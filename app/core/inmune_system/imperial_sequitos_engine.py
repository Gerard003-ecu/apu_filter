# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Séquitos Engine (Caballos de Batalla de Consenso y Mónadas)║
║ Ruta   : app/core/inmune_system/imperial_sequitos_engine.py                  ║
║ Versión: 5.1.0-Poincare-Nested-Phases-Krein-Williamson-Nekhoroshev-Kepler    ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Motor táctico que supervisa la propagación de variables, la integración
simpléctica de Maupertuis–Jacobi sobre variedades de Darboux (M, ω), el
consenso de opinión en sub-tríadas de-confinadas (DeGroot–Fiedler), la
fidelidad de Uhlmann y la violación multipartita de Bell–CHSH. En v5.1.0
se teje, granular y rigurosamente, la **mecánica celeste de Henri Poincaré**
alineada al refinamiento 4.1.0 del motor tesserario:

  • Sección transversal Σ ⊂ T*Q con transversalidad *dinámica* n_Σ · X_H ≠ 0
    (no la tautología Ω n_Σ ≠ 0).
  • 1-forma de Poincaré–Cartan ϑ = p dq − H dt y su acción de periodo.
  • Corchete de Poisson {f,g} = (∇f)ᵀ Ω (∇g) (pairing compensado).
  • Función generatriz F₂(q, P) de Hamilton–Jacobi con hessiana simetrizada
    (cerradura d²F₂ = 0 ⇔ canonicidad).
  • Gram–Schmidt simpléctico de Parasjuk–de Gosson con completación canónica
    w ← −Ωv cuando el par degenera.
  • Forma de volumen de Liouville Ωⁿ / n! y pairing uᵀΩv.
  • Clasificación de Williamson del equilibrio (elíptico / hiperbólico /
    foco–foco / parabólico) vía K = J Hess H.
  • Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F, logaritmo real
    por Schur y **clasificación de Krein–Moser** de {μ_k}.
  • Reducción al mapa de Poincaré (2n−2)×(2n−2) (se extrae μ = 1 doble).
  • Función de Mel'nikov ℳ(t₀) con ceros *simples* (signo ⊗ |ℳ′| > 0).
  • Número de rotación ρ ∈ ℝ/ℤ (mod 2π) por fracción continua de Farey.
  • Condición de twist de Moser ∂ρ/∂I ≠ 0 (Poincaré–Birkhoff).
  • Certificado KAM |k·ω| ≥ γ/|k|^τ sobre retículos adaptativos, con
    tiempo de Nekhoroshev T_N ~ exp(c ε^{−1/(2n)}).
  • Espectro de Lyapunov por QR de Benettin; emparejamiento hamiltoniano
    λᵢ ↔ −λ_{2n+1−i} *sólo* sobre monodromías en Sp(2n).
  • Variables acción-ángulo I_i = (1/2π) ∮ p_i dq_i, θ̇_i = ∂H/∂I_i.
  • Elementos orbitales osculadores de Kepler (a, e, i, Ω, ω, ν) con
    residuo de vis-viva y periodo medio n = √(μ/a³).
  • Integración Störmer–Verlet de Maupertuis–Jacobi con residual
    simpléctico del jacobiano linealizado.

Tres fases anidadas (el objeto terminal de Φₖ es el objeto inicial de Φₖ₊₁):

  Φ_I   : _NumericalCore.synthesize_poincare_kleisli_germ
            → _PoincareKleisliGerm          ≡  germen inicial de Φ_II
  Φ_II  : _DeGrootConsensus.induce_poincare_consensus_germ
            → _PoincareConsensusGerm        ≡  germen inicial de Φ_III
  Φ_III : _CHSHVerifier.certify_poincare_sequitos
            → PoincareSequitosCertificate
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterator, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Core.ImperialSequitosEngine")

__version__: Final[str] = (
    "5.1.0-Poincare-Nested-Phases-Krein-Williamson-Nekhoroshev-Kepler"
)

# =============================================================================
# CONSTANTES DE PRECISIÓN METROLÓGICA Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_REG: Final[float] = 1e-15
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_LIMIT: Final[float] = 1e-12
_SPECTRAL_TOL: Final[float] = 1e-9
_TOLERANCE_DEGROOT_COHERENT: Final[float] = 1e-6
_TOLERANCE_DEGROOT_DEGRADED: Final[float] = 1e-4
_TSIRELSON_BOUND: Final[float] = float(2.0 * np.sqrt(2.0))
_CLASSICAL_CHSH_BOUND: Final[float] = 2.0
_PR_NOSIGNAL_BOUND: Final[float] = 4.0
_LOG_EXP_CLIP: Final[float] = 700.0
_CORRELATOR_BOUND: Final[float] = 1.0

# ── Constantes de Poincaré / KAM / Floquet / Nekhoroshev ────────────────────
_HARD_POINCARE_SECTION_TOL: Final[float] = 1.0e-10
_HARD_FLOQUET_MULTIPLIER_TOL: Final[float] = 1.0e-2
_HARD_LYAPUNOV_TOL: Final[float] = 1.0e-6
_HARD_KAM_GAMMA_FLOOR: Final[float] = 1.0e-6
_HARD_KREIN_UNIT_TOL: Final[float] = 1.0e-8
_HARD_TWIST_FLOOR: Final[float] = 1.0e-10
_HARD_CARTAN_CLOSURE_TOL: Final[float] = 1.0e-6
_KAM_HARMONIC_CAP: Final[int] = 24
_KAM_LATTICE_CELL_CAP: Final[int] = 180_000
_MELNIKOV_PHASE_SAMPLES: Final[int] = 64
_LYAPUNOV_QR_ITERATIONS: Final[int] = 2048
_ROTATION_CF_DEPTH: Final[int] = 24
_DIOPHANTINE_TAU_FLOOR: Final[float] = 1.0 + 1.0e-6
_CARTAN_INTEGRAL_SAMPLES: Final[int] = 256
_ACTION_ANGLE_QUADRATURE: Final[int] = 512
_PARABOLIC_MULTIPLIER_TOL: Final[float] = 1.0e-6
_TWO_PI: Final[float] = 2.0 * float(np.pi)
_NEKHOROSHEV_PREFACTOR: Final[float] = 0.5
_WILLIAMSON_IMAG_RATIO: Final[float] = 1.0e-8
_VIS_VIVA_REL_TOL: Final[float] = 1.0e-8

_PAULI: Final[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
)


class SymplecticDimensionError(ValueError):
    """Dimensión impar incompatible con Sp(2n, ℝ)."""


# =============================================================================
# DTOs GLOBALES — Fase I, II, III y certificado terminal
# =============================================================================
@dataclass(frozen=True)
class SequitosEngineStepResult:
    r"""
    DTO inmutable de salida de la FPU para la integración simpléctica
    de Séquitos (Störmer–Verlet) con Acción de Maupertuis–Jacobi.
    """

    next_state_z: NDArray[np.float64]
    hamiltonian_energy: float
    volume_drift: float
    maupertuis_action: float
    is_liouville_conserved: bool
    symplectic_residual: float = 0.0
    cartan_increment: float = 0.0


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE I — NÚCLEO DE BANACH, MÓNADA WRITER, KLEISLI–GIRY Y POINCARÉ      ██
# ██  Objetos: sumas compensadas, kernels, sección Σ, corchete de Poisson,   ██
# ██           F₂ Hamilton–Jacobi, Gram–Schmidt simpléctico, Williamson.     ██
# ██  Morfismo terminal (I.ω): synthesize_poincare_kleisli_germ              ██
# ██                           → objeto inicial de la FASE II                ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _MarkovKernelCertificate:
    """Certificado de un núcleo de Markov (morfismo de Kleisli–Giry)."""

    row_stochastic_residual: float
    spectral_radius: float
    perron_residual: float
    is_stochastic: bool
    is_reversible: bool


@dataclass(frozen=True)
class _SymplecticFormCertificate:
    """Certificado de la 2-forma canónica Ω (Darboux)."""

    skew_residual: float
    almost_complex_residual: float
    determinant: float
    frobenius_norm: float
    is_darboux: bool
    liouville_volume: float = 1.0


@dataclass(frozen=True)
class _PoincareSectionWitness:
    r"""
    Sección transversal de Poincaré Σ = { q_k = q_k^* }.

    Transversalidad *dinámica*: n_Σ · X_H ≠ 0. La no-degeneración
    Ω n_Σ ≠ 0 es sanity del chart de Darboux.
    """

    normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    transversal_certificate: float
    is_transversal: bool
    flow_flux: float = 0.0
    is_dynamically_transversal: bool = False


@dataclass(frozen=True)
class _HamiltonJacobiWitness:
    r"""
    Función generatriz F₂(q, P) de Hamilton–Jacobi.

        p = ∂F₂/∂q ,   Q = ∂F₂/∂P .

    Residuo HJ = ‖H − Hᵀ‖_F (obstrucción a la cerradura d²F₂ = 0).
    """

    q_old: np.ndarray
    p_old: np.ndarray
    q_new: np.ndarray
    P_new: np.ndarray
    p_check: np.ndarray
    hessian_F2: np.ndarray
    hj_residual: float
    is_canonical: bool


@dataclass(frozen=True)
class _WilliamsonWitness:
    r"""
    Clasificación de Williamson de un equilibrio hamiltoniano.

    K = J A, J = Ω⁻¹ = −Ω, A = Hess H. Espectro cerrado por
    (λ, −λ, λ̄, −λ̄): pares elípticos (±iω), hiperbólicos (±λ),
    foco–foco (±α±iβ) y parabólicos (λ = 0).
    """

    elliptic_pairs: int
    hyperbolic_pairs: int
    focus_focus_pairs: int
    parabolic_multiplicity: int
    is_linearly_stable: bool
    hamiltonian_eigenvalues: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _MarkovKleisliGerm:
    """Gérmen de Kleisli–Markov (objeto terminal histórico de la Fase I)."""

    n_agents: int
    kernel: np.ndarray
    affinity: np.ndarray
    laplacian: np.ndarray
    stationary: np.ndarray
    degrees: np.ndarray
    reg_floor: float
    certificate: _MarkovKernelCertificate


@dataclass(frozen=True)
class _PoincareKleisliGerm:
    r"""
    **Gérmen de Poincaré–Kleisli.**

    **Objeto terminal de la FASE I / objeto inicial de la FASE II.**

    Envuelve el gérmen de Kleisli–Markov y añade la geometría de Poincaré:
    2-forma certificada Ω, sección transversal Σ (dinámica), volumen de
    Liouville y piso de Wilkinson. Sobre este gérmen, la FASE II define
    DeGroot, Floquet–Lyapunov, Mel'nikov, rotación, KAM y Lyapunov.
    """

    markov_germ: _MarkovKleisliGerm
    omega: np.ndarray
    form_certificate: _SymplecticFormCertificate
    section: _PoincareSectionWitness
    reg_floor: float
    max_iter: int
    tol: float
    flow_vector: Optional[np.ndarray] = None
    liouville_volume: float = 1.0


class _NumericalCore:
    """
    Fase I. Álgebra numérica de precisión metrológica y geometría
    simpléctica de Poincaré. Provee el topos lineal subyacente:
    sumación compensada, 2-forma de Liouville, corchete de Poisson,
    función generatriz F₂, sección transversal Σ, Gram–Schmidt
    simpléctico y clasificación de Williamson.
    """

    # ── I.1  Sumas compensadas ────────────────────────────────────────────
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
                raise ValueError("kahan_babuska_neumaier_sum: no-finito.")
            t = total + xf
            if abs(total) >= abs(xf):
                c += (total - t) + xf
            else:
                c += (xf - t) + total
            total = t
        return float(total + c)

    kahan_neumann_sum = kahan_babuska_neumaier_sum

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

    @staticmethod
    def compensated_real_trace(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("compensated_real_trace: matriz cuadrada requerida.")
        return _NumericalCore.kahan_babuska_neumaier_sum(np.real(np.diag(a)))

    # ── I.2  Normas, validación y Higham ──────────────────────────────────
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        return 0.0 if a.size == 0 else float(la.norm(a, "fro"))

    @staticmethod
    def euclidean_norm(vec: np.ndarray) -> float:
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return 0.0
        return float(np.sqrt(max(
            _NumericalCore.kahan_babuska_neumaier_sum(v * v), 0.0)))

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
    def assert_vec(name: str, vec: np.ndarray,
                   dim: Optional[int] = None) -> np.ndarray:
        v = np.asarray(vec).reshape(-1)
        if dim is not None and v.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {v.size}.")
        _NumericalCore.assert_finite(name, v)
        return v

    @staticmethod
    def higham_nearest_hermitian(matrix: np.ndarray) -> np.ndarray:
        a = np.asarray(matrix)
        _NumericalCore.assert_square("higham_nearest_hermitian", a)
        return 0.5 * (a + a.T.conj())

    symmetrize_hermitian = higham_nearest_hermitian

    @staticmethod
    def higham_nearest_spd(
        matrix: np.ndarray, floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        herm = _NumericalCore.higham_nearest_hermitian(matrix)
        evals, evecs = la.eigh(herm)
        evals = np.maximum(np.real(evals), float(floor))
        return evecs @ (evals[:, None] * evecs.T.conj())

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro_norm = _NumericalCore.frobenius_norm(matrix)
        return float(max(
            fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
            _WILKINSON_DEFLATION_FLOOR))

    @staticmethod
    def multiply_probabilities(p: float, q: float) -> float:
        pf = float(p)
        qf = float(q)
        if not (np.isfinite(pf) and np.isfinite(qf)):
            raise ValueError("multiply_probabilities: argumentos no finitos.")
        if pf <= 0.0 or qf <= 0.0:
            return 0.0
        lp = np.log(min(pf, 1.0)) + np.log(min(qf, 1.0))
        if lp < -_LOG_EXP_CLIP:
            return 0.0
        return float(np.exp(lp))

    @staticmethod
    def restochasticize_rows(
        matrix: np.ndarray, floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        a = np.real(np.asarray(matrix, dtype=np.float64))
        _NumericalCore.assert_square("restochasticize_rows", a)
        w = np.maximum(a, 0.0)
        for i in range(w.shape[0]):
            mass = _NumericalCore.kahan_babuska_neumaier_sum(w[i])
            if mass > max(floor, _MACHINE_EPS):
                w[i] = w[i] / mass
            else:
                w[i] = 0.0
                w[i, i] = 1.0
        return w

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

    # ── I.3  2-forma simpléctica canónica de Darboux ──────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""Ω = [[0, I_q], [−I_q, 0]] ,  Ωᵀ = −Ω ,  Ω² = −I ,  det Ω = 1 ."""
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim={dim} debe ser par y positivo (Darboux).")
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def liouville_volume(omega: np.ndarray) -> float:
        r"""vol = Ωⁿ / n!  (Pfaffiano; = 1 en Darboux)."""
        w = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("omega", w)
        if w.shape[0] % 2 != 0:
            raise SymplecticDimensionError("Liouville exige dimensión par.")
        det_o = float(np.real(la.det(w)))
        return float(np.sqrt(max(det_o, 0.0)))

    @staticmethod
    def certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        _NumericalCore.assert_square("omega", omega)
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = _NumericalCore.frobenius_norm(omega + omega.T)
        almost_c = _NumericalCore.frobenius_norm(omega @ omega + ident)
        det_o = float(np.real(la.det(omega)))
        fro = _NumericalCore.frobenius_norm(omega)
        scale = max(fro, 1.0)
        vol = _NumericalCore.liouville_volume(omega)
        is_darboux = bool(
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            frobenius_norm=fro,
            is_darboux=is_darboux,
            liouville_volume=float(vol),
        )

    # ── I.4  Sección transversal de Poincaré (dinámica) ───────────────────
    @staticmethod
    def build_poincare_section(
        two_n: int,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> _PoincareSectionWitness:
        r"""
        Σ = { q_{section_index} = q^* } ,  n_Σ = e_{section_index} .

        Transversalidad *dinámica*: n_Σ · X_H ≠ 0. Si no hay X_H, se
        cae al sanity Ω n_Σ ≠ 0 (automático si n_Σ ≠ 0).
        """
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        if not (0 <= section_index < n):
            raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        if omega is None:
            omega = _NumericalCore.generate_canonical_symplectic_form(two_n)
        v = np.asarray(omega, dtype=np.float64) @ n_sigma
        chart_cert = _NumericalCore.frobenius_norm(v)
        chart_ok = bool(chart_cert > _MACHINE_EPS)
        flow_flux = 0.0
        dyn_ok = False
        if flow_vector is not None:
            xh = np.asarray(flow_vector, dtype=np.float64).ravel()
            if xh.size != two_n:
                raise ValueError(f"flow_vector dim {xh.size} ≠ two_n={two_n}.")
            flow_flux = abs(_NumericalCore.kahan_babuska_neumaier_sum(n_sigma * xh))
            dyn_ok = bool(chart_ok and flow_flux > _MACHINE_EPS)
        return _PoincareSectionWitness(
            normal=n_sigma,
            section_index=int(section_index),
            section_offset=float(section_offset),
            energy_level=float(energy_level),
            transversal_certificate=float(chart_cert),
            is_transversal=chart_ok,
            flow_flux=float(flow_flux),
            is_dynamically_transversal=dyn_ok if flow_vector is not None else chart_ok,
        )

    # ── I.5  Corchete de Poisson {f,g} ────────────────────────────────────
    @staticmethod
    def poisson_bracket(
        grad_f: np.ndarray, grad_g: np.ndarray, omega: np.ndarray,
    ) -> float:
        r"""{f,g} = (∇f)ᵀ Ω (∇g) = Σ (∂f/∂q · ∂g/∂p − ∂f/∂p · ∂g/∂q)."""
        gf = np.asarray(grad_f, dtype=np.float64).ravel()
        gg = np.asarray(grad_g, dtype=np.float64).ravel()
        if gf.size != gg.size:
            raise ValueError("grad_f y grad_g deben coincidir en dimensión.")
        if omega.shape != (gf.size, gf.size):
            raise ValueError("Ω incompatible con las gradientes.")
        return _NumericalCore.symplectic_pairing(gf, gg, omega)

    # ── I.6  Función generatriz F₂ de Hamilton–Jacobi ─────────────────────
    @staticmethod
    def hamilton_jacobi_F2(
        q_old: np.ndarray,
        p_old: np.ndarray,
        q_new: np.ndarray,
        hessian_F2: Optional[np.ndarray] = None,
    ) -> _HamiltonJacobiWitness:
        r"""
        Germen cuadrático de F₂(q, P) = P·q + ½ (q−q_old)ᵀ H_sym (q−q_old),
        con H_sym = ½(H+Hᵀ) para imponer d²F₂ = 0 (hessiana cerrada ⇔
        canonicidad). Entonces

            P_new = p_old + H_sym (q_new − q_old),
            p_check = P_new − H_sym (q_new − q_old)  (= p_old).

        Residuo HJ = ‖H − Hᵀ‖_F.
        """
        qo = np.asarray(q_old, dtype=np.float64).ravel()
        po = np.asarray(p_old, dtype=np.float64).ravel()
        qn = np.asarray(q_new, dtype=np.float64).ravel()
        if qo.shape != po.shape or qo.shape != qn.shape:
            raise ValueError("q_old, p_old, q_new deben compartir shape.")
        n = qo.size
        H = np.eye(n, dtype=np.float64) if hessian_F2 is None \
            else np.asarray(hessian_F2, dtype=np.float64)
        _NumericalCore.assert_square("hessian_F2", H, dim=n)
        H_sym = 0.5 * (H + H.T)
        dq = qn - qo
        P_new = po + H_sym @ dq
        p_check = P_new - H_sym @ dq
        hj_res = _NumericalCore.frobenius_norm(H - H.T)
        is_canon = bool(hj_res <= _WILKINSON_DRIFT_LIMIT * max(1.0, _NumericalCore.frobenius_norm(H)))
        return _HamiltonJacobiWitness(
            q_old=qo, p_old=po, q_new=qn,
            P_new=P_new, p_check=p_check, hessian_F2=H_sym,
            hj_residual=float(hj_res),
            is_canonical=is_canon,
        )

    # ── I.7  Gram–Schmidt simpléctico (Parasjuk–de Gosson) ────────────────
    @staticmethod
    def symplectic_gram_schmidt(
        vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        r"""
        Construye S ∈ Sp(2n,ℝ) con pares conjugados (e_k, f_k):

            ω(e_i, e_j) = 0,  ω(f_i, f_j) = 0,  ω(e_i, f_j) = δ_{ij}.

        Si el par (v, w) degenera, se completa canónicamente w ← −Ω v,
        pues ω(v, −Ωv) = ‖v‖² (Ω² = −I).
        """
        B = np.asarray(vectors, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("omega", O)
        dim = O.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError("SGS exige dimensión par (Sp(2n)).")
        if B.shape != (dim, dim):
            raise ValueError(f"vectors debe ser ({dim},{dim}); {B.shape}.")
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
                    "SGS: par %d degenerado (ω=%.3e); se regulariza.", k, omega_vw)
                omega_vw = np.sign(omega_vw or 1.0) * _MACHINE_EPS
            S[:, k] = v
            S[:, k + n] = w / omega_vw
        res = _NumericalCore.frobenius_norm(S.T @ O @ S - O)
        if res > 1e-6:
            logger.warning("SGS: residuo simpléctico %.3e > umbral.", res)
        return S

    # ── I.7.b  Clasificación de Williamson ────────────────────────────────
    @staticmethod
    def williamson_classify(
        hessian: np.ndarray, omega: np.ndarray,
    ) -> _WilliamsonWitness:
        r"""Espectro de Williamson de H = ½ xᵀ A x, A = Hess H simétrica."""
        A = np.asarray(hessian, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        _NumericalCore.assert_square("hessian", A)
        _NumericalCore.assert_square("omega", O, dim=A.shape[0])
        A_sym = np.real(_NumericalCore.higham_nearest_hermitian(A))
        K = -O @ A_sym
        ev = la.eigvals(K)
        elliptic = hyperbolic = focus = parabolic = 0
        for z in ev:
            re, im = float(np.real(z)), float(np.imag(z))
            scale = max(abs(re), abs(im), 1.0)
            if abs(re) <= _WILLIAMSON_IMAG_RATIO * scale and abs(
                    im) <= _WILLIAMSON_IMAG_RATIO * scale:
                parabolic += 1
            elif abs(re) <= _WILLIAMSON_IMAG_RATIO * scale:
                elliptic += 1
            elif abs(im) <= _WILLIAMSON_IMAG_RATIO * scale:
                hyperbolic += 1
            else:
                focus += 1
        elliptic_pairs = elliptic // 2
        hyperbolic_pairs = hyperbolic // 2
        focus_focus_pairs = focus // 4
        is_stable = bool(
            hyperbolic_pairs == 0
            and focus_focus_pairs == 0
            and parabolic == 0
            and elliptic_pairs == A.shape[0] // 2
        )
        return _WilliamsonWitness(
            elliptic_pairs=int(elliptic_pairs),
            hyperbolic_pairs=int(hyperbolic_pairs),
            focus_focus_pairs=int(focus_focus_pairs),
            parabolic_multiplicity=int(parabolic),
            is_linearly_stable=bool(is_stable),
            hamiltonian_eigenvalues=np.asarray(ev),
        )

    # ── I.8  Involución de Cartan y residual simpléctico ──────────────────
    @staticmethod
    def cartan_involution(M: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""ι(M) = Ω M^{−T} Ωᵀ (involution de Cartan del par (GL, Sp))."""
        m = np.asarray(M, dtype=np.float64)
        dim = m.shape[0]
        ident = np.eye(dim, dtype=np.float64)
        try:
            x_inv_t = la.solve(m.T, ident, assume_a="gen")
            return omega @ x_inv_t @ omega.T
        except (la.LinAlgError, ValueError) as exc:
            logger.warning("Cartan fallido (%s); pinv.", exc)
            return omega @ la.pinv(m.T) @ omega.T

    @staticmethod
    def symplectic_residual(M: np.ndarray, omega: np.ndarray) -> float:
        """‖Mᵀ Ω M − Ω‖_F ."""
        m = np.asarray(M, dtype=np.float64)
        return _NumericalCore.frobenius_norm(m.T @ omega @ m - omega)

    @staticmethod
    def symplectic_inverse(S: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """S⁻¹ = −Ω Sᵀ Ω  (pues Ω⁻¹ = −Ω)."""
        return -omega @ np.asarray(S).T @ omega

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ───────────────────────────────
    @staticmethod
    def synthesize_poincare_kleisli_germ(
        affinity_matrix: np.ndarray,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        symmetrize: bool = True,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        max_iter: int = 100,
        tol: float = 1.0e-12,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareKleisliGerm:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla el gérmen de Poincaré–Kleisli

            𝒢_I = (𝒢_Markov, Ω, Cert(Ω), Σ, ε_W, K, τ, X_H, vol_Liouville)

        donde 𝒢_Markov es el gérmen de Kleisli–Giry (núcleo de Markov
        row-stochastic + Laplaciano normalizado), Ω es la 2-forma de
        Liouville de dimensión 2·n_agentes (inmersión de la opinión en
        T*Q ≅ ℝ^{2n}), Cert(Ω) su certificado, Σ la sección transversal
        de Poincaré (dinámica si se da X_H).

        Este método **es** el arranque formal de `_DeGrootConsensus.__init__`.
        """
        markov_germ = _KleisliComposer.synthesize_markov_kleisli_germ(
            affinity_matrix, regularizer=regularizer, symmetrize=symmetrize,
        )
        n_agents = markov_germ.n_agents
        two_n = 2 * n_agents
        omega = _NumericalCore.generate_canonical_symplectic_form(two_n)
        form_cert = _NumericalCore.certify_symplectic_form(omega)
        if not form_cert.is_darboux:
            logger.warning(
                "Darboux degradado: skew=%.3e, Ω²+I=%.3e, det=%.16f",
                form_cert.skew_residual, form_cert.almost_complex_residual,
                form_cert.determinant,
            )
        flow = None if flow_vector is None else np.asarray(
            flow_vector, dtype=np.float64).ravel()
        if flow is not None and flow.size != two_n:
            raise ValueError(f"flow_vector dim {flow.size} ≠ two_n={two_n}.")
        section = _NumericalCore.build_poincare_section(
            two_n, section_index=section_index,
            section_offset=section_offset, energy_level=energy_level,
            flow_vector=flow, omega=omega,
        )
        floor = max(
            float(markov_germ.reg_floor),
            _NumericalCore.wilkinson_deflation_floor(affinity_matrix),
        )
        return _PoincareKleisliGerm(
            markov_germ=markov_germ,
            omega=omega,
            form_certificate=form_cert,
            section=section,
            reg_floor=float(floor),
            max_iter=int(max_iter),
            tol=float(tol),
            flow_vector=flow,
            liouville_volume=float(form_cert.liouville_volume),
        )


class _KleisliComposer:
    """
    Fase I (continuación monádica). Categoría de Kleisli de dos mónadas:
    la mónada Writer_([0,1], ×) y la mónada de Giry finita.
    """

    @staticmethod
    def unit(value: Any) -> Tuple[Any, float]:
        return value, 1.0

    @staticmethod
    def bind(
        ta: Tuple[Any, float],
        k: Callable[[Any], Tuple[Any, float]],
    ) -> Tuple[Any, float]:
        value, prob = ta
        pf = float(prob)
        if not np.isfinite(pf) or pf < _MACHINE_EPS:
            return None, 0.0
        res, q = k(value)
        return res, _NumericalCore.multiply_probabilities(pf, float(q))

    @staticmethod
    def compose(
        f: Callable[[Any], Tuple[Any, float]],
        g: Callable[[Any], Tuple[Any, float]],
    ) -> Callable[[Any], Tuple[Any, float]]:
        def composed(x: Any) -> Tuple[Any, float]:
            return _KleisliComposer.bind(f(x), g)
        return composed

    @staticmethod
    def compose_markov_kernels(
        p_kernel: np.ndarray,
        q_kernel: np.ndarray,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        p = np.asarray(p_kernel, dtype=np.float64)
        q = np.asarray(q_kernel, dtype=np.float64)
        _NumericalCore.assert_square("p_kernel", p)
        _NumericalCore.assert_square("q_kernel", q)
        if p.shape[1] != q.shape[0]:
            raise ValueError(
                f"Kernels incompatibles para Kleisli: {p.shape} ⋆ {q.shape}.")
        return _NumericalCore.restochasticize_rows(p @ q, floor=floor)

    @staticmethod
    def markov_power(
        kernel: np.ndarray,
        steps: int,
        floor: float = _HIGHAM_TIKHONOV_REG,
    ) -> np.ndarray:
        w = _NumericalCore.restochasticize_rows(
            np.asarray(kernel, dtype=np.float64), floor=floor)
        n = w.shape[0]
        if steps <= 0:
            return np.eye(n, dtype=np.float64)
        result = np.eye(n, dtype=np.float64)
        base = w
        k = int(steps)
        while k:
            if k & 1:
                result = _NumericalCore.restochasticize_rows(
                    result @ base, floor=floor)
            base = _NumericalCore.restochasticize_rows(base @ base, floor=floor)
            k >>= 1
        return result

    @staticmethod
    def synthesize_markov_kleisli_germ(
        affinity_matrix: np.ndarray,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        symmetrize: bool = True,
    ) -> _MarkovKleisliGerm:
        raw = np.asarray(affinity_matrix)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError("affinity_matrix plana no es cuadrado perfecto.")
            raw = raw.reshape(side, side)
        _NumericalCore.assert_square("affinity_matrix", raw)
        _NumericalCore.assert_finite("affinity_matrix", raw)
        a = np.real(np.asarray(raw, dtype=np.float64))
        if symmetrize:
            a = np.real(_NumericalCore.higham_nearest_hermitian(a))
        a = np.maximum(a, 0.0)
        n = a.shape[0]
        if n == 0:
            raise ValueError("affinity_matrix no puede ser 0×0.")
        floor = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        floor = max(floor, _NumericalCore.wilkinson_deflation_floor(a))
        degrees = np.array([
            _NumericalCore.kahan_babuska_neumaier_sum(a[i]) for i in range(n)
        ], dtype=np.float64)
        w = _NumericalCore.restochasticize_rows(a, floor=floor)
        inv_sqrt = np.zeros(n, dtype=np.float64)
        live = degrees > floor
        inv_sqrt[live] = 1.0 / np.sqrt(degrees[live])
        d_is = np.diag(inv_sqrt)
        lap = np.eye(n, dtype=np.float64) - d_is @ a @ d_is
        lap = np.real(_NumericalCore.higham_nearest_hermitian(lap))
        try:
            ev, evec = la.eig(w.T)
            ev = np.asarray(ev)
            k_perron = int(np.argmin(np.abs(ev - 1.0)))
            pi = np.real(evec[:, k_perron])
            pi = np.maximum(pi, 0.0)
            mass = _NumericalCore.kahan_babuska_neumaier_sum(pi)
            if mass > _MACHINE_EPS:
                pi = pi / mass
            else:
                pi = np.full(n, 1.0 / n, dtype=np.float64)
            spectral_radius = float(np.max(np.abs(ev)))
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Perron-Frobenius fallido (%s); π uniforme.", exc)
            pi = np.full(n, 1.0 / n, dtype=np.float64)
            spectral_radius = 1.0
        ones = np.ones(n, dtype=np.float64)
        row_res = _NumericalCore.euclidean_norm(w @ ones - ones)
        perron_res = _NumericalCore.euclidean_norm(w.T @ pi - pi)
        db = pi[:, None] * w - pi[None, :] * w.T
        db_res = _NumericalCore.frobenius_norm(db)
        scale = max(1.0, float(n))
        is_stoch = row_res <= max(_WILKINSON_DRIFT_LIMIT,
                                  _WILKINSON_DRIFT_LIMIT * scale)
        is_rev = db_res <= max(_WILKINSON_DRIFT_LIMIT,
                               _WILKINSON_DRIFT_LIMIT * scale)
        cert = _MarkovKernelCertificate(
            row_stochastic_residual=float(row_res),
            spectral_radius=float(spectral_radius),
            perron_residual=float(perron_res),
            is_stochastic=bool(is_stoch),
            is_reversible=bool(is_rev),
        )
        return _MarkovKleisliGerm(
            n_agents=n, kernel=w, affinity=a, laplacian=lap,
            stationary=pi, degrees=degrees,
            reg_floor=float(floor), certificate=cert,
        )


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE II — DEGROOT, UHLMANN Y POINCARÉ (FLOQUET, MEL'NIKOV, KAM, LYAP)   ██
# ██  Continuación directa del morfismo I.ω.                                   ██
# ██  Morfismo terminal (II.ω): induce_poincare_consensus_germ                ██
# ██                           → objeto inicial de la FASE III                ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
# PUENTE Φ_I ▸ Φ_II
# El valor de retorno de I.ω (`_PoincareKleisliGerm`) es el germen
# geométrico de `_DeGrootConsensus.__init__` / `induce_poincare_consensus_germ`.
# =============================================================================
@dataclass(frozen=True)
class _DeGrootConsensusResult:
    """Resultado certificado del consenso de DeGroot / Olfati–Saber."""

    final_opinion: np.ndarray
    fiedler_value: float
    verdict: str
    discrete_opinion: np.ndarray
    continuous_opinion: np.ndarray
    spectral_gap: float
    mixing_rate: float
    stationary: np.ndarray
    connected: bool
    cheeger_upper: float
    deviation: float
    is_reversible: bool


@dataclass(frozen=True)
class _UhlmannFidelityResult:
    """Fidelidad de Uhlmann con desigualdades de Fuchs–van de Graaf y Bures."""

    fidelity: float
    converged: bool
    bures_angle: float
    trace_distance: float
    fuchs_lower: float
    fuchs_upper: float


@dataclass(frozen=True)
class _BellCorrelationGerm:
    """Gérmen de correlación de Bell (objeto terminal histórico de la Fase II)."""

    correlation_matrix: np.ndarray
    horodecki_singular_values: np.ndarray
    tsirelson_forecast: float
    from_quantum: bool
    physical: bool


@dataclass(frozen=True)
class _KreinWitness:
    r"""
    Clasificación de Krein–Moser de los multiplicadores de Floquet.

      * elípticos   : |μ| = 1, μ ≠ ±1
      * parabólicos : μ = ±1
      * hiperbólicos: μ ∈ ℝ, |μ| ≠ 1
      * loxodrómicos: μ ∈ ℂ \ ℝ, |μ| ≠ 1

    krein_definite ⇔ elípticos simples de signatura definida.
    """

    elliptic: int
    hyperbolic: int
    loxodromic: int
    parabolic: int
    krein_definite: bool
    on_unit_circle: int


@dataclass(frozen=True)
class _FloquetLyapunovWitness:
    r"""Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F ."""

    generator: np.ndarray
    periodic_part: np.ndarray
    log_residual: float
    is_real_logarithm: bool
    floquet_multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    krein: Optional[_KreinWitness] = None
    poincare_map_multipliers: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _MelnikovWitness:
    r"""
    Función de Mel'nikov ℳ(t₀) = ∫ {H₀,H₁}(q₀(t), t+t₀) dt.

    Cero *simple*: ℳ(t₀)=0 y ℳ'(t₀)≠0 ⇒ intersección transversal
    W^s ∩ W^u ⇒ caos homoclínico.
    """

    melnikov_values: np.ndarray
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    melnikov_derivative_min: float = 0.0


@dataclass(frozen=True)
class _RotationNumberWitness:
    """Número de rotación ρ ∈ ℝ/ℤ ; fracción continua de Farey."""

    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float


@dataclass(frozen=True)
class _MoserTwistWitness:
    r"""Certificado de twist de Moser: |∂ρ/∂I| ≥ ν > 0."""

    twist_value: float
    is_twist: bool
    intersection_property: bool


@dataclass(frozen=True)
class _KamTorusWitness:
    r"""Certificado KAM diofantino |k·ω| ≥ γ/|k|^τ, con Nekhoroshev."""

    frequency_vector: np.ndarray
    diophantine_gamma: float
    diophantine_tau: float
    birkhoff_residual: float
    kam_stable: bool
    iterations: int
    nekhoroshev_time: float = 0.0
    worst_small_divisor: float = 0.0


@dataclass(frozen=True)
class _LyapunovSpectrumWitness:
    r"""
    Espectro de Lyapunov λ₁ ≥ … ≥ λ_{2n} por QR de Benettin.
    Emparejamiento hamiltoniano λᵢ ↔ −λ_{2n+1−i} *sólo* si M ∈ Sp(2n).
    """

    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    is_chaotic: bool


@dataclass(frozen=True)
class _PoincareCartanWitness:
    r"""Invariante integral de Poincaré–Cartan 𝒜 = ∫ (p·dq − H dt)."""

    action: float
    closure_residual: float
    is_closed: bool


@dataclass(frozen=True)
class _PoincareConsensusGerm:
    r"""
    **Gérmen de Poincaré–Consenso.**

    **Objeto terminal de la FASE II / objeto inicial de la FASE III.**

    Combina el gérmen de Poincaré–Kleisli (Fase I) con DeGroot,
    Floquet–Lyapunov, Mel'nikov, rotación, KAM, Lyapunov, Williamson,
    Cartan y twist de Moser.
    """

    kleisli_germ: _PoincareKleisliGerm
    degroot: _DeGrootConsensusResult
    floquet: _FloquetLyapunovWitness
    melnikov: Optional[_MelnikovWitness]
    rotation: Optional[_RotationNumberWitness]
    kam: _KamTorusWitness
    lyapunov: _LyapunovSpectrumWitness
    williamson: Optional[_WilliamsonWitness] = None
    cartan: Optional[_PoincareCartanWitness] = None
    moser_twist: Optional[_MoserTwistWitness] = None
    symplectic_floquet: Optional[_FloquetLyapunovWitness] = None


class _DeGrootConsensus:
    """
    Fase II. Consenso espectral de DeGroot y protocolo continuo, ahora
    enriquecido con los morfismos de Poincaré 5.1: Floquet–Lyapunov,
    Krein–Moser, Mel'nikov (ceros simples), rotación mod 2π, twist de
    Moser, KAM adaptativo, Nekhoroshev y Lyapunov hamiltoniano.
    """

    def __init__(
        self,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        germ: Optional[_MarkovKleisliGerm] = None,
        poincare_germ: Optional[_PoincareKleisliGerm] = None,
    ) -> None:
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        self._germ = germ
        self._poincare_germ = poincare_germ

    @property
    def germ(self) -> Optional[_MarkovKleisliGerm]:
        return self._germ

    @property
    def poincare_germ(self) -> Optional[_PoincareKleisliGerm]:
        return self._poincare_germ

    # ── II.1  Resolución del gérmen y compatibilidad ─────────────────────
    def _resolve_germ(
        self, opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, _MarkovKleisliGerm]:
        x = _NumericalCore.assert_vec("opinion_vector", opinion_vector)
        n = x.size
        need = (
            self._germ is None
            or self._germ.n_agents != n
            or np.asarray(affinity_matrix).shape != (n, n)
        )
        if need:
            germ = _KleisliComposer.synthesize_markov_kleisli_germ(
                affinity_matrix, regularizer=self._reg, symmetrize=True)
            if germ.n_agents != n:
                raise ValueError(
                    "La matriz de afinidad debe ser cuadrada y coincidir "
                    "con el vector de opinión.")
            self._germ = germ
        else:
            germ = self._germ
        return x.astype(np.float64, copy=False), germ

    @staticmethod
    def _verdict_from_deviation(deviation: float) -> str:
        if deviation > _TOLERANCE_DEGROOT_DEGRADED:
            return "VETOED"
        if deviation > _TOLERANCE_DEGROOT_COHERENT:
            return "DEGRADED"
        return "COHERENT"

    # ── II.2  Consenso clásico (compatible 4.0 / 5.0) ────────────────────
    def compute(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootConsensusResult:
        if int(steps) < 0:
            raise ValueError("steps debe ser no negativo.")
        x, germ = self._resolve_germ(opinion_vector, affinity_matrix)
        n = germ.n_agents
        t = float(steps)
        evals = np.real(la.eigvalsh(germ.laplacian)) if n else np.array([0.0])
        evals = np.sort(evals)
        lambda_0 = float(evals[0]) if evals.size else 0.0
        fiedler = float(evals[1]) if evals.size > 1 else 0.0
        ker_tol = max(germ.reg_floor, _WILKINSON_DEFLATION_FLOOR * max(n, 1))
        connected = bool(n == 1
                         or (abs(lambda_0) <= ker_tol and fiedler > ker_tol))
        spectral_gap = float(max(fiedler, 0.0))
        cheeger_upper = float(np.sqrt(max(2.0 * spectral_gap, 0.0)))
        try:
            w_ev = np.sort_complex(la.eigvals(germ.kernel))
            mods = np.sort(np.abs(w_ev))[::-1]
            second = float(mods[1]) if mods.size > 1 else 0.0
            second = min(max(second, 0.0), 1.0)
            mixing = float(-np.log(second)) if second > _MACHINE_EPS and second < 1.0 \
                else (float("inf") if second <= _MACHINE_EPS else 0.0)
        except (np.linalg.LinAlgError, ValueError):
            mixing = 0.0
        try:
            evo = la.expm(-t * germ.laplacian)
            continuous = np.real(evo @ x)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.error("Fallo en expm: %s", exc)
            current = x.copy()
            dt = 0.01
            for _ in range(max(int(steps), 1)):
                current = current + dt * (-germ.laplacian @ current)
            continuous = current
        discrete_kernel = _KleisliComposer.markov_power(
            germ.kernel, int(steps), floor=germ.reg_floor)
        discrete = np.real(discrete_kernel @ x)
        final_opinion = np.real(np.asarray(continuous, dtype=np.float64))
        mean = _NumericalCore.kahan_babuska_neumaier_sum(final_opinion) / max(n, 1)
        var = _NumericalCore.kahan_babuska_neumaier_sum(
            (final_opinion - mean) ** 2) / max(n, 1)
        deviation = float(np.sqrt(max(var, 0.0)))
        verdict = self._verdict_from_deviation(deviation)
        if not connected and n > 1 and verdict == "COHERENT":
            verdict = "DEGRADED"
        return _DeGrootConsensusResult(
            final_opinion=final_opinion, fiedler_value=fiedler, verdict=verdict,
            discrete_opinion=np.asarray(discrete, dtype=np.float64),
            continuous_opinion=final_opinion,
            spectral_gap=spectral_gap, mixing_rate=float(mixing),
            stationary=np.asarray(germ.stationary, dtype=np.float64),
            connected=connected, cheeger_upper=cheeger_upper,
            deviation=deviation,
            is_reversible=bool(germ.certificate.is_reversible),
        )

    # ── II.3  Floquet–Lyapunov + Krein–Moser ──────────────────────────────
    @staticmethod
    def classify_krein_moser(
        mu: np.ndarray,
        omega: Optional[np.ndarray] = None,
        right_eigenvectors: Optional[np.ndarray] = None,
    ) -> _KreinWitness:
        z = np.asarray(mu, dtype=np.complex128).ravel()
        elliptic = hyperbolic = loxodromic = parabolic = 0
        on_unit = 0
        krein_ok = True
        for i, m in enumerate(z):
            ab = abs(m)
            im = float(np.imag(m))
            if abs(ab - 1.0) <= _HARD_KREIN_UNIT_TOL:
                on_unit += 1
                if (abs(m - 1.0) <= _PARABOLIC_MULTIPLIER_TOL
                        or abs(m + 1.0) <= _PARABOLIC_MULTIPLIER_TOL):
                    parabolic += 1
                else:
                    elliptic += 1
                    if (omega is not None and right_eigenvectors is not None
                            and i < right_eigenvectors.shape[1]):
                        v = right_eigenvectors[:, i]
                        kv = np.vdot(v, omega @ v)
                        kappa = float(np.imag(kv))
                        if abs(kappa) <= _MACHINE_EPS:
                            krein_ok = False
            else:
                if abs(im) <= _HARD_KREIN_UNIT_TOL * max(ab, 1.0):
                    hyperbolic += 1
                else:
                    loxodromic += 1
        if elliptic == 0:
            krein_ok = False
        return _KreinWitness(
            elliptic=int(elliptic), hyperbolic=int(hyperbolic),
            loxodromic=int(loxodromic), parabolic=int(parabolic),
            krein_definite=bool(krein_ok and loxodromic == 0
                                and hyperbolic == 0),
            on_unit_circle=int(on_unit),
        )

    @staticmethod
    def poincare_map_multipliers(
        floquet_multipliers: np.ndarray,
        drop_parabolic: int = 2,
    ) -> np.ndarray:
        r"""Espectro de DP_Σ: se extrae el parabólico μ=1 doble (flujo × energía)."""
        mu = np.asarray(floquet_multipliers, dtype=np.complex128).ravel()
        if mu.size <= drop_parabolic:
            return mu.copy()
        order = np.argsort(np.abs(mu - 1.0))
        keep = np.ones(mu.size, dtype=bool)
        keep[order[:drop_parabolic]] = False
        return mu[keep]

    @staticmethod
    def _real_matrix_logarithm(a: np.ndarray) -> Tuple[np.ndarray, bool]:
        try:
            log_M = la.logm(a)
        except (la.LinAlgError, ValueError) as exc:
            logger.warning("logm/Schur falló (%s); se diagonaliza en ℂ.", exc)
            w, V = la.eig(a)
            if abs(la.det(V)) < _MACHINE_EPS:
                return np.real(a - np.eye(a.shape[0])), False
            logw = np.log(w.astype(np.complex128))
            log_M = V @ np.diag(logw) @ la.inv(V)
        imag_amp = float(np.max(np.abs(np.imag(log_M)))) if np.iscomplexobj(
            log_M) else 0.0
        scale = max(float(np.max(np.abs(log_M))), 1.0)
        is_real = bool(imag_amp <= 1e-10 * scale)
        return np.real(np.asarray(log_M, dtype=np.complex128)), is_real

    @classmethod
    def floquet_lyapunov_factorization(
        cls,
        M: np.ndarray,
        orbit_period_T: float,
        omega: Optional[np.ndarray] = None,
        drop_parabolic: bool = False,
    ) -> _FloquetLyapunovWitness:
        r"""
        M = exp(T·A_F)·R_F con A_F = T⁻¹ Log(M).

        Un logaritmo *real* existe si M no tiene autovalores reales
        negativos de bloques de Jordan impares. Si `omega` se da y
        dim(M) es par, se clasifica Krein–Moser.
        """
        m = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", m)
        _NumericalCore.assert_finite("M", m)
        if orbit_period_T <= 0.0 or not np.isfinite(orbit_period_T):
            raise ValueError("orbit_period_T debe ser positivo y finito.")
        mu, evec = la.eig(m, right=True)
        log_M, is_real_log = cls._real_matrix_logarithm(m)
        a_F = log_M / float(orbit_period_T)
        r_F = la.expm(-float(orbit_period_T) * a_F) @ m
        log_res = _NumericalCore.frobenius_norm(
            la.expm(float(orbit_period_T) * a_F) @ r_F - m)
        with np.errstate(divide="ignore", invalid="ignore"):
            char_c = np.log(mu.astype(np.complex128)) / float(orbit_period_T)
        char_exp = np.real(char_c)
        krein: Optional[_KreinWitness] = None
        pm_mu: Optional[np.ndarray] = None
        if omega is not None and m.shape[0] % 2 == 0:
            krein = cls.classify_krein_moser(
                mu, omega=omega, right_eigenvectors=evec)
            if drop_parabolic:
                pm_mu = cls.poincare_map_multipliers(mu)
        return _FloquetLyapunovWitness(
            generator=a_F, periodic_part=r_F,
            log_residual=float(log_res),
            is_real_logarithm=is_real_log,
            floquet_multipliers=mu,
            characteristic_exponents=char_exp,
            krein=krein,
            poincare_map_multipliers=pm_mu,
        )

    # ── II.4  Función de Mel'nikov (ceros simples) ────────────────────────
    @staticmethod
    def melnikov_function(
        q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray,
        omega: np.ndarray,
        n_phase_offsets: int = _MELNIKOV_PHASE_SAMPLES,
    ) -> _MelnikovWitness:
        r"""
        ℳ(t₀) = ∫ {H₀,H₁} dt = ∫ (∇H₀)ᵀ Ω (∇H₁) dt.

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
        if omega.shape != (dim, dim):
            raise ValueError("Ω incompatible con la trayectoria.")
        pb = np.einsum("ij,jk,ik->i", g0, omega, g1)
        N = pb.size
        if N < 2:
            raise ValueError("Trayectoria demasiado corta.")
        n_phase_offsets = max(int(n_phase_offsets), 4)

        def trapz(y, h):
            return 0.0 if y.size < 2 else float(
                h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

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
        chaotic_ind = float(simple_zeros) / float(n_phase_offsets)
        return _MelnikovWitness(
            melnikov_values=mel_vals,
            simple_zeros=int(simple_zeros),
            chaotic_indicator=chaotic_ind,
            is_chaotic=bool(simple_zeros > 0),
            melnikov_derivative_min=float(np.min(np.abs(dmel))),
        )

    # ── II.5  Número de rotación ρ ∈ ℝ/ℤ ──────────────────────────────────
    @staticmethod
    def rotation_number_poincare(
        orbit_points: np.ndarray, cf_depth: int = _ROTATION_CF_DEPTH,
    ) -> _RotationNumberWitness:
        r"""
        ρ = lim (1/(2π N)) Σ Δθ_i ∈ ℝ/ℤ. Racional si un convergente de
        Farey verifica |ρ − p/q| < 1/(2 q²).
        """
        pts = np.asarray(orbit_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            raise ValueError("orbit_points debe ser (N≥3, 2n).")
        dim = pts.shape[1]
        if dim % 2 != 0:
            raise ValueError("orbit_points debe tener columnas pares.")
        n = dim // 2
        q = pts[:, 0]
        p = pts[:, n]
        theta = np.unwrap(np.arctan2(p, q))
        dtheta = np.diff(theta)
        mean_dtheta = float(
            _NumericalCore.klein_sum(dtheta) / max(dtheta.size, 1))
        rho = (mean_dtheta / _TWO_PI) % 1.0
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
        dioph = float(
            0.0 if is_rational
            else min(1.0, abs(rho - approx) * (q_den ** 2)))
        return _RotationNumberWitness(
            rotation_number=float(rho), is_rational=is_rational,
            continued_fraction=tuple(cf), diophantine_constant=dioph,
        )

    # ── II.5.b  Twist de Moser ────────────────────────────────────────────
    @staticmethod
    def moser_twist(
        rotation_samples: Sequence[_RotationNumberWitness],
        action_samples: Optional[np.ndarray] = None,
    ) -> _MoserTwistWitness:
        rhos = np.array(
            [r.rotation_number for r in rotation_samples], dtype=np.float64)
        if rhos.size < 2:
            return _MoserTwistWitness(
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
            return _MoserTwistWitness(
                twist_value=0.0, is_twist=False, intersection_property=False)
        twist = float(np.median(dr[live] / da[live]))
        return _MoserTwistWitness(
            twist_value=twist,
            is_twist=bool(abs(twist) >= _HARD_TWIST_FLOOR),
            intersection_property=bool(np.ptp(rhos) > _HARD_TWIST_FLOOR),
        )

    # ── II.6  Certificado KAM diofantino + Nekhoroshev ────────────────────
    @staticmethod
    def integer_lattice(n: int, H: int) -> Iterator[np.ndarray]:
        """k ∈ ℤⁿ \\ {0} con |k|_∞ ≤ H, recorte de cardinalidad."""
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
        for d in range(n):
            for s in (-1.0, 1.0):
                for r in range(1, H + 1):
                    kv = np.zeros(n, dtype=np.float64)
                    kv[d] = s * r
                    yield kv
        rng = np.random.default_rng(1729)
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
        r"""T_N ≳ exp(c · ε^{−1/(2n)}) ; ε ≤ 0 ⇒ T_N = +∞."""
        n = max(int(n_dof), 1)
        eps = float(perturbation_eps)
        if not np.isfinite(eps) or eps <= 0.0:
            return float("inf")
        exponent = float(prefactor) * (eps ** (-1.0 / (2.0 * n)))
        if exponent > 700.0:
            return float("inf")
        return float(np.exp(exponent))

    @classmethod
    def certify_kam_torus(
        cls,
        frequency_vector: np.ndarray,
        birkhoff_residual: float = 0.0,
        harmonic_cap: int = _KAM_HARMONIC_CAP,
    ) -> _KamTorusWitness:
        r"""|k·ω| ≥ γ/|k|^τ sobre un retículo adaptativo."""
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        n = omega.size
        if n == 0:
            raise ValueError("frequency_vector vacío.")
        H = int(harmonic_cap)
        gamma_est = np.inf
        tau_est = _DIOPHANTINE_TAU_FLOOR
        iter_count = 0
        worst = np.inf
        for kv in cls.integer_lattice(n, H):
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
            g_cand = abs_kv * (norm ** _DIOPHANTINE_TAU_FLOOR)
            if g_cand < gamma_est:
                gamma_est = g_cand
        if gamma_est > _HARD_KAM_GAMMA_FLOOR and np.isfinite(gamma_est):
            kam_stable = True
        else:
            kam_stable = False
            for tau in np.linspace(_DIOPHANTINE_TAU_FLOOR, 6.0, 24):
                g_try = np.inf
                resonant = False
                for kv in cls.integer_lattice(n, H):
                    kval = float(kv @ omega)
                    norm = float(np.linalg.norm(kv))
                    if norm < 1e-12:
                        continue
                    if abs(kval) < 1e-12:
                        g_try = 0.0
                        resonant = True
                        break
                    g_try = min(g_try, abs(kval) * (norm ** tau))
                if (not resonant) and g_try > _HARD_KAM_GAMMA_FLOOR:
                    gamma_est = g_try
                    tau_est = float(tau)
                    kam_stable = True
                    break
            else:
                kam_stable = bool(
                    np.isfinite(gamma_est)
                    and gamma_est > _HARD_KAM_GAMMA_FLOOR)
        if not np.isfinite(worst):
            worst = 0.0
        t_nek = cls.nekhoroshev_time(n, float(birkhoff_residual))
        return _KamTorusWitness(
            frequency_vector=omega,
            diophantine_gamma=float(min(gamma_est, 1e12)) if np.isfinite(
                gamma_est) else 0.0,
            diophantine_tau=float(tau_est),
            birkhoff_residual=float(birkhoff_residual),
            kam_stable=bool(kam_stable),
            iterations=int(iter_count),
            nekhoroshev_time=float(t_nek),
            worst_small_divisor=float(worst),
        )

    # ── II.7  Espectro de Lyapunov (Benettin–QR) ──────────────────────────
    @staticmethod
    def _kaplan_yorke(spectrum: np.ndarray) -> float:
        n = int(spectrum.size)
        if n == 0 or spectrum[0] < 0.0:
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

    @classmethod
    def lyapunov_spectrum_benettin(
        cls,
        M: np.ndarray,
        n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
        orbit_period_T: Optional[float] = None,
        hamiltonian_pairing: bool = False,
    ) -> _LyapunovSpectrumWitness:
        r"""
        QR de Benettin. Emparejamiento hamiltoniano λᵢ ← ½(λᵢ − λ_{n+1−i})
        *sólo* si `hamiltonian_pairing` (M ∈ Sp(2n)).
        """
        a = np.asarray(M, dtype=np.float64)
        _NumericalCore.assert_square("M", a)
        _NumericalCore.assert_finite("M", a)
        n = a.shape[0]
        N = int(n_iterations)
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
            dR = np.abs(np.diag(R))
            dR = np.where(dR > _MACHINE_EPS, dR, _MACHINE_EPS)
            log_acc += np.log(dR)
            steps += 1
        denom = float(max(steps, 1))
        spectrum = np.sort(log_acc / denom)[::-1]
        if hamiltonian_pairing and n % 2 == 0:
            spectrum = 0.5 * (spectrum - spectrum[::-1])
            spectrum = np.sort(spectrum)[::-1]
            spectrum -= float(np.mean(spectrum))
        if orbit_period_T is not None:
            T = float(orbit_period_T)
            if T > 0.0 and np.isfinite(T):
                spectrum = spectrum / T
        d_ky = cls._kaplan_yorke(spectrum)
        h_ks = float(np.sum(np.clip(spectrum, 0.0, None)))
        return _LyapunovSpectrumWitness(
            spectrum=spectrum, kaplan_yorke_dimension=d_ky,
            kolmogorov_sinai_entropy=h_ks,
            is_chaotic=bool(spectrum.size > 0 and spectrum[0] > _HARD_LYAPUNOV_TOL),
        )

    # ── II.8  Invariante integral de Poincaré–Cartan ──────────────────────
    @staticmethod
    def poincare_cartan_integral(
        q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        H_trajectory: np.ndarray, dt: float,
    ) -> _PoincareCartanWitness:
        r"""
        𝒜 = ∮_γ p dq − H dt por la regla del punto medio, con residuo
        de cierre ‖(q_N, p_N) − (q_0, p_0)‖.
        """
        q = np.asarray(q_trajectory, dtype=np.float64)
        p = np.asarray(p_trajectory, dtype=np.float64)
        H = np.asarray(H_trajectory, dtype=np.float64).ravel()
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        N = q.shape[0]
        if H.size != N:
            raise ValueError("H_trajectory debe tener N entradas.")
        if not np.isfinite(dt):
            raise ValueError("dt debe ser finito.")
        terms = np.empty(N, dtype=np.float64)
        for k in range(N - 1):
            terms[k] = float(p[k] @ (q[k + 1] - q[k])) - float(H[k]) * dt
        terms[-1] = float(p[-1] @ (q[0] - q[-1])) - float(H[-1]) * dt
        action = _NumericalCore.kahan_babuska_neumaier_sum(terms)
        close = _NumericalCore.frobenius_norm(
            np.concatenate((q[-1] - q[0], p[-1] - p[0])))
        return _PoincareCartanWitness(
            action=float(action),
            closure_residual=float(close),
            is_closed=bool(close <= _HARD_CARTAN_CLOSURE_TOL),
        )

    # ── II.9  Variables acción-ángulo ─────────────────────────────────────
    @staticmethod
    def action_angle_variables(
        q_periodic: np.ndarray, p_periodic: np.ndarray,
        n_quadrature: int = _ACTION_ANGLE_QUADRATURE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        I_k = (1/2π) ∮ p_k dq_k ,  θ_k = Δ arg(q_k + i p_k) / N .
        `n_quadrature` se conserva por firma; la cuadratura es trapezoidal
        sobre la malla dada.
        """
        del n_quadrature
        q = np.asarray(q_periodic, dtype=np.float64)
        p = np.asarray(p_periodic, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        N, n = q.shape
        if N < 2:
            raise ValueError("Trayectoria demasiado corta.")
        I_vec = np.zeros(n, dtype=np.float64)
        theta_vec = np.zeros(n, dtype=np.float64)
        for k in range(n):
            dq = np.diff(q[:, k])
            dq_closed = np.concatenate([dq, [q[0, k] - q[-1, k]]])
            p_mid = 0.5 * (p[:-1, k] + p[1:, k])
            p_mid_closed = np.concatenate([p_mid, [0.5 * (p[-1, k] + p[0, k])]])
            I_vec[k] = _NumericalCore.kahan_babuska_neumaier_sum(
                p_mid_closed * dq_closed) / _TWO_PI
            angles = np.unwrap(np.arctan2(p[:, k], q[:, k]))
            theta_vec[k] = float(angles[-1] - angles[0]) / max(N - 1, 1)
        return I_vec, theta_vec

    # ── II.10  Elementos orbitales osculadores de Kepler ──────────────────
    @staticmethod
    def kepler_osculating_elements(
        position: np.ndarray, velocity: np.ndarray,
        mu_gravitational: float = 1.0,
    ) -> dict:
        r"""
        Elementos orbitales osculadores desde (r, v):

            h = r × v ,  e⃗ = (v × h)/μ − r̂ ,
            a = −μ / (2ℰ) ,  ℰ = ½|v|² − μ/|r|  (vis-viva),
            i = arccos(h_z / |h|) ,
            Ω = atan2(n_y, n_x)  con n⃗ = k̂ × h ,
            ω = ∠(n⃗, e⃗) ,  ν = ∠(e⃗, r) .

        Se adjunta el residuo de vis-viva |v² − μ(2/r − 1/a)| y el
        movimiento medio n = √(μ/a³) (elipses).
        """
        r = np.asarray(position, dtype=np.float64).ravel()
        v = np.asarray(velocity, dtype=np.float64).ravel()
        if r.size != 3 or v.size != 3:
            raise ValueError("position y velocity deben ser 3-vectores.")
        mu = float(mu_gravitational)
        if mu <= 0.0:
            raise ValueError("mu_gravitational debe ser positivo.")
        r_norm = float(np.linalg.norm(r))
        v_norm = float(np.linalg.norm(v))
        if r_norm < _MACHINE_EPS:
            raise ValueError("Radio orbital nulo.")
        h_vec = np.cross(r, v)
        h_norm = float(np.linalg.norm(h_vec))
        e_vec = np.cross(v, h_vec) / mu - r / r_norm
        e = float(np.linalg.norm(e_vec))
        energy = 0.5 * v_norm ** 2 - mu / r_norm
        if abs(energy) < _MACHINE_EPS:
            a = float("inf")
        else:
            a = float(-mu / (2.0 * energy))
        if h_norm < _MACHINE_EPS:
            inc = 0.0
        else:
            inc = float(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))
        k_hat = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k_hat, h_vec)
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm < _MACHINE_EPS:
            Omega = 0.0
        else:
            Omega = float(np.arctan2(n_vec[1], n_vec[0])) % _TWO_PI
        if n_norm < _MACHINE_EPS or e < _MACHINE_EPS:
            omega_arg = 0.0
        else:
            cos_w = float(n_vec @ e_vec / (n_norm * e))
            omega_arg = float(np.arccos(np.clip(cos_w, -1.0, 1.0)))
            if e_vec[2] < 0.0:
                omega_arg = _TWO_PI - omega_arg
        if e < _MACHINE_EPS:
            nu = 0.0
        else:
            cos_nu = float(e_vec @ r / (e * r_norm))
            nu = float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))
            if float(r @ v) < 0.0:
                nu = _TWO_PI - nu
        if np.isfinite(a) and a > 0.0:
            vis_viva = abs(v_norm ** 2 - mu * (2.0 / r_norm - 1.0 / a))
            mean_motion = float(np.sqrt(mu / (a ** 3)))
            period = _TWO_PI / mean_motion
        else:
            vis_viva = 0.0 if not np.isfinite(a) else abs(
                v_norm ** 2 - 2.0 * mu / r_norm)
            mean_motion = 0.0
            period = float("inf")
        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": inc,
            "ascending_node": Omega,
            "argument_periapsis": omega_arg,
            "true_anomaly": nu,
            "specific_energy": float(energy),
            "angular_momentum": h_vec,
            "eccentricity_vector": e_vec,
            "vis_viva_residual": float(vis_viva),
            "mean_motion": float(mean_motion),
            "period": float(period),
            "is_elliptic": bool(np.isfinite(a) and a > 0.0 and e < 1.0),
        }

    # ── II.ω  MORFISMO TERMINAL DE LA FASE II ─────────────────────────────
    def induce_poincare_consensus_germ(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
        poincare_germ: Optional[_PoincareKleisliGerm] = None,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        orbit_period_T: float = 1.0,
        symplectic_monodromy: Optional[np.ndarray] = None,
        hessian: Optional[np.ndarray] = None,
        q_traj_cartan: Optional[np.ndarray] = None,
        p_traj_cartan: Optional[np.ndarray] = None,
        H_traj_cartan: Optional[np.ndarray] = None,
    ) -> _PoincareConsensusGerm:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Ensambla el gérmen de Poincaré–Consenso

            𝒢_II = (𝒢_I, 𝔇, F_Markov, F_Sp, ℳ, ρ, KAM, Lyap, 𝒲, 𝒜, twist)

        Este método **es** el arranque formal de
        `_CHSHVerifier.certify_poincare_sequitos`.
        """
        if poincare_germ is None:
            poincare_germ = self._poincare_germ
        if poincare_germ is None:
            poincare_germ = _NumericalCore.synthesize_poincare_kleisli_germ(
                affinity_matrix, regularizer=self._reg)
            self._poincare_germ = poincare_germ

        degroot = self.compute(opinion_vector, affinity_matrix, steps)

        # Floquet del kernel de DeGroot (dinámica discreta de consenso).
        M_eff = poincare_germ.markov_germ.kernel
        floquet = self.floquet_lyapunov_factorization(M_eff, orbit_period_T)

        # Floquet hamiltoniano opcional sobre una monodromía en Sp(2n).
        symplectic_floquet: Optional[_FloquetLyapunovWitness] = None
        if symplectic_monodromy is not None:
            Msp = np.asarray(symplectic_monodromy, dtype=np.float64)
            if Msp.shape[0] == poincare_germ.omega.shape[0] and Msp.shape[0] % 2 == 0:
                symplectic_floquet = self.floquet_lyapunov_factorization(
                    Msp, orbit_period_T, omega=poincare_germ.omega,
                    drop_parabolic=True)

        melnikov: Optional[_MelnikovWitness] = None
        if (q0_trajectory is not None
                and h0_grad is not None and h1_grad is not None):
            try:
                melnikov = self.melnikov_function(
                    q0_trajectory, dt_trajectory, h0_grad, h1_grad,
                    poincare_germ.omega)
            except ValueError as exc:
                logger.error("Mel'nikov falló: %s", exc)

        rotation: Optional[_RotationNumberWitness] = None
        if orbit_points_for_rotation is not None:
            try:
                rotation = self.rotation_number_poincare(orbit_points_for_rotation)
            except ValueError as exc:
                logger.error("Rotación falló: %s", exc)

        # Frecuencias: preferir el generador hamiltoniano si existe.
        src = symplectic_floquet if symplectic_floquet is not None else floquet
        if frequency_vector is None:
            eig_A = la.eigvals(src.generator)
            n_freq = max(src.generator.shape[0] // 2, 1)
            omega_freq = np.sort(np.abs(np.imag(eig_A)))[::-1][:n_freq]
            if omega_freq.size == 0 or np.all(omega_freq < 1e-12):
                omega_freq = np.ones(n_freq, dtype=np.float64)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
        kam = self.certify_kam_torus(omega_freq, birkhoff_residual=birkhoff_residual)

        # Lyapunov: hamiltoniano sólo sobre monodromía simpléctica.
        if symplectic_monodromy is not None and symplectic_floquet is not None:
            lyapunov = self.lyapunov_spectrum_benettin(
                np.asarray(symplectic_monodromy, dtype=np.float64),
                orbit_period_T=orbit_period_T,
                hamiltonian_pairing=True,
            )
        else:
            lyapunov = self.lyapunov_spectrum_benettin(
                M_eff, orbit_period_T=orbit_period_T, hamiltonian_pairing=False)

        williamson: Optional[_WilliamsonWitness] = None
        if hessian is not None:
            try:
                williamson = _NumericalCore.williamson_classify(
                    hessian, poincare_germ.omega)
            except (ValueError, SymplecticDimensionError) as exc:
                logger.error("Williamson falló: %s", exc)

        cartan: Optional[_PoincareCartanWitness] = None
        if (q_traj_cartan is not None and p_traj_cartan is not None
                and H_traj_cartan is not None):
            try:
                cartan = self.poincare_cartan_integral(
                    q_traj_cartan, p_traj_cartan, H_traj_cartan, dt_trajectory)
            except ValueError as exc:
                logger.error("Poincaré–Cartan falló: %s", exc)

        return _PoincareConsensusGerm(
            kleisli_germ=poincare_germ,
            degroot=degroot,
            floquet=floquet,
            melnikov=melnikov,
            rotation=rotation,
            kam=kam,
            lyapunov=lyapunov,
            williamson=williamson,
            cartan=cartan,
            moser_twist=None,
            symplectic_floquet=symplectic_floquet,
        )


class _UhlmannFidelity:
    """
    Fase II (continuación cuántica). Fidelidad de Uhlmann y lifting de Bell.
    Sin cambios de firma respecto a 5.0.
    """

    def __init__(self, regularizer: float = _HIGHAM_TIKHONOV_REG) -> None:
        self._reg = max(float(regularizer), _HIGHAM_TIKHONOV_REG)

    def _prepare_state(
        self, matrix: np.ndarray, name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(matrix)
        _NumericalCore.assert_square(name, a)
        _NumericalCore.assert_finite(name, a)
        herm = _NumericalCore.higham_nearest_hermitian(a)
        evals, evecs = la.eigh(herm)
        evals = np.real(evals)
        evals = np.maximum(evals, 0.0)
        tr = _NumericalCore.kahan_babuska_neumaier_sum(evals)
        if tr > _MACHINE_EPS:
            evals = evals / tr
        else:
            evals = np.zeros_like(evals)
            evals[-1] = 1.0
        rho = evecs @ (evals[:, None] * evecs.T.conj())
        return rho, evals, evecs

    def compute(self, rho: np.ndarray, sigma: np.ndarray) -> _UhlmannFidelityResult:
        rho_p, e_r, v_r = self._prepare_state(rho, "rho")
        sig_p, e_s, v_s = self._prepare_state(sigma, "sigma")
        if rho_p.shape != sig_p.shape:
            raise ValueError("rho y sigma deben tener la misma dimensión.")
        sqrt_r = v_r @ (np.sqrt(np.clip(e_r, 0.0, None))[:, None] * v_r.T.conj())
        sqrt_s = v_s @ (np.sqrt(np.clip(e_s, 0.0, None))[:, None] * v_s.T.conj())
        svals = np.real(la.svdvals(sqrt_r @ sqrt_s))
        amp = _NumericalCore.kahan_babuska_neumaier_sum(np.clip(svals, 0.0, None))
        fidelity = float(amp * amp)
        if fidelity < 0.0 and abs(fidelity) < 1e-12:
            fidelity = 0.0
        fidelity = float(min(max(fidelity, 0.0), 1.0))
        delta = _NumericalCore.higham_nearest_hermitian(rho_p - sig_p)
        ev_d = np.real(la.eigvalsh(delta))
        td = 0.5 * _NumericalCore.kahan_babuska_neumaier_sum(np.abs(ev_d))
        td = float(min(max(td, 0.0), 1.0))
        root_f = float(np.sqrt(fidelity))
        bures = float(np.arccos(min(max(root_f, 0.0), 1.0)))
        fuchs_lo = float(1.0 - root_f)
        fuchs_hi = float(np.sqrt(max(1.0 - fidelity, 0.0)))
        converged = bool(0.0 <= fidelity <= 1.0 + _MACHINE_EPS)
        return _UhlmannFidelityResult(
            fidelity=fidelity, converged=converged, bures_angle=bures,
            trace_distance=td, fuchs_lower=fuchs_lo, fuchs_upper=fuchs_hi,
        )

    @staticmethod
    def horodecki_tensor(rho_ab: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_ab, dtype=np.complex128)
        t_mat = np.zeros((3, 3), dtype=np.float64)
        for i, si in enumerate(_PAULI):
            for j, sj in enumerate(_PAULI):
                t_mat[i, j] = float(np.real(np.trace(rho @ np.kron(si, sj))))
        return t_mat

    @staticmethod
    def _optimal_chsh_block(
        t_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        u_svd, s_vals, vt = la.svd(t_mat, full_matrices=False)
        s_vals = np.real(s_vals)
        forecast = float(2.0 * np.sqrt(max(
            s_vals[0] ** 2 + (s_vals[1] ** 2 if s_vals.size > 1 else 0.0), 0.0)))
        if s_vals.size == 0:
            return np.zeros((2, 2), dtype=np.float64), s_vals, 0.0
        u0 = u_svd[:, 0]
        u1 = u_svd[:, 1] if u_svd.shape[1] > 1 else np.zeros_like(u0)
        v0 = vt[0, :]
        v1 = vt[1, :] if vt.shape[0] > 1 else np.zeros_like(v0)
        a = (u0 + u1) / np.sqrt(2.0)
        ap = (u0 - u1) / np.sqrt(2.0)
        e = np.array([
            [float(a @ t_mat @ v0), float(a @ t_mat @ v1)],
            [float(ap @ t_mat @ v0), float(ap @ t_mat @ v1)],
        ], dtype=np.float64)
        return e, s_vals, forecast

    @staticmethod
    def _lhv_from_opinions(opinion_vector: np.ndarray) -> np.ndarray:
        x = _NumericalCore.assert_vec("opinion_vector", opinion_vector)
        mean = _NumericalCore.kahan_babuska_neumaier_sum(x) / max(x.size, 1)
        hidden = float(np.clip(np.tanh(mean), -1.0, 1.0))
        return hidden * np.ones((2, 2), dtype=np.float64)

    def induce_bell_correlation_germ(
        self,
        rho_or_correlation: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        opinion_vector: Optional[np.ndarray] = None,
    ) -> _BellCorrelationGerm:
        if opinion_vector is not None:
            e = self._lhv_from_opinions(opinion_vector)
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
            return _BellCorrelationGerm(
                correlation_matrix=e,
                horodecki_singular_values=np.array([], dtype=np.float64),
                tsirelson_forecast=float(abs(2.0 * e[0, 0])),
                from_quantum=False, physical=phys,
            )
        raw = np.asarray(rho_or_correlation)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError("rho_or_correlation plana no cuadrado perfecto.")
            raw = raw.reshape(side, side)
        _NumericalCore.assert_square("rho_or_correlation", raw)
        _NumericalCore.assert_finite("rho_or_correlation", raw)
        if sigma is not None:
            _ = self.compute(raw, np.asarray(sigma))
            sig = np.asarray(sigma)
            if raw.shape == (2, 2) and sig.shape == (2, 2):
                return _BellCorrelationGerm(
                    correlation_matrix=np.zeros((2, 2), dtype=np.float64),
                    horodecki_singular_values=np.zeros(3, dtype=np.float64),
                    tsirelson_forecast=0.0, from_quantum=False, physical=True,
                )
        if raw.shape == (4, 4):
            rho, _e, _v = self._prepare_state(raw, "rho_ab")
            t_mat = self.horodecki_tensor(rho)
            e, s_vals, forecast = self._optimal_chsh_block(t_mat)
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-9))
            return _BellCorrelationGerm(
                correlation_matrix=np.real(e).astype(np.float64, copy=False),
                horodecki_singular_values=np.asarray(s_vals, dtype=np.float64),
                tsirelson_forecast=float(min(forecast, _TSIRELSON_BOUND)),
                from_quantum=True, physical=phys,
            )
        if raw.shape == (2, 2):
            e = np.real(np.asarray(raw, dtype=np.float64))
            phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
            terms = np.array([e[0, 0], -e[0, 1], e[1, 0], e[1, 1]], dtype=np.float64)
            forecast = abs(_NumericalCore.kahan_babuska_neumaier_sum(terms))
            return _BellCorrelationGerm(
                correlation_matrix=e,
                horodecki_singular_values=np.array([], dtype=np.float64),
                tsirelson_forecast=float(forecast),
                from_quantum=False, physical=phys,
            )
        raise ValueError(
            "induce_bell_correlation_germ espera E 2×2, ρ_AB 4×4 "
            f"o un vector de opinión; recibido {raw.shape}.")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE III — CHSH, TSIRELSON, POPESCU–ROHRLICH, POINCARÉ–CARTAN, KEPLER   ██
# ██  Continuación directa del morfismo II.ω.                                  ██
# ██  Morfismo terminal (III.ω): certify_poincare_sequitos                    ██
# ██                           → PoincareSequitosCertificate                  ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
# PUENTE Φ_II ▸ Φ_III
# El valor de retorno de II.ω (`_PoincareConsensusGerm`) es el germen
# dinámico de `_CHSHVerifier.certify_poincare_sequitos`.
# =============================================================================
@dataclass(frozen=True)
class _CHSHResult:
    """Resultado certificado de la desigualdad CHSH."""

    s_value: float
    verdict: str
    classical_gap: float
    tsirelson_gap: float
    pr_gap: float
    physical: bool
    horodecki_bound: float


@dataclass(frozen=True)
class PoincareSequitosCertificate:
    r"""
    **Certificado completo de Poincaré–Séquitos.**

    Morfismo terminal global Φ_III ∘ Φ_II ∘ Φ_I. Integra los tres gérmenes
    anidados con los invariantes de mecánica celeste 5.1 y los certificados
    de CHSH/Tsirelson/Popescu–Rohrlich. Porta los testigos *reales* del
    gérmen de Fase II (no reconstrucciones dummy).
    """

    n_agents: int
    dim_two_n: int
    kernel_row_stochastic_residual: float
    kernel_perron_residual: float
    kernel_is_reversible: bool
    section_is_transversal: bool
    section_transversal_certificate: float
    section_is_dynamically_transversal: bool
    section_flow_flux: float
    degroot_final_opinion: np.ndarray
    degroot_fiedler: float
    degroot_spectral_gap: float
    degroot_mixing_rate: float
    degroot_connected: bool
    degroot_verdict: str
    floquet_max_multiplier: float
    floquet_lyapunov_max: float
    floquet_log_residual: float
    kam_diophantine_gamma: float
    kam_diophantine_tau: float
    kam_stable: bool
    lyapunov_spectrum: np.ndarray
    lyapunov_kaplan_yorke: float
    lyapunov_ks_entropy: float
    melnikov_simple_zeros: Optional[int]
    melnikov_chaotic: Optional[bool]
    rotation_number: Optional[float]
    rotation_is_rational: Optional[bool]
    poincare_cartan_integral: float
    kepler_elements: Optional[dict]
    chsh_s_value: float
    chsh_verdict: str
    heyting_verdict: str
    is_monodromy_stable: bool
    krein_definite: bool = False
    nekhoroshev_time: float = 0.0
    williamson_linearly_stable: Optional[bool] = None
    cartan_is_closed: Optional[bool] = None
    moser_is_twist: Optional[bool] = None
    section: Optional[_PoincareSectionWitness] = None
    floquet: Optional[_FloquetLyapunovWitness] = None
    kam: Optional[_KamTorusWitness] = None
    lyapunov: Optional[_LyapunovSpectrumWitness] = None


class _CHSHVerifier:
    r"""
    Fase III. Observable de Bell–CHSH extendido con los morfismos de
    Poincaré 5.1: invariante de Poincaré–Cartan, elementos de Kepler,
    Krein, Nekhoroshev y certificado global de Poincaré–Séquitos.
    """

    def __init__(self, germ: Optional[_BellCorrelationGerm] = None) -> None:
        self._germ = germ

    def _resolve_e(
        self, correlation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        e = np.asarray(correlation_matrix)
        if e.size == 0 and self._germ is not None:
            g = self._germ
            return g.correlation_matrix, g.tsirelson_forecast, g.physical
        if e.ndim == 1 and e.size == 4:
            e = e.reshape(2, 2)
        if e.shape != (2, 2):
            logger.error("Matriz de correlaciones debe ser 2×2.")
            return (np.full((2, 2), np.inf, dtype=np.float64), float("inf"), False)
        _NumericalCore.assert_finite("correlation_matrix", e)
        e = np.real(np.asarray(e, dtype=np.float64))
        phys = bool(np.all(np.abs(e) <= _CORRELATOR_BOUND + 1e-12))
        forecast = (self._germ.tsirelson_forecast
                    if self._germ is not None
                    and self._germ.correlation_matrix.shape == (2, 2)
                    else float("nan"))
        return e, forecast, phys

    # ── III.1  CHSH (compatible 4.0 / 5.0) ────────────────────────────────
    def verify(self, correlation_matrix: np.ndarray) -> _CHSHResult:
        e, forecast, phys = self._resolve_e(correlation_matrix)
        if not np.all(np.isfinite(e)):
            return _CHSHResult(
                s_value=float("inf"), verdict="VETOED",
                classical_gap=float("inf"), tsirelson_gap=float("-inf"),
                pr_gap=float("-inf"), physical=False,
                horodecki_bound=float("nan"),
            )
        terms = np.array([e[0, 0], -e[0, 1], e[1, 0], e[1, 1]], dtype=np.float64)
        s_value = abs(_NumericalCore.kahan_babuska_neumaier_sum(terms))
        if not phys:
            logger.warning("Correladores no físicos: max|E|=%.6f > 1.",
                           float(np.max(np.abs(e))))
        if (not phys) or s_value > _TSIRELSON_BOUND + 8.0 * _MACHINE_EPS:
            verdict = "VETOED"
        elif s_value > _CLASSICAL_CHSH_BOUND:
            verdict = "COHERENT"
        else:
            verdict = "DEGRADED"
        horo = forecast if np.isfinite(forecast) else s_value
        return _CHSHResult(
            s_value=float(s_value), verdict=verdict,
            classical_gap=float(s_value - _CLASSICAL_CHSH_BOUND),
            tsirelson_gap=float(_TSIRELSON_BOUND - s_value),
            pr_gap=float(_PR_NOSIGNAL_BOUND - s_value),
            physical=bool(phys), horodecki_bound=float(horo),
        )

    # ── III.2  Fusión Heyting de los veredictos ──────────────────────────
    @staticmethod
    def _heyting_meet(*tokens: str) -> str:
        order = {"VETOED": 0, "DEGRADED": 1, "COHERENT": 2}
        if not tokens:
            return "COHERENT"
        return min(tokens, key=lambda t: order.get(t, 0))

    # ── III.ω  MORFISMO TERMINAL DE LA FASE III ──────────────────────────
    def certify_poincare_sequitos(
        self,
        germ: _PoincareConsensusGerm,
        correlation_matrix: np.ndarray,
        q_trajectory: Optional[np.ndarray] = None,
        p_trajectory: Optional[np.ndarray] = None,
        H_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        kepler_position: Optional[np.ndarray] = None,
        kepler_velocity: Optional[np.ndarray] = None,
        mu_gravitational: float = 1.0,
    ) -> PoincareSequitosCertificate:
        r"""
        **III.ω — Morfismo terminal global Φ_III ∘ Φ_II ∘ Φ_I.**

        Consume el gérmen de Poincaré–Consenso (cierre de Fase II) y
        sintetiza el `PoincareSequitosCertificate` integrando los
        testigos *reales* (sección, Floquet, KAM, Lyapunov, Williamson,
        Cartan, Krein) con CHSH y Kepler.
        """
        chsh = self.verify(correlation_matrix)

        cartan_w = germ.cartan
        if cartan_w is None and (q_trajectory is not None
                                 and p_trajectory is not None
                                 and H_trajectory is not None):
            try:
                cartan_w = _DeGrootConsensus.poincare_cartan_integral(
                    q_trajectory, p_trajectory, H_trajectory, dt_trajectory)
            except ValueError as exc:
                logger.error("Poincaré–Cartan falló: %s", exc)
                cartan_w = None
        cartan_value = float(cartan_w.action) if cartan_w is not None else float("nan")

        kepler = None
        if kepler_position is not None and kepler_velocity is not None:
            try:
                kepler = _DeGrootConsensus.kepler_osculating_elements(
                    kepler_position, kepler_velocity, mu_gravitational)
            except ValueError as exc:
                logger.error("Kepler falló: %s", exc)

        kg = germ.kleisli_germ
        mg = kg.markov_germ
        dg = germ.degroot
        # Preferir Floquet hamiltoniano si existe.
        fl = germ.symplectic_floquet if germ.symplectic_floquet is not None else germ.floquet
        mu = fl.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
        lyap_max = float(germ.lyapunov.spectrum[0]) \
            if germ.lyapunov.spectrum.size else 0.0
        krein_def = bool(fl.krein.krein_definite if fl.krein is not None else False)

        dyn = kg.section.is_dynamically_transversal
        chart = kg.section.is_transversal
        section_verdict = "COHERENT" if (dyn or chart) else "VETOED"
        floquet_verdict = ("COHERENT" if max_mu <= 1.0 + _HARD_FLOQUET_MULTIPLIER_TOL
                           else ("DEGRADED" if max_mu <= 1.05 else "VETOED"))
        lyap_verdict = ("COHERENT" if lyap_max <= _HARD_LYAPUNOV_TOL
                        else ("DEGRADED" if lyap_max <= 10 * _HARD_LYAPUNOV_TOL
                              else "VETOED"))
        kam_verdict = "COHERENT" if germ.kam.kam_stable else "VETOED"
        mel_verdict = ("VETOED" if (germ.melnikov is not None
                                    and germ.melnikov.is_chaotic)
                       else "COHERENT")
        tokens = [
            dg.verdict, floquet_verdict, lyap_verdict,
            kam_verdict, section_verdict, mel_verdict, chsh.verdict,
        ]
        if germ.williamson is not None and not germ.williamson.is_linearly_stable:
            tokens.append("DEGRADED" if germ.williamson.hyperbolic_pairs == 0
                          else "VETOED")
        if cartan_w is not None and not cartan_w.is_closed:
            tokens.append("DEGRADED")
        if fl.krein is not None:
            if fl.krein.loxodromic > 0 or fl.krein.hyperbolic > 0:
                tokens.append("VETOED")
            elif not fl.krein.krein_definite:
                tokens.append("DEGRADED")
        if germ.moser_twist is not None and not germ.moser_twist.is_twist:
            tokens.append("DEGRADED")
        final_verdict = self._heyting_meet(*tokens)
        is_stable = bool(final_verdict == "COHERENT"
                         and (dyn or chart)
                         and max_mu <= 1.0 + _HARD_FLOQUET_MULTIPLIER_TOL)
        return PoincareSequitosCertificate(
            n_agents=mg.n_agents,
            dim_two_n=2 * mg.n_agents,
            kernel_row_stochastic_residual=mg.certificate.row_stochastic_residual,
            kernel_perron_residual=mg.certificate.perron_residual,
            kernel_is_reversible=bool(mg.certificate.is_reversible),
            section_is_transversal=bool(kg.section.is_transversal),
            section_transversal_certificate=float(
                kg.section.transversal_certificate),
            section_is_dynamically_transversal=bool(
                kg.section.is_dynamically_transversal),
            section_flow_flux=float(kg.section.flow_flux),
            degroot_final_opinion=np.asarray(dg.final_opinion, dtype=np.float64),
            degroot_fiedler=float(dg.fiedler_value),
            degroot_spectral_gap=float(dg.spectral_gap),
            degroot_mixing_rate=float(dg.mixing_rate),
            degroot_connected=bool(dg.connected),
            degroot_verdict=str(dg.verdict),
            floquet_max_multiplier=float(max_mu),
            floquet_lyapunov_max=float(lyap_max),
            floquet_log_residual=float(fl.log_residual),
            kam_diophantine_gamma=float(germ.kam.diophantine_gamma),
            kam_diophantine_tau=float(germ.kam.diophantine_tau),
            kam_stable=bool(germ.kam.kam_stable),
            lyapunov_spectrum=np.asarray(germ.lyapunov.spectrum, dtype=np.float64),
            lyapunov_kaplan_yorke=float(germ.lyapunov.kaplan_yorke_dimension),
            lyapunov_ks_entropy=float(germ.lyapunov.kolmogorov_sinai_entropy),
            melnikov_simple_zeros=(int(germ.melnikov.simple_zeros)
                                   if germ.melnikov else None),
            melnikov_chaotic=(bool(germ.melnikov.is_chaotic)
                              if germ.melnikov else None),
            rotation_number=(float(germ.rotation.rotation_number)
                             if germ.rotation else None),
            rotation_is_rational=(bool(germ.rotation.is_rational)
                                  if germ.rotation else None),
            poincare_cartan_integral=float(cartan_value),
            kepler_elements=kepler,
            chsh_s_value=float(chsh.s_value),
            chsh_verdict=str(chsh.verdict),
            heyting_verdict=final_verdict,
            is_monodromy_stable=is_stable,
            krein_definite=krein_def,
            nekhoroshev_time=float(germ.kam.nekhoroshev_time),
            williamson_linearly_stable=(
                bool(germ.williamson.is_linearly_stable)
                if germ.williamson is not None else None),
            cartan_is_closed=(bool(cartan_w.is_closed)
                              if cartan_w is not None else None),
            moser_is_twist=(bool(germ.moser_twist.is_twist)
                            if germ.moser_twist is not None else None),
            section=kg.section,
            floquet=fl,
            kam=germ.kam,
            lyapunov=germ.lyapunov,
        )


# =============================================================================
# MOTOR PRINCIPAL — INTEGRACIÓN Φ_III ∘ Φ_II ∘ Φ_I CON POINCARÉ
# =============================================================================
class ImperialSequitosEngine:
    r"""
    Motor de alta fidelidad para la Capa 1.5 (Séquitos Imperiales) con
    mecánica celeste de Poincaré 5.1.

    Compone las tres fases anidadas y expone la integración simpléctica
    de Maupertuis–Jacobi (Störmer–Verlet), el consenso de DeGroot, la
    fidelidad de Uhlmann, la violación de CHSH y los morfismos de
    Poincaré: Floquet–Lyapunov, Krein, Mel'nikov, rotación, twist, KAM,
    Nekhoroshev, Lyapunov hamiltoniano, Williamson, acción-ángulo,
    elementos de Kepler y Poincaré–Cartan.
    """

    def __init__(
        self,
        regularizer: float = _HIGHAM_TIKHONOV_REG,
        dimension_n: int = 2,
    ) -> None:
        self._reg: Final[float] = max(float(regularizer), _HIGHAM_TIKHONOV_REG)
        self._n: Final[int] = int(dimension_n)
        self._dim: Final[int] = 2 * int(dimension_n)
        self._J_canonical: Final[NDArray[np.float64]] = np.block([
            [np.zeros((self._n, self._n), dtype=np.float64),
             np.eye(self._n, dtype=np.float64)],
            [-np.eye(self._n, dtype=np.float64),
             np.zeros((self._n, self._n), dtype=np.float64)],
        ])
        self._markov_germ: _MarkovKleisliGerm = (
            _KleisliComposer.synthesize_markov_kleisli_germ(
                np.eye(2, dtype=np.float64), regularizer=self._reg))
        self._poincare_germ: _PoincareKleisliGerm = (
            _NumericalCore.synthesize_poincare_kleisli_germ(
                np.eye(2, dtype=np.float64), regularizer=self._reg))
        self._degroot = _DeGrootConsensus(
            regularizer=self._reg, germ=self._markov_germ,
            poincare_germ=self._poincare_germ)
        self._uhlmann = _UhlmannFidelity(regularizer=self._reg)
        self._bell_germ: _BellCorrelationGerm = (
            self._uhlmann.induce_bell_correlation_germ(
                np.zeros((2, 2), dtype=np.float64)))
        self._chsh = _CHSHVerifier(germ=self._bell_germ)
        self._last_poincare_germ: Optional[_PoincareConsensusGerm] = None

    # ── Integración simpléctica de Maupertuis–Jacobi ─────────────────────
    def compute_symplectic_maupertuis_step(
        self,
        current_z: NDArray[np.float64],
        dt: float,
        metric_G: NDArray[np.float64],
        potential_V_func: Callable[[NDArray[np.float64]], float],
        grad_V_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        total_energy_H0: float,
    ) -> SequitosEngineStepResult:
        r"""
        Paso de Störmer–Verlet en coordenadas Darboux:

            p_{1/2} = p_0 − (dt/2) ∇V(q_0) ,
            q_1     = q_0 + dt G⁻¹ p_{1/2} ,
            p_1     = p_{1/2} − (dt/2) ∇V(q_1) .

        Se reporta el residual simpléctico del jacobiano linealizado
        (Hessiana cinética) y el incremento de Poincaré–Cartan
        p̄·Δq − H Δt del paso.
        """
        c_z = np.asarray(current_z, dtype=np.float64).ravel()
        dim = c_z.size
        n = dim // 2
        q_0 = c_z[:n].copy()
        p_0 = c_z[n:].copy()
        G_mat = np.asarray(metric_G, dtype=np.float64)
        if G_mat.ndim != 2 or G_mat.shape[0] != n or G_mat.shape[1] != n:
            G_inv = np.eye(n, dtype=np.float64)
        else:
            try:
                G_inv = la.inv(G_mat)
            except (la.LinAlgError, ValueError):
                G_inv = np.eye(n, dtype=np.float64)
        grad_V_0 = np.asarray(grad_V_func(q_0), dtype=np.float64).ravel()
        p_half = p_0 - 0.5 * dt * grad_V_0
        q_1 = q_0 + dt * (G_inv @ p_half)
        grad_V_1 = np.asarray(grad_V_func(q_1), dtype=np.float64).ravel()
        p_1 = p_half - 0.5 * dt * grad_V_1
        next_z = np.concatenate([q_1, p_1])
        kinetic_energy = 0.5 * float(p_1.T @ G_inv @ p_1)
        potential_energy = float(potential_V_func(q_1))
        current_energy = kinetic_energy + potential_energy
        j_canon = np.block([
            [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
            [-np.eye(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
        ])
        h_hessian = np.block([
            [np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
            [np.zeros((n, n), dtype=np.float64), G_inv],
        ])
        jacobian_M = np.eye(dim, dtype=np.float64) + dt * (j_canon @ h_hessian)
        det_M = float(np.real(la.det(jacobian_M)))
        volume_drift = abs(det_M - 1.0)
        symplectic_res = _NumericalCore.symplectic_residual(jacobian_M, j_canon)
        kinetic_margin = 2.0 * (total_energy_H0 - potential_energy)
        refractive_n = np.sqrt(max(0.0, kinetic_margin))
        dq = q_1 - q_0
        dq_norm = float(la.norm(dq))
        maupertuis_action = refractive_n * dq_norm
        p_mid = 0.5 * (p_0 + p_1)
        cartan_inc = float(p_mid @ dq) - current_energy * float(dt)
        is_conserved = volume_drift <= _WILKINSON_LIMIT
        return SequitosEngineStepResult(
            next_state_z=next_z,
            hamiltonian_energy=current_energy,
            volume_drift=volume_drift,
            maupertuis_action=maupertuis_action,
            is_liouville_conserved=is_conserved,
            symplectic_residual=float(symplectic_res),
            cartan_increment=float(cartan_inc),
        )

    # ── Re-sincronización de gérmenes ────────────────────────────────────
    def _resync_markov_germ(
        self,
        affinity_matrix: np.ndarray,
        flow_vector: Optional[np.ndarray] = None,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
    ) -> _PoincareKleisliGerm:
        germ = _NumericalCore.synthesize_poincare_kleisli_germ(
            affinity_matrix, regularizer=self._reg,
            symmetrize=True, flow_vector=flow_vector,
            section_index=section_index, section_offset=section_offset,
            energy_level=energy_level)
        self._markov_germ = germ.markov_germ
        self._poincare_germ = germ
        self._degroot = _DeGrootConsensus(
            regularizer=self._reg, germ=germ.markov_germ,
            poincare_germ=germ)
        return germ

    # ── Fase I expuesta ──────────────────────────────────────────────────
    def kahan_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_neumann_sum(arr)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        return _NumericalCore.kahan_babuska_neumaier_sum(arr)

    def kleisli_compose(
        self,
        f: Callable[[Any], Tuple[Any, float]],
        g: Callable[[Any], Tuple[Any, float]],
    ) -> Callable[[Any], Tuple[Any, float]]:
        return _KleisliComposer.compose(f, g)

    def kleisli_unit(self, value: Any) -> Tuple[Any, float]:
        return _KleisliComposer.unit(value)

    def compose_markov_kernels(
        self, p_kernel: np.ndarray, q_kernel: np.ndarray,
    ) -> np.ndarray:
        return _KleisliComposer.compose_markov_kernels(
            p_kernel, q_kernel, floor=self._reg)

    def synthesize_markov_kleisli_germ(
        self, affinity_matrix: np.ndarray, symmetrize: bool = True,
    ) -> _MarkovKleisliGerm:
        germ = _KleisliComposer.synthesize_markov_kleisli_germ(
            affinity_matrix, regularizer=self._reg, symmetrize=symmetrize)
        self._markov_germ = germ
        self._degroot = _DeGrootConsensus(
            regularizer=self._reg, germ=germ,
            poincare_germ=self._poincare_germ)
        return germ

    def synthesize_poincare_kleisli_germ(
        self, affinity_matrix: np.ndarray,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareKleisliGerm:
        """Réplica pública del morfismo I.ω (arranque de Φ_II)."""
        germ = _NumericalCore.synthesize_poincare_kleisli_germ(
            affinity_matrix, regularizer=self._reg,
            section_index=section_index,
            section_offset=section_offset,
            energy_level=energy_level,
            flow_vector=flow_vector)
        self._poincare_germ = germ
        self._markov_germ = germ.markov_germ
        self._degroot = _DeGrootConsensus(
            regularizer=self._reg, germ=germ.markov_germ,
            poincare_germ=germ)
        return germ

    def markov_kleisli_germ_certificate(self) -> _MarkovKernelCertificate:
        return self._markov_germ.certificate

    def poincare_kleisli_germ_certificate(self) -> _SymplecticFormCertificate:
        return self._poincare_germ.form_certificate

    def williamson_classify(
        self, hessian: np.ndarray, omega: Optional[np.ndarray] = None,
    ) -> _WilliamsonWitness:
        O = self._poincare_germ.omega if omega is None else np.asarray(omega)
        return _NumericalCore.williamson_classify(hessian, O)

    # ── Fase II expuesta ─────────────────────────────────────────────────
    def compute_degroot_spectral_consensus(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> Tuple[np.ndarray, float, str]:
        result = self.compute_degroot_spectral_consensus_certified(
            opinion_vector, affinity_matrix, steps)
        return result.final_opinion, result.fiedler_value, result.verdict

    def compute_degroot_spectral_consensus_certified(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootConsensusResult:
        self._resync_markov_germ(affinity_matrix)
        return self._degroot.compute(opinion_vector, affinity_matrix, steps)

    def compute_uhlmann_fidelity(
        self, rho: np.ndarray, sigma: np.ndarray,
    ) -> float:
        return self._uhlmann.compute(rho, sigma).fidelity

    def compute_uhlmann_fidelity_certified(
        self, rho: np.ndarray, sigma: np.ndarray,
    ) -> _UhlmannFidelityResult:
        return self._uhlmann.compute(rho, sigma)

    def induce_bell_correlation_germ(
        self,
        rho_or_correlation: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        opinion_vector: Optional[np.ndarray] = None,
    ) -> _BellCorrelationGerm:
        germ = self._uhlmann.induce_bell_correlation_germ(
            rho_or_correlation, sigma=sigma, opinion_vector=opinion_vector)
        self._bell_germ = germ
        self._chsh = _CHSHVerifier(germ=germ)
        return germ

    def induce_poincare_consensus_germ(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        orbit_period_T: float = 1.0,
        flow_vector: Optional[np.ndarray] = None,
        symplectic_monodromy: Optional[np.ndarray] = None,
        hessian: Optional[np.ndarray] = None,
        q_traj_cartan: Optional[np.ndarray] = None,
        p_traj_cartan: Optional[np.ndarray] = None,
        H_traj_cartan: Optional[np.ndarray] = None,
    ) -> _PoincareConsensusGerm:
        """Réplica pública del morfismo II.ω (arranque de Φ_III)."""
        poincare_germ = self._resync_markov_germ(
            affinity_matrix, flow_vector=flow_vector)
        germ = self._degroot.induce_poincare_consensus_germ(
            opinion_vector, affinity_matrix, steps=steps,
            poincare_germ=poincare_germ,
            q0_trajectory=q0_trajectory, dt_trajectory=dt_trajectory,
            h0_grad=h0_grad, h1_grad=h1_grad,
            orbit_points_for_rotation=orbit_points_for_rotation,
            frequency_vector=frequency_vector,
            birkhoff_residual=birkhoff_residual,
            orbit_period_T=orbit_period_T,
            symplectic_monodromy=symplectic_monodromy,
            hessian=hessian,
            q_traj_cartan=q_traj_cartan,
            p_traj_cartan=p_traj_cartan,
            H_traj_cartan=H_traj_cartan,
        )
        self._last_poincare_germ = germ
        return germ

    def attach_moser_twist(
        self,
        germ: _PoincareConsensusGerm,
        rotation_samples: Sequence[_RotationNumberWitness],
        action_samples: Optional[np.ndarray] = None,
    ) -> _PoincareConsensusGerm:
        """Adjunta un certificado de twist de Moser al gérmen (inmutable)."""
        twist = _DeGrootConsensus.moser_twist(
            rotation_samples, action_samples=action_samples)
        attached = _PoincareConsensusGerm(
            kleisli_germ=germ.kleisli_germ, degroot=germ.degroot,
            floquet=germ.floquet, melnikov=germ.melnikov,
            rotation=germ.rotation, kam=germ.kam, lyapunov=germ.lyapunov,
            williamson=germ.williamson, cartan=germ.cartan,
            moser_twist=twist, symplectic_floquet=germ.symplectic_floquet,
        )
        self._last_poincare_germ = attached
        return attached

    # ── Fase III expuesta ────────────────────────────────────────────────
    def verify_chsh_violation(
        self, correlation_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        result = self._chsh.verify(correlation_matrix)
        return result.s_value, result.verdict

    def verify_chsh_violation_certified(
        self, correlation_matrix: np.ndarray,
    ) -> _CHSHResult:
        return self._chsh.verify(correlation_matrix)

    def certify_poincare_sequitos(
        self,
        poincare_consensus_germ: Optional[_PoincareConsensusGerm] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        q_trajectory: Optional[np.ndarray] = None,
        p_trajectory: Optional[np.ndarray] = None,
        H_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        kepler_position: Optional[np.ndarray] = None,
        kepler_velocity: Optional[np.ndarray] = None,
        mu_gravitational: float = 1.0,
    ) -> PoincareSequitosCertificate:
        r"""
        **Morfismo terminal global Φ_III ∘ Φ_II ∘ Φ_I.**

        Si no se suministra gérmen, usa el último inducido por
        `induce_poincare_consensus_germ`.
        """
        germ = poincare_consensus_germ or self._last_poincare_germ
        if germ is None:
            raise ValueError(
                "Debe suministrarse un gérmen de Poincaré–Consenso o "
                "haber invocado `induce_poincare_consensus_germ` previamente.")
        if correlation_matrix is None:
            correlation_matrix = self._bell_germ.correlation_matrix
        return self._chsh.certify_poincare_sequitos(
            germ=germ,
            correlation_matrix=correlation_matrix,
            q_trajectory=q_trajectory, p_trajectory=p_trajectory,
            H_trajectory=H_trajectory, dt_trajectory=dt_trajectory,
            kepler_position=kepler_position, kepler_velocity=kepler_velocity,
            mu_gravitational=mu_gravitational,
        )

    # ── Utilidades públicas de Poincaré ──────────────────────────────────
    @staticmethod
    def poisson_bracket(
        grad_f: np.ndarray, grad_g: np.ndarray, omega: np.ndarray,
    ) -> float:
        """{f,g} = (∇f)ᵀ Ω (∇g)."""
        return _NumericalCore.poisson_bracket(grad_f, grad_g, omega)

    @staticmethod
    def hamilton_jacobi_F2(
        q_old: np.ndarray, p_old: np.ndarray, q_new: np.ndarray,
        hessian_F2: Optional[np.ndarray] = None,
    ) -> _HamiltonJacobiWitness:
        """Función generatriz F₂ de Hamilton–Jacobi (hessiana simetrizada)."""
        return _NumericalCore.hamilton_jacobi_F2(
            q_old, p_old, q_new, hessian_F2=hessian_F2)

    @staticmethod
    def symplectic_gram_schmidt(
        vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        """Base S ∈ Sp(2n,ℝ) con Sᵀ Ω S = Ω (Parasjuk–de Gosson)."""
        return _NumericalCore.symplectic_gram_schmidt(vectors, omega)

    @staticmethod
    def compute_floquet_lyapunov_certified(
        M: np.ndarray, orbit_period_T: float,
        omega: Optional[np.ndarray] = None,
    ) -> _FloquetLyapunovWitness:
        """M = exp(T·A_F)·R_F, con Krein si Ω se da."""
        return _DeGrootConsensus.floquet_lyapunov_factorization(
            M, orbit_period_T, omega=omega)

    @staticmethod
    def compute_melnikov_certified(
        q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray, omega: np.ndarray,
    ) -> _MelnikovWitness:
        """ℳ(t₀) = ∫ {H₀,H₁}(q₀(t), t+t₀) dt (ceros simples)."""
        return _DeGrootConsensus.melnikov_function(
            q0_trajectory, dt, h0_grad, h1_grad, omega)

    @staticmethod
    def compute_rotation_number_certified(
        orbit_points: np.ndarray,
    ) -> _RotationNumberWitness:
        """ρ ∈ ℝ/ℤ = lim (1/(2π N)) Σ Δθ_i."""
        return _DeGrootConsensus.rotation_number_poincare(orbit_points)

    @staticmethod
    def certify_kam_torus(
        frequency_vector: np.ndarray, birkhoff_residual: float = 0.0,
    ) -> _KamTorusWitness:
        """Condición diofántica de Arnold |k·ω| ≥ γ/|k|^τ + Nekhoroshev."""
        return _DeGrootConsensus.certify_kam_torus(
            frequency_vector, birkhoff_residual=birkhoff_residual)

    @staticmethod
    def nekhoroshev_stability_time(
        n_dof: int, perturbation_eps: float,
    ) -> float:
        return _DeGrootConsensus.nekhoroshev_time(n_dof, perturbation_eps)

    @staticmethod
    def compute_lyapunov_spectrum_certified(
        M: np.ndarray,
        n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
        orbit_period_T: Optional[float] = None,
        hamiltonian_pairing: bool = False,
    ) -> _LyapunovSpectrumWitness:
        """Espectro completo de Lyapunov por QR de Benettin."""
        return _DeGrootConsensus.lyapunov_spectrum_benettin(
            M, n_iterations=n_iterations, orbit_period_T=orbit_period_T,
            hamiltonian_pairing=hamiltonian_pairing)

    @staticmethod
    def poincare_cartan_invariant(
        q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        H_trajectory: np.ndarray, dt: float,
    ) -> float:
        """Invariante ∮ p dq − H dt (acción escalar)."""
        return _DeGrootConsensus.poincare_cartan_integral(
            q_trajectory, p_trajectory, H_trajectory, dt).action

    @staticmethod
    def poincare_cartan_certified(
        q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        H_trajectory: np.ndarray, dt: float,
    ) -> _PoincareCartanWitness:
        """Testigo completo de Poincaré–Cartan (acción + cierre)."""
        return _DeGrootConsensus.poincare_cartan_integral(
            q_trajectory, p_trajectory, H_trajectory, dt)

    @staticmethod
    def action_angle_variables(
        q_periodic: np.ndarray, p_periodic: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """I_k = (1/2π) ∮ p_k dq_k ; θ_k fase canónica conjugada."""
        return _DeGrootConsensus.action_angle_variables(
            q_periodic, p_periodic)

    @staticmethod
    def kepler_osculating_elements(
        position: np.ndarray, velocity: np.ndarray,
        mu_gravitational: float = 1.0,
    ) -> dict:
        """Elementos orbitales (a, e, i, Ω, ω, ν) + vis-viva + n̄."""
        return _DeGrootConsensus.kepler_osculating_elements(
            position, velocity, mu_gravitational=mu_gravitational)

    @staticmethod
    def moser_twist_certificate(
        rotation_samples: Sequence[_RotationNumberWitness],
        action_samples: Optional[np.ndarray] = None,
    ) -> _MoserTwistWitness:
        return _DeGrootConsensus.moser_twist(rotation_samples, action_samples)


__all__ = [
    "ImperialSequitosEngine",
    "SequitosEngineStepResult",
    "PoincareSequitosCertificate",
    "SymplecticDimensionError",
]