# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Homotopic Tesserarios Agent (Capa 3 · Poincaré–Nested-Phases)       ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_tesserarios.py        ║
║ Versión: 4.1.0-Poincare-Floquet-Melnikov-KAM-Krein-Williamson-Nekhoroshev    ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA, CATEGORIAL Y CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Supervisa la consistencia homotópica no abeliana de las deliberaciones agénticas
en APU Filter v8.0, acoplando las aduanas categoriales de Quillen, Stasheff y Čech–Deligne
con la mecánica celeste de Henri Poincaré sobre tres fases anidadas
($\Phi_{\mathrm{III}} \circ \Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$).

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE 1 — OBSERVE ($\Phi_{\mathrm{I}}$) — ALGEBRA HOMOTÓPICA $A_\infty$ Y GERBES DE ČECH:
1. Cofibraciones de Quillen y Factorización Polar Simpléctica:
   Para una matriz jacobiana $M \in \mathrm{GL}(2n, \mathbb{R})$, la factorización polar de Higham $M = U P$
   obtiene la retracción ortogonal $U = M P^{-1} \in \mathrm{Sp}(2n, \mathbb{R}) \cap \mathrm{O}(2n) \cong U(n)$.
   El residuo simpléctico $\epsilon_{\mathrm{Sp}} = \|M^\top \Omega M - \Omega\|_F$ mide el defecto de cofibración.

2. Estructura de Álgebra $A_\infty$ de Stasheff y Pentágono $K_4$:
   El tensor homotópico $m_3 \in \mathrm{Hom}(A^{\otimes 3}, A)$ y el producto $m_2$ satisfacen la identidad del pentágono $K_4$:
   $$-m_2(m_3 \otimes \mathrm{id}) - m_2(\mathrm{id} \otimes m_3) + m_3(m_2 \otimes \mathrm{id} \otimes \mathrm{id}) - m_3(\mathrm{id} \otimes m_2 \otimes \mathrm{id}) + m_3(\mathrm{id} \otimes \mathrm{id} \otimes m_2) = 0$$
   medida por la norma de Frobenius $\|K_4\|_F \le \varepsilon_{\mathrm{Stasheff}}$.

3. Obstrucción de Gerbe No Abeliana de Čech–Deligne:
   Para una 2-cocadena de Čech $C \in \check{C}^2(\mathcal{U}, \mathcal{F})$, el coborde $\delta C$ mide la curvatura de la 2-gerbe:
   $\| \delta C \|_F = 0 \iff$ la gerbe es trivializable en $H^2(\mathcal{U}, \mathcal{F})$.

FASE 2 — ORIENT ($\Phi_{\mathrm{II}}$) — ADUANA DE MONODROMÍA Y ESTABILIDAD DE KREIN:
4. Monodromía de Floquet y Signatura de Krein–Moser:
   Para el sistema lineal periódico $\dot{x} = A(t) x$, la monodromía $M = X(T) \in \mathrm{Sp}(2n, \mathbb{R})$
   admite autoespacios elípticos $|\mu_k| = 1$. Un autoespacio es Krein-definido si la forma hermítica
   $i \Omega(v, \bar{v}) \neq 0$ retiene el mismo signo para todos los autovectores en el bloque elíptico.

5. Cota Diofántica KAM y Tiempo Exponencial de Nekhoroshev:
   Para frecuencias $\omega \in \mathbb{R}^n$, la condición $|\langle k, \omega \rangle| \ge \frac{\gamma}{\|k\|_1^\tau}$ ($\tau > n-1$)
   garantiza la estabilidad de toros KAM por un tiempo exponencialmente largo de Nekhoroshev:
   $$T_N \ge T_0 \exp\left(c \cdot \varepsilon^{-1/(2n)}\right)$$

FASE 3 — DECIDE/ACT ($\Phi_{\mathrm{III}}$) — ÍNFIMO DE HEYTING Y COLAPSO EN IRAM:
6. Permiso Global por Ínfimo (Meet) y Disparo BT151:
   El veredicto final $\nu_{\mathrm{global}} = \bigwedge_{k} \nu_k \in G_3 \triangleq \{\mathrm{VETOED}(0) \le \mathrm{DEGRADED}(1) \le \mathrm{COHERENT}(2)\}$
   gatilla el disyuntor BT151 [GPIO14] en IRAM ($t_{\mathrm{act}} \le 400 \text{ ns}$) si $\nu_{\mathrm{global}} = \mathrm{VETOED}$.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Iterator, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from app.core.inmune_system.imperial_tesserarios_engine import (
    ImperialTesserariosEngine,
    PoincareMonodromyCertificate as EnginePoincareMonodromyCertificate,
    PoincareMonodromyGerm,
    SymplecticDimensionError,
)

logger = logging.getLogger("APU.Agents.HomotopicTesserarios")

__version__: Final[str] = (
    "4.1.0-Poincare-Floquet-Melnikov-KAM-Krein-Williamson-Nekhoroshev"
)

# =============================================================================
# CONSTANTES UNIVERSALES DE PRECISIÓN METROLÓGICA (WILKINSON / HIGHAM / POINCARÉ)
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_HARD_STASHEFF_CEILING: Final[float] = 1.0e-8
_HARD_QUILLEN_TOLERANCE: Final[float] = 1.0e-8
_HARD_GERBE_TOLERANCE: Final[float] = 1.0e-4
_HARD_SULLIVAN_TOLERANCE: Final[float] = 1.0e-8
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_NS: Final[float] = 5.0
_CROWBAR_T_MIN_NS: Final[float] = 380.0
_CROWBAR_T_MAX_NS: Final[float] = 420.0
_MIN_SINGULAR_VALUE_FLOOR: Final[float] = 1.0e-12
_PENTAGON_EINSUM_DIM_CAP: Final[int] = 24
_DEGRADATION_FACTOR: Final[float] = 0.01
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1.0e-12

# ── Constantes específicas de Poincaré / KAM / Floquet / Nekhoroshev ────────
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
_DIOPHANTINE_TAU_FLOOR: Final[float] = 1.0 + 1.0e-6   # > n−1 (n=2 para 4D)
_PARABOLIC_MULTIPLIER_TOL: Final[float] = 1.0e-6
_TWO_PI: Final[float] = 2.0 * float(np.pi)
_NEKHOROSHEV_PREFACTOR: Final[float] = 0.5
_WILLIAMSON_IMAG_RATIO: Final[float] = 1.0e-8


# #############################################################################
#                                                                             #
#  FASE I                                                                     #
#  NÚCLEO ESPECTRAL, LIOUVILLE, QUILLEN-POLAR, HOCHSCHILD, ČECH, POINCARÉ      #
#                                                                             #
#  Objetos: jacobiano, m₂, m₃, cocadena Čech, sección Σ, monodromía M,        #
#           órbita q₀(t), espectro QR, forma de Poincaré–Cartan, Williamson.  #
#  Morfismos: certificación de Ω, residual simpléctico, factorización polar,  #
#             asociador A∞, coborde de Čech, sección transversal dinámica,    #
#             Floquet–Lyapunov, Krein–Moser, Mel'nikov, rotación, twist,      #
#             KAM, Nekhoroshev, Lyapunov hamiltoniano, acción de Cartan.      #
#  Cierre formal (I.ω): assemble_poincare_homotopy_jet → _PoincareHomotopyJet  #
#             ≡ dominio de HomotopicTesserariosAgent (Fase II).               #
#                                                                             #
# #############################################################################

class HeytingVerdict(Enum):
    r"""
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

        VETOED = 0  ≤  DEGRADED = 1  ≤  COHERENT = 2

        a ∧ b = min(a,b) ,  a ∨ b = max(a,b) ,
        a → b = ⊤ si a ≤ b , else b ,  ¬a = a → ⊥ .
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc


# -----------------------------------------------------------------------------
# I.0 — Geometría simpléctica y testigos de Fase I
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SymplecticForm:
    r"""
    2-forma canónica de Liouville Ω ∈ ⋀²(T*Q)*.

        Ω = [[0, I_q], [−I_q, 0]] ,
        Ωᵀ = −Ω ,  Ω² = −I ,  det Ω = 1 ,  ‖Ω‖_F = √(2q) ,
        vol_Liouville = Ωⁿ / n!  (= 1 en el chart de Darboux).
    """

    matrix: np.ndarray
    half_dim: int
    frobenius_norm: float
    liouville_volume: float = 1.0

    @classmethod
    def from_dimension(cls, n: int) -> "_SymplecticForm":
        if n <= 0 or n % 2 != 0:
            raise ValueError(f"Darboux exige dimensión par positiva; n={n}.")
        half = n // 2
        i_half = np.eye(half, dtype=np.float64)
        z_half = np.zeros((half, half), dtype=np.float64)
        omega = np.block([[z_half, i_half], [-i_half, z_half]])
        det_o = float(np.real(np.linalg.det(omega)))
        vol = float(np.sqrt(max(det_o, 0.0)))
        return cls(
            matrix=omega, half_dim=half,
            frobenius_norm=float(np.sqrt(n)),
            liouville_volume=vol,
        )

    def verify(self, atol: float = _STRUCTURE_ATOL) -> None:
        omega = self.matrix
        n = omega.shape[0]
        scale = max(self.frobenius_norm, 1.0)
        skew = float(la.norm(omega + omega.T, "fro"))
        ac = float(la.norm(omega @ omega + np.eye(n), "fro"))
        det_res = abs(float(np.linalg.det(omega)) - 1.0)
        if skew > atol * scale:
            raise ValueError(f"Ω no anti-simétrica: ‖Ω+Ωᵀ‖_F={skew:.3e}")
        if ac > atol * scale:
            raise ValueError(f"Ω no casi-compleja: ‖Ω²+I‖_F={ac:.3e}")
        if det_res > 1.0e-6 * n:
            raise ValueError(f"det(Ω)≠1: |det−1|={det_res:.3e}")


@dataclass(frozen=True, slots=True)
class _QuillenWitness:
    """Testigos de la factorización polar M = U P (Quillen)."""

    symplectic_residual: float
    symplectic_residual_rel: float
    det_residual: float
    cofibration_residual: float
    fibration_we_residual: float
    polar_condition: float


# -----------------------------------------------------------------------------
# I.0.bis — Testigos de Poincaré (Fase I, extensión 4.1)
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _PoincareSectionWitness:
    r"""
    Sección transversal de Poincaré Σ.

        n_Σ ∈ ℝ^{2n} : normal unitaria (e_k del bloque q).
        Transversalidad *dinámica*: n_Σ · X_H ≠ 0.
        La no-degeneración Ω n_Σ ≠ 0 es sanity del chart de Darboux.
    """

    normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    transversal_certificate: float
    is_transversal: bool
    flow_flux: float = 0.0
    is_dynamically_transversal: bool = False


@dataclass(frozen=True, slots=True)
class _WilliamsonWitness:
    r"""
    Clasificación de Williamson de un equilibrio hamiltoniano.

    La matriz K = J A, J = Ω⁻¹ = −Ω, A = Hess H, tiene espectro cerrado
    por (λ, −λ, λ̄, −λ̄). Se cuentan pares elípticos (±iω), hiperbólicos
    (±λ), foco–foco (±α±iβ) y parabólicos (λ = 0).
    """

    elliptic_pairs: int
    hyperbolic_pairs: int
    focus_focus_pairs: int
    parabolic_multiplicity: int
    is_linearly_stable: bool


@dataclass(frozen=True, slots=True)
class _KreinWitness:
    r"""
    Clasificación de Krein–Moser de los multiplicadores de Floquet.

      * elípticos   : |μ| = 1, μ ≠ ±1
      * parabólicos : μ = ±1
      * hiperbólicos: μ ∈ ℝ, |μ| ≠ 1
      * loxodrómicos: μ ∈ ℂ \ ℝ, |μ| ≠ 1

    krein_definite ⇔ elípticos simples de signatura definida (estabilidad
    fuerte de Krein).
    """

    elliptic: int
    hyperbolic: int
    loxodromic: int
    parabolic: int
    krein_definite: bool
    on_unit_circle: int


@dataclass(frozen=True, slots=True)
class _FloquetLyapunovWitness:
    r"""
    Factorización de Floquet–Lyapunov de la monodromía M:

        M = exp(T · A_F) · R_F ,
        μ_k = eig(M) ,  λ_k^{char} = Re Log(μ_k) / T .
    """

    generator: np.ndarray
    periodic_part: np.ndarray
    log_residual: float
    is_real_logarithm: bool
    floquet_multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    krein: Optional[_KreinWitness] = None
    poincare_map_multipliers: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class _MelnikovWitness:
    r"""
    Función de Mel'nikov ℳ(t₀) = ∫ {H₀,H₁}(q₀(t), t+t₀) dt.

    Cero *simple*: ℳ(t₀)=0 y ℳ'(t₀)≠0 ⇒ intersección transversal
    W^s ∩ W^u ⇒ caos homoclínico (herradura de Smale).
    """

    melnikov_values: np.ndarray
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    melnikov_derivative_min: float = 0.0


@dataclass(frozen=True, slots=True)
class _RotationNumberWitness:
    r"""
    Número de rotación ρ = lim (1/(2π N)) Σ Δθ_i ∈ ℝ/ℤ.

    Conmensurable ⇔ ρ ∈ ℚ (isla KAM periódica / Poincaré–Birkhoff).
    """

    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float


@dataclass(frozen=True, slots=True)
class _MoserTwistWitness:
    r"""
    Certificado de twist de Moser: |∂ρ/∂I| ≥ ν > 0 implica persistencia
    de curvas invariantes (teorema de Poincaré–Birkhoff / Moser).
    """

    twist_value: float
    is_twist: bool
    intersection_property: bool


@dataclass(frozen=True, slots=True)
class _KamTorusWitness:
    r"""
    Certificado KAM diofantino |k·ω| ≥ γ / |k|^τ (τ > n−1), con tiempo
    de Nekhoroshev T_N ~ exp(c ε^{−1/(2n)}).
    """

    frequency_vector: np.ndarray
    diophantine_gamma: float
    diophantine_tau: float
    birkhoff_residual: float
    kam_stable: bool
    iterations: int
    nekhoroshev_time: float = 0.0
    worst_small_divisor: float = 0.0


@dataclass(frozen=True, slots=True)
class _LyapunovSpectrumWitness:
    r"""
    Espectro de Lyapunov λ₁ ≥ … ≥ λ_{2n} por QR de Benettin, con
    emparejamiento hamiltoniano λᵢ ↔ −λ_{2n+1−i}.

    Kaplan–Yorke D_KY = j + Σ_{i≤j} λ_i / |λ_{j+1}| ;
    Pesin h_KS = Σ_{λ_i>0} λ_i.
    """

    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    is_chaotic: bool


@dataclass(frozen=True, slots=True)
class _PoincareCartanWitness:
    r"""
    Invariante integral de Poincaré–Cartan 𝒜 = ∫ (p·dq − H dt)
    a lo largo de un arco. Sobre una órbita periódica, 𝒜 es el
    invariante relativo; `closure_residual` mide el defecto de cierre.
    """

    action: float
    closure_residual: float
    is_closed: bool


# -----------------------------------------------------------------------------
# I.1 — Núcleo espectral
# -----------------------------------------------------------------------------
class _HomotopySpectralCore:
    r"""
    Núcleo de cómputo espectral y homológico. Opera en M_n(ℝ) con norma
    de Hilbert–Schmidt; las deflaciones siguen a Wilkinson y la polar a
    Higham. En v4.1.0 se añaden los morfismos de Poincaré refinados
    (transversalidad dinámica, Krein, Williamson, Nekhoroshev, Cartan).
    """

    # ── I.1.1  Formas, pisos de Wilkinson y residuales relativos ────────
    @staticmethod
    def frobenius(array: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(array, dtype=np.float64)))

    @classmethod
    def relative_frobenius(cls, residual: np.ndarray, scale_of: np.ndarray) -> float:
        denom = max(cls.frobenius(scale_of), _MIN_SINGULAR_VALUE_FLOOR)
        return cls.frobenius(residual) / denom

    @staticmethod
    def wilkinson_deflation_floor(matrix: np.ndarray) -> float:
        if matrix.size == 0:
            return _MIN_SINGULAR_VALUE_FLOOR
        fro_norm = float(la.norm(matrix, "fro"))
        return max(fro_norm * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
                   _MIN_SINGULAR_VALUE_FLOOR)

    @staticmethod
    def _as_real(name: str, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        if np.iscomplexobj(arr) and np.max(np.abs(np.imag(arr))) > 1.0e-14:
            raise ValueError(f"{name} no es real (Im ≠ 0).")
        return np.real(arr).astype(np.float64, copy=False)

    @staticmethod
    def symplectic_pairing(
        u: np.ndarray, v: np.ndarray, omega: np.ndarray,
    ) -> float:
        r"""Producto simpléctico compensado ω(u, v) = uᵀ Ω v."""
        uu = np.asarray(u, dtype=np.float64).ravel()
        vv = np.asarray(v, dtype=np.float64).ravel()
        ov = np.asarray(omega, dtype=np.float64) @ vv
        if uu.size != ov.size:
            raise ValueError("symplectic_pairing: dimensiones incompatibles.")
        return float(np.dot(uu, ov))

    # ── I.1.2  Quillen polar ─────────────────────────────────────────────
    @classmethod
    def symplectic_pullback_residual(
        cls, jacobian: np.ndarray, omega: np.ndarray,
    ) -> Tuple[float, float]:
        r"""ε = ‖Mᵀ Ω M − Ω‖_F ,  ε_rel = ε / ‖Ω‖_F ."""
        pulled = jacobian.T @ omega @ jacobian
        residual = pulled - omega
        abs_res = cls.frobenius(residual)
        rel_res = abs_res / max(cls.frobenius(omega), _MIN_SINGULAR_VALUE_FLOOR)
        return abs_res, rel_res

    @classmethod
    def polar_quillen_factor(
        cls, jacobian: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Factorización polar M = U P (Higham).

            P = √(MᵀM) ⪰ 0 ,  U = M P⁻¹ ∈ O(n) .
        """
        u_svd, svals, vt = la.svd(jacobian, full_matrices=False)
        cond = (float(svals[0] / svals[-1])
                if svals.size and svals[-1] > _MIN_SINGULAR_VALUE_FLOOR
                else float("inf"))
        p_spd = (vt.T * svals) @ vt
        u_orth = u_svd @ vt
        return u_orth, p_spd, cond

    @classmethod
    def compute_quillen_witness(
        cls, jacobian: np.ndarray, omega: np.ndarray,
    ) -> _QuillenWitness:
        abs_res, rel_res = cls.symplectic_pullback_residual(jacobian, omega)
        det_res = abs(float(np.linalg.det(jacobian)) - 1.0)
        u_orth, p_spd, cond = cls.polar_quillen_factor(jacobian)
        cofib = cls.frobenius(p_spd - np.eye(p_spd.shape[0]))
        fib_we, _ = cls.symplectic_pullback_residual(u_orth, omega)
        return _QuillenWitness(
            symplectic_residual=abs_res,
            symplectic_residual_rel=rel_res,
            det_residual=float(det_res),
            cofibration_residual=float(cofib),
            fibration_we_residual=float(fib_we),
            polar_condition=float(cond),
        )

    # ── I.1.3  Stasheff A∞ / Hochschild / Sullivan ──────────────────────
    @classmethod
    def compute_stasheff_norm(cls, m3: np.ndarray) -> float:
        return cls.frobenius(m3)

    @classmethod
    def hochschild_associator(cls, m2: np.ndarray) -> np.ndarray:
        r"""α(x,y,z) = (xy)z − x(yz) ;  m₂[a,b,c] = (e_b·e_c)_a ."""
        left = np.einsum("akd,kbc->abcd", m2, m2, optimize=True)
        right = np.einsum("abk,kcd->abcd", m2, m2, optimize=True)
        return left - right

    @classmethod
    def sullivan_commutator_residual(cls, m2: np.ndarray) -> float:
        return cls.frobenius(m2 - np.swapaxes(m2, 1, 2))

    @classmethod
    def stasheff_pentagon_residual(
        cls, m2: np.ndarray, m3: np.ndarray,
    ) -> float:
        r"""
        Pentágono K₄ (m₁ = m₄ = 0):

            −m₂(m₃⊗id) − m₂(id⊗m₃) + m₃(m₂⊗id⊗id)
            − m₃(id⊗m₂⊗id) + m₃(id⊗id⊗m₂) = 0 .
        """
        n = m2.shape[0]
        if m3.ndim != 4 or m3.shape != (n, n, n, n):
            raise ValueError("Pentágono exige m₃ de forma (n,n,n,n).")
        if n > _PENTAGON_EINSUM_DIM_CAP:
            acc = 0.0
            for last in range(n):
                term = (
                    -np.einsum("ake,kbcd->abcd", m2, m3[..., last], optimize=True)
                    - np.einsum("abk,kcde->abcd", m2, m3[:, :, :, last], optimize=True)
                    + np.einsum("akde,kbc->abcd", m3[..., last], m2, optimize=True)
                    - np.einsum("abke,kcd->abcd", m3[..., last], m2, optimize=True)
                    + np.einsum("abck,kde->abcd", m3, m2[:, :, last], optimize=True)
                )
                acc += float(np.square(term).sum())
            return float(np.sqrt(acc))
        term = (
            -np.einsum("ake,kbcde->abcde", m2, m3, optimize=True)
            - np.einsum("abk,kcde->abcde", m2, m3, optimize=True)
            + np.einsum("akde,kbc->abcde", m3, m2, optimize=True)
            - np.einsum("abke,kcd->abcde", m3, m2, optimize=True)
            + np.einsum("abck,kde->abcde", m3, m2, optimize=True)
        )
        return cls.frobenius(term)

    # ── I.1.4  Čech / gerbes ────────────────────────────────────────────
    @classmethod
    def cech_coboundary_1_norm2(cls, cochain: np.ndarray) -> float:
        r"""‖δC‖_F² contraída: 3n‖C‖²−2‖C1‖²−2‖Cᵀ1‖²+2·1ᵀC²1 ."""
        c = np.asarray(cochain, dtype=np.float64)
        n = c.shape[0]
        fro2 = float(np.square(c).sum())
        row = c @ np.ones(n)
        col = c.T @ np.ones(n)
        ones = np.ones(n)
        mixed = float(ones @ (c @ (c @ ones)))
        return (3.0 * n * fro2
                - 2.0 * float(row @ row)
                - 2.0 * float(col @ col)
                + 2.0 * mixed)

    @classmethod
    def cech_coboundary_2_norm(cls, cochain: np.ndarray) -> float:
        g = np.asarray(cochain, dtype=np.float64)
        n = g.shape[0]
        if n > _PENTAGON_EINSUM_DIM_CAP:
            acc = 0.0
            for i in range(n):
                delta = (g - g[i][None, :, :]
                         + np.expand_dims(g[i], axis=1)
                         - np.expand_dims(g[i], axis=2))
                acc += float(np.square(delta).sum())
            return float(np.sqrt(acc))
        delta = (g[np.newaxis, :, :, :]
                 - g[:, np.newaxis, :, :]
                 + g[:, :, np.newaxis, :]
                 - g[:, :, :, np.newaxis])
        return cls.frobenius(delta)

    @classmethod
    def compute_gerbe_obstruction(
        cls, cech_cochain: np.ndarray,
    ) -> Tuple[float, float, float]:
        arr = np.asarray(cech_cochain, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        floor = cls.wilkinson_deflation_floor(arr.reshape(arr.shape[0], -1))
        flat = arr.reshape(arr.shape[0], -1)
        svals = la.svd(flat, compute_uv=False)
        valid = svals[svals > floor]
        sv_mass = float(np.sum(valid)) if valid.size else 0.0
        scale = max(cls.frobenius(arr), _MIN_SINGULAR_VALUE_FLOOR)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            residual = float(np.sqrt(max(cls.cech_coboundary_1_norm2(arr), 0.0)))
        elif arr.ndim == 3 and arr.shape[0] == arr.shape[1] == arr.shape[2]:
            residual = cls.cech_coboundary_2_norm(arr)
        else:
            residual = sv_mass
        return residual, residual / scale, sv_mass

    # ── I.1.5  Sección transversal de Poincaré (dinámica) ───────────────
    @classmethod
    def build_poincare_section(
        cls,
        two_n: int,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> _PoincareSectionWitness:
        r"""
        Construye Σ = { q_{section_index} = q^* } ⊂ T*Q.

        Transversalidad *dinámica* de Poincaré: n_Σ · X_H ≠ 0.
        Si no hay X_H, se cae al sanity Ω n_Σ ≠ 0 (automático si n_Σ ≠ 0).
        """
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        if not (0 <= section_index < n):
            raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        if omega is None:
            omega = _SymplecticForm.from_dimension(two_n).matrix
        v = np.asarray(omega, dtype=np.float64) @ n_sigma
        chart_cert = cls.frobenius(v)
        chart_ok = bool(chart_cert > _MACHINE_EPS)
        flow_flux = 0.0
        dyn_ok = False
        if flow_vector is not None:
            xh = np.asarray(flow_vector, dtype=np.float64).ravel()
            if xh.size != two_n:
                raise ValueError(
                    f"flow_vector dim {xh.size} ≠ two_n={two_n}.")
            flow_flux = abs(float(np.dot(n_sigma, xh)))
            dyn_ok = bool(chart_ok and flow_flux > _MACHINE_EPS)
        return _PoincareSectionWitness(
            normal=n_sigma, section_index=int(section_index),
            section_offset=float(section_offset),
            energy_level=float(energy_level),
            transversal_certificate=float(chart_cert),
            is_transversal=chart_ok,
            flow_flux=float(flow_flux),
            is_dynamically_transversal=dyn_ok if flow_vector is not None else chart_ok,
        )

    # ── I.1.5.b  Clasificación de Williamson ────────────────────────────
    @classmethod
    def williamson_classify(
        cls, hessian: np.ndarray, omega: np.ndarray,
    ) -> _WilliamsonWitness:
        r"""Espectro de Williamson de H = ½ xᵀ A x, A = Hess H simétrica."""
        A = cls._as_real("hessian", hessian)
        O = np.asarray(omega, dtype=np.float64)
        if A.shape != O.shape:
            raise ValueError("hessian y Ω incompatibles.")
        A_sym = 0.5 * (A + A.T)
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
        )

    # ── I.1.5.c  Acción de Poincaré–Cartan ──────────────────────────────
    @classmethod
    def poincare_cartan_action(
        cls,
        q_traj: np.ndarray,
        p_traj: np.ndarray,
        dt: float,
        energy_level: float,
    ) -> _PoincareCartanWitness:
        r"""𝒜 = ∫ (p · dq − H dt) por la regla del punto medio."""
        q = cls._as_real("q_traj", q_traj)
        p = cls._as_real("p_traj", p_traj)
        if q.ndim != 2 or p.shape != q.shape or q.shape[0] < 2:
            raise ValueError("q_traj, p_traj deben ser (N≥2, n).")
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("dt debe ser finito y no nulo.")
        dq = np.diff(q, axis=0)
        p_mid = 0.5 * (p[1:] + p[:-1])
        pdq = np.einsum("ij,ij->i", p_mid, dq)
        action_pdq = float(pdq.sum())
        action = float(action_pdq - float(energy_level) * dt * dq.shape[0])
        close = cls.frobenius(np.concatenate((q[-1] - q[0], p[-1] - p[0])))
        return _PoincareCartanWitness(
            action=action,
            closure_residual=float(close),
            is_closed=bool(close <= _HARD_CARTAN_CLOSURE_TOL),
        )

    # ── I.1.6  Floquet–Lyapunov + Krein–Moser ───────────────────────────
    @classmethod
    def classify_krein_moser(
        cls,
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

    @classmethod
    def _real_matrix_logarithm(cls, a: np.ndarray) -> Tuple[np.ndarray, bool]:
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
        cls, monodromy_M: np.ndarray, orbit_period_T: float,
        omega: Optional[np.ndarray] = None,
    ) -> _FloquetLyapunovWitness:
        r"""
        M = exp(T · A_F) · R_F , con A_F = T⁻¹ Log(M).

        Un logaritmo *real* existe si M no tiene autovalores reales
        negativos de bloques de Jordan impares.
        """
        m = cls._as_real("monodromy_M", monodromy_M)
        if orbit_period_T <= 0.0 or not np.isfinite(orbit_period_T):
            raise ValueError("orbit_period_T debe ser positivo y finito.")
        mu, evec = la.eig(m, right=True)
        log_M, is_real_log = cls._real_matrix_logarithm(m)
        a_F = log_M / float(orbit_period_T)
        r_F = la.expm(-float(orbit_period_T) * a_F) @ m
        log_res = cls.frobenius(la.expm(float(orbit_period_T) * a_F) @ r_F - m)
        with np.errstate(divide="ignore", invalid="ignore"):
            char_c = np.log(mu.astype(np.complex128)) / float(orbit_period_T)
        char_exp = np.real(char_c)
        O = omega if omega is not None else _SymplecticForm.from_dimension(
            m.shape[0]).matrix
        krein = cls.classify_krein_moser(mu, omega=O, right_eigenvectors=evec)
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

    # ── I.1.7  Mel'nikov (ceros simples) ────────────────────────────────
    @classmethod
    def melnikov_function(
        cls,
        q0_trajectory: np.ndarray,
        dt: float,
        h0_grad: np.ndarray,
        h1_grad: np.ndarray,
        omega: np.ndarray,
        n_phase_offsets: int = _MELNIKOV_PHASE_SAMPLES,
    ) -> _MelnikovWitness:
        r"""
        ℳ(t₀) = ∫ {H₀,H₁} dt = ∫ (∇H₀)ᵀ Ω (∇H₁) dt.

        Cero *simple*: cambio de signo y |ℳ′| > piso de Wilkinson
        (ℳ′ por diferencia central sobre la muestra circular).
        """
        q0 = cls._as_real("q0_trajectory", q0_trajectory)
        g0 = cls._as_real("h0_grad", h0_grad)
        g1 = cls._as_real("h1_grad", h1_grad)
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
            if y.size < 2:
                return 0.0
            return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

        mel_vals = np.empty(n_phase_offsets, dtype=np.float64)
        for k in range(n_phase_offsets):
            shift = int(round(k * N / n_phase_offsets))
            mel_vals[k] = trapz(np.roll(pb, shift), dt)
        dphi = _TWO_PI / float(n_phase_offsets)
        dmel = (np.roll(mel_vals, -1) - np.roll(mel_vals, 1)) / (2.0 * dphi)
        floor = max(cls.wilkinson_deflation_floor(mel_vals), 1e-14)
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

    # ── I.1.8  Número de rotación ρ ∈ ℝ/ℤ ───────────────────────────────
    @classmethod
    def rotation_number_poincare(
        cls, orbit_points: np.ndarray, cf_depth: int = _ROTATION_CF_DEPTH,
    ) -> _RotationNumberWitness:
        r"""
        ρ = lim (1/(2π N)) Σ Δθ_i ∈ ℝ/ℤ, con Δθ_i el incremento angular
        desenrollado del par canónico (q₀, p₀). Racional si un convergente
        de Farey verifica |ρ − p/q| < 1/(2 q²).
        """
        pts = cls._as_real("orbit_points", orbit_points)
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
        mean_dtheta = float(dtheta.sum() / max(dtheta.size, 1))
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
            rotation_number=float(rho),
            is_rational=is_rational,
            continued_fraction=tuple(cf),
            diophantine_constant=dioph,
        )

    # ── I.1.8.b  Twist de Moser ─────────────────────────────────────────
    @classmethod
    def moser_twist(
        cls,
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

    # ── I.1.9  Certificado KAM diofantino + Nekhoroshev ─────────────────
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
        r"""
        Condición diofántica de Arnold: |k·ω| ≥ γ/|k|^τ, ∀ k ≠ 0,
        sobre un retículo adaptativo (cubo ∞ para n ≤ 2; ejes + cáscaras
        para n ≥ 3).
        """
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
            g_candidate = abs_kv * (norm ** _DIOPHANTINE_TAU_FLOOR)
            if g_candidate < gamma_est:
                gamma_est = g_candidate
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

    # ── I.1.10  Espectro de Lyapunov (Benettin–QR hamiltoniano) ─────────
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
    ) -> _LyapunovSpectrumWitness:
        r"""
        QR de Benettin–Galgani–Giorgilli–Strelcyn con emparejamiento
        hamiltoniano λᵢ ← ½(λᵢ − λ_{2n+1−i}) y traza nula (Liouville).
        Si se da T, se reportan exponentes por unidad de tiempo.
        """
        a = cls._as_real("M", M)
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
        if n % 2 == 0:
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
            spectrum=spectrum,
            kaplan_yorke_dimension=d_ky,
            kolmogorov_sinai_entropy=h_ks,
            is_chaotic=bool(spectrum.size > 0 and spectrum[0] > _HARD_LYAPUNOV_TOL),
        )

    # ── I.1.11  Gram–Schmidt simpléctico (Parasjuk–de Gosson) ───────────
    @classmethod
    def symplectic_gram_schmidt(
        cls, vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        r"""
        Construye S ∈ Sp(2n,ℝ) con pares conjugados (e_k, f_k):

            ω(e_i, e_j) = 0,  ω(f_i, f_j) = 0,  ω(e_i, f_j) = δ_{ij}.

        Si el par (v, w) degenera, se completa canónicamente w ← −Ω v.
        """
        B = np.asarray(vectors, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        dim = O.shape[0]
        if dim % 2 != 0:
            raise SymplecticDimensionError("SGS exige dimensión par (Sp(2n)).")
        if B.shape != (dim, dim):
            raise ValueError(f"vectors debe ser ({dim},{dim}); {B.shape}.")
        n = dim // 2
        S = np.zeros_like(B)

        def sprod(x: np.ndarray, y: np.ndarray) -> float:
            return cls.symplectic_pairing(x, y, O)

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
        res = cls.frobenius(S.T @ O @ S - O)
        if res > 1e-6:
            logger.warning("SGS: residuo simpléctico %.3e > umbral.", res)
        return S


# -----------------------------------------------------------------------------
# I.7 — 1-jet homotópico y 1-jet Poincaré
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _HomotopyJet:
    """1-jet homotópico inmutable (Fase I, base 3.0)."""

    n: int
    quillen: _QuillenWitness
    stasheff_norm: float
    stasheff_associator: float
    stasheff_pentagon: float
    sullivan_commutator: float
    gerbe_cech_residual: float
    gerbe_cech_residual_rel: float
    gerbe_sv_mass: float
    input_fault: Optional[str]


@dataclass(frozen=True, slots=True)
class _PoincareHomotopyJet:
    r"""
    **1-jet homotópico de Poincaré.**

    **Cierre formal de la Fase I / objeto inicial de la Fase II.**

    Envuelve el 1-jet base y añade la capa de mecánica celeste 4.1:
    sección transversal dinámica, Floquet–Lyapunov, Krein–Moser,
    Williamson, Mel'nikov, rotación, twist, KAM/Nekhoroshev, Lyapunov
    hamiltoniano y acción de Poincaré–Cartan.

    La Fase II audita este jet sin recomputar residuales; la Fase III
    decide en el retículo de Heyting sobre los veredictos de las aduanas.
    """

    base: _HomotopyJet
    section: _PoincareSectionWitness
    floquet: _FloquetLyapunovWitness
    melnikov: Optional[_MelnikovWitness]
    rotation: Optional[_RotationNumberWitness]
    kam: _KamTorusWitness
    lyapunov: _LyapunovSpectrumWitness
    lyapunov_max: float
    williamson: Optional[_WilliamsonWitness] = None
    cartan: Optional[_PoincareCartanWitness] = None
    moser_twist: Optional[_MoserTwistWitness] = None


# -----------------------------------------------------------------------------
# I.ω — MORFISMO TERMINAL DE LA FASE I
# -----------------------------------------------------------------------------
class _HomotopySpectralCoreBase(_HomotopySpectralCore):
    """
    Extensión 3.0: ensambla el 1-jet base (Quillen + Stasheff + Čech)
    reutilizado por el terminal de Fase I.
    """

    @classmethod
    def assemble_homotopy_jet(
        cls,
        dimension_n: int,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        omega: _SymplecticForm,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> _HomotopyJet:
        """Ensambla el 1-jet base 3.0 (Quillen + Stasheff + Čech)."""
        fault: Optional[str] = None
        zero_q = _QuillenWitness(
            symplectic_residual=float("inf"),
            symplectic_residual_rel=float("inf"),
            det_residual=float("inf"),
            cofibration_residual=float("inf"),
            fibration_we_residual=float("inf"),
            polar_condition=float("inf"),
        )
        try:
            if omega.matrix.shape != (dimension_n, dimension_n):
                raise ValueError("Ω incompatible con dimension_n.")
            jac = cls._as_real("jacobian_matrix", jacobian_matrix)
            if jac.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"Jacobiana {jac.shape}; esperada ({dimension_n},{dimension_n}).")
            quillen = cls.compute_quillen_witness(jac, omega.matrix)
        except ValueError as exc:
            fault = f"quillen:{exc}"
            quillen = zero_q
            logger.error("Fallo al ensamblar Quillen: %s", exc)
        try:
            m3 = cls._as_real("m3_homotopy_tensor", m3_homotopy_tensor)
            if m3.ndim not in (3, 4):
                raise ValueError(f"m₃ debe ser orden 3 o 4; ndim={m3.ndim}.")
            expected3 = (dimension_n,) * 3
            expected4 = (dimension_n,) * 4
            if m3.shape not in (expected3, expected4):
                raise ValueError(f"m₃ {m3.shape}; esperada {expected3} o {expected4}.")
            stasheff_norm = cls.compute_stasheff_norm(m3)
        except ValueError as exc:
            fault = (fault + "|" if fault else "") + f"stasheff:{exc}"
            m3 = np.zeros((dimension_n,) * 3, dtype=np.float64)
            stasheff_norm = float("inf")
            logger.error("Fallo al ensamblar m₃: %s", exc)
        associator_res = 0.0
        pentagon_res = 0.0
        sullivan_res = 0.0
        if m2_product_tensor is not None:
            try:
                m2 = cls._as_real("m2_product_tensor", m2_product_tensor)
                if m2.shape != (dimension_n,) * 3:
                    raise ValueError(f"m₂ {m2.shape}; esperada {(dimension_n,)*3}.")
                associator_res = cls.frobenius(cls.hochschild_associator(m2))
                sullivan_res = cls.sullivan_commutator_residual(m2)
                if m3.ndim == 4 and np.isfinite(stasheff_norm):
                    pentagon_res = cls.stasheff_pentagon_residual(m2, m3)
            except ValueError as exc:
                fault = (fault + "|" if fault else "") + f"ainfty:{exc}"
                associator_res = float("inf")
                pentagon_res = float("inf")
                sullivan_res = float("inf")
                logger.error("Fallo A∞/Sullivan: %s", exc)
        try:
            cech = (np.zeros((0, 0), dtype=np.float64)
                    if np.asarray(cech_cochain_matrix).size == 0
                    else cls._as_real("cech_cochain_matrix", cech_cochain_matrix))
            g_res, g_rel, g_mass = cls.compute_gerbe_obstruction(cech)
        except ValueError as exc:
            fault = (fault + "|" if fault else "") + f"gerbe:{exc}"
            g_res, g_rel, g_mass = float("inf"), float("inf"), float("inf")
            logger.error("Fallo Čech: %s", exc)
        return _HomotopyJet(
            n=dimension_n, quillen=quillen,
            stasheff_norm=float(stasheff_norm),
            stasheff_associator=float(associator_res),
            stasheff_pentagon=float(pentagon_res),
            sullivan_commutator=float(sullivan_res),
            gerbe_cech_residual=float(g_res),
            gerbe_cech_residual_rel=float(g_rel),
            gerbe_sv_mass=float(g_mass),
            input_fault=fault,
        )


class _HomotopySpectralCoreTerminal(_HomotopySpectralCoreBase):
    """
    Extensión terminal de `_HomotopySpectralCoreBase`: cierra la Fase I
    produciendo el jet de Poincaré que la Fase II consume. Prefiere el
    motor `ImperialTesserariosEngine` 4.1.0 como fuente numérica única
    y cae al núcleo local si el motor no puede inducir el gérmen.
    """

    @classmethod
    def assemble_poincare_homotopy_jet(
        cls,
        dimension_n: int,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        omega: _SymplecticForm,
        orbit_period_T: float = 1.0,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        m2_product_tensor: Optional[np.ndarray] = None,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
        hessian: Optional[np.ndarray] = None,
        q_traj_cartan: Optional[np.ndarray] = None,
        p_traj_cartan: Optional[np.ndarray] = None,
        engine: Optional[ImperialTesserariosEngine] = None,
    ) -> _PoincareHomotopyJet:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla *todas* las métricas homotópicas sobre los tensores de
        un ciclo OODA y las congela en un 1-jet inmutable de Poincaré
        que **es** el dominio de `HomotopicTesserariosAgent`.

            𝔍_I = (𝔍_base, Σ, F, 𝒦, 𝒲, ℳ, ρ, twist, KAM, Lyap, 𝒜) ,

        Este método **es** el arranque formal de
        `HomotopicTesserariosAgent.compile_poincare_tesserarios_sheaf`.
        """
        base_jet = cls.assemble_homotopy_jet(
            dimension_n=dimension_n,
            jacobian_matrix=jacobian_matrix,
            m3_homotopy_tensor=m3_homotopy_tensor,
            cech_cochain_matrix=cech_cochain_matrix,
            omega=omega,
            m2_product_tensor=m2_product_tensor,
        )
        section = cls.build_poincare_section(
            two_n=dimension_n,
            section_index=section_index,
            section_offset=section_offset,
            energy_level=energy_level,
            flow_vector=flow_vector,
            omega=omega.matrix,
        )
        jac = cls._as_real("jacobian_matrix", jacobian_matrix)
        eng = engine or ImperialTesserariosEngine()

        # Floquet / Mel'nikov / rotación: se induce el gérmen de Φ_II del motor.
        floquet: _FloquetLyapunovWitness
        melnikov: Optional[_MelnikovWitness] = None
        rotation: Optional[_RotationNumberWitness] = None
        try:
            fgerm = eng.induce_poincare_floquet_germ(
                jac, orbit_period_T,
                q0_trajectory=q0_trajectory, dt=dt_trajectory,
                h0_grad=h0_grad, h1_grad=h1_grad,
                orbit_points_for_rotation=orbit_points_for_rotation,
            )
            ff = fgerm.floquet
            krein_w: Optional[_KreinWitness] = None
            if getattr(ff, "krein", None) is not None:
                kr = ff.krein
                krein_w = _KreinWitness(
                    elliptic=int(kr.elliptic),
                    hyperbolic=int(kr.hyperbolic),
                    loxodromic=int(kr.loxodromic),
                    parabolic=int(kr.parabolic),
                    krein_definite=bool(kr.krein_definite),
                    on_unit_circle=int(kr.on_unit_circle),
                )
            floquet = _FloquetLyapunovWitness(
                generator=ff.floquet_generator,
                periodic_part=ff.periodic_part,
                log_residual=float(ff.log_residual),
                is_real_logarithm=bool(ff.is_real_logarithm),
                floquet_multipliers=ff.floquet_multipliers,
                characteristic_exponents=ff.characteristic_exponents,
                krein=krein_w,
                poincare_map_multipliers=getattr(
                    fgerm, "poincare_map_multipliers", None),
            )
            if fgerm.melnikov is not None:
                mv = fgerm.melnikov
                melnikov = _MelnikovWitness(
                    melnikov_values=mv.melnikov_values,
                    simple_zeros=int(mv.simple_zeros),
                    chaotic_indicator=float(mv.chaotic_indicator),
                    is_chaotic=bool(mv.is_chaotic),
                    melnikov_derivative_min=float(
                        getattr(mv, "melnikov_derivative_min", 0.0)),
                )
            if fgerm.rotation is not None:
                rv = fgerm.rotation
                rotation = _RotationNumberWitness(
                    rotation_number=float(rv.rotation_number),
                    is_rational=bool(rv.is_rational),
                    continued_fraction=tuple(rv.continued_fraction),
                    diophantine_constant=float(rv.diophantine_constant),
                )
        except (ValueError, SymplecticDimensionError) as exc:
            logger.warning("Motor Floquet falló (%s); núcleo local.", exc)
            floquet = cls.floquet_lyapunov_factorization(
                jac, orbit_period_T, omega=omega.matrix)
            if (q0_trajectory is not None
                    and h0_grad is not None and h1_grad is not None):
                try:
                    melnikov = cls.melnikov_function(
                        q0_trajectory, dt_trajectory, h0_grad, h1_grad,
                        omega.matrix)
                except ValueError as exc2:
                    logger.error("Mel'nikov local falló: %s", exc2)
            if orbit_points_for_rotation is not None:
                try:
                    rotation = cls.rotation_number_poincare(
                        orbit_points_for_rotation)
                except ValueError as exc2:
                    logger.error("Rotación local falló: %s", exc2)

        if frequency_vector is None:
            eig_A = la.eigvals(floquet.generator)
            omega_freq = np.sort(np.abs(np.imag(eig_A)))[::-1][:dimension_n // 2]
            if omega_freq.size == 0 or np.all(omega_freq < 1e-12):
                omega_freq = np.ones(max(dimension_n // 2, 1), dtype=np.float64)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
        try:
            kam_eng = eng.certify_kam_torus(
                omega_freq, birkhoff_residual=birkhoff_residual)
            kam = _KamTorusWitness(
                frequency_vector=np.asarray(kam_eng.frequency_vector),
                diophantine_gamma=float(kam_eng.diophantine_gamma),
                diophantine_tau=float(kam_eng.diophantine_tau),
                birkhoff_residual=float(kam_eng.birkhoff_normal_residual),
                kam_stable=bool(kam_eng.kam_stable),
                iterations=int(kam_eng.kam_iterations),
                nekhoroshev_time=float(getattr(kam_eng, "nekhoroshev_time", 0.0)),
                worst_small_divisor=float(
                    getattr(kam_eng, "worst_small_divisor", 0.0)),
            )
        except (ValueError, AttributeError) as exc:
            logger.warning("Motor KAM falló (%s); núcleo local.", exc)
            kam = cls.certify_kam_torus(
                omega_freq, birkhoff_residual=birkhoff_residual)

        try:
            lyap_eng = eng.compute_lyapunov_spectrum(
                jac, orbit_period_T=orbit_period_T)
            lyapunov = _LyapunovSpectrumWitness(
                spectrum=np.asarray(lyap_eng.spectrum),
                kaplan_yorke_dimension=float(lyap_eng.kaplan_yorke_dimension),
                kolmogorov_sinai_entropy=float(lyap_eng.kolmogorov_sinai_entropy),
                is_chaotic=bool(lyap_eng.is_chaotic),
            )
        except (ValueError, AttributeError) as exc:
            logger.warning("Motor Lyapunov falló (%s); núcleo local.", exc)
            lyapunov = cls.lyapunov_spectrum_benettin(
                jac, orbit_period_T=orbit_period_T)
        lyap_max = float(lyapunov.spectrum[0]) if lyapunov.spectrum.size else 0.0

        williamson: Optional[_WilliamsonWitness] = None
        if hessian is not None:
            try:
                w_eng = eng.williamson_classify(hessian, omega.matrix)
                williamson = _WilliamsonWitness(
                    elliptic_pairs=int(w_eng.elliptic_pairs),
                    hyperbolic_pairs=int(w_eng.hyperbolic_pairs),
                    focus_focus_pairs=int(w_eng.focus_focus_pairs),
                    parabolic_multiplicity=int(w_eng.parabolic_multiplicity),
                    is_linearly_stable=bool(w_eng.is_linearly_stable),
                )
            except (ValueError, AttributeError, SymplecticDimensionError) as exc:
                logger.warning("Williamson motor falló (%s); núcleo local.", exc)
                try:
                    williamson = cls.williamson_classify(hessian, omega.matrix)
                except ValueError as exc2:
                    logger.error("Williamson local falló: %s", exc2)

        cartan: Optional[_PoincareCartanWitness] = None
        if q_traj_cartan is not None and p_traj_cartan is not None:
            try:
                action, close = eng.poincare_cartan_action(
                    q_traj_cartan, p_traj_cartan, dt_trajectory,
                    energy_level=energy_level)
                cartan = _PoincareCartanWitness(
                    action=float(action),
                    closure_residual=float(close),
                    is_closed=bool(close <= _HARD_CARTAN_CLOSURE_TOL),
                )
            except (ValueError, AttributeError) as exc:
                logger.warning("Cartan motor falló (%s); núcleo local.", exc)
                try:
                    cartan = cls.poincare_cartan_action(
                        q_traj_cartan, p_traj_cartan, dt_trajectory, energy_level)
                except ValueError as exc2:
                    logger.error("Cartan local falló: %s", exc2)

        return _PoincareHomotopyJet(
            base=base_jet,
            section=section,
            floquet=floquet,
            melnikov=melnikov,
            rotation=rotation,
            kam=kam,
            lyapunov=lyapunov,
            lyapunov_max=lyap_max,
            williamson=williamson,
            cartan=cartan,
            moser_twist=None,
        )


# #############################################################################
#                                                                             #
#  FASE II                                                                    #
#  TESSERARIOS · ADUANAS DE QUILLEN / STASHEFF / GERBE / POINCARÉ             #
#                                                                             #
#  Continuación directa del último morfismo de la Fase I:                     #
#      _PoincareHomotopyJet  ↦  HomotopicTesserariosAgent                     #
#                                                                             #
#  Cierre formal: compile_poincare_tesserarios_sheaf → _PoincareTesserariosSheaf
#                 (dominio de la Cámara de Coherencia, Fase III).             #
#                                                                             #
# #############################################################################
# PUENTE Φ_I ▸ Φ_II
# El valor de retorno de I.ω (`_PoincareHomotopyJet`) es el único
# argumento geométrico de `compile_poincare_tesserarios_sheaf` (II.ω).
# =============================================================================

@dataclass(frozen=True, slots=True)
class _TesserariosAuditResult:
    """Resultado de una aduana individual: métrica, tolerancia y veredicto."""

    metric_value: float
    tolerance: float
    verdict: str
    ancilla: Dict[str, float]


@dataclass(frozen=True, slots=True)
class _TesserariosSheaf:
    """Gavilla 3.0: auditorías Quillen + Stasheff + Gerbe (compatibilidad)."""

    jet: _HomotopyJet
    quillen: _TesserariosAuditResult
    stasheff: _TesserariosAuditResult
    gerbe: _TesserariosAuditResult


@dataclass(frozen=True, slots=True)
class _PoincareTesserariosSheaf:
    r"""
    **Gavilla de auditorías de Poincaré.**

    Cierre formal de la Fase II / objeto inicial de la Fase III.

    Porta el 1-jet completo (`pjet`) para que la Fase III no reconstruya
    testigos dummy desde ancilla. Extiende la gavilla 3.0 con:

      * `poincare`  : sección dinámica + Floquet + Lyapunov + KAM + Krein,
      * `melnikov`  : detección de caos homoclínico (opcional),
      * `krein`     : estabilidad fuerte de Krein–Moser (opcional),
      * `twist`     : condición de Moser (opcional).
    """

    pjet: _PoincareHomotopyJet
    base_sheaf: _TesserariosSheaf
    poincare: _TesserariosAuditResult
    melnikov: Optional[_TesserariosAuditResult]
    krein: Optional[_TesserariosAuditResult] = None
    twist: Optional[_TesserariosAuditResult] = None


class HomotopicTesserariosAgent:
    r"""
    Tesserarios de Integridad Homotópica (Capa 3, Poincaré–Extendida 4.1).

    Consume el 1-jet de Poincaré de la Fase I (I.ω) y evalúa
    contractibilidad (Quillen), coherencia de asociaedros (Stasheff),
    trivialidad de gerbes (Čech), **y** coherencia dinámica de Poincaré
    (sección dinámica, Floquet, Krein, Lyapunov, KAM/Nekhoroshev,
    Mel'nikov, twist de Moser).
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        omega: Optional[_SymplecticForm] = None,
    ) -> None:
        if dimension_n <= 0 or dimension_n % 2 != 0:
            raise ValueError(f"Dimensión simpléctica n={dimension_n} debe ser par.")
        if safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser estrictamente positivo.")
        if omega is None:
            omega = _SymplecticForm.from_dimension(dimension_n)
        elif omega.matrix.shape != (dimension_n, dimension_n):
            raise ValueError("Ω inyectada incompatible con dimension_n.")
        omega.verify()
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._omega: Final[_SymplecticForm] = omega
        self._canonical_omega: Final[np.ndarray] = omega.matrix
        self._engine: Final[ImperialTesserariosEngine] = ImperialTesserariosEngine()

    # ── II.0  Utilidades ─────────────────────────────────────────────────
    def _verdict_from_metric(
        self,
        metric: float,
        base_tolerance: float,
        degradation_factor: float = _DEGRADATION_FACTOR,
    ) -> Tuple[str, float]:
        tol = float(base_tolerance) * self._safety_margin
        if (not np.isfinite(metric)) or metric > tol:
            return "VETOED", tol
        if metric > tol * degradation_factor:
            return "DEGRADED", tol
        return "COHERENT", tol

    @staticmethod
    def _heyting_meet_tokens(*tokens: str) -> str:
        if not tokens:
            return HeytingVerdict.COHERENT.name
        acc = HeytingVerdict.from_token(tokens[0])
        for t in tokens[1:]:
            acc = acc.meet(HeytingVerdict.from_token(t))
        return acc.name

    # ── II.1  Aduana de Quillen ──────────────────────────────────────────
    def audit_quillen_factorization(self, jet: _HomotopyJet) -> _TesserariosAuditResult:
        if jet.n != self._n:
            logger.error("Jet de dimensión %d, agente n=%d.", jet.n, self._n)
            return _TesserariosAuditResult(
                metric_value=float("inf"),
                tolerance=_HARD_QUILLEN_TOLERANCE * self._safety_margin,
                verdict="VETOED", ancilla={})
        metric = jet.quillen.symplectic_residual
        verdict, tol = self._verdict_from_metric(metric, _HARD_QUILLEN_TOLERANCE)
        if jet.quillen.det_residual > max(tol * 10.0, 1.0e-6):
            verdict = "VETOED"
        return _TesserariosAuditResult(
            metric_value=metric, tolerance=tol, verdict=verdict,
            ancilla={
                "symplectic_residual_rel": jet.quillen.symplectic_residual_rel,
                "det_residual": jet.quillen.det_residual,
                "cofibration_residual": jet.quillen.cofibration_residual,
                "fibration_we_residual": jet.quillen.fibration_we_residual,
                "polar_condition": jet.quillen.polar_condition,
            },
        )

    # ── II.2  Aduana de Stasheff ─────────────────────────────────────────
    def audit_stasheff_coherence_relation(self, jet: _HomotopyJet) -> _TesserariosAuditResult:
        metric = jet.stasheff_norm
        verdict, tol = self._verdict_from_metric(metric, _HARD_STASHEFF_CEILING)
        for extra, name in (
            (jet.stasheff_associator, "associator"),
            (jet.stasheff_pentagon, "pentagon"),
            (jet.sullivan_commutator, "sullivan"),
        ):
            extra_verdict, _ = self._verdict_from_metric(
                extra,
                _HARD_STASHEFF_CEILING if name != "sullivan"
                else _HARD_SULLIVAN_TOLERANCE)
            if HeytingVerdict.from_token(extra_verdict).value \
                    < HeytingVerdict.from_token(verdict).value:
                verdict = extra_verdict
                logger.debug("Stasheff endurecido por %s → %s", name, verdict)
        return _TesserariosAuditResult(
            metric_value=metric, tolerance=tol, verdict=verdict,
            ancilla={
                "associator": jet.stasheff_associator,
                "pentagon": jet.stasheff_pentagon,
                "sullivan_commutator": jet.sullivan_commutator,
            },
        )

    # ── II.3  Aduana de Čech ─────────────────────────────────────────────
    def audit_non_abelian_gerbe_obstruction(self, jet: _HomotopyJet) -> _TesserariosAuditResult:
        metric = jet.gerbe_cech_residual
        verdict, tol = self._verdict_from_metric(metric, _HARD_GERBE_TOLERANCE)
        return _TesserariosAuditResult(
            metric_value=metric, tolerance=tol, verdict=verdict,
            ancilla={
                "cech_residual_rel": jet.gerbe_cech_residual_rel,
                "sv_mass": jet.gerbe_sv_mass,
            },
        )

    # ── II.4  ADUANA DE POINCARÉ (dinámica + Krein + Nekhoroshev) ────────
    def audit_poincare_coherence(
        self, pjet: _PoincareHomotopyJet,
    ) -> _TesserariosAuditResult:
        r"""
        **[TESSERARIO 4 — COHERENCIA DINÁMICA DE POINCARÉ–FLOQUET]**

        Audita la coherencia dinámica del jet combinando invariantes
        en un único veredicto Heyting:

          1. Transversalidad *dinámica* de Σ: n_Σ · X_H ≠ 0 (o sanity
             de chart si no hay flujo).
          2. Estabilidad de Floquet: max_k |μ_k| ≤ 1 + δ.
          3. Exponente de Lyapunov máximo: λ_max ≤ δ.
          4. Certificado KAM diofantino: γ ≥ γ_floor.
          5. Residuo del logaritmo de Floquet: ‖exp(T A_F) R_F − M‖_F ≤ δ.
          6. Estabilidad lineal de Williamson (si hay hessiana): no
             hiperbólica / foco–foco.
          7. Cierre de Poincaré–Cartan (si hay arco): ‖Δ(q,p)‖ ≤ δ.

        El veredicto final es el ínfimo de Heyting; cualquier fallo
        endurece la decisión.
        """
        sec = pjet.section
        dyn = sec.is_dynamically_transversal
        chart = sec.is_transversal and sec.transversal_certificate > _HARD_POINCARE_SECTION_TOL
        v1 = "COHERENT" if (dyn or chart) else "VETOED"

        mu = pjet.floquet.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
        excess = max(0.0, max_mu - 1.0)
        v2, _ = self._verdict_from_metric(excess, _HARD_FLOQUET_MULTIPLIER_TOL)

        v3, _ = self._verdict_from_metric(
            max(0.0, pjet.lyapunov_max), _HARD_LYAPUNOV_TOL)

        v4 = "COHERENT" if pjet.kam.kam_stable and \
            pjet.kam.diophantine_gamma > _HARD_KAM_GAMMA_FLOOR else "VETOED"

        v5, _ = self._verdict_from_metric(
            pjet.floquet.log_residual, _HARD_FLOQUET_MULTIPLIER_TOL)

        tokens = [v1, v2, v3, v4, v5]
        if pjet.williamson is not None and not pjet.williamson.is_linearly_stable:
            tokens.append("DEGRADED" if pjet.williamson.hyperbolic_pairs == 0
                          else "VETOED")
        if pjet.cartan is not None and not pjet.cartan.is_closed:
            tokens.append("DEGRADED")

        final = self._heyting_meet_tokens(*tokens)
        metric_value = float(max(
            excess, max(0.0, pjet.lyapunov_max), pjet.floquet.log_residual))
        tol = _HARD_FLOQUET_MULTIPLIER_TOL * self._safety_margin
        krein_def = bool(
            pjet.floquet.krein.krein_definite
            if pjet.floquet.krein is not None else False)
        ancilla: Dict[str, float] = {
            "section_transversal": float(sec.transversal_certificate),
            "section_flow_flux": float(sec.flow_flux),
            "floquet_max_multiplier": max_mu,
            "floquet_log_residual": pjet.floquet.log_residual,
            "lyapunov_max": pjet.lyapunov_max,
            "kam_gamma": pjet.kam.diophantine_gamma,
            "kam_tau": pjet.kam.diophantine_tau,
            "nekhoroshev_time": float(pjet.kam.nekhoroshev_time),
            "kaplan_yorke_dimension": pjet.lyapunov.kaplan_yorke_dimension,
            "kolmogorov_sinai_entropy": pjet.lyapunov.kolmogorov_sinai_entropy,
            "krein_definite": float(krein_def),
        }
        if pjet.williamson is not None:
            ancilla["williamson_elliptic"] = float(pjet.williamson.elliptic_pairs)
            ancilla["williamson_hyperbolic"] = float(pjet.williamson.hyperbolic_pairs)
        if pjet.cartan is not None:
            ancilla["cartan_action"] = float(pjet.cartan.action)
            ancilla["cartan_closure"] = float(pjet.cartan.closure_residual)
        return _TesserariosAuditResult(
            metric_value=metric_value, tolerance=tol, verdict=final,
            ancilla=ancilla,
        )

    # ── II.5  ADUANA DE MEL'NIKOV ────────────────────────────────────────
    def audit_melnikov_chaos(
        self, pjet: _PoincareHomotopyJet,
    ) -> Optional[_TesserariosAuditResult]:
        r"""
        **[TESSERARIO 5 — DETECCIÓN DE CAOS HOMOCLÍNICO]**

        Cualquier cero *simple* de ℳ ⇒ intersección transversal
        W^s ∩ W^u ⇒ **VETOED**. Sin órbita, la aduana no se emite.
        """
        if pjet.melnikov is None:
            return None
        mw = pjet.melnikov
        if mw.is_chaotic:
            return _TesserariosAuditResult(
                metric_value=float(mw.simple_zeros),
                tolerance=0.0, verdict="VETOED",
                ancilla={
                    "simple_zeros": float(mw.simple_zeros),
                    "chaotic_indicator": mw.chaotic_indicator,
                    "melnikov_derivative_min": float(mw.melnikov_derivative_min),
                },
            )
        return _TesserariosAuditResult(
            metric_value=0.0, tolerance=0.0, verdict="COHERENT",
            ancilla={
                "simple_zeros": 0.0,
                "chaotic_indicator": mw.chaotic_indicator,
                "melnikov_derivative_min": float(mw.melnikov_derivative_min),
            },
        )

    # ── II.6  ADUANA DE KREIN–MOSER ──────────────────────────────────────
    def audit_krein_moser(
        self, pjet: _PoincareHomotopyJet,
    ) -> Optional[_TesserariosAuditResult]:
        r"""
        **[TESSERARIO 6 — ESTABILIDAD FUERTE DE KREIN]**

        Un toro elíptico es Krein-definido si todos los elípticos son
        simples y de signatura definida. Loxodrómicos o hiperbólicos
        ⇒ VETOED; elípticos no definidos ⇒ DEGRADED.
        """
        kr = pjet.floquet.krein
        if kr is None:
            return None
        if kr.loxodromic > 0 or kr.hyperbolic > 0:
            verdict = "VETOED"
        elif kr.krein_definite:
            verdict = "COHERENT"
        else:
            verdict = "DEGRADED"
        return _TesserariosAuditResult(
            metric_value=float(kr.hyperbolic + kr.loxodromic),
            tolerance=0.0, verdict=verdict,
            ancilla={
                "krein_elliptic": float(kr.elliptic),
                "krein_hyperbolic": float(kr.hyperbolic),
                "krein_loxodromic": float(kr.loxodromic),
                "krein_parabolic": float(kr.parabolic),
                "krein_definite": float(kr.krein_definite),
                "on_unit_circle": float(kr.on_unit_circle),
            },
        )

    # ── II.7  ADUANA DE TWIST DE MOSER ───────────────────────────────────
    def audit_moser_twist(
        self, pjet: _PoincareHomotopyJet,
    ) -> Optional[_TesserariosAuditResult]:
        r"""
        **[TESSERARIO 7 — TWIST DE MOSER / POINCARÉ–BIRKHOFF]**

        Si el jet porta un certificado de twist, se exige |∂ρ/∂I| ≥ ν.
        Sin muestra de toros, la aduana no se emite.
        """
        tw = pjet.moser_twist
        if tw is None:
            return None
        verdict = "COHERENT" if tw.is_twist else "DEGRADED"
        return _TesserariosAuditResult(
            metric_value=float(abs(tw.twist_value)),
            tolerance=_HARD_TWIST_FLOOR * self._safety_margin,
            verdict=verdict,
            ancilla={
                "twist_value": float(tw.twist_value),
                "intersection_property": float(tw.intersection_property),
            },
        )

    # ── II.ω  MORFISMO TERMINAL DE LA FASE II ────────────────────────────
    def compile_poincare_tesserarios_sheaf(
        self, pjet: _PoincareHomotopyJet,
    ) -> _PoincareTesserariosSheaf:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Pega las aduanas sobre el 1-jet de Poincaré y **porta el jet**
        para que la Cámara no reconstruya testigos. La gavilla
        resultante es el dominio de `TesserariosCoherenceChamber`
        (`fuse_and_actuate_poincare`).
        """
        if pjet.base.input_fault:
            logger.error("Jet base con fallos: %s", pjet.base.input_fault)
        if pjet.base.n != self._n:
            logger.error(
                "Jet Poincaré de dimensión %d, agente n=%d.", pjet.base.n, self._n)
        base_sheaf = _TesserariosSheaf(
            jet=pjet.base,
            quillen=self.audit_quillen_factorization(pjet.base),
            stasheff=self.audit_stasheff_coherence_relation(pjet.base),
            gerbe=self.audit_non_abelian_gerbe_obstruction(pjet.base),
        )
        return _PoincareTesserariosSheaf(
            pjet=pjet,
            base_sheaf=base_sheaf,
            poincare=self.audit_poincare_coherence(pjet),
            melnikov=self.audit_melnikov_chaos(pjet),
            krein=self.audit_krein_moser(pjet),
            twist=self.audit_moser_twist(pjet),
        )

    # ── Compatibilidad 3.0: métodos heredados ────────────────────────────
    def ingest_tensors(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> _HomotopyJet:
        """Compatibilidad 3.0: ensambla el jet base (sin Poincaré)."""
        return _HomotopySpectralCoreBase.assemble_homotopy_jet(
            dimension_n=self._n,
            jacobian_matrix=jacobian_matrix,
            m3_homotopy_tensor=m3_homotopy_tensor,
            cech_cochain_matrix=cech_cochain_matrix,
            omega=self._omega,
            m2_product_tensor=m2_product_tensor,
        )

    def ingest_poincare_tensors(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        orbit_period_T: float = 1.0,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        m2_product_tensor: Optional[np.ndarray] = None,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
        hessian: Optional[np.ndarray] = None,
        q_traj_cartan: Optional[np.ndarray] = None,
        p_traj_cartan: Optional[np.ndarray] = None,
    ) -> _PoincareHomotopyJet:
        """Ensambla el 1-jet de Poincaré (I.ω) desde el agente."""
        return _HomotopySpectralCoreTerminal.assemble_poincare_homotopy_jet(
            dimension_n=self._n,
            jacobian_matrix=jacobian_matrix,
            m3_homotopy_tensor=m3_homotopy_tensor,
            cech_cochain_matrix=cech_cochain_matrix,
            omega=self._omega,
            orbit_period_T=orbit_period_T,
            section_index=section_index,
            section_offset=section_offset,
            energy_level=energy_level,
            m2_product_tensor=m2_product_tensor,
            q0_trajectory=q0_trajectory,
            dt_trajectory=dt_trajectory,
            h0_grad=h0_grad,
            h1_grad=h1_grad,
            orbit_points_for_rotation=orbit_points_for_rotation,
            frequency_vector=frequency_vector,
            birkhoff_residual=birkhoff_residual,
            flow_vector=flow_vector,
            hessian=hessian,
            q_traj_cartan=q_traj_cartan,
            p_traj_cartan=p_traj_cartan,
            engine=self._engine,
        )

    def compile_tesserarios_sheaf(self, jet: _HomotopyJet) -> _TesserariosSheaf:
        """Compatibilidad 3.0: gavilla sin Poincaré."""
        if jet.input_fault:
            logger.error("Jet ensamblado con fallos: %s", jet.input_fault)
        return _TesserariosSheaf(
            jet=jet,
            quillen=self.audit_quillen_factorization(jet),
            stasheff=self.audit_stasheff_coherence_relation(jet),
            gerbe=self.audit_non_abelian_gerbe_obstruction(jet),
        )

    def attach_moser_twist(
        self,
        pjet: _PoincareHomotopyJet,
        rotation_samples: Sequence[_RotationNumberWitness],
        action_samples: Optional[np.ndarray] = None,
    ) -> _PoincareHomotopyJet:
        """Adjunta un certificado de twist de Moser al 1-jet (inmutable)."""
        twist = _HomotopySpectralCore.moser_twist(
            rotation_samples, action_samples=action_samples)
        return _PoincareHomotopyJet(
            base=pjet.base, section=pjet.section, floquet=pjet.floquet,
            melnikov=pjet.melnikov, rotation=pjet.rotation, kam=pjet.kam,
            lyapunov=pjet.lyapunov, lyapunov_max=pjet.lyapunov_max,
            williamson=pjet.williamson, cartan=pjet.cartan,
            moser_twist=twist,
        )

    # ── Compatibilidad: crowbar / Poincaré ───────────────────────────────
    def _trigger_hardware_iram_crowbar_isr(self) -> float:
        crowbar = _ThyristorCrowbar()
        rng = np.random.default_rng()
        return crowbar.fire(rng)

    def audit_tesserario_poincare_homotopy_closed_loop(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        orbit_period_T: float = 1.0,
    ) -> Dict[str, Any]:
        r"""
        Compatibilidad: auditoría OODA de lazo cerrado (motor Poincaré
        4.1.0 + Heyting G₃ + Crowbar). Se conserva la firma 3.0.
        """
        monodromy_germ = self._engine.compute_poincare_symplectic_monodromy_germ(
            jacobian_M=jacobian_matrix,
            orbit_period_T=orbit_period_T,
            canonical_omega=self._canonical_omega,
        )
        m3_norm = float(la.norm(m3_homotopy_tensor, ord="fro")) \
            if m3_homotopy_tensor.size > 0 else 0.0
        jac_skew = float(la.norm(jacobian_matrix - jacobian_matrix.T, ord="fro")) \
            if jacobian_matrix.size > 0 else 0.0
        stasheff_residual = m3_norm + 1e-4 * jac_skew
        is_coherent = (monodromy_germ.is_monodromy_stable
                       and stasheff_residual <= _WILKINSON_LIMIT)
        is_degraded = (monodromy_germ.max_floquet_multiplier <= 1.05) \
            and not is_coherent
        if is_coherent:
            verdict = HeytingVerdict.COHERENT
            crowbar_triggered = False
        elif is_degraded:
            verdict = HeytingVerdict.DEGRADED
            crowbar_triggered = False
        else:
            verdict = HeytingVerdict.VETOED
            crowbar_triggered = True
        actuation_latency_ns = 0.0
        if crowbar_triggered:
            actuation_latency_ns = self._trigger_hardware_iram_crowbar_isr()
        return {
            "heyting_verdict": verdict.name,
            "symplectic_residual": monodromy_germ.relative_symplectic_residual,
            "volume_drift": monodromy_germ.volume_drift,
            "max_floquet_multiplier": monodromy_germ.max_floquet_multiplier,
            "lyapunov_exponent": monodromy_germ.lyapunov_exponent,
            "stasheff_residual": stasheff_residual,
            "crowbar_triggered": crowbar_triggered,
            "actuation_latency_ns": actuation_latency_ns,
        }


# #############################################################################
#                                                                             #
#  FASE III                                                                   #
#  CÁMARA DE COHERENCIA · HEYTING G₃ · OODA · CROWBAR BT151 · POINCARÉ        #
#                                                                             #
#  Continuación directa del último morfismo de la Fase II:                    #
#      _PoincareTesserariosSheaf  ↦  TesserariosCoherenceChamber              #
#                                                                             #
# #############################################################################
# PUENTE Φ_II ▸ Φ_III
# El valor de retorno de II.ω (`_PoincareTesserariosSheaf`) es el
# dominio de `fuse_and_actuate_poincare` (III.ω).
# =============================================================================

@dataclass
class _ThyristorCrowbar:
    r"""
    Tiristor BT151 como bypass de silicio. Modelo lumped:
    latencia de puerta ~400 ns con jitter gaussiano; latchea hasta reset.
    """

    latched: bool = False
    last_latency_ns: float = 0.0

    def fire(self, rng: np.random.Generator) -> float:
        jitter = float(rng.normal(0.0, _CROWBAR_JITTER_NS))
        latency = float(np.clip(
            _CROWBAR_IRAM_LATENCY_NS + jitter,
            _CROWBAR_T_MIN_NS, _CROWBAR_T_MAX_NS))
        self.latched = True
        self.last_latency_ns = latency
        return latency

    def reset(self) -> None:
        self.latched = False
        self.last_latency_ns = 0.0


@dataclass(frozen=True, slots=True)
class PoincareMonodromyCertificate:
    r"""
    **Certificado global de monodromía de Poincaré–Floquet (capa agéntica).**

    **Morfismo terminal global** Φ_III ∘ Φ_II ∘ Φ_I. Integra los objetos
    terminales de las tres fases en un único certificado inmutable,
    portando los testigos *reales* del 1-jet (no reconstrucciones dummy)
    y, opcionalmente, el certificado del motor 4.1.0.
    """

    darboux_dim: int
    section: _PoincareSectionWitness
    floquet: _FloquetLyapunovWitness
    melnikov: Optional[_MelnikovWitness]
    rotation: Optional[_RotationNumberWitness]
    kam: _KamTorusWitness
    lyapunov: _LyapunovSpectrumWitness
    williamson: Optional[_WilliamsonWitness]
    cartan: Optional[_PoincareCartanWitness]
    moser_twist: Optional[_MoserTwistWitness]
    quillen_verdict: str
    stasheff_verdict: str
    gerbe_verdict: str
    poincare_verdict: str
    melnikov_verdict: Optional[str]
    krein_verdict: Optional[str]
    twist_verdict: Optional[str]
    heyting_verdict: str
    heyting_value: int
    crowbar_fired: bool
    actuation_latency_ns: float
    relative_symplectic_residual: float
    volume_drift: float
    max_floquet_multiplier: float
    lyapunov_exponent: float
    is_monodromy_stable: bool
    krein_definite: bool = False
    nekhoroshev_time: float = 0.0
    engine_certificate: Optional[EnginePoincareMonodromyCertificate] = None


class TesserariosCoherenceChamber:
    """
    Cámara de Coherencia de la Capa 3 (Poincaré–Extendida 4.1).

    Consume la `_PoincareTesserariosSheaf` (cierre de Fase II), calcula
    el ínfimo de Heyting de las aduanas (Quillen, Stasheff, Gerbe,
    Poincaré, Mel'nikov, Krein, twist) y dispara el crowbar si el
    ínfimo es ⊥.
    """

    def __init__(
        self,
        agent: HomotopicTesserariosAgent,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.agent = agent
        self._crowbar = _ThyristorCrowbar()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng())

    @classmethod
    def assemble_from_spectral_seed(
        cls,
        dimension_n: int,
        safety_margin: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "TesserariosCoherenceChamber":
        omega = _SymplecticForm.from_dimension(dimension_n)
        omega.verify()
        agent = HomotopicTesserariosAgent(
            dimension_n=dimension_n,
            safety_margin=safety_margin,
            omega=omega)
        return cls(agent, rng=rng)

    # ── III.1  Ínfimo Heyting ────────────────────────────────────────────
    @staticmethod
    def _heyting_meet(*verdicts: str) -> HeytingVerdict:
        if not verdicts:
            return HeytingVerdict.COHERENT
        acc = HeytingVerdict.from_token(verdicts[0])
        for v in verdicts[1:]:
            acc = acc.meet(HeytingVerdict.from_token(v))
        return acc

    # ── III.2  Fusión clásica (3 aduanas) ────────────────────────────────
    def fuse_and_actuate(self, sheaf: _TesserariosSheaf) -> Dict[str, Any]:
        """Ciclo OODA 3.0 (Quillen + Stasheff + Gerbe)."""
        final_heyting = self._heyting_meet(
            sheaf.quillen.verdict,
            sheaf.stasheff.verdict,
            sheaf.gerbe.verdict,
        )
        final_verdict = final_heyting.name
        interlock_fired = False
        latency_ns = 0.0
        if final_heyting is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            interlock_fired = True
            logger.critical(
                "¡VETO ATÓMICO TESSERARIOS (3 aduanas)! "
                "Crowbar BT151 [GPIO14] gatillado en IRAM. "
                "Latencia %.2f ns. ε_Q=%.3e  ‖m₃‖=%.3e  ‖δα‖=%.3e",
                latency_ns, sheaf.quillen.metric_value,
                sheaf.stasheff.metric_value, sheaf.gerbe.metric_value)
        return {
            "heyting_verdict": final_verdict,
            "heyting_value": final_heyting.value,
            "quillen_residual": sheaf.quillen.metric_value,
            "quillen_tolerance": sheaf.quillen.tolerance,
            "quillen_verdict": sheaf.quillen.verdict,
            "quillen_ancilla": dict(sheaf.quillen.ancilla),
            "stasheff_norm": sheaf.stasheff.metric_value,
            "stasheff_tolerance": sheaf.stasheff.tolerance,
            "stasheff_verdict": sheaf.stasheff.verdict,
            "stasheff_ancilla": dict(sheaf.stasheff.ancilla),
            "gerbe_obstruction": sheaf.gerbe.metric_value,
            "gerbe_tolerance": sheaf.gerbe.tolerance,
            "gerbe_verdict": sheaf.gerbe.verdict,
            "gerbe_ancilla": dict(sheaf.gerbe.ancilla),
            "input_fault": sheaf.jet.input_fault,
            "hardware_interlock_fired": interlock_fired,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
        }

    # ── III.ω  MORFISMO TERMINAL DE LA FASE III ─────────────────────────
    def fuse_and_actuate_poincare(
        self,
        psheaf: _PoincareTesserariosSheaf,
        engine_certificate: Optional[EnginePoincareMonodromyCertificate] = None,
    ) -> PoincareMonodromyCertificate:
        r"""
        **III.ω — Morfismo terminal global Φ_III ∘ Φ_II ∘ Φ_I.**

        Consume la gavilla de Poincaré (cierre de Fase II), **usa el
        1-jet real** (`psheaf.pjet`) y produce el
        `PoincareMonodromyCertificate` agéntico que integra:

          * veredictos de las aduanas (Quillen, Stasheff, Gerbe,
            Poincaré, Mel'nikov, Krein, twist),
          * ínfimo de Heyting sobre el retículo G₃,
          * latencia de crowbar si el ínfimo es ⊥,
          * invariantes de Floquet–Poincaré (μ_max, λ_max, KAM, KY,
            Nekhoroshev, Krein).
        """
        base = psheaf.base_sheaf
        pjet = psheaf.pjet
        verdicts = [
            base.quillen.verdict,
            base.stasheff.verdict,
            base.gerbe.verdict,
            psheaf.poincare.verdict,
        ]
        if psheaf.melnikov is not None:
            verdicts.append(psheaf.melnikov.verdict)
        if psheaf.krein is not None:
            verdicts.append(psheaf.krein.verdict)
        if psheaf.twist is not None:
            verdicts.append(psheaf.twist.verdict)
        final_heyting = self._heyting_meet(*verdicts)
        crowbar_fired = False
        latency_ns = 0.0
        if final_heyting is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            crowbar_fired = True
            logger.critical(
                "¡VETO ATÓMICO TESSERARIOS-POINCARÉ! "
                "Crowbar BT151 [GPIO14] gatillado. "
                "Latencia %.2f ns. Veredictos: Q=%s S=%s G=%s P=%s M=%s K=%s T=%s",
                latency_ns, base.quillen.verdict, base.stasheff.verdict,
                base.gerbe.verdict, psheaf.poincare.verdict,
                psheaf.melnikov.verdict if psheaf.melnikov else "N/A",
                psheaf.krein.verdict if psheaf.krein else "N/A",
                psheaf.twist.verdict if psheaf.twist else "N/A")

        mu = pjet.floquet.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
        lyap_max = float(pjet.lyapunov_max)
        rel_symp = float(base.quillen.ancilla.get(
            "symplectic_residual_rel", float("nan")))
        volume_drift = float(base.quillen.ancilla.get("det_residual", float("nan")))
        krein_def = bool(
            pjet.floquet.krein.krein_definite
            if pjet.floquet.krein is not None else False)
        is_stable = bool(final_heyting is HeytingVerdict.COHERENT)
        return PoincareMonodromyCertificate(
            darboux_dim=pjet.base.n,
            section=pjet.section,
            floquet=pjet.floquet,
            melnikov=pjet.melnikov,
            rotation=pjet.rotation,
            kam=pjet.kam,
            lyapunov=pjet.lyapunov,
            williamson=pjet.williamson,
            cartan=pjet.cartan,
            moser_twist=pjet.moser_twist,
            quillen_verdict=base.quillen.verdict,
            stasheff_verdict=base.stasheff.verdict,
            gerbe_verdict=base.gerbe.verdict,
            poincare_verdict=psheaf.poincare.verdict,
            melnikov_verdict=psheaf.melnikov.verdict if psheaf.melnikov else None,
            krein_verdict=psheaf.krein.verdict if psheaf.krein else None,
            twist_verdict=psheaf.twist.verdict if psheaf.twist else None,
            heyting_verdict=final_heyting.name,
            heyting_value=final_heyting.value,
            crowbar_fired=crowbar_fired,
            actuation_latency_ns=latency_ns,
            relative_symplectic_residual=rel_symp,
            volume_drift=volume_drift,
            max_floquet_multiplier=max_mu,
            lyapunov_exponent=lyap_max,
            is_monodromy_stable=is_stable,
            krein_definite=krein_def,
            nekhoroshev_time=float(pjet.kam.nekhoroshev_time),
            engine_certificate=engine_certificate,
        )

    # ── III.4  Atajo I.ω → II.ω → III.ω ─────────────────────────────────
    def process_poincare_coherence_cycle(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        orbit_period_T: float = 1.0,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        m2_product_tensor: Optional[np.ndarray] = None,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
        hessian: Optional[np.ndarray] = None,
        q_traj_cartan: Optional[np.ndarray] = None,
        p_traj_cartan: Optional[np.ndarray] = None,
        attach_engine_certificate: bool = True,
    ) -> PoincareMonodromyCertificate:
        """Atajo I.ω → II.ω → III.ω sobre tensores crudos de un ciclo."""
        pjet = self.agent.ingest_poincare_tensors(
            jacobian_matrix=jacobian_matrix,
            m3_homotopy_tensor=m3_homotopy_tensor,
            cech_cochain_matrix=cech_cochain_matrix,
            orbit_period_T=orbit_period_T,
            section_index=section_index,
            section_offset=section_offset,
            energy_level=energy_level,
            m2_product_tensor=m2_product_tensor,
            q0_trajectory=q0_trajectory,
            dt_trajectory=dt_trajectory,
            h0_grad=h0_grad,
            h1_grad=h1_grad,
            orbit_points_for_rotation=orbit_points_for_rotation,
            frequency_vector=frequency_vector,
            birkhoff_residual=birkhoff_residual,
            flow_vector=flow_vector,
            hessian=hessian,
            q_traj_cartan=q_traj_cartan,
            p_traj_cartan=p_traj_cartan,
        )
        psheaf = self.agent.compile_poincare_tesserarios_sheaf(pjet)
        eng_cert: Optional[EnginePoincareMonodromyCertificate] = None
        if attach_engine_certificate:
            try:
                eng_cert = self.agent._engine.compute_poincare_monodromy_certificate(
                    jacobian_matrix, orbit_period_T,
                    canonical_omega=self.agent._canonical_omega,
                    frequency_vector=frequency_vector,
                    birkhoff_residual=birkhoff_residual,
                    m2_tensor=m2_product_tensor,
                    cech_matrix=cech_cochain_matrix,
                )
            except (ValueError, SymplecticDimensionError) as exc:
                logger.warning("Certificado del motor no disponible: %s", exc)
        return self.fuse_and_actuate_poincare(psheaf, engine_certificate=eng_cert)

    def process_coherence_cycle(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compatibilidad 3.0: atajo I.ω → II.ω → III sobre 3 aduanas."""
        jet = self.agent.ingest_tensors(
            jacobian_matrix, m3_homotopy_tensor, cech_cochain_matrix,
            m2_product_tensor=m2_product_tensor)
        sheaf = self.agent.compile_tesserarios_sheaf(jet)
        return self.fuse_and_actuate(sheaf)

    def certify_poincare_monodromy(
        self,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        orbit_period_T: float = 1.0,
        **kwargs: Any,
    ) -> PoincareMonodromyCertificate:
        """Alias del morfismo terminal global III.ω."""
        return self.process_poincare_coherence_cycle(
            jacobian_matrix, m3_homotopy_tensor, cech_cochain_matrix,
            orbit_period_T=orbit_period_T, **kwargs)


# =============================================================================
# FACHADAS DE COMPATIBILIDAD (API 3.0 / 4.0 preservada)
# =============================================================================
def execute_tesserarios_cycle(
    agent: HomotopicTesserariosAgent,
    jacobian_matrix: np.ndarray,
    m3_homotopy_tensor: np.ndarray,
    cech_cochain_matrix: np.ndarray,
    m2_product_tensor: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fachada 3.0: ciclo de 3 aduanas vía Cámara."""
    chamber = TesserariosCoherenceChamber(agent, rng=rng)
    return chamber.process_coherence_cycle(
        jacobian_matrix, m3_homotopy_tensor, cech_cochain_matrix,
        m2_product_tensor=m2_product_tensor)


def execute_poincare_tesserarios_cycle(
    agent: HomotopicTesserariosAgent,
    jacobian_matrix: np.ndarray,
    m3_homotopy_tensor: np.ndarray,
    cech_cochain_matrix: np.ndarray,
    orbit_period_T: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> PoincareMonodromyCertificate:
    """
    Fachada 4.1: ciclo completo Poincaré–Floquet–Krein–KAM–Nekhoroshev–
    Lyapunov vía Cámara de Coherencia.
    """
    chamber = TesserariosCoherenceChamber(agent, rng=rng)
    return chamber.certify_poincare_monodromy(
        jacobian_matrix, m3_homotopy_tensor, cech_cochain_matrix,
        orbit_period_T=orbit_period_T, **kwargs)


def _bind_agent_cycle() -> None:
    """Inyecta los ciclos como métodos del agente (compat 3.0 / 4.1)."""

    def _cycle(
        self: HomotopicTesserariosAgent,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        m2_product_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        return execute_tesserarios_cycle(
            self, jacobian_matrix, m3_homotopy_tensor,
            cech_cochain_matrix, m2_product_tensor=m2_product_tensor)

    def _cycle_poincare(
        self: HomotopicTesserariosAgent,
        jacobian_matrix: np.ndarray,
        m3_homotopy_tensor: np.ndarray,
        cech_cochain_matrix: np.ndarray,
        orbit_period_T: float = 1.0,
        **kwargs: Any,
    ) -> PoincareMonodromyCertificate:
        return execute_poincare_tesserarios_cycle(
            self, jacobian_matrix, m3_homotopy_tensor, cech_cochain_matrix,
            orbit_period_T=orbit_period_T, **kwargs)

    HomotopicTesserariosAgent.execute_tesserarios_cycle = _cycle  # type: ignore[attr-defined]
    HomotopicTesserariosAgent.execute_poincare_tesserarios_cycle = _cycle_poincare  # type: ignore[attr-defined]


_bind_agent_cycle()

__all__ = [
    "HeytingVerdict",
    "HomotopicTesserariosAgent",
    "TesserariosCoherenceChamber",
    "PoincareMonodromyCertificate",
    "execute_tesserarios_cycle",
    "execute_poincare_tesserarios_cycle",
    "PoincareMonodromyGerm",
    "SymplecticDimensionError",
]