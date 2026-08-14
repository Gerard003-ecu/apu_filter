# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Banach Algebra Auditor (Soberano de Estabilidad Funcional)          ║
║ Ruta   : app/physics/banach_algebra_auditor.py                               ║
║ Versión: 3.1.0-Banach-Gelfand-Neumann-Wilkinson-Heyting-PhD-Strict           ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y CRITERIO DE COMPLETITUD TOPOLÓGICA (Rigor PhD):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el resolvedor espectral y algebraico que impone la estructura
de un Álgebra de Banach univalente y compleja $$\mathcal{A}$$ sobre el espacio de 
operadores lineales acotados $$\mathcal{B}(\mathcal{H}_n)$$ que actúan en la Malla.

En el análisis funcional moderno, un Álgebra de Banach combina las propiedades 
topológicas de un espacio métrico completo con la estructura algebraica de un 
anillo multiplicativo asociativo sobre el cuerpo $$\mathbb{C}$$. En la base de la 
pirámide, la completez garantiza que el colapso secuencial de las intenciones de 
la IA (secuencias de Cauchy) converja incondicionalmente a autoestados de decisión 
estables dentro de la Malla.

INVARIANTES TOPOLÓGICOS Y TEORÍA ESPECTRAL DE OPERADORES PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────

  [I1] Completitud en el Espacio Métrico (Espacio de Banach):
       Toda sucesión de Cauchy de operadores $$\{T_n\}_{n=1}^{\infty}$$ en $$\mathcal{A}$$ 
       converge de manera absoluta hacia un operador de límite $$T^* \in \mathcal{A}$$:
       $$\lim_{n, m \to \infty} \|T_n - T_m\|_{\mathrm{op}} = 0 \implies \exists T^* \in \mathcal{A} \quad \text{tal que} \quad \lim_{n \to \infty} \|T_n - T^*\|_{\mathrm{op}} = 0$$

  [I2] Submultiplicatividad Estricta de la Norma Espectral:
       La norma del operador de la composición de dos transiciones de fase está 
       estricta y contractivamente acotada por el producto de sus normas individuales:
       $$\|X Y\|_{\mathrm{op}} \le \|X\|_{\mathrm{op}} \cdot \|Y\|_{\mathrm{op}} \quad \forall X, Y \in \mathcal{A} \quad\big[105, 110\big]$$
       Para blindar la mantisa contra la fricción aritmética, se evalúa síncronamente
       el residuo espectral contra la tolerancia elástica de Wilkinson:
       $$r_{\mathrm{sub}} = \|XY\|_2 - \|X\|_2 \|Y\|_2 \le \gamma_n \quad\big[106, 110\big]$$

  [I3] Univalencia e Invarianza de la Identidad Semántica:
       El elemento neutro multiplicativo de la interfaz de despacho $$e = I_n$$ debe 
       satisfacer idénticamente la normalización unitaria de la FPU:
       $$\|e\|_{\mathrm{op}} = 1.0 \quad\big[110\big]$$

  [I4] Isometría Cuántica de la C*-Identidad:
       La inyección de cartuchos semánticos sobre la MAC exige el cumplimiento 
       incondicional de la estructura de C*-álgebra sobre el operador normal:
       $$\|A^\top A\|_2 \equiv \|A\|_2^2 \quad\big[115\big]$$

  [I5] Isometría de Gelfand y Fórmula de Radio Espectral:
       Para todo operador normal $$T \in \mathcal{A}$$, la isometría de Gelfand-Naimark 
       exige que su norma coincida con el radio espectral $$\rho(T)$$, actuando como 
       el límite y el ínfimo de las raíces normadas de las potencias de Banach:
       $$\rho(T) = \max_{\lambda \in \sigma(T)} \lvert\lambda\rvert = \lim_{k \to \infty} \|T^k\|_2^{1/k} \equiv \inf_{k \ge 1} \|T^k\|_2^{1/k} \quad\big[113\big]$$

  [I6] Convergencia Uniforme de la Serie de Neumann bajo Perturbaciones:
       Dada una transición $$T$$ expuesta a una fluctuación de-normalizada del entorno $$\delta T$$, 
       el operador perturbado $$T + \delta T$$ es invertible si y solo si la perturbación se 
       confina estrictamente dentro del radio de convergencia de la serie geométrica:
       $$(T + \delta T)^{-1} = \sum_{k=0}^{\infty} (-1)^k \left( T^{-1} \delta T \right)^k T^{-1} \quad\big[111\big]$$
       Lo cual se verifica síncronamente en la FPU mediante la contracción espectral:
       $$\rho\left( T^{-1} \delta T \right) < 1.0 \quad\big[111\big]$$

ARQUITECTURA EN TRES FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
La transferencia de estado se rige por un acoplamiento monoidal covariante:

  Fase 1 ──► OBSERVE: SANEAMIENTO DE NORMAS (Phase1_BanachNormObserver)
             Calcula la norma espectral operatoria $$\|T\|_2 = \sigma_{\max}(T)$$ y verifica
             la inecuación de submultiplicatividad y el residuo C* en la FPU.
             Entrega: Phase1NormObservation como precondición formal de la Fase 2.

  Fase 2 ──► ORIENT: ANÁLISIS ESPECTRAL DE GELFAND (Phase2_GelfandSpectralOrienter)
             Hereda formalmente la Phase1NormObservation [6]. Computa la forma de Schur 
             compleja para obtener el radio espectral $$\rho(T)$$ y audita el límite asintótico.
             Entrega: Phase2GelfandOrientation como precondición formal de la Fase 3.

  Fase 3 ──► DECIDE & ACT: ESTABILIZACIÓN DE NEUMANN (Phase3_NeumannStabilityDecider)
             Hereda formalmente la Phase2GelfandOrientation. Resuelve la convergencia 
             espectral de la Serie de Neumann [8]. Consolida los veredictos mediante la 
             operación Supremo (join, $$\sqcup$$) sobre el retículo distributivo de Heyting $$\Omega_3$$:
             $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\} \quad\big[108\big]$$
             $$v_{\mathrm{final}} = v_{\mathrm{sub}} \sqcup v_{\mathrm{Gelfand}} \sqcup v_{\mathrm{Neumann}} \in \Omega_3 \quad\big[108\big]$$
             Si el retículo colapsa a VETOED ($$\top$$), detona 'HeytingLatticeVeto'.

  Funtor Supremo del Auditor:
             $$\mathcal{Z}_{\mathrm{motor}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[116\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.BanachAlgebraAuditor")

__version__: Final[str] = "3.0.0-Banach-Gelfand-Neumann-Wilkinson-Heyting-PhD"

# ══════════════════════════════════════════════════════════════════════════════
# §0. CONSTANTES NUMÉRICAS (Wilkinson / IEEE-754 binary64 / LAPACK)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)          # u = 2⁻⁵³
_SAFE_DENOM: Final[float] = 1.0e-300                                  # evita 0/0
_DEFAULT_REL_TOL: Final[float] = 1.0e-11
_DEFAULT_HARD_TOL: Final[float] = 1.0e-5
_SPECTRAL_MARGIN: Final[float] = 1.0e-12
_DEFAULT_POTENCY_LIMIT: Final[int] = 6
_MAX_POTENCY_LIMIT: Final[int] = 16
_HASH_ROUND_DIGITS: Final[int] = 16


def _wilkinson_gamma(n: int) -> float:
    """
    Constante clásica de Wilkinson γ_n = n u / (1 − n u), n u < 1.

    Cota a priori del factor de acumulación de redondeo en un producto
    interno de longitud n (Higham, Accuracy and Stability, Thm. 3.1).
    """
    dim = max(int(n), 1)
    nu = dim * _MACHINE_EPS
    if nu >= 1.0:
        return 1.0
    return nu / (1.0 - nu)


def _svd_norm_relerr_bound(n: int) -> float:
    """
    Cota relativa típica del error de σ_max calculado por SVD/QR (LAPACK
    xGESDD / xGESVD): O(n u) en norma 2. Se usa para no vetar residuos
    que son artefactos del redondeo y no del álgebra.
    """
    return max(4.0 * _wilkinson_gamma(max(n, 1)), 32.0 * _MACHINE_EPS)


def _clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES FUNCIONALES
# ══════════════════════════════════════════════════════════════════════════════
class BanachHeytingVerdict(IntEnum):
    """
    Clasificador de subobjetos en el topos de Banach.

    Álgebra de Heyting linealmente ordenada (cadena finita):

        ⊥ = COHERENT  ≼  DEGRADED  ≼  VETOED = ⊤_veto

    Join = max, meet = min, implicación a → b = ⊤ si a ≼ b, else b.
    La monotonía del join garantiza que una degradación no puede
    «curarse» en una fase posterior (propagación monótona de defectos).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


def heyting_join(*verdicts: BanachHeytingVerdict) -> BanachHeytingVerdict:
    """Supremo (disyunción interna) de una familia finita de veredictos."""
    if not verdicts:
        return BanachHeytingVerdict.COHERENT
    return BanachHeytingVerdict(max(v.value for v in verdicts))


def heyting_meet(*verdicts: BanachHeytingVerdict) -> BanachHeytingVerdict:
    """Ínfimo (conjunción interna) de una familia finita de veredictos."""
    if not verdicts:
        return BanachHeytingVerdict.VETOED
    return BanachHeytingVerdict(min(v.value for v in verdicts))


def heyting_implies(
    antecedent: BanachHeytingVerdict,
    consequent: BanachHeytingVerdict,
) -> BanachHeytingVerdict:
    """Implicación de Heyting en la cadena COHERENT ≼ DEGRADED ≼ VETOED."""
    if antecedent.value <= consequent.value:
        return BanachHeytingVerdict.COHERENT
    return consequent


def classify_relative_defect(
    defect: float,
    scale: float,
    soft_tol: float,
    hard_tol: float,
    floor: float = 0.0,
) -> BanachHeytingVerdict:
    """
    Clasifica un defecto no negativo relativo a `scale`.

    Se compara defect / max(scale, floor, 1) contra (soft_tol, hard_tol).
    El piso evita vetar residuos de redondeo cuando la escala es nula.
    """
    denom = max(abs(scale), floor, 1.0)
    rel = _clamp_nonneg(float(defect)) / denom
    if rel > hard_tol:
        return BanachHeytingVerdict.VETOED
    if rel > soft_tol:
        return BanachHeytingVerdict.DEGRADED
    return BanachHeytingVerdict.COHERENT


class BanachAlgebraError(Exception):
    """Excepción raíz para violaciones en el álgebra de Banach computada."""


class SubmultiplicativityViolation(BanachAlgebraError):
    """‖XY‖₂ > ‖X‖₂·‖Y‖₂ fuera de la cota de Wilkinson."""


class IdentityDegeneracyError(BanachAlgebraError):
    """‖I_n‖₂ difiere de 1 más allá de la tolerancia dura."""


class NeumannSeriesDivergence(BanachAlgebraError):
    """ρ(T⁻¹ dT) ≥ 1 − margen, o T no es invertible."""


class HeytingLatticeVeto(BanachAlgebraError):
    """Colapso del retículo de Heyting al supremo terminal VETOED."""

    def __init__(
        self,
        message: str,
        *,
        verdict: BanachHeytingVerdict = BanachHeytingVerdict.VETOED,
        cause: Optional[BanachAlgebraError] = None,
    ) -> None:
        super().__init__(message)
        self.verdict = verdict
        self.cause = cause


# ══════════════════════════════════════════════════════════════════════════════
# §B. PRIMITIVAS NUMÉRICAS ESTABLES (capa 0, compartida por las tres fases)
# ══════════════════════════════════════════════════════════════════════════════
def _as_float64_square(X: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """
    Inmersión estricta en M_n(ℝ) ⊂ B(H_n).

    Copia C-contigua float64 para:
      (i)  inmunizar el auditor frente a aliasing / mutación externa,
      (ii) garantizar que LAPACK reciba el dtype IEEE-754 esperado.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"El operador {name} debe ser numpy.ndarray, no {type(X)!r}.")
    if X.ndim != 2:
        raise ValueError(f"El operador {name} debe ser bidimensional; ndim={X.ndim}.")
    rows, cols = int(X.shape[0]), int(X.shape[1])
    if rows != cols or rows == 0:
        raise ValueError(
            f"El operador {name} debe ser cuadrado y de dimensión positiva: {X.shape}."
        )
    materialised = np.array(X, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(materialised)):
        raise ValueError(f"El operador {name} contiene valores no finitos (NaN/Inf).")
    return materialised


def _assert_conformal_pair(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    name_x: str,
    name_y: str,
) -> None:
    if X.shape != Y.shape:
        raise ValueError(
            f"{name_x} y {name_y} deben compartir dimensión: {X.shape} != {Y.shape}."
        )


def _svd_spectrum(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """Valores singulares σ₁ ≥ … ≥ σ_n ≥ 0 (LAPACK xGESDD)."""
    sigma = la.svdvals(A)
    return np.asarray(sigma, dtype=np.float64)


def _operator_norm_2(A: NDArray[np.float64]) -> float:
    """Norma operatoria ‖A‖₂ = σ_max(A). Convención: ‖0‖₂ = 0."""
    sigma = _svd_spectrum(A)
    if sigma.size == 0:
        return 0.0
    return float(sigma[0])


def _frobenius_norm(A: NDArray[np.float64]) -> float:
    """‖A‖_F = (∑_{ij} a_{ij}²)^{1/2} = (∑_k σ_k²)^{1/2}."""
    return float(la.norm(A, ord="fro"))


def _spectral_condition(sigma: NDArray[np.float64]) -> float:
    """κ₂(A) = σ_max / σ_min ∈ [1, +∞]. Convención: κ₂(0) = +∞."""
    if sigma.size == 0:
        return float("inf")
    smax = float(sigma[0])
    smin = float(sigma[-1])
    if smax <= 0.0 or smin <= 0.0:
        return float("inf")
    return smax / smin


def _numerical_rank(sigma: NDArray[np.float64], n: int) -> int:
    """
    Rango numérico: #{σ_k > n u σ_max}.

    Es el criterio clásico de Golub–Van Loan para el rango en precisión
    finita. Coincide con el rango algebraico cuando A está bien escalada
    y no hay un hueco singular comparable a n u ‖A‖₂.
    """
    if sigma.size == 0:
        return 0
    smax = float(sigma[0])
    if smax <= 0.0:
        return 0
    tol = max(n, 1) * _MACHINE_EPS * smax
    return int(np.count_nonzero(sigma > tol))


def _stable_power_norm_root(T: NDArray[np.float64], k: int) -> float:
    """
    Evalúa ‖Tᵏ‖₂^{1/k} sin overflow.

    Escalado homogéneo: T = ‖T‖₂ · S con ‖S‖₂ ≃ 1, de modo que

        ‖Tᵏ‖₂^{1/k}  =  ‖T‖₂ · ‖Sᵏ‖₂^{1/k}.

    La raíz se toma en log-espacio para k ≥ 1:

        exp( (1/k) log ‖Sᵏ‖₂ ) · ‖T‖₂.

    Si Sᵏ subfluye a la matriz nula, se devuelve 0 (k pequeño en la
    práctica de esta aduana; potency_limit ≤ 16).
    """
    if k < 1:
        raise ValueError(f"El exponente de Gelfand debe ser ≥ 1; recibido k={k}.")
    base_norm = _operator_norm_2(T)
    if base_norm <= 0.0:
        return 0.0
    if k == 1:
        return base_norm
    scaled = T / base_norm
    powered = np.linalg.matrix_power(scaled, k)
    powered_norm = _operator_norm_2(powered)
    if powered_norm <= 0.0:
        return 0.0
    return float(base_norm * np.exp(np.log(powered_norm) / k))


def _spectral_radius_via_schur(T: NDArray[np.float64]) -> Tuple[float, NDArray[np.complex128]]:
    """
    Radio espectral vía forma de Schur compleja T = Q U Q*.

    ρ(T) = máx_i |u_{ii}|. La forma de Schur es el algoritmo numéricamente
    estable (QR implícito de Francis) subyacente a xGEEV; devolver U
    permite residuales de triangularidad y el espectro sin un segundo
    paso de eigenvalores.
    """
    schur_u, _ = la.schur(T, output="complex")
    diag = np.diag(schur_u)
    if diag.size == 0:
        return 0.0, np.zeros((0,), dtype=np.complex128)
    rho = float(np.max(np.abs(diag)))
    return rho, np.asarray(diag, dtype=np.complex128)


def _departure_from_normality(T: NDArray[np.float64]) -> float:
    """
    Defecto de normalidad de Henrici:

        Δ(T) := ‖TᵀT − TTᵀ‖_F.

    Δ(T) = 0 ⇔ T es normal ⇔ ‖T‖₂ = ρ(T) ⇔ T es unitariamente
    diagonalizable sobre ℂ. Controla la rapidez de convergencia de
    Gelfand: si Δ(T) ≫ 0, ‖Tᵏ‖^{1/k} puede permanecer ≫ ρ durante
    muchos k (pseudospectro de Trefethen).
    """
    gram_left = T.T @ T
    gram_right = T @ T.T
    return _frobenius_norm(gram_left - gram_right)


def _matrix_invariants(A: NDArray[np.float64]) -> Tuple[int, float, float, float]:
    """Invariantes baratos para la huella forense: (n, tr, ‖·‖_F, suma)."""
    return (
        int(A.shape[0]),
        float(np.trace(A)),
        _frobenius_norm(A),
        float(np.sum(A)),
    )


def _round_for_hash(x: float) -> str:
    if not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{float(x):.{_HASH_ROUND_DIGITS}e}"


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES (contratos categóricos de handoff entre fases)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1NormObservation:
    """
    Artefacto terminal de la FASE 1 (Observe).

    Es la *única* precondición estricta de la FASE 2: un objeto del
    tipo Obs₁ que certifica (o degrada/veta) los axiomas [I2]–[I4]
    sobre el par (X, Y) ∈ B(H_n)×B(H_n).
    """

    dimension: int
    norm_x: float
    norm_y: float
    norm_composed: float
    identity_norm: float
    frobenius_x: float
    frobenius_y: float
    condition_x: float
    condition_y: float
    numerical_rank_x: int
    numerical_rank_y: int
    star_identity_residual_x: float
    star_identity_residual_y: float
    submultiplicativity_residual: float
    identity_residual: float
    wilkinson_bound: float
    relative_submultiplicative_violation: float
    is_submultiplicative: bool
    is_identity_coherent: bool
    is_cstar_coherent: bool
    verdict: BanachHeytingVerdict
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Phase2GelfandOrientation:
    """
    Artefacto terminal de la FASE 2 (Orient).

    Precondición estricta de la FASE 3. Contiene ρ(T), la órbita de
    Gelfand {‖Tᵏ‖₂^{1/k} : k = 1..K}, el defecto de normalidad y el
    gap espectral ‖T‖₂ − ρ(T) ≥ 0.
    """

    phase1_observation: Phase1NormObservation
    spectral_radius: float
    operator_norm_t: float
    spectral_norm_gap: float
    gelfand_bounds: Tuple[float, ...]
    gelfand_residual: float
    gelfand_monotonic_defect: float
    normality_defect: float
    is_numerically_normal: bool
    is_gelfand_consistent: bool
    spectrum_sample: Tuple[complex, ...]
    verdict: BanachHeytingVerdict
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BanachGovernanceState:
    """
    Objeto terminal de la gobernanza de Banach (Decide & Act).

    Certificado maestro con trazabilidad forense, residual de inversión,
    radio de Neumann, cota de Kato–Rellich y veredicto en el retículo.
    """

    phase2_orientation: Phase2GelfandOrientation
    neumann_radius: float
    neumann_operator_norm: float
    inversion_residual: float
    rcond_t: float
    condition_number_t: float
    kato_inverse_bound: float
    is_neumann_stable: bool
    is_neumann_sufficient_by_norm: bool
    singular_base: bool
    final_verdict: BanachHeytingVerdict
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 1 — OBSERVACIÓN AXIOMÁTICA DE NORMAS
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_BanachNormObserver:
    """
    Fase 1 (Observe) : norma espectral, unidad, submultiplicatividad, C*.

    Morfismo Φ₁ : B(H_n)×B(H_n) → Obs₁.

    Descomposición granular (cada método es un juicio atómico; el sello
    terminal `_observe_norms` es el único objeto que la Fase 2 acepta):

        validación  →  espectro singular  →  axiomas de norma
                    →  [I3] identidad     →  [I2] submultiplicatividad
                    →  [I4] C*-identidad  →  join de Heyting
                    →  sello terminal Obs₁
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_REL_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
    ) -> None:
        if not (0.0 < float(soft_tol) <= float(hard_tol)):
            raise ValueError(
                f"Se exige 0 < soft_tol ≤ hard_tol; recibido "
                f"soft_tol={soft_tol}, hard_tol={hard_tol}."
            )
        self._soft_tol = float(soft_tol)
        self._hard_tol = float(hard_tol)

    # ── D.1  Validación de pertenencia a M_n(ℝ) ──────────────────────────
    def _materialise_pair(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Inmersión conjunta y conformidad dimensional (producto interno)."""
        x_mat = _as_float64_square(X, "X")
        y_mat = _as_float64_square(Y, "Y")
        _assert_conformal_pair(x_mat, y_mat, "X", "Y")
        return x_mat, y_mat

    # ── D.2  Lectura espectral singular ──────────────────────────────────
    def _read_singular_portrait(
        self,
        A: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float, float, int]:
        """
        Retrato singular: (σ, ‖A‖₂, ‖A‖_F, κ₂(A), rango numérico).

        Desigualdad de normas (siempre auditada implícitamente):
            ‖A‖₂  ≤  ‖A‖_F  ≤  √n · ‖A‖₂.
        """
        n = int(A.shape[0])
        sigma = _svd_spectrum(A)
        op_norm = float(sigma[0]) if sigma.size else 0.0
        fro = _frobenius_norm(A)
        cond = _spectral_condition(sigma)
        rank = _numerical_rank(sigma, n)
        return sigma, op_norm, fro, cond, rank

    # ── D.3  Identidad C*  ‖AᵀA‖₂ = ‖A‖₂² ────────────────────────────────
    def _cstar_identity_residual(self, A: NDArray[np.float64], op_norm: float) -> float:
        """
        Residuo de la identidad C* en B(H_n):

            r₊(A) := | ‖AᵀA‖₂ − ‖A‖₂² |.

        En aritmética exacta r₊ ≡ 0. Un residuo grande denuncia o bien
        un error de SVD o una matriz no real-adjuntamente coherente
        (no aplicable aquí: trabajamos en M_n(ℝ) con la adjunta = transpuesta).
        """
        gram_norm = _operator_norm_2(A.T @ A)
        return abs(gram_norm - op_norm * op_norm)

    # ── D.4  Axioma [I3] : univalencia de I_n ────────────────────────────
    def _audit_identity_univalence(
        self,
        n: int,
        soft_tol: float,
        hard_tol: float,
    ) -> Tuple[float, float, BanachHeytingVerdict]:
        """
        Verifica ‖I_n‖₂ = 1.

        Es un test de *auto-consistencia del oráculo de norma*: el
        espectro singular de I_n es {1}ⁿ, luego σ_max = 1 exactamente.
        Cualquier desviación es error de SVD / redondeo, no del álgebra.
        """
        identity = np.eye(n, dtype=np.float64, order="C")
        identity_norm = _operator_norm_2(identity)
        identity_residual = abs(identity_norm - 1.0)
        verdict = classify_relative_defect(
            defect=identity_residual,
            scale=1.0,
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            floor=1.0,
        )
        return identity_norm, identity_residual, verdict

    # ── D.5  Axioma [I2] : submultiplicatividad con cota de Wilkinson ────
    def _audit_submultiplicativity(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        norm_x: float,
        norm_y: float,
        soft_tol: float,
        hard_tol: float,
    ) -> Tuple[float, float, float, float, BanachHeytingVerdict]:
        """
        Audita ‖XY‖₂ ≤ ‖X‖₂·‖Y‖₂.

        Residuo firmado:
            r := ‖XY‖₂ − ‖X‖₂·‖Y‖₂.

        En aritmética exacta r ≤ 0. El producto GEMM introduce un
        backward-error ‖E‖ ≤ γ_n ‖X‖‖Y‖ y cada norma SVD aporta
        O(n u) relativo. Se veta sólo el exceso positivo por encima
        de esa cota combinada, nunca el cumplimiento teórico.
        """
        n = int(X.shape[0])
        composed = X @ Y
        norm_composed = _operator_norm_2(composed)
        residual = norm_composed - (norm_x * norm_y)
        positive_violation = _clamp_nonneg(residual)

        gemm_bound = _wilkinson_gamma(n) * max(norm_x * norm_y, 1.0)
        svd_bound = _svd_norm_relerr_bound(n) * max(norm_x * norm_y, norm_composed, 1.0)
        wilkinson_bound = gemm_bound + 3.0 * svd_bound

        excess_over_roundoff = _clamp_nonneg(positive_violation - wilkinson_bound)
        scale = max(1.0, norm_x * norm_y, norm_composed)
        relative_violation = excess_over_roundoff / scale
        verdict = classify_relative_defect(
            defect=excess_over_roundoff,
            scale=scale,
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            floor=1.0,
        )
        return norm_composed, residual, wilkinson_bound, relative_violation, verdict

    # ── D.6  Join de átomos de la Fase 1 ─────────────────────────────────
    def _compose_phase1_heyting(
        self,
        *atoms: BanachHeytingVerdict,
    ) -> BanachHeytingVerdict:
        """Join monótono de los juicios atómicos de la Fase 1."""
        return heyting_join(*atoms)

    # ── D.7  SELLO TERMINAL DE LA FASE 1 ─────────────────────────────────
    def _observe_norms(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        soft_tol: Optional[float] = None,
        hard_tol: Optional[float] = None,
    ) -> Phase1NormObservation:
        """
        Φ₁ — morfismo terminal de la Fase 1.

        Audita sobre el par (X, Y):
          • ‖X‖₂, ‖Y‖₂, ‖XY‖₂, κ₂, rango numérico, ‖·‖_F;
          • [I3]  ‖I_n‖₂ = 1;
          • [I2]  ‖XY‖₂ ≤ ‖X‖₂·‖Y‖₂  (con cota de Wilkinson);
          • [I4]  ‖AᵀA‖₂ = ‖A‖₂²     (residuo C*).

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₁ := Phase1NormObservation. El juicio

            Φ₁(X, Y) ∈ Obs₁

        es un objeto inmutable del topos de Banach y constituye, por
        contrato categórico, la *unidad de arranque* de la Fase 2:

            Φ₂  se aplica únicamente a  (Φ₁(X, Y), T) ,
            y su primer método `_ingest_phase1_precondition` es la
            continuación lógica y tipada de este sello.

        Cualquier consumidor que no pase por este método viola la
        adjunción Observe ⊣ Orient.
        """
        soft = float(self._soft_tol if soft_tol is None else soft_tol)
        hard = float(self._hard_tol if hard_tol is None else hard_tol)
        if not (0.0 < soft <= hard):
            raise ValueError(f"Tolerancias inválidas: soft={soft}, hard={hard}.")

        x_mat, y_mat = self._materialise_pair(X, Y)
        n = int(x_mat.shape[0])

        _, norm_x, fro_x, cond_x, rank_x = self._read_singular_portrait(x_mat)
        _, norm_y, fro_y, cond_y, rank_y = self._read_singular_portrait(y_mat)

        cstar_x = self._cstar_identity_residual(x_mat, norm_x)
        cstar_y = self._cstar_identity_residual(y_mat, norm_y)
        cstar_scale = max(1.0, norm_x * norm_x, norm_y * norm_y)
        cstar_verdict = self._compose_phase1_heyting(
            classify_relative_defect(cstar_x, cstar_scale, soft, hard, floor=1.0),
            classify_relative_defect(cstar_y, cstar_scale, soft, hard, floor=1.0),
        )

        identity_norm, identity_residual, identity_verdict = (
            self._audit_identity_univalence(n, soft, hard)
        )

        (
            norm_composed,
            residual,
            wilkinson_bound,
            relative_violation,
            submult_verdict,
        ) = self._audit_submultiplicativity(
            x_mat, y_mat, norm_x, norm_y, soft, hard
        )

        combined = self._compose_phase1_heyting(
            submult_verdict, identity_verdict, cstar_verdict
        )

        atoms: Tuple[str, ...] = (
            f"n={n}",
            f"‖X‖₂={norm_x:.6e}",
            f"‖Y‖₂={norm_y:.6e}",
            f"‖XY‖₂={norm_composed:.6e}",
            f"r_sub={residual:.6e}",
            f"γ_bound={wilkinson_bound:.6e}",
            f"‖I‖₂={identity_norm:.16e}",
            f"κ₂(X)={cond_x:.6e}",
            f"κ₂(Y)={cond_y:.6e}",
            f"rk(X)={rank_x}",
            f"rk(Y)={rank_y}",
            f"C*(X)={cstar_x:.6e}",
            f"C*(Y)={cstar_y:.6e}",
            f"join={combined.name}",
        )

        if combined == BanachHeytingVerdict.VETOED:
            logger.error("[BANACH:Φ₁] Observación VETOED. %s", " | ".join(atoms))
        elif combined == BanachHeytingVerdict.DEGRADED:
            logger.warning("[BANACH:Φ₁] Observación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[BANACH:Φ₁] Observación COHERENT. n=%d", n)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 2.
        return Phase1NormObservation(
            dimension=n,
            norm_x=norm_x,
            norm_y=norm_y,
            norm_composed=norm_composed,
            identity_norm=identity_norm,
            frobenius_x=fro_x,
            frobenius_y=fro_y,
            condition_x=cond_x,
            condition_y=cond_y,
            numerical_rank_x=rank_x,
            numerical_rank_y=rank_y,
            star_identity_residual_x=cstar_x,
            star_identity_residual_y=cstar_y,
            submultiplicativity_residual=residual,
            identity_residual=identity_residual,
            wilkinson_bound=wilkinson_bound,
            relative_submultiplicative_violation=relative_violation,
            is_submultiplicative=submult_verdict != BanachHeytingVerdict.VETOED,
            is_identity_coherent=identity_verdict != BanachHeytingVerdict.VETOED,
            is_cstar_coherent=cstar_verdict != BanachHeytingVerdict.VETOED,
            verdict=combined,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 2 — ORIENTACIÓN ESPECTRAL DE GELFAND–SCHUR
#     Continuación formal del sello terminal de la Fase 1.
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_GelfandSpectralOrienter(Phase1_BanachNormObserver):
    """
    Fase 2 (Orient) : radio espectral, Gelfand, normalidad.

    Morfismo Φ₂ : Obs₁ × B(H_n) → Obs₂.

    El primer método de esta fase, `_ingest_phase1_precondition`, es la
    *continuación tipada* de `_observe_norms`. No existe camino legal
    hacia Gelfand que no atraviese ese ingest.
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_REL_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        potency_limit: int = _DEFAULT_POTENCY_LIMIT,
    ) -> None:
        super().__init__(soft_tol=soft_tol, hard_tol=hard_tol)
        self._potency_limit = self._validate_potency_limit(potency_limit)

    @staticmethod
    def _validate_potency_limit(potency_limit: int) -> int:
        k = int(potency_limit)
        if k < 1 or k > _MAX_POTENCY_LIMIT:
            raise ValueError(
                f"potency_limit debe vivir en [1, {_MAX_POTENCY_LIMIT}]; recibido {k}."
            )
        return k

    # ── E.1  INICIO DE FASE 2 = continuación del sello Φ₁ ────────────────
    def _ingest_phase1_precondition(
        self,
        phase1_obs: Phase1NormObservation,
    ) -> Phase1NormObservation:
        """
        Continuación formal de `Phase1_BanachNormObserver._observe_norms`.

        Teorema de handoff (Obs₁ ↪ Fase 2)
        ──────────────────────────────────
        Hipótesis: `phase1_obs` es el valor de retorno de `_observe_norms`
        (objeto congelado de tipo Phase1NormObservation).

        Tesis: el objeto es una precondición *habitable* de Φ₂. Se verifica:
          (i)  tipado e inmutabilidad (dataclass frozen);
          (ii) dimensión positiva y normas finitas no negativas;
          (iii) el veredicto de Heyting está en el retículo {0,1,2}.

        No se «repara» un VETOED de Fase 1: el join monótono lo
        transportará hasta el veredicto terminal. Este método es el
        único puerto de entrada de la Fase 2.
        """
        if not isinstance(phase1_obs, Phase1NormObservation):
            raise TypeError(
                "Fase 2 exige el sello terminal de Fase 1 "
                f"(Phase1NormObservation); recibido {type(phase1_obs)!r}."
            )
        if phase1_obs.dimension <= 0:
            raise ValueError(
                f"Obs₁ tiene dimensión no positiva: n={phase1_obs.dimension}."
            )
        for label, value in (
            ("norm_x", phase1_obs.norm_x),
            ("norm_y", phase1_obs.norm_y),
            ("norm_composed", phase1_obs.norm_composed),
            ("identity_norm", phase1_obs.identity_norm),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Obs₁.{label} no es una norma válida: {value!r}.")
        if not isinstance(phase1_obs.verdict, BanachHeytingVerdict):
            raise TypeError("Obs₁.verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[BANACH:Φ₂←Φ₁] Ingestión de Obs₁ (verdict=%s, n=%d).",
            phase1_obs.verdict.name,
            phase1_obs.dimension,
        )
        return phase1_obs

    # ── E.2  Radio espectral estable ─────────────────────────────────────
    def _compute_spectral_radius(
        self,
        T: NDArray[np.float64],
    ) -> Tuple[float, Tuple[complex, ...]]:
        """ρ(T) = máx |λ| vía Schur complejo, con espectro muestral."""
        rho, diag = _spectral_radius_via_schur(T)
        spectrum = tuple(complex(z) for z in diag.tolist())
        return rho, spectrum

    # ── E.3  Órbita de Gelfand escalada ──────────────────────────────────
    def _gelfand_orbit(
        self,
        T: NDArray[np.float64],
        potency_limit: int,
    ) -> Tuple[float, ...]:
        """
        Sucesión g_k = ‖Tᵏ‖₂^{1/k}, k = 1..K.

        Teorema de Gelfand: g_k → ρ(T). Además g_k ≥ ρ(T) para toda
        norma de álgebra de Banach (en exacto). La sucesión no es
        necesariamente monótona; se registra el defecto de monotonía
        como diagnóstico, no como veto.
        """
        bounds = [_stable_power_norm_root(T, k) for k in range(1, potency_limit + 1)]
        return tuple(bounds)

    # ── E.4  Consistencia Gelfand ↔ espectro ─────────────────────────────
    def _classify_gelfand_consistency(
        self,
        rho: float,
        gelfand_bounds: Sequence[float],
        soft_tol: float,
        hard_tol: float,
        n: int,
    ) -> Tuple[float, float, BanachHeytingVerdict]:
        """
        Dos juicios:
          (a) Dirección prohibida: g_K < ρ − O(n u) · escala  → veto.
              (viola ρ ≤ ‖Tᵏ‖^{1/k}.)
          (b) Defecto de monotonía: media de incrementos negativos de
              {g_k}. Informativo; no veta (Gelfand no es monótona).
        """
        if not gelfand_bounds:
            raise ValueError("La órbita de Gelfand no puede ser vacía.")
        last_bound = float(gelfand_bounds[-1])
        gelfand_residual = last_bound - rho
        impossible = _clamp_nonneg(-gelfand_residual)
        scale = max(1.0, rho, last_bound)
        # Se relaja la cota dura con el error SVD de las potencias.
        amp = 1.0 + _svd_norm_relerr_bound(n)
        verdict = classify_relative_defect(
            defect=impossible,
            scale=scale,
            soft_tol=soft_tol * amp,
            hard_tol=hard_tol * amp,
            floor=1.0,
        )
        decrements = [
            _clamp_nonneg(float(gelfand_bounds[i]) - float(gelfand_bounds[i + 1]))
            for i in range(len(gelfand_bounds) - 1)
        ]
        monotonic_defect = float(np.mean(decrements)) if decrements else 0.0
        return gelfand_residual, monotonic_defect, verdict

    # ── E.5  Gap espectral y normalidad ──────────────────────────────────
    def _spectral_gap_and_normality(
        self,
        T: NDArray[np.float64],
        rho: float,
        operator_norm: float,
        soft_tol: float,
        hard_tol: float,
    ) -> Tuple[float, float, bool, BanachHeytingVerdict]:
        """
        [I5]  gap := ‖T‖₂ − ρ(T) ≥ 0.
        Δ(T) pequeño ⇒ T numéricamente normal ⇒ gap ≃ 0.

        Un gap *negativo* (ρ > ‖T‖₂) es numéricamente imposible y se veta.
        """
        gap = operator_norm - rho
        impossible = _clamp_nonneg(-gap)
        scale = max(1.0, operator_norm, rho)
        gap_verdict = classify_relative_defect(
            defect=impossible,
            scale=scale,
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            floor=1.0,
        )
        defect = _departure_from_normality(T)
        # Umbral de normalidad: Δ(T) ≤ hard_tol · max(1, ‖T‖_F²).
        fro = max(_frobenius_norm(T), _SAFE_DENOM)
        is_normal = defect <= hard_tol * max(1.0, fro * fro)
        return gap, defect, is_normal, gap_verdict

    # ── E.6  SELLO TERMINAL DE LA FASE 2 ─────────────────────────────────
    def _orient_gelfand(
        self,
        phase1_obs: Phase1NormObservation,
        T: NDArray[np.float64],
        potency_limit: Optional[int] = None,
        soft_tol: Optional[float] = None,
        hard_tol: Optional[float] = None,
    ) -> Phase2GelfandOrientation:
        """
        Φ₂ — morfismo terminal de la Fase 2.

        Calcula ρ(T) y verifica la fórmula asintótica de Gelfand

            ρ(T)  =  lim_{k→∞} ‖Tᵏ‖₂^{1/k}

        sobre un horizonte finito K = potency_limit, con potencias
        escaladas. El join con Obs₁ transporta cualquier degradación.

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₂ := Phase2GelfandOrientation. El juicio

            Φ₂(Φ₁(X, Y), T) ∈ Obs₂

        es la *unidad de arranque* de la Fase 3: su primer método
        `_ingest_phase2_precondition` es la continuación lógica de
        este sello.
        """
        obs1 = self._ingest_phase1_precondition(phase1_obs)
        soft = float(self._soft_tol if soft_tol is None else soft_tol)
        hard = float(self._hard_tol if hard_tol is None else hard_tol)
        horizon = self._validate_potency_limit(
            self._potency_limit if potency_limit is None else potency_limit
        )

        t_mat = _as_float64_square(T, "T")
        if int(t_mat.shape[0]) != obs1.dimension:
            raise ValueError(
                f"T tiene dimensión {t_mat.shape[0]}, "
                f"incompatible con Obs₁.n={obs1.dimension}."
            )

        rho, spectrum = self._compute_spectral_radius(t_mat)
        operator_norm_t = _operator_norm_2(t_mat)
        gelfand_bounds = self._gelfand_orbit(t_mat, horizon)
        gelfand_residual, mono_defect, gelfand_verdict = (
            self._classify_gelfand_consistency(
                rho, gelfand_bounds, soft, hard, obs1.dimension
            )
        )
        gap, normality_defect, is_normal, gap_verdict = (
            self._spectral_gap_and_normality(
                t_mat, rho, operator_norm_t, soft, hard
            )
        )

        combined = heyting_join(obs1.verdict, gelfand_verdict, gap_verdict)
        is_gelfand_consistent = gelfand_verdict != BanachHeytingVerdict.VETOED

        atoms: Tuple[str, ...] = (
            f"ρ={rho:.6e}",
            f"‖T‖₂={operator_norm_t:.6e}",
            f"gap={gap:.6e}",
            f"g_K={gelfand_bounds[-1]:.6e}",
            f"r_G={gelfand_residual:.6e}",
            f"Δ(T)={normality_defect:.6e}",
            f"normal={is_normal}",
            f"join={combined.name}",
        )

        if combined == BanachHeytingVerdict.VETOED:
            logger.error("[BANACH:Φ₂] Orientación VETOED. %s", " | ".join(atoms))
        elif combined == BanachHeytingVerdict.DEGRADED:
            logger.warning("[BANACH:Φ₂] Orientación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[BANACH:Φ₂] Orientación COHERENT. ρ=%.6e", rho)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 3.
        return Phase2GelfandOrientation(
            phase1_observation=obs1,
            spectral_radius=rho,
            operator_norm_t=operator_norm_t,
            spectral_norm_gap=gap,
            gelfand_bounds=gelfand_bounds,
            gelfand_residual=gelfand_residual,
            gelfand_monotonic_defect=mono_defect,
            normality_defect=normality_defect,
            is_numerically_normal=is_normal,
            is_gelfand_consistent=is_gelfand_consistent,
            spectrum_sample=spectrum,
            verdict=combined,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 3 — CERTIFICACIÓN DE NEUMANN–KATO Y VETO TERMINAL
#     Continuación formal del sello terminal de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_NeumannStabilityDecider(Phase2_GelfandSpectralOrienter):
    """
    Fase 3 (Decide & Act) : invertibilidad perturbativa y veredicto.

    Morfismo Φ₃ : Obs₂ × B(H_n)×B(H_n) → Gov.

    Consume la orientación espectral de Fase 2 y decide si el operador
    perturbado T + dT es invertible incondicionalmente en B(H_n) vía
    la serie de Neumann de W := T⁻¹ dT.
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_REL_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        potency_limit: int = _DEFAULT_POTENCY_LIMIT,
        spectral_margin: float = _SPECTRAL_MARGIN,
    ) -> None:
        super().__init__(
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            potency_limit=potency_limit,
        )
        margin = float(spectral_margin)
        if not (0.0 <= margin < 1.0):
            raise ValueError(
                f"spectral_margin debe vivir en [0, 1); recibido {margin}."
            )
        self._spectral_margin = margin

    # ── F.1  INICIO DE FASE 3 = continuación del sello Φ₂ ────────────────
    def _ingest_phase2_precondition(
        self,
        phase2_orient: Phase2GelfandOrientation,
    ) -> Phase2GelfandOrientation:
        """
        Continuación formal de `Phase2_GelfandSpectralOrienter._orient_gelfand`.

        Teorema de handoff (Obs₂ ↪ Fase 3)
        ──────────────────────────────────
        Hipótesis: `phase2_orient` es el valor de retorno de `_orient_gelfand`.

        Tesis: Obs₂ es habitable. Se exige:
          (i)  tipado Phase2GelfandOrientation;
          (ii) Obs₁ anidado válido (re-ingesta de Fase 1);
          (iii) ρ(T) finito y no negativo;
          (iv) órbita de Gelfand no vacía.

        El veredicto de Obs₂ se transporta monótonamente; no se repara.
        """
        if not isinstance(phase2_orient, Phase2GelfandOrientation):
            raise TypeError(
                "Fase 3 exige el sello terminal de Fase 2 "
                f"(Phase2GelfandOrientation); recibido {type(phase2_orient)!r}."
            )
        self._ingest_phase1_precondition(phase2_orient.phase1_observation)
        if not np.isfinite(phase2_orient.spectral_radius) or phase2_orient.spectral_radius < 0.0:
            raise ValueError(
                f"Obs₂.spectral_radius inválido: {phase2_orient.spectral_radius!r}."
            )
        if not phase2_orient.gelfand_bounds:
            raise ValueError("Obs₂.gelfand_bounds no puede ser vacío.")
        logger.debug(
            "[BANACH:Φ₃←Φ₂] Ingestión de Obs₂ (verdict=%s, ρ=%.6e).",
            phase2_orient.verdict.name,
            phase2_orient.spectral_radius,
        )
        return phase2_orient

    # ── F.2  Inversión certificada por SVD ───────────────────────────────
    def _svd_invert_with_certificate(
        self,
        T: NDArray[np.float64],
    ) -> Tuple[Optional[NDArray[np.float64]], float, float, float, bool]:
        """
        Invierte T por SVD truncada a rango numérico pleno.

            T = U Σ Vᵀ ,   T⁻¹ = V Σ⁺ Uᵀ ,
            Σ⁺_{ii} = 1/σ_i  si σ_i > n u σ_max, else T se declara singular.

        Certificados devueltos:
          • rcond   = σ_min / σ_max,
          • κ₂(T)   = 1 / rcond,
          • residual ‖T T⁻¹ − I_n‖₂  (debería ser O(n u κ₂) si es invertible).

        No se usa `la.inv` a ciegas: una matriz mal condicionada produce
        una «inversa» de residual enorme que invalidaría Neumann.
        """
        n = int(T.shape[0])
        u, sigma, vh = la.svd(T, full_matrices=False, overwrite_a=False, check_finite=True)
        sigma = np.asarray(sigma, dtype=np.float64)
        smax = float(sigma[0]) if sigma.size else 0.0
        smin = float(sigma[-1]) if sigma.size else 0.0
        if smax <= 0.0:
            return None, 0.0, float("inf"), float("inf"), True
        rcond = smin / smax
        cond = 1.0 / rcond if rcond > 0.0 else float("inf")
        rank_tol = max(n, 1) * _MACHINE_EPS * smax
        if smin <= rank_tol:
            return None, rcond, cond, float("inf"), True

        t_inv = (vh.T * (1.0 / sigma)) @ u.T
        identity = np.eye(n, dtype=np.float64, order="C")
        residual = _operator_norm_2(T @ t_inv - identity)
        # Residual desorbitado ⇒ se trata como singularidad efectiva.
        residual_tol = max(n, 1) * _MACHINE_EPS * max(cond, 1.0) * 64.0
        if not np.isfinite(residual) or residual > max(residual_tol, self._hard_tol):
            logger.critical(
                "[BANACH:Φ₃] Residual de inversión excesivo: "
                "‖TT⁻¹−I‖₂=%.4e, κ₂=%.4e, umbral=%.4e.",
                residual,
                cond,
                residual_tol,
            )
            return None, rcond, cond, residual, True
        return t_inv, rcond, cond, residual, False

    # ── F.3  Operador de perturbación de Neumann ─────────────────────────
    def _build_neumann_operator(
        self,
        t_inv: NDArray[np.float64],
        dT: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float]:
        """
        Construye W = T⁻¹ dT y mide (‖W‖₂, ρ(W)).

        Identidad algebraica:
            T + dT  =  T (I + W) ,
        luego T + dT es invertible ⇔ I + W es invertible ⇔ −1 ∉ σ(W)
        y, en particular, si ρ(W) < 1 entonces la serie Σ (−W)ᵏ converge
        absolutamente en norma.
        """
        w_op = t_inv @ dT
        w_norm = _operator_norm_2(w_op)
        rho_w, _ = _spectral_radius_via_schur(w_op)
        return w_op, w_norm, rho_w

    # ── F.4  Cota de Kato–Rellich para la inversa perturbada ─────────────
    def _kato_rellich_inverse_bound(
        self,
        t_inv_norm: float,
        w_norm: float,
    ) -> float:
        """
        Si ‖W‖₂ < 1, la fórmula de Neumann da

            ‖(T + dT)⁻¹‖₂  ≤  ‖T⁻¹‖₂ / (1 − ‖W‖₂).

        Se devuelve esa cota; +∞ si ‖W‖₂ ≥ 1 (la hipótesis no aplica).
        """
        if not np.isfinite(w_norm) or w_norm >= 1.0 - self._spectral_margin:
            return float("inf")
        if not np.isfinite(t_inv_norm):
            return float("inf")
        return float(t_inv_norm / (1.0 - w_norm))

    # ── F.5  Clasificación de Neumann en el retículo ─────────────────────
    def _classify_neumann(
        self,
        singular_base: bool,
        rho_w: float,
        w_norm: float,
    ) -> Tuple[bool, bool, BanachHeytingVerdict]:
        """
        Estabilidad espectral: ρ(W) < 1 − margen.
        Suficiencia normativa : ‖W‖₂ < 1 − margen  ⇒  ρ(W) < 1.
        Singularidad de T     ⇒ veto inmediato.
        """
        finite_rho = bool(np.isfinite(rho_w))
        finite_norm = bool(np.isfinite(w_norm))
        threshold = 1.0 - self._spectral_margin
        is_stable = (not singular_base) and finite_rho and (rho_w < threshold)
        sufficient_by_norm = (
            (not singular_base) and finite_norm and (w_norm < threshold)
        )
        if singular_base or not is_stable:
            return is_stable, sufficient_by_norm, BanachHeytingVerdict.VETOED
        # Zona gris: estable espectralmente pero cerca del círculo unidad.
        proximity = 1.0 - rho_w
        if proximity < 10.0 * self._spectral_margin:
            return is_stable, sufficient_by_norm, BanachHeytingVerdict.DEGRADED
        return is_stable, sufficient_by_norm, BanachHeytingVerdict.COHERENT

    # ── F.6  Huella forense ──────────────────────────────────────────────
    def _forensic_provenance(
        self,
        phase2_orient: Phase2GelfandOrientation,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
        rho_w: float,
        w_norm: float,
        inversion_residual: float,
        final_verdict: BanachHeytingVerdict,
    ) -> str:
        """
        SHA-256 de invariantes + residuos de las tres fases.

        No se hashea la matriz cruda (coste y sensibilidad a bit-flips
        irrelevantes): se hashean invariantes algebraicos y los juicios.
        """
        phase1 = phase2_orient.phase1_observation
        t_inv = _matrix_invariants(T)
        dt_inv = _matrix_invariants(dT)
        payload = "|".join(
            (
                f"n={phase1.dimension}",
                _round_for_hash(phase1.submultiplicativity_residual),
                _round_for_hash(phase1.identity_residual),
                _round_for_hash(phase1.star_identity_residual_x),
                _round_for_hash(phase2_orient.gelfand_residual),
                _round_for_hash(phase2_orient.spectral_radius),
                _round_for_hash(phase2_orient.normality_defect),
                _round_for_hash(rho_w),
                _round_for_hash(w_norm),
                _round_for_hash(inversion_residual),
                f"T={t_inv}",
                f"dT={dt_inv}",
                f"V={int(final_verdict.value)}",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── F.7  SELLO TERMINAL DE LA FASE 3 ─────────────────────────────────
    def _evaluate_neumann(
        self,
        phase2_orient: Phase2GelfandOrientation,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
        raise_on_veto: bool = True,
    ) -> BanachGovernanceState:
        """
        Φ₃ — morfismo terminal de la Fase 3 y del funtor Z_Banach.

        Condición espectral (necesaria y suficiente sobre σ(W)):

            ρ(T⁻¹ dT)  <  1  −  margen_IEEE.

        Condición normativa (suficiente, Bauer–Fike / cota ρ ≤ ‖·‖):

            ‖T⁻¹ dT‖₂  <  1  −  margen_IEEE.

        Si el retículo colapsa a VETOED y `raise_on_veto`, se eleva
        HeytingLatticeVeto (con causa específica cuando es posible).
        """
        obs2 = self._ingest_phase2_precondition(phase2_orient)

        t_mat = _as_float64_square(T, "T")
        dt_mat = _as_float64_square(dT, "dT")
        _assert_conformal_pair(t_mat, dt_mat, "T", "dT")
        if int(t_mat.shape[0]) != obs2.phase1_observation.dimension:
            raise ValueError(
                f"T/dT tienen dimensión {t_mat.shape[0]}, "
                f"incompatible con Obs₁.n={obs2.phase1_observation.dimension}."
            )

        t_inv, rcond_t, cond_t, inv_residual, singular_base = (
            self._svd_invert_with_certificate(t_mat)
        )

        rho_w = float("inf")
        w_norm = float("inf")
        kato_bound = float("inf")
        if t_inv is not None and not singular_base:
            _, w_norm, rho_w = self._build_neumann_operator(t_inv, dt_mat)
            t_inv_norm = _operator_norm_2(t_inv)
            kato_bound = self._kato_rellich_inverse_bound(t_inv_norm, w_norm)
        else:
            singular_base = True
            logger.critical(
                "[BANACH:Φ₃] Operador base T espectralmente singular o "
                "numéricamente no invertible (rcond=%.4e, κ₂=%.4e).",
                rcond_t,
                cond_t,
            )

        is_stable, sufficient_by_norm, neumann_verdict = self._classify_neumann(
            singular_base=singular_base,
            rho_w=rho_w,
            w_norm=w_norm,
        )
        final_verdict = heyting_join(obs2.verdict, neumann_verdict)

        cause: Optional[BanachAlgebraError] = None
        if singular_base or not is_stable:
            cause = NeumannSeriesDivergence(
                f"Serie de Neumann no certificable: singular={singular_base}, "
                f"ρ(W)={rho_w:.6e}, ‖W‖₂={w_norm:.6e}."
            )
        if (
            not obs2.phase1_observation.is_submultiplicative
            and obs2.phase1_observation.verdict == BanachHeytingVerdict.VETOED
        ):
            cause = SubmultiplicativityViolation(
                "Submultiplicatividad vetada en Fase 1; el join terminal colapsa."
            )
        if (
            not obs2.phase1_observation.is_identity_coherent
            and obs2.phase1_observation.verdict == BanachHeytingVerdict.VETOED
        ):
            cause = IdentityDegeneracyError(
                "Univalencia de I_n vetada en Fase 1; el join terminal colapsa."
            )

        if final_verdict == BanachHeytingVerdict.VETOED:
            logger.critical(
                "[BANACH:Φ₃] VETO TERMINAL. ρ(W)=%.4e ‖W‖₂=%.4e singular=%s.",
                rho_w,
                w_norm,
                singular_base,
            )
            if raise_on_veto:
                raise HeytingLatticeVeto(
                    f"Colapso de Banach: veredicto supremo={final_verdict.name}. "
                    f"ρ(W)={rho_w:.6e} (se exige < {1.0 - self._spectral_margin:.6e}).",
                    verdict=final_verdict,
                    cause=cause,
                )
        elif final_verdict == BanachHeytingVerdict.DEGRADED:
            logger.warning(
                "[BANACH:Φ₃] Estado DEGRADED. ρ(W)=%.4e ‖W‖₂=%.4e.",
                rho_w,
                w_norm,
            )

        provenance = self._forensic_provenance(
            phase2_orient=obs2,
            T=t_mat,
            dT=dt_mat,
            rho_w=rho_w,
            w_norm=w_norm,
            inversion_residual=_finite_or(inv_residual, float("inf")),
            final_verdict=final_verdict,
        )

        phase1 = obs2.phase1_observation
        atoms: Tuple[str, ...] = (
            f"ρ(W)={rho_w:.6e}",
            f"‖W‖₂={w_norm:.6e}",
            f"rcond={rcond_t:.6e}",
            f"κ₂(T)={cond_t:.6e}",
            f"‖TT⁻¹−I‖₂={inv_residual:.6e}",
            f"Kato={kato_bound:.6e}",
            f"stable={is_stable}",
            f"suff_norm={sufficient_by_norm}",
            f"join={final_verdict.name}",
        )
        diagnostic_note = (
            f"Veredicto final: {final_verdict.name}. "
            f"Submultiplicatividad: {phase1.is_submultiplicative}. "
            f"Identidad coherente: {phase1.is_identity_coherent}. "
            f"C* coherente: {phase1.is_cstar_coherent}. "
            f"Consistencia Gelfand: {obs2.is_gelfand_consistent}. "
            f"Normalidad numérica: {obs2.is_numerically_normal}. "
            f"Neumann estable: {is_stable} "
            f"(ρ={rho_w:.4e}, ‖W‖={w_norm:.4e}, Kato≤{kato_bound:.4e}). "
            f"Huella: {provenance[:16]}…"
        )

        return BanachGovernanceState(
            phase2_orientation=obs2,
            neumann_radius=float(rho_w),
            neumann_operator_norm=float(w_norm),
            inversion_residual=_finite_or(inv_residual, float("inf")),
            rcond_t=float(rcond_t),
            condition_number_t=float(cond_t),
            kato_inverse_bound=float(kato_bound),
            is_neumann_stable=bool(is_stable),
            is_neumann_sufficient_by_norm=bool(sufficient_by_norm),
            singular_base=bool(singular_base),
            final_verdict=final_verdict,
            timestamp_utc=time.time(),
            provenance_hash=provenance,
            diagnostic_note=diagnostic_note,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §G. SOBERANO PRINCIPAL — COMPOSICIÓN FUNTORIAL Φ₃ ∘ Φ₂ ∘ Φ₁
# ══════════════════════════════════════════════════════════════════════════════
class BanachAlgebraAuditor(Phase3_NeumannStabilityDecider):
    """
    Soberano y guardián del análisis funcional computado.

    Garantiza invertibilidad perturbativa, coherencia de norma y
    consistencia espectral en el álgebra de Banach A = B(H_n).

    Composición funtorial estricta:

        Z_Banach(T, dT)  =  Φ₃( Φ₂( Φ₁(T, dT), T ), T, dT )

    Cada Φ_{k+1} *ingiere* el sello terminal de Φ_k; no hay atajos.
    En circuitos eléctricos, T modela el operador de malla/impedancia
    y dT una perturbación de componentes: la serie de Neumann es la
    convergencia del resolvente de la red perturbada.
    """

    def execute_banach_audit(
        self,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
        raise_on_veto: bool = True,
    ) -> BanachGovernanceState:
        """
        Ejecuta el ciclo completo de gobernanza en el topos de Banach.

        Parámetros
        ----------
        T : operador base (n, n). Debe ser numéricamente invertible
            para que la Fase 3 certifique Neumann.
        dT : perturbación incremental (n, n), conforme con T.
        raise_on_veto : si True, el colapso VETOED eleva HeytingLatticeVeto.

        Retorna
        -------
        BanachGovernanceState
            Certificado maestro, con Obs₂ ⊃ Obs₁ anidados y huella SHA-256.
        """
        t_mat = _as_float64_square(T, "T")
        dt_mat = _as_float64_square(dT, "dT")
        _assert_conformal_pair(t_mat, dt_mat, "T", "dT")

        # Fase 1 (Observe) — sello Obs₁, arranque formal de la Fase 2.
        phase1_obs = self._observe_norms(t_mat, dt_mat)

        # Fase 2 (Orient)  — sello Obs₂, arranque formal de la Fase 3.
        phase2_orient = self._orient_gelfand(phase1_obs, t_mat)

        # Fase 3 (Decide)  — certificado de gobernanza y veto terminal.
        governance_state = self._evaluate_neumann(
            phase2_orient=phase2_orient,
            T=t_mat,
            dT=dt_mat,
            raise_on_veto=raise_on_veto,
        )
        logger.info(
            "[BANACH:Z] Auditoría cerrada. verdict=%s hash=%s",
            governance_state.final_verdict.name,
            governance_state.provenance_hash[:16],
        )
        return governance_state

    def audit_batch(
        self,
        pairs: Iterable[Tuple[NDArray[np.float64], NDArray[np.float64]]],
        raise_on_veto: bool = False,
    ) -> Tuple[BanachGovernanceState, ...]:
        """
        Auditoría de una familia finita de pares (T, dT).

        Por defecto no eleva veto (raise_on_veto=False) para no abortar
        el lote: cada certificado porta su propio veredicto.
        """
        states = [
            self.execute_banach_audit(T=t_op, dT=dt_op, raise_on_veto=raise_on_veto)
            for t_op, dt_op in pairs
        ]
        return tuple(states)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "BanachHeytingVerdict",
    "heyting_join",
    "heyting_meet",
    "heyting_implies",
    "classify_relative_defect",
    "BanachAlgebraError",
    "SubmultiplicativityViolation",
    "IdentityDegeneracyError",
    "NeumannSeriesDivergence",
    "HeytingLatticeVeto",
    "Phase1NormObservation",
    "Phase2GelfandOrientation",
    "BanachGovernanceState",
    "Phase1_BanachNormObserver",
    "Phase2_GelfandSpectralOrienter",
    "Phase3_NeumannStabilityDecider",
    "BanachAlgebraAuditor",
]