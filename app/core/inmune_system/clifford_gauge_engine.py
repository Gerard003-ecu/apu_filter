# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Clifford Gauge Engine (Motor de Calibre de Clifford STA)             ║
║ Ruta   : app/core/inmune_system/clifford_gauge_engine.py                      ║
║ Versión: 4.0.0-Doctoral-STA-Vierbein-Hodge-Nested                             ║
║                                                                               ║
║ SINOPSIS MATEMÁTICA Y DE PRECISIÓN EN LA FPU:                                 ║
║ Motor algebraico ciego: opera tensores de Clifford-Minkowski en la FPU        ║
║ sin emitir vetos lógicos. La arquitectura es un funtor anidado                ║
║                                                                               ║
║     Quad(R^4, G)  --Fase 1-->  Marco(e, η, vol)                               ║
║                   --Fase 2-->  Cl(V, G) ⊂ M_4(C)                              ║
║                   --Fase 3-->  Ω^2(ad P) , S_YM , P_4                         ║
║                                                                               ║
║ Convención de signatura:                                                      ║
║   STA de Hestenes η = diag(+1, -1, -1, -1) ≅ Cℓ_{1,3}.                        ║
║   El germen Fase 1 → Fase 2 es el vierbein e_μ^a tal que                      ║
║                                                                               ║
║         G_{μν} = e_μ^a η_{ab} e_ν^b ,   γ_μ = e_μ^a γ_a.                      ║
║                                                                               ║
║   El germen Fase 2 → Fase 3 es el producto conmutador                         ║
║                                                                               ║
║         A × B := (AB − BA)/2 ,   F = dA + g A ∧ A.                            ║
║                                                                               ║
║ Implementa:                                                                   ║
║   Fase 1: simetrización, inercia de Sylvester, inversión espectral            ║
║           estable (SPD / indefinida / degenerada), vierbein, volumen.         ║
║   Fase 2: Dirac STA, inmersión R^16 ↪ M_4(C), Gram de traza,                  ║
║           reversión / involución / conjugación, productos geométrico,         ║
║           interior, exterior y conmutador, forma de Lorentz.                  ║
║   Fase 3: curvatura de gauge, Hodge de 2-formas, acción de Yang-Mills         ║
║           tensorial, densidad de Pontryagin, descomposición autodual.         ║
║                                                                               ║
║ Organización por herencia estricta:                                           ║
║   FASE 1: Phase1_MetricInquirer                                               ║
║   FASE 2: Phase2_CliffordArithmetic(Phase1_MetricInquirer)                    ║
║   FASE 3: Phase3_YangMillsAction(Phase2_CliffordArithmetic)                   ║
║   Motor : CliffordGaugeEngine(Phase3_YangMillsAction)                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Core.CliffordGaugeEngine")

__version__: Final[str] = "4.0.0"

# ───────────────────────────────────────────────────────────────────────────────
# Constantes de control numérico (IEEE-754 binary64)
# ───────────────────────────────────────────────────────────────────────────────

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1e-12
_ANTICOMMUTATOR_LIMIT: Final[float] = 1e-10
_SPACETIME_DIM: Final[int] = 4
_CLIFFORD_DIM: Final[int] = 16

# Signatura plana STA: η = diag(+1, −1, −1, −1).
_ETA_STA: Final[np.ndarray] = np.diag(
    np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
)
_ETA_STA.setflags(write=False)

# ───────────────────────────────────────────────────────────────────────────────
# Matrices de Pauli en M_2(C)
# ───────────────────────────────────────────────────────────────────────────────

_PAULI_1: Final[np.ndarray] = np.array(
    [[0.0, 1.0],
     [1.0, 0.0]],
    dtype=np.complex128,
)
_PAULI_2: Final[np.ndarray] = np.array(
    [[0.0, -1j],
     [1j, 0.0]],
    dtype=np.complex128,
)
_PAULI_3: Final[np.ndarray] = np.array(
    [[1.0, 0.0],
     [0.0, -1.0]],
    dtype=np.complex128,
)
_PAULI_1.setflags(write=False)
_PAULI_2.setflags(write=False)
_PAULI_3.setflags(write=False)

# ───────────────────────────────────────────────────────────────────────────────
# Pares bivectoriales en la base canónica STA
#
#   5  -> γ0γ1     6 -> γ0γ2     7 -> γ0γ3
#   8  -> γ1γ2     9 -> γ1γ3    10 -> γ2γ3
# ───────────────────────────────────────────────────────────────────────────────

_BIVECTOR_PAIRS: Final[Tuple[Tuple[int, int, int], ...]] = (
    (5, 0, 1),
    (6, 0, 2),
    (7, 0, 3),
    (8, 1, 2),
    (9, 1, 3),
    (10, 2, 3),
)


def _permutation_sign(perm: Tuple[int, ...]) -> float:
    inversions = 0
    for i, pi in enumerate(perm):
        for pj in perm[i + 1:]:
            if pi > pj:
                inversions += 1
    return 1.0 if inversions % 2 == 0 else -1.0


def _build_levi_civita_4() -> np.ndarray:
    r"""
    Símbolo de Levi-Civita ε_{μνρσ} en 4D, con ε_{0123} = +1.

    Es densidad tensorial de peso +1; no se sube/baja con G.
    """
    eps = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for perm in itertools.permutations(range(4)):
        eps[perm] = _permutation_sign(perm)
    eps.setflags(write=False)
    return eps


_LEVI_CIVITA_4: Final[np.ndarray] = _build_levi_civita_4()


# ───────────────────────────────────────────────────────────────────────────────
# Clasificación espectral de la forma cuadrática G
# ───────────────────────────────────────────────────────────────────────────────

class MetricSignature(str, Enum):
    r"""Inercia de Sylvester de G_{μν}, módulo degeneración numérica."""

    LORENTZIAN_STA = "lorentzian_+---"
    LORENTZIAN_EAST_COAST = "lorentzian_-+++"
    EUCLIDEAN = "euclidean_++++"
    NEGATIVE_DEFINITE = "negative_----"
    DEGENERATE = "degenerate"
    INDEFINITE_UNCLASSIFIED = "indefinite_unclassified"


# ───────────────────────────────────────────────────────────────────────────────
# Reportes de estado inmutables
# ───────────────────────────────────────────────────────────────────────────────

def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    r"""Copia C-contigua de solo lectura; el dataclass frozen no congela ndarrays."""
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class MetricInversionReport:
    r"""
    Estado espectral de la métrica de fondo G_{μν}.

    Atributos:
        g_matrix:            G simetrizada, 4×4 real.
        g_inv:               Inversa o pseudo-inversa de Moore-Penrose espectral.
        condition_number:    κ₂(G) = |λ|_max / |λ|_min sobre el soporte.
        cholesky_factor:     L si G ≻ 0; I en otro caso.
        is_spd:              Verdadero syss inercia = (4, 0, 0).
        bilateral_residual:  max(‖G G⁺ − I‖_F, ‖G⁺ G − I‖_F).
        eigenvalues:         Espectro real de G_sym (eigh).
        inertia:             (n₊, n₋, n₀).
        determinant:         det G (vía slogdet).
        is_degenerate:       n₀ > 0.
        eigenvectors:        Q en G = Q Λ Qᵀ.
        signature:           Clase de Sylvester.
        volume_density:      √|det G| (0 si degenerada).
        higham_residual:     ‖GG⁺−I‖_F / (4 ε ‖G‖_F ‖G⁺‖_F).
    """

    g_matrix: np.ndarray
    g_inv: np.ndarray
    condition_number: float
    cholesky_factor: np.ndarray
    is_spd: bool
    bilateral_residual: float

    eigenvalues: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    inertia: Tuple[int, int, int] = (0, 0, 0)
    determinant: float = 0.0
    is_degenerate: bool = False
    eigenvectors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    signature: MetricSignature = MetricSignature.INDEFINITE_UNCLASSIFIED
    volume_density: float = 0.0
    higham_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class OrthonormalFrameReport:
    r"""
    Germen métrico de la Fase 2: realización (V, G) ≅ marco ortonormal.

        G = e η eᵀ ,   vol = √|det G|.

    Si la inercia coincide con STA, η = η_STA y el funtor de Clifford
    admite la representación de Dirac real-equivariante.
    """

    metric: MetricInversionReport
    vierbein: np.ndarray
    vierbein_inv: np.ndarray
    eta: np.ndarray
    volume_density: float
    signature: MetricSignature
    reconstruction_residual: float
    sta_compatible: bool


@dataclass(frozen=True, slots=True)
class CliffordGeneratorReport:
    r"""
    Apertura de la Fase 2: generadores inducidos γ_μ = e_μ^a γ_a.

    La identidad de Clifford {γ_μ, γ_ν} = 2 G_{μν} I se certifica
    únicamente cuando sta_compatible es verdadero.
    """

    frame: OrthonormalFrameReport
    generators: Tuple[np.ndarray, ...]
    anticommutator_residual: float
    sta_compatible: bool


@dataclass(frozen=True, slots=True)
class CliffordArithmeticResult:
    r"""
    Resultado de un morfismo en el álgebra de Banach Cl_{1,3} ↪ (M_4(C), ‖·‖₂).

    Atributos:
        vector_rep:              Coeficientes en R^16.
        matrix_rep:              Imagen en M_4(C).
        quadratic_form:          Q(A) = ⟨A Ã⟩_0.
        is_rotor_unit:           Par y Q ≈ 1.
        sha256_hash:             Firma canónica little-endian.
        even_grade_norm:         ‖A_even‖₂.
        odd_grade_norm:          ‖A_odd‖₂.
        banach_norm:             Norma de Frobenius matricial / 2.
        spectral_radius:         ρ(M) = max |σ(M)|.
        reconstruction_residual: ‖π(ι(s)) − s‖₂.
    """

    vector_rep: np.ndarray
    matrix_rep: np.ndarray
    quadratic_form: float
    is_rotor_unit: bool
    sha256_hash: str

    even_grade_norm: float = 0.0
    odd_grade_norm: float = 0.0
    banach_norm: float = 0.0
    spectral_radius: float = 0.0
    reconstruction_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class YangMillsActionReport:
    r"""
    Acción puntual de Yang-Mills y diagnósticos topológicos.

        S_YM = (1/8) F_{μν} F^{μν}     (convención heredada, 2-forma real)
        P_4  = (1/4) ε_{μνρσ} F^{μν} F^{ρσ}

    Atributos adicionales:
        volume_density:         √|det G|.
        pontryagin_density:     Densidad algebraica F ∧ F.
        hodge_action:           (1/8) ⟨F, *F⟩_ε (diagnóstico).
        self_dual_norm:         ‖F⁺‖_F.
        anti_self_dual_norm:    ‖F⁻‖_F.
        tetrad_action_residual: |S_coord − S_tetrad|.
        signature:              Signatura de G.
    """

    curvature_bivector: np.ndarray
    curvature_matrix: np.ndarray
    ym_action: float
    is_action_finite: bool
    sha256_hash: str

    condition_number: float = 0.0
    frobenius_power: float = 0.0
    volume_density: float = 0.0
    pontryagin_density: float = 0.0
    hodge_action: float = 0.0
    self_dual_norm: float = 0.0
    anti_self_dual_norm: float = 0.0
    tetrad_action_residual: float = 0.0
    signature: MetricSignature = MetricSignature.INDEFINITE_UNCLASSIFIED


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: Ingesta, inversión espectral y marco ortonormal de G_{μν}           ║
# ║                                                                              ║
# ║ Objeto: espacio cuadrático real (R^4, G).                                    ║
# ║ Cierre formal: synthesize_orthonormal_frame  →  germen de la Fase 2.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase1_MetricInquirer:
    r"""
    FASE 1 — Geometría cuadrática de la métrica de fondo.

    Responsabilidades:
      1. Simetrización exacta de G.
      2. Diagnóstico espectral simétrico (eigh) e inercia de Sylvester.
      3. Clasificación: SPD, lorentziana STA, east-coast, degenerada.
      4. Inversión estable:
           · Cholesky si G ≻ 0,
           · inversión espectral si no degenerada,
           · pseudo-inversa espectral si degenerada.
      5. Residuo bilateral y residuo normalizado de Higham.
      6. Cierre: vierbein e tal que G = e η eᵀ  (germen de Cl(V, G)).
    """

    __slots__ = ("_tol",)

    def __init__(self, tolerance: float = _WILKINSON_LIMIT) -> None:
        self._tol: Final[float] = float(tolerance)

    def _relative_tolerance(self, scale: float) -> float:
        return max(self._tol, 4.0 * _MACHINE_EPS * max(1.0, float(scale)))

    @staticmethod
    def _classify_signature(
        positive_count: int,
        negative_count: int,
        zero_count: int,
    ) -> MetricSignature:
        if zero_count > 0:
            return MetricSignature.DEGENERATE
        if positive_count == 4 and negative_count == 0:
            return MetricSignature.EUCLIDEAN
        if positive_count == 0 and negative_count == 4:
            return MetricSignature.NEGATIVE_DEFINITE
        if positive_count == 1 and negative_count == 3:
            return MetricSignature.LORENTZIAN_STA
        if positive_count == 3 and negative_count == 1:
            return MetricSignature.LORENTZIAN_EAST_COAST
        return MetricSignature.INDEFINITE_UNCLASSIFIED

    @staticmethod
    def _volume_from_eigenvalues(
        eigvals: np.ndarray,
        eig_tol: float,
        is_degenerate: bool,
    ) -> float:
        if is_degenerate:
            return 0.0
        abs_e = np.abs(eigvals)
        if np.any(abs_e <= eig_tol):
            return 0.0
        try:
            return float(math.exp(0.5 * float(np.sum(np.log(abs_e)))))
        except OverflowError:
            return float("inf")

    @staticmethod
    def _inverse_from_eigendecomposition(
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        eig_tol: float,
    ) -> np.ndarray:
        r"""
        Pseudo-inversa espectral simétrica:

            G⁺ = Q diag(1/λ_i) Qᵀ ,   |λ_i| ≤ τ ↦ 0.
        """
        inv_vals = np.zeros_like(eigvals, dtype=np.float64)
        mask = np.abs(eigvals) > eig_tol
        inv_vals[mask] = 1.0 / eigvals[mask]
        g_inv = (eigvecs * inv_vals) @ eigvecs.T
        return 0.5 * (g_inv + g_inv.T)

    def evaluate_metric_tensor(self, G: np.ndarray) -> MetricInversionReport:
        r"""
        Audita G_{μν}: inversa estable, Cholesky condicionado, inercia,
        volumen, signatura y residuos de Higham.
        """
        G_arr = np.asarray(G, dtype=np.float64)

        if G_arr.shape != (_SPACETIME_DIM, _SPACETIME_DIM):
            raise ValueError(
                f"El tensor métrico G debe ser estrictamente 4×4. Obtenido: {G_arr.shape}"
            )
        if not np.all(np.isfinite(G_arr)):
            raise ValueError("El tensor métrico G contiene valores no finitos.")

        norm_g = float(la.norm(G_arr, "fro"))
        sym_error = float(la.norm(G_arr - G_arr.T, "fro")) / max(norm_g, 1.0)
        if sym_error > self._tol:
            logger.warning(
                "Asimetría residual detectada en G: %.3e. Se simetriza exactamente.",
                sym_error,
            )

        G_sym = 0.5 * (G_arr + G_arr.T)
        eigvals, eigvecs = la.eigh(G_sym, check_finite=False)

        abs_eig = np.abs(eigvals)
        max_abs = float(np.max(abs_eig)) if abs_eig.size else 0.0
        scale = max(1.0, norm_g, max_abs)
        eig_tol = self._relative_tolerance(scale)

        positive_count = int(np.sum(eigvals > eig_tol))
        negative_count = int(np.sum(eigvals < -eig_tol))
        zero_count = int(_SPACETIME_DIM - positive_count - negative_count)
        is_degenerate = zero_count > 0
        signature = self._classify_signature(
            positive_count, negative_count, zero_count
        )

        nonzero_abs = abs_eig[abs_eig > eig_tol]
        min_abs = float(np.min(nonzero_abs)) if nonzero_abs.size else 0.0
        condition_number = (
            float(max_abs / min_abs) if min_abs > 0.0 else float("inf")
        )

        is_spd = (
            positive_count == _SPACETIME_DIM
            and negative_count == 0
            and zero_count == 0
        )

        L = np.eye(_SPACETIME_DIM, dtype=np.float64)
        G_inv: Optional[np.ndarray] = None

        if is_spd:
            try:
                L = la.cholesky(G_sym, lower=True, check_finite=False)
                L_inv = la.solve_triangular(
                    L,
                    np.eye(_SPACETIME_DIM, dtype=np.float64),
                    lower=True,
                    check_finite=False,
                )
                G_inv = L_inv.T @ L_inv
            except la.LinAlgError:
                logger.warning(
                    "Cholesky falló pese a diagnóstico SPD. Cayendo a inversión espectral."
                )
                is_spd = False
                G_inv = None

        if G_inv is None:
            G_inv = self._inverse_from_eigendecomposition(eigvals, eigvecs, eig_tol)

        G_inv = 0.5 * (G_inv + G_inv.T)

        identity = np.eye(_SPACETIME_DIM, dtype=np.float64)
        residual_right = float(la.norm(G_sym @ G_inv - identity, "fro"))
        residual_left = float(la.norm(G_inv @ G_sym - identity, "fro"))
        bilateral_residual = max(residual_right, residual_left)

        norm_inv = float(la.norm(G_inv, "fro"))
        higham_denom = (
            float(_SPACETIME_DIM) * _MACHINE_EPS * max(norm_g, 1.0) * max(norm_inv, 1.0)
        )
        higham_residual = bilateral_residual / max(higham_denom, _MACHINE_EPS)

        sign, logdet = np.linalg.slogdet(G_sym)
        if sign == 0.0:
            determinant = 0.0
        else:
            try:
                determinant = float(sign * math.exp(logdet))
            except OverflowError:
                determinant = float("inf") if sign > 0 else float("-inf")

        volume_density = self._volume_from_eigenvalues(
            eigvals, eig_tol, is_degenerate
        )

        return MetricInversionReport(
            g_matrix=_immutable(G_sym, np.float64),
            g_inv=_immutable(G_inv, np.float64),
            condition_number=condition_number,
            cholesky_factor=_immutable(L, np.float64),
            is_spd=is_spd,
            bilateral_residual=bilateral_residual,
            eigenvalues=_immutable(eigvals, np.float64),
            inertia=(positive_count, negative_count, zero_count),
            determinant=determinant,
            is_degenerate=is_degenerate,
            eigenvectors=_immutable(eigvecs, np.float64),
            signature=signature,
            volume_density=volume_density,
            higham_residual=higham_residual,
        )

    def synthesize_orthonormal_frame(
        self,
        G: np.ndarray,
        metric_report: Optional[MetricInversionReport] = None,
    ) -> OrthonormalFrameReport:
        r"""
        CIERRE FORMAL DE LA FASE 1 / GERMEN DE LA FASE 2.

        Construye un vierbein real e_μ^a y una η diagonal de inercia tales que

            G_{μν} = e_μ^a η_{ab} e_ν^b.

        Algoritmo (ley de Sylvester + raíz espectral):
          1. G = Q Λ Qᵀ.
          2. Se reordenan ejes para coincidir con η_STA cuando la inercia es
             (1, 3): tiempo (λ > 0) primero, luego el bloque espacial.
          3. e = Q_ord diag(√|λ_ord|),  η = diag(sign(λ_ord)).
          4. e⁺ se obtiene por pseudo-inversa espectral de e.

        El funtor de Clifford de la Fase 2,
        ``induce_clifford_structure``, actúa exactamente sobre este germen:

            γ_μ = e_μ^a γ_a^{STA}  ⇒  {γ_μ, γ_ν} = 2 G_{μν} I

        si y sólo si signature = LORENTZIAN_STA.
        """
        report = (
            metric_report
            if metric_report is not None
            else self.evaluate_metric_tensor(G)
        )

        eigvals = np.array(report.eigenvalues, dtype=np.float64, copy=True)
        eigvecs = np.array(report.eigenvectors, dtype=np.float64, copy=True)

        if eigvals.shape != (_SPACETIME_DIM,) or eigvecs.shape != (
            _SPACETIME_DIM,
            _SPACETIME_DIM,
        ):
            report = self.evaluate_metric_tensor(report.g_matrix)
            eigvals = np.array(report.eigenvalues, dtype=np.float64, copy=True)
            eigvecs = np.array(report.eigenvectors, dtype=np.float64, copy=True)

        scale = max(1.0, float(la.norm(report.g_matrix, "fro")))
        eig_tol = self._relative_tolerance(scale)

        # Orden STA: un eje temporal (λ máximo) seguido de tres espaciales.
        order = np.argsort(-eigvals)
        eigvals_ord = eigvals[order]
        eigvecs_ord = eigvecs[:, order]

        signs = np.zeros(_SPACETIME_DIM, dtype=np.float64)
        sqrt_abs = np.zeros(_SPACETIME_DIM, dtype=np.float64)
        for i, lam in enumerate(eigvals_ord):
            if lam > eig_tol:
                signs[i] = 1.0
                sqrt_abs[i] = math.sqrt(lam)
            elif lam < -eig_tol:
                signs[i] = -1.0
                sqrt_abs[i] = math.sqrt(-lam)
            else:
                signs[i] = 0.0
                sqrt_abs[i] = 0.0

        eta = np.diag(signs)
        vierbein = eigvecs_ord * sqrt_abs

        inv_sqrt = np.zeros_like(sqrt_abs)
        mask = sqrt_abs > eig_tol
        inv_sqrt[mask] = 1.0 / sqrt_abs[mask]
        vierbein_inv = (eigvecs_ord * inv_sqrt).T

        reconstructed = vierbein @ eta @ vierbein.T
        reconstruction_residual = float(
            la.norm(reconstructed - report.g_matrix, "fro")
        )

        sta_compatible = (
            report.signature is MetricSignature.LORENTZIAN_STA
            and reconstruction_residual
            <= self._relative_tolerance(scale) * max(scale, 1.0)
        )

        if not sta_compatible and report.signature is MetricSignature.LORENTZIAN_STA:
            logger.warning(
                "Inercia STA pero el vierbein no reconstruye G (residuo %.3e).",
                reconstruction_residual,
            )

        return OrthonormalFrameReport(
            metric=report,
            vierbein=_immutable(vierbein, np.float64),
            vierbein_inv=_immutable(vierbein_inv, np.float64),
            eta=_immutable(eta, np.float64),
            volume_density=report.volume_density,
            signature=report.signature,
            reconstruction_residual=reconstruction_residual,
            sta_compatible=sta_compatible,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: Aritmética de Clifford STA sobre el germen de la Fase 1             ║
# ║                                                                              ║
# ║ Apertura: induce_clifford_structure(synthesize_orthonormal_frame(·)).        ║
# ║ Cierre formal: compute_commutator_product  →  germen de la Fase 3.           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_CliffordArithmetic(Phase1_MetricInquirer):
    r"""
    FASE 2 — Álgebra de Clifford STA como álgebra de Banach de dimensión 16.

    Continuación estricta de ``Phase1_MetricInquirer.synthesize_orthonormal_frame``.

    Responsabilidades:
      1. Inducir γ_μ = e_μ^a γ_a sobre el vierbein (apertura).
      2. Base de Dirac de 16 elementos y relaciones de anticonmutación.
      3. Inmersión ι: R^16 → M_4(C) y proyección dual π vía Gram de traza.
      4. Automorfismos: reversión, involución de grado, conjugación de Clifford.
      5. Productos geométrico, interior, exterior y conmutador.
      6. Forma cuadrática de Lorentz Q(A) = ⟨A Ã⟩_0.
      7. Cierre: A × B = (AB − BA)/2, germen de F = dA + g A∧A.
    """

    __slots__ = (
        "_basis_matrices",
        "_grades",
        "_reversion_signs",
        "_involution_signs",
        "_conjugation_signs",
        "_projection_gram_inv",
    )

    def __init__(self, tolerance: float = _WILKINSON_LIMIT) -> None:
        super().__init__(tolerance)
        self._basis_matrices: Final[Tuple[np.ndarray, ...]] = tuple(
            self._build_dirac_basis()
        )
        self._grades: Final[np.ndarray] = self._component_grades()
        self._reversion_signs: Final[np.ndarray] = self._grade_automorphism_signs(
            "reversion"
        )
        self._involution_signs: Final[np.ndarray] = self._grade_automorphism_signs(
            "involution"
        )
        self._conjugation_signs: Final[np.ndarray] = self._grade_automorphism_signs(
            "conjugation"
        )
        self._projection_gram_inv: Final[np.ndarray] = (
            self._build_projection_gram_inverse()
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Apertura formal: continuación del último método de la Fase 1
    # ───────────────────────────────────────────────────────────────────────────

    def induce_clifford_structure(
        self,
        G: np.ndarray,
        metric_report: Optional[MetricInversionReport] = None,
    ) -> CliffordGeneratorReport:
        r"""
        APERTURA FORMAL DE LA FASE 2.

        Continuación directa de ``synthesize_orthonormal_frame``:

            e, η, vol = synthesize_orthonormal_frame(G),
            γ_μ       = e_μ^a γ_a^{STA},
            {γ_μ, γ_ν} − 2 G_{μν} I  ≃  0.

        Si la signatura no es STA, no se induce representación de Dirac
        (Sylvester: Cl_{p,q} ≇ Cl_{1,3} como álgebras reales).
        """
        frame = self.synthesize_orthonormal_frame(G, metric_report=metric_report)

        if not frame.sta_compatible:
            logger.info(
                "Marco no STA-compatible (signatura=%s, residuo=%.3e). "
                "No se inducen generadores de Dirac.",
                frame.signature.value,
                frame.reconstruction_residual,
            )
            return CliffordGeneratorReport(
                frame=frame,
                generators=tuple(),
                anticommutator_residual=float("inf"),
                sta_compatible=False,
            )

        gammas_flat = self._basis_matrices[1:5]
        curved: List[np.ndarray] = []
        for mu in range(_SPACETIME_DIM):
            gamma_mu = np.zeros((4, 4), dtype=np.complex128)
            for a in range(_SPACETIME_DIM):
                coef = float(frame.vierbein[mu, a])
                if coef != 0.0:
                    gamma_mu = gamma_mu + coef * gammas_flat[a]
            gamma_mu.setflags(write=False)
            curved.append(gamma_mu)

        identity = np.eye(4, dtype=np.complex128)
        G_sym = frame.metric.g_matrix
        max_err = 0.0
        for mu in range(_SPACETIME_DIM):
            for nu in range(_SPACETIME_DIM):
                lhs = curved[mu] @ curved[nu] + curved[nu] @ curved[mu]
                rhs = 2.0 * G_sym[mu, nu] * identity
                max_err = max(max_err, float(la.norm(lhs - rhs, "fro")))

        if max_err > _ANTICOMMUTATOR_LIMIT:
            logger.warning(
                "Identidad de Clifford inducida violada: ‖{γμ,γν}−2Gμν I‖=%.3e",
                max_err,
            )

        return CliffordGeneratorReport(
            frame=frame,
            generators=tuple(curved),
            anticommutator_residual=max_err,
            sta_compatible=True,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Validación y utilidades numéricas
    # ───────────────────────────────────────────────────────────────────────────

    def _validate_multivector(
        self,
        S: np.ndarray,
        name: str = "multivector",
    ) -> np.ndarray:
        arr = np.asarray(S, dtype=np.float64)
        if arr.shape != (_CLIFFORD_DIM,):
            raise ValueError(
                f"El {name} debe ser estrictamente 16D. Obtenido: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El {name} contiene valores no finitos.")
        return arr

    @staticmethod
    def _kbn_sum(values: np.ndarray) -> float:
        r"""Sumación compensada Kahan–Babuška–Neumaier."""
        total = 0.0
        compensation = 0.0
        for value in values:
            val = float(value)
            if not math.isfinite(val):
                return val
            y = val - compensation
            t = total + y
            compensation = (t - total) - y
            total = t
        return total

    def _kbn_matvec(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.empty(A.shape[0], dtype=np.float64)
        for i in range(A.shape[0]):
            out[i] = self._kbn_sum(A[i, :] * x)
        return out

    @staticmethod
    def _canonical_bytes(array: np.ndarray) -> bytes:
        r"""Serialización little-endian independiente de la arquitectura."""
        arr = np.ascontiguousarray(array)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        header += np.array([1 if np.iscomplexobj(arr) else 0], dtype="<i8").tobytes()
        if np.iscomplexobj(arr):
            real = np.ascontiguousarray(arr.real, dtype=np.float64).astype("<f8")
            imag = np.ascontiguousarray(arr.imag, dtype=np.float64).astype("<f8")
            return header + real.tobytes() + imag.tobytes()
        real = np.ascontiguousarray(arr, dtype=np.float64).astype("<f8")
        return header + real.tobytes()

    def _sha256_arrays(self, *arrays: np.ndarray) -> str:
        sha = hashlib.sha256()
        for array in arrays:
            sha.update(self._canonical_bytes(array))
        return sha.hexdigest()

    # ───────────────────────────────────────────────────────────────────────────
    # Construcción de la base de Dirac y automorfismos de grado
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _component_grades() -> np.ndarray:
        r"""
        Grados de la base estándar:

            0      : escalar
            1..4   : vectores
            5..10  : bivectores
            11..14 : trivectores
            15     : pseudoescalar
        """
        grades = np.zeros(_CLIFFORD_DIM, dtype=np.int8)
        grades[1:5] = 1
        grades[5:11] = 2
        grades[11:15] = 3
        grades[15] = 4
        grades.setflags(write=False)
        return grades

    def _grade_automorphism_signs(self, kind: str) -> np.ndarray:
        r"""
        Signos de los tres anti-automorfismos fundamentales:

            reversión    \tilde A_k = (-1)^{k(k-1)/2} A_k
            involución   \hat A_k   = (-1)^k A_k
            conjugación  \bar A_k   = (-1)^{k(k+1)/2} A_k
        """
        signs = np.ones(_CLIFFORD_DIM, dtype=np.float64)
        for index, grade in enumerate(self._grades):
            k = int(grade)
            if kind == "reversion":
                exponent = k * (k - 1) // 2
            elif kind == "involution":
                exponent = k
            elif kind == "conjugation":
                exponent = k * (k + 1) // 2
            else:
                raise ValueError(f"Automorfismo de grado desconocido: {kind}")
            signs[index] = -1.0 if (exponent % 2 == 1) else 1.0
        signs.setflags(write=False)
        return signs

    @staticmethod
    def _verify_gamma_relations(gammas: Tuple[np.ndarray, ...]) -> None:
        r"""
        Verifica γ_μ γ_ν + γ_ν γ_μ = 2 η_{μν} I  con η = diag(+1, −1, −1, −1).
        """
        identity = np.eye(4, dtype=np.complex128)
        for mu in range(4):
            for nu in range(4):
                lhs = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
                rhs = 2.0 * _ETA_STA[mu, nu] * identity
                err = float(la.norm(lhs - rhs, "fro"))
                if err > _ANTICOMMUTATOR_LIMIT:
                    logger.warning(
                        "Relación de Clifford violada numéricamente en γ(%d,%d): %.3e",
                        mu,
                        nu,
                        err,
                    )

    def _build_dirac_basis(self) -> List[np.ndarray]:
        r"""
        Base canónica de 16 matrices de Dirac en M_4(C) para Cℓ_{1,3}.

            0      : I
            1..4   : γ_μ
            5..10  : γ_μ γ_ν  (μ < ν)
            11..14 : trivectores
            15     : γ5 = γ0 γ1 γ2 γ3
        """
        identity_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)

        gamma0 = np.block([
            [identity_2, zero_2],
            [zero_2, -identity_2],
        ])
        gamma1 = np.block([
            [zero_2, _PAULI_1],
            [-_PAULI_1, zero_2],
        ])
        gamma2 = np.block([
            [zero_2, _PAULI_2],
            [-_PAULI_2, zero_2],
        ])
        gamma3 = np.block([
            [zero_2, _PAULI_3],
            [-_PAULI_3, zero_2],
        ])

        gammas = (gamma0, gamma1, gamma2, gamma3)
        self._verify_gamma_relations(gammas)
        gamma5 = gamma0 @ gamma1 @ gamma2 @ gamma3

        basis: List[np.ndarray] = [
            np.eye(4, dtype=np.complex128),
            gamma0,
            gamma1,
            gamma2,
            gamma3,
            gamma0 @ gamma1,
            gamma0 @ gamma2,
            gamma0 @ gamma3,
            gamma1 @ gamma2,
            gamma1 @ gamma3,
            gamma2 @ gamma3,
            gamma0 @ gamma1 @ gamma2,
            gamma0 @ gamma1 @ gamma3,
            gamma0 @ gamma2 @ gamma3,
            gamma1 @ gamma2 @ gamma3,
            gamma5,
        ]
        for matrix in basis:
            matrix.setflags(write=False)
        return basis

    def _build_projection_gram_inverse(self) -> np.ndarray:
        r"""
        Inversa del Gram de proyección

            Γ_{kl} = (1/4) Re Tr(e_k^† e_l).

        Se invierte espectralmente para absorber deriva y bases no ortogonales.
        """
        n = _CLIFFORD_DIM
        gram = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            e_i_adj = self._basis_matrices[i].conj().T
            for j in range(i, n):
                value = 0.25 * float(
                    np.real(np.trace(e_i_adj @ self._basis_matrices[j]))
                )
                gram[i, j] = value
                gram[j, i] = value

        gram = 0.5 * (gram + gram.T)
        eigvals, eigvecs = la.eigh(gram, check_finite=False)
        max_w = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        tol = self._relative_tolerance(max_w)

        inv_w = np.zeros_like(eigvals, dtype=np.float64)
        mask = np.abs(eigvals) > tol
        inv_w[mask] = 1.0 / eigvals[mask]
        gram_inv = (eigvecs * inv_w) @ eigvecs.T
        gram_inv = 0.5 * (gram_inv + gram_inv.T)
        gram_inv.setflags(write=False)
        return gram_inv

    # ───────────────────────────────────────────────────────────────────────────
    # Inmersión ι : R^16 → M_4(C)  y  proyección dual π : M_4(C) → R^16
    # ───────────────────────────────────────────────────────────────────────────

    def embed_multivector(self, S: np.ndarray) -> np.ndarray:
        r"""
        Inmersión lineal ι(S) = Σ_k s_k E_k, con acumulación KBN por parte
        real e imaginaria.
        """
        S_arr = self._validate_multivector(S, "multivector a incrustar")

        sum_real = np.zeros((4, 4), dtype=np.float64)
        comp_real = np.zeros((4, 4), dtype=np.float64)
        sum_imag = np.zeros((4, 4), dtype=np.float64)
        comp_imag = np.zeros((4, 4), dtype=np.float64)

        for k, coef in enumerate(S_arr):
            if coef == 0.0:
                continue
            term = coef * self._basis_matrices[k]

            y_real = term.real - comp_real
            t_real = sum_real + y_real
            comp_real = (t_real - sum_real) - y_real
            sum_real = t_real

            y_imag = term.imag - comp_imag
            t_imag = sum_imag + y_imag
            comp_imag = (t_imag - sum_imag) - y_imag
            sum_imag = t_imag

        return sum_real + 1j * sum_imag

    def project_matrix(self, M: np.ndarray) -> np.ndarray:
        r"""
        Pullback π(M):  y_k = (1/4) Re Tr(E_k^† M),  s = Γ^{-1} y.
        """
        M_arr = np.asarray(M, dtype=np.complex128)
        if M_arr.shape != (4, 4):
            raise ValueError(
                f"La matriz a proyectar debe ser 4×4. Obtenido: {M_arr.shape}"
            )
        if not np.all(np.isfinite(M_arr)):
            raise ValueError("La matriz a proyectar contiene valores no finitos.")

        observations = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        for k in range(_CLIFFORD_DIM):
            basis_adj = self._basis_matrices[k].conj().T
            traced = np.trace(basis_adj @ M_arr)
            observations[k] = 0.25 * float(np.real(traced))

        coefficients = self._kbn_matvec(self._projection_gram_inv, observations)
        scale = max(1.0, float(la.norm(coefficients)))
        coefficients[np.abs(coefficients) < self._relative_tolerance(scale)] = 0.0
        return coefficients

    # ───────────────────────────────────────────────────────────────────────────
    # Automorfismos de Clifford y proyecciones de grado
    # ───────────────────────────────────────────────────────────────────────────

    def compute_reversion(self, S: np.ndarray) -> np.ndarray:
        r"""Reversión \(\tilde A_k = (-1)^{k(k-1)/2} A_k\)."""
        S_arr = self._validate_multivector(S, "multivector a revertir")
        return self._reversion_signs * S_arr

    def compute_grade_involution(self, S: np.ndarray) -> np.ndarray:
        r"""Involución de grado \(\hat A_k = (-1)^k A_k\)."""
        S_arr = self._validate_multivector(S, "multivector a involutionar")
        return self._involution_signs * S_arr

    def compute_clifford_conjugation(self, S: np.ndarray) -> np.ndarray:
        r"""Conjugación \(\bar A_k = (-1)^{k(k+1)/2} A_k = \widehat{\tilde A}\)."""
        S_arr = self._validate_multivector(S, "multivector a conjugar")
        return self._conjugation_signs * S_arr

    def compute_grade_projection(self, S: np.ndarray, grade: int) -> np.ndarray:
        if grade not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Grado inválido: {grade}. Debe ser 0,1,2,3,4.")
        S_arr = self._validate_multivector(S, "multivector a proyectar")
        out = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        mask = self._grades == grade
        out[mask] = S_arr[mask]
        return out

    # ───────────────────────────────────────────────────────────────────────────
    # Productos: geométrico, interior, exterior
    # ───────────────────────────────────────────────────────────────────────────

    def compute_geometric_product(
        self,
        A_vec: np.ndarray,
        B_vec: np.ndarray,
    ) -> np.ndarray:
        r"""Producto geométrico AB ↦ π(ι(A) ι(B))."""
        A = self._validate_multivector(A_vec, "multivector A")
        B = self._validate_multivector(B_vec, "multivector B")
        return self.project_matrix(self.embed_multivector(A) @ self.embed_multivector(B))

    def compute_outer_product(
        self,
        A_vec: np.ndarray,
        B_vec: np.ndarray,
    ) -> np.ndarray:
        r"""
        Producto exterior de Clifford:

            A ∧ B = Σ_{r,s} ⟨A_r B_s⟩_{r+s}.
        """
        A = self._validate_multivector(A_vec, "multivector A")
        B = self._validate_multivector(B_vec, "multivector B")
        out = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        for r in range(5):
            Ar = self.compute_grade_projection(A, r)
            if float(la.norm(Ar)) == 0.0:
                continue
            for s in range(5):
                grade = r + s
                if grade > 4:
                    continue
                Bs = self.compute_grade_projection(B, s)
                if float(la.norm(Bs)) == 0.0:
                    continue
                prod = self.compute_geometric_product(Ar, Bs)
                out += self.compute_grade_projection(prod, grade)
        return out

    def compute_inner_product(
        self,
        A_vec: np.ndarray,
        B_vec: np.ndarray,
    ) -> np.ndarray:
        r"""
        Contracción (producto interior):

            A · B = Σ_{r,s} ⟨A_r B_s⟩_{|r−s|}.
        """
        A = self._validate_multivector(A_vec, "multivector A")
        B = self._validate_multivector(B_vec, "multivector B")
        out = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        for r in range(5):
            Ar = self.compute_grade_projection(A, r)
            if float(la.norm(Ar)) == 0.0:
                continue
            for s in range(5):
                Bs = self.compute_grade_projection(B, s)
                if float(la.norm(Bs)) == 0.0:
                    continue
                prod = self.compute_geometric_product(Ar, Bs)
                out += self.compute_grade_projection(prod, abs(r - s))
        return out

    def compute_clifford_dual(self, S: np.ndarray) -> np.ndarray:
        r"""
        Dual de Clifford por la derecha: A ↦ A I,  I = γ0γ1γ2γ3 (índice 15).

        En STA, I² = −1 y el dual relaciona k-formas con (4−k)-formas.
        """
        S_arr = self._validate_multivector(S, "multivector a dualizar")
        pseudo = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        pseudo[15] = 1.0
        return self.compute_geometric_product(S_arr, pseudo)

    def compute_lorentz_quadratic_form(self, S: np.ndarray) -> float:
        r"""
        Forma cuadrática de Lorentz Q(A) = ⟨A Ã⟩_0.

        Sobre vectores: Q(x) = x₀² − x₁² − x₂² − x₃².
        """
        S_arr = self._validate_multivector(S, "multivector de forma cuadrática")
        M = self.embed_multivector(S_arr)
        M_rev = self.embed_multivector(self.compute_reversion(S_arr))
        scalar = float(self.project_matrix(M @ M_rev)[0])
        if abs(scalar) < self._tol:
            scalar = 0.0
        return scalar

    def _pack_arithmetic_result(
        self,
        S_prod: np.ndarray,
        M_prod: np.ndarray,
    ) -> CliffordArithmeticResult:
        quadratic_form = self.compute_lorentz_quadratic_form(S_prod)

        odd_mask = (self._grades % 2) == 1
        even_mask = ~odd_mask
        odd_grade_norm = float(la.norm(S_prod[odd_mask]))
        even_grade_norm = float(la.norm(S_prod[even_mask]))

        banach_norm = float(la.norm(M_prod, "fro")) / 2.0
        spectrum = la.eigvals(M_prod, check_finite=False)
        spectral_radius = float(np.max(np.abs(spectrum))) if spectrum.size else 0.0

        roundtrip = self.project_matrix(self.embed_multivector(S_prod))
        reconstruction_residual = float(la.norm(roundtrip - S_prod))

        rotor_tolerance = max(self._tol, 10.0 * _MACHINE_EPS)
        is_rotor_unit = bool(
            (odd_grade_norm <= rotor_tolerance)
            and (abs(quadratic_form - 1.0) <= rotor_tolerance)
        )

        sha256_hash = self._sha256_arrays(
            S_prod,
            np.real(M_prod),
            np.imag(M_prod),
            np.array(
                [quadratic_form, odd_grade_norm, even_grade_norm, banach_norm],
                dtype=np.float64,
            ),
        )

        return CliffordArithmeticResult(
            vector_rep=_immutable(S_prod, np.float64),
            matrix_rep=_immutable(M_prod, np.complex128),
            quadratic_form=quadratic_form,
            is_rotor_unit=is_rotor_unit,
            sha256_hash=sha256_hash,
            even_grade_norm=even_grade_norm,
            odd_grade_norm=odd_grade_norm,
            banach_norm=banach_norm,
            spectral_radius=spectral_radius,
            reconstruction_residual=reconstruction_residual,
        )

    def multiply_multivectors(
        self,
        A_vec: np.ndarray,
        B_vec: np.ndarray,
    ) -> CliffordArithmeticResult:
        r"""
        Producto geométrico con diagnóstico de rotor unitario, norma de Banach
        y radio espectral de la representación de Dirac.
        """
        A = self._validate_multivector(A_vec, "multivector A")
        B = self._validate_multivector(B_vec, "multivector B")
        M_prod = self.embed_multivector(A) @ self.embed_multivector(B)
        S_prod = self.project_matrix(M_prod)
        return self._pack_arithmetic_result(S_prod, M_prod)

    def compute_commutator_product(
        self,
        A_vec: np.ndarray,
        B_vec: np.ndarray,
    ) -> CliffordArithmeticResult:
        r"""
        CIERRE FORMAL DE LA FASE 2 / GERMEN DE LA FASE 3.

        Producto conmutador de Clifford (producto cruzado de Hestenes):

            A × B := (AB − BA) / 2.

        Sobre 1-formas con valores en el álgebra interna, este morfismo es
        exactamente el término no abeliano de la curvatura de gauge:

            F = dA + g A ∧ A ,   (A ∧ A)_{μν} = A_μ × A_ν.

        ``Phase3_YangMillsAction.compute_curvature_field`` consume este
        método de forma estricta: no reimplementa el conmutador.
        """
        A = self._validate_multivector(A_vec, "multivector A")
        B = self._validate_multivector(B_vec, "multivector B")
        M_A = self.embed_multivector(A)
        M_B = self.embed_multivector(B)
        M_comm = 0.5 * (M_A @ M_B - M_B @ M_A)
        S_comm = self.project_matrix(M_comm)
        return self._pack_arithmetic_result(S_comm, M_comm)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: Curvatura Faraday, Hodge y acción de Yang-Mills                     ║
# ║                                                                              ║
# ║ Apertura: compute_curvature_field usa compute_commutator_product.            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_YangMillsAction(Phase2_CliffordArithmetic):
    r"""
    FASE 3 — Curvatura de gauge, operador de Hodge y acción de Yang-Mills.

    Continuación estricta de ``compute_commutator_product``:

        F_{μν} = (dA)_{μν} + g (A_μ × A_ν)|_{γ_μ∧γ_ν}.

    Responsabilidades:
      1. Validar curvaturas 16D o bivectoriales 6D.
      2. Calcular F = dA + g A∧A con el producto conmutador de la Fase 2.
      3. Contracción métrica S_YM = (1/8) F_{μν} F^{μν}.
      4. Hodge de 2-formas, densidad de Pontryagin y partes autoduales.
      5. Residuo tetrada vs. coordenadas (invarianza de marco).
    """

    __slots__ = ()

    def _validate_curvature(
        self,
        F: np.ndarray,
        name: str = "curvatura",
    ) -> np.ndarray:
        r"""
        Acepta R^16 o el chart compacto 6D [F01, F02, F03, F12, F13, F23].
        """
        arr = np.asarray(F, dtype=np.float64)
        if arr.shape == (6,):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"La {name} compacta 6D contiene valores no finitos.")
            full = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
            full[5:11] = arr
            return full
        return self._validate_multivector(arr, name)

    @staticmethod
    def _two_form_from_multivector(F16: np.ndarray) -> np.ndarray:
        F = np.zeros((_SPACETIME_DIM, _SPACETIME_DIM), dtype=np.float64)
        for bivector_index, mu, nu in _BIVECTOR_PAIRS:
            value = float(F16[bivector_index])
            F[mu, nu] = value
            F[nu, mu] = -value
        return F

    @staticmethod
    def _hodge_star_two_form(
        F_contravariant: np.ndarray,
        volume_density: float,
    ) -> np.ndarray:
        r"""
        (*F)_{μν} = (1/2) √|g| ε_{μνρσ} F^{ρσ}.

        Si vol = 0 (métrica degenerada) el Hodge no está definido: se
        devuelve NaN.
        """
        if not math.isfinite(volume_density) or volume_density <= 0.0:
            return np.full((_SPACETIME_DIM, _SPACETIME_DIM), np.nan, dtype=np.float64)
        dual = 0.5 * volume_density * np.einsum(
            "mnrs,rs->mn",
            _LEVI_CIVITA_4,
            F_contravariant,
        )
        return 0.5 * (dual - dual.T)

    def compute_curvature_field(
        self,
        potential_A: Optional[np.ndarray],
        field_dA: np.ndarray,
        coupling_g: float = 1.0,
    ) -> np.ndarray:
        r"""
        APERTURA FORMAL DE LA FASE 3.

        Campo de curvatura covariante

            F = dA + g A ∧ A

        usando, en el sector no abeliano, el cierre de la Fase 2:

            (A ∧ A)_{μν} = A_μ × A_ν = compute_commutator_product(A_μ, A_ν).

        Modos:
          1. potential_A is None o g = 0  →  F = ⟨dA⟩_2.
          2. shape (16,) o (6,)           →  abeliano, A∧A = 0 para 1-formas.
          3. shape (4, 16) o (4, 6)       →  A_μ no abeliano.
        """
        dA_full = self._validate_curvature(field_dA, "field_dA")
        F = self.compute_grade_projection(dA_full, grade=2)

        g = float(coupling_g)
        if not math.isfinite(g):
            raise ValueError("El acoplamiento coupling_g debe ser finito.")
        if potential_A is None or g == 0.0:
            return F

        A = np.asarray(potential_A, dtype=np.float64)

        if A.ndim == 1 and A.shape == (6,):
            tmp = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
            tmp[5:11] = A
            A = tmp

        if A.ndim == 2 and A.shape == (4, 6):
            tmp = np.zeros((4, _CLIFFORD_DIM), dtype=np.float64)
            tmp[:, 5:11] = A
            A = tmp

        if A.ndim == 1:
            A_full = self._validate_multivector(A, "potential_A")
            non_vector_norm = float(
                la.norm(A_full[np.where(self._grades != 1)[0]])
            )
            if non_vector_norm > self._tol:
                logger.info(
                    "potential_A 1D contiene grados no vectoriales. "
                    "El modo 1D se interpreta como abeliano y no añade A∧A."
                )
            return F

        if A.ndim == 2 and A.shape == (4, _CLIFFORD_DIM):
            for mu in range(_SPACETIME_DIM):
                self._validate_multivector(A[mu], f"potential_A[{mu}]")

            for bivector_index, mu, nu in _BIVECTOR_PAIRS:
                # Germen Fase 2 → Fase 3: A_μ × A_ν.
                comm = self.compute_commutator_product(A[mu], A[nu])
                F[bivector_index] += g * float(comm.vector_rep[bivector_index])

            scale = max(1.0, float(la.norm(F)))
            F[np.abs(F) < self._relative_tolerance(scale)] = 0.0
            return F

        raise ValueError(
            "potential_A debe ser None, shape (16,), shape (6,), "
            "shape (4,16) o shape (4,6)."
        )

    def evaluate_yang_mills_action(
        self,
        F_vec: np.ndarray,
        G_metric: np.ndarray,
        frame: Optional[OrthonormalFrameReport] = None,
    ) -> YangMillsActionReport:
        r"""
        Acción covariante puntual de Yang-Mills

            S_YM = (1/8) F_{μν} F^{μν} ,   F^{μν} = G^{μα} G^{νβ} F_{αβ},

        más diagnósticos:
          · densidad de Pontryagin P₄ = (1/4) ε_{μνρσ} F^{μν} F^{ρσ},
          · acción de Hodge (1/8) F_{μν} (*F)^{μν},
          · normas autodual / anti-autodual,
          · residuo de invariancia tetrada.
        """
        F = self._validate_curvature(F_vec, "F_vec")
        G_arr = np.asarray(G_metric, dtype=np.float64)

        if G_arr.shape != (_SPACETIME_DIM, _SPACETIME_DIM):
            raise ValueError(
                f"La métrica G debe ser 4×4. Obtenido: {G_arr.shape}"
            )
        if not np.all(np.isfinite(G_arr)):
            raise ValueError("La métrica G contiene valores no finitos.")

        if frame is not None:
            metric_report = frame.metric
        else:
            metric_report = self.evaluate_metric_tensor(G_arr)
            frame = self.synthesize_orthonormal_frame(
                metric_report.g_matrix, metric_report=metric_report
            )

        if metric_report.is_degenerate:
            logger.warning(
                "Métrica de fondo degenerada. La acción de Yang-Mills usa "
                "pseudo-inversa y puede perder coercitividad."
            )
        elif not metric_report.is_spd:
            logger.info(
                "Métrica de fondo indefinida. Compatible con signatura "
                "Minkowski; la acción puede no ser positiva."
            )

        F_clean = self.compute_grade_projection(F, grade=2)
        F_matrix = self.embed_multivector(F_clean)
        F_covariant = self._two_form_from_multivector(F_clean)

        G_inv = metric_report.g_inv
        F_contravariant = G_inv @ F_covariant @ G_inv.T
        F_contravariant = 0.5 * (F_contravariant - F_contravariant.T)

        metric_contraction = float(
            np.einsum("mn,mn->", F_covariant, F_contravariant)
        )
        ym_action = 0.125 * metric_contraction

        frobenius_power = float(
            np.real(np.trace(F_matrix.conj().T @ F_matrix)) / 4.0
        )

        pontryagin_density = 0.25 * float(
            np.einsum(
                "mnrs,mn,rs->",
                _LEVI_CIVITA_4,
                F_contravariant,
                F_contravariant,
            )
        )

        volume_density = metric_report.volume_density
        F_hodge = self._hodge_star_two_form(F_contravariant, volume_density)

        if np.all(np.isfinite(F_hodge)):
            # (*F)^{μν} = G^{μα} G^{νβ} (*F)_{αβ}
            F_hodge_contra = G_inv @ F_hodge @ G_inv.T
            hodge_action = 0.125 * float(
                np.einsum("mn,mn->", F_covariant, F_hodge_contra)
            )
            if metric_report.signature is MetricSignature.EUCLIDEAN:
                F_plus = F_covariant + F_hodge
                F_minus = F_covariant - F_hodge
            else:
                # En Lorentz, *² = −Id sobre 2-formas: autodualidad compleja.
                F_plus = F_covariant + 1j * F_hodge
                F_minus = F_covariant - 1j * F_hodge
            self_dual_norm = float(la.norm(F_plus, "fro"))
            anti_self_dual_norm = float(la.norm(F_minus, "fro"))
        else:
            hodge_action = float("nan")
            self_dual_norm = float("nan")
            anti_self_dual_norm = float("nan")

        tetrad_action_residual = 0.0
        if frame is not None and frame.sta_compatible:
            e_inv = frame.vierbein_inv
            F_ortho = e_inv @ F_covariant @ e_inv.T
            F_ortho = 0.5 * (F_ortho - F_ortho.T)
            eta_inv = np.diag(np.diag(frame.eta))
            # η = η^{-1} sobre {±1}; ceros se conservan en el caso degenerado.
            F_ortho_contra = eta_inv @ F_ortho @ eta_inv.T
            tetrad_contraction = float(
                np.einsum("mn,mn->", F_ortho, F_ortho_contra)
            )
            tetrad_action_residual = abs(0.125 * tetrad_contraction - ym_action)

        is_action_finite = bool(
            math.isfinite(ym_action) and math.isfinite(frobenius_power)
        )

        sha256_hash = self._sha256_arrays(
            F_clean,
            metric_report.g_matrix,
            metric_report.g_inv,
            np.array(
                [
                    ym_action,
                    metric_contraction,
                    frobenius_power,
                    pontryagin_density,
                ],
                dtype=np.float64,
            ),
        )

        return YangMillsActionReport(
            curvature_bivector=_immutable(F_clean[5:11], np.float64),
            curvature_matrix=_immutable(F_matrix, np.complex128),
            ym_action=ym_action,
            is_action_finite=is_action_finite,
            sha256_hash=sha256_hash,
            condition_number=metric_report.condition_number,
            frobenius_power=frobenius_power,
            volume_density=volume_density,
            pontryagin_density=pontryagin_density,
            hodge_action=hodge_action,
            self_dual_norm=self_dual_norm,
            anti_self_dual_norm=anti_self_dual_norm,
            tetrad_action_residual=tetrad_action_residual,
            signature=metric_report.signature,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Motor final: CliffordGaugeEngine                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CliffordGaugeEngine(Phase3_YangMillsAction):
    r"""
    Motor Imperial Espectral de Calibre de Clifford.

    Consagra la FPU ciega de-confinada de Spacetime Algebra.

    Cadena de herencia (fases anidadas):

        CliffordGaugeEngine
          └─ Phase3_YangMillsAction          F = dA + g A×A , S_YM , *F
               └─ Phase2_CliffordArithmetic  Cl(V,G) ↪ M_4(C)
                    └─ Phase1_MetricInquirer (R^4, G) → (e, η, vol)

    Pipeline:
        G → marco ortonormal → generadores → curvatura → acción.
    """

    __slots__ = ()

    def run_clifford_gauge_pipeline(
        self,
        G_metric: np.ndarray,
        potential_A: Optional[np.ndarray],
        field_dA: np.ndarray,
        coupling_g: float = 1.0,
    ) -> Tuple[MetricInversionReport, YangMillsActionReport]:
        r"""
        Pipeline completo de las tres fases anidadas:

            1. Evalúa G y sintetiza el vierbein (cierre Fase 1).
            2. Induce γ_μ si la signatura es STA (apertura Fase 2).
            3. Calcula F vía A×B (cierre Fase 2 / apertura Fase 3).
            4. Evalúa S_YM, Hodge y Pontryagin sobre el mismo marco.
        """
        metric_report = self.evaluate_metric_tensor(G_metric)
        frame = self.synthesize_orthonormal_frame(
            metric_report.g_matrix, metric_report=metric_report
        )
        self.induce_clifford_structure(
            metric_report.g_matrix, metric_report=metric_report
        )

        curvature = self.compute_curvature_field(
            potential_A=potential_A,
            field_dA=field_dA,
            coupling_g=coupling_g,
        )

        action_report = self.evaluate_yang_mills_action(
            F_vec=curvature,
            G_metric=metric_report.g_matrix,
            frame=frame,
        )
        return metric_report, action_report


__all__ = [
    "CliffordGaugeEngine",
    "MetricInversionReport",
    "OrthonormalFrameReport",
    "CliffordGeneratorReport",
    "CliffordArithmeticResult",
    "YangMillsActionReport",
    "MetricSignature",
    "Phase1_MetricInquirer",
    "Phase2_CliffordArithmetic",
    "Phase3_YangMillsAction",
]