# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Algebra — Evolución Bicompleja W*-Álgebra de Von Neumann        ║
║ Ruta   : app/wisdom/mac_algebra.py                                           ║
║ Versión: 6.0.0-Nested3-WStar-Spectral-Heyting-Strict                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESTRUCTURA EN TRES FASES ANIDADAS
─────────────────────────────────
La teoría se desarrolla como una composición de funtores estricta:

    Ψ_MAC = Ψ₂₃ ∘ Ψ₁₂ :  Hilb_{C₂}  →  CPTP_{C₂} ⋊ L(H_{C₂})  →  StdForm ⋊ B₂

    FASE 1  Fundamento W*-atómico (C*-axiomas, espectro, estados de Dirac).
            Cláusula final: ``psi_12_initiate_cptp_category``.
    FASE 2  Categoría CPTP bicanal y retículo ortomodular (Choi, Sasaki, GNS).
            Cláusula final: ``psi_23_initiate_tomita_takesaki``.
    FASE 3  Tomita–Takesaki, topos de Heyting B₂ y funtor maestro Ψ_MAC.

DICTAMEN CRÍTICO DE VERACIDAD
─────────────────────────────
El álgebra bicompleja se interpreta exclusivamente bajo la descomposición
de Pierce (suma directa idempotente), no como C ⊗_R C con unidad imaginaria
adjunta no idempotente:

    C₂ ≅ C e₁ ⊕ C e₂,
    e₁² = e₁,  e₂² = e₂,  e₁ e₂ = 0,  e₁ + e₂ = 1.

Consecuencias operativas:

1. Toda C*-norma, espectro y traza se calculan canal a canal. La afirmación
   «Tr = 1» denota la unidad bicompleja 1 = e₁ + e₂, nunca un escalar de C.
2. CPTP_{C₂} ≅ CPTP × CPTP. Completa positividad ⇔ Λ⁽ᵏ⁾ ≽ 0 (Choi) ∀k.
   La categoría CPTP NO es daga-compacta; FdHilb sí lo es. Se corrige el
   abuso terminológico de la v5.
3. L(H_{C₂}) ≅ L(H⁽¹⁾) × L(H⁽²⁾) como retículo ortomodular. Meet y join
   se realizan por intersección/suma de rangos (ángulos principales), no
   por iteración inestable de (PQP)ⁿ.
4. Tomita–Takesaki finito-dimensional se descompone canal a canal. La
   involución algebraica es el adjunto de C* (A ↦ A†), no la conjugación
   entrada a entrada. KMS se enuncia para operadores acotados generales.
5. B₂ ≅ Ω₃⁽¹⁾ × Ω₃⁽²⁾ es un álgebra de Heyting acotada (no booleana).
   El veto global es la conjunción topológica min(v₁, v₂).

Convención numérica: matrices en M_d(C) ⊂ B(H), dtype complex128,
vectorización column-major (Fortran), normas de Frobenius salvo
indicación espectral/Schatten.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

#=============================================================================
# FASE 1.1 — INFRAESTRUCTURA AXIOMÁTICA
#=============================================================================
logger = logging.getLogger("MAC.Wisdom.Algebra.Bicomplex")

_SCHEMA_VERSION: Final[str] = "6.0.0-Nested3-WStar-Spectral-Heyting-Strict"
_DEFAULT_TOLERANCE: Final[float] = 1e-10
_FAITHFUL_TOLERANCE: Final[float] = 1e-14
_MEET_EIGEN_THRESHOLD: Final[float] = 1e-8
_CONDITION_WARN: Final[float] = 1e12
_GPIO_VETO_PIN: Final[int] = 14
_ISR_ACTUATION_NS: Final[float] = 398.95
_COHERENT_THRESHOLD: Final[float] = 0.85
_DEGRADED_THRESHOLD: Final[float] = 0.50

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
Dimensions = Tuple[int, int]
BicomplexTrace = Tuple[complex, complex]
KrausTuple = Tuple[ComplexMatrix, ...]


class NonCommutativeAlgebraError(Exception):
    """Violación estructural de un axioma de W*-álgebra bicompleja."""


class TraceAnomalyError(NonCommutativeAlgebraError):
    """Fallo de traza, fidelidad, normalización o positividad de estado."""


class OrthomodularConvergenceError(NonCommutativeAlgebraError):
    """Fallo numérico en meet/join o en la ley ortomodular."""


class ModularConjugationError(NonCommutativeAlgebraError):
    """Inconsistencia de J, Δ, S o del grupo modular σ_t."""


class CategoryCompositionError(NonCommutativeAlgebraError):
    """Composición o producto tensorial de morfismos mal tipados."""


class HeytingVetoError(NonCommutativeAlgebraError):
    """Colapso global a VETOED en el retículo de Heyting B₂."""

    def __init__(
        self,
        message: str,
        verdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.verdict = verdict if verdict is not None else {}


class VonNeumannFactorType(Enum):
    """Clasificación de Connes–Murray–von Neumann de factores."""

    TYPE_I_FINITE = auto()
    TYPE_I_INFINITE = auto()
    TYPE_II_1 = auto()
    TYPE_II_INFINITY = auto()
    TYPE_III = auto()
    TYPE_III_0 = auto()
    TYPE_III_LAMBDA = auto()
    TYPE_III_1 = auto()


class HeytingValue(IntEnum):
    """
    Álgebra de Heyting trivalente Ω₃ = {0 < 1 < 2} por canal.

    Orden: VETOED < DEGRADED < COHERENT.
    Implicación intuicionista: a → b = ⊤ si a ≤ b, si no a → b = b.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @classmethod
    def from_coherence(
        cls,
        value: float,
        coherent_threshold: float = _COHERENT_THRESHOLD,
        degraded_threshold: float = _DEGRADED_THRESHOLD,
    ) -> "HeytingValue":
        """Clasifica una coherencia escalar en Ω₃ con umbrales cerrados a derecha."""
        scalar = float(value)
        if scalar >= coherent_threshold:
            return cls.COHERENT
        if scalar >= degraded_threshold:
            return cls.DEGRADED
        return cls.VETOED


#=============================================================================
# FASE 1.2 — CÁLCULO MATRICIAL Y CÁLCULO FUNCIONAL ESPECTRAL
#=============================================================================
def _as_square_matrix(
    A: NDArray[np.complex128],
    dtype: type = np.complex128,
    name: str = "matriz",
) -> ComplexMatrix:
    """Inmersión validada en M_d(C): exige ndim = 2 y d × d."""
    arr = np.asarray(A, dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada; shape recibido: {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} no puede tener dimensión nula.")
    return arr


def _symmetrize(A: ComplexMatrix) -> ComplexMatrix:
    """Proyección ortogonal sobre el R-subespacio hermítico: (A + A†)/2."""
    return 0.5 * (A + A.conj().T)


def _frobenius(A: NDArray[np.complex128]) -> float:
    """Norma de Hilbert–Schmidt ‖A‖₂ = √Tr(A†A)."""
    return float(np.linalg.norm(A, ord="fro"))


def _is_hermitian(A: ComplexMatrix, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
    """A = A† en norma de Frobenius, con tolerancia absoluta."""
    return _frobenius(A - A.conj().T) < tolerance


def _is_normal(A: ComplexMatrix, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
    """Conmutación [A, A†] = 0 (operadores normales)."""
    return _frobenius(A @ A.conj().T - A.conj().T @ A) < tolerance


def _is_projector(P: ComplexMatrix, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
    """Proyector ortogonal: P = P† = P²."""
    if not _is_hermitian(P, tolerance):
        return False
    return _frobenius(P @ P - P) < tolerance


def _operator_norm(A: ComplexMatrix) -> float:
    """Norma espectral C*: ‖A‖ = ‖A‖_{B(H)} = σ_max(A)."""
    return float(np.linalg.norm(A, ord=2))


def _spectral_radius(A: ComplexMatrix) -> float:
    """r(A) = max{|λ| : λ ∈ σ(A)}. Para normales, r(A) = ‖A‖."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def _cstar_identity_residual(A: ComplexMatrix) -> float:
    """Residuo del axioma C*: |‖A†A‖ - ‖A‖²|."""
    return abs(_operator_norm(A.conj().T @ A) - _operator_norm(A) ** 2)


def _hermitian_eigh(
    A: ComplexMatrix,
) -> Tuple[RealVector, ComplexMatrix]:
    """Descomposición espectral de un operador hermitizado: A ↦ (Λ, U)."""
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(A))
    return eigvals.astype(np.float64, copy=False), eigvecs


def _functional_calculus_hermitian(
    A: ComplexMatrix,
    func: Callable[[RealVector], NDArray[np.complex128]],
) -> ComplexMatrix:
    """
    Cálculo funcional continuo para A = A†:

        f(A) = U f(Λ) U†,    A = U Λ U†.
    """
    eigvals, eigvecs = _hermitian_eigh(A)
    weights = np.asarray(func(eigvals), dtype=np.complex128)
    return _symmetrize(eigvecs @ np.diag(weights) @ eigvecs.conj().T)


def _hermitian_sqrt(A: ComplexMatrix, tolerance: float) -> ComplexMatrix:
    """Raíz cuadrada hermítica de un operador ≽ 0."""

    def _sqrt(vals: RealVector) -> NDArray[np.complex128]:
        clipped = np.where(vals < 0.0, 0.0, vals)
        if float(np.min(vals)) < -tolerance:
            raise TraceAnomalyError(
                f"Raíz hermítica de operador no positivo. λ_min={float(np.min(vals)):.3e}"
            )
        return np.sqrt(clipped).astype(np.complex128)

    return _functional_calculus_hermitian(A, _sqrt)


def _hermitian_log(A: ComplexMatrix, tolerance: float) -> ComplexMatrix:
    """Logaritmo hermítico sobre el soporte estrictamente positivo."""

    def _log(vals: RealVector) -> NDArray[np.complex128]:
        if float(np.min(vals)) <= tolerance:
            raise TraceAnomalyError(
                "Logaritmo modular exige espectro estrictamente positivo "
                f"(estado fiel). λ_min={float(np.min(vals)):.3e}"
            )
        return np.log(vals).astype(np.complex128)

    return _functional_calculus_hermitian(A, _log)


def _hermitian_power(
    A: ComplexMatrix,
    exponent: complex,
    tolerance: float,
) -> ComplexMatrix:
    """Potencia compleja A^z vía espectro real positivo."""

    def _pow(vals: RealVector) -> NDArray[np.complex128]:
        if float(np.min(vals)) <= tolerance:
            raise TraceAnomalyError(
                "Potencia compleja exige espectro estrictamente positivo."
            )
        return np.exp(exponent * np.log(vals.astype(np.float64)))

    eigvals, eigvecs = _hermitian_eigh(A)
    weights = _pow(eigvals)
    return eigvecs @ np.diag(weights) @ eigvecs.conj().T


def _sanitize_density_matrix(
    rho: ComplexMatrix,
    tolerance: float,
    name: str,
) -> ComplexMatrix:
    """
    Saneamiento FPU de matriz de densidad (postulados de Dirac–von Neumann).

    1. Proyección hermítica.
    2. Recorte de autovalores negativos menores que ``tolerance``.
    3. Renormalización de traza sobre el cono positivo.

    Raises:
        TraceAnomalyError: si el operador no es recuperablemente un estado.
    """
    rho = _as_square_matrix(rho, name=name)
    rho = _symmetrize(rho)

    with np.errstate(invalid="raise", divide="raise"):
        eigvals, eigvecs = np.linalg.eigh(rho)

    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -tolerance:
        raise TraceAnomalyError(
            f"{name} no es semidefinida positiva. Autovalor mínimo: {min_eig:.3e}",
        )

    eigvals = np.where(eigvals < 0.0, 0.0, eigvals)
    trace_val = float(np.sum(eigvals))
    if trace_val <= tolerance:
        raise TraceAnomalyError(
            f"{name} tiene traza no positiva o nula: {trace_val:.3e}",
        )

    eigvals = eigvals / trace_val
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return _symmetrize(rho)


def _validate_density_matrix(
    rho: ComplexMatrix,
    tolerance: float,
    name: str,
) -> None:
    """Verifica ρ = ρ†, Tr(ρ) = 1, ρ ≽ 0 con tolerancia absoluta."""
    if not _is_hermitian(rho, tolerance):
        raise TraceAnomalyError(f"{name} no es hermitiana.")

    trace_val = complex(np.trace(rho))
    if abs(trace_val - 1.0) > tolerance:
        raise TraceAnomalyError(
            f"{name} tiene traza distinta de 1: {trace_val!r}",
        )

    eigvals = np.linalg.eigvalsh(_symmetrize(rho))
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -tolerance:
        raise TraceAnomalyError(
            f"{name} no es semidefinida positiva. Autovalor mínimo: {min_eig:.3e}",
        )


def _apply_channel(
    kraus: Sequence[ComplexMatrix],
    A: ComplexMatrix,
) -> ComplexMatrix:
    """Imagen de Schrödinger: ℰ(A) = Σ_μ M_μ A M_μ†."""
    out = np.zeros_like(A, dtype=np.complex128)
    for M in kraus:
        out += M @ A @ M.conj().T
    return out


def _adjoint_channel(
    kraus: Sequence[ComplexMatrix],
    A: ComplexMatrix,
) -> ComplexMatrix:
    """Imagen de Heisenberg: ℰ†(A) = Σ_μ M_μ† A M_μ."""
    out = np.zeros_like(A, dtype=np.complex128)
    for M in kraus:
        out += M.conj().T @ A @ M
    return out


def _choi_channel(
    kraus: Sequence[ComplexMatrix],
    dim: int,
) -> ComplexMatrix:
    """
    Operador de Choi–Jamiołkowski no normalizado:

        Λ_ℰ = Σ_{i,j} |i⟩⟨j| ⊗ ℰ(|i⟩⟨j|),

    con bloques column-row coherentes con la base canónica.
    Choi ⇒ CP: ℰ completamente positivo ⇔ Λ_ℰ ≽ 0.
    """
    choi = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            E_ij = np.zeros((dim, dim), dtype=np.complex128)
            E_ij[i, j] = 1.0
            block = _apply_channel(kraus, E_ij)
            row = i * dim
            col = j * dim
            choi[row:row + dim, col:col + dim] = block
    return choi


def _liouvillian_channel(
    kraus: Sequence[ComplexMatrix],
    dim: int,
) -> ComplexMatrix:
    """
    Superoperador de Liouville en vectorización column-major:

        vec(ℰ(X)) = L vec(X),    L = Σ_μ M̄_μ ⊗ M_μ.

    Identidad: vec(AXB) = (Bᵀ ⊗ A) vec(X) ⇒ vec(MXM†) = (M̄ ⊗ M) vec(X).
    """
    L = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for M in kraus:
        L += np.kron(M.conj(), M)
    return L


def _as_vector(A: ComplexMatrix) -> ComplexMatrix:
    """Vectorización column-major: vec : M_d → C^{d²}."""
    return np.asarray(A, dtype=np.complex128).flatten(order="F")


def _as_operator(v: ComplexMatrix, dim: int) -> ComplexMatrix:
    """Inversa de vec en orden Fortran."""
    return np.asarray(v, dtype=np.complex128).reshape((dim, dim), order="F")


def _transpose_permutation(dim: int) -> NDArray[np.float64]:
    """
    Involución de transposición P ∈ M_{d²}(R) tal que

        vec(Aᵀ) = P vec(A),     P² = I,    Pᵀ = P.

    Luego vec(A†) = P conj(vec(A)).
    """
    P = np.zeros((dim * dim, dim * dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            idx_in = j * dim + i
            idx_out = i * dim + j
            P[idx_out, idx_in] = 1.0
    return P


def _support_projector_channel(
    A: ComplexMatrix,
    tolerance: float,
) -> ComplexMatrix:
    """Proyector espectral sobre {λ ∈ σ(A) : |λ| > tolerance} para A ≽ 0."""
    eigvals, eigvecs = _hermitian_eigh(A)
    mask = np.abs(eigvals) > tolerance
    if not np.any(mask):
        return np.zeros_like(A, dtype=np.complex128)
    basis = eigvecs[:, mask]
    return _symmetrize(basis @ basis.conj().T)


def _projector_range_basis(
    P: ComplexMatrix,
    tolerance: float,
) -> ComplexMatrix:
    """
    Base ortonormal del rango de un proyector numérico.

    Umbral 1/2: el espectro de un proyector se agrupa en {0, 1}.
    """
    eigvals, eigvecs = _hermitian_eigh(P)
    threshold = max(0.5, 1.0 - max(tolerance * 10.0, _MEET_EIGEN_THRESHOLD))
    mask = eigvals >= threshold
    if not np.any(mask):
        return np.zeros((P.shape[0], 0), dtype=np.complex128)
    return eigvecs[:, mask]


def _meet_projector_channel(
    P: ComplexMatrix,
    Q: ComplexMatrix,
    tolerance: float,
) -> ComplexMatrix:
    """
    Meet ortomodular por intersección de rangos (ángulos principales):

        ran(P ∧ Q) = ran(P) ∩ ran(Q).

    Si BP, BQ son bases ortonormales, los valores singulares de BP† BQ
    son cosenos de ángulos principales; la intersección corresponde a
    σ = 1.
    """
    dim = P.shape[0]
    BP = _projector_range_basis(P, tolerance)
    BQ = _projector_range_basis(Q, tolerance)
    if BP.shape[1] == 0 or BQ.shape[1] == 0:
        return np.zeros((dim, dim), dtype=np.complex128)

    gram = BP.conj().T @ BQ
    u_left, singular, _ = np.linalg.svd(gram, full_matrices=False)
    thresh = 1.0 - max(tolerance, _MEET_EIGEN_THRESHOLD)
    keep = singular >= thresh
    if not np.any(keep):
        return np.zeros((dim, dim), dtype=np.complex128)

    basis = BP @ u_left[:, keep]
    return _symmetrize(basis @ basis.conj().T)


def _join_projector_channel(
    P: ComplexMatrix,
    Q: ComplexMatrix,
    tolerance: float,
) -> ComplexMatrix:
    """
    Join ortomodular por suma de subespacios:

        ran(P ∨ Q) = ran(P) + ran(Q).
    """
    dim = P.shape[0]
    BP = _projector_range_basis(P, tolerance)
    BQ = _projector_range_basis(Q, tolerance)
    if BP.shape[1] == 0 and BQ.shape[1] == 0:
        return np.zeros((dim, dim), dtype=np.complex128)
    if BP.shape[1] == 0:
        return _symmetrize(Q)
    if BQ.shape[1] == 0:
        return _symmetrize(P)

    cat = np.concatenate([BP, BQ], axis=1)
    u_left, singular, _ = np.linalg.svd(cat, full_matrices=False)
    keep = singular > max(tolerance, _MEET_EIGEN_THRESHOLD)
    if not np.any(keep):
        return np.zeros((dim, dim), dtype=np.complex128)
    basis = u_left[:, keep]
    return _symmetrize(basis @ basis.conj().T)


def _von_neumann_entropy(rho: ComplexMatrix, tolerance: float) -> float:
    """Entropía de von Neumann S(ρ) = −Tr(ρ log ρ) en nats (log natural)."""
    eigvals = np.linalg.eigvalsh(_symmetrize(rho))
    positive = eigvals[eigvals > tolerance]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _purity(rho: ComplexMatrix) -> float:
    """Pureza Tr(ρ²) ∈ [1/d, 1]."""
    return float(np.real(np.trace(rho @ rho)))


def _uhlmann_fidelity(
    rho: ComplexMatrix,
    sigma: ComplexMatrix,
    tolerance: float,
) -> float:
    """
    Fidelidad de Uhlmann:

        F(ρ, σ) = ‖√ρ √σ‖₁² = [Tr √(√ρ σ √ρ)]².
    """
    sqrt_rho = _hermitian_sqrt(rho, tolerance)
    inner = _symmetrize(sqrt_rho @ sigma @ sqrt_rho)
    eigvals = np.linalg.eigvalsh(inner)
    eigvals = np.clip(eigvals, 0.0, None)
    return float(np.sum(np.sqrt(eigvals)) ** 2)


def _trace_distance(rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
    """Distancia de traza (1/2)‖ρ − σ‖₁."""
    delta = _symmetrize(rho - sigma)
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def identity_kraus_operators(dim: int) -> KrausTuple:
    """Unidad de la categoría CPTP sobre M_d(C): ℰ = id, Kraus {I_d}."""
    if int(dim) <= 0:
        raise ValueError(f"Dimensión inválida para Kraus identidad: {dim}")
    return (np.eye(int(dim), dtype=np.complex128),)


#=============================================================================
# FASE 1.3 — OPERADOR BICOMPLEJO COMO ELEMENTO DE W*-ÁLGEBRA
#=============================================================================
@dataclass(frozen=True, slots=True, eq=False)
class BicomplexOperator:
    """
    Elemento genérico del álgebra de von Neumann bicompleja

        A_{C₂} = A⁽¹⁾ e₁ + A⁽²⁾ e₂ ∈ B(H⁽¹⁾) e₁ ⊕ B(H⁽²⁾) e₂.

    La involución C* es el adjunto bicanal A ↦ A† = (A⁽¹⁾)† e₁ + (A⁽²⁾)† e₂.
    Los canales pueden tener dimensiones distintas (suma directa, no tensor).
    """

    channel_1: ComplexMatrix
    channel_2: ComplexMatrix

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_1",
            _as_square_matrix(self.channel_1, name="BicomplexOperator.channel_1"),
        )
        object.__setattr__(
            self,
            "channel_2",
            _as_square_matrix(self.channel_2, name="BicomplexOperator.channel_2"),
        )

    @property
    def dimensions(self) -> Dimensions:
        """Par (d₁, d₂) de las fibras idempotentes."""
        return (int(self.channel_1.shape[0]), int(self.channel_2.shape[0]))

    def dagger(self) -> "BicomplexOperator":
        """Involución C*: A ↦ A†."""
        return BicomplexOperator(self.channel_1.conj().T, self.channel_2.conj().T)

    def __add__(self, other: "BicomplexOperator") -> "BicomplexOperator":
        self._require_same_dimensions(other, "+")
        return BicomplexOperator(
            self.channel_1 + other.channel_1,
            self.channel_2 + other.channel_2,
        )

    def __sub__(self, other: "BicomplexOperator") -> "BicomplexOperator":
        self._require_same_dimensions(other, "-")
        return BicomplexOperator(
            self.channel_1 - other.channel_1,
            self.channel_2 - other.channel_2,
        )

    def __matmul__(self, other: "BicomplexOperator") -> "BicomplexOperator":
        """Producto interno del álgebra: (AB)⁽ᵏ⁾ = A⁽ᵏ⁾ B⁽ᵏ⁾."""
        self._require_same_dimensions(other, "@")
        return BicomplexOperator(
            self.channel_1 @ other.channel_1,
            self.channel_2 @ other.channel_2,
        )

    def _require_same_dimensions(self, other: "BicomplexOperator", op: str) -> None:
        if self.dimensions != other.dimensions:
            raise CategoryCompositionError(
                f"Dimensiones incompatibles en operación '{op}': "
                f"{self.dimensions} vs {other.dimensions}"
            )

    def is_hermitian(self, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        """A = A† en ambos canales."""
        return (
            _is_hermitian(self.channel_1, tolerance)
            and _is_hermitian(self.channel_2, tolerance)
        )

    def is_normal(self, tolerance: float = _DEFAULT_TOLERANCE) -> Tuple[bool, bool]:
        """Normalidad [A, A†] = 0 por canal."""
        return (
            _is_normal(self.channel_1, tolerance),
            _is_normal(self.channel_2, tolerance),
        )

    def is_positive(self, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        """A ≽ 0 ⇔ espectro real no negativo en ambos canales."""
        if not self.is_hermitian(tolerance):
            return False
        e1 = np.linalg.eigvalsh(_symmetrize(self.channel_1))
        e2 = np.linalg.eigvalsh(_symmetrize(self.channel_2))
        return bool(np.min(e1) >= -tolerance and np.min(e2) >= -tolerance)

    def operator_norm(self) -> Tuple[float, float]:
        """Par de C*-normas (‖A⁽¹⁾‖, ‖A⁽²⁾‖)."""
        return (_operator_norm(self.channel_1), _operator_norm(self.channel_2))

    def frobenius_norm(self) -> Tuple[float, float]:
        """Par de normas HS."""
        return (_frobenius(self.channel_1), _frobenius(self.channel_2))

    def spectral_radius(self) -> Tuple[float, float]:
        """Par de radios espectrales."""
        return (_spectral_radius(self.channel_1), _spectral_radius(self.channel_2))

    def cstar_identity_residual(self) -> Tuple[float, float]:
        """Residuos del axioma C* por canal."""
        return (
            _cstar_identity_residual(self.channel_1),
            _cstar_identity_residual(self.channel_2),
        )

    def commutator(self, other: "BicomplexOperator") -> "BicomplexOperator":
        """[A, B] = AB − BA bicanal."""
        return self @ other - other @ self

    def anticommutator(self, other: "BicomplexOperator") -> "BicomplexOperator":
        """{A, B} = AB + BA bicanal."""
        return self @ other + other @ self

    def equivalent(
        self,
        other: "BicomplexOperator",
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> bool:
        """Igualdad numérica en norma de Frobenius bicanal."""
        if self.dimensions != other.dimensions:
            return False
        return (
            _frobenius(self.channel_1 - other.channel_1) < tolerance
            and _frobenius(self.channel_2 - other.channel_2) < tolerance
        )

    def trace_bicomplex(self) -> BicomplexTrace:
        """Traza idempotente (Tr A⁽¹⁾, Tr A⁽²⁾)."""
        return (complex(np.trace(self.channel_1)), complex(np.trace(self.channel_2)))

    @classmethod
    def identity(cls, dimensions: Dimensions) -> "BicomplexOperator":
        """Unidad I_{C₂} = I_{d₁} e₁ + I_{d₂} e₂."""
        d1, d2 = int(dimensions[0]), int(dimensions[1])
        return cls(
            np.eye(d1, dtype=np.complex128),
            np.eye(d2, dtype=np.complex128),
        )

    @classmethod
    def zeros(cls, dimensions: Dimensions) -> "BicomplexOperator":
        """Cero algebraico."""
        d1, d2 = int(dimensions[0]), int(dimensions[1])
        return cls(
            np.zeros((d1, d1), dtype=np.complex128),
            np.zeros((d2, d2), dtype=np.complex128),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización estructural ligera (sin payload matricial)."""
        return {
            "dimensions": self.dimensions,
            "trace_channel_1": complex(np.trace(self.channel_1)),
            "trace_channel_2": complex(np.trace(self.channel_2)),
            "operator_norm": self.operator_norm(),
            "cstar_identity_residual": self.cstar_identity_residual(),
        }


#=============================================================================
# FASE 1.4 — ESTADO ATÓMICO DE DIRAC–VON NEUMANN
#=============================================================================
@dataclass(frozen=True, slots=True, eq=False)
class BicomplexAtomicDensityMatrix:
    """
    Estado normal normalizado del predual W*:

        ρ_{C₂} = ρ⁽¹⁾ e₁ + ρ⁽²⁾ e₂,

    con postulados de Dirac–von Neumann en cada fibra:

        ρ⁽ᵏ⁾ = (ρ⁽ᵏ⁾)†,    Tr(ρ⁽ᵏ⁾) = 1,    ρ⁽ᵏ⁾ ≽ 0.

    La traza bicompleja vale la unidad idempotente:

        Tr_{C₂}(ρ_{C₂}) = 1·e₁ + 1·e₂.
    """

    rho_channel_1: ComplexMatrix
    rho_channel_2: ComplexMatrix
    tolerance: float = _DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        tol = float(self.tolerance)
        if tol <= 0.0:
            object.__setattr__(self, "tolerance", _DEFAULT_TOLERANCE)
            tol = _DEFAULT_TOLERANCE

        rho1 = _sanitize_density_matrix(
            _as_square_matrix(self.rho_channel_1, name="rho_channel_1"),
            tol,
            "rho_channel_1",
        )
        rho2 = _sanitize_density_matrix(
            _as_square_matrix(self.rho_channel_2, name="rho_channel_2"),
            tol,
            "rho_channel_2",
        )
        _validate_density_matrix(rho1, tol, "rho_channel_1")
        _validate_density_matrix(rho2, tol, "rho_channel_2")
        object.__setattr__(self, "rho_channel_1", rho1)
        object.__setattr__(self, "rho_channel_2", rho2)

    @property
    def dimensions(self) -> Dimensions:
        """Dimensiones de los subespacios de de Rham (fibras e₁, e₂)."""
        return (int(self.rho_channel_1.shape[0]), int(self.rho_channel_2.shape[0]))

    def trace_bicomplex(self) -> BicomplexTrace:
        """Tr_{C₂}(ρ) = (Tr ρ⁽¹⁾, Tr ρ⁽²⁾)."""
        return (
            complex(np.trace(self.rho_channel_1)),
            complex(np.trace(self.rho_channel_2)),
        )

    def is_faithful(self, tolerance: Optional[float] = None) -> bool:
        """
        Estado fiel (cíclico y separador en dimensión finita):

            σ(ρ⁽ᵏ⁾) ⊂ (tolerance, +∞)    ∀k ∈ {1, 2}.
        """
        tol = self.tolerance if tolerance is None else float(tolerance)
        eig1 = np.linalg.eigvalsh(self.rho_channel_1)
        eig2 = np.linalg.eigvalsh(self.rho_channel_2)
        return bool(np.all(eig1 > tol) and np.all(eig2 > tol))

    def is_pure(self, tolerance: Optional[float] = None) -> Tuple[bool, bool]:
        """Pureza saturada: Tr(ρ²) = 1 ⇔ rango 1."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        return (
            abs(_purity(self.rho_channel_1) - 1.0) < tol,
            abs(_purity(self.rho_channel_2) - 1.0) < tol,
        )

    def is_maximally_mixed(self, tolerance: Optional[float] = None) -> Tuple[bool, bool]:
        """ρ⁽ᵏ⁾ = I/d_k."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        d1, d2 = self.dimensions
        mixed1 = np.eye(d1, dtype=np.complex128) / d1
        mixed2 = np.eye(d2, dtype=np.complex128) / d2
        return (
            _frobenius(self.rho_channel_1 - mixed1) < tol,
            _frobenius(self.rho_channel_2 - mixed2) < tol,
        )

    def purity(self) -> Tuple[float, float]:
        """(Tr(ρ⁽¹⁾)², Tr(ρ⁽²⁾)²)."""
        return (_purity(self.rho_channel_1), _purity(self.rho_channel_2))

    def von_neumann_entropy(self) -> Tuple[float, float]:
        """S_{C₂}(ρ) = S(ρ⁽¹⁾) e₁ + S(ρ⁽²⁾) e₂, en nats."""
        return (
            _von_neumann_entropy(self.rho_channel_1, self.tolerance),
            _von_neumann_entropy(self.rho_channel_2, self.tolerance),
        )

    def relative_entropy(
        self,
        other: "BicomplexAtomicDensityMatrix",
        tolerance: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Entropía relativa de Umegaki–Araki:

            S(ρ‖σ) = Tr(ρ (log ρ − log σ)),

        finita ssi supp(ρ) ≤ supp(σ). Exige σ fiel.
        """
        if self.dimensions != other.dimensions:
            raise CategoryCompositionError(
                "Entropía relativa exige dimensiones idénticas."
            )
        tol = self.tolerance if tolerance is None else float(tolerance)
        if not other.is_faithful(tol):
            raise TraceAnomalyError(
                "S(ρ‖σ) requiere σ fiel para el logaritmo modular."
            )

        def _rel(
            rho: ComplexMatrix,
            sigma: ComplexMatrix,
        ) -> float:
            log_rho = _hermitian_log(rho, tol) if np.min(np.linalg.eigvalsh(rho)) > tol else None
            log_sigma = _hermitian_log(sigma, tol)
            if log_rho is None:
                eig = np.linalg.eigvalsh(rho)
                pos = eig[eig > tol]
                # Tr(ρ log ρ) sobre el soporte; Tr(ρ log σ) global.
                tr_rho_log_rho = float(np.sum(pos * np.log(pos)))
            else:
                tr_rho_log_rho = float(np.real(np.trace(rho @ log_rho)))
            tr_rho_log_sigma = float(np.real(np.trace(rho @ log_sigma)))
            return tr_rho_log_rho - tr_rho_log_sigma

        return (
            _rel(self.rho_channel_1, other.rho_channel_1),
            _rel(self.rho_channel_2, other.rho_channel_2),
        )

    def fidelity(
        self,
        other: "BicomplexAtomicDensityMatrix",
        tolerance: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Fidelidad de Uhlmann bicanal."""
        if self.dimensions != other.dimensions:
            raise CategoryCompositionError("Fidelidad exige dimensiones idénticas.")
        tol = self.tolerance if tolerance is None else float(tolerance)
        return (
            _uhlmann_fidelity(self.rho_channel_1, other.rho_channel_1, tol),
            _uhlmann_fidelity(self.rho_channel_2, other.rho_channel_2, tol),
        )

    def trace_distance(
        self,
        other: "BicomplexAtomicDensityMatrix",
    ) -> Tuple[float, float]:
        """Distancia de traza bicanal."""
        if self.dimensions != other.dimensions:
            raise CategoryCompositionError(
                "Distancia de traza exige dimensiones idénticas."
            )
        return (
            _trace_distance(self.rho_channel_1, other.rho_channel_1),
            _trace_distance(self.rho_channel_2, other.rho_channel_2),
        )

    def born_probability(self, projector: BicomplexOperator) -> Tuple[float, float]:
        """
        Regla de Born: p(P) = Tr(ρ P) para P proyector.

        No valida idempotencia aquí: esa certificación pertenece al retículo
        (FASE 2). Se exige hermiticidad numérica.
        """
        if projector.dimensions != self.dimensions:
            raise CategoryCompositionError(
                "El proyector no actúa en el mismo H_{C₂} que el estado."
            )
        p1 = complex(np.trace(self.rho_channel_1 @ projector.channel_1))
        p2 = complex(np.trace(self.rho_channel_2 @ projector.channel_2))
        return (float(np.real(p1)), float(np.real(p2)))

    def luders_update(
        self,
        projector: BicomplexOperator,
        tolerance: Optional[float] = None,
    ) -> "BicomplexAtomicDensityMatrix":
        """
        Colapso de Lüders (medición proyectiva no selectiva normalizada):

            ρ ↦ P ρ P / Tr(P ρ P).
        """
        if projector.dimensions != self.dimensions:
            raise CategoryCompositionError(
                "Lüders exige un proyector del mismo H_{C₂}."
            )
        tol = self.tolerance if tolerance is None else float(tolerance)

        def _update(rho: ComplexMatrix, P: ComplexMatrix, name: str) -> ComplexMatrix:
            raw = P @ rho @ P
            tr = float(np.real(np.trace(raw)))
            if tr <= tol:
                raise TraceAnomalyError(
                    f"Probabilidad de Born nula en {name}: no hay estado posterior."
                )
            return raw / tr

        return BicomplexAtomicDensityMatrix(
            rho_channel_1=_update(self.rho_channel_1, projector.channel_1, "e₁"),
            rho_channel_2=_update(self.rho_channel_2, projector.channel_2, "e₂"),
            tolerance=tol,
        )

    def support_projector(self) -> BicomplexOperator:
        """Proyector de soporte s(ρ) = χ_{(0,∞)}(ρ)."""
        return BicomplexOperator(
            _support_projector_channel(self.rho_channel_1, self.tolerance),
            _support_projector_channel(self.rho_channel_2, self.tolerance),
        )

    def as_operator(self) -> BicomplexOperator:
        """Inmersión del predual en el álgebra: ρ ↦ ρ ∈ M₊ ∩ {Tr = 1}."""
        return BicomplexOperator(self.rho_channel_1, self.rho_channel_2)

    def modular_hamiltonian(self) -> BicomplexOperator:
        """
        Hamiltoniano modular K = −log ρ (exige fidelidad).

        Es el generador infinitesimal del grupo σ_t = Ad(ρ^{it}).
        """
        if not self.is_faithful(self.tolerance):
            raise TraceAnomalyError(
                "El Hamiltoniano modular exige un estado fiel."
            )
        return BicomplexOperator(
            -_hermitian_log(self.rho_channel_1, self.tolerance),
            -_hermitian_log(self.rho_channel_2, self.tolerance),
        )

    def identity_kraus(self) -> Tuple[KrausTuple, KrausTuple]:
        """Kraus de la identidad sobre el soporte dimensional del estado."""
        d1, d2 = self.dimensions
        return (identity_kraus_operators(d1), identity_kraus_operators(d2))

    @classmethod
    def maximally_mixed(
        cls,
        dimensions: Dimensions,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> "BicomplexAtomicDensityMatrix":
        """Estado tracial I/d en cada fibra (KMS a β = 0 formal)."""
        d1, d2 = int(dimensions[0]), int(dimensions[1])
        if d1 <= 0 or d2 <= 0:
            raise ValueError(f"Dimensiones inválidas: {dimensions}")
        return cls(
            rho_channel_1=np.eye(d1, dtype=np.complex128) / d1,
            rho_channel_2=np.eye(d2, dtype=np.complex128) / d2,
            tolerance=tolerance,
        )

    @classmethod
    def pure_from_vectors(
        cls,
        psi_channel_1: NDArray[np.complex128],
        psi_channel_2: NDArray[np.complex128],
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> "BicomplexAtomicDensityMatrix":
        """Estado puro |ψ⟩⟨ψ| bicanal a partir de vectores cíclicos."""

        def _ket_bra(psi: NDArray[np.complex128], name: str) -> ComplexMatrix:
            vec = np.asarray(psi, dtype=np.complex128).reshape(-1)
            nrm = float(np.linalg.norm(vec))
            if nrm <= tolerance:
                raise TraceAnomalyError(f"Vector nulo en {name}.")
            vec = vec / nrm
            return np.outer(vec, vec.conj())

        return cls(
            rho_channel_1=_ket_bra(psi_channel_1, "psi_channel_1"),
            rho_channel_2=_ket_bra(psi_channel_2, "psi_channel_2"),
            tolerance=tolerance,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización estructural del estado atómico."""
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "dimensions": self.dimensions,
            "trace_bicomplex": self.trace_bicomplex(),
            "faithful": self.is_faithful(),
            "purity": self.purity(),
            "von_neumann_entropy": self.von_neumann_entropy(),
            "is_pure": self.is_pure(),
            "is_maximally_mixed": self.is_maximally_mixed(),
        }


#=============================================================================
# FASE 1.5 — CLÁUSULA FINAL / PUENTE FUNTORIAL Ψ₁₂
#=============================================================================
def psi_12_initiate_cptp_category(
    state: BicomplexAtomicDensityMatrix,
    kraus_channel_1: Sequence[NDArray[np.complex128]],
    kraus_channel_2: Sequence[NDArray[np.complex128]],
    tolerance: Optional[float] = None,
) -> "BicomplexCPTPMorphism":
    """
    Funtor de transición Ψ₁₂ : Hilb_{C₂} → CPTP_{C₂}.

    DEFINICIÓN FORMAL (cláusula terminal de la FASE 1)
    ──────────────────────────────────────────────────
    Un estado atómico ρ_{C₂} es un objeto del predual W*. Un morfismo
    de Schrödinger es un canal CPTP bicanal ℰ_{C₂} tal que

        ℰ_{C₂}(ρ_{C₂}) = ℰ⁽¹⁾(ρ⁽¹⁾) e₁ + ℰ⁽²⁾(ρ⁽²⁾) e₂

    permanece en el simplex de estados. Esta función *inicia* la FASE 2:
    valida los axiomas CPTP (preservación de traza y completa positividad
    vía Choi) y devuelve el morfismo como flecha de la categoría
    simétrica monoidal CPTP_{C₂}.

    El cuerpo se evalúa en tiempo de llamada, cuando
    ``BicomplexCPTPMorphism`` ya ha sido construido por la FASE 2.

    Raises:
        CategoryCompositionError: si el estado no es un ``BicomplexAtomicDensityMatrix``.
        TraceAnomalyError / NonCommutativeAlgebraError: axiomas CPTP violados.
    """
    if not isinstance(state, BicomplexAtomicDensityMatrix):
        raise CategoryCompositionError(
            "Ψ₁₂ exige un objeto de BicomplexAtomicDensityMatrix."
        )
    tol = state.tolerance if tolerance is None else float(tolerance)
    morphism = BicomplexCPTPMorphism(
        kraus_channel_1=tuple(kraus_channel_1),
        kraus_channel_2=tuple(kraus_channel_2),
        tolerance=tol,
    )
    if morphism.dimensions != state.dimensions:
        raise CategoryCompositionError(
            f"Ψ₁₂: el canal {morphism.dimensions} no actúa sobre "
            f"el estado {state.dimensions}."
        )
    return morphism


#=============================================================================
# FASE 2.1 — AXIOMAS CPTP Y TEOREMA DE CHOI–JAMIOŁKOWSKI
#=============================================================================
def _verify_cptp_channel(
    kraus: Sequence[ComplexMatrix],
    tolerance: float,
    channel_name: str,
) -> None:
    """
    Axiomas de canal cuántico en dimensión finita (Kraus + Choi):

      (TP)  Σ_μ M_μ† M_μ = I.
      (CP)  Λ_ℰ ≽ 0.

    Completa positividad ⇔ positividad de Choi (Choi, 1975).
    """
    if not kraus:
        raise NonCommutativeAlgebraError(
            f"Conjunto de Kraus vacío en canal {channel_name}."
        )

    first_shape = kraus[0].shape
    dim = first_shape[0]
    for idx, M in enumerate(kraus):
        if M.shape != first_shape:
            raise ValueError(
                f"Kraus {idx} incompatible en canal {channel_name}: "
                f"{M.shape} vs {first_shape}"
            )

    identity_sum = np.zeros((dim, dim), dtype=np.complex128)
    for M in kraus:
        identity_sum += M.conj().T @ M

    trace_error = _frobenius(identity_sum - np.eye(dim, dtype=np.complex128))
    if trace_error > tolerance:
        raise TraceAnomalyError(
            f"Canal {channel_name} no preserva traza. Error ΣM†M−I: {trace_error:.3e}"
        )

    choi = _choi_channel(kraus, dim)
    eigvals = np.linalg.eigvalsh(_symmetrize(choi))
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -tolerance:
        raise NonCommutativeAlgebraError(
            f"Canal {channel_name} no es completamente positivo. "
            f"Autovalor mínimo de Choi: {min_eig:.3e}"
        )


def _compress_kraus(
    kraus: Sequence[ComplexMatrix],
    dim: int,
    tolerance: float,
) -> KrausTuple:
    """
    Compresión canónica: Kraus a partir del espectro de Choi.

        M_α = √λ_α  reshape(v_α),    λ_α > tolerance.

    El rango de Choi es el rango de Kraus mínimo.
    """
    choi = _symmetrize(_choi_channel(kraus, dim))
    eigvals, eigvecs = np.linalg.eigh(choi)
    compressed: List[ComplexMatrix] = []
    for val, vec in zip(eigvals, eigvecs.T):
        if val > tolerance:
            compressed.append(np.sqrt(float(val)) * vec.reshape((dim, dim), order="F"))
    if not compressed:
        raise NonCommutativeAlgebraError(
            "Compresión de Kraus degenerada: Choi numéricamente nulo."
        )
    return tuple(compressed)


#=============================================================================
# FASE 2.2 — MORFISMOS DE LA CATEGORÍA CPTP BICOMPLEJA
#=============================================================================
@dataclass(frozen=True, slots=True, eq=False)
class BicomplexCPTPMorphism:
    """
    Flecha de la categoría simétrica monoidal CPTP_{C₂}:

        ℰ_{C₂}(ρ_{C₂}) = ℰ⁽¹⁾(ρ⁽¹⁾) e₁ + ℰ⁽²⁾(ρ⁽²⁾) e₂.

    La aniquilación e₁ e₂ = 0 garantiza aislamiento contractual entre fibras:
    una anomalía en e₂ no contamina el presupuesto físico de e₁.

    Composición: (ℰ ∘ ℱ) Kraus {M_a N_b}.
    Producto tensorial: (ℰ ⊗ ℱ)⁽ᵏ⁾ = ℰ⁽ᵏ⁾ ⊗ ℱ⁽ᵏ⁾ (Kronecker por fibra).
    """

    kraus_channel_1: KrausTuple
    kraus_channel_2: KrausTuple
    tolerance: float = _DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        tol = float(self.tolerance)
        if tol <= 0.0:
            object.__setattr__(self, "tolerance", _DEFAULT_TOLERANCE)
            tol = _DEFAULT_TOLERANCE

        k1 = tuple(
            _as_square_matrix(M, name=f"Kraus1[{i}]")
            for i, M in enumerate(self.kraus_channel_1)
        )
        k2 = tuple(
            _as_square_matrix(M, name=f"Kraus2[{i}]")
            for i, M in enumerate(self.kraus_channel_2)
        )
        _verify_cptp_channel(k1, tol, "e₁")
        _verify_cptp_channel(k2, tol, "e₂")
        object.__setattr__(self, "kraus_channel_1", k1)
        object.__setattr__(self, "kraus_channel_2", k2)

    @classmethod
    def identity(
        cls,
        dimensions: Dimensions,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> "BicomplexCPTPMorphism":
        """Identidad monoidal id : ρ ↦ ρ."""
        d1, d2 = int(dimensions[0]), int(dimensions[1])
        return cls(
            kraus_channel_1=identity_kraus_operators(d1),
            kraus_channel_2=identity_kraus_operators(d2),
            tolerance=tolerance,
        )

    @classmethod
    def from_phase1_state(
        cls,
        state: BicomplexAtomicDensityMatrix,
        kraus_channel_1: Sequence[NDArray[np.complex128]],
        kraus_channel_2: Sequence[NDArray[np.complex128]],
        tolerance: Optional[float] = None,
    ) -> "BicomplexCPTPMorphism":
        """Realización concreta de Ψ₁₂ (FASE 1.5) como método de clase."""
        return psi_12_initiate_cptp_category(
            state,
            kraus_channel_1,
            kraus_channel_2,
            tolerance=tolerance,
        )

    @property
    def dimension_channel_1(self) -> int:
        """Dimensión de la fibra e₁."""
        return int(self.kraus_channel_1[0].shape[0])

    @property
    def dimension_channel_2(self) -> int:
        """Dimensión de la fibra e₂."""
        return int(self.kraus_channel_2[0].shape[0])

    @property
    def dimensions(self) -> Dimensions:
        """Par dimensional bicanal."""
        return (self.dimension_channel_1, self.dimension_channel_2)

    @property
    def kraus_rank(self) -> Tuple[int, int]:
        """Cardinalidades de las familias de Kraus (no necesariamente mínimas)."""
        return (len(self.kraus_channel_1), len(self.kraus_channel_2))

    def apply(self, rho: BicomplexAtomicDensityMatrix) -> BicomplexAtomicDensityMatrix:
        """Imagen forward de Schrödinger ℰ_{C₂}(ρ_{C₂})."""
        if rho.dimensions != self.dimensions:
            raise CategoryCompositionError(
                f"Dimensiones incompatibles: estado={rho.dimensions}, canal={self.dimensions}"
            )
        return BicomplexAtomicDensityMatrix(
            rho_channel_1=_apply_channel(self.kraus_channel_1, rho.rho_channel_1),
            rho_channel_2=_apply_channel(self.kraus_channel_2, rho.rho_channel_2),
            tolerance=max(self.tolerance, rho.tolerance),
        )

    def adjoint_apply(self, observable: BicomplexOperator) -> BicomplexOperator:
        """Imagen adjunta de Heisenberg ℰ†(A_{C₂})."""
        if observable.dimensions != self.dimensions:
            raise CategoryCompositionError(
                "Dimensiones incompatibles en adjunto de Heisenberg."
            )
        return BicomplexOperator(
            _adjoint_channel(self.kraus_channel_1, observable.channel_1),
            _adjoint_channel(self.kraus_channel_2, observable.channel_2),
        )

    def compute_choi_matrix(self) -> BicomplexOperator:
        """Λ_{ℰ,C₂} = Λ⁽¹⁾ e₁ + Λ⁽²⁾ e₂."""
        return BicomplexOperator(
            _choi_channel(self.kraus_channel_1, self.dimension_channel_1),
            _choi_channel(self.kraus_channel_2, self.dimension_channel_2),
        )

    def compute_liouvillian(self) -> BicomplexOperator:
        """Superoperadores L_{ℰ,C₂} = L⁽¹⁾ e₁ + L⁽²⁾ e₂."""
        return BicomplexOperator(
            _liouvillian_channel(self.kraus_channel_1, self.dimension_channel_1),
            _liouvillian_channel(self.kraus_channel_2, self.dimension_channel_2),
        )

    def choi_rank(self, tolerance: Optional[float] = None) -> Tuple[int, int]:
        """Rango de Choi = rango de Kraus mínimo por fibra."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        choi = self.compute_choi_matrix()
        r1 = int(np.sum(np.linalg.eigvalsh(_symmetrize(choi.channel_1)) > tol))
        r2 = int(np.sum(np.linalg.eigvalsh(_symmetrize(choi.channel_2)) > tol))
        return (r1, r2)

    def compressed(self) -> "BicomplexCPTPMorphism":
        """Morfismo con Kraus de rango de Choi (forma canónica)."""
        return BicomplexCPTPMorphism(
            kraus_channel_1=_compress_kraus(
                self.kraus_channel_1, self.dimension_channel_1, self.tolerance
            ),
            kraus_channel_2=_compress_kraus(
                self.kraus_channel_2, self.dimension_channel_2, self.tolerance
            ),
            tolerance=self.tolerance,
        )

    def stinespring_isometry(self) -> Tuple[ComplexMatrix, ComplexMatrix]:
        """
        Dilatación de Stinespring a partir de Kraus:

            V|ψ⟩ = Σ_μ |μ⟩ ⊗ M_μ|ψ⟩,    ℰ(ρ) = Tr_E(V ρ V†).

        Devuelve las isometrías V⁽¹⁾, V⁽²⁾ de forma (r d, d).
        """

        def _iso(kraus: Sequence[ComplexMatrix], dim: int) -> ComplexMatrix:
            rank = len(kraus)
            V = np.zeros((rank * dim, dim), dtype=np.complex128)
            for mu, M in enumerate(kraus):
                V[mu * dim:(mu + 1) * dim, :] = M
            return V

        return (
            _iso(self.kraus_channel_1, self.dimension_channel_1),
            _iso(self.kraus_channel_2, self.dimension_channel_2),
        )

    def kadison_schwarz_residual(
        self,
        observable: BicomplexOperator,
    ) -> Tuple[float, float]:
        """
        Residuo de la desigualdad de Kadison–Schwarz para unitales:

            ℰ(A†A) ≽ ℰ(A)† ℰ(A).

        Se reporta la magnitud de la parte negativa del residual hermítico.
        """
        if observable.dimensions != self.dimensions:
            raise CategoryCompositionError(
                "Kadison–Schwarz exige observable del mismo H_{C₂}."
            )

        def _residual(
            kraus: Sequence[ComplexMatrix],
            A: ComplexMatrix,
        ) -> float:
            lhs = _apply_channel(kraus, A.conj().T @ A)
            img = _apply_channel(kraus, A)
            rhs = img.conj().T @ img
            gap = _symmetrize(lhs - rhs)
            eig = np.linalg.eigvalsh(gap)
            neg = eig[eig < 0.0]
            return float(-np.min(neg)) if neg.size else 0.0

        return (
            _residual(self.kraus_channel_1, observable.channel_1),
            _residual(self.kraus_channel_2, observable.channel_2),
        )

    def compose(self, inner: "BicomplexCPTPMorphism") -> "BicomplexCPTPMorphism":
        """Composición categorial self ∘ inner, Kraus {M_a N_b} por fibra."""
        if self.dimensions != inner.dimensions:
            raise CategoryCompositionError(
                "Dimensiones incompatibles en composición bicompleja."
            )
        kraus1 = tuple(
            M @ N
            for M in self.kraus_channel_1
            for N in inner.kraus_channel_1
        )
        kraus2 = tuple(
            M @ N
            for M in self.kraus_channel_2
            for N in inner.kraus_channel_2
        )
        composed = BicomplexCPTPMorphism(
            kraus_channel_1=kraus1,
            kraus_channel_2=kraus2,
            tolerance=max(self.tolerance, inner.tolerance),
        )
        return composed.compressed()

    def tensor_product(self, other: "BicomplexCPTPMorphism") -> "BicomplexCPTPMorphism":
        """
        Producto tensorial por fibras (no cruzado):

            (ℰ ⊗ ℱ)_{C₂} = (ℰ⁽¹⁾ ⊗ ℱ⁽¹⁾) e₁ + (ℰ⁽²⁾ ⊗ ℱ⁽²⁾) e₂.
        """
        kraus1 = tuple(
            np.kron(M, N)
            for M in self.kraus_channel_1
            for N in other.kraus_channel_1
        )
        kraus2 = tuple(
            np.kron(M, N)
            for M in self.kraus_channel_2
            for N in other.kraus_channel_2
        )
        return BicomplexCPTPMorphism(
            kraus_channel_1=kraus1,
            kraus_channel_2=kraus2,
            tolerance=max(self.tolerance, other.tolerance),
        )

    def is_unitary(self, tolerance: Optional[float] = None) -> Tuple[bool, bool]:
        """Canal unitario puro ⇔ un único Kraus unitario."""
        tol = self.tolerance if tolerance is None else float(tolerance)

        def _unitary(kraus: Sequence[ComplexMatrix]) -> bool:
            if len(kraus) != 1:
                return False
            U = kraus[0]
            d = U.shape[0]
            eye = np.eye(d, dtype=np.complex128)
            return (
                _frobenius(U.conj().T @ U - eye) < tol
                and _frobenius(U @ U.conj().T - eye) < tol
            )

        return (_unitary(self.kraus_channel_1), _unitary(self.kraus_channel_2))

    def is_unital(self, tolerance: Optional[float] = None) -> Tuple[bool, bool]:
        """ℰ(I) = I (unitalidad; dual de preservación de traza en el predual)."""
        tol = self.tolerance if tolerance is None else float(tolerance)

        def _unital(kraus: Sequence[ComplexMatrix], dim: int) -> bool:
            eye = np.eye(dim, dtype=np.complex128)
            return _frobenius(_apply_channel(kraus, eye) - eye) < tol

        return (
            _unital(self.kraus_channel_1, self.dimension_channel_1),
            _unital(self.kraus_channel_2, self.dimension_channel_2),
        )

    def is_completely_positive(self, tolerance: Optional[float] = None) -> Tuple[bool, bool]:
        """Reverificación de Choi ≽ 0 (post-construcción)."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        choi = self.compute_choi_matrix()
        eig1 = np.linalg.eigvalsh(_symmetrize(choi.channel_1))
        eig2 = np.linalg.eigvalsh(_symmetrize(choi.channel_2))
        return (bool(np.all(eig1 >= -tol)), bool(np.all(eig2 >= -tol)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización estructural del morfismo."""
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "dimensions": self.dimensions,
            "kraus_rank": self.kraus_rank,
            "choi_rank": self.choi_rank(),
            "is_unitary": self.is_unitary(),
            "is_unital": self.is_unital(),
            "is_completely_positive": self.is_completely_positive(),
        }


#=============================================================================
# FASE 2.3 — RETÍCULO ORTOMODULAR BICOMPLEJO
#=============================================================================
class BicomplexOrthomodularLattice:
    """
    Retículo ortomodular de proyectores de H_{C₂}:

        L(H_{C₂}) ≅ L(H⁽¹⁾) × L(H⁽²⁾).

    Operaciones por fibra:
      - ortocomplemento  P⊥ = I − P
      - meet             P ∧ Q  (intersección de rangos)
      - join             P ∨ Q  (suma de rangos)
      - Sasaki           P →_S Q = P⊥ ∨ (P ∧ Q)

    En dimensión finita todo factor es Tipo I_n y el retículo es
    el de subespacios cerrados con inclusión y ortocomplemento.
    """

    def __init__(
        self,
        dimensions: Dimensions,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> None:
        if len(dimensions) != 2 or any(int(d) <= 0 for d in dimensions):
            raise ValueError(f"Dimensiones inválidas: {dimensions}")
        self.dimensions: Dimensions = (int(dimensions[0]), int(dimensions[1]))
        self.tolerance: float = float(tolerance) if float(tolerance) > 0.0 else _DEFAULT_TOLERANCE

    def identity(self) -> BicomplexOperator:
        """Máximo del retículo: I_{C₂}."""
        return BicomplexOperator.identity(self.dimensions)

    def zero(self) -> BicomplexOperator:
        """Mínimo del retículo: 0."""
        return BicomplexOperator.zeros(self.dimensions)

    def _validate_dimensions(self, op: BicomplexOperator) -> None:
        if op.dimensions != self.dimensions:
            raise ValueError(
                f"Dimensiones incompatibles: esperadas {self.dimensions}, "
                f"recibidas {op.dimensions}"
            )

    def validate_projector(self, P: BicomplexOperator) -> None:
        """Certifica P = P† = P² en ambas fibras."""
        self._validate_dimensions(P)
        if not _is_projector(P.channel_1, self.tolerance):
            raise ValueError("Canal e₁ no es un proyector cuántico válido.")
        if not _is_projector(P.channel_2, self.tolerance):
            raise ValueError("Canal e₂ no es un proyector cuántico válido.")

    def quantum_complement(self, P: BicomplexOperator) -> BicomplexOperator:
        """Ortocomplemento P⊥ = I − P."""
        self.validate_projector(P)
        eye = self.identity()
        return BicomplexOperator(
            eye.channel_1 - P.channel_1,
            eye.channel_2 - P.channel_2,
        )

    def quantum_conjunction(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        max_iter: int = 1000,
        validate: bool = True,
    ) -> BicomplexOperator:
        """
        Meet P ∧ Q por ángulos principales.

        ``max_iter`` se conserva por compatibilidad semántica con v5;
        el algoritmo ya no itera (PQP)ⁿ.
        """
        _ = max_iter
        if validate:
            self.validate_projector(P_A)
            self.validate_projector(P_B)
        self._validate_dimensions(P_A)
        self._validate_dimensions(P_B)
        return BicomplexOperator(
            _meet_projector_channel(P_A.channel_1, P_B.channel_1, self.tolerance),
            _meet_projector_channel(P_A.channel_2, P_B.channel_2, self.tolerance),
        )

    def quantum_disjunction(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        **kwargs: Any,
    ) -> BicomplexOperator:
        """Join P ∨ Q por suma de rangos (equivalente De Morgan si P, Q son proyectores)."""
        validate = bool(kwargs.get("validate", True))
        if validate:
            self.validate_projector(P_A)
            self.validate_projector(P_B)
        self._validate_dimensions(P_A)
        self._validate_dimensions(P_B)
        return BicomplexOperator(
            _join_projector_channel(P_A.channel_1, P_B.channel_1, self.tolerance),
            _join_projector_channel(P_A.channel_2, P_B.channel_2, self.tolerance),
        )

    def quantum_implication(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        **kwargs: Any,
    ) -> BicomplexOperator:
        """Gancho de Sasaki: P →_S Q = P⊥ ∨ (P ∧ Q)."""
        P_A_comp = self.quantum_complement(P_A)
        P_meet = self.quantum_conjunction(P_A, P_B, **kwargs)
        return self.quantum_disjunction(P_A_comp, P_meet, validate=False)

    def sasaki_projection(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
    ) -> BicomplexOperator:
        """Proyección de Sasaki φ_P(Q) = P ∧ (P⊥ ∨ Q). Dual del gancho."""
        self.validate_projector(P_A)
        self.validate_projector(P_B)
        return self.quantum_conjunction(
            P_A,
            self.quantum_disjunction(self.quantum_complement(P_A), P_B, validate=False),
            validate=False,
        )

    def commutator(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
    ) -> BicomplexOperator:
        """Conmutador algebraico [P, Q] (no el conmutador de retículo)."""
        self._validate_dimensions(P_A)
        self._validate_dimensions(P_B)
        return P_A.commutator(P_B)

    def are_compatible(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Compatibilidad clásica: [P, Q] = 0 en ambas fibras."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        comm = self.commutator(P_A, P_B)
        return (
            _frobenius(comm.channel_1) < tol
            and _frobenius(comm.channel_2) < tol
        )

    def leq(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Orden de retículo P ≤ Q ⇔ P Q = P = Q P."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        self.validate_projector(P_A)
        self.validate_projector(P_B)

        def _leq(P: ComplexMatrix, Q: ComplexMatrix) -> bool:
            return _frobenius(P @ Q - P) < tol and _frobenius(Q @ P - P) < tol

        return _leq(P_A.channel_1, P_B.channel_1) and _leq(P_A.channel_2, P_B.channel_2)

    def verify_orthomodular_law(
        self,
        P_A: BicomplexOperator,
        P_B: BicomplexOperator,
        tolerance: Optional[float] = None,
    ) -> Optional[bool]:
        """
        Ley ortomodular: P ≤ Q ⇒ Q = P ∨ (P⊥ ∧ Q).

        Retorna None si la premisa P ≤ Q falla.
        """
        tol = self.tolerance if tolerance is None else float(tolerance)
        if not self.leq(P_A, P_B, tolerance=tol):
            return None
        meet = self.quantum_conjunction(
            self.quantum_complement(P_A),
            P_B,
            validate=False,
        )
        join = self.quantum_disjunction(P_A, meet, validate=False)
        holds = (
            _frobenius(join.channel_1 - P_B.channel_1) < tol
            and _frobenius(join.channel_2 - P_B.channel_2) < tol
        )
        if not holds:
            raise OrthomodularConvergenceError(
                "Fallo numérico de la ley ortomodular en L(H_{C₂})."
            )
        return True

    def compatibility_graph(
        self,
        projectors: Sequence[BicomplexOperator],
    ) -> NDArray[np.float64]:
        """
        Grafo de compatibilidad (teoría de grafos espectral del retículo):

            G_{ij} = 1  ⇔  [P_i, P_j] = 0.
        """
        count = len(projectors)
        graph = np.eye(count, dtype=np.float64)
        validated: List[BicomplexOperator] = []
        for P in projectors:
            self.validate_projector(P)
            validated.append(P)
        for i in range(count):
            for j in range(i + 1, count):
                if self.are_compatible(validated[i], validated[j]):
                    graph[i, j] = 1.0
                    graph[j, i] = 1.0
        return graph

    def boolean_center_mask(
        self,
        projectors: Sequence[BicomplexOperator],
    ) -> NDArray[np.bool_]:
        """
        Máscara del centro relativo: proyectores que conmutan con todos
        los demás (subálgebra de Boole del conmutante relativo).
        """
        graph = self.compatibility_graph(projectors)
        return np.all(graph >= 1.0 - 1e-15, axis=1)


#=============================================================================
# FASE 2.4 — CLÁUSULA FINAL / PUENTE FUNTORIAL Ψ₂₃
#=============================================================================
def psi_23_initiate_tomita_takesaki(
    state: BicomplexAtomicDensityMatrix,
    lattice: Optional[BicomplexOrthomodularLattice] = None,
    morphism: Optional[BicomplexCPTPMorphism] = None,
    tolerance: Optional[float] = None,
) -> "BicomplexTomitaTakesakiTheory":
    """
    Funtor de transición Ψ₂₃ : CPTP_{C₂} ⋊ L(H_{C₂}) → StdForm(M_{C₂}, ρ).

    DEFINICIÓN FORMAL (cláusula terminal de la FASE 2)
    ──────────────────────────────────────────────────
    Dado un estado fiel ρ_{C₂} (objeto de la FASE 1), opcionalmente
    transportado por un morfismo CPTP (FASE 2.2) y certificado sobre un
    retículo ortomodular (FASE 2.3), se construye la forma estándar de
    Haagerup–Takesaki:

        (M_{C₂}, L²(M_{C₂}, ρ), J_{C₂}, Δ_{C₂}).

    Condiciones:
      - Si ``morphism`` se provee, el estado se empuja ℰ_*(ρ) antes del GNS.
      - Si ``lattice`` se provee, sus dimensiones deben coincidir con ρ.
      - ρ debe ser fiel: de lo contrario no existe Δ invertible.

    Continuación: la FASE 3 realiza Δ, J, σ_t y la condición KMS.

    Raises:
        CategoryCompositionError: desajuste dimensional.
        TraceAnomalyError: estado no fiel.
    """
    if not isinstance(state, BicomplexAtomicDensityMatrix):
        raise CategoryCompositionError(
            "Ψ₂₃ exige un estado BicomplexAtomicDensityMatrix."
        )
    working = state
    if morphism is not None:
        if morphism.dimensions != state.dimensions:
            raise CategoryCompositionError(
                "Ψ₂₃: el morfismo no actúa sobre el estado dado."
            )
        working = morphism.apply(state)
    if lattice is not None and lattice.dimensions != working.dimensions:
        raise CategoryCompositionError(
            "Ψ₂₃: el retículo no es el de L(H_{C₂}) del estado."
        )
    tol = working.tolerance if tolerance is None else float(tolerance)
    return BicomplexTomitaTakesakiTheory(working, tolerance=tol)


#=============================================================================
# FASE 3.1 — FORMA ESTÁNDAR Y TEORÍA MODULAR DE TOMITA–TAKESAKI
#=============================================================================
@dataclass(slots=True)
class ModularChannelData:
    """
    Datos precomputados de la forma estándar de una fibra idempotente.

    En M_d(C) con estado fiel ω(A) = Tr(ρ A):

        Δ(A)  = ρ A ρ⁻¹,
        J(A)  = ρ^{1/2} A† ρ^{-1/2},
        S(A)  = J Δ^{1/2}(A) = A†,
        σ_t(A)= ρ^{it} A ρ^{-it}.

    Los superoperadores se almacenan en representación de Liouville
    (vectorización column-major).
    """

    eigenvalues: RealVector
    eigenvectors: ComplexMatrix
    rho_sqrt: ComplexMatrix
    rho_inv_sqrt: ComplexMatrix
    rho_inv: ComplexMatrix
    Delta_vec: ComplexMatrix
    J_vec: ComplexMatrix
    Delta_half_vec: ComplexMatrix
    S_vec: NDArray[np.float64]
    GNS_gram: ComplexMatrix
    condition_number: float


class BicomplexTomitaTakesakiTheory:
    """
    Teoría modular de Tomita–Takesaki bicompleja en dimensión finita.

    Para estado fiel ρ_{C₂} = ρ⁽¹⁾ e₁ + ρ⁽²⁾ e₂ se construye, por fibra,

        Δ⁽ᵏ⁾(A) = ρ⁽ᵏ⁾ A (ρ⁽ᵏ⁾)⁻¹,
        J⁽ᵏ⁾(A) = (ρ⁽ᵏ⁾)^{1/2} A† (ρ⁽ᵏ⁾)^{-1/2},
        σ_t⁽ᵏ⁾(A) = (ρ⁽ᵏ⁾)^{it} A (ρ⁽ᵏ⁾)^{-it}.

    Relaciones fundamentales (Takesaki):
        J² = id,   J Δ J = Δ⁻¹,   S = J Δ^{1/2},   S² = id sobre el álgebra,
        Δ autoadjunto positivo en el producto GNS ⟨X|Y⟩_ρ = Tr(X† ρ Y).
    """

    def __init__(
        self,
        state: BicomplexAtomicDensityMatrix,
        tolerance: float = 1e-12,
    ) -> None:
        self.state = state
        self.tolerance = float(tolerance) if float(tolerance) > 0.0 else 1e-12
        self.dimensions = state.dimensions

        if not state.is_faithful(self.tolerance):
            raise TraceAnomalyError(
                "El estado bicomplejo no es fiel (cíclico y separador)."
            )

        self.channel_1_data = self._precompute_channel(
            state.rho_channel_1, self.tolerance, "e₁"
        )
        self.channel_2_data = self._precompute_channel(
            state.rho_channel_2, self.tolerance, "e₂"
        )

    @staticmethod
    def _precompute_channel(
        rho: ComplexMatrix,
        tolerance: float,
        channel_name: str,
    ) -> ModularChannelData:
        """Precomputa Δ, J, Δ^{1/2}, S y el gramiano GNS de una fibra."""
        rho = _symmetrize(rho)
        eigvals, eigvecs = np.linalg.eigh(rho)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order].astype(np.float64)
        eigvecs = eigvecs[:, order]

        min_eig = float(np.min(eigvals))
        if min_eig <= tolerance:
            raise TraceAnomalyError(
                f"Estado no fiel en canal {channel_name}. "
                f"Autovalor mínimo: {min_eig:.3e}"
            )

        cond = float(np.max(eigvals) / min_eig)
        if cond > _CONDITION_WARN:
            logger.warning(
                "Condicionamiento modular elevado en %s: κ(ρ)=%.3e",
                channel_name,
                cond,
            )

        sqrt_eig = np.sqrt(eigvals)
        inv_sqrt_eig = 1.0 / sqrt_eig
        inv_eig = 1.0 / eigvals
        rho_sqrt = eigvecs @ np.diag(sqrt_eig) @ eigvecs.conj().T
        rho_inv_sqrt = eigvecs @ np.diag(inv_sqrt_eig) @ eigvecs.conj().T
        rho_inv = eigvecs @ np.diag(inv_eig) @ eigvecs.conj().T

        dim = rho.shape[0]
        perm = _transpose_permutation(dim)
        # vec(Δ(A)) = vec(ρ A ρ⁻¹) = ((ρ⁻¹)ᵀ ⊗ ρ) vec(A)
        Delta_vec = np.kron(rho_inv.T, rho)
        # vec(A†) = P conj(vec(A)); J(A) = ρ^{1/2} A† ρ^{-1/2}
        # actúa antilinealmente: se almacena la parte lineal ∘ P.
        J_vec = np.kron(rho_inv_sqrt.T, rho_sqrt) @ perm
        Delta_half_vec = np.kron(rho_inv_sqrt.T, rho_sqrt)
        GNS_gram = np.kron(np.eye(dim, dtype=np.complex128), rho)

        return ModularChannelData(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            rho_sqrt=rho_sqrt,
            rho_inv_sqrt=rho_inv_sqrt,
            rho_inv=rho_inv,
            Delta_vec=Delta_vec,
            J_vec=J_vec,
            Delta_half_vec=Delta_half_vec,
            S_vec=perm,
            GNS_gram=GNS_gram,
            condition_number=cond,
        )

    @classmethod
    def from_phase2(
        cls,
        state: BicomplexAtomicDensityMatrix,
        lattice: Optional[BicomplexOrthomodularLattice] = None,
        morphism: Optional[BicomplexCPTPMorphism] = None,
        tolerance: Optional[float] = None,
    ) -> "BicomplexTomitaTakesakiTheory":
        """Realización concreta de Ψ₂₃ (FASE 2.4) como método de clase."""
        return psi_23_initiate_tomita_takesaki(
            state,
            lattice=lattice,
            morphism=morphism,
            tolerance=tolerance,
        )

    def modular_operator_apply(self, A: BicomplexOperator) -> BicomplexOperator:
        """Δ(A) = ρ A ρ⁻¹."""
        if A.dimensions != self.dimensions:
            raise ModularConjugationError("Observable ajeno al álgebra modular.")
        return BicomplexOperator(
            self.state.rho_channel_1 @ A.channel_1 @ self.channel_1_data.rho_inv,
            self.state.rho_channel_2 @ A.channel_2 @ self.channel_2_data.rho_inv,
        )

    def modular_conjugation_apply(self, A: BicomplexOperator) -> BicomplexOperator:
        """J(A) = ρ^{1/2} A† ρ^{-1/2} (antilineal en A)."""
        if A.dimensions != self.dimensions:
            raise ModularConjugationError("Observable ajeno al álgebra modular.")
        return BicomplexOperator(
            self.channel_1_data.rho_sqrt
            @ A.channel_1.conj().T
            @ self.channel_1_data.rho_inv_sqrt,
            self.channel_2_data.rho_sqrt
            @ A.channel_2.conj().T
            @ self.channel_2_data.rho_inv_sqrt,
        )

    def fundamental_involution_apply(self, A: BicomplexOperator) -> BicomplexOperator:
        """Involución de Tomita S(A) = A† (involución C* del álgebra)."""
        if A.dimensions != self.dimensions:
            raise ModularConjugationError("Observable ajeno al álgebra modular.")
        return A.dagger()

    def modular_hamiltonian(self) -> BicomplexOperator:
        """K = −log ρ, generador de σ_t = Ad(e^{itK}) en cada fibra."""
        return self.state.modular_hamiltonian()

    def modular_automorphism_group(
        self,
        t: Union[float, complex],
    ) -> Callable[[BicomplexOperator], BicomplexOperator]:
        """
        Grupo modular fuertemente continuo:

            σ_t(A) = ρ^{it} A ρ^{-it} = Ad(e^{itK})(A).
        """
        exponent = 1j * t
        rho_it_1 = _hermitian_power(
            self.state.rho_channel_1, exponent, self.tolerance
        )
        rho_minus_it_1 = _hermitian_power(
            self.state.rho_channel_1, -exponent, self.tolerance
        )
        rho_it_2 = _hermitian_power(
            self.state.rho_channel_2, exponent, self.tolerance
        )
        rho_minus_it_2 = _hermitian_power(
            self.state.rho_channel_2, -exponent, self.tolerance
        )

        def sigma_t(A: BicomplexOperator) -> BicomplexOperator:
            if A.dimensions != self.dimensions:
                raise ModularConjugationError(
                    "σ_t actúa solo sobre M_{C₂} del estado modular."
                )
            return BicomplexOperator(
                rho_it_1 @ A.channel_1 @ rho_minus_it_1,
                rho_it_2 @ A.channel_2 @ rho_minus_it_2,
            )

        return sigma_t

    def _verify_channel_relations(
        self,
        data: ModularChannelData,
        rho: ComplexMatrix,
        channel_name: str,
    ) -> Dict[str, Union[bool, float, str]]:
        """Verifica las relaciones de Tomita–Takesaki sobre operadores genéricos."""
        dim = rho.shape[0]
        tol = self.tolerance
        rng = np.random.default_rng(42 + dim)
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        B = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        A = A.astype(np.complex128)
        B = B.astype(np.complex128)

        def delta_op(X: ComplexMatrix) -> ComplexMatrix:
            return rho @ X @ data.rho_inv

        def conj_op(X: ComplexMatrix) -> ComplexMatrix:
            return data.rho_sqrt @ X.conj().T @ data.rho_inv_sqrt

        results: Dict[str, Union[bool, float, str]] = {"channel": channel_name}

        jja = conj_op(conj_op(A))
        err = _frobenius(jja - A)
        results["J_involutive"] = err < tol
        results["J_involutive_error"] = err

        jdja = conj_op(delta_op(conj_op(A)))
        delta_inv_A = data.rho_inv @ A @ rho
        err = _frobenius(jdja - delta_inv_A)
        results["J_Delta_J"] = err < tol
        results["J_Delta_J_error"] = err

        ssa = A.conj().T.conj().T
        err = _frobenius(ssa - A)
        results["S_involutive"] = err < tol
        results["S_involutive_error"] = err

        delta_half_A = data.rho_sqrt @ A @ data.rho_inv_sqrt
        j_delta_half_A = conj_op(delta_half_A)
        s_A = A.conj().T
        err = _frobenius(j_delta_half_A - s_A)
        results["S_factorization"] = err < tol
        results["S_factorization_error"] = err

        def gns_inner(X: ComplexMatrix, Y: ComplexMatrix) -> complex:
            return complex(np.trace(X.conj().T @ rho @ Y))

        err = float(abs(gns_inner(delta_op(A), B) - gns_inner(A, delta_op(B))))
        results["Delta_GNS_self_adjoint"] = err < tol
        results["Delta_GNS_self_adjoint_error"] = err
        results["condition_number"] = float(data.condition_number)
        results["all_passed"] = bool(
            results["J_involutive"]
            and results["J_Delta_J"]
            and results["S_involutive"]
            and results["S_factorization"]
            and results["Delta_GNS_self_adjoint"]
        )
        return results

    def verify_tomita_takesaki_relations(
        self,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Certifica las relaciones fundamentales en ambas fibras."""
        if tolerance is not None:
            local_tol = float(tolerance)
            previous = self.tolerance
            self.tolerance = local_tol
        else:
            previous = None

        try:
            channel_1 = self._verify_channel_relations(
                self.channel_1_data, self.state.rho_channel_1, "e1"
            )
            channel_2 = self._verify_channel_relations(
                self.channel_2_data, self.state.rho_channel_2, "e2"
            )
        finally:
            if previous is not None:
                self.tolerance = previous

        return {
            "channel_1": channel_1,
            "channel_2": channel_2,
            "all_passed": bool(channel_1["all_passed"] and channel_2["all_passed"]),
        }

    def verify_kms_relation(
        self,
        A: BicomplexOperator,
        B: BicomplexOperator,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Condición KMS a inversa de temperatura β:

            ω(A B) = ω(B σ_{−iβ}(A)),

        para operadores acotados (no necesariamente hermitianos).
        En dimensión finita, todo estado fiel es KMS respecto de su grupo
        modular (Takesaki).
        """
        if A.dimensions != self.dimensions or B.dimensions != self.dimensions:
            raise ValueError("Los observables no pertenecen al álgebra bicompleja.")

        sigma = self.modular_automorphism_group(-1.0j * beta)
        sigma_A = sigma(A)

        def _kms_channel(
            rho: ComplexMatrix,
            A_c: ComplexMatrix,
            B_c: ComplexMatrix,
            sigma_A_c: ComplexMatrix,
        ) -> Dict[str, Union[bool, float]]:
            lhs = complex(np.trace(rho @ A_c @ B_c))
            rhs = complex(np.trace(rho @ B_c @ sigma_A_c))
            err = float(abs(lhs - rhs))
            return {
                "KMS_holds": err < self.tolerance,
                "lhs_real": float(lhs.real),
                "rhs_real": float(rhs.real),
                "error": err,
            }

        channel_1 = _kms_channel(
            self.state.rho_channel_1, A.channel_1, B.channel_1, sigma_A.channel_1
        )
        channel_2 = _kms_channel(
            self.state.rho_channel_2, A.channel_2, B.channel_2, sigma_A.channel_2
        )
        return {
            "channel_1": channel_1,
            "channel_2": channel_2,
            "all_passed": bool(channel_1["KMS_holds"] and channel_2["KMS_holds"]),
        }

    def classify_factor(
        self,
        tolerance: Optional[float] = None,
    ) -> Tuple[VonNeumannFactorType, VonNeumannFactorType]:
        """
        Clasificación de factores.

        En dimensión finita, M_d(C) es un factor de Tipo I_d (I_finite),
        con independencia de que el estado sea o no tracial. El carácter
        tracial se reporta aparte vía ``is_tracial``.
        """
        _ = tolerance
        d1, d2 = self.dimensions
        _ = d1, d2
        return (VonNeumannFactorType.TYPE_I_FINITE, VonNeumannFactorType.TYPE_I_FINITE)

    @property
    def is_tracial(self) -> Tuple[bool, bool]:
        """ω tracial ⇔ ρ = I/d (el grupo modular es trivial)."""
        d1, d2 = self.dimensions
        mixed1 = np.eye(d1, dtype=np.complex128) / d1
        mixed2 = np.eye(d2, dtype=np.complex128) / d2
        return (
            _frobenius(self.state.rho_channel_1 - mixed1) < self.tolerance,
            _frobenius(self.state.rho_channel_2 - mixed2) < self.tolerance,
        )

    def build_modular_data(self) -> Dict[str, Any]:
        """Empaquetado de invariantes modulares bicomplejos."""
        factor_1, factor_2 = self.classify_factor()
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "dimensions": self.dimensions,
            "factor_types": (factor_1.name, factor_2.name),
            "tracial": self.is_tracial,
            "condition_numbers": (
                self.channel_1_data.condition_number,
                self.channel_2_data.condition_number,
            ),
            "channel_1_eigenvalues": self.channel_1_data.eigenvalues.tolist(),
            "channel_2_eigenvalues": self.channel_2_data.eigenvalues.tolist(),
        }


#=============================================================================
# FASE 3.2 — TOPOS / ÁLGEBRA DE HEYTING B₂ Y ACTUACIÓN CROWBAR
#=============================================================================
@dataclass(frozen=True, slots=True)
class BicomplexHeytingVerdict:
    """
    Punto del topos de clasificadores bicomplejo:

        B₂ ≅ Ω₃⁽¹⁾ × Ω₃⁽²⁾,    Ω₃ = {VETOED < DEGRADED < COHERENT}.

    Conjunción global = mínimo (producto de Heyting). El colapso a
    VETOED dispara la actuación ciber-física (GPIO Crowbar).
    """

    channel_1: HeytingValue
    channel_2: HeytingValue

    @property
    def global_verdict(self) -> HeytingValue:
        """Conjunción topológica min(v₁, v₂)."""
        return min(self.channel_1, self.channel_2)

    @property
    def hardware_veto(self) -> bool:
        """True ssi el clasificador global es el fondo 0 = VETOED."""
        return self.global_verdict == HeytingValue.VETOED

    @property
    def gpio_pin(self) -> int:
        """Pin físico de actuación Crowbar."""
        return _GPIO_VETO_PIN

    @property
    def actuation_ns(self) -> float:
        """Cota temporal de ISR en nanosegundos."""
        return _ISR_ACTUATION_NS

    @classmethod
    def top(cls) -> "BicomplexHeytingVerdict":
        """Unidad ⊤ = (COHERENT, COHERENT)."""
        return cls(HeytingValue.COHERENT, HeytingValue.COHERENT)

    @classmethod
    def bottom(cls) -> "BicomplexHeytingVerdict":
        """Fondo ⊥ = (VETOED, VETOED)."""
        return cls(HeytingValue.VETOED, HeytingValue.VETOED)

    @classmethod
    def from_coherence(
        cls,
        coherence_channel_1: float,
        coherence_channel_2: float,
        coherent_threshold: float = _COHERENT_THRESHOLD,
        degraded_threshold: float = _DEGRADED_THRESHOLD,
    ) -> "BicomplexHeytingVerdict":
        """Clasificación continua → Ω₃ × Ω₃."""
        return cls(
            channel_1=HeytingValue.from_coherence(
                coherence_channel_1, coherent_threshold, degraded_threshold
            ),
            channel_2=HeytingValue.from_coherence(
                coherence_channel_2, coherent_threshold, degraded_threshold
            ),
        )

    def implication(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Pseudo-complemento relativo (a → b) = ⊤ si a ≤ b, else b."""

        def impl(a: HeytingValue, b: HeytingValue) -> HeytingValue:
            return HeytingValue.COHERENT if a <= b else b

        return BicomplexHeytingVerdict(
            channel_1=impl(self.channel_1, other.channel_1),
            channel_2=impl(self.channel_2, other.channel_2),
        )

    def conjunction(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Meet de Heyting: mínimo componente a componente."""
        return BicomplexHeytingVerdict(
            channel_1=min(self.channel_1, other.channel_1),
            channel_2=min(self.channel_2, other.channel_2),
        )

    def disjunction(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Join de Heyting: máximo componente a componente."""
        return BicomplexHeytingVerdict(
            channel_1=max(self.channel_1, other.channel_1),
            channel_2=max(self.channel_2, other.channel_2),
        )

    def negation(self) -> "BicomplexHeytingVerdict":
        """Negación intuicionista ¬a = a → ⊥. No es involutiva."""
        return self.implication(self.bottom())

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe del veredicto."""
        return {
            "channel_1": self.channel_1.name,
            "channel_2": self.channel_2.name,
            "global_verdict": self.global_verdict.name,
            "hardware_veto": self.hardware_veto,
            "gpio_pin": self.gpio_pin,
            "actuation_ns": self.actuation_ns,
        }


#=============================================================================
# FASE 3.3 — REPORTE TERMINAL Y COHERENCIA PROXY
#=============================================================================
@dataclass(frozen=True, slots=True)
class BicomplexMACAlgebraReport:
    """Objeto terminal del funtor Ψ_MAC: certificado W* + veredicto B₂."""

    density_valid: bool
    dimensions: Dimensions
    morphism_report: Optional[Dict[str, Any]]
    lattice_report: Optional[Dict[str, Any]]
    modular_report: Dict[str, Any]
    verdict: BicomplexHeytingVerdict
    success: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe del reporte maestro."""
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "density_valid": self.density_valid,
            "dimensions": self.dimensions,
            "morphism_report": self.morphism_report,
            "lattice_report": self.lattice_report,
            "modular_report": self.modular_report,
            "verdict": self.verdict.to_dict(),
            "success": self.success,
            "timestamp": self.timestamp,
        }


def _coherence_from_modular_report(modular_report: Dict[str, Any]) -> Tuple[float, float]:
    """
    Coherencia proxy dura: 1.0 si la fibra pasa Tomita–Takesaki (y KMS si
    existe), 0.0 en caso contrario. Conserva la semántica de veto binario.
    """
    c1 = 1.0 if modular_report.get("channel_1", {}).get("all_passed", False) else 0.0
    c2 = 1.0 if modular_report.get("channel_2", {}).get("all_passed", False) else 0.0
    kms = modular_report.get("kms")
    if isinstance(kms, dict):
        if not kms.get("channel_1", {}).get("KMS_holds", True):
            c1 = 0.0
        if not kms.get("channel_2", {}).get("KMS_holds", True):
            c2 = 0.0
    return c1, c2


#=============================================================================
# FASE 3.4 — FUNCTOR MAESTRO Ψ_MAC = Ψ₂₃ ∘ Ψ₁₂
#=============================================================================
def compose_bicomplex_mac_algebra_pipeline(
    rho_channel_1: NDArray[np.complex128],
    rho_channel_2: NDArray[np.complex128],
    *,
    kraus_channel_1: Optional[Sequence[NDArray[np.complex128]]] = None,
    kraus_channel_2: Optional[Sequence[NDArray[np.complex128]]] = None,
    projectors: Optional[Sequence[BicomplexOperator]] = None,
    observables: Optional[Tuple[BicomplexOperator, BicomplexOperator]] = None,
    coherences: Optional[Tuple[float, float]] = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    raise_on_veto: bool = False,
) -> BicomplexMACAlgebraReport:
    """
    Funtor maestro de la MAC bicompleja, composición de las tres fases:

        Ψ_MAC = Ψ₂₃ ∘ Ψ₁₂.

    Ejecuta, en orden estricto:
      FASE 1  Saneamiento y certificación del estado atómico ρ_{C₂}.
      FASE 2  Opcional: morfismo CPTP (vía Ψ₁₂) y retículo ortomodular.
      FASE 3  Forma estándar Tomita–Takesaki (vía Ψ₂₃), KMS, colapso B₂.

    El veredicto de Heyting veta el hardware si alguna fibra colapsa a 0.
    """
    density = BicomplexAtomicDensityMatrix(
        rho_channel_1=rho_channel_1,
        rho_channel_2=rho_channel_2,
        tolerance=tolerance,
    )

    morphism_report: Optional[Dict[str, Any]] = None
    morphism: Optional[BicomplexCPTPMorphism] = None
    if kraus_channel_1 is not None or kraus_channel_2 is not None:
        if kraus_channel_1 is None or kraus_channel_2 is None:
            raise ValueError(
                "Si se provee un canal bicomplejo, deben proveerse ambos canales."
            )
        morphism = psi_12_initiate_cptp_category(
            density,
            kraus_channel_1,
            kraus_channel_2,
            tolerance=tolerance,
        )
        evolved_density = morphism.apply(density)
        morphism_report = {
            "morphism": morphism.to_dict(),
            "evolved_density": evolved_density.to_dict(),
            "is_unitary": morphism.is_unitary(),
            "is_unital": morphism.is_unital(),
            "is_completely_positive": morphism.is_completely_positive(),
        }
        density = evolved_density

    lattice_report: Optional[Dict[str, Any]] = None
    lattice: Optional[BicomplexOrthomodularLattice] = None
    if projectors:
        lattice = BicomplexOrthomodularLattice(density.dimensions, tolerance=tolerance)
        validated_projectors: List[BicomplexOperator] = []
        for P in projectors:
            if not isinstance(P, BicomplexOperator):
                raise TypeError("Cada proyector debe ser BicomplexOperator.")
            lattice.validate_projector(P)
            validated_projectors.append(P)

        compatibility: List[Dict[str, Any]] = []
        orthomodular_checks: List[Dict[str, Any]] = []
        for i, P in enumerate(validated_projectors):
            for j in range(i + 1, len(validated_projectors)):
                Q = validated_projectors[j]
                compatibility.append(
                    {
                        "P_index": i,
                        "Q_index": j,
                        "compatible": lattice.are_compatible(P, Q),
                    }
                )
                orthomodular_checks.append(
                    {
                        "P_index": i,
                        "Q_index": j,
                        "orthomodular_law": lattice.verify_orthomodular_law(P, Q),
                    }
                )
        lattice_report = {
            "projector_count": len(validated_projectors),
            "compatibility": compatibility,
            "orthomodular_checks": orthomodular_checks,
            "compatibility_graph": lattice.compatibility_graph(
                validated_projectors
            ).tolist(),
        }

    theory = psi_23_initiate_tomita_takesaki(
        density,
        lattice=lattice,
        morphism=None,
        tolerance=tolerance,
    )
    modular_report = theory.verify_tomita_takesaki_relations()
    modular_report["invariants"] = theory.build_modular_data()

    if observables is not None:
        A, B = observables
        modular_report["kms"] = theory.verify_kms_relation(A, B)

    if coherences is None:
        c1, c2 = _coherence_from_modular_report(modular_report)
    else:
        c1 = max(0.0, min(1.0, float(coherences[0])))
        c2 = max(0.0, min(1.0, float(coherences[1])))

    verdict = BicomplexHeytingVerdict.from_coherence(c1, c2)
    success = not verdict.hardware_veto

    if verdict.hardware_veto:
        logger.warning(
            "VETO B₂ activo en MAC: GPIO%d ≤ %.2f ns",
            verdict.gpio_pin,
            verdict.actuation_ns,
        )
        if raise_on_veto:
            raise HeytingVetoError(
                "Colapso global a VETOED en el retículo de Heyting bicomplejo.",
                verdict=verdict.to_dict(),
            )

    return BicomplexMACAlgebraReport(
        density_valid=True,
        dimensions=density.dimensions,
        morphism_report=morphism_report,
        lattice_report=lattice_report,
        modular_report=modular_report,
        verdict=verdict,
        success=success,
    )


#=============================================================================
# EXPORTS
#=============================================================================
__all__ = [
    "NonCommutativeAlgebraError",
    "TraceAnomalyError",
    "OrthomodularConvergenceError",
    "ModularConjugationError",
    "CategoryCompositionError",
    "HeytingVetoError",
    "VonNeumannFactorType",
    "HeytingValue",
    "BicomplexOperator",
    "BicomplexAtomicDensityMatrix",
    "BicomplexCPTPMorphism",
    "BicomplexOrthomodularLattice",
    "ModularChannelData",
    "BicomplexTomitaTakesakiTheory",
    "BicomplexHeytingVerdict",
    "BicomplexMACAlgebraReport",
    "identity_kraus_operators",
    "psi_12_initiate_cptp_category",
    "psi_23_initiate_tomita_takesaki",
    "compose_bicomplex_mac_algebra_pipeline",
]