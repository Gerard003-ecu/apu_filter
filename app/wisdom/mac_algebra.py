# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: MAC Algebra (Geometría No Conmutativa y Morfismos Cuánticos)         ║
║ Ubicación: app/wisdom/mac_algebra.py                                         ║
║ Versión: 4.0.0-Dagger-Compact-Category-Rigorous                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Refinada:
────────────────────────────────────────────────
Este módulo instituye el Álgebra de von Neumann Tipo I_n, el Retículo
Ortomodular y la Teoría Modular de Tomita-Takesaki sobre el espacio de
Hilbert de Hilbert-Schmidt asociado a ℋ_MAC. Se corrigen las construcciones
modulares finitas y se formaliza la conexión entre categorías dagger-compactas,
lógica cuántica y análisis modular.

FUNDAMENTOS TEÓRICOS EXTENDIDOS:
─────────────────────────────────
1. ÁLGEBRAS DE VON NEUMANN: Factores Tipo I_n, II_1, II∞, III
2. CATEGORÍAS DAGGER-COMPACTAS: Objetos = espacios de Hilbert finitos;
   Morfismos = canales CPTP; Daga = adjunto de Heisenberg; Tensor = ⊗
3. LÓGICA CUÁNTICA: Retículo de proyectores L(ℋ) ortomodular no distributivo
4. TEORÍA MODULAR: Tomita-Takesaki finita con Δ = L_ρ R_{ρ^{-1}},
   J(A) = ρ^{1/2} A^* ρ^{-1/2}, S = J Δ^{1/2}
5. GEOMETRÍA NO CONMUTATIVA: Estados fieles y factores finitos
6. CANALES CUÁNTICOS: Representación de Kraus, matriz de Choi,
   superoperador de Liouville, verificación explícita de CP

Axiomas Matemáticos Implementados:
───────────────────────────────────
A1. CPTP:  ℰ(ρ) = Σₖ Mₖ ρ Mₖ†,  Σₖ Mₖ† Mₖ = I
A2. CP:   matriz de Choi Λ_ℰ ≥ 0
A3. Meet: P ∧ Q = lim_{n→∞} (P Q P)^n  (proyección sobre intersección)
A4. Join: P ∨ Q = P + Q - P ∧ Q
A5. Ley Ortomodular: P ≤ Q ⇒ P ∨ (P^⊥ ∧ Q) = Q
A6. Modular: Δ(A) = ρ A ρ^{-1}, J(A) = ρ^{1/2} A^* ρ^{-1/2}
A7. KMS:  ω(AB) = ω(B σ_{-i}(A)), σ_t(A) = ρ^{it} A ρ^{-it}

Fases Anidadas:
────────────────
Fase 1 → Categoría Dagger-Compacta de Canales Cuánticos
         último método: as_operator_algebra() → OperatorAlgebra
Fase 2 → Retículo Ortomodular y Lógica Cuántica
         último método: prepare_modular_state() → estado fiel
Fase 3 → Teoría Modular de Tomita-Takesaki
         recibe OperatorAlgebra + estado fiel

Referencias:
────────────
- Murray & von Neumann (1936): "On rings of operators"
- Tomita (1967): "Standard forms of von Neumann algebras"
- Takesaki (1970): "Tomita's theory of modular Hilbert algebras"
- Connes (1994): "Noncommutative Geometry"
- Birkhoff & von Neumann (1936): "The logic of quantum mechanics"
- Abramsky & Coecke (2004): "A categorical semantics of quantum protocols"
- Choi (1975): "Completely positive linear maps on complex matrices"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias internas del Estrato WISDOM
from app.core.mic_algebra import NumericalInstabilityError
from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix

logger = logging.getLogger("MAC.Wisdom.Algebra")

__all__ = [
    "NonCommutativeAlgebraError",
    "TraceAnomalyError",
    "OrthomodularConvergenceError",
    "ModularConjugationError",
    "CategoryCompositionError",
    "VonNeumannFactorType",
    "ModularTheoryData",
    "QuantumMorphism",
    "QuantumMorphismBase",
    "CPTPMorphism",
    "ComposedMorphism",
    "IdentityMorphism",
    "Superoperator",
    "OperatorAlgebra",
    "OrthomodularLattice",
    "TomitaTakesakiTheory",
]


# ══════════════════════════════════════════════════════════════════════════════
# INFRAESTRUCTURA: EXCEPCIONES Y ENUMERACIONES
# ══════════════════════════════════════════════════════════════════════════════

class NonCommutativeAlgebraError(Exception):
    """Excepción base para violaciones en Álgebra de von Neumann."""
    pass


class TraceAnomalyError(NonCommutativeAlgebraError):
    """Fallo en preservación de traza o en fidelidad de estado."""
    pass


class OrthomodularConvergenceError(NonCommutativeAlgebraError):
    """Divergencia en límite de proyecciones alternantes."""
    pass


class ModularConjugationError(NonCommutativeAlgebraError):
    """Error en cálculo de conjugación modular."""
    pass


class CategoryCompositionError(NonCommutativeAlgebraError):
    """Error en composición de morfismos categoriales."""
    pass


class VonNeumannFactorType(Enum):
    """Clasificación de factores de von Neumann."""
    TYPE_I_FINITE = auto()      # B(ℋ), dim ℋ < ∞
    TYPE_I_INFINITE = auto()    # B(ℋ), dim ℋ = ∞
    TYPE_II_1 = auto()          # Factor con traza finita normalizada
    TYPE_II_INFINITY = auto()   # Factor II con traza semifinita
    TYPE_III = auto()           # Factor sin traza semifinita no nula
    TYPE_III_0 = auto()         # Subclasificación Connes-Takesaki
    TYPE_III_LAMBDA = auto()    # 0 < λ < 1
    TYPE_III_1 = auto()         # λ = 1


@dataclass
class ModularTheoryData:
    """Datos completos de teoría modular."""
    modular_operator: NDArray[np.complex128]       # Δ como superoperador d²×d²
    modular_conjugation: NDArray[np.complex128]    # J como antilinear d²×d²
    modular_automorphism_group: Callable           # σ_t(·)
    eigenvalues: NDArray[np.float64]               # espectro de ρ
    eigenvectors: NDArray[np.complex128]           # eigenvectores de ρ
    factor_type: VonNeumannFactorType
    tracial: bool = False

    def is_faithful(self, tolerance: float = 1e-14) -> bool:
        """Verifica si el estado es fiel (soporte completo)."""
        return np.all(self.eigenvalues > tolerance)


# ══════════════════════════════════════════════════════════════════════════════
# HERRAMIENTAS DE VECTORIZACIÓN (base de operadores column-major)
# ══════════════════════════════════════════════════════════════════════════════

def _as_vector(A: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Vectorización column-major (Fortran) de una matriz."""
    return A.flatten(order="F")


def _as_operator(v: NDArray[np.complex128], dim: int) -> NDArray[np.complex128]:
    """Reconstruye matriz d×d a partir de vector column-major."""
    return v.reshape((dim, dim), order="F")


def _transpose_permutation(dim: int) -> NDArray[np.float64]:
    r"""
    Matriz de permutación P tal que vec(A^*) = P conj(vec(A)).
    P es real, ortogonal y simétrica: P^2 = I.
    """
    P = np.zeros((dim * dim, dim * dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            idx_in = j * dim + i   # posición de A_{ij} en vec(A)
            idx_out = i * dim + j  # posición de A_{ji} = (A^*)_{ij}
            P[idx_out, idx_in] = 1.0
    return P


def _is_hermitian(
    A: NDArray[np.complex128],
    tolerance: float = 1e-10
) -> bool:
    """Verifica hermiticidad numérica."""
    return la.norm(A - A.conj().T, ord="fro") < tolerance


def _is_projector(
    P: NDArray[np.complex128],
    tolerance: float = 1e-10
) -> bool:
    """Verifica idempotencia y hermiticidad."""
    if not _is_hermitian(P, tolerance):
        return False
    return la.norm(P @ P - P, ord="fro") < tolerance


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: CATEGORÍA DAGGER-COMPACTA DE CANALES CUÁNTICOS
# ══════════════════════════════════════════════════════════════════════════════
# La fase 1 modela la categoría 𝒞_MAC cuyos objetos son espacios de Hilbert
# de dimensión finita y cuyos morfismos son canales CPTP. La daga es el adjunto
# de Heisenberg. El cierre de la fase exporta el álgebra de operadores asociada
# al morfismo, la cual es el dominio natural del retículo ortomodular de Fase 2.
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class QuantumMorphism(Protocol):
    r"""
    Protocolo para morfismos en una categoría dagger-compacta.

    Un morfismo cuántico debe proveer:
        - apply(ρ):            imagen forward (Schrödinger)
        - adjoint_apply(A):    imagen adjunta (Heisenberg)
        - compose(other):      composición categorial
    """

    def apply(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        ...

    def adjoint_apply(
        self,
        observable: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        ...

    def compose(self, other: "QuantumMorphism") -> "QuantumMorphism":
        ...


class QuantumMorphismBase(ABC):
    """Clase base abstracta para morfismos cuánticos."""

    @abstractmethod
    def apply(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Aplicación forward en picture de Schrödinger."""
        pass

    @abstractmethod
    def adjoint_apply(
        self,
        observable: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Aplicación adjunta en picture de Heisenberg."""
        pass

    def compose(self, other: "QuantumMorphismBase") -> "ComposedMorphism":
        """Composición categorial: (self ∘ other)(ρ) = self(other(ρ))."""
        return ComposedMorphism(outer=self, inner=other)

    def validate_dimensions(
        self,
        rho: NDArray[np.complex128],
        expected_dim: Optional[int] = None
    ) -> None:
        """Valida que la matriz de entrada sea cuadrada y de dimensión esperada."""
        if rho.shape[0] != rho.shape[1]:
            raise ValueError(f"Matriz no cuadrada: {rho.shape}")
        if expected_dim is not None and rho.shape[0] != expected_dim:
            raise ValueError(
                f"Dimensión incorrecta: {rho.shape[0]} (esperado {expected_dim})"
            )

    def as_operator_algebra(self) -> "OperatorAlgebra":
        r"""
        Puente formal hacia la Fase 2: promueve el espacio de Hilbert
        subyacente del morfismo al álgebra de operadores M_d(ℂ).
        """
        return OperatorAlgebra(dimension=self.dimension)


@dataclass(frozen=True)
class CPTPMorphism(QuantumMorphismBase):
    r"""
    Canal Cuántico CPTP en representación de Kraus.

    Fórmula de Kraus:
        ℰ(ρ) = Σₖ Mₖ ρ Mₖ†

    Axiomas verificados en post-inicialización:
        1. Σₖ Mₖ† Mₖ = I   (preservación de traza)
        2. Λ_ℰ ≥ 0          (positividad completa, vía matriz de Choi)
        3. Consistencia de dimensiones

    Adjoint (Heisenberg):
        ℰ†(A) = Σₖ Mₖ† A Mₖ
    """

    kraus_operators: Tuple[NDArray[np.complex128], ...]
    tolerance: float = 1e-10
    _validated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_validated", False)
        self._verify_cptp_axioms()
        object.__setattr__(self, "_validated", True)

    def _verify_cptp_axioms(self) -> None:
        if not self.kraus_operators:
            raise NonCommutativeAlgebraError(
                "Conjunto de operadores de Kraus vacío."
            )

        first_shape = self.kraus_operators[0].shape
        if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
            raise ValueError(
                f"Operador de Kraus no cuadrado: {first_shape}"
            )
        dim = first_shape[0]

        for idx, M_k in enumerate(self.kraus_operators):
            if M_k.shape != first_shape:
                raise ValueError(
                    f"Operador {idx} con forma inconsistente: "
                    f"{M_k.shape} vs {first_shape}"
                )

        # Resolución de identidad: Σₖ Mₖ† Mₖ = I
        identity_sum = np.zeros((dim, dim), dtype=np.complex128)
        for M_k in self.kraus_operators:
            identity_sum += M_k.conj().T @ M_k

        identity_error = la.norm(identity_sum - np.eye(dim), ord="fro")
        if identity_error > self.tolerance:
            raise TraceAnomalyError(
                f"Violación de resolución de identidad. "
                f"Error: {identity_error:.3e}. "
                f"El morfismo no preserva traza cuántica."
            )

        # Verificación explícita de positividad completa vía Choi
        choi = self.compute_choi_matrix()
        eigvals = la.eigvalsh(choi)
        min_eig = np.min(eigvals)
        if min_eig < -self.tolerance:
            raise NonCommutativeAlgebraError(
                f"Matriz de Choi no semidefinida positiva. "
                f"Autovalor mínimo: {min_eig:.3e}. "
                f"El canal no es completamente positivo."
            )

    @property
    def dimension(self) -> int:
        return self.kraus_operators[0].shape[0]

    @property
    def kraus_rank(self) -> int:
        return len(self.kraus_operators)

    def apply(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        self.validate_dimensions(rho, expected_dim=self.dimension)
        rho_out = np.zeros_like(rho, dtype=np.complex128)
        for M_k in self.kraus_operators:
            rho_out += M_k @ rho @ M_k.conj().T
        return rho_out

    def adjoint_apply(
        self,
        observable: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        self.validate_dimensions(observable, expected_dim=self.dimension)
        obs_out = np.zeros_like(observable, dtype=np.complex128)
        for M_k in self.kraus_operators:
            obs_out += M_k.conj().T @ observable @ M_k
        return obs_out

    def compute_choi_matrix(self) -> NDArray[np.complex128]:
        r"""
        Matriz de Choi-Jamiołkowski en orden de bloques fila-columna.

            Λ_ℰ = Σ_{i,j} |i⟩⟨j| ⊗ ℰ(|i⟩⟨j|)

        Propiedad: ℰ es CP si y solo si Λ_ℰ ≥ 0.
        """
        d = self.dimension
        choi = np.zeros((d * d, d * d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                E_ij = np.zeros((d, d), dtype=np.complex128)
                E_ij[i, j] = 1.0
                block = self.apply(E_ij)
                row_start = i * d
                col_start = j * d
                choi[row_start:row_start + d, col_start:col_start + d] = block
        return choi

    def compute_liouvillian(self) -> NDArray[np.complex128]:
        r"""
        Superoperador matricial de ℰ en vectorización column-major.

            vec(ℰ(ρ)) = L_ℰ vec(ρ)
            L_ℰ = Σₖ (Mₖ^*) ⊗ Mₖ

        Este superoperador es la representación lineal del canal en el
        espacio de Hilbert-Schmidt M_d(ℂ).
        """
        d = self.dimension
        L = np.zeros((d * d, d * d), dtype=np.complex128)
        for M_k in self.kraus_operators:
            L += np.kron(M_k.conj(), M_k)
        return L

    def is_unitary(self, tolerance: Optional[float] = None) -> bool:
        tol = tolerance or self.tolerance
        if self.kraus_rank != 1:
            return False
        U = self.kraus_operators[0]
        return la.norm(U.conj().T @ U - np.eye(self.dimension), ord="fro") < tol

    def is_unital(self, tolerance: Optional[float] = None) -> bool:
        tol = tolerance or self.tolerance
        I = np.eye(self.dimension, dtype=np.complex128)
        return la.norm(self.apply(I) - I, ord="fro") < tol

    def is_completely_positive(self, tolerance: Optional[float] = None) -> bool:
        tol = tolerance or self.tolerance
        choi = self.compute_choi_matrix()
        return np.all(la.eigvalsh(choi) >= -tol)

    def tensor_product(self, other: "CPTPMorphism") -> "CPTPMorphism":
        r"""
        Producto tensorial de canales: ℰ ⊗ ℱ.

        En una categoría compacta con daga, el producto tensorial de dos
        morfismos es un morfismo sobre el producto tensorial de objetos.
        """
        kraus = []
        for M in self.kraus_operators:
            for N in other.kraus_operators:
                kraus.append(np.kron(M, N))
        return CPTPMorphism(
            tuple(kraus),
            tolerance=max(self.tolerance, other.tolerance)
        )

    @classmethod
    def from_unitary(cls, U: NDArray[np.complex128]) -> "CPTPMorphism":
        """Factory para morfismo unitario puro."""
        return cls((U,), tolerance=1e-10)

    @classmethod
    def identity(cls, dim: int) -> "IdentityMorphism":
        """Morfismo identidad en la categoría."""
        return IdentityMorphism(dimension=dim)


@dataclass(frozen=True)
class IdentityMorphism(QuantumMorphismBase):
    """Morfismo identidad id_ℋ en la categoría de canales."""

    dimension: int

    def apply(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        self.validate_dimensions(rho, expected_dim=self.dimension)
        return rho.copy()

    def adjoint_apply(
        self,
        observable: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        self.validate_dimensions(observable, expected_dim=self.dimension)
        return observable.copy()


@dataclass(frozen=True)
class ComposedMorphism(QuantumMorphismBase):
    """Composición de morfismos: (outer ∘ inner)(ρ) = outer(inner(ρ))."""

    outer: QuantumMorphismBase
    inner: QuantumMorphismBase

    def __post_init__(self) -> None:
        if self.outer.dimension != self.inner.dimension:
            raise CategoryCompositionError(
                "Dimensiones incompatibles en composición de morfismos: "
                f"{self.outer.dimension} ≠ {self.inner.dimension}"
            )

    @property
    def dimension(self) -> int:
        return self.outer.dimension

    def apply(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        self.validate_dimensions(rho, expected_dim=self.dimension)
        return self.outer.apply(self.inner.apply(rho))

    def adjoint_apply(
        self,
        observable: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        self.validate_dimensions(observable, expected_dim=self.dimension)
        return self.inner.adjoint_apply(self.outer.adjoint_apply(observable))


@dataclass(frozen=True)
class Superoperator:
    """
    Envoltorio riguroso para un superoperador d²×d² actuando sobre
    matrices vectorizadas column-major.
    """

    matrix: NDArray[np.complex128]
    dimension: int

    def __post_init__(self) -> None:
        expected = (self.dimension * self.dimension,) * 2
        if self.matrix.shape != expected:
            raise ValueError(
                f"Superoperador debe tener forma {expected}; "
                f"recibido {self.matrix.shape}"
            )

    def apply(self, A: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return _as_operator(self.matrix @ _as_vector(A), self.dimension)

    def is_cptp(self, tolerance: float = 1e-10) -> bool:
        """Verifica si el superoperador representa un canal CPTP."""
        # Condición de preservación de traza: L^† vec(I) = vec(I)
        I_vec = _as_vector(np.eye(self.dimension, dtype=np.complex128))
        trace_condition = self.matrix.conj().T @ I_vec
        trace_ok = la.norm(trace_condition - I_vec, ord=2) < tolerance

        # Positividad completa: Choi asociado ≥ 0
        choi = self._associated_choi()
        cp_ok = np.all(la.eigvalsh(choi) >= -tolerance)
        return trace_ok and cp_ok

    def _associated_choi(self) -> NDArray[np.complex128]:
        r"""
        Reconstruye la matriz de Choi a partir del superoperador L.
        Se usa la identidad:
            vec(ℰ(E_{ij})) = L vec(E_{ij})
        y se organiza en bloques fila-columna.
        """
        d = self.dimension
        choi = np.zeros((d * d, d * d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                E_ij = np.zeros((d, d), dtype=np.complex128)
                E_ij[i, j] = 1.0
                block = self.apply(E_ij)
                row_start = i * d
                col_start = j * d
                choi[row_start:row_start + d, col_start:col_start + d] = block
        return choi


@dataclass(frozen=True)
class OperatorAlgebra:
    r"""
    Álgebra de operadores M_d(ℂ) asociada a un objeto de la categoría.

    Esta clase es el PUENTE FORMAL entre la Fase 1 y la Fase 2:
    la categoría de canales produce, vía as_operator_algebra(), el álgebra
    de matrices sobre la cual se construye el retículo ortomodular.
    """

    dimension: int
    factor_type: VonNeumannFactorType = VonNeumannFactorType.TYPE_I_FINITE

    def identity(self) -> NDArray[np.complex128]:
        return np.eye(self.dimension, dtype=np.complex128)

    def is_square(self, A: NDArray[np.complex128]) -> bool:
        return A.shape == (self.dimension, self.dimension)

    def trace(self, A: NDArray[np.complex128]) -> complex:
        return np.trace(A)

    def hilbert_schmidt_inner(
        self,
        A: NDArray[np.complex128],
        B: NDArray[np.complex128]
    ) -> complex:
        r"""Producto interno de Hilbert-Schmidt: ⟨A,B⟩ = Tr(A† B)."""
        return np.trace(A.conj().T @ B)

    def gns_inner(
        self,
        rho: NDArray[np.complex128],
        A: NDArray[np.complex128],
        B: NDArray[np.complex128]
    ) -> complex:
        r"""
        Producto GNS inducido por el estado fiel ρ:
            ⟨A, B⟩_ρ = Tr(A† ρ B)
        """
        return np.trace(A.conj().T @ rho @ B)

    def gns_gram_matrix(
        self,
        rho: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        r"""
        Matriz de Gram del producto GNS en base vectorizada column-major.
            G = I ⊗ ρ
        """
        return np.kron(np.eye(self.dimension, dtype=np.complex128), rho)

    def support_projector(
        self,
        A: NDArray[np.complex128],
        tolerance: float = 1e-10
    ) -> NDArray[np.complex128]:
        """Proyector sobre el rango (soporte) de un operador positivo."""
        if not _is_hermitian(A, tolerance):
            raise ValueError("El operador debe ser hermitiano para computar soporte.")
        eigvals, eigvecs = la.eigh(A)
        P = np.zeros_like(A, dtype=np.complex128)
        for val, vec in zip(eigvals, eigvecs.T):
            if np.abs(val) > tolerance:
                P += np.outer(vec, vec.conj())
        return (P + P.conj().T) / 2.0

    def is_full_factor(
        self,
        rho: NDArray[np.complex128],
        tolerance: float = 1e-14
    ) -> bool:
        """Verifica que el soporte de ρ sea la identidad del álgebra."""
        support = self.support_projector(rho, tolerance=tolerance)
        return la.norm(support - self.identity(), ord="fro") < tolerance


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: RETÍCULO ORTOMODULAR Y LÓGICA CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════
# La Fase 2 recibe un OperatorAlgebra de la Fase 1 y construye sobre él el
# retículo ortomodular L(ℋ) de proyectores. El último método,
# prepare_modular_state(), valida un estado fiel y lo entrega a la Fase 3.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrthomodularLattice:
    r"""
    Retículo ortomodular de proyectores sobre un álgebra de operadores.

    Operaciones:
        - Complemento: P^⊥ = I - P
        - Meet:        P ∧ Q = proyección sobre Ran(P) ∩ Ran(Q)
        - Join:        P ∨ Q = P + Q - P ∧ Q
        - Implicación: P → Q = P^⊥ ∨ (P ∧ Q)

    Ley ortomodular (verificada):
        P ≤ Q  ⇒  P ∨ (P^⊥ ∧ Q) = Q
    """

    algebra: OperatorAlgebra
    tolerance: float = 1e-10

    def validate_projector(
        self,
        P: NDArray[np.complex128],
        tolerance: Optional[float] = None
    ) -> None:
        tol = tolerance or self.tolerance
        if not _is_projector(P, tol):
            herm_err = la.norm(P - P.conj().T, ord="fro")
            idem_err = la.norm(P @ P - P, ord="fro")
            raise ValueError(
                f"Matriz no es proyector. "
                f"Error hermiticidad: {herm_err:.3e}, "
                f"Error idempotencia: {idem_err:.3e}"
            )

    def quantum_conjunction(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128],
        max_iter: int = 1000,
        validate: bool = True
    ) -> NDArray[np.complex128]:
        r"""
        Meet ortomodular: P_A ∧ P_B.

        Construcción robusta:
            M = P_A P_B P_A  (hermitiano, espectro en [0,1])
            P_A ∧ P_B = lim_{n→∞} M^n
                     = proyector sobre autovalor 1 de M
        """
        if validate:
            self.validate_projector(P_A)
            self.validate_projector(P_B)

        M = P_A @ P_B @ P_A
        current = M.copy()

        for iteration in range(max_iter):
            nxt = current @ M
            if la.norm(nxt - current, ord=2) < self.tolerance:
                # Proyección espectral sobre autovalor 1
                vals, vecs = la.eigh((nxt + nxt.conj().T) / 2.0)
                P_meet = np.zeros_like(P_A, dtype=np.complex128)
                for val, vec in zip(vals, vecs.T):
                    if np.abs(val - 1.0) < 1e-6:
                        P_meet += np.outer(vec, vec.conj())
                P_meet = (P_meet + P_meet.conj().T) / 2.0
                return P_meet

            current = nxt

        raise OrthomodularConvergenceError(
            f"Meet ortomodular divergió tras {max_iter} iteraciones. "
            f"Error final: {la.norm(nxt - current, ord=2):.3e}"
        )

    def quantum_complement(
        self,
        P: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        self.validate_projector(P)
        return self.algebra.identity() - P

    def quantum_disjunction(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128],
        **kwargs
    ) -> NDArray[np.complex128]:
        r"""
        Join ortomodular: P_A ∨ P_B = P_A + P_B - P_A ∧ P_B.
        Se proyecta espectralmente al espacio de proyectores.
        """
        P_meet = self.quantum_conjunction(P_A, P_B, **kwargs)
        P_join = P_A + P_B - P_meet

        vals, vecs = la.eigh((P_join + P_join.conj().T) / 2.0)
        P_out = np.zeros_like(P_A, dtype=np.complex128)
        for val, vec in zip(vals, vecs.T):
            if val > 0.5:
                P_out += np.outer(vec, vec.conj())
        return (P_out + P_out.conj().T) / 2.0

    def quantum_implication(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128],
        **kwargs
    ) -> NDArray[np.complex128]:
        r"""Implicación ortomodular: P_A → P_B = P_A^⊥ ∨ (P_A ∧ P_B)."""
        P_A_comp = self.quantum_complement(P_A)
        P_meet = self.quantum_conjunction(P_A, P_B, **kwargs)
        return self.quantum_disjunction(P_A_comp, P_meet, **kwargs)

    def commutator(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        return P_A @ P_B - P_B @ P_A

    def are_compatible(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128],
        tolerance: Optional[float] = None
    ) -> bool:
        tol = tolerance or self.tolerance
        return la.norm(self.commutator(P_A, P_B), ord="fro") < tol

    def verify_orthomodular_law(
        self,
        P_A: NDArray[np.complex128],
        P_B: NDArray[np.complex128],
        tolerance: Optional[float] = None
    ) -> Optional[bool]:
        r"""
        Verifica la ley ortomodular para P_A ≤ P_B.

        Retorna None si la premisa no se cumple.
        """
        tol = tolerance or self.tolerance
        if la.norm(P_A @ P_B - P_A, ord="fro") > tol:
            return None  # P_A no es subproyector de P_B

        P_A_comp = self.quantum_complement(P_A)
        meet = self.quantum_conjunction(P_A_comp, P_B, validate=True)
        join = self.quantum_disjunction(P_A, meet, validate=True)
        return la.norm(join - P_B, ord="fro") < tol

    def boolean_sublattice(
        self,
        projectors: List[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        r"""
        Si la lista de proyectores conmuta dos a dos, genera el subretículo
        booleano (distributivo) cerrado bajo meet, join y complemento.
        """
        for i, P in enumerate(projectors):
            self.validate_projector(P)
            for j in range(i + 1, len(projectors)):
                if not self.are_compatible(P, projectors[j]):
                    raise NonCommutativeAlgebraError(
                        "Los proyectores no conmutan; no generan subretículo booleano."
                    )

        # Cerradura bajo operaciones booleanas
        closure = {tuple(np.round(p.flatten(order="F"), 12)) for p in projectors}
        closure = closure | {
            tuple(np.round(self.quantum_complement(p).flatten(order="F"), 12))
            for p in projectors
        }
        # Repetir hasta estabilidad
        stable = False
        while not stable:
            current = list(closure)
            new = set()
            for a in current:
                for b in current:
                    A = np.array(a).reshape(p.shape, order="F")
                    B = np.array(b).reshape(p.shape, order="F")
                    meet = self.quantum_conjunction(A, B, validate=False)
                    join = self.quantum_disjunction(A, B, validate=False)
                    new.add(tuple(np.round(meet.flatten(order="F"), 12)))
                    new.add(tuple(np.round(join.flatten(order="F"), 12)))
            if new.issubset(closure):
                stable = True
            closure |= new

        return [
            np.array(key).reshape(projectors[0].shape, order="F")
            for key in closure
        ]

    def prepare_modular_state(
        self,
        rho: NDArray[np.complex128],
        tolerance: float = 1e-14
    ) -> NDArray[np.complex128]:
        r"""
        Puente formal hacia la Fase 3: valida un estado fiel respecto al álgebra.

        Condiciones:
            1. ρ es hermitiano
            2. Tr(ρ) = 1
            3. ρ > 0   (estado fiel)
            4. Soporte(ρ) = I  (factor completo)
        """
        if not self.algebra.is_square(rho):
            raise ValueError("El estado no pertenece al álgebra de operadores.")

        if not _is_hermitian(rho, tolerance):
            raise ValueError("El estado no es hermitiano.")

        if np.abs(self.algebra.trace(rho) - 1.0) > tolerance:
            raise ValueError(f"Traza del estado distinta de 1: {self.algebra.trace(rho)}")

        eigvals = la.eigvalsh(rho)
        if np.any(eigvals <= tolerance):
            min_eig = np.min(eigvals)
            raise TraceAnomalyError(
                f"Estado no es fiel (cíclico y separador). "
                f"Eigenvalor mínimo: {min_eig:.3e}."
            )

        if not self.algebra.is_full_factor(rho, tolerance=tolerance):
            raise TraceAnomalyError(
                "Soporte del estado no coincide con la identidad del álgebra."
            )

        logger.info(
            f"Estado fiel validado para teoría modular. dim={self.algebra.dimension}"
        )
        return rho.copy()


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: TEORÍA MODULAR DE TOMITA-TAKESAKI
# ══════════════════════════════════════════════════════════════════════════════
# La Fase 3 recibe un OperatorAlgebra (Fase 1) y un estado fiel validado por
# la Fase 2, y construye los operadores modulares correctos para M_d(ℂ):
#
#   Δ(A) = ρ A ρ^{-1}
#   J(A) = ρ^{1/2} A^* ρ^{-1/2}
#   S(A) = A^* = J Δ^{1/2}(A)
#   σ_t(A) = ρ^{it} A ρ^{-it}
#
# Todos los operadores se representan también como superoperadores d²×d² para
# verificación numérica rigurosa.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TomitaTakesakiTheory:
    r"""
    Teoría Modular de Tomita-Takesaki para el álgebra M_d(ℂ) con estado fiel ρ.

    Construcción finita correcta:
        - Δ = L_ρ R_{ρ^{-1}}
        - J(A) = ρ^{1/2} A^* ρ^{-1/2}
        - S = J Δ^{1/2}
        - σ_t(A) = ρ^{it} A ρ^{-it}
    """

    algebra: OperatorAlgebra
    state: NDArray[np.complex128]
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.algebra.is_square(self.state):
            raise ValueError("El estado no pertenece al álgebra.")
        self._validate_faithful_state()
        self._precompute_modular_operators()

    def _validate_faithful_state(self, tolerance: Optional[float] = None) -> None:
        tol = tolerance or self.tolerance
        if not _is_hermitian(self.state, tol):
            raise ValueError("Estado no hermitiano.")
        if np.abs(self.algebra.trace(self.state) - 1.0) > tol:
            raise TraceAnomalyError("Traza del estado distinta de 1.")
        eigvals = la.eigvalsh(self.state)
        if np.any(eigvals <= tol):
            raise TraceAnomalyError(
                f"Estado no fiel. Eigenvalor mínimo: {np.min(eigvals):.3e}"
            )

    def _precompute_modular_operators(self) -> None:
        rho = self.state
        d = self.algebra.dimension

        # Descomposición espectral ρ = Σ λ_i |ψ_i⟩⟨ψ_i|
        eigvals, eigvecs = la.eigh(rho)
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues = eigvals[idx]
        self.eigenvectors = eigvecs[:, idx]

        # Potencias de ρ para J, Δ y σ_t
        sqrt_eig = np.sqrt(self.eigenvalues)
        inv_sqrt_eig = 1.0 / sqrt_eig
        inv_eig = 1.0 / self.eigenvalues

        self.rho_sqrt = (
            self.eigenvectors
            @ np.diag(sqrt_eig)
            @ self.eigenvectors.conj().T
        )
        self.rho_inv_sqrt = (
            self.eigenvectors
            @ np.diag(inv_sqrt_eig)
            @ self.eigenvectors.conj().T
        )
        self.rho_inv = (
            self.eigenvectors
            @ np.diag(inv_eig)
            @ self.eigenvectors.conj().T
        )

        # Superoperadores en vectorización column-major
        # Δ(A) = ρ A ρ^{-1}  =>  Δ_vec = (ρ^{-1})^T ⊗ ρ
        self.Delta_vec = np.kron(self.rho_inv.T, rho)

        # J(A) = ρ^{1/2} A^* ρ^{-1/2}  =>  J_vec = (ρ^{-1/2})^T ⊗ ρ^{1/2} · P
        P = _transpose_permutation(d)
        self.J_vec = np.kron(self.rho_inv_sqrt.T, self.rho_sqrt) @ P

        # Δ^{1/2}(A) = ρ^{1/2} A ρ^{-1/2}
        self.Delta_half_vec = np.kron(self.rho_inv_sqrt.T, self.rho_sqrt)

        # S(A) = A^*  =>  S_vec = P
        self.S_vec = P

        # Gram GNS: G = I ⊗ ρ
        self.GNS_gram = np.kron(np.eye(d, dtype=np.complex128), rho)

    def _apply_linear(
        self,
        superop: NDArray[np.complex128],
        A: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Aplica un superoperador lineal a una matriz."""
        return _as_operator(superop @ _as_vector(A), self.algebra.dimension)

    def modular_operator_apply(
        self,
        A: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        r"""Δ(A) = ρ A ρ^{-1}."""
        return self.state @ A @ self.rho_inv

    def modular_conjugation_apply(
        self,
        A: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        r"""J(A) = ρ^{1/2} A^* ρ^{-1/2}."""
        return self.rho_sqrt @ A.conj() @ self.rho_inv_sqrt

    def fundamental_involution_apply(
        self,
        A: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        r"""S(A) = A^*."""
        return A.conj()

    def modular_automorphism_group(
        self,
        t: Union[float, complex]
    ) -> Callable[[NDArray[np.complex128]], NDArray[np.complex128]]:
        r"""
        Grupo modular de automorfismos:
            σ_t(A) = ρ^{it} A ρ^{-it}

        El parámetro t puede ser real (evolución modular) o complejo
        (extensión analítica, e.g. t = -i para la condición KMS).
        """
        rho_it = (
            self.eigenvectors
            @ np.diag(self.eigenvalues ** (1j * t))
            @ self.eigenvectors.conj().T
        )
        rho_minus_it = (
            self.eigenvectors
            @ np.diag(self.eigenvalues ** (-1j * t))
            @ self.eigenvectors.conj().T
        )

        def sigma_t(A: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return rho_it @ A @ rho_minus_it

        return sigma_t

    def verify_tomita_takesaki_relations(
        self,
        tolerance: Optional[float] = None
    ) -> Dict[str, Union[bool, float]]:
        r"""
        Verifica las relaciones fundamentales de Tomita-Takesaki:

            1. J² = I
            2. J Δ J = Δ^{-1}
            3. S² = I
            4. S = J Δ^{1/2}
            5. Δ es autoadjunto en producto GNS
        """
        tol = tolerance or self.tolerance
        d = self.algebra.dimension

        # Operadores de prueba hermitianos aleatorios
        rng = np.random.default_rng(42)
        A = rng.random((d, d)) + 1j * rng.random((d, d))
        A = (A + A.conj().T) / 2.0
        B = rng.random((d, d)) + 1j * rng.random((d, d))
        B = (B + B.conj().T) / 2.0

        results: Dict[str, Union[bool, float]] = {}

        # 1. J² = I
        JJA = self.modular_conjugation_apply(
            self.modular_conjugation_apply(A)
        )
        err = la.norm(JJA - A, ord="fro")
        results["J_involutive"] = err < tol
        results["J_involutive_error"] = float(err)

        # 2. J Δ J = Δ^{-1}
        JDJA = self.modular_conjugation_apply(
            self.modular_operator_apply(
                self.modular_conjugation_apply(A)
            )
        )
        Delta_inv_A = self.rho_inv @ A @ self.state
        err = la.norm(JDJA - Delta_inv_A, ord="fro")
        results["J_Delta_J"] = err < tol
        results["J_Delta_J_error"] = float(err)

        # 3. S² = I
        SSA = self.fundamental_involution_apply(
            self.fundamental_involution_apply(A)
        )
        err = la.norm(SSA - A, ord="fro")
        results["S_involutive"] = err < tol
        results["S_involutive_error"] = float(err)

        # 4. S = J Δ^{1/2}
        delta_half_A = self._apply_linear(self.Delta_half_vec, A)
        J_delta_half_A = self.modular_conjugation_apply(delta_half_A)
        S_A = self.fundamental_involution_apply(A)
        err = la.norm(J_delta_half_A - S_A, ord="fro")
        results["S_factorization"] = err < tol
        results["S_factorization_error"] = float(err)

        # 5. Δ autoadjunto en GNS: ⟨Δ(A), B⟩_ρ = ⟨A, Δ(B)⟩_ρ
        lhs = self.algebra.gns_inner(
            self.state,
            self.modular_operator_apply(A),
            B
        )
        rhs = self.algebra.gns_inner(
            self.state,
            A,
            self.modular_operator_apply(B)
        )
        err = float(np.abs(lhs - rhs))
        results["Delta_GNS_self_adjoint"] = err < tol
        results["Delta_GNS_self_adjoint_error"] = err

        return results

    def verify_kms_relation(
        self,
        A: NDArray[np.complex128],
        B: NDArray[np.complex128],
        beta: float = 1.0
    ) -> Dict[str, Union[bool, float]]:
        r"""
        Verifica la condición KMS a temperatura inversa β:

            ω(A B) = ω(B σ_{-iβ}(A))

        Para β = 1 y ω(X) = Tr(ρ X):
            σ_{-i}(A) = ρ A ρ^{-1}
            ω(B σ_{-i}(A)) = Tr(ρ B ρ A ρ^{-1}) = Tr(ρ A B)
        """
        if not _is_hermitian(A) or not _is_hermitian(B):
            raise ValueError("A y B deben ser hermitianos para verificar KMS.")

        omega_AB = np.trace(self.state @ A @ B)

        # t = -i β
        t = -1.0j * beta
        sigma = self.modular_automorphism_group(t)
        sigma_A = sigma(A)
        omega_B_sigma = np.trace(self.state @ B @ sigma_A)

        results: Dict[str, Union[bool, float]] = {
            "KMS_holds": np.abs(omega_AB - omega_B_sigma) < self.tolerance,
            "lhs": float(omega_AB.real),
            "rhs": float(omega_B_sigma.real),
            "error": float(np.abs(omega_AB - omega_B_sigma)),
        }
        return results

    def classify_factor(
        self,
        tolerance: Optional[float] = None
    ) -> VonNeumannFactorType:
        """
        Clasifica el factor de von Neumann asociado al estado.

        En dimensión finita el álgebra completa M_d(ℂ) es siempre Tipo I_n.
        Se detecta adicionalmente si el estado es tracial (ρ = I/d).
        """
        tol = tolerance or self.tolerance
        d = self.algebra.dimension

        # En dimensión finita sólo puede ser Tipo I_n
        if d < float("inf"):
            factor_type = VonNeumannFactorType.TYPE_I_FINITE
        else:
            factor_type = VonNeumannFactorType.TYPE_I_INFINITE

        # Detección de estado tracial
        tracial_state = np.eye(d, dtype=np.complex128) / d
        self.tracial = la.norm(self.state - tracial_state, ord="fro") < tol

        # Espectro de Δ: distinción adicional informativa
        delta_eigs = np.real(la.eigvalsh(self.Delta_vec))
        unique = np.unique(np.round(delta_eigs, 6))
        if len(unique) == 1 and np.abs(unique[0] - 1.0) < tol:
            self.tracial = True

        return factor_type

    def build_modular_data(self) -> ModularTheoryData:
        """Empaqueta todos los datos modulares en una estructura inmutable."""
        factor_type = self.classify_factor()
        return ModularTheoryData(
            modular_operator=self.Delta_vec,
            modular_conjugation=self.J_vec,
            modular_automorphism_group=self.modular_automorphism_group,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            factor_type=factor_type,
            tracial=self.tracial,
        )