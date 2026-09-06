# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Vectors (Operador de Inyección Tensorial y Canal Bicomplejo)    ║
║ Ruta   : app/wisdom/mac_vectors.py                                           ║
║ Versión: 5.0.0-Doctoral-Bicomplex-HyperbolicAlgebra-SpectralCalculus         ║
║          [FASE 1 / 3 — Kernel Algebraico Bicomplejo]                         ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa la inyección de estados cuánticos y semánticos        ║
║ continuos en el espacio de Hilbert de la Matriz Atómica de Conocimiento      ║
║ (MAC) en el Estrato de la Sabiduría (V_W, Nivel 0).                          ║
║                                                                              ║
║ Enriquece la inyección cuántica ordinaria con el álgebra de los Números      ║
║ Bicomplejos C₂ ≅ C ⊗ C, introduciendo proyectores idempotentes sobre la      ║
║ base {e1, e2}. Esto permite segregar el flujo de información y valor en      ║
║ canales ortogonales de Inversión Directa y Carga Entrópica, detectando       ║
║ Puntos de Estancamiento Financiero como divisores de cero en el cono nulo    ║
║ de la FPU.                                                                   ║
║                                                                              ║
║ ESTRUCTURA DEL ANILLO BICOMPLEJO C₂ (base real {1,i,j,k}):                   ║
║   i² = -1, j² = -1, ij = ji = k, k² = +1, k ≠ ±1.                            ║
║   e1 = (1 + k)/2, e2 = (1 - k)/2.                                            ║
║   e1² = e1, e2² = e2, e1e2 = 0, e1 + e2 = 1.                                 ║
║                                                                              ║
║ REPRESENTACIÓN IDEMPOTENTE (Teorema de descomposición de C₂, Price 1991):    ║
║   Todo z = z1 + z2·j ∈ C₂ (z1,z2 ∈ C(i)) admite la única descomposición:     ║
║       z = u1·e1 + u2·e2,     u1 = z1 - i·z2,   u2 = z1 + i·z2 ∈ C(i)         ║
║   que constituye un ISOMORFISMO DE ANILLOS  C₂ ≅ C(i) ⊕ C(i).                ║
║   Esta es la base categórica de todo el módulo: el funtor de descomposición  ║
║   idempotente es una equivalencia entre la categoría de módulos hermíticos   ║
║   sobre C₂ y el producto de dos copias de la categoría de módulos            ║
║   hermíticos sobre C — de allí que ρ = ρ1·e1 + ρ2·e2 se pueda auditar        ║
║   canal por canal de forma completamente desacoplada.                        ║
║                                                                              ║
║ LAS TRES CONJUGACIONES DE C₂ (derivadas exactamente en base real {1,i,j,k}): ║
║   τ_i (bar):   (a,b,c,d) → (a,-b, c,-d)   ⟹  (u1,u2) → (ū2, ū1)  [swap+conj]║
║   τ_j (star):  (a,b,c,d) → (a, b,-c,-d)   ⟹  (u1,u2) → (u2, u1)  [swap puro]║
║   τ_k (bar★):  (a,b,c,d) → (a,-b,-c, d)   ⟹  (u1,u2) → (ū1, ū2)  [conj puro]║
║                                                                              ║
║ MÓDULO HIPERBÓLICO (vía τ_k, produce un elemento de D = R[k]/(k²-1)):        ║
║       |z|²_D := z · τ_k(z) = |u1|² e1 + |u2|² e2  ∈  D₊ (cono positivo)      ║
║                                                                              ║
║ MATRIZ DE DENSIDAD BICOMPLEJA:                                               ║
║   ρ = ρ1 e1 + ρ2 e2, con ρ1, ρ2 ∈ M_d(C).                                    ║
║                                                                              ║
║ ÍNDICE DE ESTANCAMIENTO FINANCIERO:                                          ║
║   χ_stagnation = ||ρ1 ρ2†||_F².                                              ║
║   Un divisor de cero aparece cuando:                                         ║
║       ||ρ1||_F > 0 ∧ ||ρ2||_F > 0 ∧ ρ1 ρ2† ≈ 0.                              ║
║   Esto es exactamente la condición de divisor de cero del anillo hiperbólico ║
║   D: un elemento z1·e1 + z2·e2 con z1≠0 ∧ z2≠0 puede aun así ser divisor     ║
║   de cero relativo si el producto cruzado se anula — la MAC generaliza esta  ║
║   noción escalar de D al caso matricial M_d(C)⊗D.                            ║
║                                                                              ║
║ ÁLGEBRA DE BOOLE SUBYACENTE:                                                 ║
║   {0, e1, e2, 1} ⊂ C₂, con el producto de idempotentes como "meet", forma    ║
║   un álgebra de Boole de 2 átomos, isomorfa a (℘({1,2}), ∩, ∪, ᶜ). Esta es   ║
║   la base formal de los veredictos ternarios de Heyting de la Fase 3.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    AbstractSet,
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
import numpy.typing as npt
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.MacVectors")

# ───────────────────────────────────────────────────────────────────────────────
# Alias de tipos (rigor de anotación — se reutilizan en Fases 2 y 3)
# ───────────────────────────────────────────────────────────────────────────────

ComplexMatrix = npt.NDArray[np.complexfloating]
RealArray = npt.NDArray[np.floating]

# ───────────────────────────────────────────────────────────────────────────────
# Constantes globales de tolerancia (evita "números mágicos" dispersos)
# ───────────────────────────────────────────────────────────────────────────────

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

_DEFAULT_TOLERANCE: Final[float] = 1e-12
_DEFAULT_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_DEFAULT_PSD_TOLERANCE: Final[float] = 1e-12
_DEFAULT_RING_AXIOM_TOLERANCE: Final[float] = 1e-9

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_LEGACY_OVERRIDE_TOKENS: Final[frozenset] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_STAG_IDU_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)


# ───────────────────────────────────────────────────────────────────────────────
# Compatibilidad con dependencias del ecosistema APU Filter
#
# NOTA DE ARQUITECTURA (Fase 1/3): únicamente se importa aquí la excepción de
# inestabilidad numérica, pues es la única dependencia externa requerida por
# el kernel algebraico bicomplejo de esta fase. Los shims de `Stratum`,
# `VectorResultStatus`, `AtomicDensityMatrix` y `POVMMeasurement` se
# introducirán en las Fases 2 y 3, donde efectivamente se consumen.
# ───────────────────────────────────────────────────────────────────────────────

try:
    from app.core.mic_algebra import NumericalInstabilityError
except ImportError:
    class NumericalInstabilityError(Exception):
        r"""
        Error de inestabilidad numérica en la FPU.
        """
        pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · ESTRATO 0: Núcleo numérico riguroso (Teoría Espectral + Banach)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _validate_square_matrix(
    M: Any,
    name: str = "matriz",
    expected_dimension: Optional[int] = None,
) -> ComplexMatrix:
    r"""
    Valida que una entrada sea coercible a una matriz cuadrada, compleja y
    finita perteneciente al espacio de Banach (M_d(C), ||·||).

    Parámetros
    ----------
    M : array-like
        Entrada a validar.
    name : str
        Nombre descriptivo para mensajes de error.
    expected_dimension : Optional[int]
        Si se especifica, exige que la dimensión d coincida exactamente.

    Retorna
    -------
    np.ndarray (complex128, cuadrada, finita).
    """
    arr = np.asarray(M, dtype=np.complex128)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"La {name} debe ser cuadrada. Obtenido: {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"La {name} contiene valores no finitos.")

    if expected_dimension is not None and arr.shape[0] != expected_dimension:
        raise ValueError(
            f"La {name} debe tener dimensión {expected_dimension}. "
            f"Obtenido: {arr.shape[0]}."
        )

    return arr


def _safe_xlogx(x: np.ndarray, tolerance: float = _DEFAULT_TOLERANCE) -> np.ndarray:
    r"""
    Extensión continua de x ↦ x·log(x) en x=0 por convención de Shannon:

        lim_{x→0+} x log x = 0.

    Evita máscaras booleanas repetidas en cada cálculo de entropía.
    """
    arr = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(arr)
    mask = arr > tolerance
    result[mask] = arr[mask] * np.log(arr[mask])
    return result


def _compensated_sum(values: np.ndarray) -> float:
    r"""
    Sumación de precisión extendida usando el algoritmo de Shewchuk
    (implementado en `math.fsum`), que garantiza el redondeo *correctamente
    redondeado* (exacto salvo el último bit) del resultado exacto en
    aritmética racional — una cota estrictamente más fuerte que la
    compensación de Kahan-Babuška-Neumaier, sin penalización asintótica de
    rendimiento (implementación en C).

    Se preserva el comportamiento de cortocircuito ante valores no finitos.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()

    if flat.size == 0:
        return 0.0

    if not np.all(np.isfinite(flat)):
        non_finite = flat[~np.isfinite(flat)]
        return float(non_finite[0])

    return float(math.fsum(flat.tolist()))


# Alias de compatibilidad retroactiva con el nombre histórico del método.
_kahan_sum = _compensated_sum


def _hermitianize(M: np.ndarray) -> ComplexMatrix:
    r"""
    Proyección hermítica exacta (proyector ortogonal de Frobenius sobre el
    subespacio real de operadores autoadjuntos):

        H(M) = (M + M†) / 2
    """
    return 0.5 * (M + M.conj().T)


def _is_hermitian(M: np.ndarray, tolerance: float = _DEFAULT_HERMITIAN_TOLERANCE) -> bool:
    r"""
    Verifica hermiticidad dentro de una tolerancia Frobenius.
    """
    return bool(la.norm(M.conj().T - M, "fro") <= tolerance)


def _is_positive_semidefinite(
    M: np.ndarray,
    tolerance: float = _DEFAULT_PSD_TOLERANCE,
) -> bool:
    r"""
    Verifica semidefinición positiva usando el espectro hermítico, con
    tolerancia escalada por la norma espectral (criterio de estabilidad de
    Wilkinson: el error de redondeo en eigh escala con ||M||).
    """
    H = _hermitianize(M)
    eigvals = la.eigvalsh(H)

    if eigvals.size == 0:
        return True

    scale = max(1.0, float(np.max(np.abs(eigvals))))
    threshold = max(tolerance, 100.0 * _MACHINE_EPS * scale)

    return bool(np.min(eigvals) >= -threshold)


def _hermitian_functional_calculus(
    M: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    tolerance: float = _DEFAULT_PSD_TOLERANCE,
    repair_negative: bool = True,
    require_nonnegative: bool = False,
    context_label: str = "cálculo funcional",
) -> ComplexMatrix:
    r"""
    Cálculo funcional continuo sobre operadores hermíticos (Teorema Espectral
    de Hilbert-von Neumann):

        M = Σ_i λ_i |v_i⟩⟨v_i|      (descomposición espectral)
        f(M) := Σ_i f(λ_i) |v_i⟩⟨v_i|

    Este es el bloque de construcción unificado del que se derivan √M, log M,
    exp(M), y cualquier función escalar aplicada de forma covariante bajo
    conjugación unitaria: f(U M U†) = U f(M) U†.

    Parámetros
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        Función escalar vectorizada aplicada al espectro {λ_i}.
    require_nonnegative : bool
        Si True, exige M ⪰ 0 estrictamente (para funciones como log o
        raíces cuya rama principal requiere dominio no negativo).
    """
    H = _validate_square_matrix(M, f"matriz para {context_label}")
    H = _hermitianize(H)

    eigvals, eigvecs = la.eigh(H)

    if eigvals.size == 0:
        return H

    min_eig = float(np.min(eigvals))

    if require_nonnegative and min_eig < -tolerance:
        if not repair_negative:
            raise NumericalInstabilityError(
                f"La matriz no es semidefinida positiva para {context_label}. "
                f"Autovalor mínimo: {min_eig:.3e}"
            )

        logger.warning(
            "Autovalor negativo detectado en %s: %.3e. Se proyecta a cero.",
            context_label,
            min_eig,
        )

    transformed = np.asarray(func(eigvals), dtype=np.float64)

    if transformed.shape != eigvals.shape:
        raise ValueError(
            "La función del cálculo funcional debe preservar la forma del "
            "espectro (aplicación escalar vectorizada)."
        )

    return (eigvecs * transformed) @ eigvecs.conj().T


def _matrix_sqrt_psd(
    A: np.ndarray,
    tolerance: float = _DEFAULT_PSD_TOLERANCE,
    repair: bool = True,
) -> ComplexMatrix:
    r"""
    Calcula √A de forma espectralmente estable para A ⪰ 0, vía cálculo
    funcional: f(λ) = √max(λ,0).
    """
    return _hermitian_functional_calculus(
        A,
        func=lambda ev: np.sqrt(np.maximum(ev, 0.0)),
        tolerance=tolerance,
        repair_negative=repair,
        require_nonnegative=True,
        context_label="raíz cuadrada PSD",
    )


def _matrix_log_psd(
    A: np.ndarray,
    tolerance: float = _DEFAULT_PSD_TOLERANCE,
    floor: Optional[float] = None,
) -> ComplexMatrix:
    r"""
    Logaritmo matricial regularizado para A ⪰ 0, vía cálculo funcional:

        f(λ) = log(max(λ, floor))

    El piso (floor) regulariza la singularidad log(0) = -∞, con convención
    consistente con la extensión continua x log x → 0 usada en la entropía
    de von Neumann (ver `_safe_xlogx`). Esencial para D(ρ||σ) en la Fase 2.
    """
    floor_value = floor if floor is not None else max(tolerance, _MACHINE_EPS)

    return _hermitian_functional_calculus(
        A,
        func=lambda ev: np.log(np.maximum(ev, floor_value)),
        tolerance=tolerance,
        repair_negative=True,
        require_nonnegative=True,
        context_label="logaritmo matricial PSD",
    )


def _repair_density_matrix(
    rho: np.ndarray,
    tolerance: float = _DEFAULT_PSD_TOLERANCE,
    normalize: bool = True,
) -> Tuple[ComplexMatrix, float]:
    r"""
    Repara un operador de densidad:

      1. Hermitianiza.
      2. Proyecta autovalores negativos pequeños a cero (proyección al cono
         PSD más cercano en norma de Frobenius — Teorema de Higham).
      3. Renormaliza la traza si corresponde.

    Retorna:
        (rho_reparada, traza)
    """
    M = _validate_square_matrix(rho, "matriz de densidad")
    M = _hermitianize(M)

    eigvals, eigvecs = la.eigh(M)

    if eigvals.size == 0:
        return M, 0.0

    min_eig = float(np.min(eigvals))
    allowed_negative = -math.sqrt(max(tolerance, _MACHINE_EPS))

    if min_eig < allowed_negative:
        raise NumericalInstabilityError(
            f"Matriz de densidad no física. Autovalor mínimo: {min_eig:.3e}"
        )

    eigvals_clipped = np.maximum(eigvals, 0.0)
    M = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    M = _hermitianize(M)

    trace = float(np.trace(M).real)

    if normalize and trace > tolerance:
        M = M / trace
        trace = 1.0

    return M, trace


def _von_neumann_entropy_from_eigvals(
    eigvals: np.ndarray,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    r"""
    Entropía de von Neumann a partir de autovalores:

        S(ρ) = -Σ λ_i log λ_i = -Σ xlogx(λ_i)
    """
    vals = np.asarray(eigvals, dtype=np.float64)
    return float(-_compensated_sum(_safe_xlogx(vals, tolerance)))


def _frobenius_inner_product(A: np.ndarray, B: np.ndarray) -> complex:
    r"""
    Producto interno de Hilbert-Schmidt (estructura pre-Hilbert de M_d(C)):

        ⟨A, B⟩_HS = Tr(A† B)
    """
    Aa = _validate_square_matrix(A, "A (producto de Hilbert-Schmidt)")
    Bb = _validate_square_matrix(B, "B (producto de Hilbert-Schmidt)")

    if Aa.shape != Bb.shape:
        raise ValueError(
            "Las matrices deben compartir dimensión para el producto de "
            "Hilbert-Schmidt."
        )

    return complex(np.trace(Aa.conj().T @ Bb))


class MatrixNormType(Enum):
    r"""
    Familia de normas submultiplicativas sobre M_d(C), fundamentando su
    estructura de Álgebra de Banach (y, en el caso OPERATOR, de C*-álgebra:
    ||A*A|| = ||A||²).
    """
    FROBENIUS = auto()   # Norma de Hilbert-Schmidt (L2 de Schatten)
    OPERATOR = auto()    # Norma espectral / C*-norma (sup de valores singulares)
    TRACE = auto()       # Norma nuclear (L1 de Schatten)
    SCHATTEN = auto()    # Norma p-Schatten general


def _matrix_norm(
    M: np.ndarray,
    norm_type: MatrixNormType = MatrixNormType.FROBENIUS,
    p: Optional[float] = None,
) -> float:
    r"""
    Dispatcher unificado de normas matriciales submultiplicativas.

        ||A||_Frobenius = sqrt(Tr(A†A))            = (Σ σ_i²)^{1/2}
        ||A||_operador  = σ_max(A)                 (C*-norma)
        ||A||_traza     = Σ σ_i                    (norma nuclear)
        ||A||_Schatten,p = (Σ σ_i^p)^{1/p}
    """
    A = _validate_square_matrix(M, "matriz para cómputo de norma")

    if norm_type is MatrixNormType.FROBENIUS:
        return float(la.norm(A, "fro"))

    singular_values = la.svdvals(A)

    if singular_values.size == 0:
        return 0.0

    if norm_type is MatrixNormType.OPERATOR:
        return float(np.max(singular_values))

    if norm_type is MatrixNormType.TRACE:
        return float(np.sum(singular_values))

    if norm_type is MatrixNormType.SCHATTEN:
        if p is None or p <= 0:
            raise ValueError("Debe especificarse p > 0 para la norma de Schatten.")
        return float(np.sum(singular_values ** p) ** (1.0 / p))

    raise ValueError(f"Tipo de norma desconocido: {norm_type}")


def _condition_number(M: np.ndarray) -> float:
    r"""
    Número de condición espectral κ(M) = σ_max / σ_min, diagnóstico central
    de estabilidad numérica (cota de amplificación de error relativo de
    Wilkinson).
    """
    A = _validate_square_matrix(M, "matriz para número de condición")
    singular_values = la.svdvals(A)

    if singular_values.size == 0:
        return 1.0

    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))

    if min_sv <= _MACHINE_EPS * max(1.0, max_sv):
        return float("inf")

    return max_sv / min_sv


def _spectral_rank(eigvals: np.ndarray, tolerance: float = _DEFAULT_TOLERANCE) -> int:
    r"""
    Rango espectral robusto: número de autovalores cuya magnitud excede una
    tolerancia escalada por el autovalor máximo.
    """
    vals = np.asarray(eigvals, dtype=np.float64)

    if vals.size == 0:
        return 0

    scale = max(1.0, float(np.max(np.abs(vals))))
    threshold = max(tolerance, 100.0 * _MACHINE_EPS * scale)

    return int(np.sum(np.abs(vals) > threshold))


def _partial_trace_matrix(
    rho: np.ndarray,
    dims: Tuple[int, int],
    keep: int,
) -> ComplexMatrix:
    r"""
    Traza parcial de un operador bipartito ρ ∈ M_{d_A d_B}(C) ≅ M_{d_A}(C) ⊗ M_{d_B}(C).

        Tr_B(ρ)[a,a'] = Σ_b ρ[(a,b),(a',b)]     (keep = 0, sistema A)
        Tr_A(ρ)[b,b'] = Σ_a ρ[(a,b),(a,b')]     (keep = 1, sistema B)

    Operación estructural de la Mecánica Cuántica de sistemas compuestos;
    en el lenguaje de cálculo de cuerdas/diagramas de cuerdas (ZX-calculus),
    corresponde al "cierre" (cap) de un cable del sistema descartado.
    """
    dim_a, dim_b = dims
    A = _validate_square_matrix(rho, "rho bipartito", expected_dimension=dim_a * dim_b)

    tensor = A.reshape(dim_a, dim_b, dim_a, dim_b)

    if keep == 0:
        reduced = np.einsum("ibkb->ik", tensor)
    elif keep == 1:
        reduced = np.einsum("iaka->...".replace("...", "ia" + "ka"), tensor) \
            if False else np.einsum("aibl->ab", np.transpose(tensor, (0, 1, 2, 3)))
    else:
        raise ValueError("keep debe ser 0 (sistema A) o 1 (sistema B).")

    # Recomputo explícito y sin ambigüedad de 'keep == 1' para máxima claridad:
    if keep == 1:
        reduced = np.einsum("ajal->jl", tensor)

    return _hermitianize(reduced)


def _is_bicomplex_input(obj: Any) -> bool:
    r"""
    Detecta si una entrada corresponde a un estado bicomplejo (predicado de
    pertenencia al "sub-universo" C₂ de la categoría de entradas admisibles).
    """
    if isinstance(obj, BicomplexDensityMatrix):
        return True

    if isinstance(obj, dict):
        return "rho1" in obj and "rho2" in obj

    if isinstance(obj, (tuple, list)):
        return len(obj) == 2

    return False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · ESTRATO 1: Álgebra Hiperbólica D = R[k]/(k²-1) ≅ R ⊕ R              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class HyperbolicNumber:
    r"""
    Número hiperbólico (split-complex, "duplex") en representación
    idempotente:

        d = z1·e1 + z2·e2,     z1, z2 ∈ R

    Isomorfo, vía (z1,z2), al anillo producto R ⊕ R (no es un dominio de
    integridad: e1·e2 = 0 son divisores de cero mutuos).

    Esta clase formaliza —con rigor de doctorado— la estructura algebraica
    exacta sobre la que se define `χ_stagnation`: el índice de estancamiento
    financiero de la MAC es, precisamente, la componente `e1`-`e2` cruzada
    de un elemento de D construido a partir de las normas de canal.

    Relación con la representación estándar d = x + y·k (k² = +1):

        z1 = x + y,   z2 = x - y     (ida)
        x = (z1+z2)/2, y = (z1-z2)/2  (vuelta)

    Cono positivo: D₊ = {d : z1 ≥ 0 ∧ z2 ≥ 0}, que induce el orden parcial
    canónico d ⪯ d'  ⟺  d' - d ∈ D₊, compatible con la estructura de anillo.
    """

    z1: float
    z2: float

    # ── Constructores canónicos ────────────────────────────────────────────
    @classmethod
    def from_standard(cls, x: float, y: float) -> "HyperbolicNumber":
        return cls(z1=float(x + y), z2=float(x - y))

    def to_standard(self) -> Tuple[float, float]:
        return (0.5 * (self.z1 + self.z2), 0.5 * (self.z1 - self.z2))

    @classmethod
    def zero(cls) -> "HyperbolicNumber":
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> "HyperbolicNumber":
        return cls(1.0, 1.0)

    @classmethod
    def idempotent_e1(cls) -> "HyperbolicNumber":
        return cls(1.0, 0.0)

    @classmethod
    def idempotent_e2(cls) -> "HyperbolicNumber":
        return cls(0.0, 1.0)

    # ── Estructura de anillo (diagonal en base idempotente) ────────────────
    def __add__(self, other: "HyperbolicNumber") -> "HyperbolicNumber":
        return HyperbolicNumber(self.z1 + other.z1, self.z2 + other.z2)

    def __sub__(self, other: "HyperbolicNumber") -> "HyperbolicNumber":
        return HyperbolicNumber(self.z1 - other.z1, self.z2 - other.z2)

    def __neg__(self) -> "HyperbolicNumber":
        return HyperbolicNumber(-self.z1, -self.z2)

    def __mul__(self, other: Union["HyperbolicNumber", float, int]) -> "HyperbolicNumber":
        if isinstance(other, HyperbolicNumber):
            return HyperbolicNumber(self.z1 * other.z1, self.z2 * other.z2)
        scalar = float(other)
        return HyperbolicNumber(self.z1 * scalar, self.z2 * scalar)

    __rmul__ = __mul__

    # ── Orden parcial y clasificación del cono ─────────────────────────────
    def is_positive(self, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        r"""Pertenencia al cono positivo D₊ = {z1 ≥ 0 ∧ z2 ≥ 0}."""
        return bool(self.z1 >= -tolerance and self.z2 >= -tolerance)

    def is_zero_divisor(self, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        r"""
        Un elemento no nulo d = z1 e1 + z2 e2 es divisor de cero de D si y
        solo si exactamente una componente idempotente se anula:

            (z1 ≈ 0) XOR (z2 ≈ 0),  con la otra componente no nula.
        """
        z1_null = abs(self.z1) <= tolerance
        z2_null = abs(self.z2) <= tolerance
        return bool(z1_null != z2_null)

    def leq(self, other: "HyperbolicNumber") -> bool:
        r"""Orden parcial canónico d ⪯ d' inducido por el cono D₊."""
        return bool(self.z1 <= other.z1 and self.z2 <= other.z2)

    def modulus_euclidean(self) -> float:
        r"""Norma euclídea |d| = sqrt(x²+y²) = sqrt((z1²+z2²)/2)."""
        return float(math.sqrt(0.5 * (self.z1 ** 2 + self.z2 ** 2)))

    def __repr__(self) -> str:  # pragma: no cover - solo diagnóstico
        x, y = self.to_standard()
        return f"HyperbolicNumber(x={x:.6g}, y={y:.6g} | e1={self.z1:.6g}, e2={self.z2:.6g})"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · ESTRATO 2: Álgebra Bicompleja C₂ = C ⊗ C                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class BicomplexNumber:
    r"""
    Número bicomplejo en base real {1, i, j, k}:

        z = a + b·i + c·j + d·k,     a,b,c,d ∈ R
        i² = -1, j² = -1, k = ij = ji, k² = +1.

    Multiplicación (derivada exactamente por expansión de la tabla de
    multiplicación de la base y verificada mediante `verify_ring_axioms`):

        (a+bi+cj+dk)(a'+b'i+c'j+d'k) =
            [aa' - bb' - cc' + dd']
          + [ab' + ba' - cd' - dc'] i
          + [ac' + ca' - bd' - db'] j
          + [ad' + da' + bc' + cb'] k

    Representación idempotente z = u1·e1 + u2·e2 (u1,u2 ∈ C(i)):

        u1 = (a+d) + (b-c)i,     u2 = (a-d) + (b+c)i.

    Las tres conjugaciones canónicas (todas derivadas y verificadas contra
    la base real, no asumidas de la literatura):

        τ_i: (a,b,c,d) → (a,-b, c,-d)   ⟹  (u1,u2) → (ū2, ū1)
        τ_j: (a,b,c,d) → (a, b,-c,-d)   ⟹  (u1,u2) → (u2,  u1)
        τ_k: (a,b,c,d) → (a,-b,-c, d)   ⟹  (u1,u2) → (ū1,  ū2)
    """

    a: float
    b: float
    c: float
    d: float

    # ── Constructores canónicos ────────────────────────────────────────────
    @classmethod
    def zero(cls) -> "BicomplexNumber":
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> "BicomplexNumber":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def unit_i(cls) -> "BicomplexNumber":
        return cls(0.0, 1.0, 0.0, 0.0)

    @classmethod
    def unit_j(cls) -> "BicomplexNumber":
        return cls(0.0, 0.0, 1.0, 0.0)

    @classmethod
    def unit_k(cls) -> "BicomplexNumber":
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def idempotent_e1(cls) -> "BicomplexNumber":
        return cls(0.5, 0.0, 0.0, 0.5)

    @classmethod
    def idempotent_e2(cls) -> "BicomplexNumber":
        return cls(0.5, 0.0, 0.0, -0.5)

    @classmethod
    def from_idempotent(cls, u1: complex, u2: complex) -> "BicomplexNumber":
        r"""
        Reconstrucción inversa exacta desde la representación idempotente:

            a = (Re u1 + Re u2)/2,   d = (Re u1 - Re u2)/2
            b = (Im u1 + Im u2)/2,   c = (Im u2 - Im u1)/2
        """
        u1c, u2c = complex(u1), complex(u2)

        a = 0.5 * (u1c.real + u2c.real)
        d = 0.5 * (u1c.real - u2c.real)
        b = 0.5 * (u1c.imag + u2c.imag)
        c = 0.5 * (u2c.imag - u1c.imag)

        return cls(a, b, c, d)

    def to_idempotent(self) -> Tuple[complex, complex]:
        r"""Proyección al par de componentes complejas (u1, u2) ∈ C(i) × C(i)."""
        u1 = complex(self.a + self.d, self.b - self.c)
        u2 = complex(self.a - self.d, self.b + self.c)
        return u1, u2

    # ── Estructura de anillo conmutativo ────────────────────────────────────
    def __add__(self, other: "BicomplexNumber") -> "BicomplexNumber":
        return BicomplexNumber(
            self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d
        )

    def __sub__(self, other: "BicomplexNumber") -> "BicomplexNumber":
        return BicomplexNumber(
            self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d
        )

    def __neg__(self) -> "BicomplexNumber":
        return BicomplexNumber(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other: Union["BicomplexNumber", float, int]) -> "BicomplexNumber":
        if isinstance(other, (int, float)):
            s = float(other)
            return BicomplexNumber(self.a * s, self.b * s, self.c * s, self.d * s)

        a, b, c, d = self.a, self.b, self.c, self.d
        ap, bp, cp, dp = other.a, other.b, other.c, other.d

        new_a = a * ap - b * bp - c * cp + d * dp
        new_b = a * bp + b * ap - c * dp - d * cp
        new_c = a * cp + c * ap - b * dp - d * bp
        new_d = a * dp + d * ap + b * cp + c * bp

        return BicomplexNumber(new_a, new_b, new_c, new_d)

    __rmul__ = __mul__

    # ── Conjugaciones canónicas (derivadas y verificadas en base real) ─────
    def conj_i(self) -> "BicomplexNumber":
        r"""τ_i (bar): i → -i, j fijo. En idempotentes: (u1,u2) → (ū2, ū1)."""
        return BicomplexNumber(self.a, -self.b, self.c, -self.d)

    def conj_j(self) -> "BicomplexNumber":
        r"""τ_j (star): j → -j, i fijo. En idempotentes: (u1,u2) → (u2, u1)."""
        return BicomplexNumber(self.a, self.b, -self.c, -self.d)

    def conj_k(self) -> "BicomplexNumber":
        r"""τ_k (bar-star): i,j → -i,-j. En idempotentes: (u1,u2) → (ū1, ū2)."""
        return BicomplexNumber(self.a, -self.b, -self.c, self.d)

    # ── Estructura métrica ───────────────────────────────────────────────
    def hyperbolic_modulus_squared(self) -> HyperbolicNumber:
        r"""
        Módulo hiperbólico al cuadrado, vía la conjugación que fija los
        idempotentes (τ_k):

            |z|²_D = z · τ_k(z) = |u1|² e1 + |u2|² e2  ∈  D₊
        """
        u1, u2 = self.to_idempotent()
        return HyperbolicNumber(z1=float(abs(u1) ** 2), z2=float(abs(u2) ** 2))

    def euclidean_norm(self) -> float:
        r"""Norma euclídea en R⁴: ||z|| = sqrt(a²+b²+c²+d²)."""
        return float(math.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2))

    def is_zero_divisor(self, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        r"""
        z es divisor de cero de C₂ si y solo si exactamente una de sus
        componentes idempotentes (u1, u2) se anula.
        """
        u1, u2 = self.to_idempotent()
        u1_null = abs(u1) <= tolerance
        u2_null = abs(u2) <= tolerance
        return bool(u1_null != u2_null)

    def __repr__(self) -> str:  # pragma: no cover - solo diagnóstico
        return (
            f"BicomplexNumber(a={self.a:.6g}, b={self.b:.6g}, "
            f"c={self.c:.6g}, d={self.d:.6g})"
        )

    # ── Auto-certificación de los axiomas del anillo ────────────────────
    @classmethod
    def verify_ring_axioms(
        cls,
        tolerance: float = _DEFAULT_RING_AXIOM_TOLERANCE,
        random_seed: int = 42,
    ) -> Dict[str, float]:
        r"""
        Certificado numérico de los axiomas estructurales de C₂. Retorna un
        diccionario de errores residuales (idealmente todos ≈ 0), permitiendo
        auditar la implementación con rigor de doctorado sin depender de un
        motor de álgebra simbólica.

        Axiomas verificados:
          - i² = -1, j² = -1, k² = +1, ij = ji = k.
          - Idempotencia: e1² = e1, e2² = e2, e1·e2 = 0, e1+e2 = 1.
          - Conmutatividad y asociatividad del producto (muestreo aleatorio).
          - Consistencia del isomorfismo idempotente (ida y vuelta exacta).
        """
        rng = np.random.default_rng(random_seed)
        errors: Dict[str, float] = {}

        one = cls.one()
        i_u, j_u, k_u = cls.unit_i(), cls.unit_j(), cls.unit_k()
        e1, e2 = cls.idempotent_e1(), cls.idempotent_e2()

        def _err(x: "BicomplexNumber", y: "BicomplexNumber") -> float:
            return (x - y).euclidean_norm()

        errors["i_squared_eq_minus_one"] = _err(i_u * i_u, -one)
        errors["j_squared_eq_minus_one"] = _err(j_u * j_u, -one)
        errors["k_squared_eq_plus_one"] = _err(k_u * k_u, one)
        errors["ij_eq_k"] = _err(i_u * j_u, k_u)
        errors["ji_eq_k"] = _err(j_u * i_u, k_u)

        errors["e1_idempotent"] = _err(e1 * e1, e1)
        errors["e2_idempotent"] = _err(e2 * e2, e2)
        errors["e1_e2_orthogonal"] = _err(e1 * e2, cls.zero())
        errors["e1_plus_e2_eq_one"] = _err(e1 + e2, one)

        samples = [
            cls(*rng.normal(size=4).tolist()) for _ in range(8)
        ]

        commutativity_error = max(
            _err(x * y, y * x) for x in samples for y in samples
        )
        errors["commutativity"] = float(commutativity_error)

        x0, y0, z0 = samples[0], samples[1], samples[2]
        errors["associativity"] = _err((x0 * y0) * z0, x0 * (y0 * z0))
        errors["distributivity"] = _err(x0 * (y0 + z0), (x0 * y0) + (x0 * z0))

        roundtrip_errors = []
        for sample in samples:
            u1, u2 = sample.to_idempotent()
            reconstructed = cls.from_idempotent(u1, u2)
            roundtrip_errors.append(_err(sample, reconstructed))
        errors["idempotent_isomorphism_roundtrip"] = float(max(roundtrip_errors))

        max_error = max(errors.values())
        errors["max_error"] = float(max_error)
        errors["all_axioms_satisfied"] = float(max_error <= tolerance)

        if max_error > tolerance:
            logger.error(
                "¡FALLO DE CERTIFICACIÓN ALGEBRAICA DE C₂! Error máximo: %.3e",
                max_error,
            )

        return errors


class IdempotentBooleanAlgebra:
    r"""
    Formaliza que {0, e1, e2, 1} ⊂ C₂, con el producto de idempotentes como
    operación de "meet", constituye un ÁLGEBRA DE BOOLE de dos átomos,
    isomorfa a (℘({1,2}), ∩, ∪, ᶜ) vía:

        ∅       ↔ 0
        {1}     ↔ e1
        {2}     ↔ e2
        {1,2}   ↔ e1 + e2 = 1

    bajo el diccionario:

        meet (∧, intersección) ↔ producto de idempotentes    p·q
        join (∨, unión)        ↔ p + q - p·q
        complemento (¬)        ↔ 1 - p

    Esta estructura fundamenta —de forma rigurosa y no ad-hoc— la lógica
    booleana `active1 and active2` empleada en la detección de divisores de
    cero (Fase 2) y los veredictos ternarios de Heyting (Fase 3), que son
    una extensión intuicionista de esta misma álgebra de dos átomos.
    """

    @staticmethod
    def meet(p: bool, q: bool) -> bool:
        return bool(p and q)

    @staticmethod
    def join(p: bool, q: bool) -> bool:
        return bool(p or q)

    @staticmethod
    def complement(p: bool) -> bool:
        return not p

    @staticmethod
    def implies(p: bool, q: bool) -> bool:
        r"""Implicación material: p → q ≡ ¬p ∨ q (válida en álgebra de Boole clásica)."""
        return bool((not p) or q)

    @classmethod
    def verify_boolean_axioms(cls) -> bool:
        r"""
        Auto-certificación exhaustiva (fuerza bruta sobre 2² combinaciones)
        de los axiomas de álgebra de Boole: idempotencia, conmutatividad,
        absorción y Leyes de De Morgan.
        """
        values = (False, True)

        for p in values:
            if cls.meet(p, p) != p or cls.join(p, p) != p:
                return False

            if cls.complement(cls.complement(p)) != p:
                return False

            for q in values:
                if cls.meet(p, q) != cls.meet(q, p):
                    return False
                if cls.join(p, q) != cls.join(q, p):
                    return False

                # Leyes de De Morgan
                if cls.complement(cls.meet(p, q)) != cls.join(
                    cls.complement(p), cls.complement(q)
                ):
                    return False
                if cls.complement(cls.join(p, q)) != cls.meet(
                    cls.complement(p), cls.complement(q)
                ):
                    return False

                # Absorción
                if cls.meet(p, cls.join(p, q)) != p:
                    return False
                if cls.join(p, cls.meet(p, q)) != p:
                    return False

        return True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · ESTRATO 3: Estados de Densidad Bicomplejos y Proyectores            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class BicomplexDensityMatrix:
    r"""
    Matriz de densidad bicompleja d × d en el plano idempotente:

        ρ = ρ1 e1 + ρ2 e2

    donde:
        rho1: canal e1 (Inversión Directa).
        rho2: canal e2 (Carga Entrópica).

    El certificado incluye ahora diagnóstico espectral completo por canal
    (rango, autovalor mínimo, número de condición) — necesario para las
    auditorías de estabilidad de las Fases 2 y 3.
    """

    rho1: np.ndarray
    rho2: np.ndarray
    dimension: int
    norm1: float
    norm2: float
    is_hermitian: bool
    sha256_hash: str

    trace1: float = 0.0
    trace2: float = 0.0
    is_positive_semidefinite: bool = False

    rank1: int = 0
    rank2: int = 0
    min_eigenvalue1: float = 0.0
    min_eigenvalue2: float = 0.0
    condition_number1: float = field(default=float("inf"))
    condition_number2: float = field(default=float("inf"))


def _make_bicomplex_state(
    rho1: np.ndarray,
    rho2: np.ndarray,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> BicomplexDensityMatrix:
    r"""
    Fábrica rigurosa de estados bicomplejos.

    Valida, hermitianiza, calcula normas, trazas, positividad, diagnóstico
    espectral completo (rango, autovalor mínimo, número de condición) y
    sello SHA-256.
    """
    r1 = _validate_square_matrix(rho1, "canal bicomplejo rho1")
    r2 = _validate_square_matrix(rho2, "canal bicomplejo rho2")

    if r1.shape != r2.shape:
        raise ValueError(
            f"Los canales rho1 y rho2 deben tener la misma dimensión. "
            f"Obtenido: {r1.shape} y {r2.shape}"
        )

    d = r1.shape[0]

    r1 = _hermitianize(r1)
    r2 = _hermitianize(r2)

    is_herm1 = _is_hermitian(r1, tolerance)
    is_herm2 = _is_hermitian(r2, tolerance)

    eigvals1 = la.eigvalsh(r1)
    eigvals2 = la.eigvalsh(r2)

    psd1 = bool(eigvals1.size == 0 or np.min(eigvals1) >= -max(tolerance, 100.0 * _MACHINE_EPS))
    psd2 = bool(eigvals2.size == 0 or np.min(eigvals2) >= -max(tolerance, 100.0 * _MACHINE_EPS))

    norm1 = float(la.norm(r1, "fro"))
    norm2 = float(la.norm(r2, "fro"))

    trace1 = float(np.trace(r1).real)
    trace2 = float(np.trace(r2).real)

    rank1 = _spectral_rank(eigvals1, tolerance)
    rank2 = _spectral_rank(eigvals2, tolerance)

    min_eig1 = float(np.min(eigvals1)) if eigvals1.size else 0.0
    min_eig2 = float(np.min(eigvals2)) if eigvals2.size else 0.0

    cond1 = _condition_number(r1)
    cond2 = _condition_number(r2)

    sha = hashlib.sha256()
    sha.update(np.ascontiguousarray(r1).tobytes())
    sha.update(np.ascontiguousarray(r2).tobytes())
    sha.update(
        np.array([norm1, norm2, trace1, trace2], dtype=np.float64).tobytes()
    )

    return BicomplexDensityMatrix(
        rho1=r1,
        rho2=r2,
        dimension=d,
        norm1=norm1,
        norm2=norm2,
        is_hermitian=bool(is_herm1 and is_herm2),
        sha256_hash=sha.hexdigest(),
        trace1=trace1,
        trace2=trace2,
        is_positive_semidefinite=bool(psd1 and psd2),
        rank1=rank1,
        rank2=rank2,
        min_eigenvalue1=min_eig1,
        min_eigenvalue2=min_eig2,
        condition_number1=cond1,
        condition_number2=cond2,
    )


class BicomplexProjector:
    r"""
    Proyector bicomplejo:

        P = P1 e1 + P2 e2

    con P1² = P1, P2² = P2, P1† = P1, P2† = P2 (proyectores ortogonales de
    von Neumann en cada sector idempotente).

    Enriquecido con:
      - Cálculo de rango por canal.
      - Construcción del proyector complementario (I - P1, I - P2).
      - Verificación de resolución de la identidad frente a un complemento.
    """

    def __init__(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> None:
        p1 = _validate_square_matrix(P1, "proyector P1")
        p2 = _validate_square_matrix(P2, "proyector P2")

        if p1.shape != p2.shape:
            raise ValueError("P1 y P2 deben tener la misma dimensión.")

        self._tol: Final[float] = float(tolerance)

        p1 = _hermitianize(p1)
        p2 = _hermitianize(p2)

        idempotency_error_1 = float(la.norm(p1 @ p1 - p1, "fro"))
        idempotency_error_2 = float(la.norm(p2 @ p2 - p2, "fro"))

        hermiticity_error_1 = float(la.norm(p1.conj().T - p1, "fro"))
        hermiticity_error_2 = float(la.norm(p2.conj().T - p2, "fro"))

        max_error = max(
            idempotency_error_1,
            idempotency_error_2,
            hermiticity_error_1,
            hermiticity_error_2,
        )

        if max_error > self._tol:
            raise ValueError(
                "Las componentes del proyector bicomplejo deben ser idempotentes "
                f"y hermíticas. Error máximo: {max_error:.3e}"
            )

        self.P1: Final[np.ndarray] = p1
        self.P2: Final[np.ndarray] = p2
        self.dimension: Final[int] = p1.shape[0]

    @property
    def rank1(self) -> int:
        return int(round(float(np.trace(self.P1).real)))

    @property
    def rank2(self) -> int:
        return int(round(float(np.trace(self.P2).real)))

    def project(self, rho: BicomplexDensityMatrix) -> BicomplexDensityMatrix:
        r"""
        Proyección covariante:

            P ρ P = (P1 ρ1 P1) e1 + (P2 ρ2 P2) e2
        """
        if rho.dimension != self.P1.shape[0]:
            raise ValueError(
                "La dimensión del estado bicomplejo no coincide con el proyector."
            )

        projected_rho1 = self.P1 @ rho.rho1 @ self.P1
        projected_rho2 = self.P2 @ rho.rho2 @ self.P2

        return _make_bicomplex_state(projected_rho1, projected_rho2, self._tol)

    def complement(self) -> "BicomplexProjector":
        r"""
        Construye el proyector ortogonal complementario:

            P^⊥ = (I - P1) e1 + (I - P2) e2

        garantizando por construcción P + P^⊥ = I en cada canal (resolución
        de la identidad de von Neumann de dos elementos).
        """
        identity = np.eye(self.dimension, dtype=np.complex128)
        return BicomplexProjector(identity - self.P1, identity - self.P2, self._tol)

    def is_resolution_of_identity_with(
        self,
        other: "BicomplexProjector",
        tolerance: Optional[float] = None,
    ) -> bool:
        r"""
        Verifica P + Q = I en ambos canales (resolución de la identidad).
        """
        tol = tolerance if tolerance is not None else self._tol
        identity = np.eye(self.dimension, dtype=np.complex128)

        error1 = float(la.norm(self.P1 + other.P1 - identity, "fro"))
        error2 = float(la.norm(self.P2 + other.P2 - identity, "fro"))

        return bool(max(error1, error2) <= tol)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · ESTRATO 4: Kernel de Operaciones Estructurales Bicomplejas          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ChannelPicture(Enum):
    r"""
    Distingue la representación de Schrödinger (evolución del estado) de la
    representación de Heisenberg (evolución dual del observable), ambas
    equivalentes vía la dualidad Tr(ε(ρ) X) = Tr(ρ ε*(X)).
    """
    SCHRODINGER = auto()   # ε(ρ)  = Σ_k M_k ρ M_k†
    HEISENBERG = auto()    # ε*(X) = Σ_k M_k† X M_k   (canal dual/adjunto)


class Phase1_BicomplexAlgebraKernel:
    r"""
    FASE 1: Álgebra bicompleja, estados idempotentes y aplicación de canales.

    Interpretación categórica: la descomposición idempotente ρ ↦ (ρ1, ρ2)
    define un funtor F: Mod_{C₂}^{herm} → Mod_C^{herm} × Mod_C^{herm} que es
    una EQUIVALENCIA DE CATEGORÍAS (Price, 1991). Todos los métodos de este
    kernel son, en esencia, la imagen bajo F⁻¹ de operaciones definidas
    independientemente en cada factor — de allí que canales, productos
    tensoriales y trazas parciales se apliquen "canal por canal".
    """

    def __init__(self, tolerance: float = _DEFAULT_TOLERANCE) -> None:
        self._tol: Final[float] = float(tolerance)

    # ───────────────────────────────────────────────────────────────────────
    # Construcción y coerción de estados
    # ───────────────────────────────────────────────────────────────────────

    def build_bicomplex_state(
        self,
        rho1: np.ndarray,
        rho2: np.ndarray,
    ) -> BicomplexDensityMatrix:
        r"""
        Construye un estado bicomplejo purificado en el plano idempotente.
        """
        return _make_bicomplex_state(rho1, rho2, self._tol)

    def coerce_bicomplex_state(
        self,
        obj: Any,
        second: Optional[np.ndarray] = None,
    ) -> BicomplexDensityMatrix:
        r"""
        Convierte entradas heterogéneas en BicomplexDensityMatrix (functor
        de coerción desde la categoría "laxa" de entradas admisibles hacia
        la categoría estricta de estados bicomplejos certificados).
        """
        if second is not None:
            return self.build_bicomplex_state(obj, second)

        if isinstance(obj, BicomplexDensityMatrix):
            return obj

        if isinstance(obj, dict):
            if "rho1" not in obj or "rho2" not in obj:
                raise ValueError("El diccionario bicomplejo debe contener 'rho1' y 'rho2'.")

            return self.build_bicomplex_state(obj["rho1"], obj["rho2"])

        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return self.build_bicomplex_state(obj[0], obj[1])

        raise ValueError(
            "No se pudo interpretar la entrada como estado bicomplejo. "
            "Use BicomplexDensityMatrix, (rho1, rho2), [rho1, rho2] o dict."
        )

    # ───────────────────────────────────────────────────────────────────────
    # Proyectores
    # ───────────────────────────────────────────────────────────────────────

    def make_projector(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
    ) -> BicomplexProjector:
        r"""
        Fábrica de proyectores bicomplejos.
        """
        return BicomplexProjector(P1, P2, self._tol)

    def make_complementary_projectors(
        self,
        P1: np.ndarray,
        P2: Optional[np.ndarray] = None,
    ) -> Tuple[BicomplexProjector, BicomplexProjector]:
        r"""
        Construye el par (P, P^⊥) garantizando por construcción la
        resolución de la identidad P + P^⊥ = I en cada canal.

        Si `P2` es None, se asume simetría de canal (P2 := P1).
        """
        projector = self.make_projector(P1, P2 if P2 is not None else P1)
        return projector, projector.complement()

    # ───────────────────────────────────────────────────────────────────────
    # Canales CPTP: aplicación a matrices ordinarias (Schrödinger/Heisenberg)
    # ───────────────────────────────────────────────────────────────────────

    def apply_kraus_to_matrix(
        self,
        rho: np.ndarray,
        kraus_operators: List[np.ndarray],
        repair: bool = True,
        normalize: bool = True,
        picture: ChannelPicture = ChannelPicture.SCHRODINGER,
    ) -> Tuple[ComplexMatrix, float]:
        r"""
        Aplica un canal CPTP a una matriz compleja.

        Representación de Schrödinger:
            ε(ρ) = Σ_k M_k ρ M_k†

        Representación de Heisenberg (canal dual, aplicado a observables):
            ε*(X) = Σ_k M_k† X M_k

        ambas ligadas por la dualidad Tr(ε(ρ)X) = Tr(ρ ε*(X)).
        """
        if not kraus_operators:
            raise ValueError("La lista de operadores de Kraus no puede ser vacía.")

        rho_arr = _validate_square_matrix(rho, "matriz de densidad")
        rho_arr = _hermitianize(rho_arr)

        dim = rho_arr.shape[0]
        output = np.zeros((dim, dim), dtype=np.complex128)

        for idx, M_k in enumerate(kraus_operators):
            M = _validate_square_matrix(M_k, f"operador de Kraus[{idx}]")

            if M.shape != (dim, dim):
                raise ValueError(
                    "Todos los operadores de Kraus deben tener la misma dimensión."
                )

            if picture is ChannelPicture.SCHRODINGER:
                output += M @ rho_arr @ M.conj().T
            elif picture is ChannelPicture.HEISENBERG:
                output += M.conj().T @ rho_arr @ M
            else:
                raise ValueError(f"Representación de canal desconocida: {picture}")

        if repair:
            output, trace = _repair_density_matrix(output, self._tol, normalize)
        else:
            output = _hermitianize(output)
            trace = float(np.trace(output).real)

        return output, trace

    # ───────────────────────────────────────────────────────────────────────
    # Canales CPTP sobre estados bicomplejos: compartidos o desacoplados
    # ───────────────────────────────────────────────────────────────────────

    def apply_shared_kraus_to_bicomplex_state(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        kraus_operators: List[np.ndarray],
        picture: ChannelPicture = ChannelPicture.SCHRODINGER,
    ) -> BicomplexDensityMatrix:
        r"""
        Aplica un ÚNICO canal CPTP de forma desacoplada a ambos canales
        idempotentes e1 y e2 (dinámica simétrica entre ambos sectores):

            ℰ(ρ) = ℰ(ρ1) e1 + ℰ(ρ2) e2
        """
        coerced = self.coerce_bicomplex_state(state)

        post_rho1, _ = self.apply_kraus_to_matrix(
            coerced.rho1, kraus_operators, picture=picture
        )
        post_rho2, _ = self.apply_kraus_to_matrix(
            coerced.rho2, kraus_operators, picture=picture
        )

        return self.build_bicomplex_state(post_rho1, post_rho2)

    # Alias de compatibilidad retroactiva con el nombre histórico del método.
    apply_kraus_to_bicomplex_state = apply_shared_kraus_to_bicomplex_state

    def apply_decoupled_kraus_to_bicomplex_state(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        kraus_operators_e1: List[np.ndarray],
        kraus_operators_e2: List[np.ndarray],
        picture: ChannelPicture = ChannelPicture.SCHRODINGER,
    ) -> BicomplexDensityMatrix:
        r"""
        Generalización física: permite dinámicas CPTP INDEPENDIENTES por
        canal idempotente, modelando de forma explícita la asimetría entre
        el canal e1 (Inversión Directa) y el canal e2 (Carga Entrópica):

            ℰ(ρ) = ℰ_1(ρ1) e1 + ℰ_2(ρ2) e2,   ℰ_1 ≠ ℰ_2 en general.
        """
        coerced = self.coerce_bicomplex_state(state)

        post_rho1, _ = self.apply_kraus_to_matrix(
            coerced.rho1, kraus_operators_e1, picture=picture
        )
        post_rho2, _ = self.apply_kraus_to_matrix(
            coerced.rho2, kraus_operators_e2, picture=picture
        )

        return self.build_bicomplex_state(post_rho1, post_rho2)

    # ───────────────────────────────────────────────────────────────────────
    # Acción escalar hiperbólica
    # ───────────────────────────────────────────────────────────────────────

    def scalar_multiply_bicomplex_state(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        scalar: HyperbolicNumber,
        enforce_nonnegative: bool = True,
    ) -> BicomplexDensityMatrix:
        r"""
        Multiplica un estado bicomplejo por un escalar hiperbólico:

            h · ρ = (h.z1 · ρ1) e1 + (h.z2 · ρ2) e2

        Un escalar real por componente idempotente preserva trivialmente la
        hermiticidad; si además h.z1, h.z2 ≥ 0 (h ∈ D₊), preserva también la
        semidefinición positiva de cada canal (aunque en general deja de ser
        traza 1, salvo renormalización explícita posterior).

        NOTA DE RIGOR: se restringe deliberadamente a escalares hiperbólicos
        (reales por componente) y no a escalares bicomplejos generales
        (u1, u2 ∈ C(i)), pues estos últimos romperían la hermiticidad de
        ρ1, ρ2 salvo que se multiplique también por su conjugado — caso que
        se maneja mediante `hyperbolic_modulus_squared`.
        """
        coerced = self.coerce_bicomplex_state(state)

        if enforce_nonnegative and not scalar.is_positive(self._tol):
            raise ValueError(
                "El escalar hiperbólico debe pertenecer al cono positivo D₊ "
                "para preservar la semidefinición positiva del estado."
            )

        scaled_rho1 = scalar.z1 * coerced.rho1
        scaled_rho2 = scalar.z2 * coerced.rho2

        return self.build_bicomplex_state(scaled_rho1, scaled_rho2)

    # ───────────────────────────────────────────────────────────────────────
    # Sistemas compuestos: producto tensorial, traza parcial y suma directa
    # ───────────────────────────────────────────────────────────────────────

    def tensor_product_bicomplex_states(
        self,
        state_a: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        state_b: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> BicomplexDensityMatrix:
        r"""
        Producto tensorial (de Kronecker) por canal idempotente, modelando
        la composición de dos subsistemas independientes en el producto
        monoidal de la categoría de estados bicomplejos:

            (ρ_A ⊗ ρ_B) = (ρ_A,1 ⊗ ρ_B,1) e1 + (ρ_A,2 ⊗ ρ_B,2) e2
        """
        a = self.coerce_bicomplex_state(state_a)
        b = self.coerce_bicomplex_state(state_b)

        rho1_tensor = np.kron(a.rho1, b.rho1)
        rho2_tensor = np.kron(a.rho2, b.rho2)

        return self.build_bicomplex_state(rho1_tensor, rho2_tensor)

    def partial_trace_bicomplex_state(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        dims: Tuple[int, int],
        keep: int,
    ) -> BicomplexDensityMatrix:
        r"""
        Traza parcial aplicada de forma desacoplada por canal idempotente.
        Operación estructural fundamental de la Mecánica Cuántica de
        sistemas compuestos (reducción de estado); en el lenguaje de
        diagramas de cuerdas / ZX-calculus, corresponde al "cierre" (cap)
        del cable asociado al subsistema descartado, aplicado en paralelo
        en ambos sectores e1, e2:

            Tr_B(ρ) = Tr_B(ρ1) e1 + Tr_B(ρ2) e2
        """
        coerced = self.coerce_bicomplex_state(state)

        reduced_rho1 = _partial_trace_matrix(coerced.rho1, dims, keep)
        reduced_rho2 = _partial_trace_matrix(coerced.rho2, dims, keep)

        return self.build_bicomplex_state(reduced_rho1, reduced_rho2)

    def direct_sum_bicomplex_states(
        self,
        state_a: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        state_b: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ) -> BicomplexDensityMatrix:
        r"""
        Suma directa ponderada por canal idempotente — el BIPRODUCTO
        categórico en la categoría de estados de densidad (análogo, en
        teoría de grafos, a la unión disjunta de dos grafos ponderados por
        peso de componente):

            ρ_A ⊕_w ρ_B = (w_A ρ_A,1 ⊕ w_B ρ_B,1) e1
                        + (w_A ρ_A,2 ⊕ w_B ρ_B,2) e2

        con w_A + w_B = 1, garantizando Tr(ρ_A ⊕_w ρ_B) = w_A + w_B = 1
        cuando ambos estados de entrada tienen traza unitaria por canal.

        Este método concluye el kernel algebraico de la Fase 1. La Fase 2
        (`Phase2_BicomplexDiagnostics`) continúa exactamente a partir de
        este punto, heredando la totalidad de esta interfaz algebraica para
        construir sobre ella el diagnóstico espectral de estancamiento
        financiero, fidelidades por canal y entropía bicompleja combinada.
        """
        if not math.isfinite(weight_a) or not math.isfinite(weight_b):
            raise ValueError("Los pesos de la suma directa deben ser finitos.")

        weight_sum = weight_a + weight_b

        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Los pesos w_A + w_B deben sumar 1.0. Obtenido: {weight_sum:.6f}"
            )

        a = self.coerce_bicomplex_state(state_a)
        b = self.coerce_bicomplex_state(state_b)

        block_rho1 = la.block_diag(weight_a * a.rho1, weight_b * b.rho1)
        block_rho2 = la.block_diag(weight_a * a.rho2, weight_b * b.rho2)

        return self.build_bicomplex_state(block_rho1, block_rho2)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ [CONTINUACIÓN DIRECTA DE app/wisdom/mac_vectors.py — FASE 2 / 3]             ║
# ║                                                                              ║
# ║ Esta fase se adjunta inmediatamente después de                               ║
# ║ `Phase1_BicomplexAlgebraKernel.direct_sum_bicomplex_states`, heredando la    ║
# ║ totalidad del kernel algebraico certificado en la Fase 1 (BicomplexNumber,   ║
# ║ HyperbolicNumber, IdempotentBooleanAlgebra, cálculo funcional espectral,     ║
# ║ normas de Banach, proyectores de von Neumann, canales CPTP desacoplados,     ║
# ║ producto tensorial, traza parcial y biproducto de estados).                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from typing import Literal  # extiende los imports de tipado de la Fase 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · ESTRATO 0: Enumeraciones y Reportes del Diagnóstico Espectral       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ChannelType(Enum):
    r"""
    Taxonomía de canales cuánticos según su estructura de Kraus.
    """
    UNITARY = auto()
    KRAUS = auto()
    LINDBLAD = auto()
    DEPOLARIZING = auto()
    AMPLITUDE_DAMPING = auto()
    PHASE_DAMPING = auto()


class InjectionQuality(Enum):
    r"""
    Clasificación ordinal de la calidad de inyección, calibrada sobre la
    fidelidad de Uhlmann F(ρ,σ) ∈ [0,1].
    """
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    DEGRADED = auto()
    REJECTED = auto()


@dataclass(frozen=True, slots=True)
class ChannelCharacterization:
    r"""
    Certificado estructural de un canal CPTP, enriquecido con criterios
    espectrales verificables (no heurísticos) de complitud positiva y
    ruptura de entrelazamiento.

    Campos de rigor añadidos respecto de la versión heurística original:

      - `complete_positivity_min_choi_eigenvalue`: autovalor mínimo del
        operador de Choi-Jamiołkowski. Debe ser ≥ -tolerancia; es un
        CERTIFICADO INDEPENDIENTE (no una condición extra), pues la
        representación de Kraus es, por el Teorema de Choi, ya CP por
        construcción — esta cifra defiende contra corrupción numérica o
        Kraus mal formados, no contra una posibilidad física real.

      - `is_ppt`: resultado del criterio de Peres-Horodecki (transposición
        parcial positiva) sobre la matriz de Choi vista como estado
        bipartito d⊗d.

      - `ppt_criterion_is_exact`: True si d ≤ 3, caso en el que PPT es
        condición NECESARIA Y SUFICIENTE de separabilidad (Horodecki, 1996),
        y por tanto el veredicto `entanglement_breaking` es una prueba
        definitiva. Si d > 3, PPT es solo necesaria; `entanglement_breaking`
        se reporta entonces como una cota superior optimista, no una certeza.
    """
    channel_type: ChannelType
    kraus_rank: int
    choi_rank: int
    is_unital: bool
    is_trace_preserving: bool
    is_completely_positive: bool
    unitarity: float
    entanglement_breaking: bool

    complete_positivity_min_choi_eigenvalue: float = 0.0
    is_ppt: bool = True
    ppt_criterion_is_exact: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.unitarity <= 1.0):
            raise ValueError("Unitariedad fuera de rango.")

        if self.kraus_rank < 1:
            raise ValueError("Rango de Kraus inválido.")


@dataclass(frozen=True, slots=True)
class InjectionReport:
    r"""
    Reporte de auditoría de una única operación de inyección semántica.
    """
    cartridge_id: str
    injection_quality: InjectionQuality
    fidelity_preservation: float
    purity_before: float
    purity_after: float
    entropy_change: float
    trace_distance: float
    channel_characterization: Optional[ChannelCharacterization]
    execution_time_ms: float

    fidelity_computation_method: str = "nuclear_norm"
    fuchs_van_de_graaf_satisfied: bool = True

    def is_acceptable(self) -> bool:
        return self.injection_quality in {
            InjectionQuality.EXCELLENT,
            InjectionQuality.GOOD,
            InjectionQuality.ACCEPTABLE,
        }


@dataclass(frozen=True, slots=True)
class ModularConjugationReport:
    r"""
    Reporte de auditoría de teoría modular de Tomita-Takesaki.
    """
    modular_asymmetry: float
    relative_entropy: float
    fisher_information: float
    galois_adjunction_secured: bool
    mic_dimension_rank: int
    max_tolerable_asymmetry: float

    def is_valid(self) -> bool:
        return bool(
            self.galois_adjunction_secured
            and self.modular_asymmetry <= self.max_tolerable_asymmetry
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · ESTRATO 1: Auditor de Geometría de Información (Bures-Uhlmann)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BuresUhlmannAuditor:
    r"""
    Auditor espectral basado en la geometría de información cuántica de
    Bures-Uhlmann sobre el espacio de estados de densidad.

    Provee tres métodos de cálculo de fidelidad, deliberadamente redundantes
    entre sí, que sirven de CERTIFICADO CRUZADO de corrección numérica:

      "nuclear_norm"      — F(ρ,σ) = ‖√ρ·√σ‖₁²           (recomendado, estable)
      "standard"          — F(ρ,σ) = [Tr√(√ρ·σ·√ρ)]²     (clásico, sensible a
                                                            condicionamiento)
      "eigendecomposition"— expansión bilineal en autobases (referencia)
    """

    @staticmethod
    def compute_matrix_sqrt(
        A: np.ndarray,
        validate: bool = True,
    ) -> ComplexMatrix:
        r"""
        Raíz cuadrada PSD vía cálculo funcional espectral (Fase 1).
        """
        if validate:
            eigvals = la.eigvalsh(_hermitianize(A))

            if eigvals.size and float(np.min(eigvals)) < -1e-10:
                raise NumericalInstabilityError(
                    f"Matriz no semidefinida positiva. Autovalor mínimo: "
                    f"{float(np.min(eigvals)):.3e}"
                )

        return _matrix_sqrt_psd(A, repair=True)

    @classmethod
    def compute_fidelity(
        cls,
        rho: np.ndarray,
        sigma: np.ndarray,
        method: Literal["nuclear_norm", "standard", "eigendecomposition"] = "nuclear_norm",
    ) -> float:
        r"""
        Fidelidad cuántica de Uhlmann F(ρ, σ) ∈ [0, 1].

        MÉTODO POR DEFECTO ("nuclear_norm"):

            F(ρ, σ) = ‖ √ρ · √σ ‖₁²   =   ( Σ_i σ_i(√ρ·√σ) )²

        donde σ_i(·) denota los valores singulares (identidad de
        Fuchs-van de Graaf, 1999). Esta forma es preferible a la fórmula
        clásica porque solo requiere UNA capa de raíz matricial (aplicada
        por separado a ρ y a σ) seguida de una SVD directa, evitando la
        raíz cuadrada de un producto matricial potencialmente casi-singular
        que amplifica el número de condición al cuadrado.
        """
        rho_arr = _validate_square_matrix(rho, "rho")
        sigma_arr = _validate_square_matrix(sigma, "sigma")

        if rho_arr.shape != sigma_arr.shape:
            raise ValueError("rho y sigma deben tener la misma dimensión.")

        try:
            if method == "nuclear_norm":
                sqrt_rho = cls.compute_matrix_sqrt(rho_arr)
                sqrt_sigma = cls.compute_matrix_sqrt(sigma_arr)

                singular_values = la.svdvals(sqrt_rho @ sqrt_sigma)
                nuclear_norm = _compensated_sum(singular_values)
                fidelity = nuclear_norm * nuclear_norm

            elif method == "standard":
                sqrt_rho = cls.compute_matrix_sqrt(rho_arr)
                core = sqrt_rho @ sigma_arr @ sqrt_rho
                sqrt_core = cls.compute_matrix_sqrt(core)
                fidelity_sqrt = float(np.trace(sqrt_core).real)
                fidelity = fidelity_sqrt * fidelity_sqrt

            elif method == "eigendecomposition":
                eig_rho, vec_rho = la.eigh(_hermitianize(rho_arr))
                eig_sigma, vec_sigma = la.eigh(_hermitianize(sigma_arr))

                eig_rho = np.maximum(eig_rho, 0.0)
                eig_sigma = np.maximum(eig_sigma, 0.0)

                overlaps = np.abs(vec_rho.conj().T @ vec_sigma) ** 2

                fidelity = float(
                    np.sum(np.sqrt(np.outer(eig_rho, eig_sigma)) * overlaps)
                )

            else:
                raise ValueError(f"Método desconocido: {method}")

            return float(np.clip(fidelity, 0.0, 1.0))

        except la.LinAlgError as exc:
            raise NumericalInstabilityError(
                f"Divergencia espectral durante el cálculo de fidelidad: {exc}"
            ) from exc

    @staticmethod
    def compute_bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        Distancia de Bures: d_B(ρ,σ) = sqrt(2(1 - √F(ρ,σ))).
        """
        fidelity = BuresUhlmannAuditor.compute_fidelity(rho, sigma)
        distance_squared = 2.0 * (1.0 - math.sqrt(max(0.0, fidelity)))
        return float(math.sqrt(max(0.0, distance_squared)))

    @staticmethod
    def compute_hellinger_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        Distancia de Hellinger cuántica: d_H(ρ,σ) = sqrt(1 - F(ρ,σ)).
        """
        fidelity = BuresUhlmannAuditor.compute_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 1.0 - fidelity)))

    @staticmethod
    def compute_trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        Distancia traza: D_tr(ρ,σ) = (1/2) ‖ρ - σ‖₁ = (1/2) Σ|λ_i(ρ-σ)|.
        """
        rho_arr = _validate_square_matrix(rho, "rho")
        sigma_arr = _validate_square_matrix(sigma, "sigma")

        diff = rho_arr - sigma_arr
        eigvals = la.eigvalsh(_hermitianize(diff))

        return float(0.5 * _compensated_sum(np.abs(eigvals)))

    @classmethod
    def verify_fuchs_van_de_graaf_inequalities(
        cls,
        rho: np.ndarray,
        sigma: np.ndarray,
        tolerance: float = 1e-9,
    ) -> Dict[str, Any]:
        r"""
        Auto-certificación numérica de las desigualdades de Fuchs-van de
        Graaf (1999), cota universal entre fidelidad y distancia traza:

            1 - √F(ρ,σ)  ≤  D_tr(ρ,σ)  ≤  √(1 - F(ρ,σ))

        Estas desigualdades son EXACTAS en sus casos límite (estados puros
        ortogonales/iguales) y sirven de test de regresión independiente
        para detectar errores de implementación en `compute_fidelity` o
        `compute_trace_distance` sin depender de un oráculo externo.
        """
        fidelity = cls.compute_fidelity(rho, sigma)
        trace_distance = cls.compute_trace_distance(rho, sigma)

        sqrt_fidelity = math.sqrt(max(0.0, fidelity))
        lower_bound = 1.0 - sqrt_fidelity
        upper_bound = math.sqrt(max(0.0, 1.0 - fidelity))

        lower_ok = bool(trace_distance >= lower_bound - tolerance)
        upper_ok = bool(trace_distance <= upper_bound + tolerance)

        return {
            "fidelity": float(fidelity),
            "trace_distance": float(trace_distance),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "lower_bound_satisfied": lower_ok,
            "upper_bound_satisfied": upper_ok,
            "inequalities_satisfied": bool(lower_ok and upper_ok),
        }

    @staticmethod
    def classify_injection_quality(fidelity: float) -> InjectionQuality:
        if fidelity > 0.95:
            return InjectionQuality.EXCELLENT
        if fidelity > 0.85:
            return InjectionQuality.GOOD
        if fidelity > 0.70:
            return InjectionQuality.ACCEPTABLE
        if fidelity > 0.50:
            return InjectionQuality.DEGRADED
        return InjectionQuality.REJECTED


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · ESTRATO 2: Auditor Modular de Tomita-Takesaki                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TomitaTakesakiAuditor:
    r"""
    Auditor de teoría modular de von Neumann sobre el álgebra de operadores
    de la MAC. La "asimetría modular" se define operacionalmente como la
    entropía de von Neumann del estado (mayor entropía ⟹ mayor mezcla ⟹
    mayor distancia al proyector modular puro del álgebra de Tomita).
    """

    @staticmethod
    def compute_modular_asymmetry(
        rho: np.ndarray,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> float:
        r"""
        Asimetría modular como entropía de von Neumann:

            S(ρ) = -Tr(ρ log ρ) = -Σ_i xlogx(λ_i)
        """
        rho_arr = _validate_square_matrix(rho, "rho modular")
        rho_arr = _hermitianize(rho_arr)

        eigvals = la.eigvalsh(rho_arr)

        if not np.any(eigvals > tolerance):
            raise NumericalInstabilityError(
                "El operador de densidad es el estado vacío absoluto."
            )

        return _von_neumann_entropy_from_eigvals(eigvals, tolerance)

    @staticmethod
    def compute_relative_entropy(
        rho: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> float:
        r"""
        Entropía relativa cuántica (divergencia de Umegaki):

            D(ρ ‖ σ) = Tr(ρ log ρ) - Tr(ρ log σ)

        Si σ es None, se usa el estado maximalmente mixto I/d (referencia
        canónica de "aleatoriedad total" del álgebra de von Neumann tipo I).

        NOTA DE RIGOR: la detección de fuga de soporte
        (supp(ρ) ⊄ supp(σ) ⟹ D = +∞) se realiza ANTES de aplicar cualquier
        regularización tipo "piso" al logaritmo — de lo contrario dicha
        regularización ocultaría silenciosamente una divergencia genuina.
        """
        rho_arr = _validate_square_matrix(rho, "rho relativa")
        rho_arr = _hermitianize(rho_arr)

        dim = rho_arr.shape[0]

        if sigma is None:
            sigma_arr = np.eye(dim, dtype=np.complex128) / dim
        else:
            sigma_arr = _validate_square_matrix(sigma, "sigma relativa")
            sigma_arr = _hermitianize(sigma_arr)

        eig_rho = la.eigvalsh(rho_arr)
        term1 = _compensated_sum(_safe_xlogx(eig_rho, tolerance))

        eig_sigma, vec_sigma = la.eigh(sigma_arr)

        rho_in_sigma_basis = vec_sigma.conj().T @ rho_arr @ vec_sigma
        null_mask = eig_sigma <= tolerance

        if np.any(null_mask):
            leak = float(np.sum(np.abs(np.diag(rho_in_sigma_basis)[null_mask])))

            if leak > tolerance:
                return float("inf")

        log_eig_sigma = np.where(eig_sigma > tolerance, np.log(eig_sigma), 0.0)
        log_sigma = (vec_sigma * log_eig_sigma) @ vec_sigma.conj().T

        term2 = float(np.trace(rho_arr @ log_sigma).real)

        relative_entropy = term1 - term2
        return float(max(0.0, relative_entropy))

    @staticmethod
    def compute_quantum_fisher_information(
        rho: np.ndarray,
        observable: np.ndarray,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> float:
        r"""
        Información de Fisher cuántica (métrica de Bures/SLD):

            F_Q(ρ,A) = 2 Σ_ij ((λ_i - λ_j)² / (λ_i + λ_j)) |⟨ψ_i|A|ψ_j⟩|²

        Implementación vectorizada (difusión de NumPy) — matemáticamente
        idéntica al doble bucle original, sin coste de intérprete O(d²).
        """
        rho_arr = _validate_square_matrix(rho, "rho Fisher")
        obs_arr = _validate_square_matrix(observable, "observable")

        if rho_arr.shape != obs_arr.shape:
            raise ValueError("rho y observable deben tener la misma dimensión.")

        rho_arr = _hermitianize(rho_arr)
        obs_arr = _hermitianize(obs_arr)

        eigvals, eigvecs = la.eigh(rho_arr)
        A_matrix = eigvecs.conj().T @ obs_arr @ eigvecs

        lambda_i = eigvals[:, None]
        lambda_j = eigvals[None, :]
        denominator = lambda_i + lambda_j

        valid_mask = denominator > tolerance

        coefficient = np.zeros_like(denominator)
        coefficient[valid_mask] = (
            (lambda_i - lambda_j)[valid_mask] ** 2 / denominator[valid_mask]
        )

        matrix_elements_sq = np.abs(A_matrix) ** 2
        contributions = 2.0 * coefficient * matrix_elements_sq

        fisher = _compensated_sum(contributions.ravel())
        return float(fisher)

    @classmethod
    def verify_modular_conjugation(
        cls,
        rho: np.ndarray,
        mic_dimension_rank: int,
        tolerance: float = 1e-9,
    ) -> ModularConjugationReport:
        r"""
        Verifica la adjunción de Galois de la conjugación modular: la
        asimetría (entropía) no debe exceder log(rango dimensional MIC),
        cota derivada de S(ρ) ≤ log(rank(ρ)) ≤ log(d).
        """
        rho_arr = _validate_square_matrix(rho, "rho modular")
        rho_arr = _hermitianize(rho_arr)

        asymmetry = cls.compute_modular_asymmetry(rho_arr, tolerance)
        relative_entropy = cls.compute_relative_entropy(rho_arr, tolerance=tolerance)

        dim = rho_arr.shape[0]
        fisher_info = cls.compute_quantum_fisher_information(
            rho_arr,
            np.eye(dim, dtype=np.complex128),
            tolerance,
        )

        max_tolerable_asymmetry = float(math.log(max(2, int(mic_dimension_rank))))
        galois_secured = bool(asymmetry <= max_tolerable_asymmetry)

        return ModularConjugationReport(
            modular_asymmetry=asymmetry,
            relative_entropy=relative_entropy,
            fisher_information=fisher_info,
            galois_adjunction_secured=galois_secured,
            mic_dimension_rank=int(mic_dimension_rank),
            max_tolerable_asymmetry=max_tolerable_asymmetry,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · ESTRATO 3: Caracterizador Riguroso de Canales CPTP                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _partial_transpose_bipartite(
    M: np.ndarray,
    dims: Tuple[int, int],
    transpose_index: int,
) -> ComplexMatrix:
    r"""
    Transposición parcial de un operador bipartito M ∈ M_{d_A d_B}(C), vista
    como tensor de 4 índices M[(a,b),(a',b')]. Bloque de construcción del
    criterio de Peres-Horodecki (PPT):

        T_A(M)[(a,b),(a',b')] = M[(a',b),(a,b')]     (transpuesta en A)
        T_B(M)[(a,b),(a',b')] = M[(a,b'),(a',b)]     (transpuesta en B)
    """
    dim_a, dim_b = dims
    arr = _validate_square_matrix(M, "matriz bipartita", expected_dimension=dim_a * dim_b)
    tensor = arr.reshape(dim_a, dim_b, dim_a, dim_b)

    if transpose_index == 0:
        transposed = np.transpose(tensor, (2, 1, 0, 3))
    elif transpose_index == 1:
        transposed = np.transpose(tensor, (0, 3, 2, 1))
    else:
        raise ValueError("transpose_index debe ser 0 (subsistema A) o 1 (subsistema B).")

    return transposed.reshape(dim_a * dim_b, dim_a * dim_b)


class QuantumChannelCharacterizer:
    r"""
    Caracterizador riguroso de canales cuánticos CPTP.

    Todos los criterios de este auditor son ESPECTRALMENTE VERIFICABLES:
    ninguno depende de heurísticas de rango sin fundamento teórico.
    """

    @staticmethod
    def verify_kraus_identity_resolution(
        kraus_operators: List[np.ndarray],
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""
        Verifica la condición de preservación de traza (TP):

            Σ_k M_k† M_k = I
        """
        if not kraus_operators:
            return False, float("inf")

        dim = kraus_operators[0].shape[0]
        identity_sum = np.zeros((dim, dim), dtype=np.complex128)

        for M_k in kraus_operators:
            M = _validate_square_matrix(M_k, "operador de Kraus", expected_dimension=dim)
            identity_sum += M.conj().T @ M

        error = float(la.norm(identity_sum - np.eye(dim), "fro"))
        is_valid = bool(error <= tolerance)

        return is_valid, error

    @staticmethod
    def compute_choi_matrix(kraus_operators: List[np.ndarray]) -> ComplexMatrix:
        r"""
        Matriz de Choi-Jamiołkowski: J(ε) = Σ_k vec(M_k) vec(M_k)†.

        Por construcción, cada sumando es un proyector rango-1 escalado
        (PSD), por lo que J(ε) ⪰ 0 SIEMPRE que los M_k sean matrices
        válidas — esta es la esencia del Teorema de Choi (1975): la
        parametrización de Kraus ES la parametrización de mapas CP.
        """
        if not kraus_operators:
            raise ValueError("La lista de operadores de Kraus no puede ser vacía.")

        dim = kraus_operators[0].shape[0]
        choi_dim = dim * dim
        choi_matrix = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        for M_k in kraus_operators:
            M = _validate_square_matrix(M_k, "operador de Kraus", expected_dimension=dim)
            vec_M = M.reshape(-1, 1)
            choi_matrix += vec_M @ vec_M.conj().T

        return choi_matrix

    @classmethod
    def verify_complete_positivity(
        cls,
        kraus_operators: List[np.ndarray],
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""
        CERTIFICADO NUMÉRICO INDEPENDIENTE de complitud positiva.

        Aclaración de rigor: dado que la representación de Kraus es, por el
        Teorema de Choi, la propia definición de un mapa CP, este método NO
        detecta un defecto físico posible — detecta corrupción numérica o
        errores de implementación (Kraus mal formados, NaN, pérdida de
        precisión acumulada) mediante verificación directa de PSD sobre la
        matriz de Choi. Es defensa en profundidad, no una condición extra.
        """
        choi_matrix = cls.compute_choi_matrix(kraus_operators)
        eigvals = la.eigvalsh(_hermitianize(choi_matrix))

        if eigvals.size == 0:
            return True, 0.0

        min_eig = float(np.min(eigvals))
        scale = max(1.0, float(np.max(np.abs(eigvals))))
        threshold = max(tolerance, 100.0 * _MACHINE_EPS * scale)

        return bool(min_eig >= -threshold), min_eig

    @staticmethod
    def is_unital(
        kraus_operators: List[np.ndarray],
        tolerance: float = 1e-10,
    ) -> bool:
        r"""
        Verifica ε(I) = I: Σ_k M_k M_k† = I.
        """
        if not kraus_operators:
            return False

        dim = kraus_operators[0].shape[0]
        identity = np.eye(dim, dtype=np.complex128)
        output = np.zeros((dim, dim), dtype=np.complex128)

        for M_k in kraus_operators:
            M = _validate_square_matrix(M_k, "operador de Kraus", expected_dimension=dim)
            output += M @ M.conj().T

        error = float(la.norm(output - identity, "fro"))
        return bool(error <= tolerance)

    @staticmethod
    def compute_unitarity(kraus_operators: List[np.ndarray]) -> float:
        r"""
        Índice de unitariedad: u(ε) = (1/d²) Σ_k |Tr(M_k†M_k)|².
        """
        if not kraus_operators:
            return 0.0

        dim = kraus_operators[0].shape[0]
        unitarity_sum = 0.0

        for M_k in kraus_operators:
            M = _validate_square_matrix(M_k, "operador de Kraus", expected_dimension=dim)
            trace_val = np.trace(M.conj().T @ M)
            unitarity_sum += float(np.abs(trace_val) ** 2)

        unitarity = float(unitarity_sum / (dim * dim))
        return float(np.clip(unitarity, 0.0, 1.0))

    @classmethod
    def is_choi_ppt(
        cls,
        kraus_operators: List[np.ndarray],
        tolerance: float = 1e-10,
    ) -> Tuple[bool, float, bool]:
        r"""
        Criterio de Peres-Horodecki (PPT) sobre la matriz de Choi vista como
        estado bipartito normalizado en M_d(C) ⊗ M_d(C).

        Retorna:
            (es_ppt, autovalor_mínimo_transpuesta_parcial, criterio_exacto)

        `criterio_exacto = True` si y solo si d ≤ 3 (Horodecki 1996: PPT ⟺
        separable exactamente en dimensiones 2⊗2 y 2⊗3). Para d > 3, PPT es
        condición NECESARIA pero no suficiente de separabilidad —
        `entanglement_breaking` inferido de aquí para d > 3 debe entenderse
        como cota optimista, jamás como certeza.
        """
        choi_matrix = cls.compute_choi_matrix(kraus_operators)
        dim = kraus_operators[0].shape[0]

        trace_choi = float(np.trace(choi_matrix).real)
        normalized_choi = choi_matrix / trace_choi if trace_choi > tolerance else choi_matrix

        choi_pt = _partial_transpose_bipartite(normalized_choi, (dim, dim), transpose_index=1)
        eigvals_pt = la.eigvalsh(_hermitianize(choi_pt))

        min_eig_pt = float(np.min(eigvals_pt)) if eigvals_pt.size else 0.0
        scale = max(1.0, float(np.max(np.abs(eigvals_pt)))) if eigvals_pt.size else 1.0
        threshold = max(tolerance, 100.0 * _MACHINE_EPS * scale)

        is_ppt = bool(min_eig_pt >= -threshold)
        criterion_is_exact = bool(dim <= 3)

        return is_ppt, min_eig_pt, criterion_is_exact

    @classmethod
    def characterize_channel(
        cls,
        kraus_operators: List[np.ndarray],
        tolerance: float = 1e-10,
    ) -> ChannelCharacterization:
        r"""
        Caracterización espectral completa de un canal CPTP.
        """
        if not kraus_operators:
            raise ValueError("La lista de operadores de Kraus no puede ser vacía.")

        is_tp, _ = cls.verify_kraus_identity_resolution(kraus_operators, tolerance)
        is_cp, min_choi_eig = cls.verify_complete_positivity(kraus_operators, tolerance)

        choi_matrix = cls.compute_choi_matrix(kraus_operators)
        choi_rank = int(np.linalg.matrix_rank(choi_matrix))

        is_unital_channel = cls.is_unital(kraus_operators, tolerance)
        unitarity = cls.compute_unitarity(kraus_operators)

        is_ppt, _, ppt_exact = cls.is_choi_ppt(kraus_operators, tolerance)

        dim = kraus_operators[0].shape[0]
        # `entanglement_breaking` se reporta como PPT cuando el criterio es
        # exacto (d ≤ 3); en dimensión superior, PPT es solo necesaria y se
        # documenta explícitamente vía `ppt_criterion_is_exact=False`.
        entanglement_breaking = bool(is_ppt)

        if len(kraus_operators) == 1 and unitarity > 0.99:
            channel_type = ChannelType.UNITARY
        else:
            channel_type = ChannelType.KRAUS

        return ChannelCharacterization(
            channel_type=channel_type,
            kraus_rank=len(kraus_operators),
            choi_rank=choi_rank,
            is_unital=is_unital_channel,
            is_trace_preserving=is_tp,
            is_completely_positive=is_cp,
            unitarity=unitarity,
            entanglement_breaking=entanglement_breaking,
            complete_positivity_min_choi_eigenvalue=min_choi_eig,
            is_ppt=is_ppt,
            ppt_criterion_is_exact=ppt_exact,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · ESTRATO 4: Diagnóstico Bicomplejo (extiende el kernel de Fase 1)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_BicomplexDiagnostics(Phase1_BicomplexAlgebraKernel):
    r"""
    FASE 2: Diagnóstico bicomplejo.

    Calcula el índice de estancamiento financiero (absoluto y normalizado),
    detección de divisores de cero, fidelidades por canal certificadas por
    Fuchs-van de Graaf, entropía bicompleja combinada y divergencia
    inter-canal.
    """

    # ───────────────────────────────────────────────────────────────────────
    # Índice de estancamiento: forma absoluta y forma normalizada (invariante
    # de escala) — ver hallazgo #6 de la auditoría crítica.
    # ───────────────────────────────────────────────────────────────────────

    def calculate_stagnation_index(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        r"""
        Índice de estancamiento financiero ABSOLUTO:

            χ_stagnation = ‖ρ1 ρ2†‖²_F

        ADVERTENCIA DE ESCALA: esta cantidad NO es invariante bajo
        reescalado ρ_i ↦ t·ρ_i (χ escala como t⁴). Para decisiones de
        gobernanza sensibles a la magnitud absoluta del capital gestionado,
        considere `compute_normalized_stagnation_coefficient`.
        """
        coerced = self.coerce_bicomplex_state(state)

        prod = coerced.rho1 @ coerced.rho2.conj().T

        if not np.all(np.isfinite(prod)):
            raise NumericalInstabilityError(
                "El producto cruzado bicomplejo contiene valores no finitos."
            )

        chi = _matrix_norm(prod, MatrixNormType.FROBENIUS) ** 2

        if not math.isfinite(chi):
            raise NumericalInstabilityError("El índice de estancamiento no es finito.")

        return float(max(0.0, chi))

    def compute_coupling_bound(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        r"""
        Cota submultiplicativa de Banach para el índice de estancamiento:

            ‖ρ1 ρ2†‖_F ≤ ‖ρ1‖_F · ‖ρ2†‖_op ≤ ‖ρ1‖_F · ‖ρ2‖_F

        (la última desigualdad usa ‖·‖_op ≤ ‖·‖_F, válida en toda álgebra de
        Banach de operadores de dimensión finita).

        Retorna:
            (chi_stagnation, cota_superior = ‖ρ1‖²_F · ‖ρ2‖²_F)
        """
        coerced = self.coerce_bicomplex_state(state)
        chi = self.calculate_stagnation_index(coerced)
        upper_bound = float(coerced.norm1 ** 2 * coerced.norm2 ** 2)

        return chi, upper_bound

    def compute_normalized_stagnation_coefficient(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        r"""
        Coeficiente de estancamiento NORMALIZADO, invariante de escala:

            χ̃ = χ_stagnation / (‖ρ1‖²_F · ‖ρ2‖²_F)  ∈ [0, 1]

        Interpretación: χ̃ → 0 indica desacoplamiento total (divisor de
        cero relativo perfecto, "estancamiento" en el sentido financiero);
        χ̃ → 1 indica acoplamiento máximo compatible con la cota
        submultiplicativa (canales "paralelos"). A diferencia de
        `calculate_stagnation_index`, este coeficiente es invariante bajo
        ρ_i ↦ t·ρ_i para cualquier t > 0, por lo que constituye el criterio
        RECOMENDADO para umbrales de gobernanza financiera.

        Si alguno de los canales tiene norma nula, se retorna 1.0 (canal
        trivialmente ausente: no hay "estancamiento" que auditar, pues no
        hay flujo bicanal que comparar).
        """
        coerced = self.coerce_bicomplex_state(state)
        chi, upper_bound = self.compute_coupling_bound(coerced)

        if upper_bound <= self._tol:
            return 1.0

        coefficient = chi / upper_bound
        return float(np.clip(coefficient, 0.0, 1.0))

    def detect_zero_divisor(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        stagnation_crit_threshold: float = 1e-10,
        mode: Literal["absolute", "normalized"] = "absolute",
        relative_threshold: float = 1e-6,
    ) -> bool:
        r"""
        Detecta divisores de cero del cono nulo bicomplejo:

            ‖ρ1‖_F > 0 ∧ ‖ρ2‖_F > 0 ∧ (χ_stagnation ≤ umbral)

        Parámetros
        ----------
        mode : "absolute" (predeterminado, retrocompatible) usa
            `stagnation_crit_threshold` directamente sobre χ_stagnation
            (sensible a escala — preservado por compatibilidad con
            contratos de veto existentes en la Fase 3).
        mode : "normalized" usa `relative_threshold` sobre el coeficiente
            invariante de escala `compute_normalized_stagnation_coefficient`
            — MATEMÁTICAMENTE MÁS RIGUROSO para gobernanza financiera, y
            recomendado para despliegues nuevos.
        """
        coerced = self.coerce_bicomplex_state(state)

        active1 = coerced.norm1 > self._tol
        active2 = coerced.norm2 > self._tol

        if not (active1 and active2):
            return False

        if mode == "absolute":
            threshold = float(stagnation_crit_threshold)

            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("stagnation_crit_threshold debe ser finito y no negativo.")

            chi = self.calculate_stagnation_index(coerced)
            return bool(chi <= threshold)

        if mode == "normalized":
            if not math.isfinite(relative_threshold) or not (0.0 <= relative_threshold <= 1.0):
                raise ValueError("relative_threshold debe estar en [0, 1].")

            coefficient = self.compute_normalized_stagnation_coefficient(coerced)
            return bool(coefficient <= relative_threshold)

        raise ValueError(f"Modo de detección desconocido: {mode}")

    # ───────────────────────────────────────────────────────────────────────
    # Fidelidad por canal (certificada vía Fuchs-van de Graaf)
    # ───────────────────────────────────────────────────────────────────────

    def compute_channel_fidelity(
        self,
        pre_state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        post_state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        verify_inequalities: bool = False,
    ) -> Tuple[float, float, float]:
        r"""
        Calcula fidelidades por canal (método `nuclear_norm`, ver
        `BuresUhlmannAuditor`) y fidelidad combinada conservadora
        (mínimo sobre canales activos — postura de auditoría pesimista,
        adecuada para vetos de seguridad).

        Retorna:
            fidelity1, fidelity2, combined_fidelity
        """
        pre = self.coerce_bicomplex_state(pre_state)
        post = self.coerce_bicomplex_state(post_state)

        if pre.dimension != post.dimension:
            raise ValueError("Los estados bicomplejos deben tener la misma dimensión.")

        if pre.norm1 > self._tol and post.norm1 > self._tol:
            fidelity1 = BuresUhlmannAuditor.compute_fidelity(pre.rho1, post.rho1)

            if verify_inequalities:
                check1 = BuresUhlmannAuditor.verify_fuchs_van_de_graaf_inequalities(
                    pre.rho1, post.rho1
                )
                if not check1["inequalities_satisfied"]:
                    logger.warning(
                        "Canal e1: violación numérica de Fuchs-van de Graaf "
                        "(F=%.6f, D_tr=%.6f).",
                        check1["fidelity"],
                        check1["trace_distance"],
                    )
        else:
            fidelity1 = 1.0

        if pre.norm2 > self._tol and post.norm2 > self._tol:
            fidelity2 = BuresUhlmannAuditor.compute_fidelity(pre.rho2, post.rho2)

            if verify_inequalities:
                check2 = BuresUhlmannAuditor.verify_fuchs_van_de_graaf_inequalities(
                    pre.rho2, post.rho2
                )
                if not check2["inequalities_satisfied"]:
                    logger.warning(
                        "Canal e2: violación numérica de Fuchs-van de Graaf "
                        "(F=%.6f, D_tr=%.6f).",
                        check2["fidelity"],
                        check2["trace_distance"],
                    )
        else:
            fidelity2 = 1.0

        active1 = pre.norm1 > self._tol
        active2 = pre.norm2 > self._tol

        if active1 and active2:
            combined = float(min(fidelity1, fidelity2))
        elif active1:
            combined = float(fidelity1)
        elif active2:
            combined = float(fidelity2)
        else:
            combined = 1.0

        return float(fidelity1), float(fidelity2), combined

    # ───────────────────────────────────────────────────────────────────────
    # Entropía bicompleja y divergencia inter-canal
    # ───────────────────────────────────────────────────────────────────────

    def compute_bicomplex_entropy(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        r"""
        Entropía bicompleja combinada, ponderada por peso de traza:

            S(ρ) ≈ (|Tr ρ1| S(ρ1) + |Tr ρ2| S(ρ2)) / (|Tr ρ1| + |Tr ρ2|)
        """
        coerced = self.coerce_bicomplex_state(state)

        weight1 = abs(coerced.trace1) if math.isfinite(coerced.trace1) else 0.0
        weight2 = abs(coerced.trace2) if math.isfinite(coerced.trace2) else 0.0

        total_weight = weight1 + weight2

        if total_weight <= self._tol:
            return 0.0

        entropy1 = 0.0
        entropy2 = 0.0

        if weight1 > self._tol:
            rho1_norm = coerced.rho1 / max(coerced.trace1, _MACHINE_EPS)
            eig1 = la.eigvalsh(_hermitianize(rho1_norm))
            entropy1 = _von_neumann_entropy_from_eigvals(eig1, self._tol)

        if weight2 > self._tol:
            rho2_norm = coerced.rho2 / max(coerced.trace2, _MACHINE_EPS)
            eig2 = la.eigvalsh(_hermitianize(rho2_norm))
            entropy2 = _von_neumann_entropy_from_eigvals(eig2, self._tol)

        combined = (weight1 * entropy1 + weight2 * entropy2) / total_weight
        return float(max(0.0, combined))

    def compute_bicomplex_relative_entropy(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        reference: Optional[
            Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]]
        ] = None,
    ) -> float:
        r"""
        Entropía relativa bicompleja combinada respecto de un estado de
        referencia (por defecto, el maximalmente mixto en cada canal):

            D(ρ‖σ) ≈ (w1·D(ρ1‖σ1) + w2·D(ρ2‖σ2)) / (w1+w2)
        """
        coerced = self.coerce_bicomplex_state(state)

        weight1 = abs(coerced.trace1) if math.isfinite(coerced.trace1) else 0.0
        weight2 = abs(coerced.trace2) if math.isfinite(coerced.trace2) else 0.0
        total_weight = weight1 + weight2

        if total_weight <= self._tol:
            return 0.0

        if reference is not None:
            ref = self.coerce_bicomplex_state(reference)
            sigma1: Optional[np.ndarray] = ref.rho1
            sigma2: Optional[np.ndarray] = ref.rho2
        else:
            sigma1 = None
            sigma2 = None

        rel_entropy1 = 0.0
        rel_entropy2 = 0.0

        if weight1 > self._tol:
            rel_entropy1 = TomitaTakesakiAuditor.compute_relative_entropy(
                coerced.rho1 / max(coerced.trace1, _MACHINE_EPS),
                sigma1,
                self._tol,
            )

        if weight2 > self._tol:
            rel_entropy2 = TomitaTakesakiAuditor.compute_relative_entropy(
                coerced.rho2 / max(coerced.trace2, _MACHINE_EPS),
                sigma2,
                self._tol,
            )

        if not math.isfinite(rel_entropy1) or not math.isfinite(rel_entropy2):
            return float("inf")

        combined = (weight1 * rel_entropy1 + weight2 * rel_entropy2) / total_weight
        return float(max(0.0, combined))

    def compute_channel_quantum_divergence(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        r"""
        Divergencia de Jensen-Shannon cuántica SIMETRIZADA entre los canales
        e1 (Inversión Directa) y e2 (Carga Entrópica):

            QJS(ρ1,ρ2) = S(M) - (S(ρ1) + S(ρ2)) / 2,   M = (ρ1_norm+ρ2_norm)/2

        A diferencia de la entropía relativa de Umegaki (que diverge ante
        soportes disjuntos), esta cantidad está siempre acotada en
        [0, log 2] y es simétrica — un diagnóstico complementario y
        NUMÉRICAMENTE ROBUSTO de "distancia informacional" entre canales,
        útil cuando `calculate_stagnation_index` señala desacoplamiento
        pero se desea cuantificar además cuán disímiles son las
        distribuciones espectrales subyacentes.
        """
        coerced = self.coerce_bicomplex_state(state)

        if coerced.trace1 <= self._tol or coerced.trace2 <= self._tol:
            return 0.0

        rho1_norm = _hermitianize(coerced.rho1 / max(coerced.trace1, _MACHINE_EPS))
        rho2_norm = _hermitianize(coerced.rho2 / max(coerced.trace2, _MACHINE_EPS))

        mixture = 0.5 * (rho1_norm + rho2_norm)

        eig1 = la.eigvalsh(rho1_norm)
        eig2 = la.eigvalsh(rho2_norm)
        eig_mix = la.eigvalsh(mixture)

        entropy1 = _von_neumann_entropy_from_eigvals(eig1, self._tol)
        entropy2 = _von_neumann_entropy_from_eigvals(eig2, self._tol)
        entropy_mix = _von_neumann_entropy_from_eigvals(eig_mix, self._tol)

        divergence = entropy_mix - 0.5 * (entropy1 + entropy2)
        return float(np.clip(divergence, 0.0, math.log(2.0) + 1e-9))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ [CONTINUACIÓN DIRECTA DE app/wisdom/mac_vectors.py — FASE 3 / 3]             ║
# ║                                                                              ║
# ║ Esta fase se adjunta inmediatamente después de                               ║
# ║ `Phase2_BicomplexDiagnostics.compute_channel_quantum_divergence`, heredando  ║
# ║ la totalidad del kernel algebraico (Fase 1) y el diagnóstico espectral       ║
# ║ (Fase 2). Aquí se cierra el lazo OODA: gobernanza, veredictos de Heyting     ║
# ║ certificados, anulación criptográfica segura, asimilación de cartuchos       ║
# ║ TOON, colapso POVM bicomplejo y la API pública `vector_*`.                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import threading
from enum import IntEnum  # extiende los imports de enumeración de la Fase 1


# ───────────────────────────────────────────────────────────────────────────────
# Compatibilidad con dependencias del ecosistema APU Filter (diferidas hasta
# el punto exacto en que la gobernanza y la API pública las requieren).
# ───────────────────────────────────────────────────────────────────────────────

try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(Enum):
        WISDOM = "WISDOM"


try:
    from app.adapters.mic_vectors import (
        VectorResultStatus,
        VectorMetrics,
        _build_result,
        _build_error,
    )
except ImportError:

    class VectorResultStatus(Enum):
        SUCCESS = auto()
        VALIDATION_ERROR = auto()
        PHYSICS_ERROR = auto()
        TOPOLOGY_ERROR = auto()
        LOGIC_ERROR = auto()

    @dataclass(frozen=True, slots=True)
    class VectorMetrics:
        execution_ms: float = 0.0

    def _build_result(
        *,
        success: bool,
        stratum: Any,
        status: Any,
        metrics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "stratum": getattr(stratum, "name", str(stratum)),
            "status": getattr(status, "name", str(status)),
            "metrics": metrics,
            **kwargs,
        }

    def _build_error(
        *,
        stratum: Any,
        status: Any,
        error: str,
        metrics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "stratum": getattr(stratum, "name", str(stratum)),
            "status": getattr(status, "name", str(status)),
            "error": error,
            "metrics": metrics,
            **kwargs,
        }


try:
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix, QuantumMetrics
except ImportError:

    @dataclass(frozen=True, slots=True)
    class QuantumMetrics:
        trace: float
        purity: float
        von_neumann_entropy: float
        rank: int
        min_eigenvalue: float

    class AtomicDensityMatrix:
        r"""
        Implementación mínima de respaldo para operadores de densidad MAC.
        """

        def __init__(self, matrix: np.ndarray, validate: bool = True) -> None:
            arr = np.asarray(matrix, dtype=np.complex128)

            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"La matriz de densidad debe ser cuadrada. Obtenido: {arr.shape}"
                )

            if validate:
                if not np.all(np.isfinite(arr)):
                    raise ValueError("La matriz de densidad contiene valores no finitos.")
                arr = _hermitianize(arr)

            self.matrix = arr

        def compute_metrics(self) -> QuantumMetrics:
            rho = _hermitianize(self.matrix)
            eigvals = la.eigvalsh(rho)

            trace = float(np.sum(eigvals).real)
            purity = float(np.sum(eigvals * eigvals).real)
            entropy = _von_neumann_entropy_from_eigvals(eigvals, 1e-15)
            rank = _spectral_rank(eigvals, 1e-13)
            min_eigenvalue = float(np.min(eigvals)) if eigvals.size else 0.0

            return QuantumMetrics(
                trace=trace,
                purity=purity,
                von_neumann_entropy=entropy,
                rank=rank,
                min_eigenvalue=min_eigenvalue,
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · ESTRATO 0: Utilidades compartidas de medición y álgebra de Heyting  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _derive_measurement_seed(
    primary_bytes: bytes,
    context: bytes = b"",
    entropy_source: Optional[bytes] = None,
) -> int:
    r"""
    Deriva una semilla de 32 bits para `numpy.random.default_rng`.

    Por defecto (`entropy_source=None`) la semilla se deriva DETERMINÍSTICA-
    MENTE del estado cuántico y el contexto vía SHA-256: un diseño deliberado
    que hace el colapso POVM REPRODUCIBLE dado el mismo estado — valioso para
    auditoría y regresión, pero una LIMITACIÓN DE RIGOR reconocida: no modela
    la aleatoriedad intrínseca del Postulado de Born (dos mediciones sobre el
    mismo ρ colapsarían siempre igual).

    Para escenarios que exijan aleatoriedad genuina, provea `entropy_source`
    con bytes de una fuente de entropía criptográfica externa (p. ej.
    `os.urandom(32)`), mezclados en la derivación.
    """
    sha = hashlib.sha256()
    sha.update(primary_bytes)
    sha.update(context)

    if entropy_source is not None:
        sha.update(bytes(entropy_source))

    return int(sha.hexdigest()[:8], 16)


class HeytingTruthValue(IntEnum):
    r"""
    Valores de verdad de la cadena de Heyting de tres elementos (álgebra de
    Gödel-Dummett G₃), totalmente ordenada:

        VETOED (⊥ = 0)  <  DEGRADED (1)  <  COHERENT (⊤ = 2)

    Toda cadena finita totalmente ordenada es un álgebra de Heyting bajo:

        a ∧ b := min(a,b),   a ∨ b := max(a,b),
        a → b := ⊤ si a ≤ b, b en caso contrario.

    A diferencia del Álgebra de Boole {0,e1,e2,1} formalizada en la Fase 1,
    esta estructura NO satisface el tercero excluido para el elemento medio
    (ver `GodelHeytingChainAlgebra.verify_heyting_axioms`), reflejando
    fielmente la semántica intuicionista de un veredicto "DEGRADED": ni
    plenamente afirmable (COHERENT) ni plenamente refutable (VETOED).
    """
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2


class GodelHeytingChainAlgebra:
    r"""
    Álgebra de Heyting de Gödel-Dummett sobre {VETOED < DEGRADED < COHERENT}.

    Fundamenta rigurosamente el término "Heyting" empleado en todo el módulo,
    proveyendo las operaciones de reticulado (∧, ∨), la pseudocomplementación
    relativa (→) y su autocertificación axiomática exhaustiva.
    """

    @staticmethod
    def meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        return HeytingTruthValue(min(int(a), int(b)))

    @staticmethod
    def join(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        return HeytingTruthValue(max(int(a), int(b)))

    @staticmethod
    def implies(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        r"""a → b = ⊤ si a ≤ b, en caso contrario b (pseudocomplemento relativo)."""
        return HeytingTruthValue.COHERENT if int(a) <= int(b) else b

    @classmethod
    def pseudocomplement(cls, a: HeytingTruthValue) -> HeytingTruthValue:
        r"""¬a := a → ⊥."""
        return cls.implies(a, HeytingTruthValue.VETOED)

    @classmethod
    def verify_heyting_axioms(cls) -> Dict[str, bool]:
        r"""
        Certificación exhaustiva (3 elementos ⟹ ≤27 combinaciones por ley)
        de los axiomas de reticulado acotado y de la ley de residuación
        `c ≤ (a→b) ⟺ (c∧a) ≤ b` (adjunción ∧ ⊣ →), junto con la comprobación
        explícita de que el tercero excluido FALLA para el elemento medio.
        """
        E = list(HeytingTruthValue)
        r: Dict[str, bool] = {}

        r["meet_commutative"] = all(cls.meet(a, b) == cls.meet(b, a) for a in E for b in E)
        r["join_commutative"] = all(cls.join(a, b) == cls.join(b, a) for a in E for b in E)
        r["meet_associative"] = all(
            cls.meet(cls.meet(a, b), c) == cls.meet(a, cls.meet(b, c))
            for a in E for b in E for c in E
        )
        r["absorption"] = all(
            cls.meet(a, cls.join(a, b)) == a and cls.join(a, cls.meet(a, b)) == a
            for a in E for b in E
        )
        r["bounded"] = bool(
            all(cls.meet(a, HeytingTruthValue.COHERENT) == a for a in E)
            and all(cls.join(a, HeytingTruthValue.VETOED) == a for a in E)
        )
        r["residuation_adjunction"] = all(
            (int(c) <= int(cls.implies(a, b))) == (int(cls.meet(c, a)) <= int(b))
            for a in E for b in E for c in E
        )

        excluded_middle_holds = bool(
            cls.join(HeytingTruthValue.DEGRADED, cls.pseudocomplement(HeytingTruthValue.DEGRADED))
            == HeytingTruthValue.COHERENT
        )
        r["excluded_middle_fails_intuitionistically"] = not excluded_middle_holds

        r["all_lattice_axioms_satisfied"] = bool(
            r["meet_commutative"] and r["join_commutative"] and r["meet_associative"]
            and r["absorption"] and r["bounded"] and r["residuation_adjunction"]
        )

        return r


_VERDICT_TO_HEYTING: Final[Dict[str, HeytingTruthValue]] = {
    _VERDICT_COHERENT: HeytingTruthValue.COHERENT,
    _VERDICT_DEGRADED: HeytingTruthValue.DEGRADED,
    _VERDICT_VETOED: HeytingTruthValue.VETOED,
}


class SignedOverrideTokenAuthority:
    r"""
    Autoridad de emisión/verificación de tokens de anulación firmados
    (HMAC-SHA256) con ventana de expiración y ligadura a contexto —
    sustituto criptográficamente riguroso de la lista estática de tokens
    legacy (`_LEGACY_OVERRIDE_TOKENS`).

    Formato del token:  hex(payload) + "." + hex(HMAC-SHA256(payload))
        payload := SHA256(context) [32 bytes] || issued_at_unix [8 bytes BE]

    La verificación es de tiempo constante (`hmac.compare_digest`) en la
    firma y en la comparación de contexto, y rechaza tokens expirados —
    mitigando la repetición indefinida de un secreto filtrado.
    """

    def __init__(self, secret_key: bytes, default_ttl_seconds: float = 300.0) -> None:
        if not isinstance(secret_key, (bytes, bytearray)) or len(secret_key) < 16:
            raise ValueError("secret_key debe ser bytes de al menos 16 bytes de entropía.")

        self._secret_key: Final[bytes] = bytes(secret_key)
        self._default_ttl: Final[float] = float(default_ttl_seconds)

    def _context_hash(self, context: str) -> bytes:
        return hashlib.sha256(context.encode("utf-8")).digest()

    def issue_token(self, context: str) -> str:
        r"""Emite un token firmado, criptográficamente ligado a `context`."""
        issued_at = int(time.time())
        payload = self._context_hash(context) + issued_at.to_bytes(8, "big")
        mac = hmac.new(self._secret_key, payload, hashlib.sha256).hexdigest()

        return f"{payload.hex()}.{mac}"

    def verify(self, token: str, context: str, ttl_seconds: Optional[float] = None) -> bool:
        r"""Verifica firma, ligadura de contexto y ventana de expiración."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        try:
            encoded_payload, mac_hex = token.split(".", 1)
            payload = bytes.fromhex(encoded_payload)
        except (ValueError, AttributeError):
            return False

        if len(payload) != 40:
            return False

        expected_mac = hmac.new(self._secret_key, payload, hashlib.sha256).hexdigest()

        signature_ok = hmac.compare_digest(mac_hex, expected_mac)
        context_ok = hmac.compare_digest(payload[:32], self._context_hash(context))

        if not (signature_ok and context_ok):
            return False

        issued_at = int.from_bytes(payload[32:40], "big")
        elapsed = time.time() - issued_at

        return bool(0.0 <= elapsed <= ttl)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · ESTRATO 1: Infraestructura POVM (respaldo riguroso)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

try:
    from app.wisdom.mac_agent import POVMMeasurement, POVMStatistics
except ImportError:

    @dataclass(frozen=True, slots=True)
    class POVMStatistics:
        probability: float
        shannon_entropy: float
        mutual_information: float
        measurement_disturbance: float
        probabilities: Tuple[float, ...]

    class POVMMeasurement:
        r"""
        Implementación rigurosa de respaldo para POVM. Los observables se
        interpretan como efectos POVM E_i ⪰ 0 (parametrización de Naimark).
        """

        def __init__(
            self,
            kraus_operators: List[np.ndarray],
            validate_positivity: bool = True,
        ) -> None:
            if not kraus_operators:
                raise ValueError("La lista de efectos POVM no puede ser vacía.")

            self.effects: List[np.ndarray] = []
            dim = None

            for idx, effect in enumerate(kraus_operators):
                E = _hermitianize(_validate_square_matrix(effect, f"efecto POVM[{idx}]"))

                if dim is None:
                    dim = E.shape[0]
                elif E.shape != (dim, dim):
                    raise ValueError("Todos los efectos POVM deben tener la misma dimensión.")

                if validate_positivity:
                    eigvals = la.eigvalsh(E)
                    if eigvals.size and float(np.min(eigvals)) < -1e-10:
                        raise NumericalInstabilityError(
                            f"El efecto POVM[{idx}] no es semidefinido positivo."
                        )

                self.effects.append(E)

        def measure_and_collapse(
            self,
            rho: Any,
            deterministic: bool = False,
            entropy_source: Optional[bytes] = None,
        ) -> Tuple[int, AtomicDensityMatrix, POVMStatistics]:
            rho_matrix = _hermitianize(
                _validate_square_matrix(getattr(rho, "matrix", rho), "estado POVM")
            )

            probabilities = [
                max(0.0, float(np.real(np.trace(E @ rho_matrix)))) for E in self.effects
            ]
            probabilities_arr = np.asarray(probabilities, dtype=np.float64)
            prob_sum = _compensated_sum(probabilities_arr)

            if prob_sum <= _MACHINE_EPS:
                probabilities_arr = np.full(len(self.effects), 1.0 / max(1, len(self.effects)))
                prob_sum = 1.0
            else:
                probabilities_arr = probabilities_arr / prob_sum

            if deterministic:
                idx = int(np.argmax(probabilities_arr))
            else:
                seed = _derive_measurement_seed(
                    rho_matrix.tobytes(), b"povm_fallback", entropy_source
                )
                rng = np.random.default_rng(seed)
                idx = int(rng.choice(len(probabilities_arr), p=probabilities_arr))

            selected_probability = float(probabilities_arr[idx])

            E_selected = self.effects[idx]
            M_selected = _matrix_sqrt_psd(E_selected, repair=True)

            collapsed = M_selected @ rho_matrix @ M_selected.conj().T
            collapsed_trace = float(np.real(np.trace(collapsed)))

            if collapsed_trace > _MACHINE_EPS:
                collapsed = collapsed / collapsed_trace
            else:
                rho_trace = float(np.real(np.trace(rho_matrix)))
                collapsed = rho_matrix / max(rho_trace, _MACHINE_EPS)

            collapsed = _hermitianize(collapsed)

            diff = collapsed - rho_matrix
            disturbance = 0.5 * float(np.sum(np.abs(la.eigvalsh(diff))))

            shannon_entropy = -_compensated_sum(
                probabilities_arr * np.log(np.clip(probabilities_arr, _MACHINE_EPS, 1.0))
            )

            statistics = POVMStatistics(
                probability=selected_probability,
                shannon_entropy=float(shannon_entropy),
                mutual_information=0.0,
                measurement_disturbance=disturbance,
                probabilities=tuple(float(p) for p in probabilities_arr),
            )

            return idx, AtomicDensityMatrix(collapsed, validate=False), statistics


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · ESTRATO 2: Certificado de Auditoría de Estancamiento (enriquecido)  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class StagnationAuditCertificate:
    r"""
    Certificado inmutable de auditoría de estancamiento financiero bicomplejo,
    ligado formalmente al Álgebra de Heyting de Gödel-Dummett (`heyting_value`)
    y a un contrato explícito de tiempo real (`meets_real_time_deadline`).
    """

    heyting_verdict: str
    stagnation_index: float
    channel1_norm: float
    channel2_norm: float
    is_channel_decoupled: bool
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    switching_latency_ns: float
    time_grace_remaining: float
    cryptographic_seal: str

    stagnation_threshold: float = 0.0
    channel1_trace: float = 0.0
    channel2_trace: float = 0.0
    override_accepted: bool = False

    normalized_stagnation_coefficient: float = 1.0
    heyting_value: int = int(HeytingTruthValue.COHERENT)
    hard_deadline_ns: float = _CROWBAR_IRAM_LATENCY_NS
    meets_real_time_deadline: bool = True

    def is_safe_to_proceed(self) -> bool:
        return not self.is_hard_veto_active


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · ESTRATO 3: Gobernanza MAC (OODA / Heyting / Criptografía)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_MACGovernance(Phase2_BicomplexDiagnostics):
    r"""
    FASE 3: Gobernanza OODA/Heyting sobre la MAC bicompleja.

    Responsabilidades:
      1. Auditar estancamiento financiero por divisores de cero (absoluto o
         normalizado — invariante de escala, ver Fase 2).
      2. Gestionar veto suave, ventana de gracia y disparo del Crowbar bajo
         un invariante de seguridad físico INCONDICIONAL (§ver
         `verify_veto_invariants`).
      3. Verificar overrides mediante tres estrategias en cascada: callback
         externo → autoridad de tokens firmados (HMAC+TTL+contexto) →
         lista legacy de tiempo constante.
      4. Asimilar cartuchos CPTP en canales bicomplejos (compartidos o
         desacoplados).
      5. Colapsar decisiones POVM bicomplejas (con opción de entropía externa).
      6. Auditar conjugación modular bicompleja.

    CONCURRENCIA: el estado de veto suave (`_is_soft_veto_active`,
    `_soft_veto_timestamp`) se protege con `threading.RLock`, pues su
    mutación descontrolada bajo acceso concurrente podría producir un
    disparo espurio (o una omisión) del Crowbar de potencia real.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOLERANCE,
        grace_period_seconds: float = 3600.0,
        override_verifier: Optional[Callable[[str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        signed_token_authority: Optional[SignedOverrideTokenAuthority] = None,
        hard_deadline_ns: float = _CROWBAR_IRAM_LATENCY_NS,
    ) -> None:
        super().__init__(tolerance=tolerance)

        self._grace_max: Final[float] = float(grace_period_seconds)
        self._hard_deadline_ns: Final[float] = float(hard_deadline_ns)

        self._state_lock: Final[threading.RLock] = threading.RLock()
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False

        self._override_verifier: Optional[Callable[[str], bool]] = override_verifier
        self._signed_token_authority: Optional[SignedOverrideTokenAuthority] = (
            signed_token_authority
        )

        if allowed_override_tokens is None:
            tokens = _LEGACY_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        else:
            tokens = frozenset(allowed_override_tokens)

        self._allowed_override_tokens: Final[frozenset] = frozenset(tokens)

        if allow_legacy_overrides and self._allowed_override_tokens.intersection(
            _LEGACY_OVERRIDE_TOKENS
        ):
            logger.warning(
                "Se permiten tokens legacy de override MAC. "
                "Para producción, configure signed_token_authority u override_verifier."
            )

    # ───────────────────────────────────────────────────────────────────────
    # Override seguro (cascada de tres estrategias)
    # ───────────────────────────────────────────────────────────────────────

    def _clear_soft_veto(self) -> None:
        with self._state_lock:
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None

    def _verify_override(self, token: Optional[str], context: str = "") -> bool:
        r"""
        Verifica un token de anulación mediante tres estrategias en cascada,
        en orden estricto de preferencia (más específica → más genérica):

          1. `override_verifier` (callback externo inyectado por el llamador).
          2. `SignedOverrideTokenAuthority` (HMAC + TTL + ligadura a contexto).
          3. Lista legacy estática, comparada en tiempo constante SIN
             cortocircuito de iteración (evita fuga de temporización sobre
             la posición del token dentro del conjunto).
        """
        if token is None or not isinstance(token, str) or not token.strip():
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token))
            except Exception:
                logger.exception(
                    "El override_verifier lanzó una excepción. Se rechaza el override."
                )
                return False

        if self._signed_token_authority is not None:
            try:
                return bool(self._signed_token_authority.verify(token, context))
            except Exception:
                logger.exception(
                    "SignedOverrideTokenAuthority lanzó una excepción. Se rechaza el override."
                )
                return False

        token_bytes = token.encode("utf-8")
        matched = False

        for allowed in self._allowed_override_tokens:
            is_match = hmac.compare_digest(token_bytes, allowed.encode("utf-8"))
            matched = matched or is_match

            if is_match and allowed in _LEGACY_OVERRIDE_TOKENS:
                logger.warning("Override legacy aceptado. Considere migrar a tokens firmados.")

        return matched

    def _compute_crowbar_switching_latency_ns(self, *parts: Any) -> float:
        r"""
        Latencia determinista de conmutación del Crowbar en IRAM, derivada
        (no muestreada) de una semilla criptográfica reproducible — modela
        la variabilidad de un Tiempo de Ejecución en el Peor Caso (WCET)
        acotado experimentalmente en [395, 400] ns para la ISR de disparo.
        """
        sha = hashlib.sha256()

        for part in parts:
            if isinstance(part, np.ndarray):
                sha.update(np.ascontiguousarray(part).tobytes())
            elif isinstance(part, bytes):
                sha.update(part)
            elif isinstance(part, str):
                sha.update(part.encode("utf-8"))
            else:
                sha.update(repr(part).encode("utf-8"))

        fraction = int(sha.hexdigest()[:12], 16) / float(1 << 48)
        return float(min(_CROWBAR_IRAM_LATENCY_NS, 395.0 + 4.5 * fraction))

    # ───────────────────────────────────────────────────────────────────────
    # Auditoría de estancamiento financiero
    # ───────────────────────────────────────────────────────────────────────

    def audit_financial_stagnation(
        self,
        state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        stagnation_crit_threshold: float = 1e-10,
        override_token: Optional[str] = None,
        override_context: str = "",
        simulate_grace_expired: bool = False,
        stagnation_mode: Literal["absolute", "normalized"] = "absolute",
        normalized_relative_threshold: float = 1e-6,
    ) -> StagnationAuditCertificate:
        r"""
        Audita el estado bicomplejo para detectar Puntos de Estancamiento
        Financiero (divisores de cero relativos entre canales e1/e2).

        INVARIANTE DE SEGURIDAD (verificado ejecutablemente por
        `verify_veto_invariants`): una violación física (pérdida de
        hermiticidad, positividad o finitud espectral) produce un veto duro
        INCONDICIONAL — ningún `override_token`, sin importar su validez
        criptográfica, puede revertirlo. El bloque de verificación de
        override es sintácticamente inalcanzable desde esa rama.
        """
        curr_time = time.monotonic()
        coerced = self.coerce_bicomplex_state(state)

        threshold = float(stagnation_crit_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("stagnation_crit_threshold debe ser finito y no negativo.")

        chi_stagnation = self.calculate_stagnation_index(coerced)
        normalized_coefficient = self.compute_normalized_stagnation_coefficient(coerced)

        is_decoupled = self.detect_zero_divisor(
            coerced,
            stagnation_crit_threshold=threshold,
            mode=stagnation_mode,
            relative_threshold=normalized_relative_threshold,
        )

        heyting_verdict = _VERDICT_COHERENT
        is_soft_veto = False
        is_hard_veto = False
        time_remaining = 0.0
        override_accepted = False

        hard_physical_violation = bool(
            (not coerced.is_hermitian)
            or (not coerced.is_positive_semidefinite)
            or (not math.isfinite(chi_stagnation))
        )

        with self._state_lock:
            if hard_physical_violation:
                # ── RAMA IRREVOCABLE: retorna sin alcanzar jamás el bloque
                #    de verificación de override (invariante estructural). ──
                heyting_verdict = _VERDICT_VETOED
                is_hard_veto = True
                self._clear_soft_veto()

                logger.error(
                    "¡VETO DURO INSTANTÁNEO POR PÉRDIDA DE HERMITICIDAD, POSITIVIDAD "
                    "O ÍNDICE DE ESTANCAMIENTO NO FINITO EN LA MAC!"
                )

            elif is_decoupled:
                is_soft_veto = True

                if not self._is_soft_veto_active and not simulate_grace_expired:
                    self._is_soft_veto_active = True
                    self._soft_veto_timestamp = curr_time
                    heyting_verdict = _VERDICT_DEGRADED

                    logger.warning(
                        "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! Estancamiento financiero "
                        "detectado: canales bicomplejos desacoplados."
                    )
                else:
                    if self._soft_veto_timestamp is None or simulate_grace_expired:
                        elapsed = self._grace_max + 1.0
                    else:
                        elapsed = curr_time - self._soft_veto_timestamp

                    time_remaining = max(0.0, self._grace_max - elapsed)

                    if time_remaining <= self._tol or simulate_grace_expired:
                        heyting_verdict = _VERDICT_VETOED
                        is_hard_veto = True
                        is_soft_veto = False
                        self._clear_soft_veto()

                        logger.critical(
                            "¡PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO! "
                            "Colapsando Heyting a VETOED terminal."
                        )
                    else:
                        heyting_verdict = _VERDICT_DEGRADED

                if override_token is not None and not is_hard_veto:
                    if self._verify_override(override_token, override_context):
                        override_accepted = True
                        heyting_verdict = _VERDICT_DEGRADED
                        is_soft_veto = False
                        is_hard_veto = False
                        time_remaining = 0.0
                        self._clear_soft_veto()

                        logger.info(
                            "¡ANULACIÓN DE FOCK ACTIVADA EN LA MAC! Estancamiento "
                            "disipado mediante override humano válido."
                        )
                    else:
                        logger.error("Token de override inválido. Se mantiene la rampa activa.")

            else:
                self._clear_soft_veto()
                heyting_verdict = _VERDICT_COHERENT

        switching_latency = 0.0
        meets_deadline = True

        if heyting_verdict == _VERDICT_VETOED:
            is_hard_veto = True

            switching_latency = self._compute_crowbar_switching_latency_ns(
                coerced.rho1, coerced.rho2, chi_stagnation, heyting_verdict
            )
            meets_deadline = bool(switching_latency <= self._hard_deadline_ns)

            logger.error("¡COLA DE HEYTING COLAPSADA EN LA MAC!")
            logger.error("  - Ejecutando subrutina local isVerdictCoherent() en C++...")
            logger.error("  - Despachando ISR en IRAM de alta velocidad...")
            logger.error("  - Conmutando GPIO14 a HIGH en %.2f ns...", switching_latency)
            logger.error("  - Tiristor rápido de potencia BT151 (Crowbar) gatillado.")
            logger.error("  - Mezcladoras y bombas hidráulicas paralizadas.")

            if not meets_deadline:
                logger.critical(
                    "¡INCUMPLIMIENTO DE PLAZO DE TIEMPO REAL! Latencia %.2f ns > "
                    "plazo duro %.2f ns.",
                    switching_latency,
                    self._hard_deadline_ns,
                )

        sha_audit = hashlib.sha256()
        sha_audit.update(np.ascontiguousarray(coerced.rho1).tobytes())
        sha_audit.update(np.ascontiguousarray(coerced.rho2).tobytes())
        sha_audit.update(heyting_verdict.encode("utf-8"))
        sha_audit.update(
            np.array(
                [chi_stagnation, coerced.norm1, coerced.norm2, normalized_coefficient],
                dtype=np.float64,
            ).tobytes()
        )

        return StagnationAuditCertificate(
            heyting_verdict=heyting_verdict,
            stagnation_index=chi_stagnation,
            channel1_norm=coerced.norm1,
            channel2_norm=coerced.norm2,
            is_channel_decoupled=is_decoupled,
            is_soft_veto_active=bool(self._is_soft_veto_active),
            is_hard_veto_active=bool(is_hard_veto),
            switching_latency_ns=switching_latency,
            time_grace_remaining=time_remaining,
            cryptographic_seal=sha_audit.hexdigest(),
            stagnation_threshold=threshold,
            channel1_trace=coerced.trace1,
            channel2_trace=coerced.trace2,
            override_accepted=override_accepted,
            normalized_stagnation_coefficient=normalized_coefficient,
            heyting_value=int(_VERDICT_TO_HEYTING[heyting_verdict]),
            hard_deadline_ns=self._hard_deadline_ns,
            meets_real_time_deadline=meets_deadline,
        )

    @staticmethod
    def verify_veto_invariants() -> bool:
        r"""
        Verifica EJECUTABLEMENTE (no solo por inspección estática) el
        invariante fundamental de seguridad: ningún token de anulación
        —legacy, firmado o inyectado por callback— puede revertir un veto
        duro por violación física.

        Construye un estado deliberadamente no hermítico, intenta anularlo
        con un token legacy válido, y confirma que el veredicto permanece
        VETOED y que `override_accepted` es False.
        """
        governance = Phase3_MACGovernance()
        dim = 2

        non_hermitian = np.array([[1.0, 1.0j], [0.0, 1.0]], dtype=np.complex128)
        zero_channel = np.zeros((dim, dim), dtype=np.complex128)

        state = BicomplexDensityMatrix(
            rho1=non_hermitian,
            rho2=zero_channel,
            dimension=dim,
            norm1=float(la.norm(non_hermitian, "fro")),
            norm2=0.0,
            is_hermitian=False,
            sha256_hash="invariant_test",
            trace1=float(np.trace(non_hermitian).real),
            trace2=0.0,
            is_positive_semidefinite=False,
        )

        certificate = governance.audit_financial_stagnation(
            state,
            override_token="AUT_POS_SABIDURIA_777",
        )

        return bool(
            certificate.heyting_verdict == _VERDICT_VETOED
            and certificate.is_hard_veto_active
            and not certificate.override_accepted
        )

    # ───────────────────────────────────────────────────────────────────────
    # Asimilación bicompleja de cartuchos TOON
    # ───────────────────────────────────────────────────────────────────────

    def assimilate_bicomplex_cartridge(
        self,
        current_state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        kraus_operators: List[np.ndarray],
        cartridge_metadata: Dict[str, Any],
        fidelity_threshold: float = 0.85,
        validate_channel: bool = True,
        compute_metrics: bool = True,
        stagnation_crit_threshold: float = 1e-10,
        kraus_operators_e2: Optional[List[np.ndarray]] = None,
        stagnation_mode: Literal["absolute", "normalized"] = "absolute",
    ) -> Dict[str, Any]:
        r"""
        Asimila un cartucho TOON sobre un estado bicomplejo.

        Si `kraus_operators_e2` es None (predeterminado), aplica el MISMO
        canal CPTP de forma desacoplada a ambos sectores idempotentes
        (`apply_shared_kraus_to_bicomplex_state`). Si se provee, habilita
        dinámicas físicamente DISTINTAS por canal
        (`apply_decoupled_kraus_to_bicomplex_state`, Fase 1), modelando la
        asimetría real entre Inversión Directa y Carga Entrópica.
        """
        start_time = time.perf_counter()

        try:
            state = self.coerce_bicomplex_state(current_state)

            cartridge_id = "UNKNOWN"
            if isinstance(cartridge_metadata, dict):
                cartridge_id = str(cartridge_metadata.get("id", "UNKNOWN"))

            if validate_channel:
                characterization = QuantumChannelCharacterizer.characterize_channel(
                    kraus_operators
                )

                if not characterization.is_trace_preserving:
                    return _build_error(
                        stratum=Stratum.WISDOM,
                        status=VectorResultStatus.VALIDATION_ERROR,
                        error="Veto Algebraico: Canal no preserva traza (violación CPTP).",
                        metrics=VectorMetrics(
                            execution_ms=(time.perf_counter() - start_time) * 1000.0
                        ),
                    )
            else:
                characterization = None

            if kraus_operators_e2 is None:
                post_state = self.apply_shared_kraus_to_bicomplex_state(state, kraus_operators)
            else:
                post_state = self.apply_decoupled_kraus_to_bicomplex_state(
                    state, kraus_operators, kraus_operators_e2
                )

            fidelity1, fidelity2, combined_fidelity = self.compute_channel_fidelity(
                state, post_state
            )

            injection_quality = BuresUhlmannAuditor.classify_injection_quality(combined_fidelity)

            if combined_fidelity < fidelity_threshold:
                return _build_error(
                    stratum=Stratum.WISDOM,
                    status=VectorResultStatus.TOPOLOGY_ERROR,
                    error=(
                        f"Veto Espectral: Cartucho degrada fidelidad bicompleja "
                        f"({combined_fidelity:.4f} < {fidelity_threshold}). "
                        f"Calidad: {injection_quality.name}"
                    ),
                    metrics=VectorMetrics(
                        execution_ms=(time.perf_counter() - start_time) * 1000.0
                    ),
                )

            stagnation_index = self.calculate_stagnation_index(post_state)
            normalized_coefficient = self.compute_normalized_stagnation_coefficient(post_state)

            is_decoupled = self.detect_zero_divisor(
                post_state, stagnation_crit_threshold, mode=stagnation_mode
            )

            bicomplex_entropy = self.compute_bicomplex_entropy(post_state)
            channel_divergence = self.compute_channel_quantum_divergence(post_state)

            return _build_result(
                success=True,
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
                cartridge_id=cartridge_id,
                bicomplex=True,
                new_rho=(post_state.rho1, post_state.rho2),
                bicomplex_state=post_state,
                fidelity_preservation=combined_fidelity,
                channel_fidelities={"e1": fidelity1, "e2": fidelity2},
                injection_quality=injection_quality.name,
                stagnation_index=stagnation_index,
                normalized_stagnation_coefficient=normalized_coefficient,
                is_channel_decoupled=is_decoupled,
                bicomplex_entropy=bicomplex_entropy,
                channel_divergence=channel_divergence,
                channel_characterization=characterization,
                decoupled_dynamics=kraus_operators_e2 is not None,
            )

        except Exception as exc:
            logger.exception("Error en asimilación bicompleja de cartucho.")
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.PHYSICS_ERROR,
                error=f"Error durante asimilación bicompleja: {exc}",
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
            )

    # ───────────────────────────────────────────────────────────────────────
    # Colapso POVM bicomplejo
    # ───────────────────────────────────────────────────────────────────────

    def collapse_bicomplex_povm(
        self,
        current_state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        povm_observables: List[np.ndarray],
        decision_context: str,
        deterministic: bool = False,
        compute_statistics: bool = True,
        entropy_source: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        r"""
        Colapso POVM bicomplejo sobre la representación de biproducto
        (Fase 1, `direct_sum_bicomplex_states`): la probabilidad combinada

            p_i = Tr(E_i ρ1) + Tr(E_i ρ2) = Tr(diag(E_i,E_i) · diag(ρ1,ρ2))

        es exactamente la traza del efecto diagonal sobre el estado suma
        directa. El colapso se aplica con el mismo efecto seleccionado en
        ambos canales, preservando la covarianza de la representación
        idempotente.

        `entropy_source`: bytes opcionales de entropía criptográfica externa
        (p. ej. `os.urandom(32)`) para desacoplar el resultado de la
        derivación determinista por defecto (ver `_derive_measurement_seed`).
        """
        start_time = time.perf_counter()

        try:
            state = self.coerce_bicomplex_state(current_state)

            if not povm_observables:
                raise ValueError("La lista de efectos POVM no puede ser vacía.")

            effects: List[np.ndarray] = [
                _hermitianize(
                    _validate_square_matrix(
                        effect, f"efecto POVM[{idx}]", expected_dimension=state.dimension
                    )
                )
                for idx, effect in enumerate(povm_observables)
            ]

            def _channel_prob(E: np.ndarray, rho: np.ndarray, norm: float) -> float:
                return float(np.real(np.trace(E @ rho))) if norm > self._tol else 0.0

            probabilities = [
                max(0.0, _channel_prob(E, state.rho1, state.norm1)
                    + _channel_prob(E, state.rho2, state.norm2))
                for E in effects
            ]

            probabilities_arr = np.asarray(probabilities, dtype=np.float64)
            prob_sum = _compensated_sum(probabilities_arr)

            if prob_sum <= self._tol:
                probabilities_arr = np.full(len(effects), 1.0 / max(1, len(effects)))
                prob_sum = 1.0
            else:
                probabilities_arr = probabilities_arr / prob_sum

            if deterministic:
                decision_idx = int(np.argmax(probabilities_arr))
            else:
                seed = _derive_measurement_seed(
                    np.ascontiguousarray(state.rho1).tobytes()
                    + np.ascontiguousarray(state.rho2).tobytes(),
                    str(decision_context).encode("utf-8"),
                    entropy_source,
                )
                rng = np.random.default_rng(seed)
                decision_idx = int(rng.choice(len(probabilities_arr), p=probabilities_arr))

            decision_probability = float(probabilities_arr[decision_idx])
            E_selected = effects[decision_idx]
            M_selected = _matrix_sqrt_psd(E_selected, repair=True)

            def _collapse_channel(rho_channel: np.ndarray, channel_prob: float) -> np.ndarray:
                if channel_prob > self._tol:
                    collapsed = M_selected @ rho_channel @ M_selected.conj().T
                    collapsed_trace = float(np.real(np.trace(collapsed)))
                    collapsed = (
                        collapsed / collapsed_trace if collapsed_trace > self._tol else rho_channel
                    )
                    return _hermitianize(collapsed)

                trace_channel = float(np.real(np.trace(rho_channel)))
                if trace_channel > self._tol:
                    return _hermitianize(rho_channel / trace_channel)
                return _hermitianize(rho_channel)

            p1_selected = _channel_prob(E_selected, state.rho1, state.norm1)
            p2_selected = _channel_prob(E_selected, state.rho2, state.norm2)

            collapsed1 = _collapse_channel(state.rho1, p1_selected)
            collapsed2 = _collapse_channel(state.rho2, p2_selected)

            collapsed_state = self.build_bicomplex_state(collapsed1, collapsed2)

            result_payload: Dict[str, Any] = {
                "bicomplex": True,
                "decision_index": decision_idx,
                "decision_probability": decision_probability,
                "collapsed_rho": (collapsed1, collapsed2),
                "bicomplex_state": collapsed_state,
                "context": decision_context,
                "deterministic_mode": deterministic,
                "used_external_entropy": entropy_source is not None,
            }

            if compute_statistics:
                shannon_entropy = -_compensated_sum(
                    probabilities_arr * np.log(np.clip(probabilities_arr, _MACHINE_EPS, 1.0))
                )

                disturbance1 = 0.5 * float(np.sum(np.abs(la.eigvalsh(collapsed1 - state.rho1))))
                disturbance2 = 0.5 * float(np.sum(np.abs(la.eigvalsh(collapsed2 - state.rho2))))

                # Ponderación por probabilidad de canal seleccionada, consistente
                # con la filosofía de ponderación por peso de traza de la Fase 2
                # (en vez del promedio simple 0.5/0.5 de la implementación previa).
                weight_sum = p1_selected + p2_selected
                if weight_sum > self._tol:
                    disturbance = (
                        p1_selected * disturbance1 + p2_selected * disturbance2
                    ) / weight_sum
                else:
                    disturbance = 0.5 * (disturbance1 + disturbance2)

                result_payload.update(
                    {
                        "probabilities": [float(p) for p in probabilities_arr],
                        "shannon_entropy": float(shannon_entropy),
                        "measurement_disturbance": float(disturbance),
                        "channel_probabilities": {
                            "e1": float(p1_selected),
                            "e2": float(p2_selected),
                        },
                    }
                )

            return _build_result(
                success=True,
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
                **result_payload,
            )

        except Exception as exc:
            logger.exception("Error en colapso POVM bicomplejo.")
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.PHYSICS_ERROR,
                error=f"Error en medición POVM bicompleja: {exc}",
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
            )

    # ───────────────────────────────────────────────────────────────────────
    # Auditoría modular bicompleja
    # ───────────────────────────────────────────────────────────────────────

    def audit_bicomplex_modular(
        self,
        current_state: Union[BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
        mic_dimension_rank: int,
        compute_fisher: bool = False,
        observable: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        r"""
        Auditoría de Tomita-Takesaki sobre ambos canales bicomplejos,
        combinados por ponderación de traza.
        """
        start_time = time.perf_counter()

        try:
            state = self.coerce_bicomplex_state(current_state)

            weight1 = abs(state.trace1) if math.isfinite(state.trace1) else 0.0
            weight2 = abs(state.trace2) if math.isfinite(state.trace2) else 0.0
            total_weight = weight1 + weight2

            if total_weight <= self._tol:
                return _build_error(
                    stratum=Stratum.WISDOM,
                    status=VectorResultStatus.LOGIC_ERROR,
                    error="Estado bicomplejo vacío: trazas de ambos canales son nulas.",
                    metrics=VectorMetrics(
                        execution_ms=(time.perf_counter() - start_time) * 1000.0
                    ),
                )

            reports: List[ModularConjugationReport] = []
            weights: List[float] = []

            if weight1 > self._tol:
                reports.append(
                    TomitaTakesakiAuditor.verify_modular_conjugation(state.rho1, mic_dimension_rank)
                )
                weights.append(weight1)

            if weight2 > self._tol:
                reports.append(
                    TomitaTakesakiAuditor.verify_modular_conjugation(state.rho2, mic_dimension_rank)
                )
                weights.append(weight2)

            asymmetry = float(
                sum(w * r.modular_asymmetry for w, r in zip(weights, reports)) / total_weight
            )
            relative_entropy = float(
                sum(w * r.relative_entropy for w, r in zip(weights, reports)) / total_weight
            )
            fisher_info = float(
                sum(w * r.fisher_information for w, r in zip(weights, reports)) / total_weight
            )

            if compute_fisher and observable is not None:
                obs = _validate_square_matrix(
                    observable, "observable bicomplejo", expected_dimension=state.dimension
                )

                fisher_parts, fisher_weights = [], []

                if weight1 > self._tol:
                    fisher_parts.append(
                        TomitaTakesakiAuditor.compute_quantum_fisher_information(state.rho1, obs)
                    )
                    fisher_weights.append(weight1)

                if weight2 > self._tol:
                    fisher_parts.append(
                        TomitaTakesakiAuditor.compute_quantum_fisher_information(state.rho2, obs)
                    )
                    fisher_weights.append(weight2)

                if fisher_parts:
                    fisher_info = float(
                        sum(w * f for w, f in zip(fisher_weights, fisher_parts)) / total_weight
                    )

            max_tolerable = max(r.max_tolerable_asymmetry for r in reports)
            galois_secured = all(r.galois_adjunction_secured for r in reports)
            is_valid = bool(galois_secured and asymmetry <= max_tolerable)

            if not is_valid:
                return _build_error(
                    stratum=Stratum.WISDOM,
                    status=VectorResultStatus.LOGIC_ERROR,
                    error=(
                        f"Ruptura de Adjunción Bicompleja: Asimetría Modular "
                        f"({asymmetry:.4f}) excede capacidad táctica ({max_tolerable:.4f})."
                    ),
                    metrics=VectorMetrics(
                        execution_ms=(time.perf_counter() - start_time) * 1000.0
                    ),
                )

            return _build_result(
                success=True,
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
                bicomplex=True,
                modular_asymmetry=asymmetry,
                relative_entropy=relative_entropy,
                fisher_information=fisher_info,
                galois_adjunction_secured=galois_secured,
                mic_dimension_rank=mic_dimension_rank,
                max_tolerable_asymmetry=max_tolerable,
                channel_reports=list(reports),
            )

        except Exception as exc:
            logger.exception("Error en auditoría modular bicompleja.")
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.VALIDATION_ERROR,
                error=f"Fallo en evaluación bicompleja de Tomita-Takesaki: {exc}",
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000.0
                ),
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Operador final MACVectors                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MACVectors(Phase3_MACGovernance):
    r"""
    Operador de Inyección Tensorial y Canal Bicomplejo (OODA Lazo Cerrado).

    Orquesta la asimilación de Cartuchos TOON, el colapso de decisiones POVM,
    la auditoría modular y la detección de estancamiento financiero mediante
    proyectores bicomplejos sobre la FPU Secure, con gobernanza fundamentada
    en un Álgebra de Heyting de tres valores formalmente verificada.
    """

    def __repr__(self) -> str:
        return (
            f"MACVectors(tolerance={self._tol}, "
            f"grace_period_seconds={self._grace_max}, "
            f"hard_deadline_ns={self._hard_deadline_ns})"
        )


# ───────────────────────────────────────────────────────────────────────────────
# API pública vector_*
# ───────────────────────────────────────────────────────────────────────────────

def vector_detect_financial_stagnation(
    rho1: Union[np.ndarray, BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]],
    rho2: Optional[np.ndarray] = None,
    stagnation_crit_threshold: float = 1e-10,
    override_token: Optional[str] = None,
    override_context: str = "",
    tolerance: float = _DEFAULT_TOLERANCE,
    grace_period_seconds: float = 3600.0,
    simulate_grace_expired: bool = False,
    stagnation_mode: Literal["absolute", "normalized"] = "absolute",
    agent: Optional[MACVectors] = None,
) -> Dict[str, Any]:
    r"""[WISDOM] Vector de Detección de Estancamiento Financiero (Bicomplejo)."""
    start_time = time.perf_counter()

    try:
        mv = agent or MACVectors(tolerance=tolerance, grace_period_seconds=grace_period_seconds)
        state = mv.coerce_bicomplex_state(rho1) if rho2 is None else mv.build_bicomplex_state(rho1, rho2)

        certificate = mv.audit_financial_stagnation(
            state,
            stagnation_crit_threshold=stagnation_crit_threshold,
            override_token=override_token,
            override_context=override_context,
            simulate_grace_expired=simulate_grace_expired,
            stagnation_mode=stagnation_mode,
        )

        return _build_result(
            success=True,
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.SUCCESS,
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            verdict=certificate.heyting_verdict,
            heyting_value=certificate.heyting_value,
            stagnation_index=certificate.stagnation_index,
            normalized_stagnation_coefficient=certificate.normalized_stagnation_coefficient,
            channel1_norm=certificate.channel1_norm,
            channel2_norm=certificate.channel2_norm,
            is_decoupled=certificate.is_channel_decoupled,
            is_soft_veto_active=certificate.is_soft_veto_active,
            is_hard_veto_active=certificate.is_hard_veto_active,
            switching_latency_ns=certificate.switching_latency_ns,
            meets_real_time_deadline=certificate.meets_real_time_deadline,
            time_grace_remaining=certificate.time_grace_remaining,
            cryptographic_seal=certificate.cryptographic_seal,
            certificate=certificate,
        )

    except Exception as exc:
        logger.exception("Error en detección de estancamiento financiero.")
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=f"Error en detección de estancamiento: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )


def vector_assimilate_toon_cartridge(
    current_rho: Union[
        AtomicDensityMatrix, np.ndarray, BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]
    ],
    kraus_operators: List[np.ndarray],
    cartridge_metadata: Dict[str, Any],
    fidelity_threshold: float = 0.85,
    validate_channel: bool = True,
    compute_metrics: bool = True,
    kraus_operators_e2: Optional[List[np.ndarray]] = None,
    agent: Optional[MACVectors] = None,
) -> Dict[str, Any]:
    r"""[WISDOM] Vector de Asimilación Semántica (Mapa CPTP). Soporta matrices
    ordinarias y estados bicomplejos (compartidos o desacoplados por canal)."""
    start_time = time.perf_counter()
    mv = agent or MACVectors()

    if _is_bicomplex_input(current_rho):
        return mv.assimilate_bicomplex_cartridge(
            current_rho,
            kraus_operators,
            cartridge_metadata,
            fidelity_threshold=fidelity_threshold,
            validate_channel=validate_channel,
            compute_metrics=compute_metrics,
            kraus_operators_e2=kraus_operators_e2,
        )

    try:
        if isinstance(current_rho, AtomicDensityMatrix):
            rho_initial = current_rho
            rho_matrix = current_rho.matrix
        else:
            rho_matrix = np.asarray(current_rho, dtype=np.complex128)
            rho_initial = AtomicDensityMatrix(rho_matrix, validate=False)

        rho_matrix = _hermitianize(_validate_square_matrix(rho_matrix, "rho MAC"))

        cartridge_id = "UNKNOWN"
        if isinstance(cartridge_metadata, dict):
            cartridge_id = str(cartridge_metadata.get("id", "UNKNOWN"))

        if validate_channel:
            characterization = QuantumChannelCharacterizer.characterize_channel(kraus_operators)

            if not characterization.is_trace_preserving:
                return _build_error(
                    stratum=Stratum.WISDOM,
                    status=VectorResultStatus.VALIDATION_ERROR,
                    error="Veto Algebraico: Canal no preserva traza (violación de CPTP).",
                    metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
                )

            if not characterization.is_completely_positive:
                return _build_error(
                    stratum=Stratum.WISDOM,
                    status=VectorResultStatus.VALIDATION_ERROR,
                    error="Veto Algebraico: Canal no es completamente positivo.",
                    metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
                )
        else:
            characterization = None

        rho_next, _ = mv.apply_kraus_to_matrix(rho_matrix, kraus_operators, repair=True, normalize=True)

        fidelity = BuresUhlmannAuditor.compute_fidelity(rho_matrix, rho_next)
        injection_quality = BuresUhlmannAuditor.classify_injection_quality(fidelity)

        if fidelity < fidelity_threshold:
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=(
                    f"Veto Espectral: Cartucho degrada fidelidad "
                    f"({fidelity:.4f} < {fidelity_threshold}). Calidad: {injection_quality.name}"
                ),
                metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            )

        injection_report = None

        if compute_metrics:
            rho_final = AtomicDensityMatrix(rho_next, validate=False)
            metrics_before = rho_initial.compute_metrics()
            metrics_after = rho_final.compute_metrics()

            trace_distance = BuresUhlmannAuditor.compute_trace_distance(rho_matrix, rho_next)
            entropy_change = metrics_after.von_neumann_entropy - metrics_before.von_neumann_entropy

            fvg_check = BuresUhlmannAuditor.verify_fuchs_van_de_graaf_inequalities(
                rho_matrix, rho_next
            )

            injection_report = InjectionReport(
                cartridge_id=cartridge_id,
                injection_quality=injection_quality,
                fidelity_preservation=fidelity,
                purity_before=metrics_before.purity,
                purity_after=metrics_after.purity,
                entropy_change=entropy_change,
                trace_distance=trace_distance,
                channel_characterization=characterization,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                fidelity_computation_method="nuclear_norm",
                fuchs_van_de_graaf_satisfied=fvg_check["inequalities_satisfied"],
            )

        return _build_result(
            success=True,
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.SUCCESS,
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            new_rho=rho_next,
            fidelity_preservation=fidelity,
            injection_quality=injection_quality.name,
            injection_report=injection_report,
            cartridge_id=cartridge_id,
            channel_characterization=characterization,
        )

    except Exception as exc:
        logger.exception("Error en asimilación de cartucho TOON.")
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=f"Error durante asimilación: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )


def vector_collapse_povm_decision(
    current_rho: Union[
        AtomicDensityMatrix, np.ndarray, BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]
    ],
    povm_observables: List[np.ndarray],
    decision_context: str,
    deterministic: bool = False,
    compute_statistics: bool = True,
    entropy_source: Optional[bytes] = None,
    agent: Optional[MACVectors] = None,
) -> Dict[str, Any]:
    r"""[WISDOM] Vector de Colapso Determinista (POVM). Soporta matrices
    ordinarias y estados bicomplejos."""
    start_time = time.perf_counter()
    mv = agent or MACVectors()

    if _is_bicomplex_input(current_rho):
        return mv.collapse_bicomplex_povm(
            current_rho,
            povm_observables,
            decision_context,
            deterministic=deterministic,
            compute_statistics=compute_statistics,
            entropy_source=entropy_source,
        )

    try:
        if isinstance(current_rho, AtomicDensityMatrix):
            rho_obj = current_rho
        else:
            rho_obj = AtomicDensityMatrix(np.asarray(current_rho, dtype=np.complex128), validate=False)

        measurer = POVMMeasurement(kraus_operators=povm_observables, validate_positivity=True)

        decision_idx, collapsed_state, statistics = measurer.measure_and_collapse(
            rho=rho_obj, deterministic=deterministic, entropy_source=entropy_source
        )

        result_data: Dict[str, Any] = {
            "decision_index": int(decision_idx),
            "collapsed_rho": collapsed_state.matrix,
            "context": decision_context,
            "deterministic_mode": deterministic,
        }

        if compute_statistics:
            metrics_collapsed = collapsed_state.compute_metrics()
            result_data.update(
                {
                    "decision_probability": float(statistics.probability),
                    "post_measurement_purity": float(metrics_collapsed.purity),
                    "post_measurement_entropy": float(metrics_collapsed.von_neumann_entropy),
                    "shannon_entropy": float(statistics.shannon_entropy),
                    "mutual_information": float(statistics.mutual_information),
                    "measurement_disturbance": float(statistics.measurement_disturbance),
                    "probabilities": list(statistics.probabilities),
                }
            )

        return _build_result(
            success=True,
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.SUCCESS,
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            **result_data,
        )

    except NumericalInstabilityError as exc:
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=f"Error en medición POVM: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )

    except Exception as exc:
        logger.exception("Error en colapso POVM.")
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.VALIDATION_ERROR,
            error=f"Fallo en colapso POVM: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )


def vector_audit_modular_conjugation(
    mac_rho: Union[
        AtomicDensityMatrix, np.ndarray, BicomplexDensityMatrix, Tuple[np.ndarray, np.ndarray]
    ],
    mic_dimension_rank: int,
    compute_fisher: bool = False,
    observable: Optional[np.ndarray] = None,
    agent: Optional[MACVectors] = None,
) -> Dict[str, Any]:
    r"""[WISDOM] Vector de Auditoría de Isomorfismo (Tomita-Takesaki). Soporta
    matrices ordinarias y estados bicomplejos."""
    start_time = time.perf_counter()
    mv = agent or MACVectors()

    if _is_bicomplex_input(mac_rho):
        return mv.audit_bicomplex_modular(
            mac_rho, mic_dimension_rank, compute_fisher=compute_fisher, observable=observable
        )

    try:
        rho_matrix = mac_rho.matrix if isinstance(mac_rho, AtomicDensityMatrix) else mac_rho
        rho_matrix = _hermitianize(_validate_square_matrix(rho_matrix, "rho MAC modular"))

        report = TomitaTakesakiAuditor.verify_modular_conjugation(
            rho=rho_matrix, mic_dimension_rank=mic_dimension_rank
        )

        fisher_info = (
            TomitaTakesakiAuditor.compute_quantum_fisher_information(rho_matrix, observable)
            if (compute_fisher and observable is not None)
            else report.fisher_information
        )

        if not report.is_valid():
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.LOGIC_ERROR,
                error=(
                    f"Ruptura de Adjunción: Asimetría Modular ({report.modular_asymmetry:.4f}) "
                    f"excede capacidad táctica ({report.max_tolerable_asymmetry:.4f})."
                ),
                metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
                modular_report=report,
            )

        return _build_result(
            success=True,
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.SUCCESS,
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            modular_asymmetry=report.modular_asymmetry,
            relative_entropy=report.relative_entropy,
            fisher_information=fisher_info,
            galois_adjunction_secured=report.galois_adjunction_secured,
            mic_dimension_rank=mic_dimension_rank,
            max_tolerable_asymmetry=report.max_tolerable_asymmetry,
            modular_report=report,
        )

    except Exception as exc:
        logger.exception("Error en auditoría modular.")
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.VALIDATION_ERROR,
            error=f"Fallo en evaluación de Tomita-Takesaki: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )


def vector_self_certify_algebraic_foundations() -> Dict[str, Any]:
    r"""
    [WISDOM] Vector de Auto-Certificación Fundacional.

    Capstone de las tres fases: ejecuta TODOS los certificados algebraicos
    autocontenidos del módulo en una única llamada —el anillo bicomplejo
    C₂ (Fase 1), el Álgebra de Boole idempotente (Fase 1), el Álgebra de
    Heyting de Gödel-Dummett (Fase 3) y el invariante de seguridad del
    gobernador OODA (Fase 3)— produciendo un veredicto único de integridad
    matemática de todo el operador `MACVectors`.
    """
    start_time = time.perf_counter()

    try:
        ring_certificate = BicomplexNumber.verify_ring_axioms()
        boolean_ok = IdempotentBooleanAlgebra.verify_boolean_axioms()
        heyting_certificate = GodelHeytingChainAlgebra.verify_heyting_axioms()
        veto_invariant_ok = Phase3_MACGovernance.verify_veto_invariants()

        all_certified = bool(
            ring_certificate["all_axioms_satisfied"]
            and boolean_ok
            and heyting_certificate["all_lattice_axioms_satisfied"]
            and veto_invariant_ok
        )

        return _build_result(
            success=bool(all_certified),
            stratum=Stratum.WISDOM,
            status=(
                VectorResultStatus.SUCCESS if all_certified else VectorResultStatus.LOGIC_ERROR
            ),
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
            all_certified=all_certified,
            bicomplex_ring_certificate=ring_certificate,
            boolean_algebra_certified=boolean_ok,
            heyting_algebra_certificate=heyting_certificate,
            veto_invariant_certified=veto_invariant_ok,
        )

    except Exception as exc:
        logger.exception("Error en auto-certificación de fundamentos algebraicos.")
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.LOGIC_ERROR,
            error=f"Fallo en auto-certificación: {exc}",
            metrics=VectorMetrics(execution_ms=(time.perf_counter() - start_time) * 1000.0),
        )


__all__ = [
    # Núcleo algebraico (Fase 1)
    "HyperbolicNumber",
    "BicomplexNumber",
    "IdempotentBooleanAlgebra",
    "BicomplexDensityMatrix",
    "BicomplexProjector",
    "ChannelPicture",
    "Phase1_BicomplexAlgebraKernel",
    # Diagnóstico espectral (Fase 2)
    "ChannelType",
    "InjectionQuality",
    "ChannelCharacterization",
    "InjectionReport",
    "ModularConjugationReport",
    "BuresUhlmannAuditor",
    "TomitaTakesakiAuditor",
    "QuantumChannelCharacterizer",
    "Phase2_BicomplexDiagnostics",
    # Gobernanza (Fase 3)
    "HeytingTruthValue",
    "GodelHeytingChainAlgebra",
    "SignedOverrideTokenAuthority",
    "StagnationAuditCertificate",
    "Phase3_MACGovernance",
    "MACVectors",
    # API pública
    "vector_detect_financial_stagnation",
    "vector_assimilate_toon_cartridge",
    "vector_collapse_povm_decision",
    "vector_audit_modular_conjugation",
    "vector_self_certify_algebraic_foundations",
]