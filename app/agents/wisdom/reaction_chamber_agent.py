# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Reaction Chamber Agent — Evolución Doctoral en 3 Fases Anidadas     ║
║ Ruta   : app/agents/wisdom/reaction_chamber_agent.py                         ║
║ Versión: 3.0.0-Hodge-Smith-Heyting-CFL-IRAM-Governance                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS
========
Soberano de Calibre de la Cámara de Reacción Catalítica Cuántica.

Audita al reactor `reaction_chamber.py` (v4.0.0) mediante un topos de
morfismos anidados Φ₁ → Φ₂ → Φ₃.  Cada fase es un objeto cuya flecha
terminal es el objeto inicial de la siguiente:

    FASE 1  IntegerExactArithmetic.construct_observation_kernel
         │  + CalibratedObserver (validación Banach / sello SHA-256)
         └─► phase1_close_and_open_phase2  ──►  Phase1ReactionHandoff
                │
    FASE 2  HomologicalSpectralAuditor  (continúa el kernel de observación)
         │  Smith / Hückel / Fiedler / CFL / Hodge–Tellegen
         └─► phase2_close_and_open_phase3  ──►  Phase2ReactionHandoff
                │
    FASE 3  HeytingGovernance  (continúa el estado de gobierno espectral)
         │  Clasificador trivalente, gracia, override HMAC, Crowbar, sello
         └─► phase3_close_loop  ──►  ReactionChamberVerdict

CONTINUIDAD FORMAL
==================
    Φ₁→₂ : Phase1ReactionHandoff  →  phase2_from_phase1
    Φ₂→₃ : Phase2ReactionHandoff  →  phase3_from_phase2

El último método de la Fase k invoca de inmediato el morfismo de admisión
de la Fase k+1, de modo que la frontera no es un comentario sino un tipo.

FUNDAMENTOS
===========
Homología entera (Smith):
    ∂₁ : C₁ → C₀  sobre ℤ
    SNF(∂₁) = diag(s₁ | s₂ | … | s_r, 0, …)
    Torsión  ⇔  ∃ k con s_k > 1
    β₀ = dim ker Δ₀  =  #componentes
    β₁ = |E| − |V| + β₀     (ciclo: β₀ = β₁ = 1, χ = 0)

Hückel / Rayleigh–Ritz:
    H = α I + β A
    E(ψ) = ⟨ψ, Hψ⟩ / ⟨ψ, ψ⟩  ≥ λ_min(H)     (Courant–Fischer)
    Estado fundamental  ⇔  E(ψ) ≈ λ_min  y  ‖(H − E I)ψ̂‖ ≈ 0
    Para C₆ 2-regular:  H = (α + 2β) I + |β| L   (β < 0)
    spec(H) = α + 2β cos(2π k / 6)

Fiedler / Cheeger:
    L = D − A  ⪰  0 ,   λ₁ = 0  (si conexo),   λ₂ = a(G)
    h²/2  ≤  λ₂  ≤  2 h         (desigualdades de Cheeger–Buser)

CFL / von Neumann:
    ψ ← (I − α L) ψ
    |1 − α λ| ≤ 1  ∀ λ ∈ spec(L)  ⇒  0 ≤ α ≤ 2 / λ_max
    Cota de trabajo del reactor:  α_crit = 1 / (2 λ_max) = 1/8

Hodge–Tellegen:
    L_adj = D − A ,   L_∂ = ∂ ∂ᵀ
    Compatibilidad:  ‖L_adj − L_∂‖_F  ≤  τ

Heyting trivalente Ω₃ = {⊥, U, ⊤}:
    ⊥ = VETOED ,  U = DEGRADED ,  ⊤ = COHERENT
    a ∧ b = min ,  a ∨ b = max ,  a → b = ⊤ si a ≤ b else b
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time

import numpy as np
import scipy.linalg as la

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)


__all__ = [
    "ReactionChamberAgent",
    "ReactionChamberVerdict",
    "Phase1ReactionHandoff",
    "Phase2ReactionHandoff",
    "HeytingElement",
    "IntegerExactArithmetic",
    "ObservationKernel",
]


__version__ = "3.0.0"

logger = logging.getLogger("APU.Agents.Wisdom.ReactionChamberAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

# Invariantes del reactor v4 (C₆): λ_max(L) = 4 ⇒ α_crit = 1/8.
_RING_SIZE: Final[int] = 6
_LAMBDA_MAX_C6: Final[float] = 4.0
_ALPHA_SHARP_C6: Final[float] = 2.0 / _LAMBDA_MAX_C6
_ALPHA_CRITICAL_C6: Final[float] = 1.0 / (2.0 * _LAMBDA_MAX_C6)

_HUCKEL_ALPHA: Final[float] = 0.20
_HUCKEL_BETA: Final[float] = -0.05


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 — SUSTRATO ARITMÉTICO EXACTO Y OBSERVACIÓN CALIBRADA
# Banach ℓ² / KBN / Bareiss / SNF / kernel de observación
# ═════════════════════════════════════════════════════════════════════════════
#
# Objetos: (ℝⁿ, ‖·‖₂), anillo ℤ con eliminación fraction-free, sello SHA-256.
# El morfismo terminal es
#     IntegerExactArithmetic.construct_observation_kernel
# que phase1_close_and_open_phase2 instala como Phase1ReactionHandoff,
# objeto inicial de la Fase 2.


def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """Proyecta −0.0 → +0.0 para firmas criptográficas deterministas."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Serialización contigua C-order, con ceros signados canónicos."""
    a = np.ascontiguousarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = _canonicalize_signed_zero(a)
    return a.tobytes()


def _is_finite_array(arr: np.ndarray) -> bool:
    """Pertenencia a ℝⁿ (rechaza NaN e infinitos)."""
    return bool(np.all(np.isfinite(arr)))


def _two_sum(a: float, b: float) -> Tuple[float, float]:
    """Transformación libre de error de Knuth–Dekker: a + b = s + e exacto."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def _kbn_sum(values: Iterable[float]) -> float:
    """Sumación compensada Kahan–Babuška–Neumaier."""
    s = 0.0
    c = 0.0
    for raw in values:
        x = float(raw)
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return float(s + c)


def _kbn_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Producto interno euclídeo con compensación KBN."""
    if a.shape != b.shape:
        raise ValueError(
            "Producto interno indefinido para tensores de distinta forma"
        )
    s = 0.0
    c = 0.0
    for x, y in zip(a.ravel(), b.ravel()):
        term = float(x) * float(y)
        t = s + term
        if abs(s) >= abs(term):
            c += (s - t) + term
        else:
            c += (term - t) + s
        s = t
    return float(s + c)


def _frobenius_kbn(matrix: np.ndarray) -> float:
    """Norma de Frobenius ‖A‖_F = √Σ Aᵢⱼ² con KBN."""
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    return math.sqrt(max(_kbn_sum(float(x) * float(x) for x in flat), 0.0))


def _bareiss_det(matrix: List[List[int]]) -> int:
    """
    Determinante exacto sobre ℤ por eliminación fraction-free de Bareiss.

    Invariante: cada división por el pivote precedente es exacta en ℤ
    (teorema de Bareiss).  Overflow imposible: ints de Python son ℤ.
    """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0][0])
    if any(len(row) != n for row in matrix):
        raise ValueError("Bareiss exige matriz cuadrada")

    A = [row[:] for row in matrix]
    sign = 1
    prev = 1

    for k in range(n - 1):
        if A[k][k] == 0:
            swap_row = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    swap_row = i
                    break
            if swap_row is None:
                return 0
            A[k], A[swap_row] = A[swap_row], A[k]
            sign = -sign

        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = A[i][j] * pivot - A[i][k] * A[k][j]
                if prev != 1:
                    if numerator % prev != 0:
                        raise ArithmeticError(
                            "División no exacta en Bareiss (invariante violado)"
                        )
                    A[i][j] = numerator // prev
                else:
                    A[i][j] = numerator

        prev = pivot
        for i in range(k + 1, n):
            A[i][k] = 0
        for j in range(k + 1, n):
            A[k][j] = 0

    return int(sign * A[n - 1][n - 1])


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Identidad de Bézout: g = a x + b y, g = gcd(a, b) ≥ 0."""
    if b == 0:
        if a >= 0:
            return a, 1, 0
        return -a, -1, 0
    g, x1, y1 = _extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


class IntegerExactArithmetic:
    """
    Álgebra exacta sobre ℤ: Bareiss, SNF, divisores determinántales.

    El morfismo terminal `construct_observation_kernel` empaqueta el estado
    observado en un objeto que la Fase 2 consume como complejo de cadenas.
    """

    __slots__ = ()

    @staticmethod
    def smith_normal_form(matrix: List[List[int]]) -> Tuple[List[int], int]:
        """
        Forma normal de Smith sobre ℤ por operaciones elementales unimodulares.

        Retorna (factores_invariantes no nulos s₁ | s₂ | … | s_r, rango r)
        con sᵢ ≥ 1 y sᵢ | sᵢ₊₁.
        """
        if not matrix or not matrix[0]:
            return [], 0

        A = [list(map(int, row)) for row in matrix]
        m = len(A)
        n = len(A[0])
        if any(len(row) != n for row in A):
            raise ValueError("SNF exige matriz rectangular bien formada")

        def swap_rows(i: int, j: int) -> None:
            if i != j:
                A[i], A[j] = A[j], A[i]

        def swap_cols(i: int, j: int) -> None:
            if i == j:
                return
            for row in A:
                row[i], row[j] = row[j], row[i]

        def add_row(dest: int, src: int, k: int) -> None:
            if k == 0:
                return
            for j in range(n):
                A[dest][j] += k * A[src][j]

        def add_col(dest: int, src: int, k: int) -> None:
            if k == 0:
                return
            for i in range(m):
                A[i][dest] += k * A[i][src]

        def mul_row(i: int, k: int) -> None:
            for j in range(n):
                A[i][j] *= k

        def mul_col(j: int, k: int) -> None:
            for i in range(m):
                A[i][j] *= k

        def min_subpivot(r0: int, c0: int) -> Optional[Tuple[int, int, int]]:
            best: Optional[Tuple[int, int, int]] = None
            for i in range(r0, m):
                for j in range(c0, n):
                    val = abs(A[i][j])
                    if val == 0:
                        continue
                    if best is None or val < best[0]:
                        best = (val, i, j)
            return best

        rank = 0
        for d in range(min(m, n)):
            while True:
                found = min_subpivot(d, d)
                if found is None:
                    break
                _, pi, pj = found
                swap_rows(d, pi)
                swap_cols(d, pj)

                pivot = A[d][d]
                if pivot < 0:
                    mul_row(d, -1)
                    pivot = A[d][d]

                reduced = True
                for i in range(d + 1, m):
                    if A[i][d] == 0:
                        continue
                    q = A[i][d] // pivot
                    add_row(i, d, -q)
                    if A[i][d] != 0:
                        reduced = False
                for j in range(d + 1, n):
                    if A[d][j] == 0:
                        continue
                    q = A[d][j] // pivot
                    add_col(j, d, -q)
                    if A[d][j] != 0:
                        reduced = False

                if not reduced:
                    continue

                mixed = False
                for i in range(d + 1, m):
                    for j in range(d + 1, n):
                        if A[i][j] % pivot != 0:
                            add_row(d, i, 1)
                            mixed = True
                            break
                    if mixed:
                        break
                if mixed:
                    continue
                break

            if all(A[d][j] == 0 for j in range(n)) and all(
                A[i][d] == 0 for i in range(m)
            ):
                break
            if A[d][d] == 0:
                break
            rank += 1

        factors = [abs(A[i][i]) for i in range(rank) if A[i][i] != 0]

        # Encadenar divisibilidad sᵢ | sᵢ₊₁ por mezcla unimodular.
        changed = True
        guard = 0
        while changed and guard < rank * rank + 2:
            changed = False
            guard += 1
            for i in range(len(factors) - 1):
                a, b = factors[i], factors[i + 1]
                if a == 0:
                    continue
                g = math.gcd(a, b)
                if g == a:
                    continue
                # sᵢ ← g, sᵢ₊₁ ← sᵢ sᵢ₊₁ / g  (identidad clásica de SNF)
                factors[i] = g
                factors[i + 1] = (a // g) * b
                changed = True

        factors = [abs(int(s)) for s in factors if s != 0]
        return factors, len(factors)

    @staticmethod
    def determinantal_divisors(matrix: List[List[int]]) -> List[int]:
        """
        Δₖ = gcd{ menores k×k }.  Δ₀ := 1.
        Relación clásica: sₖ = Δₖ / Δₖ₋₁ cuando Δₖ₋₁ | Δₖ.
        """
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        r = min(m, n)
        deltas: List[int] = [1]
        for k in range(1, r + 1):
            g = 0
            for rows in combinations(range(m), k):
                for cols in combinations(range(n), k):
                    sub = [[matrix[i][j] for j in cols] for i in rows]
                    try:
                        det = _bareiss_det(sub)
                    except Exception:
                        det = 0
                    g = math.gcd(g, abs(det))
            deltas.append(g)
            if g == 0:
                break
        return deltas

    @classmethod
    def torsion_from_snf(
        cls,
        matrix: List[List[int]],
        *,
        cross_check_minors: bool = True,
    ) -> Tuple[bool, Tuple[int, ...], Dict[str, Any]]:
        """
        Torsión homológica ⇔ algún factor invariante sₖ > 1.

        Para min(m, n) ≤ 6 se cruzan SNF elemental y divisores determinántales.
        """
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return False, (), {"snf_status": "empty"}

        factors, rank = cls.smith_normal_form(matrix)
        has_torsion = any(s > 1 for s in factors)

        diagnostics: Dict[str, Any] = {
            "snf_status": "ok",
            "invariant_factors": tuple(factors),
            "snf_rank": int(rank),
            "has_torsion": bool(has_torsion),
        }

        if cross_check_minors and min(m, n) <= 6:
            deltas = cls.determinantal_divisors(matrix)
            minor_factors: List[int] = []
            torsion_minors = False
            for k in range(1, len(deltas)):
                if deltas[k] == 0 or deltas[k - 1] == 0:
                    break
                if deltas[k] % deltas[k - 1] != 0:
                    torsion_minors = True
                    break
                s = deltas[k] // deltas[k - 1]
                if s > 1:
                    torsion_minors = True
                if s != 0:
                    minor_factors.append(int(s))
            diagnostics["determinantal_divisors"] = tuple(deltas)
            diagnostics["minor_invariant_factors"] = tuple(minor_factors)
            if torsion_minors != has_torsion:
                diagnostics["snf_minors_disagreement"] = True
                has_torsion = has_torsion or torsion_minors

        return bool(has_torsion), tuple(factors), diagnostics

    def construct_observation_kernel(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
        session_sha256: str,
    ) -> "ObservationKernel":
        """
        ── MORFISMO TERMINAL DE LA FASE 1 / INICIAL DE LA FASE 2 ──────────

        Instala el complejo observado (ψ, A, ∂, α, sello) como kernel
        de observación.  La Fase 2 no revalida la pertenencia a ℝⁿ sino
        que opera sobre este objeto ya calibrado.
        """
        degrees = np.sum(adjacency_matrix, axis=1)
        laplacian = np.diag(degrees) - adjacency_matrix
        laplacian = 0.5 * (laplacian + laplacian.T)
        return ObservationKernel(
            state_vector=state_vector,
            adjacency_matrix=adjacency_matrix,
            boundary_matrix=boundary_matrix,
            diffusion_rate=float(diffusion_rate),
            session_sha256=session_sha256,
            degree_vector=np.asarray(degrees, dtype=np.float64),
            laplacian=np.asarray(laplacian, dtype=np.float64),
            n_nodes=int(state_vector.size),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ObservationKernel:
    """
    Kernel de observación (objeto inicial de la Fase 2).

    Continúa `construct_observation_kernel`: el complejo (ψ, A, ∂, L, α)
    queda cerrado y es el dominio de Smith–Hückel–Fiedler–CFL–Hodge.
    """

    state_vector: np.ndarray
    adjacency_matrix: np.ndarray
    boundary_matrix: np.ndarray
    diffusion_rate: float
    session_sha256: str
    degree_vector: np.ndarray
    laplacian: np.ndarray
    n_nodes: int


@dataclass(frozen=True, slots=True, eq=False)
class Phase1ReactionHandoff:
    """
    Frontera formal Φ₁→₂.

    Salida cerrada de la Fase 1 y entrada abierta de la Fase 2.
    """

    state_vector: np.ndarray
    adjacency_matrix: np.ndarray
    boundary_matrix: np.ndarray
    diffusion_rate: float
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str
    observation_kernel: Optional[ObservationKernel] = None


@dataclass(frozen=True, slots=True, eq=False)
class Phase2ReactionHandoff:
    """
    Frontera formal Φ₂→₃.

    Salida cerrada de la Fase 2 y entrada abierta de la Fase 3.
    """

    state_vector: np.ndarray
    adjacency_matrix: np.ndarray
    boundary_matrix: np.ndarray
    diffusion_rate: float
    session_sha256: str
    aromatic_energy: float
    is_aromatic_stable: bool
    has_torsion_anomaly: bool
    is_cfl_stable: bool
    is_connected: bool
    fiedler_value: float
    diagnostics: Dict[str, Any]
    next_entrypoint: str
    invariant_factors: Tuple[int, ...] = ()
    hodge_residual: float = 0.0
    lambda_max_laplacian: float = 0.0
    rayleigh_residual: float = 0.0


class HeytingElement(Enum):
    """
    Álgebra de Heyting trivalente Ω₃ del soberano.

        ⊥ = VETOED ⊂ U = DEGRADED ⊂ ⊤ = COHERENT
        a ∧ b = min ,  a ∨ b = max
        a → b = ⊤  si a ≤ b,  si no b
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingElement") -> "HeytingElement":
        return HeytingElement(min(self.value, other.value))

    def join(self, other: "HeytingElement") -> "HeytingElement":
        return HeytingElement(max(self.value, other.value))

    def implies(self, other: "HeytingElement") -> "HeytingElement":
        if self.value <= other.value:
            return HeytingElement.COHERENT
        return other

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class ReactionChamberVerdict:
    """
    Certificado formal de regularidad espectral y homológica de la cámara.

    Se conservan los campos originales por compatibilidad; la trazabilidad
    se agrega al final con valores por defecto.
    """

    heyting_verdict: str
    aromatic_energy: float
    is_aromatic_stable: bool
    has_torsion_anomaly: bool
    is_cfl_stable: bool
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    switching_latency_ns: float
    time_grace_remaining: float
    cryptographic_seal: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.cryptographic_seal)


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE — 3 FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class ReactionChamberAgent(IntegerExactArithmetic):
    """
    Soberano de Calibre de la Cámara de Reacción Catalítica (OODA cerrado).

    Hereda el sustrato exacto de la Fase 1 (SNF / Bareiss / kernel) e
    instala sobre él las auditorías de la Fase 2 y el gobierno Heyting
    de la Fase 3.

      FASE 1 — OBSERVE:
        Validación Banach, adyacencia, frontera, difusión, sello SHA-256.
        Terminal: phase1_close_and_open_phase2.

      FASE 2 — ORIENT:
        Admite Phase1ReactionHandoff.  Smith, Hückel, Fiedler, CFL, Hodge.
        Terminal: phase2_close_and_open_phase3.

      FASE 3 — DECIDE / ACT:
        Admite Phase2ReactionHandoff.  Heyting, gracia, HMAC, Crowbar, sello.
        Terminal: phase3_close_loop.
    """

    def __init__(
        self,
        fiedler_threshold: float = 1e-5,
        diffusion_crit_rate: float = _ALPHA_CRITICAL_C6,
        grace_period: float = 3600.0,
        tolerance: float = 1e-12,
        strict_homology: bool = True,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 1.2,
        authorized_tokens: Optional[Tuple[str, ...]] = None,
        hmac_key: Optional[bytes] = None,
        require_hodge_compatibility: bool = False,
        hodge_tolerance: float = 1e-8,
    ) -> None:
        """
        Inicializa el soberano de la cámara.

        Parámetros
        ----------
        fiedler_threshold : float
            Umbral mínimo de conectividad algebraica λ₂.
        diffusion_crit_rate : float
            Tasa crítica CFL (por defecto 1/(2 λ_max) = 0.125).
        grace_period : float
            Ventana de gracia del veto suave, en segundos.
        tolerance : float
            Tolerancia numérica base.
        strict_homology : bool
            Si True, fallo de SNF / desacuerdo de menores ⇒ anomalía.
        rng_seed : Optional[int]
            Semilla opcional para jitter determinista del Crowbar.
        jitter_sigma : float
            Desviación estándar del jitter de actuación en ns.
        authorized_tokens : Optional[Tuple[str, ...]]
            Tokens canónicos autorizados para override humano.
        hmac_key : Optional[bytes]
            Clave opcional para tokens HMAC-SHA256 (`payload:signature`).
        require_hodge_compatibility : bool
            Si True, ‖L_A − ∂∂ᵀ‖_F > hodge_tolerance es veto duro.
        hodge_tolerance : float
            Tolerancia de compatibilidad Hodge–Tellegen.
        """
        if fiedler_threshold <= 0.0:
            raise ValueError("fiedler_threshold debe ser estrictamente positiva.")
        if diffusion_crit_rate <= 0.0:
            raise ValueError("diffusion_crit_rate debe ser estrictamente positiva.")
        if grace_period < 0.0:
            raise ValueError("grace_period no puede ser negativa.")
        if tolerance <= 0.0:
            raise ValueError("tolerance debe ser estrictamente positiva.")
        if jitter_sigma < 0.0:
            raise ValueError("jitter_sigma no puede ser negativa.")
        if hodge_tolerance <= 0.0:
            raise ValueError("hodge_tolerance debe ser estrictamente positiva.")

        self._fiedler_min = float(fiedler_threshold)
        self._cfl_crit = float(diffusion_crit_rate)
        self._grace_max = float(grace_period)
        self._tol = float(tolerance)
        self._reg = max(1e-15, self._tol * 1e-3)
        self._strict_homology = bool(strict_homology)
        self._jitter_sigma = float(jitter_sigma)
        self._require_hodge = bool(require_hodge_compatibility)
        self._hodge_tol = float(hodge_tolerance)

        self._rng = np.random.default_rng(rng_seed)

        if authorized_tokens is None:
            self._authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_REACTOR_IDU_2026",
                "HMAC_SUTURA_FOCK_SECURE",
            }
        else:
            self._authorized_tokens = set(authorized_tokens)

        if isinstance(hmac_key, str):
            self._hmac_key: Optional[bytes] = hmac_key.encode("utf-8")
        else:
            self._hmac_key = hmac_key

        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDADORES, SELLO, KERNEL
    # ═════════════════════════════════════════════════════════════════════════

    def _validate_state_vector(self, state_vector: Any) -> np.ndarray:
        """Inmersión ψ ∈ ℝⁿ: no vacío, finito, ceros canónicos."""
        if state_vector is None:
            raise ValueError("state_vector es obligatorio.")
        psi = np.asarray(state_vector, dtype=np.float64).ravel()
        if psi.size == 0:
            raise ValueError("state_vector no puede ser vacío.")
        if not _is_finite_array(psi):
            raise ValueError("state_vector contiene valores NaN o infinitos.")
        return _canonicalize_signed_zero(psi)

    def _validate_adjacency_matrix(
        self,
        adjacency_matrix: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida A ∈ M_n(ℝ): forma, finitud, simetrización de Hilbert–Schmidt.

            A ← (A + Aᵀ) / 2
        """
        if adjacency_matrix is None:
            raise ValueError("adjacency_matrix es obligatoria.")
        A = np.asarray(adjacency_matrix, dtype=np.float64)
        if A.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"adjacency_matrix debe tener forma ({n_nodes}, {n_nodes}). "
                f"Obtenida: {A.shape}"
            )
        if not _is_finite_array(A):
            raise ValueError("adjacency_matrix contiene valores NaN o infinitos.")
        A_sym = 0.5 * (A + A.T)
        return _canonicalize_signed_zero(A_sym)

    def _validate_boundary_matrix(self, boundary_matrix: Any) -> np.ndarray:
        """Valida ∂₁ ∈ M_{p,q}(ℝ) bidimensional y finita."""
        if boundary_matrix is None:
            raise ValueError("boundary_matrix es obligatoria.")
        B = np.asarray(boundary_matrix, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError("boundary_matrix debe ser bidimensional.")
        if not _is_finite_array(B):
            raise ValueError("boundary_matrix contiene valores NaN o infinitos.")
        return B

    def _to_integer_matrix(self, boundary_matrix: np.ndarray) -> List[List[int]]:
        """Retracción ℝ → ℤ: exige coeficientes enteros a 10⁻⁸."""
        arr = np.asarray(boundary_matrix, dtype=np.float64)
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, atol=1e-8, rtol=0.0):
            raise ValueError(
                "boundary_matrix debe contener coeficientes enteros exactos."
            )
        return [[int(x) for x in row] for row in rounded.tolist()]

    def _adjacency_diagnostics(
        self,
        adjacency: np.ndarray,
    ) -> Dict[str, Any]:
        """Invariantes combinatorios de A: simetría residual, signo, regularidad."""
        skew = _frobenius_kbn(adjacency - adjacency.T)
        diag_norm = float(np.linalg.norm(np.diag(adjacency)))
        min_entry = float(np.min(adjacency))
        degrees = np.sum(adjacency, axis=1)
        return {
            "adjacency_skew_frobenius": skew,
            "adjacency_diag_norm": diag_norm,
            "adjacency_min_entry": min_entry,
            "degree_min": float(np.min(degrees)),
            "degree_max": float(np.max(degrees)),
            "is_nonnegative": bool(min_entry >= -self._tol),
            "is_loopless": bool(diag_norm <= self._tol),
        }

    def _session_hash(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
    ) -> str:
        """Sello SHA-256 de sesión sobre invariantes observados (IEEE-hex)."""
        h = hashlib.sha256()
        h.update(b"PHASE1/REACTION_AGENT_OBSERVE")
        h.update(_canonical_bytes(state_vector))
        h.update(_canonical_bytes(adjacency_matrix))
        h.update(_canonical_bytes(boundary_matrix))
        h.update(float(diffusion_rate).hex().encode("utf-8"))
        h.update(np.int64(state_vector.size).tobytes())
        return h.hexdigest()

    def phase1_close_and_open_phase2(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
    ) -> Phase1ReactionHandoff:
        """
        Fase 1.1 — Observación, validación, sello y kernel.

        Definición formal de frontera:
            Φ₁→₂ : Datos crudos ↦ Observación validada.

        Este es el último método de la Fase 1.  Construye el kernel de
        observación (morfismo terminal del sustrato exacto) y lo admite
        de inmediato en `phase2_from_phase1`, que es el primer método
        de la Fase 2.
        """
        psi = self._validate_state_vector(state_vector)
        n_nodes = int(psi.size)
        A = self._validate_adjacency_matrix(adjacency_matrix, n_nodes)
        B = self._validate_boundary_matrix(boundary_matrix)

        if not math.isfinite(diffusion_rate):
            raise ValueError(f"diffusion_rate no es finita: {diffusion_rate}")

        session_sha256 = self._session_hash(
            state_vector=psi,
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=float(diffusion_rate),
        )

        kernel = self.construct_observation_kernel(
            state_vector=psi,
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=float(diffusion_rate),
            session_sha256=session_sha256,
        )

        adj_diag = self._adjacency_diagnostics(A)
        diagnostics: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "boundary_shape": tuple(int(x) for x in B.shape),
            "diffusion_rate": float(diffusion_rate),
            "session_sha256_prefix": session_sha256[:16],
            "state_l2_norm": float(la.norm(psi)),
            "laplacian_trace": float(np.trace(kernel.laplacian)),
            "agent_version": __version__,
        }
        diagnostics.update(adj_diag)

        handoff = Phase1ReactionHandoff(
            state_vector=psi,
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=float(diffusion_rate),
            session_sha256=session_sha256,
            diagnostics=diagnostics,
            next_entrypoint="phase2_from_phase1",
            observation_kernel=kernel,
        )

        # Acoplamiento formal Φ₁→₂: la frontera debe ser admitida por Fase 2.
        _ = self.phase2_from_phase1(handoff)

        logger.debug(
            "Fase Observe [REACTION_AGENT]: sesión sellada. SHA prefix=%s",
            session_sha256[:16],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ORIENT: SMITH, HÜCKEL, FIEDLER, CFL, HODGE–TELLEGEN
    # Continúa IntegerExactArithmetic.construct_observation_kernel
    # / phase1_close_and_open_phase2
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(
        self,
        handoff: Phase1ReactionHandoff,
    ) -> Phase1ReactionHandoff:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Primer método de la Fase 2: reconoce el tipo Phase1ReactionHandoff
        y verifica que el punto de entrada sea exactamente Φ₁→₂.
        """
        if not isinstance(handoff, Phase1ReactionHandoff):
            raise TypeError(
                "phase2_from_phase1 exige Phase1ReactionHandoff; "
                f"recibido {type(handoff)!r}"
            )
        if handoff.next_entrypoint != "phase2_from_phase1":
            raise ValueError(
                "Phase1ReactionHandoff inválido: el punto de entrada esperado es "
                "'phase2_from_phase1'."
            )
        if not handoff.session_sha256:
            raise ValueError("Phase1ReactionHandoff sin sello de sesión.")
        return handoff

    def _kernel_or_rebuild(
        self,
        handoff: Phase1ReactionHandoff,
    ) -> ObservationKernel:
        """Recupera el kernel de observación o lo reconstruye (compatibilidad)."""
        if handoff.observation_kernel is not None:
            return handoff.observation_kernel
        return self.construct_observation_kernel(
            state_vector=handoff.state_vector,
            adjacency_matrix=handoff.adjacency_matrix,
            boundary_matrix=handoff.boundary_matrix,
            diffusion_rate=handoff.diffusion_rate,
            session_sha256=handoff.session_sha256,
        )

    def evaluate_smith_normal_form_torsion(
        self,
        boundary_matrix: np.ndarray,
    ) -> bool:
        """
        Evalúa torsión homológica sobre ℤ en ∂₁.

        True  ⇒ existe torsión (anomalía).
        False ⇒ no se detectó torsión.
        """
        int_matrix = self._to_integer_matrix(boundary_matrix)
        m = len(int_matrix)
        n = len(int_matrix[0]) if m > 0 else 0
        if m == 0 or n == 0:
            return False

        try:
            has_torsion, _, diag = self.torsion_from_snf(
                int_matrix,
                cross_check_minors=(min(m, n) <= 6),
            )
            if diag.get("snf_minors_disagreement") and self._strict_homology:
                return True
            return bool(has_torsion)
        except Exception as exc:
            logger.warning("Fallo en SNF exacta: %s", exc)
            if self._strict_homology:
                return True
            try:
                from sympy import Matrix, ZZ
                from sympy.matrices.normalforms import smith_normal_form

                S = smith_normal_form(Matrix(int_matrix), domain=ZZ)
                diag_vals = [
                    abs(int(S[i, i]))
                    for i in range(min(S.rows, S.cols))
                ]
                return any(d > 1 for d in diag_vals)
            except Exception as exc2:
                logger.warning("Fallback SymPy SNF falló: %s", exc2)
                return bool(self._strict_homology)

    def _evaluate_smith_full(
        self,
        boundary_matrix: np.ndarray,
    ) -> Tuple[bool, Tuple[int, ...], Dict[str, Any]]:
        """SNF completa con factores invariantes y diagnóstico."""
        try:
            int_matrix = self._to_integer_matrix(boundary_matrix)
        except Exception as exc:
            if self._strict_homology:
                return True, (), {"snf_status": "non_integral", "smith_error": str(exc)}
            return False, (), {"snf_status": "non_integral", "smith_error": str(exc)}

        m = len(int_matrix)
        n = len(int_matrix[0]) if m else 0
        if m == 0 or n == 0:
            return False, (), {"snf_status": "empty"}

        try:
            return self.torsion_from_snf(
                int_matrix,
                cross_check_minors=(min(m, n) <= 6),
            )
        except Exception as exc:
            logger.warning("SNF completa falló: %s", exc)
            has = bool(self._strict_homology)
            return has, (), {"snf_status": "error", "smith_error": str(exc)}

    def _evaluate_huckel_resonance_internal(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        alpha: float = _HUCKEL_ALPHA,
        beta: float = _HUCKEL_BETA,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Auditoría de resonancia de Hückel (Rayleigh–Ritz + residuo espectral).

            H = α I + β A ,  H = Hᵀ
            E(ψ) = ψ̂ᵀ H ψ̂  ≥ λ_min     (Courant–Fischer)
            r = ‖H ψ̂ − E ψ̂‖₂

        Estabilidad del fundamental: E ≈ λ_min  y  r ≈ 0.
        """
        psi = self._validate_state_vector(state_vector)
        n = psi.size
        A = self._validate_adjacency_matrix(adjacency_matrix, n)

        H = alpha * np.eye(n, dtype=np.float64) + beta * A
        H = 0.5 * (H + H.T)

        norm = float(la.norm(psi))
        if norm < self._reg:
            return 0.0, False, {
                "huckel_status": "zero_state",
                "lambda_min": math.nan,
                "rayleigh_residual": math.nan,
            }

        psi_hat = psi / norm
        Hpsi = H @ psi_hat
        energy = _kbn_dot(psi_hat, Hpsi)

        if not math.isfinite(energy):
            return float("nan"), False, {
                "huckel_status": "nonfinite_energy",
                "lambda_min": math.nan,
                "rayleigh_residual": math.nan,
            }

        residual_vec = Hpsi - energy * psi_hat
        residual = float(la.norm(residual_vec))

        eigvals = la.eigvalsh(H)
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])
        gap_to_ground = float(energy - lambda_min)
        first_excited = float(eigvals[1]) if eigvals.size >= 2 else lambda_min
        spectral_gap = float(first_excited - lambda_min)

        rel_tol = max(self._tol, 1e-10) * (1.0 + abs(lambda_min))
        is_near_ground = gap_to_ground <= rel_tol
        is_eigen = residual <= max(self._tol, 1e-8) * (1.0 + abs(energy))
        is_stable = bool(is_near_ground and is_eigen)

        analytical: Optional[Tuple[float, ...]] = None
        if n == _RING_SIZE:
            analytical = tuple(sorted(
                alpha + 2.0 * beta * math.cos(2.0 * math.pi * k / n)
                for k in range(n)
            ))

        diagnostics: Dict[str, Any] = {
            "huckel_status": "ok",
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "energy_gap_to_ground": gap_to_ground,
            "huckel_spectral_gap": spectral_gap,
            "rayleigh_residual": residual,
            "alpha": float(alpha),
            "beta": float(beta),
            "is_near_ground": is_near_ground,
            "is_approximate_eigenstate": is_eigen,
        }
        if analytical is not None:
            weyl = max(abs(float(a) - float(b)) for a, b in zip(eigvals, analytical))
            diagnostics["huckel_analytical_levels"] = analytical
            diagnostics["huckel_weyl_residual"] = float(weyl)

        return float(energy), is_stable, diagnostics

    def evaluate_huckel_resonance(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        alpha: float = _HUCKEL_ALPHA,
        beta: float = _HUCKEL_BETA,
    ) -> Tuple[float, bool]:
        """API legada — Rayleigh de Hückel y estabilidad del fundamental."""
        energy, is_stable, _ = self._evaluate_huckel_resonance_internal(
            state_vector=state_vector,
            adjacency_matrix=adjacency_matrix,
            alpha=alpha,
            beta=beta,
        )
        return energy, is_stable

    def _audit_connectivity(
        self,
        adjacency_matrix: np.ndarray,
        laplacian: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Conectividad espectral de Fiedler y núcleo laplaciano.

            L = D − A ⪰ 0
            λ₁ ≈ 0  (kernel de constantes ssi 1-conexo numérico)
            λ₂ ≥ τ_Fiedler
            Cheeger:  λ₂/2 ≤ h ≤ √(2 λ₂)
        """
        A = np.asarray(adjacency_matrix, dtype=np.float64)
        n = A.shape[0]
        if n <= 1:
            return math.inf, True, {
                "fiedler_status": "single_node",
                "lambda_max_laplacian": 0.0,
            }

        if laplacian is None:
            degrees = np.sum(A, axis=1)
            L = np.diag(degrees) - A
            L = 0.5 * (L + L.T)
        else:
            L = np.asarray(laplacian, dtype=np.float64)

        eigvals = la.eigvalsh(L)
        lam1 = float(eigvals[0]) if eigvals.size else math.nan
        fiedler = float(eigvals[1]) if eigvals.size >= 2 else math.inf
        lam_max = float(eigvals[-1]) if eigvals.size else math.nan

        kernel_ok = abs(lam1) <= max(self._tol, 1e-10)
        is_connected = bool(fiedler >= self._fiedler_min - self._tol)

        cheeger_upper = math.sqrt(max(2.0 * max(fiedler, 0.0), 0.0))
        cheeger_lower = 0.5 * max(fiedler, 0.0)

        diagnostics: Dict[str, Any] = {
            "fiedler_value": fiedler,
            "fiedler_threshold": self._fiedler_min,
            "lambda_1": lam1,
            "lambda_max_laplacian": lam_max,
            "laplacian_kernel_ok": kernel_ok,
            "cheeger_lower_bound": cheeger_lower,
            "cheeger_upper_bound": cheeger_upper,
            "fiedler_status": "connected" if is_connected else "disconnected",
        }
        return fiedler, is_connected, diagnostics

    def _audit_cfl(
        self,
        diffusion_rate: float,
        lambda_max: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Estabilidad de von Neumann / CFL del Euler explícito.

            0 ≤ α < α_crit_trabajo
            α_sharp = 2 / λ_max     (cota espectral estricta)
            α_crit  = 1 / (2 λ_max) (margen del reactor, por defecto 0.125)
        """
        if not math.isfinite(diffusion_rate):
            return False, {"cfl_status": "nonfinite_rate"}
        if diffusion_rate < 0.0:
            return False, {"cfl_status": "negative_rate"}

        lam_max = float(lambda_max) if lambda_max is not None else _LAMBDA_MAX_C6
        if not math.isfinite(lam_max) or lam_max <= 0.0:
            lam_max = _LAMBDA_MAX_C6

        alpha_sharp = 2.0 / lam_max
        alpha_work = min(self._cfl_crit, 1.0 / (2.0 * lam_max))
        is_stable = diffusion_rate < alpha_work - self._tol
        is_sharp_stable = diffusion_rate <= alpha_sharp + self._tol

        diagnostics: Dict[str, Any] = {
            "diffusion_rate": float(diffusion_rate),
            "cfl_critical": float(self._cfl_crit),
            "cfl_alpha_sharp": float(alpha_sharp),
            "cfl_alpha_work": float(alpha_work),
            "cfl_margin": float(alpha_work - diffusion_rate),
            "cfl_sharp_stable": bool(is_sharp_stable),
            "lambda_max_used": float(lam_max),
            "cfl_status": "stable" if is_stable else "unstable",
        }
        return bool(is_stable), diagnostics

    def _audit_hodge_tellegen(
        self,
        kernel: ObservationKernel,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compatibilidad Hodge–Tellegen:

            L_A = D − A
            L_∂ = ∂ ∂ᵀ
            ρ = ‖L_A − L_∂‖_F

        Si las dimensiones no coinciden se reporta incompatibilidad de soporte.
        """
        L_a = kernel.laplacian
        B = kernel.boundary_matrix
        n = kernel.n_nodes

        if B.shape[0] != n:
            return math.inf, False, {
                "hodge_status": "shape_mismatch",
                "hodge_residual": math.inf,
                "boundary_shape": tuple(int(x) for x in B.shape),
                "laplacian_shape": tuple(int(x) for x in L_a.shape),
            }

        L_d = B @ B.T
        residual = _frobenius_kbn(L_a - L_d)
        compatible = bool(residual <= self._hodge_tol)

        ones = np.ones(n, dtype=np.float64)
        inj = B @ (B.T @ ones)
        conservation = abs(_kbn_sum(float(x) for x in inj))

        return residual, compatible, {
            "hodge_status": "compatible" if compatible else "incompatible",
            "hodge_residual": float(residual),
            "hodge_tolerance": float(self._hodge_tol),
            "tellegen_ones_residual": float(conservation),
        }

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1ReactionHandoff,
    ) -> Phase2ReactionHandoff:
        """
        Fase 2.2 — Cierre formal de Fase 2 y apertura de Fase 3.

        Definición formal de frontera:
            Φ₂→₃ : Observación validada ↦ Invariantes auditados.

        Este es el último método de la Fase 2.  Consume el kernel de
        observación de la Fase 1, calcula Smith / Hückel / Fiedler / CFL /
        Hodge y admite de inmediato el handoff en `phase3_from_phase2`,
        primer método de la Fase 3.
        """
        _ = self.phase2_from_phase1(phase1_handoff)
        kernel = self._kernel_or_rebuild(phase1_handoff)

        has_torsion, factors, smith_diag = self._evaluate_smith_full(
            phase1_handoff.boundary_matrix
        )

        energy, is_aromatic, huckel_diag = self._evaluate_huckel_resonance_internal(
            state_vector=phase1_handoff.state_vector,
            adjacency_matrix=phase1_handoff.adjacency_matrix,
        )

        fiedler, is_connected, fiedler_diag = self._audit_connectivity(
            phase1_handoff.adjacency_matrix,
            laplacian=kernel.laplacian,
        )

        lam_max = float(fiedler_diag.get("lambda_max_laplacian", _LAMBDA_MAX_C6))
        if not math.isfinite(lam_max):
            lam_max = _LAMBDA_MAX_C6

        is_cfl, cfl_diag = self._audit_cfl(
            phase1_handoff.diffusion_rate,
            lambda_max=lam_max,
        )

        hodge_residual, hodge_ok, hodge_diag = self._audit_hodge_tellegen(kernel)

        betti_0 = 1 if is_connected else 0
        rank_snf = int(smith_diag.get("snf_rank", 0) or 0)
        n_edges = int(phase1_handoff.boundary_matrix.shape[1])
        n_nodes = int(kernel.n_nodes)
        betti_1_est = max(0, n_edges - rank_snf) if rank_snf else max(0, n_edges - n_nodes + betti_0)

        diagnostics: Dict[str, Any] = dict(phase1_handoff.diagnostics)
        diagnostics.update(smith_diag)
        diagnostics.update(huckel_diag)
        diagnostics.update(fiedler_diag)
        diagnostics.update(cfl_diag)
        diagnostics.update(hodge_diag)
        diagnostics["has_torsion_anomaly"] = bool(has_torsion)
        diagnostics["hodge_required"] = bool(self._require_hodge)
        diagnostics["hodge_ok"] = bool(hodge_ok)
        diagnostics["betti_0_est"] = int(betti_0)
        diagnostics["betti_1_est"] = int(betti_1_est)

        handoff = Phase2ReactionHandoff(
            state_vector=phase1_handoff.state_vector,
            adjacency_matrix=phase1_handoff.adjacency_matrix,
            boundary_matrix=phase1_handoff.boundary_matrix,
            diffusion_rate=phase1_handoff.diffusion_rate,
            session_sha256=phase1_handoff.session_sha256,
            aromatic_energy=float(energy),
            is_aromatic_stable=bool(is_aromatic),
            has_torsion_anomaly=bool(has_torsion),
            is_cfl_stable=bool(is_cfl),
            is_connected=bool(is_connected),
            fiedler_value=float(fiedler),
            diagnostics=diagnostics,
            next_entrypoint="phase3_from_phase2",
            invariant_factors=tuple(int(s) for s in factors),
            hodge_residual=float(hodge_residual) if math.isfinite(hodge_residual) else float("inf"),
            lambda_max_laplacian=float(lam_max),
            rayleigh_residual=float(huckel_diag.get("rayleigh_residual", math.nan)),
        )

        # Acoplamiento formal Φ₂→₃: la frontera debe ser admitida por Fase 3.
        _ = self.phase3_from_phase2(handoff)

        logger.debug(
            "Fase Orient [REACTION_AGENT]: energy=%.6e, torsion=%s, cfl=%s, hodge=%.3e",
            energy,
            has_torsion,
            is_cfl,
            hodge_residual,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: HEYTING, GRACIA, OVERRIDE, CROWBAR, SELLO
    # Continúa phase2_close_and_open_phase3 / Phase2ReactionHandoff
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(
        self,
        handoff: Phase2ReactionHandoff,
    ) -> Phase2ReactionHandoff:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Primer método de la Fase 3: reconoce Phase2ReactionHandoff y
        verifica el punto de entrada Φ₂→₃.
        """
        if not isinstance(handoff, Phase2ReactionHandoff):
            raise TypeError(
                "phase3_from_phase2 exige Phase2ReactionHandoff; "
                f"recibido {type(handoff)!r}"
            )
        if handoff.next_entrypoint != "phase3_from_phase2":
            raise ValueError(
                "Phase2ReactionHandoff inválido: el punto de entrada esperado es "
                "'phase3_from_phase2'."
            )
        if not handoff.session_sha256:
            raise ValueError("Phase2ReactionHandoff sin sello de sesión.")
        return handoff

    def _verify_hmac_override(self, token: Optional[str]) -> bool:
        """
        Fase 3.1 — Override humano.

        Se aceptan:
          - Tokens canónicos autorizados (conjunto cerrado).
          - Tokens HMAC-SHA256 `payload:signature` si hay `hmac_key`,
            comparados en tiempo constante.
        """
        if token is None:
            return False
        token_str = str(token).strip()
        if not token_str:
            return False
        if token_str in self._authorized_tokens:
            return True
        if self._hmac_key is not None and ":" in token_str:
            payload, signature = token_str.rsplit(":", 1)
            expected = hmac.new(
                self._hmac_key,
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            try:
                return hmac.compare_digest(signature, expected)
            except (TypeError, ValueError):
                return False
        return False

    def _hard_reasons(self, handoff: Phase2ReactionHandoff) -> List[str]:
        """Generadores del elemento ⊥ del Heyting: anomalías no perdonables."""
        reasons: List[str] = []
        if handoff.has_torsion_anomaly:
            reasons.append("smith_torsion")
        if not handoff.is_cfl_stable:
            reasons.append("cfl_violation")
        if not handoff.is_connected:
            reasons.append("spectral_disconnection")
        if not math.isfinite(handoff.aromatic_energy):
            reasons.append("nonfinite_energy")
        if self._require_hodge and handoff.hodge_residual > self._hodge_tol:
            reasons.append("hodge_incompatibility")
        return reasons

    def phase3_decide_heyting(
        self,
        handoff: Phase2ReactionHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> Dict[str, Any]:
        """
        Fase 3.2 — Clasificador de Heyting trivalente Ω₃.

        Reglas (morfismos de Ω₃):
          ⊥ inmediato:
              torsión / CFL / desconexión / energía no finita / Hodge (si se exige)
          U  (veto suave):
              pérdida de resonancia aromática del fundamental
          U --gracia expirada sin override--> ⊥
          U --override HMAC válido--> U⁺  (DEGRADED, no restaura ⊤)
          ⊤ ssi ninguna anomalía
        """
        curr_time = float(current_time if current_time is not None else time.time())
        hard_reasons = self._hard_reasons(handoff)

        override_valid = self._verify_hmac_override(override_token)
        override_sha = None
        if override_valid and override_token is not None:
            override_sha = hashlib.sha256(
                str(override_token).encode("utf-8")
            ).hexdigest()

        element = HeytingElement.COHERENT
        is_hard_veto = False
        is_soft_veto = False
        time_remaining = 0.0

        if hard_reasons:
            element = HeytingElement.VETOED
            is_hard_veto = True
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None
            logger.critical(
                "¡VETO DURO INSTANTÁNEO POR COLAPSO HOMOLÓGICO / CFL / "
                "DESCONEXIÓN ESPECTRAL! Razones: %s",
                ", ".join(hard_reasons),
            )

        elif not handoff.is_aromatic_stable:
            is_soft_veto = True
            if simulate_grace_expired:
                time_remaining = 0.0
            elif not self._is_soft_veto_active:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                time_remaining = self._grace_max
                logger.warning(
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! Pérdida de resonancia "
                    "aromática. Ventana de gracia de %.0f segundos iniciada.",
                    self._grace_max,
                )
            else:
                elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                time_remaining = max(0.0, self._grace_max - elapsed)

            if override_valid:
                element = HeytingElement.DEGRADED
                is_soft_veto = False
                is_hard_veto = False
                time_remaining = 0.0
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.info(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] INYECTADO! "
                    "Aniquilando anomalía en Fock. Sello: %s",
                    override_sha[:16] if override_sha else "UNKNOWN",
                )
            elif time_remaining <= self._tol or simulate_grace_expired:
                element = HeytingElement.VETOED
                is_hard_veto = True
                is_soft_veto = False
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "¡PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO! "
                    "Colapsando Heyting a VETOED terminal."
                )
            else:
                element = HeytingElement.DEGRADED
        else:
            element = HeytingElement.COHERENT
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None

        return {
            "heyting_verdict": element.name,
            "heyting_element": element.value,
            "is_hard_veto": bool(is_hard_veto),
            "is_soft_veto": bool(is_soft_veto and element is HeytingElement.DEGRADED),
            "hard_reasons": hard_reasons,
            "override_valid": bool(override_valid),
            "override_sha256": override_sha,
            "time_grace_remaining": float(time_remaining),
            "current_time": curr_time,
            "simulate_grace_expired": bool(simulate_grace_expired),
        }

    def phase3_actuate_crowbar_interlock(
        self,
        heyting_verdict: str,
    ) -> Tuple[bool, float]:
        """
        Fase 3.3 — Actuación simulada del interlock Crowbar.

        Si el veredicto es ⊥ (VETOED) se emula la conmutación IRAM
        con latencia gaussiana recortada a [385, 415] ns.  No se
        emite I/O de hardware real: sólo telemetría.
        """
        if heyting_verdict != HeytingElement.VETOED.name:
            return False, 0.0

        jitter = float(self._rng.normal(loc=398.5, scale=self._jitter_sigma))
        latency_ns = float(np.clip(jitter, 385.0, 415.0))

        logger.critical(
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO DE LA CÁMARA!\n"
            "  - Ejecutando subrutina local isVerdictCoherent() en C++...\n"
            "  - Despachando ISR en IRAM en menos de 400 ns...\n"
            "  - Conmutando pin de hardware GPIO14 a HIGH en %.2f ns...\n"
            "  - ¡Tiristor rápido de potencia BT151 (Crowbar) gatillado!\n"
            "  - Mezcladoras y bombas hidráulicas paralizadas en el milisegundo cero.",
            latency_ns,
        )
        return True, latency_ns

    def _canonical_seal_payload(
        self,
        handoff: Phase2ReactionHandoff,
        decision: Dict[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> bytes:
        """Codificación canónica (hex IEEE + enteros) para el sello SHA-256."""
        parts: List[bytes] = [
            b"PHASE3/REACTION_AGENT_GOVERNANCE",
            __version__.encode("utf-8"),
            _canonical_bytes(handoff.state_vector),
            _canonical_bytes(handoff.adjacency_matrix),
            _canonical_bytes(handoff.boundary_matrix),
            str(decision["heyting_verdict"]).encode("utf-8"),
            float(handoff.aromatic_energy).hex().encode("utf-8"),
            b"1" if handoff.is_aromatic_stable else b"0",
            b"1" if handoff.has_torsion_anomaly else b"0",
            b"1" if handoff.is_cfl_stable else b"0",
            b"1" if handoff.is_connected else b"0",
            float(handoff.fiedler_value).hex().encode("utf-8"),
            float(handoff.hodge_residual).hex().encode("utf-8"),
            b"1" if decision["is_hard_veto"] else b"0",
            b"1" if decision["is_soft_veto"] else b"0",
            float(decision["time_grace_remaining"]).hex().encode("utf-8"),
            b"1" if interlock_fired else b"0",
            float(switching_latency_ns).hex().encode("utf-8"),
            handoff.session_sha256.encode("utf-8"),
            str(decision.get("override_sha256") or "").encode("utf-8"),
            repr(tuple(handoff.invariant_factors)).encode("utf-8"),
        ]
        return b"|".join(parts)

    def _phase3_cryptographic_seal(
        self,
        handoff: Phase2ReactionHandoff,
        decision: Dict[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> str:
        """Fase 3.4 — Sello SHA-256 de lazo cerrado (payload canónico)."""
        payload = self._canonical_seal_payload(
            handoff=handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=switching_latency_ns,
        )
        return hashlib.sha256(payload).hexdigest()

    def phase3_issue_verdict(
        self,
        handoff: Phase2ReactionHandoff,
        decision: Dict[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> ReactionChamberVerdict:
        """Fase 3.5 — Emisión del veredicto certificado e inmutable."""
        seal = self._phase3_cryptographic_seal(
            handoff=handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=switching_latency_ns,
        )
        phase_chain = (
            "PHASE1/OBSERVE",
            "PHASE2/ORIENT",
            "PHASE3/DECIDE/ACT",
        )
        diagnostics = dict(handoff.diagnostics)
        diagnostics["decision"] = decision
        diagnostics["hardware"] = {
            "interlock_fired": bool(interlock_fired),
            "switching_latency_ns": float(switching_latency_ns),
            "crowbar_spec_ns": float(_CROWBAR_IRAM_LATENCY_NS),
        }
        diagnostics["phase_chain"] = phase_chain
        diagnostics["agent_version"] = __version__

        return ReactionChamberVerdict(
            heyting_verdict=str(decision["heyting_verdict"]),
            aromatic_energy=float(handoff.aromatic_energy),
            is_aromatic_stable=bool(handoff.is_aromatic_stable),
            has_torsion_anomaly=bool(handoff.has_torsion_anomaly),
            is_cfl_stable=bool(handoff.is_cfl_stable),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            is_hard_veto_active=bool(decision["is_hard_veto"]),
            switching_latency_ns=float(switching_latency_ns),
            time_grace_remaining=float(decision["time_grace_remaining"]),
            cryptographic_seal=seal,
            session_sha256=handoff.session_sha256,
            phase_chain=phase_chain,
            diagnostics=diagnostics,
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2ReactionHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> ReactionChamberVerdict:
        """
        Fase 3.6 — Orquestación completa de la Fase 3 (morfismo terminal).

        Ejecuta: admisión Φ₂→₃ → decisión Heyting → Crowbar → certificado.
        """
        _ = self.phase3_from_phase2(phase2_handoff)

        decision = self.phase3_decide_heyting(
            handoff=phase2_handoff,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
        )
        interlock_fired, latency = self.phase3_actuate_crowbar_interlock(
            decision["heyting_verdict"]
        )
        verdict = self.phase3_issue_verdict(
            handoff=phase2_handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=latency,
        )

        if verdict.heyting_verdict == HeytingElement.VETOED.name:
            logger.critical(
                "Soberano de la Cámara emitió VETO DURO. Sello: %s",
                verdict.cryptographic_seal[:16],
            )
        else:
            logger.info(
                "Soberano de la Cámara regulado síncronamente. Veredicto: %s. "
                "Sello: %s",
                verdict.heyting_verdict,
                verdict.cryptographic_seal[:16],
            )
        return verdict

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE OODA
    # ═════════════════════════════════════════════════════════════════════════

    def audit_lazo_cerrado(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> ReactionChamberVerdict:
        """
        API principal — Orquesta el ciclo OODA completo.

        Equivalencia de fases:
          OBSERVE    → phase1_close_and_open_phase2
          ORIENT     → phase2_close_and_open_phase3
          DECIDE/ACT → phase3_close_loop
        """
        phase1_handoff = self.phase1_close_and_open_phase2(
            state_vector=state_vector,
            adjacency_matrix=adjacency_matrix,
            boundary_matrix=boundary_matrix,
            diffusion_rate=diffusion_rate,
        )
        phase2_handoff = self.phase2_close_and_open_phase3(
            phase1_handoff=phase1_handoff,
        )
        return self.phase3_close_loop(
            phase2_handoff=phase2_handoff,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
        )