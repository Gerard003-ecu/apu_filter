# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Reaction Chamber Agent — Evolución Doctoral en 3 Fases Anidadas     ║
║ Ruta   : app/agents/wisdom/reaction_chamber_agent.py                         ║
║ Versión: 3.1.0-Hodge-Smith-Heyting-CFL-IRAM-Governance                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS
========
Soberano de Calibre de la Cámara de Reacción Catalítica Cuántica.

Audita al reactor `reaction_chamber.py` (v4.1.0) mediante un topos de
morfismos anidados Φ₁ → Φ₂ → Φ₃.  Cada fase es un objeto cuya flecha
terminal es el objeto inicial de la siguiente:

    FASE 1  IntegerExactArithmetic.construct_observation_kernel
         │  + validación Banach / Dot2 / Bareiss / SNF / sello SHA-256
         └─► phase1_close_and_open_phase2  ──►  Phase1ReactionHandoff
                │
    FASE 2  Auditoría homológico-espectral (continúa el kernel)
         │  Smith / Hückel(Kato–Temple) / Fiedler(Cheeger exacto) /
         │  CFL(propagador) / Hodge–Tellegen(ponderado) / libro de Betti
         └─► phase2_close_and_open_phase3  ──►  Phase2ReactionHandoff
                │
    FASE 3  Gobierno de Heyting (continúa el estado espectral)
         │  Átomos Ω₃, meet, gracia, override HMAC(exp,nonce), Crowbar, sello
         └─► phase3_close_loop  ──►  ReactionChamberVerdict

CONTINUIDAD FORMAL
==================
    Φ₁→₂ : Phase1ReactionHandoff  →  phase2_from_phase1
    Φ₂→₃ : Phase2ReactionHandoff  →  phase3_from_phase2

El último método de la Fase k invoca de inmediato el morfismo de admisión
de la Fase k+1: la frontera es un tipo, no un comentario.

FUNDAMENTOS
===========
Homología entera (Smith):
    ∂₁ : C₁ → C₀  sobre ℤ,   SNF(∂₁) = diag(s₁ | … | s_r, 0, …)
    Torsión ⇔ ∃k: s_k > 1.   Para una matriz de incidencia signada genuina
    (totalmente unimodular) todos los s_k = 1; por tanto torsión en ∂₁
    certifica corrupción del complejo, no topología exótica.
    β₀ = dim ker L,   β₁ = |E| − rank ∂₁,   χ = |V| − |E| = β₀ − β₁.

Hückel / Rayleigh–Ritz:
    H = αI + βA,  E(ψ) = ⟨ψ̂, Hψ̂⟩ ≥ λ_min          (Courant–Fischer)
    r = ‖(H − E)ψ̂‖₂ ,  dist(E, spec H) ≤ r          (Weyl / Bauer–Fike)
    λ_min ≥ E − r²/(λ₂ − E)  si  r < λ₂ − E          (Kato–Temple)
    Cₙ:  spec(H) = { α + 2β cos(2πk/n) }.

Fiedler / Cheeger (Laplaciano combinatorio L = D − A):
    λ₂/2 ≤ h(G) ≤ √(2 d_max λ₂),  h(G) = min_{0<|S|≤n/2} w(S,Sᶜ)/|S|.

CFL / von Neumann:
    P = I − αL,  ρ(P) = max|1 − αλ| ≤ 1 ⇔ 0 ≤ α ≤ 2/λ_max
    P ≥ 0 entrada a entrada ⇔ α d_max ≤ 1  (semigrupo estocástico)
    Cota de trabajo del reactor:  α_crit = 1/(2λ_max) = 1/8 para C₆.

Hodge–Tellegen:
    L_A = D − A,  L_∂ = ∂ W ∂ᵀ,  ‖L_A − L_∂‖_F ≤ τ,  ∂ᵀ1 = 0 (Kirchhoff).

Heyting trivalente Ω₃ = {⊥ < U < ⊤}:
    ⊥ = VETOED, U = DEGRADED, ⊤ = COHERENT
    a∧b = min, a∨b = max, a→b = ⊤ si a ≤ b si no b, ¬a = a→⊥
    ¬¬U = ⊤ ≠ U  (no booleana: falla el tercero excluido).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from itertools import combinations
from math import comb
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import scipy.linalg as la


__all__ = [
    "ReactionChamberAgent",
    "ReactionChamberVerdict",
    "Phase1ReactionHandoff",
    "Phase2ReactionHandoff",
    "HeytingElement",
    "IntegerExactArithmetic",
    "ObservationKernel",
]

__version__ = "3.1.0"

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

# Presupuestos combinatorios (rigor acotado, no rigor infinito).
_CHEEGER_EXACT_MAX_NODES: Final[int] = 12
_MINOR_BUDGET: Final[int] = 20_000

# Aritmética libre de error.
_VELTKAMP_FACTOR: Final[float] = float(2 ** 27 + 1)
_FMA = getattr(math, "fma", None)  # Python ≥ 3.13


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 — SUSTRATO ARITMÉTICO EXACTO Y OBSERVACIÓN CALIBRADA
# Banach ℓ² / Dot2 / Bareiss / SNF con testigo unimodular / kernel
# ═════════════════════════════════════════════════════════════════════════════
#
# Objetos: (ℝⁿ, ‖·‖₂), anillo ℤ con eliminación fraction-free, sello SHA-256.
# El morfismo terminal es
#     IntegerExactArithmetic.construct_observation_kernel
# que phase1_close_and_open_phase2 instala como Phase1ReactionHandoff,
# objeto inicial de la Fase 2.


# ── 1.a  Canonicalización y serialización ───────────────────────────────────

def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """Proyecta −0.0 → +0.0 (clase de equivalencia IEEE) para firmas deterministas."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """
    Serialización canónica inyectiva: cabecera `dtype|ndim|shape|` + datos
    contiguos C-order.  La cabecera impide colisiones entre arrays con los
    mismos bytes pero distinta forma (p. ej. (2,3) vs (3,2)).
    """
    a = np.ascontiguousarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = _canonicalize_signed_zero(a)
    shape = "x".join(str(int(d)) for d in a.shape) or "0"
    header = f"{a.dtype.str}|{a.ndim}|{shape}|".encode("ascii")
    return header + a.tobytes()


def _is_finite_array(arr: np.ndarray) -> bool:
    """Pertenencia a ℝⁿ (rechaza NaN e infinitos)."""
    return bool(np.all(np.isfinite(arr)))


# ── 1.b  Transformaciones libres de error ───────────────────────────────────

def _two_sum(a: float, b: float) -> Tuple[float, float]:
    """Knuth–Møller: a + b = s + e exactamente en punto flotante."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def _two_prod(a: float, b: float) -> Tuple[float, float]:
    """
    Dekker–Veltkamp / FMA: a · b = p + e exactamente.

    Con FMA nativo e = fma(a, b, −p).  Sin FMA se usa la partición de
    Veltkamp (factor 2²⁷ + 1) que es exacta salvo desbordamiento.
    """
    p = a * b
    if _FMA is not None:
        return p, _FMA(a, b, -p)
    c = _VELTKAMP_FACTOR * a
    a_hi = c - (c - a)
    a_lo = a - a_hi
    c = _VELTKAMP_FACTOR * b
    b_hi = c - (c - b)
    b_lo = b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def _kbn_sum(values: Iterable[float]) -> float:
    """Sumación compensada Kahan–Babuška–Neumaier (error O(ε) + O(nε²))."""
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


def _dot2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Producto interno Dot2 (Ogita–Rump–Oishi, 2005).

    Resultado con precisión equivalente a doble-doble: el error relativo
    es O(ε) + O(n ε² cond).  Supera a KBN porque también captura el
    error del producto vía `_two_prod`.
    """
    if a.shape != b.shape:
        raise ValueError("Producto interno indefinido para tensores de distinta forma")
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    p, s = _two_prod(float(x[0]), float(y[0]))
    for i in range(1, x.size):
        h, r = _two_prod(float(x[i]), float(y[i]))
        p, q = _two_sum(p, h)
        s += q + r
    return float(p + s)


def _norm2(vec: np.ndarray) -> float:
    """‖v‖₂ = √⟨v, v⟩ con Dot2."""
    return math.sqrt(max(_dot2(vec, vec), 0.0))


def _frobenius(matrix: np.ndarray) -> float:
    """‖A‖_F = √Σ Aᵢⱼ² con Dot2 sobre el aplanado."""
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    return math.sqrt(max(_dot2(flat, flat), 0.0))


# ── 1.c  Aritmética exacta sobre ℤ ──────────────────────────────────────────

def _bareiss_det(matrix: List[List[int]]) -> int:
    """
    Determinante exacto sobre ℤ por eliminación fraction-free de Bareiss.

    Invariante (teorema de Bareiss): cada división por el pivote precedente
    es exacta en ℤ; el algoritmo lo verifica y aborta si se viola.
    Sin desbordamiento: los enteros de Python son ℤ.
    """
    n = len(matrix)
    if n == 0:
        return 1
    if any(len(row) != n for row in matrix):
        raise ValueError("Bareiss exige matriz cuadrada")
    if n == 1:
        return int(matrix[0][0])

    A = [[int(v) for v in row] for row in matrix]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            swap_row = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if swap_row is None:
                return 0
            A[k], A[swap_row] = A[swap_row], A[k]
            sign = -sign
        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = A[i][j] * pivot - A[i][k] * A[k][j]
                if numerator % prev != 0:
                    raise ArithmeticError(
                        "División no exacta en Bareiss (invariante violado)"
                    )
                A[i][j] = numerator // prev
            A[i][k] = 0
        prev = pivot
    return int(sign * A[n - 1][n - 1])


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Bézout iterativo: g = a·x + b·y con g = gcd(a, b) ≥ 0."""
    old_r, r = int(a), int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _gcd_lcm_unimodular(a: int, b: int) -> Tuple[int, int]:
    """
    Identidad SNF  diag(a, b) ~ diag(gcd, lcm)  con testigo unimodular.

    U = [[x, y], [−b/g, a/g]]  tiene  det U = (a·x + b·y)/g = 1.
    Se verifica el determinante: si no es 1 la identidad no es válida y
    se aborta (nunca se confía en la fórmula sin el testigo).
    """
    if a <= 0 or b <= 0:
        raise ValueError("gcd/lcm unimodular exige enteros positivos")
    g, x, y = _extended_gcd(a, b)
    det = x * (a // g) + y * (b // g)
    if det != 1:
        raise ArithmeticError("Testigo unimodular inválido en SNF (det ≠ 1)")
    return g, (a // g) * b


class IntegerExactArithmetic:
    """
    Álgebra exacta sobre ℤ: Bareiss, SNF, divisores determinántales,
    incidencia signada.

    El morfismo terminal `construct_observation_kernel` empaqueta el estado
    observado en el objeto que la Fase 2 consume como complejo de cadenas.
    """

    __slots__ = ()

    # ── 1.c.i  Forma normal de Smith ────────────────────────────────────────

    @staticmethod
    def smith_normal_form(matrix: List[List[int]]) -> Tuple[List[int], int]:
        """
        Forma normal de Smith sobre ℤ por operaciones elementales unimodulares.

        Retorna (factores invariantes s₁ | s₂ | … | s_r, rango r), sᵢ ≥ 1.

        Terminación: cada reiteración del bucle interno reduce estrictamente
        |pivote| (algoritmo euclidiano bidimensional) o cierra el bloque.
        """
        if not matrix or not matrix[0]:
            return [], 0
        A = [[int(v) for v in row] for row in matrix]
        m, n = len(A), len(A[0])
        if any(len(row) != n for row in A):
            raise ValueError("SNF exige matriz rectangular bien formada")

        rank = 0
        for d in range(min(m, n)):
            while True:
                best: Optional[Tuple[int, int, int]] = None
                for i in range(d, m):
                    for j in range(d, n):
                        v = abs(A[i][j])
                        if v and (best is None or v < best[0]):
                            best = (v, i, j)
                if best is None:
                    break
                _, pi, pj = best
                if pi != d:
                    A[d], A[pi] = A[pi], A[d]
                if pj != d:
                    for row in A:
                        row[d], row[pj] = row[pj], row[d]
                if A[d][d] < 0:
                    A[d] = [-v for v in A[d]]
                pivot = A[d][d]

                clean = True
                for i in range(d + 1, m):
                    if A[i][d]:
                        q = A[i][d] // pivot
                        if q:
                            A[i] = [a - q * b for a, b in zip(A[i], A[d])]
                        if A[i][d]:
                            clean = False
                for j in range(d + 1, n):
                    if A[d][j]:
                        q = A[d][j] // pivot
                        if q:
                            for row in A:
                                row[j] -= q * row[d]
                        if A[d][j]:
                            clean = False
                if not clean:
                    continue

                offender: Optional[int] = None
                for i in range(d + 1, m):
                    if any(A[i][j] % pivot for j in range(d + 1, n)):
                        offender = i
                        break
                if offender is not None:
                    A[d] = [a + b for a, b in zip(A[d], A[offender])]
                    continue
                break

            if A[d][d] == 0:
                break
            rank += 1

        factors = [abs(A[i][i]) for i in range(rank)]

        # Encadenamiento sᵢ | sᵢ₊₁ con testigo unimodular por cada paso.
        changed = True
        guard = 0
        while changed and guard <= len(factors) ** 2 + 2:
            changed = False
            guard += 1
            for i in range(len(factors) - 1):
                a, b = factors[i], factors[i + 1]
                if b % a == 0:
                    continue
                factors[i], factors[i + 1] = _gcd_lcm_unimodular(a, b)
                changed = True

        return [int(s) for s in factors], len(factors)

    # ── 1.c.ii  Divisores determinántales (control cruzado acotado) ─────────

    @staticmethod
    def determinantal_divisors(
        matrix: List[List[int]],
        *,
        minor_budget: int = _MINOR_BUDGET,
    ) -> Optional[List[int]]:
        """
        Δₖ = gcd{menores k×k}, Δ₀ := 1;  sₖ = Δₖ/Δₖ₋₁.

        Coste Σₖ C(m,k)·C(n,k).  Si excede `minor_budget` retorna None:
        la rigurosidad se declara acotada en lugar de fingirse.
        """
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        r = min(m, n)
        total = sum(comb(m, k) * comb(n, k) for k in range(1, r + 1))
        if total > minor_budget:
            return None
        deltas: List[int] = [1]
        for k in range(1, r + 1):
            g = 0
            for rows in combinations(range(m), k):
                for cols in combinations(range(n), k):
                    sub = [[matrix[i][j] for j in cols] for i in rows]
                    g = math.gcd(g, abs(_bareiss_det(sub)))
                    if g == 1:
                        break
                if g == 1:
                    break
            deltas.append(g)
            if g == 0:
                break
        return deltas

    # ── 1.c.iii  Estructura de incidencia ───────────────────────────────────

    @staticmethod
    def is_signed_incidence(matrix: List[List[int]]) -> bool:
        """
        ∂₁ es incidencia signada ⇔ cada columna contiene exactamente un +1,
        un −1 y ceros.  Tales matrices son totalmente unimodulares ⇒ SNF
        con todos los factores = 1 ⇒ H₀ libre.
        """
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        for j in range(n):
            col = [matrix[i][j] for i in range(m)]
            if sorted(v for v in col if v != 0) != [-1, 1]:
                return False
        return True

    @classmethod
    def torsion_from_snf(
        cls,
        matrix: List[List[int]],
        *,
        cross_check_minors: bool = True,
        minor_budget: int = _MINOR_BUDGET,
    ) -> Tuple[bool, Tuple[int, ...], Dict[str, Any]]:
        """Torsión ⇔ ∃ sₖ > 1.  Cruza SNF elemental con divisores determinántales."""
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return False, (), {"snf_status": "empty", "snf_rank": 0}

        factors, rank = cls.smith_normal_form(matrix)
        has_torsion = any(s > 1 for s in factors)
        diagnostics: Dict[str, Any] = {
            "snf_status": "ok",
            "invariant_factors": tuple(factors),
            "snf_rank": int(rank),
            "has_torsion": bool(has_torsion),
            "is_signed_incidence": cls.is_signed_incidence(matrix),
            "minors_cross_checked": False,
        }

        if cross_check_minors:
            deltas = cls.determinantal_divisors(matrix, minor_budget=minor_budget)
            if deltas is None:
                diagnostics["minors_status"] = "budget_exceeded"
            else:
                diagnostics["minors_cross_checked"] = True
                minor_factors: List[int] = []
                torsion_minors = False
                for k in range(1, len(deltas)):
                    if deltas[k] == 0 or deltas[k - 1] == 0:
                        break
                    if deltas[k] % deltas[k - 1] != 0:
                        torsion_minors = True
                        break
                    s = deltas[k] // deltas[k - 1]
                    torsion_minors |= s > 1
                    minor_factors.append(int(s))
                diagnostics["determinantal_divisors"] = tuple(deltas)
                diagnostics["minor_invariant_factors"] = tuple(minor_factors)
                diagnostics["minors_rank"] = len(minor_factors)
                if torsion_minors != has_torsion or len(minor_factors) != rank:
                    diagnostics["snf_minors_disagreement"] = True
                    has_torsion = has_torsion or torsion_minors

        return bool(has_torsion), tuple(factors), diagnostics

    # ── 1.d  MORFISMO TERMINAL DE LA FASE 1 / INICIAL DE LA FASE 2 ──────────

    def construct_observation_kernel(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
        session_sha256: str,
    ) -> "ObservationKernel":
        """
        Instala el complejo observado (ψ, A, ∂, L, α, sello) como kernel de
        observación.  La Fase 2 no revalida pertenencia a ℝⁿ: opera sobre
        este objeto ya calibrado.  L = D − A se simetriza exactamente.
        """
        A = np.asarray(adjacency_matrix, dtype=np.float64)
        degrees = np.sum(A, axis=1)
        laplacian = np.diag(degrees) - A
        laplacian = 0.5 * (laplacian + laplacian.T)
        B = np.asarray(boundary_matrix, dtype=np.float64)
        return ObservationKernel(
            state_vector=np.asarray(state_vector, dtype=np.float64),
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=float(diffusion_rate),
            session_sha256=session_sha256,
            degree_vector=np.asarray(degrees, dtype=np.float64),
            laplacian=np.asarray(laplacian, dtype=np.float64),
            n_nodes=int(np.asarray(state_vector).size),
            n_edges=int(B.shape[1]) if B.ndim == 2 else 0,
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
    n_edges: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class Phase1ReactionHandoff:
    """Frontera formal Φ₁→₂: salida cerrada de Fase 1, entrada abierta de Fase 2."""

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
    """Frontera formal Φ₂→₃: salida cerrada de Fase 2, entrada abierta de Fase 3."""

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
    hodge_compatible: bool = True
    lambda_max_laplacian: float = 0.0
    rayleigh_residual: float = 0.0
    betti_0: int = 1
    betti_1: int = 0
    cheeger_constant: float = math.nan
    is_signed_incidence: bool = False


class HeytingElement(Enum):
    """
    Álgebra de Heyting trivalente Ω₃ (cadena ⊥ < U < ⊤).

        a ∧ b = min,  a ∨ b = max,  a → b = ⊤ si a ≤ b si no b,  ¬a = a → ⊥
        Adjunción:  (a ∧ b) ≤ c  ⇔  a ≤ (b → c)
        No booleana:  ¬¬U = ⊤ ≠ U.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def leq(self, other: "HeytingElement") -> bool:
        return self.value <= other.value

    def meet(self, other: "HeytingElement") -> "HeytingElement":
        return HeytingElement(min(self.value, other.value))

    def join(self, other: "HeytingElement") -> "HeytingElement":
        return HeytingElement(max(self.value, other.value))

    def implies(self, other: "HeytingElement") -> "HeytingElement":
        return HeytingElement.COHERENT if self.value <= other.value else other

    def negate(self) -> "HeytingElement":
        """Pseudo-complemento ¬a = a → ⊥ (mayor b con a ∧ b = ⊥)."""
        return self.implies(HeytingElement.VETOED)

    def boolean_shadow(self) -> "HeytingElement":
        """Proyección clásica ¬¬a (colapso a la subálgebra booleana {⊥, ⊤})."""
        return self.negate().negate()

    @classmethod
    def verify_axioms(cls) -> bool:
        """
        Certificado exhaustivo (27 ternas): adjunción de Heyting, absorción,
        ¬a es el mayor pseudo-complemento y ⊤/⊥ son unidades.
        """
        els = list(cls)
        for a in els:
            if a.meet(a.negate()) is not cls.VETOED:
                return False
            if a.meet(cls.COHERENT) is not a or a.join(cls.VETOED) is not a:
                return False
            for b in els:
                if a.meet(a.join(b)) is not a or a.join(a.meet(b)) is not a:
                    return False
                for c in els:
                    if (a.meet(b).leq(c)) != (a.leq(b.implies(c))):
                        return False
        return True

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class ReactionChamberVerdict:
    """Certificado formal de regularidad espectral y homológica de la cámara."""

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

      FASE 1 — OBSERVE : validación Banach, sello, kernel.
                         Terminal: phase1_close_and_open_phase2.
      FASE 2 — ORIENT  : Smith, Hückel, Fiedler/Cheeger, CFL, Hodge, Betti.
                         Terminal: phase2_close_and_open_phase3.
      FASE 3 — DECIDE/ACT : átomos Ω₃, gracia, HMAC, Crowbar, sello.
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
        minor_budget: int = _MINOR_BUDGET,
        cheeger_exact_max_nodes: int = _CHEEGER_EXACT_MAX_NODES,
    ) -> None:
        """
        Parámetros
        ----------
        fiedler_threshold : umbral mínimo de λ₂.
        diffusion_crit_rate : tasa crítica CFL de trabajo (1/(2λ_max) = 0.125).
        grace_period : ventana de gracia del veto suave (s).
        tolerance : tolerancia numérica base.
        strict_homology : fallo/desacuerdo SNF ⇒ anomalía.
        rng_seed, jitter_sigma : jitter determinista del Crowbar (ns).
        authorized_tokens : tokens canónicos de override.
        hmac_key : clave HMAC-SHA256 para tokens `payload:signature`
                   (payload admite `;exp=<unix>` para caducidad).
        require_hodge_compatibility, hodge_tolerance : veto duro por Hodge.
        minor_budget : presupuesto de menores para el control cruzado SNF.
        cheeger_exact_max_nodes : n máximo para h(G) exacta por enumeración.
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
        if minor_budget < 0 or cheeger_exact_max_nodes < 0:
            raise ValueError("Los presupuestos combinatorios no pueden ser negativos.")

        self._fiedler_min = float(fiedler_threshold)
        self._cfl_crit = float(diffusion_crit_rate)
        self._grace_max = float(grace_period)
        self._tol = float(tolerance)
        self._reg = max(1e-15, self._tol * 1e-3)
        self._strict_homology = bool(strict_homology)
        self._jitter_sigma = float(jitter_sigma)
        self._require_hodge = bool(require_hodge_compatibility)
        self._hodge_tol = float(hodge_tolerance)
        self._minor_budget = int(minor_budget)
        self._cheeger_max_n = int(cheeger_exact_max_nodes)

        self._rng = np.random.default_rng(rng_seed)

        if authorized_tokens is None:
            self._authorized_tokens: Set[str] = {
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

        self._consumed_override_digests: Set[str] = set()
        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDADORES, DIAGNÓSTICO, SELLO, KERNEL
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

    def _validate_adjacency_matrix(self, adjacency_matrix: Any, n_nodes: int) -> np.ndarray:
        """A ∈ M_n(ℝ): forma, finitud, proyección de Hilbert–Schmidt A ← (A + Aᵀ)/2."""
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
        return _canonicalize_signed_zero(0.5 * (A + A.T))

    def _validate_boundary_matrix(self, boundary_matrix: Any, n_nodes: int) -> np.ndarray:
        """
        ∂₁ : C₁ → C₀ como matriz (|V| × |E|) real, finita.

        Se exige dim C₀ = n_nodes: el complejo de cadenas debe compartir
        soporte con el estado ψ.  Se admite |E| = 0 (grafo sin aristas).
        """
        if boundary_matrix is None:
            raise ValueError("boundary_matrix es obligatoria.")
        B = np.asarray(boundary_matrix, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError("boundary_matrix debe ser bidimensional (|V| × |E|).")
        if B.shape[0] != n_nodes:
            raise ValueError(
                f"boundary_matrix debe tener {n_nodes} filas (dim C₀ = |V|). "
                f"Obtenida: {B.shape}"
            )
        if not _is_finite_array(B):
            raise ValueError("boundary_matrix contiene valores NaN o infinitos.")
        return _canonicalize_signed_zero(B)

    def _validate_diffusion_rate(self, diffusion_rate: Any) -> float:
        """α ∈ ℝ finito (su signo se juzga en Fase 2, no aquí)."""
        try:
            rate = float(diffusion_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"diffusion_rate no es numérica: {diffusion_rate!r}") from exc
        if not math.isfinite(rate):
            raise ValueError(f"diffusion_rate no es finita: {rate}")
        return rate

    def _to_integer_matrix(self, boundary_matrix: np.ndarray) -> List[List[int]]:
        """Retracción ℝ → ℤ: exige coeficientes enteros a 10⁻⁸."""
        arr = np.asarray(boundary_matrix, dtype=np.float64)
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, atol=1e-8, rtol=0.0):
            raise ValueError("boundary_matrix debe contener coeficientes enteros exactos.")
        return [[int(x) for x in row] for row in rounded.tolist()]

    def _adjacency_diagnostics(self, adjacency: np.ndarray) -> Dict[str, Any]:
        """Invariantes combinatorios de A: simetría residual, signo, lazos, regularidad, binariedad."""
        skew = _frobenius(adjacency - adjacency.T)
        diag_norm = _norm2(np.diag(adjacency))
        min_entry = float(np.min(adjacency))
        degrees = np.sum(adjacency, axis=1)
        is_binary = bool(np.all(np.isclose(adjacency, np.rint(adjacency), atol=self._tol))
                         and np.all(np.rint(adjacency) >= 0)
                         and np.all(np.rint(adjacency) <= 1))
        return {
            "adjacency_skew_frobenius": skew,
            "adjacency_diag_norm": diag_norm,
            "adjacency_min_entry": min_entry,
            "degree_min": float(np.min(degrees)),
            "degree_max": float(np.max(degrees)),
            "is_regular": bool(np.ptp(degrees) <= self._tol),
            "is_nonnegative": bool(min_entry >= -self._tol),
            "is_loopless": bool(diag_norm <= self._tol),
            "is_binary": is_binary,
        }

    def _session_hash(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        boundary_matrix: np.ndarray,
        diffusion_rate: float,
    ) -> str:
        """Sello SHA-256 de sesión sobre la serialización canónica (con formas y dtype)."""
        h = hashlib.sha256()
        h.update(b"PHASE1/REACTION_AGENT_OBSERVE/" + __version__.encode("ascii"))
        h.update(_canonical_bytes(state_vector))
        h.update(_canonical_bytes(adjacency_matrix))
        h.update(_canonical_bytes(boundary_matrix))
        h.update(float(diffusion_rate).hex().encode("utf-8"))
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

            Φ₁→₂ : Datos crudos ↦ Observación validada.

        Último método de la Fase 1: construye el kernel (morfismo terminal
        del sustrato exacto) y lo admite de inmediato en
        `phase2_from_phase1`, primer método de la Fase 2.
        """
        psi = self._validate_state_vector(state_vector)
        n_nodes = int(psi.size)
        A = self._validate_adjacency_matrix(adjacency_matrix, n_nodes)
        B = self._validate_boundary_matrix(boundary_matrix, n_nodes)
        rate = self._validate_diffusion_rate(diffusion_rate)

        session_sha256 = self._session_hash(psi, A, B, rate)
        kernel = self.construct_observation_kernel(
            state_vector=psi,
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=rate,
            session_sha256=session_sha256,
        )

        diagnostics: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "n_edges": int(kernel.n_edges),
            "boundary_shape": tuple(int(x) for x in B.shape),
            "diffusion_rate": rate,
            "session_sha256_prefix": session_sha256[:16],
            "state_l2_norm": _norm2(psi),
            "laplacian_trace": float(np.trace(kernel.laplacian)),
            "agent_version": __version__,
        }
        diagnostics.update(self._adjacency_diagnostics(A))

        handoff = Phase1ReactionHandoff(
            state_vector=psi,
            adjacency_matrix=A,
            boundary_matrix=B,
            diffusion_rate=rate,
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
    # FASE 2 — ORIENT: SMITH, HÜCKEL, FIEDLER/CHEEGER, CFL, HODGE, BETTI
    # Continúa construct_observation_kernel / phase1_close_and_open_phase2
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1ReactionHandoff) -> Phase1ReactionHandoff:
        """Fase 2.0 — Admisión formal Φ₁→₂ (tipo, punto de entrada, sello)."""
        if not isinstance(handoff, Phase1ReactionHandoff):
            raise TypeError(
                f"phase2_from_phase1 exige Phase1ReactionHandoff; recibido {type(handoff)!r}"
            )
        if handoff.next_entrypoint != "phase2_from_phase1":
            raise ValueError(
                "Phase1ReactionHandoff inválido: se esperaba 'phase2_from_phase1'."
            )
        if not handoff.session_sha256:
            raise ValueError("Phase1ReactionHandoff sin sello de sesión.")
        return handoff

    def _kernel_or_rebuild(self, handoff: Phase1ReactionHandoff) -> ObservationKernel:
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

    # ── 2.1  Smith ──────────────────────────────────────────────────────────

    def _evaluate_smith_full(
        self,
        boundary_matrix: np.ndarray,
    ) -> Tuple[bool, Tuple[int, ...], Dict[str, Any]]:
        """
        SNF completa: torsión, factores invariantes, rango ℤ, incidencia.

        Política `strict_homology`: no-integralidad, error o desacuerdo
        SNF/menores ⇒ anomalía.
        """
        try:
            int_matrix = self._to_integer_matrix(boundary_matrix)
        except Exception as exc:
            return bool(self._strict_homology), (), {
                "snf_status": "non_integral",
                "smith_error": str(exc),
                "snf_rank": 0,
            }
        m = len(int_matrix)
        n = len(int_matrix[0]) if m else 0
        if m == 0 or n == 0:
            return False, (), {"snf_status": "empty", "snf_rank": 0,
                               "is_signed_incidence": False}
        try:
            has, factors, diag = self.torsion_from_snf(
                int_matrix,
                cross_check_minors=True,
                minor_budget=self._minor_budget,
            )
            if diag.get("snf_minors_disagreement") and self._strict_homology:
                has = True
            return bool(has), factors, diag
        except Exception as exc:
            logger.warning("SNF exacta falló: %s", exc)
            return bool(self._strict_homology), (), {
                "snf_status": "error", "smith_error": str(exc), "snf_rank": 0,
            }

    def evaluate_smith_normal_form_torsion(self, boundary_matrix: np.ndarray) -> bool:
        """API legada — True ⇒ torsión (anomalía)."""
        has, _, _ = self._evaluate_smith_full(np.asarray(boundary_matrix, dtype=np.float64))
        return bool(has)

    # ── 2.2  Hückel ─────────────────────────────────────────────────────────

    def _is_cycle_graph(self, adjacency: np.ndarray) -> bool:
        """
        Cₙ ⇔ A binaria simétrica sin lazos, 2-regular y conexa (BFS).
        Un grafo 2-regular conexo es necesariamente un ciclo.
        """
        n = adjacency.shape[0]
        if n < 3:
            return False
        R = np.rint(adjacency)
        if not np.allclose(adjacency, R, atol=self._tol):
            return False
        if np.any(R < 0) or np.any(R > 1) or np.any(np.diag(R) != 0):
            return False
        if not np.all(R.sum(axis=1) == 2):
            return False
        seen = {0}
        frontier = [0]
        while frontier:
            v = frontier.pop()
            for u in np.nonzero(R[v])[0]:
                if int(u) not in seen:
                    seen.add(int(u))
                    frontier.append(int(u))
        return len(seen) == n

    def _audit_huckel(
        self,
        kernel: ObservationKernel,
        alpha: float = _HUCKEL_ALPHA,
        beta: float = _HUCKEL_BETA,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Resonancia de Hückel con certificados a posteriori.

            H = αI + βA,  E = ⟨ψ̂, Hψ̂⟩ ≥ λ_min           (Courant–Fischer)
            r = ‖(H − E)ψ̂‖₂,  dist(E, spec H) ≤ r      (Weyl)
            λ_min ≥ E − r²/(λ₂ − E)  si r < λ₂ − E      (Kato–Temple)
            F = ‖P₀ψ̂‖²  fidelidad al subespacio fundamental (degeneración incluida)
        """
        psi = kernel.state_vector
        A = kernel.adjacency_matrix
        n = kernel.n_nodes
        H = alpha * np.eye(n, dtype=np.float64) + beta * A
        H = 0.5 * (H + H.T)

        norm = _norm2(psi)
        nan_diag = {"lambda_min": math.nan, "rayleigh_residual": math.nan}
        if norm < self._reg:
            return 0.0, False, {"huckel_status": "zero_state", **nan_diag}

        psi_hat = psi / norm
        Hpsi = H @ psi_hat
        energy = _dot2(psi_hat, Hpsi)
        if not math.isfinite(energy):
            return float("nan"), False, {"huckel_status": "nonfinite_energy", **nan_diag}

        residual = _norm2(Hpsi - energy * psi_hat)
        eigvals, eigvecs = la.eigh(H)
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])
        first_excited = float(eigvals[1]) if n >= 2 else lambda_min
        spectral_gap = first_excited - lambda_min
        gap_to_ground = float(energy - lambda_min)

        scale = 1.0 + abs(lambda_min) + abs(lambda_max)
        tau = max(self._tol, 1e-10) * scale
        courant_fischer_ok = gap_to_ground >= -tau

        ground_mask = eigvals <= lambda_min + tau
        P0 = eigvecs[:, ground_mask]
        fidelity = float(np.sum((P0.T @ psi_hat) ** 2))

        nearest = float(np.min(np.abs(eigvals - energy)))
        weyl_ok = nearest <= residual + tau

        kato_temple_bound: Optional[float] = None
        kato_temple_ok: Optional[bool] = None
        delta = first_excited - energy
        if n >= 2 and delta > residual:
            kato_temple_bound = residual ** 2 / delta
            kato_temple_ok = gap_to_ground <= kato_temple_bound + tau

        is_near_ground = gap_to_ground <= tau
        is_eigen = residual <= max(self._tol, 1e-8) * (1.0 + abs(energy))
        is_stable = bool(is_near_ground and is_eigen and courant_fischer_ok)

        diagnostics: Dict[str, Any] = {
            "huckel_status": "ok",
            "alpha": float(alpha),
            "beta": float(beta),
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "huckel_spectral_gap": float(spectral_gap),
            "ground_degeneracy": int(np.count_nonzero(ground_mask)),
            "energy_gap_to_ground": gap_to_ground,
            "rayleigh_residual": residual,
            "ground_fidelity": fidelity,
            "courant_fischer_ok": bool(courant_fischer_ok),
            "weyl_nearest_eigenvalue_distance": nearest,
            "weyl_ok": bool(weyl_ok),
            "kato_temple_bound": kato_temple_bound,
            "kato_temple_ok": kato_temple_ok,
            "is_near_ground": bool(is_near_ground),
            "is_approximate_eigenstate": bool(is_eigen),
            "is_cycle_graph": False,
        }
        if self._is_cycle_graph(A):
            analytical = tuple(sorted(
                alpha + 2.0 * beta * math.cos(2.0 * math.pi * k / n) for k in range(n)
            ))
            diagnostics["is_cycle_graph"] = True
            diagnostics["huckel_analytical_levels"] = analytical
            diagnostics["huckel_analytical_residual"] = float(
                max(abs(float(a) - float(b)) for a, b in zip(eigvals, analytical))
            )
        return float(energy), is_stable, diagnostics

    def evaluate_huckel_resonance(
        self,
        state_vector: np.ndarray,
        adjacency_matrix: np.ndarray,
        alpha: float = _HUCKEL_ALPHA,
        beta: float = _HUCKEL_BETA,
    ) -> Tuple[float, bool]:
        """API legada — Rayleigh de Hückel y estabilidad del fundamental."""
        psi = self._validate_state_vector(state_vector)
        A = self._validate_adjacency_matrix(adjacency_matrix, psi.size)
        kernel = self.construct_observation_kernel(
            psi, A, np.zeros((psi.size, 0)), 0.0, "LEGACY/HUCKEL",
        )
        energy, stable, _ = self._audit_huckel(kernel, alpha=alpha, beta=beta)
        return energy, stable

    # ── 2.3  Fiedler / Cheeger ──────────────────────────────────────────────

    def _cheeger_constant_exact(self, adjacency: np.ndarray) -> Optional[float]:
        """
        h(G) = min_{0<|S|≤n/2} w(S, Sᶜ)/|S| por enumeración exhaustiva de
        cortes (2ⁿ subconjuntos).  Sólo para n ≤ cheeger_exact_max_nodes.
        """
        n = adjacency.shape[0]
        if n < 2 or n > self._cheeger_max_n:
            return None
        best = math.inf
        half = n // 2
        for mask in range(1, 1 << n):
            size = bin(mask).count("1")
            if size > half:
                continue
            s = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=np.float64, count=n)
            cut = float(s @ adjacency @ (1.0 - s))
            best = min(best, cut / size)
        return float(best)

    def _audit_connectivity(
        self,
        kernel: ObservationKernel,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Conectividad espectral y núcleo laplaciano.

            L ⪰ 0,  β₀ = #{λ ≤ τ},  conexo ⇔ β₀ = 1 ∧ λ₂ ≥ τ_Fiedler
            Cheeger:  λ₂/2 ≤ h(G) ≤ √(2 d_max λ₂)  (A ≥ 0)
        """
        n = kernel.n_nodes
        A = kernel.adjacency_matrix
        if n <= 1:
            return math.inf, True, {
                "fiedler_status": "single_node", "lambda_max_laplacian": 0.0,
                "betti_0": 1, "laplacian_spectrum": (0.0,), "is_psd": True,
            }
        eigvals = la.eigvalsh(kernel.laplacian)
        lam1 = float(eigvals[0])
        fiedler = float(eigvals[1])
        lam_max = float(eigvals[-1])
        tau0 = max(self._tol, 1e-10) * (1.0 + abs(lam_max))

        is_psd = lam1 >= -tau0
        kernel_ok = abs(lam1) <= tau0
        betti_0 = int(np.count_nonzero(eigvals <= tau0))
        is_connected = bool(is_psd and betti_0 == 1 and fiedler >= self._fiedler_min - self._tol)

        d_max = float(np.max(kernel.degree_vector))
        nonneg = bool(np.min(A) >= -self._tol)
        cheeger_lower = 0.5 * max(fiedler, 0.0)
        cheeger_upper = math.sqrt(max(2.0 * d_max * max(fiedler, 0.0), 0.0))
        h_exact = self._cheeger_constant_exact(A) if nonneg else None
        cheeger_certified: Optional[bool] = None
        if h_exact is not None and math.isfinite(h_exact):
            cheeger_certified = bool(cheeger_lower - tau0 <= h_exact <= cheeger_upper + tau0)

        return fiedler, is_connected, {
            "fiedler_status": "connected" if is_connected else "disconnected",
            "fiedler_value": fiedler,
            "fiedler_threshold": self._fiedler_min,
            "lambda_1": lam1,
            "lambda_max_laplacian": lam_max,
            "laplacian_spectrum": tuple(float(v) for v in eigvals),
            "is_psd": bool(is_psd),
            "laplacian_kernel_ok": bool(kernel_ok),
            "betti_0": betti_0,
            "degree_max": d_max,
            "cheeger_lower_bound": cheeger_lower,
            "cheeger_upper_bound": cheeger_upper,
            "cheeger_constant_exact": h_exact,
            "cheeger_inequality_certified": cheeger_certified,
        }

    # ── 2.4  CFL / von Neumann ──────────────────────────────────────────────

    def _audit_cfl(
        self,
        diffusion_rate: float,
        laplacian_spectrum: Optional[Sequence[float]] = None,
        degree_max: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Estabilidad del Euler explícito ψ ← (I − αL)ψ.

            ρ(P) = max_λ |1 − αλ| ≤ 1        (von Neumann)  ⇔ α ≤ 2/λ_max
            κ = max_{λ>0} |1 − αλ| < 1       (contracción en 1^⊥)
            P ≥ 0  ⇔  α d_max ≤ 1            (positividad / estocasticidad)
            trabajo:  0 ≤ α < min(α_crit, 1/(2λ_max))
        """
        if not math.isfinite(diffusion_rate):
            return False, {"cfl_status": "nonfinite_rate"}
        if diffusion_rate < 0.0:
            return False, {"cfl_status": "negative_rate", "diffusion_rate": float(diffusion_rate)}

        spec = np.asarray(laplacian_spectrum if laplacian_spectrum is not None
                          else [0.0, _LAMBDA_MAX_C6], dtype=np.float64)
        lam_max = float(np.max(spec)) if spec.size and math.isfinite(float(np.max(spec))) else _LAMBDA_MAX_C6
        if lam_max <= 0.0:
            lam_max = _LAMBDA_MAX_C6

        alpha_sharp = 2.0 / lam_max
        alpha_work = min(self._cfl_crit, 1.0 / (2.0 * lam_max))
        is_stable = bool(diffusion_rate < alpha_work - self._tol)

        amplification = 1.0 - diffusion_rate * spec
        rho = float(np.max(np.abs(amplification)))
        positive = spec > max(self._tol, 1e-10) * (1.0 + lam_max)
        kappa = float(np.max(np.abs(amplification[positive]))) if np.any(positive) else 0.0
        d_max = float(degree_max) if degree_max is not None else lam_max
        positivity = bool(diffusion_rate * d_max <= 1.0 + self._tol)

        return is_stable, {
            "cfl_status": "stable" if is_stable else "unstable",
            "diffusion_rate": float(diffusion_rate),
            "cfl_critical": float(self._cfl_crit),
            "cfl_alpha_sharp": float(alpha_sharp),
            "cfl_alpha_work": float(alpha_work),
            "cfl_margin": float(alpha_work - diffusion_rate),
            "cfl_sharp_stable": bool(diffusion_rate <= alpha_sharp + self._tol),
            "lambda_max_used": float(lam_max),
            "propagator_spectral_radius": rho,
            "von_neumann_ok": bool(rho <= 1.0 + self._tol),
            "consensus_contraction_factor": kappa,
            "consensus_contractive": bool(kappa < 1.0 - self._tol),
            "positivity_preserving": positivity,
            "mass_conserving": True,  # 1ᵀ(I − αL) = 1ᵀ pues L = Lᵀ, L1 = 0
        }

    # ── 2.5  Hodge–Tellegen ─────────────────────────────────────────────────

    def _infer_edge_weights(self, B: np.ndarray, A: np.ndarray) -> Optional[np.ndarray]:
        """Si ∂ es incidencia signada, w_e = A[i_e, j_e]; si no, None."""
        weights: List[float] = []
        for e in range(B.shape[1]):
            nz = np.nonzero(B[:, e])[0]
            if nz.size != 2:
                return None
            vals = sorted(float(v) for v in B[nz, e])
            if vals != [-1.0, 1.0]:
                return None
            weights.append(float(A[nz[0], nz[1]]))
        return np.asarray(weights, dtype=np.float64)

    def _audit_hodge_tellegen(self, kernel: ObservationKernel) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compatibilidad Hodge–Tellegen del complejo con la métrica de A.

            L_∂ = ∂∂ᵀ  (binario)   L_∂,W = ∂ W ∂ᵀ  (ponderado inferido)
            ρ = ‖L_A − L_∂‖_F ≤ τ ,   Kirchhoff: ‖∂ᵀ1‖_∞ = 0
            rank_ℝ(∂) por SVD;  dim ker ∂ᵀ = n − rank = β₀ (formas armónicas)
        """
        B = kernel.boundary_matrix
        A = kernel.adjacency_matrix
        L_a = kernel.laplacian
        n = kernel.n_nodes
        ones = np.ones(n, dtype=np.float64)

        if B.shape[1] == 0:
            residual = _frobenius(L_a)
            return residual, bool(residual <= self._hodge_tol), {
                "hodge_status": "edgeless",
                "hodge_residual": residual,
                "hodge_tolerance": float(self._hodge_tol),
                "kirchhoff_column_sum_inf": 0.0,
                "numerical_rank_boundary": 0,
                "harmonic_dimension": int(n),
            }

        L_unw = B @ B.T
        residual_unw = _frobenius(L_a - L_unw)
        weights = self._infer_edge_weights(B, A)
        residual_w: Optional[float] = None
        if weights is not None:
            residual_w = _frobenius(L_a - (B * weights) @ B.T)
        residual = residual_w if residual_w is not None else residual_unw
        compatible = bool(residual <= self._hodge_tol)

        kirchhoff = float(np.max(np.abs(B.T @ ones)))
        svals = la.svdvals(B)
        rank_tol = float(svals[0]) * max(B.shape) * _MACHINE_EPS if svals.size else 0.0
        rank_num = int(np.count_nonzero(svals > max(rank_tol, self._tol)))

        return residual, compatible, {
            "hodge_status": "compatible" if compatible else "incompatible",
            "hodge_residual": float(residual),
            "hodge_residual_unweighted": float(residual_unw),
            "hodge_residual_weighted": residual_w,
            "hodge_tolerance": float(self._hodge_tol),
            "edge_weights_inferred": weights is not None,
            "kirchhoff_column_sum_inf": kirchhoff,
            "kirchhoff_ok": bool(kirchhoff <= self._tol),
            "numerical_rank_boundary": rank_num,
            "harmonic_dimension": int(n - rank_num),
        }

    # ── 2.6  Libro mayor homológico ─────────────────────────────────────────

    def _homology_ledger(
        self,
        kernel: ObservationKernel,
        smith_diag: Dict[str, Any],
        fiedler_diag: Dict[str, Any],
        hodge_diag: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        β₀ = dim ker L (espectral), rank_ℤ ∂₁ (Smith), β₁ = |E| − rank,
        χ = |V| − |E|.  Consistencia: rank_ℤ = rank_ℝ = |V| − β₀ y χ = β₀ − β₁.
        """
        n_v, n_e = kernel.n_nodes, kernel.n_edges
        betti_0 = int(fiedler_diag.get("betti_0", 1))
        rank_snf = smith_diag.get("snf_rank")
        rank_num = hodge_diag.get("numerical_rank_boundary")
        rank_expected = n_v - betti_0
        rank_used = int(rank_snf) if rank_snf else rank_expected
        betti_1 = max(0, n_e - rank_used)
        chi = n_v - n_e
        rank_consistent = (rank_snf is None or int(rank_snf) == rank_expected) and \
                          (rank_num is None or int(rank_num) == rank_expected)
        return {
            "betti_0": betti_0,
            "betti_1": int(betti_1),
            "euler_characteristic": int(chi),
            "rank_boundary_smith": rank_snf,
            "rank_boundary_numerical": rank_num,
            "rank_boundary_expected": int(rank_expected),
            "rank_consistent": bool(rank_consistent),
            "euler_poincare_ok": bool(betti_0 - betti_1 == chi),
            "is_single_cycle": bool(betti_0 == 1 and betti_1 == 1 and chi == 0),
        }

    # ── 2.7  Cierre de Fase 2 / apertura de Fase 3 ──────────────────────────

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1ReactionHandoff,
    ) -> Phase2ReactionHandoff:
        """
        Fase 2.7 — Cierre formal de Fase 2 y apertura de Fase 3.

            Φ₂→₃ : Observación validada ↦ Invariantes auditados.

        Último método de la Fase 2: consume el kernel, ejecuta Smith /
        Hückel / Fiedler / CFL / Hodge / Betti y admite de inmediato el
        handoff en `phase3_from_phase2`, primer método de la Fase 3.
        """
        _ = self.phase2_from_phase1(phase1_handoff)
        kernel = self._kernel_or_rebuild(phase1_handoff)

        has_torsion, factors, smith_diag = self._evaluate_smith_full(kernel.boundary_matrix)
        energy, is_aromatic, huckel_diag = self._audit_huckel(kernel)
        fiedler, is_connected, fiedler_diag = self._audit_connectivity(kernel)

        lam_max = float(fiedler_diag.get("lambda_max_laplacian", _LAMBDA_MAX_C6))
        if not math.isfinite(lam_max):
            lam_max = _LAMBDA_MAX_C6
        is_cfl, cfl_diag = self._audit_cfl(
            kernel.diffusion_rate,
            laplacian_spectrum=fiedler_diag.get("laplacian_spectrum"),
            degree_max=fiedler_diag.get("degree_max"),
        )
        hodge_residual, hodge_ok, hodge_diag = self._audit_hodge_tellegen(kernel)
        homology = self._homology_ledger(kernel, smith_diag, fiedler_diag, hodge_diag)

        diagnostics: Dict[str, Any] = {
            "phase1": dict(phase1_handoff.diagnostics),
            "smith": smith_diag,
            "huckel": huckel_diag,
            "fiedler": {k: v for k, v in fiedler_diag.items() if k != "laplacian_spectrum"},
            "cfl": cfl_diag,
            "hodge": hodge_diag,
            "homology": homology,
            # Claves planas legadas
            "n_nodes": int(kernel.n_nodes),
            "n_edges": int(kernel.n_edges),
            "session_sha256_prefix": kernel.session_sha256[:16],
            "has_torsion_anomaly": bool(has_torsion),
            "fiedler_value": float(fiedler),
            "lambda_max_laplacian": float(lam_max),
            "rayleigh_residual": float(huckel_diag.get("rayleigh_residual", math.nan)),
            "hodge_required": bool(self._require_hodge),
            "hodge_ok": bool(hodge_ok),
            "betti_0": int(homology["betti_0"]),
            "betti_1": int(homology["betti_1"]),
            "agent_version": __version__,
        }

        handoff = Phase2ReactionHandoff(
            state_vector=kernel.state_vector,
            adjacency_matrix=kernel.adjacency_matrix,
            boundary_matrix=kernel.boundary_matrix,
            diffusion_rate=kernel.diffusion_rate,
            session_sha256=kernel.session_sha256,
            aromatic_energy=float(energy),
            is_aromatic_stable=bool(is_aromatic),
            has_torsion_anomaly=bool(has_torsion),
            is_cfl_stable=bool(is_cfl),
            is_connected=bool(is_connected),
            fiedler_value=float(fiedler),
            diagnostics=diagnostics,
            next_entrypoint="phase3_from_phase2",
            invariant_factors=tuple(int(s) for s in factors),
            hodge_residual=float(hodge_residual) if math.isfinite(hodge_residual) else math.inf,
            hodge_compatible=bool(hodge_ok),
            lambda_max_laplacian=float(lam_max),
            rayleigh_residual=float(huckel_diag.get("rayleigh_residual", math.nan)),
            betti_0=int(homology["betti_0"]),
            betti_1=int(homology["betti_1"]),
            cheeger_constant=float(fiedler_diag.get("cheeger_constant_exact") or math.nan),
            is_signed_incidence=bool(smith_diag.get("is_signed_incidence", False)),
        )

        # Acoplamiento formal Φ₂→₃: la frontera debe ser admitida por Fase 3.
        _ = self.phase3_from_phase2(handoff)

        logger.debug(
            "Fase Orient [REACTION_AGENT]: E=%.6e torsion=%s cfl=%s β=(%d,%d) hodge=%.3e",
            energy, has_torsion, is_cfl, handoff.betti_0, handoff.betti_1, hodge_residual,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: HEYTING, GRACIA, OVERRIDE, CROWBAR, SELLO
    # Continúa phase2_close_and_open_phase3 / Phase2ReactionHandoff
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2ReactionHandoff) -> Phase2ReactionHandoff:
        """Fase 3.0 — Admisión formal Φ₂→₃ (tipo, punto de entrada, sello)."""
        if not isinstance(handoff, Phase2ReactionHandoff):
            raise TypeError(
                f"phase3_from_phase2 exige Phase2ReactionHandoff; recibido {type(handoff)!r}"
            )
        if handoff.next_entrypoint != "phase3_from_phase2":
            raise ValueError(
                "Phase2ReactionHandoff inválido: se esperaba 'phase3_from_phase2'."
            )
        if not handoff.session_sha256:
            raise ValueError("Phase2ReactionHandoff sin sello de sesión.")
        return handoff

    # ── 3.1  Override humano ────────────────────────────────────────────────

    @staticmethod
    def _constant_time_member(token: str, pool: Set[str]) -> bool:
        """Pertenencia sin cortocircuito (acumula sobre todo el conjunto)."""
        found = False
        for candidate in sorted(pool):
            found |= hmac.compare_digest(token.encode("utf-8"), candidate.encode("utf-8"))
        return found

    @staticmethod
    def _parse_expiry(payload: str) -> Optional[float]:
        """Extrae `exp=<unix>` de un payload `k=v;k=v…`; None si ausente."""
        for part in payload.split(";"):
            part = part.strip()
            if part.startswith("exp="):
                try:
                    return float(part[4:])
                except ValueError:
                    return math.nan
        return None

    def _verify_hmac_override(
        self,
        token: Optional[str],
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Fase 3.1 — Override humano.

        Acepta (a) tokens canónicos del conjunto cerrado, comparados en
        tiempo constante; (b) tokens `payload:signature` HMAC-SHA256 con
        caducidad opcional `exp=` y anti-replay por jarra de nonces.
        """
        info: Dict[str, Any] = {"override_kind": None, "override_reason": None}
        if token is None:
            return False, info
        token_str = str(token).strip()
        if not token_str:
            return False, info

        if self._constant_time_member(token_str, self._authorized_tokens):
            info["override_kind"] = "static"
            return True, info

        if self._hmac_key is not None and ":" in token_str:
            payload, signature = token_str.rsplit(":", 1)
            expected = hmac.new(self._hmac_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            try:
                valid = hmac.compare_digest(signature.strip().lower(), expected)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                info["override_reason"] = "bad_signature"
                return False, info
            exp = self._parse_expiry(payload)
            now = float(current_time if current_time is not None else time.time())
            if exp is not None and (math.isnan(exp) or now > exp):
                info["override_reason"] = "expired"
                return False, info
            digest = hashlib.sha256(token_str.encode("utf-8")).hexdigest()
            if digest in self._consumed_override_digests:
                info["override_reason"] = "replay"
                return False, info
            self._consumed_override_digests.add(digest)
            info["override_kind"] = "hmac"
            info["override_expiry"] = exp
            return True, info

        info["override_reason"] = "unknown_token"
        return False, info

    # ── 3.2  Átomos de Heyting y decisión ───────────────────────────────────

    def _heyting_atoms(self, handoff: Phase2ReactionHandoff) -> Dict[str, HeytingElement]:
        """
        Átomos aᵢ ∈ Ω₃ de cada auditoría.  Anomalías no perdonables
        generan ⊥; la pérdida de resonancia genera U (perdonable).
        """
        top, bot, mid = HeytingElement.COHERENT, HeytingElement.VETOED, HeytingElement.DEGRADED
        return {
            "smith_torsion": bot if handoff.has_torsion_anomaly else top,
            "cfl_violation": bot if not handoff.is_cfl_stable else top,
            "spectral_disconnection": bot if not handoff.is_connected else top,
            "nonfinite_energy": bot if not math.isfinite(handoff.aromatic_energy) else top,
            "hodge_incompatibility": (
                bot if (self._require_hodge and handoff.hodge_residual > self._hodge_tol) else top
            ),
            "aromatic_resonance": mid if not handoff.is_aromatic_stable else top,
        }

    def _hard_reasons(self, handoff: Phase2ReactionHandoff) -> List[str]:
        """Generadores de ⊥ (derivados de los átomos)."""
        return [k for k, v in self._heyting_atoms(handoff).items() if v is HeytingElement.VETOED]

    def _clear_soft_veto(self) -> None:
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def phase3_decide_heyting(
        self,
        handoff: Phase2ReactionHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> Dict[str, Any]:
        """
        Fase 3.2 — Clasificador de Heyting trivalente Ω₃.

            base = ⋀ᵢ aᵢ
            base = ⊥                      ⇒ veto duro inmediato
            base = U, gracia vigente      ⇒ DEGRADED (veto suave)
            base = U, override válido     ⇒ DEGRADED bajo custodia humana
            base = U, gracia expirada     ⇒ ⊥ terminal
            base = ⊤                      ⇒ COHERENT
        """
        curr_time = float(current_time if current_time is not None else time.time())
        atoms = self._heyting_atoms(handoff)
        base = reduce(HeytingElement.meet, atoms.values(), HeytingElement.COHERENT)
        hard_reasons = [k for k, v in atoms.items() if v is HeytingElement.VETOED]

        override_valid, override_info = self._verify_hmac_override(override_token, curr_time)
        override_sha = (
            hashlib.sha256(str(override_token).encode("utf-8")).hexdigest()
            if override_valid and override_token is not None else None
        )

        element = base
        is_hard_veto = False
        is_soft_veto = False
        time_remaining = 0.0
        clock_regression = False

        if base is HeytingElement.VETOED:
            is_hard_veto = True
            self._clear_soft_veto()
            logger.critical(
                "¡VETO DURO INSTANTÁNEO! Átomos en ⊥: %s", ", ".join(hard_reasons),
            )
        elif base is HeytingElement.DEGRADED:
            is_soft_veto = True
            if simulate_grace_expired:
                time_remaining = 0.0
            elif not self._is_soft_veto_active:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                time_remaining = self._grace_max
                logger.warning(
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! Pérdida de resonancia aromática. "
                    "Ventana de gracia de %.0f s iniciada.", self._grace_max,
                )
            else:
                elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                if elapsed < 0.0:
                    clock_regression = True
                    logger.warning("Regresión de reloj detectada (%.3f s); se ignora.", elapsed)
                    elapsed = 0.0
                time_remaining = max(0.0, self._grace_max - elapsed)

            if override_valid:
                element = HeytingElement.DEGRADED
                is_soft_veto = False
                time_remaining = 0.0
                self._clear_soft_veto()
                logger.info(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] INYECTADO! Sello: %s",
                    override_sha[:16] if override_sha else "UNKNOWN",
                )
            elif time_remaining <= self._tol or simulate_grace_expired:
                element = HeytingElement.VETOED
                is_hard_veto = True
                is_soft_veto = False
                self._clear_soft_veto()
                logger.critical(
                    "¡PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO! Colapso a VETOED."
                )
        else:
            self._clear_soft_veto()

        return {
            "heyting_verdict": element.name,
            "heyting_element": element.value,
            "heyting_base": base.name,
            "heyting_atoms": {k: v.name for k, v in atoms.items()},
            "heyting_pseudo_complement": element.negate().name,
            "boolean_shadow": element.boolean_shadow().name,
            "is_hard_veto": bool(is_hard_veto),
            "is_soft_veto": bool(is_soft_veto and element is HeytingElement.DEGRADED),
            "hard_reasons": hard_reasons,
            "override_valid": bool(override_valid),
            "override_sha256": override_sha,
            "override_info": override_info,
            "time_grace_remaining": float(time_remaining),
            "clock_regression": clock_regression,
            "current_time": curr_time,
            "simulate_grace_expired": bool(simulate_grace_expired),
        }

    # ── 3.3  Crowbar ────────────────────────────────────────────────────────

    def phase3_actuate_crowbar_interlock(self, heyting_verdict: str) -> Tuple[bool, float]:
        """
        Fase 3.3 — Actuación simulada del interlock Crowbar.

        Si ⊥, se emula la conmutación IRAM con latencia gaussiana censurada
        a [385, 415] ns.  Sólo telemetría; sin I/O de hardware real.
        """
        if heyting_verdict != HeytingElement.VETOED.name:
            return False, 0.0
        jitter = float(self._rng.normal(loc=398.5, scale=self._jitter_sigma))
        latency_ns = float(np.clip(jitter, 385.0, 415.0))
        logger.critical(
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO DE LA CÁMARA!\n"
            "  - Ejecutando isVerdictCoherent() en C++...\n"
            "  - Despachando ISR en IRAM (< %.0f ns)...\n"
            "  - GPIO14 → HIGH en %.2f ns (%s spec)...\n"
            "  - ¡Tiristor BT151 (Crowbar) gatillado! Mezcladoras y bombas paralizadas.",
            _CROWBAR_IRAM_LATENCY_NS, latency_ns,
            "dentro de" if latency_ns <= _CROWBAR_IRAM_LATENCY_NS else "FUERA de",
        )
        return True, latency_ns

    # ── 3.4  Sello ──────────────────────────────────────────────────────────

    def _canonical_seal_payload(
        self,
        handoff: Phase2ReactionHandoff,
        decision: Dict[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> bytes:
        """Codificación canónica (hex IEEE, enteros, formas) del certificado."""
        atoms = decision.get("heyting_atoms", {}) or {}
        parts: List[bytes] = [
            b"PHASE3/REACTION_AGENT_GOVERNANCE",
            __version__.encode("utf-8"),
            _canonical_bytes(handoff.state_vector),
            _canonical_bytes(handoff.adjacency_matrix),
            _canonical_bytes(handoff.boundary_matrix),
            str(decision["heyting_verdict"]).encode("utf-8"),
            repr(sorted(atoms.items())).encode("utf-8"),
            repr(sorted(decision.get("hard_reasons", []))).encode("utf-8"),
            float(handoff.aromatic_energy).hex().encode("utf-8"),
            b"1" if handoff.is_aromatic_stable else b"0",
            b"1" if handoff.has_torsion_anomaly else b"0",
            b"1" if handoff.is_cfl_stable else b"0",
            b"1" if handoff.is_connected else b"0",
            float(handoff.fiedler_value).hex().encode("utf-8"),
            float(handoff.hodge_residual).hex().encode("utf-8"),
            str(int(handoff.betti_0)).encode("ascii"),
            str(int(handoff.betti_1)).encode("ascii"),
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
        """Fase 3.4 — Sello SHA-256 de lazo cerrado."""
        return hashlib.sha256(
            self._canonical_seal_payload(handoff, decision, interlock_fired, switching_latency_ns)
        ).hexdigest()

    # ── 3.5  Veredicto y verificación ───────────────────────────────────────

    def phase3_issue_verdict(
        self,
        handoff: Phase2ReactionHandoff,
        decision: Dict[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> ReactionChamberVerdict:
        """Fase 3.5 — Emisión del veredicto certificado e inmutable."""
        seal = self._phase3_cryptographic_seal(handoff, decision, interlock_fired, switching_latency_ns)
        phase_chain = ("PHASE1/OBSERVE", "PHASE2/ORIENT", "PHASE3/DECIDE/ACT")
        diagnostics = dict(handoff.diagnostics)
        diagnostics["decision"] = decision
        diagnostics["hardware"] = {
            "interlock_fired": bool(interlock_fired),
            "switching_latency_ns": float(switching_latency_ns),
            "crowbar_spec_ns": float(_CROWBAR_IRAM_LATENCY_NS),
            "within_spec": bool((not interlock_fired) or switching_latency_ns <= _CROWBAR_IRAM_LATENCY_NS),
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

    def phase3_verify_seal(self, handoff: Phase2ReactionHandoff, verdict: ReactionChamberVerdict) -> bool:
        """
        Verificación idempotente: recomputa el sello desde (handoff, decisión,
        hardware) contenidos en el veredicto y lo compara en tiempo constante.
        """
        decision = verdict.diagnostics.get("decision")
        hardware = verdict.diagnostics.get("hardware")
        if not isinstance(decision, dict) or not isinstance(hardware, dict):
            return False
        if handoff.session_sha256 != verdict.session_sha256:
            return False
        recomputed = self._phase3_cryptographic_seal(
            handoff, decision, bool(hardware["interlock_fired"]), float(hardware["switching_latency_ns"]),
        )
        return hmac.compare_digest(recomputed, verdict.cryptographic_seal)

    # ── 3.6  Morfismo terminal ──────────────────────────────────────────────

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2ReactionHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> ReactionChamberVerdict:
        """
        Fase 3.6 — Orquestación completa de la Fase 3 (morfismo terminal).

        Admisión Φ₂→₃ → decisión Heyting → Crowbar → certificado → autoverificación.
        """
        _ = self.phase3_from_phase2(phase2_handoff)
        decision = self.phase3_decide_heyting(
            handoff=phase2_handoff,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
        )
        interlock_fired, latency = self.phase3_actuate_crowbar_interlock(decision["heyting_verdict"])
        verdict = self.phase3_issue_verdict(phase2_handoff, decision, interlock_fired, latency)

        if not self.phase3_verify_seal(phase2_handoff, verdict):  # pragma: no cover
            raise RuntimeError("Autoverificación del sello criptográfico fallida.")

        if verdict.heyting_verdict == HeytingElement.VETOED.name:
            logger.critical("Soberano de la Cámara emitió VETO DURO. Sello: %s",
                            verdict.cryptographic_seal[:16])
        else:
            logger.info("Soberano de la Cámara regulado. Veredicto: %s. Sello: %s",
                        verdict.heyting_verdict, verdict.cryptographic_seal[:16])
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
        API principal — ciclo OODA completo.

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
        phase2_handoff = self.phase2_close_and_open_phase3(phase1_handoff=phase1_handoff)
        return self.phase3_close_loop(
            phase2_handoff=phase2_handoff,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
        )


# ═════════════════════════════════════════════════════════════════════════════
# CERTIFICADO DE IMPORTACIÓN — Ω₃ es un álgebra de Heyting (27 ternas)
# ═════════════════════════════════════════════════════════════════════════════

if not HeytingElement.verify_axioms():  # pragma: no cover
    raise RuntimeError("Ω₃ no satisface los axiomas de Heyting: módulo corrupto.")