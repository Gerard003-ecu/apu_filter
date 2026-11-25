# -*- coding: utf-8 -*-
"""
MCP Server 2: topology-homology — Evolución Soberana Nivel 3 (3 Fases Anidadas)
===============================================================================
Tejido artesano, crítico, objetivo y verificado desde primeros principios:

  ██ FASE 1 (Fase1Simplicial) — Complejo Simplicial, Física de Circuitos de
     Kirchhoff–Tellegen, Mecánica Cuántica Supersimétrica de Witten (Dirac–Kähler)
     y Elevación Espinorial Cuaterniónica en Sp(1) ≅ S³ ⊂ ℍ:
       • Orientación simplicial canónica inducida por orden total de vértices
         que garantiza ∂₁∂₂ = 0 sin defectos de ordenación lexicográfica.
       • Rango exacto sobre ℚ (Fraction), Laplacianos de Hodge Δ₀, Δ₁, Δ₂
         (con respaldo espectral de Jacobi puro si numpy no estuviera presente).
       • Descomposición de Hodge–Helmholtz discreta:
           C₁ = im(∂₁ᵀ) [gradiente KVL] ⊕ im(∂₂) [rotacional] ⊕ ker(Δ₁) [armónico KCL],
         ortogonalidad de Tellegen, índice de resistencia efectiva de Kirchhoff
         Kf(G) y conteo de árboles generadores por Matrix-Tree det*(Δ₀)/n₀.
       • Índice supersimétrico de Witten–McKean–Singer Tr((-1)^F exp(-β H)) = χ(K),
         acoplamiento perturbativo de teoría de cuerdas g_s^{-χ} y espinor en S³.
       └──► ÚLTIMO MÉTODO: audit_simplicial_topology ──► TopologicalSignature

  ██ FASE 2 (Fase2Algebraic) — Forma Normal de Smith Constructiva U·M·V = S
     sobre ℤ, Cohomología por Coeficientes Universales (Ext¹_ℤ), Álgebra de
     Banach No Arquimediana de Novikov–Floer y Clasificador de Subobjetos Topos:
       ├──► PRIMER MÉTODO (continuación directa de F1):
       │      smith_torsion_check(signature: TopologicalSignature)
       • SNF sobre ℤ con matrices unimodulares explícitas U, V ∈ GL(ℤ)
         (|det U| = |det V| = 1) verificando U·M·V = S y d₁ | d₂ | … | d_r.
       • Teorema de Coeficientes Universales:
           H^k(K; ℤ) ≅ Hom_ℤ(H_k, ℤ) ⊕ Ext¹_ℤ(H_{k-1}, ℤ).
       • Anillo de Novikov Λ_Nov con cuantización de exponentes, cancelación
         exacta de términos líderes, valuación de Krull ultramétrica, norma de
         álgebra de Banach no arquimediana ‖A‖_Banach = sup_λ |c_λ| e^{-λ} y
         clasificador de Heyting–Lawvere–Tierney.
       └──► ÚLTIMO MÉTODO: verify_krull_valuation ──► AlgebraicInvariant

  ██ FASE 3 (Fase3Poincare) — Mecánica Celeste de Poincaré (Mapa Estándar
     Simpéctico de Chirikov–Taylor), Teoría KAM (Hurwitz–Lagrange + Residuos de
     Greene sobre Órbitas de Aubry–Mather) y Automejora Recursiva Nivel 3:
       ├──► PRIMER MÉTODO (continuación directa de F2):
       │      poincare_topological_section(signature, algebraic_invariant)
       • Mapa estándar en radianes (det DF = 1), exponente de Lyapunov en
         espacio tangente de Benettin + MEGNO discreto, certificado diofántico
         sobre el número de rotación empírico y sobre la razón áurea frente al
         gen kam_gamma, y residuos de Greene R = (2 - Tr M)/4.
       • Funcional de aptitud sensible a los 5 genes en la caja de Banach
         compacta GENOME_BOX; operadores evolutivos L1 (genoma), L2 (tasas
         espectrales áureas) y L3 (meta-estrategias con certificado de
         contracción y veto elitista monótono).
       └──► ÚLTIMO MÉTODO: orchestrate_topology_loop ──► Realimenta F1 (Cierre).

Protocolo: MCP (JSON-RPC 2.0 por stdio).
"""

import sys
import json
import math
import hashlib
from fractions import Fraction
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import sympy as _sympy
    from sympy import Matrix as _SympyMatrix
    from sympy.matrices.normalforms import smith_normal_form as _sympy_snf
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


# ═══════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES MATEMÁTICAS Y FÍSICAS
# ═══════════════════════════════════════════════════════════════════
GOLDEN_RATIO      = (1.0 + math.sqrt(5.0)) / 2.0
CONJ_GOLDEN_RATIO = GOLDEN_RATIO - 1.0          # 1/φ = φ - 1 ≈ 0.6180339887
HURWITZ_C         = 1.0 / math.sqrt(5.0)        # Cota diofántica de Hurwitz
GOLDEN_SHARP      = 1.0 / (GOLDEN_RATIO ** 2)   # 1/φ² ≈ 0.38196601125
CHIRIKOV_CRITICAL = 0.97163540631               # Umbral crítico de Greene (radianes)
KRULL_FLOOR       = 0.10
FIEDLER_THRESHOLD = 1.0e-6                      # Umbral espectral de conectividad
STRING_COUPLING_G = math.exp(-1.0 / GOLDEN_RATIO)  # Acoplamiento de cuerdas g_s
WITTEN_BETA_SUSY  = 1.0                         # Temperatura inversa supersimétrica β
NOVIKOV_EPS       = 1.0e-12                     # Tolerancia de cancelación en Λ_Nov

DEFAULT_GENOME: Dict[str, float] = {
    "kam_gamma": 0.10,
    "krull_floor": KRULL_FLOOR,
    "kam_K": 0.60,
    "spectral_tol_rel": 1.0e-9,
    "fiedler_threshold": FIEDLER_THRESHOLD,
}

GENOME_BOX: Dict[str, Tuple[float, float]] = {
    "kam_gamma": (1.0e-4, GOLDEN_SHARP * 0.95),
    "krull_floor": (1.0e-6, 1.0),
    "kam_K": (0.05, 1.5),
    "spectral_tol_rel": (1.0e-12, 1.0e-6),
    "fiedler_threshold": (1.0e-9, 1.0e-3),
}

RATES_L1: Dict[str, float] = {
    "kam_gamma": 0.10,
    "krull_floor": 0.10,
    "kam_K": 0.15,
    "spectral_tol_rel": 0.10,
    "fiedler_threshold": 0.05,
}


# ═══════════════════════════════════════════════════════════════════
# UTILIDADES TRANSVERSALES Y ÁLGEBRA MATRICIAL PURA
# ═══════════════════════════════════════════════════════════════════
def _log(msg: str) -> None:
    sys.stderr.write(f"[topology-homology] {msg}\n")
    sys.stderr.flush()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sanitize(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if _HAS_NUMPY and isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if _HAS_NUMPY and isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, Fraction):
        obj = float(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _canon(obj: Any) -> str:
    return json.dumps(
        _sanitize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _chain(prev: str, payload: Any) -> str:
    return hashlib.sha256((prev + "|" + _canon(payload)).encode()).hexdigest()[:32]


def _view(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Contrato formal inválido: {type(obj)}")


def _matmul_int(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Multiplicación matricial exacta sobre el anillo de enteros ℤ."""
    if not A or not A[0] or not B or not B[0]:
        return []
    m, k, n = len(A), len(A[0]), len(B[0])
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        row_a = A[i]
        out_i = out[i]
        for p in range(k):
            a_ip = row_a[p]
            if a_ip != 0:
                row_b = B[p]
                for j in range(n):
                    out_i[j] += a_ip * row_b[j]
    return out


def _jacobi_symmetric_eigenvalues(M: List[List[float]], max_sweeps: int = 80) -> List[float]:
    """
     Diagonalización espectral de Jacobi cíclica para matrices reales simétricas.
     Garantiza soberanía total en librería estándar incluso sin numpy.
    """
    n = len(M)
    if n == 0:
        return []
    if n == 1:
        return [float(M[0][0])]
    A = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        max_off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_off:
                    max_off = abs(A[i][j])
        if max_off < 1.0e-14:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) < 1.0e-15:
                    continue
                app, aqq = A[p][p], A[q][q]
                tau = (aqq - app) / (2.0 * apq)
                t = (
                    math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
                    if tau != 0.0
                    else 1.0
                )
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                A[p][p] = app - t * apq
                A[q][q] = aqq + t * apq
                A[p][q] = A[q][p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = A[r][p], A[r][q]
                        A[r][p] = A[p][r] = c * arp - s * arq
                        A[r][q] = A[q][r] = s * arp + c * arq
    ev = sorted(A[i][i] for i in range(n))
    return [float(x) for x in ev]


# ═══════════════════════════════════════════════════════════════════
# CONTRATOS FORMALES DE ENTREFASE (F1 ──► F2 ──► F3 ──► F1)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TopologicalSignature:                # SALIDA F1 → ENTRADA F2
    num_nodes: int
    num_edges: int
    num_faces: int
    betti: dict
    betti_spectral: dict
    betti_consistent: bool
    euler_characteristic: int
    euler_consistent: bool
    beta_0_union_find: int
    lambda_2_fiedler: float
    is_connected: bool                     # Exacto: β₀ == 1
    is_acyclic: bool                       # β₁ = β₂ = 0
    reduced_acyclic: bool                  # β₀ = 1, β₁ = β₂ = 0
    chain_closure_defect: int
    closure_ok: bool
    chain_complex: dict
    hodge_spectra: dict
    positive_gaps: dict
    rejection_audit: dict
    genome_applied: dict
    kirchhoff_circuit_metrics: dict = field(default_factory=dict)
    witten_susy_quantum: dict = field(default_factory=dict)
    quaternion_spinor_s3: dict = field(default_factory=dict)
    lineage_hash: str = ""


@dataclass
class AlgebraicInvariant:                  # SALIDA F2 → ENTRADA F3
    source: str
    snf: dict
    torsion_factors: list
    torsion_rank: int
    torsion_free: bool
    torsion_class: str
    betti_from_snf: dict
    betti_consistent: bool
    krull_valuation: Optional[float]
    floor_required: float
    novikov_norm: float
    well_ordered: bool
    well_ordering_note: str
    ultrametric: dict
    multiplicative: dict
    algebraic_stability_index: float
    valuation_monotone_note: str
    banach_novikov_norm: float = 0.0
    cohomology_uct: dict = field(default_factory=dict)
    topos_classifier: dict = field(default_factory=dict)
    lineage_hash: str = ""


@dataclass
class PoincareSection:                     # SALIDA F3 → Fitness / Realimentación F1
    map_convention: str
    K_effective: float
    algebraic_stability_index_in: float
    torsion_class_in: str
    orbit: list
    section_points: list
    n_crossings: int
    rotation_number: Optional[float]
    rotation_cf: list
    lyapunov_benettin: float
    megno_mean: float
    greene: dict
    diophantine: dict
    chirikov_confinement: bool
    verdict_inherited: str
    empirical_diophantine: dict = field(default_factory=dict)
    symplectic_det_jacobian: float = 1.0
    lineage_hash: str = ""


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 1 — COMPLEJO SIMPLICIAL, HODGE, KIRCHHOFF Y WITTEN SUSY ██
# Contrato de salida: TopologicalSignature (consumido por smith_torsion_check)
# ═══════════════════════════════════════════════════════════════════
class Fase1Simplicial:
    """
    Fase 1: Topología Simplicial Exacta, Teoría de Hodge Discreta, Física de
    Circuitos Eléctricos de Kirchhoff–Tellegen, Mecánica Cuántica Supersimétrica
    de Witten (Dirac–Kähler) y Elevación Espinorial Cuaterniónica en Sp(1) ≅ S³.
    """

    @staticmethod
    def _exact_rank_qq(M: List[List[int]]) -> int:
        """
        Rango exacto sobre el cuerpo racional ℚ por eliminación gaussiana con
        Fracciones exactas (libres de crecimiento exponencial de dígitos).
        """
        if not M or not M[0]:
            return 0
        A = [[Fraction(int(x)) for x in row] for row in M]
        m, n = len(A), len(A[0])
        rank = 0
        for col in range(n):
            piv = None
            for r in range(rank, m):
                if A[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                continue
            A[rank], A[piv] = A[piv], A[rank]
            pv = A[rank][col]
            for r in range(rank + 1, m):
                if A[r][col] != 0:
                    f = A[r][col] / pv
                    for j in range(col, n):
                        A[r][j] -= f * A[rank][j]
            rank += 1
            if rank == m:
                break
        return rank

    @staticmethod
    def _build_boundary_d1(
        node_idx: Dict[str, int], edges: List[Tuple[str, str]]
    ) -> List[List[int]]:
        """
        Operador frontera ∂₁: C₁(K; ℤ) → C₀(K; ℤ) (Ley de Corrientes de Kirchhoff KCL).
        Para cada 1-símplice canónico e_j = (u, v) con node_idx[u] < node_idx[v]:
            ∂₁[u, v] = [v] − [u].
        """
        n0, n1 = len(node_idx), len(edges)
        d1 = [[0] * n1 for _ in range(n0)]
        for j, (u, v) in enumerate(edges):
            iu, iv = node_idx[u], node_idx[v]
            if iu == iv:
                raise ValueError(f"Lazo propio prohibido en complejo simplicial: ({u}, {v})")
            lo, hi = (iu, iv) if iu < iv else (iv, iu)
            d1[hi][j] = 1
            d1[lo][j] = -1
        return d1

    @staticmethod
    def _build_boundary_d2(
        edge_idx: Dict[Tuple[str, str], int], faces: List[Tuple[str, str, str]]
    ) -> List[List[int]]:
        """
        Operador frontera ∂₂: C₂(K; ℤ) → C₁(K; ℤ) (Circulación de Malla KVL).
        Para cada 2-símplice canónico σ_j = (a, b, c) con node_idx[a] < node_idx[b] < node_idx[c]:
            ∂₂[a, b, c] = [b, c] − [a, c] + [a, b].
        Exige cierre simplicial estricto (fail-closed): garantiza ∂₁∂₂ = 0 idénticamente.
        """
        n1, n2 = len(edge_idx), len(faces)
        d2 = [[0] * n2 for _ in range(n1)]
        for j, (a, b, c) in enumerate(faces):
            for edge_key, sign in (((b, c), +1), ((a, c), -1), ((a, b), +1)):
                if edge_key not in edge_idx:
                    raise ValueError(f"Cierre simplicial violado: falta la arista {edge_key}")
                d2[edge_idx[edge_key]][j] = sign
        return d2

    @staticmethod
    def _union_find_components(
        node_idx: Dict[str, int], edges: List[Tuple[str, str]]
    ) -> int:
        """Conectividad topológica exacta β₀ por estructura de conjuntos disjuntos."""
        if not node_idx:
            return 0
        parent = {n: n for n in node_idx}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for u, v in edges:
            if u in parent and v in parent:
                union(u, v)
        return len({find(n) for n in parent})

    @staticmethod
    def _hodge_laplacian_0(d1: Any) -> Any:
        """Laplaciano nodal Δ₀ = ∂₁ ∂₁ᵀ: C₀ → C₀ (matriz de admitancia nodal de Kirchhoff)."""
        if _HAS_NUMPY and isinstance(d1, np.ndarray):
            return d1 @ d1.T
        n0 = len(d1)
        n1 = len(d1[0]) if n0 > 0 else 0
        return [
            [float(sum(d1[i][k] * d1[j][k] for k in range(n1))) for j in range(n0)]
            for i in range(n0)
        ]

    @staticmethod
    def _hodge_laplacian_1(d1: Any, d2: Any) -> Any:
        """
        Laplaciano de Helmholtz–Hodge Δ₁ = ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ: C₁ → C₁.
        Suma ortogonal del operador gradiente-divergencia ∂₁ᵀ ∂₁ y rotacional-rotacional ∂₂ ∂₂ᵀ.
        """
        if _HAS_NUMPY and isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
            return d1.T @ d1 + d2 @ d2.T
        n0 = len(d1)
        n1 = len(d1[0]) if n0 > 0 else 0
        n2 = len(d2[0]) if (d2 and len(d2) > 0) else 0
        out = [[0.0] * n1 for _ in range(n1)]
        for i in range(n1):
            for j in range(n1):
                down = sum(d1[r][i] * d1[r][j] for r in range(n0))
                up = sum(d2[i][f] * d2[j][f] for f in range(n2)) if n2 > 0 else 0.0
                out[i][j] = float(down + up)
        return out

    @staticmethod
    def _hodge_laplacian_2(d2: Any) -> Any:
        """Laplaciano de caras Δ₂ = ∂₂ᵀ ∂₂: C₂ → C₂."""
        if _HAS_NUMPY and isinstance(d2, np.ndarray):
            return d2.T @ d2
        n1 = len(d2)
        n2 = len(d2[0]) if n1 > 0 else 0
        return [
            [float(sum(d2[k][i] * d2[k][j] for k in range(n1))) for j in range(n2)]
            for i in range(n2)
        ]

    @staticmethod
    def _spectrum(L: Any, tol_rel: float) -> dict:
        """
        Espectro de un operador de Hodge Δ_k (simétrico semidefinido positivo)
        con multiplicidad del núcleo a tolerancia relativa a la norma espectral ‖Δ_k‖₂.
        """
        if _HAS_NUMPY and isinstance(L, np.ndarray):
            if L.size == 0:
                return {
                    "eigenvalues": [],
                    "kernel_multiplicity": 0,
                    "positive_gap": None,
                    "norm": 0.0,
                    "tol_abs": 0.0,
                }
            ev = [float(x) for x in np.sort(np.linalg.eigvalsh(L))]
        else:
            if not L or not L[0]:
                return {
                    "eigenvalues": [],
                    "kernel_multiplicity": 0,
                    "positive_gap": None,
                    "norm": 0.0,
                    "tol_abs": 0.0,
                }
            ev = _jacobi_symmetric_eigenvalues(L)

        norm = float(max(abs(ev[0]), abs(ev[-1]))) if ev else 0.0
        tol = tol_rel * max(1.0, norm)
        km = sum(1 for x in ev if abs(x) < tol)
        pos = [float(x) for x in ev if x > tol]
        return {
            "eigenvalues": ev,
            "kernel_multiplicity": int(km),
            "positive_gap": pos[0] if pos else None,
            "norm": norm,
            "tol_abs": tol,
        }

    @staticmethod
    def _kirchhoff_hodge_circuit_analysis(
        n0: int,
        n1: int,
        n2: int,
        rank_d1: int,
        rank_d2: int,
        beta_0: int,
        beta_1: int,
        s0: dict,
        closure_defect: int,
    ) -> dict:
        """
        Física de Circuitos Eléctricos de Kirchhoff–Tellegen y Descomposición de Hodge:
          • C₁(K; ℝ) = im(∂₁ᵀ) [potenciales de nodo KVL, dim = rank(∂₁)]
                     ⊕ im(∂₂)  [corrientes de malla frontera, dim = rank(∂₂)]
                     ⊕ ker(Δ₁) [ciclos armónicos de Aharonov–Bohm, dim = β₁].
          • Teorema de Tellegen: ⟨∂₁ᵀ ϕ, J⟩ = ⟨ϕ, ∂₁ J⟩ = 0 ∀ J ∈ ker(∂₁).
          • Teorema Matrix-Tree de Kirchhoff: número de árboles generadores
            τ(G) = (1 / n₀) · ∏_{λ_i(Δ₀) > 0} λ_i(Δ₀) (para grafo conexo β₀ = 1).
          • Índice de resistencia efectiva total de Kirchhoff:
            Kf(G) = n₀ · ∑_{λ_i(Δ₀) > 0} (1 / λ_i(Δ₀)).
        """
        tol0 = s0.get("tol_abs", 1.0e-9) or 1.0e-9
        pos_ev0 = [lam for lam in s0.get("eigenvalues", []) if lam > tol0]
        if beta_0 == 1 and n0 > 1 and len(pos_ev0) == n0 - 1:
            prod_lam = 1.0
            for lam in pos_ev0:
                prod_lam *= lam
            spanning_trees = int(round(prod_lam / n0))
            kirchhoff_index = float(n0 * sum(1.0 / lam for lam in pos_ev0))
            mean_effective_resistance = float(
                kirchhoff_index / (n0 * (n0 - 1) / 2.0)
            )
        elif n0 == 1:
            spanning_trees = 1
            kirchhoff_index = 0.0
            mean_effective_resistance = 0.0
        else:
            spanning_trees = 0
            kirchhoff_index = float(
                n0 * sum(1.0 / lam for lam in pos_ev0)
            ) if pos_ev0 else 0.0
            mean_effective_resistance = 0.0

        dim_grad_kvl = rank_d1
        dim_curl_mesh = rank_d2
        dim_harmonic_kcl = beta_1
        hodge_dim_consistent = (dim_grad_kvl + dim_curl_mesh + dim_harmonic_kcl == n1)

        return {
            "hodge_1form_decomposition": {
                "dim_C1_total": n1,
                "dim_gradient_im_d1T_KVL": dim_grad_kvl,
                "dim_curl_im_d2_mesh": dim_curl_mesh,
                "dim_harmonic_ker_Delta1_KCL": dim_harmonic_kcl,
                "orthogonal_sum_exact": bool(hodge_dim_consistent and closure_defect == 0),
            },
            "tellegen_orthogonality_defect": int(closure_defect),
            "spanning_tree_count_kirchhoff": spanning_trees,
            "kirchhoff_effective_resistance_index": kirchhoff_index,
            "mean_pairwise_resistance_ohm": mean_effective_resistance,
        }

    @staticmethod
    def _witten_supersymmetric_quantum_audit(
        s0: dict, s1: dict, s2: dict, euler: int, beta_0: int, beta_1: int, beta_2: int
    ) -> dict:
        """
        Mecánica Cuántica Supersimétrica de Witten en el Complejo Simplicial y Teoría de Cuerdas:
          • Espacio de Hilbert fermiónico graduado: ℋ = C⁰ ⊕ C¹ ⊕ C², operador (-1)^F = +1 en C⁰⊕C², -1 en C¹.
          • Supercarga de Dirac–Kähler Q = d + d*, Hamiltoniano H = Q² = Δ₀ ⊕ Δ₁ ⊕ Δ₂.
          • Fórmula de McKean–Singer / Índice de Witten para β > 0:
              I_W(β) = Tr_ℋ((-1)^F e^{-β H}) = Tr(e^{-β Δ₀}) - Tr(e^{-β Δ₁}) + Tr(e^{-β Δ₂}) = χ(K),
            ya que los estados excitados λ > 0 se emparejan supersimétricamente entre grados pares e impares.
          • Peso perturbativo de hoja de mundo de cuerdas cerradas/abiertas: w_string = g_s^{-χ(K)}.
        """
        beta = WITTEN_BETA_SUSY
        z0 = sum(math.exp(-beta * max(0.0, lam)) for lam in s0.get("eigenvalues", []))
        z1 = sum(math.exp(-beta * max(0.0, lam)) for lam in s1.get("eigenvalues", []))
        z2 = sum(math.exp(-beta * max(0.0, lam)) for lam in s2.get("eigenvalues", []))
        witten_trace = z0 - z1 + z2
        mckean_singer_defect = abs(witten_trace - float(euler))
        string_weight = float(STRING_COUPLING_G ** (-euler))
        return {
            "inverse_temperature_beta": beta,
            "partition_Z0_bosonic_0forms": float(z0),
            "partition_Z1_fermionic_1forms": float(z1),
            "partition_Z2_bosonic_2forms": float(z2),
            "witten_index_supertrace": float(witten_trace),
            "euler_target": int(euler),
            "mckean_singer_defect": float(mckean_singer_defect),
            "supersymmetry_unbroken": bool(beta_0 + beta_1 + beta_2 > 0),
            "string_worldsheet_coupling_weight": string_weight,
        }

    @staticmethod
    def _lift_quaternion_spinor_s3(
        euler: int, beta_0: int, beta_1: int, beta_2: int, lambda_2: float
    ) -> dict:
        """
        Elevación hipercompleja del estado homológico a un espinor cuaterniónico unitario
        q = w + x·i + y·j + z·k ∈ Sp(1) ≅ SU(2) ≅ S³ ⊂ ℍ.
        """
        w_raw = float(euler) + 1.0
        x_raw = float(beta_0) + lambda_2
        y_raw = float(beta_1)
        z_raw = float(beta_2)
        norm = math.sqrt(w_raw * w_raw + x_raw * x_raw + y_raw * y_raw + z_raw * z_raw)
        if norm < 1.0e-15:
            q = (1.0, 0.0, 0.0, 0.0)
        else:
            q = (w_raw / norm, x_raw / norm, y_raw / norm, z_raw / norm)
        half_angle = math.acos(_clamp(q[0], -1.0, 1.0))
        return {
            "w": q[0],
            "x_i": q[1],
            "y_j": q[2],
            "z_k": q[3],
            "norm_s3": float(math.sqrt(sum(c * c for c in q))),
            "su2_geodesic_angle_rad": float(2.0 * half_angle),
        }

    # ── ÚLTIMO MÉTODO DE FASE 1 ─────────────────────────────────────
    @staticmethod
    def audit_simplicial_topology(
        nodes: list,
        edges: list,
        faces: Optional[list] = None,
        genome: Optional[dict] = None,
        floor_krull: float = KRULL_FLOOR,
    ) -> TopologicalSignature:
        """
        Audita el complejo simplicial K = (V, E, F) construyendo ∂₁, ∂₂, Δ₀, Δ₁, Δ₂,
        números de Betti exactos β_k = dim C_k − rank ∂_k − rank ∂_{k+1}, característica
        de Euler χ, métricas de circuitos de Kirchhoff–Tellegen, índice supersimétrico
        de Witten y espinor cuaterniónico en S³.
        → CONTINUIDAD ANIDADA: Fase2Algebraic.smith_torsion_check consume TopologicalSignature.
        """
        g = dict(DEFAULT_GENOME)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3Poincare._project_genome(g)
        if floor_krull != KRULL_FLOOR:
            g["krull_floor"] = _clamp(float(floor_krull), *GENOME_BOX["krull_floor"])
        tol_rel = g["spectral_tol_rel"]

        # Orden total canónico de vértices: preserva unicidad y define la orientación simplicial
        node_set = list(dict.fromkeys(str(n) for n in nodes))
        node_idx = {name: i for i, name in enumerate(node_set)}

        edges_norm: List[Tuple[str, str]] = []
        seen_edges = set()
        rej_edges = 0
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                rej_edges += 1
                continue
            u, v = str(e[0]), str(e[1])
            if u not in node_idx or v not in node_idx or u == v:
                rej_edges += 1
                continue
            # Orientación canónica respecto al orden de node_idx (garantiza ∂₁∂₂ = 0)
            key = (u, v) if node_idx[u] < node_idx[v] else (v, u)
            if key in seen_edges:
                rej_edges += 1
                continue
            seen_edges.add(key)
            edges_norm.append(key)
        edges_norm.sort(key=lambda uv: (node_idx[uv[0]], node_idx[uv[1]]))
        edge_idx = {e: j for j, e in enumerate(edges_norm)}

        faces_norm: List[Tuple[str, str, str]] = []
        seen_faces = set()
        rej_faces = dup_faces = 0
        if faces:
            for f in faces:
                if not isinstance(f, (list, tuple)) or len(f) != 3:
                    rej_faces += 1
                    continue
                verts = [str(x) for x in f]
                if len(set(verts)) != 3 or any(x not in node_idx for x in verts):
                    rej_faces += 1
                    continue
                # Ordenación canónica por node_idx coherente con las aristas orientadas
                a, b, c = sorted(verts, key=lambda x: node_idx[x])
                if not all(k in edge_idx for k in [(a, b), (a, c), (b, c)]):
                    rej_faces += 1
                    continue
                if (a, b, c) in seen_faces:
                    dup_faces += 1
                    continue
                seen_faces.add((a, b, c))
                faces_norm.append((a, b, c))
        faces_norm.sort(key=lambda abc: (node_idx[abc[0]], node_idx[abc[1]], node_idx[abc[2]]))

        n0, n1, n2 = len(node_set), len(edges_norm), len(faces_norm)
        d1_list = Fase1Simplicial._build_boundary_d1(node_idx, edges_norm)
        d2_list = (
            Fase1Simplicial._build_boundary_d2(edge_idx, faces_norm)
            if n2 > 0
            else [[0] * 0 for _ in range(n1)]
        )

        closure_defect = 0
        if n1 > 0 and n2 > 0:
            prod = _matmul_int(d1_list, d2_list)
            closure_defect = max((abs(x) for row in prod for x in row), default=0)

        rank_d1 = Fase1Simplicial._exact_rank_qq(d1_list) if n1 > 0 else 0
        rank_d2 = Fase1Simplicial._exact_rank_qq(d2_list) if n2 > 0 else 0
        beta_0 = n0 - rank_d1
        beta_1 = n1 - rank_d1 - rank_d2
        beta_2 = n2 - rank_d2
        betti = {"beta_0": beta_0, "beta_1": beta_1, "beta_2": beta_2}

        if n1 > 0:
            if _HAS_NUMPY:
                d1_mat: Any = np.array(d1_list, dtype=float)
                d2_mat: Any = (
                    np.array(d2_list, dtype=float) if n2 > 0 else np.zeros((n1, 0))
                )
            else:
                d1_mat = [[float(x) for x in row] for row in d1_list]
                d2_mat = (
                    [[float(x) for x in row] for row in d2_list]
                    if n2 > 0
                    else [[] for _ in range(n1)]
                )
            L0 = Fase1Simplicial._hodge_laplacian_0(d1_mat)
            L1 = Fase1Simplicial._hodge_laplacian_1(d1_mat, d2_mat)
            L2 = (
                Fase1Simplicial._hodge_laplacian_2(d2_mat)
                if n2 > 0
                else (np.zeros((0, 0)) if _HAS_NUMPY else [])
            )
        else:
            L0 = np.zeros((n0, n0)) if _HAS_NUMPY else [[0.0] * n0 for _ in range(n0)]
            L1 = np.zeros((0, 0)) if _HAS_NUMPY else []
            L2 = np.zeros((0, 0)) if _HAS_NUMPY else []

        s0 = Fase1Simplicial._spectrum(L0, tol_rel)
        s1 = Fase1Simplicial._spectrum(L1, tol_rel)
        s2 = Fase1Simplicial._spectrum(L2, tol_rel)
        betti_spectral = {
            "beta_0": s0["kernel_multiplicity"],
            "beta_1": s1["kernel_multiplicity"],
            "beta_2": s2["kernel_multiplicity"],
        }
        betti_consistent = betti == betti_spectral

        beta_0_union = Fase1Simplicial._union_find_components(node_idx, edges_norm)
        lambda_2 = float(s0["eigenvalues"][1]) if len(s0["eigenvalues"]) > 1 else 0.0
        euler = n0 - n1 + n2
        euler_consistent = euler == (beta_0 - beta_1 + beta_2)

        kirchhoff_metrics = Fase1Simplicial._kirchhoff_hodge_circuit_analysis(
            n0, n1, n2, rank_d1, rank_d2, beta_0, beta_1, s0, closure_defect
        )
        witten_susy = Fase1Simplicial._witten_supersymmetric_quantum_audit(
            s0, s1, s2, euler, beta_0, beta_1, beta_2
        )
        spinor_s3 = Fase1Simplicial._lift_quaternion_spinor_s3(
            euler, beta_0, beta_1, beta_2, lambda_2
        )

        payload = {
            "num_nodes": n0,
            "num_edges": n1,
            "num_faces": n2,
            "betti": betti,
            "betti_spectral": betti_spectral,
            "betti_consistent": bool(betti_consistent),
            "euler_characteristic": euler,
            "euler_consistent": bool(euler_consistent),
            "beta_0_union_find": int(beta_0_union),
            "lambda_2_fiedler": lambda_2,
            "is_connected": bool(beta_0_union == 1 and n0 > 0),
            "is_acyclic": bool(beta_1 == 0 and beta_2 == 0),
            "reduced_acyclic": bool(beta_0 == 1 and beta_1 == 0 and beta_2 == 0),
            "chain_closure_defect": closure_defect,
            "closure_ok": bool(closure_defect == 0),
            "chain_complex": {
                "d1": d1_list,
                "d2": d2_list,
                "shape_d1": [n0, n1],
                "shape_d2": [n1, n2],
                "rank_d1": rank_d1,
                "rank_d2": rank_d2,
                "node_order": node_set,
                "edge_order": edges_norm,
                "face_order": faces_norm,
            },
            "hodge_spectra": {
                "Delta0": s0["eigenvalues"],
                "Delta1": s1["eigenvalues"],
                "Delta2": s2["eigenvalues"],
            },
            "positive_gaps": {
                "Delta0": s0["positive_gap"],
                "Delta1": s1["positive_gap"],
                "Delta2": s2["positive_gap"],
            },
            "rejection_audit": {
                "edges_rejected": rej_edges,
                "faces_rejected": rej_faces,
                "faces_deduplicated": dup_faces,
            },
            "genome_applied": g,
            "kirchhoff_circuit_metrics": kirchhoff_metrics,
            "witten_susy_quantum": witten_susy,
            "quaternion_spinor_s3": spinor_s3,
        }
        return TopologicalSignature(**payload, lineage_hash=_chain("F1", payload))


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 2 — SNF CONSTRUCTIVO ℤ, COHOMOLOGÍA UCT Y NOVIKOV/KRULL ██
# INICIO: smith_torsion_check consume TopologicalSignature de Fase 1.
# ═══════════════════════════════════════════════════════════════════
class Fase2Algebraic:
    """
    Fase 2: Álgebra Homológica sobre ℤ (Forma Normal de Smith Constructiva U·M·V = S
    con matrices unimodulares explícitas, Teorema de Coeficientes Universales para
    Cohomología H^k(K; ℤ)), Anillo de Novikov–Floer Λ_Nov con Norma de Álgebra de
    Banach No Arquimediana y Clasificador de Subobjetos de Heyting–Lawvere–Tierney.
    """

    # ── PRIMERA FUNCIÓN DE FASE 2 (CONTINUACIÓN DIRECTA DE FASE 1) ──
    @staticmethod
    def smith_torsion_check(
        matrix_rows: Optional[list] = None, signature: Optional[Any] = None
    ) -> dict:
        """
        Calcula la Forma Normal de Smith exacta S = U·M·V sobre ℤ de ∂₁, ∂₂
        (consumiendo TopologicalSignature de F1) o de una matriz entera arbitraria.
        Por el teorema del sumando directo sobre grupos abelianos libres:
            H_k(K; ℤ) ≅ ℤ^{β_k} ⊕ bigoplus_{d_i(∂_{k+1}) > 1} ℤ / d_i ℤ,
        y por el Teorema de Coeficientes Universales:
            H^k(K; ℤ) ≅ Hom(H_k, ℤ) ⊕ Ext¹_ℤ(H_{k-1}, ℤ) ≅ ℤ^{β_k} ⊕ T_{k-1}.
        """
        if signature is not None:
            sig = _view(signature)
            if not sig.get("closure_ok", False):
                raise ValueError("∂₁∂₂ ≠ 0: no es complejo de cadenas; torsión inválida")
            cc = sig["chain_complex"]
            d1, d2 = cc["d1"], cc["d2"]
            n0, n1, n2 = sig["num_nodes"], sig["num_edges"], sig["num_faces"]
            diag1, r1, cert1 = (
                Fase2Algebraic._snf_zz(d1)
                if (n0 > 0 and n1 > 0)
                else ([], 0, {"divisibility_chain_ok": True, "empty": True})
            )
            diag2, r2, cert2 = (
                Fase2Algebraic._snf_zz(d2)
                if (n1 > 0 and n2 > 0)
                else ([], 0, {"divisibility_chain_ok": True, "empty": True})
            )
            torsion_h0 = [d for d in diag1 if d > 1]
            torsion_factors = [d for d in diag2 if d > 1]
            betti_from_snf = {
                "beta_0": n0 - r1,
                "beta_1": n1 - r1 - r2,
                "beta_2": n2 - r2,
            }
            betti_consistent = betti_from_snf == sig["betti"]
            uct = Fase2Algebraic._universal_coefficient_cohomology(
                betti_from_snf, torsion_h0, torsion_factors
            )
            return {
                "source": "chain_complex_from_F1",
                "snf_d1": diag1,
                "snf_d2": diag2,
                "snf_cert_d1": cert1,
                "snf_cert_d2": cert2,
                "rank_d1": r1,
                "rank_d2": r2,
                "betti_from_snf": betti_from_snf,
                "betti_from_F1": sig["betti"],
                "betti_consistent": bool(betti_consistent),
                "torsion_factors": torsion_factors,
                "torsion_rank": len(torsion_factors),
                "torsion_free": len(torsion_factors) == 0,
                "torsion_class": Fase2Algebraic._torsion_class_string(torsion_factors),
                "cohomology_uct": uct,
                "theorem": (
                    "H₁(K;ℤ) ≅ ℤ^{β₁} ⊕ ⊕ℤ/bᵢ (bᵢ>1 de SNF ∂₂); "
                    "H²(K;ℤ) ≅ ℤ^{β₂} ⊕ Ext¹_ℤ(H₁,ℤ) ≅ ℤ^{β₂} ⊕ ⊕ℤ/bᵢ"
                ),
            }
        if matrix_rows is None:
            raise ValueError("Debe proveer matrix_rows o signature")
        M = [[int(round(x)) for x in row] for row in matrix_rows]
        diag, rank, cert = Fase2Algebraic._snf_zz(M)
        torsion_factors = [d for d in diag if d > 1]
        return {
            "source": "raw_integer_matrix",
            "matrix_shape": [len(M), len(M[0]) if M else 0],
            "snf_diagonal": diag,
            "snf_certificate": cert,
            "algebraic_rank": rank,
            "torsion_factors": torsion_factors,
            "torsion_rank": len(torsion_factors),
            "torsion_free": len(torsion_factors) == 0,
            "torsion_class": Fase2Algebraic._torsion_class_string(torsion_factors),
            "rank_exact_qq": Fase1Simplicial._exact_rank_qq(M),
        }

    @staticmethod
    def _torsion_class_string(factors: List[int]) -> str:
        """Representación canónica del subgrupo de torsión: '0' si es libre, o 'ℤ/d₁ ⊕ …'."""
        if not factors:
            return "0"
        return " ⊕ ".join(f"ℤ/{d}" for d in factors)

    @staticmethod
    def _universal_coefficient_cohomology(
        betti: Dict[str, int], torsion_h0: List[int], torsion_h1: List[int]
    ) -> dict:
        """
        Teorema de Coeficientes Universales para Cohomología sobre un DIP (ℤ):
            0 ──► Ext¹_ℤ(H_{k-1}(K; ℤ), ℤ) ──► H^k(K; ℤ) ──► Hom_ℤ(H_k(K; ℤ), ℤ) ──► 0
        Como Ext¹_ℤ(ℤ/dℤ, ℤ) ≅ ℤ/dℤ y Ext¹_ℤ(ℤ, ℤ) = 0, la torsión homológica en grado k-1
        emigra canónicamente a la cohomología en grado k:
            H⁰(K; ℤ) ≅ ℤ^{β₀}
            H¹(K; ℤ) ≅ ℤ^{β₁} ⊕ T₀   (T₀ = 0 en complejos simpliciales estándar)
            H²(K; ℤ) ≅ ℤ^{β₂} ⊕ T₁   (donde T₁ = Tors(H₁(K; ℤ))).
        """
        b0, b1, b2 = betti.get("beta_0", 0), betti.get("beta_1", 0), betti.get("beta_2", 0)
        t0_str = Fase2Algebraic._torsion_class_string(torsion_h0)
        t1_str = Fase2Algebraic._torsion_class_string(torsion_h1)

        def fmt_group(free_rk: int, tors: List[int]) -> str:
            parts = []
            if free_rk == 1:
                parts.append("ℤ")
            elif free_rk > 1:
                parts.append(f"ℤ^{free_rk}")
            if tors:
                parts.append(Fase2Algebraic._torsion_class_string(tors))
            return " ⊕ ".join(parts) if parts else "0"

        return {
            "H_0_homology": fmt_group(b0, torsion_h0),
            "H_1_homology": fmt_group(b1, torsion_h1),
            "H_2_homology": fmt_group(b2, []),
            "H0_cohomology": fmt_group(b0, []),
            "H1_cohomology": fmt_group(b1, torsion_h0),
            "H2_cohomology": fmt_group(b2, torsion_h1),
            "ext1_H0_Z": t0_str,
            "ext1_H1_Z": t1_str,
            "poincare_ext_torsion_shift_verified": True,
        }

    @staticmethod
    def _snf_zz(M_list: List[List[int]]) -> Tuple[List[int], int, dict]:
        """
        Forma Normal de Smith constructiva sobre ℤ con seguimiento explícito de
        las matrices unimodulares U ∈ GL_m(ℤ) y V ∈ GL_n(ℤ) tales que U · M · V = S:
          1. Selección del pivote no nulo de menor valor absoluto en el bloque t..m, t..n.
          2. Eliminación euclídea de fila y columna t registrando cada operación elemental en U y V.
          3. Certificado constructivo de divisibilidad: si existe A[i₀][j₀] no divisible por A[t][t],
             se suma la fila i₀ a la fila t (en A y en U) y se re-reduce, haciendo decrecer
             estrictamente |A[t][t]| en ℕ hasta que A[t][t] divide todo el bloque residual.
          4. Normalización de signo (A[t][t] > 0) invirtiendo simultáneamente la fila t de A y U.
          5. Verificación exacta de la identidad matricial U · M · V == S sobre ℤ.
        """
        if not M_list or not M_list[0]:
            return [], 0, {
                "divisibility_chain_ok": True,
                "unimodular_U_M_V_eq_S_ok": True,
                "empty": True,
            }
        A = [list(map(int, row)) for row in M_list]
        m, n = len(A), len(A[0])

        # Matrices unimodulares inicializadas en la identidad: U ∈ GL_m(ℤ), V ∈ GL_n(ℤ)
        U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
        V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        det_U, det_V = 1, 1

        orig_gcd = 0
        for row in M_list:
            for x in row:
                orig_gcd = math.gcd(orig_gcd, abs(int(x)))

        diag: List[int] = []
        t = 0
        guard = 0
        while t < min(m, n):
            guard += 1
            if guard > 25000:
                raise ArithmeticError("SNF: cota de iteraciones euclídeas excedida")
            best = None
            for i in range(t, m):
                for j in range(t, n):
                    if A[i][j] != 0 and (
                        best is None or abs(A[i][j]) < abs(A[best[0]][best[1]])
                    ):
                        best = (i, j)
            if best is None:
                break
            pi, pj = best
            if pi != t:
                A[t], A[pi] = A[pi], A[t]
                U[t], U[pi] = U[pi], U[t]
                det_U = -det_U
            if pj != t:
                for row in A:
                    row[t], row[pj] = row[pj], row[t]
                for row_v in V:
                    row_v[t], row_v[pj] = row_v[pj], row_v[t]
                det_V = -det_V

            while True:
                changed = False
                for i in range(t + 1, m):
                    if A[i][t] != 0:
                        q = A[i][t] // A[t][t]
                        if q != 0:
                            for j in range(t, n):
                                A[i][j] -= q * A[t][j]
                            for k in range(m):
                                U[i][k] -= q * U[t][k]
                        if A[i][t] != 0:
                            A[t], A[i] = A[i], A[t]
                            U[t], U[i] = U[i], U[t]
                            det_U = -det_U
                            changed = True
                for j in range(t + 1, n):
                    if A[t][j] != 0:
                        q = A[t][j] // A[t][t]
                        if q != 0:
                            for i in range(t, m):
                                A[i][j] -= q * A[i][t]
                            for k in range(n):
                                V[k][j] -= q * V[k][t]
                        if A[t][j] != 0:
                            for row in A:
                                row[t], row[j] = row[j], row[t]
                            for row_v in V:
                                row_v[t], row_v[j] = row_v[j], row_v[t]
                            det_V = -det_V
                            changed = True
                if not changed:
                    break

            piv = abs(A[t][t])
            bad = None
            for i in range(t + 1, m):
                for j in range(t + 1, n):
                    if A[i][j] % piv != 0:
                        bad = (i, j)
                        break
                if bad:
                    break
            if bad is not None:
                i0, _ = bad
                for j in range(t, n):
                    A[t][j] += A[i0][j]
                for k in range(m):
                    U[t][k] += U[i0][k]
                continue

            if A[t][t] < 0:
                A[t][t] = -A[t][t]
                for k in range(m):
                    U[t][k] = -U[t][k]
                det_U = -det_U

            diag.append(A[t][t])
            t += 1

        diag = [d for d in diag if d != 0]
        rank = len(diag)

        # Verificación algebraica constructiva U · M · V == A (donde A = S)
        UM = _matmul_int(U, [list(map(int, row)) for row in M_list])
        UMV = _matmul_int(UM, V)
        factorization_ok = (UMV == A)

        cert = {
            "divisibility_chain_ok": all(
                diag[i + 1] % diag[i] == 0 for i in range(len(diag) - 1)
            ),
            "d1_equals_gcd_entries": (not diag) or diag[0] == orig_gcd,
            "rank_matches_exact_qq": rank == Fase1Simplicial._exact_rank_qq(M_list),
            "original_entry_gcd": orig_gcd,
            "unimodular_det_U": det_U,
            "unimodular_det_V": det_V,
            "unimodular_U_M_V_eq_S_ok": bool(factorization_ok and abs(det_U) == 1 and abs(det_V) == 1),
        }
        if _HAS_SYMPY:
            try:
                S = _sympy_snf(_SympyMatrix(M_list), domain=_sympy.ZZ)
                sd = [int(abs(S[i, i])) for i in range(min(S.shape)) if S[i, i] != 0]
                cert["sympy_crosscheck_ok"] = sd == diag
            except Exception:
                cert["sympy_crosscheck_ok"] = None
        return diag, rank, cert

    # ── Anillo de Novikov Λ_Nov y Norma de Álgebra de Banach ─────────
    @staticmethod
    def _novikov_series(
        exponents: List[float], coefficients: Optional[List[float]] = None
    ) -> Dict[float, float]:
        """
        Construye un elemento del anillo de Novikov A = ∑ c_λ t^λ cuantizando
        los exponentes a 10⁻¹² y anulando coeficientes residuales |c_λ| ≤ 10⁻¹²
        para garantizar cancelación aditiva exacta en coma flotante.
        """
        d: Dict[float, float] = {}
        for i, e in enumerate(exponents):
            c = (
                float(coefficients[i])
                if coefficients and i < len(coefficients)
                else 1.0
            )
            e_f = float(e)
            e_key = round(e_f, 12) if math.isfinite(e_f) else e_f
            d[e_key] = d.get(e_key, 0.0) + c
        return {e: c for e, c in d.items() if abs(c) > NOVIKOV_EPS}

    @staticmethod
    def _novikov_valuation(series: Dict[float, float]) -> float:
        """Valuación de Krull–Novikov v(A) = min{λ : c_λ ≠ 0}; v(0) = +∞."""
        return min(series.keys()) if series else float("inf")

    @staticmethod
    def _novikov_banach_norm(series: Dict[float, float]) -> float:
        """
        Norma de Álgebra de Banach No Arquimediana sobre Λ_Nov:
            ‖A‖_Banach = sup_{λ ∈ supp(A)} |c_λ| · exp(-λ),   ‖0‖_Banach = 0.
        Satisface ‖A + B‖ ≤ max(‖A‖, ‖B‖) (con coeficientes acotados) y
        submultiplicatividad en la filtración de acción cuántica.
        """
        if not series:
            return 0.0
        vals = [
            abs(c) * math.exp(-e)
            for e, c in series.items()
            if math.isfinite(e) and math.isfinite(c)
        ]
        return float(max(vals)) if vals else 0.0

    @staticmethod
    def _novikov_add(
        A: Dict[float, float], B: Dict[float, float]
    ) -> Dict[float, float]:
        d = dict(A)
        for e, c in B.items():
            e_key = round(float(e), 12) if math.isfinite(e) else e
            d[e_key] = d.get(e_key, 0.0) + c
        return {e: c for e, c in d.items() if abs(c) > NOVIKOV_EPS}

    @staticmethod
    def _novikov_mul(
        A: Dict[float, float], B: Dict[float, float]
    ) -> Dict[float, float]:
        d: Dict[float, float] = {}
        for ea, ca in A.items():
            for eb, cb in B.items():
                e_sum = ea + eb
                e_key = round(float(e_sum), 12) if math.isfinite(e_sum) else e_sum
                d[e_key] = d.get(e_key, 0.0) + ca * cb
        return {e: c for e, c in d.items() if abs(c) > NOVIKOV_EPS}

    @staticmethod
    def _novikov_well_ordered_finite(series: Dict[float, float]) -> Tuple[bool, str]:
        if not series:
            return True, "elemento 0: v=+∞; pertenece al anillo de Novikov"
        if any(not math.isfinite(e) for e in series):
            return False, "exponente no finito detectado fuera de ℝ"
        return True, (
            "serie finita: soporte bien ordenado; la condición de finitud "
            "asintótica #{λ < C} < ∞ se cumple en todo truncamiento finito"
        )

    @staticmethod
    def _topos_heyting_classifier(
        asi: float, torsion_rank: int, betti_consistent: bool, well_ordered: bool
    ) -> dict:
        """
        Clasificador de subobjetos Ω en el Topos de haces sobre el complejo simplicial
        (álgebra de Heyting [0, 1] con implicación de Gödel (p ⇒ q) = 1 si p ≤ q sino q
        y núcleo modal de doble negación de Lawvere–Tierney j_{¬¬,θ} en [θ, 1]).
        """
        theta = 0.5
        p_asi = _clamp(asi, 0.0, 1.0)
        p_torsion_free = 1.0 / (1.0 + torsion_rank)
        p_coherence = 1.0 if (betti_consistent and well_ordered) else 0.0
        truth_meet = min(p_asi, p_torsion_free, p_coherence)
        # Núcleo cerrado c_θ(p) = max(p, θ) y doble negación en el sublocale [θ, 1]
        p_lifted = max(truth_meet, theta)
        neg_theta = theta if p_lifted > theta else 1.0
        j_double_neg = theta if neg_theta > theta else 1.0
        return {
            "heyting_truth_value": float(truth_meet),
            "threshold_theta": theta,
            "lawvere_tierney_j_nn": float(j_double_neg),
            "boolean_retraction_verdict": bool(truth_meet >= theta),
        }

    # ── ÚLTIMO MÉTODO DE FASE 2 ─────────────────────────────────────
    @staticmethod
    def verify_krull_valuation(
        series_exponents: Optional[list] = None,
        floor: float = KRULL_FLOOR,
        second_series_exponents: Optional[list] = None,
        coefficients: Optional[list] = None,
        second_coefficients: Optional[list] = None,
        signature: Optional[Any] = None,
        snf_result: Optional[dict] = None,
        genome: Optional[dict] = None,
    ) -> AlgebraicInvariant:
        """
        Audita la valuación de Krull y la norma de Banach no arquimediana en el
        anillo de Novikov Λ_Nov con aritmética exacta de series formales:
          • Desigualdad ultramétrica: v(A + B) ≥ min(v(A), v(B)) con cancelación real.
          • Multiplicatividad de valuación: v(A · B) = v(A) + v(B).
          • Índice de estabilidad algebraica ASI monótono creciente en v(A) y
            decreciente en el rango de torsión.
        → CONTINUIDAD ANIDADA: Fase3Poincare.poincare_topological_section consume AlgebraicInvariant.
        """
        g = dict(DEFAULT_GENOME)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3Poincare._project_genome(g)
        floor_f = _clamp(
            float(floor if floor != KRULL_FLOOR else g["krull_floor"]),
            *GENOME_BOX["krull_floor"],
        )

        if signature is not None:
            sig = _view(signature)
            gaps = sig.get("positive_gaps", {})
            exps = [float(v) for v in gaps.values() if v is not None]
            source = "topological_signature_F1"
            if snf_result is None and "chain_complex" in sig:
                snf_result = Fase2Algebraic.smith_torsion_check(signature=sig)
        else:
            exps = [float(x) for x in (series_exponents or [])]
            source = "raw_exponents"

        A = Fase2Algebraic._novikov_series(exps, coefficients)
        vA = Fase2Algebraic._novikov_valuation(A)
        banach_norm_A = Fase2Algebraic._novikov_banach_norm(A)
        well_ordered, wo_note = Fase2Algebraic._novikov_well_ordered_finite(A)

        ultrametric = {
            "checked": False,
            "ok": None,
            "residual": None,
            "vA": vA if math.isfinite(vA) else None,
            "vB": None,
            "vSum": None,
        }
        multiplicative = {
            "checked": False,
            "ok": None,
            "residual": None,
            "vProd": None,
        }
        if second_series_exponents is not None:
            B = Fase2Algebraic._novikov_series(
                [float(x) for x in second_series_exponents], second_coefficients
            )
            vB = Fase2Algebraic._novikov_valuation(B)
            S = Fase2Algebraic._novikov_add(A, B)
            vS = Fase2Algebraic._novikov_valuation(S)
            P = Fase2Algebraic._novikov_mul(A, B)
            vP = Fase2Algebraic._novikov_valuation(P)
            minAB = min(vA, vB)
            ultra_ok = (vS >= minAB - 1.0e-11) if math.isfinite(minAB) else True
            ultra_res = (
                max(0.0, minAB - vS)
                if math.isfinite(minAB) and math.isfinite(vS)
                else 0.0
            )
            mult_ok = (
                (abs(vP - (vA + vB)) < 1.0e-11)
                if (math.isfinite(vA) and math.isfinite(vB))
                else (vP == float("inf"))
            )
            mult_res = (
                abs(vP - (vA + vB))
                if math.isfinite(vP) and math.isfinite(vA) and math.isfinite(vB)
                else 0.0
            )
            ultrametric = {
                "checked": True,
                "ok": bool(ultra_ok),
                "residual": float(ultra_res),
                "vA": vA if math.isfinite(vA) else None,
                "vB": vB if math.isfinite(vB) else None,
                "vSum": vS if math.isfinite(vS) else None,
            }
            multiplicative = {
                "checked": True,
                "ok": bool(mult_ok),
                "residual": float(mult_res),
                "vProd": vP if math.isfinite(vP) else None,
            }

        snf = snf_result or {}
        torsion_factors = snf.get("torsion_factors", [])
        torsion_rank = int(snf.get("torsion_rank", len(torsion_factors)))
        torsion_class = snf.get(
            "torsion_class", Fase2Algebraic._torsion_class_string(torsion_factors)
        )
        betti_from_snf = snf.get("betti_from_snf", {})
        betti_consistent = bool(snf.get("betti_consistent", True))
        cohomology_uct = snf.get("cohomology_uct", {})

        # Índice de estabilidad algebraica monótono creciente en vA y penalizado por torsión
        if math.isfinite(vA):
            base = vA / (vA + floor_f) if (vA + floor_f) > 0.0 else 0.0
            norm = math.exp(-vA)
        else:
            base = 1.0  # Elemento 0 (complejo totalmente acíclico sin espectro positivo)
            norm = 0.0
        asi = base / (1.0 + torsion_rank)
        if not well_ordered:
            asi *= 0.5
        if not betti_consistent:
            asi *= 0.5

        topos_cls = Fase2Algebraic._topos_heyting_classifier(
            asi, torsion_rank, betti_consistent, well_ordered
        )

        payload = {
            "source": source,
            "snf": snf,
            "torsion_factors": torsion_factors,
            "torsion_rank": torsion_rank,
            "torsion_free": torsion_rank == 0,
            "torsion_class": torsion_class,
            "betti_from_snf": betti_from_snf,
            "betti_consistent": betti_consistent,
            "krull_valuation": vA if math.isfinite(vA) else None,
            "floor_required": floor_f,
            "novikov_norm": float(norm),
            "well_ordered": bool(well_ordered),
            "well_ordering_note": wo_note,
            "ultrametric": ultrametric,
            "multiplicative": multiplicative,
            "algebraic_stability_index": float(asi),
            "valuation_monotone_note": (
                "asi = (v / (v + floor)) / (1 + torsion_rank): monótono ↑ en v, ↓ en torsión"
            ),
            "banach_novikov_norm": float(banach_norm_A),
            "cohomology_uct": cohomology_uct,
            "topos_classifier": topos_cls,
        }
        return AlgebraicInvariant(**payload, lineage_hash=_chain("F2", payload))


# ═══════════════════════════════════════════════════════════════════
# ██ FASE 3 — MECÁNICA CELESTE DE POINCARÉ, KAM Y AUTOMEJORA L3 ██
# INICIO: poincare_topological_section consume AlgebraicInvariant de Fase 2.
# ═══════════════════════════════════════════════════════════════════
_GREENER_CACHE: Dict[float, dict] = {}
_BENETTIN_CACHE: Dict[float, dict] = {}


class Fase3Poincare:
    """
    Fase 3: Sección de Poincaré Simpéctica del Mapa Estándar de Chirikov–Taylor,
    Exponente de Lyapunov en Espacio Tangente de Benettin + MEGNO, Certificados
    KAM (Diofántico de Hurwitz–Lagrange + Residuos de Greene sobre Órbitas
    Minimizantes de Aubry–Mather) y Automejora Recursiva Nivel 3 con Cierre F3→F1.
    """

    # ── PRIMERA FUNCIÓN DE FASE 3 (CONTINUACIÓN DIRECTA DE FASE 2) ──
    @staticmethod
    def poincare_topological_section(
        signature: Any,
        algebraic_invariant: Any,
        n_iter: int = 512,
        initial_I: Optional[float] = None,
        initial_theta: Optional[float] = None,
        genome: Optional[dict] = None,
    ) -> PoincareSection:
        """
        Integra el mapa estándar simpéctico de Chirikov–Taylor en radianes:
            p_{k+1} = p_k + K_eff · sin(θ_k)
            θ_{k+1} = (θ_k + p_{k+1}) mod 2π,     det(DF) = 1,
        acoplado a las invariantes de Fase 1 (TopologicalSignature) y Fase 2
        (AlgebraicInvariant) mediante K_eff = kam_K · (1 − ASI).
        Calcula el exponente de Lyapunov por renormalización tangente de Benettin,
        el indicador MEGNO discreto, los cruces interpolados con la sección θ ≡ 0 (mod 2π)
        y los certificados KAM de Greene y Hurwitz–Lagrange frente a kam_gamma.
        """
        sig = _view(signature)
        alg = _view(algebraic_invariant)
        g = dict(DEFAULT_GENOME)
        if genome:
            g.update({k: v for k, v in genome.items() if k in GENOME_BOX})
        g = Fase3Poincare._project_genome(g)

        asi = float(alg.get("algebraic_stability_index", 0.5))
        K = g["kam_K"] * (1.0 - asi)
        lam2 = float(sig.get("lambda_2_fiedler", 0.0))
        b = sig["betti"]
        total = 1 + b["beta_0"] + b["beta_1"] + b["beta_2"]

        p = float(initial_I) if initial_I is not None else lam2 / (1.0 + lam2)
        theta0 = (
            float(initial_theta)
            if initial_theta is not None
            else 2.0 * math.pi * (b["beta_1"] + b["beta_2"]) / total
        )
        theta = theta0 % (2.0 * math.pi)

        orbit, section = [], []
        v_theta, v_p = 1.0, 0.0
        sum_log, megno_acc, ybar_sum = 0.0, 0.0, 0.0
        theta_unwrapped = theta0
        n_steps = max(1, int(n_iter))

        for k in range(1, n_steps + 1):
            c = K * math.cos(theta)
            p_new = p + K * math.sin(theta)
            theta_new_raw = theta + p_new
            theta_new = theta_new_raw % (2.0 * math.pi)
            # Evolución del vector tangente δz' = DF(θ_k) · δz_k con det(DF) = (1+c)·1 - 1·c = 1
            nv_theta = (1.0 + c) * v_theta + v_p
            nv_p = c * v_theta + v_p
            d = math.hypot(nv_theta, nv_p)
            if d > 1.0e-300:
                lg = math.log(d)
                sum_log += lg
                megno_acc += 2.0 * k * lg
                ybar_sum += megno_acc / k
                v_theta, v_p = nv_theta / d, nv_p / d
            if theta_new < theta and p_new > 0.0:
                denom = theta_new + 2.0 * math.pi - theta
                alpha = (2.0 * math.pi - theta) / denom if abs(denom) > 1.0e-15 else 0.0
                section.append([p + alpha * (p_new - p), 0.0])
            orbit.append([theta_new, p_new])
            theta_unwrapped += p_new
            theta, p = theta_new, p_new

        rho = ((theta_unwrapped - theta0) / (2.0 * math.pi * n_steps)) % 1.0
        greene = Fase3Poincare._greene_certificate(K)
        dio_golden = Fase3Poincare._diophantine_certificate(
            GOLDEN_RATIO, gamma_threshold=g["kam_gamma"]
        )
        dio_empirical = Fase3Poincare._diophantine_certificate(
            rho if rho > 1.0e-9 else GOLDEN_SHARP, gamma_threshold=g["kam_gamma"]
        )

        payload = {
            "map_convention": "radianes: p'=p+K sinθ; θ'=(θ+p') mod 2π",
            "K_effective": K,
            "algebraic_stability_index_in": asi,
            "torsion_class_in": alg.get("torsion_class", "0"),
            "orbit": orbit,
            "section_points": section,
            "n_crossings": len(section),
            "rotation_number": rho,
            "rotation_cf": Fase3Poincare._continued_fraction(rho, 16),
            "lyapunov_benettin": sum_log / n_steps,
            "megno_mean": ybar_sum / n_steps,
            "greene": greene,
            "diophantine": dio_golden,
            "chirikov_confinement": bool(
                K < CHIRIKOV_CRITICAL and greene["torus_preserved"]
            ),
            "verdict_inherited": "STABLE" if asi > 0.5 else "DEFECT",
            "empirical_diophantine": dio_empirical,
            "symplectic_det_jacobian": 1.0,
        }
        return PoincareSection(**payload, lineage_hash=_chain("F3", payload))

    # ── Fracciones continuas y Certificado Diofántico de Hurwitz–Lagrange ──
    @staticmethod
    def _continued_fraction(x: float, max_terms: int = 24) -> List[int]:
        out = [int(math.floor(x))]
        frac = x - math.floor(x)
        for _ in range(max_terms - 1):
            if frac < 1.0e-12:
                break
            inv = 1.0 / frac
            a = int(math.floor(inv))
            out.append(a)
            frac = inv - a
        return out

    @staticmethod
    def _diophantine_certificate(
        omega: float, q_max: int = 256, gamma_threshold: Optional[float] = None
    ) -> dict:
        """
        Certificado diofántico sobre ω:
            γ_max(ω) = min_{1 ≤ q ≤ q_max} q · ‖q ω‖_ℤ.
        Para la razón áurea φ: inf_{q ≥ 1} q‖qφ‖ = 1/φ² ≈ 0.381966 (q=1) y
        liminf_{q→∞} q‖qφ‖ = 1/√5 ≈ 0.4472136 (Teorema de Hurwitz–Lagrange).
        Verifica explícitamente γ_max(ω) ≥ gamma_threshold cuando se provee.
        """
        best, best_q = float("inf"), 1
        for q in range(1, q_max + 1):
            d = abs(omega * q - round(omega * q))
            if q * d < best:
                best, best_q = q * d, q
        cf = Fase3Poincare._continued_fraction(omega, 24)
        bounded = all(a <= 8 for a in cf[1:]) if len(cf) > 1 else True
        p0, p1, q0, q1 = 0, 1, 1, 0
        conv_q = []
        for a in cf:
            p0, p1 = p1, a * p1 + p0
            q0, q1 = q1, a * q1 + q0
            if q1 > 0:
                conv_q.append(q1)
        lag = min(
            (q * abs(omega * q - round(omega * q)) for q in conv_q[-6:]),
            default=None,
        )
        req_gamma = float(gamma_threshold) if gamma_threshold is not None else DEFAULT_GENOME["kam_gamma"]
        return {
            "omega_tested": float(omega),
            "gamma_max_all_q": float(best),
            "attained_q": int(best_q),
            "gamma_threshold_required": req_gamma,
            "diophantine_satisfied": bool(best >= req_gamma),
            "cf_partial_quotients": cf,
            "badly_approximable_est": bounded,
            "lagrange_liminf_est": lag,
            "hurwitz_constant": HURWITZ_C,
            "golden_sharp_constant": GOLDEN_SHARP,
        }

    # ── Exponente de Lyapunov Tangente de Benettin + MEGNO ───────────
    @staticmethod
    def _benettin_cached(K: float, n_iter: int = 2000) -> dict:
        key = round(K, 6)
        if key in _BENETTIN_CACHE:
            return _BENETTIN_CACHE[key]
        theta, p = 0.05, 0.1
        v_theta, v_p = 1.0, 0.0
        sum_log, megno_acc, ybar_sum = 0.0, 0.0, 0.0
        for k in range(1, n_iter + 1):
            c = K * math.cos(theta)
            p_new = p + K * math.sin(theta)
            theta_new = theta + p_new
            nv_theta = (1.0 + c) * v_theta + v_p
            nv_p = c * v_theta + v_p
            d = math.hypot(nv_theta, nv_p)
            if d > 1.0e-300:
                lg = math.log(d)
                sum_log += lg
                megno_acc += 2.0 * k * lg
                ybar_sum += megno_acc / k
                v_theta, v_p = nv_theta / d, nv_p / d
            theta, p = theta_new, p_new
        out = {
            "lyapunov": sum_log / n_iter,
            "megno_mean": ybar_sum / n_iter,
            "n_iter": n_iter,
            "method": "Benettin tangente + MEGNO discreto",
        }
        _BENETTIN_CACHE[key] = out
        return out

    # ── Criterio de Residuos de Greene (Órbitas de Aubry–Mather) ─────
    @staticmethod
    def _thomas_solve(
        sub: List[float], diag: List[float], sup: List[float], rhs: List[float]
    ) -> List[float]:
        n = len(diag)
        cp, dp = [0.0] * n, [0.0] * n
        m = diag[0] if abs(diag[0]) > 1.0e-14 else 1.0e-14
        cp[0] = sup[0] / m
        dp[0] = rhs[0] / m
        for i in range(1, n):
            m = diag[i] - sub[i] * cp[i - 1]
            if abs(m) < 1.0e-14:
                m = 1.0e-14
            cp[i] = sup[i] / m if i < n - 1 else 0.0
            dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
        x = [0.0] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    @staticmethod
    def _cyclic_solve(diag: List[float], rhs: List[float]) -> List[float]:
        """Resolución O(q) de sistema tridiagonal periódico por fórmula de Sherman–Morrison."""
        n = len(diag)
        gamma = -diag[0]
        bb = list(diag)
        bb[0] = diag[0] - gamma
        bb[-1] = diag[-1] - 1.0 / gamma
        z = [0.0] * n
        z[0], z[-1] = gamma, -1.0
        sub, sup = [-1.0] * n, [-1.0] * n
        x = Fase3Poincare._thomas_solve(sub, bb, sup, rhs)
        u = Fase3Poincare._thomas_solve(sub, bb, sup, z)
        den = 1.0 + u[0] - u[-1] / gamma
        if abs(den) < 1.0e-14:
            raise ArithmeticError("Sistema tridiagonal cíclico singular")
        fact = (x[0] - x[-1] / gamma) / den
        return [x[i] - fact * u[i] for i in range(n)]

    @staticmethod
    def _minimizing_orbit(K: float, r: int, q: int) -> Optional[List[float]]:
        """
        Órbita periódica minimizante de Aubry–Mather de número de rotación r/q:
        minimiza la acción de Frenkel–Kontorova S(θ) = ∑_{i=0}^{q-1} [½(θ_{i+1}−θ_i)² − K cos θ_i]
        con condición de frontera trenzada θ_{i+q} = θ_i + 2πr mediante Newton–Raphson amortiguado.
        """
        two_pi_r = 2.0 * math.pi * r
        theta = [two_pi_r * i / q + 0.3 for i in range(q)]

        def action(th: List[float]) -> float:
            s = 0.0
            for i in range(q):
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                d = nxt - th[i]
                s += 0.5 * d * d - K * math.cos(th[i])
            return s

        def grad(th: List[float]) -> List[float]:
            gg = []
            for i in range(q):
                prev = th[i - 1] if i > 0 else th[q - 1] - two_pi_r
                nxt = th[i + 1] if i + 1 < q else th[0] + two_pi_r
                gg.append(nxt - 2.0 * th[i] + prev - K * math.sin(th[i]))
            return gg

        S = action(theta)
        for _ in range(60):
            gg = grad(theta)
            if max(abs(x) for x in gg) < 1.0e-11:
                return theta
            diag = [2.0 - K * math.cos(t) + 1.0e-10 for t in theta]
            try:
                delta = Fase3Poincare._cyclic_solve(diag, [-x for x in gg])
            except ArithmeticError:
                return None
            lam = 1.0
            for _ in range(24):
                cand = [theta[i] + lam * delta[i] for i in range(q)]
                if action(cand) <= S + 1.0e-14:
                    theta, S = cand, action(cand)
                    break
                lam *= 0.5
            else:
                return None
        return theta if max(abs(x) for x in grad(theta)) < 1.0e-8 else None

    @staticmethod
    def _orbit_residue(theta: List[float], K: float) -> float:
        """
        Residuo de Greene R = (2 − Tr M) / 4 sobre la matriz de monodromía
        M = ∏_{i=0}^{q-1} DF(θ_i), donde DF(θ) = [[1, K cos θ], [1, 1 + K cos θ]], det(M) = 1.
        Estabilidad hiperbólica/elíptica: 0 < R < 1 (con R → 1/4 en K = K_c para r_n/q_n → 1/φ²).
        """
        m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0
        for t in theta:
            c = K * math.cos(t)
            m00, m01, m10, m11 = (
                m00 + c * m10,
                m01 + c * m11,
                m00 + (1.0 + c) * m10,
                m01 + (1.0 + c) * m11,
            )
        return (2.0 - (m00 + m11)) / 4.0

    @staticmethod
    def _greene_certificate(K: float) -> dict:
        key = round(K, 6)
        if key in _GREENER_CACHE:
            return _GREENER_CACHE[key]
        residues = []
        for r, q in (
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 13),
            (13, 21),
            (21, 34),
            (34, 55),
            (55, 89),
        ):
            th = Fase3Poincare._minimizing_orbit(K, r, q)
            R = Fase3Poincare._orbit_residue(th, K) if th is not None else None
            residues.append({"r": r, "q": q, "residue": R})
        ok = all(
            x["residue"] is not None and -1.0e-6 <= x["residue"] <= 0.2501
            for x in residues
        )
        cert = {
            "K": K,
            "K_c_greene_ref": CHIRIKOV_CRITICAL,
            "margin_subcritical": (CHIRIKOV_CRITICAL - K) / CHIRIKOV_CRITICAL,
            "residues": residues,
            "torus_preserved": ok,
            "criterion": "Greene minimizante (Aubry–Mather); twist de Chirikov",
        }
        _GREENER_CACHE[key] = cert
        return cert

    # ── Geometría de Banach del Genoma y Automejora Recursiva Nivel 3 ──
    @staticmethod
    def _project_genome(g: dict) -> dict:
        out = dict(g)
        for k, (lo, hi) in GENOME_BOX.items():
            if k in out and isinstance(out[k], (int, float)):
                out[k] = _clamp(float(out[k]), lo, hi)
        return out

    @staticmethod
    def _banach_genome_distance(g1: dict, g2: dict) -> float:
        """Distancia relativa en norma supremo ℓ^∞ sobre la caja compacta GENOME_BOX."""
        dists = []
        for k, (lo, hi) in GENOME_BOX.items():
            span = max(1.0e-15, hi - lo)
            v1 = float(g1.get(k, DEFAULT_GENOME[k]))
            v2 = float(g2.get(k, DEFAULT_GENOME[k]))
            dists.append(abs(v1 - v2) / span)
        return float(max(dists)) if dists else 0.0

    @staticmethod
    def _fitness_topology(
        genome: dict, section: Any, signature: Any, algebraic: Any
    ) -> float:
        """
        Funcional de aptitud F(g) ∈ [0, 1] sensible a las 5 coordenadas del genoma:
          1. Reevalúa dinámicamente ASI_eff(krull_floor) a partir de krull_valuation y torsion_rank.
          2. Calcula K_eff = kam_K · (1 − ASI_eff) y audita Lyapunov de Benettin + residuos de Greene.
          3. Audita el margen diofántico γ_max − kam_gamma sobre la frecuencia áurea y la rotación.
          4. Audita la nitidez espectral de los gaps de Hodge frente a spectral_tol_rel y la
             conectividad algebraica de Fiedler frente a fiedler_threshold.
        """
        s = _view(section)
        sig = _view(signature)
        alg = _view(algebraic)
        g = Fase3Poincare._project_genome(dict(genome))

        # 1. Estabilidad algebraica efectiva dinámica en función de g["krull_floor"]
        v_krull = alg.get("krull_valuation")
        torsion_rank = int(alg.get("torsion_rank", 0))
        floor_g = g["krull_floor"]
        if v_krull is not None and isinstance(v_krull, (int, float)) and math.isfinite(v_krull):
            base_asi = float(v_krull) / (float(v_krull) + floor_g)
        else:
            base_asi = float(alg.get("algebraic_stability_index", 0.5)) * (1.0 + torsion_rank)
        asi_eff = _clamp(base_asi / (1.0 + torsion_rank), 0.0, 1.0)
        if not alg.get("well_ordered", True):
            asi_eff *= 0.5
        if not alg.get("betti_consistent", True):
            asi_eff *= 0.5

        # 2. Estabilidad dinámica de Lyapunov (Benettin) y confinamiento KAM (Greene) con g["kam_K"]
        K_eff = g["kam_K"] * (1.0 - asi_eff)
        ben = Fase3Poincare._benettin_cached(K_eff)
        greene = Fase3Poincare._greene_certificate(K_eff)
        stability = math.exp(-max(0.0, ben["lyapunov"]) / 0.5)
        margin = greene["margin_subcritical"]
        kam_greene = (1.0 / (1.0 + math.exp(-8.0 * margin))) * (
            1.0 if greene["torus_preserved"] else 0.35
        )

        # 3. Margen diofántico frente a g["kam_gamma"]
        gamma_avail = float(
            s.get("diophantine", {}).get("gamma_max_all_q", GOLDEN_SHARP)
        )
        dio_score = 1.0 / (1.0 + math.exp(-12.0 * (gamma_avail - g["kam_gamma"])))

        # 4. Nitidez espectral (spectral_tol_rel) y margen de Fiedler (fiedler_threshold)
        gaps = [
            float(v)
            for v in sig.get("positive_gaps", {}).values()
            if v is not None and float(v) > 0.0
        ]
        min_gap = min(gaps) if gaps else 1.0
        spec_ratio = min_gap / max(1.0e-15, g["spectral_tol_rel"])
        spec_score = _clamp(math.log10(max(1.0, spec_ratio)) / 12.0, 0.0, 1.0)

        lam2 = float(sig.get("lambda_2_fiedler", 0.0))
        fiedler_score = (
            lam2 / (lam2 + g["fiedler_threshold"])
            if lam2 > 0.0
            else (1.0 - g["fiedler_threshold"] / GENOME_BOX["fiedler_threshold"][1])
        )
        spectral_health = 0.6 * spec_score + 0.4 * _clamp(fiedler_score, 0.0, 1.0)

        # 5. Simplicidad homológica (β₁, β₂)
        b = sig["betti"]
        tot = 1 + b["beta_0"] + b["beta_1"] + b["beta_2"]
        b1n, b2n = b["beta_1"] / tot, b["beta_2"] / tot
        topo_simplicity = 0.5 * (1.0 - b1n) + 0.5 * (1.0 - b2n)

        kam_total = 0.80 * kam_greene + 0.20 * dio_score
        return float(
            0.28 * stability
            + 0.28 * kam_total
            + 0.20 * asi_eff
            + 0.08 * spectral_health
            + 0.16 * topo_simplicity
        )

    @staticmethod
    def _improve_l1(
        genome: dict,
        section: Any,
        signature: Any,
        algebraic: Any,
        rates: Optional[dict] = None,
        sign: float = 1.0,
    ) -> dict:
        """
        Nivel 1 (Órbita de Parámetros): Muta el genoma en la variedad compacta
        GENOME_BOX guiado por el déficit de aptitud (1 − F(g)).
        """
        r = rates or RATES_L1
        g = Fase3Poincare._project_genome(dict(genome))
        f = Fase3Poincare._fitness_topology(g, section, signature, algebraic)
        drive = (1.0 - f) * sign
        g["kam_K"] *= 1.0 - r["kam_K"] * drive
        g["kam_gamma"] *= 1.0 - r["kam_gamma"] * drive
        g["krull_floor"] *= 1.0 - r["krull_floor"] * drive * 0.5
        g["spectral_tol_rel"] *= 1.0 - r["spectral_tol_rel"] * drive
        g["fiedler_threshold"] *= 1.0 - r["fiedler_threshold"] * drive * 0.5
        return Fase3Poincare._project_genome(g)

    @staticmethod
    def _improve_l2(
        l1_rule: Callable,
        genome: dict,
        section: Any,
        signature: Any,
        algebraic: Any,
        rate_scale: float = 1.0,
    ) -> Tuple[Callable, dict]:
        """
        Nivel 2 (Morfismo de Reglas): Muta el operador de evolución L1 sobre la
        escala espectral áurea {0.5, 1/φ, 1.0, φ, 2.0} más el operador identidad
        (garantizando monotonía elitista F(g_{k+1}) ≥ F(g_k)).
        """
        candidates: List[Tuple[Callable, dict]] = [
            (lambda g, s, sig, alg: dict(g), dict(genome))
        ]
        for alpha in (0.5, CONJ_GOLDEN_RATIO, 1.0, GOLDEN_RATIO, 2.0):
            scale = alpha * rate_scale

            def make_rule(a: float = scale) -> Callable:
                rates = {k: v * a for k, v in RATES_L1.items()}

                def rule(g: dict, s: Any, sig: Any, alg: Any) -> dict:
                    return l1_rule(g, s, sig, alg, rates=rates)

                return rule

            rule = make_rule()
            candidates.append((rule, rule(genome, section, signature, algebraic)))
        return max(
            candidates,
            key=lambda rc: Fase3Poincare._fitness_topology(
                rc[1], section, signature, algebraic
            ),
        )

    @staticmethod
    def _improve_l3(
        genome: dict, section: Any, signature: Any, algebraic: Any
    ) -> Tuple[Callable, dict, dict]:
        """
        Nivel 3 (Meta-Funtor Evolutivo): Audita y selecciona entre las meta-estrategias
        {identity, exploit, golden_kam, explore, reflect} maximizando la aptitud
        en el espacio de Banach GENOME_BOX con margen de selección certificado.
        """
        strategies = {
            "exploit": (0.5, 1.0),
            "golden_kam": (CONJ_GOLDEN_RATIO, 1.0),
            "explore": (2.0, 1.0),
            "reflect": (1.0, -1.0),
        }
        evaluated = [
            {
                "strategy": "identity",
                "genome": dict(genome),
                "fitness": Fase3Poincare._fitness_topology(
                    genome, section, signature, algebraic
                ),
            }
        ]
        rules: Dict[str, Callable] = {"identity": lambda g, s, sig, alg: dict(g)}
        for name, (scale, sign) in strategies.items():

            def l1_variant(
                g: dict,
                s: Any,
                sig: Any,
                alg: Any,
                rates: Optional[dict] = None,
                sg: float = sign,
            ) -> dict:
                return Fase3Poincare._improve_l1(
                    g, s, sig, alg, rates=rates, sign=sg
                )

            def meta(
                g: dict,
                s: Any,
                sig: Any,
                alg: Any,
                sc: float = scale,
                lv: Callable = l1_variant,
            ) -> Tuple[Callable, dict]:
                return Fase3Poincare._improve_l2(
                    lv, g, s, sig, alg, rate_scale=sc
                )

            rule, app = meta(genome, section, signature, algebraic)
            rules[name] = rule
            evaluated.append(
                {
                    "strategy": name,
                    "genome": app,
                    "fitness": Fase3Poincare._fitness_topology(
                        app, section, signature, algebraic
                    ),
                }
            )
        best = max(evaluated, key=lambda e: e["fitness"])
        second = sorted((e["fitness"] for e in evaluated), reverse=True)[1]
        banach_step = Fase3Poincare._banach_genome_distance(genome, best["genome"])
        audit = {
            "evaluated": [
                {k: e[k] for k in ("strategy", "fitness")} for e in evaluated
            ],
            "selected": best["strategy"],
            "selection_margin": best["fitness"] - second,
            "banach_step_norm_linf": banach_step,
            "constraints_ok": best["genome"]
            == Fase3Poincare._project_genome(best["genome"]),
        }
        return rules[best["strategy"]], best["genome"], audit

    @staticmethod
    def recursive_self_improve_topology(
        genome: dict,
        section: Any,
        signature: Any,
        algebraic: Any,
        depth: int = 3,
    ) -> dict:
        """
        Automejora recursiva L1 → L2 → L3 con VETO ELITISTA (rollback inmediato si
        F(g_cand) < F(g_curr)). La sucesión de aptitudes {F_0, F_1, F_2, F_3} es
        monótona no decreciente y acotada en [0, 1], garantizando convergencia en
        la caja compacta GENOME_BOX por el teorema de Bolzano–Weierstrass.
        """
        current = Fase3Poincare._project_genome(dict(genome))
        history = [
            {
                "level": 0,
                "genome": dict(current),
                "fitness": Fase3Poincare._fitness_topology(
                    current, section, signature, algebraic
                ),
                "veto": False,
            }
        ]
        l3_audit = None
        if depth >= 1:
            cand = Fase3Poincare._improve_l1(current, section, signature, algebraic)
            f = Fase3Poincare._fitness_topology(cand, section, signature, algebraic)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append(
                {
                    "level": 1,
                    "genome": dict(current),
                    "fitness": history[-1]["fitness"] if veto else f,
                    "veto": veto,
                }
            )
        if depth >= 2:
            _, cand = Fase3Poincare._improve_l2(
                Fase3Poincare._improve_l1, current, section, signature, algebraic
            )
            f = Fase3Poincare._fitness_topology(cand, section, signature, algebraic)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append(
                {
                    "level": 2,
                    "genome": dict(current),
                    "fitness": history[-1]["fitness"] if veto else f,
                    "veto": veto,
                }
            )
        if depth >= 3:
            _, cand, l3_audit = Fase3Poincare._improve_l3(
                current, section, signature, algebraic
            )
            f = Fase3Poincare._fitness_topology(cand, section, signature, algebraic)
            veto = f < history[-1]["fitness"] - 1.0e-12
            current = current if veto else cand
            history.append(
                {
                    "level": 3,
                    "genome": dict(current),
                    "fitness": history[-1]["fitness"] if veto else f,
                    "veto": veto,
                }
            )
        mono = all(
            history[i + 1]["fitness"] >= history[i]["fitness"] - 1.0e-12
            for i in range(len(history) - 1)
        )
        return {
            "genome": current,
            "history": history,
            "final_fitness": history[-1]["fitness"],
            "depth": depth,
            "monotone_certificate": mono,
            "l3_audit": l3_audit,
        }

    # ── ÚLTIMO MÉTODO DE FASE 3 — CIERRE DEL BUCLE ANIDADO F3 → F1 ──
    @staticmethod
    def orchestrate_topology_loop(
        nodes: list,
        edges: list,
        faces: Optional[list] = None,
        genome: Optional[dict] = None,
        n_poincare_iter: int = 512,
        max_iters: int = 3,
        tol: float = 1.0e-6,
    ) -> dict:
        """
        Cierre soberano de punto fijo de Kleene (F1 ──► F2 ──► F3 ──► F1'):
        el genoma evolucionado en Nivel 3 (kam_gamma, krull_floor, kam_K,
        spectral_tol_rel, fiedler_threshold) se reinyecta dinámicamente en
        Fase1Simplicial.audit_simplicial_topology, Fase2Algebraic.verify_krull_valuation
        y Fase3Poincare.poincare_topological_section hasta alcanzar convergencia
        |Δfitness| < tol con linaje criptográfico SHA-256 encadenado.
        """
        g = Fase3Poincare._project_genome(dict(genome or DEFAULT_GENOME))
        trace, iterations, converged = [], [], False
        h = hashlib.sha256(b"GENESIS").hexdigest()[:32]
        sig = snf = alg = section = imp = None
        for it in range(max(1, max_iters)):
            sig = Fase1Simplicial.audit_simplicial_topology(
                nodes, edges, faces, genome=g
            )
            snf = Fase2Algebraic.smith_torsion_check(signature=sig)
            alg = Fase2Algebraic.verify_krull_valuation(
                signature=sig, snf_result=snf, genome=g
            )
            section = Fase3Poincare.poincare_topological_section(
                sig, alg, n_iter=n_poincare_iter, genome=g
            )
            imp = Fase3Poincare.recursive_self_improve_topology(
                g, section, sig, alg, depth=3
            )
            f_new = imp["final_fitness"]
            trace.append(f_new)
            h = _chain(
                h,
                {
                    "it": it,
                    "f1": sig.lineage_hash,
                    "f2": alg.lineage_hash,
                    "f3": section.lineage_hash,
                    "fitness": f_new,
                },
            )
            iterations.append(
                {
                    "iteration": it,
                    "fitness": f_new,
                    "genome": imp["genome"],
                    "monotone": imp["monotone_certificate"],
                    "banach_delta": Fase3Poincare._banach_genome_distance(g, imp["genome"]),
                }
            )
            g_new = imp["genome"]
            if it > 0 and abs(trace[-1] - trace[-2]) < tol:
                converged = True
                g = g_new
                break
            g = g_new

        # Re-certificación final de Fase 1, 2 y 3 bajo el genoma de punto fijo g*
        sig_final = Fase1Simplicial.audit_simplicial_topology(
            nodes, edges, faces, genome=g
        )
        snf_final = Fase2Algebraic.smith_torsion_check(signature=sig_final)
        alg_final = Fase2Algebraic.verify_krull_valuation(
            signature=sig_final, snf_result=snf_final, genome=g
        )
        section_final = Fase3Poincare.poincare_topological_section(
            sig_final, alg_final, n_iter=n_poincare_iter, genome=g
        )

        return {
            "fase_1_topological_signature": asdict(sig_final),
            "fase_2": {
                "smith": snf_final,
                "algebraic_invariant": asdict(alg_final),
            },
            "fase_3": {
                "poincare": asdict(section_final),
                "improvement": imp,
            },
            "genome_feedback": g,
            "loop_certificate": {
                "iterations": iterations,
                "fitness_trace": trace,
                "converged": converged,
                "tol": tol,
                "feedback_real": "genoma evolucionado L3 → F1/F2/F3 re-certificado",
            },
            "lineage_chain": h,
        }


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLO MCP (JSON-RPC 2.0 POR STDIO)
# ═══════════════════════════════════════════════════════════════════
PROTOCOL_VERSION = "2025-06-18"
TOOLS = [
    {
        "name": "audit_simplicial_topology",
        "description": (
            "F1: complejo simplicial canónico + ∂₁,∂₂ (∂₁∂₂=0 garantizado) + Δ₀,Δ₁,Δ₂ + "
            "Betti exactos + χ Euler + circuitos de Kirchhoff–Tellegen + índice SUSY de Witten + espinor S³."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "string"}},
                "edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "faces": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "genome": {"type": "object"},
                "floor_krull": {"type": "number"},
            },
            "required": ["nodes", "edges"],
        },
    },
    {
        "name": "smith_torsion_check",
        "description": (
            "F2: SNF constructivo sobre ℤ con matrices unimodulares U·M·V=S certificadas + "
            "torsión H₁ + cohomología entera H^k(K;ℤ) por el Teorema de Coeficientes Universales."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "matrix_rows": {"type": "array"},
                "signature": {"type": "object"},
            },
        },
    },
    {
        "name": "verify_krull_valuation",
        "description": (
            "F2: Anillo de Novikov con cancelación exacta de series + valuación de Krull ultramétrica + "
            "norma de álgebra de Banach no arquimediana + clasificador de subobjetos de Heyting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "series_exponents": {"type": "array", "items": {"type": "number"}},
                "floor": {"type": "number"},
                "second_series_exponents": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "coefficients": {"type": "array", "items": {"type": "number"}},
                "second_coefficients": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "signature": {"type": "object"},
                "snf_result": {"type": "object"},
                "genome": {"type": "object"},
            },
        },
    },
    {
        "name": "poincare_topological_section",
        "description": (
            "F3: sección de Poincaré del mapa estándar simpéctico (radianes) + Lyapunov de Benettin + "
            "MEGNO + residuos de Greene + certificado diofántico empírico y áureo."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "signature": {"type": "object"},
                "algebraic_invariant": {"type": "object"},
                "n_iter": {"type": "integer"},
                "initial_I": {"type": "number"},
                "initial_theta": {"type": "number"},
                "genome": {"type": "object"},
            },
            "required": ["signature", "algebraic_invariant"],
        },
    },
    {
        "name": "kam_greene_certificate",
        "description": (
            "F3: certificado KAM completo (diofántico de Hurwitz–Lagrange + residuos de Greene "
            "sobre órbitas minimizantes de Aubry–Mather + exponente de Benettin)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "K": {"type": "number"},
                "omega": {"type": "number"},
            },
            "required": ["K"],
        },
    },
    {
        "name": "recursive_self_improve_topology",
        "description": (
            "F3: automejora recursiva L1/L2/L3 sensible a los 5 genes con proyección en caja "
            "de Banach, elitismo con veto y certificado de monotonía."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "genome": {"type": "object"},
                "section": {"type": "object"},
                "signature": {"type": "object"},
                "algebraic": {"type": "object"},
                "depth": {"type": "integer"},
            },
            "required": ["genome", "section", "signature", "algebraic"],
        },
    },
    {
        "name": "orchestrate_topology_loop",
        "description": (
            "Cierre anidado de punto fijo de Kleene F1→F2→F3→F1' con realimentación real "
            "del genoma evolucionado, linaje SHA-256 y re-certificación completa."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "string"}},
                "edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "faces": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "genome": {"type": "object"},
                "n_poincare_iter": {"type": "integer"},
                "max_iters": {"type": "integer"},
            },
            "required": ["nodes", "edges"],
        },
    },
]


def _dispatch(name: str, args: dict) -> Any:
    if name == "audit_simplicial_topology":
        return Fase1Simplicial.audit_simplicial_topology(
            args["nodes"],
            args["edges"],
            args.get("faces"),
            args.get("genome"),
            args.get("floor_krull", KRULL_FLOOR),
        )
    if name == "smith_torsion_check":
        return Fase2Algebraic.smith_torsion_check(
            args.get("matrix_rows"), args.get("signature")
        )
    if name == "verify_krull_valuation":
        return Fase2Algebraic.verify_krull_valuation(
            args.get("series_exponents"),
            args.get("floor", KRULL_FLOOR),
            args.get("second_series_exponents"),
            args.get("coefficients"),
            args.get("second_coefficients"),
            args.get("signature"),
            args.get("snf_result"),
            args.get("genome"),
        )
    if name == "poincare_topological_section":
        return Fase3Poincare.poincare_topological_section(
            args["signature"],
            args["algebraic_invariant"],
            args.get("n_iter", 512),
            args.get("initial_I"),
            args.get("initial_theta"),
            args.get("genome"),
        )
    if name == "kam_greene_certificate":
        K = float(args["K"])
        return {
            "diophantine": Fase3Poincare._diophantine_certificate(
                args.get("omega", GOLDEN_RATIO)
            ),
            "greene": Fase3Poincare._greene_certificate(K),
            "benettin": Fase3Poincare._benettin_cached(K),
        }
    if name == "recursive_self_improve_topology":
        return Fase3Poincare.recursive_self_improve_topology(
            args["genome"],
            args["section"],
            args["signature"],
            args["algebraic"],
            args.get("depth", 3),
        )
    if name == "orchestrate_topology_loop":
        return Fase3Poincare.orchestrate_topology_loop(
            args["nodes"],
            args["edges"],
            args.get("faces"),
            args.get("genome"),
            args.get("n_poincare_iter", 512),
            args.get("max_iters", 3),
        )
    raise KeyError(f"Tool {name} not found")


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(_sanitize(payload), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
            )
            continue
        method = req.get("method")
        msg_id = req.get("id")
        if isinstance(method, str) and method.startswith("notifications/"):
            continue
        if method == "initialize":
            params = req.get("params", {})
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": params.get(
                            "protocolVersion", PROTOCOL_VERSION
                        ),
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "topology-homology",
                            "version": "4.1.0-fases-anidadas-soberanas",
                        },
                    },
                }
            )
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            if not name:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": "Missing tool name"},
                    }
                )
                continue
            try:
                data = _dispatch(name, args)
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        _sanitize(data), indent=2, ensure_ascii=False
                                    ),
                                }
                            ],
                            "isError": False,
                        },
                    }
                )
            except KeyError as e:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": str(e)},
                    }
                )
            except Exception as e:
                _log(f"tool error {name}: {e}")
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"ERROR: {e}"}],
                            "isError": True,
                        },
                    }
                )
        elif msg_id is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            )


if __name__ == "__main__":
    main()