# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Topological Analyzer Agent (Soberano de Persistencia Homológica)                    ║
║ Ruta   : app/agents/tactics/topological_analyzer_agent.py                                    ║
║ Versión: 4.1.0-Poincaré-Celestial-TDA-Nested-Phases                                          ║
║ Tratados: Edelsbrunner–Harer, Cohen-Steiner–Edelsbrunner–Harer, Cheeger, Fiedler,            ║
║           Kirchhoff, Forman, Ollivier, Poincaré (Mécanique Céleste), Kac (1947)              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
ANIDAMIENTO FORMAL DE FASES
────────────────────────────────────────────────────────────────────────────────────────────────
  FASE 1 ──► Phase1TDADossier ──► lift_observe_to_celestial_filtration_bundle
           └── objeto terminal = objeto inicial de la FASE 2 (CelestialFiltrationBundle)

  FASE 2 ──► Phase2TDADossier ──► seed_tda_recurrence_from_orientation
           └── objeto terminal = objeto inicial de la FASE 3 (TDARecurrenceSeed)

  FASE 3 ──► certify_poincare_kac_from_tda_seed ──► TopologicalAnalyzerCertificate
           └── lazo OODA cerrado (Heyting Ω₃ + Crowbar ESP32 + SHA-256)

INVARIANTES (honestos)
  1. Persistencia H_k(𝔽₂) de la filtración de flag/Vietoris–Rips del grafo ponderado.
  2. Estabilidad de bottleneck: d_B(Dgm(f), Dgm(g)) ≤ ‖f − g‖_∞.
  3. Cheeger–Fiedler para el Laplaciano *normalizado*: λ₂/2 ≤ h ≤ √(2λ₂).
  4. Kirchhoff: τ(K) = n⁻¹ ∏_{i≥2} λ_i(D − A); τ = 0 ⇔ no conexo.
  5. Forman–Ricci (combinatorio exacto) y Ollivier–Ricci (W₁ del lazy walk, n pequeño).
  6. Gauss–Bonnet discreto: Σ_v (2 − deg v) = 2χ.  Poincaré–Hopf exige un campo; no se finge.
  7. Morse–Forman: β_k ≤ m_k tras matching de bosque generador; no se usan umbrales de grado.
  8. Retorno de Poincaré sobre Φ(K_ε) ∈ ℝ⁶; Birkhoff solo si hay twist anular auténtico.
  9. Melnikov no se fabrica: sólo si el llamador aporta (H₀, H₁, silla).
 10. Recurrencia de Poincaré–Kac sobre el *lazy random walk* del grafo (dinámica nativa).
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from app.wisdom.godel_engine import (
    CrowbarCircuitPhysicsEngine,
    CrowbarPhysicalTelemetry,
    HeytingVerdict as HeytingOmega3,
    MelnikovChaosCertificate,
    MelnikovFunctionEngine,
    PoincareBirkhoffCertificate,
    PoincareBirkhoffEngine,
    PoincareRecurrenceCertificate,
    PoincareRecurrenceEngine,
    PoincareReturnMapCertificate,
    PoincareReturnMapEngine,
)

logger = logging.getLogger("MIC.Tactics.TopologicalAnalyzerAgent")

__version__: Final[str] = "4.1.0-Poincaré-Celestial-TDA-Nested-Phases"

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
_MAX_HOMOLOGY_DIM: Final[int] = 2
_OLLIVIER_LAZY: Final[float] = 0.5
_OLLIVIER_QUANT: Final[int] = 240


class TopologicalObstructionError(Exception):
    """Obstrucción homológica o colapso de la filtración persistente."""


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMBINATORIAS
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def _symmetric_adjacency(matrix: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(matrix, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("La matriz de adyacencia debe ser cuadrada.")
    return 0.5 * (adjacency + adjacency.T)


def _edge_list(adjacency: np.ndarray) -> List[Tuple[int, int, float]]:
    n = adjacency.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > _WILKINSON_LIMIT:
                edges.append((i, j, float(adjacency[i, j])))
    return edges


def _boolean_adjacency(adjacency: np.ndarray) -> np.ndarray:
    return adjacency > _WILKINSON_LIMIT


def _degrees_boolean(adjacency: np.ndarray) -> np.ndarray:
    return _boolean_adjacency(adjacency).sum(axis=1).astype(np.float64)


def _count_triangles(adjacency: np.ndarray) -> int:
    mask = _boolean_adjacency(adjacency)
    n = mask.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not mask[i, j]:
                continue
            for k in range(j + 1, n):
                if mask[i, k] and mask[j, k]:
                    count += 1
    return count


def _graph_distances(mask: np.ndarray) -> np.ndarray:
    """Distancias geodésicas por BFS; desconexos → n (diámetro-proxy finito)."""
    n = int(mask.shape[0])
    dist = np.full((n, n), n, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for src in range(n):
        queue = [src]
        seen = {src}
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            neighbors = np.where(mask[u])[0]
            for v in neighbors:
                v_int = int(v)
                if v_int in seen:
                    continue
                seen.add(v_int)
                dist[src, v_int] = dist[src, u] + 1.0
                queue.append(v_int)
    return dist


def _as_diagram(
    pairs: Dict[int, List[Tuple[float, float]]],
    essential: Dict[int, List[Tuple[float, float]]],
    dim: int,
) -> np.ndarray:
    pts = list(pairs.get(dim, [])) + list(essential.get(dim, []))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVACIÓN PERSISTENTE — VIETORIS–RIPS / FLAG, HOMOLOGÍA 𝔽₂ Y RETORNO DE POINCARÉ
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.1 SUMA COMPENSADA DE KAHAN–BABUŠKA–NEUMAIER
# ───────────────────────────────────────────────────────────────────────────────────────────────
class KahanSummation:
    r"""Suma compensada: error de redondeo O(ε_mach) independiente del cardinal."""

    @staticmethod
    def sum_axis(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        data = np.asarray(matrix, dtype=np.float64)
        if axis == 0:
            data = data.T
        n_rows, n_cols = data.shape
        result = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            acc = 0.0
            compensation = 0.0
            for j in range(n_cols):
                y = data[i, j] - compensation
                t = acc + y
                compensation = (t - acc) - y
                acc = t
            result[i] = acc
        return result


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.2 FILTRACIÓN DE FLAG / VIETORIS–RIPS DEL GRAFO PONDERADO
# ───────────────────────────────────────────────────────────────────────────────────────────────
class VietorisRipsFiltration:
    r"""
    Complejo de cliques filtrado por el peso máximo de arista (flag complex):
        VR_ε = { σ ⊆ V : max_{{u,v} ⊂ σ} w(u,v) ≤ ε, |σ| ≤ max_dim+1 }.
    Vértices nacen en 0; aristas en w; triángulos en max(w_e). Homología sobre 𝔽₂.
    """

    @staticmethod
    def build_filtration(
        adjacency: np.ndarray,
        max_dim: int = _MAX_HOMOLOGY_DIM,
    ) -> Tuple[List[int], List[float], List[List[int]]]:
        n = adjacency.shape[0]
        edge_weights: Dict[Tuple[int, int], float] = {
            (i, j): float(adjacency[i, j])
            for i in range(n)
            for j in range(i + 1, n)
            if adjacency[i, j] > _WILKINSON_LIMIT
        }
        simplices: List[Tuple[int, float, Tuple[int, ...]]] = [(0, 0.0, (v,)) for v in range(n)]
        for (i, j), weight in edge_weights.items():
            simplices.append((1, weight, (i, j)))
        if max_dim >= 2:
            for a in range(n):
                for b in range(a + 1, n):
                    if (a, b) not in edge_weights:
                        continue
                    for c in range(b + 1, n):
                        if (a, c) in edge_weights and (b, c) in edge_weights:
                            w_max = max(edge_weights[(a, b)], edge_weights[(a, c)], edge_weights[(b, c)])
                            simplices.append((2, w_max, (a, b, c)))
        simplices.sort(key=lambda item: (item[1], item[0], item[2]))
        dims = [item[0] for item in simplices]
        filt = [item[1] for item in simplices]
        idx_of = {item[2]: k for k, item in enumerate(simplices)}
        boundary: List[List[int]] = [[] for _ in simplices]
        for k, (dim, _f, verts) in enumerate(simplices):
            if dim == 1:
                i, j = verts
                boundary[k] = sorted([idx_of[(i,)], idx_of[(j,)]])
            elif dim == 2:
                a, b, c = verts
                boundary[k] = sorted([idx_of[(a, b)], idx_of[(a, c)], idx_of[(b, c)]])
        return dims, filt, boundary


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.3 REDUCCIÓN DE FRONTERA SOBRE 𝔽₂ (EDELSBRUNNER–HARER §VII.1)
# ───────────────────────────────────────────────────────────────────────────────────────────────
class GF2BoundaryReducer:
    r"""Reducción estándar: cada columna reducida tiene pivote único o es nula."""

    @staticmethod
    def _xor_sorted(left: List[int], right: List[int]) -> List[int]:
        i = j = 0
        out: List[int] = []
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                out.append(left[i])
                i += 1
            elif left[i] > right[j]:
                out.append(right[j])
                j += 1
            else:
                i += 1
                j += 1
        if i < len(left):
            out.extend(left[i:])
        if j < len(right):
            out.extend(right[j:])
        return out

    @classmethod
    def reduce(cls, boundary: List[List[int]]) -> Tuple[List[List[int]], List[Optional[int]]]:
        n_cols = len(boundary)
        reduced = [list(col) for col in boundary]
        low: List[Optional[int]] = [None] * n_cols
        low_to_col: Dict[int, int] = {}
        for j in range(n_cols):
            col = reduced[j]
            while col:
                pivot = col[-1]
                if pivot in low_to_col:
                    col = cls._xor_sorted(col, reduced[low_to_col[pivot]])
                else:
                    break
            reduced[j] = col
            if col:
                low[j] = col[-1]
                low_to_col[col[-1]] = j
        return reduced, low

    @staticmethod
    def extract_persistence(
        reduced: List[List[int]],
        low: List[Optional[int]],
        dims: List[int],
        filt: List[float],
        max_dim: int = _MAX_HOMOLOGY_DIM,
    ) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, List[Tuple[float, float]]]]:
        n_cols = len(low)
        negatives = [j for j in range(n_cols) if reduced[j]]
        pivots = {low[j] for j in negatives if low[j] is not None}
        pairs: Dict[int, List[Tuple[float, float]]] = {d: [] for d in range(max_dim + 1)}
        essential: Dict[int, List[Tuple[float, float]]] = {d: [] for d in range(max_dim + 1)}
        for j in negatives:
            i = low[j]
            if i is not None and dims[i] <= max_dim:
                pairs[dims[i]].append((filt[i], filt[j]))
        for j in range(n_cols):
            if not reduced[j] and j not in pivots and dims[j] <= max_dim:
                essential[dims[j]].append((filt[j], float("inf")))
        return pairs, essential


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.4 DISTANCIAS DE BOTTLENECK Y WASSERSTEIN-1 EN EL PLANO DE PERSISTENCIA
# ───────────────────────────────────────────────────────────────────────────────────────────────
class BottleneckDistanceEngine:
    r"""
    d_B(D₁, D₂) = inf_γ sup_x ‖x − γ(x)‖_∞  sobre biyecciones con diagonal.
    Las clases esenciales (death = ∞) se emparejan entre sí por coordenada de nacimiento;
    un esencial no emparejado ⇒ d_B = +∞ (se reporta como cota numérica grande).
    """

    _ESSENTIAL_SENTINEL: Final[float] = 1.0e12

    @staticmethod
    def _split(diagram: np.ndarray) -> Tuple[List[Tuple[float, float]], List[float]]:
        finite: List[Tuple[float, float]] = []
        births_ess: List[float] = []
        if diagram is None or np.asarray(diagram).size == 0:
            return finite, births_ess
        for birth, death in np.asarray(diagram, dtype=np.float64):
            if math.isinf(float(death)):
                births_ess.append(float(birth))
            else:
                finite.append((float(birth), float(death)))
        return finite, births_ess

    @staticmethod
    def _augmented_cost(p1: List[Tuple[float, float]], p2: List[Tuple[float, float]]) -> np.ndarray:
        def diag(birth: float, death: float) -> Tuple[float, float]:
            mid = 0.5 * (birth + death)
            return (mid, mid)

        d1 = list(p1) + [diag(b, d) for (b, d) in p2]
        d2 = list(p2) + [diag(b, d) for (b, d) in p1]
        n = len(d1)
        cost = np.zeros((n, n), dtype=np.float64)
        for i, (bi, di) in enumerate(d1):
            for j, (bj, dj) in enumerate(d2):
                cost[i, j] = max(abs(bi - bj), abs(di - dj))
        return cost

    @classmethod
    def distance(
        cls,
        dgm1: np.ndarray,
        dgm2: np.ndarray,
        tol: float = 1e-7,
        max_bisect: int = 55,
    ) -> float:
        f1, e1 = cls._split(dgm1)
        f2, e2 = cls._split(dgm2)
        if len(e1) != len(e2):
            return cls._ESSENTIAL_SENTINEL
        essential_cost = 0.0
        if e1:
            births_a = np.sort(np.asarray(e1, dtype=np.float64))
            births_b = np.sort(np.asarray(e2, dtype=np.float64))
            essential_cost = float(np.max(np.abs(births_a - births_b)))
        if not f1 and not f2:
            return essential_cost
        cost = cls._augmented_cost(f1, f2)
        lo, hi = 0.0, float(cost.max()) if cost.size else 0.0
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            adjacent = (cost <= mid + 1e-14).astype(np.int8)
            matching = maximum_bipartite_matching(csr_matrix(adjacent), perm_type="column")
            if bool(np.all(matching >= 0)):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return max(hi, essential_cost)

    @classmethod
    def wasserstein_1(cls, dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        f1, e1 = cls._split(dgm1)
        f2, e2 = cls._split(dgm2)
        total = 0.0
        if e1 or e2:
            n_ess = max(len(e1), len(e2))
            a = list(e1) + [0.0] * (n_ess - len(e1))
            b = list(e2) + [0.0] * (n_ess - len(e2))
            total += float(np.sum(np.abs(np.sort(a) - np.sort(b))))
        if not f1 and not f2:
            return total
        cost = cls._augmented_cost(f1, f2)
        row, col = linear_sum_assignment(cost)
        return total + float(cost[row, col].sum())


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.5 SISTEMA DINÁMICO DE LA FILTRACIÓN Y MAPA DE RETORNO DE POINCARÉ
# ───────────────────────────────────────────────────────────────────────────────────────────────
class FiltrationDynamicalSystem:
    r"""
    Φ(K_ε) = (β₀(ε), β₁(ε), V(ε), E(ε), χ(ε), ε̂) ∈ ℝ⁶.
    ε es el «tiempo lento» de continuación de Poincaré de las clases persistentes.
    """
    FEATURE_DIM: Final[int] = 6

    @staticmethod
    def trajectory_from_filtration(
        dims: List[int],
        filt: List[float],
        pairs: Dict[int, List[Tuple[float, float]]],
        essential: Dict[int, List[Tuple[float, float]]],
        num_samples: int = 80,
    ) -> np.ndarray:
        if not filt:
            return np.zeros((1, FiltrationDynamicalSystem.FEATURE_DIM), dtype=np.float64)
        lo, hi = float(min(filt)), float(max(filt))
        pad = max(1e-3, 0.05 * (hi - lo + 1e-15))
        grid = np.linspace(lo - pad, hi + pad, num_samples)
        traj = np.zeros((num_samples, FiltrationDynamicalSystem.FEATURE_DIM), dtype=np.float64)
        span = hi - lo + 1e-9
        for k, eps in enumerate(grid):
            v_eps = sum(1 for dim, value in zip(dims, filt) if dim == 0 and value <= eps) or 1
            e_eps = sum(1 for dim, value in zip(dims, filt) if dim == 1 and value <= eps)
            t_eps = sum(1 for dim, value in zip(dims, filt) if dim == 2 and value <= eps)
            b0 = sum(1 for b, d in pairs.get(0, []) if b <= eps < d)
            b0 += sum(1 for b, _d in essential.get(0, []) if b <= eps)
            b1 = sum(1 for b, d in pairs.get(1, []) if b <= eps < d)
            b1 += sum(1 for b, _d in essential.get(1, []) if b <= eps)
            chi = v_eps - e_eps + t_eps
            traj[k] = [float(b0), float(b1), float(v_eps), float(e_eps), float(chi), float(eps / span)]
        return traj

    @staticmethod
    def spectral_curve(
        adjacency: np.ndarray,
        filt: List[float],
        num_samples: int = 40,
    ) -> np.ndarray:
        """Continuación de Poincaré de λ₂(ε) sobre el 1-esqueleto subnivel."""
        if not filt:
            return np.zeros((1, 2), dtype=np.float64)
        lo, hi = float(min(filt)), float(max(filt))
        grid = np.linspace(lo, hi, num_samples)
        curve = np.zeros((num_samples, 2), dtype=np.float64)
        n = adjacency.shape[0]
        for k, eps in enumerate(grid):
            sub = np.where(adjacency <= eps, adjacency, 0.0)
            sub = np.where(adjacency > _WILKINSON_LIMIT, sub, 0.0)
            deg = sub.sum(axis=1)
            comb = np.diag(deg) - sub
            eig = np.sort(np.maximum(0.0, la.eigvalsh(0.5 * (comb + comb.T))))
            lam2 = float(eig[1]) if n > 1 else 0.0
            curve[k] = [eps, lam2]
        return curve

    @staticmethod
    def compute_return_map(trajectory: np.ndarray) -> Optional[PoincareReturnMapCertificate]:
        try:
            times = np.arange(trajectory.shape[0], dtype=np.float64)
            section_normal = np.zeros(trajectory.shape[1], dtype=np.float64)
            section_normal[0] = 1.0
            offset = float(np.median(trajectory[:, 0]))
            return PoincareReturnMapEngine.compute_return_map(
                trajectory=trajectory,
                time_samples=times,
                section_normal=section_normal,
                section_offset=offset,
            )
        except Exception as exc:
            logger.debug("[Fase 1] Return map omitted: %s", exc)
            return None


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.6 EXPEDIENTE DE FASE 1 (OBSERVE)
# ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Phase1TDADossier:
    """Expediente inmutable de la Fase 1 (Observe) del ciclo OODA."""
    adjacency_matrix: NDArray[np.float64]
    degree_vector: NDArray[np.float64]
    bottleneck_distance: float
    is_dossier_valid: bool
    persistence_diagram_h0: Optional[NDArray[np.float64]] = None
    persistence_diagram_h1: Optional[NDArray[np.float64]] = None
    persistence_diagram_h2: Optional[NDArray[np.float64]] = None
    wasserstein_distance_h0: float = 0.0
    wasserstein_distance_h1: float = 0.0
    num_simplices: int = 0
    filtration_range: Tuple[float, float] = (0.0, 0.0)
    return_map: Optional[PoincareReturnMapCertificate] = None
    filtration_signature_sha256: str = ""


def observe_filtration_dossier(
    adjacency_matrix: NDArray[np.float64],
    reference_diagram: Optional[NDArray[np.float64]],
    bottleneck_tolerance: float,
) -> Tuple[Phase1TDADossier, List[int], List[float], Dict[int, List[Tuple[float, float]]], Dict[int, List[Tuple[float, float]]]]:
    """Síntesis estructural de la observación persistente (núcleo de Observe)."""
    adjacency = _symmetric_adjacency(adjacency_matrix)
    degree_vector = KahanSummation.sum_axis(adjacency, axis=1)
    dims, filt, boundary = VietorisRipsFiltration.build_filtration(adjacency)
    reduced, low = GF2BoundaryReducer.reduce(boundary)
    pairs, essential = GF2BoundaryReducer.extract_persistence(reduced, low, dims, filt)
    dgm_h0 = _as_diagram(pairs, essential, 0)
    dgm_h1 = _as_diagram(pairs, essential, 1)
    dgm_h2 = _as_diagram(pairs, essential, 2)
    bottleneck_dist = 0.0
    wass_h0 = 0.0
    wass_h1 = 0.0
    if reference_diagram is not None and np.asarray(reference_diagram).size:
        ref = np.asarray(reference_diagram, dtype=np.float64)
        bottleneck_dist = BottleneckDistanceEngine.distance(dgm_h0, ref)
        wass_h0 = BottleneckDistanceEngine.wasserstein_1(dgm_h0, ref)
        wass_h1 = BottleneckDistanceEngine.wasserstein_1(dgm_h1, np.zeros((0, 2)))
    trajectory = FiltrationDynamicalSystem.trajectory_from_filtration(dims, filt, pairs, essential)
    return_map_cert = FiltrationDynamicalSystem.compute_return_map(trajectory)
    hasher = hashlib.sha256()
    hasher.update(repr(filt).encode("utf-8"))
    hasher.update(str(dims).encode("utf-8"))
    hasher.update(str(time.time_ns()).encode("utf-8"))
    frange = (float(min(filt)) if filt else 0.0, float(max(filt)) if filt else 0.0)
    dossier = Phase1TDADossier(
        adjacency_matrix=np.copy(adjacency),
        degree_vector=degree_vector,
        bottleneck_distance=bottleneck_dist,
        is_dossier_valid=bool(bottleneck_dist <= bottleneck_tolerance),
        persistence_diagram_h0=dgm_h0,
        persistence_diagram_h1=dgm_h1,
        persistence_diagram_h2=dgm_h2,
        wasserstein_distance_h0=wass_h0,
        wasserstein_distance_h1=wass_h1,
        num_simplices=len(dims),
        filtration_range=frange,
        return_map=return_map_cert,
        filtration_signature_sha256=hasher.hexdigest(),
    )
    return dossier, dims, filt, pairs, essential


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §1.7 ENLACE TERMINAL FASE 1 → INICIO FASE 2
#      Elevación del expediente persistente al fibrado de filtración celeste
# ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CelestialFiltrationBundle:
    r"""
    OBJETO TERMINAL DE LA FASE 1 Y OBJETO INICIAL DE LA FASE 2.

    Fibrado (K_ε, Φ(ε), λ₂(ε), ω_feat) de la mecánica celeste de Poincaré sobre TDA:
      • ε es el tiempo lento de continuación de las clases (órbitas de Morse),
      • Φ(K_ε) ∈ ℝ⁶ es la sección de features,
      • λ₂(ε) es la continuación espectral de Fiedler (frecuencia del toro),
      • el mapa de retorno P : Σ → Σ vive en la hipersuperficie β₀ = mediana.
    """
    dossier: Phase1TDADossier
    dims: Tuple[int, ...]
    filtration_values: Tuple[float, ...]
    pairs_h0: Tuple[Tuple[float, float], ...]
    pairs_h1: Tuple[Tuple[float, float], ...]
    essential_h0: Tuple[Tuple[float, float], ...]
    essential_h1: Tuple[Tuple[float, float], ...]
    feature_trajectory: NDArray[np.float64]
    spectral_curve: NDArray[np.float64]
    dirichlet_energy_fiedler_proxy: float
    configuration_dim: int


def lift_observe_to_celestial_filtration_bundle(
    dossier: Phase1TDADossier,
    dims: Optional[List[int]] = None,
    filt: Optional[List[float]] = None,
    pairs: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    essential: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> CelestialFiltrationBundle:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 1 / PRIMER MORFISMO DE LA FASE 2.

    Reconstruye (si hace falta) la filtración y eleva el dossier al fibrado celeste
    (K_ε, Φ, λ₂(ε)). Toda la orientación espectral de la Fase 2 se alimenta de aquí.
    """
    adjacency = dossier.adjacency_matrix
    if dims is None or filt is None or pairs is None or essential is None:
        dims, filt, boundary = VietorisRipsFiltration.build_filtration(adjacency)
        reduced, low = GF2BoundaryReducer.reduce(boundary)
        pairs, essential = GF2BoundaryReducer.extract_persistence(reduced, low, dims, filt)
    trajectory = FiltrationDynamicalSystem.trajectory_from_filtration(dims, filt, pairs, essential)
    curve = FiltrationDynamicalSystem.spectral_curve(adjacency, filt)
    n = adjacency.shape[0]
    deg = dossier.degree_vector
    comb = np.diag(deg) - adjacency
    eig = np.sort(np.maximum(0.0, la.eigvalsh(0.5 * (comb + comb.T))))
    energy = float(eig[1]) if n > 1 else 0.0
    return CelestialFiltrationBundle(
        dossier=dossier,
        dims=tuple(int(d) for d in dims),
        filtration_values=tuple(float(v) for v in filt),
        pairs_h0=tuple((float(a), float(b)) for a, b in pairs.get(0, [])),
        pairs_h1=tuple((float(a), float(b)) for a, b in pairs.get(1, [])),
        essential_h0=tuple((float(a), float(b)) for a, b in essential.get(0, [])),
        essential_h1=tuple((float(a), float(b)) for a, b in essential.get(1, [])),
        feature_trajectory=trajectory,
        spectral_curve=curve,
        dirichlet_energy_fiedler_proxy=energy,
        configuration_dim=n,
    )


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENTACIÓN ESPECTRAL — FIEDLER, CHEEGER, KIRCHHOFF, FORMAN/OLLIVIER, GAUSS–BONNET
#         (continúa desde CelestialFiltrationBundle)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.1 ESPECTRO DEL LAPLACIANO NORMALIZADO Y VECTOR DE FIEDLER
# ───────────────────────────────────────────────────────────────────────────────────────────────
class SpectralGraphEngine:
    r"""
    ℒ = I − D^{−1/2} A D^{−1/2},  0 = λ₁ ≤ λ₂ ≤ ⋯ ≤ λ_n ≤ 2.
    El corte de Fiedler sign(φ₂) es *un* corte; su razón de Cheeger acota h por arriba.
    """

    @staticmethod
    def compute(adjacency: np.ndarray, degree_vector: np.ndarray) -> Dict[str, Any]:
        n = adjacency.shape[0]
        inv_sqrt = np.power(np.maximum(degree_vector, _WILKINSON_LIMIT), -0.5)
        d_inv = np.diag(inv_sqrt)
        l_norm = np.eye(n, dtype=np.float64) - d_inv @ adjacency @ d_inv
        l_norm = 0.5 * (l_norm + l_norm.T)
        eigvals, eigvecs = la.eigh(l_norm)
        eigvals = np.maximum(0.0, eigvals)
        fiedler_vector = eigvecs[:, 1] if eigvecs.shape[1] > 1 else np.zeros(n, dtype=np.float64)
        signs = np.sign(fiedler_vector)
        signs[signs == 0] = 1.0
        side_a = np.where(signs >= 0)[0]
        side_b = np.where(signs < 0)[0]
        cut_size = 0.0
        for i in side_a:
            for j in side_b:
                cut_size += float(adjacency[int(i), int(j)])
        vol_a = float(degree_vector[side_a].sum()) if side_a.size else 0.0
        vol_b = float(degree_vector[side_b].sum()) if side_b.size else 0.0
        denom = max(min(vol_a, vol_b), _WILKINSON_LIMIT)
        return {
            "L_norm": l_norm,
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "fiedler_vector": fiedler_vector,
            "side_a": side_a,
            "side_b": side_b,
            "cut_size": cut_size,
            "cheeger_ratio_empirical": cut_size / denom,
            "vol_a": vol_a,
            "vol_b": vol_b,
        }


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.2 TEOREMA MATRIZ-ÁRBOL DE KIRCHHOFF
# ───────────────────────────────────────────────────────────────────────────────────────────────
class KirchhoffMatrixTreeEngine:
    r"""τ(K) = n⁻¹ ∏_{i=2}^n λ_i(D − A).  τ = 0 ⇔ K no conexo.  Cofactor = testigo."""

    @staticmethod
    def compute(adjacency: np.ndarray, degree_vector: np.ndarray) -> Dict[str, float]:
        n = adjacency.shape[0]
        combinatorial = 0.5 * ((np.diag(degree_vector) - adjacency) + (np.diag(degree_vector) - adjacency).T)
        eigvals = np.sort(np.maximum(0.0, la.eigvalsh(combinatorial)))
        if n < 2 or eigvals[1] < 1e-12:
            tau = 0.0
        else:
            log_tau = float(np.sum(np.log(np.maximum(eigvals[1:], 1e-300)))) - math.log(n)
            tau = math.exp(log_tau) if log_tau < 700 else float("inf")
        cofactor = 0.0
        if n >= 2:
            minor = combinatorial[1:, 1:]
            try:
                cofactor = float(abs(np.linalg.det(minor)))
            except np.linalg.LinAlgError:
                cofactor = tau
        nonzero = eigvals[eigvals > _SPECTRAL_TOL]
        return {
            "spanning_trees": tau,
            "cofactor_witness": cofactor,
            "regularized_determinant": float(np.prod(nonzero)) if nonzero.size else 0.0,
            "spectral_zeta_at_1": float(np.sum(1.0 / nonzero)) if nonzero.size else float("inf"),
            "algebraic_connectivity": float(eigvals[1]) if n > 1 else 0.0,
        }


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.3 CHEEGER, FORMAN–RICCI Y OLLIVIER–RICCI (W₁)
# ───────────────────────────────────────────────────────────────────────────────────────────────
class CheegerFormanOllivierEngine:
    r"""
    Cheeger (normalizado): λ₂/2 ≤ h ≤ √(2λ₂).
    Forman (arista unitaria): Ric_F(u,v) = 4 − deg(u) − deg(v).
    Ollivier: κ(x,y) = 1 − W₁(μ_x^α, μ_y^α)/d(x,y) con μ lazy (α = 1/2), W₁ cuantizado.
    """

    @staticmethod
    def cheeger_bracketing(fiedler_value: float) -> Dict[str, float]:
        h_low = 0.5 * fiedler_value
        h_high = math.sqrt(max(0.0, 2.0 * fiedler_value))
        return {"h_lower": float(h_low), "h_upper": float(h_high), "gap": float(h_high - h_low)}

    @staticmethod
    def forman_ricci_edges(adjacency: np.ndarray) -> Dict[str, Any]:
        degrees = _degrees_boolean(adjacency)
        curvatures: List[float] = []
        edges: List[Tuple[int, int, float]] = []
        for i, j, _w in _edge_list(adjacency):
            kappa = 4.0 - float(degrees[i]) - float(degrees[j])
            curvatures.append(kappa)
            edges.append((i, j, kappa))
        if not curvatures:
            return {"mean_curvature": 0.0, "min_curvature": 0.0, "max_curvature": 0.0, "edges": edges}
        arr = np.asarray(curvatures, dtype=np.float64)
        return {
            "mean_curvature": float(arr.mean()),
            "min_curvature": float(arr.min()),
            "max_curvature": float(arr.max()),
            "edges": edges,
        }

    @staticmethod
    def _lazy_measure(index: int, mask: np.ndarray, alpha: float) -> np.ndarray:
        n = mask.shape[0]
        mu = np.zeros(n, dtype=np.float64)
        mu[index] = alpha
        neighbors = np.where(mask[index])[0]
        if neighbors.size:
            mu[neighbors] += (1.0 - alpha) / float(neighbors.size)
        else:
            mu[index] = 1.0
        return mu

    @staticmethod
    def _w1_quantized(p: np.ndarray, q: np.ndarray, metric: np.ndarray, quant: int = _OLLIVIER_QUANT) -> float:
        a = np.maximum(0, np.round(p * quant).astype(np.int64))
        b = np.maximum(0, np.round(q * quant).astype(np.int64))
        if a.sum() != quant:
            a[int(np.argmax(p))] += quant - int(a.sum())
        if b.sum() != quant:
            b[int(np.argmax(q))] += quant - int(b.sum())
        src = np.repeat(np.arange(p.size), np.maximum(a, 0))
        dst = np.repeat(np.arange(q.size), np.maximum(b, 0))
        if src.size == 0 or dst.size == 0 or src.size != dst.size:
            return float(np.abs(p - q).sum())
        cost = metric[np.ix_(src, dst)]
        row, col = linear_sum_assignment(cost)
        return float(cost[row, col].sum()) / float(quant)

    @classmethod
    def ollivier_ricci_edges(cls, adjacency: np.ndarray, alpha: float = _OLLIVIER_LAZY) -> Dict[str, Any]:
        mask = _boolean_adjacency(adjacency)
        n = mask.shape[0]
        edges = _edge_list(adjacency)
        if not edges:
            return {"mean_curvature": 0.0, "min_curvature": 0.0, "max_curvature": 0.0, "edges": []}
        if n > 24:
            # Proxy de Jaccard (NO es Ollivier): κ ≈ |N∩N'| / max(deg, deg').
            degrees = mask.sum(axis=1).astype(float)
            kappas = []
            packed = []
            for i, j, _w in edges:
                common = float(np.sum(mask[i] & mask[j]))
                kappa = common / max(float(degrees[i]), float(degrees[j]), 1.0)
                kappas.append(kappa)
                packed.append((i, j, kappa))
            arr = np.asarray(kappas, dtype=np.float64)
            return {
                "mean_curvature": float(arr.mean()),
                "min_curvature": float(arr.min()),
                "max_curvature": float(arr.max()),
                "edges": packed,
            }
        dist = _graph_distances(mask)
        kappas = []
        packed = []
        for i, j, _w in edges:
            mu_i = cls._lazy_measure(i, mask, alpha)
            mu_j = cls._lazy_measure(j, mask, alpha)
            w1 = cls._w1_quantized(mu_i, mu_j, dist)
            length = max(float(dist[i, j]), 1.0)
            kappa = 1.0 - w1 / length
            kappas.append(kappa)
            packed.append((i, j, kappa))
        arr = np.asarray(kappas, dtype=np.float64)
        return {
            "mean_curvature": float(arr.mean()),
            "min_curvature": float(arr.min()),
            "max_curvature": float(arr.max()),
            "edges": packed,
        }


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.4 GAUSS–BONNET DISCRETO Y MORSE–FORMAN (BOSQUE GENERADOR)
# ───────────────────────────────────────────────────────────────────────────────────────────────
class GaussBonnetFormanMorseEngine:
    r"""
    Gauss–Bonnet: Σ_v (2 − deg v) = 2V − 2E = 2χ.
    Morse–Forman sobre el 1-esqueleto: matching de un bosque generador
        m₀ = β₀,  m₁ = E − (V − β₀) = E − V + β₀.
    Entonces β₀ ≤ m₀, β₁ ≤ m₁ (igualdad en 1-complejos).
    β₂ ≤ T es la cota trivial de celdas (sin matching 2-dimensional).
    """

    @staticmethod
    def compute(
        adjacency: np.ndarray,
        num_triangles: int,
        betti_0: int,
        betti_1: int,
        betti_2: int,
    ) -> Dict[str, float]:
        n = adjacency.shape[0]
        degrees = _degrees_boolean(adjacency)
        num_edges = int(degrees.sum() / 2)
        chi_combinatorial = float(n - num_edges + num_triangles)
        gauss = float(np.sum(2.0 - degrees))
        gauss_ok = abs(gauss - 2.0 * (n - num_edges)) < 1e-8
        m0 = max(betti_0, 0)
        m1 = max(num_edges - n + betti_0, 0)
        m2 = max(num_triangles, 0)
        chi_spectral = float(betti_0 - betti_1 + betti_2)
        return {
            "gauss_bonnet_total_curvature": gauss,
            "gauss_bonnet_consistent": float(1.0 if gauss_ok else 0.0),
            "chi_combinatorial": chi_combinatorial,
            "chi_spectral": chi_spectral,
            "euler_residual": float(chi_spectral - (n - num_edges)),
            "forman_m0": float(m0),
            "forman_m1": float(m1),
            "forman_m2": float(m2),
            "morse_holds": float(1.0 if (betti_0 <= m0 and betti_1 <= m1 and betti_2 <= m2) else 0.0),
            "morse_residual_max": float(max(m0 - betti_0, m1 - betti_1, m2 - betti_2, 0.0)),
            "index_sum_gauss": gauss / 2.0,
        }


class BettiRecoveryEngine:
    r"""β₀, β₁ por rank-nullity real de ∂₁ (homología de ℝ, no 𝔽₂). β₂ no se finge."""

    @staticmethod
    def from_persistence(essential: Dict[int, List[Tuple[float, float]]]) -> Tuple[int, int, int]:
        return len(essential.get(0, [])), len(essential.get(1, [])), len(essential.get(2, []))

    @staticmethod
    def from_incidence_rank(adjacency: np.ndarray) -> Tuple[int, int, int]:
        n = adjacency.shape[0]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j] > _WILKINSON_LIMIT]
        if not edges:
            return n, 0, 0
        incidence = np.zeros((n, len(edges)), dtype=np.float64)
        for k, (i, j) in enumerate(edges):
            incidence[i, k] = -1.0
            incidence[j, k] = 1.0
        rank_d1 = int(np.linalg.matrix_rank(incidence, tol=1e-9))
        return n - rank_d1, len(edges) - rank_d1, 0


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.5 BIRKHOFF DEL RETORNO DE FILTRACIÓN (MELNIKOV SÓLO SI HAY HAMILTONIANO)
# ───────────────────────────────────────────────────────────────────────────────────────────────
class FiltrationBirkhoffMelnikovEngine:
    r"""
    Birkhoff se evalúa sobre el twist anular (β₀, β₁) del fibrado, no sobre un mapa
    de Chirikov inventado. Melnikov se delega al engine si y sólo si hay (H₀, H₁, silla).
    """

    @staticmethod
    def _annulus_from_features(block: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def twist(xi: np.ndarray) -> np.ndarray:
            theta, action = float(xi[0]), float(np.clip(xi[1], 0.0, 1.0))
            cartesian = np.array([action * math.cos(theta), action * math.sin(theta)], dtype=np.float64)
            image = block @ cartesian
            action_new = float(np.clip(np.linalg.norm(image), 0.0, 1.0))
            theta_new = math.atan2(float(image[1]), float(image[0])) % (2.0 * math.pi)
            return np.array([theta_new, action_new], dtype=np.float64)
        return twist

    @classmethod
    def analyze(
        cls,
        bundle: CelestialFiltrationBundle,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
        saddle_point: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[PoincareBirkhoffCertificate], Optional[MelnikovChaosCertificate], float]:
        birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
        return_map = bundle.dossier.return_map
        traj = bundle.feature_trajectory
        if return_map is not None and traj.shape[0] >= 4 and traj.shape[1] >= 2:
            delta = traj[1:, :2] - traj[:-1, :2]
            gram = delta.T @ delta / max(delta.shape[0], 1)
            try:
                block = la.sqrtm(gram + 1e-9 * np.eye(2)).real
            except Exception:
                block = np.eye(2, dtype=np.float64) * 0.3
            try:
                birkhoff_cert = PoincareBirkhoffEngine.audit_twist_map(
                    cls._annulus_from_features(block),
                    return_map.rotation_number,
                )
            except Exception as exc:
                logger.debug("[Fase 2] Birkhoff omitted: %s", exc)
        melnikov_cert: Optional[MelnikovChaosCertificate] = None
        if hamiltonian_0 is not None and hamiltonian_1 is not None and saddle_point is not None:
            try:
                orbit = MelnikovFunctionEngine._numerical_homoclinic_orbit(hamiltonian_0, saddle_point)
                t_samples = np.linspace(-25.0, 25.0, orbit.shape[0])
                t0_grid = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
                melnikov_cert = MelnikovFunctionEngine.compute_melnikov_function(
                    hamiltonian_0, hamiltonian_1, orbit, t_samples, t0_grid
                )
            except Exception as exc:
                logger.debug("[Fase 2] Melnikov omitted: %s", exc)
        lyap = float(return_map.lyapunov_max) if return_map is not None else 0.0
        return birkhoff_cert, melnikov_cert, lyap


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.6 EXPEDIENTE DE FASE 2 (ORIENT)
# ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Phase2TDADossier:
    """Expediente inmutable de la Fase 2 (Orient) del ciclo OODA."""
    dossier_p1: Phase1TDADossier
    normalized_laplacian: NDArray[np.float64]
    sorted_spectrum: NDArray[np.float64]
    betti_0: int
    betti_1: int
    fiedler_value: float
    euler_characteristic: int
    cheeger_lower_bound: float
    betti_2: int = 0
    fiedler_vector: Optional[NDArray[np.float64]] = None
    cheeger_upper_bound: float = 0.0
    cheeger_ratio_empirical: float = 0.0
    bipartition_a: Tuple[int, ...] = ()
    bipartition_b: Tuple[int, ...] = ()
    spanning_trees: float = 0.0
    regularized_determinant: float = 0.0
    ollivier_ricci_mean: float = 0.0
    ollivier_ricci_min: float = 0.0
    forman_ricci_mean: float = 0.0
    poincare_hopf_index_sum: float = 0.0
    morse_residual_max: float = 0.0
    gauss_bonnet_consistent: bool = True
    morse_holds: bool = True
    birkhoff_certificate: Optional[PoincareBirkhoffCertificate] = None
    melnikov_certificate: Optional[MelnikovChaosCertificate] = None
    lyapunov_spectral: float = 0.0
    celestial_bundle: Optional[CelestialFiltrationBundle] = None


def orient_from_celestial_filtration_bundle(
    bundle: CelestialFiltrationBundle,
    enable_dynamic: bool = True,
    hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
    hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
    saddle_point: Optional[np.ndarray] = None,
) -> Phase2TDADossier:
    r"""
    PRIMER MÉTODO CONSUMIDOR DE `CelestialFiltrationBundle` (continuación de la Fase 1).
    """
    dossier_p1 = bundle.dossier
    adjacency = dossier_p1.adjacency_matrix
    degrees = dossier_p1.degree_vector
    n = len(degrees)
    if n == 0:
        raise ValueError("[TDA_AGENT] La matriz de adyacencia está vacía.")
    spec = SpectralGraphEngine.compute(adjacency, degrees)
    sorted_spectrum = np.asarray(spec["eigenvalues"], dtype=np.float64)
    fiedler_value = float(sorted_spectrum[1]) if len(sorted_spectrum) > 1 else 0.0
    b0_rank, b1_rank, _b2_rank = BettiRecoveryEngine.from_incidence_rank(adjacency)
    b0_persist = len(bundle.essential_h0)
    b1_persist = len(bundle.essential_h1)
    betti_0 = int(b0_rank)
    betti_1 = int(b1_rank)
    if b0_persist and b0_persist != b0_rank:
        logger.debug("β₀ 𝔽₂=%s vs ℝ=%s (torsión posible)", b0_persist, b0_rank)
    if b1_persist != b1_rank:
        logger.debug("β₁ 𝔽₂=%s vs ℝ=%s (torsión posible)", b1_persist, b1_rank)
    num_triangles = _count_triangles(adjacency)
    betti_2 = 0
    euler_char = betti_0 - betti_1 + betti_2
    cheeger = CheegerFormanOllivierEngine.cheeger_bracketing(fiedler_value)
    kirchhoff = KirchhoffMatrixTreeEngine.compute(adjacency, degrees)
    ricci_oll = CheegerFormanOllivierEngine.ollivier_ricci_edges(adjacency)
    ricci_forman = CheegerFormanOllivierEngine.forman_ricci_edges(adjacency)
    gb = GaussBonnetFormanMorseEngine.compute(adjacency, num_triangles, betti_0, betti_1, betti_2)
    birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
    melnikov_cert: Optional[MelnikovChaosCertificate] = None
    lyap = 0.0
    if enable_dynamic:
        birkhoff_cert, melnikov_cert, lyap = FiltrationBirkhoffMelnikovEngine.analyze(
            bundle, hamiltonian_0=hamiltonian_0, hamiltonian_1=hamiltonian_1, saddle_point=saddle_point
        )
    return Phase2TDADossier(
        dossier_p1=dossier_p1,
        normalized_laplacian=spec["L_norm"],
        sorted_spectrum=sorted_spectrum,
        betti_0=betti_0,
        betti_1=betti_1,
        fiedler_value=fiedler_value,
        euler_characteristic=euler_char,
        cheeger_lower_bound=cheeger["h_lower"],
        betti_2=betti_2,
        fiedler_vector=spec["fiedler_vector"],
        cheeger_upper_bound=cheeger["h_upper"],
        cheeger_ratio_empirical=spec["cheeger_ratio_empirical"],
        bipartition_a=tuple(int(x) for x in spec["side_a"]),
        bipartition_b=tuple(int(x) for x in spec["side_b"]),
        spanning_trees=kirchhoff["spanning_trees"],
        regularized_determinant=kirchhoff["regularized_determinant"],
        ollivier_ricci_mean=ricci_oll["mean_curvature"],
        ollivier_ricci_min=ricci_oll["min_curvature"],
        forman_ricci_mean=ricci_forman["mean_curvature"],
        poincare_hopf_index_sum=gb["index_sum_gauss"],
        morse_residual_max=gb["morse_residual_max"],
        gauss_bonnet_consistent=bool(gb["gauss_bonnet_consistent"]),
        morse_holds=bool(gb["morse_holds"]),
        birkhoff_certificate=birkhoff_cert,
        melnikov_certificate=melnikov_cert,
        lyapunov_spectral=lyap,
        celestial_bundle=bundle,
    )


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §2.7 ENLACE TERMINAL FASE 2 → INICIO FASE 3
#      Semilla de recurrencia: lazy random walk nativo del grafo
# ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TDARecurrenceSeed:
    r"""
    OBJETO TERMINAL DE LA FASE 2 Y OBJETO INICIAL DE LA FASE 3.

    Dinámica nativa de Poincaré sobre K: el lazy random walk
        P = (I + D⁺ A)/2,
    que es irreducible en cada componente conexa y aperiódico. El conjunto medible A
    es el lado de Fiedler de mayor volumen (corte de Cheeger). La medida estacionaria
    es π_i = deg(i) / ∑ deg, y Kac predice E[τ_A] = 1/π(A).
    """
    dossier_p2: Phase2TDADossier
    stochastic_matrix: NDArray[np.float64]
    measurable_set: NDArray[np.bool_]
    state_space_size: int
    stationary_measure: NDArray[np.float64]
    kac_stationary_prediction: float
    provenance_hash: str


def seed_tda_recurrence_from_orientation(dossier_p2: Phase2TDADossier) -> TDARecurrenceSeed:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 2 / PRIMER MORFISMO DE LA FASE 3.
    """
    adjacency = dossier_p2.dossier_p1.adjacency_matrix
    n = adjacency.shape[0]
    deg = np.maximum(dossier_p2.dossier_p1.degree_vector, 0.0)
    lazy = np.eye(n, dtype=np.float64) * 0.5
    for i in range(n):
        if deg[i] <= _WILKINSON_LIMIT:
            lazy[i, i] = 1.0
            continue
        lazy[i] += 0.5 * adjacency[i] / deg[i]
    row_sums = lazy.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    stochastic = lazy / row_sums
    measurable = np.zeros(n, dtype=bool)
    if dossier_p2.bipartition_a:
        for idx in dossier_p2.bipartition_a:
            if 0 <= idx < n:
                measurable[idx] = True
    if measurable.sum() == 0 or measurable.sum() == n:
        measurable[:] = False
        measurable[: max(1, n // 2)] = True
    vol = float(np.sum(deg))
    pi = deg / vol if vol > 0 else np.full(n, 1.0 / max(n, 1))
    pi_a = float(pi[measurable].sum())
    kac_stat = 1.0 / pi_a if pi_a > 0 else float("inf")
    digest = hashlib.sha256()
    digest.update(np.array2string(stochastic, precision=6).encode("utf-8"))
    digest.update(measurable.tobytes())
    return TDARecurrenceSeed(
        dossier_p2=dossier_p2,
        stochastic_matrix=stochastic,
        measurable_set=measurable,
        state_space_size=n,
        stationary_measure=pi,
        kac_stationary_prediction=kac_stat,
        provenance_hash=digest.hexdigest(),
    )


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3: ADJUDICACIÓN HEYTING, CROWBAR, RECURRENCIA DE POINCARÉ–KAC Y CERTIFICACIÓN
#         (continúa desde TDARecurrenceSeed)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ───────────────────────────────────────────────────────────────────────────────────────────────
# §3.1 RECURRENCIA DE POINCARÉ–KAC DESDE LA SEMILLA DEL LAZY WALK
# ───────────────────────────────────────────────────────────────────────────────────────────────
def certify_poincare_kac_from_tda_seed(
    seed: TDARecurrenceSeed,
    num_walks: int = 150,
    max_steps: int = 5000,
) -> PoincareRecurrenceCertificate:
    r"""PRIMER MÉTODO CONSUMIDOR DE `TDARecurrenceSeed` (continuación de la Fase 2)."""
    return PoincareRecurrenceEngine.from_stochastic_matrix(
        seed.stochastic_matrix,
        seed.measurable_set,
        num_walks=num_walks,
        max_steps=max_steps,
        seed=42,
    )


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §3.2 CERTIFICADO TERMINAL INMUTABLE
# ───────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TopologicalAnalyzerCertificate:
    """Certificado inmutable de lazo cerrado en el retículo de Heyting Ω₃."""
    betti_0: int
    betti_1: int
    fiedler_value: float
    bottleneck_distance: float
    euler_characteristic: int
    cheeger_bound: float
    heyting_verdict: str
    is_topologically_coherent: bool
    betti_2: int = 0
    cheeger_upper_bound: float = 0.0
    cheeger_ratio_empirical: float = 0.0
    wasserstein_distance_h0: float = 0.0
    wasserstein_distance_h1: float = 0.0
    spanning_trees: float = 0.0
    ollivier_ricci_mean: float = 0.0
    forman_ricci_mean: float = 0.0
    poincare_hopf_index_sum: float = 0.0
    morse_residual: float = 0.0
    gauss_bonnet_consistent: bool = True
    birkhoff_applicable: bool = False
    melnikov_chaos_detected: bool = False
    lyapunov_spectral: float = 0.0
    crowbar_telemetry: Optional[CrowbarPhysicalTelemetry] = None
    recurrence_certificate: Optional[PoincareRecurrenceCertificate] = None
    kac_stationary_prediction: float = 0.0
    verdict_rationale: str = ""
    signature_sha256: str = ""
    timestamp_utc: float = 0.0


class CrowbarDispatchEngine:
    """Delega el transitorio RLC al CrowbarCircuitPhysicsEngine (ESP32 + BT151)."""

    @staticmethod
    def dispatch_if_needed(verdict: HeytingOmega3, rationale: str) -> CrowbarPhysicalTelemetry:
        return CrowbarCircuitPhysicsEngine.simulate_crowbar_actuation(
            trip_required=(verdict == HeytingOmega3.VETOED),
            fault_reason=rationale,
        )


# ───────────────────────────────────────────────────────────────────────────────────────────────
# §3.3 SOBERANO TOPOLÓGICO — ORQUESTADOR OODA ANIDADO
# ───────────────────────────────────────────────────────────────────────────────────────────────
class TopologicalAnalyzerAgent:
    r"""
    Soberano de calibre de lazo cerrado OODA.

      Fase 1 : `observe_phase1` → `lift_observe_to_celestial_filtration_bundle`
      Fase 2 : `orient_from_celestial_filtration_bundle` → `seed_tda_recurrence_from_orientation`
      Fase 3 : `certify_poincare_kac_from_tda_seed` + Heyting Ω₃ + Crowbar + SHA-256.
    """

    def __init__(
        self,
        fiedler_threshold: float = 0.05,
        bottleneck_tolerance: float = 1.0e-4,
        enable_dynamic_analysis: bool = True,
    ) -> None:
        self._fiedler_threshold = float(fiedler_threshold)
        self._bottleneck_tolerance = float(bottleneck_tolerance)
        self._enable_dynamic = bool(enable_dynamic_analysis)

    def observe_phase1(
        self,
        adjacency_matrix: NDArray[np.float64],
        reference_diagram: Optional[NDArray[np.float64]] = None,
    ) -> Phase1TDADossier:
        dossier, _dims, _filt, _pairs, _ess = observe_filtration_dossier(
            adjacency_matrix, reference_diagram, self._bottleneck_tolerance
        )
        return dossier

    def observe_bundle(
        self,
        adjacency_matrix: NDArray[np.float64],
        reference_diagram: Optional[NDArray[np.float64]] = None,
    ) -> CelestialFiltrationBundle:
        dossier, dims, filt, pairs, essential = observe_filtration_dossier(
            adjacency_matrix, reference_diagram, self._bottleneck_tolerance
        )
        return lift_observe_to_celestial_filtration_bundle(dossier, dims, filt, pairs, essential)

    def orient_phase2(self, dossier_p1: Phase1TDADossier) -> Phase2TDADossier:
        bundle = lift_observe_to_celestial_filtration_bundle(dossier_p1)
        return orient_from_celestial_filtration_bundle(bundle, enable_dynamic=self._enable_dynamic)

    def decide_act_phase3(
        self,
        dossier_p2: Phase2TDADossier,
        recurrence_cert: Optional[PoincareRecurrenceCertificate] = None,
        kac_stationary_prediction: float = 0.0,
    ) -> TopologicalAnalyzerCertificate:
        p1 = dossier_p2.dossier_p1
        b0, b1, b2 = dossier_p2.betti_0, dossier_p2.betti_1, dossier_p2.betti_2
        fiedler = dossier_p2.fiedler_value
        d_bottleneck = p1.bottleneck_distance
        chaos_veto = bool(
            dossier_p2.melnikov_certificate.smale_horseshoe_expected
            if dossier_p2.melnikov_certificate is not None
            else False
        )
        arnold = bool(dossier_p2.lyapunov_spectral > 1e-3)
        is_single = b0 == 1
        is_acyclic = b1 == 0
        is_2_acyclic = b2 == 0
        is_fiedler = fiedler >= self._fiedler_threshold
        is_bottleneck = d_bottleneck <= self._bottleneck_tolerance
        is_morse = bool(dossier_p2.morse_holds)
        if is_single and is_acyclic and is_2_acyclic and is_fiedler and is_bottleneck and is_morse and not chaos_veto:
            verdict = HeytingOmega3.COHERENT
            rationale = (
                f"Topología coherente: β₀={b0}, β₁={b1}, λ₂={fiedler:.4f}, "
                f"d_B={d_bottleneck:.2e}, Gauss–Bonnet={dossier_p2.gauss_bonnet_consistent}"
            )
        elif (not is_single) or (not is_acyclic) or (not is_2_acyclic) or chaos_veto:
            verdict = HeytingOmega3.VETOED
            rationale = (
                f"Obstrucción homológica: β₀={b0} (esp. 1), β₁={b1} (esp. 0), β₂={b2}. "
                f"Melnikov={chaos_veto}, Arnold_λ={arnold}"
            )
        else:
            verdict = HeytingOmega3.DEGRADED
            rationale = (
                f"Degradación marginal: λ₂={fiedler:.4f} (umbral {self._fiedler_threshold:.4f}), "
                f"d_B={d_bottleneck:.2e}, Morse={is_morse}"
            )
        if verdict == HeytingOmega3.VETOED:
            logger.error("[TDA_AGENT_VETOED] %s Gatillando ISR IRAM ESP32 / BT151.", rationale)
        crowbar = CrowbarDispatchEngine.dispatch_if_needed(verdict, rationale)
        now = time.time()
        hasher = hashlib.sha256()
        hasher.update(f"{b0}|{b1}|{b2}|{verdict.name}".encode("utf-8"))
        hasher.update(f"{fiedler:.10f}|{d_bottleneck:.10f}".encode("utf-8"))
        hasher.update(f"{dossier_p2.spanning_trees:.6e}|{dossier_p2.forman_ricci_mean:.6e}".encode("utf-8"))
        hasher.update(f"{now:.6f}".encode("utf-8"))
        return TopologicalAnalyzerCertificate(
            betti_0=b0,
            betti_1=b1,
            fiedler_value=fiedler,
            bottleneck_distance=d_bottleneck,
            euler_characteristic=dossier_p2.euler_characteristic,
            cheeger_bound=dossier_p2.cheeger_lower_bound,
            heyting_verdict=verdict.name,
            is_topologically_coherent=bool(verdict == HeytingOmega3.COHERENT),
            betti_2=b2,
            cheeger_upper_bound=dossier_p2.cheeger_upper_bound,
            cheeger_ratio_empirical=dossier_p2.cheeger_ratio_empirical,
            wasserstein_distance_h0=p1.wasserstein_distance_h0,
            wasserstein_distance_h1=p1.wasserstein_distance_h1,
            spanning_trees=dossier_p2.spanning_trees,
            ollivier_ricci_mean=dossier_p2.ollivier_ricci_mean,
            forman_ricci_mean=dossier_p2.forman_ricci_mean,
            poincare_hopf_index_sum=dossier_p2.poincare_hopf_index_sum,
            morse_residual=dossier_p2.morse_residual_max,
            gauss_bonnet_consistent=dossier_p2.gauss_bonnet_consistent,
            birkhoff_applicable=bool(
                dossier_p2.birkhoff_certificate.birkhoff_theorem_applicable
                if dossier_p2.birkhoff_certificate
                else False
            ),
            melnikov_chaos_detected=chaos_veto,
            lyapunov_spectral=dossier_p2.lyapunov_spectral,
            crowbar_telemetry=crowbar,
            recurrence_certificate=recurrence_cert,
            kac_stationary_prediction=kac_stationary_prediction,
            verdict_rationale=rationale,
            signature_sha256=hasher.hexdigest(),
            timestamp_utc=now,
        )

    def audit_topological_persistence_and_betti(
        self,
        adjacency_matrix: NDArray[np.float64],
        reference_diagram: Optional[NDArray[np.float64]] = None,
        stochastic_transition_matrix: Optional[NDArray[np.float64]] = None,
        recurrence_set: Optional[NDArray[np.bool_]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
        saddle_point: Optional[np.ndarray] = None,
    ) -> TopologicalAnalyzerCertificate:
        r"""Ciclo OODA anidado Φ₃ ∘ Φ₂ ∘ Φ₁ sobre la topología presupuestal."""
        dossier, dims, filt, pairs, essential = observe_filtration_dossier(
            adjacency_matrix, reference_diagram, self._bottleneck_tolerance
        )
        bundle = lift_observe_to_celestial_filtration_bundle(dossier, dims, filt, pairs, essential)
        dossier_p2 = orient_from_celestial_filtration_bundle(
            bundle,
            enable_dynamic=self._enable_dynamic,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            saddle_point=saddle_point,
        )
        seed = seed_tda_recurrence_from_orientation(dossier_p2)
        rec_cert: Optional[PoincareRecurrenceCertificate] = None
        try:
            if stochastic_transition_matrix is not None and recurrence_set is not None:
                rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
                    stochastic_transition_matrix, recurrence_set, num_walks=120, max_steps=5000, seed=42
                )
            else:
                rec_cert = certify_poincare_kac_from_tda_seed(seed)
        except Exception as exc:
            logger.debug("[Fase 3] Poincaré–Kac omitted: %s", exc)
        return self.decide_act_phase3(
            dossier_p2,
            recurrence_cert=rec_cert,
            kac_stationary_prediction=seed.kac_stationary_prediction,
        )


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# §3.4 BANCO DE PRUEBAS DE VALIDACIÓN EXPERIMENTAL
# ═══════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("═" * 96)
    print("DEMOSTRACIÓN FORMAL: TOPOLOGICAL ANALYZER AGENT v4.1.0 — FASES ANIDADAS")
    print("VR/FLAG · BOTTLENECK · CHEEGER · KIRCHHOFF · FORMAN/OLLIVIER · GAUSS–BONNET · KAC")
    print("═" * 96)

    agent = TopologicalAnalyzerAgent(
        fiedler_threshold=0.05,
        bottleneck_tolerance=1.0e-4,
        enable_dynamic_analysis=True,
    )

    adj_nominal = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cert_nom = agent.audit_topological_persistence_and_betti(adj_nominal)
    print(f"\n[TEST 1 — CAMINO P₄] Veredicto: {cert_nom.heyting_verdict}")
    print(f"  • β₀={cert_nom.betti_0}, β₁={cert_nom.betti_1}, β₂={cert_nom.betti_2}")
    print(f"  • λ₂={cert_nom.fiedler_value:.4f}  |  h ∈ [{cert_nom.cheeger_bound:.4f}, {cert_nom.cheeger_upper_bound:.4f}]")
    print(f"  • τ(K) árboles generadores       : {cert_nom.spanning_trees:.6e}")
    print(f"  • Forman–Ricci medio             : {cert_nom.forman_ricci_mean:.6f}")
    print(f"  • Ollivier–Ricci medio           : {cert_nom.ollivier_ricci_mean:.6f}")
    print(f"  • Gauss–Bonnet                   : {cert_nom.gauss_bonnet_consistent}")
    print(f"  • Σκ/2 (χ de 1-esqueleto)        : {cert_nom.poincare_hopf_index_sum:.4f}")
    print(f"  • Crowbar                        : {bool(cert_nom.crowbar_telemetry and cert_nom.crowbar_telemetry.interlock_tripped)}")
    print(f"  • SHA-256                        : {cert_nom.signature_sha256[:32]}…")
    assert cert_nom.heyting_verdict == "COHERENT"
    assert cert_nom.betti_0 == 1
    assert cert_nom.betti_1 == 0

    adj_cycle = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cert_cyc = agent.audit_topological_persistence_and_betti(adj_cycle)
    print(f"\n[TEST 2 — CICLO C₄] Veredicto: {cert_cyc.heyting_verdict}")
    print(f"  • β₀={cert_cyc.betti_0}, β₁={cert_cyc.betti_1}")
    print(f"  • τ(K)                           : {cert_cyc.spanning_trees:.6e}")
    print(f"  • Crowbar tripped                : {bool(cert_cyc.crowbar_telemetry and cert_cyc.crowbar_telemetry.interlock_tripped)}")
    if cert_cyc.crowbar_telemetry and cert_cyc.crowbar_telemetry.interlock_tripped:
        print(f"  • Latencia IRAM+BT151            : {cert_cyc.crowbar_telemetry.total_clearance_latency_ns:.2f} ns")
    assert cert_cyc.heyting_verdict == "VETOED"
    assert cert_cyc.betti_1 == 1

    adj_islands = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cert_isl = agent.audit_topological_persistence_and_betti(adj_islands)
    print(f"\n[TEST 3 — ISLAS DESCONECTADAS] Veredicto: {cert_isl.heyting_verdict}")
    print(f"  • β₀={cert_isl.betti_0}, β₁={cert_isl.betti_1}")
    print(f"  • τ(K) (0 si desconexo)          : {cert_isl.spanning_trees:.6e}")
    assert cert_isl.heyting_verdict == "VETOED"
    assert cert_isl.betti_0 == 2

    print("\n[TEST 4 — RECURRENCIA DE POINCARÉ–KAC INYECTADA]")
    rng = np.random.default_rng(2026)
    p_syn = rng.random((6, 6)) + 0.1
    p_syn /= p_syn.sum(axis=1, keepdims=True)
    measurable = np.array([True, False, True, False, False, True])
    cert_kac = agent.audit_topological_persistence_and_betti(
        adj_nominal, stochastic_transition_matrix=p_syn, recurrence_set=measurable
    )
    if cert_kac.recurrence_certificate is not None:
        rec = cert_kac.recurrence_certificate
        print(f"  • |A|/|X|                        : {int(measurable.sum())}/6")
        print(f"  • τ̄_A empírico                   : {rec.mean_return_time_empirical:.4f}")
        print(f"  • Kac 1/μ(A)                     : {rec.kac_lemma_prediction:.4f}")
        print(f"  • Residuo                        : {rec.kac_error_residual:.4f}")

    print("\n[TEST 5 — FILTRACIÓN Y RETORNO (Fase 1)]")
    p1 = agent.observe_phase1(adj_cycle)
    print(f"  • Símplices                      : {p1.num_simplices}")
    print(f"  • Rango ε                        : [{p1.filtration_range[0]:.4f}, {p1.filtration_range[1]:.4f}]")
    if p1.return_map is not None:
        print(f"  • Retornos a Σ                   : {p1.return_map.num_return_points}")
        print(f"  • ρ(P)                           : {p1.return_map.rotation_number:.6f}")
        print(f"  • λ_max / KAM                    : {p1.return_map.lyapunov_max:.6e} / {p1.return_map.kam_stable}")

    print("\n[TEST 6 — ANIDAMIENTO EXPLÍCITO Fase 1 → 2 → 3]")
    dossier_e, dims_e, filt_e, pairs_e, ess_e = observe_filtration_dossier(adj_nominal, None, 1e-4)
    bundle_e = lift_observe_to_celestial_filtration_bundle(dossier_e, dims_e, filt_e, pairs_e, ess_e)
    p2_e = orient_from_celestial_filtration_bundle(bundle_e, enable_dynamic=True)
    seed_e = seed_tda_recurrence_from_orientation(p2_e)
    rec_e = certify_poincare_kac_from_tda_seed(seed_e)
    print(f"  • Fibrado dim / λ₂ proxy         : {bundle_e.configuration_dim} / {bundle_e.dirichlet_energy_fiedler_proxy:.4f}")
    print(f"  • Curva espectral muestras       : {bundle_e.spectral_curve.shape[0]}")
    print(f"  • β₀, β₁ orientados              : {p2_e.betti_0}, {p2_e.betti_1}")
    print(f"  • Semilla |A| / n                : {int(seed_e.measurable_set.sum())}/{seed_e.state_space_size}")
    print(f"  • Kac estacionario 1/π(A)        : {seed_e.kac_stationary_prediction:.4f}")
    print(f"  • Kac empírico (engine)          : {rec_e.mean_return_time_empirical:.4f} vs {rec_e.kac_lemma_prediction:.4f}")

    print("\n" + "═" * 96)
    print("✓ SUITE TDA-POINCARÉ v4.1.0 COMPLETADA.")
    print("  · Fase 1: VR/flag · persistencia 𝔽₂ · bottleneck · Φ(K_ε) → fibrado celeste.")
    print("  · Fase 2: Fiedler · Cheeger · Kirchhoff · Forman/Ollivier · Gauss–Bonnet → semilla.")
    print("  · Fase 3: lazy walk Poincaré–Kac · Heyting Ω₃ · Crowbar ESP32 · SHA-256.")
    print("═" * 96)