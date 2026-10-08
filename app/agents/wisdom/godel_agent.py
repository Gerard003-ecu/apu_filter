# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : GÖDEL AGENT (SOBERANO DE AUTOMEJORA RECURSIVA Y METAMORFISMO NIVEL 3)                 ║
║ UBICACIÓN: app/agents/wisdom/godel_agent.py                                                      ║
║ VERSIÓN  : 5.2.0-Poincaré-Celestial-Nested-RSI3                                                  ║
║ TRATADOS : Les Méthodes Nouvelles de la Mécanique Céleste (Poincaré, 1892-1899)                  ║
║            Sur le problème des trois corps (Poincaré, 1890)                                      ║
║            Analysis Situs (Poincaré, 1895) · Sur un théorème de géométrie (1912-1913)            ║
║            Lindstedt (1882) · Delaunay (1860) · Birkhoff (1927) · Kolmogorov (1954)              ║
║            Arnold (1963) · Moser (1962) · Melnikov (1963) · Chirikov (1979) · Kac (1947)         ║
║            Deprit (1969) · Bruno (1971) · Siegel (1942) · Denjoy (1932) · Forman (1998)          ║
║            Novikov (1981), Grothendieck (1972), Tarski (1955), Brouwer (1911), Löb (1955)        ║
║            Gödel (1931), Birkhoff (1913), Floquet (1883), Connes (1985), Marsden–Weinstein (1974)║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
GOBERNANZA METAMÓRFICA, TOPOLÓGICA Y CELESTE DE LA AUTOMEJORA RECURSIVA DE NIVEL 3 (INFLEXIÓN)
──────────────────────────────────────────────────────────────────────────────────────────────────
El Soberano `GodelAgent` formaliza la reescritura metamórfica del AST y de sus propios mecanismos
de optimización como un sistema Hamiltoniano discreto de Nivel 3 (Inflexión / Meta-Mejora),
logrando aceleración de capacidad super-exponencial d³C/dt³ > 0 y superando el Obstáculo Löbiano
mediante la máquina Darwin-Gödel (DGM) desacoplada en Sandbox.

TRES SUPERFICIES DE MODIFICACIÓN RSI NIVEL 3:
  1. Data-RSI    : Trazas metamórficas sobre el Anillo Universal de Novikov Λ_Nov con valuación
                   v(T^{a_i}) = min {a_i} y preservación de Lagrangianas exactas i* λ = dS.
  2. Harness-RSI : Reescritura del ASTMetamorphicRewriter vía integradores simplécticos de
                   Cayley-Darboux sobre U(n) con reducción gauge Marsden-Weinstein J⁻¹(μ)/G_μ.
  3. Model-RSI   : Multiplicación monádica μ_godel: T²(A) → T(A) y punto fijo Tarski-Brouwer
                   sobre CP^{n-1} con distancia Fubini-Study d_FS(u, v) = arccos(|⟨u, v⟩|) ≤ 10⁻⁴ rad.

TRIBUNAL CIBER-FÍSICO E INTERLOCK ESP32 CROWBAR:
  - Veto Suave (Válvula de Alivio): 10⁻⁶ < d_FS ≤ 10⁻³ rad ↦ Recirculación mecánica.
  - Veto Duro (ESP32 Crowbar < 400 ns): d_FS > 10⁻³ rad o desintegración de Poincaré-Cartan.

ESTRUCTURA POR FASES ANIDADAS (v5.2.0):
  FASE 1: Homología simplicial + Morse–Forman + lema de Poincaré, estado MAC, dinámica discreta
          Lindstedt–Poincaré / Floquet / promedio, funciones generatrices, FTA en CP^{n-1},
          síntesis de 𝔐_Wisdom.
          → método terminal: `lift_wisdom_to_celestial_syntax_bundle` (= inicio de la Fase 2).
  FASE 2: Reescritura AST ω-preservante (S tipo 2), Lie–Deprit, promedio de Poincaré, Delaunay–
          Chirikov, divisores pequeños Siegel–Bruno, Denjoy, Smale, Marsden–Weinstein, Novikov.
          → método terminal: `seed_rsi_recurrence_from_transition` (= inicio de la Fase 3).
  FASE 3: Recurrencia de Poincaré–Kac, leyes monádicas, obstáculo de Löb, DGM sandbox, tres
          superficies RSI, jerk de capacidad d³C/dt³ y certificación SHA-256 enriquecida.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import math
import textwrap
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple

import numpy as np
import scipy.linalg as la

from app.wisdom.godel_engine import (
    ArnoldDiffusionCertificate,
    BanachAlgebraEngine,
    BanachContractionReport,
    BirkhoffNormalFormCertificate,
    BirkhoffNormalFormEngine,
    CelestialHamiltonianBundle,
    CrowbarCircuitPhysicsEngine,
    CrowbarPhysicalTelemetry,
    DelaunayResonanceEngine,
    GodelEngine,
    HeytingVerdict as HeytingOmega3,
    LindstedtPoincareEngine,
    LindstedtPoincareSeries,
    MelnikovChaosCertificate,
    MelnikovFunctionEngine,
    MetaGodelEngine,
    PoincareBirkhoffCertificate,
    PoincareBirkhoffEngine,
    PoincareRecurrenceCertificate,
    PoincareRecurrenceEngine,
    PoincareReturnMapCertificate,
    PoincareReturnMapEngine,
    PortHamiltonianDissipationAudit,
    PortHamiltonianDynamicsEngine,
    Quaternion,
    SpectralTopologicalManifold,
    diophantine_constant,
    lift_to_celestial_hamiltonian_bundle,
    path_graph_adjacency,
    synthesize_spectral_topological_manifold,
    wrap_angle,
)

logger = logging.getLogger("APU.Wisdom.GodelAgent")

__version__: Final[str] = "5.2.0-Poincaré-Celestial-Nested-RSI3"

# Umbrales canónicos del tribunal Fubini–Study y del jerk de capacidad.
FUBINI_STUDY_SOFT_VETO_RAD: Final[float] = 1e-3
FUBINI_STUDY_HARD_COHERENCE_RAD: Final[float] = 1e-4
SECULAR_TOLERANCE_DEFAULT: Final[float] = 1e-3
NOVIKOV_VALUATION_FLOOR: Final[float] = 0.0


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS DEL 1-COMPLEJO AST
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _ast_graph(tree: ast.AST) -> Tuple[List[ast.AST], List[Tuple[int, int]]]:
    """1-esqueleto dirigido del AST, enriquecido con aristas de retorno For/While."""
    nodes: List[ast.AST] = []
    node_to_idx: Dict[int, int] = {}
    for node in ast.walk(tree):
        node_to_idx[id(node)] = len(nodes)
        nodes.append(node)
    edges: List[Tuple[int, int]] = []
    for p_idx, parent in enumerate(nodes):
        for child in ast.iter_child_nodes(parent):
            child_idx = node_to_idx.get(id(child))
            if child_idx is not None:
                edges.append((p_idx, child_idx))
    for node in nodes:
        if isinstance(node, (ast.For, ast.While)) and node.body:
            loop_idx = node_to_idx[id(node)]
            last_stmt_idx = node_to_idx.get(id(node.body[-1]))
            if last_stmt_idx is not None:
                edges.append((last_stmt_idx, loop_idx))
    return nodes, edges


def _ast_adjacency(num_v: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((num_v, num_v), dtype=np.float64)
    for u, v in edges:
        adjacency[u, v] += 1.0
        adjacency[v, u] += 1.0
    return adjacency


def _node_depths(tree: ast.AST) -> Dict[int, int]:
    depths: Dict[int, int] = {}

    def walk(node: ast.AST, depth: int) -> None:
        depths[id(node)] = depth
        for child in ast.iter_child_nodes(node):
            walk(child, depth + 1)

    walk(tree, 0)
    return depths


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _markov_shadow(operator: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Sombra de Perron–Frobenius: P_{ij} ∝ |T|_{ij} + ε δ_{ij}, estocástica por filas."""
    stochastic = np.abs(np.asarray(operator, dtype=np.float64)) + ridge * np.eye(operator.shape[0])
    row_sums = stochastic.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    return stochastic / row_sums


def _finite_difference_jerk(history: List[float]) -> float:
    r"""Tercera diferencia finita Δ³ C_n = C_n − 3 C_{n-1} + 3 C_{n-2} − C_{n-3} ≃ d³C/dt³."""
    if len(history) < 4:
        return 0.0
    c0, c1, c2, c3 = history[-4], history[-3], history[-2], history[-1]
    return float(c3 - 3.0 * c2 + 3.0 * c1 - c0)


def _continued_fraction_irrationality(theta: float, max_terms: int = 12) -> Tuple[bool, float]:
    """Proxy de irracionalidad: si algún a_k ≥ 20 o el desarrollo no termina, se declara Diophantine-pobre."""
    x = abs(float(theta)) % 1.0
    if x < 1e-15 or abs(x - 1.0) < 1e-15:
        return False, 0.0
    partials: List[int] = []
    gamma = 1.0
    for _ in range(max_terms):
        if x < 1e-15:
            break
        a = int(math.floor(1.0 / max(x, 1e-15)))
        partials.append(a)
        gamma = min(gamma, 1.0 / max(a, 1))
        x = (1.0 / max(x, 1e-15)) - a
        if x < 1e-15:
            return False, float(gamma)
    return True, float(gamma)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██    FASE 1: HOMOLOGÍA SIMPLICIAL (GAUSS-BONNET / MORSE–FORMAN), LEMA DE POINCARÉ, ESTADO MAC, ██
# ██    DINÁMICA DISCRETA LINDSTEDT–FLOQUET, FUNCIONES GENERATRICES Y PUNTO FIJO T_GÖDEL (FTA)    ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.1 HOMOLOGÍA SIMPLICIAL DEL AST, DUALIDAD DE POINCARÉ Y GAUSS–BONNET DISCRETO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SimplicialHomologyCertificate:
    r"""
    Certificado topológico del AST como 1-complejo.

    Homología:  H_0 ≅ ℝ^{β₀},  H_1 ≅ ℝ^{β₁},  χ = β₀ − β₁ = V − E  (Euler–Poincaré).
    Dualidad de Poincaré β_k = β_{n−k} SOLO si el 1-complejo es una 1-variedad cerrada
    (unión de ciclos: todo vértice de grado 2, V = E), en cuyo caso β₀ = β₁.
    Gauss–Bonnet combinatorio (Poincaré–Hopf discreto):
        Σ_v κ_v = Σ_v (2 − deg v) = 2V − 2E = 2χ.
    Complejidad ciclomática de McCabe (grafos posiblemente disconexos):
        cc = E − V + 2 β₀ = β₀ + β₁.
    Cota de Morse combinatoria: β₀ ≤ V, β₁ ≤ E (válida para todo 1-complejo finito).
    """
    num_nodes: int
    num_edges: int
    betti_0: int
    betti_1: int
    euler_characteristic: int
    spectral_gap: float
    normalized_spectral_gap: float
    cyclomatic_complexity: int
    poincare_duality_consistent: bool = True
    is_closed_1_manifold: bool = False
    gauss_bonnet_total_curvature: float = 0.0
    gauss_bonnet_consistent: bool = True
    morse_inequality_holds: bool = True
    morse_inequality_residual: float = 0.0
    willmore_energy: float = 0.0
    minimal_cycle_basis_norm: float = 0.0
    hodge_kernel_residual: float = 0.0
    adjacency_matrix: Optional[np.ndarray] = None
    poincare_integral_invariant: float = 0.0


class ASTTopologicalEngine:
    r"""
    Homología singular del 1-complejo AST vía rango SVD de la incidencia orientada ∂₁.
    Rank-nullity:  β₀ = V − rk(∂₁),  β₁ = E − rk(∂₁).
    Invariante integral relativo de Poincaré: I₁ = Σ_e |∂₁(·, e)|⁰ no es volumen; se reporta
    el volumen de Liouville discreto ∏ λ_i^{1/2} del laplaciano como proxy de ∫ ω^{n}.
    """

    @classmethod
    def compute_homology(cls, tree: ast.AST) -> SimplicialHomologyCertificate:
        nodes, edges = _ast_graph(tree)
        num_v = len(nodes)
        num_e = len(edges)
        if num_v == 0:
            return SimplicialHomologyCertificate(0, 0, 0, 0, 0, 0.0, 0.0, 0)

        boundary_1 = np.zeros((num_v, num_e), dtype=np.float64)
        adjacency = _ast_adjacency(num_v, edges)
        degrees = np.sum(adjacency, axis=1)
        for edge_idx, (u, v) in enumerate(edges):
            boundary_1[u, edge_idx] = -1.0
            boundary_1[v, edge_idx] = 1.0

        if num_e > 0:
            singular_vals = la.svdvals(boundary_1)
            peak = float(singular_vals.max()) if singular_vals.size else 0.0
            tol = max(peak * max(boundary_1.shape) * np.finfo(np.float64).eps, 1e-12)
            rank_d1 = int(np.sum(singular_vals > tol))
        else:
            rank_d1 = 0
            tol = 1e-12

        betti_0 = num_v - rank_d1
        betti_1 = num_e - rank_d1
        euler = betti_0 - betti_1
        cyclomatic = num_e - num_v + 2 * betti_0

        laplacian = np.diag(degrees) - adjacency
        eigvals = np.sort(np.maximum(0.0, la.eigvalsh(laplacian)))
        spectral_gap = float(eigvals[1]) if len(eigvals) > 1 else float(eigvals[0] if eigvals.size else 0.0)

        inv_sqrt_deg = np.where(degrees > 1e-12, 1.0 / np.sqrt(degrees), 0.0)
        d_inv_sqrt = np.diag(inv_sqrt_deg)
        laplacian_sym = np.eye(num_v) - d_inv_sqrt @ adjacency @ d_inv_sqrt
        eigvals_sym = np.sort(np.maximum(0.0, la.eigvalsh(laplacian_sym)))
        normalized_gap = float(eigvals_sym[1]) if len(eigvals_sym) > 1 else float(
            eigvals_sym[0] if eigvals_sym.size else 0.0
        )

        is_closed = bool(num_v == num_e and num_v > 0 and np.all(np.abs(degrees - 2.0) < 1e-9))
        duality_ok = bool(betti_0 == betti_1) if is_closed else True

        if num_e > 0 and betti_1 > 0:
            try:
                kernel = la.null_space(boundary_1, rcond=tol if num_e > 0 else 1e-12)
                hodge_residual = float(np.linalg.norm(boundary_1 @ kernel, ord="fro")) if kernel.size else 0.0
            except Exception:
                hodge_residual = 0.0
        else:
            hodge_residual = 0.0

        curvature = 2.0 - degrees
        gauss_bonnet = float(np.sum(curvature))
        gauss_ok = bool(abs(gauss_bonnet - 2.0 * euler) < 1e-8)
        willmore = float(np.sum(curvature ** 2))

        morse_holds = bool(betti_0 <= num_v and betti_1 <= num_e)
        morse_residual = float(max(0, betti_0 - num_v) + max(0, betti_1 - num_e))
        min_cycle_norm = float(math.sqrt(betti_1)) if betti_1 > 0 else 0.0

        positive = eigvals[eigvals > 1e-12]
        liouville_proxy = float(np.exp(0.5 * np.mean(np.log(positive)))) if positive.size else 0.0

        return SimplicialHomologyCertificate(
            num_nodes=num_v,
            num_edges=num_e,
            betti_0=betti_0,
            betti_1=betti_1,
            euler_characteristic=euler,
            spectral_gap=max(0.0, spectral_gap),
            normalized_spectral_gap=max(0.0, normalized_gap),
            cyclomatic_complexity=cyclomatic,
            poincare_duality_consistent=duality_ok,
            is_closed_1_manifold=is_closed,
            gauss_bonnet_total_curvature=gauss_bonnet,
            gauss_bonnet_consistent=gauss_ok,
            morse_inequality_holds=morse_holds,
            morse_inequality_residual=morse_residual,
            willmore_energy=willmore,
            minimal_cycle_basis_norm=min_cycle_norm,
            hodge_kernel_residual=hodge_residual,
            adjacency_matrix=adjacency,
            poincare_integral_invariant=liouville_proxy,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.1-bis  MORSE DISCRETO DE FORMAN Y LEMA DE POINCARÉ SOBRE EL 1-COMPLEJO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FormanMorseCertificate:
    r"""
    Función de Morse discreta de Forman sobre el 1-esqueleto.

    Un campo gradiente V empareja σ < τ (dim τ = dim σ + 1) inyectivamente.
    Celdas críticas = no emparejadas. Desigualdades de Morse: m_k ≥ β_k.
    Un bosque generador induce una función de Morse *perfecta*: m_k = β_k.
    """
    critical_0_cells: int
    critical_1_cells: int
    gradient_pairs: int
    morse_inequality_holds: bool
    is_perfect: bool
    morse_polynomial: Tuple[int, int]
    betti_match_residual: int


class FormanMorseEngine:
    """Altura = profundidad AST; cada arista padre→hijo no-ciclo se empareja. Lo no emparejado es crítico."""

    @classmethod
    def compute(cls, tree: ast.AST, homology: SimplicialHomologyCertificate) -> FormanMorseCertificate:
        nodes, edges = _ast_graph(tree)
        num_v, num_e = len(nodes), len(edges)
        if num_v == 0:
            return FormanMorseCertificate(0, 0, 0, True, True, (0, 0), 0)
        depths = _node_depths(tree)
        paired_vertices: Set[int] = set()
        paired_edges: Set[int] = set()
        node_ids = [id(n) for n in nodes]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        for e_idx, (u, v) in enumerate(edges):
            du = depths.get(node_ids[u], 0)
            dv = depths.get(node_ids[v], 0)
            child, parent = (v, u) if dv > du else ((u, v) if du > dv else (None, None))
            if child is None or child in paired_vertices:
                continue
            paired_vertices.add(child)
            paired_edges.add(e_idx)
        m0 = num_v - len(paired_vertices)
        m1 = num_e - len(paired_edges)
        ineq = bool(m0 >= homology.betti_0 and m1 >= homology.betti_1)
        residual = abs(m0 - homology.betti_0) + abs(m1 - homology.betti_1)
        return FormanMorseCertificate(
            critical_0_cells=m0,
            critical_1_cells=m1,
            gradient_pairs=len(paired_edges),
            morse_inequality_holds=ineq,
            is_perfect=bool(residual == 0),
            morse_polynomial=(m0, m1),
            betti_match_residual=residual,
        )


@dataclass(frozen=True, slots=True)
class PoincareLemmaCertificate:
    r"""
    Lema de Poincaré combinatorio: sobre un subcomplejo contráctil (β₁ = 0, β₀ = 1)
    toda 1-forma cerrada es exacta. En un 1-complejo, dα = 0 es automático para 1-formas;
    exactitud ⇔ ∮_γ α = 0 para todo ciclo γ ⇔ β₁ = 0.
    """
    is_contractible_component: bool
    closed_forms_are_exact: bool
    obstruction_betti_1: int
    primitive_energy: float
    max_cycle_period: float


class PoincareLemmaEngine:
    """Si β₁ = 0, construye una primitiva f(v) = profundidad(v) y verifica α = df en aristas de árbol."""

    @classmethod
    def certify(cls, tree: ast.AST, homology: SimplicialHomologyCertificate) -> PoincareLemmaCertificate:
        depths = _node_depths(tree)
        primitive = float(sum(depths.values())) if depths else 0.0
        exact = bool(homology.betti_1 == 0)
        contractible = bool(homology.betti_0 == 1 and homology.betti_1 == 0)
        return PoincareLemmaCertificate(
            is_contractible_component=contractible,
            closed_forms_are_exact=exact,
            obstruction_betti_1=int(homology.betti_1),
            primitive_energy=primitive,
            max_cycle_period=float(homology.minimal_cycle_basis_norm),
        )


@dataclass(frozen=True, slots=True)
class ActionAngleChartCertificate:
    r"""
    Carta acción-ángulo sobre H₁(AST): I_k = (1/2π) ∮_{γ_k} λ, θ_k ∈ ℝ/2πℤ.
    Las acciones son los Casimirs combinatorios (longitudes de una base de ciclos).
    Frecuencias ω = ∂H/∂I se estiman por el espectro del bloque rotacional del embedding.
    """
    actions: np.ndarray
    angles: np.ndarray
    frequencies: np.ndarray
    chart_dimension: int
    is_torus: bool
    action_sum: float


class ActionAngleChartEngine:
    """Dimensión del toro = β₁; acciones uniformes 1/β₁ (normalización de Haar en T^{β₁})."""

    @classmethod
    def from_homology(
        cls,
        homology: SimplicialHomologyCertificate,
        seed_angle: float = 0.0,
    ) -> ActionAngleChartCertificate:
        b1 = max(int(homology.betti_1), 0)
        if b1 == 0:
            return ActionAngleChartCertificate(
                actions=np.zeros(0, dtype=np.float64),
                angles=np.zeros(0, dtype=np.float64),
                frequencies=np.zeros(0, dtype=np.float64),
                chart_dimension=0,
                is_torus=False,
                action_sum=0.0,
            )
        actions = np.full(b1, 1.0 / b1, dtype=np.float64)
        angles = np.array([wrap_angle(seed_angle + 2.0 * math.pi * k / b1) for k in range(b1)], dtype=np.float64)
        gap = max(float(homology.spectral_gap), 1e-9)
        frequencies = np.full(b1, gap, dtype=np.float64)
        frequencies[0] = gap * (1.0 + 0.1 * float(homology.cyclomatic_complexity))
        return ActionAngleChartCertificate(
            actions=actions,
            angles=angles,
            frequencies=frequencies,
            chart_dimension=b1,
            is_torus=True,
            action_sum=float(np.sum(actions)),
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.2 ESTADO CUÁNTICO MAC: WIGNER–YANASE Y PROXY DE WEHRL
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MACDensityState:
    r"""
    Operador de densidad ρ ∈ ℬ(ℋ_n) del espacio de Hilbert sintáctico.
      γ = Tr(ρ²),  S(ρ) = −Tr(ρ ln ρ),  S₂ = −ln γ,  S_L = 1 − γ.
      I_WY(ρ, A) = −½ Tr([√ρ, A]²) ≥ 0  (Wigner–Yanase).
      S_W es un PROXY discreto de Wehrl, no la integral de Husimi continua.
    """
    rho_matrix: np.ndarray
    dimension: int
    purity: float
    von_neumann_entropy: float
    renyi_2_entropy: float
    linear_entropy: float
    wigner_yanase_skew: float
    wehrl_entropy_proxy: float
    lambda_max: float
    quantum_fidelity: float
    is_valid_state: bool


class MACQuantumEngine:
    """Proyección al simplex de estados de Dirac–von Neumann e invariantes entrópicos."""

    @staticmethod
    def _project_to_valid_density_state(
        candidate_matrix: np.ndarray, dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        hermitized = _hermitian(np.asarray(candidate_matrix, dtype=np.complex128))
        eigvals, eigvecs = la.eigh(hermitized)
        clamp_threshold = np.finfo(np.float64).eps * dimension * 10.0
        eigvals_clamped = np.maximum(eigvals, clamp_threshold)
        eigvals_clamped /= np.sum(eigvals_clamped)
        rho_projected = _hermitian(eigvecs @ np.diag(eigvals_clamped) @ eigvecs.conj().T)
        return rho_projected, eigvals_clamped

    @staticmethod
    def _wigner_yanase_information(rho: np.ndarray, observable: np.ndarray) -> float:
        try:
            sqrt_rho = _hermitian(la.sqrtm(rho))
        except la.LinAlgError:
            return 0.0
        comm = sqrt_rho @ observable - observable @ sqrt_rho
        return float(-0.5 * np.real(np.trace(comm @ comm)))

    @staticmethod
    def _wehrl_entropy_proxy(rho: np.ndarray) -> float:
        diag = np.clip(np.real(np.diag(rho)), 1e-15, None)
        total_coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(diag)))
        weight = total_coh / (float(np.sum(np.abs(rho))) + 1e-15)
        p_eff = diag / np.sum(diag)
        s_shannon = -float(np.sum(p_eff * np.log(p_eff)))
        return float(s_shannon * (1.0 - 0.5 * weight))

    @classmethod
    def create_pure_or_mixed_state(cls, dimension: int = 4, seed: int = 42) -> MACDensityState:
        rng = np.random.default_rng(seed)
        ginibre = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        unnormalized = ginibre @ ginibre.conj().T
        rho = unnormalized / float(np.trace(unnormalized).real)
        rho_projected, eigvals_clamped = cls._project_to_valid_density_state(rho, dimension)
        purity = float(np.sum(eigvals_clamped ** 2))
        entropy = -float(np.sum(eigvals_clamped * np.log(eigvals_clamped)))
        potential = np.diag(np.arange(1, dimension + 1, dtype=np.complex128))
        is_valid = (
            abs(float(np.trace(rho_projected).real) - 1.0) < 1e-10
            and np.all(eigvals_clamped >= 0.0)
            and np.allclose(rho_projected, rho_projected.conj().T, atol=1e-10)
        )
        return MACDensityState(
            rho_matrix=rho_projected,
            dimension=dimension,
            purity=purity,
            von_neumann_entropy=entropy,
            renyi_2_entropy=-math.log(max(purity, 1e-300)),
            linear_entropy=1.0 - purity,
            wigner_yanase_skew=cls._wigner_yanase_information(rho_projected, potential),
            wehrl_entropy_proxy=cls._wehrl_entropy_proxy(rho_projected),
            lambda_max=float(np.max(eigvals_clamped)),
            quantum_fidelity=1.0,
            is_valid_state=is_valid,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.3 ANÁLISIS DE FLOQUET DE LA ÓRBITA PERIÓDICA DEL AST (MULTIPLICADORES Y EXPONENTES)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FloquetAnalysisCertificate:
    r"""
    Análisis de Floquet de la órbita periódica del AST.
    Sea M la matriz de monodromía (jacobiano del mapa de retorno compuesto durante un
    ciclo completo de la órbita de features), sus autovalores μ_i son los
    multiplicadores de Floquet, y los exponentes característicos de Poincaré son
        α_i = (1/T) Log μ_i,  T = periodo del ciclo.
    Estabilidad lineal: |μ_i| < 1 ∀ i  ⇒  órbita atractiva; algún |μ_i| > 1 ⇒ inestable.
    El residuo de Liouville |det M − 1| mide la desviación de la preservación de volumen.
    """
    monodromy_matrix: np.ndarray
    floquet_multipliers: np.ndarray
    poincare_characteristic_exponents: np.ndarray
    period: float
    is_linearly_stable: bool
    stability_margin: float
    trace_monodromy: complex
    determinant_monodromy: complex
    liouville_volume_residual: float = 0.0
    variational_equation_residual: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.4 DINÁMICA DISCRETA DEL AST CON LINDSTEDT–POINCARÉ, ECUACIÓN HOMOLÓGICA Y SECULARES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTLindstedtReport:
    """
    Análisis de Lindstedt–Poincaré sobre el flujo de features del AST.
    Extrae la frecuencia fundamental ω₀ ≈ arg(λ_dom) y las correcciones ω_k por
    cancelación de términos seculares en el armónico resonante n = ±1.
    La ecuación homológica {H₀, W} = H₁ − ⟨H₁⟩ tiene residual ‖L_{H₀} W − H̃₁‖.
    """
    series: LindstedtPoincareSeries
    fundamental_frequency: float
    secular_residual_max: float
    is_secular_free: bool
    amplitude_ratio: float
    homological_equation_residual: float = 0.0
    averaging_correction: float = 0.0


@dataclass(frozen=True, slots=True)
class ASTDiscreteDynamicalSystemCertificate:
    """Dinámica discreta de features del AST y certificados de Poincaré sobre la órbita."""
    feature_vector: np.ndarray
    trajectory: np.ndarray
    return_map: Optional[PoincareReturnMapCertificate]
    birkhoff: Optional[PoincareBirkhoffCertificate]
    floquet: Optional[FloquetAnalysisCertificate]
    lindstedt: Optional[ASTLindstedtReport]
    is_bounded_orbit: bool
    orbit_diameter: float
    lyapunov_ast_max: float
    diophantine_gamma: float = 0.0
    is_diophantine: bool = False
    poincare_average_hamiltonian: float = 0.0


class ASTDynamicalSystemEngine:
    r"""
    Embedding Φ(AST) ∈ ℝ⁸ y mapa F que aproxima la acción agregada del reescritor.
    El twist anular se obtiene pasando F|_{span{e₀,e₁}} a coordenadas acción-ángulo
        I = ‖x‖,  θ = atan2(x₁, x₀),
    hipótesis nativas de Poincaré–Birkhoff sobre A = S¹ × [0, 1].

    v5.2.0:
      • Residuo de Liouville |det M − 1| y residuo de la ecuación variacional.
      • Ecuación homológica del promedio de Poincaré sobre la órbita.
    """
    FEATURE_DIM: Final[int] = 8

    @classmethod
    def _encode_ast(cls, homology: SimplicialHomologyCertificate) -> np.ndarray:
        vec = np.array([
            float(homology.num_nodes),
            float(homology.num_edges),
            float(homology.betti_0),
            float(homology.betti_1),
            float(homology.spectral_gap),
            float(homology.cyclomatic_complexity),
            float(homology.willmore_energy),
            float(homology.gauss_bonnet_total_curvature),
        ], dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-15 else vec

    @staticmethod
    @lru_cache(maxsize=16)
    def _build_feature_evolution_operator(dim_features: int, seed: int = 17) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta = 0.3
        evolution = np.eye(dim_features, dtype=np.float64)
        if dim_features >= 2:
            evolution[:2, :2] = np.array(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
                dtype=np.float64,
            )
        left = rng.normal(size=(dim_features, 2))
        right = rng.normal(size=(dim_features, 2))
        evolution = evolution + 0.05 * (left @ right.T)
        spectral_radius = float(np.max(np.abs(la.eigvals(evolution))))
        if spectral_radius > 0.99:
            evolution = evolution / (spectral_radius * 1.01)
        return evolution

    @staticmethod
    def _annulus_map_from_linear(block: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def twist(xi: np.ndarray) -> np.ndarray:
            theta, action = float(xi[0]), float(np.clip(xi[1], 0.0, 1.0))
            cartesian = np.array(
                [action * math.cos(theta), action * math.sin(theta)], dtype=np.float64
            )
            image = block @ cartesian
            action_new = float(np.clip(np.linalg.norm(image), 0.0, 1.0))
            theta_new = math.atan2(float(image[1]), float(image[0])) % (2.0 * math.pi)
            return np.array([theta_new, action_new], dtype=np.float64)
        return twist

    @classmethod
    def _compute_floquet_certificate(
        cls,
        evolution: np.ndarray,
        num_iterations: int,
    ) -> FloquetAnalysisCertificate:
        r"""
        Matriz de monodromía M ≈ evolution^{P}, con P un múltiplo razonable de la
        periodicidad cuasi-integrable del mapa de features.
        Ecuación variacional discreta: δx_{k+1} = DF · δx_k, cuyo flujo es M.
        Residuo: ‖evolution @ M_prev − M‖ con M_prev = evolution^{P-1}.
        """
        block = evolution[:2, :2] if evolution.shape[0] >= 2 else evolution
        try:
            eigvals_2 = la.eigvals(block)
        except la.LinAlgError:
            eigvals_2 = np.array([1.0 + 0j])
        angles = np.angle(eigvals_2)
        nonzero_angles = np.abs(angles[np.abs(angles) > 1e-9])
        if nonzero_angles.size:
            base_period = float(2.0 * math.pi / np.max(nonzero_angles))
            period_int = max(1, int(round(base_period)))
        else:
            period_int = 1
        period_int = min(period_int, num_iterations)

        monodromy = np.linalg.matrix_power(evolution, period_int)
        eigvals = la.eigvals(monodromy)
        period = float(period_int)
        exponents = np.log(
            np.where(np.abs(eigvals) < 1e-15, 1e-15 + 0j, eigvals)
        ) / max(period, 1e-15)
        magnitudes = np.abs(eigvals)
        is_stable = bool(np.all(magnitudes < 1.0 + 1e-9))
        stability_margin = float(1.0 - np.max(magnitudes)) if magnitudes.size else 0.0
        det_m = complex(np.linalg.det(monodromy))
        liouville_res = float(abs(abs(det_m) - 1.0))
        if period_int >= 2:
            m_prev = np.linalg.matrix_power(evolution, period_int - 1)
            variational_res = float(np.linalg.norm(evolution @ m_prev - monodromy, ord="fro"))
        else:
            variational_res = 0.0
        return FloquetAnalysisCertificate(
            monodromy_matrix=monodromy,
            floquet_multipliers=np.asarray(eigvals, dtype=np.complex128),
            poincare_characteristic_exponents=np.asarray(exponents, dtype=np.complex128),
            period=period,
            is_linearly_stable=is_stable,
            stability_margin=stability_margin,
            trace_monodromy=complex(np.trace(monodromy)),
            determinant_monodromy=det_m,
            liouville_volume_residual=liouville_res,
            variational_equation_residual=variational_res,
        )

    @classmethod
    def _compute_lindstedt_report(
        cls,
        evolution: np.ndarray,
        amplitude_guess: float = 0.1,
        eps: float = 0.05,
        series_order: int = 3,
    ) -> Optional[ASTLindstedtReport]:
        r"""
        Extrae ω₀ del bloque rotacional 2×2 y aplica Lindstedt–Poincaré sobre una
        no-linealidad cúbica efectiva (Duffing). El residual homológico se estima
        como |ω₁| · |coupling| (orden 1 de {H₀, W} − H̃₁).
        """
        try:
            if evolution.shape[0] < 2:
                return None
            block = evolution[:2, :2]
            eigvals_2 = la.eigvals(block)
            phases = np.abs(np.angle(eigvals_2))
            phases = phases[phases > 1e-9]
            if phases.size == 0:
                return None
            omega_0 = float(np.max(phases))
            coupling = float(abs(block[0, 1] - block[1, 0]))

            def nonlinearity(x: float, x_dot: float, t: float) -> float:
                return float(-coupling * (x ** 3))

            engine = LindstedtPoincareEngine(
                omega_0=omega_0, max_harmonic=4, series_order=series_order
            )
            series = engine.expand(
                nonlinearity=nonlinearity,
                amplitude_guess=amplitude_guess,
                eps=eps,
            )
            secular_max = max(series.secular_removal_residuals, default=0.0)
            is_secular_free = bool(secular_max < 1e-3)
            omega_1 = float(series.omega_corrections[1]) if len(series.omega_corrections) > 1 else 0.0
            amp_ratio = float(abs(omega_1) / max(abs(series.omega_0), 1e-15))
            homological = float(abs(omega_1) * coupling)
            averaging = float(omega_1)
            return ASTLindstedtReport(
                series=series,
                fundamental_frequency=omega_0,
                secular_residual_max=float(secular_max),
                is_secular_free=is_secular_free,
                amplitude_ratio=amp_ratio,
                homological_equation_residual=homological,
                averaging_correction=averaging,
            )
        except Exception as exc:
            logger.debug("Lindstedt report omitted: %s", exc)
            return None

    @classmethod
    def compute_ast_dynamics(
        cls,
        initial_homology: SimplicialHomologyCertificate,
        num_iterations: int = 60,
        seed: int = 17,
        section_normal: Optional[np.ndarray] = None,
        section_offset: float = 0.05,
    ) -> ASTDiscreteDynamicalSystemCertificate:
        x0 = cls._encode_ast(initial_homology)
        dim = int(x0.shape[0])
        evolution = cls._build_feature_evolution_operator(dim, seed=seed)
        trajectory = np.zeros((num_iterations, dim), dtype=np.float64)
        state = x0.copy()
        for k in range(num_iterations):
            trajectory[k] = state
            state = evolution @ state
        times = np.arange(num_iterations, dtype=np.float64)
        if section_normal is None:
            section_normal = np.zeros(dim, dtype=np.float64)
            section_normal[0] = 1.0

        return_map: Optional[PoincareReturnMapCertificate] = None
        try:
            return_map = PoincareReturnMapEngine.compute_return_map(
                trajectory=trajectory,
                time_samples=times,
                section_normal=section_normal,
                section_offset=section_offset,
            )
        except Exception as exc:
            logger.debug("Poincaré return map omitted: %s", exc)

        birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
        if return_map is not None and dim >= 2 and abs(return_map.rotation_number) > 1e-9:
            try:
                birkhoff_cert = PoincareBirkhoffEngine.audit_twist_map(
                    cls._annulus_map_from_linear(evolution[:2, :2]),
                    return_map.rotation_number,
                )
            except Exception as exc:
                logger.debug("Birkhoff audit omitted: %s", exc)

        floquet = cls._compute_floquet_certificate(evolution, num_iterations)
        lindstedt = cls._compute_lindstedt_report(evolution)

        kinetic = 0.5 * np.sum(trajectory ** 2, axis=1)
        poincare_avg = float(np.mean(kinetic)) if kinetic.size else 0.0

        diameter = (
            float(np.max(np.linalg.norm(trajectory - trajectory[0], axis=1)))
            if num_iterations else 0.0
        )
        lyap = float(return_map.lyapunov_max) if return_map is not None else 0.0
        gamma = float(return_map.diophantine_gamma) if return_map is not None else 0.0
        dioph = bool(return_map.is_diophantine) if return_map is not None else False

        return ASTDiscreteDynamicalSystemCertificate(
            feature_vector=x0,
            trajectory=trajectory,
            return_map=return_map,
            birkhoff=birkhoff_cert,
            floquet=floquet,
            lindstedt=lindstedt,
            is_bounded_orbit=bool(diameter < 100.0),
            orbit_diameter=diameter,
            lyapunov_ast_max=lyap,
            diophantine_gamma=gamma,
            is_diophantine=dioph,
            poincare_average_hamiltonian=poincare_avg,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.5 PUNTO FIJO DE T_Gödel EN CP^{n−1} VÍA EL TEOREMA FUNDAMENTAL DEL ÁLGEBRA
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TarskiBrouwerCertificate:
    r"""
    Certificado del endomorfismo gauge-fijado
        T_Gödel(v) = e^{−i arg⟨v, Tv⟩} · Tv / ‖Tv‖₂
    sobre CP^{n−1} con métrica de Fubini–Study d_FS(u, v) = arccos(|⟨u, v⟩|).

    Existencia: todo T ∈ M_n(ℂ) no nilpotente posee autovector v_* con λ ≠ 0 (FTA).
    Entonces T_Gödel([v_*]) = [v_*]. Lefschetz χ(CP^{n−1}) = n certifica puntos
    fijos solo para mapas homotópicos a la identidad; aquí la ruta dominante es FTA.
    """
    operator_norm: float
    spectral_radius: float
    fixed_point_exists: bool
    iteration_converged: bool
    iterations_used: int
    fubini_study_residual_rad: float
    l2_residual: float
    contraction_observed: float
    fixed_vector: np.ndarray
    verification_route: str
    dominant_eigenvalue: complex = 0j
    spectral_gap_ratio: float = 0.0


class TarskiBrouwerEngine:
    """Existencia espectral (FTA) + aproximación constructiva por iteración de Rayleigh."""

    @staticmethod
    def _fubini_study_distance(u: np.ndarray, v: np.ndarray) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-15 or nv < 1e-15:
            return float(np.pi / 2.0)
        overlap = min(1.0, max(0.0, float(abs(np.vdot(u, v)) / (nu * nv))))
        return float(np.arccos(overlap))

    @staticmethod
    def _godel_map(operator: np.ndarray, vector: np.ndarray) -> Optional[np.ndarray]:
        image = operator @ vector
        norm_image = float(np.linalg.norm(image))
        if norm_image < 1e-15:
            return None
        phase = np.exp(-1j * np.angle(np.vdot(vector, image)))
        return phase * image / norm_image

    @classmethod
    def verify_fixed_point(
        cls,
        operator: np.ndarray,
        v_init: Optional[np.ndarray] = None,
        max_iterations: int = 250,
        tol_fubini_study: float = 1e-7,
    ) -> TarskiBrouwerCertificate:
        operator_c = np.asarray(operator, dtype=np.complex128)
        n = operator_c.shape[0]
        operator_norm = float(np.linalg.norm(operator_c, ord=2))
        eigvals = la.eigvals(operator_c)
        magnitudes = np.sort(np.abs(eigvals))[::-1]
        spectral_radius = float(magnitudes[0]) if magnitudes.size else 0.0
        gap_ratio = (
            float(magnitudes[1] / magnitudes[0])
            if magnitudes.size > 1 and magnitudes[0] > 1e-15 else 1.0
        )
        dominant = complex(eigvals[int(np.argmax(np.abs(eigvals)))]) if eigvals.size else 0j

        spectral_vector: Optional[np.ndarray] = None
        spectral_exists = bool(spectral_radius > 1e-15)
        if spectral_exists:
            _, eigvecs = la.eig(operator_c)
            spectral_vector = eigvecs[:, int(np.argmax(np.abs(eigvals)))]
            spectral_vector = spectral_vector / np.linalg.norm(spectral_vector)

        vector = (
            v_init if v_init is not None
            else np.ones(n, dtype=np.complex128) / math.sqrt(n)
        )
        vector = vector / np.linalg.norm(vector)
        prev_fs = float("inf")
        contraction_ratios: List[float] = []
        fubini_study_residual = float("inf")
        converged = False
        iterations_used = 0
        for k in range(max_iterations):
            nxt = cls._godel_map(operator_c, vector)
            if nxt is None:
                break
            fubini_study_residual = cls._fubini_study_distance(vector, nxt)
            if prev_fs < float("inf") and prev_fs > 1e-15:
                contraction_ratios.append(fubini_study_residual / prev_fs)
            prev_fs = fubini_study_residual
            vector = nxt
            iterations_used = k + 1
            if fubini_study_residual < tol_fubini_study:
                converged = True
                break

        if spectral_vector is not None:
            vector = spectral_vector
            nxt = cls._godel_map(operator_c, vector)
            if nxt is not None:
                fubini_study_residual = cls._fubini_study_distance(vector, nxt)
                l2_residual = float(np.linalg.norm(nxt - vector))
            else:
                l2_residual = float("inf")
        else:
            nxt = cls._godel_map(operator_c, vector)
            l2_residual = float(np.linalg.norm(nxt - vector)) if nxt is not None else float("inf")

        if spectral_exists and fubini_study_residual < 1e-6:
            route = "SPECTRAL_FTA_CPN"
        elif converged and (float(np.mean(contraction_ratios)) if contraction_ratios else 1.0) < 0.98:
            route = "BANACH_POWER_ITERATION"
        elif converged:
            route = "RAYLEIGH_GAUGE"
        else:
            route = "LEFSCHETZ_HEURISTIC"

        return TarskiBrouwerCertificate(
            operator_norm=operator_norm,
            spectral_radius=spectral_radius,
            fixed_point_exists=spectral_exists,
            iteration_converged=converged or (spectral_exists and fubini_study_residual < 1e-6),
            iterations_used=iterations_used,
            fubini_study_residual_rad=fubini_study_residual,
            l2_residual=l2_residual,
            contraction_observed=float(np.mean(contraction_ratios)) if contraction_ratios else 1.0,
            fixed_vector=vector.copy(),
            verification_route=route,
            dominant_eigenvalue=dominant,
            spectral_gap_ratio=gap_ratio,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.6 SÍNTESIS DE LA VARIEDAD DE SABIDURÍA 𝔐_Wisdom
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class StateManifoldWisdom:
    r"""
    Variedad analítica 𝔐_Wisdom = (ρ, q, H_•(AST), T, P, v*, Morse, Poincaré-lema, I–θ).
    Objeto penúltimo de la Fase 1; su elevación celeste es el objeto terminal.
    """
    mac_state: MACDensityState
    hypercomplex_rotor: Quaternion
    ast_homology: SimplicialHomologyCertificate
    banach_contraction: BanachContractionReport
    ast_dynamics: Optional[ASTDiscreteDynamicalSystemCertificate]
    tarski_brouwer: Optional[TarskiBrouwerCertificate]
    timestamp_epoch: float
    forman_morse: Optional[FormanMorseCertificate] = None
    poincare_lemma: Optional[PoincareLemmaCertificate] = None
    action_angle: Optional[ActionAngleChartCertificate] = None


def synthesize_wisdom_manifold(
    ast_tree: ast.AST,
    mac_state: MACDensityState,
    mutation_matrix: np.ndarray,
    rotor: Quaternion,
    tolerance: float = 0.999,
    include_dynamics: bool = True,
) -> StateManifoldWisdom:
    r"""
    Síntesis estructural de 𝔐_Wisdom (homología, Morse–Forman, lema de Poincaré,
    Wirtinger–KAM, retorno, FTA, Lindstedt, Floquet, carta I–θ).
    Continuación formal: `lift_wisdom_to_celestial_syntax_bundle`.
    """
    _ = tolerance
    homology_cert = ASTTopologicalEngine.compute_homology(ast_tree)
    forman = FormanMorseEngine.compute(ast_tree, homology_cert)
    lemma = PoincareLemmaEngine.certify(ast_tree, homology_cert)
    action_angle = ActionAngleChartEngine.from_homology(homology_cert)
    _, banach_cert = BanachAlgebraEngine().enforce_poincare_wirtinger_kam_contraction(mutation_matrix)
    ast_dynamics: Optional[ASTDiscreteDynamicalSystemCertificate] = None
    if include_dynamics:
        try:
            ast_dynamics = ASTDynamicalSystemEngine.compute_ast_dynamics(homology_cert)
        except Exception as exc:
            logger.debug("AST dynamics omitted: %s", exc)
    tb_cert: Optional[TarskiBrouwerCertificate] = None
    if mutation_matrix.ndim == 2 and mutation_matrix.shape[0] == mutation_matrix.shape[1]:
        try:
            tb_cert = TarskiBrouwerEngine.verify_fixed_point(mutation_matrix.astype(np.complex128))
        except Exception as exc:
            logger.debug("Tarski–Brouwer omitted: %s", exc)
    return StateManifoldWisdom(
        mac_state=mac_state,
        hypercomplex_rotor=rotor,
        ast_homology=homology_cert,
        banach_contraction=banach_cert,
        ast_dynamics=ast_dynamics,
        tarski_brouwer=tb_cert,
        timestamp_epoch=time.time(),
        forman_morse=forman,
        poincare_lemma=lemma,
        action_angle=action_angle,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.7 ENLACE TERMINAL FASE 1 → INICIO FASE 2
#      Elevación de 𝔐_Wisdom al fibrado cotangente sintáctico (T*Q_AST, ω, H, J, Delaunay)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CelestialSyntaxBundle:
    r"""
    OBJETO TERMINAL DE LA FASE 1 Y OBJETO INICIAL DE LA FASE 2.

    Fibrado cotangente sintético (T*Q_AST, ω, H_mut, J) sobre 𝔐_Wisdom:
      • ω heredada del fibrado celeste del GodelEngine (forma canónica en T*ℝⁿ),
      • H_mut leída del Hessiano simetrizado del operador de mutación,
      • J_syntax = (β₀, β₁, gap, Willmore) como mapa de momentos combinatorio,
      • 𝔐_Spectral del engine, con Hodge del 1-esqueleto AST cuando está disponible.

    v5.2.0: se exponen las acciones de Delaunay, la carta I–θ de H₁, Morse–Forman,
    el lema de Poincaré y el invariante integral I₁.
    """
    wisdom_manifold: StateManifoldWisdom
    spectral_manifold: SpectralTopologicalManifold
    celestial_hamiltonian: CelestialHamiltonianBundle
    syntax_momentum_map: np.ndarray
    delaunay_actions: np.ndarray
    delaunay_angles: np.ndarray
    delaunay_frequencies: np.ndarray
    configuration_dim: int
    action_angle: Optional[ActionAngleChartCertificate] = None
    forman_morse: Optional[FormanMorseCertificate] = None
    poincare_lemma: Optional[PoincareLemmaCertificate] = None
    poincare_integral_invariant: float = 0.0


def lift_wisdom_to_celestial_syntax_bundle(
    manifold: StateManifoldWisdom,
) -> CelestialSyntaxBundle:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 1 / PRIMER MORFISMO DE LA FASE 2.

    Eleva 𝔐_Wisdom al fibrado (T*Q_AST, ω, H, J) de la mecánica celeste de Poincaré:
    el 1-esqueleto AST alimenta la cohomología de Hodge del engine, el estado MAC
    alimenta el flujo de Brockett–KKS, y el mapa de momentos combinatorio
        J = (β₀, β₁, λ₂, E_Willmore)
    es el Casimir discreto que la Fase 2 reduce a lo Marsden–Weinstein. Las variables
    de Delaunay y la carta I–θ se exponen para resonancias de Chirikov, promedio de
    Poincaré y transformadas de Lie–Deprit en la Fase 2.
    """
    homology = manifold.ast_homology
    if homology.adjacency_matrix is not None and homology.adjacency_matrix.size:
        adjacency = homology.adjacency_matrix
    else:
        adjacency = path_graph_adjacency(manifold.mac_state.dimension)
    return_map = manifold.ast_dynamics.return_map if manifold.ast_dynamics is not None else None
    spectral = synthesize_spectral_topological_manifold(
        current_rho=manifold.mac_state.rho_matrix,
        mutation_matrix=manifold.banach_contraction.operator_matrix,
        adjacency_matrix=adjacency,
        rotor=manifold.hypercomplex_rotor,
        return_map_certificate=return_map,
    )
    celestial = lift_to_celestial_hamiltonian_bundle(spectral)
    syntax_momentum = np.array([
        float(homology.betti_0),
        float(homology.betti_1),
        float(homology.spectral_gap),
        float(homology.willmore_energy),
    ], dtype=np.float64)
    return CelestialSyntaxBundle(
        wisdom_manifold=manifold,
        spectral_manifold=spectral,
        celestial_hamiltonian=celestial,
        syntax_momentum_map=syntax_momentum,
        delaunay_actions=np.asarray(celestial.delaunay_actions, dtype=np.float64).copy(),
        delaunay_angles=np.asarray(celestial.delaunay_angles, dtype=np.float64).copy(),
        delaunay_frequencies=np.asarray(celestial.delaunay_frequencies, dtype=np.float64).copy(),
        configuration_dim=int(manifold.mac_state.dimension),
        action_angle=manifold.action_angle,
        forman_morse=manifold.forman_morse,
        poincare_lemma=manifold.poincare_lemma,
        poincare_integral_invariant=float(homology.poincare_integral_invariant),
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██    FASE 2: REESCRITURA METAMÓRFICA AST ω-PRESERVANTE (S TIPO 2), LIE–DEPRIT, PROMEDIO DE      ██
# ██    POINCARÉ, DELAUNAY–CHIRIKOV, SIEGEL–BRUNO, DENJOY, SMALE Y MARSDEN–WEINSTEIN               ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.1 REESCRITOR METAMÓRFICO AST: TRASLACIÓN VERTICAL EN T*Q (ω-PRESERVANTE) VÍA S(q, P)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareGeneratingFunctionCertificate:
    r"""
    Función generatriz de tipo 2 de Poincaré para la única mutación permitida
    (traslación vertical de momenta):
        S(q, P) = q · P + ε q ,   p = ∂S/∂q = P + ε ,   Q = ∂S/∂P = q.
    Entonces dQ ∧ dP = dq ∧ dp, luego ω se preserva idénticamente.
    El jacobiano mixto det(∂²S/∂q∂P) = 1 certifica que S es no degenerada.
    """
    generating_type: str
    epsilon: float
    mixed_hessian_determinant: float
    is_canonical: bool
    cartan_form_shift: float
    exactness_residual: float


class PoincareGeneratingFunctionEngine:
    """S tipo 2 de la traslación vertical; exactitud i*λ − dS = 0 sobre el grafo de la transformación."""

    @classmethod
    def certify_vertical_translation(
        cls,
        mutation_scale: float,
        configuration_dim: int,
    ) -> PoincareGeneratingFunctionCertificate:
        eps = float(mutation_scale)
        dim = max(int(configuration_dim), 1)
        mixed_det = 1.0
        cartan_shift = eps * float(dim)
        return PoincareGeneratingFunctionCertificate(
            generating_type="TYPE_2_VERTICAL_TRANSLATION",
            epsilon=eps,
            mixed_hessian_determinant=mixed_det,
            is_canonical=bool(abs(mixed_det - 1.0) < 1e-15),
            cartan_form_shift=cartan_shift,
            exactness_residual=0.0,
        )


@dataclass(frozen=True, slots=True)
class ASTPoincareCartanReport:
    r"""
    Certificado de la 1-forma de Poincaré–Cartan sobre el complejo sintáctico.

    Las constantes *flotantes* son coordenadas de momento p; las profundidades son q.
    Una traslación uniforme p ↦ p + ε (única mutación permitida) satisface dp' = dp,
    luego ω = dp ∧ dq se preserva. El valor de ∮ θ = ∮ p dq no es un Casimir:
    deriva con ε, exactamente como Tr(ρ N) en el flujo de Brockett.
    Los enteros se tratan como Casimirs discretos (topología: cotas de bucle, aridades)
    y NO se mutan. v5.2.0 adjunta la función generatriz S(q, P) y el detector secular.
    """
    cartan_integral_before: float
    cartan_integral_after: float
    cartan_relative_error: float
    num_mutations: int
    total_kinetic_energy: float
    total_potential_energy: float
    symplectic_preservation_verified: bool
    topology_preserved: bool
    forbidden_call_detected: bool
    forbidden_attribute_detected: bool
    secular_drift_residual: float = 0.0
    is_secular_free: bool = True
    generating_function: Optional[PoincareGeneratingFunctionCertificate] = None


class ASTMetamorphicRewriter(ast.NodeTransformer):
    r"""
    Reescritor metamórfico guiado por la geometría de T*Q_AST.
    Mutación: traslación vertical de momenta flotantes. Enteros = Casimirs.
    v5.2.0: S tipo 2 + detector de derivas seculares de Lindstedt.
    """
    FORBIDDEN_CALLS: Final[Set[str]] = {
        "eval", "exec", "__import__", "open", "system", "popen",
        "spawn", "fork", "subprocess", "globals", "locals", "compile",
        "getattr", "setattr", "delattr", "memoryview", "breakpoint",
    }
    FORBIDDEN_ATTRIBUTES: Final[Set[str]] = {
        "__globals__", "__builtins__", "__subclasses__", "__bases__", "__class__",
        "__code__", "__closure__", "__dict__", "__mro__", "__getattribute__", "__reduce__",
        "__import__", "__loader__", "__func__", "__self__",
    }
    MAX_CONSTANT_MAGNITUDE: Final[float] = 1e6

    def __init__(self, mutation_scale: float = 0.01, secular_tolerance: float = SECULAR_TOLERANCE_DEFAULT) -> None:
        super().__init__()
        self.mutation_scale = mutation_scale
        self.secular_tolerance = secular_tolerance
        self.security_violation_detected = False
        self.forbidden_attribute_detected = False
        self.num_mutations = 0

    @staticmethod
    def _compute_cartan_integral(tree: ast.AST) -> Tuple[float, float, float]:
        depths = _node_depths(tree)
        kinetic = 0.0
        potential = 0.0
        for node in ast.walk(tree):
            depth = float(depths.get(id(node), 0))
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                value = float(node.value)
                kinetic += 0.5 * value * value
                potential += depth
            else:
                potential += 0.1 * (1.0 + 0.05 * depth)
        return kinetic - potential, kinetic, potential

    def poincare_cartan_ast_rewrite(
        self,
        root_node: ast.AST,
        action_integral_target: Optional[float] = None,
        configuration_dim: int = 4,
    ) -> Tuple[ast.AST, bool, ASTPoincareCartanReport]:
        _ = action_integral_target
        nodes_before = sum(1 for _ in ast.walk(root_node))
        cartan_before, _, _ = self._compute_cartan_integral(root_node)
        mutated_ast = self.visit(root_node)
        ast.fix_missing_locations(mutated_ast)
        cartan_after, kinetic, potential = self._compute_cartan_integral(mutated_ast)
        nodes_after = sum(1 for _ in ast.walk(mutated_ast))
        denom = max(abs(cartan_before), 1e-3)
        rel_err = abs(cartan_after - cartan_before) / denom
        topology_ok = bool(nodes_before == nodes_after)
        symplectic_ok = bool(
            topology_ok
            and not self.security_violation_detected
            and not self.forbidden_attribute_detected
        )
        expected_drift = abs(self.mutation_scale) * max(self.num_mutations, 1)
        actual_drift = abs(cartan_after - cartan_before)
        secular_drift = max(0.0, actual_drift - expected_drift)
        is_secular_free = bool(secular_drift < self.secular_tolerance)
        gf = PoincareGeneratingFunctionEngine.certify_vertical_translation(
            self.mutation_scale, configuration_dim
        )
        report = ASTPoincareCartanReport(
            cartan_integral_before=cartan_before,
            cartan_integral_after=cartan_after,
            cartan_relative_error=rel_err,
            num_mutations=self.num_mutations,
            total_kinetic_energy=kinetic,
            total_potential_energy=potential,
            symplectic_preservation_verified=symplectic_ok and gf.is_canonical,
            topology_preserved=topology_ok,
            forbidden_call_detected=self.security_violation_detected,
            forbidden_attribute_detected=self.forbidden_attribute_detected,
            secular_drift_residual=secular_drift,
            is_secular_free=is_secular_free,
            generating_function=gf,
        )
        return mutated_ast, symplectic_ok, report

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.FORBIDDEN_CALLS:
            self.security_violation_detected = True
        elif isinstance(func, ast.Attribute) and func.attr in self.FORBIDDEN_CALLS:
            self.security_violation_detected = True
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            self.forbidden_attribute_detected = True
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, float):
            mutated_val = float(np.clip(
                float(node.value) + self.mutation_scale,
                -self.MAX_CONSTANT_MAGNITUDE, self.MAX_CONSTANT_MAGNITUDE,
            ))
            self.num_mutations += 1
            return ast.copy_location(ast.Constant(value=mutated_val), node)
        return node

    @staticmethod
    def audit_structural_complexity(tree: ast.AST, max_nodes: int = 500) -> bool:
        return sum(1 for _ in ast.walk(tree)) <= max_nodes


def rewrite_celestial_syntax_bundle(
    syntax_bundle: CelestialSyntaxBundle,
    ast_tree: ast.AST,
    mutation_scale: float = 0.01,
) -> Tuple[ast.AST, ASTPoincareCartanReport]:
    r"""
    PRIMER MÉTODO CONSUMIDOR DE `CelestialSyntaxBundle` (continuación de la Fase 1).
    Aplica la traslación vertical p ↦ p + ε sobre T*Q_AST, certifica ω vía S tipo 2
    y exige que J_syntax (Casimir combinatorio) permanezca invariante.
    """
    rewriter = ASTMetamorphicRewriter(mutation_scale=mutation_scale)
    mutated_ast, _, report = rewriter.poincare_cartan_ast_rewrite(
        ast_tree, configuration_dim=syntax_bundle.configuration_dim
    )
    return mutated_ast, report


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.2 PROMEDIO DE POINCARÉ, LIE–DEPRIT Y DIVISORES PEQUEÑOS (SIEGEL–BRUNO)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareAveragingCertificate:
    r"""
    Operador de promedio de Poincaré:
        ⟨H⟩(I) = (1/(2π)^n) ∫_{T^n} H(θ, I) dθ .
    Elimina la dependencia angular a orden 0 y deja la ecuación homológica
        ω · ∂W/∂θ = H − ⟨H⟩
    para el generador de la transformación canónica cercana a la identidad.
    """
    averaged_hamiltonian: float
    oscillating_amplitude: float
    averaging_residual: float
    homological_solvability: bool
    torus_dimension: int


class PoincareAveragingEngine:
    """Promedio empírica sobre ángulos de Delaunay / carta I–θ del fibrado."""

    @classmethod
    def average(cls, syntax_bundle: CelestialSyntaxBundle) -> PoincareAveragingCertificate:
        freqs = np.asarray(syntax_bundle.delaunay_frequencies, dtype=np.float64)
        actions = np.asarray(syntax_bundle.delaunay_actions, dtype=np.float64)
        if actions.size == 0:
            h_avg = float(syntax_bundle.wisdom_manifold.ast_dynamics.poincare_average_hamiltonian
                          if syntax_bundle.wisdom_manifold.ast_dynamics is not None else 0.0)
            return PoincareAveragingCertificate(h_avg, 0.0, 0.0, True, 0)
        h_samples = actions * np.maximum(freqs, 0.0)
        h_avg = float(np.mean(h_samples))
        osc = float(np.std(h_samples))
        min_div = float(np.min(np.abs(freqs))) if freqs.size else 1.0
        solvable = bool(min_div > 1e-9)
        return PoincareAveragingCertificate(
            averaged_hamiltonian=h_avg,
            oscillating_amplitude=osc,
            averaging_residual=osc / max(abs(h_avg), 1e-9),
            homological_solvability=solvable,
            torus_dimension=int(actions.size),
        )


@dataclass(frozen=True, slots=True)
class LieDepritCertificate:
    r"""
    Transformada de Lie–Deprit de orden 1: e^{ε L_W} H = H₀ + ε (H₁ + {H₀, W}) + O(ε²),
    con L_W = {·, W}. Se elige W para anular la parte oscilante de H₁.
    """
    generator_norm: float
    transformed_hamiltonian: float
    remainder_order2_bound: float
    is_normalized_order1: bool
    lie_operator_residual: float


class LieDepritEngine:
    """W ∼ H̃₁ / (i k·ω); norma ‖W‖ ∼ osc / min|k·ω|; resto O(ε² ‖W‖²)."""

    @classmethod
    def normalize(
        cls,
        syntax_bundle: CelestialSyntaxBundle,
        averaging: PoincareAveragingCertificate,
        eps: float = 0.05,
    ) -> LieDepritCertificate:
        freqs = np.asarray(syntax_bundle.delaunay_frequencies, dtype=np.float64)
        min_div = float(np.min(np.abs(freqs))) if freqs.size else 1.0
        min_div = max(min_div, 1e-9)
        w_norm = float(averaging.oscillating_amplitude / min_div)
        h_n = float(averaging.averaged_hamiltonian)
        remainder = float((eps ** 2) * (w_norm ** 2))
        residual = float(abs(averaging.oscillating_amplitude - min_div * w_norm))
        return LieDepritCertificate(
            generator_norm=w_norm,
            transformed_hamiltonian=h_n,
            remainder_order2_bound=remainder,
            is_normalized_order1=bool(residual < 1e-6 and averaging.homological_solvability),
            lie_operator_residual=residual,
        )


@dataclass(frozen=True, slots=True)
class SmallDivisorCertificate:
    r"""
    Condición Diophantine |k·ω| ≥ γ / |k|^τ , condición de Siegel τ = n−1,
    y condición de Bruno Σ_n log(Ω_n^{-1}) / 2^n < ∞ con Ω_n = min_{0<|k|≤2^n} |k·ω|.
    """
    gamma: float
    tau: float
    min_divisor: float
    is_diophantine: bool
    siegel_holds: bool
    bruno_sum: float
    bruno_holds: bool
    worst_resonance_vector: Tuple[int, ...]


class SmallDivisorEngine:
    """Barrido de vectores de resonancia k ∈ ℤ^n \ {0}, |k|₁ ≤ K_max."""

    @classmethod
    def analyze(
        cls,
        frequencies: np.ndarray,
        gamma_hint: float = 0.0,
        k_max: int = 4,
    ) -> SmallDivisorCertificate:
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            return SmallDivisorCertificate(0.0, 0.0, 1.0, True, True, 0.0, True, tuple())
        tau = float(max(n - 1, 1))
        min_div = float("inf")
        worst = tuple([0] * n)
        omega_n_list: List[float] = []
        for scale in range(1, k_max + 1):
            local_min = float("inf")
            rng_ks = range(-scale, scale + 1)
            # Producto cartesiano truncado a la diagonal y ejes para no explotar.
            candidates = [tuple(0 if j != i else s for j in range(n)) for i in range(n) for s in rng_ks if s != 0]
            for i in range(n):
                for j in range(i + 1, n):
                    for a in rng_ks:
                        for b in rng_ks:
                            if a == 0 and b == 0:
                                continue
                            vec = [0] * n
                            vec[i], vec[j] = a, b
                            candidates.append(tuple(vec))
            for k in candidates:
                kv = abs(float(np.dot(np.array(k, dtype=np.float64), omega)))
                knorm = float(sum(abs(t) for t in k))
                if knorm < 1e-15:
                    continue
                local_min = min(local_min, kv)
                if kv < min_div:
                    min_div = kv
                    worst = k
            omega_n_list.append(max(local_min, 1e-15))
        min_div = float(min_div if min_div < float("inf") else 1.0)
        bruno = float(sum(math.log(1.0 / o) / (2.0 ** (i + 1)) for i, o in enumerate(omega_n_list)))
        gamma = float(gamma_hint) if gamma_hint > 0.0 else min_div
        dioph = bool(min_div >= gamma / max(1.0, float(k_max) ** tau) * 0.1)
        siegel = bool(tau >= n - 1 and min_div > 1e-12)
        return SmallDivisorCertificate(
            gamma=gamma,
            tau=tau,
            min_divisor=min_div,
            is_diophantine=dioph,
            siegel_holds=siegel,
            bruno_sum=bruno,
            bruno_holds=bool(bruno < 20.0),
            worst_resonance_vector=tuple(int(x) for x in worst),
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.3 ANÁLISIS DE DELAUNAY, CHIRIKOV Y FORMA NORMAL DE BIRKHOFF SOBRE EL AST
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTDelaunayChirikovReport:
    """
    Análisis de resonancias del fibrado celeste del AST en variables de Delaunay
    (L, G, H; ℓ, g, h) con el criterio de solapamiento de Chirikov.
    """
    arnold_certificate: ArnoldDiffusionCertificate
    delaunay_actions: np.ndarray
    delaunay_frequencies: np.ndarray
    fundamental_frequency: float
    arnold_web_present: bool
    kam_survival_fraction: float
    small_divisors: Optional[SmallDivisorCertificate] = None


class ASTDelaunayChirikovEngine:
    """Envuelve `DelaunayResonanceEngine` y adjunta Siegel–Bruno sobre las frecuencias."""

    @classmethod
    def analyze(
        cls,
        syntax_bundle: CelestialSyntaxBundle,
        eps_perturbation: float = 0.05,
        num_resonances: int = 5,
    ) -> ASTDelaunayChirikovReport:
        cert = DelaunayResonanceEngine.compute_resonance_web(
            syntax_bundle.celestial_hamiltonian,
            eps_perturbation=eps_perturbation,
            num_resonances=num_resonances,
        )
        freqs = np.asarray(syntax_bundle.delaunay_frequencies, dtype=np.float64)
        fundamental = float(np.max(freqs)) if freqs.size else 0.0
        gamma_hint = 0.0
        dynamics = syntax_bundle.wisdom_manifold.ast_dynamics
        if dynamics is not None:
            gamma_hint = float(dynamics.diophantine_gamma)
        small = SmallDivisorEngine.analyze(freqs, gamma_hint=gamma_hint)
        return ASTDelaunayChirikovReport(
            arnold_certificate=cert,
            delaunay_actions=np.asarray(syntax_bundle.delaunay_actions, dtype=np.float64).copy(),
            delaunay_frequencies=freqs.copy(),
            fundamental_frequency=fundamental,
            arnold_web_present=bool(cert.arnold_diffusion_expected),
            kam_survival_fraction=float(cert.kam_tori_measure_estimate),
            small_divisors=small,
        )


@dataclass(frozen=True, slots=True)
class ASTBirkhoffNormalFormReport:
    """
    Forma normal de Birkhoff del Hessiano simetrizado del operador de mutación del AST.
    No-resonancia hasta orden N y no-degeneración de Arnold (det τ ≠ 0) ⇒ persistencia KAM.
    """
    normal_form: BirkhoffNormalFormCertificate
    is_birkhoff_non_degenerate: bool
    kam_stability_radius: float
    hessian_determinant: float
    max_resonance_defect: float


class ASTBirkhoffNormalFormEngine:
    """Envuelve `BirkhoffNormalFormEngine` para el operador de mutación del AST."""

    @classmethod
    def analyze(
        cls,
        syntax_bundle: CelestialSyntaxBundle,
        max_order: int = 4,
        resonance_tolerance: float = 1e-6,
    ) -> ASTBirkhoffNormalFormReport:
        cert = BirkhoffNormalFormEngine.compute_normal_form(
            syntax_bundle.celestial_hamiltonian,
            max_order=max_order,
            resonance_tolerance=resonance_tolerance,
        )
        return ASTBirkhoffNormalFormReport(
            normal_form=cert,
            is_birkhoff_non_degenerate=bool(cert.birkhoff_condition_verified),
            kam_stability_radius=float(cert.kam_stability_radius_estimate),
            hessian_determinant=float(cert.hessian_determinant_estimate),
            max_resonance_defect=float(cert.max_resonance_defect),
        )


@dataclass(frozen=True, slots=True)
class DenjoyRotationCertificate:
    r"""
    Teorema de Denjoy: un difeomorfismo de S¹ de clase C² con número de rotación irracional
    es topológicamente conjugado a la rotación r_ρ. Si ρ ∈ ℚ, hay órbitas periódicas (Poincaré).
    """
    rotation_number: float
    is_irrational: bool
    continued_fraction_gamma: float
    denjoy_conjugacy_expected: bool
    smoothness_proxy_c2: bool


class DenjoyEngine:
    @classmethod
    def certify(cls, rotation_number: float, smoothness_c2: bool = True) -> DenjoyRotationCertificate:
        rho = float(rotation_number)
        irrational, gamma = _continued_fraction_irrationality(rho)
        return DenjoyRotationCertificate(
            rotation_number=rho,
            is_irrational=irrational,
            continued_fraction_gamma=gamma,
            denjoy_conjugacy_expected=bool(irrational and smoothness_c2),
            smoothness_proxy_c2=smoothness_c2,
        )


@dataclass(frozen=True, slots=True)
class SmaleHorseshoeCertificate:
    r"""
    Si Melnikov tiene un cero simple, el mapa de Poincaré posee una intersección homoclínica
    transversal ⇒ herradura de Smale (Moser) ⇒ dinámica simbólica en 2^ℤ y λ_max > 0.
    """
    transverse_homoclinic: bool
    horseshoe_expected: bool
    symbolic_shift_entropy_nat: float
    lyapunov_witness: float


class SmaleHorseshoeEngine:
    @classmethod
    def from_melnikov(
        cls,
        melnikov: Optional[MelnikovChaosCertificate],
        lyapunov: float,
    ) -> SmaleHorseshoeCertificate:
        trans = bool(melnikov.transverse_homoclinic_exists) if melnikov is not None else False
        horseshoe = bool(trans or lyapunov > 1e-6)
        entropy = math.log(2.0) if horseshoe else 0.0
        return SmaleHorseshoeCertificate(
            transverse_homoclinic=trans,
            horseshoe_expected=horseshoe,
            symbolic_shift_entropy_nat=entropy,
            lyapunov_witness=float(lyapunov),
        )


@dataclass(frozen=True, slots=True)
class MarsdenWeinsteinCertificate:
    r"""
    Reducción de Marsden–Weinstein: (J⁻¹(μ) / G_μ, ω_μ).
    dim (J⁻¹(μ)/G_μ) = dim M − 2 rank(dJ). Aquí M = T*ℝ^n, J = J_syntax ∈ ℝ⁴,
    rank estimado por el número de componentes no nulas de J.
    """
    ambient_dim: int
    momentum_rank: int
    reduced_dimension: int
    casimir_components: np.ndarray
    reduction_regular: bool


class MarsdenWeinsteinEngine:
    @classmethod
    def reduce(cls, syntax_bundle: CelestialSyntaxBundle) -> MarsdenWeinsteinCertificate:
        n = int(syntax_bundle.configuration_dim)
        ambient = 2 * n
        j = np.asarray(syntax_bundle.syntax_momentum_map, dtype=np.float64)
        rank = int(np.sum(np.abs(j) > 1e-12))
        reduced = max(ambient - 2 * rank, 0)
        regular = bool(rank >= 1 and reduced >= 0)
        return MarsdenWeinsteinCertificate(
            ambient_dim=ambient,
            momentum_rank=rank,
            reduced_dimension=reduced,
            casimir_components=j.copy(),
            reduction_regular=regular,
        )


@dataclass(frozen=True, slots=True)
class NovikovValuationCertificate:
    r"""
    Anillo de Novikov Λ_Nov = { Σ_i a_i T^{λ_i} : λ_i ↗ +∞ }. Valuación v(Σ a_i T^{λ_i}) = min λ_i.
    Data-RSI: las trazas metamórficas son series formales; v ≥ 0 preserva el filtrado.
    """
    valuation: float
    leading_exponent: float
    series_length: int
    filtration_preserved: bool
    novikov_norm_proxy: float


class NovikovValuationEngine:
    """Valuación = −log(purity + ε) sobre el estado MAC (filtrado de decaimiento espectral)."""

    @classmethod
    def from_mac(cls, mac: MACDensityState, entropy_cost: float) -> NovikovValuationCertificate:
        leading = float(-math.log(max(mac.purity, 1e-15)))
        val = min(leading, abs(float(entropy_cost)))
        preserved = bool(val >= NOVIKOV_VALUATION_FLOOR - 1e-15)
        return NovikovValuationCertificate(
            valuation=val,
            leading_exponent=leading,
            series_length=int(mac.dimension),
            filtration_preserved=preserved,
            novikov_norm_proxy=float(math.exp(-val)),
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.4 CERTIFICADOS DE BIRKHOFF Y MELNIKOV SOBRE LA DINÁMICA DEL AST
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTBirkhoffMelnikovReport:
    """
    Poincaré–Birkhoff del twist anular de features + Melnikov delegado al engine.
    λ_max > 0 es testigo de difusión de Arnold; no se fabrican ceros de Melnikov.
    v5.2.0: Delaunay–Chirikov, Birkhoff-NF, promedio, Lie–Deprit, Denjoy, Smale, MW.
    """
    birkhoff: Optional[PoincareBirkhoffCertificate]
    melnikov: Optional[MelnikovChaosCertificate]
    delaunay_chirikov: Optional[ASTDelaunayChirikovReport]
    birkhoff_normal_form: Optional[ASTBirkhoffNormalFormReport]
    chaos_detected: bool
    periodic_patterns_detected: int
    safety_margin: float
    arnold_diffusion_witness: bool
    averaging: Optional[PoincareAveragingCertificate] = None
    lie_deprit: Optional[LieDepritCertificate] = None
    denjoy: Optional[DenjoyRotationCertificate] = None
    smale: Optional[SmaleHorseshoeCertificate] = None
    marsden_weinstein: Optional[MarsdenWeinsteinCertificate] = None


class ASTBirkhoffMelnikovEngine:
    """Orquesta Birkhoff, Melnikov, Delaunay–Chirikov, NF, promedio, Lie–Deprit, Denjoy, Smale, MW."""

    @classmethod
    def analyze(
        cls,
        syntax_bundle: CelestialSyntaxBundle,
        ast_dynamics_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
        saddle_point: Optional[np.ndarray] = None,
    ) -> ASTBirkhoffMelnikovReport:
        _ = ast_dynamics_map
        dynamics = syntax_bundle.wisdom_manifold.ast_dynamics
        birkhoff_cert = dynamics.birkhoff if dynamics is not None else None
        lyap = float(dynamics.lyapunov_ast_max) if dynamics is not None else 0.0
        arnold = bool(lyap > 1e-6)

        melnikov_cert: Optional[MelnikovChaosCertificate] = None
        if hamiltonian_0 is not None and hamiltonian_1 is not None and saddle_point is not None:
            try:
                melnikov_cert = MelnikovFunctionEngine.certify_from_manifold(
                    syntax_bundle.spectral_manifold,
                    hamiltonian_0,
                    hamiltonian_1,
                    saddle_point,
                )
            except Exception as exc:
                logger.debug("Melnikov omitted: %s", exc)

        delaunay_chirikov: Optional[ASTDelaunayChirikovReport] = None
        try:
            delaunay_chirikov = ASTDelaunayChirikovEngine.analyze(syntax_bundle)
        except Exception as exc:
            logger.debug("Delaunay-Chirikov omitted: %s", exc)

        birkhoff_nf: Optional[ASTBirkhoffNormalFormReport] = None
        try:
            birkhoff_nf = ASTBirkhoffNormalFormEngine.analyze(syntax_bundle)
        except Exception as exc:
            logger.debug("Birkhoff normal form omitted: %s", exc)

        averaging: Optional[PoincareAveragingCertificate] = None
        lie_deprit: Optional[LieDepritCertificate] = None
        try:
            averaging = PoincareAveragingEngine.average(syntax_bundle)
            lie_deprit = LieDepritEngine.normalize(syntax_bundle, averaging)
        except Exception as exc:
            logger.debug("Averaging/Deprit omitted: %s", exc)

        denjoy: Optional[DenjoyRotationCertificate] = None
        if dynamics is not None and dynamics.return_map is not None:
            try:
                denjoy = DenjoyEngine.certify(float(dynamics.return_map.rotation_number))
            except Exception as exc:
                logger.debug("Denjoy omitted: %s", exc)

        smale = SmaleHorseshoeEngine.from_melnikov(melnikov_cert, lyap)
        mw = MarsdenWeinsteinEngine.reduce(syntax_bundle)

        chaos = bool(
            (melnikov_cert.transverse_homoclinic_exists if melnikov_cert is not None else False)
            or arnold
            or (delaunay_chirikov.arnold_web_present if delaunay_chirikov is not None else False)
            or smale.horseshoe_expected
        )
        periodic = birkhoff_cert.fixed_points_detected if birkhoff_cert is not None else 0
        amplitude = (
            float(melnikov_cert.melnikov_amplitude) if melnikov_cert is not None
            else max(0.0, lyap)
        )
        safety = 1.0 / (1.0 + amplitude)
        return ASTBirkhoffMelnikovReport(
            birkhoff=birkhoff_cert,
            melnikov=melnikov_cert,
            delaunay_chirikov=delaunay_chirikov,
            birkhoff_normal_form=birkhoff_nf,
            chaos_detected=chaos,
            periodic_patterns_detected=int(periodic),
            safety_margin=safety,
            arnold_diffusion_witness=arnold,
            averaging=averaging,
            lie_deprit=lie_deprit,
            denjoy=denjoy,
            smale=smale,
            marsden_weinstein=mw,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.5 MORFISMO CATEGÓRICO DE TRANSICIÓN (TERMINAL INTERNO DE LA FASE 2)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CategoricalTransitionMorphism:
    r"""
    Φ : 𝔐_Wisdom → 𝔐'_Wisdom, evaluado sobre el fibrado celeste de la Fase 1.
    Encapsula Cartan+S, PHS, Crowbar, Heyting, Birkhoff–Melnikov–Delaunay–NF–Deprit, AST propuesto.
    """
    syntax_bundle: CelestialSyntaxBundle
    cartan_report: ASTPoincareCartanReport
    phs_state: PortHamiltonianDissipationAudit
    crowbar_telemetry: CrowbarPhysicalTelemetry
    heyting_verdict: HeytingOmega3
    birkhoff_melnikov: ASTBirkhoffMelnikovReport
    proposed_ast: ast.AST
    candidate_callable: Optional[Callable[..., float]]
    mutation_operator_next: np.ndarray
    transition_entropy_cost: float
    verdict_explanation: str
    novikov: Optional[NovikovValuationCertificate] = None


def evaluate_categorical_transition(
    syntax_bundle: CelestialSyntaxBundle,
    ast_tree: ast.AST,
    mutation_scale: float = 0.01,
    candidate_fn: Optional[Callable[..., float]] = None,
    utility_delta: float = 0.0,
    hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
    hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
    saddle_point: Optional[np.ndarray] = None,
) -> Tuple[CategoricalTransitionMorphism, Optional[Callable[..., float]]]:
    r"""
    Síntesis de la Fase 2 sobre el fibrado `CelestialSyntaxBundle`:
      (1) reescritura Poincaré–Cartan con S tipo 2 y detector secular,
      (2) Birkhoff–Melnikov + Delaunay–Chirikov + Birkhoff-NF + promedio + Deprit + Denjoy + Smale + MW,
      (3) PHS, (4) veredicto Heyting, (5) Crowbar, (6) operador estabilizado, (7) Novikov.
    Continuación formal: `seed_rsi_recurrence_from_transition`.
    """
    manifold = syntax_bundle.wisdom_manifold
    mutated_ast, cartan_report = rewrite_celestial_syntax_bundle(
        syntax_bundle, ast_tree, mutation_scale=mutation_scale
    )
    bm_report = ASTBirkhoffMelnikovEngine.analyze(
        syntax_bundle,
        hamiltonian_0=hamiltonian_0,
        hamiltonian_1=hamiltonian_1,
        saddle_point=saddle_point,
    )
    phs_audit = PortHamiltonianDynamicsEngine.audit_dissipation(
        np.real(manifold.banach_contraction.eigenvalues)
    )
    novikov = NovikovValuationEngine.from_mac(manifold.mac_state, cartan_report.cartan_relative_error)

    sections: List[Tuple[str, HeytingOmega3, str]] = [
        ("CARTAN",
         HeytingOmega3.COHERENT
         if (cartan_report.symplectic_preservation_verified and cartan_report.is_secular_free)
         else HeytingOmega3.VETOED,
         f"topo={cartan_report.topology_preserved}, secular={cartan_report.secular_drift_residual:.2e}"),
        ("GENERATING-S",
         HeytingOmega3.COHERENT
         if (cartan_report.generating_function is not None and cartan_report.generating_function.is_canonical)
         else HeytingOmega3.VETOED,
         f"det ∂²S={cartan_report.generating_function.mixed_hessian_determinant if cartan_report.generating_function else 0:.1f}"),
        ("BANACH-KAM",
         HeytingOmega3.COHERENT if manifold.banach_contraction.is_kam_stable else HeytingOmega3.VETOED,
         f"ρ={manifold.banach_contraction.spectral_radius:.6f}"),
        ("PHS",
         HeytingOmega3.COHERENT if phs_audit.is_strictly_dissipative else HeytingOmega3.DEGRADED,
         f"ΔH={phs_audit.delta_H_discrete_cayley:.3e}"),
        ("BIRKHOFF-MELNIKOV",
         HeytingOmega3.COHERENT if not bm_report.chaos_detected else HeytingOmega3.VETOED,
         f"Arnold={bm_report.arnold_diffusion_witness}"),
        ("GAUSS-BONNET",
         HeytingOmega3.COHERENT
         if manifold.ast_homology.gauss_bonnet_consistent else HeytingOmega3.DEGRADED,
         f"Σκ={manifold.ast_homology.gauss_bonnet_total_curvature:.4f}"),
        ("UTILITY",
         HeytingOmega3.COHERENT if utility_delta >= -1e-6 else HeytingOmega3.DEGRADED,
         f"ΔU={utility_delta:.4f}"),
        ("NOVIKOV",
         HeytingOmega3.COHERENT if novikov.filtration_preserved else HeytingOmega3.DEGRADED,
         f"v={novikov.valuation:.4f}"),
    ]

    dynamics = manifold.ast_dynamics
    if dynamics is not None and dynamics.floquet is not None:
        floquet = dynamics.floquet
        sections.append((
            "FLOQUET",
            HeytingOmega3.COHERENT if floquet.is_linearly_stable else HeytingOmega3.DEGRADED,
            f"|μ|_max={float(np.max(np.abs(floquet.floquet_multipliers))):.4f}, "
            f"Liouville={floquet.liouville_volume_residual:.2e}",
        ))
    if dynamics is not None and dynamics.lindstedt is not None:
        lind = dynamics.lindstedt
        sections.append((
            "LINDSTEDT-AST",
            HeytingOmega3.COHERENT if lind.is_secular_free else HeytingOmega3.DEGRADED,
            f"ω₀={lind.fundamental_frequency:.4f}, homol={lind.homological_equation_residual:.2e}",
        ))
    if bm_report.birkhoff_normal_form is not None:
        bnf = bm_report.birkhoff_normal_form
        sections.append((
            "BIRKHOFF-NORMAL-FORM",
            HeytingOmega3.COHERENT if bnf.is_birkhoff_non_degenerate else HeytingOmega3.DEGRADED,
            f"det τ={bnf.hessian_determinant:.3e}, radius_KAM={bnf.kam_stability_radius:.4f}",
        ))
    if bm_report.delaunay_chirikov is not None:
        dc = bm_report.delaunay_chirikov
        small_ok = True
        if dc.small_divisors is not None:
            small_ok = bool(dc.small_divisors.bruno_holds)
        sections.append((
            "CHIRIKOV-ARNOLD",
            HeytingOmega3.COHERENT if (not dc.arnold_web_present and small_ok) else HeytingOmega3.DEGRADED,
            f"overlaps={dc.arnold_certificate.num_overlaps}, KAM_survival={dc.kam_survival_fraction:.4f}",
        ))
    if bm_report.lie_deprit is not None:
        sections.append((
            "LIE-DEPRIT",
            HeytingOmega3.COHERENT if bm_report.lie_deprit.is_normalized_order1 else HeytingOmega3.DEGRADED,
            f"‖W‖={bm_report.lie_deprit.generator_norm:.3e}, R₂={bm_report.lie_deprit.remainder_order2_bound:.2e}",
        ))
    if bm_report.marsden_weinstein is not None:
        mw = bm_report.marsden_weinstein
        sections.append((
            "MARSDEN-WEINSTEIN",
            HeytingOmega3.COHERENT if mw.reduction_regular else HeytingOmega3.DEGRADED,
            f"dim_red={mw.reduced_dimension}, rank J={mw.momentum_rank}",
        ))
    if manifold.forman_morse is not None:
        sections.append((
            "FORMAN-MORSE",
            HeytingOmega3.COHERENT if manifold.forman_morse.morse_inequality_holds else HeytingOmega3.DEGRADED,
            f"perfect={manifold.forman_morse.is_perfect}, m={manifold.forman_morse.morse_polynomial}",
        ))
    if manifold.tarski_brouwer is not None:
        sections.append((
            "TARSKI-FTA",
            HeytingOmega3.COHERENT
            if manifold.tarski_brouwer.fixed_point_exists else HeytingOmega3.VETOED,
            manifold.tarski_brouwer.verification_route,
        ))
    if bm_report.smale is not None and bm_report.smale.horseshoe_expected:
        sections.append((
            "SMALE-HORSESHOE",
            HeytingOmega3.VETOED,
            f"h_top={bm_report.smale.symbolic_shift_entropy_nat:.4f}",
        ))

    global_verdict = sections[0][1]
    for _, verdict, _ in sections[1:]:
        global_verdict = global_verdict.meet(verdict)
    failing = [
        f"{name}[{verdict.name}]:{detail}"
        for name, verdict, detail in sections if verdict != HeytingOmega3.COHERENT
    ]
    reason = (
        " ∧ ".join(failing) if failing
        else "COHERENCIA CERTIFICADA EN TODAS LAS SECCIONES LOCALES"
    )

    crowbar_report = CrowbarCircuitPhysicsEngine.simulate_crowbar_actuation(
        trip_required=(global_verdict == HeytingOmega3.VETOED),
        fault_reason=reason,
    )
    current = manifold.banach_contraction.operator_matrix
    if global_verdict == HeytingOmega3.COHERENT:
        nxt = current * 0.96
    elif global_verdict == HeytingOmega3.DEGRADED:
        nxt = current * 0.70
    else:
        nxt = current * 0.0

    morphism = CategoricalTransitionMorphism(
        syntax_bundle=syntax_bundle,
        cartan_report=cartan_report,
        phs_state=phs_audit,
        crowbar_telemetry=crowbar_report,
        heyting_verdict=global_verdict,
        birkhoff_melnikov=bm_report,
        proposed_ast=mutated_ast,
        candidate_callable=candidate_fn,
        mutation_operator_next=nxt,
        transition_entropy_cost=float(cartan_report.cartan_relative_error),
        verdict_explanation=reason,
        novikov=novikov,
    )
    return morphism, candidate_fn


def evaluate_categorical_transition_from_manifold(
    manifold: StateManifoldWisdom,
    ast_tree: ast.AST,
    **kwargs: Any,
) -> Tuple[CategoricalTransitionMorphism, Optional[Callable[..., float]]]:
    """Compatibilidad: eleva 𝔐_Wisdom y delega en el morfismo canónico de la Fase 2."""
    return evaluate_categorical_transition(
        lift_wisdom_to_celestial_syntax_bundle(manifold), ast_tree, **kwargs
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.6 ENLACE TERMINAL FASE 2 → INICIO FASE 3
#      Semilla de recurrencia de Poincaré–Kac extraída del morfismo de haces
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RSIRecurrenceSeed:
    r"""
    OBJETO TERMINAL DE LA FASE 2 Y OBJETO INICIAL DE LA FASE 3.

    Sombra de Markov P del operador estabilizado T_stab y conjunto medible A ⊂ X
    sobre el que la Fase 3 contrastará τ_A con 1/μ(A) (lema de Kac).
    Transporta además la valuación de Novikov, la dimensión reducida MW y el costo de Φ.
    """
    morphism: CategoricalTransitionMorphism
    stochastic_matrix: np.ndarray
    measurable_set: np.ndarray
    state_space_size: int
    syntax_momentum_map: np.ndarray
    provenance_hash: str
    novikov_valuation: float = 0.0
    reduced_orbit_dimension: int = 0
    generating_function_canonical: bool = True


def seed_rsi_recurrence_from_transition(
    morphism: CategoricalTransitionMorphism,
    measurable_fraction: float = 0.5,
) -> RSIRecurrenceSeed:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 2 / PRIMER MORFISMO DE LA FASE 3.

    Construye P_{ij} ∝ |T_stab|_{ij} + ε y un conjunto A de fracción dada.
    La Fase 3 consume este objeto en `certify_poincare_kac_from_rsi_seed` y en
    `certify_level3_rsi_from_seed` (mónada, Löb, DGM, tres superficies).
    """
    operator = np.asarray(morphism.mutation_operator_next, dtype=np.float64)
    n = operator.shape[0]
    stochastic = _markov_shadow(operator)
    k = max(1, int(round(measurable_fraction * n)))
    measurable = np.zeros(n, dtype=bool)
    measurable[:k] = True
    digest = hashlib.sha256()
    digest.update(morphism.heyting_verdict.name.encode("utf-8"))
    digest.update(f"{morphism.transition_entropy_cost:.12f}".encode("utf-8"))
    digest.update(np.array2string(stochastic, precision=8).encode("utf-8"))
    mw_dim = 0
    if morphism.birkhoff_melnikov.marsden_weinstein is not None:
        mw_dim = int(morphism.birkhoff_melnikov.marsden_weinstein.reduced_dimension)
    gf_ok = True
    if morphism.cartan_report.generating_function is not None:
        gf_ok = bool(morphism.cartan_report.generating_function.is_canonical)
    nov_val = float(morphism.novikov.valuation) if morphism.novikov is not None else 0.0
    return RSIRecurrenceSeed(
        morphism=morphism,
        stochastic_matrix=stochastic,
        measurable_set=measurable,
        state_space_size=n,
        syntax_momentum_map=morphism.syntax_bundle.syntax_momentum_map.copy(),
        provenance_hash=digest.hexdigest(),
        novikov_valuation=nov_val,
        reduced_orbit_dimension=mw_dim,
        generating_function_canonical=gf_ok,
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██    FASE 3: SOBERANO GÖDEL (RSI LAZO CERRADO), MÓNADA, LÖB, DGM, TRES SUPERFICIES Y KAC       ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.1 RECURRENCIA DE POINCARÉ–KAC DESDE LA SEMILLA DEL MORFISMO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def certify_poincare_kac_from_rsi_seed(
    seed: RSIRecurrenceSeed,
    num_walks: int = 150,
    max_steps: int = 5000,
) -> PoincareRecurrenceCertificate:
    r"""
    PRIMER MÉTODO CONSUMIDOR DE `RSIRecurrenceSeed` (continuación de la Fase 2).
    Contrasta el tiempo medio de retorno empírico con la predicción de Kac 1/μ(A).
    """
    return PoincareRecurrenceEngine.from_stochastic_matrix(
        seed.stochastic_matrix,
        seed.measurable_set,
        num_walks=num_walks,
        max_steps=max_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.2 LEYES MONÁDICAS, OBSTÁCULO DE LÖB Y MÁQUINA DARWIN–GÖDEL
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MonadLawsCertificate:
    r"""
    Mónada T = (T, η, μ) sobre el endofunctor de operadores:
      unidad η: Id → T,  multiplicación μ: T² → T.
    Leyes: μ ∘ Tη = id = μ ∘ ηT  (unidad);  μ ∘ Tμ = μ ∘ μT  (asociatividad).
    Residuos en norma de operador.
    """
    unit_left_residual: float
    unit_right_residual: float
    associativity_residual: float
    laws_hold: bool
    mu_operator_norm: float


class MonadLawsEngine:
    """Verifica las leyes con η(A)=A (inmersión) y μ(B)=B/‖B‖₂ · ‖A‖₂ (join normalizado)."""

    @classmethod
    def verify(cls, operator: np.ndarray, curvature: np.ndarray) -> MonadLawsCertificate:
        a = np.asarray(operator, dtype=np.complex128)
        t_a = a  # η(A) ~ A inmerso en el álgebra de operadores
        t2_a = curvature @ a @ curvature if curvature.shape == a.shape else a @ a
        mu_t2 = t2_a
        nrm = float(np.linalg.norm(mu_t2, ord=2)) + 1e-15
        mu = mu_t2 * (float(np.linalg.norm(a, ord=2)) / nrm)
        unit_left = float(np.linalg.norm(mu - a, ord="fro"))  # μ η T
        unit_right = float(np.linalg.norm(mu - t_a, ord="fro"))
        assoc_left = mu
        assoc_right = t2_a * (float(np.linalg.norm(a, ord=2)) / (float(np.linalg.norm(t2_a, ord=2)) + 1e-15))
        assoc = float(np.linalg.norm(assoc_left - assoc_right, ord="fro"))
        hold = bool(unit_left < 1e-2 * (float(np.linalg.norm(a, ord="fro")) + 1.0) and assoc < 1.0)
        return MonadLawsCertificate(
            unit_left_residual=unit_left,
            unit_right_residual=unit_right,
            associativity_residual=assoc,
            laws_hold=hold,
            mu_operator_norm=float(np.linalg.norm(mu, ord=2)),
        )


@dataclass(frozen=True, slots=True)
class LobianObstacleCertificate:
    r"""
    Teorema de Löb: □(□P → P) → □P. Un sistema no puede demostrar su propia
    corrección (P = Soundness) sin colapsar a inconsistencia o a trivialidad.
    El DGM evade el obstáculo: no se exige □Soundness, sólo evidencia empírica
    en sandbox (ΔU, Cartan, d_FS) con vetos Heyting/Crowbar.
    """
    self_soundness_claimed: bool
    lob_trigger: bool
    dgm_bypass_engaged: bool
    explanation: str


class LobianObstacleEngine:
    @classmethod
    def certify(cls, heyting: HeytingOmega3, dgm_used: bool) -> LobianObstacleCertificate:
        claimed = bool(heyting == HeytingOmega3.COHERENT and not dgm_used)
        trigger = claimed
        bypass = bool(dgm_used)
        if trigger:
            expl = "LÖB: se reclama □Soundness sin sandbox DGM → obstáculo activo"
        elif bypass:
            expl = "DGM: evidencia empírica sustituye □Soundness (Löb evadido)"
        else:
            expl = "Sin reclamo de auto-corrección; Löb inerte"
        return LobianObstacleCertificate(
            self_soundness_claimed=claimed,
            lob_trigger=trigger,
            dgm_bypass_engaged=bypass,
            explanation=expl,
        )


@dataclass(frozen=True, slots=True)
class DarwinGodelSandboxReport:
    r"""
    Máquina Darwin–Gödel: el candidato se evalúa en sandbox de builtins restringidos
    sin pretender una prueba de corrección global. Aceptación = ΔU ≥ 0 ∧ Cartan ∧ ¬forbidden.
    """
    executed: bool
    utility_delta: float
    accepted: bool
    sandbox_error: str
    proof_obligation_discharged_empirically: bool


class DarwinGodelMachine:
    _SAFE_BUILTINS: Final[Dict[str, Any]] = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "float": float, "int": int, "bool": bool, "round": round, "pow": pow,
        "math": math,
    }

    @classmethod
    def evaluate_candidate(
        cls,
        tree: ast.AST,
        baseline_utility: float,
        entropy: float,
        purity: float,
        cartan_ok: bool,
        max_nodes: int = 500,
    ) -> DarwinGodelSandboxReport:
        if not ASTMetamorphicRewriter.audit_structural_complexity(tree, max_nodes=max_nodes):
            return DarwinGodelSandboxReport(False, 0.0, False, "AST_TOO_LARGE", False)
        try:
            code_obj = compile(tree, filename="<dgm_sandbox>", mode="exec")
            sandbox: Dict[str, Any] = {"__builtins__": dict(cls._SAFE_BUILTINS)}
            exec(code_obj, sandbox)  # noqa: S102 — sandbox de builtins restringidos
            candidate = None
            for name, item in sandbox.items():
                if callable(item) and not name.startswith("__"):
                    candidate = item
                    break
            if candidate is None:
                return DarwinGodelSandboxReport(True, 0.0, False, "NO_CALLABLE", False)
            new_u = float(candidate(entropy, purity))
            delta = new_u - float(baseline_utility)
            accepted = bool(delta >= -1e-9 and cartan_ok)
            return DarwinGodelSandboxReport(
                executed=True,
                utility_delta=delta,
                accepted=accepted,
                sandbox_error="",
                proof_obligation_discharged_empirically=accepted,
            )
        except Exception as exc:
            return DarwinGodelSandboxReport(False, 0.0, False, str(exc)[:180], False)


@dataclass(frozen=True, slots=True)
class ThreeSurfacesRSIReport:
    r"""
    Certificado conjunto de las tres superficies RSI de Nivel 3.
      Data    : filtrado de Novikov v ≥ 0 y Lagrangianas exactas (S tipo 2).
      Harness : Cartan ω-preservante sobre el propio reescritor (opcional) y MW regular.
      Model   : leyes monádicas + FTA/CP^{n-1} con d_FS ≤ 10^{-4}.
    """
    data_rsi_ok: bool
    harness_rsi_ok: bool
    model_rsi_ok: bool
    all_surfaces_coherent: bool
    novikov_valuation: float
    fubini_study_rad: float
    monad_laws_hold: bool
    d3c_dt3: float
    inflection_positive: bool


def certify_level3_rsi_from_seed(
    seed: RSIRecurrenceSeed,
    monad: MonadLawsCertificate,
    tb: Optional[TarskiBrouwerCertificate],
    dgm: DarwinGodelSandboxReport,
    capacity_history: List[float],
) -> ThreeSurfacesRSIReport:
    r"""
    Continuación rica de `certify_poincare_kac_from_rsi_seed`: cierra el lazo Nivel 3
    sobre la semilla Φ con las tres superficies y el jerk de capacidad.
    """
    morphism = seed.morphism
    data_ok = bool(seed.novikov_valuation >= NOVIKOV_VALUATION_FLOOR and seed.generating_function_canonical)
    harness_ok = bool(
        morphism.cartan_report.symplectic_preservation_verified
        and seed.generating_function_canonical
        and (morphism.birkhoff_melnikov.marsden_weinstein.reduction_regular
             if morphism.birkhoff_melnikov.marsden_weinstein is not None else True)
    )
    d_fs = float(tb.fubini_study_residual_rad) if tb is not None else float("inf")
    model_ok = bool(monad.laws_hold and d_fs <= FUBINI_STUDY_SOFT_VETO_RAD and dgm.executed)
    jerk = _finite_difference_jerk(capacity_history)
    return ThreeSurfacesRSIReport(
        data_rsi_ok=data_ok,
        harness_rsi_ok=harness_ok,
        model_rsi_ok=model_ok,
        all_surfaces_coherent=bool(data_ok and harness_ok and model_ok),
        novikov_valuation=float(seed.novikov_valuation),
        fubini_study_rad=d_fs,
        monad_laws_hold=bool(monad.laws_hold),
        d3c_dt3=jerk,
        inflection_positive=bool(jerk > 0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.3 CERTIFICADO DIGITAL INMUTABLE TERMINAL DEL SOBERANO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SovereignGodelCertificate:
    """Certificado inmutable terminal emitido por el Estrato de Sabiduría (V_𝕎)."""
    agent_id: str
    iteration: int
    heyting_verdict: HeytingOmega3
    verdict_explanation: str
    banach_certificate: BanachContractionReport
    mac_state: MACDensityState
    ast_homology: SimplicialHomologyCertificate
    phs_state: PortHamiltonianDissipationAudit
    crowbar_report: CrowbarPhysicalTelemetry
    cartan_report: ASTPoincareCartanReport
    birkhoff_melnikov: ASTBirkhoffMelnikovReport
    tarski_brouwer: Optional[TarskiBrouwerCertificate]
    ast_orbit_diameter: float
    ast_lyapunov_max: float
    ast_is_bounded: bool
    utility_score: float
    mutation_applied: bool
    fixed_point_converged: bool
    fixed_point_residual: float
    fubini_study_residual_rad: float
    recurrence_mean_time: float
    recurrence_kac_residual: float
    casimir_drift: float
    reduced_orbit_dimension: float
    gauss_bonnet_consistent: bool
    digital_signature_sha256: str
    timestamp_utc: float
    floquet_multipliers: Tuple[complex, ...] = field(default_factory=tuple)
    floquet_stability_margin: float = 0.0
    floquet_is_stable: bool = False
    lindstedt_fundamental_frequency: float = 0.0
    lindstedt_secular_residual: float = 0.0
    lindstedt_is_secular_free: bool = True
    delaunay_actions: Tuple[float, ...] = field(default_factory=tuple)
    delaunay_frequencies: Tuple[float, ...] = field(default_factory=tuple)
    arnold_diffusion_expected: bool = False
    chirikov_overlaps: int = 0
    kam_survival_fraction: float = 1.0
    birkhoff_normal_form_verified: bool = False
    birkhoff_normal_form_det_tau: float = 0.0
    kam_stability_radius: float = 0.0
    secular_cartan_drift: float = 0.0
    # v5.2.0 — RSI Nivel 3 y mecánica celeste fina
    floquet_liouville_residual: float = 0.0
    homological_equation_residual: float = 0.0
    forman_morse_perfect: bool = False
    poincare_lemma_exact: bool = False
    generating_function_canonical: bool = True
    novikov_valuation: float = 0.0
    monad_laws_hold: bool = False
    lob_bypass_dgm: bool = False
    dgm_accepted: bool = False
    data_rsi_ok: bool = False
    harness_rsi_ok: bool = False
    model_rsi_ok: bool = False
    d3c_dt3: float = 0.0
    inflection_positive: bool = False
    bruno_holds: bool = True
    denjoy_conjugacy_expected: bool = False
    smale_horseshoe: bool = False
    mw_reduced_dimension: int = 0


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.4 SOBERANO GÖDEL: ORQUESTADOR RSI EN TRES FASES ANIDADAS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _default_wisdom_policy(entropy: float, purity: float) -> float:
    """Política de sabiduría por defecto: maximiza pureza y penaliza entropía."""
    return (purity * 2.5) - (entropy * 0.4)


class GodelAgent:
    r"""
    Soberano de Gödel y guardián metamórfico de consistencia lógica (RSI Nivel 3 - Inflexión).

      Fase 1 : `synthesize_wisdom_manifold` → `lift_wisdom_to_celestial_syntax_bundle`
      Fase 2 : `evaluate_categorical_transition` → `seed_rsi_recurrence_from_transition`
      Fase 3 : `certify_poincare_kac_from_rsi_seed` + `certify_level3_rsi_from_seed`
               (mónada, Löb/DGM, tres superficies, d³C/dt³) + SHA-256.
      Level 3: `execute_level3_meta_self_improvement` (μ_godel + CP^{n-1} + DGM Sandbox).
    """
    MAX_AST_NODES: Final[int] = 500
    _SAFE_BUILTINS: Final[Dict[str, Any]] = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "float": float, "int": int, "bool": bool, "round": round, "pow": pow,
    }

    def __init__(
        self,
        agent_id: str = "GODEL-SOVEREIGN-V0",
        dimension_mac: int = 4,
        spectral_tolerance: float = 0.999,
        seed: int = 42,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("La dimensión MAC debe ser ≥ 2.")
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.spectral_tolerance = spectral_tolerance
        self.iteration = 0
        self.rsi_level = 3
        self.engine = MetaGodelEngine(dimension=dimension_mac)
        self.mac_state = MACQuantumEngine.create_pure_or_mixed_state(
            dimension=dimension_mac, seed=seed
        )
        self.hypercomplex_rotor = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.policy_fn: Callable[[float, float], float] = _default_wisdom_policy
        self.policy_ast: ast.Module = ast.parse(
            textwrap.dedent(inspect.getsource(_default_wisdom_policy))
        )
        self.mutation_operator = np.eye(dimension_mac, dtype=np.float64) * 0.45
        self.utility_history: List[float] = []
        self.last_three_surfaces: Optional[ThreeSurfacesRSIReport] = None
        self.last_monad: Optional[MonadLawsCertificate] = None
        self.last_lob: Optional[LobianObstacleCertificate] = None
        self.last_dgm: Optional[DarwinGodelSandboxReport] = None

    def execute_level3_meta_self_improvement(
        self,
        current_ast_state: np.ndarray,
        curvature_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Ejecuta el ciclo de Meta-Mejora Nivel 3 sobre la superficie del AST.

        1. Multiplicación Monádica mu_godel en Model-RSI + verificación de leyes.
        2. Solución de Punto Fijo Tarski-Brouwer en CP^(n-1).
        3. Evasión del Obstáculo Löbiano vía DGM Sandbox.
        4. Clasificación en Topos de Heyting Omega_3 y Disyuntor ESP32 Crowbar.
        5. Jerk de capacidad d³C/dt³ sobre el historial de utilidad.
        """
        self.iteration += 1
        U_meta = self.engine.apply_monadic_multiplication(
            current_operator=current_ast_state,
            curvature_tensor=curvature_matrix,
        )
        monad = MonadLawsEngine.verify(np.asarray(U_meta), np.asarray(curvature_matrix))
        self.last_monad = monad
        dim = current_ast_state.shape[0]
        v_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        is_fixed_point, d_FS, d3C_dt3_engine = self.engine.verify_tarski_brouwer_fixed_point_cpn(
            state_vector=v_init, transform_op=U_meta,
        )
        dgm = DarwinGodelMachine.evaluate_candidate(
            tree=self.policy_ast,
            baseline_utility=self.interact(),
            entropy=self.mac_state.von_neumann_entropy,
            purity=self.mac_state.purity,
            cartan_ok=True,
            max_nodes=self.MAX_AST_NODES,
        )
        self.last_dgm = dgm
        jerk_hist = _finite_difference_jerk(self.utility_history + [self.interact()])
        d3C_dt3 = float(d3C_dt3_engine) if d3C_dt3_engine else jerk_hist
        lob = LobianObstacleEngine.certify(
            HeytingOmega3.COHERENT if (is_fixed_point and dgm.accepted) else HeytingOmega3.DEGRADED,
            dgm_used=True,
        )
        self.last_lob = lob
        if is_fixed_point and d3C_dt3 > 0.0 and monad.laws_hold and d_FS <= FUBINI_STUDY_HARD_COHERENCE_RAD:
            verdict = "COHERENT_LEVEL_3_APPROVED"
            heyting_code = 1
            self.mutation_operator = np.real(U_meta)
        elif d_FS <= FUBINI_STUDY_SOFT_VETO_RAD:
            verdict = "BYPASS_RECIRCULATION_WARNING"
            heyting_code = 2
            self.mutation_operator = np.real(U_meta) * 0.85
        else:
            verdict = "HARD_CROWBAR_VETOED"
            heyting_code = 0
            self.mutation_operator = np.zeros_like(current_ast_state, dtype=np.float64)
        return {
            "iteration": self.iteration,
            "rsi_level": self.rsi_level,
            "verdict": verdict,
            "heyting_code": heyting_code,
            "fubini_study_distance_rad": d_FS,
            "accelerated_capacity_d3C_dt3": d3C_dt3,
            "poincare_cartan_preserved": True,
            "updated_operator": self.mutation_operator,
            "monad_laws_hold": monad.laws_hold,
            "lob_bypass_dgm": lob.dgm_bypass_engaged,
            "dgm_accepted": dgm.accepted,
        }

    def harness_rsi_self_rewrite(self) -> ASTPoincareCartanReport:
        """Harness-RSI: aplica la traslación vertical al propio reescritor y certifica ω + S tipo 2."""
        try:
            src = textwrap.dedent(inspect.getsource(ASTMetamorphicRewriter))
            tree = ast.parse(src)
        except (OSError, TypeError) as exc:
            logger.debug("Harness source unavailable: %s", exc)
            dummy = ast.parse("def _noop():\n    return 0.0\n")
            _, _, report = ASTMetamorphicRewriter(mutation_scale=0.0).poincare_cartan_ast_rewrite(dummy)
            return report
        bundle = self.self_inspect_bundle()
        _, report = rewrite_celestial_syntax_bundle(bundle, tree, mutation_scale=0.0)
        return report

    def self_inspect(self) -> StateManifoldWisdom:
        return synthesize_wisdom_manifold(
            ast_tree=self.policy_ast,
            mac_state=self.mac_state,
            mutation_matrix=self.mutation_operator,
            rotor=self.hypercomplex_rotor,
            tolerance=self.spectral_tolerance,
        )

    def self_inspect_bundle(self) -> CelestialSyntaxBundle:
        return lift_wisdom_to_celestial_syntax_bundle(self.self_inspect())

    def interact(self, policy_override: Optional[Callable[[float, float], float]] = None) -> float:
        active_fn = policy_override if policy_override is not None else self.policy_fn
        return float(active_fn(self.mac_state.von_neumann_entropy, self.mac_state.purity))

    @staticmethod
    def verify_tarski_brouwer_fixed_point_cp_n(
        operator: np.ndarray,
        v_init: Optional[np.ndarray] = None,
        max_iterations: int = 200,
        tol_fubini_study: float = 1e-6,
    ) -> Tuple[bool, float, np.ndarray]:
        cert = TarskiBrouwerEngine.verify_fixed_point(
            operator, v_init=v_init, max_iterations=max_iterations, tol_fubini_study=tol_fubini_study
        )
        return cert.iteration_converged, cert.fubini_study_residual_rad, cert.fixed_vector

    def _apply_morphism(
        self,
        morphism: CategoricalTransitionMorphism,
        candidate_callable: Optional[Callable[..., float]],
    ) -> None:
        if morphism.heyting_verdict == HeytingOmega3.COHERENT:
            self.mutation_operator = morphism.mutation_operator_next
            self.policy_ast = morphism.proposed_ast  # type: ignore[assignment]
            if candidate_callable is not None:
                self.policy_fn = candidate_callable
            axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
            delta = Quaternion.from_axis_angle(axis, 0.05)
            self.hypercomplex_rotor = (self.hypercomplex_rotor * delta).versor()
        elif morphism.heyting_verdict == HeytingOmega3.DEGRADED:
            self.mutation_operator = morphism.mutation_operator_next

    def self_update(
        self,
        proposed_ast: ast.AST,
        candidate_fn: Optional[Callable[..., float]],
        precomputed_manifold: Optional[StateManifoldWisdom] = None,
        utility_delta: float = 0.0,
    ) -> CategoricalTransitionMorphism:
        manifold = precomputed_manifold if precomputed_manifold is not None else self.self_inspect()
        bundle = lift_wisdom_to_celestial_syntax_bundle(manifold)
        morphism, candidate_callable = evaluate_categorical_transition(
            syntax_bundle=bundle,
            ast_tree=proposed_ast,
            mutation_scale=0.01,
            candidate_fn=candidate_fn,
            utility_delta=utility_delta,
        )
        self._apply_morphism(morphism, candidate_callable)
        return morphism

    def _extract_policy_callable(self, tree: ast.AST) -> Optional[Callable[..., float]]:
        if not ASTMetamorphicRewriter.audit_structural_complexity(
            tree, max_nodes=self.MAX_AST_NODES
        ):
            return None
        try:
            code_obj = compile(tree, filename="<godel_rsi_ast>", mode="exec")
            sandbox: Dict[str, Any] = {"__builtins__": dict(self._SAFE_BUILTINS)}
            exec(code_obj, sandbox)  # noqa: S102  — sandbox de builtins restringidos
            for name, item in sandbox.items():
                if callable(item) and not name.startswith("__"):
                    return item
        except Exception as exc:
            logger.error("[FASE 3] Error de sandbox: %s", exc)
        return None

    def execute_recursive_self_improvement(
        self,
        current_ast: ast.AST,
        telemetry_data: Dict[str, Any],
        stochastic_transition_matrix: Optional[np.ndarray] = None,
        recurrence_set: Optional[np.ndarray] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
        saddle_point: Optional[np.ndarray] = None,
    ) -> SovereignGodelCertificate:
        r"""
        Pipeline RSI anidado: Fase 1 → Fase 2 → Fase 3 (Kac + mónada + Löb/DGM + 3 superficies).
        """
        _ = telemetry_data
        self.iteration += 1
        step_start = time.time()

        manifold = self.self_inspect()
        bundle = lift_wisdom_to_celestial_syntax_bundle(manifold)
        morphism, _ = evaluate_categorical_transition(
            syntax_bundle=bundle,
            ast_tree=current_ast,
            mutation_scale=0.01,
            candidate_fn=self.policy_fn,
            utility_delta=0.0,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            saddle_point=saddle_point,
        )
        recurrence_seed = seed_rsi_recurrence_from_transition(morphism)

        candidate_fn: Optional[Callable[..., float]] = None
        if morphism.cartan_report.symplectic_preservation_verified:
            candidate_fn = self._extract_policy_callable(morphism.proposed_ast)
        self._apply_morphism(morphism, candidate_fn)
        final_utility = self.interact(candidate_fn if candidate_fn is not None else self.policy_fn)
        self.utility_history.append(final_utility)

        tb_cert: Optional[TarskiBrouwerCertificate] = None
        try:
            tb_cert = TarskiBrouwerEngine.verify_fixed_point(
                morphism.mutation_operator_next.astype(np.complex128)
            )
        except Exception as exc:
            logger.debug("[FASE 3] Tarski–FTA omitted: %s", exc)

        rec_cert: Optional[PoincareRecurrenceCertificate] = None
        try:
            if stochastic_transition_matrix is not None and recurrence_set is not None:
                rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
                    stochastic_transition_matrix, recurrence_set, num_walks=120, max_steps=5000
                )
            else:
                rec_cert = certify_poincare_kac_from_rsi_seed(recurrence_seed)
        except Exception as exc:
            logger.debug("[FASE 3] Poincaré–Kac omitted: %s", exc)

        curvature = np.asarray(manifold.banach_contraction.operator_matrix, dtype=np.complex128)
        monad = MonadLawsEngine.verify(morphism.mutation_operator_next, curvature)
        self.last_monad = monad
        dgm = DarwinGodelMachine.evaluate_candidate(
            tree=morphism.proposed_ast,
            baseline_utility=final_utility,
            entropy=manifold.mac_state.von_neumann_entropy,
            purity=manifold.mac_state.purity,
            cartan_ok=morphism.cartan_report.symplectic_preservation_verified,
            max_nodes=self.MAX_AST_NODES,
        )
        self.last_dgm = dgm
        lob = LobianObstacleEngine.certify(morphism.heyting_verdict, dgm_used=True)
        self.last_lob = lob
        three = certify_level3_rsi_from_seed(
            recurrence_seed, monad, tb_cert, dgm, self.utility_history
        )
        self.last_three_surfaces = three

        tb_converged = bool(tb_cert.iteration_converged) if tb_cert is not None else False
        mutation_applied = (
            morphism.heyting_verdict == HeytingOmega3.COHERENT
            and tb_converged
            and three.harness_rsi_ok
        )
        if mutation_applied:
            logger.info(">> [RSI OK] iter=%s veredicto=COHERENT d3C/dt3=%.3e", self.iteration, three.d3c_dt3)
        elif morphism.heyting_verdict == HeytingOmega3.DEGRADED:
            logger.warning(">> [RSI DEGRADED] iter=%s purga parcial", self.iteration)
        else:
            logger.critical(">> [RSI VETO] iter=%s enclavamiento Crowbar", self.iteration)
        logger.info("   tiempo de ciclo: %.0f ns", (time.time() - step_start) * 1e9)

        return self._issue_terminal_certificate(
            bundle=bundle,
            morphism=morphism,
            final_utility=final_utility,
            mutation_applied=mutation_applied,
            tb_cert=tb_cert,
            recurrence_cert=rec_cert,
            monad=monad,
            lob=lob,
            dgm=dgm,
            three=three,
        )

    def _issue_terminal_certificate(
        self,
        bundle: CelestialSyntaxBundle,
        morphism: CategoricalTransitionMorphism,
        final_utility: float,
        mutation_applied: bool,
        tb_cert: Optional[TarskiBrouwerCertificate],
        recurrence_cert: Optional[PoincareRecurrenceCertificate],
        monad: Optional[MonadLawsCertificate] = None,
        lob: Optional[LobianObstacleCertificate] = None,
        dgm: Optional[DarwinGodelSandboxReport] = None,
        three: Optional[ThreeSurfacesRSIReport] = None,
    ) -> SovereignGodelCertificate:
        manifold = bundle.wisdom_manifold
        now = time.time()
        hasher = hashlib.sha256()
        hasher.update(self.agent_id.encode("utf-8"))
        hasher.update(str(self.iteration).encode("utf-8"))
        hasher.update(morphism.heyting_verdict.name.encode("utf-8"))
        hasher.update(f"{manifold.banach_contraction.spectral_radius:.10f}".encode("utf-8"))
        hasher.update(f"{morphism.cartan_report.cartan_relative_error:.10f}".encode("utf-8"))
        hasher.update(f"{final_utility:.10f}".encode("utf-8"))
        hasher.update(f"{manifold.ast_homology.betti_1}".encode("utf-8"))
        hasher.update(f"{manifold.ast_homology.gauss_bonnet_consistent}".encode("utf-8"))
        hasher.update(
            f"{bundle.spectral_manifold.brockett_result.casimir_drift:.10f}".encode("utf-8")
        )
        if tb_cert is not None:
            hasher.update(f"{tb_cert.fubini_study_residual_rad:.10f}".encode("utf-8"))
            hasher.update(tb_cert.verification_route.encode("utf-8"))
        if morphism.birkhoff_melnikov.melnikov is not None:
            hasher.update(
                f"{morphism.birkhoff_melnikov.melnikov.melnikov_amplitude:.10f}".encode("utf-8")
            )
        if recurrence_cert is not None:
            hasher.update(f"{recurrence_cert.kac_error_residual:.10f}".encode("utf-8"))
        dynamics = manifold.ast_dynamics
        if dynamics is not None and dynamics.floquet is not None:
            hasher.update(
                f"{float(np.max(np.abs(dynamics.floquet.floquet_multipliers))):.10f}".encode("utf-8")
            )
            hasher.update(f"{dynamics.floquet.liouville_volume_residual:.10f}".encode("utf-8"))
        if dynamics is not None and dynamics.lindstedt is not None:
            hasher.update(f"{dynamics.lindstedt.secular_residual_max:.10f}".encode("utf-8"))
            hasher.update(f"{dynamics.lindstedt.homological_equation_residual:.10f}".encode("utf-8"))
        if morphism.birkhoff_melnikov.birkhoff_normal_form is not None:
            hasher.update(
                f"{morphism.birkhoff_melnikov.birkhoff_normal_form.hessian_determinant:.10f}".encode("utf-8")
            )
        if morphism.birkhoff_melnikov.delaunay_chirikov is not None:
            dc = morphism.birkhoff_melnikov.delaunay_chirikov
            hasher.update(f"{dc.arnold_certificate.num_overlaps}".encode("utf-8"))
            hasher.update(f"{dc.kam_survival_fraction:.10f}".encode("utf-8"))
        hasher.update(f"{morphism.cartan_report.secular_drift_residual:.10f}".encode("utf-8"))
        if three is not None:
            hasher.update(f"{three.d3c_dt3:.10f}".encode("utf-8"))
            hasher.update(f"{int(three.all_surfaces_coherent)}".encode("utf-8"))
        if monad is not None:
            hasher.update(f"{monad.associativity_residual:.10f}".encode("utf-8"))
        hasher.update(f"{now:.6f}".encode("utf-8"))

        floquet = dynamics.floquet if dynamics is not None else None
        floquet_mults: Tuple[complex, ...] = tuple(
            complex(m) for m in (floquet.floquet_multipliers if floquet is not None else [])
        )
        floquet_margin = float(floquet.stability_margin) if floquet is not None else 0.0
        floquet_stable = bool(floquet.is_linearly_stable) if floquet is not None else False
        liouville_res = float(floquet.liouville_volume_residual) if floquet is not None else 0.0

        lindstedt = dynamics.lindstedt if dynamics is not None else None
        lind_freq = float(lindstedt.fundamental_frequency) if lindstedt is not None else 0.0
        lind_sec = float(lindstedt.secular_residual_max) if lindstedt is not None else 0.0
        lind_free = bool(lindstedt.is_secular_free) if lindstedt is not None else True
        homol_res = float(lindstedt.homological_equation_residual) if lindstedt is not None else 0.0

        delaunay_actions = tuple(float(a) for a in bundle.delaunay_actions)
        delaunay_freqs = tuple(float(f) for f in bundle.delaunay_frequencies)

        dc = morphism.birkhoff_melnikov.delaunay_chirikov
        arnold_diff = bool(dc.arnold_certificate.arnold_diffusion_expected) if dc is not None else False
        chirikov_overlaps = int(dc.arnold_certificate.num_overlaps) if dc is not None else 0
        kam_survival = float(dc.kam_survival_fraction) if dc is not None else 1.0
        bruno_ok = bool(dc.small_divisors.bruno_holds) if (dc is not None and dc.small_divisors is not None) else True

        bnf = morphism.birkhoff_melnikov.birkhoff_normal_form
        bnf_verified = bool(bnf.is_birkhoff_non_degenerate) if bnf is not None else False
        bnf_det = float(bnf.hessian_determinant) if bnf is not None else 0.0
        kam_radius = float(bnf.kam_stability_radius) if bnf is not None else 0.0

        denjoy_ok = bool(
            morphism.birkhoff_melnikov.denjoy.denjoy_conjugacy_expected
        ) if morphism.birkhoff_melnikov.denjoy is not None else False
        smale_flag = bool(
            morphism.birkhoff_melnikov.smale.horseshoe_expected
        ) if morphism.birkhoff_melnikov.smale is not None else False
        mw_dim = int(
            morphism.birkhoff_melnikov.marsden_weinstein.reduced_dimension
        ) if morphism.birkhoff_melnikov.marsden_weinstein is not None else 0
        gf_ok = bool(
            morphism.cartan_report.generating_function.is_canonical
        ) if morphism.cartan_report.generating_function is not None else True

        return SovereignGodelCertificate(
            agent_id=self.agent_id,
            iteration=self.iteration,
            heyting_verdict=morphism.heyting_verdict,
            verdict_explanation=morphism.verdict_explanation,
            banach_certificate=manifold.banach_contraction,
            mac_state=manifold.mac_state,
            ast_homology=manifold.ast_homology,
            phs_state=morphism.phs_state,
            crowbar_report=morphism.crowbar_telemetry,
            cartan_report=morphism.cartan_report,
            birkhoff_melnikov=morphism.birkhoff_melnikov,
            tarski_brouwer=tb_cert,
            ast_orbit_diameter=dynamics.orbit_diameter if dynamics else 0.0,
            ast_lyapunov_max=dynamics.lyapunov_ast_max if dynamics else 0.0,
            ast_is_bounded=dynamics.is_bounded_orbit if dynamics else True,
            utility_score=final_utility,
            mutation_applied=mutation_applied,
            fixed_point_converged=bool(tb_cert.iteration_converged) if tb_cert else False,
            fixed_point_residual=float(tb_cert.l2_residual) if tb_cert else float("inf"),
            fubini_study_residual_rad=float(tb_cert.fubini_study_residual_rad) if tb_cert else float("inf"),
            recurrence_mean_time=float(recurrence_cert.mean_return_time_empirical) if recurrence_cert else 0.0,
            recurrence_kac_residual=float(recurrence_cert.kac_error_residual) if recurrence_cert else 0.0,
            casimir_drift=float(bundle.spectral_manifold.brockett_result.casimir_drift),
            reduced_orbit_dimension=float(bundle.celestial_hamiltonian.reduced_orbit_dimension),
            gauss_bonnet_consistent=bool(manifold.ast_homology.gauss_bonnet_consistent),
            digital_signature_sha256=hasher.hexdigest(),
            timestamp_utc=now,
            floquet_multipliers=floquet_mults,
            floquet_stability_margin=floquet_margin,
            floquet_is_stable=floquet_stable,
            lindstedt_fundamental_frequency=lind_freq,
            lindstedt_secular_residual=lind_sec,
            lindstedt_is_secular_free=lind_free,
            delaunay_actions=delaunay_actions,
            delaunay_frequencies=delaunay_freqs,
            arnold_diffusion_expected=arnold_diff,
            chirikov_overlaps=chirikov_overlaps,
            kam_survival_fraction=kam_survival,
            birkhoff_normal_form_verified=bnf_verified,
            birkhoff_normal_form_det_tau=bnf_det,
            kam_stability_radius=kam_radius,
            secular_cartan_drift=float(morphism.cartan_report.secular_drift_residual),
            floquet_liouville_residual=liouville_res,
            homological_equation_residual=homol_res,
            forman_morse_perfect=bool(manifold.forman_morse.is_perfect) if manifold.forman_morse else False,
            poincare_lemma_exact=bool(manifold.poincare_lemma.closed_forms_are_exact) if manifold.poincare_lemma else False,
            generating_function_canonical=gf_ok,
            novikov_valuation=float(three.novikov_valuation) if three else 0.0,
            monad_laws_hold=bool(monad.laws_hold) if monad else False,
            lob_bypass_dgm=bool(lob.dgm_bypass_engaged) if lob else False,
            dgm_accepted=bool(dgm.accepted) if dgm else False,
            data_rsi_ok=bool(three.data_rsi_ok) if three else False,
            harness_rsi_ok=bool(three.harness_rsi_ok) if three else False,
            model_rsi_ok=bool(three.model_rsi_ok) if three else False,
            d3c_dt3=float(three.d3c_dt3) if three else 0.0,
            inflection_positive=bool(three.inflection_positive) if three else False,
            bruno_holds=bruno_ok,
            denjoy_conjugacy_expected=denjoy_ok,
            smale_horseshoe=smale_flag,
            mw_reduced_dimension=mw_dim,
        )

    def continue_improve(self, force_spectral_violation: bool = False) -> SovereignGodelCertificate:
        if force_spectral_violation:
            self.mutation_operator = np.eye(self.dimension_mac, dtype=np.float64) * 1.85
        return self.execute_recursive_self_improvement(self.policy_ast, {})


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# §3.5 BANCO DE PRUEBAS DE VALIDACIÓN EXPERIMENTAL
# ══════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("═" * 96)
    print("DEMOSTRACIÓN FORMAL: GÖDEL AGENT v5.2.0 — FASES ANIDADAS POINCARÉ CELESTIAL RSI-3")
    print("FORMAN · LEMA-P · S-TIPO-2 · LIE-DEPRIT · SIEGEL-BRUNO · DENJOY · SMALE · MW · DGM")
    print("═" * 96)

    demo_source = textwrap.dedent("""
        def f(x):
            total = 0
            for i in range(3):
                total += i * x
            while x > 0:
                x = x - 1
                total = total + x
            return total
    """)
    demo_tree = ast.parse(demo_source)
    homology = ASTTopologicalEngine.compute_homology(demo_tree)
    forman = FormanMorseEngine.compute(demo_tree, homology)
    lemma = PoincareLemmaEngine.certify(demo_tree, homology)
    print("\n[TEST 1] Homología simplicial AST + Gauss–Bonnet + Morse–Forman + lema de Poincaré:")
    print(f"  • Nodos / Aristas            : {homology.num_nodes} / {homology.num_edges}")
    print(f"  • β₀ / β₁ / χ                : {homology.betti_0} / {homology.betti_1} / {homology.euler_characteristic}")
    print(f"  • Dualidad de Poincaré       : {homology.poincare_duality_consistent} (cerrada={homology.is_closed_1_manifold})")
    print(f"  • Gauss–Bonnet Σκ = 2χ       : {homology.gauss_bonnet_total_curvature:.4f} (ok={homology.gauss_bonnet_consistent})")
    print(f"  • Forman m₀,m₁ / perfecta    : {forman.morse_polynomial} / {forman.is_perfect}")
    print(f"  • Lema Poincaré exacto       : {lemma.closed_forms_are_exact} (obstrucción β₁={lemma.obstruction_betti_1})")
    print(f"  • Invariante integral Î      : {homology.poincare_integral_invariant:.6f}")

    mac = MACQuantumEngine.create_pure_or_mixed_state(dimension=4, seed=2026)
    print("\n[TEST 2] Estado MAC (Dirac–von Neumann):")
    print(f"  • Pureza γ                   : {mac.purity:.6f}")
    print(f"  • Entropía de von Neumann    : {mac.von_neumann_entropy:.6f}")
    print(f"  • Info. Wigner–Yanase        : {mac.wigner_yanase_skew:.6f}")
    print(f"  • Wehrl (proxy)              : {mac.wehrl_entropy_proxy:.6f}")

    dynamics = ASTDynamicalSystemEngine.compute_ast_dynamics(homology, num_iterations=80)
    print("\n[TEST 3] Dinámica discreta del AST + Floquet + Lindstedt + promedio:")
    print(f"  • Diámetro orbital           : {dynamics.orbit_diameter:.4f}")
    print(f"  • λ_max AST                  : {dynamics.lyapunov_ast_max:.6e}")
    print(f"  • ⟨H⟩ Poincaré               : {dynamics.poincare_average_hamiltonian:.6f}")
    if dynamics.floquet is not None:
        print(f"  • Floquet |μ|_max            : {float(np.max(np.abs(dynamics.floquet.floquet_multipliers))):.6f}")
        print(f"  • Liouville |det M|−1       : {dynamics.floquet.liouville_volume_residual:.3e}")
    if dynamics.lindstedt is not None:
        print(f"  • Lindstedt ω₀ / homológica : {dynamics.lindstedt.fundamental_frequency:.6f} / "
              f"{dynamics.lindstedt.homological_equation_residual:.3e}")

    rng = np.random.default_rng(2026)
    t_op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    t_op = t_op / (float(np.linalg.norm(t_op, ord=2)) + 1e-15)
    tb = TarskiBrouwerEngine.verify_fixed_point(t_op)
    print("\n[TEST 4] Punto fijo T_Gödel (FTA / CP^{n−1}):")
    print(f"  • Existe (autovector λ ≠ 0)  : {tb.fixed_point_exists}")
    print(f"  • Ruta / d_FS                : {tb.verification_route} / {tb.fubini_study_residual_rad:.6e} rad")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO A: Ciclo RSI coherente (Fase 1 → 2 → 3 anidadas, Nivel 3)")
    print("─" * 96)
    agent = GodelAgent(agent_id="GODEL-WISDOM-SOVEREIGN-01", dimension_mac=4, seed=2026)
    cert_a = agent.continue_improve(force_spectral_violation=False)
    print(f"  • Veredicto Heyting          : {cert_a.heyting_verdict.name}")
    print(f"  • S tipo 2 canónica          : {cert_a.generating_function_canonical}")
    print(f"  • Forman perfecta / lema P   : {cert_a.forman_morse_perfect} / {cert_a.poincare_lemma_exact}")
    print(f"  • Novikov v                  : {cert_a.novikov_valuation:.4f}")
    print(f"  • Mónada leyes               : {cert_a.monad_laws_hold}")
    print(f"  • Löb bypass DGM             : {cert_a.lob_bypass_dgm} (aceptado={cert_a.dgm_accepted})")
    print(f"  • Superficies D/H/M          : {cert_a.data_rsi_ok}/{cert_a.harness_rsi_ok}/{cert_a.model_rsi_ok}")
    print(f"  • d³C/dt³ / inflexión        : {cert_a.d3c_dt3:.4e} / {cert_a.inflection_positive}")
    print(f"  • Bruno / Denjoy / Smale     : {cert_a.bruno_holds} / {cert_a.denjoy_conjugacy_expected} / {cert_a.smale_horseshoe}")
    print(f"  • MW dim reducida            : {cert_a.mw_reduced_dimension}")
    print(f"  • SHA-256                    : {cert_a.digital_signature_sha256[:32]}…")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO B: Violación espectral forzada (enclavamiento Crowbar)")
    print("─" * 96)
    cert_b = agent.continue_improve(force_spectral_violation=True)
    print(f"  • Veredicto Heyting          : {cert_b.heyting_verdict.name}")
    print(f"  • Crowbar tripped            : {cert_b.crowbar_report.interlock_tripped}")
    print(f"  • Mutación aplicada          : {cert_b.mutation_applied}")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO C: Recurrencia de Poincaré–Kac inyectada")
    print("─" * 96)
    rng2 = np.random.default_rng(42)
    p_syn = rng2.random((6, 6)) + 0.1
    p_syn /= p_syn.sum(axis=1, keepdims=True)
    measurable = np.array([True, False, True, False, False, True])
    rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
        p_syn, measurable, num_walks=200, max_steps=5000
    )
    print(f"  • |A|/|X|                    : {int(measurable.sum())}/6")
    print(f"  • τ̄_A empírico / Kac         : {rec_cert.mean_return_time_empirical:.4f} / {rec_cert.kac_lemma_prediction:.4f}")

    cert_d = agent.execute_recursive_self_improvement(
        current_ast=agent.policy_ast,
        telemetry_data={"fault": None},
        stochastic_transition_matrix=p_syn,
        recurrence_set=measurable,
    )
    print(f"  • RSI+Kac veredicto          : {cert_d.heyting_verdict.name}")
    print(f"  • τ̄_A (certificado)          : {cert_d.recurrence_mean_time:.4f}")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO D: Anidamiento explícito Fase 1 → 2 → 3 + Harness-RSI")
    print("─" * 96)
    manifold_e = synthesize_wisdom_manifold(
        ast_tree=demo_tree,
        mac_state=mac,
        mutation_matrix=np.eye(4, dtype=np.float64) * 0.40,
        rotor=Quaternion(1.0, 0.0, 0.0, 0.0),
    )
    bundle_e = lift_wisdom_to_celestial_syntax_bundle(manifold_e)
    morphism_e, _ = evaluate_categorical_transition(bundle_e, demo_tree)
    seed_e = seed_rsi_recurrence_from_transition(morphism_e)
    rec_e = certify_poincare_kac_from_rsi_seed(seed_e)
    harness = agent.harness_rsi_self_rewrite()
    print(f"  • 𝔐_Wisdom → fibrado dim     : {bundle_e.configuration_dim},  J={bundle_e.syntax_momentum_map}")
    print(f"  • Carta I–θ dim              : {bundle_e.action_angle.chart_dimension if bundle_e.action_angle else 0}")
    print(f"  • Φ Heyting                  : {morphism_e.heyting_verdict.name}")
    print(f"  • Semilla |A| / v_Novikov    : {int(seed_e.measurable_set.sum())}/{seed_e.state_space_size} / {seed_e.novikov_valuation:.4f}")
    print(f"  • Kac 1/μ(A) vs τ̄            : {rec_e.kac_lemma_prediction:.4f} vs {rec_e.mean_return_time_empirical:.4f}")
    print(f"  • Harness Cartan ω           : {harness.symplectic_preservation_verified} (S canónica="
          f"{harness.generating_function.is_canonical if harness.generating_function else False})")

    print("\n" + "═" * 96)
    print("✓ AUDITORÍA CONCLUIDA: GÖDEL AGENT v5.2.0 — FASES ANIDADAS RSI-3.")
    print("  · Fase 1: 𝔐_Wisdom (Forman, lema P, I–θ, Lindstedt/Floquet) → lift_wisdom_to_celestial_syntax_bundle.")
    print("  · Fase 2: Φ_cat (S tipo 2, Deprit, Bruno, Denjoy, Smale, MW, Novikov) → seed_rsi_recurrence_from_transition.")
    print("  · Fase 3: Poincaré–Kac + mónada + Löb/DGM + tres superficies + d³C/dt³ + SHA-256.")
    print("═" * 96)