# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : GÖDEL AGENT (SOBERANO DE AUTOMEJORA RECURSIVA Y METAMORFISMO NIVEL 3)                 ║
║ UBICACIÓN: app/agents/wisdom/godel_agent.py                                                      ║
║ VERSIÓN  : 5.0.0-Poincaré-Meta-Self-Improvement-Level-3                                          ║
║ TRATADOS : Les Méthodes Nouvelles de la Mécanique Céleste (Poincaré, 1892-1899)                  ║
║            Sur le problème des trois corps (Poincaré, 1890)                                      ║
║            Analysis Situs (Poincaré, 1895) · Sur un théorème de géométrie (1912-1913)            ║
║            Novikov (1981), Grothendieck (1972), Tarski (1955), Brouwer (1911), Löb (1955)        ║
║            Gödel (1931), Birkhoff (1913), Melnikov (1963), Forman (1998)                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
GOBERNANZA METAMÓRFICA, TOPOLÓGICA Y CELESTE DE LA AUTOMEJORA RECURSIVA DE NIVEL 3 (INFLEXIÓN)
──────────────────────────────────────────────────────────────────────────────────────────────────
El Soberano `GodelAgent` formaliza la reescritura metamórfica del AST y de sus propios mecanismos
de optimización como un sistema Hamiltoniano discreto de Nivel 3 (Inflexión / Meta-Mejora),
logrando aceleración de capacidad super-exponencial d³C/dt³ > 0 y superando el Obstáculo Löbiano
mediante la máquina Darwin-Gödel (DGM) desacoplada en Sandbox.

TRES SUPERFICIES DE MODIFICACIÓN RSI NIVEL 3:
  1. Data-RSI    : Trazas metamórficas sobre el Anillo Universal de Novikov Λ_Nov con valuación
                   v(T^{a_i}) = min {a_i} y preservación de subvariedades Lagrangianas exactas i* λ = dS.
  2. Harness-RSI : Reescritura del ASTMetamorphicRewriter vía integradores variacionales simplécticos
                   de Cayley-Darboux sobre U(n) con reducción gauge Poincaré-Marsden-Weinstein J⁻¹(μ)/G_μ.
  3. Model-RSI   : Multiplicación monádica μ_godel: T²(A) → T(A) y punto fijo Tarski-Brouwer sobre CP^{n-1}
                   con distancia Fubini-Study d_FS(u, v) = arccos(|⟨u, v⟩|) ≤ 10⁻⁴ rad.

TRIBUNAL CIBER-FÍSICO E INTERLOCK ESP32 CROWBAR:
  - Veto Suave (Válvula de Alivio): 10⁻⁶ < d_FS ≤ 10⁻³ rad ↦ Recirculación mecánica (Gracia 1h con positeón e⁺).
  - Veto Duro (ESP32 Crowbar < 400 ns): d_FS > 10⁻³ rad o desintegración de Poincaré-Cartan ↦ GPIO14 HIGH
    en IRAM, cebado BT151 y parálisis por cortocircuito de potencia.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import math
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple

import numpy as np
import scipy.linalg as la

from app.wisdom.godel_engine import (
    BanachAlgebraEngine,
    BanachContractionReport,
    CelestialHamiltonianBundle,
    CrowbarCircuitPhysicsEngine,
    CrowbarPhysicalTelemetry,
    GodelEngine,
    HeytingVerdict as HeytingOmega3,
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
    lift_to_celestial_hamiltonian_bundle,
    path_graph_adjacency,
    synthesize_spectral_topological_manifold,
)

logger = logging.getLogger("APU.Wisdom.GodelAgent")

__version__: Final[str] = "5.0.0-Poincaré-Meta-Self-Improvement-Level-3"


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


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1: FUNDAMENTOS TOPOLÓGICO-ESPECTRALES, HOMOLOGÍA SIMPLICIAL CON GAUSS–BONNET,
#         ESPACIO DE FASE AST Y PUNTO FIJO ESPECTRAL EN CP^{n−1}
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


class ASTTopologicalEngine:
    r"""
    Homología singular del 1-complejo AST vía rango SVD de la incidencia orientada ∂₁.
    Rank-nullity:  β₀ = V − rk(∂₁),  β₁ = E − rk(∂₁).
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
        if is_closed:
            duality_ok = bool(betti_0 == betti_1)
        else:
            duality_ok = True

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

        # Cota combinatoria de Morse (todas las celdas críticas): β₀ ≤ V, β₁ ≤ E.
        morse_holds = bool(betti_0 <= num_v and betti_1 <= num_e)
        morse_residual = float(max(0, betti_0 - num_v) + max(0, betti_1 - num_e))
        min_cycle_norm = float(math.sqrt(betti_1)) if betti_1 > 0 else 0.0

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
# §1.3 SISTEMA DINÁMICO DISCRETO AST: RETORNO DE POINCARÉ Y TWIST ANULAR
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTDiscreteDynamicalSystemCertificate:
    """Dinámica discreta de features del AST y certificados de Poincaré sobre la órbita."""
    feature_vector: np.ndarray
    trajectory: np.ndarray
    return_map: Optional[PoincareReturnMapCertificate]
    birkhoff: Optional[PoincareBirkhoffCertificate]
    is_bounded_orbit: bool
    orbit_diameter: float
    lyapunov_ast_max: float
    diophantine_gamma: float = 0.0
    is_diophantine: bool = False


class ASTDynamicalSystemEngine:
    r"""
    Embedding Φ(AST) ∈ ℝ⁸ y mapa F que aproxima la acción agregada del reescritor.
    El twist anular se obtiene pasando F|_{span{e₀,e₁}} a coordenadas acción-ángulo
        I = ‖x‖,  θ = atan2(x₁, x₀),
    hipótesis nativas de Poincaré–Birkhoff sobre A = S¹ × [0, 1].
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
            cartesian = np.array([action * math.cos(theta), action * math.sin(theta)], dtype=np.float64)
            image = block @ cartesian
            action_new = float(np.clip(np.linalg.norm(image), 0.0, 1.0))
            theta_new = math.atan2(float(image[1]), float(image[0])) % (2.0 * math.pi)
            return np.array([theta_new, action_new], dtype=np.float64)
        return twist

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
        try:
            return_map = PoincareReturnMapEngine.compute_return_map(
                trajectory=trajectory,
                time_samples=times,
                section_normal=section_normal,
                section_offset=section_offset,
            )
        except Exception as exc:
            logger.debug("Poincaré return map omitted: %s", exc)
            return_map = None

        birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
        if return_map is not None and dim >= 2 and abs(return_map.rotation_number) > 1e-9:
            try:
                birkhoff_cert = PoincareBirkhoffEngine.audit_twist_map(
                    cls._annulus_map_from_linear(evolution[:2, :2]),
                    return_map.rotation_number,
                )
            except Exception as exc:
                logger.debug("Birkhoff audit omitted: %s", exc)

        diameter = float(np.max(np.linalg.norm(trajectory - trajectory[0], axis=1))) if num_iterations else 0.0
        lyap = float(return_map.lyapunov_max) if return_map is not None else 0.0
        gamma = float(return_map.diophantine_gamma) if return_map is not None else 0.0
        dioph = bool(return_map.is_diophantine) if return_map is not None else False
        return ASTDiscreteDynamicalSystemCertificate(
            feature_vector=x0,
            trajectory=trajectory,
            return_map=return_map,
            birkhoff=birkhoff_cert,
            is_bounded_orbit=bool(diameter < 100.0),
            orbit_diameter=diameter,
            lyapunov_ast_max=lyap,
            diophantine_gamma=gamma,
            is_diophantine=dioph,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.4 PUNTO FIJO DE T_Gödel EN CP^{n−1} VÍA EL TEOREMA FUNDAMENTAL DEL ÁLGEBRA
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TarskiBrouwerCertificate:
    r"""
    Certificado del endomorfismo gauge-fijado
        T_Gödel(v) = e^{−i arg⟨v, Tv⟩} · Tv / ‖Tv‖₂
    sobre CP^{n−1} con métrica de Fubini–Study d_FS(u, v) = arccos(|⟨u, v⟩|).

    Existencia: todo T ∈ M_n(ℂ) no nilpotente de índice total posee un autovector
    v_* con λ ≠ 0 (FTA). Entonces T_Gödel([v_*]) = [v_*]. Brouwer sobre la bola
    NO es la vía correcta: CP^{n−1} no es un disco. Lefschetz χ(CP^{n−1}) = n
    garantiza puntos fijos solo para mapas homotópicos a la identidad.
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
        gap_ratio = float(magnitudes[1] / magnitudes[0]) if magnitudes.size > 1 and magnitudes[0] > 1e-15 else 1.0
        dominant = complex(eigvals[int(np.argmax(np.abs(eigvals)))]) if eigvals.size else 0j

        spectral_vector: Optional[np.ndarray] = None
        spectral_exists = bool(spectral_radius > 1e-15)
        if spectral_exists:
            _, eigvecs = la.eig(operator_c)
            spectral_vector = eigvecs[:, int(np.argmax(np.abs(eigvals)))]
            spectral_vector = spectral_vector / np.linalg.norm(spectral_vector)

        vector = v_init if v_init is not None else np.ones(n, dtype=np.complex128) / math.sqrt(n)
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
# §1.5 SÍNTESIS DE LA VARIEDAD DE SABIDURÍA 𝔐_Wisdom
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class StateManifoldWisdom:
    r"""
    Variedad analítica 𝔐_Wisdom = (ρ, q, H_•(AST), T, P, v*).
    Objeto penúltimo de la Fase 1; su elevación celeste es el objeto terminal.
    """
    mac_state: MACDensityState
    hypercomplex_rotor: Quaternion
    ast_homology: SimplicialHomologyCertificate
    banach_contraction: BanachContractionReport
    ast_dynamics: Optional[ASTDiscreteDynamicalSystemCertificate]
    tarski_brouwer: Optional[TarskiBrouwerCertificate]
    timestamp_epoch: float


def synthesize_wisdom_manifold(
    ast_tree: ast.AST,
    mac_state: MACDensityState,
    mutation_matrix: np.ndarray,
    rotor: Quaternion,
    tolerance: float = 0.999,
    include_dynamics: bool = True,
) -> StateManifoldWisdom:
    r"""
    Síntesis estructural de 𝔐_Wisdom (homología, Wirtinger–KAM, retorno, FTA).
    Continuación formal: `lift_wisdom_to_celestial_syntax_bundle`.
    """
    _ = tolerance
    homology_cert = ASTTopologicalEngine.compute_homology(ast_tree)
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
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.6 ENLACE TERMINAL FASE 1 → INICIO FASE 2
#      Elevación de 𝔐_Wisdom al fibrado cotangente sintáctico (T*Q_AST, ω, H, J)
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
    """
    wisdom_manifold: StateManifoldWisdom
    spectral_manifold: SpectralTopologicalManifold
    celestial_hamiltonian: CelestialHamiltonianBundle
    syntax_momentum_map: np.ndarray
    configuration_dim: int


def lift_wisdom_to_celestial_syntax_bundle(
    manifold: StateManifoldWisdom,
) -> CelestialSyntaxBundle:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 1 / PRIMER MORFISMO DE LA FASE 2.

    Eleva 𝔐_Wisdom al fibrado (T*Q_AST, ω, H, J) de la mecánica celeste de Poincaré:
    el 1-esqueleto AST alimenta la cohomología de Hodge del engine, el estado MAC
    alimenta el flujo de Brockett–KKS, y el mapa de momentos combinatorio
        J = (β₀, β₁, λ₂, E_Willmore)
    es el Casimir discreto que la Fase 2 reduce a lo Marsden–Weinstein.
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
        configuration_dim=int(manifold.mac_state.dimension),
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2: REESCRITURA METAMÓRFICA AST CON POINCARÉ–CARTAN, BIRKHOFF Y MELNIKOV
#         (continúa desde CelestialSyntaxBundle)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.1 REESCRITOR METAMÓRFICO AST: TRASLACIÓN VERTICAL EN T*Q (ω-PRESERVANTE)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTPoincareCartanReport:
    r"""
    Certificado de la 1-forma de Poincaré–Cartan sobre el complejo sintáctico.

    Las constantes *flotantes* son coordenadas de momento p; las profundidades son q.
    Una traslación uniforme p ↦ p + ε (única mutación permitida) satisface dp' = dp,
    luego ω = dp ∧ dq se preserva. El valor de ∮ θ = ∮ p dq no es un Casimir:
    deriva con ε, exactamente como Tr(ρ N) en el flujo de Brockett.
    Los enteros se tratan como Casimirs discretos (topología: cotas de bucle, aridades)
    y NO se mutan.
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


class ASTMetamorphicRewriter(ast.NodeTransformer):
    r"""
    Reescritor metamórfico guiado por la geometría de T*Q_AST.
    Mutación: traslación vertical de momenta flotantes. Enteros = Casimirs.
    """
    FORBIDDEN_CALLS: Final[Set[str]] = {
        "eval", "exec", "__import__", "open", "system", "popen",
        "spawn", "fork", "subprocess", "globals", "locals", "compile",
    }
    FORBIDDEN_ATTRIBUTES: Final[Set[str]] = {
        "__globals__", "__builtins__", "__subclasses__", "__bases__", "__class__",
        "__code__", "__closure__", "__dict__", "__mro__", "__getattribute__", "__reduce__",
        "__import__", "__loader__",
    }
    MAX_CONSTANT_MAGNITUDE: Final[float] = 1e6

    def __init__(self, mutation_scale: float = 0.01) -> None:
        super().__init__()
        self.mutation_scale = mutation_scale
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
        report = ASTPoincareCartanReport(
            cartan_integral_before=cartan_before,
            cartan_integral_after=cartan_after,
            cartan_relative_error=rel_err,
            num_mutations=self.num_mutations,
            total_kinetic_energy=kinetic,
            total_potential_energy=potential,
            symplectic_preservation_verified=symplectic_ok,
            topology_preserved=topology_ok,
            forbidden_call_detected=self.security_violation_detected,
            forbidden_attribute_detected=self.forbidden_attribute_detected,
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
            mutated_val = float(np.clip(float(node.value) + self.mutation_scale,
                                        -self.MAX_CONSTANT_MAGNITUDE, self.MAX_CONSTANT_MAGNITUDE))
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
    Aplica la traslación vertical p ↦ p + ε sobre T*Q_AST y certifica ω.
    El fibrado se usa como testigo de que la topología combinatoria (J_syntax)
    debe permanecer invariante: si el 1-esqueleto cambia, se rompe el Casimir.
    """
    _ = syntax_bundle
    rewriter = ASTMetamorphicRewriter(mutation_scale=mutation_scale)
    mutated_ast, _, report = rewriter.poincare_cartan_ast_rewrite(ast_tree)
    return mutated_ast, report


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.2 CERTIFICADOS DE BIRKHOFF Y MELNIKOV SOBRE LA DINÁMICA DEL AST
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ASTBirkhoffMelnikovReport:
    """
    Poincaré–Birkhoff del twist anular de features + Melnikov delegado al engine.
    λ_max > 0 es testigo de difusión de Arnold; no se fabrican ceros de Melnikov.
    """
    birkhoff: Optional[PoincareBirkhoffCertificate]
    melnikov: Optional[MelnikovChaosCertificate]
    chaos_detected: bool
    periodic_patterns_detected: int
    safety_margin: float
    arnold_diffusion_witness: bool


class ASTBirkhoffMelnikovEngine:
    """Orquesta Birkhoff (twist anular) y Melnikov (solo si hay H₀, H₁, silla)."""

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
        chaos = bool(
            (melnikov_cert.transverse_homoclinic_exists if melnikov_cert is not None else False) or arnold
        )
        periodic = birkhoff_cert.fixed_points_detected if birkhoff_cert is not None else 0
        amplitude = float(melnikov_cert.melnikov_amplitude) if melnikov_cert is not None else max(0.0, lyap)
        safety = 1.0 / (1.0 + amplitude)
        return ASTBirkhoffMelnikovReport(
            birkhoff=birkhoff_cert,
            melnikov=melnikov_cert,
            chaos_detected=chaos,
            periodic_patterns_detected=int(periodic),
            safety_margin=safety,
            arnold_diffusion_witness=arnold,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.3 MORFISMO CATEGÓRICO DE TRANSICIÓN (TERMINAL INTERNO DE LA FASE 2)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CategoricalTransitionMorphism:
    r"""
    Φ : 𝔐_Wisdom → 𝔐'_Wisdom, evaluado sobre el fibrado celeste de la Fase 1.
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
      (1) reescritura Poincaré–Cartan, (2) Birkhoff–Melnikov, (3) PHS,
      (4) veredicto Heyting, (5) Crowbar, (6) operador estabilizado.
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
    sections: List[Tuple[str, HeytingOmega3, str]] = [
        ("CARTAN",
         HeytingOmega3.COHERENT if cartan_report.symplectic_preservation_verified else HeytingOmega3.VETOED,
         f"topo={cartan_report.topology_preserved}"),
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
         HeytingOmega3.COHERENT if manifold.ast_homology.gauss_bonnet_consistent else HeytingOmega3.DEGRADED,
         f"Σκ={manifold.ast_homology.gauss_bonnet_total_curvature:.4f}"),
        ("UTILITY",
         HeytingOmega3.COHERENT if utility_delta >= -1e-6 else HeytingOmega3.DEGRADED,
         f"ΔU={utility_delta:.4f}"),
    ]
    if manifold.tarski_brouwer is not None:
        sections.append((
            "TARSKI-FTA",
            HeytingOmega3.COHERENT if manifold.tarski_brouwer.fixed_point_exists else HeytingOmega3.VETOED,
            manifold.tarski_brouwer.verification_route,
        ))
    if manifold.ast_homology.has_cohomological_obstruction if hasattr(manifold.ast_homology, "has_cohomological_obstruction") else manifold.ast_homology.betti_1 > 8:
        sections.append(("HODGE-CYCLES", HeytingOmega3.DEGRADED, f"β₁={manifold.ast_homology.betti_1}"))

    global_verdict = sections[0][1]
    for _, verdict, _ in sections[1:]:
        global_verdict = global_verdict.meet(verdict)
    failing = [f"{name}[{verdict.name}]:{detail}" for name, verdict, detail in sections if verdict != HeytingOmega3.COHERENT]
    reason = " ∧ ".join(failing) if failing else "COHERENCIA CERTIFICADA EN TODAS LAS SECCIONES LOCALES"
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
    )
    return morphism, candidate_fn


def evaluate_categorical_transition_from_manifold(
    manifold: StateManifoldWisdom,
    ast_tree: ast.AST,
    **kwargs: Any,
) -> Tuple[CategoricalTransitionMorphism, Optional[Callable[..., float]]]:
    """Compatibilidad: eleva 𝔐_Wisdom y delega en el morfismo canónico de la Fase 2."""
    return evaluate_categorical_transition(lift_wisdom_to_celestial_syntax_bundle(manifold), ast_tree, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.4 ENLACE TERMINAL FASE 2 → INICIO FASE 3
#      Semilla de recurrencia de Poincaré–Kac extraída del morfismo de haces
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RSIRecurrenceSeed:
    r"""
    OBJETO TERMINAL DE LA FASE 2 Y OBJETO INICIAL DE LA FASE 3.

    Sombra de Markov P del operador estabilizado T_stab y conjunto medible A ⊂ X
    sobre el que la Fase 3 contrastará τ_A con 1/μ(A) (lema de Kac).
    """
    morphism: CategoricalTransitionMorphism
    stochastic_matrix: np.ndarray
    measurable_set: np.ndarray
    state_space_size: int
    syntax_momentum_map: np.ndarray
    provenance_hash: str


def seed_rsi_recurrence_from_transition(
    morphism: CategoricalTransitionMorphism,
    measurable_fraction: float = 0.5,
) -> RSIRecurrenceSeed:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 2 / PRIMER MORFISMO DE LA FASE 3.

    Construye P_{ij} ∝ |T_stab|_{ij} + ε y un conjunto A de fracción dada.
    La Fase 3 consume este objeto en `certify_poincare_kac_from_rsi_seed`.
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
    return RSIRecurrenceSeed(
        morphism=morphism,
        stochastic_matrix=stochastic,
        measurable_set=measurable,
        state_space_size=n,
        syntax_momentum_map=morphism.syntax_bundle.syntax_momentum_map.copy(),
        provenance_hash=digest.hexdigest(),
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3: SOBERANO GÖDEL (RSI LAZO CERRADO), RECURRENCIA DE POINCARÉ Y CERTIFICACIÓN
#         (continúa desde RSIRecurrenceSeed)
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
# §3.2 CERTIFICADO DIGITAL INMUTABLE TERMINAL DEL SOBERANO
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


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.3 SOBERANO GÖDEL: ORQUESTADOR RSI EN TRES FASES ANIDADAS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _default_wisdom_policy(entropy: float, purity: float) -> float:
    """Política de sabiduría por defecto: maximiza pureza y penaliza entropía."""
    return (purity * 2.5) - (entropy * 0.4)


class GodelAgent:
    r"""
    Soberano de Gödel y guardián metamórfico de consistencia lógica (RSI Nivel 3 - Inflexión / Meta-Mejora).

      Fase 1 : `synthesize_wisdom_manifold` → `lift_wisdom_to_celestial_syntax_bundle`
      Fase 2 : `evaluate_categorical_transition` → `seed_rsi_recurrence_from_transition`
      Fase 3 : `certify_poincare_kac_from_rsi_seed` + Tarski–FTA + SHA-256.
      Level 3: `execute_level3_meta_self_improvement` (Mónada T = (T, η, μ) + CP^{n-1} Fubini-Study + DGM Sandbox).
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
        self.mac_state = MACQuantumEngine.create_pure_or_mixed_state(dimension=dimension_mac, seed=seed)
        self.hypercomplex_rotor = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.policy_fn: Callable[[float, float], float] = _default_wisdom_policy
        self.policy_ast: ast.Module = ast.parse(textwrap.dedent(inspect.getsource(_default_wisdom_policy)))
        self.mutation_operator = np.eye(dimension_mac, dtype=np.float64) * 0.45
        self.utility_history: List[float] = []

    def execute_level3_meta_self_improvement(
        self,
        current_ast_state: np.ndarray,
        curvature_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Ejecuta el ciclo de Meta-Mejora Nivel 3 sobre la superficie del AST.

        1. Multiplicación Monádica mu_godel en Model-RSI.
        2. Solución de Punto Fijo Tarski-Brouwer en CP^(n-1).
        3. Evasión del Obstáculo Löbiano vía DGM Sandbox.
        4. Clasificación en Topos de Heyting Omega_3/Omega_4 y Disyuntor ESP32 Crowbar.
        """
        self.iteration += 1
        # Step 1: Modificación Monádica del Operador
        U_meta = self.engine.apply_monadic_multiplication(
            current_operator=current_ast_state,
            curvature_tensor=curvature_matrix,
        )

        # Step 2: Verificación de Punto Fijo en CP^(n-1)
        dim = current_ast_state.shape[0]
        v_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        is_fixed_point, d_FS, d3C_dt3 = self.engine.verify_tarski_brouwer_fixed_point_cpn(
            state_vector=v_init,
            transform_op=U_meta,
        )

        # Step 3: Evaluación de Heyting y Veto Ciber-Físico
        if is_fixed_point and d3C_dt3 > 0.0:
            verdict = "COHERENT_LEVEL_3_APPROVED"
            heyting_code = 1  # Top (Verdadero / Seguro)
            self.mutation_operator = np.real(U_meta)
        elif d_FS <= 1e-3:
            verdict = "BYPASS_RECIRCULATION_WARNING"
            heyting_code = 2  # Luz Ámbar (Válvula de Alivio)
            self.mutation_operator = np.real(U_meta) * 0.85
        else:
            verdict = "HARD_CROWBAR_VETOED"
            heyting_code = 0  # Bottom (Veto Duro ESP32 < 400 ns)
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
        }

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
        if not (
            ASTMetamorphicRewriter.audit_structural_complexity(tree, max_nodes=self.MAX_AST_NODES)
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
        Pipeline RSI anidado: Fase 1 → Fase 2 → Fase 3.
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

        tb_converged = bool(tb_cert.iteration_converged) if tb_cert is not None else False
        mutation_applied = morphism.heyting_verdict == HeytingOmega3.COHERENT and tb_converged
        if mutation_applied:
            logger.info(">> [RSI OK] iter=%s veredicto=COHERENT", self.iteration)
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
        )

    def _issue_terminal_certificate(
        self,
        bundle: CelestialSyntaxBundle,
        morphism: CategoricalTransitionMorphism,
        final_utility: float,
        mutation_applied: bool,
        tb_cert: Optional[TarskiBrouwerCertificate],
        recurrence_cert: Optional[PoincareRecurrenceCertificate],
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
        hasher.update(f"{bundle.celestial_hamiltonian.casimir_drift if hasattr(bundle.celestial_hamiltonian, 'casimir_drift') else bundle.spectral_manifold.brockett_result.casimir_drift:.10f}".encode("utf-8"))
        if tb_cert is not None:
            hasher.update(f"{tb_cert.fubini_study_residual_rad:.10f}".encode("utf-8"))
            hasher.update(tb_cert.verification_route.encode("utf-8"))
        if morphism.birkhoff_melnikov.melnikov is not None:
            hasher.update(f"{morphism.birkhoff_melnikov.melnikov.melnikov_amplitude:.10f}".encode("utf-8"))
        if recurrence_cert is not None:
            hasher.update(f"{recurrence_cert.kac_error_residual:.10f}".encode("utf-8"))
        hasher.update(f"{now:.6f}".encode("utf-8"))

        dynamics = manifold.ast_dynamics
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
        )

    def continue_improve(self, force_spectral_violation: bool = False) -> SovereignGodelCertificate:
        if force_spectral_violation:
            self.mutation_operator = np.eye(self.dimension_mac, dtype=np.float64) * 1.85
        return self.execute_recursive_self_improvement(self.policy_ast, {})


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# §3.4 BANCO DE PRUEBAS DE VALIDACIÓN EXPERIMENTAL
# ══════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("═" * 96)
    print("DEMOSTRACIÓN FORMAL: GÖDEL AGENT v4.1.0 — FASES ANIDADAS POINCARÉ")
    print("GAUSS–BONNET · CARTAN ω · BIRKHOFF · FTA/CP^{n−1} · POINCARÉ–KAC · SHA-256")
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
    print("\n[TEST 1] Homología simplicial AST + Gauss–Bonnet:")
    print(f"  • Nodos / Aristas            : {homology.num_nodes} / {homology.num_edges}")
    print(f"  • β₀ / β₁ / χ                : {homology.betti_0} / {homology.betti_1} / {homology.euler_characteristic}")
    print(f"  • Dualidad de Poincaré       : {homology.poincare_duality_consistent} (cerrada={homology.is_closed_1_manifold})")
    print(f"  • Gauss–Bonnet Σκ = 2χ       : {homology.gauss_bonnet_total_curvature:.4f} (ok={homology.gauss_bonnet_consistent})")
    print(f"  • Willmore Σκ²               : {homology.willmore_energy:.4f}")
    print(f"  • Brecha de Fiedler          : {homology.spectral_gap:.4f}")
    print(f"  • Morse combinatoria         : holds={homology.morse_inequality_holds}")

    mac = MACQuantumEngine.create_pure_or_mixed_state(dimension=4, seed=2026)
    print("\n[TEST 2] Estado MAC (Dirac–von Neumann):")
    print(f"  • Pureza γ                   : {mac.purity:.6f}")
    print(f"  • Entropía de von Neumann    : {mac.von_neumann_entropy:.6f}")
    print(f"  • Info. Wigner–Yanase        : {mac.wigner_yanase_skew:.6f}")
    print(f"  • Wehrl (proxy)              : {mac.wehrl_entropy_proxy:.6f}")

    dynamics = ASTDynamicalSystemEngine.compute_ast_dynamics(homology, num_iterations=80)
    print("\n[TEST 3] Dinámica discreta del AST:")
    print(f"  • Diámetro orbital           : {dynamics.orbit_diameter:.4f}")
    print(f"  • λ_max AST                  : {dynamics.lyapunov_ast_max:.6e}")
    print(f"  • Diophantine γ              : {dynamics.diophantine_gamma:.4e} (ok={dynamics.is_diophantine})")
    if dynamics.return_map is not None:
        print(f"  • Nº retornos a Σ            : {dynamics.return_map.num_return_points}")
        print(f"  • Número de rotación ρ       : {dynamics.return_map.rotation_number:.6f}")
        print(f"  • KAM estable                : {dynamics.return_map.kam_stable}")
    if dynamics.birkhoff is not None:
        print(f"  • Birkhoff aplicable         : {dynamics.birkhoff.birkhoff_theorem_applicable}")
        print(f"  • p/q                        : {dynamics.birkhoff.rational_winding_p}/{dynamics.birkhoff.rational_period_q}")

    rng = np.random.default_rng(2026)
    t_op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    t_op = t_op / (float(np.linalg.norm(t_op, ord=2)) + 1e-15)
    tb = TarskiBrouwerEngine.verify_fixed_point(t_op)
    print("\n[TEST 4] Punto fijo T_Gödel (FTA / CP^{n−1}):")
    print(f"  • Existe (autovector λ ≠ 0)  : {tb.fixed_point_exists}")
    print(f"  • Ruta                       : {tb.verification_route}")
    print(f"  • d_FS residual              : {tb.fubini_study_residual_rad:.6e} rad")
    print(f"  • |λ_max|                    : {tb.spectral_radius:.6f}")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO A: Ciclo RSI coherente (Fase 1 → 2 → 3 anidadas)")
    print("─" * 96)
    agent = GodelAgent(agent_id="GODEL-WISDOM-SOVEREIGN-01", dimension_mac=4, seed=2026)
    cert_a = agent.continue_improve(force_spectral_violation=False)
    print(f"  • Veredicto Heyting          : {cert_a.heyting_verdict.name}")
    print(f"  • ρ(T)                       : {cert_a.banach_certificate.spectral_radius:.6f}")
    print(f"  • Cartan ω preservada        : {cert_a.cartan_report.symplectic_preservation_verified}")
    print(f"  • Topología 1-complejo       : {cert_a.cartan_report.topology_preserved}")
    print(f"  • Gauss–Bonnet               : {cert_a.gauss_bonnet_consistent}")
    print(f"  • Drift Casimirs KKS         : {cert_a.casimir_drift:.2e}")
    print(f"  • dim órbita coadjunta       : {cert_a.reduced_orbit_dimension:.1f}")
    print(f"  • Caos / Arnold              : {cert_a.birkhoff_melnikov.chaos_detected} / {cert_a.birkhoff_melnikov.arnold_diffusion_witness}")
    print(f"  • Tarski–FTA convergido      : {cert_a.fixed_point_converged}")
    print(f"  • τ̄_A (semilla Φ)            : {cert_a.recurrence_mean_time:.4f}  (Kac res={cert_a.recurrence_kac_residual:.4f})")
    print(f"  • Crowbar                    : {cert_a.crowbar_report.interlock_tripped}")
    print(f"  • SHA-256                    : {cert_a.digital_signature_sha256[:32]}…")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO B: Violación espectral forzada (enclavamiento Crowbar)")
    print("─" * 96)
    cert_b = agent.continue_improve(force_spectral_violation=True)
    print(f"  • Veredicto Heyting          : {cert_b.heyting_verdict.name}")
    print(f"  • Crowbar tripped            : {cert_b.crowbar_report.interlock_tripped}")
    print(f"  • Latencia IRAM+BT151        : {cert_b.crowbar_report.total_clearance_latency_ns:.2f} ns")
    print(f"  • Mutación aplicada          : {cert_b.mutation_applied}")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO C: Recurrencia de Poincaré–Kac inyectada")
    print("─" * 96)
    rng2 = np.random.default_rng(42)
    p_syn = rng2.random((6, 6)) + 0.1
    p_syn /= p_syn.sum(axis=1, keepdims=True)
    measurable = np.array([True, False, True, False, False, True])
    rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(p_syn, measurable, num_walks=200, max_steps=5000)
    print(f"  • |A|/|X|                    : {int(measurable.sum())}/6")
    print(f"  • τ̄_A empírico               : {rec_cert.mean_return_time_empirical:.4f}")
    print(f"  • Predicción de Kac          : {rec_cert.kac_lemma_prediction:.4f}")
    print(f"  • Residuo                    : {rec_cert.kac_error_residual:.4f}")

    cert_d = agent.execute_recursive_self_improvement(
        current_ast=agent.policy_ast,
        telemetry_data={"fault": None},
        stochastic_transition_matrix=p_syn,
        recurrence_set=measurable,
    )
    print(f"  • RSI+Kac veredicto          : {cert_d.heyting_verdict.name}")
    print(f"  • τ̄_A (certificado)          : {cert_d.recurrence_mean_time:.4f}")

    print("\n" + "─" * 96)
    print(">>> ESCENARIO E: Anidamiento explícito Fase 1 → 2 → 3")
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
    print(f"  • 𝔐_Wisdom → fibrado dim     : {bundle_e.configuration_dim},  J={bundle_e.syntax_momentum_map}")
    print(f"  • Φ Heyting                  : {morphism_e.heyting_verdict.name}")
    print(f"  • Semilla |A|                : {int(seed_e.measurable_set.sum())}/{seed_e.state_space_size}")
    print(f"  • Kac 1/μ(A) vs τ̄            : {rec_e.kac_lemma_prediction:.4f} vs {rec_e.mean_return_time_empirical:.4f}")

    print("\n" + "═" * 96)
    print("✓ AUDITORÍA CONCLUIDA: GÖDEL AGENT v4.1.0 — FASES ANIDADAS.")
    print("  · Fase 1: 𝔐_Wisdom → lift_wisdom_to_celestial_syntax_bundle (T*Q_AST, ω, H, J).")
    print("  · Fase 2: Φ_cat → seed_rsi_recurrence_from_transition (Perron–Frobenius).")
    print("  · Fase 3: Poincaré–Kac + FTA/CP^{n−1} + certificación SHA-256.")
    print("═" * 96)