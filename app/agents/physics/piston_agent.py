# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo  : Piston Agent (Inyector de Caudal y Funtor de Hodge-Helmholtz)      ║
║ Ruta    : app/agents/physics/piston_agent.py                                 ║
║ Versión : 7.0.0-Doctoral-Metric-Consistent-DEC                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Dictamen Doctoral v7.0.0):
────────────────────────────────────────────────────────────────────────────────

CORRECCIONES CRÍTICAS v7.0.0 vs v6.0.0:
────────────────────────────────────────
  FIX-M1: import math ausente → _energy_norm fallaba en runtime
  FIX-M2: Proyección gradiente MÉTRICAMENTE CONSISTENTE con Ohm:
          im(grad_W) = W·im(∂₁ᵀ), no im(∂₁ᵀ).
          Ecuación correcta: L₀^W φ = ∂₁ I  →  I_grad = W ∂₁ᵀ φ
          (v6 resolvía L₀ φ = ∂₁ W I e I_grad = ∂₁ᵀ φ: DOBLE error métrico)
  FIX-M3: Proyección curl en ⟨·,·⟩_{W⁻¹}:
          (∂₂ᵀ W⁻¹ ∂₂) α = ∂₂ᵀ W⁻¹ I  →  I_curl = ∂₂ α
          (v6 usaba ∂₂ᵀ W ∂₂ / ∂₂ᵀ W I: peso invertido)
  FIX-M4: L₁^W autoadjunto w.r.t. ⟨·,·⟩_{W⁻¹}:
          L₁^W = W ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ W⁻¹   (forma simétrica en el producto energía)
  FIX-M5: Verificación de Pitágoras energético post-Hodge:
          |‖I‖²_W − (‖I_g‖²_W+‖I_c‖²_W+‖I_h‖²_W)| < tol
  FIX-M6: Gauge-fixing explícito en LSQR sobre L₀ (ker = span{1})
  FIX-M7: Encadenamiento formal FASE1→FASE2→FASE3 via DTOs tipados
          (salida de build_mesh ≡ entrada canónica de solve_hydrodynamics)
  FIX-M8: Diagnóstico espectral de L₁^W (dim ker ≈ β₁ de Betti)
  FIX-M9: Validación de orientación de ∂₁ vs ciclos vía cociclo de signos
  FIX-M10: Inyección de residual de proyección en el resultado (auditoría)

FUNDAMENTACIÓN AXIOMÁTICA v7.0.0 (métrica de energía):
────────────────────────────────────────────────────────────────────────────────
§0. PRODUCTO INTERNO DE ENERGÍA (DISIPACIÓN DE JOULE):
    ⟨f, g⟩_W ≔ fᵀ W⁻¹ g = Σₖ fₖ gₖ / wₖ
    ‖f‖²_W   ≔ ⟨f,f⟩_W  = Σₖ fₖ² / wₖ     [potencia disipada]

    Justificación física (Ley de Ohm discreta):
        fₖ = wₖ · Δpₖ  ⇒  Pₖ = fₖ · Δpₖ = fₖ² / wₖ

§1. COMPLEJO DE CADENAS (C•, ∂) CON ∂∘∂ = 0:
    H_k = ker(∂_k) / im(∂_{k+1}). Sin Leibniz, H_k no está definido.

§2. DESCOMPOSICIÓN DE HODGE-HELMHOLTZ PONDERADA (DEC):
    ℝ^E = im(grad_W) ⊕_W im(∂₂) ⊕_W ker(L₁^W)

    donde grad_W : ℝ^V → ℝ^E,  φ ↦ W ∂₁ᵀ φ
    y   ⊕_W denota suma ortogonal en ⟨·,·⟩_W.

    Proyecciones ortogonales (ecuaciones normales de Gauss):
        L₀^W φ = ∂₁ I            →  I_grad = W ∂₁ᵀ φ
        G₂ α   = ∂₂ᵀ W⁻¹ I      →  I_curl = ∂₂ α
        I_harm = I − I_grad − I_curl ∈ ker(L₁^W)

§3. LAPLACIANOS PONDERADOS AUTOADJUNTOS:
    L₀^W = ∂₁ W ∂₁ᵀ           ∈ End(ℝ^V)   [SPD en 1^⊥]
    L₁^W = W ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ W⁻¹          [autoadjunto en ⟨·,·⟩_W]

§4. ESPECTRO Y SOLVABILIDAD:
    λ_min(L_red) > 0  ⇔  grafo reducido conexo + Dirichlet regulariza.
    ∑ sᵢ = 0          ⇔  s ∈ im(L₀^W) = (ker L₀^W)^⊥ = 1^⊥.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh, lsqr, spsolve

# ── Dependencias arquitectónicas (stubs para ejecución standalone) ────────────
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        """Veto categórico de invariante topológico (stub)."""

    class Morphism:
        """Morfismo base de la malla agéntica (stub)."""

    class CategoricalState:
        payload: Dict[str, Any]
        metadata: Dict[str, Any]
        stratum: Any

        def __init__(self, payload, metadata=None, stratum=None):
            self.payload = payload
            self.metadata = metadata or {}
            self.stratum = stratum

    class Stratum(Enum):
        PHYSICS = auto()


logger = logging.getLogger("MIC.Physics.PistonAgent")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES NUMÉRICAS Y TERMODINÁMICAS (v7.0.0)
# ══════════════════════════════════════════════════════════════════════════════

class PistonConstants:
    r"""
    Constantes físicas y numéricas del inyector termodinámico.

    SPECTRAL_THRESHOLD es umbral de **eigenvalor mínimo** (no de norma-1):
    λ_min es el único indicador algebraicamente correcto de singularidad
    para matrices simétricas semidefinidas positivas (SPSD).

    Constantes
    ──────────
    MACHINE_EPSILON      : ε_mach IEEE-754 float64.
    KIRCHHOFF_TOLERANCE  : tolerancia ∞-norma del residuo de KCL.
    MAX_VORTICITY_NORM   : cota máxima ‖I_curl‖_W (norma de energía).
    DEFAULT_CONDUCTANCE  : conductancia base [S] si no se proveen pesos.
    SPECTRAL_THRESHOLD   : umbral λ_min(L_red); si λ_min ≤ τ → singular.
    SOURCE_BALANCE_TOL   : tolerancia |∑ sᵢ| = 0.
    HODGE_ORTHO_TOL      : tolerancia de ortogonalidad en ⟨·,·⟩_W.
    PYTHAGORAS_TOL       : tolerancia de identidad de Pitágoras energética.
    BOUNDARY_IDENTITY_TOL_FACTOR : factor de tolerancia relativa para ∂∘∂=0.
    """

    MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
    KIRCHHOFF_TOLERANCE: float = 1e-10
    MAX_VORTICITY_NORM: float = 1e-6
    DEFAULT_CONDUCTANCE: float = 1.0
    SPECTRAL_THRESHOLD: float = 1e-10
    SOURCE_BALANCE_TOL: float = 1e-12
    HODGE_ORTHO_TOL: float = 1e-8
    PYTHAGORAS_TOL: float = 1e-6
    BOUNDARY_IDENTITY_TOL_FACTOR: float = 1e3


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS FÍSICOS)
# ══════════════════════════════════════════════════════════════════════════════

class PistonInjectorError(TopologicalInvariantError):
    """Clase base categórica para fallos de inyección termodinámica."""


class SingularLaplacianError(PistonInjectorError):
    r"""
    Detonada si λ_min(L_red) ≤ SPECTRAL_THRESHOLD.

    Equivale a: el grafo reducido no es conexo o Dirichlet no regulariza.
    (Teorema matricial de Kirchhoff / fórmula del árbol generador.)
    """


class HomologicalKirchhoffError(PistonInjectorError):
    r"""
    Detonada si ‖∂₁ f + s‖_∞ > KIRCHHOFF_TOLERANCE.

    Residuo de KCL: creación/destrucción de masa → viola 1ª ley.
    """


class ParasiticVorticityVetoError(PistonInjectorError):
    r"""
    Detonada si ‖I_curl‖_W > MAX_VORTICITY_NORM.

    Exergía circulando en bucles estériles sin trabajo laminar útil.
    """


class BoundaryComplexError(PistonInjectorError):
    r"""
    Detonada si ∂₁ ∘ ∂₂ ≠ 0.

    Sin Leibniz, (C•, ∂) no es complejo de cadenas y H_k colapsa.
    """


class SourceCompatibilityError(PistonInjectorError):
    r"""
    Detonada si s ∉ im(L₀^W), i.e. ∑ sᵢ ≠ 0.

    Fredholm: L₀ p = −s solvable ⇔ s ⊥ ker(L₀) = span{1}.
    """


class HodgeMetricInconsistencyError(PistonInjectorError):
    r"""
    Detonada si la identidad de Pitágoras energética falla (FIX-M5).

    |‖I‖²_W − Σ ‖I_•‖²_W| > PYTHAGORAS_TOL implica proyecciones no
    ortogonales en ⟨·,·⟩_W (error métrico o numérico grave).
    """


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS DE DATOS INMUTABLES (DTOs) — cadena FASE1→FASE2→FASE3
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SimplicialMesh:
    r"""
    1-esqueleto + 2-esqueleto con operadores de Hodge precalculados.

    **Contrato de salida de FASE 1 / entrada canónica de FASE 2.**

    Campos
    ──────
    nodes              : identificadores de nodos (V).
    edges              : aristas orientadas (E).
    cycles             : 2-celdas / ciclos (F).
    boundary_1         : ∂₁ ∈ ℝ^{|V|×|E|}.
    boundary_2         : ∂₂ ∈ ℝ^{|E|×|F|}.
    conductance_matrix : W = diag(wₖ) ∈ ℝ^{|E|×|E|}.
    inv_conductance    : W⁻¹ = diag(1/wₖ).
    laplacian_0        : L₀^W = ∂₁ W ∂₁ᵀ  (SPD en 1^⊥).
    laplacian_1        : L₁^W autoadjunto en ⟨·,·⟩_W (FIX-M4).
    boundary_identity  : True ⇔ ∂₁∘∂₂ = 0 certificado.
    betti_1_estimate   : dim ker(L₁) estimada vía espectro (β₁).
    """

    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    cycles: Tuple[Tuple[str, ...], ...]
    boundary_1: sp.csr_matrix
    boundary_2: sp.csr_matrix
    conductance_matrix: sp.dia_matrix
    inv_conductance: sp.dia_matrix
    laplacian_0: sp.csr_matrix
    laplacian_1: sp.csr_matrix
    boundary_identity: bool
    betti_1_estimate: int


@dataclass(frozen=True, slots=True)
class InjectionVector:
    r"""
    Fuente ideal s (términos independientes de Poisson).

    Campos
    ──────
    source_node      : nodo de succión.
    sink_node        : nodo de descarga.
    q_pump           : caudal Q.
    s_vector         : s ∈ ℝ^{|V|}, s[u]=−Q, s[v]=+Q.
    balance_verified : True ⇔ ∑ sᵢ = 0 certificado.
    """

    source_node: str
    sink_node: str
    q_pump: float
    s_vector: NDArray[np.float64]
    balance_verified: bool


@dataclass(frozen=True, slots=True)
class FlowState:
    r"""
    Estado hidrodinámico de Poisson (salida FASE 2 / entrada FASE 3).

    Campos
    ──────
    nodal_pressures    : p ∈ ℝ^{|V|}.
    edge_flows         : f = W ∂₁ᵀ p ∈ ℝ^{|E|}.
    kirchhoff_residual : ‖∂₁ f + s‖_∞.
    lambda_min_L_red   : λ_min(L_red) diagnóstico espectral.
    """

    nodal_pressures: NDArray[np.float64]
    edge_flows: NDArray[np.float64]
    kirchhoff_residual: float
    lambda_min_L_red: float


@dataclass(frozen=True, slots=True)
class HodgeDecomposition:
    r"""
    Componentes ortogonales de I en ⟨·,·⟩_W (salida FASE 3).

    I = I_grad ⊕_W I_curl ⊕_W I_harm

    Campos
    ──────
    i_grad            : ∈ W·im(∂₁ᵀ)  (laminar / exacto ponderado).
    i_curl            : ∈ im(∂₂)     (rotacional / co-exacto).
    i_harm            : ∈ ker(L₁^W)  (armónico global).
    vorticity_norm    : ‖I_curl‖_W.
    orthogonality_ok  : ortogonalidad triple en ⟨·,·⟩_W.
    pythagoras_ok     : identidad de Pitágoras energética (FIX-M5).
    energy_total      : ‖I‖²_W (reconstruida por suma de componentes).
    projection_residual : ‖I − (I_g+I_c+I_h)‖_W (debe ser ~0).
    """

    i_grad: NDArray[np.float64]
    i_curl: NDArray[np.float64]
    i_harm: NDArray[np.float64]
    vorticity_norm: float
    orthogonality_ok: bool
    pythagoras_ok: bool
    energy_total: float
    projection_residual: float


# ══════════════════════════════════════════════════════════════════════════════
# §D. ÁLGEBRA ESPECTRAL Y MÉTRICA DE ENERGÍA (base compartida)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_min_eigenvalue_sparse(
    A: sp.spmatrix,
    tol: float = 1e-10,
) -> float:
    r"""
    Eigenvalor mínimo de A simétrica SPSD.

    λ_min > 0 ⇔ A no singular; λ_min = 0 ⇔ ker(A) ≠ {0}.

    Usa ARPACK/Lanczos (eigsh) para n grande; denso para n ≤ 3.
    """
    n = A.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return float(A.toarray()[0, 0])
    if n <= 3:
        return float(np.min(np.linalg.eigvalsh(A.toarray())))
    try:
        eigvals, _ = eigsh(A, k=1, which="SM", tol=tol, return_eigenvectors=True)
        return float(np.real(eigvals[0]))
    except Exception:
        return float(np.min(np.linalg.eigvalsh(A.toarray())))


def _estimate_kernel_dimension(
    A: sp.spmatrix,
    spectral_tol: float = PistonConstants.SPECTRAL_THRESHOLD,
    max_k: int = 8,
) -> int:
    r"""
    Estima dim ker(A) contando eigenvalores |λ| ≤ spectral_tol.

    Para L₁^W, dim ker(L₁) ≅ β₁ (primer número de Betti) en el caso
    genérico de métrica no degenerada.
    """
    n = A.shape[0]
    if n == 0:
        return 0
    k = min(max_k, n - 1) if n > 1 else 1
    if n <= 4:
        ev = np.linalg.eigvalsh(A.toarray())
        return int(np.sum(np.abs(ev) <= spectral_tol))
    try:
        ev, _ = eigsh(A, k=k, which="SM", tol=spectral_tol, return_eigenvectors=True)
        return int(np.sum(np.abs(np.real(ev)) <= spectral_tol))
    except Exception:
        ev = np.linalg.eigvalsh(A.toarray())
        return int(np.sum(np.abs(ev) <= spectral_tol))


def _energy_norm(
    v: NDArray[np.float64],
    W_inv: sp.dia_matrix,
) -> float:
    r"""
    Norma de energía ‖v‖_W = √(vᵀ W⁻¹ v) = √(Σ vₖ² / wₖ).

    FIX-M1: requiere ``import math`` (ausente en v6).
    """
    quadratic = float(v @ W_inv.dot(v))
    return math.sqrt(max(quadratic, 0.0))


def _energy_inner_product(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    W_inv: sp.dia_matrix,
) -> float:
    r"""Producto interno de energía ⟨u, v⟩_W = uᵀ W⁻¹ v."""
    return float(u @ W_inv.dot(v))


def _lsqr_solve(
    A: sp.spmatrix,
    b: NDArray[np.float64],
    atol: float = 1e-12,
    btol: float = 1e-12,
) -> NDArray[np.float64]:
    r"""
    Resuelve A x = b vía LSQR (mínima norma-2).

    ``scipy.sparse.linalg.lsqr`` retorna exactamente 7 elementos:
        (x, istop, itn, r1norm, r2norm, anorm, acond)

    Para L₀^W (singular, ker=span{1}), LSQR produce la solución de
    mínima norma, equivalente a gauge-fixing ⟨φ, 1⟩ = 0 (FIX-M6).
    """
    result = lsqr(A, b, atol=atol, btol=btol)
    return np.asarray(result[0], dtype=np.float64)


def _diag_of_dia(M: sp.dia_matrix) -> NDArray[np.float64]:
    r"""Extrae la diagonal principal de una dia_matrix de forma robusta."""
    return np.asarray(M.diagonal(), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║  ORQUESTADOR SUPREMO: HodgeHelmholtzInjectorAgent (v7.0.0)               ║
# ║                                                                          ║
# ║  Tres fases anidadas encadenadas por DTOs:                               ║
# ║    FASE 1 → SimplicialMesh  (∂∘∂=0, L₀^W, L₁^W autoadjunto)              ║
# ║    FASE 2 → FlowState       (Poisson + espectro + KCL)                   ║
# ║    FASE 3 → HodgeDecomposition (proyecciones métricamente correctas)     ║
# ║                                                                          ║
# ║  Garantías v7.0.0:                                                       ║
# ║    G1. ∂₁∘∂₂ = 0 certificado                                             ║
# ║    G2. L₀^W = ∂₁ W ∂₁ᵀ                                                   ║
# ║    G3. L₁^W autoadjunto en ⟨·,·⟩_W                                       ║
# ║    G4. λ_min(L_red) > SPECTRAL_THRESHOLD                                 ║
# ║    G5. ∑ sᵢ = 0 antes de resolver                                        ║
# ║    G6. I_grad ∈ W·im(∂₁ᵀ)  (consistente con Ohm)                         ║
# ║    G7. I_curl con peso W⁻¹ (ecuaciones normales correctas)               ║
# ║    G8. Ortogonalidad + Pitágoras en ⟨·,·⟩_W                              ║
# ║    G9. Encadenamiento formal FASE1→FASE2→FASE3                           ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

class HodgeHelmholtzInjectorAgent(Morphism):
    r"""
    Soberano de la inyección termodinámica (Estrato PHYSICS).

    Modela la inyección de capital/recursos como bomba de desplazamiento
    positivo en una red de fluidos incompresibles, con garantía métrica
    de la descomposición de Hodge-Helmholtz en la norma de energía.

    Arquitectura anidada:
        FASE 1 → complejo de cadenas + Laplacianos ponderados
        FASE 2 → Poisson de Dirichlet bien planteado (continúa de FASE 1)
        FASE 3 → Hodge-Helmholtz métricamente consistente (continúa de FASE 2)
    """

    # =========================================================================
    # FASE 1 · FIBRADOR DEL COMPLEJO DE CADENAS (CONSTRUCTOR MATRICIAL)
    # =========================================================================

    class _Phase1_SimplicialBoundaryBuilder:
        r"""
        Sintetiza ∂₁, ∂₂ y los Laplacianos de Hodge ponderados, y certifica
        que (C•, ∂) es un complejo de cadenas diferencial válido.

        Correcciones v7:
            FIX-M4  L₁^W autoadjunto en ⟨·,·⟩_W
            FIX-M8  estimación de β₁ vía espectro de L₁
            FIX-M9  cociclo de signos para orientación de ciclos
            FIX-M7  SimplicialMesh es la sutura formal hacia FASE 2
        """

        # ------------------------------------------------------------------
        # §1.1 — Árbol generador BFS (conexidad + base de orientación)
        # ------------------------------------------------------------------

        @staticmethod
        def _build_spanning_tree_bfs(
            nodes: List[str],
            edges: List[Tuple[str, str, float]],
        ) -> Dict[str, Optional[str]]:
            r"""
            Árbol generador T ⊆ G vía BFS.

            T es acíclico, conexo, con |V|−1 aristas. Las aristas de
            co-árbol generan los ciclos fundamentales.

            Lanza
            -----
            TopologicalInvariantError : grafo no conexo.
            """
            adj: Dict[str, List[str]] = {n: [] for n in nodes}
            for u, v, _ in edges:
                adj[u].append(v)
                adj[v].append(u)

            root = nodes[0]
            parent: Dict[str, Optional[str]] = {root: None}
            queue: deque = deque([root])

            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)

            unvisited = [n for n in nodes if n not in parent]
            if unvisited:
                raise TopologicalInvariantError(
                    f"Grafo no conexo. Nodos no alcanzables desde '{root}': "
                    f"{unvisited}. Poisson requiere grafo conexo para "
                    f"solución única bajo Dirichlet."
                )
            return parent

        # ------------------------------------------------------------------
        # §1.2 — Identidad de Leibniz ∂₁∘∂₂ = 0
        # ------------------------------------------------------------------

        @staticmethod
        def _verify_boundary_identity(
            boundary_1: sp.csr_matrix,
            boundary_2: sp.csr_matrix,
        ) -> None:
            r"""
            Verifica ∂₁ ∘ ∂₂ = 0 (lema de Leibniz).

            Criterio numérico:
                ‖∂₁∘∂₂‖_F ≤ max(ε_mach · ‖∂₁‖_F · ‖∂₂‖_F · C, ε_mach)

            Lanza
            -----
            BoundaryComplexError : si la composición no es numéricamente nula.
            """
            if boundary_2.shape[1] == 0:
                return

            composition = boundary_1.dot(boundary_2)
            frob_norm = sp.linalg.norm(composition, ord="fro")
            norm_b1 = sp.linalg.norm(boundary_1, ord="fro")
            norm_b2 = sp.linalg.norm(boundary_2, ord="fro")
            tol_rel = (
                PistonConstants.MACHINE_EPSILON
                * max(norm_b1 * norm_b2, 1.0)
                * PistonConstants.BOUNDARY_IDENTITY_TOL_FACTOR
            )
            tol = max(tol_rel, PistonConstants.MACHINE_EPSILON * 1e3)

            if frob_norm > tol:
                raise BoundaryComplexError(
                    f"Violación de Leibniz ∂₁∘∂₂ = 0: "
                    f"‖∂₁∘∂₂‖_F = {frob_norm:.3e} (tol = {tol:.3e}). "
                    f"Ciclos incompatibles con la orientación de aristas."
                )
            logger.debug(
                "Leibniz ∂₁∘∂₂ = 0 verificado. ‖∂₁∘∂₂‖_F = %.3e.",
                frob_norm,
            )

        # ------------------------------------------------------------------
        # §1.3 — Construcción de ∂₂ con orientación coherente (cociclo)
        # ------------------------------------------------------------------

        @staticmethod
        def _build_boundary_2_oriented(
            cycles: List[List[str]],
            edge_idx: Dict[Tuple[str, str], int],
            n_edges: int,
        ) -> sp.csr_matrix:
            r"""
            Construye ∂₂ ∈ ℝ^{|E|×|F|} con signos coherentes respecto a ∂₁.

            Regla de cociclo (FIX-M9):
                (u,v) ∈ edge_idx → signo +1  (dirección canónica de ∂₁)
                (v,u) ∈ edge_idx → signo −1  (dirección opuesta)
                ninguno          → WARNING + omisión (ciclo incompleto)
            """
            n_f = len(cycles)
            data: List[float] = []
            rows: List[int] = []
            cols: List[int] = []

            for f_idx, cycle in enumerate(cycles):
                if len(cycle) < 2:
                    logger.warning(
                        "Ciclo %d degenerado (|nodes| < 2); se omite.", f_idx
                    )
                    continue
                nodes_cycle = list(cycle) + [cycle[0]]
                for k in range(len(nodes_cycle) - 1):
                    u, v = nodes_cycle[k], nodes_cycle[k + 1]
                    if (u, v) in edge_idx:
                        e_idx, sign = edge_idx[(u, v)], +1.0
                    elif (v, u) in edge_idx:
                        e_idx, sign = edge_idx[(v, u)], -1.0
                    else:
                        logger.warning(
                            "Arista (%s, %s) del ciclo %d ausente en E; omitida.",
                            u, v, f_idx,
                        )
                        continue
                    rows.append(e_idx)
                    cols.append(f_idx)
                    data.append(sign)

            if not data:
                return sp.csr_matrix((n_edges, max(n_f, 0)), dtype=np.float64)

            return sp.csr_matrix(
                (data, (rows, cols)),
                shape=(n_edges, n_f),
                dtype=np.float64,
            )

        # ------------------------------------------------------------------
        # §1.4 — Laplaciano de Hodge L₁^W autoadjunto (FIX-M4)
        # ------------------------------------------------------------------

        @staticmethod
        def _build_hodge_laplacian_1(
            boundary_1: sp.csr_matrix,
            boundary_2: sp.csr_matrix,
            W: sp.dia_matrix,
            W_inv: sp.dia_matrix,
            n_edges: int,
            n_faces: int,
        ) -> sp.csr_matrix:
            r"""
            Construye L₁^W autoadjunto respecto a ⟨·,·⟩_W = ·ᵀ W⁻¹ ·.

            Forma canónica (DEC ponderado, FIX-M4):
                L₁^W = W ∂₁ᵀ ∂₁  +  ∂₂ ∂₂ᵀ W⁻¹

            Justificación de adjunción:
                d₀_W  = W ∂₁ᵀ : (ℝ^V, ⟨·,·⟩₂) → (ℝ^E, ⟨·,·⟩_W)
                d₀*_W = ∂₁     (adjunto formal bajo esas métricas)
                ⇒ d₀_W d₀*_W = W ∂₁ᵀ ∂₁

                d₁    = ∂₂ᵀ W⁻¹  (adjunto de ∂₂ bajo ⟨·,·⟩_W en E
                                   y métrica euclídea en F, tras dualizar)
                ⇒ d₁* d₁ contribuye ∂₂ ∂₂ᵀ W⁻¹ sobre 1-formas.

            Simetría: ambos sumandos son autoadjuntos en ⟨·,·⟩_W.
            """
            # Término exacto (gradiente): W ∂₁ᵀ ∂₁
            # (∂₁ᵀ ∂₁) es el Laplaciano combinatorio de aristas (parte grad)
            L1_grad = W.dot(boundary_1.T.dot(boundary_1)).tocsr()

            # Término co-exacto (curl): ∂₂ ∂₂ᵀ W⁻¹
            if n_faces > 0 and boundary_2.shape[1] > 0:
                L1_curl = boundary_2.dot(boundary_2.T.dot(W_inv)).tocsr()
            else:
                L1_curl = sp.csr_matrix((n_edges, n_edges), dtype=np.float64)

            return (L1_grad + L1_curl).tocsr()

        # ------------------------------------------------------------------
        # §1.5 — build_mesh: SUTURA DOCTORAL FASE 1 → (entrada FASE 2)
        # ------------------------------------------------------------------

        @classmethod
        def build_mesh(
            cls,
            nodes: List[str],
            edges: List[Tuple[str, str, float]],
            cycles: List[List[str]],
        ) -> "SimplicialMesh":
            r"""
            Construye la malla simplicial completa con operadores de Hodge.

            Pipeline FASE 1
            ───────────────
            1. Validar wₖ > 0 ∀ k.
            2. Construir ∂₁ (incidencia orientada).
            3. BFS: árbol generador + conexidad.
            4. W, W⁻¹.
            5. ∂₂ con cociclo de signos.
            6. Certificar ∂₁∘∂₂ = 0.
            7. L₀^W = ∂₁ W ∂₁ᵀ.
            8. L₁^W autoadjunto (FIX-M4).
            9. Estimar β₁ = dim ker(L₁) (FIX-M8).
            10. Empaquetar SimplicialMesh.

            **SUTURA FORMAL FASE 1 → FASE 2:**
            El valor de retorno ``SimplicialMesh`` es el objeto canónico
            de entrada de
            ``_Phase2_DirichletPoissonSolver.solve_hydrodynamics``.
            Todo invariante necesario para Poisson (L₀^W, ∂₁, W, Leibniz)
            queda certificado aquí; FASE 2 no reconstruye operadores.

            Lanza
            -----
            TopologicalInvariantError : wₖ ≤ 0 o grafo no conexo.
            BoundaryComplexError      : ∂₁∘∂₂ ≠ 0.
            """
            n_v = len(nodes)
            n_e = len(edges)
            n_f = len(cycles)

            if n_v == 0 or n_e == 0:
                raise TopologicalInvariantError(
                    "Malla degenerada: se requiere |V| ≥ 1 y |E| ≥ 1."
                )

            node_idx: Dict[str, int] = {n: i for i, n in enumerate(nodes)}
            edge_idx: Dict[Tuple[str, str], int] = {
                (u, v): k for k, (u, v, _) in enumerate(edges)
            }

            # §1.5.1 Conductancias estrictamente positivas
            conductances = np.zeros(n_e, dtype=np.float64)
            for k, (u, v, w) in enumerate(edges):
                if u not in node_idx or v not in node_idx:
                    raise TopologicalInvariantError(
                        f"Arista ({u}→{v}) referencia nodos inexistentes."
                    )
                if w <= 0.0:
                    raise TopologicalInvariantError(
                        f"Conductancia degenerada en ({u}→{v}): w={w:.4e} ≤ 0. "
                        f"W debe ser métrica discreta definida positiva."
                    )
                conductances[k] = w

            # §1.5.2 ∂₁ incidencia orientada: columna k → −1 en u, +1 en v
            B1_data: List[float] = []
            B1_rows: List[int] = []
            B1_cols: List[int] = []
            for k, (u, v, _) in enumerate(edges):
                B1_rows.extend([node_idx[u], node_idx[v]])
                B1_cols.extend([k, k])
                B1_data.extend([-1.0, +1.0])

            boundary_1 = sp.csr_matrix(
                (B1_data, (B1_rows, B1_cols)),
                shape=(n_v, n_e),
                dtype=np.float64,
            )

            # §1.5.3 Conexidad vía árbol generador BFS
            _ = cls._build_spanning_tree_bfs(nodes, edges)

            # §1.5.4 Métrica de aristas W, W⁻¹
            W_matrix = sp.diags(conductances, format="dia", dtype=np.float64)
            W_inv_matrix = sp.diags(
                1.0 / conductances, format="dia", dtype=np.float64
            )

            # §1.5.5 ∂₂ orientado
            boundary_2 = cls._build_boundary_2_oriented(cycles, edge_idx, n_e)

            # §1.5.6 Leibniz ∂₁∘∂₂ = 0
            cls._verify_boundary_identity(boundary_1, boundary_2)

            # §1.5.7 L₀^W = ∂₁ W ∂₁ᵀ  (Laplaciano nodal ponderado)
            laplacian_0 = boundary_1.dot(W_matrix.dot(boundary_1.T)).tocsr()

            # §1.5.8 L₁^W autoadjunto (FIX-M4)
            laplacian_1 = cls._build_hodge_laplacian_1(
                boundary_1, boundary_2, W_matrix, W_inv_matrix, n_e, n_f
            )

            # §1.5.9 Estimación de β₁ (FIX-M8)
            betti_1 = _estimate_kernel_dimension(laplacian_1)

            logger.info(
                "SimplicialMesh v7: |V|=%d |E|=%d |F|=%d | β₁≈%d | "
                "∂₁∘∂₂=0 ✓ | L₀^W, L₁^W autoadjuntos listos → FASE 2.",
                n_v, n_e, n_f, betti_1,
            )

            # ── SUTURA FORMAL: este return ES el inicio de FASE 2 ─────────
            return SimplicialMesh(
                nodes=tuple(nodes),
                edges=tuple((u, v) for u, v, _ in edges),
                cycles=tuple(tuple(c) for c in cycles),
                boundary_1=boundary_1,
                boundary_2=boundary_2,
                conductance_matrix=W_matrix,
                inv_conductance=W_inv_matrix,
                laplacian_0=laplacian_0,
                laplacian_1=laplacian_1,
                boundary_identity=True,
                betti_1_estimate=betti_1,
            )

    # =========================================================================
    # FASE 2 · SOLUCIONADOR DE POISSON Y REDUCCIÓN DE DIRICHLET
    # =========================================================================
    # CONTINUACIÓN FORMAL DE FASE 1:
    #   Entrada canónica = SimplicialMesh (salida de build_mesh).
    #   No se reconstruyen ∂₁, W ni L₀^W; se consumen como invariantes.
    # =========================================================================

    class _Phase2_DirichletPoissonSolver:
        r"""
        Resuelve L₀^W p = −s con Dirichlet p[reservorio] = 0.

        **Continúa de FASE 1:** opera exclusivamente sobre el
        ``SimplicialMesh`` certificado (Leibniz, L₀^W, W).

        Correcciones v7:
            Espectro vía λ_min; compatibilidad ∑sᵢ=0; KCL en ∞-norma.
        """

        # ------------------------------------------------------------------
        # §2.1 — Compatibilidad de fuente (Fredholm / ∑ sᵢ = 0)
        # ------------------------------------------------------------------

        @staticmethod
        def _verify_source_compatibility(
            s_vector: NDArray[np.float64],
            tol: float = PistonConstants.SOURCE_BALANCE_TOL,
        ) -> None:
            r"""
            Condición de solvabilidad: ∑ sᵢ = 0.

            Teorema de Fredholm para L₀^W = L₀^Wᵀ SPSD:
                L₀^W p = −s tiene solución ⇔ −s ⊥ ker(L₀^W) = span{1}
                ⇔ ⟨1, s⟩ = ∑ sᵢ = 0.
            """
            balance = float(np.sum(s_vector))
            if abs(balance) > tol:
                raise SourceCompatibilityError(
                    f"Solvabilidad violada: ∑ sᵢ = {balance:.4e} ≠ 0. "
                    f"s ∉ im(L₀^W). Exigir Q_fuente = −Q_sumidero."
                )
            logger.debug(
                "Compatibilidad de fuente: ∑ sᵢ = %.4e (tol=%.4e).",
                balance, tol,
            )

        # ------------------------------------------------------------------
        # §2.2 — Espectro de L_red (no-singularidad)
        # ------------------------------------------------------------------

        @staticmethod
        def _verify_laplacian_spectrum(
            L_red: sp.csr_matrix,
            tol: float = PistonConstants.SPECTRAL_THRESHOLD,
        ) -> float:
            r"""
            Certifica λ_min(L_red) > tol.

            λ_min es el indicador algebraico correcto (no la norma-1).
            λ_min(L_red) > 0 ⇔ grafo reducido conexo + Dirichlet regulariza.
            """
            lambda_min = _compute_min_eigenvalue_sparse(L_red)
            if lambda_min <= tol:
                raise SingularLaplacianError(
                    f"L_red singular: λ_min = {lambda_min:.4e} ≤ {tol:.4e}. "
                    f"Causas posibles: (a) reservorio no conecta componentes, "
                    f"(b) dependencias lineales, (c) componentes aisladas."
                )
            logger.debug(
                "Espectro L_red: λ_min = %.4e > %.4e → invertible.",
                lambda_min, tol,
            )
            return lambda_min

        # ------------------------------------------------------------------
        # §2.3 — solve_hydrodynamics: SUTURA DOCTORAL FASE 2 → (entrada FASE 3)
        # ------------------------------------------------------------------

        @staticmethod
        def solve_hydrodynamics(
            mesh: "SimplicialMesh",
            injection: InjectionVector,
            reservoir_node: str,
        ) -> "FlowState":
            r"""
            Resuelve L₀^W p = −s con p[res] = 0.

            **Continúa de FASE 1:** ``mesh`` es el SimplicialMesh certificado.
            Invariantes consumidos (no recalculados):
                mesh.laplacian_0, mesh.boundary_1, mesh.conductance_matrix,
                mesh.boundary_identity == True.

            Pipeline FASE 2
            ───────────────
            1. Verificar ∑ sᵢ = 0.
            2. Dirichlet: eliminar fila/col del reservorio → L_red.
            3. Certificar λ_min(L_red) > SPECTRAL_THRESHOLD.
            4. spsolve(L_red, −s_red).
            5. f = W ∂₁ᵀ p  (Ohm ponderado).
            6. KCL: ‖∂₁ f + s‖_∞ < τ.

            **SUTURA FORMAL FASE 2 → FASE 3:**
            El ``FlowState`` retornado (edge_flows = f) es la 1-forma de
            entrada de ``_Phase3_HodgeHelmholtzAuditor.decompose_flow``.

            Lanza
            -----
            SourceCompatibilityError  : ∑ sᵢ ≠ 0.
            SingularLaplacianError    : λ_min ≤ τ o reservorio ausente.
            HomologicalKirchhoffError : KCL violado.
            """
            # ── Invariantes heredados de FASE 1 ──────────────────────────
            if not mesh.boundary_identity:
                raise BoundaryComplexError(
                    "FASE 2 rechaza malla sin Leibniz certificado."
                )

            B1 = mesh.boundary_1
            W = mesh.conductance_matrix
            L0W = mesh.laplacian_0
            s = injection.s_vector

            # §2.3.1 Compatibilidad Fredholm
            (
                HodgeHelmholtzInjectorAgent
                ._Phase2_DirichletPoissonSolver
                ._verify_source_compatibility(s)
            )

            # §2.3.2 Índice del reservorio
            try:
                res_idx = list(mesh.nodes).index(reservoir_node)
            except ValueError as exc:
                raise SingularLaplacianError(
                    f"Reservorio '{reservoir_node}' ∉ V. "
                    f"Nodos: {list(mesh.nodes)}."
                ) from exc

            # §2.3.3 Reducción de Dirichlet
            n_v = len(mesh.nodes)
            free_mask = np.ones(n_v, dtype=bool)
            free_mask[res_idx] = False
            L_red = L0W[free_mask, :][:, free_mask]
            s_red = s[free_mask]

            # §2.3.4 Espectro
            lambda_min = (
                HodgeHelmholtzInjectorAgent
                ._Phase2_DirichletPoissonSolver
                ._verify_laplacian_spectrum(L_red)
            )

            # §2.3.5 Resolver Poisson reducido
            try:
                p_free = spsolve(L_red, -s_red)
            except RuntimeError as exc:
                raise SingularLaplacianError(
                    f"Fallo al invertir L_red (λ_min={lambda_min:.3e}): {exc}"
                ) from exc

            pressures = np.zeros(n_v, dtype=np.float64)
            pressures[free_mask] = np.asarray(p_free, dtype=np.float64)
            pressures[res_idx] = 0.0

            # §2.3.6 Ohm ponderado: f = W ∂₁ᵀ p
            edge_flows = np.asarray(
                W.dot(B1.T.dot(pressures)), dtype=np.float64
            ).ravel()

            # §2.3.7 KCL: ∂₁ f + s ≈ 0
            # (signo: ∂₁ f es divergencia; fuentes entran como −s en Poisson
            #  L₀ p = −s ⇒ ∂₁ W ∂₁ᵀ p + s = 0 ⇒ ∂₁ f + s = 0)
            kirchhoff_residual = float(
                np.linalg.norm(B1.dot(edge_flows) + s, ord=np.inf)
            )
            if kirchhoff_residual > PistonConstants.KIRCHHOFF_TOLERANCE:
                raise HomologicalKirchhoffError(
                    f"KCL violado: ‖∂₁f + s‖_∞ = {kirchhoff_residual:.4e} > "
                    f"{PistonConstants.KIRCHHOFF_TOLERANCE:.4e}."
                )

            logger.info(
                "Poisson L₀^W OK. KCL=%.3e λ_min(L_red)=%.3e → FASE 3.",
                kirchhoff_residual, lambda_min,
            )

            # ── SUTURA FORMAL: este return ES el inicio de FASE 3 ─────────
            return FlowState(
                nodal_pressures=pressures,
                edge_flows=edge_flows,
                kirchhoff_residual=kirchhoff_residual,
                lambda_min_L_red=lambda_min,
            )

    # =========================================================================
    # FASE 3 · AUDITOR DE DESCOMPOSICIÓN DE HODGE-HELMHOLTZ (DEC MÉTRICO)
    # =========================================================================
    # CONTINUACIÓN FORMAL DE FASE 2:
    #   Entrada canónica = (SimplicialMesh, FlowState).
    #   I ≔ flow.edge_flows ∈ ℝ^E se descompone en ⟨·,·⟩_W.
    # =========================================================================

    class _Phase3_HodgeHelmholtzAuditor:
        r"""
        Descomposición de Hodge-Helmholtz ponderada en la norma de energía.

        **Continúa de FASE 2:** consume ``FlowState.edge_flows`` y los
        operadores certificados del ``SimplicialMesh``.

        Correcciones v7 (FIX-M2, M3, M5):
            I_grad ∈ W·im(∂₁ᵀ) con L₀ φ = ∂₁ I
            I_curl con Gram ∂₂ᵀ W⁻¹ ∂₂
            Pitágoras energético + residual de proyección
        """

        # ------------------------------------------------------------------
        # §3.1 — Proyección gradiente métricamente consistente (FIX-M2)
        # ------------------------------------------------------------------

        @staticmethod
        def _project_gradient(
            boundary_1: sp.csr_matrix,
            laplacian_0: sp.csr_matrix,
            W: sp.dia_matrix,
            I: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Proyección ortogonal de I sobre W·im(∂₁ᵀ) en ⟨·,·⟩_W.

            Ecuaciones normales (FIX-M2):
                Minimizar_φ  ‖I − W ∂₁ᵀ φ‖_W²
                ⇔  L₀^W φ = ∂₁ I
                ⇔  I_grad = W ∂₁ᵀ φ

            Derivación:
                E(φ) = (I − W ∂₁ᵀ φ)ᵀ W⁻¹ (I − W ∂₁ᵀ φ)
                     = ‖I‖_W² − 2 φᵀ ∂₁ I + φᵀ (∂₁ W ∂₁ᵀ) φ
                ∇E = 0 ⇒ L₀^W φ = ∂₁ I.

            Consistencia con Ohm (FASE 2):
                f = W ∂₁ᵀ p  ⇒  f ∈ W·im(∂₁ᵀ)  exactamente.
                Por tanto, para el flujo de Poisson, I_curl ≈ 0 y
                I_grad ≈ I (salvo error numérico).

            Gauge: LSQR sobre L₀^W (singular) ⇒ ⟨φ, 1⟩ = 0 (FIX-M6).
            """
            rhs_phi = boundary_1.dot(I)  # ∂₁ I  ∈ ℝ^{|V|}
            phi = _lsqr_solve(laplacian_0, rhs_phi)
            I_grad = np.asarray(
                W.dot(boundary_1.T.dot(phi)), dtype=np.float64
            ).ravel()
            return I_grad

        # ------------------------------------------------------------------
        # §3.2 — Proyección curl con peso W⁻¹ (FIX-M3)
        # ------------------------------------------------------------------

        @staticmethod
        def _project_curl(
            boundary_2: sp.csr_matrix,
            W_inv: sp.dia_matrix,
            I: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Proyección ortogonal de I sobre im(∂₂) en ⟨·,·⟩_W.

            Ecuaciones normales (FIX-M3):
                Minimizar_α  ‖I − ∂₂ α‖_W²
                ⇔  (∂₂ᵀ W⁻¹ ∂₂) α = ∂₂ᵀ W⁻¹ I
                ⇔  I_curl = ∂₂ α

            v6 usaba (∂₂ᵀ W ∂₂, ∂₂ᵀ W I): peso invertido → no ortogonal
            en la norma de disipación de Joule.
            """
            if boundary_2.shape[1] == 0:
                return np.zeros_like(I)

            # Gram G₂ = ∂₂ᵀ W⁻¹ ∂₂ ∈ ℝ^{|F|×|F|}
            G2 = boundary_2.T.dot(W_inv.dot(boundary_2))
            rhs_alpha = boundary_2.T.dot(W_inv.dot(I))
            alpha = _lsqr_solve(G2, rhs_alpha)
            I_curl = np.asarray(
                boundary_2.dot(alpha), dtype=np.float64
            ).ravel()
            return I_curl

        # ------------------------------------------------------------------
        # §3.3 — Ortogonalidad triple en ⟨·,·⟩_W
        # ------------------------------------------------------------------

        @staticmethod
        def _verify_hodge_orthogonality(
            I_grad: NDArray[np.float64],
            I_curl: NDArray[np.float64],
            I_harm: NDArray[np.float64],
            W_inv: sp.dia_matrix,
        ) -> bool:
            r"""
            Verifica ⟨I_a, I_b⟩_W = 0 para a ≠ b ∈ {grad, curl, harm}.
            """
            tol = PistonConstants.HODGE_ORTHO_TOL
            inner_gc = abs(_energy_inner_product(I_grad, I_curl, W_inv))
            inner_gh = abs(_energy_inner_product(I_grad, I_harm, W_inv))
            inner_ch = abs(_energy_inner_product(I_curl, I_harm, W_inv))
            ok = inner_gc < tol and inner_gh < tol and inner_ch < tol
            if not ok:
                logger.warning(
                    "Ortogonalidad Hodge violada: "
                    "⟨g,c⟩=%.3e ⟨g,h⟩=%.3e ⟨c,h⟩=%.3e (tol=%.3e).",
                    inner_gc, inner_gh, inner_ch, tol,
                )
            else:
                logger.debug(
                    "Ortogonalidad Hodge OK "
                    "(⟨g,c⟩=%.2e ⟨g,h⟩=%.2e ⟨c,h⟩=%.2e).",
                    inner_gc, inner_gh, inner_ch,
                )
            return ok

        # ------------------------------------------------------------------
        # §3.4 — Pitágoras energético (FIX-M5)
        # ------------------------------------------------------------------

        @staticmethod
        def _verify_pythagoras(
            I: NDArray[np.float64],
            I_grad: NDArray[np.float64],
            I_curl: NDArray[np.float64],
            I_harm: NDArray[np.float64],
            W_inv: sp.dia_matrix,
        ) -> Tuple[bool, float, float]:
            r"""
            Identidad de Pitágoras en ⊕_W:
                ‖I‖²_W = ‖I_grad‖²_W + ‖I_curl‖²_W + ‖I_harm‖²_W

            Retorna
            -------
            (ok, energy_components_sum, energy_original)
            """
            e_g = _energy_norm(I_grad, W_inv) ** 2
            e_c = _energy_norm(I_curl, W_inv) ** 2
            e_h = _energy_norm(I_harm, W_inv) ** 2
            e_sum = e_g + e_c + e_h
            e_I = _energy_norm(I, W_inv) ** 2
            scale = max(e_I, 1.0)
            ok = abs(e_I - e_sum) / scale <= PistonConstants.PYTHAGORAS_TOL
            if not ok:
                logger.warning(
                    "Pitágoras energético falló: ‖I‖²_W=%.6e vs "
                    "Σ‖I_•‖²_W=%.6e (Δ_rel=%.3e).",
                    e_I, e_sum, abs(e_I - e_sum) / scale,
                )
            return ok, e_sum, e_I

        # ------------------------------------------------------------------
        # §3.5 — decompose_flow: cierre de la cadena FASE 1→2→3
        # ------------------------------------------------------------------

        @classmethod
        def decompose_flow(
            cls,
            mesh: "SimplicialMesh",
            flow: "FlowState",
        ) -> "HodgeDecomposition":
            r"""
            Descompone I = I_grad ⊕_W I_curl ⊕_W I_harm.

            **Continúa de FASE 2:** ``flow.edge_flows`` es la 1-forma I
            producida por Ohm ponderado sobre el potencial de Poisson.

            Pipeline FASE 3
            ───────────────
            1. I_grad: L₀^W φ = ∂₁ I → I_grad = W ∂₁ᵀ φ   (FIX-M2)
            2. I_curl: G₂ α = ∂₂ᵀ W⁻¹ I → I_curl = ∂₂ α  (FIX-M3)
            3. I_harm = I − I_grad − I_curl
            4. Ortogonalidad triple en ⟨·,·⟩_W
            5. Pitágoras energético (FIX-M5)
            6. Residual de proyección (FIX-M10)
            7. Veto si ‖I_curl‖_W > MAX_VORTICITY_NORM

            Lanza
            -----
            ParasiticVorticityVetoError   : vorticidad parásita.
            HodgeMetricInconsistencyError : Pitágoras fallido (opcional
                                            como veto duro; aquí se reporta
                                            en el DTO y se advierte).
            """
            B1 = mesh.boundary_1
            B2 = mesh.boundary_2
            W = mesh.conductance_matrix
            W_inv = mesh.inv_conductance
            L0W = mesh.laplacian_0
            I = np.asarray(flow.edge_flows, dtype=np.float64).copy()

            # §3.5.1 Gradiente ponderado (FIX-M2)
            I_grad = cls._project_gradient(B1, L0W, W, I)

            # §3.5.2 Curl con W⁻¹ (FIX-M3)
            I_curl = cls._project_curl(B2, W_inv, I)

            # §3.5.3 Armónico residual
            I_harm = I - I_grad - I_curl

            # §3.5.4 Norma de vorticidad
            vort_norm_W = _energy_norm(I_curl, W_inv)

            # §3.5.5 Ortogonalidad
            ortho_ok = cls._verify_hodge_orthogonality(
                I_grad, I_curl, I_harm, W_inv
            )

            # §3.5.6 Pitágoras (FIX-M5)
            pyth_ok, energy_total, _energy_I = cls._verify_pythagoras(
                I, I_grad, I_curl, I_harm, W_inv
            )

            # §3.5.7 Residual de proyección (FIX-M10)
            # Por construcción algebraica I_harm = I − I_g − I_c ⇒ residual 0;
            # se mide ‖I − (I_g+I_c+I_h)‖_W para detectar corrupción numérica.
            recon = I_grad + I_curl + I_harm
            proj_residual = _energy_norm(I - recon, W_inv)

            logger.info(
                "Hodge-Helmholtz DEC ⟨·,·⟩_W: "
                "‖I_g‖_W=%.3e ‖I_c‖_W=%.3e ‖I_h‖_W=%.3e | "
                "ortho=%s pyth=%s res=%.2e | β₁≈%d.",
                _energy_norm(I_grad, W_inv),
                vort_norm_W,
                _energy_norm(I_harm, W_inv),
                "✓" if ortho_ok else "✗",
                "✓" if pyth_ok else "✗",
                proj_residual,
                mesh.betti_1_estimate,
            )

            if not pyth_ok:
                # Veto blando: se registra; el DTO carga pythagoras_ok=False
                # para que orquestadores aguas abajo decidan.
                logger.error(
                    "Inconsistencia métrica Hodge (Pitágoras). "
                    "Revisar W, ∂₁, ∂₂ o tolerancia numérica."
                )

            # §3.5.8 Veto de vorticidad parásita
            if vort_norm_W > PistonConstants.MAX_VORTICITY_NORM:
                raise ParasiticVorticityVetoError(
                    f"Veto termodinámico: ‖I_curl‖_W = {vort_norm_W:.4e} > "
                    f"{PistonConstants.MAX_VORTICITY_NORM:.4e}. "
                    f"Exergía en bucles estériles. Revisar topología o "
                    f"elevar MAX_VORTICITY_NORM si los ciclos son intencionales."
                )

            return HodgeDecomposition(
                i_grad=I_grad,
                i_curl=I_curl,
                i_harm=I_harm,
                vorticity_norm=vort_norm_W,
                orthogonality_ok=ortho_ok,
                pythagoras_ok=pyth_ok,
                energy_total=energy_total,
                projection_residual=proj_residual,
            )

    # ── Constructor y métodos públicos del agente ─────────────────────────

    def __init__(self) -> None:
        """Fases como métodos de clase; sin estado mutable de instancia."""

    # ----------------------------------------------------------------------
    # §4.1 — Vector de fuentes con ∑ sᵢ = 0
    # ----------------------------------------------------------------------

    def _create_injection_vector(
        self,
        nodes: List[str],
        source: str,
        sink: str,
        q_pump: float,
    ) -> InjectionVector:
        r"""
        Ensambla s con conservación de masa por construcción y verificación.

            s[source] = −Q,  s[sink] = +Q,  resto = 0
            ⇒ ∑ sᵢ = 0

        Corrige residuales de punto flotante si |∑ sᵢ| > 0 numéricamente.
        """
        node_list = list(nodes)
        try:
            u_idx = node_list.index(source)
        except ValueError as exc:
            raise TopologicalInvariantError(
                f"Fuente '{source}' ∉ V. Nodos: {node_list}."
            ) from exc
        try:
            v_idx = node_list.index(sink)
        except ValueError as exc:
            raise TopologicalInvariantError(
                f"Sumidero '{sink}' ∉ V. Nodos: {node_list}."
            ) from exc

        if u_idx == v_idx:
            raise TopologicalInvariantError(
                f"Fuente y sumidero coinciden ('{source}'); flujo nulo."
            )

        s_vector = np.zeros(len(node_list), dtype=np.float64)
        s_vector[u_idx] -= q_pump
        s_vector[v_idx] += q_pump

        balance = float(np.sum(s_vector))
        if abs(balance) > PistonConstants.SOURCE_BALANCE_TOL:
            s_vector[v_idx] -= balance
            logger.debug(
                "Corrección FP de balance Δ=%.4e en '%s'.", balance, sink
            )

        return InjectionVector(
            source_node=source,
            sink_node=sink,
            q_pump=q_pump,
            s_vector=s_vector,
            balance_verified=True,
        )

    # ----------------------------------------------------------------------
    # §4.2 — Orquestación FASE 1 → FASE 2 → FASE 3
    # ----------------------------------------------------------------------

    def execute_injection(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        cycles: List[List[str]],
        pump_source: str,
        pump_sink: str,
        reservoir_node: str,
        q_pump: float,
    ) -> Dict[str, Any]:
        r"""
        Transición de fase geométrica completa de la inyección.

        Encadenamiento formal
        ─────────────────────
        FASE 1  build_mesh            → SimplicialMesh
        §4.1    _create_injection_vector → InjectionVector
        FASE 2  solve_hydrodynamics(mesh, injection, ·) → FlowState
        FASE 3  decompose_flow(mesh, flow) → HodgeDecomposition
        empaquetado de invariantes para Dirac / Sheaf orchestrators
        """
        logger.info(
            "Inyección termodinámica Q=%.4f [%s → %s] (reservorio: %s).",
            q_pump, pump_source, pump_sink, reservoir_node,
        )

        # FASE 1 — Fibración topológica (sutura → FASE 2)
        mesh = self._Phase1_SimplicialBoundaryBuilder.build_mesh(
            nodes, edges, cycles
        )

        # Vector de fuentes balanceado
        injection = self._create_injection_vector(
            nodes, pump_source, pump_sink, q_pump
        )

        # FASE 2 — Poisson de Dirichlet (sutura → FASE 3)
        flow_state = self._Phase2_DirichletPoissonSolver.solve_hydrodynamics(
            mesh, injection, reservoir_node
        )

        # FASE 3 — Hodge-Helmholtz métrico
        hodge = self._Phase3_HodgeHelmholtzAuditor.decompose_flow(
            mesh, flow_state
        )

        result: Dict[str, Any] = {
            "status": "ANNIHILATED_PARASITIC_VORTICITY",
            "injection_q": float(q_pump),
            "nodal_pressures": flow_state.nodal_pressures.tolist(),
            "edge_flows": flow_state.edge_flows.tolist(),
            "laminar_flow_norm_W": float(
                _energy_norm(hodge.i_grad, mesh.inv_conductance)
            ),
            "vorticity_norm_W": float(hodge.vorticity_norm),
            "harmonic_norm_W": float(
                _energy_norm(hodge.i_harm, mesh.inv_conductance)
            ),
            "hodge_energy_total": float(hodge.energy_total),
            "hodge_orthogonal": hodge.orthogonality_ok,
            "hodge_pythagoras_ok": hodge.pythagoras_ok,
            "projection_residual_W": float(hodge.projection_residual),
            "topological_invariants": {
                "kirchhoff_residual": float(flow_state.kirchhoff_residual),
                "lambda_min_L_red": float(flow_state.lambda_min_L_red),
                "boundary_identity": mesh.boundary_identity,
                "source_balanced": injection.balance_verified,
                "betti_1_estimate": mesh.betti_1_estimate,
                "num_nodes": len(mesh.nodes),
                "num_edges": len(mesh.edges),
                "num_cycles": len(mesh.cycles),
            },
            "flow_components": {
                "i_grad": hodge.i_grad.tolist(),
                "i_curl": hodge.i_curl.tolist(),
                "i_harm": hodge.i_harm.tolist(),
            },
        }

        logger.info(
            "Inyección OK. ‖I_c‖_W=%.3e ‖I_g‖_W=%.3e KCL=%.3e λ_min=%.3e β₁≈%d.",
            result["vorticity_norm_W"],
            result["laminar_flow_norm_W"],
            result["topological_invariants"]["kirchhoff_residual"],
            result["topological_invariants"]["lambda_min_L_red"],
            result["topological_invariants"]["betti_1_estimate"],
        )
        return result

    # ----------------------------------------------------------------------
    # §4.3 — Funtor categórico φ: CategoricalState → CategoricalState
    # ----------------------------------------------------------------------

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        Funtor MIC: extrae payload, ejecuta FASES 1→2→3, retorna estado
        purificado en Stratum.PHYSICS con tensor hidrodinámico e invariantes.
        """
        payload = state.payload

        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        cycles = payload.get("cycles", [])
        pump_source = payload.get("pump_source")
        pump_sink = payload.get("pump_sink")
        reservoir_node = payload.get("reservoir_node")
        q_pump = float(payload.get("q_pump", 1.0))

        missing = [
            name
            for name, val in (
                ("nodes", nodes),
                ("edges", edges),
                ("pump_source", pump_source),
                ("pump_sink", pump_sink),
                ("reservoir_node", reservoir_node),
            )
            if not val
        ]
        if missing:
            raise TopologicalInvariantError(
                f"Payload degenerado: faltan tensores {missing}."
            )

        result_tensors = self.execute_injection(
            nodes, edges, cycles,
            pump_source, pump_sink, reservoir_node, q_pump,
        )

        new_payload = dict(payload)
        new_payload["hydrodynamic_tensor"] = result_tensors

        return CategoricalState(
            payload=new_payload,
            metadata=state.metadata,
            stratum=Stratum.PHYSICS,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "PistonConstants",
    "PistonInjectorError",
    "SingularLaplacianError",
    "HomologicalKirchhoffError",
    "ParasiticVorticityVetoError",
    "BoundaryComplexError",
    "SourceCompatibilityError",
    "HodgeMetricInconsistencyError",
    "SimplicialMesh",
    "InjectionVector",
    "FlowState",
    "HodgeDecomposition",
    "HodgeHelmholtzInjectorAgent",
]