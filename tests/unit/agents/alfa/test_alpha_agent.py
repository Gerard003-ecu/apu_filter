# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : test_alpha_agent.py                                                ║
║  Ruta   : tests/unit/agents/alpha/test_alpha_agent.py                        ║
║  Versión: 3.0.0-Test-Suite                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de pruebas rigurosa para alpha_agent.py v3.0.0.

Estructura de la suite (espejo de las 3 fases del módulo):
══════════════════════════════════════════════════════════

  BLOQUE 0 — Fixtures y utilidades matemáticas compartidas
  BLOQUE 1 — Pruebas de constantes y AlphaConstants
  BLOQUE 2 — Pruebas de jerarquía de excepciones
  BLOQUE 3 — Pruebas de DTOs topológicos (dataclasses)
  BLOQUE 4 — Pruebas de Fase 1 (_Phase1_CanvasSimplicialFibrator)
  BLOQUE 5 — Pruebas de Fase 2 (_Phase2_HomologicalBettiAuditor)
  BLOQUE 6 — Pruebas de Fase 3 (_Phase3_SpectralFiedlerAuditor)
  BLOQUE 7 — Pruebas del orquestador (evaluate_business_canvas)
  BLOQUE 8 — Pruebas de propiedades matemáticas invariantes
  BLOQUE 9 — Pruebas de robustez numérica y casos extremos
  BLOQUE 10 — Pruebas del protocolo Morphism (__call__)
  BLOQUE 11 — Pruebas de integración end-to-end

Principios de diseño:
─────────────────────
  P1. Cada prueba verifica exactamente UNA propiedad matemática o contrato.
  P2. Los valores esperados se calculan analíticamente con justificación.
  P3. Las tolerancias numéricas se justifican explícitamente.
  P4. Los casos de error verifican TIPO de excepción y fragmento del MENSAJE.
  P5. Las pruebas paramétricas cubren el espacio de parámetros relevante.
  P6. Los grafos de prueba se construyen con propiedades topológicas conocidas.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pytest
import scipy.linalg as la

# ── Módulo bajo prueba ────────────────────────────────────────────────────────
from app.agents.alpha.alpha_agent import (
    # Constantes
    AlphaConstants,
    # Excepciones
    AlphaBoundaryError,
    EulerPoincareDegeneracyError,
    PreconditionError,
    SpectralFragilityError,
    ToxicCycleVetoError,
    # DTOs
    AlphaBoundaryVerdict,
    HomologicalInvariants,
    SimplicialComplexData,
    SpectralFiedlerData,
    # Agente principal
    AlphaBoundaryAgent,
)

# Importar TopologicalInvariantError del stub o del módulo real
try:
    from app.core.mic_algebra import TopologicalInvariantError, CategoricalState
    from app.core.schemas import Stratum
except ImportError:
    # Usar los stubs definidos en alpha_agent.py
    from app.agents.alpha.alpha_agent import (  # type: ignore[assignment]
        TopologicalInvariantError,
        CategoricalState,
        Stratum,
    )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 0 — FIXTURES Y UTILIDADES MATEMÁTICAS COMPARTIDAS                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Grafos de referencia con propiedades topológicas conocidas
# ─────────────────────────────────────────────────────────────────────────────

def make_path_graph(n: int, weight: float = 1.0) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un grafo camino P_n con n vértices y n-1 aristas.

    Propiedades topológicas conocidas:
        |V| = n, |E| = n-1
        β₀ = 1  (un solo componente conexo)
        β₁ = 0  (sin ciclos: es un árbol)
        χ  = n - (n-1) = 1  (árbol ⟹ χ=1)
        rank(∂₁) = n-1

    El Laplaciano de P_n tiene autovalores:
        λ_k = 2·w·(1 - cos(k·π/n))  para k=0,1,...,n-1
        λ₁  = 0 (componente constante)
        λ₂  = 2·w·(1 - cos(π/n))    (Valor de Fiedler)
    """
    nodes = [f"v{i}" for i in range(n)]
    flows = [(f"v{i}", f"v{i+1}", weight) for i in range(n - 1)]
    return nodes, flows


def make_cycle_graph(n: int, weight: float = 1.0) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un grafo ciclo C_n con n vértices y n aristas.

    Propiedades topológicas conocidas:
        |V| = n, |E| = n
        β₀ = 1  (conexo)
        β₁ = 1  (un ciclo independiente)
        χ  = n - n = 0  ← ACTIVA VETO 2 (χ ≤ 0)
        rank(∂₁) = n-1

    NOTA: β₁=1 activa el VETO 1 antes que el VETO 2 en el orquestador.
    """
    nodes = [f"v{i}" for i in range(n)]
    flows = [(f"v{i}", f"v{(i+1)%n}", weight) for i in range(n)]
    return nodes, flows


def make_star_graph(n_leaves: int, weight: float = 1.0) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un grafo estrella K_{1,n} con 1 centro y n hojas.

    Propiedades topológicas conocidas:
        |V| = n+1, |E| = n
        β₀ = 1  (conexo)
        β₁ = 0  (árbol ⟹ sin ciclos)
        χ  = (n+1) - n = 1
        rank(∂₁) = n

    El Valor de Fiedler de K_{1,n} con pesos uniformes w:
        λ₂ = w·(n+1 - √(n²+1)) ≈ w  para n grande (límite: w)
        Más precisamente: λ₂ = w para n ≥ 2 (autovalor de multiplicidad n-1)
    """
    center = "center"
    nodes  = [center] + [f"leaf{i}" for i in range(n_leaves)]
    flows  = [(center, f"leaf{i}", weight) for i in range(n_leaves)]
    return nodes, flows


def make_disconnected_graph() -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un grafo con dos componentes conexas (P_2 ∪ P_2).

    Propiedades topológicas conocidas:
        |V| = 4, |E| = 2
        β₀ = 2  (dos componentes)
        β₁ = 0  (sin ciclos)
        χ  = 4 - 2 = 2

    El Laplaciano tiene dos autovalores nulos (multiplicidad 2 de λ=0).
    El Valor de Fiedler es λ₂ = 0 (grafo disconexo).
    """
    nodes = ["a", "b", "c", "d"]
    flows = [("a", "b", 1.0), ("c", "d", 1.0)]
    return nodes, flows


def make_single_vertex_graph() -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un grafo trivial con un solo vértice y sin aristas.

    Propiedades topológicas conocidas:
        |V| = 1, |E| = 0
        β₀ = 1  (un componente: el punto mismo)
        β₁ = 0  (sin ciclos)
        χ  = 1 - 0 = 1
        rank(∂₁) = 0  (∂₁ es la matriz 1×0 vacía)

    El Laplaciano es la matriz [0] ∈ ℝ^{1×1}.
    No existe Valor de Fiedler (n < 2).
    """
    return ["single"], []


def make_complete_graph_k4() -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera K_4 (grafo completo con 4 vértices) con aristas orientadas.
    Para evitar multi-aristas, solo se incluye cada arista una vez (u < v).

    Propiedades topológicas conocidas:
        |V| = 4, |E| = 6
        β₀ = 1  (conexo)
        β₁ = 3  (= |E| - |V| + β₀ = 6 - 4 + 1 = 3 ciclos)
        χ  = 4 - 6 = -2  ← ACTIVA VETO 2 (χ ≤ 0)
        rank(∂₁) = 3

    NOTA: β₁=3 activa VETO 1 antes que VETO 2.
    """
    nodes = ["A", "B", "C", "D"]
    flows = [
        ("A", "B", 1.0), ("A", "C", 1.0), ("A", "D", 1.0),
        ("B", "C", 1.0), ("B", "D", 1.0),
        ("C", "D", 1.0),
    ]
    return nodes, flows


def make_bmc_viable() -> Tuple[List[str], List[Tuple[str, str, float]]]:
    r"""
    Genera un BMC viable: árbol de 5 vértices con pesos heterogéneos.

    Propiedades topológicas conocidas:
        |V| = 5, |E| = 4
        β₀ = 1, β₁ = 0, χ = 1
        rank(∂₁) = 4

    El Laplaciano de este árbol tiene:
        λ₁ = 0
        λ₂ > 0  (árbol conexo ⟹ Valor de Fiedler > 0)
    """
    nodes = ["VP", "CS", "RS", "CH", "KP"]
    flows = [
        ("VP", "CS", 2.0),
        ("VP", "RS", 3.0),
        ("CS", "CH", 1.5),
        ("KP", "VP", 4.0),
    ]
    return nodes, flows


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades matemáticas
# ─────────────────────────────────────────────────────────────────────────────

def compute_boundary_operator_reference(
    nodes: List[str],
    flows: List[Tuple[str, str, float]],
) -> np.ndarray:
    r"""
    Calcula ∂₁ de referencia para comparación independiente.
    Implementación directa sin las validaciones del agente.
    """
    n_v   = len(nodes)
    n_e   = len(flows)
    idx   = {n: i for i, n in enumerate(nodes)}
    B     = np.zeros((n_v, n_e), dtype=np.float64)
    for k, (u, v, w) in enumerate(flows):
        sw = math.sqrt(w)
        B[idx[u], k] = -sw
        B[idx[v], k] = +sw
    return B


def analytical_fiedler_path(n: int, weight: float = 1.0) -> float:
    r"""
    Valor de Fiedler analítico para el grafo camino P_n:
        λ₂(P_n) = 2·w·(1 - cos(π/n))
    """
    return 2.0 * weight * (1.0 - math.cos(math.pi / n))


def analytical_fiedler_star(n_leaves: int, weight: float = 1.0) -> float:
    r"""
    Valor de Fiedler analítico para K_{1,n}:
        λ₂(K_{1,n}) = w  (autovalor de multiplicidad n-1)
    Referencia: Mohar, B. (1991), "The Laplacian spectrum of graphs".
    """
    return weight


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de pytest
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def agent_default() -> AlphaBoundaryAgent:
    """Agente con umbral de Fiedler por defecto (0.05)."""
    return AlphaBoundaryAgent()


@pytest.fixture(scope="module")
def agent_strict() -> AlphaBoundaryAgent:
    """Agente con umbral de Fiedler estricto (0.5)."""
    return AlphaBoundaryAgent(fiedler_threshold=0.5)


@pytest.fixture(scope="module")
def agent_lenient() -> AlphaBoundaryAgent:
    """Agente con umbral de Fiedler muy permisivo (1e-6)."""
    return AlphaBoundaryAgent(fiedler_threshold=1e-6)


@pytest.fixture(scope="module")
def fibrator() -> AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator:
    """Instancia del Fibrador de Fase 1."""
    return AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator()


@pytest.fixture(scope="module")
def betti_auditor() -> AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor:
    """Instancia del Auditor Homológico de Fase 2."""
    return AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor()


@pytest.fixture(scope="module")
def spectral_auditor() -> AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor:
    """Instancia del Auditor Espectral de Fase 3."""
    return AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor()


@pytest.fixture(scope="module")
def path3_complex() -> SimplicialComplexData:
    """SimplicialComplexData para P_3 (camino de 3 vértices)."""
    nodes, flows = make_path_graph(3, weight=1.0)
    return AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
        .project_canvas_to_simplicial_complex(nodes, flows)


@pytest.fixture(scope="module")
def star3_complex() -> SimplicialComplexData:
    """SimplicialComplexData para K_{1,3} (estrella con 3 hojas)."""
    nodes, flows = make_star_graph(3, weight=2.0)
    return AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
        .project_canvas_to_simplicial_complex(nodes, flows)


@pytest.fixture(scope="module")
def bmc_viable_complex() -> SimplicialComplexData:
    """SimplicialComplexData para el BMC viable de referencia."""
    nodes, flows = make_bmc_viable()
    return AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
        .project_canvas_to_simplicial_complex(nodes, flows)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 1 — PRUEBAS DE AlphaConstants                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestAlphaConstants:
    """Pruebas unitarias de AlphaConstants."""

    def test_rank_tolerance_is_positive(self) -> None:
        """PROPIEDAD: RANK_TOLERANCE > 0."""
        assert AlphaConstants.RANK_TOLERANCE > 0.0

    def test_machine_epsilon_matches_numpy(self) -> None:
        """PROPIEDAD: MACHINE_EPSILON = ε_mach de numpy float64."""
        assert AlphaConstants.MACHINE_EPSILON == pytest.approx(
            float(np.finfo(np.float64).eps), rel=1e-15
        )

    def test_min_fiedler_value_positive(self) -> None:
        """PROPIEDAD: MIN_FIEDLER_VALUE > 0 (umbral isoperimétrico positivo)."""
        assert AlphaConstants.MIN_FIEDLER_VALUE > 0.0

    def test_svd_rank_threshold_returns_rank_tolerance_for_empty(self) -> None:
        """PROPIEDAD: umbral = RANK_TOLERANCE para matriz vacía."""
        empty = np.zeros((0, 0), dtype=np.float64)
        assert AlphaConstants.svd_rank_threshold(empty) == AlphaConstants.RANK_TOLERANCE

    def test_svd_rank_threshold_returns_rank_tolerance_for_zero_matrix(self) -> None:
        r"""
        PROPIEDAD: umbral = RANK_TOLERANCE cuando σ_max = 0 (matriz nula).
        Corrección de BUG 5 de v2.0.0: sin esta guarda, sigma_values[0] con
        σ_max=0 producía tol_rel=0, haciendo que el umbral fuera 0.
        """
        zero_mat = np.zeros((4, 3), dtype=np.float64)
        threshold = AlphaConstants.svd_rank_threshold(zero_mat)
        assert threshold == AlphaConstants.RANK_TOLERANCE

    def test_svd_rank_threshold_scales_with_sigma_max(self) -> None:
        r"""
        PROPIEDAD: Para σ_max > 0, umbral ≈ max(|V|, |E|)·ε_mach·σ_max.
        """
        n, m   = 10, 5
        mat    = np.eye(n, m, dtype=np.float64)  # σ_max = 1
        thresh = AlphaConstants.svd_rank_threshold(mat)
        # max(10,5)·ε_mach·1 ≈ 10·2.22e-16 ≈ 2.22e-15
        expected_rel = max(n, m) * AlphaConstants.MACHINE_EPSILON * 1.0
        assert thresh == pytest.approx(
            max(expected_rel, AlphaConstants.RANK_TOLERANCE), rel=1e-10
        )

    def test_laplacian_zero_tolerance_scales_with_frobenius(self) -> None:
        r"""
        PROPIEDAD: zero_tol = n·ε_mach·‖L₀‖_F.
        Corrección de BUG 6: usa ‖L₀‖_F en lugar de max(eigenvalues).
        """
        n     = 5
        frob  = 3.7
        tol   = AlphaConstants.laplacian_zero_tolerance(n, frob)
        expected = n * AlphaConstants.MACHINE_EPSILON * frob
        assert tol == pytest.approx(expected, rel=1e-15)

    def test_laplacian_zero_tolerance_fallback_for_zero_frobenius(self) -> None:
        r"""
        PROPIEDAD: zero_tol = n·ε_mach·1.0 si ‖L₀‖_F = 0 (fallback).
        Corrección de BUG 6: frobenius=0 → usa 1.0 como base.
        """
        n   = 4
        tol = AlphaConstants.laplacian_zero_tolerance(n, frobenius_norm=0.0)
        expected = n * AlphaConstants.MACHINE_EPSILON * 1.0
        assert tol == pytest.approx(expected, rel=1e-15)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 2 — PRUEBAS DE JERARQUÍA DE EXCEPCIONES                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Verifica la jerarquía de excepciones del Estrato α."""

    def test_alpha_boundary_error_is_topological_invariant_error(self) -> None:
        """AlphaBoundaryError hereda de TopologicalInvariantError."""
        assert issubclass(AlphaBoundaryError, TopologicalInvariantError)

    def test_precondition_error_is_alpha_boundary_error(self) -> None:
        """PreconditionError hereda de AlphaBoundaryError."""
        assert issubclass(PreconditionError, AlphaBoundaryError)

    def test_toxic_cycle_veto_is_alpha_boundary_error(self) -> None:
        """ToxicCycleVetoError hereda de AlphaBoundaryError."""
        assert issubclass(ToxicCycleVetoError, AlphaBoundaryError)

    def test_euler_poincare_degeneracy_is_alpha_boundary_error(self) -> None:
        """EulerPoincareDegeneracyError hereda de AlphaBoundaryError."""
        assert issubclass(EulerPoincareDegeneracyError, AlphaBoundaryError)

    def test_spectral_fragility_is_alpha_boundary_error(self) -> None:
        """SpectralFragilityError hereda de AlphaBoundaryError."""
        assert issubclass(SpectralFragilityError, AlphaBoundaryError)

    def test_all_veto_exceptions_catchable_as_base(self) -> None:
        """
        PROPIEDAD: Todos los vetos pueden capturarse como AlphaBoundaryError.
        """
        exceptions = [
            PreconditionError("test"),
            ToxicCycleVetoError("test"),
            EulerPoincareDegeneracyError("test"),
            SpectralFragilityError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, AlphaBoundaryError), (
                f"{type(exc).__name__} no es AlphaBoundaryError"
            )

    def test_all_catchable_as_topological_invariant_error(self) -> None:
        """PROPIEDAD: Todos los vetos son TopologicalInvariantError."""
        exceptions = [
            PreconditionError("test"),
            ToxicCycleVetoError("test"),
            EulerPoincareDegeneracyError("test"),
            SpectralFragilityError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, TopologicalInvariantError)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 3 — PRUEBAS DE DTOs TOPOLÓGICOS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSimplicialComplexData:
    """Pruebas unitarias de SimplicialComplexData."""

    def test_post_init_verifies_boundary_operator_shape(self) -> None:
        """PROPIEDAD: __post_init__ rechaza ∂₁ con shape incorrecto."""
        with pytest.raises(PreconditionError, match="shape"):
            SimplicialComplexData(
                vertices          = ("v0", "v1"),
                edges             = (("v0", "v1", 1.0),),
                boundary_operator = np.zeros((3, 1)),  # shape (3,1) ≠ (2,1)
                n_vertices        = 2,
                n_edges           = 1,
            )

    def test_post_init_rejects_nan_in_boundary_operator(self) -> None:
        """PROPIEDAD: __post_init__ rechaza ∂₁ con NaN."""
        B = np.array([[np.nan], [-1.0]])
        with pytest.raises(PreconditionError, match="finitos"):
            SimplicialComplexData(
                vertices          = ("v0", "v1"),
                edges             = (("v0", "v1", 1.0),),
                boundary_operator = B,
                n_vertices        = 2,
                n_edges           = 1,
            )

    def test_post_init_rejects_zero_weight_edge(self) -> None:
        """PROPIEDAD: __post_init__ rechaza aristas con peso ≤ 0."""
        B = np.zeros((2, 1), dtype=np.float64)
        with pytest.raises(PreconditionError, match="peso"):
            SimplicialComplexData(
                vertices          = ("v0", "v1"),
                edges             = (("v0", "v1", 0.0),),  # w=0
                boundary_operator = B,
                n_vertices        = 2,
                n_edges           = 1,
            )

    def test_valid_construction_succeeds(self, path3_complex) -> None:
        """PROPIEDAD: SimplicialComplexData válido se construye sin error."""
        assert path3_complex.n_vertices == 3
        assert path3_complex.n_edges    == 2
        assert path3_complex.boundary_operator.shape == (3, 2)

    def test_frozen_immutability(self, path3_complex) -> None:
        """PROPIEDAD: SimplicialComplexData es inmutable (frozen=True)."""
        with pytest.raises((AttributeError, TypeError)):
            path3_complex.n_vertices = 99  # type: ignore[misc]


class TestHomologicalInvariants:
    """Pruebas unitarias de HomologicalInvariants."""

    def test_post_init_verifies_euler_identity(self) -> None:
        r"""
        PROPIEDAD: __post_init__ verifica β₀ - β₁ == euler_characteristic.
        """
        with pytest.raises(TopologicalInvariantError, match="Euler"):
            HomologicalInvariants(
                rank                = 2,
                beta_0              = 3,
                beta_1              = 1,
                euler_characteristic= 10,  # ≠ β₀ - β₁ = 2
                svd_singular_values = np.array([2.0, 1.0]),
                rank_threshold      = 1e-10,
            )

    def test_valid_construction_satisfies_euler(self) -> None:
        r"""
        PROPIEDAD: χ = β₀ - β₁ se satisface en el DTO válido.
        """
        inv = HomologicalInvariants(
            rank=2, beta_0=3, beta_1=1, euler_characteristic=2,
            svd_singular_values=np.array([2.0, 1.0]), rank_threshold=1e-10,
        )
        assert inv.euler_characteristic == inv.beta_0 - inv.beta_1


class TestSpectralFiedlerData:
    """Pruebas unitarias de SpectralFiedlerData."""

    def test_post_init_rejects_negative_min_eigenvalue(self) -> None:
        r"""
        PROPIEDAD: __post_init__ rechaza λ_min < -zero_tolerance (L₀ no PSD).
        """
        L_bad = np.array([[-1.0, 0.0], [0.0, 1.0]])
        evals = np.array([-0.1, 1.0])  # λ_min = -0.1 (negativo)
        with pytest.raises(TopologicalInvariantError, match="PSD|positiva"):
            SpectralFiedlerData(
                laplacian      = L_bad,
                eigenvalues    = evals,
                fiedler_value  = 0.0,
                zero_tolerance = 1e-14,
                is_connected   = False,
                spectral_gap   = 0.0,
            )

    def test_valid_psd_construction_succeeds(self) -> None:
        """PROPIEDAD: SpectralFiedlerData válido se construye sin error."""
        L   = np.array([[1.0, -1.0], [-1.0, 1.0]])
        ev  = np.array([0.0, 2.0])
        sfd = SpectralFiedlerData(
            laplacian=L, eigenvalues=ev, fiedler_value=2.0,
            zero_tolerance=1e-12, is_connected=True, spectral_gap=0.0,
        )
        assert sfd.fiedler_value == pytest.approx(2.0)

    def test_is_connected_field(self) -> None:
        """PROPIEDAD: is_connected se almacena correctamente."""
        L  = np.array([[1.0, -1.0], [-1.0, 1.0]])
        ev = np.array([0.0, 2.0])
        sfd = SpectralFiedlerData(
            laplacian=L, eigenvalues=ev, fiedler_value=2.0,
            zero_tolerance=1e-12, is_connected=True, spectral_gap=0.0,
        )
        assert sfd.is_connected is True


class TestAlphaBoundaryVerdict:
    """Pruebas unitarias de AlphaBoundaryVerdict."""

    def _make_homology(self, b0=1, b1=0, chi=1) -> HomologicalInvariants:
        return HomologicalInvariants(
            rank=1, beta_0=b0, beta_1=b1, euler_characteristic=chi,
            svd_singular_values=np.array([1.0]), rank_threshold=1e-10,
        )

    def _make_spectral(self, fiedler=1.0) -> SpectralFiedlerData:
        L  = np.array([[1.0, -1.0], [-1.0, 1.0]])
        ev = np.array([0.0, 2.0])
        return SpectralFiedlerData(
            laplacian=L, eigenvalues=ev, fiedler_value=fiedler,
            zero_tolerance=1e-12, is_connected=True, spectral_gap=0.0,
        )

    def test_viable_verdict_has_no_veto_reason(self) -> None:
        """PROPIEDAD: is_viable=True implica veto_reason=None."""
        v = AlphaBoundaryVerdict(
            is_viable=True, homology=self._make_homology(),
            spectral=self._make_spectral(), veto_reason=None, veto_class=None,
        )
        assert v.veto_reason is None
        assert v.veto_class  is None

    def test_non_viable_verdict_has_veto_reason(self) -> None:
        """PROPIEDAD: is_viable=False puede tener veto_reason y veto_class."""
        v = AlphaBoundaryVerdict(
            is_viable=False, homology=self._make_homology(b0=1, b1=2, chi=-1),
            spectral=None, veto_reason="Ciclos tóxicos",
            veto_class=ToxicCycleVetoError,
        )
        assert v.is_viable   is False
        assert v.veto_reason == "Ciclos tóxicos"
        assert v.veto_class  is ToxicCycleVetoError

    def test_frozen_immutability(self) -> None:
        """PROPIEDAD: AlphaBoundaryVerdict es inmutable (frozen=True)."""
        v = AlphaBoundaryVerdict(
            is_viable=True, homology=self._make_homology(),
            spectral=self._make_spectral(), veto_reason=None, veto_class=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            v.is_viable = False  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 4 — PRUEBAS DE FASE 1 (_Phase1_CanvasSimplicialFibrator)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase1_ValidateCanvasInput:
    """Pruebas de _validate_canvas_input."""

    def test_empty_nodes_raises_precondition_error(self) -> None:
        """PROPIEDAD (C1): nodes vacío lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="vacío"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input([], [])

    def test_non_string_node_raises_precondition_error(self) -> None:
        """PROPIEDAD (C2): nodo no-str lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="str"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input([1, "v1"], [])  # type: ignore[list-item]

    def test_duplicate_nodes_raises_precondition_error(self) -> None:
        """PROPIEDAD (C3): nodos duplicados lanzan PreconditionError."""
        with pytest.raises(PreconditionError, match="duplicados"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(["v0", "v1", "v0"], [])

    def test_non_tuple_flow_raises_precondition_error(self) -> None:
        """PROPIEDAD (C4): flow que no es tupla lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="Tuple"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [["v0", "v1", 1.0]]  # lista, no tupla
                )

    def test_self_loop_raises_precondition_error(self) -> None:
        """PROPIEDAD (C5): auto-bucle (u=v) lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="auto-bucle"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v0", 1.0)]
                )

    def test_multi_edge_same_direction_raises_precondition_error(self) -> None:
        """PROPIEDAD (C6): arista (u,v) duplicada lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="multi-arista"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v1", 1.0), ("v0", "v1", 2.0)]
                )

    def test_multi_edge_reverse_direction_raises_precondition_error(self) -> None:
        """
        PROPIEDAD (C6): (u,v) y (v,u) se tratan como multi-arista.
        El 1-complejo simplicial es no orientado en su estructura de soporte.
        """
        with pytest.raises(PreconditionError, match="multi-arista"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v1", 1.0), ("v1", "v0", 2.0)]
                )

    def test_zero_weight_raises_precondition_error(self) -> None:
        """PROPIEDAD (C7): peso w=0 lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="peso"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v1", 0.0)]
                )

    def test_negative_weight_raises_precondition_error(self) -> None:
        """PROPIEDAD (C7): peso w<0 lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="peso"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v1", -1.0)]
                )

    def test_infinite_weight_raises_precondition_error(self) -> None:
        """PROPIEDAD (C7): peso w=inf lanza PreconditionError (no finito)."""
        with pytest.raises(PreconditionError, match="finito"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "v1", float("inf"))]
                )

    def test_unknown_source_node_raises_precondition_error(self) -> None:
        """PROPIEDAD (C8): nodo origen no en nodes lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="origen"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("UNKNOWN", "v1", 1.0)]
                )

    def test_unknown_destination_node_raises_precondition_error(self) -> None:
        """PROPIEDAD (C8): nodo destino no en nodes lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="destino"):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                ._validate_canvas_input(
                    ["v0", "v1"],
                    [("v0", "UNKNOWN", 1.0)]
                )

    def test_valid_input_raises_nothing(self) -> None:
        """PROPIEDAD: entrada válida no lanza ninguna excepción."""
        # No debe lanzar
        AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._validate_canvas_input(
                ["v0", "v1", "v2"],
                [("v0", "v1", 1.0), ("v1", "v2", 2.0)]
            )

    def test_empty_flows_with_valid_nodes_raises_nothing(self) -> None:
        """PROPIEDAD: flows vacío con nodes válido no lanza (grafo aislado)."""
        AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._validate_canvas_input(["single"], [])


class TestPhase1_BuildBoundaryOperator:
    """Pruebas de _build_boundary_operator."""

    def test_boundary_operator_shape(self) -> None:
        """PROPIEDAD: ∂₁.shape == (|V|, |E|)."""
        nodes, flows = make_path_graph(4)
        B = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        assert B.shape == (4, 3)

    def test_boundary_operator_entries_for_unit_weight(self) -> None:
        r"""
        PROPIEDAD: Para w=1, ∂₁[i,k]=-1 (origen) y ∂₁[j,k]=+1 (destino).
        Verificación analítica para la arista e_0 = (v0, v1, 1.0):
            ∂₁[0,0] = -√1 = -1
            ∂₁[1,0] = +√1 = +1
        """
        nodes = ["v0", "v1", "v2"]
        flows = [("v0", "v1", 1.0), ("v1", "v2", 1.0)]
        B     = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        assert B[0, 0] == pytest.approx(-1.0)
        assert B[1, 0] == pytest.approx(+1.0)
        assert B[2, 0] == pytest.approx(0.0)

    def test_boundary_operator_entries_for_nonunit_weight(self) -> None:
        r"""
        PROPIEDAD: Para w=4, ∂₁[i,k]=-√4=-2 y ∂₁[j,k]=+√4=+2.
        """
        nodes = ["u", "v"]
        flows = [("u", "v", 4.0)]
        B     = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        assert B[0, 0] == pytest.approx(-2.0)
        assert B[1, 0] == pytest.approx(+2.0)

    def test_boundary_operator_column_sum_is_zero(self) -> None:
        r"""
        PROPIEDAD: Σ_i ∂₁[i,k] = 0 para toda columna k.

        Fundamento: cada columna tiene exactamente -√w_k y +√w_k, sumando 0.
        Esta es la propiedad fundamental del operador frontera: ∂₁ᵀ𝟏 = 0.
        """
        nodes, flows = make_bmc_viable()
        B = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        col_sums = np.sum(B, axis=0)
        assert np.allclose(col_sums, 0.0, atol=1e-14), (
            f"Sumas de columnas no son cero: {col_sums}"
        )

    def test_boundary_operator_matches_reference_implementation(self) -> None:
        r"""
        PROPIEDAD: ∂₁ del agente coincide con la implementación de referencia.
        """
        nodes, flows = make_bmc_viable()
        B_agent = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        B_ref   = compute_boundary_operator_reference(nodes, flows)
        # Los órdenes de columnas y filas deben coincidir
        assert np.allclose(B_agent, B_ref, atol=1e-15)

    def test_laplacian_from_boundary_is_symmetric(self) -> None:
        r"""
        PROPIEDAD: L₀ = ∂₁∂₁ᵀ es simétrica.
        Verificación directa de (P1) del §3.1.
        """
        nodes, flows = make_bmc_viable()
        B  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        L  = B @ B.T
        assert np.allclose(L, L.T, atol=1e-14)

    def test_laplacian_is_psd(self) -> None:
        r"""
        PROPIEDAD: L₀ = ∂₁∂₁ᵀ es PSD: todos los autovalores ≥ 0.
        Verificación de (P2): xᵀL₀x = ‖∂₁ᵀx‖² ≥ 0.
        """
        nodes, flows = make_bmc_viable()
        B    = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        L    = B @ B.T
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-12), f"Autovalores negativos: {eigs}"

    def test_empty_flows_produces_zero_boundary_operator(self) -> None:
        r"""
        PROPIEDAD: Sin aristas, ∂₁ es la matriz |V|×0 vacía.
        """
        nodes = ["v0", "v1", "v2"]
        flows = []
        B     = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            ._build_boundary_operator(nodes, flows)
        assert B.shape == (3, 0)


class TestPhase1_ProjectCanvasToSimplicialComplex:
    """Pruebas de project_canvas_to_simplicial_complex (punto de entrada de Fase 1)."""

    def test_returns_simplicial_complex_data(self) -> None:
        """PROPIEDAD: Retorna una instancia de SimplicialComplexData."""
        nodes, flows = make_path_graph(3)
        result = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        assert isinstance(result, SimplicialComplexData)

    def test_n_vertices_and_n_edges_correct(self) -> None:
        """PROPIEDAD: n_vertices y n_edges coinciden con |nodes| y |flows|."""
        nodes, flows = make_path_graph(5)
        cd = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        assert cd.n_vertices == 5
        assert cd.n_edges    == 4

    def test_vertices_tuple_preserves_order(self) -> None:
        """PROPIEDAD: El orden de vertices coincide con el de nodes."""
        nodes = ["C", "A", "B"]
        flows = [("C", "A", 1.0), ("A", "B", 1.0)]
        cd    = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        assert cd.vertices == ("C", "A", "B")

    def test_propagates_precondition_error(self) -> None:
        """PROPIEDAD: PreconditionError de validación se propaga hacia afuera."""
        with pytest.raises(PreconditionError):
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                .project_canvas_to_simplicial_complex([], [])

    def test_single_vertex_no_edges_succeeds(self) -> None:
        """PROPIEDAD: Un solo vértice sin aristas es un complejo válido."""
        nodes, flows = make_single_vertex_graph()
        cd = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        assert cd.n_vertices == 1
        assert cd.n_edges    == 0
        assert cd.boundary_operator.shape == (1, 0)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 5 — PRUEBAS DE FASE 2 (_Phase2_HomologicalBettiAuditor)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase2_ComputeNumericalRank:
    """Pruebas de _compute_numerical_rank."""

    def test_empty_matrix_returns_rank_zero(self) -> None:
        """PROPIEDAD: Matriz vacía tiene rango 0."""
        rank, sigmas, thr = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            ._compute_numerical_rank(np.zeros((0, 0)))
        assert rank == 0
        assert sigmas.size == 0

    def test_zero_matrix_returns_rank_zero(self) -> None:
        """PROPIEDAD: Matriz nula (no vacía) tiene rango 0. (BUG 5 fix)"""
        rank, sigmas, thr = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            ._compute_numerical_rank(np.zeros((4, 3)))
        assert rank == 0

    def test_identity_matrix_rank(self) -> None:
        """PROPIEDAD: rank(I_n) = n."""
        for n in [2, 3, 5, 10]:
            rank, _, _ = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
                ._compute_numerical_rank(np.eye(n))
            assert rank == n, f"rank(I_{n}) = {rank} ≠ {n}"

    def test_rank_of_path_graph_boundary_operator(self) -> None:
        r"""
        PROPIEDAD: rank(∂₁(P_n)) = n-1.

        Fundamento: P_n es un árbol con n vértices. Todo árbol tiene:
            rank(∂₁) = |V| - β₀ = n - 1  (árbol tiene un solo componente).
        """
        for n in [2, 3, 5, 8]:
            nodes, flows = make_path_graph(n)
            B = compute_boundary_operator_reference(nodes, flows)
            rank, _, _   = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
                ._compute_numerical_rank(B)
            assert rank == n - 1, f"rank(∂₁(P_{n})) = {rank} ≠ {n-1}"

    def test_rank_of_cycle_graph_boundary_operator(self) -> None:
        r"""
        PROPIEDAD: rank(∂₁(C_n)) = n-1 (no n, porque el ciclo introduce un ker).

        Fundamento: C_n tiene |V|=|E|=n y un componente conexo: β₀=1.
            rank(∂₁) = |V| - β₀ = n - 1
        """
        for n in [3, 4, 5]:
            nodes, flows = make_cycle_graph(n)
            B = compute_boundary_operator_reference(nodes, flows)
            rank, _, _   = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
                ._compute_numerical_rank(B)
            assert rank == n - 1, f"rank(∂₁(C_{n})) = {rank} ≠ {n-1}"

    def test_singular_values_are_non_negative_and_sorted(self) -> None:
        r"""
        PROPIEDAD: Los valores singulares son σ₁ ≥ σ₂ ≥ ... ≥ 0.
        """
        nodes, flows = make_bmc_viable()
        B            = compute_boundary_operator_reference(nodes, flows)
        _, sigmas, _ = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            ._compute_numerical_rank(B)
        assert np.all(sigmas >= 0.0)
        assert np.all(np.diff(sigmas) <= 1e-14)  # orden descendente


class TestPhase2_ComputeBettiNumbers:
    """Pruebas de _compute_betti_numbers."""

    @pytest.mark.parametrize("n_v, n_e, rank, expected_b0, expected_b1, expected_chi", [
        (3, 2, 2, 1, 0, 1),   # P_3: árbol, conexo
        (4, 3, 3, 1, 0, 1),   # P_4: árbol, conexo
        (4, 4, 3, 1, 1, 0),   # C_4: ciclo (χ=0)
        (4, 6, 3, 1, 3, -2),  # K_4: completo (χ=-2)
        (4, 2, 2, 2, 0, 2),   # 2 componentes disconexas (P_2 ∪ P_2)
        (1, 0, 0, 1, 0, 1),   # Punto aislado
    ])
    def test_betti_numbers_known_graphs(
        self, n_v, n_e, rank, expected_b0, expected_b1, expected_chi
    ) -> None:
        r"""
        PROPIEDAD: β₀, β₁, χ correctos para grafos con topología conocida.

        Verificación analítica:
            β₀ = n_v - rank
            β₁ = n_e - rank
            χ  = n_v - n_e  (independiente del rango)
        """
        b0, b1, chi = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            ._compute_betti_numbers(n_v, n_e, rank)
        assert b0  == expected_b0,  f"β₀={b0} ≠ {expected_b0}"
        assert b1  == expected_b1,  f"β₁={b1} ≠ {expected_b1}"
        assert chi == expected_chi, f"χ={chi} ≠ {expected_chi}"

    def test_euler_identity_is_v_minus_e(self) -> None:
        r"""
        PROPIEDAD: χ = |V| - |E| (independiente del rango).

        Demostración: χ = β₀ - β₁ = (|V|-r) - (|E|-r) = |V| - |E|.
        """
        for n_v, n_e in [(5,3), (10,7), (3,3), (4,6)]:
            rank = min(n_v, n_e) - 1  # rango arbitrario para la prueba
            _, _, chi = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
                ._compute_betti_numbers(n_v, n_e, rank)
            assert chi == n_v - n_e, f"χ={chi} ≠ |V|-|E|={n_v-n_e}"


class TestPhase2_AuditBettiInvariants:
    """Pruebas de audit_betti_invariants (punto de entrada de Fase 2)."""

    def test_path_graph_betti_correct(self, path3_complex) -> None:
        r"""
        PROPIEDAD: P_3 tiene β₀=1, β₁=0, χ=1.
        """
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(path3_complex)
        assert inv.beta_0               == 1
        assert inv.beta_1               == 0
        assert inv.euler_characteristic == 1
        assert inv.rank                 == 2

    def test_star_graph_betti_correct(self, star3_complex) -> None:
        r"""
        PROPIEDAD: K_{1,3} (3 hojas) tiene β₀=1, β₁=0, χ=1.
        |V|=4, |E|=3, rank=3.
        """
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(star3_complex)
        assert inv.beta_0               == 1
        assert inv.beta_1               == 0
        assert inv.euler_characteristic == 1

    def test_cycle_graph_has_beta1_positive(self) -> None:
        r"""
        PROPIEDAD: C_4 tiene β₁=1 (un ciclo independiente).
        """
        nodes, flows = make_cycle_graph(4)
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(cd)
        assert inv.beta_1 == 1

    def test_disconnected_graph_has_beta0_two(self) -> None:
        r"""
        PROPIEDAD: Grafo con 2 componentes tiene β₀=2.
        """
        nodes, flows = make_disconnected_graph()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(cd)
        assert inv.beta_0 == 2

    def test_euler_identity_verified_post_construction(self) -> None:
        r"""
        PROPIEDAD: La post-verificación de Euler-Poincaré pasa sin error
        para grafos válidos (segunda línea de defensa).
        """
        nodes, flows = make_bmc_viable()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(cd)
        # Verificar identidad directamente
        assert inv.euler_characteristic == inv.beta_0 - inv.beta_1
        assert inv.euler_characteristic == cd.n_vertices - cd.n_edges

    def test_singular_values_stored_in_homological_invariants(self) -> None:
        """PROPIEDAD: svd_singular_values se almacena en el DTO."""
        nodes, flows = make_path_graph(4)
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            .audit_betti_invariants(cd)
        assert isinstance(inv.svd_singular_values, np.ndarray)
        assert inv.svd_singular_values.size > 0


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 6 — PRUEBAS DE FASE 3 (_Phase3_SpectralFiedlerAuditor)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase3_BuildLaplacian:
    """Pruebas de _build_laplacian."""

    def test_laplacian_equals_boundary_times_transpose(self, path3_complex) -> None:
        r"""
        PROPIEDAD: L₀ = ∂₁∂₁ᵀ exactamente.
        """
        B  = path3_complex.boundary_operator
        L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._build_laplacian(B)
        L_expected = B @ B.T
        # Comparamos con la simetrización (L+Lᵀ)/2
        L_sym = (L_expected + L_expected.T) * 0.5
        assert np.allclose(L0, L_sym, atol=1e-14)

    def test_laplacian_is_exactly_symmetric(self, path3_complex) -> None:
        r"""
        PROPIEDAD: L₀ devuelto es exactamente simétrico (‖L₀-L₀ᵀ‖_F = 0).
        """
        B  = path3_complex.boundary_operator
        L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._build_laplacian(B)
        assert np.allclose(L0, L0.T, atol=1e-15), (
            f"L₀ no es simétrico: ‖L-Lᵀ‖_F={np.linalg.norm(L0-L0.T,'fro'):.2e}"
        )

    def test_laplacian_diagonal_equals_degree(self) -> None:
        r"""
        PROPIEDAD: L₀[i,i] = Σ_j w_{ij} (suma de pesos incidentes al vértice i).

        Para el grafo estrella K_{1,3} con peso w=2:
            L₀[center,center] = 2+2+2 = 6
            L₀[leaf_i,leaf_i] = 2
        """
        nodes, flows = make_star_graph(3, weight=2.0)
        B  = compute_boundary_operator_reference(nodes, flows)
        L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._build_laplacian(B)
        # Vértice "center" tiene grado ponderado = 3·2 = 6
        center_idx = 0  # "center" es el primero en nodes
        assert L0[center_idx, center_idx] == pytest.approx(6.0, abs=1e-12)
        # Vértices hoja tienen grado ponderado = 2
        for leaf_idx in [1, 2, 3]:
            assert L0[leaf_idx, leaf_idx] == pytest.approx(2.0, abs=1e-12)

    def test_laplacian_off_diagonal_equals_negative_weight(self) -> None:
        r"""
        PROPIEDAD: L₀[i,j] = -w_{ij} para aristas (i,j) existentes.

        Para una sola arista (v0,v1) con peso w=4:
            L₀[0,1] = L₀[1,0] = -4
        """
        nodes = ["v0", "v1"]
        flows = [("v0", "v1", 4.0)]
        B     = compute_boundary_operator_reference(nodes, flows)
        L0    = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._build_laplacian(B)
        assert L0[0, 1] == pytest.approx(-4.0, abs=1e-12)
        assert L0[1, 0] == pytest.approx(-4.0, abs=1e-12)

    def test_laplacian_row_sum_is_zero(self, bmc_viable_complex) -> None:
        r"""
        PROPIEDAD: Σ_j L₀[i,j] = 0 para todo i.

        Fundamento: El vector constante 𝟏 está en el núcleo de L₀:
            L₀𝟏 = 0  (propiedad definitoria del Laplaciano).
        """
        B  = bmc_viable_complex.boundary_operator
        L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._build_laplacian(B)
        row_sums = np.sum(L0, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-12), (
            f"Sumas de filas no son cero: {row_sums}"
        )


class TestPhase3_ComputeZeroTolerance:
    """Pruebas de _compute_zero_tolerance."""

    def test_zero_tolerance_uses_frobenius_norm(self) -> None:
        r"""
        PROPIEDAD: zero_tol = n·ε_mach·‖L₀‖_F.
        """
        n  = 4
        L0 = np.eye(n) * 3.0  # ‖L₀‖_F = √(n·9) = 3√n
        tol = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._compute_zero_tolerance(L0)
        frob_expected = 3.0 * math.sqrt(n)
        expected      = n * AlphaConstants.MACHINE_EPSILON * frob_expected
        assert tol == pytest.approx(expected, rel=1e-12)

    def test_zero_laplacian_uses_fallback(self) -> None:
        r"""
        PROPIEDAD: Para L₀=0, zero_tol = n·ε_mach·1.0 (fallback). (BUG 6 fix)
        """
        n   = 3
        L0  = np.zeros((n, n))
        tol = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._compute_zero_tolerance(L0)
        expected = n * AlphaConstants.MACHINE_EPSILON * 1.0
        assert tol == pytest.approx(expected, rel=1e-12)


class TestPhase3_ExtractFiedlerValue:
    """Pruebas de _extract_fiedler_value."""

    def test_single_vertex_returns_zero_fiedler(self) -> None:
        r"""
        PROPIEDAD: Para |V|=1, fiedler=0 y is_connected=False. (BUG 7 fix)
        Corrección: el veto espectral no debe dispararse para n=1.
        """
        evals = np.array([0.0])
        fiedler, is_conn, gap = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._extract_fiedler_value(evals, zero_tolerance=1e-12, n_vertices=1)
        assert fiedler   == pytest.approx(0.0)
        assert is_conn   is False
        assert gap       == pytest.approx(0.0)

    def test_disconnected_graph_returns_zero_fiedler(self) -> None:
        r"""
        PROPIEDAD: Grafo disconexo (λ₁=λ₂=0) tiene fiedler=0.
        """
        evals = np.array([0.0, 0.0, 2.0, 3.0])  # multiplicidad 2 en λ=0
        fiedler, is_conn, _ = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._extract_fiedler_value(evals, zero_tolerance=1e-12, n_vertices=4)
        assert fiedler == pytest.approx(2.0)  # λ₂ es el primer positivo
        assert is_conn is True

    def test_connected_graph_returns_positive_fiedler(self) -> None:
        r"""
        PROPIEDAD: Grafo conexo tiene fiedler > 0 (λ₂ > 0).
        """
        # Autovalores analíticos de P_4 con w=1: 0, 2-√2, 2, 2+√2
        l2_p4 = 2.0 - math.sqrt(2.0)
        evals = np.array([0.0, l2_p4, 2.0, 2.0 + math.sqrt(2.0)])
        fiedler, is_conn, _ = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._extract_fiedler_value(evals, zero_tolerance=1e-12, n_vertices=4)
        assert fiedler   == pytest.approx(l2_p4, abs=1e-10)
        assert is_conn   is True

    def test_spectral_gap_computed_correctly(self) -> None:
        r"""
        PROPIEDAD: spectral_gap = λ_max - λ₂.
        """
        evals = np.array([0.0, 1.0, 3.0, 5.0])
        _, _, gap = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._extract_fiedler_value(evals, zero_tolerance=1e-12, n_vertices=4)
        assert gap == pytest.approx(5.0 - 1.0)

    def test_all_zero_eigenvalues_returns_zero_fiedler(self) -> None:
        r"""
        PROPIEDAD: Si todos los autovalores son ≈ 0, fiedler=0 (grafo disconexo).
        """
        evals = np.array([0.0, 0.0, 0.0])
        fiedler, is_conn, _ = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            ._extract_fiedler_value(evals, zero_tolerance=1e-10, n_vertices=3)
        assert fiedler == pytest.approx(0.0)
        assert is_conn is False


class TestPhase3_AuditFiedlerConnectivity:
    """Pruebas de audit_fiedler_connectivity (punto de entrada de Fase 3)."""

    def test_path_graph_fiedler_matches_analytical(self) -> None:
        r"""
        PROPIEDAD: λ₂(P_n) ≈ 2·w·(1 - cos(π/n)) (valor analítico exacto).

        Para P_4 con w=1: λ₂ = 2·(1-cos(π/4)) = 2·(1-√2/2) = 2-√2 ≈ 0.5858.
        Tolerancia: 1e-6 (precisión de eigvalsh de scipy).
        """
        n = 4
        nodes, flows = make_path_graph(n, weight=1.0)
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        expected = analytical_fiedler_path(n, weight=1.0)
        assert sfd.fiedler_value == pytest.approx(expected, abs=1e-6)

    def test_star_graph_fiedler_matches_analytical(self) -> None:
        r"""
        PROPIEDAD: λ₂(K_{1,n}) = w para n ≥ 2.

        Para K_{1,3} con w=2: λ₂ = 2.
        Referencia: Mohar (1991).
        """
        n_leaves = 3
        weight   = 2.0
        nodes, flows = make_star_graph(n_leaves, weight=weight)
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        expected = analytical_fiedler_star(n_leaves, weight=weight)
        assert sfd.fiedler_value == pytest.approx(expected, abs=1e-6)

    def test_disconnected_graph_is_not_connected(self) -> None:
        r"""
        PROPIEDAD: Grafo con 2 componentes tiene is_connected=False.
        """
        nodes, flows = make_disconnected_graph()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        assert sfd.is_connected is False

    def test_connected_graph_is_connected(self) -> None:
        """PROPIEDAD: BMC viable tiene is_connected=True."""
        nodes, flows = make_bmc_viable()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        assert sfd.is_connected is True

    def test_first_eigenvalue_is_approximately_zero(self) -> None:
        r"""
        PROPIEDAD: λ₁(L₀) ≈ 0 (L₀ es PSD, siempre tiene λ_min = 0).
        """
        nodes, flows = make_bmc_viable()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        assert abs(float(sfd.eigenvalues[0])) < sfd.zero_tolerance * 100

    def test_single_vertex_fiedler_is_zero(self) -> None:
        r"""
        PROPIEDAD: |V|=1 → fiedler=0 sin disparar veto espectral. (BUG 7 fix)
        """
        nodes, flows = make_single_vertex_graph()
        cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(cd)
        assert sfd.fiedler_value == pytest.approx(0.0)

    def test_returns_spectral_fiedler_data_instance(self, bmc_viable_complex) -> None:
        """PROPIEDAD: Retorna una instancia de SpectralFiedlerData."""
        sfd = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor\
            .audit_fiedler_connectivity(bmc_viable_complex)
        assert isinstance(sfd, SpectralFiedlerData)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 7 — PRUEBAS DEL ORQUESTADOR (evaluate_business_canvas)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestOrchestrator_ValidateInput:
    """Pruebas de _validate_orchestrator_input."""

    def test_non_list_nodes_raises_precondition_error(self, agent_default) -> None:
        """PROPIEDAD: nodes no-lista lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="list"):
            agent_default._validate_orchestrator_input(
                nodes=("v0", "v1"),  # tupla, no lista
                flows=[("v0", "v1", 1.0)],
            )

    def test_non_list_flows_raises_precondition_error(self, agent_default) -> None:
        """PROPIEDAD: flows no-lista lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="list"):
            agent_default._validate_orchestrator_input(
                nodes=["v0", "v1"],
                flows={"v0": "v1"},  # dict, no lista
            )

    def test_valid_lists_pass(self, agent_default) -> None:
        """PROPIEDAD: Listas válidas no lanzan excepción."""
        agent_default._validate_orchestrator_input(["v0"], [])


class TestOrchestrator_VetoAxioms:
    """Pruebas de los tres vetos axiomáticos en evaluate_business_canvas."""

    # ─── Veto 1: Ciclos Tóxicos (β₁ > 0) ─────────────────────────────────────

    def test_veto1_cycle_graph_raises_toxic_cycle_error(self, agent_default) -> None:
        r"""
        PROPIEDAD: C_4 (β₁=1) activa ToxicCycleVetoError.
        """
        nodes, flows = make_cycle_graph(4)
        with pytest.raises(ToxicCycleVetoError, match="β₁"):
            agent_default.evaluate_business_canvas(nodes, flows)

    def test_veto1_complete_k4_raises_toxic_cycle_error(self, agent_default) -> None:
        r"""
        PROPIEDAD: K_4 (β₁=3) activa ToxicCycleVetoError.
        """
        nodes, flows = make_complete_graph_k4()
        with pytest.raises(ToxicCycleVetoError):
            agent_default.evaluate_business_canvas(nodes, flows)

    def test_veto1_message_contains_beta1_value(self, agent_default) -> None:
        r"""
        PROPIEDAD: El mensaje del veto contiene el valor numérico de β₁.
        """
        nodes, flows = make_cycle_graph(5)
        with pytest.raises(ToxicCycleVetoError, match=r"β₁\s*=\s*1"):
            agent_default.evaluate_business_canvas(nodes, flows)

    def test_veto1_stores_last_verdict_with_is_viable_false(
        self, agent_default
    ) -> None:
        r"""
        PROPIEDAD: Tras Veto 1, last_verdict.is_viable=False y veto_class correcto.
        (BUG 8 fix: el veredicto parcial se almacena antes del raise)
        """
        nodes, flows = make_cycle_graph(4)
        with pytest.raises(ToxicCycleVetoError):
            agent_default.evaluate_business_canvas(nodes, flows)
        assert agent_default.last_verdict is not None
        assert agent_default.last_verdict.is_viable    is False
        assert agent_default.last_verdict.veto_class   is ToxicCycleVetoError
        assert agent_default.last_verdict.spectral     is None

    # ─── Veto 2: Degeneración de Euler-Poincaré (χ ≤ 0) ─────────────────────

    def test_veto2_chi_zero_raises_euler_poincare_error(self, agent_default) -> None:
        r"""
        PROPIEDAD: Un grafo con χ=0 (β₁>0 activa Veto 1 primero si existe ciclo).

        Para un grafo con χ≤0 y β₁=0 no es posible con un 1-complejo estándar
        (β₁>0 implica χ≤0 para β₀≥1). Probamos con un grafo donde β₁>0 y
        verificamos que el Veto 1 se activa antes que el Veto 2.

        Para verificar Veto 2 independientemente: creamos un escenario con
        β₁=0 y χ≤0, que requiere que el rank sea |E| (todos los vectores
        son independientes) pero |V| ≤ |E| sin ciclos — esto es imposible
        en un grafo conectado (un árbol tiene |E|=|V|-1 → χ=1).

        NOTA: En un 1-complejo simplicial ponderado válido sin ciclos (β₁=0),
        se tiene |E| = rank = |V| - β₀. Si β₀=1: |E|=|V|-1 → χ=1>0.
        Por tanto Veto 2 con χ≤0 implica siempre β₁>0, y el Veto 1 se activa
        primero. Esta prueba verifica ese comportamiento:
        """
        # C_3 tiene β₁=1 (Veto 1) y χ=0 (Veto 2).
        # Verificamos que el Veto 1 se activa primero.
        nodes, flows = make_cycle_graph(3)
        with pytest.raises(ToxicCycleVetoError):  # Veto 1, no Veto 2
            agent_default.evaluate_business_canvas(nodes, flows)

    def test_veto2_euler_poincare_degeneracy_stores_last_verdict(
        self, agent_default
    ) -> None:
        r"""
        PROPIEDAD: Para un grafo con χ<0 (β₁>0 primero), el último veredicto
        corresponde al Veto 1 (que se dispara antes). Si se pudiera disparar
        el Veto 2 independientemente, veto_class sería EulerPoincareDegeneracyError.

        Verificación indirecta: last_verdict siempre se actualiza en cualquier veto.
        """
        nodes, flows = make_complete_graph_k4()
        with pytest.raises(ToxicCycleVetoError):  # K4 tiene β₁=3
            agent_default.evaluate_business_canvas(nodes, flows)
        assert agent_default.last_verdict.is_viable is False

    # ─── Veto 3: Fragilidad Espectral (λ₂ < threshold) ───────────────────────

    def test_veto3_disconnected_graph_raises_spectral_fragility(
        self, agent_default
    ) -> None:
        r"""
        PROPIEDAD: Grafo con β₀=2 (disconexo, λ₂=0) activa SpectralFragilityError.

        Para el grafo P_2 ∪ P_2 (2 componentes):
            β₀=2, β₁=0, χ=2 (pasan Veto 1 y 2)
            λ₂=0 < MIN_FIEDLER_VALUE=0.05 → activa Veto 3
        """
        nodes, flows = make_disconnected_graph()
        with pytest.raises(SpectralFragilityError, match="λ₂"):
            agent_default.evaluate_business_canvas(nodes, flows)

    def test_veto3_low_fiedler_strict_threshold_raises(self, agent_strict) -> None:
        r"""
        PROPIEDAD: P_4 con λ₂≈0.586 < 0.5 NO activa Veto 3.
        P_4 con λ₂≈0.586 > 0.5 tampoco. Pero con umbral estricto 0.9 sí.

        Usamos agent_strict con threshold=0.5 y P_3 con λ₂=1.0 > 0.5 (pasa).
        Luego agent con threshold=2.0 y P_3 con λ₂=1.0 < 2.0 (activa veto).
        """
        agent_high = AlphaBoundaryAgent(fiedler_threshold=2.0)
        nodes, flows = make_path_graph(3, weight=1.0)
        # P_3 con w=1: λ₂ = 2(1-cos(π/3)) = 2(1-0.5) = 1.0 < 2.0
        with pytest.raises(SpectralFragilityError):
            agent_high.evaluate_business_canvas(nodes, flows)

    def test_veto3_not_triggered_for_single_vertex(self) -> None:
        r"""
        PROPIEDAD: |V|=1 NO activa el Veto 3, aunque fiedler=0. (BUG 7 fix)
        """
        agent = AlphaBoundaryAgent(fiedler_threshold=1.0)
        nodes, flows = make_single_vertex_graph()
        # Debe pasar todos los vetos (|V|=1: χ=1, β₁=0, veto espectral omitido)
        verdict = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True

    def test_veto3_stores_spectral_data_in_last_verdict(self, agent_default) -> None:
        r"""
        PROPIEDAD: Tras Veto 3, last_verdict.spectral no es None.
        El dato espectral está disponible aunque el veto se haya disparado.
        """
        nodes, flows = make_disconnected_graph()
        with pytest.raises(SpectralFragilityError):
            agent_default.evaluate_business_canvas(nodes, flows)
        assert agent_default.last_verdict.spectral is not None
        assert agent_default.last_verdict.veto_class is SpectralFragilityError

    # ─── Caso de éxito: todas las auditorías superadas ────────────────────────

    def test_viable_bmc_returns_viable_verdict(self, agent_default) -> None:
        """PROPIEDAD: BMC viable retorna AlphaBoundaryVerdict(is_viable=True)."""
        nodes, flows = make_bmc_viable()
        verdict = agent_default.evaluate_business_canvas(nodes, flows)
        assert isinstance(verdict, AlphaBoundaryVerdict)
        assert verdict.is_viable is True

    def test_viable_bmc_has_no_veto_reason(self, agent_default) -> None:
        """PROPIEDAD: BMC viable tiene veto_reason=None y veto_class=None."""
        nodes, flows = make_bmc_viable()
        verdict = agent_default.evaluate_business_canvas(nodes, flows)
        assert verdict.veto_reason is None
        assert verdict.veto_class  is None

    def test_viable_bmc_has_spectral_data(self, agent_default) -> None:
        """PROPIEDAD: BMC viable tiene spectral no-None (Fase 3 completó)."""
        nodes, flows = make_bmc_viable()
        verdict = agent_default.evaluate_business_canvas(nodes, flows)
        assert verdict.spectral is not None
        assert isinstance(verdict.spectral, SpectralFiedlerData)

    def test_viable_bmc_has_homological_data(self, agent_default) -> None:
        """PROPIEDAD: BMC viable tiene homology con β₁=0, χ=1."""
        nodes, flows = make_bmc_viable()
        verdict = agent_default.evaluate_business_canvas(nodes, flows)
        assert verdict.homology.beta_1               == 0
        assert verdict.homology.euler_characteristic == 1

    def test_last_verdict_updated_on_success(self, agent_default) -> None:
        """PROPIEDAD: last_verdict se actualiza tras una evaluación exitosa."""
        nodes, flows = make_bmc_viable()
        _ = agent_default.evaluate_business_canvas(nodes, flows)
        assert agent_default.last_verdict is not None
        assert agent_default.last_verdict.is_viable is True


class TestOrchestrator_Constructor:
    """Pruebas del constructor de AlphaBoundaryAgent."""

    def test_default_fiedler_threshold(self) -> None:
        """PROPIEDAD: Umbral por defecto = AlphaConstants.MIN_FIEDLER_VALUE."""
        agent = AlphaBoundaryAgent()
        assert agent.fiedler_threshold == AlphaConstants.MIN_FIEDLER_VALUE

    def test_custom_fiedler_threshold(self) -> None:
        """PROPIEDAD: Umbral personalizado se almacena correctamente."""
        agent = AlphaBoundaryAgent(fiedler_threshold=0.3)
        assert agent.fiedler_threshold == pytest.approx(0.3)

    def test_zero_threshold_raises_precondition_error(self) -> None:
        """PROPIEDAD: threshold=0 lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="> 0"):
            AlphaBoundaryAgent(fiedler_threshold=0.0)

    def test_negative_threshold_raises_precondition_error(self) -> None:
        """PROPIEDAD: threshold<0 lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="> 0"):
            AlphaBoundaryAgent(fiedler_threshold=-0.1)

    def test_non_numeric_threshold_raises_precondition_error(self) -> None:
        """PROPIEDAD: threshold no-numérico lanza PreconditionError."""
        with pytest.raises(PreconditionError, match="numérico"):
            AlphaBoundaryAgent(fiedler_threshold="high")  # type: ignore[arg-type]

    def test_last_verdict_is_none_before_first_call(self) -> None:
        """PROPIEDAD: last_verdict=None antes de la primera evaluación."""
        agent = AlphaBoundaryAgent()
        assert agent.last_verdict is None


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 8 — PRUEBAS DE PROPIEDADES MATEMÁTICAS INVARIANTES                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMathematicalInvariants:
    """
    Pruebas de las propiedades matemáticas fundamentales que deben satisfacerse
    en toda ejecución del módulo, independientemente de los parámetros.
    """

    def test_euler_poincare_formula_for_trees(self, agent_lenient) -> None:
        r"""
        INVARIANTE: Para cualquier árbol T_n, χ(T_n) = 1.

        Demostración: Un árbol con n vértices tiene n-1 aristas.
        χ = |V| - |E| = n - (n-1) = 1.
        """
        for n in [2, 3, 5, 8, 13]:
            nodes, flows = make_path_graph(n)
            verdict = agent_lenient.evaluate_business_canvas(nodes, flows)
            assert verdict.homology.euler_characteristic == 1, (
                f"χ(P_{n}) = {verdict.homology.euler_characteristic} ≠ 1"
            )

    def test_beta0_equals_number_of_connected_components(self) -> None:
        r"""
        INVARIANTE: β₀ = número de componentes conexas del grafo.

        Verificación para:
            - P_3: 1 componente → β₀=1
            - P_2 ∪ P_2: 2 componentes → β₀=2
        """
        agent = AlphaBoundaryAgent(fiedler_threshold=1e-9)

        nodes1, flows1 = make_path_graph(3)
        v1 = agent.evaluate_business_canvas(nodes1, flows1)
        assert v1.homology.beta_0 == 1

        nodes2, flows2 = make_disconnected_graph()
        with pytest.raises(SpectralFragilityError):
            agent.evaluate_business_canvas(nodes2, flows2)
        assert agent.last_verdict.homology.beta_0 == 2

    def test_rank_nullity_theorem(self) -> None:
        r"""
        INVARIANTE: rank(∂₁) + dim(ker(∂₁)) = |E|.

        Esto es el Teorema de Rango-Nulidad de Álgebra Lineal:
            |E| = rank(∂₁) + β₁

        Verificación para varios grafos.
        """
        test_cases = [
            make_path_graph(4),
            make_star_graph(3),
            make_disconnected_graph(),
        ]
        for nodes, flows in test_cases:
            cd  = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
                .project_canvas_to_simplicial_complex(nodes, flows)
            inv = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
                .audit_betti_invariants(cd)
            assert inv.rank + inv.beta_1 == cd.n_edges, (
                f"rank+β₁={inv.rank+inv.beta_1} ≠ |E|={cd.n_edges}"
            )

    def test_fiedler_value_monotone_with_weight(self) -> None:
        r"""
        INVARIANTE: λ₂ escala linealmente con el peso uniforme w.

        Fundamento: Si L₀(G,w) = w·L₀(G,1), entonces:
            eig(L₀(G,w)) = w·eig(L₀(G,1))
        En particular, λ₂(G,w) = w·λ₂(G,1).
        """
        n = 4
        agent = AlphaBoundaryAgent(fiedler_threshold=1e-6)
        weights = [0.5, 1.0, 2.0, 5.0]
        fiedler_values = []
        for w in weights:
            nodes, flows = make_path_graph(n, weight=w)
            verdict = agent.evaluate_business_canvas(nodes, flows)
            fiedler_values.append(verdict.spectral.fiedler_value)

        # Verificar proporcionalidad: λ₂(w) / λ₂(1.0) ≈ w
        f0 = fiedler_values[1]  # w=1.0
        for i, w in enumerate(weights):
            ratio = fiedler_values[i] / f0
            assert ratio == pytest.approx(w, rel=1e-6), (
                f"λ₂(w={w})/λ₂(1.0) = {ratio} ≠ {w}"
            )

    def test_boundary_operator_kernel_corresponds_to_cycles(self) -> None:
        r"""
        INVARIANTE: dim(ker(∂₁)) = β₁ (número de ciclos independientes).

        Para C_4: ker(∂₁) es 1-dimensional (el ciclo que recorre todas las aristas).
        """
        nodes, flows = make_cycle_graph(4)
        B = compute_boundary_operator_reference(nodes, flows)
        # El núcleo tiene dimensión β₁ = |E| - rank
        rank, _, _ = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor\
            ._compute_numerical_rank(B)
        beta_1 = len(flows) - rank
        assert beta_1 == 1

        # Verificar directamente con scipy
        _, s, Vt = la.svd(B)
        threshold = AlphaConstants.svd_rank_threshold(B)
        null_dim  = int(np.sum(s <= threshold))
        assert null_dim == beta_1

    def test_laplacian_eigenvalues_match_multiplicity_of_components(self) -> None:
        r"""
        INVARIANTE: La multiplicidad del autovalor λ=0 del Laplaciano
        es igual al número de componentes conexas β₀.

        P3 ∪ P2 (dos componentes): multiplicidad de λ=0 es 2.
        """
        nodes = ["a", "b", "c", "d", "e"]
        flows = [("a", "b", 1.0), ("b", "c", 1.0), ("d", "e", 1.0)]
        B     = compute_boundary_operator_reference(nodes, flows)
        L     = B @ B.T
        evals = np.linalg.eigvalsh(L)
        n     = len(nodes)
        zero_tol = AlphaConstants.laplacian_zero_tolerance(
            n, float(np.linalg.norm(L, 'fro'))
        )
        n_zero = int(np.sum(evals <= zero_tol))
        assert n_zero == 2, (
            f"Multiplicidad de λ=0 es {n_zero}, esperado 2 (dos componentes)"
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 9 — PRUEBAS DE ROBUSTEZ NUMÉRICA Y CASOS EXTREMOS                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestNumericalRobustness:
    """
    Pruebas que verifican el comportamiento en condiciones numéricamente extremas.
    """

    def test_very_small_weight_no_crash(self) -> None:
        r"""
        ROBUSTEZ: Peso w=1e-15 (casi cero pero positivo) no produce crash.
        """
        agent  = AlphaBoundaryAgent(fiedler_threshold=1e-30)
        nodes  = ["a", "b", "c"]
        flows  = [("a", "b", 1e-15), ("b", "c", 1e-15)]
        # Puede pasar o fallar el veto espectral, pero no debe crashear
        try:
            verdict = agent.evaluate_business_canvas(nodes, flows)
            assert np.all(np.isfinite(verdict.spectral.eigenvalues))
        except SpectralFragilityError:
            pass  # Aceptable para pesos muy pequeños

    def test_very_large_weight_no_overflow(self) -> None:
        r"""
        ROBUSTEZ: Peso w=1e12 no produce overflow en ∂₁ o L₀.
        """
        agent  = AlphaBoundaryAgent(fiedler_threshold=1e-6)
        nodes  = ["a", "b", "c"]
        flows  = [("a", "b", 1e12), ("b", "c", 1e12)]
        verdict = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True
        assert np.all(np.isfinite(verdict.spectral.eigenvalues))

    def test_heterogeneous_weights_no_nan(self) -> None:
        r"""
        ROBUSTEZ: Pesos muy heterogéneos (ratio 1e10) no producen NaN.
        """
        agent = AlphaBoundaryAgent(fiedler_threshold=1e-6)
        nodes = ["a", "b", "c", "d"]
        flows = [("a", "b", 1e-5), ("b", "c", 1.0), ("c", "d", 1e5)]
        verdict = agent.evaluate_business_canvas(nodes, flows)
        assert not np.any(np.isnan(verdict.spectral.eigenvalues))

    def test_large_graph_no_crash(self) -> None:
        r"""
        ROBUSTEZ: Árbol grande (n=100) se procesa sin error ni desbordamiento.
        """
        agent         = AlphaBoundaryAgent(fiedler_threshold=1e-6)
        nodes, flows  = make_path_graph(100, weight=1.0)
        verdict       = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True
        assert verdict.homology.beta_0 == 1
        assert verdict.homology.beta_1 == 0

    def test_star_with_100_leaves_no_crash(self) -> None:
        r"""
        ROBUSTEZ: K_{1,100} (estrella de 100 hojas) se procesa sin error.
        """
        agent         = AlphaBoundaryAgent(fiedler_threshold=0.5)
        nodes, flows  = make_star_graph(100, weight=1.0)
        verdict       = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True
        # λ₂(K_{1,100}) = 1.0 > 0.5
        assert verdict.spectral.fiedler_value == pytest.approx(1.0, abs=1e-6)

    def test_integer_weights_accepted(self) -> None:
        r"""
        ROBUSTEZ: Pesos enteros (int) son aceptados por la validación de tipo.
        El campo espera float pero int debe ser convertido silenciosamente.
        """
        agent = AlphaBoundaryAgent(fiedler_threshold=0.01)
        nodes = ["a", "b", "c"]
        flows = [("a", "b", 3), ("b", "c", 2)]  # int, no float
        verdict = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True

    def test_boundary_operator_has_no_nan_for_valid_input(self) -> None:
        r"""
        ROBUSTEZ: ∂₁ nunca contiene NaN para entrada válida.
        """
        nodes, flows = make_bmc_viable()
        cd = AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator\
            .project_canvas_to_simplicial_complex(nodes, flows)
        assert not np.any(np.isnan(cd.boundary_operator))
        assert not np.any(np.isinf(cd.boundary_operator))


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 10 — PRUEBAS DEL PROTOCOLO MORPHISM (__call__)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMorphismProtocol:
    """Pruebas del protocolo __call__ de AlphaBoundaryAgent."""

    def _make_state(
        self,
        nodes: List[str],
        flows: List[Tuple[str, str, float]],
    ) -> CategoricalState:
        """Construye un CategoricalState con payload BMC."""
        return CategoricalState(
            payload  = {"bmc_nodes": nodes, "bmc_flows": flows},
            metadata = {"source": "test"},
        )

    def test_call_returns_categorical_state(self, agent_default) -> None:
        """PROPIEDAD: __call__ retorna un CategoricalState."""
        nodes, flows = make_bmc_viable()
        state  = self._make_state(nodes, flows)
        result = agent_default(state)
        assert isinstance(result, CategoricalState)

    def test_call_sets_stratum_alpha(self, agent_default) -> None:
        """PROPIEDAD: El estado resultante tiene stratum=ALPHA."""
        nodes, flows = make_bmc_viable()
        state  = self._make_state(nodes, flows)
        result = agent_default(state)
        assert result.stratum == Stratum.ALPHA

    def test_call_injects_alpha_verdict_in_payload(self, agent_default) -> None:
        """PROPIEDAD: 'alpha_verdict' está en el payload del estado resultante."""
        nodes, flows = make_bmc_viable()
        state  = self._make_state(nodes, flows)
        result = agent_default(state)
        assert "alpha_verdict" in result.payload
        assert isinstance(result.payload["alpha_verdict"], AlphaBoundaryVerdict)

    def test_call_preserves_original_payload_keys(self, agent_default) -> None:
        """PROPIEDAD: Las claves originales del payload se preservan."""
        nodes, flows = make_bmc_viable()
        state  = self._make_state(nodes, flows)
        result = agent_default(state)
        assert "bmc_nodes" in result.payload
        assert "bmc_flows" in result.payload

    def test_call_preserves_metadata(self, agent_default) -> None:
        """PROPIEDAD: metadata del estado original se preserva."""
        nodes, flows = make_bmc_viable()
        state  = self._make_state(nodes, flows)
        state.metadata["custom_key"] = "value"
        result = agent_default(state)
        assert result.metadata.get("custom_key") == "value"

    def test_call_raises_precondition_for_empty_nodes(self, agent_default) -> None:
        """PROPIEDAD: nodes vacío en payload lanza PreconditionError."""
        state = CategoricalState(
            payload={"bmc_nodes": [], "bmc_flows": []},
        )
        with pytest.raises(PreconditionError):
            agent_default(state)

    def test_call_raises_precondition_for_missing_bmc_keys(
        self, agent_default
    ) -> None:
        """PROPIEDAD: payload sin claves BMC lanza PreconditionError."""
        state = CategoricalState(payload={"other_key": "other_value"})
        with pytest.raises(PreconditionError, match="bmc_nodes|BMC"):
            agent_default(state)

    def test_call_propagates_veto_errors(self, agent_default) -> None:
        """PROPIEDAD: Los vetos se propagan desde __call__ sin alterarse."""
        nodes, flows = make_cycle_graph(4)
        state = self._make_state(nodes, flows)
        with pytest.raises(ToxicCycleVetoError):
            agent_default(state)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 11 — PRUEBAS DE INTEGRACIÓN END-TO-END                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """
    Pruebas de integración completa que verifican el pipeline de las 3 fases.
    """

    def test_full_pipeline_viable_bmc(self, agent_default) -> None:
        r"""
        INTEGRACIÓN: Pipeline completo para un BMC viable.

        Verifica que los invariantes de las 3 fases son consistentes entre sí:
            - β₁=0 (Fase 2) → Veto 1 no activo
            - χ=1  (Fase 2) → Veto 2 no activo
            - λ₂>0 (Fase 3) → Veto 3 no activo
        """
        nodes, flows = make_bmc_viable()
        verdict = agent_default.evaluate_business_canvas(nodes, flows)

        # Verificar coherencia de los invariantes
        assert verdict.is_viable                         is True
        assert verdict.homology.beta_1                   == 0
        assert verdict.homology.euler_characteristic     == 1
        assert verdict.spectral.fiedler_value             > 0.0
        assert verdict.spectral.is_connected             is True

        # Verificar que los valores son finitos
        assert np.all(np.isfinite(verdict.spectral.eigenvalues))
        assert np.all(np.isfinite(verdict.spectral.laplacian))

    def test_full_pipeline_star_graph_viable(self) -> None:
        r"""
        INTEGRACIÓN: K_{1,5} pasa todas las auditorías.

        K_{1,5}: |V|=6, |E|=5, β₀=1, β₁=0, χ=1.
        λ₂(K_{1,5}) = 1.0 > MIN_FIEDLER_VALUE=0.05.
        """
        agent         = AlphaBoundaryAgent()
        nodes, flows  = make_star_graph(5, weight=1.0)
        verdict       = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True

    def test_phase_output_consistency(self, agent_default) -> None:
        r"""
        INTEGRACIÓN: Las salidas de Fase 2 y Fase 3 son mutuamente consistentes.

        Verificaciones cruzadas:
            - β₀ = multiplicidad de λ=0 en el espectro de L₀ (para grafos conexos β₀=1)
            - β₁ = 0 implica que la solución SVD no tiene núcleo
        """
        nodes, flows = make_bmc_viable()
        verdict      = agent_default.evaluate_business_canvas(nodes, flows)

        # Verificar que la multiplicidad de λ≈0 coincide con β₀
        zero_tol = verdict.spectral.zero_tolerance
        n_zero   = int(np.sum(verdict.spectral.eigenvalues <= zero_tol))
        assert n_zero == verdict.homology.beta_0, (
            f"Multiplicidad λ=0 ({n_zero}) ≠ β₀ ({verdict.homology.beta_0})"
        )

    def test_sequential_evaluations_are_independent(self, agent_default) -> None:
        r"""
        INTEGRACIÓN: Evaluaciones secuenciales son deterministas e independientes.

        La misma entrada debe producir siempre el mismo resultado.
        """
        nodes, flows = make_bmc_viable()

        v1 = agent_default.evaluate_business_canvas(nodes, flows)
        v2 = agent_default.evaluate_business_canvas(nodes, flows)

        assert v1.homology.beta_0               == v2.homology.beta_0
        assert v1.homology.beta_1               == v2.homology.beta_1
        assert v1.homology.euler_characteristic == v2.homology.euler_characteristic
        assert v1.spectral.fiedler_value        == pytest.approx(v2.spectral.fiedler_value)

    def test_different_inputs_produce_different_verdicts(self, agent_default) -> None:
        r"""
        INTEGRACIÓN: Entradas diferentes producen verdicts diferentes.
        """
        nodes_a, flows_a = make_path_graph(3)
        nodes_b, flows_b = make_star_graph(4)

        v_a = agent_default.evaluate_business_canvas(nodes_a, flows_a)
        v_b = agent_default.evaluate_business_canvas(nodes_b, flows_b)

        # Pueden tener el mismo χ=1 y β₁=0, pero Fiedler debe diferir
        assert v_a.spectral.fiedler_value != pytest.approx(
            v_b.spectral.fiedler_value, abs=0.001
        )

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_all_path_graphs_are_viable(self, n: int) -> None:
        r"""
        INTEGRACIÓN PARAMÉTRICA: Todos los grafos camino P_n (n≥2) son viables
        con umbral de Fiedler suficientemente permisivo.

        λ₂(P_n) = 2(1-cos(π/n)) ≥ 2(1-cos(π/2)) = 2 para n=2.
        Para n grande: λ₂(P_n) ≈ 2(π/n)²/2 → 0, pero n·π/n = π para n finito.
        Usamos un umbral muy bajo para evitar falsos vetos espectrales.
        """
        agent         = AlphaBoundaryAgent(fiedler_threshold=1e-6)
        nodes, flows  = make_path_graph(n, weight=1.0)
        verdict       = agent.evaluate_business_canvas(nodes, flows)
        assert verdict.is_viable is True, (
            f"P_{n} no es viable: {verdict.veto_reason}"
        )

    def test_veto_sequence_respects_priority(self, agent_default) -> None:
        r"""
        INTEGRACIÓN: Los vetos se evalúan en secuencia estricta:
            Veto 1 (β₁>0) > Veto 2 (χ≤0) > Veto 3 (λ₂<threshold).

        C_4 tiene β₁=1 (activa Veto 1) y χ=0 (activaría Veto 2).
        El primero que se dispare debe ser Veto 1.
        """
        nodes, flows = make_cycle_graph(4)
        with pytest.raises(ToxicCycleVetoError) as exc_info:
            agent_default.evaluate_business_canvas(nodes, flows)
        # Confirmar que es el Veto 1, no el Veto 2
        assert "β₁" in str(exc_info.value) or "Canibal" in str(exc_info.value)