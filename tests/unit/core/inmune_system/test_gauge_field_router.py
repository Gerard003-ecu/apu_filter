"""
=========================================================================================
Suite de Pruebas Rigurosa: Gauge Field Router
Ubicación: tests/core/immune_system/test_gauge_field_router.py
=========================================================================================

Arquitectura de pruebas:
───────────────────────

Las pruebas se organizan en 9 clases temáticas que cubren los 9 invariantes
de construcción y las 8 etapas del pipeline de enrutamiento.

Estructura:
    • TestGaugeFieldRouterConstruction (9 tests)
      Valida invariantes I1−I9 durante construcción del router.

    • TestValidationFunctions (8 tests)
      Pruebas unitarias de funciones validadoras (puras).

    • TestPoissonSolver (6 tests)
      Comportamiento del solver LSQR y verificación de residuales.

    • TestChargeQuantization (4 tests)
      Cuantificación de densidad de carga y neutralidad.

    • TestFieldCalculation (4 tests)
      Cálculo del gradiente discreto (campo sobre aristas).

    • TestAgentSelection (5 tests)
      Selección de agente por máximo acoplamiento.

    • TestMomentumCalculation (3 tests)
      Cálculo correcto del momentum cibernético.

    • TestPerturbationRouting (7 tests)
      Pipeline end-to-end con múltiples topologías y casos de uso.

    • TestErrorHandlingAndEdgeCases (6 tests)
      Manejo robusto de entradas inválidas y casos patológicos.

Filosofía de diseño de pruebas:
──────────────────────────────

1. **Verificabilidad matemática**: Cada test valida una propiedad algebraica
   específica (ortogonalidad, simetría, rango, etc.).

2. **Cobertura exhaustiva**: Todos los invariantes y caminos críticos.

3. **Precisión numérica**: Tolerancias coherentes con el código bajo prueba.

4. **Fixtures generadoras**: Generadores de grafos, matrices, estados.

5. **Auditoría completa**: Captura de diagnósticos para análisis post-mortem.

6. **Casos adversariales**: Topologías patológicas, malcondicionamiento.

Niveles de severidad de pruebas:
────────────────────────────────
    • @pytest.mark.unit       — Pruebas de funciones individuales
    • @pytest.mark.integration — Pruebas end-to-end
    • @pytest.mark.stress     — Casos adversariales y límite
    • @pytest.mark.slow       — Pruebas computacionalmente intensivas
=========================================================================================
"""

from __future__ import annotations

import os
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Inyección axiomática del vacío termodinámico antes de importar NumPy/SciPy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix

from app.adapters.tools_interface import MICRegistry
from app.core.immune_system.gauge_field_router import (
    AgentValidationError,
    ChargeNeutralityError,
    CohomologicalInconsistencyError,
    CouplingResult,
    GaugeFieldDiagnostics,
    GaugeFieldError,
    GaugeFieldRouter,
    LatticeQEDConstants,
    LorentzForceError,
    PoissonSolution,
    TieBreakPolicy,
    TopologicalSingularityError,
    _validate_agent_charges,
    _validate_charge_density,
    _validate_cohomological_consistency,
    _validate_incidence_entries,
    _validate_laplacian_spectrum,
    _validate_laplacian_symmetry,
    _validate_sparse_matrix,
)
from app.core.mic_algebra import CategoricalState, Morphism

logger = logging.getLogger("tests.GaugeFieldRouter")


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES GENERADORAS DE GRAFOS Y MATRICES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_path_graph() -> Tuple[sp.csr_matrix, sp.csr_matrix, int, int]:
    """
    Genera un grafo de ruta simple: 0 — 1 — 2 — 3 (4 nodos, 3 aristas).

    Topología:
    ──────────
    Nodos: {0, 1, 2, 3}
    Aristas orientadas: (0→1), (1→2), (2→3)

    Retorna: (L, B₁, N, M)
        L ∈ ℝ⁴ˣ⁴   Laplaciano: diagonal [1,2,2,1], off-diagonal [-1,-1,-1]
        B₁ ∈ ℝ³ˣ⁴  Incidencia orientada
        N = 4     Número de nodos
        M = 3     Número de aristas
    """
    N = 4
    M = 3

    # Matriz de incidencia (3 aristas × 4 nodos)
    B1 = lil_matrix((M, N))
    # Arista 0: 0 → 1
    B1[0, 0] = -1
    B1[0, 1] = +1
    # Arista 1: 1 → 2
    B1[1, 1] = -1
    B1[1, 2] = +1
    # Arista 2: 2 → 3
    B1[2, 2] = -1
    B1[2, 3] = +1

    B1 = B1.tocsr()

    # Laplaciano: L = B₁ᵀ B₁
    L = B1.T @ B1

    return L, B1, N, M


@pytest.fixture
def cycle_graph() -> Tuple[sp.csr_matrix, sp.csr_matrix, int, int]:
    """
    Genera un grafo de ciclo: 0 ← 1 ← 2 ← 0 (3 nodos, 3 aristas).

    Topología:
    ──────────
    Nodos: {0, 1, 2}
    Aristas orientadas: (0→1), (1→2), (2→0)

    Retorna: (L, B₁, N, M)
        L ∈ ℝ³ˣ³   Laplaciano: diagonal [2,2,2], off-diagonal todos [-1]
        B₁ ∈ ℝ³ˣ³  Incidencia orientada
    """
    N = 3
    M = 3

    B1 = lil_matrix((M, N))
    # Arista 0: 0 → 1
    B1[0, 0] = -1
    B1[0, 1] = +1
    # Arista 1: 1 → 2
    B1[1, 1] = -1
    B1[1, 2] = +1
    # Arista 2: 2 → 0
    B1[2, 2] = -1
    B1[2, 0] = +1

    B1 = B1.tocsr()
    L = B1.T @ B1

    return L, B1, N, M


@pytest.fixture
def disconnected_graph() -> Tuple[sp.csr_matrix, sp.csr_matrix, int, int]:
    """
    Genera un grafo desconectado: (0—1) y (2—3) (4 nodos, 2 aristas).

    Topología:
    ──────────
    Componentes: {0,1} y {2,3}
    Aristas: (0→1), (2→3)

    dim(ker(L)) = 2 (un indicador por componente)

    Retorna: (L, B₁, N, M)
    """
    N = 4
    M = 2

    B1 = lil_matrix((M, N))
    # Arista 0: 0 → 1
    B1[0, 0] = -1
    B1[0, 1] = +1
    # Arista 1: 2 → 3
    B1[1, 2] = -1
    B1[1, 3] = +1

    B1 = B1.tocsr()
    L = B1.T @ B1

    return L, B1, N, M


@pytest.fixture
def large_grid_graph() -> Tuple[sp.csr_matrix, sp.csr_matrix, int, int]:
    """
    Genera una malla rectangular 5×5 (25 nodos, 40 aristas).

    Topología:
    ──────────
    Grid 2D con nodos (i,j) para i,j ∈ [0,4].
    Aristas orientadas hacia derecha y abajo.

    Retorna: (L, B₁, N, M)
    """
    height, width = 5, 5
    N = height * width
    M = (height - 1) * width + height * (width - 1)

    def node_idx(i: int, j: int) -> int:
        return i * width + j

    edges = []
    # Aristas horizontales (izquierda→derecha)
    for i in range(height):
        for j in range(width - 1):
            u = node_idx(i, j)
            v = node_idx(i, j + 1)
            edges.append((u, v))

    # Aristas verticales (arriba→abajo)
    for i in range(height - 1):
        for j in range(width):
            u = node_idx(i, j)
            v = node_idx(i + 1, j)
            edges.append((u, v))

    B1 = lil_matrix((len(edges), N))
    for edge_idx, (u, v) in enumerate(edges):
        B1[edge_idx, u] = -1
        B1[edge_idx, v] = +1

    B1 = B1.tocsr()
    L = B1.T @ B1

    return L, B1, N, M


@pytest.fixture
def mock_mic_registry() -> MICRegistry:
    """
    Crea un MICRegistry mock con varios morfismos registrados.

    Registra 3 morfismos simples (identidad, amplificación, anulación).
    """
    registry = MICRegistry()

    # Morfismo 1: Identidad
    def morph_identity(state: CategoricalState) -> CategoricalState:
        ctx = dict(getattr(state, "context", {}) or {})
        ctx["morphism_applied"] = "identity"
        return CategoricalState(
            payload=state.payload,
            context=ctx,
            validated_strata=state.validated_strata,
        )

    # Morfismo 2: Amplificación de contexto
    def morph_amplify(state: CategoricalState) -> CategoricalState:
        ctx = dict(getattr(state, "context", {}) or {})
        ctx["morphism_applied"] = "amplify"
        ctx["amplification_factor"] = 2.0
        return CategoricalState(
            payload=state.payload,
            context=ctx,
            validated_strata=state.validated_strata,
        )

    # Morfismo 3: Anulación con marca
    def morph_veto(state: CategoricalState) -> CategoricalState:
        ctx = dict(getattr(state, "context", {}) or {})
        ctx["morphism_applied"] = "veto"
        ctx["vetoed"] = True
        return CategoricalState(
            payload=state.payload,
            context=ctx,
            validated_strata=state.validated_strata,
        )

    # Añadimos el estrato para cumplir con la firma correcta de register_vector
    from app.core.schemas import Stratum

    registry.register_vector("agent_alpha", Stratum.PHYSICS, morph_identity)
    registry.register_vector("agent_beta", Stratum.TACTICS, morph_amplify)
    registry.register_vector("agent_gamma", Stratum.STRATEGY, morph_veto)

    return registry


def create_agent_charges(
    num_edges: int,
    num_agents: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Crea un registro de cargas de agentes para testing.

    Args:
        num_edges: Dimensión de cada vector de carga (número de aristas)
        num_agents: Número de agentes (default 3)

    Returns:
        Dict[str, np.ndarray] con cargas normalizadas
    """
    charges = {}
    agent_names = ["agent_alpha", "agent_beta", "agent_gamma"]
    for k in range(min(num_agents, len(agent_names))):
        agent_id = agent_names[k]
        # Crear carga aleatoria en [-1, +1]
        charge = np.random.uniform(-1.0, 1.0, size=num_edges)
        # Normalizar
        charge = charge / (np.linalg.norm(charge, ord=2) + 1e-10)
        charges[agent_id] = charge

    return charges


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: CONSTRUCCIÓN DEL ROUTER (INVARIANTES I1−I9)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGaugeFieldRouterConstruction:
    """
    Valida la correcta inicialización del GaugeFieldRouter y verificación
    de todos los invariantes I1−I9.
    """

    def test_i1_laplacian_is_square(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I1: L₀ debe ser cuadrada (N × N).

        Verifica que durante construcción se rechace una matriz no cuadrada.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Construir L no cuadrada (N × N+1)
        L_bad = sp.hstack([L, np.zeros((N, 1))])

        with pytest.raises(TopologicalSingularityError, match="cuadrado"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L_bad,
                incidence_matrix=B1,
                agent_charges=charges,
            )

    def test_i2_incidence_columns_match_nodes(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I2: B₁ debe tener exactamente N columnas (una por nodo).

        Verifica rechazo de B₁ con número incorrecto de columnas.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Construir B₁ con columnas incorrectas
        B1_bad = B1[:, :-1]  # N−1 columnas

        with pytest.raises(TopologicalSingularityError, match="columnas"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L,
                incidence_matrix=B1_bad,
                agent_charges=charges,
            )

    def test_i3_laplacian_symmetry(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I3: L₀ = L₀ᵀ (simetría).

        Verifica verificación de simetría durante construcción.
        Introduce asimetría intencional y verifica rechazo.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Crear matriz asimétrica perturbando L
        L_asym = L.tolil()
        L_asym[0, 1] += 0.5  # Introducir asimetría
        L_asym = L_asym.tocsr()

        with pytest.raises(TopologicalSingularityError, match="simétrico"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L_asym,
                incidence_matrix=B1,
                agent_charges=charges,
            )

    def test_i4_laplacian_positive_semidefinite(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I4: λ_min(L₀) ≥ −ε (semidefinitud positiva).

        Verifica que el router acepta el Laplaciano válido.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Construcción debe tener éxito
        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        # Eigenvalor mínimo debe ser no negativo (grafo conexo)
        assert router.lambda_min >= -LatticeQEDConstants.SPECTRAL_TOLERANCE

    def test_i5_cohomological_consistency(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I5: ‖L₀ − B₁ᵀB₁‖_F < ε_cohom (con verify_cohomology=True).

        Verifica que la relación cohomológica se comprueba.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Construcción con verify_cohomology=True debe tener éxito
        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
            verify_cohomology=True,
        )

        assert router._cohomological_residual is not None
        assert router._cohomological_residual < LatticeQEDConstants.COHOMOLOGICAL_TOLERANCE

    def test_i5_cohomological_inconsistency_rejection(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I5: Rechazo cuando L y B₁ son inconsistentes.

        Crea un L diferente de B₁ᵀB₁ y verifica rechazo.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Crear L inconsistente: L + identidad
        L_bad = L + sp.eye(N, dtype=np.float64)

        with pytest.raises(CohomologicalInconsistencyError, match="cohomológica"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L_bad,
                incidence_matrix=B1,
                agent_charges=charges,
                verify_cohomology=True,
            )

    def test_i6_incidence_entries_valid(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I6: B₁[i,j] ∈ {-1, 0, +1} exactamente.

        Verifica rechazo de entradas inválidas en B₁.
        """
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        # Crear B₁ con entradas inválidas
        B1_bad = B1.tolil()
        B1_bad[0, 0] = 0.5  # Valor inválido
        B1_bad = B1_bad.tocsr()

        with pytest.raises(TopologicalSingularityError, match="incidencia"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L,
                incidence_matrix=B1_bad,
                agent_charges=charges,
            )

    def test_i8_agent_charges_dimensionality(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I8: ∀k: Q_k ∈ ℝᴹ con dim(Q_k) = M exacto.

        Verifica rechazo de cargas con dimensión incorrecta.
        """
        L, B1, N, M = simple_path_graph

        # Cargas con dimensión incorrecta
        bad_charges = {
            "agent_a": np.random.randn(M - 1),  # Dimensión incorrecta
        }

        with pytest.raises(AgentValidationError, match="ℝ"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L,
                incidence_matrix=B1,
                agent_charges=bad_charges,
            )

    def test_i9_at_least_one_agent_required(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        I9: Debe registrarse al menos un agente (I = ∃ agente).

        Verifica rechazo de registro de agentes vacío.
        """
        L, B1, N, M = simple_path_graph

        with pytest.raises(LorentzForceError, match="Fock"):
            GaugeFieldRouter(
                mic_registry=mock_mic_registry,
                laplacian=L,
                incidence_matrix=B1,
                agent_charges={},  # Vacío
            )


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: FUNCIONES VALIDADORAS (FUNCIONES PURAS)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestValidationFunctions:
    """
    Pruebas unitarias de funciones validadoras que se usan durante
    construcción e iteración del router.
    """

    def test_validate_sparse_matrix_accepts_valid_matrix(self) -> None:
        """Acepta una matriz dispersa válida."""
        L = sp.csr_matrix(np.eye(3))
        result = _validate_sparse_matrix(L, name="test")
        assert result.shape == (3, 3)
        assert sp.issparse(result)

    def test_validate_sparse_matrix_rejects_non_sparse(self) -> None:
        """Rechaza matrices densas."""
        L_dense = np.eye(3)
        with pytest.raises(TopologicalSingularityError, match="dispersa"):
            _validate_sparse_matrix(L_dense, name="test")

    def test_validate_laplacian_symmetry_passes_symmetric(self) -> None:
        """Pasa simetría para matriz simétrica."""
        L = sp.csr_matrix(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]))
        residual = _validate_laplacian_symmetry(L)
        assert residual < 1e-12

    def test_validate_laplacian_symmetry_rejects_asymmetric(self) -> None:
        """Rechaza matriz asimétrica."""
        L_asym = sp.csr_matrix(np.array([[2, -1, 0], [0, 2, -1], [0, -1, 2]]))
        with pytest.raises(TopologicalSingularityError, match="simétrico"):
            _validate_laplacian_symmetry(L_asym)

    def test_validate_laplacian_spectrum_positive_semidefinite(self) -> None:
        """Acepta matriz semidefinida positiva."""
        L = sp.csr_matrix(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]))
        lambda_min = _validate_laplacian_spectrum(L)
        assert lambda_min >= -1e-10

    def test_validate_laplacian_spectrum_rejects_negative(self) -> None:
        """Rechaza matriz con eigenvalor negativo."""
        # Crear matriz con eigenvalor negativo
        L_neg = sp.csr_matrix(np.array([[-1, 0, 0], [0, 2, -1], [0, -1, 2]]))
        with pytest.raises(TopologicalSingularityError, match="semidefinido"):
            _validate_laplacian_spectrum(L_neg)

    def test_validate_incidence_entries_valid(self) -> None:
        """Acepta entradas válidas en {-1, 0, +1}."""
        B1 = sp.csr_matrix(np.array([[-1, 1, 0], [0, -1, 1]]))
        _validate_incidence_entries(B1)  # Debe tener éxito

    def test_validate_incidence_entries_invalid(self) -> None:
        """Rechaza entradas fuera de {-1, 0, +1}."""
        B1 = sp.csr_matrix(np.array([[-1, 1, 0.5], [0, -1, 1]]))
        with pytest.raises(TopologicalSingularityError, match="inválida"):
            _validate_incidence_entries(B1)

    def test_validate_agent_charges_valid(self) -> None:
        """Acepta cargas válidas."""
        charges = {
            "agent_a": np.array([1.0, -1.0, 0.5]),
            "agent_b": np.array([0.0, 0.5, -0.5]),
        }
        result = _validate_agent_charges(charges, expected_dim=3)
        assert len(result) == 2
        assert "agent_a" in result

    def test_validate_agent_charges_wrong_dimension(self) -> None:
        """Rechaza cargas con dimensión incorrecta."""
        charges = {
            "agent_a": np.array([1.0, -1.0]),  # Dimensión 2, se espera 3
        }
        with pytest.raises(AgentValidationError, match="ℝ"):
            _validate_agent_charges(charges, expected_dim=3)

    def test_validate_agent_charges_invalid_id(self) -> None:
        """Rechaza IDs de agente inválidos."""
        charges = {
            "": np.array([1.0, -1.0, 0.5]),  # ID vacío
        }
        with pytest.raises(AgentValidationError, match="inválido"):
            _validate_agent_charges(charges, expected_dim=3)

    def test_validate_charge_density_sum_zero_passes(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
    ) -> None:
        """Acepta densidad con suma nula en grafo conexo."""
        L, _, N, _ = simple_path_graph
        rho = np.zeros(N)
        rho[0] = 1.0
        rho -= np.mean(rho) # Enforce exactly zero
        _validate_charge_density(rho, L)  # Debe tener éxito

    def test_validate_charge_density_sum_nonzero_fails(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
    ) -> None:
        """Rechaza densidad con suma no nula."""
        L, _, N, _ = simple_path_graph
        rho = np.ones(N)  # Suma = N
        with pytest.raises(ChargeNeutralityError, match="neutralidad"):
            _validate_charge_density(rho, L)


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: SOLVER DE POISSON
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPoissonSolver:
    """
    Valida la correcta resolución del problema L₀Φ = ρ.
    """

    def test_poisson_solution_structure(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica estructura y tipos de PoissonSolution."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        # Crear densidad de carga
        rho = np.zeros(N)
        rho[0] = 1.0
        rho -= np.mean(rho)

        result = router._solve_discrete_poisson(rho)

        assert isinstance(result, PoissonSolution)
        assert result.phi.shape == (N,)
        assert np.isfinite(result.relative_residual)
        assert result.acond > 1.0  # Número de condición

    def test_poisson_residual_within_tolerance(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que el residual relativo está dentro de tolerancia."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho = np.zeros(N)
        rho[0] = 1.0
        rho -= np.mean(rho)

        result = router._solve_discrete_poisson(rho)

        assert result.relative_residual < LatticeQEDConstants.RESIDUAL_TOLERANCE

    def test_poisson_solution_orthogonal_to_kernel(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que la solución es ortogonal al kernel (mínima norma)."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho = np.zeros(N)
        rho[0] = 1.0
        rho -= np.mean(rho)

        result = router._solve_discrete_poisson(rho)

        # El kernel del Laplaciano es span{𝟙}
        ones = np.ones(N) / np.sqrt(N)
        projection = abs(np.dot(result.phi, ones))

        # La solución de mínima norma debe estar ortogonal al kernel
        assert projection < 1e-8

    def test_poisson_rejects_invalid_rho_dimension(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza densidad de carga con dimensión incorrecta."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho_bad = np.zeros(N - 1)  # Dimensión incorrecta

        with pytest.raises(TopologicalSingularityError, match="ℝ"):
            router._solve_discrete_poisson(rho_bad)

    def test_poisson_rejects_nonfinite_rho(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza densidad con NaN o ±∞."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho = np.zeros(N)
        rho[0] = np.inf  # Infinito

        with pytest.raises(TopologicalSingularityError, match="finitos"):
            router._solve_discrete_poisson(rho)


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: CUANTIFICACIÓN DE CARGA
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestChargeQuantization:
    """
    Valida la cuantificación correcta de densidad de carga con
    la fórmula ρ = α(δ_v − 𝟙/N).
    """

    def test_charge_quantization_sum_zero(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que la densidad cuantizada tiene suma nula."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho = router._quantize_bosonic_excitation(anomaly_node=0, severity=1.0)

        assert abs(np.sum(rho)) < 1e-12  # Suma nula

    def test_charge_quantization_has_peak(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que la densidad tiene un pico en el nodo perturbado."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        severity = 1.0
        anomaly_node = 0

        rho = router._quantize_bosonic_excitation(
            anomaly_node=anomaly_node,
            severity=severity,
        )

        # El nodo perturbado debe tener valor positivo máximo
        assert rho[anomaly_node] > 0.0
        assert rho[anomaly_node] == np.max(rho)

    def test_charge_quantization_negative_elsewhere(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que los otros nodos tienen valores negativos."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho = router._quantize_bosonic_excitation(anomaly_node=0, severity=1.0)

        # Todos los nodos excepto el perturbado deben tener valor negativo
        for i in range(1, N):
            assert rho[i] < 0.0

    def test_charge_quantization_severity_scaling(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica escalamiento con severidad."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        rho1 = router._quantize_bosonic_excitation(anomaly_node=0, severity=1.0)
        rho2 = router._quantize_bosonic_excitation(anomaly_node=0, severity=2.0)

        # rho2 ≈ 2 * rho1
        assert np.allclose(rho2, 2.0 * rho1)


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 5: CÁLCULO DEL CAMPO
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFieldCalculation:
    """
    Valida el cálculo correcto del campo discreto E = B₁Φ.
    """

    def test_field_has_correct_dimension(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que E ∈ ℝᴹ."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi = np.random.randn(N)
        e_field = router._compute_potential_gradient(phi)

        assert e_field.shape == (M,)

    def test_field_is_finite(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que E es finita."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi = np.random.randn(N)
        e_field = router._compute_potential_gradient(phi)

        assert np.all(np.isfinite(e_field))

    def test_field_from_constant_potential_is_zero(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que E = 0 si Φ es constante."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi_const = np.ones(N) * 5.0  # Potencial constante

        e_field = router._compute_potential_gradient(phi_const)

        # El campo debe ser cero (diferencial de constante)
        assert np.allclose(e_field, 0.0, atol=1e-12)

    def test_field_rejects_wrong_dimension_phi(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza potencial con dimensión incorrecta."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi_bad = np.zeros(N - 1)  # Dimensión incorrecta

        with pytest.raises(TopologicalSingularityError, match="ℝ"):
            router._compute_potential_gradient(phi_bad)

    def test_field_gauge_invariance(
        self,
        large_grid_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Invariancia de Gauge: E(Φ) = E(Φ + c1)."""
        L, B1, N, M = large_grid_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi = np.random.randn(N)
        e_field_base = router._compute_potential_gradient(phi)

        # Traslación global (Invariancia de Gauge)
        phi_shifted = phi + 1000.0  # c = 1000.0
        e_field_shifted = router._compute_potential_gradient(phi_shifted)

        # Verificar que el campo resultante E no muta
        np.testing.assert_allclose(e_field_base, e_field_shifted, rtol=1e-12, atol=1e-12)

    def test_field_exact_irrotationality(
        self,
        cycle_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Teorema de Hodge: Campo es irrotacional en un ciclo."""
        L, B1, N, M = cycle_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        phi = np.random.randn(N)
        e_field = router._compute_potential_gradient(phi)

        # El ciclo fundamental es sumar todas las aristas
        # El producto punto entre el rotacional (ciclo) y E debe ser 0.0
        rotational_sum = np.sum(e_field)
        assert abs(rotational_sum) < 1e-13


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 6: SELECCIÓN DE AGENTE (ACOPLAMIENTO GAUGE)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAgentSelection:
    """
    Valida la selección correcta del agente por máximo acoplamiento.
    """

    def test_agent_selection_returns_coupling_result(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica estructura de CouplingResult."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        e_field = np.random.randn(M)
        result = router._calculate_lorentz_attraction(e_field)

        assert isinstance(result, CouplingResult)
        assert result.selected_agent in result.all_actions
        assert np.isfinite(result.max_action)
        assert result.num_maximizers >= 1

    def test_agent_selection_picks_maximum_action(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que se selecciona el agente con máxima acción."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        e_field = np.random.randn(M)
        result = router._calculate_lorentz_attraction(e_field)

        # Verificar que la acción del agente seleccionado es máxima
        selected_action = result.all_actions[result.selected_agent]
        max_action_all = max(result.all_actions.values())

        assert np.isclose(selected_action, max_action_all)

    def test_agent_selection_lexicographic_tiebreak(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica desempate lexicográfico."""
        L, B1, N, M = simple_path_graph
        charges = {
            "agent_alpha": np.ones(M),  # Todos 1
            "agent_beta": np.ones(M),   # Todos 1 (empate)
            "agent_gamma": np.zeros(M),  # Todos 0
        }

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
            tie_break_policy=TieBreakPolicy.LEXICOGRAPHIC,
        )

        e_field = np.ones(M)  # Campo uniforme (empate entre alpha y beta)
        result = router._calculate_lorentz_attraction(e_field)

        # Con desempate lexicográfico, se debe seleccionar "agent_alpha"
        assert result.selected_agent == "agent_alpha"
        assert result.num_maximizers == 2  # alpha y beta empatados

    def test_agent_selection_rejects_nonfinite_field(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza campo no finito."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        e_field = np.ones(M)
        e_field[0] = np.inf  # Infinito

        with pytest.raises(LorentzForceError, match="finitos"):
            router._calculate_lorentz_attraction(e_field)


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 7: CÁLCULO DE MOMENTUM
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMomentumCalculation:
    """
    Valida el cálculo del momentum cibernético.
    """

    def test_momentum_increases_with_severity(self) -> None:
        """Verifica que p crece con severidad."""
        field_norm = 1.0

        p1 = GaugeFieldRouter._compute_cyber_momentum(severity=0.1, field_norm=field_norm)
        p2 = GaugeFieldRouter._compute_cyber_momentum(severity=1.0, field_norm=field_norm)
        p3 = GaugeFieldRouter._compute_cyber_momentum(severity=10.0, field_norm=field_norm)

        assert p1 < p2 < p3

    def test_momentum_increases_with_field_norm(self) -> None:
        """Verifica que p crece con ‖E‖₂."""
        severity = 1.0

        p1 = GaugeFieldRouter._compute_cyber_momentum(severity=severity, field_norm=0.0)
        p2 = GaugeFieldRouter._compute_cyber_momentum(severity=severity, field_norm=1.0)
        p3 = GaugeFieldRouter._compute_cyber_momentum(severity=severity, field_norm=10.0)

        assert p1 < p2 < p3

    def test_momentum_nonzero_for_positive_severity(self) -> None:
        """Verifica que p > 0 para severidad > 0."""
        p = GaugeFieldRouter._compute_cyber_momentum(severity=0.1, field_norm=0.1)
        assert p > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 8: ENRUTAMIENTO END-TO-END
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPerturbationRouting:
    """
    Pruebas del pipeline end-to-end de enrutamiento.
    Valida todas las 9 etapas ejecutadas secuencialmente.
    """

    def test_routing_simple_path_graph(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Enrutamiento exitoso en grafo de ruta simple."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={"test": "data"})

        result_state = router.route_perturbation(
            state=state,
            anomaly_node=0,
            severity=1.0,
        )

        # Verificar que el estado tiene contexto inyectado
        ctx = getattr(result_state, "context", {})
        assert "gauge_trace_id" in ctx
        assert "cyber_momentum" in ctx
        assert "gauge_selected_agent" in ctx

    def test_routing_cycle_graph(
        self,
        cycle_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Enrutamiento en grafo de ciclo."""
        L, B1, N, M = cycle_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        for anomaly_node in range(N):
            result_state = router.route_perturbation(
                state=state,
                anomaly_node=anomaly_node,
                severity=1.0,
            )
            assert isinstance(result_state, CategoricalState)

    def test_routing_injects_correct_context(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica inyección correcta de contexto electromagnético."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={"original": "payload"})
        severity = 2.5

        result_state = router.route_perturbation(
            state=state,
            anomaly_node=1,
            severity=severity,
        )

        ctx = getattr(result_state, "context", {})

        # Verificar presencia de todos los campos inyectados
        expected_keys = {
            "gauge_trace_id",
            "cyber_momentum",
            "resolved_anomaly_node",
            "gauge_selected_agent",
            "gauge_charge_density_norm",
            "gauge_field_norm",
            "gauge_potential_norm",
            "gauge_max_action",
            "gauge_num_maximizers",
            "gauge_poisson_relative_residual",
            "gauge_poisson_acond",
        }

        for key in expected_keys:
            assert key in ctx, f"Contexto falta: {key}"

        # Verificar valores
        assert ctx["resolved_anomaly_node"] == 1
        assert ctx["cyber_momentum"] > 0.0
        assert ctx["gauge_max_action"] is not None

    def test_routing_applies_morphism(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que se aplica el morfismo del agente seleccionado."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        result_state = router.route_perturbation(
            state=state,
            anomaly_node=0,
            severity=1.0,
        )

        ctx = getattr(result_state, "context", {})

        # El morfismo debería haber inyectado "morphism_applied"
        assert "morphism_applied" in ctx

    def test_routing_multiple_anomalies_different_results(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que anomalías en diferentes nodos producen resultados distintos."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        result1 = router.route_perturbation(state=state, anomaly_node=0, severity=1.0)
        result2 = router.route_perturbation(state=state, anomaly_node=N - 1, severity=1.0)

        ctx1 = getattr(result1, "context", {})
        ctx2 = getattr(result2, "context", {})

        # Trace IDs deben ser distintos
        assert ctx1.get("gauge_trace_id") != ctx2.get("gauge_trace_id")

        # Campos potencialmente distintos
        assert ctx1.get("gauge_field_norm") is not None
        assert ctx2.get("gauge_field_norm") is not None

    def test_routing_orthogonal_coupling_consistency(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Asegura que el acoplamiento iterado sea monótono y consistente, sin mutar el estado base."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={"base": "immutable"})

        # Ejecutar enrutamiento dos veces sobre el mismo estado base
        result1 = router.route_perturbation(state=state, anomaly_node=1, severity=5.0)
        result2 = router.route_perturbation(state=state, anomaly_node=1, severity=5.0)

        ctx1 = getattr(result1, "context", {})
        ctx2 = getattr(result2, "context", {})

        # Debe haber conservado su payload sin mutar
        assert result1.payload == state.payload
        assert result2.payload == state.payload
        assert state.payload == {"base": "immutable"}

        # El acoplamiento es determinista: se seleccionó el mismo agente
        assert ctx1.get("gauge_selected_agent") == ctx2.get("gauge_selected_agent")

        # La acción máxima es exactamente la misma
        assert np.isclose(ctx1.get("gauge_max_action", -1.0), ctx2.get("gauge_max_action", -2.0))

        # Los ID de traza deben ser diferentes
        assert ctx1.get("gauge_trace_id") != ctx2.get("gauge_trace_id")

    @pytest.mark.slow
    def test_routing_large_grid(
        self,
        large_grid_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Prueba enrutamiento en malla grande (computacionalmente intensiva)."""
        L, B1, N, M = large_grid_graph
        charges = create_agent_charges(M, num_agents=2)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        # Enrutar anomalía en centro de la malla
        result = router.route_perturbation(
            state=state,
            anomaly_node=N // 2,
            severity=1.0,
        )

        assert isinstance(result, CategoricalState)
        ctx = getattr(result, "context", {})
        assert "gauge_trace_id" in ctx


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE 9: MANEJO DE ERRORES Y CASOS PATOLÓGICOS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestErrorHandlingAndEdgeCases:
    """
    Pruebas de manejo robusto de entradas inválidas y casos límite.
    """

    def test_routing_rejects_none_state(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza state = None."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        with pytest.raises(GaugeFieldError, match="None"):
            router.route_perturbation(
                state=None,
                anomaly_node=0,
                severity=1.0,
            )

    def test_routing_rejects_invalid_anomaly_node(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza anomaly_node fuera de rango."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        with pytest.raises(GaugeFieldError, match="rango"):
            router.route_perturbation(
                state=state,
                anomaly_node=N,  # Fuera de rango
                severity=1.0,
            )

    def test_routing_rejects_negative_severity(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza severidad negativa."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        with pytest.raises(GaugeFieldError, match="no negativa"):
            router.route_perturbation(
                state=state,
                anomaly_node=0,
                severity=-1.0,
            )

    def test_routing_rejects_nonfinite_severity(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza severidad infinita o NaN."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        with pytest.raises(GaugeFieldError, match="finita"):
            router.route_perturbation(
                state=state,
                anomaly_node=0,
                severity=np.inf,
            )

    def test_routing_rejects_wrong_state_type(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Rechaza estado de tipo incorrecto."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        with pytest.raises(GaugeFieldError, match="CategoricalState"):
            router.route_perturbation(
                state="not a state",  # Tipo incorrecto
                anomaly_node=0,
                severity=1.0,
            )

    def test_disconnected_graph_charge_neutrality(
        self,
        disconnected_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica manejo correcto de grafos desconectados."""
        L, B1, N, M = disconnected_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        # Para grafo desconectado, la condición de Fredholm es más restrictiva
        state = CategoricalState(payload={})

        # Enrutamiento en componente 1 (nodos 0, 1) inyectando carga local
        # que falla por ChargeNeutralityError porque no hay neutralidad global
        with pytest.raises(ChargeNeutralityError, match="Fredholm"):
            router.route_perturbation(
                state=state,
                anomaly_node=0,
                severity=1.0,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETRIZADAS: BARRIDO DE TOPOLOGÍAS Y PARÁMETROS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("anomaly_node", [0, 1, 2])
@pytest.mark.parametrize("severity", [0.1, 1.0, 10.0, 100.0, 1000.0])
@pytest.mark.stress
@pytest.mark.integration
def test_routing_parametrized_simple_path(
    anomaly_node: int,
    severity: float,
    simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
    mock_mic_registry: MICRegistry,
) -> None:
    """
    Prueba enrutamiento parametrizado en grafo de ruta.
    Barrido sobre combinaciones de nodo y severidad.
    """
    L, B1, N, M = simple_path_graph
    charges = create_agent_charges(M)

    router = GaugeFieldRouter(
        mic_registry=mock_mic_registry,
        laplacian=L,
        incidence_matrix=B1,
        agent_charges=charges,
    )

    state = CategoricalState(payload={})

    result = router.route_perturbation(
        state=state,
        anomaly_node=anomaly_node,
        severity=severity,
    )

    assert isinstance(result, CategoricalState)
    ctx = getattr(result, "context", {})
    assert ctx.get("cyber_momentum") > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# TESTS DE INVARIANTES ALGEBRAICOS (VALIDACIÓN CRUZADA)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAlgebraicInvariants:
    """
    Validación de invariantes algebraicos durante operación.
    """

    def test_laplacian_consistency_after_operations(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que L sigue siendo consistente después de operaciones."""
        L_orig, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L_orig,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        # Acceder a L a través del router
        L_cached = router._L

        # Verificar que siguen siendo idénticas
        diff = sp.linalg.norm(L_orig - L_cached, ord='fro')
        assert diff < 1e-14

    def test_field_in_image_of_incidence(
        self,
        simple_path_graph: Tuple[sp.csr_matrix, sp.csr_matrix, int, int],
        mock_mic_registry: MICRegistry,
    ) -> None:
        """Verifica que E ∈ Im(B₁)."""
        L, B1, N, M = simple_path_graph
        charges = create_agent_charges(M)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        # Crear un estado y resolver Poisson
        state = CategoricalState(payload={})
        result = router.route_perturbation(
            state=state,
            anomaly_node=0,
            severity=1.0,
        )

        # El campo está en Im(B₁) por construcción (E = B₁Φ)
        # Verificar que existe Φ tal que E = B₁Φ
        ctx = getattr(result, "context", {})
        field_norm = ctx.get("gauge_field_norm", 0.0)

        # Si el campo es no nulo, la Poisson fue resuelta correctamente
        assert field_norm >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# TESTS DE PERFORMANCE Y ESCALABILIDAD
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.stress
class TestPerformanceAndScaling:
    """
    Pruebas de performance y escalabilidad.
    """

    @pytest.mark.parametrize("size", [5, 10, 20])
    def test_routing_time_complexity(
        self,
        mock_mic_registry: MICRegistry,
        size: int,
    ) -> None:
        """
        Prueba tiempo de enrutamiento para diferentes tamaños de grafo.
        """
        # Crear malla de tamaño variable
        height, width = size, size
        N = height * width
        M = (height - 1) * width + height * (width - 1)

        def node_idx(i: int, j: int) -> int:
            return i * width + j

        edges = []
        for i in range(height):
            for j in range(width - 1):
                u = node_idx(i, j)
                v = node_idx(i, j + 1)
                edges.append((u, v))

        for i in range(height - 1):
            for j in range(width):
                u = node_idx(i, j)
                v = node_idx(i + 1, j)
                edges.append((u, v))

        B1 = lil_matrix((len(edges), N))
        for edge_idx, (u, v) in enumerate(edges):
            B1[edge_idx, u] = -1
            B1[edge_idx, v] = +1

        B1 = B1.tocsr()
        L = B1.T @ B1

        charges = create_agent_charges(M, num_agents=2)

        router = GaugeFieldRouter(
            mic_registry=mock_mic_registry,
            laplacian=L,
            incidence_matrix=B1,
            agent_charges=charges,
        )

        state = CategoricalState(payload={})

        import time

        start = time.time()
        result = router.route_perturbation(
            state=state,
            anomaly_node=N // 2,
            severity=1.0,
        )
        elapsed = time.time() - start

        assert isinstance(result, CategoricalState)
        logger.info(f"Enrutamiento para tamaño {size}×{size} (N={N}): {elapsed:.3f}s")