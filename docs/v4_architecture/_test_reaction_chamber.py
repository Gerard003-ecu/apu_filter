"""
Módulo: test_reaction_chamber.py
Suite de pruebas para el Reactor Catalítico Hexagonal (versión 4.2)
Verifica la coherencia matemática, termodinámica y topológica del sistema.

Fundamentos Teóricos:
- Topología: Grafo cíclico C₆ con grupo de simetría D₆
- Termodinámica: Energía libre de Gibbs G = H - TS
- Mecánica Cuántica: Espacio de Hilbert ℂ⁶ con proyección a S⁵
- Química Orgánica: Regla de Hückel (4n+2 electrones π para aromaticidad)
"""
import math
import logging
import pytest
from collections import Counter
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

from reaction_chamber import (
    CarbonNode,
    HilbertState,
    ThermodynamicPotential,
    HexagonalTopology,
    CatalyticReactor,
    CatalystAgent,
    R_GAS_CONSTANT,
    BOLTZMANN_SCALE,
    GIBBS_CONVERGENCE_EPS,
    DAMPING_GAMMA,
    DAMPING_OMEGA,
    ENTROPY_MIN_PROB,
    CFL_STABILITY_FACTOR
)

# =============================================================================
# CONSTANTES DE PRUEBA
# =============================================================================

# Tolerancias numéricas para comparaciones
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12

# Parámetros del grafo cíclico C₆
HEXAGON_VERTICES = 6
HEXAGON_DEGREE = 2  # Cada vértice tiene exactamente 2 vecinos

# Eigenvalores de la matriz Laplaciana de C₆: λₖ = 2 - 2cos(2πk/6), k = 0,...,5
# λ₀ = 0, λ₁ = λ₅ = 1, λ₂ = λ₄ = 3, λ₃ = 4
LAPLACIAN_EIGENVALUES_C6 = [0.0, 1.0, 3.0, 4.0, 3.0, 1.0]
SPECTRAL_GAP_C6 = 1.0  # λ₁ = min(λₖ para k > 0)


# =============================================================================
# FUNCIONES AUXILIARES DE PRUEBA
# =============================================================================

def floats_close(a: float, b: float, rel_tol: float = FLOAT_REL_TOL, 
                 abs_tol: float = FLOAT_ABS_TOL) -> bool:
    """Compara dos floats con tolerancia numérica."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def vectors_close(v1: List[float], v2: List[float], 
                  rel_tol: float = FLOAT_REL_TOL,
                  abs_tol: float = FLOAT_ABS_TOL) -> bool:
    """Compara dos vectores elemento a elemento con tolerancia."""
    if len(v1) != len(v2):
        return False
    return all(floats_close(a, b, rel_tol, abs_tol) for a, b in zip(v1, v2))


def calculate_vector_norm(vector: List[float]) -> float:
    """Calcula la norma euclidiana de un vector."""
    return math.sqrt(sum(x**2 for x in vector))


def calculate_inner_product(v1: List[float], v2: List[float]) -> float:
    """Calcula el producto interno estándar de dos vectores reales."""
    return sum(a * b for a, b in zip(v1, v2))


# =============================================================================
# FIXTURES DE PRUEBA
# =============================================================================

@pytest.fixture
def mock_mic():
    """
    Mock del Microservicio de Intención (MICRegistry).
    
    Simula el registro de servicios para cada nodo de carbono.
    """
    mic = MagicMock()
    mic.resolve.return_value = MagicMock(execute=MagicMock(return_value={}))
    return mic


@pytest.fixture
def mock_telemetry():
    """
    Mock del sistema de telemetría.
    
    Registra llamadas para verificar que los eventos se emitan correctamente.
    """
    telemetry = MagicMock()
    telemetry.record_reaction_success = MagicMock()
    telemetry.record_error = MagicMock()
    telemetry.record_cycle_metrics = MagicMock()
    return telemetry


@pytest.fixture
def mock_catalyst():
    """
    Mock del Agente Catalizador que cumple con el protocolo CatalystAgent.
    
    Propiedades:
    - efficiency_factor: 0.3 (reduce la energía de activación en 30%)
    - catalytic_strength: 0.7 (fuerza de orientación catalítica)
    """
    class MockCatalyst:
        @property
        def efficiency_factor(self) -> float:
            return 0.3
        
        @property
        def catalytic_strength(self) -> float:
            return 0.7
        
        def orient(self, context: Dict, gradient: Dict) -> Dict:
            """Simula una orientación catalítica que añade un ajuste al contexto."""
            return {"catalyst_adjustment": 0.1}
    
    return MockCatalyst()


@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """Contexto inicial vacío."""
    return {}


@pytest.fixture
def minimal_context() -> Dict[str, Any]:
    """
    Contexto con los datos mínimos necesarios para activar todos los nodos.
    
    Precursores requeridos:
    - C1: data_loaded
    - C2: physical_constraints
    - C3: (ninguno específico)
    - C4: financial_params
    - C5: semantic_model
    - C6: (resultado de los anteriores)
    """
    return {
        "physical_constraints": "valid",
        "financial_params": "valid",
        "semantic_model": "valid",
        "data_loaded": True
    }


@pytest.fixture
def full_context(minimal_context) -> Dict[str, Any]:
    """
    Contexto con todos los nodos en estado resonante (aromático).
    
    Representa un sistema que ha completado el ciclo de reacción
    con los 6 electrones π necesarios para aromaticidad (regla de Hückel: 4n+2, n=1).
    """
    # Crear copia profunda para evitar efectos secundarios
    context = dict(minimal_context)
    for node in CarbonNode:
        context[f"{node.name}_status"] = "resonant"
    return context


@pytest.fixture
def topology() -> HexagonalTopology:
    """
    Topología hexagonal para pruebas.
    
    Representa el grafo cíclico C₆ con:
    - 6 vértices (nodos de carbono)
    - 6 aristas (enlaces σ)
    - Grupo de simetría D₆ (dihédrico)
    """
    return HexagonalTopology()


@pytest.fixture
def hilbert_state() -> HilbertState:
    """
    Estado de Hilbert inicial para pruebas.
    
    Vector: [1.0, 0.5, 0.0, -0.5, -1.0, 0.0]
    Norma: √(1 + 0.25 + 0 + 0.25 + 1 + 0) = √2.5 ≈ 1.581
    """
    return HilbertState(vector=[1.0, 0.5, 0.0, -0.5, -1.0, 0.0])


@pytest.fixture
def normalized_hilbert_state(hilbert_state) -> HilbertState:
    """Estado de Hilbert normalizado (proyectado a S⁵)."""
    return hilbert_state.normalize()


@pytest.fixture
def thermodynamic_potential() -> ThermodynamicPotential:
    """
    Potencial termodinámico inicial para pruebas.
    
    Parámetros:
    - H (entalpía): 100.0 kJ/mol
    - S (entropía): 0.5 kJ/(mol·K)
    - T₀ (temperatura base): 298.0 K
    - γ (acoplamiento térmico): 15.0 K
    - σ (estrés topológico): 0.8
    
    Temperatura efectiva: T = T₀ + γ·σ² = 298 + 15·0.64 = 307.6 K
    Gibbs: G = H - T·S·κ
    """
    return ThermodynamicPotential(
        enthalpy=100.0,
        entropy=0.5,
        base_temperature=298.0,
        temperature_coupling=15.0,
        topological_stress=0.8
    )


@pytest.fixture
def reactor(mock_mic, mock_catalyst, mock_telemetry) -> CatalyticReactor:
    """Reactor catalítico configurado para pruebas."""
    return CatalyticReactor(mock_mic, mock_catalyst, mock_telemetry)


# =============================================================================
# PRUEBAS DE ENUMERACIONES (CarbonNode)
# =============================================================================

class TestCarbonNode:
    """Pruebas para la enumeración CarbonNode."""
    
    def test_node_count_satisfies_huckel_rule(self):
        """
        Verifica que el número de nodos satisfaga la regla de Hückel.
        
        Para aromaticidad: 4n + 2 electrones π, donde n = 1 → 6 electrones.
        """
        assert len(CarbonNode) == 6
    
    def test_indices_are_sequential(self):
        """Verifica que los índices sean secuenciales desde 0."""
        expected_indices = list(range(6))
        actual_indices = [node.index for node in CarbonNode]
        assert actual_indices == expected_indices
    
    def test_indices_match_node_position(self):
        """Verifica que cada índice corresponda a la posición correcta."""
        index_map = {
            CarbonNode.C1_INGESTION: 0,
            CarbonNode.C2_PHYSICS: 1,
            CarbonNode.C3_TOPOLOGY: 2,
            CarbonNode.C4_STRATEGY: 3,
            CarbonNode.C5_SEMANTICS: 4,
            CarbonNode.C6_MATTER: 5
        }
        for node, expected_index in index_map.items():
            assert node.index == expected_index, f"{node.name} debería tener índice {expected_index}"
    
    def test_labels_are_human_readable(self):
        """Verifica que las etiquetas sean legibles."""
        for node in CarbonNode:
            assert isinstance(node.label, str)
            assert len(node.label) > 0
            assert node.label.startswith("C")
    
    def test_service_names_are_valid_identifiers(self):
        """Verifica que los nombres de servicio sean identificadores válidos."""
        for node in CarbonNode:
            service_name = node.service_name
            assert isinstance(service_name, str)
            assert service_name.isidentifier() or "_" in service_name
    
    def test_specific_labels(self):
        """Verifica etiquetas específicas."""
        assert CarbonNode.C1_INGESTION.label == "C1 Ingestion"
        assert CarbonNode.C2_PHYSICS.label == "C2 Physics"
        assert CarbonNode.C6_MATTER.label == "C6 Matter"
    
    def test_specific_service_names(self):
        """Verifica nombres de servicio específicos."""
        service_map = {
            CarbonNode.C1_INGESTION: "load_data",
            CarbonNode.C2_PHYSICS: "stabilize_flux",
            CarbonNode.C6_MATTER: "materialization"
        }
        for node, expected_service in service_map.items():
            assert node.service_name == expected_service


# =============================================================================
# PRUEBAS DE ESPACIO DE HILBERT
# =============================================================================

class TestHilbertState:
    """
    Pruebas para el estado de Hilbert.
    
    El espacio de estados es ℂ⁶ (o ℝ⁶ en la implementación simplificada),
    con la esfera S⁵ como variedad de estados normalizados.
    """
    
    def test_initialization_with_correct_dimension(self, hilbert_state):
        """Verifica que el estado tenga la dimensión correcta (6 para C₆)."""
        assert len(hilbert_state.vector) == HEXAGON_VERTICES
    
    def test_initial_phase_is_zero(self, hilbert_state):
        """Verifica que la fase inicial sea cero."""
        assert hilbert_state.phase == 0.0
    
    def test_norm_calculation(self, hilbert_state):
        """
        Verifica el cálculo correcto de la norma euclidiana.
        
        ‖ψ‖ = √(Σᵢ |ψᵢ|²)
        """
        expected_norm = calculate_vector_norm(hilbert_state.vector)
        assert floats_close(hilbert_state.norm, expected_norm)
    
    def test_norm_is_positive_for_nonzero_vector(self, hilbert_state):
        """Verifica que la norma sea positiva para vectores no nulos."""
        assert hilbert_state.norm > 0.0
    
    def test_norm_is_zero_for_zero_vector(self):
        """Verifica que la norma sea cero para el vector nulo."""
        zero_state = HilbertState(vector=[0.0] * HEXAGON_VERTICES)
        assert zero_state.norm == 0.0
    
    def test_inner_product_with_self_equals_norm_squared(self, hilbert_state):
        """
        Verifica que ⟨ψ|ψ⟩ = ‖ψ‖².
        
        Esta es una propiedad fundamental de los espacios de Hilbert.
        """
        inner_product = hilbert_state.inner_product(hilbert_state)
        norm_squared = hilbert_state.norm ** 2
        assert floats_close(inner_product, norm_squared)
    
    def test_inner_product_with_orthogonal_vector(self, hilbert_state):
        """
        Verifica que el producto interno con un vector ortogonal sea cero.
        
        Vector original: [1.0, 0.5, 0.0, -0.5, -1.0, 0.0]
        Vector ortogonal: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        
        ⟨ψ|φ⟩ = 0·1 + 0.5·1 + 0·0 + (-0.5)·1 + (-1)·0 + 0·0 = 0
        """
        orthogonal = HilbertState(vector=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        inner_product = hilbert_state.inner_product(orthogonal)
        assert floats_close(inner_product, 0.0, abs_tol=1e-9)
    
    def test_inner_product_symmetry(self, hilbert_state):
        """
        Verifica la simetría del producto interno: ⟨ψ|φ⟩ = ⟨φ|ψ⟩* (conjugado).
        
        Para vectores reales: ⟨ψ|φ⟩ = ⟨φ|ψ⟩
        """
        other = HilbertState(vector=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert floats_close(
            hilbert_state.inner_product(other),
            other.inner_product(hilbert_state)
        )
    
    def test_normalize_produces_unit_vector(self, hilbert_state):
        """Verifica que la normalización produzca un vector de norma 1."""
        normalized = hilbert_state.normalize()
        assert floats_close(normalized.norm, 1.0)
    
    def test_normalize_preserves_direction(self, hilbert_state):
        """Verifica que la normalización preserve la dirección del vector."""
        original_vector = hilbert_state.vector.copy()
        original_norm = hilbert_state.norm
        normalized = hilbert_state.normalize()
        
        # Verificar que cada componente se escale por el mismo factor
        for i in range(HEXAGON_VERTICES):
            expected = original_vector[i] / original_norm
            assert floats_close(normalized.vector[i], expected)
    
    def test_normalize_zero_vector_is_safe(self):
        """Verifica que normalizar un vector cero no cause errores."""
        zero_state = HilbertState(vector=[0.0] * HEXAGON_VERTICES)
        # No debe lanzar excepción
        normalized = zero_state.normalize()
        assert normalized.norm == 0.0
    
    def test_damping_reduces_norm(self, hilbert_state):
        """
        Verifica que el amortiguamiento reduzca la norma del estado.
        
        El amortiguamiento modela la disipación de energía en el sistema.
        """
        original_norm = hilbert_state.norm
        hilbert_state.apply_damping(cycle=1)
        assert hilbert_state.norm < original_norm
    
    def test_damping_at_cycle_zero_is_identity(self, hilbert_state):
        """Verifica que el amortiguamiento en ciclo 0 no modifique el estado."""
        original_vector = hilbert_state.vector.copy()
        hilbert_state.apply_damping(cycle=0)
        assert vectors_close(hilbert_state.vector, original_vector)
    
    def test_damping_is_monotonic(self, hilbert_state):
        """Verifica que el amortiguamiento sea monótono con los ciclos."""
        norms = []
        for cycle in range(5):
            state_copy = HilbertState(vector=hilbert_state.vector.copy())
            state_copy.apply_damping(cycle)
            norms.append(state_copy.norm)
        
        # La norma en ciclo 0 debe ser la original
        assert floats_close(norms[0], hilbert_state.norm)
        # Las normas subsecuentes deben decrecer
        for i in range(1, len(norms)):
            assert norms[i] < norms[0]
    
    def test_orthogonal_projection_produces_orthogonal_result(self, hilbert_state):
        """
        Verifica que la proyección ortogonal produzca un vector ortogonal a la base.
        
        Sea B = {b₁, b₂, ...} una base ortonormal del subespacio.
        La proyección ortogonal es: ψ_⊥ = ψ - Σᵢ ⟨bᵢ|ψ⟩ bᵢ
        """
        # Crear un vector base normalizado
        basis_vector = HilbertState(vector=[1.0] * HEXAGON_VERTICES).normalize()
        
        # Proyectar
        hilbert_state.project_orthogonal([basis_vector])
        
        # Verificar ortogonalidad
        inner_product = hilbert_state.inner_product(basis_vector)
        assert floats_close(inner_product, 0.0, abs_tol=1e-9)
    
    def test_orthogonal_projection_reduces_norm(self, hilbert_state):
        """
        Verifica que la proyección ortogonal no aumente la norma.
        
        Por el teorema de Pitágoras: ‖ψ‖² = ‖ψ_∥‖² + ‖ψ_⊥‖²
        Por lo tanto: ‖ψ_⊥‖ ≤ ‖ψ‖
        """
        original_norm = hilbert_state.norm
        basis_vector = HilbertState(vector=[1.0] * HEXAGON_VERTICES).normalize()
        hilbert_state.project_orthogonal([basis_vector])
        assert hilbert_state.norm <= original_norm + FLOAT_ABS_TOL


# =============================================================================
# PRUEBAS DE TOPOLOGÍA HEXAGONAL
# =============================================================================

class TestHexagonalTopology:
    """
    Pruebas para la topología hexagonal.
    
    El grafo C₆ tiene las siguientes propiedades:
    - Grafo regular de grado 2
    - Diámetro: 3
    - Circunferencia: 6
    - Número cromático: 2
    - Autovalores del Laplaciano: {0, 1, 1, 3, 3, 4}
    """
    
    def test_adjacency_matrix_structure(self, topology):
        """
        Verifica la estructura de la matriz de adyacencia.
        
        Para C₆: A[i,j] = 1 si |i-j| ≡ 1 (mod 6), 0 en otro caso.
        """
        expected_adjacency = [
            [0, 1, 0, 0, 0, 1],  # C1 ↔ C2, C6
            [1, 0, 1, 0, 0, 0],  # C2 ↔ C1, C3
            [0, 1, 0, 1, 0, 0],  # C3 ↔ C2, C4
            [0, 0, 1, 0, 1, 0],  # C4 ↔ C3, C5
            [0, 0, 0, 1, 0, 1],  # C5 ↔ C4, C6
            [1, 0, 0, 0, 1, 0]   # C6 ↔ C5, C1
        ]
        assert topology.adjacency == expected_adjacency
    
    def test_adjacency_matrix_is_symmetric(self, topology):
        """Verifica que la matriz de adyacencia sea simétrica (grafo no dirigido)."""
        adj = topology.adjacency
        for i in range(HEXAGON_VERTICES):
            for j in range(HEXAGON_VERTICES):
                assert adj[i][j] == adj[j][i]
    
    def test_degree_sequence_is_uniform(self, topology):
        """
        Verifica que la secuencia de grados sea uniforme.
        
        En C₆, cada vértice tiene exactamente 2 vecinos (grafo 2-regular).
        """
        expected_degrees = [HEXAGON_DEGREE] * HEXAGON_VERTICES
        assert topology.degree == expected_degrees
    
    def test_laplacian_matrix_construction(self, topology):
        """
        Verifica la construcción de la matriz Laplaciana.
        
        L = D - A, donde D es la matriz diagonal de grados.
        Para C₆: L[i,i] = 2, L[i,j] = -1 si adyacentes, 0 en otro caso.
        """
        expected_laplacian = [
            [2, -1, 0, 0, 0, -1],
            [-1, 2, -1, 0, 0, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 2, -1, 0],
            [0, 0, 0, -1, 2, -1],
            [-1, 0, 0, 0, -1, 2]
        ]
        for i in range(HEXAGON_VERTICES):
            for j in range(HEXAGON_VERTICES):
                assert topology.laplacian[i][j] == expected_laplacian[i][j]
    
    def test_laplacian_is_symmetric(self, topology):
        """Verifica que la matriz Laplaciana sea simétrica."""
        lap = topology.laplacian
        for i in range(HEXAGON_VERTICES):
            for j in range(HEXAGON_VERTICES):
                assert lap[i][j] == lap[j][i]
    
    def test_laplacian_rows_sum_to_zero(self, topology):
        """
        Verifica que cada fila del Laplaciano sume cero.
        
        Esta es una propiedad fundamental: L·1 = 0 (el vector constante es autovector con λ=0).
        """
        for row in topology.laplacian:
            assert sum(row) == 0
    
    def test_spectral_gap_value(self, topology):
        """
        Verifica el valor de la brecha espectral (Fiedler value).
        
        Para C₆: λ₁ = 2 - 2·cos(2π/6) = 2 - 2·(1/2) = 1.0
        
        La brecha espectral determina la velocidad de convergencia de la difusión.
        """
        assert floats_close(topology.spectral_gap, SPECTRAL_GAP_C6)
    
    def test_neighbor_indices_for_all_nodes(self, topology):
        """
        Verifica los índices de vecinos para todos los nodos.
        
        Para el nodo i en C₆: vecinos son (i-1) mod 6 y (i+1) mod 6.
        """
        expected_neighbors = {
            0: (5, 1),  # C1: vecinos C6 y C2
            1: (0, 2),  # C2: vecinos C1 y C3
            2: (1, 3),  # C3: vecinos C2 y C4
            3: (2, 4),  # C4: vecinos C3 y C5
            4: (3, 5),  # C5: vecinos C4 y C6
            5: (4, 0)   # C6: vecinos C5 y C1
        }
        for node_idx, expected in expected_neighbors.items():
            assert topology.neighbor_indices(node_idx) == expected
    
    def test_diffusion_preserves_total_mass(self, topology):
        """
        Verifica que la difusión preserve la masa total (propiedad de conservación).
        
        Σᵢ u_new[i] = Σᵢ u_old[i]
        """
        state = [1.0, 2.0, 0.5, 1.5, 0.0, 1.0]
        total_mass_before = sum(state)
        diffused = topology.diffuse_stress(state, diffusion_rate=0.1)
        total_mass_after = sum(diffused)
        assert floats_close(total_mass_before, total_mass_after, abs_tol=1e-6)
    
    def test_diffusion_with_safe_rate(self, topology):
        """Verifica que la difusión con tasa segura produzca valores razonables."""
        state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        safe_rate = 0.49  # Menor que CFL_max = 0.5 para grado 2
        
        diffused = topology.diffuse_stress(state, diffusion_rate=safe_rate)
        
        # Verificar que los valores sean razonables (no exploten)
        for value in diffused:
            assert -1.0 <= value <= 2.0
    
    def test_diffusion_with_unsafe_rate_triggers_warning(self, topology):
        """
        Verifica que una tasa de difusión insegura genere advertencia.
        
        Condición CFL: Δt ≤ 1/(2·d_max) para estabilidad, donde d_max es el grado máximo.
        """
        state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        unsafe_rate = 0.6  # Mayor que CFL_max
        
        with pytest.warns(UserWarning, match="CFL|stability|unstable"):
            topology.diffuse_stress(state, diffusion_rate=unsafe_rate)
    
    def test_diffusion_with_dirichlet_boundary(self, topology):
        """
        Verifica condiciones de frontera de Dirichlet (valor fijo).
        
        Los nodos con condición de Dirichlet mantienen su valor prescrito.
        """
        state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dirichlet_bc = {'dirichlet': {0: 2.0, 3: -1.0}}
        
        diffused = topology.diffuse_stress(
            state, diffusion_rate=0.1, boundary_conditions=dirichlet_bc
        )
        
        assert floats_close(diffused[0], 2.0)
        assert floats_close(diffused[3], -1.0)
    
    def test_diffusion_with_neumann_boundary(self, topology):
        """
        Verifica condiciones de frontera de Neumann (flujo prescrito).
        
        Un flujo positivo incrementa el valor, uno negativo lo decrementa.
        """
        state = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        neumann_bc = {'neumann': {0: 0.5, 5: -0.2}}
        
        diffused = topology.diffuse_stress(
            state, diffusion_rate=0.1, boundary_conditions=neumann_bc
        )
        
        # Flujo positivo debe incrementar
        assert diffused[0] > state[0]
        # Flujo negativo debe decrementar
        assert diffused[5] < state[5]
    
    def test_diffusion_smooths_peak(self, topology):
        """
        Verifica que la difusión suavice un pico de concentración.
        
        Este es el comportamiento físico esperado de la ecuación del calor.
        """
        state = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Pico en nodo 0
        
        diffused = topology.diffuse_stress(state, diffusion_rate=0.2)
        
        # El pico debe reducirse
        assert diffused[0] < state[0]
        # Los vecinos deben aumentar
        assert diffused[1] > state[1]
        assert diffused[5] > state[5]


# =============================================================================
# PRUEBAS DE POTENCIAL TERMODINÁMICO
# =============================================================================

class TestThermodynamicPotential:
    """
    Pruebas para el potencial termodinámico.
    
    Relaciones fundamentales:
    - Temperatura: T = T₀ + γ·σ² (principio de equipartición modificado)
    - Gibbs: G = H - T·S·κ (energía libre)
    - Inestabilidad: I = ln(1 + |ΔG|) + σ
    """
    
    def test_initialization_stores_parameters(self, thermodynamic_potential):
        """Verifica que los parámetros se almacenen correctamente."""
        assert thermodynamic_potential.enthalpy == 100.0
        assert thermodynamic_potential.entropy == 0.5
        assert thermodynamic_potential.base_temperature == 298.0
        assert thermodynamic_potential.temperature_coupling == 15.0
        assert thermodynamic_potential.topological_stress == 0.8
    
    def test_temperature_calculation(self, thermodynamic_potential):
        """
        Verifica el cálculo de la temperatura efectiva.
        
        T = T₀ + γ·σ²
        T = 298.0 + 15.0 · (0.8)² = 298.0 + 9.6 = 307.6 K
        """
        expected_temperature = 298.0 + 15.0 * (0.8 ** 2)
        assert floats_close(thermodynamic_potential.temperature, expected_temperature)
    
    def test_temperature_increases_with_stress(self):
        """Verifica que la temperatura aumente con el estrés topológico."""
        pot_low_stress = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.2
        )
        pot_high_stress = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.9
        )
        assert pot_high_stress.temperature > pot_low_stress.temperature
    
    def test_gibbs_free_energy_calculation(self, thermodynamic_potential):
        """
        Verifica el cálculo de la energía libre de Gibbs.
        
        G = H - T·S·κ
        """
        T = thermodynamic_potential.temperature
        expected_gibbs = 100.0 - (T * 0.5 * BOLTZMANN_SCALE)
        assert floats_close(thermodynamic_potential.gibbs_free_energy, expected_gibbs)
    
    def test_gibbs_decreases_with_entropy(self):
        """
        Verifica que G disminuya cuando S aumenta (a T constante).
        
        ∂G/∂S = -T < 0 siempre que T > 0
        """
        pot_low_s = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.3, base_temperature=298.0,
            temperature_coupling=0.0, topological_stress=0.0
        )
        pot_high_s = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.8, base_temperature=298.0,
            temperature_coupling=0.0, topological_stress=0.0
        )
        assert pot_high_s.gibbs_free_energy < pot_low_s.gibbs_free_energy
    
    def test_instability_index_calculation(self, thermodynamic_potential):
        """
        Verifica el cálculo del índice de inestabilidad.
        
        I = ln(1 + |ΔG|) + σ
        """
        gibbs = thermodynamic_potential.gibbs_free_energy
        expected_instability = math.log1p(abs(gibbs)) + 0.8
        assert floats_close(thermodynamic_potential.instability, expected_instability)
    
    def test_instability_increases_with_stress(self):
        """Verifica que la inestabilidad aumente con el estrés topológico."""
        pot_low = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.1
        )
        pot_high = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.9
        )
        assert pot_high.instability > pot_low.instability
    
    def test_update_modifies_state(self, thermodynamic_potential):
        """Verifica que update() modifique el estado correctamente."""
        thermodynamic_potential.update(200.0, 0.8, 0.5)
        
        assert thermodynamic_potential.enthalpy == 200.0
        assert thermodynamic_potential.entropy == 0.8
        assert thermodynamic_potential.topological_stress == 0.5
    
    def test_update_recalculates_temperature(self, thermodynamic_potential):
        """Verifica que update() recalcule la temperatura."""
        thermodynamic_potential.update(200.0, 0.8, 0.5)
        
        expected_temperature = 298.0 + 15.0 * (0.5 ** 2)
        assert floats_close(thermodynamic_potential.temperature, expected_temperature)
    
    def test_entropy_minimum_is_enforced(self):
        """Verifica que la entropía no sea menor que ENTROPY_MIN_PROB."""
        potential = ThermodynamicPotential(
            enthalpy=10.0, entropy=ENTROPY_MIN_PROB / 2,  # Intentar valor menor
            base_temperature=298.0, temperature_coupling=15.0,
            topological_stress=0.5
        )
        assert potential.entropy >= ENTROPY_MIN_PROB
    
    def test_enthalpy_minimum_is_enforced(self):
        """Verifica que la entalpía sea ajustada si es demasiado baja."""
        potential = ThermodynamicPotential(
            enthalpy=0.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.5
        )
        # Debería ser ajustado a un valor mínimo positivo
        assert potential.enthalpy > 0.0
    
    def test_gibbs_is_finite_at_extreme_values(self):
        """Verifica que G sea finito incluso con valores extremos."""
        potential = ThermodynamicPotential(
            enthalpy=1e6, entropy=0.99, base_temperature=1000.0,
            temperature_coupling=100.0, topological_stress=10.0
        )
        assert math.isfinite(potential.gibbs_free_energy)
        assert not math.isnan(potential.gibbs_free_energy)


# =============================================================================
# PRUEBAS DE ENTROPÍA DE SHANNON
# =============================================================================

class TestShannonEntropy:
    """
    Pruebas para el cálculo de entropía de Shannon.
    
    H(X) = -Σᵢ p(xᵢ) · log(p(xᵢ))
    
    Propiedades:
    - H ≥ 0 siempre
    - H = 0 si y solo si la distribución es determinista
    - H es máxima para distribución uniforme
    """
    
    def test_empty_context_has_zero_entropy(self):
        """Un contexto vacío tiene entropía 0 (no hay incertidumbre)."""
        assert CatalyticReactor._calculate_shannon_entropy({}) == 0.0
    
    def test_single_element_has_zero_entropy(self):
        """Un solo elemento tiene entropía 0 (distribución determinista)."""
        context = {"key": "value"}
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        assert floats_close(entropy, 0.0)
    
    def test_identical_elements_have_zero_entropy(self):
        """
        Elementos idénticos tienen entropía 0.
        
        Si todas las firmas son iguales, p = 1 y H = -1·log(1) = 0.
        """
        context = {"k1": "same", "k2": "same", "k3": "same"}
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        assert floats_close(entropy, 0.0, abs_tol=1e-9)
    
    def test_uniform_distribution_has_maximum_entropy(self):
        """
        Distribución uniforme tiene entropía máxima.
        
        Para n clases equiprobables: H = log(n)
        """
        # 4 elementos con firmas diferentes
        context = {"int": 42, "float": 3.14, "str": "hello", "bool": True}
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        
        # Cada tipo tiene probabilidad 1/4
        max_entropy = math.log(4)  # ≈ 1.386
        assert entropy <= max_entropy + FLOAT_ABS_TOL
        assert entropy > 0.0
    
    def test_entropy_is_non_negative(self):
        """La entropía siempre es no negativa."""
        contexts = [
            {},
            {"a": 1},
            {"a": 1, "b": 2, "c": 3},
            {"a": "x", "b": "y", "c": "x"},
        ]
        for ctx in contexts:
            entropy = CatalyticReactor._calculate_shannon_entropy(ctx)
            assert entropy >= 0.0
    
    def test_entropy_increases_with_diversity(self):
        """La entropía aumenta con la diversidad de tipos."""
        ctx_uniform = {"a": 1, "b": 1, "c": 1}  # Todos iguales
        ctx_diverse = {"a": 1, "b": "str", "c": 3.14}  # Diversos
        
        h_uniform = CatalyticReactor._calculate_shannon_entropy(ctx_uniform)
        h_diverse = CatalyticReactor._calculate_shannon_entropy(ctx_diverse)
        
        assert h_diverse > h_uniform
    
    def test_entropy_handles_none_values(self):
        """La entropía maneja valores None correctamente."""
        context = {"key1": None, "key2": None}
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        # Ambos son None, misma firma → entropía 0
        assert floats_close(entropy, 0.0, abs_tol=1e-9)
    
    def test_entropy_handles_large_objects(self):
        """La entropía maneja objetos grandes eficientemente."""
        context = {"key": "a" * 1000000}
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        assert math.isfinite(entropy)
        assert not math.isnan(entropy)
    
    def test_entropy_with_numeric_types(self):
        """La entropía distingue correctamente tipos numéricos."""
        context = {
            "int": 42,
            "float": 42.0,
            "complex": 42 + 0j,
            "bool": True  # Note: bool es subclase de int en Python
        }
        entropy = CatalyticReactor._calculate_shannon_entropy(context)
        # Debería haber algo de entropía por diferentes tipos/valores
        assert entropy >= 0.0


# =============================================================================
# PRUEBAS DE HAMILTONIANO Y ACTIVACIÓN
# =============================================================================

class TestHamiltonian:
    """
    Pruebas para el cálculo del Hamiltoniano.
    
    El Hamiltoniano representa la barrera de activación para cada nodo.
    H = α + β·(1 - estabilización_vecinos) + γ·(precursores_faltantes)
    """
    
    def test_base_hamiltonian_is_positive(self, reactor, empty_context):
        """El Hamiltoniano base es siempre positivo (α > 0)."""
        for node in CarbonNode:
            hamiltonian = reactor._calculate_hamiltonian(node, empty_context)
            assert hamiltonian > 0.0
    
    def test_hamiltonian_minimum_value(self, reactor, empty_context):
        """El Hamiltoniano nunca es menor que el mínimo (α ≈ 0.2)."""
        for node in CarbonNode:
            hamiltonian = reactor._calculate_hamiltonian(node, empty_context)
            assert hamiltonian >= 0.2
    
    def test_neighbor_resonance_reduces_hamiltonian(self, reactor, minimal_context):
        """Los vecinos resonantes reducen el Hamiltoniano (estabilización)."""
        context_with_neighbor = dict(minimal_context)
        context_with_neighbor[f"{CarbonNode.C2_PHYSICS.name}_status"] = "resonant"
        
        context_without_neighbor = dict(minimal_context)
        
        h_with = reactor._calculate_hamiltonian(CarbonNode.C1_INGESTION, context_with_neighbor)
        h_without = reactor._calculate_hamiltonian(CarbonNode.C1_INGESTION, context_without_neighbor)
        
        assert h_with < h_without
    
    def test_missing_precursors_increase_hamiltonian(self, reactor, minimal_context):
        """Los precursores faltantes aumentan el Hamiltoniano (penalización)."""
        context_no_precursors = dict(minimal_context)
        del context_no_precursors["financial_params"]
        
        h_missing = reactor._calculate_hamiltonian(CarbonNode.C4_STRATEGY, context_no_precursors)
        h_complete = reactor._calculate_hamiltonian(CarbonNode.C4_STRATEGY, minimal_context)
        
        assert h_missing > h_complete
    
    def test_hamiltonian_never_negative(self, reactor, full_context):
        """El Hamiltoniano nunca es negativo, incluso con máxima estabilización."""
        for node in CarbonNode:
            hamiltonian = reactor._calculate_hamiltonian(node, full_context)
            assert hamiltonian >= 0.0
    
    def test_catalyst_reduces_activation_energy(self, reactor, minimal_context, mock_catalyst):
        """El catalizador reduce la energía de activación efectiva."""
        node = CarbonNode.C4_STRATEGY
        base_ea = reactor._calculate_hamiltonian(node, minimal_context)
        effective_ea = base_ea * (1.0 - mock_catalyst.efficiency_factor)
        
        assert effective_ea < base_ea
        assert effective_ea > 0.0


# =============================================================================
# PRUEBAS DE AROMATICIDAD Y ESTABILIDAD
# =============================================================================

class TestAromaticity:
    """
    Pruebas para la detección de aromaticidad.
    
    Un sistema es aromático si:
    1. Todos los 6 nodos están en estado resonante (6 electrones π)
    2. No hay errores registrados
    3. No hay nodos saltados
    
    Esto cumple la regla de Hückel: 4n + 2 electrones, con n = 1.
    """
    
    def test_full_resonance_is_aromatic(self, full_context):
        """6 electrones π (todos resonantes) → aromático."""
        assert CatalyticReactor._is_aromatic(full_context)
    
    def test_missing_one_electron_not_aromatic(self, full_context):
        """5 electrones π → no aromático (no cumple 4n+2)."""
        del full_context[f"{CarbonNode.C6_MATTER.name}_status"]
        assert not CatalyticReactor._is_aromatic(full_context)
    
    def test_error_breaks_aromaticity(self, full_context):
        """Un error en cualquier nodo rompe la aromaticidad."""
        full_context[f"{CarbonNode.C3_TOPOLOGY.name}_error"] = "topology_error"
        assert not CatalyticReactor._is_aromatic(full_context)
    
    def test_skipped_node_breaks_aromaticity(self, full_context):
        """Un nodo saltado rompe la aromaticidad."""
        full_context[f"{CarbonNode.C2_PHYSICS.name}_skipped"] = True
        assert not CatalyticReactor._is_aromatic(full_context)
    
    def test_empty_context_not_aromatic(self, empty_context):
        """Contexto vacío → no aromático."""
        assert not CatalyticReactor._is_aromatic(empty_context)


class TestThermodynamicConvergence:
    """Pruebas para la detección de convergencia termodinámica."""
    
    def test_large_delta_not_converged(self, reactor):
        """Un delta grande en G indica no convergencia."""
        potential = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.8
        )
        previous_gibbs = 10.0  # Delta muy grande
        
        assert not reactor._check_thermodynamic_convergence(potential, previous_gibbs, cycle=2)
    
    def test_small_delta_is_converged(self, reactor):
        """Un delta pequeño en G indica convergencia."""
        potential = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.8
        )
        # Delta menor que GIBBS_CONVERGENCE_EPS
        previous_gibbs = potential.gibbs_free_energy + GIBBS_CONVERGENCE_EPS / 2
        
        assert reactor._check_thermodynamic_convergence(potential, previous_gibbs, cycle=2)
    
    def test_first_cycle_not_converged(self, reactor):
        """El primer ciclo nunca está convergido (necesita historial)."""
        potential = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=298.0,
            temperature_coupling=15.0, topological_stress=0.8
        )
        # Incluso con delta = 0
        assert not reactor._check_thermodynamic_convergence(
            potential, potential.gibbs_free_energy, cycle=1
        )


class TestStabilization:
    """Pruebas para el procedimiento de estabilización."""
    
    def test_stabilization_reduces_instability(
        self, reactor, hilbert_state, thermodynamic_potential
    ):
        """La estabilización reduce la inestabilidad del sistema."""
        # Crear estado inestable
        thermodynamic_potential.update(1000.0, 0.9, 2.0)
        original_instability = thermodynamic_potential.instability
        
        # Intentar estabilización
        reactor._attempt_stabilization(
            CarbonNode.C4_STRATEGY,
            hilbert_state,
            thermodynamic_potential,
            cycle=1
        )
        
        assert thermodynamic_potential.instability < original_instability
    
    def test_stabilization_reaches_threshold(
        self, reactor, hilbert_state, thermodynamic_potential
    ):
        """La estabilización lleva el sistema bajo el umbral."""
        thermodynamic_potential.update(500.0, 0.7, 1.5)
        
        reactor._attempt_stabilization(
            CarbonNode.C3_TOPOLOGY,
            hilbert_state,
            thermodynamic_potential,
            cycle=2
        )
        
        assert thermodynamic_potential.instability <= reactor.INSTABILITY_THRESHOLD
    
    def test_extreme_instability_causes_collapse(
        self, reactor, hilbert_state, thermodynamic_potential
    ):
        """Inestabilidad extrema causa colapso del reactor."""
        thermodynamic_potential.update(10000.0, 0.99, 5.0)
        
        # Mockear damping para que no cambie el estado
        with patch.object(HilbertState, 'apply_damping', return_value=None):
            with pytest.raises(RuntimeError, match="Reactor Collapse"):
                reactor._attempt_stabilization(
                    CarbonNode.C4_STRATEGY,
                    hilbert_state,
                    thermodynamic_potential,
                    cycle=1
                )


# =============================================================================
# PRUEBAS DE INTEGRACIÓN DEL REACTOR
# =============================================================================

class TestReactorIgnition:
    """Pruebas de integración para el ciclo de ignición del reactor."""
    
    def test_successful_ignition_with_full_context(
        self, reactor, full_context, mock_telemetry
    ):
        """Ignición exitosa con contexto completo."""
        result = reactor.ignite(full_context)
        
        assert result is not None
        assert "_metastable_cycle" not in result
        mock_telemetry.record_reaction_success.assert_called()
    
    def test_metastable_state_detection(
        self, reactor, minimal_context, mock_telemetry
    ):
        """Detección de estado metaestable sin aromaticidad completa."""
        with patch.object(CatalyticReactor, '_is_aromatic', return_value=False), \
             patch.object(CatalyticReactor, '_check_thermodynamic_convergence', return_value=True):
            
            result = reactor.ignite(minimal_context)
            assert "_metastable_cycle" in result
    
    def test_failure_with_empty_context(
        self, reactor, empty_context, mock_telemetry
    ):
        """Fallo con contexto vacío."""
        with pytest.raises(RuntimeError, match="aromatic stability"):
            reactor.ignite(empty_context)
        
        mock_telemetry.record_error.assert_called()
    
    def test_node_skipping_on_high_barrier(self, reactor, minimal_context, mock_mic):
        """Nodos se saltan cuando la barrera es muy alta."""
        high_barrier = reactor.ACTIVATION_BARRIER_CEILING + 0.1
        
        with patch.object(
            CatalyticReactor, '_calculate_hamiltonian', return_value=high_barrier
        ):
            result = reactor.ignite(minimal_context)
            
            # Al menos un nodo debería estar saltado
            skipped_keys = [k for k in result if "_skipped" in k]
            assert len(skipped_keys) > 0
    
    def test_node_error_handling(self, reactor, minimal_context):
        """Manejo correcto de errores en nodos."""
        def failing_react_node(node, context, ea, stress):
            if node == CarbonNode.C3_TOPOLOGY:
                raise ValueError("Simulated topology failure")
            return context, 10.0
        
        with patch.object(
            CatalyticReactor, '_react_node', side_effect=failing_react_node
        ):
            result = reactor.ignite(minimal_context)
            
            assert f"{CarbonNode.C3_TOPOLOGY.name}_error" in result
    
    def test_catalyst_orientation_applied(
        self, reactor, minimal_context, mock_catalyst
    ):
        """El catalizador orienta correctamente el contexto."""
        result = reactor.ignite(minimal_context)
        
        # Verificar que el ajuste del catalizador fue aplicado
        assert "catalyst_adjustment" in result
        assert result["catalyst_adjustment"] == 0.1
    
    def test_correct_number_of_resonance_cycles(
        self, reactor, minimal_context, caplog
    ):
        """El reactor usa el número correcto de ciclos."""
        caplog.set_level(logging.INFO)
        
        with patch.object(CatalyticReactor, '_is_aromatic', return_value=False), \
             patch.object(
                 CatalyticReactor, '_check_thermodynamic_convergence',
                 side_effect=[False, True]
             ):
            
            reactor.ignite(minimal_context)
            
            # Verificar ciclos ejecutados
            assert "Ciclo de Resonancia 1" in caplog.text
            assert "Ciclo de Resonancia 2" in caplog.text
    
    def test_topological_diffusion_smooths_stress(self, topology):
        """La difusión topológica suaviza picos de estrés."""
        # Estado con pico en nodo 0
        initial_state = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Aplicar múltiples pasos de difusión
        state = initial_state.copy()
        for _ in range(10):
            state = topology.diffuse_stress(state, diffusion_rate=0.2)
        
        # El estrés debería estar más uniformemente distribuido
        variance_initial = sum((x - sum(initial_state)/6)**2 for x in initial_state)
        variance_final = sum((x - sum(state)/6)**2 for x in state)
        
        assert variance_final < variance_initial


# =============================================================================
# PRUEBAS DE RENDIMIENTO
# =============================================================================

class TestPerformance:
    """Pruebas de rendimiento para componentes críticos."""
    
    @pytest.mark.slow
    def test_reactor_ignition_performance(self, reactor, full_context, benchmark):
        """Benchmark del ciclo de ignición completo."""
        result = benchmark(reactor.ignite, full_context)
        assert result is not None
    
    def test_shannon_entropy_large_context_performance(self, benchmark):
        """Benchmark de entropía con contexto grande."""
        large_context = {f"key_{i}": f"value_{i % 10}" for i in range(10000)}
        
        result = benchmark(CatalyticReactor._calculate_shannon_entropy, large_context)
        assert result >= 0.0
    
    def test_topology_diffusion_performance(self, topology, benchmark):
        """Benchmark de difusión topológica."""
        state = [1.0 if i == 0 else 0.0 for i in range(HEXAGON_VERTICES)]
        
        result = benchmark(topology.diffuse_stress, state, 0.1)
        assert len(result) == HEXAGON_VERTICES
    
    def test_hilbert_normalization_performance(self, benchmark):
        """Benchmark de normalización en espacio de Hilbert."""
        state = HilbertState(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        result = benchmark(state.normalize)
        assert floats_close(result.norm, 1.0)


# =============================================================================
# PRUEBAS DE PROPIEDADES MATEMÁTICAS
# =============================================================================

class TestMathematicalProperties:
    """
    Pruebas de propiedades matemáticas fundamentales.
    
    Estas pruebas verifican que la implementación respete
    los invariantes matemáticos del sistema.
    """
    
    def test_laplacian_is_positive_semidefinite(self, topology):
        """
        El Laplaciano es positivo semidefinido.
        
        Para todo vector v: v^T L v ≥ 0
        """
        L = topology.laplacian
        
        # Probar con varios vectores
        test_vectors = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
        ]
        
        for v in test_vectors:
            # Calcular v^T L v
            Lv = [sum(L[i][j] * v[j] for j in range(6)) for i in range(6)]
            quadratic_form = sum(v[i] * Lv[i] for i in range(6))
            assert quadratic_form >= -FLOAT_ABS_TOL
    
    def test_hilbert_space_linearity(self):
        """
        El espacio de Hilbert es lineal.
        
        ⟨αψ + βφ | χ⟩ = α⟨ψ|χ⟩ + β⟨φ|χ⟩
        """
        psi = HilbertState(vector=[1, 0, 0, 0, 0, 0])
        phi = HilbertState(vector=[0, 1, 0, 0, 0, 0])
        chi = HilbertState(vector=[1, 1, 0, 0, 0, 0])
        
        alpha, beta = 2.0, 3.0
        
        # αψ + βφ
        combined_vector = [alpha * psi.vector[i] + beta * phi.vector[i] for i in range(6)]
        combined = HilbertState(vector=combined_vector)
        
        # Verificar linealidad
        lhs = combined.inner_product(chi)
        rhs = alpha * psi.inner_product(chi) + beta * phi.inner_product(chi)
        
        assert floats_close(lhs, rhs)
    
    def test_gibbs_duhem_relation(self):
        """
        Verificar consistencia con la relación de Gibbs-Duhem.
        
        A temperatura constante: dG = dH - T·dS
        """
        T = 300.0  # Temperatura fija
        
        pot1 = ThermodynamicPotential(
            enthalpy=100.0, entropy=0.5, base_temperature=T,
            temperature_coupling=0.0, topological_stress=0.0
        )
        pot2 = ThermodynamicPotential(
            enthalpy=150.0, entropy=0.7, base_temperature=T,
            temperature_coupling=0.0, topological_stress=0.0
        )
        
        dH = pot2.enthalpy - pot1.enthalpy
        dS = pot2.entropy - pot1.entropy
        dG = pot2.gibbs_free_energy - pot1.gibbs_free_energy
        
        expected_dG = dH - T * dS * BOLTZMANN_SCALE
        
        assert floats_close(dG, expected_dG, rel_tol=1e-6)
    
    def test_cyclic_symmetry_of_hexagon(self, topology):
        """
        El hexágono tiene simetría cíclica C₆.
        
        Rotar los índices no cambia las propiedades topológicas.
        """
        def rotate_state(state, k):
            """Rota el estado k posiciones."""
            n = len(state)
            return [state[(i - k) % n] for i in range(n)]
        
        state = [1.0, 0.5, 0.0, -0.5, -1.0, 0.0]
        
        for k in range(6):
            rotated = rotate_state(state, k)
            
            # La difusión debería conmutar con la rotación
            diffused_then_rotated = rotate_state(
                topology.diffuse_stress(state, 0.1), k
            )
            rotated_then_diffused = topology.diffuse_stress(rotated, 0.1)
            
            assert vectors_close(
                diffused_then_rotated, 
                rotated_then_diffused,
                rel_tol=1e-6
            )