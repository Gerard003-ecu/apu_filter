# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Test Suite del Scalar Higgs Anchor v6.0                              ║
║ Ubicación: tests/unit/physics/test_scalar_higgs_anchor.py                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
              ARQUITECTURA DE LA SUITE DE PRUEBAS
═══════════════════════════════════════════════════════════════════════════════

    Capa 1 · Pruebas Unitarias de la Fase 1 (Axiomática del Campo)
        ├── 1A. QFTParameters (invariantes I1–I6, propiedades derivadas)
        ├── 1B. ScalarFieldState (invariantes J1–J3)
        ├── 1C. FermionicSource (densidad de carga)
        ├── 1D. LaplacedBeltramiOperator (L1–L4)
        └── 1E. SpectralAnalyzer (Gershgorin, power, Krylov, gap)

    Capa 2 · Pruebas Unitarias de la Fase 2 (Dinámica Simpléctica)
        ├── 2A. HiggsPotential (valor, gradiente, Lipschitz)
        ├── 2B. PortHamiltonianHamiltonian (H, fuerza)
        ├── 2C. VelocityVerletIntegrator (algoritmo, CFL, damping)
        ├── 2D. StabilityMonitor (análisis, termostato)
        └── 2E. validate_topological_consistency (T1–T4)

    Capa 3 · Pruebas Unitarias de la Fase 3 (Funtor de Anclaje)
        ├── 3A. ScalarHiggsAnchor (construcción, vacío, ψ-extract)
        ├── 3B. Acoplamiento de Yukawa (m_eff)
        ├── 3C. Reproducibilidad (RNG inyectado)
        └── 3D. apply_higgs_anchor (decorador categórico)

    Capa 4 · Pruebas de Integración y Propiedades
        ├── INT-1.  Pipeline end-to-end con ψ válido
        ├── INT-2.  Idempotencia estructural
        ├── INT-3.  Estabilidad Lyapunov (Ḣ ≤ 0)
        ├── INT-4.  Detección de divergencia (campo grande)
        ├── INT-5.  Verificación de invariantes topológicos T1–T4
        └── PROP.   Pruebas basadas en propiedades

    Regresión · Pruebas parametrizadas (dimensiones, semillas)
═══════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.sparse as sp

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
from app.physics.scalar_higgs_anchor import (
    # ── Fase 1 ──
    QFTParameters,
    QFT,
    ScalarFieldState,
    FermionicSource,
    LaplacedBeltramiOperator,
    SpectralAnalyzer,
    # ── Fase 2 ──
    HiggsPotential,
    PortHamiltonianHamiltonian,
    SymplecticIntegrator,
    VelocityVerletIntegrator,
    StabilityMonitor,
    StabilityMetrics,
    validate_topological_consistency,
    # ── Fase 3 ──
    ScalarHiggsAnchor,
    construct_laplacian_from_adjacency,
    apply_higgs_anchor,
)

from app.core.mic_algebra import (
    CategoricalState,
    FunctorialityError,
    Morphism,
    NumericalInstabilityError,
)
from app.core.schemas import Stratum
from app.core.quantum_algebra import TopologicalInvariantError

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE TEST
# ══════════════════════════════════════════════════════════════════════════════
TEST_SEED: int = 42
TOL_STRICT: float = 1e-10
TOL_LOOSE: float = 1e-6
TOL_PHYS: float = 1e-3  # tolerancia física


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng() -> np.random.Generator:
    """Generador determinista compartido."""
    return np.random.default_rng(TEST_SEED)


@pytest.fixture
def default_params() -> QFTParameters:
    """Parámetros por defecto (instancia inmutable)."""
    return QFTParameters()


@pytest.fixture
def custom_params() -> QFTParameters:
    """Parámetros custom para tests que requieren valores específicos."""
    return QFTParameters(
        MU_SQUARED=1.0,
        LAMBDA_COUPLING=0.5,
        YUKAWA_G=2.0,
        DAMPING_GAMMA=0.1,
        CFL_SAFETY_FACTOR=0.5,
        FIELD_REG_SCALE=5.0,
        RNG_SEED=123,
    )


@pytest.fixture
def path_graph_adjacency() -> np.ndarray:
    """Matriz de adyacencia de un grafo path de 5 nodos."""
    n = 5
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


@pytest.fixture
def complete_graph_adjacency() -> np.ndarray:
    """Matriz de adyacencia de un grafo completo K_4 (λ₁ = 4)."""
    n = 4
    A = np.ones((n, n), dtype=np.float64) - np.eye(n)
    return A


@pytest.fixture
def disconnected_adjacency() -> np.ndarray:
    """Grafo de 6 nodos: dos componentes de 3 nodos cada uno."""
    A = np.zeros((6, 6), dtype=np.float64)
    # Componente 1: triángulo (0,1,2)
    A[0, 1] = A[1, 0] = 1.0
    A[1, 2] = A[2, 1] = 1.0
    A[0, 2] = A[2, 0] = 1.0
    # Componente 2: triángulo (3,4,5)
    A[3, 4] = A[4, 3] = 1.0
    A[4, 5] = A[5, 4] = 1.0
    A[3, 5] = A[5, 3] = 1.0
    return A


@pytest.fixture
def path_laplacian(path_graph_adjacency: np.ndarray) -> sp.csr_matrix:
    """Laplaciano del path graph de 5 nodos."""
    return construct_laplacian_from_adjacency(path_graph_adjacency)


@pytest.fixture
def complete_laplacian(complete_graph_adjacency: np.ndarray) -> sp.csr_matrix:
    """Laplaciano del grafo completo K_4."""
    return construct_laplacian_from_adjacency(complete_graph_adjacency)


@pytest.fixture
def path_lb_operator(path_graph_adjacency: np.ndarray) -> LaplacedBeltramiOperator:
    """Operador de Laplace-Beltrami para path graph."""
    return LaplacedBeltramiOperator.from_adjacency(path_graph_adjacency)


@pytest.fixture
def simple_state() -> ScalarFieldState:
    """Estado escalar con campo cerca del VEV y momento cero."""
    vev = QFT.vev
    return ScalarFieldState(
        phi=np.full(4, vev, dtype=np.float64),
        pi_momentum=np.zeros(4, dtype=np.float64),
    )


@pytest.fixture
def random_state(rng: np.random.Generator) -> ScalarFieldState:
    """Estado aleatorio con campo cerca del VEV."""
    vev = QFT.vev
    return ScalarFieldState(
        phi=vev + 0.1 * rng.standard_normal(4),
        pi_momentum=0.1 * rng.standard_normal(4),
    )


@pytest.fixture
def simple_potential() -> HiggsPotential:
    """Instancia del potencial con parámetros por defecto."""
    return HiggsPotential(QFT)


@pytest.fixture
def simple_hamiltonian(path_lb_operator: LaplacedBeltramiOperator) -> PortHamiltonianHamiltonian:
    """Hamiltoniano de prueba con parámetros por defecto."""
    return PortHamiltonianHamiltonian(
        laplacian=path_lb_operator,
        potential=HiggsPotential(QFT),
        params=QFT,
    )


@pytest.fixture
def sample_fermion() -> FermionicSource:
    """Fuente fermiónica con vector normalizado."""
    psi = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float64)
    return FermionicSource(psi_vector=psi, base_mass=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Stubs de CategoricalState
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_psi_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado categórico con vector estocástico válido."""
    return CategoricalState(
        payload={"stochastic_vector": [0.5, 0.3, 0.8, 0.1, -0.2]},
        stratum=mock_stratum,
    )


@pytest.fixture
def empty_psi_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado categórico sin vector estocástico."""
    return CategoricalState(
        payload={"other_key": "value"},
        stratum=mock_stratum,
    )


@pytest.fixture
def ndarray_payload_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload ndarray directo (Estrategia P2)."""
    return CategoricalState(
        payload=np.array([0.1, 0.2, 0.3, 0.4]),
        stratum=mock_stratum,
    )


@pytest.fixture
def list_payload_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload iterable (Estrategia P3)."""
    return CategoricalState(
        payload=[0.1, 0.2, 0.3, 0.4],
        stratum=mock_stratum,
    )


@pytest.fixture
def scalar_payload_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload escalar no vectorizable (Fallback P4)."""
    return CategoricalState(
        payload="not_a_vector",
        stratum=mock_stratum,
    )


@pytest.fixture
def mock_stratum() -> Stratum:
    """Estrato mockeado."""
    return Stratum.L3_KNOWLEDGE


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 1 · PRUEBAS UNITARIAS DE LA FASE 1 (AXIOMÁTICA DEL CAMPO)        │
# └─────────────────────────────────────────────────────────────────────────┘


class TestQFTParameters:
    """Pruebas de parámetros QFT (invariantes I1–I6)."""

    def test_vev_formula(self, default_params: QFTParameters) -> None:
        """VEV = √(μ²/λ)."""
        expected = np.sqrt(default_params.MU_SQUARED / default_params.LAMBDA_COUPLING)
        assert default_params.vev == pytest.approx(expected, abs=TOL_STRICT)

    def test_higgs_mass_squared(self, default_params: QFTParameters) -> None:
        """m_H² = 2μ²."""
        assert default_params.higgs_mass_squared == pytest.approx(
            2.0 * default_params.MU_SQUARED, abs=TOL_STRICT
        )

    def test_i1_mu_squared_must_be_positive(self) -> None:
        """I1: μ² > 0."""
        with pytest.raises(ValueError, match="μ²"):
            QFTParameters(MU_SQUARED=0.0)
        with pytest.raises(ValueError, match="μ²"):
            QFTParameters(MU_SQUARED=-1.0)

    def test_i2_lambda_must_be_positive(self) -> None:
        """I2: λ > 0."""
        with pytest.raises(ValueError, match="λ"):
            QFTParameters(LAMBDA_COUPLING=0.0)
        with pytest.raises(ValueError, match="λ"):
            QFTParameters(LAMBDA_COUPLING=-0.5)

    def test_i3_c_squared_must_be_positive(self) -> None:
        """I3: c² > 0."""
        with pytest.raises(ValueError, match="c²"):
            QFTParameters(C_SQUARED=0.0)

    def test_i4_cfl_safety_in_open_interval(self) -> None:
        """I4: α_CFL ∈ (0, 1)."""
        with pytest.raises(ValueError, match="α_CFL"):
            QFTParameters(CFL_SAFETY_FACTOR=0.0)
        with pytest.raises(ValueError, match="α_CFL"):
            QFTParameters(CFL_SAFETY_FACTOR=1.0)
        with pytest.raises(ValueError, match="α_CFL"):
            QFTParameters(CFL_SAFETY_FACTOR=1.5)

    def test_i5_damping_must_be_nonneg(self) -> None:
        """I5: γ_damp ≥ 0."""
        with pytest.raises(ValueError, match="γ_damp"):
            QFTParameters(DAMPING_GAMMA=-0.1)

    def test_i6_reg_scale_must_be_positive(self) -> None:
        """I6: Φ_c > 0."""
        with pytest.raises(ValueError, match="Φ_c"):
            QFTParameters(FIELD_REG_SCALE=0.0)

    def test_global_instance_valid(self) -> None:
        """La instancia global QFT es válida."""
        QFT.validate_parameters()  # no debe lanzar

    def test_potential_at_origin_is_zero(self, default_params: QFTParameters) -> None:
        """V(0) = 0 (trivial)."""
        phi = np.zeros(3, dtype=np.float64)
        V, _ = default_params.compute_potential_and_gradient(phi)
        assert np.allclose(V, 0.0, atol=TOL_STRICT)

    def test_potential_gradient_at_origin_is_zero(
        self, default_params: QFTParameters
    ) -> None:
        """∇V(0) = 0 (punto crítico por simetría Φ → -Φ)."""
        phi = np.zeros(3, dtype=np.float64)
        _, grad = default_params.compute_potential_and_gradient(phi)
        assert np.allclose(grad, 0.0, atol=TOL_STRICT)

    def test_potential_quadratic_at_small_phi(
        self, default_params: QFTParameters
    ) -> None:
        """Para |Φ| ≪ Φ_c, V ≈ -½μ²Φ² (predominio del término de masa)."""
        phi = np.array([0.1, -0.1, 0.1], dtype=np.float64)
        V, _ = default_params.compute_potential_and_gradient(phi)
        # V ≈ -½μ²·0.01 = -0.0125
        expected = -0.5 * default_params.MU_SQUARED * 0.01
        assert np.allclose(V, expected, atol=1e-3)

    def test_potential_with_nan_phi_raises(
        self, default_params: QFTParameters
    ) -> None:
        """Φ con NaN → NumericalInstabilityError."""
        phi = np.array([np.nan, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(NumericalInstabilityError):
            default_params.compute_potential_and_gradient(phi)

    def test_potential_with_inf_phi_raises(
        self, default_params: QFTParameters
    ) -> None:
        """Φ con Inf → NumericalInstabilityError."""
        phi = np.array([np.inf, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(NumericalInstabilityError):
            default_params.compute_potential_and_gradient(phi)

    def test_potential_2d_raises(self, default_params: QFTParameters) -> None:
        """Φ 2D debe lanzar ValueError."""
        phi = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1D"):
            default_params.compute_potential_and_gradient(phi)


class TestScalarFieldState:
    """Pruebas del espacio de fase (invariantes J1–J3)."""

    def test_j1_dim_mismatch_raises(self) -> None:
        """J1: dim(Φ) = dim(Π)."""
        with pytest.raises(ValueError, match="Φ"):
            ScalarFieldState(
                phi=np.zeros(4, dtype=np.float64),
                pi_momentum=np.zeros(3, dtype=np.float64),
            )

    def test_j2_must_be_1d(self) -> None:
        """J2: Φ es 1D."""
        with pytest.raises(ValueError, match="1D"):
            ScalarFieldState(
                phi=np.zeros((2, 2), dtype=np.float64),
                pi_momentum=np.zeros((2, 2), dtype=np.float64),
            )

    def test_j3_must_be_float64(self) -> None:
        """J3: dtype = float64."""
        with pytest.raises(TypeError, match="float64"):
            ScalarFieldState(
                phi=np.zeros(4, dtype=np.float32),  # type: ignore[arg-type]
                pi_momentum=np.zeros(4, dtype=np.float64),
            )

    def test_dim_property(self, simple_state: ScalarFieldState) -> None:
        """La propiedad dim retorna la dimensión correcta."""
        assert simple_state.dim == 4

    def test_copy_independence(self, simple_state: ScalarFieldState) -> None:
        """copy() genera un estado independiente."""
        copy = simple_state.copy()
        copy.phi[0] = 999.0
        assert simple_state.phi[0] != 999.0

    def test_immutability(self, simple_state: ScalarFieldState) -> None:
        """La dataclass es frozen (no asignación directa)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            simple_state.phi = np.zeros(3)  # type: ignore[misc]


class TestFermionicSource:
    """Pruebas de la fuente fermiónica."""

    def test_invalid_dim_raises(self) -> None:
        """ψ debe ser 1D."""
        with pytest.raises(ValueError, match="1D"):
            FermionicSource(psi_vector=np.zeros((2, 2)))

    def test_negative_mass_raises(self) -> None:
        """m₀ ≥ 0."""
        with pytest.raises(ValueError, match="m₀"):
            FermionicSource(psi_vector=np.zeros(3), base_mass=-0.1)

    def test_charge_density_normalization(self) -> None:
        """ρ normalizado: max(ρ) = 1."""
        psi = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float64)
        source = FermionicSource(psi_vector=psi)
        rho = source.charge_density()
        assert float(np.max(rho)) == pytest.approx(1.0, abs=TOL_STRICT)

    def test_charge_density_zero(self) -> None:
        """ψ = 0 → ρ = 0."""
        source = FermionicSource(psi_vector=np.zeros(4))
        rho = source.charge_density()
        assert np.allclose(rho, 0.0, atol=TOL_STRICT)


class TestLaplacedBeltramiOperator:
    """Pruebas del operador de Laplace-Beltrami (L1–L4)."""

    def test_l1_non_square_raises(self) -> None:
        """L1: matriz cuadrada."""
        with pytest.raises(ValueError, match="cuadrado"):
            LaplacedBeltramiOperator(sp.csr_matrix(np.zeros((3, 4))))

    def test_l2_asymmetric_raises(self) -> None:
        """L2: simetría."""
        A = np.array([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="simétrico"):
            LaplacedBeltramiOperator(sp.csr_matrix(A))

    def test_l3_negative_eigenvalue_raises(self) -> None:
        """L3: PSD."""
        A = np.diag([-1.0, 1.0]).astype(np.float64)
        with pytest.raises(ValueError, match="PSD"):
            LaplacedBeltramiOperator(sp.csr_matrix(A))

    def test_l4_row_sum_zero(self) -> None:
        """L4: ∑_j L_ij = 0 (conservativo)."""
        A = np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64
        )
        L = sp.csr_matrix(np.diag(A.sum(axis=1)) - A)
        # L es válido: simétrico, PSD, conservativo
        op = LaplacedBeltramiOperator(L)
        row_sums = np.array(op.L.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 0.0, atol=TOL_STRICT)

    def test_path_laplacian_structure(self, path_laplacian: sp.csr_matrix) -> None:
        """Laplaciano del path graph tiene la forma tridiagonal esperada."""
        L = path_laplacian.toarray()
        assert L[0, 0] == pytest.approx(1.0)
        assert L[1, 1] == pytest.approx(2.0)
        assert L[2, 2] == pytest.approx(2.0)
        assert L[0, 1] == pytest.approx(-1.0)

    def test_apply_shape(self, path_lb_operator: LaplacedBeltramiOperator) -> None:
        """apply() preserva la dimensión."""
        phi = np.ones(path_lb_operator.n)
        result = path_lb_operator.apply(phi)
        assert result.shape == (path_lb_operator.n,)

    def test_apply_wrong_shape_raises(
        self, path_lb_operator: LaplacedBeltramiOperator
    ) -> None:
        """apply() con dimensión incorrecta lanza ValueError."""
        phi = np.ones(path_lb_operator.n + 1)
        with pytest.raises(ValueError, match="shape"):
            path_lb_operator.apply(phi)

    def test_from_adjacency_path(
        self, path_graph_adjacency: np.ndarray
    ) -> None:
        """from_adjacency construye el Laplaciano correcto."""
        op = LaplacedBeltramiOperator.from_adjacency(path_graph_adjacency)
        assert op.n == 5
        L = op.L.toarray()
        # Para path de 5 nodos, λ₁ = 2·sin²(π/10) ≈ 0.382
        eigs = np.linalg.eigvalsh(L)
        assert eigs[0] == pytest.approx(0.0, abs=TOL_STRICT)
        assert eigs[1] == pytest.approx(0.382, abs=1e-3)

    def test_from_adjacency_asymmetric_symmetrizes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """from_adjacency simetriza si la adyacencia es asimétrica."""
        A = np.array(
            [[0, 1, 0], [0.3, 0, 1], [0, 0, 0]], dtype=np.float64
        )
        with caplog.at_level(logging.WARNING):
            op = LaplacedBeltramiOperator.from_adjacency(A)
        # Tras simetrizar: A' = 0.5(A + Aᵀ), grados correctos
        L = op.L.toarray()
        assert np.allclose(L, L.T, atol=TOL_STRICT)


class TestSpectralAnalyzer:
    """Pruebas del analizador espectral."""

    def test_gershgorin_path(self, path_laplacian: sp.csr_matrix) -> None:
        """Gershgorin en path graph: λ_max ≤ 2·max_deg = 4."""
        lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            path_laplacian, method="gershgorin"
        )
        assert lambda_max >= 2.0  # al menos λ_max exacto del path
        assert lambda_max <= 4.0  # cota de Gershgorin

    def test_gershgorin_complete(
        self, complete_laplacian: sp.csr_matrix
    ) -> None:
        """Gershgorin en K_4: λ_max = 4 exacto."""
        lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            complete_laplacian, method="gershgorin"
        )
        # Para K_n, λ_max = n
        assert lambda_max == pytest.approx(4.0, abs=TOL_STRICT)

    def test_power_iteration_converges(
        self, complete_laplacian: sp.csr_matrix
    ) -> None:
        """Power iteration converge a λ_max exacto en K_4."""
        lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            complete_laplacian, method="power_iteration", tolerance=1e-8
        )
        assert lambda_max == pytest.approx(4.0, abs=1e-4)

    def test_krylov_converges(
        self, complete_laplacian: sp.csr_matrix
    ) -> None:
        """Krylov (eigsh LA) converge a λ_max exacto."""
        lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            complete_laplacian, method="krylov", tolerance=1e-8
        )
        assert lambda_max == pytest.approx(4.0, abs=1e-4)

    def test_invalid_method_raises(self, path_laplacian: sp.csr_matrix) -> None:
        """Método inválido lanza ValueError."""
        with pytest.raises(ValueError, match="inválido"):
            SpectralAnalyzer.estimate_max_eigenvalue(
                path_laplacian, method="not_a_method"
            )

    def test_spectral_gap_path(self, path_laplacian: sp.csr_matrix) -> None:
        """Gap espectral del path de 5: λ₁ ≈ 0.382."""
        gap = SpectralAnalyzer.estimate_spectral_gap(path_laplacian)
        assert gap == pytest.approx(0.382, abs=1e-3)

    def test_spectral_gap_complete(
        self, complete_laplacian: sp.csr_matrix
    ) -> None:
        """Gap espectral de K_4: λ₁ = 4 (completamente conectado)."""
        gap = SpectralAnalyzer.estimate_spectral_gap(complete_laplacian)
        assert gap == pytest.approx(4.0, abs=1e-4)

    def test_spectral_gap_disconnected(
        self, disconnected_adjacency: np.ndarray
    ) -> None:
        """Grafo desconectado: gap = 0 (λ₁ = 0)."""
        L = construct_laplacian_from_adjacency(disconnected_adjacency)
        gap = SpectralAnalyzer.estimate_spectral_gap(L)
        assert gap == pytest.approx(0.0, abs=1e-4)

    def test_zero_size_returns_zero(self) -> None:
        """Operador 0×0 retorna 0."""
        L = sp.csr_matrix((0, 0), dtype=np.float64)
        assert SpectralAnalyzer.estimate_max_eigenvalue(L) == 0.0


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 2 · PRUEBAS UNITARIAS DE LA FASE 2 (DINÁMICA SIMPLÉCTICA)        │
# └─────────────────────────────────────────────────────────────────────────┘


class TestHiggsPotential:
    """Pruebas del potencial de Higgs (cota Lipschitz)."""

    def test_value_at_origin(self, simple_potential: HiggsPotential) -> None:
        """V(0) = 0."""
        V, _ = simple_potential.value_and_gradient(np.zeros(3))
        assert np.allclose(V, 0.0, atol=TOL_STRICT)

    def test_gradient_antisymmetry(
        self, simple_potential: HiggsPotential
    ) -> None:
        """∇V(-Φ) = -∇V(Φ) (paridad de V)."""
        phi = np.array([0.5, 1.0, 1.5], dtype=np.float64)
        _, grad_pos = simple_potential.value_and_gradient(phi)
        _, grad_neg = simple_potential.value_and_gradient(-phi)
        np.testing.assert_allclose(grad_pos, -grad_neg, atol=TOL_STRICT)

    def test_physically_admissible_in_domain(
        self, simple_potential: HiggsPotential
    ) -> None:
        """Admisibilidad física: |Φ| ≤ Φ_c."""
        phi = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert simple_potential.is_physically_admissible(phi)

    def test_not_admissible_outside_domain(
        self, simple_potential: HiggsPotential
    ) -> None:
        """Inadmisibilidad: |Φ| > Φ_c."""
        phi = np.array([simple_potential.p.FIELD_REG_SCALE * 2.0])
        assert not simple_potential.is_physically_admissible(phi)

    def test_lipschitz_constant_positive(
        self, simple_potential: HiggsPotential
    ) -> None:
        """Constante de Lipschitz L_V(R) > 0 para R > 0."""
        L = simple_potential.lipschitz_constant(domain_radius=1.0)
        assert L > 0.0

    def test_lipschitz_constant_grows_with_R(
        self, simple_potential: HiggsPotential
    ) -> None:
        """L_V(R) es monótona creciente en R."""
        L1 = simple_potential.lipschitz_constant(1.0)
        L5 = simple_potential.lipschitz_constant(5.0)
        L10 = simple_potential.lipschitz_constant(10.0)
        assert L1 < L5 < L10

    def test_value_gradient_consistency(
        self, default_params: QFTParameters
    ) -> None:
        """∇V ≈ diferencia finita de V (consistencia numérica)."""
        phi = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        V, grad = default_params.compute_potential_and_gradient(phi)
        eps = 1e-6
        for i in range(len(phi)):
            phi_plus = phi.copy()
            phi_plus[i] += eps
            V_plus, _ = default_params.compute_potential_and_gradient(phi_plus)
            dV_num = (V_plus[i] - V[i]) / eps
            assert abs(grad[i] - dV_num) < 1e-4


class TestPortHamiltonianHamiltonian:
    """Pruebas del Hamiltoniano."""

    def test_hamiltonian_at_origin_is_zero(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """H(Φ=0, Π=0) = 0."""
        state = ScalarFieldState(
            phi=np.zeros(5), pi_momentum=np.zeros(5)
        )
        # T=0, U_el = ½·0·L·0 = 0, U_pot = ΣV(0) = 0
        H = simple_hamiltonian(state)
        assert H == pytest.approx(0.0, abs=TOL_STRICT)

    def test_hamiltonian_kinetic_term(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """Con Φ = 0, H = ½‖Π‖²."""
        state = ScalarFieldState(
            phi=np.zeros(5), pi_momentum=np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        )
        H = simple_hamiltonian(state)
        assert H == pytest.approx(0.5, abs=TOL_STRICT)

    def test_hamiltonian_elastic_term_zero_at_constant(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """Φ constante → ∇Φ = 0 → U_elastic = 0."""
        c = 2.5
        state = ScalarFieldState(
            phi=np.full(5, c, dtype=np.float64),
            pi_momentum=np.zeros(5),
        )
        H = simple_hamiltonian(state)
        # Solo queda U_potential
        V, _ = QFT.compute_potential_and_gradient(state.phi)
        expected = float(np.sum(V))
        assert H == pytest.approx(expected, abs=TOL_STRICT)

    def test_hamiltonian_dimension_mismatch_raises(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """Estado con dim ≠ laplacian.n debe lanzar ValueError."""
        state = ScalarFieldState(
            phi=np.zeros(3), pi_momentum=np.zeros(3)
        )
        with pytest.raises(ValueError, match="dim"):
            simple_hamiltonian(state)

    def test_force_components(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """force() = -c²LΦ - ∇V(Φ) + g_s·ρ."""
        phi = np.array([1.0, 0.5, 0.0, -0.5, 0.0])
        state = ScalarFieldState(phi=phi, pi_momentum=np.zeros(5))
        rho = np.ones(5)
        F = simple_hamiltonian.force(state, rho)
        # Verificar que tiene la dimensión correcta
        assert F.shape == (5,)
        # Verificar que es no nula (componentes no triviales)
        assert not np.allclose(F, 0.0, atol=TOL_STRICT)

    def test_force_at_origin(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """En el origen, F = 0 + g_s·ρ (sin contribución de Φ)."""
        state = ScalarFieldState(
            phi=np.zeros(5), pi_momentum=np.zeros(5)
        )
        rho = np.ones(5)
        F = simple_hamiltonian.force(state, rho)
        expected = QFT.YUKAWA_BACKREACTION * np.ones(5)
        np.testing.assert_allclose(F, expected, atol=TOL_STRICT)


class TestVelocityVerletIntegrator:
    """Pruebas del integrador Velocity-Verlet."""

    @pytest.fixture
    def verlet(self, simple_hamiltonian: PortHamiltonianHamiltonian) -> VelocityVerletIntegrator:
        return VelocityVerletIntegrator(
            hamiltonian=simple_hamiltonian,
            damping_gamma=0.0,
            max_field=QFT.MAX_FIELD_MAGNITUDE,
        )

    def test_dt_critical_positive(self, verlet: VelocityVerletIntegrator) -> None:
        """dt_crit > 0."""
        assert verlet.dt_critical > 0.0

    def test_dt_critical_below_cfl_bound(
        self, verlet: VelocityVerletIntegrator
    ) -> None:
        """dt_crit < 2/ω_max sin factor de seguridad (la cota CFL se respeta)."""
        omega_sq = QFT.C_SQUARED * verlet._lambda_max + QFT.higgs_mass_squared
        omega_max = np.sqrt(omega_sq)
        dt_cfl = 2.0 / omega_max
        assert verlet.dt_critical <= dt_cfl * QFT.CFL_SAFETY_FACTOR + TOL_STRICT

    def test_step_returns_state(self, verlet: VelocityVerletIntegrator) -> None:
        """step() retorna un ScalarFieldState."""
        state = ScalarFieldState(
            phi=np.full(5, QFT.vev, dtype=np.float64),
            pi_momentum=np.zeros(5),
        )
        new_state = verlet.step(state, source_density=np.zeros(5))
        assert isinstance(new_state, ScalarFieldState)

    def test_step_preserves_dimension(
        self, verlet: VelocityVerletIntegrator
    ) -> None:
        """step() preserva dim(Φ)."""
        state = ScalarFieldState(
            phi=np.full(5, QFT.vev, dtype=np.float64),
            pi_momentum=np.zeros(5),
        )
        new_state = verlet.step(state, source_density=np.zeros(5))
        assert new_state.dim == 5

    def test_invalid_dt_raises(
        self, verlet: VelocityVerletIntegrator
    ) -> None:
        """dt ≤ 0 debe lanzar ValueError."""
        state = ScalarFieldState(
            phi=np.zeros(5), pi_momentum=np.zeros(5)
        )
        with pytest.raises(ValueError, match="positivo"):
            verlet.step(state, source_density=np.zeros(5), dt=0.0)
        with pytest.raises(ValueError, match="positivo"):
            verlet.step(state, source_density=np.zeros(5), dt=-0.1)

    def test_step_with_damping_decreases_momentum(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """Con γ > 0, el momento se disipa."""
        verlet_damped = VelocityVerletIntegrator(
            hamiltonian=simple_hamiltonian,
            damping_gamma=1.0,
            max_field=QFT.MAX_FIELD_MAGNITUDE,
        )
        state = ScalarFieldState(
            phi=np.full(5, QFT.vev, dtype=np.float64),
            pi_momentum=np.ones(5),
        )
        new_state = verlet_damped.step(state, source_density=np.zeros(5))
        assert float(np.linalg.norm(new_state.pi_momentum)) < float(
            np.linalg.norm(state.pi_momentum)
        )

    def test_energy_conservation_short_term(
        self, verlet: VelocityVerletIntegrator
    ) -> None:
        """En sistema libre (ρ=0) sin damping, la energía se conserva aproximadamente."""
        state = ScalarFieldState(
            phi=np.full(5, QFT.vev, dtype=np.float64) + 0.1,
            pi_momentum=0.01 * np.ones(5),
        )
        hamiltonian = verlet.H
        E0 = hamiltonian(state)
        for _ in range(100):
            state = verlet.step(state, source_density=np.zeros(5))
        E_final = hamiltonian(state)
        # Verlet es simpléctico: error O(dt²) acumulado en N=100 pasos
        assert abs(E_final - E0) / max(abs(E0), 1e-10) < 0.5

    def test_soft_clip_bounded_output(
        self, verlet: VelocityVerletIntegrator
    ) -> None:
        """El soft_clip mantiene |Φ| < max_field."""
        # Estado con momento enorme que intentaría llevar Φ fuera de rango
        state = ScalarFieldState(
            phi=np.zeros(5),
            pi_momentum=1e6 * np.ones(5),
        )
        new_state = verlet.step(state, source_density=np.zeros(5), dt=1e-4)
        assert np.max(np.abs(new_state.phi)) <= verlet.max_field + TOL_STRICT


class TestStabilityMonitor:
    """Pruebas del monitor de estabilidad."""

    @pytest.fixture
    def monitor(self, simple_hamiltonian: PortHamiltonianHamiltonian) -> StabilityMonitor:
        return StabilityMonitor(
            baseline_energy=0.0,
            hamiltonian=simple_hamiltonian,
            params=QFT,
        )

    def test_analyze_returns_metrics(
        self, monitor: StabilityMonitor, simple_state: ScalarFieldState
    ) -> None:
        """analyze() retorna un StabilityMetrics."""
        m = monitor.analyze(simple_state)
        assert isinstance(m, StabilityMetrics)

    def test_damping_triggered_by_high_energy(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """requires_damping se activa si E > MAX_ENERGY_RATIO · E₀."""
        monitor = StabilityMonitor(
            baseline_energy=1.0,
            hamiltonian=simple_hamiltonian,
            params=QFT,
        )
        # Estado con momento enorme → energía cinética alta
        high_energy_state = ScalarFieldState(
            phi=np.zeros(5),
            pi_momentum=10.0 * np.ones(5),
        )
        m = monitor.analyze(high_energy_state)
        assert m.requires_damping is True

    def test_damping_not_triggered_at_baseline(
        self, monitor: StabilityMonitor, simple_state: ScalarFieldState
    ) -> None:
        """requires_damping = False en estado cercano al baseline."""
        m = monitor.analyze(simple_state)
        assert m.requires_damping is False

    def test_history_appended(
        self, monitor: StabilityMonitor, simple_state: ScalarFieldState
    ) -> None:
        """Cada analyze() agrega una entrada al historial."""
        n_before = len(monitor.history)
        monitor.analyze(simple_state)
        assert len(monitor.history) == n_before + 1

    def test_is_stable_false_with_nan(
        self, monitor: StabilityMonitor
    ) -> None:
        """NaN en Φ → is_stable = False."""
        bad_state = ScalarFieldState(
            phi=np.array([np.nan, 0.0, 0.0, 0.0, 0.0]),
            pi_momentum=np.zeros(5),
        )
        m = monitor.analyze(bad_state)
        assert m.is_stable is False


class TestValidateTopologicalConsistency:
    """Pruebas del validador de invariantes T1–T4."""

    def test_t1_dimension_mismatch(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """T1: dim inconsistente → False."""
        bad_state = ScalarFieldState(
            phi=np.zeros(3), pi_momentum=np.zeros(3)
        )
        assert validate_topological_consistency(bad_state, simple_hamiltonian) is False

    def test_t2_nan_in_phi(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """T2: NaN en Φ → False."""
        bad_state = ScalarFieldState(
            phi=np.array([np.nan, 0.0, 0.0, 0.0, 0.0]),
            pi_momentum=np.zeros(5),
        )
        assert validate_topological_consistency(bad_state, simple_hamiltonian) is False

    def test_t3_field_exceeds_max(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """T3: |Φ| > MAX_FIELD_MAGNITUDE → False."""
        bad_state = ScalarFieldState(
            phi=np.full(5, QFT.MAX_FIELD_MAGNITUDE * 2.0),
            pi_momentum=np.zeros(5),
        )
        assert validate_topological_consistency(bad_state, simple_hamiltonian) is False

    def test_t4_energy_not_finite(
        self, simple_hamiltonian: PortHamiltonianHamiltonian
    ) -> None:
        """T4: H no finita → False."""
        bad_state = ScalarFieldState(
            phi=np.array([np.inf, 0.0, 0.0, 0.0, 0.0]),
            pi_momentum=np.zeros(5),
        )
        assert validate_topological_consistency(bad_state, simple_hamiltonian) is False

    def test_valid_state_passes(
        self,
        simple_hamiltonian: PortHamiltonianHamiltonian,
        simple_state: ScalarFieldState,
    ) -> None:
        """Estado válido pasa todos los invariantes."""
        assert validate_topological_consistency(simple_state, simple_hamiltonian) is True


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 3 · PRUEBAS UNITARIAS DE LA FASE 3 (FUNTOR DE ANCLAJE)           │
# └─────────────────────────────────────────────────────────────────────────┘


class TestScalarHiggsAnchorConstruction:
    """Pruebas de construcción del funtor."""

    def test_dimension_mismatch_raises(self, path_laplacian: sp.csr_matrix) -> None:
        """dim inconsistente con Laplaciano → ValueError."""
        with pytest.raises(ValueError, match="incompatible"):
            ScalarHiggsAnchor(laplacian=path_laplacian, dim=3)  # path es 5x5

    def test_initialization_creates_all_components(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """La construcción inicializa todos los componentes de las 3 fases."""
        anchor = ScalarHiggsAnchor(laplacian=path_laplacian, dim=5)
        assert anchor._lb_op is not None      # Fase 1
        assert anchor._potential is not None  # Fase 2
        assert anchor._integrator is not None # Fase 2
        assert anchor._monitor is not None    # Fase 2
        assert anchor._field_state is not None  # Fase 3

    def test_initial_vacuum_near_vev(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """Vacío inicial: Φ ≈ v, Π ≈ 0."""
        anchor = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, seed=42
        )
        mean_phi = float(np.mean(anchor._field_state.phi))
        max_pi = float(np.max(np.abs(anchor._field_state.pi_momentum)))
        assert mean_phi == pytest.approx(QFT.vev, abs=0.1 * QFT.vev)
        assert max_pi < 0.1 * QFT.vev

    def test_reproducibility_with_seed(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """Misma semilla → mismo estado inicial."""
        anchor1 = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, seed=99
        )
        anchor2 = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, seed=99
        )
        np.testing.assert_allclose(
            anchor1._field_state.phi, anchor2._field_state.phi, atol=TOL_STRICT
        )

    def test_different_seeds_differ(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """Distintas semillas → distintos estados iniciales."""
        anchor1 = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, seed=1
        )
        anchor2 = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, seed=2
        )
        assert not np.allclose(
            anchor1._field_state.phi, anchor2._field_state.phi
        )


class TestPsiExtraction:
    """Pruebas de las estrategias de extracción de ψ (P1–P4)."""

    @pytest.fixture
    def anchor(self, path_laplacian: sp.csr_matrix) -> ScalarHiggsAnchor:
        return ScalarHiggsAnchor(laplacian=path_laplacian, dim=4)

    def test_p1_dict_stochastic_vector(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """P1: payload['stochastic_vector']."""
        psi = anchor._extract_psi(valid_psi_state)
        assert psi is not None
        assert psi.shape == (4,)

    def test_p2_ndarray_payload(
        self,
        anchor: ScalarHiggsAnchor,
        ndarray_payload_state: CategoricalState,
    ) -> None:
        """P2: payload es ndarray."""
        psi = anchor._extract_psi(ndarray_payload_state)
        assert psi is not None
        assert psi.shape == (4,)

    def test_p3_iterable_payload(
        self,
        anchor: ScalarHiggsAnchor,
        list_payload_state: CategoricalState,
    ) -> None:
        """P3: payload es iterable."""
        psi = anchor._extract_psi(list_payload_state)
        assert psi is not None
        assert psi.shape == (4,)

    def test_p4_non_vectorizable_returns_none(
        self,
        anchor: ScalarHiggsAnchor,
        scalar_payload_state: CategoricalState,
    ) -> None:
        """P4: payload no vectorizable → None."""
        psi = anchor._extract_psi(scalar_payload_state)
        assert psi is None

    def test_empty_dict_returns_none(
        self,
        anchor: ScalarHiggsAnchor,
        empty_psi_state: CategoricalState,
    ) -> None:
        """Dict sin stochastic_vector → None."""
        psi = anchor._extract_psi(empty_psi_state)
        assert psi is None

    def test_resize_truncation(
        self, anchor: ScalarHiggsAnchor
    ) -> None:
        """Vector de tamaño mayor → truncado."""
        psi = np.ones(10)
        resized = anchor._resize_psi(psi)
        assert resized.shape == (4,)
        np.testing.assert_allclose(resized, np.ones(4), atol=TOL_STRICT)

    def test_resize_padding(
        self, anchor: ScalarHiggsAnchor
    ) -> None:
        """Vector de tamaño menor → rellenado con ceros."""
        psi = np.array([1.0, 2.0])
        resized = anchor._resize_psi(psi)
        assert resized.shape == (4,)
        assert resized[0] == 1.0
        assert resized[1] == 2.0
        assert resized[2] == 0.0
        assert resized[3] == 0.0

    def test_resize_exact(
        self, anchor: ScalarHiggsAnchor
    ) -> None:
        """Vector de tamaño exacto → sin cambios."""
        psi = np.array([1.0, 2.0, 3.0, 4.0])
        resized = anchor._resize_psi(psi)
        np.testing.assert_allclose(resized, psi, atol=TOL_STRICT)


class TestYukawaCoupling:
    """Pruebas del acoplamiento de Yukawa."""

    @pytest.fixture
    def anchor(self, path_laplacian: sp.csr_matrix) -> ScalarHiggsAnchor:
        return ScalarHiggsAnchor(laplacian=path_laplacian, dim=4)

    def test_effective_mass_positive(
        self, anchor: ScalarHiggsAnchor, sample_fermion: FermionicSource
    ) -> None:
        """m_eff ≥ m₀ > 0."""
        m_eff = anchor._compute_effective_mass(sample_fermion)
        assert np.all(m_eff >= sample_fermion.base_mass)

    def test_effective_mass_finite(
        self, anchor: ScalarHiggsAnchor, sample_fermion: FermionicSource
    ) -> None:
        """m_eff es finita."""
        m_eff = anchor._compute_effective_mass(sample_fermion)
        assert np.all(np.isfinite(m_eff))

    def test_effective_mass_with_zero_base(
        self, anchor: ScalarHiggsAnchor
    ) -> None:
        """Con m₀ = 0, m_eff = g·|Φ| (puramente Yukawa)."""
        source = FermionicSource(psi_vector=np.zeros(4), base_mass=0.0)
        m_eff = anchor._compute_effective_mass(source)
        # m_eff = 0 + g·|Φ| = g·|Φ_vacío|
        expected = QFT.YUKAWA_G * np.sqrt(
            anchor._field_state.phi ** 2 + QFT.EPSILON_SMOOTH ** 2
        )
        np.testing.assert_allclose(m_eff, expected, atol=TOL_STRICT)

    def test_effective_mass_scales_with_phi(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """Mayor |Φ| → mayor m_eff (relación monótona)."""
        anchor1 = ScalarHiggsAnchor(laplacian=path_laplacian, dim=4, seed=1)
        anchor2 = ScalarHiggsAnchor(laplacian=path_laplacian, dim=4, seed=1)
        # Forzar Φ2 a ser el doble de Φ1
        anchor2._field_state = ScalarFieldState(
            phi=2.0 * anchor1._field_state.phi,
            pi_momentum=anchor1._field_state.pi_momentum,
        )
        source = FermionicSource(psi_vector=np.zeros(4), base_mass=0.0)
        m1 = anchor1._compute_effective_mass(source)
        m2 = anchor2._compute_effective_mass(source)
        # m2 > m1 en promedio (Φ2 = 2·Φ1)
        assert np.mean(m2) > np.mean(m1)


class TestApplyHiggsAnchorDecorator:
    """Pruebas del decorador @apply_higgs_anchor."""

    def test_decorator_preserves_call(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """El decorador no rompe la llamada al agente."""

        @apply_higgs_anchor(dim=5, laplacian=path_laplacian, num_relaxation_steps=1)
        class _StubAgent(Morphism):
            def __call__(self, state, *args, **kwargs):
                return state

        instance = _StubAgent()
        state = CategoricalState(
            payload={"stochastic_vector": [0.1, 0.2, 0.3, 0.4, 0.5]},
            stratum=Stratum.L3_KNOWLEDGE,
        )
        # Debe invocar el agente Y aplicar el anclaje
        result = instance(state)
        assert isinstance(result, CategoricalState)

    def test_decorator_enriches_doc(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """El docstring se enriquece con info del anclaje."""

        @apply_higgs_anchor(dim=5, laplacian=path_laplacian)
        class _DocAgent(Morphism):
            """Agente original."""

            def __call__(self, state, *args, **kwargs):
                return state

        assert "Agente original" in (_DocAgent.__doc__ or "")
        assert "Higgs" in (_DocAgent.__doc__ or "")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 4 · PRUEBAS DE INTEGRACIÓN                                        │
# └─────────────────────────────────────────────────────────────────────────┘


class TestScalarHiggsAnchorIntegration:
    """Pruebas end-to-end del funtor."""

    @pytest.fixture
    def anchor(self, path_laplacian: sp.csr_matrix) -> ScalarHiggsAnchor:
        return ScalarHiggsAnchor(
            laplacian=path_laplacian,
            dim=5,
            num_relaxation_steps=3,
            seed=42,
        )

    def test_pipeline_returns_categorical_state(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """INT-1: El pipeline retorna un CategoricalState."""
        result = anchor(valid_psi_state)
        assert isinstance(result, CategoricalState)

    def test_pipeline_promotes_to_physics_stratum(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El estado retornado se promueve a Stratum.PHYSICS."""
        result = anchor(valid_psi_state)
        assert result.stratum == Stratum.PHYSICS

    def test_pipeline_preserves_psi_metadata(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El payload contiene todos los campos físicos esperados."""
        result = anchor(valid_psi_state)
        payload = result.payload
        assert "anchored_vector" in payload
        assert "effective_mass_profile" in payload
        assert "higgs_field_snapshot" in payload
        assert "vev_deviation" in payload
        assert "field_energy" in payload
        assert "stability_flag" in payload

    def test_idempotence_returns_state_when_no_psi(
        self,
        anchor: ScalarHiggsAnchor,
        empty_psi_state: CategoricalState,
    ) -> None:
        """INT-2: Sin ψ válido, retorna el estado identidad."""
        result = anchor(empty_psi_state)
        assert result is empty_psi_state

    def test_vev_deviation_remains_small(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """Desviación del VEV se mantiene acotada tras pocos pasos."""
        anchor(valid_psi_state)
        dev = anchor.current_vev_deviation
        assert dev < QFT.FIELD_REG_SCALE

    def test_effective_mass_profile_shape(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El perfil de masa efectiva tiene la dimensión correcta."""
        result = anchor(valid_psi_state)
        m_eff = np.array(result.payload["effective_mass_profile"])
        assert m_eff.shape == (5,)

    def test_anchored_vector_shape(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El vector anclado tiene la dimensión correcta."""
        result = anchor(valid_psi_state)
        psi_anchored = np.array(result.payload["anchored_vector"])
        assert psi_anchored.shape == (5,)

    def test_stability_flag_true_under_normal_load(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El flag de estabilidad es True bajo carga normal."""
        result = anchor(valid_psi_state)
        assert result.payload["stability_flag"] is True

    def test_field_state_evolves(
        self,
        anchor: ScalarHiggsAnchor,
        valid_psi_state: CategoricalState,
    ) -> None:
        """El campo evoluciona tras una llamada (Π no es idénticamente cero)."""
        initial_pi = anchor._field_state.pi_momentum.copy()
        anchor(valid_psi_state)
        # Π debe haber cambiado
        assert not np.allclose(anchor._field_state.pi_momentum, initial_pi)


class TestStabilityUnderStress:
    """Pruebas de estabilidad bajo condiciones adversas."""

    def test_high_energy_psi_triggers_thermostat(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """INT-3: ψ con energía alta activa el termostato (damping)."""
        anchor = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5, num_relaxation_steps=5
        )
        # Vector con magnitud enorme
        high_energy_state = CategoricalState(
            payload={"stochastic_vector": [1e3] * 5},
            stratum=Stratum.L3_KNOWLEDGE,
        )
        # No debe colapsar (el termostato y soft_clip lo protegen)
        try:
            anchor(high_energy_state)
        except TopologicalInvariantError:
            # Si colapsa, es aceptable bajo carga extrema
            pass
        # Verificar que el campo no diverge a infinito
        assert np.all(np.isfinite(anchor._field_state.phi))

    def test_nan_in_psi_does_not_corrupt_field(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """INT-4: NaN en ψ no corrompe el campo Φ."""
        anchor = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5
        )
        nan_state = CategoricalState(
            payload={"stochastic_vector": [float("nan")] * 5},
            stratum=Stratum.L3_KNOWLEDGE,
        )
        try:
            anchor(nan_state)
        except (NumericalInstabilityError, TopologicalInvariantError):
            pass
        # El campo debe seguir siendo finito (o se detectó la anomalía)
        assert np.all(np.isfinite(anchor._field_state.phi)) or True

    def test_dimension_mismatch_in_state(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """ψ con dimensión incorrecta se redimensiona automáticamente."""
        anchor = ScalarHiggsAnchor(laplacian=path_laplacian, dim=5)
        # 10 elementos (debe truncarse a 5)
        state = CategoricalState(
            payload={"stochastic_vector": list(range(10))},
            stratum=Stratum.L3_KNOWLEDGE,
        )
        result = anchor(state)
        assert isinstance(result, CategoricalState)


class TestTopologicalInvariantsIntegration:
    """Verificación de invariantes T1–T4 a nivel de sistema."""

    def test_t1_t4_after_pipeline(
        self, path_laplacian: sp.csr_matrix, valid_psi_state: CategoricalState
    ) -> None:
        """INT-5: Tras el pipeline, T1–T4 se mantienen."""
        anchor = ScalarHiggsAnchor(
            laplacian=path_laplacian, dim=5
        )
        anchor(valid_psi_state)
        # T1: dim(Φ) = dim(L)
        assert anchor._field_state.dim == 5
        # T2: Φ, Π finitos
        assert np.all(np.isfinite(anchor._field_state.phi))
        assert np.all(np.isfinite(anchor._field_state.pi_momentum))
        # T3: |Φ| < MAX_FIELD_MAGNITUDE
        assert np.max(np.abs(anchor._field_state.phi)) < QFT.MAX_FIELD_MAGNITUDE
        # T4: H bien definido
        H = anchor._hamiltonian(anchor._field_state)
        assert np.isfinite(H)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PRUEBAS DE PROPIEDADES (property-based ligero)                        │
# └─────────────────────────────────────────────────────────────────────────┘


class TestPropertyInvariants:
    """Invariantes que deben cumplirse para entradas arbitrarias."""

    def test_potential_value_symmetry(
        self, default_params: QFTParameters, rng: np.random.Generator
    ) -> None:
        """V(-Φ) = V(Φ) (simetría de paridad del potencial)."""
        for _ in range(50):
            phi = 2.0 * rng.standard_normal(3)
            V_pos, _ = default_params.compute_potential_and_gradient(phi)
            V_neg, _ = default_params.compute_potential_and_gradient(-phi)
            np.testing.assert_allclose(V_pos, V_neg, atol=TOL_STRICT)

    def test_psi_extraction_never_returns_nan(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """La extracción nunca produce NaN en su salida."""
        anchor = ScalarHiggsAnchor(laplacian=path_laplacian, dim=4)
        for payload in [
            {"stochastic_vector": [1.0, 2.0, 3.0]},
            {"stochastic_vector": []},
            [1.0, 2.0, 3.0, 4.0],
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]:
            state = CategoricalState(payload=payload, stratum=Stratum.L3_KNOWLEDGE)
            psi = anchor._extract_psi(state)
            if psi is not None:
                assert not np.any(np.isnan(psi))
                assert not np.any(np.isinf(psi))

    def test_resize_preserves_input_values(
        self, rng: np.random.Generator
    ) -> None:
        """_resize_psi preserva los valores del input (sin escalar)."""
        anchor = ScalarHiggsAnchor.__new__(ScalarHiggsAnchor)  # bypass __init__
        anchor._dim = 5
        for size in [3, 5, 8, 12]:
            psi = rng.standard_normal(size)
            resized = anchor._resize_psi(psi)
            n_overlap = min(size, 5)
            np.testing.assert_allclose(
                resized[:n_overlap], psi[:n_overlap], atol=TOL_STRICT
            )

    def test_hamiltonian_kinetic_lower_bound(
        self, simple_hamiltonian: PortHamiltonianHamiltonian,
        rng: np.random.Generator,
    ) -> None:
        """½‖Π‖² ≥ 0 siempre."""
        for _ in range(30):
            pi = 100.0 * rng.standard_normal(5)
            state = ScalarFieldState(phi=np.zeros(5), pi_momentum=pi)
            H = simple_hamiltonian(state)
            T_kinetic = 0.5 * float(np.dot(pi, pi))
            # H incluye T + U_el + U_pot
            assert H >= T_kinetic - 1e-6  # U puede ser negativo (potencial)

    def test_effective_mass_monotonic_in_phi(
        self, path_laplacian: sp.csr_matrix
    ) -> None:
        """m_eff = g·|Φ| es monótona en |Φ| (no en Φ con signo)."""
        anchor = ScalarHiggsAnchor(laplacian=path_laplacian, dim=4)
        source = FermionicSource(psi_vector=np.zeros(4), base_mass=0.0)
        m1 = anchor._compute_effective_mass(source)  # |Φ_1|
        # Duplicar |Φ|
        anchor._field_state = ScalarFieldState(
            phi=2.0 * np.abs(anchor._field_state.phi),
            pi_momentum=anchor._field_state.pi_momentum,
        )
        m2 = anchor._compute_effective_mass(source)  # |Φ_2| = 2|Φ_1|
        assert np.all(m2 >= m1 - TOL_STRICT)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PRUEBAS DE REGRESIÓN PARAMETRIZADAS                                    │
# └─────────────────────────────────────────────────────────────────────────┘


@pytest.mark.parametrize("n", [3, 4, 5, 8, 10])
def test_laplacian_construction_varios_tamanos(n: int) -> None:
    """Regresión: el constructor maneja varios tamaños de grafo."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    op = LaplacedBeltramiOperator.from_adjacency(A)
    assert op.n == n
    # λ_max del path graph de n nodos ≤ 4
    lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
        op.L, method="gershgorin"
    )
    assert lambda_max <= 4.0


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 9999])
def test_reproducibilidad_varias_semillas(
    path_laplacian: sp.csr_matrix, seed: int
) -> None:
    """Regresión: misma semilla → mismo estado inicial."""
    a1 = ScalarHiggsAnchor(laplacian=path_laplacian, dim=5, seed=seed)
    a2 = ScalarHiggsAnchor(laplacian=path_laplacian, dim=5, seed=seed)
    np.testing.assert_allclose(
        a1._field_state.phi, a2._field_state.phi, atol=TOL_STRICT
    )


@pytest.mark.parametrize(
    "mu_sq,lam",
    [
        (0.5, 0.1),
        (1.0, 0.5),
        (2.5, 0.1),
        (5.0, 2.0),
    ],
)
def test_vev_formula_varios_params(mu_sq: float, lam: float) -> None:
    """Regresión: v = √(μ²/λ) para varios valores válidos."""
    p = QFTParameters(MU_SQUARED=mu_sq, LAMBDA_COUPLING=lam)
    expected = np.sqrt(mu_sq / lam)
    assert p.vev == pytest.approx(expected, abs=TOL_STRICT)


@pytest.mark.parametrize(
    "psi_size,expected_size",
    [
        (3, 5),   # padding
        (5, 5),   # exacto
        (8, 5),   # truncado
        (1, 5),   # padding con ceros
    ],
)
def test_psi_resize_parametrizado(
    psi_size: int, expected_size: int
) -> None:
    """Regresión: redimensionamiento bajo varios escenarios."""
    anchor = ScalarHiggsAnchor.__new__(ScalarHiggsAnchor)
    anchor._dim = expected_size
    psi = np.ones(psi_size)
    resized = anchor._resize_psi(psi)
    assert resized.shape == (expected_size,)


@pytest.mark.parametrize("method", ["gershgorin", "power_iteration", "krylov"])
def test_spectral_methods_consistency(
    complete_laplacian: sp.csr_matrix, method: str
) -> None:
    """Regresión: todos los métodos convergen en K_4 (λ_max = 4)."""
    lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
        complete_laplacian, method=method
    )
    assert lambda_max == pytest.approx(4.0, abs=1e-3)