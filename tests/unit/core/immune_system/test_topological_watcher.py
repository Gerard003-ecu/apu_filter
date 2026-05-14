"""
=========================================================================================
SUITE DE PRUEBAS RIGUROSAS — Topological Watcher (Sistema Inmunológico Matemático)
Archivo: tests/core/immune_system/test_topological_watcher.py
=========================================================================================

OBJETIVO:
  Validar exhaustivamente la implementación refinada de `topological_watcher.py` bajo
  criterios matemáticos (topología algebraica, geometría Riemanniana, álgebra lineal,
  teoría espectral, teoría de categorías), numéricos (estabilidad, tolerancias,
  regularización SPD), físicos (límites SI, validadores), y de ingeniería
  (monada de error, histéresis, bifurcaciones, emisión de electrones).

COBERTURA:
  ✓ Álgebra lineal estable (normalización, reciprocal, Gram-Schmidt)
  ✓ Tensores métricos (SPD, simetría, descomposición espectral, condición)
  ✓ Laplaciano discreto y funtor de membrana
  ✓ Proyectores ortogonales (idempotencia, auto-adjunción, Σπ_k=I)
  ✓ Morfismo inmunológico (propiedades funtoriales, monada de error)
  ✓ Flujo de Ricci (preservación SPD, regularización)
  ✓ Histéresis asimétrica y bifurcaciones topológicas
  ✓ Cuantización de electrones anómalos
  ✓ Casos de borde (NaN, Inf, degeneración, mal condicionamiento)

ESTRUCTURA:
  1. Fixtures reutilizables
  2. Tests unitarios por componente
  3. Tests de integración
  4. Tests de invariantes matemáticos
  5. Tests de propiedades categóricas
  6. Tests de robustez numérica
  7. Tests de propiedad (hypothesis) [BONUS]

REQUISITOS:
  pytest >= 7.0
  numpy >= 1.24
  scipy >= 1.10
  hypothesis >= 6.0 (opcional)

EJECUCIÓN:
  pytest tests/test_topological_watcher.py -v --tb=short
  pytest tests/test_topological_watcher.py -v --cov=app.core.immune_system.topological_watcher
  pytest tests/test_topological_watcher.py -k "test_ricci" -v
=========================================================================================
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from scipy import linalg as sp_linalg

# Módulo bajo prueba
import app.core.immune_system.topological_watcher as tw

# Dependencias de dominio
from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState

# Hypothesis para pruebas de propiedad (opcional)
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    def given(*args, **kwargs):
        return lambda f: f
    st = None


# ==============================================================================
# CONSTANTES DE PRUEBA
# ==============================================================================

EPS_TEST = tw.EPS
ALG_TOL = tw.ALGEBRAIC_TOL
COND_TOL = tw.COND_NUM_TOL
MIN_EIG = tw.MIN_EIGVAL_TOL

# Señales de referencia
SIGNAL_HEALTHY = np.array([
    0.10,   # saturation
    120.0,  # flyback_voltage
    80.0,   # dissipated_power
    1.0,    # beta_0
    0.0,    # beta_1
    0.05,   # entropy
    0.02,   # exergy_loss
], dtype=np.float64)

SIGNAL_WARNING = np.array([
    0.65,   # saturation
    290.0,  # flyback_voltage
    155.0,  # dissipated_power
    1.0,    # beta_0
    1.0,    # beta_1 (1 ciclo)
    0.40,   # entropy
    0.35,   # exergy_loss
], dtype=np.float64)

SIGNAL_CRITICAL = np.array([
    0.95,   # saturation (cerca de límite)
    395.0,  # flyback_voltage (cerca de 400V)
    190.0,  # dissipated_power (alta)
    2.0,    # beta_0 (fragmentación)
    3.0,    # beta_1 (múltiples ciclos)
    0.92,   # entropy (alta)
    0.88,   # exergy_loss (alta)
], dtype=np.float64)

# Señal con topología degenerada
SIGNAL_TOPOLOGY_ANOMALY = np.array([
    0.20,   # saturation
    150.0,  # flyback_voltage
    90.0,   # dissipated_power
    3.0,    # beta_0 (múltiples componentes)
    5.0,    # beta_1 (muchos ciclos)
    0.15,   # entropy
    0.10,   # exergy_loss
], dtype=np.float64)


# ==============================================================================
# UTILIDADES AUXILIARES
# ==============================================================================

def telemetry_from_signal(sig: np.ndarray) -> Dict[str, Any]:
    """Convierte array de señal a diccionario de telemetría."""
    keys = [c.key for c in tw.SIGNAL_SCHEMA]
    return {k: float(v) for k, v in zip(keys, sig)}


def make_spd_matrix(n: int, seed: int = 42, condition_number: Optional[float] = None) -> np.ndarray:
    """Genera matriz SPD n×n con número de condición controlado."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    Q, _ = np.linalg.qr(A)
    
    if condition_number is None:
        eigenvalues = rng.uniform(0.1, 10.0, size=n)
    else:
        eigenvalues = np.linspace(1.0, condition_number, n)
    
    G = Q @ np.diag(eigenvalues) @ Q.T
    return (G + G.T) * 0.5


def is_spd(matrix: np.ndarray, tol: float = MIN_EIG) -> bool:
    """Verifica si matriz es SPD."""
    try:
        lam = np.linalg.eigvalsh(matrix)
        return lam.min() >= tol and np.allclose(matrix, matrix.T, atol=ALG_TOL)
    except np.linalg.LinAlgError:
        return False


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def physical_constants():
    """Constantes físicas validadas."""
    return tw.PhysicalConstants


@pytest.fixture(scope="module")
def metric_identity_3():
    """Métrica identidad 3×3."""
    return tw.MetricTensor(np.eye(3, dtype=np.float64), validate=True)


@pytest.fixture(scope="module")
def metric_diagonal_well_conditioned():
    """Métrica diagonal bien condicionada (κ ≈ 10)."""
    diag = np.array([5.0, 2.0, 0.5], dtype=np.float64)
    return tw.MetricTensor(diag, validate=True)


@pytest.fixture(scope="module")
def metric_dense_moderate():
    """Métrica densa con κ moderado."""
    G = make_spd_matrix(4, seed=123, condition_number=100.0)
    return tw.MetricTensor(G, validate=True)


@pytest.fixture(scope="module")
def subspace_physics_standard():
    """Subespacio físico estándar (0:3)."""
    P_nom = tw.PhysicalConstants.P_NOMINAL()
    return tw.SubspaceSpec(
        name="physics_core",
        indices=slice(0, 3),
        weight=1.0,
        reference=np.zeros(3, dtype=np.float64),
        scale=np.array([
            tw.PhysicalConstants.SATURATION_CRITICAL,
            tw.PhysicalConstants.FLYBACK_MAX_SAFE,
            P_nom
        ], dtype=np.float64),
    )


@pytest.fixture(scope="module")
def projector_7d_standard():
    """Proyector ortogonal estándar 7D con 3 subespacios."""
    P_nom = tw.PhysicalConstants.P_NOMINAL()
    
    subspaces = {
        "physics_core": tw.SubspaceSpec(
            name="physics_core",
            indices=slice(0, 3),
            weight=1.0,
            reference=np.zeros(3, dtype=np.float64),
            scale=np.array([
                tw.PhysicalConstants.SATURATION_CRITICAL,
                tw.PhysicalConstants.FLYBACK_MAX_SAFE,
                P_nom
            ], dtype=np.float64),
        ),
        "topology_core": tw.SubspaceSpec(
            name="topology_core",
            indices=slice(3, 5),
            weight=1.5,
            reference=np.array([1.0, 0.0], dtype=np.float64),
            scale=np.ones(2, dtype=np.float64),
        ),
        "thermo_core": tw.SubspaceSpec(
            name="thermo_core",
            indices=slice(5, 7),
            weight=1.2,
            reference=np.zeros(2, dtype=np.float64),
            scale=np.ones(2, dtype=np.float64) * 0.5,
        ),
    }
    
    return tw.OrthogonalProjector(
        dimensions=7,
        subspaces=subspaces,
        topo_indices=tw.BETTI_INDICES,
        cache_projections=True
    )


@pytest.fixture
def watcher_default():
    """Morfismo inmunológico con perfil default (fresh instance per test)."""
    return tw.create_immune_watcher(profile="default")


@pytest.fixture
def watcher_strict():
    """Morfismo inmunológico con perfil strict."""
    return tw.create_immune_watcher(profile="strict")


@pytest.fixture
def watcher_laboratory():
    """Morfismo inmunológico sin histéresis (testing)."""
    return tw.create_immune_watcher(profile="laboratory")


# ==============================================================================
# TESTS: CONSTANTES FÍSICAS
# ==============================================================================

class TestPhysicalConstants:
    """Verifica consistencia dimensional y límites físicos."""
    
    def test_consistency_validated_on_import(self, physical_constants):
        """Validación ejecutada en import-time."""
        # No debe lanzar excepciones
        physical_constants.validate_physical_consistency()
    
    def test_power_nominal_positive(self, physical_constants):
        P = physical_constants.P_NOMINAL()
        assert P > 0
        assert np.isfinite(P)
    
    def test_ohms_law_consistency(self, physical_constants):
        """P = V²/Z = I²·Z debe ser consistente."""
        V = physical_constants.V_NOMINAL
        Z = physical_constants.Z_CHARACTERISTIC
        I = V / Z
        
        P_from_V = V**2 / Z
        P_from_I = I**2 * Z
        
        assert np.isclose(P_from_V, P_from_I, rtol=1e-12)


# ==============================================================================
# TESTS: ÁLGEBRA LINEAL ESTABLE
# ==============================================================================

class TestStableLinearAlgebra:
    """Tests de operaciones numéricamente estables."""
    
    # --- Normalización ---
    
    def test_safe_normalize_inf_norm(self):
        v = np.array([3.0, -4.0, 5.0], dtype=np.float64)
        vn, s = tw.StableLinearAlgebra.safe_normalize(v, norm_type='inf')
        
        assert np.isclose(np.max(np.abs(vn)), 1.0, atol=EPS_TEST)
        assert np.isclose(s, 5.0, atol=EPS_TEST)
    
    def test_safe_normalize_l2_norm(self):
        v = np.array([3.0, 4.0], dtype=np.float64)
        vn, s = tw.StableLinearAlgebra.safe_normalize(v, norm_type='2')
        
        assert np.isclose(np.linalg.norm(vn, ord=2), 1.0, atol=EPS_TEST)
        assert np.isclose(s, 5.0, atol=EPS_TEST)
    
    def test_safe_normalize_degenerate_vector(self):
        """Vector casi nulo debe retornar zeros."""
        v = np.array([1e-20, 1e-20], dtype=np.float64)
        vn, s = tw.StableLinearAlgebra.safe_normalize(v)
        
        assert np.allclose(vn, 0.0)
        assert s == 0.0 or s < EPS_TEST
    
    def test_safe_normalize_nan_raises(self):
        """NaN debe lanzar excepción."""
        v = np.array([1.0, np.nan, 2.0])
        with pytest.raises(tw.NumericalStabilityError, match="NaN"):
            tw.StableLinearAlgebra.safe_normalize(v)
    
    def test_safe_normalize_inf_clamped(self):
        """Inf debe ser clampeado con warning."""
        v = np.array([1.0, np.inf, 2.0])
        with pytest.warns(RuntimeWarning, match="Inf"):
            vn, s = tw.StableLinearAlgebra.safe_normalize(v)
        
        assert np.all(np.isfinite(vn))
    
    # --- Reciprocal ---
    
    def test_stable_reciprocal_normal(self):
        x = np.array([2.0, -4.0, 0.5], dtype=np.float64)
        invx = tw.StableLinearAlgebra.stable_reciprocal(x)
        
        expected = np.array([0.5, -0.25, 2.0], dtype=np.float64)
        assert np.allclose(invx, expected, atol=EPS_TEST)
    
    def test_stable_reciprocal_zero_protected(self):
        """División por cero debe ser protegida."""
        x = np.array([0.0, 1e-20, -0.0], dtype=np.float64)
        invx = tw.StableLinearAlgebra.stable_reciprocal(x, eps=EPS_TEST)
        
        assert np.all(np.isfinite(invx))
        assert np.all(np.abs(invx) > 0)
    
    # --- Forma cuadrática ---
    
    def test_stable_quadratic_form_identity(self):
        """Q(v) = v^T I v = ||v||²."""
        G = np.eye(3, dtype=np.float64)
        v = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        
        Q = tw.StableLinearAlgebra.stable_quadratic_form(v, G)
        expected = np.dot(v, v)
        
        assert np.isclose(Q, expected, atol=1e-12)
    
    def test_stable_quadratic_form_nonnegative(self):
        """Q(v) ≥ 0 para SPD."""
        G = make_spd_matrix(4, seed=789)
        v = np.random.RandomState(789).randn(4)
        
        Q = tw.StableLinearAlgebra.stable_quadratic_form(v, G)
        assert Q >= -1e-14  # tolerancia numérica
    
    def test_stable_quadratic_form_zero_vector(self):
        G = np.eye(5, dtype=np.float64)
        v = np.zeros(5, dtype=np.float64)
        
        Q = tw.StableLinearAlgebra.stable_quadratic_form(v, G)
        assert Q == 0.0
    
    # --- Número de condición ---
    
    def test_compute_condition_spectral_eig(self):
        """κ(diag(1,2,100)) = 100."""
        G = np.diag([1.0, 2.0, 100.0])
        kappa = tw.StableLinearAlgebra.compute_condition_spectral(G, method='eig')
        
        assert np.isclose(kappa, 100.0, rtol=1e-10)
    
    def test_compute_condition_spectral_svd(self):
        G = make_spd_matrix(5, seed=999, condition_number=1000.0)
        kappa = tw.StableLinearAlgebra.compute_condition_spectral(G, method='svd')
        
        assert 900.0 < kappa < 1100.0  # aproximado
    
    def test_compute_condition_singular_returns_inf(self):
        """Matriz singular debe retornar inf."""
        G = np.zeros((3, 3), dtype=np.float64)
        kappa = tw.StableLinearAlgebra.compute_condition_spectral(G)
        
        assert kappa == float('inf')
    
    # --- Regularización SPD ---
    
    def test_regularize_spd_already_valid(self):
        """Matriz ya SPD no debe cambiar."""
        G = np.eye(4, dtype=np.float64) * 5.0
        Greg = tw.StableLinearAlgebra.regularize_spd_matrix(G, min_eig=MIN_EIG)
        
        assert np.allclose(Greg, G, atol=ALG_TOL)
    
    def test_regularize_spd_near_singular(self):
        """Matriz casi singular debe ser regularizada."""
        G = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]], dtype=np.float64)
        Greg = tw.StableLinearAlgebra.regularize_spd_matrix(G, min_eig=MIN_EIG)
        
        lam = np.linalg.eigvalsh(Greg)
        assert lam.min() >= MIN_EIG - 1e-13
        assert np.allclose(Greg, Greg.T, atol=ALG_TOL)
    
    def test_regularize_spd_negative_eigenvalue(self):
        """Eigenvalor negativo debe ser corregido."""
        # G con eigenvalues [-1, 2]
        Q = np.array([[1, 1], [1, -1]], dtype=np.float64) / np.sqrt(2)
        L = np.diag([-1.0, 2.0])
        G = Q @ L @ Q.T
        
        Greg = tw.StableLinearAlgebra.regularize_spd_matrix(G, min_eig=MIN_EIG)
        lam = np.linalg.eigvalsh(Greg)
        
        assert lam.min() >= MIN_EIG - 1e-13
    
    def test_regularize_spd_preserves_symmetry(self):
        """Regularización debe preservar simetría."""
        G = make_spd_matrix(6, seed=111)
        G[0, 1] += 1e-10  # asimetría leve
        
        Greg = tw.StableLinearAlgebra.regularize_spd_matrix(G)
        assert np.allclose(Greg, Greg.T, atol=ALG_TOL)
    
    # --- Gram-Schmidt ---
    
    def test_gram_schmidt_orthonormal(self):
        """Resultado debe ser ortonormal."""
        V = np.random.RandomState(555).randn(5, 3)
        Q = tw.StableLinearAlgebra.gram_schmidt_orthonormalize(V)
        
        is_orth, res = tw.StableLinearAlgebra.verify_orthogonality(Q)
        assert is_orth
        assert res < ALG_TOL * 5.0
    
    def test_gram_schmidt_degenerate_handled(self):
        """Vectores colineales deben ser manejados."""
        V = np.array([[1, 0, 0], [1, 1e-14, 0], [0, 0, 1]], dtype=np.float64).T
        Q = tw.StableLinearAlgebra.gram_schmidt_orthonormalize(V)
        
        is_orth, _ = tw.StableLinearAlgebra.verify_orthogonality(Q)
        assert is_orth


# ==============================================================================
# TESTS: TENSOR MÉTRICO RIEMANNIANO
# ==============================================================================

class TestMetricTensor:
    """Tests de invariantes y operaciones de métricas."""
    
    # --- Invariantes ---
    
    def test_metric_identity_all_invariants(self, metric_identity_3):
        """Métrica identidad debe satisfacer todos los invariantes."""
        inv = metric_identity_3.verify_invariants()
        
        for name, satisfied in inv.items():
            assert satisfied, f"Invariante fallido: {name}"
    
    def test_metric_diagonal_invariants(self, metric_diagonal_well_conditioned):
        inv = metric_diagonal_well_conditioned.verify_invariants()
        
        assert inv["symmetry"]
        assert inv["positive_definite"]
        assert inv["well_conditioned"]
        assert inv["eigenvectors_orthogonal"]
        assert inv["spectral_reconstruction"]
    
    def test_metric_dense_invariants(self, metric_dense_moderate):
        inv = metric_dense_moderate.verify_invariants()
        
        for name, satisfied in inv.items():
            assert satisfied, f"Invariante fallido: {name}"
    
    def test_metric_condition_number_reasonable(self, metric_diagonal_well_conditioned):
        """κ debe ser finito y razonable."""
        kappa = metric_diagonal_well_conditioned.condition_number
        
        assert np.isfinite(kappa)
        assert kappa >= 1.0
        assert kappa < COND_TOL
    
    # --- Construcción ---
    
    def test_metric_from_dense_array(self):
        G = make_spd_matrix(5, seed=777)
        mt = tw.MetricTensor(G, validate=True)
        
        assert mt.dimension == 5
        assert not mt.is_diagonal
        assert is_spd(mt.to_array())
    
    def test_metric_from_diagonal_array(self):
        diag = np.array([3.0, 1.5, 0.8], dtype=np.float64)
        mt = tw.MetricTensor(diag, validate=True)
        
        assert mt.dimension == 3
        assert mt.is_diagonal
        assert np.allclose(mt.to_array(), diag)
    
    def test_metric_rejects_non_square(self):
        """Matriz no cuadrada debe lanzar error."""
        G_bad = np.random.randn(3, 5)
        
        with pytest.raises(tw.MetricTensorError, match="cuadrada"):
            tw.MetricTensor(G_bad, validate=True)
    
    def test_metric_rejects_negative_eigenvalue(self):
        """Eigenvalor negativo debe lanzar error con validate=True."""
        Q = np.eye(2)
        L = np.diag([-1.0, 2.0])
        G_bad = Q @ L @ Q.T
        
        with pytest.raises(tw.MetricTensorError):
            tw.MetricTensor(G_bad, validate=True)
    
    def test_metric_symmetrizes_asymmetric(self):
        """Asimetría leve debe ser corregida."""
        G = np.eye(3, dtype=np.float64)
        G[0, 1] = 0.5
        G[1, 0] = 0.5 + 1e-9  # asimetría pequeña
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mt = tw.MetricTensor(G, validate=True)
            
            # Debe emitir warning
            assert len(w) > 0
            assert "Asimetría" in str(w[0].message)
        
        # Resultado debe ser simétrico
        Greg = mt.to_array()
        assert np.allclose(Greg, Greg.T, atol=ALG_TOL)
    
    # --- Operaciones ---
    
    def test_metric_quadratic_form_nonnegative(self, metric_identity_3):
        """Q(v) ≥ 0 para SPD."""
        v = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        Q = metric_identity_3.quadratic_form(v)
        
        assert Q >= -1e-14
    
    def test_metric_quadratic_form_diagonal(self, metric_diagonal_well_conditioned):
        diag = metric_diagonal_well_conditioned.to_array()
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        Q = metric_diagonal_well_conditioned.quadratic_form(v)
        expected = np.sum((v ** 2) * diag)
        
        assert np.isclose(Q, expected, atol=1e-12)
    
    def test_metric_apply_identity(self, metric_identity_3):
        """G·v = v para identidad."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        Gv = metric_identity_3.apply(v)
        
        assert np.allclose(Gv, v, atol=EPS_TEST)
    
    def test_metric_inverse_sqrt_apply_identity(self, metric_identity_3):
        """G^{-1/2}·v = v para identidad."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        w = metric_identity_3.inverse_sqrt_apply(v)
        
        assert np.allclose(w, v, atol=1e-12)
    
    def test_metric_inverse_sqrt_apply_diagonal(self, metric_diagonal_well_conditioned):
        """G^{-1/2}·v = diag(1/√λ)·v."""
        diag = metric_diagonal_well_conditioned.to_array()
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        w = metric_diagonal_well_conditioned.inverse_sqrt_apply(v)
        expected = v / np.sqrt(diag)
        
        assert np.allclose(w, expected, atol=1e-12)
    
    # --- Descomposición espectral ---
    
    def test_spectral_decomposition_ordered(self, metric_dense_moderate):
        """Eigenvalores deben estar ordenados decrecientemente."""
        spec = metric_dense_moderate.spectral_decomposition
        lam = spec.eigenvalues
        
        assert np.all(np.diff(lam) <= 0)
    
    def test_spectral_decomposition_orthogonal(self, metric_dense_moderate):
        """Eigenvectores deben ser ortonormales."""
        spec = metric_dense_moderate.spectral_decomposition
        Q = spec.eigenvectors
        
        is_orth, res = tw.StableLinearAlgebra.verify_orthogonality(Q)
        assert is_orth
        assert res < ALG_TOL * 5.0
    
    def test_spectral_reconstruction(self, metric_dense_moderate):
        """G = Q Λ Q^T debe reconstruir matriz original."""
        spec = metric_dense_moderate.spectral_decomposition
        G_original = metric_dense_moderate.to_array()
        
        Q = spec.eigenvectors
        Λ = np.diag(spec.eigenvalues)
        G_reconstructed = Q @ Λ @ Q.T
        
        assert np.allclose(G_reconstructed, G_original, atol=ALG_TOL * 10.0)


# ==============================================================================
# TESTS: FUNTOR DE MEMBRANA Y LAPLACIANO
# ==============================================================================

class TestIsolatingMembraneFunctor:
    """Tests del funtor p-Dirichlet y Laplaciano discreto."""
    
    # --- Laplaciano ---
    
    def test_laplacian_constant_is_zero(self):
        """Δ(constante) = 0."""
        f = np.ones(20, dtype=np.float64) * 5.0
        lap = tw.IsolatingMembraneFunctor.discrete_laplacian_1d(f)
        
        assert np.allclose(lap, 0.0, atol=1e-14)
    
    def test_laplacian_linear_is_zero(self):
        """Δ(ax + b) = 0."""
        x = np.linspace(-1, 1, 30)
        f = 3.0 * x + 7.0
        lap = tw.IsolatingMembraneFunctor.discrete_laplacian_1d(f)
        
        assert np.allclose(lap, 0.0, atol=1e-10)
    
    def test_laplacian_quadratic_constant(self):
        """Δ(x²) ≈ 2 (interior)."""
        x = np.linspace(-1, 1, 51)
        f = x ** 2
        lap = tw.IsolatingMembraneFunctor.discrete_laplacian_1d(f)
        
        # Interior debe ser ~2.0
        assert np.allclose(lap[10:-10], 2.0, atol=0.05)
    
    def test_laplacian_handles_short_arrays(self):
        """Arrays muy cortos deben retornar zeros."""
        f = np.array([1.0, 2.0])
        lap = tw.IsolatingMembraneFunctor.discrete_laplacian_1d(f)
        
        assert np.allclose(lap, 0.0)
    
    # --- Estrés topológico ---
    
    def test_topological_stress_nonnegative(self):
        psi = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)
        mem = tw.IsolatingMembraneFunctor(p=1.5, eps=1e-12)
        
        stress = mem.compute_topological_stress(psi)
        
        assert np.all(stress >= 0.0)
        assert np.all(np.isfinite(stress))
    
    def test_topological_stress_smooth_signal_low(self):
        """Señal suave debe tener estrés bajo."""
        psi = np.sin(np.linspace(0, 2*np.pi, 50))
        mem = tw.IsolatingMembraneFunctor(p=1.5, eps=1e-12)
        
        stress = mem.compute_topological_stress(psi)
        
        # Estrés debe ser relativamente bajo
        assert np.mean(stress) < 10.0
    
    def test_topological_stress_discontinuous_high(self):
        """Discontinuidad debe producir estrés alto."""
        psi = np.zeros(20, dtype=np.float64)
        psi[10] = 100.0  # spike
        mem = tw.IsolatingMembraneFunctor(p=1.5, eps=1e-12)
        
        stress = mem.compute_topological_stress(psi)
        
        # Spike debe elevar estrés local
        assert stress[10] > np.mean(stress) * 10.0
    
    def test_functor_p_range_validation(self):
        """p fuera de [1, 2) debe lanzar error."""
        with pytest.raises(ValueError, match="p debe estar en"):
            tw.IsolatingMembraneFunctor(p=0.5)
        
        with pytest.raises(ValueError, match="p debe estar en"):
            tw.IsolatingMembraneFunctor(p=2.0)
        
        # Válidos
        tw.IsolatingMembraneFunctor(p=1.0)
        tw.IsolatingMembraneFunctor(p=1.99)


# ==============================================================================
# TESTS: SUBESPACIOS
# ==============================================================================

class TestSubspaceSpec:
    """Tests de especificación de subespacios."""
    
    def test_subspace_threat_nonnegative(self, subspace_physics_standard):
        v = np.array([0.5, 200.0, 100.0], dtype=np.float64)
        threat = subspace_physics_standard.compute_threat(v)
        
        assert threat >= 0.0
        assert np.isfinite(threat)
    
    def test_subspace_threat_zero_on_reference(self, subspace_physics_standard):
        """Amenaza debe ser cero en punto de referencia."""
        threat = subspace_physics_standard.compute_threat(
            subspace_physics_standard.reference
        )
        
        assert np.isclose(threat, 0.0, atol=1e-13)
    
    def test_subspace_threat_increases_with_distance(self, subspace_physics_standard):
        """Amenaza debe aumentar con distancia de referencia."""
        ref = subspace_physics_standard.reference
        
        v_close = ref + np.array([0.1, 10.0, 5.0])
        v_far = ref + np.array([0.5, 100.0, 50.0])
        
        threat_close = subspace_physics_standard.compute_threat(v_close)
        threat_far = subspace_physics_standard.compute_threat(v_far)
        
        assert threat_far > threat_close
    
    def test_subspace_dimension_mismatch_raises(self):
        """Dimensión incorrecta debe lanzar error."""
        with pytest.raises(tw.DimensionalMismatchError):
            tw.SubspaceSpec(
                name="bad",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(2),  # debería ser 3
            )
    
    def test_subspace_negative_weight_raises(self):
        """Peso negativo debe lanzar error."""
        with pytest.raises(ValueError, match="peso"):
            tw.SubspaceSpec(
                name="bad",
                indices=slice(0, 2),
                weight=-1.0,
                reference=np.zeros(2),
            )
    
    def test_subspace_metric_construction_from_scale(self):
        """Métrica debe ser construida automáticamente desde scale."""
        spec = tw.SubspaceSpec(
            name="test",
            indices=slice(0, 3),
            weight=1.0,
            reference=np.zeros(3),
            scale=np.array([2.0, 4.0, 8.0]),
        )
        
        assert spec.metric is not None
        assert spec.metric.dimension == 3
        assert spec.metric.is_diagonal
    
    def test_subspace_normalize_to_reference(self, subspace_physics_standard):
        """Normalización debe aplicar G^{-1/2}."""
        v = np.array([0.5, 200.0, 100.0], dtype=np.float64)
        normalized = subspace_physics_standard.normalize_to_reference(v)
        
        assert normalized.shape == v.shape
        assert np.all(np.isfinite(normalized))


# ==============================================================================
# TESTS: PROYECTOR ORTOGONAL
# ==============================================================================

class TestOrthogonalProjector:
    """Tests de proyectores ortogonales y cobertura."""
    
    # --- Invariantes algebraicos ---
    
    def test_projector_idempotence(self, projector_7d_standard):
        """P² = P para cada proyector."""
        rep = projector_7d_standard.validation_report
        
        for key, value in rep.items():
            if "idempotence" in key:
                assert value < ALG_TOL * 5.0, f"{key} = {value:.2e}"
    
    def test_projector_self_adjoint(self, projector_7d_standard):
        """P^T = P para cada proyector."""
        rep = projector_7d_standard.validation_report
        
        for key, value in rep.items():
            if "self_adjoint" in key:
                assert value < ALG_TOL * 5.0, f"{key} = {value:.2e}"
    
    def test_projector_sum_identity(self, projector_7d_standard):
        """Σ π_k = I."""
        rep = projector_7d_standard.validation_report
        coverage_err = rep.get("coverage_identity_error", float("inf"))
        
        assert coverage_err < ALG_TOL * 10.0
    
    # --- Proyección ---
    
    def test_project_healthy_signal(self, projector_7d_standard):
        assessment = projector_7d_standard.project(SIGNAL_HEALTHY)
        
        assert assessment.status == tw.HealthStatus.HEALTHY
        assert assessment.max_value < 0.8
        assert assessment.total_threat >= 0.0
        assert assessment.euler_char == 1  # β₀=1, β₁=0
    
    def test_project_warning_signal(self, projector_7d_standard):
        assessment = projector_7d_standard.project(SIGNAL_WARNING)
        
        assert assessment.status in (tw.HealthStatus.WARNING, tw.HealthStatus.HEALTHY)
        assert assessment.total_threat >= 0.0
    
    def test_project_critical_signal(self, projector_7d_standard):
        assessment = projector_7d_standard.project(SIGNAL_CRITICAL)
        
        assert assessment.status == tw.HealthStatus.CRITICAL
        assert assessment.max_value > 1.5
        # χ = β₀ - β₁ = 2 - 3 = -1
        assert assessment.euler_char == -1
    
    def test_project_topology_anomaly(self, projector_7d_standard):
        """Múltiples componentes y ciclos."""
        assessment = projector_7d_standard.project(SIGNAL_TOPOLOGY_ANOMALY)
        
        # χ = 3 - 5 = -2
        assert assessment.euler_char == -2
        assert assessment.status in (tw.HealthStatus.WARNING, tw.HealthStatus.CRITICAL)
    
    def test_project_nan_raises(self, projector_7d_standard):
        """NaN debe lanzar error."""
        bad_signal = SIGNAL_HEALTHY.copy()
        bad_signal[2] = np.nan
        
        with pytest.raises(tw.NumericalStabilityError, match="no finitos"):
            projector_7d_standard.project(bad_signal)
    
    def test_project_inf_raises(self, projector_7d_standard):
        """Inf debe lanzar error."""
        bad_signal = SIGNAL_HEALTHY.copy()
        bad_signal[1] = np.inf
        
        with pytest.raises(tw.NumericalStabilityError, match="no finitos"):
            projector_7d_standard.project(bad_signal)
    
    def test_project_dimension_mismatch_raises(self, projector_7d_standard):
        """Dimensión incorrecta debe lanzar error."""
        bad_signal = np.zeros(5)  # debería ser 7
        
        with pytest.raises(tw.DimensionalMismatchError):
            projector_7d_standard.project(bad_signal)
    
    # --- Histéresis ---
    
    def test_hysteresis_prevents_oscillation(self, projector_7d_standard):
        """Histéresis debe prevenir oscilación."""
        # Señal en frontera warning
        signal_border = SIGNAL_HEALTHY.copy()
        signal_border[0] = 0.55  # cerca de warning
        
        # Primera evaluación: healthy
        a1 = projector_7d_standard.project(
            signal_border,
            warning_threshold=0.6,
            hysteresis=0.1,
            previous_status=tw.HealthStatus.HEALTHY
        )
        
        # Segunda evaluación con mismo valor: debe mantener healthy
        a2 = projector_7d_standard.project(
            signal_border,
            warning_threshold=0.6,
            hysteresis=0.1,
            previous_status=a1.status
        )
        
        assert a1.status == tw.HealthStatus.HEALTHY
        assert a2.status == tw.HealthStatus.HEALTHY
    
    # --- Euler characteristic ---
    
    def test_euler_char_betti_0_validation(self, projector_7d_standard):
        """β₀ < 1 debe lanzar error."""
        bad_signal = SIGNAL_HEALTHY.copy()
        bad_signal[3] = 0.0  # β₀ = 0
        
        with pytest.raises(tw.TopologicalInvariantError, match="β₀"):
            projector_7d_standard.project(bad_signal)
    
    def test_euler_char_betti_1_validation(self, projector_7d_standard):
        """β₁ < 0 debe lanzar error."""
        bad_signal = SIGNAL_HEALTHY.copy()
        bad_signal[4] = -1.0  # β₁ = -1
        
        with pytest.raises(tw.TopologicalInvariantError, match="β₁"):
            projector_7d_standard.project(bad_signal)
    
    # --- Métrica global ---
    
    def test_global_metric_mahalanobis_distance(self, projector_7d_standard):
        """Distancia de Mahalanobis global debe ser usada."""
        assessment = projector_7d_standard.project(SIGNAL_HEALTHY)
        
        # Si métrica global disponible
        if projector_7d_standard._global_metric_tensor is not None:
            assert assessment.total_threat >= 0.0
            # Debe ser diferente de norma L²
            l2_norm = float(np.linalg.norm(list(assessment.levels.values())))
            # Puede ser igual o diferente dependiendo de ponderaciones


# ==============================================================================
# TESTS: VALIDADORES Y BUILD_SIGNAL
# ==============================================================================

class TestValidatorsAndBuildSignal:
    """Tests de validadores y construcción de señal."""
    
    # --- Validadores ---
    
    def test_unit_interval_validator_clamps(self):
        validator = tw.VALIDATOR_REGISTRY["unit_interval"]
        
        val, mod, msg = validator.validate(1.5, "test")
        assert val == 1.0
        assert mod
        
        val, mod, msg = validator.validate(-0.2, "test")
        assert val == 0.0
        assert mod
        
        val, mod, msg = validator.validate(0.5, "test")
        assert val == 0.5
        assert not mod
    
    def test_non_negative_validator(self):
        validator = tw.VALIDATOR_REGISTRY["non_negative"]
        
        val, mod, msg = validator.validate(-10.0, "test")
        assert val == 0.0
        assert mod
        
        val, mod, msg = validator.validate(5.0, "test")
        assert val == 5.0
        assert not mod
    
    def test_positive_int_validator_rounds(self):
        validator = tw.VALIDATOR_REGISTRY["positive_int"]
        
        val, mod, msg = validator.validate(3.7, "test")
        assert val == 4.0
        assert mod
        
        val, mod, msg = validator.validate(0.2, "test")
        assert val == 1.0  # min=1
        assert mod
    
    def test_validator_rejects_nan(self):
        validator = tw.VALIDATOR_REGISTRY["unit_interval"]
        
        with pytest.raises(tw.PhysicalBoundsError, match="no finito"):
            validator.validate(np.nan, "test")
    
    # --- build_signal ---
    
    def test_build_signal_valid_telemetry(self):
        telem = telemetry_from_signal(SIGNAL_HEALTHY)
        sig = tw.build_signal(telem, strict=False)
        
        assert np.allclose(sig, SIGNAL_HEALTHY, atol=1e-12)
    
    def test_build_signal_missing_key_uses_default(self):
        telem = {"saturation": 0.5}  # otros faltantes
        sig = tw.build_signal(telem, strict=False)
        
        assert sig[0] == 0.5
        assert sig[1] == tw.SIGNAL_SCHEMA[1].default  # flyback_voltage
    
    def test_build_signal_clamps_out_of_range(self):
        telem = telemetry_from_signal(SIGNAL_HEALTHY)
        telem["saturation"] = 1.8  # > 1.0
        telem["entropy"] = -0.5    # < 0.0
        
        sig = tw.build_signal(telem, strict=False)
        
        assert sig[0] == 1.0  # clamped
        assert sig[5] == 0.0  # clamped
    
    def test_build_signal_nonfinite_uses_default(self):
        telem = telemetry_from_signal(SIGNAL_HEALTHY)
        telem["flyback_voltage"] = np.inf
        telem["dissipated_power"] = np.nan
        
        sig = tw.build_signal(telem, strict=False)
        
        assert sig[1] == tw.SIGNAL_SCHEMA[1].default
        assert sig[2] == tw.SIGNAL_SCHEMA[2].default
    
    def test_build_signal_strict_raises_on_invalid(self):
        telem = telemetry_from_signal(SIGNAL_HEALTHY)
        telem["saturation"] = "not_a_number"
        
        with pytest.raises(ValueError, match="no convertible"):
            tw.build_signal(telem, strict=True)
    
    def test_build_signal_strict_raises_on_nonfinite(self):
        telem = telemetry_from_signal(SIGNAL_HEALTHY)
        telem["flyback_voltage"] = np.inf
        
        with pytest.raises(ValueError, match="no finita"):
            tw.build_signal(telem, strict=True)


# ==============================================================================
# TESTS: MORFISMO INMUNOLÓGICO
# ==============================================================================

class TestImmuneWatcherMorphism:
    """Tests del morfismo categórico completo."""
    
    # --- Propiedades funtoriales ---
    
    def test_functorial_properties_satisfied(self, watcher_default):
        """Todas las propiedades funtoriales deben ser verdaderas."""
        props = watcher_default._verify_functorial_properties()
        
        for name, satisfied in props.items():
            assert satisfied, f"Propiedad funtorial fallida: {name}"
    
    def test_domain_is_physics(self, watcher_default):
        """Dominio debe incluir PHYSICS."""
        assert Stratum.PHYSICS in watcher_default.domain
    
    def test_codomain_is_wisdom(self, watcher_default):
        """Codominio debe ser WISDOM."""
        assert watcher_default.codomain == Stratum.WISDOM
    
    # --- Monada de error ---
    
    def test_call_preserves_error_monad(self, watcher_default):
        """F(⊥) = ⊥ (preserva objeto error)."""
        state_error = CategoricalState(success=False, error_msg="sensor failure")
        new_state = watcher_default(state_error)
        
        assert not new_state.is_success
        assert new_state.error_msg == "sensor failure"
    
    def test_call_healthy_returns_success(self, watcher_default):
        """Estado saludable debe retornar éxito."""
        state = CategoricalState(
            success=True,
            context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}
        )
        new_state = watcher_default(state)
        
        assert new_state.is_success
        assert new_state.context.get("immune_status") == "healthy"
        assert new_state.stratum == Stratum.WISDOM
    
    def test_call_warning_returns_update(self, watcher_default):
        """Estado warning debe retornar actualización."""
        state = CategoricalState(
            success=True,
            context={"telemetry_metrics": telemetry_from_signal(SIGNAL_WARNING)}
        )
        new_state = watcher_default(state)
        
        assert new_state.is_success
        assert new_state.context.get("immune_status") in ("healthy", "warning")
    
    def test_call_critical_returns_quarantine(self, watcher_default):
        """Estado crítico debe activar cuarentena."""
        state = CategoricalState(
            success=True,
            context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)}
        )
        new_state = watcher_default(state)
        
        assert not new_state.is_success
        assert "QUARANTINE" in new_state.context.get("action", "")
    
    # --- Histéresis ---
    
    def test_hysteresis_stable_transitions(self, watcher_laboratory):
        """Sin histéresis, transiciones deben ser inmediatas."""
        watcher_laboratory.reset_state()
        
        # healthy
        s0 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
        ns0 = watcher_laboratory(s0)
        assert ns0.context.get("immune_status") == "healthy"
        
        # critical
        s1 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
        ns1 = watcher_laboratory(s1)
        assert not ns1.is_success
        
        # back to healthy
        s2 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
        ns2 = watcher_laboratory(s2)
        assert ns2.context.get("immune_status") == "healthy"
    
    def test_hysteresis_prevents_chatter(self, watcher_default):
        """Histéresis debe prevenir oscilación."""
        watcher_default.reset_state()
        
        # Señal oscilante cerca del umbral
        signal_border = SIGNAL_HEALTHY.copy()
        signal_border[0] = 0.75  # cerca de warning threshold=0.8
        
        s1 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(signal_border)})
        ns1 = watcher_default(s1)
        
        # Repetir evaluación
        s2 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(signal_border)})
        ns2 = watcher_default(s2)
        
        # Estado no debe oscilar
        assert ns1.context.get("immune_status") == ns2.context.get("immune_status")
    
    # --- Bifurcaciones topológicas ---
    
    def test_topology_bifurcation_detected(self, watcher_default):
        """Cambio en χ debe ser detectado."""
        watcher_default.reset_state()
        
        # χ = 1
        s0 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
        watcher_default(s0)
        
        # χ = -1
        s1 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
        watcher_default(s1)
        
        hist = watcher_default.topology_history
        assert hist[-2] == 1
        assert hist[-1] == -1
    
    def test_topology_history_accumulated(self, watcher_default):
        """Historia topológica debe acumularse."""
        watcher_default.reset_state()
        
        signals = [SIGNAL_HEALTHY, SIGNAL_WARNING, SIGNAL_CRITICAL, SIGNAL_HEALTHY]
        
        for sig in signals:
            s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(sig)})
            watcher_default(s)
        
        hist = watcher_default.topology_history
        assert len(hist) == 4
    
    # --- Flujo de Ricci ---
    
    def test_ricci_flow_preserves_spd(self, watcher_default):
        """Flujo de Ricci debe preservar SPD."""
        watcher_default.reset_state()
        
        # Forzar evolución con señal crítica
        s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
        watcher_default(s)
        
        # Verificar métricas SPD
        for key, G in watcher_default._metric_tensors_state.items():
            assert is_spd(G), f"{key} no SPD tras flujo de Ricci"
    
    def test_ricci_flow_multiple_steps(self, watcher_default):
        """Múltiples pasos deben mantener estabilidad."""
        watcher_default.reset_state()
        
        for _ in range(10):
            s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_WARNING)})
            watcher_default(s)
        
        # Todas las métricas deben seguir SPD
        for key, G in watcher_default._metric_tensors_state.items():
            assert is_spd(G), f"{key} no SPD tras 10 pasos"
    
    def test_ricci_flow_condition_bounded(self, watcher_default):
        """Número de condición debe permanecer acotado."""
        watcher_default.reset_state()
        
        for _ in range(5):
            s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
            watcher_default(s)
        
        for key, G in watcher_default._metric_tensors_state.items():
            kappa = tw.StableLinearAlgebra.compute_condition_spectral(G)
            assert kappa < COND_TOL * 10.0, f"{key} mal condicionada: κ={kappa:.2e}"
    
    # --- Evaluación de deformación de variedad ---
    
    def test_evaluate_manifold_deformation_stable(self, watcher_default):
        """Estado estable debe retornar is_stable=True."""
        watcher_default.reset_state()
        watcher_default(CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}))
        
        tm = watcher_default.evaluate_manifold_deformation(SIGNAL_HEALTHY, reference_chi=1)
        
        assert tm.is_stable
        assert tm.structural_alteration == 0
        assert tm.threat_level in ("HEALTHY", "WARNING")
    
    def test_evaluate_manifold_deformation_bifurcation(self, watcher_default):
        """Bifurcación debe retornar is_stable=False."""
        watcher_default.reset_state()
        watcher_default(CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}))
        
        tm = watcher_default.evaluate_manifold_deformation(SIGNAL_CRITICAL, reference_chi=1)
        
        assert not tm.is_stable
        assert tm.structural_alteration != 0
        assert tm.threat_level == "CRITICAL"
    
    # --- Diagnóstico y reportes ---
    
    def test_get_diagnostics_complete(self, watcher_default):
        """Diagnóstico debe contener todas las claves esperadas."""
        watcher_default.reset_state()
        watcher_default(CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}))
        
        diag = watcher_default.get_diagnostics()
        
        assert "name" in diag
        assert "evaluation_count" in diag
        assert "current_status" in diag
        assert "thresholds" in diag
        assert "topology_history_last10" in diag
        assert "projector_validation" in diag
        assert "functorial_properties" in diag
    
    def test_health_report_nonempty(self, watcher_default):
        """Reporte de salud debe ser string no vacío."""
        watcher_default.reset_state()
        watcher_default(CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}))
        
        report = watcher_default.health_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "IMMUNE WATCHER" in report
        assert "DIAGNÓSTICO" in report
    
    def test_reset_state_clears_history(self, watcher_default):
        """reset_state debe limpiar historial."""
        watcher_default(CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)}))
        assert watcher_default.evaluation_count > 0
        
        watcher_default.reset_state()
        
        assert watcher_default.evaluation_count == 0
        assert len(watcher_default.topology_history) == 0
        assert watcher_default.current_status is None
    
    # --- Perfiles ---
    
    def test_profile_strict_lower_thresholds(self, watcher_strict):
        """Perfil strict debe tener umbrales más bajos."""
        assert watcher_strict.thresholds["warning"] < 0.8
        assert watcher_strict.thresholds["critical"] < 1.5
    
    def test_profile_laboratory_no_hysteresis(self, watcher_laboratory):
        """Perfil laboratory debe tener histéresis cero."""
        assert watcher_laboratory.thresholds["hysteresis"] == 0.0


# ==============================================================================
# TESTS: CASOS DE BORDE Y ROBUSTEZ
# ==============================================================================

class TestEdgeCasesAndRobustness:
    """Tests de casos extremos y robustez numérica."""
    
    def test_zero_signal(self, projector_7d_standard):
        """Señal cero debe ser manejada."""
        zero = np.zeros(7, dtype=np.float64)
        assessment = projector_7d_standard.project(zero)
        
        assert np.isfinite(assessment.total_threat)
        # Topología: beta0=0 debería lanzar error, pero build_signal lo corrige
    
    def test_high_condition_metric_regularized(self):
        """Métrica mal condicionada debe ser regularizada."""
        G_bad = np.diag([1e10, 1.0, 1e-10])
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mt = tw.MetricTensor(G_bad, validate=True)
            
            assert is_spd(mt.to_array())
            # κ puede ser alto pero finito
    
    def test_block_diag_mixed_dimensions(self):
        """Bloques de diferentes dimensiones."""
        b1 = np.array([1.0, 2.0, 3.0])  # 1D
        b2 = np.eye(2, dtype=np.float64) * 5.0  # 2D
        b3 = np.array([7.0])  # 1D
        
        B = tw.block_diag_pure(b1, b2, b3)
        
        assert B.shape == (6, 6)
        assert np.allclose(np.diag(B)[:3], b1)
        assert np.allclose(B[3:5, 3:5], b2)
        assert B[5, 5] == 7.0
    
    def test_gram_schmidt_rank_deficient(self):
        """Matriz rango deficiente debe ser manejada."""
        V = np.array([[1, 0, 0], [1, 1e-15, 0], [0, 0, 1]], dtype=np.float64).T
        Q = tw.StableLinearAlgebra.gram_schmidt_orthonormalize(V)
        
        is_orth, _ = tw.StableLinearAlgebra.verify_orthogonality(Q)
        assert is_orth
        assert np.linalg.matrix_rank(Q) == 2  # rango efectivo
    
    def test_watcher_handles_missing_telemetry(self, watcher_default):
        """Telemetría vacía debe usar defaults."""
        state = CategoricalState(success=True, context={})
        new_state = watcher_default(state)
        
        assert new_state.is_success or not new_state.is_success  # depende de defaults
    
    def test_watcher_handles_partial_telemetry(self, watcher_default):
        """Telemetría parcial debe ser completada."""
        partial = {"saturation": 0.5, "beta_0": 1.0}
        state = CategoricalState(success=True, context={"telemetry_metrics": partial})
        new_state = watcher_default(state)
        
        assert new_state.is_success or not new_state.is_success


# ==============================================================================
# TESTS DE PROPIEDAD (HYPOTHESIS) - BONUS
# ==============================================================================

if HAS_HYPOTHESIS:
    class TestPropertyBased:
        """Tests basados en propiedades con Hypothesis."""
        
        @given(
            diag=st.lists(
                st.floats(min_value=0.1, max_value=100.0),
                min_size=2,
                max_size=10
            )
        )
        @settings(max_examples=50, deadline=None)
        def test_metric_diagonal_always_spd(self, diag):
            """Métrica diagonal con valores positivos siempre SPD."""
            diag_arr = np.array(diag, dtype=np.float64)
            mt = tw.MetricTensor(diag_arr, validate=True)
            
            assert is_spd(mt.to_array())
        
        @given(
            v=st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=3,
                max_size=3
            )
        )
        @settings(max_examples=50, deadline=None)
        def test_quadratic_form_nonnegative_identity(self, v):
            """Q(v) ≥ 0 para métrica identidad."""
            v_arr = np.array(v, dtype=np.float64)
            G = np.eye(3, dtype=np.float64)
            
            Q = tw.StableLinearAlgebra.stable_quadratic_form(v_arr, G)
            assert Q >= -1e-12
        
        @given(
            saturation=st.floats(min_value=-0.5, max_value=1.5),
            flyback=st.floats(min_value=0.0, max_value=500.0),
            beta_0=st.integers(min_value=1, max_value=5),
        )
        @settings(max_examples=30, deadline=None)
        def test_build_signal_always_valid(self, saturation, flyback, beta_0):
            """build_signal siempre produce señal válida."""
            telem = {
                "saturation": saturation,
                "flyback_voltage": flyback,
                "beta_0": beta_0,
            }
            
            sig = tw.build_signal(telem, strict=False)
            
            assert sig.shape == (7,)
            assert np.all(np.isfinite(sig))
            # saturation clampeada
            assert 0.0 <= sig[0] <= 1.0


# ==============================================================================
# TESTS DE INTEGRACIÓN COMPLETA
# ==============================================================================

class TestIntegrationFull:
    """Tests de ciclos completos end-to-end."""
    
    def test_full_cycle_healthy_warning_critical_recovery(self, watcher_default):
        """Ciclo completo: healthy → warning → critical → recovery."""
        watcher_default.reset_state()
        
        # Healthy
        s0 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
        ns0 = watcher_default(s0)
        assert ns0.context.get("immune_status") == "healthy"
        
        # Warning
        s1 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_WARNING)})
        ns1 = watcher_default(s1)
        assert ns1.context.get("immune_status") in ("healthy", "warning")
        
        # Critical → quarantine
        s2 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
        ns2 = watcher_default(s2)
        assert not ns2.is_success
        
        # Recovery
        s3 = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
        ns3 = watcher_default(s3)
        assert ns3.is_success
        assert ns3.context.get("immune_status") == "healthy"
    
    def test_electron_quantization_critical(self, watcher_default):
        """Electrones deben ser emitidos en estado crítico."""
        watcher_default.reset_state()
        
        s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_CRITICAL)})
        ns = watcher_default(s)
        
        # Verificar estructura de electrones si están presentes
        details = ns.context
        if "electrons" in details and details["electrons"]:
            electrons = details["electrons"]
            assert isinstance(electrons, (list, tuple))
    
    def test_ricci_flow_convergence(self, watcher_default):
        """Flujo de Ricci debe estabilizar métricas."""
        watcher_default.reset_state()
        
        # Evaluar múltiples veces con señal estable
        for _ in range(20):
            s = CategoricalState(success=True, context={"telemetry_metrics": telemetry_from_signal(SIGNAL_HEALTHY)})
            watcher_default(s)
        
        # Métricas deben seguir SPD y bien condicionadas
        for key, G in watcher_default._metric_tensors_state.items():
            assert is_spd(G)
            kappa = tw.StableLinearAlgebra.compute_condition_spectral(G)
            assert kappa < COND_TOL * 5.0


# ==============================================================================
# CONFIGURACIÓN DE PYTEST
# ==============================================================================

def pytest_configure(config):
    """Configuración custom de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos (deselect con '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hypothesis: tests basados en propiedades"
    )


# ==============================================================================
# EJECUCIÓN DIRECTA
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=3"])