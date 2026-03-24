"""
Suite de Pruebas Rigurosas: Tensores Métricos Riemannianos
Ubicación: tests/unit/core/immune_system/metric_tensors.py

Cobertura Matemática:
─────────────────────
1. Axiomas de espacio métrico y propiedades SPD
2. Invariantes algebraicos (simetría, positividad, condicionamiento)
3. Propiedades espectrales (Gershgorin, Sylvester, Cholesky)
4. Pipeline de validación (cada etapa individualmente)
5. Regularización Tikhonov (corrección, optimalidad, consistencia)
6. Casos límite y patologías numéricas
7. Inmutabilidad y seguridad de tipo
8. Diagnósticos externos
9. Propiedades geométricas del cono SPD

Convenciones:
─────────────
- Cada clase de test agrupa un aspecto matemático o funcional coherente
- Los nombres de test siguen: test_<qué>_<condición>_<resultado_esperado>
- Las tolerancias numéricas se justifican explícitamente
- Se evita dependencia entre tests (cada uno es autocontenido)
"""
from __future__ import annotations

import copy
import re
from typing import Final

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from app.core.immune_system.metric_tensors import (
    G_PHYSICS,
    G_TOPOLOGY,
    G_THERMODYNAMICS,
    MetricTensorFactory,
    RegularizationReport,
    SpectralProfile,
    get_tensor_diagnostics,
    _FLOAT_DTYPE,
    _NEAR_ZERO_FROBENIUS_TOL,
    _REGULARIZATION_ABORT_THRESHOLD,
    _REGULARIZATION_WARN_THRESHOLD,
    _SPD_INTERIOR_FACTOR,
    _SYMMETRY_ATOL_BASE,
    _SYMMETRY_RTOL,
    _TIKHONOV_DELTA_ATOL,
)
from app.core.immune_system.topological_watcher import (
    COND_NUM_TOL,
    MIN_EIGVAL_TOL,
    MetricTensorError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

# Tolerancia para comparaciones float64 (O(ε_mach) ≈ 2.2e-16)
_FLOAT64_RTOL: Final[float] = 1e-12
_FLOAT64_ATOL: Final[float] = 1e-14

# Tensores precompilados agrupados para parametrización
_ALL_PRECOMPILED_TENSORS = [
    ("G_PHYSICS", G_PHYSICS, 3),
    ("G_TOPOLOGY", G_TOPOLOGY, 2),
    ("G_THERMODYNAMICS", G_THERMODYNAMICS, 2),
]

_TENSOR_IDS = [name for name, _, _ in _ALL_PRECOMPILED_TENSORS]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=_ALL_PRECOMPILED_TENSORS, ids=_TENSOR_IDS)
def tensor_info(request):
    """Fixture parametrizada que provee (nombre, tensor, dimensión) para cada tensor."""
    return request.param


@pytest.fixture
def factory():
    """Provee la clase fábrica para acceso directo a métodos."""
    return MetricTensorFactory


@pytest.fixture
def identity_3x3():
    """Matriz identidad 3×3 como tensor SPD trivial."""
    return np.eye(3, dtype=_FLOAT_DTYPE)


@pytest.fixture
def identity_2x2():
    """Matriz identidad 2×2 como tensor SPD trivial."""
    return np.eye(2, dtype=_FLOAT_DTYPE)


@pytest.fixture
def well_conditioned_spd_3x3():
    """Matriz SPD 3×3 bien condicionada con eigenvalores conocidos."""
    # Construida como QΛQᵀ con eigenvalores {1, 2, 3}
    Q = np.linalg.qr(np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
    ], dtype=_FLOAT_DTYPE))[0]
    Lambda = np.diag([1.0, 2.0, 3.0])
    return Q @ Lambda @ Q.T


@pytest.fixture
def near_singular_spd_2x2():
    """Matriz SPD 2×2 con λ_min muy pequeño pero positivo."""
    tiny = MIN_EIGVAL_TOL * 0.1  # Por debajo del umbral
    return np.array([
        [1.0, 0.0],
        [0.0, tiny],
    ], dtype=_FLOAT_DTYPE)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 1: PROPIEDADES ALGEBRAICAS FUNDAMENTALES DE TENSORES PRECOMPILADOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrecompiledTensorAlgebraicProperties:
    """
    Verifica axiomas y propiedades algebraicas de los tensores precompilados
    que deben satisfacerse como invariantes del módulo.
    """

    def test_dtype_is_float64(self, tensor_info):
        """Todo tensor debe tener dtype float64 para estabilidad numérica."""
        name, G, _ = tensor_info
        assert G.dtype == _FLOAT_DTYPE, (
            f"{name}: dtype esperado {_FLOAT_DTYPE}, obtenido {G.dtype}"
        )

    def test_ndim_is_two(self, tensor_info):
        """Todo tensor debe ser bidimensional (rango tensorial 2)."""
        name, G, _ = tensor_info
        assert G.ndim == 2, f"{name}: ndim esperado 2, obtenido {G.ndim}"

    def test_shape_is_square(self, tensor_info):
        """Todo tensor debe ser cuadrado (endomorfismo)."""
        name, G, _ = tensor_info
        n_rows, n_cols = G.shape
        assert n_rows == n_cols, (
            f"{name}: forma no cuadrada {n_rows}×{n_cols}"
        )

    def test_expected_dimension(self, tensor_info):
        """Cada tensor debe tener la dimensión prescrita por su dominio."""
        name, G, expected_dim = tensor_info
        assert G.shape == (expected_dim, expected_dim), (
            f"{name}: dimensión esperada {expected_dim}×{expected_dim}, "
            f"obtenida {G.shape[0]}×{G.shape[1]}"
        )

    def test_all_entries_finite(self, tensor_info):
        """Ninguna entrada puede ser NaN o ±∞."""
        name, G, _ = tensor_info
        assert np.all(np.isfinite(G)), (
            f"{name}: contiene entradas no finitas"
        )

    def test_exact_symmetry(self, tensor_info):
        """
        Simetría bit-a-bit: G[i,j] == G[j,i] para todo i,j.

        Post-simetrización por (G + Gᵀ)/2, la simetría debe ser exacta
        en aritmética IEEE 754 (no solo aproximada).
        """
        name, G, _ = tensor_info
        np.testing.assert_array_equal(
            G, G.T,
            err_msg=f"{name}: simetría no es exacta bit-a-bit"
        )

    def test_symmetry_within_tolerance(self, tensor_info):
        """
        Verificación de simetría con tolerancia adaptativa (redundante con
        exacta, pero verifica el mecanismo de tolerancia).
        """
        name, G, _ = tensor_info
        frobenius_norm = np.linalg.norm(G, "fro")
        tol = max(_SYMMETRY_ATOL_BASE, _SYMMETRY_RTOL * frobenius_norm)
        asymmetry_norm = np.linalg.norm(G - G.T, "fro")
        assert asymmetry_norm <= tol, (
            f"{name}: ||G - Gᵀ||_F = {asymmetry_norm:.6e} > tol = {tol:.6e}"
        )

    def test_positive_diagonal(self, tensor_info):
        """
        Condición necesaria para SPD: todos los elementos diagonales
        deben ser positivos. Para G ≻ 0, eᵢᵀ G eᵢ = gᵢᵢ > 0.
        """
        name, G, _ = tensor_info
        diagonal = np.diag(G)
        assert np.all(diagonal > 0), (
            f"{name}: diagonal contiene elementos no positivos: {diagonal}"
        )

    def test_positive_determinant(self, tensor_info):
        """
        Condición necesaria para SPD: det(G) > 0.
        Para matrices SPD, det(G) = ∏ᵢ λᵢ > 0.
        """
        name, G, _ = tensor_info
        det = np.linalg.det(G)
        assert det > 0, f"{name}: det(G) = {det:.6e} ≤ 0"

    def test_positive_trace(self, tensor_info):
        """
        Condición necesaria para SPD: tr(G) > 0.
        Para matrices SPD, tr(G) = Σᵢ λᵢ > 0.
        """
        name, G, _ = tensor_info
        trace = np.trace(G)
        assert trace > 0, f"{name}: tr(G) = {trace:.6e} ≤ 0"

    def test_trace_equals_eigenvalue_sum(self, tensor_info):
        """
        Invariante espectral: tr(G) = Σᵢ λᵢ.
        Verifica consistencia entre traza y eigenvalores.
        """
        name, G, _ = tensor_info
        trace = np.trace(G)
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalue_sum = np.sum(eigenvalues)
        np.testing.assert_allclose(
            trace, eigenvalue_sum,
            rtol=_FLOAT64_RTOL,
            err_msg=f"{name}: tr(G) ≠ Σλᵢ"
        )

    def test_determinant_equals_eigenvalue_product(self, tensor_info):
        """
        Invariante espectral: det(G) = ∏ᵢ λᵢ.
        Verifica consistencia entre determinante y eigenvalores.
        """
        name, G, _ = tensor_info
        det = np.linalg.det(G)
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalue_product = np.prod(eigenvalues)
        np.testing.assert_allclose(
            det, eigenvalue_product,
            rtol=_FLOAT64_RTOL,
            err_msg=f"{name}: det(G) ≠ ∏λᵢ"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 2: PROPIEDADES SPD (SIMÉTRICA DEFINIDA POSITIVA)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSPDProperties:
    """
    Verifica la propiedad de definida positiva mediante múltiples criterios
    independientes para máxima confiabilidad.
    """

    def test_cholesky_succeeds(self, tensor_info):
        """
        Criterio de Cholesky: G ≻ 0 ⟺ ∃L triangular inferior con
        diagonal positiva tal que G = LLᵀ.
        """
        name, G, _ = tensor_info
        try:
            L = np.linalg.cholesky(G)
            # Verificar que L es triangular inferior
            assert np.allclose(L, np.tril(L)), (
                f"{name}: factor de Cholesky no es triangular inferior"
            )
            # Verificar que la diagonal de L es positiva
            assert np.all(np.diag(L) > 0), (
                f"{name}: factor de Cholesky tiene diagonal no positiva"
            )
            # Verificar reconstrucción: LLᵀ ≈ G
            np.testing.assert_allclose(
                L @ L.T, G,
                rtol=_FLOAT64_RTOL,
                atol=_FLOAT64_ATOL,
                err_msg=f"{name}: LLᵀ ≠ G"
            )
        except LinAlgError:
            pytest.fail(f"{name}: descomposición de Cholesky falló (no SPD)")

    def test_all_eigenvalues_positive(self, tensor_info):
        """Todos los eigenvalores deben ser estrictamente positivos."""
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues > 0), (
            f"{name}: eigenvalores no todos positivos: {eigenvalues}"
        )

    def test_eigenvalues_above_threshold(self, tensor_info):
        """
        λ_min(G) ≥ MIN_EIGVAL_TOL: el espectro inferior está acotado
        por el umbral de estabilidad numérica.
        """
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        lambda_min = eigenvalues[0]
        assert lambda_min >= MIN_EIGVAL_TOL, (
            f"{name}: λ_min = {lambda_min:.6e} < MIN_EIGVAL_TOL = {MIN_EIGVAL_TOL:.6e}"
        )

    def test_sylvester_criterion(self, tensor_info):
        """
        Criterio de Sylvester: G ≻ 0 ⟺ todos los menores principales
        líderes son positivos: Δₖ = det(G[0:k, 0:k]) > 0 para k=1,...,n.
        """
        name, G, n = tensor_info
        for k in range(1, n + 1):
            leading_minor = np.linalg.det(G[:k, :k])
            assert leading_minor > 0, (
                f"{name}: menor principal líder Δ_{k} = {leading_minor:.6e} ≤ 0"
            )

    def test_quadratic_form_positive_on_basis_vectors(self, tensor_info):
        """
        Para G ≻ 0, la forma cuadrática xᵀGx > 0 para todo x ≠ 0.
        Verificamos en vectores base canónicos eᵢ.
        """
        name, G, n = tensor_info
        for i in range(n):
            e_i = np.zeros(n, dtype=_FLOAT_DTYPE)
            e_i[i] = 1.0
            quad_form = e_i @ G @ e_i
            assert quad_form > 0, (
                f"{name}: eᵢᵀ G eᵢ = {quad_form:.6e} ≤ 0 para i={i}"
            )

    def test_quadratic_form_positive_on_random_vectors(self, tensor_info):
        """
        Verificación estocástica: xᵀGx > 0 para vectores aleatorios.
        Usa semilla fija para reproducibilidad.
        """
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)
        for trial in range(100):
            x = rng.standard_normal(n)
            x = x / np.linalg.norm(x)  # Normalizar a la esfera unitaria
            quad_form = x @ G @ x
            assert quad_form > 0, (
                f"{name}: xᵀGx = {quad_form:.6e} ≤ 0 para vector aleatorio "
                f"(trial {trial})"
            )

    def test_quadratic_form_bounded_by_eigenvalues(self, tensor_info):
        """
        Para G ≻ 0 y ||x||₂ = 1:
            λ_min ≤ xᵀGx ≤ λ_max

        Esto es el teorema min-max de Courant-Fischer.
        """
        name, G, n = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[-1]

        rng = np.random.default_rng(seed=123)
        for _ in range(100):
            x = rng.standard_normal(n)
            x = x / np.linalg.norm(x)
            quad_form = x @ G @ x
            assert lambda_min - _FLOAT64_ATOL <= quad_form <= lambda_max + _FLOAT64_ATOL, (
                f"{name}: xᵀGx = {quad_form:.6e} fuera de "
                f"[{lambda_min:.6e}, {lambda_max:.6e}]"
            )

    def test_inverse_exists_and_is_spd(self, tensor_info):
        """
        Si G ≻ 0, entonces G⁻¹ existe y G⁻¹ ≻ 0.
        Además, los eigenvalores de G⁻¹ son 1/λᵢ(G).
        """
        name, G, _ = tensor_info
        try:
            G_inv = np.linalg.inv(G)
        except LinAlgError:
            pytest.fail(f"{name}: G no es invertible")

        # Verificar G⁻¹ es SPD
        eigenvalues_inv = np.linalg.eigvalsh(G_inv)
        assert np.all(eigenvalues_inv > 0), (
            f"{name}: G⁻¹ no es SPD, eigenvalores: {eigenvalues_inv}"
        )

        # Verificar eigenvalores de G⁻¹ = 1/eigenvalores de G
        eigenvalues_G = np.linalg.eigvalsh(G)
        expected_inv_eigenvalues = 1.0 / eigenvalues_G
        np.testing.assert_allclose(
            np.sort(eigenvalues_inv),
            np.sort(expected_inv_eigenvalues),
            rtol=_FLOAT64_RTOL,
            err_msg=f"{name}: eigenvalores de G⁻¹ ≠ 1/eigenvalores de G"
        )

        # Verificar G · G⁻¹ ≈ I
        product = G @ G_inv
        np.testing.assert_allclose(
            product,
            np.eye(G.shape[0], dtype=_FLOAT_DTYPE),
            atol=_FLOAT64_ATOL * G.shape[0],
            err_msg=f"{name}: G · G⁻¹ ≠ I"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 3: CONDICIONAMIENTO ESPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpectralConditioning:
    """
    Verifica propiedades del número de condición y estructura espectral.
    """

    def test_condition_number_bounded(self, tensor_info):
        """κ₂(G) = λ_max/λ_min ≤ COND_NUM_TOL."""
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        kappa = eigenvalues[-1] / eigenvalues[0]
        assert kappa <= COND_NUM_TOL, (
            f"{name}: κ₂(G) = {kappa:.4e} > COND_NUM_TOL = {COND_NUM_TOL:.4e}"
        )

    def test_condition_number_finite(self, tensor_info):
        """El número de condición debe ser un valor finito positivo."""
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        kappa = eigenvalues[-1] / eigenvalues[0]
        assert np.isfinite(kappa), f"{name}: κ₂(G) no es finito"
        assert kappa >= 1.0, f"{name}: κ₂(G) = {kappa} < 1 (imposible para SPD)"

    def test_condition_number_consistency_with_numpy(self, tensor_info):
        """
        Verifica que κ₂ calculado por eigenvalores sea consistente
        con np.linalg.cond (que usa SVD).
        """
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        kappa_eigen = eigenvalues[-1] / eigenvalues[0]
        kappa_numpy = np.linalg.cond(G, p=2)
        np.testing.assert_allclose(
            kappa_eigen, kappa_numpy,
            rtol=_FLOAT64_RTOL,
            err_msg=f"{name}: κ₂ por eigenvalores ≠ κ₂ por SVD"
        )

    def test_eigenvalues_ordered_ascending(self, tensor_info):
        """eigvalsh retorna eigenvalores en orden ascendente."""
        name, G, _ = tensor_info
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:]), (
            f"{name}: eigenvalores no ordenados: {eigenvalues}"
        )

    def test_gershgorin_discs_positive(self, tensor_info):
        """
        Teorema de Gershgorin: cada eigenvalor está en algún disco
        Dᵢ = {z : |z - gᵢᵢ| ≤ rᵢ} con rᵢ = Σⱼ≠ᵢ |gᵢⱼ|.

        Si gᵢᵢ - rᵢ > 0 para todo i (dominancia diagonal estricta),
        todos los eigenvalores son positivos.
        """
        name, G, n = tensor_info
        for i in range(n):
            diagonal_element = G[i, i]
            off_diagonal_sum = np.sum(np.abs(G[i, :])) - np.abs(diagonal_element)
            gershgorin_lower = diagonal_element - off_diagonal_sum
            # Nota: no necesariamente dominante diagonal post-regularización,
            # pero los discos deben estar en la semirrecta positiva
            assert gershgorin_lower > -_FLOAT64_ATOL or diagonal_element > off_diagonal_sum, (
                f"{name}: disco de Gershgorin fila {i} cruza el origen: "
                f"gᵢᵢ={diagonal_element:.6e}, rᵢ={off_diagonal_sum:.6e}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 4: INMUTABILIDAD Y SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestImmutability:
    """
    Verifica que los tensores precompilados son inmutables y que la
    inmutabilización funciona correctamente.
    """

    def test_tensor_is_readonly(self, tensor_info):
        """El array precompilado debe tener write=False."""
        name, G, _ = tensor_info
        assert not G.flags.writeable, (
            f"{name}: tensor es mutable (write=True)"
        )

    def test_mutation_raises_error(self, tensor_info):
        """Intentar modificar un tensor inmutable debe lanzar ValueError."""
        name, G, _ = tensor_info
        with pytest.raises(ValueError, match="read-only|not writeable"):
            G[0, 0] = 999.0

    def test_mutation_slice_raises_error(self, tensor_info):
        """Intentar modificar una rebanada debe lanzar ValueError."""
        name, G, _ = tensor_info
        with pytest.raises(ValueError, match="read-only|not writeable"):
            G[0, :] = 0.0

    def test_make_immutable_creates_copy(self, factory, identity_3x3):
        """_make_immutable debe crear una copia independiente."""
        original = identity_3x3.copy()
        immutable = factory._make_immutable(original)

        # Modificar original no debe afectar inmutable
        original[0, 0] = 999.0
        assert immutable[0, 0] == 1.0, (
            "Modificación del original afectó al inmutable"
        )

    def test_make_immutable_preserves_values(self, factory, identity_3x3):
        """_make_immutable debe preservar todos los valores exactamente."""
        immutable = factory._make_immutable(identity_3x3)
        np.testing.assert_array_equal(immutable, identity_3x3)

    def test_make_immutable_preserves_dtype(self, factory, identity_3x3):
        """_make_immutable debe preservar dtype float64."""
        immutable = factory._make_immutable(identity_3x3)
        assert immutable.dtype == _FLOAT_DTYPE


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 5: VALIDACIÓN ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralValidation:
    """
    Verifica que _validate_structural_preconditions rechaza correctamente
    todas las entradas inválidas.
    """

    def test_rejects_none(self, factory):
        """None no es convertible a array numérico."""
        with pytest.raises(MetricTensorError, match="no convertible"):
            factory._validate_structural_preconditions("test", None)

    def test_rejects_string(self, factory):
        """Un string no es una matriz numérica."""
        with pytest.raises(MetricTensorError, match="no convertible"):
            factory._validate_structural_preconditions("test", "not a matrix")

    def test_rejects_1d_array(self, factory):
        """Un vector 1D no es una matriz 2D."""
        vec = np.array([1.0, 2.0, 3.0], dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="2D"):
            factory._validate_structural_preconditions("test", vec)

    def test_rejects_3d_array(self, factory):
        """Un tensor 3D no es una matriz 2D."""
        tensor = np.ones((2, 2, 2), dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="2D"):
            factory._validate_structural_preconditions("test", tensor)

    def test_rejects_rectangular_matrix(self, factory):
        """Una matriz rectangular no es cuadrada."""
        rect = np.ones((2, 3), dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="no cuadrada"):
            factory._validate_structural_preconditions("test", rect)

    def test_rejects_empty_matrix(self, factory):
        """Una matriz 0×0 no tiene dimensión positiva."""
        empty = np.array([], dtype=_FLOAT_DTYPE).reshape(0, 0)
        with pytest.raises(MetricTensorError, match="vacía"):
            factory._validate_structural_preconditions("test", empty)

    def test_rejects_wrong_dimension_g_phys(self, factory):
        """G_phys debe ser 3×3, no 2×2."""
        wrong_dim = np.eye(2, dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="dimensión incorrecta"):
            factory._validate_structural_preconditions("G_phys", wrong_dim)

    def test_rejects_wrong_dimension_g_topo(self, factory):
        """G_topo debe ser 2×2, no 3×3."""
        wrong_dim = np.eye(3, dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="dimensión incorrecta"):
            factory._validate_structural_preconditions("G_topo", wrong_dim)

    def test_rejects_wrong_dimension_g_thermo(self, factory):
        """G_thermo debe ser 2×2, no 4×4."""
        wrong_dim = np.eye(4, dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="dimensión incorrecta"):
            factory._validate_structural_preconditions("G_thermo", wrong_dim)

    def test_accepts_unknown_name_any_dimension(self, factory):
        """Un nombre no registrado acepta cualquier dimensión cuadrada."""
        G = np.eye(5, dtype=_FLOAT_DTYPE)
        result = factory._validate_structural_preconditions("custom", G)
        assert result.shape == (5, 5)

    def test_rejects_nan_entries(self, factory):
        """Entradas NaN son detectadas y rechazadas."""
        G = np.eye(3, dtype=_FLOAT_DTYPE)
        G[1, 1] = np.nan
        with pytest.raises(MetricTensorError, match="no finita"):
            factory._validate_structural_preconditions("test", G)

    def test_rejects_inf_entries(self, factory):
        """Entradas ±∞ son detectadas y rechazadas."""
        G = np.eye(3, dtype=_FLOAT_DTYPE)
        G[0, 2] = np.inf
        with pytest.raises(MetricTensorError, match="no finita"):
            factory._validate_structural_preconditions("test", G)

    def test_rejects_negative_inf(self, factory):
        """Entradas -∞ son detectadas."""
        G = np.eye(2, dtype=_FLOAT_DTYPE)
        G[1, 0] = -np.inf
        with pytest.raises(MetricTensorError, match="no finita"):
            factory._validate_structural_preconditions("test", G)

    def test_reports_non_finite_count_and_indices(self, factory):
        """El mensaje de error incluye conteo e índices de entradas no finitas."""
        G = np.eye(3, dtype=_FLOAT_DTYPE)
        G[0, 1] = np.nan
        G[2, 2] = np.inf
        with pytest.raises(MetricTensorError, match=r"2 entrada\(s\)"):
            factory._validate_structural_preconditions("test", G)

    def test_returns_float64_copy(self, factory):
        """La validación debe retornar una copia float64 independiente."""
        G_int = np.eye(3, dtype=np.int32)
        result = factory._validate_structural_preconditions("test", G_int)
        assert result.dtype == _FLOAT_DTYPE
        # Verificar que es una copia
        assert result is not G_int

    def test_returns_independent_copy(self, factory):
        """Modificar la salida no debe afectar la entrada."""
        G_orig = np.eye(3, dtype=_FLOAT_DTYPE)
        result = factory._validate_structural_preconditions("test", G_orig)
        result[0, 0] = 999.0
        assert G_orig[0, 0] == 1.0

    def test_accepts_list_of_lists(self, factory):
        """Debe aceptar listas anidadas convertibles a array 2D."""
        G_list = [[1.0, 0.0], [0.0, 1.0]]
        result = factory._validate_structural_preconditions("test", G_list)
        np.testing.assert_array_equal(result, np.eye(2, dtype=_FLOAT_DTYPE))


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 6: ANÁLISIS Y SIMETRIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymmetryAnalysis:
    """
    Verifica los métodos de análisis de simetría y simetrización.
    """

    def test_symmetric_matrix_has_zero_asymmetry(self, factory):
        """Una matriz simétrica tiene asimetría relativa cero."""
        G = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=_FLOAT_DTYPE)
        asymmetry = factory._assess_input_symmetry("test", G)
        assert asymmetry == 0.0

    def test_asymmetric_matrix_detected(self, factory):
        """Una matriz asimétrica tiene asimetría relativa positiva."""
        G = np.array([[2.0, 1.5], [0.5, 3.0]], dtype=_FLOAT_DTYPE)
        asymmetry = factory._assess_input_symmetry("test", G)
        assert asymmetry > 0

    def test_asymmetry_quantification(self, factory):
        """Verifica el cálculo correcto de asimetría relativa."""
        G = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=_FLOAT_DTYPE)
        # G - Gᵀ = [[0, 1], [-1, 0]], ||G - Gᵀ||_F = √2
        # ||G||_F = √(4 + 1 + 0 + 9) = √14
        expected = np.sqrt(2.0) / np.sqrt(14.0)
        asymmetry = factory._assess_input_symmetry("test", G)
        np.testing.assert_allclose(asymmetry, expected, rtol=_FLOAT64_RTOL)

    def test_near_zero_matrix_asymmetry(self, factory):
        """Matriz cercana a cero retorna asimetría absoluta."""
        tiny = _NEAR_ZERO_FROBENIUS_TOL * 0.1
        G = np.array([[tiny, tiny / 2], [0, tiny]], dtype=_FLOAT_DTYPE)
        asymmetry = factory._assess_input_symmetry("test", G)
        # Debería ser asimetría absoluta, no relativa
        assert np.isfinite(asymmetry)

    def test_symmetrize_identity(self, factory):
        """Simetrizar la identidad debe dar la identidad."""
        I = np.eye(3, dtype=_FLOAT_DTYPE)
        result = factory._symmetrize_by_projection(I)
        np.testing.assert_array_equal(result, I)

    def test_symmetrize_symmetric_matrix(self, factory):
        """Simetrizar una matriz simétrica no la cambia (idempotencia)."""
        G = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=_FLOAT_DTYPE)
        result = factory._symmetrize_by_projection(G)
        np.testing.assert_array_equal(result, G)

    def test_symmetrize_asymmetric_matrix(self, factory):
        """
        Simetrizar (G + Gᵀ)/2 produce la media de las entradas
        simétricas.
        """
        G = np.array([[2.0, 3.0], [1.0, 4.0]], dtype=_FLOAT_DTYPE)
        result = factory._symmetrize_by_projection(G)
        expected = np.array([[2.0, 2.0], [2.0, 4.0]], dtype=_FLOAT_DTYPE)
        np.testing.assert_array_equal(result, expected)

    def test_symmetrize_result_is_symmetric(self, factory):
        """El resultado de simetrización es exactamente simétrico."""
        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            G = rng.standard_normal((4, 4))
            result = factory._symmetrize_by_projection(G)
            np.testing.assert_array_equal(result, result.T)

    def test_symmetrize_is_closest_symmetric(self, factory):
        """
        π(G) = argmin_{S ∈ Sym(n)} ||G - S||_F

        Verificar que π(G) es estrictamente más cercana a G que cualquier
        otra matriz simétrica (o igual si G ya es simétrica).
        """
        rng = np.random.default_rng(seed=99)
        G = rng.standard_normal((3, 3))
        projected = factory._symmetrize_by_projection(G)
        dist_projected = np.linalg.norm(G - projected, "fro")

        # Cualquier otra perturbación simétrica debe estar al menos tan lejos
        for _ in range(50):
            perturbation = rng.standard_normal((3, 3))
            perturbation = 0.5 * (perturbation + perturbation.T)
            other_symmetric = projected + 0.01 * perturbation
            dist_other = np.linalg.norm(G - other_symmetric, "fro")
            assert dist_projected <= dist_other + _FLOAT64_ATOL

    def test_symmetrize_idempotent(self, factory):
        """π² = π: aplicar dos veces da el mismo resultado."""
        rng = np.random.default_rng(seed=77)
        G = rng.standard_normal((4, 4))
        once = factory._symmetrize_by_projection(G)
        twice = factory._symmetrize_by_projection(once)
        np.testing.assert_array_equal(once, twice)

    def test_compute_symmetry_tolerance_scales_with_norm(self, factory):
        """La tolerancia de simetría escala con ||G||_F."""
        G_small = np.eye(2, dtype=_FLOAT_DTYPE) * 1e-10
        G_large = np.eye(2, dtype=_FLOAT_DTYPE) * 1e10
        tol_small = factory._compute_symmetry_tolerance(G_small)
        tol_large = factory._compute_symmetry_tolerance(G_large)
        assert tol_large > tol_small

    def test_compute_symmetry_tolerance_has_floor(self, factory):
        """La tolerancia nunca es menor que _SYMMETRY_ATOL_BASE."""
        G_zero = np.zeros((2, 2), dtype=_FLOAT_DTYPE)
        tol = factory._compute_symmetry_tolerance(G_zero)
        assert tol >= _SYMMETRY_ATOL_BASE


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 7: PERFIL ESPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpectralProfile:
    """
    Verifica el cálculo y propiedades del perfil espectral.
    """

    def test_identity_spectral_profile(self, factory):
        """Perfil espectral de la identidad: todos los eigenvalores son 1."""
        I = np.eye(3, dtype=_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(I)
        assert profile.lambda_min == 1.0
        assert profile.lambda_max == 1.0
        assert profile.condition_number == 1.0
        assert profile.spectral_gap == 0.0
        np.testing.assert_array_equal(profile.eigenvalues, [1.0, 1.0, 1.0])

    def test_diagonal_matrix_spectral_profile(self, factory):
        """
        Perfil espectral de una matriz diagonal con eigenvalores conocidos.
        """
        D = np.diag([1.0, 3.0, 5.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(D)
        assert profile.lambda_min == 1.0
        assert profile.lambda_max == 5.0
        np.testing.assert_allclose(profile.condition_number, 5.0, rtol=_FLOAT64_RTOL)

    def test_spectral_gap_normalized_by_lambda_max(self, factory):
        """
        Brecha espectral = (λ₂ - λ₁)/λ_max.
        Para D = diag(1, 3, 5): gap = (3 - 1)/5 = 0.4.
        """
        D = np.diag([1.0, 3.0, 5.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(D)
        expected_gap = (3.0 - 1.0) / 5.0
        np.testing.assert_allclose(
            profile.spectral_gap, expected_gap, rtol=_FLOAT64_RTOL
        )

    def test_spectral_gap_bounded_in_zero_one(self, tensor_info):
        """
        Para matrices SPD, la brecha espectral normalizada por λ_max
        está en [0, 1).
        """
        name, G, _ = tensor_info
        profile = MetricTensorFactory._compute_spectral_profile(G)
        assert 0.0 <= profile.spectral_gap < 1.0, (
            f"{name}: brecha espectral {profile.spectral_gap} fuera de [0, 1)"
        )

    def test_frobenius_norm_correct(self, factory):
        """Verificar que ||G||_F coincide con np.linalg.norm."""
        D = np.diag([1.0, 2.0, 3.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(D)
        expected = np.linalg.norm(D, "fro")
        np.testing.assert_allclose(profile.frobenius_norm, expected, rtol=_FLOAT64_RTOL)

    def test_frobenius_norm_equals_sqrt_eigenvalue_squares_sum(self, factory):
        """
        Para matrices simétricas: ||G||_F = √(Σ λᵢ²).
        Esto es porque ||G||_F² = tr(GᵀG) = tr(G²) = Σ λᵢ².
        """
        D = np.diag([2.0, 3.0, 5.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(D)
        expected = np.sqrt(np.sum(profile.eigenvalues ** 2))
        np.testing.assert_allclose(
            profile.frobenius_norm, expected, rtol=_FLOAT64_RTOL
        )

    def test_eigenvalues_are_immutable(self, factory):
        """Los eigenvalores en SpectralProfile deben ser read-only."""
        I = np.eye(3, dtype=_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(I)
        with pytest.raises(ValueError, match="read-only|not writeable"):
            profile.eigenvalues[0] = 999.0

    def test_rejects_non_symmetric_input(self, factory):
        """_compute_spectral_profile debe rechazar matrices no simétricas."""
        G = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="no simétrica"):
            factory._compute_spectral_profile(G)

    def test_is_strictly_positive_true(self):
        """is_strictly_positive retorna True cuando λ_min ≥ threshold."""
        eigenvalues = np.array([1.0, 2.0], dtype=_FLOAT_DTYPE)
        eigenvalues.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigenvalues,
            lambda_min=1.0,
            lambda_max=2.0,
            condition_number=2.0,
            spectral_gap=0.5,
            frobenius_norm=np.sqrt(5.0),
        )
        assert profile.is_strictly_positive(threshold=0.5)

    def test_is_strictly_positive_false(self):
        """is_strictly_positive retorna False cuando λ_min < threshold."""
        eigenvalues = np.array([0.001, 2.0], dtype=_FLOAT_DTYPE)
        eigenvalues.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigenvalues,
            lambda_min=0.001,
            lambda_max=2.0,
            condition_number=2000.0,
            spectral_gap=0.9995,
            frobenius_norm=2.0,
        )
        assert not profile.is_strictly_positive(threshold=0.01)

    def test_is_well_conditioned_true(self):
        """is_well_conditioned retorna True cuando κ ≤ max_kappa."""
        eigenvalues = np.array([1.0, 2.0], dtype=_FLOAT_DTYPE)
        eigenvalues.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigenvalues,
            lambda_min=1.0,
            lambda_max=2.0,
            condition_number=2.0,
            spectral_gap=0.5,
            frobenius_norm=np.sqrt(5.0),
        )
        assert profile.is_well_conditioned(max_kappa=10.0)

    def test_is_well_conditioned_false_for_inf(self):
        """is_well_conditioned retorna False para κ = ∞."""
        eigenvalues = np.array([0.0, 1.0], dtype=_FLOAT_DTYPE)
        eigenvalues.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigenvalues,
            lambda_min=0.0,
            lambda_max=1.0,
            condition_number=np.inf,
            spectral_gap=0.0,
            frobenius_norm=1.0,
        )
        assert not profile.is_well_conditioned()


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 8: VERIFICACIÓN SPD POR CHOLESKY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCholeskyVerification:
    """
    Verifica el método de certificación SPD por Cholesky.
    """

    def test_spd_matrix_passes(self, factory):
        """Una matriz SPD pasa la verificación de Cholesky."""
        G = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=_FLOAT_DTYPE)
        assert factory._verify_spd_by_cholesky("test", G) is True

    def test_identity_passes(self, factory, identity_3x3):
        """La identidad pasa Cholesky trivialmente."""
        assert factory._verify_spd_by_cholesky("test", identity_3x3) is True

    def test_negative_definite_fails(self, factory):
        """Una matriz negativa definida falla Cholesky."""
        G = -np.eye(2, dtype=_FLOAT_DTYPE)
        assert factory._verify_spd_by_cholesky("test", G) is False

    def test_semi_definite_fails(self, factory):
        """Una matriz semidefinida positiva (singular) falla Cholesky."""
        G = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=_FLOAT_DTYPE)
        assert factory._verify_spd_by_cholesky("test", G) is False

    def test_indefinite_fails(self, factory):
        """Una matriz indefinida falla Cholesky."""
        G = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=_FLOAT_DTYPE)
        assert factory._verify_spd_by_cholesky("test", G) is False

    def test_zero_matrix_fails(self, factory):
        """La matriz cero no es SPD."""
        G = np.zeros((2, 2), dtype=_FLOAT_DTYPE)
        assert factory._verify_spd_by_cholesky("test", G) is False


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 9: REGULARIZACIÓN TIKHONOV
# ═══════════════════════════════════════════════════════════════════════════════

class TestTikhonovRegularization:
    """
    Verifica la regularización de Tikhonov: G_reg = G + δI.
    """

    def test_no_regularization_when_well_conditioned(self, factory):
        """No se aplica regularización si λ_min ≥ target."""
        G = np.eye(3, dtype=_FLOAT_DTYPE) * 2.0
        profile = factory._compute_spectral_profile(G)
        G_reg, delta = factory._apply_tikhonov_regularization("test", G, profile)
        assert delta == 0.0
        np.testing.assert_array_equal(G_reg, G)

    def test_regularization_applied_when_needed(self, factory, near_singular_spd_2x2):
        """Se aplica regularización cuando λ_min < target."""
        profile = factory._compute_spectral_profile(near_singular_spd_2x2)
        G_reg, delta = factory._apply_tikhonov_regularization(
            "test", near_singular_spd_2x2, profile
        )
        assert delta > _TIKHONOV_DELTA_ATOL
        # Verificar que λ_min(G_reg) ≈ target
        target = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR
        reg_eigenvalues = np.linalg.eigvalsh(G_reg)
        np.testing.assert_allclose(
            reg_eigenvalues[0], target, rtol=_FLOAT64_RTOL
        )

    def test_regularization_preserves_eigenvectors(self, factory):
        """
        G_reg = G + δI preserva los autovectores de G.
        Los autovectores de G y G + δI deben ser los mismos.
        """
        G = np.array([[2.0, 1.0], [1.0, 0.5 * MIN_EIGVAL_TOL]], dtype=_FLOAT_DTYPE)
        G = 0.5 * (G + G.T)

        # Eigendecomposición original
        _, V_orig = np.linalg.eigh(G)

        profile = factory._compute_spectral_profile(G)
        G_reg, delta = factory._apply_tikhonov_regularization("test", G, profile)

        if delta > _TIKHONOV_DELTA_ATOL:
            _, V_reg = np.linalg.eigh(G_reg)
            # Los autovectores pueden diferir en signo
            for i in range(G.shape[0]):
                dot_product = abs(np.dot(V_orig[:, i], V_reg[:, i]))
                np.testing.assert_allclose(
                    dot_product, 1.0, atol=1e-10,
                    err_msg=f"Autovector {i} cambió tras regularización"
                )

    def test_regularization_shifts_eigenvalues_by_delta(self, factory):
        """λᵢ(G + δI) = λᵢ(G) + δ exactamente (en aritmética exacta)."""
        G = np.diag([0.001, 1.0, 5.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(G)
        G_reg, delta = factory._apply_tikhonov_regularization("test", G, profile)

        if delta > _TIKHONOV_DELTA_ATOL:
            expected_eigenvalues = profile.eigenvalues + delta
            actual_eigenvalues = np.linalg.eigvalsh(G_reg)
            np.testing.assert_allclose(
                actual_eigenvalues, expected_eigenvalues,
                rtol=_FLOAT64_RTOL,
                atol=_FLOAT64_ATOL,
            )

    def test_delta_computation_correct(self, factory):
        """
        δ = target_λ_min - λ_min donde target_λ_min = MIN_EIGVAL_TOL · factor.
        """
        small_eigval = MIN_EIGVAL_TOL * 0.5
        G = np.diag([small_eigval, 2.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(G)
        _, delta = factory._apply_tikhonov_regularization("test", G, profile)

        expected_delta = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR - small_eigval
        np.testing.assert_allclose(delta, expected_delta, rtol=_FLOAT64_RTOL)

    def test_regularized_matrix_is_spd(self, factory):
        """La matriz regularizada debe ser SPD."""
        G = np.diag([1e-15, 1.0]).astype(_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(G)
        G_reg, _ = factory._apply_tikhonov_regularization("test", G, profile)
        assert factory._verify_spd_by_cholesky("test", G_reg)

    def test_regularized_matrix_preserves_symmetry(self, factory):
        """La regularización G + δI preserva simetría exactamente."""
        G = np.array([[1.0, 0.5], [0.5, 0.001]], dtype=_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(G)
        G_reg, _ = factory._apply_tikhonov_regularization("test", G, profile)
        np.testing.assert_array_equal(G_reg, G_reg.T)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 10: DEFORMACIÓN POR REGULARIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegularizationDeformation:
    """
    Verifica el cálculo de deformación relativa por regularización.
    """

    def test_zero_deformation_no_regularization(self, factory):
        """Sin regularización, la deformación es cero."""
        G = np.eye(3, dtype=_FLOAT_DTYPE)
        deformation = factory._compute_regularization_deformation(G, G, 0.0)
        assert deformation == 0.0

    def test_deformation_formula_correct(self, factory):
        """
        Deformación = ||G_reg - G||_F / ||G||_F = δ√n / ||G||_F.
        """
        G = np.diag([1.0, 2.0, 3.0]).astype(_FLOAT_DTYPE)
        delta = 0.5
        G_reg = G + delta * np.eye(3, dtype=_FLOAT_DTYPE)
        deformation = factory._compute_regularization_deformation(G, G_reg, delta)

        expected_num = delta * np.sqrt(3)
        expected_den = np.linalg.norm(G, "fro")
        expected = expected_num / expected_den

        np.testing.assert_allclose(deformation, expected, rtol=_FLOAT64_RTOL)

    def test_near_zero_matrix_raises_error(self, factory):
        """Matriz esencialmente nula genera error (no admite deformación relativa)."""
        G_zero = np.zeros((2, 2), dtype=_FLOAT_DTYPE)
        G_reg = np.eye(2, dtype=_FLOAT_DTYPE) * 0.01
        with pytest.raises(MetricTensorError, match="esencialmente nula"):
            factory._compute_regularization_deformation(G_zero, G_reg, 0.01)

    def test_small_norm_matrix_raises_error(self, factory):
        """Matriz con norma menor que tolerancia genera error."""
        tiny = _NEAR_ZERO_FROBENIUS_TOL * 0.1
        G = np.array([[tiny, 0], [0, tiny]], dtype=_FLOAT_DTYPE)
        G_reg = G + np.eye(2, dtype=_FLOAT_DTYPE) * 0.01
        with pytest.raises(MetricTensorError):
            factory._compute_regularization_deformation(G, G_reg, 0.01)

    def test_deformation_proportional_to_delta(self, factory):
        """La deformación escala linealmente con δ."""
        G = np.eye(2, dtype=_FLOAT_DTYPE) * 5.0
        deformations = []
        deltas = [0.01, 0.02, 0.04]
        for d in deltas:
            G_reg = G + d * np.eye(2, dtype=_FLOAT_DTYPE)
            def_val = factory._compute_regularization_deformation(G, G_reg, d)
            deformations.append(def_val)

        # Verificar linealidad: def(2δ) / def(δ) ≈ 2
        ratio_1 = deformations[1] / deformations[0]
        ratio_2 = deformations[2] / deformations[0]
        np.testing.assert_allclose(ratio_1, 2.0, rtol=_FLOAT64_RTOL)
        np.testing.assert_allclose(ratio_2, 4.0, rtol=_FLOAT64_RTOL)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 11: CONSISTENCIA ESPECTRAL POST-TIKHONOV
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpectralConsistency:
    """
    Verifica la consistencia λᵢ(G + δI) = λᵢ(G) + δ.
    """

    def test_no_check_without_regularization(self, factory):
        """Sin regularización (δ < atol), no se verifica nada."""
        eigvals = np.array([1.0, 2.0], dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=1.0,
            lambda_max=2.0,
            condition_number=2.0,
            spectral_gap=0.5,
            frobenius_norm=np.sqrt(5.0),
        )
        # No debería lanzar ninguna excepción ni advertencia
        factory._verify_spectral_consistency("test", profile, profile, 0.0)

    def test_consistent_eigenvalues_pass(self, factory):
        """Eigenvalores consistentes no generan advertencia."""
        G = np.diag([1.0, 3.0, 5.0]).astype(_FLOAT_DTYPE)
        original_profile = factory._compute_spectral_profile(G)

        delta = 0.5
        G_reg = G + delta * np.eye(3, dtype=_FLOAT_DTYPE)
        reg_profile = factory._compute_spectral_profile(G_reg)

        # No debería lanzar excepción
        factory._verify_spectral_consistency(
            "test", original_profile, reg_profile, delta
        )

    def test_eigenvalue_shift_is_exact(self, factory):
        """
        Verifica que para matrices diagonales, el shift es exacto.
        """
        eigenvalues = [0.5, 2.0, 7.0]
        G = np.diag(eigenvalues).astype(_FLOAT_DTYPE)
        delta = 1.0
        G_reg = G + delta * np.eye(3, dtype=_FLOAT_DTYPE)

        expected = np.array([e + delta for e in eigenvalues], dtype=_FLOAT_DTYPE)
        actual = np.linalg.eigvalsh(G_reg)

        np.testing.assert_allclose(actual, expected, rtol=_FLOAT64_RTOL)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 12: VERIFICACIÓN SPD ESTRICTA POST-REGULARIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrictSPDAssertion:
    """
    Verifica que _assert_spd_strict detecta correctamente todas las
    violaciones de las condiciones SPD.
    """

    def _make_profile(self, eigenvalues, condition_number=None):
        """Helper para crear SpectralProfile de prueba."""
        eigvals = np.array(eigenvalues, dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])
        if condition_number is None:
            condition_number = lambda_max / lambda_min if lambda_min > 0 else np.inf
        return SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            condition_number=condition_number,
            spectral_gap=0.0,
            frobenius_norm=np.sqrt(np.sum(eigvals ** 2)),
        )

    def test_valid_spd_passes(self, factory):
        """Una matriz SPD válida pasa todas las verificaciones."""
        G = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=_FLOAT_DTYPE)
        profile = factory._compute_spectral_profile(G)
        factory._assert_spd_strict("test", G, profile)  # No debería lanzar

    def test_non_finite_entries_detected(self, factory):
        """Entradas no finitas post-regularización son detectadas."""
        G = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=_FLOAT_DTYPE)
        profile = self._make_profile([0.5, 1.5])
        with pytest.raises(MetricTensorError, match="no finitos"):
            factory._assert_spd_strict("test", G, profile)

    def test_negative_lambda_max_detected(self, factory):
        """λ_max ≤ 0 es detectado."""
        G = -np.eye(2, dtype=_FLOAT_DTYPE)
        profile = self._make_profile([-1.0, -1.0], condition_number=1.0)
        with pytest.raises(MetricTensorError, match="λ_max"):
            factory._assert_spd_strict("test", G, profile)

    def test_lambda_min_below_threshold_detected(self, factory):
        """λ_min <= MIN_EIGVAL_TOL es detectado (divergencia de número de condición inminente)."""
        tiny = MIN_EIGVAL_TOL
        G = np.diag([tiny, 1.0]).astype(_FLOAT_DTYPE)
        profile = self._make_profile([tiny, 1.0])
        with pytest.raises(MetricTensorError, match="λ_min.*?<="):
            factory._assert_spd_strict("test", G, profile)

    def test_sylvester_criterion_in_strict_spd(self, factory):
        """Criterio de Sylvester (menores principales positivos) falla y es detectado."""
        # Una matriz con λ_min y λ_max positivos pero que no es SPD debido
        # a componentes fuera de la diagonal muy grandes
        G = np.array([[1.0, 10.0], [10.0, 1.0]], dtype=_FLOAT_DTYPE)
        profile = self._make_profile([0.5, 1.5])  # valores falsos para saltar los checks previos
        with pytest.raises(MetricTensorError, match="Criterio de Sylvester"):
            factory._assert_spd_strict("test", G, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 13: CONTROL DE CONDICIONAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

class TestConditionNumberControl:
    """
    Verifica _assert_well_conditioned.
    """

    def _make_profile_with_kappa(self, kappa):
        """Helper para crear perfil con κ₂ específico."""
        eigvals = np.array([1.0, kappa], dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        return SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=1.0,
            lambda_max=kappa,
            condition_number=kappa,
            spectral_gap=(kappa - 1.0) / kappa,
            frobenius_norm=np.sqrt(1.0 + kappa ** 2),
        )

    def test_well_conditioned_passes(self, factory):
        """κ₂ ≤ COND_NUM_TOL pasa."""
        profile = self._make_profile_with_kappa(COND_NUM_TOL * 0.5)
        factory._assert_well_conditioned("test", profile)  # No debería lanzar

    def test_exactly_at_threshold_passes(self, factory):
        """κ₂ = COND_NUM_TOL exacto pasa."""
        profile = self._make_profile_with_kappa(COND_NUM_TOL)
        factory._assert_well_conditioned("test", profile)  # No debería lanzar

    def test_above_threshold_fails(self, factory):
        """κ₂ > COND_NUM_TOL falla."""
        profile = self._make_profile_with_kappa(COND_NUM_TOL * 1.1)
        with pytest.raises(MetricTensorError, match="mal condicionado"):
            factory._assert_well_conditioned("test", profile)

    def test_infinite_condition_number_fails(self, factory):
        """κ₂ = ∞ falla."""
        eigvals = np.array([0.0, 1.0], dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=0.0,
            lambda_max=1.0,
            condition_number=np.inf,
            spectral_gap=0.0,
            frobenius_norm=1.0,
        )
        with pytest.raises(MetricTensorError, match="no finito"):
            factory._assert_well_conditioned("test", profile)

    def test_nan_condition_number_fails(self, factory):
        """κ₂ = NaN falla."""
        eigvals = np.array([-1.0, 1.0], dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        profile = SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=-1.0,
            lambda_max=1.0,
            condition_number=np.nan,
            spectral_gap=0.0,
            frobenius_norm=np.sqrt(2.0),
        )
        with pytest.raises(MetricTensorError, match="no finito"):
            factory._assert_well_conditioned("test", profile)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 14: PIPELINE COMPLETO _validate_and_regularize
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    Pruebas de integración del pipeline completo.
    """

    def test_identity_passes_without_regularization(self, factory):
        """La identidad pasa sin regularización."""
        I = np.eye(3, dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", I)
        np.testing.assert_allclose(result, I, rtol=_FLOAT64_RTOL)

    def test_well_conditioned_spd_passes(self, factory, well_conditioned_spd_3x3):
        """Una matriz SPD bien condicionada pasa el pipeline."""
        result = factory._validate_and_regularize("test", well_conditioned_spd_3x3)
        assert result.dtype == _FLOAT_DTYPE
        assert not result.flags.writeable

    def test_near_singular_gets_regularized(self, factory):
        """Una matriz con λ_min pequeño se regulariza correctamente."""
        tiny = MIN_EIGVAL_TOL * 0.01
        G = np.diag([tiny, 1.0]).astype(_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)

        # El resultado debe ser SPD
        eigenvalues = np.linalg.eigvalsh(result)
        assert eigenvalues[0] >= MIN_EIGVAL_TOL

    def test_asymmetric_input_gets_symmetrized(self, factory):
        """Una entrada ligeramente asimétrica se simetriza."""
        G = np.array([[2.0, 1.0001], [0.9999, 3.0]], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        np.testing.assert_array_equal(result, result.T)

    def test_negative_definite_after_regularization_aborts(self, factory):
        """
        Una matriz fuertemente negativa definida genera deformación excesiva
        y es rechazada.
        """
        G = -np.eye(2, dtype=_FLOAT_DTYPE) * 10.0
        with pytest.raises(MetricTensorError):
            factory._validate_and_regularize("test", G)

    def test_output_is_immutable(self, factory):
        """La salida del pipeline es inmutable."""
        G = np.eye(2, dtype=_FLOAT_DTYPE) * 2.0
        result = factory._validate_and_regularize("test", G)
        assert not result.flags.writeable

    def test_output_dtype_is_float64(self, factory):
        """La salida es siempre float64."""
        G = np.eye(2, dtype=np.float32) * 2.0
        result = factory._validate_and_regularize("test", G)
        assert result.dtype == _FLOAT_DTYPE

    def test_pipeline_rejects_scalar(self, factory):
        """Un escalar no es un tensor 2D."""
        with pytest.raises(MetricTensorError):
            factory._validate_and_regularize("test", np.float64(5.0))

    def test_pipeline_preserves_well_conditioned_matrix(self, factory):
        """
        Una matriz SPD bien condicionada debe pasar sin deformación
        (o con deformación despreciable).
        """
        G = np.array([[5.0, 1.0], [1.0, 4.0]], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        np.testing.assert_allclose(result, G, rtol=_FLOAT64_RTOL)

    def test_pipeline_with_integer_input(self, factory):
        """Acepta entradas enteras convertibles a float64."""
        G = np.array([[3, 1], [1, 2]], dtype=np.int64)
        result = factory._validate_and_regularize("test", G)
        assert result.dtype == _FLOAT_DTYPE
        expected = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=_FLOAT_DTYPE)
        np.testing.assert_allclose(result, expected, rtol=_FLOAT64_RTOL)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 15: CONSTRUCTORES DE TENSORES ESPECÍFICOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorConstructors:
    """
    Verifica los constructores de tensores específicos por dominio.
    """

    def test_physics_tensor_shape(self, factory):
        """G_phys es 3×3."""
        G = factory.build_physics_tensor()
        assert G.shape == (3, 3)

    def test_topology_tensor_shape(self, factory):
        """G_topo es 2×2."""
        G = factory.build_topology_tensor()
        assert G.shape == (2, 2)

    def test_thermo_tensor_shape(self, factory):
        """G_thermo es 2×2."""
        G = factory.build_thermo_tensor()
        assert G.shape == (2, 2)

    def test_physics_tensor_diagonal_dominance(self, factory):
        """
        G_phys debe satisfacer dominancia diagonal estricta por filas:
        gᵢᵢ > Σⱼ≠ᵢ |gᵢⱼ| para todo i.
        """
        G = factory.build_physics_tensor()
        n = G.shape[0]
        for i in range(n):
            diagonal = G[i, i]
            off_diagonal_sum = np.sum(np.abs(G[i, :])) - abs(diagonal)
            assert diagonal > off_diagonal_sum, (
                f"Fila {i}: gᵢᵢ={diagonal:.4f} ≤ Σ|gᵢⱼ|={off_diagonal_sum:.4f}"
            )

    def test_topology_tensor_sylvester(self, factory):
        """
        G_topo debe satisfacer criterio de Sylvester:
        g₀₀ > 0 y det(G) > 0.
        """
        G = factory.build_topology_tensor()
        assert G[0, 0] > 0
        det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
        assert det > 0

    def test_thermo_tensor_sylvester(self, factory):
        """
        G_thermo debe satisfacer criterio de Sylvester.
        """
        G = factory.build_thermo_tensor()
        assert G[0, 0] > 0
        det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
        assert det > 0

    def test_physics_tensor_known_entries(self, factory):
        """
        Verifica las entradas conocidas de G_phys (pueden diferir por
        regularización, pero deben estar muy cercanas).
        """
        G = factory.build_physics_tensor()
        expected_diagonal = [2.50, 1.50, 1.00]
        for i, expected in enumerate(expected_diagonal):
            np.testing.assert_allclose(
                G[i, i], expected, atol=0.01,
                err_msg=f"G_phys[{i},{i}] inesperado"
            )

    def test_topology_tensor_known_entries(self, factory):
        """Verifica las entradas conocidas de G_topo."""
        G = factory.build_topology_tensor()
        np.testing.assert_allclose(G[0, 0], 1.00, atol=0.01)
        np.testing.assert_allclose(G[1, 1], 3.00, atol=0.01)
        np.testing.assert_allclose(G[0, 1], 0.60, atol=0.01)

    def test_thermo_tensor_known_entries(self, factory):
        """Verifica las entradas conocidas de G_thermo."""
        G = factory.build_thermo_tensor()
        np.testing.assert_allclose(G[0, 0], 1.80, atol=0.01)
        np.testing.assert_allclose(G[1, 1], 2.20, atol=0.01)
        np.testing.assert_allclose(G[0, 1], 0.75, atol=0.01)

    def test_constructors_are_deterministic(self, factory):
        """Llamar al constructor múltiples veces da el mismo resultado."""
        G1 = factory.build_physics_tensor()
        G2 = factory.build_physics_tensor()
        np.testing.assert_array_equal(G1, G2)

        G3 = factory.build_topology_tensor()
        G4 = factory.build_topology_tensor()
        np.testing.assert_array_equal(G3, G4)

        G5 = factory.build_thermo_tensor()
        G6 = factory.build_thermo_tensor()
        np.testing.assert_array_equal(G5, G6)

    def test_constructors_return_independent_copies(self, factory):
        """Cada llamada al constructor retorna un objeto independiente."""
        G1 = factory.build_physics_tensor()
        G2 = factory.build_physics_tensor()
        assert G1 is not G2
        # Verificar que son arrays diferentes en memoria
        assert not np.shares_memory(G1, G2)

    def test_topology_tensor_analytic_eigenvalues(self, factory):
        """
        Verifica eigenvalores analíticos de G_topo:
        tr = 4.0, det = 2.64
        λ± = (4 ± √5.44) / 2 ≈ (0.8338, 3.1662)
        """
        G = factory.build_topology_tensor()
        eigenvalues = np.linalg.eigvalsh(G)
        discriminant = np.sqrt(4.0 ** 2 - 4.0 * 2.64)
        expected_min = (4.0 - discriminant) / 2.0
        expected_max = (4.0 + discriminant) / 2.0
        np.testing.assert_allclose(eigenvalues[0], expected_min, rtol=1e-6)
        np.testing.assert_allclose(eigenvalues[1], expected_max, rtol=1e-6)

    def test_thermo_tensor_analytic_eigenvalues(self, factory):
        """
        Verifica eigenvalores analíticos de G_thermo:
        tr = 4.0, det = 3.3975
        λ± = (4 ± √2.41) / 2 ≈ (1.2238, 2.7762)
        """
        G = factory.build_thermo_tensor()
        eigenvalues = np.linalg.eigvalsh(G)
        discriminant = np.sqrt(4.0 ** 2 - 4.0 * 3.3975)
        expected_min = (4.0 - discriminant) / 2.0
        expected_max = (4.0 + discriminant) / 2.0
        np.testing.assert_allclose(eigenvalues[0], expected_min, rtol=1e-6)
        np.testing.assert_allclose(eigenvalues[1], expected_max, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 16: REGULARIZATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegularizationReport:
    """
    Verifica la estructura y logging del RegularizationReport.
    """

    def _make_simple_profile(self, lambda_min=1.0, lambda_max=5.0):
        """Helper para crear perfiles simples."""
        eigvals = np.array([lambda_min, lambda_max], dtype=_FLOAT_DTYPE)
        eigvals.setflags(write=False)
        kappa = lambda_max / lambda_min if lambda_min > 0 else np.inf
        return SpectralProfile(
            eigenvalues=eigvals,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            condition_number=kappa,
            spectral_gap=(lambda_max - lambda_min) / lambda_max if lambda_max > 0 else 0.0,
            frobenius_norm=np.sqrt(lambda_min ** 2 + lambda_max ** 2),
        )

    def test_report_is_frozen(self):
        """RegularizationReport es inmutable (frozen dataclass)."""
        profile = self._make_simple_profile()
        report = RegularizationReport(
            tensor_name="test",
            original_profile=profile,
            regularized_profile=profile,
            tikhonov_delta=0.0,
            frobenius_deformation=0.0,
            required_regularization=False,
            input_asymmetry=0.0,
        )
        with pytest.raises(AttributeError):
            report.tensor_name = "modified"

    def test_report_stores_input_asymmetry(self):
        """El informe incluye la asimetría de entrada."""
        profile = self._make_simple_profile()
        report = RegularizationReport(
            tensor_name="test",
            original_profile=profile,
            regularized_profile=profile,
            tikhonov_delta=0.0,
            frobenius_deformation=0.0,
            required_regularization=False,
            input_asymmetry=0.001,
        )
        assert report.input_asymmetry == 0.001

    def test_report_log_summary_no_regularization(self, caplog):
        """Log de resumen para tensor sin regularización."""
        profile = self._make_simple_profile()
        report = RegularizationReport(
            tensor_name="test_tensor",
            original_profile=profile,
            regularized_profile=profile,
            tikhonov_delta=0.0,
            frobenius_deformation=0.0,
            required_regularization=False,
            input_asymmetry=0.0,
        )
        with caplog.at_level(logging.DEBUG, logger="MIC.ImmuneSystem.MetricTensors"):
            report.log_summary()
        assert "test_tensor" in caplog.text
        assert "sin regularización" in caplog.text

    def test_report_log_summary_with_regularization(self, caplog):
        """Log de resumen para tensor con regularización."""
        orig = self._make_simple_profile(lambda_min=0.001, lambda_max=5.0)
        reg = self._make_simple_profile(lambda_min=0.1, lambda_max=5.1)
        report = RegularizationReport(
            tensor_name="test_reg",
            original_profile=orig,
            regularized_profile=reg,
            tikhonov_delta=0.099,
            frobenius_deformation=0.02,
            required_regularization=True,
            input_asymmetry=0.0,
        )
        with caplog.at_level(logging.INFO, logger="MIC.ImmuneSystem.MetricTensors"):
            report.log_summary()
        assert "test_reg" in caplog.text
        assert "Regularización aplicada" in caplog.text


# Necesario para los tests de logging
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 17: DIAGNÓSTICOS EXTERNOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorDiagnostics:
    """
    Verifica la función get_tensor_diagnostics.
    """

    def test_diagnostics_of_identity(self):
        """Diagnósticos de la identidad son correctos."""
        diag = get_tensor_diagnostics(np.eye(3, dtype=_FLOAT_DTYPE), "I_3")
        assert diag["name"] == "I_3"
        assert diag["shape"] == (3, 3)
        assert diag["is_symmetric"] is True
        assert diag["is_spd"] is True
        assert diag["lambda_min"] == 1.0
        assert diag["lambda_max"] == 1.0
        assert diag["condition_number"] == 1.0
        assert diag["is_well_conditioned"] is True
        assert diag["is_strictly_positive"] is True

    def test_diagnostics_of_precompiled_tensors(self, tensor_info):
        """Diagnósticos de tensores precompilados son coherentes."""
        name, G, expected_dim = tensor_info
        diag = get_tensor_diagnostics(G, name)
        assert diag["name"] == name
        assert diag["shape"] == (expected_dim, expected_dim)
        assert diag["is_symmetric"] is True
        assert diag["is_spd"] is True
        assert diag["is_well_conditioned"] is True
        assert diag["is_strictly_positive"] is True
        assert diag["lambda_min"] >= MIN_EIGVAL_TOL
        assert diag["condition_number"] <= COND_NUM_TOL

    def test_diagnostics_of_invalid_input_returns_error(self):
        """Entrada inválida retorna diccionario con error."""
        diag = get_tensor_diagnostics("not_a_matrix", "invalid")
        assert "error" in diag
        assert "error_type" in diag
        assert diag["name"] == "invalid"

    def test_diagnostics_of_non_square_returns_error(self):
        """Matriz no cuadrada retorna error estructural."""
        G = np.ones((2, 3), dtype=_FLOAT_DTYPE)
        diag = get_tensor_diagnostics(G, "rect")
        assert "error" in diag

    def test_diagnostics_includes_asymmetry_info(self):
        """Los diagnósticos incluyen información de asimetría."""
        G = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=_FLOAT_DTYPE)
        diag = get_tensor_diagnostics(G, "sym_test")
        assert "asymmetry_norm" in diag
        assert "symmetry_tolerance" in diag
        assert diag["asymmetry_norm"] == 0.0

    def test_diagnostics_eigenvalues_are_list(self):
        """Los eigenvalores en diagnósticos son lista (serializable JSON)."""
        diag = get_tensor_diagnostics(np.eye(2, dtype=_FLOAT_DTYPE), "test")
        assert isinstance(diag["eigenvalues"], list)
        assert len(diag["eigenvalues"]) == 2

    def test_diagnostics_non_spd_matrix(self):
        """Diagnósticos de matriz no SPD reportan is_spd=False."""
        G = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=_FLOAT_DTYPE)
        diag = get_tensor_diagnostics(G, "non_spd")
        assert diag["is_spd"] is False

    def test_diagnostics_default_name(self):
        """El nombre por defecto es 'unnamed'."""
        diag = get_tensor_diagnostics(np.eye(2, dtype=_FLOAT_DTYPE))
        assert diag["name"] == "unnamed"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 18: PROPIEDADES GEOMÉTRICAS DEL CONO SPD
# ═══════════════════════════════════════════════════════════════════════════════

class TestSPDConeProperties:
    """
    Verifica propiedades geométricas del cono de matrices SPD 𝒮₊₊ⁿ.

    El cono SPD es un cono convexo abierto:
    - Si G₁, G₂ ∈ 𝒮₊₊ⁿ y α, β > 0, entonces αG₁ + βG₂ ∈ 𝒮₊₊ⁿ
    - Si G ∈ 𝒮₊₊ⁿ, entonces G⁻¹ ∈ 𝒮₊₊ⁿ
    - Si G ∈ 𝒮₊₊ⁿ y P invertible, entonces PGPᵀ ∈ 𝒮₊₊ⁿ (congruencia)
    """

    def test_positive_linear_combination_is_spd(self, tensor_info):
        """
        Combinación cónica: si G₁, G₂ ∈ 𝒮₊₊ⁿ y α, β > 0,
        entonces αG₁ + βG₂ ∈ 𝒮₊₊ⁿ.
        """
        name, G, n = tensor_info
        alpha, beta = 0.7, 1.3
        # Combinar con la identidad (siempre SPD)
        I = np.eye(n, dtype=_FLOAT_DTYPE)
        combination = alpha * G + beta * I
        eigenvalues = np.linalg.eigvalsh(combination)
        assert np.all(eigenvalues > 0), (
            f"{name}: combinación cónica no es SPD: λ = {eigenvalues}"
        )

    def test_pairwise_positive_combination_is_spd(self):
        """
        Combinar pares de tensores precompilados del mismo tamaño
        produce matrices SPD.
        """
        alpha, beta = 0.4, 0.6
        # G_TOPOLOGY y G_THERMODYNAMICS son ambos 2×2
        combination = alpha * G_TOPOLOGY + beta * G_THERMODYNAMICS
        eigenvalues = np.linalg.eigvalsh(combination)
        assert np.all(eigenvalues > 0)

    def test_congruence_preserves_spd(self, tensor_info):
        """
        Transformación de congruencia: si G ∈ 𝒮₊₊ⁿ y P invertible,
        entonces PGPᵀ ∈ 𝒮₊₊ⁿ.
        """
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)
        # Generar P invertible
        P = rng.standard_normal((n, n))
        while abs(np.linalg.det(P)) < 0.1:
            P = rng.standard_normal((n, n))

        congruent = P @ G @ P.T
        # Simetrizar para compensar errores numéricos
        congruent = 0.5 * (congruent + congruent.T)
        eigenvalues = np.linalg.eigvalsh(congruent)
        assert np.all(eigenvalues > 0), (
            f"{name}: congruencia PGPᵀ no es SPD: λ = {eigenvalues}"
        )

    def test_scalar_multiple_is_spd(self, tensor_info):
        """Si G ∈ 𝒮₊₊ⁿ y c > 0, entonces cG ∈ 𝒮₊₊ⁿ."""
        name, G, _ = tensor_info
        for c in [0.01, 0.5, 1.0, 2.0, 100.0]:
            scaled = c * G
            eigenvalues = np.linalg.eigvalsh(scaled)
            assert np.all(eigenvalues > 0), (
                f"{name}: {c}·G no es SPD: λ = {eigenvalues}"
            )

    def test_sum_of_spd_is_spd(self):
        """La suma de dos matrices SPD es SPD."""
        total = G_TOPOLOGY + G_THERMODYNAMICS
        eigenvalues = np.linalg.eigvalsh(total)
        assert np.all(eigenvalues > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 19: DISTANCIA DE MAHALANOBIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMahalanobisDistance:
    """
    Verifica propiedades de la forma cuadrática de Mahalanobis d_G(x)² = xᵀGx
    que define la pseudo-distancia inducida por los tensores métricos.
    """

    def test_distance_is_non_negative_truncated(self):
        """Si la distancia de Mahalanobis cae por debajo de cero debido a
        los límites IEEE 754, el tensor debe truncarlo algebraicamente a cero.
        Probamos esto instanciando un MetricTensor y forzando una forma
        cuadrática numéricamente inestable o pasándole un vector específico,
        pero dado que la lógica de truncamiento está en topological_watcher.py,
        lo probamos importando de allí o simulando la operación."""
        from app.core.immune_system.topological_watcher import MetricTensor
        # Creamos una matriz SPD
        G = np.array([[1e-10, 0], [0, 1e-10]], dtype=_FLOAT_DTYPE)
        tensor = MetricTensor(G, validate=False)

        # En MetricTensor.quadratic_form, ya se clampea a 0.0
        # Simulemos un resultado ligeramente negativo que produciría np.dot
        # Forzamos _matrix a tener un valor negativo minúsculo (esto no debería
        # pasar para SPD pero sirve para probar el clamping de la función)
        tensor._matrix = np.array([[-1e-16, 0], [0, -1e-16]], dtype=_FLOAT_DTYPE)
        v = np.array([1.0, 1.0], dtype=_FLOAT_DTYPE)

        # La forma cuadrática normal daría negativo
        raw_result = float(v @ tensor._matrix @ v)
        assert raw_result < 0.0

        # Pero quadratic_form debe clampearlo a 0.0
        clamped_result = tensor.quadratic_form(v)
        assert clamped_result == 0.0

    def test_distance_is_non_negative(self, tensor_info):
        """xᵀGx ≥ 0 para todo x (definida positiva implica ≥ 0)."""
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)
        for _ in range(200):
            x = rng.standard_normal(n)
            assert x @ G @ x >= -_FLOAT64_ATOL, (
                f"{name}: xᵀGx < 0 para algún x"
            )

    def test_distance_zero_only_at_origin(self, tensor_info):
        """xᵀGx = 0 ⟺ x = 0 (definida positiva)."""
        name, G, n = tensor_info
        # x = 0 → xᵀGx = 0
        zero = np.zeros(n, dtype=_FLOAT_DTYPE)
        assert zero @ G @ zero == 0.0

        # x ≠ 0 → xᵀGx > 0
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            x = rng.standard_normal(n)
            if np.linalg.norm(x) > 1e-15:
                assert x @ G @ x > 0, (
                    f"{name}: xᵀGx = 0 para x ≠ 0"
                )

    def test_distance_scales_quadratically(self, tensor_info):
        """(cx)ᵀG(cx) = c² · xᵀGx (homogeneidad cuadrática)."""
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)
        x = rng.standard_normal(n)
        base_distance = x @ G @ x

        for c in [0.5, 2.0, 3.0, 10.0]:
            scaled_distance = (c * x) @ G @ (c * x)
            expected = c ** 2 * base_distance
            np.testing.assert_allclose(
                scaled_distance, expected,
                rtol=_FLOAT64_RTOL,
                err_msg=f"{name}: homogeneidad cuadrática falla para c={c}"
            )

    def test_triangle_inequality_for_sqrt_distance(self, tensor_info):
        """
        La raíz cuadrada de la forma cuadrática define una norma:
        ||x||_G = √(xᵀGx) satisface la desigualdad triangular
        ||x + y||_G ≤ ||x||_G + ||y||_G.
        """
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)

        for _ in range(100):
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)

            norm_x = np.sqrt(x @ G @ x)
            norm_y = np.sqrt(y @ G @ y)
            norm_sum = np.sqrt((x + y) @ G @ (x + y))

            assert norm_sum <= norm_x + norm_y + _FLOAT64_ATOL, (
                f"{name}: desigualdad triangular violada"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 20: CASOS LÍMITE Y PATOLOGÍAS NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """
    Verifica comportamiento en casos límite y entradas patológicas.
    """

    def test_very_large_entries(self, factory):
        """Matrices con entradas muy grandes (pero finitas) deben ser manejadas."""
        G = np.eye(2, dtype=_FLOAT_DTYPE) * 1e15
        result = factory._validate_and_regularize("test", G)
        assert np.all(np.isfinite(result))
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues > 0)

    def test_very_small_positive_entries(self, factory):
        """
        Matrices con entradas positivas muy pequeñas pueden necesitar
        regularización.
        """
        tiny = MIN_EIGVAL_TOL * 0.01
        G = np.array([[tiny, 0], [0, tiny]], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        eigenvalues = np.linalg.eigvalsh(result)
        assert eigenvalues[0] >= MIN_EIGVAL_TOL

    def test_1x1_matrix(self, factory):
        """Una matriz 1×1 positiva es SPD trivialmente."""
        G = np.array([[5.0]], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        assert result.shape == (1, 1)
        assert result[0, 0] >= MIN_EIGVAL_TOL

    def test_1x1_matrix_needs_regularization(self, factory):
        """Una matriz 1×1 con valor pequeño se regulariza."""
        tiny = MIN_EIGVAL_TOL * 0.1
        G = np.array([[tiny]], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        assert result[0, 0] >= MIN_EIGVAL_TOL

    def test_deformation_threshold_boundary(self, factory):
        """
        Verifica que la deformación en el límite del umbral de aborto
        es rechazada correctamente.
        """
        # Construir una matriz donde la regularización necesaria
        # causa deformación > 25%
        # ||δI||_F / ||G||_F > 0.25 cuando δ√n / ||G||_F > 0.25
        # Para n=2: δ√2 / ||G||_F > 0.25
        # Si ||G||_F ≈ 1 y δ ≈ 0.5: 0.5√2 / 1 ≈ 0.71 > 0.25 → aborta
        G = np.array([[0.5, 0], [0, -0.3]], dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError):
            factory._validate_and_regularize("test", G)

    def test_mixed_sign_eigenvalues(self, factory):
        """
        Matriz con eigenvalores de signos mixtos requiere gran
        regularización → probablemente excede umbral de deformación.
        """
        G = np.array([[1.0, 0.0], [0.0, -0.5]], dtype=_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError):
            factory._validate_and_regularize("test", G)

    def test_nearly_zero_off_diagonal(self, factory):
        """Matriz diagonal con off-diagonal ≈ 0 se maneja correctamente."""
        G = np.array([
            [2.0, 1e-16],
            [1e-16, 3.0],
        ], dtype=_FLOAT_DTYPE)
        result = factory._validate_and_regularize("test", G)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues > 0)

    def test_large_condition_number_rejected(self, factory):
        """
        Una matriz con κ₂ > COND_NUM_TOL es rechazada en el pipeline.
        """
        # Construir matriz con κ₂ grande
        lambda_min = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR
        lambda_max = lambda_min * (COND_NUM_TOL + 1.0)
        G = np.diag([lambda_min, lambda_max]).astype(_FLOAT_DTYPE)
        with pytest.raises(MetricTensorError, match="mal condicionado"):
            factory._validate_and_regularize("test", G)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 21: PROPIEDADES DEL ESPACIO PRODUCTO
# ═══════════════════════════════════════════════════════════════════════════════

class TestProductSpaceProperties:
    """
    Verifica que los tensores son consistentes como componentes del
    espacio producto V_phys ⊕ V_topo ⊕ V_thermo ⊂ ℝ⁷.
    """

    def test_total_dimension_is_seven(self):
        """
        La suma de dimensiones de los subespacios es 7:
        dim(V_phys) + dim(V_topo) + dim(V_thermo) = 3 + 2 + 2 = 7.
        """
        total_dim = G_PHYSICS.shape[0] + G_TOPOLOGY.shape[0] + G_THERMODYNAMICS.shape[0]
        assert total_dim == 7

    def test_block_diagonal_metric_is_spd(self):
        """
        El tensor métrico bloque-diagonal del espacio producto
        G = diag(G_phys, G_topo, G_thermo) ∈ ℝ⁷ˣ⁷ es SPD.
        """
        from scipy.linalg import block_diag
        G_total = block_diag(
            np.array(G_PHYSICS),
            np.array(G_TOPOLOGY),
            np.array(G_THERMODYNAMICS),
        )
        assert G_total.shape == (7, 7)

        # Verificar SPD
        eigenvalues = np.linalg.eigvalsh(G_total)
        assert np.all(eigenvalues > 0), (
            f"Tensor bloque-diagonal no es SPD: λ = {eigenvalues}"
        )

        # Verificar simetría
        np.testing.assert_array_equal(G_total, G_total.T)

    def test_block_diagonal_eigenvalues_are_union(self):
        """
        Los eigenvalores de diag(A, B, C) son la unión de los eigenvalores
        de A, B y C.
        """
        from scipy.linalg import block_diag
        G_total = block_diag(
            np.array(G_PHYSICS),
            np.array(G_TOPOLOGY),
            np.array(G_THERMODYNAMICS),
        )

        eigs_total = np.sort(np.linalg.eigvalsh(G_total))
        eigs_parts = np.sort(np.concatenate([
            np.linalg.eigvalsh(G_PHYSICS),
            np.linalg.eigvalsh(G_TOPOLOGY),
            np.linalg.eigvalsh(G_THERMODYNAMICS),
        ]))

        np.testing.assert_allclose(
            eigs_total, eigs_parts,
            rtol=_FLOAT64_RTOL,
            err_msg="Eigenvalores del bloque ≠ unión de eigenvalores por bloque"
        )

    def test_block_diagonal_condition_number(self):
        """
        κ₂(diag(A, B, C)) = max(λ_max(A,B,C)) / min(λ_min(A,B,C)).
        """
        all_eigenvalues = np.concatenate([
            np.linalg.eigvalsh(G_PHYSICS),
            np.linalg.eigvalsh(G_TOPOLOGY),
            np.linalg.eigvalsh(G_THERMODYNAMICS),
        ])
        global_lambda_min = np.min(all_eigenvalues)
        global_lambda_max = np.max(all_eigenvalues)
        global_kappa = global_lambda_max / global_lambda_min

        assert np.isfinite(global_kappa)
        assert global_lambda_min >= MIN_EIGVAL_TOL
        # El condicionamiento global puede ser peor que cada bloque individual
        # pero debería ser finito y razonable


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 22: CONSTANTES DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleConstants:
    """
    Verifica que las constantes del módulo tienen valores coherentes
    y satisfacen las relaciones necesarias.
    """

    def test_symmetry_rtol_is_positive_and_small(self):
        """Tolerancia relativa de simetría debe ser positiva y pequeña."""
        assert _SYMMETRY_RTOL > 0
        assert _SYMMETRY_RTOL < 1e-10  # Mucho menor que errores típicos

    def test_symmetry_atol_base_is_positive_and_tiny(self):
        """Tolerancia absoluta base debe ser positiva y muy pequeña."""
        assert _SYMMETRY_ATOL_BASE > 0
        assert _SYMMETRY_ATOL_BASE < _SYMMETRY_RTOL

    def test_spd_interior_factor_greater_than_one(self):
        """Factor de interior SPD debe ser > 1 para estar en el interior."""
        assert _SPD_INTERIOR_FACTOR > 1.0

    def test_regularization_thresholds_ordered(self):
        """El umbral de advertencia debe ser menor que el de aborto."""
        assert 0 < _REGULARIZATION_WARN_THRESHOLD < _REGULARIZATION_ABORT_THRESHOLD < 1.0

    def test_tikhonov_delta_atol_positive(self):
        """Umbral de δ Tikhonov es positivo y diminuto."""
        assert _TIKHONOV_DELTA_ATOL > 0
        assert _TIKHONOV_DELTA_ATOL < MIN_EIGVAL_TOL

    def test_near_zero_frobenius_tol_positive(self):
        """Tolerancia de norma de Frobenius cero es positiva."""
        assert _NEAR_ZERO_FROBENIUS_TOL > 0
        assert _NEAR_ZERO_FROBENIUS_TOL < 1e-10

    def test_float_dtype_is_float64(self):
        """El dtype del módulo es float64."""
        assert _FLOAT_DTYPE == np.float64

    def test_min_eigval_tol_is_positive(self):
        """Tolerancia mínima de eigenvalor es positiva."""
        assert MIN_EIGVAL_TOL > 0

    def test_cond_num_tol_is_positive_and_finite(self):
        """Tolerancia de número de condición es positiva y finita."""
        assert COND_NUM_TOL > 0
        assert np.isfinite(COND_NUM_TOL)

    def test_expected_dimensions_consistency(self, factory):
        """Las dimensiones esperadas son coherentes con la suma total de 7."""
        dims = factory._EXPECTED_DIMENSIONS
        assert dims["G_phys"] == 3
        assert dims["G_topo"] == 2
        assert dims["G_thermo"] == 2
        assert sum(dims.values()) == 7


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 23: TESTS DE NO-REGRESIÓN PARA VALORES ESPECÍFICOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonRegression:
    """
    Tests de no-regresión que verifican valores numéricos específicos
    de los tensores precompilados. Si estos fallan, indica que las
    constantes base o la regularización cambiaron.
    """

    def test_g_physics_diagonal_values(self):
        """Valores diagonales de G_PHYSICS dentro de tolerancia de regularización."""
        np.testing.assert_allclose(G_PHYSICS[0, 0], 2.50, atol=0.05)
        np.testing.assert_allclose(G_PHYSICS[1, 1], 1.50, atol=0.05)
        np.testing.assert_allclose(G_PHYSICS[2, 2], 1.00, atol=0.05)

    def test_g_physics_off_diagonal_values(self):
        """Valores fuera de diagonal de G_PHYSICS."""
        np.testing.assert_allclose(G_PHYSICS[0, 1], 0.85, atol=0.05)
        np.testing.assert_allclose(G_PHYSICS[0, 2], 0.30, atol=0.05)
        np.testing.assert_allclose(G_PHYSICS[1, 2], 0.45, atol=0.05)

    def test_g_topology_values(self):
        """Valores de G_TOPOLOGY."""
        np.testing.assert_allclose(G_TOPOLOGY[0, 0], 1.00, atol=0.05)
        np.testing.assert_allclose(G_TOPOLOGY[1, 1], 3.00, atol=0.05)
        np.testing.assert_allclose(G_TOPOLOGY[0, 1], 0.60, atol=0.05)

    def test_g_thermodynamics_values(self):
        """Valores de G_THERMODYNAMICS."""
        np.testing.assert_allclose(G_THERMODYNAMICS[0, 0], 1.80, atol=0.05)
        np.testing.assert_allclose(G_THERMODYNAMICS[1, 1], 2.20, atol=0.05)
        np.testing.assert_allclose(G_THERMODYNAMICS[0, 1], 0.75, atol=0.05)

    def test_g_physics_has_three_positive_eigenvalues(self):
        """G_PHYSICS tiene exactamente 3 eigenvalores positivos."""
        eigenvalues = np.linalg.eigvalsh(G_PHYSICS)
        assert len(eigenvalues) == 3
        assert np.all(eigenvalues > 0)

    def test_g_topology_has_two_positive_eigenvalues(self):
        """G_TOPOLOGY tiene exactamente 2 eigenvalores positivos."""
        eigenvalues = np.linalg.eigvalsh(G_TOPOLOGY)
        assert len(eigenvalues) == 2
        assert np.all(eigenvalues > 0)

    def test_g_thermodynamics_has_two_positive_eigenvalues(self):
        """G_THERMODYNAMICS tiene exactamente 2 eigenvalores positivos."""
        eigenvalues = np.linalg.eigvalsh(G_THERMODYNAMICS)
        assert len(eigenvalues) == 2
        assert np.all(eigenvalues > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 24: TESTS DE ROBUSTEZ NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericalRobustness:
    """
    Verifica estabilidad numérica bajo perturbaciones controladas.
    """

    def test_small_perturbation_preserves_spd(self, tensor_info):
        """
        Perturbaciones simétricas pequeñas de un tensor SPD deben
        preservar la propiedad SPD (estabilidad del interior del cono).
        """
        name, G, n = tensor_info
        lambda_min = np.linalg.eigvalsh(G)[0]

        rng = np.random.default_rng(seed=42)
        perturbation = rng.standard_normal((n, n))
        perturbation = 0.5 * (perturbation + perturbation.T)
        # Escalar perturbación para que sea mucho menor que λ_min
        scale = lambda_min * 0.01 / np.linalg.norm(perturbation, 'fro')
        G_perturbed = np.array(G) + scale * perturbation  # copia writable

        eigenvalues = np.linalg.eigvalsh(G_perturbed)
        assert np.all(eigenvalues > 0), (
            f"{name}: pequeña perturbación destruyó SPD: λ = {eigenvalues}"
        )

    def test_eigenvalue_perturbation_bound(self, tensor_info):
        """
        Teorema de Weyl: |λᵢ(A + E) - λᵢ(A)| ≤ ||E||₂
        para matrices simétricas A, E.
        """
        name, G, n = tensor_info
        rng = np.random.default_rng(seed=42)
        E = rng.standard_normal((n, n))
        E = 0.5 * (E + E.T) * 0.001  # Perturbación simétrica pequeña

        eigs_G = np.linalg.eigvalsh(G)
        G_perturbed = np.array(G) + E
        eigs_perturbed = np.linalg.eigvalsh(G_perturbed)

        norm_E = np.linalg.norm(E, 2)  # Norma espectral
        max_shift = np.max(np.abs(eigs_perturbed - eigs_G))

        assert max_shift <= norm_E + _FLOAT64_ATOL, (
            f"{name}: violación de cota de Weyl: "
            f"max|Δλ| = {max_shift:.6e} > ||E||₂ = {norm_E:.6e}"
        )

    def test_cholesky_inverse_consistency(self, tensor_info):
        """
        Si G = LLᵀ, entonces G⁻¹ = L⁻ᵀL⁻¹.
        Verificar consistencia de inversión vía Cholesky.
        """
        name, G, _ = tensor_info
        L = np.linalg.cholesky(G)
        L_inv = np.linalg.inv(L)
        G_inv_cholesky = L_inv.T @ L_inv
        G_inv_direct = np.linalg.inv(G)

        np.testing.assert_allclose(
            G_inv_cholesky, G_inv_direct,
            rtol=_FLOAT64_RTOL * 10,  # Algo más holgado por acumulación
            atol=_FLOAT64_ATOL * 10,
            err_msg=f"{name}: inversión vía Cholesky inconsistente"
        )

    def test_frobenius_norm_submultiplicative(self, tensor_info):
        """
        ||AB||_F ≤ ||A||_F · ||B||_F (submultiplicatividad).
        Verificar para G · G.
        """
        name, G, _ = tensor_info
        G_sq = G @ G
        norm_G = np.linalg.norm(G, "fro")
        norm_G_sq = np.linalg.norm(G_sq, "fro")
        assert norm_G_sq <= norm_G ** 2 + _FLOAT64_ATOL


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 25: TESTS DE INTERFAZ PÚBLICA (__all__)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPublicInterface:
    """
    Verifica que la interfaz pública del módulo es correcta y completa.
    """

    def test_all_exports_exist(self):
        """Todos los nombres en __all__ deben existir en el módulo."""
        import app.core.immune_system.metric_tensors as module
        for name in module.__all__:
            assert hasattr(module, name), f"'{name}' en __all__ pero no existe"

    def test_expected_exports_present(self):
        """Los exports esperados están presentes."""
        import app.core.immune_system.metric_tensors as module
        expected = [
            "G_PHYSICS", "G_TOPOLOGY", "G_THERMODYNAMICS",
            "MetricTensorFactory", "SpectralProfile",
            "RegularizationReport", "get_tensor_diagnostics",
        ]
        for name in expected:
            assert name in module.__all__, f"'{name}' falta en __all__"

    def test_no_private_exports(self):
        """Ningún nombre privado (con _) está en __all__."""
        import app.core.immune_system.metric_tensors as module
        for name in module.__all__:
            assert not name.startswith("_"), (
                f"Nombre privado '{name}' exportado en __all__"
            )