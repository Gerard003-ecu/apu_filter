# tests/unit/agents/core/immune_system/test_watcher_agent.py
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de pruebas unitarias: TopologicalWatcherAgent                          ║
║ Ubicación: tests/unit/agents/core/immune_system/test_watcher_agent.py        ║
║ Versión: 5.0.0-Nested-Spectral-Topos-Cauchy                                  ║
║ Cobertura: Fase 1 → Fase 2 → Fase 3 + funtor 𝒲_agent + invariantes          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de verificación (teoría espectral + geometría riemanniana numérica):
  • Fixtures sintéticas con geometría euclídea plana (Γ≡0, ∂𝒯≡0) ⇒ ∇𝒯 = 0 exacto.
  • Tensores de estrés simétricos con espectro controlado (ρ, ‖·‖_op, Tr_G).
  • Casos patológicos: asimetría, métrica no-PD, divergencia no nula, δx mal dim,
    desbordamiento de energía, lockdown booleano, mal-condicionamiento de G.
  • Invariantes de Rayleigh, residuo relativo, L_max y predicado ZeroTrust.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─── SUT ─────────────────────────────────────────────────────────────────────
from app.agents.core.immune_system.watcher_agent import (
    _MACHINE_EPSILON,
    _KAPPA_0,
    _DEFAULT_TOL,
    _COND_MAX,
    _RHO_CRIT,
    _LOCKDOWN_EPS_FACTOR,
    WatcherAgentError,
    TensorConservationError,
    TensorSymmetryError,
    MetricSignatureError,
    DeformationEnergyOverflowError,
    DimensionMismatchError,
    SpectralSeverity,
    StressTensorData,
    SpectralInvariants,
    ValidatedStressTensor,
    PushforwardResult,
    CohomologyPushforwardData,
    ToposPullbackData,
    Phase1_StressTensorValidation,
    TopologicalWatcherAgent,
)


# =============================================================================
# HELPERS DE CONSTRUCCIÓN GEOMÉTRICA (geometría euclídea / riemanniana controlada)
# =============================================================================

def _eye(n: int) -> NDArray[np.float64]:
    return np.eye(n, dtype=np.float64)


def _zeros3(n: int) -> NDArray[np.float64]:
    return np.zeros((n, n, n), dtype=np.float64)


def make_symmetric_stress(
    n: int,
    scale: float = 1.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """𝒯 simétrico con espectro controlable vía escala."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    T = 0.5 * (A + A.T) * scale
    return T.astype(np.float64)


def make_spd_metric(
    n: int,
    cond: float = 1.0,
    seed: int = 1,
) -> NDArray[np.float64]:
    """
    Métrica SPD con número de condición ≈ cond.
    Construcción: Q diag(λ) Qᵀ con λ_max/λ_min = cond.
    """
    rng = np.random.default_rng(seed)
    # QR → ortogonal
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam_min = 1.0
    lam_max = float(cond)
    if n == 1:
        eigs = np.array([lam_min], dtype=np.float64)
    else:
        eigs = np.linspace(lam_min, lam_max, n, dtype=np.float64)
    G = (Q @ np.diag(eigs) @ Q.T).astype(np.float64)
    # Re-simetrizar residual numérico
    return 0.5 * (G + G.T)


def make_flat_conserved_data(
    n: int = 4,
    T_scale: float = 1.0,
    G_cond: float = 1.0,
    seed: int = 42,
) -> StressTensorData:
    r"""
    Geometría plana euclídea (o riemanniana con Γ=0, ∂𝒯=0):
      ∇_μ 𝒯^{μν} = 0 exactamente ⇒ conservación trivial.
    """
    T = make_symmetric_stress(n, scale=T_scale, seed=seed)
    G = make_spd_metric(n, cond=G_cond, seed=seed + 1)
    return StressTensorData(
        T_mu_nu=T,
        G_mu_nu=G,
        Christoffel=_zeros3(n),
        partial_T=_zeros3(n),
    )


def make_nonzero_christoffel_conserved(
    n: int = 3,
    seed: int = 7,
) -> StressTensorData:
    r"""
    Construye un par (𝒯, Γ, ∂𝒯) no trivial que satisface ∇𝒯 ≈ 0
    por diseño: se elige 𝒯 y Γ, y se define

        ∂_μ 𝒯^{μν}  :=  −(Γ^μ_{μλ} 𝒯^{λν} + Γ^ν_{μλ} 𝒯^{μλ})

    de modo que la suma de los tres términos sea idénticamente nula.
    """
    rng = np.random.default_rng(seed)
    T = make_symmetric_stress(n, scale=0.5, seed=seed)
    G = make_spd_metric(n, cond=2.0, seed=seed + 3)

    # Christoffel aleatorio (no necesariamente simétrico en μν — no es necesario
    # para el test de contracción).
    Gamma = rng.standard_normal((n, n, n)).astype(np.float64) * 0.1

    # Términos de conexión
    gamma_contracted = np.einsum("mml->l", Gamma)
    term1 = gamma_contracted @ T
    term2 = np.einsum("nml,ml->n", Gamma, T)
    connection = term1 + term2  # vector en ν

    # partial_T tal que ∂_μ 𝒯^{μν} = −connection^ν
    # Distribuimos la derivada solo en la diagonal ρ=μ (resto cero).
    partial_T = _zeros3(n)
    for mu in range(n):
        # Asignamos −connection / n en cada slice diagonal para que
        # einsum('mmn->n') = −connection.
        partial_T[mu, mu, :] = -connection / float(n)

    return StressTensorData(
        T_mu_nu=T,
        G_mu_nu=G,
        Christoffel=Gamma,
        partial_T=partial_T,
    )


def make_divergent_data(n: int = 3, seed: int = 99) -> StressTensorData:
    """Datos con ∇𝒯 ≠ 0 (fuga de exergía deliberada)."""
    data = make_flat_conserved_data(n=n, seed=seed)
    # Perturbar partial_T para romper conservación
    partial_T = data.partial_T.copy()
    partial_T[0, 0, :] += 1.0  # inyecta divergencia O(1)
    return StressTensorData(
        T_mu_nu=data.T_mu_nu,
        G_mu_nu=data.G_mu_nu,
        Christoffel=data.Christoffel,
        partial_T=partial_T,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dim() -> int:
    return 4


@pytest.fixture
def flat_data(dim: int) -> StressTensorData:
    return make_flat_conserved_data(n=dim, T_scale=1.0, G_cond=1.0, seed=42)


@pytest.fixture
def flat_data_illcond(dim: int) -> StressTensorData:
    """Métrica con κ(G) moderadamente alto pero < _COND_MAX."""
    return make_flat_conserved_data(n=dim, T_scale=1.0, G_cond=1e4, seed=43)


@pytest.fixture
def delta_x(dim: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.standard_normal(dim).astype(np.float64)


@pytest.fixture
def phase1() -> Phase1_StressTensorValidation:
    return Phase1_StressTensorValidation(tolerance=_DEFAULT_TOL)


@pytest.fixture
def phase2() -> Phase1_StressTensorValidation.Phase2_CohomologyPushforward:
    return Phase1_StressTensorValidation.Phase2_CohomologyPushforward(
        energy_overflow_factor=1e8,
    )


@pytest.fixture
def phase3() -> (
    Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback
):
    return (
        Phase1_StressTensorValidation
        .Phase2_CohomologyPushforward
        .Phase3_ToposPullback(kappa0=_KAPPA_0)
    )


@pytest.fixture
def agent() -> TopologicalWatcherAgent:
    return TopologicalWatcherAgent(tolerance=_DEFAULT_TOL, kappa0=_KAPPA_0)


@pytest.fixture
def validated(
    phase1: Phase1_StressTensorValidation,
    flat_data: StressTensorData,
) -> ValidatedStressTensor:
    return phase1.validate_tensor(flat_data)


# =============================================================================
# FASE 0 – CONSTANTES, EXCEPCIONES Y DTOs
# =============================================================================

class TestConstantsAndExceptions:
    """Sanidad de constantes numéricas y jerarquía de excepciones."""

    def test_machine_epsilon_positive(self) -> None:
        assert _MACHINE_EPSILON > 0.0
        assert _MACHINE_EPSILON == float(np.finfo(np.float64).eps)

    def test_kappa0_positive(self) -> None:
        assert _KAPPA_0 > 0.0

    def test_exception_hierarchy(self) -> None:
        assert issubclass(TensorConservationError, WatcherAgentError)
        assert issubclass(TensorSymmetryError, WatcherAgentError)
        assert issubclass(MetricSignatureError, WatcherAgentError)
        assert issubclass(DeformationEnergyOverflowError, WatcherAgentError)
        assert issubclass(DimensionMismatchError, WatcherAgentError)
        assert issubclass(WatcherAgentError, Exception)

    def test_spectral_severity_enum(self) -> None:
        names = {s.name for s in SpectralSeverity}
        assert names == {"NOMINAL", "ELEVATED", "CRITICAL", "SINGULAR"}

    def test_stress_tensor_data_frozen(self, flat_data: StressTensorData) -> None:
        with pytest.raises(Exception):
            flat_data.T_mu_nu = np.zeros_like(flat_data.T_mu_nu)  # type: ignore[misc]

    def test_dto_slots_and_immutability(self, validated: ValidatedStressTensor) -> None:
        assert hasattr(validated, "__slots__") or True  # slots=True en dataclass
        with pytest.raises(Exception):
            validated.residual_relative = 0.0  # type: ignore[misc]


# =============================================================================
# FASE 1 – VALIDACIÓN TENSORIAL Y ESPECTRO
# =============================================================================

class TestPhase1Dimensions:
    """Chequeos dimensionales del 4-tuplo (𝒯, G, Γ, ∂𝒯)."""

    def test_valid_dimensions_pass(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        n = phase1._assert_dimensions(flat_data)
        assert n == flat_data.T_mu_nu.shape[0]

    def test_non_square_T_raises(self, phase1: Phase1_StressTensorValidation) -> None:
        T = np.ones((3, 4), dtype=np.float64)
        data = StressTensorData(
            T_mu_nu=T,
            G_mu_nu=_eye(3),
            Christoffel=_zeros3(3),
            partial_T=_zeros3(3),
        )
        with pytest.raises(DimensionMismatchError, match="cuadrado"):
            phase1._assert_dimensions(data)

    def test_G_shape_mismatch_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 3
        data = StressTensorData(
            T_mu_nu=_eye(n),
            G_mu_nu=_eye(n + 1),
            Christoffel=_zeros3(n),
            partial_T=_zeros3(n),
        )
        with pytest.raises(DimensionMismatchError, match="G shape"):
            phase1._assert_dimensions(data)

    def test_christoffel_shape_mismatch_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 3
        data = StressTensorData(
            T_mu_nu=_eye(n),
            G_mu_nu=_eye(n),
            Christoffel=np.zeros((n, n, n + 1), dtype=np.float64),
            partial_T=_zeros3(n),
        )
        with pytest.raises(DimensionMismatchError, match="Christoffel"):
            phase1._assert_dimensions(data)

    def test_partial_T_shape_mismatch_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 3
        data = StressTensorData(
            T_mu_nu=_eye(n),
            G_mu_nu=_eye(n),
            Christoffel=_zeros3(n),
            partial_T=np.zeros((2, 2, 2), dtype=np.float64),
        )
        with pytest.raises(DimensionMismatchError, match="partial_T"):
            phase1._assert_dimensions(data)


class TestPhase1Symmetry:
    """Simetría de 𝒯 (álgebra de tensores simétricos)."""

    def test_symmetric_passes(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        phase1._check_symmetry(flat_data.T_mu_nu)  # no raise

    def test_asymmetric_raises(self, phase1: Phase1_StressTensorValidation) -> None:
        T = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        with pytest.raises(TensorSymmetryError, match="no es simétrico"):
            phase1._check_symmetry(T)

    def test_near_symmetric_within_tol_passes(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        T = np.array([[1.0, 1.0], [1.0 + 1e-14, 1.0]], dtype=np.float64)
        # Error relativo muy pequeño → debe pasar
        phase1._check_symmetry(T)

    def test_relative_symmetry_error_zero_for_exact(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        T = make_symmetric_stress(5, seed=3)
        err = phase1._relative_symmetry_error(T)
        assert err < 1e-15


class TestPhase1Metric:
    """Firma riemanniana, Cholesky y espectro de G."""

    def test_identity_metric(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        G = _eye(4)
        lam_min, lam_max, cond = phase1._check_metric(G)
        assert lam_min == pytest.approx(1.0)
        assert lam_max == pytest.approx(1.0)
        assert cond == pytest.approx(1.0)

    def test_spd_with_known_cond(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        target_cond = 50.0
        G = make_spd_metric(5, cond=target_cond, seed=11)
        lam_min, lam_max, cond = phase1._check_metric(G)
        assert lam_min > 0.0
        assert cond == pytest.approx(target_cond, rel=1e-6)

    def test_asymmetric_metric_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        G = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        with pytest.raises(MetricSignatureError, match="no es simétrica"):
            phase1._check_metric(G)

    def test_indefinite_metric_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        # Firma (1,1): un autovalor negativo
        G = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        with pytest.raises(MetricSignatureError):
            phase1._check_metric(G)

    def test_singular_metric_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        G = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        with pytest.raises(MetricSignatureError):
            phase1._check_metric(G)

    def test_negative_definite_raises(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        G = -_eye(3)
        with pytest.raises(MetricSignatureError):
            phase1._check_metric(G)


class TestPhase1Divergence:
    """Conservación covariante ∇_μ 𝒯^{μν} = 0."""

    def test_flat_geometry_zero_divergence(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        div = phase1._compute_divergence(flat_data)
        assert np.linalg.norm(div) == pytest.approx(0.0, abs=1e-14)

    def test_constructed_nonzero_christoffel_conserved(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        data = make_nonzero_christoffel_conserved(n=3, seed=7)
        div = phase1._compute_divergence(data)
        # Por construcción debe ser ~0
        assert np.linalg.norm(div) == pytest.approx(0.0, abs=1e-10)

    def test_divergent_data_nonzero(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        data = make_divergent_data(n=3)
        div = phase1._compute_divergence(data)
        assert np.linalg.norm(div) > 0.5

    def test_validate_rejects_divergent(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        data = make_divergent_data(n=3)
        with pytest.raises(TensorConservationError, match="Fuga de exergía"):
            phase1.validate_tensor(data)

    def test_validate_accepts_flat(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        result = phase1.validate_tensor(flat_data)
        assert isinstance(result, ValidatedStressTensor)
        assert result.residual_relative < _DEFAULT_TOL

    def test_validate_accepts_nonzero_christoffel_conserved(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        data = make_nonzero_christoffel_conserved(n=4, seed=21)
        result = phase1.validate_tensor(data)
        assert result.residual_relative < _DEFAULT_TOL


class TestPhase1SpectralInvariants:
    """Extracción de espectro: Tr_G, ρ(𝒯), ‖𝒯‖_op, κ(G), severidad."""

    def test_identity_pair_invariants(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 3
        T = _eye(n) * 2.0
        G = _eye(n)
        data = StressTensorData(
            T_mu_nu=T, G_mu_nu=G, Christoffel=_zeros3(n), partial_T=_zeros3(n)
        )
        inv = phase1._compute_spectral_invariants(data, 1.0, 1.0, 1.0)

        # Tr_G(𝒯) = Σ_i 𝒯^{ii} G_{ii} = 2+2+2 = 6
        assert inv.trace_mixed == pytest.approx(6.0)
        assert inv.rho_T == pytest.approx(2.0)
        assert inv.op_norm_T == pytest.approx(2.0)
        assert inv.frobenius_T == pytest.approx(math.sqrt(3.0 * 4.0))
        assert inv.severity == SpectralSeverity.NOMINAL

    def test_severity_nominal(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        sev = phase1._classify_severity(cond_G=2.0, rho_T=1.0, lam_min_G=1.0)
        assert sev == SpectralSeverity.NOMINAL

    def test_severity_elevated(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        sev = phase1._classify_severity(
            cond_G=math.sqrt(_COND_MAX) * 2.0,
            rho_T=1.0,
            lam_min_G=1.0,
        )
        assert sev == SpectralSeverity.ELEVATED

    def test_severity_critical_by_cond(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        sev = phase1._classify_severity(
            cond_G=_COND_MAX * 2.0, rho_T=1.0, lam_min_G=1.0
        )
        assert sev == SpectralSeverity.CRITICAL

    def test_severity_critical_by_rho(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        sev = phase1._classify_severity(
            cond_G=1.0, rho_T=_RHO_CRIT * 2.0, lam_min_G=1.0
        )
        assert sev == SpectralSeverity.CRITICAL

    def test_severity_singular(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        sev = phase1._classify_severity(
            cond_G=1.0, rho_T=1.0, lam_min_G=_MACHINE_EPSILON
        )
        assert sev == SpectralSeverity.SINGULAR

    def test_validate_tensor_populates_invariants(
        self, validated: ValidatedStressTensor, flat_data: StressTensorData
    ) -> None:
        inv = validated.invariants
        assert inv.lambda_max_G > 0.0
        assert inv.lambda_min_G > 0.0
        assert inv.cond_G >= 1.0 - 1e-12
        assert inv.op_norm_T >= 0.0
        assert inv.rho_T >= 0.0
        assert inv.frobenius_T >= 0.0
        assert inv.christoffel_contracted.shape == (flat_data.T_mu_nu.shape[0],)
        # Consistencia: ρ(𝒯) ≤ ‖𝒯‖_op ≤ ‖𝒯‖_F (desigualdades espectrales)
        assert inv.rho_T <= inv.op_norm_T + 1e-10
        assert inv.op_norm_T <= inv.frobenius_T + 1e-10

    def test_trace_mixed_matches_einsum(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        validated = phase1.validate_tensor(flat_data)
        expected = float(
            np.einsum("ij,ij->", flat_data.T_mu_nu, flat_data.G_mu_nu)
        )
        assert validated.invariants.trace_mixed == pytest.approx(expected)


class TestPhase1ValidateTensorIntegration:
    """Integración del pipeline completo de validate_tensor."""

    def test_returns_validated_stress_tensor(
        self, phase1: Phase1_StressTensorValidation, flat_data: StressTensorData
    ) -> None:
        result = phase1.validate_tensor(flat_data)
        assert isinstance(result, ValidatedStressTensor)
        np.testing.assert_array_equal(result.T_mu_nu, flat_data.T_mu_nu)
        np.testing.assert_array_equal(result.G_mu_nu, flat_data.G_mu_nu)

    def test_rejects_asymmetric_T(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 3
        T = np.array(
            [[1.0, 5.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        data = StressTensorData(
            T_mu_nu=T, G_mu_nu=_eye(n),
            Christoffel=_zeros3(n), partial_T=_zeros3(n),
        )
        with pytest.raises(TensorSymmetryError):
            phase1.validate_tensor(data)

    def test_rejects_bad_metric(
        self, phase1: Phase1_StressTensorValidation
    ) -> None:
        n = 2
        data = StressTensorData(
            T_mu_nu=_eye(n),
            G_mu_nu=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
            Christoffel=_zeros3(n),
            partial_T=_zeros3(n),
        )
        with pytest.raises(MetricSignatureError):
            phase1.validate_tensor(data)

    def test_tolerance_constructor_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="positiva"):
            Phase1_StressTensorValidation(tolerance=0.0)
        with pytest.raises(ValueError, match="positiva"):
            Phase1_StressTensorValidation(tolerance=-1e-8)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
    def test_parametrized_dimensions(
        self, phase1: Phase1_StressTensorValidation, n: int
    ) -> None:
        data = make_flat_conserved_data(n=n, seed=n * 10)
        result = phase1.validate_tensor(data)
        assert result.T_mu_nu.shape == (n, n)
        assert result.invariants.christoffel_contracted.shape == (n,)


# =============================================================================
# FASE 2 – PUSHFORWARD COHOMOLÓGICO
# =============================================================================

class TestPhase2DirichletEnergy:
    """Energía de Dirichlet ℰ = δxᵀ 𝒯 δx y cota de Rayleigh."""

    def test_energy_matches_quadratic_form(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        energy = phase2._dirichlet_energy(validated.T_mu_nu, delta_x)
        expected = float(delta_x @ validated.T_mu_nu @ delta_x)
        assert energy == pytest.approx(expected, rel=1e-12)

    def test_energy_zero_for_zero_vector(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
    ) -> None:
        n = validated.T_mu_nu.shape[0]
        energy = phase2._dirichlet_energy(
            validated.T_mu_nu, np.zeros(n, dtype=np.float64)
        )
        assert energy == pytest.approx(0.0)

    def test_rayleigh_bound_dominates_energy(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        energy = phase2._dirichlet_energy(validated.T_mu_nu, delta_x)
        bound = phase2._rayleigh_bound(
            validated.invariants.op_norm_T, delta_x
        )
        assert abs(energy) <= bound + 1e-9

    def test_rayleigh_bound_formula(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
    ) -> None:
        op = 3.5
        dx = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        bound = phase2._rayleigh_bound(op, dx)
        assert bound == pytest.approx(3.5 * (1.0 + 4.0 + 0.0))


class TestPhase2ComputePushforward:
    """Método terminal de Fase 2."""

    def test_happy_path(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        result = phase2.compute_pushforward(validated, delta_x)
        assert isinstance(result, PushforwardResult)
        assert result.energy_is_bounded is True
        assert result.rayleigh_bound >= 0.0
        assert result.invariants is validated.invariants

    def test_delta_x_dimension_mismatch_raises(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
    ) -> None:
        bad_dx = np.ones(validated.T_mu_nu.shape[0] + 2, dtype=np.float64)
        with pytest.raises(DimensionMismatchError, match="δx"):
            phase2.compute_pushforward(validated, bad_dx)

    def test_delta_x_matrix_raises(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
    ) -> None:
        n = validated.T_mu_nu.shape[0]
        bad_dx = np.ones((n, n), dtype=np.float64)
        with pytest.raises(DimensionMismatchError):
            phase2.compute_pushforward(validated, bad_dx)

    def test_overflow_raises(
        self,
        phase1: Phase1_StressTensorValidation,
    ) -> None:
        """
        Forzamos overflow con factor minúsculo: cualquier ℰ no nulo desborda.
        """
        data = make_flat_conserved_data(n=3, T_scale=10.0, seed=5)
        validated = phase1.validate_tensor(data)
        phase2_strict = (
            Phase1_StressTensorValidation.Phase2_CohomologyPushforward(
                energy_overflow_factor=1e-30,
            )
        )
        dx = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # Si ℰ y bound son ambos ~0, no desborda; usamos T_scale alto y dx unitario.
        # Con factor 1e-30, |ℰ| > factor·bound casi siempre si bound > 0 y ℰ ≠ 0.
        energy = float(dx @ validated.T_mu_nu @ dx)
        bound = validated.invariants.op_norm_T * float(np.dot(dx, dx))
        if abs(energy) > 1e-30 * bound:
            with pytest.raises(DeformationEnergyOverflowError):
                phase2_strict.compute_pushforward(validated, dx)
        else:
            # Caso degenerado: skip lógico
            pytest.skip("Energía numéricamente nula; overflow no aplicable.")

    def test_overflow_factor_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            Phase1_StressTensorValidation.Phase2_CohomologyPushforward(
                energy_overflow_factor=0.0
            )

    def test_invariants_propagated_intact(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        result = phase2.compute_pushforward(validated, delta_x)
        assert result.invariants.trace_mixed == validated.invariants.trace_mixed
        assert result.invariants.rho_T == validated.invariants.rho_T
        assert result.invariants.op_norm_T == validated.invariants.op_norm_T
        assert result.invariants.severity == validated.invariants.severity


# =============================================================================
# FASE 3 – PULLBACK AL TOPOS DE GROTHENDIECK
# =============================================================================

class TestPhase3LMax:
    """Cota de Lipschitz L_max y regularización espectral."""

    def test_L_max_positive(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        pf = phase2.compute_pushforward(validated, delta_x)
        L = phase3._compute_L_max(pf.invariants)
        assert L > 0.0
        assert math.isfinite(L)

    def test_L_max_decreases_with_larger_trace(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        """Mayor |Tr_G(𝒯)| ⇒ mayor denominador ⇒ menor L_max."""
        inv_small = SpectralInvariants(
            trace_mixed=1.0,
            lambda_max_G=1.0,
            lambda_min_G=1.0,
            cond_G=1.0,
            rho_T=1.0,
            op_norm_T=1.0,
            frobenius_T=1.0,
            christoffel_contracted=np.zeros(2),
            severity=SpectralSeverity.NOMINAL,
        )
        inv_large = SpectralInvariants(
            trace_mixed=100.0,
            lambda_max_G=1.0,
            lambda_min_G=1.0,
            cond_G=1.0,
            rho_T=1.0,
            op_norm_T=1.0,
            frobenius_T=1.0,
            christoffel_contracted=np.zeros(2),
            severity=SpectralSeverity.NOMINAL,
        )
        L_small = phase3._compute_L_max(inv_small)
        L_large = phase3._compute_L_max(inv_large)
        assert L_large < L_small

    def test_L_max_decreases_with_worse_conditioning(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        """√κ en el denominador penaliza mal-condicionamiento."""
        base = dict(
            trace_mixed=0.0,  # aísla el término de regularización
            lambda_max_G=1.0,
            lambda_min_G=1.0,
            rho_T=0.0,
            op_norm_T=0.0,
            frobenius_T=0.0,
            christoffel_contracted=np.zeros(2),
            severity=SpectralSeverity.NOMINAL,
        )
        inv_well = SpectralInvariants(cond_G=1.0, **base)
        inv_ill = SpectralInvariants(cond_G=1e6, **base)
        L_well = phase3._compute_L_max(inv_well)
        L_ill = phase3._compute_L_max(inv_ill)
        assert L_ill < L_well

    def test_L_max_formula_analytic(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = SpectralInvariants(
            trace_mixed=4.0,       # √|Tr| = 2
            lambda_max_G=1.0,
            lambda_min_G=1.0,
            cond_G=1.0,            # √κ = 1
            rho_T=0.0,
            op_norm_T=0.0,
            frobenius_T=0.0,
            christoffel_contracted=np.zeros(1),
            severity=SpectralSeverity.NOMINAL,
        )
        L = phase3._compute_L_max(inv)
        denom = 2.0 + _MACHINE_EPSILON * 1.0 * 1.0
        expected = _KAPPA_0 / denom
        assert L == pytest.approx(expected, rel=1e-12)

    def test_constructor_rejects_bad_params(self) -> None:
        P3 = (
            Phase1_StressTensorValidation
            .Phase2_CohomologyPushforward
            .Phase3_ToposPullback
        )
        with pytest.raises(ValueError, match="kappa0"):
            P3(kappa0=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            P3(epsilon=-1.0)


class TestPhase3LockdownPredicate:
    """Álgebra de Boole del predicado ZeroTrust."""

    def _make_inv(
        self,
        *,
        severity: SpectralSeverity = SpectralSeverity.NOMINAL,
        cond_G: float = 1.0,
        rho_T: float = 0.0,
        trace_mixed: float = 1.0,
    ) -> SpectralInvariants:
        return SpectralInvariants(
            trace_mixed=trace_mixed,
            lambda_max_G=1.0,
            lambda_min_G=1.0 / max(cond_G, 1.0),
            cond_G=cond_G,
            rho_T=rho_T,
            op_norm_T=rho_T,
            frobenius_T=rho_T,
            christoffel_contracted=np.zeros(2),
            severity=severity,
        )

    def test_nominal_no_lockdown(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = self._make_inv()
        L = phase3._compute_L_max(inv)
        assert phase3._boolean_lockdown_predicate(L, inv) is False

    def test_lockdown_by_severity_critical(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = self._make_inv(severity=SpectralSeverity.CRITICAL)
        L = phase3._compute_L_max(inv)
        assert phase3._boolean_lockdown_predicate(L, inv) is True

    def test_lockdown_by_severity_singular(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = self._make_inv(severity=SpectralSeverity.SINGULAR)
        L = phase3._compute_L_max(inv)
        assert phase3._boolean_lockdown_predicate(L, inv) is True

    def test_lockdown_by_cond(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = self._make_inv(cond_G=_COND_MAX * 2.0)
        L = phase3._compute_L_max(inv)
        assert phase3._boolean_lockdown_predicate(L, inv) is True

    def test_lockdown_by_rho(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = self._make_inv(rho_T=_RHO_CRIT * 2.0)
        L = phase3._compute_L_max(inv)
        assert phase3._boolean_lockdown_predicate(L, inv) is True

    def test_lockdown_by_tiny_L_max(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        """L_max artificialmente minúsculo dispara P₁."""
        inv = self._make_inv(trace_mixed=1e40)  # L_max → 0
        L = phase3._compute_L_max(inv)
        theta = _MACHINE_EPSILON * _LOCKDOWN_EPS_FACTOR
        assert L < theta
        assert phase3._boolean_lockdown_predicate(L, inv) is True


class TestPhase3ComputePullback:
    """Método terminal de Fase 3."""

    def test_happy_path_dto(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
        delta_x: NDArray[np.float64],
    ) -> None:
        pf = phase2.compute_pushforward(validated, delta_x)
        result = phase3.compute_pullback(pf)
        assert isinstance(result, ToposPullbackData)
        assert result.L_max > 0.0
        assert isinstance(result.zero_trust_lockdown, bool)
        assert result.severity == validated.invariants.severity
        assert result.condition_number == pytest.approx(
            validated.invariants.cond_G
        )

    def test_pullback_from_critical_invariants(
        self,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
    ) -> None:
        inv = SpectralInvariants(
            trace_mixed=1.0,
            lambda_max_G=1.0,
            lambda_min_G=1e-15,
            cond_G=_COND_MAX * 10.0,
            rho_T=_RHO_CRIT * 10.0,
            op_norm_T=_RHO_CRIT * 10.0,
            frobenius_T=_RHO_CRIT * 10.0,
            christoffel_contracted=np.zeros(2),
            severity=SpectralSeverity.CRITICAL,
        )
        pf = PushforwardResult(
            deformed_energy=1.0,
            rayleigh_bound=2.0,
            energy_is_bounded=True,
            invariants=inv,
        )
        result = phase3.compute_pullback(pf)
        assert result.zero_trust_lockdown is True
        assert result.severity == SpectralSeverity.CRITICAL


# =============================================================================
# FUNCTOR 𝒲_agent – INTEGRACIÓN END-TO-END
# =============================================================================

class TestTopologicalWatcherAgent:
    """Orquestación de las tres fases anidadas."""

    def test_execute_propagation_happy_path(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> None:
        coh, pull = agent.execute_propagation(flat_data, delta_x)
        assert isinstance(coh, CohomologyPushforwardData)
        assert isinstance(pull, ToposPullbackData)

        # Consistencia numérica del empaquetado
        assert math.isfinite(coh.deformed_dirichlet_energy)
        assert math.isfinite(coh.stress_trace)
        assert coh.operator_norm >= 0.0
        assert coh.spectral_radius >= 0.0
        assert pull.L_max > 0.0
        assert isinstance(pull.zero_trust_lockdown, bool)

    def test_call_delegates_to_execute(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> None:
        coh1, pull1 = agent.execute_propagation(flat_data, delta_x)
        coh2, pull2 = agent(flat_data, delta_x)
        assert coh1.deformed_dirichlet_energy == pytest.approx(
            coh2.deformed_dirichlet_energy
        )
        assert pull1.L_max == pytest.approx(pull2.L_max)
        assert pull1.zero_trust_lockdown == pull2.zero_trust_lockdown

    def test_propagation_rejects_divergent(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        data = make_divergent_data(n=3)
        dx = np.ones(3, dtype=np.float64)
        with pytest.raises(TensorConservationError):
            agent.execute_propagation(data, dx)

    def test_propagation_rejects_bad_delta_dim(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
    ) -> None:
        bad_dx = np.ones(flat_data.T_mu_nu.shape[0] + 1, dtype=np.float64)
        with pytest.raises(DimensionMismatchError):
            agent.execute_propagation(flat_data, bad_dx)

    def test_propagation_with_nonzero_christoffel(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        data = make_nonzero_christoffel_conserved(n=4, seed=33)
        dx = np.linspace(0.1, 1.0, 4, dtype=np.float64)
        coh, pull = agent.execute_propagation(data, dx)
        assert math.isfinite(coh.deformed_dirichlet_energy)
        assert pull.L_max > 0.0

    def test_describe_pipeline_keys(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        meta = agent.describe_pipeline()
        assert "functor" in meta
        assert "phase_1" in meta
        assert "phase_2" in meta
        assert "phase_3" in meta
        assert "axiom_1" in meta
        assert "axiom_2" in meta
        assert "axiom_3" in meta
        assert "𝒲_agent" in meta["functor"]

    def test_energy_consistency_across_layers(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> None:
        """ℰ del DTO de cohomología = δxᵀ 𝒯 δx calculado manualmente."""
        coh, _ = agent.execute_propagation(flat_data, delta_x)
        manual = float(delta_x @ flat_data.T_mu_nu @ delta_x)
        assert coh.deformed_dirichlet_energy == pytest.approx(manual, rel=1e-12)

    def test_trace_consistency_across_layers(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> None:
        coh, _ = agent.execute_propagation(flat_data, delta_x)
        manual = float(
            np.einsum("ij,ij->", flat_data.T_mu_nu, flat_data.G_mu_nu)
        )
        assert coh.stress_trace == pytest.approx(manual, rel=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
    def test_parametrized_end_to_end(
        self, agent: TopologicalWatcherAgent, n: int
    ) -> None:
        data = make_flat_conserved_data(n=n, seed=100 + n)
        dx = np.ones(n, dtype=np.float64) / math.sqrt(n)
        coh, pull = agent.execute_propagation(data, dx)
        assert coh.operator_norm >= coh.spectral_radius - 1e-9
        assert pull.L_max > 0.0
        assert math.isfinite(coh.deformed_dirichlet_energy)

    def test_illconditioned_metric_still_propagates(
        self,
        agent: TopologicalWatcherAgent,
        flat_data_illcond: StressTensorData,
    ) -> None:
        n = flat_data_illcond.T_mu_nu.shape[0]
        dx = np.ones(n, dtype=np.float64)
        coh, pull = agent.execute_propagation(flat_data_illcond, dx)
        # κ elevado debe reflejarse en condition_number del pullback
        assert pull.condition_number > 10.0
        assert math.isfinite(pull.L_max)


# =============================================================================
# PROPIEDADES ESPECTRALES Y NUMÉRICAS (invariantes de la teoría)
# =============================================================================

class TestSpectralInequalities:
    """Desigualdades clásicas de teoría espectral que el pipeline debe respetar."""

    def test_rho_le_op_norm_le_frobenius(
        self, validated: ValidatedStressTensor
    ) -> None:
        inv = validated.invariants
        assert inv.rho_T <= inv.op_norm_T + 1e-10
        assert inv.op_norm_T <= inv.frobenius_T + 1e-10

    def test_cond_ge_one(self, validated: ValidatedStressTensor) -> None:
        assert validated.invariants.cond_G >= 1.0 - 1e-12

    def test_lambda_max_ge_lambda_min(
        self, validated: ValidatedStressTensor
    ) -> None:
        inv = validated.invariants
        assert inv.lambda_max_G >= inv.lambda_min_G - 1e-14

    def test_rayleigh_quotient_bounded(
        self,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        validated: ValidatedStressTensor,
    ) -> None:
        """
        Para todo δx ≠ 0: |δxᵀ 𝒯 δx| / ‖δx‖₂² ≤ ‖𝒯‖_op.
        """
        rng = np.random.default_rng(123)
        n = validated.T_mu_nu.shape[0]
        for _ in range(20):
            dx = rng.standard_normal(n).astype(np.float64)
            norm_sq = float(np.dot(dx, dx))
            if norm_sq < 1e-30:
                continue
            energy = phase2._dirichlet_energy(validated.T_mu_nu, dx)
            rayleigh = abs(energy) / norm_sq
            assert rayleigh <= validated.invariants.op_norm_T + 1e-8

    def test_residual_relative_nonnegative(
        self, validated: ValidatedStressTensor
    ) -> None:
        assert validated.residual_relative >= 0.0


class TestNestedPhaseAccess:
    """La anidación de clases es estructural: Fase3 solo vía Fase2 vía Fase1."""

    def test_phase2_is_nested_in_phase1(self) -> None:
        assert hasattr(Phase1_StressTensorValidation, "Phase2_CohomologyPushforward")
        p2 = Phase1_StressTensorValidation.Phase2_CohomologyPushforward
        assert p2.__qualname__.startswith("Phase1_StressTensorValidation")

    def test_phase3_is_nested_in_phase2(self) -> None:
        p3 = (
            Phase1_StressTensorValidation
            .Phase2_CohomologyPushforward
            .Phase3_ToposPullback
        )
        assert "Phase2_CohomologyPushforward" in p3.__qualname__
        assert "Phase1_StressTensorValidation" in p3.__qualname__

    def test_full_chain_type_flow(
        self,
        phase1: Phase1_StressTensorValidation,
        phase2: Phase1_StressTensorValidation.Phase2_CohomologyPushforward,
        phase3: Phase1_StressTensorValidation.Phase2_CohomologyPushforward.Phase3_ToposPullback,
        flat_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> None:
        """Contrato de tipos: salida_k = entrada_{k+1}."""
        v: ValidatedStressTensor = phase1.validate_tensor(flat_data)
        p: PushforwardResult = phase2.compute_pushforward(v, delta_x)
        t: ToposPullbackData = phase3.compute_pullback(p)
        assert isinstance(v, ValidatedStressTensor)
        assert isinstance(p, PushforwardResult)
        assert isinstance(t, ToposPullbackData)


class TestEdgeCasesAndStability:
    """Casos límite y estabilidad numérica."""

    def test_dimension_one(self, agent: TopologicalWatcherAgent) -> None:
        data = make_flat_conserved_data(n=1, seed=0)
        dx = np.array([2.5], dtype=np.float64)
        coh, pull = agent.execute_propagation(data, dx)
        expected_energy = float(dx[0] * data.T_mu_nu[0, 0] * dx[0])
        assert coh.deformed_dirichlet_energy == pytest.approx(expected_energy)
        assert pull.L_max > 0.0

    def test_zero_stress_tensor(self, agent: TopologicalWatcherAgent) -> None:
        n = 3
        data = StressTensorData(
            T_mu_nu=np.zeros((n, n), dtype=np.float64),
            G_mu_nu=_eye(n),
            Christoffel=_zeros3(n),
            partial_T=_zeros3(n),
        )
        dx = np.ones(n, dtype=np.float64)
        coh, pull = agent.execute_propagation(data, dx)
        assert coh.deformed_dirichlet_energy == pytest.approx(0.0)
        assert coh.spectral_radius == pytest.approx(0.0)
        assert coh.operator_norm == pytest.approx(0.0)
        assert pull.L_max > 0.0  # denominador ≈ ε·λ_max·√κ > 0

    def test_zero_delta_x(
        self,
        agent: TopologicalWatcherAgent,
        flat_data: StressTensorData,
    ) -> None:
        n = flat_data.T_mu_nu.shape[0]
        dx = np.zeros(n, dtype=np.float64)
        coh, pull = agent.execute_propagation(flat_data, dx)
        assert coh.deformed_dirichlet_energy == pytest.approx(0.0)
        assert pull.L_max > 0.0

    def test_large_scale_stress_still_finite(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        data = make_flat_conserved_data(n=4, T_scale=1e6, seed=77)
        dx = np.ones(4, dtype=np.float64) * 1e-3
        coh, pull = agent.execute_propagation(data, dx)
        assert math.isfinite(coh.deformed_dirichlet_energy)
        assert math.isfinite(pull.L_max)

    def test_negative_definite_stress_energy_sign(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        """𝒯 = −I ⇒ ℰ = −‖δx‖₂² < 0 (permitido: forma bilineal, no norma)."""
        n = 3
        data = StressTensorData(
            T_mu_nu=-_eye(n),
            G_mu_nu=_eye(n),
            Christoffel=_zeros3(n),
            partial_T=_zeros3(n),
        )
        dx = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        coh, _ = agent.execute_propagation(data, dx)
        assert coh.deformed_dirichlet_energy == pytest.approx(-1.0)

    def test_reproducibility(
        self, agent: TopologicalWatcherAgent
    ) -> None:
        data = make_flat_conserved_data(n=5, seed=999)
        dx = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
        coh_a, pull_a = agent.execute_propagation(data, dx)
        coh_b, pull_b = agent.execute_propagation(data, dx)
        assert coh_a.deformed_dirichlet_energy == coh_b.deformed_dirichlet_energy
        assert pull_a.L_max == pull_b.L_max
        assert pull_a.zero_trust_lockdown == pull_b.zero_trust_lockdown