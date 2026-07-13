# tests/unit/omega/test_raychaudhuri_focal_fibrator.py
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de pruebas unitarias: RaychaudhuriFocalFibrator                        ║
║ Ubicación: tests/unit/omega/test_raychaudhuri_focal_fibrator.py              ║
║ Versión: 3.0.0-Nested-Caustic-Spectral-Topos                                 ║
║ Cobertura: Fase 1 → Fase 2 → Fase 3 + funtor 𝒲_Raychaudhuri + teoremas      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de verificación (GR + teoría espectral + ODE):
  • Congruencias sintéticas con ω≡0, σ sim+traceless, B = (θ/(n−1))I + σ.
  • SEC satisfecha/violada con 𝒯 y u controlados.
  • Cáustica: cota Hawking–Penrose, solución exacta α=0 y α>0, IVP numérico.
  • Casos patológicos: vorticidad, σ no traceless, θ₀≥0, u no normalizado,
    dimensiones inconsistentes, métrica no-PD.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─── SUT ─────────────────────────────────────────────────────────────────────
from app.omega.raychaudhuri_focal_fibrator import (
    _MACHINE_EPSILON,
    _VORTICITY_TOLERANCE,
    _SYMMETRY_REL_TOL,
    _TRACELESS_TOL,
    _DECOMPOSITION_TOL,
    _U_NORMALIZATION_TOL,
    _CAUSTIC_THETA_THRESHOLD,
    _MAX_AFFINE_PARAMETER,
    _COND_MAX,
    _MIN_SPATIAL_DIM,
    RaychaudhuriFibratorError,
    FocalDivergenceVetoError,
    StrongEnergyViolationError,
    VorticityAnomalyError,
    ShearAnomalyError,
    MetricSignatureError,
    DimensionMismatchError,
    NormalizationError,
    CausticNotReachedError,
    FocalViabilityFlags,
    CausticMethod,
    MetricSignature,
    KinematicExpansionData,
    EnergyConditionCertificate,
    FocalLengthResult,
    Phase1_RaychaudhuriKinematics,
    RaychaudhuriFocalFibrator,
)


# =============================================================================
# HELPERS DE CONSTRUCCIÓN GEOMÉTRICA
# =============================================================================

def _eye(n: int) -> NDArray[np.float64]:
    return np.eye(n, dtype=np.float64)


def _zeros(n: int) -> NDArray[np.float64]:
    return np.zeros((n, n), dtype=np.float64)


def make_spd_metric(
    n: int,
    cond: float = 1.0,
    seed: int = 1,
) -> NDArray[np.float64]:
    """Métrica SPD con κ(G) ≈ cond."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam_min = 1.0
    lam_max = float(cond)
    if n == 1:
        eigs = np.array([lam_min], dtype=np.float64)
    else:
        eigs = np.linspace(lam_min, lam_max, n, dtype=np.float64)
    G = (Q @ np.diag(eigs) @ Q.T).astype(np.float64)
    return 0.5 * (G + G.T)


def make_traceless_symmetric_shear(
    n: int,
    scale: float = 0.1,
    seed: int = 0,
    G_inv: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    r"""
    σ simétrico y traceless respecto a G (euclídeo si G_inv=I):
      σ ← (A+Aᵀ)/2 − (Tr_G(σ)/n) · G   (proyección al subespacio traceless).
    En el caso euclídeo G=I: restamos (tr/n) I.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    sigma = 0.5 * (A + A.T) * scale
    if G_inv is None:
        tr = float(np.trace(sigma))
        sigma = sigma - (tr / n) * _eye(n)
    else:
        tr = float(np.einsum("ij,ij->", sigma, G_inv))
        # Para G=I, G_inv=I; corrección euclídea genérica suficiente en tests
        # con métrica no trivial usamos proyección aproximada sobre I en carta local.
        sigma = sigma - (tr / n) * _eye(n)
        # Re-simetrizar
        sigma = 0.5 * (sigma + sigma.T)
    return sigma.astype(np.float64)


def make_consistent_congruence(
    n: int = 4,
    theta: float = -2.0,
    shear_scale: float = 0.05,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construye (B, σ, ω) consistentes con la descomposición euclídea:

        B = (θ/(n−1)) I + σ + ω,   ω ≡ 0,   Tr(σ)=0,   σ=σᵀ.

    Así residual de descomposición ≈ 0 y θ_computed = Tr(B) = θ.
    """
    assert n >= _MIN_SPATIAL_DIM
    sigma = make_traceless_symmetric_shear(n, scale=shear_scale, seed=seed)
    omega = _zeros(n)
    expansion_part = (theta / float(n - 1)) * _eye(n)
    B = expansion_part + sigma + omega
    return B.astype(np.float64), sigma, omega


def make_normalized_u(
    n: int,
    G: NDArray[np.float64] | None = None,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Vector u con G(u,u)=+1 (riemanniano)."""
    rng = np.random.default_rng(seed)
    if G is None:
        G = _eye(n)
    v = rng.standard_normal(n).astype(np.float64)
    # Normalizar: u = v / √G(v,v)
    gvv = float(v @ G @ v)
    if gvv <= 0.0:
        v = np.ones(n, dtype=np.float64)
        gvv = float(v @ G @ v)
    u = v / math.sqrt(gvv)
    return u


def make_sec_satisfying_stress(
    n: int,
    u: NDArray[np.float64],
    G: NDArray[np.float64] | None = None,
    sec_target: float = 0.5,
) -> NDArray[np.float64]:
    r"""
    Construye 𝒯 simétrico tal que SEC(u) ≈ sec_target ≥ 0.

    Caso euclídeo G=I, u unitario:
      𝒯 = a u⊗u + b (I − u⊗u)  (isótropo en el ortocomplemento)
      T(u,u) = a
      Tr(𝒯) = a + b(n−1)
      SEC = a − ½(a + b(n−1)) · 1 = ½a − ½b(n−1)

    Elegimos b=0, a = 2·sec_target ⇒ SEC = sec_target.
    """
    if G is None:
        G = _eye(n)
    a = 2.0 * sec_target
    # 𝒯 = a u⊗u  (simétrico, SEC = a − ½ a · G(u,u) = a/2 si G(u,u)=1)
    T = a * np.outer(u, u)
    # Proyectar a parte simétrica (ya lo es)
    return 0.5 * (T + T.T).astype(np.float64)


def make_sec_violating_stress(
    n: int,
    u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """𝒯 = −c u⊗u con c>0 ⇒ SEC < 0."""
    c = 2.0
    T = -c * np.outer(u, u)
    return 0.5 * (T + T.T).astype(np.float64)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def n() -> int:
    return 4


@pytest.fixture
def metric(n: int) -> NDArray[np.float64]:
    return _eye(n)


@pytest.fixture
def congruence(n: int) -> Tuple[NDArray, NDArray, NDArray]:
    return make_consistent_congruence(n=n, theta=-2.0, shear_scale=0.05, seed=42)


@pytest.fixture
def u_vec(n: int, metric: NDArray[np.float64]) -> NDArray[np.float64]:
    return make_normalized_u(n, G=metric, seed=7)


@pytest.fixture
def stress_ok(
    n: int, u_vec: NDArray[np.float64], metric: NDArray[np.float64]
) -> NDArray[np.float64]:
    return make_sec_satisfying_stress(n, u_vec, G=metric, sec_target=0.5)


@pytest.fixture
def phase1(metric: NDArray[np.float64]) -> Phase1_RaychaudhuriKinematics:
    return Phase1_RaychaudhuriKinematics(metric=metric)


@pytest.fixture
def kinematics(
    phase1: Phase1_RaychaudhuriKinematics,
    congruence: Tuple[NDArray, NDArray, NDArray],
) -> KinematicExpansionData:
    B, sigma, omega = congruence
    return phase1.compute_kinematics(B, sigma, omega)


@pytest.fixture
def phase2(
    phase1: Phase1_RaychaudhuriKinematics,
) -> Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition:
    return Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition(
        metric=phase1._G,
        metric_inv=phase1._G_inv,
        signature=MetricSignature.RIEMANNIAN,
    )


@pytest.fixture
def energy_cert(
    phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
    kinematics: KinematicExpansionData,
    stress_ok: NDArray[np.float64],
    u_vec: NDArray[np.float64],
) -> EnergyConditionCertificate:
    return phase2.certify_energy(kinematics, stress_ok, u_vec)


@pytest.fixture
def phase3(
    n: int,
) -> Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition.Phase3_FocalCollapse:
    return (
        Phase1_RaychaudhuriKinematics
        .Phase2_StrongEnergyCondition
        .Phase3_FocalCollapse(
            spatial_dim=n,
            method=CausticMethod.ANALYTICAL_EXACT,
        )
    )


@pytest.fixture
def agent(n: int, metric: NDArray[np.float64]) -> RaychaudhuriFocalFibrator:
    return RaychaudhuriFocalFibrator(
        spatial_dimensions=n,
        metric=metric,
        caustic_method=CausticMethod.ANALYTICAL_EXACT,
    )


# =============================================================================
# FASE 0 – CONSTANTES, EXCEPCIONES, FLAGS, ENUMS
# =============================================================================

class TestConstantsAndExceptions:
    def test_machine_epsilon_positive(self) -> None:
        assert _MACHINE_EPSILON > 0.0
        assert _MACHINE_EPSILON == float(np.finfo(np.float64).eps)

    def test_vorticity_tol_positive(self) -> None:
        assert _VORTICITY_TOLERANCE > 0.0

    def test_min_spatial_dim(self) -> None:
        assert _MIN_SPATIAL_DIM >= 2

    def test_exception_hierarchy(self) -> None:
        assert issubclass(FocalDivergenceVetoError, RaychaudhuriFibratorError)
        assert issubclass(StrongEnergyViolationError, RaychaudhuriFibratorError)
        assert issubclass(VorticityAnomalyError, RaychaudhuriFibratorError)
        assert issubclass(ShearAnomalyError, RaychaudhuriFibratorError)
        assert issubclass(MetricSignatureError, RaychaudhuriFibratorError)
        assert issubclass(DimensionMismatchError, RaychaudhuriFibratorError)
        assert issubclass(NormalizationError, RaychaudhuriFibratorError)
        assert issubclass(CausticNotReachedError, RaychaudhuriFibratorError)

    def test_focal_viability_flags_order_unit(self) -> None:
        assert FocalViabilityFlags.ALL.is_order_unit() is True
        assert FocalViabilityFlags.NONE.is_order_unit() is False
        assert (
            FocalViabilityFlags.ZERO_VORTICITY
            | FocalViabilityFlags.STRONG_ENERGY_SATISFIED
        ).is_order_unit() is False

    def test_caustic_method_enum(self) -> None:
        names = {m.name for m in CausticMethod}
        assert names == {
            "ANALYTICAL_BOUND",
            "ANALYTICAL_EXACT",
            "NUMERICAL_IVP",
        }

    def test_metric_signature_enum(self) -> None:
        names = {s.name for s in MetricSignature}
        assert names == {"RIEMANNIAN", "LORENTZIAN"}

    def test_dto_immutability(self, kinematics: KinematicExpansionData) -> None:
        with pytest.raises(Exception):
            kinematics.expansion_scalar = 0.0  # type: ignore[misc]


# =============================================================================
# FASE 1 – CINEMÁTICA: MÉTRICA Y DIMENSIONES
# =============================================================================

class TestPhase1Metric:
    def test_identity_metric_ok(self, n: int) -> None:
        p1 = Phase1_RaychaudhuriKinematics(metric=_eye(n))
        assert p1._lam_min_G == pytest.approx(1.0)
        assert p1._lam_max_G == pytest.approx(1.0)
        assert p1._cond_G == pytest.approx(1.0)

    def test_spd_known_cond(self, n: int) -> None:
        G = make_spd_metric(n, cond=25.0, seed=3)
        p1 = Phase1_RaychaudhuriKinematics(metric=G)
        assert p1._cond_G == pytest.approx(25.0, rel=1e-5)
        assert p1._lam_min_G > 0.0

    def test_asymmetric_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[0, 1] = 0.5
        with pytest.raises(MetricSignatureError, match="simétrica"):
            Phase1_RaychaudhuriKinematics(metric=G)

    def test_indefinite_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[-1, -1] = -1.0
        with pytest.raises(MetricSignatureError):
            Phase1_RaychaudhuriKinematics(metric=G)

    def test_singular_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[-1, -1] = 0.0
        with pytest.raises(MetricSignatureError):
            Phase1_RaychaudhuriKinematics(metric=G)

    def test_dimension_one_metric_raises(self) -> None:
        with pytest.raises(MetricSignatureError):
            Phase1_RaychaudhuriKinematics(metric=_eye(1))

    def test_non_square_metric_raises(self) -> None:
        with pytest.raises(MetricSignatureError):
            Phase1_RaychaudhuriKinematics(
                metric=np.ones((3, 4), dtype=np.float64)
            )

    def test_vorticity_tol_rejects_nonpositive(self, n: int) -> None:
        with pytest.raises(ValueError, match="vorticity_tol"):
            Phase1_RaychaudhuriKinematics(metric=_eye(n), vorticity_tol=0.0)


class TestPhase1Dimensions:
    def test_valid_dimensions(
        self,
        phase1: Phase1_RaychaudhuriKinematics,
        congruence: Tuple[NDArray, NDArray, NDArray],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        assert phase1._assert_dimensions(B, sigma, omega) == n

    def test_B_non_square_raises(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        with pytest.raises(DimensionMismatchError, match="cuadrado"):
            phase1._assert_dimensions(
                np.ones((n, n + 1)), _zeros(n), _zeros(n)
            )

    def test_sigma_shape_mismatch(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        with pytest.raises(DimensionMismatchError, match="σ"):
            phase1._assert_dimensions(_eye(n), _zeros(n + 1), _zeros(n))

    def test_omega_shape_mismatch(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        with pytest.raises(DimensionMismatchError, match="ω"):
            phase1._assert_dimensions(_eye(n), _zeros(n), _zeros(n + 1))

    def test_G_dim_mismatch_with_B(self, n: int) -> None:
        p1 = Phase1_RaychaudhuriKinematics(metric=_eye(n))
        with pytest.raises(DimensionMismatchError, match="G shape"):
            p1._assert_dimensions(_eye(n + 1), _zeros(n + 1), _zeros(n + 1))


# =============================================================================
# FASE 1 – SHEAR Y VORTICIDAD
# =============================================================================

class TestPhase1Shear:
    def test_valid_traceless_symmetric(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        sigma = make_traceless_symmetric_shear(n, scale=0.2, seed=1)
        sigma_sq = phase1._check_shear(sigma)
        assert sigma_sq >= 0.0

    def test_asymmetric_shear_raises(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        sigma = np.zeros((n, n), dtype=np.float64)
        sigma[0, 1] = 1.0  # no simétrico
        with pytest.raises(ShearAnomalyError, match="simétrico"):
            phase1._check_shear(sigma)

    def test_non_traceless_shear_raises(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        sigma = _eye(n)  # Tr = n ≠ 0
        with pytest.raises(ShearAnomalyError, match="traceless"):
            phase1._check_shear(sigma)

    def test_zero_shear_ok(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        assert phase1._check_shear(_zeros(n)) == pytest.approx(0.0)


class TestPhase1Vorticity:
    def test_zero_vorticity_ok(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        assert phase1._check_vorticity(_zeros(n)) == pytest.approx(0.0)

    def test_antisymmetric_nonzero_raises(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        omega = _zeros(n)
        omega[0, 1] = 0.5
        omega[1, 0] = -0.5
        with pytest.raises(VorticityAnomalyError, match="solenoidal"):
            phase1._check_vorticity(omega)

    def test_non_antisymmetric_raises(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        omega = _zeros(n)
        omega[0, 1] = 0.3
        omega[1, 0] = 0.1  # no anti
        with pytest.raises(VorticityAnomalyError, match="antisimétrico"):
            phase1._check_vorticity(omega)

    def test_tiny_vorticity_within_tol(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        """ω² por debajo de tolerancia → aceptado."""
        omega = _zeros(n)
        eps = 1e-10
        omega[0, 1] = eps
        omega[1, 0] = -eps
        # ω² ~ O(eps²) << tol
        omega_sq = phase1._check_vorticity(omega)
        assert omega_sq <= _VORTICITY_TOLERANCE


# =============================================================================
# FASE 1 – DESCOMPOSICIÓN Y COMPUTE_KINEMATICS
# =============================================================================

class TestPhase1Decomposition:
    def test_consistent_congruence_near_zero_residual(
        self,
        phase1: Phase1_RaychaudhuriKinematics,
        congruence: Tuple[NDArray, NDArray, NDArray],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        theta = float(np.trace(B))
        res = phase1._check_decomposition(B, theta, sigma, omega, n)
        assert res < _DECOMPOSITION_TOL

    def test_inconsistent_logs_but_returns(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        """B arbitrario ⇒ residual alto, pero no lanza (warning only)."""
        B = np.ones((n, n), dtype=np.float64)
        sigma = make_traceless_symmetric_shear(n, seed=2)
        omega = _zeros(n)
        theta = float(np.trace(B))
        res = phase1._check_decomposition(B, theta, sigma, omega, n)
        assert res > _DECOMPOSITION_TOL


class TestPhase1ComputeKinematics:
    def test_happy_path(
        self,
        phase1: Phase1_RaychaudhuriKinematics,
        congruence: Tuple[NDArray, NDArray, NDArray],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        kin = phase1.compute_kinematics(B, sigma, omega)
        assert isinstance(kin, KinematicExpansionData)
        assert kin.spatial_dim == n
        assert kin.vorticity_magnitude == pytest.approx(0.0, abs=1e-14)
        assert kin.shear_magnitude >= 0.0
        assert kin.op_norm_B >= 0.0
        assert kin.frobenius_B >= 0.0
        assert kin.metric_cond >= 1.0 - 1e-12
        assert kin.residual_decomposition < _DECOMPOSITION_TOL

    def test_theta_equals_trace_B(
        self,
        phase1: Phase1_RaychaudhuriKinematics,
        congruence: Tuple[NDArray, NDArray, NDArray],
    ) -> None:
        B, sigma, omega = congruence
        kin = phase1.compute_kinematics(B, sigma, omega)
        assert kin.expansion_scalar == pytest.approx(float(np.trace(B)))

    def test_prescribed_theta_recovered(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        theta_target = -3.5
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=theta_target, shear_scale=0.0, seed=0
        )
        kin = phase1.compute_kinematics(B, sigma, omega)
        assert kin.expansion_scalar == pytest.approx(theta_target, abs=1e-12)
        assert kin.shear_magnitude == pytest.approx(0.0, abs=1e-12)

    def test_rejects_vorticity(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        B, sigma, _ = make_consistent_congruence(n=n, theta=-1.0)
        omega = _zeros(n)
        omega[0, 1] = 1.0
        omega[1, 0] = -1.0
        with pytest.raises(VorticityAnomalyError):
            phase1.compute_kinematics(B, sigma, omega)

    def test_rejects_bad_shear(
        self, phase1: Phase1_RaychaudhuriKinematics, n: int
    ) -> None:
        B, _, omega = make_consistent_congruence(n=n, theta=-1.0)
        with pytest.raises(ShearAnomalyError):
            phase1.compute_kinematics(B, _eye(n), omega)

    def test_spectral_inequalities(
        self, kinematics: KinematicExpansionData
    ) -> None:
        # ‖B‖_op ≤ ‖B‖_F
        assert kinematics.op_norm_B <= kinematics.frobenius_B + 1e-10

    @pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 7])
    def test_parametrized_dimensions(self, n_dim: int) -> None:
        p1 = Phase1_RaychaudhuriKinematics(metric=_eye(n_dim))
        B, sigma, omega = make_consistent_congruence(
            n=n_dim, theta=-1.5, seed=n_dim
        )
        kin = p1.compute_kinematics(B, sigma, omega)
        assert kin.spatial_dim == n_dim
        assert kin.expansion_scalar == pytest.approx(-1.5, abs=1e-10)


# =============================================================================
# FASE 2 – CONDICIÓN DE ENERGÍA FUERTE (SEC)
# =============================================================================

class TestPhase2DimensionsAndSymmetry:
    def test_stress_shape_mismatch_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        bad_T = _eye(n + 1)
        with pytest.raises(DimensionMismatchError, match="𝒯"):
            phase2.certify_energy(kinematics, bad_T, u_vec)

    def test_u_shape_mismatch_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        stress_ok: NDArray[np.float64],
        n: int,
    ) -> None:
        with pytest.raises(DimensionMismatchError, match="u"):
            phase2.certify_energy(
                kinematics, stress_ok, np.ones(n + 1, dtype=np.float64)
            )

    def test_asymmetric_stress_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        T = _zeros(n)
        T[0, 1] = 1.0
        with pytest.raises(StrongEnergyViolationError, match="simétrico"):
            phase2.certify_energy(kinematics, T, u_vec)


class TestPhase2Normalization:
    def test_normalized_u_passes(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        u_vec: NDArray[np.float64],
        metric: NDArray[np.float64],
    ) -> None:
        g_uu = phase2._normalize_check_u(u_vec)
        assert g_uu == pytest.approx(1.0, abs=_U_NORMALIZATION_TOL)

    def test_unnormalized_u_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        n: int,
    ) -> None:
        u = 3.0 * np.ones(n, dtype=np.float64)
        with pytest.raises(NormalizationError, match="no normalizado"):
            phase2._normalize_check_u(u)

    def test_zero_u_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        n: int,
    ) -> None:
        with pytest.raises(NormalizationError):
            phase2._normalize_check_u(np.zeros(n, dtype=np.float64))

    def test_lorentzian_signature_target(self, n: int) -> None:
        """Firma lorentziana espera G(u,u)=−1; con G euclídea + u real no se alcanza."""
        p1 = Phase1_RaychaudhuriKinematics(metric=_eye(n))
        p2 = Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition(
            metric=p1._G,
            metric_inv=p1._G_inv,
            signature=MetricSignature.LORENTZIAN,
        )
        u = make_normalized_u(n)  # G(u,u)=+1
        with pytest.raises(NormalizationError):
            p2._normalize_check_u(u)


class TestPhase2SECValue:
    def test_sec_formula_analytic(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        n: int,
    ) -> None:
        u = np.zeros(n, dtype=np.float64)
        u[0] = 1.0  # unitario euclídeo
        sec_target = 0.75
        T = make_sec_satisfying_stress(n, u, sec_target=sec_target)
        sec_val, tr = phase2._compute_sec_value(T, u)
        assert sec_val == pytest.approx(sec_target, rel=1e-10)

    def test_certify_energy_happy_path(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        cert = phase2.certify_energy(kinematics, stress_ok, u_vec)
        assert isinstance(cert, EnergyConditionCertificate)
        assert cert.is_strongly_attractive is True
        assert cert.ricci_contraction >= 0.0
        assert cert.sec_value >= -_MACHINE_EPSILON
        assert cert.kinematics is kinematics
        assert cert.u_normalization == pytest.approx(1.0, abs=1e-8)

    def test_sec_violation_raises(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        T_bad = make_sec_violating_stress(n, u_vec)
        with pytest.raises(StrongEnergyViolationError, match="SEC"):
            phase2.certify_energy(kinematics, T_bad, u_vec)

    def test_ricci_equals_sec_under_einstein(
        self,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        kinematics: KinematicExpansionData,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        cert = phase2.certify_energy(kinematics, stress_ok, u_vec)
        assert cert.ricci_contraction == pytest.approx(
            max(0.0, cert.sec_value), abs=1e-14
        )

    def test_kinematics_propagated_intact(
        self,
        energy_cert: EnergyConditionCertificate,
        kinematics: KinematicExpansionData,
    ) -> None:
        assert energy_cert.kinematics.expansion_scalar == kinematics.expansion_scalar
        assert energy_cert.kinematics.shear_magnitude == kinematics.shear_magnitude
        assert energy_cert.kinematics.spatial_dim == kinematics.spatial_dim

    def test_u_norm_tol_rejects_nonpositive(self, n: int) -> None:
        p1 = Phase1_RaychaudhuriKinematics(metric=_eye(n))
        with pytest.raises(ValueError, match="u_norm_tol"):
            Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition(
                metric=p1._G, metric_inv=p1._G_inv, u_norm_tol=0.0
            )


# =============================================================================
# FASE 3 – RAYCHAUDHURI RHS, HAWKING–PENROSE, CÁUSTICA
# =============================================================================

class TestPhase3RaychaudhuriRHS:
    def test_rhs_negative_for_convergent(
        self, phase3, kinematics: KinematicExpansionData
    ) -> None:
        theta = kinematics.expansion_scalar  # < 0
        sigma_sq = kinematics.shear_magnitude
        ricci = 0.5
        dtheta = phase3.raychaudhuri_rhs(theta, sigma_sq, ricci)
        # −θ²/(n−1) − σ² − R < 0 siempre si σ²,R ≥ 0
        assert dtheta < 0.0

    def test_rhs_formula(
        self, phase3, n: int
    ) -> None:
        theta, sigma_sq, ricci = -3.0, 0.25, 0.5
        expected = -(theta ** 2) / (n - 1) - sigma_sq - ricci
        assert phase3.raychaudhuri_rhs(theta, sigma_sq, ricci) == pytest.approx(
            expected
        )

    def test_rhs_zero_only_if_all_zero(self, phase3) -> None:
        # θ=0, σ²=0, R=0 ⇒ dθ/dτ = 0
        assert phase3.raychaudhuri_rhs(0.0, 0.0, 0.0) == pytest.approx(0.0)


class TestPhase3HawkingPenrose:
    def test_bound_formula(self, phase3, n: int) -> None:
        theta0 = -2.0
        tau_hp = phase3.hawking_penrose_bound(theta0, n)
        assert tau_hp == pytest.approx((n - 1) / 2.0)

    def test_bound_rejects_nonnegative_theta(self, phase3, n: int) -> None:
        with pytest.raises(FocalDivergenceVetoError):
            phase3.hawking_penrose_bound(0.0, n)
        with pytest.raises(FocalDivergenceVetoError):
            phase3.hawking_penrose_bound(1.0, n)

    def test_more_negative_theta_shorter_bound(self, phase3, n: int) -> None:
        t1 = phase3.hawking_penrose_bound(-1.0, n)
        t2 = phase3.hawking_penrose_bound(-4.0, n)
        assert t2 < t1


class TestPhase3AnalyticalExact:
    def test_alpha_zero_caustic(
        self, phase3, n: int
    ) -> None:
        """α=0 ⇒ τ_c = −(n−1)/θ₀  (= τ_HP)."""
        theta0 = -2.0
        tau_c, f_opt, area = phase3._analytical_exact_caustic(theta0, alpha=0.0)
        expected = (n - 1) / abs(theta0)
        assert tau_c == pytest.approx(expected)
        assert area == pytest.approx(0.0)
        assert f_opt == pytest.approx(expected)

    def test_alpha_positive_caustic_finite(
        self, phase3
    ) -> None:
        theta0 = -2.0
        alpha = 0.5
        tau_c, f_opt, area = phase3._analytical_exact_caustic(theta0, alpha)
        assert tau_c > 0.0
        assert math.isfinite(tau_c)
        assert area == pytest.approx(0.0)
        assert f_opt >= 0.0

    def test_alpha_positive_below_hp_bound(
        self, phase3, n: int
    ) -> None:
        """Con α>0 la cáustica llega antes o en τ_HP."""
        theta0 = -2.0
        alpha = 1.0
        tau_c, _, _ = phase3._analytical_exact_caustic(theta0, alpha)
        tau_hp = phase3.hawking_penrose_bound(theta0, n)
        assert tau_c <= tau_hp + 1e-9

    def test_alpha_negative_clipped(
        self, phase3, n: int
    ) -> None:
        """α<0 se recorta a 0 (defensivo)."""
        theta0 = -1.0
        tau_c, _, _ = phase3._analytical_exact_caustic(theta0, alpha=-0.5)
        assert tau_c == pytest.approx((n - 1) / abs(theta0))


class TestPhase3NumericalIVP:
    @pytest.fixture
    def phase3_num(
        self, n: int
    ) -> Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition.Phase3_FocalCollapse:
        return (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse(
                spatial_dim=n,
                method=CausticMethod.NUMERICAL_IVP,
                caustic_theta=-1.0e4,  # umbral menos extremo para tests rápidos
            )
        )

    def test_numerical_reaches_caustic(
        self, phase3_num, n: int
    ) -> None:
        theta0 = -2.0
        sigma_sq = 0.0
        ricci = 0.0
        tau_c, f_opt, area = phase3_num._numerical_caustic(
            theta0, sigma_sq, ricci
        )
        tau_hp = (n - 1) / abs(theta0)
        assert tau_c > 0.0
        # Debe estar en (0, τ_HP] (o muy cerca; umbral finito anticipa un poco)
        assert tau_c <= tau_hp * 1.05
        assert math.isfinite(f_opt)

    def test_numerical_with_alpha(
        self, phase3_num, n: int
    ) -> None:
        theta0 = -3.0
        tau_c, f_opt, area = phase3_num._numerical_caustic(
            theta0, sigma_sq=0.2, ricci_term=0.3
        )
        assert tau_c > 0.0
        assert tau_c <= (n - 1) / abs(theta0) * 1.05


class TestPhase3ComputeFocalLength:
    def test_happy_path_analytical(
        self,
        phase3,
        energy_cert: EnergyConditionCertificate,
    ) -> None:
        result = phase3.compute_focal_length(energy_cert)
        assert isinstance(result, FocalLengthResult)
        assert result.optimal_focal_length >= 0.0
        assert result.caustic_affine_parameter > 0.0
        assert result.hawking_penrose_bound > 0.0
        assert result.initial_dtheta_dtau < 0.0
        assert result.caustic_method == CausticMethod.ANALYTICAL_EXACT
        assert result.viability_flags.is_order_unit()

    def test_tau_c_le_tau_hp(
        self,
        phase3,
        energy_cert: EnergyConditionCertificate,
    ) -> None:
        result = phase3.compute_focal_length(energy_cert)
        assert result.caustic_affine_parameter <= (
            result.hawking_penrose_bound * (1.0 + 1e-6)
        )

    def test_rejects_nonnegative_theta(
        self, phase3, n: int, metric: NDArray[np.float64]
    ) -> None:
        """θ₀ ≥ 0 ⇒ FocalDivergenceVetoError."""
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=+1.0, shear_scale=0.0, seed=0
        )
        p1 = Phase1_RaychaudhuriKinematics(metric=metric)
        kin = p1.compute_kinematics(B, sigma, omega)
        # Certificado sintético con θ>0
        cert = EnergyConditionCertificate(
            ricci_contraction=0.1,
            sec_value=0.1,
            stress_trace=0.2,
            u_normalization=1.0,
            is_strongly_attractive=True,
            kinematics=kin,
        )
        with pytest.raises(FocalDivergenceVetoError, match="no negativa"):
            phase3.compute_focal_length(cert)

    def test_analytical_bound_method(self, n: int, energy_cert) -> None:
        p3 = (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse(
                spatial_dim=n,
                method=CausticMethod.ANALYTICAL_BOUND,
            )
        )
        result = p3.compute_focal_length(energy_cert)
        assert result.caustic_method == CausticMethod.ANALYTICAL_BOUND
        assert result.caustic_affine_parameter == pytest.approx(
            result.hawking_penrose_bound
        )

    def test_numerical_method_e2e(self, n: int, energy_cert) -> None:
        p3 = (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse(
                spatial_dim=n,
                method=CausticMethod.NUMERICAL_IVP,
                caustic_theta=-1.0e4,
            )
        )
        result = p3.compute_focal_length(energy_cert)
        assert result.caustic_method == CausticMethod.NUMERICAL_IVP
        assert result.caustic_affine_parameter > 0.0
        assert FocalViabilityFlags.CAUSTIC_REACHABLE in result.viability_flags

    def test_constructor_rejects_bad_params(self) -> None:
        P3 = (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse
        )
        with pytest.raises(ValueError, match="spatial_dim"):
            P3(spatial_dim=1)
        with pytest.raises(ValueError, match="max_affine"):
            P3(spatial_dim=3, max_affine=0.0)
        with pytest.raises(ValueError, match="caustic_theta"):
            P3(spatial_dim=3, caustic_theta=1.0)

    def test_flags_all_set_on_success(
        self, phase3, energy_cert: EnergyConditionCertificate
    ) -> None:
        result = phase3.compute_focal_length(energy_cert)
        f = result.viability_flags
        assert FocalViabilityFlags.ZERO_VORTICITY in f
        assert FocalViabilityFlags.STRONG_ENERGY_SATISFIED in f
        assert FocalViabilityFlags.NEGATIVE_EXPANSION in f
        assert FocalViabilityFlags.CONVERGENT_RAYCHAUDHURI in f
        assert FocalViabilityFlags.CAUSTIC_REACHABLE in f


# =============================================================================
# FUNCTOR 𝒲_Raychaudhuri – INTEGRACIÓN END-TO-END
# =============================================================================

class TestRaychaudhuriFocalFibrator:
    def test_execute_focalization_happy_path(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        result = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        assert isinstance(result, FocalLengthResult)
        assert result.optimal_focal_length >= 0.0
        assert result.caustic_affine_parameter > 0.0
        assert result.viability_flags.is_order_unit()
        assert result.initial_dtheta_dtau < 0.0

    def test_call_delegates(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        r1 = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        r2 = agent(B, sigma, omega, stress_ok, u_vec)
        assert r1.caustic_affine_parameter == pytest.approx(
            r2.caustic_affine_parameter
        )
        assert r1.optimal_focal_length == pytest.approx(r2.optimal_focal_length)

    def test_rejects_vorticity(
        self,
        agent: RaychaudhuriFocalFibrator,
        n: int,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, _ = make_consistent_congruence(n=n, theta=-2.0)
        omega = _zeros(n)
        omega[0, 1] = 1.0
        omega[1, 0] = -1.0
        with pytest.raises(VorticityAnomalyError):
            agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)

    def test_rejects_sec_violation(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        T_bad = make_sec_violating_stress(n, u_vec)
        with pytest.raises(StrongEnergyViolationError):
            agent.execute_focalization(B, sigma, omega, T_bad, u_vec)

    def test_rejects_divergent_expansion(
        self,
        agent: RaychaudhuriFocalFibrator,
        n: int,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = make_consistent_congruence(n=n, theta=+1.0)
        with pytest.raises(FocalDivergenceVetoError):
            agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)

    def test_rejects_unnormalized_u(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        u_bad = 5.0 * np.ones(n, dtype=np.float64)
        with pytest.raises(NormalizationError):
            agent.execute_focalization(B, sigma, omega, stress_ok, u_bad)

    def test_numerical_agent(
        self,
        n: int,
        metric: NDArray[np.float64],
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        agent_num = RaychaudhuriFocalFibrator(
            spatial_dimensions=n,
            metric=metric,
            use_numerical_integration=True,
        )
        B, sigma, omega = congruence
        result = agent_num.execute_focalization(
            B, sigma, omega, stress_ok, u_vec
        )
        assert result.caustic_method == CausticMethod.NUMERICAL_IVP
        assert result.caustic_affine_parameter > 0.0

    def test_describe_pipeline_keys(
        self, agent: RaychaudhuriFocalFibrator
    ) -> None:
        meta = agent.describe_pipeline()
        for key in (
            "functor",
            "version",
            "phase_1",
            "phase_2",
            "phase_3",
            "axiom_1",
            "axiom_2",
            "axiom_3",
            "caustic_method",
            "signature",
            "spatial_dim",
        ):
            assert key in meta
        assert "𝒲_Raychaudhuri" in meta["functor"]

    def test_hp_bound_consistency(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        result = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        theta0 = float(np.trace(B))
        expected_hp = (n - 1) / abs(theta0)
        assert result.hawking_penrose_bound == pytest.approx(expected_hp, rel=1e-10)
        assert result.caustic_affine_parameter <= expected_hp * (1.0 + 1e-6)

    @pytest.mark.parametrize("n_dim", [2, 3, 4, 5])
    def test_parametrized_end_to_end(self, n_dim: int) -> None:
        G = _eye(n_dim)
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n_dim,
            metric=G,
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        B, sigma, omega = make_consistent_congruence(
            n=n_dim, theta=-2.5, shear_scale=0.02, seed=n_dim
        )
        u = make_normalized_u(n_dim, G=G, seed=n_dim + 1)
        T = make_sec_satisfying_stress(n_dim, u, G=G, sec_target=0.3)
        result = agent.execute_focalization(B, sigma, omega, T, u)
        assert result.viability_flags.is_order_unit()
        assert result.caustic_affine_parameter > 0.0
        assert math.isfinite(result.optimal_focal_length)

    def test_constructor_rejects_dim_one(self) -> None:
        with pytest.raises(ValueError, match="spatial_dimensions"):
            RaychaudhuriFocalFibrator(spatial_dimensions=1)

    def test_zero_shear_alpha_zero_matches_hp(
        self, n: int, metric: NDArray[np.float64]
    ) -> None:
        """σ=0 y SEC→0 (R≈0) ⇒ τ_c = τ_HP exacto (α=0)."""
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n,
            metric=metric,
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        theta0 = -4.0
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=theta0, shear_scale=0.0, seed=0
        )
        u = make_normalized_u(n, G=metric, seed=1)
        # 𝒯=0 ⇒ SEC=0 ⇒ R=0
        T = _zeros(n)
        result = agent.execute_focalization(B, sigma, omega, T, u)
        expected = (n - 1) / abs(theta0)
        assert result.caustic_affine_parameter == pytest.approx(expected, rel=1e-10)
        assert result.hawking_penrose_bound == pytest.approx(expected, rel=1e-10)


# =============================================================================
# ANIDACIÓN ESTRUCTURAL Y FLUJO DE TIPOS
# =============================================================================

class TestNestedPhaseAccess:
    def test_phase2_nested_in_phase1(self) -> None:
        assert hasattr(
            Phase1_RaychaudhuriKinematics, "Phase2_StrongEnergyCondition"
        )
        p2 = Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition
        assert p2.__qualname__.startswith("Phase1_RaychaudhuriKinematics")

    def test_phase3_nested_in_phase2(self) -> None:
        p3 = (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse
        )
        assert "Phase2_StrongEnergyCondition" in p3.__qualname__
        assert "Phase1_RaychaudhuriKinematics" in p3.__qualname__

    def test_full_chain_type_flow(
        self,
        phase1: Phase1_RaychaudhuriKinematics,
        phase2: Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition,
        phase3,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        kin: KinematicExpansionData = phase1.compute_kinematics(B, sigma, omega)
        cert: EnergyConditionCertificate = phase2.certify_energy(
            kin, stress_ok, u_vec
        )
        focal: FocalLengthResult = phase3.compute_focal_length(cert)
        assert isinstance(kin, KinematicExpansionData)
        assert isinstance(cert, EnergyConditionCertificate)
        assert isinstance(focal, FocalLengthResult)
        assert cert.kinematics is kin


# =============================================================================
# CASOS LÍMITE Y ESTABILIDAD NUMÉRICA
# =============================================================================

class TestEdgeCasesAndStability:
    def test_dimension_two_minimal(self) -> None:
        n = 2
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n, metric=_eye(n)
        )
        B, sigma, omega = make_consistent_congruence(n=n, theta=-1.0)
        u = make_normalized_u(n)
        T = make_sec_satisfying_stress(n, u, sec_target=0.1)
        result = agent.execute_focalization(B, sigma, omega, T, u)
        assert result.caustic_affine_parameter > 0.0
        assert result.hawking_penrose_bound == pytest.approx(1.0)

    def test_large_negative_theta_short_caustic(self, n: int) -> None:
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n, metric=_eye(n)
        )
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=-1.0e3, shear_scale=0.0
        )
        u = make_normalized_u(n)
        T = _zeros(n)
        result = agent.execute_focalization(B, sigma, omega, T, u)
        assert result.caustic_affine_parameter == pytest.approx(
            (n - 1) / 1.0e3, rel=1e-8
        )

    def test_strong_shear_shortens_caustic(self, n: int) -> None:
        """Mayor σ² (α↑) ⇒ τ_c menor o igual que el caso σ=0."""
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n,
            metric=_eye(n),
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        u = make_normalized_u(n, seed=0)
        T = _zeros(n)

        B0, s0, w0 = make_consistent_congruence(
            n=n, theta=-2.0, shear_scale=0.0, seed=1
        )
        B1, s1, w1 = make_consistent_congruence(
            n=n, theta=-2.0, shear_scale=0.5, seed=1
        )
        r0 = agent.execute_focalization(B0, s0, w0, T, u)
        r1 = agent.execute_focalization(B1, s1, w1, T, u)
        assert r1.caustic_affine_parameter <= r0.caustic_affine_parameter + 1e-9

    def test_reproducibility(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        r_a = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        r_b = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        assert r_a.caustic_affine_parameter == r_b.caustic_affine_parameter
        assert r_a.optimal_focal_length == r_b.optimal_focal_length
        assert r_a.viability_flags == r_b.viability_flags

    def test_illconditioned_metric_still_works(self, n: int) -> None:
        G = make_spd_metric(n, cond=1e4, seed=99)
        agent = RaychaudhuriFocalFibrator(
            spatial_dimensions=n, metric=G
        )
        B, sigma, omega = make_consistent_congruence(n=n, theta=-2.0, seed=5)
        # u normalizado respecto a G
        u = make_normalized_u(n, G=G, seed=6)
        # 𝒯 simple: proyección uu con G(u,u)=1
        T = make_sec_satisfying_stress(n, u, G=G, sec_target=0.2)
        # Nota: con G≠I, Tr_G(σ) del helper euclídeo puede no ser exacto;
        # usamos shear_scale=0 para evitar fallo traceless métrico.
        B0, s0, w0 = make_consistent_congruence(
            n=n, theta=-2.0, shear_scale=0.0, seed=5
        )
        result = agent.execute_focalization(B0, s0, w0, T, u)
        assert result.caustic_affine_parameter > 0.0
        assert math.isfinite(result.optimal_focal_length)

    def test_dtheta_dtau_matches_manual(
        self,
        agent: RaychaudhuriFocalFibrator,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        result = agent.execute_focalization(B, sigma, omega, stress_ok, u_vec)
        theta0 = float(np.trace(B))
        # Reconstruir σ² y R del pipeline es interno; verificamos signo y finitud
        assert result.initial_dtheta_dtau < 0.0
        assert math.isfinite(result.initial_dtheta_dtau)
        # Cota: |dθ/dτ| ≥ θ₀²/(n−1)
        assert abs(result.initial_dtheta_dtau) >= (
            theta0 ** 2
        ) / (n - 1) - 1e-9