# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de pruebas unitarias: PenroseSingularityAgent                          ║
║ Ubicación: tests/unit/omega/test_penrose_singularity_agent.py                ║
║ Versión: 4.0.0-Nested-Hawking-Penrose-Spectral-Topos                         ║
║ Cobertura: Fase 1 → Fase 2 → Fase 3 + endofuntor 𝒫 + teorema de Penrose     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de verificación (GR + teorema de singularidad + integración fibrador):
  • Auditoría energética con θ₀<0, SEC≥0, u normalizado, cota τ_HP.
  • Cáustica vía RaychaudhuriFocalFibrator (real o stub controlado).
  • Veredicto Penrose: τ_c ≤ τ_HP, retículo booleano ALL, vetos ontológicos.
  • Casos patológicos: métrica degenerada, θ₀≥0, SEC violada, u no normalizado,
    tensores ausentes, fuga topológica (τ_c > τ_HP), dims inconsistentes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

# ─── SUT ─────────────────────────────────────────────────────────────────────
from app.omega.penrose_singularity_agent import (
    _MACHINE_EPSILON,
    _SYMMETRY_REL_TOL,
    _U_NORMALIZATION_TOL,
    _PENROSE_MARGIN_TOL,
    _COND_MAX,
    _MIN_SPATIAL_DIM,
    _DEFAULT_SPATIAL_DIM,
    PenroseAgentError,
    PenroseSingularityVetoError,
    EnergyConvergenceViolationError,
    MetricTensorDegeneracyError,
    DimensionMismatchError,
    NormalizationError,
    TensorSymmetryError,
    MissingTensorError,
    FibratorIntegrationError,
    ViabilityLatticeError,
    SingularityCertificateFlags,
    MetricSignatureKind,
    EnergyAuditResult,
    CausticResult,
    SingularityDiagnostics,
    CollapsedAffineState,
    Phase1_HawkingEnergyAuditor,
    PenroseSingularityAgent,
)

# Fibrador (dependencia real)
try:
    from app.omega.raychaudhuri_focal_fibrator import (
        RaychaudhuriFocalFibrator,
        FocalLengthResult,
        FocalViabilityFlags,
        CausticMethod,
        MetricSignature,
    )
    _HAS_FIBRATOR = True
except ImportError:
    _HAS_FIBRATOR = False
    FocalLengthResult = None  # type: ignore[misc, assignment]
    FocalViabilityFlags = None  # type: ignore[misc, assignment]
    CausticMethod = None  # type: ignore[misc, assignment]
    MetricSignature = None  # type: ignore[misc, assignment]


# =============================================================================
# HELPERS DE CONSTRUCCIÓN GEOMÉTRICA
# =============================================================================

def _eye(n: int) -> NDArray[np.float64]:
    return np.eye(n, dtype=np.float64)


def _zeros(n: int) -> NDArray[np.float64]:
    return np.zeros((n, n), dtype=np.float64)


def make_spd_metric(
    n: int, cond: float = 1.0, seed: int = 1
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, float(cond), n, dtype=np.float64)
    G = (Q @ np.diag(eigs) @ Q.T).astype(np.float64)
    return 0.5 * (G + G.T)


def make_traceless_symmetric_shear(
    n: int, scale: float = 0.05, seed: int = 0
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    sigma = 0.5 * (A + A.T) * scale
    tr = float(np.trace(sigma))
    sigma = sigma - (tr / n) * _eye(n)
    return 0.5 * (sigma + sigma.T).astype(np.float64)


def make_consistent_congruence(
    n: int = 4,
    theta: float = -2.0,
    shear_scale: float = 0.05,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""B = (θ/(n−1))I + σ, ω≡0, Tr(σ)=0."""
    sigma = make_traceless_symmetric_shear(n, scale=shear_scale, seed=seed)
    omega = _zeros(n)
    B = (theta / float(n - 1)) * _eye(n) + sigma
    return B.astype(np.float64), sigma, omega


def make_normalized_u(
    n: int,
    G: NDArray[np.float64] | None = None,
    seed: int = 0,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    if G is None:
        G = _eye(n)
    v = rng.standard_normal(n).astype(np.float64)
    gvv = float(v @ G @ v)
    if gvv <= 0.0:
        v = np.ones(n, dtype=np.float64)
        gvv = float(v @ G @ v)
    return (v / math.sqrt(gvv)).astype(np.float64)


def make_sec_satisfying_stress(
    n: int,
    u: NDArray[np.float64],
    sec_target: float = 0.5,
) -> NDArray[np.float64]:
    """𝒯 = a u⊗u con a=2·sec_target ⇒ SEC≈sec_target (G=I, ‖u‖=1)."""
    a = 2.0 * sec_target
    T = a * np.outer(u, u)
    return 0.5 * (T + T.T).astype(np.float64)


def make_sec_violating_stress(
    n: int, u: NDArray[np.float64]
) -> NDArray[np.float64]:
    T = -2.0 * np.outer(u, u)
    return 0.5 * (T + T.T).astype(np.float64)


def make_fake_focal_result(
    *,
    tau_c: float = 1.0,
    f_opt: float = 1.0,
    tau_hp: float = 1.5,
    dtheta: float = -1.0,
    flags_ok: bool = True,
) -> Any:
    """FocalLengthResult sintético (compatible duck-typing con el SUT)."""
    if _HAS_FIBRATOR and FocalLengthResult is not None:
        flags = (
            FocalViabilityFlags.ALL
            if flags_ok and hasattr(FocalViabilityFlags, "ALL")
            else getattr(FocalViabilityFlags, "NONE", 0)
        )
        # Intentar construir el dataclass real
        try:
            return FocalLengthResult(
                optimal_focal_length=f_opt,
                caustic_affine_parameter=tau_c,
                hawking_penrose_bound=tau_hp,
                initial_dtheta_dtau=dtheta,
                caustic_method=CausticMethod.ANALYTICAL_EXACT,
                viability_flags=flags,
                area_collapse_factor=0.0,
            )
        except TypeError:
            pass

    # Fallback duck-typed
    @dataclass
    class _FakeFlags:
        def is_order_unit(self) -> bool:
            return flags_ok

    @dataclass
    class _FakeFocal:
        optimal_focal_length: float = f_opt
        caustic_affine_parameter: float = tau_c
        hawking_penrose_bound: float = tau_hp
        initial_dtheta_dtau: float = dtheta
        caustic_method: Any = type("M", (), {"name": "ANALYTICAL_EXACT"})()
        viability_flags: Any = field(default_factory=_FakeFlags)
        area_collapse_factor: float = 0.0

    return _FakeFocal()


class FakeCategoricalState:
    """Stub de CategoricalState con almacén de tensores."""

    def __init__(self, tensors: Optional[Dict[str, Any]] = None) -> None:
        self._tensors: Dict[str, Any] = dict(tensors or {})

    def get_tensor(self, key: str) -> Any:
        if key not in self._tensors:
            raise KeyError(key)
        return self._tensors[key]

    def update(self, key: str, value: Any) -> "FakeCategoricalState":
        new = FakeCategoricalState(self._tensors)
        new._tensors[key] = value
        return new


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
def stress_ok(n: int, u_vec: NDArray[np.float64]) -> NDArray[np.float64]:
    return make_sec_satisfying_stress(n, u_vec, sec_target=0.5)


@pytest.fixture
def phase1(metric: NDArray[np.float64]) -> Phase1_HawkingEnergyAuditor:
    return Phase1_HawkingEnergyAuditor(metric=metric)


@pytest.fixture
def energy_audit(
    phase1: Phase1_HawkingEnergyAuditor,
    congruence: Tuple[NDArray, NDArray, NDArray],
    stress_ok: NDArray[np.float64],
    u_vec: NDArray[np.float64],
    n: int,
) -> EnergyAuditResult:
    B, _, _ = congruence
    return phase1.audit_energy(B, stress_ok, u_vec, n)


@pytest.fixture
def phase3() -> (
    Phase1_HawkingEnergyAuditor
    .Phase2_RaychaudhuriCausticIntegrator
    .Phase3_PenroseSingularityAuditor
):
    return (
        Phase1_HawkingEnergyAuditor
        .Phase2_RaychaudhuriCausticIntegrator
        .Phase3_PenroseSingularityAuditor(
            margin_tol=_PENROSE_MARGIN_TOL,
            require_fibrator_flags=True,
        )
    )


def _make_caustic_from_audit(
    energy: EnergyAuditResult,
    *,
    tau_c: Optional[float] = None,
    flags_ok: bool = True,
) -> CausticResult:
    """Construye CausticResult sintético a partir de un audit (sin fibrador real)."""
    if tau_c is None:
        tau_c = 0.5 * energy.tau_hp  # holgura generosa
    focal = make_fake_focal_result(
        tau_c=tau_c,
        f_opt=tau_c * 0.8,
        tau_hp=energy.tau_hp,
        dtheta=-1.0,
        flags_ok=flags_ok,
    )
    return CausticResult(
        focal_distance=float(focal.optimal_focal_length),
        tau_caustic=float(tau_c),
        hawking_penrose_bound_fibrator=float(energy.tau_hp),
        initial_dtheta_dtau=-1.0,
        area_collapse_factor=0.0,
        fibrator_method="ANALYTICAL_EXACT",
        fibrator_flags_ok=flags_ok,
        energy_audit=energy,
        fibrator_diagnostics=focal,
    )


@pytest.fixture
def caustic_ok(energy_audit: EnergyAuditResult) -> CausticResult:
    return _make_caustic_from_audit(energy_audit)


@pytest.fixture
def full_tensors(
    congruence: Tuple[NDArray, NDArray, NDArray],
    stress_ok: NDArray[np.float64],
    u_vec: NDArray[np.float64],
) -> Dict[str, NDArray[np.float64]]:
    B, sigma, omega = congruence
    return {
        PenroseSingularityAgent.KEY_VELOCITY: B,
        PenroseSingularityAgent.KEY_SHEAR: sigma,
        PenroseSingularityAgent.KEY_VORTICITY: omega,
        PenroseSingularityAgent.KEY_STRESS: stress_ok,
        PenroseSingularityAgent.KEY_ATTENTION: u_vec,
    }


@pytest.fixture
def agent(n: int, metric: NDArray[np.float64]) -> PenroseSingularityAgent:
    if not _HAS_FIBRATOR:
        pytest.skip("RaychaudhuriFocalFibrator no disponible")
    return PenroseSingularityAgent(
        spatial_dimensions=n,
        metric=metric,
        caustic_method=CausticMethod.ANALYTICAL_EXACT,
    )


# =============================================================================
# FASE 0 – CONSTANTES, EXCEPCIONES, FLAGS, ENUMS
# =============================================================================

class TestConstantsAndExceptions:
    def test_machine_epsilon(self) -> None:
        assert _MACHINE_EPSILON > 0.0
        assert _MACHINE_EPSILON == float(np.finfo(np.float64).eps)

    def test_min_spatial_dim(self) -> None:
        assert _MIN_SPATIAL_DIM >= 2

    def test_default_spatial_dim(self) -> None:
        assert _DEFAULT_SPATIAL_DIM >= _MIN_SPATIAL_DIM

    def test_exception_hierarchy(self) -> None:
        assert issubclass(PenroseSingularityVetoError, PenroseAgentError)
        assert issubclass(EnergyConvergenceViolationError, PenroseAgentError)
        assert issubclass(MetricTensorDegeneracyError, PenroseAgentError)
        assert issubclass(DimensionMismatchError, PenroseAgentError)
        assert issubclass(NormalizationError, PenroseAgentError)
        assert issubclass(TensorSymmetryError, PenroseAgentError)
        assert issubclass(MissingTensorError, PenroseAgentError)
        assert issubclass(FibratorIntegrationError, PenroseAgentError)
        assert issubclass(ViabilityLatticeError, PenroseAgentError)

    def test_certificate_flags_order_unit(self) -> None:
        assert SingularityCertificateFlags.ALL.is_order_unit() is True
        assert SingularityCertificateFlags.NONE.is_order_unit() is False
        partial = (
            SingularityCertificateFlags.NEGATIVE_EXPANSION
            | SingularityCertificateFlags.STRONG_ENERGY_SATISFIED
        )
        assert partial.is_order_unit() is False

    def test_metric_signature_kind(self) -> None:
        names = {s.name for s in MetricSignatureKind}
        assert names == {"RIEMANNIAN", "LORENTZIAN"}

    def test_dto_immutability(self, energy_audit: EnergyAuditResult) -> None:
        with pytest.raises(Exception):
            energy_audit.theta_0 = 0.0  # type: ignore[misc]


# =============================================================================
# FASE 1 – MÉTRICA Y DIMENSIONES
# =============================================================================

class TestPhase1Metric:
    def test_identity_metric_ok(self, n: int) -> None:
        p1 = Phase1_HawkingEnergyAuditor(metric=_eye(n))
        assert p1._lam_min_G == pytest.approx(1.0)
        assert p1._lam_max_G == pytest.approx(1.0)
        assert p1._cond_G == pytest.approx(1.0)

    def test_spd_known_cond(self, n: int) -> None:
        G = make_spd_metric(n, cond=16.0, seed=3)
        p1 = Phase1_HawkingEnergyAuditor(metric=G)
        assert p1._cond_G == pytest.approx(16.0, rel=1e-5)

    def test_asymmetric_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[0, 1] = 0.7
        with pytest.raises(MetricTensorDegeneracyError, match="simétrica"):
            Phase1_HawkingEnergyAuditor(metric=G)

    def test_indefinite_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[-1, -1] = -1.0
        with pytest.raises(MetricTensorDegeneracyError):
            Phase1_HawkingEnergyAuditor(metric=G)

    def test_singular_metric_raises(self, n: int) -> None:
        G = _eye(n)
        G[-1, -1] = 0.0
        with pytest.raises(MetricTensorDegeneracyError):
            Phase1_HawkingEnergyAuditor(metric=G)

    def test_dim_one_raises(self) -> None:
        with pytest.raises(MetricTensorDegeneracyError):
            Phase1_HawkingEnergyAuditor(metric=_eye(1))

    def test_non_square_raises(self) -> None:
        with pytest.raises(MetricTensorDegeneracyError):
            Phase1_HawkingEnergyAuditor(
                metric=np.ones((3, 4), dtype=np.float64)
            )

    def test_u_norm_tol_rejects_nonpositive(self, n: int) -> None:
        with pytest.raises(ValueError, match="u_norm_tol"):
            Phase1_HawkingEnergyAuditor(metric=_eye(n), u_norm_tol=0.0)


class TestPhase1Dimensions:
    def test_valid_dimensions(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        assert phase1._assert_dimensions(B, stress_ok, u_vec, n) == n

    def test_B_non_square_raises(
        self, phase1: Phase1_HawkingEnergyAuditor, n: int, u_vec
    ) -> None:
        with pytest.raises(DimensionMismatchError, match="cuadrado"):
            phase1._assert_dimensions(
                np.ones((n, n + 1)), _eye(n), u_vec, n
            )

    def test_spatial_dim_mismatch(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        with pytest.raises(DimensionMismatchError, match="spatial_dim"):
            phase1._assert_dimensions(B, stress_ok, u_vec, n + 1)

    def test_stress_shape_mismatch(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        with pytest.raises(DimensionMismatchError, match="𝒯"):
            phase1._assert_dimensions(B, _eye(n + 1), u_vec, n)

    def test_u_shape_mismatch(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        with pytest.raises(DimensionMismatchError, match="u"):
            phase1._assert_dimensions(
                B, stress_ok, np.ones(n + 1, dtype=np.float64), n
            )


# =============================================================================
# FASE 1 – EXPANSIÓN, SEC, NORMALIZACIÓN, HP
# =============================================================================

class TestPhase1Expansion:
    def test_negative_theta_ok(
        self, phase1: Phase1_HawkingEnergyAuditor, n: int
    ) -> None:
        B = (-2.0 / (n - 1)) * _eye(n)
        theta = phase1._compute_initial_expansion(B)
        assert theta == pytest.approx(-2.0)

    def test_nonnegative_theta_raises(
        self, phase1: Phase1_HawkingEnergyAuditor, n: int
    ) -> None:
        with pytest.raises(EnergyConvergenceViolationError, match="no convergente"):
            phase1._compute_initial_expansion(_eye(n))
        with pytest.raises(EnergyConvergenceViolationError):
            phase1._compute_initial_expansion(_zeros(n))


class TestPhase1Normalization:
    def test_normalized_u_passes(
        self, phase1: Phase1_HawkingEnergyAuditor, u_vec: NDArray[np.float64]
    ) -> None:
        g_uu = phase1._normalize_check_u(u_vec)
        assert g_uu == pytest.approx(1.0, abs=_U_NORMALIZATION_TOL)

    def test_unnormalized_raises(
        self, phase1: Phase1_HawkingEnergyAuditor, n: int
    ) -> None:
        with pytest.raises(NormalizationError, match="no normalizado"):
            phase1._normalize_check_u(3.0 * np.ones(n, dtype=np.float64))

    def test_zero_u_raises(
        self, phase1: Phase1_HawkingEnergyAuditor, n: int
    ) -> None:
        with pytest.raises(NormalizationError):
            phase1._normalize_check_u(np.zeros(n, dtype=np.float64))


class TestPhase1SEC:
    def test_sec_satisfying(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        u_vec: NDArray[np.float64],
        stress_ok: NDArray[np.float64],
    ) -> None:
        ricci, sec, tr = phase1._compute_ricci_contraction(stress_ok, u_vec)
        assert sec >= -_MACHINE_EPSILON
        assert ricci == pytest.approx(max(0.0, sec))
        assert ricci >= 0.0

    def test_sec_violation_raises(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        T_bad = make_sec_violating_stress(n, u_vec)
        with pytest.raises(EnergyConvergenceViolationError, match="SEC"):
            phase1._compute_ricci_contraction(T_bad, u_vec)

    def test_asymmetric_stress_raises(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        n: int,
    ) -> None:
        T = _zeros(n)
        T[0, 1] = 1.0
        with pytest.raises(TensorSymmetryError, match="simétrico"):
            phase1._check_stress_symmetry(T)

    def test_zero_stress_sec_zero(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        ricci, sec, tr = phase1._compute_ricci_contraction(_zeros(n), u_vec)
        assert sec == pytest.approx(0.0, abs=1e-14)
        assert ricci == pytest.approx(0.0, abs=1e-14)


class TestPhase1HawkingPenroseBound:
    def test_formula(self, n: int) -> None:
        tau = Phase1_HawkingEnergyAuditor.hawking_penrose_bound(-2.0, n)
        assert tau == pytest.approx((n - 1) / 2.0)

    def test_rejects_nonnegative_theta(self, n: int) -> None:
        with pytest.raises(EnergyConvergenceViolationError):
            Phase1_HawkingEnergyAuditor.hawking_penrose_bound(0.0, n)
        with pytest.raises(EnergyConvergenceViolationError):
            Phase1_HawkingEnergyAuditor.hawking_penrose_bound(1.0, n)

    def test_rejects_bad_dim(self) -> None:
        with pytest.raises(DimensionMismatchError):
            Phase1_HawkingEnergyAuditor.hawking_penrose_bound(-1.0, 1)

    def test_more_negative_shorter(self, n: int) -> None:
        t1 = Phase1_HawkingEnergyAuditor.hawking_penrose_bound(-1.0, n)
        t2 = Phase1_HawkingEnergyAuditor.hawking_penrose_bound(-5.0, n)
        assert t2 < t1


class TestPhase1AuditEnergy:
    def test_happy_path(self, energy_audit: EnergyAuditResult, n: int) -> None:
        assert isinstance(energy_audit, EnergyAuditResult)
        assert energy_audit.theta_0 < 0.0
        assert energy_audit.ricci_contraction >= 0.0
        assert energy_audit.tau_hp > 0.0
        assert energy_audit.spatial_dim == n
        assert energy_audit.metric_cond >= 1.0 - 1e-12
        assert energy_audit.op_norm_B >= 0.0
        assert energy_audit.frobenius_B >= 0.0
        assert energy_audit.u_normalization == pytest.approx(1.0, abs=1e-8)

    def test_tau_hp_consistency(
        self, energy_audit: EnergyAuditResult
    ) -> None:
        expected = (energy_audit.spatial_dim - 1) / abs(energy_audit.theta_0)
        assert energy_audit.tau_hp == pytest.approx(expected, rel=1e-12)

    def test_theta_equals_trace(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        audit = phase1.audit_energy(B, stress_ok, u_vec, n)
        assert audit.theta_0 == pytest.approx(float(np.trace(B)))

    def test_spectral_inequality_B(
        self, energy_audit: EnergyAuditResult
    ) -> None:
        assert energy_audit.op_norm_B <= energy_audit.frobenius_B + 1e-10

    def test_rejects_positive_theta(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = make_consistent_congruence(n=n, theta=+1.0)
        with pytest.raises(EnergyConvergenceViolationError):
            phase1.audit_energy(B, stress_ok, u_vec, n)

    def test_rejects_sec_violation(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        T_bad = make_sec_violating_stress(n, u_vec)
        with pytest.raises(EnergyConvergenceViolationError):
            phase1.audit_energy(B, T_bad, u_vec, n)

    def test_rejects_unnormalized_u(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        with pytest.raises(NormalizationError):
            phase1.audit_energy(
                B, stress_ok, 5.0 * np.ones(n, dtype=np.float64), n
            )

    @pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 7])
    def test_parametrized_dimensions(self, n_dim: int) -> None:
        p1 = Phase1_HawkingEnergyAuditor(metric=_eye(n_dim))
        B, _, _ = make_consistent_congruence(n=n_dim, theta=-1.5, seed=n_dim)
        u = make_normalized_u(n_dim, seed=n_dim + 1)
        T = make_sec_satisfying_stress(n_dim, u, sec_target=0.2)
        audit = p1.audit_energy(B, T, u, n_dim)
        assert audit.spatial_dim == n_dim
        assert audit.theta_0 == pytest.approx(-1.5, abs=1e-10)
        assert audit.tau_hp == pytest.approx((n_dim - 1) / 1.5, rel=1e-10)


# =============================================================================
# FASE 2 – CÁUSTICA (con mock del fibrador y/o integración real)
# =============================================================================

class TestPhase2CausticWithMock:
    """Fase 2 aislada con fibrador mockeado."""

    def test_compute_caustic_happy_path(
        self,
        energy_audit: EnergyAuditResult,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
        metric: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        fake = make_fake_focal_result(
            tau_c=0.5 * energy_audit.tau_hp,
            f_opt=0.4,
            tau_hp=energy_audit.tau_hp,
            flags_ok=True,
        )

        with patch.object(
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator,
            "__init__",
            lambda self, *a, **k: None,
        ):
            p2 = object.__new__(
                Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator
            )
            p2._dim = n
            p2._caustic_method = "ANALYTICAL_EXACT"
            p2._fibrator = MagicMock(return_value=fake)

            result = p2.compute_caustic(
                energy_audit, B, sigma, omega, stress_ok, u_vec
            )

        assert isinstance(result, CausticResult)
        assert result.tau_caustic == pytest.approx(0.5 * energy_audit.tau_hp)
        assert result.energy_audit is energy_audit
        assert result.fibrator_flags_ok is True
        assert result.focal_distance == pytest.approx(0.4)

    def test_dimension_mismatch_shear(
        self,
        energy_audit: EnergyAuditResult,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, omega = congruence
        p2 = object.__new__(
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator
        )
        p2._dim = n
        p2._caustic_method = "ANALYTICAL_EXACT"
        p2._fibrator = MagicMock()

        with pytest.raises(DimensionMismatchError, match="σ"):
            p2.compute_caustic(
                energy_audit, B, _zeros(n + 1), omega, stress_ok, u_vec
            )

    def test_fibrator_exception_wrapped(
        self,
        energy_audit: EnergyAuditResult,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        p2 = object.__new__(
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator
        )
        p2._dim = n
        p2._caustic_method = "ANALYTICAL_EXACT"
        p2._fibrator = MagicMock(
            side_effect=RuntimeError("fibrator boom")
        )

        with pytest.raises(FibratorIntegrationError, match="inesperado"):
            p2.compute_caustic(
                energy_audit, B, sigma, omega, stress_ok, u_vec
            )


@pytest.mark.skipif(not _HAS_FIBRATOR, reason="fibrador no instalado")
class TestPhase2CausticRealFibrator:
    def test_integration_with_real_fibrator(
        self,
        energy_audit: EnergyAuditResult,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
        metric: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        p2 = Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator(
            spatial_dim=n,
            metric=metric,
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        result = p2.compute_caustic(
            energy_audit, B, sigma, omega, stress_ok, u_vec
        )
        assert result.tau_caustic > 0.0
        assert math.isfinite(result.focal_distance)
        assert result.energy_audit is energy_audit
        # τ_c no debe superar τ_HP del audit (teorema)
        assert result.tau_caustic <= energy_audit.tau_hp * (1.0 + 1e-5)

    def test_constructor_rejects_dim_one(self) -> None:
        with pytest.raises(ValueError, match="spatial_dim"):
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator(
                spatial_dim=1
            )


# =============================================================================
# FASE 3 – VEREDICTO DE PENROSE
# =============================================================================

class TestPhase3RelativeMargin:
    def test_positive_margin(self, phase3) -> None:
        m = phase3._relative_margin(tau_c=1.0, tau_hp=2.0)
        assert m == pytest.approx(0.5)

    def test_zero_margin(self, phase3) -> None:
        m = phase3._relative_margin(tau_c=2.0, tau_hp=2.0)
        assert m == pytest.approx(0.0)

    def test_negative_margin_violation(self, phase3) -> None:
        m = phase3._relative_margin(tau_c=3.0, tau_hp=2.0)
        assert m < 0.0


class TestPhase3VerifyAndCertify:
    def test_happy_path(
        self, phase3, caustic_ok: CausticResult
    ) -> None:
        collapsed = phase3.verify_and_certify(caustic_ok)
        assert isinstance(collapsed, CollapsedAffineState)
        assert collapsed.diagnostics.is_singularity_inevitable is True
        assert collapsed.diagnostics.certificate_flags.is_order_unit()
        assert collapsed.diagnostics.relative_margin >= -_PENROSE_MARGIN_TOL
        assert collapsed.focal_distance == caustic_ok.focal_distance
        assert collapsed.caustic is caustic_ok

    def test_tau_c_equals_tau_hp_accepted(
        self, phase3, energy_audit: EnergyAuditResult
    ) -> None:
        """τ_c = τ_HP está en el borde y debe aceptarse."""
        caustic = _make_caustic_from_audit(
            energy_audit, tau_c=energy_audit.tau_hp
        )
        collapsed = phase3.verify_and_certify(caustic)
        assert collapsed.diagnostics.is_singularity_inevitable is True
        assert collapsed.diagnostics.relative_margin == pytest.approx(
            0.0, abs=1e-12
        )

    def test_tau_c_exceeds_tau_hp_raises(
        self, phase3, energy_audit: EnergyAuditResult
    ) -> None:
        caustic = _make_caustic_from_audit(
            energy_audit, tau_c=energy_audit.tau_hp * 1.5
        )
        with pytest.raises(PenroseSingularityVetoError, match="Fuga topológica"):
            phase3.verify_and_certify(caustic)

    def test_invalid_tau_c_negative_raises(
        self, phase3, energy_audit: EnergyAuditResult
    ) -> None:
        caustic = _make_caustic_from_audit(energy_audit, tau_c=-1.0)
        with pytest.raises(PenroseSingularityVetoError, match="inválido"):
            phase3.verify_and_certify(caustic)

    def test_invalid_tau_c_nan_raises(
        self, phase3, energy_audit: EnergyAuditResult
    ) -> None:
        caustic = _make_caustic_from_audit(energy_audit, tau_c=float("nan"))
        with pytest.raises(PenroseSingularityVetoError):
            phase3.verify_and_certify(caustic)

    def test_fibrator_flags_required(
        self, energy_audit: EnergyAuditResult
    ) -> None:
        p3 = (
            Phase1_HawkingEnergyAuditor
            .Phase2_RaychaudhuriCausticIntegrator
            .Phase3_PenroseSingularityAuditor(
                require_fibrator_flags=True
            )
        )
        caustic = _make_caustic_from_audit(
            energy_audit, flags_ok=False
        )
        with pytest.raises(ViabilityLatticeError, match="Retículo"):
            p3.verify_and_certify(caustic)

    def test_fibrator_flags_optional_when_disabled(
        self, energy_audit: EnergyAuditResult
    ) -> None:
        p3 = (
            Phase1_HawkingEnergyAuditor
            .Phase2_RaychaudhuriCausticIntegrator
            .Phase3_PenroseSingularityAuditor(
                require_fibrator_flags=False
            )
        )
        caustic = _make_caustic_from_audit(
            energy_audit, flags_ok=False
        )
        collapsed = p3.verify_and_certify(caustic)
        assert collapsed.diagnostics.is_singularity_inevitable is True

    def test_diagnostics_fields(
        self, phase3, caustic_ok: CausticResult, energy_audit: EnergyAuditResult
    ) -> None:
        collapsed = phase3.verify_and_certify(caustic_ok)
        d = collapsed.diagnostics
        assert d.initial_expansion_scalar == energy_audit.theta_0
        assert d.ricci_contraction == energy_audit.ricci_contraction
        assert d.max_theoretical_affine_limit == energy_audit.tau_hp
        assert d.actual_caustic_parameter == caustic_ok.tau_caustic
        assert d.spatial_dim == energy_audit.spatial_dim
        assert (
            SingularityCertificateFlags.CAUSTIC_WITHIN_HP_BOUND
            in d.certificate_flags
        )
        assert (
            SingularityCertificateFlags.NEGATIVE_EXPANSION
            in d.certificate_flags
        )

    def test_margin_tol_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="margin_tol"):
            (
                Phase1_HawkingEnergyAuditor
                .Phase2_RaychaudhuriCausticIntegrator
                .Phase3_PenroseSingularityAuditor(margin_tol=-1e-6)
            )

    def test_certificate_flags_all_present(
        self, phase3, caustic_ok: CausticResult
    ) -> None:
        collapsed = phase3.verify_and_certify(caustic_ok)
        f = collapsed.diagnostics.certificate_flags
        for flag in (
            SingularityCertificateFlags.NEGATIVE_EXPANSION,
            SingularityCertificateFlags.STRONG_ENERGY_SATISFIED,
            SingularityCertificateFlags.U_NORMALIZED,
            SingularityCertificateFlags.CAUSTIC_WITHIN_HP_BOUND,
            SingularityCertificateFlags.FIBRATOR_VIABLE,
            SingularityCertificateFlags.METRIC_RIEMANNIAN,
        ):
            assert flag in f


# =============================================================================
# ENDOFUNTORE 𝒫 – INTEGRACIÓN END-TO-END
# =============================================================================

@pytest.mark.skipif(not _HAS_FIBRATOR, reason="fibrador no instalado")
class TestPenroseSingularityAgentE2E:
    def test_execute_singularity_audit_happy_path(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        collapsed = agent.execute_singularity_audit(
            B, sigma, omega, stress_ok, u_vec
        )
        assert isinstance(collapsed, CollapsedAffineState)
        assert collapsed.diagnostics.is_singularity_inevitable is True
        assert collapsed.diagnostics.certificate_flags.is_order_unit()
        assert collapsed.diagnostics.actual_caustic_parameter <= (
            collapsed.diagnostics.max_theoretical_affine_limit
            * (1.0 + _PENROSE_MARGIN_TOL + 1e-9)
        )
        assert collapsed.focal_distance >= 0.0
        assert math.isfinite(collapsed.focal_distance)

    def test_call_on_categorical_state(
        self,
        agent: PenroseSingularityAgent,
        full_tensors: Dict[str, NDArray[np.float64]],
    ) -> None:
        state = FakeCategoricalState(full_tensors)
        new_state = agent(state)
        assert isinstance(new_state, FakeCategoricalState)
        assert PenroseSingularityAgent.KEY_FOCAL_DISTANCE in new_state._tensors
        f_opt = new_state._tensors[PenroseSingularityAgent.KEY_FOCAL_DISTANCE]
        assert math.isfinite(float(f_opt))
        # Diagnósticos adjuntos si el estado lo soporta
        diag = new_state._tensors.get(
            PenroseSingularityAgent.KEY_SINGULARITY_DIAG
        )
        if diag is not None:
            assert isinstance(diag, SingularityDiagnostics)
            assert diag.is_singularity_inevitable is True

    def test_missing_tensor_raises(
        self, agent: PenroseSingularityAgent
    ) -> None:
        state = FakeCategoricalState({})
        with pytest.raises(MissingTensorError, match="ausentes"):
            agent(state)

    def test_partial_tensors_raises(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
    ) -> None:
        B, sigma, _ = congruence
        state = FakeCategoricalState({
            PenroseSingularityAgent.KEY_VELOCITY: B,
            PenroseSingularityAgent.KEY_SHEAR: sigma,
            # faltan vorticity, stress, attention
        })
        with pytest.raises(MissingTensorError):
            agent(state)

    def test_rejects_positive_theta(
        self,
        agent: PenroseSingularityAgent,
        n: int,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = make_consistent_congruence(n=n, theta=+1.0)
        with pytest.raises(EnergyConvergenceViolationError):
            agent.execute_singularity_audit(B, sigma, omega, stress_ok, u_vec)

    def test_rejects_sec_violation(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        T_bad = make_sec_violating_stress(n, u_vec)
        with pytest.raises(EnergyConvergenceViolationError):
            agent.execute_singularity_audit(B, sigma, omega, T_bad, u_vec)

    def test_rejects_unnormalized_u(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        with pytest.raises(NormalizationError):
            agent.execute_singularity_audit(
                B, sigma, omega, stress_ok,
                4.0 * np.ones(n, dtype=np.float64),
            )

    def test_rejects_vorticity_via_fibrator(
        self,
        agent: PenroseSingularityAgent,
        n: int,
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, _ = make_consistent_congruence(n=n, theta=-2.0)
        omega = _zeros(n)
        omega[0, 1] = 1.0
        omega[1, 0] = -1.0
        with pytest.raises(FibratorIntegrationError):
            agent.execute_singularity_audit(
                B, sigma, omega, stress_ok, u_vec
            )

    def test_hp_bound_consistency(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, sigma, omega = congruence
        collapsed = agent.execute_singularity_audit(
            B, sigma, omega, stress_ok, u_vec
        )
        theta0 = float(np.trace(B))
        expected_hp = (n - 1) / abs(theta0)
        assert collapsed.diagnostics.max_theoretical_affine_limit == pytest.approx(
            expected_hp, rel=1e-10
        )

    def test_zero_shear_alpha_zero(
        self, n: int, metric: NDArray[np.float64]
    ) -> None:
        """σ=0, 𝒯=0 ⇒ τ_c = τ_HP exacto; Penrose certifica en el borde."""
        agent = PenroseSingularityAgent(
            spatial_dimensions=n,
            metric=metric,
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        theta0 = -3.0
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=theta0, shear_scale=0.0, seed=0
        )
        u = make_normalized_u(n, G=metric, seed=1)
        T = _zeros(n)
        collapsed = agent.execute_singularity_audit(B, sigma, omega, T, u)
        expected = (n - 1) / abs(theta0)
        assert collapsed.diagnostics.actual_caustic_parameter == pytest.approx(
            expected, rel=1e-8
        )
        assert collapsed.diagnostics.max_theoretical_affine_limit == pytest.approx(
            expected, rel=1e-10
        )

    def test_describe_pipeline_keys(
        self, agent: PenroseSingularityAgent
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
            "axiom_4",
            "caustic_method",
            "signature",
            "spatial_dim",
            "margin_tol",
        ):
            assert key in meta
        assert "PenroseSingularityAgent" in meta["functor"] or "𝒫" in meta["functor"]

    def test_constructor_rejects_dim_one(self) -> None:
        with pytest.raises(ValueError, match="spatial_dimensions"):
            PenroseSingularityAgent(spatial_dimensions=1)

    def test_numerical_method(
        self,
        n: int,
        metric: NDArray[np.float64],
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        agent_num = PenroseSingularityAgent(
            spatial_dimensions=n,
            metric=metric,
            use_numerical_integration=True,
        )
        B, sigma, omega = congruence
        collapsed = agent_num.execute_singularity_audit(
            B, sigma, omega, stress_ok, u_vec
        )
        assert collapsed.diagnostics.is_singularity_inevitable is True
        assert collapsed.diagnostics.actual_caustic_parameter > 0.0

    @pytest.mark.parametrize("n_dim", [2, 3, 4, 5])
    def test_parametrized_end_to_end(self, n_dim: int) -> None:
        G = _eye(n_dim)
        agent = PenroseSingularityAgent(
            spatial_dimensions=n_dim,
            metric=G,
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        B, sigma, omega = make_consistent_congruence(
            n=n_dim, theta=-2.5, shear_scale=0.02, seed=n_dim
        )
        u = make_normalized_u(n_dim, G=G, seed=n_dim + 1)
        T = make_sec_satisfying_stress(n_dim, u, sec_target=0.3)
        collapsed = agent.execute_singularity_audit(B, sigma, omega, T, u)
        assert collapsed.diagnostics.certificate_flags.is_order_unit()
        assert collapsed.diagnostics.spatial_dim == n_dim
        assert collapsed.diagnostics.actual_caustic_parameter <= (
            collapsed.diagnostics.max_theoretical_affine_limit * 1.001
        )

    def test_reproducibility(
        self,
        agent: PenroseSingularityAgent,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
    ) -> None:
        B, sigma, omega = congruence
        a = agent.execute_singularity_audit(B, sigma, omega, stress_ok, u_vec)
        b = agent.execute_singularity_audit(B, sigma, omega, stress_ok, u_vec)
        assert a.diagnostics.actual_caustic_parameter == (
            b.diagnostics.actual_caustic_parameter
        )
        assert a.focal_distance == b.focal_distance
        assert a.diagnostics.certificate_flags == b.diagnostics.certificate_flags


# =============================================================================
# ANIDACIÓN ESTRUCTURAL Y FLUJO DE TIPOS
# =============================================================================

class TestNestedPhaseAccess:
    def test_phase2_nested_in_phase1(self) -> None:
        assert hasattr(
            Phase1_HawkingEnergyAuditor, "Phase2_RaychaudhuriCausticIntegrator"
        )
        p2 = Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator
        assert p2.__qualname__.startswith("Phase1_HawkingEnergyAuditor")

    def test_phase3_nested_in_phase2(self) -> None:
        p3 = (
            Phase1_HawkingEnergyAuditor
            .Phase2_RaychaudhuriCausticIntegrator
            .Phase3_PenroseSingularityAuditor
        )
        assert "Phase2_RaychaudhuriCausticIntegrator" in p3.__qualname__
        assert "Phase1_HawkingEnergyAuditor" in p3.__qualname__

    def test_full_chain_with_synthetic_caustic(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        phase3,
        congruence: Tuple[NDArray, NDArray, NDArray],
        stress_ok: NDArray[np.float64],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        """Cadena Fase1 → (cáustica sintética) → Fase3 sin fibrador real."""
        B, _, _ = congruence
        energy: EnergyAuditResult = phase1.audit_energy(
            B, stress_ok, u_vec, n
        )
        caustic: CausticResult = _make_caustic_from_audit(energy)
        collapsed: CollapsedAffineState = phase3.verify_and_certify(caustic)
        assert isinstance(energy, EnergyAuditResult)
        assert isinstance(caustic, CausticResult)
        assert isinstance(collapsed, CollapsedAffineState)
        assert caustic.energy_audit is energy
        assert collapsed.caustic is caustic


# =============================================================================
# CASOS LÍMITE Y ESTABILIDAD
# =============================================================================

class TestEdgeCasesAndStability:
    def test_dimension_two_minimal_audit(self) -> None:
        n = 2
        p1 = Phase1_HawkingEnergyAuditor(metric=_eye(n))
        B, _, _ = make_consistent_congruence(n=n, theta=-1.0)
        u = make_normalized_u(n)
        T = make_sec_satisfying_stress(n, u, sec_target=0.1)
        audit = p1.audit_energy(B, T, u, n)
        assert audit.tau_hp == pytest.approx(1.0)
        assert audit.spatial_dim == 2

    @pytest.mark.skipif(not _HAS_FIBRATOR, reason="fibrador no instalado")
    def test_large_negative_theta(self, n: int) -> None:
        agent = PenroseSingularityAgent(
            spatial_dimensions=n, metric=_eye(n)
        )
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=-1.0e3, shear_scale=0.0
        )
        u = make_normalized_u(n)
        T = _zeros(n)
        collapsed = agent.execute_singularity_audit(B, sigma, omega, T, u)
        assert collapsed.diagnostics.actual_caustic_parameter == pytest.approx(
            (n - 1) / 1.0e3, rel=1e-6
        )

    @pytest.mark.skipif(not _HAS_FIBRATOR, reason="fibrador no instalado")
    def test_strong_shear_still_within_hp(self, n: int) -> None:
        agent = PenroseSingularityAgent(
            spatial_dimensions=n,
            metric=_eye(n),
            caustic_method=CausticMethod.ANALYTICAL_EXACT,
        )
        B, sigma, omega = make_consistent_congruence(
            n=n, theta=-2.0, shear_scale=0.4, seed=9
        )
        u = make_normalized_u(n, seed=10)
        T = make_sec_satisfying_stress(n, u, sec_target=0.5)
        collapsed = agent.execute_singularity_audit(B, sigma, omega, T, u)
        d = collapsed.diagnostics
        assert d.actual_caustic_parameter <= d.max_theoretical_affine_limit * 1.001
        assert d.relative_margin >= -1e-6

    def test_audit_with_zero_stress(
        self,
        phase1: Phase1_HawkingEnergyAuditor,
        congruence: Tuple[NDArray, NDArray, NDArray],
        u_vec: NDArray[np.float64],
        n: int,
    ) -> None:
        B, _, _ = congruence
        audit = phase1.audit_energy(B, _zeros(n), u_vec, n)
        assert audit.ricci_contraction == pytest.approx(0.0, abs=1e-14)
        assert audit.sec_value == pytest.approx(0.0, abs=1e-14)

    def test_margin_formula_in_diagnostics(
        self, phase3, energy_audit: EnergyAuditResult
    ) -> None:
        tau_c = 0.25 * energy_audit.tau_hp
        caustic = _make_caustic_from_audit(energy_audit, tau_c=tau_c)
        collapsed = phase3.verify_and_certify(caustic)
        expected_margin = (energy_audit.tau_hp - tau_c) / energy_audit.tau_hp
        assert collapsed.diagnostics.relative_margin == pytest.approx(
            expected_margin, rel=1e-12
        )

    @pytest.mark.skipif(not _HAS_FIBRATOR, reason="fibrador no instalado")
    def test_state_preserves_other_tensors(
        self,
        agent: PenroseSingularityAgent,
        full_tensors: Dict[str, NDArray[np.float64]],
    ) -> None:
        tensors = dict(full_tensors)
        tensors["custom_payload"] = np.array([1.0, 2.0, 3.0])
        state = FakeCategoricalState(tensors)
        new_state = agent(state)
        assert "custom_payload" in new_state._tensors
        np.testing.assert_array_equal(
            new_state._tensors["custom_payload"],
            np.array([1.0, 2.0, 3.0]),
        )