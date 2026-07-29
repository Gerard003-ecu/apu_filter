# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_levi_civita_agent.py                                           ║
║ Objetivo: Validación rigurosa de LeviCivitaConnectionAgent v8.0.0            ║
║           (fases anidadas, axiomas LC, geodésica G-norma, transporte)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Organización por continuación formal
------------------------------------
.. code-block:: text

    TestPhase0_ExceptionsAndDTO
        │
    TestPhase1_ChristoffelEngine
        │  build_christoffel → ChristoffelData
        ▼
    TestPhase2_TorsionFreeConnection
        │  verify_axioms(ChristoffelData) → ConnectionDiagnostics
        ▼
    TestPhase3_GeodesicOrchestrator
        │  enforce_geodesic_flow / geodesic_rhs / ♭♯ / parallel_transport
        ▼
    TestCategoricalContract / TestEcosystemIntegration / TestNumericalStress

Ejecución
---------
    pytest app/omega/tests/test_levi_civita_agent.py -v --tb=short
    pytest app/omega/tests/test_levi_civita_agent.py -k "Phase1 or Phase2" -v
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# SUT (System Under Test)
# ---------------------------------------------------------------------------
from app.omega.levi_civita_agent import (
    BianchiIdentityError,
    CallableMetricDerivative,
    ChristoffelData,
    ChristoffelInstabilityError,
    ConnectionDiagnostics,
    CotangentVector,
    DimensionalMismatchError,
    GeodesicDeviationError,
    GeodesicStepReport,
    LeviCivitaConnectionAgent,
    LeviCivitaParameterError,
    MetricCompatibilityError,
    MetricDerivativeProvider,
    StaticMetricDerivative,
    TangentVector,
    TopologicalTorsionError,
    _BIANCHI_TOLERANCE,
    _CHRISTOFFEL_FINITE_TOL,
    _DEFAULT_DT,
    _DT_MIN,
    _GEODESIC_NORM_DRIFT_TOL,
    _MACHINE_EPS,
    _METRIC_COMPAT_TOLERANCE,
    _TORSION_TOLERANCE,
)


# =============================================================================
# FIXTURES Y UTILIDADES NUMÉRICAS
# =============================================================================

_RTOL = 1.0e-12
_ATOL = 1.0e-14
_SOFT_ATOL = 1.0e-10


def _sym(A: NDArray[np.float64]) -> NDArray[np.float64]:
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)


def _spd_from_eig(
    eigenvalues: NDArray[np.float64],
    seed: int = 0,
) -> NDArray[np.float64]:
    """Construye G = Q diag(λ) Qᵀ SPD con autovalores dados."""
    n = int(eigenvalues.shape[0])
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q, _ = la.qr(A)
    return _sym(Q @ np.diag(eigenvalues) @ Q.T)


def _g_norm(G: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return math.sqrt(max(float(v @ G @ v), 0.0))


def _assert_allclose(
    a: Any,
    b: Any,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
    msg: str = "",
) -> None:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


@pytest.fixture
def eye4() -> NDArray[np.float64]:
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def eye3() -> NDArray[np.float64]:
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def diag_metric() -> NDArray[np.float64]:
    """Métrica diagonal anisotrópica bien condicionada."""
    return np.diag([1.0, 2.0, 0.5, 4.0]).astype(np.float64)


@pytest.fixture
def ill_conditioned_spd() -> NDArray[np.float64]:
    """SPD con κ ≈ 1e8 (estres de precondicionador)."""
    return _spd_from_eig(
        np.array([1.0, 1.0e2, 1.0e4, 1.0e8], dtype=np.float64),
        seed=42,
    )


@pytest.fixture
def random_spd4() -> NDArray[np.float64]:
    return _spd_from_eig(
        np.array([0.3, 1.1, 2.7, 5.0], dtype=np.float64),
        seed=7,
    )


@pytest.fixture
def agent_static_eye(eye4: NDArray[np.float64]) -> LeviCivitaConnectionAgent:
    return LeviCivitaConnectionAgent(
        metric_tensor=eye4,
        enforce_norm_conservation=True,
        parallel_transport_order=2,
    )


@pytest.fixture
def agent_diag(
    diag_metric: NDArray[np.float64],
) -> LeviCivitaConnectionAgent:
    return LeviCivitaConnectionAgent(
        metric_tensor=diag_metric,
        enforce_norm_conservation=True,
        parallel_transport_order=2,
    )


@pytest.fixture
def tangent_unit(eye4: NDArray[np.float64]) -> TangentVector:
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return TangentVector(coordinates=v)


@pytest.fixture
def tangent_generic() -> TangentVector:
    v = np.array([0.5, -0.3, 0.8, 0.1], dtype=np.float64)
    return TangentVector(coordinates=v)


# =============================================================================
# FASE 0 — EXCEPCIONES, DTOs, PARÁMETROS
# =============================================================================


class TestPhase0_ExceptionsAndDTO:
    """Validación de contratos de tipos, DTOs y parámetros de control."""

    def test_invalid_torsion_tolerance_raises(self, eye4: NDArray[np.float64]) -> None:
        with pytest.raises(LeviCivitaParameterError, match="torsion_tolerance"):
            LeviCivitaConnectionAgent(metric_tensor=eye4, torsion_tolerance=0.0)
        with pytest.raises(LeviCivitaParameterError, match="torsion_tolerance"):
            LeviCivitaConnectionAgent(metric_tensor=eye4, torsion_tolerance=-1e-9)

    def test_invalid_metric_compat_tolerance_raises(
        self, eye4: NDArray[np.float64]
    ) -> None:
        with pytest.raises(LeviCivitaParameterError, match="metric_compat"):
            LeviCivitaConnectionAgent(
                metric_tensor=eye4, metric_compat_tolerance=-1.0
            )

    def test_invalid_bianchi_tolerance_raises(
        self, eye4: NDArray[np.float64]
    ) -> None:
        with pytest.raises(LeviCivitaParameterError, match="bianchi"):
            LeviCivitaConnectionAgent(metric_tensor=eye4, bianchi_tolerance=0.0)

    def test_invalid_parallel_transport_order_raises(
        self, eye4: NDArray[np.float64]
    ) -> None:
        with pytest.raises(LeviCivitaParameterError, match="parallel_transport_order"):
            LeviCivitaConnectionAgent(
                metric_tensor=eye4, parallel_transport_order=3
            )

    def test_christoffel_data_shape_invariant(
        self, eye4: NDArray[np.float64]
    ) -> None:
        n = 4
        Gamma = np.zeros((n, n, n), dtype=np.float64)
        dG = np.zeros((n, n, n), dtype=np.float64)
        G = eye4.copy()
        G_inv = eye4.copy()
        data = ChristoffelData(
            Gamma=Gamma,
            frobenius_norm=0.0,
            infinity_norm=0.0,
            dG=dG,
            G=G,
            G_inv=G_inv,
            dimension=n,
            is_static=True,
            condition_number_reg=1.0,
            inverse_residual=0.0,
            inverse_residual_bound=1e-12,
            spectral_gap_min=1.0,
        )
        assert data.dimension == 4
        assert data.is_static is True

    def test_christoffel_data_rejects_bad_gamma_shape(
        self, eye4: NDArray[np.float64]
    ) -> None:
        n = 4
        with pytest.raises(ValueError, match="Gamma"):
            ChristoffelData(
                Gamma=np.zeros((n, n), dtype=np.float64),  # wrong rank
                frobenius_norm=0.0,
                infinity_norm=0.0,
                dG=np.zeros((n, n, n), dtype=np.float64),
                G=eye4,
                G_inv=eye4,
                dimension=n,
                is_static=True,
                condition_number_reg=1.0,
                inverse_residual=0.0,
                inverse_residual_bound=1e-12,
                spectral_gap_min=1.0,
            )

    def test_christoffel_data_rejects_negative_frob(
        self, eye4: NDArray[np.float64]
    ) -> None:
        n = 4
        with pytest.raises(ValueError, match="normas"):
            ChristoffelData(
                Gamma=np.zeros((n, n, n)),
                frobenius_norm=-1.0,
                infinity_norm=0.0,
                dG=np.zeros((n, n, n)),
                G=eye4,
                G_inv=eye4,
                dimension=n,
                is_static=True,
                condition_number_reg=1.0,
                inverse_residual=0.0,
                inverse_residual_bound=1e-12,
                spectral_gap_min=1.0,
            )

    def test_connection_diagnostics_all_passed_logic(self) -> None:
        ok = ConnectionDiagnostics(
            torsion_norm=0.0,
            covd_metric_norm=0.0,
            riemann_norm=0.0,
            ricci_norm=0.0,
            bianchi_norm=0.0,
            condition_number_reg=1.0,
            torsion_passed=True,
            metric_compat_passed=True,
            bianchi_passed=True,
            is_static=True,
            dimension=4,
        )
        assert ok.all_passed() is True
        s = ok.summary()
        assert s["all_passed"] is True
        assert s["dimension"] == 4

        bad = ConnectionDiagnostics(
            torsion_norm=1.0,
            covd_metric_norm=0.0,
            riemann_norm=0.0,
            ricci_norm=0.0,
            bianchi_norm=0.0,
            condition_number_reg=1.0,
            torsion_passed=False,
            metric_compat_passed=True,
            bianchi_passed=True,
            is_static=False,
            dimension=3,
        )
        assert bad.all_passed() is False

    def test_static_metric_derivative_protocol(self) -> None:
        p = StaticMetricDerivative()
        dG = p.derivative(3)
        assert dG.shape == (3, 3, 3)
        assert np.all(dG == 0.0)
        with pytest.raises(DimensionalMismatchError):
            p.derivative(0)

    def test_callable_metric_derivative_adapter(self) -> None:
        def fn(n: int) -> NDArray[np.float64]:
            out = np.zeros((n, n, n), dtype=np.float64)
            out[0, 0, 0] = 1.0
            out[0, 0, 0] = 1.0  # keep
            return out

        prov = CallableMetricDerivative(fn)
        dG = prov.derivative(2)
        assert dG.shape == (2, 2, 2)
        assert isinstance(prov, MetricDerivativeProvider) or hasattr(
            prov, "derivative"
        )


# =============================================================================
# FASE 1 — CHRISTOFFEL ENGINE
# =============================================================================


class TestPhase1_ChristoffelEngine:
    """
    build_christoffel(G) → ChristoffelData
    Precondición formal de la Fase 2.
    """

    def test_static_identity_gamma_vanishes(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        data = agent_static_eye._christoffel_data
        assert data.is_static is True
        assert data.dimension == 4
        _assert_allclose(data.Gamma, 0.0, atol=_ATOL, msg="Γ debe ser 0 si dG=0")
        assert data.frobenius_norm == pytest.approx(0.0, abs=_ATOL)
        assert data.infinity_norm == pytest.approx(0.0, abs=_ATOL)

    def test_static_diag_gamma_vanishes(
        self, agent_diag: LeviCivitaConnectionAgent
    ) -> None:
        data = agent_diag._christoffel_data
        assert data.is_static is True
        _assert_allclose(data.Gamma, 0.0, atol=_ATOL)

    def test_G_symmetric_and_inverse_residual(
        self, agent_diag: LeviCivitaConnectionAgent
    ) -> None:
        data = agent_diag._christoffel_data
        G, G_inv = data.G, data.G_inv
        _assert_allclose(G, G.T, msg="G debe ser simétrica")
        I = np.eye(data.dimension)
        _assert_allclose(G @ G_inv, I, atol=1e-10, msg="G G^{-1} ≈ I")
        _assert_allclose(G_inv @ G, I, atol=1e-10, msg="G^{-1} G ≈ I")
        assert data.inverse_residual < data.inverse_residual_bound * 10 + 1e-9
        assert data.spectral_gap_min > 0.0

    def test_condition_number_reg_at_least_one(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        assert agent_static_eye._christoffel_data.condition_number_reg >= 1.0 - 1e-12

    def test_ill_conditioned_metric_preconditioned(
        self, ill_conditioned_spd: NDArray[np.float64]
    ) -> None:
        agent = LeviCivitaConnectionAgent(metric_tensor=ill_conditioned_spd)
        data = agent._christoffel_data
        assert data.dimension == 4
        assert np.all(np.isfinite(data.Gamma))
        assert data.inverse_residual < 1.0e-6
        # κ reportado debe ser finito y ≥ 1
        assert math.isfinite(data.condition_number_reg)
        assert data.condition_number_reg >= 1.0

    def test_asymmetric_input_is_symmetrized(self) -> None:
        G_asym = np.array(
            [
                [2.0, 0.3, 0.0],
                [0.1, 1.0, 0.0],  # 0.1 ≠ 0.3
                [0.0, 0.0, 1.5],
            ],
            dtype=np.float64,
        )
        agent = LeviCivitaConnectionAgent(metric_tensor=G_asym)
        G = agent.metric_tensor
        _assert_allclose(G, G.T, atol=1e-14, msg="post-precond G simétrica")

    def test_non_square_metric_raises(self) -> None:
        with pytest.raises(DimensionalMismatchError):
            LeviCivitaConnectionAgent(
                metric_tensor=np.ones((3, 4), dtype=np.float64)
            )

    def test_empty_metric_raises(self) -> None:
        with pytest.raises(DimensionalMismatchError):
            LeviCivitaConnectionAgent(
                metric_tensor=np.zeros((0, 0), dtype=np.float64)
            )

    def test_nan_metric_raises(self) -> None:
        G = np.eye(3, dtype=np.float64)
        G[1, 1] = np.nan
        with pytest.raises(ValueError, match="no finitos"):
            LeviCivitaConnectionAgent(metric_tensor=G)

    def test_inf_metric_raises(self) -> None:
        G = np.eye(2, dtype=np.float64)
        G[0, 0] = np.inf
        with pytest.raises(ValueError, match="no finitos"):
            LeviCivitaConnectionAgent(metric_tensor=G)

    def test_phase1_build_christoffel_direct(self, eye3: NDArray[np.float64]) -> None:
        engine = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = engine.build_christoffel(eye3)
        assert isinstance(data, ChristoffelData)
        assert data.dimension == 3
        assert data.is_static is True
        _assert_allclose(data.Gamma, 0.0)
        _assert_allclose(data.dG, 0.0)

    def test_dG_symmetrization_when_asymmetric_provider(self) -> None:
        """Provider con dG asimétrico en i,j → simetrizado en Fase 1."""

        class AsymProvider:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                # dG[0,0,1] ≠ dG[0,1,0]
                dG[0, 0, 1] = 0.4
                dG[0, 1, 0] = 0.1
                return dG

        engine = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine(
            metric_derivative=AsymProvider()  # type: ignore[arg-type]
        )
        data = engine.build_christoffel(np.eye(2, dtype=np.float64))
        dG = data.dG
        # Tras simetrización: dG[k,i,j] = dG[k,j,i]
        _assert_allclose(
            dG,
            dG.transpose(0, 2, 1),
            atol=_ATOL,
            msg="dG debe ser simétrico en (i,j)",
        )
        # valor esperado: promedio 0.25
        assert dG[0, 0, 1] == pytest.approx(0.25, abs=1e-14)
        assert data.is_static is False

    def test_nontrivial_dG_produces_nonzero_gamma(self) -> None:
        """
        Métrica euclídea con ∂₀G₁₁ = 2 ⇒
        Γ con componentes no nulas vía fórmula de Koszul.
        """

        class PartialG:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                # ∂_0 G_{11} = 2 (índices 0-based: i=j=1)
                dG[0, 1, 1] = 2.0
                return dG

        engine = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine(
            metric_derivative=PartialG()  # type: ignore[arg-type]
        )
        data = engine.build_christoffel(np.eye(2, dtype=np.float64))
        assert data.is_static is False
        assert data.frobenius_norm > 0.0
        assert data.infinity_norm > 0.0
        # Koszul con G=I, G^{-1}=I:
        # Γ^r_{mn} = 1/2 (∂_m G_{rn} + ∂_n G_{rm} - ∂_r G_{mn})
        # Γ^1_{01} = 1/2 (∂_0 G_{11} + ∂_1 G_{10} - ∂_1 G_{01}) = 1/2 * 2 = 1
        assert data.Gamma[1, 0, 1] == pytest.approx(1.0, abs=1e-12)
        # simetría en m,n
        assert data.Gamma[1, 1, 0] == pytest.approx(1.0, abs=1e-12)

    def test_koszul_symmetry_gamma_lower_indices(
        self, random_spd4: NDArray[np.float64]
    ) -> None:
        """Aunque dG=0 aquí, la construcción debe preservar Γ^r_{mn}=Γ^r_{nm}."""
        agent = LeviCivitaConnectionAgent(metric_tensor=random_spd4)
        Gamma = agent.christoffel_symbols
        _assert_allclose(
            Gamma,
            Gamma.transpose(0, 2, 1),
            atol=_ATOL,
            msg="Γ^r_{mn} = Γ^r_{nm}",
        )

    def test_properties_return_copies(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        G1 = agent_static_eye.metric_tensor
        G1[0, 0] = 999.0
        G2 = agent_static_eye.metric_tensor
        assert G2[0, 0] != 999.0
        Gamma1 = agent_static_eye.christoffel_symbols
        Gamma1[0, 0, 0] = 42.0
        assert agent_static_eye.christoffel_symbols[0, 0, 0] != 42.0


# =============================================================================
# FASE 2 — TORSIÓN NULA + COMPATIBILIDAD MÉTRICA
#          (continuación formal de ChristoffelData)
# =============================================================================


class TestPhase2_TorsionFreeConnection:
    """
    verify_axioms(ChristoffelData) → ConnectionDiagnostics
    Precondición operativa de la Fase 3.
    """

    def test_static_identity_axioms_pass(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        diag = agent_static_eye.connection_diagnostics()
        assert isinstance(diag, ConnectionDiagnostics)
        assert diag.torsion_passed is True
        assert diag.metric_compat_passed is True
        assert diag.bianchi_passed is True
        assert diag.all_passed() is True
        assert diag.is_static is True
        assert diag.torsion_norm < _TORSION_TOLERANCE
        assert diag.covd_metric_norm < _METRIC_COMPAT_TOLERANCE
        assert diag.riemann_norm == pytest.approx(0.0, abs=_SOFT_ATOL)
        assert diag.ricci_norm == pytest.approx(0.0, abs=_SOFT_ATOL)
        assert diag.bianchi_norm == pytest.approx(0.0, abs=_SOFT_ATOL)

    def test_static_diag_axioms_pass(
        self, agent_diag: LeviCivitaConnectionAgent
    ) -> None:
        diag = agent_diag.connection_diagnostics()
        assert diag.all_passed() is True
        assert diag.riemann_norm == pytest.approx(0.0, abs=_SOFT_ATOL)

    def test_phase2_accepts_phase1_output_pipeline(
        self, eye4: NDArray[np.float64]
    ) -> None:
        """Continuación formal: Phase1 → ChristoffelData → Phase2."""
        p1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = p1.build_christoffel(eye4)
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        diag = p2.verify_axioms(data)
        assert diag.all_passed()
        assert diag.dimension == data.dimension

    def test_torsion_tensor_antisymmetric_in_lower(
        self, eye3: NDArray[np.float64]
    ) -> None:
        p1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = p1.build_christoffel(eye3)
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        T = p2._compute_torsion_tensor(data.Gamma)
        # T^r_{mn} = -T^r_{nm}
        _assert_allclose(T, -T.transpose(0, 2, 1), atol=_ATOL)
        assert float(la.norm(T, "fro")) < _TORSION_TOLERANCE

    def test_artificial_torsion_raises(self, eye3: NDArray[np.float64]) -> None:
        p1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = p1.build_christoffel(eye3)
        # Contaminar Γ con parte antisimétrica
        Gamma_bad = data.Gamma.copy()
        Gamma_bad[0, 1, 2] += 0.5
        Gamma_bad[0, 2, 1] -= 0.1  # no cancela
        data_bad = ChristoffelData(
            Gamma=Gamma_bad,
            frobenius_norm=float(la.norm(Gamma_bad, "fro")),
            infinity_norm=float(np.max(np.abs(Gamma_bad))),
            dG=data.dG,
            G=data.G,
            G_inv=data.G_inv,
            dimension=data.dimension,
            is_static=data.is_static,
            condition_number_reg=data.condition_number_reg,
            inverse_residual=data.inverse_residual,
            inverse_residual_bound=data.inverse_residual_bound,
            spectral_gap_min=data.spectral_gap_min,
        )
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection(
            torsion_tolerance=_TORSION_TOLERANCE
        )
        with pytest.raises(TopologicalTorsionError, match="Torsión"):
            p2.verify_axioms(data_bad)

    def test_metric_compatibility_static_zero(
        self, agent_diag: LeviCivitaConnectionAgent
    ) -> None:
        data = agent_diag._christoffel_data
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        covd = p2._compute_covd_metric(data.Gamma, data.dG, data.G)
        assert float(la.norm(covd, "fro")) < _METRIC_COMPAT_TOLERANCE

    def test_covd_metric_failure_raises(self, eye3: NDArray[np.float64]) -> None:
        """
        Fuerza ∇G ≠ 0: dG no nulo con Γ=0 (inconsistente con LC).
        """
        p1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = p1.build_christoffel(eye3)
        dG_bad = np.zeros_like(data.dG)
        dG_bad[0, 0, 0] = 1.0  # ∂_0 G_00 = 1, pero Γ=0
        data_bad = ChristoffelData(
            Gamma=data.Gamma,  # sigue 0
            frobenius_norm=0.0,
            infinity_norm=0.0,
            dG=dG_bad,
            G=data.G,
            G_inv=data.G_inv,
            dimension=data.dimension,
            is_static=False,
            condition_number_reg=data.condition_number_reg,
            inverse_residual=data.inverse_residual,
            inverse_residual_bound=data.inverse_residual_bound,
            spectral_gap_min=data.spectral_gap_min,
        )
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection(
            metric_compat_tolerance=_METRIC_COMPAT_TOLERANCE
        )
        with pytest.raises(MetricCompatibilityError, match="∇G"):
            p2.verify_axioms(data_bad)

    def test_riemann_quadratic_vanishes_for_zero_gamma(
        self, eye4: NDArray[np.float64]
    ) -> None:
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        Gamma = np.zeros((4, 4, 4), dtype=np.float64)
        R = p2._compute_riemann_quadratic(Gamma)
        assert R.shape == (4, 4, 4, 4)
        _assert_allclose(R, 0.0)

    def test_ricci_contraction_shape_and_zero(
        self, eye3: NDArray[np.float64]
    ) -> None:
        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        R = np.zeros((3, 3, 3, 3), dtype=np.float64)
        Ric = p2._compute_ricci(R)
        assert Ric.shape == (3, 3)
        _assert_allclose(Ric, 0.0)

    def test_bianchi_holds_for_static(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        diag = agent_static_eye.connection_diagnostics()
        assert diag.bianchi_norm < _BIANCHI_TOLERANCE
        assert diag.bianchi_passed is True

    def test_consistent_nontrivial_dG_axioms(
        self,
    ) -> None:
        """
        Con dG simétrico y Γ de Koszul, torsión y ∇G deben pasar
        (compatibilidad métrica es identidad de la definición LC).
        """

        class FlatDerivative:
            """∂_k G_{ij} constante simétrica pequeña."""

            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                # ∂_0 G_{00} = 0.02 (simétrico trivial en i,j)
                dG[0, 0, 0] = 0.02
                return dG

        agent = LeviCivitaConnectionAgent(
            metric_tensor=np.eye(2, dtype=np.float64),
            metric_derivative=FlatDerivative(),  # type: ignore[arg-type]
            metric_compat_tolerance=1.0e-9,
        )
        diag = agent.connection_diagnostics()
        assert diag.torsion_passed is True
        assert diag.metric_compat_passed is True
        assert diag.is_static is False
        # Γ no idénticamente nulo
        assert agent._christoffel_data.frobenius_norm > 0.0


# =============================================================================
# FASE 3 — GEODÉSICA, TRANSPORTE, ISOMORFISMOS
# =============================================================================


class TestPhase3_GeodesicRHS:
    """geodesic_rhs: a = −Γ(v,v)."""

    def test_rhs_vanishes_for_static(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_generic: TangentVector,
    ) -> None:
        a = agent_static_eye.geodesic_rhs(tangent_generic.coordinates)
        _assert_allclose(a, 0.0, atol=_ATOL, msg="a=0 si Γ=0")

    def test_rhs_dimension_mismatch_raises(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        with pytest.raises(DimensionalMismatchError):
            agent_static_eye.geodesic_rhs(np.array([1.0, 2.0], dtype=np.float64))

    def test_rhs_quadratic_homogeneity(
        self,
    ) -> None:
        """a(λv) = λ² a(v) por bilinealidad de Γ(v,v)."""

        class PartialG:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                dG[0, 1, 1] = 2.0
                return dG

        agent = LeviCivitaConnectionAgent(
            metric_tensor=np.eye(2, dtype=np.float64),
            metric_derivative=PartialG(),  # type: ignore[arg-type]
            enforce_norm_conservation=False,
        )
        v = np.array([1.0, 1.0], dtype=np.float64)
        a1 = agent.geodesic_rhs(v)
        a2 = agent.geodesic_rhs(3.0 * v)
        _assert_allclose(a2, 9.0 * a1, atol=1e-12)


class TestPhase3_EnforceGeodesicFlow:
    """enforce_geodesic_flow: RK4 + norma G + reportes."""

    def test_static_flow_preserves_vector(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_generic: TangentVector,
    ) -> None:
        v2 = agent_static_eye.enforce_geodesic_flow(tangent_generic, dt=1e-2)
        assert isinstance(v2, TangentVector)
        _assert_allclose(
            v2.coordinates,
            tangent_generic.coordinates,
            atol=1e-12,
            msg="flujo trivial si Γ=0",
        )

    def test_return_report_tuple(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_unit: TangentVector,
    ) -> None:
        out = agent_static_eye.enforce_geodesic_flow(
            tangent_unit, dt=1e-3, return_report=True
        )
        assert isinstance(out, tuple) and len(out) == 2
        v2, report = out
        assert isinstance(v2, TangentVector)
        assert isinstance(report, GeodesicStepReport)
        assert report.dt == pytest.approx(1e-3)
        assert report.norm_drift_G < _GEODESIC_NORM_DRIFT_TOL
        assert report.v_initial_norm_G == pytest.approx(1.0, abs=1e-12)
        assert report.is_stable is True

    def test_G_norm_conservation_diag(
        self,
        agent_diag: LeviCivitaConnectionAgent,
        diag_metric: NDArray[np.float64],
    ) -> None:
        v0 = np.array([0.4, -0.2, 0.5, 0.1], dtype=np.float64)
        n0 = _g_norm(diag_metric, v0)
        tv = TangentVector(coordinates=v0)
        v1, rep = agent_diag.enforce_geodesic_flow(  # type: ignore[misc]
            tv, dt=1e-3, return_report=True
        )
        n1 = _g_norm(diag_metric, v1.coordinates)
        assert abs(n1 - n0) / max(n0, _MACHINE_EPS) < 1e-9
        assert rep.renormalized is True or rep.norm_drift_G < 1e-9

    def test_without_renormalization_flag(
        self,
        eye4: NDArray[np.float64],
        tangent_generic: TangentVector,
    ) -> None:
        agent = LeviCivitaConnectionAgent(
            metric_tensor=eye4,
            enforce_norm_conservation=False,
        )
        _, rep = agent.enforce_geodesic_flow(  # type: ignore[misc]
            tangent_generic, dt=1e-3, return_report=True, renormalize=False
        )
        assert rep.renormalized is False

    def test_dt_too_small_raises(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_unit: TangentVector,
    ) -> None:
        with pytest.raises(LeviCivitaParameterError, match="dt"):
            agent_static_eye.enforce_geodesic_flow(tangent_unit, dt=_DT_MIN * 0.5)

    def test_wrong_type_velocity_raises(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        with pytest.raises(TypeError, match="TangentVector"):
            agent_static_eye.enforce_geodesic_flow(
                np.array([1.0, 0, 0, 0]), dt=1e-3  # type: ignore[arg-type]
            )

    def test_dimension_mismatch_velocity_raises(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        bad = TangentVector(coordinates=np.array([1.0, 2.0], dtype=np.float64))
        with pytest.raises(DimensionalMismatchError):
            agent_static_eye.enforce_geodesic_flow(bad, dt=1e-3)

    def test_nan_velocity_raises(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        bad = TangentVector(
            coordinates=np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float64)
        )
        with pytest.raises(ValueError, match="no finitas"):
            agent_static_eye.enforce_geodesic_flow(bad, dt=1e-3)

    def test_default_dt_used(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_unit: TangentVector,
    ) -> None:
        v2, rep = agent_static_eye.enforce_geodesic_flow(  # type: ignore[misc]
            tangent_unit, dt=None, return_report=True
        )
        assert rep.dt == pytest.approx(_DEFAULT_DT)
        assert isinstance(v2, TangentVector)

    def test_multi_step_static_orbit(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_generic: TangentVector,
    ) -> None:
        v = tangent_generic
        for _ in range(20):
            v = agent_static_eye.enforce_geodesic_flow(v, dt=1e-2)  # type: ignore[assignment]
            assert isinstance(v, TangentVector)
        _assert_allclose(
            v.coordinates,
            tangent_generic.coordinates,
            atol=1e-10,
        )

    def test_velocity_dependent_dt_max(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        # Con Γ=0, dt_max es enorme (capa de seguridad)
        dt1 = agent_static_eye._compute_max_stable_dt_for_speed(1.0)
        dt10 = agent_static_eye._compute_max_stable_dt_for_speed(10.0)
        assert dt1 > 0.0 and dt10 > 0.0
        # Con Γ no nulo, dt_max ∝ 1/‖v‖_G
        class PartialG:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                dG[0, 0, 0] = 1.0
                return dG

        agent = LeviCivitaConnectionAgent(
            metric_tensor=np.eye(2, dtype=np.float64),
            metric_derivative=PartialG(),  # type: ignore[arg-type]
        )
        d_slow = agent._compute_max_stable_dt_for_speed(0.1)
        d_fast = agent._compute_max_stable_dt_for_speed(10.0)
        assert d_slow > d_fast


class TestPhase3_ParallelTransport:
    """Transporte paralelo Euler (1) y Heun (2)."""

    def test_static_parallel_transport_is_identity(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
    ) -> None:
        V = TangentVector(
            coordinates=np.array([0.2, 0.3, -0.1, 0.4], dtype=np.float64)
        )
        ydot = TangentVector(
            coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        V2 = agent_static_eye.parallel_transport(V, ydot, dt=1e-2)
        _assert_allclose(V2.coordinates, V.coordinates, atol=1e-12)

    def test_heun_vs_euler_static_agree(
        self, eye4: NDArray[np.float64]
    ) -> None:
        V = TangentVector(coordinates=np.ones(4, dtype=np.float64))
        ydot = TangentVector(
            coordinates=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        )
        a1 = LeviCivitaConnectionAgent(
            metric_tensor=eye4, parallel_transport_order=1
        )
        a2 = LeviCivitaConnectionAgent(
            metric_tensor=eye4, parallel_transport_order=2
        )
        V1 = a1.parallel_transport(V, ydot, dt=1e-3)
        V2 = a2.parallel_transport(V, ydot, dt=1e-3)
        _assert_allclose(V1.coordinates, V2.coordinates, atol=1e-12)

    def test_parallel_transport_type_checks(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        V = TangentVector(coordinates=np.ones(4, dtype=np.float64))
        with pytest.raises(TypeError):
            agent_static_eye.parallel_transport(
                V, np.ones(4), dt=1e-3  # type: ignore[arg-type]
            )

    def test_parallel_transport_dim_mismatch(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        V = TangentVector(coordinates=np.ones(3, dtype=np.float64))
        ydot = TangentVector(coordinates=np.ones(4, dtype=np.float64))
        with pytest.raises(DimensionalMismatchError):
            agent_static_eye.parallel_transport(V, ydot, dt=1e-3)

    def test_nontrivial_transport_changes_vector(self) -> None:
        class PartialG:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                dG[0, 1, 1] = 2.0
                return dG

        agent = LeviCivitaConnectionAgent(
            metric_tensor=np.eye(2, dtype=np.float64),
            metric_derivative=PartialG(),  # type: ignore[arg-type]
            parallel_transport_order=2,
            enforce_norm_conservation=False,
        )
        V = TangentVector(coordinates=np.array([0.0, 1.0], dtype=np.float64))
        ydot = TangentVector(coordinates=np.array([1.0, 0.0], dtype=np.float64))
        V2 = agent.parallel_transport(V, ydot, dt=0.1)
        # Con Γ no nulo, V debe cambiar
        assert not np.allclose(V2.coordinates, V.coordinates, atol=1e-14)


class TestPhase3_MusicalIsomorphisms:
    """♭ / ♯ y puentes logística ↔ finanzas."""

    def test_flat_sharp_roundtrip_identity(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_generic: TangentVector,
    ) -> None:
        omega = agent_static_eye._musical_engine.apply_flat_isomorphism(
            tangent_generic
        )
        assert isinstance(omega, CotangentVector)
        v_back = agent_static_eye._musical_engine.apply_sharp_isomorphism(omega)
        _assert_allclose(
            v_back.coordinates,
            tangent_generic.coordinates,
            atol=1e-10,
            msg="♯♭ = id",
        )

    def test_flat_sharp_roundtrip_diag(
        self,
        agent_diag: LeviCivitaConnectionAgent,
        diag_metric: NDArray[np.float64],
    ) -> None:
        v = TangentVector(
            coordinates=np.array([0.7, -0.2, 0.3, 0.5], dtype=np.float64)
        )
        omega = agent_diag._musical_engine.apply_flat_isomorphism(v)
        # ω = G v
        _assert_allclose(
            omega.coordinates,
            diag_metric @ v.coordinates,
            atol=1e-10,
        )
        v2 = agent_diag._musical_engine.apply_sharp_isomorphism(omega)
        _assert_allclose(v2.coordinates, v.coordinates, atol=1e-10)

    def test_transport_to_finance_oracle(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
        tangent_unit: TangentVector,
    ) -> None:
        omega, report = agent_static_eye.transport_to_finance_oracle(
            tangent_unit, dt=1e-3, apply_geodesic_correction=True
        )
        assert isinstance(omega, CotangentVector)
        assert isinstance(report, GeodesicStepReport)
        _assert_allclose(omega.coordinates, tangent_unit.coordinates, atol=1e-12)

    def test_transport_to_finance_without_geodesic(
        self,
        agent_diag: LeviCivitaConnectionAgent,
    ) -> None:
        v = TangentVector(
            coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        omega, report = agent_diag.transport_to_finance_oracle(
            v, apply_geodesic_correction=False
        )
        assert report.dt == pytest.approx(0.0)
        # ω_0 = G_{00} v^0 = 1.0 * 1.0
        assert omega.coordinates[0] == pytest.approx(1.0, abs=1e-12)

    def test_transport_to_logistics_manifold(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
    ) -> None:
        w = CotangentVector(
            coordinates=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        )
        v, rep = agent_static_eye.transport_to_logistics_manifold(
            w, apply_post_geodesic=False
        )
        assert isinstance(v, TangentVector)
        assert rep is None
        _assert_allclose(v.coordinates, w.coordinates, atol=1e-12)

    def test_transport_to_logistics_with_post_geodesic(
        self,
        agent_static_eye: LeviCivitaConnectionAgent,
    ) -> None:
        w = CotangentVector(
            coordinates=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        )
        v, rep = agent_static_eye.transport_to_logistics_manifold(
            w, apply_post_geodesic=True, dt=1e-3
        )
        assert isinstance(v, TangentVector)
        assert isinstance(rep, GeodesicStepReport)

    def test_transport_to_logistics_wrong_type(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        with pytest.raises(TypeError, match="CotangentVector"):
            agent_static_eye.transport_to_logistics_manifold(
                TangentVector(coordinates=np.ones(4))  # type: ignore[arg-type]
            )


# =============================================================================
# CONTRATO CATEGÓRICO + DIAGNÓSTICOS + PROPIEDADES
# =============================================================================


class TestCategoricalContract:
    def test_forward_static(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        # Usar CategoricalState del módulo (stub o real)
        from app.omega.levi_civita_agent import CategoricalState

        payload = np.array([0.5, -0.5, 0.25, 0.0], dtype=np.float64)
        state = CategoricalState(payload=payload, label="psi0")
        out = agent_static_eye.forward(state)
        assert isinstance(out, CategoricalState)
        _assert_allclose(out.payload, payload, atol=1e-12)
        assert "levi_civita_forward" in str(out.label)

    def test_forward_dim_mismatch_raises(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        from app.omega.levi_civita_agent import CategoricalState

        state = CategoricalState(
            payload=np.array([1.0, 2.0], dtype=np.float64), label="bad"
        )
        with pytest.raises(DimensionalMismatchError):
            agent_static_eye.forward(state)

    def test_backward_delegates_to_forward(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        from app.omega.levi_civita_agent import CategoricalState

        payload = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state = CategoricalState(payload=payload, label="x")
        b = agent_static_eye.backward(state)
        f = agent_static_eye.forward(state)
        _assert_allclose(b.payload, f.payload, atol=1e-14)

    def test_geodesic_flow_report_keys(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        rep = agent_static_eye.geodesic_flow_report()
        required = {
            "metric_dimension",
            "christoffel_frob_norm",
            "christoffel_inf_norm",
            "is_static_metric",
            "condition_number_reg",
            "inverse_residual",
            "inverse_residual_bound",
            "spectral_gap_min",
            "torsion_norm",
            "covd_metric_norm",
            "riemann_norm",
            "ricci_norm",
            "bianchi_norm",
            "torsion_passed",
            "metric_compat_passed",
            "bianchi_passed",
            "all_axioms_passed",
            "dt_default",
            "dt_max_stable_unit",
            "enforce_norm_conservation",
            "parallel_transport_order",
            "agent",
        }
        assert required.issubset(set(rep.keys()))
        assert rep["all_axioms_passed"] is True
        assert rep["metric_dimension"] == 4
        assert "v8" in rep["agent"]

    def test_public_properties(
        self, agent_diag: LeviCivitaConnectionAgent, diag_metric: NDArray[np.float64]
    ) -> None:
        assert agent_diag.metric_dimension == 4
        assert agent_diag.dimension == 4
        assert agent_diag.is_static is True
        assert agent_diag.dt_max_stable_unit > 0.0
        _assert_allclose(
            np.diag(agent_diag.metric_tensor),
            np.diag(diag_metric),
            atol=1e-10,
        )


# =============================================================================
# ESTRÉS NUMÉRICO Y REGRESIONES
# =============================================================================


class TestNumericalStress:
    def test_high_dimension_static(self) -> None:
        n = 16
        agent = LeviCivitaConnectionAgent(metric_tensor=np.eye(n, dtype=np.float64))
        assert agent.dimension == n
        assert agent.connection_diagnostics().all_passed()
        v = TangentVector(coordinates=np.linspace(0.1, 1.0, n, dtype=np.float64))
        v2 = agent.enforce_geodesic_flow(v, dt=1e-3)
        assert isinstance(v2, TangentVector)
        _assert_allclose(v2.coordinates, v.coordinates, atol=1e-11)

    def test_many_rk4_steps_norm_G(
        self, agent_diag: LeviCivitaConnectionAgent, diag_metric: NDArray[np.float64]
    ) -> None:
        v = np.array([0.3, 0.4, 0.1, -0.2], dtype=np.float64)
        n0 = _g_norm(diag_metric, v)
        tv = TangentVector(coordinates=v)
        for _ in range(50):
            tv = agent_diag.enforce_geodesic_flow(tv, dt=5e-4)  # type: ignore[assignment]
            assert isinstance(tv, TangentVector)
        n1 = _g_norm(diag_metric, np.asarray(tv.coordinates))
        rel = abs(n1 - n0) / max(n0, _MACHINE_EPS)
        assert rel < 1e-8, f"deriva acumulada de ‖v‖_G: {rel:.3e}"

    def test_random_spd_batch_axioms(self) -> None:
        rng = np.random.default_rng(123)
        for i in range(8):
            ev = rng.uniform(0.2, 5.0, size=3)
            G = _spd_from_eig(ev.astype(np.float64), seed=100 + i)
            agent = LeviCivitaConnectionAgent(metric_tensor=G)
            assert agent.connection_diagnostics().all_passed()
            assert agent._christoffel_data.is_static is True

    def test_geodesic_rhs_consistency_with_flow_derivative(
        self, agent_static_eye: LeviCivitaConnectionAgent
    ) -> None:
        """
        Para Γ=0: (v(dt)−v(0))/dt ≈ rhs = 0.
        """
        v0 = np.array([0.9, -0.1, 0.2, 0.3], dtype=np.float64)
        dt = 1e-4
        tv = TangentVector(coordinates=v0)
        v1 = agent_static_eye.enforce_geodesic_flow(
            tv, dt=dt, renormalize=False
        )
        assert isinstance(v1, TangentVector)
        num_deriv = (v1.coordinates - v0) / dt
        rhs = agent_static_eye.geodesic_rhs(v0)
        _assert_allclose(num_deriv, rhs, atol=1e-8)

    def test_nontrivial_gamma_energy_with_renorm(self) -> None:
        class PartialG:
            def derivative(self, n: int) -> NDArray[np.float64]:
                dG = np.zeros((n, n, n), dtype=np.float64)
                dG[0, 0, 0] = 0.5
                dG[0, 1, 1] = 0.2
                dG[0, 1, 0] = 0.0  # se simetrizará con [0,0,1]
                return dG

        G = np.eye(2, dtype=np.float64)
        agent = LeviCivitaConnectionAgent(
            metric_tensor=G,
            metric_derivative=PartialG(),  # type: ignore[arg-type]
            enforce_norm_conservation=True,
        )
        v0 = np.array([0.6, 0.8], dtype=np.float64)  # ‖v‖_2 = 1 = ‖v‖_G
        n0 = _g_norm(G, v0)
        tv = TangentVector(coordinates=v0)
        tv2, rep = agent.enforce_geodesic_flow(  # type: ignore[misc]
            tv, dt=1e-3, return_report=True
        )
        n1 = _g_norm(G, tv2.coordinates)
        assert abs(n1 - n0) < 1e-10
        assert rep.renormalized is True


class TestPhaseContinuationContract:
    """
    Verifica el contrato de continuación formal entre fases:
    Phase1.build_christoffel → Phase2.verify_axioms → Phase3 ops.
    """

    def test_full_pipeline_types(self, random_spd4: NDArray[np.float64]) -> None:
        p1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine()
        data = p1.build_christoffel(random_spd4)
        assert isinstance(data, ChristoffelData)

        p2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection()
        diag = p2.verify_axioms(data)
        assert isinstance(diag, ConnectionDiagnostics)
        assert diag.all_passed()

        # Fase 3 vía orquestador que ya compuso p1/p2
        agent = LeviCivitaConnectionAgent(metric_tensor=random_spd4)
        v = TangentVector(
            coordinates=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        )
        v2 = agent.enforce_geodesic_flow(v, dt=1e-3)
        assert isinstance(v2, TangentVector)
        a = agent.geodesic_rhs(v.coordinates)
        assert a.shape == (4,)
        assert agent.connection_diagnostics().all_passed()

    def test_orchestrator_caches_consistent_with_phases(
        self, eye4: NDArray[np.float64]
    ) -> None:
        agent = LeviCivitaConnectionAgent(metric_tensor=eye4)
        data = agent.phase1.build_christoffel(eye4)
        diag = agent.phase2.verify_axioms(data)
        # Caché del orquestador coherente
        _assert_allclose(agent._Gamma, data.Gamma)
        assert agent._connection_diagnostics.torsion_norm == pytest.approx(
            diag.torsion_norm
        )
        assert agent._n == data.dimension


# =============================================================================
# MARKERS / SMOKE
# =============================================================================


class TestSmokeImportAndRepr:
    def test_agent_constructs_default(self) -> None:
        agent = LeviCivitaConnectionAgent()
        assert agent.dimension >= 1
        assert agent.geodesic_flow_report()["all_axioms_passed"] is True

    def test_exports_available(self) -> None:
        import app.omega.levi_civita_agent as m

        for name in [
            "LeviCivitaConnectionAgent",
            "ChristoffelData",
            "ConnectionDiagnostics",
            "GeodesicStepReport",
            "TopologicalTorsionError",
            "MetricCompatibilityError",
            "StaticMetricDerivative",
        ]:
            assert hasattr(m, name), f"falta export {name}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "--tb=short"])