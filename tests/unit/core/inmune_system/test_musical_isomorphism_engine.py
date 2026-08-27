# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Suite de Pruebas: Musical Isomorphism Engine                                        ║
║  Ruta   : tests/unit/core/immune_system/test_musical_isomorphism_engine.py           ║
║  Versión: 4.0.0-Topos-Spectral-Categorical                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Organización por continuación formal
------------------------------------
.. code-block:: text

    TestPhase0_TypesVarianceDTO
        │
    TestPhase1_MetricSpectralPreconditioner
        │  precondition(G) → PreconditionedMetric
        ▼
    TestPhase2_FlatIsomorphism
        │  ♭, ‖·‖_G, Riesz pairing
        ▼
    TestPhase3_SharpIsomorphism
        │  ♯, roundtrips, isometría, audit Z₂
        ▼
    TestEngineOrchestrator / TestNumericalStress / TestContinuationContract

Ejecución
---------
    pytest app/core/immune_system/tests/test_musical_isomorphism_engine.py -v --tb=short
    pytest ... -k "Phase1 or Phase2" -v
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# SUT
# ---------------------------------------------------------------------------
from app.core.immune_system.musical_isomorphism_engine import (
    CategoricalVariance,
    ContravariantFunctor,
    CotangentVector,
    CovariantFunctor,
    FlatIsomorphism,
    FunctorialityError,
    InversionAudit,
    MetricSpectralPreconditioner,
    MusicalIsomorphismEngine,
    NumericalInstabilityError,
    PairingReport,
    PreconditionedMetric,
    RoundtripReport,
    SharpIsomorphism,
    TangentVector,
    _CONDITION_THRESHOLD,
    _INVERSION_BASE_TOLERANCE,
    _MACHINE_EPSILON,
    _SYMMETRY_TOLERANCE,
    _TIKHONOV_EPSILON_RATIO,
)


# =============================================================================
# UTILIDADES Y FIXTURES
# =============================================================================

_RTOL = 1.0e-12
_ATOL = 1.0e-14
_SOFT = 1.0e-9


def _sym(A: NDArray[np.float64]) -> NDArray[np.float64]:
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)


def _spd_from_eig(
    eigenvalues: NDArray[np.float64],
    seed: int = 0,
) -> NDArray[np.float64]:
    n = int(eigenvalues.shape[0])
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.normal(size=(n, n)))
    return _sym(Q @ np.diag(np.asarray(eigenvalues, dtype=np.float64)) @ Q.T)


def _assert_allclose(
    a: Any,
    b: Any,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
    msg: str = "",
) -> None:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def _g_norm(G: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return math.sqrt(max(float(v @ G @ v), 0.0))


@pytest.fixture
def eye2() -> NDArray[np.float64]:
    return np.eye(2, dtype=np.float64)


@pytest.fixture
def eye3() -> NDArray[np.float64]:
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def eye4() -> NDArray[np.float64]:
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def diag_metric() -> NDArray[np.float64]:
    return np.diag([1.0, 2.0, 0.5, 4.0]).astype(np.float64)


@pytest.fixture
def random_spd3() -> NDArray[np.float64]:
    return _spd_from_eig(
        np.array([0.4, 1.2, 3.5], dtype=np.float64), seed=11
    )


@pytest.fixture
def ill_conditioned_spd() -> NDArray[np.float64]:
    return _spd_from_eig(
        np.array([1.0, 1e2, 1e5, 1e10], dtype=np.float64), seed=99
    )


@pytest.fixture
def singular_psd() -> NDArray[np.float64]:
    """Rango deficiente: λ = (0, 1, 2)."""
    return _spd_from_eig(
        np.array([0.0, 1.0, 2.0], dtype=np.float64), seed=3
    )


@pytest.fixture
def precond() -> MetricSpectralPreconditioner:
    return MetricSpectralPreconditioner()


@pytest.fixture
def pm_eye4(
    precond: MetricSpectralPreconditioner, eye4: NDArray[np.float64]
) -> PreconditionedMetric:
    return precond.precondition(eye4)


@pytest.fixture
def pm_diag(
    precond: MetricSpectralPreconditioner, diag_metric: NDArray[np.float64]
) -> PreconditionedMetric:
    return precond.precondition(diag_metric)


@pytest.fixture
def engine_eye4(eye4: NDArray[np.float64]) -> MusicalIsomorphismEngine:
    return MusicalIsomorphismEngine(metric_tensor=eye4)


@pytest.fixture
def engine_diag(diag_metric: NDArray[np.float64]) -> MusicalIsomorphismEngine:
    return MusicalIsomorphismEngine(metric_tensor=diag_metric)


@pytest.fixture
def tangent_unit4() -> TangentVector:
    return TangentVector(
        coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )


@pytest.fixture
def tangent_generic4() -> TangentVector:
    return TangentVector(
        coordinates=np.array([0.5, -0.3, 0.8, 0.1], dtype=np.float64)
    )


# =============================================================================
# FASE 0 — TIPOS, VARIANZA Z₂, DTOs
# =============================================================================


class TestPhase0_TypesVarianceDTO:
    """Contratos algebraicos de tipos y DTOs."""

    # --- CategoricalVariance ------------------------------------------------

    def test_variance_cayley_table(self) -> None:
        C = CategoricalVariance.COVARIANT
        K = CategoricalVariance.CONTRAVARIANT
        assert C * C == CategoricalVariance.COVARIANT
        assert C * K == CategoricalVariance.CONTRAVARIANT
        assert K * C == CategoricalVariance.CONTRAVARIANT
        assert K * K == CategoricalVariance.COVARIANT

    def test_variance_mul_type_error(self) -> None:
        with pytest.raises(TypeError):
            _ = CategoricalVariance.COVARIANT * 1  # type: ignore[operator]

    def test_variance_repr(self) -> None:
        r = repr(CategoricalVariance.COVARIANT)
        assert "COVARIANT" in r
        assert "+1" in r or "1" in r

    # --- TangentVector / CotangentVector ------------------------------------

    def test_tangent_valid(self) -> None:
        v = TangentVector(coordinates=np.array([1.0, 2.0], dtype=np.float64))
        assert v.dim == 2
        assert v.norm == pytest.approx(math.sqrt(5.0))

    def test_tangent_coerces_float32(self) -> None:
        v = TangentVector(
            coordinates=np.array([1.0, 0.0], dtype=np.float32)
        )
        assert v.coordinates.dtype == np.float64

    def test_tangent_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            TangentVector(coordinates=np.eye(2, dtype=np.float64))

    def test_tangent_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="n ≥ 1"):
            TangentVector(coordinates=np.array([], dtype=np.float64))

    def test_tangent_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="no finitas"):
            TangentVector(
                coordinates=np.array([1.0, np.nan], dtype=np.float64)
            )

    def test_tangent_rejects_non_array(self) -> None:
        with pytest.raises(TypeError):
            TangentVector(coordinates=[1.0, 2.0])  # type: ignore[arg-type]

    def test_cotangent_valid_and_copy(self) -> None:
        w = CotangentVector(
            coordinates=np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        assert w.dim == 3
        c = w.copy_coords()
        c[0] = 99.0
        assert w.coordinates[0] == 0.0

    def test_cotangent_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="no finitas"):
            CotangentVector(
                coordinates=np.array([np.inf, 0.0], dtype=np.float64)
            )

    # --- PreconditionedMetric invariants ------------------------------------

    def test_pm_from_pipeline_ok(self, pm_eye4: PreconditionedMetric) -> None:
        assert pm_eye4.matrix_dimension == 4
        assert pm_eye4.condition_number_reg >= 1.0 - 1e-12
        assert np.all(pm_eye4.eigenvalues_reg > 0)
        s = pm_eye4.spectral_summary()
        assert "lambda_min_reg" in s
        assert "inversion_residual" in s

    def test_pm_rejects_bad_shape(self, eye2: NDArray[np.float64]) -> None:
        with pytest.raises(ValueError, match="G"):
            PreconditionedMetric(
                G=np.eye(3),
                G_inv=eye2,
                eigenvalues_raw=np.ones(2),
                eigenvalues_reg=np.ones(2),
                eigenvectors=eye2,
                condition_number_raw=1.0,
                condition_number_reg=1.0,
                spectral_gap_absolute=1.0,
                spectral_gap_cheeger=1.0,
                null_space_dim=0,
                tikhonov_epsilon=0.0,
                regularization_applied=False,
                matrix_dimension=2,
            )

    def test_pm_rejects_nonpositive_eigs_reg(
        self, eye2: NDArray[np.float64]
    ) -> None:
        with pytest.raises(ValueError, match="eigenvalues_reg"):
            PreconditionedMetric(
                G=eye2,
                G_inv=eye2,
                eigenvalues_raw=np.array([1.0, 1.0]),
                eigenvalues_reg=np.array([-1.0, 1.0]),
                eigenvectors=eye2,
                condition_number_raw=1.0,
                condition_number_reg=1.0,
                spectral_gap_absolute=1.0,
                spectral_gap_cheeger=1.0,
                null_space_dim=0,
                tikhonov_epsilon=0.0,
                regularization_applied=False,
                matrix_dimension=2,
            )

    # --- Funtores -----------------------------------------------------------

    def test_covariant_functor_variance(self) -> None:
        f = CovariantFunctor()
        assert f.variance == CategoricalVariance.COVARIANT
        assert f.domain_category == "C"
        assert f.codomain_category == "D"
        assert f.map_object(42) == 42

    def test_contravariant_functor_variance(self) -> None:
        f = ContravariantFunctor()
        assert f.variance == CategoricalVariance.CONTRAVARIANT
        assert f.domain_category == "C^{op}"


# =============================================================================
# FASE 1 — MetricSpectralPreconditioner
# =============================================================================


class TestPhase1_MetricSpectralPreconditioner:
    """precondition(G) → PreconditionedMetric."""

    def test_identity_no_regularization(
        self,
        precond: MetricSpectralPreconditioner,
        eye4: NDArray[np.float64],
    ) -> None:
        pm = precond.precondition(eye4)
        assert pm.regularization_applied is False
        assert pm.tikhonov_epsilon == pytest.approx(0.0)
        assert pm.null_space_dim == 0
        assert pm.condition_number_reg == pytest.approx(1.0, abs=1e-10)
        _assert_allclose(pm.G, eye4, atol=1e-12)
        _assert_allclose(pm.G_inv, eye4, atol=1e-12)
        assert pm.inversion_residual <= pm.inversion_tolerance

    def test_diag_metric_inverse(
        self,
        precond: MetricSpectralPreconditioner,
        diag_metric: NDArray[np.float64],
    ) -> None:
        pm = precond.precondition(diag_metric)
        expected_inv = np.diag(1.0 / np.diag(diag_metric))
        # autovalores pueden reordenar vía base espectral; verificar GG^{-1}=I
        I = np.eye(4)
        _assert_allclose(pm.G @ pm.G_inv, I, atol=1e-10)
        _assert_allclose(pm.G_inv @ pm.G, I, atol=1e-10)
        # G simétrica SPD con mismos autovalores
        ev_g = np.sort(np.linalg.eigvalsh(pm.G))
        ev_d = np.sort(np.diag(diag_metric))
        _assert_allclose(ev_g, ev_d, atol=1e-10)

    def test_asymmetry_projected(
        self, precond: MetricSpectralPreconditioner
    ) -> None:
        G = np.array(
            [[2.0, 0.4, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.5]],
            dtype=np.float64,
        )
        pm = precond.precondition(G)
        _assert_allclose(pm.G, pm.G.T, atol=1e-14)

    def test_non_square_raises(
        self, precond: MetricSpectralPreconditioner
    ) -> None:
        with pytest.raises(ValueError, match="cuadrado"):
            precond.precondition(np.ones((2, 3), dtype=np.float64))

    def test_not_2d_raises(self, precond: MetricSpectralPreconditioner) -> None:
        with pytest.raises(ValueError, match="2-D"):
            precond.precondition(np.ones((2, 2, 2), dtype=np.float64))

    def test_empty_raises(self, precond: MetricSpectralPreconditioner) -> None:
        with pytest.raises(ValueError):
            precond.precondition(np.zeros((0, 0), dtype=np.float64))

    def test_nan_raises(self, precond: MetricSpectralPreconditioner) -> None:
        G = np.eye(2)
        G[0, 1] = np.nan
        with pytest.raises(ValueError, match="no finita|NaN"):
            precond.precondition(G)

    def test_inf_raises(self, precond: MetricSpectralPreconditioner) -> None:
        G = np.eye(2)
        G[1, 1] = np.inf
        with pytest.raises(ValueError, match="no finita|Inf"):
            precond.precondition(G)

    def test_type_error_list(
        self, precond: MetricSpectralPreconditioner
    ) -> None:
        with pytest.raises(TypeError):
            precond.precondition([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_singular_triggers_regularization(
        self,
        precond: MetricSpectralPreconditioner,
        singular_psd: NDArray[np.float64],
    ) -> None:
        pm = precond.precondition(singular_psd)
        assert pm.regularization_applied is True
        assert pm.tikhonov_epsilon > 0.0
        assert pm.null_space_dim >= 1
        assert np.all(pm.eigenvalues_reg > 0)
        assert math.isfinite(pm.condition_number_reg)
        I = np.eye(pm.matrix_dimension)
        r = float(la.norm(pm.G @ pm.G_inv - I, "fro"))
        assert r < 1e-6

    def test_ill_conditioned_regularized_or_stable(
        self,
        precond: MetricSpectralPreconditioner,
        ill_conditioned_spd: NDArray[np.float64],
    ) -> None:
        pm = precond.precondition(ill_conditioned_spd)
        assert np.all(pm.eigenvalues_reg > 0)
        assert math.isfinite(pm.condition_number_reg)
        assert pm.condition_number_reg >= 1.0
        assert pm.inversion_residual <= pm.inversion_tolerance * 10 + 1e-12

    def test_bilateral_inversion_residuals(
        self,
        precond: MetricSpectralPreconditioner,
        random_spd3: NDArray[np.float64],
    ) -> None:
        pm = precond.precondition(random_spd3)
        n = pm.matrix_dimension
        I = np.eye(n)
        scale = math.sqrt(float(n))
        r_l = float(la.norm(pm.G @ pm.G_inv - I, "fro")) / scale
        r_r = float(la.norm(pm.G_inv @ pm.G - I, "fro")) / scale
        assert max(r_l, r_r) <= pm.inversion_tolerance + 1e-15
        assert pm.inversion_residual == pytest.approx(
            max(r_l, r_r), rel=1e-6, abs=1e-15
        )

    def test_cholesky_crosscheck_well_conditioned(
        self,
        eye3: NDArray[np.float64],
    ) -> None:
        pc = MetricSpectralPreconditioner(enable_cholesky_crosscheck=True)
        pm = pc.precondition(eye3)
        # identidad bien condicionada → cross-check disponible y pequeño
        if pm.cholesky_crosscheck_residual is not None:
            assert pm.cholesky_crosscheck_residual < 1e-10

    def test_spectral_gap_identity(
        self, pm_eye4: PreconditionedMetric
    ) -> None:
        assert pm_eye4.spectral_gap_absolute == pytest.approx(1.0, abs=1e-12)
        assert pm_eye4.spectral_gap_cheeger > 0.0

    def test_operator_norms_match_eigenvalues(
        self, pm_diag: PreconditionedMetric
    ) -> None:
        assert pm_diag.operator_norm_G == pytest.approx(
            float(pm_diag.eigenvalues_reg[-1]), abs=1e-12
        )
        assert pm_diag.operator_norm_G_inv == pytest.approx(
            float(1.0 / pm_diag.eigenvalues_reg[0]), rel=1e-10
        )

    def test_G_and_G_inv_symmetric(
        self, pm_diag: PreconditionedMetric
    ) -> None:
        _assert_allclose(pm_diag.G, pm_diag.G.T, atol=1e-14)
        _assert_allclose(pm_diag.G_inv, pm_diag.G_inv.T, atol=1e-14)

    def test_invalid_constructor_params(self) -> None:
        with pytest.raises(ValueError):
            MetricSpectralPreconditioner(symmetry_tolerance=0.0)
        with pytest.raises(ValueError):
            MetricSpectralPreconditioner(condition_threshold=0.5)
        with pytest.raises(ValueError):
            MetricSpectralPreconditioner(tikhonov_epsilon_ratio=-1.0)
        with pytest.raises(ValueError):
            MetricSpectralPreconditioner(target_kappa=0.5)

    def test_zero_metric_regularized(
        self, precond: MetricSpectralPreconditioner
    ) -> None:
        pm = precond.precondition(np.zeros((2, 2), dtype=np.float64))
        assert pm.regularization_applied is True
        assert np.all(pm.eigenvalues_reg > 0)
        assert math.isfinite(pm.condition_number_reg)

    def test_negative_definite_regularized(
        self, precond: MetricSpectralPreconditioner
    ) -> None:
        G = -np.eye(2, dtype=np.float64)
        pm = precond.precondition(G)
        assert pm.regularization_applied is True
        assert np.all(pm.eigenvalues_reg > 0)


# =============================================================================
# FASE 2 — FlatIsomorphism (continuación de PreconditionedMetric)
# =============================================================================


class TestPhase2_FlatIsomorphism:
    """♭ : TM → T*M, normas G, Riesz."""

    def test_init_requires_pm(self) -> None:
        with pytest.raises(TypeError, match="PreconditionedMetric"):
            FlatIsomorphism(np.eye(2))  # type: ignore[arg-type]

    def test_flat_identity_metric(
        self, pm_eye4: PreconditionedMetric, tangent_generic4: TangentVector
    ) -> None:
        flat = FlatIsomorphism(pm_eye4)
        omega = flat.apply_flat_isomorphism(tangent_generic4)
        assert isinstance(omega, CotangentVector)
        _assert_allclose(
            omega.coordinates,
            tangent_generic4.coordinates,
            atol=1e-14,
            msg="♭=id sobre G=I",
        )

    def test_flat_diag_metric(
        self, pm_diag: PreconditionedMetric, diag_metric: NDArray[np.float64]
    ) -> None:
        flat = FlatIsomorphism(pm_diag)
        v = TangentVector(
            coordinates=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        )
        omega = flat.apply_flat_isomorphism(v)
        # ω = G v en la base espectral reconstruida ≈ diag · v en autobase;
        # verificar dualidad vía pairing: ⟨♭v, e_i⟩
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1.0
            # G(v,e) debe = ⟨♭v, e⟩
            G_inner = float(v.coordinates @ pm_diag.G @ e)
            pair = float(omega.coordinates @ e)
            assert pair == pytest.approx(G_inner, abs=1e-10)

    def test_flat_wrong_type_raises(self, pm_eye4: PreconditionedMetric) -> None:
        flat = FlatIsomorphism(pm_eye4)
        with pytest.raises(TypeError, match="TangentVector"):
            flat.apply_flat_isomorphism(
                CotangentVector(coordinates=np.ones(4))  # type: ignore[arg-type]
            )

    def test_flat_dim_mismatch_raises(
        self, pm_eye4: PreconditionedMetric
    ) -> None:
        flat = FlatIsomorphism(pm_eye4)
        bad = TangentVector(coordinates=np.array([1.0, 2.0], dtype=np.float64))
        with pytest.raises(FunctorialityError, match="dimensional"):
            flat.apply_flat_isomorphism(bad)

    def test_g_norm_identity(
        self, pm_eye4: PreconditionedMetric, tangent_generic4: TangentVector
    ) -> None:
        flat = FlatIsomorphism(pm_eye4)
        assert flat.g_norm(tangent_generic4) == pytest.approx(
            tangent_generic4.norm, abs=1e-14
        )

    def test_g_norm_diag(
        self, pm_diag: PreconditionedMetric, diag_metric: NDArray[np.float64]
    ) -> None:
        flat = FlatIsomorphism(pm_diag)
        v = np.array([0.5, -0.2, 0.3, 0.1], dtype=np.float64)
        expected = _g_norm(pm_diag.G, v)
        assert flat.g_norm(v) == pytest.approx(expected, abs=1e-12)

    def test_g_norm_dim_mismatch(self, pm_eye4: PreconditionedMetric) -> None:
        flat = FlatIsomorphism(pm_eye4)
        with pytest.raises(FunctorialityError):
            flat.g_norm(np.array([1.0, 2.0], dtype=np.float64))

    def test_metric_inner_symmetry(
        self, pm_diag: PreconditionedMetric
    ) -> None:
        flat = FlatIsomorphism(pm_diag)
        v = TangentVector(
            coordinates=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        )
        w = TangentVector(
            coordinates=np.array([0.4, -0.1, 0.2, 0.0], dtype=np.float64)
        )
        assert flat.metric_inner(v, w) == pytest.approx(
            flat.metric_inner(w, v), abs=1e-14
        )

    def test_riesz_pairing_identity(
        self, pm_eye4: PreconditionedMetric, tangent_generic4: TangentVector
    ) -> None:
        flat = FlatIsomorphism(pm_eye4)
        w = TangentVector(
            coordinates=np.array([0.2, 0.3, -0.1, 0.5], dtype=np.float64)
        )
        rep = flat.verify_riesz_pairing(tangent_generic4, w)
        assert isinstance(rep, PairingReport)
        assert rep.passed is True
        assert rep.residual <= rep.tolerance
        assert rep.flat_pairing == pytest.approx(rep.G_inner, abs=1e-12)

    def test_riesz_pairing_diag(self, pm_diag: PreconditionedMetric) -> None:
        flat = FlatIsomorphism(pm_diag)
        v = TangentVector(
            coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        w = TangentVector(
            coordinates=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        )
        rep = flat.verify_riesz_pairing(v, w)
        # G diagonal ⇒ G(e0,e1)=0
        assert rep.G_inner == pytest.approx(0.0, abs=1e-12)
        assert rep.passed is True

    def test_diagnostics_report_keys(self, pm_eye4: PreconditionedMetric) -> None:
        flat = FlatIsomorphism(pm_eye4)
        d = flat.diagnostics_report()
        assert d["phase"].startswith("FlatIsomorphism")
        assert "♭" in d["isomorphism"]
        assert "condition_number_reg" in d

    def test_linearity_of_flat(self, pm_diag: PreconditionedMetric) -> None:
        flat = FlatIsomorphism(pm_diag)
        v1 = TangentVector(
            coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        v2 = TangentVector(
            coordinates=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        )
        a, b = 2.5, -1.5
        lhs = flat.apply_flat_isomorphism(
            TangentVector(coordinates=a * v1.coordinates + b * v2.coordinates)
        )
        rhs = (
            a * flat.apply_flat_isomorphism(v1).coordinates
            + b * flat.apply_flat_isomorphism(v2).coordinates
        )
        _assert_allclose(lhs.coordinates, rhs, atol=1e-12)


# =============================================================================
# FASE 3 — SharpIsomorphism + auditoría
# =============================================================================


class TestPhase3_SharpIsomorphism:
    """♯, roundtrips, isometría, Z₂."""

    def test_sharp_identity(
        self, pm_eye4: PreconditionedMetric
    ) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        w = CotangentVector(
            coordinates=np.array([0.7, -0.2, 0.1, 0.3], dtype=np.float64)
        )
        v = sharp.apply_sharp_isomorphism(w)
        _assert_allclose(v.coordinates, w.coordinates, atol=1e-14)

    def test_sharp_wrong_type(self, pm_eye4: PreconditionedMetric) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        with pytest.raises(TypeError, match="CotangentVector"):
            sharp.apply_sharp_isomorphism(
                TangentVector(coordinates=np.ones(4))  # type: ignore[arg-type]
            )

    def test_sharp_dim_mismatch(self, pm_eye4: PreconditionedMetric) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        with pytest.raises(FunctorialityError):
            sharp.apply_sharp_isomorphism(
                CotangentVector(
                    coordinates=np.array([1.0, 2.0], dtype=np.float64)
                )
            )

    def test_roundtrip_sharp_flat_identity(
        self, pm_eye4: PreconditionedMetric, tangent_generic4: TangentVector
    ) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        rep = sharp.verify_roundtrip_identity(tangent_generic4)
        assert isinstance(rep, RoundtripReport)
        assert rep.direction == "sharp_flat"
        assert rep.passed is True
        assert rep.residual <= rep.tolerance
        assert rep.residual < 1e-12

    def test_roundtrip_flat_sharp_identity(
        self, pm_eye4: PreconditionedMetric
    ) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        w = CotangentVector(
            coordinates=np.array([0.2, 0.4, -0.5, 0.1], dtype=np.float64)
        )
        rep = sharp.verify_cotroundtrip_identity(w)
        assert rep.direction == "flat_sharp"
        assert rep.passed is True
        assert rep.residual < 1e-12

    def test_roundtrip_diag_metric(
        self, pm_diag: PreconditionedMetric
    ) -> None:
        sharp = SharpIsomorphism(pm_diag)
        v = TangentVector(
            coordinates=np.array([0.3, -0.7, 0.2, 0.5], dtype=np.float64)
        )
        rep = sharp.verify_roundtrip_identity(v)
        assert rep.passed is True
        # reconstrucción explícita
        omega = sharp.apply_flat_isomorphism(v)
        v2 = sharp.apply_sharp_isomorphism(omega)
        _assert_allclose(v2.coordinates, v.coordinates, atol=1e-10)

    def test_isometry_flat(
        self, pm_diag: PreconditionedMetric
    ) -> None:
        sharp = SharpIsomorphism(pm_diag)
        v = TangentVector(
            coordinates=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        )
        iso = sharp.verify_isometry_flat(v)
        assert iso["passed"] is True
        assert iso["g_norm"] == pytest.approx(
            iso["g_inv_norm_flat"], abs=1e-10
        )

    def test_linearity_of_sharp(self, pm_diag: PreconditionedMetric) -> None:
        sharp = SharpIsomorphism(pm_diag)
        w1 = CotangentVector(
            coordinates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        w2 = CotangentVector(
            coordinates=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        )
        a, b = 1.25, -0.75
        lhs = sharp.apply_sharp_isomorphism(
            CotangentVector(
                coordinates=a * w1.coordinates + b * w2.coordinates
            )
        )
        rhs = (
            a * sharp.apply_sharp_isomorphism(w1).coordinates
            + b * sharp.apply_sharp_isomorphism(w2).coordinates
        )
        _assert_allclose(lhs.coordinates, rhs, atol=1e-12)

    def test_g_inv_norm_consistency(
        self, pm_eye4: PreconditionedMetric
    ) -> None:
        sharp = SharpIsomorphism(pm_eye4)
        w = CotangentVector(
            coordinates=np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
        )
        # G=I ⇒ ‖ω‖_{G⁻¹}=‖ω‖₂
        assert sharp.g_inv_norm(w) == pytest.approx(5.0, abs=1e-14)

    # --- auditoría funtorial ------------------------------------------------

    def test_audit_cov_cov(self) -> None:
        f1, f2 = CovariantFunctor(), CovariantFunctor()
        # dom(f1)=C, cod(f2)=D → incompatibles por defecto
        with pytest.raises(FunctorialityError, match="Composición inválida"):
            SharpIsomorphism.audit_functor_composition(
                f1, f2, verify_domain_compatibility=True
            )
        rep = SharpIsomorphism.audit_functor_composition(
            f1, f2, verify_domain_compatibility=False
        )
        assert rep["result_variance"] == CategoricalVariance.COVARIANT
        assert rep["composition_valid"] is True  # solo se desactivó dom-check

    def test_audit_cont_cont_variance(self) -> None:
        f1, f2 = ContravariantFunctor(), ContravariantFunctor()
        rep = SharpIsomorphism.audit_functor_composition(
            f1, f2, verify_domain_compatibility=False
        )
        assert rep["result_variance"] == CategoricalVariance.COVARIANT

    def test_audit_cov_cont_variance(self) -> None:
        f1 = CovariantFunctor()
        f2 = ContravariantFunctor()
        rep = SharpIsomorphism.audit_functor_composition(
            f1, f2, verify_domain_compatibility=False
        )
        assert rep["result_variance"] == CategoricalVariance.CONTRAVARIANT

    def test_audit_compatible_categories(self) -> None:
        class F2(CovariantFunctor):
            @property
            def codomain_category(self) -> str:
                return "C"  # cod(f2) = dom(f1)

        f1 = CovariantFunctor()  # dom = C
        f2 = F2()
        rep = SharpIsomorphism.audit_functor_composition(
            f1, f2, verify_domain_compatibility=True
        )
        assert rep["domain_compatible"] is True
        assert rep["composition_valid"] is True

    def test_audit_missing_variance_raises(self) -> None:
        class Bare:
            pass

        with pytest.raises(TypeError, match="variance"):
            SharpIsomorphism.audit_functor_composition(
                Bare(), CovariantFunctor(),  # type: ignore[arg-type]
                verify_domain_compatibility=False,
            )


# =============================================================================
# ORQUESTADOR — MusicalIsomorphismEngine
# =============================================================================


class TestEngineOrchestrator:
    def test_construct_default(self) -> None:
        engine = MusicalIsomorphismEngine()
        assert engine.dimension >= 1
        assert engine.preconditioned_metric.matrix_dimension == engine.dimension

    def test_construct_eye(
        self, engine_eye4: MusicalIsomorphismEngine
    ) -> None:
        assert engine_eye4.dimension == 4
        assert engine_eye4.preconditioned_metric.regularization_applied is False

    def test_metric_tensor_copies(
        self, engine_eye4: MusicalIsomorphismEngine
    ) -> None:
        G = engine_eye4.metric_tensor
        G[0, 0] = 999.0
        assert engine_eye4.metric_tensor[0, 0] != 999.0
        Gi = engine_eye4.metric_inverse
        Gi[0, 0] = 999.0
        assert engine_eye4.metric_inverse[0, 0] != 999.0

    def test_flat_sharp_api(
        self,
        engine_diag: MusicalIsomorphismEngine,
        tangent_generic4: TangentVector,
    ) -> None:
        omega = engine_diag.apply_flat_isomorphism(tangent_generic4)
        v2 = engine_diag.apply_sharp_isomorphism(omega)
        _assert_allclose(
            v2.coordinates, tangent_generic4.coordinates, atol=1e-10
        )

    def test_full_cycle_report(
        self,
        engine_diag: MusicalIsomorphismEngine,
        tangent_unit4: TangentVector,
    ) -> None:
        rep = engine_diag.full_cycle_report(tangent_unit4)
        required = {
            "matrix_dimension",
            "condition_number_reg",
            "input_vector_norm",
            "input_vector_g_norm",
            "flat_covector_norm",
            "flat_covector_g_inv_norm",
            "sharp_vector_norm",
            "roundtrip_sharp_flat",
            "roundtrip_flat_sharp",
            "isometry",
            "riesz_pairing",
            "engine",
        }
        assert required.issubset(set(rep.keys()))
        assert rep["roundtrip_sharp_flat"]["passed"] is True
        assert rep["roundtrip_flat_sharp"]["passed"] is True
        assert rep["isometry"]["passed"] is True
        assert rep["riesz_pairing"]["passed"] is True
        assert "v4" in rep["engine"]

    def test_full_cycle_wrong_type(
        self, engine_eye4: MusicalIsomorphismEngine
    ) -> None:
        with pytest.raises(TypeError):
            engine_eye4.full_cycle_report(
                CotangentVector(coordinates=np.ones(4))  # type: ignore[arg-type]
            )

    def test_inject_custom_preconditioner(
        self, eye3: NDArray[np.float64]
    ) -> None:
        pc = MetricSpectralPreconditioner(
            tikhonov_epsilon_ratio=1e-10,
            enable_cholesky_crosscheck=True,
        )
        engine = MusicalIsomorphismEngine(
            metric_tensor=eye3, preconditioner=pc
        )
        assert engine.dimension == 3
        v = TangentVector(
            coordinates=np.array([1.0, 2.0, 3.0], dtype=np.float64)
        )
        assert engine.verify_roundtrip_identity(v).passed

    def test_engine_inherits_phase2_diagnostics(
        self, engine_eye4: MusicalIsomorphismEngine
    ) -> None:
        d = engine_eye4.diagnostics_report()
        assert "condition_number_reg" in d


# =============================================================================
# CONTINUACIÓN FORMAL ENTRE FASES
# =============================================================================


class TestContinuationContract:
    def test_phase1_to_phase2_to_phase3(
        self, random_spd3: NDArray[np.float64]
    ) -> None:
        p1 = MetricSpectralPreconditioner()
        pm = p1.precondition(random_spd3)
        assert isinstance(pm, PreconditionedMetric)

        flat = FlatIsomorphism(pm)
        v = TangentVector(
            coordinates=np.array([0.2, -0.5, 0.7], dtype=np.float64)
        )
        omega = flat.apply_flat_isomorphism(v)
        assert isinstance(omega, CotangentVector)
        pair = flat.verify_riesz_pairing(v, v)
        assert pair.passed

        sharp = SharpIsomorphism(pm)
        v2 = sharp.apply_sharp_isomorphism(omega)
        assert isinstance(v2, TangentVector)
        rt = sharp.verify_roundtrip_identity(v)
        crt = sharp.verify_cotroundtrip_identity(omega)
        assert rt.passed and crt.passed

    def test_engine_consistent_with_manual_pipeline(
        self, diag_metric: NDArray[np.float64]
    ) -> None:
        p1 = MetricSpectralPreconditioner()
        pm = p1.precondition(diag_metric)
        engine = MusicalIsomorphismEngine(metric_tensor=diag_metric)
        _assert_allclose(engine.preconditioned_metric.G, pm.G, atol=1e-12)
        _assert_allclose(
            engine.preconditioned_metric.G_inv, pm.G_inv, atol=1e-12
        )

    def test_orchestrator_uses_same_n(
        self, random_spd3: NDArray[np.float64]
    ) -> None:
        engine = MusicalIsomorphismEngine(metric_tensor=random_spd3)
        assert engine.dimension == 3
        assert engine.preconditioned_metric.matrix_dimension == 3


# =============================================================================
# ESTRÉS NUMÉRICO Y PROPIEDADES ALGEBRAICAS
# =============================================================================


class TestNumericalStress:
    def test_high_dimension_roundtrip(self) -> None:
        n = 24
        G = _spd_from_eig(
            np.linspace(0.5, 5.0, n, dtype=np.float64), seed=5
        )
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        rng = np.random.default_rng(0)
        for _ in range(5):
            v = TangentVector(
                coordinates=rng.normal(size=n).astype(np.float64)
            )
            rep = engine.verify_roundtrip_identity(v)
            assert rep.passed
            crt = engine.verify_cotroundtrip_identity(
                engine.apply_flat_isomorphism(v)
            )
            assert crt.passed

    def test_batch_random_spd_full_cycle(self) -> None:
        rng = np.random.default_rng(42)
        for i in range(10):
            n = int(rng.integers(2, 7))
            ev = rng.uniform(0.2, 4.0, size=n).astype(np.float64)
            G = _spd_from_eig(ev, seed=200 + i)
            engine = MusicalIsomorphismEngine(metric_tensor=G)
            v = TangentVector(
                coordinates=rng.normal(size=n).astype(np.float64)
            )
            report = engine.full_cycle_report(v)
            assert report["roundtrip_sharp_flat"]["passed"]
            assert report["isometry"]["passed"]
            assert report["riesz_pairing"]["passed"]

    def test_ill_conditioned_roundtrip(
        self, ill_conditioned_spd: NDArray[np.float64]
    ) -> None:
        engine = MusicalIsomorphismEngine(metric_tensor=ill_conditioned_spd)
        v = TangentVector(
            coordinates=np.array([1.0, 0.5, -0.3, 0.2], dtype=np.float64)
        )
        # puede haber regularización; roundtrip debe pasar con tol adaptativa
        rep = engine.verify_roundtrip_identity(v)
        assert rep.passed
        assert rep.residual <= rep.tolerance

    def test_singular_metric_engine_usable(
        self, singular_psd: NDArray[np.float64]
    ) -> None:
        engine = MusicalIsomorphismEngine(metric_tensor=singular_psd)
        assert engine.preconditioned_metric.regularization_applied is True
        v = TangentVector(
            coordinates=np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        omega = engine.apply_flat_isomorphism(v)
        v2 = engine.apply_sharp_isomorphism(omega)
        assert np.all(np.isfinite(v2.coordinates))
        assert engine.verify_roundtrip_identity(v).passed

    def test_musical_adjoint_property(
        self, engine_diag: MusicalIsomorphismEngine
    ) -> None:
        r"""
        ⟨♭v, w⟩_eucl = G(v,w) = ⟨v, ♯^{-1} wait⟩:
        más directamente: ⟨♭v, w⟩ = ⟨v, ♭w⟩ por simetría de G.
        """
        v = TangentVector(
            coordinates=np.array([0.5, 0.1, -0.2, 0.3], dtype=np.float64)
        )
        w = TangentVector(
            coordinates=np.array([0.0, 0.4, 0.2, -0.1], dtype=np.float64)
        )
        bv = engine_diag.apply_flat_isomorphism(v)
        bw = engine_diag.apply_flat_isomorphism(w)
        left = float(bv.coordinates @ w.coordinates)
        right = float(v.coordinates @ bw.coordinates)
        assert left == pytest.approx(right, abs=1e-12)

    def test_double_flat_sharp_involution_approx(
        self, engine_diag: MusicalIsomorphismEngine, tangent_generic4: TangentVector
    ) -> None:
        """(♯♭)² ≈ id y (♯♭) ≈ id ⇒ idempotencia del proyector id."""
        v = tangent_generic4
        for _ in range(3):
            v = engine_diag.apply_sharp_isomorphism(
                engine_diag.apply_flat_isomorphism(v)
            )
        _assert_allclose(
            v.coordinates, tangent_generic4.coordinates, atol=1e-9
        )

    def test_lipschitz_bound_flat(
        self, engine_diag: MusicalIsomorphismEngine
    ) -> None:
        r"""‖♭v‖₂ ≤ ‖G‖₂ ‖v‖₂ = λ_max ‖v‖₂."""
        pm = engine_diag.preconditioned_metric
        lam_max = pm.operator_norm_G
        v = TangentVector(
            coordinates=np.array([0.6, -0.8, 0.0, 0.0], dtype=np.float64)
        )
        omega = engine_diag.apply_flat_isomorphism(v)
        assert omega.norm <= lam_max * v.norm + 1e-10

    def test_lipschitz_bound_sharp(
        self, engine_diag: MusicalIsomorphismEngine
    ) -> None:
        r"""‖♯ω‖₂ ≤ ‖G⁻¹‖₂ ‖ω‖₂ = (1/λ_min) ‖ω‖₂."""
        pm = engine_diag.preconditioned_metric
        op_inv = pm.operator_norm_G_inv
        w = CotangentVector(
            coordinates=np.array([0.0, 0.6, 0.8, 0.0], dtype=np.float64)
        )
        v = engine_diag.apply_sharp_isomorphism(w)
        assert v.norm <= op_inv * w.norm + 1e-10


# =============================================================================
# SMOKE / EXPORTS
# =============================================================================


class TestSmokeExports:
    def test_exports(self) -> None:
        import app.core.immune_system.musical_isomorphism_engine as m

        for name in [
            "CategoricalVariance",
            "TangentVector",
            "CotangentVector",
            "PreconditionedMetric",
            "MetricSpectralPreconditioner",
            "FlatIsomorphism",
            "SharpIsomorphism",
            "MusicalIsomorphismEngine",
            "CovariantFunctor",
            "ContravariantFunctor",
            "InversionAudit",
            "RoundtripReport",
            "PairingReport",
            "FunctorialityError",
            "NumericalInstabilityError",
        ]:
            assert hasattr(m, name), f"falta export {name}"

    def test_machine_eps_positive(self) -> None:
        assert _MACHINE_EPSILON > 0.0
        assert _MACHINE_EPSILON < 1e-10


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "--tb=short"])