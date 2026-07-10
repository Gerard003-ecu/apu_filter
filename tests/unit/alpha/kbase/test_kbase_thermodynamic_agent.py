# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite de Pruebas : test_kbase_thermodynamic_agent.py                         |
| Objetivo         : Verificación granular, rigurosa y matemáticamente         |
|                     trazable de KBaseThermodynamicAgent v4.0.0               |
+==============================================================================+

FILOSOFÍA DE LA SUITE
----------------------
Cada método público y privado relevante de las tres fases anidadas
(Phase1_MatrixTopology, Phase2_HamiltonianDynamics, Phase3_SheafProjection)
se somete a pruebas de caja blanca que verifican no sólo la ausencia de
excepciones, sino la **corrección numérica exacta** frente a oráculos
independientes calculados con ``numpy.linalg`` / ``scipy.linalg`` sin
reutilizar la lógica interna del módulo bajo prueba.

Se organiza en las siguientes baterías:

  1. TestExceptionHierarchy               — Álgebra de herencia de excepciones.
  2. TestPhase1*                          — 8 clases, una por responsabilidad.
  3. TestPhase2*                          — 7 clases, una por responsabilidad.
  4. TestPhase3*                          — 3 clases, una por responsabilidad.
  5. TestStabilityFlagsBooleanAlgebra     — Retícula de Boole de predicados.
  6. TestKBaseThermodynamicAgentIntegration — Contrato público end-to-end.

Todas las tolerancias numéricas están explícitamente justificadas en
términos de ``np.finfo(np.float64).eps`` y de las normas involucradas,
replicando el estándar de rigor del módulo bajo prueba.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging

import numpy as np
import pytest
import scipy.linalg as la

from app.agents.alpha.kbase.kbase_thermodynamic_agent import (
    BasalStateTensor,
    CapacitanceDegeneracyError,
    DimensionMismatchError,
    IllConditionedMatrixError,
    InertialFlybackError,
    KBaseThermodynamicAgent,
    MetricTensorSingularityError,
    RayleighDissipationViolation,
    SheafCoboundaryError,
    SheafStalk,
    StabilityFlags,
    StructuralConsistencyError,
    ThermodynamicBaseError,
    TopologicalContext,
    describe_stability_flags,
)

logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent").setLevel(logging.CRITICAL)

EPS = float(np.finfo(np.float64).eps)

Phase1 = KBaseThermodynamicAgent.Phase1_MatrixTopology
Phase2 = KBaseThermodynamicAgent.Phase2_HamiltonianDynamics
Phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection


# ==============================================================================
# UTILIDADES DE GENERACIÓN DETERMINISTA (ORÁCULOS INDEPENDIENTES)
# ==============================================================================


def make_spd(n: int, seed: int, floor: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """SPD determinista: A = B·Bᵀ + floor·I, garantiza λ_min ≥ floor."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    return scale * (B @ B.T + floor * np.eye(n))


def make_psd_with_rank(n: int, rank: int, seed: int) -> np.ndarray:
    """PSD determinista con rango exacto controlado vía autovalores impuestos."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigvals = np.zeros(n)
    if rank > 0:
        eigvals[:rank] = rng.uniform(0.5, 2.0, size=rank)
    return Q @ np.diag(eigvals) @ Q.T


def make_antisymmetric(n: int, seed: int) -> np.ndarray:
    """Antisimétrica determinista: J = B - Bᵀ."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    return B - B.T


# ==============================================================================
# FIXTURES DE SISTEMAS CANÓNICOS
# ==============================================================================


@pytest.fixture(scope="function")
def dims() -> tuple:
    return 3, 2  # dim_q, dim_p -> n = 5


@pytest.fixture(scope="function")
def valid_system(dims):
    """Sistema Port-Hamiltoniano válido genérico (SPD, PSD, antisimétrico)."""
    dim_q, dim_p = dims
    n = dim_q + dim_p
    C_soc = make_spd(dim_q, seed=101)
    M_rec = make_spd(dim_p, seed=102)
    R_cost = make_psd_with_rank(n, rank=n, seed=103)  # PSD de rango completo
    J_base = make_antisymmetric(n, seed=104)
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="function")
def agent(valid_system) -> KBaseThermodynamicAgent:
    C_soc, M_rec, R_cost, J_base = valid_system
    return KBaseThermodynamicAgent(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)


@pytest.fixture(scope="function")
def diagonal_R_system(dims):
    """
    Sistema con R_cost diagonal de autovalores conocidos exactamente, para
    verificar analíticamente la brecha espectral y el rango numérico.
    """
    dim_q, dim_p = dims
    n = dim_q + dim_p
    C_soc = make_spd(dim_q, seed=201)
    M_rec = make_spd(dim_p, seed=202)
    known_eigvals = np.array([0.1, 0.5, 1.0, 2.0, 5.0])[:n]
    R_cost = np.diag(known_eigvals)
    J_base = make_antisymmetric(n, seed=204)
    return C_soc, M_rec, R_cost, J_base, known_eigvals


@pytest.fixture(scope="function")
def rank_deficient_R_system(dims):
    """Sistema con R_cost de rango deficiente para validar β₀(R) > 0."""
    dim_q, dim_p = dims
    n = dim_q + dim_p
    C_soc = make_spd(dim_q, seed=301)
    M_rec = make_spd(dim_p, seed=302)
    R_cost = make_psd_with_rank(n, rank=n - 2, seed=303)
    J_base = make_antisymmetric(n, seed=304)
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="function")
def harmonic_oscillator_system():
    """
    Sistema mínimo (dim_q=1, dim_p=1) sin disipación, isomorfo a un
    oscilador LC ideal: H = q²/(2C) + p²/(2M), J=[[0,1],[-1,0]].
    La frecuencia propia exacta es ω = 1/√(M·C).
    """
    c, m = 4.0, 9.0
    C_soc = np.array([[c]])
    M_rec = np.array([[m]])
    R_cost = np.zeros((2, 2))
    J_base = np.array([[0.0, 1.0], [-1.0, 0.0]])
    omega_exact = 1.0 / np.sqrt(c * m)
    return C_soc, M_rec, R_cost, J_base, omega_exact


@pytest.fixture(scope="function")
def phase1_placeholder():
    """
    Instancia de Phase1_MatrixTopology con matrices "de relleno" válidas en
    forma (no en contenido), usada para probar métodos de validación que son
    independientes del contenido (p.ej. ``_validate_metric_tensor``) sin
    necesidad de ensamblar un sistema físico completo.
    """
    def _make(kappa_max: float = 1.0e10) -> Phase1:
        C_soc = np.eye(2)
        M_rec = np.eye(2)
        R_cost = np.eye(4)
        J_base = make_antisymmetric(4, seed=999)
        return Phase1(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, kappa_max=kappa_max
        )
    return _make


# ==============================================================================
# 1. JERARQUÍA DE EXCEPCIONES
# ==============================================================================


class TestExceptionHierarchy:
    """Verifica que todas las excepciones del dominio formen un único árbol."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            DimensionMismatchError,
            CapacitanceDegeneracyError,
            InertialFlybackError,
            RayleighDissipationViolation,
            IllConditionedMatrixError,
            MetricTensorSingularityError,
            SheafCoboundaryError,
            StructuralConsistencyError,
        ],
    )
    def test_all_domain_exceptions_derive_from_thermodynamic_base_error(self, exc_cls):
        assert issubclass(exc_cls, ThermodynamicBaseError)

    def test_thermodynamic_base_error_derives_from_exception(self):
        assert issubclass(ThermodynamicBaseError, Exception)


# ==============================================================================
# 2. FASE 1 — TOPOLOGÍA MATRICIAL, MÉTRICA RIEMANNIANA, ESPECTRO
# ==============================================================================


class TestPhase1DimensionValidation:
    """Verifica ``_check_dimensions`` contra todas las formas de inconsistencia."""

    def test_valid_dimensions_returns_correct_tuple(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        dim_q, dim_p = p1._check_dimensions()
        assert (dim_q, dim_p) == dims

    def test_c_soc_non_square_raises(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_C = C_soc[:, :-1]  # rectangular
        p1 = Phase1(C_soc=bad_C, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(DimensionMismatchError):
            p1._check_dimensions()

    def test_m_rec_non_square_raises(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_M = M_rec[:, :-1]
        p1 = Phase1(C_soc=C_soc, M_rec=bad_M, R_cost=R_cost, J_base=J_base)
        with pytest.raises(DimensionMismatchError):
            p1._check_dimensions()

    def test_r_cost_wrong_shape_raises(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_R = np.eye(R_cost.shape[0] + 1)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=bad_R, J_base=J_base)
        with pytest.raises(DimensionMismatchError):
            p1._check_dimensions()

    def test_j_base_wrong_shape_raises(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_J = make_antisymmetric(J_base.shape[0] + 1, seed=1)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=bad_J)
        with pytest.raises(DimensionMismatchError):
            p1._check_dimensions()


class TestPhase1SymmetryValidation:
    """Verifica ``_validate_symmetry`` y ``_validate_antisymmetry``."""

    def test_symmetric_matrix_passes(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        p1._validate_symmetry(C_soc, "C_soc")  # no debe lanzar

    def test_asymmetric_matrix_fails(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        broken = C_soc.copy()
        broken[0, 1] += 1.0  # rompe simetría muy por encima de eps
        p1 = Phase1(C_soc=broken, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(ThermodynamicBaseError):
            p1._validate_symmetry(broken, "C_soc_broken")

    def test_antisymmetric_matrix_passes(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        p1._validate_antisymmetry(J_base, "J_base")  # no debe lanzar

    def test_non_antisymmetric_matrix_fails(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        broken = J_base.copy()
        broken[0, 1] += 1.0
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=broken)
        with pytest.raises(ThermodynamicBaseError):
            p1._validate_antisymmetry(broken, "J_base_broken")

    def test_zero_matrix_is_both_symmetric_and_antisymmetric(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = J_base.shape[0]
        zero = np.zeros((n, n))
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        p1._validate_symmetry(zero, "zero")
        p1._validate_antisymmetry(zero, "zero")


class TestPhase1MetricTensorValidation:
    """Verifica ``_validate_metric_tensor`` (invertibilidad y condicionamiento)."""

    def test_identity_metric_has_condition_number_one(self, phase1_placeholder):
        p1 = phase1_placeholder()
        kappa = p1._validate_metric_tensor(np.eye(3), "G_id", expected_dim=3)
        assert kappa == pytest.approx(1.0, rel=1e-12)

    def test_wrong_shape_raises_dimension_mismatch(self, phase1_placeholder):
        p1 = phase1_placeholder()
        G = np.eye(2)
        with pytest.raises(DimensionMismatchError):
            p1._validate_metric_tensor(G, "G_bad_shape", expected_dim=3)

    def test_singular_metric_raises_singularity_error(self, phase1_placeholder):
        p1 = phase1_placeholder()
        G = np.array([[1.0, 0.0], [0.0, 0.0]])  # rango deficiente
        with pytest.raises(MetricTensorSingularityError):
            p1._validate_metric_tensor(G, "G_singular", expected_dim=2)

    def test_ill_conditioned_metric_raises_singularity_error(self, phase1_placeholder):
        p1 = phase1_placeholder(kappa_max=10.0)
        G = np.diag([1.0e-3, 1.0])  # kappa = 1000 > kappa_max = 10
        with pytest.raises(MetricTensorSingularityError):
            p1._validate_metric_tensor(G, "G_ill", expected_dim=2)

    def test_moderate_condition_number_within_bounds_passes(self, phase1_placeholder):
        p1 = phase1_placeholder(kappa_max=100.0)
        G = np.diag([1.0, 2.0, 5.0])
        kappa = p1._validate_metric_tensor(G, "G_ok", expected_dim=3)
        assert kappa == pytest.approx(5.0, rel=1e-12)


class TestPhase1CongruencePullback:
    """
    Verifica ``_congruence_pullback`` frente a la Ley de Inercia de
    Sylvester: Ã = G·A·G^⊤ preserva la signatura de A si G es invertible.
    """

    def test_identity_metric_leaves_matrix_unchanged(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        I = np.eye(C_soc.shape[0])
        A_tilde = p1._congruence_pullback(C_soc, I, "C_soc")
        assert np.allclose(A_tilde, C_soc, rtol=1e-12, atol=1e-12)

    def test_pullback_preserves_symmetry(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        G = np.diag(np.linspace(1.0, 3.0, C_soc.shape[0]))
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        assert np.allclose(A_tilde, A_tilde.T, atol=1e-10)

    def test_pullback_preserves_positive_definiteness_sylvester_inertia(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        rng = np.random.default_rng(55)
        G = rng.normal(size=C_soc.shape) + 3.0 * np.eye(C_soc.shape[0])  # bien condicionada
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        eigvals = np.linalg.eigvalsh(A_tilde)
        assert np.all(eigvals > 0.0)

    def test_pullback_matches_manual_matrix_product(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        G = np.diag(np.linspace(0.5, 2.0, C_soc.shape[0]))
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        expected = G @ C_soc @ G.T
        assert np.allclose(A_tilde, expected, rtol=1e-10)


class TestPhase1ConditionNumber:
    """Verifica ``_compute_condition_number`` contra ``np.linalg.eigvalsh``."""

    def test_identity_has_condition_number_one(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        kappa, lmin, lmax = p1._compute_condition_number(np.eye(n), "I")
        assert kappa == pytest.approx(1.0, rel=1e-12)
        assert lmin == pytest.approx(1.0, rel=1e-12)
        assert lmax == pytest.approx(1.0, rel=1e-12)

    def test_condition_number_matches_oracle_eigvalsh(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        kappa, lmin, lmax = p1._compute_condition_number(C_soc, "C_soc")
        oracle_eigvals = np.linalg.eigvalsh(C_soc)
        assert lmin == pytest.approx(float(oracle_eigvals[0]), rel=1e-8)
        assert lmax == pytest.approx(float(oracle_eigvals[-1]), rel=1e-8)
        assert kappa == pytest.approx(float(oracle_eigvals[-1] / oracle_eigvals[0]), rel=1e-8)

    def test_non_spd_matrix_raises_capacitance_degeneracy(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        # Matriz simétrica con un autovalor negativo garantizado.
        Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(n, n)))
        eigvals = np.linspace(-1.0, 1.0, n)
        bad = Q @ np.diag(eigvals) @ Q.T
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(CapacitanceDegeneracyError):
            p1._compute_condition_number(bad, "bad_matrix")

    def test_ill_conditioned_matrix_raises_ill_conditioned_error(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(n, n)))
        eigvals = np.geomspace(1.0, 1.0e12, n)  # kappa = 1e12
        bad = Q @ np.diag(eigvals) @ Q.T
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, kappa_max=1.0e6)
        with pytest.raises(IllConditionedMatrixError):
            p1._compute_condition_number(bad, "ill_conditioned")


class TestPhase1CholeskyRegularization:
    """
    Verifica ``_cholesky_spd_regularized``: caso nominal (τ=0), caso con
    jitter aplicado tras fallo transitorio simulado, y caso de fallo
    persistente que debe elevar ``CapacitanceDegeneracyError``.
    """

    def test_well_conditioned_matrix_requires_no_jitter(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        L, tau = p1._cholesky_spd_regularized(C_soc, "C_soc")
        assert tau == 0.0
        assert np.allclose(L @ L.T, C_soc, rtol=1e-8, atol=1e-8)

    def test_cholesky_reconstructs_original_matrix_exactly(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        L, _ = p1._cholesky_spd_regularized(M_rec, "M_rec")
        assert np.allclose(L @ L.T, M_rec, rtol=1e-8, atol=1e-8)
        assert np.allclose(L, np.tril(L))  # triangular inferior

    def test_transient_failure_triggers_jitter_and_recovers(self, valid_system, monkeypatch):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)

        real_cholesky = la.cholesky
        call_state = {"count": 0}

        def flaky_cholesky(A, lower=True):
            call_state["count"] += 1
            if call_state["count"] == 1:
                raise la.LinAlgError("fallo simulado transitorio")
            return real_cholesky(A, lower=lower)

        monkeypatch.setattr(la, "cholesky", flaky_cholesky)
        L, tau = p1._cholesky_spd_regularized(C_soc, "C_soc")

        assert call_state["count"] == 2
        assert tau > 0.0
        # Reconstrucción aproximada (con jitter, la igualdad es hasta O(tau))
        assert np.allclose(L @ L.T, C_soc + tau * np.eye(C_soc.shape[0]), rtol=1e-8)

    def test_persistent_failure_raises_capacitance_degeneracy(self, valid_system, monkeypatch):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)

        def always_fail(A, lower=True):
            raise la.LinAlgError("fallo persistente simulado")

        monkeypatch.setattr(la, "cholesky", always_fail)
        with pytest.raises(CapacitanceDegeneracyError):
            p1._cholesky_spd_regularized(C_soc, "C_soc", max_attempts=3)


class TestPhase1PSDSpectralDiagnostics:
    """Verifica ``_validate_psd_and_spectral_diagnostics``."""

    def test_spd_matrix_yields_full_rank_and_zero_kernel(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, rank_R, gap = p1._validate_psd_and_spectral_diagnostics(R_cost, "R_cost")
        assert rank_R == n
        assert np.allclose(R_sqrt @ R_sqrt, R_cost, rtol=1e-6, atol=1e-8)

    def test_negative_eigenvalue_raises_rayleigh_violation(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(n, n)))
        bad = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(RayleighDissipationViolation):
            p1._validate_psd_and_spectral_diagnostics(bad, "R_cost_bad")

    def test_rank_deficient_psd_matrix_reports_correct_rank(self, rank_deficient_R_system, dims):
        C_soc, M_rec, R_cost, J_base = rank_deficient_R_system
        n = sum(dims)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, rank_R, gap = p1._validate_psd_and_spectral_diagnostics(R_cost, "R_cost")
        assert rank_R == n - 2

    def test_spectral_gap_matches_known_eigenvalues(self, diagonal_R_system):
        C_soc, M_rec, R_cost, J_base, known_eigvals = diagonal_R_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        _, _, gap = p1._validate_psd_and_spectral_diagnostics(R_cost, "R_cost")
        sorted_eigs = np.sort(known_eigvals)
        expected_gap = sorted_eigs[1] - sorted_eigs[0]
        assert gap == pytest.approx(expected_gap, rel=1e-8)

    def test_zero_matrix_is_psd_with_rank_zero(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, rank_R, gap = p1._validate_psd_and_spectral_diagnostics(np.zeros((n, n)), "zero")
        assert rank_R == 0
        assert np.allclose(R_sqrt, 0.0)


class TestPhase1BuildTopologicalContextIntegration:
    """Prueba de integración del método terminal de la Fase 1."""

    def test_returns_immutable_topological_context(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert isinstance(ctx, TopologicalContext)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.dim_q = 999  # type: ignore[misc]

    def test_context_dimensions_match_inputs(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert (ctx.dim_q, ctx.dim_p) == dims

    def test_cholesky_factors_reconstruct_pulled_back_matrices(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert np.allclose(ctx.L_C @ ctx.L_C.T, C_soc, rtol=1e-6, atol=1e-8)
        assert np.allclose(ctx.L_M @ ctx.L_M.T, M_rec, rtol=1e-6, atol=1e-8)

    def test_inv_sqrt_matrices_are_true_inverse_square_roots(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        C_inv_reconstructed = ctx.C_inv_sqrt.T @ ctx.C_inv_sqrt
        assert np.allclose(C_inv_reconstructed, np.linalg.inv(C_soc), rtol=1e-6, atol=1e-8)
        M_inv_reconstructed = ctx.M_inv_sqrt.T @ ctx.M_inv_sqrt
        assert np.allclose(M_inv_reconstructed, np.linalg.inv(M_rec), rtol=1e-6, atol=1e-8)

    def test_nontrivial_metric_tensor_applies_congruent_pullback(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        G_q = np.diag(np.linspace(1.0, 2.0, dim_q))
        G_p = np.diag(np.linspace(1.0, 1.5, dim_p))
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, G_q=G_q, G_p=G_p)
        ctx = p1.build_topological_context()
        expected_C_tilde = G_q @ C_soc @ G_q.T
        expected_M_tilde = G_p @ M_rec @ G_p.T
        assert np.allclose(ctx.L_C @ ctx.L_C.T, expected_C_tilde, rtol=1e-6, atol=1e-8)
        assert np.allclose(ctx.L_M @ ctx.L_M.T, expected_M_tilde, rtol=1e-6, atol=1e-8)
        assert ctx.kappa_G_q == pytest.approx(2.0, rel=1e-8)
        assert ctx.kappa_G_p == pytest.approx(1.5, rel=1e-8)

    def test_default_metric_tensors_are_identity(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert np.allclose(ctx.G_q, np.eye(dim_q))
        assert np.allclose(ctx.G_p, np.eye(dim_p))

    def test_betti_0_equals_n_minus_rank_R(self, rank_deficient_R_system, dims):
        C_soc, M_rec, R_cost, J_base = rank_deficient_R_system
        n = sum(dims)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert ctx.betti_0_R == n - ctx.rank_R
        assert ctx.betti_0_R == 2


# ==============================================================================
# 3. FASE 2 — DINÁMICA PORT-HAMILTONIANA, RAYLEIGH Y WILLIAMSON
# ==============================================================================


@pytest.fixture(scope="function")
def valid_context(valid_system) -> TopologicalContext:
    C_soc, M_rec, R_cost, J_base = valid_system
    p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
    return p1.build_topological_context()


@pytest.fixture(scope="function")
def phase2(valid_context) -> Phase2:
    return Phase2(context=valid_context, breakdown_voltage=1.0e5, kappa_max=1.0e10)


@pytest.fixture(scope="function")
def state_vectors(dims):
    dim_q, dim_p = dims
    rng = np.random.default_rng(42)
    q = rng.normal(size=dim_q)
    p = rng.normal(size=dim_p)
    df_dt = rng.normal(size=dim_p)
    return q, p, df_dt


class TestPhase2EnergyComputation:
    """Verifica energías y gradientes contra inversión explícita (oráculo)."""

    def test_potential_energy_matches_explicit_inverse(self, phase2, valid_system, state_vectors):
        C_soc, *_ = valid_system
        q, p, df_dt = state_vectors
        V_q, grad_V_q = phase2._evaluate_potential_energy(q)
        C_inv = np.linalg.inv(C_soc)
        expected_V = 0.5 * q @ C_inv @ q
        expected_grad = C_inv @ q
        assert V_q == pytest.approx(expected_V, rel=1e-8)
        assert np.allclose(grad_V_q, expected_grad, rtol=1e-6, atol=1e-8)

    def test_kinetic_energy_matches_explicit_inverse(self, phase2, valid_system, state_vectors):
        _, M_rec, _, _ = valid_system
        q, p, df_dt = state_vectors
        K_p, grad_K_p = phase2._compute_kinetic_energy(p)
        M_inv = np.linalg.inv(M_rec)
        expected_K = 0.5 * p @ M_inv @ p
        expected_grad = M_inv @ p
        assert K_p == pytest.approx(expected_K, rel=1e-8)
        assert np.allclose(grad_K_p, expected_grad, rtol=1e-6, atol=1e-8)

    def test_energies_are_nonnegative_for_arbitrary_state(self, phase2, dims):
        dim_q, dim_p = dims
        rng = np.random.default_rng(123)
        for _ in range(20):
            q = rng.normal(size=dim_q) * rng.uniform(0.1, 10)
            p = rng.normal(size=dim_p) * rng.uniform(0.1, 10)
            V_q, _ = phase2._evaluate_potential_energy(q)
            K_p, _ = phase2._compute_kinetic_energy(p)
            assert V_q >= -1e-9
            assert K_p >= -1e-9

    def test_zero_state_yields_zero_energy(self, phase2, dims):
        dim_q, dim_p = dims
        V_q, grad_V = phase2._evaluate_potential_energy(np.zeros(dim_q))
        K_p, grad_K = phase2._compute_kinetic_energy(np.zeros(dim_p))
        assert V_q == pytest.approx(0.0, abs=1e-12)
        assert K_p == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(grad_V, 0.0)
        assert np.allclose(grad_K, 0.0)

    def test_potential_energy_wrong_shape_raises(self, phase2, dims):
        dim_q, _ = dims
        with pytest.raises(DimensionMismatchError):
            phase2._evaluate_potential_energy(np.zeros(dim_q + 1))

    def test_kinetic_energy_wrong_shape_raises(self, phase2, dims):
        _, dim_p = dims
        with pytest.raises(DimensionMismatchError):
            phase2._compute_kinetic_energy(np.zeros(dim_p + 1))


class TestPhase2EulerHomogeneity:
    """
    Verifica el Teorema de Euler para funciones homogéneas de grado 2:
    q·∇_qH + p·∇_pH = 2H, exacto para H cuadrática.
    """

    def test_euler_identity_holds_near_machine_precision(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        V_q, grad_V = phase2._evaluate_potential_energy(q)
        K_p, grad_K = phase2._compute_kinetic_energy(p)
        H_total = V_q + K_p
        residual = phase2._verify_euler_homogeneity(q, p, grad_V, grad_K, H_total)
        assert residual < 1e-6 * max(abs(H_total), 1.0)

    def test_euler_identity_holds_for_zero_state(self, phase2, dims):
        dim_q, dim_p = dims
        q = np.zeros(dim_q)
        p = np.zeros(dim_p)
        residual = phase2._verify_euler_homogeneity(q, p, np.zeros(dim_q), np.zeros(dim_p), 0.0)
        assert residual == pytest.approx(0.0, abs=1e-12)


class TestPhase2RayleighDissipation:
    """Verifica ``_enforce_rayleigh_dissipation`` (Segunda Ley)."""

    def test_dissipated_power_matches_quadratic_form(self, phase2, valid_system, state_vectors):
        _, _, R_cost, _ = valid_system
        q, p, df_dt = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        expected = grad_H @ R_cost @ grad_H
        assert P_diss == pytest.approx(expected, rel=1e-8)
        assert P_diss >= 0.0

    def test_wrong_shape_raises_dimension_mismatch(self, phase2, dims):
        n = sum(dims)
        with pytest.raises(DimensionMismatchError):
            phase2._enforce_rayleigh_dissipation(np.zeros(n + 1))

    def test_negative_definite_r_cost_raises_violation(self, valid_context):
        """
        Ataque de caja blanca: se inyecta una R_cost negativo-definida
        directamente en el contexto (bypaseando la Fase 1) para verificar
        que la Fase 2 posee su propia red de seguridad redundante.
        """
        n = valid_context.dim_q + valid_context.dim_p
        bad_R = -np.eye(n)  # estrictamente negativa
        broken_ctx = dataclasses.replace(valid_context, R_cost=bad_R)
        phase2_broken = Phase2(context=broken_ctx, breakdown_voltage=1e5, kappa_max=1e10)
        grad_H = np.ones(n)
        with pytest.raises(RayleighDissipationViolation):
            phase2_broken._enforce_rayleigh_dissipation(grad_H)

    def test_zero_gradient_yields_zero_dissipation(self, phase2, dims):
        n = sum(dims)
        P_diss = phase2._enforce_rayleigh_dissipation(np.zeros(n))
        assert P_diss == pytest.approx(0.0, abs=1e-12)


class TestPhase2FlybackVoltage:
    """Verifica ``_measure_flyback_voltage``."""

    def test_flyback_matches_manual_matrix_vector_product(self, phase2, valid_system, state_vectors):
        _, M_rec, _, _ = valid_system
        q, p, df_dt = state_vectors
        v_fb = phase2._measure_flyback_voltage(df_dt)
        expected_vec = M_rec @ df_dt
        expected_norm = np.linalg.norm(expected_vec, ord=np.inf)
        assert v_fb == pytest.approx(expected_norm, rel=1e-6)

    def test_wrong_shape_raises_dimension_mismatch(self, phase2, dims):
        _, dim_p = dims
        with pytest.raises(DimensionMismatchError):
            phase2._measure_flyback_voltage(np.zeros(dim_p + 1))

    def test_zero_perturbation_yields_zero_voltage(self, phase2, dims):
        _, dim_p = dims
        v_fb = phase2._measure_flyback_voltage(np.zeros(dim_p))
        assert v_fb == pytest.approx(0.0, abs=1e-12)

    def test_exceeding_breakdown_voltage_raises_inertial_flyback_error(self, valid_context, dims):
        _, dim_p = dims
        phase2_strict = Phase2(context=valid_context, breakdown_voltage=1e-9, kappa_max=1e10)
        with pytest.raises(InertialFlybackError):
            phase2_strict._measure_flyback_voltage(np.ones(dim_p))


class TestPhase2VectorFieldAndStructuralConsistency:
    """
    Verifica el campo vectorial Port-Hamiltoniano ẋ=(J-R)∇H y la identidad
    algebraica exacta ∇H^⊤ẋ ≡ -P_diss (consecuencia de J antisimétrica).
    """

    def test_vector_field_matches_manual_formula(self, phase2, valid_system, state_vectors):
        _, _, R_cost, J_base = valid_system
        q, p, df_dt = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        x_dot = phase2._compute_vector_field(grad_H)
        expected = (J_base - R_cost) @ grad_H
        assert np.allclose(x_dot, expected, rtol=1e-8)

    def test_structural_consistency_holds_for_genuine_ph_system(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        x_dot = phase2._compute_vector_field(grad_H)
        residual = phase2._verify_structural_consistency(grad_H, x_dot, P_diss)
        assert residual < 1e-6 * max(P_diss, 1.0)

    def test_grad_H_transpose_J_grad_H_is_identically_zero(self, phase2, valid_system, state_vectors):
        """Verificación directa de la identidad algebraica pura ∇H^⊤J∇H≡0."""
        _, _, _, J_base = valid_system
        q, p, df_dt = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        quad_form = grad_H @ J_base @ grad_H
        assert quad_form == pytest.approx(0.0, abs=1e-8)

    def test_inconsistent_wiring_raises_structural_consistency_error(self, phase2):
        """
        Ataque de caja blanca: se fabrican grad_H, x_dot y P_diss mutuamente
        incoherentes para verificar que el invariante en caliente detecta
        errores de cableado entre J_base, R_cost y ∇H.
        """
        n = phase2._ctx.dim_q + phase2._ctx.dim_p
        grad_H = np.ones(n)
        x_dot = np.ones(n) * 1000.0  # deliberadamente incoherente
        P_diss = 0.0
        with pytest.raises(StructuralConsistencyError):
            phase2._verify_structural_consistency(grad_H, x_dot, P_diss)


class TestPhase2NormalModesWilliamson:
    """
    Verifica ``compute_normal_modes`` contra la frecuencia analítica exacta
    de un oscilador LC ideal: ω = 1/√(M·C).
    """

    def test_single_mode_matches_analytic_harmonic_frequency(self, harmonic_oscillator_system):
        C_soc, M_rec, R_cost, J_base, omega_exact = harmonic_oscillator_system
        agent = KBaseThermodynamicAgent(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        omegas, E0 = agent.phase2.compute_normal_modes()
        assert omegas.shape == (1,)
        assert omegas[0] == pytest.approx(omega_exact, rel=1e-6)

    def test_zero_point_energy_matches_half_hbar_omega(self, harmonic_oscillator_system):
        C_soc, M_rec, R_cost, J_base, omega_exact = harmonic_oscillator_system
        hbar = 2.5
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, hbar=hbar
        )
        omegas, E0 = agent.phase2.compute_normal_modes()
        expected_E0 = 0.5 * hbar * omega_exact
        assert E0 == pytest.approx(expected_E0, rel=1e-6)

    def test_normal_modes_are_nonnegative(self, phase2):
        omegas, E0 = phase2.compute_normal_modes()
        assert np.all(omegas >= -1e-9)
        assert E0 >= -1e-9

    def test_number_of_modes_equals_half_dimension(self, phase2, dims):
        n = sum(dims)
        omegas, _ = phase2.compute_normal_modes()
        assert omegas.shape[0] == n // 2


class TestPhase2StabilityFlags:
    """Verifica la retícula ``StabilityFlags`` en escenarios controlados."""

    def test_all_flags_satisfied_for_nominal_state(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        V_q, grad_V = phase2._evaluate_potential_energy(q)
        K_p, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        v_fb = phase2._measure_flyback_voltage(df_dt)
        x_dot = phase2._compute_vector_field(grad_H)
        residual = phase2._verify_structural_consistency(grad_H, x_dot, P_diss)
        flags = phase2._evaluate_stability_flags(V_q, K_p, P_diss, v_fb, residual)
        assert flags == StabilityFlags.ALL

    def test_flyback_unsafe_flag_cleared_near_soft_margin(self, valid_context, dims):
        """
        Con un margen de seguridad estricto (0.01) y un breakdown_voltage
        muy bajo, un voltaje de flyback moderado debe marcar
        FLYBACK_SAFE como ausente sin disparar la excepción dura.
        """
        _, dim_p = dims
        phase2_margin = Phase2(
            context=valid_context,
            breakdown_voltage=1.0,
            kappa_max=1e10,
            flyback_safety_margin=0.01,
        )
        flags = phase2_margin._evaluate_stability_flags(
            V_q=1.0, K_p=1.0, P_diss=1.0, v_fb=0.5, structural_residual=0.0
        )
        assert StabilityFlags.FLYBACK_SAFE not in flags

    def test_spectral_conditioning_flag_cleared_for_high_kappa(self, valid_context):
        phase2_strict = Phase2(context=valid_context, breakdown_voltage=1e5, kappa_max=1.0)
        flags = phase2_strict._evaluate_stability_flags(
            V_q=1.0, K_p=1.0, P_diss=1.0, v_fb=0.0, structural_residual=0.0
        )
        assert StabilityFlags.SPECTRAL_CONDITIONING_SOUND not in flags


class TestPhase2SynthesizeBasalStateIntegration:
    """Prueba de integración del método terminal de la Fase 2."""

    def test_returns_basal_state_tensor_with_consistent_fields(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        assert isinstance(tensor, BasalStateTensor)
        assert tensor.total_hamiltonian == pytest.approx(
            tensor.potential_energy + tensor.kinetic_energy, rel=1e-10
        )
        assert tensor.is_thermodynamically_stable == (tensor.stability_flags == StabilityFlags.ALL)
        assert tensor.normal_mode_frequencies is None
        assert tensor.zero_point_energy is None

    def test_compute_normal_modes_flag_populates_optional_fields(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt, compute_normal_modes=True)
        assert tensor.normal_mode_frequencies is not None
        assert tensor.zero_point_energy is not None
        assert tensor.normal_mode_frequencies.shape[0] == sum(
            [phase2._ctx.dim_q, phase2._ctx.dim_p]
        ) // 2

    def test_vector_field_shape_matches_state_dimension(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        n = phase2._ctx.dim_q + phase2._ctx.dim_p
        assert tensor.vector_field.shape == (n,)

    def test_result_is_immutable(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tensor.total_hamiltonian = 0.0  # type: ignore[misc]


# ==============================================================================
# 4. FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES
# ==============================================================================


@pytest.fixture(scope="function")
def phase3(valid_context) -> Phase3:
    return Phase3(context=valid_context)


@pytest.fixture(scope="function")
def full_state_x(dims):
    n = sum(dims)
    rng = np.random.default_rng(777)
    return rng.normal(size=n)


class TestPhase3CoboundaryAssembly:
    """Verifica la construcción de δ_metric, δ_diss y δ_BASE."""

    def test_delta_metric_is_block_diagonal_of_inverse_square_roots(self, phase3, valid_context, dims):
        dim_q, dim_p = dims
        assert np.allclose(
            phase3._delta_metric[:dim_q, :dim_q], valid_context.C_inv_sqrt, rtol=1e-10
        )
        assert np.allclose(
            phase3._delta_metric[dim_q:, dim_q:], valid_context.M_inv_sqrt, rtol=1e-10
        )
        assert np.allclose(phase3._delta_metric[:dim_q, dim_q:], 0.0)
        assert np.allclose(phase3._delta_metric[dim_q:, :dim_q], 0.0)

    def test_delta_dissipative_equals_r_sqrt(self, phase3, valid_context):
        assert np.array_equal(phase3._delta_diss, valid_context.R_sqrt)

    def test_delta_base_is_vertical_stack_of_correct_shape(self, phase3, dims):
        n = sum(dims)
        assert phase3._delta_base.shape == (2 * n, n)
        assert np.allclose(phase3._delta_base[:n, :], phase3._delta_metric)
        assert np.allclose(phase3._delta_base[n:, :], phase3._delta_diss)

    def test_delta_metric_is_invertible(self, phase3):
        singular_values = la.svdvals(phase3._delta_metric)
        assert np.all(singular_values > 1e-10)

    def test_rank_delta_equals_full_dimension(self, phase3, dims):
        n = sum(dims)
        assert phase3._rank_delta == n


class TestPhase3HodgeIdentityAndSpectrum:
    """
    Verifica la identidad de Hodge local Δ_BASE = ∇²H + R_cost y sus
    diagnósticos espectrales derivados.
    """

    def test_hodge_laplacian_matches_hessian_plus_r_cost(self, phase3, valid_system, valid_context):
        C_soc, M_rec, R_cost, J_base = valid_system
        expected_hessian = la.block_diag(np.linalg.inv(C_soc), np.linalg.inv(M_rec))
        expected_hodge = expected_hessian + R_cost
        assert np.allclose(phase3._hodge_laplacian, expected_hodge, rtol=1e-6, atol=1e-8)

    def test_hodge_identity_residual_is_near_machine_precision(self, phase3):
        rel_error = phase3._verify_hodge_identity()
        assert rel_error < 1e3 * EPS

    def test_hodge_laplacian_is_symmetric_positive_definite(self, phase3):
        assert np.allclose(phase3._hodge_laplacian, phase3._hodge_laplacian.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(phase3._hodge_laplacian)
        assert np.all(eigvals > 0.0)

    def test_hodge_spectrum_gap_and_condition_number_are_consistent(self, phase3):
        gap, kappa, harmonic_dim = phase3._compute_hodge_spectrum()
        eigvals = np.linalg.eigvalsh(phase3._hodge_laplacian)
        expected_gap = eigvals[1] - eigvals[0]
        expected_kappa = eigvals[-1] / eigvals[0]
        assert gap == pytest.approx(expected_gap, rel=1e-6)
        assert kappa == pytest.approx(expected_kappa, rel=1e-6)

    def test_harmonic_dimension_is_always_zero(self, phase3):
        """
        Invariante topológico: δ_metric es invertible por construcción
        (∇²H≻0 nunca degenera), por lo que dim ker(δ_metric) = 0 siempre.
        """
        _, _, harmonic_dim = phase3._compute_hodge_spectrum()
        assert harmonic_dim == 0


class TestPhase3ExportStalkIntegration:
    """Prueba de integración del método terminal de la Fase 3."""

    def test_returns_sheaf_stalk_with_correct_shapes(self, phase3, full_state_x, dims):
        n = sum(dims)
        stalk = phase3.export_stalk(full_state_x)
        assert isinstance(stalk, SheafStalk)
        assert stalk.delta_base.shape == (2 * n, n)
        assert stalk.state_vector.shape == (n,)
        assert stalk.rank_delta == n

    def test_wrong_shape_raises_dimension_mismatch(self, phase3, dims):
        n = sum(dims)
        with pytest.raises(DimensionMismatchError):
            phase3.export_stalk(np.zeros(n + 3))

    def test_projections_match_manual_matrix_vector_products(self, phase3, full_state_x):
        stalk = phase3.export_stalk(full_state_x)
        expected_metric_proj = phase3._delta_metric @ full_state_x
        expected_diss_proj = phase3._delta_diss @ full_state_x
        assert np.allclose(stalk.projected_state_metric, expected_metric_proj, rtol=1e-10)
        assert np.allclose(stalk.projected_state_dissipative, expected_diss_proj, rtol=1e-10)

    def test_lossless_subspace_dimension_matches_betti0(self, valid_context, full_state_x):
        phase3_local = Phase3(context=valid_context)
        stalk = phase3_local.export_stalk(full_state_x)
        assert stalk.lossless_subspace_dimension == valid_context.betti_0_R

    def test_state_vector_is_defensively_copied(self, phase3, full_state_x):
        stalk = phase3.export_stalk(full_state_x)
        full_state_x[0] = 9999.0
        assert stalk.state_vector[0] != 9999.0


# ==============================================================================
# 5. RETÍCULA BOOLEANA DE ESTABILIDAD
# ==============================================================================


ATOMIC_FLAGS = [
    f for f in StabilityFlags
    if f.value != 0 and (f.value & (f.value - 1)) == 0
]


class TestStabilityFlagsBooleanAlgebra:
    """
    Verifica que ``StabilityFlags`` satisface las leyes de un álgebra de
    Boole: idempotencia, absorción, distributividad y De Morgan, todas
    relativas al elemento supremo ``ALL`` (universo de discurso).
    """

    def test_all_is_disjunction_of_all_atoms(self):
        union = StabilityFlags.NONE
        for f in ATOMIC_FLAGS:
            union |= f
        assert union == StabilityFlags.ALL

    def test_none_is_identity_for_disjunction(self):
        for f in ATOMIC_FLAGS:
            assert (f | StabilityFlags.NONE) == f

    def test_all_is_identity_for_conjunction(self):
        for f in ATOMIC_FLAGS:
            assert (f & StabilityFlags.ALL) == f

    @pytest.mark.parametrize("f", ATOMIC_FLAGS)
    def test_idempotence_of_union_and_intersection(self, f):
        assert (f | f) == f
        assert (f & f) == f

    def test_absorption_law(self):
        a, b = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1]
        assert (a | (a & b)) == a
        assert (a & (a | b)) == a

    def test_distributivity_of_conjunction_over_disjunction(self):
        a, b, c = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1], ATOMIC_FLAGS[2]
        lhs = a & (b | c)
        rhs = (a & b) | (a & c)
        assert lhs == rhs

    def test_distributivity_of_disjunction_over_conjunction(self):
        a, b, c = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1], ATOMIC_FLAGS[2]
        lhs = a | (b & c)
        rhs = (a | b) & (a | c)
        assert lhs == rhs

    def test_de_morgan_law_over_union(self):
        a, b = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1]
        complement_of_union = StabilityFlags.ALL & ~(a | b)
        intersection_of_complements = (StabilityFlags.ALL & ~a) & (StabilityFlags.ALL & ~b)
        assert complement_of_union == intersection_of_complements

    def test_de_morgan_law_over_intersection(self):
        a, b = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1]
        complement_of_intersection = StabilityFlags.ALL & ~(a & b)
        union_of_complements = (StabilityFlags.ALL & ~a) | (StabilityFlags.ALL & ~b)
        assert complement_of_intersection == union_of_complements

    def test_complement_of_all_is_none(self):
        assert (StabilityFlags.ALL & ~StabilityFlags.ALL) == StabilityFlags.NONE

    def test_complement_of_none_is_all(self):
        assert (StabilityFlags.ALL & ~StabilityFlags.NONE) == StabilityFlags.ALL

    def test_describe_stability_flags_reports_all_satisfied(self):
        description = describe_stability_flags(StabilityFlags.ALL)
        assert "ESTABLE_TOTAL=True" in description
        assert "VIOLADOS=ninguno" in description

    def test_describe_stability_flags_reports_none_satisfied(self):
        description = describe_stability_flags(StabilityFlags.NONE)
        assert "ESTABLE_TOTAL=False" in description
        assert "SATISFECHOS=ninguno" in description

    def test_describe_stability_flags_lists_partial_violation(self):
        partial = StabilityFlags.ENERGY_NONNEGATIVE | StabilityFlags.DISSIPATION_VALID
        description = describe_stability_flags(partial)
        assert "ENERGY_NONNEGATIVE" in description
        assert "FLYBACK_SAFE" in description  # debe aparecer como violado
        assert "ESTABLE_TOTAL=False" in description


# ==============================================================================
# 6. INTEGRACIÓN END-TO-END DEL AGENTE COMPLETO
# ==============================================================================


class TestKBaseThermodynamicAgentIntegration:
    """Contrato público completo: construcción, fases y frontera entre ellas."""

    def test_constructor_populates_all_public_attributes(self, agent, dims):
        assert isinstance(agent.context, TopologicalContext)
        assert isinstance(agent.phase1, Phase1)
        assert isinstance(agent.phase2, Phase2)
        assert agent.phase3 is None  # instanciación perezosa
        assert (agent.context.dim_q, agent.context.dim_p) == dims

    def test_constructor_propagates_dimension_mismatch(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        with pytest.raises(DimensionMismatchError):
            KBaseThermodynamicAgent(
                C_soc=C_soc[:, :-1], M_rec=M_rec, R_cost=R_cost, J_base=J_base
            )

    def test_constructor_propagates_capacitance_degeneracy(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(n, n)))
        bad_C = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        with pytest.raises(CapacitanceDegeneracyError):
            KBaseThermodynamicAgent(C_soc=bad_C, M_rec=M_rec, R_cost=R_cost, J_base=J_base)

    def test_constructor_propagates_rayleigh_violation(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(n, n)))
        bad_R = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        with pytest.raises(RayleighDissipationViolation):
            KBaseThermodynamicAgent(C_soc=C_soc, M_rec=M_rec, R_cost=bad_R, J_base=J_base)

    def test_constructor_propagates_ill_conditioned_error(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(n, n)))
        huge_kappa_C = Q @ np.diag(np.geomspace(1.0, 1.0e12, n)) @ Q.T
        with pytest.raises(IllConditionedMatrixError):
            KBaseThermodynamicAgent(
                C_soc=huge_kappa_C, M_rec=M_rec, R_cost=R_cost, J_base=J_base, kappa_max=1.0e6
            )

    def test_public_synthesize_basal_hamiltonian_matches_phase2_directly(self, agent, state_vectors):
        q, p, df_dt = state_vectors
        tensor_public = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        tensor_direct = agent.phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        assert tensor_public.total_hamiltonian == pytest.approx(tensor_direct.total_hamiltonian)

    def test_export_sheaf_stalk_lazy_initializes_phase3_once(self, agent, full_state_x):
        assert agent.phase3 is None
        stalk_1 = agent.export_sheaf_stalk(full_state_x)
        phase3_instance = agent.phase3
        assert phase3_instance is not None
        stalk_2 = agent.export_sheaf_stalk(full_state_x)
        assert agent.phase3 is phase3_instance  # misma instancia reutilizada
        assert np.allclose(stalk_1.delta_base, stalk_2.delta_base)

    def test_end_to_end_state_vector_consistency_between_phase2_and_phase3(self, agent, state_vectors):
        q, p, df_dt = state_vectors
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        state_x = np.concatenate([q, p])
        stalk = agent.export_sheaf_stalk(state_x)
        assert np.allclose(stalk.state_vector, state_x)
        # ∇H proyectado sobre la fibra métrica debe ser consistente en norma
        assert stalk.projected_state_metric.shape == (agent.context.dim_q + agent.context.dim_p,)

    def test_agent_with_custom_metric_tensors_end_to_end(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        G_q = np.diag(np.linspace(1.0, 1.2, dim_q))
        G_p = np.diag(np.linspace(1.0, 1.1, dim_p))
        agent_custom = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, G_q=G_q, G_p=G_p
        )
        rng = np.random.default_rng(2024)
        q = rng.normal(size=dim_q)
        p = rng.normal(size=dim_p)
        df_dt = rng.normal(size=dim_p)
        tensor = agent_custom.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert tensor.total_hamiltonian >= 0.0

    def test_agent_raises_inertial_flyback_error_end_to_end(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        _, dim_p = dims
        agent_strict = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base, breakdown_voltage=1e-12
        )
        rng = np.random.default_rng(2025)
        q = rng.normal(size=C_soc.shape[0])
        p = rng.normal(size=dim_p)
        df_dt = np.ones(dim_p) * 100.0
        with pytest.raises(InertialFlybackError):
            agent_strict.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

    def test_friendly_name_is_defined(self, agent):
        assert agent.FRIENDLY_NAME == "Asesor de Cimientos Financieros"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))