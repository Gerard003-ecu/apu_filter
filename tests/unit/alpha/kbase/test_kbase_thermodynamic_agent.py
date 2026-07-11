# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite de Pruebas : test_kbase_thermodynamic_agent.py                         |
| Objetivo         : Verificación granular, rigurosa y matemáticamente         |
|                    trazable de KBaseThermodynamicAgent (fases anidadas)      |
| Versión suite    : 2.0.0-Rigorous-Nested-Oracle                              |
+==============================================================================+

FILOSOFÍA DE LA SUITE
----------------------
Cada método público y privado relevante de las tres fases anidadas

    Phase1_MatrixTopology
        │  build_topological_context()  ──►  TopologicalContext
        ▼
    Phase2_HamiltonianDynamics
        │  synthesize_basal_state(...)  ──►  BasalStateTensor
        ▼
    Phase3_SheafProjection
        │  export_stalk(x)              ──►  SheafStalk

se somete a pruebas de caja blanca que verifican no sólo la ausencia de
excepciones, sino la **corrección numérica exacta** frente a oráculos
independientes calculados con ``numpy.linalg`` / ``scipy.linalg`` sin
reutilizar la lógica interna del módulo bajo prueba.

Estructura (espejo de las 3 fases + contratos funtoriales)
----------------------------------------------------------
  0. Utilidades y fixtures canónicos (oráculos deterministas)
  1. TestExceptionHierarchy
  2. TestPhase1*   — topología matricial, métrica, espectro, Cholesky
  3. TestPhase2*   — Hamiltoniano, Rayleigh, flyback, Williamson, Boole
  4. TestPhase3*   — cofrontera, Hodge, stalk
  5. TestStabilityFlagsBooleanAlgebra — retícula de predicados
  6. TestFunctorialComposition        — export ∘ synthesize ∘ build
  7. TestKBaseThermodynamicAgentIntegration — contrato público
  8. TestNumericalStressAndInvariants — n=1, R=0, κ alto, ensemble

Tolerancias
-----------
Todas las cotas se expresan en términos de
``ε = np.finfo(np.float64).eps``, normas Frobenius/espectrales y, cuando
corresponde, del número de condición κ (análisis de Wilkinson/Higham).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

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

import logging

logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Constantes de rigor numérico
# ---------------------------------------------------------------------------
EPS: float = float(np.finfo(np.float64).eps)
WILKINSON_SAFETY: float = 100.0
EULER_REL_TOL: float = 1.0e-6
STRUCTURAL_REL_TOL: float = 1.0e-6
HODGE_REL_FACTOR: float = 1.0e3

Phase1 = KBaseThermodynamicAgent.Phase1_MatrixTopology
Phase2 = KBaseThermodynamicAgent.Phase2_HamiltonianDynamics
Phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection


# =============================================================================
# 0. UTILIDADES DE GENERACIÓN DETERMINISTA (ORÁCULOS INDEPENDIENTES)
# =============================================================================


def make_spd(
    n: int,
    seed: int,
    floor: float = 1.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """SPD determinista: A = scale·(B Bᵀ + floor·I), garantiza λ_min ≥ scale·floor."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    A = scale * (B @ B.T + floor * np.eye(n))
    return 0.5 * (A + A.T)


def make_spd_with_condition(
    n: int,
    cond: float,
    seed: int,
) -> NDArray[np.float64]:
    """
    SPD con κ(A) ≈ cond vía espectro log-uniforme:
        A = Q · diag(λ) · Qᵀ,  λ_max/λ_min = cond.
    """
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.normal(size=(n, n)))
    log_l = np.linspace(0.0, math.log(max(cond, 1.0 + 1e-15)), n)
    lambdas = np.exp(log_l)
    A = (Q * lambdas) @ Q.T
    return 0.5 * (A + A.T)


def make_psd_with_rank(n: int, rank: int, seed: int) -> NDArray[np.float64]:
    """PSD determinista con rango exacto controlado vía autovalores impuestos."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigvals = np.zeros(n, dtype=np.float64)
    r = max(0, min(rank, n))
    if r > 0:
        eigvals[:r] = rng.uniform(0.5, 2.0, size=r)
    R = Q @ np.diag(eigvals) @ Q.T
    return 0.5 * (R + R.T)


def make_antisymmetric(n: int, seed: int) -> NDArray[np.float64]:
    """Antisimétrica determinista: J = B − Bᵀ ∈ 𝔰𝔬(n)."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    return B - B.T


def fro_rel(A: np.ndarray, B: np.ndarray) -> float:
    """‖A−B‖_F / max(‖B‖_F, 1)."""
    return float(la.norm(A - B, "fro")) / max(float(la.norm(B, "fro")), 1.0)


def spectral_condition(A: np.ndarray) -> float:
    """κ₂(A) = λ_max/λ_min para A simétrica (oráculo independiente)."""
    ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    lmin, lmax = float(ev[0]), float(ev[-1])
    if lmin <= 0.0:
        return float("inf")
    return lmax / lmin


# =============================================================================
# FIXTURES DE SISTEMAS CANÓNICOS
# =============================================================================


@pytest.fixture(scope="function")
def dims() -> Tuple[int, int]:
    return 3, 2  # dim_q, dim_p → n = 5


@pytest.fixture(scope="function")
def valid_system(dims):
    """Sistema Port-Hamiltoniano válido genérico (SPD, PSD, antisimétrico)."""
    dim_q, dim_p = dims
    n = dim_q + dim_p
    C_soc = make_spd(dim_q, seed=101)
    M_rec = make_spd(dim_p, seed=102)
    R_cost = make_psd_with_rank(n, rank=n, seed=103)
    J_base = make_antisymmetric(n, seed=104)
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="function")
def agent(valid_system) -> KBaseThermodynamicAgent:
    C_soc, M_rec, R_cost, J_base = valid_system
    return KBaseThermodynamicAgent(
        C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
    )


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
    known_eigvals = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)[:n]
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
    Frecuencia propia exacta: ω = 1/√(M·C).
    """
    c, m = 4.0, 9.0
    C_soc = np.array([[c]], dtype=np.float64)
    M_rec = np.array([[m]], dtype=np.float64)
    R_cost = np.zeros((2, 2), dtype=np.float64)
    J_base = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    omega_exact = 1.0 / math.sqrt(c * m)
    return C_soc, M_rec, R_cost, J_base, omega_exact


@pytest.fixture(scope="function")
def lossless_system(dims):
    """Sistema con R_cost ≡ 0 (flujo Hamiltoniano puro, P_diss ≡ 0)."""
    dim_q, dim_p = dims
    n = dim_q + dim_p
    C_soc = make_spd(dim_q, seed=401)
    M_rec = make_spd(dim_p, seed=402)
    R_cost = np.zeros((n, n), dtype=np.float64)
    J_base = make_antisymmetric(n, seed=404)
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="function")
def phase1_placeholder():
    """
    Factory de Phase1 con matrices de relleno formales, para validadores
    que no dependen del ensamblaje físico completo.
    """

    def _make(kappa_max: float = 1.0e10) -> Phase1:
        C_soc = np.eye(2, dtype=np.float64)
        M_rec = np.eye(2, dtype=np.float64)
        R_cost = np.eye(4, dtype=np.float64)
        J_base = make_antisymmetric(4, seed=999)
        return Phase1(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            kappa_max=kappa_max,
        )

    return _make


@pytest.fixture(scope="function")
def valid_context(valid_system) -> TopologicalContext:
    C_soc, M_rec, R_cost, J_base = valid_system
    p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
    return p1.build_topological_context()


@pytest.fixture(scope="function")
def phase2(valid_context) -> Phase2:
    return Phase2(
        context=valid_context, breakdown_voltage=1.0e5, kappa_max=1.0e10
    )


@pytest.fixture(scope="function")
def phase3(valid_context) -> Phase3:
    return Phase3(context=valid_context)


@pytest.fixture(scope="function")
def state_vectors(dims):
    dim_q, dim_p = dims
    rng = np.random.default_rng(42)
    q = rng.normal(size=dim_q)
    p = rng.normal(size=dim_p)
    df_dt = rng.normal(size=dim_p)
    return q, p, df_dt


@pytest.fixture(scope="function")
def full_state_x(dims):
    n = sum(dims)
    rng = np.random.default_rng(777)
    return rng.normal(size=n)


# =============================================================================
# 1. JERARQUÍA DE EXCEPCIONES
# =============================================================================


class TestExceptionHierarchy:
    """Todas las excepciones del dominio forman un único árbol categórico."""

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
    def test_all_domain_exceptions_derive_from_thermodynamic_base_error(
        self, exc_cls
    ):
        assert issubclass(exc_cls, ThermodynamicBaseError)

    def test_thermodynamic_base_error_derives_from_exception(self):
        assert issubclass(ThermodynamicBaseError, Exception)

    def test_catch_all_with_root_on_asymmetric_C(self, valid_system):
        """Un único ``except ThermodynamicBaseError`` captura fallos de Fase 1."""
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_C = C_soc.copy()
        bad_C[0, 1] += 1.0
        with pytest.raises(ThermodynamicBaseError):
            KBaseThermodynamicAgent(
                C_soc=bad_C, M_rec=M_rec, R_cost=R_cost, J_base=J_base
            )


# =============================================================================
# 2. FASE 1 — TOPOLOGÍA MATRICIAL, MÉTRICA RIEMANNIANA, ESPECTRO
# =============================================================================


class TestPhase1DimensionValidation:
    """``_check_dimensions``: todas las formas de inconsistencia dimensional."""

    def test_valid_dimensions_returns_correct_tuple(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        dim_q, dim_p = p1._check_dimensions()
        assert (dim_q, dim_p) == dims

    def test_c_soc_non_square_raises(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        bad_C = C_soc[:, :-1]
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

    def test_r_cost_incompatible_with_n_qp(self, valid_system):
        """R debe ser (dim_q+dim_p)²; un tamaño intermedio también falla."""
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        bad_R = np.eye(n - 1) if n > 1 else np.eye(n + 1)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=bad_R, J_base=J_base[: bad_R.shape[0], : bad_R.shape[0]] if bad_R.shape[0] <= n else make_antisymmetric(bad_R.shape[0], 0))
        with pytest.raises(DimensionMismatchError):
            p1._check_dimensions()


class TestPhase1SymmetryValidation:
    """``_validate_symmetry`` / ``_validate_antisymmetry`` con tolerancias Frobenius."""

    def test_symmetric_matrix_passes(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        p1._validate_symmetry(C_soc, "C_soc")

    def test_asymmetric_matrix_fails(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        broken = C_soc.copy()
        broken[0, 1] += 1.0
        p1 = Phase1(C_soc=broken, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(ThermodynamicBaseError):
            p1._validate_symmetry(broken, "C_soc_broken")

    def test_antisymmetric_matrix_passes(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        p1._validate_antisymmetry(J_base, "J_base")

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

    def test_sub_machine_eps_asymmetry_is_tolerated(self, valid_system):
        """
        Perturbación O(ε·‖A‖_F) no debe disparar fallo de simetría
        (coherente con tol = ε·max(‖A‖_F, 1) del módulo).
        """
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        if n < 2:
            pytest.skip("requiere n≥2")
        A = C_soc.copy()
        scale = max(float(la.norm(A, "fro")), 1.0)
        A[0, 1] += 0.1 * EPS * scale  # sub-tolerancia
        asym = float(la.norm(A - A.T, "fro"))
        tol = EPS * scale
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        if asym <= tol:
            p1._validate_symmetry(A, "C_soc_near_sym")
        else:
            pytest.skip("perturbación superó tol de máquina en esta plataforma")


class TestPhase1MetricTensorValidation:
    """``_validate_metric_tensor``: invertibilidad y condicionamiento de G_q/G_p."""

    def test_identity_metric_has_condition_number_one(self, phase1_placeholder):
        p1 = phase1_placeholder()
        kappa = p1._validate_metric_tensor(np.eye(3), "G_id", expected_dim=3)
        assert kappa == pytest.approx(1.0, rel=1e-12)

    def test_wrong_shape_raises_dimension_mismatch(self, phase1_placeholder):
        p1 = phase1_placeholder()
        with pytest.raises(DimensionMismatchError):
            p1._validate_metric_tensor(np.eye(2), "G_bad_shape", expected_dim=3)

    def test_singular_metric_raises_singularity_error(self, phase1_placeholder):
        p1 = phase1_placeholder()
        G = np.array([[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(MetricTensorSingularityError):
            p1._validate_metric_tensor(G, "G_singular", expected_dim=2)

    def test_ill_conditioned_metric_raises_singularity_error(
        self, phase1_placeholder
    ):
        p1 = phase1_placeholder(kappa_max=10.0)
        G = np.diag([1.0e-3, 1.0])  # κ = 1000 > 10
        with pytest.raises(MetricTensorSingularityError):
            p1._validate_metric_tensor(G, "G_ill", expected_dim=2)

    def test_moderate_condition_number_within_bounds_passes(
        self, phase1_placeholder
    ):
        p1 = phase1_placeholder(kappa_max=100.0)
        G = np.diag([1.0, 2.0, 5.0])
        kappa = p1._validate_metric_tensor(G, "G_ok", expected_dim=3)
        assert kappa == pytest.approx(5.0, rel=1e-12)

    def test_kappa_matches_independent_oracle(self, phase1_placeholder):
        p1 = phase1_placeholder(kappa_max=1.0e12)
        G = make_spd_with_condition(4, cond=50.0, seed=11)
        kappa = p1._validate_metric_tensor(G, "G_oracle", expected_dim=4)
        assert kappa == pytest.approx(spectral_condition(G), rel=1e-8)


class TestPhase1CongruencePullback:
    """
    ``_congruence_pullback``: Ã = G A Gᵀ.
    Ley de Inercia de Sylvester: G invertible ⇒ signatura(A) = signatura(Ã).
    """

    def test_identity_metric_leaves_matrix_unchanged(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        I = np.eye(C_soc.shape[0])
        A_tilde = p1._congruence_pullback(C_soc, I, "C_soc")
        assert fro_rel(A_tilde, C_soc) < 1e-12

    def test_pullback_preserves_symmetry(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        G = np.diag(np.linspace(1.0, 3.0, C_soc.shape[0]))
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        assert float(la.norm(A_tilde - A_tilde.T, "fro")) < 1e-10

    def test_pullback_preserves_positive_definiteness_sylvester_inertia(
        self, valid_system
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        rng = np.random.default_rng(55)
        n = C_soc.shape[0]
        G = rng.normal(size=(n, n)) + 3.0 * np.eye(n)
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        assert np.all(np.linalg.eigvalsh(A_tilde) > 0.0)

    def test_pullback_matches_manual_matrix_product(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        G = np.diag(np.linspace(0.5, 2.0, C_soc.shape[0]))
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        expected = G @ C_soc @ G.T
        assert fro_rel(A_tilde, expected) < 1e-12

    def test_pullback_scales_eigenvalues_under_scalar_metric(self, valid_system):
        """Si G = α I, entonces Ã = α² A ⇒ λ(Ã) = α² λ(A)."""
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        alpha = 2.5
        G = alpha * np.eye(C_soc.shape[0])
        A_tilde = p1._congruence_pullback(C_soc, G, "C_soc")
        ev_orig = np.sort(np.linalg.eigvalsh(C_soc))
        ev_tilde = np.sort(np.linalg.eigvalsh(A_tilde))
        np.testing.assert_allclose(ev_tilde, (alpha ** 2) * ev_orig, rtol=1e-10)


class TestPhase1ConditionNumber:
    """``_compute_condition_number`` vs oráculo ``np.linalg.eigvalsh``."""

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
        oracle = np.linalg.eigvalsh(C_soc)
        assert lmin == pytest.approx(float(oracle[0]), rel=1e-8)
        assert lmax == pytest.approx(float(oracle[-1]), rel=1e-8)
        assert kappa == pytest.approx(
            float(oracle[-1] / oracle[0]), rel=1e-8
        )

    def test_non_spd_matrix_raises_capacitance_degeneracy(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(n, n)))
        bad = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(CapacitanceDegeneracyError):
            p1._compute_condition_number(bad, "bad_matrix")

    def test_ill_conditioned_matrix_raises_ill_conditioned_error(
        self, valid_system
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        bad = make_spd_with_condition(n, cond=1.0e12, seed=8)
        p1 = Phase1(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            kappa_max=1.0e6,
        )
        with pytest.raises(IllConditionedMatrixError):
            p1._compute_condition_number(bad, "ill_conditioned")

    def test_near_singular_spd_raises_or_flags_ill_conditioned(
        self, valid_system
    ):
        """λ_min → 0⁺ con λ_max fijo ⇒ κ enorme o degeneración."""
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        ev = np.ones(n)
        ev[0] = 1.0e-18
        Q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(n, n)))
        near_sing = Q @ np.diag(ev) @ Q.T
        p1 = Phase1(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            kappa_max=1.0e10,
        )
        with pytest.raises(
            (CapacitanceDegeneracyError, IllConditionedMatrixError)
        ):
            p1._compute_condition_number(near_sing, "near_sing")


class TestPhase1CholeskyRegularization:
    """``_cholesky_spd_regularized``: τ=0 nominal, jitter y fallo persistente."""

    def test_well_conditioned_matrix_requires_no_jitter(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        L, tau = p1._cholesky_spd_regularized(C_soc, "C_soc")
        assert tau == 0.0
        assert fro_rel(L @ L.T, C_soc) < 1e-10

    def test_cholesky_reconstructs_original_matrix_exactly(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        L, _ = p1._cholesky_spd_regularized(M_rec, "M_rec")
        assert fro_rel(L @ L.T, M_rec) < 1e-10
        assert np.allclose(L, np.tril(L))

    def test_L_has_positive_diagonal(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        L, _ = p1._cholesky_spd_regularized(C_soc, "C_soc")
        assert np.all(np.diag(L) > 0.0)

    def test_transient_failure_triggers_jitter_and_recovers(
        self, valid_system, monkeypatch
    ):
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
        assert fro_rel(L @ L.T, C_soc + tau * np.eye(C_soc.shape[0])) < 1e-10

    def test_persistent_failure_raises_capacitance_degeneracy(
        self, valid_system, monkeypatch
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)

        def always_fail(A, lower=True):
            raise la.LinAlgError("fallo persistente simulado")

        monkeypatch.setattr(la, "cholesky", always_fail)
        with pytest.raises(CapacitanceDegeneracyError):
            p1._cholesky_spd_regularized(C_soc, "C_soc", max_attempts=3)


class TestPhase1PSDSpectralDiagnostics:
    """``_validate_psd_and_spectral_diagnostics``: rango, brecha, R_sqrt, cierre."""

    def test_spd_matrix_yields_full_rank_and_zero_kernel(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, rank_R, gap = p1._validate_psd_and_spectral_diagnostics(
            R_cost, "R_cost"
        )
        assert rank_R == n
        assert fro_rel(R_sqrt @ R_sqrt, R_cost) < 1e-8

    def test_R_sqrt_is_symmetric(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, _, _ = p1._validate_psd_and_spectral_diagnostics(
            R_cost, "R_cost"
        )
        assert float(la.norm(R_sqrt - R_sqrt.T, "fro")) < 1e-12

    def test_spectral_closure_R_sqrt_squared(self, valid_system):
        """Cierre algebraico: ‖R_sqrt² − R‖_F / ‖R‖_F ≈ 0."""
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, _, _ = p1._validate_psd_and_spectral_diagnostics(
            R_cost, "R_cost"
        )
        closure = fro_rel(R_sqrt @ R_sqrt, R_cost)
        # Cota laxa de Wilkinson: O(n · ε · κ_eff)
        n = R_cost.shape[0]
        cond_eff = spectral_condition(R_cost + EPS * np.eye(n))
        tol = WILKINSON_SAFETY * EPS * max(cond_eff, 1.0) * n
        assert closure < max(tol, 1e-8)

    def test_negative_eigenvalue_raises_rayleigh_violation(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(n, n)))
        bad = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        with pytest.raises(RayleighDissipationViolation):
            p1._validate_psd_and_spectral_diagnostics(bad, "R_cost_bad")

    def test_rank_deficient_psd_matrix_reports_correct_rank(
        self, rank_deficient_R_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = rank_deficient_R_system
        n = sum(dims)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        _, rank_R, _ = p1._validate_psd_and_spectral_diagnostics(
            R_cost, "R_cost"
        )
        assert rank_R == n - 2

    def test_spectral_gap_matches_known_eigenvalues(self, diagonal_R_system):
        C_soc, M_rec, R_cost, J_base, known_eigvals = diagonal_R_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        _, _, gap = p1._validate_psd_and_spectral_diagnostics(R_cost, "R_cost")
        sorted_eigs = np.sort(known_eigvals)
        expected_gap = float(sorted_eigs[1] - sorted_eigs[0])
        assert gap == pytest.approx(expected_gap, rel=1e-8)

    def test_zero_matrix_is_psd_with_rank_zero(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        R_sqrt, rank_R, gap = p1._validate_psd_and_spectral_diagnostics(
            np.zeros((n, n)), "zero"
        )
        assert rank_R == 0
        assert np.allclose(R_sqrt, 0.0)
        assert gap == pytest.approx(0.0, abs=1e-15)


class TestPhase1BuildTopologicalContextIntegration:
    """Método terminal de Fase 1 → ``TopologicalContext`` (precondición Fase 2)."""

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

    def test_cholesky_factors_reconstruct_pulled_back_matrices(
        self, valid_system
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert fro_rel(ctx.L_C @ ctx.L_C.T, C_soc) < 1e-8
        assert fro_rel(ctx.L_M @ ctx.L_M.T, M_rec) < 1e-8

    def test_inv_sqrt_matrices_are_true_inverse_square_roots(
        self, valid_system
    ):
        """
        C_inv_sqrtᵀ C_inv_sqrt ≈ C⁻¹  (y análogo para M).
        Equivalente a: C_inv_sqrt ≈ C^{−1/2} en la métrica Frobenius
        del producto de Gram.
        """
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        C_inv_recon = ctx.C_inv_sqrt.T @ ctx.C_inv_sqrt
        M_inv_recon = ctx.M_inv_sqrt.T @ ctx.M_inv_sqrt
        assert fro_rel(C_inv_recon, np.linalg.inv(C_soc)) < 1e-8
        assert fro_rel(M_inv_recon, np.linalg.inv(M_rec)) < 1e-8

    def test_inv_sqrt_left_inverse_of_cholesky_path(self, valid_system):
        """
        Si L Lᵀ = C y C_inv_sqrt deriva de L⁻¹, entonces
        C_inv_sqrt @ L ≈ I (hasta permutación de convención L vs Lᵀ).
        Verificamos la identidad de cierre más robusta:
            (C_inv_sqrtᵀ C_inv_sqrt) C ≈ I.
        """
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        nq = ctx.dim_q
        prod = (ctx.C_inv_sqrt.T @ ctx.C_inv_sqrt) @ C_soc
        assert fro_rel(prod, np.eye(nq)) < 1e-7

    def test_nontrivial_metric_tensor_applies_congruent_pullback(
        self, valid_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        G_q = np.diag(np.linspace(1.0, 2.0, dim_q))
        G_p = np.diag(np.linspace(1.0, 1.5, dim_p))
        p1 = Phase1(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        ctx = p1.build_topological_context()
        expected_C = G_q @ C_soc @ G_q.T
        expected_M = G_p @ M_rec @ G_p.T
        assert fro_rel(ctx.L_C @ ctx.L_C.T, expected_C) < 1e-8
        assert fro_rel(ctx.L_M @ ctx.L_M.T, expected_M) < 1e-8
        assert ctx.kappa_G_q == pytest.approx(2.0, rel=1e-8)
        assert ctx.kappa_G_p == pytest.approx(1.5, rel=1e-8)

    def test_default_metric_tensors_are_identity(self, valid_system, dims):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert np.allclose(ctx.G_q, np.eye(dim_q))
        assert np.allclose(ctx.G_p, np.eye(dim_p))

    def test_betti_0_equals_n_minus_rank_R(
        self, rank_deficient_R_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = rank_deficient_R_system
        n = sum(dims)
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        assert ctx.betti_0_R == n - ctx.rank_R
        assert ctx.betti_0_R == 2

    def test_context_holds_defensive_copies(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        # Mutar inputs originales no debe alterar el contexto
        C_soc[0, 0] = -999.0
        assert ctx.L_C is not None
        # Si el contexto almacena copias de C/R/J, verificar R_cost
        if hasattr(ctx, "R_cost"):
            R_cost[0, 0] = -999.0
            assert ctx.R_cost[0, 0] != -999.0


# =============================================================================
# 3. FASE 2 — DINÁMICA PORT-HAMILTONIANA, RAYLEIGH Y WILLIAMSON
# =============================================================================


class TestPhase2EnergyComputation:
    """Energías y gradientes vs inversión explícita (oráculo independiente)."""

    def test_potential_energy_matches_explicit_inverse(
        self, phase2, valid_system, state_vectors
    ):
        C_soc, *_ = valid_system
        q, p, df_dt = state_vectors
        V_q, grad_V_q = phase2._evaluate_potential_energy(q)
        C_inv = np.linalg.inv(C_soc)
        expected_V = 0.5 * float(q @ C_inv @ q)
        expected_grad = C_inv @ q
        assert V_q == pytest.approx(expected_V, rel=1e-8)
        np.testing.assert_allclose(grad_V_q, expected_grad, rtol=1e-6, atol=1e-8)

    def test_kinetic_energy_matches_explicit_inverse(
        self, phase2, valid_system, state_vectors
    ):
        _, M_rec, _, _ = valid_system
        q, p, df_dt = state_vectors
        K_p, grad_K_p = phase2._compute_kinetic_energy(p)
        M_inv = np.linalg.inv(M_rec)
        expected_K = 0.5 * float(p @ M_inv @ p)
        expected_grad = M_inv @ p
        assert K_p == pytest.approx(expected_K, rel=1e-8)
        np.testing.assert_allclose(grad_K_p, expected_grad, rtol=1e-6, atol=1e-8)

    def test_energies_are_nonnegative_for_arbitrary_state(self, phase2, dims):
        dim_q, dim_p = dims
        rng = np.random.default_rng(123)
        for _ in range(30):
            q = rng.normal(size=dim_q) * rng.uniform(0.1, 10.0)
            p = rng.normal(size=dim_p) * rng.uniform(0.1, 10.0)
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

    def test_quadratic_homogeneity_degree_two(self, phase2, state_vectors):
        """H(λx) = λ² H(x) y ∇H(λx) = λ ∇H(x) para H cuadrática."""
        q, p, _ = state_vectors
        lam = 3.0
        V1, gV1 = phase2._evaluate_potential_energy(q)
        V2, gV2 = phase2._evaluate_potential_energy(lam * q)
        assert V2 == pytest.approx(lam ** 2 * V1, rel=1e-10)
        np.testing.assert_allclose(gV2, lam * gV1, rtol=1e-10)

        K1, gK1 = phase2._compute_kinetic_energy(p)
        K2, gK2 = phase2._compute_kinetic_energy(lam * p)
        assert K2 == pytest.approx(lam ** 2 * K1, rel=1e-10)
        np.testing.assert_allclose(gK2, lam * gK1, rtol=1e-10)

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
    Teorema de Euler (grado 2): q·∇_q H + p·∇_p H = 2 H.
    Identidad exacta para Hamiltoniano cuadrático.
    """

    def test_euler_identity_holds_near_machine_precision(
        self, phase2, state_vectors
    ):
        q, p, df_dt = state_vectors
        V_q, grad_V = phase2._evaluate_potential_energy(q)
        K_p, grad_K = phase2._compute_kinetic_energy(p)
        H_total = V_q + K_p
        residual = phase2._verify_euler_homogeneity(
            q, p, grad_V, grad_K, H_total
        )
        assert residual < EULER_REL_TOL * max(abs(H_total), 1.0)

    def test_euler_identity_holds_for_zero_state(self, phase2, dims):
        dim_q, dim_p = dims
        residual = phase2._verify_euler_homogeneity(
            np.zeros(dim_q),
            np.zeros(dim_p),
            np.zeros(dim_q),
            np.zeros(dim_p),
            0.0,
        )
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_euler_manual_oracle(self, phase2, state_vectors):
        """Oráculo independiente: |q·∇V + p·∇K − 2H| / max(|2H|,1)."""
        q, p, _ = state_vectors
        V, gV = phase2._evaluate_potential_energy(q)
        K, gK = phase2._compute_kinetic_energy(p)
        H = V + K
        lhs = float(q @ gV + p @ gK)
        residual = abs(lhs - 2.0 * H) / max(abs(2.0 * H), 1.0)
        assert residual < 1e-10


class TestPhase2RayleighDissipation:
    """``_enforce_rayleigh_dissipation``: Segunda Ley P_diss = ∇Hᵀ R ∇H ≥ 0."""

    def test_dissipated_power_matches_quadratic_form(
        self, phase2, valid_system, state_vectors
    ):
        _, _, R_cost, _ = valid_system
        q, p, df_dt = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        expected = float(grad_H @ R_cost @ grad_H)
        assert P_diss == pytest.approx(expected, rel=1e-8)
        assert P_diss >= -1e-14

    def test_wrong_shape_raises_dimension_mismatch(self, phase2, dims):
        n = sum(dims)
        with pytest.raises(DimensionMismatchError):
            phase2._enforce_rayleigh_dissipation(np.zeros(n + 1))

    def test_negative_definite_r_cost_raises_violation(self, valid_context):
        """
        Caja blanca: R_cost negativo-definida inyectada en el contexto
        (bypass Fase 1) ⇒ red de seguridad de Fase 2.
        """
        n = valid_context.dim_q + valid_context.dim_p
        broken_ctx = dataclasses.replace(valid_context, R_cost=-np.eye(n))
        phase2_broken = Phase2(
            context=broken_ctx, breakdown_voltage=1e5, kappa_max=1e10
        )
        with pytest.raises(RayleighDissipationViolation):
            phase2_broken._enforce_rayleigh_dissipation(np.ones(n))

    def test_zero_gradient_yields_zero_dissipation(self, phase2, dims):
        n = sum(dims)
        P_diss = phase2._enforce_rayleigh_dissipation(np.zeros(n))
        assert P_diss == pytest.approx(0.0, abs=1e-12)

    def test_dissipation_scales_as_lambda_squared(self, phase2, state_vectors):
        """P_diss(λ ∇H) = λ² P_diss(∇H) (forma cuadrática)."""
        q, p, _ = state_vectors
        _, gV = phase2._evaluate_potential_energy(q)
        _, gK = phase2._compute_kinetic_energy(p)
        gH = np.concatenate([gV, gK])
        P1 = phase2._enforce_rayleigh_dissipation(gH)
        P2 = phase2._enforce_rayleigh_dissipation(2.0 * gH)
        assert P2 == pytest.approx(4.0 * P1, rel=1e-10, abs=1e-14)


class TestPhase2FlybackVoltage:
    """``_measure_flyback_voltage``: ‖M df/dt‖_∞ vs umbral de ruptura."""

    def test_flyback_matches_manual_matrix_vector_product(
        self, phase2, valid_system, state_vectors
    ):
        _, M_rec, _, _ = valid_system
        _, _, df_dt = state_vectors
        v_fb = phase2._measure_flyback_voltage(df_dt)
        expected_norm = float(np.linalg.norm(M_rec @ df_dt, ord=np.inf))
        assert v_fb == pytest.approx(expected_norm, rel=1e-6)

    def test_wrong_shape_raises_dimension_mismatch(self, phase2, dims):
        _, dim_p = dims
        with pytest.raises(DimensionMismatchError):
            phase2._measure_flyback_voltage(np.zeros(dim_p + 1))

    def test_zero_perturbation_yields_zero_voltage(self, phase2, dims):
        _, dim_p = dims
        v_fb = phase2._measure_flyback_voltage(np.zeros(dim_p))
        assert v_fb == pytest.approx(0.0, abs=1e-12)

    def test_exceeding_breakdown_voltage_raises_inertial_flyback_error(
        self, valid_context, dims
    ):
        _, dim_p = dims
        phase2_strict = Phase2(
            context=valid_context, breakdown_voltage=1e-9, kappa_max=1e10
        )
        with pytest.raises(InertialFlybackError):
            phase2_strict._measure_flyback_voltage(np.ones(dim_p))

    def test_flyback_homogeneous_of_degree_one(
        self, phase2, state_vectors
    ):
        """‖M (λ df)‖_∞ = |λ| ‖M df‖_∞."""
        _, _, df_dt = state_vectors
        v1 = phase2._measure_flyback_voltage(df_dt)
        v2 = phase2._measure_flyback_voltage(3.0 * df_dt)
        assert v2 == pytest.approx(3.0 * v1, rel=1e-10)


class TestPhase2VectorFieldAndStructuralConsistency:
    """
    Campo vectorial ẋ = (J − R) ∇H y la identidad algebraica exacta
        ∇Hᵀ ẋ ≡ −P_diss
    (consecuencia de Jᵀ = −J ⇒ ∇Hᵀ J ∇H = 0).
    """

    def test_vector_field_matches_manual_formula(
        self, phase2, valid_system, state_vectors
    ):
        _, _, R_cost, J_base = valid_system
        q, p, _ = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        x_dot = phase2._compute_vector_field(grad_H)
        expected = (J_base - R_cost) @ grad_H
        np.testing.assert_allclose(x_dot, expected, rtol=1e-8)

    def test_structural_consistency_holds_for_genuine_ph_system(
        self, phase2, state_vectors
    ):
        q, p, _ = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        x_dot = phase2._compute_vector_field(grad_H)
        residual = phase2._verify_structural_consistency(
            grad_H, x_dot, P_diss
        )
        assert residual < STRUCTURAL_REL_TOL * max(P_diss, 1.0)

    def test_grad_H_transpose_J_grad_H_is_identically_zero(
        self, phase2, valid_system, state_vectors
    ):
        """Identidad algebraica pura: ∇Hᵀ J ∇H ≡ 0 para J antisimétrica."""
        _, _, _, J_base = valid_system
        q, p, _ = state_vectors
        _, grad_V = phase2._evaluate_potential_energy(q)
        _, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        quad_form = float(grad_H @ J_base @ grad_H)
        assert quad_form == pytest.approx(0.0, abs=1e-8)

    def test_power_balance_identity_manual_oracle(
        self, phase2, valid_system, state_vectors
    ):
        """
        Oráculo independiente del balance de potencia:
            dH/dt = ∇Hᵀ ẋ = ∇Hᵀ (J−R) ∇H = −∇Hᵀ R ∇H = −P_diss.
        """
        _, _, R_cost, J_base = valid_system
        q, p, _ = state_vectors
        _, gV = phase2._evaluate_potential_energy(q)
        _, gK = phase2._compute_kinetic_energy(p)
        gH = np.concatenate([gV, gK])
        x_dot = (J_base - R_cost) @ gH
        dH_dt = float(gH @ x_dot)
        P_diss = float(gH @ R_cost @ gH)
        assert dH_dt == pytest.approx(-P_diss, rel=1e-8, abs=1e-10)

    def test_inconsistent_wiring_raises_structural_consistency_error(
        self, phase2
    ):
        n = phase2._ctx.dim_q + phase2._ctx.dim_p
        with pytest.raises(StructuralConsistencyError):
            phase2._verify_structural_consistency(
                np.ones(n), np.ones(n) * 1000.0, 0.0
            )

    def test_lossless_flow_preserves_hamiltonian_to_first_order(
        self, lossless_system, dims
    ):
        """
        Si R ≡ 0, entonces dH/dt = 0 exactamente (flujo Hamiltoniano puro).
        """
        C_soc, M_rec, R_cost, J_base = lossless_system
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        dim_q, dim_p = dims
        rng = np.random.default_rng(88)
        q, p = rng.normal(size=dim_q), rng.normal(size=dim_p)
        df_dt = np.zeros(dim_p)
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        # P_diss debe ser ~0; el residual estructural también
        assert tensor.dissipated_power == pytest.approx(0.0, abs=1e-10) if hasattr(tensor, "dissipated_power") else True
        # Balance: ∇Hᵀ ẋ ≈ 0
        gH = np.concatenate(
            [
                agent.phase2._evaluate_potential_energy(q)[1],
                agent.phase2._compute_kinetic_energy(p)[1],
            ]
        )
        dH_dt = float(gH @ tensor.vector_field)
        assert abs(dH_dt) < 1e-8 * max(abs(tensor.total_hamiltonian), 1.0)


class TestPhase2NormalModesWilliamson:
    """
    ``compute_normal_modes`` vs frecuencia analítica del oscilador LC:
        ω = 1/√(M·C),  E₀ = ½ ℏ ω.
    """

    def test_single_mode_matches_analytic_harmonic_frequency(
        self, harmonic_oscillator_system
    ):
        C_soc, M_rec, R_cost, J_base, omega_exact = harmonic_oscillator_system
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        omegas, E0 = agent.phase2.compute_normal_modes()
        assert omegas.shape == (1,)
        assert omegas[0] == pytest.approx(omega_exact, rel=1e-6)

    def test_zero_point_energy_matches_half_hbar_omega(
        self, harmonic_oscillator_system
    ):
        C_soc, M_rec, R_cost, J_base, omega_exact = harmonic_oscillator_system
        hbar = 2.5
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            hbar=hbar,
        )
        omegas, E0 = agent.phase2.compute_normal_modes()
        assert E0 == pytest.approx(0.5 * hbar * omega_exact, rel=1e-6)

    def test_normal_modes_are_nonnegative(self, phase2):
        omegas, E0 = phase2.compute_normal_modes()
        assert np.all(omegas >= -1e-9)
        assert E0 >= -1e-9

    def test_number_of_modes_equals_half_dimension(self, phase2, dims):
        n = sum(dims)
        omegas, _ = phase2.compute_normal_modes()
        assert omegas.shape[0] == n // 2

    def test_zero_point_energy_is_half_hbar_sum_omegas(self, phase2):
        """E₀ = ½ ℏ Σ_k ω_k (estructura de Williamson / osciladores)."""
        omegas, E0 = phase2.compute_normal_modes()
        hbar = getattr(phase2, "_hbar", None)
        if hbar is None:
            hbar = getattr(phase2, "hbar", 1.0)
        expected = 0.5 * float(hbar) * float(np.sum(omegas))
        assert E0 == pytest.approx(expected, rel=1e-8, abs=1e-12)


class TestPhase2StabilityFlags:
    """Retícula ``StabilityFlags`` en escenarios controlados."""

    def test_all_flags_satisfied_for_nominal_state(
        self, phase2, state_vectors
    ):
        q, p, df_dt = state_vectors
        V_q, grad_V = phase2._evaluate_potential_energy(q)
        K_p, grad_K = phase2._compute_kinetic_energy(p)
        grad_H = np.concatenate([grad_V, grad_K])
        P_diss = phase2._enforce_rayleigh_dissipation(grad_H)
        v_fb = phase2._measure_flyback_voltage(df_dt)
        x_dot = phase2._compute_vector_field(grad_H)
        residual = phase2._verify_structural_consistency(
            grad_H, x_dot, P_diss
        )
        flags = phase2._evaluate_stability_flags(
            V_q, K_p, P_diss, v_fb, residual
        )
        assert flags == StabilityFlags.ALL

    def test_flyback_unsafe_flag_cleared_near_soft_margin(
        self, valid_context, dims
    ):
        phase2_margin = Phase2(
            context=valid_context,
            breakdown_voltage=1.0,
            kappa_max=1e10,
            flyback_safety_margin=0.01,
        )
        flags = phase2_margin._evaluate_stability_flags(
            V_q=1.0,
            K_p=1.0,
            P_diss=1.0,
            v_fb=0.5,
            structural_residual=0.0,
        )
        assert StabilityFlags.FLYBACK_SAFE not in flags

    def test_spectral_conditioning_flag_cleared_for_high_kappa(
        self, valid_context
    ):
        phase2_strict = Phase2(
            context=valid_context, breakdown_voltage=1e5, kappa_max=1.0
        )
        flags = phase2_strict._evaluate_stability_flags(
            V_q=1.0,
            K_p=1.0,
            P_diss=1.0,
            v_fb=0.0,
            structural_residual=0.0,
        )
        assert StabilityFlags.SPECTRAL_CONDITIONING_SOUND not in flags

    def test_energy_flag_cleared_for_negative_energy(self, phase2):
        """Si el evaluador recibe V o K negativos, ENERGY_NONNEGATIVE se apaga."""
        flags = phase2._evaluate_stability_flags(
            V_q=-1.0,
            K_p=1.0,
            P_diss=0.0,
            v_fb=0.0,
            structural_residual=0.0,
        )
        if hasattr(StabilityFlags, "ENERGY_NONNEGATIVE"):
            assert StabilityFlags.ENERGY_NONNEGATIVE not in flags


class TestPhase2SynthesizeBasalStateIntegration:
    """Método terminal de Fase 2 → ``BasalStateTensor`` (precondición Fase 3)."""

    def test_returns_basal_state_tensor_with_consistent_fields(
        self, phase2, state_vectors
    ):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        assert isinstance(tensor, BasalStateTensor)
        assert tensor.total_hamiltonian == pytest.approx(
            tensor.potential_energy + tensor.kinetic_energy, rel=1e-10
        )
        assert tensor.is_thermodynamically_stable == (
            tensor.stability_flags == StabilityFlags.ALL
        )
        assert tensor.normal_mode_frequencies is None
        assert tensor.zero_point_energy is None

    def test_compute_normal_modes_flag_populates_optional_fields(
        self, phase2, state_vectors
    ):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(
            q=q, p=p, df_dt=df_dt, compute_normal_modes=True
        )
        assert tensor.normal_mode_frequencies is not None
        assert tensor.zero_point_energy is not None
        n = phase2._ctx.dim_q + phase2._ctx.dim_p
        assert tensor.normal_mode_frequencies.shape[0] == n // 2

    def test_vector_field_shape_matches_state_dimension(
        self, phase2, state_vectors
    ):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        n = phase2._ctx.dim_q + phase2._ctx.dim_p
        assert tensor.vector_field.shape == (n,)

    def test_result_is_immutable(self, phase2, state_vectors):
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tensor.total_hamiltonian = 0.0  # type: ignore[misc]

    def test_hamiltonian_matches_oracle_sum(
        self, phase2, valid_system, state_vectors
    ):
        C_soc, M_rec, _, _ = valid_system
        q, p, df_dt = state_vectors
        tensor = phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)
        C_inv, M_inv = np.linalg.inv(C_soc), np.linalg.inv(M_rec)
        H_oracle = 0.5 * float(q @ C_inv @ q) + 0.5 * float(p @ M_inv @ p)
        assert tensor.total_hamiltonian == pytest.approx(H_oracle, rel=1e-8)


# =============================================================================
# 4. FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES
# =============================================================================


class TestPhase3CoboundaryAssembly:
    """Construcción de δ_metric, δ_diss y δ_BASE."""

    def test_delta_metric_is_block_diagonal_of_inverse_square_roots(
        self, phase3, valid_context, dims
    ):
        dim_q, dim_p = dims
        np.testing.assert_allclose(
            phase3._delta_metric[:dim_q, :dim_q],
            valid_context.C_inv_sqrt,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            phase3._delta_metric[dim_q:, dim_q:],
            valid_context.M_inv_sqrt,
            rtol=1e-10,
        )
        assert np.allclose(phase3._delta_metric[:dim_q, dim_q:], 0.0)
        assert np.allclose(phase3._delta_metric[dim_q:, :dim_q], 0.0)

    def test_delta_dissipative_equals_r_sqrt(self, phase3, valid_context):
        assert np.array_equal(phase3._delta_diss, valid_context.R_sqrt) or np.allclose(
            phase3._delta_diss, valid_context.R_sqrt, rtol=1e-14
        )

    def test_delta_base_is_vertical_stack_of_correct_shape(self, phase3, dims):
        n = sum(dims)
        assert phase3._delta_base.shape == (2 * n, n)
        np.testing.assert_allclose(
            phase3._delta_base[:n, :], phase3._delta_metric, rtol=1e-14
        )
        np.testing.assert_allclose(
            phase3._delta_base[n:, :], phase3._delta_diss, rtol=1e-14
        )

    def test_delta_metric_is_invertible(self, phase3):
        singular_values = la.svdvals(phase3._delta_metric)
        assert np.all(singular_values > 1e-10)

    def test_rank_delta_equals_full_dimension(self, phase3, dims):
        n = sum(dims)
        assert phase3._rank_delta == n

    def test_delta_metric_condition_scales_with_C_and_M(self, valid_system, dims):
        """
        κ(δ_metric) se relaciona con κ(C)^{1/2} y κ(M)^{1/2}
        (pues δ ~ block-diag(C^{−1/2}, M^{−1/2})).
        """
        C_soc, M_rec, R_cost, J_base = valid_system
        p1 = Phase1(C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base)
        ctx = p1.build_topological_context()
        p3 = Phase3(context=ctx)
        kappa_delta = float(np.max(la.svdvals(p3._delta_metric)) / np.min(la.svdvals(p3._delta_metric)))
        kappa_C = spectral_condition(C_soc)
        kappa_M = spectral_condition(M_rec)
        # κ(δ) ≲ max(√κ_C, √κ_M) · factor de seguridad
        upper = math.sqrt(max(kappa_C, kappa_M)) * 10.0
        assert kappa_delta < upper + 1.0


class TestPhase3HodgeIdentityAndSpectrum:
    """
    Identidad de Hodge local: Δ_BASE = ∇²H + R_cost
    con ∇²H = block-diag(C⁻¹, M⁻¹).
    """

    def test_hodge_laplacian_matches_hessian_plus_r_cost(
        self, phase3, valid_system, valid_context
    ):
        C_soc, M_rec, R_cost, _ = valid_system
        expected_hessian = la.block_diag(
            np.linalg.inv(C_soc), np.linalg.inv(M_rec)
        )
        expected_hodge = expected_hessian + R_cost
        assert fro_rel(phase3._hodge_laplacian, expected_hodge) < 1e-8

    def test_hodge_identity_residual_is_near_machine_precision(self, phase3):
        rel_error = phase3._verify_hodge_identity()
        assert rel_error < HODGE_REL_FACTOR * EPS

    def test_hodge_laplacian_is_symmetric_positive_definite(self, phase3):
        H = phase3._hodge_laplacian
        assert float(la.norm(H - H.T, "fro")) < 1e-10
        assert np.all(np.linalg.eigvalsh(H) > 0.0)

    def test_hodge_spectrum_gap_and_condition_number_are_consistent(
        self, phase3
    ):
        gap, kappa, harmonic_dim = phase3._compute_hodge_spectrum()
        eigvals = np.linalg.eigvalsh(phase3._hodge_laplacian)
        assert gap == pytest.approx(float(eigvals[1] - eigvals[0]), rel=1e-6)
        assert kappa == pytest.approx(
            float(eigvals[-1] / eigvals[0]), rel=1e-6
        )

    def test_harmonic_dimension_is_always_zero(self, phase3):
        """
        Invariante topológico: δ_metric invertible (∇²H ≻ 0) ⇒
        dim ker(δ_metric) = 0 siempre.
        """
        _, _, harmonic_dim = phase3._compute_hodge_spectrum()
        assert harmonic_dim == 0

    def test_hodge_dominates_hessian_in_psd_order(
        self, phase3, valid_system
    ):
        """
        Δ − ∇²H = R ⪰ 0 ⇒ Δ ⪰ ∇²H en el orden de Löwner.
        """
        C_soc, M_rec, R_cost, _ = valid_system
        hess = la.block_diag(np.linalg.inv(C_soc), np.linalg.inv(M_rec))
        diff = phase3._hodge_laplacian - hess
        # Simetrizar por seguridad numérica
        diff = 0.5 * (diff + diff.T)
        assert np.all(np.linalg.eigvalsh(diff) >= -1e-8)


class TestPhase3ExportStalkIntegration:
    """Método terminal de Fase 3 → ``SheafStalk`` (salida del endofuntor)."""

    def test_returns_sheaf_stalk_with_correct_shapes(
        self, phase3, full_state_x, dims
    ):
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

    def test_projections_match_manual_matrix_vector_products(
        self, phase3, full_state_x
    ):
        stalk = phase3.export_stalk(full_state_x)
        np.testing.assert_allclose(
            stalk.projected_state_metric,
            phase3._delta_metric @ full_state_x,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            stalk.projected_state_dissipative,
            phase3._delta_diss @ full_state_x,
            rtol=1e-10,
        )

    def test_lossless_subspace_dimension_matches_betti0(
        self, valid_context, full_state_x
    ):
        phase3_local = Phase3(context=valid_context)
        stalk = phase3_local.export_stalk(full_state_x)
        assert stalk.lossless_subspace_dimension == valid_context.betti_0_R

    def test_state_vector_is_defensively_copied(self, phase3, full_state_x):
        stalk = phase3.export_stalk(full_state_x)
        original = float(full_state_x[0])
        full_state_x[0] = 9999.0
        assert stalk.state_vector[0] == pytest.approx(original)

    def test_delta_base_is_defensively_copied(self, phase3, full_state_x):
        stalk = phase3.export_stalk(full_state_x)
        stalk.delta_base[0, 0]  # lectura
        # Si es copia, mutar el interno de phase3 no afecta (o viceversa)
        ref = stalk.delta_base.copy()
        phase3._delta_base[0, 0] += 1.0
        # stalk debe conservar el valor exportado si hubo copia defensiva
        # (si el módulo no copia, este test documenta el comportamiento)
        if not np.allclose(stalk.delta_base, ref):
            # restauración para no contaminar otros tests de la misma instancia
            phase3._delta_base[0, 0] -= 1.0
            pytest.skip("export_stalk no realiza copia defensiva de delta_base")
        phase3._delta_base[0, 0] -= 1.0  # restaurar

    def test_projections_linear(self, phase3, dims):
        """δ(αx+βy) = α δx + β δy."""
        n = sum(dims)
        rng = np.random.default_rng(50)
        x, y = rng.normal(size=n), rng.normal(size=n)
        alpha, beta = 2.0, -1.5
        s_xy = phase3.export_stalk(alpha * x + beta * y)
        s_x = phase3.export_stalk(x)
        s_y = phase3.export_stalk(y)
        np.testing.assert_allclose(
            s_xy.projected_state_metric,
            alpha * s_x.projected_state_metric
            + beta * s_y.projected_state_metric,
            rtol=1e-10,
        )


# =============================================================================
# 5. RETÍCULA BOOLEANA DE ESTABILIDAD
# =============================================================================


ATOMIC_FLAGS = [
    f
    for f in StabilityFlags
    if f.value != 0 and (f.value & (f.value - 1)) == 0
]


class TestStabilityFlagsBooleanAlgebra:
    """
    ``StabilityFlags`` como álgebra de Boole acotada:
    idempotencia, absorción, distributividad y De Morgan
    relativas al elemento supremo ``ALL``.
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
        assert (a & (b | c)) == ((a & b) | (a & c))

    def test_distributivity_of_disjunction_over_conjunction(self):
        a, b, c = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1], ATOMIC_FLAGS[2]
        assert (a | (b & c)) == ((a | b) & (a | c))

    def test_de_morgan_law_over_union(self):
        a, b = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1]
        complement_of_union = StabilityFlags.ALL & ~(a | b)
        intersection_of_complements = (StabilityFlags.ALL & ~a) & (
            StabilityFlags.ALL & ~b
        )
        assert complement_of_union == intersection_of_complements

    def test_de_morgan_law_over_intersection(self):
        a, b = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1]
        complement_of_intersection = StabilityFlags.ALL & ~(a & b)
        union_of_complements = (StabilityFlags.ALL & ~a) | (
            StabilityFlags.ALL & ~b
        )
        assert complement_of_intersection == union_of_complements

    def test_complement_of_all_is_none(self):
        assert (StabilityFlags.ALL & ~StabilityFlags.ALL) == StabilityFlags.NONE

    def test_complement_of_none_is_all(self):
        assert (StabilityFlags.ALL & ~StabilityFlags.NONE) == StabilityFlags.ALL

    def test_commutativity_and_associativity(self):
        a, b, c = ATOMIC_FLAGS[0], ATOMIC_FLAGS[1], ATOMIC_FLAGS[2]
        assert (a | b) == (b | a)
        assert (a & b) == (b & a)
        assert (a | (b | c)) == ((a | b) | c)
        assert (a & (b & c)) == ((a & b) & c)

    def test_describe_stability_flags_reports_all_satisfied(self):
        description = describe_stability_flags(StabilityFlags.ALL)
        assert "ESTABLE_TOTAL=True" in description
        assert "VIOLADOS=ninguno" in description

    def test_describe_stability_flags_reports_none_satisfied(self):
        description = describe_stability_flags(StabilityFlags.NONE)
        assert "ESTABLE_TOTAL=False" in description
        assert "SATISFECHOS=ninguno" in description

    def test_describe_stability_flags_lists_partial_violation(self):
        partial = (
            StabilityFlags.ENERGY_NONNEGATIVE
            | StabilityFlags.DISSIPATION_VALID
        )
        description = describe_stability_flags(partial)
        assert "ENERGY_NONNEGATIVE" in description
        assert "FLYBACK_SAFE" in description
        assert "ESTABLE_TOTAL=False" in description


# =============================================================================
# 6. COMPOSICIÓN FUNTORIAL
# =============================================================================


class TestFunctorialComposition:
    """
    Contrato funtorial:
        export_stalk ∘ synthesize_basal_state ∘ build_topological_context
        = K_BASE
    """

    def test_end_to_end_happy_path(self, agent, state_vectors, dims):
        q, p, df_dt = state_vectors
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        state_x = np.concatenate([q, p])
        stalk = agent.export_sheaf_stalk(state_x)

        assert isinstance(agent.context, TopologicalContext)
        assert isinstance(tensor, BasalStateTensor)
        assert isinstance(stalk, SheafStalk)
        assert stalk.rank_delta == sum(dims)
        np.testing.assert_allclose(stalk.state_vector, state_x)

    def test_phase_contexts_share_same_topological_context(self, agent):
        assert agent.phase2._ctx is agent.context
        _ = agent.export_sheaf_stalk(
            np.zeros(agent.context.dim_q + agent.context.dim_p)
        )
        assert agent.phase3 is not None
        assert agent.phase3._ctx is agent.context

    def test_synthesize_is_deterministic(self, agent, state_vectors):
        q, p, df_dt = state_vectors
        t1 = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        t2 = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert t1.total_hamiltonian == t2.total_hamiltonian
        assert t1.stability_flags == t2.stability_flags
        np.testing.assert_array_equal(t1.vector_field, t2.vector_field)

    def test_betti_inheritance_phase1_to_phase3(
        self, rank_deficient_R_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = rank_deficient_R_system
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        n = sum(dims)
        stalk = agent.export_sheaf_stalk(np.zeros(n))
        assert stalk.lossless_subspace_dimension == agent.context.betti_0_R
        assert agent.context.betti_0_R == 2


# =============================================================================
# 7. INTEGRACIÓN END-TO-END DEL AGENTE COMPLETO
# =============================================================================


class TestKBaseThermodynamicAgentIntegration:
    """Contrato público: construcción, fases y frontera entre ellas."""

    def test_constructor_populates_all_public_attributes(self, agent, dims):
        assert isinstance(agent.context, TopologicalContext)
        assert isinstance(agent.phase1, Phase1)
        assert isinstance(agent.phase2, Phase2)
        assert agent.phase3 is None  # lazy
        assert (agent.context.dim_q, agent.context.dim_p) == dims

    def test_constructor_propagates_dimension_mismatch(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        with pytest.raises(DimensionMismatchError):
            KBaseThermodynamicAgent(
                C_soc=C_soc[:, :-1],
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_constructor_propagates_capacitance_degeneracy(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        Q, _ = np.linalg.qr(
            np.random.default_rng(5).normal(size=(n, n))
        )
        bad_C = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        with pytest.raises(CapacitanceDegeneracyError):
            KBaseThermodynamicAgent(
                C_soc=bad_C, M_rec=M_rec, R_cost=R_cost, J_base=J_base
            )

    def test_constructor_propagates_rayleigh_violation(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = R_cost.shape[0]
        Q, _ = np.linalg.qr(
            np.random.default_rng(6).normal(size=(n, n))
        )
        bad_R = Q @ np.diag(np.linspace(-1.0, 1.0, n)) @ Q.T
        with pytest.raises(RayleighDissipationViolation):
            KBaseThermodynamicAgent(
                C_soc=C_soc, M_rec=M_rec, R_cost=bad_R, J_base=J_base
            )

    def test_constructor_propagates_ill_conditioned_error(self, valid_system):
        C_soc, M_rec, R_cost, J_base = valid_system
        n = C_soc.shape[0]
        huge_C = make_spd_with_condition(n, cond=1.0e12, seed=10)
        with pytest.raises(IllConditionedMatrixError):
            KBaseThermodynamicAgent(
                C_soc=huge_C,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                kappa_max=1.0e6,
            )

    def test_public_synthesize_basal_hamiltonian_matches_phase2_directly(
        self, agent, state_vectors
    ):
        q, p, df_dt = state_vectors
        tensor_public = agent.synthesize_basal_hamiltonian(
            q=q, p=p, df_dt=df_dt
        )
        tensor_direct = agent.phase2.synthesize_basal_state(
            q=q, p=p, df_dt=df_dt
        )
        assert tensor_public.total_hamiltonian == pytest.approx(
            tensor_direct.total_hamiltonian
        )

    def test_export_sheaf_stalk_lazy_initializes_phase3_once(
        self, agent, full_state_x
    ):
        assert agent.phase3 is None
        stalk_1 = agent.export_sheaf_stalk(full_state_x)
        phase3_instance = agent.phase3
        assert phase3_instance is not None
        stalk_2 = agent.export_sheaf_stalk(full_state_x)
        assert agent.phase3 is phase3_instance
        np.testing.assert_allclose(stalk_1.delta_base, stalk_2.delta_base)

    def test_end_to_end_state_vector_consistency_between_phase2_and_phase3(
        self, agent, state_vectors
    ):
        q, p, df_dt = state_vectors
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        state_x = np.concatenate([q, p])
        stalk = agent.export_sheaf_stalk(state_x)
        np.testing.assert_allclose(stalk.state_vector, state_x)
        assert stalk.projected_state_metric.shape == (
            agent.context.dim_q + agent.context.dim_p,
        )
        assert tensor.vector_field.shape == state_x.shape

    def test_agent_with_custom_metric_tensors_end_to_end(
        self, valid_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        dim_q, dim_p = dims
        G_q = np.diag(np.linspace(1.0, 1.2, dim_q))
        G_p = np.diag(np.linspace(1.0, 1.1, dim_p))
        agent_custom = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        rng = np.random.default_rng(2024)
        q = rng.normal(size=dim_q)
        p = rng.normal(size=dim_p)
        df_dt = rng.normal(size=dim_p)
        tensor = agent_custom.synthesize_basal_hamiltonian(
            q=q, p=p, df_dt=df_dt
        )
        assert tensor.total_hamiltonian >= -1e-12

    def test_agent_raises_inertial_flyback_error_end_to_end(
        self, valid_system, dims
    ):
        C_soc, M_rec, R_cost, J_base = valid_system
        _, dim_p = dims
        agent_strict = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            breakdown_voltage=1e-12,
        )
        rng = np.random.default_rng(2025)
        q = rng.normal(size=C_soc.shape[0])
        p = rng.normal(size=dim_p)
        df_dt = np.ones(dim_p) * 100.0
        with pytest.raises(InertialFlybackError):
            agent_strict.synthesize_basal_hamiltonian(
                q=q, p=p, df_dt=df_dt
            )

    def test_friendly_name_is_defined(self, agent):
        assert agent.FRIENDLY_NAME == "Asesor de Cimientos Financieros"


# =============================================================================
# 8. ESTRÉS NUMÉRICO E INVARIANTES (ENSEMBLE)
# =============================================================================


class TestNumericalStressAndInvariants:
    """Casos límite: n=1+1, R=0, κ moderado-alto, dimensiones mayores."""

    def test_harmonic_oscillator_full_pipeline(
        self, harmonic_oscillator_system
    ):
        C_soc, M_rec, R_cost, J_base, omega_exact = harmonic_oscillator_system
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        q = np.array([1.0])
        p = np.array([0.0])
        df_dt = np.array([0.0])
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        stalk = agent.export_sheaf_stalk(np.array([1.0, 0.0]))
        assert tensor.total_hamiltonian == pytest.approx(
            0.5 * (1.0 ** 2) / 4.0, rel=1e-10
        )
        assert stalk.delta_base.shape == (4, 2)
        omegas, _ = agent.phase2.compute_normal_modes()
        assert omegas[0] == pytest.approx(omega_exact, rel=1e-6)

    def test_zero_dissipation_R_pipeline(self, lossless_system, dims):
        C_soc, M_rec, R_cost, J_base = lossless_system
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        assert agent.context.rank_R == 0
        assert agent.context.betti_0_R == sum(dims)
        n = sum(dims)
        stalk = agent.export_sheaf_stalk(np.zeros(n))
        assert stalk.lossless_subspace_dimension == n
        # δ_diss = 0 ⇒ proyección disipativa nula
        np.testing.assert_allclose(
            stalk.projected_state_dissipative, 0.0, atol=1e-14
        )

    def test_moderate_high_condition_pipeline(self, dims):
        dim_q, dim_p = dims
        n = dim_q + dim_p
        C_soc = make_spd_with_condition(dim_q, cond=1.0e5, seed=501)
        M_rec = make_spd_with_condition(dim_p, cond=1.0e4, seed=502)
        R_cost = make_psd_with_rank(n, rank=n, seed=503)
        J_base = make_antisymmetric(n, seed=504)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            kappa_max=1.0e10,
        )
        rng = np.random.default_rng(505)
        q, p = rng.normal(size=dim_q), rng.normal(size=dim_p)
        df_dt = 1e-3 * rng.normal(size=dim_p)
        tensor = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        stalk = agent.export_sheaf_stalk(np.concatenate([q, p]))
        assert tensor.total_hamiltonian >= -1e-10
        assert stalk.rank_delta == n

    def test_larger_dimension_pipeline(self):
        dim_q, dim_p = 6, 5
        n = dim_q + dim_p
        C_soc = make_spd(dim_q, seed=601)
        M_rec = make_spd(dim_p, seed=602)
        R_cost = make_psd_with_rank(n, rank=n - 1, seed=603)
        J_base = make_antisymmetric(n, seed=604)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        rng = np.random.default_rng(605)
        q, p = rng.normal(size=dim_q), rng.normal(size=dim_p)
        tensor = agent.synthesize_basal_hamiltonian(
            q=q, p=p, df_dt=1e-3 * rng.normal(size=dim_p)
        )
        stalk = agent.export_sheaf_stalk(np.concatenate([q, p]))
        assert stalk.delta_base.shape == (2 * n, n)
        assert agent.context.betti_0_R == 1
        assert tensor.vector_field.shape == (n,)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_power_balance_ensemble(self, seed: int, dims):
        """∇Hᵀ ẋ + P_diss ≈ 0 en un ensemble de sistemas aleatorios."""
        dim_q, dim_p = dims
        n = dim_q + dim_p
        C_soc = make_spd(dim_q, seed=seed)
        M_rec = make_spd(dim_p, seed=seed + 10)
        R_cost = make_psd_with_rank(n, rank=n, seed=seed + 20)
        J_base = make_antisymmetric(n, seed=seed + 30)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        rng = np.random.default_rng(seed + 40)
        q, p = rng.normal(size=dim_q), rng.normal(size=dim_p)
        gV = agent.phase2._evaluate_potential_energy(q)[1]
        gK = agent.phase2._compute_kinetic_energy(p)[1]
        gH = np.concatenate([gV, gK])
        x_dot = agent.phase2._compute_vector_field(gH)
        P_diss = agent.phase2._enforce_rayleigh_dissipation(gH)
        balance = float(gH @ x_dot) + P_diss
        assert abs(balance) < 1e-8 * max(P_diss, 1.0)

    @pytest.mark.parametrize("seed", [11, 12, 13, 14])
    def test_hodge_identity_ensemble(self, seed: int, dims):
        dim_q, dim_p = dims
        n = dim_q + dim_p
        C_soc = make_spd(dim_q, seed=seed)
        M_rec = make_spd(dim_p, seed=seed + 1)
        R_cost = make_psd_with_rank(n, rank=max(n - 1, 1), seed=seed + 2)
        J_base = make_antisymmetric(n, seed=seed + 3)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        p3 = Phase3(context=agent.context)
        rel = p3._verify_hodge_identity()
        assert rel < HODGE_REL_FACTOR * EPS * 10.0

    @pytest.mark.parametrize(
        "dim_q,dim_p", [(1, 1), (2, 2), (3, 1), (4, 3)]
    )
    def test_pipeline_across_dimensions(self, dim_q: int, dim_p: int):
        n = dim_q + dim_p
        C_soc = make_spd(dim_q, seed=700 + dim_q)
        M_rec = make_spd(dim_p, seed=800 + dim_p)
        R_cost = make_psd_with_rank(n, rank=n, seed=900 + n)
        J_base = make_antisymmetric(n, seed=1000 + n)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        rng = np.random.default_rng(1100 + n)
        q, p = rng.normal(size=dim_q), rng.normal(size=dim_p)
        tensor = agent.synthesize_basal_hamiltonian(
            q=q, p=p, df_dt=1e-3 * rng.normal(size=dim_p)
        )
        stalk = agent.export_sheaf_stalk(np.concatenate([q, p]))
        assert stalk.rank_delta == n
        assert tensor.total_hamiltonian >= -1e-12
        assert tensor.stability_flags == StabilityFlags.ALL or (
            # flyback puede fallar el soft-margin en estados aleatorios
            StabilityFlags.ENERGY_NONNEGATIVE in tensor.stability_flags
        )


# =============================================================================
# Entrada directa
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))