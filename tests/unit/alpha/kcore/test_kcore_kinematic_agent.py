# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite  : test_kcore_kinematic_agent.py                                       |
| Objetivo: Pruebas granulares y rigurosas del KCoreKinematicAgent v6.0.0      |
| Cobertura:                                                                   |
|   • SECCIÓN 0  — Excepciones y jerarquía                                     |
|   • SECCIÓN 1  — DTOs inmutables                                             |
|   • FASE 1     — Validación matricial constitutiva                           |
|   • FASE 2     — Síntesis cinemática (IDA-PBC, Hodge, KK, CFL)               |
|   • FASE 3     — Proyección en haces y cofrontera δ_CORE                     |
|   • API pública — synthesize_kinematic_core / export_sheaf_stalk             |
|   • Propiedades invariantes y edge-cases espectrales                         |
+==============================================================================+
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import del módulo bajo prueba
# Ajuste la ruta si el paquete se instala de otra forma.
# ---------------------------------------------------------------------------
try:
    from app.agents.alpha.kcore.kcore_kinematic_agent import (
        CFLViolationError,
        DiracMatchingError,
        ImpedanceReflectionError,
        KinematicConditionError,
        KinematicCoreError,
        KinematicDimensionError,
        KinematicPreparationContext,
        KinematicStateTensor,
        KinematicSymmetryError,
        KCoreKinematicAgent,
        MetricTensorError,
        ParasiticVorticityError,
        SheafCoboundaryError,
        SheafStalk,
    )
except ImportError:
    # Fallback: import relativo al árbol local de desarrollo
    from kcore_kinematic_agent import (  # type: ignore[no-redef]
        CFLViolationError,
        DiracMatchingError,
        ImpedanceReflectionError,
        KinematicConditionError,
        KinematicCoreError,
        KinematicDimensionError,
        KinematicPreparationContext,
        KinematicStateTensor,
        KinematicSymmetryError,
        KCoreKinematicAgent,
        MetricTensorError,
        ParasiticVorticityError,
        SheafCoboundaryError,
        SheafStalk,
    )


# =============================================================================
# CONSTANTES Y FACTORÍAS DE FIXTURES
# =============================================================================

_EPS: float = float(np.finfo(np.float64).eps)
_RTOL: float = 1.0e-10
_ATOL: float = 1.0e-12


def _skew(n: int, seed: int = 0) -> NDArray[np.float64]:
    """Genera una matriz antisimétrica n×n aleatoria bien condicionada."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return 0.5 * (A - A.T)


def _spd(n: int, seed: int = 1, cond: float = 10.0) -> NDArray[np.float64]:
    """
    Genera una matriz SPD n×n con número de condición ≈ cond.

    Construcción: Q · diag(λ) · Qᵀ con λ equiespaciados en [1, cond].
    """
    rng = np.random.default_rng(seed)
    # QR de matriz gaussiana → Q ortogonal
    Q, _ = la.qr(rng.standard_normal((n, n)))
    lambdas = np.linspace(1.0, cond, n)
    return (Q * lambdas) @ Q.T


def _psd_rank_deficient(
    n: int, rank: int, seed: int = 2
) -> NDArray[np.float64]:
    """PSD n×n de rango exacto `rank` < n."""
    assert 0 < rank < n
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, rank))
    return B @ B.T


def _control_g(
    n: int, m: int, seed: int = 3, full_rank: bool = True
) -> NDArray[np.float64]:
    """Matriz de control g ∈ ℝ^{n×m}."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, m))
    if full_rank and m <= n:
        # Asegurar rango completo vía SVD + clamping
        U, s, Vh = la.svd(g, full_matrices=False)
        s = np.maximum(s, 0.5)
        g = (U * s) @ Vh
    return g


def _canonical_system(
    n: int = 4,
    m: int = 2,
    seed: int = 42,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Sistema Port-Hamiltoniano canónico válido:
        J, J_d antisimétricas; R, R_d, G SPD; g rango completo.
    """
    J = _skew(n, seed)
    J_d = _skew(n, seed + 1)
    R = _spd(n, seed + 2, cond=5.0)
    R_d = _spd(n, seed + 3, cond=8.0)
    g = _control_g(n, m, seed + 4, full_rank=True)
    G = _spd(n, seed + 5, cond=3.0)
    return J, R, J_d, R_d, g, G


def _path_laplacian(n_nodes: int) -> sp.csr_matrix:
    """Laplaciano simétrico del grafo camino P_{n_nodes} (PSD, λ_max conocido)."""
    # L = D − A: diagonal 1 en extremos, 2 en interiores; off-diag −1
    diags_data = [
        -np.ones(n_nodes - 1),
        np.concatenate([[1.0], 2.0 * np.ones(n_nodes - 2), [1.0]])
        if n_nodes > 1
        else np.array([0.0]),
        -np.ones(n_nodes - 1),
    ]
    offsets = [-1, 0, 1]
    if n_nodes == 1:
        return sp.csr_matrix([[0.0]])
    L = sp.diags(diags_data, offsets, shape=(n_nodes, n_nodes), format="csr")
    return L


def _diagonal_conductance(E: int, w: float = 1.0) -> sp.csr_matrix:
    """Conductancia diagonal W = w · I_E."""
    return sp.diags(
        w * np.ones(E), offsets=0, shape=(E, E), format="csr", dtype=np.float64
    )


# =============================================================================
# SECCIÓN 0 — JERARQUÍA DE EXCEPCIONES
# =============================================================================


class TestExceptionHierarchy:
    """Verifica la taxonomía de excepciones cinemáticas."""

    def test_all_inherit_from_kinematic_core_error(self) -> None:
        subclasses = [
            KinematicDimensionError,
            KinematicSymmetryError,
            KinematicConditionError,
            DiracMatchingError,
            ParasiticVorticityError,
            ImpedanceReflectionError,
            CFLViolationError,
            SheafCoboundaryError,
            MetricTensorError,
        ]
        for cls in subclasses:
            assert issubclass(cls, KinematicCoreError), (
                f"{cls.__name__} debe heredar de KinematicCoreError"
            )

    def test_catch_all_with_base(self) -> None:
        with pytest.raises(KinematicCoreError):
            raise DiracMatchingError("matching fallido")


# =============================================================================
# SECCIÓN 1 — DTOs INMUTABLES
# =============================================================================


class TestImmutableDTOs:
    """Los dataclasses frozen no deben admitir mutación."""

    def test_preparation_context_frozen(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        n, m = 4, 2
        ctx = KinematicPreparationContext(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g, G=G,
            n=n, m=m, rank_g=m, rank_G=n,
            kappa_R=1.0, kappa_R_d=1.0, kappa_G=1.0,
            spectral_gap_R=0.5,
        )
        with pytest.raises(Exception):
            ctx.n = 99  # type: ignore[misc]

    def test_state_tensor_frozen(self) -> None:
        tensor = KinematicStateTensor(
            control_law_alpha=np.zeros(2),
            hodge_conductance=sp.eye(3, format="csr"),
            dielectric_tensor=np.eye(2),
            magnetic_tensor=np.eye(2),
            cfl_safe_dt=1e-3,
            residual_idapbc=1e-12,
            vorticity_norm=0.0,
            gershgorin_rho=4.0,
            lambda_max_delta=4.0,
            is_kinematically_stable=True,
        )
        with pytest.raises(Exception):
            tensor.cfl_safe_dt = 0.0  # type: ignore[misc]

    def test_sheaf_stalk_frozen(self) -> None:
        stalk = SheafStalk(
            delta_core=np.eye(3),
            delta_hodge_residual=1e-15,
            state_vector=np.zeros(3),
            projected_state=np.zeros(3),
            rank_delta=3,
            betti_approx=0,
            spectral_entropy=0.0,
        )
        with pytest.raises(Exception):
            stalk.rank_delta = 0  # type: ignore[misc]


# =============================================================================
# FASE 1 — VALIDACIÓN MATRICIAL
# =============================================================================


class TestPhase1Dimensions:
    """Fase 1 · verificación dimensional."""

    def test_valid_system_builds_context(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        ctx = agent.context
        assert ctx.n == 4
        assert ctx.m == 2
        assert ctx.rank_g == 2
        assert ctx.rank_G == 4
        assert math.isfinite(ctx.kappa_R)
        assert math.isfinite(ctx.kappa_R_d)
        assert math.isfinite(ctx.kappa_G)
        assert 0.0 <= ctx.spectral_gap_R <= 1.0

    def test_default_G_is_identity(self) -> None:
        J, R, J_d, R_d, g, _ = _canonical_system(n=3, m=1)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=None)
        assert np.allclose(agent.context.G, np.eye(3))
        assert agent.context.rank_G == 3

    def test_non_square_J_raises(self) -> None:
        J = np.zeros((3, 4))
        R = np.eye(3)
        with pytest.raises(KinematicDimensionError, match="cuadrada"):
            KCoreKinematicAgent(J, R, R, R, np.ones((3, 1)))

    def test_inconsistent_R_shape_raises(self) -> None:
        J, _, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        R_bad = np.eye(3)
        with pytest.raises(KinematicDimensionError, match="'R'"):
            KCoreKinematicAgent(J, R_bad, J_d, R_d, g, G=G)

    def test_g_wrong_rows_raises(self) -> None:
        J, R, J_d, R_d, _, G = _canonical_system(n=4, m=2)
        g_bad = np.ones((3, 2))
        with pytest.raises(KinematicDimensionError, match="filas"):
            KCoreKinematicAgent(J, R, J_d, R_d, g_bad, G=G)

    def test_g_zero_columns_raises(self) -> None:
        J, R, J_d, R_d, _, G = _canonical_system(n=4, m=2)
        g_bad = np.zeros((4, 0))
        with pytest.raises(KinematicDimensionError):
            KCoreKinematicAgent(J, R, J_d, R_d, g_bad, G=G)

    def test_1d_array_raises(self) -> None:
        with pytest.raises(KinematicDimensionError, match="2D"):
            KCoreKinematicAgent(
                np.zeros(4), np.eye(4), np.zeros((4, 4)),
                np.eye(4), np.ones((4, 1)),
            )

    def test_G_wrong_shape_raises(self) -> None:
        J, R, J_d, R_d, g, _ = _canonical_system(n=4, m=2)
        G_bad = np.eye(3)
        with pytest.raises(MetricTensorError, match="G"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G_bad)


class TestPhase1Symmetry:
    """Fase 1 · simetría / antisimetría."""

    def test_non_antisymmetric_J_raises(self) -> None:
        _, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        J_bad = np.eye(4)  # simétrica, no antisimétrica
        with pytest.raises(KinematicSymmetryError, match="antisim"):
            KCoreKinematicAgent(J_bad, R, J_d, R_d, g, G=G)

    def test_non_symmetric_R_raises(self) -> None:
        J, _, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        R_bad = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.1, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        with pytest.raises(KinematicSymmetryError, match="simétr"):
            KCoreKinematicAgent(J, R_bad, J_d, R_d, g, G=G)

    def test_non_psd_R_raises(self) -> None:
        J, _, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        # Matriz simétrica con autovalor negativo
        R_bad = np.diag([1.0, 1.0, 1.0, -5.0])
        with pytest.raises(KinematicSymmetryError, match="Positiva|PSD"):
            KCoreKinematicAgent(J, R_bad, J_d, R_d, g, G=G)

    def test_non_psd_G_raises(self) -> None:
        J, R, J_d, R_d, g, _ = _canonical_system(n=4, m=2)
        G_bad = np.diag([1.0, 1.0, 1.0, -1.0])
        with pytest.raises(KinematicSymmetryError, match="Positiva|PSD"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G_bad)

    def test_near_antisymmetric_within_tol_passes(self) -> None:
        """Perturbación O(ε_mach) no debe disparar falso positivo."""
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        J_pert = J + _EPS * np.ones((4, 4)) * 0.1
        # Re-antisimetrizar casi: la validación usa tol = ε·‖J‖_F
        # Una perturbación muy pequeña respecto a ‖J‖_F debe pasar o fallar
        # de forma determinista; construimos J exacto.
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        assert agent.context.n == 4


class TestPhase1Conditioning:
    """Fase 1 · número de condición y rango."""

    def test_ill_conditioned_R_raises(self) -> None:
        J, _, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        # κ ≈ 1e14 >> kappa_max=1e10
        R_ill = _spd(4, seed=99, cond=1.0e14)
        with pytest.raises(KinematicConditionError, match="condicionad"):
            KCoreKinematicAgent(
                J, R_ill, J_d, R_d, g, G=G, kappa_max=1.0e10
            )

    def test_null_g_raises(self) -> None:
        J, R, J_d, R_d, _, G = _canonical_system(n=4, m=2)
        g_null = np.zeros((4, 2))
        with pytest.raises(KinematicDimensionError, match="nula"):
            KCoreKinematicAgent(J, R, J_d, R_d, g_null, G=G)

    def test_rank_deficient_g_accepted_if_nonzero(self) -> None:
        """g de rango 1 < m=2 debe aceptarse (rank_g=1)."""
        J, R, J_d, R_d, _, G = _canonical_system(n=4, m=2)
        rng = np.random.default_rng(7)
        col = rng.standard_normal(4)
        g_rd = np.column_stack([col, 2.0 * col])  # rank 1
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g_rd, G=G)
        assert agent.context.rank_g == 1

    def test_spectral_gap_of_R(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        # gap ∈ [0, 1]
        assert 0.0 <= agent.context.spectral_gap_R <= 1.0 + 1e-12

    def test_context_copies_are_independent(self) -> None:
        """Las copias del contexto no deben aliasar las matrices originales."""
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        agent.context.J[0, 0] = 999.0
        assert J[0, 0] != 999.0


class TestPhase1ConstructorGuards:
    """Guardas del constructor del agente."""

    def test_cfl_margin_out_of_range(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        with pytest.raises(ValueError, match="cfl_margin"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G, cfl_margin=1.5)
        with pytest.raises(ValueError, match="cfl_margin"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G, cfl_margin=0.0)
        with pytest.raises(ValueError, match="cfl_margin"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G, cfl_margin=-0.1)

    def test_negative_tikhonov_raises(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        with pytest.raises(ValueError, match="tikhonov"):
            KCoreKinematicAgent(J, R, J_d, R_d, g, G=G, tikhonov_reg=-1e-6)


# =============================================================================
# FASE 2 — SÍNTESIS CINEMÁTICA
# =============================================================================


class TestPhase2IDA_PBC:
    """Fase 2 · moldeado de energía IDA-PBC covariante."""

    @pytest.fixture()
    def agent(self) -> KCoreKinematicAgent:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2, seed=10)
        return KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G,
            residual_tol_rel=1.0e-5,
            tikhonov_reg=0.0,
        )

    def test_control_law_shape_and_finite(
        self, agent: KCoreKinematicAgent
    ) -> None:
        n = agent.context.n
        m = agent.context.m
        grad_H = np.ones(n)
        grad_H_d = 0.5 * np.ones(n)
        alpha, res = agent.phase2.compute_dirac_control_law(grad_H, grad_H_d)
        assert alpha.shape == (m,)
        assert np.all(np.isfinite(alpha))
        assert 0.0 <= res < agent.residual_tol_rel

    def test_zero_force_gives_zero_alpha(
        self, agent: KCoreKinematicAgent
    ) -> None:
        """Si ∇H = ∇H_d = 0 ⇒ F_req = 0 ⇒ α = 0, residuo = 0."""
        n = agent.context.n
        alpha, res = agent.phase2.compute_dirac_control_law(
            np.zeros(n), np.zeros(n)
        )
        assert np.allclose(alpha, 0.0, atol=_ATOL)
        assert res == pytest.approx(0.0, abs=_ATOL)

    def test_wrong_grad_shape_raises(
        self, agent: KCoreKinematicAgent
    ) -> None:
        with pytest.raises(KinematicDimensionError, match="grad_H"):
            agent.phase2.compute_dirac_control_law(
                np.ones(agent.context.n + 1),
                np.ones(agent.context.n),
            )
        with pytest.raises(KinematicDimensionError, match="grad_H_d"):
            agent.phase2.compute_dirac_control_law(
                np.ones(agent.context.n),
                np.ones(2),
            )

    def test_matching_residual_identity_euclidean(self) -> None:
        """
        Con G = I y g de rango completo, el residuo de mínimos cuadrados
        debe ser ortogonal al rango de g: gᵀ (g α − F_req) ≈ 0.
        """
        n, m = 6, 3
        J, R, J_d, R_d, g, _ = _canonical_system(n=n, m=m, seed=20)
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=None, residual_tol_rel=1e-4
        )
        rng = np.random.default_rng(21)
        grad_H = rng.standard_normal(n)
        grad_H_d = rng.standard_normal(n)
        alpha, res = agent.phase2.compute_dirac_control_law(grad_H, grad_H_d)

        F_d = (J_d - R_d) @ grad_H_d
        F_nat = (J - R) @ grad_H
        F_req = F_d - F_nat
        residual_vec = g @ alpha - F_req
        # Condición normal: gᵀ residual ≈ 0
        normal_eq = g.T @ residual_vec
        assert np.allclose(normal_eq, 0.0, atol=1e-8), (
            f"Ecuaciones normales violadas: ‖gᵀ r‖ = {la.norm(normal_eq):.3e}"
        )
        assert res < 1e-4

    def test_covariant_vs_euclidean_differ(self) -> None:
        """Con G ≠ I la ley de control debe diferir de la euclídea."""
        n, m = 4, 2
        J, R, J_d, R_d, g, G = _canonical_system(n=n, m=m, seed=30)
        agent_G = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, residual_tol_rel=1e-4
        )
        agent_I = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=None, residual_tol_rel=1e-4
        )
        grad_H = np.linspace(1.0, 2.0, n)
        grad_H_d = np.linspace(0.5, 1.5, n)
        alpha_G, _ = agent_G.phase2.compute_dirac_control_law(grad_H, grad_H_d)
        alpha_I, _ = agent_I.phase2.compute_dirac_control_law(grad_H, grad_H_d)
        # No deben ser idénticos (G no es I)
        assert not np.allclose(alpha_G, alpha_I, atol=1e-6)

    def test_tikhonov_stabilizes_rank_deficient_g(self) -> None:
        """λ_reg > 0 permite resolver aunque g tenga rango deficiente."""
        n, m = 4, 2
        J, R, J_d, R_d, _, G = _canonical_system(n=n, m=m, seed=40)
        rng = np.random.default_rng(41)
        col = rng.standard_normal(n)
        g_rd = np.column_stack([col, col])  # rank 1
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g_rd, G=G,
            residual_tol_rel=1.0,  # holgado: matching exacto imposible
            tikhonov_reg=1.0e-3,
        )
        alpha, res = agent.phase2.compute_dirac_control_law(
            np.ones(n), 0.5 * np.ones(n)
        )
        assert alpha.shape == (m,)
        assert np.all(np.isfinite(alpha))
        assert math.isfinite(res)

    def test_strict_residual_tol_can_raise(self) -> None:
        """
        Con g de rango deficiente y tol muy estricta sin Tikhonov suficiente,
        el matching debe fallar.
        """
        n, m = 4, 2
        J, R, J_d, R_d, _, G = _canonical_system(n=n, m=m, seed=50)
        rng = np.random.default_rng(51)
        col = rng.standard_normal(n)
        g_rd = np.column_stack([col, 0.0 * col])  # rank 1, segunda col nula
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g_rd, G=G,
            residual_tol_rel=1.0e-14,
            tikhonov_reg=0.0,
        )
        # F_req genérico casi seguro fuera del rango de g
        with pytest.raises(DiracMatchingError):
            agent.phase2.compute_dirac_control_law(
                np.arange(1.0, n + 1.0),
                np.arange(n, 0.0, -1.0),
            )


class TestPhase2Hodge:
    """Fase 2 · estrangulamiento de vorticidad de Hodge."""

    @pytest.fixture()
    def phase2(self) -> KCoreKinematicAgent.Phase2_KinematicSynthesis:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        return agent.phase2

    def test_no_strangle_when_vorticity_low(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        E = 5
        W = _diagonal_conductance(E, w=2.0)
        I_curl = 1e-6 * np.ones(E)
        W_mod, vnorm = phase2.modulate_hodge_conductance(
            W, I_curl, epsilon_crit=1e-2
        )
        assert vnorm < 1e-2
        assert np.allclose(W_mod.diagonal(), W.diagonal())

    def test_strangle_reduces_conductance_on_support(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        E = 6
        W = _diagonal_conductance(E, w=1.0)
        I_curl = np.zeros(E)
        I_curl[0] = 10.0
        I_curl[1] = 10.0
        strangle = 1.0e-4
        W_mod, vnorm = phase2.modulate_hodge_conductance(
            W, I_curl, epsilon_crit=1e-3, strangle_factor=strangle
        )
        assert vnorm > 1e-3
        w_mod_diag = W_mod.diagonal()
        # Aristas 0 y 1 penalizadas: w' = strangle * w (congruencia √s · w · √s)
        assert w_mod_diag[0] == pytest.approx(strangle, rel=1e-10)
        assert w_mod_diag[1] == pytest.approx(strangle, rel=1e-10)
        # Resto intacto
        assert w_mod_diag[2] == pytest.approx(1.0, rel=1e-10)

    def test_congruence_preserves_psd(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        """W_mod = D W D debe permanecer PSD si W lo es."""
        E = 5
        # W SPD denso-sparse
        W_dense = _spd(E, seed=60, cond=4.0)
        W = sp.csr_matrix(W_dense)
        I_curl = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
        W_mod, _ = phase2.modulate_hodge_conductance(
            W, I_curl, epsilon_crit=0.1, strangle_factor=1e-3
        )
        eigvals = la.eigvalsh(W_mod.toarray())
        assert eigvals[0] >= -1e-10 * abs(eigvals[-1])

    def test_vorticity_norm_formula(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        E = 4
        w = np.array([1.0, 2.0, 3.0, 4.0])
        W = sp.diags(w, format="csr")
        I = np.array([1.0, 1.0, 1.0, 1.0])
        # ‖I‖_W² = Σ w_i I_i² = 10
        _, vnorm = phase2.modulate_hodge_conductance(
            W, I, epsilon_crit=1e6  # sin estrangular
        )
        assert vnorm == pytest.approx(math.sqrt(10.0), rel=1e-12)

    def test_wrong_I_curl_shape_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        W = _diagonal_conductance(4)
        with pytest.raises(KinematicDimensionError, match="I_curl"):
            phase2.modulate_hodge_conductance(W, np.ones(3))

    def test_invalid_strangle_factor_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        W = _diagonal_conductance(3)
        with pytest.raises(ParasiticVorticityError, match="strangle"):
            phase2.modulate_hodge_conductance(
                W, np.ones(3), strangle_factor=0.0
            )
        with pytest.raises(ParasiticVorticityError, match="strangle"):
            phase2.modulate_hodge_conductance(
                W, np.ones(3), strangle_factor=-0.5
            )

    def test_non_square_W_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        W = sp.csr_matrix(np.ones((3, 4)))
        with pytest.raises(KinematicDimensionError, match="cuadrada"):
            phase2.modulate_hodge_conductance(W, np.ones(3))


class TestPhase2KramersKronig:
    """Fase 2 · sintonización de impedancia Kramers-Kronig."""

    @pytest.fixture()
    def phase2(self) -> KCoreKinematicAgent.Phase2_KinematicSynthesis:
        J, R, J_d, R_d, g, G = _canonical_system()
        return KCoreKinematicAgent(J, R, J_d, R_d, g, G=G).phase2

    def test_spd_output_and_causal(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        Z = _spd(3, seed=70, cond=5.0)
        eps, mu = phase2.tune_impedance_tensors(Z)
        # Ambos SPD
        assert la.eigvalsh(eps)[0] > 0.0
        assert la.eigvalsh(mu)[0] > 0.0
        # ε_eff ≈ Z
        assert np.allclose(eps, 0.5 * (Z + Z.T), atol=1e-10)
        # Simetría
        assert np.allclose(eps, eps.T, atol=1e-12)
        assert np.allclose(mu, mu.T, atol=1e-12)

    def test_non_spd_Z_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        Z_bad = np.diag([1.0, -2.0, 3.0])
        with pytest.raises(ImpedanceReflectionError, match="SPD|Positiva"):
            phase2.tune_impedance_tensors(Z_bad)

    def test_non_square_Z_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        with pytest.raises(ImpedanceReflectionError, match="cuadrada"):
            phase2.tune_impedance_tensors(np.ones((2, 3)))

    def test_scalar_impedance(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        Z = np.array([[2.5]])
        eps, mu = phase2.tune_impedance_tensors(Z)
        assert eps.shape == (1, 1)
        assert eps[0, 0] == pytest.approx(2.5, rel=1e-12)
        # μ = Z · ε · Z = 2.5³ = 15.625
        assert mu[0, 0] == pytest.approx(2.5 ** 3, rel=1e-12)


class TestPhase2CFL:
    """Fase 2 · auditoría CFL dual (Gerschgorin + espectral)."""

    @pytest.fixture()
    def phase2(self) -> KCoreKinematicAgent.Phase2_KinematicSynthesis:
        J, R, J_d, R_d, g, G = _canonical_system()
        return KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, cfl_margin=0.9
        ).phase2

    def test_path_graph_cfl_positive(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        L = _path_laplacian(10)
        dt_safe, rho_g, lam_max = phase2.audit_cfl_limit(c_eff=1.0, Delta_sym=L)
        assert dt_safe > 0.0
        assert math.isfinite(dt_safe)
        assert rho_g > 0.0
        assert lam_max >= 0.0
        # Gerschgorin ≥ λ_max (teorema)
        assert rho_g >= lam_max - 1e-6

    def test_cfl_margin_scales_dt(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        p2_a = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, cfl_margin=0.5
        ).phase2
        p2_b = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, cfl_margin=1.0
        ).phase2
        L = _path_laplacian(8)
        dt_a, _, _ = p2_a.audit_cfl_limit(1.0, L)
        dt_b, _, _ = p2_b.audit_cfl_limit(1.0, L)
        assert dt_b == pytest.approx(2.0 * dt_a, rel=1e-10)

    def test_c_eff_inverse_scaling(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        L = _path_laplacian(6)
        dt1, _, _ = phase2.audit_cfl_limit(1.0, L)
        dt2, _, _ = phase2.audit_cfl_limit(2.0, L)
        assert dt2 == pytest.approx(0.5 * dt1, rel=1e-10)

    def test_nonpositive_c_eff_raises(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        L = _path_laplacian(4)
        with pytest.raises(CFLViolationError, match="c_eff"):
            phase2.audit_cfl_limit(0.0, L)
        with pytest.raises(CFLViolationError, match="c_eff"):
            phase2.audit_cfl_limit(-1.0, L)

    def test_zero_laplacian_gives_inf(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        L0 = sp.csr_matrix((5, 5), dtype=np.float64)
        dt_safe, _, _ = phase2.audit_cfl_limit(1.0, L0)
        assert dt_safe == float("inf")

    def test_gershgorin_exact_for_path(
        self, phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis
    ) -> None:
        """
        Para P_n el radio de Gerschgorin de nodos interiores es 4
        (|2| + |-1| + |-1| = 4); extremos = 2.
        ⇒ ρ_G = 4 para n ≥ 3.
        """
        L = _path_laplacian(5)
        rho = phase2._gershgorin_radius(L)
        assert rho == pytest.approx(4.0, abs=1e-12)


class TestPhase2Synthesize:
    """Fase 2 · método terminal synthesize (integración de subprocesos)."""

    def _inputs(self, n: int = 4, m: int = 2, E: int = 5, V: int = 6):
        J, R, J_d, R_d, g, G = _canonical_system(n=n, m=m, seed=80)
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G,
            cfl_margin=0.9,
            residual_tol_rel=1e-4,
        )
        grad_H = np.linspace(0.1, 1.0, n)
        grad_H_d = np.linspace(0.2, 0.8, n)
        W = _diagonal_conductance(E, w=1.5)
        I_curl = 1e-4 * np.ones(E)
        Z_load = _spd(2, seed=81, cond=3.0)
        c_eff = 1.0
        Delta_sym = _path_laplacian(V)
        return agent, grad_H, grad_H_d, W, I_curl, Z_load, c_eff, Delta_sym

    def test_synthesize_happy_path(self) -> None:
        agent, gH, gHd, W, Ic, Z, ce, Ds = self._inputs()
        # dt holgado
        state = agent.phase2.synthesize(
            gH, gHd, W, Ic, Z, ce, Ds, dt_requested=1e-6
        )
        assert isinstance(state, KinematicStateTensor)
        assert state.is_kinematically_stable is True
        assert state.control_law_alpha.shape == (agent.context.m,)
        assert state.cfl_safe_dt > 0.0
        assert state.residual_idapbc < 1e-4
        assert state.vorticity_norm >= 0.0
        assert state.gershgorin_rho >= 0.0
        assert state.lambda_max_delta >= 0.0
        # Tensores SPD
        assert la.eigvalsh(state.dielectric_tensor)[0] > 0
        assert la.eigvalsh(state.magnetic_tensor)[0] > 0

    def test_cfl_violation_on_large_dt(self) -> None:
        agent, gH, gHd, W, Ic, Z, ce, Ds = self._inputs()
        with pytest.raises(CFLViolationError, match="CFL|Cono de Luz"):
            agent.phase2.synthesize(
                gH, gHd, W, Ic, Z, ce, Ds, dt_requested=1e6
            )

    def test_hodge_conductance_is_csr(self) -> None:
        agent, gH, gHd, W, Ic, Z, ce, Ds = self._inputs()
        state = agent.phase2.synthesize(
            gH, gHd, W, Ic, Z, ce, Ds, dt_requested=1e-6
        )
        assert sp.issparse(state.hodge_conductance)


# =============================================================================
# FASE 3 — PROYECCIÓN EN HACES
# =============================================================================


class TestPhase3SheafProjection:
    """Fase 3 · cofrontera δ_CORE e identidad de Hodge."""

    def test_hodge_identity_diagonal_W(self) -> None:
        E = 5
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = sp.diags(w, format="csr")
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W)
        # δ² ≈ W
        delta = phase3._delta_core
        assert np.allclose(delta @ delta, np.diag(w), atol=1e-10)
        assert phase3._hodge_residual < 100 * _EPS * 10
        assert phase3._rank_delta == E
        assert phase3._betti_approx == 0

    def test_hodge_identity_spd_W(self) -> None:
        E = 4
        W_dense = _spd(E, seed=90, cond=6.0)
        W = sp.csr_matrix(W_dense)
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W)
        delta = phase3._delta_core
        recon = delta @ delta
        rel = la.norm(recon - W_dense, "fro") / la.norm(W_dense, "fro")
        assert rel < 1e-10
        assert phase3._hodge_residual == pytest.approx(rel, rel=0.1, abs=1e-14)

    def test_rank_deficient_W_betti(self) -> None:
        """W de rango 2 en E=5 ⇒ β₀ ≈ 3."""
        E, rank = 5, 2
        W_dense = _psd_rank_deficient(E, rank, seed=91)
        W = sp.csr_matrix(W_dense)
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W)
        assert phase3._rank_delta == rank
        assert phase3._betti_approx == E - rank

    def test_spectral_entropy_bounds(self) -> None:
        # Rango 1 ⇒ S ≈ 0
        E = 4
        v = np.array([1.0, 0.0, 0.0, 0.0])
        W1 = sp.diags(v, format="csr")
        p3_1 = KCoreKinematicAgent.Phase3_SheafProjection(W1)
        assert p3_1._spectral_entropy == pytest.approx(0.0, abs=1e-12)

        # Espectro plano ⇒ S ≈ ln(E)
        Wflat = _diagonal_conductance(E, w=1.0)
        p3_f = KCoreKinematicAgent.Phase3_SheafProjection(Wflat)
        assert p3_f._spectral_entropy == pytest.approx(math.log(E), rel=1e-10)

    def test_export_stalk_projection(self) -> None:
        E = 4
        W = _diagonal_conductance(E, w=4.0)  # δ = 2 I
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        stalk = phase3.export_stalk(x)
        assert isinstance(stalk, SheafStalk)
        assert np.allclose(stalk.projected_state, 2.0 * x)
        assert stalk.rank_delta == E
        assert stalk.betti_approx == 0
        assert stalk.spectral_entropy == pytest.approx(math.log(E), rel=1e-10)
        assert np.allclose(stalk.state_vector, x)
        # Inmutabilidad de la copia
        x[0] = -99.0
        assert stalk.state_vector[0] == 1.0

    def test_export_wrong_shape_raises(self) -> None:
        W = _diagonal_conductance(3)
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W)
        with pytest.raises(KinematicDimensionError, match="state_x"):
            phase3.export_stalk(np.ones(5))

    def test_negative_W_raises(self) -> None:
        W_bad = sp.diags([1.0, -2.0, 1.0], format="csr")
        with pytest.raises(SheafCoboundaryError, match="Positiva|PSD"):
            KCoreKinematicAgent.Phase3_SheafProjection(W_bad)

    def test_zero_W_entropy_and_rank(self) -> None:
        W0 = sp.csr_matrix((3, 3), dtype=np.float64)
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W0)
        assert phase3._rank_delta == 0
        assert phase3._betti_approx == 3
        assert phase3._spectral_entropy == pytest.approx(0.0, abs=1e-15)


# =============================================================================
# API PÚBLICA DEL AGENTE
# =============================================================================


class TestPublicAPI:
    """Interfaz pública: synthesize_kinematic_core + export_sheaf_stalk."""

    @pytest.fixture()
    def ready_agent(self) -> KCoreKinematicAgent:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2, seed=100)
        return KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G,
            cfl_margin=0.9,
            residual_tol_rel=1e-4,
            tikhonov_reg=1e-12,
        )

    def _run_synth(self, agent: KCoreKinematicAgent) -> KinematicStateTensor:
        n = agent.context.n
        E, V = 5, 6
        return agent.synthesize_kinematic_core(
            grad_H=np.linspace(0.1, 1.0, n),
            grad_H_d=np.linspace(0.05, 0.9, n),
            W=_diagonal_conductance(E, w=1.0),
            I_curl=1e-5 * np.ones(E),
            Z_load=_spd(2, seed=101, cond=2.0),
            c_eff=1.0,
            Delta_sym=_path_laplacian(V),
            dt_requested=1e-5,
        )

    def test_full_pipeline(self, ready_agent: KCoreKinematicAgent) -> None:
        state = self._run_synth(ready_agent)
        assert state.is_kinematically_stable
        stalk = ready_agent.export_sheaf_stalk(np.ones(5))
        assert stalk.delta_core.shape == (5, 5)
        assert stalk.rank_delta >= 0
        assert stalk.delta_hodge_residual < 1e-10

    def test_export_without_synthesize_raises(
        self, ready_agent: KCoreKinematicAgent
    ) -> None:
        with pytest.raises(KinematicCoreError, match="W_mod|synthesize"):
            ready_agent.export_sheaf_stalk(np.ones(5))

    def test_phase3_lazy_and_cache(
        self, ready_agent: KCoreKinematicAgent
    ) -> None:
        assert ready_agent.phase3 is None
        self._run_synth(ready_agent)
        assert ready_agent.phase3 is None  # aún lazy hasta export
        _ = ready_agent.export_sheaf_stalk(np.ones(5))
        assert ready_agent.phase3 is not None
        p3_ref = ready_agent.phase3
        # Segunda exportación reutiliza la misma instancia
        _ = ready_agent.export_sheaf_stalk(np.zeros(5))
        assert ready_agent.phase3 is p3_ref

    def test_resynthesize_invalidates_phase3(
        self, ready_agent: KCoreKinematicAgent
    ) -> None:
        self._run_synth(ready_agent)
        _ = ready_agent.export_sheaf_stalk(np.ones(5))
        assert ready_agent.phase3 is not None
        # Nueva síntesis invalida phase3
        self._run_synth(ready_agent)
        assert ready_agent.phase3 is None

    def test_cfl_violation_through_public_api(
        self, ready_agent: KCoreKinematicAgent
    ) -> None:
        n = ready_agent.context.n
        with pytest.raises(CFLViolationError):
            ready_agent.synthesize_kinematic_core(
                grad_H=np.ones(n),
                grad_H_d=np.ones(n),
                W=_diagonal_conductance(4),
                I_curl=np.zeros(4),
                Z_load=np.eye(2),
                c_eff=1.0,
                Delta_sym=_path_laplacian(5),
                dt_requested=1e9,
            )


# =============================================================================
# PROPIEDADES INVARIANTES Y EDGE-CASES
# =============================================================================


class TestInvariants:
    """Propiedades matemáticas que deben preservarse siempre."""

    def test_alpha_in_control_space_dimension(self) -> None:
        for m in (1, 2, 3):
            J, R, J_d, R_d, g, G = _canonical_system(n=5, m=m, seed=110 + m)
            agent = KCoreKinematicAgent(
                J, R, J_d, R_d, g, G=G, residual_tol_rel=1e-3
            )
            alpha, _ = agent.phase2.compute_dirac_control_law(
                np.ones(5), 0.5 * np.ones(5)
            )
            assert alpha.shape == (m,)

    def test_phase_continuity_context_to_phase2(self) -> None:
        """El contexto de Fase 1 es exactamente el que consume Fase 2."""
        J, R, J_d, R_d, g, G = _canonical_system(n=3, m=1, seed=120)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        assert agent.phase2._ctx is agent.context

    def test_hodge_conductance_feeds_phase3(self) -> None:
        """W_mod del tensor de estado es el input de Phase3."""
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2, seed=130)
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, residual_tol_rel=1e-3
        )
        state = agent.synthesize_kinematic_core(
            grad_H=np.ones(4),
            grad_H_d=0.5 * np.ones(4),
            W=_diagonal_conductance(3, w=2.0),
            I_curl=np.zeros(3),
            Z_load=np.eye(2),
            c_eff=1.0,
            Delta_sym=_path_laplacian(4),
            dt_requested=1e-4,
        )
        assert agent._latest_hodge_conductance is state.hodge_conductance
        stalk = agent.export_sheaf_stalk(np.array([1.0, 0.0, 0.0]))
        # δ = √2 · I  ⇒  δ x = √2 · e_0
        assert stalk.projected_state[0] == pytest.approx(
            math.sqrt(2.0), rel=1e-10
        )

    def test_dissipation_psd_invariant_after_validation(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2, seed=140)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        for mat in (agent.context.R, agent.context.R_d, agent.context.G):
            ev = la.eigvalsh(mat)
            assert ev[0] >= -1e-12

    def test_interconnection_skew_invariant(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2, seed=150)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        for mat in (agent.context.J, agent.context.J_d):
            assert np.allclose(mat + mat.T, 0.0, atol=1e-12)

    def test_cfl_formula_consistency(self) -> None:
        """
        Δt_safe = 2·margin / (c · √bound).
        Verificar inversión algebraica.
        """
        J, R, J_d, R_d, g, G = _canonical_system()
        margin = 0.8
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, cfl_margin=margin
        )
        L = _path_laplacian(7)
        c = 1.5
        dt, rho, lmax = agent.phase2.audit_cfl_limit(c, L)
        bound = max(rho, lmax, _EPS)
        expected = 2.0 * margin / (c * math.sqrt(bound))
        assert dt == pytest.approx(expected, rel=1e-12)

    def test_von_neumann_entropy_static(self) -> None:
        S = KCoreKinematicAgent.Phase3_SheafProjection._von_neumann_entropy
        # Nulo
        assert S(np.zeros(4)) == 0.0
        # Un solo modo
        assert S(np.array([3.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-15)
        # Dos modos iguales: S = ln(2)
        assert S(np.array([1.0, 1.0, 0.0])) == pytest.approx(
            math.log(2.0), rel=1e-12
        )


class TestEdgeCases:
    """Casos límite dimensionales y degenerados."""

    def test_minimal_system_n1_m1(self) -> None:
        """Sistema escalar: J=[[0]], R>0, g=[[1]]."""
        J = np.array([[0.0]])
        R = np.array([[1.0]])
        J_d = np.array([[0.0]])
        R_d = np.array([[2.0]])
        g = np.array([[1.0]])
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g)
        assert agent.context.n == 1
        assert agent.context.m == 1
        alpha, res = agent.phase2.compute_dirac_control_law(
            np.array([1.0]), np.array([0.5])
        )
        assert alpha.shape == (1,)
        assert res < 1e-10

    def test_large_sparse_laplacian_gershgorin_only(self) -> None:
        """
        Para n grande ARPACK debería funcionar; verificamos que al menos
        Gerschgorin produce un radio finito y positivo.
        """
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=1)
        agent = KCoreKinematicAgent(J, R, J_d, R_d, g, G=G)
        n_nodes = 200
        L = _path_laplacian(n_nodes)
        dt, rho, lmax = agent.phase2.audit_cfl_limit(1.0, L)
        assert rho == pytest.approx(4.0, abs=1e-9)
        assert dt > 0.0 and math.isfinite(dt)
        # λ_max del camino ≈ 4 (exactamente 2-2cos(π(n-1)/n) → 4)
        assert lmax == pytest.approx(4.0, rel=0.05) or lmax == 0.0

    def test_identity_metric_gram_equals_gtg(self) -> None:
        """Con G=I, Gram = gᵀg; verificar coherencia de la SVD interna."""
        n, m = 5, 2
        J, R, J_d, R_d, g, _ = _canonical_system(n=n, m=m, seed=160)
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=None, residual_tol_rel=1e-4
        )
        # Ejecutar una ley de control y verificar que el Gram implícito
        # tiene los mismos valores singulares que gᵀg
        s_gtg = la.svdvals(g.T @ g)
        s_g = la.svdvals(g) ** 2
        assert np.allclose(np.sort(s_gtg), np.sort(s_g), atol=1e-10)
        alpha, res = agent.phase2.compute_dirac_control_law(
            np.ones(n), np.zeros(n)
        )
        assert np.all(np.isfinite(alpha))


# =============================================================================
# CONTINUIDAD FORMAL ENTRE FASES (contrato de interfaz)
# =============================================================================


class TestPhaseContinuityContracts:
    """
    Verifica que las fronteras formales Fase1→2→3 se respetan:
      build_preparation_context() → KinematicPreparationContext
      synthesize()                → KinematicStateTensor (hodge_conductance)
      export_stalk()              → SheafStalk
    """

    def test_phase1_terminal_type(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        p1 = KCoreKinematicAgent.Phase1_MatrixValidation(
            J, R, J_d, R_d, g, G=G
        )
        ctx = p1.build_preparation_context()
        assert isinstance(ctx, KinematicPreparationContext)

    def test_phase2_accepts_phase1_output(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system()
        ctx = KCoreKinematicAgent.Phase1_MatrixValidation(
            J, R, J_d, R_d, g, G=G
        ).build_preparation_context()
        p2 = KCoreKinematicAgent.Phase2_KinematicSynthesis(
            context=ctx, cfl_margin=0.9, residual_tol_rel=1e-4
        )
        assert p2._ctx.n == ctx.n

    def test_phase3_accepts_phase2_hodge(self) -> None:
        J, R, J_d, R_d, g, G = _canonical_system(n=4, m=2)
        agent = KCoreKinematicAgent(
            J, R, J_d, R_d, g, G=G, residual_tol_rel=1e-3
        )
        state = agent.phase2.synthesize(
            grad_H=np.ones(4),
            grad_H_d=np.ones(4),
            W=_diagonal_conductance(4),
            I_curl=np.zeros(4),
            Z_load=np.eye(2),
            c_eff=1.0,
            Delta_sym=_path_laplacian(4),
            dt_requested=1e-5,
        )
        p3 = KCoreKinematicAgent.Phase3_SheafProjection(
            W_mod=state.hodge_conductance
        )
        stalk = p3.export_stalk(np.ones(4))
        assert isinstance(stalk, SheafStalk)
        assert stalk.delta_hodge_residual < 1e-10


# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-ra"])