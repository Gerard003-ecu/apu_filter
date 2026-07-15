# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite  : test_floquet_agent.py                                               |
| Objetivo: Validación rigurosa del endofuntor Floquet v3.0.0                  |
|           (proyector covariante · monodromía · canal CPTP · topos MIC)       |
| Versión: 1.0.0-Rigorous-Nested-Oracle                                        |
+==============================================================================+

Arquitectura (espejo de las 3 fases anidadas)
---------------------------------------------
  0. Utilidades / fixtures / oráculos independientes
  1. TestExceptionHierarchy
  2. TestPhase1*  — síntesis covariante, idempotencia, obstrucción trivial
  3. TestPhase2*  — monodromía M=2P−P², espectro ⊆{0,1}, estabilidad
  4. TestPhase3*  — Kraus C=I bilateral, Von Neumann, pureza, positrón
  5. TestFunctorialComposition — execute ∘ audit ∘ synthesize
  6. TestFloquetMonodromyAgentIntegration — contrato público
  7. TestNumericalStressAndInvariants — ensemble, n variable, G anisotrópica

Ejecución
---------
  pytest test_floquet_agent.py -v --tb=short
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import del SUT con fallback de path local
# ---------------------------------------------------------------------------
try:
    from app.omega.floquet_agent import (
        DimensionalAudit,
        DimensionalMismatchError,
        FloquetInstabilityError,
        FloquetMonodromyAgent,
        FloquetMonodromyState,
        FloquetParameterError,
        KrausCompletenessError,
        KrausTraceViolationError,
        ProjectorDefectError,
        ProjectorSynthesisResult,
        QuantumChannelEvolution,
        TopologicalInvariantError,
        _DEFAULT_KRAUS_TOL,
        _DEFAULT_PROJECTOR_TOL,
        _DEFAULT_STABILITY_TOL,
        _MACHINE_EPS,
    )
except ImportError:
    from floquet_agent import (  # type: ignore
        DimensionalAudit,
        DimensionalMismatchError,
        FloquetInstabilityError,
        FloquetMonodromyAgent,
        FloquetMonodromyState,
        FloquetParameterError,
        KrausCompletenessError,
        KrausTraceViolationError,
        ProjectorDefectError,
        ProjectorSynthesisResult,
        QuantumChannelEvolution,
        TopologicalInvariantError,
        _DEFAULT_KRAUS_TOL,
        _DEFAULT_PROJECTOR_TOL,
        _DEFAULT_STABILITY_TOL,
        _MACHINE_EPS,
    )

# TopologicalInvariantError puede venir del stub o del ecosistema
try:
    from app.core.mic_algebra import TopologicalInvariantError as _TIE_ROOT
except ImportError:
    _TIE_ROOT = TopologicalInvariantError  # type: ignore[misc]


# =============================================================================
# 0. UTILIDADES Y ORÁCULOS INDEPENDIENTES
# =============================================================================

EPS = float(np.finfo(np.float64).eps)
_RNG = np.random.default_rng(20260328)


def make_spd(n: int, cond: float = 5.0, seed: int = 0) -> NDArray[np.float64]:
    """SPD con κ ≈ cond."""
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.standard_normal((n, n)))
    log_l = np.linspace(0.0, math.log(max(cond, 1.0 + 1e-15)), n)
    A = (Q * np.exp(log_l)) @ Q.T
    return 0.5 * (A + A.T)


def make_projector_euclid(n: int, rank: int, seed: int = 0) -> NDArray[np.float64]:
    """
    Proyector euclídeo ortogonal exacto de rango r:
        P = Q[:, :r] Q[:, :r]ᵀ,  P²=P, Pᵀ=P, spec ⊆ {0,1}.
    """
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.standard_normal((n, n)))
    r = max(0, min(rank, n))
    if r == 0:
        return np.zeros((n, n), dtype=np.float64)
    P = Q[:, :r] @ Q[:, :r].T
    return 0.5 * (P + P.T)


def fro_rel(A: np.ndarray, B: np.ndarray) -> float:
    return float(la.norm(A - B, "fro")) / max(float(la.norm(B, "fro")), 1.0)


def von_neumann_oracle(rho: np.ndarray) -> float:
    """Oráculo independiente S(ρ) = −Σ λ log λ (traza-1)."""
    H = 0.5 * (rho + rho.T)
    tr = float(np.trace(H))
    if tr <= 1e-30:
        return 0.0
    Hn = H / tr
    ev = np.clip(np.linalg.eigvalsh(Hn), 0.0, None)
    pos = ev[ev > 1e-15]
    if pos.size == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


def purity_oracle(rho: np.ndarray) -> float:
    H = 0.5 * (rho + rho.T)
    tr = float(np.trace(H))
    if tr <= 1e-30:
        return 0.0
    Hn = H / tr
    return float(np.trace(Hn @ Hn))


def kraus_completeness_oracle(
    E0: np.ndarray, E1: np.ndarray
) -> Tuple[float, float]:
    """(‖C−I‖_F, max|λ(C−I)|)."""
    d = E0.shape[0]
    C = E0.T @ E0 + E1.T @ E1
    diff = 0.5 * ((C - np.eye(d)) + (C - np.eye(d)).T)
    fro = float(la.norm(diff, "fro"))
    eig_res = float(np.max(np.abs(np.linalg.eigvalsh(diff))))
    return fro, eig_res


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(params=[2, 3, 4, 6])
def dim(request) -> int:
    return int(request.param)


@pytest.fixture
def G_euclid(dim) -> NDArray[np.float64]:
    return np.eye(dim, dtype=np.float64)


@pytest.fixture
def G_aniso(dim) -> NDArray[np.float64]:
    return make_spd(dim, cond=8.0, seed=dim + 17)


@pytest.fixture
def agent_euclid(dim, G_euclid) -> FloquetMonodromyAgent:
    return FloquetMonodromyAgent(
        metric_tensor=G_euclid,
        stability_tolerance=1e-9,
        kraus_tolerance=1e-9,
        projector_tolerance=1e-8,
    )


@pytest.fixture
def agent_aniso(dim, G_aniso) -> FloquetMonodromyAgent:
    return FloquetMonodromyAgent(
        metric_tensor=G_aniso,
        stability_tolerance=1e-9,
        kraus_tolerance=1e-9,
        projector_tolerance=1e-8,
    )


@pytest.fixture
def random_psi(dim) -> NDArray[np.float64]:
    v = _RNG.standard_normal(dim)
    nrm = float(la.norm(v))
    return v / max(nrm, 1e-15)


@pytest.fixture
def random_grad(dim) -> NDArray[np.float64]:
    return _RNG.standard_normal(dim)


# =============================================================================
# 1. JERARQUÍA DE EXCEPCIONES
# =============================================================================


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            FloquetInstabilityError,
            KrausTraceViolationError,
            KrausCompletenessError,
            DimensionalMismatchError,
            ProjectorDefectError,
            FloquetParameterError,
        ],
    )
    def test_all_derive_from_topological_root(self, exc_cls):
        assert issubclass(exc_cls, _TIE_ROOT) or issubclass(
            exc_cls, TopologicalInvariantError
        )

    def test_catch_all_on_nonsquare_G(self):
        with pytest.raises((TopologicalInvariantError, DimensionalMismatchError)):
            FloquetMonodromyAgent(metric_tensor=np.ones((3, 2)))


# =============================================================================
# 2. FASE 1 — SÍNTESIS COVARIANTE DEL PROYECTOR
# =============================================================================


class TestPhase1Construction:
    def test_rejects_nonsquare_G(self):
        with pytest.raises(DimensionalMismatchError):
            FloquetMonodromyAgent(metric_tensor=np.ones((2, 3)))

    def test_rejects_asymmetric_G(self):
        G = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(TopologicalInvariantError):
            FloquetMonodromyAgent(metric_tensor=G)

    def test_rejects_indefinite_G(self):
        G = np.diag([1.0, -1.0])
        with pytest.raises(TopologicalInvariantError):
            FloquetMonodromyAgent(metric_tensor=G)

    def test_rejects_nonpositive_tolerances(self, G_euclid):
        with pytest.raises(FloquetParameterError):
            FloquetMonodromyAgent(G_euclid, stability_tolerance=0.0)
        with pytest.raises(FloquetParameterError):
            FloquetMonodromyAgent(G_euclid, kraus_tolerance=-1.0)
        with pytest.raises(FloquetParameterError):
            FloquetMonodromyAgent(G_euclid, projector_tolerance=0.0)

    def test_dimension_property(self, agent_euclid, dim):
        assert agent_euclid.dimension == dim
        assert agent_euclid.metric_tensor.shape == (dim, dim)


class TestPhase1TrivialObstruction:
    """Axioma §1: ‖n‖_G ≈ 0 ⇒ P = I (estado válido, no excepción)."""

    def test_zero_gradient_yields_identity(self, agent_euclid, dim):
        synth = agent_euclid.phase1.synthesize_projector(np.zeros(dim))
        assert isinstance(synth, ProjectorSynthesisResult)
        assert synth.is_trivial_obstruction is True
        assert synth.reflector is None
        np.testing.assert_allclose(synth.P_hat, np.eye(dim), atol=1e-15)
        assert synth.idempotence_residual < 1e-14
        assert synth.symmetry_residual < 1e-14
        assert synth.normal_G_norm < 1e-15

    def test_near_zero_gradient_trivial(self, agent_euclid, dim):
        grad = 1e-20 * np.ones(dim)
        synth = agent_euclid.phase1.synthesize_projector(grad)
        # Puede ser trivial o no según ‖G grad‖_G; si es trivial, P=I
        if synth.is_trivial_obstruction:
            np.testing.assert_allclose(synth.P_hat, np.eye(dim), atol=1e-14)


class TestPhase1NontrivialProjector:
    def test_wrong_grad_shape_raises(self, agent_euclid, dim):
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.phase1.synthesize_projector(np.ones(dim + 1))

    def test_projector_idempotent(self, agent_euclid, dim, random_grad):
        synth = agent_euclid.phase1.synthesize_projector(random_grad)
        if synth.is_trivial_obstruction:
            pytest.skip("obstrucción trivial en esta semilla")
        P = synth.P_hat
        idem = float(la.norm(P @ P - P, "fro"))
        assert idem == pytest.approx(synth.idempotence_residual, rel=1e-9, abs=1e-15)
        assert idem < _DEFAULT_PROJECTOR_TOL * max(dim, 1)

    def test_projector_nearly_symmetric(self, agent_euclid, dim, random_grad):
        synth = agent_euclid.phase1.synthesize_projector(random_grad)
        assert synth.symmetry_residual < 1e-6 * max(dim, 1)

    def test_normal_G_norm_positive(self, agent_euclid, dim):
        grad = np.ones(dim)
        synth = agent_euclid.phase1.synthesize_projector(grad)
        if not synth.is_trivial_obstruction:
            assert synth.normal_G_norm > 0.0

    def test_pullback_uses_metric(self, agent_aniso, dim, G_aniso):
        """
        n = G ∇H; ‖n‖_G = sqrt(nᵀ G n).
        Oráculo independiente del normal_G_norm reportado.
        """
        grad = _RNG.standard_normal(dim)
        synth = agent_aniso.phase1.synthesize_projector(grad)
        n = G_aniso @ grad
        expected_norm = float(np.sqrt(max(n @ (G_aniso @ n), 0.0)))
        if expected_norm < 1e-15:
            assert synth.is_trivial_obstruction
        else:
            assert synth.normal_G_norm == pytest.approx(expected_norm, rel=1e-10)

    def test_result_is_frozen(self, agent_euclid, dim):
        synth = agent_euclid.phase1.synthesize_projector(np.ones(dim))
        with pytest.raises(Exception):
            synth.dim = 99  # type: ignore[misc]


class TestPhase1ProjectorDefect:
    """Inyección de proyector defectuoso en Fase 2 (defensa en profundidad)."""

    def test_phase2_rejects_non_idempotent_P(self, agent_euclid, dim):
        # Fabricar un "synthesis" con P no idempotente
        bad_P = np.ones((dim, dim)) / dim  # no es proyector si dim>1 de forma genérica
        # Ajustar: matriz genérica no idempotente
        bad_P = _RNG.standard_normal((dim, dim))
        bad_P = 0.5 * (bad_P + bad_P.T)
        # Forzar no-idempotencia
        if float(la.norm(bad_P @ bad_P - bad_P, "fro")) < 1e-6:
            bad_P = bad_P + 0.3 * np.eye(dim)

        fake = ProjectorSynthesisResult(
            P_hat=bad_P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(bad_P @ bad_P - bad_P, "fro")),
            symmetry_residual=float(la.norm(bad_P - bad_P.T, "fro")),
            dim=dim,
        )
        with pytest.raises(ProjectorDefectError):
            agent_euclid.phase2.audit_monodromy(fake)


# =============================================================================
# 3. FASE 2 — MONODROMÍA DE FLOQUET
# =============================================================================


class TestPhase2MonodromyAlgebra:
    """
    Identidad: si P²=P entonces M=2P−P²=P y spec(M)⊆{0,1}.
    """

    def test_monodromy_equals_P_for_idempotent(self, agent_euclid, dim):
        P = make_projector_euclid(dim, rank=max(dim // 2, 1), seed=7)
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(P @ P - P, "fro")),
            symmetry_residual=float(la.norm(P - P.T, "fro")),
            dim=dim,
        )
        mono = agent_euclid.phase2.audit_monodromy(synth)
        assert fro_rel(mono.monodromy_matrix, P) < 1e-10
        assert mono.spectral_radius <= 1.0 + 1e-9
        assert mono.is_asymptotically_stable is True
        # Multiplicadores ≈ 0 o 1
        for mu in np.real_if_close(mono.multipliers):
            assert min(abs(mu - 0.0), abs(mu - 1.0)) < 1e-8

    def test_identity_projector_spectral_radius_one(self, agent_euclid, dim):
        P = np.eye(dim)
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=0.0,
            is_trivial_obstruction=True,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            dim=dim,
        )
        mono = agent_euclid.phase2.audit_monodromy(synth)
        assert mono.spectral_radius == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(mono.multipliers, np.ones(dim), atol=1e-12)

    def test_zero_projector_spectral_radius_zero(self, agent_euclid, dim):
        P = np.zeros((dim, dim))
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            dim=dim,
        )
        mono = agent_euclid.phase2.audit_monodromy(synth)
        assert mono.spectral_radius == pytest.approx(0.0, abs=1e-12)

    def test_rank_projector_multiplicity(self, agent_euclid, dim):
        """#{μ≈1} = rank(P), #{μ≈0} = n−rank(P)."""
        r = max(1, dim // 2)
        P = make_projector_euclid(dim, rank=r, seed=11)
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(P @ P - P, "fro")),
            symmetry_residual=0.0,
            dim=dim,
        )
        mono = agent_euclid.phase2.audit_monodromy(synth)
        mult = np.real_if_close(mono.multipliers)
        n_one = int(np.sum(np.abs(mult - 1.0) < 1e-7))
        n_zero = int(np.sum(np.abs(mult - 0.0) < 1e-7))
        assert n_one == r
        assert n_zero == dim - r

    def test_pipeline_phase1_to_phase2_stable(
        self, agent_euclid, dim, random_grad
    ):
        synth = agent_euclid.phase1.synthesize_projector(random_grad)
        mono = agent_euclid.phase2.audit_monodromy(synth)
        assert mono.is_asymptotically_stable
        assert mono.spectral_radius <= 1.0 + mono.tolerance_used
        assert mono.projector_idempotence_residual < 1e-6 * max(dim, 1)

    def test_asymmetric_P_marks_complex_manifold(self, agent_euclid, dim):
        if dim < 2:
            pytest.skip("requiere d≥2")
        # Proyector no simétrico artificial con P²≈P (proyección oblicua)
        # Usamos un proyector oblicuo: P = u vᵀ / (vᵀ u) con u≠v
        rng = np.random.default_rng(99)
        u = rng.standard_normal(dim)
        v = rng.standard_normal(dim)
        denom = float(v @ u)
        if abs(denom) < 1e-8:
            v = u + 0.5 * np.ones(dim)
            denom = float(v @ u)
        P = np.outer(u, v) / denom
        # Verificar idempotencia de proyector oblicuo: P²=P
        assert float(la.norm(P @ P - P, "fro")) < 1e-8

        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(P @ P - P, "fro")),
            symmetry_residual=float(la.norm(P - P.T, "fro")),
            dim=dim,
        )
        mono = agent_euclid.phase2.audit_monodromy(synth)
        # Si M no es simétrica, is_complex_manifold=True
        if float(la.norm(mono.monodromy_matrix - mono.monodromy_matrix.T, "fro")) > 1e-10:
            assert mono.is_complex_manifold is True
        assert mono.spectral_radius <= 1.0 + 1e-6


class TestPhase2Instability:
    def test_unstable_monodromy_raises(self, agent_euclid, dim):
        """
        P con radio espectral de M > 1 (no proyector): debe fallar
        por ProjectorDefect o FloquetInstability.
        """
        # Matriz con autovalores grandes
        bad = 3.0 * np.eye(dim)
        fake = ProjectorSynthesisResult(
            P_hat=bad,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(bad @ bad - bad, "fro")),
            symmetry_residual=0.0,
            dim=dim,
        )
        with pytest.raises((ProjectorDefectError, FloquetInstabilityError)):
            agent_euclid.phase2.audit_monodromy(fake)


# =============================================================================
# 4. FASE 3 — CANAL CUÁNTICO CPTP
# =============================================================================


class TestPhase3KrausCompleteness:
    """Axioma §3: C = E0ᵀE0 + E1ᵀE1 = I (bilateral)."""

    def test_completeness_for_orthogonal_projector(self, agent_euclid, dim):
        P = make_projector_euclid(dim, rank=max(1, dim // 2), seed=21)
        E0, E1 = P, np.eye(dim) - P
        fro, eig_res = kraus_completeness_oracle(E0, E1)
        assert fro < 1e-10
        assert eig_res < 1e-10

        # Vía el método interno
        f2, e2 = agent_euclid.phase3._verify_kraus_completeness(E0, E1)
        assert f2 == pytest.approx(fro, abs=1e-12)
        assert e2 == pytest.approx(eig_res, abs=1e-12)

    def test_completeness_identity_and_zero(self, agent_euclid, dim):
        for P in (np.eye(dim), np.zeros((dim, dim))):
            E0, E1 = P, np.eye(dim) - P
            fro, eig_res = agent_euclid.phase3._verify_kraus_completeness(E0, E1)
            assert fro < 1e-12
            assert eig_res < 1e-12

    def test_algebraic_identity_for_symmetric_idempotent(self, dim):
        """
        Demostración numérica:
          PᵀP+(I−P)ᵀ(I−P) = P²+(I−P)² = P+I−2P+P = I
        si Pᵀ=P y P²=P.
        """
        P = make_projector_euclid(dim, rank=max(1, dim - 1), seed=22)
        C = P.T @ P + (np.eye(dim) - P).T @ (np.eye(dim) - P)
        assert fro_rel(C, np.eye(dim)) < 1e-12

    def test_defective_operators_raise_completeness_error(self, agent_euclid, dim):
        """E0, E1 arbitrarios que no completan la identidad."""
        E0 = 2.0 * np.eye(dim)  # C = 4I + ... ≠ I
        E1 = np.eye(dim)
        with pytest.raises((KrausCompletenessError, KrausTraceViolationError)):
            agent_euclid.phase3._verify_kraus_completeness(E0, E1)

    def test_unilateral_psd_would_pass_but_bilateral_fails(self, agent_euclid, dim):
        """
        Guardia anti-regresión del bug v2:
        C = 1.5 I ⇒ C−I = 0.5 I ⪰ 0 (pasaría el criterio unilateral)
        pero max|λ(C−I)| = 0.5 ⇏ 0 (bilateral falla).
        """
        # Construir E0, E1 tales que C = 1.5 I
        # E0 = sqrt(1.5) I, E1 = 0 ⇒ C = 1.5 I
        E0 = math.sqrt(1.5) * np.eye(dim)
        E1 = np.zeros((dim, dim))
        with pytest.raises((KrausCompletenessError, KrausTraceViolationError)):
            agent_euclid.phase3._verify_kraus_completeness(E0, E1)


class TestPhase3VonNeumannEntropy:
    def test_pure_state_entropy_zero(self, agent_euclid, dim, random_psi):
        rho = agent_euclid.phase3._density_from_pure(random_psi)
        S = agent_euclid.phase3._von_neumann_entropy_rho(rho)
        assert S == pytest.approx(0.0, abs=1e-10)
        assert S == pytest.approx(von_neumann_oracle(rho), abs=1e-10)

    def test_maximally_mixed_entropy(self, agent_euclid, dim):
        """S(I/d) = log(d)."""
        rho = np.eye(dim) / dim
        S = agent_euclid.phase3._von_neumann_entropy_rho(rho)
        assert S == pytest.approx(math.log(dim), rel=1e-10)

    def test_purity_pure_state_is_one(self, agent_euclid, dim, random_psi):
        rho = agent_euclid.phase3._density_from_pure(random_psi)
        assert agent_euclid.phase3._purity(rho) == pytest.approx(1.0, abs=1e-10)

    def test_purity_mixed_less_than_one(self, agent_euclid, dim):
        if dim < 2:
            pytest.skip("requiere d≥2")
        rho = np.eye(dim) / dim
        assert agent_euclid.phase3._purity(rho) == pytest.approx(1.0 / dim, rel=1e-10)

    def test_channel_preserves_trace(self, agent_euclid, dim, random_psi):
        P = make_projector_euclid(dim, rank=max(1, dim // 2), seed=30)
        E0, E1 = P, np.eye(dim) - P
        rho = agent_euclid.phase3._density_from_pure(random_psi)
        rho_post = agent_euclid.phase3._apply_kraus_channel(rho, E0, E1)
        assert float(np.trace(rho_post)) == pytest.approx(
            float(np.trace(rho)), rel=1e-10, abs=1e-12
        )

    def test_channel_preserves_hermiticity_and_psd(
        self, agent_euclid, dim, random_psi
    ):
        P = make_projector_euclid(dim, rank=max(1, dim // 2), seed=31)
        E0, E1 = P, np.eye(dim) - P
        rho = agent_euclid.phase3._density_from_pure(random_psi)
        rho_post = agent_euclid.phase3._apply_kraus_channel(rho, E0, E1)
        H = 0.5 * (rho_post + rho_post.T)
        assert float(la.norm(rho_post - rho_post.T, "fro")) < 1e-12
        assert np.all(np.linalg.eigvalsh(H) >= -1e-12)


class TestPhase3ExecuteChannel:
    def _synth_mono(self, agent, grad):
        synth = agent.phase1.synthesize_projector(grad)
        mono = agent.phase2.audit_monodromy(synth)
        return synth, mono

    def test_execute_returns_evolution_dto(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        evo = agent_euclid.phase3.execute_quantum_channel(
            random_psi, random_grad, synth, mono
        )
        assert isinstance(evo, QuantumChannelEvolution)
        assert evo.coherent_state.shape == (dim,)
        assert evo.rho_post.shape == (dim, dim)
        assert evo.dimensional_audit.is_coherent is True
        assert evo.monodromy_state is mono
        assert evo.kraus_residual_fro < 1e-8
        assert evo.kraus_residual_eig < 1e-8

    def test_coherent_state_equals_P_psi(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        evo = agent_euclid.phase3.execute_quantum_channel(
            random_psi, random_grad, synth, mono
        )
        expected = synth.P_hat @ random_psi
        np.testing.assert_allclose(evo.coherent_state, expected, rtol=1e-10)

    def test_von_neumann_fields_match_oracle(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        evo = agent_euclid.phase3.execute_quantum_channel(
            random_psi, random_grad, synth, mono
        )
        rho_pre = np.outer(random_psi, random_psi)
        assert evo.von_neumann_pre == pytest.approx(
            von_neumann_oracle(rho_pre), abs=1e-9
        )
        assert evo.von_neumann_post == pytest.approx(
            von_neumann_oracle(evo.rho_post), abs=1e-9
        )
        assert evo.delta_entropy == pytest.approx(
            evo.von_neumann_post - evo.von_neumann_pre, abs=1e-12
        )
        assert evo.purity_post == pytest.approx(
            purity_oracle(evo.rho_post), abs=1e-9
        )

    def test_pure_input_has_zero_pre_entropy(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        evo = agent_euclid.phase3.execute_quantum_channel(
            random_psi, random_grad, synth, mono
        )
        # Estado puro ⇒ S_pre = 0
        assert evo.von_neumann_pre == pytest.approx(0.0, abs=1e-10)

    def test_dimensional_mismatch_blocks(self, agent_euclid, dim, random_grad):
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.phase3.execute_quantum_channel(
                np.ones(dim + 2), random_grad, synth, mono
            )

    def test_antimatter_on_generic_state(
        self, agent_euclid, dim, random_grad
    ):
        """
        Si ψ no está en im(P), la componente disipada es no nula
        ⇒ positrón forense (salvo casos degenerados).
        """
        if dim < 2:
            pytest.skip("requiere d≥2")
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        # ψ genérico
        psi = _RNG.standard_normal(dim)
        evo = agent_euclid.phase3.execute_quantum_channel(
            psi, random_grad, synth, mono
        )
        diss = (np.eye(dim) - synth.P_hat) @ psi
        diss_norm = float(la.norm(diss))
        if diss_norm > 1e-10 and not synth.is_trivial_obstruction:
            # P=I ⇒ E1=0 ⇒ sin disipación
            assert evo.antimatter_emission is not None
            assert evo.antimatter_emission.homological_charge == -1
            assert len(evo.antimatter_emission.authorization_signature) == 64

    def test_no_antimatter_when_psi_in_image(
        self, agent_euclid, dim, random_grad
    ):
        """Si ψ = P ψ (ya proyectado), E1ψ=0 ⇒ sin positrón por norma."""
        synth, mono = self._synth_mono(agent_euclid, random_grad)
        psi = synth.P_hat @ _RNG.standard_normal(dim)
        evo = agent_euclid.phase3.execute_quantum_channel(
            psi, random_grad, synth, mono
        )
        diss = (np.eye(dim) - synth.P_hat) @ psi
        if float(la.norm(diss)) < 1e-12:
            # Puede no emitir si disipación nula
            assert evo.dissipated_entropy == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# 5. COMPOSICIÓN FUNTORIAL
# =============================================================================


class TestFunctorialComposition:
    def test_end_to_end_happy_path(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        evo = agent_euclid.purify_and_tune_cavity(random_psi, random_grad)
        assert isinstance(evo, QuantumChannelEvolution)
        assert evo.dimensional_audit.is_coherent
        assert evo.monodromy_state is not None
        assert evo.monodromy_state.is_asymptotically_stable
        assert evo.kraus_residual_eig < 1e-8
        assert evo.coherent_state.shape == (dim,)
        assert evo.rho_post.shape == (dim, dim)

    def test_composition_matches_manual_phases(
        self, agent_euclid, dim, random_psi, random_grad
    ):
        # Manual
        s = agent_euclid.phase1.synthesize_projector(random_grad)
        m = agent_euclid.phase2.audit_monodromy(s)
        e_manual = agent_euclid.phase3.execute_quantum_channel(
            random_psi, random_grad, s, m
        )
        # Pipeline
        e_pipe = agent_euclid.purify_and_tune_cavity(random_psi, random_grad)
        np.testing.assert_allclose(
            e_pipe.coherent_state, e_manual.coherent_state, rtol=1e-12
        )
        assert e_pipe.von_neumann_post == pytest.approx(
            e_manual.von_neumann_post, abs=1e-12
        )
        assert e_pipe.kraus_residual_fro == pytest.approx(
            e_manual.kraus_residual_fro, abs=1e-15
        )

    def test_deterministic(self, agent_euclid, random_psi, random_grad):
        e1 = agent_euclid.purify_and_tune_cavity(random_psi, random_grad)
        e2 = agent_euclid.purify_and_tune_cavity(random_psi, random_grad)
        np.testing.assert_array_equal(e1.coherent_state, e2.coherent_state)
        assert e1.delta_entropy == e2.delta_entropy
        assert e1.kraus_residual_eig == e2.kraus_residual_eig

    def test_trivial_obstruction_full_pipeline(self, agent_euclid, dim, random_psi):
        """∇H=0 ⇒ P=I ⇒ canal = identidad sobre ψ, S_post=S_pre=0."""
        evo = agent_euclid.purify_and_tune_cavity(random_psi, np.zeros(dim))
        np.testing.assert_allclose(
            evo.coherent_state, random_psi, rtol=1e-12, atol=1e-12
        )
        assert evo.von_neumann_pre == pytest.approx(0.0, abs=1e-10)
        assert evo.von_neumann_post == pytest.approx(0.0, abs=1e-10)
        assert evo.purity_post == pytest.approx(1.0, abs=1e-10)
        assert evo.monodromy_state is not None
        assert evo.monodromy_state.spectral_radius == pytest.approx(1.0, abs=1e-12)

    def test_anisotropic_metric_pipeline(
        self, agent_aniso, dim, random_psi, random_grad
    ):
        evo = agent_aniso.purify_and_tune_cavity(random_psi, random_grad)
        assert evo.dimensional_audit.dimension == dim
        assert evo.kraus_residual_fro < 1e-7
        assert evo.monodromy_state is not None
        assert evo.monodromy_state.is_asymptotically_stable


# =============================================================================
# 6. INTEGRACIÓN DEL AGENTE (forward / propiedades)
# =============================================================================


class TestFloquetMonodromyAgentIntegration:
    def test_metric_tensor_is_defensive_copy(self, agent_euclid, dim):
        G = agent_euclid.metric_tensor
        G[0, 0] = -999.0
        assert agent_euclid.metric_tensor[0, 0] != -999.0

    def test_phase_ports_exposed(self, agent_euclid):
        assert agent_euclid.phase1_synthesizer is agent_euclid.phase1
        assert agent_euclid.phase2_auditor is agent_euclid.phase2
        assert agent_euclid.phase3_channel is agent_euclid.phase3

    def test_forward_categorical_state(self, agent_euclid, dim, random_psi):
        try:
            from app.core.mic_algebra import CategoricalState
        except ImportError:
            from floquet_agent import CategoricalState  # type: ignore

        state = CategoricalState(payload=random_psi, label="test")
        out = agent_euclid.forward(state)
        assert out.payload.shape == (dim,)
        # forward usa grad canónico e_0
        evo = agent_euclid.purify_and_tune_cavity(
            random_psi, np.eye(dim)[0]
        )
        np.testing.assert_allclose(out.payload, evo.coherent_state, rtol=1e-12)

    def test_forward_wrong_shape_raises(self, agent_euclid, dim):
        try:
            from app.core.mic_algebra import CategoricalState
        except ImportError:
            from floquet_agent import CategoricalState  # type: ignore

        state = CategoricalState(payload=np.ones(dim + 3), label="bad")
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.forward(state)

    def test_backward_equals_forward_convention(
        self, agent_euclid, dim, random_psi
    ):
        try:
            from app.core.mic_algebra import CategoricalState
        except ImportError:
            from floquet_agent import CategoricalState  # type: ignore

        state = CategoricalState(payload=random_psi, label="x")
        np.testing.assert_allclose(
            agent_euclid.backward(state).payload,
            agent_euclid.forward(state).payload,
            rtol=1e-12,
        )


# =============================================================================
# 7. ESTRÉS NUMÉRICO E INVARIANTES
# =============================================================================


class TestNumericalStressAndInvariants:
    def test_d1_full_pipeline(self):
        G = np.array([[2.5]])
        agent = FloquetMonodromyAgent(metric_tensor=G)
        psi = np.array([1.5])
        grad = np.array([0.3])
        evo = agent.purify_and_tune_cavity(psi, grad)
        assert evo.coherent_state.shape == (1,)
        assert evo.rho_post.shape == (1, 1)
        assert evo.kraus_residual_eig < 1e-12

    def test_high_condition_metric(self):
        d = 4
        G = make_spd(d, cond=1e6, seed=77)
        agent = FloquetMonodromyAgent(
            metric_tensor=G, projector_tolerance=1e-6, kraus_tolerance=1e-7
        )
        psi = _RNG.standard_normal(d)
        grad = _RNG.standard_normal(d)
        evo = agent.purify_and_tune_cavity(psi, grad)
        assert evo.monodromy_state is not None
        assert evo.monodromy_state.is_asymptotically_stable
        assert evo.kraus_residual_fro < 1e-5

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_kraus_completeness_ensemble(self, seed: int):
        d = 5
        G = make_spd(d, cond=4.0, seed=seed)
        agent = FloquetMonodromyAgent(metric_tensor=G)
        grad = np.random.default_rng(seed + 100).standard_normal(d)
        psi = np.random.default_rng(seed + 200).standard_normal(d)
        evo = agent.purify_and_tune_cavity(psi, grad)
        assert evo.kraus_residual_eig < 1e-7
        assert evo.kraus_residual_fro < 1e-6
        # Traza de ρ_post ≈ ‖ψ‖²
        assert float(np.trace(evo.rho_post)) == pytest.approx(
            float(psi @ psi), rel=1e-8, abs=1e-10
        )

    @pytest.mark.parametrize("seed", [10, 11, 12, 13])
    def test_monodromy_radius_le_one_ensemble(self, seed: int):
        d = 4
        agent = FloquetMonodromyAgent(metric_tensor=np.eye(d))
        grad = np.random.default_rng(seed).standard_normal(d)
        synth = agent.phase1.synthesize_projector(grad)
        mono = agent.phase2.audit_monodromy(synth)
        assert mono.spectral_radius <= 1.0 + 1e-9
        assert mono.projector_idempotence_residual < 1e-7

    @pytest.mark.parametrize("d", [2, 3, 5, 8])
    def test_pipeline_across_dimensions(self, d: int):
        agent = FloquetMonodromyAgent(metric_tensor=make_spd(d, cond=3.0, seed=d))
        psi = np.random.default_rng(d + 50).standard_normal(d)
        grad = np.random.default_rng(d + 60).standard_normal(d)
        evo = agent.purify_and_tune_cavity(psi, grad)
        assert evo.coherent_state.shape == (d,)
        assert evo.dimensional_audit.dimension == d
        assert 0.0 <= evo.purity_post <= 1.0 + 1e-10
        assert evo.von_neumann_pre >= -1e-12
        assert evo.von_neumann_post >= -1e-12

    def test_projector_eigenvalues_in_unit_interval_via_monodromy(self):
        """Para P ortogonal idempotente, μ_k ∈ {0,1} exactamente."""
        d, r = 6, 2
        P = make_projector_euclid(d, rank=r, seed=123)
        agent = FloquetMonodromyAgent(metric_tensor=np.eye(d))
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=float(la.norm(P @ P - P, "fro")),
            symmetry_residual=0.0,
            dim=d,
        )
        mono = agent.phase2.audit_monodromy(synth)
        mult = np.sort(np.real_if_close(mono.multipliers))
        expected = np.array([0.0] * (d - r) + [1.0] * r)
        np.testing.assert_allclose(mult, expected, atol=1e-10)

    def test_cp_map_increases_entropy_possible(self):
        """
        Documenta que CPTP **puede** aumentar S_vN (no se impone ΔS≤0).
        Canal de proyección: estado puro fuera de im(P) → mezcla o puro en im(P).
        """
        d = 2
        agent = FloquetMonodromyAgent(metric_tensor=np.eye(d))
        # P = |0⟩⟨0|
        P = np.array([[1.0, 0.0], [0.0, 0.0]])
        synth = ProjectorSynthesisResult(
            P_hat=P,
            reflector=None,
            normal_G_norm=1.0,
            is_trivial_obstruction=False,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            dim=d,
        )
        mono = agent.phase2.audit_monodromy(synth)
        # Superposición equitativa
        psi = np.array([1.0, 1.0]) / math.sqrt(2.0)
        evo = agent.phase3.execute_quantum_channel(
            psi, np.array([1.0, 0.0]), synth, mono
        )
        # S_pre = 0 (puro); S_post ≥ 0
        assert evo.von_neumann_pre == pytest.approx(0.0, abs=1e-12)
        assert evo.von_neumann_post >= -1e-12
        # ΔS puede ser > 0 — no assert ΔS ≤ 0 (regresión del axioma falso de v2)
        assert evo.delta_entropy == pytest.approx(
            evo.von_neumann_post - evo.von_neumann_pre, abs=1e-12
        )


class TestMachineEpsConsistency:
    def test_machine_eps_positive(self):
        assert _MACHINE_EPS == float(np.finfo(np.float64).eps)

    def test_default_tolerances_sane(self):
        assert 0.0 < _DEFAULT_STABILITY_TOL < 1e-3
        assert 0.0 < _DEFAULT_KRAUS_TOL < 1e-3
        assert 0.0 < _DEFAULT_PROJECTOR_TOL < 1e-3


# =============================================================================
# Entrada directa
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))