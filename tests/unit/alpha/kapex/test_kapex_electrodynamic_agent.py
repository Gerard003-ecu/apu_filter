# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite  : test_kapex_electrodynamic_agent.py                                  |
| Objetivo: Validación rigurosa del endofuntor K_APEX v7.0.0                   |
|           (métrica · calibre 𝔰𝔬(n) · haces · retícula de Boole)              |
+==============================================================================+

Arquitectura de la suite (espejo de las 3 fases anidadas)
--------------------------------------------------------
  Sección 0  — Fixtures y constructores canónicos (SPD, isometrías, plaquetas)
  Sección 1  — Fase 1: validación métrica, Cholesky, cierre espectral, bilateral
  Sección 2  — Fase 2: inyección, Eikonal, Poynting, curvatura 𝔰𝔬(n), Boole
  Sección 3  — Fase 3: cofrontera, identidad de Hodge, espectro de Δ_APEX
  Sección 4  — Composición funtorial export ∘ synthesize ∘ build_context
  Sección 5  — Retícula de Boole y excepciones categóricas
  Sección 6  — Propiedades numéricas de estrés (κ alto, n=1, n grande)

Ejecución
---------
  pytest test_kapex_electrodynamic_agent.py -v --tb=short
  pytest test_kapex_electrodynamic_agent.py -k "yang_mills or hodge" -v
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import del SUT (System Under Test) con fallback de path local
# ---------------------------------------------------------------------------
try:
    from app.agents.alpha.kapex.kapex_electrodynamic_agent import (
        ApexConditionError,
        ApexDimensionError,
        ApexParameterError,
        ApexPreparationContext,
        ApexStateTensor,
        ApexSymmetryError,
        ApexViabilityFlags,
        ElectrodynamicApexError,
        EikonalRefractionError,
        FinancialBlackHoleError,
        GaugeCovarianceError,
        GaugePotentialError,
        HolonomyVetoError,
        KApexElectrodynamicAgent,
        MetricInverseError,
        SheafMetricError,
        SheafStalkApex,
        SpectralClosureError,
        describe_viability_flags,
        _MACHINE_EPS,
        _ANTISYM_REL_TOL,
    )
except ImportError:
    from kapex_electrodynamic_agent import (  # type: ignore
        ApexConditionError,
        ApexDimensionError,
        ApexParameterError,
        ApexPreparationContext,
        ApexStateTensor,
        ApexSymmetryError,
        ApexViabilityFlags,
        ElectrodynamicApexError,
        EikonalRefractionError,
        FinancialBlackHoleError,
        GaugeCovarianceError,
        GaugePotentialError,
        HolonomyVetoError,
        KApexElectrodynamicAgent,
        MetricInverseError,
        SheafMetricError,
        SheafStalkApex,
        SpectralClosureError,
        describe_viability_flags,
        _MACHINE_EPS,
        _ANTISYM_REL_TOL,
    )


# =============================================================================
# SECCIÓN 0 — FIXTURES Y CONSTRUCTORES CANÓNICOS
# =============================================================================

_EPS = float(np.finfo(np.float64).eps)
_RNG = np.random.default_rng(20260328)


def _random_spd(n: int, cond: float = 10.0, seed: int | None = None) -> NDArray[np.float64]:
    """
    Genera A ≻ 0 con κ(A) ≈ cond mediante espectro controlado:
        A = Q · diag(λ) · Qᵀ,  λ_max/λ_min = cond.
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    # Ortogonal vía QR de Gaussiana
    Q, _ = la.qr(rng.standard_normal((n, n)))
    # Espectro log-uniforme entre 1 y cond
    log_lambdas = np.linspace(0.0, math.log(cond), n)
    lambdas = np.exp(log_lambdas)
    return (Q * lambdas) @ Q.T


def _random_psd(
    n: int, rank: int | None = None, seed: int | None = None
) -> NDArray[np.float64]:
    """Genera R ⪰ 0 de rango prescrito (default: full rank)."""
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    r = n if rank is None else max(0, min(rank, n))
    if r == 0:
        return np.zeros((n, n), dtype=np.float64)
    B = rng.standard_normal((n, r))
    R = B @ B.T
    return 0.5 * (R + R.T)


def _canonical_metric_bundle(
    n: int = 4,
    cond_G: float = 10.0,
    rank_R: int | None = None,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Tripleta canónica (G, G⁻¹, R) bien condicionada para camino feliz.
    G⁻¹ se calcula por resolución exacta (no por inversión densa ingenua
    cuando n es moderado: usamos la.inv que es estable para κ bajo).
    """
    G = _random_spd(n, cond=cond_G, seed=seed)
    G = 0.5 * (G + G.T)
    G_inv = la.inv(G)
    G_inv = 0.5 * (G_inv + G_inv.T)
    R = _random_psd(n, rank=rank_R, seed=seed + 1)
    return G, G_inv, R


def _random_so_generator(n: int, seed: int | None = None) -> NDArray[np.float64]:
    """Generador aleatorio de 𝔰𝔬(n): ½(M − Mᵀ)."""
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    M = rng.standard_normal((n, n))
    return 0.5 * (M - M.T)


def _random_matrix(n: int, seed: int | None = None) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    return rng.standard_normal((n, n))


def _G_isometry(G: NDArray[np.float64], seed: int = 7) -> NDArray[np.float64]:
    """
    Construye Q ∈ O_G(n): Qᵀ G Q = G.

    Método: si G = L Lᵀ (Cholesky), entonces
        Q = L · U · L⁻¹  con U ∈ O(n) arbitraria
    satisface Qᵀ G Q = G.
    """
    n = G.shape[0]
    L = la.cholesky(0.5 * (G + G.T), lower=True)
    U, _ = la.qr(_RNG.standard_normal((n, n)))
    # Forzar det(U)=+1 si es posible (no estrictamente necesario para O_G)
    if la.det(U) < 0:
        U[:, 0] *= -1.0
    L_inv = la.solve_triangular(L, np.eye(n), lower=True)
    Q = L @ U @ L_inv
    return Q


def _make_agent(
    n: int = 4,
    cond_G: float = 10.0,
    rank_R: int | None = None,
    kappa_max: float = 1.0e10,
    eikonal_slack: float = 0.1,
    holonomy_tol_rel: float = 1.0e-4,
    seed: int = 42,
) -> KApexElectrodynamicAgent:
    G, G_inv, R = _canonical_metric_bundle(n, cond_G, rank_R, seed)
    return KApexElectrodynamicAgent(
        G_mu_nu=G,
        G_inv=G_inv,
        R_cost=R,
        kappa_max=kappa_max,
        eikonal_slack=eikonal_slack,
        holonomy_tol_rel=holonomy_tol_rel,
    )


def _happy_path_fields(
    n: int, agent: KApexElectrodynamicAgent
) -> dict:
    """
    Campos de entrada que satisfacen Eikonal, Poynting y holonomía trivial
    (A₁=A₂ ⇒ F=0 ⇒ S_YM=0).
    """
    rng = np.random.default_rng(99)
    d_Phi = rng.standard_normal(n)
    # Gradiente de fase grande ⇒ Eikonal holgado
    phase_gradient = 5.0 * rng.standard_normal(n)
    # E, H alineados ⇒ P_in > 0; grad_H pequeño ⇒ P_diss pequeño
    E_field = rng.standard_normal(n)
    H_field = E_field.copy()  # E·H = ‖E‖² ≥ 0
    grad_H = 1.0e-3 * rng.standard_normal(n)
    # Plaqueta trivial: A₁ = A₂ ⇒ F = 0
    A = _random_matrix(n, seed=11)
    return dict(
        d_Phi=d_Phi,
        phase_gradient=phase_gradient,
        sigma_stress=0.2,
        E_field=E_field,
        H_field=H_field,
        grad_H=grad_H,
        A_gauge_1=A,
        A_gauge_2=A.copy(),  # holonomía trivial
        alpha_fermat=0.5,
    )


# =============================================================================
# SECCIÓN 1 — FASE 1: VALIDACIÓN MÉTRICA
# =============================================================================


class TestPhase1Dimensions:
    """Verificación dimensional estricta de G, G_inv, R."""

    def test_accepts_consistent_square_matrices(self):
        agent = _make_agent(n=3)
        assert agent.context.dim == 3

    def test_rejects_nonsquare_G(self):
        G = np.ones((3, 2))
        G_inv = np.eye(3)
        R = np.eye(3)
        with pytest.raises(ApexDimensionError, match="G_mu_nu"):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_rejects_dimension_mismatch_G_inv(self):
        G = np.eye(3)
        G_inv = np.eye(4)
        R = np.eye(3)
        with pytest.raises(ApexDimensionError):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_rejects_dimension_mismatch_R(self):
        G = np.eye(3)
        G_inv = np.eye(3)
        R = np.eye(2)
        with pytest.raises(ApexDimensionError):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_rejects_zero_dimension(self):
        G = np.zeros((0, 0))
        with pytest.raises(ApexDimensionError):
            KApexElectrodynamicAgent(G, G, G)


class TestPhase1Symmetry:
    """Simetría Frobenius de G, G_inv, R."""

    def test_rejects_asymmetric_G(self):
        G = np.array([[2.0, 1.0], [0.0, 2.0]])  # no simétrica
        G_inv = la.inv(0.5 * (G + G.T))
        R = np.eye(2)
        with pytest.raises(ApexSymmetryError, match="G_mu_nu"):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_accepts_numerically_symmetric_within_eps(self):
        """Perturbación O(ε_mach) no debe disparar simetría."""
        n = 4
        G, G_inv, R = _canonical_metric_bundle(n, seed=1)
        G_perturbed = G.copy()
        G_perturbed[0, 1] += 0.1 * _EPS * la.norm(G, "fro")
        # Re-simetrizar G_inv coherente
        G_sym = 0.5 * (G_perturbed + G_perturbed.T)
        # Si la asimetría es sub-tolerancia, el validador debe aceptar
        # (el código simetriza internamente solo en SPD/Cholesky, pero
        #  _validate_symmetry usa tol = ε·max(‖A‖_F,1)).
        asym = la.norm(G_perturbed - G_perturbed.T, "fro")
        tol = _EPS * max(la.norm(G_perturbed, "fro"), 1.0)
        if asym <= tol:
            agent = KApexElectrodynamicAgent(G_perturbed, la.inv(G_sym), R)
            assert agent.context.dim == n


class TestPhase1SPDAndCondition:
    """SPD y número de condición espectral."""

    def test_rejects_indefinite_G(self):
        G = np.diag([1.0, -1.0])  # indefinida
        with pytest.raises(ElectrodynamicApexError, match="SPD|Definida Positiva"):
            KApexElectrodynamicAgent(G, la.inv(np.diag([1.0, 1.0])), np.eye(2))

    def test_rejects_ill_conditioned_G(self):
        n = 3
        G = _random_spd(n, cond=1.0e14, seed=5)
        G_inv = la.inv(G)
        R = np.eye(n)
        with pytest.raises(ApexConditionError, match="mal condicionada"):
            KApexElectrodynamicAgent(G, G_inv, R, kappa_max=1.0e8)

    def test_kappa_crosscheck_identity(self):
        """κ(G⁻¹) ≈ κ(G) por identidad exacta (rel < 10⁻³ en camino feliz)."""
        agent = _make_agent(n=5, cond_G=50.0, seed=8)
        rel = abs(agent.context.kappa_G_inv - agent.context.kappa_G) / max(
            agent.context.kappa_G, 1.0
        )
        assert rel < 1.0e-3

    def test_n1_spd_scalar(self):
        """Caso degenerado n=1: SPD se reduce a G[0,0] > 0."""
        G = np.array([[2.5]])
        G_inv = np.array([[0.4]])
        R = np.array([[1.0]])
        agent = KApexElectrodynamicAgent(G, G_inv, R)
        assert agent.context.dim == 1
        assert agent.context.kappa_G == pytest.approx(1.0)


class TestPhase1InverseConsistency:
    """Consistencia métrica bilateral G G⁻¹ ≈ I ≈ G⁻¹ G."""

    def test_rejects_wrong_inverse(self):
        G = np.eye(3) * 2.0
        G_inv_wrong = np.eye(3) * 0.1  # debería ser 0.5
        R = np.eye(3)
        with pytest.raises(MetricInverseError):
            KApexElectrodynamicAgent(G, G_inv_wrong, R)

    def test_bilateral_residual_small_on_happy_path(self):
        agent = _make_agent(n=6, cond_G=20.0)
        assert agent.context.inverse_residual < 1.0e-10

    def test_inverse_residual_scales_gracefully_with_kappa(self):
        """Para κ moderado el residuo bilateral permanece O(κ·ε·n)."""
        n = 4
        cond = 1.0e6
        agent = _make_agent(n=n, cond_G=cond, seed=13)
        # Cota de Wilkinson laxa (el validador ya pasó): residuo << 1
        assert agent.context.inverse_residual < 1.0e-6


class TestPhase1SpectralDiagnostics:
    """PSD de R_cost, rango, β₀, brecha y cierre espectral."""

    def test_rank_and_betti_consistent(self):
        n, rank = 5, 3
        agent = _make_agent(n=n, rank_R=rank, seed=21)
        assert agent.context.rank_R == rank
        assert agent.context.betti_0_R == n - rank

    def test_full_rank_R_betti_zero(self):
        agent = _make_agent(n=4, rank_R=4, seed=22)
        assert agent.context.betti_0_R == 0
        assert agent.context.rank_R == 4

    def test_zero_R_full_kernel(self):
        G, G_inv, _ = _canonical_metric_bundle(n=3, seed=23)
        R = np.zeros((3, 3))
        agent = KApexElectrodynamicAgent(G, G_inv, R)
        assert agent.context.rank_R == 0
        assert agent.context.betti_0_R == 3

    def test_rejects_negative_eigenvalue_R(self):
        G, G_inv, _ = _canonical_metric_bundle(n=2, seed=24)
        R = np.diag([1.0, -0.5])  # no PSD
        with pytest.raises(ApexSymmetryError, match="PSD|Entropía"):
            KApexElectrodynamicAgent(G, G_inv, R)

    def test_spectral_closure_residual_near_machine_eps(self):
        agent = _make_agent(n=5, seed=25)
        assert agent.context.spectral_closure_residual < 1.0e-10

    def test_R_sqrt_squared_recovers_R(self):
        """‖R_sqrt² − R‖_F / ‖R‖_F ≈ 0 (cierre algebraico)."""
        agent = _make_agent(n=4, seed=26)
        R = agent.context.R_cost
        Rs = agent.context.R_sqrt
        closure = la.norm(Rs @ Rs - R, "fro") / max(la.norm(R, "fro"), 1.0)
        assert closure == pytest.approx(
            agent.context.spectral_closure_residual, rel=1e-6, abs=1e-14
        )
        assert closure < 1.0e-10

    def test_R_sqrt_is_symmetric(self):
        agent = _make_agent(n=4, seed=27)
        Rs = agent.context.R_sqrt
        assert la.norm(Rs - Rs.T, "fro") < 1.0e-14


class TestPhase1CholeskyRegularization:
    """Cholesky y jitter de Tikhonov."""

    def test_epsilon_G_zero_for_well_conditioned(self):
        agent = _make_agent(n=3, cond_G=5.0, seed=30)
        assert agent.context.epsilon_G == 0.0

    def test_L_G_reconstructs_G(self):
        agent = _make_agent(n=4, seed=31)
        L = agent.context.L_G
        G = agent.context.G_mu_nu
        recon = L @ L.T
        # Si no hubo regularización, reconstrucción exacta hasta redondeo
        if agent.context.epsilon_G == 0.0:
            rel = la.norm(recon - G, "fro") / max(la.norm(G, "fro"), 1.0)
            assert rel < 1.0e-12


class TestPhase1Parameters:
    """Validación de parámetros de construcción del orquestador."""

    def test_rejects_kappa_max_le_one(self):
        G, G_inv, R = _canonical_metric_bundle(n=2)
        with pytest.raises(ApexParameterError, match="kappa_max"):
            KApexElectrodynamicAgent(G, G_inv, R, kappa_max=1.0)

    def test_rejects_eikonal_slack_out_of_range(self):
        G, G_inv, R = _canonical_metric_bundle(n=2)
        with pytest.raises(ApexParameterError, match="eikonal_slack"):
            KApexElectrodynamicAgent(G, G_inv, R, eikonal_slack=1.0)
        with pytest.raises(ApexParameterError, match="eikonal_slack"):
            KApexElectrodynamicAgent(G, G_inv, R, eikonal_slack=-0.1)

    def test_rejects_nonpositive_holonomy_tol(self):
        G, G_inv, R = _canonical_metric_bundle(n=2)
        with pytest.raises(ApexParameterError, match="holonomy_tol_rel"):
            KApexElectrodynamicAgent(G, G_inv, R, holonomy_tol_rel=0.0)


class TestPhase1ContextImmutability:
    """ApexPreparationContext es frozen y contiene copias defensivas."""

    def test_context_is_frozen(self):
        agent = _make_agent(n=3)
        with pytest.raises(Exception):
            agent.context.dim = 99  # type: ignore[misc]

    def test_context_holds_defensive_copies(self):
        G, G_inv, R = _canonical_metric_bundle(n=3, seed=40)
        agent = KApexElectrodynamicAgent(G, G_inv, R)
        G[0, 0] = -999.0  # mutar original
        assert agent.context.G_mu_nu[0, 0] != -999.0


# =============================================================================
# SECCIÓN 2 — FASE 2: SÍNTESIS ELECTRODINÁMICA
# =============================================================================


class TestPhase2GaugeInjection:
    """Inyección de potencial de calibre s = dΦ · exp(−½ Tr G)."""

    def test_suppression_factor_formula(self):
        agent = _make_agent(n=3, seed=50)
        expected = float(np.exp(-0.5 * np.trace(agent.context.G_mu_nu)))
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.suppression_factor == pytest.approx(expected, rel=1e-12)

    def test_s_val_scales_with_d_Phi(self):
        agent = _make_agent(n=3, seed=51)
        fields = _happy_path_fields(3, agent)
        t1 = agent.synthesize_apex_field(**fields)
        fields["d_Phi"] = 2.0 * fields["d_Phi"]
        t2 = agent.synthesize_apex_field(**fields)
        np.testing.assert_allclose(
            t2.gauge_injection_vector,
            2.0 * t1.gauge_injection_vector,
            rtol=1e-12,
        )

    def test_rejects_wrong_d_Phi_shape(self):
        agent = _make_agent(n=3)
        fields = _happy_path_fields(3, agent)
        fields["d_Phi"] = np.ones(5)
        with pytest.raises(ApexDimensionError, match="d_Phi"):
            agent.synthesize_apex_field(**fields)

    def test_gauge_collapse_on_extreme_trace(self):
        """
        Tr(G) ≫ 1 ⇒ exp(−½ Tr G) < ε_mach ⇒ GaugePotentialError.
        Construimos G = λ I con λ grande.
        """
        n = 2
        lam = 100.0  # Tr = 200 ⇒ exp(-100) ≪ ε
        G = lam * np.eye(n)
        G_inv = (1.0 / lam) * np.eye(n)
        R = np.eye(n)
        agent = KApexElectrodynamicAgent(G, G_inv, R)
        fields = _happy_path_fields(n, agent)
        with pytest.raises(GaugePotentialError, match="supresión|colapsada"):
            agent.synthesize_apex_field(**fields)

    def test_suppression_invariant_under_orthogonal_conjugation(self):
        """
        Tr(Qᵀ G Q) = Tr(G) ⇒ factor de supresión invariante.
        (Verifica que el escalar depende solo de la traza.)
        """
        n = 3
        G, G_inv, R = _canonical_metric_bundle(n, seed=52)
        Q, _ = la.qr(_RNG.standard_normal((n, n)))
        G2 = Q.T @ G @ Q
        G2 = 0.5 * (G2 + G2.T)
        G2_inv = la.inv(G2)
        a1 = KApexElectrodynamicAgent(G, G_inv, R)
        a2 = KApexElectrodynamicAgent(G2, G2_inv, R)
        s1 = float(np.exp(-0.5 * np.trace(a1.context.G_mu_nu)))
        s2 = float(np.exp(-0.5 * np.trace(a2.context.G_mu_nu)))
        assert s1 == pytest.approx(s2, rel=1e-10)


class TestPhase2Eikonal:
    """Ecuación Eikonal de absorción / refracción de Fermat."""

    def test_refractive_index_bounds(self):
        """n(σ*) = 1 + tanh(α·σ*) ∈ (0, 2)."""
        agent = _make_agent(n=3, seed=60)
        fields = _happy_path_fields(3, agent)
        for sigma in (-10.0, 0.0, 10.0):
            fields["sigma_stress"] = sigma
            # Asegurar Eikonal con gradiente grande
            fields["phase_gradient"] = 20.0 * np.ones(3)
            tensor = agent.synthesize_apex_field(**fields)
            assert 0.0 < tensor.fermat_refractive_index < 2.0 + 1e-12

    def test_eikonal_failure_raises(self):
        agent = _make_agent(n=3, eikonal_slack=0.05, seed=61)
        fields = _happy_path_fields(3, agent)
        fields["phase_gradient"] = 1.0e-12 * np.ones(3)  # norma ~ 0
        fields["sigma_stress"] = 1.0  # n > 1
        with pytest.raises(EikonalRefractionError, match="Eikonal"):
            agent.synthesize_apex_field(**fields)

    def test_eikonal_norm_sq_is_quadratic_form(self):
        agent = _make_agent(n=4, seed=62)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        g = fields["phase_gradient"]
        expected = float(g @ agent.context.G_inv @ g)
        assert tensor.eikonal_norm_sq == pytest.approx(expected, rel=1e-12)


class TestPhase2Poynting:
    """Flujo exergético: P_diss = ‖R_sqrt ∇H‖² ≥ 0."""

    def test_dissipation_nonnegative(self):
        agent = _make_agent(n=4, seed=70)
        fields = _happy_path_fields(4, agent)
        # Gradiente arbitrario
        fields["grad_H"] = _RNG.standard_normal(4)
        # E·H grande para no caer en black hole
        fields["E_field"] = 10.0 * np.ones(4)
        fields["H_field"] = 10.0 * np.ones(4)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.poynting_dissipation >= -1.0e-14

    def test_dissipation_equals_quadratic_form(self):
        """‖R_sqrt ∇H‖² == ∇Hᵀ R ∇H (identidad algebraica)."""
        agent = _make_agent(n=4, seed=71)
        fields = _happy_path_fields(4, agent)
        gH = _RNG.standard_normal(4)
        fields["grad_H"] = gH
        fields["E_field"] = 50.0 * np.ones(4)
        fields["H_field"] = 50.0 * np.ones(4)
        tensor = agent.synthesize_apex_field(**fields)
        R = agent.context.R_cost
        expected = float(gH @ R @ gH)
        # Clamp numérico: forma cuadrática puede ser levemente negativa
        expected = max(expected, 0.0)
        assert tensor.poynting_dissipation == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_black_hole_when_dissipation_dominates(self):
        agent = _make_agent(n=3, rank_R=3, seed=72)
        fields = _happy_path_fields(3, agent)
        fields["E_field"] = np.zeros(3)       # P_in = 0
        fields["H_field"] = np.zeros(3)
        fields["grad_H"] = np.ones(3)         # P_diss > 0 si R ≻ 0
        # Si rank_R=3 y R no es nula, P_diss > 0 ⇒ black hole
        if agent.context.rank_R > 0:
            with pytest.raises(FinancialBlackHoleError):
                agent.synthesize_apex_field(**fields)

    def test_exergy_balance_identity(self):
        agent = _make_agent(n=3, seed=73)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.poynting_exergy_flux == pytest.approx(
            tensor.poynting_income - tensor.poynting_dissipation,
            rel=1e-12,
            abs=1e-14,
        )


class TestPhase2YangMillsCurvature:
    """
    Curvatura F ∈ 𝔰𝔬(n), acción S_YM correcta y holonomía.
    Cubre la corrección del bug de v6: S_YM = ½ Tr(Fᵀ G F G⁻¹).
    """

    def test_trivial_plaquette_vanishing_action(self):
        """A₁ = A₂ ⇒ F = 0 ⇒ S_YM = 0."""
        agent = _make_agent(n=4, holonomy_tol_rel=1e-12, seed=80)
        fields = _happy_path_fields(4, agent)
        A = _random_matrix(4, seed=80)
        fields["A_gauge_1"] = A
        fields["A_gauge_2"] = A.copy()
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.yang_mills_action == pytest.approx(0.0, abs=1e-14)
        assert tensor.curvature_antisymmetry_residual == pytest.approx(0.0, abs=1e-14)

    def test_curvature_is_antisymmetric(self):
        """Para A₁ ≠ A₂ genéricos, F = −Fᵀ hasta 10⁻⁸ relativo."""
        agent = _make_agent(n=5, holonomy_tol_rel=1.0e6, seed=81)
        # holonomy_tol generoso para no veto por S_YM
        fields = _happy_path_fields(5, agent)
        fields["A_gauge_1"] = _random_matrix(5, seed=1)
        fields["A_gauge_2"] = _random_matrix(5, seed=2)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.curvature_antisymmetry_residual <= _ANTISYM_REL_TOL

    def test_so_generators_produce_antisymmetric_F(self):
        """Si A_i ya ∈ 𝔰𝔬(n), la proyección es idempotente y F ∈ 𝔰𝔬(n)."""
        n = 4
        agent = _make_agent(n=n, holonomy_tol_rel=1.0e6, seed=82)
        fields = _happy_path_fields(n, agent)
        fields["A_gauge_1"] = _random_so_generator(n, seed=3)
        fields["A_gauge_2"] = _random_so_generator(n, seed=4)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.curvature_antisymmetry_residual <= 1.0e-12

    def test_S_YM_nonnegative(self):
        agent = _make_agent(n=4, holonomy_tol_rel=1.0e6, seed=83)
        fields = _happy_path_fields(4, agent)
        fields["A_gauge_1"] = _random_matrix(4, seed=5)
        fields["A_gauge_2"] = _random_matrix(4, seed=6)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.yang_mills_action >= -1.0e-14

    def test_S_YM_formula_matches_manual_contraction(self):
        """
        Verifica S_YM = ½ Tr(Fᵀ G F G⁻¹) contra cálculo manual independiente
        (detección del bug de v6 que introducía un F sobrante).
        """
        n = 3
        agent = _make_agent(n=n, holonomy_tol_rel=1.0e6, seed=84)
        A1 = _random_matrix(n, seed=10)
        A2 = _random_matrix(n, seed=11)

        # Replicar _compute_curvature
        A1a = 0.5 * (A1 - A1.T)
        A2a = 0.5 * (A2 - A2.T)
        F = (A2a - A1a) + (A1a @ A2a - A2a @ A1a)
        F = 0.5 * (F - F.T)  # proyección final

        G = agent.context.G_mu_nu
        G_inv = agent.context.G_inv
        S_manual = 0.5 * float(np.trace(F.T @ G @ F @ G_inv))
        S_manual = max(S_manual, 0.0)

        fields = _happy_path_fields(n, agent)
        fields["A_gauge_1"] = A1
        fields["A_gauge_2"] = A2
        tensor = agent.synthesize_apex_field(**fields)

        assert tensor.yang_mills_action == pytest.approx(S_manual, rel=1e-10, abs=1e-14)

        # Guardia anti-regresión del bug v6: la forma errónea NO debe coincidir
        # (salvo casos degenerados F≈0 o F²≈F).
        S_buggy = 0.5 * float(np.trace(F.T @ G @ F @ F @ G_inv))
        if la.norm(F, "fro") > 1e-6 and abs(S_manual) > 1e-12:
            # Si F no es idempotente, buggy ≠ correcto
            if abs(S_buggy - S_manual) / max(abs(S_manual), 1.0) > 1e-6:
                assert tensor.yang_mills_action != pytest.approx(S_buggy, rel=1e-4)

    def test_holonomy_veto_on_large_curvature(self):
        agent = _make_agent(n=3, holonomy_tol_rel=1.0e-15, seed=85)
        fields = _happy_path_fields(3, agent)
        fields["A_gauge_1"] = _random_matrix(3, seed=20)
        fields["A_gauge_2"] = _random_matrix(3, seed=21)
        # Con tol extremadamente estricta y A₁ ≠ A₂, debe vetar
        with pytest.raises(HolonomyVetoError):
            agent.synthesize_apex_field(**fields)

    def test_project_to_so_is_idempotent(self):
        n = 5
        agent = _make_agent(n=n, seed=86)
        M = _random_matrix(n, seed=30)
        Pi = agent.phase2._project_to_so
        assert la.norm(Pi(Pi(M)) - Pi(M), "fro") < 1e-14

    def test_lie_commutator_closes_so(self):
        """[B,C]ᵀ = −[B,C] si B,C ∈ 𝔰𝔬(n)."""
        n = 4
        agent = _make_agent(n=n, seed=87)
        B = _random_so_generator(n, seed=31)
        C = _random_so_generator(n, seed=32)
        comm = agent.phase2._lie_commutator(B, C)
        assert la.norm(comm + comm.T, "fro") < 1e-13


class TestPhase2GaugeCovariance:
    """Invarianza de S_YM bajo isometrías de G_μν (diagnóstico opt-in)."""

    def test_covariance_under_G_isometry(self):
        n = 3
        agent = _make_agent(n=n, holonomy_tol_rel=1.0e6, seed=90)
        fields = _happy_path_fields(n, agent)
        fields["A_gauge_1"] = _random_matrix(n, seed=40)
        fields["A_gauge_2"] = _random_matrix(n, seed=41)
        Q = _G_isometry(agent.context.G_mu_nu, seed=7)
        fields["Q_isometry_diagnostic"] = Q
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.gauge_covariance_residual is not None
        assert tensor.gauge_covariance_residual < 1.0e-6

    def test_rejects_non_isometry(self):
        n = 3
        agent = _make_agent(n=n, holonomy_tol_rel=1.0e6, seed=91)
        fields = _happy_path_fields(n, agent)
        fields["A_gauge_1"] = _random_matrix(n, seed=42)
        fields["A_gauge_2"] = _random_matrix(n, seed=43)
        # Q arbitraria: no preserva G
        fields["Q_isometry_diagnostic"] = _random_matrix(n, seed=44)
        with pytest.raises(GaugeCovarianceError, match="no preserva"):
            agent.synthesize_apex_field(**fields)

    def test_covariance_residual_none_when_diagnostic_omitted(self):
        agent = _make_agent(n=3, seed=92)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.gauge_covariance_residual is None


class TestPhase2ViabilityFlags:
    """Retícula de Boole de predicados de viabilidad."""

    def test_happy_path_all_flags(self):
        agent = _make_agent(n=4, cond_G=5.0, holonomy_tol_rel=1e-3, seed=100)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.viability_flags.is_order_unit()
        assert tensor.is_electrodynamically_viable is True
        assert ApexViabilityFlags.HOLONOMY_TRIVIAL in tensor.viability_flags
        assert ApexViabilityFlags.CURVATURE_ANTISYMMETRIC in tensor.viability_flags
        assert ApexViabilityFlags.SPECTRAL_CLOSURE_SOUND in tensor.viability_flags

    def test_describe_viability_flags_readable(self):
        flags = (
            ApexViabilityFlags.EXERGY_NONNEGATIVE
            | ApexViabilityFlags.HOLONOMY_TRIVIAL
        )
        text = describe_viability_flags(flags)
        assert "EXERGY_NONNEGATIVE" in text
        assert "HOLONOMY_TRIVIAL" in text
        assert "VIABLE_TOTAL=False" in text

    def test_meet_join_lattice_laws(self):
        a = ApexViabilityFlags.EXERGY_NONNEGATIVE
        b = ApexViabilityFlags.HOLONOMY_TRIVIAL
        assert a.meet(b) == ApexViabilityFlags.NONE
        assert a.join(b) == (a | b)
        assert a.meet(ApexViabilityFlags.ALL) == a
        assert a.join(ApexViabilityFlags.ALL) == ApexViabilityFlags.ALL
        assert a.meet(ApexViabilityFlags.NONE) == ApexViabilityFlags.NONE

    def test_metric_well_conditioned_flag_off_near_kappa_max(self):
        """
        Si κ(G) > ½ κ_max, la bandera METRIC_WELL_CONDITIONED no se enciende.
        """
        n = 3
        cond = 80.0
        kappa_max = 100.0  # ½ κ_max = 50 < 80 ⇒ bandera apagada
        G, G_inv, R = _canonical_metric_bundle(n, cond_G=cond, seed=101)
        agent = KApexElectrodynamicAgent(
            G, G_inv, R, kappa_max=kappa_max, holonomy_tol_rel=1e-3
        )
        # κ real puede diferir ligeramente del cond pedido
        if agent.context.kappa_G > 0.5 * kappa_max:
            fields = _happy_path_fields(n, agent)
            tensor = agent.synthesize_apex_field(**fields)
            assert ApexViabilityFlags.METRIC_WELL_CONDITIONED not in tensor.viability_flags
            assert not tensor.is_electrodynamically_viable


# =============================================================================
# SECCIÓN 3 — FASE 3: PROYECCIÓN EN HACES
# =============================================================================


class TestPhase3SheafConstruction:
    """δ_metric, δ_diss, δ_APEX, Δ_APEX."""

    def test_lazy_init_phase3(self):
        agent = _make_agent(n=3, seed=110)
        assert agent.phase3 is None
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert agent.phase3 is not None
        assert isinstance(stalk, SheafStalkApex)

    def test_delta_shapes(self):
        agent = _make_agent(n=4, seed=111)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        n = 4
        assert stalk.delta_metric.shape == (n, n)
        assert stalk.delta_dissipative.shape == (n, n)
        assert stalk.delta_apex.shape == (2 * n, n)
        assert stalk.hodge_laplacian.shape == (n, n)
        assert stalk.rank_delta == n

    def test_hodge_identity_near_machine_precision(self):
        """‖δᵀ G δ − I‖_F / n ≈ 0."""
        agent = _make_agent(n=5, cond_G=10.0, seed=112)
        fields = _happy_path_fields(5, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.hodge_metric_residual < 1.0e-10

        # Verificación manual independiente
        d = stalk.delta_metric
        G = agent.context.G_mu_nu
        I = np.eye(5)
        rel = la.norm(d.T @ G @ d - I, "fro") / 5
        assert rel == pytest.approx(stalk.hodge_metric_residual, rel=1e-9, abs=1e-15)

    def test_hodge_laplacian_spd(self):
        agent = _make_agent(n=4, seed=113)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        eigvals = la.eigvalsh(stalk.hodge_laplacian)
        assert eigvals[0] > 0.0
        # Δ = I + δ_dissᵀ δ_diss  ⇒  λ_min ≥ 1
        assert eigvals[0] >= 1.0 - 1e-10

    def test_hodge_laplacian_formula(self):
        """Δ = I + δ_dissᵀ δ_diss (reconstrucción exacta)."""
        agent = _make_agent(n=3, seed=114)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        I = np.eye(3)
        expected = I + stalk.delta_dissipative.T @ stalk.delta_dissipative
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(
            stalk.hodge_laplacian, expected, rtol=1e-12, atol=1e-14
        )

    def test_delta_apex_stacking(self):
        agent = _make_agent(n=3, seed=115)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        np.testing.assert_allclose(
            stalk.delta_apex[:3], stalk.delta_metric, rtol=1e-14
        )
        np.testing.assert_allclose(
            stalk.delta_apex[3:], stalk.delta_dissipative, rtol=1e-14
        )

    def test_projections_are_linear_images(self):
        agent = _make_agent(n=4, seed=116)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        s = tensor.gauge_injection_vector
        stalk = agent.export_sheaf_stalk(s)
        np.testing.assert_allclose(
            stalk.projected_source_metric,
            stalk.delta_metric @ s,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            stalk.projected_source_dissipative,
            stalk.delta_dissipative @ s,
            rtol=1e-12,
        )

    def test_lossless_subspace_inherits_betti(self):
        n, rank = 5, 2
        agent = _make_agent(n=n, rank_R=rank, seed=117)
        fields = _happy_path_fields(n, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.lossless_subspace_dimension == agent.context.betti_0_R
        assert stalk.lossless_subspace_dimension == n - rank

    def test_spectral_gap_nonnegative(self):
        agent = _make_agent(n=4, seed=118)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.hodge_spectral_gap >= -1e-14
        assert stalk.hodge_condition_number >= 1.0

    def test_rejects_wrong_s_val_shape(self):
        agent = _make_agent(n=3, seed=119)
        with pytest.raises(ApexDimensionError, match="s_val"):
            agent.export_sheaf_stalk(np.ones(7))

    def test_delta_diss_equals_R_sqrt_times_delta_metric(self):
        agent = _make_agent(n=4, seed=120)
        fields = _happy_path_fields(4, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        expected = agent.context.R_sqrt @ stalk.delta_metric
        np.testing.assert_allclose(
            stalk.delta_dissipative, expected, rtol=1e-12, atol=1e-14
        )


# =============================================================================
# SECCIÓN 4 — COMPOSICIÓN FUNTORIAL
# =============================================================================


class TestFunctorialComposition:
    """
    Contrato:
        export_stalk ∘ synthesize ∘ build_context  =  K_APEX
    """

    def test_end_to_end_happy_path(self):
        n = 4
        agent = _make_agent(n=n, cond_G=8.0, seed=130)
        fields = _happy_path_fields(n, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)

        assert isinstance(agent.context, ApexPreparationContext)
        assert isinstance(tensor, ApexStateTensor)
        assert isinstance(stalk, SheafStalkApex)
        assert tensor.is_electrodynamically_viable
        assert stalk.rank_delta == n
        np.testing.assert_array_equal(
            stalk.source_injection, tensor.gauge_injection_vector
        )

    def test_phase1_context_shared_across_phases(self):
        agent = _make_agent(n=3, seed=131)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        # Misma métrica subyacente
        assert agent.phase2._ctx is agent.context
        assert agent.phase3 is not None
        assert agent.phase3._ctx is agent.context
        assert stalk.lossless_subspace_dimension == agent.context.betti_0_R

    def test_repeated_export_reuses_phase3(self):
        agent = _make_agent(n=3, seed=132)
        fields = _happy_path_fields(3, agent)
        tensor = agent.synthesize_apex_field(**fields)
        s = tensor.gauge_injection_vector
        s1 = agent.export_sheaf_stalk(s)
        phase3_ref = agent.phase3
        s2 = agent.export_sheaf_stalk(s)
        assert agent.phase3 is phase3_ref
        np.testing.assert_allclose(
            s1.hodge_laplacian, s2.hodge_laplacian, rtol=1e-14
        )

    def test_synthesize_is_deterministic(self):
        agent = _make_agent(n=3, seed=133)
        fields = _happy_path_fields(3, agent)
        t1 = agent.synthesize_apex_field(**fields)
        t2 = agent.synthesize_apex_field(**fields)
        np.testing.assert_array_equal(
            t1.gauge_injection_vector, t2.gauge_injection_vector
        )
        assert t1.yang_mills_action == t2.yang_mills_action
        assert t1.viability_flags == t2.viability_flags


# =============================================================================
# SECCIÓN 5 — EXCEPCIONES CATEGÓRICAS Y JERARQUÍA
# =============================================================================


class TestExceptionHierarchy:
    """Toda excepción del módulo hereda de ElectrodynamicApexError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ApexDimensionError,
            ApexParameterError,
            ApexSymmetryError,
            ApexConditionError,
            MetricInverseError,
            SpectralClosureError,
            GaugePotentialError,
            EikonalRefractionError,
            FinancialBlackHoleError,
            HolonomyVetoError,
            GaugeCovarianceError,
            SheafMetricError,
        ],
    )
    def test_is_subclass_of_root(self, exc_cls):
        assert issubclass(exc_cls, ElectrodynamicApexError)

    def test_catch_all_with_root(self):
        G = np.array([[1.0, 2.0], [0.0, 1.0]])  # asimétrica
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(G, np.eye(2), np.eye(2))


# =============================================================================
# SECCIÓN 6 — ESTRÉS NUMÉRICO Y CASOS LÍMITE
# =============================================================================


class TestNumericalStress:
    """Casos límite: n=1, κ alto, R=0, dimensiones mayores."""

    def test_n1_full_pipeline(self):
        G = np.array([[3.0]])
        G_inv = np.array([[1.0 / 3.0]])
        R = np.array([[0.5]])
        agent = KApexElectrodynamicAgent(
            G, G_inv, R, holonomy_tol_rel=1e-3
        )
        fields = dict(
            d_Phi=np.array([1.0]),
            phase_gradient=np.array([5.0]),
            sigma_stress=0.1,
            E_field=np.array([2.0]),
            H_field=np.array([2.0]),
            grad_H=np.array([0.01]),
            A_gauge_1=np.array([[0.0]]),
            A_gauge_2=np.array([[0.0]]),
        )
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert tensor.is_electrodynamically_viable
        assert stalk.delta_apex.shape == (2, 1)
        assert stalk.hodge_laplacian[0, 0] >= 1.0 - 1e-12

    def test_zero_cost_R_vanishing_dissipation(self):
        n = 3
        G, G_inv, _ = _canonical_metric_bundle(n, seed=140)
        R = np.zeros((n, n))
        agent = KApexElectrodynamicAgent(G, G_inv, R, holonomy_tol_rel=1e-3)
        fields = _happy_path_fields(n, agent)
        fields["grad_H"] = _RNG.standard_normal(n)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.poynting_dissipation == pytest.approx(0.0, abs=1e-14)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        # δ_diss = 0 ⇒ Δ = I ⇒ gap = 0, κ = 1
        np.testing.assert_allclose(
            stalk.delta_dissipative, 0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            stalk.hodge_laplacian, np.eye(n), rtol=1e-12
        )
        assert stalk.hodge_condition_number == pytest.approx(1.0, abs=1e-10)
        assert stalk.lossless_subspace_dimension == n

    def test_moderate_high_condition(self):
        """κ ~ 1e7 aún debe completar el pipeline sin excepción."""
        n = 4
        agent = _make_agent(
            n=n, cond_G=1.0e7, kappa_max=1.0e10, holonomy_tol_rel=1e-3, seed=141
        )
        fields = _happy_path_fields(n, agent)
        # Gradiente grande para Eikonal con G_inv posiblemente anisotrópica
        fields["phase_gradient"] = 50.0 * _RNG.standard_normal(n)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.hodge_metric_residual < 1.0e-6  # tol escalada por κ

    def test_larger_dimension(self):
        n = 12
        agent = _make_agent(n=n, cond_G=15.0, holonomy_tol_rel=1e-3, seed=142)
        fields = _happy_path_fields(n, agent)
        fields["phase_gradient"] = 10.0 * _RNG.standard_normal(n)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.delta_apex.shape == (2 * n, n)
        assert tensor.curvature_antisymmetry_residual <= _ANTISYM_REL_TOL

    def test_identity_metric_simplifies_hodge(self):
        """Si G = I, entonces δ_metric = I y Δ = I + R (pues R_sqrt² = R)."""
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = _random_psd(n, rank=n, seed=143)
        agent = KApexElectrodynamicAgent(G, G_inv, R, holonomy_tol_rel=1e-3)
        fields = _happy_path_fields(n, agent)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        np.testing.assert_allclose(stalk.delta_metric, np.eye(n), atol=1e-12)
        # Δ = I + R_sqrtᵀ R_sqrt = I + R  (R_sqrt simétrica)
        expected_Delta = np.eye(n) + agent.context.R_sqrt @ agent.context.R_sqrt
        expected_Delta = 0.5 * (expected_Delta + expected_Delta.T)
        np.testing.assert_allclose(
            stalk.hodge_laplacian, expected_Delta, rtol=1e-10, atol=1e-12
        )


class TestMachineEpsConsistency:
    """El módulo exporta _MACHINE_EPS coherente con numpy."""

    def test_machine_eps_matches_numpy(self):
        assert _MACHINE_EPS == float(np.finfo(np.float64).eps)

    def test_antisym_tol_is_sane(self):
        assert 0.0 < _ANTISYM_REL_TOL < 1.0e-4


# =============================================================================
# SECCIÓN 7 — PROPIEDADES INVARIANTES (property-style, semillas fijas)
# =============================================================================


class TestInvariantProperties:
    """Invariantes algebraicos muestreados sobre un ensemble finito."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_S_YM_nonnegative_ensemble(self, seed: int):
        n = 4
        agent = _make_agent(n=n, holonomy_tol_rel=1.0e6, seed=seed)
        fields = _happy_path_fields(n, agent)
        fields["A_gauge_1"] = _random_matrix(n, seed=seed + 100)
        fields["A_gauge_2"] = _random_matrix(n, seed=seed + 200)
        tensor = agent.synthesize_apex_field(**fields)
        assert tensor.yang_mills_action >= -1e-13
        assert tensor.curvature_antisymmetry_residual <= _ANTISYM_REL_TOL

    @pytest.mark.parametrize("seed", [10, 11, 12, 13])
    def test_hodge_identity_ensemble(self, seed: int):
        n = 5
        agent = _make_agent(n=n, cond_G=30.0, seed=seed)
        fields = _happy_path_fields(n, agent)
        fields["phase_gradient"] = 15.0 * _RNG.standard_normal(n)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        # tol ~ 100 · κ · ε
        tol = 100.0 * agent.context.kappa_G * _EPS
        assert stalk.hodge_metric_residual <= tol + 1e-15

    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_pipeline_across_dimensions(self, n: int):
        agent = _make_agent(n=n, holonomy_tol_rel=1e-3, seed=50 + n)
        fields = _happy_path_fields(n, agent)
        fields["phase_gradient"] = 8.0 * np.ones(n)
        tensor = agent.synthesize_apex_field(**fields)
        stalk = agent.export_sheaf_stalk(tensor.gauge_injection_vector)
        assert stalk.rank_delta == n
        assert tensor.suppression_factor > 0.0