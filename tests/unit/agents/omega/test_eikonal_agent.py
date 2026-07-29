# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Suite  : test_eikonal_agent.py                                               |
| Objetivo: Validación rigurosa del endofuntor Eikonal v3.0.0                  |
|           (diafragma · superficie eikonal · Fermat/geodésica · lente)        |
| Versión: 1.0.0-Rigorous-Nested-Oracle                                        |
+==============================================================================+

Arquitectura (espejo de las 3 fases anidadas)
---------------------------------------------
  0. Utilidades / fixtures / oráculos independientes
  1. TestExceptionHierarchy
  2. TestPhase1*  — ρ física, pureza, diafragma, colapso
  3. TestPhase2*  — G⁻¹ SPD, eikonal, slack, refracción
  4. TestPhase3*  — Simpson/trapecio, Fermat, residuo geodésico
  5. TestFunctorialComposition — lens ∘ path ∘ eikonal ∘ aperture
  6. TestEikonalAgentIntegration — contrato público
  7. TestNumericalStressAndInvariants — ensemble, κ, d variable

Ejecución
---------
  pytest test_eikonal_agent.py -v --tb=short
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import del SUT con fallback local
# ---------------------------------------------------------------------------
try:
    from app.omega.eikonal_agent import (
        ApertureModulationResult,
        DimensionalMismatchError,
        EikonalAgent,
        EikonalControlInput,
        EikonalParameterError,
        EikonalPhaseState,
        EikonalRefractionError,
        EikonalSingularityError,
        EikonalSurfaceResult,
        FermatOpticalDeviationError,
        FermatPathResult,
        MetricSignatureError,
        QuantumPurityCollapseError,
        SpectralDensityAudit,
        TopologicalInvariantError,
        _DEFAULT_EIKONAL_SLACK,
        _DEFAULT_KAPPA_MAX,
        _MACHINE_EPS,
    )
except ImportError:
    from eikonal_agent import (  # type: ignore
        ApertureModulationResult,
        DimensionalMismatchError,
        EikonalAgent,
        EikonalControlInput,
        EikonalParameterError,
        EikonalPhaseState,
        EikonalRefractionError,
        EikonalSingularityError,
        EikonalSurfaceResult,
        FermatOpticalDeviationError,
        FermatPathResult,
        MetricSignatureError,
        QuantumPurityCollapseError,
        SpectralDensityAudit,
        TopologicalInvariantError,
        _DEFAULT_EIKONAL_SLACK,
        _DEFAULT_KAPPA_MAX,
        _MACHINE_EPS,
    )

try:
    from app.core.mic_algebra import TopologicalInvariantError as _TIE_ROOT
except ImportError:
    _TIE_ROOT = TopologicalInvariantError  # type: ignore[misc]

try:
    from app.omega.levi_civita_agent import TangentVector
except ImportError:
    try:
        from eikonal_agent import TangentVector  # type: ignore
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class TangentVector:  # type: ignore[no-redef]
            coordinates: NDArray[np.float64]


# =============================================================================
# 0. UTILIDADES Y ORÁCULOS INDEPENDIENTES
# =============================================================================

EPS = float(np.finfo(np.float64).eps)
_RNG = np.random.default_rng(20260329)


def make_spd(n: int, cond: float = 5.0, seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.standard_normal((n, n)))
    log_l = np.linspace(0.0, math.log(max(cond, 1.0 + 1e-15)), n)
    A = (Q * np.exp(log_l)) @ Q.T
    return 0.5 * (A + A.T)


def make_density(
    n: int,
    pure: bool = False,
    seed: int = 0,
    rank: int | None = None,
) -> NDArray[np.float64]:
    """
    Matriz de densidad real simétrica: ρ ⪰ 0, Tr(ρ)=1.
    pure=True ⇒ |ψ⟩⟨ψ|; rank controla el rango si no es pura.
    """
    rng = np.random.default_rng(seed)
    if pure:
        v = rng.standard_normal(n)
        v = v / la.norm(v)
        return np.outer(v, v)
    r = n if rank is None else max(1, min(rank, n))
    B = rng.standard_normal((n, r))
    R = B @ B.T
    R = 0.5 * (R + R.T)
    tr = float(np.trace(R))
    return R / max(tr, 1e-300)


def purity_oracle(rho: np.ndarray) -> float:
    """Tr(ρ²) para ρ simétrica de traza 1 (o no: se normaliza)."""
    H = 0.5 * (rho + rho.T)
    tr = float(np.trace(H))
    if tr <= 1e-30:
        return 0.0
    Hn = H / tr
    return float(np.trace(Hn @ Hn))


def von_neumann_oracle(rho: np.ndarray) -> float:
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


def fro_rel(A: np.ndarray, B: np.ndarray) -> float:
    return float(la.norm(A - B, "fro")) / max(float(la.norm(B, "fro")), 1.0)


def eikonal_norm_sq_oracle(
    grad: np.ndarray, G_inv: np.ndarray
) -> float:
    return float(grad @ G_inv @ grad)


def simpson_integral_oracle(norms: np.ndarray, dt: float) -> Tuple[float, str]:
    """Oráculo Simpson/trapecio independiente (misma política T par → T-1)."""
    n = len(norms)
    if n == 0:
        return 0.0, "empty"
    if n < 3:
        if n == 1:
            return float(norms[0]) * dt, "trapezoid"
        return 0.5 * (float(norms[0]) + float(norms[-1])) * dt, "trapezoid"
    if n % 2 == 0:
        norms = norms[:-1]
        n = len(norms)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((dt / 3.0) * np.dot(w, norms)), "simpson"


def speed_norms_G(V: np.ndarray, G: np.ndarray) -> np.ndarray:
    q = np.einsum("ti,ij,tj->t", V, G, V)
    return np.sqrt(np.maximum(q, 0.0))


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(params=[2, 3, 4])
def dim(request) -> int:
    return int(request.param)


@pytest.fixture
def G_euclid(dim) -> NDArray[np.float64]:
    return np.eye(dim, dtype=np.float64)


@pytest.fixture
def G_aniso(dim) -> NDArray[np.float64]:
    return make_spd(dim, cond=6.0, seed=dim + 11)


@pytest.fixture
def agent_euclid(dim, G_euclid) -> EikonalAgent:
    return EikonalAgent(
        metric_tensor=G_euclid,
        l_max_absolute=50,
        kappa_coupling=1.0,
        eikonal_slack=0.1,
        kappa_max=1e12,
    )


@pytest.fixture
def agent_aniso(dim, G_aniso) -> EikonalAgent:
    return EikonalAgent(
        metric_tensor=G_aniso,
        l_max_absolute=40,
        kappa_coupling=0.8,
        eikonal_slack=0.15,
    )


@pytest.fixture
def pure_rho(dim) -> NDArray[np.float64]:
    return make_density(dim, pure=True, seed=42)


@pytest.fixture
def mixed_rho(dim) -> NDArray[np.float64]:
    return make_density(dim, pure=False, seed=43, rank=max(2, dim // 2 + 1))


def _make_control(
    d: int,
    *,
    G: NDArray[np.float64] | None = None,
    rho: NDArray[np.float64] | None = None,
    s_mac: float = 0.5,
    sigma: float = 0.3,
    phase_scale: float = 5.0,
    T: int = 5,
    use_geo: bool = False,
    cavity_tol: float = 1.0,
    seed: int = 0,
) -> EikonalControlInput:
    """
    Control feliz: ∇S grande ⇒ Eikonal holgado; path corto; geo off por defecto
    (stubs Levi-Civita con Γ=0 dan R_geo=0).
    """
    rng = np.random.default_rng(seed)
    if rho is None:
        rho = make_density(d, pure=True, seed=seed + 1)
    logits = rng.standard_normal(d)
    grad = phase_scale * rng.standard_normal(d)
    # Garantizar ‖∇S‖² grande respecto a n²≈4
    if G is not None:
        G_inv = la.inv(0.5 * (G + G.T))
        # Escalar grad hasta s_norm_sq >= 10
        sn = float(grad @ G_inv @ grad)
        if sn < 10.0:
            grad = grad * math.sqrt(10.0 / max(sn, 1e-15))
    else:
        if float(grad @ grad) < 10.0:
            grad = grad * math.sqrt(10.0 / max(float(grad @ grad), 1e-15))
    V = rng.standard_normal((T, d)) * 0.1
    return EikonalControlInput(
        raw_llm_logits=logits,
        rho_llm=rho,
        s_mac_entropy=s_mac,
        logistic_stress_norm=sigma,
        phase_gradient=grad,
        path_velocities=V,
        use_geodesic_correction=use_geo,
        cavity_tol=cavity_tol,
    )


# =============================================================================
# 1. JERARQUÍA DE EXCEPCIONES
# =============================================================================


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            QuantumPurityCollapseError,
            EikonalSingularityError,
            FermatOpticalDeviationError,
            DimensionalMismatchError,
            MetricSignatureError,
            EikonalParameterError,
            EikonalRefractionError,
        ],
    )
    def test_all_derive_from_topological_root(self, exc_cls):
        assert issubclass(exc_cls, _TIE_ROOT) or issubclass(
            exc_cls, TopologicalInvariantError
        )

    def test_catch_all_nonsquare_G(self):
        with pytest.raises(
            (TopologicalInvariantError, DimensionalMismatchError, MetricSignatureError)
        ):
            EikonalAgent(metric_tensor=np.ones((3, 2)))


# =============================================================================
# 2. FASE 1 — DIAFRAGMA DINÁMICO CUÁNTICO
# =============================================================================


class TestPhase1ConstructionAndParams:
    def test_rejects_l_max_lt_one(self, G_euclid):
        with pytest.raises(EikonalParameterError):
            EikonalAgent(G_euclid, l_max_absolute=0)

    def test_rejects_nonpositive_kappa(self, G_euclid):
        with pytest.raises(EikonalParameterError):
            EikonalAgent(G_euclid, kappa_coupling=0.0)

    def test_rejects_bad_slack(self, G_euclid):
        with pytest.raises(EikonalParameterError):
            EikonalAgent(G_euclid, eikonal_slack=1.0)
        with pytest.raises(EikonalParameterError):
            EikonalAgent(G_euclid, eikonal_slack=-0.1)

    def test_rejects_kappa_max_le_one(self, G_euclid):
        with pytest.raises(EikonalParameterError):
            EikonalAgent(G_euclid, kappa_max=1.0)

    def test_rejects_asymmetric_G(self):
        G = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(MetricSignatureError):
            EikonalAgent(metric_tensor=G)

    def test_rejects_indefinite_G(self):
        with pytest.raises(MetricSignatureError):
            EikonalAgent(metric_tensor=np.diag([1.0, -1.0]))


class TestPhase1DensityAudit:
    def test_pure_state_purity_one(self, agent_euclid, pure_rho, dim):
        audit = agent_euclid.phase1.audit_density_matrix(pure_rho)
        assert isinstance(audit, SpectralDensityAudit)
        assert audit.purity == pytest.approx(1.0, abs=1e-10)
        assert audit.von_neumann_entropy == pytest.approx(0.0, abs=1e-10)
        assert audit.is_physical
        assert audit.dim == dim
        assert audit.trace_after_pruning == pytest.approx(1.0, abs=1e-12)
        assert audit.purity == pytest.approx(purity_oracle(pure_rho), abs=1e-10)

    def test_maximally_mixed_purity(self, agent_euclid, dim):
        rho = np.eye(dim) / dim
        audit = agent_euclid.phase1.audit_density_matrix(rho)
        assert audit.purity == pytest.approx(1.0 / dim, rel=1e-10)
        assert audit.von_neumann_entropy == pytest.approx(math.log(dim), rel=1e-10)

    def test_mixed_state_purity_in_unit_interval(
        self, agent_euclid, mixed_rho
    ):
        audit = agent_euclid.phase1.audit_density_matrix(mixed_rho)
        assert 0.0 < audit.purity <= 1.0 + 1e-12
        assert audit.von_neumann_entropy >= -1e-12
        assert audit.purity == pytest.approx(purity_oracle(mixed_rho), rel=1e-8)

    def test_rejects_nonsquare_rho(self, agent_euclid):
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.phase1.audit_density_matrix(np.ones((3, 2)))

    def test_rejects_asymmetric_rho(self, agent_euclid, dim):
        if dim < 2:
            pytest.skip("d≥2")
        bad = np.eye(dim)
        bad[0, 1] = 0.5  # rompe simetría
        with pytest.raises(QuantumPurityCollapseError):
            agent_euclid.phase1.audit_density_matrix(bad)

    def test_prunes_negative_eigenvalues(self, agent_euclid, dim):
        """ρ con autovalor negativo se recorta; pureza se calcula post-recorte."""
        # Construir simétrica con un λ < 0 y renormalizar no es físico;
        # inyectamos y verificamos n_negative > 0 y pureza finita.
        rng = np.random.default_rng(9)
        Q, _ = la.qr(rng.standard_normal((dim, dim)))
        ev = np.linspace(-0.2, 1.0, dim)
        rho = Q @ np.diag(ev) @ Q.T
        rho = 0.5 * (rho + rho.T)
        audit = agent_euclid.phase1.audit_density_matrix(rho)
        assert audit.negative_eigvals_pruned >= 1
        assert audit.trace_after_pruning == pytest.approx(1.0, abs=1e-12)
        assert audit.purity > 0.0
        assert np.all(audit.eigenvalues_psd >= -1e-15)

    def test_zero_trace_after_clip_raises(self, agent_euclid, dim):
        """Todo negativo ⇒ traza 0 tras clip ⇒ colapso."""
        rho = -np.eye(dim)
        with pytest.raises(QuantumPurityCollapseError):
            agent_euclid.phase1.audit_density_matrix(rho)

    def test_s_vn_matches_oracle(self, agent_euclid, mixed_rho):
        audit = agent_euclid.phase1.audit_density_matrix(mixed_rho)
        assert audit.von_neumann_entropy == pytest.approx(
            von_neumann_oracle(mixed_rho), abs=1e-9
        )


class TestPhase1ApertureModulation:
    def test_modulate_returns_dto(self, agent_euclid, pure_rho):
        res = agent_euclid.phase1.modulate_aperture(0.5, pure_rho)
        assert isinstance(res, ApertureModulationResult)
        assert res.l_cutoff >= 1
        assert res.l_cutoff <= res.l_max
        assert 0.0 < res.attenuation_factor <= 1.0 + 1e-15
        assert res.spectral_certificate.purity == pytest.approx(1.0, abs=1e-10)

    def test_cutoff_formula_oracle(self, agent_euclid, pure_rho):
        s_mac = 1.2
        kappa = agent_euclid.phase1._kappa
        l_max = agent_euclid.phase1._l_max
        purity = 1.0  # puro
        att = math.exp(-(kappa * s_mac) / purity)
        expected = max(1, int(math.floor(l_max * att)))
        res = agent_euclid.phase1.modulate_aperture(s_mac, pure_rho)
        assert res.l_cutoff == expected
        assert res.attenuation_factor == pytest.approx(att, rel=1e-12)

    def test_zero_entropy_full_aperture(self, agent_euclid, pure_rho):
        """S_MAC=0 ⇒ attenuation=1 ⇒ l_cutoff = l_max."""
        res = agent_euclid.phase1.modulate_aperture(0.0, pure_rho)
        assert res.attenuation_factor == pytest.approx(1.0)
        assert res.l_cutoff == res.l_max

    def test_high_entropy_closes_diaphragm(self, agent_euclid, pure_rho):
        res_lo = agent_euclid.phase1.modulate_aperture(0.0, pure_rho)
        res_hi = agent_euclid.phase1.modulate_aperture(10.0, pure_rho)
        assert res_hi.l_cutoff <= res_lo.l_cutoff
        assert res_hi.attenuation_factor < res_lo.attenuation_factor

    def test_lower_purity_closes_more(self, agent_euclid, dim):
        """A menor pureza, mayor atenuación para el mismo S_MAC."""
        pure = make_density(dim, pure=True, seed=1)
        mixed = np.eye(dim) / dim
        s = 1.0
        r_pure = agent_euclid.phase1.modulate_aperture(s, pure)
        r_mixed = agent_euclid.phase1.modulate_aperture(s, mixed)
        assert r_mixed.attenuation_factor <= r_pure.attenuation_factor + 1e-15
        assert r_mixed.l_cutoff <= r_pure.l_cutoff

    def test_negative_s_mac_raises(self, agent_euclid, pure_rho):
        with pytest.raises(EikonalParameterError):
            agent_euclid.phase1.modulate_aperture(-0.1, pure_rho)

    def test_alias_compute_dynamic_cutoff(self, agent_euclid, pure_rho):
        a = agent_euclid.phase1.modulate_aperture(0.7, pure_rho).l_cutoff
        b = agent_euclid.phase1.compute_dynamic_cutoff(0.7, pure_rho)
        assert a == b

    def test_purity_collapse_on_numerical_zero(
        self, agent_euclid, dim, monkeypatch
    ):
        """Si pureza < EPS tras audit, colapso (inyección vía rho casi nula imposible
        si renormalizamos; usamos monkeypatch del audit)."""
        fake = SpectralDensityAudit(
            purity=1e-20,
            von_neumann_entropy=10.0,
            eigenvalues_psd=np.ones(dim) / dim,
            negative_eigvals_pruned=0,
            hermiticity_residual=0.0,
            trace_after_pruning=1.0,
            is_physical=False,
            dim=dim,
        )
        monkeypatch.setattr(
            agent_euclid.phase1,
            "audit_density_matrix",
            lambda rho: fake,
        )
        with pytest.raises(QuantumPurityCollapseError):
            agent_euclid.phase1.modulate_aperture(0.1, np.eye(dim) / dim)


# =============================================================================
# 3. FASE 2 — SUPERFICIE EIKONAL
# =============================================================================


class TestPhase2MetricInverse:
    def test_G_inv_is_true_inverse(self, agent_euclid, G_euclid, dim):
        G_inv = agent_euclid.phase2.G_inv
        I = np.eye(dim)
        assert fro_rel(G_euclid @ G_inv, I) < 1e-10
        assert fro_rel(G_inv @ G_euclid, I) < 1e-10

    def test_aniso_inverse_consistency(self, agent_aniso, G_aniso, dim):
        G_inv = agent_aniso.phase2.G_inv
        I = np.eye(dim)
        assert fro_rel(G_aniso @ G_inv, I) < 1e-8
        assert agent_aniso.phase2._inv_residual < 1e-8

    def test_kappa_identity(self, agent_aniso):
        """κ(G⁻¹) ≈ κ(G)."""
        rel = abs(
            agent_aniso.phase2._kappa_G_inv - agent_aniso.phase2._kappa_G
        ) / max(agent_aniso.phase2._kappa_G, 1.0)
        assert rel < 1e-3

    def test_ill_conditioned_G_raises(self, dim):
        G = make_spd(dim, cond=1e14, seed=5)
        with pytest.raises(EikonalSingularityError):
            EikonalAgent(metric_tensor=G, kappa_max=1e8)

    def test_metric_inverse_property_defensive(self, agent_euclid, dim):
        Gi = agent_euclid.metric_inverse
        Gi[0, 0] = -999.0
        assert agent_euclid.metric_inverse[0, 0] != -999.0


class TestPhase2EikonalEquation:
    def test_resolve_returns_dto(self, agent_euclid, dim):
        grad = 3.0 * np.ones(dim)
        n = 1.2
        res = agent_euclid.phase2.resolve_eikonal(grad, n)
        assert isinstance(res, EikonalSurfaceResult)
        expected = eikonal_norm_sq_oracle(grad, agent_euclid.phase2.G_inv)
        assert res.phase_norm_sq == pytest.approx(expected, rel=1e-12)
        assert res.n_refract == pytest.approx(n)
        assert res.n_sq == pytest.approx(n ** 2)
        assert res.dim == dim

    def test_euclid_norm_is_squared_l2(self, agent_euclid, dim):
        grad = np.arange(1, dim + 1, dtype=np.float64)
        res = agent_euclid.phase2.resolve_eikonal(grad, 0.5)
        assert res.phase_norm_sq == pytest.approx(float(grad @ grad), rel=1e-12)

    def test_wrong_grad_shape_raises(self, agent_euclid, dim):
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.phase2.resolve_eikonal(np.ones(dim + 1), 1.0)

    def test_negative_n_raises(self, agent_euclid, dim):
        with pytest.raises(EikonalParameterError):
            agent_euclid.phase2.resolve_eikonal(np.ones(dim), -0.1)

    def test_eikonal_refraction_failure(self, agent_euclid, dim):
        """∇S ≈ 0 no alcanza n²(1−slack)."""
        with pytest.raises(EikonalRefractionError):
            agent_euclid.phase2.resolve_eikonal(
                1e-12 * np.ones(dim), n_refract=1.5
            )

    def test_slack_threshold_formula(self, G_euclid, dim):
        slack = 0.2
        agent = EikonalAgent(G_euclid, eikonal_slack=slack)
        n = 2.0
        n_sq = n ** 2
        thr = n_sq * (1.0 - slack)
        # Justo por debajo del umbral
        # ‖∇S‖² = thr - eps
        target = thr * 0.99
        grad = np.zeros(dim)
        grad[0] = math.sqrt(target)
        with pytest.raises(EikonalRefractionError):
            agent.phase2.resolve_eikonal(grad, n)
        # Justo por encima
        grad[0] = math.sqrt(thr * 1.01)
        res = agent.phase2.resolve_eikonal(grad, n)
        assert res.phase_norm_sq >= res.eikonal_threshold

    def test_margin_sound_flag(self, agent_euclid, dim):
        # Gradiente muy grande ⇒ margin_sound True
        grad = 20.0 * np.ones(dim)
        res = agent_euclid.phase2.resolve_eikonal(grad, 1.0)
        assert res.margin_sound is True

    def test_alias_resolve_eikonal_equation(self, agent_euclid, dim):
        grad = 4.0 * np.ones(dim)
        a = agent_euclid.phase2.resolve_eikonal(grad, 1.1).phase_norm_sq
        b = agent_euclid.phase2.resolve_eikonal_equation(grad, 1.1)
        assert a == pytest.approx(b)

    def test_zero_gradient_with_zero_n(self, agent_euclid, dim):
        """n=0, ∇S=0: threshold=0, s_norm_sq=0 ⇒ no lanza (borde)."""
        res = agent_euclid.phase2.resolve_eikonal(np.zeros(dim), 0.0)
        assert res.phase_norm_sq == pytest.approx(0.0, abs=1e-15)
        assert res.eikonal_threshold == pytest.approx(0.0, abs=1e-15)


# =============================================================================
# 4. FASE 3 — FERMAT Y GEODÉSICA
# =============================================================================


class TestPhase3FermatIntegration:
    def test_simpson_matches_oracle(self, agent_euclid, dim, G_euclid):
        T, dt = 5, 1e-3  # impar
        V = _RNG.standard_normal((T, dim))
        n_ref = 1.3
        action, rule, used = agent_euclid.phase3.integrate_fermat(V, n_ref, dt=dt)
        norms = speed_norms_G(V, G_euclid)
        integ, rule_o = simpson_integral_oracle(norms, dt)
        assert rule == rule_o == "simpson"
        assert used == T
        assert action == pytest.approx(n_ref * integ, rel=1e-12)

    def test_even_T_truncates_to_simpson(self, agent_euclid, dim, G_euclid):
        T, dt = 6, 1e-3  # par → usa 5
        V = _RNG.standard_normal((T, dim))
        action, rule, used = agent_euclid.phase3.integrate_fermat(V, 1.0, dt=dt)
        assert rule == "simpson"
        assert used == T - 1
        norms = speed_norms_G(V, G_euclid)
        integ, _ = simpson_integral_oracle(norms, dt)
        assert action == pytest.approx(integ, rel=1e-12)

    def test_short_path_trapezoid(self, agent_euclid, dim):
        V = _RNG.standard_normal((2, dim))
        action, rule, used = agent_euclid.phase3.integrate_fermat(V, 1.0, dt=0.01)
        assert rule == "trapezoid"
        assert used == 2
        assert action >= 0.0

    def test_single_point(self, agent_euclid, dim):
        V = np.ones((1, dim))
        action, rule, used = agent_euclid.phase3.integrate_fermat(V, 2.0, dt=0.5)
        assert rule == "trapezoid"
        assert used == 1
        # n * ‖v‖_G * dt
        speed = math.sqrt(float(np.ones(dim) @ np.ones(dim)))  # G=I
        assert action == pytest.approx(2.0 * speed * 0.5, rel=1e-12)

    def test_empty_path_zero_action(self, agent_euclid, dim):
        V = np.zeros((0, dim))
        action, rule, used = agent_euclid.phase3.integrate_fermat(V, 1.0)
        assert action == 0.0
        assert rule == "empty"
        assert used == 0

    def test_action_scales_with_n(self, agent_euclid, dim):
        V = _RNG.standard_normal((5, dim))
        a1, _, _ = agent_euclid.phase3.integrate_fermat(V, 1.0, dt=1e-3)
        a2, _, _ = agent_euclid.phase3.integrate_fermat(V, 2.0, dt=1e-3)
        assert a2 == pytest.approx(2.0 * a1, rel=1e-12)

    def test_action_nonnegative_for_real_paths(self, agent_euclid, dim):
        V = _RNG.standard_normal((7, dim))
        action, _, _ = agent_euclid.phase3.integrate_fermat(V, 1.5, dt=1e-3)
        assert action >= -1e-14

    def test_wrong_shape_raises(self, agent_euclid, dim):
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.phase3.integrate_fermat(np.ones((3, dim + 1)), 1.0)

    def test_nonpositive_dt_raises(self, agent_euclid, dim):
        with pytest.raises(EikonalParameterError):
            agent_euclid.phase3.integrate_fermat(np.ones((3, dim)), 1.0, dt=0.0)

    def test_alias_audit_fermat_action(self, agent_euclid, dim):
        V = _RNG.standard_normal((5, dim))
        a = agent_euclid.phase3.audit_fermat_action(V, 1.2, dt=1e-3)
        b, _, _ = agent_euclid.phase3.integrate_fermat(V, 1.2, dt=1e-3)
        assert a == pytest.approx(b)


class TestPhase3Geodesic:
    def test_geodesic_stub_zero_deviation(self, agent_euclid, dim):
        """Con stub Γ=0 y flujo constante, R_geo = 0."""
        v0 = TangentVector(coordinates=np.ones(dim))
        path, dev = agent_euclid.phase3.enforce_geodesic_path(v0, n_steps=4, dt=1.0)
        assert path.shape == (5, dim)  # n_steps+1
        assert dev == pytest.approx(0.0, abs=1e-12)

    def test_n_steps_lt_one_raises(self, agent_euclid, dim):
        with pytest.raises(EikonalParameterError):
            agent_euclid.phase3.enforce_geodesic_path(
                TangentVector(coordinates=np.ones(dim)), n_steps=0
            )

    def test_audit_path_without_geo(self, agent_euclid, dim):
        V = _RNG.standard_normal((5, dim)) * 0.2
        res = agent_euclid.phase3.audit_path(
            V, 1.1, use_geodesic_correction=False, cavity_tol=1e-10
        )
        assert isinstance(res, FermatPathResult)
        assert res.geodesic_deviation == 0.0
        assert res.fermat_action >= 0.0
        assert res.n_refract == pytest.approx(1.1)

    def test_audit_path_with_geo_stub(self, agent_euclid, dim):
        V = _RNG.standard_normal((4, dim)) * 0.1
        res = agent_euclid.phase3.audit_path(
            V, 1.0, use_geodesic_correction=True, cavity_tol=1.0
        )
        assert res.geodesic_deviation == pytest.approx(0.0, abs=1e-12)
        assert res.path_velocities.shape[0] == 5  # n_steps+1 del enforce

    def test_geodesic_tol_violation(self, agent_euclid, dim, monkeypatch):
        """Si la desviación reportada > cavity_tol ⇒ FermatOpticalDeviationError."""

        def fake_enforce(initial_velocity, n_steps, dt=1.0):
            V = np.zeros((n_steps + 1, dim))
            return V, 1.0  # desviación grande

        monkeypatch.setattr(
            agent_euclid.phase3, "enforce_geodesic_path", fake_enforce
        )
        with pytest.raises(FermatOpticalDeviationError):
            agent_euclid.phase3.audit_path(
                np.ones((3, dim)),
                1.0,
                use_geodesic_correction=True,
                cavity_tol=1e-6,
            )


# =============================================================================
# 5. COMPOSICIÓN FUNTORIAL
# =============================================================================


class TestFunctorialComposition:
    def test_end_to_end_happy_path(self, agent_euclid, dim, G_euclid, pure_rho):
        control = _make_control(
            dim, G=G_euclid, rho=pure_rho, use_geo=False, seed=100
        )
        state = agent_euclid.execute_optical_guidance(control)
        assert isinstance(state, EikonalPhaseState)
        assert state.dynamic_l_cutoff >= 1
        assert state.fermat_action_integral >= 0.0
        assert state.phase_gradient_norm > 0.0
        assert state.spectral_certificate is not None
        assert state.spectral_certificate.purity == pytest.approx(1.0, abs=1e-10)
        assert state.eikonal_surface is not None
        assert state.eikonal_surface.margin_sound or True  # puede o no
        assert state.fermat_path is not None
        assert state.aperture is not None
        assert state.aperture.l_cutoff == state.dynamic_l_cutoff
        assert state.refracted_state is not None

    def test_manual_phases_match_pipeline(
        self, agent_euclid, dim, G_euclid, pure_rho
    ):
        control = _make_control(
            dim, G=G_euclid, rho=pure_rho, use_geo=False, seed=101
        )
        # Manual
        ap = agent_euclid.phase1.modulate_aperture(
            control.s_mac_entropy, control.rho_llm
        )
        n = float(
            agent_euclid._lens_fibrator._compute_fermat_refractive_index(
                control.logistic_stress_norm
            )
        )
        surf = agent_euclid.phase2.resolve_eikonal(control.phase_gradient, n)
        ferm = agent_euclid.phase3.audit_path(
            control.path_velocities,
            n,
            use_geodesic_correction=False,
            cavity_tol=control.cavity_tol,
        )
        # Pipeline
        state = agent_euclid.execute_optical_guidance(control)
        assert state.dynamic_l_cutoff == ap.l_cutoff
        assert state.phase_gradient_norm == pytest.approx(surf.phase_norm_sq)
        assert state.fermat_action_integral == pytest.approx(ferm.fermat_action)

    def test_deterministic(self, agent_euclid, dim, G_euclid, pure_rho):
        control = _make_control(dim, G=G_euclid, rho=pure_rho, seed=102)
        s1 = agent_euclid.execute_optical_guidance(control)
        s2 = agent_euclid.execute_optical_guidance(control)
        assert s1.phase_gradient_norm == s2.phase_gradient_norm
        assert s1.fermat_action_integral == s2.fermat_action_integral
        assert s1.dynamic_l_cutoff == s2.dynamic_l_cutoff

    def test_anisotropic_pipeline(self, agent_aniso, dim, G_aniso, pure_rho):
        control = _make_control(
            dim, G=G_aniso, rho=pure_rho, use_geo=False, seed=103
        )
        state = agent_aniso.execute_optical_guidance(control)
        assert state.eikonal_surface is not None
        assert state.eikonal_surface.inverse_residual < 1e-8
        assert state.spectral_certificate is not None

    def test_high_s_mac_reduces_cutoff(
        self, agent_euclid, dim, G_euclid, pure_rho
    ):
        c_lo = _make_control(
            dim, G=G_euclid, rho=pure_rho, s_mac=0.0, seed=104
        )
        c_hi = _make_control(
            dim, G=G_euclid, rho=pure_rho, s_mac=5.0, seed=104
        )
        # mismos tensores; solo s_mac cambia — reconstruir con mismos campos
        s_lo = agent_euclid.execute_optical_guidance(c_lo)
        s_hi = agent_euclid.execute_optical_guidance(c_hi)
        assert s_hi.dynamic_l_cutoff <= s_lo.dynamic_l_cutoff

    def test_eikonal_failure_propagates(
        self, agent_euclid, dim, G_euclid, pure_rho
    ):
        control = _make_control(dim, G=G_euclid, rho=pure_rho, seed=105)
        # Forzar gradiente nulo
        bad = EikonalControlInput(
            raw_llm_logits=control.raw_llm_logits,
            rho_llm=control.rho_llm,
            s_mac_entropy=control.s_mac_entropy,
            logistic_stress_norm=2.0,  # n > 1
            phase_gradient=np.zeros(dim),
            path_velocities=control.path_velocities,
            use_geodesic_correction=False,
        )
        with pytest.raises(EikonalRefractionError):
            agent_euclid.execute_optical_guidance(bad)


# =============================================================================
# 6. INTEGRACIÓN DEL AGENTE
# =============================================================================


class TestEikonalAgentIntegration:
    def test_validate_control_dimensions(self, agent_euclid, dim, pure_rho):
        good = _make_control(dim, rho=pure_rho, seed=110)
        # logits mal
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.execute_optical_guidance(
                EikonalControlInput(
                    raw_llm_logits=np.ones(dim + 1),
                    rho_llm=good.rho_llm,
                    s_mac_entropy=0.1,
                    logistic_stress_norm=0.1,
                    phase_gradient=good.phase_gradient,
                    path_velocities=good.path_velocities,
                    use_geodesic_correction=False,
                )
            )
        with pytest.raises(DimensionalMismatchError):
            agent_euclid.execute_optical_guidance(
                EikonalControlInput(
                    raw_llm_logits=good.raw_llm_logits,
                    rho_llm=np.eye(dim + 1),
                    s_mac_entropy=0.1,
                    logistic_stress_norm=0.1,
                    phase_gradient=good.phase_gradient,
                    path_velocities=good.path_velocities,
                    use_geodesic_correction=False,
                )
            )

    def test_negative_s_mac_in_control(self, agent_euclid, dim, pure_rho):
        c = _make_control(dim, rho=pure_rho, seed=111)
        with pytest.raises(EikonalParameterError):
            agent_euclid.execute_optical_guidance(
                EikonalControlInput(
                    raw_llm_logits=c.raw_llm_logits,
                    rho_llm=c.rho_llm,
                    s_mac_entropy=-1.0,
                    logistic_stress_norm=0.1,
                    phase_gradient=c.phase_gradient,
                    path_velocities=c.path_velocities,
                    use_geodesic_correction=False,
                )
            )

    def test_metric_tensor_defensive_copy(self, agent_euclid):
        G = agent_euclid.metric_tensor
        G[0, 0] = -999.0
        assert agent_euclid.metric_tensor[0, 0] != -999.0

    def test_dimension_property(self, agent_euclid, dim):
        assert agent_euclid.dimension == dim

    def test_phase_ports_exposed(self, agent_euclid):
        assert agent_euclid.phase1_modulator is agent_euclid.phase1
        assert agent_euclid.phase2_resolver is agent_euclid.phase2
        assert agent_euclid.phase3_auditor is agent_euclid.phase3

    def test_forward_backward_categorical(self, agent_euclid, dim):
        try:
            from app.core.mic_algebra import CategoricalState
        except ImportError:
            from eikonal_agent import CategoricalState  # type: ignore

        psi = np.ones(dim)
        st = CategoricalState(payload=psi, label="x")
        out = agent_euclid.forward(st)
        assert "eikonal_forward" in out.label
        np.testing.assert_allclose(out.payload, psi)
        np.testing.assert_allclose(
            agent_euclid.backward(st).payload, agent_euclid.forward(st).payload
        )

    def test_forward_wrong_shape(self, agent_euclid, dim):
        try:
            from app.core.mic_algebra import CategoricalState
        except ImportError:
            from eikonal_agent import CategoricalState  # type: ignore

        with pytest.raises(DimensionalMismatchError):
            agent_euclid.forward(
                CategoricalState(payload=np.ones(dim + 2), label="bad")
            )

    def test_state_is_frozen(self, agent_euclid, dim, G_euclid, pure_rho):
        state = agent_euclid.execute_optical_guidance(
            _make_control(dim, G=G_euclid, rho=pure_rho, seed=112)
        )
        with pytest.raises(Exception):
            state.dynamic_l_cutoff = 0  # type: ignore[misc]

    def test_lens_receives_cutoff(self, agent_euclid, dim, G_euclid, pure_rho):
        control = _make_control(
            dim, G=G_euclid, rho=pure_rho, s_mac=0.0, seed=113
        )
        state = agent_euclid.execute_optical_guidance(control)
        assert agent_euclid._lens_fibrator._l_cutoff == state.dynamic_l_cutoff


# =============================================================================
# 7. ESTRÉS NUMÉRICO E INVARIANTES
# =============================================================================


class TestNumericalStressAndInvariants:
    def test_d1_full_pipeline(self):
        G = np.array([[2.0]])
        agent = EikonalAgent(metric_tensor=G, eikonal_slack=0.2)
        rho = np.array([[1.0]])
        control = EikonalControlInput(
            raw_llm_logits=np.array([0.5]),
            rho_llm=rho,
            s_mac_entropy=0.2,
            logistic_stress_norm=0.1,
            phase_gradient=np.array([5.0]),
            path_velocities=np.array([[0.1], [0.2], [0.1]]),
            use_geodesic_correction=False,
        )
        state = agent.execute_optical_guidance(control)
        assert state.dynamic_l_cutoff >= 1
        assert state.phase_gradient_norm > 0.0

    def test_mixed_rho_pipeline(self, dim, G_euclid):
        agent = EikonalAgent(G_euclid)
        rho = make_density(dim, pure=False, seed=200)
        control = _make_control(dim, G=G_euclid, rho=rho, seed=201)
        state = agent.execute_optical_guidance(control)
        assert 0.0 < state.spectral_certificate.purity <= 1.0 + 1e-12
        assert state.spectral_certificate.von_neumann_entropy >= -1e-12

    def test_moderate_high_condition(self, dim):
        G = make_spd(dim, cond=1e5, seed=210)
        agent = EikonalAgent(G, kappa_max=1e10, eikonal_slack=0.2)
        rho = make_density(dim, pure=True, seed=211)
        control = _make_control(dim, G=G, rho=rho, phase_scale=20.0, seed=212)
        state = agent.execute_optical_guidance(control)
        assert state.eikonal_surface is not None
        assert state.eikonal_surface.kappa_G < 1e10

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_eikonal_norm_oracle_ensemble(self, seed: int):
        d = 4
        G = make_spd(d, cond=5.0, seed=seed)
        agent = EikonalAgent(G, eikonal_slack=0.25)
        grad = 10.0 * np.random.default_rng(seed + 50).standard_normal(d)
        n = 1.0 + 0.1 * seed
        res = agent.phase2.resolve_eikonal(grad, n)
        expected = eikonal_norm_sq_oracle(grad, agent.phase2.G_inv)
        assert res.phase_norm_sq == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("seed", [10, 11, 12, 13])
    def test_fermat_simpson_ensemble(self, seed: int):
        d = 3
        G = np.eye(d)
        agent = EikonalAgent(G)
        V = np.random.default_rng(seed).standard_normal((7, d))
        n_ref = 1.4
        dt = 1e-3
        action, rule, used = agent.phase3.integrate_fermat(V, n_ref, dt=dt)
        norms = speed_norms_G(V, G)
        integ, rule_o = simpson_integral_oracle(norms, dt)
        assert rule == rule_o
        assert action == pytest.approx(n_ref * integ, rel=1e-11)

    @pytest.mark.parametrize("d", [2, 3, 5, 6])
    def test_pipeline_across_dimensions(self, d: int):
        G = make_spd(d, cond=3.0, seed=d + 300)
        agent = EikonalAgent(G, l_max_absolute=30, eikonal_slack=0.15)
        rho = make_density(d, pure=True, seed=d + 301)
        control = _make_control(d, G=G, rho=rho, T=5, seed=d + 302)
        state = agent.execute_optical_guidance(control)
        assert state.aperture is not None
        assert state.aperture.spectral_certificate.dim == d
        assert state.eikonal_surface is not None
        assert state.eikonal_surface.dim == d
        assert 1 <= state.dynamic_l_cutoff <= 30

    def test_cutoff_monotone_in_s_mac(self, agent_euclid, pure_rho):
        cutoffs = [
            agent_euclid.phase1.modulate_aperture(s, pure_rho).l_cutoff
            for s in (0.0, 0.5, 1.0, 2.0, 5.0)
        ]
        assert cutoffs == sorted(cutoffs, reverse=True)

    def test_purity_bounds_pure_and_mixed(self, agent_euclid, dim):
        pure = make_density(dim, pure=True, seed=400)
        mixed = np.eye(dim) / dim
        assert agent_euclid.phase1.audit_density_matrix(pure).purity == pytest.approx(
            1.0, abs=1e-12
        )
        assert agent_euclid.phase1.audit_density_matrix(mixed).purity == pytest.approx(
            1.0 / dim, rel=1e-12
        )

    def test_bilateral_inverse_on_aniso(self, dim):
        G = make_spd(dim, cond=20.0, seed=410)
        agent = EikonalAgent(G)
        G_inv = agent.metric_inverse
        I = np.eye(dim)
        r = max(fro_rel(G @ G_inv, I), fro_rel(G_inv @ G, I))
        assert r < 1e-9


class TestMachineEpsConsistency:
    def test_machine_eps(self):
        assert _MACHINE_EPS == float(np.finfo(np.float64).eps)

    def test_default_slack_in_unit_interval(self):
        assert 0.0 <= _DEFAULT_EIKONAL_SLACK < 1.0

    def test_default_kappa_max_gt_one(self):
        assert _DEFAULT_KAPPA_MAX > 1.0


# =============================================================================
# Entrada directa
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))