# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Ruta     : tests/unit/boole/omega/test_generative_optic_hodge_suturator.py   ║
║ Módulo   : tests.unit.boole.omega.test_generative_optic_hodge_suturator      ║
║ Propósito: Batería de pruebas unitarias granulares y rigurosas para          ║
║            generative_optic_hodge_suturator.py                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ Cobertura principal:                                                         ║
║   • Regularización métrica Tikhonov / Cholesky espectral.                    ║
║   • Proyección Householder-Grassmann idempotente y G-autoadjunta.            ║
║   • Transporte Eikonal-Floquet con disipación geodésica controlada.          ║
║   • Lente Riemann-Topos: curvatura, gap de Hodge, entropía y coherencia.     ║
║   • Cuadratura esférica vectorizada de armónicos Y_l^m.                      ║
║   • Pipeline tricapa extremo a extremo con certificados y pasividad.         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.linalg as la


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIÓN ROBUSTA DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
_MODULE_CANDIDATES = (
    "app.omega.generative_optic_hodge_suturator",
    "app.boole.omega.generative_optic_hodge_suturator",
    "generative_optic_hodge_suturator",
)

sut = None
for candidate in _MODULE_CANDIDATES:
    try:
        sut = importlib.import_module(candidate)
        break
    except Exception:
        sut = None

if sut is None:
    pytest.skip(
        "No se pudo importar generative_optic_hodge_suturator desde ninguno de "
        f"{_MODULE_CANDIDATES}.",
        allow_module_level=True,
    )

_REQUIRED_SYMBOLS = (
    "StableRiemannianInverter",
    "MetricTensorBundle",
    "SemanticParabolicMirror",
    "EikonalFloquetAgentSutured",
    "OpticalRiemannLens",
    "GenerativeOpticHodgeSuturator",
    "SphericalHarmonicsVectorizer",
    "GeodesicEnergyConserver",
    "MetricSignatureError",
    "EikonalRefractionError",
    "FloquetInstabilityError",
    "GrassmannRankDeficiency",
    "SphericalQuadratureError",
    "Stratum",
)

_missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(sut, name)]
if _missing:
    pytest.fail(
        "El módulo importado no expone la API requerida para los tests: "
        f"{', '.join(_missing)}"
    )

StableRiemannianInverter = sut.StableRiemannianInverter
MetricTensorBundle = sut.MetricTensorBundle
SemanticParabolicMirror = sut.SemanticParabolicMirror
EikonalFloquetAgentSutured = sut.EikonalFloquetAgentSutured
OpticalRiemannLens = sut.OpticalRiemannLens
GenerativeOpticHodgeSuturator = sut.GenerativeOpticHodgeSuturator
SphericalHarmonicsVectorizer = sut.SphericalHarmonicsVectorizer
GeodesicEnergyConserver = sut.GeodesicEnergyConserver

MetricSignatureError = sut.MetricSignatureError
EikonalRefractionError = sut.EikonalRefractionError
FloquetInstabilityError = sut.FloquetInstabilityError
GrassmannRankDeficiency = sut.GrassmannRankDeficiency
SphericalQuadratureError = sut.SphericalQuadratureError
Stratum = sut.Stratum


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS Y FIXTURES DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS = float(np.finfo(np.float64).eps)


@pytest.fixture()
def rng() -> np.random.Generator:
    """Generador pseudoaleatorio determinista para pruebas reproducibles."""
    return np.random.default_rng(2026)


def _make_spd_metric(
    dim: int,
    condition: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Construye una métrica SPD con número de condición aproximado controlado.
    """
    rng = rng or np.random.default_rng()
    dim = int(max(1, dim))
    condition = float(max(1.0, condition))

    eigenvalues = np.linspace(1.0, condition, dim)
    Q, _ = la.qr(rng.standard_normal((dim, dim)))
    G = (Q * eigenvalues) @ Q.T
    G = (G + G.T) / 2.0
    return G.astype(np.float64)


def _make_ill_conditioned_spd_metric(dim: int = 8) -> np.ndarray:
    """
    Métrica diagonal SPD severamente mal condicionada para pruebas de Tikhonov.
    """
    eig = np.logspace(-8.0, 4.0, dim)
    return np.diag(eig).astype(np.float64)


def _make_semantic_direction(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Dirección semántica real unitaria."""
    v = rng.standard_normal(dim)
    norm = float(la.norm(v))
    if norm <= _MACHINE_EPS:
        v = np.ones(dim)
        norm = float(la.norm(v))
    return (v / norm).astype(np.float64)


def _make_complex_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Estado complejo aleatorio no nulo."""
    state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    norm = float(la.norm(state))
    if norm <= _MACHINE_EPS:
        state = np.ones(dim, dtype=np.complex128)
        norm = float(la.norm(state))
    return state.astype(np.complex128)


def _make_eikonal_gradient(
    metric_bundle: MetricTensorBundle,
    refractive_index: float,
    rng: np.random.Generator,
    factor: float = 2.0,
) -> np.ndarray:
    """
    Construye ∂S que satisface holgadamente la ecuación eikonal:
        ∂Sᵀ G^{-1} ∂S ≥ n²(1-slack).
    """
    dim = metric_bundle.dimension
    ds = rng.standard_normal(dim).astype(np.float64)

    lhs = float(np.real(np.vdot(ds, metric_bundle.G_inv @ ds)))
    required = (float(refractive_index) ** 2) * 0.9 * float(factor)

    if lhs <= _MACHINE_EPS:
        ds = np.ones(dim, dtype=np.float64)
        lhs = float(np.real(np.vdot(ds, metric_bundle.G_inv @ ds)))

    if lhs <= _MACHINE_EPS:
        raise ValueError("No se pudo construir gradiente eikonal no degenerado")

    ds *= float(np.sqrt(required / lhs))
    return ds


def _make_phase1_bundle(
    rng: np.random.Generator,
    dim: int = 6,
    grassmann_k: int = 3,
    condition: float = 5.0,
    refractive_index: float = 1.5,
):
    """
    Construye un Phase1ProjectionBundle válido para pruebas de Fase 2.
    """
    G = _make_spd_metric(dim=dim, condition=condition, rng=rng)
    semantic = _make_semantic_direction(dim=dim, rng=rng)
    raw_state = _make_complex_state(dim=dim, rng=rng)

    mirror = SemanticParabolicMirror(G)
    bundle = mirror.project_and_prepare_transport_bundle(
        semantic_direction=semantic,
        raw_state_vector=raw_state,
        grassmann_dimension=grassmann_k,
    )

    ds = _make_eikonal_gradient(
        metric_bundle=bundle.metric_bundle,
        refractive_index=refractive_index,
        rng=rng,
        factor=2.0,
    )

    return mirror, bundle, ds


def _make_suturator(
    cavity_tolerance: float = 1e-8,
    max_curvature: float = 2.0,
    n_theta: int = 8,
    n_phi: int = 8,
    max_l: int = 3,
) -> GenerativeOpticHodgeSuturator:
    """
    Orquestador tricapa con cuadratura esférica pequeña para tests rápidos.
    """
    return GenerativeOpticHodgeSuturator(
        target_stratum=Stratum.WISDOM,
        cavity_tolerance=cavity_tolerance,
        max_curvature=max_curvature,
        max_condition_number=1e8,
        quadrature_theta_nodes=n_theta,
        quadrature_phi_nodes=n_phi,
        max_spherical_l=max_l,
    )


# ══════════════════════════════════════════════════════════════════════════════
# [SUTURA 1] REGULARIZACIÓN MÉTRICA ESTABLE
# ══════════════════════════════════════════════════════════════════════════════
class TestStableRiemannianInverter:
    """Pruebas granulares de regularización Tikhonov/espectral."""

    def test_spd_metric_returns_consistent_inverse_and_square_roots(self, rng):
        dim = 6
        G = _make_spd_metric(dim=dim, condition=5.0, rng=rng)

        bundle = StableRiemannianInverter.regularize_spd_metric(
            G,
            max_condition_number=1e8,
        )

        I = np.eye(dim, dtype=np.float64)

        assert bundle.dimension == dim
        assert bundle.tikhonov_report.regularization_applied is False
        assert bundle.tikhonov_report.regularized_condition_number <= 10.0

        assert la.norm(bundle.G_reg @ bundle.G_inv - I, ord="fro") < 1e-8
        assert la.norm(bundle.G_sqrt @ bundle.G_sqrt - bundle.G_reg, ord="fro") < 1e-8
        assert la.norm(bundle.G_inv_sqrt @ bundle.G_sqrt - I, ord="fro") < 1e-8

    def test_ill_conditioned_metric_is_regularized_to_bound(self):
        G = _make_ill_conditioned_spd_metric(dim=8)
        max_condition_number = 1e6

        bundle = StableRiemannianInverter.regularize_spd_metric(
            G,
            max_condition_number=max_condition_number,
        )

        report = bundle.tikhonov_report

        assert report.regularization_applied is True
        assert report.original_condition_number > max_condition_number
        assert report.regularized_condition_number <= max_condition_number * 1.01 + 1e-8
        assert report.spectral_shift >= 0.0

        I = np.eye(8, dtype=np.float64)
        assert la.norm(bundle.G_reg @ bundle.G_inv - I, ord="fro") < 1e-5

    def test_non_square_metric_raises(self):
        G = np.ones((3, 4), dtype=np.float64)

        with pytest.raises(MetricSignatureError):
            StableRiemannianInverter.regularize_spd_metric(G)

    def test_indefinite_metric_raises(self):
        G = np.diag([-2.0, 1.0, 3.0]).astype(np.float64)

        with pytest.raises(MetricSignatureError):
            StableRiemannianInverter.regularize_spd_metric(G)

    def test_zero_metric_raises(self):
        G = np.zeros((4, 4), dtype=np.float64)

        with pytest.raises(MetricSignatureError):
            StableRiemannianInverter.regularize_spd_metric(G)

    def test_stable_riemannian_inverse_on_well_conditioned_metric(self, rng):
        dim = 5
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)

        G_inv, report = StableRiemannianInverter.stable_riemannian_inverse(G)

        I = np.eye(dim, dtype=np.float64)
        assert report.regularization_applied is False
        assert la.norm(G @ G_inv - I, ord="fro") < 1e-8

    def test_metric_bundle_inner_product_is_positive(self, rng):
        dim = 4
        G = _make_spd_metric(dim=dim, condition=3.0, rng=rng)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        norm_g = bundle.norm(v)
        inner_self = bundle.inner(v, v)

        assert norm_g >= 0.0
        assert inner_self >= 0.0
        assert np.isclose(inner_self, norm_g**2, rtol=1e-10, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PROYECCIÓN HOUSEHOLDER-GRASSMANN
# ══════════════════════════════════════════════════════════════════════════════
class TestSemanticParabolicMirror:
    """Pruebas de proyectores idempotentes sobre Gr(k,n)."""

    def test_full_dimension_projector_is_identity(self, rng):
        dim = 6
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        cert = mirror.compute_householder_projector(
            semantic_direction=semantic,
            target_dimension=dim,
        )

        assert cert.grassmann_dimension == dim
        assert np.allclose(cert.projector_matrix, np.eye(dim), atol=1e-10)
        assert cert.idempotence_residual <= 1e-10

    def test_zero_dimension_projector_is_zero(self, rng):
        dim = 5
        G = _make_spd_metric(dim=dim, condition=3.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        cert = mirror.compute_householder_projector(
            semantic_direction=semantic,
            target_dimension=0,
        )

        assert cert.grassmann_dimension == 0
        assert np.allclose(cert.projector_matrix, np.zeros((dim, dim)), atol=1e-10)
        assert cert.idempotence_residual <= 1e-10

    def test_projector_is_idempotent_and_g_self_adjoint(self, rng):
        dim = 8
        k = 4
        G = _make_spd_metric(dim=dim, condition=5.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        cert = mirror.compute_householder_projector(
            semantic_direction=semantic,
            target_dimension=k,
        )

        P = cert.projector_matrix
        G_reg = mirror.metric_bundle.G_reg

        idempotence_residual = la.norm(P @ P - P, ord="fro")
        g_self_adjoint_residual = la.norm(G_reg @ P - P.T @ G_reg, ord="fro")
        g_self_adjoint_residual /= max(1.0, la.norm(G_reg, ord="fro"))

        assert idempotence_residual <= 1e-7
        assert g_self_adjoint_residual <= 1e-7
        assert cert.idempotence_residual <= 1e-7

    def test_projector_preserves_semantic_direction(self, rng):
        dim = 7
        k = 3
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        cert = mirror.compute_householder_projector(
            semantic_direction=semantic,
            target_dimension=k,
        )

        P = cert.projector_matrix
        preserved_residual = la.norm(P @ semantic - semantic) / la.norm(semantic)

        assert preserved_residual <= 1e-7

    def test_projector_has_expected_spectral_rank(self, rng):
        dim = 8
        k = 5
        G = _make_spd_metric(dim=dim, condition=3.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        cert = mirror.compute_householder_projector(
            semantic_direction=semantic,
            target_dimension=k,
        )

        eigvals = la.eigvals(cert.projector_matrix)
        numerical_rank = int(np.sum(np.abs(eigvals) > 0.5))

        assert numerical_rank == k

    @pytest.mark.parametrize("invalid_k", [-1, 9])
    def test_invalid_grassmann_dimension_raises(self, invalid_k):
        dim = 8
        G = np.eye(dim, dtype=np.float64)
        semantic = np.ones(dim, dtype=np.float64)
        semantic /= la.norm(semantic)

        mirror = SemanticParabolicMirror(G)

        with pytest.raises(GrassmannRankDeficiency):
            mirror.compute_householder_projector(
                semantic_direction=semantic,
                target_dimension=invalid_k,
            )

    def test_project_and_prepare_transport_bundle_is_consistent(self, rng):
        dim = 6
        k = 3
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)
        semantic = _make_semantic_direction(dim=dim, rng=rng)
        raw_state = _make_complex_state(dim=dim, rng=rng)

        mirror = SemanticParabolicMirror(G)
        bundle = mirror.project_and_prepare_transport_bundle(
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=k,
        )

        P = bundle.certificate.projector_matrix
        expected_projected = P @ bundle.raw_state_vector

        assert bundle.state_projected.shape == raw_state.shape
        assert bundle.raw_state_vector.dtype == np.complex128
        assert bundle.grassmann_dimension == k
        assert np.allclose(bundle.state_projected, expected_projected, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: TRANSPORTE EIKONAL-FLOQUET
# ══════════════════════════════════════════════════════════════════════════════
class TestEikonalFloquetAgentSutured:
    """Pruebas de Fase 2 con sutura Cholesky/Tikhonov y disipación geodésica."""

    def _make_agent(self, bundle, refractive_index: float = 1.5):
        return EikonalFloquetAgentSutured(
            metric_bundle=bundle.metric_bundle,
            refractive_index_n=refractive_index,
            cavity_tolerance=1e-10,
        )

    def test_eikonal_condition_is_satisfied_for_scaled_gradient(self, rng):
        _, bundle, ds = _make_phase1_bundle(rng=rng, dim=6, grassmann_k=3)
        agent = self._make_agent(bundle, refractive_index=1.5)

        lhs, rhs, ok = agent.validate_eikonal_condition_sutured(ds)

        assert ok is True
        assert lhs >= rhs
        assert np.isfinite(lhs)
        assert np.isfinite(rhs)

    def test_eikonal_condition_rejects_small_gradient(self, rng):
        _, bundle, ds = _make_phase1_bundle(rng=rng, dim=6, grassmann_k=3)
        agent = self._make_agent(bundle, refractive_index=1.5)

        small_ds = ds * 1e-8
        lhs, rhs, ok = agent.validate_eikonal_condition_sutured(small_ds)

        assert ok is False
        assert lhs < rhs

    def test_execute_transport_phase_raises_for_eikonal_violation(self, rng):
        _, bundle, ds = _make_phase1_bundle(rng=rng, dim=6, grassmann_k=3)
        agent = self._make_agent(bundle, refractive_index=1.5)

        M = 0.5 * np.eye(bundle.metric_bundle.dimension)

        with pytest.raises(EikonalRefractionError):
            agent.execute_transport_phase(
                phase1_bundle=bundle,
                phase_gradient_ds=ds * 1e-10,
                monodromy_matrix_m=M,
                period_t=1.0,
            )

    def test_floquet_stability_accepts_contractive_monodromy(self, rng):
        _, bundle, _ = _make_phase1_bundle(rng=rng, dim=5, grassmann_k=2)
        agent = self._make_agent(bundle, refractive_index=1.5)

        M = 0.5 * np.eye(5)
        multipliers, max_modulus, lyapunov = agent.analyze_floquet_stability(
            M,
            period_t=1.0,
        )

        assert max_modulus <= 1.0 + 1e-10
        assert multipliers.shape == (5,)
        assert lyapunov.shape == (5,)
        assert np.all(np.isfinite(lyapunov))

    def test_floquet_instability_raises(self, rng):
        _, bundle, _ = _make_phase1_bundle(rng=rng, dim=4, grassmann_k=2)
        agent = self._make_agent(bundle, refractive_index=1.5)

        M = 1.2 * np.eye(4)

        with pytest.raises(FloquetInstabilityError):
            agent.analyze_floquet_stability(M, period_t=1.0)

    def test_nonpositive_period_raises(self, rng):
        _, bundle, _ = _make_phase1_bundle(rng=rng, dim=4, grassmann_k=2)
        agent = self._make_agent(bundle, refractive_index=1.5)

        M = 0.5 * np.eye(4)

        with pytest.raises(ValueError):
            agent.analyze_floquet_stability(M, period_t=0.0)

    def test_execute_transport_phase_enforces_dissipative_bound(self, rng):
        _, bundle, ds = _make_phase1_bundle(rng=rng, dim=6, grassmann_k=3)
        agent = self._make_agent(bundle, refractive_index=1.5)

        M = 0.5 * np.eye(6)
        result = agent.execute_transport_phase(
            phase1_bundle=bundle,
            phase_gradient_ds=ds,
            monodromy_matrix_m=M,
            period_t=1.0,
        )

        raw_norm_2 = float(la.norm(bundle.raw_state_vector))
        transported_norm_2 = float(la.norm(result.state_transported))

        raw_norm_g = bundle.metric_bundle.norm(bundle.raw_state_vector)
        transported_norm_g = bundle.metric_bundle.norm(result.state_transported)

        assert result.certificate.is_causally_admissible is True
        assert result.certificate.max_floquet_modulus <= 1.0 + 1e-8
        assert transported_norm_2 <= raw_norm_2 + 1e-10
        assert transported_norm_g <= raw_norm_g + 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# [SUTURA 2] CONSERVACIÓN GEODÉSICA DE ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════
class TestGeodesicEnergyConserver:
    """Pruebas directas del preservador geodésico."""

    def test_riemannian_energy_is_nonnegative(self, rng):
        dim = 5
        G = _make_spd_metric(dim=dim, condition=3.0, rng=rng)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        energy = GeodesicEnergyConserver.compute_riemannian_energy(v, bundle.G_reg)

        assert energy >= 0.0
        assert np.isfinite(energy)

    def test_geodesic_normalization_reaches_target_norm(self, rng):
        dim = 6
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        target_norm = 2.0

        v_norm, report = GeodesicEnergyConserver.enforce_geodesic_normalization(
            v,
            bundle.G_reg,
            target_norm,
        )

        current_norm = bundle.norm(v_norm)

        assert abs(current_norm - target_norm) <= 1e-8
        assert report.normalization_applied in (True, False)
        assert np.isfinite(report.energy_drift)

    def test_dissipative_bound_never_amplifies(self, rng):
        dim = 5
        G = _make_spd_metric(dim=dim, condition=3.0, rng=rng)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        raw = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        raw /= max(la.norm(raw), _MACHINE_EPS)

        v = raw * 5.0

        v_out, report = GeodesicEnergyConserver.enforce_dissipative_bound(
            v,
            raw,
            bundle.G_reg,
        )

        assert la.norm(v_out) <= la.norm(raw) + 1e-10
        assert bundle.norm(v_out) <= bundle.norm(raw) + 1e-10
        assert report.normalization_applied is True

    def test_zero_raw_state_forces_zero_output(self, rng):
        dim = 4
        G = np.eye(dim, dtype=np.float64)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        raw = np.zeros(dim, dtype=np.complex128)
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        v_out, _ = GeodesicEnergyConserver.enforce_dissipative_bound(
            v,
            raw,
            bundle.G_reg,
        )

        assert la.norm(v_out) <= 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: LENTE RIEMANN-TOPOS
# ══════════════════════════════════════════════════════════════════════════════
class TestOpticalRiemannLens:
    """Pruebas de métricas categóricas y compresión disipativa."""

    def _make_lens(self, dim: int = 6, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        G = _make_spd_metric(dim=dim, condition=4.0, rng=rng)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        lens = OpticalRiemannLens(
            metric_bundle=bundle,
            max_curvature_k=2.0,
            n_theta=8,
            n_phi=8,
            max_l=3,
        )

        return lens, bundle

    def test_riemann_curvature_norm_is_finite_and_nonnegative(self, rng):
        lens, _ = self._make_lens(dim=6, rng=rng)
        curvature = lens.compute_riemann_curvature_norm()

        assert np.isfinite(curvature)
        assert curvature >= 0.0

    def test_hodge_laplacian_gap_is_nonnegative(self, rng):
        lens, _ = self._make_lens(dim=5, rng=rng)
        gap = lens.compute_hodge_laplacian_gap()

        assert np.isfinite(gap)
        assert gap >= 0.0

    def test_hodge_gap_positive_for_non_degenerate_spectrum(self):
        G = np.diag([1.0, 2.0, 3.0]).astype(np.float64)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        lens = OpticalRiemannLens(
            metric_bundle=bundle,
            max_curvature_k=2.0,
        )

        gap = lens.compute_hodge_laplacian_gap()
        assert gap > 0.0

    def test_von_neumann_entropy_of_pure_state_is_zero(self, rng):
        dim = 5
        lens, _ = self._make_lens(dim=dim, rng=rng)

        psi = _make_complex_state(dim=dim, rng=rng)
        psi /= la.norm(psi)

        rho = np.outer(psi, psi.conj())
        entropy = lens.compute_von_neumann_entropy(rho)

        assert abs(entropy) <= 1e-10

    def test_von_neumann_entropy_of_maximally_mixed_state(self, rng):
        dim = 6
        lens, _ = self._make_lens(dim=dim, rng=rng)

        rho = np.eye(dim, dtype=np.complex128) / float(dim)
        entropy = lens.compute_von_neumann_entropy(rho)

        assert abs(entropy - np.log(dim)) <= 1e-10

    def test_topos_coherence_extremes(self):
        dim = 8
        G = np.eye(dim, dtype=np.float64)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        lens = OpticalRiemannLens(metric_bundle=bundle)

        basis_state = np.zeros(dim, dtype=np.complex128)
        basis_state[0] = 1.0

        uniform_state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

        coherence_basis = lens.compute_topos_coherence_index(basis_state)
        coherence_uniform = lens.compute_topos_coherence_index(uniform_state)

        assert abs(coherence_basis - 0.0) <= 1e-12
        assert abs(coherence_uniform - 1.0) <= 1e-12

    def test_compression_ratio_is_passive_for_attenuated_state(self, rng):
        dim = 6
        lens, _ = self._make_lens(dim=dim, rng=rng)

        raw = _make_complex_state(dim=dim, rng=rng)
        focused = 0.5 * raw

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
        )

        assert 0.0 < result.certificate.compression_ratio <= 1.0
        assert la.norm(result.state_focused) <= la.norm(raw) + 1e-10

    def test_amplified_focused_state_is_scaled_down(self, rng):
        dim = 6
        lens, _ = self._make_lens(dim=dim, rng=rng)

        raw = _make_complex_state(dim=dim, rng=rng)
        focused = 3.0 * raw

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
        )

        assert result.certificate.compression_ratio <= 1.0
        assert la.norm(result.state_focused) <= la.norm(raw) + 1e-10

    def test_zero_raw_state_returns_unit_ratio(self, rng):
        dim = 4
        lens, _ = self._make_lens(dim=dim, rng=rng)

        raw = np.zeros(dim, dtype=np.complex128)
        focused = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
        )

        assert result.certificate.compression_ratio == pytest.approx(1.0, abs=1e-12)
        assert la.norm(result.state_focused) <= 1e-12

    def test_focused_zero_state_receives_numeric_floor(self, rng):
        dim = 5
        lens, _ = self._make_lens(dim=dim, rng=rng)

        raw = _make_complex_state(dim=dim, rng=rng)
        focused = np.zeros(dim, dtype=np.complex128)

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
        )

        assert result.certificate.compression_ratio > 0.0
        assert result.certificate.compression_ratio <= 1.0

    def test_spherical_compression_with_compatible_field(self):
        dim = 16
        G = np.eye(dim, dtype=np.float64)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        lens = OpticalRiemannLens(
            metric_bundle=bundle,
            max_curvature_k=2.0,
            n_theta=4,
            n_phi=4,
            max_l=2,
        )

        raw = np.ones(dim, dtype=np.complex128) / 4.0
        focused = 0.8 * raw
        spherical_field = np.ones((4, 4), dtype=np.complex128)

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
            spherical_field=spherical_field,
            spherical_l_cut=1,
        )

        assert result.coefficients is not None
        assert result.certificate.spherical_degree_used == 1
        assert result.certificate.compression_ratio <= 1.0
        assert result.state_focused.size == dim

    def test_spherical_mismatch_falls_back_gracefully(self):
        dim = 16
        G = np.eye(dim, dtype=np.float64)
        bundle = StableRiemannianInverter.regularize_spd_metric(G)

        lens = OpticalRiemannLens(
            metric_bundle=bundle,
            max_curvature_k=2.0,
            n_theta=4,
            n_phi=4,
            max_l=2,
        )

        raw = np.ones(dim, dtype=np.complex128) / 4.0
        focused = 0.8 * raw
        incompatible_field = np.ones((3, 3), dtype=np.complex128)

        result = lens.apply_dissipative_compression(
            focused_state=focused,
            raw_state=raw,
            spherical_field=incompatible_field,
            spherical_l_cut=1,
        )

        assert result.coefficients is None
        assert result.certificate.compression_ratio <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# [SUTURA 3] ARMÓNICOS ESFÉRICOS VECTORIZADOS
# ══════════════════════════════════════════════════════════════════════════════
class TestSphericalHarmonicsVectorizer:
    """Pruebas de cuadratura Gauss-Legendre y armónicos esféricos."""

    def test_quadrature_cache_shapes(self):
        n_theta = 8
        n_phi = 10
        max_l = 3

        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=n_theta,
            n_phi=n_phi,
            max_l=max_l,
        )

        assert cache.theta_nodes.shape == (n_theta,)
        assert cache.phi_nodes.shape == (n_phi,)
        assert cache.weights.shape == (n_theta, n_phi)
        assert cache.harmonics_tensor.shape == (
            max_l + 1,
            2 * max_l + 1,
            n_theta,
            n_phi,
        )

    def test_constant_field_projects_to_y00(self):
        n_theta = 32
        n_phi = 32
        max_l = 4

        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=n_theta,
            n_phi=n_phi,
            max_l=max_l,
        )

        grid = np.ones((n_theta, n_phi), dtype=np.complex128)
        coeffs = SphericalHarmonicsVectorizer.compute_coefficients_vectorized(
            grid,
            cache,
        )

        expected_c00 = np.sqrt(4.0 * np.pi)
        c00 = coeffs[0, 0]

        assert abs(c00 - expected_c00) <= 1e-6

    def test_constant_field_reconstruction_roundtrip(self):
        n_theta = 24
        n_phi = 24
        max_l = 3

        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=n_theta,
            n_phi=n_phi,
            max_l=max_l,
        )

        grid = np.ones((n_theta, n_phi), dtype=np.complex128)
        coeffs = SphericalHarmonicsVectorizer.compute_coefficients_vectorized(
            grid,
            cache,
        )

        reconstructed = SphericalHarmonicsVectorizer.reconstruct_field_vectorized(
            coeffs,
            cache,
        )

        relative_error = la.norm(grid - reconstructed) / la.norm(grid)
        assert relative_error <= 1e-6

    def test_filter_coefficients_zero_cut_keeps_only_l0(self):
        n_theta = 16
        n_phi = 16
        max_l = 4

        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=n_theta,
            n_phi=n_phi,
            max_l=max_l,
        )

        grid = np.ones((n_theta, n_phi), dtype=np.complex128)
        coeffs = SphericalHarmonicsVectorizer.compute_coefficients_vectorized(
            grid,
            cache,
        )

        filtered, l_used = SphericalHarmonicsVectorizer.filter_coefficients(
            coeffs,
            l_cut=0,
        )

        assert l_used == 0
        assert abs(filtered[0, 0]) > 0.0
        assert np.allclose(filtered[1:, :], 0.0, atol=1e-15)

    def test_energy_retained_is_bounded(self):
        n_theta = 16
        n_phi = 16
        max_l = 3

        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=n_theta,
            n_phi=n_phi,
            max_l=max_l,
        )

        grid = np.ones((n_theta, n_phi), dtype=np.complex128)
        coeffs = SphericalHarmonicsVectorizer.compute_coefficients_vectorized(
            grid,
            cache,
        )

        filtered, _ = SphericalHarmonicsVectorizer.filter_coefficients(coeffs, 0)

        total_energy = SphericalHarmonicsVectorizer.coefficient_energy(coeffs)
        filtered_energy = SphericalHarmonicsVectorizer.coefficient_energy(filtered)

        retained = filtered_energy / (total_energy + _MACHINE_EPS)

        assert 0.0 <= retained <= 1.0
        assert retained > 0.99


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR TRICAPA EXTREMO A EXTREMO
# ══════════════════════════════════════════════════════════════════════════════
class TestGenerativeOpticHodgeSuturatorEndToEnd:
    """Pruebas de integración completa del pipeline functorial."""

    def _make_basis_inputs(self, dim: int = 8):
        G = np.eye(dim, dtype=np.float64)

        semantic = np.zeros(dim, dtype=np.float64)
        semantic[0] = 1.0

        raw_state = np.zeros(dim, dtype=np.complex128)
        raw_state[0] = 1.0

        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0

        M = 0.5 * np.eye(dim, dtype=np.float64)

        return G, ds, M, semantic, raw_state

    def test_basis_state_pipeline_is_stable(self):
        dim = 8
        G, ds, M, semantic, raw_state = self._make_basis_inputs(dim=dim)

        suturator = _make_suturator()
        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=4,
            period_t=1.0,
        )

        assert certificate.global_stability_index > 0.95
        assert certificate.is_globally_stable is True
        assert certificate.phase3_cert.compression_ratio > 0.0
        assert certificate.phase3_cert.compression_ratio <= 1.0

    def test_pipeline_final_state_is_passive(self):
        dim = 8
        G, ds, M, semantic, raw_state = self._make_basis_inputs(dim=dim)

        suturator = _make_suturator()
        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=4,
            period_t=1.0,
        )

        assert la.norm(certificate.final_state) <= la.norm(raw_state) + 1e-10

    def test_invalid_metric_raises_end_to_end(self):
        dim = 4
        G = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)
        _, ds, M, semantic, raw_state = self._make_basis_inputs(dim=dim)

        suturator = _make_suturator()

        with pytest.raises(MetricSignatureError):
            suturator.execute_optic_suturation(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                monodromy_matrix_m=M,
                semantic_direction=semantic,
                raw_state_vector=raw_state,
                grassmann_dimension=2,
                period_t=1.0,
            )

    def test_eikonal_violation_raises_end_to_end(self):
        dim = 6
        G, _, M, semantic, raw_state = self._make_basis_inputs(dim=dim)
        ds = np.zeros(dim, dtype=np.float64)

        suturator = _make_suturator()

        with pytest.raises(EikonalRefractionError):
            suturator.execute_optic_suturation(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                monodromy_matrix_m=M,
                semantic_direction=semantic,
                raw_state_vector=raw_state,
                grassmann_dimension=3,
                period_t=1.0,
            )

    def test_floquet_instability_raises_end_to_end(self):
        dim = 6
        G, ds, _, semantic, raw_state = self._make_basis_inputs(dim=dim)
        M = 1.2 * np.eye(dim, dtype=np.float64)

        suturator = _make_suturator()

        with pytest.raises(FloquetInstabilityError):
            suturator.execute_optic_suturation(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                monodromy_matrix_m=M,
                semantic_direction=semantic,
                raw_state_vector=raw_state,
                grassmann_dimension=3,
                period_t=1.0,
            )

    def test_invalid_grassmann_dimension_raises_end_to_end(self):
        dim = 6
        G, ds, M, semantic, raw_state = self._make_basis_inputs(dim=dim)

        suturator = _make_suturator()

        with pytest.raises(GrassmannRankDeficiency):
            suturator.execute_optic_suturation(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                monodromy_matrix_m=M,
                semantic_direction=semantic,
                raw_state_vector=raw_state,
                grassmann_dimension=dim + 1,
                period_t=1.0,
            )

    def test_optional_spherical_field_end_to_end(self):
        dim = 16
        G = np.eye(dim, dtype=np.float64)

        semantic = np.ones(dim, dtype=np.float64)
        semantic /= la.norm(semantic)

        raw_state = np.ones(dim, dtype=np.complex128)
        raw_state /= la.norm(raw_state)

        ds = semantic * 3.0
        M = 0.5 * np.eye(dim, dtype=np.float64)
        spherical_field = np.ones((4, 4), dtype=np.complex128)

        suturator = _make_suturator(
            n_theta=4,
            n_phi=4,
            max_l=2,
        )

        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=8,
            period_t=1.0,
            spherical_field=spherical_field,
            spherical_l_cut=1,
        )

        assert certificate.phase3_cert.spherical_degree_used == 1
        assert certificate.final_state.size == dim
        assert certificate.phase3_cert.compression_ratio <= 1.0
        assert la.norm(certificate.final_state) <= la.norm(raw_state) + 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTEZ NUMÉRICA Y REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════
class TestNumericalRobustness:
    """Pruebas de borde, repetibilidad y estabilidad numérica."""

    def test_complex_state_with_real_metric_is_supported(self):
        dim = 8
        G, ds, M, semantic, _ = TestGenerativeOpticHodgeSuturatorEndToEnd()._make_basis_inputs(dim=dim)

        raw_state = np.zeros(dim, dtype=np.complex128)
        raw_state[0] = np.exp(1j * 0.35)

        suturator = _make_suturator()
        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=4,
            period_t=1.0,
        )

        assert np.iscomplexobj(certificate.final_state)
        assert la.norm(certificate.final_state) <= la.norm(raw_state) + 1e-10

    def test_repeatability_of_pipeline(self):
        dim = 8
        G, ds, M, semantic, raw_state = TestGenerativeOpticHodgeSuturatorEndToEnd()._make_basis_inputs(dim=dim)

        suturator = _make_suturator()

        cert1 = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=4,
            period_t=1.0,
        )

        cert2 = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=4,
            period_t=1.0,
        )

        assert np.allclose(cert1.final_state, cert2.final_state, atol=1e-12)
        assert cert1.global_stability_index == pytest.approx(
            cert2.global_stability_index,
            abs=1e-12,
        )

    def test_large_dimension_identity_pipeline(self):
        dim = 32
        G = np.eye(dim, dtype=np.float64)

        semantic = np.zeros(dim, dtype=np.float64)
        semantic[0] = 1.0

        raw_state = np.zeros(dim, dtype=np.complex128)
        raw_state[0] = 1.0

        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0

        M = 0.5 * np.eye(dim, dtype=np.float64)

        suturator = _make_suturator()
        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            monodromy_matrix_m=M,
            semantic_direction=semantic,
            raw_state_vector=raw_state,
            grassmann_dimension=16,
            period_t=1.0,
        )

        assert certificate.global_stability_index > 0.90
        assert certificate.phase3_cert.compression_ratio <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))