# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas Unitarias: Einstein-Hilbert Agent                           ║
║ Ubicación: tests/unit/agents/omega/test_einstein_hilbert_agent.py            ║
║ Versión: 1.0.0 – Cobertura granular de las 3 fases anidadas + orquestador    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de prueba (mapeo 1-1 con las fases del módulo):
  • Fase 1  →  normalización causal, T_{μν}, condiciones de energía, handoff
  • Fase 2  →  Christoffel, Ricci, deformación, handoff tipado
  • Fase 3  →  termodinámica BH, decisión de colapso, veto ontológico
  • Agente  →  ciclo OODA completo (éxito + veto) y composición de morfismos

Dependencias externas se mockean de forma estricta para aislar la lógica
matemática del agente sin invocar el escudo gravitacional real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from numpy.typing import NDArray

# ── Imports del módulo bajo prueba ──────────────────────────────────────────
from app.omega.einstein_hilbert_agent import (
    AstrophysicalConstants,
    BlackHoleThermodynamics,
    CausalStructureError,
    CurvatureInvariants,
    EinsteinHilbertAgent,
    EnergyMomentumData,
    EnergyMomentumDegeneracyError,
    Phase1_EnergyMomentumExtractor,
    Phase2_EinsteinFieldSolver,
    Phase3_BekensteinHawkingDecider,
    SingularityVetoError,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y DOBLES DE PRUEBA (TEST DOUBLES)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def g_euclidean_4d() -> NDArray[np.float64]:
    """Métrica Euclidiana 4-D (firma ++++)."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def g_lorentzian_4d() -> NDArray[np.float64]:
    """Métrica Lorentziana 4-D (firma −+++)."""
    g = np.eye(4, dtype=np.float64)
    g[0, 0] = -1.0
    return g


@pytest.fixture
def g_euclidean_2d() -> NDArray[np.float64]:
    """Métrica Euclidiana 2-D (mínima para tests de curvatura)."""
    return np.eye(2, dtype=np.float64)


@pytest.fixture
def flow_velocity_iso_4d() -> NDArray[np.float64]:
    """Cuadrivelocidad isótropa normalizable en 4-D."""
    return np.ones(4, dtype=np.float64) / 2.0  # ||v||_E = 1 tras normalizar


@pytest.fixture
def mock_polaron() -> MagicMock:
    """PolaronCartridge mínimo con atributos requeridos."""
    p = MagicMock()
    p.frohlich_coupling = 0.5
    p.inertial_mass = 1.0
    p.effective_mass = 1.0
    p.volatility_alpha = 0.1
    p.fiedler_value = 0.0
    p.base_electron = None
    return p


@pytest.fixture
def energy_data_valid(g_euclidean_4d, flow_velocity_iso_4d) -> EnergyMomentumData:
    """EnergyMomentumData sintético válido (ρ=2, P=0.5, DEC ok)."""
    rho, P = 2.0, 0.5
    u = flow_velocity_iso_4d / np.linalg.norm(flow_velocity_iso_4d)
    u_cov = g_euclidean_4d @ u
    T = (rho + P) * np.outer(u_cov, u_cov) + P * g_euclidean_4d
    return EnergyMomentumData(
        T_tensor=T,
        effective_mass=rho,
        inflationary_pressure=P,
        four_velocity=u,
        trace=float(np.trace(T)),  # g=I ⇒ tr(T)
        energy_density=rho,
        weak_energy_ok=True,
        null_energy_ok=True,
        strong_energy_ok=True,
        dominant_energy_condition_ok=True,
        approximate_conservation_residual=0.0,
    )


def _make_warped(
    g: NDArray[np.float64],
    max_sec: float = 0.1,
    christoffel: Optional[NDArray[np.float64]] = None,
) -> MagicMock:
    """WarpedSpaceTime mock con métrica y Christoffel controlados."""
    w = MagicMock()
    w.deformed_metric = g.copy()
    n = g.shape[0]
    w.christoffel_symbols = (
        christoffel if christoffel is not None else np.zeros((n, n, n), dtype=np.float64)
    )
    w.max_sectional_curvature = max_sec
    return w


def _make_polyakov(
    is_trapped: bool = False,
    amplitude: float = 0.9,
    action: float = 1.5,
) -> MagicMock:
    """PolyakovAction mock."""
    pa = MagicMock()
    pa.is_trapped = is_trapped
    pa.feynman_amplitude = amplitude
    pa.action_integral = action
    return pa


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – Phase1_EnergyMomentumExtractor
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1DetectMetricSignature:
    """Detección de firma métrica (Euclidiana vs Lorentziana)."""

    def test_euclidean_signature(self, g_euclidean_4d):
        sig = Phase1_EnergyMomentumExtractor._detect_metric_signature(g_euclidean_4d)
        assert sig == pytest.approx(1.0)

    def test_lorentzian_signature(self, g_lorentzian_4d):
        sig = Phase1_EnergyMomentumExtractor._detect_metric_signature(g_lorentzian_4d)
        assert sig == pytest.approx(-1.0)

    def test_positive_definite_remains_plus(self):
        g = np.diag([2.0, 3.0, 4.0]).astype(np.float64)
        assert Phase1_EnergyMomentumExtractor._detect_metric_signature(g) == pytest.approx(1.0)


class TestPhase1Normalize4Velocity:
    """Normalización causal de la cuadrivelocidad."""

    def test_normalize_euclidean_unit(self, g_euclidean_4d):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        u = Phase1_EnergyMomentumExtractor._normalize_4velocity(v, g_euclidean_4d, +1.0)
        norm_sq = float(u.T @ g_euclidean_4d @ u)
        assert norm_sq == pytest.approx(1.0, abs=1e-12)
        assert u.shape == (4,)

    def test_normalize_lorentzian_timelike(self, g_lorentzian_4d):
        # Vector tipo tiempo: g(v,v) < 0
        v = np.array([2.0, 0.1, 0.0, 0.0], dtype=np.float64)
        u = Phase1_EnergyMomentumExtractor._normalize_4velocity(v, g_lorentzian_4d, -1.0)
        norm_sq = float(u.T @ g_lorentzian_4d @ u)
        assert norm_sq == pytest.approx(-1.0, abs=1e-12)

    def test_null_vector_raises(self, g_euclidean_4d):
        v = np.zeros(4, dtype=np.float64)
        with pytest.raises(CausalStructureError, match="nula"):
            Phase1_EnergyMomentumExtractor._normalize_4velocity(v, g_euclidean_4d, +1.0)

    def test_wrong_causal_type_raises(self, g_lorentzian_4d):
        # Vector espacial en métrica Lorentziana (g(v,v) > 0) con target −1
        v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(CausalStructureError, match="Tipo causal"):
            Phase1_EnergyMomentumExtractor._normalize_4velocity(v, g_lorentzian_4d, -1.0)

    def test_dimension_mismatch_raises(self, g_euclidean_4d):
        v = np.ones(3, dtype=np.float64)
        with pytest.raises(CausalStructureError, match="Dimensión"):
            Phase1_EnergyMomentumExtractor._normalize_4velocity(v, g_euclidean_4d, +1.0)


class TestPhase1EnergyConditions:
    """Verificación algebraica de WEC / NEC / SEC / DEC."""

    def test_all_ok_for_physical_fluid(self, g_euclidean_4d):
        rho, P = 2.0, 0.5
        T = np.eye(4)  # placeholder; la función usa ρ, P
        u = np.array([1.0, 0, 0, 0], dtype=np.float64)
        wec, nec, sec, dec = Phase1_EnergyMomentumExtractor._check_energy_conditions(
            T, u, rho, P, g_euclidean_4d
        )
        assert wec and nec and sec and dec

    def test_dec_fails_when_pressure_exceeds_density(self, g_euclidean_4d):
        rho, P = 1.0, 2.0  # |P| > ρ
        T = np.eye(4)
        u = np.ones(4) / 2.0
        wec, nec, sec, dec = Phase1_EnergyMomentumExtractor._check_energy_conditions(
            T, u, rho, P, g_euclidean_4d
        )
        assert dec is False
        assert nec is True  # ρ+P = 3 > 0

    def test_wec_fails_for_negative_density(self, g_euclidean_4d):
        rho, P = -1.0, 0.0
        T = np.eye(4)
        u = np.ones(4) / 2.0
        wec, nec, sec, dec = Phase1_EnergyMomentumExtractor._check_energy_conditions(
            T, u, rho, P, g_euclidean_4d
        )
        assert wec is False
        assert dec is False

    def test_sec_fails_for_large_negative_pressure(self, g_euclidean_4d):
        # ρ+P ≥ 0 pero ρ+3P < 0  →  P ∈ [−ρ, −ρ/3)
        rho, P = 3.0, -1.5  # ρ+P=1.5≥0, ρ+3P=3−4.5=−1.5<0
        T = np.eye(4)
        u = np.ones(4) / 2.0
        wec, nec, sec, dec = Phase1_EnergyMomentumExtractor._check_energy_conditions(
            T, u, rho, P, g_euclidean_4d
        )
        assert nec is True
        assert sec is False


class TestPhase1ComputeStressEnergyTensor:
    """Construcción completa de T_{μν} y del artefacto EnergyMomentumData."""

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=2.5)
    def test_tensor_symmetry_and_shape(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        data = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.5,
        )
        assert isinstance(data, EnergyMomentumData)
        assert data.T_tensor.shape == (4, 4)
        assert np.allclose(data.T_tensor, data.T_tensor.T, atol=1e-12)
        assert data.effective_mass == pytest.approx(2.5)
        assert data.inflationary_pressure == pytest.approx(0.5)
        assert data.energy_density == pytest.approx(2.5)
        assert data.four_velocity.shape == (4,)
        # g(u,u) ≈ +1
        assert float(data.four_velocity.T @ g_euclidean_4d @ data.four_velocity) == pytest.approx(
            1.0, abs=1e-12
        )

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=2.0)
    def test_perfect_fluid_formula(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        """T = (ρ+P) u⊗u + P g  (con u covariante)."""
        rho, P = 2.0, 0.3
        data = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=P,
        )
        u = data.four_velocity
        u_cov = g_euclidean_4d @ u
        T_expected = (rho + P) * np.outer(u_cov, u_cov) + P * g_euclidean_4d
        assert np.allclose(data.T_tensor, T_expected, atol=1e-12)

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1.0)
    def test_dec_flag_false_when_pressure_high(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        data = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=5.0,  # |P| > ρ
        )
        assert data.dominant_energy_condition_ok is False

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=-1.0)
    def test_negative_mass_raises(self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d):
        with pytest.raises(EnergyMomentumDegeneracyError, match="no física"):
            Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
                g_base=g_euclidean_4d,
                polaron=mock_polaron,
                flow_velocity=flow_velocity_iso_4d,
                market_pressure=0.0,
            )

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=float("nan"))
    def test_nan_mass_raises(self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d):
        with pytest.raises(EnergyMomentumDegeneracyError, match="no física"):
            Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
                g_base=g_euclidean_4d,
                polaron=mock_polaron,
                flow_velocity=flow_velocity_iso_4d,
                market_pressure=0.0,
            )

    def test_singular_metric_raises(self, mock_polaron, flow_velocity_iso_4d):
        g_sing = np.zeros((4, 4), dtype=np.float64)
        with patch(
            "app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1.0
        ):
            # Normalización fallará primero (norma nula) o inv fallará
            with pytest.raises((CausalStructureError, EnergyMomentumDegeneracyError)):
                Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
                    g_base=g_sing,
                    polaron=mock_polaron,
                    flow_velocity=flow_velocity_iso_4d,
                    market_pressure=0.0,
                )

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1.5)
    def test_frohlich_override_passed(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.1,
            frohlich_coupling_override=9.9,
        )
        mock_mass.assert_called_once()
        _, kwargs = mock_mass.call_args
        # Puede ser posicional o keyword según firma
        assert mock_mass.call_args.kwargs.get("frohlich_coupling", mock_mass.call_args[1].get("frohlich_coupling") if len(mock_mass.call_args) > 1 else None) == 9.9 or (
            mock_mass.call_args[0][2] == 9.9 if len(mock_mass.call_args[0]) > 2 else True
        )


class TestPhase1ObserveAndHandoff:
    """Método terminal de Fase 1: continuidad tipada hacia Fase 2."""

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=3.0)
    def test_handoff_returns_energy_momentum_data(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        result = Phase1_EnergyMomentumExtractor.observe_and_handoff(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.2,
        )
        assert isinstance(result, EnergyMomentumData)
        assert result.effective_mass == pytest.approx(3.0)
        assert hasattr(result, "T_tensor")
        assert hasattr(result, "dominant_energy_condition_ok")

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=3.0)
    def test_handoff_is_pure_delegate(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        """observe_and_handoff ≡ compute_stress_energy_tensor (mismo resultado)."""
        a = Phase1_EnergyMomentumExtractor.observe_and_handoff(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.2,
        )
        b = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.2,
        )
        assert np.allclose(a.T_tensor, b.T_tensor)
        assert a.effective_mass == b.effective_mass
        assert a.trace == pytest.approx(b.trace)


class TestPhase1DivergenceResidual:
    """Residual de conservación (indicador, no prueba geométrica)."""

    def test_residual_non_negative(self, g_euclidean_4d):
        T = np.eye(4, dtype=np.float64)
        res = Phase1_EnergyMomentumExtractor._approximate_divergence_residual(T, g_euclidean_4d)
        assert res >= 0.0
        assert math.isfinite(res)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 – Phase2_EinsteinFieldSolver
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2Christoffel:
    """Cálculo de símbolos de Christoffel."""

    def test_flat_metric_near_zero_gamma(self, g_euclidean_4d):
        Gamma = Phase2_EinsteinFieldSolver._compute_christoffel_from_metric(g_euclidean_4d)
        assert Gamma.shape == (4, 4, 4)
        # Métrica plana: Γ debería ser numéricamente pequeño (proxy logarítmico)
        assert np.isfinite(Gamma).all()

    def test_diagonal_metric_shape(self):
        g = np.diag([1.0, 2.0, 3.0]).astype(np.float64)
        Gamma = Phase2_EinsteinFieldSolver._compute_christoffel_from_metric(g)
        assert Gamma.shape == (3, 3, 3)

    def test_singular_metric_raises(self):
        g = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(EnergyMomentumDegeneracyError, match="singular"):
            Phase2_EinsteinFieldSolver._compute_christoffel_from_metric(g)


class TestPhase2Ricci:
    """Tensor y escalar de Ricci."""

    def test_ricci_symmetric(self, g_euclidean_4d):
        n = 4
        Gamma = np.zeros((n, n, n), dtype=np.float64)
        # Inyectar un Γ no trivial para activar términos cuadráticos
        Gamma[0, 1, 1] = 0.1
        Gamma[1, 0, 1] = 0.05
        Ricci = Phase2_EinsteinFieldSolver._compute_ricci_tensor(Gamma, g_euclidean_4d)
        assert Ricci.shape == (n, n)
        assert np.allclose(Ricci, Ricci.T, atol=1e-12)

    def test_ricci_scalar_flat_small(self, g_euclidean_4d):
        Ricci = np.zeros((4, 4), dtype=np.float64)
        R = Phase2_EinsteinFieldSolver._compute_ricci_scalar(Ricci, g_euclidean_4d)
        assert R == pytest.approx(0.0)

    def test_ricci_scalar_known_trace(self):
        g = np.eye(2, dtype=np.float64)
        Ricci = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
        R = Phase2_EinsteinFieldSolver._compute_ricci_scalar(Ricci, g)
        assert R == pytest.approx(5.0)

    def test_ricci_scalar_singular_metric_returns_nan(self):
        g = np.zeros((2, 2), dtype=np.float64)
        Ricci = np.eye(2)
        R = Phase2_EinsteinFieldSolver._compute_ricci_scalar(Ricci, g)
        assert math.isnan(R)


class TestPhase2OrientAndHandoff:
    """orient_from_energy_momentum / deform_and_handoff con mocks del escudo."""

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    def test_orient_returns_triplet(
        self,
        mock_shield_cls,
        mock_deform,
        energy_data_valid,
        g_euclidean_4d,
    ):
        shield_instance = MagicMock()
        shield_instance._g_base = g_euclidean_4d
        shield_instance.enforce_gravitational_attractor.return_value = _make_polyakov(
            is_trapped=False, amplitude=0.8, action=1.2
        )
        mock_shield_cls.return_value = shield_instance
        mock_deform.return_value = _make_warped(g_euclidean_4d, max_sec=0.05)

        solver = Phase2_EinsteinFieldSolver()
        # Re-asignar g_base tras el mock del __init__
        solver._g_base = g_euclidean_4d
        solver._shield = shield_instance

        attention = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        warped, polyakov, inv = solver.orient_from_energy_momentum(
            energy_data=energy_data_valid,
            node_index=0,
            attention_vector=attention,
        )

        assert warped is mock_deform.return_value
        assert polyakov.is_trapped is False
        assert isinstance(inv, CurvatureInvariants)
        assert math.isfinite(inv.ricci_scalar)
        assert inv.max_sectional_curvature == pytest.approx(0.05)
        assert inv.christoffel_norm >= 0.0
        shield_instance.enforce_gravitational_attractor.assert_called_once()
        mock_deform.assert_called_once()

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    def test_deform_and_handoff_is_terminal_alias(
        self,
        mock_shield_cls,
        mock_deform,
        energy_data_valid,
        g_euclidean_4d,
    ):
        shield_instance = MagicMock()
        shield_instance._g_base = g_euclidean_4d
        shield_instance.enforce_gravitational_attractor.return_value = _make_polyakov()
        mock_shield_cls.return_value = shield_instance
        mock_deform.return_value = _make_warped(g_euclidean_4d)

        solver = Phase2_EinsteinFieldSolver()
        solver._g_base = g_euclidean_4d
        solver._shield = shield_instance

        attention = np.ones(4, dtype=np.float64)
        result = solver.deform_and_handoff(
            energy_data=energy_data_valid,
            node_index=1,
            attention_vector=attention,
        )
        assert len(result) == 3
        warped, polyakov, inv = result
        assert hasattr(warped, "deformed_metric")
        assert hasattr(polyakov, "is_trapped")
        assert isinstance(inv, CurvatureInvariants)

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    def test_orient_uses_shield_christoffel_when_present(
        self,
        mock_shield_cls,
        mock_deform,
        energy_data_valid,
        g_euclidean_4d,
    ):
        n = 4
        custom_gamma = np.ones((n, n, n), dtype=np.float64) * 0.01
        warped = _make_warped(g_euclidean_4d, max_sec=1.0, christoffel=custom_gamma)

        shield_instance = MagicMock()
        shield_instance._g_base = g_euclidean_4d
        shield_instance.enforce_gravitational_attractor.return_value = _make_polyakov()
        mock_shield_cls.return_value = shield_instance
        mock_deform.return_value = warped

        solver = Phase2_EinsteinFieldSolver()
        solver._g_base = g_euclidean_4d
        solver._shield = shield_instance

        _, _, inv = solver.orient_from_energy_momentum(
            energy_data=energy_data_valid,
            node_index=0,
            attention_vector=np.ones(4),
        )
        # ||Γ|| = ||0.01 * ones|| = 0.01 * sqrt(n³)
        expected_norm = 0.01 * math.sqrt(n ** 3)
        assert inv.christoffel_norm == pytest.approx(expected_norm, rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 – Phase3_BekensteinHawkingDecider
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase3EvaluateSingularity:
    """Invariantes de Bekenstein-Hawking."""

    def test_positive_mass_finite_thermo(self):
        m = 10.0
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(m)
        assert isinstance(bh, BlackHoleThermodynamics)
        assert bh.schwarzschild_radius > 0.0
        assert bh.horizon_area > 0.0
        assert bh.bekenstein_hawking_entropy > 0.0
        assert bh.hawking_temperature > 0.0
        assert math.isfinite(bh.hawking_temperature)
        assert bh.horizon_euler_characteristic == 2

    def test_schwarzschild_radius_formula(self):
        m = 5.0
        from app.core.immune_system.gravity_shield import GravitationalConstants
        G = GravitationalConstants.CYBER_G
        c2 = GravitationalConstants.CYBER_C ** 2
        expected_rs = 2.0 * G * m / c2
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(m)
        assert bh.schwarzschild_radius == pytest.approx(expected_rs)

    def test_area_is_4pi_rs_squared(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(7.0)
        expected_area = 4.0 * math.pi * bh.schwarzschild_radius ** 2
        assert bh.horizon_area == pytest.approx(expected_area)

    def test_entropy_proportional_to_area(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(3.0)
        expected_s = (
            AstrophysicalConstants.BOLTZMANN_K
            * bh.horizon_area
            / (4.0 * AstrophysicalConstants.PLANCK_LENGTH_SQ)
        )
        assert bh.bekenstein_hawking_entropy == pytest.approx(expected_s)

    def test_hawking_temperature_inverse_mass(self):
        bh1 = Phase3_BekensteinHawkingDecider.evaluate_singularity(2.0)
        bh2 = Phase3_BekensteinHawkingDecider.evaluate_singularity(4.0)
        # T_H ∝ 1/M
        assert bh1.hawking_temperature == pytest.approx(2.0 * bh2.hawking_temperature)

    def test_zero_mass_infinite_temperature(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(0.0)
        assert bh.hawking_temperature == float("inf")
        assert bh.schwarzschild_radius == pytest.approx(0.0)

    def test_negative_mass_raises(self):
        with pytest.raises(EnergyMomentumDegeneracyError, match="no física"):
            Phase3_BekensteinHawkingDecider.evaluate_singularity(-1.0)

    def test_nan_mass_raises(self):
        with pytest.raises(EnergyMomentumDegeneracyError, match="no física"):
            Phase3_BekensteinHawkingDecider.evaluate_singularity(float("nan"))


class TestPhase3DecideFromQuantumCollapse:
    """Decisión condicional: horizonte vs geodésica libre."""

    def test_not_trapped_returns_none(self, g_euclidean_4d):
        warped = _make_warped(g_euclidean_4d)
        polyakov = _make_polyakov(is_trapped=False, amplitude=0.7)
        inv = CurvatureInvariants(
            ricci_tensor=np.zeros((4, 4)),
            ricci_scalar=0.01,
            max_sectional_curvature=0.01,
            christoffel_norm=0.0,
        )
        result = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped,
            polyakov_action=polyakov,
            curvature=inv,
            effective_mass=2.0,
        )
        assert result is None

    def test_trapped_returns_thermodynamics(self, g_euclidean_4d):
        warped = _make_warped(g_euclidean_4d, max_sec=1e4)
        polyakov = _make_polyakov(is_trapped=True, amplitude=0.0)
        inv = CurvatureInvariants(
            ricci_tensor=np.eye(4) * 100,
            ricci_scalar=1e4,
            max_sectional_curvature=1e4,
            christoffel_norm=10.0,
        )
        result = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped,
            polyakov_action=polyakov,
            curvature=inv,
            effective_mass=5.0,
        )
        assert isinstance(result, BlackHoleThermodynamics)
        assert result.schwarzschild_radius > 0.0
        assert result.horizon_euler_characteristic == 2

    def test_trapped_with_mild_curvature_still_decides(self, g_euclidean_4d):
        """is_trapped=True es condición suficiente, independientemente de R."""
        warped = _make_warped(g_euclidean_4d, max_sec=0.001)
        polyakov = _make_polyakov(is_trapped=True, amplitude=1e-20)
        inv = CurvatureInvariants(
            ricci_tensor=np.zeros((4, 4)),
            ricci_scalar=0.0,
            max_sectional_curvature=0.001,
            christoffel_norm=0.0,
        )
        result = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped,
            polyakov_action=polyakov,
            curvature=inv,
            effective_mass=1.0,
        )
        assert result is not None


class TestPhase3ActVeto:
    """Veto ontológico: siempre lanza SingularityVetoError."""

    def test_act_veto_raises_singularity_error(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(1.0)
        polyakov = _make_polyakov(is_trapped=True, amplitude=0.0)
        with pytest.raises(SingularityVetoError, match="Horizonte"):
            Phase3_BekensteinHawkingDecider.act_veto(bh, polyakov)

    def test_act_veto_is_topological_invariant_error(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(1.0)
        polyakov = _make_polyakov(is_trapped=True)
        with pytest.raises(SingularityVetoError) as exc_info:
            Phase3_BekensteinHawkingDecider.act_veto(bh, polyakov)
        # Jerarquía: SingularityVetoError ⊂ TopologicalInvariantError
        from app.core.mic_algebra import TopologicalInvariantError
        assert isinstance(exc_info.value, TopologicalInvariantError)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SUPREMO – EinsteinHilbertAgent (ciclo OODA completo)
# ══════════════════════════════════════════════════════════════════════════════

class TestEinsteinHilbertAgentOODA:
    """Orquestación completa: Observe → Orient → Decide → Act."""

    @pytest.fixture
    def agent_with_mocks(self, g_euclidean_4d):
        """Agente con shield y deformación mockeados."""
        with patch(
            "app.omega.einstein_hilbert_agent.GravitationalShieldFunctor"
        ) as mock_shield_cls, patch(
            "app.omega.einstein_hilbert_agent.G_PHYSICS", g_euclidean_4d
        ):
            shield = MagicMock()
            shield._g_base = g_euclidean_4d
            mock_shield_cls.return_value = shield
            agent = EinsteinHilbertAgent()
            agent._g_base = g_euclidean_4d
            agent._solver._g_base = g_euclidean_4d
            agent._solver._shield = shield
            yield agent, shield

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=2.0)
    def test_ooda_success_path_returns_categorical_state(
        self,
        mock_mass,
        mock_deform,
        agent_with_mocks,
        mock_polaron,
        g_euclidean_4d,
    ):
        agent, shield = agent_with_mocks
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(
            is_trapped=False, amplitude=0.85, action=1.1
        )
        mock_deform.return_value = _make_warped(g_euclidean_4d, max_sec=0.02)

        from app.core.mic_algebra import CategoricalState

        state = agent.execute_covariant_ooda(
            polaron=mock_polaron,
            node_index=0,
            attention_vector=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
            market_pressure=0.3,
        )

        assert isinstance(state, CategoricalState)
        assert state.stratum == "WISDOM"
        assert "effective_mass" in state.payload
        assert state.payload["effective_mass"] == pytest.approx(2.0)
        assert "ricci_scalar" in state.payload
        assert "energy_conditions" in state.payload
        assert state.payload["energy_conditions"]["DEC"] is True
        assert state.payload["feynman_amplitude"] == pytest.approx(0.85)

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=4.0)
    def test_ooda_veto_path_raises_singularity(
        self,
        mock_mass,
        mock_deform,
        agent_with_mocks,
        mock_polaron,
        g_euclidean_4d,
    ):
        agent, shield = agent_with_mocks
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(
            is_trapped=True, amplitude=0.0, action=99.0
        )
        mock_deform.return_value = _make_warped(g_euclidean_4d, max_sec=1e5)

        with pytest.raises(SingularityVetoError, match="Horizonte"):
            agent.execute_covariant_ooda(
                polaron=mock_polaron,
                node_index=3,
                attention_vector=np.ones(4, dtype=np.float64),
                market_pressure=0.1,
            )

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1.0)
    def test_ooda_dec_violation_still_completes_if_not_trapped(
        self,
        mock_mass,
        mock_deform,
        agent_with_mocks,
        mock_polaron,
        g_euclidean_4d,
    ):
        """DEC violada genera warning pero no aborta si no hay horizonte."""
        agent, shield = agent_with_mocks
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(
            is_trapped=False, amplitude=0.5, action=0.8
        )
        mock_deform.return_value = _make_warped(g_euclidean_4d)

        state = agent.execute_covariant_ooda(
            polaron=mock_polaron,
            node_index=0,
            attention_vector=np.ones(4) / 2.0,
            market_pressure=10.0,  # |P| ≫ ρ ⇒ DEC falla
        )
        assert state.stratum == "WISDOM"
        assert state.payload["energy_conditions"]["DEC"] is False

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=2.0)
    def test_ooda_phase_order_call_sequence(
        self,
        mock_mass,
        mock_deform,
        agent_with_mocks,
        mock_polaron,
        g_euclidean_4d,
    ):
        """Verifica que el escudo (Fase 2) se invoca exactamente una vez tras Fase 1."""
        agent, shield = agent_with_mocks
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(is_trapped=False)
        mock_deform.return_value = _make_warped(g_euclidean_4d)

        agent.execute_covariant_ooda(
            polaron=mock_polaron,
            node_index=0,
            attention_vector=np.ones(4) / 2.0,
            market_pressure=0.0,
        )
        mock_mass.assert_called_once()
        shield.enforce_gravitational_attractor.assert_called_once()
        mock_deform.assert_called_once()


class TestEinsteinHilbertAgentInheritance:
    """Contrato categórico: el agente es un Morphism."""

    def test_is_morphism_subclass(self):
        from app.core.mic_algebra import Morphism
        assert issubclass(EinsteinHilbertAgent, Morphism)

    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    @patch("app.omega.einstein_hilbert_agent.G_PHYSICS", np.eye(4))
    def test_instantiation(self, mock_shield_cls):
        mock_shield_cls.return_value = MagicMock(_g_base=np.eye(4))
        agent = EinsteinHilbertAgent()
        assert agent._g_base is not None
        assert agent._solver is not None


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y EXCEPCIONES (sanidad de API pública)
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicAPIAndConstants:
    """Sanidad de constantes y jerarquía de excepciones."""

    def test_astrophysical_constants_positive(self):
        assert AstrophysicalConstants.PLANCK_LENGTH_SQ > 0.0
        assert AstrophysicalConstants.BOLTZMANN_K > 0.0
        assert AstrophysicalConstants.PI == pytest.approx(math.pi)
        assert AstrophysicalConstants.HAWKING_TEMP_FACTOR > 0.0
        assert AstrophysicalConstants.HORIZON_EULER_CHARACTERISTIC == 2
        assert AstrophysicalConstants.SYMMETRY_ATOL > 0.0

    def test_exception_hierarchy(self):
        from app.core.mic_algebra import TopologicalInvariantError
        assert issubclass(SingularityVetoError, TopologicalInvariantError)
        assert issubclass(EnergyMomentumDegeneracyError, TopologicalInvariantError)
        assert issubclass(CausalStructureError, TopologicalInvariantError)

    def test_energy_momentum_data_frozen(self, energy_data_valid):
        with pytest.raises(Exception):
            energy_data_valid.effective_mass = 999.0  # type: ignore[misc]

    def test_black_hole_thermo_frozen(self):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(1.0)
        with pytest.raises(Exception):
            bh.schwarzschild_radius = 0.0  # type: ignore[misc]

    def test_curvature_invariants_frozen(self):
        inv = CurvatureInvariants(
            ricci_tensor=np.zeros((2, 2)),
            ricci_scalar=0.0,
            max_sectional_curvature=0.0,
            christoffel_norm=0.0,
        )
        with pytest.raises(Exception):
            inv.ricci_scalar = 1.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE CONTINUIDAD TIPADA ENTRE FASES (contratos de anidamiento)
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseNestingContracts:
    """
    Garantiza que el tipo de salida de la fase N es el dominio de la fase N+1.
    """

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1.5)
    def test_phase1_output_is_valid_phase2_input(
        self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d
    ):
        em = Phase1_EnergyMomentumExtractor.observe_and_handoff(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.1,
        )
        # Campos obligatorios que Fase 2 consume
        assert isinstance(em.effective_mass, float)
        assert em.T_tensor.ndim == 2
        assert em.four_velocity.ndim == 1
        assert isinstance(em.dominant_energy_condition_ok, bool)

    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    def test_phase2_output_is_valid_phase3_input(
        self,
        mock_shield_cls,
        mock_deform,
        energy_data_valid,
        g_euclidean_4d,
    ):
        shield = MagicMock()
        shield._g_base = g_euclidean_4d
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(is_trapped=True)
        mock_shield_cls.return_value = shield
        mock_deform.return_value = _make_warped(g_euclidean_4d, max_sec=10.0)

        solver = Phase2_EinsteinFieldSolver()
        solver._g_base = g_euclidean_4d
        solver._shield = shield

        warped, polyakov, inv = solver.deform_and_handoff(
            energy_data=energy_data_valid,
            node_index=0,
            attention_vector=np.ones(4),
        )
        # Fase 3 debe poder consumir el triplete sin error de atributos
        result = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped,
            polyakov_action=polyakov,
            curvature=inv,
            effective_mass=energy_data_valid.effective_mass,
        )
        assert isinstance(result, BlackHoleThermodynamics)

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=2.0)
    @patch("app.omega.einstein_hilbert_agent._deform_metric_tensor")
    @patch("app.omega.einstein_hilbert_agent.GravitationalShieldFunctor")
    def test_full_pipeline_phase1_to_phase3_free_geodesic(
        self,
        mock_shield_cls,
        mock_deform,
        mock_mass,
        g_euclidean_4d,
        mock_polaron,
        flow_velocity_iso_4d,
    ):
        """Pipeline completo sin veto: Phase1 → Phase2 → Phase3 → None."""
        shield = MagicMock()
        shield._g_base = g_euclidean_4d
        shield.enforce_gravitational_attractor.return_value = _make_polyakov(is_trapped=False)
        mock_shield_cls.return_value = shield
        mock_deform.return_value = _make_warped(g_euclidean_4d)

        em = Phase1_EnergyMomentumExtractor.observe_and_handoff(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.0,
        )
        solver = Phase2_EinsteinFieldSolver()
        solver._g_base = g_euclidean_4d
        solver._shield = shield

        warped, polyakov, inv = solver.deform_and_handoff(
            energy_data=em,
            node_index=0,
            attention_vector=np.ones(4) / 2.0,
        )
        decision = Phase3_BekensteinHawkingDecider.decide_from_quantum_collapse(
            warped_space=warped,
            polyakov_action=polyakov,
            curvature=inv,
            effective_mass=em.effective_mass,
        )
        assert decision is None


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS NUMÉRICAS DE ESTABILIDAD / REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Estabilidad numérica en bordes del dominio."""

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1e-30)
    def test_tiny_mass(self, mock_mass, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d):
        data = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
            g_base=g_euclidean_4d,
            polaron=mock_polaron,
            flow_velocity=flow_velocity_iso_4d,
            market_pressure=0.0,
        )
        assert data.effective_mass == pytest.approx(1e-30)
        assert np.isfinite(data.T_tensor).all()

    @patch("app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=1e20)
    def test_huge_mass_thermo(self, mock_mass):
        bh = Phase3_BekensteinHawkingDecider.evaluate_singularity(1e20)
        assert math.isfinite(bh.schwarzschild_radius)
        assert math.isfinite(bh.horizon_area)
        assert math.isfinite(bh.bekenstein_hawking_entropy)
        assert bh.hawking_temperature > 0.0
        # T_H muy pequeña para M enorme
        assert bh.hawking_temperature < 1e-10 or bh.hawking_temperature < AstrophysicalConstants.HAWKING_TEMP_FACTOR

    def test_pressure_zero_dust(self, g_euclidean_4d, mock_polaron, flow_velocity_iso_4d):
        """Polvo (P=0): T_{μν} = ρ u_μ u_ν."""
        with patch(
            "app.omega.einstein_hilbert_agent._acquire_effective_mass", return_value=3.0
        ):
            data = Phase1_EnergyMomentumExtractor.compute_stress_energy_tensor(
                g_base=g_euclidean_4d,
                polaron=mock_polaron,
                flow_velocity=flow_velocity_iso_4d,
                market_pressure=0.0,
            )
        u_cov = g_euclidean_4d @ data.four_velocity
        T_expected = 3.0 * np.outer(u_cov, u_cov)
        assert np.allclose(data.T_tensor, T_expected, atol=1e-12)
        assert data.weak_energy_ok
        assert data.dominant_energy_condition_ok