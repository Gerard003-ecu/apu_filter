# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para kbase_thermodynamic_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.alfa.kbase.kbase_thermodynamic_engine import (
    KBaseThermodynamicEngine,
    HeytingVerdict,
    SpectralChart,
    DensityOperatorChart,
    PhaseOneKBasePacket,
    PhaseTwoKBasePacket,
    KBaseTelemetry,
    SpectralThermoReport
)
from app.core.mic_algebra import Morphism

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_eps(self):
        from app.alfa.kbase.kbase_thermodynamic_engine import _MACHINE_EPS
        assert _MACHINE_EPS > 0.0

    def test_wilkinson_floor(self):
        from app.alfa.kbase.kbase_thermodynamic_engine import _WILKINSON_FLOOR
        assert _WILKINSON_FLOOR > 0.0

class TestHeytingVerdict:
    def test_heyting_rank(self):
        assert HeytingVerdict.VETOED.rank == 0
        assert HeytingVerdict.CERTIFIED.rank == 3

    def test_heyting_meet(self):
        assert HeytingVerdict.COHERENT.meet(HeytingVerdict.DEGRADED) == HeytingVerdict.DEGRADED

    def test_heyting_join(self):
        assert HeytingVerdict.COHERENT.join(HeytingVerdict.DEGRADED) == HeytingVerdict.COHERENT

    def test_heyting_implies(self):
        assert HeytingVerdict.DEGRADED.implies(HeytingVerdict.COHERENT) == HeytingVerdict.CERTIFIED
        assert HeytingVerdict.COHERENT.implies(HeytingVerdict.DEGRADED) == HeytingVerdict.DEGRADED

    def test_heyting_negate(self):
        assert HeytingVerdict.VETOED.negate() == HeytingVerdict.CERTIFIED
        assert HeytingVerdict.CERTIFIED.negate() == HeytingVerdict.VETOED

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_spectral_chart_frozen(self):
        chart = SpectralChart(
            eigenvalues=np.array([1.0], dtype=np.float64),
            eigenvectors=np.eye(1, dtype=np.float64),
            inv_metric=np.eye(1, dtype=np.float64),
            condition_number=1.0, volume_density=1.0, operator_norm=1.0,
            frobenius_norm=1.0, spectral_gap=1.0, regularized=False, tikhonov_alpha=1e-15
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            chart.condition_number = 2.0
            
    def test_density_operator_chart_frozen(self):
        chart = DensityOperatorChart(
            eigenvalues=np.array([1.0], dtype=np.float64),
            eigenvectors=np.eye(1, dtype=np.float64),
            trace=1.0, min_eigenvalue=1.0, is_density=True,
            von_neumann_entropy=0.0, purity=1.0, energy_expectation=1.0,
            quantum_free_energy=1.0, log_partition=1.0, relative_entropy_to_gibbs=0.0,
            stationarity_defect=0.0, banach_submultiplicative_defect=0.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            chart.trace = 2.0

class TestPhase1_ObserveOrient:
    @pytest.fixture
    def engine(self):
        return KBaseThermodynamicEngine(dimension_n=3)
        
    def test_engine_init_invalid_dim(self):
        with pytest.raises(ValueError):
            KBaseThermodynamicEngine(dimension_n=0)

    def test_kahan_sum_valid(self, engine):
        arr = np.array([1e16, 1.0, -1e16], dtype=np.float64)
        res = engine.kahan_sum(arr)
        assert abs(res - 1.0) < 1e-10

    def test_kahan_sum_invalid_dim(self, engine):
        with pytest.raises(ValueError):
            engine.kahan_sum(np.eye(2))

    def test_validate_square_matrix(self, engine):
        M = np.eye(3, dtype=np.float64)
        engine._validate_square_matrix(M, "M") # Should not raise
        with pytest.raises(ValueError):
            engine._validate_square_matrix(np.eye(2, dtype=np.float64), "M")

    def test_validate_hermitian(self, engine):
        M = np.eye(3, dtype=np.float64)
        engine._validate_hermitian(M, "M")
        M[0, 1] = 1.0
        with pytest.raises(ValueError):
            engine._validate_hermitian(M, "M")

    def test_factorize_metric_spd(self, engine):
        M = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)
        chart = engine._factorize_metric(M)
        assert not chart.regularized
        np.testing.assert_allclose(chart.eigenvalues, [2.0, 3.0, 4.0])
        assert abs(chart.condition_number - 2.0) < 1e-10

    def test_factorize_metric_regularized(self, engine):
        M = np.zeros((3, 3), dtype=np.float64)
        chart = engine._factorize_metric(M)
        assert chart.regularized
        assert abs(chart.condition_number - 1.0) < 1e-10

    def test_compute_riemannian_pullback_covariant(self, engine):
        G = np.eye(3, dtype=np.float64) * 2.0
        M = np.ones((3, 3), dtype=np.float64)
        res = engine.compute_riemannian_pullback(G, M, "covariant")
        np.testing.assert_allclose(res, M * 4.0)

    def test_compute_riemannian_pullback_contravariant(self, engine):
        G = np.eye(3, dtype=np.float64) * 2.0
        M = np.ones((3, 3), dtype=np.float64)
        res = engine.compute_riemannian_pullback(G, M, "contravariant")
        np.testing.assert_allclose(res, M * 0.25)

    def test_compute_riemannian_pullback_jacobian(self, engine):
        G = np.eye(3, dtype=np.float64)
        M = np.eye(3, dtype=np.float64)
        J = np.eye(3, dtype=np.float64) * 3.0
        res = engine.compute_riemannian_pullback(G, M, "covariant", jacobian=J)
        np.testing.assert_allclose(res, np.eye(3) * 9.0)

    def test_pullback_spectral_entropy(self, engine):
        M = np.eye(3, dtype=np.float64)
        S = engine._pullback_spectral_entropy(M)
        expected = -math.log(1.0/3.0)
        assert abs(S - expected) < 1e-10

    def test_compute_gibbs_free_energy(self, engine):
        G, H = engine.compute_gibbs_free_energy(10.0, 2.0, 3.0, 300.0, 0.01)
        assert abs(H - 16.0) < 1e-10
        assert abs(G - 13.0) < 1e-10

    def test_compute_gibbs_free_energy_invalid_temp(self, engine):
        with pytest.raises(ValueError):
            engine.compute_gibbs_free_energy(10.0, 2.0, 3.0, -5.0, 0.01)

    def test_complex_step_dG_dT(self, engine):
        dG = engine._complex_step_dG_dT(10.0, 5.0, 300.0)
        assert abs(dG - (-5.0)) < 1e-10

    def test_evaluate_clausius_duhem_coherent(self, engine):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gradT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        phi, coherent = engine.evaluate_clausius_duhem(0.0, q, gradT, 300.0)
        assert coherent
        assert phi > 0.0

    def test_evaluate_clausius_duhem_incoherent(self, engine):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gradT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        phi, coherent = engine.evaluate_clausius_duhem(0.0, q, gradT, 300.0)
        assert not coherent
        assert phi < 0.0

    def test_fourier_audit(self, engine):
        q = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        gradT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        chart = engine._factorize_metric(np.eye(3, dtype=np.float64))
        k, res, d = engine._fourier_audit(q, gradT, chart)
        assert abs(k - 2.0) < 1e-10
        assert abs(res) < 1e-10
        assert abs(d - 4.0) < 1e-10

    def test_calculate_spectral_irreversibility(self, engine):
        H = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=np.complex128)
        rho = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        irr = engine.calculate_spectral_irreversibility(rho, H)
        assert irr > 0.0

    def test_factorize_density_valid(self, engine):
        H = np.eye(3, dtype=np.float64)
        rho = np.eye(3, dtype=np.float64) / 3.0
        chart = engine._factorize_density(rho, H, 300.0)
        assert chart.is_density
        assert abs(chart.trace - 1.0) < 1e-10

    def test_factorize_density_invalid(self, engine):
        H = np.eye(3, dtype=np.float64)
        rho = np.eye(3, dtype=np.float64)
        chart = engine._factorize_density(rho, H, 300.0)
        assert not chart.is_density

    def test_phase1_observe_orient(self, engine):
        G = np.eye(3, dtype=np.float64)
        M = np.eye(3, dtype=np.float64)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gradT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        rho = np.eye(3, dtype=np.float64) / 3.0
        H = np.eye(3, dtype=np.float64)
        
        packet = engine._phase1_observe_orient(
            G=G, tensor_M=M, tensor_type="covariant",
            internal_energy_U=10.0, pressure_P=1.0, volume_V=1.0,
            temperature_T=300.0, entropy_S=0.1, entropy_production_rate=0.01,
            heat_flux_q=q, temp_gradient_gradT=gradT, rho=rho, hamiltonian_H=H
        )
        assert isinstance(packet, PhaseOneKBasePacket)
        assert packet.clausius_coherent
        assert abs(packet.temperature - 300.0) < 1e-10

class TestPhase2_Decide:
    pass

class TestPhase3_Act:
    pass
