# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para kapex_electrodynamic_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.alfa.kapex.kapex_electrodynamic_engine import (
    KapexElectrodynamicEngine,
    HeytingVerdict,
    SpectralChart,
    PhaseOneKapexPacket,
    PhaseTwoKapexPacket,
    KapexTelemetry,
    SpectralIntegrityReport
)
from app.core.mic_algebra import Morphism

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_eps(self):
        from app.alfa.kapex.kapex_electrodynamic_engine import _MACHINE_EPS
        assert _MACHINE_EPS > 0.0

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

class TestPhase1_ObserveOrient:
    @pytest.fixture
    def engine(self):
        return KapexElectrodynamicEngine(dimension_n=3)

    def test_engine_init_invalid_dim(self):
        with pytest.raises(ValueError):
            KapexElectrodynamicEngine(dimension_n=0)

    def test_kahan_sum_valid(self, engine):
        arr = np.array([1e16, 1.0, -1e16], dtype=np.float64)
        res = engine.kahan_sum(arr)
        assert abs(res - 1.0) < 1e-10

    def test_project_to_so_n(self, engine):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        pi_A, res = engine.project_to_so_n(A)
        assert res > 0.0
        np.testing.assert_allclose(pi_A + pi_A.T, np.zeros((3, 3)))

    def test_killing_pairing(self, engine):
        X = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        Y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        val = engine.killing_pairing(X, Y)
        # X @ Y = [[-1, 0, 0], [0, -1, 0], [0, 0, 0]], Tr = -2. n=3 => (3-2)*(-2) = -2
        assert abs(val - (-2.0)) < 1e-10

    def test_factorize_metric_spd(self, engine):
        M = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)
        chart = engine._factorize_metric(M)
        assert not chart.regularized
        np.testing.assert_allclose(chart.eigenvalues, [2.0, 3.0, 4.0])

    def test_solve_eikonal_residual(self, engine):
        grad_S = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        M = np.eye(3, dtype=np.float64)
        res, p_up = engine.solve_eikonal_residual(grad_S, M, refraction_index=1.0)
        assert abs(res) < 1e-10
        np.testing.assert_allclose(p_up, [1.0, 0.0, 0.0])

    def test_compute_yang_mills_action(self, engine):
        A0 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        A2 = np.zeros((3, 3))
        M = np.eye(3, dtype=np.float64)
        F, action = engine.compute_yang_mills_action([A0, A1, A2], M)
        assert action >= 0.0
        assert F.shape == (3, 3, 3, 3)

    def test_compute_poynting_strategic_flux(self, engine):
        E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        B = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        M = np.eye(3, dtype=np.float64)
        S, u = engine.compute_poynting_strategic_flux(E, B, M)
        np.testing.assert_allclose(S, [0.0, 0.0, 1.0])
        assert abs(u - 1.0) < 1e-10

    def test_compute_maxwell_stress_energy(self, engine):
        E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        B = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        chart = engine._factorize_metric(np.eye(3, dtype=np.float64))
        T, two_form = engine.compute_maxwell_stress_energy(E, B, chart)
        assert T.shape == (3, 3)
        assert two_form.shape == (3, 3)
