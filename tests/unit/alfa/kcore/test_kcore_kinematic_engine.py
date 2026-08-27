# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para kcore_kinematic_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.alfa.kcore.kcore_kinematic_engine import (
    KCoreKinematicEngine,
    HeytingVerdict,
    SpectralChart,
    PortHamiltonianChart,
    HodgeChart,
    KineticChart,
    PhaseOneKinematicPacket,
    PhaseTwoKinematicPacket,
    KinematicTelemetry,
    SpectralKinematicReport
)
from app.core.mic_algebra import Morphism

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_eps(self):
        from app.alfa.kcore.kcore_kinematic_engine import _MACHINE_EPS
        assert _MACHINE_EPS > 0.0

    def test_wilkinson_floor(self):
        from app.alfa.kcore.kcore_kinematic_engine import _WILKINSON_FLOOR
        assert _WILKINSON_FLOOR > 0.0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_spectral_chart_frozen(self):
        chart = SpectralChart(
            eigenvalues=np.array([1.0], dtype=np.float64),
            eigenvectors=np.eye(1, dtype=np.float64),
            inv_metric=np.eye(1, dtype=np.float64),
            sqrt_metric=np.eye(1, dtype=np.float64),
            condition_number=1.0, operator_norm=1.0,
            frobenius_norm=1.0, spectral_gap=1.0, regularized=False, 
            tikhonov_alpha=1e-15, kernel_dimension=0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            chart.condition_number = 2.0

class TestPhase1_ObserveOrient:
    @pytest.fixture
    def engine(self):
        return KCoreKinematicEngine(dimension_n=3)

    def test_engine_init_invalid_dim(self):
        with pytest.raises(ValueError):
            KCoreKinematicEngine(dimension_n=0)

    def test_kahan_sum_valid(self, engine):
        arr = np.array([1e16, 1.0, -1e16], dtype=np.float64)
        res = engine.kahan_sum(arr)
        assert abs(res - 1.0) < 1e-10

    def test_skew_project(self, engine):
        M = np.array([[1.0, 2.0], [0.0, 3.0]])
        proj, res = engine._skew_project(M)
        np.testing.assert_allclose(proj, np.array([[0.0, 1.0], [-1.0, 0.0]]))
        assert res > 0.0

    def test_psd_project(self, engine):
        M = np.array([[1.0, 0.0], [0.0, -1.0]])
        proj, res = engine._psd_project(M)
        np.testing.assert_allclose(proj, np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert res > 0.0

    def test_factorize_metric_spd(self, engine):
        M = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)
        chart = engine._factorize_metric(M)
        assert not chart.regularized
        np.testing.assert_allclose(chart.eigenvalues, [2.0, 3.0, 4.0])
        assert abs(chart.condition_number - 2.0) < 1e-10

    def test_tikhonov_svd_solve(self, engine):
        A = np.eye(3, dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x, sigma, res, rank, cond = engine._tikhonov_svd_solve(A, b, reg=1e-15)
        np.testing.assert_allclose(x, b)
        assert rank == 3
        assert abs(cond - 1.0) < 1e-10

    def test_compute_ida_pbc_matching(self, engine):
        g = np.eye(3, dtype=np.float64)
        G = np.eye(3, dtype=np.float64)
        J = np.zeros((3, 3), dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        Jd = np.zeros((3, 3), dtype=np.float64)
        Rd = np.eye(3, dtype=np.float64) * 2.0
        grad_H = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        grad_Hd = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        alpha, res = engine.compute_ida_pbc_matching(
            g, G, J, R, Jd, Rd, grad_H, grad_Hd
        )
        assert abs(res) < 1e-10
        # g alpha = (Jd-Rd)grad_Hd - (J-R)grad_H
        # F = (-2I)(1) - (-1I)(1) = -2 + 1 = -1
        np.testing.assert_allclose(alpha, [-1.0, -1.0, -1.0], atol=1e-10)

    def test_compute_helmholtz_hodge_decomposition(self, engine):
        B0 = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
        w = np.array([1.0, 1.0], dtype=np.float64)
        I = np.array([1.0, 1.0], dtype=np.float64)
        
        grad, curl, harm, err = engine.compute_helmholtz_hodge_decomposition(B0, w, I)
        assert grad.shape == (2,)
        assert curl.shape == (2,)
        assert harm.shape == (2,)
        assert err >= 0.0
