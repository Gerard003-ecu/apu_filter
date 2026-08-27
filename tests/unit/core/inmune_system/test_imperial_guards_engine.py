# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.imperial_guards_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.imperial_guards_engine import (
    ImperialGuardsEngine,
    _SpectralTripleCertificate,
    _SpectralTripleGerm,
    _DiracSpectrumResult,
    _PetzMetricResult,
    _HodgeCheegerGerm,
    _LaplacianResult,
    _CheegerResult,
    _EulerPoincareResult,
    _MACHINE_EPS,
    _HIGHAM_TIKHONOV_FLOOR,
    _DEFAULT_REGULARIZER,
    _WILKINSON_DEFLATION_SCALE,
    _WILKINSON_DEFLATION_FLOOR,
    _WILKINSON_DRIFT_LIMIT,
    _PSD_ABS_TOL,
    _PSD_REL_TOL,
    _HERMITIAN_REL_TOL,
    _IMAGINARY_TOL,
    _LOG_MEAN_REL_TOL,
    _COMPLEX_STEP_DEFAULT_H,
    _COMPLEX_STEP_MIN_H,
    _COMPLEX_STEP_FD_FALLBACK,
    _LOG_EXP_CLIP,
    _SPECTRAL_DIM_MIN_MODES,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _HIGHAM_TIKHONOV_FLOOR > 0
        assert _DEFAULT_REGULARIZER > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _WILKINSON_DEFLATION_FLOOR > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _PSD_ABS_TOL > 0
        assert _PSD_REL_TOL > 0
        assert _HERMITIAN_REL_TOL > 0
        assert _IMAGINARY_TOL > 0
        assert _LOG_MEAN_REL_TOL > 0
        assert _COMPLEX_STEP_DEFAULT_H > 0
        assert _COMPLEX_STEP_MIN_H > 0
        assert _COMPLEX_STEP_FD_FALLBACK > 0
        assert _LOG_EXP_CLIP > 0
        assert _SPECTRAL_DIM_MIN_MODES > 0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_spectral_triple_certificate_frozen(self):
        cert = _SpectralTripleCertificate(0.0, 1.0, 0.1, 1.0, 0.0, 2, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.is_density = False

    def test_spectral_triple_germ_frozen(self):
        cert = _SpectralTripleCertificate(0.0, 1.0, 0.1, 1.0, 0.0, 2, True)
        germ = _SpectralTripleGerm(2, np.eye(2), np.ones(2), np.eye(2), 1e-15, 1e-20, cert)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            germ.dim = 3

class TestPhase1_NumericalFoundations:
    """Granular tests for Phase 1 methods."""
    def test_kahan_sum(self):
        engine = ImperialGuardsEngine()
        arr = np.array([1.0, 1e-16, -1.0], dtype=np.float64)
        assert abs(engine.kahan_sum(arr) - 1e-16) < 1e-15

    def test_compute_complex_step_gradient(self):
        engine = ImperialGuardsEngine()
        def f(x): return np.sum(x**2)
        x = np.array([1.0, 2.0], dtype=np.float64)
        grad = engine.compute_complex_step_gradient(f, x)
        assert np.allclose(grad, [2.0, 4.0])

class TestPhase2_SpectralQuantum:
    """Granular tests for Phase 2 methods."""
    def test_compute_dirac_operator_spectrum(self):
        engine = ImperialGuardsEngine()
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.float64)
        dirac, evals = engine.compute_dirac_operator_spectrum(rho)
        assert dirac.shape == (2,)
        assert evals.shape == (2,)
        assert np.all(evals >= 0)

    def test_compute_petz_fisher_rao_metric(self):
        engine = ImperialGuardsEngine()
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.float64)
        A = np.array([[1.0, 0.5], [0.5, 0.0]], dtype=np.float64)
        val = engine.compute_petz_fisher_rao_metric(rho, A, A)
        assert val >= 0

class TestPhase3_TopologicalGeometric:
    """Granular tests for Phase 3 methods."""
    def test_compute_simplicial_normalized_laplacian(self):
        engine = ImperialGuardsEngine()
        delta0 = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
        L = engine.compute_simplicial_normalized_laplacian(delta0)
        assert L.shape == (3, 3)
        evals = np.linalg.eigvalsh(L)
        assert abs(evals[0]) < 1e-8

    def test_estimate_cheeger_constant_bounds(self):
        engine = ImperialGuardsEngine()
        evals = np.array([0.0, 0.5, 1.5], dtype=np.float64)
        h_min, h_max = engine.estimate_cheeger_constant_bounds(evals)
        assert h_min == 0.25
        assert h_max == 1.0

    def test_compute_euler_poincare_characteristic(self):
        engine = ImperialGuardsEngine()
        delta0 = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
        chi = engine.compute_euler_poincare_characteristic(delta0, None)
        assert isinstance(chi, int)

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization(self):
        engine = ImperialGuardsEngine()
        assert engine.regularizer >= _HIGHAM_TIKHONOV_FLOOR
