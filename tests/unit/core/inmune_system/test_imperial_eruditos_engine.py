# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.imperial_eruditos_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.imperial_eruditos_engine import (
    ImperialEruditosEngine,
    _SymplecticFormCertificate,
    _FloerCylinderGerm,
    _FloerResult,
    _CechNerveGerm,
    _CechCohomologyResult,
    _NumericalCore,
    _FloerHomologyVerifier,
    _AttentionCechCohomology,
    _MACHINE_EPS,
    _HIGHAM_TIKHONOV_FLOOR,
    _WILKINSON_DEFLATION_FLOOR,
    _WILKINSON_DEFLATION_SCALE,
    _WILKINSON_DRIFT_LIMIT,
    _CSMD_STEP,
    _CSMD_FD_FALLBACK,
    _CECH_TRIPLE_CAP,
    _LOG_EXP_CLIP,
    _MASLOV_DEGENERACY,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _HIGHAM_TIKHONOV_FLOOR > 0
        assert _WILKINSON_DEFLATION_FLOOR > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _CSMD_STEP > 0
        assert _CSMD_FD_FALLBACK > 0
        assert _CECH_TRIPLE_CAP > 0
        assert _LOG_EXP_CLIP > 0
        assert _MASLOV_DEGENERACY > 0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_symplectic_form_certificate_frozen(self):
        cert = _SymplecticFormCertificate(0.1, 0.2, 1.0, 1.0, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.is_darboux = False

    def test_floer_cylinder_germ_frozen(self):
        cert = _SymplecticFormCertificate(0.0, 0.0, 1.0, 1.0, True)
        germ = _FloerCylinderGerm(2, 1, np.eye(2), 1e-20, 1e-15, cert)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            germ.two_n = 4

class TestPhase1_NumericalCore:
    """Granular tests for Phase 1 methods."""
    def test_kahan_sum(self):
        arr = np.array([1.0, 1e-16, -1.0], dtype=np.float64)
        assert abs(_NumericalCore.kahan_sum(arr) - 1e-16) < 1e-15

    def test_generate_canonical_symplectic_form(self):
        omega = _NumericalCore.generate_canonical_symplectic_form(4)
        assert omega.shape == (4, 4)
        assert np.allclose(omega @ omega, -np.eye(4))

    def test_compute_gradient_csmd(self):
        def f(x): return np.sum(x**2)
        x = np.array([1.0, 2.0], dtype=np.float64)
        grad = _NumericalCore.compute_gradient_csmd(f, x)
        assert np.allclose(grad, [2.0, 4.0])

    def test_compute_symplectic_gradient(self):
        def f(x): return np.sum(x**2)
        x = np.array([1.0, 2.0], dtype=np.float64)
        grad = _NumericalCore.compute_symplectic_gradient(f, x)
        assert grad.shape == (2,)

class TestPhase2_Floer:
    """Granular tests for Phase 2 methods."""
    def test_verify_floer_homology_trajectory(self):
        engine = ImperialEruditosEngine()
        z0 = np.array([0.0, 0.0], dtype=np.float64)
        z1 = np.array([1.0, 1.0], dtype=np.float64)
        M = np.eye(2, dtype=np.float64)
        residual, action = engine.verify_floer_homology_trajectory(z0, z1, M)
        assert residual >= 0
        assert action >= 0

class TestPhase3_Cech:
    """Granular tests for Phase 3 methods."""
    def test_compute_attention_cech_cohomology(self):
        engine = ImperialEruditosEngine()
        A = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        obs, modes = engine.compute_attention_cech_cohomology(A)
        assert obs >= 0
        assert len(modes) > 0

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization_invalid(self):
        engine = ImperialEruditosEngine(-1.0)
        assert engine._reg == _HIGHAM_TIKHONOV_FLOOR
