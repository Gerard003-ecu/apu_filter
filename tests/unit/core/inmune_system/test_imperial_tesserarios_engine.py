# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.imperial_tesserarios_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.imperial_tesserarios_engine import (
    ImperialTesserariosEngine,
    _SymplecticFormCertificate,
    _SymplecticQuillenGerm,
    _SymplecticProjectionResult,
    _QuillenFactorizationResult,
    _AInfinityCechGerm,
    _StasheffAssociatorResult,
    _CechObstructionResult,
    _NumericalCore,
    _SymplecticProjector,
    _StasheffAssociator,
    _CechObstructionCalculator,
    _MACHINE_EPS,
    _REG_FLOOR_TIKHONOV,
    _WILKINSON_DEFLATION_SCALE,
    _WILKINSON_DEFLATION_FLOOR,
    _WILKINSON_DRIFT_LIMIT,
    _DEFAULT_MAX_ITER,
    _DEFAULT_TOL,
    _STASHEFF_PENTAGON_CAP,
    _CECH_TRIPLE_CAP,
    _CECH_QUAD_CAP,
    _MU_CLIP,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _REG_FLOOR_TIKHONOV > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _WILKINSON_DEFLATION_FLOOR > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _DEFAULT_MAX_ITER > 0
        assert _DEFAULT_TOL > 0
        assert _STASHEFF_PENTAGON_CAP > 0
        assert _CECH_TRIPLE_CAP > 0
        assert _CECH_QUAD_CAP > 0
        assert len(_MU_CLIP) == 2

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_symplectic_form_certificate_frozen(self):
        cert = _SymplecticFormCertificate(0.0, 0.0, 1.0, 1.0, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.is_darboux = False

    def test_symplectic_quillen_germ_frozen(self):
        cert = _SymplecticFormCertificate(0.0, 0.0, 1.0, 1.0, True)
        germ = _SymplecticQuillenGerm(2, 1, np.eye(2), 1e-15, 100, 1e-12, cert)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            germ.two_n = 4

class TestPhase1_NumericalCore:
    """Granular tests for Phase 1 methods."""
    def test_kahan_sum(self):
        engine = ImperialTesserariosEngine()
        arr = np.array([1.0, 1e-16, -1.0], dtype=np.float64)
        assert abs(engine.kahan_sum(arr) - 1e-16) < 1e-15

    def test_generate_canonical_symplectic_form(self):
        engine = ImperialTesserariosEngine()
        omega = engine.generate_canonical_symplectic_form(2)
        assert np.allclose(omega, [[0.0, 1.0], [-1.0, 0.0]])

    def test_symplectic_quillen_germ_certificate(self):
        engine = ImperialTesserariosEngine()
        cert = engine.symplectic_quillen_germ_certificate()
        assert cert.is_darboux

class TestPhase2_PolarQuillen:
    """Granular tests for Phase 2 methods."""
    def test_project_to_symplectic_group(self):
        engine = ImperialTesserariosEngine()
        M = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64)
        S, res = engine.project_to_symplectic_group(M)
        assert S.shape == (2, 2)
        assert res < 1e-8
        omega = engine.generate_canonical_symplectic_form(2)
        assert np.allclose(S.T @ omega @ S, omega)

    def test_compute_quillen_factorization(self):
        engine = ImperialTesserariosEngine()
        M = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64)
        fibration, cofibration, res = engine.compute_quillen_factorization(M)
        assert fibration.shape == (2, 2)
        assert cofibration.shape == (2, 2)
        assert res >= 0

class TestPhase3_StasheffCech:
    """Granular tests for Phase 3 methods."""
    def test_compute_stasheff_m3_associator(self):
        engine = ImperialTesserariosEngine()
        m2 = np.zeros((2, 2, 2), dtype=np.float64)
        m3 = engine.compute_stasheff_m3_associator(m2)
        assert m3.shape == (2, 2, 2, 2)
        assert np.all(m3 == 0)

    def test_compute_cech_hypercohomology_gerbe(self):
        engine = ImperialTesserariosEngine()
        cochain = np.eye(2, dtype=np.float64)
        val, sv = engine.compute_cech_hypercohomology_gerbe(cochain)
        assert val >= 0
        assert sv.shape == (2,)

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization(self):
        engine = ImperialTesserariosEngine()
        assert engine._reg >= _REG_FLOOR_TIKHONOV
