# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.pretorio_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.pretorio_engine import (
    PretorioEngine,
    _HypercohomologyGerm,
    _HypercohomologyResult,
    _BrouwerResult,
    _UltrafilterGerm,
    _UltrafilterResult,
    _NumericalCore,
    _HypercohomologyChecker,
    _BrouwerChecker,
    _UltrafilterEvaluator,
    _MACHINE_EPS,
    _WILKINSON_DEFLATION_FLOOR,
    _HIGHAM_TIKHONOV_REG,
    _WILKINSON_DEFLATION_SCALE,
    _WILKINSON_DRIFT_LIMIT,
    _BROUWER_VETO_HS,
    _BROUWER_DEGRADED_HS,
    _BROUWER_VETO_TRACE,
    _BROUWER_DEGRADED_TRACE,
    _HYPER_VETO_SCALE,
    _PSD_NEG_TOL,
    _HEYTING_ORDER,
    _HEYTING_GODEL,
    _CANONICAL_VERDICTS,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _WILKINSON_DEFLATION_FLOOR > 0
        assert _HIGHAM_TIKHONOV_REG > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _BROUWER_VETO_HS > 0
        assert _BROUWER_DEGRADED_HS > 0
        assert _BROUWER_VETO_TRACE > 0
        assert _BROUWER_DEGRADED_TRACE > 0
        assert _HYPER_VETO_SCALE > 0
        assert _PSD_NEG_TOL > 0
        assert "COHERENT" in _HEYTING_ORDER

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_hypercohomology_germ_frozen(self):
        germ = _HypercohomologyGerm(np.eye(2), np.eye(2), True, True, True, True, np.eye(2), 1e-15, 1.0, 1.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            germ.reg_floor = 1e-12

    def test_ultrafilter_result_frozen(self):
        res = _UltrafilterResult("VIABLE", False, 1, 1, 0, 0, 1.0, 1.0, 1.0, 0.0, True, "COHERENT")
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            res.consensus = "RECHAZAR"

class TestPhase1_NumericalCore:
    """Granular tests for Phase 1 methods."""
    def test_kahan_compensated_trace(self):
        engine = PretorioEngine()
        M = np.array([[1.0, 0.5], [0.5, 2.0]], dtype=np.float64)
        assert engine.kahan_compensated_trace(M) == 3.0

    def test_weyl_toeplitz_symmetrization(self):
        engine = PretorioEngine()
        M = np.array([[1.0, 1.0j], [-1.0j, 1.0]], dtype=np.complex128)
        sym = engine.weyl_toeplitz_symmetrization(M)
        assert np.allclose(sym, M)

    def test_higham_nearest_density(self):
        engine = PretorioEngine()
        M = np.array([[1.1, 0.0], [0.0, -0.1]], dtype=np.float64)
        rho = engine.higham_nearest_density(M)
        assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)
        assert abs(np.trace(rho) - 1.0) < 1e-10

class TestPhase2_HypercohomologyBrouwer:
    """Granular tests for Phase 2 methods."""
    def test_verify_cech_derham_hypercohomology(self):
        engine = PretorioEngine()
        d1 = np.zeros((2, 2), dtype=np.float64)
        d2 = np.zeros((2, 2), dtype=np.float64)
        res, verdict = engine.verify_cech_derham_hypercohomology(d1, d2)
        assert res == 0.0
        assert verdict == "COHERENT"

    def test_verify_brouwer_fixed_point(self):
        engine = PretorioEngine()
        rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
        b_res, t_res, verdict = engine.verify_brouwer_fixed_point(rho, rho)
        assert b_res < 1e-10
        assert t_res < 1e-10
        assert verdict == "COHERENT"

class TestPhase3_Ultrafilter:
    """Granular tests for Phase 3 methods."""
    def test_evaluate_ultrafilter_consensus_coherent(self):
        engine = PretorioEngine()
        verdicts = ["COHERENT", "COHERENT"]
        consensus, fired = engine.evaluate_ultrafilter_consensus(verdicts)
        assert consensus == "VIABLE"
        assert fired is False

    def test_evaluate_ultrafilter_consensus_vetoed(self):
        engine = PretorioEngine()
        verdicts = ["COHERENT", "VETOED"]
        consensus, fired = engine.evaluate_ultrafilter_consensus(verdicts)
        assert consensus == "RECHAZAR"
        assert fired is True

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization(self):
        engine = PretorioEngine()
        assert engine._reg >= _HIGHAM_TIKHONOV_REG
