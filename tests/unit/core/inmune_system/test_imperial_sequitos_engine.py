# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.imperial_sequitos_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.imperial_sequitos_engine import (
    ImperialSequitosEngine,
    _MarkovKernelCertificate,
    _MarkovKleisliGerm,
    _DeGrootConsensusResult,
    _UhlmannFidelityResult,
    _BellCorrelationGerm,
    _CHSHResult,
    _KleisliComposer,
    _NumericalCore,
    _DeGrootConsensus,
    _UhlmannFidelity,
    _CHSHVerifier,
    _MACHINE_EPS,
    _HIGHAM_TIKHONOV_REG,
    _WILKINSON_DEFLATION_FLOOR,
    _WILKINSON_DEFLATION_SCALE,
    _WILKINSON_DRIFT_LIMIT,
    _TOLERANCE_DEGROOT_COHERENT,
    _TOLERANCE_DEGROOT_DEGRADED,
    _TSIRELSON_BOUND,
    _CLASSICAL_CHSH_BOUND,
    _PR_NOSIGNAL_BOUND,
    _LOG_EXP_CLIP,
    _CORRELATOR_BOUND,
    _PAULI,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _HIGHAM_TIKHONOV_REG > 0
        assert _WILKINSON_DEFLATION_FLOOR > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _TOLERANCE_DEGROOT_COHERENT > 0
        assert _TOLERANCE_DEGROOT_DEGRADED > 0
        assert _TSIRELSON_BOUND > 0
        assert _CLASSICAL_CHSH_BOUND > 0
        assert _PR_NOSIGNAL_BOUND > 0
        assert _LOG_EXP_CLIP > 0
        assert _CORRELATOR_BOUND > 0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_markov_kernel_certificate_frozen(self):
        cert = _MarkovKernelCertificate(0.0, 1.0, 0.0, True, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.is_stochastic = False

    def test_chsh_result_frozen(self):
        res = _CHSHResult(2.0, "DEGRADED", 0.0, 0.0, 0.0, True, 2.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            res.s_value = 3.0

class TestPhase1_NumericalCore:
    """Granular tests for Phase 1 methods."""
    def test_kahan_sum(self):
        engine = ImperialSequitosEngine()
        arr = np.array([1.0, 1e-16, -1.0], dtype=np.float64)
        assert abs(engine.kahan_sum(arr) - 1e-16) < 1e-15

    def test_kleisli_compose(self):
        engine = ImperialSequitosEngine()
        def f(x): return x + 1, 0.8
        def g(x): return x * 2, 0.5
        comp = engine.kleisli_compose(f, g)
        val, prob = comp(2)
        assert val == 6
        assert abs(prob - 0.4) < 1e-8

    def test_kleisli_unit(self):
        engine = ImperialSequitosEngine()
        val, prob = engine.kleisli_unit(5)
        assert val == 5
        assert prob == 1.0

    def test_compose_markov_kernels(self):
        engine = ImperialSequitosEngine()
        P = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float64)
        Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        R = engine.compose_markov_kernels(P, Q)
        assert np.allclose(R, P)

class TestPhase2_ConsensusUhlmann:
    """Granular tests for Phase 2 methods."""
    def test_compute_degroot_spectral_consensus(self):
        engine = ImperialSequitosEngine()
        op = np.array([1.0, 0.0], dtype=np.float64)
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        final, fiedler, verdict = engine.compute_degroot_spectral_consensus(op, A, 100)
        assert final.shape == (2,)
        assert fiedler >= 0
        assert isinstance(verdict, str)

    def test_compute_uhlmann_fidelity(self):
        engine = ImperialSequitosEngine()
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        sigma = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        fid = engine.compute_uhlmann_fidelity(rho, sigma)
        assert abs(fid - 1.0) < 1e-8

class TestPhase3_CHSH:
    """Granular tests for Phase 3 methods."""
    def test_verify_chsh_violation(self):
        engine = ImperialSequitosEngine()
        E = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=np.float64)
        s, verdict = engine.verify_chsh_violation(E)
        assert abs(s - 1.0) < 1e-8
        assert verdict == "DEGRADED"

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization(self):
        engine = ImperialSequitosEngine()
        assert engine._reg >= _HIGHAM_TIKHONOV_REG
