# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para imperial_guards_agent."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_agent import (
    Phase1SpectralObservation,
    Phase2LogisticObservation,
    Phase3TribunalDecision,
    ImperialGuardsCertificate,
    Phase1SpectralGuardianMixin,
    Phase2LogisticGuardianMixin,
    ImperialGuardsAgent,
    _WILKINSON_DRIFT_LIMIT,
    _DEFAULT_CHEEGER_THRESHOLD
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_positive(self):
        assert _WILKINSON_DRIFT_LIMIT > 0.0
        assert _DEFAULT_CHEEGER_THRESHOLD > 0.0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_phase1_observation_frozen(self):
        obs = Phase1SpectralObservation(
            dirac_spectrum_size=2,
            lambda_min_dirac=1.0,
            lipschitz_coefficient=0.5,
            partial_verdict="COHERENT",
            veto_reasons=(),
            degraded_reasons=(),
            diagnostics={}
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obs.dirac_spectrum_size = 3

    def test_imperial_guards_certificate_frozen(self):
        cert = ImperialGuardsCertificate(
            phase="Phase3",
            heyting_verdict="COHERENT",
            lipschitz_coefficient=0.5,
            dirac_spectral_gap=1.0,
            fiedler_connectivity=0.5,
            cheeger_lower_bound=0.125,
            pyramidal_stability=0.5,
            cohomological_residual=0.0,
            hardware_interlock_fired=False,
            actuation_latency_ns=0.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.hardware_interlock_fired = True

class TestPhase1SpectralGuardianMixin:
    """Granular tests for Phase 1 methods."""
    def test_initialization(self):
        mixin = Phase1SpectralGuardianMixin(4)
        assert mixin._n == 4

    def test_kahan_compensated_sum(self):
        mixin = Phase1SpectralGuardianMixin(2)
        terms = np.array([1.0, 1e-16, 1e-16], dtype=np.float64)
        res = mixin.kahan_compensated_sum(terms)
        assert res >= 1.0

    def test_compute_lipschitz_coefficient(self):
        mixin = Phase1SpectralGuardianMixin(2)
        lam = 1.0
        coeff = mixin._compute_lipschitz_coefficient(lam)
        assert abs(coeff - 0.5) < 1e-9

        # invalid lambda
        assert math.isinf(mixin._compute_lipschitz_coefficient(0.0))
        assert math.isinf(mixin._compute_lipschitz_coefficient(-1.0))

    def test_phase1_audit_spectral_heterogeomorphic_curve(self):
        mixin = Phase1SpectralGuardianMixin(2)
        eig = np.array([1.0, 2.0], dtype=np.float64)
        obs = mixin.phase1_audit_spectral_heterogeomorphic_curve(eig)
        assert obs.dirac_spectrum_size == 2
        assert obs.partial_verdict == "COHERENT"

    def test_phase1_audit_invalid(self):
        mixin = Phase1SpectralGuardianMixin(2)
        obs = mixin.phase1_audit_spectral_heterogeomorphic_curve(np.array([], dtype=np.float64))
        assert obs.partial_verdict == "VETOED"

class TestPhase2LogisticGuardianMixin:
    """Granular tests for Phase 2 methods."""
    def test_fiedler_gap(self):
        mixin = Phase2LogisticGuardianMixin(4)
        eigs = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
        gap, diag = mixin._fiedler_gap(eigs)
        assert abs(gap - 0.5) < 1e-9

    def test_phase2_audit_logistic_from_phase1(self):
        mixin = Phase2LogisticGuardianMixin(4)
        eigs = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
        obs = mixin.phase2_audit_logistic_from_phase1(
            phase1_observation=None,
            eigenvalues_L=eigs,
            betti_0=1,
            betti_1=0
        )
        assert obs.partial_verdict == "COHERENT"
        assert abs(obs.fiedler_connectivity - 0.5) < 1e-9

class TestImperialGuardsAgent:
    """Integration tests for the main class."""
    def test_execute_full_audit(self):
        agent = ImperialGuardsAgent(4)
        dirac_eigs = np.array([1.0, 2.0], dtype=np.float64)
        laplacian_eigs = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
        
        cert = agent.execute_sovereign_governance(
            eigenvalues_dirac=dirac_eigs,
            eigenvalues_L=laplacian_eigs,
            betti_0=1,
            betti_1=0
        )
        assert cert.heyting_verdict == "COHERENT"
        assert cert.hardware_interlock_fired is False

    def test_execute_full_audit_vetoed(self):
        agent = ImperialGuardsAgent(4)
        dirac_eigs = np.array([-1.0], dtype=np.float64)
        laplacian_eigs = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
        
        cert = agent.execute_sovereign_governance(
            eigenvalues_dirac=dirac_eigs,
            eigenvalues_L=laplacian_eigs,
            betti_0=1,
            betti_1=0
        )
        assert cert.heyting_verdict == "VETOED"
        assert cert.hardware_interlock_fired is True
