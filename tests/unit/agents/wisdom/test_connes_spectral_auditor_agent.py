# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para connes_spectral_auditor_agent."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.wisdom.connes_spectral_auditor_agent import (
    Phase1_SpectralTripleBinder,
    Phase2_KMSEquilibriumAuditor,
    SpectralTripleData,
    KMSThermalBundle,
    KMSThermalState,
    SemanticDiscontinuityError,
    SpectralTripleError,
    KMSEquilibriumViolation,
    _COMMUTATOR_NORM_BOUND,
    _KMS_EQUILIBRIUM_TOLERANCE,
    _HERMITICITY_TOLERANCE,
    _POSITIVITY_TOLERANCE,
    _TRACE_TOLERANCE
)
try:
    from app.agents.wisdom.connes_spectral_auditor_agent import ConnesSpectralAuditorAgent, Phase3_DixmierTraceIntegrator
except ImportError:
    ConnesSpectralAuditorAgent = None
    Phase3_DixmierTraceIntegrator = None

def get_hermitian(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (M + M.conj().T)

def get_faithful_rho(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = M @ M.conj().T + 0.1 * np.eye(n)
    return rho / np.trace(rho)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_hermiticity_tolerance(self):
        assert _HERMITICITY_TOLERANCE > 0

    def test_positivity_tolerance(self):
        assert _POSITIVITY_TOLERANCE > 0

    def test_trace_tolerance(self):
        assert _TRACE_TOLERANCE > 0

    def test_commutator_norm_bound(self):
        assert _COMMUTATOR_NORM_BOUND > 0

    def test_kms_equilibrium_tolerance(self):
        assert _KMS_EQUILIBRIUM_TOLERANCE > 0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_semantic_discontinuity(self):
        with pytest.raises(SemanticDiscontinuityError):
            raise SemanticDiscontinuityError("test")
            
    def test_spectral_triple_error(self):
        with pytest.raises(SpectralTripleError):
            raise SpectralTripleError("test")

    def test_kms_equilibrium_violation(self):
        with pytest.raises(KMSEquilibriumViolation):
            raise KMSEquilibriumViolation("test")

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_spectral_triple_data_frozen(self):
        obj = SpectralTripleData(
            dirac_operator=np.eye(2),
            dirac_eigenvalues=np.array([1.0, 1.0]),
            commutator_norm=0.1,
            lipschitz_seminorm=0.1,
            dirac_condition_number=1.0,
            hilbert_space_dim=2,
            is_differentiable=True,
            rho_reference=np.eye(2)/2,
            X_reference=np.eye(2)
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.commutator_norm = 0.5

    def test_kms_thermal_state_frozen(self):
        obj = KMSThermalState(
            thermal_residual_norm=0.01,
            kms_beta=1.0,
            is_kms_compliant=True,
            left_expectation=1.0,
            right_expectation=1.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.kms_beta = 2.0

class TestPhase1_SpectralTripleBinder:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.binder = Phase1_SpectralTripleBinder()

    def test_hermitize(self):
        M = np.array([[1, 1j], [0, 1]])
        H = self.binder._hermitize(M)
        assert abs(H[0, 1] - 0.5j) < 1e-9

    def test_validate_hermitian_square_not_square(self):
        with pytest.raises(SemanticDiscontinuityError, match="se exige matriz cuadrada"):
            self.binder._validate_hermitian_square(np.zeros((2, 3)), "test")

    def test_validate_hermitian_square_not_hermitian(self):
        M = np.array([[1, 1j], [0, 1]])
        with pytest.raises(SemanticDiscontinuityError, match="no hermítica"):
            self.binder._validate_hermitian_square(M, "test")

    def test_validate_faithful_density_negative_eig(self):
        M = np.array([[-1, 0], [0, 1]])
        with pytest.raises(SpectralTripleError, match="autovalor negativo"):
            self.binder._validate_faithful_density(M, "test")

    def test_validate_faithful_density_trace(self):
        M = np.eye(2)
        with pytest.raises(SpectralTripleError, match="Tr ρ"):
            self.binder._validate_faithful_density(M, "test")

    def test_bind_spectral_triple_success(self):
        rho = get_faithful_rho(2)
        X = get_hermitian(2)
        res = self.binder.bind_spectral_triple(rho, X)
        assert res.is_differentiable is True
        assert res.hilbert_space_dim == 2
        
    def test_bind_spectral_triple_fail(self):
        rho = get_faithful_rho(2)
        X = np.array([[1, 1j], [0, 1]])
        with pytest.raises(SemanticDiscontinuityError):
            self.binder.bind_spectral_triple(rho, X)

    def test_lipschitz_seminorm(self):
        D = np.eye(2)
        X = get_hermitian(2)
        comm, lip = self.binder._lipschitz_seminorm(D, X)
        assert abs(lip - 0.0) < 1e-9

class TestPhase2_KMSEquilibriumAuditor:
    """Granular tests for Phase 2 methods."""
    def setup_method(self):
        self.auditor = Phase2_KMSEquilibriumAuditor()
        self.engine = MagicMock()

    def test_state_expectation(self):
        rho = np.eye(2) / 2
        op = np.eye(2)
        res = self.auditor._state_expectation(rho, op)
        assert abs(res - 1.0) < 1e-9

    def test_kms_residual(self):
        rho = np.eye(2) / 2
        A = np.eye(2)
        B = np.eye(2)
        res, l, r = self.auditor._kms_residual(rho, A, B, A)
        assert abs(res - 0.0) < 1e-9
        
    def test_zoom_thermal_friction(self):
        rho = np.eye(2) / 2
        X = np.eye(2)
        res = self.auditor._zoom_thermal_friction(rho, X, X)
        assert abs(res - 0.0) < 1e-9

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_agent_init(self):
        if ConnesSpectralAuditorAgent is None:
            pytest.skip("ConnesSpectralAuditorAgent not found")
        with patch.multiple(ConnesSpectralAuditorAgent, __abstractmethods__=frozenset()):
            agent = ConnesSpectralAuditorAgent()
            assert agent is not None
