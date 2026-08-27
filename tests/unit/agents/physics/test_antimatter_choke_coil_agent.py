# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para antimatter_choke_coil_agent.py."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.physics.antimatter_choke_coil_agent import (
    _MACHINE_EPSILON, _HERMITICITY_TOLERANCE, _BEKENSTEIN_ABS_TOLERANCE,
    VacuumCustodianError, DomainIntegrityViolationError, NonHermitianOperatorError,
    SpectralContaminationError, BekensteinLimitViolation, CausalityViolationError,
    SymplecticCollapseError, PhaseSpaceTopologyError,
    SpectralDecompositionData, HermiticityAuditData, BekensteinBoundData,
    SymplecticDissipationData, Phase1HermiticityHandoff, Phase2BekensteinHandoff,
    VacuumGovernanceState,
    Phase1_HermiticityAuditor, Phase2_BekensteinBoundEnforcer,
    Phase3_SymplecticPortHamiltonianCertifier, AntimatterChokeCoilAgent
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_epsilon(self):
        assert _MACHINE_EPSILON > 0.0

    def test_hermiticity_tolerance(self):
        assert _HERMITICITY_TOLERANCE > 0.0

    def test_bekenstein_tolerance(self):
        assert _BEKENSTEIN_ABS_TOLERANCE > 0.0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_domain_integrity_violation_error(self):
        assert issubclass(DomainIntegrityViolationError, VacuumCustodianError)

    def test_non_hermitian_operator_error(self):
        assert issubclass(NonHermitianOperatorError, VacuumCustodianError)

    def test_spectral_contamination_error(self):
        assert issubclass(SpectralContaminationError, VacuumCustodianError)
        
    def test_bekenstein_limit_violation(self):
        assert issubclass(BekensteinLimitViolation, VacuumCustodianError)

    def test_symplectic_collapse_error(self):
        assert issubclass(SymplecticCollapseError, VacuumCustodianError)

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_spectral_decomposition_data_frozen(self):
        obj = SpectralDecompositionData(
            eigenvalues_real=np.array([1.0], dtype=np.float64),
            eigenvalues_imaginary_norm=0.0,
            spectral_radius=1.0,
            trace_real=1.0,
            trace_imaginary_norm=0.0,
            condition_number=1.0,
            is_spectrally_clean=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.spectral_radius = 2.0
            
    def test_hermiticity_audit_data_frozen(self):
        obj = HermiticityAuditData(
            residual_norm=0.0,
            is_hermitian=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.is_hermitian = False
            
    def test_bekenstein_bound_data_frozen(self):
        obj = BekensteinBoundData(
            entropy_emitted=1.0,
            bekenstein_bound=2.0,
            is_entropically_safe=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.is_entropically_safe = False
            
    def test_symplectic_dissipation_data_frozen(self):
        obj = SymplecticDissipationData(
            symplectic_residual=0.0,
            dissipation_rate=1.0,
            is_symplectically_invariant=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.dissipation_rate = 2.0

class TestPhase1_HermiticityAuditor:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.auditor = Phase1_HermiticityAuditor()

    def test_adaptive_tolerance(self):
        tol = self.auditor._adaptive_tolerance(1e-12, 1.0)
        assert tol >= 1e-12
        
    def test_coerce_finite_scalar_valid(self):
        val = self.auditor._coerce_finite_scalar("test", 1.5)
        assert abs(val - 1.5) < 1e-12

    def test_coerce_finite_scalar_invalid_bool(self):
        with pytest.raises(DomainIntegrityViolationError):
            self.auditor._coerce_finite_scalar("test", True)
            
    def test_coerce_finite_matrix_valid(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        res = self.auditor._coerce_finite_matrix("test", mat)
        np.testing.assert_allclose(res, mat)

    def test_coerce_finite_matrix_invalid_shape(self):
        with pytest.raises(DomainIntegrityViolationError):
            self.auditor._coerce_finite_matrix("test", np.array([1.0, 2.0]))

    def test_coerce_finite_vector_valid(self):
        vec = np.array([1.0, 2.0], dtype=np.float64)
        res = self.auditor._coerce_finite_vector("test", vec)
        np.testing.assert_allclose(res, vec)
        
    def test_spectral_decomposition_valid(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
        res = self.auditor._spectral_decomposition_and_validation(mat)
        assert res.is_spectrally_clean
        
    def test_spectral_decomposition_invalid(self):
        mat = np.array([[1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128)
        with pytest.raises(SpectralContaminationError):
            self.auditor._spectral_decomposition_and_validation(mat)

    def test_audit_operator_hermiticity_valid(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
        res = self.auditor._audit_operator_hermiticity(mat)
        assert res.is_hermitian
        
    def test_audit_operator_hermiticity_invalid(self):
        mat = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=np.complex128)
        with pytest.raises(NonHermitianOperatorError):
            self.auditor._audit_operator_hermiticity(mat)
            
    def test_phase1_audit_and_handoff(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
        res = self.auditor._phase1_audit_and_handoff_to_phase2(mat)
        assert isinstance(res, Phase1HermiticityHandoff)

class TestPhase2_BekensteinBoundEnforcer:
    """Granular tests for Phase 2 methods."""
    def setup_method(self):
        self.enforcer = Phase2_BekensteinBoundEnforcer()
        
    def test_certify_nonnegative_scalar_valid(self):
        val = self.enforcer._certify_nonnegative_scalar("test", 1.0)
        assert abs(val - 1.0) < 1e-12
        
    def test_certify_nonnegative_scalar_invalid(self):
        with pytest.raises(DomainIntegrityViolationError):
            self.enforcer._certify_nonnegative_scalar("test", -1.0, strict_positive=True)

class TestPhase3_SymplecticPortHamiltonianCertifier:
    """Granular tests for Phase 3 methods."""
    def setup_method(self):
        self.certifier = Phase3_SymplecticPortHamiltonianCertifier()

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def setup_method(self):
        with patch.object(AntimatterChokeCoilAgent, '__abstractmethods__', frozenset()):
            self.agent = AntimatterChokeCoilAgent()
