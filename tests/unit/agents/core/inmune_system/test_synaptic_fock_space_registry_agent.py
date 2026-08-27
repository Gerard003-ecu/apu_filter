# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para synaptic_fock_space_registry_agent."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.core.inmune_system.synaptic_fock_space_registry_agent import (
    SynapticFockSpaceRegistryAgent,
    Phase1_FockIsometryAuditor,
    Phase2_SymplecticBogoliubovValidator,
    Phase3_LindbladPassivityValidator,
    stable_density_matrix_higham,
    Phase1FockIsometryCertificate,
    Phase2SymplecticBogoliubovCertificate,
    Phase3LindbladPassivityCertificate,
    FockRegistrySovereignState,
    FockSovereignVerdict,
    CrowbarAction,
    SynapticFockSpaceRegistryAgentError,
    DensityMatrixAnomalyError,
    _DEFAULT_TOL,
    _HIGHAM_REGULARIZATION_FLOOR
)

# Dummy classes for missing registry imports if they are needed
class DummyElectronCartridge:
    homological_charge = -1
    @property
    def quantum_state_hash(self): return "hash_e"

class DummyPositronCartridge:
    homological_charge = 1
    @property
    def quantum_state_hash(self): return "hash_p"

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_positive(self):
        assert _DEFAULT_TOL > 0.0
        assert _HIGHAM_REGULARIZATION_FLOOR > 0.0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_density_matrix_anomaly_error_inherits(self):
        assert issubclass(DensityMatrixAnomalyError, SynapticFockSpaceRegistryAgentError)

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_phase1_certificate_frozen(self):
        cert = Phase1FockIsometryCertificate(
            is_pauli_respected=True,
            is_fock_bounded=True,
            entropy_eviction_ratio=0.5,
            occupancy=10,
            max_capacity=100,
            density_matrix_trace=1.0,
            density_matrix_purity=1.0,
            higham_regularization_applied=False,
            sparse_computation=False,
            verdict=FockSovereignVerdict.COHERENT
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.occupancy = 11

    def test_fock_registry_sovereign_state_frozen(self):
        dummy1 = Phase1FockIsometryCertificate(True, True, 0.0, 0, 0, 1.0, 1.0, False, False, FockSovereignVerdict.COHERENT)
        dummy2 = Phase2SymplecticBogoliubovCertificate(True, 0.0, True, None, True, 0, 0, FockSovereignVerdict.COHERENT)
        dummy3 = Phase3LindbladPassivityCertificate(True, 0.0, True, True, 0.0, 0.0, True, True, 0.0, True, 10, False, FockSovereignVerdict.COHERENT)
        state = FockRegistrySovereignState(dummy1, dummy2, dummy3, FockSovereignVerdict.COHERENT, False, CrowbarAction.NONE, "timestamp", "hash")
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            state.diagnostic_note = "new note"

class DummyAuditor(Phase1_FockIsometryAuditor):
    pass

class DummyValidator(Phase2_SymplecticBogoliubovValidator):
    pass

class DummyLindblad(Phase3_LindbladPassivityValidator):
    pass

class TestStableDensityMatrixHigham:
    """Tests for stable_density_matrix_higham function."""
    def test_stable_density_matrix_psd(self):
        rho = np.array([[1.0, 0.0], [0.0, -0.1]], dtype=np.complex128)
        rho_proj, shift, applied = stable_density_matrix_higham(rho)
        assert applied is True
        vals = np.linalg.eigvalsh(rho_proj)
        assert all(v >= -1e-10 for v in vals)
        assert abs(np.sum(vals) - 1.0) < _DEFAULT_TOL

    def test_stable_density_matrix_valid(self):
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.complex128)
        rho_proj, shift, applied = stable_density_matrix_higham(rho)
        assert applied is False
        assert abs(shift) < 1e-15

class TestPhase1_FockIsometryAuditor:
    """Granular tests for Phase 1 methods."""
    def test_compute_entropy_ratio(self):
        auditor = DummyAuditor()
        registry = MagicMock()
        registry.size = 1
        registry._embedding_cache = {'a': np.array([1.0, 0.0])}
        vec = np.array([1.0, 0.0])
        ratio, sparse = auditor._compute_entropy_ratio(registry, vec)
        assert abs(ratio - 1.0) < _DEFAULT_TOL

    def test_audit_fock_isometry_empty(self):
        auditor = DummyAuditor()
        registry = MagicMock()
        registry.size = 0
        registry.max_capacity = 100
        registry._registry = {}
        registry._embedding_cache = {}
        vec = np.array([])
        cert = auditor.audit_fock_isometry(registry, vec)
        assert cert.is_fock_bounded is True
        assert cert.occupancy == 0

class TestPhase2_SymplecticBogoliubovValidator:
    """Granular tests for Phase 2 methods."""
    def test_audit_symplectic_bogoliubov_valid(self):
        validator = DummyValidator()
        dummy1 = Phase1FockIsometryCertificate(True, True, 0.0, 10, 100, 1.0, 1.0, False, False, FockSovereignVerdict.COHERENT)
        u_k = complex(math.cosh(1.0), 0.0)
        v_k = complex(math.sinh(1.0), 0.0)
        cert = validator.audit_symplectic_bogoliubov(u_k, v_k, dummy1)
        assert cert.is_symplectic_invariant is True
        assert cert.verdict == FockSovereignVerdict.COHERENT

    def test_audit_symplectic_bogoliubov_invalid(self):
        validator = DummyValidator()
        dummy1 = Phase1FockIsometryCertificate(True, True, 0.0, 10, 100, 1.0, 1.0, False, False, FockSovereignVerdict.COHERENT)
        cert = validator.audit_symplectic_bogoliubov(1.0, 1.0, dummy1)
        assert cert.is_symplectic_invariant is False
        assert cert.verdict == FockSovereignVerdict.VETOED

class TestPhase3_LindbladPassivityValidator:
    """Granular tests for Phase 3 methods."""
    def test_audit_lindblad_evolution_valid(self):
        validator = DummyLindblad()
        rho_pre = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        rho_post = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        H_eff = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        dummy2 = Phase2SymplecticBogoliubovCertificate(True, 0.0, True, None, True, 1, 0, FockSovereignVerdict.COHERENT)
        cert = validator.audit_lindblad_evolution(rho_pre, rho_post, H_eff, dummy2)
        assert cert.is_trace_preserved is True
        assert cert.is_density_matrix_psd is True
        assert cert.is_lyapunov_passive is True
        assert cert.verdict == FockSovereignVerdict.COHERENT

class TestSynapticFockSpaceRegistryAgent:
    """Integration tests for the main class."""
    def test_execute_sovereign_governance_coherent(self):
        with patch.object(SynapticFockSpaceRegistryAgent, '__abstractmethods__', frozenset()):
            agent = SynapticFockSpaceRegistryAgent()
            
            registry = MagicMock()
            registry.size = 1
            registry.max_capacity = 100
            registry._registry = {}
            registry._embedding_cache = {'a': np.array([1.0, 0.0])}
            vec = np.array([1.0, 0.0])
            u_k = complex(1.0, 0.0)
            v_k = complex(0.0, 0.0)
            rho_pre = np.array([[1.0]], dtype=np.complex128)
            rho_post = np.array([[1.0]], dtype=np.complex128)
            H_eff = np.array([[1.0]], dtype=np.complex128)
            
            state = agent.execute_sovereign_governance(
                registry, vec, u_k, v_k, rho_pre, rho_post, H_eff
            )
            assert state.final_verdict == FockSovereignVerdict.COHERENT
            assert state.crowbar_triggered is False

    def test_execute_sovereign_governance_vetoed(self):
        with patch.object(SynapticFockSpaceRegistryAgent, '__abstractmethods__', frozenset()):
            agent = SynapticFockSpaceRegistryAgent(raise_on_veto=False)
            
            registry = MagicMock()
            registry.size = 101 # exceeds capacity
            registry.max_capacity = 100
            registry._registry = {}
            registry._embedding_cache = {}
            
            vec = np.array([1.0, 0.0])
            u_k = complex(1.0, 0.0)
            v_k = complex(0.0, 0.0)
            rho_pre = np.array([[1.0]], dtype=np.complex128)
            rho_post = np.array([[1.0]], dtype=np.complex128)
            H_eff = np.array([[1.0]], dtype=np.complex128)
            
            state = agent.execute_sovereign_governance(
                registry, vec, u_k, v_k, rho_pre, rho_post, H_eff
            )
            assert state.final_verdict == FockSovereignVerdict.VETOED
            assert state.crowbar_triggered is True
