# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Synaptic Fock Space Registry Agent (Topos Memoria Cuántica)║
║ Ruta: tests/unit/agents/core/inmune_system/test_synaptic_fock_space_registry_agent.py ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.synaptic_fock_space_registry_agent import (
    SynapticFockSpaceRegistryAgent,
    Phase1_FockIsometryAuditor,
    Phase2_SymplecticBogoliubovValidator,
    Phase3_LindbladPassivityValidator,
    FockSovereignVerdict,
    CrowbarAction,
    Phase1FockIsometryCertificate,
    Phase2SymplecticBogoliubovCertificate,
    Phase3LindbladPassivityCertificate,
    FockRegistrySovereignState,
    stable_density_matrix_higham,
)
from app.core.inmune_system.synaptic_fock_space_registry import SynapticFockSpaceRegistry


class TestHighamDensityMatrixProjection:
    """Evaluación de regularización espectral de Higham para operadores densidad."""

    def test_stable_density_matrix_higham_non_psd(self):
        """Proyecta una matriz con autovalor negativo hacia el cono PSD con traza 1."""
        rho_bad = np.array([[1.2 + 0.0j, 0.5 + 0.0j],
                            [0.5 + 0.0j, -0.2 + 0.0j]], dtype=np.complex128)

        rho_proj, shift, applied = stable_density_matrix_higham(rho_bad)
        assert applied is True
        assert np.isclose(np.trace(rho_proj).real, 1.0)
        eigenvals = np.linalg.eigvalsh(rho_proj)
        assert np.all(eigenvals >= -1e-12)


class TestPhase2SymplecticBogoliubovValidator:
    """Evaluación de invarianza simpléctica |u_k|^2 - |v_k|^2 = 1."""

    def setup_method(self):
        self.validator = Phase2_SymplecticBogoliubovValidator()

    def test_symplectic_invariance_success(self):
        """Verifica coeficientes de Bogoliubov válidos."""
        u_k = 1.25 + 0.0j
        v_k = 0.75 + 0.0j

        cert1 = Phase1FockIsometryCertificate(
            is_pauli_respected=True,
            is_fock_bounded=True,
            entropy_eviction_ratio=0.1,
            occupancy=10,
            max_capacity=100,
            density_matrix_trace=1.0,
            density_matrix_purity=0.9,
            higham_regularization_applied=False,
            sparse_computation=False,
            verdict=FockSovereignVerdict.COHERENT,
        )

        cert2 = self.validator.audit_symplectic_bogoliubov(
            u_k=u_k,
            v_k=v_k,
            phase1_cert=cert1,
        )
        assert cert2.is_symplectic_invariant is True
        assert cert2.verdict == FockSovereignVerdict.COHERENT

    def test_symplectic_invariance_violation_vetoed(self):
        """Detona veredicto VETOED ante desintegración de la norma simpléctica."""
        u_k = 1.0 + 0.0j
        v_k = 1.0 + 0.0j

        cert1 = Phase1FockIsometryCertificate(
            is_pauli_respected=True,
            is_fock_bounded=True,
            entropy_eviction_ratio=0.1,
            occupancy=10,
            max_capacity=100,
            density_matrix_trace=1.0,
            density_matrix_purity=0.9,
            higham_regularization_applied=False,
            sparse_computation=False,
            verdict=FockSovereignVerdict.COHERENT,
        )

        cert2 = self.validator.audit_symplectic_bogoliubov(
            u_k=u_k,
            v_k=v_k,
            phase1_cert=cert1,
        )
        assert cert2.is_symplectic_invariant is False
        assert cert2.verdict == FockSovereignVerdict.VETOED


class TestSynapticFockSpaceRegistryAgentPipeline:
    """Integración completa del soberano de la memoria cuántica."""

    def setup_method(self):
        self.agent = SynapticFockSpaceRegistryAgent(raise_on_veto=False)
        self.registry = SynapticFockSpaceRegistry(max_cartridges=100)

    def test_execute_sovereign_governance_coherent(self):
        """Ejecuta el ciclo OODA completo con veredicto final COHERENT."""
        u_vec = np.array([1.0, 0.0], dtype=np.float64)
        u_k = 1.25 + 0.0j
        v_k = 0.75 + 0.0j
        rho_pre = np.eye(2, dtype=np.complex128) / 2.0
        rho_post = np.eye(2, dtype=np.complex128) / 2.0
        H_eff = np.eye(2, dtype=np.complex128)

        state = self.agent.execute_sovereign_governance(
            registry=self.registry,
            current_decision_vector=u_vec,
            u_k=u_k,
            v_k=v_k,
            rho_pre=rho_pre,
            rho_post=rho_post,
            hamiltonian_eff=H_eff,
        )
        assert isinstance(state, FockRegistrySovereignState)
        assert state.final_verdict == FockSovereignVerdict.COHERENT
        assert state.crowbar_triggered is False
