# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Antimatter Choke Coil Agent (Custodio del Vacío)         ║
║ Ruta: tests/unit/agents/physics/test_antimatter_choke_coil_agent.py          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np
import scipy.linalg as la

from app.agents.physics.antimatter_choke_coil_agent import (
    AntimatterChokeCoilAgent,
    Phase1_HermiticityAuditor,
    Phase2_BekensteinBoundEnforcer,
    Phase3_SymplecticPortHamiltonianCertifier,
    NonHermitianOperatorError,
    BekensteinLimitViolation,
    SymplecticCollapseError,
    DomainIntegrityViolationError,
    SpectralContaminationError,
    VacuumGovernanceState,
    HermiticityAuditData,
    BekensteinBoundData,
    SymplecticDissipationData,
    _HBAR_EFF,
    _C_EFF,
    _K_B,
)


class TestPhase1HermiticityAuditor:
    """Evaluación rigurosa de Fase 1: Auditoría de autoadjunción y espectro real."""

    def setup_method(self):
        self.auditor = Phase1_HermiticityAuditor()

    def test_audit_hermitian_operator_success(self):
        """Demuestra $\|A - A^\dagger\|_F \le \varepsilon_{herm}$ en operador autoadjunto."""
        A = np.array([[2.0 + 0.0j, 1.0 - 1.0j],
                      [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
        
        handoff = self.auditor._phase1_audit_and_handoff_to_phase2(A)
        assert handoff.hermiticity_audit.is_hermitian is True
        assert handoff.hermiticity_audit.residual_norm < 1e-12
        assert handoff.operator_dimension == 2
        assert handoff.spectral_certificate is not None
        assert handoff.spectral_certificate.is_spectrally_clean is True

    def test_non_hermitian_operator_raises_exception(self):
        """Detona NonHermitianOperatorError si la componente imaginaria viola la simetría CPT."""
        A = np.array([[2.0 + 0.0j, 1.0 + 5.0j],
                      [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
        
        with pytest.raises(NonHermitianOperatorError):
            self.auditor._phase1_audit_and_handoff_to_phase2(A)

    def test_boolean_scalar_rejection(self):
        """Verifica rechazo categórico de booleanos en el espacio R."""
        with pytest.raises(DomainIntegrityViolationError):
            self.auditor._coerce_finite_scalar("test_bool", True)


class TestPhase2BekensteinBoundEnforcer:
    """Evaluación rigurosa de Fase 2: Inecuación de cota holográfica de Bekenstein."""

    def setup_method(self):
        self.enforcer = Phase2_BekensteinBoundEnforcer()

    def test_bekenstein_bound_safe_entropy(self):
        """Verifica $S \le \frac{2\pi k_B E R}{\hbar c}$ en régimen sub-crítico."""
        A = np.eye(2, dtype=np.complex128)
        p1_handoff = self.enforcer._phase1_audit_and_handoff_to_phase2(A)
        
        E_gamma = 1.0e-10
        R_sys = 1.0
        bound = (2.0 * np.pi * _K_B * E_gamma * R_sys) / (_HBAR_EFF * _C_EFF)
        S_emitted = bound * 0.5
        
        p2_handoff = self.enforcer._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=p1_handoff,
            gamma_energy=E_gamma,
            system_radius_R=R_sys,
            emitted_entropy_S=S_emitted,
        )
        assert p2_handoff.bekenstein_audit.is_entropically_safe is True
        assert p2_handoff.bekenstein_audit.entropy_emitted == S_emitted

    def test_bekenstein_bound_violation_raises(self):
        """Detona BekensteinLimitViolation si la entropía excede el volumen holográfico."""
        A = np.eye(2, dtype=np.complex128)
        p1_handoff = self.enforcer._phase1_audit_and_handoff_to_phase2(A)
        
        E_gamma = 1.0e-10
        R_sys = 1.0
        bound = (2.0 * np.pi * _K_B * E_gamma * R_sys) / (_HBAR_EFF * _C_EFF)
        S_excessive = bound * 2.0
        
        with pytest.raises(BekensteinLimitViolation):
            self.enforcer._phase2_enforce_and_handoff_to_phase3(
                phase1_handoff=p1_handoff,
                gamma_energy=E_gamma,
                system_radius_R=R_sys,
                emitted_entropy_S=S_excessive,
            )


class TestPhase3SymplecticPortHamiltonianCertifier:
    """Evaluación rigurosa de Fase 3: Invarianza simpléctica y disipación Port-Hamiltoniana."""

    def setup_method(self):
        self.certifier = Phase3_SymplecticPortHamiltonianCertifier()

    def test_symplectic_preservation_and_dissipation(self):
        """Certifica $M^\top \Omega M = \Omega$ y disipación Rayleigh $\dot{H} \le 0$."""
        M = np.eye(2, dtype=np.float64)
        grad_H = np.array([1.0, 1.0], dtype=np.float64)
        J = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        R = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
        
        dissipation = self.certifier._certify_symplectic_port_hamiltonian(
            jacobian_M=M,
            grad_H=grad_H,
            J_matrix=J,
            R_matrix=R,
        )
        assert dissipation.is_symplectically_invariant is True
        assert dissipation.dissipation_rate <= 0.0

    def test_non_symplecticity_raises_collapse(self):
        """Detona SymplecticCollapseError si det(M) != 1."""
        M = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        grad_H = np.array([1.0, 1.0], dtype=np.float64)
        J = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64)
        
        with pytest.raises(SymplecticCollapseError):
            self.certifier._certify_symplectic_port_hamiltonian(
                jacobian_M=M,
                grad_H=grad_H,
                J_matrix=J,
                R_matrix=R,
            )


class TestAntimatterChokeCoilAgentFullPipeline:
    """Integración completa del endofuntor de gobierno de vacío."""

    def setup_method(self):
        self.agent = AntimatterChokeCoilAgent()

    def test_execute_vacuum_governance_full_success(self):
        """Ejecuta la composición funtorial completa $\Phi_3 \circ \Phi_2 \circ \Phi_1$."""
        operator_A = np.eye(2, dtype=np.complex128)
        E_gamma = 1.0e-10
        R_sys = 1.0
        bound = (2.0 * np.pi * _K_B * E_gamma * R_sys) / (_HBAR_EFF * _C_EFF)
        S_emitted = bound * 0.1
        
        M = np.eye(2, dtype=np.float64)
        grad_H = np.array([0.5, -0.5], dtype=np.float64)
        J = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * 0.1
        
        state = self.agent.execute_vacuum_governance(
            operator_A=operator_A,
            gamma_energy=E_gamma,
            system_radius_R=R_sys,
            emitted_entropy_S=S_emitted,
            jacobian_M=M,
            grad_H=grad_H,
            J_matrix=J,
            R_matrix=R,
        )
        assert isinstance(state, VacuumGovernanceState)
        assert state.is_epistemologically_valid is True
        assert state.hermiticity_audit.is_hermitian is True
        assert state.bekenstein_audit.is_entropically_safe is True
        assert state.symplectic_audit.is_symplectically_invariant is True
