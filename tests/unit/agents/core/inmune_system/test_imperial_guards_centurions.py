# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Guards Centurions (Control Port-Hamiltoniano)  ║
║ Ruta: tests/unit/agents/core/inmune_system/test_imperial_guards_centurions.py║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_centurions import (
    HeytingVerdict,
    PortHamiltonianCenturion,
    ThermodynamicCenturion,
    CenturionsCoherenceChamber,
)


class TestPortHamiltonianCenturion:
    """Evaluación de la dinámica Port-Hamiltoniana e identidades disipativas IDA-PBC."""

    def setup_method(self):
        self.n = 4
        self.M_d = np.eye(self.n, dtype=np.float64)
        self.R_d = np.eye(self.n, dtype=np.float64) * 0.5
        self.x_star = np.zeros(self.n, dtype=np.float64)
        self.centurion = PortHamiltonianCenturion(
            dimension_n=self.n,
            inertia_matrix=self.M_d,
            damping_matrix_rd=self.R_d,
            target_state=self.x_star,
        )

    def test_hamiltonian_and_gradient(self):
        """Verifica $H_d(x) = \frac{1}{2} e^\top M_d^{-1} e$ y $\\nabla H_d(x)$."""
        x = np.array([1.0, 0.0, 2.0, 0.0], dtype=np.float64)
        H = self.centurion.compute_hamiltonian(x)
        assert np.isclose(H, 2.5)  # 0.5 * (1^2 + 2^2) = 2.5

        grad = self.centurion.compute_gradient(x)
        assert np.allclose(grad, x)

    def test_evaluate_power_curtain_coherent(self):
        """Verifica la potencia disipada $\\dot{H}_d \le 0$ en régimen pasivo."""
        x = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        u = np.zeros(self.n, dtype=np.float64)

        audit = self.centurion.evaluate_power_curtain(x, u)
        assert audit.dissipation_power > 0.0
        assert audit.verdict == "COHERENT"


class TestThermodynamicCenturion:
    """Evaluación de entropía de von Neumann y condición KMS."""

    def setup_method(self):
        self.dim = 2
        self.centurion = ThermodynamicCenturion(dimension_h=self.dim, basal_temperature=1.0)

    def test_von_neumann_entropy_pure_state(self):
        """Verifica $S(\\rho) = 0$ para un estado puro."""
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        S = self.centurion.compute_von_neumann_entropy(rho_pure)
        assert np.isclose(S, 0.0, atol=1e-7)

    def test_kms_condition_verification(self):
        """Verifica el residuo de la condición KMS a temperatura $\\beta=1$."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        A = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        B = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        kms_res, verdict = self.centurion.verify_kms_condition(rho, A, B, beta=1.0)
        assert kms_res >= 0.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")


class TestCenturionsCoherenceChamberPipeline:
    """Integración de la cámara de coherencia de los centuriones."""

    def setup_method(self):
        n = 4
        M_d = np.eye(n, dtype=np.float64)
        R_d = np.eye(n, dtype=np.float64) * 0.5
        x_star = np.zeros(n, dtype=np.float64)

        self.chamber = CenturionsCoherenceChamber.assemble_from_spectral_seed(
            dimension_n=n,
            inertia_matrix=M_d,
            damping_matrix_rd=R_d,
            target_state=x_star,
            dimension_h=2,
            basal_temperature=1.0,
        )

    def test_process_coherence_cycle(self):
        """Ejecuta el ciclo de coherencia unificado de Capa 2."""
        state_x = np.array([0.2, 0.2, 0.0, 0.0], dtype=np.float64)
        ext_u = np.zeros(4, dtype=np.float64)
        rho = np.eye(2, dtype=np.complex128) / 2.0
        A = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        B = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        result = self.chamber.process_coherence_cycle(
            state_x=state_x,
            external_u=ext_u,
            density_rho=rho,
            obs_A=A,
            obs_B=B,
            beta_kms=1.0,
        )
        assert "heyting_verdict" in result
        assert result["heyting_verdict"] in ("COHERENT", "DEGRADED", "VETOED")
