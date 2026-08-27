# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Guards Agent (Censura de Calibre de de Rham)    ║
║ Ruta: tests/unit/agents/core/inmune_system/test_imperial_guards_agent.py   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_agent import (
    ImperialGuardsAgent,
    Phase1SpectralObservation,
    Phase2LogisticObservation,
    Phase3TribunalDecision,
    ImperialGuardsCertificate,
)


class TestImperialGuardsAgent:
    """Evaluación del tribunal de la Guardia Imperial y cotas de Connes / Cheeger."""

    def setup_method(self):
        self.agent = ImperialGuardsAgent(config_dim_n=4, cheeger_threshold=0.15)

    def test_kahan_summation_accuracy(self):
        """Demuestra la sumación compensada de Kahan-Neumaier contra deriva flotante."""
        terms = np.array([1e16, 1.0, -1e16], dtype=np.float64)
        res = self.agent.kahan_compensated_sum(terms)
        assert np.isclose(res, 1.0)

    def test_phase1_spectral_audit_coherent(self):
        """Audita la cota de Connes con autovalor mínimo suficiente."""
        eigs_dirac = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        obs1 = self.agent.phase1_audit_spectral_heterogeomorphic_curve(eigs_dirac)

        assert isinstance(obs1, Phase1SpectralObservation)
        assert obs1.lambda_min_dirac == 0.5
        assert obs1.partial_verdict == "COHERENT"

    def test_phase2_logistic_audit_coherent(self):
        """Audita la conectividad de Fiedler y retornos de Betti en un complejo simplicial conexo."""
        eigs_L = np.array([0.0, 0.9, 1.2, 2.0], dtype=np.float64)
        obs2 = self.agent.phase2_audit_logistic_from_phase1(
            phase1_observation=None,
            eigenvalues_L=eigs_L,
            betti_0=1,
            betti_1=0,
        )

        assert isinstance(obs2, Phase2LogisticObservation)
        assert obs2.fiedler_connectivity == 0.9
        assert obs2.cohomological_residual == 0.0
        assert obs2.partial_verdict == "COHERENT"

    def test_execute_guardians_cycle_coherent(self):
        """Ejecuta el ciclo OODA completo resultando en veredicto COHERENT."""
        eigs_dirac = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        eigs_L = np.array([0.0, 0.9, 1.2, 2.0], dtype=np.float64)

        cert = self.agent.execute_guardians_cycle(
            eigenvalues_dirac=eigs_dirac,
            eigenvalues_L=eigs_L,
            betti_0=1,
            betti_1=0,
        )
        assert isinstance(cert, ImperialGuardsCertificate)
        assert cert.heyting_verdict == "COHERENT"
        assert cert.hardware_interlock_fired is False

    def test_execute_guardians_cycle_vetoed_by_loops(self):
        """Detona VETOED si existen ciclos lógicos parásitos (beta_1 > 0)."""
        eigs_dirac = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        eigs_L = np.array([0.0, 0.9, 1.2, 2.0], dtype=np.float64)

        cert = self.agent.execute_guardians_cycle(
            eigenvalues_dirac=eigs_dirac,
            eigenvalues_L=eigs_L,
            betti_0=1,
            betti_1=2,
        )
        assert cert.heyting_verdict == "VETOED"
        assert cert.hardware_interlock_fired is True
        assert "logical_loops_detected" in cert.veto_reasons
