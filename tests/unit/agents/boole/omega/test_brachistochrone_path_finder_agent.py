# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Brachistochrone Path Finder Agent (Estrato Ómega)        ║
║ Ruta: tests/unit/agents/boole/omega/test_brachistochrone_path_finder_agent.py║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.boole.omega.brachistochrone_path_finder_agent import (
    BrachistochronePathFinderAgent,
    Phase1_PotentialEnergyObserver,
    Phase2_ConformalMetricSuturator,
    Phase3_FermatBrachistochroneDecider,
    BrachistochroneHeytingVerdict,
    CrowbarBypassAction,
    StinespringPotentialDilation,
    ConformalGeometryBundle,
    BrachistochroneGovernanceState,
    EnergyBarrierViolationError,
)


class TestPhase1PotentialEnergyObserver:
    """Evaluación de la dilatación de Stinespring y barrera de energía de Maupertuis."""

    def setup_method(self):
        self.observer = Phase1_PotentialEnergyObserver()

    def test_observe_potential_well_success(self):
        """Verifica la dilatación del pozo de potencial cuando $H_0 > V(q)$."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([0.0, 0.0], dtype=np.float64)
        v_start = np.array([1.0, 0.0], dtype=np.float64)
        H0 = 10.0
        V_samples = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        dilation = self.observer.observe_potential_well(
            g_base=G,
            potential_v=V_samples,
            initial_h0=H0,
            q_start=q_start,
            v_start=v_start,
            potential_at_start=0.0,
        )
        assert isinstance(dilation, StinespringPotentialDilation)
        assert dilation.is_energy_well_safe is True
        assert dilation.phase1_verdict == BrachistochroneHeytingVerdict.COHERENT

    def test_energy_barrier_violation_raises(self):
        """Detona EnergyBarrierViolationError cuando V(q) >= H_0."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([0.0, 0.0], dtype=np.float64)
        v_start = np.array([0.0, 0.0], dtype=np.float64)
        H0 = 1.0
        V_samples = np.array([2.0], dtype=np.float64)

        with pytest.raises(EnergyBarrierViolationError):
            self.observer.observe_potential_well(
                g_base=G,
                potential_v=V_samples,
                initial_h0=H0,
                q_start=q_start,
                v_start=v_start,
                potential_at_start=2.0,
            )


class TestBrachistochronePathFinderAgentPipeline:
    """Integración completa de la braquistócrona de Fermat en el estrato Ómega."""

    def setup_method(self):
        self.agent = BrachistochronePathFinderAgent(raise_on_veto=False)

    def test_execute_brachistochrone_governance_success(self):
        """Ejecuta la gobernanza sobre un pozo de potencial armónico."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([1.0, 1.0], dtype=np.float64)
        v_start = np.array([0.1, 0.1], dtype=np.float64)
        V_fn = self.agent.harmonic_potential(omega=1.0)
        H0 = 10.0

        state = self.agent.execute_brachistochrone_governance(
            g_base=G,
            q_start=q_start,
            v_start=v_start,
            potential_v_fn=V_fn,
            initial_h0=H0,
            t_max=0.5,
            dt=0.01,
        )
        assert isinstance(state, BrachistochroneGovernanceState)
        assert state.is_epistemologically_valid is True
        assert state.verdict in (BrachistochroneHeytingVerdict.COHERENT, BrachistochroneHeytingVerdict.DEGRADED)
        assert state.transit_time > 0.0
