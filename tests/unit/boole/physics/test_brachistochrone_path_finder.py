# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Brachistochrone Path Finder Motor (Física Boole)        ║
║ Ruta: tests/unit/boole/physics/test_brachistochrone_path_finder.py          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.boole.physics.brachistochrone_path_finder import (
    BrachistochronePathFinder,
    Phase1_PotentialWellInquirer,
    Phase2_ConformalKoszulSuturator,
    Phase3_FermatGeodesicSolver,
    ConformalPotentialDilation,
    ConformalManifoldBundle,
    BrachistochronePhysicalState,
    EnergyWellColapsoError,
)


class TestPhase1PotentialWellInquirer:
    """Evaluación de la dilatación conforme de potencial en el motor físico."""

    def setup_method(self):
        self.inquirer = Phase1_PotentialWellInquirer()

    def test_evaluate_potential_barrier_success(self):
        """Sanea la barrera de potencial y calcula el factor de Cholesky."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([0.0, 0.0], dtype=np.float64)
        v_start = np.array([1.0, 0.0], dtype=np.float64)
        H0 = 10.0
        V_samples = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        dilation = self.inquirer.evaluate_potential_barrier(
            g_base=G,
            potential_v=V_samples,
            initial_h0=H0,
            q_start=q_start,
            v_start=v_start,
            potential_at_start=0.0,
        )
        assert isinstance(dilation, ConformalPotentialDilation)
        assert dilation.is_safe is True
        assert dilation.energy_gap_min > 0.0

    def test_energy_well_collapse_raises(self):
        """Detona EnergyWellColapsoError ante colapso de la barrera V >= H_0."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([0.0, 0.0], dtype=np.float64)
        v_start = np.array([0.0, 0.0], dtype=np.float64)
        H0 = 1.0
        V_samples = np.array([2.0], dtype=np.float64)

        with pytest.raises(EnergyWellColapsoError):
            self.inquirer.evaluate_potential_barrier(
                g_base=G,
                potential_v=V_samples,
                initial_h0=H0,
                q_start=q_start,
                v_start=v_start,
                potential_at_start=2.0,
            )


class TestBrachistochronePathFinderEnginePipeline:
    """Integración del motor físico puro de la braquistócrona de Fermat-Jacobi."""

    def setup_method(self):
        self.engine = BrachistochronePathFinder()

    def test_compute_brachistochrone_path_success(self):
        """Calcula la geodésica conforme sobre un pozo de gravedad lineal."""
        G = np.eye(2, dtype=np.float64)
        q_start = np.array([0.0, 0.0], dtype=np.float64)
        v_start = np.array([0.1, 0.1], dtype=np.float64)
        V_fn = self.engine.linear_gravity_potential(g_acc=1.0, axis=1)
        H0 = 5.0

        state = self.engine.compute_brachistochrone_path(
            g_base=G,
            q_start=q_start,
            v_start=v_start,
            potential_v_fn=V_fn,
            initial_h0=H0,
            t_max=0.5,
            dt=0.01,
        )
        assert isinstance(state, BrachistochronePhysicalState)
        assert state.is_globally_stable is True
        assert state.transit_time_t > 0.0
        assert state.trajectory.shape[1] == 2
