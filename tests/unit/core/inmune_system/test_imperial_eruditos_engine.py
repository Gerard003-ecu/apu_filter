# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Eruditos Engine (Floer y Cohomología de Čech)   ║
║ Ruta: tests/unit/core/inmune_system/test_imperial_eruditos_engine.py         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.imperial_eruditos_engine import (
    ImperialEruditosEngine,
    _FloerResult,
    _CechCohomologyResult,
)


class TestImperialEruditosEngine:
    """Evaluación del motor cohomológico de Floer y Čech."""

    def setup_method(self):
        self.engine = ImperialEruditosEngine(regularizer=1e-15)

    def test_verify_floer_homology_trajectory(self):
        """Verifica la regularidad del cilindro pseudo-holomorfo de Floer."""
        u_start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        u_end = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
        jacobian = np.eye(4, dtype=np.float64)

        res_residual, res_potential = self.engine.verify_floer_homology_trajectory(
            u_start, u_end, jacobian
        )
        assert res_residual >= 0.0
        assert res_potential >= 0.0

    def test_compute_attention_cech_cohomology(self):
        """Calcula la obstrucción de Čech sobre los pesos de atención."""
        att_weights = np.eye(4, dtype=np.float64)
        cech_obs, active_modes = self.engine.compute_attention_cech_cohomology(att_weights)

        assert cech_obs >= 0.0
        assert isinstance(active_modes, np.ndarray)

    def test_compute_symplectic_gradient(self):
        """Calcula el campo vectorial simpléctico por CSMD."""
        def H(x):
            return 0.5 * np.sum(x**2)

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        X_H = self.engine.compute_symplectic_gradient(H, x)

        assert X_H.shape == (4,)
        assert np.all(np.isfinite(X_H))
