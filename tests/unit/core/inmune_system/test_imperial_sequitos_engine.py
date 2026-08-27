# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Sequitos Engine (Consenso y Bell-CHSH)          ║
║ Ruta: tests/unit/core/inmune_system/test_imperial_sequitos_engine.py         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.imperial_sequitos_engine import ImperialSequitosEngine


class TestImperialSequitosEngine:
    """Evaluación del motor de los séquitos imperiales."""

    def setup_method(self):
        self.engine = ImperialSequitosEngine(regularizer=1e-15)

    def test_kleisli_compose(self):
        """Verifica la composición monádica en la categoría de Kleisli."""
        def f(x):
            return x + 1, 0.9

        def g(y):
            return y * 2, 0.8

        h = self.engine.kleisli_compose(f, g)
        val, prob = h(3)
        assert val == 8
        assert np.isclose(prob, 0.72)

    def test_compute_degroot_spectral_consensus(self):
        """Verifica la convergencia espectral de DeGroot."""
        opinions = np.array([1.0, 0.0], dtype=np.float64)
        affinity = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)

        final_ops, fiedler, verdict = self.engine.compute_degroot_spectral_consensus(
            opinions, affinity, steps=50
        )
        assert len(final_ops) == 2
        assert fiedler >= 0.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_compute_uhlmann_fidelity(self):
        """Verifica la fidelidad de Uhlmann $F(\\rho, \\sigma) \\in [0, 1]$."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        sigma = np.eye(2, dtype=np.complex128) / 2.0

        fid = self.engine.compute_uhlmann_fidelity(rho, sigma)
        assert np.isclose(fid, 1.0)

    def test_verify_chsh_violation(self):
        """Verifica la violación cuántica multipartita de Bell-CHSH."""
        E = np.array([[0.6, -0.6], [0.6, 0.6]], dtype=np.float64)
        s_val, verdict = self.engine.verify_chsh_violation(E)

        assert np.isclose(s_val, 2.4)
        assert verdict == "COHERENT"
