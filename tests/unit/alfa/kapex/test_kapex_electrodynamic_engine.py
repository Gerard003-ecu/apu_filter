# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: KAPEX Electrodynamic Engine (Eikonal, Yang-Mills)        ║
║ Ruta: tests/unit/alfa/kapex/test_kapex_electrodynamic_engine.py              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.alfa.kapex.kapex_electrodynamic_engine import (
    KapexElectrodynamicEngine,
    HeytingVerdict,
)


class TestKapexElectrodynamicEngine:
    """Evaluación del motor electrodinámico de ápice (Eikonal, Yang-Mills y Poynting)."""

    def setup_method(self):
        self.n = 3
        self.engine = KapexElectrodynamicEngine(dimension_n=self.n, reg_param=1e-15)

    def test_solve_eikonal_residual(self):
        """Verifica el residuo de la EDP eikonal $g^{\\mu\\nu} \\partial_\\mu S \\partial_\\nu S - n(q)^2 = 0$."""
        grad_S = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        g = np.eye(3, dtype=np.float64)
        n_refraction = 1.0

        residual, p_up = self.engine.solve_eikonal_residual(grad_S, g, n_refraction)
        assert np.isclose(residual, 0.0)
        assert np.allclose(p_up, grad_S)

    def test_compute_yang_mills_action(self):
        """Verifica la acción de Yang-Mills sobre $\\mathfrak{so}(n)$."""
        A = [np.zeros((3, 3), dtype=np.float64) for _ in range(3)]
        g = np.eye(3, dtype=np.float64)

        F, action = self.engine.compute_yang_mills_action(A, g)
        assert F.shape == (3, 3, 3, 3)
        assert np.isclose(action, 0.0)

    def test_compute_poynting_strategic_flux(self):
        """Verifica el vector de flujo de Poynting $S^\\mu$ en $n=3$."""
        E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        B = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        g = np.eye(3, dtype=np.float64)

        S_up, energy = self.engine.compute_poynting_strategic_flux(E, B, g)
        assert np.allclose(S_up, [0.0, 0.0, 1.0])
        assert np.isclose(energy, 1.0)

    def test_execute_electrodynamic_cycle(self):
        """Ejecuta el ciclo OODA electrodinámico completo."""
        grad_S = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        A = [np.zeros((3, 3), dtype=np.float64) for _ in range(3)]
        g = np.eye(3, dtype=np.float64)
        E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        B = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        telemetry = self.engine.execute_electrodynamic_cycle(
            grad_S, A, g, E, B, refraction_index=1.0
        )
        assert isinstance(telemetry, dict)
        assert "heyting_verdict" in telemetry
        assert telemetry["heyting_verdict"] in ("CERTIFIED", "COHERENT", "DEGRADED", "VETOED")
