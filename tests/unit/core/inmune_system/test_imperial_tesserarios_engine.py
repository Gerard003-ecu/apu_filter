# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Tesserarios Engine (Homotopía y Gerbes de Čech) ║
║ Ruta: tests/unit/core/inmune_system/test_imperial_tesserarios_engine.py     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.imperial_tesserarios_engine import ImperialTesserariosEngine


class TestImperialTesserariosEngine:
    """Evaluación del motor homotópico de los tesserarios imperiales."""

    def setup_method(self):
        self.engine = ImperialTesserariosEngine(regularizer=1e-15)

    def test_compute_quillen_factorization(self):
        """Verifica la factorización de Quillen $M = P \\cdot I$ en la categoría modelo."""
        M = np.eye(4, dtype=np.float64)
        P, I, residual = self.engine.compute_quillen_factorization(M)

        assert P.shape == (4, 4)
        assert I.shape == (4, 4)
        assert residual >= 0.0

    def test_project_to_symplectic_group(self):
        """Verifica la proyección ortogonal simpléctica sobre $Sp(2n, \\mathbb{R})$."""
        M = np.eye(4, dtype=np.float64)
        M_sym, residual = self.engine.project_to_symplectic_group(M)

        assert M_sym.shape == (4, 4)
        assert np.isclose(residual, 0.0)

    def test_compute_stasheff_m3_associator(self):
        """Calcula el tensor de homotopía $m_3$ en el asociaedro de Stasheff."""
        m2 = np.zeros((2, 2, 2), dtype=np.float64)
        m2[0, 0, 0] = 1.0  # álgebra asociativa trivial

        m3 = self.engine.compute_stasheff_m3_associator(m2)
        assert m3.shape == (2, 2, 2, 2)
        assert np.allclose(m3, 0.0)

    def test_compute_cech_hypercohomology_gerbe(self):
        """Evalúa la clase de obstrucción de Čech para Gerbes no abelianos."""
        cech_matrix = np.eye(4, dtype=np.float64)
        obs_val, s_vals = self.engine.compute_cech_hypercohomology_gerbe(cech_matrix)

        assert obs_val >= 0.0
        assert len(s_vals) == 4
