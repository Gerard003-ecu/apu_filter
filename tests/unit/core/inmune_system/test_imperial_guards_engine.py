# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Guards Engine (Cálculo Espectral y Topológico)  ║
║ Ruta: tests/unit/core/inmune_system/test_imperial_guards_engine.py          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.imperial_guards_engine import ImperialGuardsEngine


class TestImperialGuardsEngine:
    """Evaluación del motor espectral y topológico de las guardias imperiales."""

    def setup_method(self):
        self.engine = ImperialGuardsEngine(regularizer=1e-15)

    def test_kahan_sum(self):
        """Verifica la sumación compensada KBN en coma flotante."""
        arr = np.array([1e-16, 1.0, -1.0, 2.0, 1e-16], dtype=np.float64)
        res = self.engine.kahan_sum(arr)
        assert np.isclose(res, 2.0 + 2e-16)

    def test_compute_dirac_operator_spectrum(self):
        """Verifica el espectro del operador de Dirac de Connes $\\not\\!D = \\rho^{-1/2}$."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        dirac_eigs, evals = self.engine.compute_dirac_operator_spectrum(rho)

        assert len(dirac_eigs) == 2
        assert len(evals) == 2
        assert np.allclose(dirac_eigs, np.sqrt(2.0))

    def test_compute_petz_fisher_rao_metric(self):
        """Verifica la métrica de Petz-Fisher-Rao cuántica $g_\\rho(A, B)$."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        A = np.eye(2, dtype=np.complex128)
        B = np.eye(2, dtype=np.complex128)

        metric = self.engine.compute_petz_fisher_rao_metric(rho, A, B)
        assert metric > 0.0

    def test_compute_simplicial_normalized_laplacian(self):
        """Verifica la construcción del Laplaciano simplicial normalizado."""
        # Matriz de incidencia |E|x|V| = 2x3
        b0 = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)
        L_norm = self.engine.compute_simplicial_normalized_laplacian(b0)

        assert L_norm.shape == (3, 3)
        assert np.allclose(L_norm, L_norm.T)

    def test_estimate_cheeger_constant_bounds(self):
        """Estima las cotas de Cheeger $h(G)$ a partir del valor de Fiedler."""
        eigs_L = np.array([0.0, 0.5, 1.2], dtype=np.float64)
        h_lower, h_upper = self.engine.estimate_cheeger_constant_bounds(eigs_L)

        assert h_lower == 0.25
        assert np.isclose(h_upper, 1.0)

    def test_compute_euler_poincare_characteristic(self):
        """Calcula la característica de Euler-Poincaré $\\chi(K) = |V| - |E| + |F|$."""
        b0 = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)  # 3V, 2E
        chi = self.engine.compute_euler_poincare_characteristic(b0, None)

        assert chi == 3 - 2 + 0  # 1
