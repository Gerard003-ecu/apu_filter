# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Centurions Engine (Topología y Termodinámica)   ║
║ Ruta: tests/unit/core/inmune_system/test_imperial_centurions_engine.py      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.imperial_centurions_engine import ImperialCenturionsEngine


class TestImperialCenturionsEngine:
    """Evaluación del motor de los centuriones: conservación simpléctica y flujo de Tomita-Takesaki."""

    def setup_method(self):
        self.n = 2  # dimension_n = 2 (T*Q de dim 4)
        self.engine = ImperialCenturionsEngine(dimension_n=self.n)

    def test_verify_symplectic_preservation(self):
        """Verifies $M^\top \Omega M - \Omega \equiv 0$ para matriz de transición simpléctica."""
        # 2n = 4
        M_sym = np.eye(4, dtype=np.float64)
        residual_norm, is_viable = self.engine.verify_symplectic_preservation(M_sym)
        assert np.isclose(residual_norm, 0.0)
        assert is_viable is True

    def test_purify_density_operator(self):
        """Verifica la purificación espectral de la matriz de densidad $\\operatorname{Tr}(\\rho) \equiv 1.0$."""
        rho_mixed = np.array([[0.8, 0.2], [0.2, 0.4]], dtype=np.complex128)
        rho_purified = self.engine.purify_density_operator(rho_mixed)

        assert np.isclose(np.trace(rho_purified).real, 1.0)
        evals = np.linalg.eigvalsh(rho_purified)
        assert np.all(evals >= 0.0)

    def test_evolve_tomita_takesaki_flow(self):
        """Verifica la rotación de Wick $t \mapsto -i\\beta$ del flujo modular."""
        rho = np.eye(4, dtype=np.complex128) / 4.0
        A = np.eye(4, dtype=np.complex128)

        evolved_A = self.engine.evolve_tomita_takesaki_flow(A, rho, time_parameter=-1j * 1.0)
        assert evolved_A.shape == (4, 4)
        assert np.allclose(evolved_A, A)

    def test_compute_quantum_relative_entropy(self):
        """Verifica la entropía relativa de Umegaki $S(\\rho \\parallel \\sigma) \ge 0$."""
        rho = np.eye(4, dtype=np.complex128) / 4.0
        sigma = np.eye(4, dtype=np.complex128) / 4.0

        umegaki_S, uhlmann_F = self.engine.compute_quantum_relative_entropy(rho, sigma)
        assert np.isclose(umegaki_S, 0.0, atol=1e-7)
        assert np.isclose(uhlmann_F, 1.0, atol=1e-7)
