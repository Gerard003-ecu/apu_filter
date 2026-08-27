# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: KBase Thermodynamic Engine (Gibbs-Helmholtz y Clausius)   ║
║ Ruta: tests/unit/alfa/kbase/test_kbase_thermodynamic_engine.py              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.alfa.kbase.kbase_thermodynamic_engine import (
    KBaseThermodynamicEngine,
    HeytingVerdict,
)


class TestKBaseThermodynamicEngine:
    """Evaluación del motor termodinámico de base (potenciales de Gibbs-Helmholtz y Clausius-Duhem)."""

    def setup_method(self):
        self.n = 3
        self.engine = KBaseThermodynamicEngine(dimension_n=self.n, reg_param=1e-15)

    def test_compute_riemannian_pullback(self):
        """Verifica la congruencia métrica $T^\\flat = g T g$."""
        g = np.eye(3, dtype=np.float64)
        M = np.diag([1.0, 2.0, 3.0])

        pullback = self.engine.compute_riemannian_pullback(g, M, tensor_type="covariant")
        assert np.allclose(pullback, M)

    def test_compute_gibbs_free_energy(self):
        """Verifica los potenciales de Gibbs y entalpía $G = H - T S$, $H = U + P V$."""
        U, P, V, T, S = 100.0, 10.0, 2.0, 300.0, 0.1
        gibbs, H = self.engine.compute_gibbs_free_energy(U, P, V, T, S)

        assert H == 120.0  # 100 + 10*2
        assert gibbs == 90.0  # 120 - 300*0.1

    def test_evaluate_clausius_duhem(self):
        """Verifica la desigualdad de Clausius-Duhem $\\Phi \\ge 0$."""
        sigma_int = 0.05
        heat_q = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        grad_T = np.array([-1.0, 0.0, 0.0], dtype=np.float64)  # conducción ley de Fourier
        T = 300.0

        dissipation, is_coherent = self.engine.evaluate_clausius_duhem(sigma_int, heat_q, grad_T, T)
        assert dissipation > 0.0
        assert is_coherent is True

    def test_calculate_spectral_irreversibility(self):
        """Calcula el defecto de estacionariedad $\\mathcal{I} = \\|[H, \\rho]\\|_\\text{HS}^2$."""
        rho = np.eye(3, dtype=np.complex128) / 3.0
        H = np.eye(3, dtype=np.complex128)

        irrev = self.engine.calculate_spectral_irreversibility(rho, H)
        assert np.isclose(irrev, 0.0)

    def test_execute_thermodynamic_cycle(self):
        """Ejecuta el ciclo OODA termodinámico completo."""
        g = np.eye(3, dtype=np.float64)
        M = np.diag([1.0, 2.0, 3.0])
        heat_q = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        grad_T = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        rho = np.eye(3, dtype=np.complex128) / 3.0
        H = np.eye(3, dtype=np.complex128)

        telemetry = self.engine.execute_thermodynamic_cycle(
            G=g,
            tensor_M=M,
            tensor_type="covariant",
            internal_energy_U=100.0,
            pressure_P=10.0,
            volume_V=2.0,
            temperature_T=300.0,
            entropy_S=0.1,
            entropy_production_rate=0.05,
            heat_flux_q=heat_q,
            temp_gradient_gradT=grad_T,
            rho=rho,
            hamiltonian_H=H,
        )
        assert isinstance(telemetry, dict)
        assert "heyting_verdict" in telemetry
        assert telemetry["heyting_verdict"] in ("CERTIFIED", "COHERENT", "DEGRADED", "VETOED")
