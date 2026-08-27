# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: KCore Kinematic Engine (IDA-PBC, Hodge, Cinética)         ║
║ Ruta: tests/unit/alfa/kcore/test_kcore_kinematic_engine.py                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.alfa.kcore.kcore_kinematic_engine import (
    KCoreKinematicEngine,
    HeytingVerdict,
)


class TestKCoreKinematicEngine:
    """Evaluación del motor cinemático del núcleo (IDA-PBC port-Hamiltoniano y decomposición de Hodge)."""

    def setup_method(self):
        self.n = 4
        self.engine = KCoreKinematicEngine(dimension_n=self.n, reg_param=1e-15)

    def test_compute_ida_pbc_matching(self):
        """Verifica la ecuación de matching IDA-PBC $g \\alpha = (J_d - R_d) \\nabla H_d - (J - R) \\nabla H$."""
        g = np.eye(4, dtype=np.float64)
        G = np.eye(4, dtype=np.float64)
        J = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float64)
        R = np.eye(4, dtype=np.float64)
        J_d = J.copy()
        R_d = np.eye(4, dtype=np.float64) * 2.0
        grad_H = np.ones(4, dtype=np.float64)
        grad_Hd = np.zeros(4, dtype=np.float64)

        alpha, residual = self.engine.compute_ida_pbc_matching(
            g, G, J, R, J_d, R_d, grad_H, grad_Hd
        )
        assert alpha.shape == (4,)
        assert residual >= 0.0

    def test_compute_helmholtz_hodge_decomposition(self):
        """Verifica la descomposición de Hodge en el 1-complejo celular $C^1 = \\text{im}(d_0) \\oplus \\text{im}(\\delta_2) \\oplus \\mathcal{H}^1$."""
        boundary = np.array([
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
        flow = np.ones(4, dtype=np.float64)

        I_grad, I_curl, I_harm, ortho_err = self.engine.compute_helmholtz_hodge_decomposition(
            boundary, weights, flow
        )
        assert I_grad.shape == (4,)
        assert I_curl.shape == (4,)
        assert I_harm.shape == (4,)
        assert ortho_err >= 0.0

    def test_compute_compensated_kinetic_metrics(self):
        """Verifica la energía cinética $T = \\frac{1}{2} \\dot{q}^T M \\dot{q}$ y potencia $P = \\dot{q}^T F$."""
        M = np.eye(4, dtype=np.float64)
        v = np.ones(4, dtype=np.float64)
        F = np.ones(4, dtype=np.float64)

        T_kin, P_mech = self.engine.compute_compensated_kinetic_metrics(M, v, F)
        assert np.isclose(T_kin, 2.0)  # 0.5 * 4 * 1^2
        assert np.isclose(P_mech, 4.0)  # 4 * 1*1

    def test_execute_kinematic_cycle(self):
        """Ejecuta el ciclo OODA cinemático completo."""
        g = np.eye(4, dtype=np.float64)
        G = np.eye(4, dtype=np.float64)
        J = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float64)
        R = np.eye(4, dtype=np.float64)
        J_d = J.copy()
        R_d = np.eye(4, dtype=np.float64) * 2.0
        grad_H = np.ones(4, dtype=np.float64)
        grad_Hd = np.zeros(4, dtype=np.float64)

        boundary = np.array([
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
        flow = np.ones(4, dtype=np.float64)

        M = np.eye(4, dtype=np.float64)
        v = np.ones(4, dtype=np.float64)
        F = np.ones(4, dtype=np.float64)

        telemetry = self.engine.execute_kinematic_cycle(
            g, G, J, R, J_d, R_d, grad_H, grad_Hd,
            boundary, weights, flow,
            M, v, F
        )
        assert isinstance(telemetry, dict)
        assert "heyting_verdict" in telemetry
        assert telemetry["heyting_verdict"] in ("CERTIFIED", "COHERENT", "DEGRADED", "VETOED")
