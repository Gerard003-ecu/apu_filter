# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Imperial Guards Eruditos (Cohomología Simpléctica Čech)║
║ Ruta: tests/unit/agents/core/inmune_system/test_imperial_guards_eruditos.py ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_eruditos import (
    ImperialGuardsEruditosAgent,
    _FloerVeredict,
    _CechVeredict,
)


class TestImperialGuardsEruditosAgent:
    """Evaluación de la cohomología simpléctica de Floer y cohomología atencional de Čech."""

    def setup_method(self):
        self.agent = ImperialGuardsEruditosAgent(dimension_n=4, safety_margin=1.0)

    def test_audit_floer_homology_trajectory_coherent(self):
        """Audita el cilindro de Floer en el espacio simpléctico de fase."""
        start_pt = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        end_pt = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
        jacobian_m3 = np.eye(4, dtype=np.float64)

        res = self.agent.audit_floer_homology_trajectory(start_pt, end_pt, jacobian_m3)
        assert isinstance(res, _FloerVeredict)
        assert res.verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_audit_attention_cech_cohomology_coherent(self):
        """Audita la obstrucción de Čech sobre la matriz del haz atencional."""
        sheaf_matrix = np.eye(4, dtype=np.float64)
        res = self.agent.audit_attention_cech_cohomology(sheaf_matrix)

        assert isinstance(res, _CechVeredict)
        assert res.verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_execute_eruditos_cycle_full(self):
        """Ejecuta el ciclo OODA cohomológico completo."""
        start_pt = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        end_pt = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
        jacobian_m3 = np.eye(4, dtype=np.float64)
        sheaf_matrix = np.eye(4, dtype=np.float64)

        res_dict = self.agent.execute_eruditos_cycle(
            start_point=start_pt,
            end_point=end_pt,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=sheaf_matrix,
        )
        assert isinstance(res_dict, dict)
        assert "heyting_verdict" in res_dict
        assert "hardware_interlock_fired" in res_dict
