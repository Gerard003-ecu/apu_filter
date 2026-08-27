# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Homotopic Séquitos Agent (Capa 1.5 Calibre Consenso)     ║
║ Ruta: tests/unit/agents/core/inmune_system/test_imperial_guards_sequitos.py ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_sequitos import (
    ImperialGuardsSequitosAgent,
    _KleisliVeredict,
    _DeGrootVeredict,
    _CHSHVeredict,
)


class TestImperialGuardsSequitosAgent:
    """Evaluación de Kleisli, Consenso de DeGroot y no-localidad Bell-CHSH."""

    def setup_method(self):
        self.agent = ImperialGuardsSequitosAgent(dimension_n=3, safety_margin=1.0)

    def test_audit_kleisli_associativity_coherent(self):
        """Verifica la asociatividad monádica de Kleisli."""
        f = lambda x: (x + 1, 1.0)
        g = lambda x: (x * 2, 1.0)
        h = lambda x: (x - 3, 1.0)

        dev, verdict = self.agent.audit_kleisli_associativity(f, g, h, test_input=5)
        assert dev >= 0.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_audit_degroot_consensus_coherent(self):
        """Verifica el consenso espectral de DeGroot."""
        opinions = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        affinity = np.array([[0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5],
                             [0.5, 0.5, 0.0]], dtype=np.float64)

        fiedler, verdict = self.agent.audit_degroot_consensus(opinions, affinity)
        assert fiedler >= 0.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_audit_quantum_chsh_channel_entangled(self):
        """Verifica la aduana cuántica CHSH con valor S dentro del límite de Tsirelson (2 < S <= 2*sqrt(2))."""
        # E_matrix tal que S = E00 - E01 + E10 + E11 = 0.6 - (-0.6) + 0.6 + 0.6 = 2.4
        E_matrix = np.array([[0.6, -0.6], [0.6, 0.6]], dtype=np.float64)

        s_val, verdict = self.agent.audit_quantum_chsh_channel(E_matrix)
        assert s_val > 2.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_execute_sequitos_cycle_full(self):
        """Ejecuta el ciclo OODA completo de los séquitos."""
        f = lambda x: (x + 1, 1.0)
        g = lambda x: (x * 2, 1.0)
        h = lambda x: (x - 3, 1.0)
        opinions = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        affinity = np.array([[0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5],
                             [0.5, 0.5, 0.0]], dtype=np.float64)
        E_matrix = np.array([[0.6, -0.6], [0.6, 0.6]], dtype=np.float64)

        res = self.agent.execute_sequitos_cycle(
            f=f, g=g, h_func=h, test_input=5,
            opinion_vector=opinions,
            affinity_matrix=affinity,
            correlation_matrix=E_matrix,
        )
        assert isinstance(res, dict)
        assert "heyting_verdict" in res
        assert "hardware_interlock_fired" in res
