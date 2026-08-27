# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Homotopic Tesserarios Agent (Homotopía de Rham)         ║
║ Ruta: tests/unit/agents/core/inmune_system/test_imperial_guards_tesserarios.py║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_tesserarios import (
    HomotopicTesserariosAgent,
    TesserariosCoherenceChamber,
    execute_tesserarios_cycle,
)


class TestHomotopicTesserariosAgent:
    """Evaluación de Quillen, Asociaedros de Stasheff y Obstrucción de Gerbes Čech."""

    def setup_method(self):
        self.n = 4
        self.agent = HomotopicTesserariosAgent(dimension_n=self.n, safety_margin=1.0)

    def test_ingest_tensors_and_audit(self):
        """Infiere tensores homotópicos y verifica las tres aduanas."""
        jacobian = np.eye(self.n, dtype=np.float64)
        m3_tensor = np.zeros((self.n, self.n, self.n, self.n), dtype=np.float64)
        cech_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        jet = self.agent.ingest_tensors(jacobian, m3_tensor, cech_matrix)
        sheaf = self.agent.compile_tesserarios_sheaf(jet)

        assert sheaf.quillen.verdict in ("COHERENT", "DEGRADED", "VETOED")
        assert sheaf.stasheff.verdict in ("COHERENT", "DEGRADED", "VETOED")
        assert sheaf.gerbe.verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_tesserarios_coherence_chamber_pipeline(self):
        """Ejecuta el ciclo de coherencia homotópico completo."""
        chamber = TesserariosCoherenceChamber.assemble_from_spectral_seed(dimension_n=4, safety_margin=1.0)
        jacobian = np.eye(4, dtype=np.float64)
        m3_tensor = np.zeros((4, 4, 4, 4), dtype=np.float64)
        cech_matrix = np.zeros((4, 4), dtype=np.float64)

        res = chamber.process_coherence_cycle(jacobian, m3_tensor, cech_matrix)
        assert isinstance(res, dict)
        assert "heyting_verdict" in res
        assert res["heyting_verdict"] == "COHERENT"
        assert res["hardware_interlock_fired"] is False

    def test_execute_tesserarios_cycle_facade(self):
        """Prueba la fachada pública execute_tesserarios_cycle."""
        jacobian = np.eye(4, dtype=np.float64)
        m3_tensor = np.zeros((4, 4, 4, 4), dtype=np.float64)
        cech_matrix = np.zeros((4, 4), dtype=np.float64)

        res = execute_tesserarios_cycle(self.agent, jacobian, m3_tensor, cech_matrix)
        assert res["heyting_verdict"] == "COHERENT"
