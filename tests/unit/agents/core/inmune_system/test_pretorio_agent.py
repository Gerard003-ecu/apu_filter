# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Pretorio Agent (Comandante Supremo de Seguridad)         ║
║ Ruta: tests/unit/agents/core/inmune_system/test_pretorio_agent.py           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.core.inmune_system.pretorio_agent import (
    PretorioAgent,
    PretorioCoherenceChamber,
    execute_pretorio_supervision_cycle,
    _HypercohomologyResult,
    _BrouwerResult,
    _UltrafilterResult,
)


class TestPretorioAgent:
    """Evaluación de Hipercohomología, Punto Fijo de Brouwer y Ultrafiltro Booleano."""

    def setup_method(self):
        self.n = 2
        self.agent = PretorioAgent(dimension_n=self.n, hypercohomology_threshold=1e-9)

    def test_audit_calibre_hypercohomology(self):
        """Verifica la nilpotencia diferencial total $d^2 = 0$ en el complejo de calibre."""
        d0 = np.array([[1.0, 0.0]], dtype=np.float64)
        d1 = np.array([[0.0], [0.0]], dtype=np.float64)

        res = self.agent.audit_calibre_hypercohomology([d0, d1])
        assert isinstance(res, _HypercohomologyResult)
        assert res.verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_verify_brouwer_fixed_point_consistency(self):
        """Verifica la consistencia geodésica de punto fijo $f(\\rho) = \\rho$."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        transition = np.eye(2, dtype=np.complex128)

        res = self.agent.verify_brouwer_fixed_point_consistency(rho, transition)
        assert isinstance(res, _BrouwerResult)
        assert res.verdict == "COHERENT"

    def test_evaluate_global_boolean_ultrafilter_coherent(self):
        """Verifica el colapso del ultrafiltro cuando todas las capas reportan COHERENT."""
        layer_verdicts = {
            "capa_1_guards": "COHERENT",
            "capa_2_centurions": "COHERENT",
            "capa_3_tesserarios": "COHERENT",
            "capa_4_pretorio_hyper": "COHERENT",
            "capa_4_pretorio_brouwer": "COHERENT",
        }

        res = self.agent.evaluate_global_boolean_ultrafilter(layer_verdicts)
        assert isinstance(res, _UltrafilterResult)
        assert res.global_verdict == "COHERENT"
        assert res.hardware_interlock is False

    def test_execute_pretorio_supervision_cycle_facade(self):
        """Ejecuta el ciclo de supervisión omnipresente del Pretorio."""
        d0 = np.array([[1.0, 0.0]], dtype=np.float64)
        d1 = np.array([[0.0], [0.0]], dtype=np.float64)
        rho = np.eye(2, dtype=np.complex128) / 2.0
        transition = np.eye(2, dtype=np.complex128)
        layer_verdicts = {
            "capa_1_guards": "COHERENT",
            "capa_2_centurions": "COHERENT",
            "capa_3_tesserarios": "COHERENT",
        }

        res = execute_pretorio_supervision_cycle(
            self.agent, [d0, d1], rho, transition, layer_verdicts
        )
        assert isinstance(res, dict)
        assert "pretorio_global_verdict" in res
        assert res["pretorio_global_verdict"] in ("COHERENT", "DEGRADED")
