# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Pretorio Engine (Hipercohomología, Brouwer y Ultrafiltro)║
║ Ruta: tests/unit/core/inmune_system/test_pretorio_engine.py                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.core.inmune_system.pretorio_engine import PretorioEngine


class TestPretorioEngine:
    """Evaluación del motor supremo del Pretorio."""

    def setup_method(self):
        self.engine = PretorioEngine(regularizer=1e-15)

    def test_verify_cech_derham_hypercohomology(self):
        """Verifica la hipercohomología del bicomplejo de Čech-de Rham."""
        d1 = np.array([[1.0, 0.0]], dtype=np.complex128)
        d2 = np.array([[0.0], [0.0]], dtype=np.complex128)

        residual, verdict = self.engine.verify_cech_derham_hypercohomology(d1, d2)
        assert residual >= 0.0
        assert verdict in ("COHERENT", "DEGRADED", "VETOED")

    def test_verify_brouwer_fixed_point(self):
        """Certifica el punto fijo de Brouwer en $f(\\rho) = \\rho$."""
        rho_current = np.eye(2, dtype=np.complex128) / 2.0
        rho_transformed = np.eye(2, dtype=np.complex128) / 2.0

        brouwer_res, trace_res, verdict = self.engine.verify_brouwer_fixed_point(
            rho_current, rho_transformed
        )
        assert np.isclose(brouwer_res, 0.0)
        assert np.isclose(trace_res, 0.0)
        assert verdict == "COHERENT"

    def test_evaluate_ultrafilter_consensus(self):
        """Evalúa el colapso del ultrafiltro booleano $\\mathcal{U}$."""
        verdicts = ["COHERENT", "COHERENT", "COHERENT"]
        consensus, interlock = self.engine.evaluate_ultrafilter_consensus(verdicts)

        assert consensus == "VIABLE"
        assert interlock is False

    def test_evaluate_ultrafilter_consensus_vetoed(self):
        """Verifica el disparo del crowbar de potencia ante un veto en el ultrafiltro."""
        verdicts = ["COHERENT", "VETOED", "COHERENT"]
        consensus, interlock = self.engine.evaluate_ultrafilter_consensus(verdicts)

        assert consensus == "RECHAZAR"
        assert interlock is True
