# -*- coding: utf-8 -*-
"""
Conftest para tests/unit/core/

Sutura #5: Firma extendida de RiskChallenger.challenge_verdict
================================================================

Contexto:
    La firma de producción de ``RiskChallenger.challenge_verdict`` es:

        challenge_verdict(
            report, financial_metrics, thermal_state,
            topo_bundle, session_context=None,
        )

    Los tests legacy llaman ``challenger.challenge_verdict(naive_report)``
    con un único argumento. Esta sutura NO modifica producción; en su lugar
    crea un wrapper de tests que detecta llamadas con un solo argumento y
    completa con mocks dimensionales coherentes.

Doctrina:
    Producción es Sagrada: NO se modifica ``app/strategy/business_agent.py``.
    Los tests se ajustan envolviendo el método en un shim de mocks.

Estrategia:
    Monkeypatch ``RiskChallenger.challenge_verdict`` con una función
    adaptativa que detecta el número de argumentos posicionales y completa
    con valores por defecto.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patch_risk_challenger_signature():
    """
    Shim adaptativo para ``challenge_verdict``: si los tests llaman con
    1 argumento (forma legacy), completa con mocks dimensionales.
    """
    try:
        from app.strategy.business_agent import RiskChallenger
    except ImportError:
        yield
        return

    import inspect
    from functools import wraps

    _original = RiskChallenger.challenge_verdict
    _sig = inspect.signature(_original)

    @wraps(_original)
    def _adaptive_challenge_verdict(self, *args, **kwargs):
        # Si ya viene con los 4+ argumentos correctos, llamar directamente.
        if len(args) >= 5 or "financial_metrics" in kwargs:
            return _original(self, *args, **kwargs)

        # Forma legacy: 1 argumento posicional (report).
        # Inyectar MagicMocks específicos para los 4 argumentos dimensionales
        # con atributos básicos que la producción consulta.
        fin = MagicMock()
        fin.irr = 0.10
        fin.npv = 1_000_000.0
        fin.payback_period = 8.0
        fin.profitability_index = 1.2
        fin.var_95 = 0.05
        fin.cvar_95 = 0.07
        fin.roi = 0.15
        fin.modified_irr = 0.09

        therm = MagicMock()
        therm.entropy = 0.5
        therm.temperature = 1.0
        therm.free_energy = -0.1
        therm.dissipation_rate = 0.02
        therm.metastability_index = 0.9
        therm.internal_energy = 100.0

        topo = MagicMock()
        topo.pyramid_stability = 0.7
        topo.structural_coherence = 0.8
        topo.betti = MagicMock()
        topo.betti.beta_0 = 1
        topo.betti.beta_1 = 0
        topo.betti.beta_2 = 0
        topo.spectral = MagicMock()
        topo.spectral.number_of_components = 1
        topo.graph_metrics = MagicMock()
        topo.persistence = None

        report = args[0] if args else kwargs.get("report")

        return _original(
            self,
            report,
            fin,
            therm,
            topo,
            session_context=None,
        )

    # Sólo parchear si la firma original lo requiere
    required_params = [
        p for p in _sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.name != "self"
    ]
    if len(required_params) >= 4:
        RiskChallenger.challenge_verdict = _adaptive_challenge_verdict

    try:
        yield
    finally:
        RiskChallenger.challenge_verdict = _original