# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Tomita-Takesaki Telescopic Engine (Teoría Modular GNS)   ║
║ Ruta: tests/unit/wisdom/test_tomita_takesaki_telescopic_engine.py            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.wisdom.tomita_takesaki_telescopic_engine import (
    TomitaTakesakiTelescopicEngine,
    Phase1_GNSConstruction,
    Phase2_AnalyticModularFlow,
    Phase3_UmegakiExtraction,
    GNSFibrationData,
    ModularFlowData,
    UmegakiExtractionState,
    GNSConstructionError,
    ModularFlowSingularityError,
    UmegakiDivergenceError,
)


class TestPhase1GNSConstruction:
    """Evaluación de la fibración GNS y diagonalización de la matriz densidad."""

    def setup_method(self):
        self.gns_builder = Phase1_GNSConstruction()

    def test_gns_construction_faithful_state(self):
        """Construye la representación GNS sobre un estado fiel."""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=np.complex128)

        gns_data = self.gns_builder.extract_modular_operator(rho_mac=rho)
        assert isinstance(gns_data, GNSFibrationData)
        assert gns_data.hilbert_space_dim == 2
        assert gns_data.faithful_spectral_floor > 0.0
        assert gns_data.purity_gap > 0.0

    def test_gns_construction_singular_state_raises(self):
        """Detona GNSConstructionError ante matriz singular no fiel (autovalores 0)."""
        rho_singular = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

        with pytest.raises(GNSConstructionError):
            self.gns_builder.extract_modular_operator(rho_mac=rho_singular)


class TestPhase2AnalyticModularFlow:
    """Evaluación del flujo modular analítico de Tomita-Takesaki."""

    def setup_method(self):
        self.flow_engine = Phase2_AnalyticModularFlow()

    def test_modular_zoom_transformation(self):
        """Aplica la deformación modular."""
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.complex128)
        X = np.array([[1.0, 0.5], [0.5, 2.0]], dtype=np.complex128)

        gns_data = self.flow_engine.extract_modular_operator(rho_mac=rho)
        flow_data = self.flow_engine.execute_modular_zoom(
            gns_data,
            X,
            0.5,
        )
        assert isinstance(flow_data, ModularFlowData)
        assert flow_data.X_deformed.shape == (2, 2)
        assert flow_data.flow_condition_number >= 1.0


class TestTomitaTakesakiTelescopicEnginePipeline:
    """Integración completa del motor microscópico de Tomita-Takesaki."""

    def setup_method(self):
        self.engine = TomitaTakesakiTelescopicEngine()

    def test_execute_modular_audit_success(self):
        """Ejecuta la auditoría completa de divergencia de Umegaki."""
        rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=np.complex128)
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

        state = self.engine.execute_modular_audit(
            rho_mac=rho,
            X_observable=X,
            lambda_magnification=0.1,
        )
        assert isinstance(state, UmegakiExtractionState)
        assert state.is_epistemologically_safe is True
        assert state.umegaki_relative_entropy >= 0.0
        assert state.fidelity_uhlmann <= 1.0 + 1e-9
