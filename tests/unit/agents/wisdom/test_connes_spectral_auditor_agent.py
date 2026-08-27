# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Connes Spectral Auditor Agent (Geometría No Conmutativa)║
║ Ruta: tests/unit/agents/wisdom/test_connes_spectral_auditor_agent.py        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.wisdom.tomita_takesaki_telescopic_engine import TomitaTakesakiTelescopicEngine
from app.agents.wisdom.connes_spectral_auditor_agent import (
    ConnesSpectralAuditorAgent,
    Phase1_SpectralTripleBinder,
    Phase2_KMSEquilibriumAuditor,
    Phase3_DixmierTraceIntegrator,
    SpectralTripleData,
    KMSThermalBundle,
    ConnesAuditState,
    SemanticDiscontinuityError,
    SpectralTripleError,
    KMSEquilibriumViolation,
)


class TestPhase1SpectralTripleBinder:
    """Evaluación del triple espectral y el operador de Dirac."""

    def setup_method(self):
        self.binder = Phase1_SpectralTripleBinder()

    def test_bind_spectral_triple_success(self):
        """Demuestra la construcción del operador de Dirac y norma lipschitziana."""
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=np.complex128)
        X = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

        triple = self.binder.bind_spectral_triple(rho_mac=rho, X_observable=X)
        assert isinstance(triple, SpectralTripleData)
        assert triple.hilbert_space_dim == 2
        assert triple.dirac_condition_number >= 1.0
        assert triple.is_differentiable is True

    def test_non_psd_density_matrix_raises_spectral_error(self):
        """Detona SpectralTripleError si rho no es definida positiva."""
        rho_invalid = np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
        X = np.eye(2, dtype=np.complex128)

        with pytest.raises(SpectralTripleError):
            self.binder.bind_spectral_triple(rho_mac=rho_invalid, X_observable=X)


class TestPhase2KMSEquilibriumAuditor:
    """Evaluación del estado térmico de KMS y flujo modular de Tomita-Takesaki."""

    def setup_method(self):
        self.engine = TomitaTakesakiTelescopicEngine()
        self.auditor = Phase2_KMSEquilibriumAuditor()
        self.binder = Phase1_SpectralTripleBinder()

    def test_certify_kms_equilibrium_success(self):
        """Verifica la condición KMS con observable conmutativo en la base propia."""
        rho = np.array([[0.6, 0.0], [0.0, 0.4]], dtype=np.complex128)
        X = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)

        triple = self.binder.bind_spectral_triple(rho_mac=rho, X_observable=X)
        bundle = self.auditor.certify_kms_equilibrium(
            spectral_triple=triple,
            engine=self.engine,
            lambda_zoom=0.1,
            beta=1.0,
        )
        assert isinstance(bundle, KMSThermalBundle)
        assert bundle.kms_canonical.is_kms_compliant is True
        assert bundle.is_zoom_thermally_stable is True


class TestConnesSpectralAuditorAgentPipeline:
    """Integración completa de la geometría no conmutativa de Connes."""

    def setup_method(self):
        self.engine = TomitaTakesakiTelescopicEngine()
        self.agent = ConnesSpectralAuditorAgent(engine=self.engine, kms_beta=1.0, spectral_dim_p=1.0)

    def test_execute_spectral_audit_full_success(self):
        """Ejecuta la tubería completa de Connes."""
        rho = np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.complex128)
        X = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)

        state = self.agent.execute_spectral_audit(
            rho_mac=rho,
            X_observable=X,
            lambda_zoom=0.2,
        )
        assert isinstance(state, ConnesAuditState)
        assert state.is_epistemologically_safe is True
        assert state.dixmier_trace.dixmier_volume >= 0.0
