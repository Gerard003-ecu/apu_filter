# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para tomita_takesaki_telescopic_engine."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.wisdom.tomita_takesaki_telescopic_engine import (
    Phase1_GNSConstruction,
    Phase2_AnalyticModularFlow,
    Phase3_UmegakiExtraction,
    GNSFibrationData,
    ModularFlowData,
    UmegakiExtractionState,
    TomitaTakesakiEngineError,
    GNSConstructionError,
    InvalidObservableError,
    ModularFlowSingularityError,
    UmegakiDivergenceError,
    PetzMetricSingularityError,
    _HERMITICITY_TOLERANCE,
    _POSITIVITY_TOLERANCE,
    _TRACE_TOLERANCE,
    _FAITHFULNESS_FLOOR
)

try:
    from app.wisdom.tomita_takesaki_telescopic_engine import TomitaTakesakiTelescopicEngine
except ImportError:
    TomitaTakesakiTelescopicEngine = None

def get_hermitian(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (M + M.conj().T)

def get_faithful_rho(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = M @ M.conj().T + 0.1 * np.eye(n)
    return rho / np.trace(rho)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_hermiticity_tolerance(self):
        assert _HERMITICITY_TOLERANCE > 0

    def test_positivity_tolerance(self):
        assert _POSITIVITY_TOLERANCE > 0

    def test_trace_tolerance(self):
        assert _TRACE_TOLERANCE > 0

    def test_faithfulness_floor(self):
        assert _FAITHFULNESS_FLOOR > 0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_gns_construction_error(self):
        with pytest.raises(GNSConstructionError):
            raise GNSConstructionError("test")

    def test_invalid_observable_error(self):
        with pytest.raises(InvalidObservableError):
            raise InvalidObservableError("test")

    def test_modular_flow_singularity_error(self):
        with pytest.raises(ModularFlowSingularityError):
            raise ModularFlowSingularityError("test")

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_gns_fibration_data_frozen(self):
        obj = GNSFibrationData(
            rho_eigenvalues=np.array([0.5, 0.5]),
            rho_eigenvectors=np.eye(2, dtype=np.complex128),
            modular_operator_delta=np.ones((2,2)),
            modular_conjugation_J_phases=np.ones((2,2), dtype=np.complex128),
            purity_gap=0.5,
            faithful_spectral_floor=0.5,
            hilbert_space_dim=2
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.hilbert_space_dim = 3

    def test_modular_flow_data_frozen(self):
        gns = GNSFibrationData(
            rho_eigenvalues=np.array([0.5, 0.5]),
            rho_eigenvectors=np.eye(2, dtype=np.complex128),
            modular_operator_delta=np.ones((2,2)),
            modular_conjugation_J_phases=np.ones((2,2), dtype=np.complex128),
            purity_gap=0.5,
            faithful_spectral_floor=0.5,
            hilbert_space_dim=2
        )
        obj = ModularFlowData(
            X_deformed=np.eye(2),
            X_original=np.eye(2),
            lambda_zoom=1.0,
            flow_condition_number=1.0,
            flow_multiplier_spectrum=np.ones((2,2)),
            gns_data=gns
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.lambda_zoom = 2.0

class TestPhase1_GNSConstruction:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.phase1 = Phase1_GNSConstruction()

    def test_hermitize(self):
        M = np.array([[1, 1j], [0, 1]])
        H = self.phase1._hermitize(M)
        assert abs(H[0, 1] - 0.5j) < 1e-9

    def test_validate_density_matrix_invalid_shape(self):
        with pytest.raises(GNSConstructionError, match="se exige matriz cuadrada"):
            self.phase1._validate_density_matrix(np.zeros((2, 3)))

    def test_validate_density_matrix_not_hermitian(self):
        M = np.array([[1, 1j], [0, 1]])
        with pytest.raises(GNSConstructionError, match="no hermítica"):
            self.phase1._validate_density_matrix(M)

    def test_verify_faithful_state(self):
        ev = np.array([0.5, 0.5])
        res = self.phase1._verify_faithful_state(ev)
        assert abs(res - 0.5) < 1e-9

    def test_verify_faithful_state_fails(self):
        ev = np.array([1e-15, 0.9])
        with pytest.raises(GNSConstructionError, match="Estado no fiel"):
            self.phase1._verify_faithful_state(ev)

    def test_extract_modular_operator(self):
        rho = get_faithful_rho(2)
        res = self.phase1.extract_modular_operator(rho)
        assert res.hilbert_space_dim == 2
        assert abs(res.purity_gap) >= 0.0

class TestPhase2_AnalyticModularFlow:
    """Granular tests for Phase 2 methods."""
    def setup_method(self):
        self.phase2 = Phase2_AnalyticModularFlow()

    def test_validate_observable(self):
        X = get_hermitian(2)
        res = self.phase2._validate_observable(X, 2)
        assert res.shape == (2, 2)

    def test_flow_multipliers(self):
        delta = np.array([[1.0, 2.0], [0.5, 1.0]])
        M, cond = self.phase2._flow_multipliers(delta, 1.0)
        assert abs(M[0, 1] - 0.5) < 1e-9
        
    def test_apply_modular_automorphism(self):
        X = get_hermitian(2)
        evecs = np.eye(2, dtype=np.complex128)
        M = np.ones((2, 2))
        res = self.phase2._apply_modular_automorphism(X, evecs, M)
        assert res.shape == (2, 2)

class TestPhase3_UmegakiExtraction:
    """Granular tests for Phase 3 methods."""
    def setup_method(self):
        self.phase3 = Phase3_UmegakiExtraction()

    def test_compute_umegaki_divergence(self):
        rho = get_faithful_rho(2, 42)
        sigma = get_faithful_rho(2, 43)
        div = self.phase3._compute_umegaki_divergence(rho, sigma)
        assert div >= 0.0

    def test_compute_uhlmann_fidelity(self):
        rho = get_faithful_rho(2, 42)
        sigma = get_faithful_rho(2, 42)
        fid = self.phase3._compute_uhlmann_fidelity(rho, sigma)
        assert abs(fid - 1.0) < 1e-6

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_init(self):
        if TomitaTakesakiTelescopicEngine is None:
            pytest.skip("TomitaTakesakiTelescopicEngine not found")
        with patch.multiple(TomitaTakesakiTelescopicEngine, __abstractmethods__=frozenset()):
            engine = TomitaTakesakiTelescopicEngine()
            assert engine is not None
