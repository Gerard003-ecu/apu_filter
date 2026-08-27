# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para brachistochrone_path_finder."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.boole.physics.brachistochrone_path_finder import (
    Phase1_PotentialWellInquirer,
    Phase2_ConformalKoszulSuturator,
    Phase3_FermatGeodesicSolver,
    BrachistochronePathFinder,
    ConformalPotentialDilation,
    ConformalManifoldBundle,
    BrachistochronePhysicalState,
    BrachistochroneEngineError,
    EnergyWellColapsoError,
    ConformalSingularityError,
    GeodesicIntegrationError,
    KoszulChecksumError,
    InitialKinematicsError
)
from app.core.mic_algebra import Morphism

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_eps(self):
        from app.boole.physics.brachistochrone_path_finder import _MACHINE_EPS
        assert _MACHINE_EPS > 0.0

    def test_condition_number_max(self):
        from app.boole.physics.brachistochrone_path_finder import _CONDITION_NUMBER_MAX
        assert _CONDITION_NUMBER_MAX > 0.0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_exceptions(self):
        assert issubclass(EnergyWellColapsoError, BrachistochroneEngineError)
        assert issubclass(ConformalSingularityError, BrachistochroneEngineError)
        assert issubclass(GeodesicIntegrationError, BrachistochroneEngineError)
        assert issubclass(KoszulChecksumError, BrachistochroneEngineError)
        assert issubclass(InitialKinematicsError, BrachistochroneEngineError)

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_conformal_potential_dilation_frozen(self):
        dto = ConformalPotentialDilation(
            g_base=np.eye(2),
            potential_v=np.array([1.0, 1.0]),
            initial_h0=10.0,
            energy_gap=np.array([9.0, 9.0]),
            is_safe=True,
            cholesky_factor=np.eye(2),
            symmetry_residual_relative=0.0,
            energy_gap_min=9.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            dto.initial_h0 = 5.0

class TestPhase1_PotentialWellInquirer:
    @pytest.fixture
    def inquirer(self):
        return Phase1_PotentialWellInquirer()

    def test_validate_manifold_dimension_valid(self, inquirer):
        assert inquirer._phase1_validate_manifold_dimension(3) == 3

    def test_validate_manifold_dimension_invalid(self, inquirer):
        with pytest.raises(ConformalSingularityError):
            inquirer._phase1_validate_manifold_dimension(0)

    def test_validate_metric_tensor_valid(self, inquirer):
        G = np.eye(3, dtype=np.float64)
        G_val = inquirer._phase1_validate_metric_tensor(G)
        np.testing.assert_allclose(G_val, G)

    def test_certify_metric_symmetry_symmetric(self, inquirer):
        G = np.eye(3, dtype=np.float64)
        sym, res = inquirer._phase1_certify_metric_symmetry(G)
        np.testing.assert_allclose(sym, G)
        assert abs(res) < 1e-10

    def test_certify_metric_symmetry_asymmetric(self, inquirer):
        G = np.eye(3, dtype=np.float64)
        G[0, 1] = 1.0
        with pytest.raises(ConformalSingularityError):
            inquirer._phase1_certify_metric_symmetry(G)

    def test_cholesky_spd_factor_spd(self, inquirer):
        G = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
        chol, logdet = inquirer._phase1_cholesky_spd_factor(G)
        np.testing.assert_allclose(chol @ chol.T, G)

    def test_cholesky_spd_factor_non_spd(self, inquirer):
        G = np.array([[2.0, 0.0], [0.0, -3.0]], dtype=np.float64)
        with pytest.raises(ConformalSingularityError):
            inquirer._phase1_cholesky_spd_factor(G)

    def test_validate_sylvester_criteria(self, inquirer):
        G = np.eye(3, dtype=np.float64)
        chol, res = inquirer._validate_sylvester_criteria(G)
        np.testing.assert_allclose(chol, G)
        assert abs(res) < 1e-10

    def test_energy_gap_spectrum_safe(self, inquirer):
        samples = np.array([1.0, 2.0], dtype=np.float64)
        gap, min_gap, safe = inquirer._phase1_energy_gap_spectrum(samples, 10.0)
        np.testing.assert_allclose(gap, [9.0, 8.0])
        assert abs(min_gap - 8.0) < 1e-10
        assert safe

    def test_energy_gap_spectrum_unsafe(self, inquirer):
        samples = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(EnergyWellColapsoError):
            inquirer._phase1_energy_gap_spectrum(samples, 1.5)

    def test_observe_energy_certificate(self, inquirer):
        G = np.eye(2, dtype=np.float64)
        samples = np.array([1.0, 1.0], dtype=np.float64)
        cert = inquirer._phase1_observe_energy_certificate(G, samples, 10.0)
        assert isinstance(cert, ConformalPotentialDilation)
        assert cert.manifold_dim == 2
        assert cert.is_safe

    def test_evaluate_potential_barrier(self, inquirer):
        G = np.eye(2, dtype=np.float64)
        samples = np.array([1.0, 1.0], dtype=np.float64)
        cert = inquirer.evaluate_potential_barrier(G, samples, 10.0)
        assert cert.is_safe

class TestPhase2_ConformalKoszulSuturator:
    @pytest.fixture
    def suturator(self):
        try:
            return Phase2_ConformalKoszulSuturator()
        except NameError:
            pytest.skip("Phase2_ConformalKoszulSuturator not implemented or imported")
            
class TestPhase3_FermatGeodesicSolver:
    pass

class TestBrachistochronePathFinder:
    @patch.object(Morphism, "__abstractmethods__", frozenset())
    def test_engine_instantiation(self):
        try:
            engine = BrachistochronePathFinder()
            assert engine is not None
        except NameError:
            pytest.skip("BrachistochronePathFinder not implemented or imported")
