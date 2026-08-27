# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para app.core.inmune_system.imperial_centurions_engine."""
from __future__ import annotations

import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.inmune_system.imperial_centurions_engine import (
    ImperialCenturionsEngine,
    _SymplecticFormCertificate,
    _PortHamiltonianGerm,
    _StructureCertificate,
    _IDAPBCResult,
    _SymplecticPreservationResult,
    _ModularSpectralGerm,
    _PurificationResult,
    _ModularFlowResult,
    _QuantumRelativeEntropyResult,
    _NumericalCore,
    _IDAPBCController,
    _SymplecticPreservationChecker,
    _DensityPurifier,
    _TomitaTakesakiFlow,
    _QuantumEntropyCalculator,
    _MACHINE_EPS,
    _HIGHAM_REG_FLOOR,
    _WILKINSON_DRIFT_LIMIT,
    _WILKINSON_DEFLATION_SCALE,
    _LOG_EXP_CLIP,
    _KMS_STRIP_TOL,
    _HERMITIAN_TOL,
    _DEFAULT_PURITY_MARGIN,
    _DEFAULT_BETA,
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_exist_and_positive(self):
        assert _MACHINE_EPS > 0
        assert _HIGHAM_REG_FLOOR > 0
        assert _WILKINSON_DRIFT_LIMIT > 0
        assert _WILKINSON_DEFLATION_SCALE > 0
        assert _LOG_EXP_CLIP > 0
        assert _KMS_STRIP_TOL > 0
        assert _HERMITIAN_TOL > 0
        assert _DEFAULT_PURITY_MARGIN > 0
        assert _DEFAULT_BETA > 0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_symplectic_form_certificate_frozen(self):
        cert = _SymplecticFormCertificate(0.1, 0.2, 1.0, 1.0, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.is_darboux = False

    def test_port_hamiltonian_germ_frozen(self):
        cert = _SymplecticFormCertificate(0.0, 0.0, 1.0, 1.0, True)
        germ = _PortHamiltonianGerm(2, 4, np.eye(4), 1e-10, cert)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            germ.n = 3

class TestPhase1_NumericalCore:
    """Granular tests for Phase 1 methods."""
    def test_kahan_sum(self):
        arr = np.array([1.0, 1e-16, -1.0], dtype=np.float64)
        assert abs(_NumericalCore.kahan_sum(arr) - 1e-16) < 1e-15
    
    def test_kahan_babuska_neumaier_sum(self):
        arr = np.array([1e10, 1.0, -1e10], dtype=np.float64)
        assert abs(_NumericalCore.kahan_babuska_neumaier_sum(arr) - 1.0) < 1e-9

    def test_generate_canonical_symplectic_form(self):
        omega = _NumericalCore.generate_canonical_symplectic_form(4)
        assert omega.shape == (4, 4)
        assert np.allclose(omega @ omega, -np.eye(4))
        assert np.allclose(omega.T, -omega)

    def test_certify_symplectic_form(self):
        omega = _NumericalCore.generate_canonical_symplectic_form(4)
        cert = _NumericalCore.certify_symplectic_form(omega)
        assert cert.is_darboux is True
        assert abs(cert.determinant - 1.0) < 1e-8

    def test_higham_nearest_spd(self):
        M = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        spd = _NumericalCore.higham_nearest_spd(M)
        assert np.allclose(spd, spd.T)
        evals = np.linalg.eigvalsh(spd)
        assert np.all(evals > 0)

class TestPhase2_IDAPBC:
    """Granular tests for Phase 2 methods."""
    def test_compute_control_law(self):
        engine = ImperialCenturionsEngine(2)
        q = np.array([0.1, 0.2], dtype=np.float64)
        p = np.array([0.3, 0.4], dtype=np.float64)
        grad_H = np.ones(4, dtype=np.float64)
        grad_Hd = np.ones(4, dtype=np.float64)
        g_actuator = np.eye(4, dtype=np.float64)
        J = np.zeros((4,4), dtype=np.float64)
        R = np.eye(4, dtype=np.float64)
        Jd = np.zeros((4,4), dtype=np.float64)
        Rd = np.eye(4, dtype=np.float64)
        G = np.eye(4, dtype=np.float64)

        alpha, exergy = engine.compute_ida_pbc_control_law(q, p, grad_H, grad_Hd, g_actuator, J, R, Jd, Rd, G)
        assert alpha.shape == (4, 1) or alpha.shape == (4, 4)
        assert exergy >= 0

class TestPhase3_Modular:
    """Granular tests for Phase 3 methods."""
    def test_purify_density_operator(self):
        engine = ImperialCenturionsEngine(2)
        rho = np.array([[1.5, 0.0, 0.0, 0.0],
                        [0.0, -0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        purified = engine.purify_density_operator(rho)
        evals = np.linalg.eigvalsh(purified)
        assert np.all(evals >= -1e-12)
        assert abs(np.sum(evals) - 1.0) < 1e-8

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_engine_initialization_invalid(self):
        with pytest.raises(ValueError):
            ImperialCenturionsEngine(0)

    def test_engine_valid(self):
        engine = ImperialCenturionsEngine(2)
        assert engine._n == 2
        assert engine._2n == 4
