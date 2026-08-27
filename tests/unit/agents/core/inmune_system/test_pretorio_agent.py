# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para pretorio_agent."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
import scipy.linalg as la
from unittest.mock import MagicMock, patch

from app.agents.core.inmune_system.pretorio_agent import (
    _MACHINE_EPS, _WILKINSON_DRIFT_LIMIT, _WILKINSON_DEFLATION_SCALE,
    _MIN_SINGULAR_VALUE_FLOOR, _STRUCTURE_ATOL, _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
    _HYPER_DEGRADATION_FACTOR, _ACYCLIC_SOFT_VETO, _ACYCLIC_SOFT_DEGRADE,
    _BROUWER_HARD_TOLERANCE, _BROUWER_DEGRADED_TOLERANCE,
    HeytingVerdict, _HodgeSpectrum, _BrouwerSpectrum, _PretorioJet,
    _PretorioSpectralCore, _HypercohomologyResult, _BrouwerResult,
    _UltrafilterResult, _PretorioEdict
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_types_and_values(self):
        assert _MACHINE_EPS > 0
        assert _WILKINSON_DRIFT_LIMIT == 1.0e-12
        assert _WILKINSON_DEFLATION_SCALE == 10.0
        assert _MIN_SINGULAR_VALUE_FLOOR == 1.0e-12

class TestExceptionHierarchy:
    """Verify exception inheritance chain (if any)."""
    def test_heyting_verdict_unknown(self):
        with pytest.raises(ValueError):
            HeytingVerdict.from_token("UNKNOWN")

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_heyting_verdict_methods(self):
        v = HeytingVerdict.COHERENT
        v2 = HeytingVerdict.DEGRADED
        assert v.meet(v2) == HeytingVerdict.DEGRADED
        assert v.join(v2) == HeytingVerdict.COHERENT
        assert v.negate() == HeytingVerdict.VETOED
        assert v.implies(v2) == HeytingVerdict.DEGRADED
        assert v.booleanize_closed() == HeytingVerdict.COHERENT
        assert v2.booleanize_closed() == HeytingVerdict.VETOED
        assert v2.booleanize_open() == HeytingVerdict.COHERENT
        assert HeytingVerdict.from_token_or_bottom("UNKNOWN") == HeytingVerdict.VETOED

    def test_hodge_spectrum_frozen(self):
        obj = _HodgeSpectrum(0.1, (), (), (), (), (), 0.1, True, ())
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.max_nilpotency = 0.2

    def test_brouwer_spectrum_frozen(self):
        obj = _BrouwerSpectrum(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, True)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.residual = 0.2

    def test_pretorio_jet_frozen(self):
        hodge = _HodgeSpectrum(0.1, (), (), (), (), (), 0.1, True, ())
        brouwer = _BrouwerSpectrum(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, True)
        obj = _PretorioJet(2, hodge, brouwer, None)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.n = 3

class TestPhase1_PretorioSpectralCore:
    """Granular tests for Phase 1 methods."""
    def test_frobenius(self):
        val = _PretorioSpectralCore.frobenius(np.array([3.0, 4.0]))
        assert abs(val - 5.0) < 1e-12

    def test_hermitize(self):
        mat = np.array([[1, 1j], [-1j, 1]])
        herm = _PretorioSpectralCore.hermitize(mat)
        np.testing.assert_allclose(herm, mat)

    def test_wilkinson_floor(self):
        val = _PretorioSpectralCore.wilkinson_floor(1.0, 2)
        assert val > 0

    def test_numerical_rank(self):
        mat = np.eye(2)
        rank = _PretorioSpectralCore.numerical_rank(mat)
        assert rank == 2

    def test_regularize_density(self):
        rho = np.eye(2)
        reg, evals = _PretorioSpectralCore.regularize_density(rho, 2)
        np.testing.assert_allclose(reg, np.eye(2) / 2.0)
        np.testing.assert_allclose(evals, [0.5, 0.5])

    def test_as_matrix(self):
        mat = _PretorioSpectralCore._as_matrix("test", np.eye(2))
        assert mat.shape == (2, 2)
        with pytest.raises(ValueError):
            _PretorioSpectralCore._as_matrix("test", np.array([1, 2]))

    def test_space_dims(self):
        d1 = np.ones((3, 2))
        d2 = np.ones((4, 3))
        dims = _PretorioSpectralCore._space_dims([d1, d2])
        assert dims == (2, 3, 4)

    def test_validate_complex(self):
        d1 = np.ones((3, 2))
        d2 = np.ones((4, 3))
        mats, valid, fault = _PretorioSpectralCore._validate_complex([d1, d2])
        assert valid is True

    def test_validate_complex_invalid(self):
        d1 = np.ones((3, 2))
        d2 = np.ones((4, 4))
        mats, valid, fault = _PretorioSpectralCore._validate_complex([d1, d2])
        assert valid is False

    def test_nilpotency_residuals(self):
        d1 = np.zeros((3, 2))
        d2 = np.zeros((4, 3))
        abs_res, rel_res = _PretorioSpectralCore.nilpotency_residuals([d1, d2])
        assert abs_res[0] < 1e-12

    def test_hodge_laplacian(self):
        d1 = np.eye(2)
        delta = _PretorioSpectralCore.hodge_laplacian([d1], [2, 2], 0)
        np.testing.assert_allclose(delta, np.eye(2))

    def test_hodge_invariants(self):
        d1 = np.zeros((2, 2))
        betti, soft, gaps = _PretorioSpectralCore.hodge_invariants([d1], [2, 2])
        assert betti == (2, 2)

    def test_compute_hodge_spectrum(self):
        d1 = np.zeros((2, 2))
        spec = _PretorioSpectralCore.compute_hodge_spectrum([d1])
        assert spec.complex_valid is True

    def test_compute_brouwer_spectrum(self):
        rho = np.eye(2) / 2.0
        t_map = np.eye(2)
        spec = _PretorioSpectralCore.compute_brouwer_spectrum(rho, t_map, 2)
        assert abs(spec.residual) < 1e-12
        assert spec.banach_contraction is False # norm of eye is 1.0, not < 1.0

    def test_heyting_meet_tokens(self):
        res = _PretorioSpectralCore.heyting_meet_tokens(["COHERENT", "DEGRADED"])
        assert res == HeytingVerdict.DEGRADED

    def test_heyting_lower_median(self):
        res = _PretorioSpectralCore.heyting_lower_median(["COHERENT", "DEGRADED", "VETOED"])
        assert res == HeytingVerdict.DEGRADED

    def test_weighted_social_choice(self):
        layer_v = {"capa_1_guards": "COHERENT", "capa_2_centurions": "VETOED"}
        winner, masses, anom = _PretorioSpectralCore.weighted_social_choice(layer_v)
        assert winner == HeytingVerdict.VETOED

    def test_assemble_pretorio_jet(self):
        d1 = np.zeros((2, 2))
        rho = np.eye(2) / 2.0
        t_map = np.eye(2)
        jet = _PretorioSpectralCore.assemble_pretorio_jet(2, [d1], rho, t_map)
        assert jet.n == 2
        assert jet.input_fault is None
