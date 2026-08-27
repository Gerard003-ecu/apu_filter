# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para imperial_guards_tesserarios."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
import scipy.linalg as la
from unittest.mock import MagicMock, patch

from app.agents.core.inmune_system.imperial_guards_tesserarios import (
    _MACHINE_EPS, _WILKINSON_DEFLATION_SCALE, _HARD_STASHEFF_CEILING,
    _HARD_QUILLEN_TOLERANCE, _HARD_GERBE_TOLERANCE, _HARD_SULLIVAN_TOLERANCE,
    _STRUCTURE_ATOL, _MIN_SINGULAR_VALUE_FLOOR, _DEGRADATION_FACTOR,
    HeytingVerdict, _SymplecticForm, _QuillenWitness, _HomotopyJet,
    _HomotopySpectralCore, _TesserariosAuditResult, _TesserariosSheaf,
    HomotopicTesserariosAgent
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_types_and_values(self):
        assert _MACHINE_EPS > 0
        assert _WILKINSON_DEFLATION_SCALE == 10.0
        assert _HARD_STASHEFF_CEILING == 1.0e-8
        assert _HARD_QUILLEN_TOLERANCE == 1.0e-8
        assert _HARD_GERBE_TOLERANCE == 1.0e-4
        assert _HARD_SULLIVAN_TOLERANCE == 1.0e-8
        assert _MIN_SINGULAR_VALUE_FLOOR == 1.0e-12

class TestExceptionHierarchy:
    """Verify exception inheritance chain (if any) or standard errors."""
    def test_heyting_verdict_unknown(self):
        with pytest.raises(ValueError):
            HeytingVerdict.from_token("INVALID")

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_heyting_verdict_logic(self):
        v1 = HeytingVerdict.COHERENT
        v2 = HeytingVerdict.DEGRADED
        assert v1.meet(v2) == HeytingVerdict.DEGRADED
        assert v1.join(v2) == HeytingVerdict.COHERENT
        assert v1.negate() == HeytingVerdict.VETOED
        assert v2.implies(v1) == HeytingVerdict.COHERENT

    def test_symplectic_form_valid(self):
        form = _SymplecticForm.from_dimension(2)
        assert form.half_dim == 1
        assert abs(form.frobenius_norm - math.sqrt(2)) < 1e-12
        form.verify()

    def test_symplectic_form_invalid_dim(self):
        with pytest.raises(ValueError):
            _SymplecticForm.from_dimension(3)

    def test_symplectic_form_verify_failure(self):
        form = _SymplecticForm(matrix=np.eye(2), half_dim=1, frobenius_norm=math.sqrt(2))
        with pytest.raises(ValueError, match="no es anti-simétrica"):
            form.verify()

    def test_quillen_witness_frozen(self):
        obj = _QuillenWitness(0.1, 0.1, 0.1, 0.1, 0.1, 1.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.det_residual = 0.2

    def test_homotopy_jet_frozen(self):
        qw = _QuillenWitness(0.1, 0.1, 0.1, 0.1, 0.1, 1.0)
        obj = _HomotopyJet(2, qw, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, None)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.n = 4

    def test_tesserarios_audit_result(self):
        obj = _TesserariosAuditResult(0.1, 0.2, "COHERENT", {})
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.metric_value = 0.5

class TestPhase1_HomotopySpectralCore:
    """Granular tests for Phase 1 methods."""
    def test_frobenius(self):
        val = _HomotopySpectralCore.frobenius(np.array([3.0, 4.0]))
        assert abs(val - 5.0) < 1e-12

    def test_relative_frobenius(self):
        val = _HomotopySpectralCore.relative_frobenius(np.array([2.0]), np.array([4.0]))
        assert abs(val - 0.5) < 1e-12

    def test_wilkinson_deflation_floor(self):
        mat = np.eye(2)
        val = _HomotopySpectralCore.wilkinson_deflation_floor(mat)
        assert val > 0

    def test_as_real_valid(self):
        res = _HomotopySpectralCore._as_real("test", np.array([1.0, 2.0]))
        assert res.dtype == np.float64

    def test_as_real_invalid(self):
        with pytest.raises(ValueError):
            _HomotopySpectralCore._as_real("test", np.array([1.0, np.nan]))
        with pytest.raises(ValueError):
            _HomotopySpectralCore._as_real("test", np.array([1.0 + 1j]))

    def test_symplectic_pullback_residual(self):
        omega = _SymplecticForm.from_dimension(2).matrix
        jac = np.eye(2)
        abs_res, rel_res = _HomotopySpectralCore.symplectic_pullback_residual(jac, omega)
        assert abs(abs_res) < 1e-12
        assert abs(rel_res) < 1e-12

    def test_polar_quillen_factor(self):
        jac = np.eye(2) * 2.0
        u, p, cond = _HomotopySpectralCore.polar_quillen_factor(jac)
        np.testing.assert_allclose(u, np.eye(2))
        np.testing.assert_allclose(p, np.eye(2) * 2.0)
        assert abs(cond - 1.0) < 1e-12

    def test_compute_quillen_witness(self):
        omega = _SymplecticForm.from_dimension(2).matrix
        jac = np.eye(2)
        wit = _HomotopySpectralCore.compute_quillen_witness(jac, omega)
        assert abs(wit.symplectic_residual) < 1e-12

    def test_stasheff_norm(self):
        m3 = np.ones((2, 2, 2))
        val = _HomotopySpectralCore.compute_stasheff_norm(m3)
        assert abs(val - math.sqrt(8)) < 1e-12

    def test_hochschild_associator(self):
        m2 = np.zeros((2, 2, 2))
        asc = _HomotopySpectralCore.hochschild_associator(m2)
        assert _HomotopySpectralCore.frobenius(asc) < 1e-12

    def test_sullivan_commutator_residual(self):
        m2 = np.zeros((2, 2, 2))
        res = _HomotopySpectralCore.sullivan_commutator_residual(m2)
        assert res < 1e-12

    def test_stasheff_pentagon_residual(self):
        m2 = np.zeros((2, 2, 2))
        m3 = np.zeros((2, 2, 2, 2))
        res = _HomotopySpectralCore.stasheff_pentagon_residual(m2, m3)
        assert res < 1e-12

    def test_cech_coboundary_1_norm2(self):
        c = np.zeros((2, 2))
        val = _HomotopySpectralCore.cech_coboundary_1_norm2(c)
        assert val < 1e-12

    def test_cech_coboundary_2_norm(self):
        g = np.zeros((2, 2, 2))
        val = _HomotopySpectralCore.cech_coboundary_2_norm(g)
        assert val < 1e-12

    def test_compute_gerbe_obstruction(self):
        c = np.zeros((2, 2))
        res, rel, mass = _HomotopySpectralCore.compute_gerbe_obstruction(c)
        assert res < 1e-12

    def test_assemble_homotopy_jet(self):
        omega = _SymplecticForm.from_dimension(2)
        jac = np.eye(2)
        m3 = np.zeros((2, 2, 2))
        cech = np.zeros((2, 2))
        jet = _HomotopySpectralCore.assemble_homotopy_jet(2, jac, m3, cech, omega)
        assert jet.n == 2
        assert jet.input_fault is None

class TestPhase2_HomotopicTesserariosAgent:
    """Granular tests for Phase 2 methods."""
    def setup_method(self):
        self.agent = HomotopicTesserariosAgent(dimension_n=2)
        self.rng = np.random.default_rng(42)

    def test_init_invalid_dim(self):
        with pytest.raises(ValueError):
            HomotopicTesserariosAgent(dimension_n=3)

    def test_ingest_tensors(self):
        jac = np.eye(2)
        m3 = np.zeros((2, 2, 2))
        cech = np.zeros((2, 2))
        jet = self.agent.ingest_tensors(jac, m3, cech)
        assert jet.n == 2

    def test_verdict_from_metric(self):
        v, t = self.agent._verdict_from_metric(0.0, 1.0)
        assert v == "COHERENT"
        v2, t2 = self.agent._verdict_from_metric(0.5, 1.0, degradation_factor=0.1)
        assert v2 == "DEGRADED"

    def test_audit_quillen_factorization(self):
        jac = np.eye(2)
        m3 = np.zeros((2, 2, 2))
        cech = np.zeros((2, 2))
        jet = self.agent.ingest_tensors(jac, m3, cech)
        res = self.agent.audit_quillen_factorization(jet)
        assert res.verdict == "COHERENT"

