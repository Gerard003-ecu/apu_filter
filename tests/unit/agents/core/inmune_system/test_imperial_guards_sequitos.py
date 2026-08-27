# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para imperial_guards_sequitos."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.core.inmune_system.imperial_guards_sequitos import (
    _MACHINE_EPS, _INTERLOCK_LATENCY_BUDGET_NS, _INTERLOCK_JITTER_NS,
    _WILKINSON_REL_SCALE, _KLEISLI_COHERENT_TOL, _KLEISLI_DEGRADED_TOL,
    _DEGROOT_COHERENT_DEV, _DEGROOT_DEGRADED_DEV, _TSIRELSON_BOUND,
    _CLASSICAL_CHSH_BOUND, _PR_NOSIGNAL_BOUND, _TSIRELSON_GUARD_BASE,
    _CORRELATOR_BOUND, _HEYTING_ORDER, _HEYTING_GODEL, _REVERSE_HEYTING,
    _KleisliRawResult, _DeGrootRawResult, _CHSHRawResult, _HeytingAuditGerm,
    _AuditCore, _KleisliVeredict, _DeGrootVeredict, _CHSHVeredict,
    _OODAActuationGerm, _HeytingClassifier, _OODAResult, _OODAController
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_types_and_values(self):
        assert _MACHINE_EPS > 0
        assert _INTERLOCK_LATENCY_BUDGET_NS == 400.0
        assert _INTERLOCK_JITTER_NS == 5.0
        assert _WILKINSON_REL_SCALE == 10.0
        assert _KLEISLI_COHERENT_TOL == 1e-10
        assert _KLEISLI_DEGRADED_TOL == 1e-8
        assert _DEGROOT_COHERENT_DEV == 1e-6
        assert _DEGROOT_DEGRADED_DEV == 1e-4
        assert abs(_TSIRELSON_BOUND - 2.0 * math.sqrt(2.0)) < 1e-12
        assert _CLASSICAL_CHSH_BOUND == 2.0
        assert _PR_NOSIGNAL_BOUND == 4.0
        assert _TSIRELSON_GUARD_BASE == 1e-3
        assert _CORRELATOR_BOUND == 1.0

    def test_heyting_dicts(self):
        assert _HEYTING_ORDER["COHERENT"] == 0
        assert _HEYTING_ORDER["DEGRADED"] == 1
        assert _HEYTING_ORDER["VETOED"] == 2
        assert _HEYTING_GODEL["COHERENT"] == 1.0
        assert _HEYTING_GODEL["DEGRADED"] == 0.5
        assert _HEYTING_GODEL["VETOED"] == 0.0
        assert _REVERSE_HEYTING[0] == "COHERENT"

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_kleisli_raw_result_immutability(self):
        obj = _KleisliRawResult(deviation=0.1)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.deviation = 0.2
        assert obj.engine_ok is True

    def test_degroot_raw_result_post_init(self):
        opinions = np.array([1.0, 2.0], dtype=np.float64)
        obj = _DeGrootRawResult(final_opinions=opinions, fiedler_value=0.5, deviation=0.1)
        np.testing.assert_allclose(obj.discrete_opinions, opinions)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.fiedler_value = 0.8

    def test_chsh_raw_result(self):
        obj = _CHSHRawResult(s_value=2.5)
        assert obj.engine_ok is True
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.s_value = 1.0

    def test_heyting_audit_germ(self):
        kl = _KleisliRawResult(deviation=0.1)
        dg = _DeGrootRawResult(final_opinions=np.array([1.0], dtype=np.float64), fiedler_value=1.0, deviation=0.0)
        ch = _CHSHRawResult(s_value=2.0)
        germ = _HeytingAuditGerm(kleisli=kl, degroot=dg, chsh=ch, n_agents=3, safety_margin=1.5, kleisli_scale=1.0, degroot_scale=1.0)
        assert germ.safety_margin == 1.5

    def test_kleisli_veredict(self):
        obj = _KleisliVeredict(deviation=0.1, verdict="COHERENT")
        assert obj.godel_value == 0.0

    def test_degroot_veredict(self):
        obj = _DeGrootVeredict(fiedler_value=1.0, deviation=0.1, verdict="COHERENT")
        assert obj.connected is True

    def test_chsh_veredict(self):
        obj = _CHSHVeredict(s_value=2.5, verdict="COHERENT")
        assert obj.physical is True

    def test_ooda_actuation_germ(self):
        kl = _KleisliVeredict(deviation=0.1, verdict="COHERENT")
        dg = _DeGrootVeredict(fiedler_value=1.0, deviation=0.1, verdict="COHERENT")
        ch = _CHSHVeredict(s_value=2.5, verdict="COHERENT")
        germ = _OODAActuationGerm(kleisli=kl, degroot=dg, chsh=ch, heyting_join="COHERENT", godel_meet=1.0, n_agents=3, safety_margin=1.0)
        assert germ.heyting_join == "COHERENT"

    def test_ooda_result_dict(self):
        res = _OODAResult("COHERENT", 0.0, "COHERENT", 1.0, "COHERENT", 2.5, "COHERENT", False, 100.0, 1.0, True, 0.0, True, 0.0, True, True)
        d = res.as_public_dict()
        assert d["heyting_verdict"] == "COHERENT"

class TestPhase1_AuditCore:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.mock_engine = MagicMock()
        self.core = _AuditCore(engine=self.mock_engine, n_agents=3)
        self.rng = np.random.default_rng(42)

    def test_as_vec_valid(self):
        vec = self.core._as_vec("test", [1, 2, 3])
        assert vec.dtype == np.float64
        assert vec.shape == (3,)

    def test_as_vec_invalid(self):
        with pytest.raises(ValueError, match="no puede ser vacío"):
            self.core._as_vec("test", [])
        with pytest.raises(ValueError, match="contiene no-finitos"):
            self.core._as_vec("test", [1.0, np.inf])

    def test_as_matrix_valid(self):
        mat = self.core._as_matrix("test", [[1, 2], [3, 4]], square=True)
        assert mat.shape == (2, 2)
        flat = self.core._as_matrix("test", [1, 2, 3, 4], square=True)
        assert flat.shape == (2, 2)

    def test_as_matrix_invalid(self):
        with pytest.raises(ValueError, match="no es un cuadrado perfecto"):
            self.core._as_matrix("test", [1, 2, 3])
        with pytest.raises(ValueError, match="debe ser cuadrada"):
            self.core._as_matrix("test", [[1, 2, 3], [4, 5, 6]], square=True)

    def test_prob_of(self):
        val, p = self.core._prob_of(("val", 0.5), "test")
        assert val == "val"
        assert abs(p - 0.5) < 1e-12

    def test_prob_of_invalid(self):
        with pytest.raises(ValueError, match="debe retornar"):
            self.core._prob_of("not_tuple", "test")
        with pytest.raises(ValueError, match="probabilidad no finita"):
            self.core._prob_of(("val", np.nan), "test")

    def test_value_mismatch_same(self):
        res = self.core._value_mismatch(np.array([1.0]), np.array([1.0]))
        assert abs(res) < 1e-12

    def test_value_mismatch_diff(self):
        res = self.core._value_mismatch(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert abs(res - math.sqrt(2.0)) < 1e-12

    def test_compute_kleisli_deviation_success(self):
        f = lambda x: (x, 1.0)
        g = lambda x: (x, 1.0)
        h = lambda x: (x, 1.0)
        self.mock_engine.kleisli_compose.return_value = lambda x: (x, 1.0)
        res = self.core.compute_kleisli_deviation(f, g, h, 0)
        assert res.engine_ok is True
        assert abs(res.deviation) < 1e-12

    def test_compute_kleisli_deviation_failure(self):
        self.mock_engine.kleisli_compose.side_effect = Exception("error")
        res = self.core.compute_kleisli_deviation(lambda x: (x, 1.0), lambda x: (x, 1.0), lambda x: (x, 1.0), 0)
        assert res.engine_ok is False
        assert math.isinf(res.deviation)

    def test_compute_degroot_metrics_success(self):
        op = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        aff = np.eye(3, dtype=np.float64)
        self.mock_engine.compute_degroot_spectral_consensus.return_value = (op, 0.5, "COHERENT")
        res = self.core.compute_degroot_metrics(op, aff)
        assert res.engine_ok is True
        assert abs(res.fiedler_value - 0.5) < 1e-12

    def test_compute_degroot_metrics_failure(self):
        res = self.core.compute_degroot_metrics(np.array([1.0]), np.eye(2))
        assert res.engine_ok is False
        assert math.isinf(res.deviation)

    def test_compute_chsh_s_value_success(self):
        corr = np.eye(2, dtype=np.float64)
        self.mock_engine.verify_chsh_violation.return_value = (2.5, "COHERENT")
        res = self.core.compute_chsh_s_value(corr)
        assert res.engine_ok is True
        assert abs(res.s_value - 2.5) < 1e-12

    def test_compute_chsh_s_value_failure(self):
        self.mock_engine.verify_chsh_violation.side_effect = Exception("error")
        res = self.core.compute_chsh_s_value(np.eye(2))
        assert res.engine_ok is False
        assert math.isinf(res.s_value)

    def test_synthesize_heyting_audit_germ(self):
        f = lambda x: (x, 1.0)
        self.mock_engine.kleisli_compose.return_value = lambda x: (x, 1.0)
        self.mock_engine.compute_degroot_spectral_consensus.return_value = (np.array([1.0]), 1.0, "COHERENT")
        self.mock_engine.verify_chsh_violation.return_value = (2.5, "COHERENT")
        germ = self.core.synthesize_heyting_audit_germ(f, f, f, 0, np.array([1.0]), np.eye(1), np.eye(2), 1.5)
        assert germ.safety_margin == 1.5
        assert germ.kleisli.engine_ok is True

class TestPhase2_HeytingClassifier:
    """Granular tests for Phase 2 methods."""
    def setup_method(self):
        self.classifier = _HeytingClassifier(safety_margin=1.5)

    def test_canonicalize(self):
        assert self.classifier.canonicalize("COHERENT") == "COHERENT"
        assert self.classifier.canonicalize("UNKNOWN") == "VETOED"

    def test_join(self):
        assert self.classifier.join("COHERENT", "DEGRADED") == "DEGRADED"
        assert self.classifier.join("DEGRADED", "VETOED") == "VETOED"

    def test_meet(self):
        assert self.classifier.meet("COHERENT", "DEGRADED") == "COHERENT"
        assert self.classifier.meet("DEGRADED", "VETOED") == "DEGRADED"

    def test_scaled_tol(self):
        tol = self.classifier.scaled_tol(1.0, scale=2.0)
        assert tol >= 1.5

    def test_verdict_from_deviation(self):
        res = self.classifier.verdict_from_deviation(np.inf, 1e-5, 1e-3)
        assert res == "VETOED"
        res2 = self.classifier.verdict_from_deviation(1e-6, 1e-5, 1e-3)
        assert res2 == "COHERENT"

    def test_classify_kleisli(self):
        raw = _KleisliRawResult(deviation=0.0, engine_ok=True)
        res = self.classifier.classify_kleisli(raw, scale=1.0)
        assert res.verdict == "COHERENT"

    def test_classify_degroot(self):
        raw = _DeGrootRawResult(final_opinions=np.array([1.0]), fiedler_value=1.0, deviation=0.0, engine_ok=True, connected=False)
        res = self.classifier.classify_degroot(raw, scale=1.0)
        assert res.verdict == "DEGRADED"

    def test_classify_chsh_vetoed(self):
        raw = _CHSHRawResult(s_value=5.0, engine_ok=True, physical=False)
        res = self.classifier.classify_chsh(raw)
        assert res.verdict == "VETOED"

    def test_induce_ooda_actuation_germ(self):
        kl = _KleisliRawResult(deviation=0.0)
        dg = _DeGrootRawResult(final_opinions=np.array([1.0]), fiedler_value=1.0, deviation=0.0)
        ch = _CHSHRawResult(s_value=2.5)
        germ = _HeytingAuditGerm(kleisli=kl, degroot=dg, chsh=ch, n_agents=3, safety_margin=1.0, kleisli_scale=1.0, degroot_scale=1.0)
        res = self.classifier.induce_ooda_actuation_germ(germ)
        assert res.heyting_join in ["COHERENT", "DEGRADED", "VETOED"]

class TestPhase3_OODAController:
    """Granular tests for Phase 3 methods."""
    def setup_method(self):
        self.controller = _OODAController()
        kl = _KleisliVeredict(deviation=0.0, verdict="COHERENT")
        dg = _DeGrootVeredict(fiedler_value=1.0, deviation=0.0, verdict="COHERENT")
        ch = _CHSHVeredict(s_value=2.5, verdict="COHERENT")
        self.germ = _OODAActuationGerm(kleisli=kl, degroot=dg, chsh=ch, heyting_join="COHERENT", godel_meet=1.0, n_agents=3, safety_margin=1.0)

    def test_observe(self):
        kl, dg, ch = self.controller.observe(self.germ)
        assert kl.verdict == "COHERENT"

    def test_orient(self):
        join = self.controller.orient(self.germ)
        assert join == "COHERENT"

    def test_decide(self):
        assert self.controller.decide("COHERENT") is False
        assert self.controller.decide("VETOED") is True

