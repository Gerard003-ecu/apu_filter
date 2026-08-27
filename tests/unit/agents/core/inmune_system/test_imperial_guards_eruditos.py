# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para imperial_guards_eruditos."""
from __future__ import annotations
import dataclasses
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.core.inmune_system.imperial_guards_eruditos import (
    _FloerAudit,
    _CechAudit,
    _HeytingAuditGerm,
    _AuditCore,
    _FloerVeredict,
    _CechVeredict,
    _OODAActuationGerm,
    _HeytingClassifier,
    _OODAResult,
    _OODAController,
    ImperialGuardsEruditosAgent,
    _MACHINE_EPS,
    _FLOER_THRESHOLD,
    _CECH_THRESHOLD
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_positive(self):
        assert _MACHINE_EPS > 0.0
        assert _FLOER_THRESHOLD > 0.0
        assert _CECH_THRESHOLD > 0.0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_floer_audit_frozen(self):
        audit = _FloerAudit(
            floer_residual=0.0,
            action_potential=1.0,
            liouville_action=0.0,
            dirichlet_energy=0.0,
            symplectic_monodromy_residual=0.0,
            conley_zehnder_index=0.0,
            maslov_degeneracy=0.0,
            is_nondegenerate=True,
            is_symplectic_monodromy=True,
            engine_ok=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            audit.floer_residual = 1.0

    def test_cech_audit_frozen(self):
        audit = _CechAudit(
            cech_obstruction=0.0,
            active_modes=np.array([1.0]),
            cocycle_defect=0.0,
            harmonic_energy=0.0,
            betti_0=1,
            betti_1=0,
            effective_rank=1,
            nuclear_mass=1.0,
            engine_ok=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            audit.cech_obstruction = 1.0

class TestAuditCore:
    def test_as_vec(self):
        vec = _AuditCore._as_vec("test", [1, 2, 3])
        assert np.allclose(vec, np.array([1.0, 2.0, 3.0]))

    def test_as_matrix(self):
        mat = _AuditCore._as_matrix("test", [[1, 2], [3, 4]])
        assert np.allclose(mat, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_floer_audit(self):
        engine_mock = MagicMock()
        engine_mock.verify_floer_homology_trajectory.return_value = (0.0, 1.0)
        core = _AuditCore(engine_mock, 2)
        
        audit = core.floer_audit(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.eye(2))
        assert audit.floer_residual == 0.0
        assert audit.action_potential == 1.0

    def test_cech_audit(self):
        engine_mock = MagicMock()
        engine_mock.compute_attention_cech_cohomology.return_value = (0.0, np.array([1.0]))
        core = _AuditCore(engine_mock, 2)
        
        audit = core.cech_audit(np.eye(2))
        assert audit.cech_obstruction == 0.0

class TestHeytingClassifier:
    def test_verdict_from_metric(self):
        classifier = _HeytingClassifier(1.0)
        verdict = classifier.verdict_from_metric(0.0, 1.0)
        assert verdict == "COHERENT"

    def test_classify_floer(self):
        classifier = _HeytingClassifier(1.0)
        audit = _FloerAudit(
            floer_residual=0.0,
            action_potential=1.0
        )
        verdict = classifier.classify_floer(audit, 1.0)
        assert verdict.verdict == "COHERENT"

class TestOODAController:
    def test_decide(self):
        assert _OODAController.decide("VETOED") is True
        assert _OODAController.decide("COHERENT") is False

    def test_run(self):
        controller = _OODAController()
        germ = _OODAActuationGerm(
            floer=_FloerVeredict(0.0, 1.0, "COHERENT"),
            cech=_CechVeredict(0.0, 1, "COHERENT"),
            heyting_join="COHERENT",
            godel_meet=1.0,
            two_n=2,
            safety_margin=1.0
        )
        res = controller.run(germ)
        assert res.hardware_interlock_fired is False
        assert res.heyting_verdict == "COHERENT"

class TestImperialGuardsEruditosAgent:
    @patch('app.agents.core.inmune_system.imperial_guards_eruditos.ImperialEruditosEngine')
    def test_initialization(self, engine_mock):
        agent = ImperialGuardsEruditosAgent(2)
        assert agent.dimension == 2
        assert agent.safety_margin == 1.0
