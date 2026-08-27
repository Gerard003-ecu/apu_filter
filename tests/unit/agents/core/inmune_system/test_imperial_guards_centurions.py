# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para imperial_guards_centurions."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np

from app.agents.core.inmune_system.imperial_guards_centurions import (
    HeytingVerdict,
    _HamiltonianBundle,
    _SpectralCore,
    PortHamiltonianCenturion,
    ThermodynamicCenturion,
    _PowerCurtainAudit,
    _MACHINE_EPS,
    _WILKINSON_LIMIT
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_constants_positive(self):
        assert _MACHINE_EPS > 0.0
        assert _WILKINSON_LIMIT > 0.0

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_hamiltonian_bundle_frozen(self):
        bundle = _HamiltonianBundle(
            n=2,
            M_d=np.eye(2),
            M_d_inv=np.eye(2),
            R_d=np.eye(2),
            J_d=np.array([[0, 1], [-1, 0]]),
            cond_M=1.0,
            spectral_gap_R=1.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            bundle.n = 4

    def test_power_curtain_audit_frozen(self):
        audit = _PowerCurtainAudit(
            dissipation_power=1.0,
            interconnection_leak=0.0,
            port_supply_rate=0.0,
            predicted_hdot=-1.0,
            gradient_norm=1.0,
            hamiltonian=0.5,
            antiwindup_engaged=False,
            extra_damping=0.0,
            verdict="COHERENT"
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            audit.verdict = "VETOED"

class TestHeytingVerdict:
    def test_heyting_logic(self):
        assert HeytingVerdict.COHERENT.meet(HeytingVerdict.VETOED) == HeytingVerdict.VETOED
        assert HeytingVerdict.COHERENT.join(HeytingVerdict.VETOED) == HeytingVerdict.COHERENT
        assert HeytingVerdict.DEGRADED.negate() == HeytingVerdict.VETOED
        assert HeytingVerdict.from_token("COHERENT") == HeytingVerdict.COHERENT

class TestSpectralCore:
    def test_hermitize(self):
        mat = np.array([[1, 2j], [3, 4]])
        herm = _SpectralCore.hermitize(mat)
        assert np.allclose(herm, herm.conj().T)

    def test_skew_symmetrize(self):
        mat = np.array([[1, 2j], [3, 4]])
        skew = _SpectralCore.skew_symmetrize(mat)
        assert np.allclose(skew, -skew.conj().T)

    def test_regularize_spd(self):
        mat = np.array([[1, 0], [0, -1]], dtype=np.float64)
        spd = _SpectralCore.regularize_spd(mat)
        evals = np.linalg.eigvalsh(spd)
        assert np.all(evals >= _WILKINSON_LIMIT)

    def test_assemble_standard_J(self):
        J = _SpectralCore.assemble_standard_J(4)
        assert np.allclose(J.T, -J)
        assert np.allclose(J @ J, -np.eye(4))

    def test_prepare_hamiltonian_bundle(self):
        bundle = _SpectralCore.prepare_hamiltonian_bundle(2, np.eye(2), np.eye(2))
        assert bundle.n == 2
        assert np.allclose(bundle.M_d, np.eye(2))
        assert np.allclose(bundle.R_d, np.eye(2))
        assert bundle.cond_M >= 1.0

class TestPortHamiltonianCenturion:
    def test_initialization(self):
        target = np.array([0.0, 0.0])
        centurion = PortHamiltonianCenturion(
            dimension_n=2,
            inertia_matrix=np.eye(2),
            damping_matrix_rd=np.eye(2),
            target_state=target
        )
        assert centurion._n == 2

    def test_compute_error_and_hamiltonian(self):
        target = np.array([0.0, 0.0])
        centurion = PortHamiltonianCenturion(
            dimension_n=2,
            inertia_matrix=np.eye(2),
            damping_matrix_rd=np.eye(2),
            target_state=target
        )
        x = np.array([1.0, 1.0])
        err = centurion.compute_error(x)
        assert np.allclose(err, x)
        H = centurion.compute_hamiltonian(x)
        assert abs(H - 1.0) < 1e-9

    def test_evaluate_power_curtain(self):
        target = np.array([0.0, 0.0])
        centurion = PortHamiltonianCenturion(
            dimension_n=2,
            inertia_matrix=np.eye(2),
            damping_matrix_rd=np.eye(2),
            target_state=target
        )
        x = np.array([1.0, 1.0])
        u = np.array([0.0, 0.0])
        audit = centurion.evaluate_power_curtain(x, u)
        assert audit.verdict == "COHERENT"
        assert audit.dissipation_power > 0

class TestThermodynamicCenturion:
    def test_initialization(self):
        centurion = ThermodynamicCenturion(2)
        assert centurion._dim == 2

    def test_compute_von_neumann_entropy(self):
        centurion = ThermodynamicCenturion(2)
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy = centurion.compute_von_neumann_entropy(rho)
        assert abs(entropy) < 1e-9
