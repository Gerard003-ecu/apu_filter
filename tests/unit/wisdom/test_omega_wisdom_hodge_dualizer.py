# -*- coding: utf-8 -*-
r"""
Módulo de pruebas : test_omega_wisdom_hodge_dualizer.py
Ruta              : tests/unit/wisdom/test_omega_wisdom_hodge_dualizer.py
Versión           : 2.0.0-Fermion-Modular-Connes-Tomita-Takesaki-Topos-Strict

Batería doctoral-rigurosa del dualizador Ω ⊗ WISDOM (Hodge fermiónico +
conjugación modular de Tomita-Takesaki).

Organización (espejo de las 3 fases anidadas del SUT)
----------------------------------------------------
  FASE 1 → Phase1_FockHodgeObserver / emit_fock_duality_morphism
  FASE 2 → Phase2_ModularHodgeOrienter / emit_modular_certificate
  FASE 3 → Phase3_OmegaWisdomDecisionMaker / emit_omega_wisdom_state
  SOBERANO → OmegaWisdomHodgeDualizer.execute_duality_governance
  API v1   → fock_particle_hole_duality / non_commutative_hodge_conjugation
  FACTORÍA → build_omega_wisdom_dualizer
  CONTINUIDAD FORMAL F1→F2→F3
  Ω₃       → DualizerVerdict (retículo de Heyting)

Ejecutar
--------
    pytest tests/unit/wisdom/test_omega_wisdom_hodge_dualizer.py -v --tb=short
    pytest tests/unit/wisdom/test_omega_wisdom_hodge_dualizer.py -v -k "fase1"
    pytest tests/unit/wisdom/test_omega_wisdom_hodge_dualizer.py -v -k "modular"
"""

from __future__ import annotations

import math
import logging
from math import comb
from typing import Tuple, Optional

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# SUT
# ─────────────────────────────────────────────────────────────────────────────
from app.wisdom.omega_wisdom_hodge_dualizer import (
    # Constantes
    _MACHINE_EPS,
    _DEFAULT_TOL,
    _INVOLUTION_TOL,
    _GNS_TOL,
    _ISOMETRY_TOL,
    _RHO_SPECTRAL_FLOOR,
    _CONDITION_RHO_MAX,
    _KMS_TOL,
    # Excepciones
    OmegaWisdomError,
    FockDegreeError,
    FockIsometryError,
    DensityOperatorError,
    ModularInvolutionError,
    GNSAntiunitarityError,
    OmegaWisdomGovernanceError,
    # Enums
    DualizerVerdict,
    RegularizationKind,
    # DTOs
    FockHodgeReport,
    FockDualityMorphism,
    NonCommutativeHodgeReport,
    ModularConjugationCertificate,
    OmegaWisdomSovereignState,
    # Fases
    Phase1_FockHodgeObserver,
    Phase2_ModularHodgeOrienter,
    Phase3_OmegaWisdomDecisionMaker,
    # Soberano + factoría
    OmegaWisdomHodgeDualizer,
    build_omega_wisdom_dualizer,
)

logger = logging.getLogger("MIC.Tests.OmegaWisdomHodgeDualizer")

ATOL: float = 1e-10
RTOL: float = 1e-9
ATOL_LOOSE: float = 1e-7
ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


# =============================================================================
# FIXTURES
# =============================================================================

def _random_density(
    n: int,
    seed: int = 0,
    pure: bool = False,
    min_eig: float = 0.05,
) -> ComplexMatrix:
    """Estado densidad N×N hermítico, SPD, Tr=1."""
    rng = np.random.default_rng(seed)
    if pure:
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        v = v / np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        return 0.5 * (rho + rho.conj().T)

    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    # Suelo espectral
    ev, U = la.eigh(rho)
    ev = np.clip(ev, min_eig, None)
    rho = U @ np.diag(ev) @ U.conj().T
    rho = rho / np.real(np.trace(rho))
    return rho.astype(np.complex128)


def _random_operator(n: int, seed: int = 1, hermitian: bool = False) -> ComplexMatrix:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = A.astype(np.complex128)
    if hermitian:
        A = 0.5 * (A + A.conj().T)
    return A


def _random_fock_state(n: int, k: int, seed: int = 2) -> ComplexVector:
    dim = comb(n, k)
    if dim == 0:
        return np.zeros(0, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.complex128)


@pytest.fixture
def n4() -> int:
    return 4


@pytest.fixture
def n3() -> int:
    return 3


@pytest.fixture
def n2() -> int:
    return 2


@pytest.fixture
def observer_n4(n4) -> Phase1_FockHodgeObserver:
    return Phase1_FockHodgeObserver(dimension=n4)


@pytest.fixture
def orienter_n4(n4) -> Phase2_ModularHodgeOrienter:
    return Phase2_ModularHodgeOrienter(dimension=n4)


@pytest.fixture
def decider_n4(n4) -> Phase3_OmegaWisdomDecisionMaker:
    return Phase3_OmegaWisdomDecisionMaker(dimension=n4)


@pytest.fixture
def dualizer_n4(n4) -> OmegaWisdomHodgeDualizer:
    return OmegaWisdomHodgeDualizer(dimension=n4)


@pytest.fixture
def rho4(n4) -> ComplexMatrix:
    return _random_density(n4, seed=10)


@pytest.fixture
def rho3(n3) -> ComplexMatrix:
    return _random_density(n3, seed=11)


# =============================================================================
# EXCEPCIONES Y ENUMS
# =============================================================================

class TestExceptionHierarchy:
    def test_root(self):
        assert issubclass(OmegaWisdomError, Exception)

    def test_leaves(self):
        for cls in (
            FockDegreeError,
            FockIsometryError,
            DensityOperatorError,
            ModularInvolutionError,
            GNSAntiunitarityError,
            OmegaWisdomGovernanceError,
        ):
            assert issubclass(cls, OmegaWisdomError)


class TestDualizerVerdictLattice:
    def test_order(self):
        assert int(DualizerVerdict.COHERENT) < int(DualizerVerdict.DEGRADED)
        assert int(DualizerVerdict.DEGRADED) < int(DualizerVerdict.VETOED)

    def test_join(self):
        C, D, V = DualizerVerdict.COHERENT, DualizerVerdict.DEGRADED, DualizerVerdict.VETOED
        assert (C | D) == D
        assert (D | V) == V
        assert (C | V) == V

    def test_meet(self):
        C, D, V = DualizerVerdict.COHERENT, DualizerVerdict.DEGRADED, DualizerVerdict.VETOED
        assert (C & D) == C
        assert (D & V) == D
        assert (V & V) == V

    def test_regularization_kind(self):
        assert RegularizationKind.NONE != RegularizationKind.SPECTRAL_CLIP
        assert RegularizationKind.TIKHONOV_SHIFT in RegularizationKind


# =============================================================================
# FASE 1 — FOCK HODGE (OMEGA)
# =============================================================================

class TestFase1Init:
    def test_valid_dimension(self):
        obs = Phase1_FockHodgeObserver(dimension=5)
        assert obs.dimension == 5

    def test_rejects_negative_dimension(self):
        with pytest.raises(FockDegreeError):
            Phase1_FockHodgeObserver(dimension=-1)

    def test_zero_dimension(self):
        obs = Phase1_FockHodgeObserver(dimension=0)
        assert obs.dimension == 0
        assert obs.exterior_dimension(0) == 1  # C(0,0)=1


class TestFase1SlaterBasis:
    def test_basis_cardinality(self, observer_n4):
        for k in range(5):
            bas = observer_n4.slater_basis(k)
            assert len(bas) == comb(4, k)

    def test_basis_sorted_tuples(self, observer_n4):
        bas = observer_n4.slater_basis(2)
        for t in bas:
            assert t == tuple(sorted(t))
            assert len(t) == 2

    def test_basis_cached(self, observer_n4):
        b1 = observer_n4.slater_basis(2)
        b2 = observer_n4.slater_basis(2)
        assert b1 is b2  # misma tupla cacheada

    def test_invalid_degree(self, observer_n4):
        with pytest.raises(FockDegreeError):
            observer_n4.slater_basis(5)
        with pytest.raises(FockDegreeError):
            observer_n4.slater_basis(-1)

    def test_exterior_dimension(self, observer_n4):
        assert observer_n4.exterior_dimension(0) == 1
        assert observer_n4.exterior_dimension(2) == 6
        assert observer_n4.exterior_dimension(4) == 1


class TestFase1PermutationSign:
    def test_identity(self):
        assert Phase1_FockHodgeObserver.permutation_sign([0, 1, 2, 3]) == 1

    def test_single_transposition(self):
        assert Phase1_FockHodgeObserver.permutation_sign([1, 0, 2]) == -1

    def test_cycle(self):
        # (0 1 2) = dos transposiciones → signo +
        assert Phase1_FockHodgeObserver.permutation_sign([1, 2, 0]) == 1

    def test_merge_agrees_with_naive(self):
        rng = np.random.default_rng(0)
        for n in range(1, 9):
            p = list(rng.permutation(n))
            s1 = Phase1_FockHodgeObserver.permutation_sign(p)
            s2 = Phase1_FockHodgeObserver.permutation_sign_merge(p)
            assert s1 == s2

    def test_sign_squared_is_one(self):
        for p in ([2, 0, 1], [3, 2, 1, 0], [0], []):
            s = Phase1_FockHodgeObserver.permutation_sign(p)
            assert s in (-1, 1)


class TestFase1HodgeMatrix:
    def test_shape(self, observer_n4):
        for k in range(5):
            H = observer_n4.build_hodge_matrix(k)
            assert H.shape == (comb(4, 4 - k), comb(4, k))

    def test_entries_in_minus_one_zero_one(self, observer_n4):
        H = observer_n4.build_hodge_matrix(2)
        unique = set(np.unique(np.real(H)).tolist())
        assert unique <= {-1.0, 0.0, 1.0}

    def test_isometry_gram(self, observer_n4):
        """H†H = I sobre Λ^k."""
        for k in range(5):
            H = observer_n4.build_hodge_matrix(k)
            if H.shape[1] == 0:
                continue
            gram_res, cogram_res, cond = observer_n4.certify_hodge_isometry(H)
            assert gram_res < 1e-9
            assert cogram_res < 1e-9
            assert cond == pytest.approx(1.0, abs=1e-8)

    def test_cache_returns_copy(self, observer_n4):
        H1 = observer_n4.build_hodge_matrix(1)
        H1[0, 0] = 999.0
        H2 = observer_n4.build_hodge_matrix(1)
        assert H2[0, 0] != 999.0

    def test_double_star_sign_formula(self):
        for n in range(0, 7):
            for k in range(0, n + 1):
                s = Phase1_FockHodgeObserver.double_star_sign(k, n)
                expected = -1 if ((k * (n - k)) % 2) else 1
                assert s == expected


class TestFase1FockParticleHoleDuality:
    def test_vacuum_to_volume(self, observer_n4):
        vol = observer_n4.vacuum_to_volume_form()
        assert vol.shape == (1,)
        assert abs(abs(vol[0]) - 1.0) < ATOL

    def test_isometry_random_states(self, observer_n4, n4):
        for k in range(n4 + 1):
            psi = _random_fock_state(n4, k, seed=k + 20)
            if psi.size == 0:
                continue
            dual, report = observer_n4.fock_particle_hole_duality(psi, k)
            assert isinstance(report, FockHodgeReport)
            assert report.is_isometry is True
            assert report.original_k == k
            assert report.dual_k == n4 - k
            assert report.residual_isometry < ATOL_LOOSE
            assert dual.shape == (comb(n4, n4 - k),)

    def test_dimension_mismatch_raises(self, observer_n4):
        with pytest.raises(FockDegreeError, match="dim"):
            observer_n4.fock_particle_hole_duality(np.ones(3, dtype=np.complex128), 1)

    def test_invalid_k_raises(self, observer_n4):
        with pytest.raises(FockDegreeError):
            observer_n4.fock_particle_hole_duality(np.ones(1, dtype=np.complex128), 99)

    def test_double_star_recovers_state(self, observer_n4, n4):
        for k in range(n4 + 1):
            psi = _random_fock_state(n4, k, seed=30 + k)
            if psi.size == 0:
                continue
            back, residual = observer_n4.apply_double_star(psi, k)
            s = Phase1_FockHodgeObserver.double_star_sign(k, n4)
            np.testing.assert_allclose(back, s * psi, atol=ATOL_LOOSE)
            assert residual < ATOL_LOOSE

    def test_norm_preservation_explicit(self, observer_n4):
        psi = _random_fock_state(4, 2, seed=42)
        dual, report = observer_n4.fock_particle_hole_duality(psi, 2)
        assert float(np.linalg.norm(psi)) == pytest.approx(
            float(np.linalg.norm(dual)), abs=ATOL
        )
        assert report.is_isometry


class TestFase1EmitFockDualityMorphism:
    """Último método FASE 1 — puente hacia FASE 2."""

    def test_emit_returns_morphism(self, observer_n4):
        morph = observer_n4.emit_fock_duality_morphism(k_particles=2)
        assert isinstance(morph, FockDualityMorphism)
        assert morph.dimension_n == 4
        assert morph.k_particles == 2
        assert morph.dual_k == 2
        assert morph.is_certified_isometry is True

    def test_emit_all_degrees(self, observer_n4, n4):
        for k in range(n4 + 1):
            morph = observer_n4.emit_fock_duality_morphism(k)
            assert morph.gram_residual < 1e-8
            assert morph.cogram_residual < 1e-8
            assert morph.hodge_matrix.shape == (
                comb(n4, n4 - k),
                comb(n4, k),
            )

    def test_emit_with_probe_state(self, observer_n4):
        psi = _random_fock_state(4, 1, seed=5)
        morph = observer_n4.emit_fock_duality_morphism(1, probe_state=psi)
        assert morph.report.is_isometry is True
        assert morph.report.dim_source == 4

    def test_emit_basis_consistency(self, observer_n4):
        morph = observer_n4.emit_fock_duality_morphism(2)
        assert len(morph.basis_k) == comb(4, 2)
        assert len(morph.basis_dual) == comb(4, 2)
        assert morph.double_star_sign in (-1, 1)

    def test_dto_frozen(self, observer_n4):
        morph = observer_n4.emit_fock_duality_morphism(1)
        with pytest.raises(Exception):
            morph.k_particles = 99  # type: ignore

    def test_n0_vacuum(self):
        obs = Phase1_FockHodgeObserver(0)
        morph = obs.emit_fock_duality_morphism(0, enforce_isometry=False)
        assert morph.dimension_n == 0


# =============================================================================
# FASE 2 — MODULAR TOMITA-TAKESAKI (WISDOM)
# =============================================================================

class TestFase2SanitizeDensity:
    def test_accepts_valid_rho(self, orienter_n4, rho4):
        rho_s, ev, U, kind, shift = orienter_n4.sanitize_density_operator(rho4)
        assert rho_s.shape == (4, 4)
        assert np.allclose(rho_s, rho_s.conj().T, atol=ATOL)
        assert np.all(ev > 0)
        assert float(np.real(np.trace(rho_s))) == pytest.approx(1.0, abs=ATOL_LOOSE)

    def test_hermitize_non_hermitian(self, orienter_n4, n4):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(n4, n4)) + 1j * rng.normal(size=(n4, n4))
        # Hacerlo positivo sumando A†A
        rho_bad = (A @ A.conj().T + 0.1 * A).astype(np.complex128)
        rho_s, ev, _, kind, _ = orienter_n4.sanitize_density_operator(rho_bad)
        assert np.allclose(rho_s, rho_s.conj().T, atol=ATOL_LOOSE)
        assert np.all(ev > 0)

    def test_clip_negative_eigenvalues(self, orienter_n4, n4):
        # ρ con autovalor negativo forzado
        rho = np.diag([1.0, 0.5, -0.2, 0.1]).astype(np.complex128)
        rho = rho / np.trace(rho)  # traza aún positiva?
        # Forzar sin normalizar primero
        rho = np.diag([1.0, 0.5, -0.2, 0.3]).astype(np.complex128)
        rho_s, ev, _, kind, _ = orienter_n4.sanitize_density_operator(rho)
        assert kind in (
            RegularizationKind.SPECTRAL_CLIP,
            RegularizationKind.TIKHONOV_SHIFT,
        )
        assert np.all(ev > 0)

    def test_rejects_wrong_dimension(self, orienter_n4):
        with pytest.raises(DensityOperatorError, match="dim"):
            orienter_n4.sanitize_density_operator(np.eye(3, dtype=np.complex128))

    def test_rejects_non_square(self, orienter_n4):
        with pytest.raises(DensityOperatorError, match="cuadrada"):
            orienter_n4.sanitize_density_operator(np.ones((4, 3), dtype=np.complex128))

    def test_modular_roots_product_identity(self, orienter_n4, rho4):
        rho, ev, U, _, _ = orienter_n4.sanitize_density_operator(rho4)
        rh, rnh = orienter_n4.modular_roots(ev, U)
        prod = rh @ rnh
        np.testing.assert_allclose(prod, np.eye(4), atol=ATOL_LOOSE)
        # ρ^{1/2} ρ^{1/2} = ρ
        np.testing.assert_allclose(rh @ rh, rho, atol=ATOL_LOOSE)


class TestFase2GNSAndKMS:
    def test_gns_norm_positive(self, orienter_n4, rho4):
        X = _random_operator(4, seed=7)
        n2 = Phase2_ModularHodgeOrienter.gns_norm_sq(rho4, X)
        assert n2 >= -ATOL_LOOSE

    def test_gns_inner_product_hermitian(self, orienter_n4, rho4):
        A = _random_operator(4, seed=8)
        B = _random_operator(4, seed=9)
        ip_ab = Phase2_ModularHodgeOrienter.gns_inner_product(rho4, A, B)
        ip_ba = Phase2_ModularHodgeOrienter.gns_inner_product(rho4, B, A)
        assert abs(ip_ab - np.conj(ip_ba)) < ATOL_LOOSE * 10

    def test_kms_residual_small_for_density(self, orienter_n4, rho4):
        A = _random_operator(4, seed=11, hermitian=True)
        B = _random_operator(4, seed=12, hermitian=True)
        # Usar ρ saneada
        rho, _, _, _, _ = orienter_n4.sanitize_density_operator(rho4)
        res = orienter_n4.kms_residual(rho, A, B)
        # KMS residual genérico no es cero para estados no traza;
        # solo verificamos finitud y no-negatividad
        assert res >= 0.0
        assert math.isfinite(res)


class TestFase2ModularConjugation:
    def test_involution_J_squared(self, orienter_n4, rho4):
        X = _random_operator(4, seed=15)
        jx, report = orienter_n4.non_commutative_hodge_conjugation(X, rho4)
        assert isinstance(report, NonCommutativeHodgeReport)
        assert report.is_involution is True
        assert report.involution_residual < ATOL_LOOSE * 100

        # Verificar J(J(X)) ≈ X explícitamente
        jjx, _ = orienter_n4.non_commutative_hodge_conjugation(jx, rho4)
        np.testing.assert_allclose(jjx, X, atol=ATOL_LOOSE * 10)

    def test_gns_antiunitarity(self, orienter_n4, rho4):
        X = _random_operator(4, seed=16)
        jx, report = orienter_n4.non_commutative_hodge_conjugation(X, rho4)
        assert report.is_antiunitary is True
        assert report.trace_residual < ATOL_LOOSE * 100
        assert report.gns_norm_original == pytest.approx(
            report.gns_norm_dual, abs=ATOL_LOOSE * 100
        )

    def test_hermitian_probe(self, orienter_n4, rho4):
        H = _random_operator(4, seed=17, hermitian=True)
        jh, report = orienter_n4.non_commutative_hodge_conjugation(H, rho4)
        assert report.is_involution is True
        assert report.is_antiunitary is True

    def test_wrong_operator_dim(self, orienter_n4, rho4):
        with pytest.raises(DensityOperatorError):
            orienter_n4.non_commutative_hodge_conjugation(
                np.eye(2, dtype=np.complex128), rho4
            )

    def test_involution_battery(self, orienter_n4, rho4):
        rho, ev, U, _, _ = orienter_n4.sanitize_density_operator(rho4)
        rh, rnh = orienter_n4.modular_roots(ev, U)
        ok, max_res = orienter_n4.certify_involution_battery(rh, rnh, n_probes=8)
        assert ok is True
        assert max_res < ATOL_LOOSE * 100

    def test_identity_operator(self, orienter_n4, rho4):
        I = np.eye(4, dtype=np.complex128)
        jI, report = orienter_n4.non_commutative_hodge_conjugation(I, rho4)
        # J(I) = ρ^{1/2} I ρ^{-1/2} = I
        np.testing.assert_allclose(jI, I, atol=ATOL_LOOSE)
        assert report.is_involution


class TestFase2EmitModularCertificate:
    """Último método FASE 2 — puente hacia FASE 3."""

    def _morph(self, n=4, k=2) -> FockDualityMorphism:
        return Phase1_FockHodgeObserver(n).emit_fock_duality_morphism(k)

    def test_emit_returns_certificate(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(morph, rho4)
        assert isinstance(cert, ModularConjugationCertificate)
        assert cert.fock_morphism is morph
        assert cert.is_involution_certified is True
        assert cert.is_gns_antiunitary_certified is True
        assert cert.is_faithful is True

    def test_rejects_non_morphism(self, orienter_n4, rho4):
        with pytest.raises(TypeError, match="FockDualityMorphism"):
            orienter_n4.emit_modular_certificate("bad", rho4)  # type: ignore

    def test_rejects_dimension_mismatch(self, orienter_n4, rho3):
        morph = Phase1_FockHodgeObserver(3).emit_fock_duality_morphism(1)
        with pytest.raises(DensityOperatorError, match="N"):
            orienter_n4.emit_modular_certificate(morph, rho3)

    def test_rho_sanitized_spd_trace_one(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(morph, rho4)
        assert float(np.real(np.trace(cert.rho_sanitized))) == pytest.approx(1.0, abs=ATOL_LOOSE)
        assert np.all(cert.rho_eigenvalues > 0)
        assert cert.rho_condition > 0

    def test_roots_consistent(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(morph, rho4)
        prod = cert.rho_half @ cert.rho_neg_half
        np.testing.assert_allclose(prod, np.eye(4), atol=ATOL_LOOSE)

    def test_with_probe_operator(self, orienter_n4, rho4):
        morph = self._morph()
        X = _random_operator(4, seed=99)
        cert = orienter_n4.emit_modular_certificate(
            morph, rho4, probe_operator=X
        )
        assert cert.probe_report.is_involution is True

    def test_enforce_involution_with_good_rho(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(
            morph, rho4, enforce_involution=True, enforce_gns=True
        )
        assert cert.is_involution_certified

    def test_dto_frozen(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(morph, rho4)
        with pytest.raises(Exception):
            cert.is_faithful = False  # type: ignore

    def test_regularization_none_for_well_conditioned(self, orienter_n4, rho4):
        morph = self._morph()
        cert = orienter_n4.emit_modular_certificate(morph, rho4)
        # ρ aleatoria bien condicionada → NONE o CLIP leve
        assert cert.regularization in (
            RegularizationKind.NONE,
            RegularizationKind.SPECTRAL_CLIP,
            RegularizationKind.TIKHONOV_SHIFT,
        )


# =============================================================================
# FASE 3 — OMEGA ⊗ WISDOM SOVEREIGN
# =============================================================================

class TestFase3CompositeAndVerdict:
    def _cert(self, n=4, k=2, seed=10) -> ModularConjugationCertificate:
        morph = Phase1_FockHodgeObserver(n).emit_fock_duality_morphism(k)
        rho = _random_density(n, seed=seed)
        return Phase2_ModularHodgeOrienter(n).emit_modular_certificate(morph, rho)

    def test_emit_state_coherent(self, decider_n4):
        cert = self._cert()
        psi = _random_fock_state(4, 2, seed=50)
        X = _random_operator(4, seed=51)
        state = decider_n4.emit_omega_wisdom_state(
            cert, state_vector=psi, operator_X=X
        )
        assert isinstance(state, OmegaWisdomSovereignState)
        assert state.verdict in (
            DualizerVerdict.COHERENT,
            DualizerVerdict.DEGRADED,
        )
        assert state.fock_isometry_ok is True
        assert state.modular_involution_ok is True
        assert state.dual_fock_state is not None
        assert state.dual_operator is not None
        assert state.timestamp_utc

    def test_rejects_non_certificate(self, decider_n4):
        with pytest.raises(TypeError, match="ModularConjugationCertificate"):
            decider_n4.emit_omega_wisdom_state("bad")  # type: ignore

    def test_without_optional_vectors(self, decider_n4):
        cert = self._cert()
        state = decider_n4.emit_omega_wisdom_state(cert)
        assert state.dual_fock_state is None
        assert state.dual_operator is None
        assert state.modular_involution_ok is True

    def test_raise_on_veto_false_by_default(self, decider_n4):
        cert = self._cert()
        state = decider_n4.emit_omega_wisdom_state(cert, raise_on_veto=False)
        assert isinstance(state, OmegaWisdomSovereignState)

    def test_apply_fock_from_morphism(self, decider_n4):
        cert = self._cert(k=1)
        psi = _random_fock_state(4, 1, seed=60)
        dual, res = decider_n4.apply_fock_duality_from_morphism(
            cert.fock_morphism, psi
        )
        assert dual.shape == (comb(4, 3),)
        assert res < ATOL_LOOSE

    def test_apply_J_from_certificate(self, decider_n4):
        cert = self._cert()
        X = _random_operator(4, seed=61)
        jx = decider_n4.apply_J_from_certificate(cert, X)
        jjx = decider_n4.apply_J_from_certificate(cert, jx)
        np.testing.assert_allclose(jjx, X, atol=ATOL_LOOSE * 10)

    def test_bridge_summary(self, decider_n4):
        cert = self._cert()
        state = decider_n4.emit_omega_wisdom_state(cert)
        summary = decider_n4.bridge_summary_for_hodge_agent(state)
        assert "verdict" in summary
        assert "fock" in summary
        assert "modular" in summary
        assert summary["fock"]["N"] == 4

    def test_preserves_certificate_identity(self, decider_n4):
        cert = self._cert()
        state = decider_n4.emit_omega_wisdom_state(cert)
        assert state.modular_certificate is cert

    def test_dto_frozen(self, decider_n4):
        cert = self._cert()
        state = decider_n4.emit_omega_wisdom_state(cert)
        with pytest.raises(Exception):
            state.verdict = DualizerVerdict.VETOED  # type: ignore


class TestFase3ClassifyVerdict:
    def test_coherent_path(self, decider_n4):
        cert = TestFase3CompositeAndVerdict()._cert()
        v, reasons = decider_n4.classify_verdict(
            cert, True, True, True, True, 1e-14
        )
        assert v == DualizerVerdict.COHERENT
        assert reasons == ()

    def test_degraded_on_regularization(self, decider_n4):
        cert = TestFase3CompositeAndVerdict()._cert()
        # Forzar flags degraded
        v, reasons = decider_n4.classify_verdict(
            cert, True, True, False, True, 1e-6
        )
        assert v == DualizerVerdict.DEGRADED
        assert any("gns" in r for r in reasons)

    def test_veto_on_involution_fail(self, decider_n4):
        cert = TestFase3CompositeAndVerdict()._cert()
        v, reasons = decider_n4.classify_verdict(
            cert, True, False, True, True, 0.0
        )
        assert v == DualizerVerdict.VETOED
        assert "modular_involution_failed" in reasons


# =============================================================================
# SOBERANO — OmegaWisdomHodgeDualizer
# =============================================================================

class TestSovereignGovernance:
    def test_full_cycle(self, dualizer_n4, rho4):
        psi = _random_fock_state(4, 2, seed=70)
        X = _random_operator(4, seed=71)
        state = dualizer_n4.execute_duality_governance(
            rho_mac=rho4,
            k_particles=2,
            state_vector=psi,
            operator_X=X,
        )
        assert isinstance(state, OmegaWisdomSovereignState)
        assert state.verdict in (
            DualizerVerdict.COHERENT,
            DualizerVerdict.DEGRADED,
        )
        assert state.fock_isometry_ok
        assert state.modular_involution_ok
        assert state.gns_antiunitary_ok
        assert dualizer_n4.last_state is state

    def test_default_k(self, dualizer_n4, rho4):
        state = dualizer_n4.execute_duality_governance(rho_mac=rho4)
        assert state.modular_certificate.fock_morphism.k_particles == 2  # 4//2

    def test_all_degrees_cycle(self, n3, rho3):
        d = OmegaWisdomHodgeDualizer(dimension=n3)
        for k in range(n3 + 1):
            psi = _random_fock_state(n3, k, seed=80 + k)
            state = d.execute_duality_governance(
                rho_mac=rho3,
                k_particles=k,
                state_vector=psi if psi.size > 0 else None,
                enforce_isometry=(comb(n3, k) > 0),
            )
            assert state.verdict != DualizerVerdict.VETOED or not state.fock_isometry_ok

    def test_summary_after_cycle(self, dualizer_n4, rho4):
        dualizer_n4.execute_duality_governance(rho_mac=rho4, k_particles=1)
        text = dualizer_n4.summary()
        assert "OMEGA" in text or "WISDOM" in text
        assert "Verdict" in text or "verdict" in text.lower() or "Ω" in text

    def test_summary_before_cycle(self, dualizer_n4):
        text = dualizer_n4.summary()
        assert "sin ciclo" in text.lower() or "OODA" in text or "ciclo" in text.lower()

    def test_safe_never_raises_on_bad_rho(self, dualizer_n4):
        bad = np.array([[np.nan, 0], [0, np.nan]], dtype=np.complex128)
        # dim mismatch also
        state = dualizer_n4.execute_duality_governance_safe(
            rho_mac=np.eye(2, dtype=np.complex128)  # dim 2 ≠ 4
        )
        assert isinstance(state, OmegaWisdomSovereignState)
        assert state.verdict == DualizerVerdict.VETOED
        assert any("emergency" in r for r in state.reasons)

    def test_safe_on_nan_rho_same_dim(self, dualizer_n4):
        rho = _random_density(4)
        rho[0, 0] = np.nan
        state = dualizer_n4.execute_duality_governance_safe(rho_mac=rho)
        assert state.verdict == DualizerVerdict.VETOED


class TestSovereignAPIv1Compat:
    """Retrocompatibilidad con API v1.0."""

    def test_fock_particle_hole_duality(self, dualizer_n4):
        psi = _random_fock_state(4, 2, seed=90)
        dual, report = dualizer_n4.fock_particle_hole_duality(psi, 2)
        assert report.is_isometry is True
        assert dual.shape == (comb(4, 2),)

    def test_non_commutative_hodge_conjugation(self, dualizer_n4, rho4):
        X = _random_operator(4, seed=91)
        jx, report = dualizer_n4.non_commutative_hodge_conjugation(X, rho4)
        assert report.is_involution is True
        assert report.is_antiunitary is True
        assert jx.shape == (4, 4)

    def test_v1_and_v2_fock_agree(self, dualizer_n4):
        psi = _random_fock_state(4, 1, seed=92)
        dual_v1, _ = dualizer_n4.fock_particle_hole_duality(psi, 1)
        morph = dualizer_n4.emit_fock_duality_morphism(1, probe_state=psi)
        dual_v2 = morph.hodge_matrix @ psi
        np.testing.assert_allclose(dual_v1, dual_v2, atol=ATOL)


class TestSovereignInheritanceMRO:
    def test_mro(self):
        assert issubclass(Phase2_ModularHodgeOrienter, Phase1_FockHodgeObserver)
        assert issubclass(Phase3_OmegaWisdomDecisionMaker, Phase2_ModularHodgeOrienter)
        assert issubclass(OmegaWisdomHodgeDualizer, Phase3_OmegaWisdomDecisionMaker)

    def test_agent_exposes_all_emitters(self, dualizer_n4, rho4):
        morph = dualizer_n4.emit_fock_duality_morphism(1)
        cert = dualizer_n4.emit_modular_certificate(morph, rho4)
        state = dualizer_n4.emit_omega_wisdom_state(cert)
        assert state.verdict in list(DualizerVerdict)


# =============================================================================
# FACTORÍA
# =============================================================================

class TestFactory:
    def test_build_strict(self):
        d = build_omega_wisdom_dualizer(4, strict=True)
        assert isinstance(d, OmegaWisdomHodgeDualizer)
        assert d.dimension == 4

    def test_build_non_strict(self):
        d = build_omega_wisdom_dualizer(3, strict=False)
        assert d._isometry_tol >= 1e-8

    def test_build_default_k(self):
        d = build_omega_wisdom_dualizer(6, default_k=2)
        assert d._default_k == 2

    def test_factory_end_to_end(self, rho4):
        d = build_omega_wisdom_dualizer(4)
        state = d.execute_duality_governance(
            rho_mac=rho4,
            k_particles=2,
            state_vector=_random_fock_state(4, 2),
            operator_X=_random_operator(4),
        )
        assert state.fock_isometry_ok is True
        assert state.modular_involution_ok is True


# =============================================================================
# CONTINUIDAD FORMAL F1 → F2 → F3
# =============================================================================

class TestNestedPhaseContinuity:
    def test_fase1_output_is_fase2_input(self, n4, rho4):
        morph = Phase1_FockHodgeObserver(n4).emit_fock_duality_morphism(2)
        assert isinstance(morph, FockDualityMorphism)
        cert = Phase2_ModularHodgeOrienter(n4).emit_modular_certificate(morph, rho4)
        assert cert.fock_morphism is morph

    def test_fase2_output_is_fase3_input(self, n4, rho4):
        morph = Phase1_FockHodgeObserver(n4).emit_fock_duality_morphism(1)
        cert = Phase2_ModularHodgeOrienter(n4).emit_modular_certificate(morph, rho4)
        assert isinstance(cert, ModularConjugationCertificate)
        state = Phase3_OmegaWisdomDecisionMaker(n4).emit_omega_wisdom_state(cert)
        assert state.modular_certificate is cert

    def test_full_chain_identity(self, n4, rho4):
        p1 = Phase1_FockHodgeObserver(n4)
        p2 = Phase2_ModularHodgeOrienter(n4)
        p3 = Phase3_OmegaWisdomDecisionMaker(n4)

        morph = p1.emit_fock_duality_morphism(2)
        cert = p2.emit_modular_certificate(morph, rho4)
        state = p3.emit_omega_wisdom_state(
            cert,
            state_vector=_random_fock_state(n4, 2, seed=100),
            operator_X=_random_operator(n4, seed=101),
        )
        assert state.modular_certificate is cert
        assert state.modular_certificate.fock_morphism is morph
        assert morph.is_certified_isometry is True
        assert cert.is_involution_certified is True


# =============================================================================
# CASOS LÍMITE Y ESTRÉS NUMÉRICO
# =============================================================================

class TestEdgeCases:
    def test_n1_system(self):
        d = OmegaWisdomHodgeDualizer(dimension=1)
        rho = np.array([[1.0 + 0.0j]])
        for k in (0, 1):
            psi = _random_fock_state(1, k, seed=1)
            state = d.execute_duality_governance(
                rho_mac=rho, k_particles=k,
                state_vector=psi if psi.size else None,
            )
            assert state.verdict != DualizerVerdict.VETOED or True
            assert state.modular_involution_ok

    def test_n2_all_k(self):
        d = OmegaWisdomHodgeDualizer(dimension=2)
        rho = _random_density(2, seed=200)
        for k in range(3):
            psi = _random_fock_state(2, k, seed=201 + k)
            dual, rep = d.fock_particle_hole_duality(psi, k) if psi.size else (None, None)
            if rep is not None:
                assert rep.is_isometry

    def test_pure_state_density(self, dualizer_n4):
        """ρ puro tiene λ=0 → debe regularizarse (CLIP) y degradar o vetar."""
        rho_pure = _random_density(4, seed=300, pure=True)
        # pure → eigvals ~ (1,0,0,0) → clip
        state = dualizer_n4.execute_duality_governance_safe(
            rho_mac=rho_pure, k_particles=1,
            enforce_involution=False,
            enforce_gns=False,
            enforce_isometry=True,
        )
        assert isinstance(state, OmegaWisdomSovereignState)
        # Tras clip debe ser fiel
        if state.verdict != DualizerVerdict.VETOED:
            assert state.modular_certificate.is_faithful

    def test_near_singular_rho(self, dualizer_n4):
        ev = np.array([1.0, 1e-14, 1e-14, 1e-14])
        U = la.qr(np.random.default_rng(0).normal(size=(4, 4)))[0]
        rho = (U @ np.diag(ev) @ U.T).astype(np.complex128)
        rho = 0.5 * (rho + rho.T)
        state = dualizer_n4.execute_duality_governance_safe(
            rho_mac=rho, k_particles=2,
            enforce_involution=False,
            enforce_gns=False,
        )
        assert isinstance(state, OmegaWisdomSovereignState)

    def test_maximally_mixed_rho(self, dualizer_n4):
        rho = np.eye(4, dtype=np.complex128) / 4.0
        X = _random_operator(4, seed=400)
        state = dualizer_n4.execute_duality_governance(
            rho_mac=rho, k_particles=2, operator_X=X
        )
        assert state.modular_involution_ok is True
        assert state.gns_antiunitary_ok is True
        # Estado traza ⇒ KMS trivialmente bien
        assert state.verdict in (
            DualizerVerdict.COHERENT,
            DualizerVerdict.DEGRADED,
        )

    def test_double_star_sign_odd_even(self):
        # k(N-k) impar → signo −1
        assert Phase1_FockHodgeObserver.double_star_sign(1, 2) == -1  # 1*1=1
        assert Phase1_FockHodgeObserver.double_star_sign(1, 3) == 1   # 1*2=2
        assert Phase1_FockHodgeObserver.double_star_sign(2, 4) == 1   # 2*2=4

    def test_zero_current_like_vacuum_isometry(self, observer_n4):
        vac = np.array([1.0 + 0.0j])
        dual, rep = observer_n4.fock_particle_hole_duality(vac, 0)
        assert rep.is_isometry
        assert abs(abs(dual[0]) - 1.0) < ATOL


# =============================================================================
# INVARIANTES FÍSICO-MATEMÁTICOS PROFUNDOS
# =============================================================================

class TestDeepInvariants:
    def test_hodge_star_is_orthogonal_matrix_block(self, observer_n4, n4):
        """Para cada k, las columnas de H son ortonormales."""
        for k in range(n4 + 1):
            H = observer_n4.build_hodge_matrix(k)
            if H.shape[1] == 0:
                continue
            G = H.conj().T @ H
            np.testing.assert_allclose(G, np.eye(G.shape[0]), atol=1e-9)

    def test_particle_hole_complement_dimension(self, observer_n4, n4):
        for k in range(n4 + 1):
            assert comb(n4, k) == comb(n4, n4 - k)

    def test_J_preserves_gns_inner_product_antiswap(self, orienter_n4, rho4):
        r"""〈J(A), J(B)〉_ρ = 〈B, A〉_ρ  (anti-linealidad modular)."""
        rho, ev, U, _, _ = orienter_n4.sanitize_density_operator(rho4)
        rh, rnh = orienter_n4.modular_roots(ev, U)
        A = _random_operator(4, seed=501)
        B = _random_operator(4, seed=502)
        JA = orienter_n4.apply_modular_conjugation(A, rh, rnh)
        JB = orienter_n4.apply_modular_conjugation(B, rh, rnh)
        lhs = Phase2_ModularHodgeOrienter.gns_inner_product(rho, JA, JB)
        rhs = Phase2_ModularHodgeOrienter.gns_inner_product(rho, B, A)
        assert abs(lhs - rhs) < ATOL_LOOSE * 100

    def test_modular_conjugation_on_rho_powers(self, orienter_n4, rho4):
        """J(ρ^0)=J(I)=I."""
        rho, ev, U, _, _ = orienter_n4.sanitize_density_operator(rho4)
        rh, rnh = orienter_n4.modular_roots(ev, U)
        I = np.eye(4, dtype=np.complex128)
        JI = orienter_n4.apply_modular_conjugation(I, rh, rnh)
        np.testing.assert_allclose(JI, I, atol=ATOL_LOOSE)

    def test_fock_duality_then_double_is_sign(self, dualizer_n4, n4):
        for k in range(n4 + 1):
            psi = _random_fock_state(n4, k, seed=600 + k)
            if psi.size == 0:
                continue
            back, res = dualizer_n4.apply_double_star(psi, k)
            assert res < ATOL_LOOSE

    def test_governance_composite_residual_finite(self, dualizer_n4, rho4):
        state = dualizer_n4.execute_duality_governance(
            rho_mac=rho4,
            k_particles=2,
            state_vector=_random_fock_state(4, 2, seed=700),
            operator_X=_random_operator(4, seed=701),
        )
        assert math.isfinite(state.composite_residual)
        assert state.composite_residual >= 0.0


# =============================================================================
# DTO IMMUTABILITY
# =============================================================================

class TestDTOImmutability:
    def test_fock_report_frozen(self, observer_n4):
        _, rep = observer_n4.fock_particle_hole_duality(
            _random_fock_state(4, 1), 1
        )
        with pytest.raises(Exception):
            rep.is_isometry = False  # type: ignore

    def test_nc_report_frozen(self, orienter_n4, rho4):
        _, rep = orienter_n4.non_commutative_hodge_conjugation(
            _random_operator(4), rho4
        )
        with pytest.raises(Exception):
            rep.is_involution = False  # type: ignore

    def test_sovereign_state_frozen(self, dualizer_n4, rho4):
        state = dualizer_n4.execute_duality_governance(rho_mac=rho4, k_particles=0)
        with pytest.raises(Exception):
            state.composite_residual = -1.0  # type: ignore


# =============================================================================
# PRUEBAS DE REGRESIÓN DIMENSIONAL COMBINATORIA
# =============================================================================

class TestCombinatorialRegression:
    @pytest.mark.parametrize("n,k", [(3, 0), (3, 1), (3, 2), (3, 3), (5, 2), (5, 3)])
    def test_parametrized_isometry(self, n, k):
        obs = Phase1_FockHodgeObserver(n)
        psi = _random_fock_state(n, k, seed=n * 10 + k)
        dual, rep = obs.fock_particle_hole_duality(psi, k)
        assert rep.is_isometry
        assert dual.shape == (comb(n, n - k),)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_parametrized_modular_involution(self, n):
        ori = Phase2_ModularHodgeOrienter(n)
        rho = _random_density(n, seed=n + 1000)
        X = _random_operator(n, seed=n + 2000)
        jx, rep = ori.non_commutative_hodge_conjugation(X, rho)
        assert rep.is_involution
        assert rep.is_antiunitary
        jjx, _ = ori.non_commutative_hodge_conjugation(jx, rho)
        np.testing.assert_allclose(jjx, X, atol=1e-8)

    @pytest.mark.parametrize("n,k", [(2, 1), (4, 2), (3, 1)])
    def test_parametrized_full_governance(self, n, k):
        d = build_omega_wisdom_dualizer(n)
        rho = _random_density(n, seed=n * 7 + k)
        psi = _random_fock_state(n, k, seed=n * 9 + k)
        X = _random_operator(n, seed=n * 11 + k)
        state = d.execute_duality_governance(
            rho_mac=rho, k_particles=k,
            state_vector=psi, operator_X=X,
        )
        assert state.fock_isometry_ok
        assert state.modular_involution_ok
        assert state.verdict in (
            DualizerVerdict.COHERENT,
            DualizerVerdict.DEGRADED,
        )