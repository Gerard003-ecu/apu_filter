r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — Quantum CPTP Channel Validator                        ║
║  Ruta   : tests/unit/wisdom/test_cptp_validator.py                           ║
║  Versión: 6.0.0-Choi-Jamiolkowski-Kraus-Nested-Strict                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Cobertura por fases anidadas:                                               ║
║    FASE 1 — Dimensión Hilbert, tipado Kraus, calibre UV                      ║
║    FASE 2 — TP residual, Choi–Jamiołkowski, hermiticidad, Tr_B dual          ║
║    FASE 3 — CP espectral, PPT, retícula Ω_CPTP, sellado, rechazo monádico    ║
║    E2E    — audit_quantum_channel + fábricas + invariantes algebraicos       ║
║                                                                              ║
║  Framework: pytest + numpy.testing                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import FrozenInstanceError
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.mic_algebra import TopologicalInvariantError
from app.core.schemas import Stratum
from app.wisdom.cptp_validator import (
    CPTPChannelValidator,
    CPTPValidationReport,
    ComplexMatrix,
    _EPS_DEFAULT,
    _EPS_HERMITICITY,
    _EPS_SPECTRAL,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y TOLERANCIAS
# ═══════════════════════════════════════════════════════════════════════════════
_RTOL: float = 1.0e-10
_ATOL: float = 1.0e-12
_TOL: float = 1.0e-12
_SEED: int = 7
_D2: int = 2
_D3: int = 3
_D4: int = 4


# ═══════════════════════════════════════════════════════════════════════════════
# GENERADORES ALGEBRAICOS DE CANALES Y OPERADORES
# ═══════════════════════════════════════════════════════════════════════════════
def _eye(d: int) -> ComplexMatrix:
    return np.eye(d, dtype=np.complex128)


def _zero(d: int) -> ComplexMatrix:
    return np.zeros((d, d), dtype=np.complex128)


def _random_complex_matrix(d: int, rng: np.random.Generator) -> ComplexMatrix:
    return (
        rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    ).astype(np.complex128)


def _normalize_kraus_to_tp(ops: List[ComplexMatrix], d: int) -> List[ComplexMatrix]:
    r"""
    Renormaliza un ensamble arbitrario para forzar Σ M†M = I
    vía S^{-1/2} por la izquierda (si S ≻ 0).
    """
    s = np.zeros((d, d), dtype=np.complex128)
    for m in ops:
        s += m.conj().T @ m
    s = 0.5 * (s + s.conj().T)
    # Pseudo-inversa de raíz para soporte completo
    eigvals, eigvecs = la.eigh(s)
    eigvals_safe = np.maximum(eigvals, _EPS_SPECTRAL)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.conj().T
    return [(m @ inv_sqrt).astype(np.complex128) for m in ops]


def _non_cp_kraus(d: int = 2) -> List[ComplexMatrix]:
    """
    Ensamble que completa TP pero induce Choi con autovalor negativo
    si se manipula; aquí usamos un mapa positivo pero no CP clásico
    vía un único Kraus no-completo + corrección artificial.

    Construcción directa: canal de transposición (positivo, no CP)
    no admite Kraus CP; simulamos violando CP con un 'Kraus' sintético
    que no corresponde a un mapa CP — para TP usamos identidad parcial
    y añadimos un operador que rompe la positividad del Choi ensamblado
    de forma controlada en tests de CP (vía patch o ensamble ad-hoc).

    Alternativa fiable: construir Choi no-PSD y no pasar por Kraus.
    Para tests unitarios de fase 3 se mockea λ_min; para E2E se usa
    un ensamble cuya suma de outers no es PSD... lo cual es imposible
    si Λ = Σ |v⟩⟨v|. Por construcción Kraus ⇒ CP.

    Por tanto la violación CP E2E se logra solo si corrompemos Λ
    post-construcción (tests de fase 3 / reject). La violación TP
    sí es directa con Kraus incompletos.
    """
    # Kraus incompleto: √p · I  con p < 1  → no TP (y sí CP)
    p = 0.5
    return [math.sqrt(p) * _eye(d)]


def _unitary_kraus(d: int, rng: np.random.Generator) -> List[ComplexMatrix]:
    """Canal unitario: un solo Kraus unitario U."""
    a = _random_complex_matrix(d, rng)
    u, _ = la.qr(a)
    # QR complejo da unitario en Q
    return [np.asarray(u, dtype=np.complex128)]


def _replace_channel_kraus() -> List[ComplexMatrix]:
    r"""
    Canal de reemplazo a |0⟩⟨0| en d=2 (entanglement-breaking, CPTP):
    M_0 = |0⟩⟨0|, M_1 = |0⟩⟨1|.
    """
    m0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    m1 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    return [m0, m1]


def _choi_from_kraus(ops: List[ComplexMatrix], d: int) -> ComplexMatrix:
    """Réplica de referencia del isomorfismo vec_F para aserciones cruzadas."""
    choi_dim = d * d
    choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)
    for m in ops:
        v = m.flatten(order="F")
        choi += np.outer(v, v.conj())
    return 0.5 * (choi + choi.conj().T)


def _tp_residual(ops: List[ComplexMatrix], d: int) -> float:
    s = np.zeros((d, d), dtype=np.complex128)
    for m in ops:
        s += m.conj().T @ m
    return float(la.norm(s - np.eye(d), ord="fro"))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


@pytest.fixture
def validator() -> CPTPChannelValidator:
    return CPTPChannelValidator(target_stratum=Stratum.WISDOM)


@pytest.fixture
def identity_d2() -> List[ComplexMatrix]:
    return CPTPChannelValidator.identity_kraus(_D2)


@pytest.fixture
def identity_d3() -> List[ComplexMatrix]:
    return CPTPChannelValidator.identity_kraus(_D3)


@pytest.fixture
def amplitude_damping_050() -> List[ComplexMatrix]:
    return CPTPChannelValidator.amplitude_damping_kraus(0.5)


@pytest.fixture
def depolarizing_d2_p01() -> List[ComplexMatrix]:
    return CPTPChannelValidator.depolarizing_kraus(_D2, 0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE ASERCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def assert_hermitian(m: ComplexMatrix, tol: float = 1e-10) -> None:
    defect = float(la.norm(m - m.conj().T, ord="fro"))
    assert defect <= tol, f"‖M−M†‖_F={defect:.3e} > {tol:.3e}"


def assert_report_schema(r: CPTPValidationReport) -> None:
    assert isinstance(r, CPTPValidationReport)
    assert isinstance(r.is_cptp, bool)
    assert isinstance(r.is_separable, bool)
    assert isinstance(r.choi_rank, int) and r.choi_rank >= 0
    assert isinstance(r.kraus_count, int) and r.kraus_count >= 0
    assert r.kraus_completeness_residual >= 0.0
    assert r.tp_diamond_defect >= 0.0
    assert r.wilkinson_tolerance >= 0.0
    assert math.isfinite(r.choi_min_eigenvalue) or math.isnan(r.choi_min_eigenvalue)
    assert math.isfinite(r.choi_trace)
    if r.is_cptp:
        assert r.kraus_completeness_residual <= r.wilkinson_tolerance + _ATOL
        assert r.choi_min_eigenvalue >= -r.wilkinson_tolerance - _ATOL


def assert_is_tp(ops: List[ComplexMatrix], d: int, tol: float = 1e-10) -> None:
    assert _tp_residual(ops, d) <= tol


# ═══════════════════════════════════════════════════════════════════════════════
#                                    FASE 1
#              Consistencia dimensional y normalización de Kraus
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1ValidateHilbertDimension:
    """FASE 1.1 — _phase1_validate_hilbert_dimension"""

    def test_accepts_positive_integers(self, validator: CPTPChannelValidator) -> None:
        for d in (1, 2, 3, 8, 16):
            assert validator._phase1_validate_hilbert_dimension(d) == d

    def test_accepts_numpy_integer(self, validator: CPTPChannelValidator) -> None:
        assert validator._phase1_validate_hilbert_dimension(np.int64(4)) == 4

    def test_rejects_zero(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="Dimensión"):
            validator._phase1_validate_hilbert_dimension(0)

    def test_rejects_negative(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="Dimensión"):
            validator._phase1_validate_hilbert_dimension(-2)

    def test_rejects_float(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="Dimensión"):
            validator._phase1_validate_hilbert_dimension(2.5)  # type: ignore[arg-type]

    def test_d1_admissible_with_debug(
        self, validator: CPTPChannelValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="MIC.Wisdom.CPTPValidator"):
            assert validator._phase1_validate_hilbert_dimension(1) == 1
        assert any("d=1" in r.message for r in caplog.records)


class TestPhase1ValidateKrausEnsemble:
    """FASE 1.2 — _phase1_validate_kraus_ensemble"""

    def test_accepts_valid_ensemble(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        out = validator._phase1_validate_kraus_ensemble(identity_d2, _D2)
        assert len(out) == 1
        assert out[0].shape == (_D2, _D2)
        assert out[0].dtype == np.complex128

    def test_rejects_empty(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="vacío"):
            validator._phase1_validate_kraus_ensemble([], _D2)

    def test_rejects_none(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="vacío"):
            validator._phase1_validate_kraus_ensemble(None, _D2)  # type: ignore[arg-type]

    def test_rejects_wrong_shape(self, validator: CPTPChannelValidator) -> None:
        bad = [np.ones((2, 3), dtype=np.complex128)]
        with pytest.raises(TopologicalInvariantError, match="dimensional"):
            validator._phase1_validate_kraus_ensemble(bad, _D2)

    def test_rejects_non_ndarray(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="ndarray"):
            validator._phase1_validate_kraus_ensemble([[1, 0], [0, 1]], _D2)  # type: ignore[list-item]

    def test_rejects_nan(self, validator: CPTPChannelValidator) -> None:
        m = _eye(_D2)
        m[0, 0] = np.nan + 0.0j
        with pytest.raises(TopologicalInvariantError, match="NaN/Inf"):
            validator._phase1_validate_kraus_ensemble([m], _D2)

    def test_rejects_inf(self, validator: CPTPChannelValidator) -> None:
        m = _eye(_D2)
        m[1, 1] = np.inf + 0.0j
        with pytest.raises(TopologicalInvariantError, match="NaN/Inf"):
            validator._phase1_validate_kraus_ensemble([m], _D2)

    def test_coerces_dtype_to_complex128(self, validator: CPTPChannelValidator) -> None:
        m = np.eye(_D2, dtype=np.float64)
        out = validator._phase1_validate_kraus_ensemble([m], _D2)
        assert out[0].dtype == np.complex128

    def test_multiple_operators_indexed_error(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = [_eye(_D2), np.ones((3, 3), dtype=np.complex128)]
        with pytest.raises(TopologicalInvariantError, match="Kraus\\[1\\]"):
            validator._phase1_validate_kraus_ensemble(ops, _D2)


class TestPhase1NormalizeKrausGauge:
    """FASE 1.3 — _phase1_normalize_kraus_gauge"""

    def test_preserves_nonzero(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        out = validator._phase1_normalize_kraus_gauge(identity_d2, _TOL)
        assert len(out) == 1

    def test_discards_numerical_zeros(
        self, validator: CPTPChannelValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        ops = [_eye(_D2), _zero(_D2), 1e-20 * _eye(_D2)]
        with caplog.at_level(logging.INFO, logger="MIC.Wisdom.CPTPValidator"):
            out = validator._phase1_normalize_kraus_gauge(ops, _TOL)
        assert len(out) == 1
        assert any("descartados" in r.message for r in caplog.records)

    def test_all_zero_raises(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="nulos"):
            validator._phase1_normalize_kraus_gauge([_zero(_D2), _zero(_D2)], _TOL)


class TestPhase1ObserveKrausCertificate:
    """FASE 1.4 — _phase1_observe_kraus_certificate (composición terminal)"""

    def test_composition_returns_clean_and_d(
        self, validator: CPTPChannelValidator, identity_d3: List[ComplexMatrix]
    ) -> None:
        clean, d = validator._phase1_observe_kraus_certificate(
            identity_d3, _D3, _TOL
        )
        assert d == _D3
        assert len(clean) == 1
        assert clean[0].shape == (_D3, _D3)

    def test_invalid_dimension_short_circuits(
        self, validator: CPTPChannelValidator
    ) -> None:
        with pytest.raises(TopologicalInvariantError, match="Dimensión"):
            validator._phase1_observe_kraus_certificate([_eye(2)], 0, _TOL)

    def test_output_is_phase2_input_signature(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        """Contrato funtorial F1→F2: (clean, d) alimenta residual TP."""
        clean, d = validator._phase1_observe_kraus_certificate(
            identity_d2, _D2, _TOL
        )
        r_tp, s_op = validator._phase2_trace_preserving_residual(clean, d)
        assert r_tp <= _TOL
        assert s_op.shape == (_D2, _D2)


# ═══════════════════════════════════════════════════════════════════════════════
#                                    FASE 2
#              Trace-Preserving y construcción de Choi–Jamiołkowski
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2TracePreservingResidual:
    """FASE 2.1 — _phase2_trace_preserving_residual"""

    def test_identity_has_zero_residual(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        r, s = validator._phase2_trace_preserving_residual(identity_d2, _D2)
        assert r <= _ATOL
        np.testing.assert_allclose(s, _eye(_D2), atol=1e-12)

    def test_amplitude_damping_is_tp(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        r, s = validator._phase2_trace_preserving_residual(amplitude_damping_050, _D2)
        assert r <= 1e-10
        assert_hermitian(s)

    def test_incomplete_kraus_positive_residual(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = _non_cp_kraus(_D2)  # √0.5 I
        r, _ = validator._phase2_trace_preserving_residual(ops, _D2)
        assert r > 0.1

    def test_s_operator_is_psd(
        self, validator: CPTPChannelValidator, depolarizing_d2_p01: List[ComplexMatrix]
    ) -> None:
        _, s = validator._phase2_trace_preserving_residual(depolarizing_d2_p01, _D2)
        eigs = la.eigvalsh(s)
        assert np.all(eigs >= -1e-10)

    def test_unitary_channel_tp(
        self, validator: CPTPChannelValidator, rng: np.random.Generator
    ) -> None:
        ops = _unitary_kraus(_D3, rng)
        r, _ = validator._phase2_trace_preserving_residual(ops, _D3)
        assert r <= 1e-10


class TestPhase2ConstructChoiMatrix:
    """FASE 2.2 — _phase2_construct_choi_matrix"""

    def test_shape_d_squared(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        assert choi.shape == (_D2 * _D2, _D2 * _D2)

    def test_identity_choi_is_maximally_entangled_projector(
        self, validator: CPTPChannelValidator
    ) -> None:
        r"""
        Para ℰ = id, Λ = |Ω⟩⟨Ω| · d  (no normalizada) = Σ_{ij} |i⟩|i⟩⟨j|⟨j|.
        rank(Λ) = 1 y Tr(Λ) = d.
        """
        d = _D2
        choi = validator._phase2_construct_choi_matrix(
            CPTPChannelValidator.identity_kraus(d), d
        )
        assert_hermitian(choi)
        tr = float(np.real(np.trace(choi)))
        np.testing.assert_allclose(tr, d, atol=1e-10)
        eigs = la.eigvalsh(choi)
        rank = int(np.sum(eigs > 1e-10))
        assert rank == 1
        np.testing.assert_allclose(np.max(eigs), d, atol=1e-10)

    def test_matches_reference_construction(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(amplitude_damping_050, _D2)
        ref = _choi_from_kraus(amplitude_damping_050, _D2)
        np.testing.assert_allclose(choi, ref, atol=1e-12)

    def test_choi_is_psd_for_valid_kraus(
        self, validator: CPTPChannelValidator, depolarizing_d2_p01: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(depolarizing_d2_p01, _D2)
        eigs = la.eigvalsh(choi)
        assert np.all(eigs >= -1e-10)

    def test_rank_equals_kraus_count_for_lin_indep(
        self, validator: CPTPChannelValidator
    ) -> None:
        """Reemplazo a |0⟩⟨0|: 2 Kraus independientes ⇒ rank Choi = 2."""
        ops = _replace_channel_kraus()
        choi = validator._phase2_construct_choi_matrix(ops, _D2)
        eigs = la.eigvalsh(choi)
        rank = int(np.sum(eigs > 1e-10))
        assert rank == 2


class TestPhase2CertifyChoiHermiticity:
    """FASE 2.3 — _phase2_certify_choi_hermiticity"""

    def test_accepts_hermitian(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        out = validator._phase2_certify_choi_hermiticity(choi, _TOL)
        assert out is not None
        assert_hermitian(out)

    def test_rejects_strongly_non_hermitian(
        self, validator: CPTPChannelValidator
    ) -> None:
        bad = np.array(
            [[0, 10], [0, 0]], dtype=np.complex128
        )  # no hermítica; pad a 4×4
        bad4 = np.zeros((4, 4), dtype=np.complex128)
        bad4[0, 1] = 10.0 + 0.0j
        with pytest.raises(TopologicalInvariantError, match="hermiticidad"):
            validator._phase2_certify_choi_hermiticity(bad4, _TOL)


class TestPhase2ChoiPartialTraceTpDefect:
    """FASE 2.4 — _phase2_choi_partial_trace_tp_defect"""

    def test_identity_defect_near_zero(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        delta = validator._phase2_choi_partial_trace_tp_defect(choi, _D2)
        assert delta <= 1e-10

    def test_amplitude_damping_defect_near_zero(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(amplitude_damping_050, _D2)
        delta = validator._phase2_choi_partial_trace_tp_defect(choi, _D2)
        assert delta <= 1e-10

    def test_incomplete_kraus_positive_defect(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = _non_cp_kraus(_D2)
        choi = validator._phase2_construct_choi_matrix(ops, _D2)
        delta = validator._phase2_choi_partial_trace_tp_defect(choi, _D2)
        assert delta > 0.1

    def test_checksum_consistent_with_kraus_residual(
        self, validator: CPTPChannelValidator, depolarizing_d2_p01: List[ComplexMatrix]
    ) -> None:
        """Si r_TP≈0 entonces δ_⋄≈0 (checksum cruzado)."""
        r_tp, _ = validator._phase2_trace_preserving_residual(depolarizing_d2_p01, _D2)
        choi = validator._phase2_construct_choi_matrix(depolarizing_d2_p01, _D2)
        delta = validator._phase2_choi_partial_trace_tp_defect(choi, _D2)
        assert r_tp <= 1e-10
        assert delta <= 1e-9


class TestPhase2OrientChoiGeometry:
    """FASE 2.5 — _phase2_orient_choi_geometry (composición terminal Orient)"""

    def test_four_tuple_invariants(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        r_tp, choi, tr, delta = validator._phase2_orient_choi_geometry(
            amplitude_damping_050, _D2, _TOL
        )
        assert r_tp <= 1e-10
        assert choi.shape == (4, 4)
        assert_hermitian(choi)
        np.testing.assert_allclose(tr, float(np.real(np.trace(choi))), atol=1e-14)
        # TP ⇒ Tr(Λ) = d
        np.testing.assert_allclose(tr, _D2, atol=1e-10)
        assert delta <= 1e-10

    def test_output_feeds_phase3_signature(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        """Contrato funtorial F2→F3: choi entra directo a CP espectral."""
        _r, choi, _tr, _d = validator._phase2_orient_choi_geometry(
            identity_d2, _D2, _TOL
        )
        eigs, min_e, rank, is_cp = validator._phase3_spectral_complete_positivity(
            choi, _TOL
        )
        assert is_cp is True
        assert min_e >= -_TOL
        assert rank == 1
        assert len(eigs) == 4


# ═══════════════════════════════════════════════════════════════════════════════
#                                    FASE 3
#              CP espectral, PPT, retícula, sellado y rechazo
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3SpectralCompletePositivity:
    """FASE 3.1 — _phase3_spectral_complete_positivity"""

    def test_identity_is_cp_rank_one(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        eigs, min_e, rank, is_cp = validator._phase3_spectral_complete_positivity(
            choi, _TOL
        )
        assert is_cp is True
        assert min_e >= -_TOL
        assert rank == 1
        assert np.all(np.isfinite(eigs))

    def test_negative_eigenvalue_not_cp(
        self, validator: CPTPChannelValidator
    ) -> None:
        # Choi artificial con λ_min < 0
        choi = np.diag([1.0, 0.5, 0.1, -0.3]).astype(np.complex128)
        _e, min_e, rank, is_cp = validator._phase3_spectral_complete_positivity(
            choi, _TOL
        )
        assert is_cp is False
        assert min_e == pytest.approx(-0.3)
        assert rank == 3

    def test_rank_counts_above_tolerance(
        self, validator: CPTPChannelValidator
    ) -> None:
        choi = np.diag([1.0, 1e-14, 0.5, 0.0]).astype(np.complex128)
        _e, _m, rank, is_cp = validator._phase3_spectral_complete_positivity(
            choi, tolerance=1e-12
        )
        assert is_cp is True
        assert rank == 2  # 1.0 y 0.5


class TestPhase3PptSeparability:
    """FASE 3.2 — _phase3_ppt_separability"""

    def test_identity_choi_is_entangled_not_ppt_separable_as_state(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        r"""
        Choi de id ∝ |Ω⟩⟨Ω| es el estado máximamente entrelazado:
        PPT falla ⇒ is_separable = False.
        """
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        # Normalización no cambia el signo del espectro PPT
        is_sep = validator._phase3_ppt_separability(choi, _D2, _TOL)
        assert is_sep is False

    def test_replacement_channel_is_entanglement_breaking_ppt(
        self, validator: CPTPChannelValidator
    ) -> None:
        """Canal de reemplazo: Choi separable ⇒ PPT satisfecho."""
        ops = _replace_channel_kraus()
        choi = validator._phase2_construct_choi_matrix(ops, _D2)
        is_sep = validator._phase3_ppt_separability(choi, _D2, _TOL)
        assert is_sep is True

    def test_depolarizing_high_noise_tends_ppt(
        self, validator: CPTPChannelValidator
    ) -> None:
        """
        Depolarizante con p→1 ≈ canal completamente mezclado:
        Choi ∝ I ⇒ separable.
        """
        ops = CPTPChannelValidator.depolarizing_kraus(_D2, p=0.99)
        # renormalizar por si la convención de escala deja residuo
        ops = _normalize_kraus_to_tp(ops, _D2)
        choi = validator._phase2_construct_choi_matrix(ops, _D2)
        is_sep = validator._phase3_ppt_separability(choi, _D2, 1e-8)
        assert is_sep is True


class TestPhase3DecideCptpLattice:
    """FASE 3.3 — _phase3_decide_cptp_lattice"""

    @pytest.mark.parametrize(
        "is_tp,is_cp,expected",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
        ids=["cptp", "tp_only", "cp_only", "neither"],
    )
    def test_lattice_and(
        self,
        validator: CPTPChannelValidator,
        is_tp: bool,
        is_cp: bool,
        expected: bool,
    ) -> None:
        result = validator._phase3_decide_cptp_lattice(is_tp, is_cp, 0.0, 0.0)
        assert result is expected

    def test_error_log_on_failure(
        self, validator: CPTPChannelValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="MIC.Wisdom.CPTPValidator"):
            validator._phase3_decide_cptp_lattice(False, True, 1e-3, 0.0)
        assert any("inválido" in r.message for r in caplog.records)

    def test_info_log_on_success(
        self, validator: CPTPChannelValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="MIC.Wisdom.CPTPValidator"):
            validator._phase3_decide_cptp_lattice(True, True, 1e-14, 1e-15)
        assert any("certificado" in r.message.lower() for r in caplog.records)


class TestPhase3SealValidationReport:
    """FASE 3.4 — _phase3_seal_validation_report"""

    def test_seal_frozen_and_fields(
        self, validator: CPTPChannelValidator
    ) -> None:
        report = validator._phase3_seal_validation_report(
            is_cptp=True,
            completeness_residual=1e-14,
            min_eigen=1e-15,
            choi_rank=1,
            is_separable=False,
            choi_trace=2.0,
            tp_diamond_defect=1e-14,
            kraus_count=1,
            tolerance=_TOL,
        )
        assert_report_schema(report)
        assert report.is_cptp is True
        assert report.choi_rank == 1
        assert report.kraus_count == 1
        assert report.choi_trace == pytest.approx(2.0)
        assert report.wilkinson_tolerance == _TOL

        with pytest.raises((FrozenInstanceError, AttributeError)):
            report.is_cptp = False  # type: ignore[misc]

    def test_seal_failure_certificate(
        self, validator: CPTPChannelValidator
    ) -> None:
        report = validator._phase3_seal_validation_report(
            is_cptp=False,
            completeness_residual=0.5,
            min_eigen=-0.1,
            choi_rank=2,
            is_separable=False,
            choi_trace=1.0,
            tp_diamond_defect=0.4,
            kraus_count=1,
            tolerance=_TOL,
        )
        assert report.is_cptp is False
        assert report.choi_min_eigenvalue == pytest.approx(-0.1)


class TestPhase3RejectNonCptp:
    """FASE 3.Ω — _phase3_reject_non_cptp"""

    def test_passes_silently_when_cptp(
        self, validator: CPTPChannelValidator
    ) -> None:
        report = CPTPValidationReport(
            is_cptp=True,
            kraus_completeness_residual=0.0,
            choi_min_eigenvalue=0.0,
            choi_rank=1,
            is_separable=False,
        )
        validator._phase3_reject_non_cptp(report)  # no raise

    def test_raises_on_non_cptp(self, validator: CPTPChannelValidator) -> None:
        report = CPTPValidationReport(
            is_cptp=False,
            kraus_completeness_residual=0.3,
            choi_min_eigenvalue=-0.2,
            choi_rank=0,
            is_separable=False,
            tp_diamond_defect=0.25,
        )
        with pytest.raises(TopologicalInvariantError, match="TraceAnomalyError"):
            validator._phase3_reject_non_cptp(report)


# ═══════════════════════════════════════════════════════════════════════════════
#                    AUDIT_QUANTUM_CHANNEL — INTEGRACIÓN E2E
# ═══════════════════════════════════════════════════════════════════════════════
class TestAuditQuantumChannelIntegration:
    """Composición ObserveKraus → OrientChoi → CP/PPT/Seal/Reject"""

    def test_identity_channel_certified(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(identity_d2, _D2, _TOL)
        assert_report_schema(report)
        assert report.is_cptp is True
        assert report.choi_rank == 1
        assert report.kraus_count == 1
        assert report.is_separable is False  # Choi ∝ |Ω⟩⟨Ω|
        np.testing.assert_allclose(report.choi_trace, _D2, atol=1e-10)
        assert report.kraus_completeness_residual <= _TOL
        assert report.tp_diamond_defect <= 1e-10

    def test_identity_d3_certified(
        self, validator: CPTPChannelValidator, identity_d3: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(identity_d3, _D3, _TOL)
        assert report.is_cptp is True
        assert report.choi_rank == 1
        np.testing.assert_allclose(report.choi_trace, _D3, atol=1e-10)

    def test_amplitude_damping_certified(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(amplitude_damping_050, _D2, _TOL)
        assert report.is_cptp is True
        assert report.choi_rank == 2
        assert report.kraus_count == 2
        assert report.choi_min_eigenvalue >= -_TOL

    def test_depolarizing_certified(
        self, validator: CPTPChannelValidator, depolarizing_d2_p01: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(depolarizing_d2_p01, _D2, 1e-10)
        assert report.is_cptp is True
        assert report.choi_min_eigenvalue >= -1e-10

    def test_unitary_random_certified(
        self, validator: CPTPChannelValidator, rng: np.random.Generator
    ) -> None:
        ops = _unitary_kraus(_D4, rng)
        report = validator.audit_quantum_channel(ops, _D4, 1e-10)
        assert report.is_cptp is True
        assert report.choi_rank == 1
        assert report.is_separable is False

    def test_replacement_eb_channel_separable(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = _replace_channel_kraus()
        report = validator.audit_quantum_channel(ops, _D2, _TOL)
        assert report.is_cptp is True
        assert report.is_separable is True
        assert report.choi_rank == 2

    def test_incomplete_kraus_rejected(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = _non_cp_kraus(_D2)
        with pytest.raises(TopologicalInvariantError, match="TraceAnomalyError"):
            validator.audit_quantum_channel(ops, _D2, _TOL)

    def test_dimension_mismatch_rejected(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        with pytest.raises(TopologicalInvariantError, match="dimensional"):
            validator.audit_quantum_channel(identity_d2, _D3, _TOL)

    def test_empty_ensemble_rejected(
        self, validator: CPTPChannelValidator
    ) -> None:
        with pytest.raises(TopologicalInvariantError, match="vacío"):
            validator.audit_quantum_channel([], _D2, _TOL)

    def test_negative_tolerance_rejected(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        with pytest.raises(TopologicalInvariantError, match="tolerance"):
            validator.audit_quantum_channel(identity_d2, _D2, -1e-6)

    def test_nan_tolerance_rejected(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        with pytest.raises(TopologicalInvariantError, match="tolerance"):
            validator.audit_quantum_channel(identity_d2, _D2, float("nan"))

    def test_zero_operators_discarded_then_fail_or_pass(
        self, validator: CPTPChannelValidator
    ) -> None:
        """Identidad + ceros ⇒ calibre descarta ceros ⇒ CPTP OK."""
        ops = [_eye(_D2), _zero(_D2)]
        report = validator.audit_quantum_channel(ops, _D2, _TOL)
        assert report.is_cptp is True
        assert report.kraus_count == 1

    def test_all_zero_rejected(self, validator: CPTPChannelValidator) -> None:
        with pytest.raises(TopologicalInvariantError, match="nulos"):
            validator.audit_quantum_channel([_zero(_D2)], _D2, _TOL)

    def test_default_tolerance(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(identity_d2, _D2)
        assert report.wilkinson_tolerance == _EPS_DEFAULT
        assert report.is_cptp is True


# ═══════════════════════════════════════════════════════════════════════════════
#                         FÁBRICAS DE CANALES DE REFERENCIA
# ═══════════════════════════════════════════════════════════════════════════════
class TestFactoryIdentityKraus:
    def test_shape_and_tp(self) -> None:
        for d in (1, 2, 5):
            ops = CPTPChannelValidator.identity_kraus(d)
            assert len(ops) == 1
            assert ops[0].shape == (d, d)
            assert_is_tp(ops, d)


class TestFactoryAmplitudeDamping:
    @pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 1.0])
    def test_tp_and_cp_via_validator(
        self, validator: CPTPChannelValidator, gamma: float
    ) -> None:
        ops = CPTPChannelValidator.amplitude_damping_kraus(gamma)
        report = validator.audit_quantum_channel(ops, 2, 1e-12)
        assert report.is_cptp is True
        expected_rank = 1 if gamma == 0.0 else (1 if gamma == 0.0 else 2)
        if gamma == 0.0:
            assert report.choi_rank == 1
        elif gamma == 1.0:
            assert report.choi_rank == 2
        else:
            assert report.choi_rank == 2

    def test_gamma_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            CPTPChannelValidator.amplitude_damping_kraus(-0.1)
        with pytest.raises(ValueError, match="gamma"):
            CPTPChannelValidator.amplitude_damping_kraus(1.1)

    def test_m0_m1_structure(self) -> None:
        g = 0.3
        m0, m1 = CPTPChannelValidator.amplitude_damping_kraus(g)
        np.testing.assert_allclose(m0[0, 0], 1.0)
        np.testing.assert_allclose(m0[1, 1], math.sqrt(1 - g))
        np.testing.assert_allclose(m1[0, 1], math.sqrt(g))
        np.testing.assert_allclose(m1[1, 0], 0.0)


class TestFactoryDepolarizing:
    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
    def test_tp_property_d2(self, p: float) -> None:
        ops = CPTPChannelValidator.depolarizing_kraus(_D2, p)
        # Puede requerir renormalización numérica suave
        r = _tp_residual(ops, _D2)
        assert r <= 1e-8, f"r_TP={r} para p={p}"

    def test_p_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="p debe"):
            CPTPChannelValidator.depolarizing_kraus(_D2, -0.01)
        with pytest.raises(ValueError, match="p debe"):
            CPTPChannelValidator.depolarizing_kraus(_D2, 1.01)

    def test_p0_equals_identity_channel(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = CPTPChannelValidator.depolarizing_kraus(_D2, 0.0)
        report = validator.audit_quantum_channel(ops, _D2, 1e-10)
        assert report.is_cptp is True
        assert report.choi_rank == 1

    def test_d3_depolarizing_cptp(self, validator: CPTPChannelValidator) -> None:
        ops = CPTPChannelValidator.depolarizing_kraus(_D3, 0.2)
        # tolerancia algo más laxa por acumulación HW
        report = validator.audit_quantum_channel(ops, _D3, 1e-8)
        assert report.is_cptp is True


# ═══════════════════════════════════════════════════════════════════════════════
#                    INVARIANTES ALGEBRAICOS Y PROPIEDADES
# ═══════════════════════════════════════════════════════════════════════════════
class TestAlgebraicInvariants:
    """Propiedades que deben cumplirse ∀ canales CPTP válidos."""

    def test_report_is_frozen_slotted(self) -> None:
        r = CPTPValidationReport(
            is_cptp=True,
            kraus_completeness_residual=0.0,
            choi_min_eigenvalue=0.0,
            choi_rank=1,
            is_separable=False,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            r.choi_rank = 99  # type: ignore[misc]

    def test_validator_default_stratum_wisdom(self) -> None:
        v = CPTPChannelValidator()
        assert v._target_stratum == Stratum.WISDOM

    def test_choi_trace_equals_d_for_tp(
        self, validator: CPTPChannelValidator, rng: np.random.Generator
    ) -> None:
        for d in (2, 3):
            ops = _unitary_kraus(d, rng)
            report = validator.audit_quantum_channel(ops, d, 1e-10)
            np.testing.assert_allclose(report.choi_trace, d, atol=1e-9)

    def test_kraus_rank_bound_choi_rank(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        """rank(Λ) ≤ |{M_k}| (teorema de Choi)."""
        report = validator.audit_quantum_channel(amplitude_damping_050, _D2, _TOL)
        assert report.choi_rank <= report.kraus_count

    def test_tp_and_diamond_defect_vanish_together(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        report = validator.audit_quantum_channel(identity_d2, _D2, _TOL)
        assert report.kraus_completeness_residual <= _TOL
        assert report.tp_diamond_defect <= 1e-10

    def test_phase_nesting_single_choi_construction(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix]
    ) -> None:
        """construct_choi_matrix se invoca una sola vez en el pipeline E2E."""
        real = validator._phase2_construct_choi_matrix
        count = {"n": 0}

        def counting(ops, d):
            count["n"] += 1
            return real(ops, d)

        with patch.object(validator, "_phase2_construct_choi_matrix", side_effect=counting):
            validator.audit_quantum_channel(identity_d2, _D2, _TOL)
        assert count["n"] == 1

    def test_vec_f_convention_consistency(
        self, validator: CPTPChannelValidator
    ) -> None:
        r"""
        Para M = E_{01} = |0⟩⟨1| en d=2, vec_F(M) = (0,0,1,0)^T
        (column-major: col0=(0,0), col1=(1,0)).
        """
        m = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        choi = validator._phase2_construct_choi_matrix([m], 2)
        # Λ = |v⟩⟨v| con v = flatten_F(M)
        v = m.flatten(order="F")
        ref = np.outer(v, v.conj())
        np.testing.assert_allclose(choi, ref, atol=1e-14)

    def test_completeness_operator_eigenvalues_near_one(
        self, validator: CPTPChannelValidator, amplitude_damping_050: List[ComplexMatrix]
    ) -> None:
        r, s = validator._phase2_trace_preserving_residual(amplitude_damping_050, _D2)
        eigs = la.eigvalsh(s)
        np.testing.assert_allclose(eigs, np.ones(_D2), atol=1e-10)
        assert r <= 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
#                    FRONTERAS NUMÉRICAS Y CONDICIONAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
class TestNumericalBoundaries:
    def test_d1_scalar_channel(self, validator: CPTPChannelValidator) -> None:
        ops = CPTPChannelValidator.identity_kraus(1)
        report = validator.audit_quantum_channel(ops, 1, _TOL)
        assert report.is_cptp is True
        assert report.choi_rank == 1
        np.testing.assert_allclose(report.choi_trace, 1.0, atol=1e-12)

    def test_near_singular_damping(
        self, validator: CPTPChannelValidator
    ) -> None:
        ops = CPTPChannelValidator.amplitude_damping_kraus(1.0 - 1e-12)
        report = validator.audit_quantum_channel(ops, 2, 1e-10)
        assert report.is_cptp is True

    def test_many_redundant_kraus(
        self, validator: CPTPChannelValidator
    ) -> None:
        """
        Libertad unitaria de calibre: repartir I en dos Kraus
        M_0=M_1=I/√2 ⇒ CPTP, rank Choi = 1 < kraus_count.
        """
        s = 1.0 / math.sqrt(2.0)
        ops = [s * _eye(_D2), s * _eye(_D2)]
        report = validator.audit_quantum_channel(ops, _D2, _TOL)
        assert report.is_cptp is True
        assert report.kraus_count == 2
        assert report.choi_rank == 1

    def test_large_tolerance_accepts_mild_tp_defect(
        self, validator: CPTPChannelValidator
    ) -> None:
        # √(0.999)*I → r_TP pequeño
        ops = [math.sqrt(0.999) * _eye(_D2)]
        # con tol estricta falla
        with pytest.raises(TopologicalInvariantError):
            validator.audit_quantum_channel(ops, _D2, tolerance=1e-12)
        # con tol laxa pasa CP pero... aún no es TP al 1e-3 exacto;
        # r_TP = ‖0.999 I - I‖_F = 0.001 * √2 ≈ 1.4e-3
        report_ Tol = 1e-2
        # Usamos sellado manual evitando reject: verificamos fases
        clean, d = validator._phase1_observe_kraus_certificate(ops, _D2, 1e-2)
        r_tp, choi, tr, delta = validator._phase2_orient_choi_geometry(clean, d, 1e-2)
        assert r_tp < 1e-2
        _e, min_e, rank, is_cp = validator._phase3_spectral_complete_positivity(choi, 1e-2)
        assert is_cp is True

    def test_float32_input_coerced(
        self, validator: CPTPChannelValidator
    ) -> None:
        m = np.eye(2, dtype=np.float32)
        report = validator.audit_quantum_channel([m], 2, 1e-6)
        assert report.is_cptp is True


# ═══════════════════════════════════════════════════════════════════════════════
#                         TELEMETRÍA / LOGGING DE FASES
# ═══════════════════════════════════════════════════════════════════════════════
class TestLoggingTelemetry:
    def test_phase1_debug_on_ensemble(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="MIC.Wisdom.CPTPValidator"):
            validator._phase1_validate_kraus_ensemble(identity_d2, _D2)
        assert any("FASE1.2 Kraus" in r.message for r in caplog.records)

    def test_phase2_debug_on_tp(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="MIC.Wisdom.CPTPValidator"):
            validator._phase2_trace_preserving_residual(identity_d2, _D2)
        assert any("FASE2.1 TP" in r.message for r in caplog.records)

    def test_phase3_debug_on_cp(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        choi = validator._phase2_construct_choi_matrix(identity_d2, _D2)
        with caplog.at_level(logging.DEBUG, logger="MIC.Wisdom.CPTPValidator"):
            validator._phase3_spectral_complete_positivity(choi, _TOL)
        assert any("FASE3.1 CP" in r.message for r in caplog.records)

    def test_e2e_info_on_success(
        self, validator: CPTPChannelValidator, identity_d2: List[ComplexMatrix],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="MIC.Wisdom.CPTPValidator"):
            validator.audit_quantum_channel(identity_d2, _D2, _TOL)
        assert any("CPTP certificado" in r.message for r in caplog.records)

    def test_e2e_error_on_tp_violation(
        self, validator: CPTPChannelValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="MIC.Wisdom.CPTPValidator"):
            with pytest.raises(TopologicalInvariantError):
                validator.audit_quantum_channel(_non_cp_kraus(_D2), _D2, _TOL)
        assert any("inválido" in r.message for r in caplog.records)