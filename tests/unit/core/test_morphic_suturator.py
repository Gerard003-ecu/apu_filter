r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Suite  : Batería de Verificación Espectral — Morphic Suturator             ║
║  Ruta   : tests/unit/core/test_morphic_suturator.py                         ║
║  Versión: 2.0.0-Galois-Adjunction-Spectral-Strict-Granular-QA               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FILOSOFÍA DE LA SUITE:                                                      ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  La arquitectura de pruebas replica la topología de herencia del módulo      ║
║  bajo prueba:                                                                ║
║                                                                              ║
║      Phase1_MicAuditor                                                      ║
║              △                                                              ║
║      Phase2_MacCertifier(Phase1_MicAuditor)                                 ║
║              △                                                              ║
║      Phase3_GaloisSuturator(Phase2_MacCertifier)                            ║
║                                                                              ║
║  se traduce a:                                                              ║
║                                                                              ║
║      TestPhase1MicAuditor                                                   ║
║              △                                                              ║
║      TestPhase2MacCertifier(TestPhase1MicAuditor)                           ║
║              △                                                              ║
║      TestPhase3GaloisSuturator(TestPhase2MacCertifier)                      ║
║                                                                              ║
║  de modo que la última prueba granular de FASE 1 es literalmente el         ║
║  ancestro (objeto inicial) del primer método propio de FASE 2, y así        ║
║  sucesivamente. Una cuarta capa — TestMorphicSuturatorOrchestrator —        ║
║  cierra el ciclo Observe∥Orient→Decide→Seal/Veto.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pytest
import scipy.linalg as la

from app.core.morphic_suturator import (
    _DEFAULT_TOL,
    _EPS_HERMITICITY,
    _EPS_ORTHOGONALITY,
    _MACHINE_EPS,
    _SPECTRAL_PSD_FLOOR,
    GaloisAdjunctionBreachError,
    GaloisAdjunctionCertificate,
    LipschitzParameterError,
    MacDensityAnomalyError,
    MacHermiticityCertificate,
    MacHermiticityViolation,
    MacPositivitySpectralError,
    MacTraceAnomalyError,
    MicOrthogonalityBreachError,
    MicRankCertificate,
    MicRankDeficiencyError,
    MorphicSuturationState,
    MorphicSuturator,
    MorphicSuturatorError,
    NonFiniteInputError,
    Phase1_MicAuditor,
    Phase2_MacCertifier,
    Phase3_GaloisSuturator,
    ShapeMismatchError,
    Stratum,
    TopologicalInvariantError,
)

RealMatrix = np.ndarray
ComplexMatrix = np.ndarray
RealVector = np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# ORÁCULOS MATEMÁTICOS INDEPENDIENTES (desacoplados de la implementación)
# ═══════════════════════════════════════════════════════════════════════════════
def _fro(m: np.ndarray) -> float:
    """Norma de Frobenius, oráculo independiente."""
    return float(la.norm(m, ord="fro"))


def _is_hermitian(m: ComplexMatrix, tol: float = 1e-9) -> bool:
    return _fro(m - m.conj().T) <= tol


def _manual_gram_residual(m: RealMatrix) -> float:
    """Oráculo de ortogonalidad: ‖MMᵀ − I‖_F."""
    n = m.shape[0]
    return _fro(m @ m.T - np.eye(n))


def _manual_gershgorin(m: RealMatrix) -> float:
    """Oráculo del radio de discos de Gershgorin, calculado fila a fila."""
    n = m.shape[0]
    radii = []
    for i in range(n):
        diag = abs(m[i, i])
        off = sum(abs(m[i, j]) for j in range(n) if j != i)
        radii.append(diag + off)
    return float(max(radii))


def _manual_von_neumann_entropy(eigvals: np.ndarray) -> float:
    """Oráculo de entropía de von Neumann: S = -Σ λᵢ log(λᵢ), 0·log0 := 0."""
    clipped = np.clip(eigvals, 0.0, None)
    s = 0.0
    for lam in clipped:
        if lam > 1e-15:
            s -= lam * math.log(lam)
    return float(s)


def _haar_random_orthogonal(n: int, seed: int) -> RealMatrix:
    r"""
    Genera una matriz ortogonal Haar-aleatoria \(Q\in O(n)\) vía el método de
    Mezzadri (2007): QR de una matriz Ginibre real, con corrección de signo
    de la diagonal de R para eliminar el sesgo de la descomposición QR.

    \[
    Q^{T}Q = I_n \quad\text{exactamente, salvo error de redondeo FPU.}
    \]
    """
    rng = np.random.default_rng(seed)
    ginibre = rng.normal(size=(n, n))
    q, r = np.linalg.qr(ginibre)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    return q.astype(np.float64)


def _random_diagonally_dominant_matrix(n: int, seed: int) -> RealMatrix:
    r"""
    Genera una matriz real diagonalmente dominante (Levy–Desplanques ⇒ rango
    completo garantizado) pero genéricamente **no ortogonal** — oráculo de
    calibración para el invariante blando de ortogonalidad.
    """
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n)) + float(n) * np.eye(n)
    return m.astype(np.float64)


def _random_ginibre_density_matrix(d: int, seed: int) -> ComplexMatrix:
    r"""
    Genera un operador densidad físico \(\rho\succeq0,\operatorname{Tr}\rho=1\)
    vía el ensamble de Ginibre: \(\rho=GG^\dagger/\operatorname{Tr}(GG^\dagger)\).
    """
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = g @ g.conj().T
    rho = rho / np.real(np.trace(rho))
    return rho.astype(np.complex128)


def _random_pure_state_density(d: int, seed: int) -> ComplexMatrix:
    r"""Genera un estado puro \(\rho=|\psi\rangle\langle\psi|\), \(\psi\) Haar-aleatorio."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj()).astype(np.complex128)


# Especificaciones deterministas: (dimension, seed)
RANDOM_MIC_SPECS: List[Tuple[int, int]] = [(2, 1), (3, 2), (4, 3), (5, 4)]
RANDOM_DENSITY_SPECS: List[Tuple[int, int]] = [(2, 10), (3, 11), (4, 12), (5, 13)]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES CANÓNICAS COMPARTIDAS
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def suturator() -> MorphicSuturator:
    return MorphicSuturator()


@pytest.fixture
def identity_mic_2() -> RealMatrix:
    return MorphicSuturator.identity_mic(2)


@pytest.fixture
def identity_mic_3() -> RealMatrix:
    return MorphicSuturator.identity_mic(3)


@pytest.fixture
def mixed_mac_2() -> ComplexMatrix:
    return MorphicSuturator.maximally_mixed_mac(2)


@pytest.fixture
def identity_adjunction_2() -> Tuple[RealMatrix, RealMatrix]:
    return MorphicSuturator.identity_adjunction_pair(2)


@pytest.fixture
def singular_mic() -> RealMatrix:
    """MIC deliberadamente degenerada: columnas linealmente dependientes."""
    return np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)


@pytest.fixture
def non_hermitian_mac() -> ComplexMatrix:
    return np.array([[0.5, 1.0j], [0.0, 0.5]], dtype=np.complex128)


@pytest.fixture
def non_trace_normalized_mac() -> ComplexMatrix:
    return np.eye(2, dtype=np.complex128) * 0.5  # Tr = 1.0... ajustar a 0.8


@pytest.fixture
def broken_trace_mac() -> ComplexMatrix:
    m = np.diag([0.3, 0.5]).astype(np.complex128)  # Tr = 0.8 ≠ 1
    return m


@pytest.fixture
def negative_eigen_mac() -> ComplexMatrix:
    """Hermítica y traza=1, pero con autovalor negativo (no PSD)."""
    return np.diag([1.5, -0.5]).astype(np.complex128)


@pytest.fixture
def nan_mic() -> RealMatrix:
    m = np.eye(2, dtype=np.float64)
    m[0, 0] = np.nan
    return m


@pytest.fixture
def inf_mac() -> ComplexMatrix:
    m = np.eye(2, dtype=np.complex128)
    m[1, 1] = np.inf
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 (TEST) — Rango y Ortogonalidad de la MIC
# Ancla de la torre de herencia; toda subclase re-ejecuta esta batería.
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1MicAuditor:
    """
    Batería granular de FASE 1: valida los morfismos elementales

        Shape → Finite → Gershgorin → SpectralRadius → SVD
        → ConditionNumber → CertifyFullRank → OrthoResidual → audit_mic_rank(Ω)

    El certificado terminal ``MicRankCertificate`` (FASE 1.Ω) es consumido
    en paralelo por FASE 2 en el orquestador (no en cadena de datos directa).
    """

    # ── 1.1 · validación de forma ────────────────────────────────────────
    def test_validate_input_shape_accepts_square_matrix(self):
        m = np.eye(3, dtype=np.float64)
        assert Phase1_MicAuditor._phase1_validate_input_shape(m) == 3

    def test_validate_input_shape_rejects_non_ndarray(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_MicAuditor._phase1_validate_input_shape([[1, 0], [0, 1]])

    def test_validate_input_shape_rejects_1d_array(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_MicAuditor._phase1_validate_input_shape(np.array([1.0, 2.0]))

    def test_validate_input_shape_rejects_3d_array(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_MicAuditor._phase1_validate_input_shape(np.zeros((2, 2, 2)))

    def test_validate_input_shape_rejects_non_square(self):
        m = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(ShapeMismatchError):
            Phase1_MicAuditor._phase1_validate_input_shape(m)

    def test_validate_input_shape_rejects_zero_dimension(self):
        m = np.zeros((0, 0), dtype=np.float64)
        with pytest.raises(ShapeMismatchError):
            Phase1_MicAuditor._phase1_validate_input_shape(m)

    # ── 1.2 · finitud IEEE-754 ────────────────────────────────────────────
    def test_validate_finite_accepts_finite_matrix(self, identity_mic_2):
        assert Phase1_MicAuditor._phase1_validate_finite(identity_mic_2) is True

    def test_validate_finite_rejects_nan(self, nan_mic):
        with pytest.raises(NonFiniteInputError):
            Phase1_MicAuditor._phase1_validate_finite(nan_mic)

    def test_validate_finite_rejects_inf(self):
        m = np.eye(2, dtype=np.float64)
        m[0, 1] = np.inf
        with pytest.raises(NonFiniteInputError):
            Phase1_MicAuditor._phase1_validate_finite(m)

    # ── 1.3 · cota de Gershgorin ──────────────────────────────────────────
    def test_gershgorin_matches_manual_oracle_for_diagonal_matrix(self):
        m = np.diag([3.0, -5.0]).astype(np.float64)
        radius = Phase1_MicAuditor._phase1_gershgorin_disc_radius(m)
        assert radius == pytest.approx(_manual_gershgorin(m))
        assert radius == pytest.approx(5.0)

    def test_gershgorin_matches_manual_oracle_for_dense_matrix(self):
        m = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        radius = Phase1_MicAuditor._phase1_gershgorin_disc_radius(m)
        assert radius == pytest.approx(_manual_gershgorin(m))
        assert radius == pytest.approx(4.0)

    @pytest.mark.parametrize("d,seed", RANDOM_MIC_SPECS)
    def test_gershgorin_bounds_spectral_radius(self, d, seed):
        r"""Teorema (Círculo de Gershgorin): ρ(M) ≤ Gershgorin(M)."""
        m = _random_diagonally_dominant_matrix(d, seed)
        gersh = Phase1_MicAuditor._phase1_gershgorin_disc_radius(m)
        spectral = Phase1_MicAuditor._phase1_spectral_radius(m)
        assert spectral <= gersh + 1e-9

    # ── 1.4 · radio espectral ─────────────────────────────────────────────
    def test_spectral_radius_of_identity_is_one(self, identity_mic_3):
        assert Phase1_MicAuditor._phase1_spectral_radius(identity_mic_3) == pytest.approx(1.0)

    def test_spectral_radius_of_diagonal_matrix_matches_max_abs_eigenvalue(self):
        m = np.diag([2.0, -3.0, 1.0]).astype(np.float64)
        assert Phase1_MicAuditor._phase1_spectral_radius(m) == pytest.approx(3.0)

    @pytest.mark.parametrize("theta", [0.0, math.pi / 6, math.pi / 3, math.pi / 2, math.pi])
    def test_spectral_radius_of_orthogonal_rotation_is_unity(self, theta):
        r"""Teorema: para Q∈O(n), |λ(Q)|=1 ∀λ ⇒ ρ(Q)=1."""
        m = MorphicSuturator.rotation_mic(2, theta)
        radius = Phase1_MicAuditor._phase1_spectral_radius(m)
        assert radius == pytest.approx(1.0, abs=1e-9)

    # ── 1.5 · SVD ─────────────────────────────────────────────────────────
    def test_svd_shapes_and_reconstruction(self, identity_mic_3):
        u, s, vh = Phase1_MicAuditor._phase1_singular_value_decomposition(identity_mic_3)
        assert u.shape == (3, 3)
        assert s.shape == (3,)
        assert vh.shape == (3, 3)
        reconstructed = u @ np.diag(s) @ vh
        assert np.allclose(reconstructed, identity_mic_3, atol=1e-10)

    def test_svd_singular_values_are_sorted_descending_and_nonnegative(self):
        m = np.array([[3.0, 0.0], [4.0, 5.0]], dtype=np.float64)
        _u, s, _vh = Phase1_MicAuditor._phase1_singular_value_decomposition(m)
        assert np.all(s >= 0.0)
        assert np.all(np.diff(s) <= 1e-12)  # orden descendente

    def test_svd_raises_mic_rank_deficiency_error_on_linalg_failure(self, monkeypatch):
        def _boom(_m, **_kw):
            raise la.LinAlgError("SVD sintéticamente divergente")

        monkeypatch.setattr(la, "svd", _boom)
        with pytest.raises(MicRankDeficiencyError):
            Phase1_MicAuditor._phase1_singular_value_decomposition(np.eye(2))

    # ── 1.6 · número de condición ─────────────────────────────────────────
    def test_condition_number_of_identity_is_one(self):
        s = np.array([1.0, 1.0, 1.0])
        assert Phase1_MicAuditor._phase1_condition_number(s) == pytest.approx(1.0)

    def test_condition_number_is_infinite_for_zero_singular_value(self):
        s = np.array([2.0, 1.0, 0.0])
        assert Phase1_MicAuditor._phase1_condition_number(s) == float("inf")

    def test_condition_number_matches_ratio_of_extremes(self):
        s = np.array([10.0, 5.0, 2.0])
        assert Phase1_MicAuditor._phase1_condition_number(s) == pytest.approx(5.0)

    # ── 1.7 · certificación de rango completo ────────────────────────────
    def test_certify_full_rank_passes_for_full_rank_singular_values(self):
        s = np.array([2.0, 1.5, 1.0])
        rank, is_full = Phase1_MicAuditor._phase1_certify_full_rank(s, 3, 2.0)
        assert rank == 3
        assert is_full is True

    def test_certify_full_rank_raises_on_rank_deficiency(self):
        s = np.array([1.0, 1.0, 0.0])
        with pytest.raises(MicRankDeficiencyError):
            Phase1_MicAuditor._phase1_certify_full_rank(s, 3, float("inf"))

    def test_certify_full_rank_effective_rank_below_tolerance_threshold(self):
        s = np.array([1.0, 1e-20])  # 1e-20 por debajo del umbral de Higham
        with pytest.raises(MicRankDeficiencyError):
            Phase1_MicAuditor._phase1_certify_full_rank(s, 2, 1e20)

    # ── 1.8 · residuo de ortogonalidad ────────────────────────────────────
    def test_orthogonality_residual_zero_for_identity(self, identity_mic_3):
        residual = Phase1_MicAuditor._phase1_orthogonality_residual(identity_mic_3, 3)
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_orthogonality_residual_matches_manual_oracle(self):
        m = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        residual = Phase1_MicAuditor._phase1_orthogonality_residual(m, 2)
        assert residual == pytest.approx(_manual_gram_residual(m))
        assert residual == pytest.approx(3.0)

    @pytest.mark.parametrize("d,seed", RANDOM_MIC_SPECS)
    def test_orthogonality_residual_zero_for_haar_random_orthogonal_matrices(self, d, seed):
        q = _haar_random_orthogonal(d, seed)
        residual = Phase1_MicAuditor._phase1_orthogonality_residual(q, d)
        assert residual == pytest.approx(0.0, abs=1e-8)

    # ── 1.Ω · composición terminal Observe ────────────────────────────────
    def test_audit_mic_rank_end_to_end_identity(self, identity_mic_2):
        cert = Phase1_MicAuditor.audit_mic_rank(identity_mic_2)
        assert isinstance(cert, MicRankCertificate)
        assert cert.is_full_rank is True
        assert cert.is_orthogonal is True
        assert cert.effective_rank == 2
        assert cert.rank_deficiency_margin == 0
        assert cert.spectral_radius == pytest.approx(1.0)

    def test_audit_mic_rank_raises_on_singular_matrix(self, singular_mic):
        with pytest.raises(MicRankDeficiencyError):
            Phase1_MicAuditor.audit_mic_rank(singular_mic)

    def test_audit_mic_rank_raises_on_nonfinite_input(self, nan_mic):
        with pytest.raises(NonFiniteInputError):
            Phase1_MicAuditor.audit_mic_rank(nan_mic)

    def test_audit_mic_rank_soft_gate_reports_nonorthogonal_without_raising(self):
        m = _random_diagonally_dominant_matrix(3, seed=99)
        cert = Phase1_MicAuditor.audit_mic_rank(m)  # strict_orthogonality=False (default)
        assert cert.is_full_rank is True
        assert cert.is_orthogonal is False
        assert cert.orthogonality_deviation > 1e-6

    def test_audit_mic_rank_strict_orthogonality_raises_on_nonorthogonal_matrix(self):
        m = _random_diagonally_dominant_matrix(3, seed=99)
        with pytest.raises(MicOrthogonalityBreachError):
            Phase1_MicAuditor.audit_mic_rank(m, strict_orthogonality=True)

    def test_audit_mic_rank_strict_orthogonality_passes_for_haar_random_orthogonal(self):
        q = _haar_random_orthogonal(4, seed=7)
        cert = Phase1_MicAuditor.audit_mic_rank(q, strict_orthogonality=True)
        assert cert.is_orthogonal is True

    @pytest.mark.parametrize("d,seed", RANDOM_MIC_SPECS)
    def test_audit_mic_rank_gershgorin_and_spectral_radius_populated(self, d, seed):
        m = _random_diagonally_dominant_matrix(d, seed)
        cert = Phase1_MicAuditor.audit_mic_rank(m)
        assert cert.gershgorin_bound > 0.0
        assert cert.spectral_radius > 0.0
        assert cert.spectral_radius <= cert.gershgorin_bound + 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 (TEST) — Postulados de Dirac–von Neumann sobre la MAC
# Continuación directa de TestPhase1MicAuditor (herencia real).
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2MacCertifier(TestPhase1MicAuditor):
    """
    Batería granular de FASE 2: valida

        Shape → Finite → HermResidual → CertifyHerm → Weyl
        → TraceValue → CertifyTrace → SpectralPositivity
        → Purity → Entropy → audit_mac_density(Ω)
    """

    # ── 2.1 · validación de forma ────────────────────────────────────────
    def test_validate_input_shape_accepts_square_density(self, mixed_mac_2):
        assert Phase2_MacCertifier._phase2_validate_input_shape(mixed_mac_2) == 2

    def test_validate_input_shape_rejects_non_square_density(self):
        m = np.zeros((2, 3), dtype=np.complex128)
        with pytest.raises(ShapeMismatchError):
            Phase2_MacCertifier._phase2_validate_input_shape(m)

    def test_validate_input_shape_rejects_1d_density(self):
        with pytest.raises(ShapeMismatchError):
            Phase2_MacCertifier._phase2_validate_input_shape(np.array([1.0, 0.0]))

    # ── 2.2 · finitud IEEE-754 ────────────────────────────────────────────
    def test_validate_finite_accepts_finite_density(self, mixed_mac_2):
        assert Phase2_MacCertifier._phase2_validate_finite(mixed_mac_2) is True

    def test_validate_finite_rejects_inf_density(self, inf_mac):
        with pytest.raises(NonFiniteInputError):
            Phase2_MacCertifier._phase2_validate_finite(inf_mac)

    # ── 2.3 · residuo hermítico ───────────────────────────────────────────
    def test_hermiticity_residual_zero_for_hermitian_matrix(self, mixed_mac_2):
        residual = Phase2_MacCertifier._phase2_hermiticity_residual(mixed_mac_2)
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_hermiticity_residual_positive_for_non_hermitian(self, non_hermitian_mac):
        residual = Phase2_MacCertifier._phase2_hermiticity_residual(non_hermitian_mac)
        assert residual > 1e-3

    # ── 2.4 · certificación de hermiticidad ──────────────────────────────
    def test_certify_hermiticity_passes_within_tolerance(self):
        assert Phase2_MacCertifier._phase2_certify_hermiticity(1e-13, _DEFAULT_TOL) is True

    def test_certify_hermiticity_raises_beyond_tolerance(self):
        with pytest.raises(MacHermiticityViolation):
            Phase2_MacCertifier._phase2_certify_hermiticity(1e-3, _DEFAULT_TOL)

    def test_hermiticity_violation_is_subtype_of_density_anomaly(self):
        assert issubclass(MacHermiticityViolation, MacDensityAnomalyError)

    # ── 2.5 · proyección de Weyl ──────────────────────────────────────────
    def test_weyl_symmetrize_produces_exactly_hermitian_output(self, non_hermitian_mac):
        symmetrized = Phase2_MacCertifier._phase2_weyl_symmetrize(non_hermitian_mac)
        assert _is_hermitian(symmetrized, tol=1e-15)

    def test_weyl_symmetrize_matches_manual_formula(self, non_hermitian_mac):
        expected = 0.5 * (non_hermitian_mac + non_hermitian_mac.conj().T)
        result = Phase2_MacCertifier._phase2_weyl_symmetrize(non_hermitian_mac)
        assert np.allclose(result, expected)

    def test_weyl_symmetrize_is_noop_for_already_hermitian_matrix(self, mixed_mac_2):
        result = Phase2_MacCertifier._phase2_weyl_symmetrize(mixed_mac_2)
        assert np.allclose(result, mixed_mac_2)

    # ── 2.6 · valor de traza ──────────────────────────────────────────────
    def test_trace_value_of_maximally_mixed_state_is_one(self, mixed_mac_2):
        assert Phase2_MacCertifier._phase2_trace_value(mixed_mac_2) == pytest.approx(1.0)

    def test_trace_value_matches_manual_trace(self, broken_trace_mac):
        expected = float(np.real(np.trace(broken_trace_mac)))
        assert Phase2_MacCertifier._phase2_trace_value(broken_trace_mac) == pytest.approx(expected)

    # ── 2.7 · certificación de normalización de traza ─────────────────────
    def test_certify_trace_normalization_passes_for_unit_trace(self):
        assert Phase2_MacCertifier._phase2_certify_trace_normalization(1.0, _DEFAULT_TOL) is True

    def test_certify_trace_normalization_raises_for_broken_trace(self):
        with pytest.raises(MacTraceAnomalyError):
            Phase2_MacCertifier._phase2_certify_trace_normalization(0.8, _DEFAULT_TOL)

    def test_trace_anomaly_is_subtype_of_density_anomaly(self):
        assert issubclass(MacTraceAnomalyError, MacDensityAnomalyError)

    # ── 2.8 · positividad espectral ────────────────────────────────────────
    def test_spectral_positivity_passes_for_valid_density_matrix(self, mixed_mac_2):
        eigvals, min_eig, is_psd = Phase2_MacCertifier._phase2_spectral_positivity(mixed_mac_2)
        assert is_psd is True
        assert min_eig == pytest.approx(0.5)
        assert len(eigvals) == 2

    def test_spectral_positivity_raises_on_negative_eigenvalue(self, negative_eigen_mac):
        with pytest.raises(MacPositivitySpectralError):
            Phase2_MacCertifier._phase2_spectral_positivity(negative_eigen_mac)

    def test_positivity_spectral_error_is_subtype_of_density_anomaly(self):
        assert issubclass(MacPositivitySpectralError, MacDensityAnomalyError)

    def test_spectral_positivity_tolerates_wilkinson_floor(self):
        near_zero = np.diag([1.0 + 1e-14, -1e-14]).astype(np.complex128)
        _eigs, min_eig, is_psd = Phase2_MacCertifier._phase2_spectral_positivity(near_zero)
        assert is_psd is True  # -1e-14 > _SPECTRAL_PSD_FLOOR = -1e-13

    # ── 2.9 · pureza cuántica ──────────────────────────────────────────────
    def test_quantum_purity_of_pure_state_equals_one(self):
        rho = _random_pure_state_density(3, seed=21)
        purity, is_bounded = Phase2_MacCertifier._phase2_quantum_purity(rho, 3, _DEFAULT_TOL)
        assert purity == pytest.approx(1.0, abs=1e-9)
        assert is_bounded is True

    def test_quantum_purity_of_maximally_mixed_state_equals_inverse_dimension(self, mixed_mac_2):
        purity, is_bounded = Phase2_MacCertifier._phase2_quantum_purity(mixed_mac_2, 2, _DEFAULT_TOL)
        assert purity == pytest.approx(0.5, abs=1e-9)
        assert is_bounded is True

    def test_quantum_purity_detects_out_of_bound_artificial_matrix(self):
        r"""
        Ataque directo: matriz no física (0.1·I para d=2) con
        \(\operatorname{Tr}(\rho^2)=0.02 < 1/d=0.5\) ⇒ is_bounded=False.
        """
        artificial = (0.1 * np.eye(2)).astype(np.complex128)
        purity, is_bounded = Phase2_MacCertifier._phase2_quantum_purity(
            artificial, 2, _DEFAULT_TOL
        )
        assert purity == pytest.approx(0.02, abs=1e-9)
        assert is_bounded is False

    @pytest.mark.parametrize("d,seed", RANDOM_DENSITY_SPECS)
    def test_quantum_purity_bounded_for_random_ginibre_density_matrices(self, d, seed):
        rho = _random_ginibre_density_matrix(d, seed)
        purity, is_bounded = Phase2_MacCertifier._phase2_quantum_purity(rho, d, _DEFAULT_TOL)
        assert 1.0 / d - 1e-9 <= purity <= 1.0 + 1e-9
        assert is_bounded is True

    # ── 2.10 · entropía de von Neumann ──────────────────────────────────────
    def test_von_neumann_entropy_zero_for_pure_state(self):
        eigvals = np.array([1.0, 0.0, 0.0])
        entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)
        assert entropy == pytest.approx(0.0, abs=1e-12)

    def test_von_neumann_entropy_matches_analytic_log_d_for_maximally_mixed(self):
        for d in (2, 3, 4, 5):
            eigvals = np.full(d, 1.0 / d)
            entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)
            assert entropy == pytest.approx(math.log(d), abs=1e-9)

    def test_von_neumann_entropy_matches_manual_oracle(self):
        eigvals = np.array([0.7, 0.2, 0.1])
        expected = _manual_von_neumann_entropy(eigvals)
        entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)
        assert entropy == pytest.approx(expected, abs=1e-12)

    def test_von_neumann_entropy_clips_negative_roundoff_eigenvalues(self):
        eigvals = np.array([1.0 + 1e-16, -1e-16])
        entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)
        assert math.isfinite(entropy)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_von_neumann_entropy_handles_all_zero_support_defensively(self):
        eigvals = np.array([0.0, 0.0])
        entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)
        assert entropy == 0.0

    # ── 2.Ω · composición terminal Orient ─────────────────────────────────
    def test_audit_mac_density_end_to_end_maximally_mixed(self, mixed_mac_2):
        cert = Phase2_MacCertifier.audit_mac_density(mixed_mac_2)
        assert isinstance(cert, MacHermiticityCertificate)
        assert cert.is_hermitian is True
        assert cert.is_trace_normalized is True
        assert cert.is_positive_semidefinite is True
        assert cert.hilbert_dimension == 2
        assert cert.quantum_purity == pytest.approx(0.5, abs=1e-9)
        assert cert.von_neumann_entropy == pytest.approx(math.log(2), abs=1e-9)

    def test_audit_mac_density_raises_hermiticity_before_other_checks(self, non_hermitian_mac):
        r"""Orden de precedencia: hermiticidad se certifica antes que traza/positividad."""
        with pytest.raises(MacHermiticityViolation):
            Phase2_MacCertifier.audit_mac_density(non_hermitian_mac)

    def test_audit_mac_density_raises_trace_anomaly_for_broken_trace(self, broken_trace_mac):
        with pytest.raises(MacTraceAnomalyError):
            Phase2_MacCertifier.audit_mac_density(broken_trace_mac)

    def test_audit_mac_density_raises_positivity_error_for_negative_eigenvalue(
        self, negative_eigen_mac
    ):
        with pytest.raises(MacPositivitySpectralError):
            Phase2_MacCertifier.audit_mac_density(negative_eigen_mac)

    def test_audit_mac_density_raises_on_nonfinite_input(self, inf_mac):
        with pytest.raises(NonFiniteInputError):
            Phase2_MacCertifier.audit_mac_density(inf_mac)

    @pytest.mark.parametrize("d,seed", RANDOM_DENSITY_SPECS)
    def test_audit_mac_density_certifies_random_ginibre_states(self, d, seed):
        rho = _random_ginibre_density_matrix(d, seed)
        cert = Phase2_MacCertifier.audit_mac_density(rho)
        assert cert.is_hermitian is True
        assert cert.is_trace_normalized is True
        assert cert.is_positive_semidefinite is True
        assert cert.is_purity_bounded is True
        assert cert.von_neumann_entropy >= -1e-9

    @pytest.mark.parametrize("d,seed", RANDOM_DENSITY_SPECS)
    def test_audit_mac_density_pure_states_have_near_zero_entropy(self, d, seed):
        rho = _random_pure_state_density(d, seed)
        cert = Phase2_MacCertifier.audit_mac_density(rho)
        assert cert.quantum_purity == pytest.approx(1.0, abs=1e-8)
        assert cert.von_neumann_entropy == pytest.approx(0.0, abs=1e-7)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 (TEST) — Adjunción de Galois F ⊣ G
# Continuación directa de TestPhase2MacCertifier (herencia real).
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3GaloisSuturator(TestPhase2MacCertifier):
    """
    Batería granular de FASE 3: valida

        ValidateLipschitz → ValidateShapes → ValidateFinite
        → Reconstruct → UnitResidual → TargetDiscrepancy
        → CertifyLipschitz → Fidelity → audit_galois_adjunction(Ω)
    """

    # ── 3.1 · validación de parámetros escalares ─────────────────────────
    def test_validate_lipschitz_bound_accepts_nonnegative_finite(self):
        Phase3_GaloisSuturator._phase3_validate_lipschitz_bound(1.5, 1e-7)  # no debe lanzar

    @pytest.mark.parametrize("bad_l", [-1.0, float("nan"), float("inf")])
    def test_validate_lipschitz_bound_rejects_invalid_lipschitz(self, bad_l):
        with pytest.raises(LipschitzParameterError):
            Phase3_GaloisSuturator._phase3_validate_lipschitz_bound(bad_l, 1e-7)

    @pytest.mark.parametrize("bad_tol", [-1e-7, float("nan"), float("inf")])
    def test_validate_lipschitz_bound_rejects_invalid_tolerance(self, bad_tol):
        with pytest.raises(LipschitzParameterError):
            Phase3_GaloisSuturator._phase3_validate_lipschitz_bound(1.5, bad_tol)

    def test_validate_lipschitz_bound_accepts_zero(self):
        Phase3_GaloisSuturator._phase3_validate_lipschitz_bound(0.0, 0.0)  # no debe lanzar

    # ── 3.2 · compatibilidad dimensional ──────────────────────────────────
    def test_validate_dimensional_compatibility_accepts_consistent_shapes(self):
        x = np.zeros(3)
        y = np.zeros(2)
        f = np.zeros((2, 3))
        g = np.zeros((3, 2))
        Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(x, y, f, g)

    def test_validate_dimensional_compatibility_rejects_2d_state_x(self):
        x = np.zeros((3, 1))
        y = np.zeros(2)
        f = np.zeros((2, 3))
        g = np.zeros((3, 2))
        with pytest.raises(ShapeMismatchError):
            Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(x, y, f, g)

    def test_validate_dimensional_compatibility_rejects_mismatched_functor_f(self):
        x = np.zeros(3)
        y = np.zeros(2)
        f = np.zeros((5, 3))  # m debe ser 2, no 5
        g = np.zeros((3, 2))
        with pytest.raises(ShapeMismatchError):
            Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(x, y, f, g)

    def test_validate_dimensional_compatibility_rejects_mismatched_functor_g(self):
        x = np.zeros(3)
        y = np.zeros(2)
        f = np.zeros((2, 3))
        g = np.zeros((5, 2))  # n debe ser 3, no 5
        with pytest.raises(ShapeMismatchError):
            Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(x, y, f, g)

    def test_validate_dimensional_compatibility_rejects_non_ndarray_functor(self):
        x = np.zeros(2)
        y = np.zeros(2)
        f = [[1, 0], [0, 1]]
        g = np.eye(2)
        with pytest.raises(ShapeMismatchError):
            Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(x, y, f, g)

    # ── 3.3 · finitud conjunta ─────────────────────────────────────────────
    def test_validate_finite_accepts_all_finite_operands(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        Phase3_GaloisSuturator._phase3_validate_finite(x, y, f, g)  # no debe lanzar

    def test_validate_finite_rejects_nan_in_any_operand(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([np.nan, 2.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(NonFiniteInputError):
            Phase3_GaloisSuturator._phase3_validate_finite(x, y, f, g)

    def test_validate_finite_rejects_inf_in_functor(self):
        f = np.array([[np.inf, 0.0], [0.0, 1.0]])
        g = np.eye(2)
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(NonFiniteInputError):
            Phase3_GaloisSuturator._phase3_validate_finite(x, y, f, g)

    # ── 3.4 · reconstrucción G(F(X)) ──────────────────────────────────────
    def test_reconstruct_state_identity_functors_is_exact(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([3.0, -2.0])
        reconstructed = Phase3_GaloisSuturator._phase3_reconstruct_state(x, f, g)
        assert np.allclose(reconstructed, x)

    def test_reconstruct_state_matches_manual_composition(self):
        x = np.array([1.0, 2.0, 3.0])
        f = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # proyección 3→2
        g = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # inclusión 2→3
        expected = g @ (f @ x)
        result = Phase3_GaloisSuturator._phase3_reconstruct_state(x, f, g)
        assert np.allclose(result, expected)

    # ── 3.5 · residuo de la unidad de la adjunción ────────────────────────
    def test_unit_residual_zero_for_identity_adjunction(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([5.0, -1.0])
        reconstructed = Phase3_GaloisSuturator._phase3_reconstruct_state(x, f, g)
        residual = Phase3_GaloisSuturator._phase3_unit_residual(x, reconstructed)
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_unit_residual_matches_euclidean_norm_formula(self):
        x = np.array([1.0, 0.0])
        rec = np.array([0.0, 1.0])
        residual = Phase3_GaloisSuturator._phase3_unit_residual(x, rec)
        assert residual == pytest.approx(math.sqrt(2.0))

    # ── 3.6 · discrepancia del contra-dominio ─────────────────────────────
    def test_target_discrepancy_zero_when_fx_equals_y(self, identity_adjunction_2):
        f, _g = identity_adjunction_2
        x = np.array([2.0, 3.0])
        y = f @ x
        discrepancy = Phase3_GaloisSuturator._phase3_target_discrepancy(f, x, y)
        assert discrepancy == pytest.approx(0.0, abs=1e-12)

    def test_target_discrepancy_matches_manual_formula(self):
        f = np.eye(2)
        x = np.array([1.0, 1.0])
        y = np.array([0.0, 0.0])
        expected = float(la.norm(f @ x - y, ord=2))
        discrepancy = Phase3_GaloisSuturator._phase3_target_discrepancy(f, x, y)
        assert discrepancy == pytest.approx(expected)

    # ── 3.7 · certificación de Lipschitz ──────────────────────────────────
    def test_certify_lipschitz_condition_passes_when_within_bound(self):
        result = Phase3_GaloisSuturator._phase3_certify_lipschitz_condition(
            reconstruction_residual=1.0, target_diff=1.0, lipschitz_bound=2.0, tolerance=0.0
        )
        assert result is True

    def test_certify_lipschitz_condition_raises_when_violated(self):
        with pytest.raises(GaloisAdjunctionBreachError):
            Phase3_GaloisSuturator._phase3_certify_lipschitz_condition(
                reconstruction_residual=10.0, target_diff=1.0, lipschitz_bound=1.0, tolerance=0.0
            )

    def test_certify_lipschitz_condition_boundary_equality_does_not_violate(self):
        r"""Frontera exacta: residuo = L·Δ + ε ⇒ no debe violar (comparación estricta '>')."""
        result = Phase3_GaloisSuturator._phase3_certify_lipschitz_condition(
            reconstruction_residual=2.5, target_diff=2.0, lipschitz_bound=1.0, tolerance=0.5
        )
        assert result is True

    def test_certify_lipschitz_condition_infinitesimally_beyond_boundary_violates(self):
        with pytest.raises(GaloisAdjunctionBreachError):
            Phase3_GaloisSuturator._phase3_certify_lipschitz_condition(
                reconstruction_residual=2.5 + 1e-9,
                target_diff=2.0,
                lipschitz_bound=1.0,
                tolerance=0.5,
            )

    # ── 3.8 · fidelidad de reconstrucción ─────────────────────────────────
    def test_reconstruction_fidelity_is_one_for_identical_vectors(self):
        x = np.array([1.0, 0.0])
        fidelity, clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, x.copy())
        assert fidelity == pytest.approx(1.0, abs=1e-12)
        assert clipped is False

    def test_reconstruction_fidelity_is_zero_for_orthogonal_vectors(self):
        x = np.array([1.0, 0.0])
        rec = np.array([0.0, 1.0])
        fidelity, _clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
        assert fidelity == pytest.approx(0.0, abs=1e-12)

    def test_reconstruction_fidelity_is_one_for_antiparallel_vectors(self):
        r"""Fidelidad usa valor absoluto: la orientación no penaliza (análogo cuántico)."""
        x = np.array([1.0, 0.0])
        rec = np.array([-1.0, 0.0])
        fidelity, _clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
        assert fidelity == pytest.approx(1.0, abs=1e-12)

    def test_reconstruction_fidelity_defensive_zero_vector_x(self):
        x = np.array([0.0, 0.0])
        rec = np.array([1.0, 0.0])
        fidelity, clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
        assert fidelity == 0.0
        assert clipped is False

    def test_reconstruction_fidelity_defensive_zero_vector_reconstructed(self):
        x = np.array([1.0, 0.0])
        rec = np.array([0.0, 0.0])
        fidelity, clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
        assert fidelity == 0.0
        assert clipped is False

    def test_reconstruction_fidelity_clips_floating_point_overshoot(self, monkeypatch):
        r"""
        Ataque a la rama defensiva de recorte: fuerza \(\cos\theta>1\) por
        redondeo FPU inyectando un producto interno inflado artificialmente.
        """
        x = np.array([1.0, 1e-10])
        rec = np.array([1.0, 1e-10])

        original_dot = np.dot

        def _inflated_dot(a, b):
            return original_dot(a, b) * (1.0 + 1e-6)

        monkeypatch.setattr(np, "dot", _inflated_dot)
        fidelity, clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
        assert fidelity == pytest.approx(1.0, abs=1e-9)
        assert clipped is True

    def test_reconstruction_fidelity_bounded_in_unit_interval(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.normal(size=4)
            rec = rng.normal(size=4)
            fidelity, _clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(x, rec)
            assert 0.0 <= fidelity <= 1.0

    # ── 3.Ω · composición terminal Decide ─────────────────────────────────
    def test_audit_galois_adjunction_end_to_end_identity_pair(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([1.0, 2.0])
        y = f @ x
        cert = Phase3_GaloisSuturator.audit_galois_adjunction(x, y, f, g)
        assert isinstance(cert, GaloisAdjunctionCertificate)
        assert cert.is_adjunction_secured is True
        assert cert.adjunction_residual == pytest.approx(0.0, abs=1e-9)
        assert cert.reconstruction_fidelity == pytest.approx(1.0, abs=1e-9)
        assert cert.target_diff == pytest.approx(0.0, abs=1e-9)

    def test_audit_galois_adjunction_raises_on_invalid_lipschitz_bound(
        self, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(LipschitzParameterError):
            Phase3_GaloisSuturator.audit_galois_adjunction(
                x, y, f, g, lipschitz_bound=-1.0
            )

    def test_audit_galois_adjunction_raises_on_shape_mismatch(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 1.0])
        f = np.eye(2)  # forma incompatible con x de dim 3
        g = np.eye(2)
        with pytest.raises(ShapeMismatchError):
            Phase3_GaloisSuturator.audit_galois_adjunction(x, y, f, g)

    def test_audit_galois_adjunction_raises_breach_on_incoherent_functors(self):
        r"""
        F, G aleatorios e independientes (sin relación adjunta) deben violar
        la desigualdad de Lipschitz para una cota L pequeña.
        """
        rng = np.random.default_rng(3)
        n = 4
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        f = rng.normal(size=(n, n)) * 5.0
        g = rng.normal(size=(n, n)) * 5.0
        with pytest.raises(GaloisAdjunctionBreachError):
            Phase3_GaloisSuturator.audit_galois_adjunction(
                x, y, f, g, lipschitz_bound=1e-6, tolerance=1e-12
            )

    def test_audit_galois_adjunction_raises_on_nonfinite_operand(self, identity_adjunction_2):
        f, g = identity_adjunction_2
        x = np.array([np.nan, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(NonFiniteInputError):
            Phase3_GaloisSuturator.audit_galois_adjunction(x, y, f, g)


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — MorphicSuturator (Seal / Veto, ciclo completo)
# ═══════════════════════════════════════════════════════════════════════════════
class TestMorphicSuturatorOrchestrator:
    """Cierra el ciclo: Observe(MIC) ∥ Orient(MAC) → Decide(Galois) → Seal/Veto."""

    # ── inicialización ────────────────────────────────────────────────────
    def test_init_default_stratum_is_wisdom(self):
        s = MorphicSuturator()
        assert s._target_stratum == Stratum.WISDOM

    def test_init_accepts_custom_stratum(self):
        s = MorphicSuturator(target_stratum=Stratum.STRATEGY)
        assert s._target_stratum == Stratum.STRATEGY

    # ── ejecución exitosa (canales coherentes) ──────────────────────────────
    def test_execute_suturation_succeeds_for_fully_coherent_system(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 2.0])
        y = f @ x
        state = suturator.execute_suturation(identity_mic_2, mixed_mac_2, x, y, f, g)
        assert isinstance(state, MorphicSuturationState)
        assert state.is_sutured_coherent is True

    def test_execute_suturation_succeeds_with_pure_state_mac(self, suturator, identity_mic_2):
        psi = np.array([1.0 + 0.0j, 1.0 + 0.0j])
        rho = MorphicSuturator.pure_state_mac(psi)
        f, g = MorphicSuturator.identity_adjunction_pair(2)
        x = np.array([1.0, 1.0])
        y = f @ x
        state = suturator.execute_suturation(identity_mic_2, rho, x, y, f, g)
        assert state.is_sutured_coherent is True
        assert state.mac_audit.quantum_purity == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("d,seed", RANDOM_MIC_SPECS)
    def test_execute_suturation_succeeds_with_haar_random_orthogonal_mic(
        self, suturator, d, seed
    ):
        mic = _haar_random_orthogonal(d, seed)
        mac = MorphicSuturator.maximally_mixed_mac(d)
        f, g = MorphicSuturator.identity_adjunction_pair(d)
        rng = np.random.default_rng(seed)
        x = rng.normal(size=d)
        y = f @ x
        state = suturator.execute_suturation(mic, mac, x, y, f, g)
        assert state.is_sutured_coherent is True
        assert state.mic_audit.is_orthogonal is True

    # ── vetos por fase ────────────────────────────────────────────────────
    def test_execute_suturation_raises_mic_rank_deficiency(
        self, suturator, singular_mic, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(MicRankDeficiencyError):
            suturator.execute_suturation(singular_mic, mixed_mac_2, x, y, f, g)

    def test_execute_suturation_raises_mac_hermiticity_violation(
        self, suturator, identity_mic_2, non_hermitian_mac, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(MacHermiticityViolation):
            suturator.execute_suturation(identity_mic_2, non_hermitian_mac, x, y, f, g)

    def test_execute_suturation_raises_galois_breach(
        self, suturator, identity_mic_2, mixed_mac_2
    ):
        rng = np.random.default_rng(5)
        n = 2
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        f_incoherent = rng.normal(size=(n, n)) * 10.0
        g_incoherent = rng.normal(size=(n, n)) * 10.0
        with pytest.raises(GaloisAdjunctionBreachError):
            suturator.execute_suturation(
                identity_mic_2, mixed_mac_2, x, y, f_incoherent, g_incoherent,
                lipschitz_limit=1e-9, tolerance=1e-12,
            )

    # ── validación de tolerancia global ──────────────────────────────────
    @pytest.mark.parametrize("bad_tol", [-1.0, float("nan"), float("inf")])
    def test_execute_suturation_rejects_invalid_global_tolerance(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2, bad_tol
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with pytest.raises(MorphicSuturatorError):
            suturator.execute_suturation(
                identity_mic_2, mixed_mac_2, x, y, f, g, tolerance=bad_tol
            )

    # ── telemetría de veto y log crítico ─────────────────────────────────
    def test_execute_suturation_veto_logs_critical_and_reraises(
        self, suturator, singular_mic, mixed_mac_2, identity_adjunction_2, caplog
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])
        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.MorphicSuturator"):
            with pytest.raises(MicRankDeficiencyError):
                suturator.execute_suturation(singular_mic, mixed_mac_2, x, y, f, g)
        assert any("VETO DE SUTURA" in rec.message for rec in caplog.records)

    def test_execute_suturation_wraps_unexpected_linalg_error(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2, monkeypatch, caplog
    ):
        r"""
        Ataca la rama defensiva de captura de ``la.LinAlgError`` no
        categorizado, verificando que se envuelve como ``MorphicSuturatorError``
        y se re-propaga con log CRITICAL — sin fugar la excepción cruda de SciPy.
        """
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.0])

        def _boom(*_args, **_kwargs):
            raise la.LinAlgError("colapso sintético de LAPACK")

        monkeypatch.setattr(suturator, "audit_galois_adjunction", _boom)

        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.MorphicSuturator"):
            with pytest.raises(MorphicSuturatorError):
                suturator.execute_suturation(identity_mic_2, mixed_mac_2, x, y, f, g)
        assert any("VETO DE SUTURA" in rec.message for rec in caplog.records)

    # ── integridad del certificado sellado ────────────────────────────────
    def test_governance_state_is_frozen_dataclass(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = f @ x
        state = suturator.execute_suturation(identity_mic_2, mixed_mac_2, x, y, f, g)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_sutured_coherent = False  # type: ignore[misc]

    def test_governance_state_timestamp_is_iso8601_utc(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = f @ x
        state = suturator.execute_suturation(identity_mic_2, mixed_mac_2, x, y, f, g)
        parsed = datetime.fromisoformat(state.timestamp_utc)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_governance_state_records_wilkinson_tolerance(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = f @ x
        state = suturator.execute_suturation(
            identity_mic_2, mixed_mac_2, x, y, f, g, tolerance=1e-10
        )
        assert state.wilkinson_tolerance == pytest.approx(1e-10)

    def test_governance_state_records_stratum_name_as_string(
        self, suturator, identity_mic_2, mixed_mac_2, identity_adjunction_2
    ):
        f, g = identity_adjunction_2
        x = np.array([1.0, 1.0])
        y = f @ x
        state = suturator.execute_suturation(identity_mic_2, mixed_mac_2, x, y, f, g)
        assert state.stratum == "WISDOM"

    # ── conjunción de hard-gates (Ω.1) — tabla de verdad exhaustiva ─────────
    @staticmethod
    def _dummy_certs(
        is_full_rank: bool, is_psd: bool, is_secured: bool,
        is_orthogonal: bool = True, is_purity_bounded: bool = True,
    ):
        mic = MicRankCertificate(
            matrix_shape=(2, 2), effective_rank=2 if is_full_rank else 1,
            condition_number=1.0, is_full_rank=is_full_rank,
            orthogonality_deviation=0.0 if is_orthogonal else 1.0,
            is_orthogonal=is_orthogonal,
        )
        mac = MacHermiticityCertificate(
            is_hermitian=True, hermitician_residual=0.0, trace_value=1.0,
            is_trace_normalized=True, minimum_eigenvalue=0.0 if is_psd else -1.0,
            is_positive_semidefinite=is_psd, quantum_purity=0.5,
            is_purity_bounded=is_purity_bounded,
        )
        galois = GaloisAdjunctionCertificate(
            adjunction_residual=0.0 if is_secured else 100.0,
            is_adjunction_secured=is_secured, lipschitz_bound=1.5,
            reconstruction_fidelity=1.0 if is_secured else 0.0,
        )
        return mic, mac, galois

    @pytest.mark.parametrize("is_full_rank", [True, False])
    @pytest.mark.parametrize("is_psd", [True, False])
    @pytest.mark.parametrize("is_secured", [True, False])
    def test_seal_conjunction_truth_table(self, is_full_rank, is_psd, is_secured):
        mic, mac, galois = self._dummy_certs(is_full_rank, is_psd, is_secured)
        expected = is_full_rank and is_psd and is_secured
        result = MorphicSuturator._seal_coherence_conjunction(mic, mac, galois)
        assert result == expected

    @pytest.mark.parametrize("is_orthogonal", [True, False])
    @pytest.mark.parametrize("is_purity_bounded", [True, False])
    def test_seal_conjunction_is_independent_of_soft_invariants(
        self, is_orthogonal, is_purity_bounded
    ):
        r"""Ortogonalidad y cota de pureza son invariantes *blandos*: no afectan χ."""
        mic, mac, galois = self._dummy_certs(
            True, True, True, is_orthogonal=is_orthogonal, is_purity_bounded=is_purity_bounded
        )
        assert MorphicSuturator._seal_coherence_conjunction(mic, mac, galois) is True

    def test_seal_suturation_state_packs_all_three_audits(self, suturator):
        mic, mac, galois = self._dummy_certs(True, True, True)
        state = suturator._seal_suturation_state(mic, mac, galois, True, _DEFAULT_TOL)
        assert state.mic_audit is mic
        assert state.mac_audit is mac
        assert state.galois_audit is galois
        assert state.is_sutured_coherent is True

    def test_veto_log_and_reraise_propagates_same_exception_instance(self, caplog):
        err = MicRankDeficiencyError("residuo sintético de prueba")
        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.MorphicSuturator"):
            with pytest.raises(MicRankDeficiencyError) as exc_info:
                MorphicSuturator._veto_log_and_reraise(err)
        assert exc_info.value is err
        assert any("VETO DE SUTURA" in rec.message for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (retícula Ω₃ de vetos de sutura)
# ═══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    """Certifica la topología de la jerarquía de excepciones del módulo."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            NonFiniteInputError,
            ShapeMismatchError,
            MicRankDeficiencyError,
            MicOrthogonalityBreachError,
            MacDensityAnomalyError,
            LipschitzParameterError,
            GaloisAdjunctionBreachError,
        ],
    )
    def test_top_level_exceptions_inherit_from_suturator_error(self, exc_cls):
        assert issubclass(exc_cls, MorphicSuturatorError)

    @pytest.mark.parametrize(
        "exc_cls",
        [MacHermiticityViolation, MacTraceAnomalyError, MacPositivitySpectralError],
    )
    def test_mac_specific_exceptions_inherit_from_density_anomaly(self, exc_cls):
        assert issubclass(exc_cls, MacDensityAnomalyError)
        assert issubclass(exc_cls, MorphicSuturatorError)

    def test_morphic_suturator_error_inherits_topological_invariant_error(self):
        assert issubclass(MorphicSuturatorError, TopologicalInvariantError)

    def test_exceptions_are_mutually_distinguishable_types(self):
        assert MicRankDeficiencyError is not GaloisAdjunctionBreachError
        assert MacHermiticityViolation is not MacTraceAnomalyError

    def test_exceptions_carry_human_readable_message(self):
        err = ShapeMismatchError("mensaje de diagnóstico dimensional")
        assert "mensaje de diagnóstico dimensional" in str(err)

    def test_exceptions_are_catchable_via_common_root(self):
        with pytest.raises(MorphicSuturatorError):
            raise MacPositivitySpectralError("defecto de positividad sintético")


# ═══════════════════════════════════════════════════════════════════════════════
# FÁBRICAS DE REFERENCIA (calibración / tests del suturador)
# ═══════════════════════════════════════════════════════════════════════════════
class TestReferenceFactories:
    """Certifica la corrección algebraica de las fábricas de referencia expuestas."""

    def test_identity_mic_is_identity_matrix(self):
        m = MorphicSuturator.identity_mic(4)
        assert np.array_equal(m, np.eye(4))

    @pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi])
    def test_rotation_mic_is_orthogonal_for_all_angles(self, theta):
        m = MorphicSuturator.rotation_mic(3, theta)
        residual = _manual_gram_residual(m)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_rotation_mic_reduces_to_identity_for_n_less_than_two(self):
        m = MorphicSuturator.rotation_mic(1, math.pi / 3)
        assert np.array_equal(m, np.eye(1))

    def test_maximally_mixed_mac_has_correct_trace_and_purity(self):
        for d in (2, 3, 4, 5):
            rho = MorphicSuturator.maximally_mixed_mac(d)
            assert np.real(np.trace(rho)) == pytest.approx(1.0)
            purity = float(np.real(np.trace(rho @ rho)))
            assert purity == pytest.approx(1.0 / d)

    def test_pure_state_mac_normalizes_arbitrary_nonzero_vector(self):
        psi = np.array([3.0 + 0.0j, 4.0 + 0.0j])  # norma 5
        rho = MorphicSuturator.pure_state_mac(psi)
        assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-9)
        eigvals = la.eigvalsh(rho)
        assert np.sum(eigvals > 1e-9) == 1  # rango 1

    def test_pure_state_mac_rejects_zero_vector(self):
        psi = np.array([0.0 + 0.0j, 0.0 + 0.0j])
        with pytest.raises(ShapeMismatchError):
            MorphicSuturator.pure_state_mac(psi)

    def test_identity_adjunction_pair_returns_matching_identities(self):
        f, g = MorphicSuturator.identity_adjunction_pair(3)
        assert np.array_equal(f, np.eye(3))
        assert np.array_equal(g, np.eye(3))
        assert f is not g  # instancias independientes, no aliasing

    def test_identity_adjunction_pair_yields_zero_unit_residual_for_any_x(self):
        f, g = MorphicSuturator.identity_adjunction_pair(4)
        rng = np.random.default_rng(11)
        x = rng.normal(size=4)
        reconstructed = Phase3_GaloisSuturator._phase3_reconstruct_state(x, f, g)
        residual = Phase3_GaloisSuturator._phase3_unit_residual(x, reconstructed)
        assert residual == pytest.approx(0.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuración de ejecución local (pytest -q tests/unit/core/...)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))