# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — Sheaf Cohomology Orchestrator Agent                       ║
║  Ruta   : tests/unit/agents/boole/strategy/                                      ║
║           test_sheaf_cohomology_orchestrator_agent.py                            ║
║  Versión: 3.0.0-Test-Categorical-Krylov-Hodge                                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  FILOSOFÍA DE LA SUITE:                                                          ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Cada clase de prueba es la formalización computacional de un teorema            ║
║  matemático que el código implementa. Los tests no verifican comportamiento:     ║
║  prueban invariantes algebraicos, topológicos y termodinámicos.                  ║
║                                                                                  ║
║  ESTRUCTURA DE CLASES:                                                           ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  §T0  TestFiniteNumericalGuard      → Guardas (_FiniteNumericalGuard)            ║
║  §T1  TestPhase1_CohomologicalVeto  → Fase 1 (veto cohomológico)                 ║
║  §T2  TestPhase2_KrylovSpectral     → Fase 2 (espectro Krylov-Dirichlet)         ║
║  §T3  TestPhase3_HodgeProjection    → Fase 3 (proyección Hodge isoperimétrica)   ║
║  §T4  TestOrchestrator              → Orquestador (Z_SheafAgent = Φ₃∘Φ₂∘Φ₁)      ║
║  §T5  TestFunctorialChaining        → Anidamiento funtorial entre fases          ║
║  §T6  TestExceptionHierarchy        → Jerarquía de excepciones                   ║
║  §T7  TestDTOImmutability           → Inmutabilidad de DTOs                      ║
║  §T8  TestNumericalEdgeCases        → Casos límite numéricos extremos            ║
║  §T9  TestProvenanceAndChecksum     → Trazabilidad y checksums SHA-256           ║
║  §T10 TestMathematicalInvariants    → Invariantes algebraicos y topológicos      ║
║                                                                                  ║
║  CONVENCIONES MATEMÁTICAS:                                                       ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  · δ  = operador cofrontera C⁰ → C¹                                              ║
║  · H¹ = ker(δ¹)/im(δ⁰) = primer grupo de cohomología                             ║
║  · L  = δᵀδ = operador Laplaciano de Hodge                                       ║
║  · κ  = número de condición (σ_max/σ_min)                                        ║
║  · E  = energía de Dirichlet ‖δx‖₂²                                              ║
║  · β  = rank(δ)/dim(C¹) = índice de estabilidad cohomológica                     ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones estándar
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import struct
from datetime import datetime, timezone
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones científicas
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pytest
import scipy.linalg as la

# ─────────────────────────────────────────────────────────────────────────────
# Módulo bajo prueba
# ─────────────────────────────────────────────────────────────────────────────
from app.agents.boole.strategy.sheaf_cohomology_orchestrator_agent import (
    # Excepciones
    SheafCohomologyAgentError,
    TopologicalBifurcationError,
    PoincareLefschetzViolation,
    SpectralComputationError,
    SVDConvergenceError,
    HodgeDecompositionError,
    DirichletFrustrationError,
    PoincareBoundViolation,
    HomologicalInconsistencyError,
    LipschitzViolation,
    MinimalNormViolation,
    # DTOs
    CohomologicalVetoData,
    KrylovSpectralData,
    HodgeProjectionData,
    SheafAuditProvenance,
    SheafGovernanceState,
    # Fases
    Phase1_CohomologicalVetoCertifier,
    Phase2_KrylovSpectralAuditor,
    Phase3_IsoperimetricHodgeProjector,
    # Orquestador
    SheafCohomologyOrchestratorAgent,
    # Constantes internas
    _MACHINE_EPSILON,
    _SVD_TOLERANCE_BASE,
    _MAX_CONDITION_NUMBER_L,
    _FRUSTRATION_TOLERANCE,
    _INERTIA_DELTA_MAX,
    _NUMERICAL_SAFETY_FACTOR,
    _SPECTRAL_GAP_MIN_RATIO,
    _POINCARE_CONSTANT_MAX,
    _LIPSCHITZ_SLACK,
)

logger = logging.getLogger("TEST.MIC.SheafCohomologyAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# FÁBRICAS DE OBJETOS MATEMÁTICOS CANÓNICOS
# ═══════════════════════════════════════════════════════════════════════════════

def _full_rank_delta(m: int, n: int, seed: int = 42) -> np.ndarray:
    r"""
    Construye δ ∈ ℝ^{m×n} de rango mínimo(m,n) con valores singulares
    moderados (κ(δ) ≈ 10).

    Para que dim H¹ = 0 necesitamos rank(δ) = m (δ es inyectiva en C¹
    si m ≤ n, o rank = m si m < n).

    Estrategia:
        δ = U · Σ · Vᵀ  con Σ = diag(10, 9, ..., 10-r+1) (r = min(m,n)).
    """
    rng = np.random.default_rng(seed=seed)
    r = min(m, n)
    U, _ = la.qr(rng.standard_normal((m, m)))
    Vt, _ = la.qr(rng.standard_normal((n, n)))
    sigma_vals = np.linspace(10.0, 1.0, r)
    Sigma = np.zeros((m, n), dtype=np.float64)
    for i in range(r):
        Sigma[i, i] = sigma_vals[i]
    return (U[:, :m] @ Sigma @ Vt[:n, :]).astype(np.float64)


def _rank_deficient_delta(m: int, n: int, deficiency: int = 1, seed: int = 7) -> np.ndarray:
    r"""
    Construye δ ∈ ℝ^{m×n} con rank(δ) = m − deficiency.

    dim H¹ = m − rank(δ) = deficiency > 0.
    Esto debe disparar TopologicalBifurcationError.
    """
    rng = np.random.default_rng(seed=seed)
    r = min(m, n)
    effective_rank = max(0, r - deficiency)
    U, _ = la.qr(rng.standard_normal((m, m)))
    Vt, _ = la.qr(rng.standard_normal((n, n)))
    sigma_vals = np.array(
        [float(i + 1) for i in range(effective_rank)]
        + [0.0] * (r - effective_rank)
    )
    Sigma = np.zeros((m, n), dtype=np.float64)
    for i in range(r):
        Sigma[i, i] = sigma_vals[i]
    return (U[:, :m] @ Sigma @ Vt[:n, :]).astype(np.float64)


def _delta_with_known_kernel(n: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construye δ ∈ ℝ^{n×(n+1)} con rank = n exacto.
    ker(δ) = span{e_{n+1}} de dim = 1.

    El vector en ker(δ): v = [0,...,0,1] → δv = 0.
    La componente armónica de cualquier x ∈ C⁰ es π(x) = <x, v>·v.

    Retorna:
        (delta, kernel_vector).
    """
    delta = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        delta[i, i] = 1.0
    kernel_vector = np.zeros(n + 1, dtype=np.float64)
    kernel_vector[n] = 1.0
    return delta, kernel_vector


def _x_in_kernel(delta: np.ndarray, n_tries: int = 10, seed: int = 99) -> np.ndarray:
    r"""
    Construye x ∈ ker(δ) tal que δx = 0 usando la base del espacio nulo.
    """
    rng = np.random.default_rng(seed=seed)
    try:
        null_basis = la.null_space(delta)
    except Exception:
        return np.zeros(delta.shape[1], dtype=np.float64)

    if null_basis.shape[1] == 0:
        return np.zeros(delta.shape[1], dtype=np.float64)

    coeffs = rng.standard_normal(null_basis.shape[1])
    x = (null_basis @ coeffs).astype(np.float64)
    return x


def _x_near_kernel(
    delta: np.ndarray,
    epsilon: float = 1e-8,
    seed: int = 13,
) -> np.ndarray:
    r"""
    Construye x ≈ ker(δ) tal que ‖δx‖ = epsilon (frustración pequeña).
    """
    x_kern = _x_in_kernel(delta)
    rng = np.random.default_rng(seed=seed)
    perturbation = rng.standard_normal(x_kern.shape)
    # Normalizar la perturbación y escalarla para que ‖δ·pert‖ = epsilon
    delta_pert = delta @ perturbation
    norm_delta_pert = float(la.norm(delta_pert))
    if norm_delta_pert > 0:
        perturbation = perturbation * (epsilon / norm_delta_pert)
    return x_kern + perturbation


def _build_valid_governance_inputs(
    m: int = 3,
    n: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Construye (δ, x, x*) válidos para la gobernanza completa.

    δ: m×n de rango m → H¹ = 0.
    x: ≈ ker(δ) con frustración E(x) < ε_frust.
    x*: exactamente en ker(δ) → E(x*) = 0 ≤ E(x).

    Garantías:
        - ‖x − x*‖ < Δ_inertia.
        - ‖x*‖ ≤ ‖x‖ (mínima norma).
    """
    delta = _full_rank_delta(m, n, seed=seed)
    epsilon = _FRUSTRATION_TOLERANCE * 0.01  # muy por debajo del límite
    x = _x_near_kernel(delta, epsilon=epsilon, seed=seed)
    x_star = _x_in_kernel(delta, seed=seed)

    # Garantizar que ‖x − x*‖ < Δ_inertia
    displacement = float(la.norm(x - x_star))
    if displacement > _INERTIA_DELTA_MAX * 0.5:
        # Reescalar x_star para acercarlo
        x_star = x + (x_star - x) * (_INERTIA_DELTA_MAX * 0.4 / displacement)

    # Garantizar que ‖x*‖ ≤ ‖x‖ + ε_num
    norm_x = float(la.norm(x))
    norm_xstar = float(la.norm(x_star))
    if norm_xstar > norm_x + 1e-10:
        # Normalizar x_star para que ‖x*‖ ≤ ‖x‖
        if norm_xstar > 0:
            x_star = x_star * (norm_x / norm_xstar)

    return delta, x, x_star


def _make_valid_veto_cert(
    m: int = 3,
    n: int = 5,
    seed: int = 42,
) -> CohomologicalVetoData:
    r"""Genera un CohomologicalVetoData válido."""
    p1 = Phase1_CohomologicalVetoCertifier()
    delta = _full_rank_delta(m, n, seed=seed)
    return p1._certify_cohomological_veto_axiom(delta)


def _make_valid_spectral_cert(
    m: int = 3,
    n: int = 5,
    seed: int = 42,
) -> KrylovSpectralData:
    r"""Genera un KrylovSpectralData válido."""
    p2 = Phase2_KrylovSpectralAuditor()
    delta, x, _ = _build_valid_governance_inputs(m, n, seed=seed)
    veto = _make_valid_veto_cert(m, n, seed=seed)
    return p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)


def _make_invalid_veto_cert() -> CohomologicalVetoData:
    r"""Genera un CohomologicalVetoData con is_topologically_coherent=False."""
    return CohomologicalVetoData(
        dim_C0=4,
        dim_C1=3,
        delta_rank=2,
        h1_dimension=1,
        svd_tolerance=_SVD_TOLERANCE_BASE,
        max_singular_value=5.0,
        min_nonzero_singular_value=0.5,
        spectral_gap=4.5,
        spectral_gap_ratio=0.9,
        cohomological_stability_index=0.667,
        euler_characteristic_01=1,
        whitehead_torsion=1.0,
        poincare_lefschetz_ok=True,
        is_topologically_coherent=False,
    )


def _make_invalid_spectral_cert() -> KrylovSpectralData:
    r"""Genera un KrylovSpectralData con is_spectrally_stable=False."""
    return KrylovSpectralData(
        dirichlet_energy=0.0,
        dirichlet_energy_norm=0.0,
        frustration_tolerance=_FRUSTRATION_TOLERANCE,
        frustration_index=0.0,
        delta_condition_number=1.0,
        laplacian_condition_number=1.0,
        spectral_gap_effective=0.0,
        harmonic_component_norm=0.0,
        exact_component_norm=0.0,
        poincare_constant=0.0,
        is_frustration_bounded=True,
        is_spectrally_stable=False,
        is_poincare_bounded=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §T0 — TESTS DE GUARDAS NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class TestFiniteNumericalGuard:
    r"""
    Verifica _FiniteNumericalGuard ante entradas degeneradas.
    """

    @pytest.fixture
    def guard(self) -> Phase1_CohomologicalVetoCertifier:
        return Phase1_CohomologicalVetoCertifier()

    # ── T0.1: _as_float_array ─────────────────────────────────────────────────

    def test_accepts_valid_real_array(self, guard):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = guard._as_float_array("v", arr)
        np.testing.assert_array_equal(result, arr)

    def test_converts_int_to_float64(self, guard):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = guard._as_float_array("v", arr)
        assert result.dtype == np.float64

    def test_rejects_complex_array(self, guard):
        arr = np.array([1.0 + 2j, 3.0])
        with pytest.raises(TypeError, match="real"):
            guard._as_float_array("v", arr)

    def test_rejects_nan(self, guard):
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("v", arr)

    def test_rejects_positive_inf(self, guard):
        arr = np.array([1.0, np.inf])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("v", arr)

    def test_rejects_negative_inf(self, guard):
        arr = np.array([-np.inf, 1.0])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("v", arr)

    def test_rejects_non_numeric(self, guard):
        with pytest.raises(TypeError):
            guard._as_float_array("v", "no_soy_arreglo")

    # ── T0.2: _as_finite_matrix ───────────────────────────────────────────────

    def test_accepts_valid_2d_matrix(self, guard):
        M = np.eye(3, dtype=np.float64)
        result = guard._as_finite_matrix("M", M)
        np.testing.assert_array_equal(result, M)

    def test_rejects_1d_as_matrix(self, guard):
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            guard._as_finite_matrix("v", v)

    def test_rejects_matrix_below_min_rows(self, guard):
        M = np.ones((1, 5))
        with pytest.raises(ValueError, match="filas"):
            guard._as_finite_matrix("M", M, min_rows=2)

    def test_rejects_matrix_below_min_cols(self, guard):
        M = np.ones((3, 1))
        with pytest.raises(ValueError, match="columnas"):
            guard._as_finite_matrix("M", M, min_cols=2)

    def test_accepts_non_square_matrix(self, guard):
        M = np.ones((3, 5))
        result = guard._as_finite_matrix("M", M)
        assert result.shape == (3, 5)

    # ── T0.3: _as_finite_vector ───────────────────────────────────────────────

    def test_accepts_1d_vector(self, guard):
        v = np.array([1.0, 2.0, 3.0])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1

    def test_normalizes_column_vector(self, guard):
        v = np.array([[1.0], [2.0], [3.0]])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_normalizes_row_vector(self, guard):
        v = np.array([[1.0, 2.0, 3.0]])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1

    def test_accepts_scalar_as_dim1_vector(self, guard):
        result = guard._as_finite_vector("v", np.float64(3.14))
        assert result.ndim == 1
        assert result.size == 1

    def test_rejects_empty_when_not_allowed(self, guard):
        v = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="vacío"):
            guard._as_finite_vector("v", v, allow_empty=False)

    def test_accepts_empty_by_default(self, guard):
        v = np.array([], dtype=np.float64)
        result = guard._as_finite_vector("v", v)
        assert result.size == 0

    # ── T0.4: Normas ──────────────────────────────────────────────────────────

    def test_vector_norm_zero_vector(self, guard):
        assert guard._vector_norm(np.zeros(5)) == 0.0

    def test_vector_norm_unit_vector(self, guard):
        v = np.array([1.0, 0.0, 0.0])
        assert math.isclose(guard._vector_norm(v), 1.0, rel_tol=1e-12)

    def test_vector_norm_empty(self, guard):
        assert guard._vector_norm(np.array([])) == 0.0

    def test_vector_norm_known_value(self, guard):
        v = np.array([3.0, 4.0])
        assert math.isclose(guard._vector_norm(v), 5.0, rel_tol=1e-12)

    def test_frobenius_norm_identity(self, guard):
        I = np.eye(4)
        assert math.isclose(guard._frobenius_norm(I), 2.0, rel_tol=1e-12)

    def test_frobenius_norm_empty(self, guard):
        assert guard._frobenius_norm(np.zeros((0, 3))) == 0.0

    def test_squared_norm_positive(self, guard):
        v = np.array([3.0, 4.0])
        result = guard._squared_norm_from_vector(v)
        assert math.isclose(result, 25.0, rel_tol=1e-12)

    def test_squared_norm_zero_vector(self, guard):
        result = guard._squared_norm_from_vector(np.zeros(5))
        assert result == 0.0

    def test_squared_norm_empty(self, guard):
        result = guard._squared_norm_from_vector(np.array([]))
        assert result == 0.0

    # ── T0.5: _safe_svdvals ───────────────────────────────────────────────────

    def test_safe_svdvals_identity(self, guard):
        I = np.eye(4)
        svs = guard._safe_svdvals(I, "I")
        assert svs.size == 4
        assert np.allclose(svs, 1.0, atol=1e-12)

    def test_safe_svdvals_rectangular(self, guard):
        M = np.ones((3, 5))
        svs = guard._safe_svdvals(M, "M")
        assert svs.size == 3  # min(3,5) valores singulares

    def test_safe_svdvals_known_spectrum(self, guard):
        sigma = np.array([5.0, 3.0, 1.0])
        M = np.diag(sigma)
        svs = guard._safe_svdvals(M, "M")
        np.testing.assert_allclose(np.sort(svs)[::-1], sigma, rtol=1e-10)

    def test_safe_svdvals_empty_matrix(self, guard):
        M = np.zeros((0, 3))
        svs = guard._safe_svdvals(M, "M")
        assert svs.size == 0

    def test_safe_svdvals_all_finite(self, guard):
        rng = np.random.default_rng(seed=77)
        M = rng.standard_normal((4, 6))
        svs = guard._safe_svdvals(M, "M")
        assert np.all(np.isfinite(svs))

    # ── T0.6: _check_spectral_gap ─────────────────────────────────────────────

    def test_spectral_gap_clear_gap(self, guard):
        r"""Valores singulares [10, 9, 8, 0.001, 0.0005] → gap entre 8 y 0.001."""
        svs = np.array([10.0, 9.0, 8.0, 0.001, 0.0005])
        tol = _SVD_TOLERANCE_BASE
        rank, gap_ratio = guard._check_spectral_gap(svs, tol)
        assert rank == 3
        assert gap_ratio > _SPECTRAL_GAP_MIN_RATIO

    def test_spectral_gap_no_gap(self, guard):
        r"""Valores singulares uniformes → sin gap, se usa tolerancia estándar."""
        svs = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        tol = 0.5
        rank, gap_ratio = guard._check_spectral_gap(svs, tol)
        assert rank == 5  # todos > 0.5

    def test_spectral_gap_empty(self, guard):
        rank, gap = guard._check_spectral_gap(np.array([]), _SVD_TOLERANCE_BASE)
        assert rank == 0
        assert gap == 0.0

    def test_spectral_gap_single_value(self, guard):
        svs = np.array([5.0])
        rank, gap = guard._check_spectral_gap(svs, 1.0)
        assert rank == 1  # 5.0 > 1.0

    def test_spectral_gap_zero_sigma_max(self, guard):
        svs = np.array([0.0, 0.0])
        rank, gap = guard._check_spectral_gap(svs, _SVD_TOLERANCE_BASE)
        assert rank == 0
        assert gap == 0.0

    # ── T0.7: _pseudo_inverse ─────────────────────────────────────────────────

    def test_pseudo_inverse_identity(self, guard):
        r"""I† = I."""
        I = np.eye(4)
        I_pinv = guard._pseudo_inverse(I, "I")
        np.testing.assert_allclose(I_pinv, I, atol=1e-12)

    def test_pseudo_inverse_tall_matrix(self, guard):
        r"""Para A ∈ ℝ^{m×n} con m > n de rango n: A†A = I_n."""
        rng = np.random.default_rng(seed=11)
        A = rng.standard_normal((6, 3))
        A_pinv = guard._pseudo_inverse(A, "A")
        product = A_pinv @ A
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_pseudo_inverse_wide_matrix(self, guard):
        r"""Para A ∈ ℝ^{m×n} con m < n de rango m: AA† = I_m."""
        rng = np.random.default_rng(seed=22)
        A = rng.standard_normal((3, 6))
        A_pinv = guard._pseudo_inverse(A, "A")
        product = A @ A_pinv
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_pseudo_inverse_empty_matrix(self, guard):
        A = np.zeros((0, 3))
        A_pinv = guard._pseudo_inverse(A, "A")
        assert A_pinv.shape == (3, 0)

    # ── T0.8: _compute_input_checksum ────────────────────────────────────────

    def test_checksum_is_sha256_hex(self, guard):
        arr = np.eye(3)
        chk = guard._compute_input_checksum(arr)
        assert len(chk) == 64
        int(chk, 16)  # No lanza si es hex válido

    def test_checksum_deterministic(self, guard):
        arr = np.ones((3, 4))
        assert guard._compute_input_checksum(arr) == guard._compute_input_checksum(arr)

    def test_checksum_differs_for_different_inputs(self, guard):
        a1 = np.ones((3, 4))
        a2 = np.ones((3, 4)) * 2.0
        assert guard._compute_input_checksum(a1) != guard._compute_input_checksum(a2)

    def test_checksum_handles_none(self, guard):
        chk = guard._compute_input_checksum(None, np.eye(2))
        assert len(chk) == 64

    def test_checksum_multiple_arrays(self, guard):
        a = np.ones(3)
        b = np.zeros((2, 2))
        chk = guard._compute_input_checksum(a, b)
        assert len(chk) == 64


# ═══════════════════════════════════════════════════════════════════════════════
# §T1 — TESTS DE FASE 1: VETO COHOMOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1_CohomologicalVeto:
    r"""
    Verifica Phase1_CohomologicalVetoCertifier.

    Estructura:
        T1.1  → Tolerancia SVD adaptativa
        T1.2  → Torsión de Whitehead
        T1.3  → Poincaré-Lefschetz
        T1.4  → _certify_cohomological_veto_axiom (método principal)
    """

    @pytest.fixture
    def p1(self) -> Phase1_CohomologicalVetoCertifier:
        return Phase1_CohomologicalVetoCertifier()

    # ── T1.1: Tolerancia SVD adaptativa ──────────────────────────────────────

    def test_svd_tolerance_base_for_zero_sigma(self, p1):
        tol = p1._compute_svd_tolerance(0.0, (3, 5))
        assert tol == _SVD_TOLERANCE_BASE

    def test_svd_tolerance_base_for_negative_sigma(self, p1):
        tol = p1._compute_svd_tolerance(-1.0, (3, 5))
        assert tol == _SVD_TOLERANCE_BASE

    def test_svd_tolerance_increases_with_sigma(self, p1):
        tol_small = p1._compute_svd_tolerance(1.0, (3, 5))
        tol_large = p1._compute_svd_tolerance(1e8, (3, 5))
        assert tol_large >= tol_small

    def test_svd_tolerance_always_positive(self, p1):
        for sigma in [0.0, 1e-15, 1.0, 1e8]:
            tol = p1._compute_svd_tolerance(sigma, (4, 6))
            assert tol > 0.0

    def test_svd_tolerance_scales_with_matrix_size(self, p1):
        tol_small = p1._compute_svd_tolerance(5.0, (2, 3))
        tol_large = p1._compute_svd_tolerance(5.0, (100, 200))
        assert tol_large >= tol_small

    # ── T1.2: Torsión de Whitehead ────────────────────────────────────────────

    def test_whitehead_torsion_identity_sigma(self, p1):
        r"""log|τ_W| = Σ log(σᵢ) = 0 para σᵢ = 1."""
        svs = np.ones(4)
        torsion = p1._compute_whitehead_torsion(svs, 4)
        assert math.isclose(torsion, 0.0, abs_tol=1e-12)

    def test_whitehead_torsion_known_value(self, p1):
        r"""log|τ_W| = log(2) + log(3) para σ = [2, 3]."""
        svs = np.array([3.0, 2.0])  # orden descendente
        expected = math.log(3.0) + math.log(2.0)
        torsion = p1._compute_whitehead_torsion(svs, 2)
        assert math.isclose(torsion, expected, rel_tol=1e-10)

    def test_whitehead_torsion_zero_rank(self, p1):
        svs = np.array([5.0, 3.0])
        torsion = p1._compute_whitehead_torsion(svs, 0)
        assert torsion == 0.0

    def test_whitehead_torsion_empty_svs(self, p1):
        torsion = p1._compute_whitehead_torsion(np.array([]), 0)
        assert torsion == 0.0

    def test_whitehead_torsion_partial_rank(self, p1):
        r"""Solo usa los primeros `rank` valores singulares."""
        svs = np.array([5.0, 3.0, 1.0])
        torsion_full = p1._compute_whitehead_torsion(svs, 3)
        torsion_partial = p1._compute_whitehead_torsion(svs, 2)
        assert torsion_full > torsion_partial

    # ── T1.3: Poincaré-Lefschetz ──────────────────────────────────────────────

    def test_poincare_lefschetz_valid(self, p1):
        assert p1._verify_poincare_lefschetz(5, 3, 3) is True

    def test_poincare_lefschetz_rank_equals_min(self, p1):
        assert p1._verify_poincare_lefschetz(4, 4, 4) is True

    def test_poincare_lefschetz_rank_zero(self, p1):
        assert p1._verify_poincare_lefschetz(5, 3, 0) is True

    def test_poincare_lefschetz_violated(self, p1, caplog):
        r"""rank > min(dim C⁰, dim C¹) → False con warning."""
        with caplog.at_level(logging.WARNING):
            result = p1._verify_poincare_lefschetz(3, 3, 5)
        assert result is False
        assert any("Poincaré" in r.message for r in caplog.records)

    # ── T1.4: _certify_cohomological_veto_axiom ───────────────────────────────

    def test_certify_full_rank_passes(self, p1):
        r"""δ de rango máximo → H¹=0, certificado emitido."""
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert isinstance(cert, CohomologicalVetoData)
        assert cert.is_topologically_coherent is True
        assert cert.h1_dimension == 0

    def test_certify_returns_correct_dimensions(self, p1):
        m, n = 3, 7
        delta = _full_rank_delta(m, n)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.dim_C0 == n
        assert cert.dim_C1 == m

    def test_certify_rank_matches_expectation(self, p1):
        m, n = 4, 6
        delta = _full_rank_delta(m, n)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.delta_rank == m  # rango = m para δ inyectiva en C¹

    def test_certify_rank_deficient_raises(self, p1):
        r"""δ con rank < dim C¹ → dim H¹ > 0 → TopologicalBifurcationError."""
        delta = _rank_deficient_delta(4, 6, deficiency=1)
        with pytest.raises(TopologicalBifurcationError):
            p1._certify_cohomological_veto_axiom(delta)

    def test_certify_rank_deficient_2_raises(self, p1):
        r"""Deficiencia de rango 2 también lanza la excepción."""
        delta = _rank_deficient_delta(4, 6, deficiency=2)
        with pytest.raises(TopologicalBifurcationError):
            p1._certify_cohomological_veto_axiom(delta)

    def test_certify_h1_dimension_zero(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0

    def test_certify_sigma_max_is_finite_positive(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isfinite(cert.max_singular_value)
        assert cert.max_singular_value > 0.0

    def test_certify_sigma_min_nonzero_leq_sigma_max(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.min_nonzero_singular_value <= cert.max_singular_value

    def test_certify_spectral_gap_nonneg(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.spectral_gap >= 0.0

    def test_certify_spectral_gap_ratio_in_unit_interval(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert 0.0 <= cert.spectral_gap_ratio <= 1.0

    def test_certify_stability_index_in_unit_interval(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert 0.0 <= cert.cohomological_stability_index <= 1.0

    def test_certify_stability_index_one_for_full_rank(self, p1):
        r"""Para δ de rango máximo: β = rank/dim C¹ = 1."""
        m = 3
        delta = _full_rank_delta(m, m + 2)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isclose(cert.cohomological_stability_index, 1.0, rel_tol=1e-10)

    def test_certify_euler_characteristic_01(self, p1):
        m, n = 3, 7
        delta = _full_rank_delta(m, n)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.euler_characteristic_01 == n - m

    def test_certify_whitehead_torsion_finite(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isfinite(cert.whitehead_torsion)

    def test_certify_poincare_lefschetz_ok(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.poincare_lefschetz_ok is True

    def test_certify_rejects_nan_input(self, p1):
        delta = _full_rank_delta(3, 5)
        delta[0, 0] = np.nan
        with pytest.raises((ValueError, TopologicalBifurcationError)):
            p1._certify_cohomological_veto_axiom(delta)

    def test_certify_rejects_inf_input(self, p1):
        delta = _full_rank_delta(3, 5)
        delta[1, 2] = np.inf
        with pytest.raises((ValueError, TopologicalBifurcationError)):
            p1._certify_cohomological_veto_axiom(delta)

    def test_certify_rejects_1d_input(self, p1):
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            p1._certify_cohomological_veto_axiom(v)

    def test_certify_square_full_rank_passes(self, p1):
        r"""Matriz cuadrada de rango máximo → H¹ = 0."""
        delta = _full_rank_delta(4, 4)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0

    def test_certify_identity_is_full_rank(self, p1):
        r"""La identidad I_n tiene rank n → H¹ = 0."""
        delta = np.eye(4, dtype=np.float64)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.delta_rank == 4
        assert cert.h1_dimension == 0

    def test_certify_svd_tolerance_is_positive(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.svd_tolerance > 0.0

    def test_certify_all_fields_finite(self, p1):
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isfinite(cert.svd_tolerance)
        assert math.isfinite(cert.max_singular_value)
        assert math.isfinite(cert.min_nonzero_singular_value)
        assert math.isfinite(cert.spectral_gap)
        assert math.isfinite(cert.spectral_gap_ratio)
        assert math.isfinite(cert.cohomological_stability_index)
        assert math.isfinite(cert.whitehead_torsion)

    def test_certify_large_delta(self, p1):
        r"""Prueba con δ grande (m=10, n=20)."""
        delta = _full_rank_delta(10, 20)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0
        assert cert.delta_rank == 10

    def test_certify_single_row_delta(self, p1):
        r"""δ ∈ ℝ^{1×n}: H¹ = 1 − rank. Pasará si rank = 1."""
        delta = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0
        assert cert.delta_rank == 1


# ═══════════════════════════════════════════════════════════════════════════════
# §T2 — TESTS DE FASE 2: ESPECTRO DE KRYLOV-DIRICHLET
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2_KrylovSpectral:
    r"""
    Verifica Phase2_KrylovSpectralAuditor.

    Estructura:
        T2.1  → _kappa_from_certificate
        T2.2  → _approximate_hodge_decomposition
        T2.3  → _compute_poincare_constant
        T2.4  → _audit_krylov_spectral_stability (método principal)
    """

    @pytest.fixture
    def p2(self) -> Phase2_KrylovSpectralAuditor:
        return Phase2_KrylovSpectralAuditor()

    @pytest.fixture
    def valid_veto(self) -> CohomologicalVetoData:
        return _make_valid_veto_cert()

    # ── T2.1: _kappa_from_certificate ────────────────────────────────────────

    def test_kappa_identity_cert(self, p2):
        r"""Para δ=I: σ_max=σ_min=1 → κ(δ)=1, κ(L)=1."""
        delta = np.eye(3)
        p1 = Phase1_CohomologicalVetoCertifier()
        cert = p1._certify_cohomological_veto_axiom(delta)
        kd, kl = p2._kappa_from_certificate(cert)
        assert math.isclose(kd, 1.0, rel_tol=1e-10)
        assert math.isclose(kl, 1.0, rel_tol=1e-10)

    def test_kappa_zero_sigma_max(self, p2):
        r"""δ = 0 (σ_max=0) → κ=1, κL=1 (trivialmente estable)."""
        cert = CohomologicalVetoData(
            dim_C0=3, dim_C1=2, delta_rank=0, h1_dimension=0,
            svd_tolerance=1e-10,
            max_singular_value=0.0,
            min_nonzero_singular_value=0.0,
            spectral_gap=0.0, spectral_gap_ratio=0.0,
            cohomological_stability_index=0.0,
            euler_characteristic_01=1,
            whitehead_torsion=0.0,
            poincare_lefschetz_ok=True,
            is_topologically_coherent=True,
        )
        kd, kl = p2._kappa_from_certificate(cert)
        assert kd == 1.0
        assert kl == 1.0

    def test_kappa_sigma_min_zero(self, p2):
        r"""σ_min=0 → κ=∞."""
        cert = CohomologicalVetoData(
            dim_C0=3, dim_C1=2, delta_rank=1, h1_dimension=0,
            svd_tolerance=1e-10,
            max_singular_value=5.0,
            min_nonzero_singular_value=0.0,
            spectral_gap=5.0, spectral_gap_ratio=1.0,
            cohomological_stability_index=0.5,
            euler_characteristic_01=1,
            whitehead_torsion=1.0,
            poincare_lefschetz_ok=True,
            is_topologically_coherent=True,
        )
        kd, kl = p2._kappa_from_certificate(cert)
        assert kd == math.inf

    def test_kappa_known_ratio(self, p2):
        r"""σ_max=10, σ_min=2 → κ=5, κL=25."""
        cert = CohomologicalVetoData(
            dim_C0=4, dim_C1=3, delta_rank=3, h1_dimension=0,
            svd_tolerance=1e-10,
            max_singular_value=10.0,
            min_nonzero_singular_value=2.0,
            spectral_gap=8.0, spectral_gap_ratio=0.8,
            cohomological_stability_index=1.0,
            euler_characteristic_01=1,
            whitehead_torsion=2.3,
            poincare_lefschetz_ok=True,
            is_topologically_coherent=True,
        )
        kd, kl = p2._kappa_from_certificate(cert)
        assert math.isclose(kd, 5.0, rel_tol=1e-10)
        assert math.isclose(kl, 25.0, rel_tol=1e-10)

    # ── T2.2: _approximate_hodge_decomposition ────────────────────────────────

    def test_hodge_decomp_kernel_vector_is_harmonic(self, p2):
        r"""
        Si x ∈ ker(δ): δx = 0 → x_exact = δ†·0 = 0 → x_harm = x.
        ‖x_harm‖ = ‖x‖, ‖x_exact‖ ≈ 0.
        """
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        delta_x = delta @ x
        harm_norm, exact_norm = p2._approximate_hodge_decomposition(
            delta, x, delta_x, _SVD_TOLERANCE_BASE,
        )
        assert math.isclose(exact_norm, 0.0, abs_tol=1e-10)
        assert math.isclose(harm_norm, float(la.norm(x)), rel_tol=1e-8)

    def test_hodge_decomp_norms_nonnegative(self, p2):
        delta = _full_rank_delta(3, 5)
        x = np.ones(5) * 0.001
        delta_x = delta @ x
        harm_norm, exact_norm = p2._approximate_hodge_decomposition(
            delta, x, delta_x, _SVD_TOLERANCE_BASE,
        )
        assert harm_norm >= 0.0
        assert exact_norm >= 0.0

    def test_hodge_decomp_pythagoras(self, p2):
        r"""
        La descomposición de Hodge es ortogonal:
        ‖x‖² = ‖x_harm‖² + ‖x_exact‖² (aproximado).
        """
        delta, kernel_vec = _delta_with_known_kernel(4)
        # x con componente en ker y fuera de ker
        x = np.zeros(5)
        x[0] = 1.0  # fuera de ker (δe₀ = e₀ ≠ 0)
        x[4] = 1.0  # en ker (δe₄ = 0)
        delta_x = delta @ x
        harm_norm, exact_norm = p2._approximate_hodge_decomposition(
            delta, x, delta_x, _SVD_TOLERANCE_BASE,
        )
        x_norm_sq = float(np.dot(x, x))
        decomp_sq = harm_norm ** 2 + exact_norm ** 2
        assert math.isclose(decomp_sq, x_norm_sq, rel_tol=1e-6)

    # ── T2.3: _compute_poincare_constant ──────────────────────────────────────

    def test_poincare_constant_zero_energy(self, p2):
        r"""E(x) = 0 → C_P = 0 (no definida)."""
        result = p2._compute_poincare_constant(1.0, 0.0)
        assert result == 0.0

    def test_poincare_constant_known_value(self, p2):
        r"""‖x_exact‖ = 3, ‖δx‖ = √E = √9 = 3 → C_P = 1."""
        result = p2._compute_poincare_constant(3.0, 9.0)
        assert math.isclose(result, 1.0, rel_tol=1e-10)

    def test_poincare_constant_nonneg(self, p2):
        result = p2._compute_poincare_constant(2.0, 4.0)
        assert result >= 0.0

    def test_poincare_constant_zero_exact_norm(self, p2):
        r"""‖x_exact‖ = 0 → C_P = 0."""
        result = p2._compute_poincare_constant(0.0, 4.0)
        assert result == 0.0

    # ── T2.4: _audit_krylov_spectral_stability ────────────────────────────────

    def test_audit_passes_for_x_in_kernel(self, p2):
        r"""x ∈ ker(δ) → E(x) = 0 → pasa la auditoría."""
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        veto = _make_valid_veto_cert(3, 4)
        # Usamos el delta con kernel conocido
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        assert isinstance(cert, KrylovSpectralData)
        assert cert.is_frustration_bounded is True
        assert math.isclose(cert.dirichlet_energy, 0.0, abs_tol=1e-14)

    def test_audit_passes_for_x_near_kernel(self, p2):
        r"""x ≈ ker(δ) con pequeña frustración → pasa."""
        delta, x, _ = _build_valid_governance_inputs()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        assert cert.is_frustration_bounded is True

    def test_audit_fails_for_large_frustration(self, p2):
        r"""x alejado de ker(δ) → E(x) > ε_frust → DirichletFrustrationError."""
        delta = _full_rank_delta(3, 5)
        # x con ‖δx‖ grande: x = primer vector singular derecho de δ
        _, _, Vt = la.svd(delta, full_matrices=False)
        x = Vt[0] * 10.0  # componente en la dirección de mayor σ
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        with pytest.raises(DirichletFrustrationError):
            p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)

    def test_audit_computes_correct_dirichlet_energy(self, p2):
        r"""E(x) = ‖δx‖₂² debe ser correcto."""
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        expected = float(np.dot(delta @ x, delta @ x))
        assert math.isclose(cert.dirichlet_energy, expected, rel_tol=1e-12)

    def test_audit_rejects_wrong_x_dimension(self, p2):
        r"""x con dimensión incorrecta → ValueError."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(7)  # dim C⁰ esperada = 5
        with pytest.raises(ValueError, match="dim"):
            p2._audit_krylov_spectral_stability(delta, x)

    def test_audit_rejects_incoherent_veto_cert(self, p2):
        r"""Certificado con is_topologically_coherent=False → error."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(5)
        bad_veto = _make_invalid_veto_cert()
        with pytest.raises(TopologicalBifurcationError):
            p2._audit_krylov_spectral_stability(delta, x, veto_audit=bad_veto)

    def test_audit_rejects_veto_with_h1_positive(self, p2):
        r"""Certificado con h1_dimension > 0 → error."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(5)
        bad_veto = CohomologicalVetoData(
            dim_C0=5, dim_C1=3, delta_rank=2, h1_dimension=1,
            svd_tolerance=1e-10,
            max_singular_value=5.0, min_nonzero_singular_value=1.0,
            spectral_gap=4.0, spectral_gap_ratio=0.8,
            cohomological_stability_index=0.67,
            euler_characteristic_01=2, whitehead_torsion=0.0,
            poincare_lefschetz_ok=True,
            is_topologically_coherent=True,
        )
        with pytest.raises(TopologicalBifurcationError):
            p2._audit_krylov_spectral_stability(delta, x, veto_audit=bad_veto)

    def test_audit_rejects_dimension_mismatch_in_cert(self, p2):
        r"""Certificado con dimensiones diferentes a δ actual → ValueError."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(5)
        wrong_veto = CohomologicalVetoData(
            dim_C0=4, dim_C1=3, delta_rank=3, h1_dimension=0,
            svd_tolerance=1e-10,
            max_singular_value=5.0, min_nonzero_singular_value=1.0,
            spectral_gap=4.0, spectral_gap_ratio=0.8,
            cohomological_stability_index=1.0,
            euler_characteristic_01=1, whitehead_torsion=0.0,
            poincare_lefschetz_ok=True,
            is_topologically_coherent=True,
        )
        with pytest.raises(ValueError, match="Inconsistencia"):
            p2._audit_krylov_spectral_stability(delta, x, veto_audit=wrong_veto)

    def test_audit_executes_phase1_internally_if_no_cert(self, p2):
        r"""Sin veto_audit provisto, ejecuta Fase 1 internamente."""
        delta, x, _ = _build_valid_governance_inputs()
        cert = p2._audit_krylov_spectral_stability(delta, x)
        assert cert.is_frustration_bounded is True

    def test_audit_energy_norm_consistent_with_energy(self, p2):
        r"""Ê = E(x)/max(1,‖x‖²) debe ser consistente."""
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        x_norm_sq = float(np.dot(x, x))
        expected_norm = cert.dirichlet_energy / max(1.0, x_norm_sq)
        assert math.isclose(cert.dirichlet_energy_norm, expected_norm, rel_tol=1e-12)

    def test_audit_frustration_index_consistent(self, p2):
        r"""ρ = E(x)/ε_frust debe ser consistente."""
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        if cert.frustration_tolerance > 0:
            expected_rho = cert.dirichlet_energy / cert.frustration_tolerance
            assert math.isclose(cert.frustration_index, expected_rho, rel_tol=1e-10)

    def test_audit_condition_numbers_positive_finite(self, p2):
        delta, x, _ = _build_valid_governance_inputs()
        cert = p2._audit_krylov_spectral_stability(delta, x)
        assert math.isfinite(cert.delta_condition_number)
        assert math.isfinite(cert.laplacian_condition_number)
        assert cert.delta_condition_number >= 1.0
        assert cert.laplacian_condition_number >= 1.0

    def test_audit_laplacian_kappa_equals_delta_kappa_squared(self, p2):
        r"""κ(L) = κ(δ)² debe ser consistente."""
        delta, x, _ = _build_valid_governance_inputs()
        cert = p2._audit_krylov_spectral_stability(delta, x)
        expected_kL = cert.delta_condition_number ** 2
        assert math.isclose(
            cert.laplacian_condition_number, expected_kL, rel_tol=1e-8
        )

    def test_audit_poincare_constant_nonneg(self, p2):
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        assert cert.poincare_constant >= 0.0

    def test_audit_spectral_gap_matches_phase1(self, p2):
        r"""El gap espectral propagado desde Fase 1 debe coincidir."""
        delta, x, _ = _build_valid_governance_inputs()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        assert math.isclose(
            cert.spectral_gap_effective, veto.spectral_gap, rel_tol=1e-12
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §T3 — TESTS DE FASE 3: PROYECCIÓN DE HODGE ISOPERIMÉTRICA
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3_HodgeProjection:
    r"""
    Verifica Phase3_IsoperimetricHodgeProjector.

    Estructura:
        T3.1  → _verify_lipschitz_strong
        T3.2  → _estimate_cheeger_bound
        T3.3  → _compute_morse_reduction_index
        T3.4  → _enforce_isoperimetric_hodge_projection (método principal)
    """

    @pytest.fixture
    def p3(self) -> Phase3_IsoperimetricHodgeProjector:
        return Phase3_IsoperimetricHodgeProjector()

    @pytest.fixture
    def valid_spectral(self) -> KrylovSpectralData:
        return _make_valid_spectral_cert()

    # ── T3.1: _verify_lipschitz_strong ────────────────────────────────────────

    def test_lipschitz_satisfied_parallel_case(self, p3):
        r"""
        Si x* = x (sin desplazamiento): δx* − δx = 0, residuo = −κ·0 = 0 ≤ 0.
        """
        delta_x0 = np.array([1.0, 2.0, 3.0])
        delta_x1 = np.array([1.0, 2.0, 3.0])  # igual
        disp_norm = 0.0
        residual, ok = p3._verify_lipschitz_strong(
            delta_x0, delta_x1, disp_norm, 10.0,
            vector_norm_fn=p3._vector_norm,
        )
        assert ok is True
        assert residual <= 0.0 + _LIPSCHITZ_SLACK * max(1.0, 10.0 * disp_norm)

    def test_lipschitz_satisfied_contraction(self, p3):
        r"""
        Si ‖δx* − δx‖ << κ(δ)·‖x* − x‖: condición satisfecha.
        """
        delta_x0 = np.array([1.0, 0.0, 0.0])
        delta_x1 = np.array([0.0, 0.0, 0.0])  # δx* = 0 (x* en ker δ)
        # ‖δx* − δx‖ = ‖δx‖ = 1
        # ‖x* − x‖ = 1, κ = 10 → rhs = 10 >> 1 = lhs
        disp_norm = 1.0
        residual, ok = p3._verify_lipschitz_strong(
            delta_x0, delta_x1, disp_norm, 10.0,
            vector_norm_fn=p3._vector_norm,
        )
        assert ok is True
        assert residual < 0.0

    def test_lipschitz_violated(self, p3):
        r"""
        Si ‖δx* − δx‖ >> κ(δ)·‖x* − x‖: condición violada.
        """
        delta_x0 = np.array([1.0, 0.0, 0.0])
        delta_x1 = np.array([100.0, 0.0, 0.0])  # diferencia enorme
        disp_norm = 0.001  # desplazamiento mínimo
        residual, ok = p3._verify_lipschitz_strong(
            delta_x0, delta_x1, disp_norm, 1.0,  # κ = 1
            vector_norm_fn=p3._vector_norm,
        )
        # lhs ≈ 99 >> rhs = 1 * 0.001 = 0.001
        assert ok is False
        assert residual > 0.0

    # ── T3.2: _estimate_cheeger_bound ─────────────────────────────────────────

    def test_cheeger_zero_projected_energy(self, p3):
        r"""E(x*) = 0 → h(G) = 0."""
        result = p3._estimate_cheeger_bound(0.0, 5.0)
        assert result == 0.0

    def test_cheeger_zero_projected_norm(self, p3):
        r"""‖x*‖ = 0 → h(G) = 0."""
        result = p3._estimate_cheeger_bound(1.0, 0.0)
        assert result == 0.0

    def test_cheeger_known_value(self, p3):
        r"""E(x*) = 4, ‖x*‖ = 2 → h = 4/4 = 1."""
        result = p3._estimate_cheeger_bound(4.0, 2.0)
        assert math.isclose(result, 1.0, rel_tol=1e-12)

    def test_cheeger_nonneg(self, p3):
        result = p3._estimate_cheeger_bound(2.0, 3.0)
        assert result >= 0.0

    # ── T3.3: _compute_morse_reduction_index ──────────────────────────────────

    def test_morse_index_identity(self, p3):
        r"""δ = I_n: ker(δ) = {0} → ι_M = 0."""
        delta = np.eye(4)
        index = p3._compute_morse_reduction_index(delta, _SVD_TOLERANCE_BASE)
        assert index == 0

    def test_morse_index_with_kernel(self, p3):
        r"""δ ∈ ℝ^{n×(n+1)} con rank n: dim ker = 1 → ι_M = 1."""
        delta, _ = _delta_with_known_kernel(3)
        index = p3._compute_morse_reduction_index(delta, _SVD_TOLERANCE_BASE)
        assert index == 1

    def test_morse_index_nonneg(self, p3):
        delta = _full_rank_delta(3, 5)
        index = p3._compute_morse_reduction_index(delta, _SVD_TOLERANCE_BASE)
        assert index >= 0

    def test_morse_index_empty_matrix(self, p3):
        delta = np.zeros((0, 3))
        index = p3._compute_morse_reduction_index(delta, _SVD_TOLERANCE_BASE)
        assert index == 3  # dim ker = dim C⁰ = 3

    # ── T3.4: _enforce_isoperimetric_hodge_projection ─────────────────────────

    def test_enforce_passes_valid_projection(self, p3, valid_spectral):
        r"""Proyección válida con x* ∈ ker(δ) debe pasar."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star,
            spectral_audit=valid_spectral,
            coboundary_operator_delta=delta,
        )
        assert isinstance(cert, HodgeProjectionData)
        assert cert.is_isoperimetrically_bounded is True
        assert cert.is_energy_non_increasing is True

    def test_enforce_projection_distance_correct(self, p3):
        r"""‖x − x*‖₂ debe ser correcto."""
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        x_star = np.array([1.0, 2.0, 2.0, 0.0, 0.0])
        cert = p3._enforce_isoperimetric_hodge_projection(x, x_star)
        expected = float(la.norm(x - x_star))
        assert math.isclose(cert.projection_distance, expected, rel_tol=1e-12)

    def test_enforce_rejects_large_displacement(self, p3):
        r"""‖x − x*‖ > Δ_inertia → HomologicalInconsistencyError."""
        x = np.zeros(4)
        x_star = np.ones(4) * (_INERTIA_DELTA_MAX + 1.0)
        with pytest.raises(HomologicalInconsistencyError):
            p3._enforce_isoperimetric_hodge_projection(x, x_star)

    def test_enforce_rejects_dimension_mismatch(self, p3):
        r"""x y x* con dimensiones distintas → ValueError."""
        x = np.zeros(4)
        x_star = np.zeros(5)
        with pytest.raises(ValueError):
            p3._enforce_isoperimetric_hodge_projection(x, x_star)

    def test_enforce_rejects_bad_spectral_cert(self, p3):
        r"""Certificado con is_spectrally_stable=False → error."""
        x = np.zeros(4)
        x_star = np.zeros(4)
        bad_cert = _make_invalid_spectral_cert()
        with pytest.raises(SpectralComputationError):
            p3._enforce_isoperimetric_hodge_projection(
                x, x_star, spectral_audit=bad_cert,
            )

    def test_enforce_rejects_energy_increasing(self, p3):
        r"""
        Si E(x*) > E(x): proyección incrementa energía → HomologicalInconsistencyError.
        """
        delta, kernel_vec = _delta_with_known_kernel(3)
        x = kernel_vec.copy()  # x ∈ ker(δ): E(x) = 0
        # x_star fuera del kernel: E(x_star) > 0
        x_star = np.zeros(4)
        x_star[0] = 0.0001  # δ e₀ = e₀ → E(x_star) > 0

        p1 = Phase1_CohomologicalVetoCertifier()
        p2 = Phase2_KrylovSpectralAuditor()
        veto = p1._certify_cohomological_veto_axiom(delta)
        spectral = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)

        with pytest.raises(HomologicalInconsistencyError):
            p3._enforce_isoperimetric_hodge_projection(
                x, x_star,
                spectral_audit=spectral,
                coboundary_operator_delta=delta,
            )

    def test_enforce_rejects_delta_dim_mismatch(self, p3):
        r"""δ con dim C⁰ ≠ dim(x) → ValueError."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(4)  # incorrecto: debe ser dim 5
        x_star = np.zeros(4)
        with pytest.raises(ValueError):
            p3._enforce_isoperimetric_hodge_projection(
                x, x_star, coboundary_operator_delta=delta,
            )

    def test_enforce_without_delta_verified_false(self, p3):
        r"""Sin operador δ: verified_by_delta=False."""
        x = np.zeros(4)
        x_star = np.zeros(4)
        cert = p3._enforce_isoperimetric_hodge_projection(x, x_star)
        assert cert.verified_by_delta is False

    def test_enforce_with_delta_verified_true(self, p3):
        r"""Con operador δ: verified_by_delta=True."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.verified_by_delta is True

    def test_enforce_relative_distance_consistent(self, p3):
        r"""‖x − x*‖/max(1,‖x‖) debe ser consistente."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        expected = cert.projection_distance / max(1.0, float(la.norm(x)))
        assert math.isclose(
            cert.relative_projection_distance, expected, rel_tol=1e-12
        )

    def test_enforce_energy_reduction_ratio_in_unit_interval(self, p3):
        r"""0 ≤ E(x*)/E(x) ≤ 1 para proyección válida."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        if cert.original_dirichlet_energy > 1e-15:
            assert 0.0 <= cert.energy_reduction_ratio <= 1.0 + 1e-10

    def test_enforce_cheeger_nonneg(self, p3):
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.cheeger_bound_estimate >= 0.0

    def test_enforce_morse_index_nonneg(self, p3):
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.morse_reduction_index >= 0

    def test_enforce_lipschitz_satisfied_for_valid_projection(self, p3):
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.lipschitz_satisfied is True

    def test_enforce_minimal_norm_satisfied_for_valid_projection(self, p3):
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.minimal_norm_satisfied is True

    def test_enforce_inertia_delta_max_constant(self, p3):
        x = np.zeros(4)
        x_star = np.zeros(4)
        cert = p3._enforce_isoperimetric_hodge_projection(x, x_star)
        assert cert.inertia_delta_max == _INERTIA_DELTA_MAX

    def test_enforce_identical_x_and_xstar_passes(self, p3):
        r"""x = x* → distancia = 0, siempre pasa."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cert = p3._enforce_isoperimetric_hodge_projection(x, x.copy())
        assert cert.projection_distance == 0.0
        assert cert.is_isoperimetrically_bounded is True


# ═══════════════════════════════════════════════════════════════════════════════
# §T4 — TESTS DEL ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════════
class TestOrchestrator:
    r"""
    Verifica SheafCohomologyOrchestratorAgent como Z_SheafAgent = Φ₃∘Φ₂∘Φ₁.
    """

    @pytest.fixture
    def agent(self) -> SheafCohomologyOrchestratorAgent:
        return SheafCohomologyOrchestratorAgent(strict_mode=True)

    def test_governance_passes_valid_inputs(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert isinstance(state, SheafGovernanceState)
        assert state.is_epistemologically_valid is True

    def test_governance_returns_all_certificates(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert isinstance(state.veto_audit, CohomologicalVetoData)
        assert isinstance(state.spectral_audit, KrylovSpectralData)
        assert isinstance(state.hodge_audit, HodgeProjectionData)

    def test_governance_returns_provenance(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert isinstance(state.provenance, SheafAuditProvenance)

    def test_governance_provenance_all_phases_passed(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert state.provenance.phase1_passed is True
        assert state.provenance.phase2_passed is True
        assert state.provenance.phase3_passed is True

    def test_governance_provenance_timestamp_iso8601(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        ts = state.provenance.timestamp_iso
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_governance_provenance_checksum_sha256(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        chk = state.provenance.input_checksum_sha256
        assert len(chk) == 64
        int(chk, 16)

    def test_governance_provenance_functor_chain_contains_phases(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        chain = state.provenance.functor_chain
        assert "Φ₁" in chain
        assert "Φ₂" in chain
        assert "Φ₃" in chain
        assert "✓" in chain

    def test_governance_fails_phase1_rank_deficient(self, agent):
        r"""δ con dim H¹ > 0 → TopologicalBifurcationError."""
        delta = _rank_deficient_delta(4, 6, deficiency=1)
        x = np.zeros(6)
        x_star = np.zeros(6)
        with pytest.raises(TopologicalBifurcationError):
            agent.execute_sheaf_cohomology_governance(delta, x, x_star)

    def test_governance_fails_phase2_large_frustration(self, agent):
        r"""x con E(x) >> ε_frust → DirichletFrustrationError."""
        delta = _full_rank_delta(3, 5)
        _, _, Vt = la.svd(delta, full_matrices=False)
        x = Vt[0] * 10.0  # alta frustración
        x_star = np.zeros(5)
        with pytest.raises(DirichletFrustrationError):
            agent.execute_sheaf_cohomology_governance(delta, x, x_star)

    def test_governance_fails_phase3_large_displacement(self, agent):
        r"""‖x − x*‖ > Δ_inertia → HomologicalInconsistencyError."""
        delta, x, _ = _build_valid_governance_inputs()
        x_star_far = x + np.ones(x.shape) * (_INERTIA_DELTA_MAX + 1.0)
        with pytest.raises(HomologicalInconsistencyError):
            agent.execute_sheaf_cohomology_governance(delta, x, x_star_far)

    def test_governance_deterministic(self, agent):
        r"""Dos ejecuciones con la misma entrada producen resultados idénticos."""
        delta, x, x_star = _build_valid_governance_inputs()
        s1 = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        s2 = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert s1.veto_audit == s2.veto_audit
        assert s1.spectral_audit == s2.spectral_audit
        assert s1.hodge_audit == s2.hodge_audit

    def test_governance_checksum_same_for_same_input(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        s1 = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        s2 = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert (
            s1.provenance.input_checksum_sha256
            == s2.provenance.input_checksum_sha256
        )

    def test_governance_checksum_differs_for_different_inputs(self, agent):
        delta1, x1, xs1 = _build_valid_governance_inputs(m=3, n=5, seed=1)
        delta2, x2, xs2 = _build_valid_governance_inputs(m=3, n=5, seed=2)
        s1 = agent.execute_sheaf_cohomology_governance(delta1, x1, xs1)
        s2 = agent.execute_sheaf_cohomology_governance(delta2, x2, xs2)
        assert (
            s1.provenance.input_checksum_sha256
            != s2.provenance.input_checksum_sha256
        )

    def test_governance_strict_mode_attribute(self, agent):
        assert agent._strict_mode is True

    def test_governance_lenient_mode(self):
        agent_lenient = SheafCohomologyOrchestratorAgent(strict_mode=False)
        assert agent_lenient._strict_mode is False

    def test_governance_veto_audit_h1_is_zero(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert state.veto_audit.h1_dimension == 0

    def test_governance_spectral_audit_stable(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert state.spectral_audit.is_spectrally_stable is True
        assert state.spectral_audit.is_frustration_bounded is True

    def test_governance_hodge_audit_bounded(self, agent):
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert state.hodge_audit.is_isoperimetrically_bounded is True
        assert state.hodge_audit.is_energy_non_increasing is True

    def test_governance_logs_completion(self, agent, caplog):
        delta, x, x_star = _build_valid_governance_inputs()
        with caplog.at_level(logging.INFO):
            agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert len(caplog.records) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# §T5 — TESTS DE ANIDAMIENTO FUNTORIAL
# ═══════════════════════════════════════════════════════════════════════════════
class TestFunctorialChaining:
    r"""
    Verifica el anidamiento correcto entre fases:
    el certificado de Fase i es el objeto inicial de Fase i+1.
    """

    @pytest.fixture
    def p1(self) -> Phase1_CohomologicalVetoCertifier:
        return Phase1_CohomologicalVetoCertifier()

    @pytest.fixture
    def p2(self) -> Phase2_KrylovSpectralAuditor:
        return Phase2_KrylovSpectralAuditor()

    @pytest.fixture
    def p3(self) -> Phase3_IsoperimetricHodgeProjector:
        return Phase3_IsoperimetricHodgeProjector()

    def test_phase1_output_feeds_phase2(self, p1, p2):
        r"""El certificado de Fase 1 alimenta directamente Fase 2."""
        delta, x, _ = _build_valid_governance_inputs()
        cert1 = p1._certify_cohomological_veto_axiom(delta)
        cert2 = p2._audit_krylov_spectral_stability(delta, x, veto_audit=cert1)
        assert cert2.is_frustration_bounded is True

    def test_phase2_output_feeds_phase3(self, p2, p3):
        r"""El certificado de Fase 2 alimenta directamente Fase 3."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert2 = p2._audit_krylov_spectral_stability(delta, x)
        cert3 = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, spectral_audit=cert2,
        )
        assert cert3.is_isoperimetrically_bounded is True

    def test_full_chain_p1_p2_p3(self, p1, p2, p3):
        r"""La cadena completa Φ₁→Φ₂→Φ₃ produce resultados válidos."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert1 = p1._certify_cohomological_veto_axiom(delta)
        cert2 = p2._audit_krylov_spectral_stability(delta, x, veto_audit=cert1)
        cert3 = p3._enforce_isoperimetric_hodge_projection(
            x, x_star,
            spectral_audit=cert2,
            coboundary_operator_delta=delta,
        )
        assert cert3.is_isoperimetrically_bounded is True
        assert cert3.is_energy_non_increasing is True

    def test_phase2_rejects_incoherent_phase1_cert(self, p2):
        r"""Certificado con is_topologically_coherent=False → error en Fase 2."""
        delta, x, _ = _build_valid_governance_inputs()
        bad_cert = _make_invalid_veto_cert()
        with pytest.raises(TopologicalBifurcationError):
            p2._audit_krylov_spectral_stability(delta, x, veto_audit=bad_cert)

    def test_phase3_rejects_unstable_phase2_cert(self, p3):
        r"""Certificado con is_spectrally_stable=False → error en Fase 3."""
        x = np.zeros(4)
        x_star = np.zeros(4)
        bad_cert = _make_invalid_spectral_cert()
        with pytest.raises(SpectralComputationError):
            p3._enforce_isoperimetric_hodge_projection(
                x, x_star, spectral_audit=bad_cert,
            )

    def test_chain_dimensional_consistency_p1_p2(self, p1, p2):
        r"""Las dimensiones certificadas por Fase 1 deben coincidir con Fase 2."""
        delta, x, _ = _build_valid_governance_inputs(m=4, n=7)
        cert1 = p1._certify_cohomological_veto_axiom(delta)
        assert cert1.dim_C0 == 7
        assert cert1.dim_C1 == 4
        cert2 = p2._audit_krylov_spectral_stability(delta, x, veto_audit=cert1)
        assert cert2.is_frustration_bounded is True

    def test_energy_consistency_p2_p3(self, p1, p2, p3):
        r"""La energía E(x) debe ser consistente entre Fase 2 y Fase 3."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert1 = p1._certify_cohomological_veto_axiom(delta)
        cert2 = p2._audit_krylov_spectral_stability(delta, x, veto_audit=cert1)
        cert3 = p3._enforce_isoperimetric_hodge_projection(
            x, x_star,
            spectral_audit=cert2,
            coboundary_operator_delta=delta,
        )
        # La energía original en Fase 3 debe coincidir con la certificada en Fase 2
        assert math.isclose(
            cert3.original_dirichlet_energy,
            cert2.dirichlet_energy,
            rel_tol=1e-8,
            abs_tol=1e-14,
        )

    def test_inconsistent_energy_between_phases_raises(self, p1, p2, p3):
        r"""
        Si el certificado de Fase 2 tiene E(x) diferente al recalculado en Fase 3,
        debe lanzar HomologicalInconsistencyError.
        """
        delta, x, x_star = _build_valid_governance_inputs()
        cert2 = p2._audit_krylov_spectral_stability(delta, x)

        # Fabricamos un certificado con energía inconsistente
        bad_spectral = KrylovSpectralData(
            dirichlet_energy=cert2.dirichlet_energy + 999.0,  # incorrecto
            dirichlet_energy_norm=cert2.dirichlet_energy_norm,
            frustration_tolerance=cert2.frustration_tolerance,
            frustration_index=cert2.frustration_index,
            delta_condition_number=cert2.delta_condition_number,
            laplacian_condition_number=cert2.laplacian_condition_number,
            spectral_gap_effective=cert2.spectral_gap_effective,
            harmonic_component_norm=cert2.harmonic_component_norm,
            exact_component_norm=cert2.exact_component_norm,
            poincare_constant=cert2.poincare_constant,
            is_frustration_bounded=True,
            is_spectrally_stable=True,
            is_poincare_bounded=True,
        )
        with pytest.raises(HomologicalInconsistencyError, match="Inconsistencia"):
            p3._enforce_isoperimetric_hodge_projection(
                x, x_star,
                spectral_audit=bad_spectral,
                coboundary_operator_delta=delta,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# §T6 — TESTS DE JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    r"""
    Verifica el lattice de excepciones:

        SheafCohomologyAgentError
        ├── TopologicalBifurcationError
        │   └── PoincareLefschetzViolation
        ├── SpectralComputationError
        │   ├── SVDConvergenceError
        │   └── HodgeDecompositionError
        ├── DirichletFrustrationError
        │   └── PoincareBoundViolation
        └── HomologicalInconsistencyError
            ├── LipschitzViolation
            └── MinimalNormViolation
    """

    def test_sheaf_agent_error_is_base(self):
        exc = SheafCohomologyAgentError("test")
        assert isinstance(exc, SheafCohomologyAgentError)

    def test_topological_bifurcation_is_sheaf_error(self):
        exc = TopologicalBifurcationError("test")
        assert isinstance(exc, SheafCohomologyAgentError)

    def test_poincare_lefschetz_is_bifurcation(self):
        exc = PoincareLefschetzViolation("test")
        assert isinstance(exc, TopologicalBifurcationError)

    def test_spectral_computation_is_sheaf_error(self):
        exc = SpectralComputationError("test")
        assert isinstance(exc, SheafCohomologyAgentError)

    def test_svd_convergence_is_spectral(self):
        exc = SVDConvergenceError("test")
        assert isinstance(exc, SpectralComputationError)

    def test_hodge_decomposition_is_spectral(self):
        exc = HodgeDecompositionError("test")
        assert isinstance(exc, SpectralComputationError)

    def test_dirichlet_frustration_is_sheaf_error(self):
        exc = DirichletFrustrationError("test")
        assert isinstance(exc, SheafCohomologyAgentError)

    def test_poincare_bound_is_frustration(self):
        exc = PoincareBoundViolation("test")
        assert isinstance(exc, DirichletFrustrationError)

    def test_homological_inconsistency_is_sheaf_error(self):
        exc = HomologicalInconsistencyError("test")
        assert isinstance(exc, SheafCohomologyAgentError)

    def test_lipschitz_violation_is_inconsistency(self):
        exc = LipschitzViolation("test")
        assert isinstance(exc, HomologicalInconsistencyError)

    def test_minimal_norm_is_inconsistency(self):
        exc = MinimalNormViolation("test")
        assert isinstance(exc, HomologicalInconsistencyError)

    def test_all_leaves_catchable_as_root(self):
        leaves = [
            PoincareLefschetzViolation,
            SVDConvergenceError,
            HodgeDecompositionError,
            PoincareBoundViolation,
            LipschitzViolation,
            MinimalNormViolation,
        ]
        for ExcClass in leaves:
            try:
                raise ExcClass("test")
            except SheafCohomologyAgentError:
                pass  # ✓

    def test_exceptions_carry_message(self):
        msg = "Fractura cohomológica detectada en el haz celular."
        exc = TopologicalBifurcationError(msg)
        assert str(exc) == msg

    def test_bifurcation_catchable_as_spectral_not(self):
        r"""TopologicalBifurcationError NO es SpectralComputationError."""
        exc = TopologicalBifurcationError("test")
        assert not isinstance(exc, SpectralComputationError)

    def test_lipschitz_not_catchable_as_spectral(self):
        r"""LipschitzViolation NO es SpectralComputationError."""
        exc = LipschitzViolation("test")
        assert not isinstance(exc, SpectralComputationError)


# ═══════════════════════════════════════════════════════════════════════════════
# §T7 — TESTS DE INMUTABILIDAD DE DTOs
# ═══════════════════════════════════════════════════════════════════════════════
class TestDTOImmutability:
    r"""
    Verifica que todos los DTOs son frozen (inmutables) y tienen tipos correctos.
    """

    def _make_veto(self) -> CohomologicalVetoData:
        return _make_valid_veto_cert()

    def _make_spectral(self) -> KrylovSpectralData:
        return _make_valid_spectral_cert()

    def _make_hodge(self) -> HodgeProjectionData:
        delta, x, x_star = _build_valid_governance_inputs()
        p3 = Phase3_IsoperimetricHodgeProjector()
        return p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )

    def _make_provenance(self) -> SheafAuditProvenance:
        return SheafAuditProvenance(
            timestamp_iso="2025-01-01T00:00:00+00:00",
            input_checksum_sha256="a" * 64,
            phase1_passed=True,
            phase2_passed=True,
            phase3_passed=True,
            functor_chain="Φ₁✓→Φ₂✓→Φ₃✓",
        )

    # ── T7.1: CohomologicalVetoData ──────────────────────────────────────────

    def test_veto_cert_is_frozen(self):
        cert = self._make_veto()
        with pytest.raises((AttributeError, TypeError)):
            cert.h1_dimension = 99

    def test_veto_cert_field_types(self):
        cert = self._make_veto()
        assert isinstance(cert.dim_C0, int)
        assert isinstance(cert.dim_C1, int)
        assert isinstance(cert.delta_rank, int)
        assert isinstance(cert.h1_dimension, int)
        assert isinstance(cert.svd_tolerance, float)
        assert isinstance(cert.max_singular_value, float)
        assert isinstance(cert.min_nonzero_singular_value, float)
        assert isinstance(cert.spectral_gap, float)
        assert isinstance(cert.spectral_gap_ratio, float)
        assert isinstance(cert.cohomological_stability_index, float)
        assert isinstance(cert.euler_characteristic_01, int)
        assert isinstance(cert.whitehead_torsion, float)
        assert isinstance(cert.poincare_lefschetz_ok, bool)
        assert isinstance(cert.is_topologically_coherent, bool)

    # ── T7.2: KrylovSpectralData ──────────────────────────────────────────────

    def test_spectral_cert_is_frozen(self):
        cert = self._make_spectral()
        with pytest.raises((AttributeError, TypeError)):
            cert.is_spectrally_stable = False

    def test_spectral_cert_field_types(self):
        cert = self._make_spectral()
        assert isinstance(cert.dirichlet_energy, float)
        assert isinstance(cert.dirichlet_energy_norm, float)
        assert isinstance(cert.frustration_tolerance, float)
        assert isinstance(cert.frustration_index, float)
        assert isinstance(cert.delta_condition_number, float)
        assert isinstance(cert.laplacian_condition_number, float)
        assert isinstance(cert.spectral_gap_effective, float)
        assert isinstance(cert.harmonic_component_norm, float)
        assert isinstance(cert.exact_component_norm, float)
        assert isinstance(cert.poincare_constant, float)
        assert isinstance(cert.is_frustration_bounded, bool)
        assert isinstance(cert.is_spectrally_stable, bool)
        assert isinstance(cert.is_poincare_bounded, bool)

    # ── T7.3: HodgeProjectionData ─────────────────────────────────────────────

    def test_hodge_cert_is_frozen(self):
        cert = self._make_hodge()
        with pytest.raises((AttributeError, TypeError)):
            cert.is_isoperimetrically_bounded = False

    def test_hodge_cert_field_types(self):
        cert = self._make_hodge()
        assert isinstance(cert.projection_distance, float)
        assert isinstance(cert.relative_projection_distance, float)
        assert isinstance(cert.inertia_delta_max, float)
        assert isinstance(cert.original_dirichlet_energy, float)
        assert isinstance(cert.projected_dirichlet_energy, float)
        assert isinstance(cert.energy_reduction_ratio, float)
        assert isinstance(cert.lipschitz_residual, float)
        assert isinstance(cert.lipschitz_satisfied, bool)
        assert isinstance(cert.minimal_norm_satisfied, bool)
        assert isinstance(cert.cheeger_bound_estimate, float)
        assert isinstance(cert.morse_reduction_index, int)
        assert isinstance(cert.is_isoperimetrically_bounded, bool)
        assert isinstance(cert.is_energy_non_increasing, bool)
        assert isinstance(cert.verified_by_delta, bool)

    # ── T7.4: SheafAuditProvenance ────────────────────────────────────────────

    def test_provenance_is_frozen(self):
        prov = self._make_provenance()
        with pytest.raises((AttributeError, TypeError)):
            prov.phase1_passed = False

    def test_provenance_field_types(self):
        prov = self._make_provenance()
        assert isinstance(prov.timestamp_iso, str)
        assert isinstance(prov.input_checksum_sha256, str)
        assert isinstance(prov.phase1_passed, bool)
        assert isinstance(prov.phase2_passed, bool)
        assert isinstance(prov.phase3_passed, bool)
        assert isinstance(prov.functor_chain, str)

    # ── T7.5: SheafGovernanceState ────────────────────────────────────────────

    def test_governance_state_is_frozen(self):
        agent = SheafCohomologyOrchestratorAgent()
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        with pytest.raises((AttributeError, TypeError)):
            state.is_epistemologically_valid = False

    def test_governance_state_field_types(self):
        agent = SheafCohomologyOrchestratorAgent()
        delta, x, x_star = _build_valid_governance_inputs()
        state = agent.execute_sheaf_cohomology_governance(delta, x, x_star)
        assert isinstance(state.veto_audit, CohomologicalVetoData)
        assert isinstance(state.spectral_audit, KrylovSpectralData)
        assert isinstance(state.hodge_audit, HodgeProjectionData)
        assert isinstance(state.provenance, SheafAuditProvenance)
        assert isinstance(state.is_epistemologically_valid, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# §T8 — TESTS DE CASOS LÍMITE NUMÉRICOS
# ═══════════════════════════════════════════════════════════════════════════════
class TestNumericalEdgeCases:
    r"""
    Verifica el comportamiento ante condiciones numéricas extremas.
    """

    @pytest.fixture
    def p1(self) -> Phase1_CohomologicalVetoCertifier:
        return Phase1_CohomologicalVetoCertifier()

    @pytest.fixture
    def p2(self) -> Phase2_KrylovSpectralAuditor:
        return Phase2_KrylovSpectralAuditor()

    @pytest.fixture
    def p3(self) -> Phase3_IsoperimetricHodgeProjector:
        return Phase3_IsoperimetricHodgeProjector()

    def test_phase1_single_entry_delta(self, p1):
        r"""δ ∈ ℝ^{1×1}: caso mínimo."""
        delta = np.array([[2.5]], dtype=np.float64)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0
        assert cert.delta_rank == 1

    def test_phase1_delta_with_large_sigma(self, p1):
        r"""δ con valores singulares grandes: σ_max = 1e6."""
        delta = np.diag([1e6, 1e5, 1e4]).astype(np.float64)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0
        assert math.isclose(cert.max_singular_value, 1e6, rel_tol=1e-6)

    def test_phase1_well_conditioned_delta(self, p1):
        r"""δ bien condicionada: κ ≈ 1."""
        delta = np.eye(5, dtype=np.float64)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isclose(cert.max_singular_value, 1.0, rel_tol=1e-10)
        assert math.isclose(cert.min_nonzero_singular_value, 1.0, rel_tol=1e-10)

    def test_phase1_tall_matrix(self, p1):
        r"""δ ∈ ℝ^{10×5}: m > n, rango máximo = 5."""
        rng = np.random.default_rng(seed=333)
        Q, _ = la.qr(rng.standard_normal((10, 10)))
        delta = Q[:, :5] * 3.0
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.delta_rank == 5
        assert cert.h1_dimension == 5  # dim C¹ − rank = 10 − 5 = 5
        # Espera lanzar excepción porque h1 > 0
    # Corrección: no debe llamarse para δ tall con H¹ > 0

    def test_phase1_wide_matrix_h1_zero(self, p1):
        r"""δ ∈ ℝ^{3×8}: m < n, rango = m = 3 → H¹ = 0."""
        delta = _full_rank_delta(3, 8)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert cert.h1_dimension == 0
        assert cert.delta_rank == 3

    def test_phase2_x_all_zeros_passes(self, p2):
        r"""x = 0 → E(x) = 0 → pasa (frustración = 0)."""
        delta = _full_rank_delta(3, 5)
        x = np.zeros(5)
        cert = p2._audit_krylov_spectral_stability(delta, x)
        assert cert.dirichlet_energy == 0.0

    def test_phase2_tiny_x_passes(self, p2):
        r"""x muy pequeño → E(x) ≈ 0 → pasa."""
        delta, x, _ = _build_valid_governance_inputs()
        x_tiny = x * 1e-15
        cert = p2._audit_krylov_spectral_stability(delta, x_tiny)
        assert cert.is_frustration_bounded is True

    def test_phase3_x_equals_x_star_passes(self, p3):
        r"""x = x* → distancia = 0, siempre pasa."""
        x = np.array([0.5, 0.5, 0.5])
        cert = p3._enforce_isoperimetric_hodge_projection(x, x.copy())
        assert cert.projection_distance == 0.0

    def test_phase3_minimal_displacement(self, p3):
        r"""Desplazamiento ε_maq → pasa."""
        x = np.zeros(5)
        x_star = x.copy()
        x_star[0] = _MACHINE_EPSILON
        cert = p3._enforce_isoperimetric_hodge_projection(x, x_star)
        assert cert.is_isoperimetrically_bounded is True

    def test_phase3_displacement_at_boundary(self, p3):
        r"""Desplazamiento justo en Δ_inertia debe pasar."""
        x = np.zeros(5)
        x_star = np.zeros(5)
        x_star[0] = _INERTIA_DELTA_MAX  # exactamente en el límite
        cert = p3._enforce_isoperimetric_hodge_projection(x, x_star)
        assert cert.is_isoperimetrically_bounded is True

    def test_phase1_clear_spectral_gap_detected(self, p1):
        r"""δ con gap claro entre σ₃ y σ₄ debe tener gap_ratio > ρ_gap."""
        U, _ = la.qr(np.random.default_rng(0).standard_normal((5, 5)))
        Vt, _ = la.qr(np.random.default_rng(1).standard_normal((8, 8)))
        sigma_vals = np.array([100.0, 90.0, 80.0, 0.0001, 0.00005])
        Sigma = np.zeros((5, 8))
        for i, s in enumerate(sigma_vals):
            Sigma[i, i] = s
        delta = (U @ Sigma @ Vt[:8, :]).astype(np.float64)
        try:
            cert = p1._certify_cohomological_veto_axiom(delta)
            # H¹ puede ser 0 o > 0 dependiendo del gap
            # Lo importante: el gap_ratio puede ser > _SPECTRAL_GAP_MIN_RATIO
            assert math.isfinite(cert.spectral_gap_ratio)
        except TopologicalBifurcationError:
            pass  # También válido si H¹ > 0

    def test_safe_svdvals_large_matrix(self, p1):
        r"""SVD de matriz grande (50×80) no debe fallar."""
        rng = np.random.default_rng(seed=555)
        M = rng.standard_normal((50, 80))
        svs = p1._safe_svdvals(M, "M_large")
        assert svs.size == 50
        assert np.all(np.isfinite(svs))


# ═══════════════════════════════════════════════════════════════════════════════
# §T9 — TESTS DE TRAZABILIDAD Y CHECKSUMS
# ═══════════════════════════════════════════════════════════════════════════════
class TestProvenanceAndChecksum:
    r"""
    Verifica la trazabilidad criptográfica y el objeto SheafAuditProvenance.
    """

    @pytest.fixture
    def agent(self) -> SheafCohomologyOrchestratorAgent:
        return SheafCohomologyOrchestratorAgent()

    def _run(self, agent, seed=42):
        delta, x, x_star = _build_valid_governance_inputs(seed=seed)
        return agent.execute_sheaf_cohomology_governance(delta, x, x_star)

    def test_provenance_timestamp_not_empty(self, agent):
        state = self._run(agent)
        assert len(state.provenance.timestamp_iso) > 0

    def test_provenance_timestamp_has_timezone(self, agent):
        state = self._run(agent)
        ts = state.provenance.timestamp_iso
        assert "+" in ts or "Z" in ts

    def test_provenance_checksum_64_hex(self, agent):
        state = self._run(agent)
        chk = state.provenance.input_checksum_sha256
        assert len(chk) == 64
        int(chk, 16)

    def test_provenance_functor_chain_check_marks(self, agent):
        state = self._run(agent)
        assert "✓" in state.provenance.functor_chain

    def test_provenance_functor_chain_no_fail_marks(self, agent):
        state = self._run(agent)
        assert "✗" not in state.provenance.functor_chain

    def test_build_provenance_failed_phase(self, agent):
        prov = agent._build_provenance(
            checksum="a" * 64,
            phase1_passed=True,
            phase2_passed=False,
            phase3_passed=False,
        )
        assert prov.phase2_passed is False
        assert "✗" in prov.functor_chain

    def test_build_provenance_all_failed(self, agent):
        prov = agent._build_provenance(
            checksum="b" * 64,
            phase1_passed=False,
            phase2_passed=False,
            phase3_passed=False,
        )
        assert prov.functor_chain.count("✗") >= 3

    def test_checksum_changes_with_different_delta(self, agent):
        delta1, x, x_star = _build_valid_governance_inputs(seed=1)
        delta2, x2, x_star2 = _build_valid_governance_inputs(seed=2)
        s1 = agent.execute_sheaf_cohomology_governance(delta1, x, x_star)
        s2 = agent.execute_sheaf_cohomology_governance(delta2, x2, x_star2)
        assert (
            s1.provenance.input_checksum_sha256
            != s2.provenance.input_checksum_sha256
        )

    def test_checksum_changes_with_different_x(self, agent):
        r"""Mismo δ pero distinto x → checksum diferente."""
        delta, x1, x_star1 = _build_valid_governance_inputs(seed=10)
        _, x2, x_star2 = _build_valid_governance_inputs(seed=11)
        s1 = agent.execute_sheaf_cohomology_governance(delta, x1, x_star1)
        s2 = agent.execute_sheaf_cohomology_governance(delta, x2, x_star2)
        # Si x1 ≠ x2, los checksums deben diferir
        if not np.allclose(x1, x2):
            assert (
                s1.provenance.input_checksum_sha256
                != s2.provenance.input_checksum_sha256
            )


# ═══════════════════════════════════════════════════════════════════════════════
# §T10 — TESTS DE INVARIANTES MATEMÁTICOS
# ═══════════════════════════════════════════════════════════════════════════════
class TestMathematicalInvariants:
    r"""
    Verifica propiedades algebraicas y topológicas fundamentales.
    Estas pruebas actúan como "teoremas computacionales".
    """

    @pytest.fixture
    def p1(self) -> Phase1_CohomologicalVetoCertifier:
        return Phase1_CohomologicalVetoCertifier()

    @pytest.fixture
    def p2(self) -> Phase2_KrylovSpectralAuditor:
        return Phase2_KrylovSpectralAuditor()

    @pytest.fixture
    def p3(self) -> Phase3_IsoperimetricHodgeProjector:
        return Phase3_IsoperimetricHodgeProjector()

    # ── T10.1: Rango-Nulidad ──────────────────────────────────────────────────

    def test_rank_nullity_theorem(self, p1):
        r"""rank(δ) + dim ker(δ) = dim C⁰."""
        for m, n in [(3, 5), (2, 4), (4, 4), (1, 3)]:
            delta = _full_rank_delta(m, n)
            cert = p1._certify_cohomological_veto_axiom(delta)
            dim_kernel = n - cert.delta_rank
            assert dim_kernel >= 0
            assert cert.delta_rank + dim_kernel == n

    # ── T10.2: H¹ = 0 implica integrabilidad ─────────────────────────────────

    def test_h1_zero_implies_topological_coherence(self, p1):
        r"""dim H¹ = 0 ↔ is_topologically_coherent = True."""
        for seed in [1, 2, 3, 7, 42]:
            delta = _full_rank_delta(3, 5, seed=seed)
            cert = p1._certify_cohomological_veto_axiom(delta)
            assert cert.h1_dimension == 0
            assert cert.is_topologically_coherent is True

    # ── T10.3: Euler-Poincaré ─────────────────────────────────────────────────

    def test_euler_characteristic_formula(self, p1):
        r"""χ₀₁ = dim C⁰ − dim C¹."""
        for m, n in [(3, 5), (2, 6), (4, 4), (1, 7)]:
            delta = _full_rank_delta(m, n)
            cert = p1._certify_cohomological_veto_axiom(delta)
            expected_chi = n - m
            assert cert.euler_characteristic_01 == expected_chi

    # ── T10.4: Energía de Dirichlet ≥ 0 ──────────────────────────────────────

    def test_dirichlet_energy_nonneg(self, p2):
        r"""E(x) = ‖δx‖₂² ≥ 0 siempre."""
        delta, x, _ = _build_valid_governance_inputs()
        cert = p2._audit_krylov_spectral_stability(delta, x)
        assert cert.dirichlet_energy >= 0.0

    def test_dirichlet_energy_zero_for_kernel_vector(self, p2):
        r"""x ∈ ker(δ) → E(x) = 0."""
        delta, kernel_vec = _delta_with_known_kernel(4)
        x = kernel_vec.copy()
        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)
        assert math.isclose(cert.dirichlet_energy, 0.0, abs_tol=1e-14)

    # ── T10.5: κ(L) = κ(δ)² ──────────────────────────────────────────────────

    def test_laplacian_kappa_is_delta_kappa_squared(self, p2):
        r"""κ(L) = κ(δ)² = (σ_max/σ_min)²."""
        for seed in [1, 5, 10]:
            delta, x, _ = _build_valid_governance_inputs(seed=seed)
            cert = p2._audit_krylov_spectral_stability(delta, x)
            kd = cert.delta_condition_number
            kl = cert.laplacian_condition_number
            assert math.isclose(kl, kd * kd, rel_tol=1e-8)

    # ── T10.6: Proyección de Hodge no incrementa energía ─────────────────────

    def test_hodge_projection_non_increasing_energy(self, p3):
        r"""E(x*) ≤ E(x) para toda proyección válida."""
        for seed in [1, 2, 3]:
            delta, x, x_star = _build_valid_governance_inputs(seed=seed)
            cert = p3._enforce_isoperimetric_hodge_projection(
                x, x_star, coboundary_operator_delta=delta,
            )
            assert cert.is_energy_non_increasing is True
            assert cert.projected_dirichlet_energy <= (
                cert.original_dirichlet_energy + 1e-10
            )

    # ── T10.7: Ortogonalidad aproximada de la descomposición de Hodge ─────────

    def test_hodge_decomposition_approximate_orthogonality(self, p2):
        r"""
        ‖x‖² ≈ ‖x_harm‖² + ‖x_exact‖²  (Pitágoras para la descomposición ortogonal).
        """
        delta, kernel_vec = _delta_with_known_kernel(3)
        # x con componentes en ker y fuera de ker
        x = np.zeros(4)
        x[0] = 1.0   # fuera de ker
        x[3] = 1.0   # en ker
        delta_x = delta @ x
        harm_norm, exact_norm = p2._approximate_hodge_decomposition(
            delta, x, delta_x, _SVD_TOLERANCE_BASE,
        )
        x_norm_sq = float(np.dot(x, x))
        decomp_sq = harm_norm ** 2 + exact_norm ** 2
        assert math.isclose(decomp_sq, x_norm_sq, rel_tol=1e-6)

    # ── T10.8: Índice de Morse + rango = dim C⁰ ──────────────────────────────

    def test_morse_plus_rank_equals_dim_C0(self, p1, p3):
        r"""ι_M + rank(δ) = dim C⁰ (Teorema de Rango-Nulidad)."""
        for m, n in [(3, 5), (2, 4), (4, 7)]:
            delta = _full_rank_delta(m, n)
            cert = p1._certify_cohomological_veto_axiom(delta)
            morse = p3._compute_morse_reduction_index(
                delta, cert.svd_tolerance,
            )
            assert cert.delta_rank + morse == n

    # ── T10.9: Cauchy-Schwarz para el producto de Cheeger ────────────────────

    def test_cheeger_bound_nonneg(self, p3):
        r"""h(G) ≥ 0."""
        delta, x, x_star = _build_valid_governance_inputs()
        cert = p3._enforce_isoperimetric_hodge_projection(
            x, x_star, coboundary_operator_delta=delta,
        )
        assert cert.cheeger_bound_estimate >= 0.0

    # ── T10.10: Estabilidad β ∈ [0, 1] ───────────────────────────────────────

    def test_stability_index_in_unit_interval(self, p1):
        r"""β = rank(δ)/dim C¹ ∈ [0, 1] siempre."""
        for m, n in [(1, 3), (2, 5), (4, 4), (3, 10)]:
            delta = _full_rank_delta(m, n)
            cert = p1._certify_cohomological_veto_axiom(delta)
            assert 0.0 <= cert.cohomological_stability_index <= 1.0

    # ── T10.11: Torsión de Whitehead es finita ────────────────────────────────

    def test_whitehead_torsion_finite_for_full_rank(self, p1):
        r"""log|τ_W| debe ser finito para δ de rango positivo."""
        delta = _full_rank_delta(3, 5)
        cert = p1._certify_cohomological_veto_axiom(delta)
        assert math.isfinite(cert.whitehead_torsion)

    # ── T10.12: Escalado de la norma de frustración ───────────────────────────

    def test_energy_scales_as_sigma_squared(self, p2):
        r"""
        Si x es el vector singular derecho dominante de δ con valor α:
        E(α·v₁) = α² · σ_max² = α² · E(v₁).
        """
        delta, kernel_vec = _delta_with_known_kernel(3)
        # Para δ=[[I₃|0]]: v₁ = e₀ → δv₁ = e₀ → E(v₁) = 1
        x = np.zeros(4)
        x[0] = 1e-4  # pequeño para pasar la tolerancia

        p1 = Phase1_CohomologicalVetoCertifier()
        veto = p1._certify_cohomological_veto_axiom(delta)
        cert1 = p2._audit_krylov_spectral_stability(delta, x, veto_audit=veto)

        x2 = x * 0.5
        cert2 = p2._audit_krylov_spectral_stability(delta, x2, veto_audit=veto)

        # E(0.5·x) = 0.25 · E(x)
        assert math.isclose(
            cert2.dirichlet_energy,
            0.25 * cert1.dirichlet_energy,
            rel_tol=1e-10,
        )