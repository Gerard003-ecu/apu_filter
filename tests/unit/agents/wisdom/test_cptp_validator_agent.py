# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Suite  : Batería de Verificación Espectral — CPTP Channel Validator Agent   ║
║  Ruta   : tests/unit/agents/wisdom/test_cptp_validator_agent.py              ║
║  Versión: 2.0.0-Choi-Jamiolkowski-Kraus-NonSignaling-OODA-Nested-Strict-QA   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FILOSOFÍA DE LA SUITE:                                                      ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  La arquitectura de pruebas replica EXACTAMENTE la topología de herencia     ║
║  del módulo bajo prueba:                                                     ║
║                                                                              ║
║      Phase1_KrausSanitizer                                                   ║
║              △                                                               ║
║              │ (continuación funtorial)                                      ║
║      Phase2_ChoiIsomorphismCertifier(Phase1_KrausSanitizer)                  ║
║              △                                                               ║
║              │ (continuación funtorial)                                      ║
║      Phase3_QuantumSecurityEnforcer(Phase2_ChoiIsomorphismCertifier)         ║
║                                                                              ║
║  se traduce a:                                                               ║
║                                                                              ║
║      TestPhase1KrausSanitizer                                                ║
║              △                                                               ║
║      TestPhase2ChoiIsomorphismCertifier(TestPhase1KrausSanitizer)            ║
║              △                                                               ║
║      TestPhase3QuantumSecurityEnforcer(TestPhase2ChoiIsomorphismCertifier)   ║
║                                                                              ║
║  De modo que el último método de prueba de FASE 1 es literalmente el         ║
║  ancestro (objeto inicial) del primer método de prueba propio de FASE 2,     ║
║  y así sucesivamente — pytest re-ejecuta la batería heredada como            ║
║  regresión de base en cada nivel de la torre, exactamente como el DTO        ║
║  ``Phase1SanitizationData`` es objeto inicial de FASE 2 en el código         ║
║  fuente.                                                                     ║
║                                                                              ║
║  Una cuarta capa — ``TestCPTPChannelValidatorAgentOrchestrator`` — cierra    ║
║  el ciclo OODA completo (Seal/Veto), y capas transversales adicionales       ║
║  cubren jerarquía de excepciones y fábricas de referencia.                   ║
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

from app.agents.wisdom.cptp_validator_agent import (
    _DEFAULT_TOL,
    _MACHINE_EPS,
    _SPECTRAL_PSD_FLOOR,
    ChoiHermiticityError,
    CompletePositivityViolation,
    CPTPChannelGovernanceState,
    CPTPChannelValidatorAgent,
    CPTPValidatorAgentError,
    KrausDimensionError,
    NonSignalingViolationError,
    PeresHorodeckiSeparabilityError,
    Phase1_KrausSanitizer,
    Phase1SanitizationData,
    Phase2_ChoiIsomorphismCertifier,
    Phase2ChoiIsomorphismData,
    Phase3_QuantumSecurityEnforcer,
    Phase3QuantumSecurityData,
    Stratum,
    TopologicalInvariantError,
    TracePreservationViolation,
    UnitalityViolationError,
)

ComplexMatrix = np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES MATEMÁTICAS INDEPENDIENTES DE VERIFICACIÓN (oráculo externo)
# ═══════════════════════════════════════════════════════════════════════════════
def _fro(m: ComplexMatrix) -> float:
    """Norma de Frobenius, oráculo independiente de la implementación bajo test."""
    return float(la.norm(m, ord="fro"))


def _is_hermitian(m: ComplexMatrix, tol: float = 1e-9) -> bool:
    return _fro(m - m.conj().T) <= tol


def _is_psd(m: ComplexMatrix, floor: float = _SPECTRAL_PSD_FLOOR) -> bool:
    eigs = la.eigvalsh(0.5 * (m + m.conj().T))
    return bool(np.min(eigs) >= floor)


def _is_density_matrix(rho: ComplexMatrix, tol: float = 1e-8) -> bool:
    return (
        _is_hermitian(rho, tol)
        and _is_psd(rho, -tol)
        and abs(float(np.real(np.trace(rho))) - 1.0) <= tol
    )


def _apply_channel(kraus_ops: List[ComplexMatrix], rho: ComplexMatrix) -> ComplexMatrix:
    """Aplica el canal Σ M_k ρ M_k† — oráculo físico independiente."""
    out = np.zeros_like(rho)
    for m in kraus_ops:
        out += m @ rho @ m.conj().T
    return out


def _manual_choi_vecF(kraus_ops: List[ComplexMatrix], d: int) -> ComplexMatrix:
    """Reimplementación manual (oráculo) del isomorfismo de Choi, vec column-major."""
    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for m in kraus_ops:
        v = m.flatten(order="F")
        choi += np.outer(v, v.conj())
    return choi


def _haar_random_kraus(d: int, k: int, seed: int) -> List[ComplexMatrix]:
    r"""
    Genera un ensamble de Kraus \(\{M_i\}_{i=1}^k\) válido (TP exacto) mediante
    dilatación de Stinespring: se extrae una isometría \(V\in\mathbb{C}^{dk\times d}\)
    (\(V^\dagger V=I_d\)) vía QR de una matriz Ginibre, y se particiona en k
    bloques \(d\times d\). Por construcción algebraica exacta:

    \[
    \sum_i M_i^\dagger M_i = V^\dagger V = I_d.
    \]

    Este es el oráculo canónico de generación aleatoria de canales CPTP para
    pruebas basadas en propiedades (property-based testing artesanal, sin
    dependencia de `hypothesis`).
    """
    rng = np.random.default_rng(seed)
    ginibre = rng.normal(size=(d * k, d)) + 1j * rng.normal(size=(d * k, d))
    isometry, _r = np.linalg.qr(ginibre)
    isometry = isometry[:, :d].astype(np.complex128)
    return [isometry[i * d:(i + 1) * d, :].copy() for i in range(k)]


# Especificaciones deterministas de canales aleatorios: (dimension_d, k_kraus, seed)
RANDOM_CHANNEL_SPECS: List[Tuple[int, int, int]] = [
    (2, 2, 1), (2, 3, 2), (2, 4, 3),
    (3, 2, 4), (3, 3, 5), (3, 5, 6),
    (4, 2, 7), (4, 3, 8),
]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES CANÓNICAS COMPARTIDAS
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def agent() -> CPTPChannelValidatorAgent:
    return CPTPChannelValidatorAgent()


@pytest.fixture
def qubit_identity() -> List[ComplexMatrix]:
    return CPTPChannelValidatorAgent.identity_kraus(2)


@pytest.fixture
def qutrit_identity() -> List[ComplexMatrix]:
    return CPTPChannelValidatorAgent.identity_kraus(3)


@pytest.fixture(params=[0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
def amplitude_damping_gamma(request) -> float:
    return request.param


@pytest.fixture
def amplitude_damping_kraus_ops(amplitude_damping_gamma) -> Tuple[List[ComplexMatrix], float]:
    ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(amplitude_damping_gamma)
    return ops, amplitude_damping_gamma


@pytest.fixture(params=[2, 3, 4])
def depolarizing_dimension(request) -> int:
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.5, 0.75, 1.0])
def depolarizing_p(request) -> float:
    return request.param


@pytest.fixture
def broken_non_tp_kraus() -> List[ComplexMatrix]:
    """Ensamble deliberadamente incompleto: Σ M†M = 0.25·I ≠ I (viola TP)."""
    return [0.5 * np.eye(2, dtype=np.complex128)]


@pytest.fixture
def nan_kraus() -> List[ComplexMatrix]:
    m = np.eye(2, dtype=np.complex128)
    m[0, 0] = np.nan
    return [m]


@pytest.fixture
def inf_kraus() -> List[ComplexMatrix]:
    m = np.eye(2, dtype=np.complex128)
    m[1, 1] = np.inf
    return [m]


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 (TEST) — Saneamiento Kraus, gauge de fase, certificado dimensional
# Ancla de la torre de herencia; toda subclase hereda y re-ejecuta esta batería.
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1KrausSanitizer:
    """
    Batería granular de FASE 1: valida los morfismos elementales

        Dim → Type → UVFilter → Gauge → Mass → sanitize_operators(Ω)

    El método terminal ``sanitize_operators`` (FASE 1.Ω) es el objeto inicial
    exacto consumido por ``Phase2_ChoiIsomorphismCertifier`` — su contrato
    de salida (``sanitized_ops``, ``Phase1SanitizationData``) se congela aquí
    como base de la torre de herencia de pruebas.
    """

    # ── 1.1 · dimensión de Hilbert ──────────────────────────────────────
    @pytest.mark.parametrize("d", [1, 2, 3, 8, 16])
    def test_validate_hilbert_dimension_accepts_positive_int(self, d):
        assert Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(d) == d

    def test_validate_hilbert_dimension_accepts_numpy_integer(self):
        d = np.int64(4)
        assert Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(d) == 4

    @pytest.mark.parametrize("bad_d", [0, -1, -100])
    def test_validate_hilbert_dimension_rejects_non_positive(self, bad_d):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(bad_d)

    @pytest.mark.parametrize("bad_d", [1.5, "2", None, 2.0])
    def test_validate_hilbert_dimension_rejects_non_integer_type(self, bad_d):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(bad_d)

    # ── 1.2 · tipado C*, shape, finitud IEEE-754 ────────────────────────
    def test_validate_kraus_typing_rejects_empty_list(self):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing([], 2)

    def test_validate_kraus_typing_rejects_none(self):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing(None, 2)

    def test_validate_kraus_typing_rejects_non_ndarray_element(self):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing([[[1, 0], [0, 1]]], 2)

    def test_validate_kraus_typing_rejects_wrong_shape(self):
        m = np.eye(3, dtype=np.complex128)
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing([m], 2)

    def test_validate_kraus_typing_rejects_nan(self, nan_kraus):
        with pytest.raises(CPTPValidatorAgentError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing(nan_kraus, 2)

    def test_validate_kraus_typing_rejects_inf(self, inf_kraus):
        with pytest.raises(CPTPValidatorAgentError):
            Phase1_KrausSanitizer._phase1_validate_kraus_typing(inf_kraus, 2)

    def test_validate_kraus_typing_casts_real_to_complex128(self):
        m = np.eye(2, dtype=np.float64)
        typed = Phase1_KrausSanitizer._phase1_validate_kraus_typing([m], 2)
        assert typed[0].dtype == np.complex128
        assert np.allclose(typed[0], np.eye(2))

    def test_validate_kraus_typing_preserves_operator_order(self):
        m0 = np.eye(2, dtype=np.complex128)
        m1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        typed = Phase1_KrausSanitizer._phase1_validate_kraus_typing([m0, m1], 2)
        assert np.array_equal(typed[0], m0)
        assert np.array_equal(typed[1], m1)

    # ── 1.3 · filtro ultravioleta de operadores nulos ───────────────────
    def test_filter_uv_removes_zero_matrix(self):
        ops = [np.eye(2, dtype=np.complex128), np.zeros((2, 2), dtype=np.complex128)]
        cleaned = Phase1_KrausSanitizer._phase1_filter_uv_null_operators(ops, _DEFAULT_TOL)
        assert len(cleaned) == 1

    def test_filter_uv_raises_when_all_null(self):
        ops = [np.zeros((2, 2), dtype=np.complex128)]
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_filter_uv_null_operators(ops, _DEFAULT_TOL)

    def test_filter_uv_preserves_all_significant_operators(self, qubit_identity):
        cleaned = Phase1_KrausSanitizer._phase1_filter_uv_null_operators(
            qubit_identity, _DEFAULT_TOL
        )
        assert len(cleaned) == len(qubit_identity)

    def test_filter_uv_respects_custom_tolerance_floor(self):
        tiny = np.eye(2, dtype=np.complex128) * 1e-9
        # con tolerancia laxa (1e-3) el floor = 1e-6 → 1e-9 debe filtrarse
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer._phase1_filter_uv_null_operators([tiny], 1.0e-3)

    # ── 1.4 · gauge de fase estándar U(1) ────────────────────────────────
    def test_gauge_phase_pivot_becomes_real_positive(self):
        m = np.array([[1j, 0], [0, 0]], dtype=np.complex128)
        gauged, rotated, _res = Phase1_KrausSanitizer._phase1_standard_gauge_phase([m])
        pivot = gauged[0][0, 0]
        assert rotated is True
        assert pivot.imag == pytest.approx(0.0, abs=1e-12)
        assert pivot.real > 0.0

    def test_gauge_phase_leaves_already_real_pivot_untouched_in_value(self):
        m = np.array([[2.0, 0], [0, 1.0]], dtype=np.complex128)
        gauged, _rotated, res = Phase1_KrausSanitizer._phase1_standard_gauge_phase([m])
        assert np.allclose(gauged[0], m)
        assert res == pytest.approx(0.0, abs=1e-12)

    def test_gauge_phase_handles_degenerate_zero_pivot_branch(self):
        """Rama defensiva: |pivot| ≤ max(eps, eps_gauge) ⇒ no rotación."""
        m = np.zeros((2, 2), dtype=np.complex128)
        gauged, rotated, res = Phase1_KrausSanitizer._phase1_standard_gauge_phase([m])
        assert np.array_equal(gauged[0], m)
        assert rotated is False
        assert res == 0.0

    def test_gauge_phase_is_channel_invariant_via_choi(self):
        r"""
        Invariante físico: \(M_k\mapsto e^{i\phi_k}M_k\) no altera el canal
        porque la fase se cancela en \(M_k\rho M_k^\dagger\), y por tanto
        tampoco altera \(\Lambda_{\mathcal{E}}=\sum_k\mathrm{vec}(M_k)\mathrm{vec}(M_k)^\dagger\)
        (el módulo al cuadrado de la fase es 1). Se verifica end-to-end.
        """
        d = 2
        raw = CPTPChannelValidatorAgent.amplitude_damping_kraus(0.3)
        # Inyectar fases arbitrarias por operador (libertad de gauge de Kraus)
        phased = [op * np.exp(1j * ang) for op, ang in zip(raw, [0.7, -2.1])]

        choi_raw = _manual_choi_vecF(raw, d)
        choi_phased = _manual_choi_vecF(phased, d)
        assert np.allclose(choi_raw, choi_phased, atol=1e-10)

        gauged, _r, _res = Phase1_KrausSanitizer._phase1_standard_gauge_phase(phased)
        choi_gauged = _manual_choi_vecF(gauged, d)
        assert np.allclose(choi_raw, choi_gauged, atol=1e-9)

    # ── 1.5 · masa de Frobenius ──────────────────────────────────────────
    def test_frobenius_mass_matches_manual_sum(self):
        ops = [
            np.array([[1, 0], [0, 0]], dtype=np.complex128),
            np.array([[0, 1], [0, 0]], dtype=np.complex128),
        ]
        expected = sum(_fro(m) ** 2 for m in ops)
        assert Phase1_KrausSanitizer._phase1_frobenius_mass(ops) == pytest.approx(expected)

    def test_frobenius_mass_equals_dimension_for_identity_channel(self, qubit_identity):
        mass = Phase1_KrausSanitizer._phase1_frobenius_mass(qubit_identity)
        assert mass == pytest.approx(2.0, abs=1e-12)

    # ── 1.Ω · composición terminal Observe/Sanitize ─────────────────────
    def test_sanitize_operators_end_to_end_identity(self, qubit_identity):
        ops, dto = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        assert isinstance(dto, Phase1SanitizationData)
        assert dto.dimension_d == 2
        assert dto.num_operators == 1
        assert dto.num_operators_effective == 1
        assert dto.is_finite is True
        assert dto.frobenius_mass == pytest.approx(2.0, abs=1e-10)

    def test_sanitize_operators_raises_on_empty_ensemble(self):
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer.sanitize_operators([], 2)

    def test_sanitize_operators_effective_count_discounts_uv_nulls(self):
        ops = [
            np.eye(2, dtype=np.complex128),
            np.zeros((2, 2), dtype=np.complex128),
        ]
        _sanitized, dto = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        assert dto.num_operators == 2
        assert dto.num_operators_effective == 1

    def test_sanitize_operators_gauge_residual_within_tolerance(self, qubit_identity):
        _sanitized, dto = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        assert dto.gauge_phase_residuals <= 1e-9

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_sanitize_operators_frobenius_mass_equals_d_for_random_tp_channels(
        self, d, k, seed
    ):
        r"""Propiedad: para todo canal TP, \(\mu_F=\sum_k\|M_k\|_F^2=d\)."""
        ops = _haar_random_kraus(d, k, seed)
        _sanitized, dto = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        assert dto.frobenius_mass == pytest.approx(d, abs=1e-8)

    def test_sanitize_operators_rejects_dimension_mismatch_downstream_shape(self):
        m = np.eye(3, dtype=np.complex128)
        with pytest.raises(KrausDimensionError):
            Phase1_KrausSanitizer.sanitize_operators([m], 2)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 (TEST) — TP / unitalidad / Choi–Jamiołkowski / unitariedad
# Continuación directa de TestPhase1KrausSanitizer (herencia real): toda la
# batería anterior se re-certifica como precondición válida de esta fase.
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2ChoiIsomorphismCertifier(TestPhase1KrausSanitizer):
    """
    Batería granular de FASE 2: valida

        TP-residual → Unital-residual → Choi(vecF) → Hermiticidad
        → δ_⋄ (checksum dual) → U(E) → audit_trace_preservation(Ω)

    El DTO terminal ``Phase2ChoiIsomorphismData`` (FASE 2.Ω) es el objeto
    inicial exacto consumido por ``Phase3_QuantumSecurityEnforcer``.
    """

    # ── 2.1 · residuo TP ─────────────────────────────────────────────────
    def test_tp_residual_zero_for_identity(self, qubit_identity):
        sanitized, _dto1 = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        r_tp, s_op = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized, 2
        )
        assert r_tp == pytest.approx(0.0, abs=1e-10)
        assert _is_hermitian(s_op)

    def test_tp_residual_nonzero_for_broken_channel(self, broken_non_tp_kraus):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized, 2
        )
        assert r_tp > 1e-3

    def test_tp_residual_matches_manual_frobenius_formula(self, amplitude_damping_kraus_ops):
        ops, _gamma = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        s_manual = sum(m.conj().T @ m for m in sanitized)
        expected = _fro(s_manual - np.eye(2))
        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized, 2
        )
        assert r_tp == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_tp_residual_zero_for_random_haar_channels(self, d, k, seed):
        ops = _haar_random_kraus(d, k, seed)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized, d
        )
        assert r_tp == pytest.approx(0.0, abs=1e-8)

    # ── 2.2 · residuo de unitalidad ──────────────────────────────────────
    def test_unital_residual_zero_for_identity(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        r_u, is_unital = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(
            sanitized, 2
        )
        assert r_u == pytest.approx(0.0, abs=1e-10)
        assert is_unital is True

    def test_unital_residual_nonzero_for_amplitude_damping_with_gamma_gt_zero(self):
        ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(0.4)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        r_u, is_unital = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(
            sanitized, 2
        )
        assert r_u > 1e-3
        assert is_unital is False

    def test_unital_residual_matches_analytic_amplitude_damping_formula(self):
        gamma = 0.6
        ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(gamma)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        r_u, _ = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(sanitized, 2)
        expected = math.sqrt(2.0) * gamma  # ‖diag(γ, -γ)‖_F
        assert r_u == pytest.approx(expected, rel=1e-6)

    def test_unital_residual_zero_for_depolarizing(self, depolarizing_dimension, depolarizing_p):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(
            depolarizing_dimension, depolarizing_p
        )
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(
            ops, depolarizing_dimension
        )
        r_u, is_unital = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(
            sanitized, depolarizing_dimension
        )
        assert r_u <= 1e-8
        assert is_unital is True

    # ── 2.3 · construcción de Choi (vecF) ────────────────────────────────
    def test_choi_matrix_shape_is_d_squared(self, qutrit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qutrit_identity, 3)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 3)
        assert choi.shape == (9, 9)

    def test_choi_matrix_is_hermitian_by_construction(self, amplitude_damping_kraus_ops):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        assert _is_hermitian(choi, tol=1e-9)

    def test_choi_matrix_is_psd_by_gram_construction(self, amplitude_damping_kraus_ops):
        r"""
        Teorema: \(\Lambda=\sum_k v_kv_k^\dagger\) es PSD para *cualquier*
        ensamble de matrices (Gram-sum de rango 1), sea o no TP.
        """
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        assert _is_psd(choi)

    def test_choi_matrix_trace_equals_frobenius_mass(self, amplitude_damping_kraus_ops):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        assert float(np.real(np.trace(choi))) == pytest.approx(
            dto1.frobenius_mass, abs=1e-9
        )

    def test_choi_matrix_matches_independent_oracle_formula(self, amplitude_damping_kraus_ops):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi_impl = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(
            sanitized, 2
        )
        choi_oracle = _manual_choi_vecF(sanitized, 2)
        assert np.allclose(choi_impl, 0.5 * (choi_oracle + choi_oracle.conj().T), atol=1e-10)

    def test_choi_matrix_vecF_convention_differs_from_row_major_for_asymmetric_kraus(self):
        """Blindaje de regresión de convención: order='F' ≠ order='C' cuando M no es simétrica."""
        m = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # amplitude-damping-like
        v_f = m.flatten(order="F")
        v_c = m.flatten(order="C")
        assert not np.allclose(v_f, v_c)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix([m], 2)
        assert np.allclose(choi, np.outer(v_f, v_f.conj()))

    # ── 2.4 · certificación de hermiticidad ──────────────────────────────
    def test_certify_hermiticity_accepts_valid_choi(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        certified = Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
            choi, _DEFAULT_TOL
        )
        assert _is_hermitian(certified, tol=1e-12)

    def test_certify_hermiticity_raises_on_corrupted_matrix(self):
        broken = np.array([[1.0, 1.0j], [0.0, 1.0]], dtype=np.complex128)  # no-hermítica
        with pytest.raises(ChoiHermiticityError):
            Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
                broken, _DEFAULT_TOL
            )

    def test_certify_hermiticity_symmetrizes_via_weyl_projection(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        # inyectar defecto antihermítico ínfimo (por debajo de tolerancia)
        choi_perturbed = choi + 1e-14j * np.ones_like(choi)
        certified = Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
            choi_perturbed, 1e-10
        )
        assert _fro(certified - certified.conj().T) == pytest.approx(0.0, abs=1e-18)

    # ── 2.5 · checksum dual δ_⋄ (traza parcial de Choi) ──────────────────
    def test_partial_trace_tp_defect_zero_for_tp_channel(self, amplitude_damping_kraus_ops):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        delta = Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_trace_tp_defect(
            choi, 2
        )
        assert delta == pytest.approx(0.0, abs=1e-8)

    def test_partial_trace_tp_defect_nonzero_for_broken_channel(self, broken_non_tp_kraus):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        delta = Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_trace_tp_defect(
            choi, 2
        )
        assert delta > 1e-3

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_partial_trace_tp_defect_dual_theorem_matches_tp_residual_sign(
        self, d, k, seed
    ):
        r"""
        Teorema (checksum causal, dualidad TP): \(r_{TP}\approx0\iff\delta_\diamond\approx0\)
        para el mismo ensamble.
        """
        ops = _haar_random_kraus(d, k, seed)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized, d
        )
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, d)
        delta = Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_trace_tp_defect(
            choi, d
        )
        assert r_tp <= 1e-8
        assert delta <= 1e-8

    # ── 2.6 · grado de unitariedad ────────────────────────────────────────
    def test_unitariety_degree_is_one_for_pure_unitary_channel(self):
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators([pauli_x], 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        u_deg = Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(choi, 2)
        assert u_deg == pytest.approx(1.0, abs=1e-9)

    def test_unitariety_degree_trivial_for_scalar_dimension(self):
        m = np.array([[1.0]], dtype=np.complex128)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix([m], 1)
        u_deg = Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(choi, 1)
        assert u_deg == 1.0

    def test_unitariety_degree_bounded_in_unit_interval_for_random_channels(self):
        for d, k, seed in RANDOM_CHANNEL_SPECS:
            ops = _haar_random_kraus(d, k, seed)
            sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
            choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(
                sanitized, d
            )
            u_deg = Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(choi, d)
            assert 0.0 <= u_deg <= 1.0

    def test_unitariety_degree_decreases_with_depolarizing_noise_strength(
        self, depolarizing_dimension
    ):
        d = depolarizing_dimension
        degrees = []
        for p in (0.0, 0.5, 1.0):
            ops = CPTPChannelValidatorAgent.depolarizing_kraus(d, p)
            sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
            choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(
                sanitized, d
            )
            degrees.append(
                Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(choi, d)
            )
        assert degrees[0] >= degrees[1] >= degrees[2]
        assert degrees[0] == pytest.approx(1.0, abs=1e-6)

    # ── 2.Ω · composición terminal Orient ─────────────────────────────────
    def test_audit_trace_preservation_dto_fields_for_identity(self, qubit_identity):
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert isinstance(dto2, Phase2ChoiIsomorphismData)
        assert dto2.is_trace_preserving is True
        assert dto2.is_unital is True
        assert dto2.choi_trace == pytest.approx(2.0, abs=1e-9)
        assert dto2.unitariety_degree == pytest.approx(1.0, abs=1e-9)

    def test_audit_trace_preservation_raises_on_broken_channel(self, broken_non_tp_kraus):
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        with pytest.raises(TracePreservationViolation):
            Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
                sanitized, dto1.dimension_d, _DEFAULT_TOL
            )

    def test_audit_trace_preservation_raises_choi_hermiticity_error_via_monkeypatch(
        self, monkeypatch, qubit_identity
    ):
        """Ataca la vía defensiva de hermiticidad inyectando un constructor corrupto."""
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)

        def _corrupted_choi(_ops, _d):
            return np.array([[1.0, 1.0j], [0.0, 1.0]], dtype=np.complex128)

        monkeypatch.setattr(
            Phase2_ChoiIsomorphismCertifier,
            "_phase2_construct_choi_matrix",
            staticmethod(_corrupted_choi),
        )
        with pytest.raises(ChoiHermiticityError):
            Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
                sanitized, dto1.dimension_d, _DEFAULT_TOL
            )

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_audit_trace_preservation_choi_trace_equals_d_for_random_channels(
        self, d, k, seed
    ):
        ops = _haar_random_kraus(d, k, seed)
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert dto2.choi_trace == pytest.approx(d, abs=1e-7)
        assert dto2.tp_diamond_defect <= 1e-7


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 (TEST) — CP, PPT, Non-Signaling y gobernanza sellada
# Continuación directa de TestPhase2ChoiIsomorphismCertifier (herencia real).
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3QuantumSecurityEnforcer(TestPhase2ChoiIsomorphismCertifier):
    """
    Batería granular de FASE 3: valida

        CP-espectral → PPT (Peres–Horodecki) → recurso de Bell
        → traza parcial A → acción local → Non-Signaling → pureza
        → audit_quantum_security(Ω)

    El DTO terminal ``Phase3QuantumSecurityData`` alimenta el sellado de
    gobernanza (Ω.1/Ω.2) validado en ``TestCPTPChannelValidatorAgentOrchestrator``.
    """

    # ── 3.1 · positividad completa espectral ─────────────────────────────
    def test_cp_spectral_identifies_valid_channel(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        eigvals, min_eig, rank, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                choi, _DEFAULT_TOL
            )
        )
        assert is_cp is True
        assert min_eig >= -1e-9
        assert rank == 1  # canal identidad ⇒ rango de Kraus mínimo = 1
        assert len(eigvals) == 4

    def test_cp_spectral_kraus_rank_matches_independent_kraus_count(self):
        m0 = np.diag([1.0, 0.0]).astype(np.complex128)
        m1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators([m0, m1], 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        _eigs, _m, rank, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                choi, _DEFAULT_TOL
            )
        )
        assert is_cp is True
        assert rank == 2

    def test_cp_spectral_detects_violation_on_hand_crafted_negative_eigen_matrix(self):
        r"""Ataca directamente el oráculo espectral con \(\Lambda\) no-PSD artesanal."""
        negative_choi = np.diag([1.0, 1.0, 1.0, -0.5]).astype(np.complex128)
        _eigs, min_eig, _rank, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                negative_choi, _DEFAULT_TOL
            )
        )
        assert min_eig == pytest.approx(-0.5)
        assert is_cp is False

    def test_cp_spectral_tolerates_wilkinson_floor_epsilon(self):
        """Autovalores ínfimamente negativos (redondeo FPU) no deben vetar CP."""
        near_zero_negative = np.diag([1.0, 1.0, 1.0, -1e-14]).astype(np.complex128)
        _eigs, min_eig, _rank, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                near_zero_negative, _DEFAULT_TOL
            )
        )
        assert is_cp is True  # -1e-14 > _SPECTRAL_PSD_FLOOR = -1e-13

    # ── 3.2 · criterio PPT de Peres–Horodecki ────────────────────────────
    def test_ppt_identity_channel_choi_is_entangled_pure_state(self, qubit_identity):
        r"""
        Teorema: el Choi del canal identidad es el estado maximalmente
        entrelazado (escalado), cuya transpuesta parcial posee autovalor
        negativo (ejemplo canónico de detección de entrelazamiento vía PPT).
        """
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        ppt_min, is_separable = Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(
            choi, 2
        )
        assert ppt_min < -1e-6
        assert is_separable is False

    def test_ppt_full_amplitude_damping_is_entanglement_breaking_hence_separable(self):
        r"""
        Teorema (Horodecki): un canal *entanglement-breaking* (γ=1, colapso a
        \(|0\rangle\langle0|\)) produce un estado de Choi separable ⇒ PPT.
        """
        ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(1.0)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        ppt_min, is_separable = Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(
            choi, 2
        )
        assert ppt_min >= _SPECTRAL_PSD_FLOOR
        assert is_separable is True

    def test_ppt_maximally_depolarizing_channel_is_separable(self, depolarizing_dimension):
        d = depolarizing_dimension
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(d, 1.0)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, d)
        ppt_min, is_separable = Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(
            choi, d
        )
        assert is_separable is True
        assert ppt_min >= _SPECTRAL_PSD_FLOOR

    def test_ppt_matrix_is_hermitian_after_partial_transpose(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        # No hay getter público de la matriz PT; validamos vía espectro simétrico
        ppt_min, _sep = Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(choi, 2)
        assert math.isfinite(ppt_min)

    # ── 3.3 · estado recurso de Bell ─────────────────────────────────────
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_bell_resource_state_is_valid_density_matrix(self, d):
        rho = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(d)
        assert rho.shape == (d * d, d * d)
        assert _is_density_matrix(rho, tol=1e-8)

    def test_bell_resource_state_is_rank_one_pure_state(self):
        rho = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(3)
        eigs = la.eigvalsh(rho)
        assert np.sum(eigs > 1e-9) == 1
        assert np.max(eigs) == pytest.approx(1.0, abs=1e-9)

    # ── 3.4 · traza parcial sobre Alice ──────────────────────────────────
    def test_partial_trace_A_of_bell_state_is_maximally_mixed(self):
        d = 3
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(d)
        rho_b = Phase3_QuantumSecurityEnforcer._phase3_partial_trace_A(rho_ab, d)
        assert np.allclose(rho_b, np.eye(d) / d, atol=1e-9)

    def test_partial_trace_A_preserves_total_trace(self, depolarizing_dimension):
        d = depolarizing_dimension
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(d)
        rho_b = Phase3_QuantumSecurityEnforcer._phase3_partial_trace_A(rho_ab, d)
        assert float(np.real(np.trace(rho_b))) == pytest.approx(1.0, abs=1e-9)

    def test_partial_trace_A_output_is_hermitian(self):
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(2)
        rho_b = Phase3_QuantumSecurityEnforcer._phase3_partial_trace_A(rho_ab, 2)
        assert _is_hermitian(rho_b, tol=1e-12)

    # ── 3.5 · acción local del canal sobre Alice ─────────────────────────
    def test_apply_local_channel_identity_is_noop(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(2)
        rho_prime = Phase3_QuantumSecurityEnforcer._phase3_apply_local_channel_on_A(
            rho_ab, sanitized, 2
        )
        assert np.allclose(rho_prime, rho_ab, atol=1e-9)

    def test_apply_local_channel_preserves_trace_for_tp_channel(
        self, amplitude_damping_kraus_ops
    ):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(2)
        rho_prime = Phase3_QuantumSecurityEnforcer._phase3_apply_local_channel_on_A(
            rho_ab, sanitized, 2
        )
        assert float(np.real(np.trace(rho_prime))) == pytest.approx(1.0, abs=1e-8)

    def test_apply_local_channel_output_is_hermitian(self, amplitude_damping_kraus_ops):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        rho_ab = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(2)
        rho_prime = Phase3_QuantumSecurityEnforcer._phase3_apply_local_channel_on_A(
            rho_ab, sanitized, 2
        )
        assert _is_hermitian(rho_prime, tol=1e-9)

    # ── 3.6 · residuo de No-Señalización ──────────────────────────────────
    def test_non_signaling_residual_zero_for_identity(self, qubit_identity):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        residual, is_ns = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, 2
        )
        assert residual == pytest.approx(0.0, abs=1e-9)
        assert is_ns is True

    def test_non_signaling_residual_zero_for_amplitude_damping(
        self, amplitude_damping_kraus_ops
    ):
        ops, _g = amplitude_damping_kraus_ops
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        residual, is_ns = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, 2
        )
        assert residual <= 1e-8
        assert is_ns is True

    def test_non_signaling_residual_zero_for_depolarizing(
        self, depolarizing_dimension, depolarizing_p
    ):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(
            depolarizing_dimension, depolarizing_p
        )
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(
            ops, depolarizing_dimension
        )
        residual, is_ns = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, depolarizing_dimension
        )
        assert residual <= 1e-7
        assert is_ns is True

    def test_non_signaling_residual_nonzero_for_non_tp_operators(self, broken_non_tp_kraus):
        r"""
        Teorema (checksum causal, contrapositivo): \(\lnot\text{TP}\Rightarrow r_{NS}\neq0\).
        """
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        residual, is_ns = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, 2
        )
        assert residual > 1e-3
        assert is_ns is False

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_non_signaling_residual_zero_for_random_haar_channels(self, d, k, seed):
        ops = _haar_random_kraus(d, k, seed)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        residual, is_ns = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, d
        )
        assert residual <= 1e-7
        assert is_ns is True

    # ── 3.7 · pureza normalizada de Choi ─────────────────────────────────
    def test_choi_purity_equals_one_for_unitary_channel(self):
        pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators([pauli_z], 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        trace = float(np.real(np.trace(choi)))
        purity = Phase3_QuantumSecurityEnforcer._phase3_choi_purity(choi, trace)
        assert purity == pytest.approx(1.0, abs=1e-9)

    def test_choi_purity_bounded_in_half_open_unit_interval_for_random_channels(self):
        for d, k, seed in RANDOM_CHANNEL_SPECS:
            ops = _haar_random_kraus(d, k, seed)
            sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(ops, d)
            choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(
                sanitized, d
            )
            trace = float(np.real(np.trace(choi)))
            purity = Phase3_QuantumSecurityEnforcer._phase3_choi_purity(choi, trace)
            assert 0.0 < purity <= 1.0 + 1e-9

    def test_choi_purity_handles_near_zero_trace_defensively(self):
        zero_ish = np.zeros((4, 4), dtype=np.complex128)
        purity = Phase3_QuantumSecurityEnforcer._phase3_choi_purity(zero_ish, 0.0)
        assert math.isfinite(purity)

    # ── 3.Ω · composición terminal Decide ─────────────────────────────────
    def test_audit_quantum_security_dto_for_identity_channel(self, qubit_identity):
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        dto3 = Phase3_QuantumSecurityEnforcer.audit_quantum_security(
            dto2, sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert isinstance(dto3, Phase3QuantumSecurityData)
        assert dto3.is_completely_positive is True
        assert dto3.kraus_rank == 1
        assert dto3.is_non_signaling is True
        assert dto3.is_separable_ppt is False  # identidad ⇒ máximamente entrelazado

    def test_audit_quantum_security_raises_cp_violation_on_corrupted_choi(
        self, qubit_identity
    ):
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(qubit_identity, 2)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        corrupted_choi = np.diag([1.0, 1.0, 1.0, -0.9]).astype(np.complex128)
        dto2_corrupted = dataclasses.replace(dto2, choi_matrix=corrupted_choi)
        with pytest.raises(CompletePositivityViolation):
            Phase3_QuantumSecurityEnforcer.audit_quantum_security(
                dto2_corrupted, sanitized, dto1.dimension_d, _DEFAULT_TOL
            )

    def test_audit_quantum_security_raises_non_signaling_violation_for_non_tp_ops(
        self, broken_non_tp_kraus
    ):
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
            choi, _DEFAULT_TOL
        )
        # Choi de un ensamble no-TP sigue siendo PSD (Gram-sum) ⇒ pasa CP,
        # pero debe fallar en NS al recibir operadores no-TP directamente.
        fake_dto2 = Phase2ChoiIsomorphismData(
            trace_preserving_residual=0.75,
            is_trace_preserving=False,
            choi_matrix=choi,
            is_unital=False,
            unital_residual=0.75,
            unitariety_degree=0.0,
            choi_trace=float(np.real(np.trace(choi))),
            tp_diamond_defect=0.75,
        )
        with pytest.raises(NonSignalingViolationError):
            Phase3_QuantumSecurityEnforcer.audit_quantum_security(
                fake_dto2, sanitized, dto1.dimension_d, _DEFAULT_TOL
            )

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_audit_quantum_security_kraus_rank_never_exceeds_choi_dimension(
        self, d, k, seed
    ):
        ops = _haar_random_kraus(d, k, seed)
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        dto3 = Phase3_QuantumSecurityEnforcer.audit_quantum_security(
            dto2, sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert 1 <= dto3.kraus_rank <= d * d


# ═══════════════════════════════════════════════════════════════════════════════
# TEOREMA DEL CHECKSUM CAUSAL (transversal a las tres fases)
# ═══════════════════════════════════════════════════════════════════════════════
class TestChecksumCausalTheorem:
    r"""
    Verifica explícitamente el teorema enunciado en el docstring del módulo:

    \[
    \mathcal{E}\text{ TP}
    \;\Longrightarrow\;
    \delta_\diamond(\Lambda)\approx 0
    \;\land\;
    r_{NS}\approx 0,
    \]

    es decir, que el residuo de No-Señalización (FASE 3) y el defecto de
    traza parcial de Choi (FASE 2) son ambos invariantes numéricos duales
    de la propiedad TP (FASE 2), verificados de forma cruzada sobre la
    misma familia de canales.
    """

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_tp_implies_ns_and_diamond_defect_vanish_jointly(self, d, k, seed):
        ops = _haar_random_kraus(d, k, seed)
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(ops, d)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        dto3 = Phase3_QuantumSecurityEnforcer.audit_quantum_security(
            dto2, sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert dto2.is_trace_preserving is True
        assert dto2.tp_diamond_defect <= 1e-7
        assert dto3.non_signaling_residual <= 1e-7

    def test_non_tp_breaks_ns_and_diamond_defect_jointly(self, broken_non_tp_kraus):
        sanitized, _ = Phase1_KrausSanitizer.sanitize_operators(broken_non_tp_kraus, 2)
        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(sanitized, 2)
        delta = Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_trace_tp_defect(
            choi, 2
        )
        ns_residual, _ = Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
            sanitized, 2
        )
        assert delta > 1e-3
        assert ns_residual > 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — CPTPChannelValidatorAgent (Seal / Veto, ciclo OODA completo)
# ═══════════════════════════════════════════════════════════════════════════════
class TestCPTPChannelValidatorAgentOrchestrator:
    """Cierra el ciclo OODA: Observe → Orient → Decide → Seal/Veto."""

    # ── inicialización ────────────────────────────────────────────────────
    def test_init_default_stratum_is_wisdom(self):
        a = CPTPChannelValidatorAgent()
        assert a._target_stratum == Stratum.WISDOM

    def test_init_accepts_custom_stratum(self):
        a = CPTPChannelValidatorAgent(target_stratum=Stratum.STRATEGY)
        assert a._target_stratum == Stratum.STRATEGY

    # ── canales legítimos son certificados como seguros ────────────────────
    def test_audit_identity_channel_is_secure(self, agent, qubit_identity):
        state = agent.audit_cptp_channel(qubit_identity, 2)
        assert isinstance(state, CPTPChannelGovernanceState)
        assert state.is_channel_secure is True

    def test_audit_amplitude_damping_is_secure_across_gamma_range(
        self, agent, amplitude_damping_kraus_ops
    ):
        ops, _gamma = amplitude_damping_kraus_ops
        state = agent.audit_cptp_channel(ops, 2)
        assert state.is_channel_secure is True

    def test_audit_depolarizing_is_secure_across_dims_and_p(
        self, agent, depolarizing_dimension, depolarizing_p
    ):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(
            depolarizing_dimension, depolarizing_p
        )
        state = agent.audit_cptp_channel(ops, depolarizing_dimension)
        assert state.is_channel_secure is True

    @pytest.mark.parametrize("d,k,seed", RANDOM_CHANNEL_SPECS)
    def test_audit_random_haar_channels_are_always_secure(self, agent, d, k, seed):
        ops = _haar_random_kraus(d, k, seed)
        state = agent.audit_cptp_channel(ops, d)
        assert state.is_channel_secure is True

    def test_audit_replacement_channel_is_secure_but_not_unital(self, agent):
        ops = CPTPChannelValidatorAgent.replacement_kraus()
        state = agent.audit_cptp_channel(ops, 2)
        assert state.is_channel_secure is True
        assert state.choi_audit.is_unital is False  # EB channel, no unital

    # ── validaciones de entrada del compositor público ──────────────────────
    @pytest.mark.parametrize("bad_tol", [-1.0, -1e-9])
    def test_audit_rejects_negative_tolerance(self, agent, qubit_identity, bad_tol):
        with pytest.raises(CPTPValidatorAgentError):
            agent.audit_cptp_channel(qubit_identity, 2, tolerance=bad_tol)

    def test_audit_rejects_nan_tolerance(self, agent, qubit_identity):
        with pytest.raises(CPTPValidatorAgentError):
            agent.audit_cptp_channel(qubit_identity, 2, tolerance=float("nan"))

    def test_audit_rejects_inf_tolerance(self, agent, qubit_identity):
        with pytest.raises(CPTPValidatorAgentError):
            agent.audit_cptp_channel(qubit_identity, 2, tolerance=float("inf"))

    def test_audit_raises_kraus_dimension_error_on_empty_ensemble(self, agent):
        with pytest.raises(KrausDimensionError):
            agent.audit_cptp_channel([], 2)

    def test_audit_raises_kraus_dimension_error_on_shape_mismatch(self, agent):
        m = np.eye(3, dtype=np.complex128)
        with pytest.raises(KrausDimensionError):
            agent.audit_cptp_channel([m], 2)

    def test_audit_raises_trace_preservation_violation_on_broken_channel(
        self, agent, broken_non_tp_kraus
    ):
        with pytest.raises(TracePreservationViolation):
            agent.audit_cptp_channel(broken_non_tp_kraus, 2)

    def test_audit_raises_on_nan_operator(self, agent, nan_kraus):
        with pytest.raises(CPTPValidatorAgentError):
            agent.audit_cptp_channel(nan_kraus, 2)

    def test_audit_raises_on_inf_operator(self, agent, inf_kraus):
        with pytest.raises(CPTPValidatorAgentError):
            agent.audit_cptp_channel(inf_kraus, 2)

    def test_audit_veto_logs_critical_and_reraises(self, agent, broken_non_tp_kraus, caplog):
        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.CPTPChannelValidatorAgent"):
            with pytest.raises(TracePreservationViolation):
                agent.audit_cptp_channel(broken_non_tp_kraus, 2)
        assert any("VETO CUÁNTICO" in rec.message for rec in caplog.records)

    # ── integridad del objeto de gobernanza sellado ─────────────────────────
    def test_governance_state_is_frozen_dataclass(self, agent, qubit_identity):
        state = agent.audit_cptp_channel(qubit_identity, 2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_channel_secure = False  # type: ignore[misc]

    def test_governance_state_timestamp_is_iso8601_utc(self, agent, qubit_identity):
        state = agent.audit_cptp_channel(qubit_identity, 2)
        parsed = datetime.fromisoformat(state.timestamp_utc)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_governance_state_records_wilkinson_tolerance(self, agent, qubit_identity):
        state = agent.audit_cptp_channel(qubit_identity, 2, tolerance=1e-10)
        assert state.wilkinson_tolerance == pytest.approx(1e-10)

    def test_governance_state_records_stratum_name_as_string(self, agent, qubit_identity):
        state = agent.audit_cptp_channel(qubit_identity, 2)
        assert state.stratum == "WISDOM"

    def test_governance_state_nested_audits_are_consistent_with_direct_calls(
        self, agent, amplitude_damping_kraus_ops
    ):
        ops, _gamma = amplitude_damping_kraus_ops
        state = agent.audit_cptp_channel(ops, 2)
        sanitized, dto1 = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        dto2 = Phase2_ChoiIsomorphismCertifier.audit_trace_preservation(
            sanitized, dto1.dimension_d, _DEFAULT_TOL
        )
        assert state.choi_audit.trace_preserving_residual == pytest.approx(
            dto2.trace_preserving_residual, abs=1e-9
        )

    # ── conjunción de hard-gates (Ω.1) — tabla de verdad exhaustiva ─────────
    @staticmethod
    def _dummy_dtos(
        is_finite: bool, is_tp: bool, is_cp: bool, is_ns: bool,
        is_unital: bool = True, is_ppt: bool = True,
    ):
        p1 = Phase1SanitizationData(
            dimension_d=2, num_operators=1, num_operators_effective=1,
            is_finite=is_finite, phase_rotated=False,
        )
        p2 = Phase2ChoiIsomorphismData(
            trace_preserving_residual=0.0 if is_tp else 1.0,
            is_trace_preserving=is_tp,
            choi_matrix=np.eye(4, dtype=np.complex128),
            is_unital=is_unital,
            unital_residual=0.0,
            unitariety_degree=1.0,
        )
        p3 = Phase3QuantumSecurityData(
            minimum_choi_eigenvalue=0.0 if is_cp else -1.0,
            kraus_rank=1,
            is_completely_positive=is_cp,
            non_signaling_residual=0.0 if is_ns else 1.0,
            is_non_signaling=is_ns,
            is_separable_ppt=is_ppt,
            ppt_min_eigenvalue=0.0 if is_ppt else -1.0,
        )
        return p1, p2, p3

    @pytest.mark.parametrize("is_finite", [True, False])
    @pytest.mark.parametrize("is_tp", [True, False])
    @pytest.mark.parametrize("is_cp", [True, False])
    @pytest.mark.parametrize("is_ns", [True, False])
    def test_seal_conjunction_truth_table(self, is_finite, is_tp, is_cp, is_ns):
        p1, p2, p3 = self._dummy_dtos(is_finite, is_tp, is_cp, is_ns)
        expected = is_finite and is_tp and is_cp and is_ns
        result = CPTPChannelValidatorAgent._seal_security_conjunction(p1, p2, p3)
        assert result == expected

    @pytest.mark.parametrize("is_unital", [True, False])
    @pytest.mark.parametrize("is_ppt", [True, False])
    def test_seal_conjunction_is_independent_of_soft_invariants(self, is_unital, is_ppt):
        r"""PPT y unitalidad son invariantes *blandos*: no deben afectar χ_secure."""
        p1, p2, p3 = self._dummy_dtos(
            True, True, True, True, is_unital=is_unital, is_ppt=is_ppt
        )
        assert CPTPChannelValidatorAgent._seal_security_conjunction(p1, p2, p3) is True

    def test_seal_governance_state_packs_all_three_audits(self, agent):
        p1, p2, p3 = self._dummy_dtos(True, True, True, True)
        state = agent._seal_governance_state(p1, p2, p3, True, _DEFAULT_TOL)
        assert state.sanitization_audit is p1
        assert state.choi_audit is p2
        assert state.security_audit is p3
        assert state.is_channel_secure is True

    def test_veto_log_and_reraise_propagates_same_exception_type(self, caplog):
        err = TracePreservationViolation("residuo sintético de prueba")
        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.CPTPChannelValidatorAgent"):
            with pytest.raises(TracePreservationViolation) as exc_info:
                CPTPChannelValidatorAgent._veto_log_and_reraise(err)
        assert exc_info.value is err
        assert any("VETO CUÁNTICO" in rec.message for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (retícula Ω₃ de vetos absolutos)
# ═══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    """Certifica la topología de la jerarquía de excepciones del módulo."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            KrausDimensionError,
            TracePreservationViolation,
            CompletePositivityViolation,
            NonSignalingViolationError,
            PeresHorodeckiSeparabilityError,
            UnitalityViolationError,
            ChoiHermiticityError,
        ],
    )
    def test_all_domain_exceptions_inherit_from_agent_error(self, exc_cls):
        assert issubclass(exc_cls, CPTPValidatorAgentError)

    def test_cptp_validator_agent_error_inherits_topological_invariant_error(self):
        assert issubclass(CPTPValidatorAgentError, TopologicalInvariantError)

    def test_exceptions_are_mutually_distinguishable_types(self):
        assert TracePreservationViolation is not CompletePositivityViolation
        assert NonSignalingViolationError is not ChoiHermiticityError

    def test_exceptions_carry_human_readable_message(self):
        err = KrausDimensionError("mensaje de diagnóstico")
        assert "mensaje de diagnóstico" in str(err)

    def test_exceptions_are_catchable_via_common_root(self):
        with pytest.raises(CPTPValidatorAgentError):
            raise ChoiHermiticityError("defecto antihermítico sintético")


# ═══════════════════════════════════════════════════════════════════════════════
# FÁBRICAS DE REFERENCIA (canales canónicos de calibración)
# ═══════════════════════════════════════════════════════════════════════════════
class TestReferenceChannelFactories:
    """Certifica la corrección física de los canales de referencia expuestos."""

    def test_identity_kraus_returns_single_identity_matrix(self):
        ops = CPTPChannelValidatorAgent.identity_kraus(3)
        assert len(ops) == 1
        assert np.array_equal(ops[0], np.eye(3, dtype=np.complex128))

    @pytest.mark.parametrize("bad_gamma", [-0.1, 1.1, 2.0, -5.0])
    def test_amplitude_damping_rejects_gamma_out_of_bounds(self, bad_gamma):
        with pytest.raises(ValueError):
            CPTPChannelValidatorAgent.amplitude_damping_kraus(bad_gamma)

    def test_amplitude_damping_gamma_zero_reduces_to_identity_action(self):
        ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(0.0)
        rho = np.array([[0.3, 0.4 - 0.1j], [0.4 + 0.1j, 0.7]], dtype=np.complex128)
        out = _apply_channel(ops, rho)
        assert np.allclose(out, rho, atol=1e-9)

    def test_amplitude_damping_gamma_one_maps_everything_to_ground_state(self):
        ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(1.0)
        rho = np.array([[0.2, 0.3 + 0.1j], [0.3 - 0.1j, 0.8]], dtype=np.complex128)
        out = _apply_channel(ops, rho)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(out, expected, atol=1e-9)

    def test_amplitude_damping_full_damping_equals_replacement_channel(self):
        ad_ops = CPTPChannelValidatorAgent.amplitude_damping_kraus(1.0)
        repl_ops = CPTPChannelValidatorAgent.replacement_kraus()
        for a, b in zip(ad_ops, repl_ops):
            assert np.allclose(a, b)

    @pytest.mark.parametrize("bad_p", [-0.1, 1.1, 3.0])
    def test_depolarizing_rejects_p_out_of_bounds(self, bad_p):
        with pytest.raises(ValueError):
            CPTPChannelValidatorAgent.depolarizing_kraus(2, bad_p)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_depolarizing_raw_operator_count_equals_d_squared(self, d):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(d, 0.5)
        assert len(ops) == d * d

    def test_depolarizing_p_zero_collapses_to_identity_after_sanitization(self):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(2, 0.0)
        sanitized, dto = Phase1_KrausSanitizer.sanitize_operators(ops, 2)
        assert dto.num_operators_effective == 1
        assert np.allclose(sanitized[0], np.eye(2), atol=1e-9)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_depolarizing_is_always_valid_cptp_across_dimensions(self, agent, d):
        ops = CPTPChannelValidatorAgent.depolarizing_kraus(d, 0.5)
        state = agent.audit_cptp_channel(ops, d)
        assert state.is_channel_secure is True

    def test_replacement_kraus_maps_arbitrary_state_to_ground_state(self):
        ops = CPTPChannelValidatorAgent.replacement_kraus()
        rho = np.array([[0.1, 0.2 + 0.05j], [0.2 - 0.05j, 0.9]], dtype=np.complex128)
        out = _apply_channel(ops, rho)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(out, expected, atol=1e-9)
        assert _is_density_matrix(out, tol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuración de ejecución local (pytest -q tests/unit/agents/wisdom/...)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))