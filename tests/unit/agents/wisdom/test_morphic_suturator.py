r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Suite  : Batería de Verificación OODA — Morphic Suturator Agent             ║
║  Ruta   : tests/unit/agents/wisdom/test_morphic_suturator.py                 ║
║  Módulo bajo prueba: app/agents/wisdom/morphic_suturator_agent.py            ║
║  Versión: 2.0.0-Galois-Adjunction-OODA-Strict-FPU-Secure-Granular-QA         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FILOSOFÍA DE LA SUITE:                                                      ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Replica la topología de herencia del módulo bajo prueba:                    ║
║                                                                              ║
║      Phase1_SpectralObserver                                                 ║
║              △                                                               ║
║      Phase2_AdjunctionOrienter(Phase1_SpectralObserver)                      ║
║              △                                                               ║
║      Phase3_VerdictDecider(Phase2_AdjunctionOrienter)                        ║
║                                                                              ║
║  se traduce a:                                                               ║
║                                                                              ║
║      TestPhase1SpectralObserver                                              ║
║              △                                                               ║
║      TestPhase2AdjunctionOrienter(TestPhase1SpectralObserver)                ║
║              △                                                               ║
║      TestPhase3VerdictDecider(TestPhase2AdjunctionOrienter)                  ║
║                                                                              ║
║  Una cuarta capa — TestMorphicSuturatorAgentOrchestrator — certifica el      ║
║  contrato fail-secure del compositor público. Se verifica explícitamente     ║
║  que FASE 1 de este agente es puramente OBSERVACIONAL: hermiticidad,         ║
║  traza y positividad de la MAC se registran como telemetría, SIN abortar     ║
║  el flujo — solo forma, finitud IEEE-754 y divergencia LAPACK son            ║
║  hard-gates en esta fase.                                                    ║
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

from app.agents.wisdom.morphic_suturator_agent import (
    _DEFAULT_TOL,
    _EPS_HERMITICITY,
    _EPS_TRACE,
    _MACHINE_EPS,
    _SPECTRAL_PSD_FLOOR,
    AdjunctionBreachVeto,
    EpistemicSuturatorVerdict,
    MacSpectralAnomalyError,
    MicSpectralAnomalyError,
    MorphicSuturatorAgent,
    NonFiniteInputError,
    Phase1SpectralObservation,
    Phase1_SpectralObserver,
    Phase2AdjunctionOrientation,
    Phase2_AdjunctionOrienter,
    Phase3_VerdictDecider,
    ScalarDomainError,
    ShapeMismatchError,
    Stratum,
    SuturatorAgentError,
    SuturatorAgentVerdictState,
    ThresholdOrderingError,
    TopologicalInvariantError,
)

RealMatrix = np.ndarray
ComplexMatrix = np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# ORÁCULOS MATEMÁTICOS INDEPENDIENTES
# ═══════════════════════════════════════════════════════════════════════════════
def _fro(m: np.ndarray) -> float:
    return float(la.norm(m, ord="fro"))


def _is_hermitian(m: ComplexMatrix, tol: float = 1e-9) -> bool:
    return _fro(m - m.conj().T) <= tol


def _manual_condition_number(m: RealMatrix) -> float:
    """Oráculo independiente: κ(M) = σ_max / max(σ_min, ε_mach)."""
    s = la.svdvals(m)
    return float(np.max(s) / np.maximum(np.min(s), _MACHINE_EPS))


def _manual_von_neumann_entropy(eigvals: np.ndarray) -> float:
    """Oráculo: S = -Σ pᵢ log(pᵢ) sobre eigvals normalizados y recortados."""
    clipped = np.clip(eigvals, 0.0, None)
    total = float(np.sum(clipped))
    if total <= 1e-15:
        return 0.0
    probs = clipped / total
    s = 0.0
    for p in probs:
        if p > 1e-15:
            s -= p * math.log(p)
    return float(s)


def _haar_random_orthogonal(n: int, seed: int) -> RealMatrix:
    """Matriz ortogonal Haar-aleatoria vía QR de Ginibre (método de Mezzadri)."""
    rng = np.random.default_rng(seed)
    ginibre = rng.normal(size=(n, n))
    q, r = np.linalg.qr(ginibre)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return (q * ph).astype(np.float64)


def _random_ginibre_density_matrix(d: int, seed: int) -> ComplexMatrix:
    """Operador densidad físico ρ⪰0, Tr(ρ)=1 vía ensamble de Ginibre."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = g @ g.conj().T
    return (rho / np.real(np.trace(rho))).astype(np.complex128)


RANDOM_MIC_SPECS: List[Tuple[int, int]] = [(2, 1), (3, 2), (4, 3), (5, 4)]
RANDOM_DENSITY_SPECS: List[Tuple[int, int]] = [(2, 10), (3, 11), (4, 12)]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES CANÓNICAS COMPARTIDAS
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def agent() -> MorphicSuturatorAgent:
    return MorphicSuturatorAgent()


@pytest.fixture
def identity_mic_2() -> RealMatrix:
    return np.eye(2, dtype=np.float64)


@pytest.fixture
def identity_mic_3() -> RealMatrix:
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def singular_mic() -> RealMatrix:
    return np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)


@pytest.fixture
def nan_mic() -> RealMatrix:
    m = np.eye(2, dtype=np.float64)
    m[0, 0] = np.nan
    return m


@pytest.fixture
def mixed_mac_2() -> ComplexMatrix:
    return (np.eye(2, dtype=np.complex128) / 2.0)


@pytest.fixture
def non_hermitian_mac() -> ComplexMatrix:
    return np.array([[0.5, 1.0j], [0.0, 0.5]], dtype=np.complex128)


@pytest.fixture
def broken_trace_mac() -> ComplexMatrix:
    return np.diag([0.3, 0.5]).astype(np.complex128)


@pytest.fixture
def negative_eigen_mac() -> ComplexMatrix:
    return np.diag([1.5, -0.5]).astype(np.complex128)


@pytest.fixture
def inf_mac() -> ComplexMatrix:
    m = np.eye(2, dtype=np.complex128)
    m[1, 1] = np.inf
    return m


@pytest.fixture
def pure_state_mac_2() -> ComplexMatrix:
    psi = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    return np.outer(psi, psi.conj())


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 (TEST) — Observación espectral de la MIC y la MAC
# Ancla de la torre de herencia.
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1SpectralObserver:
    """
    Batería granular de FASE 1: valida

        ValidateMicShape → ValidateMicFinite → SingularValues → Condition
        → RankCertificate → ValidateMacShape → ValidateMacFinite
        → HermResidual → Weyl → Spectrum → TraceAnomaly → Purity/Entropy
        → Positivity → observe_spectral_state(Ω)

    Se verifica explícitamente que esta fase es OBSERVACIONAL: solo forma,
    finitud y divergencia LAPACK abortan el flujo; hermiticidad, traza y
    positividad de la MAC se registran sin lanzar excepción.
    """

    # ── 1.1 · forma de la MIC ─────────────────────────────────────────────
    def test_validate_mic_shape_accepts_square_matrix(self, identity_mic_3):
        assert Phase1_SpectralObserver._phase1_validate_mic_shape(identity_mic_3) == 3

    def test_validate_mic_shape_rejects_non_ndarray(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mic_shape([[1, 0], [0, 1]])

    def test_validate_mic_shape_rejects_1d(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mic_shape(np.array([1.0, 2.0]))

    def test_validate_mic_shape_rejects_non_square(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mic_shape(np.zeros((2, 3)))

    def test_validate_mic_shape_rejects_zero_dimension(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mic_shape(np.zeros((0, 0)))

    # ── 1.2 · finitud de la MIC ───────────────────────────────────────────
    def test_validate_mic_finite_accepts_finite(self, identity_mic_2):
        assert Phase1_SpectralObserver._phase1_validate_mic_finite(identity_mic_2) is True

    def test_validate_mic_finite_rejects_nan(self, nan_mic):
        with pytest.raises(NonFiniteInputError):
            Phase1_SpectralObserver._phase1_validate_mic_finite(nan_mic)

    def test_validate_mic_finite_rejects_inf(self):
        m = np.eye(2)
        m[0, 1] = np.inf
        with pytest.raises(NonFiniteInputError):
            Phase1_SpectralObserver._phase1_validate_mic_finite(m)

    # ── 1.3 · valores singulares de la MIC ────────────────────────────────
    def test_mic_singular_values_of_identity(self, identity_mic_3):
        s = Phase1_SpectralObserver._phase1_mic_singular_values(identity_mic_3)
        assert np.allclose(s, np.ones(3))

    def test_mic_singular_values_raises_on_linalg_divergence(self, monkeypatch):
        def _boom(_m, **_kw):
            raise la.LinAlgError("SVD sintéticamente divergente")

        monkeypatch.setattr(la, "svdvals", _boom)
        with pytest.raises(MicSpectralAnomalyError):
            Phase1_SpectralObserver._phase1_mic_singular_values(np.eye(2))

    # ── 1.4 · número de condición ─────────────────────────────────────────
    def test_mic_condition_number_of_identity_is_one(self):
        s = np.array([1.0, 1.0])
        assert Phase1_SpectralObserver._phase1_mic_condition_number(s) == pytest.approx(1.0)

    def test_mic_condition_number_matches_independent_oracle(self, identity_mic_3):
        s = Phase1_SpectralObserver._phase1_mic_singular_values(identity_mic_3)
        cond = Phase1_SpectralObserver._phase1_mic_condition_number(s)
        assert cond == pytest.approx(_manual_condition_number(identity_mic_3))

    def test_mic_condition_number_floored_by_machine_eps_for_zero_singular_value(self):
        s = np.array([2.0, 0.0])
        cond = Phase1_SpectralObserver._phase1_mic_condition_number(s)
        assert cond == pytest.approx(2.0 / _MACHINE_EPS)
        assert math.isfinite(cond)  # nunca produce inf real (a diferencia de v1.0.0 SVD)

    # ── 1.5 · certificación de rango ──────────────────────────────────────
    def test_mic_rank_certificate_true_for_full_rank(self):
        s = np.array([2.0, 1.5, 1.0])
        assert Phase1_SpectralObserver._phase1_mic_rank_certificate(s, 3) is True

    def test_mic_rank_certificate_false_for_rank_deficient(self):
        s = np.array([1.0, 1e-20])
        assert Phase1_SpectralObserver._phase1_mic_rank_certificate(s, 2) is False

    def test_mic_rank_certificate_is_soft_does_not_raise(self, singular_mic):
        s = Phase1_SpectralObserver._phase1_mic_singular_values(singular_mic)
        result = Phase1_SpectralObserver._phase1_mic_rank_certificate(s, 2)
        assert result is False  # solo etiqueta, no aborta

    # ── 1.6 · forma de la MAC ─────────────────────────────────────────────
    def test_validate_mac_shape_accepts_square(self, mixed_mac_2):
        assert Phase1_SpectralObserver._phase1_validate_mac_shape(mixed_mac_2) == 2

    def test_validate_mac_shape_rejects_non_square(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mac_shape(np.zeros((2, 3), dtype=complex))

    def test_validate_mac_shape_rejects_1d(self):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver._phase1_validate_mac_shape(np.array([1.0, 0.0]))

    # ── 1.7 · finitud de la MAC ───────────────────────────────────────────
    def test_validate_mac_finite_accepts_finite(self, mixed_mac_2):
        assert Phase1_SpectralObserver._phase1_validate_mac_finite(mixed_mac_2) is True

    def test_validate_mac_finite_rejects_inf(self, inf_mac):
        with pytest.raises(NonFiniteInputError):
            Phase1_SpectralObserver._phase1_validate_mac_finite(inf_mac)

    # ── 1.8 · residuo hermítico (blando) ──────────────────────────────────
    def test_mac_hermiticity_residual_zero_for_hermitian(self, mixed_mac_2):
        residual = Phase1_SpectralObserver._phase1_mac_hermiticity_residual(mixed_mac_2)
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_mac_hermiticity_residual_positive_for_non_hermitian(self, non_hermitian_mac):
        residual = Phase1_SpectralObserver._phase1_mac_hermiticity_residual(non_hermitian_mac)
        assert residual > 1e-3

    # ── 1.9 · proyección de Weyl ──────────────────────────────────────────
    def test_weyl_symmetrize_produces_exactly_hermitian(self, non_hermitian_mac):
        sym = Phase1_SpectralObserver._phase1_weyl_symmetrize_mac(non_hermitian_mac)
        assert _is_hermitian(sym, tol=1e-15)

    def test_weyl_symmetrize_matches_manual_formula(self, non_hermitian_mac):
        expected = 0.5 * (non_hermitian_mac + non_hermitian_mac.conj().T)
        result = Phase1_SpectralObserver._phase1_weyl_symmetrize_mac(non_hermitian_mac)
        assert np.allclose(result, expected)

    # ── 1.10 · espectro de la MAC ──────────────────────────────────────────
    def test_mac_spectrum_of_mixed_state(self, mixed_mac_2):
        eigvals = Phase1_SpectralObserver._phase1_mac_spectrum(mixed_mac_2)
        assert np.allclose(np.sort(eigvals), [0.5, 0.5])

    def test_mac_spectrum_raises_on_linalg_divergence(self, monkeypatch):
        def _boom(_m, **_kw):
            raise la.LinAlgError("diagonalización sintéticamente divergente")

        monkeypatch.setattr(la, "eigvalsh", _boom)
        with pytest.raises(MacSpectralAnomalyError):
            Phase1_SpectralObserver._phase1_mac_spectrum(np.eye(2, dtype=complex))

    # ── 1.11 · traza y anomalía (blanda, previa al recorte) ───────────────
    def test_mac_trace_and_anomaly_false_for_unit_trace(self):
        eigvals = np.array([0.5, 0.5])
        trace_val, anomaly = Phase1_SpectralObserver._phase1_mac_trace_and_anomaly(eigvals)
        assert trace_val == pytest.approx(1.0)
        assert anomaly is False

    def test_mac_trace_and_anomaly_true_for_broken_trace(self):
        eigvals = np.array([0.3, 0.5])  # Tr = 0.8
        trace_val, anomaly = Phase1_SpectralObserver._phase1_mac_trace_and_anomaly(eigvals)
        assert trace_val == pytest.approx(0.8)
        assert anomaly is True

    def test_mac_trace_and_anomaly_registers_before_negative_clipping(self):
        r"""
        Ataque de regresión: la anomalía de traza debe computarse ANTES del
        recorte de autovalores negativos (a diferencia del cálculo enmascarado
        de v1.0.0), de modo que un ρ no-PSD con Tr≈1 nominal no oculte la
        contribución negativa en la suma cruda.
        """
        eigvals = np.array([1.5, -0.5])  # suma = 1.0 mas contiene negativo
        trace_val, anomaly = Phase1_SpectralObserver._phase1_mac_trace_and_anomaly(eigvals)
        assert trace_val == pytest.approx(1.0)
        assert anomaly is False  # la traza cruda sí es 1; PSD se evalúa aparte

    # ── 1.12 · pureza y entropía (defensivamente normalizadas) ────────────
    def test_purity_and_entropy_of_maximally_mixed_qubit(self):
        eigvals = np.array([0.5, 0.5])
        purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(eigvals)
        assert purity == pytest.approx(0.5, abs=1e-9)
        assert entropy == pytest.approx(math.log(2.0), abs=1e-9)

    def test_purity_and_entropy_of_pure_state_is_one_and_zero(self):
        eigvals = np.array([1.0, 0.0])
        purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(eigvals)
        assert purity == pytest.approx(1.0, abs=1e-9)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_purity_and_entropy_matches_manual_oracle(self):
        eigvals = np.array([0.7, 0.2, 0.1])
        expected_entropy = _manual_von_neumann_entropy(eigvals)
        _purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(eigvals)
        assert entropy == pytest.approx(expected_entropy, abs=1e-12)

    def test_purity_and_entropy_renormalizes_despite_broken_trace(self):
        r"""
        Diseño explícito: la pureza/entropía se computan sobre la
        distribución renormalizada, independientemente de si la traza
        cruda ya estaba corrupta (0.8 en este caso).
        """
        eigvals = np.array([0.3, 0.5])  # Tr = 0.8
        purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(eigvals)
        probs = np.array([0.375, 0.625])
        assert purity == pytest.approx(float(np.sum(probs ** 2)), abs=1e-9)
        assert entropy == pytest.approx(_manual_von_neumann_entropy(eigvals), abs=1e-9)

    def test_purity_and_entropy_defensive_zero_for_all_negative_eigenvalues(self):
        eigvals = np.array([-1.0, -0.5])
        purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(eigvals)
        assert purity == 0.0
        assert entropy == 0.0

    # ── 1.13 · positividad espectral (blanda) ──────────────────────────────
    def test_mac_positivity_true_for_valid_state(self):
        eigvals = np.array([0.5, 0.5])
        min_eig, is_psd = Phase1_SpectralObserver._phase1_mac_positivity(eigvals)
        assert min_eig == pytest.approx(0.5)
        assert is_psd is True

    def test_mac_positivity_false_for_negative_eigenvalue(self):
        eigvals = np.array([1.5, -0.5])
        min_eig, is_psd = Phase1_SpectralObserver._phase1_mac_positivity(eigvals)
        assert min_eig == pytest.approx(-0.5)
        assert is_psd is False

    def test_mac_positivity_tolerates_wilkinson_floor(self):
        eigvals = np.array([1.0, -1e-14])
        _min_eig, is_psd = Phase1_SpectralObserver._phase1_mac_positivity(eigvals)
        assert is_psd is True  # -1e-14 > _SPECTRAL_PSD_FLOOR = -1e-13

    # ── 1.Ω · composición terminal Observe ─────────────────────────────────
    def test_observe_spectral_state_end_to_end_nominal(self, identity_mic_2, mixed_mac_2):
        obs = Phase1_SpectralObserver.observe_spectral_state(identity_mic_2, mixed_mac_2)
        assert isinstance(obs, Phase1SpectralObservation)
        assert obs.mic_is_full_rank is True
        assert obs.mac_is_psd is True
        assert obs.mac_is_hermitian is True
        assert obs.mac_trace_anomaly is False
        assert obs.mac_purity == pytest.approx(0.5, abs=1e-9)
        assert obs.mac_entropy == pytest.approx(math.log(2.0), abs=1e-9)

    def test_observe_spectral_state_raises_on_mic_shape_error(self, mixed_mac_2):
        with pytest.raises(ShapeMismatchError):
            Phase1_SpectralObserver.observe_spectral_state(np.zeros((2, 3)), mixed_mac_2)

    def test_observe_spectral_state_raises_on_mic_nonfinite(self, nan_mic, mixed_mac_2):
        with pytest.raises(NonFiniteInputError):
            Phase1_SpectralObserver.observe_spectral_state(nan_mic, mixed_mac_2)

    def test_observe_spectral_state_raises_on_mac_nonfinite(self, identity_mic_2, inf_mac):
        with pytest.raises(NonFiniteInputError):
            Phase1_SpectralObserver.observe_spectral_state(identity_mic_2, inf_mac)

    def test_observe_spectral_state_does_not_raise_on_non_hermitian_mac(
        self, identity_mic_2, non_hermitian_mac
    ):
        r"""Verificación crítica: la hermiticidad de la MAC es telemetría BLANDA aquí."""
        obs = Phase1_SpectralObserver.observe_spectral_state(identity_mic_2, non_hermitian_mac)
        assert obs.mac_is_hermitian is False
        assert obs.mac_hermiticity_residual > 1e-3  # no lanzó excepción

    def test_observe_spectral_state_does_not_raise_on_broken_trace_mac(
        self, identity_mic_2, broken_trace_mac
    ):
        r"""Verificación crítica: la traza rota de la MAC es telemetría BLANDA aquí."""
        obs = Phase1_SpectralObserver.observe_spectral_state(identity_mic_2, broken_trace_mac)
        assert obs.mac_trace_anomaly is True

    def test_observe_spectral_state_does_not_raise_on_non_psd_mac(
        self, identity_mic_2, negative_eigen_mac
    ):
        r"""Verificación crítica: la no-positividad de la MAC es telemetría BLANDA aquí."""
        obs = Phase1_SpectralObserver.observe_spectral_state(identity_mic_2, negative_eigen_mac)
        assert obs.mac_is_psd is False

    def test_observe_spectral_state_does_not_raise_on_rank_deficient_mic(
        self, singular_mic, mixed_mac_2
    ):
        r"""Verificación crítica: el rango deficiente de la MIC es telemetría BLANDA aquí."""
        obs = Phase1_SpectralObserver.observe_spectral_state(singular_mic, mixed_mac_2)
        assert obs.mic_is_full_rank is False

    @pytest.mark.parametrize("d,seed", RANDOM_DENSITY_SPECS)
    def test_observe_spectral_state_random_ginibre_states_are_physical(
        self, identity_mic_2, d, seed
    ):
        rho = _random_ginibre_density_matrix(d, seed)
        mic = _haar_random_orthogonal(d, seed)
        obs = Phase1_SpectralObserver.observe_spectral_state(mic, rho)
        assert obs.mac_is_psd is True
        assert obs.mac_is_hermitian is True
        assert obs.mac_trace_anomaly is False
        assert obs.mic_is_full_rank is True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 (TEST) — Orientación del residuo de Adjunción de Galois
# Continuación directa de TestPhase1SpectralObserver (herencia real).
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2AdjunctionOrienter(TestPhase1SpectralObserver):
    """
    Batería granular de FASE 2: valida

        ValidateScalarDomain → ValidateThresholdOrdering
        → ComputeResidual → orient_adjunction_residual(Ω)
    """

    # ── 2.1 · dominio escalar ─────────────────────────────────────────────
    def test_validate_scalar_domain_accepts_nonnegative_finite(self):
        Phase2_AdjunctionOrienter._phase2_validate_scalar_domain(0.5, 1.0)  # no debe lanzar

    def test_validate_scalar_domain_accepts_zero(self):
        Phase2_AdjunctionOrienter._phase2_validate_scalar_domain(0.0, 0.0)  # no debe lanzar

    @pytest.mark.parametrize("bad_error", [-0.1, float("nan"), float("inf")])
    def test_validate_scalar_domain_rejects_invalid_reconstruction_error(self, bad_error):
        with pytest.raises(ScalarDomainError):
            Phase2_AdjunctionOrienter._phase2_validate_scalar_domain(bad_error, 1.0)

    @pytest.mark.parametrize("bad_l", [-1.0, float("nan"), float("inf")])
    def test_validate_scalar_domain_rejects_invalid_lipschitz(self, bad_l):
        with pytest.raises(ScalarDomainError):
            Phase2_AdjunctionOrienter._phase2_validate_scalar_domain(0.5, bad_l)

    # ── 2.2 · orden de umbrales ───────────────────────────────────────────
    def test_validate_threshold_ordering_accepts_coherent_leq_veto(self):
        Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(1e-10, 1e-6)

    def test_validate_threshold_ordering_accepts_equal_thresholds(self):
        Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(1e-6, 1e-6)

    def test_validate_threshold_ordering_rejects_coherent_greater_than_veto(self):
        with pytest.raises(ThresholdOrderingError):
            Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(1e-3, 1e-6)

    @pytest.mark.parametrize("bad_val", [-1e-6, float("nan"), float("inf")])
    def test_validate_threshold_ordering_rejects_invalid_coherence_threshold(self, bad_val):
        with pytest.raises(ThresholdOrderingError):
            Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(bad_val, 1e-6)

    @pytest.mark.parametrize("bad_val", [-1e-6, float("nan"), float("inf")])
    def test_validate_threshold_ordering_rejects_invalid_veto_threshold(self, bad_val):
        with pytest.raises(ThresholdOrderingError):
            Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(1e-10, bad_val)

    # ── 2.3 · cómputo del residuo de adjunción ────────────────────────────
    def test_compute_adjunction_residual_zero_when_error_below_lipschitz(self):
        residual = Phase2_AdjunctionOrienter._phase2_compute_adjunction_residual(0.5, 1.0)
        assert residual == pytest.approx(0.0)

    def test_compute_adjunction_residual_positive_when_error_exceeds_lipschitz(self):
        residual = Phase2_AdjunctionOrienter._phase2_compute_adjunction_residual(2.0, 1.0)
        assert residual == pytest.approx(1.0)

    def test_compute_adjunction_residual_matches_manual_formula(self):
        for err, lip in [(0.0, 0.0), (3.0, 1.0), (0.5, 0.5), (10.0, 2.0)]:
            expected = max(0.0, err - lip)
            residual = Phase2_AdjunctionOrienter._phase2_compute_adjunction_residual(err, lip)
            assert residual == pytest.approx(expected)

    # ── 2.Ω · composición terminal Orient ──────────────────────────────────
    def test_orient_adjunction_residual_end_to_end(self):
        orient = Phase2_AdjunctionOrienter.orient_adjunction_residual(
            reconstruction_error=2.0,
            lipschitz_constant=1.0,
            coherence_threshold=1e-10,
            veto_threshold=1e-6,
        )
        assert isinstance(orient, Phase2AdjunctionOrientation)
        assert orient.adjunction_residual == pytest.approx(1.0)
        assert orient.reconstruction_error == pytest.approx(2.0)
        assert orient.lipschitz_constant == pytest.approx(1.0)

    def test_orient_adjunction_residual_raises_scalar_domain_error(self):
        with pytest.raises(ScalarDomainError):
            Phase2_AdjunctionOrienter.orient_adjunction_residual(
                reconstruction_error=-1.0,
                lipschitz_constant=1.0,
                coherence_threshold=1e-10,
                veto_threshold=1e-6,
            )

    def test_orient_adjunction_residual_raises_threshold_ordering_error(self):
        with pytest.raises(ThresholdOrderingError):
            Phase2_AdjunctionOrienter.orient_adjunction_residual(
                reconstruction_error=0.5,
                lipschitz_constant=1.0,
                coherence_threshold=1e-3,
                veto_threshold=1e-6,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 (TEST) — Decisión, escalamiento y sellado
# Continuación directa de TestPhase2AdjunctionOrienter (herencia real).
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3VerdictDecider(TestPhase2AdjunctionOrienter):
    """
    Batería granular de FASE 3: valida

        ClassifyBaseVerdict → EscalateVerdict → DetermineCrowbar
        → LogTelemetry → SealVerdictState → decide_and_seal(Ω)
    """

    # ── 3.1 · clasificación base (fronteras exactas) ───────────────────────
    def test_classify_base_verdict_coherent_at_boundary(self):
        v = Phase3_VerdictDecider._phase3_classify_base_verdict(1e-10, 1e-10, 1e-6)
        assert v == EpistemicSuturatorVerdict.COHERENT

    def test_classify_base_verdict_degraded_just_above_coherence(self):
        v = Phase3_VerdictDecider._phase3_classify_base_verdict(1e-10 + 1e-15, 1e-10, 1e-6)
        assert v == EpistemicSuturatorVerdict.DEGRADED

    def test_classify_base_verdict_degraded_at_veto_boundary(self):
        v = Phase3_VerdictDecider._phase3_classify_base_verdict(1e-6, 1e-10, 1e-6)
        assert v == EpistemicSuturatorVerdict.DEGRADED

    def test_classify_base_verdict_vetoed_just_above_veto_boundary(self):
        v = Phase3_VerdictDecider._phase3_classify_base_verdict(1e-6 + 1e-12, 1e-10, 1e-6)
        assert v == EpistemicSuturatorVerdict.VETOED

    def test_classify_base_verdict_coherent_for_zero_residual(self):
        v = Phase3_VerdictDecider._phase3_classify_base_verdict(0.0, 1e-10, 1e-6)
        assert v == EpistemicSuturatorVerdict.COHERENT

    # ── 3.2 · escalamiento monótono — tabla de verdad exhaustiva ───────────
    @pytest.mark.parametrize(
        "base,psd,rank,expected",
        [
            (EpistemicSuturatorVerdict.COHERENT, True, True, EpistemicSuturatorVerdict.COHERENT),
            (EpistemicSuturatorVerdict.COHERENT, True, False, EpistemicSuturatorVerdict.DEGRADED),
            (EpistemicSuturatorVerdict.COHERENT, False, True, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.COHERENT, False, False, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.DEGRADED, True, True, EpistemicSuturatorVerdict.DEGRADED),
            (EpistemicSuturatorVerdict.DEGRADED, True, False, EpistemicSuturatorVerdict.DEGRADED),
            (EpistemicSuturatorVerdict.DEGRADED, False, True, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.DEGRADED, False, False, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.VETOED, True, True, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.VETOED, True, False, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.VETOED, False, True, EpistemicSuturatorVerdict.VETOED),
            (EpistemicSuturatorVerdict.VETOED, False, False, EpistemicSuturatorVerdict.VETOED),
        ],
    )
    def test_escalate_verdict_truth_table(self, base, psd, rank, expected):
        result = Phase3_VerdictDecider._phase3_escalate_verdict(base, psd, rank)
        assert result == expected

    def test_escalate_verdict_never_decreases_severity(self):
        r"""Propiedad de monotonía: el escalamiento nunca reduce la severidad."""
        for base in EpistemicSuturatorVerdict:
            for psd in (True, False):
                for rank in (True, False):
                    result = Phase3_VerdictDecider._phase3_escalate_verdict(base, psd, rank)
                    assert int(result) >= int(base)

    # ── 3.3 · activación del Crowbar ────────────────────────────────────────
    @pytest.mark.parametrize(
        "verdict,expected",
        [
            (EpistemicSuturatorVerdict.COHERENT, False),
            (EpistemicSuturatorVerdict.DEGRADED, False),
            (EpistemicSuturatorVerdict.VETOED, True),
        ],
    )
    def test_determine_crowbar_activation(self, verdict, expected):
        assert Phase3_VerdictDecider._phase3_determine_crowbar_activation(verdict) is expected

    # ── 3.4 · telemetría graduada por severidad ─────────────────────────────
    def test_log_verdict_telemetry_vetoed_uses_error_level(self, caplog):
        with caplog.at_level(logging.ERROR, logger="MIC.Wisdom.MorphicSuturatorAgent"):
            Phase3_VerdictDecider._phase3_log_verdict_telemetry(
                EpistemicSuturatorVerdict.VETOED, 1.0, 0.5, 0.0, True
            )
        assert any("VETO DE SUTURA" in rec.message for rec in caplog.records)
        assert any(rec.levelno == logging.ERROR for rec in caplog.records)

    def test_log_verdict_telemetry_degraded_uses_warning_level(self, caplog):
        with caplog.at_level(logging.WARNING, logger="MIC.Wisdom.MorphicSuturatorAgent"):
            Phase3_VerdictDecider._phase3_log_verdict_telemetry(
                EpistemicSuturatorVerdict.DEGRADED, 1e-7, 1e-6, 0.5, False
            )
        assert any("degradada" in rec.message for rec in caplog.records)
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_log_verdict_telemetry_coherent_uses_info_level(self, caplog):
        with caplog.at_level(logging.INFO, logger="MIC.Wisdom.MorphicSuturatorAgent"):
            Phase3_VerdictDecider._phase3_log_verdict_telemetry(
                EpistemicSuturatorVerdict.COHERENT, 0.0, 1.0, 0.6931, False
            )
        assert any("aprobada" in rec.message for rec in caplog.records)
        assert any(rec.levelno == logging.INFO for rec in caplog.records)

    # ── 3.5 · sellado del certificado ────────────────────────────────────────
    @staticmethod
    def _dummy_observation(mac_is_psd=True, mic_is_full_rank=True) -> Phase1SpectralObservation:
        return Phase1SpectralObservation(
            mic_shape=(2, 2), mic_is_finite=True, mic_condition_number=1.0,
            mic_is_full_rank=mic_is_full_rank, mac_shape=(2, 2), mac_is_finite=True,
            mac_is_hermitian=True, mac_hermiticity_residual=0.0, mac_trace=1.0,
            mac_trace_anomaly=False, mac_minimum_eigenvalue=0.5 if mac_is_psd else -0.5,
            mac_is_psd=mac_is_psd, mac_purity=0.5, mac_entropy=math.log(2.0),
        )

    @staticmethod
    def _dummy_orientation(residual=0.0) -> Phase2AdjunctionOrientation:
        return Phase2AdjunctionOrientation(
            adjunction_residual=residual, reconstruction_error=0.5,
            lipschitz_constant=1.0, coherence_threshold=1e-10, veto_threshold=1e-6,
        )

    def test_seal_verdict_state_packs_all_fields_correctly(self):
        obs = self._dummy_observation()
        orient = self._dummy_orientation(residual=0.0)
        state = Phase3_VerdictDecider._phase3_seal_verdict_state(
            obs, orient, EpistemicSuturatorVerdict.COHERENT, False
        )
        assert isinstance(state, SuturatorAgentVerdictState)
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT
        assert state.mic_condition_number == pytest.approx(1.0)
        assert state.mac_entropy == pytest.approx(math.log(2.0))
        assert state.is_crowbar_active is False
        assert state.mac_purity == pytest.approx(0.5)
        assert state.mac_trace_anomaly is False
        assert state.mic_rank_deficient is False
        assert state.reconstruction_error == pytest.approx(0.5)
        assert state.lipschitz_constant == pytest.approx(1.0)
        assert state.timestamp_utc != ""

    def test_seal_verdict_state_is_frozen(self):
        obs = self._dummy_observation()
        orient = self._dummy_orientation()
        state = Phase3_VerdictDecider._phase3_seal_verdict_state(
            obs, orient, EpistemicSuturatorVerdict.COHERENT, False
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.verdict = EpistemicSuturatorVerdict.VETOED  # type: ignore[misc]

    def test_seal_verdict_state_timestamp_is_iso8601_utc(self):
        obs = self._dummy_observation()
        orient = self._dummy_orientation()
        state = Phase3_VerdictDecider._phase3_seal_verdict_state(
            obs, orient, EpistemicSuturatorVerdict.COHERENT, False
        )
        parsed = datetime.fromisoformat(state.timestamp_utc)
        assert parsed.tzinfo is not None

    def test_seal_verdict_state_reflects_rank_deficiency(self):
        obs = self._dummy_observation(mic_is_full_rank=False)
        orient = self._dummy_orientation()
        state = Phase3_VerdictDecider._phase3_seal_verdict_state(
            obs, orient, EpistemicSuturatorVerdict.DEGRADED, False
        )
        assert state.mic_rank_deficient is True

    # ── 3.Ω · composición terminal Decide + Act ────────────────────────────
    def test_decide_and_seal_end_to_end_coherent(self):
        obs = self._dummy_observation()
        orient = self._dummy_orientation(residual=0.0)
        state = Phase3_VerdictDecider.decide_and_seal(obs, orient)
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT
        assert state.is_crowbar_active is False

    def test_decide_and_seal_end_to_end_vetoed_by_residual(self):
        obs = self._dummy_observation()
        orient = self._dummy_orientation(residual=1.0)  # > veto_threshold=1e-6
        state = Phase3_VerdictDecider.decide_and_seal(obs, orient)
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.is_crowbar_active is True

    def test_decide_and_seal_end_to_end_vetoed_by_non_psd_mac_despite_zero_residual(self):
        r"""
        Teorema de escalamiento: incluso con residuo de adjunción nulo, un
        ρ_MAC no-PSD fuerza VETOED — la alucinación estructural domina sobre
        la coherencia numérica del residuo escalar.
        """
        obs = self._dummy_observation(mac_is_psd=False)
        orient = self._dummy_orientation(residual=0.0)
        state = Phase3_VerdictDecider.decide_and_seal(obs, orient)
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.is_crowbar_active is True

    def test_decide_and_seal_end_to_end_degraded_by_rank_deficient_mic(self):
        obs = self._dummy_observation(mic_is_full_rank=False)
        orient = self._dummy_orientation(residual=0.0)  # sería COHERENT sin escalar
        state = Phase3_VerdictDecider.decide_and_seal(obs, orient)
        assert state.verdict == EpistemicSuturatorVerdict.DEGRADED
        assert state.is_crowbar_active is False


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — MorphicSuturatorAgent (contrato fail-secure)
# ═══════════════════════════════════════════════════════════════════════════════
class TestMorphicSuturatorAgentOrchestrator:
    """Certifica el ciclo OODA completo y el contrato fail-secure del agente."""

    # ── inicialización ────────────────────────────────────────────────────
    def test_init_default_tolerance_and_stratum(self):
        a = MorphicSuturatorAgent()
        assert a._adjunction_tolerance == pytest.approx(1e-10)
        assert a._target_stratum == Stratum.WISDOM

    def test_init_accepts_custom_tolerance(self):
        a = MorphicSuturatorAgent(adjunction_tolerance=1e-5)
        assert a._adjunction_tolerance == pytest.approx(1e-5)

    # ── _build_failure_state ──────────────────────────────────────────────
    def test_build_failure_state_produces_vetoed_with_inf_metrics(self, caplog):
        with caplog.at_level(logging.CRITICAL, logger="MIC.Wisdom.MorphicSuturatorAgent"):
            state = MorphicSuturatorAgent._build_failure_state("razón sintética de prueba")
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.adjunction_residual == float("inf")
        assert state.mic_condition_number == float("inf")
        assert state.mac_entropy == float("inf")
        assert state.is_crowbar_active is True
        assert state.timestamp_utc != ""
        assert any("razón sintética de prueba" in rec.message for rec in caplog.records)

    def test_build_failure_state_default_fields_are_zeroed(self):
        state = MorphicSuturatorAgent._build_failure_state("x")
        assert state.mac_purity == 0.0
        assert state.mac_trace_anomaly is False
        assert state.mic_rank_deficient is False
        assert state.reconstruction_error == 0.0

    # ── ejecución nominal (COHERENT) ─────────────────────────────────────
    def test_execute_ooda_cycle_coherent_case(self, agent, identity_mic_2, mixed_mac_2):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT
        assert state.is_crowbar_active is False
        assert state.timestamp_utc != ""

    # ── resolución del umbral por defecto (coherence_threshold=None) ─────
    def test_execute_ooda_cycle_none_coherence_threshold_resolves_to_agent_tolerance(
        self, identity_mic_2, mixed_mac_2
    ):
        custom_agent = MorphicSuturatorAgent(adjunction_tolerance=1e-4)
        state = custom_agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=1e-4, lipschitz_constant=0.0,
            coherence_threshold=None,
        )
        assert state.coherence_threshold == pytest.approx(1e-4)
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT

    def test_execute_ooda_cycle_explicit_coherence_threshold_overrides_agent_tolerance(
        self, identity_mic_2, mixed_mac_2
    ):
        custom_agent = MorphicSuturatorAgent(adjunction_tolerance=1e-4)
        state = custom_agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=1e-2, lipschitz_constant=0.0,
            coherence_threshold=1e-1,
        )
        assert state.coherence_threshold == pytest.approx(1e-1)
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT

    # ── casos DEGRADED / VETOED por residuo ────────────────────────────────
    def test_execute_ooda_cycle_degraded_case(self, agent, identity_mic_2, mixed_mac_2):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=1e-7, lipschitz_constant=0.0,
            coherence_threshold=1e-10, veto_threshold=1e-6,
        )
        assert state.verdict == EpistemicSuturatorVerdict.DEGRADED
        assert state.is_crowbar_active is False

    def test_execute_ooda_cycle_vetoed_case_by_residual(self, agent, identity_mic_2, mixed_mac_2):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=1.0, lipschitz_constant=0.0,
            coherence_threshold=1e-10, veto_threshold=1e-6,
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.is_crowbar_active is True

    def test_execute_ooda_cycle_vetoed_case_by_non_psd_mac(
        self, agent, identity_mic_2, negative_eigen_mac
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, negative_eigen_mac,
            reconstruction_error=0.0, lipschitz_constant=1.0,
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.is_crowbar_active is True

    # ── contrato fail-secure: errores de dominio NO propagan por defecto ───
    def test_execute_ooda_cycle_shape_mismatch_returns_vetoed_not_raises(
        self, agent, mixed_mac_2
    ):
        bad_mic = np.zeros((2, 3))
        state = agent.execute_sutured_ooda_cycle(
            bad_mic, mixed_mac_2, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.adjunction_residual == float("inf")

    def test_execute_ooda_cycle_nonfinite_mic_returns_vetoed_not_raises(
        self, agent, nan_mic, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            nan_mic, mixed_mac_2, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED

    def test_execute_ooda_cycle_nonfinite_mac_returns_vetoed_not_raises(
        self, agent, identity_mic_2, inf_mac
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, inf_mac, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED

    def test_execute_ooda_cycle_negative_reconstruction_error_returns_vetoed_not_raises(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2, reconstruction_error=-1.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED

    def test_execute_ooda_cycle_invalid_threshold_ordering_returns_vetoed_not_raises(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=0.0, lipschitz_constant=1.0,
            coherence_threshold=1e-3, veto_threshold=1e-6,  # orden inválido
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED

    def test_execute_ooda_cycle_linalg_divergence_returns_vetoed_not_raises(
        self, agent, identity_mic_2, mixed_mac_2, monkeypatch
    ):
        def _boom(_m, **_kw):
            raise la.LinAlgError("colapso sintético de LAPACK")

        monkeypatch.setattr(la, "svdvals", _boom)
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED

    def test_execute_ooda_cycle_unanticipated_exception_returns_vetoed_not_raises(
        self, agent, identity_mic_2, mixed_mac_2, monkeypatch
    ):
        r"""Red de seguridad final: incluso un bug de programación colapsa a VETOED."""

        def _boom(*_a, **_kw):
            raise RuntimeError("bug de programación sintético no anticipado")

        monkeypatch.setattr(agent, "observe_spectral_state", _boom)
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2, reconstruction_error=0.0, lipschitz_constant=1.0
        )
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.is_crowbar_active is True

    # ── modo estricto: raise_on_veto=True ──────────────────────────────────
    def test_execute_ooda_cycle_raise_on_veto_true_raises_on_legitimate_veto(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        with pytest.raises(AdjunctionBreachVeto):
            agent.execute_sutured_ooda_cycle(
                identity_mic_2, mixed_mac_2,
                reconstruction_error=1.0, lipschitz_constant=0.0,
                raise_on_veto=True,
            )

    def test_execute_ooda_cycle_raise_on_veto_true_does_not_raise_on_coherent(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=0.0, lipschitz_constant=1.0,
            raise_on_veto=True,
        )
        assert state.verdict == EpistemicSuturatorVerdict.COHERENT

    def test_execute_ooda_cycle_raise_on_veto_true_does_not_raise_on_degraded(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2,
            reconstruction_error=1e-7, lipschitz_constant=0.0,
            coherence_threshold=1e-10, veto_threshold=1e-6,
            raise_on_veto=True,
        )
        assert state.verdict == EpistemicSuturatorVerdict.DEGRADED

    def test_execute_ooda_cycle_raise_on_veto_true_does_not_override_internal_domain_errors(
        self, agent, mixed_mac_2
    ):
        r"""
        Precedencia crítica: un error de dominio interno (ShapeMismatchError)
        NUNCA se transforma en AdjunctionBreachVeto — el modo estricto solo
        aplica a un veredicto VETOED legítimamente clasificado por decide_and_seal.
        """
        bad_mic = np.zeros((2, 3))
        state = agent.execute_sutured_ooda_cycle(
            bad_mic, mixed_mac_2,
            reconstruction_error=0.0, lipschitz_constant=1.0,
            raise_on_veto=True,
        )
        # No debe lanzar AdjunctionBreachVeto; debe retornar el estado sellado.
        assert state.verdict == EpistemicSuturatorVerdict.VETOED
        assert state.adjunction_residual == float("inf")

    # ── consistencia con llamadas directas a las fases ────────────────────
    def test_execute_ooda_cycle_matches_direct_phase_composition(
        self, agent, identity_mic_2, mixed_mac_2
    ):
        state = agent.execute_sutured_ooda_cycle(
            identity_mic_2, mixed_mac_2, reconstruction_error=0.3, lipschitz_constant=0.1
        )
        obs = agent.observe_spectral_state(identity_mic_2, mixed_mac_2)
        orient = agent.orient_adjunction_residual(
            0.3, 0.1, agent._adjunction_tolerance, 1e-6
        )
        expected_state = agent.decide_and_seal(obs, orient)
        assert state.verdict == expected_state.verdict
        assert state.adjunction_residual == pytest.approx(expected_state.adjunction_residual)
        assert state.mic_condition_number == pytest.approx(expected_state.mic_condition_number)


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (retícula Ω₃ de vetos de sutura del agente)
# ═══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    """Certifica la topología de la jerarquía de excepciones del módulo."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            NonFiniteInputError,
            ShapeMismatchError,
            MicSpectralAnomalyError,
            MacSpectralAnomalyError,
            ScalarDomainError,
            ThresholdOrderingError,
            AdjunctionBreachVeto,
        ],
    )
    def test_all_domain_exceptions_inherit_from_agent_error(self, exc_cls):
        assert issubclass(exc_cls, SuturatorAgentError)

    def test_suturator_agent_error_inherits_topological_invariant_error(self):
        assert issubclass(SuturatorAgentError, TopologicalInvariantError)

    def test_exceptions_are_mutually_distinguishable(self):
        assert MicSpectralAnomalyError is not MacSpectralAnomalyError
        assert ScalarDomainError is not ThresholdOrderingError

    def test_adjunction_breach_veto_is_catchable_via_common_root(self):
        with pytest.raises(SuturatorAgentError):
            raise AdjunctionBreachVeto("veto sintético de prueba")

    def test_exceptions_carry_human_readable_message(self):
        err = ScalarDomainError("mensaje de diagnóstico escalar")
        assert "mensaje de diagnóstico escalar" in str(err)


# ═══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DE VEREDICTOS — propiedades estructurales del enum
# ═══════════════════════════════════════════════════════════════════════════════
class TestEpistemicSuturatorVerdictLattice:
    """Certifica el orden total y los valores canónicos del retículo Ω₃."""

    def test_verdict_values_are_ordered_by_severity(self):
        assert EpistemicSuturatorVerdict.COHERENT < EpistemicSuturatorVerdict.DEGRADED
        assert EpistemicSuturatorVerdict.DEGRADED < EpistemicSuturatorVerdict.VETOED

    def test_verdict_canonical_integer_values(self):
        assert int(EpistemicSuturatorVerdict.COHERENT) == 0
        assert int(EpistemicSuturatorVerdict.DEGRADED) == 1
        assert int(EpistemicSuturatorVerdict.VETOED) == 2

    def test_verdict_has_exactly_three_members(self):
        assert len(list(EpistemicSuturatorVerdict)) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Configuración de ejecución local (pytest -q tests/unit/agents/wisdom/...)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))