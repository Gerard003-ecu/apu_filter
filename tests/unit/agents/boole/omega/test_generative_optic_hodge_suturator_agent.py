# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Ruta     : tests/unit/agents/boole/omega/                                    ║
║            test_generative_optic_hodge_suturator_agent.py                    ║
║                                                                              ║
║ Módulo   : tests.unit.agents.boole.omega.                                    ║
║            test_generative_optic_hodge_suturator_agent                       ║
║                                                                              ║
║ Propósito: Batería de pruebas unitarias granulares y rigurosas para          ║
║            generative_optic_hodge_suturator_agent.py                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ Cobertura principal:                                                         ║
║   • Retículo de Heyting Ω₃ y operaciones join/meet/implicación.              ║
║   • Triple Modular Redundancy (TMR) con estrategias de votación.             ║
║   • Timestamps lógicos de Lamport.                                           ║
║   • Proveniencia criptográfica Blake3/SHA-256.                               ║
║   • Factorización de Cholesky optimizada [OPT-1].                            ║
║   • Preservación de energía geodésica [OPT-2].                               ║
║   • Fases OODA: Observe, Orient, Decide.                                     ║
║   • Agente soberano completo: gobernanza, crowbar, fail-secure.              ║
║   • Telemetría, reproducibilidad y robustez numérica.                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIÓN ROBUSTA DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
_MODULE_CANDIDATES = (
    "app.agents.boole.omega.generative_optic_hodge_suturator_agent",
    "app.agents.omega.generative_optic_hodge_suturator_agent",
    "app.boole.omega.generative_optic_hodge_suturator_agent",
    "generative_optic_hodge_suturator_agent",
)

agent_module = None
for candidate in _MODULE_CANDIDATES:
    try:
        agent_module = importlib.import_module(candidate)
        break
    except Exception:
        agent_module = None

if agent_module is None:
    pytest.skip(
        "No se pudo importar generative_optic_hodge_suturator_agent desde "
        f"{_MODULE_CANDIDATES}.",
        allow_module_level=True,
    )

_REQUIRED_SYMBOLS = (
    "GenerativeOpticHodgeSuturatorAgent",
    "OpticSovereignVerdict",
    "CrowbarAction",
    "TMRVoteStrategy",
    "TripleModularRedundancy",
    "LamportTimestamp",
    "CryptographicProvenance",
    "Phase1OpticalObservationCertificate",
    "Phase2EikonalTransportCertificate",
    "Phase3FloquetMonodromyCertificate",
    "OpticSuturationState",
    "OpticSuturatorAgentError",
    "MetricSignatureError",
    "EikonalRefractionError",
    "FloquetInstabilityError",
    "HouseholderIdempotenceError",
    "OpticSuturationVetoError",
    "RiemannCurvatureBoundError",
    "TMRVotingError",
)

_missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(agent_module, name)]
if _missing:
    pytest.fail(
        "El módulo importado no expone la API requerida para los tests: "
        f"{', '.join(_missing)}"
    )

GenerativeOpticHodgeSuturatorAgent = agent_module.GenerativeOpticHodgeSuturatorAgent
OpticSovereignVerdict = agent_module.OpticSovereignVerdict
CrowbarAction = agent_module.CrowbarAction
TMRVoteStrategy = agent_module.TMRVoteStrategy
TripleModularRedundancy = agent_module.TripleModularRedundancy
LamportTimestamp = agent_module.LamportTimestamp
CryptographicProvenance = agent_module.CryptographicProvenance

Phase1OpticalObservationCertificate = agent_module.Phase1OpticalObservationCertificate
Phase2EikonalTransportCertificate = agent_module.Phase2EikonalTransportCertificate
Phase3FloquetMonodromyCertificate = agent_module.Phase3FloquetMonodromyCertificate
OpticSuturationState = agent_module.OpticSuturationState

OpticSuturatorAgentError = agent_module.OpticSuturatorAgentError
MetricSignatureError = agent_module.MetricSignatureError
EikonalRefractionError = agent_module.EikonalRefractionError
FloquetInstabilityError = agent_module.FloquetInstabilityError
HouseholderIdempotenceError = agent_module.HouseholderIdempotenceError
OpticSuturationVetoError = agent_module.OpticSuturationVetoError
RiemannCurvatureBoundError = agent_module.RiemannCurvatureBoundError
TMRVotingError = agent_module.TMRVotingError

# Símbolos opcionales introducidos por suturas de optimización.
OptimizedCholeskyFactorizer = getattr(agent_module, "OptimizedCholeskyFactorizer", None)
GeodesicEnergyPreserver = getattr(agent_module, "GeodesicEnergyPreserver", None)
CholeskyFactorizationError = getattr(agent_module, "CholeskyFactorizationError", None)
GeodesicEnergyDriftError = getattr(agent_module, "GeodesicEnergyDriftError", None)
CryptographicProvenanceError = getattr(
    agent_module,
    "CryptographicProvenanceError",
    None,
)

_HAS_CHOLESKY = OptimizedCholeskyFactorizer is not None
_HAS_GEODESIC_PRESERVER = GeodesicEnergyPreserver is not None


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES Y UTILIDADES DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def rng() -> np.random.Generator:
    """Generador pseudoaleatorio determinista."""
    return np.random.default_rng(2026)


@pytest.fixture()
def agent() -> GenerativeOpticHodgeSuturatorAgent:
    """Agente soberano con TMR mayoritario y sin excepción en veto."""
    return GenerativeOpticHodgeSuturatorAgent(
        raise_on_veto=False,
        tmr_strategy=TMRVoteStrategy.MAJORITY,
        node_id="test-agent-fixture",
    )


def _identity_metric(dim: int) -> np.ndarray:
    """Métrica identidad SPD."""
    return np.eye(dim, dtype=np.float64)


def _idempotent_projector(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    r"""
    Construye un proyector idempotente válido:
        P = I - vvᵀ, con ||v||₂ = 1.

    Nota:
        I - 2vvᵀ es un reflector, no un proyector.
    """
    rng = rng or np.random.default_rng()
    v = rng.standard_normal(dim)
    norm_v = float(np.linalg.norm(v))

    if norm_v <= 1e-14:
        v = np.zeros(dim, dtype=np.float64)
        v[0] = 1.0
    else:
        v = v / norm_v

    P = np.eye(dim, dtype=np.float64) - np.outer(v, v)
    return P


def _stable_monodromy(dim: int, spectral_radius: float = 0.5) -> np.ndarray:
    """Matriz de monodromía estable con radio espectral controlado."""
    return float(spectral_radius) * np.eye(dim, dtype=np.float64)


def _unstable_monodromy(dim: int, spectral_radius: float = 1.2) -> np.ndarray:
    """Matriz de monodromía inestable."""
    return float(spectral_radius) * np.eye(dim, dtype=np.float64)


def _make_coherent_inputs(dim: int = 8):
    """
    Insumos deterministas para una gobernanza COHERENT.

    Se usa:
        - G = I.
        - ∂S con holgura eikonal amplia.
        - P idempotente.
        - M estable.
        - Lente pasiva: focused < raw.
    """
    G = _identity_metric(dim)

    phase_gradient = np.zeros(dim, dtype=np.float64)
    phase_gradient[1] = 3.0

    refractive_index = 1.5

    monodromy = _stable_monodromy(dim, spectral_radius=0.5)

    v = np.zeros(dim, dtype=np.float64)
    v[0] = 1.0
    projector = np.eye(dim, dtype=np.float64) - np.outer(v, v)

    focused_logits_norm = 0.95
    raw_logits_norm = 1.0

    return (
        G,
        phase_gradient,
        refractive_index,
        monodromy,
        projector,
        focused_logits_norm,
        raw_logits_norm,
    )


def _execute_coherent(agent: GenerativeOpticHodgeSuturatorAgent, dim: int = 8):
    """Ejecuta una gobernanza coherente estándar."""
    (
        G,
        phase_gradient,
        refractive_index,
        monodromy,
        projector,
        focused_logits_norm,
        raw_logits_norm,
    ) = _make_coherent_inputs(dim=dim)

    return agent.execute_sovereign_governance(
        metric_tensor_g=G,
        phase_gradient_ds=phase_gradient,
        refractive_index_n=refractive_index,
        monodromy_matrix_m=monodromy,
        householder_projector=projector,
        focused_logits_norm=focused_logits_norm,
        raw_logits_norm=raw_logits_norm,
    )


def _make_degraded_majority_inputs(dim: int = 4):
    r"""
    Insumos para forzar veredicto global DEGRADED por mayoría TMR.

    Fase 1 degradada:
        focused_logits_norm > raw_logits_norm.

    Fase 2 degradada:
        Ecuación eikonal válida pero muy cerca del borde causal.
    """
    G = _identity_metric(dim)

    # n = 1.0, slack = 0.1 ⇒ RHS = 0.9.
    # Se coloca LHS apenas por encima para degradar sin violar.
    phase_gradient = np.zeros(dim, dtype=np.float64)
    phase_gradient[0] = float(np.sqrt(0.9 + 1e-12))

    refractive_index = 1.0
    monodromy = _stable_monodromy(dim, spectral_radius=0.5)
    projector = np.eye(dim, dtype=np.float64)

    focused_logits_norm = 1.2
    raw_logits_norm = 1.0

    return (
        G,
        phase_gradient,
        refractive_index,
        monodromy,
        projector,
        focused_logits_norm,
        raw_logits_norm,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DE HEYTING Ω₃ Y ENUMERACIONES
# ══════════════════════════════════════════════════════════════════════════════
class TestHeytingLatticeAndEnums:
    """Validación del clasificador de subobjetos Ω₃."""

    def test_verdict_order_is_partial(self):
        assert OpticSovereignVerdict.COHERENT < OpticSovereignVerdict.DEGRADED
        assert OpticSovereignVerdict.DEGRADED < OpticSovereignVerdict.VETOED

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT),
            (OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.DEGRADED, OpticSovereignVerdict.DEGRADED),
            (OpticSovereignVerdict.DEGRADED, OpticSovereignVerdict.VETOED, OpticSovereignVerdict.VETOED),
            (OpticSovereignVerdict.VETOED, OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.VETOED),
        ],
    )
    def test_join_returns_supremum(self, a, b, expected):
        assert a.join(b) == expected
        assert b.join(a) == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT),
            (OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.DEGRADED, OpticSovereignVerdict.COHERENT),
            (OpticSovereignVerdict.DEGRADED, OpticSovereignVerdict.VETOED, OpticSovereignVerdict.DEGRADED),
            (OpticSovereignVerdict.VETOED, OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT),
        ],
    )
    def test_meet_returns_infimum(self, a, b, expected):
        assert a.meet(b) == expected
        assert b.meet(a) == expected

    def test_heyting_implication(self):
        assert OpticSovereignVerdict.COHERENT.heyting_implies(OpticSovereignVerdict.DEGRADED)
        assert OpticSovereignVerdict.COHERENT.heyting_implies(OpticSovereignVerdict.VETOED)
        assert OpticSovereignVerdict.DEGRADED.heyting_implies(OpticSovereignVerdict.VETOED)

        assert not OpticSovereignVerdict.DEGRADED.heyting_implies(OpticSovereignVerdict.COHERENT)
        assert not OpticSovereignVerdict.VETOED.heyting_implies(OpticSovereignVerdict.COHERENT)

    def test_crowbar_actions_exist(self):
        assert hasattr(CrowbarAction, "NONE")
        assert hasattr(CrowbarAction, "WATCHDOG_PULSE")
        assert hasattr(CrowbarAction, "HARD_SHORT")
        assert hasattr(CrowbarAction, "EMERGENCY_HALT")

    def test_tmr_strategies_exist(self):
        assert hasattr(TMRVoteStrategy, "MAJORITY")
        assert hasattr(TMRVoteStrategy, "UNANIMOUS")
        assert hasattr(TMRVoteStrategy, "PESSIMISTIC")


# ══════════════════════════════════════════════════════════════════════════════
# TIMESTAMPS LÓGICOS DE LAMPORT
# ══════════════════════════════════════════════════════════════════════════════
class TestLamportTimestamp:
    """Validación del orden causal de Lamport."""

    def test_clock_ordering(self):
        ts1 = LamportTimestamp(
            logical_clock=1,
            node_id="node-a",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )
        ts2 = LamportTimestamp(
            logical_clock=2,
            node_id="node-a",
            physical_timestamp_utc="2026-01-01T00:00:01Z",
        )

        assert ts1 < ts2
        assert not ts2 < ts1

    def test_node_id_tie_breaking(self):
        ts_a = LamportTimestamp(
            logical_clock=10,
            node_id="node-a",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )
        ts_b = LamportTimestamp(
            logical_clock=10,
            node_id="node-b",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )

        assert ts_a < ts_b
        assert not ts_b < ts_a


# ══════════════════════════════════════════════════════════════════════════════
# PROVENIENCIA CRIPTOGRÁFICA
# ══════════════════════════════════════════════════════════════════════════════
class TestCryptographicProvenance:
    """Validación de firmas criptográficas y sensibilidad a nonce."""

    def test_digest_is_deterministic(self):
        payload = b"mic-optic-payload"
        d1 = CryptographicProvenance.compute_blake3_digest(payload)
        d2 = CryptographicProvenance.compute_blake3_digest(payload)

        assert d1 == d2
        assert len(d1) == 64

    def test_digest_changes_with_payload(self):
        d1 = CryptographicProvenance.compute_blake3_digest(b"payload-1")
        d2 = CryptographicProvenance.compute_blake3_digest(b"payload-2")

        assert d1 != d2

    def test_digest_changes_with_salt(self):
        payload = b"payload"
        d1 = CryptographicProvenance.compute_blake3_digest(payload, salt=b"salt-1")
        d2 = CryptographicProvenance.compute_blake3_digest(payload, salt=b"salt-2")

        assert d1 != d2

    def test_generate_provenance_hash_changes_with_nonce(self):
        ts = LamportTimestamp(
            logical_clock=5,
            node_id="node-test",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )

        h1 = CryptographicProvenance.generate_provenance_hash(
            "cert-1",
            "cert-2",
            lamport_ts=ts,
            nonce=b"0" * 16,
        )
        h2 = CryptographicProvenance.generate_provenance_hash(
            "cert-1",
            "cert-2",
            lamport_ts=ts,
            nonce=b"1" * 16,
        )

        assert h1 != h2
        assert len(h1) == 64
        assert len(h2) == 64

    def test_generate_provenance_hash_changes_with_clock(self):
        ts1 = LamportTimestamp(
            logical_clock=1,
            node_id="node-test",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )
        ts2 = LamportTimestamp(
            logical_clock=2,
            node_id="node-test",
            physical_timestamp_utc="2026-01-01T00:00:00Z",
        )

        h1 = CryptographicProvenance.generate_provenance_hash(
            "cert",
            lamport_ts=ts1,
            nonce=b"nonce",
        )
        h2 = CryptographicProvenance.generate_provenance_hash(
            "cert",
            lamport_ts=ts2,
            nonce=b"nonce",
        )

        assert h1 != h2


# ══════════════════════════════════════════════════════════════════════════════
# TRIPLE MODULAR REDUNDANCY (TMR)
# ══════════════════════════════════════════════════════════════════════════════
class TestTripleModularRedundancy:
    """Validación de votación mayoritaria, pesimista y unánime."""

    def test_majority_requires_at_least_three_votes(self):
        with pytest.raises(TMRVotingError):
            TripleModularRedundancy.majority_vote(
                [OpticSovereignVerdict.COHERENT, OpticSovereignVerdict.COHERENT]
            )

    def test_majority_all_same(self):
        verdict, confidence = TripleModularRedundancy.majority_vote(
            [OpticSovereignVerdict.COHERENT] * 3
        )

        assert verdict == OpticSovereignVerdict.COHERENT
        assert confidence == pytest.approx(1.0)

    def test_majority_two_same(self):
        verdict, confidence = TripleModularRedundancy.majority_vote(
            [
                OpticSovereignVerdict.DEGRADED,
                OpticSovereignVerdict.DEGRADED,
                OpticSovereignVerdict.COHERENT,
            ]
        )

        assert verdict == OpticSovereignVerdict.DEGRADED
        assert confidence == pytest.approx(2.0 / 3.0)

    def test_majority_no_clear_majority_raises(self):
        with pytest.raises(TMRVotingError):
            TripleModularRedundancy.majority_vote(
                [
                    OpticSovereignVerdict.COHERENT,
                    OpticSovereignVerdict.DEGRADED,
                    OpticSovereignVerdict.VETOED,
                ]
            )

    def test_pessimistic_vote_returns_supremum(self):
        verdict = TripleModularRedundancy.pessimistic_vote(
            [
                OpticSovereignVerdict.COHERENT,
                OpticSovereignVerdict.DEGRADED,
                OpticSovereignVerdict.COHERENT,
            ]
        )
        assert verdict == OpticSovereignVerdict.DEGRADED

        verdict = TripleModularRedundancy.pessimistic_vote(
            [
                OpticSovereignVerdict.COHERENT,
                OpticSovereignVerdict.VETOED,
                OpticSovereignVerdict.DEGRADED,
            ]
        )
        assert verdict == OpticSovereignVerdict.VETOED

    def test_unanimous_vote_ok(self):
        verdict = TripleModularRedundancy.unanimous_vote(
            [OpticSovereignVerdict.COHERENT] * 3
        )
        assert verdict == OpticSovereignVerdict.COHERENT

    def test_unanimous_vote_raises_on_disagreement(self):
        with pytest.raises(TMRVotingError):
            TripleModularRedundancy.unanimous_vote(
                [
                    OpticSovereignVerdict.COHERENT,
                    OpticSovereignVerdict.COHERENT,
                    OpticSovereignVerdict.DEGRADED,
                ]
            )


# ══════════════════════════════════════════════════════════════════════════════
# [OPT-1] FACTORIZACIÓN DE CHOLESKY OPTIMIZADA
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _HAS_CHOLESKY, reason="OptimizedCholeskyFactorizer no disponible")
class TestOptimizedCholeskyFactorizer:
    """Validación de la sutura de optimización Cholesky."""

    def test_identity_factorization_is_well_conditioned(self):
        G = np.eye(4, dtype=np.float64)
        cache = OptimizedCholeskyFactorizer.compute_cholesky_with_condition_estimate(G)

        assert cache.L_lower.shape == (4, 4)
        assert np.isfinite(cache.condition_estimate)
        assert cache.is_well_conditioned is True

    def test_solve_with_cholesky_identity(self):
        G = np.eye(3, dtype=np.float64)
        b = np.ones(3, dtype=np.float64)

        cache = OptimizedCholeskyFactorizer.compute_cholesky_with_condition_estimate(G)
        x = OptimizedCholeskyFactorizer.solve_with_cholesky(cache, b)

        assert np.allclose(x, b, atol=1e-12)

    def test_quadratic_form_identity(self):
        G = np.eye(5, dtype=np.float64)
        v = np.ones(5, dtype=np.float64)

        cache = OptimizedCholeskyFactorizer.compute_cholesky_with_condition_estimate(G)
        q = OptimizedCholeskyFactorizer.compute_quadratic_form_optimized(cache, v)

        # vᵀ I⁻¹ v = ||v||² = 5.
        assert q == pytest.approx(5.0, rel=1e-12, abs=1e-12)

    def test_non_spd_metric_raises(self):
        G = -np.eye(3, dtype=np.float64)

        with pytest.raises((CholeskyFactorizationError, MetricSignatureError)):
            OptimizedCholeskyFactorizer.compute_cholesky_with_condition_estimate(G)


# ══════════════════════════════════════════════════════════════════════════════
# [OPT-2] PRESERVACIÓN DE ENERGÍA GEODÉSICA
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(
    not _HAS_GEODESIC_PRESERVER,
    reason="GeodesicEnergyPreserver no disponible",
)
class TestGeodesicEnergyPreserver:
    """Validación de conservación de energía geodésica."""

    def test_energy_is_nonnegative(self):
        G = np.eye(4, dtype=np.float64)
        v = np.ones(4, dtype=np.float64)

        energy = GeodesicEnergyPreserver.compute_riemannian_kinetic_energy(v, G)
        assert energy >= 0.0
        assert np.isfinite(energy)

    def test_preserve_geodesic_energy_scales_velocity(self):
        G = np.eye(3, dtype=np.float64)
        v = np.ones(3, dtype=np.float64)
        target_energy = 0.5

        v_preserved, report = GeodesicEnergyPreserver.preserve_geodesic_energy(
            v,
            G,
            target_energy,
        )

        final_energy = GeodesicEnergyPreserver.compute_riemannian_kinetic_energy(
            v_preserved,
            G,
        )

        assert report.projection_applied is True
        assert final_energy == pytest.approx(target_energy, rel=1e-10, abs=1e-12)
        assert report.final_energy == pytest.approx(target_energy, rel=1e-10, abs=1e-12)

    def test_no_projection_when_already_on_target(self):
        G = np.eye(2, dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)
        target_energy = 0.5

        v_preserved, report = GeodesicEnergyPreserver.preserve_geodesic_energy(
            v,
            G,
            target_energy,
        )

        assert report.projection_applied is False
        assert np.allclose(v_preserved, v, atol=1e-14)

    def test_zero_target_energy_produces_zero_velocity(self):
        G = np.eye(2, dtype=np.float64)
        v = np.array([1.0, 2.0], dtype=np.float64)

        v_preserved, report = GeodesicEnergyPreserver.preserve_geodesic_energy(
            v,
            G,
            0.0,
        )

        assert np.linalg.norm(v_preserved) <= 1e-12
        assert report.final_energy <= 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVE
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1Observer:
    """Pruebas granulares de la Fase 1 del agente."""

    def test_spd_identity_metric_returns_coherent_certificate(self, agent):
        G = _identity_metric(6)
        cert = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )

        assert cert.is_metric_spd is True
        assert cert.metric_condition == pytest.approx(1.0, rel=1e-8, abs=1e-8)
        assert cert.kv_compression_ratio == pytest.approx(0.9, rel=1e-10)
        assert cert.verdict == OpticSovereignVerdict.COHERENT
        assert cert.observation_timestamp is not None
        assert cert.observation_timestamp.logical_clock > 0

    def test_non_spd_metric_raises(self, agent):
        G = np.diag([1.0, -1.0, 2.0]).astype(np.float64)

        with pytest.raises(MetricSignatureError):
            agent.observe_metric_and_dissipation(
                metric_tensor_g=G,
                focused_logits_norm=0.9,
                raw_logits_norm=1.0,
            )

    def test_passive_violation_degrades_phase1(self, agent):
        G = _identity_metric(4)

        cert = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=1.2,
            raw_logits_norm=1.0,
        )

        assert cert.verdict == OpticSovereignVerdict.DEGRADED

    def test_high_condition_number_degrades_phase1_without_curvature_blowup(self, agent):
        dim = 16
        eigenvalues = np.linspace(1.0, 2.0e7, dim)
        G = np.diag(eigenvalues).astype(np.float64)

        cert = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )

        assert cert.metric_condition > 1.0e7
        assert cert.verdict == OpticSovereignVerdict.DEGRADED

    def test_curvature_bound_raises_for_extreme_condition(self, agent):
        G = np.diag([1.0, 1.0e12]).astype(np.float64)

        with pytest.raises(RiemannCurvatureBoundError):
            agent.observe_metric_and_dissipation(
                metric_tensor_g=G,
                focused_logits_norm=0.9,
                raw_logits_norm=1.0,
            )

    def test_lamport_clock_advances_between_observations(self, agent):
        G = _identity_metric(3)

        cert1 = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )
        cert2 = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )

        assert cert2.observation_timestamp.logical_clock > cert1.observation_timestamp.logical_clock


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENT
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2Orient:
    """Pruebas granulares de la Fase 2 del agente."""

    def test_valid_orientation_returns_certificate(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0

        P = _idempotent_projector(dim)

        cert = agent.orient_eikonal_and_householder(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            householder_projector=P,
        )

        assert cert.is_eikonal_valid is True
        assert cert.is_householder_idempotent is True
        assert cert.causal_cone_compliance is True
        assert cert.energy_preservation_report is not None
        assert cert.orientation_timestamp is not None

    def test_eikonal_violation_raises(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        P = _idempotent_projector(dim)

        with pytest.raises(EikonalRefractionError):
            agent.orient_eikonal_and_householder(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                householder_projector=P,
            )

    def test_non_idempotent_projector_raises(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0

        v = np.zeros(dim, dtype=np.float64)
        v[0] = 1.0

        # Reflector, no proyector: P² = I, no P.
        P = np.eye(dim, dtype=np.float64) - 2.0 * np.outer(v, v)

        with pytest.raises(HouseholderIdempotenceError):
            agent.orient_eikonal_and_householder(
                metric_tensor_g=G,
                phase_gradient_ds=ds,
                refractive_index_n=1.5,
                householder_projector=P,
            )

    def test_energy_report_preserves_target_energy(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0
        P = _idempotent_projector(dim)

        cert = agent.orient_eikonal_and_householder(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            householder_projector=P,
        )

        report = cert.energy_preservation_report
        assert report is not None
        assert report.final_energy == pytest.approx(
            report.target_energy,
            rel=1e-8,
            abs=1e-10,
        )

    def test_cholesky_cache_reuse_when_available(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0
        P = _idempotent_projector(dim)

        cert1 = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )

        cholesky_cache = getattr(cert1, "cholesky_cache", None)
        if cholesky_cache is None:
            pytest.skip("El certificado de Fase 1 no expone cholesky_cache")

        cert2 = agent.orient_eikonal_and_householder(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            householder_projector=P,
            cholesky_cache=cholesky_cache,
        )

        assert cert2.is_eikonal_valid is True

    def test_orientation_lamport_clock_advances_after_observation(self, agent):
        dim = 4
        G = _identity_metric(dim)
        ds = np.zeros(dim, dtype=np.float64)
        ds[0] = 3.0
        P = _idempotent_projector(dim)

        cert1 = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.9,
            raw_logits_norm=1.0,
        )

        cert2 = agent.orient_eikonal_and_householder(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=1.5,
            householder_projector=P,
        )

        assert cert2.orientation_timestamp.logical_clock > cert1.observation_timestamp.logical_clock


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: DECIDE
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3Decide:
    """Pruebas granulares de la Fase 3 del agente."""

    def test_stable_monodromy_returns_coherent_certificate(self, agent):
        M = _stable_monodromy(dim=5, spectral_radius=0.5)

        cert = agent.decide_floquet_monodromy(
            monodromy_matrix_m=M,
            period_t=1.0,
        )

        assert cert.is_floquet_stable is True
        assert cert.max_floquet_multiplier == pytest.approx(0.5, rel=1e-12, abs=1e-12)
        assert cert.verdict == OpticSovereignVerdict.COHERENT
        assert cert.lyapunov_exponents is not None
        assert np.all(np.isfinite(cert.lyapunov_exponents))
        assert cert.stability_margin > 0.0

    def test_unstable_monodromy_raises(self, agent):
        M = _unstable_monodromy(dim=4, spectral_radius=1.2)

        with pytest.raises(FloquetInstabilityError):
            agent.decide_floquet_monodromy(
                monodromy_matrix_m=M,
                period_t=1.0,
            )

    def test_marginally_stable_monodromy_degrades(self, agent):
        # Dentro del límite estable, pero con margen muy pequeño.
        M = (1.0 + 5.0e-11) * np.eye(3, dtype=np.float64)

        cert = agent.decide_floquet_monodromy(
            monodromy_matrix_m=M,
            period_t=1.0,
            tolerance=1.0e-10,
        )

        assert cert.is_floquet_stable is True
        assert cert.verdict == OpticSovereignVerdict.DEGRADED

    def test_lamport_clock_advances_between_decisions(self, agent):
        M = _stable_monodromy(dim=3, spectral_radius=0.5)

        cert1 = agent.decide_floquet_monodromy(M, period_t=1.0)
        cert2 = agent.decide_floquet_monodromy(M, period_t=1.0)

        assert cert2.decision_timestamp.logical_clock > cert1.decision_timestamp.logical_clock


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO: GOBERNANZA COMPLETA
# ══════════════════════════════════════════════════════════════════════════════
class TestSovereignGovernanceIntegration:
    """Pruebas extremo a extremo del agente soberano."""

    def test_coherent_pipeline_returns_globally_coherent_state(self, agent):
        state = _execute_coherent(agent, dim=8)

        assert state.final_verdict == OpticSovereignVerdict.COHERENT
        assert state.phase1.verdict == OpticSovereignVerdict.COHERENT
        assert state.phase2.verdict == OpticSovereignVerdict.COHERENT
        assert state.phase3.verdict == OpticSovereignVerdict.COHERENT

        assert state.is_globally_coherent is True
        assert state.crowbar_triggered is False
        assert state.crowbar_action == CrowbarAction.NONE
        assert state.risk_level == "MINIMAL"
        assert state.tmr_confidence > 0.66

    def test_invalid_metric_fail_secure_returns_vetoed_state(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.MAJORITY,
            node_id="test-fail-secure",
        )

        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        bad_G = -np.eye(4, dtype=np.float64)

        state = agent.execute_sovereign_governance(
            metric_tensor_g=bad_G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=projector,
            focused_logits_norm=focused_logits_norm,
            raw_logits_norm=raw_logits_norm,
        )

        assert state.final_verdict == OpticSovereignVerdict.VETOED
        assert state.crowbar_triggered is True
        assert state.crowbar_action == CrowbarAction.EMERGENCY_HALT
        assert state.is_globally_coherent is False
        assert state.risk_level == "CRITICAL"
        assert state.tmr_confidence == pytest.approx(0.0)
        assert np.isinf(state.phase1.metric_condition)

    def test_invalid_metric_raises_when_raise_on_veto_enabled(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=True,
            tmr_strategy=TMRVoteStrategy.MAJORITY,
            node_id="test-raise-on-veto",
        )

        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        bad_G = -np.eye(4, dtype=np.float64)

        with pytest.raises(MetricSignatureError):
            agent.execute_sovereign_governance(
                metric_tensor_g=bad_G,
                phase_gradient_ds=phase_gradient,
                refractive_index_n=refractive_index,
                monodromy_matrix_m=monodromy,
                householder_projector=projector,
                focused_logits_norm=focused_logits_norm,
                raw_logits_norm=raw_logits_norm,
            )

    def test_eikonal_violation_fail_secure(self, agent):
        (
            G,
            _,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        zero_ds = np.zeros(4, dtype=np.float64)

        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=zero_ds,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=projector,
            focused_logits_norm=focused_logits_norm,
            raw_logits_norm=raw_logits_norm,
        )

        assert state.final_verdict == OpticSovereignVerdict.VETOED
        assert state.phase2.is_eikonal_valid is False

    def test_floquet_instability_fail_secure(self, agent):
        (
            G,
            phase_gradient,
            refractive_index,
            _,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        unstable_M = _unstable_monodromy(dim=4, spectral_radius=1.3)

        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=unstable_M,
            householder_projector=projector,
            focused_logits_norm=focused_logits_norm,
            raw_logits_norm=raw_logits_norm,
        )

        assert state.final_verdict == OpticSovereignVerdict.VETOED
        assert state.phase3.is_floquet_stable is False

    def test_non_idempotent_projector_fail_secure(self, agent):
        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            _,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        v = np.zeros(4, dtype=np.float64)
        v[0] = 1.0
        reflector = np.eye(4, dtype=np.float64) - 2.0 * np.outer(v, v)

        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=reflector,
            focused_logits_norm=focused_logits_norm,
            raw_logits_norm=raw_logits_norm,
        )

        assert state.final_verdict == OpticSovereignVerdict.VETOED
        assert state.phase2.is_householder_idempotent is False


# ══════════════════════════════════════════════════════════════════════════════
# CROWBAR, TMR Y ESTRATEGIAS DE VOTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
class TestCrowbarAndTMRStrategies:
    """Pruebas de mitigación física y estrategias TMR."""

    def test_degraded_majority_triggers_watchdog(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.MAJORITY,
            node_id="test-degraded-majority",
        )

        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_degraded_majority_inputs(dim=4)

        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=projector,
            focused_logits_norm=focused_logits_norm,
            raw_logits_norm=raw_logits_norm,
        )

        assert state.final_verdict == OpticSovereignVerdict.DEGRADED
        assert state.crowbar_triggered is True
        assert state.crowbar_action == CrowbarAction.WATCHDOG_PULSE
        assert state.is_globally_coherent is False
        assert state.risk_level == "MODERATE"

    def test_pessimistic_strategy_amplifies_single_degradation(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.PESSIMISTIC,
            node_id="test-pessimistic",
        )

        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        # Solo Fase 1 degradada por violación de pasividad.
        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=projector,
            focused_logits_norm=1.2,
            raw_logits_norm=1.0,
        )

        assert state.final_verdict == OpticSovereignVerdict.DEGRADED
        assert state.tmr_confidence == pytest.approx(1.0)
        assert state.crowbar_action == CrowbarAction.WATCHDOG_PULSE

    def test_unanimous_strategy_fail_secure_on_disagreement(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.UNANIMOUS,
            node_id="test-unanimous",
        )

        (
            G,
            phase_gradient,
            refractive_index,
            monodromy,
            projector,
            focused_logits_norm,
            raw_logits_norm,
        ) = _make_coherent_inputs(dim=4)

        # Fase 1 degradada, fases 2 y 3 coherentes ⇒ falta unanimidad.
        state = agent.execute_sovereign_governance(
            metric_tensor_g=G,
            phase_gradient_ds=phase_gradient,
            refractive_index_n=refractive_index,
            monodromy_matrix_m=monodromy,
            householder_projector=projector,
            focused_logits_norm=1.2,
            raw_logits_norm=1.0,
        )

        assert state.final_verdict == OpticSovereignVerdict.VETOED
        assert state.crowbar_action == CrowbarAction.EMERGENCY_HALT
        assert state.tmr_confidence == pytest.approx(0.0)

    def test_unanimous_strategy_coherent_when_all_agree(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.UNANIMOUS,
            node_id="test-unanimous-coherent",
        )

        state = _execute_coherent(agent, dim=6)

        assert state.final_verdict == OpticSovereignVerdict.COHERENT
        assert state.is_globally_coherent is True


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRÍA, PROVENIENCIA Y REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════
class TestTelemetryAndProvenance:
    """Pruebas de observabilidad, firma y reproducibilidad."""

    def test_telemetry_contains_core_metrics(self, agent):
        state = _execute_coherent(agent, dim=6)

        core_keys = {
            "metric_condition_number",
            "riemann_curvature_norm",
            "spectral_entropy",
            "kv_compression_ratio",
            "eikonal_residual",
            "householder_residual",
            "max_floquet_multiplier",
            "floquet_stability_margin",
        }

        assert core_keys.issubset(state.telemetry_metrics.keys())

    def test_telemetry_values_are_finite_in_coherent_state(self, agent):
        state = _execute_coherent(agent, dim=6)

        for key, value in state.telemetry_metrics.items():
            assert np.isfinite(value), f"Métrica no finita: {key}={value}"

    def test_provenance_hash_is_hexadecimal_256_bits(self, agent):
        state = _execute_coherent(agent, dim=6)

        h = state.provenance_hash
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h.lower())

    def test_lamport_timestamp_is_attached_to_state(self, agent):
        state = _execute_coherent(agent, dim=6)

        assert state.lamport_timestamp is not None
        assert state.lamport_timestamp.logical_clock > 0
        assert isinstance(state.lamport_timestamp.node_id, str)

    def test_lamport_clock_advances_between_governance_runs(self, agent):
        state1 = _execute_coherent(agent, dim=6)
        state2 = _execute_coherent(agent, dim=6)

        assert (
            state2.lamport_timestamp.logical_clock
            > state1.lamport_timestamp.logical_clock
        )

    def test_provenance_hash_changes_between_runs(self, agent):
        state1 = _execute_coherent(agent, dim=6)
        state2 = _execute_coherent(agent, dim=6)

        assert state1.provenance_hash != state2.provenance_hash


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTEZ NUMÉRICA Y CASOS LÍMITE
# ══════════════════════════════════════════════════════════════════════════════
class TestNumericalRobustness:
    """Pruebas de borde y estabilidad numérica."""

    def test_zero_raw_logits_norm_does_not_crash_observer(self, agent):
        G = _identity_metric(3)

        # raw_logits_norm = 0 puede producir ratio grande; no debe colapsar.
        cert = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=0.0,
            raw_logits_norm=0.0,
        )

        assert cert.observation_timestamp is not None
        assert isinstance(cert.kv_compression_ratio, float)

    def test_focused_greater_than_raw_is_not_coherent(self, agent):
        G = _identity_metric(4)

        cert = agent.observe_metric_and_dissipation(
            metric_tensor_g=G,
            focused_logits_norm=2.0,
            raw_logits_norm=1.0,
        )

        assert cert.verdict != OpticSovereignVerdict.COHERENT

    def test_large_dimension_identity_pipeline_remains_coherent(self):
        agent = GenerativeOpticHodgeSuturatorAgent(
            raise_on_veto=False,
            tmr_strategy=TMRVoteStrategy.MAJORITY,
            node_id="test-large-dim",
        )

        state = _execute_coherent(agent, dim=32)

        assert state.final_verdict == OpticSovereignVerdict.COHERENT
        assert state.is_globally_coherent is True

    def test_exception_hierarchy_is_consistent(self):
        assert issubclass(MetricSignatureError, OpticSuturatorAgentError)
        assert issubclass(EikonalRefractionError, OpticSuturatorAgentError)
        assert issubclass(FloquetInstabilityError, OpticSuturatorAgentError)
        assert issubclass(HouseholderIdempotenceError, OpticSuturatorAgentError)
        assert issubclass(OpticSuturationVetoError, OpticSuturatorAgentError)
        assert issubclass(RiemannCurvatureBoundError, OpticSuturatorAgentError)
        assert issubclass(TMRVotingError, OpticSuturatorAgentError)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))