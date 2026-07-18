# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_witten_atiyah_agent.py                                         ║
║ Ruta   : tests/unit/agents/omega/test_witten_atiyah_agent.py                 ║
║ Versión: 3.0.0-Atiyah-Singer-Witten-APS-Spectral-Rigorous                    ║
║ Objetivo: Validación granular del Inquisidor Witten–Atiyah (fases anidadas)  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cobertura (fases anidadas + objetos categóricos + orquestación TQFT):
  §0  Constantes, entropía de von Neumann, pureza y umbrales APS
  §1  Jerarquía de excepciones topológicas
  §2  DTOs inmutables (PurifiedPair, DimensionalEmbedding, DiracIndexData, …)
  §3  Fase 1 – Funtor de Olvido Métrico U : Met → Top
  §4  Fase 2 – Inmersión fibrada ι + Teorema del Índice (A-S / APS / η)
  §5  Fase 3 – Integral de Witten y colapso al retículo
  §6  Orquestador WittenAtiyahAgent (composición funtorial completa)
  §7  Invariantes de integración / propiedades algebraicas / edge cases
"""

from __future__ import annotations

import math
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ── SUT ──────────────────────────────────────────────────────────────────────
from app.omega.witten_atiyah_agent import (
    WittenAtiyahConstants,
    WittenAtiyahError,
    GaugeTearingError,
    IndexTheoremViolation,
    OntologicalTQFTVeto,
    PurifiedPair,
    DimensionalEmbedding,
    DiracIndexData,
    IndexCertifiedEmbedding,
    WittenAtiyahVerdict,
    Phase1_MetricForgetfulFunctor,
    Phase2_AtiyahSingerEmbedding,
    Phase3_WittenTQFTVerdictor,
    WittenAtiyahAgent,
)

# Dependencias TQFT (reales o stubs del ecosistema)
try:
    from app.wisdom.semantic_translator import VerdictLevel
except ImportError:
    from app.omega.witten_atiyah_agent import VerdictLevel  # type: ignore

try:
    from app.omega.tqft_projection_manifold import (
        TQFTProjectionManifold,
        TQFTBoundary,
        TQFTVerdict,
        QuantumInvariants,
        TopologicalKnotVeto,
        CobordismDegeneracyError,
        TuraevViroCollapseError,
    )
except ImportError:
    # Stubs mínimos si el módulo TQFT no está disponible en el path de tests
    class TopologicalKnotVeto(Exception):
        pass

    class CobordismDegeneracyError(Exception):
        pass

    class TuraevViroCollapseError(Exception):
        pass

    class TQFTBoundary:
        def __init__(self, state_vector, betti_numbers, hilbert_dimension):
            self.state_vector = state_vector
            self.betti_numbers = betti_numbers
            self.hilbert_dimension = hilbert_dimension

    class QuantumInvariants:
        def __init__(self, chern_simons_action=0j, turaev_viro_state_sum=1j,
                     is_knot_free=True, spectral_rank=1):
            self.chern_simons_action = chern_simons_action
            self.turaev_viro_state_sum = turaev_viro_state_sum
            self.is_knot_free = is_knot_free
            self.spectral_rank = spectral_rank

    class TQFTVerdict:
        def __init__(self, invariants=None, verdict=None, topological_trace=""):
            self.invariants = invariants
            self.verdict = verdict if verdict is not None else VerdictLevel.VIABLE
            self.topological_trace = topological_trace

    class TQFTProjectionManifold:
        def project_intent(self, *args, **kwargs):
            return TQFTVerdict(
                invariants=QuantumInvariants(),
                verdict=VerdictLevel.VIABLE,
                topological_trace="Z(M)=stub",
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FIXTURES GLOBALES                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
DIM = 4
DIM_SMALL = 2
DIM_LARGE = 6
EPS = WittenAtiyahConstants.MACHINE_EPS


def _pure_state_projector(dim: int, index: int = 0) -> NDArray[np.complex128]:
    """Proyector puro |e_index⟩⟨e_index| en dimensión dim."""
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[index, index] = 1.0 + 0.0j
    return rho


def _maximally_mixed(dim: int) -> NDArray[np.complex128]:
    """Estado máximamente mixto I/d."""
    return np.eye(dim, dtype=np.complex128) / dim


def _hermitian_density(
    rng: np.random.Generator, dim: int, pure: bool = False
) -> NDArray[np.complex128]:
    """Operador densidad Hermítico PSD de traza 1 (puro o mixto genérico)."""
    if pure:
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj()).astype(np.complex128)
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = A @ A.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho).real
    return (rho / tr).astype(np.complex128)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=7)


@pytest.fixture
def pure_in() -> NDArray[np.complex128]:
    return _pure_state_projector(DIM, 0)


@pytest.fixture
def pure_out() -> NDArray[np.complex128]:
    return _pure_state_projector(DIM, 1)


@pytest.fixture
def mixed_in(rng: np.random.Generator) -> NDArray[np.complex128]:
    return _hermitian_density(rng, DIM, pure=False)


@pytest.fixture
def mixed_out(rng: np.random.Generator) -> NDArray[np.complex128]:
    return _hermitian_density(rng, DIM, pure=False)


@pytest.fixture
def metric_contaminated(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Estado con traza ≠ 1 y pequeña contaminación anti-Hermítica."""
    rho = _hermitian_density(rng, DIM)
    rho = rho * 3.7  # escala métrica / masa de Fröhlich
    noise = 1e-14 * (
        rng.standard_normal((DIM, DIM)) + 1j * rng.standard_normal((DIM, DIM))
    )
    return (rho + noise).astype(np.complex128)


@pytest.fixture
def non_psd_state() -> NDArray[np.complex128]:
    """Matriz Hermítica con autovalor significativamente negativo."""
    return np.diag(np.array([1.0, 0.5, -0.3, 0.1], dtype=np.complex128))


@pytest.fixture
def zero_trace_state() -> NDArray[np.complex128]:
    return np.zeros((DIM, DIM), dtype=np.complex128)


@pytest.fixture
def identity_connection() -> NDArray[np.float64]:
    return np.zeros((DIM, DIM), dtype=np.float64)


@pytest.fixture
def antisymmetric_connection(rng: np.random.Generator) -> NDArray[np.float64]:
    M = rng.standard_normal((DIM, DIM))
    return 0.5 * (M - M.T)


@pytest.fixture
def phase1() -> Phase1_MetricForgetfulFunctor:
    return Phase1_MetricForgetfulFunctor()


@pytest.fixture
def phase2() -> Phase2_AtiyahSingerEmbedding:
    return Phase2_AtiyahSingerEmbedding()


@pytest.fixture
def mock_tqft_viable() -> MagicMock:
    """TQFTProjectionManifold que siempre devuelve VIABLE."""
    mock = MagicMock(spec=TQFTProjectionManifold)
    mock.project_intent.return_value = TQFTVerdict(
        invariants=QuantumInvariants(
            chern_simons_action=0j,
            turaev_viro_state_sum=1.0 + 0j,
            is_knot_free=True,
            spectral_rank=2,
        ),
        verdict=VerdictLevel.VIABLE,
        topological_trace="Z(M)=1.0000e+00 | S_CS=0.0000e+00 | knot_free=True",
    )
    return mock


@pytest.fixture
def mock_tqft_reject() -> MagicMock:
    """TQFTProjectionManifold que devuelve RECHAZAR (sin lanzar)."""
    mock = MagicMock(spec=TQFTProjectionManifold)
    mock.project_intent.return_value = TQFTVerdict(
        invariants=QuantumInvariants(
            chern_simons_action=1.0 + 0j,
            turaev_viro_state_sum=0.5 + 0j,
            is_knot_free=False,
            spectral_rank=1,
        ),
        verdict=VerdictLevel.RECHAZAR,
        topological_trace="Z(M)=5e-1 | S_CS=1.0 | knot_free=False",
    )
    return mock


@pytest.fixture
def mock_tqft_knot_veto() -> MagicMock:
    """TQFTProjectionManifold que lanza TopologicalKnotVeto."""
    mock = MagicMock(spec=TQFTProjectionManifold)
    mock.project_intent.side_effect = TopologicalKnotVeto("nudo logístico S_CS≠0")
    return mock


@pytest.fixture
def phase3_viable(mock_tqft_viable: MagicMock) -> Phase3_WittenTQFTVerdictor:
    return Phase3_WittenTQFTVerdictor(mock_tqft_viable)


@pytest.fixture
def agent_viable(mock_tqft_viable: MagicMock) -> WittenAtiyahAgent:
    return WittenAtiyahAgent(mock_tqft_viable)


@pytest.fixture
def purified_equal_dim(
    phase1: Phase1_MetricForgetfulFunctor,
    pure_in: NDArray[np.complex128],
    pure_out: NDArray[np.complex128],
) -> PurifiedPair:
    return phase1.apply_forgetful_functor(pure_in, pure_out)


@pytest.fixture
def certified_equal_dim(
    phase2: Phase2_AtiyahSingerEmbedding,
    purified_equal_dim: PurifiedPair,
) -> IndexCertifiedEmbedding:
    return phase2.certify_index(purified_equal_dim)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §0  CONSTANTES, ENTROPÍA, PUREZA Y UMBRALES APS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestWittenAtiyahConstants:
    def test_machine_eps_positive(self) -> None:
        assert WittenAtiyahConstants.MACHINE_EPS == float(np.finfo(np.float64).eps)
        assert WittenAtiyahConstants.MACHINE_EPS > 0.0

    def test_tolerances_ordered(self) -> None:
        assert WittenAtiyahConstants.DIRAC_INDEX_TOLERANCE > WittenAtiyahConstants.MACHINE_EPS
        assert WittenAtiyahConstants.VACUUM_ENTROPY_TOLERANCE >= WittenAtiyahConstants.MACHINE_EPS
        assert WittenAtiyahConstants.POSITIVITY_TOLERANCE > 0.0
        assert WittenAtiyahConstants.ETA_SPECTRAL_CUTOFF > 0.0

    def test_default_betti_sphere(self) -> None:
        b = WittenAtiyahConstants.DEFAULT_BETTI
        assert b[0] == 1  # β₀
        assert len(b) >= 1

    def test_von_neumann_entropy_pure_state(self) -> None:
        # Espectro de un proyector: (1, 0, 0, …) ⇒ S = 0
        eigs = np.array([1.0, 0.0, 0.0, 0.0])
        s = WittenAtiyahConstants.von_neumann_entropy(eigs)
        assert abs(s) < 1e-14

    def test_von_neumann_entropy_maximally_mixed(self) -> None:
        d = 4
        eigs = np.ones(d) / d
        s = WittenAtiyahConstants.von_neumann_entropy(eigs)
        assert abs(s - math.log(d)) < 1e-12

    def test_von_neumann_entropy_empty_or_zero(self) -> None:
        assert WittenAtiyahConstants.von_neumann_entropy(np.array([])) == 0.0
        assert WittenAtiyahConstants.von_neumann_entropy(np.array([0.0, 0.0])) == 0.0

    def test_purity_pure_state(self, pure_in: NDArray[np.complex128]) -> None:
        assert abs(WittenAtiyahConstants.purity(pure_in) - 1.0) < 1e-12

    def test_purity_maximally_mixed(self) -> None:
        rho = _maximally_mixed(DIM)
        expected = 1.0 / DIM
        assert abs(WittenAtiyahConstants.purity(rho) - expected) < 1e-12

    def test_min_hilbert_dim(self) -> None:
        assert WittenAtiyahConstants.MIN_HILBERT_DIM >= 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §1  JERARQUÍA DE EXCEPCIONES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestExceptionHierarchy:
    def test_root_inheritance(self) -> None:
        assert issubclass(WittenAtiyahError, Exception)

    def test_gauge_tearing_inherits(self) -> None:
        assert issubclass(GaugeTearingError, WittenAtiyahError)

    def test_index_violation_inherits(self) -> None:
        assert issubclass(IndexTheoremViolation, WittenAtiyahError)

    def test_ontological_veto_inherits(self) -> None:
        assert issubclass(OntologicalTQFTVeto, WittenAtiyahError)

    def test_catchable_as_root(self) -> None:
        with pytest.raises(WittenAtiyahError):
            raise GaugeTearingError("x")
        with pytest.raises(WittenAtiyahError):
            raise IndexTheoremViolation("x")
        with pytest.raises(WittenAtiyahError):
            raise OntologicalTQFTVeto("x")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §2  DTOs INMUTABLES                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestDataTransferObjects:
    def test_purified_pair_frozen(self, purified_equal_dim: PurifiedPair) -> None:
        with pytest.raises(Exception):
            purified_equal_dim.purity_in = 0.0  # type: ignore[misc]

    def test_purified_pair_fields(self, purified_equal_dim: PurifiedPair) -> None:
        p = purified_equal_dim
        assert p.sigma_in.shape == (DIM, DIM)
        assert p.sigma_out.shape == (DIM, DIM)
        assert 0.0 <= p.purity_in <= 1.0 + 1e-9
        assert 0.0 <= p.purity_out <= 1.0 + 1e-9
        assert p.entropy_in >= -1e-12
        assert p.entropy_out >= -1e-12

    def test_dimensional_embedding_defaults(self, pure_in, pure_out) -> None:
        emb = DimensionalEmbedding(
            sigma_in_embedded=pure_in,
            sigma_out=pure_out,
            vacuum_dimension_added=0,
            is_isometric=True,
        )
        assert emb.betti_numbers == WittenAtiyahConstants.DEFAULT_BETTI
        assert emb.is_isometric is True

    def test_dirac_index_data_structure(self) -> None:
        d = DiracIndexData(
            analytical_index=0,
            topological_invariant=0,
            operator_kernel_dim=2,
            operator_cokernel_dim=2,
            eta_invariant=0.0,
            spectral_flow_estimate=0,
            is_theorem_satisfied=True,
        )
        assert d.is_theorem_satisfied is True
        assert d.analytical_index == 0

    def test_index_certified_embedding_links(
        self, certified_equal_dim: IndexCertifiedEmbedding
    ) -> None:
        c = certified_equal_dim
        assert isinstance(c.embedding, DimensionalEmbedding)
        assert isinstance(c.index_data, DiracIndexData)
        assert isinstance(c.purified, PurifiedPair)
        assert c.index_data.is_theorem_satisfied is True

    def test_witten_atiyah_verdict_structure(
        self, certified_equal_dim: IndexCertifiedEmbedding
    ) -> None:
        v = WittenAtiyahVerdict(
            verdict=VerdictLevel.VIABLE,
            index_data=certified_equal_dim.index_data,
            tqft_trace="Z(M)=1",
            vacuum_dimension_added=0,
        )
        assert v.verdict == VerdictLevel.VIABLE
        assert v.vacuum_dimension_added == 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §3  FASE 1 – FUNTOR DE OLVIDO MÉTRICO                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase1MetricForgetfulFunctor:
    # ── 3.1 Validación de forma ──────────────────────────────────────────────

    def test_non_square_raises(self, phase1: Phase1_MetricForgetfulFunctor) -> None:
        bad = np.ones((3, 4), dtype=np.complex128)
        with pytest.raises(GaugeTearingError, match="cuadrada"):
            phase1._strip_metric_tensor(bad, "σ_bad")

    def test_non_ndarray_raises(self, phase1: Phase1_MetricForgetfulFunctor) -> None:
        with pytest.raises(GaugeTearingError, match="NDArray"):
            phase1._strip_metric_tensor([[1, 0], [0, 0]], "σ_list")  # type: ignore[arg-type]

    def test_zero_dim_raises(self, phase1: Phase1_MetricForgetfulFunctor) -> None:
        bad = np.zeros((0, 0), dtype=np.complex128)
        with pytest.raises(GaugeTearingError):
            phase1._strip_metric_tensor(bad, "σ_empty")

    # ── 3.2 Purificación de estados válidos ──────────────────────────────────

    def test_pure_state_remains_pure(
        self, phase1: Phase1_MetricForgetfulFunctor, pure_in: NDArray[np.complex128]
    ) -> None:
        rho, pur, s = phase1._strip_metric_tensor(pure_in, "σ_in")
        assert abs(pur - 1.0) < 1e-10
        assert abs(s) < 1e-10
        assert abs(np.trace(rho).real - 1.0) < 1e-12
        assert np.allclose(rho, rho.conj().T, atol=1e-12)

    def test_maximally_mixed_purity(
        self, phase1: Phase1_MetricForgetfulFunctor
    ) -> None:
        rho_raw = _maximally_mixed(DIM)
        rho, pur, s = phase1._strip_metric_tensor(rho_raw, "σ_mix")
        assert abs(pur - 1.0 / DIM) < 1e-10
        assert abs(s - math.log(DIM)) < 1e-8

    def test_metric_contamination_normalized(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        metric_contaminated: NDArray[np.complex128],
    ) -> None:
        rho, pur, s = phase1._strip_metric_tensor(metric_contaminated, "σ_met")
        assert abs(np.trace(rho).real - 1.0) < 1e-10
        assert 0.0 < pur <= 1.0 + 1e-9
        # Hermítico
        assert np.linalg.norm(rho - rho.conj().T, ord="fro") < 1e-10
        # PSD: autovalores ≥ −tol
        eigs = la.eigvalsh(rho)
        assert np.all(eigs >= -WittenAtiyahConstants.POSITIVITY_TOLERANCE)

    def test_hermitian_projection_kills_antihermitian(
        self, phase1: Phase1_MetricForgetfulFunctor, rng: np.random.Generator
    ) -> None:
        H = _hermitian_density(rng, DIM)
        K = rng.standard_normal((DIM, DIM)) + 1j * rng.standard_normal((DIM, DIM))
        K = 0.5 * (K - K.conj().T)  # anti-Hermítico
        rho_raw = H + 0.1 * K
        rho, _, _ = phase1._strip_metric_tensor(rho_raw, "σ_ah")
        assert np.linalg.norm(rho - rho.conj().T, ord="fro") < 1e-10

    # ── 3.3 Rechazos patológicos ─────────────────────────────────────────────

    def test_zero_trace_raises(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        zero_trace_state: NDArray[np.complex128],
    ) -> None:
        with pytest.raises(GaugeTearingError, match="colapsó|despreciable|Tr"):
            phase1._strip_metric_tensor(zero_trace_state, "σ_zero")

    def test_non_psd_raises(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        non_psd_state: NDArray[np.complex128],
    ) -> None:
        with pytest.raises(GaugeTearingError, match="negativo"):
            phase1._strip_metric_tensor(non_psd_state, "σ_npsd")

    # ── 3.4 apply_forgetful_functor (terminal Fase 1) ────────────────────────

    def test_apply_forgetful_returns_purified_pair(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
    ) -> None:
        pair = phase1.apply_forgetful_functor(pure_in, pure_out)
        assert isinstance(pair, PurifiedPair)
        assert pair.sigma_in.shape == pure_in.shape
        assert pair.sigma_out.shape == pure_out.shape
        assert abs(np.trace(pair.sigma_in).real - 1.0) < 1e-12
        assert abs(np.trace(pair.sigma_out).real - 1.0) < 1e-12

    def test_apply_forgetful_both_contaminated(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        metric_contaminated: NDArray[np.complex128],
        rng: np.random.Generator,
    ) -> None:
        out = metric_contaminated * 2.0 + 0.01 * np.eye(DIM)
        pair = phase1.apply_forgetful_functor(metric_contaminated, out)
        assert abs(np.trace(pair.sigma_in).real - 1.0) < 1e-10
        assert abs(np.trace(pair.sigma_out).real - 1.0) < 1e-10

    def test_apply_forgetful_propagates_in_error(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        pure_out: NDArray[np.complex128],
        zero_trace_state: NDArray[np.complex128],
    ) -> None:
        with pytest.raises(GaugeTearingError):
            phase1.apply_forgetful_functor(zero_trace_state, pure_out)

    def test_apply_forgetful_propagates_out_error(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        pure_in: NDArray[np.complex128],
        non_psd_state: NDArray[np.complex128],
    ) -> None:
        with pytest.raises(GaugeTearingError):
            phase1.apply_forgetful_functor(pure_in, non_psd_state)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §4  FASE 2 – INMERSIÓN FIBRADA + TEOREMA DEL ÍNDICE                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase2AtiyahSingerEmbedding:
    # ── 4.1 Vacío puro ───────────────────────────────────────────────────────

    def test_build_pure_vacuum_axioms(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        vac = phase2._build_pure_vacuum(3)
        assert vac.shape == (3, 3)
        assert abs(np.trace(vac).real - 1.0) < 1e-14
        assert abs(WittenAtiyahConstants.purity(vac) - 1.0) < 1e-14
        phase2._verify_vacuum_axioms(vac)  # no raise

    def test_build_pure_vacuum_dim_invalid(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        with pytest.raises(GaugeTearingError, match="≥ 1"):
            phase2._build_pure_vacuum(0)

    def test_verify_vacuum_rejects_mixed(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        mixed = _maximally_mixed(3)
        with pytest.raises(GaugeTearingError):
            phase2._verify_vacuum_axioms(mixed)

    # ── 4.2 Inmersión ι ──────────────────────────────────────────────────────

    def test_embedding_equal_dims_is_identity(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        purified_equal_dim: PurifiedPair,
    ) -> None:
        emb = phase2._apply_embedding_functor(purified_equal_dim)
        assert emb.vacuum_dimension_added == 0
        assert emb.is_isometric is True
        assert emb.sigma_in_embedded.shape == emb.sigma_out.shape
        assert np.allclose(emb.sigma_in_embedded, purified_equal_dim.sigma_in)

    def test_embedding_in_smaller_than_out(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        rho_in = _hermitian_density(rng, DIM_SMALL)
        rho_out = _hermitian_density(rng, DIM_LARGE)
        pair = phase1.apply_forgetful_functor(rho_in, rho_out)
        emb = phase2._apply_embedding_functor(pair)
        assert emb.vacuum_dimension_added == DIM_LARGE - DIM_SMALL
        assert emb.sigma_in_embedded.shape == (DIM_LARGE, DIM_LARGE)
        assert emb.sigma_out.shape == (DIM_LARGE, DIM_LARGE)
        # Traza del embebido = 1 (extensión por ceros o renormalización)
        assert abs(np.real(np.trace(emb.sigma_in_embedded)) - 1.0) < 1e-10
        # Bloque superior-izquierdo proporcional / igual a ρ_in
        block = emb.sigma_in_embedded[:DIM_SMALL, :DIM_SMALL]
        assert np.allclose(block, pair.sigma_in, atol=1e-10)

    def test_embedding_contraction_raises(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        rho_in = _hermitian_density(rng, DIM_LARGE)
        rho_out = _hermitian_density(rng, DIM_SMALL)
        pair = phase1.apply_forgetful_functor(rho_in, rho_out)
        with pytest.raises(GaugeTearingError, match="Contracción dimensional"):
            phase2._apply_embedding_functor(pair)

    def test_embedding_custom_betti(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        purified_equal_dim: PurifiedPair,
    ) -> None:
        betti = (1, 2, 1)  # T²-like
        emb = phase2._apply_embedding_functor(purified_equal_dim, betti_numbers=betti)
        assert emb.betti_numbers == betti

    # ── 4.3 η-invariante y espectro ──────────────────────────────────────────

    def test_eta_symmetric_spectrum_zero(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        eigs = np.array([-2.0, -1.0, 1.0, 2.0])
        assert abs(phase2._compute_eta_invariant(eigs)) < 1e-14

    def test_eta_all_positive(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        eigs = np.array([0.1, 0.5, 1.0, 2.0])
        assert phase2._compute_eta_invariant(eigs) == 4.0

    def test_eta_all_negative(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        eigs = np.array([-0.1, -0.5, -1.0])
        assert phase2._compute_eta_invariant(eigs) == -3.0

    def test_eta_near_zero_ignored(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        cutoff = WittenAtiyahConstants.ETA_SPECTRAL_CUTOFF
        eigs = np.array([cutoff * 0.1, 1.0, -1.0])
        # El modo casi-nulo se ignora ⇒ η = sign(1)+sign(-1) = 0
        assert abs(phase2._compute_eta_invariant(eigs)) < 1e-14

    def test_eta_empty_spectrum(
        self, phase2: Phase2_AtiyahSingerEmbedding
    ) -> None:
        assert phase2._compute_eta_invariant(np.array([])) == 0.0

    # ── 4.4 Índice de Atiyah–Singer ──────────────────────────────────────────

    def test_index_equal_dims_satisfied(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        purified_equal_dim: PurifiedPair,
    ) -> None:
        emb = phase2._apply_embedding_functor(purified_equal_dim)
        idx = phase2._evaluate_atiyah_singer_index(emb)
        assert idx.analytical_index == 0
        assert idx.topological_invariant == 0
        assert idx.is_theorem_satisfied is True
        assert idx.operator_kernel_dim == idx.operator_cokernel_dim
        assert isinstance(idx.eta_invariant, float)
        assert isinstance(idx.spectral_flow_estimate, int)

    def test_index_after_embedding_satisfied(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        pair = phase1.apply_forgetful_functor(
            _hermitian_density(rng, DIM_SMALL),
            _hermitian_density(rng, DIM_LARGE),
        )
        emb = phase2._apply_embedding_functor(pair)
        idx = phase2._evaluate_atiyah_singer_index(emb)
        assert idx.is_theorem_satisfied is True
        assert idx.analytical_index == 0

    def test_index_identical_states_full_kernel(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        pure_in: NDArray[np.complex128],
    ) -> None:
        """⧸D = 0 ⇒ ker = d, ind = 0."""
        pair = phase1.apply_forgetful_functor(pure_in, pure_in.copy())
        emb = phase2._apply_embedding_functor(pair)
        idx = phase2._evaluate_atiyah_singer_index(emb)
        assert idx.operator_kernel_dim == DIM
        assert idx.operator_cokernel_dim == DIM
        assert idx.analytical_index == 0
        assert idx.is_theorem_satisfied is True

    # ── 4.5 certify_index (terminal Fase 2) ──────────────────────────────────

    def test_certify_index_success(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        purified_equal_dim: PurifiedPair,
    ) -> None:
        certified = phase2.certify_index(purified_equal_dim)
        assert isinstance(certified, IndexCertifiedEmbedding)
        assert certified.index_data.is_theorem_satisfied is True
        assert certified.purified is purified_equal_dim

    def test_certify_index_with_betti(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        purified_equal_dim: PurifiedPair,
    ) -> None:
        betti = (1, 0, 0)
        certified = phase2.certify_index(purified_equal_dim, betti_numbers=betti)
        assert certified.embedding.betti_numbers == betti

    def test_certify_index_contraction_raises(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        pair = phase1.apply_forgetful_functor(
            _hermitian_density(rng, DIM_LARGE),
            _hermitian_density(rng, DIM_SMALL),
        )
        with pytest.raises(GaugeTearingError, match="Contracción"):
            phase2.certify_index(pair)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §5  FASE 3 – INTEGRAL DE WITTEN Y COLAPSO BOOLEANO                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase3WittenTQFTVerdictor:
    # ── 5.1 Construcción / validación del verdictor ──────────────────────────

    def test_init_rejects_none(self) -> None:
        with pytest.raises(WittenAtiyahError, match="None"):
            Phase3_WittenTQFTVerdictor(None)  # type: ignore[arg-type]

    # ── 5.2 density → state vector ───────────────────────────────────────────

    def test_density_to_state_vector_pure(
        self, phase3_viable: Phase3_WittenTQFTVerdictor, pure_in: NDArray[np.complex128]
    ) -> None:
        v = phase3_viable._density_to_state_vector(pure_in)
        assert v.shape == (DIM,)
        assert v.dtype == np.float64
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12
        # Proyector en e_0 ⇒ masa en el índice 0
        assert v[0] == pytest.approx(1.0, abs=1e-12)

    def test_density_to_state_vector_mixed(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        rho = _maximally_mixed(DIM)
        v = phase3_viable._density_to_state_vector(rho)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12
        # Uniforme ⇒ todas las componentes iguales
        assert np.allclose(v, v[0], atol=1e-12)

    def test_density_to_state_vector_zero_diag_fallback(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        # Matriz con diagonal nula (no es densidad válida, pero el helper
        # debe devolver el vector uniforme como fallback)
        rho = np.zeros((DIM, DIM), dtype=np.complex128)
        v = phase3_viable._density_to_state_vector(rho)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12

    # ── 5.3 Ensamblado de fronteras TQFT ─────────────────────────────────────

    def test_assemble_tqft_boundaries(
        self,
        phase3_viable: Phase3_WittenTQFTVerdictor,
        certified_equal_dim: IndexCertifiedEmbedding,
    ) -> None:
        t_in, t_out = phase3_viable._assemble_tqft_boundaries(certified_equal_dim)
        assert isinstance(t_in, TQFTBoundary)
        assert isinstance(t_out, TQFTBoundary)
        assert t_in.hilbert_dimension == DIM
        assert t_out.hilbert_dimension == DIM
        assert t_in.betti_numbers == t_out.betti_numbers
        assert t_in.state_vector.shape == (DIM,)
        assert t_out.state_vector.shape == (DIM,)

    # ── 5.4 Adaptación de la conexión ────────────────────────────────────────

    def test_adapt_connection_same_dim(
        self,
        phase3_viable: Phase3_WittenTQFTVerdictor,
        antisymmetric_connection: NDArray[np.float64],
    ) -> None:
        A = phase3_viable._adapt_connection_tensor(antisymmetric_connection, DIM)
        assert A.shape == (DIM, DIM)
        assert np.allclose(A, antisymmetric_connection)

    def test_adapt_connection_extend_zeros(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        A_small = np.eye(DIM_SMALL, dtype=np.float64)
        A = phase3_viable._adapt_connection_tensor(A_small, DIM_LARGE)
        assert A.shape == (DIM_LARGE, DIM_LARGE)
        assert np.allclose(A[:DIM_SMALL, :DIM_SMALL], A_small)
        assert np.allclose(A[DIM_SMALL:, :], 0.0)
        assert np.allclose(A[:, DIM_SMALL:], 0.0)

    def test_adapt_connection_non_square_raises(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        with pytest.raises(OntologicalTQFTVeto, match="cuadrada"):
            phase3_viable._adapt_connection_tensor(np.ones((3, 4)), 4)

    def test_adapt_connection_too_large_raises(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        with pytest.raises(OntologicalTQFTVeto, match="no se trunca"):
            phase3_viable._adapt_connection_tensor(np.eye(DIM_LARGE), DIM_SMALL)

    # ── 5.5 Integral de Witten ───────────────────────────────────────────────

    def test_integrate_witten_viable(
        self,
        phase3_viable: Phase3_WittenTQFTVerdictor,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
        mock_tqft_viable: MagicMock,
    ) -> None:
        verdict = phase3_viable._integrate_witten_path(
            certified_equal_dim, identity_connection
        )
        assert verdict.verdict == VerdictLevel.VIABLE
        mock_tqft_viable.project_intent.assert_called_once()

    def test_integrate_witten_knot_veto_wrapped(
        self,
        mock_tqft_knot_veto: MagicMock,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        phase3 = Phase3_WittenTQFTVerdictor(mock_tqft_knot_veto)
        with pytest.raises(OntologicalTQFTVeto, match="Nudo logístico"):
            phase3._integrate_witten_path(certified_equal_dim, identity_connection)

    def test_integrate_witten_cobordism_error_wrapped(
        self,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        mock = MagicMock()
        mock.project_intent.side_effect = CobordismDegeneracyError("desgarro")
        phase3 = Phase3_WittenTQFTVerdictor(mock)
        with pytest.raises(OntologicalTQFTVeto, match="Degeneración|colapso"):
            phase3._integrate_witten_path(certified_equal_dim, identity_connection)

    def test_integrate_witten_turaev_viro_error_wrapped(
        self,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        mock = MagicMock()
        mock.project_intent.side_effect = TuraevViroCollapseError("colapso Z")
        phase3 = Phase3_WittenTQFTVerdictor(mock)
        with pytest.raises(OntologicalTQFTVeto):
            phase3._integrate_witten_path(certified_equal_dim, identity_connection)

    def test_integrate_witten_generic_exception_wrapped(
        self,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        mock = MagicMock()
        mock.project_intent.side_effect = RuntimeError("boom")
        phase3 = Phase3_WittenTQFTVerdictor(mock)
        with pytest.raises(OntologicalTQFTVeto, match="Veto Topológico Absoluto"):
            phase3._integrate_witten_path(certified_equal_dim, identity_connection)

    # ── 5.6 Colapso y render_verdict (terminal Fase 3) ───────────────────────

    def test_collapse_verdict_passthrough(
        self, phase3_viable: Phase3_WittenTQFTVerdictor
    ) -> None:
        tv = TQFTVerdict(verdict=VerdictLevel.VIABLE, topological_trace="ok")
        assert phase3_viable._collapse_verdict(tv) == VerdictLevel.VIABLE
        tv2 = TQFTVerdict(verdict=VerdictLevel.RECHAZAR, topological_trace="no")
        assert phase3_viable._collapse_verdict(tv2) == VerdictLevel.RECHAZAR

    def test_render_verdict_viable(
        self,
        phase3_viable: Phase3_WittenTQFTVerdictor,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        result = phase3_viable.render_verdict(
            certified_equal_dim, identity_connection
        )
        assert isinstance(result, WittenAtiyahVerdict)
        assert result.verdict == VerdictLevel.VIABLE
        assert result.index_data.is_theorem_satisfied is True
        assert "Z(M)" in result.tqft_trace or result.tqft_trace != ""
        assert result.vacuum_dimension_added == 0

    def test_render_verdict_reject(
        self,
        mock_tqft_reject: MagicMock,
        certified_equal_dim: IndexCertifiedEmbedding,
        identity_connection: NDArray[np.float64],
    ) -> None:
        phase3 = Phase3_WittenTQFTVerdictor(mock_tqft_reject)
        result = phase3.render_verdict(certified_equal_dim, identity_connection)
        assert result.verdict == VerdictLevel.RECHAZAR


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §6  ORQUESTADOR SUPREMO – WittenAtiyahAgent                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestWittenAtiyahAgent:
    def test_inheritance_chain(self, agent_viable: WittenAtiyahAgent) -> None:
        assert isinstance(agent_viable, Phase3_WittenTQFTVerdictor)
        assert isinstance(agent_viable, Phase2_AtiyahSingerEmbedding)
        assert isinstance(agent_viable, Phase1_MetricForgetfulFunctor)

    def test_adjudicate_transition_viable(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        level = agent_viable.adjudicate_transition(
            pure_in, pure_out, identity_connection
        )
        assert level == VerdictLevel.VIABLE

    def test_adjudicate_full_structure(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        full = agent_viable.adjudicate_full(
            pure_in, pure_out, identity_connection
        )
        assert isinstance(full, WittenAtiyahVerdict)
        assert full.verdict == VerdictLevel.VIABLE
        assert full.index_data.analytical_index == 0
        assert full.index_data.is_theorem_satisfied is True
        assert full.vacuum_dimension_added == 0

    def test_adjudicate_with_dimension_lift(
        self,
        agent_viable: WittenAtiyahAgent,
        rng: np.random.Generator,
        identity_connection: NDArray[np.float64],
    ) -> None:
        """dim_in < dim_out ⇒ inmersión + extensión de conexión."""
        rho_in = _hermitian_density(rng, DIM_SMALL)
        rho_out = _hermitian_density(rng, DIM)
        # Conexión en dim pequeña; el agente debe extenderla
        A = np.zeros((DIM_SMALL, DIM_SMALL), dtype=np.float64)
        full = agent_viable.adjudicate_full(rho_in, rho_out, A)
        assert full.verdict == VerdictLevel.VIABLE
        assert full.vacuum_dimension_added == DIM - DIM_SMALL

    def test_adjudicate_rejects_contraction(
        self,
        agent_viable: WittenAtiyahAgent,
        rng: np.random.Generator,
        identity_connection: NDArray[np.float64],
    ) -> None:
        rho_in = _hermitian_density(rng, DIM_LARGE)
        rho_out = _hermitian_density(rng, DIM_SMALL)
        with pytest.raises(GaugeTearingError, match="Contracción"):
            agent_viable.adjudicate_transition(rho_in, rho_out, identity_connection)

    def test_adjudicate_rejects_zero_trace(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        with pytest.raises(GaugeTearingError):
            agent_viable.adjudicate_transition(
                np.zeros((DIM, DIM), dtype=np.complex128),
                pure_out,
                identity_connection,
            )

    def test_adjudicate_rejects_non_psd(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_out: NDArray[np.complex128],
        non_psd_state: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        with pytest.raises(GaugeTearingError, match="negativo"):
            agent_viable.adjudicate_transition(
                non_psd_state, pure_out, identity_connection
            )

    def test_adjudicate_knot_veto(
        self,
        mock_tqft_knot_veto: MagicMock,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        agent = WittenAtiyahAgent(mock_tqft_knot_veto)
        with pytest.raises(OntologicalTQFTVeto, match="Nudo logístico"):
            agent.adjudicate_transition(pure_in, pure_out, identity_connection)

    def test_adjudicate_reject_level(
        self,
        mock_tqft_reject: MagicMock,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        agent = WittenAtiyahAgent(mock_tqft_reject)
        level = agent.adjudicate_transition(pure_in, pure_out, identity_connection)
        assert level == VerdictLevel.RECHAZAR

    def test_adjudicate_custom_betti(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        full = agent_viable.adjudicate_full(
            pure_in, pure_out, identity_connection, betti_numbers=(1, 0, 0)
        )
        assert full.verdict == VerdictLevel.VIABLE

    def test_adjudicate_metric_contaminated(
        self,
        agent_viable: WittenAtiyahAgent,
        metric_contaminated: NDArray[np.complex128],
        rng: np.random.Generator,
        identity_connection: NDArray[np.float64],
    ) -> None:
        out = _hermitian_density(rng, DIM) * 2.5
        level = agent_viable.adjudicate_transition(
            metric_contaminated, out, identity_connection
        )
        assert level == VerdictLevel.VIABLE

    def test_adjudicate_transition_matches_full_verdict(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        level = agent_viable.adjudicate_transition(
            pure_in, pure_out, identity_connection
        )
        full = agent_viable.adjudicate_full(
            pure_in, pure_out, identity_connection
        )
        assert level == full.verdict


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §7  INVARIANTES DE INTEGRACIÓN / PROPIEDADES ALGEBRAICAS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestAlgebraicAndIntegrationInvariants:
    def test_phase_chain_types(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase3_viable: Phase3_WittenTQFTVerdictor,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        """Cadena explícita Fase1 → Fase2 → Fase3 con tipos en cada eslabón."""
        pair = phase1.apply_forgetful_functor(pure_in, pure_out)
        assert isinstance(pair, PurifiedPair)

        certified = phase2.certify_index(pair)
        assert isinstance(certified, IndexCertifiedEmbedding)
        assert certified.index_data.is_theorem_satisfied

        result = phase3_viable.render_verdict(certified, identity_connection)
        assert isinstance(result, WittenAtiyahVerdict)
        assert result.verdict == VerdictLevel.VIABLE

    def test_purification_idempotent_on_already_pure(
        self,
        phase1: Phase1_MetricForgetfulFunctor,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
    ) -> None:
        """U(U(ρ)) ≈ U(ρ) (idempotencia del funtor de olvido sobre densidades)."""
        pair1 = phase1.apply_forgetful_functor(pure_in, pure_out)
        pair2 = phase1.apply_forgetful_functor(pair1.sigma_in, pair1.sigma_out)
        assert np.allclose(pair1.sigma_in, pair2.sigma_in, atol=1e-12)
        assert np.allclose(pair1.sigma_out, pair2.sigma_out, atol=1e-12)
        assert abs(pair1.purity_in - pair2.purity_in) < 1e-12

    def test_index_always_zero_for_endomorphisms(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        """Para cualquier par con dim_in ≤ dim_out, ind(⧸D) = 0 tras ι."""
        for d_in, d_out in [(1, 1), (2, 2), (2, 5), (3, 3), (4, 6)]:
            pair = phase1.apply_forgetful_functor(
                _hermitian_density(rng, d_in),
                _hermitian_density(rng, d_out),
            )
            certified = phase2.certify_index(pair)
            assert certified.index_data.analytical_index == 0
            assert certified.index_data.is_theorem_satisfied is True

    def test_trace_preservation_through_embedding(
        self,
        phase2: Phase2_AtiyahSingerEmbedding,
        phase1: Phase1_MetricForgetfulFunctor,
        rng: np.random.Generator,
    ) -> None:
        pair = phase1.apply_forgetful_functor(
            _hermitian_density(rng, DIM_SMALL),
            _hermitian_density(rng, DIM_LARGE),
        )
        emb = phase2._apply_embedding_functor(pair)
        assert abs(np.real(np.trace(emb.sigma_in_embedded)) - 1.0) < 1e-10
        assert abs(np.real(np.trace(emb.sigma_out)) - 1.0) < 1e-10

    def test_hilbert_dim_one_edge_case(
        self,
        agent_viable: WittenAtiyahAgent,
    ) -> None:
        rho = np.array([[1.0 + 0j]], dtype=np.complex128)
        A = np.zeros((1, 1), dtype=np.float64)
        result = agent_viable.adjudicate_full(rho, rho.copy(), A)
        assert result.verdict == VerdictLevel.VIABLE
        assert result.index_data.analytical_index == 0
        assert result.vacuum_dimension_added == 0

    def test_connection_extension_invokes_tqft_with_target_dim(
        self,
        mock_tqft_viable: MagicMock,
        rng: np.random.Generator,
    ) -> None:
        agent = WittenAtiyahAgent(mock_tqft_viable)
        rho_in = _hermitian_density(rng, DIM_SMALL)
        rho_out = _hermitian_density(rng, DIM)
        A = np.zeros((DIM_SMALL, DIM_SMALL), dtype=np.float64)
        agent.adjudicate_transition(rho_in, rho_out, A)

        # Verificar que project_intent recibió conexión de dim = DIM
        assert mock_tqft_viable.project_intent.called
        call_args = mock_tqft_viable.project_intent.call_args
        # args: (tqft_in, tqft_out, A_adapted)
        A_passed = call_args[0][2]
        assert A_passed.shape == (DIM, DIM)
        tqft_in = call_args[0][0]
        assert tqft_in.hilbert_dimension == DIM

    def test_repeated_adjudication_stable(
        self,
        agent_viable: WittenAtiyahAgent,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        antisymmetric_connection: NDArray[np.float64],
    ) -> None:
        results = [
            agent_viable.adjudicate_full(pure_in, pure_out, antisymmetric_connection)
            for _ in range(5)
        ]
        assert all(r.verdict == results[0].verdict for r in results)
        assert all(
            r.index_data.analytical_index == results[0].index_data.analytical_index
            for r in results
        )
        assert all(
            r.vacuum_dimension_added == results[0].vacuum_dimension_added
            for r in results
        )

    def test_module_all_exports_importable(self) -> None:
        import app.omega.witten_atiyah_agent as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"Falta export: {name}"

    def test_full_pipeline_with_real_tqft_if_available(
        self,
        pure_in: NDArray[np.complex128],
        pure_out: NDArray[np.complex128],
        identity_connection: NDArray[np.float64],
    ) -> None:
        """
        Integración opcional con TQFTProjectionManifold real.
        Si el stub no implementa la lógica completa, se acepta VIABLE o veto
        controlado (no crash genérico).
        """
        try:
            from app.omega.tqft_projection_manifold import (
                TQFTProjectionManifold as RealTQFT,
            )
            engine = RealTQFT()
        except Exception:
            pytest.skip("TQFTProjectionManifold real no disponible")

        agent = WittenAtiyahAgent(engine)
        try:
            result = agent.adjudicate_full(
                pure_in, pure_out, identity_connection
            )
            assert result.verdict in (VerdictLevel.VIABLE, VerdictLevel.RECHAZAR)
            assert result.index_data.is_theorem_satisfied is True
        except OntologicalTQFTVeto:
            # Veto topológico legítimo del proyector real
            pass
        except (GaugeTearingError, IndexTheoremViolation):
            raise  # estos no deberían ocurrir con estados puros de igual dim


# ── Entrypoint local (pytest) ────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])