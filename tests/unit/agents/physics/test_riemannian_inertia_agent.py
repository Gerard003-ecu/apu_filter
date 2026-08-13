# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Pruebas : Riemannian Inertia Agent                                           ║
║ Ruta    : tests/unit/agents/physics/test_riemannian_inertia_agent.py         ║
║ Versión : 4.0.0-Topos-Heyting-Liouville-Gauge-Strict                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Contrato de la batería (composición funtorial anidada): ────────────────────────
Las pruebas reproducen el encadenamiento OODA del agente, que a su vez
gobierna al motor físico:

    Motor-F1 → Agente-F1 → Φ₁₂ → Motor-F2 → Agente-F2 → Φ₂₃
             → Motor-F3 → Agente-F3 → InertialGovernanceState

    v_final = v_Liouville ⊔ v_Skew ⊔ v_Work     sobre Ω₃

Cada clase de Fase N termina con la auditoría del morfismo de handoff, que
es exactamente el dominio sobre el que arranca la clase de la Fase N+1.

Álgebra auditada: ──────────────────────────────────────────────────────────────
    Ω₃ = {COHERENT ≤ DEGRADED ≤ VETOED}   (cadena de Heyting, no Booleana)
    ¬¬DEGRADED = VETOED ≠ DEGRADED
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray

import app.agents.physics.riemannian_inertia_agent as ria
from app.agents.physics.riemannian_inertia_agent import (
    DualPairingCollapse,
    ExteriorAlgebraCollapse,
    HeytingLatticeVeto,
    InertialGovernanceState,
    InertialHeytingVerdict,
    InverseCoherenceCollapse,
    LiouvilleVolumeCollapse,
    MetricConditionCollapse,
    MotorContractError,
    MusicalIsomorphismCollapse,
    Phase1_LiouvilleVolumeAuditor,
    Phase1ObservationBridge,
    Phase2_SkewSymmetryCertifier,
    Phase2OrientationBridge,
    Phase3_HeytingLatticeDecider,
    PhaseHandoffCollapse,
    RiemannianInertiaAgent,
    RiemannianInertiaAgentError,
    SkewSignatureCollapse,
    ThermodynamicPassivityCollapse,
)


MomentumAuditData = ria.MomentumAuditData
GyroscopicSynthesisData = ria.GyroscopicSynthesisData
ThermodynamicVetoData = ria.ThermodynamicVetoData


# ══════════════════════════════════════════════════════════════════════════════
# §T0. CONSTANTES, FÁBRICAS, MOTORES FALSOS Y FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
_SEED_METRIC: Final[int] = 7
_SEED_VECTOR: Final[int] = 11
_SEED_SKEW: Final[int] = 13
_SEED_VORTICITY: Final[int] = 17
_SEED_GRADIENT: Final[int] = 19

_DIMS_CANONICAL: Final[tuple[int, ...]] = (1, 2, 3, 4)
_ATOL_STRUCT: Final[float] = 1.0e-10

_COHERENT = InertialHeytingVerdict.COHERENT
_DEGRADED = InertialHeytingVerdict.DEGRADED
_VETOED = InertialHeytingVerdict.VETOED


def _has_real_motor() -> bool:
    candidate = getattr(ria, "RiemannianInertiaModulator", None)
    return isinstance(candidate, type) and callable(
        getattr(candidate, "execute_phase1", None)
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_spd_pair(
    dimension: int,
    condition: float = 10.0,
    seed: int = _SEED_METRIC,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Par métrico espectralmente exacto G ≻ 0, G⁻¹ = Q diag(λ⁻¹) Qᵀ."""
    if dimension < 1:
        raise ValueError("dimension debe ser ≥ 1.")
    if condition < 1.0:
        raise ValueError("condition debe ser ≥ 1.")
    if dimension == 1:
        metric = np.array([[float(condition)]], dtype=np.float64)
        return metric, np.array([[1.0 / float(condition)]], dtype=np.float64)

    generator = _rng(seed)
    raw, _ = np.linalg.qr(generator.standard_normal((dimension, dimension)))
    if float(np.linalg.det(raw)) < 0.0:
        raw[:, 0] *= -1.0
    eigenvalues = np.geomspace(1.0, float(condition), num=dimension)
    metric = (raw * eigenvalues) @ raw.T
    inverse = (raw * (1.0 / eigenvalues)) @ raw.T
    metric = np.ascontiguousarray(0.5 * (metric + metric.T), dtype=np.float64)
    inverse = np.ascontiguousarray(0.5 * (inverse + inverse.T), dtype=np.float64)
    return metric, inverse


def make_skew_matrix(
    dimension: int,
    seed: int = _SEED_SKEW,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    if dimension == 1:
        return np.zeros((1, 1), dtype=np.float64)
    generator = _rng(seed)
    raw = generator.standard_normal((dimension, dimension))
    return np.ascontiguousarray(
        float(scale) * 0.5 * (raw - raw.T),
        dtype=np.float64,
    )


def make_vector(
    dimension: int,
    seed: int = _SEED_VECTOR,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    return np.ascontiguousarray(
        float(scale) * _rng(seed).standard_normal(dimension),
        dtype=np.float64,
    )


def make_momentum_audit(
    *,
    dimension: int = 3,
    momentum_norm: float = 1.0,
    is_bounded: bool = True,
    metric_condition_number: float = 2.0,
    inverse_consistency_residual: float = 0.0,
    pairing_residual: float = 0.0,
    musical_roundtrip_residual: float = 0.0,
    kinetic_energy_primal: float | None = None,
    kinetic_energy_dual: float | None = None,
    spectral_minimum: float | None = None,
    spectral_maximum: float | None = None,
    covariant_momentum: NDArray[np.float64] | None = None,
    reconstructed_velocity: NDArray[np.float64] | None = None,
    wilkinson_bound: float = 1.0e-8,
    dual_pairing: float | None = None,
) -> Any:
    """DTO coherente de Fase 1 del motor (espectralmente consistente)."""
    if covariant_momentum is None:
        covariant_momentum = make_vector(dimension, seed=3, scale=0.5)
    if reconstructed_velocity is None:
        reconstructed_velocity = np.array(covariant_momentum, dtype=np.float64, copy=True)
    energy = float(momentum_norm) ** 2
    if kinetic_energy_primal is None:
        kinetic_energy_primal = energy
    if kinetic_energy_dual is None:
        kinetic_energy_dual = energy
    if dual_pairing is None:
        dual_pairing = energy
    if spectral_minimum is None:
        spectral_minimum = 1.0
    if spectral_maximum is None:
        spectral_maximum = max(float(metric_condition_number), spectral_minimum)
    return MomentumAuditData(
        covariant_momentum=covariant_momentum,
        reconstructed_velocity=reconstructed_velocity,
        momentum_norm=float(momentum_norm),
        kinetic_energy_primal=float(kinetic_energy_primal),
        kinetic_energy_dual=float(kinetic_energy_dual),
        dual_pairing=float(dual_pairing),
        pairing_residual=float(pairing_residual),
        musical_roundtrip_residual=float(musical_roundtrip_residual),
        is_bounded=is_bounded,
        metric_condition_number=float(metric_condition_number),
        inverse_consistency_residual=float(inverse_consistency_residual),
        wilkinson_bound=float(wilkinson_bound),
        spectral_minimum=float(spectral_minimum),
        spectral_maximum=float(spectral_maximum),
    )


def make_synthesis_data(
    *,
    dimension: int = 3,
    is_strictly_skew: bool = True,
    antisymmetry_residual: float = 0.0,
    relative_skew_residual: float = 0.0,
    vorticity_projection_residual: float = 0.0,
    wedge_gram_residual: float = 0.0,
    skew_numerical_rank: int = 2,
    rank_is_even: bool = True,
    tensor: NDArray[np.float64] | None = None,
    gyroscopic_frobenius_norm: float | None = None,
) -> Any:
    """DTO coherente de Fase 2 del motor."""
    if tensor is None:
        tensor = make_skew_matrix(dimension, seed=5)
    if gyroscopic_frobenius_norm is None:
        gyroscopic_frobenius_norm = float(np.linalg.norm(tensor, ord="fro"))
    return GyroscopicSynthesisData(
        skew_symmetric_tensor=tensor,
        vorticity_two_form=make_skew_matrix(dimension, seed=6),
        omega_vector=make_vector(dimension, seed=8),
        antisymmetry_residual=float(antisymmetry_residual),
        relative_skew_residual=float(relative_skew_residual),
        vorticity_projection_residual=float(vorticity_projection_residual),
        gyroscopic_frobenius_norm=float(gyroscopic_frobenius_norm),
        wedge_gram_residual=float(wedge_gram_residual),
        skew_numerical_rank=int(skew_numerical_rank),
        rank_is_even=bool(rank_is_even),
        is_strictly_skew=is_strictly_skew,
    )


def make_veto_data(
    *,
    dimension: int = 3,
    nilpotent_work_residual: float = 0.0,
    is_symplectically_passive: bool = True,
    work_tolerance: float = 1.0e-8,
    dirac_symmetric_residual: float = 0.0,
    pairwise_work_residual: float = 0.0,
    liouville_probe_residual: float = 0.0,
    relative_skew_residual: float = 0.0,
    effective_dirac_matrix: NDArray[np.float64] | None = None,
) -> Any:
    """DTO coherente de Fase 3 del motor."""
    if effective_dirac_matrix is None:
        effective_dirac_matrix = make_skew_matrix(dimension, seed=9)
    return ThermodynamicVetoData(
        effective_dirac_matrix=effective_dirac_matrix,
        nilpotent_work_residual=float(nilpotent_work_residual),
        dirac_symmetric_residual=float(dirac_symmetric_residual),
        work_tolerance=float(work_tolerance),
        is_symplectically_passive=is_symplectically_passive,
        pairwise_work_residual=float(pairwise_work_residual),
        liouville_probe_residual=float(liouville_probe_residual),
        relative_skew_residual=float(relative_skew_residual),
    )


def _join(*verdicts: InertialHeytingVerdict) -> InertialHeytingVerdict:
    acc = _COHERENT
    for verdict in verdicts:
        acc = acc.join(verdict)
    return acc


def make_phase1_bridge(
    *,
    momentum_data: Any | None = None,
    momentum_bound_verdict: InertialHeytingVerdict = _COHERENT,
    metric_condition_verdict: InertialHeytingVerdict = _COHERENT,
    inverse_consistency_verdict: InertialHeytingVerdict = _COHERENT,
    pairing_verdict: InertialHeytingVerdict = _COHERENT,
    musical_roundtrip_verdict: InertialHeytingVerdict = _COHERENT,
    spectral_gap_verdict: InertialHeytingVerdict = _COHERENT,
    momentum_margin: float = 0.99,
    pairing_residual: float = 0.0,
    musical_roundtrip_residual: float = 0.0,
    kinetic_energy_primal: float = 1.0,
    kinetic_energy_dual: float = 1.0,
    spectral_minimum: float = 1.0,
    spectral_maximum: float = 2.0,
) -> Phase1ObservationBridge:
    """Puente de Fase 1 con join de Heyting internamente consistente."""
    if momentum_data is None:
        momentum_data = make_momentum_audit()
    liouville = _join(
        momentum_bound_verdict,
        metric_condition_verdict,
        inverse_consistency_verdict,
        pairing_verdict,
        musical_roundtrip_verdict,
        spectral_gap_verdict,
    )
    return Phase1ObservationBridge(
        momentum_data=momentum_data,
        liouville_verdict=liouville,
        momentum_bound_verdict=momentum_bound_verdict,
        metric_condition_verdict=metric_condition_verdict,
        inverse_consistency_verdict=inverse_consistency_verdict,
        pairing_verdict=pairing_verdict,
        musical_roundtrip_verdict=musical_roundtrip_verdict,
        spectral_gap_verdict=spectral_gap_verdict,
        momentum_margin=momentum_margin,
        pairing_residual=pairing_residual,
        musical_roundtrip_residual=musical_roundtrip_residual,
        kinetic_energy_primal=kinetic_energy_primal,
        kinetic_energy_dual=kinetic_energy_dual,
        spectral_minimum=spectral_minimum,
        spectral_maximum=spectral_maximum,
    )


def make_phase2_bridge(
    *,
    phase1_bridge: Phase1ObservationBridge | None = None,
    synthesis_data: Any | None = None,
    antisymmetry_verdict: InertialHeytingVerdict = _COHERENT,
    vorticity_projection_verdict: InertialHeytingVerdict = _COHERENT,
    gram_identity_verdict: InertialHeytingVerdict = _COHERENT,
    even_rank_verdict: InertialHeytingVerdict = _COHERENT,
    gauge_signature_verdict: InertialHeytingVerdict = _COHERENT,
    relative_antisymmetry_residual: float = 0.0,
    vorticity_projection_ratio: float = 0.0,
    wedge_gram_residual: float = 0.0,
    gauge_skew_residual: float = 0.0,
    skew_numerical_rank: int = 2,
    rank_is_even: bool = True,
) -> Phase2OrientationBridge:
    """Puente de Fase 2 con join de Heyting internamente consistente."""
    if phase1_bridge is None:
        phase1_bridge = make_phase1_bridge()
    if synthesis_data is None:
        synthesis_data = make_synthesis_data()
    skew = _join(
        antisymmetry_verdict,
        vorticity_projection_verdict,
        gram_identity_verdict,
        even_rank_verdict,
        gauge_signature_verdict,
    )
    return Phase2OrientationBridge(
        phase1_bridge=phase1_bridge,
        synthesis_data=synthesis_data,
        skew_verdict=skew,
        antisymmetry_verdict=antisymmetry_verdict,
        vorticity_projection_verdict=vorticity_projection_verdict,
        gram_identity_verdict=gram_identity_verdict,
        even_rank_verdict=even_rank_verdict,
        gauge_signature_verdict=gauge_signature_verdict,
        relative_antisymmetry_residual=relative_antisymmetry_residual,
        vorticity_projection_ratio=vorticity_projection_ratio,
        wedge_gram_residual=wedge_gram_residual,
        gauge_skew_residual=gauge_skew_residual,
        skew_numerical_rank=skew_numerical_rank,
        rank_is_even=rank_is_even,
    )


class RecordingMotor:
    """Motor falso que registra el orden OODA y entrega certificados prefijados."""

    def __init__(
        self,
        *,
        phase1: Any | None = None,
        phase2: Any | None = None,
        phase3: Any | None = None,
    ) -> None:
        self.phase1 = phase1 if phase1 is not None else make_momentum_audit()
        self.phase2 = phase2 if phase2 is not None else make_synthesis_data()
        self.phase3 = phase3 if phase3 is not None else make_veto_data()
        self.calls: list[str] = []
        self.phase1_kwargs: dict[str, Any] | None = None
        self.phase2_kwargs: dict[str, Any] | None = None
        self.phase3_kwargs: dict[str, Any] | None = None

    def execute_phase1(self, **kwargs: Any) -> Any:
        self.calls.append("phase1")
        self.phase1_kwargs = dict(kwargs)
        return self.phase1

    def execute_phase2(self, **kwargs: Any) -> Any:
        self.calls.append("phase2")
        self.phase2_kwargs = dict(kwargs)
        return self.phase2

    def execute_phase3(self, **kwargs: Any) -> Any:
        self.calls.append("phase3")
        self.phase3_kwargs = dict(kwargs)
        return self.phase3


def coherent_payload(dimension: int = 3) -> dict[str, NDArray[np.float64]]:
    metric, inverse = make_spd_pair(dimension, condition=8.0, seed=_SEED_METRIC)
    return {
        "q_dot": make_vector(dimension, seed=_SEED_VECTOR, scale=0.75),
        "grad_H": make_vector(dimension, seed=_SEED_GRADIENT, scale=1.25),
        "G_tensor": metric,
        "G_inv": inverse,
        "J_base": make_skew_matrix(dimension, seed=_SEED_SKEW, scale=0.8),
        "vorticity_matrix": make_skew_matrix(
            dimension, seed=_SEED_VORTICITY, scale=1.1
        ),
    }


@pytest.fixture(scope="module")
def auditor() -> Phase1_LiouvilleVolumeAuditor:
    return Phase1_LiouvilleVolumeAuditor()


@pytest.fixture(scope="module")
def certifier() -> Phase2_SkewSymmetryCertifier:
    return Phase2_SkewSymmetryCertifier()


@pytest.fixture(scope="module")
def decider() -> Phase3_HeytingLatticeDecider:
    return Phase3_HeytingLatticeDecider()


@pytest.fixture
def coherent_agent() -> RiemannianInertiaAgent:
    return RiemannianInertiaAgent(RecordingMotor())


# ══════════════════════════════════════════════════════════════════════════════
# §T1. EXPORTACIONES, ÁLGEBRA DE HEYTING Y JERARQUÍA
# ══════════════════════════════════════════════════════════════════════════════
class TestExportsAndHierarchy:
    """Contrato público, torre de excepciones y anidamiento de fases."""

    def test_version_is_canonical(self) -> None:
        assert ria.__version__ == "4.0.0-Topos-Heyting-Liouville-Gauge-Strict"

    def test_all_exports_are_importable(self) -> None:
        for name in ria.__all__:
            assert hasattr(ria, name), f"{name} falta en el módulo."
            assert getattr(ria, name) is not None

    def test_exception_tower(self) -> None:
        assert issubclass(RiemannianInertiaAgentError, ria.TopologicalInvariantError)
        for cls in (
            MotorContractError,
            PhaseHandoffCollapse,
            LiouvilleVolumeCollapse,
            MetricConditionCollapse,
            InverseCoherenceCollapse,
            DualPairingCollapse,
            MusicalIsomorphismCollapse,
            SkewSignatureCollapse,
            ExteriorAlgebraCollapse,
            ThermodynamicPassivityCollapse,
            HeytingLatticeVeto,
        ):
            assert issubclass(cls, RiemannianInertiaAgentError)

    def test_inheritance_is_nested_by_phase(self) -> None:
        assert issubclass(Phase2_SkewSymmetryCertifier, Phase1_LiouvilleVolumeAuditor)
        assert issubclass(Phase3_HeytingLatticeDecider, Phase2_SkewSymmetryCertifier)
        assert issubclass(RiemannianInertiaAgent, Phase3_HeytingLatticeDecider)
        assert issubclass(RiemannianInertiaAgent, ria.Morphism)

    def test_policy_constants_are_ordered(self) -> None:
        assert 0.0 < ria._MOMENTUM_SOFT_LIMIT < ria._MOMENTUM_HARD_LIMIT
        assert 1.0 <= ria._CONDITION_SOFT_MAX < ria._CONDITION_HARD_MAX
        assert 0.0 < ria._INVERSE_RESIDUAL_SOFT_MAX < ria._INVERSE_RESIDUAL_HARD_MAX
        assert 0.0 < ria._SKEW_SOFT_RELATIVE_TOLERANCE < ria._SKEW_HARD_RELATIVE_TOLERANCE
        assert (
            0.0
            < ria._VORTICITY_PROJECTION_SOFT_RATIO
            < ria._VORTICITY_PROJECTION_HARD_RATIO
        )
        assert (
            0.0
            < ria._DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE
            < ria._DIRAC_SYMMETRIC_HARD_RELATIVE_TOLERANCE
        )
        assert 0.0 < ria._PAIRING_RESIDUAL_SOFT_MAX < ria._PAIRING_RESIDUAL_HARD_MAX
        assert 0.0 < ria._MUSICAL_ROUNDTRIP_SOFT_MAX < ria._MUSICAL_ROUNDTRIP_HARD_MAX
        assert 0.0 < ria._GRAM_RESIDUAL_SOFT_MAX < ria._GRAM_RESIDUAL_HARD_MAX
        assert (
            0.0
            < ria._GAUGE_SKEW_SOFT_RELATIVE_TOLERANCE
            < ria._GAUGE_SKEW_HARD_RELATIVE_TOLERANCE
        )
        assert 0.0 < ria._SPECTRAL_FLOOR_RATIO_HARD < ria._SPECTRAL_FLOOR_RATIO_SOFT
        assert ria._WORK_SOFT_ABSOLUTE_TOLERANCE > 0.0
        assert ria._MACHINE_EPSILON > 0.0


class TestHeytingAlgebraOmega3:
    r"""
    Ω₃ es una cadena, luego un álgebra de Heyting:

        a ⊔ b = max,   a ⊓ b = min,
        a ⇒ b = ⊤ si a ≤ b, si no b,
        ¬a = a ⇒ ⊥.
    """

    def test_partial_order_of_severity(self) -> None:
        assert _COHERENT.value < _DEGRADED.value < _VETOED.value
        assert _COHERENT <= _DEGRADED <= _VETOED

    def test_join_is_max_and_commutative(self) -> None:
        assert _COHERENT.join(_DEGRADED) is _DEGRADED
        assert _DEGRADED.join(_COHERENT) is _DEGRADED
        assert _DEGRADED.join(_VETOED) is _VETOED
        assert _VETOED.join(_COHERENT) is _VETOED
        assert _COHERENT.join(_COHERENT) is _COHERENT

    def test_meet_is_min_and_commutative(self) -> None:
        assert _COHERENT.meet(_DEGRADED) is _COHERENT
        assert _DEGRADED.meet(_VETOED) is _DEGRADED
        assert _VETOED.meet(_COHERENT) is _COHERENT
        assert _VETOED.meet(_VETOED) is _VETOED

    def test_join_and_meet_are_associative(self) -> None:
        assert _COHERENT.join(_DEGRADED).join(_VETOED) is _VETOED
        assert _COHERENT.join(_DEGRADED.join(_VETOED)) is _VETOED
        assert _VETOED.meet(_DEGRADED).meet(_COHERENT) is _COHERENT
        assert _VETOED.meet(_DEGRADED.meet(_COHERENT)) is _COHERENT

    @pytest.mark.parametrize("antecedent", list(InertialHeytingVerdict))
    @pytest.mark.parametrize("consequent", list(InertialHeytingVerdict))
    def test_implication_table(
        self,
        antecedent: InertialHeytingVerdict,
        consequent: InertialHeytingVerdict,
    ) -> None:
        implied = antecedent.implies(consequent)
        if antecedent.value <= consequent.value:
            assert implied is _VETOED
        else:
            assert implied is consequent

    def test_negation_is_intuitionistic(self) -> None:
        # ¬⊥ = ⊤,  ¬a = ⊥ para a ≠ ⊥.
        assert _COHERENT.negate() is _VETOED
        assert _DEGRADED.negate() is _COHERENT
        assert _VETOED.negate() is _COHERENT

    def test_double_negation_does_not_recover_degraded(self) -> None:
        """Testigo de que Ω₃ no es Booleana: ¬¬DEGRADED = VETOED ≠ DEGRADED."""
        assert _DEGRADED.negate().negate() is _VETOED
        assert _DEGRADED.negate().negate() is not _DEGRADED
        assert _COHERENT.negate().negate() is _COHERENT
        assert _VETOED.negate().negate() is _VETOED

    def test_is_terminal_only_for_top(self) -> None:
        assert _COHERENT.is_terminal is False
        assert _DEGRADED.is_terminal is False
        assert _VETOED.is_terminal is True
        assert InertialHeytingVerdict(2).is_terminal is True

    def test_join_and_meet_reject_foreign_types(self) -> None:
        with pytest.raises(TypeError):
            _COHERENT.join("COHERENT")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _COHERENT.meet(0)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _COHERENT.implies(object())  # type: ignore[arg-type]

    def test_auditor_join_of_empty_collection_is_bottom(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._heyting_join(()) is _COHERENT

    def test_auditor_meet_of_empty_collection_is_bottom_by_policy(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        # Neutro algebraico de ⊓ es ⊤, pero una colección vacía se
        # interpreta como ausencia de evidencia y colapsa a ⊥.
        assert auditor._heyting_meet(()) is _COHERENT

    def test_auditor_join_and_meet_on_mixed_verdicts(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._heyting_join((_COHERENT, _DEGRADED, _COHERENT)) is _DEGRADED
        assert auditor._heyting_meet((_VETOED, _DEGRADED, _VETOED)) is _DEGRADED

    def test_auditor_join_rejects_non_verdict(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="InertialHeytingVerdict"):
            auditor._heyting_join((_COHERENT, "VETOED"))  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# §T2. DTOs INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════
class TestImmutableGovernanceDTOs:
    """Los puentes categóricos son frozen+slots y no admiten atributos extra."""

    def test_phase1_bridge_is_frozen_and_slotted(self) -> None:
        bridge = make_phase1_bridge()
        with pytest.raises(FrozenInstanceError):
            bridge.momentum_margin = 0.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            bridge.not_a_field = True  # type: ignore[attr-defined]

    def test_phase2_bridge_is_frozen_and_slotted(self) -> None:
        bridge = make_phase2_bridge()
        with pytest.raises(FrozenInstanceError):
            bridge.skew_numerical_rank = -1  # type: ignore[misc]
        with pytest.raises(AttributeError):
            bridge.not_a_field = True  # type: ignore[attr-defined]

    def test_governance_state_is_frozen_and_slotted(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        state = decider.execute_phase3(
            make_phase2_bridge(),
            make_veto_data(),
            raise_on_veto=True,
        )
        with pytest.raises(FrozenInstanceError):
            state.is_epistemologically_valid = False  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.not_a_field = True  # type: ignore[attr-defined]

    def test_phase1_factory_join_matches_stored_verdict(self) -> None:
        bridge = make_phase1_bridge(
            momentum_bound_verdict=_DEGRADED,
            pairing_verdict=_VETOED,
        )
        assert bridge.liouville_verdict is _VETOED

    def test_phase2_factory_join_matches_stored_verdict(self) -> None:
        bridge = make_phase2_bridge(
            antisymmetry_verdict=_DEGRADED,
            gram_identity_verdict=_COHERENT,
        )
        assert bridge.skew_verdict is _DEGRADED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → PRIMITIVAS, CLASIFICADORES, EXECUTE Y HANDOFF Φ₁₂
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1ValidationPrimitives:
    """Granularidad de los coercers y extractores del auditor."""

    def test_as_finite_float_accepts_numeric_strings(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._as_finite_float("1.25", "x") == pytest.approx(1.25)

    @pytest.mark.parametrize("payload", [True, False, np.bool_(True)])
    def test_as_finite_float_rejects_booleans(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        payload: object,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="booleano"):
            auditor._as_finite_float(payload, "x")

    @pytest.mark.parametrize("payload", [math.nan, math.inf, -math.inf, object(), "nope"])
    def test_as_finite_float_rejects_non_real(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        payload: object,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError):
            auditor._as_finite_float(payload, "x")

    def test_nonnegative_and_positive_guards(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._as_nonnegative_finite_float(0.0, "x") == 0.0
        with pytest.raises(RiemannianInertiaAgentError, match="negativo"):
            auditor._as_nonnegative_finite_float(-1e-12, "x")
        assert auditor._as_positive_finite_float(1e-16, "x") > 0.0
        with pytest.raises(RiemannianInertiaAgentError, match="positivo"):
            auditor._as_positive_finite_float(0.0, "x")

    def test_as_bool_and_as_int_reject_cross_types(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._as_bool(True, "flag") is True
        with pytest.raises(RiemannianInertiaAgentError, match="booleano"):
            auditor._as_bool(1, "flag")
        assert auditor._as_int(np.int64(4), "n") == 4
        with pytest.raises(RiemannianInertiaAgentError, match="booleano"):
            auditor._as_int(True, "n")
        with pytest.raises(RiemannianInertiaAgentError, match="entero"):
            auditor._as_int(1.0, "n")

    def test_optional_positive_falls_back_on_invalid(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        class Payload:
            work_tolerance = -1.0

        assert auditor._get_optional_positive_finite_float(
            Payload(), "work_tolerance", 1.0e-10, "payload"
        ) == pytest.approx(1.0e-10)

    def test_optional_float_returns_default_when_missing_or_none(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        class Empty:
            pass

        class NoneField:
            residual = None

        assert auditor._get_optional_nonnegative_finite_float(
            Empty(), "residual", 0.5, "obj"
        ) == pytest.approx(0.5)
        assert auditor._get_optional_nonnegative_finite_float(
            NoneField(), "residual", 0.5, "obj"
        ) == pytest.approx(0.5)

    def test_required_attribute_missing_raises(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="no contiene"):
            auditor._get_required_attribute(object(), "momentum_norm", "audit")

    def test_finite_array_attribute_optional_and_required(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        class Empty:
            pass

        assert (
            auditor._validate_finite_array_attribute(
                Empty(), "tensor", "obj", required=False
            )
            is None
        )
        with pytest.raises(RiemannianInertiaAgentError, match="no contiene"):
            auditor._validate_finite_array_attribute(
                Empty(), "tensor", "obj", required=True
            )

    def test_finite_array_rejects_empty_and_non_finite(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        class Payload:
            def __init__(self, value: object) -> None:
                self.tensor = value

        with pytest.raises(RiemannianInertiaAgentError, match="vacío"):
            auditor._validate_finite_array_attribute(
                Payload(np.zeros((0,))), "tensor", "obj"
            )
        with pytest.raises(RiemannianInertiaAgentError, match="no finitos"):
            auditor._validate_finite_array_attribute(
                Payload(np.array([1.0, math.nan])), "tensor", "obj"
            )

    def test_scalar_array_is_reshaped_to_one_dimension(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        class Payload:
            tensor = np.float64(3.5)

        array = auditor._validate_finite_array_attribute(Payload(), "tensor", "obj")
        assert array is not None
        assert array.shape == (1,)
        assert array[0] == pytest.approx(3.5)

    def test_matrix_frobenius_norm_rejects_non_square(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="cuadrada"):
            auditor._matrix_frobenius_norm(
                np.zeros((2, 3), dtype=np.float64), "M"
            )

    def test_relative_residual_uses_unit_floor(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._relative_residual(0.5, 0.1) == pytest.approx(0.5)
        assert auditor._relative_residual(4.0, 2.0) == pytest.approx(2.0)

    def test_clamp01_projects_and_sanitizes_nan(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._clamp01(-1.0) == 0.0
        assert auditor._clamp01(2.0) == 1.0
        assert auditor._clamp01(0.3) == pytest.approx(0.3)
        assert auditor._clamp01(math.nan) == 0.0

    def test_validate_momentum_data_rejects_wrong_type(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="MomentumAuditData"):
            auditor._validate_momentum_data(object())  # type: ignore[arg-type]

    def test_validate_momentum_data_normalizes_condition_below_one(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        fields = auditor._validate_momentum_data(
            make_momentum_audit(metric_condition_number=0.25, spectral_maximum=1.0)
        )
        assert fields["metric_condition_number"] == pytest.approx(1.0)

    def test_validate_momentum_data_rejects_negative_norm(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="negativo"):
            auditor._validate_momentum_data(make_momentum_audit(momentum_norm=-0.1))


class TestPhase1Classifiers:
    """Clasificadores locales de Liouville, métrica, dualidad y espectro."""

    def test_momentum_bound_coherent_degraded_vetoed(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_momentum_bound(1.0, True) is _COHERENT
        assert (
            auditor._classify_momentum_bound(ria._MOMENTUM_SOFT_LIMIT + 1.0, True)
            is _DEGRADED
        )
        assert (
            auditor._classify_momentum_bound(ria._MOMENTUM_HARD_LIMIT + 1.0, True)
            is _VETOED
        )
        assert auditor._classify_momentum_bound(1.0, False) is _VETOED

    def test_metric_condition_thresholds(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_metric_condition(10.0) is _COHERENT
        assert (
            auditor._classify_metric_condition(ria._CONDITION_SOFT_MAX + 1.0)
            is _DEGRADED
        )
        assert (
            auditor._classify_metric_condition(ria._CONDITION_HARD_MAX + 1.0)
            is _VETOED
        )

    def test_inverse_consistency_thresholds(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_inverse_consistency(0.0) is _COHERENT
        assert (
            auditor._classify_inverse_consistency(ria._INVERSE_RESIDUAL_SOFT_MAX * 2.0)
            is _DEGRADED
        )
        assert (
            auditor._classify_inverse_consistency(ria._INVERSE_RESIDUAL_HARD_MAX * 2.0)
            is _VETOED
        )

    def test_dual_pairing_joins_residual_and_energy_gap(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_dual_pairing(0.0, 1.0, 1.0) is _COHERENT
        assert (
            auditor._classify_dual_pairing(ria._PAIRING_RESIDUAL_SOFT_MAX * 2.0, 1.0, 1.0)
            is _DEGRADED
        )
        assert (
            auditor._classify_dual_pairing(ria._PAIRING_RESIDUAL_HARD_MAX * 2.0, 1.0, 1.0)
            is _VETOED
        )
        # Hueco primal/dual relativo grande, residual nulo.
        assert auditor._classify_dual_pairing(0.0, 1.0, 10.0) is _VETOED

    def test_musical_roundtrip_thresholds(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_musical_roundtrip(0.0) is _COHERENT
        assert (
            auditor._classify_musical_roundtrip(ria._MUSICAL_ROUNDTRIP_SOFT_MAX * 2.0)
            is _DEGRADED
        )
        assert (
            auditor._classify_musical_roundtrip(ria._MUSICAL_ROUNDTRIP_HARD_MAX * 2.0)
            is _VETOED
        )

    def test_spectral_gap_rejects_non_positive_or_inverted_spectrum(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._classify_spectral_gap(1.0, 4.0, 4.0) is _COHERENT
        assert auditor._classify_spectral_gap(-1.0, 4.0, 4.0) is _VETOED
        assert auditor._classify_spectral_gap(1.0, 0.0, 4.0) is _VETOED
        assert auditor._classify_spectral_gap(5.0, 2.0, 1.0) is _VETOED

    def test_spectral_gap_floor_and_condition_mismatch(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        # λ_min / λ_max por debajo del umbral duro.
        assert (
            auditor._classify_spectral_gap(1.0e-20, 1.0, 1.0e20) is _VETOED
        )
        # κ espectral discrepa groseramente del κ certificado.
        assert auditor._classify_spectral_gap(1.0, 2.0, 100.0) is _VETOED

    def test_momentum_margin_formula(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        assert auditor._compute_momentum_margin(0.0) == pytest.approx(1.0)
        assert auditor._compute_momentum_margin(ria._MOMENTUM_HARD_LIMIT) == pytest.approx(
            0.0
        )
        assert auditor._compute_momentum_margin(ria._MOMENTUM_HARD_LIMIT * 2.0) == 0.0
        half = auditor._compute_momentum_margin(0.5 * ria._MOMENTUM_HARD_LIMIT)
        assert half == pytest.approx(0.5)


class TestPhase1ExecuteAndCertificates:
    """execute_phase1 proyecta el certificado del motor al retículo Ω₃."""

    def test_coherent_audit_yields_coherent_bridge(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        bridge = auditor.execute_phase1(make_momentum_audit())
        assert isinstance(bridge, Phase1ObservationBridge)
        assert bridge.liouville_verdict is _COHERENT
        assert bridge.momentum_bound_verdict is _COHERENT
        assert bridge.metric_condition_verdict is _COHERENT
        assert bridge.inverse_consistency_verdict is _COHERENT
        assert bridge.pairing_verdict is _COHERENT
        assert bridge.musical_roundtrip_verdict is _COHERENT
        assert bridge.spectral_gap_verdict is _COHERENT
        assert 0.0 <= bridge.momentum_margin <= 1.0

    def test_unbounded_flag_vetoes_even_if_norm_is_small(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        bridge = auditor.execute_phase1(
            make_momentum_audit(momentum_norm=1.0, is_bounded=False)
        )
        assert bridge.momentum_bound_verdict is _VETOED
        assert bridge.liouville_verdict is _VETOED

    def test_soft_momentum_degrades_without_veto(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        bridge = auditor.execute_phase1(
            make_momentum_audit(momentum_norm=ria._MOMENTUM_SOFT_LIMIT * 2.0)
        )
        assert bridge.momentum_bound_verdict is _DEGRADED
        assert bridge.liouville_verdict is _DEGRADED

    def test_join_is_supremum_of_granular_classifiers(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        bridge = auditor.execute_phase1(
            make_momentum_audit(
                momentum_norm=ria._MOMENTUM_SOFT_LIMIT * 2.0,
                inverse_consistency_residual=ria._INVERSE_RESIDUAL_HARD_MAX * 2.0,
            )
        )
        assert bridge.momentum_bound_verdict is _DEGRADED
        assert bridge.inverse_consistency_verdict is _VETOED
        assert bridge.liouville_verdict is _VETOED

    def test_execute_phase1_rejects_non_audit(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="MomentumAuditData"):
            auditor.execute_phase1(object())  # type: ignore[arg-type]


class TestPhase1HandoffPhi12:
    """
    Definición formal final de la Fase 1.

    Φ₁₂ es el último morfismo de esta fase y el dominio sobre el que
    la Fase 2 certifica la firma métrica.
    """

    def test_handoff_is_identity_on_consistent_bridge(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        produced = auditor.execute_phase1(make_momentum_audit())
        handed = auditor.handoff_phase1_to_phase2(produced)
        assert handed is produced
        assert handed.liouville_verdict is _COHERENT

    def test_handoff_rejects_wrong_type(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        with pytest.raises(PhaseHandoffCollapse, match="Phase1ObservationBridge"):
            auditor.handoff_phase1_to_phase2(object())  # type: ignore[arg-type]

    def test_handoff_detects_lattice_inconsistency(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        broken = Phase1ObservationBridge(
            momentum_data=make_momentum_audit(),
            liouville_verdict=_COHERENT,
            momentum_bound_verdict=_VETOED,
        )
        with pytest.raises(PhaseHandoffCollapse, match="Inconsistencia"):
            auditor.handoff_phase1_to_phase2(broken)

    def test_handoff_rejects_negative_margin(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        broken = Phase1ObservationBridge(
            momentum_data=make_momentum_audit(),
            liouville_verdict=_COHERENT,
            momentum_margin=-0.1,
        )
        with pytest.raises(RiemannianInertiaAgentError, match="negativo"):
            auditor.handoff_phase1_to_phase2(broken)

    def test_handoff_does_not_collapse_on_vetoed_but_consistent_bridge(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
    ) -> None:
        """La política de veto se reserva a la Fase 3 / fail-fast."""
        vetoed = make_phase1_bridge(momentum_bound_verdict=_VETOED)
        handed = auditor.handoff_phase1_to_phase2(vetoed)
        assert handed.liouville_verdict is _VETOED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → CONTINUACIÓN DE Φ₁₂, FIRMA MÉTRICA, CALIBRE Y HANDOFF Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2ReceivesPhase1Handoff:
    """El primer método de la Fase 2 es la continuación literal de Φ₁₂."""

    def test_receive_equals_phase1_handoff(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        bridge = auditor.execute_phase1(make_momentum_audit())
        via_phi = auditor.handoff_phase1_to_phase2(bridge)
        via_phase2 = certifier._receive_certified_observation(bridge)
        assert via_phi is via_phase2

    def test_receive_rejects_inconsistent_bridge(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        broken = Phase1ObservationBridge(
            momentum_data=make_momentum_audit(),
            liouville_verdict=_DEGRADED,
            momentum_bound_verdict=_COHERENT,
        )
        with pytest.raises(PhaseHandoffCollapse):
            certifier._receive_certified_observation(broken)


class TestPhase2ClassifiersAndGauge:
    """Antisimetría, vorticidad, Gram, rango par y G-antisimetría."""

    def test_antisymmetry_vetoes_non_strict_flag(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        residual, verdict = certifier._classify_antisymmetry(0.0, False, 1.0, 0.0)
        assert verdict is _VETOED
        assert residual == pytest.approx(0.0)

    def test_antisymmetry_uses_worst_of_computed_and_certified(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        # residual abs / max(1, ‖W‖) = 1e-4; certificado = 0.
        residual, verdict = certifier._classify_antisymmetry(
            1.0e-4, True, 1.0, 0.0
        )
        assert residual == pytest.approx(1.0e-4)
        assert verdict is _VETOED

        residual, verdict = certifier._classify_antisymmetry(
            0.0, True, 1.0, ria._SKEW_SOFT_RELATIVE_TOLERANCE * 2.0
        )
        assert verdict is _DEGRADED

        residual, verdict = certifier._classify_antisymmetry(0.0, True, 1.0, 0.0)
        assert verdict is _COHERENT

    def test_vorticity_is_coherent_when_gyroscopic_norm_vanishes(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        ratio, verdict = certifier._classify_vorticity_projection(10.0, 0.0)
        assert ratio == pytest.approx(0.0)
        assert verdict is _COHERENT

    def test_vorticity_thresholds(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        _, coherent = certifier._classify_vorticity_projection(1.0e-6, 1.0)
        assert coherent is _COHERENT
        _, degraded = certifier._classify_vorticity_projection(
            ria._VORTICITY_PROJECTION_SOFT_RATIO * 2.0, 1.0
        )
        assert degraded is _DEGRADED
        _, vetoed = certifier._classify_vorticity_projection(
            ria._VORTICITY_PROJECTION_HARD_RATIO * 2.0, 1.0
        )
        assert vetoed is _VETOED

    def test_gram_identity_thresholds(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        assert certifier._classify_gram_identity(0.0) is _COHERENT
        assert (
            certifier._classify_gram_identity(ria._GRAM_RESIDUAL_SOFT_MAX * 2.0)
            is _DEGRADED
        )
        assert (
            certifier._classify_gram_identity(ria._GRAM_RESIDUAL_HARD_MAX * 2.0)
            is _VETOED
        )

    def test_even_rank_degrades_but_never_vetoes(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        assert certifier._classify_even_rank(2, True) is _COHERENT
        assert certifier._classify_even_rank(0, True) is _COHERENT
        assert certifier._classify_even_rank(1, True) is _DEGRADED
        assert certifier._classify_even_rank(2, False) is _DEGRADED

    def test_gauge_is_skipped_when_metric_or_tensor_missing(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        residual, verdict = certifier._classify_gauge_signature(None, np.eye(2))
        assert residual == pytest.approx(0.0)
        assert verdict is _COHERENT
        residual, verdict = certifier._classify_gauge_signature(np.eye(2), None)
        assert verdict is _COHERENT

    def test_gauge_vanishes_when_metric_is_conformal_to_identity(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        tensor = make_skew_matrix(3, seed=41, scale=1.3)
        metric = 4.0 * np.eye(3, dtype=np.float64)
        residual, verdict = certifier._classify_gauge_signature(tensor, metric)
        assert residual <= 1.0e-12
        assert verdict is _COHERENT

    def test_gauge_detects_euclidean_skew_that_is_not_g_skew(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        tensor = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        metric = np.diag([1.0, 100.0]).astype(np.float64)
        residual, verdict = certifier._classify_gauge_signature(tensor, metric)
        assert residual > ria._GAUGE_SKEW_HARD_RELATIVE_TOLERANCE
        assert verdict is _VETOED

    def test_gauge_rejects_dimension_mismatch(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="dimensión"):
            certifier._classify_gauge_signature(
                make_skew_matrix(3),
                np.eye(2, dtype=np.float64),
            )

    def test_optional_metric_validation(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        assert certifier._validate_optional_metric(None) is None
        metric = certifier._validate_optional_metric(np.eye(2, dtype=np.float64))
        assert metric is not None
        assert metric.shape == (2, 2)
        with pytest.raises(RiemannianInertiaAgentError, match="cuadrada"):
            certifier._validate_optional_metric(np.zeros((2, 3)))
        with pytest.raises(RiemannianInertiaAgentError, match="no finitos"):
            certifier._validate_optional_metric(
                np.array([[1.0, math.nan], [0.0, 1.0]])
            )


class TestPhase2ExecuteAndCertificates:
    """execute_phase2 certifica W a partir del puente de Fase 1."""

    def test_coherent_synthesis_without_metric_skips_gauge(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit())
        phase2 = certifier.execute_phase2(phase1, make_synthesis_data())
        assert isinstance(phase2, Phase2OrientationBridge)
        assert phase2.phase1_bridge is phase1 or (
            phase2.phase1_bridge.liouville_verdict is _COHERENT
        )
        assert phase2.skew_verdict is _COHERENT
        assert phase2.gauge_signature_verdict is _COHERENT
        assert phase2.gauge_skew_residual == pytest.approx(0.0)

    def test_execute_phase2_with_conformal_metric_is_gauge_coherent(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit(dimension=3))
        tensor = make_skew_matrix(3, seed=43)
        phase2 = certifier.execute_phase2(
            phase1,
            make_synthesis_data(dimension=3, tensor=tensor),
            G_tensor=3.0 * np.eye(3, dtype=np.float64),
        )
        assert phase2.gauge_signature_verdict is _COHERENT
        assert phase2.skew_verdict is _COHERENT

    def test_execute_phase2_flags_non_g_skew_tensor(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit(dimension=2))
        tensor = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        phase2 = certifier.execute_phase2(
            phase1,
            make_synthesis_data(dimension=2, tensor=tensor),
            G_tensor=np.diag([1.0, 80.0]).astype(np.float64),
        )
        assert phase2.gauge_signature_verdict is _VETOED
        assert phase2.skew_verdict is _VETOED

    def test_non_strict_certificate_vetoes_without_raising(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit())
        phase2 = certifier.execute_phase2(
            phase1, make_synthesis_data(is_strictly_skew=False)
        )
        assert phase2.antisymmetry_verdict is _VETOED
        assert phase2.skew_verdict is _VETOED

    def test_odd_rank_degrades_skew_verdict(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit())
        phase2 = certifier.execute_phase2(
            phase1,
            make_synthesis_data(skew_numerical_rank=1, rank_is_even=False),
        )
        assert phase2.even_rank_verdict is _DEGRADED
        assert phase2.skew_verdict is _DEGRADED

    def test_execute_phase2_rejects_non_synthesis(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit())
        with pytest.raises(RiemannianInertiaAgentError, match="GyroscopicSynthesisData"):
            certifier.execute_phase2(phase1, object())  # type: ignore[arg-type]

    def test_validate_synthesis_recomputes_norm_when_reported_zero(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        tensor = make_skew_matrix(3, seed=47, scale=2.0)
        fields = certifier._validate_synthesis_data(
            make_synthesis_data(tensor=tensor, gyroscopic_frobenius_norm=0.0)
        )
        assert fields["gyroscopic_frobenius_norm"] == pytest.approx(
            float(np.linalg.norm(tensor, ord="fro"))
        )

    def test_validate_synthesis_rejects_negative_rank(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="negativo"):
            certifier._validate_synthesis_data(
                make_synthesis_data(skew_numerical_rank=-1)
            )


class TestPhase2HandoffPhi23:
    """
    Definición formal final de la Fase 2.

    Φ₂₃ es el último morfismo de esta fase y el dominio sobre el que
    la Fase 3 calcula el supremo termodinámico.
    """

    def test_handoff_is_identity_on_consistent_bridge(
        self,
        auditor: Phase1_LiouvilleVolumeAuditor,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        phase1 = auditor.execute_phase1(make_momentum_audit())
        phase2 = certifier.execute_phase2(phase1, make_synthesis_data())
        handed = certifier.handoff_phase2_to_phase3(phase2)
        assert handed is phase2

    def test_handoff_rejects_wrong_type(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        with pytest.raises(PhaseHandoffCollapse, match="Phase2OrientationBridge"):
            certifier.handoff_phase2_to_phase3(object())  # type: ignore[arg-type]

    def test_handoff_detects_lattice_inconsistency(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        broken = Phase2OrientationBridge(
            phase1_bridge=make_phase1_bridge(),
            synthesis_data=make_synthesis_data(),
            skew_verdict=_COHERENT,
            antisymmetry_verdict=_VETOED,
        )
        with pytest.raises(PhaseHandoffCollapse, match="Inconsistencia"):
            certifier.handoff_phase2_to_phase3(broken)

    def test_handoff_rejects_negative_rank(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        broken = Phase2OrientationBridge(
            phase1_bridge=make_phase1_bridge(),
            synthesis_data=make_synthesis_data(),
            skew_verdict=_COHERENT,
            skew_numerical_rank=-3,
        )
        with pytest.raises(PhaseHandoffCollapse, match="negativo"):
            certifier.handoff_phase2_to_phase3(broken)

    def test_handoff_revalidates_nested_phase1_bridge(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        nested = Phase1ObservationBridge(
            momentum_data=make_momentum_audit(),
            liouville_verdict=_COHERENT,
            pairing_verdict=_VETOED,
        )
        broken = Phase2OrientationBridge(
            phase1_bridge=nested,
            synthesis_data=make_synthesis_data(),
            skew_verdict=_COHERENT,
        )
        with pytest.raises(PhaseHandoffCollapse, match="Inconsistencia"):
            certifier.handoff_phase2_to_phase3(broken)

    def test_handoff_does_not_collapse_on_vetoed_but_consistent_bridge(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
    ) -> None:
        vetoed = make_phase2_bridge(antisymmetry_verdict=_VETOED)
        handed = certifier.handoff_phase2_to_phase3(vetoed)
        assert handed.skew_verdict is _VETOED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → CONTINUACIÓN DE Φ₂₃, RETÍCULO TERMODINÁMICO Y COLAPSO
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3ReceivesPhase2Handoff:
    """El primer método de la Fase 3 es la continuación literal de Φ₂₃."""

    def test_receive_equals_phase2_handoff(
        self,
        certifier: Phase2_SkewSymmetryCertifier,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        bridge = make_phase2_bridge()
        via_phi = certifier.handoff_phase2_to_phase3(bridge)
        via_phase3 = decider._receive_certified_orientation(bridge)
        assert via_phi is via_phase3

    def test_receive_rejects_inconsistent_bridge(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        broken = Phase2OrientationBridge(
            phase1_bridge=make_phase1_bridge(),
            synthesis_data=make_synthesis_data(),
            skew_verdict=_DEGRADED,
            antisymmetry_verdict=_COHERENT,
        )
        with pytest.raises(PhaseHandoffCollapse):
            decider._receive_certified_orientation(broken)


class TestPhase3ClassifiersAndVetoData:
    """Pasividad, pares, sondas de Liouville y simetría de Dirac."""

    def test_work_passivity_table(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        assert decider._classify_work_passivity(0.0, True, 1.0e-8) is _COHERENT
        assert (
            decider._classify_work_passivity(
                ria._WORK_SOFT_ABSOLUTE_TOLERANCE * 2.0, True, 1.0e-8
            )
            is _DEGRADED
        )
        assert decider._classify_work_passivity(1.0e-6, True, 1.0e-8) is _VETOED
        assert decider._classify_work_passivity(0.0, False, 1.0e-8) is _VETOED

    def test_dirac_symmetry_uses_worst_of_computed_and_certified(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        residual, verdict = decider._classify_dirac_symmetry(0.0, 1.0, 0.0)
        assert residual == pytest.approx(0.0)
        assert verdict is _COHERENT

        residual, verdict = decider._classify_dirac_symmetry(
            0.0, 1.0, ria._DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE * 2.0
        )
        assert verdict is _DEGRADED

        residual, verdict = decider._classify_dirac_symmetry(1.0, 1.0, 0.0)
        assert residual == pytest.approx(1.0)
        assert verdict is _VETOED

    def test_work_margin_formula(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        assert decider._compute_work_margin(0.0, 1.0e-8) == pytest.approx(1.0)
        assert decider._compute_work_margin(1.0e-8, 1.0e-8) == pytest.approx(0.0)
        assert decider._compute_work_margin(1.0, 1.0e-8) == 0.0
        assert decider._compute_work_margin(2.5e-9, 1.0e-8) == pytest.approx(0.75)

    def test_validate_veto_data_rejects_wrong_type(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="ThermodynamicVetoData"):
            decider._validate_veto_data(object())  # type: ignore[arg-type]

    def test_validate_veto_data_falls_back_on_non_positive_tolerance(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        fields = decider._validate_veto_data(make_veto_data(work_tolerance=0.0))
        assert fields["work_tolerance"] == pytest.approx(ria._WORK_SOFT_ABSOLUTE_TOLERANCE)

    def test_validate_veto_data_computes_frobenius_of_dirac_matrix(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        matrix = make_skew_matrix(3, seed=53, scale=2.0)
        fields = decider._validate_veto_data(
            make_veto_data(effective_dirac_matrix=matrix)
        )
        assert fields["dirac_frobenius_norm"] == pytest.approx(
            float(np.linalg.norm(matrix, ord="fro"))
        )

    def test_collect_veto_sources_lists_terminal_subobjects(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        bridge = make_phase2_bridge(
            momentum_bound_verdict=_VETOED,
            antisymmetry_verdict=_VETOED,
        )
        sources = decider._collect_veto_sources(
            bridge,
            work_verdict=_VETOED,
            work_passivity_verdict=_VETOED,
            pairwise_work_verdict=_COHERENT,
            liouville_probe_verdict=_COHERENT,
            dirac_symmetry_verdict=_COHERENT,
        )
        assert "Liouville.momentum" in sources
        assert "Skew.antisymmetry" in sources
        assert "Work.passivity" in sources
        assert "Work.join" in sources
        assert "Work.pairwise" not in sources


class TestPhase3ExecuteLatticeAndCollapse:
    """execute_phase3 calcula v_final y, si procede, colapsa el retículo."""

    def test_coherent_pipeline_is_epistemologically_valid(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        state = decider.execute_phase3(
            make_phase2_bridge(),
            make_veto_data(),
            raise_on_veto=True,
        )
        assert isinstance(state, InertialGovernanceState)
        assert state.final_supremum_verdict is _COHERENT
        assert state.is_epistemologically_valid is True
        assert state.work_verdict is _COHERENT
        assert state.work_margin == pytest.approx(1.0)

    def test_degraded_work_propagates_to_final_supremum(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        state = decider.execute_phase3(
            make_phase2_bridge(),
            make_veto_data(
                nilpotent_work_residual=ria._WORK_SOFT_ABSOLUTE_TOLERANCE * 2.0,
                work_tolerance=1.0e-8,
            ),
            raise_on_veto=True,
        )
        assert state.work_passivity_verdict is _DEGRADED
        assert state.final_supremum_verdict is _DEGRADED
        assert state.is_epistemologically_valid is True

    def test_raise_on_veto_false_returns_invalid_state(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        state = decider.execute_phase3(
            make_phase2_bridge(momentum_bound_verdict=_VETOED),
            make_veto_data(),
            raise_on_veto=False,
        )
        assert state.final_supremum_verdict is _VETOED
        assert state.is_epistemologically_valid is False

    def test_final_supremum_is_join_of_the_three_phases(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        state = decider.execute_phase3(
            make_phase2_bridge(
                pairing_verdict=_DEGRADED,
                gram_identity_verdict=_COHERENT,
            ),
            make_veto_data(
                nilpotent_work_residual=ria._WORK_SOFT_ABSOLUTE_TOLERANCE * 2.0,
                work_tolerance=1.0e-8,
            ),
            raise_on_veto=True,
        )
        assert state.phase2_bridge.phase1_bridge.liouville_verdict is _DEGRADED
        assert state.work_verdict is _DEGRADED
        assert state.final_supremum_verdict is _DEGRADED

    @pytest.mark.parametrize(
        ("bridge_kwargs", "veto_kwargs", "expected"),
        [
            ({"momentum_bound_verdict": _VETOED}, {}, LiouvilleVolumeCollapse),
            ({"metric_condition_verdict": _VETOED}, {}, MetricConditionCollapse),
            ({"inverse_consistency_verdict": _VETOED}, {}, InverseCoherenceCollapse),
            ({"pairing_verdict": _VETOED}, {}, DualPairingCollapse),
            ({"musical_roundtrip_verdict": _VETOED}, {}, MusicalIsomorphismCollapse),
            ({"antisymmetry_verdict": _VETOED}, {}, SkewSignatureCollapse),
            ({"gauge_signature_verdict": _VETOED}, {}, SkewSignatureCollapse),
            ({"gram_identity_verdict": _VETOED}, {}, ExteriorAlgebraCollapse),
            (
                {},
                {"is_symplectically_passive": False},
                ThermodynamicPassivityCollapse,
            ),
            (
                {},
                {"nilpotent_work_residual": 1.0, "work_tolerance": 1.0e-8},
                ThermodynamicPassivityCollapse,
            ),
            (
                {},
                {"pairwise_work_residual": 1.0, "work_tolerance": 1.0e-8},
                ThermodynamicPassivityCollapse,
            ),
            (
                {},
                {"liouville_probe_residual": 1.0, "work_tolerance": 1.0e-8},
                ThermodynamicPassivityCollapse,
            ),
            (
                {},
                {"dirac_symmetric_residual": 1.0},
                ThermodynamicPassivityCollapse,
            ),
            ({"spectral_gap_verdict": _VETOED}, {}, HeytingLatticeVeto),
        ],
    )
    def test_raise_on_veto_selects_most_specific_exception(
        self,
        decider: Phase3_HeytingLatticeDecider,
        bridge_kwargs: dict[str, Any],
        veto_kwargs: dict[str, Any],
        expected: type[RiemannianInertiaAgentError],
    ) -> None:
        with pytest.raises(expected, match="Colapso de software"):
            decider.execute_phase3(
                make_phase2_bridge(**bridge_kwargs),
                make_veto_data(**veto_kwargs),
                raise_on_veto=True,
            )

    def test_exception_message_enumerates_veto_sources(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        with pytest.raises(LiouvilleVolumeCollapse, match="Liouville.momentum") as captured:
            decider.execute_phase3(
                make_phase2_bridge(momentum_bound_verdict=_VETOED),
                make_veto_data(),
                raise_on_veto=True,
            )
        assert "Transacción aniquilada en RAM" in str(captured.value)
        assert "Veredicto Supremo = VETOED" in str(captured.value)

    def test_execute_phase3_rejects_non_veto_payload(
        self,
        decider: Phase3_HeytingLatticeDecider,
    ) -> None:
        with pytest.raises(RiemannianInertiaAgentError, match="ThermodynamicVetoData"):
            decider.execute_phase3(
                make_phase2_bridge(),
                object(),  # type: ignore[arg-type]
                raise_on_veto=False,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR → CONTRATO DEL MOTOR, PAYLOAD Y CICLO OODA
# ══════════════════════════════════════════════════════════════════════════════
class TestOrchestratorMotorContract:
    """El agente exige un motor funtorial de tres fases ejecutables."""

    def test_constructor_rejects_none(self) -> None:
        with pytest.raises(MotorContractError, match="None"):
            RiemannianInertiaAgent(None)  # type: ignore[arg-type]

    def test_constructor_rejects_missing_phase_methods(self) -> None:
        class Incomplete:
            def execute_phase1(self, **kwargs: Any) -> Any:
                return make_momentum_audit()

            def execute_phase2(self, **kwargs: Any) -> Any:
                return make_synthesis_data()

        with pytest.raises(MotorContractError, match="execute_phase3"):
            RiemannianInertiaAgent(Incomplete())  # type: ignore[arg-type]

    def test_constructor_rejects_non_callable_phase(self) -> None:
        class NotCallable:
            execute_phase1 = "nope"
            execute_phase2 = lambda self, **kwargs: make_synthesis_data()  # noqa: E731
            execute_phase3 = lambda self, **kwargs: make_veto_data()  # noqa: E731

        with pytest.raises(MotorContractError, match="execute_phase1"):
            RiemannianInertiaAgent(NotCallable())  # type: ignore[arg-type]

    def test_constructor_accepts_recording_motor(self) -> None:
        agent = RiemannianInertiaAgent(RecordingMotor())
        assert agent._motor is not None


class TestOrchestratorPayloadValidation:
    """Pre-auditoría dimensional: no duplica el espectro del motor."""

    def test_valid_payload_is_accepted(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        coherent_agent._validate_governance_payload(**coherent_payload(3))

    def test_rejects_uncoercible_payload(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["q_dot"] = object()  # type: ignore[assignment]
        with pytest.raises(RiemannianInertiaAgentError, match="convertible"):
            coherent_agent._validate_governance_payload(**payload)

    def test_rejects_empty_tensor(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["G_tensor"] = np.zeros((0, 0), dtype=np.float64)
        with pytest.raises(RiemannianInertiaAgentError, match="vacío"):
            coherent_agent._validate_governance_payload(**payload)

    def test_rejects_non_finite_entries(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["grad_H"] = np.array([1.0, math.nan, 0.0], dtype=np.float64)
        with pytest.raises(RiemannianInertiaAgentError, match="no finitos"):
            coherent_agent._validate_governance_payload(**payload)

    def test_rejects_non_vector_state(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["q_dot"] = np.eye(3, dtype=np.float64)
        with pytest.raises(RiemannianInertiaAgentError, match="1-D"):
            coherent_agent._validate_governance_payload(**payload)

    def test_rejects_velocity_gradient_dimension_mismatch(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["grad_H"] = make_vector(2, seed=4)
        with pytest.raises(RiemannianInertiaAgentError, match="mismo espacio"):
            coherent_agent._validate_governance_payload(**payload)

    def test_rejects_matrix_with_wrong_shape(
        self,
        coherent_agent: RiemannianInertiaAgent,
    ) -> None:
        payload = coherent_payload(3)
        payload["J_base"] = make_skew_matrix(2)
        with pytest.raises(RiemannianInertiaAgentError, match="3×3"):
            coherent_agent._validate_governance_payload(**payload)


class TestOrchestratorOODAWithFakeMotor:
    """Ciclo OODA completo contra un motor determinista de laboratorio."""

    def test_coherent_governance_invokes_phases_in_order(self) -> None:
        motor = RecordingMotor()
        agent = RiemannianInertiaAgent(motor)
        state = agent.execute_inertia_governance(
            **coherent_payload(3),
            raise_on_veto=True,
            fail_fast=False,
        )
        assert motor.calls == ["phase1", "phase2", "phase3"]
        assert state.is_epistemologically_valid is True
        assert state.final_supremum_verdict is _COHERENT
        assert motor.phase2_kwargs is not None
        assert motor.phase2_kwargs["momentum_data"] is motor.phase1
        assert motor.phase3_kwargs is not None
        assert motor.phase3_kwargs["synthesis_data"] is motor.phase2

    def test_agent_forwards_gauge_metric_into_phase2(self) -> None:
        motor = RecordingMotor()
        agent = RiemannianInertiaAgent(motor)
        payload = coherent_payload(3)
        state = agent.execute_inertia_governance(**payload)
        assert state.phase2_bridge.gauge_signature_verdict in {
            _COHERENT,
            _DEGRADED,
            _VETOED,
        }
        # Con G genérica (no conformal) un W euclídeo-skew puede degradar
        # o vetar el calibre; el orquestador no debe romper el ciclo si
        # el motor entregó un W nulo (so(n) trivial es G-skew).
        assert isinstance(state.final_supremum_verdict, InertialHeytingVerdict)

    def test_zero_gyroscopic_tensor_is_gauge_coherent_for_any_metric(self) -> None:
        motor = RecordingMotor(
            phase2=make_synthesis_data(
                tensor=np.zeros((3, 3), dtype=np.float64),
                gyroscopic_frobenius_norm=0.0,
                skew_numerical_rank=0,
            )
        )
        agent = RiemannianInertiaAgent(motor)
        state = agent.execute_inertia_governance(**coherent_payload(3))
        assert state.phase2_bridge.gauge_signature_verdict is _COHERENT
        assert state.is_epistemologically_valid is True

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_pipeline_is_valid_across_canonical_dimensions(
        self,
        dimension: int,
    ) -> None:
        motor = RecordingMotor(
            phase1=make_momentum_audit(dimension=dimension),
            phase2=make_synthesis_data(
                dimension=dimension,
                tensor=np.zeros((dimension, dimension), dtype=np.float64),
                gyroscopic_frobenius_norm=0.0,
                skew_numerical_rank=0,
            ),
            phase3=make_veto_data(dimension=dimension),
        )
        agent = RiemannianInertiaAgent(motor)
        state = agent.execute_inertia_governance(**coherent_payload(dimension))
        assert state.is_epistemologically_valid is True
        assert state.final_supremum_verdict is _COHERENT


class TestFailFastAndRaiseOnVetoPolicies:
    """fail_fast corta el OODA; raise_on_veto decide el colapso terminal."""

    def test_fail_fast_stops_before_motor_phase2_on_liouville_veto(self) -> None:
        motor = RecordingMotor(
            phase1=make_momentum_audit(is_bounded=False, momentum_norm=1.0)
        )
        agent = RiemannianInertiaAgent(motor)
        with pytest.raises(LiouvilleVolumeCollapse, match="Fail-fast en Fase 1"):
            agent.execute_inertia_governance(
                **coherent_payload(3),
                raise_on_veto=True,
                fail_fast=True,
            )
        assert motor.calls == ["phase1"]

    def test_fail_fast_stops_before_motor_phase3_on_skew_veto(self) -> None:
        motor = RecordingMotor(phase2=make_synthesis_data(is_strictly_skew=False))
        agent = RiemannianInertiaAgent(motor)
        with pytest.raises(SkewSignatureCollapse, match="Fail-fast en Fase 2"):
            agent.execute_inertia_governance(
                **coherent_payload(3),
                raise_on_veto=True,
                fail_fast=True,
            )
        assert motor.calls == ["phase1", "phase2"]

    def test_without_fail_fast_phase1_veto_reaches_phase3_then_collapses(self) -> None:
        motor = RecordingMotor(
            phase1=make_momentum_audit(is_bounded=False, momentum_norm=1.0)
        )
        agent = RiemannianInertiaAgent(motor)
        with pytest.raises(LiouvilleVolumeCollapse, match="Colapso de software"):
            agent.execute_inertia_governance(
                **coherent_payload(3),
                raise_on_veto=True,
                fail_fast=False,
            )
        assert motor.calls == ["phase1", "phase2", "phase3"]

    def test_raise_on_veto_false_completes_ooda_and_returns_invalid_state(self) -> None:
        motor = RecordingMotor(
            phase1=make_momentum_audit(
                momentum_norm=ria._MOMENTUM_HARD_LIMIT * 2.0,
                is_bounded=True,
            )
        )
        agent = RiemannianInertiaAgent(motor)
        state = agent.execute_inertia_governance(
            **coherent_payload(3),
            raise_on_veto=False,
            fail_fast=False,
        )
        assert motor.calls == ["phase1", "phase2", "phase3"]
        assert state.is_epistemologically_valid is False
        assert state.final_supremum_verdict is _VETOED

    def test_fail_fast_false_and_work_veto_raises_thermodynamic(self) -> None:
        motor = RecordingMotor(
            phase3=make_veto_data(is_symplectically_passive=False)
        )
        agent = RiemannianInertiaAgent(motor)
        with pytest.raises(ThermodynamicPassivityCollapse):
            agent.execute_inertia_governance(
                **coherent_payload(3),
                raise_on_veto=True,
                fail_fast=False,
            )
        assert motor.calls == ["phase1", "phase2", "phase3"]

    def test_degraded_does_not_trigger_fail_fast(self) -> None:
        motor = RecordingMotor(
            phase1=make_momentum_audit(momentum_norm=ria._MOMENTUM_SOFT_LIMIT * 2.0)
        )
        agent = RiemannianInertiaAgent(motor)
        state = agent.execute_inertia_governance(
            **coherent_payload(3),
            raise_on_veto=True,
            fail_fast=True,
        )
        assert motor.calls == ["phase1", "phase2", "phase3"]
        assert state.final_supremum_verdict is _DEGRADED
        assert state.is_epistemologically_valid is True


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSICIÓN ANIDADA Y MOTOR REAL
# ══════════════════════════════════════════════════════════════════════════════
class TestNestedHandoffsComposeLikeOrchestrator:
    """
    Contrato documental:

        execute_phase3 ∘ Φ₂₃ ∘ execute_phase2 ∘ Φ₁₂ ∘ execute_phase1
            ≡ gobernanza del orquestador   (sobre los mismos certificados).
    """

    def test_manual_composition_matches_orchestrator_state(self) -> None:
        audit = make_momentum_audit()
        synthesis = make_synthesis_data(
            tensor=np.zeros((3, 3), dtype=np.float64),
            gyroscopic_frobenius_norm=0.0,
            skew_numerical_rank=0,
        )
        veto = make_veto_data()
        motor = RecordingMotor(phase1=audit, phase2=synthesis, phase3=veto)
        agent = RiemannianInertiaAgent(motor)

        phase1 = agent.execute_phase1(audit)
        handed_1 = agent.handoff_phase1_to_phase2(phase1)
        phase2 = agent.execute_phase2(handed_1, synthesis, G_tensor=np.eye(3))
        handed_2 = agent.handoff_phase2_to_phase3(phase2)
        composed = agent.execute_phase3(handed_2, veto, raise_on_veto=True)

        orchestrated = agent.execute_inertia_governance(
            **coherent_payload(3),
            raise_on_veto=True,
            fail_fast=False,
        )

        assert composed.final_supremum_verdict is orchestrated.final_supremum_verdict
        assert composed.work_verdict is orchestrated.work_verdict
        assert (
            composed.is_epistemologically_valid
            is orchestrated.is_epistemologically_valid
        )
        assert composed.phase2_bridge.skew_verdict is orchestrated.phase2_bridge.skew_verdict
        assert (
            composed.phase2_bridge.phase1_bridge.liouville_verdict
            is orchestrated.phase2_bridge.phase1_bridge.liouville_verdict
        )

    def test_phi12_is_the_domain_of_phase2_and_phi23_of_phase3(self) -> None:
        agent = RiemannianInertiaAgent(RecordingMotor())
        phase1 = agent.execute_phase1(make_momentum_audit())
        # Φ₁₂ no lanza sobre un certificado coherente y es el input de F2.
        assert agent.handoff_phase1_to_phase2(phase1) is phase1
        phase2 = agent.execute_phase2(phase1, make_synthesis_data())
        assert agent.handoff_phase2_to_phase3(phase2) is phase2
        state = agent.execute_phase3(phase2, make_veto_data())
        assert state.phase2_bridge is phase2


@pytest.mark.skipif(not _has_real_motor(), reason="modulator 4.0.0 no disponible")
class TestOrchestratorWithRealModulator:
    """El agente gobierna al funtor físico real sin romper invariantes."""

    def test_real_motor_coherent_payload_is_epistemologically_valid(self) -> None:
        motor_cls = ria.RiemannianInertiaModulator
        agent = RiemannianInertiaAgent(motor_cls())
        state = agent.execute_inertia_governance(
            **coherent_payload(3),
            raise_on_veto=True,
            fail_fast=False,
        )
        assert state.is_epistemologically_valid is True
        assert state.final_supremum_verdict in {_COHERENT, _DEGRADED}
        assert state.work_verdict in {_COHERENT, _DEGRADED}
        assert state.veto_data.is_symplectically_passive is True
        assert state.work_margin >= 0.0
        assert state.liouville_probe_residual >= 0.0

    def test_real_motor_divergent_momentum_collapses_liouville(self) -> None:
        motor_cls = ria.RiemannianInertiaModulator
        agent = RiemannianInertiaAgent(motor_cls())
        payload = coherent_payload(2)
        payload["q_dot"] = np.full(2, ria._MOMENTUM_HARD_LIMIT * 3.0, dtype=np.float64)
        payload["grad_H"] = make_vector(2, seed=5)
        metric, inverse = make_spd_pair(2, condition=1.0, seed=2)
        payload["G_tensor"] = metric
        payload["G_inv"] = inverse
        payload["J_base"] = make_skew_matrix(2)
        payload["vorticity_matrix"] = make_skew_matrix(2, seed=6)
        # El motor lanza antes de que el agente clasifique; ambos son fallos
        # de integridad inercial y deben abortar la gobernanza.
        with pytest.raises(Exception):
            agent.execute_inertia_governance(**payload, raise_on_veto=True)

    def test_real_motor_symmetric_dirac_is_rejected(self) -> None:
        motor_cls = ria.RiemannianInertiaModulator
        agent = RiemannianInertiaAgent(motor_cls())
        payload = coherent_payload(3)
        payload["J_base"] = np.eye(3, dtype=np.float64)
        with pytest.raises(Exception):
            agent.execute_inertia_governance(**payload, raise_on_veto=True)

    @pytest.mark.parametrize("dimension", (1, 2, 4))
    def test_real_motor_preserves_passivity_across_dimensions(
        self,
        dimension: int,
    ) -> None:
        motor_cls = ria.RiemannianInertiaModulator
        agent = RiemannianInertiaAgent(motor_cls())
        state = agent.execute_inertia_governance(
            **coherent_payload(dimension),
            raise_on_veto=True,
            fail_fast=False,
        )
        assert state.is_epistemologically_valid is True
        assert state.veto_data.is_symplectically_passive is True
        dirac = np.asarray(state.veto_data.effective_dirac_matrix, dtype=np.float64)
        residual = float(np.linalg.norm(dirac + dirac.T, ord="fro"))
        assert residual <= _ATOL_STRUCT