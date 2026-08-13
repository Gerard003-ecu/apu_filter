# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Pruebas : Riemannian Inertia Modulator                                       ║
║ Ruta    : tests/unit/physics/test_riemannian_inertia_modulator.py            ║
║ Versión : 4.0.0-Spectral-Neumaier-deRham-Lie-Strict                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Contrato de la batería (composición funtorial anidada): ────────────────────────
Las pruebas reproducen el mismo encadenamiento que el módulo bajo examen:

    Fase 1  →  Φ₁₂ = handoff_phase1_to_phase2  →  Fase 2
    Fase 2  →  Φ₂₃ = handoff_phase2_to_phase3  →  Fase 3
    Fase 3  →  ThermodynamicVetoData
    Orquestador = Fase 3 ∘ Φ₂₃ ∘ Fase 2 ∘ Φ₁₂ ∘ Fase 1

Cada clase de Fase N termina con la auditoría del morfismo de handoff, que
es exactamente el dominio sobre el que arranca la clase de la Fase N+1.

Invariantes certificados por esta batería: ─────────────────────────────────────
    [I1] Liouville: xᵀ J_eff x ≡ 0  (Neumaier, pares, sondas).
    [I2] de Rham:   ♯ ∘ ♭ = id,  G G⁻¹ = I  (Wilkinson).
    [I3] Clausius:  ⟨∇H, J_eff ∇H⟩ ≡ 0  y  J_eff ∈ so(n).
    [I4] Gram:      ‖p ∧ ω‖_F² = 2 (‖p‖²‖ω‖² − ⟨p,ω⟩²) α².
    [I5] Lie:       Wᵀ = −W, rango numérico par, proyección idempotente.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

import app.physics.riemannian_inertia_modulator as rim
from app.physics.riemannian_inertia_modulator import (
    DualPairingError,
    ExteriorAlgebraError,
    GyroscopicSynthesisData,
    MetricCoherenceError,
    MomentumAuditData,
    MomentumDivergenceError,
    Phase1_MomentumSpectrometer,
    Phase2_GyroscopicSynthesizer,
    Phase3_SymplecticInertiaModulator,
    PhaseHandoffError,
    RiemannianInertiaError,
    RiemannianInertiaModulator,
    SkewSymmetryViolationError,
    SymplecticWorkViolationError,
    ThermodynamicVetoData,
)


# ══════════════════════════════════════════════════════════════════════════════
# §T0. CONSTANTES, FÁBRICAS Y FIXTURES DE LA BATERÍA
# ══════════════════════════════════════════════════════════════════════════════
_SEED_METRIC: Final[int] = 7
_SEED_VECTOR: Final[int] = 11
_SEED_SKEW: Final[int] = 13
_SEED_VORTICITY: Final[int] = 17
_SEED_GRADIENT: Final[int] = 19

_DIMS_CANONICAL: Final[tuple[int, ...]] = (1, 2, 3, 4, 8)
_CONDS_CANONICAL: Final[tuple[float, ...]] = (1.0, 10.0, 1.0e3)
_ATOL_STRUCT: Final[float] = 1.0e-10
_RTOL_STRUCT: Final[float] = 1.0e-10


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_spd_pair(
    dimension: int,
    condition: float = 10.0,
    seed: int = _SEED_METRIC,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fabrica un par métrico espectralmente exacto:

        G = Q diag(λ) Qᵀ,   G⁻¹ = Q diag(λ⁻¹) Qᵀ,
        λ ∈ [1, κ],   G = Gᵀ ≻ 0.
    """
    if dimension < 1:
        raise ValueError("dimension debe ser ≥ 1.")
    if condition < 1.0:
        raise ValueError("condition debe ser ≥ 1.")

    if dimension == 1:
        metric = np.array([[float(condition)]], dtype=np.float64)
        return metric, np.array([[1.0 / float(condition)]], dtype=np.float64)

    generator = _rng(seed)
    raw, _ = np.linalg.qr(generator.standard_normal((dimension, dimension)))
    # Garantiza orientación positiva para evitar reflexiones espurias.
    if np.linalg.det(raw) < 0.0:
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
    """Proyección exacta al cono so(n).  so(1) = {0}."""
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
    generator = _rng(seed)
    return np.ascontiguousarray(
        float(scale) * generator.standard_normal(dimension),
        dtype=np.float64,
    )


def make_momentum_audit(
    *,
    dimension: int = 3,
    momentum_norm: float = 1.0,
    is_bounded: bool = True,
    metric_condition_number: float = 2.0,
    inverse_consistency_residual: float = 0.0,
    covariant_momentum: NDArray[np.float64] | None = None,
) -> MomentumAuditData:
    """DTO sintético de Fase 1 para auditar Φ₁₂ sin ejecutar el motor."""
    if covariant_momentum is None:
        covariant_momentum = make_vector(dimension, seed=3, scale=0.5)
    reconstructed = np.array(covariant_momentum, dtype=np.float64, copy=True)
    return MomentumAuditData(
        covariant_momentum=covariant_momentum,
        reconstructed_velocity=reconstructed,
        momentum_norm=float(momentum_norm),
        kinetic_energy_primal=float(momentum_norm) ** 2,
        kinetic_energy_dual=float(momentum_norm) ** 2,
        dual_pairing=float(momentum_norm) ** 2,
        pairing_residual=0.0,
        musical_roundtrip_residual=0.0,
        is_bounded=is_bounded,
        metric_condition_number=float(metric_condition_number),
        inverse_consistency_residual=float(inverse_consistency_residual),
        wilkinson_bound=1.0e-8,
        spectral_minimum=1.0,
        spectral_maximum=float(metric_condition_number),
    )


def make_synthesis_data(
    *,
    dimension: int = 3,
    is_strictly_skew: bool = True,
    relative_skew_residual: float = 0.0,
    tensor: NDArray[np.float64] | None = None,
) -> GyroscopicSynthesisData:
    """DTO sintético de Fase 2 para auditar Φ₂₃ sin ejecutar el motor."""
    if tensor is None:
        tensor = make_skew_matrix(dimension, seed=5)
    return GyroscopicSynthesisData(
        skew_symmetric_tensor=tensor,
        vorticity_two_form=make_skew_matrix(dimension, seed=6),
        omega_vector=make_vector(dimension, seed=8),
        antisymmetry_residual=0.0,
        relative_skew_residual=float(relative_skew_residual),
        vorticity_projection_residual=0.0,
        gyroscopic_frobenius_norm=float(np.linalg.norm(tensor, ord="fro")),
        wedge_gram_residual=0.0,
        skew_numerical_rank=0,
        rank_is_even=True,
        is_strictly_skew=is_strictly_skew,
    )


def assert_finite_array(array: NDArray[np.float64], name: str) -> None:
    assert isinstance(array, np.ndarray), f"{name} no es ndarray."
    assert array.dtype == np.float64, f"{name} no es float64."
    assert np.all(np.isfinite(array)), f"{name} contiene no-finitos."


def assert_readonly(array: NDArray[np.float64], name: str) -> None:
    assert not array.flags.writeable, f"{name} debería ser de solo lectura."
    if array.size == 0:
        return
    with pytest.raises((ValueError, RuntimeError)):
        array.reshape(-1)[0] = 0.0


def assert_skew(
    matrix: NDArray[np.float64],
    name: str,
    atol: float = _ATOL_STRUCT,
) -> None:
    residual = float(np.linalg.norm(matrix + matrix.T, ord="fro"))
    assert residual <= atol, f"{name} no es antisimétrica: ‖A+Aᵀ‖_F = {residual:.4e}."
    diagonal = np.diag(matrix)
    assert np.max(np.abs(diagonal)) <= atol, f"{name} tiene traza diagonal no nula."


def assert_spd(matrix: NDArray[np.float64], name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert float(eigenvalues[0]) > 0.0, f"{name} no es definida positiva."


@pytest.fixture(scope="module")
def spectrometer() -> Phase1_MomentumSpectrometer:
    return Phase1_MomentumSpectrometer()


@pytest.fixture(scope="module")
def synthesizer() -> Phase2_GyroscopicSynthesizer:
    return Phase2_GyroscopicSynthesizer()


@pytest.fixture(scope="module")
def modulator_phase() -> Phase3_SymplecticInertiaModulator:
    return Phase3_SymplecticInertiaModulator()


@pytest.fixture(scope="module")
def functor() -> RiemannianInertiaModulator:
    return RiemannianInertiaModulator()


@pytest.fixture
def payload_n3() -> dict[str, NDArray[np.float64]]:
    dimension = 3
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


# ══════════════════════════════════════════════════════════════════════════════
# §T1. EXPORTACIONES, JERARQUÍA Y DTOs INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════
class TestExportsAndAlgebraicHierarchy:
    """Certifica el contrato público y la torre de excepciones."""

    def test_version_is_canonical(self) -> None:
        assert rim.__version__ == "4.0.0-Spectral-Neumaier-deRham-Lie-Strict"

    def test_all_exports_are_importable(self) -> None:
        for name in rim.__all__:
            assert hasattr(rim, name), f"{name} falta en el módulo."
            assert getattr(rim, name) is not None

    def test_exception_tower(self) -> None:
        assert issubclass(RiemannianInertiaError, rim.TopologicalInvariantError)
        for cls in (
            MetricCoherenceError,
            DualPairingError,
            MomentumDivergenceError,
            ExteriorAlgebraError,
            SkewSymmetryViolationError,
            SymplecticWorkViolationError,
            PhaseHandoffError,
        ):
            assert issubclass(cls, RiemannianInertiaError)

    def test_inheritance_is_nested_by_phase(self) -> None:
        assert issubclass(Phase2_GyroscopicSynthesizer, Phase1_MomentumSpectrometer)
        assert issubclass(Phase3_SymplecticInertiaModulator, Phase2_GyroscopicSynthesizer)
        assert issubclass(RiemannianInertiaModulator, Phase3_SymplecticInertiaModulator)
        assert issubclass(RiemannianInertiaModulator, rim.Morphism)

    def test_physical_constants_are_sane(self) -> None:
        assert rim._MACHINE_EPSILON > 0.0
        assert rim._SYMPLECTIC_TOLERANCE > rim._MACHINE_EPSILON
        assert rim._MOMENTUM_MAX_BOUND > 1.0
        assert rim._CONDITION_NUMBER_MAX > 1.0
        assert rim._VORTICITY_COUPLING_FACTOR == 1.0
        assert rim._SKEW_RELATIVE_TOLERANCE > 0.0
        assert rim._LIOUVILLE_PROBE_COUNT >= 1
        assert rim._LIOUVILLE_PROBE_SEED == 1729
        assert rim._WILKINSON_CONSTANT >= 1.0


class TestImmutableDTOs:
    """Los certificados del fibrado cotangente son inmutables y sellados."""

    def test_momentum_audit_is_frozen_and_slotted(self) -> None:
        audit = make_momentum_audit()
        with pytest.raises(FrozenInstanceError):
            audit.momentum_norm = 0.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            audit.not_a_field = True  # type: ignore[attr-defined]

    def test_gyroscopic_synthesis_is_frozen_and_slotted(self) -> None:
        data = make_synthesis_data()
        with pytest.raises(FrozenInstanceError):
            data.is_strictly_skew = False  # type: ignore[misc]
        with pytest.raises(AttributeError):
            data.not_a_field = True  # type: ignore[attr-defined]

    def test_thermodynamic_veto_is_frozen_and_slotted(self) -> None:
        veto = ThermodynamicVetoData(
            effective_dirac_matrix=make_skew_matrix(2),
            nilpotent_work_residual=0.0,
            pairwise_work_residual=0.0,
            liouville_probe_residual=0.0,
            dirac_symmetric_residual=0.0,
            relative_skew_residual=0.0,
            work_tolerance=1.0e-12,
            is_symplectically_passive=True,
        )
        with pytest.raises(FrozenInstanceError):
            veto.is_symplectically_passive = False  # type: ignore[misc]
        with pytest.raises(AttributeError):
            veto.not_a_field = True  # type: ignore[attr-defined]

    def test_freeze_array_seals_write_flag(self) -> None:
        frozen = rim._freeze_array(np.array([1.0, 2.0], dtype=np.float64))
        assert_readonly(frozen, "frozen")
        assert frozen.tolist() == [1.0, 2.0]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → PRIMITIVAS NUMÉRICAS, MÉTRICA, ISOMORFISMO MUSICAL Y HANDOFF Φ₁₂
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1ValidationPrimitives:
    """Granularidad de los validadores y de la suma de Neumaier."""

    def test_validate_vector_accepts_list_and_copies(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        raw = [1.0, -2.5, 3.25]
        vector = spectrometer._validate_vector(raw, "q_dot")
        assert vector.dtype == np.float64
        assert vector.shape == (3,)
        raw[0] = 99.0  # type: ignore[index]
        assert vector[0] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ([], "vacío"),
            ([[1.0, 2.0]], "1-D"),
            ([1.0, math.nan], "no finitos"),
            ([1.0, math.inf], "no finitos"),
        ],
    )
    def test_validate_vector_rejects_degenerate_payloads(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        payload: object,
        match: str,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match=match):
            spectrometer._validate_vector(payload, "q_dot")

    def test_validate_vector_rejects_uncoercible(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="no pudo convertirse"):
            spectrometer._validate_vector(object(), "q_dot")

    def test_validate_square_matrix_accepts_and_copies(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        raw = np.array([[2.0, 0.1], [0.1, 3.0]], dtype=np.float64)
        matrix = spectrometer._validate_square_matrix(raw, "G")
        raw[0, 0] = -1.0
        assert matrix[0, 0] == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (np.zeros((0, 0)), "vacía"),
            (np.zeros((2, 3)), "cuadrada"),
            (np.array([1.0, 2.0]), "cuadrada"),
            (np.array([[1.0, math.nan], [0.0, 1.0]]), "no finitos"),
        ],
    )
    def test_validate_square_matrix_rejects_degenerate_payloads(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        payload: object,
        match: str,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match=match):
            spectrometer._validate_square_matrix(payload, "G")

    def test_frobenius_and_euclidean_norms(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        matrix = np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        assert spectrometer._frobenius_norm(matrix, "M") == pytest.approx(5.0)
        assert spectrometer._euclidean_norm(
            np.array([3.0, 4.0], dtype=np.float64), "v"
        ) == pytest.approx(5.0)

    def test_symmetrize_and_skew_are_complementary_projections(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        raw = _rng(23).standard_normal((5, 5)).astype(np.float64)
        symmetric = spectrometer._symmetrize(raw)
        skew = spectrometer._skew_symmetrize(raw)
        np.testing.assert_allclose(symmetric + skew, raw, atol=1.0e-14)
        np.testing.assert_allclose(symmetric, symmetric.T, atol=1.0e-14)
        np.testing.assert_allclose(skew, -skew.T, atol=1.0e-14)
        # Ortogonalidad de Frobenius: ⟨sym, skew⟩_F = 0.
        pairing = float(np.sum(symmetric * skew))
        assert abs(pairing) <= 1.0e-13
        # Idempotencia.
        np.testing.assert_allclose(
            spectrometer._symmetrize(symmetric), symmetric, atol=1.0e-15
        )
        np.testing.assert_allclose(
            spectrometer._skew_symmetrize(skew), skew, atol=1.0e-15
        )

    def test_residual_tolerances_scale_with_frobenius_norm(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        small = np.eye(2, dtype=np.float64)
        large = 1.0e6 * np.eye(2, dtype=np.float64)
        tol_small = spectrometer._matrix_residual_tolerance(small)
        tol_large = spectrometer._matrix_residual_tolerance(large)
        assert tol_small == pytest.approx(rim._SYMPLECTIC_TOLERANCE)
        assert tol_large > tol_small
        assert tol_large == pytest.approx(
            rim._SYMPLECTIC_TOLERANCE * float(np.linalg.norm(large, ord="fro"))
        )

    def test_relative_skew_residual_of_exact_so_n_is_zero(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        skew = make_skew_matrix(6, seed=29)
        assert spectrometer._relative_skew_residual(skew) <= 1.0e-15

    def test_neumaier_recovers_cancelled_unit(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        terms = np.array([1.0e16, 1.0, -1.0e16], dtype=np.float64)
        naive = float(np.sum(terms))
        compensated = spectrometer._neumaier_sum(terms, "cancelación")
        assert compensated == pytest.approx(1.0, abs=1.0e-12)
        # El propósito es precisamente no colapsar a 0 como la suma ingenua.
        assert compensated != pytest.approx(0.0)
        assert naive == pytest.approx(0.0) or compensated == pytest.approx(1.0)

    def test_neumaier_sum_of_range_matches_closed_form(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        terms = np.arange(1.0, 1001.0, dtype=np.float64)
        expected = 1000.0 * 1001.0 / 2.0
        assert spectrometer._neumaier_sum(terms, "1..1000") == pytest.approx(expected)

    def test_neumaier_dot_matches_inner_product(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        left = np.array([1.0, -2.0, 3.5], dtype=np.float64)
        right = np.array([4.0, 0.5, -1.0], dtype=np.float64)
        assert spectrometer._neumaier_dot(left, right, "⟨u,v⟩") == pytest.approx(
            float(left @ right)
        )

    def test_neumaier_dot_rejects_dimension_mismatch(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="Dimensión incompatible"):
            spectrometer._neumaier_dot(
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                "⟨u,v⟩",
            )

    def test_neumaier_sum_rejects_non_finite(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="no finitos"):
            spectrometer._neumaier_sum(
                np.array([1.0, math.nan], dtype=np.float64),
                "nan",
            )


class TestPhase1MetricPairAndSpectrum:
    """Par métrico (G, G⁻¹): simetría, espectro, Wilkinson, inversa espectral."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    @pytest.mark.parametrize("condition", _CONDS_CANONICAL)
    def test_valid_metric_pair_is_accepted(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        dimension: int,
        condition: float,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=condition)
        (
            sanitized_g,
            sanitized_inv,
            cond,
            residual,
            wilkinson,
            lambda_min,
            lambda_max,
        ) = spectrometer._validate_metric_pair(metric, inverse)
        assert_spd(sanitized_g, "G")
        assert_spd(sanitized_inv, "G⁻¹")
        assert cond >= 1.0
        assert cond <= condition * 1.05 + 1.0e-9
        assert residual >= 0.0
        assert wilkinson >= rim._METRIC_INVERSE_TOLERANCE
        assert lambda_min > 0.0
        assert lambda_max >= lambda_min
        identity = np.eye(dimension, dtype=np.float64)
        np.testing.assert_allclose(sanitized_g @ sanitized_inv, identity, atol=1.0e-8)
        np.testing.assert_allclose(sanitized_inv @ sanitized_g, identity, atol=1.0e-8)

    def test_asymmetric_metric_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=4.0)
        broken = metric.copy()
        broken[0, 2] += 1.5
        with pytest.raises(MetricCoherenceError, match="no es simétrica"):
            spectrometer._validate_metric_pair(broken, inverse)

    def test_asymmetric_inverse_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=4.0)
        broken = inverse.copy()
        broken[1, 0] += 2.0
        with pytest.raises(MetricCoherenceError, match="no es simétrica"):
            spectrometer._validate_metric_pair(metric, broken)

    def test_indefinite_metric_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        indefinite = np.diag([1.0, -1.0, 2.0]).astype(np.float64)
        inverse = np.diag([1.0, -1.0, 0.5]).astype(np.float64)
        with pytest.raises(MetricCoherenceError):
            spectrometer._validate_metric_pair(indefinite, inverse)

    def test_negative_definite_metric_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        negative = -np.eye(2, dtype=np.float64)
        with pytest.raises(MetricCoherenceError, match="definida positiva"):
            spectrometer._validate_metric_pair(negative, negative)

    def test_singular_metric_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        singular = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        fake_inv = np.eye(2, dtype=np.float64)
        with pytest.raises(MetricCoherenceError):
            spectrometer._validate_metric_pair(singular, fake_inv)

    def test_shape_mismatch_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, _ = make_spd_pair(3)
        inverse, _ = make_spd_pair(2, seed=99)
        with pytest.raises(MetricCoherenceError, match="misma dimensión"):
            spectrometer._validate_metric_pair(metric, inverse)

    def test_inconsistent_inverse_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(4, condition=5.0)
        broken = 2.0 * inverse
        with pytest.raises(MetricCoherenceError, match="Wilkinson|inversa espectral"):
            spectrometer._validate_metric_pair(metric, broken)

    def test_tiny_asymmetry_below_tolerance_is_sanitized(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=3.0)
        metric = metric.copy()
        metric[0, 1] += 1.0e-16
        sanitized, _, _, _, _, _, _ = spectrometer._validate_metric_pair(metric, inverse)
        np.testing.assert_allclose(sanitized, sanitized.T, atol=1.0e-15)

    def test_ill_conditioned_beyond_hard_cap_is_rejected(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=rim._CONDITION_NUMBER_MAX * 10.0)
        with pytest.raises(MetricCoherenceError, match="mal condicionada"):
            spectrometer._validate_metric_pair(metric, inverse)

    def test_spectral_inverse_reproduces_analytic_inverse(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(5, condition=25.0, seed=41)
        evals, evecs = spectrometer._spectral_decompose_spd(metric, "G")
        spectral = spectrometer._spectral_inverse(evals, evecs)
        np.testing.assert_allclose(spectral, inverse, atol=1.0e-10, rtol=1.0e-10)

    def test_wilkinson_tolerance_grows_with_dimension_and_condition(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        small = spectrometer._wilkinson_inverse_tolerance(2, 1.0)
        large = spectrometer._wilkinson_inverse_tolerance(64, 1.0e6)
        assert large > small
        assert small >= rim._METRIC_INVERSE_TOLERANCE

    def test_condition_number_from_spectrum_matches_ratio(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        evals = np.array([0.5, 1.0, 4.0], dtype=np.float64)
        assert spectrometer._condition_number_from_spectrum(evals, "G") == pytest.approx(
            8.0
        )


class TestPhase1MusicalIsomorphismAndDualPairing:
    """♭, ♯, round-trip e identidad de apareamiento dual."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_flat_is_metric_contraction(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        dimension: int,
    ) -> None:
        metric, _ = make_spd_pair(dimension, condition=6.0)
        velocity = make_vector(dimension, seed=53)
        momentum = spectrometer._musical_flat(velocity, metric)
        np.testing.assert_allclose(momentum, metric @ velocity, atol=1.0e-14)

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_sharp_inverts_flat(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        dimension: int,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=7.0)
        velocity = make_vector(dimension, seed=59)
        momentum = spectrometer._musical_flat(velocity, metric)
        reconstructed = spectrometer._musical_sharp(momentum, inverse)
        np.testing.assert_allclose(reconstructed, velocity, atol=1.0e-10, rtol=1.0e-10)

    def test_flat_rejects_dimension_mismatch(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, _ = make_spd_pair(3)
        with pytest.raises(RiemannianInertiaError, match="incompatible"):
            spectrometer._musical_flat(np.array([1.0, 2.0], dtype=np.float64), metric)

    def test_roundtrip_residual_of_identity_pair_is_certified(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        velocity = np.array([0.25, -0.5, 0.75], dtype=np.float64)
        residual = spectrometer._certify_musical_roundtrip(velocity, velocity, 1.0)
        assert residual == pytest.approx(0.0)

    def test_roundtrip_rejects_large_deviation(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        velocity = np.array([1.0, 0.0], dtype=np.float64)
        broken = np.array([1.0, 1.0], dtype=np.float64)
        with pytest.raises(DualPairingError, match="involutivo"):
            spectrometer._certify_musical_roundtrip(velocity, broken, 1.0)

    def test_dual_pairing_identity_holds_for_spd_pair(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(4, condition=12.0, seed=61)
        velocity = make_vector(4, seed=67, scale=0.4)
        momentum = metric @ velocity
        pairing, primal, dual, residual = spectrometer._certify_dual_pairing(
            velocity, momentum, metric, inverse, 12.0
        )
        expected = float(velocity @ metric @ velocity)
        assert pairing == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
        assert primal == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
        assert dual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
        assert residual <= 1.0e-10
        assert primal >= 0.0
        assert dual >= 0.0

    def test_dual_pairing_detects_inconsistent_momentum(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=2.0)
        velocity = make_vector(3, seed=71)
        wrong_momentum = make_vector(3, seed=73, scale=4.0)
        with pytest.raises(DualPairingError, match="apareamiento dual"):
            spectrometer._certify_dual_pairing(
                velocity, wrong_momentum, metric, inverse, 2.0
            )

    def test_momentum_norm_is_sqrt_of_dual_energy(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        assert spectrometer._evaluate_momentum_norm(9.0) == pytest.approx(3.0)
        assert spectrometer._evaluate_momentum_norm(0.0) == pytest.approx(0.0)

    def test_momentum_norm_rejects_negative_energy(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="negativa"):
            spectrometer._evaluate_momentum_norm(-1.0e-3)


class TestPhase1ExecuteAndCertificates:
    """execute_phase1: certificado completo y cotas de inercia."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_execute_phase1_returns_consistent_certificate(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        dimension: int,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=9.0, seed=79)
        velocity = make_vector(dimension, seed=83, scale=0.6)
        audit = spectrometer.execute_phase1(velocity, metric, inverse)

        assert isinstance(audit, MomentumAuditData)
        assert_finite_array(audit.covariant_momentum, "p")
        assert_finite_array(audit.reconstructed_velocity, "♯p")
        assert_readonly(audit.covariant_momentum, "p")
        assert_readonly(audit.reconstructed_velocity, "♯p")
        assert audit.covariant_momentum.shape == (dimension,)
        np.testing.assert_allclose(
            audit.covariant_momentum, metric @ velocity, atol=1.0e-12
        )
        np.testing.assert_allclose(
            audit.reconstructed_velocity, velocity, atol=1.0e-9, rtol=1.0e-9
        )
        expected_energy = float(velocity @ metric @ velocity)
        assert audit.kinetic_energy_primal == pytest.approx(expected_energy, rel=1.0e-10)
        assert audit.kinetic_energy_dual == pytest.approx(expected_energy, rel=1.0e-10)
        assert audit.dual_pairing == pytest.approx(expected_energy, rel=1.0e-10)
        assert audit.momentum_norm == pytest.approx(math.sqrt(expected_energy), rel=1.0e-10)
        assert audit.is_bounded is True
        assert audit.metric_condition_number >= 1.0
        assert audit.inverse_consistency_residual >= 0.0
        assert audit.wilkinson_bound > 0.0
        assert audit.spectral_minimum > 0.0
        assert audit.spectral_maximum >= audit.spectral_minimum
        assert audit.pairing_residual >= 0.0
        assert audit.musical_roundtrip_residual >= 0.0

    def test_execute_phase1_is_isolated_from_input_mutation(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        velocity = np.array([0.2, -0.4, 0.1], dtype=np.float64)
        audit = spectrometer.execute_phase1(velocity, metric, inverse)
        snapshot = audit.covariant_momentum.copy()
        velocity[0] = 99.0
        np.testing.assert_array_equal(audit.covariant_momentum, snapshot)

    def test_execute_phase1_rejects_velocity_metric_mismatch(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        with pytest.raises(RiemannianInertiaError, match="incompatible"):
            spectrometer.execute_phase1(
                np.array([1.0, 2.0], dtype=np.float64), metric, inverse
            )

    def test_execute_phase1_vetoes_divergent_momentum(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(2, condition=1.0, seed=1)
        # ‖p‖_{G⁻¹} = ‖q̇‖₂ cuando G = I.  Escalamos por encima de P_max.
        velocity = np.array(
            [rim._MOMENTUM_MAX_BOUND * 2.0, 0.0],
            dtype=np.float64,
        )
        # make_spd_pair(2, 1.0) produce G con λ=1, no necesariamente I, pero
        # ‖p‖ = √(q̇ᵀ G q̇) ≥ √(λ_min) ‖q̇‖ ≫ P_max.
        with pytest.raises(MomentumDivergenceError, match="Divergencia inercial"):
            spectrometer.execute_phase1(velocity, metric, inverse)

    def test_zero_velocity_yields_zero_momentum(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(4, condition=5.0)
        audit = spectrometer.execute_phase1(
            np.zeros(4, dtype=np.float64), metric, inverse
        )
        assert audit.momentum_norm == pytest.approx(0.0)
        np.testing.assert_allclose(audit.covariant_momentum, 0.0, atol=1.0e-15)
        assert audit.is_bounded is True


class TestPhase1HandoffPhi12:
    """
    Definición formal final de la Fase 1.

    Φ₁₂ es, a la vez, el último morfismo de esta fase y el dominio
    sobre el que la Fase 2 construye p ∧ ω.
    """

    def test_handoff_returns_validated_covariant_momentum(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=4.0)
        velocity = make_vector(3, seed=89, scale=0.3)
        audit = spectrometer.execute_phase1(velocity, metric, inverse)
        handed = spectrometer.handoff_phase1_to_phase2(audit)
        np.testing.assert_array_equal(handed, audit.covariant_momentum)
        assert_finite_array(handed, "p_Φ₁₂")

    def test_handoff_rejects_wrong_type(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        with pytest.raises(PhaseHandoffError, match="MomentumAuditData"):
            spectrometer.handoff_phase1_to_phase2(object())  # type: ignore[arg-type]

    def test_handoff_rejects_unbounded_certificate(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        audit = make_momentum_audit(is_bounded=False, momentum_norm=1.0)
        with pytest.raises(PhaseHandoffError, match="acotado"):
            spectrometer.handoff_phase1_to_phase2(audit)

    def test_handoff_rejects_norm_above_hard_cap(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        audit = make_momentum_audit(
            is_bounded=True,
            momentum_norm=rim._MOMENTUM_MAX_BOUND * 2.0,
        )
        with pytest.raises(PhaseHandoffError, match="P_max"):
            spectrometer.handoff_phase1_to_phase2(audit)

    def test_handoff_rejects_ill_conditioned_certificate(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        audit = make_momentum_audit(
            metric_condition_number=rim._CONDITION_NUMBER_MAX * 2.0
        )
        with pytest.raises(PhaseHandoffError, match="κ"):
            spectrometer.handoff_phase1_to_phase2(audit)

    def test_handoff_rejects_corrupt_momentum_payload(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
    ) -> None:
        audit = make_momentum_audit(
            covariant_momentum=np.array([1.0, math.nan], dtype=np.float64)
        )
        with pytest.raises(RiemannianInertiaError, match="no finitos"):
            spectrometer.handoff_phase1_to_phase2(audit)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → CONTINUACIÓN DE Φ₁₂, ÁLGEBRA EXTERIOR, so(n) Y HANDOFF Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2ReceivesPhase1Handoff:
    """El primer método de la Fase 2 es la continuación literal de Φ₁₂."""

    def test_receive_certified_momentum_equals_phase1_handoff(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        metric, inverse = make_spd_pair(4, condition=5.0)
        audit = spectrometer.execute_phase1(
            make_vector(4, seed=97, scale=0.5), metric, inverse
        )
        via_phi = spectrometer.handoff_phase1_to_phase2(audit)
        via_phase2 = synthesizer._receive_certified_momentum(audit)
        np.testing.assert_array_equal(via_phi, via_phase2)

    def test_receive_rejects_the_same_unbounded_certificate(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        with pytest.raises(PhaseHandoffError):
            synthesizer._receive_certified_momentum(
                make_momentum_audit(is_bounded=False)
            )


class TestPhase2ExteriorAlgebraAndSkewCone:
    """Vorticidad, producto exterior, identidad de Gram y cono so(n)."""

    def test_vorticity_projection_discards_symmetric_strain(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        symmetric = np.array([[2.0, 1.0], [1.0, 4.0]], dtype=np.float64)
        skew = np.array([[0.0, -3.0], [3.0, 0.0]], dtype=np.float64)
        mixed = symmetric + skew
        two_form, residual = synthesizer._project_vorticity_to_two_form(mixed)
        np.testing.assert_allclose(two_form, skew, atol=1.0e-14)
        assert residual == pytest.approx(float(np.linalg.norm(symmetric, ord="fro")))
        assert_skew(two_form, "Ω_skew")

    def test_couple_vorticity_is_matrix_vector_product(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        two_form = make_skew_matrix(3, seed=101)
        momentum = make_vector(3, seed=103)
        omega = synthesizer._couple_vorticity(two_form, momentum)
        np.testing.assert_allclose(omega, two_form @ momentum, atol=1.0e-15)

    def test_couple_vorticity_rejects_dimension_mismatch(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="no coincide"):
            synthesizer._couple_vorticity(
                make_skew_matrix(3),
                make_vector(2),
            )

    def test_exterior_wedge_is_manifestly_skew(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = np.array([1.0, 0.0, 2.0], dtype=np.float64)
        right = np.array([0.0, 3.0, -1.0], dtype=np.float64)
        bivector = synthesizer._exterior_wedge(left, right, 1.0)
        assert_skew(bivector, "p ∧ ω")
        expected = np.outer(left, right) - np.outer(right, left)
        np.testing.assert_allclose(bivector, expected, atol=1.0e-15)

    def test_exterior_wedge_scales_with_coupling(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = np.array([1.0, 2.0], dtype=np.float64)
        right = np.array([3.0, -1.0], dtype=np.float64)
        unit = synthesizer._exterior_wedge(left, right, 1.0)
        scaled = synthesizer._exterior_wedge(left, right, 2.5)
        np.testing.assert_allclose(scaled, 2.5 * unit, atol=1.0e-15)

    def test_exterior_wedge_rejects_non_finite_coupling(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = np.array([1.0, 0.0], dtype=np.float64)
        with pytest.raises(ExteriorAlgebraError, match="acoplamiento"):
            synthesizer._exterior_wedge(left, left, math.inf)

    def test_gram_identity_holds_exactly_for_constructed_bivector(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = make_vector(5, seed=107)
        right = make_vector(5, seed=109)
        coupling = 0.75
        bivector = synthesizer._exterior_wedge(left, right, coupling)
        residual = synthesizer._certify_wedge_gram_identity(
            left, right, bivector, coupling
        )
        assert residual <= 1.0e-10

    def test_gram_identity_detects_corrupted_bivector(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        bivector = synthesizer._exterior_wedge(left, right, 1.0)
        bivector = bivector + np.eye(3, dtype=np.float64)
        with pytest.raises(ExteriorAlgebraError, match="Gram"):
            synthesizer._certify_wedge_gram_identity(left, right, bivector, 1.0)

    def test_parallel_vectors_produce_zero_bivector(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        left = np.array([2.0, -1.0, 0.5], dtype=np.float64)
        bivector = synthesizer._exterior_wedge(left, 3.0 * left, 1.0)
        np.testing.assert_allclose(bivector, 0.0, atol=1.0e-14)
        residual = synthesizer._certify_wedge_gram_identity(
            left, 3.0 * left, bivector, 1.0
        )
        assert residual <= 1.0e-12

    def test_project_to_skew_cone_is_idempotent(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        raw = _rng(113).standard_normal((4, 4)).astype(np.float64)
        projected, abs_residual, rel_residual = (
            synthesizer._project_to_skew_symmetric_cone(raw)
        )
        assert_skew(projected, "W_proj")
        assert abs_residual <= 1.0e-14
        assert rel_residual <= rim._SKEW_RELATIVE_TOLERANCE
        again, _, _ = synthesizer._project_to_skew_symmetric_cone(projected)
        np.testing.assert_allclose(again, projected, atol=1.0e-15)

    def test_even_rank_of_simple_bivector_in_3d(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        # p ∧ ω en R³ tiene rango 0 o 2.
        left = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        bivector = synthesizer._exterior_wedge(left, right, 1.0)
        rank, even = synthesizer._certify_even_numerical_rank(bivector)
        assert even is True
        assert rank % 2 == 0
        assert rank in {0, 2}

    def test_zero_tensor_has_even_rank_zero(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        rank, even = synthesizer._certify_even_numerical_rank(
            np.zeros((3, 3), dtype=np.float64)
        )
        assert rank == 0
        assert even is True


class TestPhase2ExecuteAndCertificates:
    """execute_phase2: síntesis de Lorentz a partir del certificado de Fase 1."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_execute_phase2_builds_so_n_lorentz_tensor(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
        dimension: int,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=6.0, seed=127)
        audit = spectrometer.execute_phase1(
            make_vector(dimension, seed=131, scale=0.55),
            metric,
            inverse,
        )
        vorticity = make_skew_matrix(dimension, seed=_SEED_VORTICITY, scale=0.9)
        synthesis = synthesizer.execute_phase2(audit, vorticity)

        assert isinstance(synthesis, GyroscopicSynthesisData)
        assert synthesis.is_strictly_skew is True
        assert_readonly(synthesis.skew_symmetric_tensor, "W")
        assert_readonly(synthesis.vorticity_two_form, "Ω")
        assert_readonly(synthesis.omega_vector, "ω")
        assert_skew(synthesis.skew_symmetric_tensor, "W")
        assert_skew(synthesis.vorticity_two_form, "Ω")
        assert synthesis.skew_symmetric_tensor.shape == (dimension, dimension)
        assert synthesis.antisymmetry_residual <= 1.0e-12
        assert synthesis.relative_skew_residual <= rim._SKEW_RELATIVE_TOLERANCE
        assert synthesis.gyroscopic_frobenius_norm >= 0.0
        assert synthesis.wedge_gram_residual >= 0.0
        assert synthesis.skew_numerical_rank >= 0
        assert synthesis.rank_is_even is True or synthesis.skew_numerical_rank % 2 == 1

        momentum = audit.covariant_momentum
        omega = synthesis.vorticity_two_form @ momentum
        np.testing.assert_allclose(synthesis.omega_vector, omega, atol=1.0e-12)
        expected = rim._VORTICITY_COUPLING_FACTOR * (
            np.outer(momentum, omega) - np.outer(omega, momentum)
        )
        np.testing.assert_allclose(
            synthesis.skew_symmetric_tensor,
            0.5 * (expected - expected.T),
            atol=1.0e-11,
        )

    def test_execute_phase2_rejects_non_audit_payload(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        with pytest.raises(PhaseHandoffError):
            synthesizer.execute_phase2(object(), make_skew_matrix(3))  # type: ignore[arg-type]

    def test_execute_phase2_rejects_vorticity_dimension_mismatch(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        audit = spectrometer.execute_phase1(make_vector(3), metric, inverse)
        with pytest.raises(RiemannianInertiaError):
            synthesizer.execute_phase2(audit, make_skew_matrix(4))

    def test_symmetric_vorticity_yields_zero_tensor(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=2.0)
        audit = spectrometer.execute_phase1(
            make_vector(3, seed=137, scale=0.4), metric, inverse
        )
        strain = np.array(
            [[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 0.8]],
            dtype=np.float64,
        )
        synthesis = synthesizer.execute_phase2(audit, strain)
        np.testing.assert_allclose(
            synthesis.skew_symmetric_tensor, 0.0, atol=1.0e-13
        )
        assert synthesis.gyroscopic_frobenius_norm == pytest.approx(0.0, abs=1.0e-13)
        assert synthesis.vorticity_projection_residual > 0.0


class TestPhase2HandoffPhi23:
    """
    Definición formal final de la Fase 2.

    Φ₂₃ es el último morfismo de esta fase y el dominio sobre el que
    la Fase 3 realiza J_eff = J + W.
    """

    def test_handoff_returns_revalidated_skew_tensor(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        audit = spectrometer.execute_phase1(make_vector(3, seed=139), metric, inverse)
        synthesis = synthesizer.execute_phase2(
            audit, make_skew_matrix(3, seed=_SEED_VORTICITY)
        )
        handed = synthesizer.handoff_phase2_to_phase3(synthesis)
        np.testing.assert_array_equal(handed, synthesis.skew_symmetric_tensor)
        assert_skew(handed, "W_Φ₂₃")

    def test_handoff_rejects_wrong_type(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        with pytest.raises(PhaseHandoffError, match="GyroscopicSynthesisData"):
            synthesizer.handoff_phase2_to_phase3(object())  # type: ignore[arg-type]

    def test_handoff_rejects_non_strict_certificate(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        data = make_synthesis_data(is_strictly_skew=False)
        with pytest.raises(PhaseHandoffError, match="antisimétrico"):
            synthesizer.handoff_phase2_to_phase3(data)

    def test_handoff_rejects_relative_skew_above_tolerance(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        data = make_synthesis_data(relative_skew_residual=1.0e-3)
        with pytest.raises(PhaseHandoffError, match="r_skew"):
            synthesizer.handoff_phase2_to_phase3(data)

    def test_handoff_rejects_tensor_that_left_so_n(
        self,
        synthesizer: Phase2_GyroscopicSynthesizer,
    ) -> None:
        data = make_synthesis_data(tensor=np.eye(3, dtype=np.float64))
        with pytest.raises(PhaseHandoffError, match="so\\(n\\)"):
            synthesizer.handoff_phase2_to_phase3(data)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → CONTINUACIÓN DE Φ₂₃, DIRAC, TRABAJO NILOPOTENTE Y VEREDICTO
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3ReceivesPhase2Handoff:
    """El primer método de la Fase 3 es la continuación literal de Φ₂₃."""

    def test_receive_certified_tensor_equals_phase2_handoff(
        self,
        spectrometer: Phase1_MomentumSpectrometer,
        synthesizer: Phase2_GyroscopicSynthesizer,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        audit = spectrometer.execute_phase1(make_vector(3, seed=149), metric, inverse)
        synthesis = synthesizer.execute_phase2(audit, make_skew_matrix(3, seed=151))
        via_phi = synthesizer.handoff_phase2_to_phase3(synthesis)
        via_phase3 = modulator_phase._receive_certified_gyroscopic_tensor(synthesis)
        np.testing.assert_array_equal(via_phi, via_phase3)

    def test_receive_rejects_non_strict_certificate(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(PhaseHandoffError):
            modulator_phase._receive_certified_gyroscopic_tensor(
                make_synthesis_data(is_strictly_skew=False)
            )


class TestPhase3DiracStructureAndWork:
    """Estructura de Dirac, Neumaier, pares estructurales y sondas de Liouville."""

    def test_validate_dirac_accepts_exact_skew(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        interconnection = make_skew_matrix(5, seed=157, scale=1.4)
        projected, abs_residual, rel_residual = modulator_phase._validate_dirac_structure(
            interconnection
        )
        assert_skew(projected, "J")
        assert abs_residual <= 1.0e-13
        assert rel_residual <= rim._SKEW_RELATIVE_TOLERANCE

    def test_validate_dirac_rejects_symmetric_injection(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(SymplecticWorkViolationError, match="antisimétrica"):
            modulator_phase._validate_dirac_structure(np.eye(3, dtype=np.float64))

    def test_modulate_dirac_is_sum_then_skew(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        interconnection = make_skew_matrix(4, seed=163)
        gyroscopic = make_skew_matrix(4, seed=167, scale=0.4)
        effective = modulator_phase._modulate_dirac_structure(interconnection, gyroscopic)
        expected = 0.5 * (
            (interconnection + gyroscopic) - (interconnection + gyroscopic).T
        )
        np.testing.assert_allclose(effective, expected, atol=1.0e-14)
        assert_skew(effective, "J_eff")

    def test_modulate_dirac_rejects_shape_mismatch(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="misma dimensión"):
            modulator_phase._modulate_dirac_structure(
                make_skew_matrix(3),
                make_skew_matrix(2),
            )

    def test_neumaier_quadratic_form_vanishes_on_so_n(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        skew = make_skew_matrix(6, seed=173, scale=2.0)
        vector = make_vector(6, seed=179, scale=3.0)
        work, abs_sum = modulator_phase._neumaier_quadratic_form(vector, skew)
        assert abs(work) <= 1.0e-10 * max(1.0, abs_sum)
        assert abs_sum >= 0.0

    def test_neumaier_quadratic_form_recovers_squared_norm_on_identity(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        vector = np.array([1.0, -2.0, 2.0], dtype=np.float64)
        work, abs_sum = modulator_phase._neumaier_quadratic_form(
            vector, np.eye(3, dtype=np.float64)
        )
        assert work == pytest.approx(9.0)
        assert abs_sum == pytest.approx(9.0)

    def test_neumaier_quadratic_form_rejects_dimension_mismatch(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(RiemannianInertiaError, match="incompatible"):
            modulator_phase._neumaier_quadratic_form(
                np.array([1.0, 2.0], dtype=np.float64),
                np.eye(3, dtype=np.float64),
            )

    def test_pairwise_structural_work_vanishes_on_so_n(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        skew = make_skew_matrix(5, seed=181)
        vector = make_vector(5, seed=191)
        residual = modulator_phase._pairwise_structural_work(vector, skew)
        assert abs(residual) <= 1.0e-12

    def test_pairwise_structural_work_equals_quadratic_form_for_any_matrix(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        matrix = _rng(193).standard_normal((4, 4)).astype(np.float64)
        vector = make_vector(4, seed=197)
        pairwise = modulator_phase._pairwise_structural_work(vector, matrix)
        naive = float(vector @ matrix @ vector)
        assert pairwise == pytest.approx(naive, rel=1.0e-12, abs=1.0e-12)

    def test_liouville_probes_are_deterministic_and_tiny_on_so_n(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        skew = make_skew_matrix(4, seed=199, scale=1.7)
        first = modulator_phase._liouville_probe_residual(skew)
        second = modulator_phase._liouville_probe_residual(skew)
        assert first == pytest.approx(second)
        assert first <= 1.0e-12

    def test_adaptive_work_tolerance_grows_with_scale(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        small_g = np.array([1.0e-3, 0.0], dtype=np.float64)
        large_g = np.array([1.0e3, 0.0], dtype=np.float64)
        interconnection = make_skew_matrix(2, seed=211, scale=2.0)
        small = modulator_phase._adaptive_work_tolerance(small_g, interconnection, 1.0)
        large = modulator_phase._adaptive_work_tolerance(large_g, interconnection, 1.0)
        assert large > small
        assert small >= 100.0 * rim._MACHINE_EPSILON


class TestPhase3ExecuteAndCertificates:
    """execute_phase3: veredicto termodinámico a partir de Φ₂₃."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    def test_execute_phase3_certifies_nilpotent_work(
        self,
        functor: RiemannianInertiaModulator,
        dimension: int,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=5.0, seed=223)
        velocity = make_vector(dimension, seed=227, scale=0.45)
        gradient = make_vector(dimension, seed=229, scale=0.85)
        interconnection = make_skew_matrix(dimension, seed=233, scale=0.7)
        vorticity = make_skew_matrix(dimension, seed=239, scale=1.05)

        audit = functor.execute_phase1(velocity, metric, inverse)
        synthesis = functor.execute_phase2(audit, vorticity)
        veto = functor.execute_phase3(gradient, interconnection, synthesis)

        assert isinstance(veto, ThermodynamicVetoData)
        assert veto.is_symplectically_passive is True
        assert_readonly(veto.effective_dirac_matrix, "J_eff")
        assert_skew(veto.effective_dirac_matrix, "J_eff")
        assert veto.effective_dirac_matrix.shape == (dimension, dimension)
        assert veto.nilpotent_work_residual <= veto.work_tolerance
        assert veto.pairwise_work_residual <= veto.work_tolerance
        assert veto.liouville_probe_residual <= veto.work_tolerance
        assert veto.dirac_symmetric_residual <= 1.0e-12
        assert veto.relative_skew_residual <= rim._SKEW_RELATIVE_TOLERANCE
        assert veto.work_tolerance > 0.0

        expected = 0.5 * (
            (interconnection + synthesis.skew_symmetric_tensor)
            - (interconnection + synthesis.skew_symmetric_tensor).T
        )
        np.testing.assert_allclose(
            veto.effective_dirac_matrix, expected, atol=1.0e-12
        )

    def test_execute_phase3_rejects_non_synthesis_payload(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(PhaseHandoffError):
            modulator_phase.execute_phase3(
                make_vector(3),
                make_skew_matrix(3),
                object(),  # type: ignore[arg-type]
            )

    def test_execute_phase3_rejects_gradient_dimension_mismatch(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        audit = functor.execute_phase1(make_vector(3, seed=241), metric, inverse)
        synthesis = functor.execute_phase2(audit, make_skew_matrix(3, seed=251))
        with pytest.raises(RiemannianInertiaError, match="incompatible"):
            functor.execute_phase3(
                np.array([1.0, 2.0], dtype=np.float64),
                make_skew_matrix(3),
                synthesis,
            )

    def test_execute_phase3_rejects_non_skew_dirac_structure(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        audit = functor.execute_phase1(make_vector(3, seed=257), metric, inverse)
        synthesis = functor.execute_phase2(audit, make_skew_matrix(3, seed=263))
        with pytest.raises(SymplecticWorkViolationError, match="antisimétrica"):
            functor.execute_phase3(
                make_vector(3, seed=269),
                np.eye(3, dtype=np.float64),
                synthesis,
            )

    def test_certify_nilpotent_work_rejects_symmetric_operator(
        self,
        modulator_phase: Phase3_SymplecticInertiaModulator,
    ) -> None:
        with pytest.raises(SymplecticWorkViolationError, match="componente simétrica"):
            modulator_phase._certify_nilpotent_work(
                make_vector(3, seed=271),
                np.diag([1.0, 2.0, 3.0]).astype(np.float64),
            )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR → FASE 3 ∘ Φ₂₃ ∘ FASE 2 ∘ Φ₁₂ ∘ FASE 1
# ══════════════════════════════════════════════════════════════════════════════
class TestOrchestratorEndToEnd:
    """RiemannianInertiaModulator.apply_inertia_modulation."""

    @pytest.mark.parametrize("dimension", _DIMS_CANONICAL)
    @pytest.mark.parametrize("condition", (1.0, 50.0))
    def test_full_pipeline_preserves_all_invariants(
        self,
        functor: RiemannianInertiaModulator,
        dimension: int,
        condition: float,
    ) -> None:
        metric, inverse = make_spd_pair(dimension, condition=condition, seed=277)
        velocity = make_vector(dimension, seed=281, scale=0.35)
        gradient = make_vector(dimension, seed=283, scale=1.1)
        interconnection = make_skew_matrix(dimension, seed=293, scale=0.65)
        vorticity = make_skew_matrix(dimension, seed=307, scale=1.2)

        veto = functor.apply_inertia_modulation(
            q_dot=velocity,
            grad_H=gradient,
            G_tensor=metric,
            G_inv=inverse,
            J_base=interconnection,
            vorticity_matrix=vorticity,
        )

        assert veto.is_symplectically_passive is True
        assert_skew(veto.effective_dirac_matrix, "J_eff")
        work = float(gradient @ veto.effective_dirac_matrix @ gradient)
        assert abs(work) <= max(veto.work_tolerance, 1.0e-10)
        # Liouville puntual: ningún eje canónico inyecta divergencia diagonal.
        assert np.max(np.abs(np.diag(veto.effective_dirac_matrix))) <= 1.0e-12

    def test_pipeline_is_deterministic(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        first = functor.apply_inertia_modulation(**payload_n3)
        second = functor.apply_inertia_modulation(**payload_n3)
        np.testing.assert_array_equal(
            first.effective_dirac_matrix, second.effective_dirac_matrix
        )
        assert first.nilpotent_work_residual == second.nilpotent_work_residual
        assert first.liouville_probe_residual == second.liouville_probe_residual
        assert first.work_tolerance == second.work_tolerance

    def test_pipeline_rejects_velocity_gradient_mismatch(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        with pytest.raises(RiemannianInertiaError, match="mismo espacio"):
            functor.apply_inertia_modulation(
                q_dot=make_vector(3),
                grad_H=make_vector(2),
                G_tensor=metric,
                G_inv=inverse,
                J_base=make_skew_matrix(3),
                vorticity_matrix=make_skew_matrix(3),
            )

    def test_pipeline_propagates_phase1_metric_failure(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3)
        broken = inverse * 3.0
        with pytest.raises(MetricCoherenceError):
            functor.apply_inertia_modulation(
                q_dot=make_vector(3),
                grad_H=make_vector(3, seed=4),
                G_tensor=metric,
                G_inv=broken,
                J_base=make_skew_matrix(3),
                vorticity_matrix=make_skew_matrix(3),
            )

    def test_pipeline_propagates_phase1_divergence(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(2, condition=1.0, seed=2)
        velocity = np.full(2, rim._MOMENTUM_MAX_BOUND * 3.0, dtype=np.float64)
        with pytest.raises(MomentumDivergenceError):
            functor.apply_inertia_modulation(
                q_dot=velocity,
                grad_H=make_vector(2, seed=5),
                G_tensor=metric,
                G_inv=inverse,
                J_base=make_skew_matrix(2),
                vorticity_matrix=make_skew_matrix(2),
            )

    def test_pipeline_propagates_phase3_dirac_failure(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        payload = dict(payload_n3)
        payload["J_base"] = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(SymplecticWorkViolationError):
            functor.apply_inertia_modulation(**payload)

    def test_general_vorticity_is_projected_and_still_passive(
        self,
        functor: RiemannianInertiaModulator,
    ) -> None:
        metric, inverse = make_spd_pair(3, condition=4.0, seed=311)
        mixed_vorticity = _rng(313).standard_normal((3, 3)).astype(np.float64)
        veto = functor.apply_inertia_modulation(
            q_dot=make_vector(3, seed=317, scale=0.4),
            grad_H=make_vector(3, seed=331, scale=0.9),
            G_tensor=metric,
            G_inv=inverse,
            J_base=make_skew_matrix(3, seed=337),
            vorticity_matrix=mixed_vorticity,
        )
        assert veto.is_symplectically_passive is True
        assert_skew(veto.effective_dirac_matrix, "J_eff")


class TestGlobalInvariantsAcrossPhases:
    """Invariantes [I1]–[I5] medidos sobre la composición completa."""

    def test_i1_liouville_quadratic_form_is_identically_nilpotent(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        veto = functor.apply_inertia_modulation(**payload_n3)
        generator = _rng(347)
        worst = 0.0
        for _ in range(32):
            probe = generator.standard_normal(3).astype(np.float64)
            probe /= max(float(np.linalg.norm(probe)), 1.0e-18)
            worst = max(
                worst,
                abs(float(probe @ veto.effective_dirac_matrix @ probe)),
            )
        assert worst <= max(veto.work_tolerance, 1.0e-11)

    def test_i2_musical_roundtrip_and_wilkinson_on_certificate(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        audit = functor.execute_phase1(
            payload_n3["q_dot"], payload_n3["G_tensor"], payload_n3["G_inv"]
        )
        assert audit.musical_roundtrip_residual <= 1.0e-10
        assert audit.inverse_consistency_residual <= audit.wilkinson_bound * 10.0
        identity = np.eye(3, dtype=np.float64)
        np.testing.assert_allclose(
            payload_n3["G_tensor"] @ payload_n3["G_inv"],
            identity,
            atol=1.0e-10,
        )

    def test_i3_work_is_orthogonal_to_hamiltonian_gradient(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        veto = functor.apply_inertia_modulation(**payload_n3)
        injected = veto.effective_dirac_matrix @ payload_n3["grad_H"]
        pairing = float(payload_n3["grad_H"] @ injected)
        assert abs(pairing) <= veto.work_tolerance

    def test_i4_gram_identity_on_phase2_certificate(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        audit = functor.execute_phase1(
            payload_n3["q_dot"], payload_n3["G_tensor"], payload_n3["G_inv"]
        )
        synthesis = functor.execute_phase2(audit, payload_n3["vorticity_matrix"])
        assert synthesis.wedge_gram_residual <= 1.0e-10

    def test_i5_gyroscopic_tensor_lives_in_so_n(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        audit = functor.execute_phase1(
            payload_n3["q_dot"], payload_n3["G_tensor"], payload_n3["G_inv"]
        )
        synthesis = functor.execute_phase2(audit, payload_n3["vorticity_matrix"])
        tensor = synthesis.skew_symmetric_tensor
        assert_skew(tensor, "W")
        spectrum = np.linalg.eigvals(tensor)
        # Espectro de so(n): puramente imaginario (o nulo).
        assert np.max(np.abs(spectrum.real)) <= 1.0e-10
        if synthesis.skew_numerical_rank > 0:
            assert synthesis.skew_numerical_rank % 2 == 0 or not synthesis.rank_is_even

    def test_nested_handoffs_compose_like_the_orchestrator(
        self,
        functor: RiemannianInertiaModulator,
        payload_n3: dict[str, NDArray[np.float64]],
    ) -> None:
        """
        Verifica el contrato documental:

            execute_phase3 ∘ Φ₂₃ ∘ execute_phase2 ∘ Φ₁₂ ∘ execute_phase1
                ≡ apply_inertia_modulation
        """
        audit = functor.execute_phase1(
            payload_n3["q_dot"], payload_n3["G_tensor"], payload_n3["G_inv"]
        )
        handed_p = functor.handoff_phase1_to_phase2(audit)
        np.testing.assert_array_equal(handed_p, audit.covariant_momentum)

        synthesis = functor.execute_phase2(audit, payload_n3["vorticity_matrix"])
        handed_w = functor.handoff_phase2_to_phase3(synthesis)
        np.testing.assert_array_equal(handed_w, synthesis.skew_symmetric_tensor)

        composed = functor.execute_phase3(
            payload_n3["grad_H"], payload_n3["J_base"], synthesis
        )
        orchestrated = functor.apply_inertia_modulation(**payload_n3)
        np.testing.assert_allclose(
            composed.effective_dirac_matrix,
            orchestrated.effective_dirac_matrix,
            atol=1.0e-14,
        )
        assert composed.nilpotent_work_residual == pytest.approx(
            orchestrated.nilpotent_work_residual
        )
        assert composed.is_symplectically_passive is True
        assert orchestrated.is_symplectically_passive is True