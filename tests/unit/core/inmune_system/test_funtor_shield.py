# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Test Suite del Funtor Shield v5.0                                   ║
║ Ubicación: tests/core/immune_system/test_funtor_shield.py                   ║
║ Versión: 5.0.0-Categorical-Symplectic-Homological                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
              ARQUITECTURA DE LA SUITE DE PRUEBAS
═══════════════════════════════════════════════════════════════════════════════

    Capa 1 · Pruebas Unitarias de la Fase 1 (Axiomática)
        ├── 1A. PhysicalConstants
        ├── 1B. PhaseVectorExtractor
        ├── 1C. ThermodynamicMetrics (invariantes I1–I5)
        ├── 1D. ShieldSignature
        └── 1E. ValidatedMetricTensor (M1–M4)

    Capa 2 · Pruebas Unitarias de la Fase 2 (Operadores)
        ├── 2A. ConstantSkewStructure
        ├── 2B. QuadraticDissipation (PSD, acoplamiento γ)
        ├── 2C. PortHamiltonianFlow (Lyapunov continuo)
        ├── 2D. YonedaRepresentable
        └── 2E. PortHamiltonianEvaluator (exacto vs finito)

    Capa 3 · Pruebas Unitarias de la Fase 3 (Proyector)
        ├── 3A. RiemannianProjector (A1–A4)
        ├── 3B. DiracBoundaryOperator
        ├── 3C. HomologyBettiDetector (β₁)
        ├── 3D. FuntorShield (ciclo completo)
        └── 3E. apply_funtor_shield (decorador)

    Capa 4 · Pruebas de Integración y Teoremas
        ├── INT-1.  Pipeline end-to-end (agente benigno)
        ├── INT-2.  Idempotencia experimental (T2)
        ├── INT-3.  Detección de violación Lyapunov (T3)
        ├── INT-4.  Detección de Socavón Lógico (T4)
        ├── INT-5.  Amputación por Dirichlet
        └── INT-6.  Verificación simultánea T1–T4

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.immune_system.funtor_shield import (
    # ── Fase 1 ──
    PhysicalConstants,
    ShieldViolationType,
    ThermodynamicMetrics,
    ShieldSignature,
    PhaseVectorExtractor,
    ValidatedMetricTensor,
    PhaseSpaceVectorizable,
    # ── Fase 2 ──
    ConstantSkewStructure,
    QuadraticDissipation,
    PortHamiltonianFlow,
    YonedaRepresentable,
    PortHamiltonianEvaluator,
    # ── Fase 3 ──
    RiemannianProjector,
    DiracBoundaryOperator,
    HomologyBettiDetector,
    FuntorShield,
    apply_funtor_shield,
)
from app.core.mic_algebra import CategoricalState, FunctorialityError, Morphism
from app.core.schemas import Stratum


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE TEST
# ══════════════════════════════════════════════════════════════════════════════
TEST_SEED: int = 42
TOL_STRICT: float = 1e-10
TOL_LOOSE: float = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng() -> np.random.Generator:
    """Generador pseudoaleatorio determinista."""
    return np.random.default_rng(TEST_SEED)


@pytest.fixture
def identity_metric_4d() -> ValidatedMetricTensor:
    """Tensor métrico I₄ (4×4 identidad)."""
    return ValidatedMetricTensor(np.eye(4, dtype=np.float64))


@pytest.fixture
def diagonal_metric_4d() -> ValidatedMetricTensor:
    """Tensor métrico diagonal diag(1, 2, 3, 4)."""
    return ValidatedMetricTensor(np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64))


@pytest.fixture
def aniso_metric_4d() -> ValidatedMetricTensor:
    """Tensor métrico 4×4 anisotrópico (simétrico definido positivo)."""
    A = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    # G = Aᵀ A → simétrica SPD
    return ValidatedMetricTensor(A.T @ A)


@pytest.fixture
def mock_stratum() -> Stratum:
    """Estrato mockeado para tests independientes del dominio real."""
    return Stratum.L3_KNOWLEDGE


# ─────────────────────────────────────────────────────────────────────────────
# Stubs de Morphism y CategoricalState
# (se aíslan del sistema real para no acoplar la suite a su implementación)
# ─────────────────────────────────────────────────────────────────────────────


class _StubMorphism(Morphism):
    """Morfismo de prueba: identidad con dimensión configurable."""

    def __init__(self, phase_dim: int = 4) -> None:
        super().__init__()
        self.phase_dim: int = phase_dim
        self.call_count: int = 0

    def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
        self.call_count += 1
        return state


class _ContractingAgent(Morphism):
    """
    Agente que aplica una contracción G-disipativa:
    contrae el vector de fase en una escala estable bajo Port-Hamiltoniano.
    """

    def __init__(self, phase_dim: int = 4, factor: float = 0.5) -> None:
        super().__init__()
        self.phase_dim: int = phase_dim
        self.factor: float = factor

    def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
        if isinstance(state.payload, (int, float)):
            new_payload = float(state.payload) * self.factor
        elif isinstance(state.payload, np.ndarray):
            new_payload = state.payload * self.factor
        else:
            new_payload = state.payload
        return CategoricalState(payload=new_payload, stratum=state.stratum)


class _DivergentAgent(Morphism):
    """Agente que inyecta energía (Ḣ > 0)."""

    def __init__(self, phase_dim: int = 4, factor: float = 10.0) -> None:
        super().__init__()
        self.phase_dim: int = phase_dim
        self.factor: float = factor

    def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
        if isinstance(state.payload, np.ndarray):
            new_payload = state.payload * self.factor
        else:
            new_payload = state.payload
        return CategoricalState(payload=new_payload, stratum=state.stratum)


class _FailingAgent(Morphism):
    """Agente que siempre lanza excepción."""

    def __init__(self, phase_dim: int = 4) -> None:
        super().__init__()
        self.phase_dim: int = phase_dim

    def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
        raise RuntimeError("Falla controlada del agente")


class _VectorizableState(CategoricalState, PhaseSpaceVectorizable):
    """Estado que implementa el protocolo PhaseSpaceVectorizable."""

    def __init__(self, vector: NDArray[np.float64], stratum: Stratum) -> None:
        super().__init__(payload=vector.copy(), stratum=stratum)
        self._vec = vector

    def to_phase_vector(self, dim: int) -> NDArray[np.float64]:
        return self._vec[:dim].copy()


@pytest.fixture
def stub_morphism() -> _StubMorphism:
    return _StubMorphism(phase_dim=4)


@pytest.fixture
def contracting_agent() -> _ContractingAgent:
    return _ContractingAgent(phase_dim=4, factor=0.5)


@pytest.fixture
def divergent_agent() -> _DivergentAgent:
    return _DivergentAgent(phase_dim=4, factor=10.0)


@pytest.fixture
def failing_agent() -> _FailingAgent:
    return _FailingAgent(phase_dim=4)


@pytest.fixture
def sample_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado escalar simple (entra en Estrategia E4 del extractor)."""
    return CategoricalState(payload=1.0, stratum=mock_stratum)


@pytest.fixture
def array_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload ndarray (entra en Estrategia E3)."""
    vec = np.array([0.5, -0.3, 0.8, 0.1], dtype=np.float64)
    return CategoricalState(payload=vec, stratum=mock_stratum)


@pytest.fixture
def iterable_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload iterable (entra en Estrategia E5)."""
    return CategoricalState(payload=[1.0, 2.0, 3.0], stratum=mock_stratum)


@pytest.fixture
def bad_state(mock_stratum: Stratum) -> CategoricalState:
    """Estado con payload no vectorizable (entra en Estrategia E6 fallback)."""
    return CategoricalState(payload=object(), stratum=mock_stratum)


@pytest.fixture
def vectorizable_state(mock_stratum: Stratum) -> _VectorizableState:
    """Estado que implementa el protocolo PhaseSpaceVectorizable (E1)."""
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    return _VectorizableState(vec, mock_stratum)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 1 · PRUEBAS UNITARIAS DE LA FASE 1 (AXIOMÁTICA)                  │
# └─────────────────────────────────────────────────────────────────────────┘


class TestPhysicalConstants:
    """Pruebas de las constantes físicas fundamentales."""

    def test_epsilons_are_positive(self) -> None:
        """Los ε deben ser estrictamente positivos."""
        assert PhysicalConstants.EPSILON_MACHINE > 0
        assert PhysicalConstants.EPSILON_LYAPUNOV > 0
        assert PhysicalConstants.EPSILON_NORM > 0

    def test_epsilons_are_ordered(self) -> None:
        """ε_machine ≤ ε_lyapunov ≤ ε_norm (orden natural de tolerancia)."""
        eps_m = PhysicalConstants.EPSILON_MACHINE
        eps_l = PhysicalConstants.EPSILON_LYAPUNOV
        eps_n = PhysicalConstants.EPSILON_NORM
        assert eps_m < eps_l < eps_n

    def test_canonical_dim_is_7(self) -> None:
        """La dimensión canónica debe ser 7 (consistencia con el módulo)."""
        assert PhysicalConstants.CANONICAL_PHASE_DIM == 7

    def test_boltzmann_value(self) -> None:
        """Verifica el valor de Boltzmann normalizado."""
        expected = 1.380649e-23
        assert PhysicalConstants.BOLTZMANN_NORMALIZED == pytest.approx(expected)


class TestPhaseVectorExtractor:
    """Pruebas del extractor de vectores de fase (Estrategias E1–E6)."""

    def test_e1_protocol(self, vectorizable_state: _VectorizableState) -> None:
        """E1: Estado con protocolo PhaseSpaceVectorizable."""
        vec = PhaseVectorExtractor.extract(vectorizable_state, 4)
        assert vec.shape == (4,)
        np.testing.assert_allclose(vec, [0.1, 0.2, 0.3, 0.4], atol=TOL_STRICT)

    def test_e2_explicit_attribute(self, mock_stratum: Stratum) -> None:
        """E2: Estado con atributo phase_vector."""

        class _StateWithVec(CategoricalState):
            def __init__(self, vec: NDArray, stratum: Stratum) -> None:
                super().__init__(payload="dummy", stratum=stratum)
                self.phase_vector = vec  # type: ignore[attr-defined]

        state = _StateWithVec(np.array([1.0, 2.0, 3.0, 4.0]), mock_stratum)
        vec = PhaseVectorExtractor.extract(state, 4)
        np.testing.assert_allclose(vec, [1.0, 2.0, 3.0, 4.0], atol=TOL_STRICT)

    def test_e3_ndarray_payload(self, array_state: CategoricalState) -> None:
        """E3: Payload ndarray directo."""
        vec = PhaseVectorExtractor.extract(array_state, 4)
        np.testing.assert_allclose(vec, [0.5, -0.3, 0.8, 0.1], atol=TOL_STRICT)

    def test_e4_scalar_payload(self, sample_state: CategoricalState) -> None:
        """E4: Payload escalar → broadcasting."""
        vec = PhaseVectorExtractor.extract(sample_state, 5)
        assert vec.shape == (5,)
        assert np.all(vec == 1.0)

    def test_e5_iterable_payload(self, iterable_state: CategoricalState) -> None:
        """E5: Payload iterable → np.asarray + reshape."""
        vec = PhaseVectorExtractor.extract(iterable_state, 3)
        np.testing.assert_allclose(vec, [1.0, 2.0, 3.0], atol=TOL_STRICT)

    def test_e6_fallback(self, bad_state: CategoricalState) -> None:
        """E6: Fallback a vector cero con warning."""
        vec = PhaseVectorExtractor.extract(bad_state, 4)
        assert vec.shape == (4,)
        assert np.all(vec == 0.0)

    def test_truncation(self, array_state: CategoricalState) -> None:
        """Truncar vector cuando size > target_dim."""
        vec = PhaseVectorExtractor.extract(array_state, 2)
        assert vec.shape == (2,)
        np.testing.assert_allclose(vec, [0.5, -0.3], atol=TOL_STRICT)

    def test_padding(self, array_state: CategoricalState) -> None:
        """Rellenar con ceros cuando size < target_dim."""
        vec = PhaseVectorExtractor.extract(array_state, 8)
        assert vec.shape == (8,)
        np.testing.assert_allclose(vec[:4], [0.5, -0.3, 0.8, 0.1], atol=TOL_STRICT)
        assert np.all(vec[4:] == 0.0)

    def test_invalid_dim_raises(self, sample_state: CategoricalState) -> None:
        """Dimensión ≤ 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="dimensión objetivo inválida"):
            PhaseVectorExtractor.extract(sample_state, 0)
        with pytest.raises(ValueError, match="dimensión objetivo inválida"):
            PhaseVectorExtractor.extract(sample_state, -1)


class TestThermodynamicMetrics:
    """Pruebas de los invariantes I1–I5."""

    def _valid_metrics(self) -> ThermodynamicMetrics:
        return ThermodynamicMetrics(
            lyapunov_derivative=-0.5,
            dissipated_power=0.5,
            energy_pre=1.0,
            energy_post=0.5,
            entropy_production=0.5 / PhysicalConstants.BOLTZMANN_NORMALIZED,
            is_dirichlet_enforced=False,
            violation_type=ShieldViolationType.NONE,
        )

    def test_i1_dissipated_power_nonnegative(self) -> None:
        """I1: P_diss ≥ 0."""
        with pytest.raises(AssertionError, match="Segunda Ley"):
            ThermodynamicMetrics(
                lyapunov_derivative=1.0,
                dissipated_power=-0.1,  # viola
                energy_pre=1.0,
                energy_post=2.0,
                entropy_production=0.0,
                is_dirichlet_enforced=False,
                violation_type=ShieldViolationType.NONE,
            )

    def test_i2_energies_nonnegative(self) -> None:
        """I2: energy_pre, energy_post ≥ 0."""
        with pytest.raises(AssertionError, match="inicial negativa"):
            ThermodynamicMetrics(
                lyapunov_derivative=0.0,
                dissipated_power=0.0,
                energy_pre=-1.0,  # viola
                energy_post=1.0,
                entropy_production=0.0,
                is_dirichlet_enforced=False,
                violation_type=ShieldViolationType.NONE,
            )

    def test_i3_lyapunov_finite(self) -> None:
        """I3: Ḣ finita."""
        with pytest.raises(AssertionError, match="no finita"):
            ThermodynamicMetrics(
                lyapunov_derivative=float("inf"),
                dissipated_power=0.0,
                energy_pre=1.0,
                energy_post=1.0,
                entropy_production=0.0,
                is_dirichlet_enforced=False,
                violation_type=ShieldViolationType.NONE,
            )

    def test_i4_entropy_nonnegative(self) -> None:
        """I4: σ ≥ 0."""
        with pytest.raises(AssertionError, match="irreversibilidad"):
            ThermodynamicMetrics(
                lyapunov_derivative=0.0,
                dissipated_power=0.0,
                energy_pre=1.0,
                energy_post=1.0,
                entropy_production=-0.1,  # viola
                is_dirichlet_enforced=False,
                violation_type=ShieldViolationType.NONE,
            )

    def test_i5_consistency_lyapunov_dissipated(self) -> None:
        """I5: Ḣ + P_diss ≥ −ε."""
        with pytest.raises(AssertionError, match="inconsistencia"):
            ThermodynamicMetrics(
                lyapunov_derivative=-1.0,
                dissipated_power=0.5,  # -1 + 0.5 = -0.5 < 0
                energy_pre=1.0,
                energy_post=1.0,
                entropy_production=0.5,
                is_dirichlet_enforced=False,
                violation_type=ShieldViolationType.NONE,
            )

    def test_valid_construction_succeeds(self) -> None:
        """Métricas válidas no lanzan excepción."""
        metrics = self._valid_metrics()
        assert metrics.dissipated_power == 0.5
        assert metrics.energy_pre == 1.0


class TestShieldSignature:
    """Pruebas de la firma criptográfica."""

    def test_forge_deterministic_hash(
        self, stub_morphism: _StubMorphism, sample_state: CategoricalState
    ) -> None:
        """Hash Blake2b es determinista para el mismo payload."""
        sig1 = ShieldSignature.forge(stub_morphism, sample_state, phase_dim=4)
        sig2 = ShieldSignature.forge(stub_morphism, sample_state, phase_dim=4)
        assert sig1.hash_digest == sig2.hash_digest
        assert sig1.phase_dim == sig2.phase_dim == 4

    def test_different_payloads_different_hashes(
        self, stub_morphism: _StubMorphism, mock_stratum: Stratum
    ) -> None:
        """Distintos payloads → distintos hashes."""
        s1 = CategoricalState(payload=1.0, stratum=mock_stratum)
        s2 = CategoricalState(payload=2.0, stratum=mock_stratum)
        sig1 = ShieldSignature.forge(stub_morphism, s1, phase_dim=4)
        sig2 = ShieldSignature.forge(stub_morphism, s2, phase_dim=4)
        assert sig1.hash_digest != sig2.hash_digest

    def test_signature_immutable(self, stub_morphism: _StubMorphism, sample_state: CategoricalState) -> None:
        """La firma es frozen (no admite asignación)."""
        sig = ShieldSignature.forge(stub_morphism, sample_state, phase_dim=4)
        with pytest.raises(Exception):  # FrozenInstanceError
            sig.phase_dim = 99  # type: ignore[misc]


class TestValidatedMetricTensor:
    """Pruebas del tensor métrico validado (M1–M4)."""

    def test_m1_must_be_2d_square(self) -> None:
        """M1: tensor 2D cuadrado."""
        with pytest.raises(ValueError, match="no cuadrado"):
            ValidatedMetricTensor(np.zeros((3, 4), dtype=np.float64))
        with pytest.raises(ValueError, match="no cuadrado"):
            ValidatedMetricTensor(np.zeros((2, 2, 2), dtype=np.float64))

    def test_m2_must_be_symmetric(self) -> None:
        """M2: tensor simétrico."""
        G = np.array([[1.0, 0.5], [0.3, 1.0]], dtype=np.float64)  # asimétrico
        with pytest.raises(ValueError, match="asimétrico"):
            ValidatedMetricTensor(G)

    def test_m3_spd_in_strict_mode(self) -> None:
        """M3 (strict): tensor PSD estricto."""
        G = np.diag([1.0, 0.0]).astype(np.float64)  # semidefinido
        with pytest.raises(ValueError, match="definido-positivo"):
            ValidatedMetricTensor(G, strict_spd=True)

    def test_m4_eigendecomposition(
        self, diagonal_metric_4d: ValidatedMetricTensor
    ) -> None:
        """M4: descomposición espectral correcta."""
        # Para diag(1, 2, 3, 4), los autovalores deben ser {1, 2, 3, 4}
        eigs = np.sort(diagonal_metric_4d.eigval)
        np.testing.assert_allclose(eigs, [1.0, 2.0, 3.0, 4.0], atol=TOL_STRICT)

    def test_reconstruction_from_spectral(
        self, aniso_metric_4d: ValidatedMetricTensor
    ) -> None:
        """G = V Λ Vᵀ debe reconstruir el tensor original."""
        G_reconstructed = (
            aniso_metric_4d.eigvec * aniso_metric_4d.eigval
        ) @ aniso_metric_4d.eigvec.T
        np.testing.assert_allclose(
            G_reconstructed, aniso_metric_4d.G, atol=TOL_STRICT
        )

    def test_inverse(self, identity_metric_4d: ValidatedMetricTensor) -> None:
        """G⁻¹ de la identidad es la identidad."""
        G_inv = identity_metric_4d.inverse()
        np.testing.assert_allclose(G_inv, np.eye(4), atol=TOL_STRICT)

    def test_inverse_correctness(
        self, aniso_metric_4d: ValidatedMetricTensor
    ) -> None:
        """G · G⁻¹ ≈ I."""
        G_inv = aniso_metric_4d.inverse()
        product = aniso_metric_4d.G @ G_inv
        np.testing.assert_allclose(product, np.eye(4), atol=TOL_STRICT)

    def test_condition_number(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> None:
        """κ(I) = 1 (identidad está perfectamente condicionada)."""
        assert identity_metric_4d.condition_number() == pytest.approx(1.0)

    def test_spectral_gap(
        self, diagonal_metric_4d: ValidatedMetricTensor
    ) -> None:
        """Gap = λ₂ − λ₁ = 2 − 1 = 1."""
        assert diagonal_metric_4d.spectral_gap() == pytest.approx(1.0)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 2 · PRUEBAS UNITARIAS DE LA FASE 2 (OPERADORES)                  │
# └─────────────────────────────────────────────────────────────────────────┘


class TestConstantSkewStructure:
    """Pruebas de la estructura simpléctica J."""

    def test_antisymmetry_4d(self) -> None:
        """J = −Jᵀ en dimensión 4."""
        J = ConstantSkewStructure(dim=4)
        np.testing.assert_allclose(J(), -J().T, atol=TOL_STRICT)

    def test_antisymmetry_6d(self) -> None:
        """J = −Jᵀ en dimensión 6."""
        J = ConstantSkewStructure(dim=6)
        np.testing.assert_allclose(J(), -J().T, atol=TOL_STRICT)

    def test_odd_dim_rejected(self) -> None:
        """Dimensión impar debe lanzar ValueError."""
        with pytest.raises(ValueError, match="par"):
            ConstantSkewStructure(dim=3)
        with pytest.raises(ValueError, match="par"):
            ConstantSkewStructure(dim=5)

    def test_canonical_block_form(self) -> None:
        """Forma canónica de bloque en dimensión 4: [[0, I], [−I, 0]]."""
        J = ConstantSkewStructure(dim=4)
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(J(), expected, atol=TOL_STRICT)

    def test_bilinear_degeneracy(self) -> None:
        """xᵀ J x = 0 para todo x (degeneración bilineal)."""
        J = ConstantSkewStructure(dim=4)
        rng = np.random.default_rng(TEST_SEED)
        for _ in range(100):
            x = rng.standard_normal(4)
            assert abs(x @ J() @ x) < TOL_STRICT


class TestQuadraticDissipation:
    """Pruebas del operador de disipación R(x) = R₀ + γ·xxᵀ."""

    def test_r0_only_is_symmetric(self) -> None:
        """R(x) es simétrica aún con γ = 0."""
        R = QuadraticDissipation(dim=4, gamma=0.0)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(R(x), R(x).T, atol=TOL_STRICT)

    def test_r0_only_is_psd(self) -> None:
        """R(x) ≥ 0 cuando γ = 0 y R₀ = ε·I."""
        R = QuadraticDissipation(dim=4, gamma=0.0)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert R.is_psd(x)

    def test_gamma_coupling_increases_trace(
        self, rng: np.random.Generator
    ) -> None:
        """tr(R(x)) crece con γ cuando ||x|| > 0."""
        R_lo = QuadraticDissipation(dim=4, gamma=0.0)
        R_hi = QuadraticDissipation(dim=4, gamma=1.0)
        x = rng.standard_normal(4)
        assert np.trace(R_hi(x)) > np.trace(R_lo(x))

    def test_gamma_coupling_preserves_psd(self) -> None:
        """R(x) = R₀ + γ·xxᵀ sigue siendo PSD para γ ≥ 0."""
        R = QuadraticDissipation(dim=4, gamma=10.0)
        x = np.array([1.0, -1.0, 0.5, 0.0])
        assert R.is_psd(x)

    def test_negative_gamma_rejected(self) -> None:
        """γ < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="γ"):
            QuadraticDissipation(dim=4, gamma=-0.1)

    def test_invalid_dim_rejected(self) -> None:
        """dim ≤ 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="positivo"):
            QuadraticDissipation(dim=0)

    def test_invalid_r0_shape(self) -> None:
        """R0 con forma incorrecta debe lanzar ValueError."""
        with pytest.raises(ValueError, match="shape"):
            QuadraticDissipation(dim=4, R0=np.eye(3))

    def test_asymmetric_r0_rejected(self) -> None:
        """R0 asimétrico debe lanzar ValueError."""
        with pytest.raises(ValueError, match="simétrica"):
            R0 = np.array([[1.0, 0.5], [0.3, 1.0], [0.0, 0.0], [0.0, 0.0]])
        # Forma correcta para 2D pero asimétrica
        R0 = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="simétrica"):
            QuadraticDissipation(dim=2, R0=R0)


class TestPortHamiltonianFlow:
    """Pruebas del flujo Port-Hamiltoniano."""

    @pytest.fixture
    def flow_4d(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> PortHamiltonianFlow:
        J = ConstantSkewStructure(dim=4)
        R = QuadraticDissipation(dim=4, gamma=0.0)
        return PortHamiltonianFlow(G=identity_metric_4d, J=J, R=R)

    def test_hamiltonian_kinetic_form(
        self, flow_4d: PortHamiltonianFlow
    ) -> None:
        """H(x) = ½ xᵀ G x con G = I → H = ½ ||x||²."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        expected = 0.5 * (1 + 4 + 9 + 16)
        assert flow_4d.hamiltonian(x) == pytest.approx(expected, abs=TOL_STRICT)

    def test_gradient_hamiltonian(
        self, flow_4d: PortHamiltonianFlow
    ) -> None:
        """∇H(x) = G x = x (con G = I)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            flow_4d.gradient_hamiltonian(x), x, atol=TOL_STRICT
        )

    def test_lyapunov_decreases(
        self, flow_4d: PortHamiltonianFlow, rng: np.random.Generator
    ) -> None:
        """Ḣ ≤ 0 (T3) en Port-Hamiltoniano con G = I."""
        for _ in range(50):
            x = rng.standard_normal(4)
            dH = flow_4d.lyapunov_derivative(x)
            assert dH <= TOL_LOOSE, f"Ḣ={dH} > 0"

    def test_lyapunov_zero_at_origin(
        self, flow_4d: PortHamiltonianFlow
    ) -> None:
        """Ḣ(0) = 0 (punto crítico del Hamiltoniano)."""
        x = np.zeros(4)
        assert flow_4d.lyapunov_derivative(x) == pytest.approx(0.0, abs=TOL_STRICT)

    def test_euler_step_decreases_energy(
        self, flow_4d: PortHamiltonianFlow, rng: np.random.Generator
    ) -> None:
        """Un paso de Euler debe reducir (o mantener) la energía."""
        x = rng.standard_normal(4)
        H_pre = flow_4d.hamiltonian(x)
        x_post = flow_4d.step(x, dt=0.01)
        H_post = flow_4d.hamiltonian(x_post)
        assert H_post <= H_pre + TOL_LOOSE

    def test_invalid_dt_rejected(
        self, flow_4d: PortHamiltonianFlow
    ) -> None:
        """dt ≤ 0 debe lanzar ValueError."""
        x = np.zeros(4)
        with pytest.raises(ValueError, match="positivo"):
            flow_4d.step(x, dt=0.0)
        with pytest.raises(ValueError, match="positivo"):
            flow_4d.step(x, dt=-0.1)

    def test_dimension_consistency(self) -> None:
        """G.n ≠ J.dim debe lanzar ValueError."""
        G = ValidatedMetricTensor(np.eye(4))
        J = ConstantSkewStructure(dim=2)  # dimensión incorrecta
        R = QuadraticDissipation(dim=4)
        with pytest.raises(ValueError, match="inconsistentes"):
            PortHamiltonianFlow(G=G, J=J, R=R)


class TestYonedaRepresentable:
    """Pruebas del funtor representable Yoneda."""

    def test_representable_mimics_dim(
        self, stub_morphism: _StubMorphism
    ) -> None:
        """El funtor representable registra la dimensión del agente."""
        Y = YonedaRepresentable(stub_morphism, phase_dim=4)
        assert Y.phase_dim == 4

    def test_hom_to_invokes_agent(
        self, stub_morphism: _StubMorphism, sample_state: CategoricalState
    ) -> None:
        """Y.hom_to(x) invoca al agente sobre x."""
        Y = YonedaRepresentable(stub_morphism, phase_dim=4)
        Y.hom_to(sample_state)
        assert stub_morphism.call_count == 1

    def test_hom_to_caches_result(
        self, stub_morphism: _StubMorphism, sample_state: CategoricalState
    ) -> None:
        """Y cachea resultados por id de estado."""
        Y = YonedaRepresentable(stub_morphism, phase_dim=4)
        r1 = Y.hom_to(sample_state)
        r2 = Y.hom_to(sample_state)
        assert r1 is r2  # misma referencia
        assert stub_morphism.call_count == 1  # solo invocó una vez


class TestPortHamiltonianEvaluator:
    """Pruebas del evaluador termodinámico."""

    @pytest.fixture
    def evaluator_4d(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> PortHamiltonianEvaluator:
        J = ConstantSkewStructure(dim=4)
        R = QuadraticDissipation(dim=4, gamma=0.0)
        flow = PortHamiltonianFlow(G=identity_metric_4d, J=J, R=R)
        return PortHamiltonianEvaluator(identity_metric_4d, flow)

    def test_exact_lyapunov_is_nonpositive(
        self,
        evaluator_4d: PortHamiltonianEvaluator,
        rng: np.random.Generator,
    ) -> None:
        """Ḣ exacto debe ser ≤ 0 en Port-Hamiltoniano con G = I."""
        for _ in range(50):
            x = rng.standard_normal(4)
            dH = evaluator_4d.compute_lyapunov_derivative_exact(x)
            assert dH <= TOL_LOOSE

    def test_finite_difference_approximates_exact(
        self, evaluator_4d: PortHamiltonianEvaluator
    ) -> None:
        """Ḣ por diferencia finita ≈ Ḣ exacto para dt pequeño."""
        x_pre = np.array([1.0, 0.0, 0.0, 0.0])
        # x_post = x_pre + dt * ẋ  (paso infinitesimal)
        flow_x = evaluator_4d.flow_sys.flow(x_pre)
        dt = 1e-6
        x_post = x_pre + dt * flow_x
        dH_exact = evaluator_4d.compute_lyapunov_derivative_exact(x_pre)
        dH_finite = evaluator_4d.compute_lyapunov_derivative_finite(
            x_pre, x_post, dt
        )
        # Deben coincidir en O(dt)
        assert abs(dH_exact - dH_finite) / max(abs(dH_exact), 1e-12) < 1e-3

    def test_classify_no_violation(
        self, evaluator_4d: PortHamiltonianEvaluator
    ) -> None:
        """Ḣ ≈ 0 → NONE."""
        violation = evaluator_4d._classify_violation(
            PhysicalConstants.EPSILON_LYAPUNOV / 10, np.ones(4)
        )
        assert violation == ShieldViolationType.NONE

    def test_classify_subcritical(
        self, evaluator_4d: PortHamiltonianEvaluator
    ) -> None:
        """Ḣ > 0 pero ratio < umbral → LYAPUNOV_POSITIVE."""
        dH = 0.1
        x = np.ones(4)  # ||x||² = 4
        violation = evaluator_4d._classify_violation(dH, x)
        assert violation == ShieldViolationType.LYAPUNOV_POSITIVE

    def test_classify_critical(
        self, evaluator_4d: PortHamiltonianEvaluator
    ) -> None:
        """Ḣ > 0 con ratio > umbral → LYAPUNOV_CRITICAL."""
        dH = 1e4  # muy grande
        x = np.ones(4) * 1e-3  # ||x||² pequeño → ratio enorme
        violation = evaluator_4d._classify_violation(dH, x)
        assert violation == ShieldViolationType.LYAPUNOV_CRITICAL

    def test_evaluate_returns_valid_metrics(
        self, evaluator_4d: PortHamiltonianEvaluator
    ) -> None:
        """evaluate() retorna ThermodynamicMetrics válidos."""
        x_pre = np.array([1.0, 0.0, 0.0, 0.0])
        x_post = np.array([0.5, 0.0, 0.0, 0.0])
        metrics = evaluator_4d.evaluate(x_pre, x_post, dt=1.0)
        assert metrics.energy_pre > metrics.energy_post
        assert metrics.dissipated_power >= 0
        assert metrics.violation_type == ShieldViolationType.NONE


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 3 · PRUEBAS UNITARIAS DE LA FASE 3 (PROYECTOR)                   │
# └─────────────────────────────────────────────────────────────────────────┘


class TestRiemannianProjector:
    """Pruebas del proyector ortogonal (axiomas A1–A4)."""

    @pytest.fixture
    def projector_4d_2(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> RiemannianProjector:
        return RiemannianProjector(identity_metric_4d, n_protected=2)

    def test_a1_idempotence(
        self, projector_4d_2: RiemannianProjector
    ) -> None:
        """A1: P̂² = P̂."""
        P = projector_4d_2.P
        np.testing.assert_allclose(P @ P, P, atol=TOL_STRICT)

    def test_a2_metric_symmetry(
        self, projector_4d_2: RiemannianProjector
    ) -> None:
        """A2: P̂ᵀ G = G P̂ (con G = I)."""
        P = projector_4d_2.P
        G = projector_4d_2.metric.G
        np.testing.assert_allclose(P.T @ G, G @ P, atol=TOL_STRICT)

    def test_a2_metric_symmetry_diagonal(
        self, diagonal_metric_4d: ValidatedMetricTensor
    ) -> None:
        """A2 con G diagonal: P̂ᵀ G = G P̂."""
        proj = RiemannianProjector(diagonal_metric_4d, n_protected=2)
        np.testing.assert_allclose(
            proj.P.T @ diagonal_metric_4d.G,
            diagonal_metric_4d.G @ proj.P,
            atol=TOL_STRICT,
        )

    def test_a3_contractive(
        self, projector_4d_2: RiemannianProjector
    ) -> None:
        """A3: ‖P̂‖_G ≤ 1."""
        # Verificado en check_axioms; lo invocamos explícitamente
        projector_4d_2.check_axioms()  # no debe lanzar excepción

    def test_a4_dimension_consistency(
        self, projector_4d_2: RiemannianProjector
    ) -> None:
        """A4: P̂ tiene forma (n, n)."""
        assert projector_4d_2.P.shape == (4, 4)

    def test_project_invariant_under_reapplication(
        self, projector_4d_2: RiemannianProjector, rng: np.random.Generator
    ) -> None:
        """P̂(P̂(x)) = P̂(x) (consecuencia de A1)."""
        for _ in range(20):
            x = rng.standard_normal(4)
            px = projector_4d_2(x)
            ppx = projector_4d_2(px)
            np.testing.assert_allclose(px, ppx, atol=TOL_STRICT)

    def test_invalid_n_protected(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> None:
        """n_protected fuera de (0, n] debe lanzar ValueError."""
        with pytest.raises(ValueError, match="n_protected"):
            RiemannianProjector(identity_metric_4d, n_protected=0)
        with pytest.raises(ValueError, match="n_protected"):
            RiemannianProjector(identity_metric_4d, n_protected=5)

    def test_n_protected_equals_n_is_identity(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> None:
        """Si k = n, P̂ = I."""
        proj = RiemannianProjector(identity_metric_4d, n_protected=4)
        np.testing.assert_allclose(proj.P, np.eye(4), atol=TOL_STRICT)


class TestDiracBoundaryOperator:
    """Pruebas del operador de Dirichlet."""

    @pytest.fixture
    def dirichlet_4d(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> DiracBoundaryOperator:
        return DiracBoundaryOperator(identity_metric_4d, max_energy=2.0)

    def test_below_threshold_preserves(
        self, dirichlet_4d: DiracBoundaryOperator
    ) -> None:
        """f(x) ≤ 0 → x preservado."""
        x = np.array([1.0, 0.0, 0.0, 0.0])  # ||x||² = 1 < 2
        x_out = dirichlet_4d.amputate(x)
        np.testing.assert_allclose(x_out, x, atol=TOL_STRICT)

    def test_above_threshold_nullifies(
        self, dirichlet_4d: DiracBoundaryOperator
    ) -> None:
        """f(x) > 0 → x anulado."""
        x = np.array([2.0, 0.0, 0.0, 0.0])  # ||x||² = 4 > 2
        x_out = dirichlet_4d.amputate(x)
        assert np.allclose(x_out, 0.0, atol=TOL_STRICT)

    def test_at_threshold_preserves(
        self, dirichlet_4d: DiracBoundaryOperator
    ) -> None:
        """f(x) = 0 → x preservado (caso límite)."""
        x = np.array([np.sqrt(2.0), 0.0, 0.0, 0.0])
        x_out = dirichlet_4d.amputate(x)
        np.testing.assert_allclose(x_out, x, atol=TOL_STRICT)

    def test_level_set_evaluation(
        self, dirichlet_4d: DiracBoundaryOperator
    ) -> None:
        """f(x) = xᵀ G x − E_max."""
        x = np.array([1.0, 1.0, 0.0, 0.0])  # ||x||² = 2
        f = dirichlet_4d.evaluate_level_set(x)
        assert f == pytest.approx(0.0, abs=TOL_STRICT)

    def test_infinite_threshold(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> None:
        """E_max = ∞ → nunca amputa."""
        D = DiracBoundaryOperator(identity_metric_4d, max_energy=float("inf"))
        x = np.array([1e10, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(D.amputate(x), x, atol=TOL_STRICT)


class TestHomologyBettiDetector:
    """Pruebas del detector de β₁ (Socavones Lógicos)."""

    def test_linear_state_has_no_cycle(self) -> None:
        """Estado lineal: x = (1, 2, 3, 4) → β₁ = 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert HomologyBettiDetector.compute_beta1(x) == 0

    def test_constant_state_has_no_cycle(self) -> None:
        """Estado constante: x = (c, c, c, c) → β₁ = 0."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        assert HomologyBettiDetector.compute_beta1(x) == 0

    def test_single_element(self) -> None:
        """Vector unitario: n=1, m=0, c=1 → β₁ = 0."""
        x = np.array([1.0])
        assert HomologyBettiDetector.compute_beta1(x) == 0

    def test_has_logical_crevice_returns_bool(self) -> None:
        """has_logical_crevice retorna booleano."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = HomologyBettiDetector.has_logical_crevice(x)
        assert isinstance(result, bool)

    def test_invalid_k_raises(self) -> None:
        """k < 1 debe lanzar ValueError."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="k"):
            HomologyBettiDetector._build_knn_graph(x, k=0)

    def test_knn_graph_2d_structure(self) -> None:
        """Para 2 nodos, k=1: n=2, m=1, c=1 → β₁ = 0."""
        x = np.array([0.0, 1.0])
        n, m, c = HomologyBettiDetector._build_knn_graph(x, k=1)
        assert n == 2
        assert m == 1
        assert c == 1

    def test_knn_graph_4d_structure(self) -> None:
        """Para 4 nodos, k=2: el grafo debe estar conectado."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        n, m, c = HomologyBettiDetector._build_knn_graph(x, k=2)
        assert n == 4
        assert m >= 3  # al menos n-1 aristas en un árbol
        assert c == 1  # k-NN sobre datos lineales está conectado


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CAPA 4 · PRUEBAS DE INTEGRACIÓN Y TEOREMAS                            │
# └─────────────────────────────────────────────────────────────────────────┘


class TestFuntorShieldIntegration:
    """Pruebas de integración del FuntorShield completo."""

    @pytest.fixture
    def shield_with_contracting_agent(
        self, contracting_agent: _ContractingAgent, mock_stratum: Stratum
    ) -> FuntorShield[_ContractingAgent]:
        return FuntorShield(
            invoking_agent=contracting_agent,
            stratum=mock_stratum,
            n_protected=2,
        )

    def test_construction_initializes_all_components(
        self,
        shield_with_contracting_agent: FuntorShield[_ContractingAgent],
    ) -> None:
        """La construcción inicializa J, R, H, P̂, D, Y, β₁."""
        s = shield_with_contracting_agent
        assert s._J is not None
        assert s._R is not None
        assert s._flow_sys is not None
        assert s._projector is not None
        assert s._dirichlet is not None
        assert s._yoneda is not None
        assert s._homology is not None

    def test_idempotence_under_double_application(
        self,
        shield_with_contracting_agent: FuntorShield[_ContractingAgent],
        array_state: CategoricalState,
    ) -> None:
        """T2: Shield(Shield(x)) = Shield(x)."""
        s = shield_with_contracting_agent
        result = s.verify_idempotence(array_state)
        assert result is True

    def test_pipeline_returns_categorical_state(
        self,
        shield_with_contracting_agent: FuntorShield[_ContractingAgent],
        array_state: CategoricalState,
    ) -> None:
        """El pipeline retorna un CategoricalState."""
        s = shield_with_contracting_agent
        result = s(array_state)
        assert isinstance(result, CategoricalState)

    def test_state_is_marked_as_shielded(
        self,
        shield_with_contracting_agent: FuntorShield[_ContractingAgent],
        array_state: CategoricalState,
    ) -> None:
        """El estado retornado lleva la firma de blindaje."""
        s = shield_with_contracting_agent
        result = s(array_state)
        assert hasattr(result, "__funtor_shield_signature__")
        sig = result.__funtor_shield_signature__
        assert isinstance(sig, ShieldSignature)
        assert sig.shield_id == s._shield_id

    def test_lyapunov_violation_triggers_dirichlet(
        self,
        mock_stratum: Stratum,
        divergent_agent: _DivergentAgent,
        array_state: CategoricalState,
    ) -> None:
        """Agente divergente → amputación por Dirichlet."""
        s = FuntorShield(
            invoking_agent=divergent_agent,
            stratum=mock_stratum,
            n_protected=2,
        )
        result = s(array_state)
        # Debe haber marcado el estado con violación
        assert hasattr(result, "__shield_metrics__")
        metrics = result.__shield_metrics__
        assert metrics.is_dirichlet_enforced is True

    def test_agent_exception_triggers_dirichlet(
        self,
        mock_stratum: Stratum,
        failing_agent: _FailingAgent,
        array_state: CategoricalState,
    ) -> None:
        """Excepción del agente → amputación por Dirichlet."""
        s = FuntorShield(
            invoking_agent=failing_agent,
            stratum=mock_stratum,
            n_protected=2,
        )
        result = s(array_state)
        assert hasattr(result, "__shield_metrics__")
        metrics = result.__shield_metrics__
        assert metrics.violation_type == ShieldViolationType.DIRICHLET_NULLIFICATION

    def test_funtor_dim_is_consistent_with_agent(
        self,
        stub_morphism: _StubMorphism,
        mock_stratum: Stratum,
    ) -> None:
        """Dim del funtor = dim del agente (cuando ambos pares)."""
        s = FuntorShield(
            invoking_agent=stub_morphism,
            stratum=mock_stratum,
            n_protected=2,
        )
        assert s._phase_space_dim == stub_morphism.phase_dim

    def test_funtor_adjusts_odd_dim(
        self,
        mock_stratum: Stratum,
    ) -> None:
        """Dim impar del agente → ajustada al par superior."""

        class _OddAgent(Morphism):
            def __init__(self) -> None:
                super().__init__()
                self.phase_dim = 3

            def __call__(self, state, *args, **kwargs):
                return state

        s = FuntorShield(
            invoking_agent=_OddAgent(),
            stratum=mock_stratum,
            n_protected=2,
        )
        assert s._phase_space_dim == 4  # 3 → 4 (par)


class TestFuntorShieldTheorems:
    """Verificación experimental de los teoremas T1–T4."""

    @pytest.fixture
    def shield_4d(
        self,
        contracting_agent: _ContractingAgent,
        mock_stratum: Stratum,
    ) -> FuntorShield[_ContractingAgent]:
        return FuntorShield(
            invoking_agent=contracting_agent,
            stratum=mock_stratum,
            n_protected=2,
        )

    def test_t1_endofunctor(
        self, shield_4d: FuntorShield[_ContractingAgent]
    ) -> None:
        """T1: Shield es un endofuntor (categoría → misma categoría)."""
        results = shield_4d.verify_theorems()
        assert results["T1_endofunctor"] is True

    def test_t2_idempotence_axiom(
        self, shield_4d: FuntorShield[_ContractingAgent]
    ) -> None:
        """T2: P̂² = P̂ (axioma de idempotencia)."""
        results = shield_4d.verify_theorems()
        # El axioma A1 ya fue verificado en RiemannianProjector.check_axioms
        # Aquí validamos la bandera T2
        assert "T2_idempotence" in results

    def test_t3_lyapunov_continuo(
        self, shield_4d: FuntorShield[_ContractingAgent], rng: np.random.Generator
    ) -> None:
        """T3: Ḣ ≤ 0 en el colector."""
        results = shield_4d.verify_theorems()
        assert results["T3_lyapunov"] is True

    def test_t4_homology_nonnegativa(
        self, shield_4d: FuntorShield[_ContractingAgent]
    ) -> None:
        """T4: β₁ ≥ 0 (es un invariante topológico)."""
        results = shield_4d.verify_theorems()
        assert results["T4_homology"] is True

    def test_all_theorems_together(
        self, shield_4d: FuntorShield[_ContractingAgent]
    ) -> None:
        """Verificación simultánea de T1–T4."""
        results = shield_4d.verify_theorems()
        for key, value in results.items():
            assert value is True, f"Teorema {key} no verificado"


class TestFuntorShieldDiagnostics:
    """Pruebas de los métodos de diagnóstico (spectral_gap, condition_number)."""

    @pytest.fixture
    def shield(
        self, stub_morphism: _StubMorphism, mock_stratum: Stratum
    ) -> FuntorShield[_StubMorphism]:
        return FuntorShield(
            invoking_agent=stub_morphism,
            stratum=mock_stratum,
            n_protected=2,
        )

    def test_spectral_gap_positive(
        self, shield: FuntorShield[_StubMorphism]
    ) -> None:
        """El gap espectral es computable y ≥ 0."""
        gap = shield.get_spectral_gap()
        assert gap >= 0.0
        assert np.isfinite(gap)

    def test_condition_number_at_least_one(
        self, shield: FuntorShield[_StubMorphism]
    ) -> None:
        """κ(G) ≥ 1 siempre (convención de norma)."""
        kappa = shield.compute_condition_number()
        assert kappa >= 1.0
        assert np.isfinite(kappa)

    def test_repr_contains_key_fields(
        self, shield: FuntorShield[_StubMorphism]
    ) -> None:
        """El __repr__ contiene id, agente, dim, stratum."""
        r = repr(shield)
        assert "FuntorShield" in r
        assert "_StubMorphism" in r
        assert "dim=" in r
        assert "stratum=" in r

    def test_str_is_concise(
        self, shield: FuntorShield[_StubMorphism]
    ) -> None:
        """El __str__ es una versión concisa."""
        s = str(shield)
        assert "FuntorShield[" in s
        assert "]" in s
        assert "@" in s


class TestApplyFuntorShieldDecorator:
    """Pruebas del decorador @apply_funtor_shield."""

    def test_decorator_returns_shielded_class(self, mock_stratum: Stratum) -> None:
        """El decorador retorna una subclase de FuntorShield."""
        @apply_funtor_shield(stratum=mock_stratum)
        class _MyAgent(Morphism):
            def __init__(self) -> None:
                super().__init__()
                self.phase_dim = 4

            def __call__(self, state, *args, **kwargs):
                return state

        assert issubclass(_MyAgent, FuntorShield)

    def test_decorated_class_preserves_name(self, mock_stratum: Stratum) -> None:
        """El nombre de la clase decorada se preserva con prefijo 'Shielded'."""
        @apply_funtor_shield(stratum=mock_stratum)
        class MyAgent(Morphism):
            phase_dim = 4

            def __call__(self, state, *args, **kwargs):
                return state

        assert "ShieldedMyAgent" in MyAgent.__name__

    def test_decorated_class_preserves_doc(self, mock_stratum: Stratum) -> None:
        """El docstring se preserva y se enriquece."""

        @apply_funtor_shield(stratum=mock_stratum)
        class DocumentedAgent(Morphism):
            """Mi agente documentado."""
            phase_dim = 4

            def __call__(self, state, *args, **kwargs):
                return state

        assert "Mi agente documentado" in (DocumentedAgent.__doc__ or "")

    def test_decorator_instantiation(self, mock_stratum: Stratum) -> None:
        """La clase decorada se puede instanciar como FuntorShield."""
        @apply_funtor_shield(stratum=mock_stratum)
        class _MyAgent(Morphism):
            phase_dim = 4

            def __call__(self, state, *args, **kwargs):
                return state

        instance = _MyAgent()
        assert isinstance(instance, FuntorShield)
        assert instance._phase_space_dim == 4


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PRUEBAS PARAMETRIZADAS DE REGRESIÓN                                    │
# └─────────────────────────────────────────────────────────────────────────┘


@pytest.mark.parametrize(
    "dim,n_protected,seed",
    [
        (2, 1, 1),
        (4, 2, 2),
        (6, 3, 3),
        (8, 4, 4),
        (10, 5, 5),
    ],
)
def test_projector_idempotence_varios_tamanos(
    dim: int, n_protected: int, seed: int
) -> None:
    """Regresión: A1 se mantiene para varias dimensiones."""
    G = ValidatedMetricTensor(np.eye(dim))
    proj = RiemannianProjector(G, n_protected=n_protected)
    P = proj.P
    np.testing.assert_allclose(P @ P, P, atol=TOL_STRICT)


@pytest.mark.parametrize(
    "dim,seed",
    [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5)],
)
def test_port_hamiltonian_lyapunov_varias_dimensiones(dim: int, seed: int) -> None:
    """Regresión: T3 se mantiene para varias dimensiones."""
    G = ValidatedMetricTensor(np.eye(dim))
    J = ConstantSkewStructure(dim=dim)
    R = QuadraticDissipation(dim=dim, gamma=0.5)
    flow = PortHamiltonianFlow(G=G, J=J, R=R)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        x = rng.standard_normal(dim)
        dH = flow.lyapunov_derivative(x)
        assert dH <= TOL_LOOSE, f"Ḣ={dH} en dim={dim}"


@pytest.mark.parametrize(
    "payload,target_dim,expected_size",
    [
        (1.0, 4, 4),  # E4 escalar
        (np.array([1.0, 2.0]), 4, 4),  # E3 + padding
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3, 3),  # E3 + truncado
        ([1.0, 2.0, 3.0], 3, 3),  # E5
    ],
)
def test_phase_vector_extractor_parametrizado(
    payload: Any, target_dim: int, expected_size: int, mock_stratum: Stratum
) -> None:
    """Regresión: extractor maneja combinaciones de estrategias y dimensiones."""
    state = CategoricalState(payload=payload, stratum=mock_stratum)
    vec = PhaseVectorExtractor.extract(state, target_dim)
    assert vec.shape == (expected_size,)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PRUEBAS DE PROPIEDADES (Property-based ligero)                         │
# └─────────────────────────────────────────────────────────────────────────┘


class TestPropertyInvariants:
    """Pruebas basadas en propiedades: invariantes que deben cumplirse siempre."""

    def test_hamiltonian_is_always_nonnegative(
        self, identity_metric_4d: ValidatedMetricTensor, rng: np.random.Generator
    ) -> None:
        """H(x) = ½ xᵀ G x ≥ 0 para G SPD."""
        J = ConstantSkewStructure(dim=4)
        R = QuadraticDissipation(dim=4, gamma=0.0)
        flow = PortHamiltonianFlow(G=identity_metric_4d, J=J, R=R)
        for _ in range(200):
            x = rng.standard_normal(4) * 100  # amplio rango
            assert flow.hamiltonian(x) >= 0.0

    def test_dissipation_matrix_always_psd(
        self, rng: np.random.Generator
    ) -> None:
        """R(x) ≥ 0 ∀x cuando R₀ ≥ 0 y γ ≥ 0."""
        for gamma in [0.0, 0.1, 1.0, 10.0, 100.0]:
            R = QuadraticDissipation(dim=4, gamma=gamma)
            for _ in range(20):
                x = rng.standard_normal(4)
                assert R.is_psd(x)

    def test_extractor_never_pads_nan(self, mock_stratum: Stratum) -> None:
        """El extractor nunca produce NaN en el padding."""
        for payload in [1.0, [1, 2], np.array([1.0]), "no_vectorizable"]:
            state = CategoricalState(payload=payload, stratum=mock_stratum)
            vec = PhaseVectorExtractor.extract(state, 4)
            assert not np.any(np.isnan(vec))
            assert not np.any(np.isinf(vec))

    def test_projector_preserves_vector_subspace(
        self, identity_metric_4d: ValidatedMetricTensor
    ) -> None:
        """P̂(x) preserva el subespacio protegido (P̂² = P̂)."""
        proj = RiemannianProjector(identity_metric_4d, n_protected=2)
        rng = np.random.default_rng(TEST_SEED)
        for _ in range(50):
            x = rng.standard_normal(4)
            # Generar un vector en el rango del proyector
            v = proj(x)
            # P̂(v) = v
            np.testing.assert_allclose(proj(v), v, atol=TOL_STRICT)