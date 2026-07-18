# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_tqft_projection_manifold.py                                    ║
║ Ruta   : tests/unit/omega/test_tqft_projection_manifold.py                   ║
║ Versión: 3.0.0-Strict-Functorial-Spectral-TQFT                               ║
║ Objetivo: Validación granular, rigurosa y espectral del endofuntor TQFT      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cobertura (fases anidadas + objetos categóricos):
  §0  Constantes, números cuánticos y umbrales espectrales
  §1  Jerarquía de excepciones topológicas
  §2  DTOs inmutables (TQFTBoundary, CobordismManifold, …)
  §3  MetricForgetfulFunctor (funtor de olvido U : Met → Top)
  §4  Fase 1 – Fibrado de Cobordismo (validación + 3-esqueleto)
  §5  Fase 2 – Motor de Invariantes (Chern–Simons + Turaev–Viro + SVD)
  §6  Fase 3 – Colapso Booleano y proyección al retículo
  §7  Orquestador TQFTProjectionManifold (composición funtorial completa)
  §8  Invariantes de integración / propiedades algebraicas
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# ── SUT ──────────────────────────────────────────────────────────────────────
from app.omega.tqft_projection_manifold import (
    TQFTConstants,
    TQFTProjectionError,
    CobordismDegeneracyError,
    TopologicalKnotVeto,
    TuraevViroCollapseError,
    TQFTBoundary,
    CobordismManifold,
    QuantumInvariants,
    TQFTVerdict,
    MetricForgetfulFunctor,
    Phase1_CobordismFibrator,
    Phase2_QuantumInvariantsEngine,
    Phase3_BooleanCollapseProjector,
    TQFTProjectionManifold,
)

# VerdictLevel puede ser el real o el stub del módulo
try:
    from app.wisdom.semantic_translator import VerdictLevel
except ImportError:
    from app.omega.tqft_projection_manifold import VerdictLevel  # type: ignore


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FIXTURES GLOBALES (espacios de Hilbert, fronteras, conexiones)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
DIM = 4
EPS = TQFTConstants.MACHINE_EPS


@pytest.fixture
def rng() -> np.random.Generator:
    """Generador determinista para reproducibilidad espectral."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def identity_connection() -> NDArray[np.float64]:
    """Conexión nula (plana) d×d – holonomía trivial."""
    return np.zeros((DIM, DIM), dtype=np.float64)


@pytest.fixture
def antisymmetric_connection(rng: np.random.Generator) -> NDArray[np.float64]:
    """Conexión anti-simétrica genérica (candidato a 𝔰𝔲(d)_ℝ)."""
    M = rng.standard_normal((DIM, DIM))
    return 0.5 * (M - M.T)


@pytest.fixture
def metric_contaminated_connection(rng: np.random.Generator) -> NDArray[np.float64]:
    """Conexión contaminada por parte simétrica (métrica) + traza no nula."""
    M = rng.standard_normal((DIM, DIM))
    # Parte simétrica dominante + escalado de traza
    S = 0.5 * (M + M.T) + 3.0 * np.eye(DIM)
    return S


@pytest.fixture
def compatible_boundaries() -> Tuple[TQFTBoundary, TQFTBoundary]:
    """Par de fronteras homológicamente compatibles (β₀ iguales, dim ℋ iguales)."""
    state_in = np.ones(DIM, dtype=np.float64) / math.sqrt(DIM)
    state_out = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    betti = (1, 0, 0)  # S²-like
    sigma_in = TQFTBoundary(
        state_vector=state_in,
        betti_numbers=betti,
        hilbert_dimension=DIM,
    )
    sigma_out = TQFTBoundary(
        state_vector=state_out,
        betti_numbers=betti,
        hilbert_dimension=DIM,
    )
    return sigma_in, sigma_out


@pytest.fixture
def incompatible_dim_boundaries() -> Tuple[TQFTBoundary, TQFTBoundary]:
    """Fronteras con dim ℋ distintas → CobordismDegeneracyError."""
    sigma_in = TQFTBoundary(
        state_vector=np.ones(DIM),
        betti_numbers=(1,),
        hilbert_dimension=DIM,
    )
    sigma_out = TQFTBoundary(
        state_vector=np.ones(DIM + 1),
        betti_numbers=(1,),
        hilbert_dimension=DIM + 1,
    )
    return sigma_in, sigma_out


@pytest.fixture
def incompatible_betti0_boundaries() -> Tuple[TQFTBoundary, TQFTBoundary]:
    """Fronteras con β₀ distintos → desgarro de componentes conexas."""
    sigma_in = TQFTBoundary(
        state_vector=np.ones(DIM),
        betti_numbers=(1, 0),
        hilbert_dimension=DIM,
    )
    sigma_out = TQFTBoundary(
        state_vector=np.ones(DIM),
        betti_numbers=(2, 0),  # β₀ ≠
        hilbert_dimension=DIM,
    )
    return sigma_in, sigma_out


@pytest.fixture
def degenerate_state_boundaries() -> Tuple[TQFTBoundary, TQFTBoundary]:
    """Frontera con vector de estado nulo → degeneración."""
    sigma_in = TQFTBoundary(
        state_vector=np.zeros(DIM),
        betti_numbers=(1,),
        hilbert_dimension=DIM,
    )
    sigma_out = TQFTBoundary(
        state_vector=np.ones(DIM),
        betti_numbers=(1,),
        hilbert_dimension=DIM,
    )
    return sigma_in, sigma_out


@pytest.fixture
def phase1() -> Phase1_CobordismFibrator:
    return Phase1_CobordismFibrator()


@pytest.fixture
def phase2() -> Phase2_QuantumInvariantsEngine:
    return Phase2_QuantumInvariantsEngine()


@pytest.fixture
def phase3() -> Phase3_BooleanCollapseProjector:
    return Phase3_BooleanCollapseProjector()


@pytest.fixture
def manifold_engine() -> TQFTProjectionManifold:
    return TQFTProjectionManifold()


@pytest.fixture
def valid_cobordism(
    phase1: Phase1_CobordismFibrator,
    compatible_boundaries: Tuple[TQFTBoundary, TQFTBoundary],
    antisymmetric_connection: NDArray[np.float64],
) -> CobordismManifold:
    """Cobordismo ya construido y validado (producto de Fase 1)."""
    sigma_in, sigma_out = compatible_boundaries
    return phase1.build_cobordism(sigma_in, sigma_out, antisymmetric_connection)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §0  CONSTANTES, NÚMEROS CUÁNTICOS Y UMBRALES ESPECTRALES                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestTQFTConstants:
    """Validación de parámetros del grupo cuántico U_q(sl₂) y umbrales."""

    def test_machine_eps_is_float64_eps(self) -> None:
        assert TQFTConstants.MACHINE_EPS == float(np.finfo(np.float64).eps)
        assert TQFTConstants.MACHINE_EPS > 0.0

    def test_chern_simons_level_positive_integer(self) -> None:
        assert isinstance(TQFTConstants.CHERN_SIMONS_K, int)
        assert TQFTConstants.CHERN_SIMONS_K >= 1

    def test_root_of_unity_order(self) -> None:
        """r = k + 2 es la convención clásica de SU(2)_k."""
        assert TQFTConstants.ROOT_OF_UNITY_R == TQFTConstants.CHERN_SIMONS_K + 2

    def test_q_deformation_is_primitive_root(self) -> None:
        q = TQFTConstants.Q_DEFORMATION
        r = TQFTConstants.ROOT_OF_UNITY_R
        # q^r = 1
        assert abs(q ** r - 1.0) < 1e-12
        # q^j ≠ 1 para 1 ≤ j < r (primalidad débil)
        for j in range(1, r):
            assert abs(q ** j - 1.0) > 1e-9

    def test_quantum_number_zero(self) -> None:
        assert abs(TQFTConstants.quantum_number(0)) < EPS

    def test_quantum_number_one(self) -> None:
        """[1]_q = 1 para cualquier q ≠ 0."""
        val = TQFTConstants.quantum_number(1)
        assert abs(val - 1.0) < 1e-12

    def test_quantum_number_symmetry(self) -> None:
        """[−n]_q = −[n]_q."""
        n = 3
        assert abs(
            TQFTConstants.quantum_number(n) + TQFTConstants.quantum_number(-n)
        ) < 1e-12

    def test_quantum_dimension_spin_zero(self) -> None:
        """dim_q(0) = [1]_q = 1."""
        assert abs(TQFTConstants.quantum_dimension(0.0) - 1.0) < 1e-12

    def test_quantum_dimension_spin_half_positive_modulus(self) -> None:
        d = TQFTConstants.quantum_dimension(0.5)
        assert abs(d) > 0.0

    def test_spectral_rel_tol_and_planck_eff_positive(self) -> None:
        assert TQFTConstants.SPECTRAL_REL_TOL > 0.0
        assert TQFTConstants.PLANCK_EFF > 0.0
        assert TQFTConstants.PLANCK_EFF >= TQFTConstants.MACHINE_EPS


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §1  JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestExceptionHierarchy:
    """La jerarquía debe respetar la herencia categórica de errores."""

    def test_root_is_topological_invariant_error(self) -> None:
        assert issubclass(TQFTProjectionError, Exception)

    def test_cobordism_degeneracy_inherits_root(self) -> None:
        assert issubclass(CobordismDegeneracyError, TQFTProjectionError)

    def test_knot_veto_inherits_root(self) -> None:
        assert issubclass(TopologicalKnotVeto, TQFTProjectionError)

    def test_turaev_viro_collapse_inherits_root(self) -> None:
        assert issubclass(TuraevViroCollapseError, TQFTProjectionError)

    def test_exceptions_are_raiseable_and_catchable(self) -> None:
        with pytest.raises(TQFTProjectionError):
            raise CobordismDegeneracyError("test")
        with pytest.raises(TQFTProjectionError):
            raise TopologicalKnotVeto("test")
        with pytest.raises(TQFTProjectionError):
            raise TuraevViroCollapseError("test")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §2  DTOs INMUTABLES (objetos de Cob y de la TQFT)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestDataTransferObjects:
    """Inmutabilidad, slots y validación de constructores."""

    def test_tqft_boundary_frozen(self, compatible_boundaries) -> None:
        sigma_in, _ = compatible_boundaries
        with pytest.raises(Exception):
            sigma_in.hilbert_dimension = 99  # type: ignore[misc]

    def test_tqft_boundary_fields(self, compatible_boundaries) -> None:
        sigma_in, sigma_out = compatible_boundaries
        assert sigma_in.hilbert_dimension == DIM
        assert sigma_in.betti_numbers == (1, 0, 0)
        assert sigma_in.state_vector.shape == (DIM,)
        assert sigma_out.hilbert_dimension == sigma_in.hilbert_dimension

    def test_tqft_boundary_rejects_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError, match="hilbert_dimension"):
            TQFTBoundary(
                state_vector=np.array([1.0]),
                betti_numbers=(1,),
                hilbert_dimension=0,
            )

    def test_cobordism_manifold_frozen(self, valid_cobordism) -> None:
        with pytest.raises(Exception):
            valid_cobordism.euler_characteristic = 999  # type: ignore[misc]

    def test_cobordism_manifold_structure(self, valid_cobordism) -> None:
        m = valid_cobordism
        assert m.connection_form_A.shape == (DIM, DIM)
        assert len(m.tetrahedral_tensors) >= 1
        for T in m.tetrahedral_tensors:
            assert T.dtype == np.complex128
            assert T.shape[0] == DIM
        assert isinstance(m.euler_characteristic, int)

    def test_quantum_invariants_defaults(self) -> None:
        qi = QuantumInvariants(
            chern_simons_action=0j,
            turaev_viro_state_sum=1.0 + 0j,
            is_knot_free=True,
        )
        assert qi.spectral_rank == 0
        assert qi.is_knot_free is True

    def test_tqft_verdict_structure(self) -> None:
        qi = QuantumInvariants(
            chern_simons_action=0j,
            turaev_viro_state_sum=1.0 + 0j,
            is_knot_free=True,
            spectral_rank=2,
        )
        v = TQFTVerdict(
            invariants=qi,
            verdict=VerdictLevel.VIABLE,
            topological_trace="Z(M)=1",
        )
        assert v.verdict == VerdictLevel.VIABLE
        assert "Z(M)" in v.topological_trace


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §3  METRIC FORGETFUL FUNCTOR  U : Met → Top                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestMetricForgetfulFunctor:
    """El funtor debe aniquilar traza y parte simétrica (métrica)."""

    def test_output_is_square_same_shape(
        self, metric_contaminated_connection: NDArray[np.float64]
    ) -> None:
        A = MetricForgetfulFunctor.apply_forgetful_map(
            None, metric_contaminated_connection
        )
        assert A.shape == metric_contaminated_connection.shape
        assert A.dtype == np.float64

    def test_result_is_antisymmetric(
        self, metric_contaminated_connection: NDArray[np.float64]
    ) -> None:
        A = MetricForgetfulFunctor.apply_forgetful_map(
            None, metric_contaminated_connection
        )
        # A + Aᵀ ≈ 0
        residual = A + A.T
        assert np.linalg.norm(residual, ord="fro") < 1e-12

    def test_trace_vanishes_after_forget(
        self, metric_contaminated_connection: NDArray[np.float64]
    ) -> None:
        A = MetricForgetfulFunctor.apply_forgetful_map(
            None, metric_contaminated_connection
        )
        # Anti-simétrica ⇒ traza nula
        assert abs(np.trace(A)) < 1e-12

    def test_already_antisymmetric_preserves_direction(
        self, antisymmetric_connection: NDArray[np.float64]
    ) -> None:
        A_raw = antisymmetric_connection
        A = MetricForgetfulFunctor.apply_forgetful_map(None, A_raw)
        # Dirección proyectiva: A ∝ A_raw (salvo normalización de Frobenius)
        if np.linalg.norm(A_raw, ord="fro") > EPS:
            # Cosine de Hilbert–Schmidt
            num = abs(np.trace(A.T @ A_raw))
            den = np.linalg.norm(A, ord="fro") * np.linalg.norm(A_raw, ord="fro")
            cosine = num / (den + EPS)
            assert cosine > 1.0 - 1e-9

    def test_zero_matrix_returns_zero(self) -> None:
        Z = np.zeros((DIM, DIM))
        A = MetricForgetfulFunctor.apply_forgetful_map(None, Z)
        assert np.allclose(A, 0.0)

    def test_non_square_raises(self) -> None:
        bad = np.ones((3, 4))
        with pytest.raises(CobordismDegeneracyError, match="cuadrada"):
            MetricForgetfulFunctor.apply_forgetful_map(None, bad)

    def test_frobenius_normalization(
        self, metric_contaminated_connection: NDArray[np.float64]
    ) -> None:
        A = MetricForgetfulFunctor.apply_forgetful_map(
            None, metric_contaminated_connection
        )
        fro = np.linalg.norm(A, ord="fro")
        # Si no es nula, debe estar normalizada a 1
        if fro > EPS:
            assert abs(fro - 1.0) < 1e-12


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §4  FASE 1 – FIBRADO DE COBORDISMO                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase1CobordismFibrator:
    """Validación de impedancia, Euler y generación del 3-esqueleto."""

    # ── 4.1 Validación de impedancia ─────────────────────────────────────────

    def test_compatible_boundaries_pass(
        self,
        phase1: Phase1_CobordismFibrator,
        compatible_boundaries,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        # No debe lanzar
        phase1._validate_impedance_matching(sigma_in, sigma_out)

    def test_incompatible_hilbert_dim_raises(
        self,
        phase1: Phase1_CobordismFibrator,
        incompatible_dim_boundaries,
    ) -> None:
        sigma_in, sigma_out = incompatible_dim_boundaries
        with pytest.raises(CobordismDegeneracyError, match="Desgarro dimensional"):
            phase1._validate_impedance_matching(sigma_in, sigma_out)

    def test_incompatible_betti0_raises(
        self,
        phase1: Phase1_CobordismFibrator,
        incompatible_betti0_boundaries,
    ) -> None:
        sigma_in, sigma_out = incompatible_betti0_boundaries
        with pytest.raises(CobordismDegeneracyError, match="β₀"):
            phase1._validate_impedance_matching(sigma_in, sigma_out)

    def test_degenerate_state_raises(
        self,
        phase1: Phase1_CobordismFibrator,
        degenerate_state_boundaries,
    ) -> None:
        sigma_in, sigma_out = degenerate_state_boundaries
        with pytest.raises(CobordismDegeneracyError, match="degenerado"):
            phase1._validate_impedance_matching(sigma_in, sigma_out)

    def test_empty_betti_raises(self, phase1: Phase1_CobordismFibrator) -> None:
        s_in = TQFTBoundary(np.ones(DIM), (), DIM)
        s_out = TQFTBoundary(np.ones(DIM), (), DIM)
        with pytest.raises(CobordismDegeneracyError, match="Betti"):
            phase1._validate_impedance_matching(s_in, s_out)

    # ── 4.2 Característica de Euler ──────────────────────────────────────────

    def test_euler_characteristic_sphere_like(
        self,
        phase1: Phase1_CobordismFibrator,
        compatible_boundaries,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        # β = (1,0,0) ⇒ χ(Σ) = 1; χ(M) ≈ (1+1)//2 = 1
        chi = phase1._estimate_euler_characteristic(sigma_in, sigma_out)
        assert chi == 1

    def test_euler_characteristic_torus_like(
        self, phase1: Phase1_CobordismFibrator
    ) -> None:
        # T²: β = (1, 2, 1) ⇒ χ = 0
        betti = (1, 2, 1)
        s = TQFTBoundary(np.ones(DIM), betti, DIM)
        chi = phase1._estimate_euler_characteristic(s, s)
        assert chi == 0

    # ── 4.3 Tensores tetraédricos ────────────────────────────────────────────

    def test_tetrahedral_tensors_count_and_dtype(
        self,
        phase1: Phase1_CobordismFibrator,
        antisymmetric_connection,
    ) -> None:
        tensors = phase1._generate_tetrahedral_tensors(DIM, antisymmetric_connection)
        assert len(tensors) == 3
        for T in tensors:
            assert T.shape == (DIM, DIM)
            assert T.dtype == np.complex128

    def test_trivial_channel_is_identity_scaled(
        self,
        phase1: Phase1_CobordismFibrator,
        identity_connection,
    ) -> None:
        tensors = phase1._generate_tetrahedral_tensors(DIM, identity_connection)
        T0 = tensors[0]
        # Canal trivial: dim_q(0)·Id = Id
        assert np.allclose(T0, np.eye(DIM, dtype=np.complex128), atol=1e-12)

    def test_holonomy_channel_unitary_for_antisymmetric(
        self,
        phase1: Phase1_CobordismFibrator,
        antisymmetric_connection,
    ) -> None:
        """exp(iA) con A anti-simétrica real ⇒ operador unitario (iA anti-hermitiana)."""
        tensors = phase1._generate_tetrahedral_tensors(DIM, antisymmetric_connection)
        T1 = tensors[1]
        # Normalizamos por |dim_q(1/2)| para recuperar la holonomía
        dim_q_half = TQFTConstants.quantum_dimension(0.5)
        U = T1 / dim_q_half
        # U† U ≈ I
        product = U.conj().T @ U
        assert np.allclose(product, np.eye(DIM), atol=1e-8)

    # ── 4.4 build_cobordism (método terminal Fase 1) ─────────────────────────

    def test_build_cobordism_success(
        self,
        phase1: Phase1_CobordismFibrator,
        compatible_boundaries,
        antisymmetric_connection,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        m = phase1.build_cobordism(sigma_in, sigma_out, antisymmetric_connection)
        assert isinstance(m, CobordismManifold)
        assert m.sigma_in is sigma_in
        assert m.sigma_out is sigma_out
        assert m.connection_form_A.shape == (DIM, DIM)
        # Conexión purificada anti-simétrica
        assert np.linalg.norm(m.connection_form_A + m.connection_form_A.T) < 1e-10
        assert len(m.tetrahedral_tensors) == 3

    def test_build_cobordism_rejects_shape_mismatch(
        self,
        phase1: Phase1_CobordismFibrator,
        compatible_boundaries,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        bad_conn = np.eye(DIM + 2)
        with pytest.raises(CobordismDegeneracyError):
            phase1.build_cobordism(sigma_in, sigma_out, bad_conn)

    def test_build_cobordism_applies_forgetful_functor(
        self,
        phase1: Phase1_CobordismFibrator,
        compatible_boundaries,
        metric_contaminated_connection,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        m = phase1.build_cobordism(
            sigma_in, sigma_out, metric_contaminated_connection
        )
        A = m.connection_form_A
        # Tras olvido: anti-simétrica y traza nula
        assert abs(np.trace(A)) < 1e-12
        assert np.linalg.norm(A + A.T, ord="fro") < 1e-12


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §5  FASE 2 – MOTOR DE INVARIANTES CUÁNTICOS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase2QuantumInvariantsEngine:
    """Chern–Simons, truncamiento espectral y suma de estados Turaev–Viro."""

    # ── 5.1 Acción de Chern–Simons ───────────────────────────────────────────

    def test_chern_simons_vanishes_on_zero_connection(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        A = np.zeros((DIM, DIM))
        s_cs = phase2._compute_chern_simons_action(A)
        assert abs(s_cs) < 1e-14

    def test_chern_simons_returns_complex(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        antisymmetric_connection,
    ) -> None:
        s_cs = phase2._compute_chern_simons_action(antisymmetric_connection)
        assert isinstance(s_cs, complex)

    def test_chern_simons_scales_with_level(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        antisymmetric_connection,
    ) -> None:
        """S_CS ∝ k; verificamos homogeneidad lineal en el nivel."""
        A = antisymmetric_connection
        s1 = phase2._compute_chern_simons_action(A)
        # Recomputamos con la fórmula explícita a k'=2k
        A_sq = A @ A
        dA = A_sq - A_sq.T
        A_cube = A @ A @ A
        omega = A @ dA + (2.0 / 3.0) * A_cube
        s_double = (2 * TQFTConstants.CHERN_SIMONS_K / (4.0 * np.pi)) * np.trace(omega)
        if abs(s1) > 1e-14:
            assert abs(complex(s_double) / s1 - 2.0) < 1e-9

    def test_chern_simons_gauge_covariance_under_orthogonal(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        rng: np.random.Generator,
    ) -> None:
        """
        Bajo conjugación ortogonal A ↦ Qᵀ A Q (Q ∈ O(d)), la traza de
        polinomios en A es invariante ⇒ S_CS invariante.
        """
        M = rng.standard_normal((DIM, DIM))
        A = 0.5 * (M - M.T)
        # Matriz ortogonal vía QR
        Q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
        A_conj = Q.T @ A @ Q
        s1 = phase2._compute_chern_simons_action(A)
        s2 = phase2._compute_chern_simons_action(A_conj)
        assert abs(s1 - s2) < 1e-10

    # ── 5.2 Truncamiento espectral (Eckart–Young) ────────────────────────────

    def test_spectral_truncate_full_rank_identity(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        I = np.eye(DIM, dtype=np.complex128)
        Z_trunc, rank = phase2._spectral_truncate(I, rel_tol=1e-12)
        assert rank == DIM
        assert np.allclose(Z_trunc, I, atol=1e-12)

    def test_spectral_truncate_low_rank(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        # Matriz de rango 1
        v = np.ones((DIM, 1), dtype=np.complex128)
        M = v @ v.T
        Z_trunc, rank = phase2._spectral_truncate(M, rel_tol=1e-10)
        assert rank == 1
        # La aproximación de rango 1 debe recuperar M
        assert np.allclose(Z_trunc, M, atol=1e-10)

    def test_spectral_truncate_empty(self, phase2: Phase2_QuantumInvariantsEngine) -> None:
        Z = np.array([], dtype=np.complex128).reshape(0, 0)
        Z_trunc, rank = phase2._spectral_truncate(Z)
        assert rank == 0

    def test_spectral_truncate_near_zero_keeps_dominant_mode(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        # Todos los valores singulares por debajo del umbral relativo extremo
        M = 1e-30 * np.eye(DIM, dtype=np.complex128)
        Z_trunc, rank = phase2._spectral_truncate(M, rel_tol=1e-2)
        # Política: conservar al menos el modo dominante
        assert rank >= 1

    # ── 5.3 Suma de estados Turaev–Viro ──────────────────────────────────────

    def test_turaev_viro_empty_tensors_raises(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        with pytest.raises(TuraevViroCollapseError, match="tetraédricos"):
            phase2._compute_turaev_viro_state_sum([])

    def test_turaev_viro_single_identity(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        I = [np.eye(DIM, dtype=np.complex128)]
        z, rank = phase2._compute_turaev_viro_state_sum(I)
        assert abs(z - float(DIM)) < 1e-10  # Tr(I_d) = d
        assert rank >= 0

    def test_turaev_viro_incompatible_dims_raises(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        T0 = np.eye(DIM, dtype=np.complex128)
        T1 = np.eye(DIM + 1, dtype=np.complex128)
        with pytest.raises(TuraevViroCollapseError, match="Incompatibilidad"):
            phase2._compute_turaev_viro_state_sum([T0, T1])

    def test_turaev_viro_collapse_below_planck(
        self, phase2: Phase2_QuantumInvariantsEngine
    ) -> None:
        # Tensores casi nulos ⇒ |Z| < PLANCK_EFF
        tiny = 1e-20 * np.eye(DIM, dtype=np.complex128)
        with pytest.raises(TuraevViroCollapseError, match="colapsada"):
            phase2._compute_turaev_viro_state_sum([tiny])

    def test_turaev_viro_from_valid_manifold(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        valid_cobordism: CobordismManifold,
    ) -> None:
        z, rank = phase2._compute_turaev_viro_state_sum(
            valid_cobordism.tetrahedral_tensors
        )
        assert isinstance(z, complex)
        assert abs(z) >= TQFTConstants.PLANCK_EFF
        assert rank >= 1

    # ── 5.4 compute_invariants (método terminal Fase 2) ──────────────────────

    def test_compute_invariants_structure(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        valid_cobordism: CobordismManifold,
    ) -> None:
        inv = phase2.compute_invariants(valid_cobordism)
        assert isinstance(inv, QuantumInvariants)
        assert isinstance(inv.chern_simons_action, complex)
        assert isinstance(inv.turaev_viro_state_sum, complex)
        assert isinstance(inv.is_knot_free, bool)
        assert inv.spectral_rank >= 0

    def test_compute_invariants_flat_connection_is_knot_free(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        compatible_boundaries,
        identity_connection,
    ) -> None:
        """Conexión nula ⇒ S_CS = 0 ⇒ knot-free."""
        sigma_in, sigma_out = compatible_boundaries
        m = phase2.build_cobordism(sigma_in, sigma_out, identity_connection)
        inv = phase2.compute_invariants(m)
        assert abs(inv.chern_simons_action) < 1e-12
        assert inv.is_knot_free is True

    def test_compute_invariants_nonzero_may_detect_knot(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        compatible_boundaries,
        rng: np.random.Generator,
    ) -> None:
        """
        Conexión anti-simétrica genérica de gran norma puede producir
        S_CS ≠ 0. Verificamos consistencia del flag con el umbral.
        """
        sigma_in, sigma_out = compatible_boundaries
        M = rng.standard_normal((DIM, DIM)) * 10.0
        A = 0.5 * (M - M.T)
        m = phase2.build_cobordism(sigma_in, sigma_out, A)
        inv = phase2.compute_invariants(m)
        threshold = TQFTConstants.SPECTRAL_REL_TOL * max(
            1.0, abs(TQFTConstants.CHERN_SIMONS_K)
        )
        expected_free = abs(inv.chern_simons_action) < threshold
        assert inv.is_knot_free is expected_free


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §6  FASE 3 – COLAPSO BOOLEANO Y PROYECCIÓN AL RETÍCULO                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestPhase3BooleanCollapseProjector:
    """Proyección al retículo {VIABLE, RECHAZAR} ≅ {⊤, ⊥}."""

    def _make_invariants(self, knot_free: bool, s_cs: complex = 0j) -> QuantumInvariants:
        return QuantumInvariants(
            chern_simons_action=s_cs,
            turaev_viro_state_sum=1.0 + 0j,
            is_knot_free=knot_free,
            spectral_rank=2,
        )

    def test_project_knot_free_to_viable(
        self, phase3: Phase3_BooleanCollapseProjector
    ) -> None:
        inv = self._make_invariants(knot_free=True)
        assert phase3._project_to_lattice(inv) == VerdictLevel.VIABLE

    def test_project_knotted_to_rechazar(
        self, phase3: Phase3_BooleanCollapseProjector
    ) -> None:
        inv = self._make_invariants(knot_free=False, s_cs=1.0 + 0j)
        assert phase3._project_to_lattice(inv) == VerdictLevel.RECHAZAR

    def test_collapse_verdict_viable(
        self, phase3: Phase3_BooleanCollapseProjector
    ) -> None:
        inv = self._make_invariants(knot_free=True)
        verdict = phase3.collapse_verdict(inv)
        assert isinstance(verdict, TQFTVerdict)
        assert verdict.verdict == VerdictLevel.VIABLE
        assert verdict.invariants is inv
        assert "Z(M)" in verdict.topological_trace
        assert "knot_free = True" in verdict.topological_trace

    def test_collapse_verdict_raises_knot_veto(
        self, phase3: Phase3_BooleanCollapseProjector
    ) -> None:
        inv = self._make_invariants(knot_free=False, s_cs=3.14 + 0j)
        with pytest.raises(TopologicalKnotVeto, match="Veto Topológico"):
            phase3.collapse_verdict(inv)

    def test_collapse_verdict_trace_contains_invariants(
        self, phase3: Phase3_BooleanCollapseProjector
    ) -> None:
        inv = self._make_invariants(knot_free=True, s_cs=0j)
        v = phase3.collapse_verdict(inv)
        assert "rank_eff = 2" in v.topological_trace
        assert "S_CS" in v.topological_trace


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §7  ORQUESTADOR SUPREMO – TQFTProjectionManifold                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestTQFTProjectionManifold:
    """Composición funtorial completa Phase3 ∘ Phase2 ∘ Phase1 ∘ U."""

    def test_project_intent_viable_flat_connection(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        identity_connection,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        result = manifold_engine.project_intent(
            sigma_in, sigma_out, identity_connection
        )
        assert isinstance(result, TQFTVerdict)
        assert result.verdict == VerdictLevel.VIABLE
        assert result.invariants.is_knot_free is True
        assert abs(result.invariants.chern_simons_action) < 1e-12

    def test_project_intent_viable_small_antisymmetric(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        rng: np.random.Generator,
    ) -> None:
        """Conexión anti-simétrica de norma pequeña ⇒ S_CS ≈ 0 ⇒ VIABLE."""
        sigma_in, sigma_out = compatible_boundaries
        M = rng.standard_normal((DIM, DIM)) * 1e-6
        A = 0.5 * (M - M.T)
        result = manifold_engine.project_intent(sigma_in, sigma_out, A)
        assert result.verdict == VerdictLevel.VIABLE

    def test_project_intent_rejects_incompatible_boundaries(
        self,
        manifold_engine: TQFTProjectionManifold,
        incompatible_dim_boundaries,
        identity_connection,
    ) -> None:
        sigma_in, sigma_out = incompatible_dim_boundaries
        with pytest.raises(CobordismDegeneracyError):
            manifold_engine.project_intent(sigma_in, sigma_out, identity_connection)

    def test_project_intent_rejects_betti0_mismatch(
        self,
        manifold_engine: TQFTProjectionManifold,
        incompatible_betti0_boundaries,
        identity_connection,
    ) -> None:
        sigma_in, sigma_out = incompatible_betti0_boundaries
        with pytest.raises(CobordismDegeneracyError, match="β₀"):
            manifold_engine.project_intent(sigma_in, sigma_out, identity_connection)

    def test_project_intent_rejects_degenerate_state(
        self,
        manifold_engine: TQFTProjectionManifold,
        degenerate_state_boundaries,
        identity_connection,
    ) -> None:
        sigma_in, sigma_out = degenerate_state_boundaries
        with pytest.raises(CobordismDegeneracyError, match="degenerado"):
            manifold_engine.project_intent(sigma_in, sigma_out, identity_connection)

    def test_project_intent_full_pipeline_trace(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        identity_connection,
    ) -> None:
        sigma_in, sigma_out = compatible_boundaries
        result = manifold_engine.project_intent(
            sigma_in, sigma_out, identity_connection
        )
        # Traza de auditoría forense completa
        trace = result.topological_trace
        assert "Z(M)" in trace
        assert "S_CS" in trace
        assert "rank_eff" in trace
        assert result.invariants.spectral_rank >= 0

    def test_inheritance_chain(self, manifold_engine: TQFTProjectionManifold) -> None:
        """El orquestador hereda las tres fases (composición por herencia)."""
        assert isinstance(manifold_engine, Phase3_BooleanCollapseProjector)
        assert isinstance(manifold_engine, Phase2_QuantumInvariantsEngine)
        assert isinstance(manifold_engine, Phase1_CobordismFibrator)

    def test_project_intent_metric_contamination_still_works(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        metric_contaminated_connection,
    ) -> None:
        """
        Incluso con conexión contaminada métricamente, el funtor de olvido
        debe permitir la proyección (éxito o veto topológico, pero no crash).
        """
        sigma_in, sigma_out = compatible_boundaries
        try:
            result = manifold_engine.project_intent(
                sigma_in, sigma_out, metric_contaminated_connection
            )
            assert result.verdict in (VerdictLevel.VIABLE, VerdictLevel.RECHAZAR)
        except TopologicalKnotVeto:
            # Veto legítimo si S_CS residual no nulo tras proyección
            pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §8  INVARIANTES DE INTEGRACIÓN / PROPIEDADES ALGEBRAICAS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TestAlgebraicAndIntegrationInvariants:
    """Propiedades que deben preservarse a través de la composición funtorial."""

    def test_end_to_end_idempotence_of_forgetful_on_purified(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        antisymmetric_connection,
    ) -> None:
        """U ∘ U = U sobre conexiones ya purificadas (idempotencia del funtor)."""
        A1 = MetricForgetfulFunctor.apply_forgetful_map(
            None, antisymmetric_connection
        )
        A2 = MetricForgetfulFunctor.apply_forgetful_map(None, A1)
        assert np.allclose(A1, A2, atol=1e-12)

    def test_cobordism_to_invariants_to_verdict_chain(
        self,
        phase1: Phase1_CobordismFibrator,
        phase2: Phase2_QuantumInvariantsEngine,
        phase3: Phase3_BooleanCollapseProjector,
        compatible_boundaries,
        identity_connection,
    ) -> None:
        """
        Cadena explícita Fase1 → Fase2 → Fase3 sin orquestador,
        verificando tipos en cada eslabón (composición monoidal estricta).
        """
        sigma_in, sigma_out = compatible_boundaries

        # Fase 1
        m = phase1.build_cobordism(sigma_in, sigma_out, identity_connection)
        assert isinstance(m, CobordismManifold)

        # Fase 2 (dominio = producto de Fase 1)
        inv = phase2.compute_invariants(m)
        assert isinstance(inv, QuantumInvariants)
        assert inv.is_knot_free is True

        # Fase 3 (dominio = producto de Fase 2)
        verdict = phase3.collapse_verdict(inv)
        assert isinstance(verdict, TQFTVerdict)
        assert verdict.verdict == VerdictLevel.VIABLE

    def test_phase2_accepts_only_cobordism_manifold_fields(
        self,
        phase2: Phase2_QuantumInvariantsEngine,
        valid_cobordism: CobordismManifold,
    ) -> None:
        """compute_invariants opera sobre todos los campos del cobordismo."""
        inv = phase2.compute_invariants(valid_cobordism)
        # Z(M) construido a partir de tetrahedral_tensors del cobordismo
        z_direct, _ = phase2._compute_turaev_viro_state_sum(
            valid_cobordism.tetrahedral_tensors
        )
        assert abs(inv.turaev_viro_state_sum - z_direct) < 1e-12
        # S_CS construido a partir de connection_form_A
        s_direct = phase2._compute_chern_simons_action(
            valid_cobordism.connection_form_A
        )
        assert abs(inv.chern_simons_action - s_direct) < 1e-12

    def test_verdict_invariants_roundtrip(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        identity_connection,
    ) -> None:
        """Los invariantes embebidos en el veredicto coinciden con el cómputo directo."""
        sigma_in, sigma_out = compatible_boundaries
        result = manifold_engine.project_intent(
            sigma_in, sigma_out, identity_connection
        )
        m = manifold_engine.build_cobordism(
            sigma_in, sigma_out, identity_connection
        )
        inv_direct = manifold_engine.compute_invariants(m)
        assert abs(
            result.invariants.chern_simons_action - inv_direct.chern_simons_action
        ) < 1e-12
        assert abs(
            result.invariants.turaev_viro_state_sum - inv_direct.turaev_viro_state_sum
        ) < 1e-12
        assert result.invariants.is_knot_free == inv_direct.is_knot_free

    def test_numerical_stability_repeated_projection(
        self,
        manifold_engine: TQFTProjectionManifold,
        compatible_boundaries,
        antisymmetric_connection,
    ) -> None:
        """Proyecciones repetidas producen invariantes numéricamente idénticos."""
        sigma_in, sigma_out = compatible_boundaries
        results = []
        for _ in range(5):
            # Cada llamada reconstruye desde la conexión cruda
            try:
                r = manifold_engine.project_intent(
                    sigma_in, sigma_out, antisymmetric_connection
                )
                results.append(r.invariants.turaev_viro_state_sum)
            except TopologicalKnotVeto as exc:
                # Si veta, todas deben vetar de forma consistente
                results.append("VETO")

        # Todas las corridas deben coincidir
        assert all(r == results[0] for r in results) or (
            all(isinstance(r, complex) for r in results)
            and all(abs(r - results[0]) < 1e-10 for r in results)
        )

    def test_hilbert_dimension_one_edge_case(
        self, manifold_engine: TQFTProjectionManifold
    ) -> None:
        """Caso borde: dim ℋ = 1 (espacio de Hilbert mínimo)."""
        s_in = TQFTBoundary(
            state_vector=np.array([1.0]),
            betti_numbers=(1,),
            hilbert_dimension=1,
        )
        s_out = TQFTBoundary(
            state_vector=np.array([1.0]),
            betti_numbers=(1,),
            hilbert_dimension=1,
        )
        A = np.zeros((1, 1))
        result = manifold_engine.project_intent(s_in, s_out, A)
        assert result.verdict == VerdictLevel.VIABLE
        assert abs(result.invariants.chern_simons_action) < 1e-12

    def test_module_all_exports_importable(self) -> None:
        """Todo lo listado en __all__ del módulo debe ser importable."""
        import app.omega.tqft_projection_manifold as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"Falta export: {name}"


# ── Entrypoint local (pytest) ────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])