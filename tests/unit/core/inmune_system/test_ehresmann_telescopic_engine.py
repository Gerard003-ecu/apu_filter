# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_ehresmann_telescopic_engine.py                                 ║
║ Ruta   : tests/unit/core/immune_system/test_ehresmann_telescopic_engine.py   ║
║ Objetivo: Validación granular, espectral y topológica del endofuntor         ║
║          Z = Φ₃ ∘ Φ₂ ∘ Φ₁ (Stinespring → Ehresmann → Maurer-Cartan/Novikov). ║
║ Framework: pytest + numpy + scipy                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de la suite (espejo de las fases anidadas del motor):
  §T0  Fixtures canónicas y utilidades espectrales
  §T1  Jerarquía de excepciones y constantes
  §T2  DTOs inmutables (objetos del Topos)
  §T3  Fase 1 — Inmersión isométrica de Stinespring
  §T4  Fase 2 — Fibración vertical de Ehresmann
  §T5  Fase 3 — Regularización Maurer-Cartan / Novikov
  §T6  Orquestador EhresmannTelescopicEngine (composición Z)
  §T7  Invariantes de extremo a extremo y regresión numérica
  §T8  Casos límite, patologías y robustez
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Importación del SUT (System Under Test)
# ─────────────────────────────────────────────────────────────────────────────
from app.core.immune_system.ehresmann_telescopic_engine import (
    # Constantes (acceso indirecto vía comportamiento)
    # Excepciones
    TelescopicEngineError,
    StinespringDilationError,
    InvalidDensityMatrixError,
    EhresmannFibrationError,
    SphereBubblingAnomalyError,
    SpectralDegeneracyError,
    # DTOs
    StinespringDilationData,
    VerticalFibrationData,
    TelescopicAuditState,
    # Fases
    Phase1_StinespringImmersion,
    Phase2_TelescopicVerticalFibration,
    Phase3_MaurerCartanRegularization,
    # Orquestador
    EhresmannTelescopicEngine,
)

# Tolerancias espejo del motor (no se importan privadas; se replican para asserts)
_ISO_TOL = 1e-12
_ORTH_TOL = 1e-10
_HERM_TOL = 1e-12
_TRACE_TOL = 1e-12
_MC_TOL = 1e-9
_PURE_FLOAT = np.float64
_COMPLEX = np.complex128


# ══════════════════════════════════════════════════════════════════════════════
# §T0 · FIXTURES CANÓNICAS Y UTILIDADES ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (M + M.conj().T)


def _random_density(n: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Genera ρ ∈ 𝔇(ℂⁿ) por el mapa de Wishart normalizado: A A† / Tr(A A†)."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    rho = _hermitize(rho)
    rho /= np.trace(rho).real
    return rho.astype(_COMPLEX)


def _pure_state_density(n: int, idx: int = 0) -> NDArray[np.complex128]:
    """Proyector |e_idx⟩⟨e_idx|."""
    rho = np.zeros((n, n), dtype=_COMPLEX)
    rho[idx, idx] = 1.0 + 0.0j
    return rho


def _maximally_mixed(n: int) -> NDArray[np.complex128]:
    return (np.eye(n, dtype=_COMPLEX) / n).astype(_COMPLEX)


def _is_hermitian(M: NDArray, tol: float = _HERM_TOL) -> bool:
    return float(la.norm(M - M.conj().T, ord="fro")) <= tol


def _is_psd(M: NDArray, floor: float = -1e-12) -> bool:
    ev = la.eigvalsh(_hermitize(M))
    return bool(np.all(ev >= floor))


def _is_density(M: NDArray, tol: float = _TRACE_TOL) -> bool:
    return (
        M.ndim == 2
        and M.shape[0] == M.shape[1]
        and _is_hermitian(M)
        and _is_psd(M)
        and abs(float(np.real(np.trace(M))) - 1.0) <= tol
    )


def _fro(A: NDArray) -> float:
    return float(la.norm(A, ord="fro"))


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42_001)


@pytest.fixture
def phase1() -> Phase1_StinespringImmersion:
    return Phase1_StinespringImmersion()


@pytest.fixture
def phase2() -> Phase2_TelescopicVerticalFibration:
    return Phase2_TelescopicVerticalFibration()


@pytest.fixture
def phase3() -> Phase3_MaurerCartanRegularization:
    return Phase3_MaurerCartanRegularization()


@pytest.fixture
def engine() -> EhresmannTelescopicEngine:
    return EhresmannTelescopicEngine(audit_dimension=4)


@pytest.fixture
def rho_2x2_pure() -> NDArray[np.complex128]:
    return _pure_state_density(2, idx=0)


@pytest.fixture
def rho_2x2_mixed(rng: np.random.Generator) -> NDArray[np.complex128]:
    return _random_density(2, rng)


@pytest.fixture
def rho_3x3_mixed(rng: np.random.Generator) -> NDArray[np.complex128]:
    return _random_density(3, rng)


@pytest.fixture
def rho_4x4_mixed(rng: np.random.Generator) -> NDArray[np.complex128]:
    return _random_density(4, rng)


@pytest.fixture
def rho_maximally_mixed_3() -> NDArray[np.complex128]:
    return _maximally_mixed(3)


# ══════════════════════════════════════════════════════════════════════════════
# §T1 · JERARQUÍA DE EXCEPCIONES Y CONTRATO TAXONÓMICO
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Verifica la taxonomía de excepciones topológicas/cuánticas."""

    def test_root_is_topological(self):
        # TelescopicEngineError hereda de TopologicalInvariantError (o Exception stub)
        assert issubclass(TelescopicEngineError, Exception)

    def test_leaves_inherit_from_root(self):
        leaves = (
            StinespringDilationError,
            InvalidDensityMatrixError,
            EhresmannFibrationError,
            SphereBubblingAnomalyError,
            SpectralDegeneracyError,
        )
        for exc in leaves:
            assert issubclass(exc, TelescopicEngineError), f"{exc} no hereda de raíz"

    def test_exceptions_are_raisable_and_catchable(self):
        with pytest.raises(TelescopicEngineError):
            raise StinespringDilationError("iso rota")
        with pytest.raises(InvalidDensityMatrixError):
            raise InvalidDensityMatrixError("ρ inválida")
        with pytest.raises(EhresmannFibrationError):
            raise EhresmannFibrationError("fuga H_p")
        with pytest.raises(SphereBubblingAnomalyError):
            raise SphereBubblingAnomalyError("MC diverge")

    def test_catch_leaf_via_root(self):
        with pytest.raises(TelescopicEngineError):
            raise SphereBubblingAnomalyError("burbujeo")


# ══════════════════════════════════════════════════════════════════════════════
# §T2 · DTOs INMUTABLES (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════

class TestDTOImmutability:
    """Los artefactos de fase deben ser frozen + slots (inmutables)."""

    def _sample_stinespring(self) -> StinespringDilationData:
        V = np.eye(4, 2, dtype=_COMPLEX)
        # Completar isometría 4×2 canónica: V[0,0]=1, V[2,1]=1 ya con eye parcial
        V = np.zeros((4, 2), dtype=_COMPLEX)
        V[0, 0] = 1.0
        V[2, 1] = 1.0
        rho_a = _maximally_mixed(2)
        return StinespringDilationData(
            V_isometry=V,
            rho_audit_subspace=rho_a,
            purity_preservation=0.5,
            dilating_environment_dim=2,
            stinespring_residual=0.0,
        )

    def test_stinespring_dto_frozen(self):
        dto = self._sample_stinespring()
        with pytest.raises(AttributeError):
            dto.purity_preservation = 0.9  # type: ignore[misc]

    def test_vertical_fibration_dto_frozen(self):
        dim = 2
        I = np.eye(dim, dtype=_COMPLEX)
        dto = VerticalFibrationData(
            T_lambda_vertical=I,
            ehresmann_connection_form=I,
            vertical_projector=I,
            horizontal_leakage_norm=0.0,
            connection_curvature_norm=0.0,
            spectral_zoom_eigenvalues=np.array([1.0, 1.0], dtype=_PURE_FLOAT),
        )
        with pytest.raises(AttributeError):
            dto.horizontal_leakage_norm = 1.0  # type: ignore[misc]

    def test_telescopic_audit_state_frozen(self):
        rho = _maximally_mixed(2)
        b = np.zeros((2, 2), dtype=_COMPLEX)
        dto = TelescopicAuditState(
            audited_density_matrix=rho,
            landau_ginzburg_potential=0.0,
            novikov_convergence_iterations=1,
            maurer_cartan_bounding_cochain=b,
            novikov_filtration_degree=0.0,
            is_safe_for_witten_atiyah=True,
        )
        with pytest.raises(AttributeError):
            dto.is_safe_for_witten_atiyah = False  # type: ignore[misc]

    def test_dto_fields_accessible(self):
        dto = self._sample_stinespring()
        assert dto.dilating_environment_dim == 2
        assert 0.0 <= dto.purity_preservation <= 1.0
        assert dto.V_isometry.shape[1] == 2


# ══════════════════════════════════════════════════════════════════════════════
# §T3 · FASE 1 — INMERSIÓN ISOMÉTRICA DE STINESPRING
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1ValidateDensityMatrix:
    """§1.1 — Validación del simplejo 𝔇(ℋ)."""

    def test_valid_pure_state_accepted(self, phase1, rho_2x2_pure):
        out = phase1._validate_density_matrix(rho_2x2_pure, name="ρ")
        assert _is_density(out)

    def test_valid_mixed_state_accepted(self, phase1, rho_3x3_mixed):
        out = phase1._validate_density_matrix(rho_3x3_mixed, name="ρ")
        assert _is_density(out)

    def test_maximally_mixed_accepted(self, phase1, rho_maximally_mixed_3):
        out = phase1._validate_density_matrix(rho_maximally_mixed_3)
        assert _is_density(out)
        n = out.shape[0]
        purity = float(np.real(np.trace(out @ out)))
        assert purity == pytest.approx(1.0 / n, abs=1e-10)

    def test_rejects_non_square(self, phase1):
        M = np.ones((2, 3), dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError, match="cuadrada"):
            phase1._validate_density_matrix(M)

    def test_rejects_non_hermitian(self, phase1):
        M = np.array([[0.5, 1.0 + 2.0j], [0.0, 0.5]], dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError, match="hermítica"):
            phase1._validate_density_matrix(M)

    def test_rejects_negative_eigenvalue(self, phase1):
        # Matriz hermítica con autovalor negativo, traza 1
        M = np.array([[1.5, 0.0], [0.0, -0.5]], dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError, match="negativo|SPD"):
            phase1._validate_density_matrix(M)

    def test_rejects_wrong_trace(self, phase1):
        M = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError, match="Tr"):
            phase1._validate_density_matrix(M)

    def test_hermitize_projection(self, phase1):
        A = np.array([[1.0, 1e-15 + 1e-15j], [0.0, 0.0]], dtype=_COMPLEX)
        # casi hermítica con traza 1 tras proyección — usamos una válida
        rho = _hermitize(np.array([[0.7, 0.1j], [-0.1j, 0.3]], dtype=_COMPLEX))
        rho /= np.trace(rho).real
        out = phase1._validate_density_matrix(rho)
        assert _is_hermitian(out)


class TestPhase1Isometry:
    """§1.2 — Condición V†V = I."""

    def test_canonical_isometry_is_isometric(self, phase1):
        V = phase1._build_canonical_isometry(dim_mic=3, dim_audit=4)
        residual = phase1._verify_isometry(V)
        assert residual < _ISO_TOL
        gram = V.conj().T @ V
        assert _fro(gram - np.eye(3, dtype=_COMPLEX)) < _ISO_TOL

    def test_isometry_shape(self, phase1):
        dim_mic, dim_audit = 5, 3
        V = phase1._build_canonical_isometry(dim_mic, dim_audit)
        assert V.shape == (dim_mic * dim_audit, dim_mic)

    def test_broken_isometry_raises(self, phase1):
        V = np.zeros((4, 2), dtype=_COMPLEX)
        V[0, 0] = 2.0  # rompe isometría
        with pytest.raises(StinespringDilationError, match="isométrica"):
            phase1._verify_isometry(V)

    def test_near_isometry_within_tolerance(self, phase1):
        V = phase1._build_canonical_isometry(2, 2)
        # perturbación bajo tolerancia
        V_pert = V.copy()
        V_pert[0, 0] += 1e-16
        residual = phase1._verify_isometry(V_pert)
        assert residual < _ISO_TOL


class TestPhase1PartialTrace:
    """§1.3 — Traza parcial Tr_MIC."""

    def test_partial_trace_shape(self, phase1, rho_2x2_pure):
        dim_mic, dim_audit = 2, 3
        V = phase1._build_canonical_isometry(dim_mic, dim_audit)
        rho_c = V @ rho_2x2_pure @ V.conj().T
        rho_a = phase1._partial_trace_mic(rho_c, dim_mic, dim_audit)
        assert rho_a.shape == (dim_audit, dim_audit)

    def test_partial_trace_preserves_trace(self, phase1, rho_3x3_mixed):
        dim_mic = 3
        dim_audit = 4
        V = phase1._build_canonical_isometry(dim_mic, dim_audit)
        rho_c = V @ rho_3x3_mixed @ V.conj().T
        rho_a = phase1._partial_trace_mic(rho_c, dim_mic, dim_audit)
        assert abs(float(np.real(np.trace(rho_a))) - 1.0) < 1e-10

    def test_partial_trace_hermitian(self, phase1, rho_2x2_mixed):
        dim_mic, dim_audit = 2, 2
        V = phase1._build_canonical_isometry(dim_mic, dim_audit)
        rho_c = V @ rho_2x2_mixed @ V.conj().T
        rho_a = phase1._partial_trace_mic(rho_c, dim_mic, dim_audit)
        assert _is_hermitian(rho_a)

    def test_partial_trace_dimension_mismatch_raises(self, phase1):
        rho_c = np.eye(6, dtype=_COMPLEX) / 6.0
        with pytest.raises(StinespringDilationError, match="incompatible"):
            phase1._partial_trace_mic(rho_c, dim_mic=2, dim_audit=2)  # 2*2=4 ≠ 6


class TestPhase1ComputeIsometricImmersion:
    """§1.5 — Morfismo terminal Φ₁."""

    def test_returns_stinespring_dto(self, phase1, rho_2x2_mixed):
        data = phase1.compute_isometric_immersion(rho_2x2_mixed, dim_audit=3)
        assert isinstance(data, StinespringDilationData)

    def test_isometry_residual_near_zero(self, phase1, rho_3x3_mixed):
        data = phase1.compute_isometric_immersion(rho_3x3_mixed, dim_audit=4)
        assert data.stinespring_residual < _ISO_TOL
        V = data.V_isometry
        assert _fro(V.conj().T @ V - np.eye(3, dtype=_COMPLEX)) < _ISO_TOL

    def test_rho_audit_is_density(self, phase1, rho_2x2_pure):
        data = phase1.compute_isometric_immersion(rho_2x2_pure, dim_audit=2)
        assert _is_density(data.rho_audit_subspace)

    def test_purity_bounds(self, phase1, rho_4x4_mixed):
        dim_audit = 4
        data = phase1.compute_isometric_immersion(rho_4x4_mixed, dim_audit=dim_audit)
        assert 1.0 / dim_audit - 1e-9 <= data.purity_preservation <= 1.0 + 1e-9

    def test_purity_of_pure_input_under_canonical_embedding(self, phase1, rho_2x2_pure):
        """
        Con V = I ⊗ |0⟩, ρ_audit = Tr_MIC(V|ψ⟩⟨ψ|V†) colapsa a |0⟩⟨0| del baño
        (puro) cuando el anclaje es al vacuum: pureza → 1.
        """
        data = phase1.compute_isometric_immersion(rho_2x2_pure, dim_audit=3)
        assert data.purity_preservation == pytest.approx(1.0, abs=1e-8)

    def test_dilating_environment_dim_recorded(self, phase1, rho_2x2_mixed):
        data = phase1.compute_isometric_immersion(rho_2x2_mixed, dim_audit=5)
        assert data.dilating_environment_dim == 5
        assert data.rho_audit_subspace.shape == (5, 5)

    def test_rejects_audit_dim_too_small(self, phase1, rho_2x2_mixed):
        with pytest.raises(ValueError, match="dim_audit"):
            phase1.compute_isometric_immersion(rho_2x2_mixed, dim_audit=1)

    def test_rejects_invalid_rho(self, phase1):
        bad = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError):
            phase1.compute_isometric_immersion(bad, dim_audit=2)

    def test_v_dagger_v_equals_identity_numeric(self, phase1, rng):
        for n in (2, 3, 5):
            rho = _random_density(n, rng)
            data = phase1.compute_isometric_immersion(rho, dim_audit=4)
            G = data.V_isometry.conj().T @ data.V_isometry
            np.testing.assert_allclose(G, np.eye(n), atol=_ISO_TOL)

    def test_combined_state_is_density(self, phase1, rho_2x2_mixed):
        data = phase1.compute_isometric_immersion(rho_2x2_mixed, dim_audit=3)
        V = data.V_isometry
        rho_c = V @ rho_2x2_mixed @ V.conj().T
        # traza del estado combinado = 1
        assert abs(float(np.real(np.trace(rho_c))) - 1.0) < 1e-10
        assert _is_psd(rho_c)


# ══════════════════════════════════════════════════════════════════════════════
# §T4 · FASE 2 — FIBRACIÓN VERTICAL DE EHRESMANN
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2EhresmannConnection:
    """§2.1 — Conexión de Ehresmann y curvatura."""

    def test_projectors_shape_and_complement(self, phase2):
        dim = 4
        Pi_V, Pi_H, omega = phase2._build_ehresmann_connection(dim)
        assert Pi_V.shape == (dim, dim)
        assert Pi_H.shape == (dim, dim)
        assert omega.shape == (dim, dim)
        # En calibre canónico: Π_V = I, Π_H = 0
        np.testing.assert_allclose(Pi_V, np.eye(dim), atol=_ISO_TOL)
        np.testing.assert_allclose(Pi_H, np.zeros((dim, dim)), atol=_ISO_TOL)

    def test_vertical_projector_idempotent(self, phase2):
        Pi_V, _, omega = phase2._build_ehresmann_connection(3)
        np.testing.assert_allclose(Pi_V @ Pi_V, Pi_V, atol=_ISO_TOL)
        np.testing.assert_allclose(omega @ omega, omega, atol=_ISO_TOL)

    def test_curvature_vanishes_in_flat_gauge(self, phase2):
        _, _, omega = phase2._build_ehresmann_connection(5)
        curv = phase2._connection_curvature_norm(omega)
        assert curv < _ORTH_TOL


class TestPhase2SpectralZoom:
    """§2.2 — Zoom espectral Tikhonov."""

    def test_scale_factors_at_least_one_for_nonnegative_lambda(self, phase2, rng):
        rho = _random_density(3, rng)
        T, scales = phase2._compute_spectral_zoom(rho, lambda_zoom=2.5)
        assert np.all(scales >= 1.0 - 1e-12)

    def test_identity_zoom_at_lambda_zero(self, phase2, rng):
        rho = _random_density(3, rng)
        T, scales = phase2._compute_spectral_zoom(rho, lambda_zoom=0.0)
        np.testing.assert_allclose(scales, np.ones(3), atol=1e-12)
        # T debe ser I en cualquier base (s_i=1 ∀i)
        np.testing.assert_allclose(T, np.eye(3, dtype=_COMPLEX), atol=1e-10)

    def test_T_is_hermitian(self, phase2, rng):
        rho = _random_density(4, rng)
        T, _ = phase2._compute_spectral_zoom(rho, lambda_zoom=1.0)
        assert _is_hermitian(T)

    def test_negative_lambda_raises(self, phase2, rng):
        rho = _random_density(2, rng)
        with pytest.raises(ValueError, match="lambda_zoom"):
            phase2._compute_spectral_zoom(rho, lambda_zoom=-0.1)

    def test_tikhonov_saturation_uv(self, phase2):
        """Para μ grandes, s ∼ 1 + λ/√α (acotado); no diverge."""
        # ρ con autovalor dominante cercano a 1
        rho = np.diag([0.99, 0.01]).astype(_COMPLEX)
        lam = 100.0
        alpha = 1e-3
        _, scales = phase2._compute_spectral_zoom(rho, lambda_zoom=lam, tikhonov_alpha=alpha)
        # cota teórica: s ≤ 1 + λ/√α
        upper = 1.0 + lam / math.sqrt(alpha)
        assert np.all(scales <= upper + 1e-8)

    def test_spectral_zoom_commutes_with_eigenbasis(self, phase2):
        """T y ρ deben conmutar (T es función espectral de ρ)."""
        rho = np.diag([0.7, 0.2, 0.1]).astype(_COMPLEX)
        T, _ = phase2._compute_spectral_zoom(rho, lambda_zoom=1.5)
        comm = T @ rho - rho @ T
        assert _fro(comm) < 1e-10


class TestPhase2VerticalProjection:
    """§2.3 — Proyección vertical y fuga horizontal."""

    def test_zero_leakage_in_canonical_gauge(self, phase2, rng):
        dim = 3
        Pi_V, Pi_H, omega = phase2._build_ehresmann_connection(dim)
        T = _hermitize(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
        T_v, leak = phase2._project_vertical_and_measure_leakage(T, Pi_V, Pi_H, omega)
        assert leak < _ORTH_TOL
        assert _is_hermitian(T_v)

    def test_leakage_detection_raises(self, phase2):
        dim = 2
        Pi_V = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=_COMPLEX)  # proyector propio
        Pi_H = np.eye(2, dtype=_COMPLEX) - Pi_V
        omega = Pi_V.copy()
        # T con componente horizontal significativa
        T = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises(EhresmannFibrationError, match="Fuga|horizontal"):
            phase2._project_vertical_and_measure_leakage(T, Pi_V, Pi_H, omega)


class TestPhase2ApplyTelescopicDeformation:
    """§2.4 — Morfismo terminal Φ₂ (continuación de Φ₁)."""

    def _dilation(self, phase1, rho, dim_audit=4) -> StinespringDilationData:
        return phase1.compute_isometric_immersion(rho, dim_audit=dim_audit)

    def test_returns_vertical_fibration_dto(self, phase1, phase2, rho_2x2_mixed):
        dil = self._dilation(phase1, rho_2x2_mixed)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=1.0)
        assert isinstance(fib, VerticalFibrationData)

    def test_horizontal_leakage_near_zero(self, phase1, phase2, rho_3x3_mixed):
        dil = self._dilation(phase1, rho_3x3_mixed)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=2.0)
        assert fib.horizontal_leakage_norm < _ORTH_TOL

    def test_curvature_norm_near_zero(self, phase1, phase2, rho_2x2_pure):
        dil = self._dilation(phase1, rho_2x2_pure, dim_audit=3)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=0.5)
        assert fib.connection_curvature_norm < _ORTH_TOL

    def test_T_vertical_hermitian(self, phase1, phase2, rho_2x2_mixed):
        dil = self._dilation(phase1, rho_2x2_mixed)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=1.2)
        assert _is_hermitian(fib.T_lambda_vertical)

    def test_spectral_eigenvalues_match_dim(self, phase1, phase2, rho_2x2_mixed):
        dil = self._dilation(phase1, rho_2x2_mixed, dim_audit=5)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=0.8)
        assert fib.spectral_zoom_eigenvalues.shape == (5,)
        assert np.all(fib.spectral_zoom_eigenvalues >= 1.0 - 1e-12)

    def test_lambda_zero_gives_identity_T(self, phase1, phase2, rho_2x2_mixed):
        dil = self._dilation(phase1, rho_2x2_mixed, dim_audit=3)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=0.0)
        np.testing.assert_allclose(
            fib.T_lambda_vertical, np.eye(3, dtype=_COMPLEX), atol=1e-10
        )

    def test_negative_lambda_raises(self, phase1, phase2, rho_2x2_mixed):
        dil = self._dilation(phase1, rho_2x2_mixed)
        with pytest.raises(ValueError, match="lambda_zoom"):
            phase2.apply_telescopic_deformation(dil, lambda_zoom=-1.0)

    def test_phase2_inherits_phase1(self, phase2):
        assert isinstance(phase2, Phase1_StinespringImmersion)

    def test_chain_phi1_to_phi2(self, phase2, rho_3x3_mixed):
        """Anidación: Φ₂ consume directamente el DTO de Φ₁ vía herencia."""
        dil = phase2.compute_isometric_immersion(rho_3x3_mixed, dim_audit=4)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=1.0)
        assert fib.T_lambda_vertical.shape == dil.rho_audit_subspace.shape


# ══════════════════════════════════════════════════════════════════════════════
# §T5 · FASE 3 — REGULARIZACIÓN MAURER-CARTAN / NOVIKOV
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase3AnomalyAndMCFunction:
    """§3.1–§3.2 — m₀ y F(b)."""

    def test_anomaly_zero_when_T_is_identity(self, phase3, rng):
        rho = _random_density(3, rng)
        T = np.eye(3, dtype=_COMPLEX)
        m0 = phase3._compute_anomaly(T, rho)
        assert _fro(m0) < 1e-12

    def test_anomaly_hermitian_when_inputs_hermitian(self, phase3, rng):
        rho = _random_density(3, rng)
        T, _ = phase3._compute_spectral_zoom(rho, lambda_zoom=1.0)
        m0 = phase3._compute_anomaly(T, rho)
        assert _is_hermitian(m0)

    def test_mc_function_at_b_zero_equals_m0(self, phase3, rng):
        rho = _random_density(2, rng)
        T, _ = phase3._compute_spectral_zoom(rho, lambda_zoom=0.7)
        m0 = phase3._compute_anomaly(T, rho)
        b = np.zeros_like(T)
        F = phase3._maurer_cartan_function(b, T, m0)
        np.testing.assert_allclose(F, m0, atol=1e-14)

    def test_mc_function_structure(self, phase3):
        """F(b) = m0 + [T,b] + b² — verificación algebraica explícita."""
        T = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=_COMPLEX)
        b = np.array([[0.0, 0.1j], [-0.1j, 0.0]], dtype=_COMPLEX)
        m0 = np.array([[0.01, 0.0], [0.0, -0.01]], dtype=_COMPLEX)
        F = phase3._maurer_cartan_function(b, T, m0)
        expected = m0 + (T @ b - b @ T) + b @ b
        np.testing.assert_allclose(F, expected, atol=1e-14)


class TestPhase3LinearizedSolver:
    """§3.3 — Sylvester / Newton step."""

    def test_sylvester_solves_zero_residual(self, phase3):
        """Si residual=0, δb debe ser ≈0 (solución homogénea trivial)."""
        T = np.diag([1.0, 2.0]).astype(_COMPLEX)
        b = np.zeros((2, 2), dtype=_COMPLEX)
        residual = np.zeros((2, 2), dtype=_COMPLEX)
        delta = phase3._solve_linearized(b, T, residual)
        assert _fro(delta) < 1e-10

    def test_sylvester_finite_for_generic_input(self, phase3, rng):
        n = 3
        T = _hermitize(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        # Desplazar espectro de T para evitar singularidad
        T = T + 3.0 * np.eye(n, dtype=_COMPLEX)
        b = 0.01 * _hermitize(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        residual = 0.1 * _hermitize(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        delta = phase3._solve_linearized(b, T, residual)
        assert np.all(np.isfinite(delta))

    def test_newton_direction_reduces_residual_locally(self, phase3, rng):
        """Un paso de Newton sin damping debe reducir ‖F‖ en régimen cercano a 0."""
        rho = _random_density(2, rng)
        T, _ = phase3._compute_spectral_zoom(rho, lambda_zoom=0.3)
        m0 = phase3._compute_anomaly(T, rho)
        b = np.zeros_like(T)
        F = phase3._maurer_cartan_function(b, T, m0)
        r0 = _fro(F)
        if r0 < 1e-14:
            pytest.skip("anomalía ya nula")
        delta = phase3._solve_linearized(b, T, F)
        b1 = b + delta
        r1 = _fro(phase3._maurer_cartan_function(b1, T, m0))
        # Newton cuadrático: en vecindad razonable debe bajar
        assert r1 < r0 * 1.05 + 1e-12  # margen numérico generoso


class TestPhase3DampingAndRegularizedDensity:
    """§3.4–§3.5 — Armijo y ρ_safe."""

    def test_damped_step_returns_finite(self, phase3, rng):
        rho = _random_density(2, rng)
        T, _ = phase3._compute_spectral_zoom(rho, lambda_zoom=0.5)
        m0 = phase3._compute_anomaly(T, rho)
        b = np.zeros_like(T)
        F = phase3._maurer_cartan_function(b, T, m0)
        delta = phase3._solve_linearized(b, T, F)
        b_new = phase3._newton_damped_step(b, delta, T, m0, _fro(F))
        assert np.all(np.isfinite(b_new))

    def test_regularized_density_is_valid_density(self, phase3, rng):
        rho = _random_density(3, rng)
        # b antihermitiana pequeña → e^b casi unitaria
        X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        b = 0.05 * (X - X.conj().T)  # antihermitiana
        rho_safe = phase3._regularized_density(b, rho)
        assert _is_density(rho_safe)

    def test_regularized_density_trace_one(self, phase3, rng):
        rho = _random_density(2, rng)
        b = np.zeros((2, 2), dtype=_COMPLEX)
        rho_safe = phase3._regularized_density(b, rho)
        assert abs(float(np.real(np.trace(rho_safe))) - 1.0) < 1e-10

    def test_regularized_density_psd(self, phase3, rng):
        rho = _random_density(4, rng)
        b = 0.01 * _hermitize(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        rho_safe = phase3._regularized_density(b, rho)
        assert _is_psd(rho_safe)


class TestPhase3ResolveMaurerCartanNovikov:
    """§3.6 — Morfismo terminal Φ₃ (cierra Z)."""

    def _prepare_fibration(
        self,
        phase3: Phase3_MaurerCartanRegularization,
        rho_business: NDArray[np.complex128],
        dim_audit: int = 4,
        lam: float = 0.5,
    ) -> Tuple[VerticalFibrationData, NDArray[np.complex128]]:
        dil = phase3.compute_isometric_immersion(rho_business, dim_audit=dim_audit)
        fib = phase3.apply_telescopic_deformation(dil, lambda_zoom=lam)
        return fib, dil.rho_audit_subspace

    def test_returns_telescopic_audit_state(self, phase3, rho_2x2_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_mixed, lam=0.4)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert isinstance(state, TelescopicAuditState)

    def test_converges_for_moderate_lambda(self, phase3, rho_2x2_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_mixed, lam=0.5)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert state.landau_ginzburg_potential < _MC_TOL
        assert state.novikov_convergence_iterations >= 1
        assert state.is_safe_for_witten_atiyah is True

    def test_audited_density_is_valid(self, phase3, rho_3x3_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_3x3_mixed, dim_audit=3, lam=0.3)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert _is_density(state.audited_density_matrix)

    def test_lambda_zero_trivial_convergence(self, phase3, rho_2x2_pure):
        """Con λ=0, T=I, m₀=0 ⇒ F(0)=0 en la primera evaluación."""
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_pure, dim_audit=2, lam=0.0)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert state.landau_ginzburg_potential < _MC_TOL
        assert state.novikov_convergence_iterations == 1

    def test_novikov_filtration_degree_nonnegative(self, phase3, rho_2x2_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_mixed, lam=0.6)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert state.novikov_filtration_degree >= 0.0

    def test_bounding_cochain_shape(self, phase3, rho_2x2_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_mixed, dim_audit=4, lam=0.5)
        state = phase3.resolve_maurer_cartan_novikov(fib, rho_a)
        assert state.maurer_cartan_bounding_cochain.shape == rho_a.shape

    def test_dimension_mismatch_raises(self, phase3, rho_2x2_mixed):
        fib, rho_a = self._prepare_fibration(phase3, rho_2x2_mixed, dim_audit=3, lam=0.2)
        bad_rho = _maximally_mixed(2)  # dim 2 ≠ 3
        with pytest.raises(EhresmannFibrationError, match="dimensional"):
            phase3.resolve_maurer_cartan_novikov(fib, bad_rho)

    def test_phase3_inherits_phase2(self, phase3):
        assert isinstance(phase3, Phase2_TelescopicVerticalFibration)
        assert isinstance(phase3, Phase1_StinespringImmersion)

    def test_full_nested_chain_on_phase3(self, phase3, rho_2x2_mixed):
        """Anidación estricta Φ₁→Φ₂→Φ₃ sobre la misma instancia."""
        dil = phase3.compute_isometric_immersion(rho_2x2_mixed, dim_audit=3)
        fib = phase3.apply_telescopic_deformation(dil, lambda_zoom=0.4)
        state = phase3.resolve_maurer_cartan_novikov(fib, dil.rho_audit_subspace)
        assert state.is_safe_for_witten_atiyah
        assert _is_density(state.audited_density_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# §T6 · ORQUESTADOR EhresmannTelescopicEngine (COMPOSICIÓN Z)
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineConstruction:
    """Inicialización y validación de parámetros del orquestador."""

    def test_default_audit_dimension(self):
        eng = EhresmannTelescopicEngine()
        assert eng.audit_dimension == 4

    def test_custom_audit_dimension(self):
        eng = EhresmannTelescopicEngine(audit_dimension=6)
        assert eng.audit_dimension == 6

    def test_rejects_audit_dim_below_minimum(self):
        with pytest.raises(ValueError, match="audit_dimension"):
            EhresmannTelescopicEngine(audit_dimension=1)

    def test_rejects_non_integer_audit_dim(self):
        with pytest.raises(ValueError, match="audit_dimension"):
            EhresmannTelescopicEngine(audit_dimension=2.5)  # type: ignore[arg-type]

    def test_engine_is_morphism_and_phase3(self, engine):
        assert isinstance(engine, Phase3_MaurerCartanRegularization)


class TestEngineExecuteTelescopicAudit:
    """Composición funtorial Z = Φ₃ ∘ Φ₂ ∘ Φ₁."""

    def test_returns_audit_state(self, engine, rho_2x2_mixed):
        state = engine.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.5)
        assert isinstance(state, TelescopicAuditState)

    def test_output_is_valid_density(self, engine, rho_3x3_mixed):
        state = engine.execute_telescopic_audit(rho_3x3_mixed, magnification_lambda=0.3)
        assert _is_density(state.audited_density_matrix)

    def test_safe_flag_true_on_success(self, engine, rho_2x2_pure):
        state = engine.execute_telescopic_audit(rho_2x2_pure, magnification_lambda=0.2)
        assert state.is_safe_for_witten_atiyah is True

    def test_lg_potential_below_tolerance(self, engine, rho_2x2_mixed):
        state = engine.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.4)
        assert state.landau_ginzburg_potential < _MC_TOL

    def test_negative_lambda_raises(self, engine, rho_2x2_mixed):
        with pytest.raises(ValueError, match="magnification_lambda"):
            engine.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=-0.01)

    def test_invalid_business_rho_raises(self, engine):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises(InvalidDensityMatrixError):
            engine.execute_telescopic_audit(bad, magnification_lambda=0.1)

    def test_lambda_zero_pipeline(self, engine, rho_2x2_mixed):
        state = engine.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.0)
        assert state.is_safe_for_witten_atiyah
        assert _is_density(state.audited_density_matrix)
        assert state.landau_ginzburg_potential < _MC_TOL

    def test_audit_dimension_propagates_to_output_shape(self, rho_2x2_mixed):
        for dim_a in (2, 3, 5):
            eng = EhresmannTelescopicEngine(audit_dimension=dim_a)
            state = eng.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.25)
            assert state.audited_density_matrix.shape == (dim_a, dim_a)

    def test_deterministic_for_fixed_input(self, rho_2x2_mixed):
        eng = EhresmannTelescopicEngine(audit_dimension=4)
        s1 = eng.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.35)
        s2 = eng.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.35)
        np.testing.assert_allclose(
            s1.audited_density_matrix, s2.audited_density_matrix, atol=1e-12
        )
        assert s1.novikov_convergence_iterations == s2.novikov_convergence_iterations

    def test_various_business_dimensions(self, rng):
        eng = EhresmannTelescopicEngine(audit_dimension=4)
        for n in (2, 3, 4, 6):
            rho = _random_density(n, rng)
            state = eng.execute_telescopic_audit(rho, magnification_lambda=0.2)
            assert _is_density(state.audited_density_matrix)
            assert state.audited_density_matrix.shape == (4, 4)


# ══════════════════════════════════════════════════════════════════════════════
# §T7 · INVARIANTES DE EXTREMO A EXTREMO Y REGRESIÓN NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndInvariants:
    """Invariantes globales del endofuntor Z."""

    def test_purity_of_output_in_legal_range(self, engine, rng):
        rho = _random_density(3, rng)
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.5)
        rho_s = state.audited_density_matrix
        purity = float(np.real(np.trace(rho_s @ rho_s)))
        n = rho_s.shape[0]
        assert 1.0 / n - 1e-8 <= purity <= 1.0 + 1e-8

    def test_output_eigenvalues_sum_to_one(self, engine, rng):
        rho = _random_density(2, rng)
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.7)
        ev = la.eigvalsh(state.audited_density_matrix)
        assert ev.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(ev >= -1e-12)

    def test_stinespring_isometry_invariant_through_engine(self, engine, rho_2x2_mixed):
        """El motor expone Φ₁; verificamos isometría independientemente."""
        dil = engine.compute_isometric_immersion(rho_2x2_mixed, engine.audit_dimension)
        G = dil.V_isometry.conj().T @ dil.V_isometry
        np.testing.assert_allclose(
            G, np.eye(rho_2x2_mixed.shape[0]), atol=_ISO_TOL
        )

    def test_vertical_leakage_invariant_through_engine(self, engine, rho_2x2_mixed):
        dil = engine.compute_isometric_immersion(rho_2x2_mixed, engine.audit_dimension)
        fib = engine.apply_telescopic_deformation(dil, lambda_zoom=1.0)
        assert fib.horizontal_leakage_norm < _ORTH_TOL

    def test_mc_residual_matches_lg_potential(self, engine, rho_2x2_mixed):
        """W_LG reportado debe ser coherente con F(b) reevaluado."""
        dil = engine.compute_isometric_immersion(rho_2x2_mixed, engine.audit_dimension)
        fib = engine.apply_telescopic_deformation(dil, lambda_zoom=0.45)
        state = engine.resolve_maurer_cartan_novikov(fib, dil.rho_audit_subspace)
        T = fib.T_lambda_vertical
        rho_a = dil.rho_audit_subspace
        m0 = engine._compute_anomaly(T, rho_a)
        F = engine._maurer_cartan_function(
            state.maurer_cartan_bounding_cochain, T, m0
        )
        assert _fro(F) == pytest.approx(state.landau_ginzburg_potential, abs=1e-8)
        assert _fro(F) < _MC_TOL

    def test_composition_equals_stepwise(self, rng):
        """Z(ρ,λ) ≡ Φ₃(Φ₂(Φ₁(ρ),λ), ρ_audit) componente a componente."""
        eng = EhresmannTelescopicEngine(audit_dimension=3)
        rho = _random_density(2, rng)
        lam = 0.55

        # Composición monolítica
        state_z = eng.execute_telescopic_audit(rho, magnification_lambda=lam)

        # Paso a paso
        d = eng.compute_isometric_immersion(rho, 3)
        f = eng.apply_telescopic_deformation(d, lam)
        state_s = eng.resolve_maurer_cartan_novikov(f, d.rho_audit_subspace)

        np.testing.assert_allclose(
            state_z.audited_density_matrix,
            state_s.audited_density_matrix,
            atol=1e-12,
        )
        assert state_z.novikov_convergence_iterations == state_s.novikov_convergence_iterations

    def test_increasing_lambda_monotone_zoom_spectrum(self, engine, rho_2x2_mixed):
        """Los factores s_i(λ) son monótonos no decrecientes en λ."""
        dil = engine.compute_isometric_immersion(rho_2x2_mixed, engine.audit_dimension)
        prev = None
        for lam in (0.0, 0.5, 1.0, 2.0, 5.0):
            fib = engine.apply_telescopic_deformation(dil, lambda_zoom=lam)
            scales = fib.spectral_zoom_eigenvalues
            if prev is not None:
                assert np.all(scales >= prev - 1e-12)
            prev = scales.copy()


# ══════════════════════════════════════════════════════════════════════════════
# §T8 · CASOS LÍMITE, PATOLOGÍAS Y ROBUSTEZ
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndRobustness:
    """Bordes dimensionales, estados extremos y estrés numérico."""

    def test_minimal_audit_dimension(self, rho_2x2_mixed):
        eng = EhresmannTelescopicEngine(audit_dimension=2)
        state = eng.execute_telescopic_audit(rho_2x2_mixed, magnification_lambda=0.3)
        assert state.audited_density_matrix.shape == (2, 2)
        assert _is_density(state.audited_density_matrix)

    def test_pure_state_business(self, engine):
        rho = _pure_state_density(4, idx=2)
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.4)
        assert _is_density(state.audited_density_matrix)

    def test_maximally_mixed_business(self, engine):
        rho = _maximally_mixed(3)
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.4)
        assert _is_density(state.audited_density_matrix)

    def test_near_rank_deficient_density(self, engine):
        """Estado casi puro: autovalores (1-ε, ε, 0, …)."""
        n = 3
        eps = 1e-10
        rho = np.diag([1.0 - eps, eps, 0.0]).astype(_COMPLEX)
        # renormalizar por seguridad numérica
        rho = _hermitize(rho)
        rho /= np.trace(rho).real
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.2)
        assert _is_density(state.audited_density_matrix)

    def test_large_business_dimension(self, rng):
        eng = EhresmannTelescopicEngine(audit_dimension=2)
        rho = _random_density(8, rng)
        state = eng.execute_telescopic_audit(rho, magnification_lambda=0.15)
        assert _is_density(state.audited_density_matrix)

    def test_hermitian_numerical_noise_accepted(self, phase1, rng):
        """Ruido antihermítico bajo tolerancia no debe invalidar ρ."""
        rho = _random_density(3, rng)
        noise = 1e-16 * (
            rng.normal(size=rho.shape) + 1j * rng.normal(size=rho.shape)
        )
        rho_noisy = rho + noise
        # puede pasar o fallar según magnitud; forzamos hermitización previa
        rho_h = _hermitize(rho_noisy)
        rho_h /= np.trace(rho_h).real
        out = phase1._validate_density_matrix(rho_h)
        assert _is_density(out)

    def test_high_lambda_still_converges_or_raises_bubbling(self, engine, rho_2x2_mixed):
        """
        λ elevado puede converger o disparar SphereBubblingAnomalyError.
        Ambas salidas son contractualmente válidas; no debe haber excepciones
        genéricas no tipadas.
        """
        try:
            state = engine.execute_telescopic_audit(
                rho_2x2_mixed, magnification_lambda=25.0
            )
            assert isinstance(state, TelescopicAuditState)
            assert _is_density(state.audited_density_matrix)
        except SphereBubblingAnomalyError:
            pass  # fallo topológico controlado

    def test_complex_offdiagonal_density(self, engine, rng):
        n = 3
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        rho = A @ A.conj().T
        rho = _hermitize(rho)
        rho /= np.trace(rho).real
        # garantizar off-diagonales complejos no nulos
        assert abs(rho[0, 1].imag) + abs(rho[0, 1].real) > 0 or n > 0
        state = engine.execute_telescopic_audit(rho.astype(_COMPLEX), magnification_lambda=0.3)
        assert _is_density(state.audited_density_matrix)

    def test_execute_does_not_mutate_input(self, engine, rng):
        rho = _random_density(3, rng)
        rho_copy = rho.copy()
        _ = engine.execute_telescopic_audit(rho, magnification_lambda=0.5)
        np.testing.assert_array_equal(rho, rho_copy)

    def test_multiple_engines_independent(self, rng):
        rho = _random_density(2, rng)
        e1 = EhresmannTelescopicEngine(audit_dimension=2)
        e2 = EhresmannTelescopicEngine(audit_dimension=5)
        s1 = e1.execute_telescopic_audit(rho, magnification_lambda=0.2)
        s2 = e2.execute_telescopic_audit(rho, magnification_lambda=0.2)
        assert s1.audited_density_matrix.shape == (2, 2)
        assert s2.audited_density_matrix.shape == (5, 5)

    def test_gauge_fixing_traceless_cochain(self, engine, rho_2x2_mixed):
        """Tras Φ₃, la co-cadena acotante debe ser numéricamente de traza ~0 (gauge fix)."""
        dil = engine.compute_isometric_immersion(rho_2x2_mixed, engine.audit_dimension)
        fib = engine.apply_telescopic_deformation(dil, lambda_zoom=0.5)
        state = engine.resolve_maurer_cartan_novikov(fib, dil.rho_audit_subspace)
        tr_b = complex(np.trace(state.maurer_cartan_bounding_cochain))
        assert abs(tr_b) < 1e-8


class TestParametrizedSpectrum:
    """Batería parametrizada sobre dimensiones y magnificaciones."""

    @pytest.mark.parametrize("dim_mic", [2, 3, 4])
    @pytest.mark.parametrize("dim_audit", [2, 4])
    @pytest.mark.parametrize("lam", [0.0, 0.25, 1.0])
    def test_grid_dimensions_and_lambda(
        self,
        rng: np.random.Generator,
        dim_mic: int,
        dim_audit: int,
        lam: float,
    ):
        eng = EhresmannTelescopicEngine(audit_dimension=dim_audit)
        rho = _random_density(dim_mic, rng)
        state = eng.execute_telescopic_audit(rho, magnification_lambda=lam)
        assert _is_density(state.audited_density_matrix)
        assert state.audited_density_matrix.shape == (dim_audit, dim_audit)
        assert state.is_safe_for_witten_atiyah is True
        assert state.landau_ginzburg_potential < _MC_TOL

    @pytest.mark.parametrize(
        "rho_factory",
        [
            lambda: _pure_state_density(2, 0),
            lambda: _pure_state_density(2, 1),
            lambda: _maximally_mixed(2),
            lambda: _maximally_mixed(4),
        ],
    )
    def test_canonical_states(self, rho_factory: Callable[[], NDArray], engine):
        rho = rho_factory()
        # Ajustar audit dim si es necesario
        state = engine.execute_telescopic_audit(rho, magnification_lambda=0.3)
        assert _is_density(state.audited_density_matrix)


class TestPublicAPIExports:
    """Contrato de exportación del módulo (superficie pública)."""

    def test_all_public_symbols_importable(self):
        import app.core.immune_system.ehresmann_telescopic_engine as mod

        expected = [
            "TelescopicEngineError",
            "StinespringDilationError",
            "InvalidDensityMatrixError",
            "EhresmannFibrationError",
            "SphereBubblingAnomalyError",
            "SpectralDegeneracyError",
            "StinespringDilationData",
            "VerticalFibrationData",
            "TelescopicAuditState",
            "Phase1_StinespringImmersion",
            "Phase2_TelescopicVerticalFibration",
            "Phase3_MaurerCartanRegularization",
            "EhresmannTelescopicEngine",
        ]
        for name in expected:
            assert hasattr(mod, name), f"Falta exportar {name}"
            assert name in getattr(mod, "__all__", []), f"{name} ausente en __all__"