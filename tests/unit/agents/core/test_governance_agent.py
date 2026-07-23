# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de pruebas : test_governance_agent.py                                 ║
║ Ruta              : tests/unit/agents/core/test_governance_agent.py          ║
║ SUT               : app/agents/core/governance_agent.py (v3.0.0)             ║
╚══════════════════════════════════════════════════════════════════════════════╝

FILOSOFÍA DE VERIFICACIÓN:
────────────────────────────────────────────────────────────────────────────────
La suite se organiza en espejo exacto de las 3 fases anidadas del SUT, más
tres bloques transversales:

  §1. Fase 1 — Retículo distributivo dual + teoría de la información.
  §2. Fase 2 — Cohomología de haces + validación cruzada de Hodge.
  §3. Fase 3 — Pullback espectral en el Topos + axioma de pegado de haces.
  §4. Integración end-to-end vía GovernanceAgent.execute_federated_governance.
  §5. Invariantes transversales: inmutabilidad de DTOs, jerarquía de mixins.

Cada método de prueba certifica UN invariante matemático o UNA rama de
decisión. Las fixtures cohomológicas (`filled_triangle`, `hollow_triangle`)
son complejos simpliciales reales verificados analíticamente, no matrices
aleatorias — el disco relleno (χ=1, H¹=0) y su frontera S¹ (χ=0, H¹=1).
"""

from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import pytest

from app.agents.core.governance_agent import (
    CohomologicalOntologyData,
    CohomologyInputError,
    GovernanceAgent,
    GovernanceAgentError,
    HodgeCrossValidationError,
    LatticeInputError,
    LatticeProjectionData,
    OntologicalParadoxVeto,
    Phase1_InformationTheoreticLatticeProjector,
    Phase2_EulerCharacteristicCohomologyAuditor,
    Phase3_SpectralToposPolicyFunctor,
    PullbackInputError,
    SheafGluingObstructionError,
    StructuralVetoMonad,
    ToposPolicyPullbackData,
    ZeroTrustViolationError,
)


# ══════════════════════════════════════════════════════════════════════════════
# §0. FIXTURES CANÓNICAS (Complejos simpliciales y objetos del Topos)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def phase1() -> Phase1_InformationTheoreticLatticeProjector:
    """Instancia aislada de Fase 1 (stub de Fase 2 activo)."""
    return Phase1_InformationTheoreticLatticeProjector()


@pytest.fixture
def phase2() -> Phase2_EulerCharacteristicCohomologyAuditor:
    """Instancia de Fase 2 (implementación real de cohomología, stub de Fase 3)."""
    return Phase2_EulerCharacteristicCohomologyAuditor()


@pytest.fixture
def phase3() -> Phase3_SpectralToposPolicyFunctor:
    """Instancia de la cadena completa Φ₃∘Φ₂∘Φ₁."""
    return Phase3_SpectralToposPolicyFunctor()


@pytest.fixture
def agent() -> GovernanceAgent:
    """Instancia del orquestador supremo GovernanceAgent."""
    return GovernanceAgent()


@pytest.fixture
def filled_triangle_boundaries():
    r"""
    Complejo de cocadenas del **disco 2-simplicial relleno** (triángulo con
    su cara 2D presente): 3 vértices, 3 aristas, 1 cara.

    Verificado analíticamente:
        δ¹δ⁰ = 0
        rank(δ⁰) = 2, rank(δ¹) = 1
        dim H⁰ = 1, dim H¹ = 0, dim H² = 0
        χ = dim_C0 − dim_C1 + dim_C2 = 3 − 3 + 1 = 1  (disco es contráctil)
        Δ₁ = δ⁰δ⁰ᵀ + δ¹ᵀδ¹ = 3·I₃  ⇒  dim ker(Δ₁) = 0  (coincide con H¹=0)

    Es el caso canónico de **coherencia ontológica absoluta**.
    """
    d0 = np.array(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    d1 = np.array([[1.0, 1.0, -1.0]])
    return d0, d1


@pytest.fixture
def hollow_triangle_boundaries():
    r"""
    Complejo de cocadenas de la **frontera del triángulo** (círculo S¹, sin
    cara 2D): 3 vértices, 3 aristas, 0 caras (dim_C2 = 0).

    Verificado analíticamente:
        δ¹ es la matriz vacía (0,3) ⇒ δ¹δ⁰ = 0 trivialmente
        rank(δ⁰) = 2, rank(δ¹) = 0
        dim H⁰ = 1, dim H¹ = 1, dim H² = 0
        χ = 3 − 3 + 0 = 0  (círculo tiene característica de Euler nula)
        Δ₁ = δ⁰δ⁰ᵀ (matriz singular, det=0) ⇒ dim ker(Δ₁) = 1

    Es el caso canónico de **paradoja ontológica** (ciclo cohomológico no
    trivial: la ausencia de la cara 2D deja un "agujero" lógico).
    """
    d0 = np.array(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    d1 = np.zeros((0, 3))
    return d0, d1


@pytest.fixture
def valid_projector_and_domain():
    r"""
    Triple (Ω, X, S) que satisface el pullback del Topos de forma exacta:
        Ω = diag(1,0,0)   — proyector ortogonal canónico rank-1
        S = ΩX            — por construcción
    """
    omega = np.diag([1.0, 0.0, 0.0])
    x_domain = np.array([[1.0], [2.0], [3.0]])
    s_allowed = omega @ x_domain
    return omega, x_domain, s_allowed


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §1. FASE 1 — RETÍCULO DISTRIBUTIVO DUAL + TEORÍA DE LA INFORMACIÓN          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestShannonEntropyComputation:
    """Certifica `_shannon_entropy_bits`: distinción entre fallo puntual y difuso."""

    def test_single_dominant_violation_yields_zero_entropy(self, phase1):
        entropy, max_entropy = phase1._shannon_entropy_bits([1.0, 0.0, 0.0])
        assert entropy == pytest.approx(0.0, abs=1e-12)
        assert max_entropy == pytest.approx(np.log2(3))

    def test_uniform_distribution_yields_maximal_entropy(self, phase1):
        entropy, max_entropy = phase1._shannon_entropy_bits([0.5, 0.5])
        assert entropy == pytest.approx(1.0)
        assert max_entropy == pytest.approx(1.0)

    def test_empty_sequence_yields_zero_entropy_and_zero_max(self, phase1):
        entropy, max_entropy = phase1._shannon_entropy_bits([])
        assert entropy == 0.0
        assert max_entropy == 0.0

    def test_all_zero_values_yields_zero_entropy_but_nonzero_max(self, phase1):
        entropy, max_entropy = phase1._shannon_entropy_bits([0.0, 0.0, 0.0])
        assert entropy == 0.0
        assert max_entropy == pytest.approx(np.log2(3))

    def test_single_element_sequence_has_zero_max_entropy(self, phase1):
        _entropy, max_entropy = phase1._shannon_entropy_bits([0.7])
        assert max_entropy == 0.0

    def test_partial_concentration_yields_intermediate_entropy(self, phase1):
        # Distribución [0.8, 0.1, 0.1] normalizada: entropía estrictamente
        # entre 0 (concentración total) y log2(3) (uniforme).
        entropy, max_entropy = phase1._shannon_entropy_bits([0.8, 0.1, 0.1])
        assert 0.0 < entropy < max_entropy


class TestDeMorganDualityCertification:
    """
    Certifica `_de_morgan_duality_residual`: guardia de regresión numérica
    de la identidad ⊔ᵢvᵢ = 1 − ⊓ᵢ(1−vᵢ).
    """

    def test_duality_holds_exactly_for_arbitrary_values(self, phase1):
        values = [0.2, 0.5, 0.9]
        supremum = max(values)
        residual = phase1._de_morgan_duality_residual(values, supremum)
        assert residual == pytest.approx(0.0, abs=1e-12)

    def test_duality_holds_for_empty_sequence(self, phase1):
        residual = phase1._de_morgan_duality_residual([], 0.0)
        assert residual == 0.0

    def test_duality_detects_deliberately_corrupted_supremum(self, phase1):
        values = [0.2, 0.5, 0.9]
        corrupted_supremum = 0.5  # valor incorrecto deliberado
        residual = phase1._de_morgan_duality_residual(values, corrupted_supremum)
        assert residual == pytest.approx(0.4, abs=1e-12)


class TestDistributiveLatticeProjection:
    """Certifica `_project_to_distributive_lattice`: join, meet y sus identidades."""

    def test_typical_violation_vector_is_structurally_sound(self, phase1):
        audit = phase1._project_to_distributive_lattice([0.1, 0.3, 0.2])
        assert audit.supremum_state == pytest.approx(0.3)
        assert audit.infimum_state == pytest.approx(0.1)
        assert audit.is_structurally_sound is True

    def test_empty_violations_yield_bottom_supremum_and_top_infimum(self, phase1):
        audit = phase1._project_to_distributive_lattice([])
        assert audit.supremum_state == 0.0  # identidad join(∅) = ⊥
        assert audit.infimum_state == 1.0   # identidad meet(∅) = ⊤
        assert audit.is_structurally_sound is True

    def test_scalar_input_is_accepted_as_singleton_sequence(self, phase1):
        audit = phase1._project_to_distributive_lattice(0.42)
        assert audit.violation_count == 1
        assert audit.supremum_state == pytest.approx(0.42)

    def test_supremum_at_top_threshold_triggers_structural_veto(self, phase1):
        with pytest.raises(StructuralVetoMonad):
            phase1._project_to_distributive_lattice([1.0])

    def test_veto_disabled_returns_unsound_audit_without_raising(self, phase1):
        audit = phase1._project_to_distributive_lattice([1.0], raise_on_veto=False)
        assert audit.is_structurally_sound is False
        assert audit.critical_count == 1

    def test_out_of_range_violation_raises_lattice_input_error(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice([1.5])

    def test_nan_violation_raises_lattice_input_error(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice([float("nan")])

    def test_string_input_is_explicitly_rejected(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice("0.5")

    def test_none_input_raises_lattice_input_error(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice(None)  # type: ignore[arg-type]

    def test_non_numeric_non_iterable_input_raises_lattice_input_error(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice(object())  # type: ignore[arg-type]

    def test_invalid_top_threshold_zero_raises(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice([0.1], top_threshold=0.0)

    def test_invalid_top_threshold_above_range_raises(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice([0.1], top_threshold=2.0)

    def test_negative_lattice_tolerance_raises(self, phase1):
        with pytest.raises(LatticeInputError):
            phase1._project_to_distributive_lattice([0.1], lattice_tolerance=-1e-6)

    def test_audit_reports_entropy_and_de_morgan_residual(self, phase1):
        audit = phase1._project_to_distributive_lattice([0.5, 0.5])
        assert audit.shannon_entropy_bits == pytest.approx(1.0)
        assert audit.de_morgan_duality_residual == pytest.approx(0.0, abs=1e-9)

    def test_all_bottom_violations_are_annotated_in_notes(self, phase1):
        audit = phase1._project_to_distributive_lattice([0.0, 0.0])
        assert any("⊥" in note for note in audit.notes)


class TestPhase1StubContractTowardsPhase2:
    """Certifica que Fase 1, aislada, falla explícitamente al alcanzar Fase 2."""

    def test_ontology_stub_raises_not_implemented_in_isolation(
        self, phase1, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        with pytest.raises(NotImplementedError):
            phase1._audit_semantic_ontology(d0, d1)

    def test_terminal_bridge_propagates_stub_failure(
        self, phase1, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        with pytest.raises(NotImplementedError):
            phase1._phase1_terminal_bridge_to_phase2([0.1], d0, d1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §2. FASE 2 — COHOMOLOGÍA DE HACES + VALIDACIÓN CRUZADA DE HODGE             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestFiniteMatrixValidatorWithInjectableExceptionType:
    """
    Certifica `_as_finite_matrix` y su inyección de `exception_cls`
    (corrección Curry-Howard: Fase 3 debe poder tipificar sus propios errores).
    """

    def test_valid_2d_matrix_passes_through(self, phase2):
        matrix = phase2._as_finite_matrix("M", np.eye(2))
        assert matrix.shape == (2, 2)

    def test_non_2d_input_raises_default_cohomology_error(self, phase2):
        with pytest.raises(CohomologyInputError):
            phase2._as_finite_matrix("M", np.array([1.0, 2.0]))

    def test_non_finite_values_raise_default_cohomology_error(self, phase2):
        matrix = np.array([[1.0, np.nan]])
        with pytest.raises(CohomologyInputError):
            phase2._as_finite_matrix("M", matrix)

    def test_exception_cls_injection_overrides_default_type(self, phase2):
        with pytest.raises(PullbackInputError):
            phase2._as_finite_matrix(
                "Omega", np.array([np.inf]), exception_cls=PullbackInputError
            )


class TestFrobeniusNormAndSVDRank:
    """Certifica `_frobenius_norm` y `_svd_rank` como primitivas cohomológicas."""

    def test_frobenius_norm_of_identity_matches_sqrt_dimension(self, phase2):
        norm = phase2._frobenius_norm(np.eye(4))
        assert norm == pytest.approx(2.0)

    def test_frobenius_norm_of_empty_matrix_is_zero(self, phase2):
        assert phase2._frobenius_norm(np.zeros((0, 0))) == 0.0

    def test_svd_rank_of_identity_matrix_equals_dimension(self, phase2):
        rank, max_singular, _tol = phase2._svd_rank("M", np.eye(3), 1e-10)
        assert rank == 3
        assert max_singular == pytest.approx(1.0)

    def test_svd_rank_of_zero_matrix_is_zero(self, phase2):
        rank, max_singular, _tol = phase2._svd_rank("M", np.zeros((3, 3)), 1e-10)
        assert rank == 0
        assert max_singular == 0.0

    def test_svd_rank_of_degenerate_shape_matrix_is_trivially_zero(self, phase2):
        rank, max_singular, tol = phase2._svd_rank("M", np.zeros((0, 3)), 1e-10)
        assert rank == 0
        assert max_singular == 0.0
        assert tol == 1e-10

    def test_svd_rank_of_known_rank_two_matrix(self, phase2, filled_triangle_boundaries):
        d0, _d1 = filled_triangle_boundaries
        rank, _max_singular, _tol = phase2._svd_rank("boundary_d0", d0, 1e-10)
        assert rank == 2


class TestHodgeLaplacianKernelDimension:
    """
    Certifica `_hodge_laplacian_kernel_dimension`: segunda vía algorítmica
    independiente para dim H¹, verificada contra complejos simpliciales
    reales con resultado analítico conocido.
    """

    def test_filled_triangle_has_trivial_hodge_kernel(
        self, phase2, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        kernel_dim, _tol = phase2._hodge_laplacian_kernel_dimension(d0, d1, 1e-9)
        assert kernel_dim == 0

    def test_hollow_triangle_has_one_dimensional_hodge_kernel(
        self, phase2, hollow_triangle_boundaries
    ):
        d0, d1 = hollow_triangle_boundaries
        kernel_dim, _tol = phase2._hodge_laplacian_kernel_dimension(d0, d1, 1e-9)
        assert kernel_dim == 1

    def test_zero_dimensional_cochain_space_has_trivial_kernel(self, phase2):
        d0 = np.zeros((0, 0))
        d1 = np.zeros((0, 0))
        kernel_dim, _tol = phase2._hodge_laplacian_kernel_dimension(d0, d1, 1e-9)
        assert kernel_dim == 0


class TestSemanticOntologyAudit:
    """Certifica `_audit_semantic_ontology`: el corazón cohomológico de Fase 2."""

    def test_filled_triangle_is_ontologically_coherent(
        self, phase2, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        audit = phase2._audit_semantic_ontology(d0, d1)

        assert audit.is_ontologically_coherent is True
        assert audit.dim_H0 == 1
        assert audit.dim_H1 == 0
        assert audit.dim_H2 == 0
        assert audit.euler_characteristic == 1
        assert audit.is_hodge_cross_validated is True
        assert audit.dim_H1_hodge == 0

    def test_hollow_triangle_raises_ontological_paradox_veto(
        self, phase2, hollow_triangle_boundaries
    ):
        d0, d1 = hollow_triangle_boundaries
        with pytest.raises(OntologicalParadoxVeto):
            phase2._audit_semantic_ontology(d0, d1)

    def test_hollow_triangle_veto_disabled_reports_full_diagnostic(
        self, phase2, hollow_triangle_boundaries
    ):
        d0, d1 = hollow_triangle_boundaries
        audit = phase2._audit_semantic_ontology(d0, d1, raise_on_veto=False)

        assert audit.is_ontologically_coherent is False
        assert audit.dim_H1 == 1
        assert audit.euler_characteristic == 0
        # La validación cruzada de Hodge coincide (ambos métodos ven H¹=1):
        # la paradoja es ontológica, no una discrepancia numérica.
        assert audit.is_hodge_cross_validated is True
        assert audit.dim_H1_hodge == 1

    def test_shape_mismatch_between_boundary_maps_raises_input_error(self, phase2):
        d0 = np.eye(2)
        d1 = np.eye(3)  # dim_C1 incompatible (2 vs 3)
        with pytest.raises(CohomologyInputError):
            phase2._audit_semantic_ontology(d0, d1)

    def test_non_zero_composition_raises_paradox_when_complex_required(self, phase2):
        d0 = np.eye(2)
        d1 = np.eye(2)  # δ¹δ⁰ = I ≠ 0
        with pytest.raises(OntologicalParadoxVeto):
            phase2._audit_semantic_ontology(d0, d1, require_cochain_complex=True)

    def test_non_zero_composition_tolerated_when_complex_not_required(self, phase2):
        d0 = np.eye(2)
        d1 = np.eye(2)
        audit = phase2._audit_semantic_ontology(
            d0, d1, require_cochain_complex=False, raise_on_veto=False
        )
        assert any("require_cochain_complex=False" in note for note in audit.notes)

    def test_negative_svd_tolerance_raises_input_error(
        self, phase2, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        with pytest.raises(CohomologyInputError):
            phase2._audit_semantic_ontology(d0, d1, svd_tolerance=-1.0)

    def test_negative_hodge_tolerance_raises_input_error(
        self, phase2, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        with pytest.raises(CohomologyInputError):
            phase2._audit_semantic_ontology(d0, d1, hodge_tolerance=-1.0)

    def test_hodge_validation_is_skipped_when_complex_condition_fails(self, phase2):
        d0 = np.eye(2)
        d1 = np.eye(2)  # complejo inválido
        audit = phase2._audit_semantic_ontology(
            d0, d1, require_cochain_complex=False, raise_on_veto=False
        )
        assert audit.dim_H1_hodge is None
        assert any("omitida" in note for note in audit.notes)


class TestNegativeDimH1ClampingDefensiveBranch:
    """
    Ejercita, vía `monkeypatch`, la rama de clamp `dim_H1_raw < 0 → 0`.

    Esta rama es matemáticamente inalcanzable para un complejo de cocadenas
    genuino (δ¹δ⁰=0 garantiza dim_H1_raw ≥ 0 por el teorema de rango-nulidad);
    se ejercita forzando valores de rango artificiales para certificar que la
    red de seguridad numérica del código no produce un Betti number negativo.
    """

    def test_forged_rank_values_are_clamped_to_zero_with_diagnostic_note(
        self, phase2, monkeypatch
    ):
        def _forged_svd_rank(self, name, matrix, base_tolerance, **kwargs):
            if name == "boundary_d0":
                return 5, 1.0, base_tolerance  # rango irrealmente alto
            return 0, 1.0, base_tolerance

        monkeypatch.setattr(
            Phase2_EulerCharacteristicCohomologyAuditor,
            "_svd_rank",
            _forged_svd_rank,
        )

        d0 = np.zeros((2, 2))
        d1 = np.zeros((1, 2))

        audit = phase2._audit_semantic_ontology(
            d0, d1, require_hodge_consistency=False, raise_on_veto=False
        )

        assert audit.dim_H1 == 0
        assert any("clamp" in note.lower() for note in audit.notes)


class TestHodgeCrossValidationDefensiveBranch:
    """
    Ejercita, vía `monkeypatch`, `HodgeCrossValidationError`.

    Bajo la hipótesis δ¹δ⁰=0, ambos algoritmos (SVD y Laplaciano de Hodge)
    están garantizados a coincidir por el Teorema de Hodge discreta; esta
    prueba forja una discrepancia artificial para certificar que el sistema
    de alarma epistémica (distinto del ontológico) responde correctamente.
    """

    def test_forged_hodge_mismatch_raises_cross_validation_error(
        self, phase2, filled_triangle_boundaries, monkeypatch
    ):
        d0, d1 = filled_triangle_boundaries

        def _forged_hodge_kernel(self, d0_arg, d1_arg, tolerance_scale):
            return 99, float(tolerance_scale)  # valor deliberadamente falso

        monkeypatch.setattr(
            Phase2_EulerCharacteristicCohomologyAuditor,
            "_hodge_laplacian_kernel_dimension",
            _forged_hodge_kernel,
        )

        with pytest.raises(HodgeCrossValidationError):
            phase2._audit_semantic_ontology(d0, d1, require_hodge_consistency=True)

    def test_forged_hodge_mismatch_tolerated_when_consistency_not_required(
        self, phase2, filled_triangle_boundaries, monkeypatch
    ):
        d0, d1 = filled_triangle_boundaries

        def _forged_hodge_kernel(self, d0_arg, d1_arg, tolerance_scale):
            return 99, float(tolerance_scale)

        monkeypatch.setattr(
            Phase2_EulerCharacteristicCohomologyAuditor,
            "_hodge_laplacian_kernel_dimension",
            _forged_hodge_kernel,
        )

        audit = phase2._audit_semantic_ontology(
            d0, d1, require_hodge_consistency=False
        )

        assert audit.is_hodge_cross_validated is False
        assert audit.is_ontologically_coherent is True  # ontología sigue siendo válida


class TestPhase2StubContractTowardsPhase3:
    """Certifica que Fase 2, aislada, falla explícitamente al alcanzar Fase 3."""

    def test_pullback_stub_raises_not_implemented_in_isolation(
        self, phase2, valid_projector_and_domain
    ):
        omega, x_domain, s_allowed = valid_projector_and_domain
        with pytest.raises(NotImplementedError):
            phase2._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_terminal_bridge_propagates_stub_failure(
        self, phase2, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain
        with pytest.raises(NotImplementedError):
            phase2._phase2_terminal_bridge_to_phase3(
                [0.1], d0, d1, x_domain, s_allowed, omega
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §3. FASE 3 — PULLBACK ESPECTRAL EN EL TOPOS + AXIOMA DE PEGADO              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestProjectorSpectralBinaryTheorem:
    """Certifica σ(Ω) ⊆ {0,1} de forma algorítmicamente independiente."""

    def test_canonical_projector_has_negligible_spectral_residual(self, phase3):
        omega = np.diag([1.0, 0.0, 0.0])
        residual = phase3._certify_projector_spectral_binary(omega, 1e-9)
        assert residual < 1e-9

    def test_non_projector_scalar_matrix_reports_exact_distance(self, phase3):
        omega = np.diag([0.5, 0.5])
        residual = phase3._certify_projector_spectral_binary(omega, 1e-9)
        assert residual == pytest.approx(0.5, abs=1e-9)

    def test_zero_dimensional_operator_has_zero_residual(self, phase3):
        assert phase3._certify_projector_spectral_binary(np.zeros((0, 0)), 1e-9) == 0.0

    def test_oblique_idempotent_projector_flagged_by_symmetrization_side_effect(
        self, phase3
    ):
        """
        Hallazgo de rigor documentado: `_certify_projector_spectral_binary`
        simetriza internamente antes de diagonalizar (Ω_sym=(Ω+Ωᵀ)/2). Para
        el proyector oblicuo genuino [[1,1],[0,0]] (idempotente, σ(Ω)={0,1}
        exacto pero no simétrico), la matriz simetrizada tiene autovalores
        ≈{1.207,-0.207} ∉ {0,1}. Documentado como especificación de facto,
        idéntica al comportamiento observado en `app_agent.py` Fase 4.
        """
        oblique_projector = np.array([[1.0, 1.0], [0.0, 0.0]])
        residual = phase3._certify_projector_spectral_binary(oblique_projector, 1e-9)
        assert residual > 0.1


class TestTraceRankIdentityCertification:
    """Certifica tr(Ω) = rank(Ω) para proyectores idempotentes."""

    def test_rank_one_projector_satisfies_trace_rank_identity(self, phase3):
        omega = np.diag([1.0, 0.0, 0.0])
        residual = phase3._certify_trace_rank_identity(omega, projector_rank=1)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_rank_two_projector_satisfies_trace_rank_identity(self, phase3):
        omega = np.diag([1.0, 1.0, 0.0])
        residual = phase3._certify_trace_rank_identity(omega, projector_rank=2)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_non_idempotent_matrix_violates_trace_rank_identity(self, phase3):
        omega = np.diag([0.5, 0.5, 0.5])  # tr=1.5, rank=3
        residual = phase3._certify_trace_rank_identity(omega, projector_rank=3)
        assert residual == pytest.approx(1.5, abs=1e-9)

    def test_undefined_rank_propagates_as_none_not_sentinel(self, phase3):
        omega = np.eye(2)
        residual = phase3._certify_trace_rank_identity(omega, projector_rank=None)
        assert residual is None

    def test_zero_dimensional_operator_has_zero_residual(self, phase3):
        residual = phase3._certify_trace_rank_identity(np.zeros((0, 0)), 0)
        assert residual == 0.0


class TestPolicyPullbackValidation:
    """Certifica `_validate_policy_pullback`: el corazón espectral de Fase 3."""

    def test_valid_canonical_projector_is_fully_verified(
        self, phase3, valid_projector_and_domain
    ):
        omega, x_domain, s_allowed = valid_projector_and_domain
        audit = phase3._validate_policy_pullback(x_domain, s_allowed, omega)

        assert audit.is_zero_trust_verified is True
        assert audit.idempotence_residual < 1e-9
        assert audit.symmetry_residual < 1e-9
        assert audit.spectral_binary_residual < 1e-9
        assert audit.trace_rank_residual == pytest.approx(0.0, abs=1e-9)
        assert audit.pullback_residual < 1e-9
        assert audit.allowed_fixed_residual < 1e-9
        assert audit.projector_rank == 1
        assert audit.projector_trace == pytest.approx(1.0)

    def test_non_idempotent_projector_raises_zero_trust_violation(self, phase3):
        omega = np.diag([0.5, 0.5, 0.5])
        x_domain = np.array([[1.0], [1.0], [1.0]])
        s_allowed = omega @ x_domain
        with pytest.raises(ZeroTrustViolationError):
            phase3._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_non_symmetric_projector_raises_when_hermiticity_required(self, phase3):
        oblique_projector = np.array([[1.0, 1.0], [0.0, 0.0]])
        x_domain = np.array([[1.0], [0.0]])
        s_allowed = oblique_projector @ x_domain
        with pytest.raises(ZeroTrustViolationError):
            phase3._validate_policy_pullback(
                x_domain, s_allowed, oblique_projector, require_hermitian_projector=True
            )

    def test_bad_pullback_reconstruction_raises_violation(self, phase3):
        omega = np.diag([1.0, 0.0, 0.0])
        x_domain = np.array([[1.0], [2.0], [3.0]])
        wrong_s = np.array([[99.0], [0.0], [0.0]])  # S ≠ ΩX
        with pytest.raises(ZeroTrustViolationError):
            phase3._validate_policy_pullback(x_domain, wrong_s, omega)

    def test_disabling_structural_flags_bypasses_projector_checks(self, phase3):
        # Ω no es un proyector genuino, pero con todos los flags estructurales
        # desactivados y un dominio trivial (X=S=0), solo se exige el
        # pullback y el punto fijo, ambos triviales.
        non_projector_omega = np.array([[0.3, 0.0], [0.0, 0.3]])
        x_domain = np.zeros((2, 1))
        s_allowed = np.zeros((2, 1))

        audit = phase3._validate_policy_pullback(
            x_domain,
            s_allowed,
            non_projector_omega,
            require_projector=False,
            require_hermitian_projector=False,
            require_spectral_projector=False,
        )

        assert audit.is_zero_trust_verified is True

    def test_non_square_omega_raises_pullback_input_error(self, phase3):
        omega = np.ones((2, 3))
        x_domain = np.zeros((2, 1))
        s_allowed = np.zeros((2, 1))
        with pytest.raises(PullbackInputError):
            phase3._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_domain_dimension_mismatch_raises_pullback_input_error(self, phase3):
        omega = np.eye(3)
        x_domain = np.zeros((2, 1))  # dimensión incompatible con Ω (3x3)
        s_allowed = np.zeros((2, 1))
        with pytest.raises(PullbackInputError):
            phase3._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_allowed_shape_mismatch_raises_pullback_input_error(self, phase3):
        omega = np.eye(2)
        x_domain = np.zeros((2, 1))
        s_allowed = np.zeros((2, 2))  # forma distinta a X
        with pytest.raises(PullbackInputError):
            phase3._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_non_finite_omega_raises_pullback_specific_input_error(self, phase3):
        omega = np.array([[np.nan, 0.0], [0.0, 1.0]])
        x_domain = np.zeros((2, 1))
        s_allowed = np.zeros((2, 1))
        with pytest.raises(PullbackInputError):
            phase3._validate_policy_pullback(x_domain, s_allowed, omega)

    def test_negative_pullback_tolerance_raises_input_error(
        self, phase3, valid_projector_and_domain
    ):
        omega, x_domain, s_allowed = valid_projector_and_domain
        with pytest.raises(PullbackInputError):
            phase3._validate_policy_pullback(
                x_domain, s_allowed, omega, pullback_tolerance=-1.0
            )

    def test_veto_disabled_returns_invalid_audit_without_raising(self, phase3):
        omega = np.diag([0.5, 0.5, 0.5])
        x_domain = np.array([[1.0], [1.0], [1.0]])
        s_allowed = omega @ x_domain
        audit = phase3._validate_policy_pullback(
            x_domain, s_allowed, omega, raise_on_veto=False
        )
        assert audit.is_zero_trust_verified is False

    def test_rank_two_projector_reconstructs_domain_correctly(self, phase3):
        omega = np.diag([1.0, 1.0, 0.0])
        x_domain = np.array([[1.0], [2.0], [3.0]])
        s_allowed = omega @ x_domain
        audit = phase3._validate_policy_pullback(x_domain, s_allowed, omega)
        assert audit.is_zero_trust_verified is True
        assert audit.projector_rank == 2

    def test_relative_pullback_residual_is_reported_and_bounded(
        self, phase3, valid_projector_and_domain
    ):
        omega, x_domain, s_allowed = valid_projector_and_domain
        audit = phase3._validate_policy_pullback(x_domain, s_allowed, omega)
        assert audit.relative_pullback_residual >= 0.0
        assert audit.relative_pullback_residual < 1e-6

    def test_zero_dimensional_domain_is_trivially_verified(self, phase3):
        omega = np.zeros((0, 0))
        x_domain = np.zeros((0, 1))
        s_allowed = np.zeros((0, 1))
        audit = phase3._validate_policy_pullback(x_domain, s_allowed, omega)
        assert audit.domain_rows == 0
        assert audit.is_zero_trust_verified is True


class TestSheafGluingObstruction:
    """
    Certifica `_certify_sheaf_gluing_obstruction`: el axioma de pegado sobre
    la cubierta {U_lattice, U_ontology, U_hodge, U_pullback}.
    """

    @staticmethod
    def _consistent_audits():
        lattice = LatticeProjectionData(
            supremum_state=0.0,
            infimum_state=1.0,
            bottom_state=0.0,
            top_threshold=0.999999999,
            violation_count=0,
            critical_count=0,
            shannon_entropy_bits=0.0,
            max_entropy_bits=0.0,
            de_morgan_duality_residual=0.0,
            is_structurally_sound=True,
        )
        cohomology = CohomologicalOntologyData(
            dim_H0=1,
            dim_H1=0,
            dim_H2=0,
            betti_0_image=2,
            betti_1_kernel=2,
            dim_C0=3,
            dim_C1=3,
            dim_C2=1,
            rank_d0=2,
            rank_d1=1,
            euler_characteristic=1,
            dim_H1_hodge=0,
            hodge_cross_validation_residual=0,
            hodge_tolerance=1e-9,
            cochain_residual=0.0,
            cochain_tolerance=1e-10,
            rank_tolerance_d0=1e-10,
            rank_tolerance_d1=1e-10,
            is_ontologically_coherent=True,
            is_hodge_cross_validated=True,
        )
        pullback = ToposPolicyPullbackData(
            pullback_residual=0.0,
            relative_pullback_residual=0.0,
            tolerance=1e-12,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            spectral_binary_residual=0.0,
            projector_trace=1.0,
            trace_rank_residual=0.0,
            allowed_fixed_residual=0.0,
            projector_rank=1,
            domain_rows=3,
            domain_columns=1,
            is_zero_trust_verified=True,
        )
        return lattice, cohomology, pullback

    def test_all_consistent_sections_yield_zero_obstruction(self, phase3):
        lattice, cohomology, pullback = self._consistent_audits()
        index, failing = phase3._certify_sheaf_gluing_obstruction(
            lattice, cohomology, pullback
        )
        assert index == 0
        assert failing == ()

    def test_lattice_failure_contributes_single_obstruction(self, phase3):
        lattice, cohomology, pullback = self._consistent_audits()
        broken = dataclasses.replace(lattice, is_structurally_sound=False)
        index, failing = phase3._certify_sheaf_gluing_obstruction(
            broken, cohomology, pullback
        )
        assert index == 1
        assert failing == ("U_lattice:Retículo-Distributivo",)

    def test_ontology_failure_contributes_single_obstruction(self, phase3):
        lattice, cohomology, pullback = self._consistent_audits()
        broken = dataclasses.replace(cohomology, is_ontologically_coherent=False)
        index, failing = phase3._certify_sheaf_gluing_obstruction(
            lattice, broken, pullback
        )
        assert index == 1
        assert failing == ("U_ontology:Cohomología-Ontológica",)

    def test_hodge_failure_is_distinct_from_ontology_failure(self, phase3):
        """
        Certifica que la obstrucción de Hodge (epistémica) es una carta
        distinta de la obstrucción ontológica, aun cuando ambas puedan
        coexistir en el mismo `CohomologicalOntologyData`.
        """
        lattice, cohomology, pullback = self._consistent_audits()
        broken = dataclasses.replace(cohomology, is_hodge_cross_validated=False)
        index, failing = phase3._certify_sheaf_gluing_obstruction(
            lattice, broken, pullback
        )
        assert index == 1
        assert failing == ("U_hodge:Validación-Cruzada-Hodge",)

    def test_pullback_failure_contributes_single_obstruction(self, phase3):
        lattice, cohomology, pullback = self._consistent_audits()
        broken = dataclasses.replace(pullback, is_zero_trust_verified=False)
        index, failing = phase3._certify_sheaf_gluing_obstruction(
            lattice, cohomology, broken
        )
        assert index == 1
        assert failing == ("U_pullback:Pullback-ZeroTrust",)

    def test_all_four_sectors_failing_accumulate_to_four(self, phase3):
        lattice, cohomology, pullback = self._consistent_audits()
        broken_lattice = dataclasses.replace(lattice, is_structurally_sound=False)
        broken_cohomology = dataclasses.replace(
            cohomology, is_ontologically_coherent=False, is_hodge_cross_validated=False
        )
        broken_pullback = dataclasses.replace(pullback, is_zero_trust_verified=False)

        index, failing = phase3._certify_sheaf_gluing_obstruction(
            broken_lattice, broken_cohomology, broken_pullback
        )

        assert index == 4
        assert set(failing) == {
            "U_lattice:Retículo-Distributivo",
            "U_ontology:Cohomología-Ontológica",
            "U_hodge:Validación-Cruzada-Hodge",
            "U_pullback:Pullback-ZeroTrust",
        }


class TestPhase3TerminalSynthesis:
    """Certifica `_phase3_terminal_synthesis`: la síntesis final de gobernanza."""

    def test_full_valid_pipeline_yields_compliant_state(
        self, phase3, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        state = phase3._phase3_terminal_synthesis(
            [0.1, 0.2], d0, d1, x_domain, s_allowed, omega
        )

        assert state.is_fully_compliant is True
        assert state.gluing_obstruction_index == 0
        assert state.gluing_obstruction_failing_charts == ()
        uuid.UUID(state.governance_id)

    def test_ontological_paradox_vetoes_synthesis_by_default(
        self, phase3, hollow_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = hollow_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        with pytest.raises(OntologicalParadoxVeto):
            phase3._phase3_terminal_synthesis(
                [0.1], d0, d1, x_domain, s_allowed, omega
            )

    def test_veto_disabled_aggregates_obstruction_from_single_sector(
        self, phase3, hollow_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = hollow_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        state = phase3._phase3_terminal_synthesis(
            [0.1], d0, d1, x_domain, s_allowed, omega, raise_on_veto=False
        )

        assert state.is_fully_compliant is False
        assert state.gluing_obstruction_index >= 1
        assert "U_ontology:Cohomología-Ontológica" in state.gluing_obstruction_failing_charts


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §4. INTEGRACIÓN END-TO-END — GOVERNANCEAGENT.EXECUTE_FEDERATED_GOVERNANCE   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestGovernanceAgentEndToEnd:
    """Certifica la composición funtorial completa Z_Gov = Φ₃∘Φ₂∘Φ₁."""

    def test_full_happy_path_certifies_compliant_governance(
        self, agent, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        state = agent.execute_federated_governance(
            [0.0, 0.1], d0, d1, x_domain, s_allowed, omega
        )

        assert state.is_fully_compliant is True
        assert state.lattice_audit.is_structurally_sound is True
        assert state.cohomology_audit.is_ontologically_coherent is True
        assert state.cohomology_audit.is_hodge_cross_validated is True
        assert state.pullback_audit.is_zero_trust_verified is True

    def test_lattice_top_state_vetoes_entire_governance(
        self, agent, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        with pytest.raises(StructuralVetoMonad):
            agent.execute_federated_governance(
                [1.0], d0, d1, x_domain, s_allowed, omega
            )

    def test_ontological_paradox_vetoes_entire_governance(
        self, agent, hollow_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = hollow_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        with pytest.raises(OntologicalParadoxVeto):
            agent.execute_federated_governance(
                [0.1], d0, d1, x_domain, s_allowed, omega
            )

    def test_zero_trust_violation_vetoes_entire_governance(
        self, agent, filled_triangle_boundaries
    ):
        d0, d1 = filled_triangle_boundaries
        bad_omega = np.diag([0.5, 0.5, 0.5])
        x_domain = np.array([[1.0], [1.0], [1.0]])
        bad_s_allowed = bad_omega @ x_domain

        with pytest.raises(ZeroTrustViolationError):
            agent.execute_federated_governance(
                [0.1], d0, d1, x_domain, bad_s_allowed, bad_omega
            )

    def test_raise_on_veto_false_aggregates_all_sector_failures(
        self, agent, hollow_triangle_boundaries
    ):
        d0, d1 = hollow_triangle_boundaries  # rompe cohomología
        bad_omega = np.diag([0.5, 0.5, 0.5])  # rompe pullback
        x_domain = np.array([[1.0], [1.0], [1.0]])
        bad_s_allowed = bad_omega @ x_domain

        state = agent.execute_federated_governance(
            [1.0],  # rompe retículo
            d0,
            d1,
            x_domain,
            bad_s_allowed,
            bad_omega,
            raise_on_veto=False,
        )

        assert state.is_fully_compliant is False
        assert state.gluing_obstruction_index >= 3
        assert "U_lattice:Retículo-Distributivo" in state.gluing_obstruction_failing_charts
        assert "U_ontology:Cohomología-Ontológica" in state.gluing_obstruction_failing_charts
        assert "U_pullback:Pullback-ZeroTrust" in state.gluing_obstruction_failing_charts

    def test_pullback_input_error_is_raised_for_malformed_omega(
        self, agent, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        _omega, x_domain, s_allowed = valid_projector_and_domain
        malformed_omega = np.array([[np.nan]])

        with pytest.raises(PullbackInputError):
            agent.execute_federated_governance(
                [0.1], d0, d1, x_domain, s_allowed, malformed_omega
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §5. INVARIANTES TRANSVERSALES: INMUTABILIDAD Y JERARQUÍA DE MIXINS          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestImmutableDataTransferObjects:
    """Certifica que los DTOs `frozen=True, slots=True` rechazan mutación."""

    def test_lattice_projection_data_rejects_attribute_mutation(self):
        dto = LatticeProjectionData(
            supremum_state=0.0,
            infimum_state=1.0,
            bottom_state=0.0,
            top_threshold=0.999999999,
            violation_count=0,
            critical_count=0,
            shannon_entropy_bits=0.0,
            max_entropy_bits=0.0,
            de_morgan_duality_residual=0.0,
            is_structurally_sound=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            dto.is_structurally_sound = False  # type: ignore[misc]

    def test_federated_governance_state_rejects_attribute_mutation(
        self, agent, filled_triangle_boundaries, valid_projector_and_domain
    ):
        d0, d1 = filled_triangle_boundaries
        omega, x_domain, s_allowed = valid_projector_and_domain

        state = agent.execute_federated_governance(
            [0.1], d0, d1, x_domain, s_allowed, omega
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_fully_compliant = False  # type: ignore[misc]

    def test_dataclasses_replace_produces_independent_copy(self):
        original = ToposPolicyPullbackData(
            pullback_residual=0.0,
            relative_pullback_residual=0.0,
            tolerance=1e-12,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            spectral_binary_residual=0.0,
            projector_trace=1.0,
            trace_rank_residual=0.0,
            allowed_fixed_residual=0.0,
            projector_rank=1,
            domain_rows=3,
            domain_columns=1,
            is_zero_trust_verified=True,
        )
        mutated_copy = dataclasses.replace(original, is_zero_trust_verified=False)

        assert original.is_zero_trust_verified is True
        assert mutated_copy.is_zero_trust_verified is False


class TestFunctorialMixinHierarchy:
    """Certifica el anidamiento funtorial Φ₃∘Φ₂∘Φ₁ vía `isinstance`."""

    def test_governance_agent_inherits_full_phase_chain(self, agent):
        assert isinstance(agent, Phase1_InformationTheoreticLatticeProjector)
        assert isinstance(agent, Phase2_EulerCharacteristicCohomologyAuditor)
        assert isinstance(agent, Phase3_SpectralToposPolicyFunctor)

    def test_phase3_instance_inherits_phase1_and_phase2(self, phase3):
        assert isinstance(phase3, Phase1_InformationTheoreticLatticeProjector)
        assert isinstance(phase3, Phase2_EulerCharacteristicCohomologyAuditor)

    def test_phase2_instance_does_not_inherit_phase3(self, phase2):
        assert isinstance(phase2, Phase1_InformationTheoreticLatticeProjector)
        assert not isinstance(phase2, Phase3_SpectralToposPolicyFunctor)

    def test_exception_hierarchy_roots_at_governance_agent_error(self):
        assert issubclass(LatticeInputError, GovernanceAgentError)
        assert issubclass(StructuralVetoMonad, GovernanceAgentError)
        assert issubclass(CohomologyInputError, GovernanceAgentError)
        assert issubclass(OntologicalParadoxVeto, GovernanceAgentError)
        assert issubclass(HodgeCrossValidationError, GovernanceAgentError)
        assert issubclass(PullbackInputError, GovernanceAgentError)
        assert issubclass(ZeroTrustViolationError, GovernanceAgentError)
        assert issubclass(SheafGluingObstructionError, GovernanceAgentError)

    def test_hodge_and_ontological_exceptions_are_siblings_not_parent_child(self):
        """
        Certifica la separación epistemológica deliberada: un fallo de
        validación cruzada de Hodge NO es un caso especial de paradoja
        ontológica ni viceversa — son ramas hermanas bajo la raíz común.
        """
        assert not issubclass(HodgeCrossValidationError, OntologicalParadoxVeto)
        assert not issubclass(OntologicalParadoxVeto, HodgeCrossValidationError)