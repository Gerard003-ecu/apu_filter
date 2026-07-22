# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de pruebas : test_app_agent.py                                        ║
║ Ruta              : tests/unit/agents/test_app_agent.py                      ║
║ SUT               : app/agents/app_agent.py (v3.0.0)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

FILOSOFÍA DE VERIFICACIÓN:
────────────────────────────────────────────────────────────────────────────────
La suite se organiza en espejo exacto de las 3 fases anidadas del SUT, más
tres bloques transversales:

  §1. Fase 1 — Fibración criptográfica + Impedancia Port-Hamiltoniana.
  §2. Fase 2 — Cierre categórico-topológico del poset DIKW (Warshall).
  §3. Fase 3 — Proyección cuántica-booleana MIC + cancelación de anomalía.
  §4. Integración end-to-end vía AppAgent.execute_gateway_governance.
  §5. Invariantes transversales: inmutabilidad de DTOs, jerarquía de mixins.
  §6. Pruebas combinatorias/defensivas de alto rigor (fuerza bruta, monkeypatch).

Cada método de prueba certifica UN invariante matemático o UNA rama de
decisión, nunca ambos a la vez (principio de granularidad).
"""

from __future__ import annotations

import dataclasses
import itertools
import uuid

import numpy as np
import pytest

from app.agents.app_agent import (
    _DIKW_IMMEDIATE_ADJACENCY,
    _DIKW_LEVEL_NAMES,
    AppAgent,
    AppAgentError,
    GatewayGovernanceState,
    GlobalAnomalyCancellationError,
    ImpedanceControlData,
    ImpedanceInputError,
    OrthogonalProjectionData,
    OrthogonalProjectionInputError,
    OrthogonalityViolationError,
    Phase1_SpectralFibrationImpedanceCertifier,
    Phase2_CategoricalDIKWClosureInstantiator,
    Phase3_QuantumBooleanMICProjectorSynthesizer,
    ResonanceCatastropheVeto,
    StateFibrationData,
    StateSymmetryBreakingError,
    ThermodynamicPassportData,
    TopologicalPassportError,
)


# ══════════════════════════════════════════════════════════════════════════════
# §0. FIXTURES CANÓNICAS (Objetos físicos y algebraicos de referencia)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def phase1() -> Phase1_SpectralFibrationImpedanceCertifier:
    """Instancia aislada de Fase 1 (stubs de Fase 2 activos)."""
    return Phase1_SpectralFibrationImpedanceCertifier()


@pytest.fixture
def phase2() -> Phase2_CategoricalDIKWClosureInstantiator:
    """Instancia de Fase 2 (implementación real de DIKW, stub de Fase 3 activo)."""
    return Phase2_CategoricalDIKWClosureInstantiator()


@pytest.fixture
def phase3() -> Phase3_QuantumBooleanMICProjectorSynthesizer:
    """Instancia de la cadena completa Φ₃∘Φ₂∘Φ₁ (todas las implementaciones reales)."""
    return Phase3_QuantumBooleanMICProjectorSynthesizer()


@pytest.fixture
def agent() -> AppAgent:
    """Instancia del orquestador supremo AppAgent."""
    return AppAgent()


@pytest.fixture
def valid_antisymmetric_J() -> np.ndarray:
    """J = -J^T exacta (tejido interconector conservativo, 3D)."""
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 2.0],
            [0.0, -2.0, 0.0],
        ]
    )


@pytest.fixture
def valid_psd_diagonally_dominant_R() -> np.ndarray:
    """
    R = R^T ⪰ 0, diagonalmente dominante (red disipativa tridiagonal, 3D).
    Autovalores analíticos: {2 - cos(kπ/4) : k=1,2,3} ⊂ (0, 3), todos > 0.
    """
    return np.array(
        [
            [2.0, -0.5, 0.0],
            [-0.5, 2.0, -0.5],
            [0.0, -0.5, 2.0],
        ]
    )


@pytest.fixture
def grad_H_vector() -> np.ndarray:
    """Gradiente de energía de referencia en R^3."""
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def canonical_projectors_3d():
    """
    Base canónica ortogonal de proyectores en R^3: P0, P1, P2 diagonales,
    mutuamente ortogonales, idempotentes, simétricos, con espectro {0,1}.
    """
    p0 = np.diag([1.0, 0.0, 0.0])
    p1 = np.diag([0.0, 1.0, 0.0])
    p2 = np.diag([0.0, 0.0, 1.0])
    return p0, p1, p2


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §1. FASE 1 — FIBRACIÓN CRIPTOGRÁFICA + IMPEDANCIA ACTIVA                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestHashLabelNormalization:
    """Certifica `_normalize_hash_label`: coerción de tipos y saneamiento textual."""

    def test_bytes_input_decoded_as_utf8(self, phase1):
        text, violations = phase1._normalize_hash_label(
            "h", b"abc123", normalize_case=False
        )
        assert text == "abc123"
        assert violations == ()

    def test_str_input_is_stripped_of_lateral_whitespace(self, phase1):
        text, violations = phase1._normalize_hash_label(
            "h", "  abc123  ", normalize_case=False
        )
        assert text == "abc123"
        assert violations == ()

    def test_case_normalization_lowercases_when_enabled(self, phase1):
        text, _ = phase1._normalize_hash_label("h", "ABC123", normalize_case=True)
        assert text == "abc123"

    def test_case_preserved_when_normalization_disabled(self, phase1):
        text, _ = phase1._normalize_hash_label("h", "ABC123", normalize_case=False)
        assert text == "ABC123"

    def test_invalid_utf8_bytes_are_flagged_as_violation(self, phase1):
        text, violations = phase1._normalize_hash_label(
            "h", b"\xff\xfe", normalize_case=False
        )
        assert text == ""
        assert len(violations) == 1
        assert "no se pudo decodificar" in violations[0]

    def test_non_str_non_bytes_type_is_flagged(self, phase1):
        text, violations = phase1._normalize_hash_label("h", 12345, normalize_case=False)
        assert text == ""
        assert "se esperaba str o bytes" in violations[0]


class TestHammingDistanceGroupTheory:
    """
    Certifica `_compute_hamming_distance_bits` como métrica del grupo
    abeliano (GF(2)^N, XOR).
    """

    def test_identical_strings_yield_zero_distance(self, phase1):
        assert phase1._compute_hamming_distance_bits("abcdef", "abcdef") == 0

    def test_empty_strings_yield_zero_distance(self, phase1):
        assert phase1._compute_hamming_distance_bits("", "") == 0

    def test_single_byte_xor_distance_matches_bit_count(self, phase1):
        # 'a'=0x61, 'e'=0x65 → XOR y conteo de bits difiere en popcount(0x04)=1
        expected = bin(0x61 ^ 0x65).count("1")
        assert phase1._compute_hamming_distance_bits("a", "e") == expected

    def test_length_mismatch_penalizes_eight_bits_per_missing_byte(self, phase1):
        # común: "ab" vs "ab" (0 bits) + 1 byte faltante * 8 bits de penalización
        distance = phase1._compute_hamming_distance_bits("abc", "ab")
        assert distance == 8

    def test_distance_is_symmetric_under_argument_swap(self, phase1):
        d_ab = phase1._compute_hamming_distance_bits("hello", "world")
        d_ba = phase1._compute_hamming_distance_bits("world", "hello")
        assert d_ab == d_ba


class TestStateFibrationIsomorphismCertification:
    """Certifica `_certify_state_fibration_isomorphism` (contrato de Fase 1)."""

    def test_matching_hashes_certify_symmetry(self, phase1):
        audit = phase1._certify_state_fibration_isomorphism("deadbeef", "deadbeef")
        assert audit.is_symmetric is True
        assert audit.hash_residual == 0.0
        assert audit.hamming_distance_bits == 0
        assert audit.constant_time_compared is True

    def test_mismatched_hashes_raise_by_default(self, phase1):
        with pytest.raises(StateSymmetryBreakingError):
            phase1._certify_state_fibration_isomorphism("deadbeef", "beefdead")

    def test_mismatched_hashes_do_not_raise_when_veto_disabled(self, phase1):
        audit = phase1._certify_state_fibration_isomorphism(
            "deadbeef", "beefdead", raise_on_veto=False
        )
        assert audit.is_symmetric is False
        assert audit.hash_residual == 1.0

    def test_empty_hash_pair_raises_when_nonempty_required(self, phase1):
        with pytest.raises(StateSymmetryBreakingError):
            phase1._certify_state_fibration_isomorphism("", "")

    def test_empty_hash_pair_certifies_symmetry_when_allowed(self, phase1):
        audit = phase1._certify_state_fibration_isomorphism(
            "", "", require_nonempty_hash=False
        )
        assert audit.is_symmetric is True

    def test_case_normalization_reconciles_differing_case_hashes(self, phase1):
        audit = phase1._certify_state_fibration_isomorphism(
            "DEADBEEF", "deadbeef", normalize_hash_case=True
        )
        assert audit.is_symmetric is True

    def test_case_sensitivity_is_default_behavior(self, phase1):
        with pytest.raises(StateSymmetryBreakingError):
            phase1._certify_state_fibration_isomorphism("DEADBEEF", "deadbeef")

    def test_bytes_input_is_supported_transparently(self, phase1):
        audit = phase1._certify_state_fibration_isomorphism(b"deadbeef", b"deadbeef")
        assert audit.is_symmetric is True


class TestFiniteNumericValidators:
    """
    Certifica `_as_finite_vector` / `_as_finite_square_matrix`, incluida la
    inyección de `exception_cls` (correspondencia Curry–Howard dominio↔error).
    """

    def test_1d_vector_passes_through_unchanged(self, phase1):
        v = phase1._as_finite_vector("x", [1.0, 2.0, 3.0])
        assert v.shape == (3,)

    def test_scalar_is_reshaped_to_length_one_vector(self, phase1):
        v = phase1._as_finite_vector("x", 5.0)
        assert v.shape == (1,)

    def test_column_matrix_is_flattened_to_vector(self, phase1):
        v = phase1._as_finite_vector("x", np.array([[1.0], [2.0]]))
        assert v.shape == (2,)

    def test_non_finite_vector_raises_default_exception_type(self, phase1):
        with pytest.raises(ImpedanceInputError):
            phase1._as_finite_vector("x", [1.0, np.nan])

    def test_non_column_2d_matrix_is_rejected_as_vector(self, phase1):
        with pytest.raises(ImpedanceInputError):
            phase1._as_finite_vector("x", np.ones((2, 2)))

    def test_exception_cls_injection_overrides_default_type(self, phase1):
        with pytest.raises(OrthogonalProjectionInputError):
            phase1._as_finite_vector(
                "x", [np.inf], exception_cls=OrthogonalProjectionInputError
            )

    def test_square_matrix_shape_mismatch_raises(self, phase1):
        with pytest.raises(ImpedanceInputError):
            phase1._as_finite_square_matrix("M", np.eye(2), dimension=3)

    def test_square_matrix_with_nan_raises(self, phase1):
        matrix = np.eye(2)
        matrix[0, 0] = np.nan
        with pytest.raises(ImpedanceInputError):
            phase1._as_finite_square_matrix("M", matrix, dimension=2)

    def test_non_2d_input_rejected_as_square_matrix(self, phase1):
        with pytest.raises(ImpedanceInputError):
            phase1._as_finite_square_matrix("M", np.array([1.0, 2.0]), dimension=2)


class TestOperatorToleranceScaling:
    """Certifica el modelo `tol = max(base, 100·n·ε·max(1,scale))`."""

    def test_tolerance_increases_monotonically_with_dimension(self, phase1):
        tol_small = phase1._operator_tolerance(2, 1.0)
        tol_large = phase1._operator_tolerance(200, 1.0)
        assert tol_large > tol_small

    def test_tolerance_respects_explicit_base_floor(self, phase1):
        tol = phase1._operator_tolerance(1, 0.0, base=1e-6)
        assert tol >= 1e-6


class TestSpectralAndSylvesterInertiaCertificates:
    """
    Certifica el teorema espectral de antisimetría y la Ley de Inercia de
    Sylvester (invariancia bajo congruencia).
    """

    def test_pure_antisymmetric_matrix_has_negligible_real_spectrum(
        self, phase1, valid_antisymmetric_J
    ):
        residual = phase1._spectral_antisymmetry_certificate(valid_antisymmetric_J)
        assert residual < 1e-9

    def test_symmetric_matrix_has_significant_real_spectrum(self, phase1):
        symmetric_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        residual = phase1._spectral_antisymmetry_certificate(symmetric_matrix)
        assert residual > 0.5

    def test_zero_dimensional_operator_has_zero_spectral_residual(self, phase1):
        assert phase1._spectral_antisymmetry_certificate(np.zeros((0, 0))) == 0.0

    def test_inertia_of_positive_definite_matrix_is_all_positive(self, phase1):
        r_matrix = np.eye(3) * 2.0
        inertia = phase1._sylvester_inertia_certificate(r_matrix, 1e-9)
        assert inertia == (3, 0, 0)

    def test_inertia_of_indefinite_matrix_reports_mixed_signature(self, phase1):
        r_matrix = np.diag([1.0, -1.0, 0.0])
        inertia = phase1._sylvester_inertia_certificate(r_matrix, 1e-9)
        assert inertia == (1, 1, 1)

    def test_inertia_is_invariant_under_congruence_transformation(self, phase1):
        """
        Ley de Inercia de Sylvester: (n+, n0, n-) es invariante bajo
        R ↦ AᵀRA para toda A no singular.
        """
        r_matrix = np.diag([2.0, 3.0, -1.0])
        congruence_map = np.array(
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]
        )
        assert abs(np.linalg.det(congruence_map)) > 1e-9  # no singular

        r_congruent = congruence_map.T @ r_matrix @ congruence_map

        inertia_original = phase1._sylvester_inertia_certificate(r_matrix, 1e-9)
        inertia_congruent = phase1._sylvester_inertia_certificate(r_congruent, 1e-6)

        assert inertia_original == inertia_congruent

    def test_zero_dimensional_matrix_has_trivial_inertia(self, phase1):
        assert phase1._sylvester_inertia_certificate(np.zeros((0, 0)), 1e-9) == (0, 0, 0)


class TestCircuitTopologyDissipationCertificate:
    """Certifica dominancia diagonal y conectividad algebraica análoga."""

    def test_diagonally_dominant_matrix_is_certified(
        self, phase1, valid_psd_diagonally_dominant_R
    ):
        is_dominant, margin, _connectivity = phase1._graph_theoretic_dissipation_topology(
            valid_psd_diagonally_dominant_R
        )
        assert is_dominant is True
        assert margin >= 0.0

    def test_non_diagonally_dominant_matrix_is_flagged(self, phase1):
        r_matrix = np.array([[1.0, 5.0], [5.0, 1.0]])
        is_dominant, margin, _connectivity = phase1._graph_theoretic_dissipation_topology(
            r_matrix
        )
        assert is_dominant is False
        assert margin < 0.0

    def test_zero_dimensional_matrix_is_trivially_dominant(self, phase1):
        is_dominant, margin, connectivity = phase1._graph_theoretic_dissipation_topology(
            np.zeros((0, 0))
        )
        assert is_dominant is True
        assert margin == 0.0
        assert connectivity == 0.0


class TestActiveImpedanceControlEnforcement:
    """Certifica `_enforce_active_impedance_control`: el corazón de Fase 1."""

    def test_zero_exogenous_work_is_passively_stable(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        audit = phase1._enforce_active_impedance_control(
            grad_H_vector, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, 0.0
        )
        assert audit.structural_valid is True
        assert audit.is_passively_stable is True
        assert audit.h_dot <= audit.tolerance

    def test_dissipation_term_matches_analytic_quadratic_form(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        # ∇H^T R_sym ∇H con grad=[1,1,1] y R tridiagonal(2,-0.5) analítico = 4.0
        audit = phase1._enforce_active_impedance_control(
            grad_H_vector, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, 0.0
        )
        assert audit.dissipation_term == pytest.approx(4.0, abs=1e-9)
        assert audit.h_dot == pytest.approx(-4.0, abs=1e-9)

    def test_excessive_exogenous_work_triggers_resonance_veto(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        with pytest.raises(ResonanceCatastropheVeto):
            phase1._enforce_active_impedance_control(
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                1000.0,
            )

    def test_veto_disabled_returns_unstable_audit_without_raising(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        audit = phase1._enforce_active_impedance_control(
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            1000.0,
            raise_on_veto=False,
        )
        assert audit.is_passively_stable is False

    def test_non_antisymmetric_j_raises_structural_veto(
        self, phase1, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        symmetric_j = np.eye(3)  # J = J^T viola J = -J^T
        with pytest.raises(ResonanceCatastropheVeto):
            phase1._enforce_active_impedance_control(
                grad_H_vector, symmetric_j, valid_psd_diagonally_dominant_R, 0.0
            )

    def test_non_symmetric_r_raises_structural_veto(
        self, phase1, valid_antisymmetric_J, grad_H_vector
    ):
        asymmetric_r = np.array(
            [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        with pytest.raises(ResonanceCatastropheVeto):
            phase1._enforce_active_impedance_control(
                grad_H_vector, valid_antisymmetric_J, asymmetric_r, 0.0
            )

    def test_negative_definite_r_raises_structural_veto(
        self, phase1, valid_antisymmetric_J, grad_H_vector
    ):
        negative_r = -np.eye(3)
        with pytest.raises(ResonanceCatastropheVeto):
            phase1._enforce_active_impedance_control(
                grad_H_vector, valid_antisymmetric_J, negative_r, 0.0
            )

    def test_dimension_mismatch_between_grad_and_matrices_raises_input_error(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R
    ):
        mismatched_grad = np.array([1.0, 2.0])  # dimensión 2 vs matrices 3x3
        with pytest.raises(ImpedanceInputError):
            phase1._enforce_active_impedance_control(
                mismatched_grad,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
            )

    def test_non_finite_exogenous_work_raises_input_error(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        with pytest.raises(ImpedanceInputError):
            phase1._enforce_active_impedance_control(
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                float("nan"),
            )

    def test_strict_negative_h_dot_certifies_asymptotic_stability(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        audit = phase1._enforce_active_impedance_control(
            grad_H_vector, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, 0.0
        )
        assert audit.is_strictly_dissipative is True
        assert audit.is_asymptotically_stable is True

    def test_zero_dimensional_system_is_trivially_stable(self, phase1):
        audit = phase1._enforce_active_impedance_control(
            np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), 0.0
        )
        assert audit.dimension == 0
        assert audit.is_passively_stable is True


class TestPhase1StubContractTowardsPhase2:
    """
    Certifica que Fase 1, aislada (sin mixin de Fase 2), preserva el
    contrato de "stub formal" — es decir, falla explícitamente en vez de
    producir un comportamiento silencioso incorrecto.
    """

    def test_passport_stub_raises_not_implemented_in_isolation(self, phase1):
        with pytest.raises(NotImplementedError):
            phase1._instantiate_thermodynamic_passport(True, False, False, False)

    def test_terminal_bridge_propagates_stub_failure(
        self, phase1, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        with pytest.raises(NotImplementedError):
            phase1._phase1_terminal_bridge_to_phase2(
                "h",
                "h",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                True,
                False,
                False,
                False,
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §2. FASE 2 — CIERRE CATEGÓRICO-TOPOLÓGICO DEL POSET DIKW                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestWarshallBooleanTransitiveClosure:
    """Certifica `_warshall_boolean_transitive_closure` sobre el semianillo booleano."""

    def test_dikw_adjacency_closes_to_expected_lower_triangular_poset(self, phase2):
        closure = phase2._warshall_boolean_transitive_closure(_DIKW_IMMEDIATE_ADJACENCY)
        expected = np.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(closure, expected)

    def test_closure_operator_is_idempotent(self, phase2):
        closure = phase2._warshall_boolean_transitive_closure(_DIKW_IMMEDIATE_ADJACENCY)
        double_closure = phase2._warshall_boolean_transitive_closure(closure)
        np.testing.assert_array_equal(closure, double_closure)

    def test_diagonal_only_graph_remains_unchanged_by_closure(self, phase2):
        adjacency = np.eye(3, dtype=bool)
        closure = phase2._warshall_boolean_transitive_closure(adjacency)
        np.testing.assert_array_equal(closure, adjacency)

    def test_fully_connected_graph_closes_to_all_true(self, phase2):
        adjacency = np.ones((3, 3), dtype=bool)
        closure = phase2._warshall_boolean_transitive_closure(adjacency)
        assert bool(np.all(closure))


class TestPosetAxiomsCertification:
    """Certifica los tres axiomas de orden parcial (categoría delgada)."""

    def test_dikw_closure_satisfies_all_three_poset_axioms(self, phase2):
        closure = phase2._warshall_boolean_transitive_closure(_DIKW_IMMEDIATE_ADJACENCY)
        reflexive, antisymmetric, transitive = phase2._certify_poset_axioms(closure)
        assert reflexive and antisymmetric and transitive

    def test_non_reflexive_relation_is_detected(self, phase2):
        broken = _DIKW_IMMEDIATE_ADJACENCY.copy()
        broken[0, 0] = False
        reflexive, _antisymmetric, _transitive = phase2._certify_poset_axioms(broken)
        assert reflexive is False

    def test_non_antisymmetric_relation_is_detected(self, phase2):
        mutually_related = np.array([[True, True], [True, True]])
        _reflexive, antisymmetric, _transitive = phase2._certify_poset_axioms(
            mutually_related
        )
        assert antisymmetric is False

    def test_non_transitive_relation_is_detected(self, phase2):
        # a→b, b→c, pero NO a→c
        chain_without_transitivity = np.array(
            [
                [True, True, False],
                [False, True, True],
                [False, False, True],
            ]
        )
        _reflexive, _antisymmetric, transitive = phase2._certify_poset_axioms(
            chain_without_transitivity
        )
        assert transitive is False


class TestThermodynamicPassportInstantiation:
    """Certifica `_instantiate_thermodynamic_passport` (implementación real, Fase 2)."""

    def test_physics_only_request_is_self_closed(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(True, False, False, False)
        assert audit.assigned_stratum_level == 0
        assert audit.is_transitively_closed is True
        assert audit.missing_lower_levels == ()

    def test_tactics_without_physics_raises_topological_error(self, phase2):
        with pytest.raises(TopologicalPassportError):
            phase2._instantiate_thermodynamic_passport(False, True, False, False)

    def test_tactics_with_physics_is_transitively_closed(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(True, True, False, False)
        assert audit.is_transitively_closed is True
        assert audit.assigned_stratum_level == 1

    def test_strategy_without_tactics_raises_topological_error(self, phase2):
        with pytest.raises(TopologicalPassportError):
            phase2._instantiate_thermodynamic_passport(True, False, True, False)

    def test_wisdom_alone_reports_all_three_missing_lower_levels(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(
            False, False, False, True, raise_on_veto=False
        )
        assert audit.assigned_stratum_level == 3
        assert audit.missing_lower_levels == (0, 1, 2)
        assert audit.is_transitively_closed is False

    def test_full_stratum_request_is_transitively_closed(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(True, True, True, True)
        assert audit.is_transitively_closed is True
        assert audit.assigned_stratum_level == 3

    def test_empty_request_raises_by_default(self, phase2):
        with pytest.raises(TopologicalPassportError):
            phase2._instantiate_thermodynamic_passport(False, False, False, False)

    def test_empty_request_is_tolerated_when_explicitly_allowed(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(
            False, False, False, False, allow_empty_request=True
        )
        assert audit.assigned_stratum_level == -1
        assert audit.is_transitively_closed is True

    def test_veto_disabled_returns_invalid_audit_without_raising(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(
            False, True, False, False, raise_on_veto=False
        )
        assert audit.is_transitively_closed is False
        assert audit.missing_lower_levels == (0,)

    def test_passport_always_reports_valid_underlying_poset_structure(self, phase2):
        audit = phase2._instantiate_thermodynamic_passport(True, False, False, False)
        assert audit.is_valid_poset_structure is True


class TestDIKWPassportExhaustiveCombinatorics:
    """
    Verificación exhaustiva por fuerza bruta (2⁴ = 16 combinaciones) de la
    consistencia entre `_instantiate_thermodynamic_passport` y la semántica
    directa de la clausura de Warshall: is_transitively_closed debe
    coincidir exactamente con requested_set ⊇ closure[assigned_level, :].
    """

    @pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=4)))
    def test_closure_semantics_hold_for_every_combination(self, phase2, flags):
        requested_levels = {i for i, active in enumerate(flags) if active}
        closure = phase2._warshall_boolean_transitive_closure(_DIKW_IMMEDIATE_ADJACENCY)

        if not requested_levels:
            audit = phase2._instantiate_thermodynamic_passport(
                *flags, allow_empty_request=True
            )
            assert audit.assigned_stratum_level == -1
            return

        assigned_level = max(requested_levels)
        required = {j for j in range(len(_DIKW_LEVEL_NAMES)) if closure[assigned_level, j]}
        expected_closed = required.issubset(requested_levels)

        audit = phase2._instantiate_thermodynamic_passport(*flags, raise_on_veto=False)

        assert audit.assigned_stratum_level == assigned_level
        assert audit.is_transitively_closed == expected_closed
        assert set(audit.missing_lower_levels) == (required - requested_levels)


class TestPhase2StubContractTowardsPhase3:
    """Certifica que Fase 2, aislada, falla explícitamente al alcanzar Fase 3."""

    def test_projection_stub_raises_not_implemented_in_isolation(self, phase2):
        with pytest.raises(NotImplementedError):
            phase2._project_intention_to_mic_basis(np.array([1.0]), np.eye(1), None)

    def test_terminal_bridge_propagates_stub_failure(
        self, phase2, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        with pytest.raises(NotImplementedError):
            phase2._phase2_terminal_bridge_to_phase3(
                "h",
                "h",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                True,
                False,
                False,
                False,
                np.array([1.0]),
                np.eye(1),
                None,
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §3. FASE 3 — PROYECCIÓN CUÁNTICA-BOOLEANA MIC + ANOMALÍA GLOBAL             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestProjectorSpectralBinaryTheorem:
    """Certifica σ(P) ⊆ {0,1} para proyectores hermíticos."""

    def test_valid_hermitian_projector_has_negligible_spectral_residual(self, phase3):
        projector = np.diag([1.0, 0.0, 0.0])
        residual = phase3._certify_projector_spectral_binary(projector, 1e-9)
        assert residual < 1e-9

    def test_non_projector_matrix_reports_expected_residual(self, phase3):
        # espectro {0.5, 0.5}: distancia mínima a {0,1} es exactamente 0.5
        non_projector = np.diag([0.5, 0.5])
        residual = phase3._certify_projector_spectral_binary(non_projector, 1e-9)
        assert residual == pytest.approx(0.5, abs=1e-9)

    def test_zero_dimensional_operator_has_zero_residual(self, phase3):
        assert phase3._certify_projector_spectral_binary(np.zeros((0, 0)), 1e-9) == 0.0


class TestBooleanOrthocomplementedLatticeAxioms:
    """Certifica los axiomas de meet/join del reticulado de von Neumann-Birkhoff."""

    def test_orthogonal_projector_pair_has_zero_meet_and_idempotent_join(self, phase3):
        p0 = np.diag([1.0, 0.0])
        p1 = np.diag([0.0, 1.0])
        meet_residual, join_residual = phase3._certify_boolean_lattice_pair(p0, p1)
        assert meet_residual == pytest.approx(0.0, abs=1e-12)
        assert join_residual == pytest.approx(0.0, abs=1e-9)

    def test_non_orthogonal_projector_pair_has_nonzero_meet(self, phase3):
        p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        # proyector ortogonal rank-1 sobre la dirección (1,1)/√2, idempotente y simétrico
        p_diagonal_direction = np.array([[0.5, 0.5], [0.5, 0.5]])
        meet_residual, _join_residual = phase3._certify_boolean_lattice_pair(
            p0, p_diagonal_direction
        )
        assert meet_residual > 0.0


class TestBornRuleProbabilityComputation:
    """Certifica p_i = ‖P_i x‖² / ‖x‖² ∈ [0,1]."""

    def test_full_alignment_yields_unit_probability(self, phase3):
        p = phase3._compute_born_rule_probability(projected_norm=2.0, x_norm=2.0)
        assert p == pytest.approx(1.0)

    def test_full_orthogonality_yields_zero_probability(self, phase3):
        p = phase3._compute_born_rule_probability(projected_norm=0.0, x_norm=2.0)
        assert p == pytest.approx(0.0)

    def test_probability_is_clamped_to_unit_interval(self, phase3):
        p = phase3._compute_born_rule_probability(projected_norm=3.0, x_norm=2.0)
        assert p == pytest.approx(1.0)

    def test_zero_norm_state_yields_zero_probability_without_division_error(self, phase3):
        p = phase3._compute_born_rule_probability(projected_norm=0.0, x_norm=0.0)
        assert p == 0.0


class TestOrthogonalProjectionToMICBasis:
    """Certifica `_project_intention_to_mic_basis`: el corazón de Fase 3."""

    def test_valid_orthogonal_projector_set_is_fully_certified(
        self, phase3, canonical_projectors_3d
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        audit = phase3._project_intention_to_mic_basis(intent, p0, [p1, p2])

        assert audit.is_mutually_orthogonal is True
        assert audit.idempotence_residual < 1e-9
        assert audit.symmetry_residual < 1e-9
        assert audit.spectral_binary_residual < 1e-9
        assert audit.boolean_meet_residual < 1e-9
        assert audit.boolean_join_idempotence_residual < 1e-9
        assert audit.born_rule_probability == pytest.approx(1.0)
        assert audit.projected_norm == pytest.approx(1.0)
        assert audit.projector_rank == 1

    def test_non_idempotent_projector_raises_orthogonality_violation(self, phase3):
        non_idempotent = np.array([[0.5, 0.0], [0.0, 0.5]])
        intent = np.array([1.0, 0.0])
        with pytest.raises(OrthogonalityViolationError):
            phase3._project_intention_to_mic_basis(intent, non_idempotent, None)

    def test_non_symmetric_but_idempotent_projector_raises_when_hermiticity_required(
        self, phase3
    ):
        # P² = P (idempotente) pero P ≠ P^T
        oblique_projector = np.array([[1.0, 1.0], [0.0, 0.0]])
        intent = np.array([1.0, 0.0])
        with pytest.raises(OrthogonalityViolationError):
            phase3._project_intention_to_mic_basis(
                intent, oblique_projector, None, require_hermitian_projectors=True
            )

    def test_symmetrization_side_effect_still_flags_genuine_oblique_projectors(
        self, phase3
    ):
        """
        Hallazgo de rigor documentado como especificación de facto: aunque
        `require_hermitian_projectors=False` omite el chequeo EXPLÍCITO de
        simetría, `_certify_projector_spectral_binary` simetriza
        internamente el operador antes de diagonalizar. Para un proyector
        oblicuo genuino (no ortogonal) con espectro algebraico exacto
        {0,1} — como `[[1,1],[0,0]]`, cuyo polinomio característico es
        λ² - λ — la matriz simetrizada (P+Pᵀ)/2 tiene autovalores
        ≈ {1.207, -0.207} ∉ {0,1}. Esto produce una violación espectral
        residual incluso con la verificación de hermiticidad desactivada.
        """
        oblique_projector = np.array([[1.0, 1.0], [0.0, 0.0]])
        intent = np.array([1.0, 0.0])
        with pytest.raises(OrthogonalityViolationError):
            phase3._project_intention_to_mic_basis(
                intent, oblique_projector, None, require_hermitian_projectors=False
            )

    def test_operator_level_non_orthogonality_raises_by_default(self, phase3):
        p0 = np.diag([1.0, 0.0])
        overlapping_projector = np.array([[0.5, 0.5], [0.5, 0.5]])
        intent = np.array([1.0, 1.0])
        with pytest.raises(OrthogonalityViolationError):
            phase3._project_intention_to_mic_basis(intent, p0, [overlapping_projector])

    def test_operator_level_check_can_be_disabled_when_state_is_trivially_orthogonal(
        self, phase3
    ):
        p0 = np.diag([1.0, 0.0])
        overlapping_projector = np.array([[0.5, 0.5], [0.5, 0.5]])
        # x = [1,-1] ∈ ker(P_overlap): P_overlap·x = [0,0] pese al solape de operador
        intent = np.array([1.0, -1.0])

        audit = phase3._project_intention_to_mic_basis(
            intent, p0, [overlapping_projector], check_operator_orthogonality=False
        )

        assert audit.is_mutually_orthogonal is True

    def test_state_level_violation_raises_even_with_operator_check_disabled(self, phase3):
        p0 = np.diag([1.0, 0.0])
        overlapping_projector = np.array([[0.5, 0.5], [0.5, 0.5]])
        intent = np.array([1.0, 1.0])  # produce superposición parásita no nula

        with pytest.raises(OrthogonalityViolationError):
            phase3._project_intention_to_mic_basis(
                intent, p0, [overlapping_projector], check_operator_orthogonality=False
            )

    def test_projector_rank_matches_expected_algebraic_rank(
        self, phase3, canonical_projectors_3d
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])
        audit = phase3._project_intention_to_mic_basis(intent, p0, [p1, p2])
        assert audit.projector_rank == 1

    def test_veto_disabled_returns_invalid_audit_without_raising(self, phase3):
        non_idempotent = np.array([[0.5, 0.0], [0.0, 0.5]])
        intent = np.array([1.0, 0.0])
        audit = phase3._project_intention_to_mic_basis(
            intent, non_idempotent, None, raise_on_veto=False
        )
        assert audit.is_mutually_orthogonal is False

    def test_non_finite_intent_vector_raises_projection_specific_input_error(self, phase3):
        with pytest.raises(OrthogonalProjectionInputError):
            phase3._project_intention_to_mic_basis([np.nan, 1.0], np.eye(2), None)

    def test_projector_dimension_mismatch_raises_projection_specific_input_error(
        self, phase3
    ):
        with pytest.raises(OrthogonalProjectionInputError):
            phase3._project_intention_to_mic_basis(np.array([1.0, 0.0]), np.eye(3), None)

    def test_non_sequence_other_projectors_raises_projection_specific_input_error(
        self, phase3
    ):
        with pytest.raises(OrthogonalProjectionInputError):
            phase3._project_intention_to_mic_basis(
                np.array([1.0, 0.0]), np.eye(2), other_projectors=42
            )


class TestGlobalAnomalyCancellation:
    """
    Certifica `_certify_global_anomaly_cancellation`: analogía formal al
    mecanismo de Green-Schwarz (cancelación de cargas discretas por sector).
    """

    @staticmethod
    def _consistent_audits():
        fibration = StateFibrationData(
            hash_residual=0.0,
            is_symmetric=True,
            hash_length_t0=8,
            hash_length_t=8,
            constant_time_compared=True,
            hamming_distance_bits=0,
            bit_length_reference=64,
        )
        impedance = ImpedanceControlData(
            dimension=1,
            h_dot=-1.0,
            dissipation_term=1.0,
            exogenous_work_gu=0.0,
            tolerance=1e-9,
            j_antisymmetry_residual=0.0,
            j_spectral_max_real_part=0.0,
            r_symmetry_residual=0.0,
            r_min_eigenvalue=1.0,
            r_inertia_positive=1,
            r_inertia_zero=0,
            r_inertia_negative=0,
            r_diagonally_dominant=True,
            r_algebraic_connectivity_analog=1.0,
            structural_valid=True,
            is_passively_stable=True,
            is_strictly_dissipative=True,
            is_asymptotically_stable=True,
        )
        passport = ThermodynamicPassportData(
            assigned_stratum_level=0,
            requested_levels=(0,),
            missing_lower_levels=(),
            is_transitively_closed=True,
            is_valid_poset_structure=True,
        )
        projection = OrthogonalProjectionData(
            intent_dimension=1,
            projector_rank=1,
            idempotence_residual=0.0,
            symmetry_residual=0.0,
            spectral_binary_residual=0.0,
            operator_orthogonality_residual=0.0,
            state_orthogonality_residual=0.0,
            boolean_meet_residual=0.0,
            boolean_join_idempotence_residual=0.0,
            born_rule_probability=1.0,
            projected_norm=1.0,
            tolerance=1e-9,
            is_mutually_orthogonal=True,
        )
        return fibration, impedance, passport, projection

    def test_all_consistent_sectors_yield_zero_anomaly_index(self, phase3):
        fibration, impedance, passport, projection = self._consistent_audits()
        index, failing = phase3._certify_global_anomaly_cancellation(
            fibration, impedance, passport, projection
        )
        assert index == 0
        assert failing == ()

    def test_fibration_failure_contributes_single_charge(self, phase3):
        fibration, impedance, passport, projection = self._consistent_audits()
        broken_fibration = dataclasses.replace(fibration, is_symmetric=False)
        index, failing = phase3._certify_global_anomaly_cancellation(
            broken_fibration, impedance, passport, projection
        )
        assert index == 1
        assert failing == ("Φ₁:Fibración-Impedancia",)

    def test_passport_failure_contributes_single_charge(self, phase3):
        fibration, impedance, passport, projection = self._consistent_audits()
        broken_passport = dataclasses.replace(passport, is_transitively_closed=False)
        index, failing = phase3._certify_global_anomaly_cancellation(
            fibration, impedance, broken_passport, projection
        )
        assert index == 1
        assert failing == ("Φ₂:Clausura-DIKW",)

    def test_projection_failure_contributes_single_charge(self, phase3):
        fibration, impedance, passport, projection = self._consistent_audits()
        broken_projection = dataclasses.replace(projection, is_mutually_orthogonal=False)
        index, failing = phase3._certify_global_anomaly_cancellation(
            fibration, impedance, passport, broken_projection
        )
        assert index == 1
        assert failing == ("Φ₃:Proyección-MIC",)

    def test_multiple_sector_failures_accumulate_additively(self, phase3):
        fibration, impedance, passport, projection = self._consistent_audits()
        broken_fibration = dataclasses.replace(fibration, is_symmetric=False)
        broken_passport = dataclasses.replace(passport, is_transitively_closed=False)
        broken_projection = dataclasses.replace(projection, is_mutually_orthogonal=False)

        index, failing = phase3._certify_global_anomaly_cancellation(
            broken_fibration, impedance, broken_passport, broken_projection
        )

        assert index == 3
        assert set(failing) == {
            "Φ₁:Fibración-Impedancia",
            "Φ₂:Clausura-DIKW",
            "Φ₃:Proyección-MIC",
        }


class TestPhase3TerminalSynthesis:
    """Certifica `_phase3_terminal_synthesis`: la síntesis final de gobernanza."""

    def test_full_valid_pipeline_yields_secure_gateway_state(
        self,
        phase3,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        state = phase3._phase3_terminal_synthesis(
            "deadbeef",
            "deadbeef",
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            0.0,
            True,
            False,
            False,
            False,
            intent,
            p0,
            [p1, p2],
        )

        assert isinstance(state, GatewayGovernanceState)
        assert state.is_gateway_secure is True
        assert state.anomaly_index == 0
        assert state.anomaly_failing_sectors == ()
        uuid.UUID(state.governance_id)  # valida formato UUID

    def test_synthesis_raises_generic_anomaly_error_when_subcheck_is_bypassed(
        self,
        phase3,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
        monkeypatch,
    ):
        """
        Ejercita la rama defensiva de `GlobalAnomalyCancellationError`:
        estructuralmente inalcanzable cuando `raise_on_veto` se propaga de
        forma uniforme, pero necesaria como red de seguridad ante un mixin
        futuro que retorne un audit inconsistente sin abortar.
        """
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        original_method = (
            Phase3_QuantumBooleanMICProjectorSynthesizer._project_intention_to_mic_basis
        )

        def _forged_inconsistent_projection(self, *args, **kwargs):
            kwargs["raise_on_veto"] = False
            audit = original_method(self, *args, **kwargs)
            return dataclasses.replace(audit, is_mutually_orthogonal=False)

        monkeypatch.setattr(
            Phase3_QuantumBooleanMICProjectorSynthesizer,
            "_project_intention_to_mic_basis",
            _forged_inconsistent_projection,
        )

        with pytest.raises(GlobalAnomalyCancellationError):
            phase3._phase3_terminal_synthesis(
                "deadbeef",
                "deadbeef",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                True,
                False,
                False,
                False,
                intent,
                p0,
                [p1, p2],
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §4. INTEGRACIÓN END-TO-END — APPAGENT.EXECUTE_GATEWAY_GOVERNANCE            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestAppAgentEndToEndGovernance:
    """Certifica la composición funtorial completa Z_Gateway = Φ₃∘Φ₂∘Φ₁."""

    def test_full_happy_path_certifies_secure_gateway(
        self,
        agent,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        state = agent.execute_gateway_governance(
            "deadbeef",
            "deadbeef",
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            0.0,
            True,
            False,
            False,
            False,
            intent,
            p0,
            [p1, p2],
        )

        assert state.is_gateway_secure is True
        assert state.anomaly_index == 0
        assert state.fibration_audit.is_symmetric is True
        assert state.impedance_audit.is_passively_stable is True
        assert state.passport_audit.is_transitively_closed is True
        assert state.projection_audit.is_mutually_orthogonal is True

    def test_hash_mismatch_vetoes_the_entire_gateway(
        self,
        agent,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        with pytest.raises(StateSymmetryBreakingError):
            agent.execute_gateway_governance(
                "aaaaaaaa",
                "bbbbbbbb",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                True,
                False,
                False,
                False,
                intent,
                p0,
                [p1, p2],
            )

    def test_resonance_catastrophe_vetoes_the_entire_gateway(
        self,
        agent,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ResonanceCatastropheVeto):
            agent.execute_gateway_governance(
                "deadbeef",
                "deadbeef",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                1.0e9,  # trabajo exógeno catastrófico
                True,
                False,
                False,
                False,
                intent,
                p0,
                [p1, p2],
            )

    def test_dikw_closure_violation_vetoes_the_entire_gateway(
        self,
        agent,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        with pytest.raises(TopologicalPassportError):
            agent.execute_gateway_governance(
                "deadbeef",
                "deadbeef",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                False,  # PHYSICS ausente
                True,  # TACTICS solicitado sin su base
                False,
                False,
                intent,
                p0,
                [p1, p2],
            )

    def test_orthogonality_violation_vetoes_the_entire_gateway(
        self, agent, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        non_idempotent_projector = np.diag([0.5, 0.5, 0.5])
        intent = np.array([1.0, 0.0, 0.0])

        with pytest.raises(OrthogonalityViolationError):
            agent.execute_gateway_governance(
                "deadbeef",
                "deadbeef",
                grad_H_vector,
                valid_antisymmetric_J,
                valid_psd_diagonally_dominant_R,
                0.0,
                True,
                False,
                False,
                False,
                intent,
                non_idempotent_projector,
                None,
            )

    def test_raise_on_veto_false_aggregates_all_sector_failures(
        self, agent, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        non_idempotent_projector = np.diag([0.5, 0.5, 0.5])
        intent = np.array([1.0, 0.0, 0.0])

        state = agent.execute_gateway_governance(
            "aaaaaaaa",  # rompe Φ1 (hash)
            "bbbbbbbb",
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            0.0,
            False,  # rompe Φ2 (DIKW): TACTICS sin PHYSICS
            True,
            False,
            False,
            intent,
            non_idempotent_projector,  # rompe Φ3 (proyección)
            None,
            raise_on_veto=False,
        )

        assert state.is_gateway_secure is False
        assert state.anomaly_index == 3
        assert set(state.anomaly_failing_sectors) == {
            "Φ₁:Fibración-Impedancia",
            "Φ₂:Clausura-DIKW",
            "Φ₃:Proyección-MIC",
        }

    def test_other_projectors_defaults_to_none_without_error(
        self, agent, valid_antisymmetric_J, valid_psd_diagonally_dominant_R, grad_H_vector
    ):
        p0 = np.diag([1.0, 0.0, 0.0])
        intent = np.array([1.0, 0.0, 0.0])

        state = agent.execute_gateway_governance(
            "deadbeef",
            "deadbeef",
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            0.0,
            True,
            False,
            False,
            False,
            intent,
            p0,
        )
        assert state.is_gateway_secure is True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ §5. INVARIANTES TRANSVERSALES: INMUTABILIDAD Y JERARQUÍA DE MIXINS          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TestImmutableDataTransferObjects:
    """Certifica que los DTOs `frozen=True, slots=True` rechazan mutación."""

    def test_state_fibration_data_rejects_attribute_mutation(self):
        dto = StateFibrationData(
            hash_residual=0.0,
            is_symmetric=True,
            hash_length_t0=1,
            hash_length_t=1,
            constant_time_compared=True,
            hamming_distance_bits=0,
            bit_length_reference=8,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            dto.is_symmetric = False  # type: ignore[misc]

    def test_gateway_governance_state_rejects_attribute_mutation(
        self,
        agent,
        valid_antisymmetric_J,
        valid_psd_diagonally_dominant_R,
        grad_H_vector,
        canonical_projectors_3d,
    ):
        p0, p1, p2 = canonical_projectors_3d
        intent = np.array([1.0, 0.0, 0.0])

        state = agent.execute_gateway_governance(
            "deadbeef",
            "deadbeef",
            grad_H_vector,
            valid_antisymmetric_J,
            valid_psd_diagonally_dominant_R,
            0.0,
            True,
            False,
            False,
            False,
            intent,
            p0,
            [p1, p2],
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_gateway_secure = False  # type: ignore[misc]

    def test_dataclasses_replace_produces_independent_copy(self):
        original = ThermodynamicPassportData(
            assigned_stratum_level=0,
            requested_levels=(0,),
            missing_lower_levels=(),
            is_transitively_closed=True,
            is_valid_poset_structure=True,
        )
        mutated_copy = dataclasses.replace(original, is_transitively_closed=False)

        assert original.is_transitively_closed is True
        assert mutated_copy.is_transitively_closed is False


class TestFunctorialMixinHierarchy:
    """Certifica el anidamiento funtorial Φ₃∘Φ₂∘Φ₁ vía `isinstance`."""

    def test_app_agent_inherits_full_phase_chain(self, agent):
        assert isinstance(agent, Phase1_SpectralFibrationImpedanceCertifier)
        assert isinstance(agent, Phase2_CategoricalDIKWClosureInstantiator)
        assert isinstance(agent, Phase3_QuantumBooleanMICProjectorSynthesizer)

    def test_phase3_instance_inherits_phase1_and_phase2(self, phase3):
        assert isinstance(phase3, Phase1_SpectralFibrationImpedanceCertifier)
        assert isinstance(phase3, Phase2_CategoricalDIKWClosureInstantiator)

    def test_phase2_instance_inherits_phase1_only(self, phase2):
        assert isinstance(phase2, Phase1_SpectralFibrationImpedanceCertifier)
        assert not isinstance(phase2, Phase3_QuantumBooleanMICProjectorSynthesizer)

    def test_exception_hierarchy_roots_at_app_agent_error(self):
        assert issubclass(StateSymmetryBreakingError, AppAgentError)
        assert issubclass(ImpedanceInputError, AppAgentError)
        assert issubclass(ResonanceCatastropheVeto, AppAgentError)
        assert issubclass(TopologicalPassportError, AppAgentError)
        assert issubclass(OrthogonalProjectionInputError, AppAgentError)
        assert issubclass(OrthogonalityViolationError, AppAgentError)
        assert issubclass(GlobalAnomalyCancellationError, AppAgentError)