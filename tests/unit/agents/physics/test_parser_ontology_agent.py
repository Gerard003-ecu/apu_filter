r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_parser_ontology_agent.py                                       ║
║ Ruta   : tests/unit/agents/physics/test_parser_ontology_agent.py             ║
║ Versión: 2.1.0-Rigorous-Spectral-Categorical-Homological                     ║
║ Objetivo: Validación granular y rigurosa del Endofuntor ParserOntologyAgent  ║
║           (Fases 1 → 2 → 3 anidadas)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de la suite (espejo de las fases del SUT):
  §T0  Constantes, DTOs e jerarquía de excepciones
  §T1  Fase 1 — Mecánica estadística y espectral del texto
  §T2  Fase 2 — Homeomorfismo categórico, homológico y espectral
  §T3  Fase 3 — Orquestador ParserOntologyAgent (composición funtorial)
  §T4  Integración de punta a punta y propiedades invariantes
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# ─── SUT ─────────────────────────────────────────────────────────────────────
from app.agents.physics.parser_ontology_agent import (
    InformationConstants,
    ParserOntologyError,
    ThermodynamicEntropyVeto,
    SpectralDegeneracyVeto,
    HomeomorphismViolationError,
    EmptyManifoldError,
    HomologicalInvariantError,
    TextThermodynamics,
    HomeomorphicValidation,
    Phase1_TextStatisticalMechanics,
    Phase2_HomeomorphicValidator,
    ParserOntologyAgent,
)

# Stubs de dependencias (si el import real falla en el SUT se usan los internos;
# aquí re-importamos para tipado y aserciones de payload).
try:
    from app.core.mic_algebra import CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    from app.agents.physics.parser_ontology_agent import (
        CategoricalState,  # type: ignore[attr-defined]
        TopologicalInvariantError,  # type: ignore[attr-defined]
        Stratum,  # type: ignore[attr-defined]
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def pure_alpha_text() -> str:
    """Texto de baja entropía: solo líneas alfabéticas (estado 0)."""
    return "\n".join(["AlphaLine"] * 8)


@pytest.fixture
def pure_numeric_text() -> str:
    """Texto de baja entropía: solo líneas numéricas (estado 2)."""
    return "\n".join(["3.14159", "42", "0", "100.0", "7"])


@pytest.fixture
def mixed_balanced_text() -> str:
    """
    Texto con las 4 clases representadas de forma equilibrada.
    Entropía cercana a H_max → candidato a veto si Φ es bajo.
    """
    return "\n".join(
        [
            "Alpha",
            "",
            "123.45",
            "mix3d!",
            "Beta",
            "   ",
            "999",
            "foo@bar",
        ]
    )


@pytest.fixture
def sequential_report_text() -> str:
    """
    Texto estructurado tipo reporte (líneas con índices naturales).
    Ideal para homeomorfismo posicional y con line_index.
    """
    return "\n".join(
        [
            "HEADER",
            "100",
            "200.5",
            "FOOTER",
            "note-1",
        ]
    )


@pytest.fixture
def empty_text() -> str:
    return ""


@pytest.fixture
def whitespace_only_text() -> str:
    return "\n   \n\t\n  "


@pytest.fixture
def mock_parser_success():
    """ReportParserCrudo que devuelve AST con line_index biyectivo."""

    def _factory(text: str) -> List[Dict[str, Any]]:
        lines = text.splitlines()
        return [
            {"line_index": i, "raw": ln, "parsed": True}
            for i, ln in enumerate(lines)
        ]

    return _factory


@pytest.fixture
def mock_parser_empty():
    def _factory(text: str) -> List[Dict[str, Any]]:
        return []

    return _factory


@pytest.fixture
def mock_parser_broken_indices():
    """AST con line_index que rompe la sobreyectividad esencial."""

    def _factory(text: str) -> List[Dict[str, Any]]:
        # Solo mapea a un subconjunto y añade un objeto fantasma
        return [
            {"line_index": 0, "raw": "a"},
            {"line_index": 99, "raw": "ghost"},  # 99 ∉ C_text
        ]

    return _factory


@pytest.fixture
def phase1() -> type:
    return Phase1_TextStatisticalMechanics


@pytest.fixture
def phase2() -> type:
    return Phase2_HomeomorphicValidator


@pytest.fixture
def agent_default() -> ParserOntologyAgent:
    return ParserOntologyAgent(
        work_function=InformationConstants.DEFAULT_WORK_FUNCTION
    )


@pytest.fixture
def agent_strict() -> ParserOntologyAgent:
    """Φ bajo para forzar vetos entrópicos en textos mixtos."""
    return ParserOntologyAgent(work_function=0.40)


# ══════════════════════════════════════════════════════════════════════════════
# §T0 — CONSTANTES, DTOs Y JERARQUÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════

class TestInformationConstants:
    """Invariantes numéricos y dimensionales de InformationConstants."""

    def test_states_count_is_four(self):
        assert InformationConstants.STATES_COUNT == 4

    def test_h_max_is_log2_of_states(self):
        assert InformationConstants.H_MAX == pytest.approx(math.log2(4))
        assert InformationConstants.H_MAX == pytest.approx(2.0)

    def test_default_work_function_in_unit_interval(self):
        phi = InformationConstants.DEFAULT_WORK_FUNCTION
        assert 0.0 < phi <= 1.0

    def test_epsilon_positive_and_small(self):
        assert 0.0 < InformationConstants.EPSILON < 1e-6

    def test_state_labels_cardinality(self):
        assert len(InformationConstants.STATE_LABELS) == InformationConstants.STATES_COUNT
        assert InformationConstants.STATE_LABELS == (
            "alpha", "empty", "numeric", "mixed"
        )

    def test_spectral_tolerances_non_negative(self):
        assert InformationConstants.MIN_SPECTRAL_GAP >= 0.0
        assert InformationConstants.SPECTRAL_ISOMORPHISM_TOL >= 0.0


class TestExceptionHierarchy:
    """Jerarquía de vetos físicos y topológicos."""

    def test_root_is_topological_invariant_error(self):
        assert issubclass(ParserOntologyError, TopologicalInvariantError)

    def test_thermodynamic_veto_is_parser_error(self):
        assert issubclass(ThermodynamicEntropyVeto, ParserOntologyError)

    def test_spectral_veto_is_parser_error(self):
        assert issubclass(SpectralDegeneracyVeto, ParserOntologyError)

    def test_homeomorphism_error_is_parser_error(self):
        assert issubclass(HomeomorphismViolationError, ParserOntologyError)

    def test_empty_manifold_is_parser_error(self):
        assert issubclass(EmptyManifoldError, ParserOntologyError)

    def test_homological_error_is_parser_error(self):
        assert issubclass(HomologicalInvariantError, ParserOntologyError)

    def test_exceptions_are_raisable_and_catchable(self):
        with pytest.raises(ParserOntologyError):
            raise ThermodynamicEntropyVeto("test")
        with pytest.raises(TopologicalInvariantError):
            raise EmptyManifoldError("test")


class TestDTOImmutability:
    """Los contratos entre fases son frozen dataclasses (slots)."""

    def test_text_thermodynamics_is_frozen(self, pure_alpha_text, phase1):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        assert isinstance(thermo, TextThermodynamics)
        with pytest.raises(Exception):
            # frozen → AttributeError o FrozenInstanceError
            thermo.shannon_entropy = 0.0  # type: ignore[misc]

    def test_text_thermodynamics_fields_present(self, pure_alpha_text, phase1):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        assert hasattr(thermo, "shannon_entropy")
        assert hasattr(thermo, "von_neumann_entropy")
        assert hasattr(thermo, "normalized_entropy")
        assert hasattr(thermo, "work_function")
        assert hasattr(thermo, "is_exergically_viable")
        assert hasattr(thermo, "state_distribution")
        assert hasattr(thermo, "exergy")
        assert hasattr(thermo, "markov_matrix")
        assert hasattr(thermo, "markov_spectrum")
        assert hasattr(thermo, "spectral_gap")
        assert hasattr(thermo, "state_sequence")

    def test_homeomorphic_validation_is_frozen(
        self, sequential_report_text, phase1, phase2, mock_parser_success
    ):
        thermo = phase1.evaluate_thermodynamic_manifold(
            sequential_report_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            hv = phase2.project_to_simplex(sequential_report_text, thermo)
        assert isinstance(hv, HomeomorphicValidation)
        with pytest.raises(Exception):
            hv.is_homeomorphic = False  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# §T1 — FASE 1: MECÁNICA ESTADÍSTICA Y ESPECTRAL DEL TEXTO
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1ClassifyLine:
    """Clasificación booleana pura de líneas → índices de estado."""

    def test_alpha(self, phase1):
        assert phase1._classify_line("HelloWorld") == 0
        assert phase1._classify_line("  ABC  ") == 0

    def test_empty(self, phase1):
        assert phase1._classify_line("") == 1
        assert phase1._classify_line("   ") == 1
        assert phase1._classify_line("\t\t") == 1

    def test_numeric(self, phase1):
        assert phase1._classify_line("42") == 2
        assert phase1._classify_line("3.14") == 2
        assert phase1._classify_line("0") == 2
        assert phase1._classify_line("100.0") == 2

    def test_numeric_rejects_multiple_dots(self, phase1):
        # "1.2.3" → replace first dot → "12.3" no es digit → mixed
        assert phase1._classify_line("1.2.3") == 3

    def test_mixed(self, phase1):
        assert phase1._classify_line("foo123") == 3
        assert phase1._classify_line("a@b") == 3
        assert phase1._classify_line("-3.14") == 3  # signo no permitido
        assert phase1._classify_line("1e5") == 3


class TestPhase1StateProbabilities:
    """Proyección al simplejo Δ³ y secuencia de estados."""

    def test_empty_raises_empty_manifold(self, phase1, empty_text):
        with pytest.raises(EmptyManifoldError):
            phase1._compute_state_probabilities(empty_text)

    def test_pure_alpha_distribution(self, phase1, pure_alpha_text):
        p, seq = phase1._compute_state_probabilities(pure_alpha_text)
        assert p.shape == (4,)
        assert_allclose(p.sum(), 1.0, atol=1e-12)
        assert p[0] == pytest.approx(1.0)
        assert_array_equal(seq, np.zeros(8, dtype=np.int64))

    def test_pure_numeric_distribution(self, phase1, pure_numeric_text):
        p, seq = phase1._compute_state_probabilities(pure_numeric_text)
        assert_allclose(p.sum(), 1.0, atol=1e-12)
        assert p[2] == pytest.approx(1.0)
        assert set(seq.tolist()) == {2}

    def test_mixed_balanced_support(self, phase1, mixed_balanced_text):
        p, seq = phase1._compute_state_probabilities(mixed_balanced_text)
        assert_allclose(p.sum(), 1.0, atol=1e-12)
        # Las 4 clases aparecen al menos una vez
        assert np.all(p > 0)
        assert set(seq.tolist()) == {0, 1, 2, 3}

    def test_simplex_non_negativity(self, phase1, mixed_balanced_text):
        p, _ = phase1._compute_state_probabilities(mixed_balanced_text)
        assert np.all(p >= 0.0)

    def test_whitespace_only_is_empty_state(self, phase1, whitespace_only_text):
        p, seq = phase1._compute_state_probabilities(whitespace_only_text)
        # Todas las líneas no vacías de contenido son empty tras strip
        assert p[1] == pytest.approx(1.0)
        assert set(seq.tolist()) == {1}


class TestPhase1Entropies:
    """Entropías de Shannon y von Neumann."""

    def test_dirac_has_zero_entropy(self, phase1):
        p = np.array([1.0, 0.0, 0.0, 0.0])
        h, s_vn = phase1._compute_entropies(p)
        assert h == pytest.approx(0.0)
        assert s_vn == pytest.approx(0.0)

    def test_uniform_has_max_entropy(self, phase1):
        p = np.ones(4) / 4.0
        h, s_vn = phase1._compute_entropies(p)
        assert h == pytest.approx(InformationConstants.H_MAX)
        assert s_vn == pytest.approx(h)

    def test_shannon_equals_von_neumann_classical(self, phase1, mixed_balanced_text):
        p, _ = phase1._compute_state_probabilities(mixed_balanced_text)
        h, s_vn = phase1._compute_entropies(p)
        assert h == pytest.approx(s_vn)

    def test_entropy_bounds(self, phase1, pure_alpha_text, mixed_balanced_text):
        p_pure, _ = phase1._compute_state_probabilities(pure_alpha_text)
        p_mix, _ = phase1._compute_state_probabilities(mixed_balanced_text)
        h_pure, _ = phase1._compute_entropies(p_pure)
        h_mix, _ = phase1._compute_entropies(p_mix)
        assert 0.0 <= h_pure <= InformationConstants.H_MAX
        assert 0.0 <= h_mix <= InformationConstants.H_MAX
        assert h_mix > h_pure  # más mezcla → más entropía


class TestPhase1MarkovOperator:
    """Operador de Markov, espectro y gap."""

    def test_markov_rows_stochastic(self, phase1, mixed_balanced_text):
        _, seq = phase1._compute_state_probabilities(mixed_balanced_text)
        M, spectrum, gap = phase1._build_markov_operator(seq)
        assert M.shape == (4, 4)
        assert_allclose(M.sum(axis=1), np.ones(4), atol=1e-10)

    def test_markov_entries_non_negative(self, phase1, sequential_report_text):
        _, seq = phase1._compute_state_probabilities(sequential_report_text)
        M, _, _ = phase1._build_markov_operator(seq)
        assert np.all(M >= -1e-15)

    def test_spectral_radius_at_most_one(self, phase1, pure_alpha_text):
        _, seq = phase1._compute_state_probabilities(pure_alpha_text)
        _, spectrum, gap = phase1._build_markov_operator(seq)
        moduli = np.abs(spectrum)
        assert moduli[0] <= 1.0 + 1e-8

    def test_constant_sequence_has_trivial_markov(self, phase1, pure_alpha_text):
        _, seq = phase1._compute_state_probabilities(pure_alpha_text)
        M, spectrum, gap = phase1._build_markov_operator(seq)
        # Todas las transiciones 0→0
        assert M[0, 0] == pytest.approx(1.0)
        # Gap de una matriz con único estado absorbente puede ser 0 o 1
        assert gap >= 0.0

    def test_single_line_markov_is_identity_like(self, phase1):
        text = "OnlyOne"
        _, seq = phase1._compute_state_probabilities(text)
        M, spectrum, gap = phase1._build_markov_operator(seq)
        # Sin transiciones: filas sin salidas → Dirac en sí mismas
        assert M.shape == (4, 4)
        assert_allclose(M.sum(axis=1), np.ones(4), atol=1e-10)


class TestPhase1EvaluateThermodynamicManifold:
    """Método terminal de la Fase 1 (puente formal hacia Fase 2)."""

    def test_success_low_entropy(self, phase1, pure_alpha_text):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.85
        )
        assert isinstance(thermo, TextThermodynamics)
        assert thermo.is_exergically_viable is True
        assert thermo.normalized_entropy <= 0.85
        assert thermo.exergy == pytest.approx(
            InformationConstants.H_MAX - thermo.shannon_entropy
        )
        assert thermo.shannon_entropy == pytest.approx(thermo.von_neumann_entropy)
        assert thermo.state_distribution.shape == (4,)
        assert thermo.markov_matrix.shape == (4, 4)
        assert thermo.spectral_gap >= 0.0

    def test_veto_high_entropy(self, phase1, mixed_balanced_text):
        # Φ muy bajo → veto casi seguro en texto equilibrado
        with pytest.raises(ThermodynamicEntropyVeto) as exc_info:
            phase1.evaluate_thermodynamic_manifold(
                mixed_balanced_text, work_function=0.10
            )
        assert "Función de Trabajo" in str(exc_info.value) or "Φ" in str(exc_info.value)

    def test_empty_raises(self, phase1, empty_text):
        with pytest.raises(EmptyManifoldError):
            phase1.evaluate_thermodynamic_manifold(empty_text, work_function=0.85)

    def test_normalized_entropy_in_unit_interval(self, phase1, pure_numeric_text):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_numeric_text, work_function=0.99
        )
        assert 0.0 <= thermo.normalized_entropy <= 1.0 + 1e-12

    def test_work_function_stored(self, phase1, pure_alpha_text):
        phi = 0.77
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=phi
        )
        assert thermo.work_function == pytest.approx(phi)

    def test_state_sequence_length_matches_lines(self, phase1, sequential_report_text):
        thermo = phase1.evaluate_thermodynamic_manifold(
            sequential_report_text, work_function=0.99
        )
        n_lines = len(sequential_report_text.splitlines())
        assert len(thermo.state_sequence) == n_lines

    def test_exergy_non_negative(self, phase1, pure_alpha_text):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        assert thermo.exergy >= -1e-12

    def test_boundary_phi_equals_h_norm_is_viable(self, phase1, pure_alpha_text):
        # Calcular H_norm y usar exactamente ese valor como Φ
        p, _ = phase1._compute_state_probabilities(pure_alpha_text)
        h, _ = phase1._compute_entropies(p)
        h_norm = h / InformationConstants.H_MAX
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=h_norm
        )
        assert thermo.is_exergically_viable is True
        assert thermo.normalized_entropy == pytest.approx(h_norm)


# ══════════════════════════════════════════════════════════════════════════════
# §T2 — FASE 2: HOMEOMORFISMO CATEGÓRICO, HOMOLÓGICO Y ESPECTRAL
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2SmallCategory:
    """Micro-framework categórico interno."""

    def test_add_object_and_morphism(self, phase2):
        cat = phase2._SmallCategory("T")
        cat.add_morphism(0, 1, "succ")
        assert 0 in cat.objects and 1 in cat.objects
        assert cat.hom_set(0, 1) is True
        assert cat.hom_set(0, 0) is True  # identidad
        assert cat.hom_set(1, 0) is False

    def test_transitive_closure(self, phase2):
        cat = phase2._SmallCategory("Path")
        cat.add_morphism(0, 1)
        cat.add_morphism(1, 2)
        assert cat.hom_set(0, 2) is True  # compuesto

    def test_adjacency_symmetric(self, phase2):
        cat = phase2._SmallCategory("G")
        cat.add_morphism(0, 1)
        cat.add_morphism(1, 2)
        A = cat.adjacency_matrix()
        assert A.shape == (3, 3)
        assert_allclose(A, A.T)

    def test_laplacian_row_sums_zero(self, phase2):
        cat = phase2._SmallCategory("G")
        cat.add_morphism(0, 1)
        cat.add_morphism(1, 2)
        L = cat.laplacian_matrix()
        assert_allclose(L.sum(axis=1), np.zeros(3), atol=1e-12)

    def test_generators_property(self, phase2):
        cat = phase2._SmallCategory("G")
        cat.add_morphism(0, 1, "a")
        gens = cat.generators
        assert (0, 1) in gens
        assert gens[(0, 1)] == "a"


class TestPhase2BuildCategories:
    """Construcción de C_text y C_parsed."""

    def test_text_category_path_graph(self, phase2, sequential_report_text):
        cat = phase2._build_text_category(sequential_report_text)
        n = len(sequential_report_text.splitlines())
        assert len(cat.objects) == n
        # Camino: i → i+1
        for i in range(n - 1):
            assert cat.hom_set(i, i + 1)

    def test_parsed_category_with_line_index(self, phase2, mock_parser_success):
        text = "a\nb\nc"
        data = mock_parser_success(text)
        cat = phase2._build_parsed_category(data)
        assert len(cat.objects) == 3
        assert cat.hom_set(0, 1)
        assert cat.hom_set(1, 2)

    def test_parsed_category_without_line_index(self, phase2):
        data = [{"raw": "x"}, {"raw": "y"}]
        cat = phase2._build_parsed_category(data)
        assert cat.objects == {0, 1}
        assert cat.hom_set(0, 1)


class TestPhase2InferFunctor:
    """Inferencia del mapeo de objetos F: Ob(C_text) → Ob(C_parsed)."""

    def test_line_index_mapping(self, phase2, sequential_report_text, mock_parser_success):
        text_cat = phase2._build_text_category(sequential_report_text)
        data = mock_parser_success(sequential_report_text)
        parsed_cat = phase2._build_parsed_category(data)
        mapping = phase2._infer_functor(text_cat, parsed_cat, data)
        for i in text_cat.objects_list:
            assert mapping[i] == i

    def test_positional_mapping_without_line_index(self, phase2):
        text = "a\nb"
        text_cat = phase2._build_text_category(text)
        data = [{"raw": "a"}, {"raw": "b"}]
        parsed_cat = phase2._build_parsed_category(data)
        mapping = phase2._infer_functor(text_cat, parsed_cat, data)
        assert mapping[0] == 0
        assert mapping[1] == 1


class TestPhase2BettiNumbers:
    """Homología simplicial H₀, H₁ del 1-complejo."""

    def test_path_graph_betti(self, phase2):
        cat = phase2._SmallCategory("Path")
        for i in range(5):
            cat.add_object(i)
            if i < 4:
                cat.add_morphism(i, i + 1)
        beta0, beta1 = phase2._compute_betti_numbers(cat)
        # Camino conexo, sin ciclos
        assert beta0 == 1
        assert beta1 == 0

    def test_cycle_graph_betti(self, phase2):
        cat = phase2._SmallCategory("C3")
        cat.add_morphism(0, 1)
        cat.add_morphism(1, 2)
        cat.add_morphism(2, 0)
        beta0, beta1 = phase2._compute_betti_numbers(cat)
        assert beta0 == 1
        assert beta1 == 1  # un ciclo

    def test_two_components(self, phase2):
        cat = phase2._SmallCategory("2comp")
        cat.add_morphism(0, 1)
        cat.add_object(2)
        cat.add_object(3)
        cat.add_morphism(2, 3)
        beta0, beta1 = phase2._compute_betti_numbers(cat)
        assert beta0 == 2
        assert beta1 == 0

    def test_empty_category(self, phase2):
        cat = phase2._SmallCategory("empty")
        beta0, beta1 = phase2._compute_betti_numbers(cat)
        assert beta0 == 0
        assert beta1 == 0

    def test_isolated_vertices(self, phase2):
        cat = phase2._SmallCategory("isol")
        cat.add_object(0)
        cat.add_object(1)
        cat.add_object(2)
        beta0, beta1 = phase2._compute_betti_numbers(cat)
        assert beta0 == 3
        assert beta1 == 0


class TestPhase2SpectralDistance:
    """Distancia L² entre espectros de grafos."""

    def test_identical_matrices_zero_distance(self, phase2):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        d = phase2._spectral_distance(A, A.copy())
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_different_dimensions_padded(self, phase2):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        B = np.array([[0.0]])
        d = phase2._spectral_distance(A, B)
        assert d >= 0.0

    def test_isospectral_path_vs_same(self, phase2):
        cat = phase2._SmallCategory("P")
        cat.add_morphism(0, 1)
        cat.add_morphism(1, 2)
        A = cat.adjacency_matrix()
        d = phase2._spectral_distance(A, A)
        assert d < InformationConstants.SPECTRAL_ISOMORPHISM_TOL


class TestPhase2CategoricalEquivalence:
    """Validación completa: funtor + homología + isospectralidad."""

    def test_equivalent_on_identity_parse(
        self, phase2, sequential_report_text, mock_parser_success
    ):
        data = mock_parser_success(sequential_report_text)
        valid, log, bt, bp, dA, dL = phase2._validate_categorical_equivalence(
            sequential_report_text, data
        )
        assert valid is True
        assert bt == bp
        assert dA < InformationConstants.SPECTRAL_ISOMORPHISM_TOL
        assert dL < InformationConstants.SPECTRAL_ISOMORPHISM_TOL
        assert "isomorfismo" in log.lower() or "equivalencia" in log.lower() or "verificado" in log.lower()

    def test_broken_indices_not_essentially_surjective(
        self, phase2, sequential_report_text, mock_parser_broken_indices
    ):
        data = mock_parser_broken_indices(sequential_report_text)
        valid, log, *_ = phase2._validate_categorical_equivalence(
            sequential_report_text, data
        )
        assert valid is False
        assert "sobreyectiv" in log.lower() or "no alcanzados" in log.lower() or "homolog" in log.lower() or "espectr" in log.lower()


class TestPhase2ProjectToSimplex:
    """Método principal de la Fase 2 (continuación de TextThermodynamics)."""

    def test_success_homeomorphic(
        self, phase1, phase2, sequential_report_text, mock_parser_success
    ):
        thermo = phase1.evaluate_thermodynamic_manifold(
            sequential_report_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            hv = phase2.project_to_simplex(sequential_report_text, thermo)

        assert isinstance(hv, HomeomorphicValidation)
        assert hv.is_homeomorphic is True
        assert hv.categorical_equivalence is True
        assert hv.parsed_simplexes == len(sequential_report_text.splitlines())
        assert hv.thermodynamics is thermo
        assert hv.betti_numbers_text == hv.betti_numbers_parsed
        assert hv.adjacency_spectra_distance < InformationConstants.SPECTRAL_ISOMORPHISM_TOL

    def test_empty_ast_raises(
        self, phase1, phase2, pure_alpha_text, mock_parser_empty
    ):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_empty
            with pytest.raises(HomeomorphismViolationError) as ei:
                phase2.project_to_simplex(pure_alpha_text, thermo)
        assert "vacío" in str(ei.value).lower() or "cero" in str(ei.value).lower() or "nulo" in str(ei.value).lower() or "empty" in str(ei.value).lower()

    def test_dfa_exception_wrapped(
        self, phase1, phase2, pure_alpha_text
    ):
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = RuntimeError("DFA crash")
            with pytest.raises(HomeomorphismViolationError) as ei:
                phase2.project_to_simplex(pure_alpha_text, thermo)
        assert "DFA" in str(ei.value) or "Autómata" in str(ei.value) or "variedad" in str(ei.value).lower()

    def test_non_viable_thermo_guard(self, phase2, pure_alpha_text, phase1):
        # Construir un TextThermodynamics artificialmente no viable
        thermo = phase1.evaluate_thermodynamic_manifold(
            pure_alpha_text, work_function=0.99
        )
        # Reconstruir con is_exergically_viable=False (frozen → object.__new__)
        bad = TextThermodynamics(
            shannon_entropy=thermo.shannon_entropy,
            von_neumann_entropy=thermo.von_neumann_entropy,
            normalized_entropy=0.99,
            work_function=0.10,
            is_exergically_viable=False,
            state_distribution=thermo.state_distribution,
            exergy=thermo.exergy,
            markov_matrix=thermo.markov_matrix,
            markov_spectrum=thermo.markov_spectrum,
            spectral_gap=thermo.spectral_gap,
            state_sequence=thermo.state_sequence,
        )
        with pytest.raises(ThermodynamicEntropyVeto):
            phase2.project_to_simplex(pure_alpha_text, bad)

    def test_broken_functor_raises(
        self, phase1, phase2, sequential_report_text, mock_parser_broken_indices
    ):
        thermo = phase1.evaluate_thermodynamic_manifold(
            sequential_report_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_broken_indices
            with pytest.raises(HomeomorphismViolationError):
                phase2.project_to_simplex(sequential_report_text, thermo)


# ══════════════════════════════════════════════════════════════════════════════
# §T3 — FASE 3: ORQUESTADOR PARSER ONTOLOGY AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestParserOntologyAgentInit:
    """Construcción y parámetros del endofuntor."""

    def test_default_work_function(self, agent_default):
        assert agent_default._work_function == pytest.approx(
            InformationConstants.DEFAULT_WORK_FUNCTION
        )

    def test_custom_work_function(self):
        agent = ParserOntologyAgent(work_function=0.55)
        assert agent._work_function == pytest.approx(0.55)

    def test_is_morphism_subclass(self):
        assert issubclass(ParserOntologyAgent, Phase2_HomeomorphicValidator)


class TestParserOntologyAgentCall:
    """Composición funtorial completa: text → CategoricalState."""

    def test_success_returns_categorical_state(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        assert isinstance(state, CategoricalState)
        assert state.stratum == Stratum.PHYSICS

    def test_payload_keys(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        required = {
            "parsed_ast",
            "simplex_count",
            "syntactic_entropy",
            "von_neumann_entropy",
            "exergy_bits",
            "state_distribution",
            "markov_spectrum",
            "spectral_gap",
            "betti_text",
            "betti_parsed",
            "adjacency_spectra_distance",
            "laplacian_spectra_distance",
            "categorical_log",
        }
        assert required.issubset(state.payload.keys())

    def test_context_keys(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        assert state.context.get("is_homeomorphic") is True
        assert state.context.get("categorical_equivalence") is True
        assert state.context.get("exergically_viable") is True
        assert "work_function_applied" in state.context

    def test_simplex_count_matches_lines(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        n = len(sequential_report_text.splitlines())
        assert state.payload["simplex_count"] == n
        assert len(state.payload["parsed_ast"]) == n

    def test_betti_agreement(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        assert state.payload["betti_text"] == state.payload["betti_parsed"]

    def test_type_error_on_non_string(self, agent_default):
        with pytest.raises(TypeError):
            agent_default(12345)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agent_default(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agent_default(b"bytes")  # type: ignore[arg-type]

    def test_empty_text_raises(self, agent_default, empty_text):
        with pytest.raises(EmptyManifoldError):
            agent_default(empty_text)

    def test_high_entropy_veto_with_strict_phi(
        self, agent_strict, mixed_balanced_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            with pytest.raises(ThermodynamicEntropyVeto):
                agent_strict(mixed_balanced_text)

    def test_entropy_fields_consistent(
        self, agent_default, pure_alpha_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(pure_alpha_text)

        h_norm = state.payload["syntactic_entropy"]
        s_vn = state.payload["von_neumann_entropy"]
        exergy = state.payload["exergy_bits"]
        assert 0.0 <= h_norm <= 1.0 + 1e-12
        # Para Dirac: H ≈ 0, exergía ≈ H_max
        assert s_vn == pytest.approx(h_norm * InformationConstants.H_MAX, abs=1e-9)
        assert exergy == pytest.approx(
            InformationConstants.H_MAX - s_vn, abs=1e-9
        )

    def test_state_distribution_is_probability(
        self, agent_default, pure_numeric_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(pure_numeric_text)

        dist = np.array(state.payload["state_distribution"])
        assert dist.shape == (4,)
        assert_allclose(dist.sum(), 1.0, atol=1e-10)
        assert np.all(dist >= -1e-15)

    def test_spectral_gap_non_negative(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        assert state.payload["spectral_gap"] >= 0.0

    def test_spectra_distances_near_zero_when_homeomorphic(
        self, agent_default, sequential_report_text, mock_parser_success
    ):
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent_default(sequential_report_text)

        assert state.payload["adjacency_spectra_distance"] < 1e-5
        assert state.payload["laplacian_spectra_distance"] < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# §T4 — INTEGRACIÓN, INVARIANTES Y PROPIEDADES
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndInvariants:
    """Propiedades que deben preservarse a través de la composición funtorial."""

    def test_phase_composition_identity_of_thermo(
        self, phase1, phase2, sequential_report_text, mock_parser_success
    ):
        """El TextThermodynamics inyectado en Fase 2 es el mismo objeto en HV."""
        thermo = phase1.evaluate_thermodynamic_manifold(
            sequential_report_text, work_function=0.99
        )
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            hv = phase2.project_to_simplex(sequential_report_text, thermo)
        assert hv.thermodynamics is thermo

    def test_agent_matches_manual_composition(
        self, sequential_report_text, mock_parser_success
    ):
        """
        ParserOntologyAgent(text) ≡
          CategoricalState ∘ project_to_simplex ∘ evaluate_thermodynamic_manifold
        """
        agent = ParserOntologyAgent(work_function=0.90)
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state_agent = agent(sequential_report_text)

            # Manual
            thermo = Phase1_TextStatisticalMechanics.evaluate_thermodynamic_manifold(
                sequential_report_text, 0.90
            )
            hv = Phase2_HomeomorphicValidator.project_to_simplex(
                sequential_report_text, thermo
            )

        assert state_agent.payload["simplex_count"] == hv.parsed_simplexes
        assert state_agent.payload["syntactic_entropy"] == pytest.approx(
            thermo.normalized_entropy
        )
        assert state_agent.payload["betti_text"] == list(hv.betti_numbers_text)

    def test_idempotent_classification(self, phase1):
        """Clasificar dos veces la misma línea es estable."""
        lines = ["ABC", "12.3", "", "x1!", "   "]
        for ln in lines:
            assert phase1._classify_line(ln) == phase1._classify_line(ln)

    def test_entropy_monotonicity_under_mixing(self, phase1):
        """
        Una distribución más mezclada no puede tener menor entropía de Shannon
        que una Dirac (propiedad fundamental).
        """
        p_dirac = np.array([1.0, 0.0, 0.0, 0.0])
        p_mix = np.array([0.4, 0.3, 0.2, 0.1])
        h_d, _ = phase1._compute_entropies(p_dirac)
        h_m, _ = phase1._compute_entropies(p_mix)
        assert h_m >= h_d - 1e-12

    def test_exports_all_public_symbols(self):
        """__all__ del módulo cubre los símbolos públicos esperados."""
        import app.agents.physics.parser_ontology_agent as mod

        expected = {
            "InformationConstants",
            "ParserOntologyError",
            "ThermodynamicEntropyVeto",
            "SpectralDegeneracyVeto",
            "HomeomorphismViolationError",
            "EmptyManifoldError",
            "HomologicalInvariantError",
            "TextThermodynamics",
            "HomeomorphicValidation",
            "Phase1_TextStatisticalMechanics",
            "Phase2_HomeomorphicValidator",
            "ParserOntologyAgent",
        }
        assert expected.issubset(set(mod.__all__))


class TestEdgeCasesAndRobustness:
    """Casos límite y robustez numérica."""

    def test_single_line_text(self, phase1, mock_parser_success):
        text = "SoloUnaLinea"
        thermo = phase1.evaluate_thermodynamic_manifold(text, work_function=0.99)
        assert thermo.normalized_entropy == pytest.approx(0.0)
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            hv = Phase2_HomeomorphicValidator.project_to_simplex(text, thermo)
        assert hv.parsed_simplexes == 1
        assert hv.betti_numbers_text[0] == 1  # un vértice → β₀=1

    def test_very_long_homogeneous_text(self, phase1):
        text = "\n".join(["X"] * 500)
        thermo = phase1.evaluate_thermodynamic_manifold(text, work_function=0.99)
        assert thermo.normalized_entropy == pytest.approx(0.0)
        assert len(thermo.state_sequence) == 500

    def test_unicode_alpha_classified_as_alpha(self, phase1):
        # isalpha() de Python acepta letras Unicode
        assert phase1._classify_line("Café") == 0 or phase1._classify_line("Café") in (0, 3)
        # "Café" es alpha en Python 3
        assert "Café".isalpha()
        assert phase1._classify_line("Café") == 0

    def test_numeric_with_leading_zeros(self, phase1):
        assert phase1._classify_line("007") == 2

    def test_markov_with_two_states_alternating(self, phase1):
        text = "\n".join(["AAA", "111", "BBB", "222", "CCC", "333"])
        _, seq = phase1._compute_state_probabilities(text)
        M, spectrum, gap = phase1._build_markov_operator(seq)
        assert_allclose(M.sum(axis=1), np.ones(4), atol=1e-10)
        # Transiciones 0→2 y 2→0 predominan
        assert M[0, 2] > 0
        assert M[2, 0] > 0

    def test_agent_work_function_propagated_to_context(
        self, sequential_report_text, mock_parser_success
    ):
        phi = 0.72
        agent = ParserOntologyAgent(work_function=phi)
        with patch(
            "app.agents.physics.parser_ontology_agent.ReportParserCrudo"
        ) as MockDFA:
            MockDFA.return_value.parse.side_effect = mock_parser_success
            state = agent(sequential_report_text)
        assert state.context["work_function_applied"] == pytest.approx(phi)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZACIÓN ADICIONAL (clasificación exhaustiva)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "line,expected",
    [
        ("Hello", 0),
        ("WORLD", 0),
        ("", 1),
        ("   ", 1),
        ("\t", 1),
        ("42", 2),
        ("0.5", 2),
        ("3.1415926535", 2),
        ("foo1", 3),
        ("1.2.3", 3),
        ("-1", 3),
        ("+2", 3),
        ("1e10", 3),
        ("a b", 3),
        ("###", 3),
        ("12a", 3),
    ],
)
def test_classify_line_parametrized(line: str, expected: int):
    assert Phase1_TextStatisticalMechanics._classify_line(line) == expected


@pytest.mark.parametrize(
    "n_alpha,n_empty,n_num,n_mix",
    [
        (10, 0, 0, 0),
        (0, 5, 0, 0),
        (0, 0, 7, 0),
        (2, 2, 2, 2),
        (5, 1, 1, 1),
    ],
)
def test_distribution_counts(n_alpha, n_empty, n_num, n_mix):
    parts = (
        ["Alpha"] * n_alpha
        + [""] * n_empty
        + ["99"] * n_num
        + ["x1!"] * n_mix
    )
    text = "\n".join(parts)
    if not parts:
        pytest.skip("empty")
    p, seq = Phase1_TextStatisticalMechanics._compute_state_probabilities(text)
    total = n_alpha + n_empty + n_num + n_mix
    expected = np.array(
        [n_alpha, n_empty, n_num, n_mix], dtype=np.float64
    ) / total
    assert_allclose(p, expected, atol=1e-12)
    assert len(seq) == total