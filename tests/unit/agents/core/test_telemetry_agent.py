# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Telemetry Agent (Suite de Validación de Propagación Causal)               ║
║  Ruta   : tests/unit/agents/core/test_telemetry_agent.py                                 ║
║  Versión: 2.0.0-Causal-Cohomology-Port-Hamiltonian-Strict-Nested                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ARQUITECTURA DE PRUEBAS (Composición Funtorial Φ₃ ∘ Φ₂ ∘ Φ₁):                           ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Este módulo de pruebas implementa una batería exhaustiva que valida la integridad       ║
║  topológica, cohomológica y termodinámica del endofuntor Z_Telemetry.                    ║
║                                                                                          ║
║  FASE 1 → Filtración de Clausura Transitiva (Topología Zero-Trust)                       ║
║           Valida: V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM                          ║
║                                                                                          ║
║  FASE 2 → Cohomología de Spans Causales (Invariancia de Grafo)                           ║
║           Valida: χ(K) = β₀ - β₁ = |V| - |E| y β₁ = 0 (anti-ciclos)                      ║
║                                                                                          ║
║  FASE 3 → Disipación Port-Hamiltoniana (Irreversibilidad Termodinámica)                  ║
║           Valida: P_diss = ∇Hᵀ R ∇H ≥ 0 y R ⪰ 0 (matriz PSD)                             ║
║                                                                                          ║
║  COBERTURA DE EXCEPCIONES TOPOLÓGICAS:                                                   ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  • TransitiveClosureViolation     → Violaciones de jerarquía DIKW                        ║
║  • CausalCohomologyError          → Ciclos homológicos (β₁ > 0)                          ║
║  • ThermodynamicReversibilityError → Energía negativa o matriz no PSD                    ║
║                                                                                          ║
║  EJECUCIÓN:                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  $ pytest tests/unit/agents/core/test_telemetry_agent.py -v --cov=app.agents.core        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════════════════════

import pytest
import numpy as np
import math
from typing import Any, Callable, Set
from numpy.typing import NDArray
from enum import Enum

# Importación del módulo bajo prueba
from app.agents.core.telemetry_agent import (
    # Excepciones Topológicas y Causales
    TelemetryAgentError,
    TransitiveClosureViolation,
    CausalCohomologyError,
    ThermodynamicReversibilityError,
    # Enumeración de Estratos DIKW
    Stratum,
    # Estructuras Inmutables (DTOs)
    FiltrationAuditData,
    CausalCohomologyData,
    ThermodynamicDissipationData,
    Phase1CausalBridge,
    Phase2CohomologyBridge,
    CausalPropagationState,
    # Fases Anidadas
    Phase1_TransitiveClosureEnforcer,
    Phase2_CausalCohomologyAuditor,
    Phase3_PortHamiltonianVerifier,
    # Orquestador Supremo
    TelemetryAgent,
    # Constantes Físico-Matemáticas
    _MACHINE_EPSILON,
    _MAX_CAUSAL_DEPTH,
    _DISSIPATION_NEGATIVE_TOLERANCE,
    _PSD_EIGENVALUE_TOLERANCE,
    _SYMMETRY_WARNING_RATIO,
)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §A. FIXTURES Y UTILITARIOS DE PRUEBA (Infraestructura Categórica)
# ═══════════════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def telemetry_agent() -> TelemetryAgent:
    """
    Fixture: Instancia del Orquestador Supremo TelemetryAgent.
    Retorna el endofuntor completo para pruebas de integración.
    """
    return TelemetryAgent()


@pytest.fixture
def phase1_enforcer() -> Phase1_TransitiveClosureEnforcer:
    """
    Fixture: Instancia de Phase1_TransitiveClosureEnforcer.
    Para pruebas unitarias de la Fase 1.
    """
    return Phase1_TransitiveClosureEnforcer()


@pytest.fixture
def phase2_auditor() -> Phase2_CausalCohomologyAuditor:
    """
    Fixture: Instancia de Phase2_CausalCohomologyAuditor.
    Para pruebas unitarias de la Fase 2.
    """
    return Phase2_CausalCohomologyAuditor()


@pytest.fixture
def phase3_verifier() -> Phase3_PortHamiltonianVerifier:
    """
    Fixture: Instancia de Phase3_PortHamiltonianVerifier.
    Para pruebas unitarias de la Fase 3.
    """
    return Phase3_PortHamiltonianVerifier()


@pytest.fixture
def valid_strata_complete() -> Set[Stratum]:
    """
    Fixture: Conjunto completo de estratos DIKW válidos.
    Cumple la clausura transitiva: PHYSICS ⊂ TACTICS ⊂ STRATEGY ⊂ WISDOM
    """
    return {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM}


@pytest.fixture
def valid_strata_partial() -> Set[Stratum]:
    """
    Fixture: Conjunto parcial válido de estratos (solo PHYSICS).
    """
    return {Stratum.PHYSICS}


@pytest.fixture
def valid_grad_H() -> NDArray[np.float64]:
    """
    Fixture: Vector gradiente Hamiltoniano válido.
    """
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def valid_R_matrix() -> NDArray[np.float64]:
    """
    Fixture: Matriz de disipación R válida (simétrica PSD).
    """
    # Matriz simétrica definida positiva
    A = np.array([[2.0, 1.0, 0.0],
                  [1.0, 2.0, 1.0],
                  [0.0, 1.0, 2.0]], dtype=np.float64)
    return A @ A.T  # Garantiza PSD


@pytest.fixture
def valid_causal_graph_params() -> tuple[int, int, int]:
    """
    Fixture: Parámetros válidos de grafo causal (árbol).
    β₁ = 0 (sin ciclos)
    """
    total_spans = 5  # |V|
    causal_edges = 4  # |E| = |V| - 1 para árbol
    connected_components = 1  # β₀
    return total_spans, causal_edges, connected_components


@pytest.fixture
def identity_matrix_3x3() -> NDArray[np.float64]:
    """
    Fixture: Matriz identidad 3x3 (PSD trivial).
    """
    return np.eye(3, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 1: FILTRACIÓN DE CLAUSURA TRANSITIVA
#   Valida: V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase1_TransitiveClosureEnforcer:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 1: FILTRACIÓN DE CLAUSURA TRANSITIVA                                            ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la topología Zero-Trust de la jerarquía DIKW.           ║
    ║  Cada método prueba un axioma específico del §1 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.1. Pruebas de Validación de Enteros (Método: _as_nonnegative_int)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_as_nonnegative_int_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Entero no negativo válido.
        VALIDA: Conversión correcta a int.
        """
        result = phase1_enforcer._as_nonnegative_int("test_int", 42)
        assert isinstance(result, int)
        assert result == 42
        assert result >= 0

    def test_as_nonnegative_int_zero(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Cero es válido.
        VALIDA: Frontera inferior aceptada.
        """
        result = phase1_enforcer._as_nonnegative_int("test_zero", 0)
        assert result == 0

    def test_as_nonnegative_int_from_np_int(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Entero de NumPy convertido correctamente.
        VALIDA: Compatibilidad con tipos NumPy.
        """
        result = phase1_enforcer._as_nonnegative_int("test_np_int", np.int64(100))
        assert isinstance(result, int)
        assert result == 100

    def test_as_nonnegative_int_negative_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Entero negativo lanza TelemetryAgentError.
        VALIDA: Restricción de no negatividad.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_nonnegative_int("test_negative", -1)
        assert "no negativo" in str(exc_info.value).lower()

    def test_as_nonnegative_int_bool_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Booleano lanza TelemetryAgentError.
        VALIDA: Rechazo de tipos booleanos.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_nonnegative_int("test_bool", True)
        assert "booleano" in str(exc_info.value).lower()

    def test_as_nonnegative_int_np_bool_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Booleano de NumPy lanza TelemetryAgentError.
        VALIDA: Rechazo de np.bool_.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_nonnegative_int("test_np_bool", np.bool_(True))
        assert "booleano" in str(exc_info.value).lower()

    def test_as_nonnegative_int_string_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: String no convertible lanza TelemetryAgentError.
        VALIDA: Validación de tipo.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_nonnegative_int("test_string", "not_a_number")
        assert "entero" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.2. Pruebas de Validación de Vectores (Método: _as_finite_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_as_finite_vector_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Vector finito válido.
        VALIDA: Conversión a NDArray[np.float64] 1-D.
        """
        result = phase1_enforcer._as_finite_vector("test_vector", valid_grad_H)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.ndim == 1
        assert result.size == 3
        assert np.all(np.isfinite(result))

    def test_as_finite_vector_from_list(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Lista convertida a vector.
        VALIDA: Flexibilidad de entrada.
        """
        result = phase1_enforcer._as_finite_vector("test_list", [1.0, 2.0, 3.0])
        assert result.ndim == 1
        assert result.size == 3

    def test_as_finite_vector_empty_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Vector vacío lanza TelemetryAgentError.
        VALIDA: No-degeneración del espacio vectorial.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_vector("test_empty", np.array([], dtype=np.float64))
        assert "vacío" in str(exc_info.value).lower()

    def test_as_finite_vector_2d_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Matriz 2D lanza TelemetryAgentError.
        VALIDA: Exigencia de vector 1-D.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_vector("test_2d", np.array([[1.0], [2.0]]))
        assert "1-d" in str(exc_info.value).lower()

    def test_as_finite_vector_nan_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Vector con NaN lanza TelemetryAgentError.
        VALIDA: Finitud absoluta de componentes.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_vector("test_nan", np.array([1.0, np.nan, 3.0]))
        assert "nan" in str(exc_info.value).lower()

    def test_as_finite_vector_inf_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Vector con infinito lanza TelemetryAgentError.
        VALIDA: Finitud absoluta de componentes.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_vector("test_inf", np.array([1.0, np.inf, 3.0]))
        assert "infinit" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.3. Pruebas de Validación de Matrices (Método: _as_finite_square_matrix)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_as_finite_square_matrix_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Matriz cuadrada finita válida.
        VALIDA: Conversión a NDArray[np.float64] 2-D cuadrada.
        """
        result = phase1_enforcer._as_finite_square_matrix("test_matrix", valid_R_matrix)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.ndim == 2
        assert result.shape[0] == result.shape[1]
        assert np.all(np.isfinite(result))

    def test_as_finite_square_matrix_from_list(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Lista de listas convertida a matriz.
        VALIDA: Flexibilidad de entrada.
        """
        result = phase1_enforcer._as_finite_square_matrix(
            "test_list_matrix",
            [[1.0, 0.0], [0.0, 1.0]]
        )
        assert result.shape == (2, 2)

    def test_as_finite_square_matrix_non_square_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Matriz no cuadrada lanza TelemetryAgentError.
        VALIDA: Exigencia de cuadratura.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_square_matrix(
                "test_rect",
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
        assert "cuadrada" in str(exc_info.value).lower()

    def test_as_finite_square_matrix_empty_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Matriz vacía lanza TelemetryAgentError.
        VALIDA: No-degeneración matricial.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_square_matrix("test_empty", np.array([]).reshape(0, 0))
        assert "vacía" in str(exc_info.value).lower()

    def test_as_finite_square_matrix_nan_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Matriz con NaN lanza TelemetryAgentError.
        VALIDA: Finitud absoluta de entradas.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._as_finite_square_matrix(
                "test_nan",
                np.array([[1.0, np.nan], [0.0, 1.0]])
            )
        assert "nan" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.4. Pruebas de Normas Numéricas (Métodos: _safe_l2_norm, _safe_fro_norm)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_safe_l2_norm_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Norma L2 calculada correctamente.
        VALIDA: ||v||₂ = sqrt(Σ vᵢ²)
        """
        norm = phase1_enforcer._safe_l2_norm(valid_grad_H)
        expected = np.sqrt(1.0**2 + 2.0**2 + 3.0**2)
        assert isinstance(norm, float)
        assert np.isclose(norm, expected)
        assert np.isfinite(norm)

    def test_safe_l2_norm_zero_vector(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Norma L2 de vector cero.
        VALIDA: Caso degenerado.
        """
        norm = phase1_enforcer._safe_l2_norm(np.array([0.0, 0.0], dtype=np.float64))
        assert norm == 0.0

    def test_safe_fro_norm_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Norma de Frobenius calculada correctamente.
        VALIDA: ||M||_F = sqrt(Σ |mᵢⱼ|²)
        """
        norm = phase1_enforcer._safe_fro_norm(valid_R_matrix)
        expected = np.sqrt(np.sum(valid_R_matrix ** 2))
        assert isinstance(norm, float)
        assert np.isclose(norm, expected)
        assert np.isfinite(norm)

    def test_safe_fro_norm_zero_matrix(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Norma de Frobenius de matriz cero.
        VALIDA: Caso degenerado.
        """
        norm = phase1_enforcer._safe_fro_norm(np.zeros((3, 3), dtype=np.float64))
        assert norm == 0.0

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.5. Pruebas de Forma Cuadrática (Método: _safe_quadratic_form)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_safe_quadratic_form_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Forma cuadrática xᵀAx calculada correctamente.
        VALIDA: Producto escalar ponderado por matriz.
        """
        result = phase1_enforcer._safe_quadratic_form(valid_grad_H, valid_R_matrix)
        expected = float(valid_grad_H.T @ valid_R_matrix @ valid_grad_H)
        assert isinstance(result, float)
        assert np.isclose(result, expected)
        assert np.isfinite(result)

    def test_safe_quadratic_form_zero_vector(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Forma cuadrática con vector cero.
        VALIDA: Resultado debe ser cero.
        """
        zero_vec = np.zeros(3, dtype=np.float64)
        result = phase1_enforcer._safe_quadratic_form(zero_vec, valid_R_matrix)
        assert result == 0.0

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.6. Pruebas de Validación de Estratos (Método: _validate_completed_strata)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_validate_completed_strata_valid_set(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
    ) -> None:
        """
        PRUEBA: Conjunto de estratos válido convertido a frozenset.
        VALIDA: Normalización de entrada.
        """
        result = phase1_enforcer._validate_completed_strata(valid_strata_complete)
        assert isinstance(result, frozenset)
        assert len(result) == 4
        assert Stratum.PHYSICS in result
        assert Stratum.WISDOM in result

    def test_validate_completed_strata_from_list(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Lista de estratos convertida correctamente.
        VALIDA: Flexibilidad de entrada.
        """
        strata_list = [Stratum.PHYSICS, Stratum.TACTICS]
        result = phase1_enforcer._validate_completed_strata(strata_list)
        assert isinstance(result, frozenset)
        assert len(result) == 2

    def test_validate_completed_strata_from_int(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Enteros convertidos a Stratum.
        VALIDA: Coerción de tipos.
        """
        strata_ints = [0, 1]  # PHYSICS=0, TACTICS=1
        result = phase1_enforcer._validate_completed_strata(strata_ints)
        assert Stratum.PHYSICS in result
        assert Stratum.TACTICS in result

    def test_validate_completed_strata_none_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: None lanza TransitiveClosureViolation.
        VALIDA: §1. Filtración de Clausura Transitiva.
        """
        with pytest.raises(TransitiveClosureViolation) as exc_info:
            phase1_enforcer._validate_completed_strata(None)
        assert "none" in str(exc_info.value).lower()

    def test_validate_completed_strata_invalid_item_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: Item inválido en conjunto lanza TransitiveClosureViolation.
        VALIDA: Integridad de elementos.
        """
        with pytest.raises(TransitiveClosureViolation) as exc_info:
            phase1_enforcer._validate_completed_strata({Stratum.PHYSICS, "invalid"})
        assert "estrato inválido" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.7. Pruebas de Clausura Transitiva (Método: _enforce_transitive_closure_filtration)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_enforce_transitive_closure_filtration_complete(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
    ) -> None:
        """
        PRUEBA: Clausura transitiva con todos los estratos.
        VALIDA: §1. V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
        """
        result = phase1_enforcer._enforce_transitive_closure_filtration(valid_strata_complete)
        assert isinstance(result, FiltrationAuditData)
        assert result.is_filtration_valid is True
        assert len(result.active_strata) == 4

    def test_enforce_transitive_closure_filtration_physics_only(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_partial: Set[Stratum],
    ) -> None:
        """
        PRUEBA: Solo PHYSICS es válido.
        VALIDA: Estrato base sin dependencias.
        """
        result = phase1_enforcer._enforce_transitive_closure_filtration(valid_strata_partial)
        assert result.is_filtration_valid is True
        assert Stratum.PHYSICS in result.active_strata

    def test_enforce_transitive_closure_filtration_wisdom_without_physics_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: WISDOM sin PHYSICS lanza TransitiveClosureViolation.
        VALIDA: §1. Jerarquía DIKW estricta.
        """
        with pytest.raises(TransitiveClosureViolation) as exc_info:
            phase1_enforcer._enforce_transitive_closure_filtration({Stratum.WISDOM})
        assert "physics" in str(exc_info.value).lower()

    def test_enforce_transitive_closure_filtration_strategy_without_tactics_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: STRATEGY sin TACTICS lanza TransitiveClosureViolation.
        VALIDA: Dependencia transitiva.
        """
        with pytest.raises(TransitiveClosureViolation) as exc_info:
            phase1_enforcer._enforce_transitive_closure_filtration(
                {Stratum.PHYSICS, Stratum.STRATEGY}
            )
        assert "tactics" in str(exc_info.value).lower()

    def test_enforce_transitive_closure_filtration_tactics_without_physics_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
    ) -> None:
        """
        PRUEBA: TACTICS sin PHYSICS lanza TransitiveClosureViolation.
        VALIDA: Base termodinámica requerida.
        """
        with pytest.raises(TransitiveClosureViolation) as exc_info:
            phase1_enforcer._enforce_transitive_closure_filtration({Stratum.TACTICS})
        assert "physics" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.8. Pruebas de Handoff Fase 1 → Fase 2 (Método: _complete_phase1_causal_filtration)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_complete_phase1_causal_filtration_valid(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 1 a Fase 2.
        VALIDA: Continuidad funtorial Φ₁ → Φ₂.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        bridge = phase1_enforcer._complete_phase1_causal_filtration(
            completed_strata=valid_strata_complete,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
            connected_components=connected_components,
        )
        assert isinstance(bridge, Phase1CausalBridge)
        assert isinstance(bridge.filtration_audit, FiltrationAuditData)
        assert bridge.total_spans == total_spans
        assert bridge.causal_edges == causal_edges
        assert bridge.connected_components == connected_components
        assert np.array_equal(bridge.grad_H, valid_grad_H)
        assert np.array_equal(bridge.R_matrix, valid_R_matrix)

    def test_phase1_causal_bridge_immutability(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Phase1CausalBridge es inmutable.
        VALIDA: Integridad del artefacto de handoff.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        bridge = phase1_enforcer._complete_phase1_causal_filtration(
            completed_strata=valid_strata_complete,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
            connected_components=connected_components,
        )
        with pytest.raises((AttributeError, TypeError)):
            bridge.total_spans = 999  # type: ignore

    def test_complete_phase1_causal_filtration_dimension_mismatch_raises(
        self,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Dimensión incompatible entre grad_H y R_matrix lanza excepción.
        VALIDA: Consistencia dimensional.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        grad_H = np.array([1.0, 2.0], dtype=np.float64)  # Dimensión 2
        R_matrix = np.eye(3, dtype=np.float64)  # Dimensión 3
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase1_enforcer._complete_phase1_causal_filtration(
                completed_strata=valid_strata_complete,
                total_spans=total_spans,
                causal_edges=causal_edges,
                grad_H=grad_H,
                R_matrix=R_matrix,
                connected_components=connected_components,
            )
        assert "dimensión incompatible" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 2: COHOMOLOGÍA DE SPANS CAUSALES
#   Valida: χ(K) = β₀ - β₁ = |V| - |E| y β₁ = 0
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase2_CausalCohomologyAuditor:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 2: COHOMOLOGÍA DE SPANS CAUSALES                                                ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida los invariantes cohomológicos del grafo causal.         ║
    ║  Cada método prueba un axioma específico del §2 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.1. Pruebas de Puente desde Fase 1 (Método: _begin_phase2_from_phase1_bridge)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_begin_phase2_from_phase1_bridge_valid(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Inicio de Fase 2 desde puente de Fase 1 válido.
        VALIDA: Continuidad funtorial Φ₁ → Φ₂.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        phase1_bridge = phase1_enforcer._complete_phase1_causal_filtration(
            completed_strata=valid_strata_complete,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
            connected_components=connected_components,
        )
        phase2_bridge = phase2_auditor._begin_phase2_from_phase1_bridge(phase1_bridge)
        assert isinstance(phase2_bridge, Phase2CohomologyBridge)
        assert isinstance(phase2_bridge.phase1_bridge, Phase1CausalBridge)
        assert isinstance(phase2_bridge.cohomology_audit, CausalCohomologyData)

    def test_begin_phase2_from_phase1_bridge_invalid_type_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Puente de Fase 1 inválido lanza TelemetryAgentError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase2_auditor._begin_phase2_from_phase1_bridge(None)  # type: ignore
        assert "phase1causalbridge" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.2. Pruebas de Auditoría Cohomológica (Método: _audit_causal_span_cohomology)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_audit_causal_span_cohomology_tree_valid(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Grafo tipo árbol (β₁ = 0) es válido.
        VALIDA: §2. β₁ = 0 (sin ciclos).
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        result = phase2_auditor._audit_causal_span_cohomology(
            total_spans=total_spans,
            causal_edges=causal_edges,
            connected_components=connected_components,
        )
        assert isinstance(result, CausalCohomologyData)
        assert result.vertices_count == total_spans
        assert result.edges_count == causal_edges
        assert result.betti_0 == connected_components
        assert result.betti_1 == 0  # Árbol: sin ciclos
        assert result.euler_characteristic == total_spans - causal_edges
        assert result.is_acyclic_directed is True

    def test_audit_causal_span_cohomology_empty_graph_valid(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Grafo vacío es válido.
        VALIDA: Caso degenerado aceptado.
        """
        result = phase2_auditor._audit_causal_span_cohomology(
            total_spans=0,
            causal_edges=0,
            connected_components=0,
        )
        assert result.vertices_count == 0
        assert result.edges_count == 0
        assert result.betti_1 == 0

    def test_audit_causal_span_cohomology_cycle_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Grafo con ciclo (β₁ > 0) lanza CausalCohomologyError.
        VALIDA: §2. Anti-ciclos causales.
        """
        # 3 vértices, 3 aristas, 1 componente → β₁ = 1 - 3 + 3 = 1
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=3,
                causal_edges=3,
                connected_components=1,
            )
        assert "ciclo" in str(exc_info.value).lower() or "β₁" in str(exc_info.value)

    def test_audit_causal_span_cohomology_negative_betti1_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: β₁ < 0 lanza CausalCohomologyError.
        VALIDA: Consistencia combinatoria.
        """
        # 5 vértices, 2 aristas, 1 componente → β₁ = 1 - 5 + 2 = -2
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=5,
                causal_edges=2,
                connected_components=1,
            )
        assert "β₁" in str(exc_info.value) or "inconsistente" in str(exc_info.value).lower()

    def test_audit_causal_span_cohomology_exceeds_max_depth_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Profundidad excede _MAX_CAUSAL_DEPTH lanza CausalCohomologyError.
        VALIDA: Límite anti-runaway.
        """
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=_MAX_CAUSAL_DEPTH + 1,
                causal_edges=_MAX_CAUSAL_DEPTH,
                connected_components=1,
            )
        assert "profundidad" in str(exc_info.value).lower() or "degeneración" in str(exc_info.value).lower()

    def test_audit_causal_span_cohomology_edges_without_vertices_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Aristas sin vértices lanza CausalCohomologyError.
        VALIDA: Consistencia del grafo.
        """
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=0,
                causal_edges=1,
                connected_components=0,
            )
        assert "vértices=0" in str(exc_info.value).lower() or "aristas" in str(exc_info.value).lower()

    def test_audit_causal_span_cohomology_components_out_of_range_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Componentes fuera de rango lanza CausalCohomologyError.
        VALIDA: 1 ≤ β₀ ≤ |V|
        """
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=5,
                causal_edges=4,
                connected_components=10,  # > vertices
            )
        assert "componentes" in str(exc_info.value).lower() or "inconsistente" in str(exc_info.value).lower()

    def test_audit_causal_span_cohomology_exceeds_max_edges_raises(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
    ) -> None:
        """
        PRUEBA: Aristas exceden máximo simple lanza CausalCohomologyError.
        VALIDA: Límite combinatorio |E| ≤ n(n-1)/2
        """
        # 3 vértices, máximo 3 aristas simples
        with pytest.raises(CausalCohomologyError) as exc_info:
            phase2_auditor._audit_causal_span_cohomology(
                total_spans=3,
                causal_edges=10,  # Excede 3*(3-1)/2 = 3
                connected_components=1,
            )
        assert "excede" in str(exc_info.value).lower() or "máximo" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.3. Pruebas de Estructura CausalCohomologyData
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_causal_cohomology_data_creation(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Creación de CausalCohomologyData.
        VALIDA: Estructura inmutable del DTO.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        result = phase2_auditor._audit_causal_span_cohomology(
            total_spans=total_spans,
            causal_edges=causal_edges,
            connected_components=connected_components,
        )
        assert isinstance(result.vertices_count, int)
        assert isinstance(result.edges_count, int)
        assert isinstance(result.betti_0, int)
        assert isinstance(result.betti_1, int)
        assert isinstance(result.euler_characteristic, int)
        assert isinstance(result.is_acyclic_directed, bool)

    def test_causal_cohomology_data_immutability(
        self,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: CausalCohomologyData es inmutable.
        VALIDA: Integridad estructural.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        result = phase2_auditor._audit_causal_span_cohomology(
            total_spans=total_spans,
            causal_edges=causal_edges,
            connected_components=connected_components,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.betti_1 = 1  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 3: DISIPACIÓN PORT-HAMILTONIANA
#   Valida: P_diss = ∇Hᵀ R ∇H ≥ 0 y R ⪰ 0
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase3_PortHamiltonianVerifier:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 3: DISIPACIÓN PORT-HAMILTONIANA                                                 ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la irreversibilidad termodinámica del sistema.          ║
    ║  Cada método prueba un axioma específico del §3 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.1. Pruebas de Puente desde Fase 2 (Método: _begin_phase3_from_phase2_bridge)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_begin_phase3_from_phase2_bridge_valid(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        phase2_auditor: Phase2_CausalCohomologyAuditor,
        phase1_enforcer: Phase1_TransitiveClosureEnforcer,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Inicio de Fase 3 desde puente de Fase 2 válido.
        VALIDA: Continuidad funtorial Φ₂ → Φ₃.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        phase1_bridge = phase1_enforcer._complete_phase1_causal_filtration(
            completed_strata=valid_strata_complete,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
            connected_components=connected_components,
        )
        phase2_bridge = phase2_auditor._begin_phase2_from_phase1_bridge(phase1_bridge)
        result = phase3_verifier._begin_phase3_from_phase2_bridge(phase2_bridge)
        assert isinstance(result, ThermodynamicDissipationData)
        assert isinstance(result.dissipated_power, float)
        assert result.dissipated_power >= 0
        assert result.is_entropically_valid is True

    def test_begin_phase3_from_phase2_bridge_invalid_type_raises(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
    ) -> None:
        """
        PRUEBA: Puente de Fase 2 inválido lanza TelemetryAgentError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase3_verifier._begin_phase3_from_phase2_bridge(None)  # type: ignore
        assert "phase2cohomologybridge" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.2. Pruebas de Disipación Port-Hamiltoniana (Método: _verify_port_hamiltonian_dissipation)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_verify_port_hamiltonian_dissipation_valid(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Disipación Port-Hamiltoniana válida (P_diss ≥ 0).
        VALIDA: §3. Segunda Ley de Termodinámica.
        """
        result = phase3_verifier._verify_port_hamiltonian_dissipation(
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
        )
        assert isinstance(result, ThermodynamicDissipationData)
        assert result.dissipated_power >= 0
        assert result.gradient_norm > 0
        assert result.spectral_min >= 0
        assert result.spectral_max > 0
        assert result.is_entropically_valid is True

    def test_verify_port_hamiltonian_dissipation_identity_matrix(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
        identity_matrix_3x3: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Matriz identidad (PSD trivial).
        VALIDA: P_diss = ||∇H||²
        """
        result = phase3_verifier._verify_port_hamiltonian_dissipation(
            grad_H=valid_grad_H,
            R_matrix=identity_matrix_3x3,
        )
        expected_power = float(np.sum(valid_grad_H ** 2))
        assert np.isclose(result.dissipated_power, expected_power)

    def test_verify_port_hamiltonian_dissipation_dimension_mismatch_raises(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Dimensión incompatible entre grad_H y R_matrix lanza excepción.
        VALIDA: Consistencia dimensional.
        """
        R_wrong = np.eye(4, dtype=np.float64)  # Dimensión 4 vs grad_H dim 3
        with pytest.raises(TelemetryAgentError) as exc_info:
            phase3_verifier._verify_port_hamiltonian_dissipation(
                grad_H=valid_grad_H,
                R_matrix=R_wrong,
            )
        assert "dimensión incompatible" in str(exc_info.value).lower()

    def test_verify_port_hamiltonian_dissipation_negative_eigenvalue_raises(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Matriz no PSD (eigenvalor negativo) lanza ThermodynamicReversibilityError.
        VALIDA: §3. R ⪰ 0.
        """
        # Matriz con eigenvalor negativo
        R_non_psd = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        grad_2d = np.array([1.0, 1.0], dtype=np.float64)
        with pytest.raises(ThermodynamicReversibilityError) as exc_info:
            phase3_verifier._verify_port_hamiltonian_dissipation(
                grad_H=grad_2d,
                R_matrix=R_non_psd,
            )
        assert "positiva semidefinida" in str(exc_info.value).lower() or "lambda_min" in str(exc_info.value)

    def test_verify_port_hamiltonian_dissipation_negative_power_raises(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: P_diss < 0 lanza ThermodynamicReversibilityError.
        VALIDA: §3. P_diss ≥ 0 (anti-entropía negativa).
        """
        # Matriz definida negativa
        R_negative = -np.eye(3, dtype=np.float64)
        with pytest.raises(ThermodynamicReversibilityError) as exc_info:
            phase3_verifier._verify_port_hamiltonian_dissipation(
                grad_H=valid_grad_H,
                R_matrix=R_negative,
            )
        assert "energía disipada" in str(exc_info.value).lower() or "p_diss" in str(exc_info.value).lower()

    def test_verify_port_hamiltonian_dissipation_asymmetric_matrix_warning(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Matriz asimétrica se simetriza (warning interno).
        VALIDA: R_sym = 0.5 * (R + R.T)
        """
        # Matriz asimétrica pero con parte simétrica PSD
        R_asym = np.array([[2.0, 1.0, 0.5],
                          [0.5, 2.0, 1.0],
                          [0.0, 1.0, 2.0]], dtype=np.float64)
        result = phase3_verifier._verify_port_hamiltonian_dissipation(
            grad_H=valid_grad_H,
            R_matrix=R_asym,
        )
        assert result.is_entropically_valid is True
        assert result.dissipated_power >= 0

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.3. Pruebas de Estructura ThermodynamicDissipationData
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_thermodynamic_dissipation_data_creation(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de ThermodynamicDissipationData.
        VALIDA: Estructura inmutable del DTO.
        """
        result = phase3_verifier._verify_port_hamiltonian_dissipation(
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
        )
        assert isinstance(result.dissipated_power, float)
        assert isinstance(result.gradient_norm, float)
        assert isinstance(result.spectral_min, float)
        assert isinstance(result.spectral_max, float)
        assert isinstance(result.is_entropically_valid, bool)

    def test_thermodynamic_dissipation_data_immutability(
        self,
        phase3_verifier: Phase3_PortHamiltonianVerifier,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: ThermodynamicDissipationData es inmutable.
        VALIDA: Integridad estructural.
        """
        result = phase3_verifier._verify_port_hamiltonian_dissipation(
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.dissipated_power = -1.0  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   ORQUESTADOR SUPREMO: TELEMETRYAGENT (Pruebas de Integración)
#   Valida: Endofuntor Z_Telemetry = Φ₃ ∘ Φ₂ ∘ Φ₁
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestTelemetryAgent_Integration:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  ORQUESTADOR SUPREMO: TELEMETRYAGENT                                                  ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Pruebas de integración que validan el endofuntor completo Z_Telemetry.               ║
    ║  Estas pruebas aseguran que la composición Φ₃ ∘ Φ₂ ∘ Φ₁ funciona correctamente.       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_telemetry_agent_execute_causal_propagation_governance_valid(
        self,
        telemetry_agent: TelemetryAgent,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Ejecución completa del gobierno de propagación causal.
        VALIDA: Endofuntor Z_Telemetry con datos válidos.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        state = telemetry_agent.execute_causal_propagation_governance(
            completed_strata=valid_strata_complete,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
            connected_components=connected_components,
        )
        assert isinstance(state, CausalPropagationState)
        assert state.is_epistemologically_valid is True
        assert state.filtration_audit.is_filtration_valid is True
        assert state.cohomology_audit.betti_1 == 0
        assert state.cohomology_audit.is_acyclic_directed is True
        assert state.dissipation_audit.dissipated_power >= 0
        assert state.dissipation_audit.is_entropically_valid is True

    def test_telemetry_agent_transitive_closure_violation(
        self,
        telemetry_agent: TelemetryAgent,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Violación de clausura transitiva lanza TransitiveClosureViolation.
        VALIDA: Propagación de excepciones de Fase 1.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        # WISDOM sin PHYSICS
        with pytest.raises(TransitiveClosureViolation):
            telemetry_agent.execute_causal_propagation_governance(
                completed_strata={Stratum.WISDOM},
                total_spans=total_spans,
                causal_edges=causal_edges,
                grad_H=valid_grad_H,
                R_matrix=valid_R_matrix,
                connected_components=connected_components,
            )

    def test_telemetry_agent_causal_cohomology_error_cycle(
        self,
        telemetry_agent: TelemetryAgent,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Ciclo causal (β₁ > 0) lanza CausalCohomologyError.
        VALIDA: Propagación de excepciones de Fase 2.
        """
        # Grafo con ciclo: 3 vértices, 3 aristas
        with pytest.raises(CausalCohomologyError):
            telemetry_agent.execute_causal_propagation_governance(
                completed_strata=valid_strata_complete,
                total_spans=3,
                causal_edges=3,
                grad_H=valid_grad_H,
                R_matrix=valid_R_matrix,
                connected_components=1,
            )

    def test_telemetry_agent_thermodynamic_reversibility_error(
        self,
        telemetry_agent: TelemetryAgent,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Matriz no PSD lanza ThermodynamicReversibilityError.
        VALIDA: Propagación de excepciones de Fase 3.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        R_negative = -np.eye(3, dtype=np.float64)
        with pytest.raises(ThermodynamicReversibilityError):
            telemetry_agent.execute_causal_propagation_governance(
                completed_strata=valid_strata_complete,
                total_spans=total_spans,
                causal_edges=causal_edges,
                grad_H=valid_grad_H,
                R_matrix=R_negative,
                connected_components=connected_components,
            )

    def test_telemetry_agent_inheritance_chain(
        self,
        telemetry_agent: TelemetryAgent,
    ) -> None:
        """
        PRUEBA: Cadena de herencia del TelemetryAgent.
        VALIDA: Arquitectura de fases anidadas.
        """
        assert isinstance(telemetry_agent, TelemetryAgent)
        assert isinstance(telemetry_agent, Phase3_PortHamiltonianVerifier)
        assert isinstance(telemetry_agent, Phase2_CausalCohomologyAuditor)
        assert isinstance(telemetry_agent, Phase1_TransitiveClosureEnforcer)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Z. PRUEBAS DE ESTRUCTURAS DE DATOS (Data Classes)
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE ESTRUCTURAS DE DATOS INMUTABLES                                           ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad de todos los DTOs del espacio causal.                           ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_filtration_audit_data_creation(
        self,
        valid_strata_complete: Set[Stratum],
    ) -> None:
        """
        PRUEBA: Creación de FiltrationAuditData.
        VALIDA: Estructura inmutable del certificado.
        """
        audit = FiltrationAuditData(
            active_strata=frozenset(valid_strata_complete),
            is_filtration_valid=True,
        )
        assert audit.is_filtration_valid is True
        assert len(audit.active_strata) == 4

    def test_causal_cohomology_data_creation(
        self,
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Creación de CausalCohomologyData.
        VALIDA: Artefacto de Fase 2.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        audit = CausalCohomologyData(
            vertices_count=total_spans,
            edges_count=causal_edges,
            betti_0=connected_components,
            betti_1=0,
            euler_characteristic=total_spans - causal_edges,
            is_acyclic_directed=True,
        )
        assert audit.is_acyclic_directed is True
        assert audit.betti_1 == 0

    def test_thermodynamic_dissipation_data_creation(
        self,
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de ThermodynamicDissipationData.
        VALIDA: Artefacto de Fase 3.
        """
        p_diss = float(valid_grad_H.T @ valid_R_matrix @ valid_grad_H)
        grad_norm = float(np.linalg.norm(valid_grad_H))
        eigenvalues = np.linalg.eigvalsh(valid_R_matrix)
        audit = ThermodynamicDissipationData(
            dissipated_power=p_diss,
            gradient_norm=grad_norm,
            spectral_min=float(np.max([0.0, np.min(eigenvalues)])),
            spectral_max=float(np.max(eigenvalues)),
            is_entropically_valid=True,
        )
        assert audit.is_entropically_valid is True
        assert audit.dissipated_power >= 0

    def test_phase1_causal_bridge_creation(
        self,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Creación de Phase1CausalBridge.
        VALIDA: Puente funtorial Φ₁ → Φ₂.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        filtration_audit = FiltrationAuditData(
            active_strata=frozenset(valid_strata_complete),
            is_filtration_valid=True,
        )
        bridge = Phase1CausalBridge(
            filtration_audit=filtration_audit,
            total_spans=total_spans,
            causal_edges=causal_edges,
            connected_components=connected_components,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
        )
        assert isinstance(bridge.filtration_audit, FiltrationAuditData)
        assert np.array_equal(bridge.grad_H, valid_grad_H)

    def test_phase2_cohomology_bridge_creation(
        self,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Creación de Phase2CohomologyBridge.
        VALIDA: Puente funtorial Φ₂ → Φ₃.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        filtration_audit = FiltrationAuditData(
            active_strata=frozenset(valid_strata_complete),
            is_filtration_valid=True,
        )
        phase1_bridge = Phase1CausalBridge(
            filtration_audit=filtration_audit,
            total_spans=total_spans,
            causal_edges=causal_edges,
            connected_components=connected_components,
            grad_H=valid_grad_H,
            R_matrix=valid_R_matrix,
        )
        cohomology_audit = CausalCohomologyData(
            vertices_count=total_spans,
            edges_count=causal_edges,
            betti_0=connected_components,
            betti_1=0,
            euler_characteristic=total_spans - causal_edges,
            is_acyclic_directed=True,
        )
        bridge = Phase2CohomologyBridge(
            phase1_bridge=phase1_bridge,
            cohomology_audit=cohomology_audit,
        )
        assert isinstance(bridge.phase1_bridge, Phase1CausalBridge)
        assert isinstance(bridge.cohomology_audit, CausalCohomologyData)

    def test_causal_propagation_state_creation(
        self,
        valid_strata_complete: Set[Stratum],
        valid_grad_H: NDArray[np.float64],
        valid_R_matrix: NDArray[np.float64],
        valid_causal_graph_params: tuple[int, int, int],
    ) -> None:
        """
        PRUEBA: Creación de CausalPropagationState (objeto final).
        VALIDA: Estado epistemológico completo del endofuntor.
        """
        total_spans, causal_edges, connected_components = valid_causal_graph_params
        filtration_audit = FiltrationAuditData(
            active_strata=frozenset(valid_strata_complete),
            is_filtration_valid=True,
        )
        cohomology_audit = CausalCohomologyData(
            vertices_count=total_spans,
            edges_count=causal_edges,
            betti_0=connected_components,
            betti_1=0,
            euler_characteristic=total_spans - causal_edges,
            is_acyclic_directed=True,
        )
        p_diss = float(valid_grad_H.T @ valid_R_matrix @ valid_grad_H)
        dissipation_audit = ThermodynamicDissipationData(
            dissipated_power=p_diss,
            gradient_norm=float(np.linalg.norm(valid_grad_H)),
            spectral_min=0.0,
            spectral_max=float(np.max(np.linalg.eigvalsh(valid_R_matrix))),
            is_entropically_valid=True,
        )
        state = CausalPropagationState(
            filtration_audit=filtration_audit,
            cohomology_audit=cohomology_audit,
            dissipation_audit=dissipation_audit,
            is_epistemologically_valid=True,
        )
        assert state.is_epistemologically_valid is True
        assert isinstance(state.filtration_audit, FiltrationAuditData)
        assert isinstance(state.cohomology_audit, CausalCohomologyData)
        assert isinstance(state.dissipation_audit, ThermodynamicDissipationData)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §∞. PRUEBAS DE CONSTANTES FÍSICO-MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhysicalMathematicalConstants:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE CONSTANTES FÍSICO-MATEMÁTICAS                                             ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida que las constantes del módulo tengan valores correctos y consistentes.        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_machine_epsilon_value(self) -> None:
        """
        PRUEBA: Valor de _MACHINE_EPSILON.
        VALIDA: Precisión de float64 de NumPy.
        """
        assert _MACHINE_EPSILON == float(np.finfo(np.float64).eps)
        assert _MACHINE_EPSILON > 0
        assert _MACHINE_EPSILON < 1e-15

    def test_max_causal_depth_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_CAUSAL_DEPTH.
        VALIDA: Límite anti-runaway para grafos.
        """
        assert _MAX_CAUSAL_DEPTH == 1024
        assert _MAX_CAUSAL_DEPTH > 0

    def test_dissipation_negative_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _DISSIPATION_NEGATIVE_TOLERANCE.
        VALIDA: Tolerancia para energía negativa.
        """
        assert _DISSIPATION_NEGATIVE_TOLERANCE == 1e-12
        assert _DISSIPATION_NEGATIVE_TOLERANCE > 0

    def test_psd_eigenvalue_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _PSD_EIGENVALUE_TOLERANCE.
        VALIDA: Tolerancia espectral para PSD.
        """
        assert _PSD_EIGENVALUE_TOLERANCE == 1e-12
        assert _PSD_EIGENVALUE_TOLERANCE > 0

    def test_symmetry_warning_ratio_value(self) -> None:
        """
        PRUEBA: Valor de _SYMMETRY_WARNING_RATIO.
        VALIDA: Ratio de advertencia de asimetría.
        """
        assert _SYMMETRY_WARNING_RATIO == 1e-8
        assert _SYMMETRY_WARNING_RATIO > 0


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. PRUEBAS DEL ENUM STRATUM
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestStratumEnum:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DEL ENUMERADO STRATUM (Jerarquía DIKW)                                       ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad de la jerarquía de estratos.                                    ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_stratum_physics_value(self) -> None:
        """
        PRUEBA: Stratum.PHYSICS = 0.
        VALIDA: Estrato base.
        """
        assert Stratum.PHYSICS.value == 0

    def test_stratum_tactics_value(self) -> None:
        """
        PRUEBA: Stratum.TACTICS = 1.
        VALIDA: Segundo estrato.
        """
        assert Stratum.TACTICS.value == 1

    def test_stratum_strategy_value(self) -> None:
        """
        PRUEBA: Stratum.STRATEGY = 2.
        VALIDA: Tercer estrato.
        """
        assert Stratum.STRATEGY.value == 2

    def test_stratum_wisdom_value(self) -> None:
        """
        PRUEBA: Stratum.WISDOM = 3.
        VALIDA: Estrato superior.
        """
        assert Stratum.WISDOM.value == 3

    def test_stratum_ordering(self) -> None:
        """
        PRUEBA: Ordenamiento correcto de estratos.
        VALIDA: PHYSICS < TACTICS < STRATEGY < WISDOM
        """
        assert Stratum.PHYSICS.value < Stratum.TACTICS.value
        assert Stratum.TACTICS.value < Stratum.STRATEGY.value
        assert Stratum.STRATEGY.value < Stratum.WISDOM.value

    def test_stratum_from_int(self) -> None:
        """
        PRUEBA: Conversión de entero a Stratum.
        VALIDA: Coerción de tipos.
        """
        assert Stratum(0) == Stratum.PHYSICS
        assert Stratum(1) == Stratum.TACTICS
        assert Stratum(2) == Stratum.STRATEGY
        assert Stratum(3) == Stratum.WISDOM

    def test_stratum_from_name(self) -> None:
        """
        PRUEBA: Conversión de nombre a Stratum.
        VALIDA: Lookup por nombre.
        """
        assert Stratum["PHYSICS"] == Stratum.PHYSICS
        assert Stratum["WISDOM"] == Stratum.WISDOM


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. EJECUCIÓN DIRECTA (Para debugging)
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución directa para debugging fuera de pytest.
    Uso: python tests/unit/agents/core/test_telemetry_agent.py
    """
    import sys
    import os
    
    # Agregar el directorio raíz al path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    
    pytest.main([__file__, "-v", "--tb=short"])