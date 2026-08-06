# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Batería de pruebas unitarias para:                                                      ║
║  app/agents/tactics/semantic_estimator_agent.py                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ORGANIZACIÓN POR FASES ANIDADAS:                                                        ║
║  ────────────────────────────────                                                        ║
║  Fase 1 → Certificación de vecindad topológica de Hilbert.                               ║
║           El último test valida que `Phase1TopologicalBridge` es el objeto inicial       ║
║           de Fase 2.                                                                     ║
║                                                                                          ║
║  Fase 2 → Auditoría del operador de fricción territorial y ensamblaje de costos.         ║
║           El último test valida que `Phase2FrictionBridge` es el objeto inicial          ║
║           de Fase 3.                                                                     ║
║                                                                                          ║
║  Fase 3 → Teorema de Rango-Nulidad e isometría parcial ortogonal.                        ║
║           El último test sintetiza `SemanticEstimatorAuditState`.                        ║
║                                                                                          ║
║  Orquestación → `SemanticEstimatorAgent` como composición terminal Φ₃ ∘ Φ₂ ∘ Φ₁.         ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("numpy", reason="El módulo auditado requiere NumPy.")
pytest.importorskip("scipy", reason="El módulo auditado requiere SciPy.")

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# §0. CARGA ROBUSTA DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

_TARGET_MODULE_NAME = "app.agents.tactics.semantic_estimator_agent"
_TARGET_REL_PATH = Path("app") / "agents" / "tactics" / "semantic_estimator_agent.py"


def _candidate_roots() -> list[Path]:
    """Devuelve raíces candidatas donde podría encontrarse el paquete `app`."""
    roots: list[Path] = [Path.cwd()]
    try:
        roots.extend(Path(__file__).resolve().parents)
    except Exception:  # pragma: no cover - defensa extrema del loader
        pass

    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _ensure_syspath_for_app() -> None:
    """Inserta en `sys.path` la raíz del proyecto si contiene el módulo objetivo."""
    for root in _candidate_roots():
        candidate = root / _TARGET_REL_PATH
        if candidate.is_file():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)


def _load_target_module():
    """
    Importa el módulo objetivo por nombre de paquete; si falla, lo carga
    directamente desde el sistema de ficheros.
    """
    _ensure_syspath_for_app()

    try:
        return importlib.import_module(_TARGET_MODULE_NAME)
    except Exception as import_error:
        for root in _candidate_roots():
            candidate = root / _TARGET_REL_PATH
            if candidate.is_file():
                spec = importlib.util.spec_from_file_location(
                    _TARGET_MODULE_NAME,
                    str(candidate),
                )
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[_TARGET_MODULE_NAME] = module
                spec.loader.exec_module(module)
                return module

        raise ImportError(
            "No se pudo importar el módulo bajo prueba "
            f"'{_TARGET_MODULE_NAME}' ni localizar '{_TARGET_REL_PATH}'."
        ) from import_error


target = _load_target_module()

# ══════════════════════════════════════════════════════════════════════════════
# §1. IMPORTACIÓN DE SÍMBOLOS AUDITADOS
# ══════════════════════════════════════════════════════════════════════════════

# Excepciones.
SemanticEstimatorAgentError = target.SemanticEstimatorAgentError
TopologicalMappingError = target.TopologicalMappingError
VectorDegeneracyError = target.VectorDegeneracyError
DimensionalIncompatibilityError = target.DimensionalIncompatibilityError
ThermodynamicFrictionAnomaly = target.ThermodynamicFrictionAnomaly
FunctorialityError = target.FunctorialityError
ProjectorIntegrityError = target.ProjectorIntegrityError

# DTOs.
TopologicalNeighborhoodData = target.TopologicalNeighborhoodData
TensorFrictionData = target.TensorFrictionData
RankNullityProjectionData = target.RankNullityProjectionData
Phase1TopologicalBridge = target.Phase1TopologicalBridge
Phase2FrictionBridge = target.Phase2FrictionBridge
SemanticEstimatorAuditState = target.SemanticEstimatorAuditState

# Clases de fase.
Phase1_TopologicalNeighborhoodCertifier = target.Phase1_TopologicalNeighborhoodCertifier
Phase2_TensorFrictionAuditor = target.Phase2_TensorFrictionAuditor
Phase3_RankNullityProjector = target.Phase3_RankNullityProjector

# Orquestador supremo.
SemanticEstimatorAgent = target.SemanticEstimatorAgent

# Entidades categóricas opcionales.
TopologicalInvariantError = getattr(target, "TopologicalInvariantError", None)
Morphism = getattr(target, "Morphism", None)

# Constantes internas.
_MACHINE_EPSILON = float(getattr(target, "_MACHINE_EPSILON", np.finfo(np.float64).eps))
_TAU_MIN_SIMILARITY = float(getattr(target, "_TAU_MIN_SIMILARITY", 0.85))
_DEGENERACY_NORM_FLOOR = float(getattr(target, "_DEGENERACY_NORM_FLOOR", 1e-15))
_MAX_FRICTION_CONDITION = float(getattr(target, "_MAX_FRICTION_CONDITION", 1e3))
_POSITIVE_FLOOR = float(getattr(target, "_POSITIVE_FLOOR", 1e-12))
_NEGATIVE_TOLERANCE = float(getattr(target, "_NEGATIVE_TOLERANCE", 1e-12))
_SVD_ABSOLUTE_TOLERANCE = float(getattr(target, "_SVD_ABSOLUTE_TOLERANCE", 1e-10))
_ORTHOGONALITY_TOLERANCE = float(getattr(target, "_ORTHOGONALITY_TOLERANCE", 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
# §2. FÁBRICAS DE DATOS DETERMINISTAS
# ══════════════════════════════════════════════════════════════════════════════


def make_unit_vector(dim: int, seed: int = 0) -> np.ndarray:
    """Construye un vector unitario determinista en R^dim."""
    if dim <= 0:
        raise ValueError("dim debe ser positivo.")

    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float64)
    norm = float(np.linalg.norm(vector))

    if norm < 1e-12:
        vector = np.zeros(dim, dtype=np.float64)
        vector[0] = 1.0
        return vector

    return vector / norm


def make_similar_pair(
    dim: int = 4,
    cosine: float = 0.95,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye un par de vectores unitarios con similitud coseno prescrita.

    Sea q un vector unitario y w un vector unitario ortogonal a q:
        v = cos(θ) q + sin(θ) w
    entonces <q, v> = cos(θ).
    """
    if dim <= 0:
        raise ValueError("dim debe ser positivo.")

    cosine = float(max(-1.0, min(1.0, cosine)))

    query = make_unit_vector(dim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(size=dim).astype(np.float64)

    # Ortogonalización de Gram-Schmidt contra query.
    noise = noise - float(np.dot(noise, query)) * query
    noise_norm = float(np.linalg.norm(noise))

    if noise_norm < 1e-12:
        noise = np.zeros(dim, dtype=np.float64)
        noise[0] = 1.0
        noise = noise - float(np.dot(noise, query)) * query
        noise_norm = float(np.linalg.norm(noise))

        if noise_norm < 1e-12:
            noise = np.zeros(dim, dtype=np.float64)
            noise[-1] = 1.0
            noise = noise - float(np.dot(noise, query)) * query
            noise_norm = float(np.linalg.norm(noise))

    noise = noise / noise_norm

    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    retrieved = cosine * query + sine * noise
    retrieved_norm = float(np.linalg.norm(retrieved))

    if retrieved_norm < 1e-12:
        raise ValueError("No fue posible construir un vector recuperado no degenerado.")

    return query, retrieved / retrieved_norm


def make_rank1_partial_isometry(m: int = 5, n: int = 3, seed: int = 3) -> np.ndarray:
    """
    Construye una isometría parcial de rango 1:
        T = u vᵀ
    con ||u|| = ||v|| = 1. Entonces σ₁(T)=1 y rank(T)=1.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m y n deben ser positivos.")

    left = make_unit_vector(m, seed=seed)
    right = make_unit_vector(n, seed=seed + 1)
    return np.outer(left, right).astype(np.float64)


def make_valid_agent_inputs(
    dim_sem: int = 4,
    dim_cost: int = 3,
    m: int = 5,
    n: int = 3,
    cosine: float = 0.95,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Construye insumos completos válidos para el orquestador.

    - query/retrieved: vectores unitarios con cos(θ) ≥ τ_min.
    - cost_vector_c: costos estrictamente positivos.
    - friction_operator_F: operador diagonal positivo bien condicionado.
    - injection_matrix_T: isometría parcial de rango 1.
    """
    query, retrieved = make_similar_pair(dim=dim_sem, cosine=cosine, seed=seed)
    cost = np.linspace(1.0, 2.0, num=dim_cost, dtype=np.float64)
    friction = np.ones(dim_cost, dtype=np.float64)
    injection = make_rank1_partial_isometry(m=m, n=n, seed=seed + 1)

    return {
        "query_vector": query,
        "retrieved_vector": retrieved,
        "cost_vector_c": cost,
        "friction_operator_F": friction,
        "injection_matrix_T": injection,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §3. FIXTURES POR FASES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def phase1() -> Phase1_TopologicalNeighborhoodCertifier:
    """Instancia fresca del certificador de Fase 1."""
    return Phase1_TopologicalNeighborhoodCertifier()


@pytest.fixture(scope="function")
def phase2() -> Phase2_TensorFrictionAuditor:
    """Instancia fresca del auditor de Fase 2."""
    return Phase2_TensorFrictionAuditor()


@pytest.fixture(scope="function")
def phase3() -> Phase3_RankNullityProjector:
    """Instancia fresca del proyector de Fase 3."""
    return Phase3_RankNullityProjector()


@pytest.fixture(scope="function")
def agent() -> SemanticEstimatorAgent:
    """Instancia fresca del orquestador supremo."""
    return SemanticEstimatorAgent()


@pytest.fixture(scope="function")
def valid_inputs() -> Dict[str, Any]:
    """Insumos completos válidos."""
    return make_valid_agent_inputs()


@pytest.fixture(scope="function")
def valid_phase1_bridge(
    phase1: Phase1_TopologicalNeighborhoodCertifier,
    valid_inputs: Dict[str, Any],
) -> Phase1TopologicalBridge:
    """Puente terminal de Fase 1."""
    return phase1._phase1_certify_and_bridge_to_phase2(**valid_inputs)


@pytest.fixture(scope="function")
def valid_phase2_bridge(
    phase2: Phase2_TensorFrictionAuditor,
    valid_phase1_bridge: Phase1TopologicalBridge,
) -> Phase2FrictionBridge:
    """Puente terminal de Fase 2."""
    return phase2._phase2_audit_and_bridge_to_phase3(valid_phase1_bridge)


# ══════════════════════════════════════════════════════════════════════════════
# §4. CONTRATO DEL MÓDULO Y TAXONOMÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleContractAndExceptionTaxonomy:
    """Contrato estructural del módulo y jerarquía de excepciones."""

    def test_module_exposes_core_types(self) -> None:
        """El módulo debe exponer todos los tipos públicos principales."""
        core_names = (
            "SemanticEstimatorAgentError",
            "TopologicalMappingError",
            "VectorDegeneracyError",
            "DimensionalIncompatibilityError",
            "ThermodynamicFrictionAnomaly",
            "FunctorialityError",
            "ProjectorIntegrityError",
            "TopologicalNeighborhoodData",
            "TensorFrictionData",
            "RankNullityProjectionData",
            "Phase1TopologicalBridge",
            "Phase2FrictionBridge",
            "SemanticEstimatorAuditState",
            "Phase1_TopologicalNeighborhoodCertifier",
            "Phase2_TensorFrictionAuditor",
            "Phase3_RankNullityProjector",
            "SemanticEstimatorAgent",
        )

        for name in core_names:
            assert hasattr(target, name), f"El módulo no expone el tipo requerido: {name}"

    def test_phase_class_hierarchy_is_nested(self) -> None:
        """La jerarquía de clases debe reflejar la composición funtorial."""
        assert issubclass(
            Phase2_TensorFrictionAuditor,
            Phase1_TopologicalNeighborhoodCertifier,
        )
        assert issubclass(
            Phase3_RankNullityProjector,
            Phase2_TensorFrictionAuditor,
        )
        assert issubclass(SemanticEstimatorAgent, Phase3_RankNullityProjector)

    def test_agent_is_morphism_if_morphism_exists(self) -> None:
        """El agente debe ser un morfismo si el topo MIC define `Morphism`."""
        if Morphism is None:
            pytest.skip("El módulo auditado no expone Morphism.")

        assert issubclass(SemanticEstimatorAgent, Morphism)

    @pytest.mark.parametrize(
        "exc",
        [
            TopologicalMappingError,
            VectorDegeneracyError,
            DimensionalIncompatibilityError,
            ThermodynamicFrictionAnomaly,
            FunctorialityError,
            ProjectorIntegrityError,
        ],
    )
    def test_exceptions_are_rooted_in_semantic_estimator_error(self, exc) -> None:
        """Toda excepción de gobernanza debe descender de SemanticEstimatorAgentError."""
        assert issubclass(exc, SemanticEstimatorAgentError)

    def test_projector_integrity_error_is_functoriality_error(self) -> None:
        """ProjectorIntegrityError debe ser una especialización de FunctorialityError."""
        assert issubclass(ProjectorIntegrityError, FunctorialityError)

    def test_root_exception_inherits_topological_invariant_error_if_available(self) -> None:
        """Si existe TopologicalInvariantError, la raíz debe heredarla."""
        if TopologicalInvariantError is None:
            pytest.skip("TopologicalInvariantError no está disponible en el módulo.")

        assert issubclass(SemanticEstimatorAgentError, TopologicalInvariantError)


# ══════════════════════════════════════════════════════════════════════════════
# §5. FASE 1 — CERTIFICACIÓN DE VECINDAD TOPOLÓGICA
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1TopologicalNeighborhoodCertifier:
    """
    Fase 1: Geometría de Hilbert.

    Objetivo:
        cos(θ) = ⟨u,v⟩ / (||u|| ||v||) ≥ τ_min.

    El último test valida que el puente terminal de Fase 1 es consumido
    por Fase 2 cuando el mixin de Fase 2 está presente.
    """

    def test_phase1_certifier_is_instantiable(self) -> None:
        """Fase 1 debe poder instanciarse sin estado mutable."""
        certifier = Phase1_TopologicalNeighborhoodCertifier()
        assert isinstance(certifier, Phase1_TopologicalNeighborhoodCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.1. Coerción de escalares
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "value",
        [0, 1.5, -2.0, np.float64(3.25), "1.5", np.array(2.0)],
    )
    def test_coerce_finite_scalar_accepts_convertible_values(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Escalares convertibles a float64 finito deben ser aceptados."""
        scalar = phase1._coerce_finite_scalar("x", value)
        assert isinstance(scalar, float)
        assert math.isfinite(scalar)

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_coerce_finite_scalar_rejects_booleans(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Los booleanos pertenecen a B₂, no a R."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_scalar("x", value)

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan"), np.inf, np.nan],
    )
    def test_coerce_finite_scalar_rejects_non_finite_values(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Escalares no finitos deben ser rechazados."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_scalar("x", value)

    @pytest.mark.parametrize("value", [[1.0], [[1.0]], (1.0, 2.0)])
    def test_coerce_finite_scalar_rejects_non_scalar_arrays(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Arreglos con dimensión > 0 no son escalares."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_scalar("x", value)

    @pytest.mark.parametrize("value", [object(), 1 + 2j, "abc"])
    def test_coerce_finite_scalar_rejects_non_convertible_values(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Valores no convertibles a float64 deben ser rechazados."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_scalar("x", value)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.2. Coerción de vectores
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "value",
        [
            [1.0, 2.0],
            (1.0, 2.0),
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0]]).reshape(-1),
        ],
    )
    def test_coerce_finite_vector_accepts_one_dimensional_vectors(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Vectores 1-D finitos deben ser aceptados."""
        vector = phase1._coerce_finite_vector("v", value)
        assert vector.ndim == 1
        assert vector.size == 2
        assert np.all(np.isfinite(vector))

    @pytest.mark.parametrize("value", [1.0, np.array(1.0)])
    def test_coerce_finite_vector_rejects_scalar_objects(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Un escalar no es un vector 1-D."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_vector("v", value)

    def test_coerce_finite_vector_rejects_2d_column_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector columna 2-D no es aceptado directamente."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_vector("v", np.array([[1.0], [2.0]]))

    def test_coerce_finite_vector_rejects_empty_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """El vector vacío viola la completitud del espacio de Hilbert."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_vector("v", np.array([]))

    def test_coerce_finite_vector_enforces_expected_dimension(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La dimensión esperada debe cumplirse exactamente."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase1._coerce_finite_vector("v", np.array([1.0, 2.0]), expected_dim=3)

    def test_coerce_finite_vector_rejects_non_finite_entries(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Componentes NaN/Inf deben ser rechazadas."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_vector("v", np.array([1.0, np.nan]))

    def test_coerce_finite_vector_rejects_non_convertible_object(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Objetos no convertibles a float64 deben ser rechazados."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_vector("v", object())

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3. Coerción de matrices
    # ─────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_matrix_accepts_2d_matrix(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una matriz 2-D finita debe ser aceptada."""
        matrix = np.eye(2, dtype=np.float64)
        out = phase1._coerce_finite_matrix("M", matrix)

        assert out.ndim == 2
        assert out.shape == (2, 2)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("value", [1.0, np.array(1.0), np.array([1.0, 2.0])])
    def test_coerce_finite_matrix_rejects_non_2d_objects(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: Any,
    ) -> None:
        """Escalares y vectores no son matrices."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_matrix("M", value)

    @pytest.mark.parametrize("shape", [(0, 2), (2, 0), (0, 0)])
    def test_coerce_finite_matrix_rejects_empty_matrices(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        shape: tuple[int, int],
    ) -> None:
        """Una matriz vacía viola la integridad matricial."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_matrix("M", np.empty(shape, dtype=np.float64))

    def test_coerce_finite_matrix_rejects_non_finite_entries(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Matrices con NaN/Inf deben ser rechazadas."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_matrix("M", np.array([[1.0, np.nan]]))

    def test_coerce_finite_matrix_rejects_non_convertible_object(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Objetos no convertibles a matriz float64 deben ser rechazados."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_finite_matrix("M", object())

    # ─────────────────────────────────────────────────────────────────────────
    # 5.4. Coerción del operador de fricción
    # ─────────────────────────────────────────────────────────────────────────

    def test_coerce_friction_operator_accepts_scalar_for_dimension_one(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un escalar es admisible como operador diagonal en dimensión 1."""
        operator = phase1._coerce_finite_operator("F", 2.0, dimension=1)
        assert operator.shape == (1,)
        assert operator[0] == pytest.approx(2.0)

    def test_coerce_friction_operator_rejects_scalar_for_dimension_gt_one(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un escalar no es admisible para dimensión > 1."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase1._coerce_friction_operator("F", 2.0, dimension=2)

    def test_coerce_friction_operator_accepts_1d_diagonal(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector 1-D del tamaño correcto representa un operador diagonal."""
        operator = phase1._coerce_friction_operator("F", np.array([1.0, 2.0]), dimension=2)
        assert operator.shape == (2,)

    def test_coerce_friction_operator_rejects_wrong_1d_size(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector diagonal de tamaño incorrecto es incompatible."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase1._coerce_friction_operator("F", np.array([1.0]), dimension=2)

    def test_coerce_friction_operator_accepts_square_matrix(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una matriz cuadrada compatible es un operador matricial válido."""
        operator = phase1._coerce_friction_operator("F", np.eye(3), dimension=3)
        assert operator.shape == (3, 3)

    def test_coerce_friction_operator_rejects_wrong_matrix_shape(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una matriz no cuadrada o de dimensión incorrecta es incompatible."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase1._coerce_friction_operator("F", np.eye(2), dimension=3)

    def test_coerce_friction_operator_rejects_3d_tensors(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un tensor 3-D no es un operador de fricción admisible."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_friction_operator("F", np.zeros((2, 2, 2)), dimension=2)

    def test_coerce_friction_operator_rejects_non_finite_entries(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un operador con NaN/Inf debe ser rechazado."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase1._coerce_friction_operator("F", np.array([np.nan]), dimension=1)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.5. Compatibilidad dimensional
    # ─────────────────────────────────────────────────────────────────────────

    def test_verify_dimensional_compatibility_same_dimension(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores de igual dimensión definen un producto interno válido."""
        a = np.ones(3)
        b = np.ones(3)
        dim = phase1._verify_dimensional_compatibility("a", a, "b", b)
        assert dim == 3

    def test_verify_dimensional_compatibility_rejects_mismatch(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores de dimensión distinta no pueden compararse."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase1._verify_dimensional_compatibility(
                "a",
                np.ones(2),
                "b",
                np.ones(3),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 5.6. Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────

    def test_safe_l2_norm_zero_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La norma L2 del vector cero es cero."""
        assert phase1._safe_l2_norm(np.zeros(4)) == pytest.approx(0.0)

    def test_safe_l2_norm_known_value(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La norma L2 de (3,4) es 5."""
        norm = phase1._safe_l2_norm(np.array([3.0, 4.0]))
        assert norm == pytest.approx(5.0)

    def test_safe_l2_norm_large_finite_value(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector con una sola componente enorme conserva finitud."""
        norm = phase1._safe_l2_norm(np.array([1e308]))
        assert math.isfinite(norm)
        assert norm == pytest.approx(1e308)

    def test_safe_l2_norm_overflow_returns_inf(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector cuyo norma excede float64 debe retornar infinito."""
        norm = phase1._safe_l2_norm(np.array([1e308] * 4))
        assert math.isinf(norm)

    @pytest.mark.parametrize("value", [np.array([np.inf, 1.0]), np.array([np.nan, 1.0])])
    def test_safe_l2_norm_nonfinite_input_returns_inf(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        value: np.ndarray,
    ) -> None:
        """Entradas no finitas deben producir norma infinita."""
        norm = phase1._safe_l2_norm(value)
        assert math.isinf(norm)

    def test_safe_frobenius_norm_known_value(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La norma de Frobenius de [[3,4]] es 5."""
        norm = phase1._safe_frobenius_norm(np.array([[3.0, 4.0]]))
        assert norm == pytest.approx(5.0)

    def test_safe_frobenius_norm_overflow_returns_inf(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una matriz con Frobenius overflow debe retornar infinito."""
        matrix = np.array([[1e308, 1e308], [1e308, 1e308]])
        norm = phase1._safe_frobenius_norm(matrix)
        assert math.isinf(norm)

    def test_safe_l1_norm_known_value(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La norma L1 de (1,-2,3) es 6."""
        norm = phase1._safe_l1_norm(np.array([1.0, -2.0, 3.0]))
        assert norm == pytest.approx(6.0)

    def test_safe_l1_norm_overflow_returns_inf(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector con L1 overflow debe retornar infinito."""
        norm = phase1._safe_l1_norm(np.array([1e308] * 4))
        assert math.isinf(norm)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.7. No degeneración vectorial
    # ─────────────────────────────────────────────────────────────────────────

    def test_verify_non_degenerate_vectors_valid(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores con norma mayor al piso deben ser certificados."""
        q = make_unit_vector(3, seed=11)
        r = make_unit_vector(3, seed=12)

        norm_q, norm_r = phase1._verify_non_degenerate_vectors(q, r)

        assert norm_q == pytest.approx(1.0)
        assert norm_r == pytest.approx(1.0)

    def test_verify_non_degenerate_vectors_rejects_zero_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """El vector cero degenera la métrica angular."""
        with pytest.raises(VectorDegeneracyError):
            phase1._verify_non_degenerate_vectors(
                np.zeros(2),
                make_unit_vector(2, seed=13),
            )

    def test_verify_non_degenerate_vectors_rejects_subnormal_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una norma por debajo del piso de degeneración debe vetarse."""
        with pytest.raises(VectorDegeneracyError):
            phase1._verify_non_degenerate_vectors(
                np.array([1e-20, 0.0]),
                make_unit_vector(2, seed=14),
            )

    def test_verify_non_degenerate_vectors_rejects_nonfinite_norm(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una norma infinita no pertenece al espacio métrico certificado."""
        with pytest.raises(VectorDegeneracyError):
            phase1._verify_non_degenerate_vectors(
                np.array([np.inf, 1.0]),
                make_unit_vector(2, seed=15),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 5.8. Similitud coseno y distancia euclidiana
    # ─────────────────────────────────────────────────────────────────────────

    def test_compute_cosine_similarity_identical_vectors(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores idénticos unitarios tienen cos(θ)=1."""
        u = make_unit_vector(4, seed=16)
        norm_u = phase1._safe_l2_norm(u)

        cos = phase1._compute_cosine_similarity(u, u, norm_u, norm_u)
        assert cos == pytest.approx(1.0)

    def test_compute_cosine_similarity_opposite_vectors(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores opuestos tienen cos(θ)=-1."""
        u = make_unit_vector(4, seed=17)
        norm_u = phase1._safe_l2_norm(u)

        cos = phase1._compute_cosine_similarity(u, -u, norm_u, norm_u)
        assert cos == pytest.approx(-1.0)

    def test_compute_cosine_similarity_orthogonal_vectors(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores ortogonales tienen cos(θ)=0."""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

        cos = phase1._compute_cosine_similarity(u, v, 1.0, 1.0)
        assert cos == pytest.approx(0.0)

    def test_compute_cosine_similarity_rejects_nonfinite_result(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Una similitud coseno no finita debe detonar veto topológico."""
        u = np.array([np.inf])
        v = np.array([1.0])

        with pytest.raises(TopologicalMappingError):
            phase1._compute_cosine_similarity(u, v, math.inf, 1.0)

    def test_compute_euclidean_distance_known_value(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """La distancia euclidiana entre (1,0) y (0,1) es sqrt(2)."""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

        dist = phase1._compute_euclidean_distance(u, v)
        assert dist == pytest.approx(math.sqrt(2.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 5.9. Certificación principal de vecindad
    # ─────────────────────────────────────────────────────────────────────────

    def test_certify_topological_neighborhood_identical_vectors_pass(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores idénticos deben certificar vecindad con margen máximo."""
        q = make_unit_vector(4, seed=18)
        audit = phase1._certify_topological_neighborhood(q, q)

        assert isinstance(audit, TopologicalNeighborhoodData)
        assert audit.is_homotopically_valid is True
        assert audit.cosine_similarity == pytest.approx(1.0)
        assert audit.angle_radians == pytest.approx(0.0)
        assert audit.angle_degrees == pytest.approx(0.0)
        assert audit.euclidean_distance == pytest.approx(0.0)
        assert audit.similarity_margin == pytest.approx(1.0 - _TAU_MIN_SIMILARITY)

    def test_certify_topological_neighborhood_similar_vectors_pass(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores con cos(θ)=0.9 deben certificar vecindad."""
        q, r = make_similar_pair(dim=4, cosine=0.9, seed=19)
        audit = phase1._certify_topological_neighborhood(q, r)

        expected_angle = math.acos(0.9)

        assert audit.is_homotopically_valid is True
        assert audit.cosine_similarity == pytest.approx(0.9, abs=1e-12)
        assert audit.angle_radians == pytest.approx(expected_angle, abs=1e-12)
        assert audit.angle_degrees == pytest.approx(math.degrees(expected_angle), abs=1e-10)
        assert audit.similarity_margin == pytest.approx(0.9 - _TAU_MIN_SIMILARITY, abs=1e-12)

    def test_certify_topological_neighborhood_below_threshold_raises(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores con cos(θ)<τ_min deben detonar TopologicalMappingError."""
        q, r = make_similar_pair(dim=4, cosine=0.5, seed=20)

        with pytest.raises(TopologicalMappingError):
            phase1._certify_topological_neighborhood(q, r)

    def test_certify_topological_neighborhood_rejects_degenerate_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un vector degenerado debe vetar la certificación angular."""
        q = np.zeros(3, dtype=np.float64)
        r = make_unit_vector(3, seed=21)

        with pytest.raises(VectorDegeneracyError):
            phase1._certify_topological_neighborhood(q, r)

    def test_certify_topological_neighborhood_rejects_dimension_mismatch(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Vectores de dimensión distinta deben vetarse."""
        q = make_unit_vector(3, seed=22)
        r = make_unit_vector(4, seed=23)

        with pytest.raises(DimensionalIncompatibilityError):
            phase1._certify_topological_neighborhood(q, r)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.10. Puente terminal de Fase 1
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase1_bridge_contains_complete_artifacts(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """El puente de Fase 1 debe contener todos los artefactos necesarios."""
        bridge = phase1._phase1_certify_and_bridge_to_phase2(**valid_inputs)

        assert isinstance(bridge, Phase1TopologicalBridge)
        assert bridge.neighborhood_audit.is_homotopically_valid is True
        assert bridge.query_vector.shape == valid_inputs["query_vector"].shape
        assert bridge.retrieved_vector.shape == valid_inputs["retrieved_vector"].shape
        assert bridge.cost_vector_c.shape == valid_inputs["cost_vector_c"].shape
        assert bridge.friction_operator_F.shape == valid_inputs["friction_operator_F"].shape
        assert bridge.injection_matrix_T.shape == valid_inputs["injection_matrix_T"].shape

    def test_phase1_bridge_accepts_scalar_friction_for_one_dimensional_cost(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
    ) -> None:
        """Un costo 1-D admite un operador de fricción escalar."""
        inputs = make_valid_agent_inputs(dim_cost=1)
        inputs["friction_operator_F"] = 2.0

        bridge = phase1._phase1_certify_and_bridge_to_phase2(**inputs)
        assert bridge.friction_operator_F.shape == (1,)
        assert bridge.friction_operator_F[0] == pytest.approx(2.0)

    def test_phase1_bridge_rejects_degenerate_query(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Una consulta degenerada debe impedir el puente."""
        invalid = dict(valid_inputs)
        invalid["query_vector"] = np.zeros_like(valid_inputs["query_vector"])

        with pytest.raises(VectorDegeneracyError):
            phase1._phase1_certify_and_bridge_to_phase2(**invalid)

    def test_phase1_bridge_rejects_invalid_cost_vector(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Un vector de costos no finito debe impedir el puente."""
        invalid = dict(valid_inputs)
        invalid["cost_vector_c"] = np.array([1.0, np.nan, 2.0])

        with pytest.raises(SemanticEstimatorAgentError):
            phase1._phase1_certify_and_bridge_to_phase2(**invalid)

    def test_phase1_bridge_rejects_invalid_friction_dimension(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Un operador de fricción incompatible debe impedir el puente."""
        invalid = dict(valid_inputs)
        invalid["friction_operator_F"] = np.ones(
            valid_inputs["cost_vector_c"].size + 1,
            dtype=np.float64,
        )

        with pytest.raises(DimensionalIncompatibilityError):
            phase1._phase1_certify_and_bridge_to_phase2(**invalid)

    def test_phase1_bridge_rejects_invalid_injection_matrix(
        self,
        phase1: Phase1_TopologicalNeighborhoodCertifier,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Una matriz de inyección no finita debe impedir el puente."""
        invalid = dict(valid_inputs)
        invalid["injection_matrix_T"] = np.array([[np.nan]])

        with pytest.raises(SemanticEstimatorAgentError):
            phase1._phase1_certify_and_bridge_to_phase2(**invalid)

    def test_phase1_bridge_is_immutable(
        self,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """El puente de Fase 1 debe ser inmutable."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            valid_phase1_bridge.cost_vector_c = np.zeros(1)

    def test_phase1_terminal_bridge_is_phase2_initial_object(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₂ ∘ Φ₁:

        El puente terminal de Fase 1, cuando Fase 2 está presente,
        debe ser consumido como objeto inicial de Fase 2.
        """
        bridge = phase2._phase1_certify_and_bridge_to_phase2(**valid_inputs)
        phase2_bridge = phase2._phase2_audit_and_bridge_to_phase3(bridge)

        assert isinstance(phase2_bridge, Phase2FrictionBridge)
        assert phase2_bridge.phase1_bridge is bridge
        assert phase2_bridge.friction_audit.is_positive_definite is True


# ══════════════════════════════════════════════════════════════════════════════
# §6. FASE 2 — AUDITORÍA DE FRICCIÓN TERRITORIAL
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2TensorFrictionAuditor:
    """
    Fase 2: Estabilidad termodinámica del operador F_ext.

    Condiciones:
        1. c ≥ 0.
        2. F ≻ 0.
        3. κ(F) ≤ κ_max.
        4. C_total = F c ≥ 0.

    El último test valida que el puente terminal de Fase 2 es consumido
    por Fase 3 cuando el mixin de Fase 3 está presente.
    """

    def test_phase2_auditor_is_instantiable(self) -> None:
        """Fase 2 debe poder instanciarse como extensión de Fase 1."""
        auditor = Phase2_TensorFrictionAuditor()
        assert isinstance(auditor, Phase2_TensorFrictionAuditor)
        assert isinstance(auditor, Phase1_TopologicalNeighborhoodCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.1. Saneamiento de vectores no negativos
    # ─────────────────────────────────────────────────────────────────────────

    def test_sanitize_nonnegative_vector_preserves_positive_values(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Valores positivos deben preservarse exactamente."""
        vector = np.array([1.0, 2.0, 3.0])
        clean = phase2._sanitize_nonnegative_vector("v", vector)

        np.testing.assert_allclose(clean, vector)

    def test_sanitize_nonnegative_vector_zeroes_tiny_negative_values(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Negativos infinitesimales dentro de tolerancia se sanean a cero."""
        vector = np.array([-1e-13, 1.0])
        clean = phase2._sanitize_nonnegative_vector("v", vector)

        assert clean[0] == pytest.approx(0.0)
        assert clean[1] == pytest.approx(1.0)

    def test_sanitize_nonnegative_vector_rejects_macro_negative_values(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Negativos no infinitesimales violan la termodinámica de costos."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._sanitize_nonnegative_vector("v", np.array([-1e-2, 1.0]))

    def test_sanitize_nonnegative_vector_rejects_mixed_tiny_and_macro_negative(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Si existe un negativo macroscópico, el veto es incondicional."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._sanitize_nonnegative_vector("v", np.array([-1e-13, -1e-2]))

    # ─────────────────────────────────────────────────────────────────────────
    # 6.2. Auditoría de operador diagonal
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_diagonal_friction_operator_valid(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador diagonal positivo debe auditar correctamente."""
        diag = np.array([1.0, 2.0])
        cost = np.array([3.0, 4.0])

        (
            spectral_min,
            spectral_max,
            spectral_mean,
            spectral_std,
            total_cost_vector,
            determinant,
        ) = phase2._audit_diagonal_friction_operator(diag, cost)

        assert spectral_min == pytest.approx(1.0)
        assert spectral_max == pytest.approx(2.0)
        assert spectral_mean == pytest.approx(1.5)
        assert spectral_std == pytest.approx(0.5)
        np.testing.assert_allclose(total_cost_vector, np.array([3.0, 8.0]))
        assert determinant == pytest.approx(2.0)

    def test_audit_diagonal_friction_operator_rejects_zero_factor(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un factor cero singulariza el operador y debe vetarse."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_diagonal_friction_operator(
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
            )

    def test_audit_diagonal_friction_operator_rejects_negative_factor(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un factor negativo macroscópico inyecta energía anómala."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_diagonal_friction_operator(
                np.array([-1e-2, 1.0]),
                np.array([1.0, 1.0]),
            )

    def test_audit_diagonal_friction_operator_tiny_negative_becomes_singular(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un negativo infinitesimal saneado a cero singulariza el operador."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_diagonal_friction_operator(
                np.array([-1e-13, 1.0]),
                np.array([1.0, 1.0]),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 6.3. Auditoría de operador matricial
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_matrix_friction_operator_identity_valid(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """F = I es definido positivo y debe auditar correctamente."""
        F = np.eye(2, dtype=np.float64)
        cost = np.array([1.0, 2.0])

        (
            spectral_min,
            spectral_max,
            _,
            _,
            total_cost_vector,
            determinant,
            symmetry_residual,
        ) = phase2._audit_matrix_friction_operator(F, cost)

        assert spectral_min == pytest.approx(1.0)
        assert spectral_max == pytest.approx(1.0)
        np.testing.assert_allclose(total_cost_vector, cost)
        assert determinant == pytest.approx(1.0)
        assert symmetry_residual == pytest.approx(0.0)

    def test_audit_matrix_friction_operator_non_symmetric_positive_passes(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Una matriz positiva no simétrica se proyecta a su parte simétrica."""
        F = np.array([[1.0, 1e-3], [0.0, 1.0]], dtype=np.float64)
        cost = np.array([1.0, 1.0])

        (
            spectral_min,
            spectral_max,
            _,
            _,
            total_cost_vector,
            determinant,
            symmetry_residual,
        ) = phase2._audit_matrix_friction_operator(F, cost)

        assert spectral_min > _POSITIVE_FLOOR
        assert spectral_max >= spectral_min
        assert np.all(np.isfinite(total_cost_vector))
        assert math.isfinite(determinant)
        assert symmetry_residual > 1e-6

    def test_audit_matrix_friction_operator_rejects_negative_entries(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Entradas negativas macroscópicas violan el saneamiento territorial."""
        F = np.array([[1.0, -0.1], [-0.1, 1.0]], dtype=np.float64)

        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_matrix_friction_operator(F, np.array([1.0, 1.0]))

    def test_audit_matrix_friction_operator_rejects_indefinite_symmetric_part(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Una parte simétrica indefinida debe vetarse aunque las entradas sean positivas."""
        F = np.array([[1.0, 10.0], [10.0, 1.0]], dtype=np.float64)

        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_matrix_friction_operator(F, np.array([1.0, 1.0]))

    # ─────────────────────────────────────────────────────────────────────────
    # 6.4. Cota del número de condición
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("kappa", [1.0, 10.0, _MAX_FRICTION_CONDITION])
    def test_verify_condition_bound_accepts_bounded_condition(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        kappa: float,
    ) -> None:
        """Números de condición dentro de la cota deben aceptarse."""
        phase2._verify_condition_bound(kappa)

    def test_verify_condition_bound_rejects_excessive_condition(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """κ(F) > κ_max detona anomalía termodinámica."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._verify_condition_bound(_MAX_FRICTION_CONDITION + 1.0)

    @pytest.mark.parametrize("kappa", [float("inf"), float("nan")])
    def test_verify_condition_bound_rejects_non_finite_condition(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        kappa: float,
    ) -> None:
        """Un número de condición no finito debe vetarse."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._verify_condition_bound(kappa)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.5. Auditoría completa del ensamblaje
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_tensor_friction_assembly_diagonal_valid(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador diagonal positivo debe producir certificado completo."""
        cost = np.array([1.0, 1.0])
        F = np.array([1.0, 2.0])

        audit = phase2._audit_tensor_friction_assembly(cost, F)

        assert isinstance(audit, TensorFrictionData)
        assert audit.operator_type == "diagonal"
        assert audit.is_positive_definite is True
        assert audit.condition_number == pytest.approx(2.0)
        np.testing.assert_allclose(audit.total_cost_vector, np.array([1.0, 2.0]))
        assert audit.total_cost_norm == pytest.approx(3.0)
        assert audit.cost_vector_norm == pytest.approx(2.0)
        assert audit.friction_determinant == pytest.approx(2.0)

    def test_audit_tensor_friction_assembly_matrix_valid(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador matricial identidad debe producir certificado completo."""
        cost = np.array([1.0, 2.0])
        F = np.eye(2, dtype=np.float64)

        audit = phase2._audit_tensor_friction_assembly(cost, F)

        assert audit.operator_type == "matricial"
        assert audit.is_positive_definite is True
        assert audit.condition_number == pytest.approx(1.0)
        np.testing.assert_allclose(audit.total_cost_vector, cost)

    def test_audit_tensor_friction_assembly_rejects_negative_cost(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un costo negativo macroscópico debe vetarse."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_tensor_friction_assembly(
                np.array([-1.0, 1.0]),
                np.ones(2),
            )

    def test_audit_tensor_friction_assembly_sanitizes_tiny_negative_cost(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un costo negativo infinitesimal debe sanearse a cero."""
        cost = np.array([-1e-13, 1.0])
        F = np.ones(2, dtype=np.float64)

        audit = phase2._audit_tensor_friction_assembly(cost, F)

        np.testing.assert_allclose(audit.total_cost_vector, np.array([0.0, 1.0]))
        assert audit.total_cost_norm == pytest.approx(1.0)
        assert audit.cost_vector_norm == pytest.approx(1.0)

    def test_audit_tensor_friction_assembly_rejects_zero_diagonal_factor(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un factor diagonal cero singulariza el operador."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_tensor_friction_assembly(
                np.ones(2),
                np.array([0.0, 1.0]),
            )

    def test_audit_tensor_friction_assembly_rejects_high_condition_diagonal(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador diagonal mal condicionado debe vetarse."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_tensor_friction_assembly(
                np.ones(2),
                np.array([1e-6, 1.0]),
            )

    def test_audit_tensor_friction_assembly_rejects_high_condition_matrix(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador matricial mal condicionado debe vetarse."""
        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._audit_tensor_friction_assembly(
                np.ones(2),
                np.diag([1e-6, 1.0]),
            )

    def test_audit_tensor_friction_assembly_rejects_dimension_mismatch(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Un operador incompatible con el costo debe vetarse."""
        with pytest.raises(DimensionalIncompatibilityError):
            phase2._audit_tensor_friction_assembly(
                np.ones(3),
                np.ones(2),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 6.6. Puente terminal de Fase 2
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase2_bridge_requires_phase1_bridge(
        self,
        phase2: Phase2_TensorFrictionAuditor,
    ) -> None:
        """Fase 2 debe rechazar cualquier objeto que no sea Phase1TopologicalBridge."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase2._phase2_audit_and_bridge_to_phase3(object())

    def test_phase2_bridge_contains_friction_audit(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """El puente de Fase 2 debe contener la auditoría de fricción."""
        bridge = phase2._phase2_audit_and_bridge_to_phase3(valid_phase1_bridge)

        assert isinstance(bridge, Phase2FrictionBridge)
        assert bridge.phase1_bridge is valid_phase1_bridge
        assert isinstance(bridge.friction_audit, TensorFrictionData)
        assert bridge.friction_audit.is_positive_definite is True

    def test_phase2_bridge_rejects_singular_friction_operator(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """Un operador de fricción singular debe vetarse en Fase 2."""
        bad_bridge = dataclasses.replace(
            valid_phase1_bridge,
            friction_operator_F=np.zeros_like(valid_phase1_bridge.cost_vector_c),
        )

        with pytest.raises(ThermodynamicFrictionAnomaly):
            phase2._phase2_audit_and_bridge_to_phase3(bad_bridge)

    def test_phase2_bridge_is_immutable(
        self,
        valid_phase2_bridge: Phase2FrictionBridge,
    ) -> None:
        """El puente de Fase 2 debe ser inmutable."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            valid_phase2_bridge.friction_audit = None

    def test_phase2_terminal_bridge_is_phase3_initial_object(
        self,
        phase3: Phase3_RankNullityProjector,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₃ ∘ Φ₂:

        El puente terminal de Fase 2, cuando Fase 3 está presente,
        debe permitir la síntesis final de Fase 3.
        """
        phase2_bridge = phase3._phase2_audit_and_bridge_to_phase3(valid_phase1_bridge)
        state = phase3._phase3_finalize_from_phase2_bridge(phase2_bridge)

        assert isinstance(state, SemanticEstimatorAuditState)
        assert state.is_epistemologically_valid is True


# ══════════════════════════════════════════════════════════════════════════════
# §7. FASE 3 — RANGO-NULIDAD E ISOMETRÍA PARCIAL
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3RankNullityProjector:
    """
    Fase 3: Teorema de Rango-Nulidad y aislamiento ortogonal.

    Condiciones:
        1. rank(T) = 1.
        2. σ₁(T) = 1.
        3. Los proyectores inducidos son idempotentes y simétricos.

    El último test sintetiza el objeto final `SemanticEstimatorAuditState`.
    """

    def test_phase3_projector_is_instantiable(self) -> None:
        """Fase 3 debe poder instanciarse como extensión de Fase 2."""
        projector = Phase3_RankNullityProjector()
        assert isinstance(projector, Phase3_RankNullityProjector)
        assert isinstance(projector, Phase2_TensorFrictionAuditor)
        assert isinstance(projector, Phase1_TopologicalNeighborhoodCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.1. Descomposición SVD
    # ─────────────────────────────────────────────────────────────────────────

    def test_compute_svd_decomposition_rank1_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz rango 1 debe producir un valor singular dominante 1."""
        T = make_rank1_partial_isometry(m=5, n=3, seed=24)
        singular_values, U = phase3._compute_svd_decomposition(T)

        assert singular_values.size > 0
        assert singular_values[0] == pytest.approx(1.0, abs=1e-12)
        assert U.shape == (5, min(5, 3))

    # ─────────────────────────────────────────────────────────────────────────
    # 7.2. Determinación del rango efectivo
    # ─────────────────────────────────────────────────────────────────────────

    def test_determine_effective_rank_rank1_exact(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Valores singulares [1,0] deben certificar rank=1."""
        singular_values = np.array([1.0, 0.0])
        rank, kernel, tolerance = phase3._determine_effective_rank(
            singular_values,
            matrix_shape=(3, 2),
        )

        assert rank == 1
        assert kernel == 1
        assert tolerance >= _SVD_ABSOLUTE_TOLERANCE

    def test_determine_effective_rank_tiny_second_singular_is_ignored(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Un segundo valor singular por debajo de tolerancia no cuenta."""
        singular_values = np.array([1.0, 1e-12])
        rank, kernel, _ = phase3._determine_effective_rank(
            singular_values,
            matrix_shape=(3, 2),
        )

        assert rank == 1
        assert kernel == 1

    def test_determine_effective_rank_rejects_rank2(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Dos valores singulares significativos violan rank=1."""
        singular_values = np.array([1.0, 1.0])

        with pytest.raises(FunctorialityError):
            phase3._determine_effective_rank(singular_values, matrix_shape=(2, 2))

    def test_determine_effective_rank_rejects_rank0(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz nula no puede certificar rank=1."""
        singular_values = np.array([0.0, 0.0])

        with pytest.raises(FunctorialityError):
            phase3._determine_effective_rank(singular_values, matrix_shape=(2, 2))

    def test_determine_effective_rank_rejects_subnormal_dominant_singular(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Un valor singular dominante subnormal debe vetarse."""
        singular_values = np.array([1e-12])

        with pytest.raises(FunctorialityError):
            phase3._determine_effective_rank(singular_values, matrix_shape=(1, 1))

    # ─────────────────────────────────────────────────────────────────────────
    # 7.3. Verificación de isometría parcial
    # ─────────────────────────────────────────────────────────────────────────

    def test_verify_partial_isometry_unit_rank1_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una isometría parcial unitaria debe tener desviaciones nulas."""
        T = make_rank1_partial_isometry(m=4, n=3, seed=25)

        (
            sigma_deviation,
            row_sym_deviation,
            row_idempotence_deviation,
            col_sym_deviation,
            col_idempotence_deviation,
        ) = phase3._verify_partial_isometry(T, sigma_max=1.0)

        assert sigma_deviation == pytest.approx(0.0, abs=1e-12)
        assert row_sym_deviation <= 1e-10
        assert row_idempotence_deviation <= 1e-10
        assert col_sym_deviation <= 1e-10
        assert col_idempotence_deviation <= 1e-10

    def test_verify_partial_isometry_scaled_matrix_reports_sigma_deviation(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz 2T tiene σ₁=2 y debe reportar desviación σ=1."""
        T = 2.0 * make_rank1_partial_isometry(m=4, n=3, seed=26)

        (
            sigma_deviation,
            row_sym_deviation,
            row_idempotence_deviation,
            col_sym_deviation,
            col_idempotence_deviation,
        ) = phase3._verify_partial_isometry(T, sigma_max=2.0)

        assert sigma_deviation == pytest.approx(1.0)
        assert row_sym_deviation <= 1e-10
        assert row_idempotence_deviation <= 1e-10
        assert col_sym_deviation <= 1e-10
        assert col_idempotence_deviation <= 1e-10

    # ─────────────────────────────────────────────────────────────────────────
    # 7.4. Imposición de rango-nulidad
    # ─────────────────────────────────────────────────────────────────────────

    def test_enforce_rank_nullity_projection_valid_outer_product(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """T = u vᵀ con ||u||=||v||=1 debe certificar rango 1 e isometría."""
        T = make_rank1_partial_isometry(m=5, n=3, seed=27)
        audit = phase3._enforce_rank_nullity_projection(T)

        assert isinstance(audit, RankNullityProjectionData)
        assert audit.matrix_shape == (5, 3)
        assert audit.effective_rank == 1
        assert audit.kernel_dimension == 2
        assert audit.largest_singular_value == pytest.approx(1.0)
        assert audit.smallest_singular_value == pytest.approx(0.0)
        assert audit.is_orthogonal_injection is True
        assert math.isinf(audit.condition_number)

    def test_enforce_rank_nullity_projection_valid_column_vector(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Un vector columna unitario es una isometría parcial rango 1."""
        T = make_unit_vector(4, seed=28).reshape(-1, 1)
        audit = phase3._enforce_rank_nullity_projection(T)

        assert audit.matrix_shape == (4, 1)
        assert audit.effective_rank == 1
        assert audit.kernel_dimension == 0
        assert audit.largest_singular_value == pytest.approx(1.0)
        assert audit.is_orthogonal_injection is True

    def test_enforce_rank_nullity_projection_valid_row_vector(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Un vector fila unitario es una isometría parcial rango 1."""
        T = make_unit_vector(4, seed=29).reshape(1, -1)
        audit = phase3._enforce_rank_nullity_projection(T)

        assert audit.matrix_shape == (1, 4)
        assert audit.effective_rank == 1
        assert audit.kernel_dimension == 3
        assert audit.largest_singular_value == pytest.approx(1.0)
        assert audit.is_orthogonal_injection is True

    def test_enforce_rank_nullity_projection_valid_sparse_projector(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """diag(1,0) es un proyector ortogonal de rango 1."""
        T = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        audit = phase3._enforce_rank_nullity_projection(T)

        assert audit.effective_rank == 1
        assert audit.kernel_dimension == 1
        assert audit.is_orthogonal_injection is True

    def test_enforce_rank_nullity_projection_accepts_near_unit_scaling(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una desviación σ₁−1 ≤ tolerancia debe aceptarse."""
        T = (1.0 + 1e-9) * make_rank1_partial_isometry(m=4, n=3, seed=30)
        audit = phase3._enforce_rank_nullity_projection(T)

        assert audit.is_orthogonal_injection is True
        assert audit.orthogonality_deviation <= max(
            _ORTHOGONALITY_TOLERANCE,
            1e-8,
        )

    def test_enforce_rank_nullity_projection_rejects_zero_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """La matriz cero tiene rango 0 y viola el axioma rank=1."""
        with pytest.raises(FunctorialityError):
            phase3._enforce_rank_nullity_projection(np.zeros((2, 2)))

    def test_enforce_rank_nullity_projection_rejects_rank2_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz identidad 2×2 tiene rango 2 y debe vetarse."""
        with pytest.raises(FunctorialityError):
            phase3._enforce_rank_nullity_projection(np.eye(2))

    def test_enforce_rank_nullity_projection_rejects_scaled_isometry(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una isometría escalada por 2 viola σ₁=1."""
        T = 2.0 * make_rank1_partial_isometry(m=4, n=3, seed=31)

        with pytest.raises(ProjectorIntegrityError):
            phase3._enforce_rank_nullity_projection(T)

    def test_enforce_rank_nullity_projection_rejects_excessive_sigma_deviation(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una desviación σ₁−1 mayor que la tolerancia debe vetarse."""
        T = (1.0 + 1e-7) * make_rank1_partial_isometry(m=4, n=3, seed=32)

        with pytest.raises(ProjectorIntegrityError):
            phase3._enforce_rank_nullity_projection(T)

    def test_enforce_rank_nullity_projection_rejects_nonfinite_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz no finita viola la integridad de dominio."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase3._enforce_rank_nullity_projection(np.array([[np.nan]]))

    def test_enforce_rank_nullity_projection_rejects_empty_matrix(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Una matriz vacía viola la integridad matricial."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase3._enforce_rank_nullity_projection(np.empty((0, 2)))

    # ─────────────────────────────────────────────────────────────────────────
    # 7.5. Finalización funtorial de Fase 3
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase3_finalize_requires_phase2_bridge(
        self,
        phase3: Phase3_RankNullityProjector,
    ) -> None:
        """Fase 3 debe rechazar cualquier objeto que no sea Phase2FrictionBridge."""
        with pytest.raises(SemanticEstimatorAgentError):
            phase3._phase3_finalize_from_phase2_bridge(object())

    def test_phase3_finalize_valid_returns_state(
        self,
        phase3: Phase3_RankNullityProjector,
        valid_phase2_bridge: Phase2FrictionBridge,
    ) -> None:
        """Un puente válido debe sintetizar un estado epistemológicamente válido."""
        state = phase3._phase3_finalize_from_phase2_bridge(valid_phase2_bridge)

        assert isinstance(state, SemanticEstimatorAuditState)
        assert state.is_epistemologically_valid is True
        assert state.neighborhood_audit.is_homotopically_valid is True
        assert state.friction_audit.is_positive_definite is True
        assert state.projection_audit.is_orthogonal_injection is True

    def test_phase3_finalize_metadata_is_complete(
        self,
        phase3: Phase3_RankNullityProjector,
        valid_phase2_bridge: Phase2FrictionBridge,
    ) -> None:
        """La metadata de gobernanza debe contener las tres fases."""
        state = phase3._phase3_finalize_from_phase2_bridge(valid_phase2_bridge)
        meta = state.governance_metadata

        expected_keys = {
            "functor_composition",
            "phase1_cosine_similarity",
            "phase1_angle_degrees",
            "phase2_condition_number",
            "phase2_operator_type",
            "phase3_effective_rank",
            "phase3_kernel_dimension",
            "phase3_orthogonality_deviation",
        }

        assert expected_keys.issubset(meta.keys())
        assert meta["functor_composition"] == "Φ₃ ∘ Φ₂ ∘ Φ₁"
        assert meta["phase3_effective_rank"] == 1

    def test_phase3_finalize_rejects_rank2_injection(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        phase3: Phase3_RankNullityProjector,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """Una matriz de inyección rango 2 debe vetarse en Fase 3."""
        bad_phase1_bridge = dataclasses.replace(
            valid_phase1_bridge,
            injection_matrix_T=np.eye(2, dtype=np.float64),
        )
        bad_phase2_bridge = phase2._phase2_audit_and_bridge_to_phase3(bad_phase1_bridge)

        with pytest.raises(FunctorialityError):
            phase3._phase3_finalize_from_phase2_bridge(bad_phase2_bridge)

    def test_phase3_finalize_rejects_scaled_injection(
        self,
        phase2: Phase2_TensorFrictionAuditor,
        phase3: Phase3_RankNullityProjector,
        valid_phase1_bridge: Phase1TopologicalBridge,
    ) -> None:
        """Una matriz de inyección escalada debe violar la isometría parcial."""
        scaled_T = 2.0 * make_rank1_partial_isometry(
            m=valid_phase1_bridge.injection_matrix_T.shape[0],
            n=valid_phase1_bridge.injection_matrix_T.shape[1],
            seed=33,
        )

        bad_phase1_bridge = dataclasses.replace(
            valid_phase1_bridge,
            injection_matrix_T=scaled_T,
        )
        bad_phase2_bridge = phase2._phase2_audit_and_bridge_to_phase3(bad_phase1_bridge)

        with pytest.raises(ProjectorIntegrityError):
            phase3._phase3_finalize_from_phase2_bridge(bad_phase2_bridge)

    def test_semantic_estimator_audit_state_is_immutable(
        self,
        phase3: Phase3_RankNullityProjector,
        valid_phase2_bridge: Phase2FrictionBridge,
    ) -> None:
        """El objeto terminal del endofuntor debe ser inmutable."""
        state = phase3._phase3_finalize_from_phase2_bridge(valid_phase2_bridge)

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_epistemologically_valid = False


# ══════════════════════════════════════════════════════════════════════════════
# §8. ORQUESTADOR SUPREMO — SEMANTIC ESTIMATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticEstimatorAgentEndToEnd:
    """
    Orquestación completa:

        Z_EstimatorAgent = Φ₃ ∘ Φ₂ ∘ Φ₁

    Estos tests validan el diagrama conmutativo completo, los vetos
    categóricos y la inmutabilidad del estado terminal.
    """

    def test_agent_is_phase3_subclass_and_morphism_if_available(self) -> None:
        """El agente debe componer todas las fases y ser morfismo si aplica."""
        agent_instance = SemanticEstimatorAgent()

        assert isinstance(agent_instance, Phase3_RankNullityProjector)
        assert isinstance(agent_instance, Phase2_TensorFrictionAuditor)
        assert isinstance(agent_instance, Phase1_TopologicalNeighborhoodCertifier)

        if Morphism is not None:
            assert isinstance(agent_instance, Morphism)

    def test_execute_valid_semantic_governance_diagonal_friction(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """El endofuntor completo debe certificar validez con fricción diagonal."""
        state = agent.execute_semantic_estimation_governance(**valid_inputs)

        assert isinstance(state, SemanticEstimatorAuditState)
        assert state.is_epistemologically_valid is True

        assert state.neighborhood_audit.is_homotopically_valid is True
        assert state.friction_audit.is_positive_definite is True
        assert state.friction_audit.operator_type == "diagonal"
        assert state.projection_audit.effective_rank == 1
        assert state.projection_audit.is_orthogonal_injection is True

    def test_execute_valid_semantic_governance_matrix_friction(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """El endofuntor completo debe certificar validez con fricción matricial."""
        inputs = dict(valid_inputs)
        inputs["friction_operator_F"] = np.eye(
            valid_inputs["cost_vector_c"].size,
            dtype=np.float64,
        )

        state = agent.execute_semantic_estimation_governance(**inputs)

        assert state.is_epistemologically_valid is True
        assert state.friction_audit.operator_type == "matricial"
        assert state.friction_audit.condition_number == pytest.approx(1.0)

    def test_callable_alias_returns_state(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """__call__ debe ser un alias del endofuntor de gobernanza."""
        state = agent(**valid_inputs)
        assert isinstance(state, SemanticEstimatorAuditState)
        assert state.is_epistemologically_valid is True

    def test_execute_rejects_low_similarity(
        self,
        agent: SemanticEstimatorAgent,
    ) -> None:
        """cos(θ) < τ_min debe detonar TopologicalMappingError."""
        inputs = make_valid_agent_inputs(cosine=0.5)

        with pytest.raises(TopologicalMappingError):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_degenerate_query(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Una consulta degenerada debe detonar VectorDegeneracyError."""
        inputs = dict(valid_inputs)
        inputs["query_vector"] = np.zeros_like(valid_inputs["query_vector"])

        with pytest.raises(VectorDegeneracyError):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_dimension_mismatch(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Vectores de dimensión distinta deben detonar incompatibilidad."""
        inputs = dict(valid_inputs)
        inputs["retrieved_vector"] = np.ones(
            valid_inputs["query_vector"].size + 1,
            dtype=np.float64,
        )

        with pytest.raises(DimensionalIncompatibilityError):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_negative_cost(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Un costo negativo macroscópico debe detonar anomalía termodinámica."""
        inputs = dict(valid_inputs)
        cost = valid_inputs["cost_vector_c"].copy()
        cost[0] = -1.0
        inputs["cost_vector_c"] = cost

        with pytest.raises(ThermodynamicFrictionAnomaly):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_zero_friction_operator(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Un operador de fricción nulo debe detonar anomalía termodinámica."""
        inputs = dict(valid_inputs)
        inputs["friction_operator_F"] = np.zeros_like(
            valid_inputs["cost_vector_c"],
            dtype=np.float64,
        )

        with pytest.raises(ThermodynamicFrictionAnomaly):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_high_condition_friction_operator(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Un operador mal condicionado debe detonar anomalía termodinámica."""
        inputs = dict(valid_inputs)
        friction = np.ones_like(valid_inputs["cost_vector_c"], dtype=np.float64)
        friction[0] = 1e-6
        inputs["friction_operator_F"] = friction

        with pytest.raises(ThermodynamicFrictionAnomaly):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_rank2_injection_matrix(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Una matriz de inyección rango 2 debe detonar FunctorialityError."""
        inputs = dict(valid_inputs)
        inputs["injection_matrix_T"] = np.eye(2, dtype=np.float64)

        with pytest.raises(FunctorialityError):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_execute_rejects_scaled_injection_matrix(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """Una matriz de inyección escalada debe violar la isometría parcial."""
        inputs = dict(valid_inputs)
        inputs["injection_matrix_T"] = 2.0 * make_rank1_partial_isometry(
            m=valid_inputs["injection_matrix_T"].shape[0],
            n=valid_inputs["injection_matrix_T"].shape[1],
            seed=34,
        )

        with pytest.raises(ProjectorIntegrityError):
            agent.execute_semantic_estimation_governance(**inputs)

    def test_governance_metadata_is_complete_and_finite(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """La metadata de gobernanza debe ser completa y finita."""
        state = agent.execute_semantic_estimation_governance(**valid_inputs)
        meta = state.governance_metadata

        expected_keys = {
            "functor_composition",
            "phase1_cosine_similarity",
            "phase1_angle_degrees",
            "phase2_condition_number",
            "phase2_operator_type",
            "phase3_effective_rank",
            "phase3_kernel_dimension",
            "phase3_orthogonality_deviation",
        }

        assert expected_keys.issubset(meta.keys())
        assert meta["functor_composition"] == "Φ₃ ∘ Φ₂ ∘ Φ₁"
        assert meta["phase2_operator_type"] in {"diagonal", "matricial"}
        assert meta["phase3_effective_rank"] == 1

        numeric_keys = (
            "phase1_cosine_similarity",
            "phase1_angle_degrees",
            "phase2_condition_number",
            "phase3_effective_rank",
            "phase3_kernel_dimension",
            "phase3_orthogonality_deviation",
        )

        for key in numeric_keys:
            assert np.isfinite(float(meta[key])), f"Metadata no finita en {key}"

    def test_semantic_estimator_audit_state_is_immutable_end_to_end(
        self,
        agent: SemanticEstimatorAgent,
        valid_inputs: Dict[str, Any],
    ) -> None:
        """El objeto terminal del endofuntor debe ser inmutable."""
        state = agent.execute_semantic_estimation_governance(**valid_inputs)

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_epistemologically_valid = False

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.governance_metadata = {}