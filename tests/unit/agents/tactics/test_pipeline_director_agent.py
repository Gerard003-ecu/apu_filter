# Archivo: tests/unit/agents/tactics/test_pipeline_director_agent.py
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Batería de pruebas unitarias para:                                                      ║
║  app/agents/tactics/pipeline_director_agent.py                                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ORGANIZACIÓN POR FASES ANIDADAS:                                                        ║
║  ────────────────────────────────                                                        ║
║  Fase 1 → Certificación espectral de nilpotencia (Schur, radio espectral, índice ν).    ║
║           El último test valida que el puente terminal de Fase 1 es consumido por        ║
║           Fase 2 cuando el mezclador (mixin) correcto está presente.                     ║
║                                                                                          ║
║  Fase 2 → Auditoría de filtración Poset DIKW (monotonicidad, auto-bucles, slacks).      ║
║           El último test valida que el puente terminal de Fase 2 es consumido por        ║
║           Fase 3 cuando el mezclador correcto está presente.                             ║
║                                                                                          ║
║  Fase 3 → Intercepción homológica de Mayer-Vietoris (Δβ₁, Euler-Poincaré, H₀, JS).      ║
║           El último test sintetiza el objeto final `CausalGovernanceState`.              ║
║                                                                                          ║
║  Orquestación → `PipelineDirectorAgent` como composición terminal Z_Causal.              ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

pytest.importorskip("numpy", reason="El módulo auditado requiere NumPy.")

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# §0. CARGA ROBUSTA DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

_TARGET_MODULE_NAME = "app.agents.tactics.pipeline_director_agent"
_TARGET_REL_PATH = Path("app") / "agents" / "tactics" / "pipeline_director_agent.py"


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
PipelineDirectorAgentError = target.PipelineDirectorAgentError
AdjacencyMatrixFormatError = target.AdjacencyMatrixFormatError
CausalLoopVetoError = target.CausalLoopVetoError
NilpotenceIndexVetoError = target.NilpotenceIndexVetoError
StratumMappingError = target.StratumMappingError
SelfLoopVetoError = target.SelfLoopVetoError
FiltrationViolationVeto = target.FiltrationViolationVeto
AdjacencySupportVetoError = target.AdjacencySupportVetoError
MayerVietorisInputError = target.MayerVietorisInputError
HomologicalFusionVeto = target.HomologicalFusionVeto
EulerPoincareMismatchError = target.EulerPoincareMismatchError

# DTOs.
NilpotenceAuditData = target.NilpotenceAuditData
PosetFiltrationData = target.PosetFiltrationData
MayerVietorisAuditData = target.MayerVietorisAuditData
CausalGovernanceState = target.CausalGovernanceState

# Clases de fase.
Phase1_SpectralNilpotenceCertifier = target.Phase1_SpectralNilpotenceCertifier
Phase2_PosetFiltrationAuditor = target.Phase2_PosetFiltrationAuditor
Phase3_MayerVietorisInterceptor = target.Phase3_MayerVietorisInterceptor

# Orquestador supremo.
PipelineDirectorAgent = target.PipelineDirectorAgent

# Entidades categóricas opcionales.
TopologicalInvariantError = getattr(target, "TopologicalInvariantError", None)
Morphism = getattr(target, "Morphism", None)

# Constantes y flags internos.
_MACHINE_EPSILON = float(getattr(target, "_MACHINE_EPSILON", np.finfo(np.float64).eps))
_BASE_SPECTRAL_TOLERANCE = float(getattr(target, "_BASE_SPECTRAL_TOLERANCE", 1e-10))
_BASE_POWER_TOLERANCE = float(getattr(target, "_BASE_POWER_TOLERANCE", 1e-8))
_DEFAULT_POWER_AUDIT_MAX_DIM = int(getattr(target, "_DEFAULT_POWER_AUDIT_MAX_DIM", 96))
_HAS_SCIPY = bool(getattr(target, "_HAS_SCIPY", False))


# ══════════════════════════════════════════════════════════════════════════════
# §2. FÁBRICAS DE DATOS DETERMINISTAS
# ══════════════════════════════════════════════════════════════════════════════


def make_upper_triangular_dag(n: int = 3, weight: float = 1.0) -> np.ndarray:
    """
    Construye una matriz de adyacencia nilpotente de un DAG cadena:
        v0 → v1 → ... → v_{n-1}
    """
    if n <= 0:
        raise ValueError("n debe ser positivo.")

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = float(weight)
    return A


def make_cycle_adjacency(n: int = 2) -> np.ndarray:
    """Construye una matriz de adyacencia con ciclo dirigido."""
    if n <= 0:
        raise ValueError("n debe ser positivo.")

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
    return A


def make_path_edges(n: int = 3) -> list[Tuple[str, str]]:
    """Aristas de una cadena DAG v0 → v1 → ... → v_{n-1}."""
    return [(f"v{i}", f"v{i + 1}") for i in range(n - 1)]


def make_path_strata(n: int = 3) -> Dict[str, int]:
    """Estratos DIKW estrictamente crecientes para la cadena."""
    return {f"v{i}": i for i in range(n)}


def make_path_node_order(n: int = 3) -> list[str]:
    """Orden canónico de nodos para la cadena."""
    return [f"v{i}" for i in range(n)]


def make_valid_betti() -> Dict[str, int]:
    """
    Números de Betti en H₁ válidos bajo el contrato aditivo:
        β₁(A∪B) = β₁(A) + β₁(B) − β₁(A∩B) = 1 + 1 − 1 = 1.
    """
    return {
        "betti_1_A": 1,
        "betti_1_B": 1,
        "betti_1_intersection": 1,
        "betti_1_union": 1,
    }


def make_valid_betti0() -> Dict[str, int]:
    """
    Números de Betti en H₀ consistentes con:
        β₀(A∪B) = β₀(A) + β₀(B) − β₀(A∩B) + rank(∂)
    con rank(∂)=0 por contrato aditivo.
    """
    return {
        "betti_0_A": 2,
        "betti_0_B": 2,
        "betti_0_intersection": 1,
        "betti_0_union": 3,
    }


def make_nilpotence_audit(
    is_strictly_nilpotent: bool = True,
    dimension: int = 2,
) -> NilpotenceAuditData:
    """Construye un certificado de Fase 1 sintético para tests de Fase 2."""
    return NilpotenceAuditData(
        dimension=dimension,
        spectral_radius=0.0 if is_strictly_nilpotent else 1.0,
        tolerance=_BASE_SPECTRAL_TOLERANCE,
        adjacency_inf_norm=1.0,
        frobenius_norm=1.0,
        nonzero_entries=1,
        directed_density=0.5,
        is_strictly_nilpotent=is_strictly_nilpotent,
    )


def make_valid_agent_kwargs(n: int = 3, **overrides: Any) -> Dict[str, Any]:
    """
    Construye insumos completos válidos para el orquestador supremo.

    Por defecto:
        - DAG cadena nilpotente.
        - Aristas soportadas por la matriz.
        - Filtración DIKW estricta.
        - Betti H₁ y H₀ consistentes.
    """
    kwargs: Dict[str, Any] = {
        "adjacency_matrix": make_upper_triangular_dag(n, weight=1.0),
        "edges": make_path_edges(n),
        "node_strata": make_path_strata(n),
        "node_order": make_path_node_order(n),
        "adjacency_zero_threshold": _BASE_SPECTRAL_TOLERANCE,
        "allow_signed_weights": False,
        "require_boolean": True,
        "deep_nilpotence_audit": True,
        "power_audit_max_dim": _DEFAULT_POWER_AUDIT_MAX_DIM,
        "allow_unknown_nodes": False,
        "strict_filtration": True,
        "cross_check_adjacency_support": True,
        "raise_on_veto": True,
    }

    kwargs.update(make_valid_betti())
    kwargs.update(make_valid_betti0())
    kwargs.update(overrides)
    return kwargs


class _EnumLike:
    """Objeto minimal con atributo `.value` para tests de coerción de estratos."""

    def __init__(self, value: Any) -> None:
        self.value = value


# ══════════════════════════════════════════════════════════════════════════════
# §3. FIXTURES POR FASES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def phase1() -> Phase1_SpectralNilpotenceCertifier:
    """Instancia fresca del certificador de Fase 1."""
    return Phase1_SpectralNilpotenceCertifier()


@pytest.fixture(scope="function")
def phase2() -> Phase2_PosetFiltrationAuditor:
    """Instancia fresca del auditor de Fase 2."""
    return Phase2_PosetFiltrationAuditor()


@pytest.fixture(scope="function")
def phase3() -> Phase3_MayerVietorisInterceptor:
    """Instancia fresca del interceptor de Fase 3."""
    return Phase3_MayerVietorisInterceptor()


@pytest.fixture(scope="function")
def agent() -> PipelineDirectorAgent:
    """Instancia fresca del orquestador supremo."""
    return PipelineDirectorAgent()


@pytest.fixture(scope="function")
def dag_inputs() -> Dict[str, Any]:
    """Insumos básicos de DAG cadena para puentes entre fases."""
    n = 3
    return {
        "adjacency_matrix": make_upper_triangular_dag(n),
        "edges": make_path_edges(n),
        "node_strata": make_path_strata(n),
        "node_order": make_path_node_order(n),
    }


@pytest.fixture(scope="function")
def valid_betti() -> Dict[str, int]:
    """Números de Betti H₁ válidos."""
    return make_valid_betti()


@pytest.fixture(scope="function")
def valid_betti0() -> Dict[str, int]:
    """Números de Betti H₀ válidos."""
    return make_valid_betti0()


# ══════════════════════════════════════════════════════════════════════════════
# §4. CONTRATO DEL MÓDULO Y TAXONOMÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleContractAndExceptionTaxonomy:
    """Contrato estructural del módulo y jerarquía de excepciones."""

    def test_module_exposes_core_types(self) -> None:
        """El módulo debe exponer todos los tipos públicos principales."""
        core_names = (
            "PipelineDirectorAgentError",
            "AdjacencyMatrixFormatError",
            "CausalLoopVetoError",
            "NilpotenceIndexVetoError",
            "StratumMappingError",
            "SelfLoopVetoError",
            "FiltrationViolationVeto",
            "AdjacencySupportVetoError",
            "MayerVietorisInputError",
            "HomologicalFusionVeto",
            "EulerPoincareMismatchError",
            "NilpotenceAuditData",
            "PosetFiltrationData",
            "MayerVietorisAuditData",
            "CausalGovernanceState",
            "Phase1_SpectralNilpotenceCertifier",
            "Phase2_PosetFiltrationAuditor",
            "Phase3_MayerVietorisInterceptor",
            "PipelineDirectorAgent",
        )

        for name in core_names:
            assert hasattr(target, name), f"El módulo no expone el tipo requerido: {name}"

    def test_phase_class_hierarchy_is_nested(self) -> None:
        """La jerarquía de clases debe reflejar la composición funtorial."""
        assert issubclass(Phase2_PosetFiltrationAuditor, Phase1_SpectralNilpotenceCertifier)
        assert issubclass(Phase3_MayerVietorisInterceptor, Phase2_PosetFiltrationAuditor)
        assert issubclass(PipelineDirectorAgent, Phase3_MayerVietorisInterceptor)

    def test_agent_is_morphism_if_morphism_exists(self) -> None:
        """El agente debe ser un morfismo si el topo MIC define `Morphism`."""
        if Morphism is None:
            pytest.skip("El módulo auditado no expone Morphism.")

        assert issubclass(PipelineDirectorAgent, Morphism)

    @pytest.mark.parametrize(
        "exc",
        [
            AdjacencyMatrixFormatError,
            CausalLoopVetoError,
            NilpotenceIndexVetoError,
            StratumMappingError,
            SelfLoopVetoError,
            FiltrationViolationVeto,
            AdjacencySupportVetoError,
            MayerVietorisInputError,
            HomologicalFusionVeto,
            EulerPoincareMismatchError,
        ],
    )
    def test_exceptions_are_rooted_in_pipeline_director_error(self, exc) -> None:
        """Toda excepción de gobernanza debe descender de PipelineDirectorAgentError."""
        assert issubclass(exc, PipelineDirectorAgentError)

    def test_root_exception_inherits_topological_invariant_error_if_available(self) -> None:
        """Si existe TopologicalInvariantError, la raíz debe heredarla."""
        if TopologicalInvariantError is None:
            pytest.skip("TopologicalInvariantError no está disponible en el módulo.")

        assert issubclass(PipelineDirectorAgentError, TopologicalInvariantError)

    def test_utc_timestamp_is_iso8601_utc(self) -> None:
        """El timestamp de gobernanza debe ser UTC ISO-8601."""
        utc_timestamp = getattr(target, "_utc_timestamp", None)
        if utc_timestamp is None:
            pytest.skip("_utc_timestamp no está disponible.")

        ts = utc_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts
        assert ts.endswith("+00:00")


# ══════════════════════════════════════════════════════════════════════════════
# §5. FASE 1 — CERTIFICACIÓN ESPECTRAL DE NILPOTENCIA
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1SpectralNilpotenceCertifier:
    """
    Fase 1: Certificación espectral de aciclicidad.

    Objetivo:
        Spec(A) = {0}  ⇔  ρ(A) = 0  ⇔  A es nilpotente.

    El último test valida que el puente terminal de Fase 1 se convierte
    en el morfismo inicial de Fase 2 cuando el mixin de Fase 2 está presente.
    """

    def test_phase1_certifier_is_instantiable(self) -> None:
        """Fase 1 debe poder instanciarse sin estado mutable."""
        certifier = Phase1_SpectralNilpotenceCertifier()
        assert isinstance(certifier, Phase1_SpectralNilpotenceCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.1. Validación y acondicionamiento de la matriz de adyacencia
    # ─────────────────────────────────────────────────────────────────────────

    def test_validate_accepts_square_finite_dag(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una matriz cuadrada finita debe ser aceptada."""
        A = make_upper_triangular_dag(3)
        out = phase1._validate_and_condition_adjacency(A)

        assert out.shape == (3, 3)
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    def test_validate_rejects_non_2d_operator(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un tensor 3D no es un endomorfismo y debe ser rechazado."""
        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(np.zeros((2, 2, 2)))

    def test_validate_rejects_non_square_matrix(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una matriz rectangular no define un grafo cerrado."""
        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(np.zeros((2, 3)))

    def test_validate_rejects_non_finite_entries(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """NaN/Inf violan el dominio del operador causal."""
        A = np.array([[0.0, np.nan], [0.0, 0.0]])

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(A)

    def test_validate_rejects_negative_zero_threshold(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un umbral negativo carece de interpretación espectral."""
        A = make_upper_triangular_dag(2)

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(A, zero_threshold=-1.0)

    def test_validate_thresholds_small_entries(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Entradas por debajo del umbral deben ser anuladas exactamente."""
        A = np.array([[0.0, 1e-12], [0.0, 0.0]])
        out = phase1._validate_and_condition_adjacency(A, zero_threshold=1e-10)

        assert np.count_nonzero(out) == 0

    def test_validate_rejects_negative_weights_by_default(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un DAG causal físico exige pesos no negativos por defecto."""
        A = np.array([[0.0, -1.0], [0.0, 0.0]])

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(A, allow_signed_weights=False)

    def test_validate_accepts_negative_weights_when_allowed(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Con allow_signed_weights=True se admiten pesos signados."""
        A = np.array([[0.0, -1.0], [0.0, 0.0]])
        out = phase1._validate_and_condition_adjacency(A, allow_signed_weights=True)

        assert out[0, 1] == pytest.approx(-1.0)

    def test_validate_require_boolean_rejects_nonbinary_weights(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """require_boolean=True exige A ∈ {0,1}^{n×n}."""
        A = np.array([[0.0, 2.0], [0.0, 0.0]])

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._validate_and_condition_adjacency(A, require_boolean=True)

    def test_validate_require_boolean_projects_to_binary(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Valores cercanos a 1 dentro de tolerancia deben binarizarse."""
        A = np.array([[0.0, 1.0 + 1e-12], [0.0, 0.0]])
        out = phase1._validate_and_condition_adjacency(
            A,
            zero_threshold=1e-10,
            require_boolean=True,
        )

        assert out[0, 1] == pytest.approx(1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.2. Descomposición de Schur
    # ─────────────────────────────────────────────────────────────────────────

    def test_schur_decompose_returns_shapes_and_condition(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La descomposición de Schur debe retornar T, Q, κ₂(Q) y bloques."""
        A = make_upper_triangular_dag(3)
        T, Q, kappa_Q, n_blocks = phase1._schur_decompose(A)

        assert T.shape == (3, 3)
        assert Q.shape == (3, 3)
        assert kappa_Q >= 1.0 or np.isinf(kappa_Q)
        assert n_blocks >= 0

        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_schur_decompose_nilpotent_has_zero_spectral_radius(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una matriz nilpotente debe tener radio espectral extraído ≈ 0."""
        A = make_upper_triangular_dag(3)
        T, _, _, n_blocks = phase1._schur_decompose(A)
        rho = phase1._extract_spectral_radius_from_schur(T, n_blocks)

        assert rho <= max(_BASE_SPECTRAL_TOLERANCE, 1e-8)

    @pytest.mark.skipif(not _HAS_SCIPY, reason="Requiere scipy.linalg.schur.")
    def test_schur_decompose_rotation_block_spectral_radius(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un bloque 2×2 rotacional debe producir radio espectral 1."""
        A = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        T, _, _, n_blocks = phase1._schur_decompose(A)
        rho = phase1._extract_spectral_radius_from_schur(T, n_blocks)

        assert rho == pytest.approx(1.0, abs=1e-8)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3. Tolerancia espectral dinámica
    # ─────────────────────────────────────────────────────────────────────────

    def test_dynamic_spectral_tolerance_dimension_zero_returns_base(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Dimensión no positiva debe retornar la tolerancia base."""
        tol = phase1._dynamic_spectral_tolerance(0, 1.0, 1.0)
        assert tol == pytest.approx(_BASE_SPECTRAL_TOLERANCE)

    def test_dynamic_spectral_tolerance_lower_bound(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La tolerancia dinámica nunca debe ser menor que la base."""
        tol = phase1._dynamic_spectral_tolerance(2, 0.0, 1.0)
        assert tol >= _BASE_SPECTRAL_TOLERANCE

    def test_dynamic_spectral_tolerance_grows_with_frobenius_norm(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La tolerancia debe crecer con la norma de Frobenius."""
        tol_small = phase1._dynamic_spectral_tolerance(3, 1.0, 1.0)
        tol_large = phase1._dynamic_spectral_tolerance(3, 1e6, 1.0)

        assert tol_large > tol_small

    def test_dynamic_spectral_tolerance_grows_with_condition_number(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La tolerancia debe crecer con κ₂(Q)."""
        tol_well = phase1._dynamic_spectral_tolerance(3, 1.0, 1.0)
        tol_ill = phase1._dynamic_spectral_tolerance(3, 1.0, 100.0)

        assert tol_ill > tol_well

    # ─────────────────────────────────────────────────────────────────────────
    # 5.4. Radio espectral de Gelfand
    # ─────────────────────────────────────────────────────────────────────────

    def test_gelfand_zero_matrix_returns_zero(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La matriz nula tiene radio espectral 0."""
        A = np.zeros((2, 2), dtype=np.float64)
        rho = phase1._gelfand_spectral_radius_estimate(A, n_steps=5)

        assert rho == pytest.approx(0.0)

    def test_gelfand_nilpotent_matrix_returns_zero(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una matriz nilpotente debe decaer a radio 0 en iteración de potencia."""
        A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        rho = phase1._gelfand_spectral_radius_estimate(A, n_steps=20)

        assert rho == pytest.approx(0.0, abs=1e-12)

    def test_gelfand_identity_matrix_returns_one(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La identidad tiene radio espectral 1."""
        A = np.eye(2, dtype=np.float64)
        rho = phase1._gelfand_spectral_radius_estimate(A, n_steps=5)

        assert rho == pytest.approx(1.0, abs=1e-12)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.5. Índice de nilpotencia
    # ─────────────────────────────────────────────────────────────────────────

    def test_certify_nilpotence_index_zero_matrix(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La matriz nula tiene índice de nilpotencia ν=1."""
        A = np.zeros((2, 2), dtype=np.float64)
        residual, index, gelfand = phase1._certify_nilpotence_index(
            A,
            power_tolerance=_BASE_POWER_TOLERANCE,
            power_audit_max_dim=96,
        )

        assert residual == pytest.approx(0.0)
        assert index == 1
        assert gelfand is None

    def test_certify_nilpotence_index_path_dag(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una cadena de n nodos tiene índice de nilpotencia ν=n."""
        A = make_upper_triangular_dag(3)
        residual, index, gelfand = phase1._certify_nilpotence_index(
            A,
            power_tolerance=_BASE_POWER_TOLERANCE,
            power_audit_max_dim=96,
        )

        assert residual == pytest.approx(0.0, abs=1e-12)
        assert index == 3
        assert gelfand is None

    def test_certify_nilpotence_index_identity_no_index(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La identidad no es nilpotente y no debe certificar índice."""
        A = np.eye(2, dtype=np.float64)
        residual, index, gelfand = phase1._certify_nilpotence_index(
            A,
            power_tolerance=_BASE_POWER_TOLERANCE,
            power_audit_max_dim=96,
        )

        assert residual is not None
        assert residual > _BASE_POWER_TOLERANCE
        assert index is None
        assert gelfand is None

    def test_certify_nilpotence_index_large_uses_gelfand(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Si n > power_audit_max_dim, se usa estimación de Gelfand."""
        A = make_upper_triangular_dag(3)
        residual, index, gelfand = phase1._certify_nilpotence_index(
            A,
            power_tolerance=_BASE_POWER_TOLERANCE,
            power_audit_max_dim=2,
        )

        assert residual is None
        assert index is None
        assert gelfand is not None

    # ─────────────────────────────────────────────────────────────────────────
    # 5.6. Extracción del radio espectral desde Schur
    # ─────────────────────────────────────────────────────────────────────────

    def test_extract_spectral_radius_from_diagonal(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una T diagonal debe retornar el máximo módulo diagonal."""
        T = np.diag([0.0, 0.5, -2.0])
        rho = phase1._extract_spectral_radius_from_schur(T, n_blocks_2x2=0)

        assert rho == pytest.approx(2.0)

    def test_extract_spectral_radius_empty_matrix(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Una matriz vacía tiene radio espectral 0."""
        T = np.empty((0, 0), dtype=np.float64)
        rho = phase1._extract_spectral_radius_from_schur(T, n_blocks_2x2=0)

        assert rho == pytest.approx(0.0)

    def test_extract_spectral_radius_from_2x2_block(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un bloque 2×2 rotacional debe producir módulo 1."""
        T = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        rho = phase1._extract_spectral_radius_from_schur(T, n_blocks_2x2=1)

        assert rho == pytest.approx(1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.7. Certificación espectral principal
    # ─────────────────────────────────────────────────────────────────────────

    def test_certify_spectral_nilpotence_empty_matrix_trivially_nilpotent(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """El operador vacío es trivialmente nilpotente."""
        audit = phase1._certify_spectral_nilpotence(np.empty((0, 0)))

        assert audit.dimension == 0
        assert audit.is_strictly_nilpotent is True
        assert audit.nilpotence_index == 0
        assert audit.power_audited is True
        assert audit.spectral_radius == pytest.approx(0.0)

    def test_certify_spectral_nilpotence_nilpotent_dag(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un DAG cadena debe ser certificado como nilpotente."""
        A = make_upper_triangular_dag(3)
        audit = phase1._certify_spectral_nilpotence(A)

        assert audit.is_strictly_nilpotent is True
        assert audit.spectral_radius <= audit.tolerance
        assert audit.dimension == 3
        assert audit.nonzero_entries == 2

    def test_certify_spectral_nilpotence_directed_density(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La densidad dirigida debe contar aristas fuera de la diagonal."""
        A = make_upper_triangular_dag(3)
        audit = phase1._certify_spectral_nilpotence(A)

        # 2 aristas fuera de diagonal sobre 3*2 = 6 posibles.
        assert audit.directed_density == pytest.approx(2.0 / 6.0)

    def test_certify_spectral_nilpotence_cycle_raises_causal_loop(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Un ciclo dirigido debe detonar CausalLoopVetoError."""
        A = make_cycle_adjacency(2)

        with pytest.raises(CausalLoopVetoError):
            phase1._certify_spectral_nilpotence(A, raise_on_veto=True)

    def test_certify_spectral_nilpotence_cycle_no_raise_returns_not_nilpotent(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Con raise_on_veto=False, un ciclo retorna certificado inválido."""
        A = make_cycle_adjacency(2)
        audit = phase1._certify_spectral_nilpotence(A, raise_on_veto=False)

        assert audit.is_strictly_nilpotent is False
        assert audit.spectral_radius > audit.tolerance

    def test_certify_spectral_nilpotence_negative_weights_raise(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Pesos negativos sin permiso deben violar el formato de adyacencia."""
        A = np.array([[0.0, -1.0], [0.0, 0.0]])

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._certify_spectral_nilpotence(A, allow_signed_weights=False)

    def test_certify_spectral_nilpotence_require_boolean_rejects_weights(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """require_boolean=True debe rechazar pesos distintos de 1."""
        A = np.array([[0.0, 2.0], [0.0, 0.0]])

        with pytest.raises(AdjacencyMatrixFormatError):
            phase1._certify_spectral_nilpotence(A, require_boolean=True)

    def test_certify_spectral_nilpotence_deep_audit_index(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """La auditoría profunda debe certificar el índice de nilpotencia."""
        A = make_upper_triangular_dag(3)
        audit = phase1._certify_spectral_nilpotence(
            A,
            deep_nilpotence_audit=True,
            power_audit_max_dim=96,
        )

        assert audit.power_audited is True
        assert audit.power_residual == pytest.approx(0.0, abs=1e-12)
        assert audit.nilpotence_index == 3

    def test_certify_spectral_nilpotence_large_dim_uses_gelfand_estimate(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """Para n > max_dim, la auditoría profunda debe usar Gelfand."""
        A = make_upper_triangular_dag(3)
        audit = phase1._certify_spectral_nilpotence(
            A,
            deep_nilpotence_audit=True,
            power_audit_max_dim=2,
            raise_on_veto=True,
        )

        assert audit.power_audited is False
        assert audit.power_residual is None
        assert audit.nilpotence_index is None
        assert audit.gelfand_radius_estimate is not None

    # ─────────────────────────────────────────────────────────────────────────
    # 5.8. Puente terminal de Fase 1
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase1_stub_audit_poset_raises_not_implemented(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
    ) -> None:
        """El stub de Fase 2 debe exigir el mixin de Fase 2."""
        audit = make_nilpotence_audit(True)

        with pytest.raises(NotImplementedError):
            phase1._audit_poset_filtration_from_nilpotence(audit, [], {})

    def test_phase1_terminal_bridge_requires_phase2_mixin(
        self,
        phase1: Phase1_SpectralNilpotenceCertifier,
        dag_inputs: Dict[str, Any],
    ) -> None:
        """El puente terminal de Fase 1 falla si Fase 2 no está mezclada."""
        with pytest.raises(NotImplementedError):
            phase1._phase1_terminal_bridge_to_phase2(
                dag_inputs["adjacency_matrix"],
                dag_inputs["edges"],
                dag_inputs["node_strata"],
            )

    def test_phase1_terminal_bridge_is_phase2_initial_object(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        dag_inputs: Dict[str, Any],
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₂ ∘ Φ₁:

        El puente terminal de Fase 1, cuando Fase 2 está presente,
        produce el certificado de Fase 1 y el certificado inicial de Fase 2.
        """
        nilpotence_audit, filtration_audit = (
            phase2._phase1_terminal_bridge_to_phase2(
                dag_inputs["adjacency_matrix"],
                dag_inputs["edges"],
                dag_inputs["node_strata"],
                node_order=dag_inputs["node_order"],
            )
        )

        assert isinstance(nilpotence_audit, NilpotenceAuditData)
        assert isinstance(filtration_audit, PosetFiltrationData)
        assert nilpotence_audit.is_strictly_nilpotent is True
        assert filtration_audit.is_monotonic_filtration is True


# ══════════════════════════════════════════════════════════════════════════════
# §6. FASE 2 — AUDITORÍA DE FILTRACIÓN POSET DIKW
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2PosetFiltrationAuditor:
    """
    Fase 2: Auditoría de monotonicidad categórica DIKW.

    Objetivo:
        stratum(u) ≤ stratum(v) para toda arista u → v.

    El último test valida que el puente terminal de Fase 2 se convierte
    en el morfismo inicial de Fase 3 cuando el mixin de Fase 3 está presente.
    """

    def test_phase2_auditor_is_instantiable(self) -> None:
        """Fase 2 debe poder instanciarse como extensión de Fase 1."""
        auditor = Phase2_PosetFiltrationAuditor()
        assert isinstance(auditor, Phase2_PosetFiltrationAuditor)
        assert isinstance(auditor, Phase1_SpectralNilpotenceCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.1. Coerción de estratos
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_coerce_stratum_level_rejects_booleans(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        value: Any,
    ) -> None:
        """Los booleanos son semánticamente ambiguos para estratos DIKW."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", value)

    @pytest.mark.parametrize("value, expected", [(0, 0), (3, 3), (np.int64(2), 2)])
    def test_coerce_stratum_level_accepts_integers(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        value: Any,
        expected: int,
    ) -> None:
        """Enteros nativos y NumPy deben convertirse directamente."""
        assert phase2._coerce_stratum_level("node", value) == expected

    def test_coerce_stratum_level_accepts_enum_like_value(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Objetos con `.value` entera deben ser aceptados."""
        assert phase2._coerce_stratum_level("node", _EnumLike(2)) == 2

    def test_coerce_stratum_level_rejects_invalid_enum_value(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Objetos con `.value` no convertible deben ser rechazados."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", _EnumLike(None))

    def test_coerce_stratum_level_accepts_integral_float(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Floats finitos e integrales deben convertirse a int."""
        assert phase2._coerce_stratum_level("node", 2.0) == 2

    def test_coerce_stratum_level_rejects_non_integral_float(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Floats no integrales deben ser rechazados."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", 2.7)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_coerce_stratum_level_rejects_non_finite_floats(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        value: float,
    ) -> None:
        """Floats no finitos deben ser rechazados."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", value)

    @pytest.mark.parametrize("value, expected", [("2", 2), (" 3 ", 3), ("2.0", 2)])
    def test_coerce_stratum_level_accepts_numeric_strings(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        value: str,
        expected: int,
    ) -> None:
        """Strings numéricos deben parsearse correctamente."""
        assert phase2._coerce_stratum_level("node", value) == expected

    def test_coerce_stratum_level_rejects_non_numeric_string(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Strings no numéricos deben ser rechazados."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", "knowledge")

    def test_coerce_stratum_level_rejects_arbitrary_object(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Objetos sin conversión válida deben ser rechazados."""
        with pytest.raises(StratumMappingError):
            phase2._coerce_stratum_level("node", object())

    # ─────────────────────────────────────────────────────────────────────────
    # 6.2. Normalización de aristas
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_edges_rejects_none(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """edges=None viola el contrato de secuencia."""
        with pytest.raises(PipelineDirectorAgentError):
            phase2._normalize_edges(None)

    def test_normalize_edges_rejects_invalid_edge_format(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una arista no desempaquetable como (u, v) debe ser rechazada."""
        with pytest.raises(PipelineDirectorAgentError):
            phase2._normalize_edges([123])

    def test_normalize_edges_rejects_triple_edge(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una tupla de longitud 3 no es una arista dirigida válida."""
        with pytest.raises(PipelineDirectorAgentError):
            phase2._normalize_edges([("a", "b", "c")])

    def test_normalize_edges_rejects_empty_node_identifier(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Identificadores vacíos deben ser rechazados."""
        with pytest.raises(PipelineDirectorAgentError):
            phase2._normalize_edges([("   ", "b")])

    def test_normalize_edges_rejects_self_loop_by_default(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Un auto-bucle (u,u) debe detonar SelfLoopVetoError."""
        with pytest.raises(SelfLoopVetoError):
            phase2._normalize_edges([("a", "a")], raise_on_self_loop=True)

    def test_normalize_edges_counts_self_loop_when_veto_disabled(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con veto desactivado, los auto-bucles se cuentan y se excluyen."""
        normalized, self_loops = phase2._normalize_edges(
            [("a", "a"), ("a", "b")],
            raise_on_self_loop=False,
        )

        assert normalized == [("a", "b")]
        assert self_loops == 1

    def test_normalize_edges_accepts_non_string_nodes_by_stringification(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Nodos no string deben convertirse mediante str()."""
        normalized, self_loops = phase2._normalize_edges([(1, 2)])

        assert normalized == [("1", "2")]
        assert self_loops == 0

    # ─────────────────────────────────────────────────────────────────────────
    # 6.3. Normalización de estratos
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("value", [None, {}])
    def test_normalize_node_strata_empty_returns_empty_dict(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        value: Any,
    ) -> None:
        """node_strata vacío o None debe retornar diccionario vacío."""
        assert phase2._normalize_node_strata(value) == {}

    def test_normalize_node_strata_rejects_empty_key(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una clave vacía debe ser rechazada."""
        with pytest.raises(StratumMappingError):
            phase2._normalize_node_strata({"   ": 1})

    def test_normalize_node_strata_rejects_negative_level(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Los estratos DIKW no pueden ser negativos."""
        with pytest.raises(StratumMappingError):
            phase2._normalize_node_strata({"a": -1})

    def test_normalize_node_strata_valid_conversion(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Estratos válidos deben normalizarse a enteros no negativos."""
        normalized = phase2._normalize_node_strata({" a ": "1", "b": 2.0})

        assert normalized == {"a": 1, "b": 2}

    # ─────────────────────────────────────────────────────────────────────────
    # 6.4. Cross-check edges ↔ matriz
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_adjacency_support_no_inputs_returns_empty(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Sin matriz o sin orden de nodos, el cross-check se omite."""
        ok_notes, inverse_blocked = phase2._audit_adjacency_support(
            None,
            [("a", "b")],
            ["a", "b"],
            adjacency_zero_threshold=1e-10,
        )

        assert ok_notes == ()
        assert inverse_blocked == ()

    def test_audit_adjacency_support_rejects_wrong_node_order_length(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """node_order debe tener longitud igual a la dimensión de A."""
        A = np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(AdjacencyMatrixFormatError):
            phase2._audit_adjacency_support(
                A,
                [("a", "b")],
                ["a"],
                adjacency_zero_threshold=1e-10,
            )

    def test_audit_adjacency_support_rejects_duplicate_node_order(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """node_order no debe contener identificadores duplicados."""
        A = np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(AdjacencyMatrixFormatError):
            phase2._audit_adjacency_support(
                A,
                [("a", "b")],
                ["a", "a"],
                adjacency_zero_threshold=1e-10,
            )

    def test_audit_adjacency_support_rejects_unsupported_direct_edge(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una arista declarada sin soporte A[i,j] ≠ 0 debe ser vetada."""
        A = np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(AdjacencySupportVetoError):
            phase2._audit_adjacency_support(
                A,
                [("a", "b")],
                ["a", "b"],
                adjacency_zero_threshold=1e-10,
            )

    def test_audit_adjacency_support_rejects_inverse_edges(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una arista inversa A[j,i] ≠ 0 introduce ciclo de longitud 2."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

        with pytest.raises(AdjacencySupportVetoError):
            phase2._audit_adjacency_support(
                A,
                [("a", "b")],
                ["a", "b"],
                adjacency_zero_threshold=1e-10,
                check_inverse_edges=True,
            )

    def test_audit_adjacency_support_valid_direct_edge(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una arista soportada y sin inversa debe pasar el cross-check."""
        A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

        ok_notes, inverse_blocked = phase2._audit_adjacency_support(
            A,
            [("a", "b")],
            ["a", "b"],
            adjacency_zero_threshold=1e-10,
        )

        assert len(ok_notes) == 1
        assert inverse_blocked == ()

    # ─────────────────────────────────────────────────────────────────────────
    # 6.5. Histograma de slacks y nodos aislados
    # ─────────────────────────────────────────────────────────────────────────

    def test_compute_slack_histogram_empty(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Sin slacks, el histograma debe ser vacío."""
        assert phase2._compute_slack_histogram([]) == ()

    def test_compute_slack_histogram_counts(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """El histograma debe contar frecuencias por slack."""
        histogram = phase2._compute_slack_histogram([0, 1, 1])

        assert histogram == ((0, 1), (1, 2))

    def test_find_isolated_strata_nodes_empty_strata(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Sin estratos, no hay nodos aislados."""
        assert phase2._find_isolated_strata_nodes({}, [("a", "b")]) == ()

    def test_find_isolated_strata_nodes_detects_isolated(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Un nodo sin aristas incidentes debe ser reportado como aislado."""
        strata = {"a": 0, "b": 1, "c": 2}
        edges = [("a", "b")]

        isolated = phase2._find_isolated_strata_nodes(strata, edges)
        assert isolated == ("c",)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.6. Auditoría principal de filtración
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_poset_filtration_valid_lax(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una arista con slack positivo debe pasar en modo laxo."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "b")],
            {"a": 0, "b": 1},
        )

        assert audit.is_monotonic_filtration is True
        assert audit.edge_count == 1
        assert audit.audited_edge_count == 1
        assert audit.min_slack == 1
        assert audit.max_slack == 1
        assert audit.self_loops_detected == 0

    def test_audit_poset_filtration_strict_rejects_same_stratum(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """En modo estricto, slack=0 debe violar la filtración."""
        with pytest.raises(FiltrationViolationVeto):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(True),
                [("a", "b")],
                {"a": 0, "b": 0},
                strict_filtration=True,
                raise_on_veto=True,
            )

    def test_audit_poset_filtration_strict_no_raise_returns_false(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con veto desactivado, una violación estricta retorna False."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "b")],
            {"a": 0, "b": 0},
            strict_filtration=True,
            raise_on_veto=False,
        )

        assert audit.is_monotonic_filtration is False
        assert audit.min_slack == 0
        assert audit.max_slack == 0

    def test_audit_poset_filtration_negative_slack_raises(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Una regresión de estrato debe violar la flecha DIKW."""
        with pytest.raises(FiltrationViolationVeto):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(True),
                [("b", "a")],
                {"a": 0, "b": 1},
                strict_filtration=False,
                raise_on_veto=True,
            )

    def test_audit_poset_filtration_unknown_nodes_disallowed(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Nodos sin estrato deben detonar StratumMappingError si no se permiten."""
        with pytest.raises(StratumMappingError):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(True),
                [("a", "x")],
                {"a": 0},
                allow_unknown_nodes=False,
            )

    def test_audit_poset_filtration_unknown_nodes_allowed(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con allow_unknown_nodes=True, los nodos sin estrato se ignoran."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "x")],
            {"a": 0},
            allow_unknown_nodes=True,
        )

        assert audit.is_monotonic_filtration is True
        assert audit.audited_edge_count == 0
        assert audit.ignored_edge_count == 1
        assert audit.unknown_nodes == ("x",)

    def test_audit_poset_filtration_degraded_when_nilpotence_false_raise_true(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Si Fase 1 no certificó nilpotencia, Fase 2 debe vetar con veto activo."""
        with pytest.raises(CausalLoopVetoError):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(False),
                [("a", "b")],
                {"a": 0, "b": 1},
                raise_on_veto=True,
            )

    def test_audit_poset_filtration_degraded_when_nilpotence_false_raise_false(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con veto desactivado, Fase 2 degradada retorna certificado inválido."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(False),
            [("a", "b")],
            {"a": 0, "b": 1},
            raise_on_veto=False,
        )

        assert audit.is_monotonic_filtration is False
        assert audit.audited_edge_count == 0
        assert audit.ignored_edge_count == 1

    def test_audit_poset_filtration_self_loop_raises_when_veto_enabled(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Un auto-bucle debe detonar SelfLoopVetoError con veto activo."""
        with pytest.raises(SelfLoopVetoError):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(True),
                [("a", "a")],
                {"a": 0},
                raise_on_veto=True,
            )

    def test_audit_poset_filtration_self_loop_not_raising_when_veto_disabled(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con veto desactivado, el auto-bucle se cuenta y se excluye."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "a")],
            {"a": 0},
            raise_on_veto=False,
        )

        assert audit.self_loops_detected == 1
        assert audit.edge_count == 1
        assert audit.audited_edge_count == 0

    def test_audit_poset_filtration_isolated_nodes_reported(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Los nodos en node_strata sin aristas deben reportarse."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "b")],
            {"a": 0, "b": 1, "c": 2},
        )

        assert audit.isolated_strata_nodes == ("c",)

    def test_audit_poset_filtration_cross_check_failure_raise_true(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """El cross-check de soporte debe vetar si falta soporte directo."""
        with pytest.raises(AdjacencySupportVetoError):
            phase2._audit_poset_filtration_from_nilpotence(
                make_nilpotence_audit(True),
                [("a", "b")],
                {"a": 0, "b": 1},
                node_order=["a", "b"],
                adjacency_matrix=np.zeros((2, 2), dtype=np.float64),
                cross_check_adjacency_support=True,
                raise_on_veto=True,
            )

    def test_audit_poset_filtration_cross_check_failure_raise_false(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """Con veto desactivado, el fallo de cross-check queda en notas."""
        audit = phase2._audit_poset_filtration_from_nilpotence(
            make_nilpotence_audit(True),
            [("a", "b")],
            {"a": 0, "b": 1},
            node_order=["a", "b"],
            adjacency_matrix=np.zeros((2, 2), dtype=np.float64),
            cross_check_adjacency_support=True,
            raise_on_veto=False,
        )

        assert audit.is_monotonic_filtration is True
        assert any("Cross-check" in note for note in audit.notes)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.7. Puente terminal de Fase 2
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase2_stub_intercept_raises_not_implemented(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
    ) -> None:
        """El stub de Fase 3 debe exigir el mixin de Fase 3."""
        with pytest.raises(NotImplementedError):
            phase2._intercept_mayer_vietoris_sequence(0, 0, 0, 0)

    def test_phase2_terminal_bridge_requires_phase3_mixin(
        self,
        phase2: Phase2_PosetFiltrationAuditor,
        dag_inputs: Dict[str, Any],
        valid_betti: Dict[str, int],
    ) -> None:
        """El puente terminal de Fase 2 falla si Fase 3 no está mezclada."""
        with pytest.raises(NotImplementedError):
            phase2._phase2_terminal_bridge_to_phase3(
                dag_inputs["adjacency_matrix"],
                dag_inputs["edges"],
                dag_inputs["node_strata"],
                **valid_betti,
            )

    def test_phase2_terminal_bridge_is_phase3_initial_object(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        dag_inputs: Dict[str, Any],
        valid_betti: Dict[str, int],
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₃ ∘ Φ₂:

        El puente terminal de Fase 2, cuando Fase 3 está presente,
        produce certificados de Fase 1, Fase 2 y Fase 3.
        """
        nilpotence_audit, filtration_audit, mv_audit = (
            phase3._phase2_terminal_bridge_to_phase3(
                dag_inputs["adjacency_matrix"],
                dag_inputs["edges"],
                dag_inputs["node_strata"],
                **valid_betti,
            )
        )

        assert isinstance(nilpotence_audit, NilpotenceAuditData)
        assert isinstance(filtration_audit, PosetFiltrationData)
        assert isinstance(mv_audit, MayerVietorisAuditData)

        assert nilpotence_audit.is_strictly_nilpotent is True
        assert filtration_audit.is_monotonic_filtration is True
        assert mv_audit.is_fusion_homologous is True


# ══════════════════════════════════════════════════════════════════════════════
# §7. FASE 3 — INTERCEPCIÓN HOMOLÓGICA DE MAYER-VIETORIS
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3MayerVietorisInterceptor:
    """
    Fase 3: Intercepción de la cohomología de fusión.

    Objetivo:
        Δβ₁ = β₁(A∪B) − [β₁(A)+β₁(B)−β₁(A∩B)] = 0

    Además verifica:
        - Desigualdad débil de Mayer-Vietoris.
        - Secuencia exacta en H₀.
        - Identidad de Euler-Poincaré.
        - Defecto Jensen-Shannon.
    """

    def test_phase3_interceptor_is_instantiable(self) -> None:
        """Fase 3 debe poder instanciarse como extensión de Fase 2."""
        interceptor = Phase3_MayerVietorisInterceptor()
        assert isinstance(interceptor, Phase3_MayerVietorisInterceptor)
        assert isinstance(interceptor, Phase2_PosetFiltrationAuditor)
        assert isinstance(interceptor, Phase1_SpectralNilpotenceCertifier)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.1. Validación de números de Betti
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_validate_betti_number_rejects_booleans(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        value: Any,
    ) -> None:
        """Los booleanos no son números de Betti."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_betti_number("beta", value)

    @pytest.mark.parametrize("value, expected", [(0, 0), (3, 3), (np.int64(2), 2)])
    def test_validate_betti_number_accepts_integers(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        value: Any,
        expected: int,
    ) -> None:
        """Enteros no negativos deben ser aceptados."""
        assert phase3._validate_betti_number("beta", value) == expected

    def test_validate_betti_number_accepts_integral_float(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Floats integrales finitos deben ser aceptados."""
        assert phase3._validate_betti_number("beta", 2.0) == 2

    def test_validate_betti_number_rejects_non_integral_float(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Floats no integrales deben ser rechazados."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_betti_number("beta", 2.5)

    def test_validate_betti_number_rejects_nan(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """NaN debe ser rechazado."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_betti_number("beta", float("nan"))

    def test_validate_betti_number_rejects_string(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Los strings no forman parte del dominio de números de Betti."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_betti_number("beta", "1")

    def test_validate_betti_number_rejects_negative(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """β_k ≥ 0 por definición."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_betti_number("beta", -1)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.2. Validación de rangos opcionales
    # ─────────────────────────────────────────────────────────────────────────

    def test_validate_optional_rank_none_returns_none(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """None debe retornar None."""
        assert phase3._validate_optional_rank("rank", None, upper_bound=3) is None

    def test_validate_optional_rank_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Un rango dentro de cotas algebraicas debe ser aceptado."""
        assert phase3._validate_optional_rank("rank", 2, upper_bound=3) == 2

    def test_validate_optional_rank_rejects_below_lower(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Un rango por debajo de la cota inferior debe ser rechazado."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_optional_rank("rank", 0, upper_bound=3, lower_bound=1)

    def test_validate_optional_rank_rejects_above_upper(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Un rango por encima de la cota superior debe ser rechazado."""
        with pytest.raises(MayerVietorisInputError):
            phase3._validate_optional_rank("rank", 4, upper_bound=3)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.3. Desigualdad débil, H₀ y Euler-Poincaré
    # ─────────────────────────────────────────────────────────────────────────

    def test_verify_weak_mayer_vietoris_bound_satisfied(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """La desigualdad débil debe detectarse como satisfecha."""
        satisfied, msg = phase3._verify_weak_mayer_vietoris_bound(
            bA=1,
            bB=1,
            bI=1,
            bU=1,
        )

        assert satisfied is True
        assert "≤" in msg

    def test_verify_weak_mayer_vietoris_bound_not_satisfied(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """La desigualdad débil debe detectarse como violada."""
        satisfied, msg = phase3._verify_weak_mayer_vietoris_bound(
            bA=0,
            bB=0,
            bI=0,
            bU=1,
        )

        assert satisfied is False
        assert ">" in msg

    def test_verify_h0_sequence_omitted_when_missing_b0(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Sin β₀ completos, la verificación H₀ se omite."""
        valid, msg = phase3._verify_h0_sequence(
            None,
            1,
            1,
            1,
            connecting_boundary_rank=0,
        )

        assert valid is None
        assert "omitida" in msg.lower()

    def test_verify_h0_sequence_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Una secuencia H₀ consistente debe ser válida."""
        valid, msg = phase3._verify_h0_sequence(
            betti_0_A=2,
            betti_0_B=2,
            betti_0_intersection=1,
            betti_0_union=3,
            connecting_boundary_rank=0,
        )

        assert valid is True

    def test_verify_h0_sequence_invalid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Una secuencia H₀ inconsistente debe ser inválida."""
        valid, msg = phase3._verify_h0_sequence(
            betti_0_A=2,
            betti_0_B=2,
            betti_0_intersection=1,
            betti_0_union=99,
            connecting_boundary_rank=0,
        )

        assert valid is False

    def test_verify_euler_poincare_omitted_when_missing_b0(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Sin β₀ completos, Euler-Poincaré se omite."""
        valid, msg = phase3._verify_euler_poincare(
            None,
            1,
            1,
            1,
            bA=1,
            bB=1,
            bI=1,
            bU=1,
        )

        assert valid is None

    def test_verify_euler_poincare_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """La identidad de Euler-Poincaré debe preservarse."""
        valid, msg = phase3._verify_euler_poincare(
            betti_0_A=2,
            betti_0_B=2,
            betti_0_intersection=1,
            betti_0_union=3,
            bA=1,
            bB=1,
            bI=1,
            bU=1,
        )

        assert valid is True

    def test_verify_euler_poincare_invalid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Una violación de Euler-Poincaré debe detectarse."""
        valid, msg = phase3._verify_euler_poincare(
            betti_0_A=2,
            betti_0_B=2,
            betti_0_intersection=1,
            betti_0_union=99,
            bA=1,
            bB=1,
            bI=1,
            bU=1,
        )

        assert valid is False

    # ─────────────────────────────────────────────────────────────────────────
    # 7.4. Defecto Jensen-Shannon
    # ─────────────────────────────────────────────────────────────────────────

    def test_jensen_shannon_defect_zero_when_exact(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Si bU == expected, el defecto JS debe ser cero."""
        defect = phase3._compute_jensen_shannon_defect(1, 1)
        assert defect == pytest.approx(0.0)

    def test_jensen_shannon_defect_bounded_and_symmetric(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """El defecto JS debe ser simétrico y estar acotado en [0,1)."""
        d1 = phase3._compute_jensen_shannon_defect(10, 0)
        d2 = phase3._compute_jensen_shannon_defect(0, 10)

        assert d1 == d2
        assert 0.0 <= d1 < 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 7.5. Intercepción principal de Mayer-Vietoris
    # ─────────────────────────────────────────────────────────────────────────

    def test_intercept_default_additive_contract_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
    ) -> None:
        """El contrato aditivo por defecto debe certificar fusión homológica."""
        audit = phase3._intercept_mayer_vietoris_sequence(**valid_betti)

        assert audit.is_fusion_homologous is True
        assert audit.delta_betti_1 == 0
        assert audit.expected_union_betti_1 == 1
        assert audit.jensen_shannon_defect == pytest.approx(0.0)

    def test_intercept_default_additive_contract_invalid_raises(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
    ) -> None:
        """Δβ₁ ≠ 0 debe detonar HomologicalFusionVeto con veto activo."""
        invalid = dict(valid_betti)
        invalid["betti_1_union"] = 2

        with pytest.raises(HomologicalFusionVeto):
            phase3._intercept_mayer_vietoris_sequence(**invalid, raise_on_veto=True)

    def test_intercept_invalid_no_raise_returns_not_homologous(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
    ) -> None:
        """Con veto desactivado, Δβ₁ ≠ 0 retorna certificado inválido."""
        invalid = dict(valid_betti)
        invalid["betti_1_union"] = 2

        audit = phase3._intercept_mayer_vietoris_sequence(
            **invalid,
            raise_on_veto=False,
        )

        assert audit.is_fusion_homologous is False
        assert audit.delta_betti_1 == 1

    def test_intercept_exact_ranks_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Rangos exactos de la secuencia deben permitir fusión homológica."""
        audit = phase3._intercept_mayer_vietoris_sequence(
            betti_1_A=2,
            betti_1_B=2,
            betti_1_intersection=3,
            betti_1_union=3,
            image_rank_h1_intersection=2,
            connecting_boundary_rank=1,
        )

        assert audit.is_fusion_homologous is True
        assert audit.expected_union_betti_1 == 3
        assert audit.delta_betti_1 == 0

    def test_intercept_image_rank_above_upper_raises(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """rank(im φ) no puede exceder min(β₁(A∩B), β₁(A)+β₁(B))."""
        with pytest.raises(MayerVietorisInputError):
            phase3._intercept_mayer_vietoris_sequence(
                betti_1_A=1,
                betti_1_B=1,
                betti_1_intersection=3,
                betti_1_union=0,
                image_rank_h1_intersection=3,
            )

    def test_intercept_boundary_rank_above_upper_raises(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """rank(∂) no puede exceder β₁(A∪B) si no se proporciona β₀(A∩B)."""
        with pytest.raises(MayerVietorisInputError):
            phase3._intercept_mayer_vietoris_sequence(
                betti_1_A=1,
                betti_1_B=1,
                betti_1_intersection=1,
                betti_1_union=1,
                connecting_boundary_rank=2,
            )

    def test_intercept_expected_negative_with_default_contract(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """Si β₁(A∩B) > β₁(A)+β₁(B), el contrato aditivo genera expected < 0."""
        audit = phase3._intercept_mayer_vietoris_sequence(
            betti_1_A=0,
            betti_1_B=0,
            betti_1_intersection=1,
            betti_1_union=0,
            raise_on_veto=False,
        )

        assert audit.expected_union_betti_1 == -1
        assert audit.is_fusion_homologous is False
        assert any("ALERTA" in note for note in audit.notes)

    def test_intercept_weak_bound_false_recorded(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
    ) -> None:
        """La violación de la desigualdad débil debe quedar registrada."""
        audit = phase3._intercept_mayer_vietoris_sequence(
            betti_1_A=0,
            betti_1_B=0,
            betti_1_intersection=0,
            betti_1_union=1,
            raise_on_veto=False,
        )

        assert audit.weak_bound_satisfied is False
        assert audit.is_fusion_homologous is False

    def test_intercept_euler_mismatch_raises_when_veto_enabled(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
    ) -> None:
        """Una violación Euler-Poincaré debe vetar si β₀ fueron fournisados."""
        kwargs = dict(valid_betti)
        kwargs.update(make_valid_betti0())
        kwargs["betti_0_union"] = 99

        with pytest.raises(EulerPoincareMismatchError):
            phase3._intercept_mayer_vietoris_sequence(**kwargs, raise_on_veto=True)

    def test_intercept_euler_mismatch_no_raise_returns_false_flag(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
    ) -> None:
        """Con veto desactivado, Euler-Poincaré inválido retorna flag False."""
        kwargs = dict(valid_betti)
        kwargs.update(make_valid_betti0())
        kwargs["betti_0_union"] = 99

        audit = phase3._intercept_mayer_vietoris_sequence(
            **kwargs,
            raise_on_veto=False,
        )

        assert audit.euler_characteristic_valid is False
        assert audit.h0_sequence_valid is False

    # ─────────────────────────────────────────────────────────────────────────
    # 7.6. Síntesis terminal de Fase 3
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase3_terminal_synthesis_valid(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        dag_inputs: Dict[str, Any],
        valid_betti: Dict[str, int],
        valid_betti0: Dict[str, int],
    ) -> None:
        """La síntesis terminal debe producir un estado causal válido."""
        state = phase3._phase3_terminal_synthesis(
            dag_inputs["adjacency_matrix"],
            dag_inputs["edges"],
            dag_inputs["node_strata"],
            node_order=dag_inputs["node_order"],
            strict_filtration=True,
            cross_check_adjacency_support=True,
            **valid_betti,
            **valid_betti0,
        )

        assert isinstance(state, CausalGovernanceState)
        assert state.is_causally_valid is True
        assert state.nilpotence_audit.is_strictly_nilpotent is True
        assert state.filtration_audit.is_monotonic_filtration is True
        assert state.mayer_vietoris_audit.is_fusion_homologous is True

    def test_phase3_terminal_synthesis_invalid_cycle_no_raise(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        valid_betti: Dict[str, int],
        valid_betti0: Dict[str, int],
    ) -> None:
        """Con veto desactivado, un ciclo retorna estado inválido."""
        state = phase3._phase3_terminal_synthesis(
            make_cycle_adjacency(2),
            [("v0", "v1")],
            {"v0": 0, "v1": 1},
            node_order=["v0", "v1"],
            cross_check_adjacency_support=False,
            raise_on_veto=False,
            **valid_betti,
            **valid_betti0,
        )

        assert state.is_causally_valid is False
        assert state.nilpotence_audit.is_strictly_nilpotent is False
        assert state.filtration_audit.is_monotonic_filtration is False

    def test_phase3_terminal_synthesis_invalid_homology_no_raise(
        self,
        phase3: Phase3_MayerVietorisInterceptor,
        dag_inputs: Dict[str, Any],
        valid_betti0: Dict[str, int],
    ) -> None:
        """Con veto desactivado, Δβ₁ ≠ 0 retorna estado inválido."""
        invalid_betti = make_valid_betti()
        invalid_betti["betti_1_union"] = 2

        state = phase3._phase3_terminal_synthesis(
            dag_inputs["adjacency_matrix"],
            dag_inputs["edges"],
            dag_inputs["node_strata"],
            node_order=dag_inputs["node_order"],
            cross_check_adjacency_support=True,
            raise_on_veto=False,
            **invalid_betti,
            **valid_betti0,
        )

        assert state.is_causally_valid is False
        assert state.mayer_vietoris_audit.is_fusion_homologous is False


# ══════════════════════════════════════════════════════════════════════════════
# §8. ORQUESTADOR SUPREMO — PIPELINE DIRECTOR AGENT
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineDirectorAgentEndToEnd:
    """
    Orquestación completa:

        Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁

    Estos tests validan el diagrama conmutativo completo, los vetos
    categóricos y la inmutabilidad del estado terminal.
    """

    def test_agent_is_phase3_subclass_and_morphism_if_available(self) -> None:
        """El agente debe componer todas las fases y ser morfismo si aplica."""
        agent_instance = PipelineDirectorAgent()

        assert isinstance(agent_instance, Phase3_MayerVietorisInterceptor)
        assert isinstance(agent_instance, Phase2_PosetFiltrationAuditor)
        assert isinstance(agent_instance, Phase1_SpectralNilpotenceCertifier)

        if Morphism is not None:
            assert isinstance(agent_instance, Morphism)

    def test_execute_valid_causal_governance(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """El endofuntor completo debe certificar causalidad válida."""
        kwargs = make_valid_agent_kwargs(n=3)
        state = agent.execute_causal_governance(**kwargs)

        assert isinstance(state, CausalGovernanceState)
        assert state.is_causally_valid is True

        assert state.nilpotence_audit.is_strictly_nilpotent is True
        assert state.filtration_audit.is_monotonic_filtration is True
        assert state.mayer_vietoris_audit.is_fusion_homologous is True
        assert state.mayer_vietoris_audit.jensen_shannon_defect == pytest.approx(0.0)

    def test_execute_valid_returns_uuid_and_utc_timestamp(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """El estado final debe incluir governance_id UUID y timestamp UTC."""
        kwargs = make_valid_agent_kwargs(n=3)
        state = agent.execute_causal_governance(**kwargs)

        # governance_id debe ser UUID válido.
        parsed = uuid.UUID(state.governance_id)
        assert str(parsed) == state.governance_id

        # generated_at_utc debe ser ISO-8601 UTC.
        assert isinstance(state.generated_at_utc, str)
        assert "T" in state.generated_at_utc
        assert state.generated_at_utc.endswith("+00:00")

    def test_execute_valid_path_nilpotence_index(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Una cadena de 3 nodos debe certificar índice de nilpotencia ν=3."""
        kwargs = make_valid_agent_kwargs(n=3)
        state = agent.execute_causal_governance(**kwargs)

        assert state.nilpotence_audit.nilpotence_index == 3
        assert state.nilpotence_audit.power_audited is True

    def test_execute_rejects_cycle(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Un ciclo espectral debe detonar CausalLoopVetoError."""
        kwargs = make_valid_agent_kwargs(n=2)
        kwargs["adjacency_matrix"] = make_cycle_adjacency(2)

        with pytest.raises(CausalLoopVetoError):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_self_loop(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Un auto-bucle en edges debe detonar SelfLoopVetoError."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["edges"] = make_path_edges(3) + [("v0", "v0")]

        with pytest.raises(SelfLoopVetoError):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_filtration_violation(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Una regresión DIKW debe detonar FiltrationViolationVeto."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["edges"] = make_path_edges(3) + [("v2", "v1")]
        kwargs["cross_check_adjacency_support"] = False

        with pytest.raises(FiltrationViolationVeto):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_strict_same_stratum_edge(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """En filtración estricta, una arista intra-estrato debe vetarse."""
        kwargs = make_valid_agent_kwargs(n=2)
        kwargs["adjacency_matrix"] = np.array([[0.0, 1.0], [0.0, 0.0]])
        kwargs["edges"] = [("v0", "v1")]
        kwargs["node_strata"] = {"v0": 0, "v1": 0}
        kwargs["node_order"] = ["v0", "v1"]
        kwargs["strict_filtration"] = True
        kwargs["cross_check_adjacency_support"] = True

        with pytest.raises(FiltrationViolationVeto):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_homological_fusion_veto(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Δβ₁ ≠ 0 debe detonar HomologicalFusionVeto."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["betti_1_union"] = 2

        with pytest.raises(HomologicalFusionVeto):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_euler_poincare_mismatch(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Una violación de Euler-Poincaré debe detonar veto específico."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["betti_0_union"] = 99

        with pytest.raises(EulerPoincareMismatchError):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_adjacency_support_veto(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Una arista declarada sin soporte matricial debe ser vetada."""
        kwargs = make_valid_agent_kwargs(n=2)
        kwargs["adjacency_matrix"] = np.zeros((2, 2), dtype=np.float64)
        kwargs["edges"] = [("v0", "v1")]
        kwargs["node_strata"] = {"v0": 0, "v1": 1}
        kwargs["node_order"] = ["v0", "v1"]
        kwargs["cross_check_adjacency_support"] = True

        with pytest.raises(AdjacencySupportVetoError):
            agent.execute_causal_governance(**kwargs)

    def test_execute_rejects_non_boolean_weights_when_required(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """require_boolean=True debe rechazar pesos distintos de 1."""
        kwargs = make_valid_agent_kwargs(n=2)
        kwargs["adjacency_matrix"] = np.array([[0.0, 2.0], [0.0, 0.0]])
        kwargs["require_boolean"] = True

        with pytest.raises(AdjacencyMatrixFormatError):
            agent.execute_causal_governance(**kwargs)

    def test_execute_accepts_signed_weights_when_allowed(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Con allow_signed_weights=True, pesos negativos nilpotentes pasan."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["adjacency_matrix"] = make_upper_triangular_dag(3, weight=-1.0)
        kwargs["allow_signed_weights"] = True
        kwargs["require_boolean"] = False

        state = agent.execute_causal_governance(**kwargs)
        assert state.is_causally_valid is True

    def test_execute_raise_on_veto_false_returns_invalid_state_for_cycle(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Con veto desactivado, un ciclo retorna estado inválido sin excepción."""
        kwargs = make_valid_agent_kwargs(n=2)
        kwargs["adjacency_matrix"] = make_cycle_adjacency(2)
        kwargs["cross_check_adjacency_support"] = False
        kwargs["raise_on_veto"] = False

        state = agent.execute_causal_governance(**kwargs)

        assert state.is_causally_valid is False
        assert state.nilpotence_audit.is_strictly_nilpotent is False
        assert state.filtration_audit.is_monotonic_filtration is False

    def test_execute_raise_on_veto_false_returns_invalid_state_for_homology(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """Con veto desactivado, Δβ₁ ≠ 0 retorna estado inválido."""
        kwargs = make_valid_agent_kwargs(n=3)
        kwargs["betti_1_union"] = 2
        kwargs["raise_on_veto"] = False

        state = agent.execute_causal_governance(**kwargs)

        assert state.is_causally_valid is False
        assert state.mayer_vietoris_audit.is_fusion_homologous is False

    def test_causal_governance_state_is_immutable(
        self,
        agent: PipelineDirectorAgent,
    ) -> None:
        """El objeto terminal del endofuntor debe ser inmutable."""
        kwargs = make_valid_agent_kwargs(n=3)
        state = agent.execute_causal_governance(**kwargs)

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_causally_valid = False

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.governance_id = "mutado"

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.nilpotence_audit = None