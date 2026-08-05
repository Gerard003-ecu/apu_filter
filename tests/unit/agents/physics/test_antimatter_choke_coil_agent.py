# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Batería de pruebas unitarias para:                                                      ║
║  app/agents/physics/antimatter_choke_coil_agent.py                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ORGANIZACIÓN POR FASES ANIDADAS:                                                        ║
║  ────────────────────────────────                                                        ║
║  Fase 1 → Hermiticidad del operador de aniquilación.                                     ║
║           El último test valida que `Phase1HermiticityHandoff` es el objeto inicial      ║
║           de Fase 2.                                                                     ║
║                                                                                          ║
║  Fase 2 → Cota termodinámica de Bekenstein.                                              ║
║           El primer insumo formal es `Phase1HermiticityHandoff`.                         ║
║           El último test valida que `Phase2BekensteinHandoff` es el objeto inicial       ║
║           de Fase 3.                                                                     ║
║                                                                                          ║
║  Fase 3 → Certificación simpléctica Port-Hamiltoniana.                                   ║
║           El primer insumo formal es `Phase2BekensteinHandoff`.                          ║
║                                                                                          ║
║  Orquestación → Endofuntor completo `AntimatterChokeCoilAgent`.                          ║
║                                                                                          ║
║  CONVENCIONES:                                                                           ║
║  ─────────────                                                                           ║
║  - Los tests acceden deliberadamente a métodos protegidos (`_...`) porque la             ║
║    especificación física y categórica está implementada en esos puntos de control.       ║
║  - Se emplean tolerancias adaptativas coherentes con el módulo auditado.                 ║
║  - No se introduce aleatoriedad no controlada: todo RNG está sembrado.                   ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Dependencias duras del módulo auditado.
pytest.importorskip("numpy", reason="El módulo auditado requiere NumPy.")
pytest.importorskip("scipy", reason="El módulo auditado requiere SciPy.")

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# §0. CARGA ROBUSTA DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

_TARGET_MODULE_NAME = "app.agents.physics.antimatter_choke_coil_agent"
_TARGET_REL_PATH = Path("app") / "agents" / "physics" / "antimatter_choke_coil_agent.py"


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
VacuumCustodianError = target.VacuumCustodianError
DomainIntegrityViolationError = target.DomainIntegrityViolationError
NonHermitianOperatorError = target.NonHermitianOperatorError
SpectralContaminationError = target.SpectralContaminationError
BekensteinLimitViolation = target.BekensteinLimitViolation
CausalityViolationError = target.CausalityViolationError
SymplecticCollapseError = target.SymplecticCollapseError
PhaseSpaceTopologyError = target.PhaseSpaceTopologyError

# DTOs.
SpectralDecompositionData = target.SpectralDecompositionData
HermiticityAuditData = target.HermiticityAuditData
BekensteinBoundData = target.BekensteinBoundData
SymplecticDissipationData = target.SymplecticDissipationData
Phase1HermiticityHandoff = target.Phase1HermiticityHandoff
Phase2BekensteinHandoff = target.Phase2BekensteinHandoff
VacuumGovernanceState = target.VacuumGovernanceState

# Clases de fase.
Phase1_HermiticityAuditor = target.Phase1_HermiticityAuditor
Phase2_BekensteinBoundEnforcer = target.Phase2_BekensteinBoundEnforcer
Phase3_SymplecticPortHamiltonianCertifier = target.Phase3_SymplecticPortHamiltonianCertifier

# Orquestador.
AntimatterChokeCoilAgent = target.AntimatterChokeCoilAgent

# Entidades categóricas opcionales.
TopologicalInvariantError = getattr(target, "TopologicalInvariantError", None)
Morphism = getattr(target, "Morphism", None)

# Constantes físicas y numéricas.
_MACHINE_EPSILON = float(getattr(target, "_MACHINE_EPSILON", np.finfo(np.float64).eps))
_HBAR_EFF = float(getattr(target, "_HBAR_EFF", 1.054_571_817e-34))
_C_EFF = float(getattr(target, "_C_EFF", 299_792_458.0))
_K_B = float(getattr(target, "_K_B", 1.380_649e-23))
_ELECTRON_MASS_KG = float(
    getattr(target, "_ELECTRON_MASS_KG", 9.109_383_7015e-31)
)

_LOGGER_NAME = getattr(
    getattr(target, "logger", None),
    "name",
    "MIC.Omega.VacuumCustodian.Granular",
)


# ══════════════════════════════════════════════════════════════════════════════
# §2. FÁBRICAS DE DATOS DETERMINISTAS
# ══════════════════════════════════════════════════════════════════════════════


def make_hermitian_operator(n: int = 2, seed: int = 7) -> np.ndarray:
    """
    Construye un operador hermítico complejo deterministicamente.

    Se fuerza matemáticamente:
        A = (A + A†) / 2
    para garantizar autoadjunción dentro de la precisión flotante.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = 0.5 * (A + A.conj().T)
    return A.astype(np.complex128)


def make_non_hermitian_operator() -> np.ndarray:
    """Devuelve un operador manifiestamente no autoadjunto."""
    return np.array(
        [[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )


def make_canonical_antisymmetric_matrix(dim: int) -> np.ndarray:
    """
    Construye una matriz antisimétrica canónica de dimensión `dim`.

    Para dim = 2n:
        J = [[0, I], [-I, 0]]
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("make_canonical_antisymmetric_matrix requiere dim par y positivo.")

    n = dim // 2
    J = np.zeros((dim, dim), dtype=np.float64)
    identity = np.eye(n, dtype=np.float64)
    J[:n, n:] = identity
    J[n:, :n] = -identity
    return J


def make_symplectic_inputs(
    dim: int = 2,
    dissipation: float = 0.1,
    seed: int = 11,
) -> Dict[str, np.ndarray]:
    """
    Genera insumos válidos para la Fase 3.

    Incluye:
        - jacobian_M = I (simpléctico).
        - grad_H determinista.
        - J antisimétrica canónica.
        - R simétrica y semidefinida positiva.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("make_symplectic_inputs requiere una dimensión par y positiva.")

    rng = np.random.default_rng(seed)
    grad_H = rng.normal(scale=0.25, size=dim).astype(np.float64)

    return {
        "jacobian_M": np.eye(dim, dtype=np.float64),
        "grad_H": grad_H,
        "J_matrix": make_canonical_antisymmetric_matrix(dim),
        "R_matrix": float(dissipation) * np.eye(dim, dtype=np.float64),
    }


def bekenstein_bound_value(energy: float, radius: float) -> float:
    """
    Calcula la cota de Bekenstein con las mismas constantes efectivas
    empleadas por el módulo auditado:

        S_max = 2π k_B E R / (ℏ c)
    """
    return (2.0 * math.pi * _K_B * float(energy) * float(radius)) / (
        _HBAR_EFF * _C_EFF
    )


def make_bekenstein_safe_inputs() -> Dict[str, float]:
    """
    Insumos físicamente seguros para la aniquilación e⁻ + e⁺ → 2γ.

    Se emplea:
        E ≈ 2 m_e c²
        R = 1 nm
        S = 1e-30 J/K
    """
    gamma_energy = 2.0 * _ELECTRON_MASS_KG * _C_EFF**2
    system_radius_R = 1.0e-9
    emitted_entropy_S = 1.0e-30

    return {
        "gamma_energy": float(gamma_energy),
        "system_radius_R": float(system_radius_R),
        "emitted_entropy_S": float(emitted_entropy_S),
    }


def make_synthetic_bekenstein_inputs(entropy_fraction: float = 0.5) -> Dict[str, float]:
    """
    Insumos sintéticos macroscópicos para evaluar la cota de Bekenstein
    en un régimen donde la tolerancia absoluta no domina la comparación.
    """
    energy = 1.0
    radius = 1.0
    bound = bekenstein_bound_value(energy, radius)
    entropy = float(entropy_fraction) * bound

    return {
        "gamma_energy": energy,
        "system_radius_R": radius,
        "emitted_entropy_S": entropy,
    }


def make_valid_agent_inputs(dim: int = 2, seed: int = 17) -> Dict[str, Any]:
    """Construye insumos completos válidos para el orquestador supremo."""
    operator_A = make_hermitian_operator(n=2, seed=seed)
    bekenstein_inputs = make_bekenstein_safe_inputs()
    symplectic_inputs = make_symplectic_inputs(dim=dim, dissipation=0.15, seed=seed + 1)

    return {
        "operator_A": operator_A,
        **bekenstein_inputs,
        **symplectic_inputs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §3. FIXTURES POR FASES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def phase1_auditor() -> Phase1_HermiticityAuditor:
    """Instancia fresca del auditor de Fase 1."""
    return Phase1_HermiticityAuditor()


@pytest.fixture(scope="function")
def phase2_enforcer() -> Phase2_BekensteinBoundEnforcer:
    """Instancia fresca del regulador de Fase 2."""
    return Phase2_BekensteinBoundEnforcer()


@pytest.fixture(scope="function")
def phase3_certifier() -> Phase3_SymplecticPortHamiltonianCertifier:
    """Instancia fresca del certificador de Fase 3."""
    return Phase3_SymplecticPortHamiltonianCertifier()


@pytest.fixture(scope="function")
def agent() -> AntimatterChokeCoilAgent:
    """Instancia fresca del endofuntor orquestador."""
    return AntimatterChokeCoilAgent()


@pytest.fixture(scope="function")
def hermitian_operator_2x2() -> np.ndarray:
    """Operador hermítico 2×2 determinista."""
    return make_hermitian_operator(n=2, seed=7)


@pytest.fixture(scope="function")
def non_hermitian_operator_2x2() -> np.ndarray:
    """Operador no hermítico 2×2."""
    return make_non_hermitian_operator()


@pytest.fixture(scope="function")
def bekenstein_safe_inputs() -> Dict[str, float]:
    """Insumos seguros para la cota de Bekenstein."""
    return make_bekenstein_safe_inputs()


@pytest.fixture(scope="function")
def phase1_handoff(
    phase1_auditor: Phase1_HermiticityAuditor,
    hermitian_operator_2x2: np.ndarray,
) -> Phase1HermiticityHandoff:
    """
    Artefacto terminal de Fase 1.

    Este fixture materializa la salida formal de Fase 1 y se convierte
    en el prefijo obligatorio de Fase 2.
    """
    return phase1_auditor._phase1_audit_and_handoff_to_phase2(
        operator_A=hermitian_operator_2x2,
    )


@pytest.fixture(scope="function")
def phase2_handoff(
    phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    phase1_handoff: Phase1HermiticityHandoff,
    bekenstein_safe_inputs: Dict[str, float],
) -> Phase2BekensteinHandoff:
    """
    Artefacto terminal de Fase 2.

    Este fixture materializa la salida formal de Fase 2 y se convierte
    en el prefijo obligatorio de Fase 3.
    """
    return phase2_enforcer._phase2_enforce_and_handoff_to_phase3(
        phase1_handoff=phase1_handoff,
        **bekenstein_safe_inputs,
    )


@pytest.fixture(scope="function")
def symplectic_inputs_2d() -> Dict[str, np.ndarray]:
    """Insumos simplécticos válidos en dimensión 2."""
    return make_symplectic_inputs(dim=2, dissipation=0.1, seed=13)


# ══════════════════════════════════════════════════════════════════════════════
# §4. CONTRATO DEL MÓDULO Y TAXONOMÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleContractAndExceptionTaxonomy:
    """
    Contrato estructural del módulo.

    Verifica que el módulo expone las clases, DTOs y excepciones
    requeridos por la arquitectura funtorial.
    """

    def test_module_exposes_core_types(self) -> None:
        """El módulo debe exponer todos los tipos públicos principales."""
        core_names = (
            "VacuumCustodianError",
            "DomainIntegrityViolationError",
            "NonHermitianOperatorError",
            "SpectralContaminationError",
            "BekensteinLimitViolation",
            "CausalityViolationError",
            "SymplecticCollapseError",
            "PhaseSpaceTopologyError",
            "SpectralDecompositionData",
            "HermiticityAuditData",
            "BekensteinBoundData",
            "SymplecticDissipationData",
            "Phase1HermiticityHandoff",
            "Phase2BekensteinHandoff",
            "VacuumGovernanceState",
            "Phase1_HermiticityAuditor",
            "Phase2_BekensteinBoundEnforcer",
            "Phase3_SymplecticPortHamiltonianCertifier",
            "AntimatterChokeCoilAgent",
        )

        for name in core_names:
            assert hasattr(target, name), f"El módulo no expone el tipo requerido: {name}"

    def test_phase_class_hierarchy_is_nested(self) -> None:
        """La jerarquía de clases debe reflejar la composición de fases."""
        assert issubclass(Phase2_BekensteinBoundEnforcer, Phase1_HermiticityAuditor)
        assert issubclass(
            Phase3_SymplecticPortHamiltonianCertifier,
            Phase2_BekensteinBoundEnforcer,
        )
        assert issubclass(
            AntimatterChokeCoilAgent,
            Phase3_SymplecticPortHamiltonianCertifier,
        )

    def test_agent_is_morphism_if_morphism_exists(self) -> None:
        """El agente debe ser un morfismo si el topo MIC define `Morphism`."""
        if Morphism is None:
            pytest.skip("El módulo auditado no expone Morphism.")

        assert issubclass(AntimatterChokeCoilAgent, Morphism)

    @pytest.mark.parametrize(
        "exc",
        [
            DomainIntegrityViolationError,
            NonHermitianOperatorError,
            SpectralContaminationError,
            BekensteinLimitViolation,
            CausalityViolationError,
            SymplecticCollapseError,
            PhaseSpaceTopologyError,
        ],
    )
    def test_exceptions_are_rooted_in_vacuum_custodian_error(self, exc) -> None:
        """Toda excepción cuántica debe descender de VacuumCustodianError."""
        assert issubclass(exc, VacuumCustodianError)

    def test_root_exception_inherits_topological_invariant_error_if_available(self) -> None:
        """Si existe TopologicalInvariantError, la raíz cuántica debe heredarla."""
        if TopologicalInvariantError is None:
            pytest.skip("TopologicalInvariantError no está disponible en el módulo.")

        assert issubclass(VacuumCustodianError, TopologicalInvariantError)


# ══════════════════════════════════════════════════════════════════════════════
# §5. FASE 1 — HERMITICIDAD DEL OPERADOR DE ANIQUILACIÓN
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1HermiticityAuditor:
    """
    Fase 1: Auditoría de hermiticidad.

    Objetivo:
        A = A†  ⇒  ||A - A†||_F <= ε

    El último test de esta clase valida que el objeto terminal de Fase 1,
    `Phase1HermiticityHandoff`, es exactamente el objeto inicial que
    Fase 2 requiere para continuar la composición funtorial.
    """

    def test_phase1_auditor_is_instantiable(self) -> None:
        """Fase 1 debe poder instanciarse sin estado mutable."""
        auditor = Phase1_HermiticityAuditor()
        assert isinstance(auditor, Phase1_HermiticityAuditor)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.1. Coerción de escalares
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("value", [0.0, 1.0, -2.5, 1e300])
    def test_coerce_finite_scalar_accepts_finite_reals(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
        value: float,
    ) -> None:
        """Todo escalar físico finito debe poder materializarse como float."""
        scalar = phase1_auditor._coerce_finite_scalar("x", value)
        assert scalar == float(value)
        assert np.isfinite(scalar)

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_coerce_finite_scalar_rejects_booleans(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
        value: Any,
    ) -> None:
        """Los booleanos pertenecen a B₂, no a R, y deben ser rechazados."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_scalar("x", value)

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan"), "inf", "-inf", "nan"],
    )
    def test_coerce_finite_scalar_rejects_non_finite_values(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
        value: Any,
    ) -> None:
        """Los escalares no finitos deben violar la integridad de dominio."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_scalar("x", value)

    @pytest.mark.parametrize("value", [object(), 1 + 2j, [1.0]])
    def test_coerce_finite_scalar_rejects_non_convertible_values(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
        value: Any,
    ) -> None:
        """Los valores no convertibles a float deben ser rechazados."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_scalar("x", value)

    def test_coerce_finite_scalar_rejects_negative_when_not_allowed(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Si allow_negative=False, cualquier valor negativo es inválido."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_scalar("x", -1e-20, allow_negative=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.2. Coerción de matrices
    # ─────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_matrix_accepts_square_finite_matrix(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Una matriz finita cuadrada debe ser aceptada."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        out = phase1_auditor._coerce_finite_matrix(
            "M",
            matrix,
            dtype=np.float64,
            square_required=True,
        )
        np.testing.assert_allclose(out, matrix)

    def test_coerce_finite_matrix_rejects_non_2d_object(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un vector no es una matriz y debe ser rechazado."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix("M", np.array([1.0, 2.0]))

    def test_coerce_finite_matrix_rejects_empty_matrix(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Una matriz vacía no pertenece al espacio de Hilbert operativo."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix("M", np.empty((0, 0)))

    def test_coerce_finite_matrix_enforces_squareness(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Si square_required=True, la matriz debe ser cuadrada."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix(
                "M",
                np.zeros((2, 3)),
                square_required=True,
            )

    def test_coerce_finite_matrix_enforces_min_dimension(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La dimensión mínima debe ser respetada."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix(
                "M",
                np.array([[1.0]]),
                min_dimension=2,
            )

    def test_coerce_finite_matrix_rejects_non_finite_entries(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Una matriz con NaN/Inf debe ser rechazada."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix("M", np.array([[np.nan, 0.0]]))

    def test_coerce_finite_matrix_rejects_ragged_sequence(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Una secuencia irregular no puede materializarse como matriz."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_matrix("M", [[1.0], [2.0, 3.0]])

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3. Coerción de vectores
    # ─────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_vector_accepts_scalar_and_column_vectors(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Escalares y vectores columna deben ser aplanados correctamente."""
        scalar_vector = phase1_auditor._coerce_finite_vector("v", 3.0)
        assert scalar_vector.shape == (1,)
        assert scalar_vector[0] == 3.0

        column_vector = phase1_auditor._coerce_finite_vector(
            "v",
            np.array([[1.0], [2.0]]),
        )
        assert column_vector.shape == (2,)
        np.testing.assert_allclose(column_vector, np.array([1.0, 2.0]))

    def test_coerce_finite_vector_rejects_empty_vector(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un vector vacío viola la completitud del espacio de fase."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_vector("v", np.array([]))

    def test_coerce_finite_vector_enforces_expected_dimension(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La dimensión esperada debe cumplirse exactamente."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_vector(
                "v",
                np.array([1.0, 2.0]),
                expected_dim=3,
            )

    def test_coerce_finite_vector_rejects_non_finite_entries(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un vector con componentes no finitas debe ser rechazado."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._coerce_finite_vector("v", np.array([1.0, np.nan]))

    # ─────────────────────────────────────────────────────────────────────────
    # 5.4. Tolerancia adaptativa
    # ─────────────────────────────────────────────────────────────────────────

    def test_adaptive_tolerance_is_lower_bounded_by_base(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La tolerancia nunca debe ser menor que la tolerancia base."""
        base = 1e-8
        tol = phase1_auditor._adaptive_tolerance(base, 0.0)
        assert tol >= base

    def test_adaptive_tolerance_grows_with_scale(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La tolerancia debe crecer con la escala del objeto auditado."""
        large_reference = np.ones(10, dtype=np.float64) * 1e9
        tol_large = phase1_auditor._adaptive_tolerance(1e-12, large_reference)
        assert tol_large > 1e-12

    def test_adaptive_tolerance_amplifies_with_condition_number(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La tolerancia debe amplificarse con el número de condición."""
        well_conditioned = np.eye(2, dtype=np.float64)
        ill_conditioned = np.diag([1.0, 1e-9]).astype(np.float64)

        tol_well = phase1_auditor._adaptive_tolerance(
            1e-12,
            well_conditioned,
            condition_amplification=True,
        )
        tol_ill = phase1_auditor._adaptive_tolerance(
            1e-12,
            ill_conditioned,
            condition_amplification=True,
        )

        assert tol_ill > tol_well

    # ─────────────────────────────────────────────────────────────────────────
    # 5.5. Auditoría de hermiticidad
    # ─────────────────────────────────────────────────────────────────────────

    def test_hermitian_identity_operator_is_certified(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """La identidad es autoadjunta y debe ser certificada."""
        A = np.eye(3, dtype=np.complex128)
        audit = phase1_auditor._audit_operator_hermiticity(A)

        assert audit.is_hermitian is True
        assert audit.operator_dimension == 3
        assert audit.residual_norm <= audit.hermiticity_tolerance
        assert audit.spectral_decomposition is not None
        assert audit.spectral_decomposition.is_spectrally_clean is True

    def test_random_hermitian_operator_is_certified(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un operador hermítico aleatorio determinista debe ser certificado."""
        A = make_hermitian_operator(n=4, seed=99)
        audit = phase1_auditor._audit_operator_hermiticity(A)

        assert audit.is_hermitian is True
        assert audit.operator_dimension == 4
        assert audit.residual_norm <= audit.hermiticity_tolerance
        assert audit.spectral_decomposition is not None
        assert np.all(np.isfinite(audit.spectral_decomposition.eigenvalues_real))

    def test_zero_operator_is_certified_and_condition_number_is_infinite(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """El operador cero es hermítico pero singular; κ(A)=∞."""
        A = np.zeros((2, 2), dtype=np.complex128)
        audit = phase1_auditor._audit_operator_hermiticity(A)

        assert audit.is_hermitian is True
        assert audit.spectral_decomposition is not None
        assert math.isinf(audit.spectral_decomposition.condition_number)

    def test_non_hermitian_operator_is_rejected(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
        non_hermitian_operator_2x2: np.ndarray,
    ) -> None:
        """Un operador no autoadjunto debe detonar NonHermitianOperatorError."""
        with pytest.raises(NonHermitianOperatorError):
            phase1_auditor._audit_operator_hermiticity(non_hermitian_operator_2x2)

    def test_non_square_operator_is_rejected(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un operador no cuadrado viola el dominio del espacio de Hilbert."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._audit_operator_hermiticity(
                np.ones((2, 3), dtype=np.complex128)
            )

    def test_non_finite_operator_is_rejected(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un operador con componentes no finitas debe ser rechazado."""
        with pytest.raises(DomainIntegrityViolationError):
            phase1_auditor._audit_operator_hermiticity(
                np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 5.6. Descomposición espectral
    # ─────────────────────────────────────────────────────────────────────────

    def test_spectral_decomposition_validates_real_spectrum(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Un operador diagonal real debe tener espectro real certificado."""
        A = np.diag([1.0, 2.0, 3.0]).astype(np.complex128)
        data = phase1_auditor._spectral_decomposition_and_validation(A)

        assert data.is_spectrally_clean is True
        np.testing.assert_allclose(data.eigenvalues_real, np.array([1.0, 2.0, 3.0]))
        assert data.eigenvalues_imaginary_norm == pytest.approx(0.0, abs=1e-15)
        assert data.trace_real == pytest.approx(6.0)
        assert data.trace_imaginary_norm == pytest.approx(0.0, abs=1e-15)

    def test_spectral_decomposition_rejects_imaginary_trace(
        self,
        phase1_auditor: Phase1_HermiticityAuditor,
    ) -> None:
        """Una traza con parte imaginaria no tolerable debe ser rechazada."""
        A = np.array([[1.0 + 2.0j]], dtype=np.complex128)
        with pytest.raises(SpectralContaminationError):
            phase1_auditor._spectral_decomposition_and_validation(A)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.7. Handoff terminal de Fase 1
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase1_handoff_contains_complete_certificate(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
    ) -> None:
        """El handoff de Fase 1 debe contener el certificado completo."""
        assert isinstance(phase1_handoff, Phase1HermiticityHandoff)
        assert phase1_handoff.hermiticity_audit.is_hermitian is True
        assert phase1_handoff.operator_dimension == (
            phase1_handoff.hermiticity_audit.operator_dimension
        )
        assert (
            phase1_handoff.spectral_certificate
            is phase1_handoff.hermiticity_audit.spectral_decomposition
        )

    def test_phase1_handoff_is_immutable(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
    ) -> None:
        """El handoff de Fase 1 debe ser inmutable."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            phase1_handoff.operator_dimension = 3

    def test_phase1_terminal_handoff_is_phase2_initial_object(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
        bekenstein_safe_inputs: Dict[str, float],
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₂ ∘ Φ₁:

        El objeto terminal de Fase 1 debe ser aceptado como prefijo formal
        por Fase 2 sin pérdida de identidad categórica.
        """
        enforcer = Phase2_BekensteinBoundEnforcer()

        phase2_handoff = enforcer._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            **bekenstein_safe_inputs,
        )

        assert isinstance(phase2_handoff, Phase2BekensteinHandoff)
        assert phase2_handoff.phase1_handoff is phase1_handoff
        assert phase2_handoff.bekenstein_audit.is_entropically_safe is True


# ══════════════════════════════════════════════════════════════════════════════
# §6. FASE 2 — REGULACIÓN TERMODINÁMICA DE BEKENSTEIN
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2BekensteinBoundEnforcer:
    """
    Fase 2: Regulación termodinámica.

    Objetivo:
        S <= 2π k_B E R / (ℏ c)

    El primer insumo formal es `Phase1HermiticityHandoff`.
    El último test valida que el objeto terminal de Fase 2 es aceptado
    por Fase 3 como prefijo formal.
    """

    def test_phase2_enforcer_is_instantiable(self) -> None:
        """Fase 2 debe poder instanciarse como extensión de Fase 1."""
        enforcer = Phase2_BekensteinBoundEnforcer()
        assert isinstance(enforcer, Phase2_BekensteinBoundEnforcer)
        assert isinstance(enforcer, Phase1_HermiticityAuditor)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.1. Certificación de escalares no negativos
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("value", [0.0, 1e-30, 1.0, 1e6])
    def test_certify_nonnegative_scalar_accepts_zero_and_positive(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        value: float,
    ) -> None:
        """Magnitudes físicas no negativas deben ser certificadas."""
        scalar = phase2_enforcer._certify_nonnegative_scalar("x", value)
        assert scalar == float(value)
        assert scalar >= 0.0

    @pytest.mark.parametrize("value", [-1e-20, -1.0, -1e30])
    def test_certify_nonnegative_scalar_rejects_negative(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        value: float,
    ) -> None:
        """Magnitudes físicas negativas deben violar el dominio."""
        with pytest.raises(DomainIntegrityViolationError):
            phase2_enforcer._certify_nonnegative_scalar("x", value)

    @pytest.mark.parametrize("value", [True, np.bool_(False)])
    def test_certify_nonnegative_scalar_rejects_booleans(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        value: Any,
    ) -> None:
        """Los booleanos no pertenecen al dominio escalar físico."""
        with pytest.raises(DomainIntegrityViolationError):
            phase2_enforcer._certify_nonnegative_scalar("x", value)

    @pytest.mark.parametrize("value", [0.0, 1e-20])
    def test_certify_nonnegative_scalar_strict_positive_rejects_non_positive(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        value: float,
    ) -> None:
        """En modo estricto, el escalar debe ser mayor que la tolerancia."""
        with pytest.raises(DomainIntegrityViolationError):
            phase2_enforcer._certify_nonnegative_scalar(
                "x",
                value,
                strict_positive=True,
            )

    def test_certify_nonnegative_scalar_strict_positive_accepts_macro_value(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """Un valor macroscópico debe superar la tolerancia absoluta."""
        scalar = phase2_enforcer._certify_nonnegative_scalar(
            "x",
            1e-6,
            strict_positive=True,
        )
        assert scalar == pytest.approx(1e-6)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.2. Certificación de radio causal
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("radius", [1e-9, 1e-3, 1.0, 10.0])
    def test_certify_positive_radius_accepts_valid_radii(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        radius: float,
    ) -> None:
        """Radios por encima de la tolerancia deben ser certificados."""
        certified_radius = phase2_enforcer._certify_positive_radius("R", radius)
        assert certified_radius == pytest.approx(radius)

    @pytest.mark.parametrize("radius", [0.0, -1e-9, -1.0])
    def test_certify_positive_radius_rejects_non_positive_radii(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        radius: float,
    ) -> None:
        """Un radio no positivo destruye la región causal."""
        with pytest.raises(BekensteinLimitViolation):
            phase2_enforcer._certify_positive_radius("R", radius)

    def test_certify_positive_radius_rejects_subtolerance_radius(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """Un radio por debajo de la tolerancia absoluta es rechazado."""
        with pytest.raises(BekensteinLimitViolation):
            phase2_enforcer._certify_positive_radius("R", 1e-15)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.3. Consistencia dimensional física
    # ─────────────────────────────────────────────────────────────────────────

    def test_dimensional_consistency_emits_warnings_for_extreme_values(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Valores extremos deben generar advertencias diagnósticas."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            phase2_enforcer._verify_dimensional_consistency(
                energy=1e11,
                radius=1e-21,
                entropy=1e11,
            )

        assert len(caplog.records) == 3

    def test_dimensional_consistency_does_not_warn_for_regular_values(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Valores regulares no deben generar advertencias."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            phase2_enforcer._verify_dimensional_consistency(
                energy=1.0,
                radius=1.0,
                entropy=1.0,
            )

        assert len(caplog.records) == 0

    # ─────────────────────────────────────────────────────────────────────────
    # 6.4. Métricas derivadas
    # ─────────────────────────────────────────────────────────────────────────

    def test_light_crossing_time_is_radius_over_c(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """El tiempo de cruce lumínico debe ser t = R/c."""
        radius = 123.456
        expected = radius / _C_EFF
        observed = phase2_enforcer._compute_light_crossing_time(radius)
        assert math.isclose(observed, expected, rel_tol=1e-15, abs_tol=0.0)

    @pytest.mark.parametrize("bound", [0.0, -1.0])
    def test_information_capacity_bits_is_zero_for_nonpositive_bound(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        bound: float,
    ) -> None:
        """Una cota no positiva no puede contener información."""
        bits = phase2_enforcer._compute_information_capacity_bits(bound)
        assert bits == 0.0

    def test_information_capacity_bits_positive_bound(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """Una cota positiva debe convertirse a bits vía Landauer."""
        bound = 10.0
        expected = bound / (_K_B * math.log(2.0))
        observed = phase2_enforcer._compute_information_capacity_bits(bound)
        assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.5. Imposición de la cota de Bekenstein
    # ─────────────────────────────────────────────────────────────────────────

    def test_enforce_bekenstein_bound_accepts_physical_safe_inputs(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        bekenstein_safe_inputs: Dict[str, float],
    ) -> None:
        """La aniquilación física segura debe pasar la cota."""
        audit = phase2_enforcer._enforce_bekenstein_gamma_bound(
            **bekenstein_safe_inputs,
        )

        assert audit.is_entropically_safe is True
        assert audit.entropy_emitted == pytest.approx(
            bekenstein_safe_inputs["emitted_entropy_S"]
        )
        assert audit.bekenstein_bound > 0.0
        assert audit.entropy_ratio >= 0.0
        assert audit.entropy_ratio <= 1.0
        assert audit.causal_light_crossing_time > 0.0
        assert audit.information_capacity_bits > 0.0

    def test_enforce_bekenstein_bound_accepts_exact_synthetic_bound(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """S = S_max debe ser aceptado dentro de tolerancias numéricas."""
        inputs = make_synthetic_bekenstein_inputs(entropy_fraction=1.0)
        audit = phase2_enforcer._enforce_bekenstein_gamma_bound(**inputs)

        assert audit.is_entropically_safe is True
        assert math.isclose(
            audit.entropy_ratio,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )

    def test_enforce_bekenstein_bound_accepts_half_synthetic_bound(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """S = S_max/2 debe ser aceptado y registrar ratio 0.5."""
        inputs = make_synthetic_bekenstein_inputs(entropy_fraction=0.5)
        audit = phase2_enforcer._enforce_bekenstein_gamma_bound(**inputs)

        assert audit.is_entropically_safe is True
        assert math.isclose(
            audit.entropy_ratio,
            0.5,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )

    def test_enforce_bekenstein_bound_rejects_entropy_excess(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """S > S_max debe detonar BekensteinLimitViolation."""
        inputs = make_synthetic_bekenstein_inputs(entropy_fraction=1.01)

        with pytest.raises(BekensteinLimitViolation):
            phase2_enforcer._enforce_bekenstein_gamma_bound(**inputs)

    def test_enforce_bekenstein_bound_accepts_zero_energy_zero_entropy(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """El vacío trivial E=0, S=0 debe ser seguro."""
        audit = phase2_enforcer._enforce_bekenstein_gamma_bound(
            gamma_energy=0.0,
            system_radius_R=1.0,
            emitted_entropy_S=0.0,
        )

        assert audit.is_entropically_safe is True
        assert audit.bekenstein_bound == pytest.approx(0.0)
        assert audit.information_capacity_bits == 0.0

    def test_enforce_bekenstein_bound_rejects_zero_energy_with_macro_entropy(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
    ) -> None:
        """E=0 con entropía macroscópica viola la cota causal."""
        with pytest.raises(BekensteinLimitViolation):
            phase2_enforcer._enforce_bekenstein_gamma_bound(
                gamma_energy=0.0,
                system_radius_R=1.0,
                emitted_entropy_S=1e-6,
            )

    def test_enforce_bekenstein_bound_rejects_nonfinite_bound(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Una cota no finita debe ser rechazada explícitamente."""
        monkeypatch.setattr(target, "_K_B", float("nan"))

        with pytest.raises(BekensteinLimitViolation):
            phase2_enforcer._enforce_bekenstein_gamma_bound(
                gamma_energy=1.0,
                system_radius_R=1.0,
                emitted_entropy_S=0.0,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 6.6. Handoff terminal de Fase 2
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase2_requires_phase1_handoff(
        self,
        phase2_enforcer: Phase2_BekensteinBoundEnforcer,
        bekenstein_safe_inputs: Dict[str, float],
    ) -> None:
        """Fase 2 debe rechazar cualquier prefijo que no sea Fase 1."""
        with pytest.raises(DomainIntegrityViolationError):
            phase2_enforcer._phase2_enforce_and_handoff_to_phase3(
                phase1_handoff=object(),
                **bekenstein_safe_inputs,
            )

    def test_phase2_handoff_preserves_phase1_certificate(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        phase1_handoff: Phase1HermiticityHandoff,
    ) -> None:
        """El handoff de Fase 2 debe preservar identidad del handoff de Fase 1."""
        assert phase2_handoff.phase1_handoff is phase1_handoff
        assert (
            phase2_handoff.phase1_handoff.hermiticity_audit
            is phase1_handoff.hermiticity_audit
        )

    def test_phase2_handoff_contains_safe_bekenstein_audit(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
    ) -> None:
        """El handoff de Fase 2 debe contener una auditoría entrópica segura."""
        assert phase2_handoff.bekenstein_audit.is_entropically_safe is True
        assert phase2_handoff.bekenstein_audit.bekenstein_bound >= 0.0
        assert np.isfinite(phase2_handoff.bekenstein_audit.entropy_ratio)

    def test_phase2_handoff_is_immutable(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
    ) -> None:
        """El handoff de Fase 2 debe ser inmutable."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            phase2_handoff.bekenstein_audit = None

    def test_phase2_terminal_handoff_is_phase3_initial_object(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        symplectic_inputs_2d: Dict[str, np.ndarray],
    ) -> None:
        """
        Lema de Continuación Funtorial Φ₃ ∘ Φ₂:

        El objeto terminal de Fase 2 debe ser aceptado como prefijo formal
        por Fase 3 y permitir la certificación simpléctica.
        """
        certifier = Phase3_SymplecticPortHamiltonianCertifier()

        state = certifier._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            **symplectic_inputs_2d,
        )

        assert isinstance(state, VacuumGovernanceState)
        assert state.is_epistemologically_valid is True


# ══════════════════════════════════════════════════════════════════════════════
# §7. FASE 3 — CERTIFICACIÓN SIMPLÉCTICA PORT-HAMILTONIANA
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3SymplecticPortHamiltonianCertifier:
    """
    Fase 3: Certificación simpléctica y disipación Port-Hamiltoniana.

    Condiciones:
        1. Mᵀ Ω M = Ω
        2. J = -Jᵀ
        3. R = Rᵀ ⪰ 0
        4. Ḣ = -∇Hᵀ R ∇H <= 0

    El primer insumo formal es `Phase2BekensteinHandoff`.
    """

    def test_phase3_certifier_is_instantiable(self) -> None:
        """Fase 3 debe poder instanciarse como extensión de Fase 2."""
        certifier = Phase3_SymplecticPortHamiltonianCertifier()
        assert isinstance(certifier, Phase3_SymplecticPortHamiltonianCertifier)
        assert isinstance(certifier, Phase2_BekensteinBoundEnforcer)
        assert isinstance(certifier, Phase1_HermiticityAuditor)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.1. Forma simpléctica canónica
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [0, -1, -2])
    def test_build_symplectic_form_rejects_nonpositive_dof(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        n: int,
    ) -> None:
        """El número de grados de libertad debe ser positivo."""
        with pytest.raises(PhaseSpaceTopologyError):
            phase3_certifier._build_symplectic_form(n)

    @pytest.mark.parametrize("n", [1, 2])
    def test_build_symplectic_form_canonical_properties(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        n: int,
    ) -> None:
        """La 2-forma canónica debe ser antisimétrica, cuadrática y unitaria."""
        omega = phase3_certifier._build_symplectic_form(n)
        dim = 2 * n

        assert omega.shape == (dim, dim)
        np.testing.assert_allclose(omega.T, -omega, atol=1e-15)
        np.testing.assert_allclose(omega @ omega, -np.eye(dim), atol=1e-15)

        det = float(np.linalg.det(omega))
        assert det == pytest.approx(1.0, abs=1e-8)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.2. Topología del espacio de fase
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dim, expected_dof", [(2, 1), (4, 2), (6, 3)])
    def test_verify_phase_space_dimension_accepts_even_dimensions(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        dim: int,
        expected_dof: int,
    ) -> None:
        """El teorema de Darboux exige dimensión par."""
        dof = phase3_certifier._verify_phase_space_dimension(dim)
        assert dof == expected_dof

    @pytest.mark.parametrize("dim", [1, 3, 5])
    def test_verify_phase_space_dimension_rejects_odd_dimensions(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        dim: int,
    ) -> None:
        """Una dimensión impar no admite estructura simpléctica."""
        with pytest.raises(PhaseSpaceTopologyError):
            phase3_certifier._verify_phase_space_dimension(dim)

    @pytest.mark.parametrize("dim", [0, -2])
    def test_verify_phase_space_dimension_rejects_nonpositive_dimensions(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        dim: int,
    ) -> None:
        """Una dimensión no positiva es topológicamente inválida."""
        with pytest.raises(PhaseSpaceTopologyError):
            phase3_certifier._verify_phase_space_dimension(dim)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.3. Antisimetría de J
    # ─────────────────────────────────────────────────────────────────────────

    def test_certify_antisymmetric_matrix_accepts_canonical_J(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una matriz J canónica debe ser certificada como antisimétrica."""
        J = make_canonical_antisymmetric_matrix(4)
        residual = phase3_certifier._certify_antisymmetric_matrix("J", J, 4)
        assert residual == pytest.approx(0.0, abs=1e-15)

    def test_certify_antisymmetric_matrix_accepts_tolerated_noise(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Ruido simétrico infinitesimal dentro de tolerancia debe pasar."""
        J = make_canonical_antisymmetric_matrix(4)
        J_noisy = J + 1e-12 * np.ones((4, 4), dtype=np.float64)

        residual = phase3_certifier._certify_antisymmetric_matrix("J", J_noisy, 4)
        assert np.isfinite(residual)
        assert residual <= 1e-10

    def test_certify_antisymmetric_matrix_rejects_symmetric_part(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una matriz simétrica no puede ser matriz de interconexión."""
        J_bad = np.eye(4, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_antisymmetric_matrix("J", J_bad, 4)

    def test_certify_antisymmetric_matrix_rejects_wrong_dimension(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """La dimensión esperada debe coincidir exactamente."""
        J = make_canonical_antisymmetric_matrix(4)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_antisymmetric_matrix("J", J, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.4. Simetría y semidefinición positiva de R
    # ─────────────────────────────────────────────────────────────────────────

    def test_certify_symmetric_positive_semidefinite_accepts_identity(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """R = I es simétrica y definida positiva."""
        R = np.eye(3, dtype=np.float64)
        R_cert, sym_res, min_eig, max_eig = (
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                3,
            )
        )

        np.testing.assert_allclose(R_cert, R)
        assert sym_res == pytest.approx(0.0, abs=1e-15)
        assert min_eig == pytest.approx(1.0)
        assert max_eig == pytest.approx(1.0)

    def test_certify_symmetric_positive_semidefinite_accepts_zero(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """R = 0 es semidefinida positiva y representa disipación nula."""
        R = np.zeros((2, 2), dtype=np.float64)
        R_cert, sym_res, min_eig, max_eig = (
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                2,
            )
        )

        np.testing.assert_allclose(R_cert, R)
        assert sym_res == pytest.approx(0.0, abs=1e-15)
        assert min_eig == pytest.approx(0.0, abs=1e-15)
        assert max_eig == pytest.approx(0.0, abs=1e-15)

    def test_certify_symmetric_positive_semidefinite_accepts_tiny_negative_within_tolerance(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Autovalores negativos dentro de tolerancia numérica se toleran."""
        R = np.diag([-1e-13, 1.0]).astype(np.float64)

        R_cert, sym_res, min_eig, max_eig = (
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                2,
            )
        )

        assert np.all(np.isfinite(R_cert))
        assert sym_res == pytest.approx(0.0, abs=1e-15)
        assert min_eig < 0.0
        assert min_eig >= -1e-12
        assert max_eig == pytest.approx(1.0)

    def test_certify_symmetric_positive_semidefinite_rejects_negative_eigenvalue(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Un autovalor negativo macroscópico viola pasividad."""
        R = np.diag([-1e-2, 1.0]).astype(np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                2,
            )

    def test_certify_symmetric_positive_semidefinite_rejects_nonsymmetric_matrix(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """R debe ser simétrica antes de evaluar semidefinición positiva."""
        R = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                2,
            )

    def test_certify_symmetric_positive_semidefinite_rejects_wrong_dimension(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """La dimensión de R debe coincidir con el espacio de fase."""
        R = np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_symmetric_positive_semidefinite_matrix(
                "R",
                R,
                3,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.5. Invarianza de volumen simpléctico
    # ─────────────────────────────────────────────────────────────────────────

    def test_symplectic_volume_preservation_accepts_identity(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """M = I preserva trivialmente la 2-forma canónica."""
        omega = phase3_certifier._build_symplectic_form(1)
        M = np.eye(2, dtype=np.float64)

        residual, det_residual = (
            phase3_certifier._verify_symplectic_volume_preservation(M, omega)
        )

        assert residual == pytest.approx(0.0, abs=1e-15)
        assert det_residual == pytest.approx(0.0, abs=1e-12)

    def test_symplectic_volume_preservation_accepts_shear(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una cizalla con determinante unitario es simpléctica en 2D."""
        omega = phase3_certifier._build_symplectic_form(1)
        M = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        residual, det_residual = (
            phase3_certifier._verify_symplectic_volume_preservation(M, omega)
        )

        assert residual <= 1e-10
        assert det_residual <= 1e-9

    def test_symplectic_volume_preservation_rejects_scaling(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una homotecia M = 2I destruye el volumen simpléctico."""
        omega = phase3_certifier._build_symplectic_form(1)
        M = 2.0 * np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._verify_symplectic_volume_preservation(M, omega)

    def test_symplectic_volume_preservation_rejects_nonfinite_transformation(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una transformación no finita debe detonar colapso simpléctico."""
        omega = phase3_certifier._build_symplectic_form(1)
        M = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._verify_symplectic_volume_preservation(M, omega)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.6. Disipación Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────

    def test_audit_dissipation_is_nonpositive_for_psd_R(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Con R ⪰ 0, la tasa de energía debe ser no positiva."""
        grad_H = np.array([1.0, 2.0], dtype=np.float64)
        R = np.eye(2, dtype=np.float64)

        h_dot = phase3_certifier._audit_port_hamiltonian_dissipation(grad_H, R)

        assert h_dot <= 0.0
        assert math.isclose(h_dot, -5.0, rel_tol=1e-15, abs_tol=0.0)

    def test_audit_dissipation_zero_for_zero_gradient(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Si ∇H = 0, la tasa de disipación es exactamente cero."""
        grad_H = np.zeros(2, dtype=np.float64)
        R = np.eye(2, dtype=np.float64)

        h_dot = phase3_certifier._audit_port_hamiltonian_dissipation(grad_H, R)
        assert h_dot == pytest.approx(0.0, abs=1e-15)

    def test_audit_dissipation_rejects_energy_injection(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una matriz R negativa inyecta energía y viola la segunda ley."""
        grad_H = np.array([1.0, 1.0], dtype=np.float64)
        R = -np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._audit_port_hamiltonian_dissipation(grad_H, R)

    def test_audit_dissipation_rejects_nonfinite_rate(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una tasa de disipación no finita debe ser rechazada."""
        grad_H = np.array([np.nan, 1.0], dtype=np.float64)
        R = np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._audit_port_hamiltonian_dissipation(grad_H, R)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.7. Certificación completa Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dim", [2, 4])
    def test_certify_symplectic_port_hamiltonian_valid(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        dim: int,
    ) -> None:
        """Un sistema Port-Hamiltoniano válido debe certificarse."""
        inputs = make_symplectic_inputs(dim=dim, dissipation=0.2, seed=dim)

        audit = phase3_certifier._certify_symplectic_port_hamiltonian(**inputs)

        assert audit.is_symplectically_invariant is True
        assert audit.phase_space_dimension == dim
        assert audit.degrees_of_freedom == dim // 2
        assert audit.dissipation_rate <= 0.0
        assert np.isfinite(audit.symplectic_residual)
        assert np.isfinite(audit.determinant_residual)
        assert np.isfinite(audit.antisymmetry_residual)
        assert np.isfinite(audit.r_symmetry_residual)
        assert audit.r_min_eigenvalue >= -1e-12

    def test_certify_symplectic_port_hamiltonian_rejects_odd_dimension(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una dimensión impar debe violar la topología del espacio de fase."""
        M = np.eye(1, dtype=np.float64)
        grad_H = np.ones(1, dtype=np.float64)
        J = np.zeros((1, 1), dtype=np.float64)
        R = np.eye(1, dtype=np.float64)

        with pytest.raises(PhaseSpaceTopologyError):
            phase3_certifier._certify_symplectic_port_hamiltonian(
                jacobian_M=M,
                grad_H=grad_H,
                J_matrix=J,
                R_matrix=R,
            )

    def test_certify_symplectic_port_hamiltonian_rejects_nonfinite_jacobian(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Un jacobiano no finito viola la integridad de dominio."""
        inputs = make_symplectic_inputs(dim=2, dissipation=0.1, seed=3)
        inputs["jacobian_M"] = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64)

        with pytest.raises(DomainIntegrityViolationError):
            phase3_certifier._certify_symplectic_port_hamiltonian(**inputs)

    def test_certify_symplectic_port_hamiltonian_rejects_wrong_grad_dimension(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """El gradiente hamiltoniano debe tener la dimensión del espacio de fase."""
        inputs = make_symplectic_inputs(dim=2, dissipation=0.1, seed=5)
        inputs["grad_H"] = np.ones(3, dtype=np.float64)

        with pytest.raises(DomainIntegrityViolationError):
            phase3_certifier._certify_symplectic_port_hamiltonian(**inputs)

    def test_certify_symplectic_port_hamiltonian_rejects_invalid_J(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una matriz J no antisimétrica debe detonar colapso simpléctico."""
        inputs = make_symplectic_inputs(dim=2, dissipation=0.1, seed=7)
        inputs["J_matrix"] = np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_symplectic_port_hamiltonian(**inputs)

    def test_certify_symplectic_port_hamiltonian_rejects_invalid_R(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
    ) -> None:
        """Una matriz R negativa debe violar la condición de pasividad."""
        inputs = make_symplectic_inputs(dim=2, dissipation=0.1, seed=9)
        inputs["R_matrix"] = -np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._certify_symplectic_port_hamiltonian(**inputs)

    # ─────────────────────────────────────────────────────────────────────────
    # 7.8. Finalización funtorial de Fase 3
    # ─────────────────────────────────────────────────────────────────────────

    def test_phase3_finalize_requires_phase2_handoff(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        symplectic_inputs_2d: Dict[str, np.ndarray],
    ) -> None:
        """Fase 3 debe rechazar cualquier prefijo que no sea Fase 2."""
        with pytest.raises(DomainIntegrityViolationError):
            phase3_certifier._phase3_finalize_from_phase2_handoff(
                phase2_handoff=object(),
                **symplectic_inputs_2d,
            )

    def test_phase3_finalize_returns_vacuum_state_from_phase2_handoff(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        phase2_handoff: Phase2BekensteinHandoff,
        symplectic_inputs_2d: Dict[str, np.ndarray],
    ) -> None:
        """La finalización debe producir un VacuumGovernanceState válido."""
        state = phase3_certifier._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            **symplectic_inputs_2d,
        )

        assert isinstance(state, VacuumGovernanceState)
        assert state.is_epistemologically_valid is True
        assert state.hermiticity_audit.is_hermitian is True
        assert state.bekenstein_audit.is_entropically_safe is True
        assert state.symplectic_audit.is_symplectically_invariant is True

    def test_phase3_finalize_propagates_phase1_and_phase2_certificates(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        phase2_handoff: Phase2BekensteinHandoff,
        symplectic_inputs_2d: Dict[str, np.ndarray],
    ) -> None:
        """Los certificados previos deben propagarse sin mutación."""
        state = phase3_certifier._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            **symplectic_inputs_2d,
        )

        assert state.hermiticity_audit is (
            phase2_handoff.phase1_handoff.hermiticity_audit
        )
        assert state.bekenstein_audit is phase2_handoff.bekenstein_audit

    def test_phase3_finalize_rejects_invalid_symplectic_input(
        self,
        phase3_certifier: Phase3_SymplecticPortHamiltonianCertifier,
        phase2_handoff: Phase2BekensteinHandoff,
        symplectic_inputs_2d: Dict[str, np.ndarray],
    ) -> None:
        """Una entrada simpléctica inválida debe impedir la finalización."""
        bad_inputs = dict(symplectic_inputs_2d)
        bad_inputs["jacobian_M"] = 2.0 * np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            phase3_certifier._phase3_finalize_from_phase2_handoff(
                phase2_handoff=phase2_handoff,
                **bad_inputs,
            )


# ══════════════════════════════════════════════════════════════════════════════
# §8. ORQUESTADOR SUPREMO — ENDOFUNTOR COMPLETO
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndVacuumGovernance:
    """
    Orquestación completa:

        Z_Vacuum = Φ₃ ∘ Φ₂ ∘ Φ₁

    Estos tests validan el diagrama conmutativo completo, la metadata
    de gobernanza y la inmutabilidad del estado terminal.
    """

    def test_agent_is_phase3_subclass_and_morphism_if_available(self) -> None:
        """El agente debe componer todas las fases y ser morfismo si aplica."""
        agent_instance = AntimatterChokeCoilAgent()

        assert isinstance(agent_instance, Phase3_SymplecticPortHamiltonianCertifier)
        assert isinstance(agent_instance, Phase2_BekensteinBoundEnforcer)
        assert isinstance(agent_instance, Phase1_HermiticityAuditor)

        if Morphism is not None:
            assert isinstance(agent_instance, Morphism)

    def test_execute_vacuum_governance_valid_end_to_end(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """El endofuntor completo debe certificar un vacío cuántico válido."""
        inputs = make_valid_agent_inputs(dim=2, seed=17)
        state = agent.execute_vacuum_governance(**inputs)

        assert isinstance(state, VacuumGovernanceState)
        assert state.is_epistemologically_valid is True

        assert state.hermiticity_audit.is_hermitian is True
        assert state.bekenstein_audit.is_entropically_safe is True
        assert state.symplectic_audit.is_symplectically_invariant is True
        assert state.symplectic_audit.dissipation_rate <= 0.0

    def test_callable_alias_returns_governance_state(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """__call__ debe ser un alias del endofuntor de gobierno."""
        inputs = make_valid_agent_inputs(dim=2, seed=19)
        state = agent(**inputs)

        assert isinstance(state, VacuumGovernanceState)
        assert state.is_epistemologically_valid is True

    def test_execute_accepts_zero_dissipation(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """R = 0 representa disipación nula y debe ser válido."""
        inputs = make_valid_agent_inputs(dim=2, seed=23)
        inputs["R_matrix"] = np.zeros((2, 2), dtype=np.float64)

        state = agent.execute_vacuum_governance(**inputs)

        assert state.is_epistemologically_valid is True
        assert state.symplectic_audit.dissipation_rate == pytest.approx(
            0.0,
            abs=1e-15,
        )

    def test_execute_accepts_shear_jacobian(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """Un jacobiano de cizalla con det=1 debe preservar el volumen."""
        inputs = make_valid_agent_inputs(dim=2, seed=29)
        inputs["jacobian_M"] = np.array(
            [[1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )

        state = agent.execute_vacuum_governance(**inputs)

        assert state.is_epistemologically_valid is True
        assert state.symplectic_audit.is_symplectically_invariant is True

    def test_execute_rejects_non_hermitian_operator(
        self,
        agent: AntimatterChokeCoilAgent,
        non_hermitian_operator_2x2: np.ndarray,
    ) -> None:
        """Fase 1 debe abortar la composición si A ≠ A†."""
        inputs = make_valid_agent_inputs(dim=2, seed=31)
        inputs["operator_A"] = non_hermitian_operator_2x2

        with pytest.raises(NonHermitianOperatorError):
            agent.execute_vacuum_governance(**inputs)

    def test_execute_rejects_non_square_operator(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """Un operador no cuadrado viola el dominio de Fase 1."""
        inputs = make_valid_agent_inputs(dim=2, seed=37)
        inputs["operator_A"] = np.ones((2, 3), dtype=np.complex128)

        with pytest.raises(DomainIntegrityViolationError):
            agent.execute_vacuum_governance(**inputs)

    def test_execute_rejects_bekenstein_violation(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """Fase 2 debe abortar si la entropía excede la cota causal."""
        inputs = make_valid_agent_inputs(dim=2, seed=41)
        inputs["emitted_entropy_S"] = 1.0e-6

        with pytest.raises(BekensteinLimitViolation):
            agent.execute_vacuum_governance(**inputs)

    def test_execute_rejects_symplectic_collapse(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """Fase 3 debe abortar si se destruye el volumen simpléctico."""
        inputs = make_valid_agent_inputs(dim=2, seed=43)
        inputs["jacobian_M"] = 2.0 * np.eye(2, dtype=np.float64)

        with pytest.raises(SymplecticCollapseError):
            agent.execute_vacuum_governance(**inputs)

    def test_execute_with_four_dimensional_phase_space(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """El endofuntor debe funcionar en espacios de fase 4D."""
        inputs = make_valid_agent_inputs(dim=4, seed=47)
        state = agent.execute_vacuum_governance(**inputs)

        assert state.is_epistemologically_valid is True
        assert state.symplectic_audit.phase_space_dimension == 4
        assert state.symplectic_audit.degrees_of_freedom == 2

    def test_governance_metadata_is_complete_and_finite(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """La metadata de gobernanza debe ser completa, finita y coherente."""
        inputs = make_valid_agent_inputs(dim=4, seed=53)
        state = agent.execute_vacuum_governance(**inputs)

        meta = state.governance_metadata

        expected_keys = {
            "functor_composition",
            "phase1_residual",
            "phase2_entropy_ratio",
            "phase3_dissipation_rate",
            "phase_space_dof",
            "information_capacity_bits",
        }
        assert expected_keys.issubset(meta.keys())

        assert meta["functor_composition"] == "Φ₃ ∘ Φ₂ ∘ Φ₁"
        assert meta["phase_space_dof"] == 2

        assert np.isfinite(float(meta["phase1_residual"]))
        assert np.isfinite(float(meta["phase2_entropy_ratio"]))
        assert np.isfinite(float(meta["phase3_dissipation_rate"]))
        assert np.isfinite(float(meta["information_capacity_bits"]))

        assert float(meta["phase1_residual"]) >= 0.0
        assert float(meta["phase2_entropy_ratio"]) >= 0.0
        assert float(meta["phase3_dissipation_rate"]) <= 0.0
        assert float(meta["information_capacity_bits"]) >= 0.0

    def test_vacuum_governance_state_is_immutable(
        self,
        agent: AntimatterChokeCoilAgent,
    ) -> None:
        """El objeto terminal del endofuntor debe ser inmutable."""
        inputs = make_valid_agent_inputs(dim=2, seed=59)
        state = agent.execute_vacuum_governance(**inputs)

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.is_epistemologically_valid = False

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.hermiticity_audit.residual_norm = 1.0

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.governance_metadata = {}