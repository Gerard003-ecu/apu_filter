r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_gromov_witten_auditor_agent.py                                 ║
║ Ruta   : tests/unit/agents/omega/test_gromov_witten_auditor_agent.py         ║
║ Objetivo: Validación granular, termodinámica, simpléctica y espectral del    ║
║          endofuntor 𝒜 = Φ₃ᴳᵂ ∘ Φ₂ᴳᵂ ∘ Φ₁ᴳᵂ                                   ║
║          (Bekenstein → Cartan/Ehresmann → Gromov–Witten/APS).                ║
║ Framework: pytest + numpy + scipy                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de la suite (espejo de las fases anidadas del agente):
  §G0  Fixtures canónicas y utilidades espectrales / termodinámicas
  §G1  Jerarquía de excepciones y contrato taxonómico
  §G2  DTOs inmutables (objetos del Topos)
  §G3  Fase 1 — Bekenstein / von Neumann / dimensionamiento del baño
  §G4  Fase 2 — Cinemática de Cartan / Ehresmann / no-demolición
  §G5  Fase 3 — Gromov–Witten / Maurer–Cartan / compensación APS
  §G6  Orquestador GromovWittenAuditorAgent (composición 𝒜)
  §G7  Invariantes de extremo a extremo y regresión numérica
  §G8  Casos límite, patologías, robustez y superficie pública
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# SUT — Agente Gromov–Witten
# ─────────────────────────────────────────────────────────────────────────────
from app.agents.omega.gromov_witten_auditor_agent import (
    # Excepciones
    GromovWittenAuditorError,
    BekensteinLimitViolationError,
    BusinessInvarianceError,
    CartanStructureError,
    SpectralCompensationError,
    AuditingPipelineError,
    # DTOs
    BekensteinDimensionData,
    CartanKinematicsData,
    MaurerCartanSolution,
    GromovWittenCompensation,
    AuditBundle,
    # Fases
    Phase1_Bekenstein,
    Phase2_EhresmannCartan,
    Phase3_GromovWitten,
    # Orquestador
    GromovWittenAuditorAgent,
)

# Motor telescópico (DTOs / excepciones compartidas)
from app.core.immune_system.ehresmann_telescopic_engine import (
    StinespringDilationData,
    VerticalFibrationData,
    TelescopicAuditState,
    InvalidDensityMatrixError,
    StinespringDilationError,
    EhresmannFibrationError,
    SphereBubblingAnomalyError,
)

# Tolerancias espejo del agente
_ENTROPY_TOL = 1e-12
_BUS_INV_TOL = 1e-10
_CARTAN_CURV_TOL = 1e-10
_MC_TOL = 1e-9
_ISO_TOL = 1e-12
_HS_FLOOR = 1e-16
_MIN_AUDIT_DIM = 2
_COMPLEX = np.complex128
_FLOAT = np.float64


# ══════════════════════════════════════════════════════════════════════════════
# §G0 · FIXTURES CANÓNICAS Y UTILIDADES ESPECTRALES / TERMODINÁMICAS
# ══════════════════════════════════════════════════════════════════════════════

def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (M + M.conj().T)


def _random_density(n: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """ρ ∈ 𝔇(ℂⁿ) vía Wishart normalizado."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    rho = _hermitize(rho)
    rho /= np.trace(rho).real
    return rho.astype(_COMPLEX)


def _pure_state_density(n: int, idx: int = 0) -> NDArray[np.complex128]:
    rho = np.zeros((n, n), dtype=_COMPLEX)
    rho[idx, idx] = 1.0 + 0.0j
    return rho


def _maximally_mixed(n: int) -> NDArray[np.complex128]:
    return (np.eye(n, dtype=_COMPLEX) / n).astype(_COMPLEX)


def _thermal_density(n: int, beta: float = 1.0) -> NDArray[np.complex128]:
    """Estado térmico ρ ∝ exp(−β H) con H = diag(0,1,…,n−1)."""
    levels = np.arange(n, dtype=_FLOAT)
    weights = np.exp(-beta * levels)
    weights /= weights.sum()
    return np.diag(weights.astype(_COMPLEX))


def _is_hermitian(M: NDArray, tol: float = 1e-12) -> bool:
    return float(la.norm(M - M.conj().T, ord="fro")) <= tol


def _is_psd(M: NDArray, floor: float = -1e-12) -> bool:
    ev = la.eigvalsh(_hermitize(M))
    return bool(np.all(ev >= floor))


def _is_density(M: NDArray, tol: float = 1e-12) -> bool:
    return (
        M.ndim == 2
        and M.shape[0] == M.shape[1]
        and _is_hermitian(M)
        and _is_psd(M)
        and abs(float(np.real(np.trace(M))) - 1.0) <= tol
    )


def _fro(A: NDArray) -> float:
    return float(la.norm(A, ord="fro"))


def _von_neumann_reference(rho: NDArray[np.complex128]) -> float:
    """S(ρ) de referencia independiente del SUT."""
    ev = la.eigvalsh(_hermitize(rho))
    ev = np.clip(ev.real, 0.0, None)
    s = ev.sum()
    if s < 1e-15:
        return 0.0
    ev = ev / s
    pos = ev[ev > 1e-15]
    if pos.size == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=77_011)


@pytest.fixture
def phase1() -> Phase1_Bekenstein:
    return Phase1_Bekenstein()


@pytest.fixture
def phase2() -> Phase2_EhresmannCartan:
    return Phase2_EhresmannCartan()


@pytest.fixture
def phase3() -> Phase3_GromovWitten:
    return Phase3_GromovWitten()


@pytest.fixture
def agent() -> GromovWittenAuditorAgent:
    return GromovWittenAuditorAgent(default_eta_invariant=0.0)


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


@pytest.fixture
def rho_thermal_4() -> NDArray[np.complex128]:
    return _thermal_density(4, beta=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# §G1 · JERARQUÍA DE EXCEPCIONES Y CONTRATO TAXONÓMICO
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Taxonomía de excepciones simplécticas / cuánticas del agente."""

    def test_root_is_exception(self):
        assert issubclass(GromovWittenAuditorError, Exception)

    def test_leaves_inherit_from_root(self):
        leaves = (
            BekensteinLimitViolationError,
            BusinessInvarianceError,
            CartanStructureError,
            SpectralCompensationError,
            AuditingPipelineError,
        )
        for exc in leaves:
            assert issubclass(exc, GromovWittenAuditorError), f"{exc.__name__}"

    def test_raisable_and_catchable_via_root(self):
        with pytest.raises(GromovWittenAuditorError):
            raise BekensteinLimitViolationError("dim insuficiente")
        with pytest.raises(GromovWittenAuditorError):
            raise BusinessInvarianceError("Δρ excesivo")
        with pytest.raises(GromovWittenAuditorError):
            raise CartanStructureError("Ω patológica")
        with pytest.raises(GromovWittenAuditorError):
            raise SpectralCompensationError("GW divergente")
        with pytest.raises(GromovWittenAuditorError):
            raise AuditingPipelineError("orquestación rota")

    def test_leaf_specific_catch(self):
        with pytest.raises(BusinessInvarianceError):
            raise BusinessInvarianceError("no-demolición violada")
        with pytest.raises(SpectralCompensationError):
            raise SpectralCompensationError("η_eff nan")


# ══════════════════════════════════════════════════════════════════════════════
# §G2 · DTOs INMUTABLES (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════

class TestDTOImmutability:
    """Los artefactos de fase deben ser frozen (inmutables)."""

    def _sample_bekenstein(self) -> BekensteinDimensionData:
        return BekensteinDimensionData(
            von_neumann_entropy=0.5,
            required_audit_dimension=3,
            rho_mic_eigenvalues=np.array([0.7, 0.3], dtype=_FLOAT),
            effective_hilbert_dimension=float(math.exp(0.5)),
            bekenstein_saturated=False,
        )

    def test_bekenstein_dto_frozen(self):
        dto = self._sample_bekenstein()
        with pytest.raises(AttributeError):
            dto.von_neumann_entropy = 1.0  # type: ignore[misc]

    def test_bekenstein_fields_accessible(self):
        dto = self._sample_bekenstein()
        assert dto.required_audit_dimension >= 2
        assert dto.effective_hilbert_dimension > 0.0
        assert dto.rho_mic_eigenvalues.shape[0] == 2

    def test_gw_compensation_frozen(self):
        dto = GromovWittenCompensation(
            gw_invariant_volume=0.01,
            gw_chern_simons_secondary=0.0,
            raw_eta_invariant=0.1,
            effective_eta_invariant=0.09,
            bubble_area_class=0.1414,
            is_ready_for_atiyah_singer=True,
        )
        with pytest.raises(AttributeError):
            dto.is_ready_for_atiyah_singer = False  # type: ignore[misc]

    def test_maurer_cartan_solution_frozen(self, rng):
        rho = _maximally_mixed(2)
        b = np.zeros((2, 2), dtype=_COMPLEX)
        # Construir TelescopicAuditState mínimo compatible con motor v3
        try:
            audit = TelescopicAuditState(
                audited_density_matrix=rho,
                landau_ginzburg_potential=0.0,
                novikov_convergence_iterations=1,
                maurer_cartan_bounding_cochain=b,
                novikov_filtration_degree=0.0,
                is_safe_for_witten_atiyah=True,
            )
        except TypeError:
            # Stub / firma alternativa
            audit = TelescopicAuditState(  # type: ignore[call-arg]
                audited_density_matrix=rho,
                landau_ginzburg_potential=0.0,
                novikov_convergence_iterations=1,
                is_safe_for_witten_atiyah=True,
            )
        dto = MaurerCartanSolution(
            b_cochain=b,
            audit_state=audit,
            residual_frobenius=0.0,
            novikov_filtration_degree=0.0,
        )
        with pytest.raises(AttributeError):
            dto.residual_frobenius = 1.0  # type: ignore[misc]

    def test_cartan_kinematics_and_bundle_require_nested_dtos(self, phase1, rho_2x2_mixed):
        """Construcción real vía pipeline parcial para validar frozen de Cartan/Bundle."""
        bek = phase1.enforce_bekenstein_bound(rho_2x2_mixed)
        assert isinstance(bek, BekensteinDimensionData)
        with pytest.raises(AttributeError):
            bek.required_audit_dimension = 99  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# §G3 · FASE 1 — BEKENSTEIN / VON NEUMANN / DIMENSIONAMIENTO DEL BAÑO
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1VonNeumannEntropy:
    """§1.1 — Entropía de von Neumann S(ρ) = −Tr(ρ ln ρ)."""

    def test_pure_state_entropy_zero(self, phase1, rho_2x2_pure):
        S, ev = phase1._compute_von_neumann_entropy(rho_2x2_pure)
        assert S == pytest.approx(0.0, abs=1e-10)
        assert ev.sum() == pytest.approx(1.0, abs=1e-12)

    def test_maximally_mixed_entropy_log_n(self, phase1, rho_maximally_mixed_3):
        n = 3
        S, ev = phase1._compute_von_neumann_entropy(rho_maximally_mixed_3)
        assert S == pytest.approx(math```python
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_gromov_witten_auditor_agent.py                                 ║
║ Ruta   : tests/unit/agents/omega/test_gromov_witten_auditor_agent.py         ║
║ Objetivo: Validación granular, termodinámica, simpléctica y espectral del    ║
║          endofuntor 𝒜 = Φ₃ᴳᵂ ∘ Φ₂ᴳᵂ ∘ Φ₁ᴳᵂ                                   ║
║          (Bekenstein → Cartan/Ehresmann → Gromov–Witten/APS).                ║
║ Framework: pytest + numpy + scipy                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de la suite (espejo de las fases anidadas del agente):
  §G0  Fixtures canónicas y utilidades espectrales / termodinámicas
  §G1  Jerarquía de excepciones y contrato taxonómico
  §G2  DTOs inmutables (objetos del Topos)
  §G3  Fase 1 — Bekenstein / von Neumann / dimensionamiento del baño
  §G4  Fase 2 — Cinemática de Cartan / Ehresmann / no-demolición
  §G5  Fase 3 — Gromov–Witten / Maurer–Cartan / compensación APS
  §G6  Orquestador GromovWittenAuditorAgent (composición 𝒜)
  §G7  Invariantes de extremo a extremo y regresión numérica
  §G8  Casos límite, patologías, robustez y superficie pública
"""

from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# SUT — Agente Gromov–Witten
# ─────────────────────────────────────────────────────────────────────────────
from app.agents.omega.gromov_witten_auditor_agent import (
    # Excepciones
    GromovWittenAuditorError,
    BekensteinLimitViolationError,
    BusinessInvarianceError,
    CartanStructureError,
    SpectralCompensationError,
    AuditingPipelineError,
    # DTOs
    BekensteinDimensionData,
    CartanKinematicsData,
    MaurerCartanSolution,
    GromovWittenCompensation,
    AuditBundle,
    # Fases
    Phase1_Bekenstein,
    Phase2_EhresmannCartan,
    Phase3_GromovWitten,
    # Orquestador
    GromovWittenAuditorAgent,
)

# Motor telescópico (DTOs / excepciones compartidas)
from app.core.immune_system.ehresmann_telescopic_engine import (
    StinespringDilationData,
    VerticalFibrationData,
    TelescopicAuditState,
    InvalidDensityMatrixError,
    StinespringDilationError,
    EhresmannFibrationError,
    SphereBubblingAnomalyError,
)

# Tolerancias espejo del agente / motor
_ENTROPY_TOL = 1e-12
_BUS_INV_TOL = 1e-10
_CARTAN_CURV_TOL = 1e-10
_MC_TOL = 1e-9
_ISO_TOL = 1e-12
_HS_FLOOR = 1e-16
_MIN_AUDIT_DIM = 2
_MAX_AUDIT_DIM = 256
_GW_CEILING = 1e12
_COMPLEX = np.complex128
_FLOAT = np.float64


# ══════════════════════════════════════════════════════════════════════════════
# §G0 · FIXTURES CANÓNICAS Y UTILIDADES ESPECTRALES / TERMODINÁMICAS
# ══════════════════════════════════════════════════════════════════════════════

def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (M + M.conj().T)


def _random_density(n: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """ρ ∈ 𝔇(ℂⁿ) vía Wishart normalizado."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    rho = _hermitize(rho)
    rho /= np.trace(rho).real
    return rho.astype(_COMPLEX)


def _pure_state_density(n: int, idx: int = 0) -> NDArray[np.complex128]:
    rho = np.zeros((n, n), dtype=_COMPLEX)
    rho[idx, idx] = 1.0 + 0.0j
    return rho


def _maximally_mixed(n: int) -> NDArray[np.complex128]:
    return (np.eye(n, dtype=_COMPLEX) / n).astype(_COMPLEX)


def _thermal_density(n: int, beta: float = 1.0) -> NDArray[np.complex128]:
    """Estado térmico ρ ∝ exp(−β H) con H = diag(0,1,…,n−1)."""
    levels = np.arange(n, dtype=_FLOAT)
    weights = np.exp(-beta * levels)
    weights /= weights.sum()
    return np.diag(weights.astype(_COMPLEX))


def _is_hermitian(M: NDArray, tol: float = 1e-12) -> bool:
    return float(la.norm(M - M.conj().T, ord="fro")) <= tol


def _is_psd(M: NDArray, floor: float = -1e-12) -> bool:
    ev = la.eigvalsh(_hermitize(M))
    return bool(np.all(ev >= floor))


def _is_density(M: NDArray, tol: float = 1e-12) -> bool:
    return (
        M.ndim == 2
        and M.shape[0] == M.shape[1]
        and _is_hermitian(M)
        and _is_psd(M)
        and abs(float(np.real(np.trace(M))) - 1.0) <= tol
    )


def _fro(A: NDArray) -> float:
    return float(la.norm(A, ord="fro"))


def _von_neumann_reference(rho: NDArray[np.complex128]) -> float:
    """S(ρ) de referencia independiente del SUT."""
    ev = la.eigvalsh(_hermitize(rho))
    ev = np.clip(ev.real, 0.0, None)
    s = float(ev.sum())
    if s < 1e-15:
        return 0.0
    ev = ev / s
    pos = ev[ev > 1e-15]
    if pos.size == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=77_011)


@pytest.fixture
def phase1() -> Phase1_Bekenstein:
    return Phase1_Bekenstein()


@pytest.fixture
def phase2() -> Phase2_EhresmannCartan:
    return Phase2_EhresmannCartan()


@pytest.fixture
def phase3() -> Phase3_GromovWitten:
    return Phase3_GromovWitten()


@pytest.fixture
def agent() -> GromovWittenAuditorAgent:
    return GromovWittenAuditorAgent(default_eta_invariant=0.0)


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


@pytest.fixture
def rho_thermal_4() -> NDArray[np.complex128]:
    return _thermal_density(4, beta=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# §G1 · JERARQUÍA DE EXCEPCIONES Y CONTRATO TAXONÓMICO
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Taxonomía de excepciones simplécticas / cuánticas del agente."""

    def test_root_is_exception(self):
        assert issubclass(GromovWittenAuditorError, Exception)

    def test_leaves_inherit_from_root(self):
        leaves = (
            BekensteinLimitViolationError,
            BusinessInvarianceError,
            CartanStructureError,
            SpectralCompensationError,
            AuditingPipelineError,
        )
        for exc in leaves:
            assert issubclass(exc, GromovWittenAuditorError), f"{exc.__name__}"

    def test_raisable_and_catchable_via_root(self):
        with pytest.raises(GromovWittenAuditorError):
            raise BekensteinLimitViolationError("dim insuficiente")
        with pytest.raises(GromovWittenAuditorError):
            raise BusinessInvarianceError("Δρ excesivo")
        with pytest.raises(GromovWittenAuditorError):
            raise CartanStructureError("Ω patológica")
        with pytest.raises(GromovWittenAuditorError):
            raise SpectralCompensationError("GW divergente")
        with pytest.raises(GromovWittenAuditorError):
            raise AuditingPipelineError("orquestación rota")

    def test_leaf_specific_catch(self):
        with pytest.raises(BusinessInvarianceError):
            raise BusinessInvarianceError("no-demolición violada")
        with pytest.raises(SpectralCompensationError):
            raise SpectralCompensationError("η_eff nan")


# ══════════════════════════════════════════════════════════════════════════════
# §G2 · DTOs INMUTABLES (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════

class TestDTOImmutability:
    """Los artefactos de fase deben ser frozen (inmutables)."""

    def _sample_bekenstein(self) -> BekensteinDimensionData:
        return BekensteinDimensionData(
            von_neumann_entropy=0.5,
            required_audit_dimension=3,
            rho_mic_eigenvalues=np.array([0.7, 0.3], dtype=_FLOAT),
            effective_hilbert_dimension=float(math.exp(0.5)),
            bekenstein_saturated=False,
        )

    def test_bekenstein_dto_frozen(self):
        dto = self._sample_bekenstein()
        with pytest.raises(AttributeError):
            dto.von_neumann_entropy = 1.0  # type: ignore[misc]

    def test_bekenstein_fields_accessible(self):
        dto = self._sample_bekenstein()
        assert dto.required_audit_dimension >= 2
        assert dto.effective_hilbert_dimension > 0.0
        assert dto.rho_mic_eigenvalues.shape[0] == 2

    def test_gw_compensation_frozen(self):
        dto = GromovWittenCompensation(
            gw_invariant_volume=0.01,
            gw_chern_simons_secondary=0.0,
            raw_eta_invariant=0.1,
            effective_eta_invariant=0.09,
            bubble_area_class=0.1414,
            is_ready_for_atiyah_singer=True,
        )
        with pytest.raises(AttributeError):
            dto.is_ready_for_atiyah_singer = False  # type: ignore[misc]
        assert dto.effective_eta_invariant == pytest.approx(0.09)

    def test_maurer_cartan_solution_frozen(self):
        rho = _maximally_mixed(2)
        b = np.zeros((2, 2), dtype=_COMPLEX)
        audit = TelescopicAuditState(
            audited_density_matrix=rho,
            landau_ginzburg_potential=0.0,
            novikov_convergence_iterations=1,
            maurer_cartan_bounding_cochain=b,
            novikov_filtration_degree=0.0,
            is_safe_for_witten_atiyah=True,
        )
        dto = MaurerCartanSolution(
            b_cochain=b,
            audit_state=audit,
            residual_frobenius=0.0,
            novikov_filtration_degree=0.0,
        )
        with pytest.raises(AttributeError):
            dto.residual_frobenius = 1.0  # type: ignore[misc]

    def test_bekenstein_from_pipeline_frozen(self, phase1, rho_2x2_mixed):
        bek = phase1.enforce_bekenstein_bound(rho_2x2_mixed)
        assert isinstance(bek, BekensteinDimensionData)
        with pytest.raises(AttributeError):
            bek.required_audit_dimension = 99  # type: ignore[misc]

    def test_cartan_and_bundle_frozen(self, phase3, rho_2x2_mixed):
        bek = phase3.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase3.certify_cartan_kinematics(rho_2x2_mixed, 0.3, bek)
        assert isinstance(cartan, CartanKinematicsData)
        with pytest.raises(AttributeError):
            cartan.is_business_unchanged = False  # type: ignore[misc]

        bundle = phase3.compensate_aps_eta(cartan, raw_eta_invariant=0.0)
        assert isinstance(bundle, AuditBundle)
        with pytest.raises(AttributeError):
            bundle.bekenstein = bek  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# §G3 · FASE 1 — BEKENSTEIN / VON NEUMANN / DIMENSIONAMIENTO DEL BAÑO
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1VonNeumannEntropy:
    """§1.1 — Entropía de von Neumann S(ρ) = −Tr(ρ ln ρ)."""

    def test_pure_state_entropy_zero(self, phase1, rho_2x2_pure):
        S, ev = phase1._compute_von_neumann_entropy(rho_2x2_pure)
        assert S == pytest.approx(0.0, abs=1e-10)
        assert ev.sum() == pytest.approx(1.0, abs=1e-12)

    def test_maximally_mixed_entropy_log_n(self, phase1, rho_maximally_mixed_3):
        n = 3
        S, ev = phase1._compute_von_neumann_entropy(rho_maximally_mixed_3)
        assert S == pytest.approx(math.log(n), abs=1e-10)
        np.testing.assert_allclose(ev, np.full(n, 1.0 / n), atol=1e-12)

    def test_entropy_matches_reference(self, phase1, rng):
        for n in (2, 3, 5):
            rho = _random_density(n, rng)
            S, _ = phase1._compute_von_neumann_entropy(rho)
            S_ref = _von_neumann_reference(rho)
            assert S == pytest.approx(S_ref, abs=1e-9)

    def test_entropy_bounds(self, phase1, rng):
        for n in (2, 4, 6):
            rho = _random_density(n, rng)
            S, ev = phase1._compute_von_neumann_entropy(rho)
            assert 0.0 <= S <= math.log(n) + _ENTROPY_TOL
            assert ev.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(ev >= -1e-15)

    def test_thermal_entropy_between_pure_and_mixed(self, phase1, rho_thermal_4):
        S, _ = phase1._compute_von_neumann_entropy(rho_thermal_4)
        assert 0.0 < S < math.log(4) + 1e-12

    def test_rejects_invalid_density(self, phase1):
        bad = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises((InvalidDensityMatrixError, GromovWittenAuditorError, Exception)):
            phase1._compute_von_neumann_entropy(bad)

    def test_eigenvalues_renormalized(self, phase1, rho_3x3_mixed):
        _, ev = phase1._compute_von_neumann_entropy(rho_3x3_mixed)
        assert abs(float(ev.sum()) - 1.0) < 1e-12


class TestPhase1BekensteinDimension:
    """§1.2 — Traducción S ↦ dim_audit."""

    def test_pure_state_minimum_dimension(self, phase1):
        """S=0 ⇒ d_eff=1 ⇒ dim = max(2, ⌈1+margin⌉) = 2."""
        required, d_eff, sat = phase1._bekenstein_dimension(0.0)
        assert d_eff == pytest.approx(1.0, abs=1e-14)
        assert required >= _MIN_AUDIT_DIM
        assert required == _MIN_AUDIT_DIM
        assert sat is False

    def test_maximally_mixed_dimension(self, phase1):
        n = 4
        S = math.log(n)
        required, d_eff, sat = phase1._bekenstein_dimension(S)
        assert d_eff == pytest.approx(float(n), abs=1e-10)
        assert required >= n
        assert sat is False

    def test_negative_entropy_raises(self, phase1):
        with pytest.raises(BekensteinLimitViolationError):
            phase1._bekenstein_dimension(-0.1)

    def test_non_finite_entropy_raises(self, phase1):
        with pytest.raises(BekensteinLimitViolationError):
            phase1._bekenstein_dimension(float("nan"))
        with pytest.raises(BekensteinLimitViolationError):
            phase1._bekenstein_dimension(float("inf"))

    def test_saturation_at_max_audit_dim(self, phase1):
        # S tan grande que e^S >> _MAX_AUDIT_DIM
        huge_S = math.log(_MAX_AUDIT_DIM + 1000.0)
        required, d_eff, sat = phase1._bekenstein_dimension(huge_S)
        assert sat is True
        assert required == _MAX_AUDIT_DIM
        assert d_eff > _MAX_AUDIT_DIM

    def test_monotonic_in_entropy(self, phase1):
        prev = 0
        for S in (0.0, 0.5, 1.0, 1.5, 2.0):
            req, _, _ = phase1._bekenstein_dimension(S)
            assert req >= prev
            prev = req


class TestPhase1EnforceBekensteinBound:
    """§1.3 — Morfismo terminal Φ₁ᴳᵂ."""

    def test_returns_bekenstein_dto(self, phase1, rho_2x2_mixed):
        data = phase1.enforce_bekenstein_bound(rho_2x2_mixed)
        assert isinstance(data, BekensteinDimensionData)

    def test_dimension_at_least_min(self, phase1, rho_2x2_pure):
        data = phase1.enforce_bekenstein_bound(rho_2x2_pure)
        assert data.required_audit_dimension >= _MIN_AUDIT_DIM

    def test_dimension_covers_effective(self, phase1, rho_3x3_mixed):
        data = phase1.enforce_bekenstein_bound(rho_3x3_mixed)
        if not data.bekenstein_saturated:
            assert data.required_audit_dimension >= math.floor(
                data.effective_hilbert_dimension
            )

    def test_entropy_consistency(self, phase1, rho_4x4_mixed):
        data = phase1.enforce_bekenstein_bound(rho_4x4_mixed)
        S_ref = _von_neumann_reference(rho_4x4_mixed)
        assert data.von_neumann_entropy == pytest.approx(S_ref, abs=1e-9)
        assert data.effective_hilbert_dimension == pytest.approx(
            math.exp(data.von_neumann_entropy), abs=1e-10
        )

    def test_eigenvalues_sum_to_one(self, phase1, rho_2x2_mixed):
        data = phase1.enforce_bekenstein_bound(rho_2x2_mixed)
        assert data.rho_mic_eigenvalues.sum() == pytest.approx(1.0, abs=1e-12)

    def test_pure_vs_mixed_dimension_ordering(self, phase1):
        pure = _pure_state_density(4, 0)
        mixed = _maximally_mixed(4)
        d_pure = phase1.enforce_bekenstein_bound(pure)
        d_mixed = phase1.enforce_bekenstein_bound(mixed)
        assert d_pure.von_neumann_entropy <= d_mixed.von_neumann_entropy
        assert d_pure.required_audit_dimension <= d_mixed.required_audit_dimension

    def test_rejects_non_hermitian(self, phase1):
        bad = np.array([[0.5, 1.0 + 2.0j], [0.0, 0.5]], dtype=_COMPLEX)
        with pytest.raises((InvalidDensityMatrixError, GromovWittenAuditorError, Exception)):
            phase1.enforce_bekenstein_bound(bad)

    def test_phase1_inherits_stinespring(self, phase1):
        assert hasattr(phase1, "compute_isometric_immersion")


# ══════════════════════════════════════════════════════════════════════════════
# §G4 · FASE 2 — CINEMÁTICA DE CARTAN / EHRESMANN / NO-DEMOLICIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2PartialTraceAudit:
    """§2.1 — Traza parcial sobre el factor de auditoría."""

    def test_partial_trace_shape(self, phase2):
        dim_mic, dim_audit = 2, 3
        # Estado producto |0⟩⟨0| ⊗ ρ_a
        rho_mic = _pure_state_density(dim_mic, 0)
        rho_a = _maximally_mixed(dim_audit)
        rho_c = np.kron(rho_mic, rho_a)
        out = phase2._partial_trace_audit(rho_c, dim_mic, dim_audit)
        assert out.shape == (dim_mic, dim_mic)

    def test_partial_trace_product_state(self, phase2):
        """Tr_audit(ρ_M ⊗ ρ_a) = ρ_M · Tr(ρ_a) = ρ_M."""
        dim_mic, dim_audit = 3, 2
        rho_mic = _maximally_mixed(dim_mic)
        rho_a = _maximally_mixed(dim_audit)
        rho_c = np.kron(rho_mic, rho_a)
        out = phase2._partial_trace_audit(rho_c, dim_mic, dim_audit)
        np.testing.assert_allclose(out, rho_mic, atol=1e-12)

    def test_partial_trace_preserves_hermiticity(self, phase2, rng):
        dim_mic, dim_audit = 2, 2
        rho_c = _random_density(dim_mic * dim_audit, rng)
        out = phase2._partial_trace_audit(rho_c, dim_mic, dim_audit)
        assert _is_hermitian(out)

    def test_dimension_mismatch_raises(self, phase2):
        rho_c = np.eye(6, dtype=_COMPLEX) / 6.0
        with pytest.raises(CartanStructureError, match="incompatible"):
            phase2._partial_trace_audit(rho_c, dim_mic=2, dim_audit=2)  # 4 ≠ 6


class TestPhase2BusinessInvariance:
    """§2.2 — Identidad de no-demolición."""

    def _prepare_dilation(
        self, phase2: Phase2_EhresmannCartan, rho: NDArray, dim_audit: int
    ) -> StinespringDilationData:
        return phase2.compute_isometric_immersion(rho, dim_audit)

    def test_identity_zoom_preserves_business(self, phase2, rho_2x2_mixed):
        dim_a = 3
        dil = self._prepare_dilation(phase2, rho_2x2_mixed, dim_a)
        T = np.eye(dim_a, dtype=_COMPLEX)
        diff, unchanged = phase2.verify_business_invariance(rho_2x2_mixed, dil, T)
        assert unchanged is True
        assert diff < _BUS_INV_TOL

    def test_spectral_zoom_preserves_business(self, phase2, rho_2x2_mixed):
        dim_a = 4
        dil = self._prepare_dilation(phase2, rho_2x2_mixed, dim_a)
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=1.0)
        diff, unchanged = phase2.verify_business_invariance(
            rho_2x2_mixed, dil, fib.T_lambda_vertical
        )
        assert unchanged is True
        assert diff < _BUS_INV_TOL

    def test_invariance_across_lambdas(self, phase2, rho_3x3_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_3x3_mixed)
        dim_a = bek.required_audit_dimension
        dil = self._prepare_dilation(phase2, rho_3x3_mixed, dim_a)
        for lam in (0.0, 0.5, 1.5, 3.0):
            fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=lam)
            diff, unchanged = phase2.verify_business_invariance(
                rho_3x3_mixed, dil, fib.T_lambda_vertical
            )
            assert unchanged, f"λ={lam}: diff={diff}"

    def test_shape_mismatch_v_raises(self, phase2, rho_2x2_mixed):
        dil = self._prepare_dilation(phase2, rho_2x2_mixed, dim_audit=3)
        T_wrong = np.eye(2, dtype=_COMPLEX)  # dim_audit debería ser 3
        with pytest.raises(CartanStructureError):
            phase2.verify_business_invariance(rho_2x2_mixed, dil, T_wrong)


class TestPhase2EhresmannGaugeDiagnosis:
    """§2.3 — Diagnóstico ℓ_H y ‖Ω‖."""

    def test_canonical_fibration_passes(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        dil = phase2.compute_isometric_immersion(
            rho_2x2_mixed, bek.required_audit_dimension
        )
        fib = phase2.apply_telescopic_deformation(dil, lambda_zoom=0.8)
        leak, curv = phase2._diagnose_ehresmann_gauge(fib)
        assert leak < _BUS_INV_TOL
        assert curv < _CARTAN_CURV_TOL

    def test_artificial_leakage_raises(self, phase2):
        dim = 2
        I = np.eye(dim, dtype=_COMPLEX)
        bad_fib = VerticalFibrationData(
            T_lambda_vertical=I,
            ehresmann_connection_form=I,
            vertical_projector=I,
            horizontal_leakage_norm=1e-3,  # >> tolerancia
            connection_curvature_norm=0.0,
            spectral_zoom_eigenvalues=np.ones(dim, dtype=_FLOAT),
        )
        with pytest.raises(CartanStructureError, match="horizontal|Fuga|ℓ_H"):
            phase2._diagnose_ehresmann_gauge(bad_fib)

    def test_artificial_curvature_raises(self, phase2):
        dim = 2
        I = np.eye(dim, dtype=_COMPLEX)
        bad_fib = VerticalFibrationData(
            T_lambda_vertical=I,
            ehresmann_connection_form=I,
            vertical_projector=I,
            horizontal_leakage_norm=0.0,
            connection_curvature_norm=1e-3,  # >> tolerancia
            spectral_zoom_eigenvalues=np.ones(dim, dtype=_FLOAT),
        )
        with pytest.raises(CartanStructureError, match="[Cc]urvatura|Ω"):
            phase2._diagnose_ehresmann_gauge(bad_fib)


class TestPhase2CertifyCartanKinematics:
    """§2.4 — Morfismo terminal Φ₂ᴳᵂ."""

    def test_returns_cartan_dto(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_mixed, 0.5, bek)
        assert isinstance(cartan, CartanKinematicsData)

    def test_business_unchanged_flag(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_mixed, 0.5, bek)
        assert cartan.is_business_unchanged is True
        assert cartan.business_state_difference < _BUS_INV_TOL

    def test_nested_dtos_present(self, phase2, rho_3x3_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_3x3_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_3x3_mixed, 0.4, bek)
        assert isinstance(cartan.dilation_data, StinespringDilationData)
        assert isinstance(cartan.fibration_data, VerticalFibrationData)
        assert isinstance(cartan.bekenstein_data, BekensteinDimensionData)
        assert cartan.bekenstein_data.required_audit_dimension == bek.required_audit_dimension

    def test_audit_dimension_matches_bekenstein(self, phase2, rho_2x2_pure):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_pure)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_pure, 0.2, bek)
        rho_a = cartan.dilation_data.rho_audit_subspace
        assert rho_a.shape == (bek.required_audit_dimension, bek.required_audit_dimension)

    def test_negative_lambda_raises(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        with pytest.raises(ValueError, match="lambda_magnification"):
            phase2.certify_cartan_kinematics(rho_2x2_mixed, -0.1, bek)

    def test_lambda_zero_cartan(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_mixed, 0.0, bek)
        assert cartan.is_business_unchanged
        # T_λ = I ⇒ scales ≈ 1
        scales = cartan.fibration_data.spectral_zoom_eigenvalues
        np.testing.assert_allclose(scales, np.ones_like(scales), atol=1e-10)

    def test_leakage_and_curvature_recorded(self, phase2, rho_2x2_mixed):
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_mixed, 1.0, bek)
        assert cartan.horizontal_leakage_norm < _BUS_INV_TOL
        assert cartan.cartan_curvature_norm < _CARTAN_CURV_TOL

    def test_phase2_inherits_phase1_and_engine(self, phase2):
        assert isinstance(phase2, Phase1_Bekenstein)
        assert hasattr(phase2, "apply_telescopic_deformation")
        assert hasattr(phase2, "enforce_bekenstein_bound")

    def test_chain_phi1_to_phi2(self, phase2, rho_2x2_mixed):
        """Anidación: Φ₂ᴳᵂ consume directamente el DTO de Φ₁ᴳᵂ."""
        bek = phase2.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase2.certify_cartan_kinematics(rho_2x2_mixed, 0.35, bek)
        assert cartan.bekenstein_data.von_neumann_entropy == bek.von_neumann_entropy


# ══════════════════════════════════════════════════════════════════════════════
# §G5 · FASE 3 — GROMOV–WITTEN / MAURER–CARTAN / COMPENSACIÓN APS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase3BoundingCochainExtraction:
    """§3.1 — Extracción de b desde TelescopicAuditState."""

    def test_extract_from_v3_dto(self, phase3):
        b_expected = np.array([[0.0, 0.1j], [-0.1j, 0.0]], dtype=_COMPLEX)
        rho = _maximally_mixed(2)
        audit = TelescopicAuditState(
            audited_density_matrix=rho,
            landau_ginzburg_potential=1e-12,
            novikov_convergence_iterations=3,
            maurer_cartan_bounding_cochain=b_expected,
            novikov_filtration_degree=0.1,
            is_safe_for_witten_atiyah=True,
        )
        b = phase3._extract_bounding_cochain(audit)
        np.testing.assert_allclose(b, b_expected)

    def test_extract_fallback(self, phase3):
        rho = _maximally_mixed(2)
        b_fb = np.zeros((2, 2), dtype=_COMPLEX)

        # DTO sin campo b: construir uno mínimo y forzar fallback
        class _StubState:
            audited_density_matrix = rho
            landau_ginzburg_potential = 0.0
            novikov_convergence_iterations = 1
            is_safe_for_witten_atiyah = True

        b = phase3._extract_bounding_cochain(_StubState(), fallback_b=b_fb)  # type: ignore[arg-type]
        np.testing.assert_allclose(b, b_fb)

    def test_extract_missing_raises(self, phase3):
        class _StubState:
            audited_density_matrix = _maximally_mixed(2)
            landau_ginzburg_potential = 0.0
            novikov_convergence_iterations = 1
            is_safe_for_witten_atiyah = True

        with pytest.raises(SpectralCompensationError, match="[Cc]o-cadena|ausente"):
            phase3._extract_bounding_cochain(_StubState())  # type: ignore[arg-type]


class TestPhase3GromovWittenInvariants:
    """§3.2 — GW₀, CS₃, area_class."""

    def test_zero_cochain_vanishing_invariants(self, phase3):
        b = np.zeros((3, 3), dtype=_COMPLEX)
        gw0, cs3, area = phase3._gromov_witten_invariants(b)
        assert gw0 == pytest.approx(0.0, abs=1e-15)
        assert area == pytest.approx(0.0, abs=1e-15)
        assert abs(cs3) < 1e-15

    def test_gw_is_half_hs_squared(self, phase3):
        b = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=_COMPLEX)
        # ‖b‖_HS² = Tr(b†b) = 1
        gw0, _, area = phase3._gromov_witten_invariants(b)
        assert gw0 == pytest.approx(0.5, abs=1e-12)
        assert area == pytest.approx(1.0, abs=1e-12)

    def test_antihermitian_cochain_cs3(self, phase3, rng):
        X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        b = 0.1 * (X - X.conj().T)  # antihermitiana
        gw0, cs3, area = phase3._gromov_witten_invariants(b)
        assert gw0 >= 0.0
        assert math.isfinite(cs3)
        assert area >= 0.0
        # Consistencia: area² ≈ 2·GW₀
        assert area ** 2 == pytest.approx(2.0 * gw0, abs=1e-10)

    def test_non_square_raises(self, phase3):
        b = np.zeros((2, 3), dtype=_COMPLEX)
        with pytest.raises(SpectralCompensationError, match="cuadrada"):
            phase3._gromov_witten_invariants(b)

    def test_gw_nonnegative(self, phase3, rng):
        for _ in range(8):
            X = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            b = 0.05 * X
            gw0, _, _ = phase3._gromov_witten_invariants(b)
            assert gw0 >= -1e-15


class TestPhase3CompensateEta:
    """§3.3 — η_eff = η_raw − GW₀."""

    def test_basic_compensation(self, phase3):
        eta_eff = phase3._compensate_eta(raw_eta=1.0, gw0=0.25)
        assert eta_eff == pytest.approx(0.75, abs=1e-14)

    def test_zero_gw_preserves_eta(self, phase3):
        assert phase3._compensate_eta(0.3, 0.0) == pytest.approx(0.3)

    def test_nan_raw_raises(self, phase3):
        with pytest.raises(SpectralCompensationError, match="η_raw"):
            phase3._compensate_eta(float("nan"), 0.1)

    def test_inf_raw_raises(self, phase3):
        with pytest.raises(SpectralCompensationError):
            phase3._compensate_eta(float("inf"), 0.0)

    def test_huge_raw_raises(self, phase3):
        with pytest.raises(SpectralCompensationError):
            phase3._compensate_eta(1e15, 0.0)


class TestPhase3ResolveMaurerCartanWithGW:
    """§3.4 — MC + GW + APS acoplados."""

    def _cartan(
        self, phase3: Phase3_GromovWitten, rho: NDArray, lam: float = 0.4
    ) -> CartanKinematicsData:
        bek = phase3.enforce_bekenstein_bound(rho)
        return phase3.certify_cartan_kinematics(rho, lam, bek)

    def test_returns_mc_and_gw_tuple(self, phase3, rho_2x2_mixed):
        cartan = self._cartan(phase3, rho_2x2_mixed)
        mc, gw = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=0.1,
        )
        assert isinstance(mc, MaurerCartanSolution)
        assert isinstance(gw, GromovWittenCompensation)

    def test_mc_residual_below_tolerance(self, phase3, rho_2x2_mixed):
        cartan = self._cartan(phase3, rho_2x2_mixed, lam=0.3)
        mc, _ = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=0.0,
        )
        assert mc.residual_frobenius < _MC_TOL
        assert mc.audit_state.landau_ginzburg_potential < _MC_TOL

    def test_audit_state_is_valid_density(self, phase3, rho_3x3_mixed):
        cartan = self._cartan(phase3, rho_3x3_mixed, lam=0.25)
        mc, _ = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=0.0,
        )
        assert _is_density(mc.audit_state.audited_density_matrix)

    def test_gw_eta_relation(self, phase3, rho_2x2_mixed):
        cartan = self._cartan(phase3, rho_2x2_mixed, lam=0.5)
        eta_raw = 0.42
        _, gw = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=eta_raw,
        )
        assert gw.raw_eta_invariant == pytest.approx(eta_raw)
        assert gw.effective_eta_invariant == pytest.approx(
            eta_raw - gw.gw_invariant_volume, abs=1e-12
        )
        assert gw.gw_invariant_volume >= 0.0
        assert math.isfinite(gw.gw_chern_simons_secondary)
        assert gw.bubble_area_class >= 0.0

    def test_lambda_zero_small_gw(self, phase3, rho_2x2_pure):
        """λ=0 ⇒ m₀=0 ⇒ b≈0 ⇒ GW₀≈0."""
        cartan = self._cartan(phase3, rho_2x2_pure, lam=0.0)
        mc, gw = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=1.0,
        )
        assert gw.gw_invariant_volume < 1e-12
        assert gw.effective_eta_invariant == pytest.approx(1.0, abs=1e-10)
        assert _fro(mc.b_cochain) < 1e-8

    def test_cochain_shape_matches_audit(self, phase3, rho_2x2_mixed):
        cartan = self._cartan(phase3, rho_2x2_mixed)
        mc, _ = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=0.0,
        )
        dim_a = cartan.dilation_data.rho_audit_subspace.shape[0]
        assert mc.b_cochain.shape == (dim_a, dim_a)

    def test_ready_flag_true_on_success(self, phase3, rho_2x2_mixed):
        cartan = self._cartan(phase3, rho_2x2_mixed, lam=0.3)
        mc, gw = phase3.resolve_maurer_cartan_with_gw(
            cartan.fibration_data,
            cartan.dilation_data.rho_audit_subspace,
            raw_eta_invariant=0.0,
        )
        assert mc.audit_state.is_safe_for_witten_atiyah is True
        assert gw.is_ready_for_atiyah_singer is True


class TestPhase3CompensateAPSEta:
    """§3.5 — Morfismo terminal Φ₃ᴳᵂ → AuditBundle."""

    def test_returns_audit_bundle(self, phase3, rho_2x2_mixed):
        bek = phase3.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase3.certify_cartan_kinematics(rho_2x2_mixed, 0.4, bek)
        bundle = phase3.compensate_aps_eta(cartan, raw_eta_invariant=0.05)
        assert isinstance(bundle, AuditBundle)

    def test_bundle_internal_consistency(self, phase3, rho_2x2_mixed):
        bek = phase3.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase3.certify_cartan_kinematics(rho_2x2_mixed, 0.35, bek)
        bundle = phase3.compensate_aps_eta(cartan, raw_eta_invariant=0.2)

        assert bundle.cartan_kinematics is cartan or (
            bundle.cartan_kinematics.business_state_difference
            == cartan.business_state_difference
        )
        assert bundle.bekenstein.required_audit_dimension == bek.required_audit_dimension
        assert _is_density(bundle.audit_state.audited_density_matrix)
        assert bundle.gw_compensation.raw_eta_invariant == pytest.approx(0.2)
        assert (
            bundle.maurer_cartan_solution.audit_state.audited_density_matrix.shape
            == bundle.audit_state.audited_density_matrix.shape
        )

    def test_rejects_if_business_changed(self, phase3, rho_2x2_mixed):
        bek = phase3.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase3.certify_cartan_kinematics(rho_2x2_mixed, 0.3, bek)
        # Fabricar un CartanKinematicsData con flag roto
        broken = CartanKinematicsData(
            business_state_difference=1.0,
            is_business_unchanged=False,
            horizontal_leakage_norm=cartan.horizontal_leakage_norm,
            cartan_curvature_norm=cartan.cartan_curvature_norm,
            dilation_data=cartan.dilation_data,
            fibration_data=cartan.fibration_data,
            bekenstein_data=cartan.bekenstein_data,
        )
        with pytest.raises(BusinessInvarianceError):
            phase3.compensate_aps_eta(broken, raw_eta_invariant=0.0)

    def test_phase3_inherits_full_chain(self, phase3):
        assert isinstance(phase3, Phase2_EhresmannCartan)
        assert isinstance(phase3, Phase1_Bekenstein)
        assert hasattr(phase3, "resolve_maurer_cartan_novikov")

    def test_full_nested_chain_on_phase3(self, phase3, rho_2x2_mixed):
        """Anidación estricta Φ₁ᴳᵂ→Φ₂ᴳᵂ→Φ₃ᴳᵂ sobre la misma instancia."""
        bek = phase3.enforce_bekenstein_bound(rho_2x2_mixed)
        cartan = phase3.certify_cartan_kinematics(rho_2x2_mixed, 0.4, bek)
        bundle = phase3.compensate_aps_eta(cartan, 0.0)
        assert bundle.gw_compensation.is_ready_for_atiyah_singer
        assert _is_density(bundle.audit_state.audited_density_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# §G6 · ORQUESTADOR GromovWittenAuditorAgent (COMPOSICIÓN 𝒜)
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentConstruction:
    """Inicialización y validación de parámetros del orquestador."""

    def test_default_eta(self):
        ag = GromovWittenAuditorAgent()
        assert ag.default_eta_invariant == pytest.approx(0.0)

    def test_custom_default_eta(self):
        ag = GromovWittenAuditorAgent(default_eta_invariant=0.7)
        assert ag.default_eta_invariant == pytest.approx(0.7)

    def test_rejects_non_finite_default_eta(self):
        with pytest.raises(ValueError, match="default_eta_invariant"):
            GromovWittenAuditorAgent(default_eta_invariant=float("nan"))
        with pytest.raises(ValueError, match="default_eta_invariant"):
            GromovWittenAuditorAgent(default_eta_invariant=float("inf"))

    def test_agent_is_phase3_and_morphism(self, agent):
        assert isinstance(agent, Phase3_GromovWitten)


class TestAgentExecuteAuditingProcess:
    """Composición funtorial 𝒜 = Φ₃ᴳᵂ ∘ Φ₂ᴳᵂ ∘ Φ₁ᴳᵂ (API tupla)."""

    def test_returns_tuple(self, agent, rho_2x2_mixed):
        result = agent.execute_auditing_process(rho_2x2_mixed, 0.4, 0.1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        state, gw = result
        assert isinstance(state, TelescopicAuditState)
        assert isinstance(gw, GromovWittenCompensation)

    def test_output_density_valid(self, agent, rho_3x3_mixed):
        state, gw = agent.execute_auditing_process(rho_3x3_mixed, 0.3, 0.0)
        assert _is_density(state.audited_density_matrix)
        assert gw.is_ready_for_atiyah_singer is True

    def test_uses_default_eta_when_none(self):
        ag = GromovWittenAuditorAgent(default_eta_invariant=0.55)
        rho = _pure_state_density(2)
        _, gw = ag.execute_auditing_process(rho, 0.0, current_eta_invariant=None)
        # λ=0 ⇒ GW≈0 ⇒ η_eff ≈ default
        assert gw.raw_eta_invariant == pytest.approx(0.55)
        assert gw.effective_eta_invariant == pytest.approx(0.55, abs=1e-8)

    def test_explicit_eta_overrides_default(self, agent, rho_2x2_pure):
        _, gw = agent.execute_auditing_process(
            rho_2x2_pure, 0.0, current_eta_invariant=1.25
        )
        assert gw.raw_eta_invariant == pytest.approx(1.25)

    def test_negative_lambda_raises(self, agent, rho_2x2_mixed):
        with pytest.raises(ValueError, match="lambda_magnification"):
            agent.execute_auditing_process(rho_2x2_mixed, -0.01, 0.0)

    def test_invalid_rho_raises(self, agent):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=_COMPLEX)
        with pytest.raises((InvalidDensityMatrixError, GromovWittenAuditorError)):
            agent.execute_auditing_process(bad, 0.1, 0.0)

    def test_non_finite_eta_raises(self, agent, rho_2x2_mixed):
        with pytest.raises(SpectralCompensationError):
            agent.execute_auditing_process(
                rho_2x2_mixed, 0.1, current_eta_invariant=float("nan")
            )

    def test_lg_potential_below_tolerance(self, agent, rho_2x2_mixed):
        state, _ = agent.execute_auditing_process(rho_2x2_mixed, 0.35, 0.0)
        assert state.landau_ginzburg_potential < _MC_TOL

    def test_deterministic_for_fixed_input(self, rho_2x2_mixed):
        ag = GromovWittenAuditorAgent(default_eta_invariant=0.0)
        s1, g1 = ag.execute_auditing_process(rho_2x2_mixed, 0.4, 0.1)
        s2, g2 = ag.execute_auditing_process(rho_2x2_mixed, 0.4, 0.1)
        np.testing.assert_allclose(
            s1.audited_density_matrix, s2.audited_density_matrix, atol=1e-12
        )
        assert g1.gw_invariant_volume == pytest.approx(g2.gw_invariant_volume)
        assert g1.effective_eta_invariant == pytest.approx(g2.effective_eta_invariant)


class TestAgentExecuteAuditingBundle:
    """API rica: AuditBundle completo."""

    def test_returns_bundle(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.4, 0.1)
        assert isinstance(bundle, AuditBundle)

    def test_bundle_vs_process_consistency(self, agent, rho_2x2_mixed):
        state, gw = agent.execute_auditing_process(rho_2x2_mixed, 0.45, 0.2)
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.45, 0.2)
        np.testing.assert_allclose(
            state.audited_density_matrix,
            bundle.audit_state.audited_density_matrix,
            atol=1e-12,
        )
        assert gw.gw_invariant_volume == pytest.approx(
            bundle.gw_compensation.gw_invariant_volume
        )
        assert gw.effective_eta_invariant == pytest.approx(
            bundle.gw_compensation.effective_eta_invariant
        )

    def test_bundle_contains_all_phases(self, agent, rho_3x3_mixed):
        bundle = agent.execute_auditing_bundle(rho_3x3_mixed, 0.3, 0.0)
        assert isinstance(bundle.bekenstein, BekensteinDimensionData)
        assert isinstance(bundle.cartan_kinematics, CartanKinematicsData)
        assert isinstance(bundle.maurer_cartan_solution, MaurerCartanSolution)
        assert isinstance(bundle.gw_compensation, GromovWittenCompensation)
        assert isinstance(bundle.audit_state, TelescopicAuditState)

    def test_bundle_bekenstein_dim_matches_audit_shape(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.25, 0.0)
        dim = bundle.bekenstein.required_audit_dimension
        assert bundle.audit_state.audited_density_matrix.shape == (dim, dim)


# ══════════════════════════════════════════════════════════════════════════════
# §G7 · INVARIANTES DE EXTREMO A EXTREMO Y REGRESIÓN NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndInvariants:
    """Invariantes globales del endofuntor 𝒜."""

    def test_composition_equals_stepwise(self, rng):
        """𝒜(ρ,λ,η) ≡ Φ₃ᴳᵂ(Φ₂ᴳᵂ(Φ₁ᴳᵂ(ρ),λ),η) componente a componente."""
        ag = GromovWittenAuditorAgent(default_eta_invariant=0.0)
        rho = _random_density(2, rng)
        lam, eta = 0.5, 0.15

        # Monolítico
        bundle_z = ag.execute_auditing_bundle(rho, lam, eta)

        # Paso a paso
        bek = ag.enforce_bekenstein_bound(rho)
        cartan = ag.certify_cartan_kinematics(rho, lam, bek)
        bundle_s = ag.compensate_aps_eta(cartan, eta)

        np.testing.assert_allclose(
            bundle_z.audit_state.audited_density_matrix,
            bundle_s.audit_state.audited_density_matrix,
            atol=1e-12,
        )
        assert bundle_z.gw_compensation.gw_invariant_volume == pytest.approx(
            bundle_s.gw_compensation.gw_invariant_volume, abs=1e-12
        )
        assert bundle_z.gw_compensation.effective_eta_invariant == pytest.approx(
            bundle_s.gw_compensation.effective_eta_invariant, abs=1e-12
        )
        assert (
            bundle_z.bekenstein.required_audit_dimension
            == bundle_s.bekenstein.required_audit_dimension
        )

    def test_business_invariance_e2e(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 1.0, 0.0)
        assert bundle.cartan_kinematics.is_business_unchanged
        assert bundle.cartan_kinematics.business_state_difference < _BUS_INV_TOL

    def test_stinespring_isometry_through_agent(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.5, 0.0)
        V = bundle.cartan_kinematics.dilation_data.V_isometry
        n = rho_2x2_mixed.shape[0]
        G = V.conj().T @ V
        np.testing.assert_allclose(G, np.eye(n, dtype=_COMPLEX), atol=_ISO_TOL)

    def test_eta_eff_equals_raw_minus_gw(self, agent, rng):
        rho = _random_density(3, rng)
        eta_raw = 0.77
        _, gw = agent.execute_auditing_process(rho, 0.4, eta_raw)
        assert gw.effective_eta_invariant == pytest.approx(
            gw.raw_eta_invariant - gw.gw_invariant_volume, abs=1e-12
        )

    def test_mc_residual_matches_lg(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.4, 0.0)
        assert bundle.maurer_cartan_solution.residual_frobenius == pytest.approx(
            bundle.audit_state.landau_ginzburg_potential, abs=1e-10
        )
        assert bundle.maurer_cartan_solution.residual_frobenius < _MC_TOL

    def test_gw_area_consistency(self, agent, rho_2x2_mixed):
        """area_class² ≈ 2 · GW₀ ≈ ‖b‖_HS²."""
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.6, 0.0)
        gw0 = bundle.gw_compensation.gw_invariant_volume
        area = bundle.gw_compensation.bubble_area_class
        b = bundle.maurer_cartan_solution.b_cochain
        hs2 = float(np.real(np.trace(b.conj().T @ b)))
        assert gw0 == pytest.approx(0.5 * hs2, abs=1e-10)
        if hs2 > _HS_FLOOR:
            assert area == pytest.approx(math.sqrt(hs2), abs=1e-10)

    def test_output_purity_in_legal_range(self, agent, rng):
        rho = _random_density(3, rng)
        state, _ = agent.execute_auditing_process(rho, 0.3, 0.0)
        rho_s = state.audited_density_matrix
        purity = float(np.real(np.trace(rho_s @ rho_s)))
        n = rho_s.shape[0]
        assert 1.0 / n - 1e-8 <= purity <= 1.0 + 1e-8

    def test_bekenstein_dim_monotonic_with_mixedness(self, agent):
        pure = _pure_state_density(4, 0)
        mixed = _maximally_mixed(4)
        b_pure = agent.execute_auditing_bundle(pure, 0.2, 0.0)
        b_mixed = agent.execute_auditing_bundle(mixed, 0.2, 0.0)
        assert (
            b_pure.bekenstein.required_audit_dimension
            <= b_mixed.bekenstein.required_audit_dimension
        )

    def test_safe_flags_aligned(self, agent, rho_2x2_mixed):
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.3, 0.0)
        assert bundle.audit_state.is_safe_for_witten_atiyah is True
        assert bundle.gw_compensation.is_ready_for_atiyah_singer is True


# ══════════════════════════════════════════════════════════════════════════════
# §G8 · CASOS LÍMITE, PATOLOGÍAS, ROBUSTEZ Y SUPERFICIE PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndRobustness:
    """Bordes dimensionales, estados extremos y estrés numérico."""

    def test_pure_state_pipeline(self, agent):
        rho = _pure_state_density(3, idx=1)
        state, gw = agent.execute_auditing_process(rho, 0.3, 0.0)
        assert _is_density(state.audited_density_matrix)
        assert gw.is_ready_for_atiyah_singer

    def test_maximally_mixed_pipeline(self, agent):
        rho = _maximally_mixed(4)
        bundle = agent.execute_auditing_bundle(rho, 0.25, 0.1)
        assert _is_density(bundle.audit_state.audited_density_matrix)
        assert bundle.bekenstein.von_neumann_entropy == pytest.approx(
            math.log(4), abs=1e-8
        )

    def test_thermal_state_pipeline(self, agent, rho_thermal_4):
        state, gw = agent.execute_auditing_process(rho_thermal_4, 0.4, 0.05)
        assert _is_density(state.audited_density_matrix)
        assert math.isfinite(gw.effective_eta_invariant)

    def test_near_rank_deficient(self, agent):
        eps = 1e-10
        rho = np.diag([1.0 - eps, eps, 0.0]).astype(_COMPLEX)
        rho = _hermitize(rho)
        rho /= np.trace(rho).real
        state, gw = agent.execute_auditing_process(rho, 0.2, 0.0)
        assert _is_density(state.audited_density_matrix)

    def test_large_business_dimension(self, rng):
        ag = GromovWittenAuditorAgent()
        rho = _random_density(8, rng)
        state, gw = ag.execute_auditing_process(rho, 0.15, 0.0)
        assert _is_density(state.audited_density_matrix)
        assert gw.gw_invariant_volume < _GW_CEILING

    def test_high_lambda_converges_or_bubbling(self, agent, rho_2x2_mixed):
        """λ elevado: convergencia o SphereBubbling controlado; sin excepciones genéricas."""
        try:
            state, gw = agent.execute_auditing_process(rho_2x2_mixed, 20.0, 0.0)
            assert isinstance(state, TelescopicAuditState)
            assert isinstance(gw, GromovWittenCompensation)
            assert _is_density(state.audited_density_matrix)
        except SphereBubblingAnomalyError:
            pass
        except GromovWittenAuditorError:
            pass

    def test_does_not_mutate_input(self, agent, rng):
        rho = _random_density(3, rng)
        rho_copy = rho.copy()
        _ = agent.execute_auditing_process(rho, 0.5, 0.0)
        np.testing.assert_array_equal(rho, rho_copy)

    def test_complex_offdiagonal_density(self, agent, rng):
        n = 3
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        rho = A @ A.conj().T
        rho = _hermitize(rho)
        rho /= np.trace(rho).real
        state, gw = agent.execute_auditing_process(rho.astype(_COMPLEX), 0.3, 0.1)
        assert _is_density(state.audited_density_matrix)
        assert math.isfinite(gw.effective_eta_invariant)

    def test_multiple_agents_independent(self, rng):
        rho = _random_density(2, rng)
        a1 = GromovWittenAuditorAgent(default_eta_invariant=0.0)
        a2 = GromovWittenAuditorAgent(default_eta_invariant=1.0)
        _, g1 = a1.execute_auditing_process(rho, 0.0, None)
        _, g2 = a2.execute_auditing_process(rho, 0.0, None)
        assert g1.raw_eta_invariant == pytest.approx(0.0)
        assert g2.raw_eta_invariant == pytest.approx(1.0)

    def test_gauge_traceless_cochain(self, agent, rho_2x2_mixed):
        """Co-cadena b con traza ~0 (gauge fix del motor)."""
        bundle = agent.execute_auditing_bundle(rho_2x2_mixed, 0.5, 0.0)
        tr_b = complex(np.trace(bundle.maurer_cartan_solution.b_cochain))
        assert abs(tr_b) < 1e-8

    def test_zero_lambda_eta_preservation(self, agent, rho_2x2_mixed):
        eta = 0.33
        _, gw = agent.execute_auditing_process(rho_2x2_mixed, 0.0, eta)
        assert gw.gw_invariant_volume < 1e-12
        assert gw.effective_eta_invariant == pytest.approx(eta, abs=1e-10)


class TestParametrizedSpectrum:
    """Batería parametrizada sobre dimensiones, magnificaciones y η."""

    @pytest.mark.parametrize("dim_mic", [2, 3, 4])
    @pytest.mark.parametrize("lam", [0.0, 0.25, 1.0])
    @pytest.mark.parametrize("eta", [0.0, 0.5])
    def test_grid_dimensions_lambda_eta(
        self,
        rng: np.random.Generator,
        dim_mic: int,
        lam: float,
        eta: float,
    ):
        ag = GromovWittenAuditorAgent()
        rho = _random_density(dim_mic, rng)
        state, gw = ag.execute_auditing_process(rho, lam, eta)
        assert _is_density(state.audited_density_matrix)
        assert gw.is_ready_for_atiyah_singer is True
        assert state.landau_ginzburg_potential < _MC_TOL
        assert gw.effective_eta_invariant == pytest.approx(
            eta - gw.gw_invariant_volume, abs=1e-10
        )
        assert gw.gw_invariant_volume >= -1e-15

    @pytest.mark.parametrize(
        "rho_factory",
        [
            lambda: _pure_state_density(2, 0),
            lambda: _pure_state_density(3, 2),
            lambda: _maximally_mixed(2),
            lambda: _maximally_mixed(4),
            lambda: _thermal_density(3, beta=0.8),
        ],
    )
    def test_canonical_states(self, rho_factory: Callable[[], NDArray], agent):
        rho = rho_factory()
        bundle = agent.execute_auditing_bundle(rho, 0.3, 0.1)
        assert _is_density(bundle.audit_state.audited_density_matrix)
        assert bundle.cartan_kinematics.is_business_unchanged
        assert bundle.bekenstein.required_audit_dimension >= _MIN_AUDIT_DIM


class TestPublicAPIExports:
    """Contrato de exportación del módulo (superficie pública)."""

    def test_all_public_symbols_importable(self):
        import app.agents.omega.gromov_witten_auditor_agent as mod

        expected = [
            "GromovWittenAuditorError",
            "BekensteinLimitViolationError",
            "BusinessInvarianceError",
            "CartanStructureError",
            "SpectralCompensationError",
            "AuditingPipelineError",
            "BekensteinDimensionData",
            "CartanKinematicsData",
            "MaurerCartanSolution",
            "GromovWittenCompensation",
            "AuditBundle",
            "Phase1_Bekenstein",
            "Phase2_EhresmannCartan",
            "Phase3_GromovWitten",
            "GromovWittenAuditorAgent",
        ]
        for name in expected:
            assert hasattr(mod, name), f"Falta exportar {name}"
            assert name in getattr(mod, "__all__", []), f"{name} ausente en __all__"

    def test_agent_public_methods(self, agent):
        assert callable(agent.execute_auditing_process)
        assert callable(agent.execute_auditing_bundle)
        assert callable(agent.enforce_bekenstein_bound)
        assert callable(agent.certify_cartan_kinematics)
        assert callable(agent.compensate_aps_eta)