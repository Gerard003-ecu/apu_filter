r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Quantum Epistemic Spectral Auditor (Batería Doctoral)         ║
║  Ruta   : tests/unit/wisdom/test_quantum_epistemic_auditor.py                ║
║  Versión: 6.0.0-Spectral-Kernel-Componential-Categorical-Test-Suite          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ESTRATEGIA DE AISLAMIENTO (honestidad de diseño):                           ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo intenta primero importar el árbol real de dependencias del      ║
║  ecosistema APU (`app.core.mic_algebra`, `app.core.schemas`, etc.). Si el    ║
║  entorno de ejecución no los provee, instala un ÁRBOL DE STUBS mínimo vía    ║
║  `sys.modules`, replicando exactamente el contrato (duck-typing) consumido   ║
║  por `quantum_epistemic_auditor.py`. Esto NO es un mock que oculte           ║
║  comportamiento — es aislamiento de unidad legítimo que permite certificar   ║
║  el KERNEL MATEMÁTICO del módulo sin arrastrar el grafo de dependencias      ║
║  completo de la aplicación.                                                  ║
║                                                                              ║
║  Estructura en 3 fases anidadas, isomorfa al módulo bajo prueba:             ║
║    FASE 1 — Kernel espectral puro (funciones sin estado).                    ║
║    FASE 2 — Componentes acoplados del auditor (métodos privados granulares). ║
║    FASE 3 — Clausura del veredicto y orquestación pública end-to-end.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math
import os
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import scipy.linalg as la


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 0 — AISLAMIENTO DE DEPENDENCIAS (previo a la importación del SUT)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _register_synthetic_stub_tree() -> None:
    """
    Instala en `sys.modules` un árbol de paquetes sintéticos que satisface
    exactamente las importaciones de `quantum_epistemic_auditor.py`, sin
    pretender ser una reimplementación fiel del ecosistema APU completo.
    """
    def _register(name: str, module: types.ModuleType) -> None:
        sys.modules[name] = module
        parent_name, _, attr = name.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], attr, module)

    for pkg_name in (
        "app", "app.core", "app.agents", "app.agents.core",
        "app.agents.core.immune_system", "app.agents.wisdom", "app.wisdom",
    ):
        _register(pkg_name, types.ModuleType(pkg_name))

    # ── app.core.mic_algebra ────────────────────────────────────────────────
    mic_algebra = types.ModuleType("app.core.mic_algebra")

    class _StubMorphism:
        """Sustituto mínimo de `Morphism`: contrato de inicialización vacío."""
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _StubTopologicalInvariantError(Exception):
        """Sustituto de la excepción raíz del framework MIC."""
        pass

    mic_algebra.Morphism = _StubMorphism
    mic_algebra.TopologicalInvariantError = _StubTopologicalInvariantError
    _register("app.core.mic_algebra", mic_algebra)

    # ── app.core.schemas ─────────────────────────────────────────────────────
    schemas = types.ModuleType("app.core.schemas")

    class _StubStratum(Enum):
        PHYSICS = auto()
        TACTICS = auto()
        STRATEGY = auto()
        WISDOM = auto()

    schemas.Stratum = _StubStratum
    _register("app.core.schemas", schemas)

    # ── app.agents.core.immune_system.watcher_agent ─────────────────────────
    watcher_agent = types.ModuleType("app.agents.core.immune_system.watcher_agent")

    @dataclass
    class _StubValidatedStressTensor:
        residual_relative: float
        T_mu_nu: np.ndarray
        G_mu_nu: np.ndarray

    watcher_agent.ValidatedStressTensor = _StubValidatedStressTensor
    _register("app.agents.core.immune_system.watcher_agent", watcher_agent)

    # ── app.wisdom.tomita_takesaki_telescopic_engine ────────────────────────
    tomita_engine = types.ModuleType("app.wisdom.tomita_takesaki_telescopic_engine")

    class _StubUmegakiExtractionState:
        pass

    tomita_engine.UmegakiExtractionState = _StubUmegakiExtractionState
    _register("app.wisdom.tomita_takesaki_telescopic_engine", tomita_engine)

    # ── app.agents.wisdom.connes_spectral_auditor_agent ─────────────────────
    connes_agent = types.ModuleType("app.agents.wisdom.connes_spectral_auditor_agent")

    class _StubConnesAuditState:
        pass

    connes_agent.ConnesAuditState = _StubConnesAuditState
    _register("app.agents.wisdom.connes_spectral_auditor_agent", connes_agent)


def _ensure_dependency_tree_available() -> None:
    """Preserva la integración genuina si el entorno real existe; si no, aísla."""
    try:
        import app.core.mic_algebra          # noqa: F401
        import app.core.schemas              # noqa: F401
        import app.agents.core.immune_system.watcher_agent  # noqa: F401
        import app.wisdom.tomita_takesaki_telescopic_engine  # noqa: F401
        import app.agents.wisdom.connes_spectral_auditor_agent  # noqa: F401
        return
    except ImportError:
        _register_synthetic_stub_tree()


_ensure_dependency_tree_available()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from quantum_epistemic_auditor import (
    # Fase 1
    SpectralAuditThresholds,
    CauchyConservationError,
    SpectralFidelityError,
    DiracCommutatorDivergenceError,
    RelativeEntropySupportError,
    UmegakiVetoError,
    WeakEnergyConditionError,
    KMSNumericalInconsistencyError,
    hermitian_eigendecomposition,
    spectral_functional_calculus,
    operator_commutator_norm,
    four_velocity_energy_density,
    kms_modular_identity_residual,
    # Fase 2
    ConnesTripleState,
    ModularFlowState,
    QuantumEpistemicSpectralAuditor,
    # Fase 3
    EpistemicVerdict,
    EpistemicCoherenceCertificate,
    _classify_bounded_metric,
)

from app.core.mic_algebra import TopologicalInvariantError
from app.core.schemas import Stratum
from app.agents.core.immune_system.watcher_agent import ValidatedStressTensor


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ UTILIDADES DE CONSTRUCCIÓN DE ESCENARIOS (compartidas por las 3 fases)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _build_swap_scenario(p: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Construye el escenario canónico $\rho = \mathrm{diag}(p, 1-p)$,
    $X = \sigma_x$ (operador de intercambio). Propiedad notable, usada
    extensivamente: bajo la regla de Lüders, $X\rho X^\dagger$ permuta la
    diagonal de $\rho$, de modo que
    $$\sigma = \tfrac12(\rho + X\rho X^\dagger) = \mathbb{I}/2 \quad \forall p,$$
    dando la fórmula cerrada $D(\rho\|\sigma) = \ln 2 - H(p)$, con $H$ la
    entropía binaria en nats.
    """
    rho = np.array([[p, 0.0], [0.0, 1.0 - p]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return rho, X


def _binary_entropy_nats(p: float) -> float:
    r"""Calcula $H(p) = -p\ln p - (1-p)\ln(1-p)$ en nats."""
    return sum(-x * math.log(x) for x in (p, 1.0 - p) if x > 0.0)


def _random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""Genera un estado $\rho$ Hermitiano, definido positivo y de traza 1."""
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    H = A @ A.conj().T + dim * np.eye(dim)
    return (H / np.trace(H)).astype(np.complex128)


def _build_valid_stress_tensor(residual_relative: float = 0.0) -> ValidatedStressTensor:
    r"""
    Construye un tensor de esfuerzos válido en la métrica de Minkowski
    (signatura $(-,+,+,+)$), con $T_{00}=1.0$ (densidad de energía positiva)
    y componentes espaciales de presión $0.3$.
    """
    g = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.complex128)
    t = np.diag([1.0, 0.3, 0.3, 0.3]).astype(np.complex128)
    return ValidatedStressTensor(residual_relative=residual_relative, T_mu_nu=t, G_mu_nu=g)


_REST_OBSERVER = np.array([1.0, 0.0, 0.0, 0.0])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — KERNEL ESPECTRAL PURO                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase1KernelEspectralPuro(unittest.TestCase):
    """Pruebas atómicas de las primitivas matemáticas sin estado de clase."""

    # ─────────────── I.0 Integridad de la retícula y la API pública ─────────

    def test_thresholds_default_bounds_are_ordered_correctly(self) -> None:
        r"""Cada par (coherencia, veto) debe satisfacer coherencia < veto."""
        t = SpectralAuditThresholds()
        self.assertLess(t.commutator_coherence_bound, t.commutator_divergence_bound)
        self.assertLess(t.umegaki_coherence_bound, t.umegaki_veto_bound)

    def test_exception_hierarchy_roots_in_topological_invariant_error(self) -> None:
        """Las 7 excepciones granulares deben heredar de la excepción raíz del MIC."""
        for exc in (
            CauchyConservationError, SpectralFidelityError,
            DiracCommutatorDivergenceError, RelativeEntropySupportError,
            UmegakiVetoError, WeakEnergyConditionError, KMSNumericalInconsistencyError,
        ):
            self.assertTrue(issubclass(exc, TopologicalInvariantError))

    def test_module_exposes_expected_public_api(self) -> None:
        """Prueba de regresión: certifica que el contrato público no se degrade en silencio."""
        import quantum_epistemic_auditor as sut
        for symbol in (
            "SpectralAuditThresholds", "ConnesTripleState", "ModularFlowState",
            "EpistemicVerdict", "EpistemicCoherenceCertificate",
            "QuantumEpistemicSpectralAuditor",
        ):
            self.assertTrue(hasattr(sut, symbol), f"Símbolo público ausente: {symbol}")

    # ─────────────── I.1 Diagonalización hermítica y fidelidad GNS ──────────

    def test_hermitian_eigendecomposition_reconstructs_original_matrix(self) -> None:
        r"""$V\,\mathrm{diag}(\lambda_i)\,V^\dagger = \rho$ (Teorema Espectral)."""
        rng = np.random.default_rng(1)
        rho = _random_density_matrix(3, rng)
        eigvals, eigvecs = hermitian_eigendecomposition(rho, 1e-9, 1e-12)
        reconstructed = eigvecs @ np.diag(eigvals.astype(complex)) @ eigvecs.conj().T
        np.testing.assert_allclose(reconstructed, rho, atol=1e-9)

    def test_hermitian_eigendecomposition_rejects_non_self_adjoint_operator(self) -> None:
        r"""$\rho \ne \rho^\dagger \implies$ `SpectralFidelityError`."""
        non_hermitian = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex)
        with self.assertRaises(SpectralFidelityError):
            hermitian_eigendecomposition(non_hermitian, 1e-9, 1e-12)

    def test_hermitian_eigendecomposition_rejects_non_faithful_state(self) -> None:
        r"""Un autovalor nulo (GNS degenerado) debe ser vetado por debajo del piso de fidelidad."""
        singular = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        with self.assertRaises(SpectralFidelityError):
            hermitian_eigendecomposition(singular, 1e-9, 1e-12)

    # ─────────────── I.2 Cálculo funcional espectral ─────────────────────────

    def test_spectral_functional_calculus_identity_function_is_reconstructive(self) -> None:
        r"""$f(\lambda)=\lambda \implies f(\rho) = \rho$ exactamente."""
        rng = np.random.default_rng(2)
        rho = _random_density_matrix(4, rng)
        eigvals, eigvecs = hermitian_eigendecomposition(rho, 1e-9, 1e-12)
        reconstructed = spectral_functional_calculus(eigvals, eigvecs, lambda l: l)
        np.testing.assert_allclose(reconstructed, rho, atol=1e-9)

    def test_spectral_functional_calculus_sqrt_inverse_satisfies_involution(self) -> None:
        r"""$D=\rho^{-1/2} \implies D^2\rho = \mathbb{I}$ (involución algebraica)."""
        rng = np.random.default_rng(3)
        rho = _random_density_matrix(3, rng)
        eigvals, eigvecs = hermitian_eigendecomposition(rho, 1e-9, 1e-12)
        D = spectral_functional_calculus(eigvals, eigvecs, lambda l: 1.0 / np.sqrt(l))
        np.testing.assert_allclose(D @ D @ rho, np.eye(3), atol=1e-8)

    def test_spectral_functional_calculus_log_matches_scipy_logm_reference(self) -> None:
        """Validación cruzada contra `scipy.linalg.logm` (garantía de no-regresión numérica)."""
        rng = np.random.default_rng(4)
        A = rng.standard_normal((3, 3))
        H = (A @ A.T + 3.0 * np.eye(3)).astype(complex)
        eigvals, eigvecs = hermitian_eigendecomposition(H, 1e-9, 1e-9)
        computed = spectral_functional_calculus(eigvals, eigvecs, np.log)
        reference = la.logm(H)
        np.testing.assert_allclose(computed, reference, atol=1e-7)

    # ─────────────── I.3 Conmutador de operadores ────────────────────────────

    def test_operator_commutator_norm_vanishes_for_commuting_diagonal_matrices(self) -> None:
        r"""Dos matrices diagonales conmutan: $\|[A,B]\|_2 = 0$."""
        A = np.diag([1.0, 2.0, 3.0]).astype(complex)
        B = np.diag([4.0, -1.0, 0.5]).astype(complex)
        self.assertAlmostEqual(operator_commutator_norm(A, B), 0.0, places=12)

    def test_operator_commutator_norm_matches_manual_off_diagonal_computation(self) -> None:
        r"""Para $D=\mathrm{diag}(d_0,d_1)$, $\|[D,X]\|_2 = |d_0-d_1|\cdot|X_{01}|$ (caso 2x2)."""
        D = np.diag([2.0, 5.0]).astype(complex)
        X = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=complex)
        expected = abs(2.0 - 5.0) * 3.0
        self.assertAlmostEqual(operator_commutator_norm(D, X), expected, places=9)

    # ─────────────── I.4 Condición de Energía Débil (Cauchy-Momentum) ────────

    def test_four_velocity_energy_density_nominal_rest_observer(self) -> None:
        r"""Para $u=(1,0,0,0)$ y $g=\mathrm{diag}(-1,1,1,1)$: $\varepsilon = T_{00}$."""
        g = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(complex)
        t = np.diag([2.5, 0.1, 0.1, 0.1]).astype(complex)
        density = four_velocity_energy_density(t, g, _REST_OBSERVER, 1e-6)
        self.assertAlmostEqual(density, 2.5, places=9)

    def test_four_velocity_energy_density_rejects_unnormalized_observer(self) -> None:
        r"""$g_{\mu\nu}u^\mu u^\nu \ne -1 \implies$ `WeakEnergyConditionError`."""
        g = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(complex)
        t = np.diag([1.0, 0.0, 0.0, 0.0]).astype(complex)
        u_bad = np.array([2.0, 0.0, 0.0, 0.0])
        with self.assertRaises(WeakEnergyConditionError):
            four_velocity_energy_density(t, g, u_bad, 1e-6)

    def test_four_velocity_energy_density_rejects_negative_energy_density(self) -> None:
        r"""$\varepsilon < 0 \implies$ violación de la Condición de Energía Débil."""
        g = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(complex)
        t = np.diag([-1.0, 0.0, 0.0, 0.0]).astype(complex)
        with self.assertRaises(WeakEnergyConditionError):
            four_velocity_energy_density(t, g, _REST_OBSERVER, 1e-6)

    # ─────────────── I.5 Identidad KMS de referencia (ciclicidad de traza) ───

    def test_kms_modular_identity_residual_vanishes_for_arbitrary_probes(self) -> None:
        r"""
        En dimensión finita (Tipo I), $\mathrm{Tr}(A\rho B) = \mathrm{Tr}(\rho BA)$
        es una consecuencia trivial de la ciclicidad de la traza, válida para
        CUALQUIER par de sondeos $A, B$ (no necesariamente hermíticos).
        """
        rng = np.random.default_rng(7)
        rho = _random_density_matrix(3, rng)
        eigvals, eigvecs = hermitian_eigendecomposition(rho, 1e-9, 1e-12)
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        residual = kms_modular_identity_residual(eigvals, eigvecs, A, B)
        self.assertLess(residual, 1e-8)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ACOPLAMIENTO ESTRUCTURAL DE COMPONENTES                           ║
# ║ (Continúa la Fase 1: cada método consume las primitivas ya certificadas)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase2AcoplamientoEstructural(unittest.TestCase):
    """Pruebas granulares de los métodos privados del auditor, en aislamiento."""

    def setUp(self) -> None:
        self.auditor = QuantumEpistemicSpectralAuditor()
        self.thresholds = SpectralAuditThresholds()

    # ─────────────── II.1 Componente Cauchy-Momentum ─────────────────────────

    def test_validate_cauchy_conservation_returns_expected_dirichlet_energy(self) -> None:
        r"""$E = \mathrm{Tr}(\mathcal{T}\mathcal{G})$ sobre tensores diagonales conocidos."""
        stress = _build_valid_stress_tensor()
        energy = self.auditor._validate_cauchy_conservation(stress, self.thresholds)
        expected = float(np.real(np.trace(stress.T_mu_nu @ stress.G_mu_nu)))
        self.assertAlmostEqual(energy, expected, places=9)

    def test_validate_cauchy_conservation_rejects_residual_leak(self) -> None:
        r"""$\nabla_\mu\mathcal{T}^{\mu\nu} \ne 0 \implies$ `CauchyConservationError`."""
        stress = _build_valid_stress_tensor(residual_relative=1e-3)
        with self.assertRaises(CauchyConservationError):
            self.auditor._validate_cauchy_conservation(stress, self.thresholds)

    def test_validate_cauchy_conservation_rejects_asymmetric_stress_tensor(self) -> None:
        r"""$\mathcal{T}_{\mu\nu} \ne \mathcal{T}_{\nu\mu} \implies$ par interno no balanceado."""
        stress = _build_valid_stress_tensor()
        stress.T_mu_nu[0, 1] = 0.5
        stress.T_mu_nu[1, 0] = 0.1  # Asimetría deliberada.
        with self.assertRaises(CauchyConservationError):
            self.auditor._validate_cauchy_conservation(stress, self.thresholds)

    def test_certify_weak_energy_condition_abstains_without_observer(self) -> None:
        """Ante `u_velocity=None`, el método se abstiene explícitamente (retorna `None`)."""
        stress = _build_valid_stress_tensor()
        self.assertIsNone(
            self.auditor._certify_weak_energy_condition(stress, None, self.thresholds)
        )

    def test_certify_weak_energy_condition_rejects_bad_normalization(self) -> None:
        stress = _build_valid_stress_tensor()
        with self.assertRaises(WeakEnergyConditionError):
            self.auditor._certify_weak_energy_condition(
                stress, np.array([3.0, 0.0, 0.0, 0.0]), self.thresholds
            )

    def test_certify_weak_energy_condition_detects_violation(self) -> None:
        stress = _build_valid_stress_tensor()
        stress.T_mu_nu[0, 0] = -1.0
        with self.assertRaises(WeakEnergyConditionError):
            self.auditor._certify_weak_energy_condition(stress, _REST_OBSERVER, self.thresholds)

    def test_certify_weak_energy_condition_nominal_rest_observer(self) -> None:
        stress = _build_valid_stress_tensor()
        density = self.auditor._certify_weak_energy_condition(
            stress, _REST_OBSERVER, self.thresholds
        )
        self.assertAlmostEqual(density, 1.0, places=9)

    # ─────────────── II.2 Triple de Connes-Dirac ─────────────────────────────

    def test_instantiate_connes_dirac_triple_dirac_operator_squares_to_rho_inverse(self) -> None:
        r"""$D^2 = \rho^{-1}$: certifica el Teorema Espectral aplicado al análogo de Dirac."""
        rho, X = _build_swap_scenario(0.6)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        reconstructed_inverse = triple.dirac_operator @ triple.dirac_operator
        np.testing.assert_allclose(reconstructed_inverse, la.inv(rho), atol=1e-9)

    def test_instantiate_connes_dirac_triple_lipschitz_limit_formula(self) -> None:
        r"""$L_{\max} = C_{\text{base}}/(1+\lambda_{\text{disp}})$ con $\lambda_{\text{disp}}=0.8$."""
        rho, X = _build_swap_scenario(0.9)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 2.0, self.thresholds)
        expected = 2.0 / (1.0 + 0.8)
        self.assertAlmostEqual(triple.lipschitz_limit, expected, places=9)

    def test_instantiate_connes_dirac_triple_propagates_non_self_adjoint_state(self) -> None:
        non_hermitian = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex)
        with self.assertRaises(SpectralFidelityError):
            self.auditor._instantiate_connes_dirac_triple(
                non_hermitian, np.eye(2, dtype=complex), 1.5, self.thresholds
            )

    def test_instantiate_connes_dirac_triple_raises_on_commutator_divergence(self) -> None:
        r"""Un autovalor ínfimo de $\rho$ magnifica $D=\rho^{-1/2}$, disparando el conmutador."""
        rho = np.array([[1.0 - 5e-12, 0.0], [0.0, 5e-12]], dtype=complex)
        X = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=complex)
        with self.assertRaises(DiracCommutatorDivergenceError):
            self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)

    # ─────────────── II.3 Flujo modular de Takesaki + Lüders ─────────────────

    def test_apply_modular_flow_identity_at_zero_zoom(self) -> None:
        r"""$\sigma_0(X) = \rho^{0}X\rho^{0} = X$ exactamente."""
        rho, X = _build_swap_scenario(0.7)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        state = self.auditor._apply_modular_flow(triple, X, 0.0)
        np.testing.assert_allclose(state.observable_zoomed, X, atol=1e-9)

    def test_apply_modular_flow_round_trip_group_inversion(self) -> None:
        r"""Propiedad de grupo del flujo modular: $\sigma_{-\lambda}(\sigma_\lambda(X)) = X$."""
        rho, X = _build_swap_scenario(0.8)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        forward = self.auditor._apply_modular_flow(triple, X, 0.37).observable_zoomed
        backward = self.auditor._apply_modular_flow(triple, forward, -0.37).observable_zoomed
        np.testing.assert_allclose(backward, X, atol=1e-8)

    def test_apply_modular_flow_rejects_non_finite_zoom_parameter(self) -> None:
        rho, X = _build_swap_scenario(0.5)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        with self.assertRaises(TopologicalInvariantError):
            self.auditor._apply_modular_flow(triple, X, float("nan"))

    def test_apply_luders_transformation_rejects_null_observable(self) -> None:
        rho, _ = _build_swap_scenario(0.5)
        with self.assertRaises(TopologicalInvariantError):
            self.auditor._apply_luders_transformation(rho, np.zeros((2, 2), dtype=complex))

    def test_apply_luders_transformation_preserves_exact_trace_normalization(self) -> None:
        rho, X = _build_swap_scenario(0.9)
        sigma = self.auditor._apply_luders_transformation(rho, X)
        self.assertAlmostEqual(float(np.real(np.trace(sigma))), 1.0, places=9)

    # ─────────────── II.4 Divergencia de Umegaki ─────────────────────────────

    def test_compute_umegaki_divergence_vanishes_for_identical_states(self) -> None:
        r"""$D(\rho\|\rho) = 0$ para cualquier estado fiel (propiedad general, no solo $p=0.5$)."""
        rho, X = _build_swap_scenario(0.9)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        divergence = self.auditor._compute_umegaki_divergence(triple, rho, self.thresholds)
        self.assertAlmostEqual(divergence, 0.0, places=7)

    def test_compute_umegaki_divergence_matches_closed_form_maximally_mixed_reference(self) -> None:
        r"""$D(\rho\|\mathbb{I}/2) = \ln 2 - H(p)$, forma cerrada del escenario canónico."""
        p = 0.9
        rho, X = _build_swap_scenario(p)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        sigma = np.eye(2, dtype=complex) / 2.0
        divergence = self.auditor._compute_umegaki_divergence(triple, sigma, self.thresholds)
        expected = math.log(2.0) - _binary_entropy_nats(p)
        self.assertAlmostEqual(divergence, expected, places=6)

    def test_compute_umegaki_divergence_raises_veto_for_highly_peaked_state(self) -> None:
        p = 0.999
        rho, X = _build_swap_scenario(p)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        sigma = np.eye(2, dtype=complex) / 2.0
        with self.assertRaises(UmegakiVetoError):
            self.auditor._compute_umegaki_divergence(triple, sigma, self.thresholds)

    def test_compute_umegaki_divergence_raises_on_araki_support_violation(self) -> None:
        r"""$\mathrm{supp}(\rho) \not\subseteq \mathrm{supp}(\sigma) \implies$ divergencia infinita."""
        rho, X = _build_swap_scenario(0.5)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        singular_sigma = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        with self.assertRaises(RelativeEntropySupportError):
            self.auditor._compute_umegaki_divergence(triple, singular_sigma, self.thresholds)

    # ─────────────── II.5 Distorsión de Dixmier (analogía finita) ────────────

    def test_compute_dixmier_distortion_unity_at_zero_zoom(self) -> None:
        rho, X = _build_swap_scenario(0.7)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        state = self.auditor._apply_modular_flow(triple, X, 0.0)
        ratio = self.auditor._compute_dixmier_distortion(X, state.observable_zoomed)
        self.assertAlmostEqual(ratio, 1.0, places=9)

    def test_compute_dixmier_distortion_reflects_conformal_scaling(self) -> None:
        rho, X = _build_swap_scenario(0.9)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, self.thresholds)
        state = self.auditor._apply_modular_flow(triple, X, 1.0)
        ratio = self.auditor._compute_dixmier_distortion(X, state.observable_zoomed)
        self.assertGreater(ratio, 0.0)
        self.assertNotAlmostEqual(ratio, 1.0, places=3)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — CLAUSURA CATEGÓRICA DEL VEREDICTO Y ORQUESTACIÓN PÚBLICA          ║
# ║ (Continúa la Fase 2: compone sus componentes ya certificados)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase3ClausuraCategoricaYOrquestacion(unittest.TestCase):
    """Pruebas de la retícula de verdicto Ω₃ y del orquestador público end-to-end."""

    def setUp(self) -> None:
        self.auditor = QuantumEpistemicSpectralAuditor()

    # ─────────────── III.1 Retícula de verdicto Ω₃ ───────────────────────────

    def test_epistemic_verdict_total_ordering(self) -> None:
        self.assertLess(EpistemicVerdict.COHERENT, EpistemicVerdict.DEGRADED)
        self.assertLess(EpistemicVerdict.DEGRADED, EpistemicVerdict.VETOED)
        self.assertEqual(
            max(EpistemicVerdict.COHERENT, EpistemicVerdict.VETOED), EpistemicVerdict.VETOED
        )

    def test_classify_bounded_metric_three_regimes_with_inclusive_boundaries(self) -> None:
        self.assertEqual(_classify_bounded_metric(0.1, 0.25, 0.5), EpistemicVerdict.COHERENT)
        self.assertEqual(_classify_bounded_metric(0.25, 0.25, 0.5), EpistemicVerdict.COHERENT)
        self.assertEqual(_classify_bounded_metric(0.4, 0.25, 0.5), EpistemicVerdict.DEGRADED)
        self.assertEqual(_classify_bounded_metric(0.5, 0.25, 0.5), EpistemicVerdict.DEGRADED)
        self.assertEqual(_classify_bounded_metric(0.9, 0.25, 0.5), EpistemicVerdict.VETOED)

    # ─────────────── III.2 Certificación numérica KMS ────────────────────────

    def test_certify_kms_numerical_consistency_passes_within_default_tolerance(self) -> None:
        rho, X = _build_swap_scenario(0.7)
        thresholds = SpectralAuditThresholds()
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, thresholds)
        rng = np.random.default_rng(99)
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        B = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        residual = self.auditor.certify_kms_numerical_consistency(triple, A, B, thresholds)
        self.assertLess(residual, 1e-6)

    def test_certify_kms_numerical_consistency_raises_for_zero_tolerance(self) -> None:
        """Con tolerancia nula, cualquier residuo de redondeo debe disparar la excepción."""
        rho, X = _build_swap_scenario(0.5)
        strict = SpectralAuditThresholds(kms_numerical_tol=0.0)
        triple = self.auditor._instantiate_connes_dirac_triple(rho, X, 1.5, strict)
        rng = np.random.default_rng(123)
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        B = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        with self.assertRaises(KMSNumericalInconsistencyError):
            self.auditor.certify_kms_numerical_consistency(triple, A, B, strict)

    # ─────────────── III.3 Orquestación pública end-to-end ───────────────────

    def test_execute_epistemic_audit_full_nominal_pipeline_returns_coherent_certificate(self) -> None:
        r"""Con $p=0.5$ ($\rho \propto \mathbb{I}$), todo el pipeline colapsa a valores exactos."""
        rho, X = _build_swap_scenario(0.5)
        stress = _build_valid_stress_tensor()
        cert = self.auditor.execute_epistemic_audit(
            stress_tensor_data=stress, rho_mac=rho, observable_X=X,
            zoom_lambda=0.1, c_base=1.5,
        )
        self.assertIsInstance(cert, EpistemicCoherenceCertificate)
        self.assertEqual(cert.verdict, EpistemicVerdict.COHERENT)
        self.assertTrue(cert.is_coherent)
        self.assertAlmostEqual(cert.commutator_norm, 0.0, places=6)
        self.assertAlmostEqual(cert.umegaki_divergence, 0.0, places=6)
        self.assertAlmostEqual(cert.lipschitz_limit, 1.5, places=9)
        self.assertAlmostEqual(cert.dixmier_volume_ratio, 1.0, places=6)
        self.assertLess(cert.kms_numerical_residual, 1e-6)
        self.assertIsNone(cert.weak_energy_density)

    def test_execute_epistemic_audit_produces_degraded_verdict_for_moderately_peaked_state(self) -> None:
        r"""Con $p=0.9$: $D(\rho\|\sigma) \approx 0.368 \in (0.25, 0.50)$ -> DEGRADED."""
        rho, X = _build_swap_scenario(0.9)
        stress = _build_valid_stress_tensor()
        cert = self.auditor.execute_epistemic_audit(stress, rho, X, zoom_lambda=0.2)
        expected = math.log(2.0) - _binary_entropy_nats(0.9)
        self.assertAlmostEqual(cert.umegaki_divergence, expected, places=6)
        self.assertEqual(cert.verdict, EpistemicVerdict.DEGRADED)
        self.assertFalse(cert.is_coherent)

    def test_execute_epistemic_audit_raises_umegaki_veto_for_highly_peaked_state(self) -> None:
        r"""Con $p=0.999$: $D(\rho\|\sigma) \approx 0.685 > 0.50$ -> `UmegakiVetoError`."""
        rho, X = _build_swap_scenario(0.999)
        stress = _build_valid_stress_tensor()
        with self.assertRaises(UmegakiVetoError):
            self.auditor.execute_epistemic_audit(stress, rho, X, zoom_lambda=0.05)

    def test_execute_epistemic_audit_propagates_cauchy_conservation_error(self) -> None:
        rho, X = _build_swap_scenario(0.5)
        bad_stress = _build_valid_stress_tensor(residual_relative=1e-3)
        with self.assertRaises(CauchyConservationError):
            self.auditor.execute_epistemic_audit(bad_stress, rho, X, zoom_lambda=0.1)

    def test_execute_epistemic_audit_propagates_weak_energy_condition_error(self) -> None:
        rho, X = _build_swap_scenario(0.5)
        stress = _build_valid_stress_tensor()
        stress.T_mu_nu[0, 0] = -1.0
        with self.assertRaises(WeakEnergyConditionError):
            self.auditor.execute_epistemic_audit(
                stress, rho, X, zoom_lambda=0.1, u_velocity=_REST_OBSERVER
            )

    def test_execute_epistemic_audit_returns_weak_energy_density_when_observer_supplied(self) -> None:
        rho, X = _build_swap_scenario(0.5)
        stress = _build_valid_stress_tensor()
        cert = self.auditor.execute_epistemic_audit(
            stress, rho, X, zoom_lambda=0.1, u_velocity=_REST_OBSERVER
        )
        self.assertAlmostEqual(cert.weak_energy_density, 1.0, places=9)

    def test_execute_epistemic_audit_propagates_spectral_fidelity_error(self) -> None:
        rho, X = _build_swap_scenario(0.5)
        rho[0, 1] = 5.0  # Rompe deliberadamente la autoadjunción.
        stress = _build_valid_stress_tensor()
        with self.assertRaises(SpectralFidelityError):
            self.auditor.execute_epistemic_audit(stress, rho, X, zoom_lambda=0.1)

    def test_execute_epistemic_audit_propagates_commutator_divergence_error(self) -> None:
        rho = np.array([[1.0 - 5e-12, 0.0], [0.0, 5e-12]], dtype=complex)
        X = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=complex)
        stress = _build_valid_stress_tensor()
        with self.assertRaises(DiracCommutatorDivergenceError):
            self.auditor.execute_epistemic_audit(stress, rho, X, zoom_lambda=0.1)

    def test_execute_epistemic_audit_respects_custom_thresholds_override(self) -> None:
        r"""Un umbral de veto más estricto ($0.30 < 0.368$) debe vetar lo que el default degradaría."""
        rho, X = _build_swap_scenario(0.9)
        stress = _build_valid_stress_tensor()
        strict = SpectralAuditThresholds(umegaki_veto_bound=0.30)
        with self.assertRaises(UmegakiVetoError):
            self.auditor.execute_epistemic_audit(
                stress, rho, X, zoom_lambda=0.1, thresholds=strict
            )


def load_full_suite() -> unittest.TestSuite:
    """Ensambla la suite completa preservando el orden narrativo de las 3 fases anidadas."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestFase1KernelEspectralPuro))
    suite.addTests(loader.loadTestsFromTestCase(TestFase2AcoplamientoEstructural))
    suite.addTests(loader.loadTestsFromTestCase(TestFase3ClausuraCategoricaYOrquestacion))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_full_suite())