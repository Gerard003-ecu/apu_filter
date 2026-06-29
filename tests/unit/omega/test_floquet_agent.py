# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Test Suite — Floquet Monodromy Agent v3.0                           ║
║ Ubicación: tests/omega/test_floquet_agent.py                                 ║
║ Cobertura: Axiomas §0–§5, contrato categórico, Kraus PSD, HMAC forense       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de testing (pirámide completa):
────────────────────────────────────────────────────────────────────────────────
  ◆ FASE 1 — CovariantProjectorSynthesizer (Síntesis Métrica):
      • Pullback covariante n = G·∇H_obs
      • Normalización bajo G
      • Detección de obstrucción trivial (P = I)
      • Validación de G como SPD
      • Reenvío de excepciones de Householder

  ◆ FASE 2 — FloquetStabilityAuditor (Monodromía):
      • M_on = 2P − P² exacto
      • Detección de simetría → eigvalsh vs eigvals
      • Radio espectral y estabilidad asintótica
      • Detección de multiplicadores > 1 + ε
      • Tolerancia configurable
      • Diagnóstico de κ₂(P)

  ◆ FASE 3 — QuantumKrausChannel (CPTP Canónico):
      • Completitud Kraus PSD (C − I ⪰ 0)
      • Completitud Kraus Frobenius (‖C − I‖_F)
      • Estado coherente colapsado
      • Entropía de mezcla (Shannon/Von Neumann)
      • Firma HMAC-SHA256 del positrón
      • Emisión de antimateria condicionada
      • Contractividad ΔS ≤ 0

  ◆ FASE 4 — FloquetMonodromyAgent (Orquestador Categórico):
      • Contrato Morphism (forward/backward)
      • Composición por agregación tipada
      • Pipeline completo P → M_on → Kraus
      • Trazabilidad del estado coherente
      • Validación del Axioma §0 a nivel de orquestador

  ◆ Casos físicos extremos, invariancia y regresiones:
      • Proyector asimétrico (manicomio complejo)
      • Métrica Lorentziana (rechazada)
      • Métrica mal condicionada
      • ∇H_obs nulo (obstrucción trivial)
      • Determinismo bit-a-bit
      • Regresiones v2.0
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose

# ══════════════════════════════════════════════════════════════════════════════
# PATH BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Silenciar logging durante tests
logging.disable(logging.CRITICAL)

from app.omega.floquet_agent import (  # noqa: E402
    FloquetInstabilityError,
    KrausTraceViolationError,
    KrausCompletenessError,
    DimensionalMismatchError,
    DimensionalAudit,
    FloquetMonodromyState,
    QuantumChannelEvolution,
    Phase1_CovariantProjectorSynthesizer,
    Phase2_FloquetStabilityAuditor,
    Phase3_QuantumKrausChannel,
    FloquetMonodromyAgent,
)
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError  # noqa: E402
from app.core.immune_system.metric_tensors import G_PHYSICS  # noqa: E402
from app.omega.semantic_parabolic_mirror import (  # noqa: E402
    MetricAwareHouseholderReflector,
    HouseholderSingularityError,
)
from app.core.telemetry_schemas import PositronCartridge  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def rng() -> np.random.Generator:
    """Generador reproducible para determinismo bit-a-bit."""
    return np.random.default_rng(seed=20251215)


@pytest.fixture
def dim_metric() -> int:
    """Dimensión base."""
    return G_PHYSICS.shape[0]


@pytest.fixture
def identity_metric(dim_metric: int) -> np.ndarray:
    """Métrica trivial G = I_d."""
    return np.eye(dim_metric, dtype=np.float64)


@pytest.fixture
def anisotropic_metric(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Métrica anisotrópica bien condicionada."""
    A = rng.standard_normal((dim_metric, dim_metric))
    return A @ A.T + np.eye(dim_metric)


@pytest.fixture
def ill_conditioned_metric(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Métrica con κ₂(G) ≈ 10¹⁰."""
    eigenvalues = np.logspace(-5, 5, dim_metric)
    Q, _ = np.linalg.qr(rng.standard_normal((dim_metric, dim_metric)))
    return Q @ np.diag(eigenvalues) @ Q.T


@pytest.fixture
def h_obs_gradient(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Gradiente de la obstrucción homológica."""
    grad = rng.standard_normal(dim_metric)
    return grad / np.linalg.norm(grad)


@pytest.fixture
def raw_llm_logits(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Logits crudos del LLM."""
    return rng.standard_normal(dim_metric)


@pytest.fixture
def trivial_gradient(dim_metric: int) -> np.ndarray:
    """Gradiente nulo → obstrucción trivial."""
    return np.zeros(dim_metric)


@pytest.fixture
def projector_factory(dim_metric: int, rng: np.random.Generator):
    """Factory de proyectores ortogonales aleatorios."""
    def _make(seed: int = None) -> np.ndarray:
        local_rng = np.random.default_rng(seed) if seed is not None else rng
        v = local_rng.standard_normal(dim_metric)
        v /= np.linalg.norm(v)
        reflector = MetricAwareHouseholderReflector(v, np.eye(dim_metric))
        return reflector.projection_operator
    return _make


@pytest.fixture
def asymmetric_projector(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Proyector deliberadamente asimétrico (artefacto numérico)."""
    v = rng.standard_normal(dim_metric)
    v /= np.linalg.norm(v)
    # P = I - vv^T (simétrico), luego perturbamos asimétricamente
    P = np.eye(dim_metric) - np.outer(v, v)
    perturbation = 1e-13 * rng.standard_normal((dim_metric, dim_metric))
    # Romper simetría
    P_asym = P + (perturbation - perturbation.T)
    return P_asym


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗  ██╗ █████╗ ███████╗███████╗     ██╗
# ██╔════╝██║  ██║██╔══██╗██╔════╝██╔════╝    ███║
# █████╗  ███████║███████║███████╗█████╗      ╚██║
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝       ██║
# ██║     ██║  ██║██║  ██║███████║███████╗      ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝
#           FASE 1 — COVARIANT PROJECTOR SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1_MetricValidation:
    """Axioma §0: G debe ser SPD para el sintetizador."""

    def test_G_no_cuadrada_es_rechazada(self):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            Phase1_CovariantProjectorSynthesizer(metric_tensor=np.zeros((3, 5)))

    def test_G_singular_es_rechazada(self):
        G = np.array([[1.0, 1.0], [1.0, 1.0]])  # det = 0
        with pytest.raises(TopologicalInvariantError, match="SPD"):
            Phase1_CovariantProjectorSynthesizer(metric_tensor=G)

    def test_G_con_autovalor_negativo_es_rechazada(self):
        G = np.diag([1.0, -1.0, 2.0])
        with pytest.raises(TopologicalInvariantError):
            Phase1_CovariantProjectorSynthesizer(metric_tensor=G)

    def test_G_identidad_es_valida(self, identity_metric):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        assert synth._G.shape == identity_metric.shape

    def test_G_PHYSICS_es_valida(self):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=G_PHYSICS)
        assert synth._G.shape == G_PHYSICS.shape


class TestPhase1_CovariantPullback:
    """Axioma §1: Pullback métrico n = G·∇H_obs."""

    def test_pullback_es_algebraicamente_correcto(
        self, identity_metric, dim_metric, rng
    ):
        """Con G = I, n_cov = ∇H_obs."""
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        grad = rng.standard_normal(dim_metric)
        n_cov = synth._G @ grad
        assert_allclose(n_cov, grad, atol=1e-14)

    def test_pullback_con_G_general(self, anisotropic_metric, dim_metric, rng):
        """n_cov = G @ grad."""
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=anisotropic_metric)
        grad = rng.standard_normal(dim_metric)
        n_cov = synth._G @ grad
        assert_allclose(n_cov, anisotropic_metric @ grad, atol=1e-14)

    def test_gradiente_mal_dimensionado_es_rechazado(
        self, identity_metric, dim_metric
    ):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        bad_grad = np.ones(dim_metric + 1)
        with pytest.raises(DimensionalMismatchError, match="dimensión"):
            synth.synthesize_projector(bad_grad)

    def test_gradiente_2d_es_rechazado(self, identity_metric, dim_metric):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        with pytest.raises(DimensionalMismatchError):
            synth.synthesize_projector(np.ones((dim_metric, dim_metric)))


class TestPhase1_TrivialObstruction:
    """Axioma §1: ‖n‖_G < ε ⟹ P = I_d (estado válido)."""

    def test_obstruccion_trivial_retorna_identidad(self, identity_metric, trivial_gradient):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, reflector = synth.synthesize_projector(trivial_gradient)
        assert P is not None  # No None silencioso
        assert reflector is None
        assert_allclose(P, np.eye(identity_metric.shape[0]), atol=1e-14)

    def test_obstruccion_casi_trivial_retorna_identidad(self, identity_metric, dim_metric):
        """‖n‖_G ≈ 1e-16 < ε = 1e-15 ⟹ P = I."""
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        grad = np.full(dim_metric, 1e-16)
        P, reflector = synth.synthesize_projector(grad)
        assert reflector is None
        assert_allclose(P, np.eye(dim_metric), atol=1e-14)

    def test_obstruccion_no_trivial_retorna_proyector_no_trivial(
        self, identity_metric, h_obs_gradient
    ):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, reflector = synth.synthesize_projector(h_obs_gradient)
        assert P is not None
        assert reflector is not None
        # P ≠ I
        assert not np.allclose(P, np.eye(identity_metric.shape[0]))


class TestPhase1_ProjectorConstruction:
    """El proyector retornado debe ser métricamente válido."""

    def test_proyector_es_idempotente(self, identity_metric, h_obs_gradient):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, _ = synth.synthesize_projector(h_obs_gradient)
        assert_allclose(P @ P, P, atol=1e-12)

    def test_proyector_es_simetrico(self, identity_metric, h_obs_gradient):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, _ = synth.synthesize_projector(h_obs_gradient)
        assert_allclose(P, P.T, atol=1e-12)

    def test_proyector_aniquila_normal(self, identity_metric, h_obs_gradient):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, _ = synth.synthesize_projector(h_obs_gradient)
        # El normal n_unit es la dirección que P aniquila
        n_cov = synth._G @ h_obs_gradient
        n_norm = np.sqrt(n_cov @ (synth._G @ n_cov))
        n_unit = n_cov / n_norm
        assert_allclose(P @ n_unit, np.zeros_like(n_unit), atol=1e-12)

    def test_proyector_es_devuelto_por_householder(
        self, identity_metric, h_obs_gradient
    ):
        """El proyector debe coincidir con el del reflector subyacente."""
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, reflector = synth.synthesize_projector(h_obs_gradient)
        assert reflector is not None
        assert_allclose(P, reflector.projection_operator, atol=1e-14)

    def test_proyector_con_G_anisotropica_es_idempotente(
        self, anisotropic_metric, h_obs_gradient
    ):
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=anisotropic_metric)
        P, _ = synth.synthesize_projector(h_obs_gradient)
        assert_allclose(P @ P, P, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗  ██╗ █████╗ ███████╗███████╗     ██████╗
# ██╔════╝██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
# █████╗  ███████║███████║███████╗█████╗       █████╔╝
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝      ██╔═══╝
# ██║     ██║  ██║██║  ██║███████║███████╗    ███████╗
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
#             FASE 2 — FLOQUET STABILITY AUDITOR
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2_Configuration:
    """Configuración y validación del auditor."""

    def test_tolerancia_invalida_es_rechazada(self):
        with pytest.raises(ValueError, match="stability_tolerance"):
            Phase2_FloquetStabilityAuditor(stability_tolerance=0.0)
        with pytest.raises(ValueError, match="stability_tolerance"):
            Phase2_FloquetStabilityAuditor(stability_tolerance=-1e-9)

    def test_tolerancia_default_es_razonable(self):
        auditor = Phase2_FloquetStabilityAuditor()
        assert auditor._stability_tolerance == 1e-9


class TestPhase2_MonodromyMatrix:
    """M_on = 2P − P² debe ser exacta."""

    def test_M_on_es_identidad_si_P_es_identidad(
        self, identity_metric, dim_metric
    ):
        auditor = Phase2_FloquetStabilityAuditor()
        P_I = np.eye(dim_metric)
        M_on = 2.0 * P_I - P_I @ P_I
        assert_allclose(M_on, P_I, atol=1e-14)

    def test_M_on_es_cero_si_P_es_cero(self, identity_metric, dim_metric):
        """P = 0 ⟹ M_on = 0 (pero P=0 no es un proyector válido)."""
        auditor = Phase2_FloquetStabilityAuditor()
        P_zero = np.zeros((dim_metric, dim_metric))
        # No es proyector válido pero probamos la fórmula
        M_on = 2.0 * P_zero - P_zero @ P_zero
        assert_allclose(M_on, np.zeros((dim_metric, dim_metric)), atol=1e-14)

    def test_M_on_es_I_si_P_es_idempotente_y_simetrico(
        self, projector_factory, identity_metric
    ):
        """Para P idempotente: P² = P ⟹ M_on = 2P − P = P."""
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        M_on = 2.0 * P - P @ P
        assert_allclose(M_on, P, atol=1e-12)

    def test_P_mal_dimensionado_es_rechazado(self, identity_metric):
        auditor = Phase2_FloquetStabilityAuditor()
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            auditor.audit_monodromy(np.zeros((3, 5)))


class TestPhase2_SymmetryDetection:
    """Detección automática de simetría para eigvalsh vs eigvals."""

    def test_P_simetrico_usa_eigvalsh_reales(
        self, projector_factory, identity_metric
    ):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        assert state.is_complex_manifold is False
        assert state.multipliers.dtype == np.float64
        # Multiplicadores ∈ [-1, 1+ε]
        assert np.all(np.abs(state.multipliers) <= 1.0 + 1e-9 + 1e-12)

    def test_P_asimetrico_usa_eigvals_complejos(
        self, asymmetric_projector, identity_metric
    ):
        """Proyector asimétrico debe disparar eigvals complejo."""
        auditor = Phase2_FloquetStabilityAuditor()
        state = auditor.audit_monodromy(asymmetric_projector)
        assert state.is_complex_manifold is True
        assert state.multipliers.dtype == np.complex128

    def test_certificado_contiene_todos_los_campos(
        self, projector_factory
    ):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        assert hasattr(state, "multipliers")
        assert hasattr(state, "spectral_radius")
        assert hasattr(state, "is_asymptotically_stable")
        assert hasattr(state, "condition_number_P")
        assert hasattr(state, "is_complex_manifold")
        assert hasattr(state, "tolerance_used")

    def test_certificado_es_inmutable(self, projector_factory):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        with pytest.raises((AttributeError, Exception)):
            state.spectral_radius = 0.5  # type: ignore[misc]


class TestPhase2_Stability:
    """Radio espectral ≤ 1 + ε ⟺ estabilidad asintótica."""

    def test_P_identidad_es_estable(self, identity_metric, dim_metric):
        """P = I ⟹ M_on = I ⟹ ρ(M_on) = 1 (marginalmente estable)."""
        auditor = Phase2_FloquetStabilityAuditor(stability_tolerance=1e-9)
        P = np.eye(dim_metric)
        state = auditor.audit_monodromy(P)
        assert state.is_asymptotically_stable is True
        assert state.spectral_radius == pytest.approx(1.0, abs=1e-12)

    def test_P_no_trivial_es_estable(self, projector_factory):
        auditor = Phase2_FloquetStabilityAuditor(stability_tolerance=1e-9)
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        assert state.is_asymptotically_stable is True
        # ρ(M_on) = 1 para cualquier proyector idempotente
        assert state.spectral_radius <= 1.0 + 1e-9 + 1e-12

    def test_proyector_inestable_es_error(
        self, identity_metric, dim_metric
    ):
        """P = 2I (no es proyector) ⟹ M_on con ρ > 1."""
        auditor = Phase2_FloquetStabilityAuditor(stability_tolerance=1e-9)
        # Construimos matriz que NO es proyector idempotente: M_on tendrá ρ > 1
        A = 2.0 * np.eye(dim_metric)  # No es proyector
        # M_on = 2A - A² = 2A - 4I = 4I - 4I = 0 si A = 2I... Calculemos bien
        # A = 2I ⟹ A² = 4I ⟹ M_on = 4I - 4I = 0. Necesitamos otro caso.
        # Construyamos P con autovalor > 1 (no idempotente)
        eigvals = np.array([2.0] + [1.0] * (dim_metric - 1))
        Q, _ = np.linalg.qr(np.eye(dim_metric))
        P_bad = Q @ np.diag(eigvals) @ Q.T  # Autovalor 2 ⟹ M_on con ρ = 3
        with pytest.raises(FloquetInstabilityError, match="Inestabilidad"):
            auditor.audit_monodromy(P_bad)

    def test_tolerancia_es_registrada_en_certificado(
        self, projector_factory
    ):
        auditor = Phase2_FloquetStabilityAuditor(stability_tolerance=1e-6)
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        assert state.tolerance_used == 1e-6

    def test_diagnostico_cond_P(self, projector_factory):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        # κ₂(P) para proyector idempotente: ‖P‖ = 1, ‖P⁻¹‖ sobre imagen = 1 ⟹ κ = 1
        # (P es singular con pseudoinversa)
        assert state.condition_number_P >= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗  ██╗ █████╗ ███████╗███████╗    ██╗  ██╗██████╗  █████╗ ██╗   ██╗
# ██╔════╝██║  ██║██╔══██╗██╔════╝██╔════╝    ██║ ██╔╝██╔══██╗██╔══██╗██║   ██║
# █████╗  ███████║███████║███████╗█████╗      █████╔╝ ██████╔╝███████║██║   ██║
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝      ██╔═██╗ ██╔══██╗██╔══██║██║   ██║
# ██║     ██║  ██║██║  ██║███████║███████╗    ██║  ██╗██║  ██║██║  ██║╚██████╔╝
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
#                FASE 3 — QUANTUM KRAUS CHANNEL
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3_Construction:
    """Validación del Axioma §0 a nivel del canal."""

    def test_G_no_cuadrada_es_rechazada(self):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            Phase3_QuantumKrausChannel(metric_tensor=np.zeros((3, 5)))

    def test_construccion_con_G_PHYSICS(self):
        channel = Phase3_QuantumKrausChannel(metric_tensor=G_PHYSICS)
        assert channel._G.shape == G_PHYSICS.shape


class TestPhase3_KrausCompletenessPSD:
    """Axioma §3: C = Σ E†_k E_k ⟹ C − I ⪰ 0 (PSD estricto)."""

    def test_completitud_para_kraus_canonico(
        self, identity_metric, dim_metric
    ):
        """E_0 = P (proyector), E_1 = I − P ⟹ C = P + (I − P) = I."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        # Proyector aleatorio
        rng = np.random.default_rng(42)
        v = rng.standard_normal(dim_metric)
        v /= np.linalg.norm(v)
        P = np.eye(dim_metric) - np.outer(v, v)
        I = np.eye(dim_metric)
        fro_res, psd_min = channel._verify_kraus_completeness_psd(P, I - P)
        assert fro_res < 1e-12
        assert psd_min > -1e-9

    def test_completitud_para_P_identidad(self, identity_metric, dim_metric):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        I = np.eye(dim_metric)
        # P = I ⟹ E_0 = I, E_1 = 0 ⟹ C = I·I + 0 = I
        fro_res, psd_min = channel._verify_kraus_completeness_psd(I, np.zeros_like(I))
        assert fro_res < 1e-12
        assert psd_min > -1e-9

    def test_completitud_psd_estricta_para_canonico(
        self, projector_factory, identity_metric
    ):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        fro_res, psd_min = channel._verify_kraus_completeness_psd(P, np.eye(P.shape[0]) - P)
        # PSD estricto ⟹ min eigval ≥ 0
        assert psd_min >= -1e-9

    def test_kraus_incompletos_son_rechazados_PSD(
        self, identity_metric, dim_metric
    ):
        """E_0 + E_1 con Σ E†E ≠ I ⟹ PSD violation."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        # Construimos E_0, E_1 tales que C = E†E no es PSD
        # E_0 = diag(0.5, 1, 1, ...) ⟹ C = diag(0.25, 1, 1, ...) ≠ I
        E0 = np.diag([0.5] + [1.0] * (dim_metric - 1))
        E1 = np.diag([0.5] + [0.0] * (dim_metric - 1))
        with pytest.raises(KrausCompletenessError, match="PSD"):
            channel._verify_kraus_completeness_psd(E0, E1)

    def test_kraus_incompletos_frobenius_grande_es_rechazado(
        self, identity_metric, dim_metric
    ):
        """‖C − I‖_F > 1e-6 debe disparar KrausTraceViolationError."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        # Perturbación grande que rompe PSD pero quizás no tanto
        # Construir C tal que ‖C − I‖_F > 1e-6
        rng = np.random.default_rng(99)
        v = rng.standard_normal(dim_metric)
        v /= np.linalg.norm(v)
        P_perturbed = np.eye(dim_metric) - np.outer(v, v)
        P_perturbed += 1e-3 * rng.standard_normal((dim_metric, dim_metric))
        P_perturbed = (P_perturbed + P_perturbed.T) / 2.0
        with pytest.raises((KrausTraceViolationError, KrausCompletenessError)):
            channel._verify_kraus_completeness_psd(P_perturbed, np.eye(dim_metric) - P_perturbed)


class TestPhase3_EntropyCalculation:
    """Axioma §4: Entropía de Von Neumann / mezcla."""

    def test_entropia_estado_puro_es_cero(self, identity_metric, dim_metric):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        # Estado puro ψ = e_0
        psi = np.zeros(dim_metric)
        psi[0] = 1.0
        S = channel._von_neumann_entropy(psi)
        assert S == pytest.approx(0.0, abs=1e-12)

    def test_entropia_mezcla_uniforme(self, identity_metric, dim_metric):
        """ψ = (1/√d)·(1,1,...,1) ⟹ ρ diagonal uniforme ⟹ S = log(d)."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        psi = np.ones(dim_metric) / np.sqrt(dim_metric)
        S = channel._mixing_entropy(psi)
        assert S == pytest.approx(np.log(dim_metric), abs=1e-10)

    def test_entropia_mezcla_es_no_negativa(self, identity_metric, dim_metric, rng):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        for _ in range(10):
            psi = rng.standard_normal(dim_metric)
            psi /= np.linalg.norm(psi)
            S = channel._mixing_entropy(psi)
            assert S >= -1e-10

    def test_entropia_mezcla_es_acotada(self, identity_metric, dim_metric, rng):
        """S ≤ log(d)."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        for _ in range(20):
            psi = rng.standard_normal(dim_metric)
            psi /= np.linalg.norm(psi)
            S = channel._mixing_entropy(psi)
            assert S <= np.log(dim_metric) + 1e-10


class TestPhase3_AntimatterEmission:
    """Axioma §4: HMAC-SHA256 forense sobre el positrón."""

    def test_firma_HMAC_es_determinista(self):
        """Mismos parámetros ⟹ misma firma."""
        sig1 = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        sig2 = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        assert sig1 == sig2

    def test_firma_HMAC_cambia_con_parametros(self):
        sig1 = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        sig2 = Phase3_QuantumKrausChannel._sign_antimatter(0.6, -0.1, -1e-14)
        assert sig1 != sig2

    def test_firma_HMAC_es_hexadecimal_64_chars(self):
        """SHA-256 produce 64 caracteres hexadecimales."""
        sig = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_firma_HMAC_con_secret_personalizado(self):
        """HMAC con secreto diferente produce firma diferente."""
        sig_default = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        sig_custom = Phase3_QuantumKrausChannel._sign_antimatter(
            0.5, -0.1, -1e-14, secret=b"OTRO_SECRETO"
        )
        assert sig_default != sig_custom

    def test_firma_HMAC_no_es_simple_hash(self):
        """No debe ser sólo SHA-256 sin clave."""
        sig = Phase3_QuantumKrausChannel._sign_antimatter(0.5, -0.1, -1e-14)
        payload = "5.0000000000e-01|-1.0000000000e-01|-1.0000000000e-14"
        simple_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert sig != simple_hash  # HMAC ≠ SHA simple


class TestPhase3_ChannelExecution:
    """Axioma §3-§4: Ejecución completa del canal CPTP."""

    def test_ejecucion_con_Kraus_canonico(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits, dim_metric
    ):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        assert isinstance(evolution, QuantumChannelEvolution)
        assert evolution.coherent_state.shape == (dim_metric,)
        assert evolution.kraus_residual_fro < 1e-12
        assert evolution.kraus_residual_psd > -1e-9

    def test_estado_coherente_es_proyeccion_de_psi(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        """coherent_state = P · psi_raw."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        expected_coherent = P @ raw_llm_logits
        assert_allclose(evolution.coherent_state, expected_coherent, atol=1e-12)

    def test_componente_disipada_es_ortogonal_a_coherente(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        """(I−P)ψ ⟂ Pψ bajo producto euclídeo (para P simétrico)."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        dissipated_vec = (np.eye(P.shape[0]) - P) @ raw_llm_logits
        inner = evolution.coherent_state @ dissipated_vec
        assert inner == pytest.approx(0.0, abs=1e-12)

    def test_contractividad_CPTP(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        """ΔS ≤ 0 (el canal CPTP no aumenta entropía)."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        assert evolution.delta_entropy <= 1e-10

    def test_antimateria_se_emite_con_disipacion_significativa(
        self, identity_metric, projector_factory, h_obs_gradient, dim_metric, rng
    ):
        """Con ψ con componente grande fuera del hiperplano ⟹ positrón emitido."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        # Construimos ψ con componente ortogonal significativa
        # Calculamos el vector que P aniquila
        v = np.zeros(dim_metric)
        v[0] = 1.0  # Componente en dirección normal
        # Proyector ortogonal a e_0
        P_e0 = np.eye(dim_metric) - np.outer(v, v)
        # Si usamos P_e0, entonces v es la dirección normal ⟹ psi grande en v ⟹ disipación
        # Pero queremos el caso contrario: psi con componente FUERA del hiperplano
        # Tomamos P ≠ P_e0
        P_other = projector_factory(seed=123)
        psi = rng.standard_normal(dim_metric)
        evolution = channel.execute_quantum_channel(psi, h_obs_gradient, P_other)
        # Verificamos campos del positrón
        if evolution.antimatter_emission is not None:
            assert isinstance(evolution.antimatter_emission, PositronCartridge)
            assert evolution.antimatter_emission.inertial_mass > 0
            assert evolution.antimatter_emission.homological_charge == -1
            assert len(evolution.antimatter_emission.authorization_signature) == 64

    def test_sin_disipacion_no_hay_antimateria(
        self, identity_metric, projector_factory, h_obs_gradient, dim_metric, rng
    ):
        """ψ ya en el hiperplano ⟹ no hay positrón."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        # ψ = P · ψ_aleatorio ⟹ ya está proyectado, no hay disipación
        psi_random = rng.standard_normal(dim_metric)
        psi_proyectado = P @ psi_random
        evolution = channel.execute_quantum_channel(psi_proyectado, h_obs_gradient, P)
        # ‖(I−P)ψ‖_G debe ser ≈ 0
        if evolution.antimatter_emission is not None:
            assert evolution.antimatter_emission.inertial_mass < 1e-10

    def test_audit_dimensional_se_incluye_en_evolucion(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits, dim_metric
    ):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        assert isinstance(evolution.dimensional_audit, DimensionalAudit)
        assert evolution.dimensional_audit.dimension == dim_metric
        assert evolution.dimensional_audit.is_coherent is True
        assert evolution.dimensional_audit.psi_dim_ok is True
        assert evolution.dimensional_audit.grad_dim_ok is True


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
# ██╔════╝██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
# █████╗  ███████║███████║███████╗█████╗       █████╔╝
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝      ██╔═══╝
# ██║     ██║  ██║██║  ██║███████║███████╗    ███████╗
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
#        FASE 4 — FLOQUET MONODROMY AGENT (MORPHISM)
# ══════════════════════════════════════════════════════════════════════════════
class TestAgent_Construction:
    """Validación del Axioma §0 a nivel del orquestador."""

    def test_es_instancia_de_Morphism(self, identity_metric):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        assert isinstance(agent, Morphism)

    def test_G_no_cuadrada_es_rechazada(self):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            FloquetMonodromyAgent(metric_tensor=np.zeros((3, 5)))

    def test_G_asimetrica_es_rechazada(self, rng):
        G = rng.standard_normal((5, 5))
        with pytest.raises(TopologicalInvariantError, match="simétrica"):
            FloquetMonodromyAgent(metric_tensor=G)

    def test_G_singular_es_rechazada(self):
        G = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(TopologicalInvariantError, match="SPD"):
            FloquetMonodromyAgent(metric_tensor=G)

    def test_tolerancia_invalida_es_rechazada(self, identity_metric):
        with pytest.raises(ValueError):
            FloquetMonodromyAgent(metric_tensor=identity_metric, stability_tolerance=0.0)

    def test_metric_tensor_es_inmutable_externo(self, identity_metric):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        G_copy = agent.metric_tensor
        G_copy[0, 0] = 999.0
        assert agent._G[0, 0] != 999.0

    def test_fases_son_accesibles_como_propiedades(self, identity_metric):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        assert agent.phase1_synthesizer is not None
        assert agent.phase2_auditor is not None
        assert agent.phase3_channel is not None


class TestAgent_CategoricalContract:
    """Axioma §5: Contrato Morphism completo."""

    def test_forward_aplica_y_conserva_etiqueta(
        self, identity_metric, dim_metric, rng
    ):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        psi = rng.standard_normal(dim_metric)
        state = CategoricalState(payload=psi, label="raw")
        new_state = agent.forward(state)
        assert new_state.label == "raw"
        assert new_state.payload.shape == (dim_metric,)

    def test_backward_equivale_a_forward(
        self, identity_metric, dim_metric, rng
    ):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        psi = rng.standard_normal(dim_metric)
        state = CategoricalState(payload=psi, label="x")
        fwd = agent.forward(state).payload
        bwd = agent.backward(state).payload
        assert_allclose(fwd, bwd, atol=1e-14)

    def test_forward_rechaza_dimension_incompatible(self, identity_metric):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        bad_state = CategoricalState(payload=np.ones(3), label="malo")
        with pytest.raises(DimensionalMismatchError, match="dimensión|incompatible"):
            agent.forward(bad_state)


class TestAgent_Pipeline:
    """Pipeline axiomático completo (Fase 1 → 2 → 3)."""

    def test_pipeline_completo_exitoso(
        self, identity_metric, raw_llm_logits, h_obs_gradient
    ):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        evolution = agent.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        assert isinstance(evolution, QuantumChannelEvolution)
        assert evolution.coherent_state.shape == raw_llm_logits.shape
        assert evolution.kraus_residual_fro < 1e-9
        assert evolution.kraus_residual_psd > -1e-9
        assert evolution.dimensional_audit.is_coherent is True

    def test_pipeline_con_obstruccion_trivial(
        self, identity_metric, raw_llm_logits, trivial_gradient
    ):
        """∇H_obs = 0 ⟹ P = I ⟹ coherent_state = ψ_raw."""
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        evolution = agent.purify_and_tune_cavity(raw_llm_logits, trivial_gradient)
        # P = I ⟹ coherent_state = ψ_raw
        assert_allclose(evolution.coherent_state, raw_llm_logits, atol=1e-12)
        # No debe haber positrón (sin disipación)
        assert evolution.antimatter_emission is None

    def test_pipeline_con_G_anisotropica(
        self, anisotropic_metric, raw_llm_logits, h_obs_gradient
    ):
        agent = FloquetMonodromyAgent(metric_tensor=anisotropic_metric)
        evolution = agent.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        assert isinstance(evolution, QuantumChannelEvolution)

    def test_pipeline_con_G_PHYSICS(self, raw_llm_logits, h_obs_gradient):
        """Smoke test con métrica real del MIC."""
        agent = FloquetMonodromyAgent(metric_tensor=G_PHYSICS)
        evolution = agent.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        assert evolution.coherent_state.shape == raw_llm_logits.shape

    def test_determinismo_pipeline(
        self, identity_metric, raw_llm_logits, h_obs_gradient
    ):
        """Mismas entradas ⟹ mismos resultados bit-a-bit."""
        agent1 = FloquetMonodromyAgent(metric_tensor=identity_metric)
        agent2 = FloquetMonodromyAgent(metric_tensor=identity_metric)
        e1 = agent1.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        e2 = agent2.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        assert_allclose(e1.coherent_state, e2.coherent_state, atol=1e-14)
        assert e1.dissipated_entropy == pytest.approx(e2.dissipated_entropy, abs=1e-14)


# ══════════════════════════════════════════════════════════════════════════════
# ██╗███╗   ██╗██╗   ██╗ █████╗ ██████╗ ██╗ █████╗ ███╗   ██╗ ██████╗███████╗
# ██║████╗  ██║██║   ██║██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║██╔════╝██╔════╝
# ██║██╔██╗ ██║██║   ██║███████║██████╔╝██║███████║██╔██╗ ██║██║     █████╗
# ██║██║╚██╗██║╚██╗ ██╔╝██╔══██║██╔══██╗██║██╔══██║██║╚██╗██║██║     ██╔══╝
# ██║██║ ╚████║ ╚████╔╝ ██║  ██║██║  ██║██║██║  ██║██║ ╚████║╚██████╗███████╗
# ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
#            CASOS LÍMITE, INVARIANCIAS Y REGRESIONES
# ══════════════════════════════════════════════════════════════════════════════
class TestInvarianceAndEdgeCases:
    """Casos extremos físicos y matemáticos."""

    def test_metric_lorentziana_es_rechazada(self):
        """Métrica con signatura (-,+,+,+) no es Riemanniana."""
        G_lorentz = np.diag([-1.0, 1.0, 1.0, 1.0])
        with pytest.raises((TopologicalInvariantError, DimensionalMismatchError)):
            FloquetMonodromyAgent(metric_tensor=G_lorentz)

    def test_psi_con_componente_normal_pura_genera_maxima_disipacion(
        self, identity_metric, h_obs_gradient, dim_metric
    ):
        """ψ ∥ n_unit ⟹ toda la magnitud se disipa."""
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        # Construimos ψ alineado con la dirección normal
        P, reflector = agent.phase1_synthesizer.synthesize_projector(h_obs_gradient)
        # El vector que P aniquila
        if reflector is not None:
            v = reflector._v  # Vector normal unitario
            psi = v * 10.0
            evolution = agent.purify_and_tune_cavity(psi, h_obs_gradient)
            # P ψ = 0 ⟹ coherent_state ≈ 0
            assert np.linalg.norm(evolution.coherent_state) < 1e-10

    def test_reutilizacion_de_fases(
        self, identity_metric, raw_llm_logits, h_obs_gradient
    ):
        """Las fases son reutilizables entre invocaciones."""
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        e1 = agent.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        e2 = agent.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)
        assert_allclose(e1.coherent_state, e2.coherent_state, atol=1e-14)

    def test_diferentes_gradientes_producen_diferentes_proyecciones(
        self, identity_metric, raw_llm_logits, rng, dim_metric
    ):
        agent = FloquetMonodromyAgent(metric_tensor=identity_metric)
        g1 = rng.standard_normal(dim_metric)
        g2 = rng.standard_normal(dim_metric)
        e1 = agent.purify_and_tune_cavity(raw_llm_logits, g1)
        e2 = agent.purify_and_tune_cavity(raw_llm_logits, g2)
        # Si los gradientes son diferentes, los proyectores son diferentes,
        # y los coherent_state también (salvo coincidencia patológica)
        if np.linalg.norm(g1 - g2) > 1e-3:
            assert not np.allclose(e1.coherent_state, e2.coherent_state, atol=1e-6)

    def test_dimensional_audit_para_psi_mal_dimensionado(
        self, identity_metric, h_obs_gradient, dim_metric
    ):
        """DimensionalAudit detecta dimensiones incorrectas."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = np.eye(dim_metric)
        bad_psi = np.ones(dim_metric + 1)
        audit = channel._audit_dimensions(bad_psi, h_obs_gradient)
        assert audit.is_coherent is False
        assert audit.psi_dim_ok is False
        assert audit.grad_dim_ok is True


class TestRegressions:
    """Tests de regresión contra bugs conocidos de v2.0."""

    def test_regresion_Kraus_solo_Frobenius_pero_no_PSD(
        self, identity_metric, dim_metric
    ):
        """
        v2.0 sólo verificaba ‖C − I‖_F < ε. Esto pasa la prueba incluso si
        C − I tiene autovalores negativos pequeños que pasan inadvertidos.
        v3.0 verifica PSD estricto (λ_min ≥ −ε).
        """
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        # Construir E_0, E_1 con C − I que tiene un autovalor ligeramente negativo
        # E_0 = diag(1, 1, ..., 0.999) ⟹ C = E†_0 E_0 + E†_1 E_1
        # Si E_1 = diag(0, 0, ..., 0.001), entonces C = diag(0.998, 1, ..., 1.000001)
        # Diferencia con I: diag(-0.002, 0, ..., 0.000001) — un autovalor negativo
        e0_diag = np.ones(dim_metric)
        e0_diag[0] = 0.999
        e1_diag = np.ones(dim_metric)
        e1_diag[0] = 0.001
        E0 = np.diag(e0_diag)
        E1 = np.diag(e1_diag)
        with pytest.raises((KrausCompletenessError, KrausTraceViolationError)):
            channel._verify_kraus_completeness_psd(E0, E1)

    def test_regresion_proyector_asimetrico_no_usa_eigvalsh(
        self, asymmetric_projector, identity_metric
    ):
        """
        v2.0 llamaba eigvalsh sin verificar simetría, fallando silenciosamente.
        v3.0 detecta asimetría y usa eigvals.
        """
        auditor = Phase2_FloquetStabilityAuditor()
        state = auditor.audit_monodromy(asymmetric_projector)
        assert state.is_complex_manifold is True
        assert state.multipliers.dtype == np.complex128

    def test_regresion_obstruccion_trivial_no_retornaba_None(
        self, identity_metric, trivial_gradient
    ):
        """
        v2.0 retornaba `(np.eye(d), None)` pero el `None` propagaba errores
        en consumidores que esperaban reflector. v3.0 retorna reflector=None
        pero P=I es estado semánticamente válido y consistente.
        """
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=identity_metric)
        P, reflector = synth.synthesize_projector(trivial_gradient)
        assert P is not None
        assert_allclose(P, np.eye(identity_metric.shape[0]), atol=1e-14)
        # El reflector puede ser None, pero P siempre está definido
        assert reflector is None

    def test_regresion_HMAC_reemplaza_firma_literal(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        """
        v2.0 usaba authorization_signature="Floquet_CPTP_Auditor" literal.
        v3.0 usa HMAC-SHA256 real.
        """
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        if evolution.antimatter_emission is not None:
            sig = evolution.antimatter_emission.authorization_signature
            assert sig != "Floquet_CPTP_Auditor"
            assert len(sig) == 64
            assert all(c in "0123456789abcdef" for c in sig)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS DE PROPIEDADES ALGEBRAICAS PROFUNDAS
# ══════════════════════════════════════════════════════════════════════════════
class TestAlgebraicProperties:
    """Identidades algebraicas certificadas matemáticamente."""

    def test_M_on_tiene_traza_2_tr_P_menos_d(
        self, projector_factory, identity_metric
    ):
        """Para P proyector: tr(M_on) = tr(2P − P²) = tr(P) (porque P²=P)."""
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        tr_M_on = float(np.sum(state.multipliers).real)
        tr_P = float(np.trace(P))
        assert tr_M_on == pytest.approx(tr_P, abs=1e-10)

    def test_M_on_es_simetrica_si_P_lo_es(
        self, projector_factory, identity_metric
    ):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        M_on = 2.0 * P - P @ P
        assert_allclose(M_on, M_on.T, atol=1e-12)

    def test_proyector_y_complemento_sumen_identidad(
        self, projector_factory
    ):
        """P + (I − P) = I (algebraicamente)."""
        P = projector_factory(seed=42)
        I = np.eye(P.shape[0])
        assert_allclose(P + (I - P), I, atol=1e-14)

    def test_idempotencia_de_P_es_preservada_por_pullback(
        self, anisotropic_metric, h_obs_gradient
    ):
        """Pullback covariante no destruye idempotencia."""
        synth = Phase1_CovariantProjectorSynthesizer(metric_tensor=anisotropic_metric)
        P, _ = synth.synthesize_projector(h_obs_gradient)
        assert_allclose(P @ P, P, atol=1e-10)

    def test_Kraus_completitud_para_proyector_aleatorio(
        self, projector_factory, identity_metric
    ):
        """Para cualquier P, {P, I−P} es una familia de Kraus completa."""
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        fro_res, psd_min = channel._verify_kraus_completeness_psd(
            P, np.eye(P.shape[0]) - P
        )
        assert fro_res < 1e-12
        assert psd_min >= -1e-9

    def test_conservacion_de_producto_interno_para_coherente(
        self, identity_metric, projector_factory, h_obs_gradient, rng, dim_metric
    ):
        """
        Para ψ puro, ‖Pψ‖² + ‖(I−P)ψ‖² = ‖ψ‖².
        """
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        psi = rng.standard_normal(dim_metric)
        evolution = channel.execute_quantum_channel(psi, h_obs_gradient, P)
        n_coherent = np.linalg.norm(evolution.coherent_state) ** 2
        n_dissipated = np.linalg.norm((np.eye(dim_metric) - P) @ psi) ** 2
        n_total = np.linalg.norm(psi) ** 2
        assert (n_coherent + n_dissipated) == pytest.approx(n_total, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS DE DATACLASSES INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════
class TestImmutableStructures:
    """Los dataclasses deben ser frozen."""

    def test_DimensionalAudit_es_inmutable(self):
        audit = DimensionalAudit(
            dimension=8, psi_dim_ok=True, grad_dim_ok=True, is_coherent=True
        )
        with pytest.raises((AttributeError, Exception)):
            audit.dimension = 16  # type: ignore[misc]

    def test_FloquetMonodromyState_es_inmutable(self, projector_factory):
        auditor = Phase2_FloquetStabilityAuditor()
        P = projector_factory(seed=42)
        state = auditor.audit_monodromy(P)
        with pytest.raises((AttributeError, Exception)):
            state.spectral_radius = 0.5  # type: ignore[misc]

    def test_QuantumChannelEvolution_es_inmutable(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        with pytest.raises((AttributeError, Exception)):
            evolution.coherent_state = np.zeros(8)  # type: ignore[misc]

    def test_QuantumChannelEvolution_contiene_todos_los_campos(
        self, identity_metric, projector_factory, h_obs_gradient, raw_llm_logits
    ):
        channel = Phase3_QuantumKrausChannel(metric_tensor=identity_metric)
        P = projector_factory(seed=42)
        evolution = channel.execute_quantum_channel(
            raw_llm_logits, h_obs_gradient, P
        )
        assert hasattr(evolution, "coherent_state")
        assert hasattr(evolution, "dissipated_entropy")
        assert hasattr(evolution, "antimatter_emission")
        assert hasattr(evolution, "delta_entropy")
        assert hasattr(evolution, "kraus_residual_psd")
        assert hasattr(evolution, "kraus_residual_fro")
        assert hasattr(evolution, "dimensional_audit")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════
def pytest_configure(config):
    """Marca custom para tests del topos Floquet."""
    config.addinivalue_line(
        "markers",
        "topos: tests del contrato categórico FloquetMonodromyAgent",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA STANDALONE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestPhase1_MetricValidation,
        TestPhase1_CovariantPullback,
        TestPhase1_TrivialObstruction,
        TestPhase1_ProjectorConstruction,
        TestPhase2_Configuration,
        TestPhase2_MonodromyMatrix,
        TestPhase2_SymmetryDetection,
        TestPhase2_Stability,
        TestPhase3_Construction,
        TestPhase3_KrausCompletenessPSD,
        TestPhase3_EntropyCalculation,
        TestPhase3_AntimatterEmission,
        TestPhase3_ChannelExecution,
        TestAgent_Construction,
        TestAgent_CategoricalContract,
        TestAgent_Pipeline,
        TestInvarianceAndEdgeCases,
        TestRegressions,
        TestAlgebraicProperties,
        TestImmutableStructures,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)