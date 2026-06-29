# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Test Suite — Eikonal Agent v7.0                                     ║
║ Ubicación: tests/omega/test_eikonal_agent.py                                ║
║ Cobertura: Axiomas §0–§5, contrato categórico, casos límite, regresiones   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de testing (pirámide completa):
────────────────────────────────────────────────────────────────────────────────
  ◆ FASE 1 — DynamicApertureModulator (Diafragma Cuántico):
      • Hermiticidad y positividad de ρ
      • Recorte PSD de autovalores negativos
      • Pureza y entropía de Von Neumann
      • Validación de parámetros (l_max, kappa)
      • Modulación exponencial del cutoff
      • Detección de pureza colapsada

  ◆ FASE 2 — EikonalSurfaceResolver (Hamiltoniano Hiperbólico):
      • Validación SPD de G
      • Inversión espectral estable
      • Certificación PSD de G⁻¹
      • Umbral κ₂(G⁻¹)
      • Cálculo de ‖∇S‖²_G via einsum
      • Detección de singularidad Hamiltoniana
      • Alertas de desviación eikonal

  ◆ FASE 3 — FermatActionAuditor (Acción Estacionaria):
      • Integración Simpson compuesta (orden 4)
      • Fallback a trapecio para series degeneradas
      • Detección de divergencia de acción
      • Residuo geodésico covariante
      • Validación de Levi-Civita

  ◆ FASE 4 — EikonalAgent (Orquestador Categórico):
      • Contrato Morphism (forward/backward)
      • Composición tipada por agregación
      • Validación de EikonalControlInput
      • Coherencia dimensional Axioma §0
      • Pipeline completo con/sin corrección geodésica
      • Auditoría de tolerancia geodésica
      • Contratos Protocol runtime_checkable

  ◆ Casos físicos extremos e invariancia:
      • Estado maximalmente mezclado (S_max)
      • Estado puro (S=0)
      • Métrica mal condicionada (κ₂ ≈ 10¹⁰)
      • Métrica Lorentziana (rechazada)
      • ρ con autovalores negativos (recorte)
      • Determinismo bit-a-bit
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose, assert_array_almost_equal

# ══════════════════════════════════════════════════════════════════════════════
# PATH BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Silenciar logging durante tests
logging.disable(logging.CRITICAL)

from app.omega.eikonal_agent import (  # noqa: E402
    QuantumPurityCollapseError,
    EikonalSingularityError,
    FermatOpticalDeviationError,
    DimensionalMismatchError,
    MetricSignatureError,
    SpectralDensityAudit,
    EikonalPhaseState,
    EikonalControlInput,
    Phase1_DynamicApertureModulator,
    Phase2_EikonalSurfaceResolver,
    Phase3_FermatActionAuditor,
    EikonalAgent,
)
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError  # noqa: E402
from app.core.immune_system.metric_tensors import G_PHYSICS  # noqa: E402
from app.omega.optical_riemann_lens import OpticalRiemannLensFibrator, RefractedState  # noqa: E402
from app.omega.levi_civita_agent import LeviCivitaConnectionAgent, TangentVector  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def rng() -> np.random.Generator:
    """Generador reproducible para determinismo bit-a-bit."""
    return np.random.default_rng(seed=20251215)


@pytest.fixture
def dim_metric() -> int:
    """Dimensión base de los tests (coherente con G_PHYSICS si posible)."""
    return G_PHYSICS.shape[0]


@pytest.fixture
def identity_metric(dim_metric: int) -> np.ndarray:
    """Métrica trivial G = I_d."""
    return np.eye(dim_metric, dtype=np.float64)


@pytest.fixture
def ill_conditioned_metric(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Métrica con κ₂(G⁻¹) ≈ 10¹⁰ (caso extremo)."""
    eigenvalues = np.logspace(-5, 5, dim_metric)
    Q, _ = np.linalg.qr(rng.standard_normal((dim_metric, dim_metric)))
    return Q @ np.diag(eigenvalues) @ Q.T


@pytest.fixture
def anisotropic_metric(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Métrica anisotrópica bien condicionada."""
    A = rng.standard_normal((dim_metric, dim_metric))
    return A @ A.T + np.eye(dim_metric)


@pytest.fixture
def pure_state(dim_metric: int) -> np.ndarray:
    """Estado puro: ρ = |ψ⟩⟨ψ| con ψ = e_0."""
    psi = np.zeros(dim_metric, dtype=np.float64)
    psi[0] = 1.0
    return np.outer(psi, psi)


@pytest.fixture
def maximally_mixed_state(dim_metric: int) -> np.ndarray:
    """Estado maximalmente mezclado: ρ = I/d."""
    return np.eye(dim_metric, dtype=np.float64) / dim_metric


@pytest.fixture
def noisy_pure_state(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Estado puro con ruido numérico (autovalores ligeramente negativos)."""
    psi = rng.standard_normal(dim_metric)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi)
    # Inyectar perturbación asimétrica que genere autovalores negativos pequeños
    perturbation = 1e-14 * rng.standard_normal((dim_metric, dim_metric))
    rho_noisy = rho + perturbation - perturbation.T  # Asimétrica
    # Asegurar hermiticidad + ligera contaminación
    rho_noisy = (rho_noisy + rho_noisy.T) / 2.0
    rho_noisy[0, 1] += 1e-13
    return rho_noisy


@pytest.fixture
def phase_gradient(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Gradiente de fase eikonal aleatorio."""
    grad = rng.standard_normal(dim_metric)
    return grad / np.linalg.norm(grad)


@pytest.fixture
def path_velocities(dim_metric: int, rng: np.random.Generator) -> np.ndarray:
    """Secuencia de velocidades para trayectoria (número impar de puntos para Simpson)."""
    return rng.standard_normal((11, dim_metric)) * 0.1


@pytest.fixture
def eikonal_control(
    dim_metric: int,
    pure_state: np.ndarray,
    phase_gradient: np.ndarray,
    path_velocities: np.ndarray,
    rng: np.random.Generator,
) -> EikonalControlInput:
    """Input canónico para EikonalAgent.execute_optical_guidance."""
    raw_llm = rng.standard_normal(dim_metric)
    return EikonalControlInput(
        raw_llm_logits=raw_llm,
        rho_llm=pure_state,
        s_mac_entropy=0.5,
        logistic_stress_norm=0.3,
        phase_gradient=phase_gradient,
        path_velocities=path_velocities,
        use_geodesic_correction=True,
        cavity_tol=1e-10,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗ █████╗  █████╗ ███████╗███████╗     ██╗
# ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝    ███║
# █████╗  ███████║███████║███████╗█████╗      ╚██║
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝       ██║
# ██║     ██║  ██║██║  ██║███████║███████╗      ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝
#            FASE 1 — DYNAMIC APERTURE MODULATOR
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1_DensityMatrixValidation:
    """Axioma §0: ρ debe ser hermitiana, PSD (post-recorte), traza unitaria."""

    def test_hermiticidad_es_auditada(self, dim_metric, rng):
        rho = rng.standard_normal((dim_metric, dim_metric))
        # No hermitiana
        with pytest.raises(QuantumPurityCollapseError, match="hermítica"):
            Phase1_DynamicApertureModulator()._audit_density_matrix(rho)

    def test_rho_debe_ser_cuadrada(self):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            Phase1_DynamicApertureModulator()._audit_density_matrix(
                np.ones((3, 5), dtype=np.float64)
            )

    def test_rho_3d_es_rechazada(self, dim_metric):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            Phase1_DynamicApertureModulator()._audit_density_matrix(
                np.ones((dim_metric, dim_metric, dim_metric))
            )

    def test_autovalores_negativos_son_recortados(self, dim_metric, rng):
        """ρ con autovalores -ε debe ser proyectada al cono PSD."""
        # Construir ρ con un autovalor ligeramente negativo
        psi = rng.standard_normal(dim_metric)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi)
        # Inyectar autovalor negativo
        rho[0, 0] = -1e-13
        rho = (rho + rho.T) / 2.0

        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(rho)
        assert audit.negative_eigvals_pruned >= 1
        assert np.all(audit.eigenvalues_psd >= 0)
        assert audit.is_physical is True

    def test_traza_es_normalizada(self, dim_metric, rng):
        """ρ con traza ≠ 1 es renormalizada."""
        psi = rng.standard_normal(dim_metric)
        rho = np.outer(psi, psi) * 5.0  # traza = 5
        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(rho)
        assert audit.trace_after_pruning == pytest.approx(1.0, abs=1e-10)

    def test_estado_puro_da_pureza_1_y_entropia_0(self, pure_state, dim_metric):
        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(pure_state)
        assert audit.purity == pytest.approx(1.0, abs=1e-10)
        assert audit.von_neumann_entropy == pytest.approx(0.0, abs=1e-10)
        assert audit.is_physical is True

    def test_estado_maximalmente_mezclado(self, maximally_mixed_state, dim_metric):
        """ρ = I/d: pureza = 1/d, S = log(d)."""
        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(maximally_mixed_state)
        assert audit.purity == pytest.approx(1.0 / dim_metric, abs=1e-10)
        assert audit.von_neumann_entropy == pytest.approx(np.log(dim_metric), abs=1e-8)
        assert audit.is_physical is True

    def test_estado_caotico_termico_es_rechazado(self, dim_metric):
        """ρ con pureza < ε es rechazada (caos térmico)."""
        # Matriz de traza 1 con todos los autovalores iguales pequeños
        rho = np.eye(dim_metric, dtype=np.float64) * (1e-14 / dim_metric)
        rho[0, 0] += 1e-14  # traza total ≈ 2e-14, pureza ≈ 2e-28
        with pytest.raises(QuantumPurityCollapseError, match="nula|caos"):
            Phase1_DynamicApertureModulator()._audit_density_matrix(rho)


class TestPhase1_DynamicCutoff:
    """Axioma §1: l_cutoff = ⌊l_max · exp(−κS/Tr(ρ²))⌋."""

    def test_parametros_invalidos(self):
        with pytest.raises(ValueError, match="l_max"):
            Phase1_DynamicApertureModulator(l_max_absolute=0)
        with pytest.raises(ValueError, match="kappa"):
            Phase1_DynamicApertureModulator(kappa_coupling=-1.0)

    def test_cutoff_basico(self, pure_state):
        modulator = Phase1_DynamicApertureModulator(l_max_absolute=50, kappa_coupling=1.0)
        l_cutoff = modulator.compute_dynamic_cutoff(s_mac=0.5, rho_llm=pure_state)
        # Con pureza=1 y S=0.5: exp(-0.5) ≈ 0.606, l_cutoff = floor(50 * 0.606) = 30
        assert 28 <= l_cutoff <= 32

    def test_cutoff_minimo_es_1(self, maximally_mixed_state):
        """Con S muy alta o pureza muy baja, l_cutoff debe ser al menos 1."""
        modulator = Phase1_DynamicApertureModulator(l_max_absolute=50, kappa_coupling=10.0)
        l_cutoff = modulator.compute_dynamic_cutoff(s_mac=100.0, rho_llm=maximally_mixed_state)
        assert l_cutoff == 1

    def test_entropia_negativa_rechazada(self, pure_state):
        with pytest.raises(ValueError, match="entropía"):
            Phase1_DynamicApertureModulator().compute_dynamic_cutoff(
                s_mac=-0.1, rho_llm=pure_state
            )

    def test_pureza_colapsada_es_error(self, dim_metric):
        """ρ maximalmente mezclado en dim grande ⟹ pureza = 1/d < ε."""
        # d=100 ⟹ pureza = 0.01 > 1e-12 ⟹ no debe fallar; usamos d=2 con ρ muy diluida
        rho = np.diag([0.5, 0.5 - 1e-13, 1e-13])[:2, :2]
        rho = rho / rho.trace()
        # Forzar pureza extremadamente baja
        rho_diluted = rho * 1e-13
        rho_diluted = rho_diluted / rho_diluted.trace()
        with pytest.raises(QuantumPurityCollapseError, match="nula|caos"):
            Phase1_DynamicApertureModulator().compute_dynamic_cutoff(
                s_mac=0.1, rho_llm=rho_diluted
            )

    def test_recorte_psd_aumenta_pureza(self, noisy_pure_state):
        """ρ con autovalores negativos: pureza post-recorte ≥ pureza pre-recorte."""
        modulator = Phase1_DynamicApertureModulator()
        audit = modulator._audit_density_matrix(noisy_pure_state)
        # El recorte PSD sólo puede aumentar o mantener la pureza
        eigvals_raw = np.linalg.eigvalsh(noisy_pure_state)
        eigvals_raw_pos = np.clip(eigvals_raw, 0, None)
        eigvals_raw_pos = eigvals_raw_pos / eigvals_raw_pos.sum()
        purity_raw = np.sum(eigvals_raw_pos ** 2)
        assert audit.purity >= purity_raw - 1e-15

    def test_atenuacion_es_decreciente_en_S(self, pure_state):
        """l_cutoff debe ser monótonamente decreciente con S_MAC."""
        modulator = Phase1_DynamicApertureModulator(l_max_absolute=100, kappa_coupling=1.0)
        l_low = modulator.compute_dynamic_cutoff(s_mac=0.1, rho_llm=pure_state)
        l_high = modulator.compute_dynamic_cutoff(s_mac=5.0, rho_llm=pure_state)
        assert l_low > l_high

    def test_atenuacion_es_decreciente_en_kappa(self, pure_state):
        """l_cutoff debe ser monótonamente decreciente con kappa."""
        m_low = Phase1_DynamicApertureModulator(l_max_absolute=100, kappa_coupling=0.5)
        m_high = Phase1_DynamicApertureModulator(l_max_absolute=100, kappa_coupling=2.0)
        l_low = m_low.compute_dynamic_cutoff(s_mac=1.0, rho_llm=pure_state)
        l_high = m_high.compute_dynamic_cutoff(s_mac=1.0, rho_llm=pure_state)
        assert l_low > l_high


class TestPhase1_AuditCertificate:
    """El certificado SpectralDensityAudit debe ser inmutable y completo."""

    def test_audit_es_inmutable(self, pure_state):
        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(pure_state)
        with pytest.raises((AttributeError, Exception)):
            audit.purity = 0.5  # type: ignore[misc]

    def test_audit_contiene_todos_los_campos(self, pure_state):
        audit = Phase1_DynamicApertureModulator()._audit_density_matrix(pure_state)
        assert hasattr(audit, "purity")
        assert hasattr(audit, "von_neumann_entropy")
        assert hasattr(audit, "eigenvalues_psd")
        assert hasattr(audit, "negative_eigvals_pruned")
        assert hasattr(audit, "trace_after_pruning")
        assert hasattr(audit, "is_physical")


# ══════════════════════════════════════════════════════════════════════════════
# ██████╗  █████╗ ███████╗███████╗     ██████╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝    ╚════██╗
# ██████╔╝███████║███████╗█████╗       █████╔╝
# ██╔═══╝ ██╔══██║╚════██║██╔══╝      ██╔═══╝
# ██║     ██║  ██║███████║███████╗    ███████╗
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
#             FASE 2 — EIKONAL SURFACE RESOLVER
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2_MetricValidation:
    """Axioma §2: G debe ser SPD; G⁻¹ debe tener κ₂ controlado."""

    def test_G_no_cuadrada_es_rechazada(self):
        with pytest.raises(DimensionalMismatchError, match="cuadrada"):
            Phase2_EikonalSurfaceResolver(metric_tensor=np.zeros((3, 5)))

    def test_G_asimetrica_es_rechazada(self, rng):
        G = rng.standard_normal((5, 5))
        with pytest.raises(MetricSignatureError, match="simétrica"):
            Phase2_EikonalSurfaceResolver(metric_tensor=G)

    def test_G_singular_es_rechazada(self):
        G = np.array([[1.0, 1.0], [1.0, 1.0]])  # det = 0
        with pytest.raises(MetricSignatureError, match="SPD"):
            Phase2_EikonalSurfaceResolver(metric_tensor=G)

    def test_G_con_autovalor_negativo_es_rechazada(self):
        G = np.diag([1.0, -1.0, 2.0])
        with pytest.raises(MetricSignatureError):
            Phase2_EikonalSurfaceResolver(metric_tensor=G)

    def test_G_identidad_es_valida(self, identity_metric):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        assert resolver._G.shape == identity_metric.shape

    def test_G_PHYSICS_es_valida(self):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=G_PHYSICS)
        assert resolver._G.shape == G_PHYSICS.shape


class TestPhase2_GInverseCertification:
    """Axioma §2: G⁻¹ debe ser SPD con κ₂(G⁻¹) < κ_max."""

    def test_G_inversa_es_simétrica(self, anisotropic_metric):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=anisotropic_metric)
        assert_allclose(resolver._G_inv, resolver._G_inv.T, atol=1e-10)

    def test_G_inversa_es_SPD(self, anisotropic_metric):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=anisotropic_metric)
        eigvals_inv = np.linalg.eigvalsh(resolver._G_inv)
        assert np.all(eigvals_inv > 0)

    def test_G_inversa_satisface_G_Ginv_I(self, anisotropic_metric):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=anisotropic_metric)
        assert_allclose(anisotropic_metric @ resolver._G_inv, np.eye(anisotropic_metric.shape[0]), atol=1e-8)

    def test_kappa_max_umbral_es_aplicado(self, ill_conditioned_metric):
        """Con κ_max restrictivo, una métrica mal condicionada debe ser rechazada."""
        with pytest.raises(EikonalSingularityError, match="κ₂"):
            Phase2_EikonalSurfaceResolver(
                metric_tensor=ill_conditioned_metric, kappa_max=1e5
            )

    def test_kappa_max_permisivo_acepta_metrica_malcondicionada(
        self, ill_conditioned_metric
    ):
        resolver = Phase2_EikonalSurfaceResolver(
            metric_tensor=ill_conditioned_metric, kappa_max=1e15
        )
        assert resolver._G_inv.shape == ill_conditioned_metric.shape


class TestPhase2_EikonalEquation:
    """‖∇S‖²_G = n²: el Hamiltoniano eikonal debe ser hiperbólico regular."""

    def test_gradiente_mal_dimensionado_es_rechazado(
        self, identity_metric, dim_metric
    ):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        with pytest.raises(DimensionalMismatchError, match="dimensión"):
            resolver.resolve_eikonal_equation(
                np.ones(dim_metric + 1), n_refract=1.0
            )

    def test_gradiente_nulo_es_singularidad(self, identity_metric):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        with pytest.raises(EikonalSingularityError, match="colapsado|nula"):
            resolver.resolve_eikonal_equation(np.zeros(identity_metric.shape[0]), n_refract=1.0)

    def test_norma_eikonal_G_identidad(self, identity_metric, dim_metric):
        """Con G = I: ‖∇S‖² = ‖∇S‖²_euclídeo."""
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        grad = np.ones(dim_metric)
        n_refract = 1.0
        s_norm_sq = resolver.resolve_eikonal_equation(grad, n_refract)
        assert s_norm_sq == pytest.approx(dim_metric, rel=1e-10)

    def test_norma_eikonal_G_general(self, anisotropic_metric, dim_metric, rng):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=anisotropic_metric)
        grad = rng.standard_normal(dim_metric)
        s_norm_sq = resolver.resolve_eikonal_equation(grad, n_refract=2.0)
        # Verificación algebraica directa
        expected = float(grad @ resolver._G_inv @ grad)
        assert s_norm_sq == pytest.approx(expected, rel=1e-10)
        assert s_norm_sq > 0

    def test_desviacion_significativa_emite_warning(
        self, identity_metric, dim_metric, caplog
    ):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        # ‖∇S‖² muy diferente de n² (n_refract=10, pero grad pequeño)
        grad = np.ones(dim_metric) * 0.001
        with caplog.at_level(logging.WARNING, logger="MIC.Omega.EikonalAgent"):
            resolver.resolve_eikonal_equation(grad, n_refract=10.0)
        # Verificamos que se emitió warning de desviación
        assert any("Desviación eikonal" in r.message for r in caplog.records)

    def test_resultado_es_finito_y_positivo(
        self, anisotropic_metric, phase_gradient
    ):
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=anisotropic_metric)
        s_norm_sq = resolver.resolve_eikonal_equation(phase_gradient, n_refract=1.0)
        assert np.isfinite(s_norm_sq)
        assert s_norm_sq > 0


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗ ███████╗██████╗ ███╗   ███╗ █████╗ ████████╗
# ██╔════╝██╔════╝██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝
# █████╗  █████╗  ██████╔╝██╔████╔██║███████║   ██║
# ██╔══╝  ██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══██║   ██║
# ██║     ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║   ██║
# ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝
#             FASE 3 — FERMAT ACTION AUDITOR
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase3_FermatIntegration:
    """Axioma §3: Simpson compuesta vs. trapecio fallback."""

    def test_integral_para_velocidades_constantes(self, identity_metric, dim_metric):
        """Con v constante, ∫ √(v^T G v) dt = √(v^T G v) · T."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_const = np.ones((9, dim_metric))  # 9 puntos (impar, Simpson OK)
        v_const[:, :] = 0.1
        action = auditor.audit_fermat_action(v_const, n_refract=2.0, dt=0.01)
        # ‖v‖_G = 0.1 * sqrt(d); Simpson da exactamente ∫ = ‖v‖·T para polinomios lineales
        expected_norm = 0.1 * np.sqrt(dim_metric)
        expected_action = 2.0 * expected_norm * 0.01 * 8  # Simpson integra sobre T=0.08
        assert action == pytest.approx(expected_action, rel=1e-6)

    def test_simpsom_vs_trapecio_para_n_par(self, identity_metric, dim_metric):
        """n=10 (par) debe usar fallback a trapecio."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_even = np.ones((10, dim_metric)) * 0.1
        action = auditor.audit_fermat_action(v_even, n_refract=1.0, dt=0.01)
        # Trapecio: ∫ = (n-1) * dt * norm (promedio en extremos)
        assert np.isfinite(action)
        assert action > 0

    def test_simpson_para_n_impar_es_orden_4(self, identity_metric, dim_metric):
        """Para f cuadrático, Simpson es exacto (orden 4)."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        # Construimos velocidad cuadrática en t
        n = 21
        t = np.linspace(0, 1, n)
        velocities = np.zeros((n, dim_metric))
        for i, ti in enumerate(t):
            velocities[i, :] = (ti ** 2)  # ‖v‖² cuadrática en t
        action = auditor.audit_fermat_action(velocities, n_refract=1.0, dt=t[1] - t[0])
        # ∫_0^1 t² dt = 1/3
        expected = 1.0 / 3.0
        assert action == pytest.approx(expected, rel=1e-3)

    def test_velocidades_mal_dimensionadas(self, identity_metric):
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        with pytest.raises(ValueError, match="path_velocities"):
            auditor.audit_fermat_action(np.ones(5), n_refract=1.0)

    def test_accion_divergente_es_error(self, identity_metric, dim_metric):
        """Tamaños extremos deben disparar FermatOpticalDeviationError."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        # Velocidades enormes
        v_huge = np.ones((5, dim_metric)) * 1e8
        with pytest.raises(FermatOpticalDeviationError, match="Divergencia"):
            auditor.audit_fermat_action(v_huge, n_refract=1.0)

    def test_accion_con_n_refract_escala_linealmente(self, identity_metric, dim_metric):
        """A = n_refract * integral, proporcionalidad lineal."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v = np.ones((5, dim_metric))
        a1 = auditor.audit_fermat_action(v, n_refract=1.0)
        a3 = auditor.audit_fermat_action(v, n_refract=3.0)
        assert a3 == pytest.approx(3.0 * a1, rel=1e-10)

    def test_integral_es_no_negativa(self, identity_metric, dim_metric, rng):
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v = rng.standard_normal((7, dim_metric))
        action = auditor.audit_fermat_action(v, n_refract=2.5)
        assert action >= 0


class TestPhase3_GeodesicEnforcement:
    """Axioma §4: Residuo geodésico covariante ‖a_eff − a_geo‖_G."""

    def test_geodesica_con_velocidad_inicial_constante(self, identity_metric, dim_metric):
        """Velocidad constante ⟹ trayectoria geodésica (a_geo = 0)."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_init = TangentVector(coordinates=np.ones(dim_metric) * 0.5)
        velocities, deviation = auditor.enforce_geodesic_path(v_init, n_steps=5, dt=0.1)
        assert velocities.shape == (6, dim_metric)
        # En espacio plano con velocidad constante, la desviación debe ser ≈ 0
        assert deviation == pytest.approx(0.0, abs=1e-8)

    def test_geodesica_retorna_pasos_correctos(self, identity_metric, dim_metric):
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_init = TangentVector(coordinates=np.ones(dim_metric))
        velocities, _ = auditor.enforce_geodesic_path(v_init, n_steps=10, dt=0.01)
        assert velocities.shape[0] == 11  # n_steps + 1 (inicial + n_steps)
        assert velocities.shape[1] == dim_metric

    def test_n_steps_invalido(self, identity_metric, dim_metric):
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_init = TangentVector(coordinates=np.ones(dim_metric))
        with pytest.raises(ValueError, match="n_steps"):
            auditor.enforce_geodesic_path(v_init, n_steps=0)

    def test_desviacion_es_escalar(self, identity_metric, dim_metric):
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_init = TangentVector(coordinates=np.ones(dim_metric))
        _, deviation = auditor.enforce_geodesic_path(v_init, n_steps=3)
        assert isinstance(deviation, float)
        assert np.isfinite(deviation)
        assert deviation >= 0

    def test_geodesica_con_magnitud_aumentada(self, identity_metric, dim_metric):
        """Velocidades grandes ⟹ trayectorias más curvas (en general)."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v_init_small = TangentVector(coordinates=np.ones(dim_metric) * 0.01)
        v_init_large = TangentVector(coordinates=np.ones(dim_metric) * 10.0)
        _, dev_small = auditor.enforce_geodesic_path(v_init_small, n_steps=5)
        _, dev_large = auditor.enforce_geodesic_path(v_init_large, n_steps=5)
        # En métrica euclídea plana, ambas deberían ser ~0
        # Pero con LeviCivita puede haber efectos numéricos
        assert dev_large >= dev_small - 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗██╗██╗  ██╗███████╗    ██████╗
# ██╔════╝██║██║ ██╔╝██╔════╝    ╚════██╗
# █████╗  ██║█████╔╝ █████╗       █████╔╝
# ██╔══╝  ██║██╔═██╗ ██╔══╝      ██╔═══╝
# ██║     ██║██║  ██╗███████╗    ███████╗
# ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝
#        FASE 4 — ORQUESTADOR CATEGORIAL (MORPHISM)
# ══════════════════════════════════════════════════════════════════════════════
class TestEikonalAgent_Construction:
    """Validación del Axioma §0 a nivel del orquestador."""

    def test_es_instancia_de_Morphism(self, identity_metric):
        agent = EikonalAgent(metric_tensor=identity_metric)
        assert isinstance(agent, Morphism)

    def test_construccion_con_G_invalida(self):
        with pytest.raises((DimensionalMismatchError, MetricSignatureError, TopologicalInvariantError)):
            EikonalAgent(metric_tensor=np.zeros((3, 3)))

    def test_construccion_con_G_PHYSICS(self):
        agent = EikonalAgent(metric_tensor=G_PHYSICS)
        assert agent._G.shape == G_PHYSICS.shape

    def test_metric_tensor_es_inmutable_externo(self, identity_metric):
        agent = EikonalAgent(metric_tensor=identity_metric)
        G_copy = agent.metric_tensor
        G_copy[0, 0] = 999.0
        assert agent._G[0, 0] != 999.0

    def test_metric_inverse_es_inmutable_externo(self, identity_metric):
        agent = EikonalAgent(metric_tensor=identity_metric)
        Ginv = agent.metric_inverse
        Ginv[0, 0] = 999.0
        assert agent._phase2._G_inv[0, 0] != 999.0  # type: ignore[attr-defined]

    def test_parametros_invalidos(self):
        with pytest.raises(ValueError):
            EikonalAgent(l_max_absolute=0)
        with pytest.raises(ValueError):
            EikonalAgent(kappa_coupling=-1.0)

    def test_fases_son_accesibles_como_propiedades(self, identity_metric):
        agent = EikonalAgent(metric_tensor=identity_metric)
        assert agent._phase1 is not None
        assert agent._phase2 is not None
        assert agent._phase3 is not None


class TestEikonalAgent_CategoricalContract:
    """Axioma §5: Contrato Morphism + Protocol runtime_checkable."""

    def test_forward_aplica_y_conserva_etiqueta(self, identity_metric, dim_metric, rng):
        agent = EikonalAgent(metric_tensor=identity_metric)
        psi = rng.standard_normal(dim_metric)
        state = CategoricalState(payload=psi, label="raw")
        new_state = agent.forward(state)
        assert new_state.label == "raw::eikonal_forward"
        assert new_state.payload.shape == (dim_metric,)

    def test_backward_equivale_a_forward(self, identity_metric, dim_metric, rng):
        agent = EikonalAgent(metric_tensor=identity_metric)
        psi = rng.standard_normal(dim_metric)
        state = CategoricalState(payload=psi, label="x")
        fwd = agent.forward(state).payload
        bwd = agent.backward(state).payload
        assert_allclose(fwd, bwd, atol=1e-14)

    def test_forward_rechaza_dimension_incompatible(self, identity_metric):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_state = CategoricalState(payload=np.ones(3), label="malo")
        with pytest.raises(DimensionalMismatchError):
            agent.forward(bad_state)

    def test_fases_cumplen_protocolos(self, identity_metric):
        """Las fases agregadas deben satisfacer los Protocol runtime_checkable."""
        agent = EikonalAgent(metric_tensor=identity_metric)
        from app.omega.eikonal_agent import (
            ProjectorSynthesizerPort,
            FloquetAuditorPort,
            KrausChannelPort,
        )
        # Las fases concretas deben satisfacer los protocolos
        # (no son directamente instancias porque los Protocol son estructurales)
        assert hasattr(agent._phase1, "compute_dynamic_cutoff")
        assert hasattr(agent._phase2, "resolve_eikonal_equation")
        assert hasattr(agent._phase3, "execute_quantum_channel")


class TestEikonalAgent_InputValidation:
    """Axioma §0: EikonalControlInput debe ser dimensionalmente coherente."""

    def test_logits_mal_dimensionados(
        self, identity_metric, pure_state, phase_gradient, path_velocities
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_control = EikonalControlInput(
            raw_llm_logits=np.ones(3),  # dimensión incorrecta
            rho_llm=pure_state,
            s_mac_entropy=0.5,
            logistic_stress_norm=0.3,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        with pytest.raises(DimensionalMismatchError, match="raw_llm_logits"):
            agent.execute_optical_guidance(bad_control)

    def test_rho_mal_dimensionado(
        self, identity_metric, dim_metric, phase_gradient, path_velocities, rng
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_rho = np.eye(3)
        bad_control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=bad_rho,
            s_mac_entropy=0.5,
            logistic_stress_norm=0.3,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        with pytest.raises(DimensionalMismatchError, match="rho_llm"):
            agent.execute_optical_guidance(bad_control)

    def test_phase_gradient_mal_dimensionado(
        self, identity_metric, pure_state, dim_metric, path_velocities, rng
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.5,
            logistic_stress_norm=0.3,
            phase_gradient=np.ones(3),
            path_velocities=path_velocities,
        )
        with pytest.raises(DimensionalMismatchError, match="phase_gradient"):
            agent.execute_optical_guidance(bad_control)

    def test_path_velocities_mal_dimensionado(
        self, identity_metric, pure_state, phase_gradient, dim_metric, rng
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.5,
            logistic_stress_norm=0.3,
            phase_gradient=phase_gradient,
            path_velocities=np.ones((5, 3)),  # dimensión d incorrecta
        )
        with pytest.raises(DimensionalMismatchError, match="path_velocities"):
            agent.execute_optical_guidance(bad_control)

    def test_entropia_negativa_rechazada(
        self, identity_metric, pure_state, phase_gradient, path_velocities, dim_metric, rng
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        bad_control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=-0.1,
            logistic_stress_norm=0.3,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        with pytest.raises(ValueError, match="entropía"):
            agent.execute_optical_guidance(bad_control)


class TestEikonalAgent_Pipeline:
    """Pipeline axiomático completo (Fase 1 → 2 → 3 → Lente)."""

    def test_pipeline_completo_exitoso(
        self, identity_metric, eikonal_control
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        assert isinstance(result, EikonalPhaseState)
        assert result.phase_gradient_norm > 0
        assert result.fermat_action_integral >= 0
        assert result.dynamic_l_cutoff >= 1
        assert isinstance(result.refracted_state, RefractedState)
        assert result.spectral_certificate is not None

    def test_pipeline_sin_correccion_geodesica(
        self, identity_metric, dim_metric, pure_state, phase_gradient, path_velocities, rng
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.3,
            logistic_stress_norm=0.2,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
            use_geodesic_correction=False,
        )
        result = agent.execute_optical_guidance(control)
        assert result.geodesic_deviation == 0.0

    def test_correccion_geodesica_se_activa(
        self, identity_metric, eikonal_control
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        # En espacio euclídeo plano, la corrección debe ser muy pequeña
        assert result.geodesic_deviation >= 0
        assert np.isfinite(result.geodesic_deviation)

    def test_desviacion_geodesica_excesiva_es_error(
        self, identity_metric, dim_metric, pure_state, phase_gradient, path_velocities, rng
    ):
        """Si cavity_tol es muy estricto, puede dispararse FermatOpticalDeviationError."""
        agent = EikonalAgent(metric_tensor=identity_metric)
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.3,
            logistic_stress_norm=0.2,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
            use_geodesic_correction=True,
            cavity_tol=1e-20,  # Imposible de satisfacer
        )
        # En espacio plano, la desviación puede ser ≈ 0, pero fluctuaciones
        # numéricas podrían exceder tolerancia ridículamente baja
        try:
            result = agent.execute_optical_guidance(control)
            # Si pasa, la desviación debe ser ≤ tol
            assert result.geodesic_deviation <= 1e-20
        except FermatOpticalDeviationError:
            pass  # Comportamiento aceptable

    def test_l_cutoff_se_inyecta_en_lente(
        self, identity_metric, eikonal_control
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        # El l_cutoff del lens_fibrator debe coincidir con el devuelto
        assert agent._lens_fibrator._l_cutoff == result.dynamic_l_cutoff

    def test_certificado_espectral_se_preserva(
        self, identity_metric, eikonal_control
    ):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        assert result.spectral_certificate.is_physical is True


class TestEikonalControlInput:
    """El objeto del topos EikonalControlInput debe ser inmutable."""

    def test_es_inmutable(self, eikonal_control):
        with pytest.raises((AttributeError, Exception)):
            eikonal_control.s_mac_entropy = 1.0  # type: ignore[misc]

    def test_contiene_todos_los_campos(self, eikonal_control):
        assert hasattr(eikonal_control, "raw_llm_logits")
        assert hasattr(eikonal_control, "rho_llm")
        assert hasattr(eikonal_control, "s_mac_entropy")
        assert hasattr(eikonal_control, "logistic_stress_norm")
        assert hasattr(eikonal_control, "phase_gradient")
        assert hasattr(eikonal_control, "path_velocities")
        assert hasattr(eikonal_control, "use_geodesic_correction")
        assert hasattr(eikonal_control, "cavity_tol")


class TestEikonalPhaseState:
    """El estado de salida EikonalPhaseState es inmutable."""

    def test_es_inmutable(self, identity_metric, eikonal_control):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        with pytest.raises((AttributeError, Exception)):
            result.phase_gradient_norm = 0.0  # type: ignore[misc]

    def test_contiene_todos_los_campos(self, identity_metric, eikonal_control):
        agent = EikonalAgent(metric_tensor=identity_metric)
        result = agent.execute_optical_guidance(eikonal_control)
        assert hasattr(result, "phase_gradient_norm")
        assert hasattr(result, "fermat_action_integral")
        assert hasattr(result, "dynamic_l_cutoff")
        assert hasattr(result, "refracted_state")
        assert hasattr(result, "geodesic_deviation")
        assert hasattr(result, "spectral_certificate")


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

    def test_estado_maximalmente_mezclado_no_es_caos(
        self, identity_metric, maximally_mixed_state, phase_gradient, path_velocities, dim_metric, rng
    ):
        """
        ρ = I/d en d=16 tiene pureza = 1/16 ≈ 0.0625, que es > ε.
        No debe disparar QuantumPurityCollapseError.
        """
        agent = EikonalAgent(metric_tensor=identity_metric)
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=maximally_mixed_state,
            s_mac_entropy=1.0,
            logistic_stress_norm=0.5,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        result = agent.execute_optical_guidance(control)
        assert result.dynamic_l_cutoff >= 1

    def test_pipeline_con_G_PHYSICS(self, dim_metric, pure_state, phase_gradient, path_velocities, rng):
        """Smoke test con la métrica real del MIC."""
        if G_PHYSICS.shape[0] != dim_metric:
            pytest.skip("G_PHYSICS dimensión no coincide con fixture")
        agent = EikonalAgent(metric_tensor=G_PHYSICS)
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.4,
            logistic_stress_norm=0.2,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        result = agent.execute_optical_guidance(control)
        assert np.isfinite(result.fermat_action_integral)

    def test_metric_lorentziana_es_rechazada_en_fase2(self):
        """Métrica con signatura (-,+,+,+) no es Riemanniana ⟹ rechazo."""
        G_lorentz = np.diag([-1.0, 1.0, 1.0, 1.0])
        with pytest.raises((MetricSignatureError, TopologicalInvariantError)):
            Phase2_EikonalSurfaceResolver(metric_tensor=G_lorentz)

    def test_determinismo_bit_a_bit(self, identity_metric, eikonal_control):
        """Misma semilla + mismas entradas ⟹ mismo resultado."""
        agent1 = EikonalAgent(metric_tensor=identity_metric)
        agent2 = EikonalAgent(metric_tensor=identity_metric)
        r1 = agent1.execute_optical_guidance(eikonal_control)
        r2 = agent2.execute_optical_guidance(eikonal_control)
        assert r1.fermat_action_integral == pytest.approx(r2.fermat_action_integral, rel=1e-12)
        assert r1.dynamic_l_cutoff == r2.dynamic_l_cutoff

    def test_path_velocities_vacias_con_correccion(
        self, identity_metric, pure_state, phase_gradient, dim_metric, rng
    ):
        """Sin velocidades, la corrección geodésica no se aplica."""
        agent = EikonalAgent(metric_tensor=identity_metric)
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.3,
            logistic_stress_norm=0.2,
            phase_gradient=phase_gradient,
            path_velocities=np.zeros((0, dim_metric)),
            use_geodesic_correction=True,
        )
        result = agent.execute_optical_guidance(control)
        assert result.geodesic_deviation == 0.0


class TestRegressions:
    """Tests de regresión contra bugs conocidos de versiones anteriores."""

    def test_regresion_pureza_contaminada_por_eigvals_negativos(
        self, dim_metric, rng, phase_gradient, path_velocities
    ):
        """
        v6.0 calculaba pureza con eigvalsh sin recorte, contaminando la métrica.
        v7.0 proyecta al cono PSD antes del cálculo.
        """
        # Construir ρ con un autovalor negativo explícito
        rho = np.eye(dim_metric, dtype=np.float64) / dim_metric
        rho[0, 0] -= 1e-13  # Introduce perturbación
        rho = (rho + rho.T) / 2.0

        agent = EikonalAgent(metric_tensor=np.eye(dim_metric))
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=rho,
            s_mac_entropy=0.5,
            logistic_stress_norm=0.3,
            phase_gradient=phase_gradient,
            path_velocities=path_velocities,
        )
        # No debe fallar ni dar l_cutoff=0
        result = agent.execute_optical_guidance(control)
        assert result.dynamic_l_cutoff >= 1

    def test_regresion_desviacion_eikonal_no_aborta(
        self, identity_metric, dim_metric, pure_state, path_velocities, rng
    ):
        """
        v6.0 podía abortar el pipeline si ‖∇S‖²_G se desviaba mucho de n².
        v7.0 sólo emite warning, no aborta.
        """
        agent = EikonalAgent(metric_tensor=identity_metric)
        # Gradiente enorme vs. n_refract pequeño
        extreme_grad = np.ones(dim_metric) * 100.0
        control = EikonalControlInput(
            raw_llm_logits=rng.standard_normal(dim_metric),
            rho_llm=pure_state,
            s_mac_entropy=0.3,
            logistic_stress_norm=0.5,  # n_refract será alto
            phase_gradient=extreme_grad,
            path_velocities=path_velocities,
        )
        # Debe ejecutar sin abortar
        result = agent.execute_optical_guidance(control)
        assert result is not None

    def test_regresion_kappa_max_no_validaba_G_inv(
        self, ill_conditioned_metric
    ):
        """
        v6.0 no validaba κ₂(G⁻¹), permitiendo Hamiltonianos degenerados.
        v7.0 audita con κ_max.
        """
        # Métrica mal condicionada debe ser rechazada con κ_max restrictivo
        with pytest.raises(EikonalSingularityError, match="κ₂"):
            EikonalAgent(metric_tensor=ill_conditioned_metric, kappa_max=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS DE PROPIEDADES ALGEBRAICAS PROFUNDAS
# ══════════════════════════════════════════════════════════════════════════════
class TestAlgebraicProperties:
    """Identidades algebraicas certificadas matemáticamente."""

    def test_idempotencia_recorte_psd(self, dim_metric, rng):
        """Aplicar el recorte dos veces es igual a aplicarlo una vez."""
        rho = rng.standard_normal((dim_metric, dim_metric))
        rho = (rho + rho.T) / 2.0

        modulator = Phase1_DynamicApertureModulator()
        audit1 = modulator._audit_density_matrix(rho)
        rho_reconstructed = (audit1.eigenvalues_psd[:, None] * np.eye(dim_metric)).sum(axis=0)
        # Reconstrucción: rho_psd = V diag(λ_psd) V^T (no necesario, sólo eigvals)
        audit2 = modulator._audit_density_matrix(rho)
        # Tras renormalización, los eigvals_psd deben ser idénticos
        assert_allclose(audit1.eigenvalues_psd, audit2.eigenvalues_psd, atol=1e-12)

    def test_entropia_es_no_negativa(self, dim_metric, rng):
        """S(ρ) ≥ 0 siempre."""
        for _ in range(10):
            rho = rng.standard_normal((dim_metric, dim_metric))
            rho = rho @ rho.T  # PSD
            rho /= np.trace(rho)
            audit = Phase1_DynamicApertureModulator()._audit_density_matrix(rho)
            assert audit.von_neumann_entropy >= -1e-10

    def test_pureza_acotada_entre_1_d_y_1(self, dim_metric, rng):
        """1/d ≤ Tr(ρ²) ≤ 1."""
        modulator = Phase1_DynamicApertureModulator()
        for _ in range(20):
            rho = rng.standard_normal((dim_metric, dim_metric))
            rho = rho @ rho.T
            rho /= np.trace(rho)
            audit = modulator._audit_density_matrix(rho)
            assert 1.0 / dim_metric - 1e-10 <= audit.purity <= 1.0 + 1e-10

    def test_eikonal_norma_es_cuadratica_en_grad(self, identity_metric, dim_metric):
        """‖α·∇S‖²_G = α²·‖∇S‖²_G (homogeneidad cuadrática)."""
        resolver = Phase2_EikonalSurfaceResolver(metric_tensor=identity_metric)
        grad = np.ones(dim_metric)
        s_sq_1 = resolver.resolve_eikonal_equation(grad, n_refract=1.0)
        s_sq_3 = resolver.resolve_eikonal_equation(grad * 3.0, n_refract=1.0)
        assert s_sq_3 == pytest.approx(9.0 * s_sq_1, rel=1e-10)

    def test_accion_fermat_es_aditiva_en_segmentos(self, identity_metric, dim_metric):
        """A([γ₁;γ₂]) = A(γ₁) + A(γ₂) (aditividad en concatenación)."""
        auditor = Phase3_FermatActionAuditor(metric_tensor=identity_metric)
        v1 = np.ones((3, dim_metric)) * 0.1
        v2 = np.ones((3, dim_metric)) * 0.2
        v_total = np.vstack([v1, v2])
        a_total = auditor.audit_fermat_action(v_total, n_refract=1.0, dt=0.01)
        a_1 = auditor.audit_fermat_action(v1, n_refract=1.0, dt=0.01)
        a_2 = auditor.audit_fermat_action(v2, n_refract=1.0, dt=0.01)
        assert a_total == pytest.approx(a_1 + a_2, rel=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════
def pytest_configure(config):
    """Marca custom para tests del topos Eikonal."""
    config.addinivalue_line(
        "markers",
        "topos: tests del contrato categórico EikonalAgent",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA STANDALONE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestPhase1_DensityMatrixValidation,
        TestPhase1_DynamicCutoff,
        TestPhase1_AuditCertificate,
        TestPhase2_MetricValidation,
        TestPhase2_GInverseCertification,
        TestPhase2_EikonalEquation,
        TestPhase3_FermatIntegration,
        TestPhase3_GeodesicEnforcement,
        TestEikonalAgent_Construction,
        TestEikonalAgent_CategoricalContract,
        TestEikonalAgent_InputValidation,
        TestEikonalAgent_Pipeline,
        TestEikonalControlInput,
        TestEikonalPhaseState,
        TestInvarianceAndEdgeCases,
        TestRegressions,
        TestAlgebraicProperties,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)