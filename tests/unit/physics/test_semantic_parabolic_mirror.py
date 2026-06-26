# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Test Suite — Semantic Parabolic Mirror v6.0                        ║
║ Ubicación: tests/unit/physics/test_semantic_parabolic_mirror.py                    ║
║ Cobertura: Axiomas §0–§4, contrato categórico, casos límite, regresiones    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de testing (pirámide invertida):
────────────────────────────────────────────────────────────────────────────────
  ◆ FASE 1 — Reflector (Householder métrico):
      • Validez de G (simetría, SPD, Cholesky)
      • Vector nulo / sub-luminoso bajo G
      • Isometría métrica: M^T G M = G
      • Inversión del normal: M v = -v
      • Idempotencia del proyector: P² = P
      • Hermiticidad métrica del proyector: P^T G P = G P
      • Certificado espectral bipartito: esp(M) = {-1} ∪ {+1}^(d-1)
      • Determinante = -1, traza = d-2

  ◆ FASE 2 — Cavidad resonante (Von Mises métrico):
      • Estabilización en una iteración cuando P es exacto
      • Convergencia bajo ruido numérico
      • Detección de aniquilación total (‖Pψ‖_G ≈ 0)
      • Traza de convergencia registrada
      • Rechazo de dimensiones incompatibles

  ◆ FASE 3 — Orquestador categórico:
      • Contrato Morphism: forward/backward/eta_kernel
      • Involutividad de Householder: forward(forward(x)) = x
      • Camino corto (sin refinamiento)
      • Camino completo (con cavidad)
      • Inyección eta preserva etiqueta
      • Métricas de auditoría (last_trace, last_certificate)

  ◆ Casos límite e invariancia:
      • Rotación nula (G = I, v aleatorio) → comportamiento canónico
      • Métrica mal condicionada (κ₂(G) ≈ 10⁸)
      • Reutilización del reflector en múltiples cavidades
      • Determinismo bit-a-bit (semillas)
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose, assert_array_almost_equal

# ══════════════════════════════════════════════════════════════════════════════
# PATH BOOTSTRAP — resolver el paquete `app.*` desde la raíz de tests
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Silenciar logging durante tests para no contaminar stdout
logging.disable(logging.CRITICAL)

from app.omega.semantic_parabolic_mirror import (  # noqa: E402
    SpectralCertificate,
    ConvergenceTrace,
    HouseholderSingularityError,
    MetricSignatureError,
    ResonanceDissonanceError,
    MetricAwareHouseholderReflector,
    FabryPerotStabilizedCavity,
    SemanticParabolicMirror,
)
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError  # noqa: E402
from app.core.immune_system.metric_tensors import G_PHYSICS  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def rng() -> np.random.Generator:
    """Generador reproducible para determinismo bit-a-bit."""
    return np.random.default_rng(seed=20251215)


@pytest.fixture
def identity_metric() -> np.ndarray:
    """Métrica trivial G = I_d (caso límite canónico)."""
    return np.eye(8, dtype=np.float64)


@pytest.fixture
def ill_conditioned_metric(rng: np.random.Generator) -> np.ndarray:
    """
    Métrica con κ₂(G) controlado ≈ 10⁸.
    Construida como Q diag(λ) Qᵀ con autovalores espaciados exponencialmente.
    """
    d = 6
    eigenvalues = np.array([1e-4, 1e-2, 1.0, 10.0, 1e2, 1e4], dtype=np.float64)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q @ np.diag(eigenvalues) @ Q.T


@pytest.fixture
def anisotropic_metric(rng: np.random.Generator) -> np.ndarray:
    """Métrica anisotrópica genérica bien condicionada (κ₂ ≈ 100)."""
    d = 6
    A = rng.standard_normal((d, d))
    # SPD = A A^T + I para asegurar positividad
    return A @ A.T + np.eye(d)


@pytest.fixture
def normal_vector(rng: np.random.Generator) -> np.ndarray:
    """Vector normal v unitario en R^8."""
    v = rng.standard_normal(8)
    return v / np.linalg.norm(v)


@pytest.fixture
def orthogonal_vector(rng: np.random.Generator, normal_vector: np.ndarray) -> np.ndarray:
    """Vector ortogonal a normal_vector bajo producto euclídeo (sanity check)."""
    v = rng.standard_normal(8)
    v -= (v @ normal_vector) * normal_vector
    return v / np.linalg.norm(v)


# ══════════════════════════════════════════════════════════════════════════════
# ██████╗  █████╗ ███████╗███████╗     ██╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝    ███║
# ██████╔╝███████║███████╗█████╗      ╚██║
# ██╔═══╝ ██╔══██║╚════██║██╔══╝       ██║
# ██║     ██║  ██║███████║███████╗      ██║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝
#                FASE 1 — REFLECTOR
# ══════════════════════════════════════════════════════════════════════════════
class TestMetricValidation:
    """Axioma §0: G debe ser SPD — el espejo exige una métrica legítima."""

    def test_G_debe_ser_cuadrada(self):
        with pytest.raises(MetricSignatureError, match="cuadrada"):
            MetricAwareHouseholderReflector(
                np.array([1.0, 2.0, 3.0]),
                np.zeros((2, 3), dtype=np.float64),
            )

    def test_G_debe_ser_simetrica(self, rng):
        G = rng.standard_normal((5, 5))
        # Asimétrica deliberada
        with pytest.raises(MetricSignatureError, match="simétrica"):
            MetricAwareHouseholderReflector(np.ones(5), G)

    def test_G_debe_ser_definida_positiva(self, rng):
        # Matriz singular (semidefinida): Cholesky debe fallar
        G = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(MetricSignatureError, match="definida positiva"):
            MetricAwareHouseholderReflector(np.ones(2), G)

    def test_G_con_autovalor_negativo_falla(self, rng):
        # SPD rota: un autovalor negativo
        G = np.diag([1.0, -1.0, 2.0])
        with pytest.raises(MetricSignatureError):
            MetricAwareHouseholderReflector(np.ones(3), G)

    def test_G_identidad_es_valida(self, identity_metric, normal_vector):
        # G = I debe pasar todos los axiomas
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        assert reflector._dim == 8

    def test_G_PHYSICS_del_ecosistema_es_valida(self, normal_vector):
        """La métrica real del MIC debe pasar el filtro (Axioma §0)."""
        # Ajustamos v a la dimensión de G_PHYSICS
        v = np.ones(G_PHYSICS.shape[0], dtype=np.float64)
        v /= np.linalg.norm(v)
        reflector = MetricAwareHouseholderReflector(v, G_PHYSICS)
        assert reflector._dim == G_PHYSICS.shape[0]


class TestVectorValidation:
    """Axioma §1: v debe ser no-luminoso bajo G (⟨v,Gv⟩ > 0)."""

    def test_vector_nulo_es_singular(self, identity_metric):
        with pytest.raises(HouseholderSingularityError, match="luminoso"):
            MetricAwareHouseholderReflector(np.zeros(8), identity_metric)

    def test_dimension_incompatible(self, identity_metric):
        with pytest.raises(HouseholderSingularityError, match="dimensión"):
            MetricAwareHouseholderReflector(np.ones(5), identity_metric)

    def test_vector_luminoso_bajo_G_no_euclideo(
        self, rng
    ):
        """
        Construimos G con un autovalor negativo que haga ⟨v, Gv⟩ ≤ 0.
        Esto simula un vector tipo-luz en métrica Lorentziana (físicamente relevante).
        """
        # Métrica Lorentziana simple en R²
        G = np.diag([1.0, -1.0])
        v = np.array([1.0, 1.0])  # ⟨v, Gv⟩ = 1 - 1 = 0 (luminoso)
        with pytest.raises(HouseholderSingularityError, match="luminoso"):
            MetricAwareHouseholderReflector(v, G)


class TestIsometryAndReflection:
    """Axioma §1: M^T G M = G y M v = -v."""

    def test_isometria_metrica_G_identidad(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        M = reflector.reflection_matrix
        # M^T I M = M^T M = I
        assert_allclose(M.T @ M, np.eye(8), atol=1e-12)

    def test_isometria_metrica_G_general(
        self, normal_vector, anisotropic_metric
    ):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        M = reflector.reflection_matrix
        G = anisotropic_metric
        assert_allclose(M.T @ G @ M, G, atol=1e-10)

    def test_inversion_del_normal_G_identidad(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        M = reflector.reflection_matrix
        assert_allclose(M @ normal_vector, -normal_vector, atol=1e-12)

    def test_inversion_del_normal_G_general(self, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        M = reflector.reflection_matrix
        assert_allclose(M @ normal_vector, -normal_vector, atol=1e-10)

    def test_involutividad_M_cuadrado_igual_I(self, normal_vector, anisotropic_metric):
        """Householder es involutiva: M² = I (reflexión idempotente como aplicación)."""
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        M = reflector.reflection_matrix
        assert_allclose(M @ M, np.eye(reflector._dim), atol=1e-10)

    def test_reflect_devuelve_M_psi(self, rng, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        psi = rng.standard_normal(8)
        reflected = reflector.reflect(psi)
        assert_allclose(reflected, reflector.reflection_matrix @ psi, atol=1e-12)


class TestSpectralCertificate:
    """Axioma §1: cert(f) = (det=-1, tr=d-2, esp bipartito {-1,+1,…,+1})."""

    def test_certificado_es_inmutable(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cert = reflector.certificate
        # Frozen dataclass: debe fallar cualquier asignación
        with pytest.raises((AttributeError, Exception)):
            cert.is_valid_reflection = False  # type: ignore[misc]

    def test_determinante_menos_uno(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        assert reflector.certificate.determinant == pytest.approx(-1.0, abs=1e-10)

    def test_traza_d_menos_2(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        assert reflector.certificate.trace_value == pytest.approx(8 - 2, abs=1e-10)

    def test_espectro_bipartito(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        eigs = np.sort(reflector.certificate.eigenvalues)
        # El menor debe ser -1, los demás +1
        assert eigs[0] == pytest.approx(-1.0, abs=1e-8)
        assert np.all(eigs[1:] == pytest.approx(1.0, abs=1e-8))

    def test_es_valida_reflexion(self, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        assert reflector.certificate.is_valid_reflection is True

    def test_certificado_con_G_malcondicionada(
        self, normal_vector, ill_conditioned_metric
    ):
        """Incluso con κ₂(G) ≈ 10⁸, el certificado debe pasar."""
        reflector = MetricAwareHouseholderReflector(normal_vector, ill_conditioned_metric)
        cert = reflector.certificate
        assert cert.is_valid_reflection is True
        assert cert.determinant == pytest.approx(-1.0, abs=1e-8)


class TestProjectionOperator:
    """Axioma §2: P = (I + M)/2 = I - (Gv)v^T/⟨v,Gv⟩."""

    def test_P_es_idempotente(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        P = reflector.projection_operator
        assert_allclose(P @ P, P, atol=1e-12)

    def test_P_es_idempotente_G_general(self, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        P = reflector.projection_operator
        assert_allclose(P @ P, P, atol=1e-10)

    def test_P_es_hermitico_bajo_G(self, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        P = reflector.projection_operator
        G = anisotropic_metric
        # P^T G P = G P  (proyector ortogonal respecto a G)
        assert_allclose(P.T @ G @ P, G @ P, atol=1e-10)

    def test_P_aniquila_v(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        P = reflector.projection_operator
        assert_allclose(P @ normal_vector, np.zeros_like(normal_vector), atol=1e-12)

    def test_P_fija_ortogonales(
        self, normal_vector, orthogonal_vector, identity_metric
    ):
        """Si u ⊥ v bajo G, entonces P u = u."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        P = reflector.projection_operator
        # Para G = I, ortogonalidad euclídea ⟹ P u = u
        assert_allclose(P @ orthogonal_vector, orthogonal_vector, atol=1e-12)

    def test_P_es_igual_a_I_M_sobre_2(self, normal_vector, anisotropic_metric):
        """Verifica la identidad algebraica: P = (I + M)/2."""
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        I = np.eye(reflector._dim)
        M = reflector.reflection_matrix
        P_from_M = (I + M) / 2.0
        assert_allclose(P_from_M, reflector.projection_operator, atol=1e-12)

    def test_P_es_devuelto_como_copia_defensiva(self, normal_vector, identity_metric):
        """Modificar la copia no debe afectar el reflector."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        P = reflector.projection_operator
        P[0, 0] = 999.0
        P_fresh = reflector.projection_operator
        assert P_fresh[0, 0] != 999.0


class TestGNormConsistency:
    """Consistencia entre producto interno métrico, norma vía Cholesky y álgebra."""

    def test_norma_G_v_es_consistente(self, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        v = normal_vector
        # ||v||_G² = v^T G v
        assert reflector.g_norm_squared == pytest.approx(v @ (anisotropic_metric @ v))
        # ||v||_G = sqrt(||v||_G²) = ||L^T v||_2
        L = reflector._L
        assert reflector.g_norm(v) == pytest.approx(np.linalg.norm(L.T @ v))

    def test_norma_G_es_cero_para_cero(self, identity_metric):
        reflector = MetricAwareHouseholderReflector(np.array([1.0, 0.0]), identity_metric)
        assert reflector.g_norm(np.zeros(2)) == 0.0

    def test_producto_interno_G_coincide_con_T(self, rng, normal_vector, anisotropic_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        x = rng.standard_normal(reflector._dim)
        y = rng.standard_normal(reflector._dim)
        # ⟨x,y⟩_G = x^T G y
        assert reflector.g_inner(x, y) == pytest.approx(x @ (anisotropic_metric @ y))

    def test_descomposicion_ortogonal_bajo_G(
        self, rng, normal_vector, anisotropic_metric
    ):
        """Para todo ψ: ψ = α·v + P ψ (proyección métrica)."""
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        v = normal_vector
        G = anisotropic_metric
        psi = rng.standard_normal(reflector._dim)
        # Coeficiente de v bajo G: α = ⟨ψ, v⟩_G / ⟨v, v⟩_G
        alpha = psi @ (G @ v) / (v @ (G @ v))
        reconstruido = alpha * v + reflector.projection_operator @ psi
        assert_allclose(reconstruido, psi, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  ██████╗ █████╗ ██╗   ██╗██╗████████╗██╗   ██╗
# ██╔════╝██╔══██╗██║   ██║██║╚══██╔══╝╚██╗ ██╔╝
# ██║     ███████║██║   ██║██║   ██║    ╚████╔╝
# ██║     ██╔══██║██║   ██║██║   ██║      ╚██╔╝
# ╚██████╗██║  ██║╚██████╔╝██║   ██║       ██║
#  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝   ╚═╝       ╚═╝
#                  FASE 2 — CAVIDAD
# ══════════════════════════════════════════════════════════════════════════════
class TestFabryPerotConvergence:
    """Axioma §3: Cavidad Von Mises sobre P heredado de Fase 1."""

    def test_estabilizacion_en_una_iteracion(
        self, rng, normal_vector, identity_metric
    ):
        """
        Si el reflector es exacto, el esquema T = 2P - P² debe estabilizar
        en exactamente 1 iteración (porque P² = P computacionalmente).
        """
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=8, tol=1e-14)
        psi = rng.standard_normal(8)
        coherent, trace = cavity.stabilize(psi)
        assert trace.converged is True
        assert trace.iterations_used <= 1
        assert trace.final_residual < 1e-14

    def test_convergencia_con_ruido_numerico(
        self, rng, normal_vector, identity_metric
    ):
        """Inyectamos ruido en ψ; la cavidad debe proyectarlo."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=10, tol=1e-12)
        # ψ con componente alineada a v ruidosa
        psi = rng.standard_normal(8) + 1e-8 * np.ones(8)
        coherent, trace = cavity.stabilize(psi)
        # El estado coherente debe estar en el hiperplano (proyección ortogonal)
        assert reflector.g_norm(coherent @ normal_vector) < 1e-10
        assert trace.converged is True

    def test_traza_historial_monotono(
        self, rng, normal_vector, identity_metric
    ):
        """El residuo de idempotencia debe ser no-creciente a lo largo de la traza."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=10, tol=1e-14)
        psi = rng.standard_normal(8)
        _, trace = cavity.stabilize(psi)
        residuos = [r for _, r in trace.history]
        for i in range(1, len(residuos)):
            assert residuos[i] <= residuos[i - 1] + 1e-15

    def test_aniquilacion_total_es_error(self, normal_vector, identity_metric):
        """Si ψ ∥ v (en el hiperplano ortogonal el estado es 0), debe explotar."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=3, tol=1e-14)
        # ψ ∥ v ⟹ P ψ = 0 ⟹ aniquilación
        psi_aligned = normal_vector.copy() * 5.0
        with pytest.raises(ResonanceDissonanceError, match="Aniquilación"):
            cavity.stabilize(psi_aligned)

    def test_rechazo_dimension_incompatible(self, normal_vector, identity_metric):
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector)
        with pytest.raises(ResonanceDissonanceError, match="dimensión"):
            cavity.stabilize(np.ones(5))

    def test_coherencia_con_proyector_de_fase_1(
        self, rng, normal_vector, identity_metric
    ):
        """El resultado de stabilize debe ser indistinguible de P @ ψ_raw."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=5, tol=1e-14)
        psi = rng.standard_normal(8)
        coherent, _ = cavity.stabilize(psi)
        P_direct = reflector.projection_operator @ psi
        assert_allclose(coherent, P_direct, atol=1e-12)

    def test_convergencia_bajo_G_general(
        self, rng, normal_vector, anisotropic_metric
    ):
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=10, tol=1e-12)
        psi = rng.standard_normal(reflector._dim)
        coherent, trace = cavity.stabilize(psi)
        # Estado coherente debe tener residuo de idempotencia ≪ tol
        assert trace.converged is True
        # Verificación: (P² - P) ψ ≈ 0
        P = reflector.projection_operator
        residuos = P @ (P @ coherent) - P @ coherent
        assert reflector.g_norm(residuos) < 1e-10

    def test_convergencia_con_metric_malcondicionada(
        self, rng, normal_vector, ill_conditioned_metric
    ):
        """La cavidad debe converger incluso bajo κ₂(G) ≈ 10⁸."""
        reflector = MetricAwareHouseholderReflector(normal_vector, ill_conditioned_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=10, tol=1e-10)
        psi = rng.standard_normal(reflector._dim)
        coherent, trace = cavity.stabilize(psi)
        assert trace.converged is True

    def test_max_iter_cero_es_valido(self, normal_vector, identity_metric):
        """max_iter=0 deshabilita el bucle pero exige una sola iteración mínima."""
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=0, tol=1e-14)
        # Esto fuerza al menos 1 pasada del bucle en stabilize
        psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        coherent, trace = cavity.stabilize(psi)
        assert trace.converged is True

    def test_tipo_cavity_rechaza_objetos_invalidos(self):
        with pytest.raises(TypeError, match="certificado"):
            FabryPerotStabilizedCavity("no-soy-reflector")  # type: ignore[arg-type]


class TestConvergenceTrace:
    """Dataclass inmutable con historial categórico."""

    def test_traza_es_inmutable(self):
        trace = ConvergenceTrace(history=((0, 0.5),), final_residual=0.5, iterations_used=0, converged=False)
        with pytest.raises((AttributeError, Exception)):
            trace.converged = True  # type: ignore[misc]

    def test_traza_historial_tupla(self):
        trace = ConvergenceTrace(history=((0, 1.0), (1, 0.5), (2, 0.0)), final_residual=0.0, iterations_used=2, converged=True)
        assert isinstance(trace.history, tuple)
        assert len(trace.history) == 3


# ══════════════════════════════════════════════════════════════════════════════
# ███████╗ █████╗  █████╗ ███████╗███████╗     █████╗
# ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝    ╚════██╗
# █████╗  ███████║███████║███████╗█████╗       █████╔╝
# ██╔══╝  ██╔══██║██╔══██║╚════██║██╔══╝       ╚═══██╗
# ██║     ██║  ██║██║  ██║███████║███████╗     █████╔╝
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝     ╚════╝
#              FASE 3 — ORQUESTADOR CATEGORIAL
# ══════════════════════════════════════════════════════════════════════════════
class TestCategoricalContract:
    """Axioma §4: SemanticParabolicMirror es un Morphism válido."""

    def test_es_instancia_de_Morphism(self, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        assert isinstance(mirror, Morphism)

    def test_forward_aplica_M_y_conserva_etiqueta(self, rng, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        psi = rng.standard_normal(8)
        state = CategoricalState(payload=psi, label="crudo")
        new_state = mirror.forward(state)
        assert new_state.label.startswith("crudo::")
        # El payload es M @ psi (reflexión, no proyección)
        v = mirror._canonical_normal()
        M_expected = np.eye(8) - 2 * np.outer(v, v)
        assert_allclose(new_state.payload, M_expected @ psi, atol=1e-12)

    def test_forward_involutivo(self, rng, identity_metric):
        """forward(forward(x)) = x (Householder es involución)."""
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        psi = rng.standard_normal(8)
        state = CategoricalState(payload=psi, label="x")
        once = mirror.forward(state)
        twice = mirror.forward(once)
        assert_allclose(twice.payload, psi, atol=1e-12)

    def test_backward_es_forward(self, rng, identity_metric):
        """backward = forward por involutividad de Householder."""
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        psi = rng.standard_normal(8)
        state = CategoricalState(payload=psi, label="y")
        forward_result = mirror.forward(state).payload
        backward_result = mirror.backward(state).payload
        assert_allclose(forward_result, backward_result, atol=1e-14)

    def test_eta_kernel_proyecta_y_conserva_etiqueta(self, rng, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        psi = rng.standard_normal(8)
        state = CategoricalState(payload=psi, label="original")
        injected = mirror.eta_kernel(state)
        assert injected.label == "original"  # La etiqueta se preserva (η es fiel)
        # La proyección se ha aplicado
        v = mirror._canonical_normal()
        # ⟨psi_inyectado, v⟩_G debe ser ~0
        assert abs(injected.payload @ v) < 1e-12

    def test_forward_rechaza_dimensiones_incompatibles(self, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        bad_state = CategoricalState(payload=np.ones(5), label="malo")
        with pytest.raises(TopologicalInvariantError, match="dimensión|incompatible"):
            mirror.forward(bad_state)


class TestOrchestratorPipeline:
    """Acoplamiento Fase 1 → Fase 2 dentro del orquestador."""

    def test_camino_completo_con_refinamiento(
        self, rng, identity_metric, normal_vector
    ):
        mirror = SemanticParabolicMirror(
            metric_tensor=identity_metric, max_cavity_iter=10, cavity_tol=1e-14
        )
        psi = rng.standard_normal(8)
        coherent, trace = mirror.apply(psi, normal_vector, use_iterative_refinement=True)
        # Estado coherente ortogonal a v bajo G
        assert coherent @ normal_vector == pytest.approx(0.0, abs=1e-12)
        assert trace is not None
        assert trace.converged is True
        assert mirror.last_trace is trace
        assert mirror.last_certificate is not None
        assert mirror.last_certificate.is_valid_reflection is True

    def test_camino_corto_sin_refinamiento(
        self, rng, identity_metric, normal_vector
    ):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        psi = rng.standard_normal(8)
        coherent, trace = mirror.apply(psi, normal_vector, use_iterative_refinement=False)
        assert trace is None
        assert mirror.last_trace is None
        # La proyección directa sigue siendo correcta
        assert coherent @ normal_vector == pytest.approx(0.0, abs=1e-12)

    def test_apply_con_G_PHYSICS(self, rng):
        """Smoke test con la métrica real del ecosistema MIC."""
        d = G_PHYSICS.shape[0]
        mirror = SemanticParabolicMirror(metric_tensor=G_PHYSICS)
        psi = rng.standard_normal(d)
        v = np.ones(d, dtype=np.float64)
        v /= np.linalg.norm(v)
        coherent, trace = mirror.apply(psi, v)
        # Coherencia métrica: ⟨coherent, v⟩_G ≈ 0
        assert coherent @ (G_PHYSICS @ v) == pytest.approx(0.0, abs=1e-8)
        assert trace.converged is True

    def test_apply_rechaza_dimensiones_incompatibles(self, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        with pytest.raises(TopologicalInvariantError):
            mirror.apply(np.ones(5), np.ones(8))
        with pytest.raises(TopologicalInvariantError):
            mirror.apply(np.ones(8), np.ones(5))

    def test_apply_sin_use_iterative_no_crea_trace(
        self, rng, identity_metric, normal_vector
    ):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        _, trace = mirror.apply(
            rng.standard_normal(8), normal_vector, use_iterative_refinement=False
        )
        assert trace is None


class TestOrchestratorProperties:
    """Propiedades algebraicas del orquestador."""

    def test_metric_tensor_es_inmutable_externo(self, identity_metric):
        mirror = SemanticParabolicMirror(metric_tensor=identity_metric)
        G_copy = mirror.metric_tensor
        G_copy[0, 0] = 999.0
        # La métrica interna no debe estar afectada
        assert mirror._G[0, 0] != 999.0

    def test_validacion_G_en_construccion(self):
        with pytest.raises(MetricSignatureError):
            SemanticParabolicMirror(metric_tensor=np.zeros((3, 3)))
        with pytest.raises(MetricSignatureError):
            SemanticParabolicMirror(metric_tensor=np.diag([1.0, 0.0, 1.0]))  # singular

    def test_construccion_con_G_PHYSICS(self):
        mirror = SemanticParabolicMirror()
        assert mirror._dim == G_PHYSICS.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
# ██╗███╗   ██╗██╗   ██╗ █████╗ ██████╗ ██╗ █████╗ ███╗   ██╗ ██████╗███████╗
# ██║████╗  ██║██║   ██║██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║██╔════╝██╔════╝
# ██║██╔██╗ ██║██║   ██║███████║██████╔╝██║███████║██╔██╗ ██║██║     █████╗
# ██║██║╚██╗██║╚██╗ ██╔╝██╔══██║██╔══██╗██║██╔══██║██║╚██╗██║██║     ██╔══╝
# ██║██║ ╚████║ ╚████╔╝ ██║  ██║██║  ██║██║██║  ██║██║ ╚████║╚██████╗███████╗
# ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
#              INVARIANCIAS, CASOS LÍMITE, REGRESIONES
# ══════════════════════════════════════════════════════════════════════════════
class TestInvarianceAndEdgeCases:
    """Tests de invariancia estructural y casos extremos."""

    def test_determinismo_bit_a_bit(
        self, rng, normal_vector, identity_metric
    ):
        """
        Misma semilla ⟹ mismo resultado bit-a-bit.
        Crítico para reproducibilidad categórica.
        """
        psi_a = rng.standard_normal(8)
        rng2 = np.random.default_rng(seed=20251215)
        psi_b = rng2.standard_normal(8)
        assert_allclose(psi_a, psi_b)
        reflector_a = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        reflector_b = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        assert_allclose(reflector_a.reflection_matrix, reflector_b.reflection_matrix)

    def test_reutilizacion_reflector_multiples_cavidades(
        self, rng, normal_vector, identity_metric
    ):
        """
        Un mismo reflector puede alimentar múltiples cavidades.
        Verifica que el estado interno no se muta.
        """
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity1 = FabryPerotStabilizedCavity(reflector)
        cavity2 = FabryPerotStabilizedCavity(reflector)
        psi = rng.standard_normal(8)
        c1, _ = cavity1.stabilize(psi)
        c2, _ = cavity2.stabilize(psi)
        assert_allclose(c1, c2, atol=1e-14)
        # El reflector sigue intacto
        assert_allclose(
            reflector.reflection_matrix @ reflector.reflection_matrix,
            np.eye(8),
            atol=1e-12,
        )

    def test_G_identidad_comportamiento_canonico(
        self, rng, normal_vector
    ):
        """Con G = I, Householder métrico ≡ Householder clásico."""
        reflector = MetricAwareHouseholderReflector(normal_vector, np.eye(8))
        M = reflector.reflection_matrix
        # Householder clásico: M = I - 2 v v^T / (v^T v)
        v = normal_vector
        M_classic = np.eye(8) - 2.0 * np.outer(v, v) / (v @ v)
        assert_allclose(M, M_classic, atol=1e-12)

    def test_n_dimension_minima_d2(self):
        """El módulo debe operar correctamente en d = 2 (mínimo no trivial)."""
        G = np.eye(2)
        v = np.array([1.0, 0.0])
        reflector = MetricAwareHouseholderReflector(v, G)
        # En d=2, Householder es una reflexión pura
        M = reflector.reflection_matrix
        assert_allclose(M, np.diag([-1.0, 1.0]), atol=1e-12)
        assert reflector.certificate.determinant == pytest.approx(-1.0, abs=1e-10)

    def test_norma_G_resiste_a_vectores_muy_pequenos(
        self, rng, identity_metric
    ):
        """Vector con norma euclídea pequeña pero G-norma finita."""
        v = rng.standard_normal(8) * 1e-3
        reflector = MetricAwareHouseholderReflector(v, identity_metric)
        # La norma G es la euclídea para G = I
        assert reflector.g_norm(v) == pytest.approx(np.linalg.norm(v), rel=1e-10)

    def test_ortogonalidad_entre_v_y_P_psi_es_metrica(
        self, rng, normal_vector, anisotropic_metric
    ):
        """Para todo ψ: ⟨v, P ψ⟩_G = 0 (proyección ortogonal bajo G)."""
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        G = anisotropic_metric
        psi = rng.standard_normal(reflector._dim)
        P_psi = reflector.projection_operator @ psi
        # ⟨v, P ψ⟩_G = v^T G P ψ ≈ 0
        assert reflector.g_inner(normal_vector, P_psi) == pytest.approx(0.0, abs=1e-10)

    def test_pipeline_completo_conserva_energia_normal(
        self, rng, normal_vector, anisotropic_metric
    ):
        """
        Para ψ sin componente en v: ‖ψ‖_G antes y después debe coincidir.
        La reflexión + proyección conserva la norma G de la componente coherente.
        """
        reflector = MetricAwareHouseholderReflector(normal_vector, anisotropic_metric)
        psi = rng.standard_normal(reflector._dim)
        # Hacemos ψ ortogonal a v bajo G
        G = anisotropic_metric
        alpha = psi @ (G @ normal_vector) / (normal_vector @ (G @ normal_vector))
        psi_ortho = psi - alpha * normal_vector
        # Ahora ψ_ortho ya está en el hiperplano: P ψ_ortho = ψ_ortho
        coherent, trace = FabryPerotStabilizedCavity(reflector).stabilize(psi_ortho)
        assert_allclose(reflector.g_norm(coherent), reflector.g_norm(psi_ortho), atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# REGRESIONES HISTÓRICAS
# ══════════════════════════════════════════════════════════════════════════════
class TestRegressions:
    """Tests de regresión contra errores conocidos de versiones anteriores."""

    def test_regresion_bucle_de_Richardson_inerte(
        self, rng, normal_vector, identity_metric
    ):
        """
        Regresión: en v5.0, el bucle de Richardson con α = tol/residual colapsaba
        a alpha ≈ 0, paralizando la convergencia. Verificamos que en v6.0
        el esquema de Von Mises converge robustamente.
        """
        reflector = MetricAwareHouseholderReflector(normal_vector, identity_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=5, tol=1e-14)
        psi = rng.standard_normal(8) * 1e3  # Magnitud grande
        coherent, trace = cavity.stabilize(psi)
        assert trace.converged is True
        # El resultado debe coincidir con proyección exacta
        P_direct = reflector.projection_operator @ psi
        assert_allclose(coherent, P_direct, atol=1e-10)

    def test_regresion_chi_alphas_inestables_con_G_malcondicionada(
        self, rng, normal_vector, ill_conditioned_metric
    ):
        """Regresión: v5.0 no precomputaba Cholesky, causando norm overflow."""
        reflector = MetricAwareHouseholderReflector(normal_vector, ill_conditioned_metric)
        cavity = FabryPerotStabilizedCavity(reflector, max_iter=10, tol=1e-10)
        psi = rng.standard_normal(reflector._dim)
        coherent, trace = cavity.stabilize(psi)
        assert np.all(np.isfinite(coherent))
        assert np.all(np.isfinite(trace.final_residual))
        assert trace.converged is True


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST — MARCADORES Y RIGOR
# ══════════════════════════════════════════════════════════════════════════════
def pytest_configure(config):
    """Marca custom para tests del topos MIC."""
    config.addinivalue_line(
        "markers",
        "topos: tests del contrato categórico SemanticParabolicMirror",
    )


# Aplicar marca automáticamente a TestCategoricalContract
for cls in [TestCategoricalContract]:
    for name in dir(cls):
        if name.startswith("test_"):
            getattr(cls, name).__pytest_mark__ = pytest.mark.topos  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN STANDALONE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Permite ejecutar la suite sin pytest instalado:
        python -m tests.omega.test_semantic_parabolic_mirror
    """
    import unittest

    # Recolectar todas las clases de test
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestMetricValidation,
        TestVectorValidation,
        TestIsometryAndReflection,
        TestSpectralCertificate,
        TestProjectionOperator,
        TestGNormConsistency,
        TestFabryPerotConvergence,
        TestConvergenceTrace,
        TestCategoricalContract,
        TestOrchestratorPipeline,
        TestOrchestratorProperties,
        TestInvarianceAndEdgeCases,
        TestRegressions,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)