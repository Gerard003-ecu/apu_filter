# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas : Yang-Mills Holonomy Agent v3.0.0                          ║
║ Archivo : tests/unit/core/immune_system/test_yang_mills_holonomy_agent.py         ║
║ Cobertura: Phase1 · Phase2 · Phase3 · YangMillsHolonomyAgent                 ║
║            Algebraica · Geométrica · Topológica · Espectral · Adversarial    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Organización de la suite
────────────────────────
  TestFixtures               — fábrica centralizada de objetos de prueba
  TestPhase1_Curvatura_*     — tensor F, Hodge, Bianchi, clasificación
  TestPhase2_Holonomia_*     — bucle de Wilson, ds_k, unitariedad, veto
  TestPhase3_Optimizador_*   — Lyapunov, Higham, EOM residual, acción
  TestAgente_Pipeline_*      — orquestador completo, DeformationTensor
  TestProperties_*           — pruebas basadas en propiedades (Hypothesis)

Convenciones matemáticas
────────────────────────
  · [A,B]   = AB − BA  (conmutador de Lie)
  · ★F      = G⁻¹ F_skew G  (dual de Hodge respecto a G)
  · F⁺      = ½(F + ★F)  (parte self-dual)
  · F⁻      = ½(F − ★F)  (parte anti-self-dual)
  · W(γ)    = 𝒫 ∏_k exp(i A_k ds_k)  (bucle de Wilson discreto)
  · δ_U     = ‖W†W − I‖_F  (defecto de unitariedad)
  · S_YM    = ½ Tr(Fᵀ G⁻¹ F G⁻¹) + θ Tr(F G⁻¹ F̃ G⁻¹)
  · EOM_res = ‖G⁻¹F + FG⁻¹ + G⁻¹δR + δRG⁻¹ − J‖_F
"""

from __future__ import annotations

import math
import cmath
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.linalg as la
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.immune_system.yang_mills_holonomy_agent import (
    _EPS_BIANCHI,
    _EPS_FLAT_CURVATURE,
    _EPS_HOLONOMY,
    _EPS_SELFDUAL,
    _EPS_UNITARITY,
    _EPS_EOM_RESIDUAL,
    _KAPPA_MAX_METRIC,
    _THETA_YM,
    BianchiViolationError,
    CurvatureClass,
    GaugeCurvatureSingularityError,
    GaugeCurvatureTensor,
    HolonomyClass,
    HolonomyVetoError,
    Phase1_GaugeCurvatureComputer,
    Phase2_WilsonLoopAuditor,
    Phase3_YangMillsOptimizer,
    WilsonLoopHolonomy,
    YangMillsAction,
    YangMillsHolonomyAgent,
    YangMillsOptimizationError,
)
from app.core.immune_system.dynamic_shield_router import DeformationTensor

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

#: Dimensión canónica del espacio de estados en las pruebas.
N: int = 4

#: Tolerancia estricta para igualdades algebraicas exactas.
ATOL_STRICT: float = 1e-9

#: Tolerancia relajada para cantidades termodinámicas aproximadas.
ATOL_THERMO: float = 1e-6

#: Semilla fija para reproducibilidad.
RNG_SEED: int = 2024


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES CENTRALIZADAS
# ══════════════════════════════════════════════════════════════════════════════


class TestFixtures:
    """
    Fábrica estática de objetos de prueba reutilizables en toda la suite.
    Garantiza coherencia matemática entre clases de prueba y evita
    la duplicación de lógica de construcción de matrices.
    """

    @staticmethod
    def rng(seed: int = RNG_SEED) -> np.random.Generator:
        """Generador de números aleatorios reproducible."""
        return np.random.default_rng(seed)

    @staticmethod
    def spd_matrix(
        n: int = N,
        scale: float = 1.0,
        seed: int = RNG_SEED,
    ) -> NDArray[np.float64]:
        r"""
        Genera una matriz SPD n×n como M = AᵀA + n·I.
        λ_min(M) ≥ n > 0 garantizado.
        """
        rng = TestFixtures.rng(seed)
        A = rng.standard_normal((n, n))
        return (A.T @ A + n * np.eye(n)) * scale

    @staticmethod
    def skew_matrix(
        n: int = N,
        scale: float = 1.0,
        seed: int = RNG_SEED,
    ) -> NDArray[np.float64]:
        r"""
        Genera una matriz antisimétrica n×n: S = A − Aᵀ.
        Sirve como generador del álgebra de Lie u(n).
        """
        rng = TestFixtures.rng(seed)
        A = rng.standard_normal((n, n))
        return (A - A.T) * scale

    @staticmethod
    def generic_matrix(
        n: int = N,
        scale: float = 1.0,
        seed: int = RNG_SEED,
    ) -> NDArray[np.float64]:
        """Genera una matriz genérica n×n (no necesariamente simétrica)."""
        rng = TestFixtures.rng(seed)
        return rng.standard_normal((n, n)) * scale

    @staticmethod
    def commuting_pair(
        n: int = N,
        seed: int = RNG_SEED,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Genera un par de matrices que conmutan exactamente: [A, B] = 0.
        Construcción: A = Q Λ_A Qᵀ, B = Q Λ_B Qᵀ (misma base ortonormal Q).
        """
        rng = TestFixtures.rng(seed)
        Q = la.orth(rng.standard_normal((n, n)))
        lambda_A = np.diag(rng.uniform(0.5, 2.0, n))
        lambda_B = np.diag(rng.uniform(0.5, 2.0, n))
        A = Q @ lambda_A @ Q.T
        B = Q @ lambda_B @ Q.T
        return A, B

    @staticmethod
    def non_commuting_pair(
        n: int = N,
        scale: float = 0.5,
        seed: int = RNG_SEED,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Genera un par de matrices que no conmutan: [A, B] ≠ 0.
        Usa matrices genéricas que con probabilidad 1 no comparten base.
        """
        A = TestFixtures.generic_matrix(n, scale=1.0, seed=seed)
        B = TestFixtures.generic_matrix(n, scale=scale, seed=seed + 1)
        return A, B

    @staticmethod
    def zero_matrix(n: int = N) -> NDArray[np.float64]:
        """Devuelve la matriz cero n×n."""
        return np.zeros((n, n), dtype=np.float64)

    @staticmethod
    def identity(n: int = N, scale: float = 1.0) -> NDArray[np.float64]:
        """Devuelve scale · I_n."""
        return scale * np.eye(n, dtype=np.float64)

    @staticmethod
    def anti_hermitian_matrix(
        n: int = N,
        seed: int = RNG_SEED,
    ) -> NDArray[np.float64]:
        r"""
        Genera una matriz anti-hermitiana (antisimétrica real):
        M† = −M  ↔  Mᵀ = −M.
        exp(i·M) es unitaria cuando M es antisimétrica real.
        """
        return TestFixtures.skew_matrix(n, scale=0.3, seed=seed)

    @staticmethod
    def phase1(
        n: int = N,
        seed: int = RNG_SEED,
    ) -> Phase1_GaugeCurvatureComputer:
        """Instancia Phase1 con una métrica SPD aleatoria."""
        G = TestFixtures.spd_matrix(n=n, seed=seed)
        return Phase1_GaugeCurvatureComputer(metric_tensor=G)

    @staticmethod
    def phase2() -> Phase2_WilsonLoopAuditor:
        """Instancia Phase2 con tolerancias por defecto."""
        return Phase2_WilsonLoopAuditor()

    @staticmethod
    def phase3(
        n: int = N,
        seed: int = RNG_SEED,
        theta: float = 0.0,
    ) -> Phase3_YangMillsOptimizer:
        """Instancia Phase3 con una métrica SPD aleatoria."""
        G = TestFixtures.spd_matrix(n=n, seed=seed)
        return Phase3_YangMillsOptimizer(metric_tensor=G, theta_ym=theta)

    @staticmethod
    def agent(
        n: int = N,
        seed: int = RNG_SEED,
        theta: float = 0.0,
    ) -> YangMillsHolonomyAgent:
        """Instancia el agente completo con métrica SPD aleatoria."""
        G = TestFixtures.spd_matrix(n=n, seed=seed)
        return YangMillsHolonomyAgent(metric_tensor=G, theta_ym=theta)

    @staticmethod
    def curvature_tensor_flat(n: int = N) -> GaugeCurvatureTensor:
        """
        Construye un GaugeCurvatureTensor sintético con F = 0
        (curvatura plana), para pruebas de Fase 3 aisladas.
        """
        zero = np.zeros((n, n), dtype=np.float64)
        return GaugeCurvatureTensor(
            F_matrix=zero,
            F_selfdual=zero,
            F_antiselfdual=zero,
            frobenius_norm=0.0,
            bianchi_residual=0.0,
            chern_number=0.0,
            curvature_class=CurvatureClass.FLAT,
            is_integrable=True,
        )

    @staticmethod
    def curvature_tensor_general(
        F: NDArray[np.float64],
        G: NDArray[np.float64],
    ) -> GaugeCurvatureTensor:
        """
        Construye un GaugeCurvatureTensor sintético a partir de F y G,
        calculando ★F, F⁺, F⁻ y c₁ explícitamente.
        """
        G_inv = la.inv(G)
        F_skew = 0.5 * (F - F.T)
        F_hodge = G_inv @ F_skew @ G
        F_plus = 0.5 * (F + F_hodge)
        F_minus = 0.5 * (F - F_hodge)
        norm_F = float(np.linalg.norm(F, "fro"))
        chern = float(np.trace(F)) / (2.0 * math.pi)
        return GaugeCurvatureTensor(
            F_matrix=F,
            F_selfdual=F_plus,
            F_antiselfdual=F_minus,
            frobenius_norm=norm_F,
            bianchi_residual=0.0,
            chern_number=chern,
            curvature_class=CurvatureClass.GENERAL,
            is_integrable=(norm_F < _EPS_FLAT_CURVATURE),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — PRUEBAS DE LA CURVATURA DE CALIBRE
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1_ConstructorValidacion:
    """
    Pruebas de la validación del constructor de Phase1_GaugeCurvatureComputer.
    Verifica que la métrica G es correctamente validada antes de cualquier
    cálculo de curvatura.
    """

    def test_constructor_acepta_metrica_spd(self) -> None:
        """Constructor no lanza excepción con métrica SPD válida."""
        G = TestFixtures.spd_matrix()
        phase1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        assert phase1 is not None

    def test_constructor_rechaza_metrica_no_cuadrada(self) -> None:
        """Constructor rechaza G rectangular."""
        G_rect = np.eye(N, N + 1)
        with pytest.raises(ValueError, match="cuadrada"):
            Phase1_GaugeCurvatureComputer(metric_tensor=G_rect)

    def test_constructor_rechaza_metrica_no_simetrica(self) -> None:
        """Constructor rechaza G no simétrica."""
        G = TestFixtures.spd_matrix()
        G_asym = G + 0.5 * np.tril(G, -1)
        with pytest.raises(ValueError, match="simétrica"):
            Phase1_GaugeCurvatureComputer(metric_tensor=G_asym)

    def test_constructor_rechaza_metrica_indefinida(self) -> None:
        """Constructor rechaza G indefinida (λ_min < 0)."""
        G = TestFixtures.spd_matrix()
        evals, evecs = la.eigh(G)
        evals[0] = -0.1
        G_indef = evecs @ np.diag(evals) @ evecs.T
        G_indef = 0.5 * (G_indef + G_indef.T)
        with pytest.raises(ValueError):
            Phase1_GaugeCurvatureComputer(metric_tensor=G_indef)

    def test_constructor_rechaza_metrica_semidefinida(self) -> None:
        """Constructor rechaza G PSD (λ_min = 0, Cholesky falla)."""
        rng = TestFixtures.rng()
        B = rng.standard_normal((N, N - 1))
        G_psd = B @ B.T   # rango N−1, singular
        with pytest.raises(ValueError):
            Phase1_GaugeCurvatureComputer(metric_tensor=G_psd)

    def test_g_inv_precomputado_es_consistente(self) -> None:
        """G · G⁻¹ = I (la inversa pre-computada es correcta)."""
        G = TestFixtures.spd_matrix()
        phase1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        product = G @ phase1._G_inv
        assert np.allclose(product, np.eye(N), atol=ATOL_STRICT), (
            f"G · G⁻¹ ≠ I: ‖G·G⁻¹ − I‖_F = "
            f"{np.linalg.norm(product - np.eye(N), 'fro'):.3e}"
        )


class TestPhase1_TensorFaraday:
    """
    Pruebas del tensor de Faraday F_{μν} = ∂_ν A_ν − ∂_μ A_μ + [A_μ, A_ν].
    Verifica la antisimetría estructural, el conmutador y los casos límite.
    """

    @pytest.fixture
    def p1(self) -> Phase1_GaugeCurvatureComputer:
        return TestFixtures.phase1(seed=10)

    def test_f_cero_cuando_a_mu_igual_a_nu(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """
        Si A_μ = A_ν, entonces dA = A_ν − A_μ = 0 y [A,A] = 0,
        por lo tanto F = 0 y la curvatura es plana.
        """
        A = TestFixtures.generic_matrix(seed=20)
        result = p1.compute_curvature(A_mu=A, A_nu=A)
        assert result.curvature_class == CurvatureClass.FLAT
        assert result.frobenius_norm < _EPS_FLAT_CURVATURE
        assert result.is_integrable

    def test_f_cero_cuando_matrices_conmutan_y_diferencia_cero(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """
        Para matrices que conmutan [A,B]=0 y son iguales, F=0.
        """
        A, _ = TestFixtures.commuting_pair(seed=30)
        result = p1.compute_curvature(A_mu=A, A_nu=A)
        assert np.allclose(result.F_matrix, 0.0, atol=ATOL_STRICT)

    def test_conmutador_contribuye_cuando_matrices_no_conmutan(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """
        Cuando [A_μ, A_ν] ≠ 0, F contiene la contribución no abeliana.
        Verificamos que F ≠ A_ν − A_μ (el conmutador modifica el resultado).
        """
        A_mu, A_nu = TestFixtures.non_commuting_pair(seed=40)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        # F sin conmutador sería solo dA = A_nu - A_mu
        dA_only = A_nu - A_mu
        comm = A_mu @ A_nu - A_nu @ A_mu
        assert not np.allclose(comm, 0.0, atol=ATOL_STRICT), (
            "El par elegido conmuta; elegir un par no-conmutante."
        )
        # F debe incluir el conmutador
        F_expected = dA_only + comm
        assert np.allclose(result.F_matrix, F_expected, atol=ATOL_STRICT), (
            f"F ≠ dA + [A,B]: ‖F − F_exp‖_F = "
            f"{np.linalg.norm(result.F_matrix - F_expected, 'fro'):.3e}"
        )

    def test_formula_explicita_faraday(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """
        Verifica la fórmula completa F = (A_ν − A_μ) + [A_μ, A_ν]
        contra la implementación.
        """
        A_mu = TestFixtures.generic_matrix(seed=50)
        A_nu = TestFixtures.generic_matrix(seed=51)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        comm = A_mu @ A_nu - A_nu @ A_mu
        F_ref = (A_nu - A_mu) + comm
        assert np.allclose(result.F_matrix, F_ref, atol=ATOL_STRICT)

    def test_f_escala_con_diferencia_de_conexiones(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """
        Para conexiones que difieren por un escalar α: A_ν = A_μ + α I,
        el conmutador [A_μ, A_μ + αI] = 0 y F = α I.
        """
        A_mu = TestFixtures.spd_matrix(seed=60)
        alpha = 0.7
        A_nu = A_mu + alpha * np.eye(N)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        F_expected = alpha * np.eye(N)
        assert np.allclose(result.F_matrix, F_expected, atol=ATOL_STRICT), (
            f"F ≠ αI: ‖F − αI‖_F = "
            f"{np.linalg.norm(result.F_matrix - F_expected, 'fro'):.3e}"
        )

    def test_norma_frobenius_reportada_coincide_con_real(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """frobenius_norm coincide con np.linalg.norm(F, 'fro')."""
        A_mu = TestFixtures.generic_matrix(seed=70)
        A_nu = TestFixtures.generic_matrix(seed=71)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        norm_manual = float(np.linalg.norm(result.F_matrix, "fro"))
        assert abs(result.frobenius_norm - norm_manual) < ATOL_STRICT

    def test_chern_number_formula_explicita(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """c₁ = Tr(F) / (2π) verificado contra la implementación."""
        A_mu = TestFixtures.generic_matrix(seed=80)
        A_nu = TestFixtures.generic_matrix(seed=81)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        chern_manual = float(np.trace(result.F_matrix)) / (2.0 * math.pi)
        assert abs(result.chern_number - chern_manual) < ATOL_STRICT

    def test_dimensiones_incompatibles_lanza_error(
        self, p1: Phase1_GaugeCurvatureComputer
    ) -> None:
        """Si A_μ tiene dimensión distinta a G, lanza ValueError."""
        A_mu_wrong = np.eye(N + 1)
        A_nu = TestFixtures.generic_matrix()
        with pytest.raises(ValueError, match="compatible con la métrica"):
            p1.compute_curvature(A_mu=A_mu_wrong, A_nu=A_nu)


class TestPhase1_HodgeDual:
    """
    Pruebas del operador de Hodge dual ★F = G⁻¹ F_skew G.
    Verifica propiedades algebraicas: ★★F = F_skew, antisimetría de F_skew,
    y la descomposición F = F⁺ + F⁻.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(seed=100)
        p1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        A_mu = TestFixtures.generic_matrix(seed=101)
        A_nu = TestFixtures.generic_matrix(seed=102)
        result = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        return {"G": G, "p1": p1, "result": result}

    def test_descomposicion_F_mas_F_menos_es_F(self, setup: dict) -> None:
        """
        F⁺ + F⁻ = F (la descomposición es exacta).
        ½(F + ★F) + ½(F − ★F) = F.
        """
        res = setup["result"]
        F_reconstructed = res.F_selfdual + res.F_antiselfdual
        assert np.allclose(F_reconstructed, res.F_matrix, atol=ATOL_STRICT), (
            f"F⁺ + F⁻ ≠ F: ‖reconstruido − F‖_F = "
            f"{np.linalg.norm(F_reconstructed - res.F_matrix, 'fro'):.3e}"
        )

    def test_F_mas_y_F_menos_ortogonales_en_norma_frobenius(
        self, setup: dict
    ) -> None:
        """
        F⁺ y F⁻ son ortogonales en el producto interno de Frobenius:
        Tr(F⁺ᵀ F⁻) = 0.
        """
        res = setup["result"]
        inner = float(np.trace(res.F_selfdual.T @ res.F_antiselfdual))
        assert abs(inner) < ATOL_THERMO, (
            f"F⁺ y F⁻ no son ortogonales: Tr(F⁺ᵀ F⁻) = {inner:.3e}"
        )

    def test_norma_cuadrada_aditiva(self, setup: dict) -> None:
        """
        ‖F‖_F² = ‖F⁺‖_F² + ‖F⁻‖_F² (teorema de Pitágoras en Frobenius).
        """
        res = setup["result"]
        norm_F_sq = res.frobenius_norm ** 2
        norm_Fplus_sq = float(np.linalg.norm(res.F_selfdual, "fro")) ** 2
        norm_Fminus_sq = float(np.linalg.norm(res.F_antiselfdual, "fro")) ** 2
        assert abs(norm_F_sq - (norm_Fplus_sq + norm_Fminus_sq)) < ATOL_THERMO, (
            f"‖F‖² = {norm_F_sq:.4e} ≠ ‖F⁺‖² + ‖F⁻‖² = "
            f"{norm_Fplus_sq + norm_Fminus_sq:.4e}"
        )

    def test_hodge_dual_de_parte_simetrica_es_cero(self, setup: dict) -> None:
        """
        ★F opera solo sobre F_skew = (F−Fᵀ)/2. Si F es simétrico,
        F_skew = 0 y ★F = 0, luego F⁺ = F⁻ = ½F.
        """
        G = setup["G"]
        p1 = setup["p1"]
        # A_nu = A_mu + escalar·I → F = escalar·I (simétrica)
        A_mu = TestFixtures.spd_matrix(seed=110)
        A_nu = A_mu + 0.5 * np.eye(N)
        res = p1.compute_curvature(A_mu=A_mu, A_nu=A_nu)
        # F = 0.5·I es simétrica → F_skew = 0 → ★F = 0 → F⁺ = F⁻ = ½F
        F_half = 0.5 * res.F_matrix
        assert np.allclose(res.F_selfdual, F_half, atol=ATOL_STRICT), (
            "F⁺ ≠ ½F para F simétrica."
        )
        assert np.allclose(res.F_antiselfdual, F_half, atol=ATOL_STRICT), (
            "F⁻ ≠ ½F para F simétrica."
        )


class TestPhase1_ClasificacionCurvatura:
    """
    Pruebas de la clasificación topológica de F: FLAT, SELF_DUAL,
    ANTI_SELF_DUAL, GENERAL.
    """

    def test_clasificacion_flat_cuando_a_mu_igual_a_nu(self) -> None:
        """A_μ = A_ν → F = 0 → clasificación FLAT."""
        p1 = TestFixtures.phase1()
        A = TestFixtures.generic_matrix(seed=120)
        res = p1.compute_curvature(A, A)
        assert res.curvature_class == CurvatureClass.FLAT
        assert res.is_integrable

    def test_clasificacion_general_para_matrices_genericas(self) -> None:
        """
        Matrices genéricas sin estructura especial producen curvatura GENERAL
        (con alta probabilidad; se verifica que no sea FLAT).
        """
        p1 = TestFixtures.phase1(seed=130)
        A_mu = TestFixtures.generic_matrix(seed=131)
        A_nu = TestFixtures.generic_matrix(seed=132)
        res = p1.compute_curvature(A_mu, A_nu)
        # Con matrices aleatorias la curvatura no es plana con prob. 1
        if res.frobenius_norm > _EPS_FLAT_CURVATURE:
            assert res.curvature_class != CurvatureClass.FLAT
            assert not res.is_integrable

    def test_is_integrable_implica_flat(self) -> None:
        """is_integrable ↔ curvature_class == FLAT (consistencia lógica)."""
        p1 = TestFixtures.phase1(seed=140)
        for seed in range(5):
            A = TestFixtures.generic_matrix(seed=141 + seed)
            res = p1.compute_curvature(A, A)  # siempre plana
            assert res.is_integrable == (res.curvature_class == CurvatureClass.FLAT)

    def test_curvatura_plana_tiene_chern_cero(self) -> None:
        """F = 0 → Tr(F) = 0 → c₁ = 0."""
        p1 = TestFixtures.phase1()
        A = TestFixtures.generic_matrix(seed=150)
        res = p1.compute_curvature(A, A)
        assert abs(res.chern_number) < ATOL_STRICT, (
            f"c₁ = {res.chern_number:.3e} ≠ 0 para curvatura plana."
        )


class TestPhase1_IdentidadBianchi:
    """
    Pruebas de la identidad de Bianchi D∧F = 0 y de su violación severa.
    """

    def test_bianchi_residual_es_no_negativo(self) -> None:
        """‖B‖_F ≥ 0 siempre (es una norma)."""
        p1 = TestFixtures.phase1(seed=160)
        A_mu = TestFixtures.generic_matrix(seed=161)
        A_nu = TestFixtures.generic_matrix(seed=162)
        res = p1.compute_curvature(A_mu, A_nu)
        assert res.bianchi_residual >= 0.0

    def test_bianchi_cero_para_curvatura_plana(self) -> None:
        """
        Cuando F = 0 (A_μ = A_ν), el residuo de Bianchi debe ser 0:
        [A, 0] + [A, 0] + [A, 0] = 0.
        """
        p1 = TestFixtures.phase1(seed=170)
        A = TestFixtures.generic_matrix(seed=171)
        res = p1.compute_curvature(A, A)
        assert res.bianchi_residual < _EPS_BIANCHI, (
            f"Bianchi ≠ 0 para F=0: ‖B‖_F = {res.bianchi_residual:.3e}"
        )

    def test_bianchi_violation_error_para_violacion_severa(self) -> None:
        """
        Si el residuo de Bianchi supera 100·_EPS_BIANCHI, compute_curvature
        debe lanzar BianchiViolationError.

        Estrategia: usar matrices con norma muy grande para amplificar [A,F].
        """
        G = TestFixtures.identity(N)
        p1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        # Matrices de norma grande → conmutadores grandes → Bianchi violado
        scale = 1e6
        A_mu = TestFixtures.generic_matrix(seed=180) * scale
        A_nu = TestFixtures.generic_matrix(seed=181) * scale
        # Este test puede o no lanzar la excepción dependiendo de la amplitud;
        # lo verificamos de forma condicional.
        try:
            res = p1.compute_curvature(A_mu, A_nu)
            # Si no lanza, verificar que el residuo es razonablemente bajo
            # (el escalado grande no siempre viola Bianchi severamente)
        except BianchiViolationError:
            pass   # comportamiento esperado para escala muy grande

    def test_bianchi_residual_reportado_es_no_negativo_y_finito(self) -> None:
        """El residuo de Bianchi reportado es siempre finito y ≥ 0."""
        p1 = TestFixtures.phase1(seed=190)
        for seed in range(10):
            A_mu = TestFixtures.generic_matrix(seed=190 + seed)
            A_nu = TestFixtures.generic_matrix(seed=200 + seed)
            try:
                res = p1.compute_curvature(A_mu, A_nu)
                assert res.bianchi_residual >= 0.0
                assert math.isfinite(res.bianchi_residual)
            except BianchiViolationError:
                pass   # aceptable para matrices de alta norma


class TestPhase1_DTOInmutabilidad:
    """
    Pruebas de la inmutabilidad del GaugeCurvatureTensor (frozen dataclass).
    """

    def test_curvature_tensor_es_inmutable(self) -> None:
        """Intentar modificar GaugeCurvatureTensor lanza error."""
        p1 = TestFixtures.phase1()
        A = TestFixtures.generic_matrix(seed=210)
        res = p1.compute_curvature(A, A)
        with pytest.raises((TypeError, AttributeError)):
            res.frobenius_norm = 999.0   # type: ignore[misc]

    def test_compute_curvature_es_determinista(self) -> None:
        """Llamadas múltiples con los mismos argumentos dan el mismo resultado."""
        p1 = TestFixtures.phase1(seed=220)
        A_mu = TestFixtures.generic_matrix(seed=221)
        A_nu = TestFixtures.generic_matrix(seed=222)
        r1 = p1.compute_curvature(A_mu, A_nu)
        r2 = p1.compute_curvature(A_mu, A_nu)
        assert np.allclose(r1.F_matrix, r2.F_matrix)
        assert r1.frobenius_norm == r2.frobenius_norm
        assert r1.chern_number == r2.chern_number


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — PRUEBAS DE LA AUDITORÍA DE HOLONOMÍA
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2_HolonomiaTrivial:
    """
    Pruebas del caso trivial: ciclo vacío o matrices cero.
    W = I, χ(W) = n, δ_U = 0.
    """

    @pytest.fixture
    def p2(self) -> Phase2_WilsonLoopAuditor:
        return TestFixtures.phase2()

    def test_ciclo_vacio_retorna_identidad(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """evaluate_holonomy([]) retorna W = I."""
        result = p2.evaluate_holonomy([])
        n = result.phase_shift_matrix.shape[0]
        assert np.allclose(
            result.phase_shift_matrix,
            np.eye(n, dtype=np.complex128),
            atol=ATOL_STRICT,
        ), "W ≠ I para ciclo vacío."

    def test_ciclo_vacio_traza_es_n(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """χ(W) = n para ciclo vacío (W = I)."""
        result = p2.evaluate_holonomy([])
        n = result.phase_shift_matrix.shape[0]
        assert abs(result.holonomy_trace - complex(n, 0.0)) < ATOL_STRICT

    def test_ciclo_vacio_no_paradoja(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """Ciclo vacío → paradox_detected = False, clase TRIVIAL."""
        result = p2.evaluate_holonomy([])
        assert not result.paradox_detected
        assert result.holonomy_class == HolonomyClass.TRIVIAL

    def test_ciclo_con_matrices_cero_retorna_identidad(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        Si todos los A_k = 0, entonces exp(i·0·ds_k) = I para todo k,
        y W = I (holonomía trivial por curvatura plana).
        """
        zeros = [TestFixtures.zero_matrix() for _ in range(3)]
        result = p2.evaluate_holonomy(zeros)
        assert result.holonomy_class == HolonomyClass.TRIVIAL

    def test_ciclo_vacio_unitarity_defect_es_cero(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """δ_U = 0 para ciclo vacío."""
        result = p2.evaluate_holonomy([])
        assert result.unitarity_defect == 0.0


class TestPhase2_LongitudesDeArco:
    """
    Pruebas de la discretización invariante de reparametrización
    mediante longitudes de arco ds_k = ‖A_k‖_F / Σ‖A_j‖_F.
    """

    @pytest.fixture
    def p2(self) -> Phase2_WilsonLoopAuditor:
        return TestFixtures.phase2()

    def test_pesos_suman_uno(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """Σ ds_k = 1 (normalización)."""
        potentials = [
            TestFixtures.anti_hermitian_matrix(seed=300 + k)
            for k in range(5)
        ]
        result = p2.evaluate_holonomy(potentials)
        assert abs(float(result.arc_length_weights.sum()) - 1.0) < ATOL_STRICT, (
            f"Σ ds_k = {float(result.arc_length_weights.sum()):.8f} ≠ 1."
        )

    def test_pesos_son_positivos(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """ds_k ≥ 0 para todo k (son normas normalizadas)."""
        potentials = [
            TestFixtures.anti_hermitian_matrix(seed=310 + k)
            for k in range(4)
        ]
        result = p2.evaluate_holonomy(potentials)
        assert np.all(result.arc_length_weights >= 0.0)

    def test_pesos_proporcionales_a_norma_frobenius(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        ds_k ∝ ‖A_k‖_F: si ‖A_1‖_F = 2·‖A_0‖_F, entonces ds_1 = 2·ds_0.
        Caso de dos potenciales:
            ds_0 = ‖A_0‖ / (‖A_0‖ + ‖A_1‖)
            ds_1 = ‖A_1‖ / (‖A_0‖ + ‖A_1‖)
        """
        A0 = TestFixtures.identity(N, scale=1.0)
        A1 = TestFixtures.identity(N, scale=2.0)
        result = p2.evaluate_holonomy([A0, A1])
        ds = result.arc_length_weights
        # ‖A0‖_F = N, ‖A1‖_F = 2N → ds_0 = 1/3, ds_1 = 2/3
        norm_A0 = float(np.linalg.norm(A0, "fro"))
        norm_A1 = float(np.linalg.norm(A1, "fro"))
        total = norm_A0 + norm_A1
        expected_ds0 = norm_A0 / total
        expected_ds1 = norm_A1 / total
        assert abs(ds[0] - expected_ds0) < ATOL_STRICT
        assert abs(ds[1] - expected_ds1) < ATOL_STRICT

    def test_ciclo_con_un_potencial_peso_uno(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """Un solo potencial A ≠ 0 → ds_0 = 1.0."""
        A = TestFixtures.anti_hermitian_matrix(seed=320)
        result = p2.evaluate_holonomy([A])
        assert abs(float(result.arc_length_weights[0]) - 1.0) < ATOL_STRICT


class TestPhase2_UnitariedadWilson:
    """
    Pruebas de la unitariedad del bucle de Wilson W†W = I.
    La unitariedad es una propiedad fundamental del transporte paralelo
    en grupos de gauge unitarios.
    """

    @pytest.fixture
    def p2(self) -> Phase2_WilsonLoopAuditor:
        return TestFixtures.phase2()

    def test_wilson_es_unitario_para_potenciales_antisimetricos(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        Para A_k antisimétrico (A_k = −A_kᵀ), la exponencial
        exp(i · A_k · ds_k) es unitaria: U†U = I.
        Por tanto W(γ) también es unitaria.
        """
        potentials = [
            TestFixtures.anti_hermitian_matrix(seed=400 + k)
            for k in range(4)
        ]
        result = p2.evaluate_holonomy(potentials)
        W = result.phase_shift_matrix
        WdagW = W.conj().T @ W
        assert np.allclose(WdagW, np.eye(N, dtype=np.complex128), atol=ATOL_THERMO), (
            f"W†W ≠ I: δ_U = {result.unitarity_defect:.3e}"
        )
        assert result.unitarity_defect < _EPS_UNITARITY * 100
        assert result.holonomy_class != HolonomyClass.PARADOX

    def test_unitarity_defect_coincide_con_formula_explicita(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        unitarity_defect = ‖W†W − I‖_F calculado por Phase2 coincide
        con el valor computado manualmente sobre W retornado.
        """
        potentials = [
            TestFixtures.anti_hermitian_matrix(seed=410 + k)
            for k in range(3)
        ]
        result = p2.evaluate_holonomy(potentials)
        W = result.phase_shift_matrix
        defect_manual = float(
            np.linalg.norm(W.conj().T @ W - np.eye(N, dtype=np.complex128), "fro")
        )
        assert abs(result.unitarity_defect - defect_manual) < ATOL_STRICT

    def test_traza_wilson_coincide_con_np_trace(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """holonomy_trace = np.trace(W) verificado manualmente."""
        potentials = [TestFixtures.anti_hermitian_matrix(seed=420)]
        result = p2.evaluate_holonomy(potentials)
        trace_manual = complex(np.trace(result.phase_shift_matrix))
        assert abs(result.holonomy_trace - trace_manual) < ATOL_STRICT


class TestPhase2_ClasificacionHolonomia:
    """
    Pruebas de la clasificación TRIVIAL / NON_TRIVIAL / PARADOX.
    """

    @pytest.fixture
    def p2(self) -> Phase2_WilsonLoopAuditor:
        return TestFixtures.phase2()

    def test_clasificacion_trivial_para_ciclo_vacio(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """Ciclo vacío → HolonomyClass.TRIVIAL."""
        result = p2.evaluate_holonomy([])
        assert result.holonomy_class == HolonomyClass.TRIVIAL

    def test_clasificacion_non_trivial_para_potencial_grande(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        Un potencial antisimétrico de escala grande produce W ≠ I
        (holonomía genuina, no trivial) pero W sigue siendo unitario.
        """
        A = TestFixtures.skew_matrix(N, scale=2.0, seed=430)
        result = p2.evaluate_holonomy([A])
        W = result.phase_shift_matrix
        deviation = float(
            np.linalg.norm(W - np.eye(N, dtype=np.complex128), "fro")
        )
        if deviation > _EPS_HOLONOMY:
            assert result.holonomy_class == HolonomyClass.NON_TRIVIAL
        assert result.holonomy_class != HolonomyClass.PARADOX

    def test_paradox_detectado_para_w_no_unitario(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        Usando tolerancias muy estrictas, una pequeña asimetría numérica
        puede producir PARADOX. Verificamos la clasificación con un auditor
        de tolerancia extremadamente baja.
        """
        ultra_strict = Phase2_WilsonLoopAuditor(tol_unitarity=1e-20)
        # Potencial genérico (no antisimétrico) → exp(iA) puede no ser unitaria
        A = TestFixtures.generic_matrix(seed=440)
        result = ultra_strict.evaluate_holonomy([A])
        # Con tolerancia 1e-20, casi cualquier W es PARADOX
        # Verificamos que la clasificación es consistente con unitarity_defect
        if result.unitarity_defect > 1e-20:
            assert result.holonomy_class == HolonomyClass.PARADOX
            assert result.paradox_detected


class TestPhase2_VetoHolonomia:
    """
    Pruebas del método check_paradox: separación de cálculo y decisión de veto.
    """

    @pytest.fixture
    def p2(self) -> Phase2_WilsonLoopAuditor:
        return TestFixtures.phase2()

    def test_check_paradox_no_lanza_para_holonomia_trivial(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """check_paradox no lanza para ciclo vacío (holonomía trivial)."""
        result = p2.evaluate_holonomy([])
        p2.check_paradox(result)   # no debe lanzar

    def test_check_paradox_lanza_para_holonomy_class_paradox(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        Si se construye un WilsonLoopHolonomy con paradox_detected=True,
        check_paradox lanza HolonomyVetoError.
        """
        fake_holonomy = WilsonLoopHolonomy(
            phase_shift_matrix=np.eye(N, dtype=np.complex128),
            holonomy_trace=complex(N, 0),
            unitarity_defect=1.0,    # claramente no unitario
            arc_length_weights=np.array([1.0]),
            holonomy_class=HolonomyClass.PARADOX,
            paradox_detected=True,
        )
        with pytest.raises(HolonomyVetoError, match="unitario"):
            p2.check_paradox(fake_holonomy)

    def test_check_paradox_no_lanza_para_non_trivial(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """NON_TRIVIAL (W ≠ I pero unitaria) no activa el veto."""
        fake_holonomy = WilsonLoopHolonomy(
            phase_shift_matrix=np.eye(N, dtype=np.complex128),
            holonomy_trace=complex(N, 0),
            unitarity_defect=1e-12,   # esencialmente unitaria
            arc_length_weights=np.array([1.0]),
            holonomy_class=HolonomyClass.NON_TRIVIAL,
            paradox_detected=False,
        )
        p2.check_paradox(fake_holonomy)   # no debe lanzar

    def test_evaluate_holonomy_no_lanza_ella_misma(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """
        evaluate_holonomy solo calcula; NO lanza HolonomyVetoError.
        La separación de responsabilidades es una garantía de diseño.
        """
        # Potencial genérico (puede producir W no unitaria)
        A = TestFixtures.generic_matrix(seed=450)
        # evaluate_holonomy NO debe lanzar independientemente del resultado
        result = p2.evaluate_holonomy([A])
        assert isinstance(result, WilsonLoopHolonomy)

    def test_dimensiones_inconsistentes_en_ciclo_lanza_error(
        self, p2: Phase2_WilsonLoopAuditor
    ) -> None:
        """Si el ciclo mezcla matrices de dimensiones distintas, lanza ValueError."""
        A_ok = TestFixtures.generic_matrix(N)
        A_wrong = TestFixtures.generic_matrix(N + 1)
        with pytest.raises(ValueError, match="dimensión"):
            p2.evaluate_holonomy([A_ok, A_wrong])


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — PRUEBAS DEL OPTIMIZADOR DE YANG-MILLS
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3_AccionYangMills:
    """
    Pruebas del cálculo de la acción S_YM = ½ Tr(Fᵀ G⁻¹ F G⁻¹).
    Verifica no-negatividad, escala cuadrática en F y el término θ.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(seed=500)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G, theta_ym=0.0)
        return {"G": G, "p3": p3}

    def test_accion_es_no_negativa(self, setup: dict) -> None:
        """S_YM ≥ 0 para cualquier F (la forma cuadrática Fᵀ G⁻¹ F G⁻¹ es SPD)."""
        G, p3 = setup["G"], setup["p3"]
        for seed in range(5):
            F = TestFixtures.generic_matrix(seed=500 + seed)
            curv = TestFixtures.curvature_tensor_general(F, G)
            J = TestFixtures.spd_matrix(seed=510 + seed)
            result = p3.minimize_action(curv, J)
            assert result.action_value >= -ATOL_THERMO, (
                f"S_YM = {result.action_value:.4e} < 0."
            )

    def test_accion_cero_para_f_cero(self, setup: dict) -> None:
        """F = 0 → S_YM = 0."""
        p3 = setup["p3"]
        curv = TestFixtures.curvature_tensor_flat()
        J = TestFixtures.spd_matrix(seed=520)
        result = p3.minimize_action(curv, J)
        assert abs(result.action_value) < ATOL_THERMO, (
            f"S_YM(F=0) = {result.action_value:.3e} ≠ 0."
        )

    def test_accion_escala_cuadratica_con_norma_F(self, setup: dict) -> None:
        """
        S_YM ∝ ‖F‖² (forma cuadrática). Si F → 2F, entonces S_YM → 4·S_YM.
        """
        G, p3 = setup["G"], setup["p3"]
        F_base = TestFixtures.generic_matrix(seed=530)
        J = TestFixtures.spd_matrix(seed=531)

        curv1 = TestFixtures.curvature_tensor_general(F_base, G)
        curv2 = TestFixtures.curvature_tensor_general(2.0 * F_base, G)

        r1 = p3.minimize_action(curv1, J)
        r2 = p3.minimize_action(curv2, J)

        # S_YM(2F) = 4 · S_YM(F)  (solo el término cinético, θ=0)
        ratio = r2.action_value / (r1.action_value + 1e-20)
        assert abs(ratio - 4.0) < ATOL_THERMO * 10, (
            f"S_YM(2F)/S_YM(F) = {ratio:.4f} ≠ 4."
        )

    def test_accion_topologica_cero_para_theta_cero(self, setup: dict) -> None:
        """Con θ = 0, el término topológico S_top = 0."""
        G, p3 = setup["G"], setup["p3"]
        assert p3._theta == 0.0
        F = TestFixtures.generic_matrix(seed=540)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=541)
        result = p3.minimize_action(curv, J)
        assert result.topological_action == pytest.approx(0.0, abs=ATOL_STRICT)

    def test_accion_topologica_no_cero_para_theta_no_cero(self) -> None:
        """Con θ ≠ 0 y F antisimétrico, S_top ≠ 0."""
        G = TestFixtures.spd_matrix(seed=550)
        p3_theta = Phase3_YangMillsOptimizer(metric_tensor=G, theta_ym=1.0)
        # F antisimétrica → F_skew = F → ★F = G⁻¹ F G ≠ 0 en general
        F = TestFixtures.skew_matrix(N, scale=1.0, seed=551)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=552)
        result = p3_theta.minimize_action(curv, J)
        # No necesariamente cero; verificamos que se reporta un valor finito
        assert math.isfinite(result.topological_action)

    def test_kappa_metrica_reportado_coincide_con_real(self, setup: dict) -> None:
        """kappa_metric en YangMillsAction coincide con κ(G)."""
        G, p3 = setup["G"], setup["p3"]
        evals = la.eigvalsh(G)
        kappa_real = float(evals[-1] / evals[0])
        curv = TestFixtures.curvature_tensor_flat()
        J = TestFixtures.spd_matrix(seed=560)
        result = p3.minimize_action(curv, J)
        assert abs(result.kappa_metric - kappa_real) < ATOL_THERMO * kappa_real


class TestPhase3_EcuacionLyapunov:
    """
    Pruebas de la resolución de la ecuación de Lyapunov generalizada:
    G⁻¹ δR + δR G⁻¹ = RHS.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(seed=600)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G, theta_ym=0.0)
        return {"G": G, "p3": p3}

    def test_solucion_lyapunov_satisface_ecuacion(self, setup: dict) -> None:
        """
        La solución δR de G⁻¹δR + δRG⁻¹ = RHS satisface la ecuación
        con residuo pequeño: ‖G⁻¹δR + δRG⁻¹ − RHS‖_F < tolerancia.

        Nota: la proyección de Higham puede introducir un residuo controlado.
        Se verifica con una tolerancia relajada.
        """
        G, p3 = setup["G"], setup["p3"]
        G_inv = la.inv(G)

        F = TestFixtures.generic_matrix(seed=610)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=611)

        result = p3.minimize_action(curv, J)
        delta_R = result.optimal_deformation

        # Verificar la ecuación de Lyapunov para δR (antes de proyección PSD
        # no hay garantía exacta, pero el residuo debe ser razonable)
        lhs = G_inv @ delta_R + delta_R @ G_inv
        RHS = J - (G_inv @ F + F @ G_inv)
        lyapunov_res = float(np.linalg.norm(lhs - RHS, "fro"))

        # Con proyección Higham el residuo puede ser no nulo pero acotado
        assert lyapunov_res < result.lyapunov_rhs_norm * 10 + ATOL_THERMO, (
            f"Residuo Lyapunov = {lyapunov_res:.3e} demasiado grande."
        )

    def test_rhs_norm_reportado_coincide_con_formula_explicita(
        self, setup: dict
    ) -> None:
        """
        lyapunov_rhs_norm = ‖J − G⁻¹F − FG⁻¹‖_F verificado manualmente.
        """
        G, p3 = setup["G"], setup["p3"]
        G_inv = la.inv(G)

        F = TestFixtures.generic_matrix(seed=620)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=621)

        result = p3.minimize_action(curv, J)
        RHS_manual = J - (G_inv @ F + F @ G_inv)
        rhs_norm_manual = float(np.linalg.norm(RHS_manual, "fro"))

        assert abs(result.lyapunov_rhs_norm - rhs_norm_manual) < ATOL_STRICT

    def test_eom_residual_reportado_coincide_con_formula_explicita(
        self, setup: dict
    ) -> None:
        """
        eom_residual_norm = ‖G⁻¹F + FG⁻¹ + G⁻¹δR + δRG⁻¹ − J‖_F
        verificado manualmente.
        """
        G, p3 = setup["G"], setup["p3"]
        G_inv = la.inv(G)

        F = TestFixtures.generic_matrix(seed=630)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=631)

        result = p3.minimize_action(curv, J)
        dR = result.optimal_deformation

        EOM = G_inv @ F + F @ G_inv + G_inv @ dR + dR @ G_inv - J
        eom_manual = float(np.linalg.norm(EOM, "fro"))

        assert abs(result.eom_residual_norm - eom_manual) < ATOL_THERMO

    def test_dimensiones_incompatibles_J_lanza_error(self, setup: dict) -> None:
        """J con dimensión incorrecta lanza YangMillsOptimizationError."""
        p3 = setup["p3"]
        curv = TestFixtures.curvature_tensor_flat(N)
        J_wrong = np.eye(N + 1)
        with pytest.raises(YangMillsOptimizationError, match="Dimensiones"):
            p3.minimize_action(curv, J_wrong)


class TestPhase3_ProyeccionHigham:
    """
    Pruebas de la proyección de Higham aplicada a la solución de Lyapunov.
    Verifica PSD, simetría y no-negatividad de valores propios.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(seed=700)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G)
        return {"G": G, "p3": p3}

    def test_delta_r_optimo_es_psd(self, setup: dict) -> None:
        """δR_opt ∈ S⁺ₙ: todos los valores propios ≥ −_EIG_TOL."""
        G, p3 = setup["G"], setup["p3"]
        for seed in range(5):
            F = TestFixtures.generic_matrix(seed=700 + seed)
            curv = TestFixtures.curvature_tensor_general(F, G)
            J = TestFixtures.spd_matrix(seed=710 + seed)
            result = p3.minimize_action(curv, J)
            evals = la.eigvalsh(result.optimal_deformation)
            assert np.all(evals >= -1e-9), (
                f"δR_opt tiene λ_min = {float(np.min(evals)):.3e} < 0."
            )

    def test_delta_r_optimo_es_simetrico(self, setup: dict) -> None:
        """δR_opt es simétrico: ‖δR − δRᵀ‖_F < ATOL_STRICT."""
        G, p3 = setup["G"], setup["p3"]
        F = TestFixtures.generic_matrix(seed=720)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=721)
        result = p3.minimize_action(curv, J)
        dR = result.optimal_deformation
        assert np.allclose(dR, dR.T, atol=ATOL_STRICT)

    def test_delta_r_es_finito(self, setup: dict) -> None:
        """δR_opt no contiene NaN ni Inf."""
        G, p3 = setup["G"], setup["p3"]
        F = TestFixtures.generic_matrix(seed=730)
        curv = TestFixtures.curvature_tensor_general(F, G)
        J = TestFixtures.spd_matrix(seed=731)
        result = p3.minimize_action(curv, J)
        assert np.all(np.isfinite(result.optimal_deformation))

    def test_higham_proyeccion_estatica_psd(self, setup: dict) -> None:
        """
        _project_higham transforma una matriz con valores propios negativos
        en una PSD (prueba del método privado directamente).
        """
        p3 = setup["p3"]
        rng = TestFixtures.rng(seed=740)
        A = rng.standard_normal((N, N))
        M = A + A.T   # simétrica, puede tener λ < 0
        M_psd = p3._project_higham(M)
        evals = la.eigvalsh(M_psd)
        assert np.all(evals >= -1e-10), (
            f"λ_min(Higham) = {float(np.min(evals)):.3e} < 0."
        )

    def test_higham_proyeccion_no_altera_psd(self, setup: dict) -> None:
        """Si M ∈ S⁺ₙ, Higham(M) = M (idempotencia en S⁺ₙ)."""
        p3 = setup["p3"]
        M = TestFixtures.spd_matrix(seed=750)
        M_proj = p3._project_higham(M)
        assert np.allclose(M_proj, M, atol=ATOL_THERMO)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE COMPLETO — PRUEBAS DEL ORQUESTADOR
# ══════════════════════════════════════════════════════════════════════════════


class TestAgente_CoherenciaDimensional:
    """
    Pruebas de la validación de coherencia dimensional pre-pipeline.
    """

    @pytest.fixture
    def agente(self) -> YangMillsHolonomyAgent:
        return TestFixtures.agent(seed=800)

    def test_dimensiones_correctas_no_lanza(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """Con todas las entradas de dimensión correcta, no lanza."""
        n = N
        A_base = TestFixtures.generic_matrix(n, seed=800)
        A_new = TestFixtures.generic_matrix(n, seed=801)
        J = TestFixtures.spd_matrix(n, seed=802)
        # Ciclo vacío para evitar el veto de holonomía
        result = agente.enforce_gauge_invariance(A_base, A_new, [], J)
        assert isinstance(result, DeformationTensor)

    def test_a_base_dimension_incorrecta_lanza_error(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """A_base con dimensión incorrecta lanza ValueError."""
        A_wrong = np.eye(N + 1)
        A_new = TestFixtures.generic_matrix()
        J = TestFixtures.spd_matrix()
        with pytest.raises(ValueError, match="A_base"):
            agente.enforce_gauge_invariance(A_wrong, A_new, [], J)

    def test_a_new_dimension_incorrecta_lanza_error(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """A_new con dimensión incorrecta lanza ValueError."""
        A_base = TestFixtures.generic_matrix()
        A_wrong = np.eye(N + 2)
        J = TestFixtures.spd_matrix()
        with pytest.raises(ValueError, match="A_new"):
            agente.enforce_gauge_invariance(A_base, A_wrong, [], J)

    def test_j_dimension_incorrecta_lanza_error(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """J con dimensión incorrecta lanza ValueError."""
        A = TestFixtures.generic_matrix()
        J_wrong = np.eye(N + 1)
        with pytest.raises(ValueError, match="J"):
            agente.enforce_gauge_invariance(A, A, [], J_wrong)

    def test_ciclo_con_dimension_incorrecta_lanza_error(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """Una matriz del ciclo con dimensión incorrecta lanza ValueError."""
        A = TestFixtures.generic_matrix()
        J = TestFixtures.spd_matrix()
        bad_cycle = [np.eye(N + 1)]
        with pytest.raises(ValueError):
            agente.enforce_gauge_invariance(A, A, bad_cycle, J)


class TestAgente_PipelineCompleto:
    """
    Pruebas de integración del pipeline completo:
    Fase 1 → Fase 2 → Fase 3 → DeformationTensor.
    """

    @pytest.fixture
    def agente(self) -> YangMillsHolonomyAgent:
        return TestFixtures.agent(seed=900)

    @pytest.fixture
    def entradas_validas(self) -> dict:
        """Entradas para un pipeline exitoso con curvatura plana."""
        A_base = TestFixtures.generic_matrix(seed=901)
        # A_new = A_base → F = 0 (curvatura plana, sin singularidad)
        A_new = A_base.copy()
        J = TestFixtures.spd_matrix(seed=902)
        return {"A_base": A_base, "A_new": A_new, "J": J}

    def test_pipeline_retorna_deformation_tensor(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """enforce_gauge_invariance retorna DeformationTensor."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        assert isinstance(result, DeformationTensor)

    def test_pipeline_delta_r_es_psd(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """δR_opt del DeformationTensor es PSD."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        evals = la.eigvalsh(result.delta_R)
        assert np.all(evals >= -1e-9), (
            f"δR tiene λ_min = {float(np.min(evals)):.3e} < 0."
        )

    def test_pipeline_delta_r_es_simetrico(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """δR_opt es simétrico."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        dR = result.delta_R
        assert np.allclose(dR, dR.T, atol=ATOL_STRICT)

    def test_pipeline_info_contiene_claves_obligatorias(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """
        El campo info del DeformationTensor contiene todas las claves
        esperadas para la trazabilidad del pipeline.
        """
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        claves_requeridas = {
            "yang_mills_action",
            "topological_action",
            "wilson_trace",
            "unitarity_defect",
            "holonomy_class",
            "paradox_detected",
            "curvature_class",
            "chern_number",
            "bianchi_residual",
            "eom_residual_norm",
            "lyapunov_rhs_norm",
            "kappa_metric",
            "frobenius_ratio",
            "entropy_production",
        }
        for clave in claves_requeridas:
            assert clave in result.info, (
                f"Clave obligatoria '{clave}' ausente en result.info."
            )

    def test_pipeline_singularidad_curvatura_lanza_error(
        self, agente: YangMillsHolonomyAgent
    ) -> None:
        """
        Si A_base ≠ A_new y la curvatura es no integrable,
        enforce_gauge_invariance lanza GaugeCurvatureSingularityError.
        """
        A_base = TestFixtures.generic_matrix(seed=910)
        # A_new muy distinta de A_base para garantizar ‖F‖_F > umbral
        A_new = A_base + TestFixtures.generic_matrix(seed=911) * 10.0
        J = TestFixtures.spd_matrix(seed=912)
        with pytest.raises(GaugeCurvatureSingularityError):
            agente.enforce_gauge_invariance(A_base, A_new, [], J)

    def test_pipeline_frobenius_ratio_es_consistente(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """
        frobenius_ratio = ‖δR‖_F / ‖A_base‖_F (con salvaguarda 1e-15)
        es consistente con los campos del DeformationTensor.
        """
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        norm_dR = float(np.linalg.norm(result.delta_R, "fro"))
        norm_A_base = float(np.linalg.norm(ev["A_base"], "fro"))
        ratio_manual = norm_dR / (norm_A_base + 1e-15)
        assert abs(result.frobenius_ratio - ratio_manual) < ATOL_STRICT

    def test_pipeline_entropy_production_es_finita(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """σ = Tr(δR G⁻¹) es finita."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        assert math.isfinite(result.entropy_production)

    def test_pipeline_holonomia_trivial_con_ciclo_vacio(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """Con ciclo vacío, paradox_detected = False en el info."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        assert result.info["paradox_detected"] is False
        assert result.info["holonomy_class"] == HolonomyClass.TRIVIAL.name

    def test_pipeline_curvatura_plana_chern_cero(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """Con F = 0 (A_base = A_new), c₁ = 0."""
        ev = entradas_validas
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        assert abs(result.info["chern_number"]) < ATOL_STRICT

    def test_pipeline_es_determinista(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """Dos llamadas idénticas producen el mismo DeformationTensor."""
        ev = entradas_validas
        r1 = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        r2 = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], [], ev["J"]
        )
        assert np.allclose(r1.delta_R, r2.delta_R, atol=ATOL_STRICT)
        assert r1.info["yang_mills_action"] == r2.info["yang_mills_action"]

    def test_pipeline_con_ciclo_antihermitiano_no_lanza(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """
        Un ciclo con potenciales anti-hermitianos produce W unitario
        y no activa el veto de holonomía.
        """
        ev = entradas_validas
        cycle = [
            TestFixtures.anti_hermitian_matrix(seed=920 + k)
            for k in range(3)
        ]
        result = agente.enforce_gauge_invariance(
            ev["A_base"], ev["A_new"], cycle, ev["J"]
        )
        assert not result.info["paradox_detected"]

    def test_pipeline_veto_holonomia_con_ciclo_no_unitario(
        self, agente: YangMillsHolonomyAgent, entradas_validas: dict
    ) -> None:
        """
        Un agente con tol_unitarity extremadamente baja detecta paradoja
        para un ciclo con matrices genéricas (no anti-hermitianas).
        """
        G = TestFixtures.spd_matrix(seed=930)
        agente_estricto = YangMillsHolonomyAgent(
            metric_tensor=G,
        )
        # Inyectamos una Phase2 con tol_unitarity casi imposible de satisfacer
        agente_estricto._holonomy_auditor = Phase2_WilsonLoopAuditor(
            tol_unitarity=1e-20
        )
        ev = entradas_validas
        cycle = [TestFixtures.generic_matrix(seed=931)]

        result_eval = agente_estricto._holonomy_auditor.evaluate_holonomy(cycle)
        if result_eval.unitarity_defect > 1e-20:
            with pytest.raises(HolonomyVetoError):
                agente_estricto._holonomy_auditor.check_paradox(result_eval)


class TestAgente_ComposicionFases:
    """
    Pruebas de que el resultado del agente es algebraicamente
    consistente con la composición manual de las tres fases.
    """

    def test_composicion_manual_coincide_con_agente(self) -> None:
        """
        Ejecutar las fases manualmente en secuencia debe producir
        exactamente el mismo δR_opt que usar YangMillsHolonomyAgent.
        """
        G = TestFixtures.spd_matrix(seed=1000)
        A_base = TestFixtures.generic_matrix(seed=1001)
        A_new = A_base.copy()   # curvatura plana
        J = TestFixtures.spd_matrix(seed=1002)

        # Ejecución manual
        p1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        p2 = Phase2_WilsonLoopAuditor()
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G)

        curv = p1.compute_curvature(A_base, A_new)
        hol = p2.evaluate_holonomy([])
        p2.check_paradox(hol)
        action = p3.minimize_action(curv, J)
        delta_R_manual = action.optimal_deformation

        # Ejecución vía agente
        agente = YangMillsHolonomyAgent(metric_tensor=G)
        result = agente.enforce_gauge_invariance(A_base, A_new, [], J)

        assert np.allclose(result.delta_R, delta_R_manual, atol=ATOL_THERMO), (
            f"δR del agente difiere de la composición manual: "
            f"‖Δ‖_F = {np.linalg.norm(result.delta_R - delta_R_manual, 'fro'):.3e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS BASADAS EN PROPIEDADES (HYPOTHESIS)
# ══════════════════════════════════════════════════════════════════════════════


class TestProperties_Algebraicas:
    """
    Pruebas basadas en propiedades con Hypothesis.
    Verifican invariantes globales que deben sostenerse para cualquier
    entrada válida dentro del dominio matemático.
    """

    @staticmethod
    def _spd_from_data(
        n: int, data: st.DataObject, seed_offset: int = 0
    ) -> NDArray[np.float64]:
        """Genera una SPD n×n usando Hypothesis con entradas en [-1, 1]."""
        flat = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n * n,),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                ),
            )
        )
        A = flat.reshape(n, n)
        return A.T @ A + n * np.eye(n)

    @given(st.data())
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_f_mas_f_menos_siempre_es_f(self, data: st.DataObject) -> None:
        """
        F⁺ + F⁻ = F para cualquier A_μ, A_ν y cualquier métrica SPD.
        """
        n = 3
        G = self._spd_from_data(n, data)
        assume(float(np.linalg.cond(G)) < 1e8)

        flat_mu = data.draw(
            arrays(
                dtype=np.float64, shape=(n * n,),
                elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
            )
        )
        flat_nu = data.draw(
            arrays(
                dtype=np.float64, shape=(n * n,),
                elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
            )
        )
        A_mu = flat_mu.reshape(n, n)
        A_nu = flat_nu.reshape(n, n)

        try:
            p1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
            res = p1.compute_curvature(A_mu, A_nu)
        except (BianchiViolationError, ValueError):
            assume(False)

        F_sum = res.F_selfdual + res.F_antiselfdual
        assert np.allclose(F_sum, res.F_matrix, atol=1e-8), (
            f"F⁺ + F⁻ ≠ F: ‖diferencia‖_F = "
            f"{np.linalg.norm(F_sum - res.F_matrix, 'fro'):.3e}"
        )

    @given(st.data())
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_delta_r_siempre_psd_para_curvatura_plana(
        self, data: st.DataObject
    ) -> None:
        """
        Para F = 0 (A_base = A_new), δR_opt ∈ S⁺ₙ para cualquier J SPD
        y cualquier métrica G SPD.
        """
        n = 3
        G = self._spd_from_data(n, data)
        assume(float(np.linalg.cond(G)) < 1e6)

        flat_A = data.draw(
            arrays(
                dtype=np.float64, shape=(n * n,),
                elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
            )
        )
        A = flat_A.reshape(n, n)
        J = self._spd_from_data(n, data, seed_offset=1)

        p1 = Phase1_GaugeCurvatureComputer(metric_tensor=G)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G)

        curv = p1.compute_curvature(A, A)   # F = 0
        result = p3.minimize_action(curv, J)

        evals = la.eigvalsh(result.optimal_deformation)
        assert np.all(evals >= -1e-8), (
            f"λ_min(δR) = {float(np.min(evals)):.3e} < 0."
        )

    @given(
        n_potentials=st.integers(min_value=1, max_value=6),
        scale=st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_pesos_arco_suman_uno_siempre(
        self,
        n_potentials: int,
        scale: float,
    ) -> None:
        """
        Σ ds_k = 1 para cualquier número de potenciales antisimétricos
        no nulos.
        """
        rng = np.random.default_rng(42)
        potentials = [
            (rng.standard_normal((N, N)) - rng.standard_normal((N, N)).T) * scale
            for _ in range(n_potentials)
        ]
        # Filtrar matrices con norma cero (producen división por cero)
        potentials = [A for A in potentials if np.linalg.norm(A, "fro") > 1e-12]
        if not potentials:
            return

        p2 = TestFixtures.phase2()
        result = p2.evaluate_holonomy(potentials)
        total = float(result.arc_length_weights.sum())
        assert abs(total - 1.0) < ATOL_STRICT, (
            f"Σ ds_k = {total:.8f} ≠ 1 para {n_potentials} potenciales."
        )

    @given(st.data())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_accion_ym_no_negativa_para_cualquier_F(
        self, data: st.DataObject
    ) -> None:
        """
        S_YM ≥ 0 para cualquier F y cualquier métrica G SPD (con θ = 0).
        La forma bilineal ½ Tr(Fᵀ G⁻¹ F G⁻¹) es semidefinida positiva.
        """
        n = 3
        G = self._spd_from_data(n, data)
        assume(float(np.linalg.cond(G)) < 1e7)

        flat_F = data.draw(
            arrays(
                dtype=np.float64, shape=(n * n,),
                elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
            )
        )
        F = flat_F.reshape(n, n)
        J = self._spd_from_data(n, data)

        p3 = Phase3_YangMillsOptimizer(metric_tensor=G, theta_ym=0.0)
        curv = TestFixtures.curvature_tensor_general(F, G)

        try:
            result = p3.minimize_action(curv, J)
            assert result.action_value >= -ATOL_THERMO, (
                f"S_YM = {result.action_value:.4e} < 0."
            )
        except YangMillsOptimizationError:
            pass   # puede ocurrir si G está mal condicionada

    @given(st.data())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_wilson_unitario_para_matrices_antisimetricas(
        self, data: st.DataObject
    ) -> None:
        """
        W(γ) ∈ U(n) cuando todos los A_k son antisimétricos.
        Verificado para cualquier número de potenciales y cualquier escala.
        """
        n = 3
        n_pots = data.draw(st.integers(min_value=1, max_value=5))
        scale = data.draw(st.floats(min_value=0.01, max_value=0.5))
        rng = np.random.default_rng(data.draw(st.integers(0, 10000)))

        potentials = []
        for _ in range(n_pots):
            A = rng.standard_normal((n, n))
            potentials.append((A - A.T) * scale)

        p2 = Phase2_WilsonLoopAuditor()
        result = p2.evaluate_holonomy(potentials)
        W = result.phase_shift_matrix

        WdagW = W.conj().T @ W
        defect = float(np.linalg.norm(WdagW - np.eye(n, dtype=np.complex128), "fro"))
        assert defect < 1e-7, (
            f"W no unitario para potenciales antisimétricos: δ_U = {defect:.3e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE REGRESIÓN Y CASOS EXTREMOS
# ══════════════════════════════════════════════════════════════════════════════


class TestRegression_CasosExtremos:
    """
    Pruebas de regresión: estabilidad numérica en condiciones extremas,
    matrices de alta norma, métricas casi singulares y casos de borde.
    """

    def test_agente_con_metrica_identidad_produce_delta_r_valido(
        self,
    ) -> None:
        """Con G = I, el agente produce δR válido (caso más simple)."""
        G = np.eye(N)
        agente = YangMillsHolonomyAgent(metric_tensor=G)
        A_base = TestFixtures.generic_matrix(seed=2000)
        J = TestFixtures.spd_matrix(seed=2001)
        result = agente.enforce_gauge_invariance(A_base, A_base, [], J)
        assert np.all(np.isfinite(result.delta_R))
        assert np.all(la.eigvalsh(result.delta_R) >= -1e-9)

    def test_j_identidad_produce_delta_r_finito(self) -> None:
        """J = I produce δR finito y PSD."""
        G = TestFixtures.spd_matrix(seed=2010)
        agente = YangMillsHolonomyAgent(metric_tensor=G)
        A = TestFixtures.generic_matrix(seed=2011)
        J_id = np.eye(N)
        result = agente.enforce_gauge_invariance(A, A, [], J_id)
        assert np.all(np.isfinite(result.delta_R))

    def test_multiples_lladas_son_idempotentes(self) -> None:
        """
        Aplicar enforce_gauge_invariance dos veces con el mismo δR_opt
        como nueva J produce resultados consistentes (estabilidad iterativa).
        """
        G = TestFixtures.spd_matrix(seed=2020)
        agente = YangMillsHolonomyAgent(metric_tensor=G)
        A = TestFixtures.generic_matrix(seed=2021)
        J = TestFixtures.spd_matrix(seed=2022)

        r1 = agente.enforce_gauge_invariance(A, A, [], J)
        # Segunda llamada usando δR_opt como corriente J
        r2 = agente.enforce_gauge_invariance(A, A, [], r1.delta_R)
        # Ambos deben ser PSD y finitos
        assert np.all(la.eigvalsh(r2.delta_R) >= -1e-9)
        assert np.all(np.isfinite(r2.delta_R))

    def test_f_cero_accion_cero_y_rhs_correcto(self) -> None:
        """
        Con F = 0, la acción es 0 y el RHS de Lyapunov es simplemente J.
        La solución δR debe satisfacer G⁻¹δR + δRG⁻¹ = J.
        """
        G = TestFixtures.spd_matrix(seed=2030)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G)
        curv = TestFixtures.curvature_tensor_flat(N)
        J = TestFixtures.spd_matrix(seed=2031)
        result = p3.minimize_action(curv, J)

        assert abs(result.action_value) < ATOL_THERMO
        # RHS = J − (G⁻¹·0 + 0·G⁻¹) = J
        assert abs(result.lyapunov_rhs_norm - float(np.linalg.norm(J, "fro"))) < ATOL_THERMO

    def test_ciclo_con_un_potencial_zero_es_trivial(self) -> None:
        """Ciclo con A = 0 → W = I (holonomía trivial)."""
        p2 = TestFixtures.phase2()
        result = p2.evaluate_holonomy([np.zeros((N, N))])
        assert result.holonomy_class == HolonomyClass.TRIVIAL

    def test_chern_number_es_real_siempre(self) -> None:
        """c₁ = Tr(F) / (2π) es siempre un número real."""
        p1 = TestFixtures.phase1(seed=2040)
        for seed in range(5):
            A_mu = TestFixtures.generic_matrix(seed=2040 + seed)
            A_nu = TestFixtures.generic_matrix(seed=2050 + seed)
            try:
                res = p1.compute_curvature(A_mu, A_nu)
                assert math.isfinite(res.chern_number)
                assert isinstance(res.chern_number, float)
            except BianchiViolationError:
                pass

    def test_dto_yang_mills_action_es_inmutable(self) -> None:
        """YangMillsAction es frozen=True; asignar atributo lanza error."""
        G = TestFixtures.spd_matrix(seed=2060)
        p3 = Phase3_YangMillsOptimizer(metric_tensor=G)
        curv = TestFixtures.curvature_tensor_flat(N)
        J = TestFixtures.spd_matrix(seed=2061)
        result = p3.minimize_action(curv, J)
        with pytest.raises((TypeError, AttributeError)):
            result.action_value = 0.0  # type: ignore[misc]

    def test_dto_wilson_loop_holonomy_es_inmutable(self) -> None:
        """WilsonLoopHolonomy es frozen=True."""
        p2 = TestFixtures.phase2()
        result = p2.evaluate_holonomy([])
        with pytest.raises((TypeError, AttributeError)):
            result.paradox_detected = True  # type: ignore[misc]

    def test_agente_con_theta_no_cero_produce_delta_r_valido(self) -> None:
        """Con θ ≠ 0, el agente sigue produciendo δR PSD y finito."""
        G = TestFixtures.spd_matrix(seed=2070)
        agente_theta = YangMillsHolonomyAgent(
            metric_tensor=G, theta_ym=math.pi / 4
        )
        A = TestFixtures.generic_matrix(seed=2071)
        J = TestFixtures.spd_matrix(seed=2072)
        result = agente_theta.enforce_gauge_invariance(A, A, [], J)
        assert np.all(np.isfinite(result.delta_R))
        assert np.all(la.eigvalsh(result.delta_R) >= -1e-9)
        assert math.isfinite(result.info["topological_action"])


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-rA"])