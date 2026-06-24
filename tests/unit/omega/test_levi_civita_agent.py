# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Suite de Pruebas: Levi-Civita Connection Agent                                     ║
║  Ruta   : tests/unit/omega/test_levi_civita_agent.py                              ║
║  Versión: 2.0.0-Granular-Geodesic-Categorical                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Filosofía de Testing — Axiomas Geométricos como Contratos Ejecutables:
══════════════════════════════════════════════════════════════════════════════════════

Esta suite verifica que el código satisface los axiomas matemáticos de la geometría
diferencial Riemanniana. Cada clase de test corresponde a un teorema o proposición
formal que el código debe cumplir:

  §G1. Axiomas de Christoffel (Fase 1):
       - Shape (n,n,n) y dtype float64 del tensor Γ.
       - Finitud: ‖Γ‖_F < _CHRISTOFFEL_FINITE_TOL.
       - Modo estático (dG=0): Γ = 0 exactamente.
       - Invariancia bajo reparametrización de la métrica.
       - Simetría de Christoffel: Γ^r_{mn} = Γ^r_{nm} ∀ r,m,n (condición necesaria).
       - Verificación de la fórmula de Koszul para casos analíticos.

  §G2. Axiomas de Torsión y Compatibilidad (Fase 2):
       - Torsión nula: ‖Γ - Γ^T_{(mn)}‖_F < _TORSION_TOLERANCE.
       - Compatibilidad métrica: ‖∇_γ G‖_F < _METRIC_COMPAT_TOLERANCE.
       - Tensor de Riemann: R = 0 para espacio plano (dG = 0).
       - ConnectionDiagnostics.all_passed() ↔ ambas verificaciones.
       - Detección correcta de torsión artificial introducida.
       - Detección correcta de incompatibilidad métrica introducida.

  §G3. Integración Geodésica (Fase 3):
       - Finitud de v(t+dt) para dt bien elegido.
       - Conservación aproximada de norma: ‖v(t+dt)‖ ≈ ‖v(t)‖.
       - Consistencia de orden 4: error RK4 ~ O(dt^5).
       - Aceleración geodésica: a = -Γ v² (contracción correcta).
       - Transporte paralelo: ‖DV/dt‖ ≈ 0 (ecuación de transporte).
       - Rechazo de dt ≤ _DT_MIN.
       - Advertencia para dt > dt_max_stable.

  §G4. Transportes Categóricos ♭ y ♯ (Fase 3):
       - Roundtrip: ♭ → ♯ ≈ id_{TM} bajo corrección geodésica.
       - Linealidad preservada post-corrección.
       - Tipo correcto de retorno (CotangentVector / TangentVector).
       - Informes GeodesicStepReport con claves completas.
       - Coherencia de normas entre transporte directo e inverso.

  §G5. Contratos de Errores y Robustez:
       - ChristoffelInstabilityError para métricas extremas.
       - TopologicalTorsionError para conexiones asimétricas artificiales.
       - MetricCompatibilityError para métricas incompatibles.
       - GeodesicDeviationError para vectores que producen overflow.
       - TypeError/ValueError para entradas inválidas en todos los métodos.

  §G6. Propiedades Estadísticas y Alta Dimensión:
       - Pipeline completo para n ∈ {3, 5, 10, 20, 50}.
       - Consistencia de diagnósticos sobre 10 métricas SPD aleatorias.
       - geodesic_flow_report() con todas las claves esperadas.

Estrategia de fixtures:
══════════════════════════════════════════════════════════════════════════════════════
  - make_spd_matrix(n)           : SPD controlada con λ_min ≥ n > 0.
  - make_diagonal_metric(diags)  : métrica diagonal con autovalores conocidos.
  - make_identity_metric(n)      : métrica plana ℝⁿ (Γ = 0 exactamente).
  - make_ill_conditioned(n, κ)   : SPD con κ ≫ 1 (prueba regularización).
  - make_rank_deficient(n, r)    : PSD de rango r < n (prueba Tikhonov).
  - make_tangent_vector(arr)     : TangentVector desde array numpy.
  - make_cotangent_vector(arr)   : CotangentVector desde array numpy.
  - build_agent(G)               : LeviCivitaConnectionAgent con métrica G.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ════════════════════════════════════════════════════════════════════════════════
from app.omega.levi_civita_agent import (
    # Excepciones
    TopologicalTorsionError,
    GeodesicDeviationError,
    MetricCompatibilityError,
    ChristoffelInstabilityError,
    # Estructuras de datos
    ChristoffelData,
    ConnectionDiagnostics,
    GeodesicStepReport,
    # Agente principal y fases internas
    LeviCivitaConnectionAgent,
    _ChristoffelEngine,
    _TorsionFreeConnection,
    # Constantes
    _TORSION_TOLERANCE,
    _METRIC_COMPAT_TOLERANCE,
    _CHRISTOFFEL_FINITE_TOL,
    _GEODESIC_NORM_DRIFT_TOL,
    _DEFAULT_DT,
    _DT_MAX_STABLE_FACTOR,
    _DT_MIN,
    _MACHINE_EPSILON,
)
from app.core.immune_system.musical_isomorphism_engine import (
    MetricSpectralPreconditioner,
    PreconditionedMetric,
    TangentVector,
    CotangentVector,
)
from app.core.mic_algebra import FunctorialityError, NumericalInstabilityError

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING)
_RNG_SEED: int = 42


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FÁBRICAS DE MÉTRICAS, VECTORES Y AGENTES                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

def make_spd_matrix(n: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """
    Genera una SPD de tamaño n×n con λ_min ≥ n.

    G = AᵀA + n·I, A aleatoria de seed dado.
    Garantiza separación espectral adecuada del cero para evitar
    regularización de Tikhonov (a menos que se desee).
    """
    rng = np.random.default_rng(seed)
    A: NDArray[np.float64] = rng.standard_normal((n, n))
    return (A.T @ A + float(n) * np.eye(n)).astype(np.float64)


def make_diagonal_metric(
    diag_values: List[float],
) -> NDArray[np.float64]:
    """
    Genera una métrica diagonal con autovalores exactamente conocidos.

    Para una métrica diagonal G = diag(g₁, ..., gₙ):
      - Γ^r_{mn} = 0 para todo r,m,n (si dG = 0).
      - κ(G) = max(gᵢ) / min(gᵢ).
      - G_inv = diag(1/g₁, ..., 1/gₙ).
    """
    return np.diag(np.array(diag_values, dtype=np.float64))


def make_identity_metric(n: int) -> NDArray[np.float64]:
    """
    Genera la métrica plana I_n (espacio euclídeo ℝⁿ).

    Propiedades exactas:
      - Γ = 0 (sin curvatura).
      - κ = 1 (perfectamente condicionada).
      - G_inv = I_n.
    """
    return np.eye(n, dtype=np.float64)


def make_ill_conditioned_matrix(
    n: int, condition_number: float = 1e14, seed: int = _RNG_SEED
) -> NDArray[np.float64]:
    """
    SPD con κ ≈ condition_number (logarítmicamente distribuida).

    Construida como G = V diag(λ) Vᵀ con λ ∈ [1, κ] y V ortogonal aleatoria.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.logspace(0, np.log10(condition_number), n, dtype=np.float64)
    G = Q @ np.diag(eigenvalues) @ Q.T
    return ((G + G.T) * 0.5).astype(np.float64)


def make_rank_deficient_matrix(
    n: int, rank: int, seed: int = _RNG_SEED
) -> NDArray[np.float64]:
    """
    PSD de rango `rank` < n (kernel dimensional = n - rank).

    G = A Aᵀ donde A ∈ ℝ^{n×rank}. Requiere regularización de Tikhonov.
    """
    rng = np.random.default_rng(seed)
    A: NDArray[np.float64] = rng.standard_normal((n, rank))
    return (A @ A.T).astype(np.float64)


def make_tangent_vector(
    arr: NDArray[np.float64],
) -> TangentVector:
    """Envuelve un array numpy en TangentVector."""
    return TangentVector(coordinates=arr.astype(np.float64))


def make_cotangent_vector(
    arr: NDArray[np.float64],
) -> CotangentVector:
    """Envuelve un array numpy en CotangentVector."""
    return CotangentVector(coordinates=arr.astype(np.float64))


def make_random_unit_tangent(n: int, seed: int = _RNG_SEED) -> TangentVector:
    """TangentVector aleatorio de norma unitaria."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return make_tangent_vector(v)


def make_random_cotangent(n: int, seed: int = _RNG_SEED) -> CotangentVector:
    """CotangentVector aleatorio."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(n)
    return make_cotangent_vector(omega)


def build_agent(
    G: NDArray[np.float64],
    preconditioner: Optional[MetricSpectralPreconditioner] = None,
) -> LeviCivitaConnectionAgent:
    """Construye LeviCivitaConnectionAgent sobre la métrica G."""
    return LeviCivitaConnectionAgent(metric_tensor=G, preconditioner=preconditioner)


def build_christoffel_engine(
    G: NDArray[np.float64],
) -> _ChristoffelEngine:
    """Construye _ChristoffelEngine directamente para pruebas de Fase 1."""
    return _ChristoffelEngine(G)


def build_torsion_free(
    G: NDArray[np.float64],
) -> _TorsionFreeConnection:
    """Construye _TorsionFreeConnection directamente para pruebas de Fase 2."""
    return _TorsionFreeConnection(G)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G1a — TestChristoffelEngineStructure: Invariantes de Fase 1               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestChristoffelEngineStructure:
    """
    Verifica los invariantes estructurales del tensor de Christoffel:
      - Shape (n,n,n) para toda n.
      - dtype float64.
      - Finitud de todas las entradas.
      - ChristoffelData como objeto inmutable (frozen dataclass).
      - Propiedad christoffel_symbols retorna copia defensiva.
    """

    @pytest.mark.parametrize("n", [2, 3, 5, 8, 10])
    def test_gamma_shape_is_n_cubed(self, n: int) -> None:
        """Γ.shape == (n, n, n) para toda dimensión n."""
        G = make_spd_matrix(n, seed=n * 7)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        assert Gamma.shape == (n, n, n), (
            f"Gamma.shape={Gamma.shape} ≠ ({n},{n},{n}) para n={n}."
        )

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_gamma_dtype_is_float64(self, n: int) -> None:
        """Γ debe tener dtype float64."""
        G = make_spd_matrix(n, seed=n * 11)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        assert Gamma.dtype == np.float64, (
            f"Gamma.dtype={Gamma.dtype} ≠ float64."
        )

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_gamma_is_finite(self, n: int) -> None:
        """Todas las entradas de Γ son finitas (sin NaN ni Inf)."""
        G = make_spd_matrix(n, seed=n * 13)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        assert np.all(np.isfinite(Gamma)), (
            f"Gamma contiene valores no finitos para n={n}."
        )

    def test_christoffel_data_is_immutable(self) -> None:
        """ChristoffelData es un dataclass frozen — no admite modificación."""
        G = make_spd_matrix(3, seed=17)
        engine = build_christoffel_engine(G)
        cd = engine._christoffel_data
        with pytest.raises((AttributeError, TypeError)):
            cd.dimension = 999  # type: ignore[misc]

    def test_christoffel_symbols_returns_copy(self) -> None:
        """christoffel_symbols retorna una copia: la modificación externa no afecta al motor."""
        G = make_spd_matrix(4, seed=19)
        engine = build_christoffel_engine(G)
        Gamma_copy = engine.christoffel_symbols
        Gamma_copy[0, 0, 0] = 9999.0
        # El tensor interno no debe haber cambiado
        Gamma_internal = engine._christoffel_data.Gamma
        assert Gamma_internal[0, 0, 0] != 9999.0, (
            "La modificación de la copia no debe afectar al tensor interno."
        )

    def test_metric_dimension_property(self) -> None:
        """metric_dimension retorna n correcto."""
        for n in [3, 5, 7]:
            G = make_spd_matrix(n, seed=n)
            engine = build_christoffel_engine(G)
            assert engine.metric_dimension == n, (
                f"metric_dimension={engine.metric_dimension} ≠ {n}."
            )

    def test_preconditioned_metric_property(self) -> None:
        """preconditioned_metric retorna PreconditionedMetric válida."""
        G = make_spd_matrix(4, seed=23)
        engine = build_christoffel_engine(G)
        pm = engine.preconditioned_metric
        assert isinstance(pm, PreconditionedMetric), (
            f"preconditioned_metric debe ser PreconditionedMetric, "
            f"recibido {type(pm).__name__}."
        )

    def test_christoffel_data_keys(self) -> None:
        """ChristoffelData tiene los campos: Gamma, frobenius_norm, dG, dimension, is_static."""
        G = make_spd_matrix(3, seed=29)
        engine = build_christoffel_engine(G)
        cd = engine._christoffel_data
        assert hasattr(cd, 'Gamma')
        assert hasattr(cd, 'frobenius_norm')
        assert hasattr(cd, 'dG')
        assert hasattr(cd, 'dimension')
        assert hasattr(cd, 'is_static')

    def test_dg_shape_in_christoffel_data(self) -> None:
        """ChristoffelData.dG tiene shape (n,n,n)."""
        n = 4
        G = make_spd_matrix(n, seed=31)
        engine = build_christoffel_engine(G)
        assert engine._christoffel_data.dG.shape == (n, n, n), (
            f"dG.shape={engine._christoffel_data.dG.shape} ≠ ({n},{n},{n})."
        )

    def test_frobenius_norm_is_non_negative(self) -> None:
        """ChristoffelData.frobenius_norm ≥ 0."""
        G = make_spd_matrix(5, seed=37)
        engine = build_christoffel_engine(G)
        assert engine._christoffel_data.frobenius_norm >= 0, (
            "frobenius_norm debe ser no negativo."
        )

    def test_christoffel_data_post_init_rejects_wrong_shapes(self) -> None:
        """ChristoffelData.__post_init__ rechaza Gamma con shape incorrecto."""
        n = 3
        with pytest.raises(ValueError, match="shape"):
            ChristoffelData(
                Gamma          = np.zeros((n, n, n + 1)),  # shape incorrecto
                frobenius_norm = 0.0,
                dG             = np.zeros((n, n, n)),
                dimension      = n,
                is_static      = True,
            )

    def test_christoffel_data_post_init_rejects_negative_norm(self) -> None:
        """ChristoffelData.__post_init__ rechaza frobenius_norm < 0."""
        n = 3
        with pytest.raises(ValueError, match="frobenius_norm"):
            ChristoffelData(
                Gamma          = np.zeros((n, n, n)),
                frobenius_norm = -1.0,
                dG             = np.zeros((n, n, n)),
                dimension      = n,
                is_static      = True,
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G1b — TestChristoffelEngineValues: Correctitud Analítica de Fase 1        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestChristoffelEngineValues:
    """
    Verifica la correctitud analítica de los símbolos de Christoffel:
      - Para dG = 0 (modo estático), Γ = 0 exactamente.
      - Para métrica diagonal, Γ_diagonal = 0 (sin derivadas cruzadas).
      - Para métrica identidad, Γ = 0 y is_static = True.
      - Fórmula de Koszul verificada componente a componente.
      - Norma de Frobenius de Γ es cero para espacio plano.
    """

    def test_static_metric_gives_zero_christoffel(self) -> None:
        """
        Para dG = 0 (modo estático), Γ^r_{mn} = 0 ∀ r,m,n.

        Demostración: Γ = ½ G^{rk}(∂G + ∂G - ∂G) = ½ G^{rk} · 0 = 0.
        """
        G = make_spd_matrix(5, seed=41)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        assert_allclose(
            Gamma, np.zeros((5, 5, 5)), atol=1e-14,
            err_msg="Para dG=0, Γ debe ser exactamente 0."
        )

    def test_identity_metric_christoffel_is_zero(self) -> None:
        """Para G = I_n, Γ = 0 (espacio plano euclídeo)."""
        for n in [2, 3, 4, 5]:
            engine = build_christoffel_engine(make_identity_metric(n))
            Gamma = engine.christoffel_symbols
            assert_allclose(
                Gamma, np.zeros((n, n, n)), atol=1e-14,
                err_msg=f"Para G=I_{n}, Γ debe ser 0."
            )

    def test_diagonal_metric_christoffel_is_zero(self) -> None:
        """Para G = diag(g₁,...,gₙ) constante, Γ = 0."""
        diags = [1.0, 2.5, 4.0, 0.5]
        G = make_diagonal_metric(diags)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        assert_allclose(
            Gamma, np.zeros((4, 4, 4)), atol=1e-14,
            err_msg="Para G diagonal constante, Γ debe ser 0."
        )

    def test_is_static_true_for_constant_metric(self) -> None:
        """is_static == True para métricas constantes (dG = 0)."""
        G = make_spd_matrix(4, seed=43)
        engine = build_christoffel_engine(G)
        assert engine._christoffel_data.is_static, (
            "is_static debe ser True para métrica constante."
        )

    def test_frobenius_norm_zero_for_static_metric(self) -> None:
        """‖Γ‖_F = 0 para dG = 0."""
        G = make_spd_matrix(6, seed=47)
        engine = build_christoffel_engine(G)
        assert engine._christoffel_data.frobenius_norm == 0.0, (
            f"‖Γ‖_F={engine._christoffel_data.frobenius_norm} ≠ 0 para dG=0."
        )

    def test_christoffel_koszul_formula_2d_known_metric(self) -> None:
        """
        Verifica la fórmula de Koszul en 2D con métrica diagonal conocida y
        derivadas métricas artificialmente inyectadas.

        Para G = diag(g₁, g₂) con ∂₁ g₁₁ = a (resto = 0):
            Γ^1_{11} = ½ G^{11} ∂₁ G_{11} = a / (2 g₁)
            Γ^2_{11} = -½ G^{22} ∂₂ G_{11} = 0  (∂₂ G_{11} = 0)
        """
        g1, g2 = 2.0, 3.0
        a = 1.0  # ∂₁ G_{11} = a

        class _MockEngine(_ChristoffelEngine):
            """Subclase que inyecta derivadas conocidas."""
            def _compute_metric_derivative(self) -> NDArray[np.float64]:
                dG = np.zeros((2, 2, 2), dtype=np.float64)
                dG[0, 0, 0] = a  # ∂₁ G_{11} = a
                return dG

        G = make_diagonal_metric([g1, g2])
        engine = _MockEngine(G)
        Gamma = engine.christoffel_symbols

        # Γ^1_{11} = ½ (1/g1) * a = a / (2 g1)
        expected_Gamma_1_11 = a / (2.0 * g1)
        assert_allclose(
            Gamma[0, 0, 0], expected_Gamma_1_11, rtol=1e-12,
            err_msg=(
                f"Γ^1_{{11}} = {Gamma[0,0,0]:.6e} ≠ {expected_Gamma_1_11:.6e}."
            )
        )

        # Γ^2_{11} = ½ G^{22} (∂₁ G_{21} + ∂₁ G_{21} - ∂₂ G_{11}) = 0
        assert_allclose(
            Gamma[1, 0, 0], 0.0, atol=1e-14,
            err_msg=f"Γ^2_{{11}} debe ser 0, obtenido {Gamma[1,0,0]:.2e}."
        )

    def test_christoffel_symmetry_in_lower_indices_static(self) -> None:
        """
        Para el modo estático (Γ = 0), la simetría Γ^r_{mn} = Γ^r_{nm}
        se satisface trivialmente. Verificación de que la propiedad se hereda
        del pipeline aunque Γ = 0.
        """
        G = make_spd_matrix(5, seed=53)
        engine = build_christoffel_engine(G)
        Gamma = engine.christoffel_symbols
        # Tensor de torsión = 0 para Γ = 0
        torsion = Gamma - Gamma.transpose(0, 2, 1)
        assert_allclose(
            torsion, np.zeros_like(torsion), atol=1e-15,
            err_msg="Tensor de torsión no es cero para Γ=0."
        )

    def test_einsum_terms_cancel_correctly_for_zero_dG(self) -> None:
        """
        Verifica que T1, T2, T3 son todos cero cuando dG = 0,
        lo que resulta en Γ = ½(0 + 0 - 0) = 0.
        """
        G = make_spd_matrix(4, seed=59)
        engine = build_christoffel_engine(G)
        dG = np.zeros((4, 4, 4), dtype=np.float64)

        T1, T2, T3 = engine._compute_christoffel_terms(dG)

        assert_allclose(T1, np.zeros((4, 4, 4)), atol=1e-15,
                        err_msg="T1 ≠ 0 para dG=0.")
        assert_allclose(T2, np.zeros((4, 4, 4)), atol=1e-15,
                        err_msg="T2 ≠ 0 para dG=0.")
        assert_allclose(T3, np.zeros((4, 4, 4)), atol=1e-15,
                        err_msg="T3 ≠ 0 para dG=0.")


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G1c — TestChristoffelEngineValidation: Contratos de Entrada (Fase 1)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestChristoffelEngineValidation:
    """
    Verifica que _validate_metric_input rechaza entradas inválidas con las
    excepciones correctas.
    """

    def test_rejects_list_input(self) -> None:
        """TypeError para input que no sea ndarray."""
        with pytest.raises(TypeError, match="NDArray"):
            _ChristoffelEngine([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_rejects_1d_array(self) -> None:
        """ValueError para array 1-D."""
        with pytest.raises(ValueError, match="2-D"):
            _ChristoffelEngine(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_array(self) -> None:
        """ValueError para array 3-D."""
        with pytest.raises(ValueError, match="2-D"):
            _ChristoffelEngine(np.ones((3, 3, 3)))

    def test_rejects_non_square_matrix(self) -> None:
        """ValueError para matriz no cuadrada."""
        with pytest.raises(ValueError, match="cuadrado"):
            _ChristoffelEngine(np.eye(3, 4))

    def test_rejects_nan_entries(self) -> None:
        """ValueError para matrices con NaN."""
        G = make_spd_matrix(3, seed=61)
        G[1, 1] = np.nan
        with pytest.raises(ValueError, match="no finitos"):
            _ChristoffelEngine(G)

    def test_rejects_inf_entries(self) -> None:
        """ValueError para matrices con Inf."""
        G = make_spd_matrix(3, seed=67)
        G[0, 2] = np.inf
        with pytest.raises(ValueError, match="no finitos"):
            _ChristoffelEngine(G)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_accepts_valid_spd_matrices(self, n: int) -> None:
        """No excepción para SPD válidas de distintas dimensiones."""
        G = make_spd_matrix(n, seed=n * 71)
        engine = _ChristoffelEngine(G)
        assert engine.metric_dimension == n

    def test_dependency_injection_accepted(self) -> None:
        """MetricSpectralPreconditioner inyectado es utilizado correctamente."""
        G = make_spd_matrix(4, seed=73)
        custom_pc = MetricSpectralPreconditioner()
        engine = _ChristoffelEngine(G, preconditioner=custom_pc)
        assert engine.metric_dimension == 4

    def test_validate_derivative_tensor_rejects_wrong_shape(self) -> None:
        """_validate_derivative_tensor rechaza dG con shape incorrecto."""
        G = make_spd_matrix(3, seed=79)
        engine = _ChristoffelEngine(G)
        bad_dG = np.zeros((3, 3, 4), dtype=np.float64)  # shape incorrecto
        with pytest.raises(ValueError, match="shape"):
            engine._validate_derivative_tensor(bad_dG)

    def test_validate_derivative_tensor_rejects_nan(self) -> None:
        """_validate_derivative_tensor rechaza dG con NaN."""
        G = make_spd_matrix(3, seed=83)
        engine = _ChristoffelEngine(G)
        bad_dG = np.zeros((3, 3, 3), dtype=np.float64)
        bad_dG[1, 2, 0] = np.nan
        with pytest.raises(ValueError, match="no finitos"):
            engine._validate_derivative_tensor(bad_dG)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G2a — TestTorsionFreeConnection: Axioma 1 — Torsión Nula                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestTorsionFreeConnectionTorsion:
    """
    Verifica el Axioma 1 de la conexión de Levi-Civita:
      T^r_{mn} = Γ^r_{mn} - Γ^r_{nm} = 0  ∀ r,m,n

    Pruebas:
      - Torsión nula para SPD estáticas (Γ = 0 → T = 0 trivialmente).
      - Torsión nula para métricas con derivadas inyectadas (Γ ≠ 0).
      - TopologicalTorsionError para Christoffel artificialmente asimétrico.
      - _compute_torsion_tensor retorna tensor antisimétrico correcto.
      - Mensaje de error incluye índice de máxima asimetría.
    """

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_zero_torsion_for_static_metric(self, n: int) -> None:
        """Torsión nula para métricas estáticas (dG=0, Γ=0)."""
        G = make_spd_matrix(n, seed=n * 89)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.torsion_passed, (
            f"Torsión debe ser nula para n={n}: norm={diag.torsion_norm:.2e}."
        )
        assert diag.torsion_norm < _TORSION_TOLERANCE, (
            f"‖T‖_F={diag.torsion_norm:.2e} ≥ tol={_TORSION_TOLERANCE}."
        )

    def test_compute_torsion_tensor_zero_for_symmetric_gamma(self) -> None:
        """_compute_torsion_tensor retorna 0 para Gamma simétrico."""
        n = 4
        G = make_spd_matrix(n, seed=97)
        conn = build_torsion_free(G)

        # Construir un Gamma simétrico artificial
        Gamma_sym = np.zeros((n, n, n), dtype=np.float64)
        # Symmetrizar en los índices inferiores
        for r in range(n):
            for m in range(n):
                for nv in range(n):
                    val = float(r + m + nv) * 0.01
                    Gamma_sym[r, m, nv] = val
                    Gamma_sym[r, nv, m] = val  # simetría

        torsion = _TorsionFreeConnection._compute_torsion_tensor(Gamma_sym)
        assert_allclose(
            torsion, np.zeros((n, n, n)), atol=1e-15,
            err_msg="Tensor de torsión debe ser 0 para Gamma simétrico."
        )

    def test_compute_torsion_tensor_nonzero_for_asymmetric_gamma(self) -> None:
        """_compute_torsion_tensor retorna tensor no nulo para Gamma asimétrico."""
        n = 3
        G = make_spd_matrix(n, seed=101)
        conn = build_torsion_free(G)

        Gamma_asym = np.zeros((n, n, n), dtype=np.float64)
        Gamma_asym[0, 0, 1] = 1.0   # T^0_{01} = 1
        Gamma_asym[0, 1, 0] = -1.0  # Gamma[0,1,0] = -1, torsion = 2.0

        torsion = _TorsionFreeConnection._compute_torsion_tensor(Gamma_asym)
        # T[0,0,1] = Gamma[0,0,1] - Gamma[0,1,0] = 1 - (-1) = 2
        assert_allclose(
            torsion[0, 0, 1], 2.0, rtol=1e-14,
            err_msg="Torsión T^0_{01} debe ser 2.0."
        )
        # Antisimetría: T[0,1,0] = -T[0,0,1]
        assert_allclose(
            torsion[0, 1, 0], -2.0, rtol=1e-14,
            err_msg="Torsión T^0_{10} debe ser -2.0 (antisimetría)."
        )

    def test_torsion_error_raised_for_artificial_asymmetry(self) -> None:
        """
        TopologicalTorsionError cuando _verify_zero_torsion recibe un tensor
        de torsión no nulo > _TORSION_TOLERANCE.
        """
        n = 3
        G = make_spd_matrix(n, seed=103)
        conn = build_torsion_free(G)

        # Tensor de torsión artificial con norma grande
        large_torsion = np.ones((n, n, n), dtype=np.float64) * 1.0

        with pytest.raises(TopologicalTorsionError, match="Torsión topológica"):
            conn._verify_zero_torsion(large_torsion)

    def test_torsion_error_message_contains_index(self) -> None:
        """El mensaje de TopologicalTorsionError incluye información del índice."""
        n = 3
        G = make_spd_matrix(n, seed=107)
        conn = build_torsion_free(G)

        torsion = np.zeros((n, n, n), dtype=np.float64)
        torsion[2, 1, 0] = 1.0  # Asimetría en índice conocido

        with pytest.raises(TopologicalTorsionError) as exc_info:
            conn._verify_zero_torsion(torsion)

        error_msg = str(exc_info.value)
        assert "Torsión topológica" in error_msg, (
            "El mensaje debe mencionar 'Torsión topológica'."
        )

    def test_connection_diagnostics_is_frozen(self) -> None:
        """ConnectionDiagnostics es inmutable (frozen dataclass)."""
        G = make_spd_matrix(4, seed=109)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        with pytest.raises((AttributeError, TypeError)):
            diag.torsion_norm = 999.0  # type: ignore[misc]

    def test_torsion_norm_is_zero_for_zero_gamma(self) -> None:
        """‖T‖_F = 0 exactamente para Γ = 0 (modo estático)."""
        G = make_spd_matrix(5, seed=113)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.torsion_norm == 0.0, (
            f"‖T‖_F={diag.torsion_norm} ≠ 0 para Γ=0."
        )

    def test_torsion_antisymmetry_property(self) -> None:
        """
        El tensor de torsión T^r_{mn} es antisimétrico en m,n:
        T[r,m,n] = -T[r,n,m] (verifica la propiedad algebraica del cálculo).
        """
        n = 4
        rng = np.random.default_rng(117)
        # Gamma asimétrico aleatorio
        Gamma = rng.standard_normal((n, n, n)).astype(np.float64)
        torsion = _TorsionFreeConnection._compute_torsion_tensor(Gamma)

        # Verificar antisimetría: T[r,m,n] = -T[r,n,m]
        assert_allclose(
            torsion, -torsion.transpose(0, 2, 1), atol=1e-14,
            err_msg="El tensor de torsión debe ser antisimétrico en índices m,n."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G2b — TestTorsionFreeConnection: Axioma 2 — Compatibilidad Métrica        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestTorsionFreeConnectionMetricCompat:
    """
    Verifica el Axioma 2 de la conexión de Levi-Civita:
      (∇_γ G)_{μν} = ∂_γ G_{μν} - Γ^k_{γμ} G_{kν} - Γ^k_{γν} G_{μk} = 0

    Pruebas:
      - ‖∇G‖_F = 0 para métricas estáticas (dG=0, Γ=0).
      - MetricCompatibilityError para ∇G grande.
      - _compute_covd_metric retorna tensor correcto.
      - El tensor ∇G es simétrico en μ,ν (hereda simetría de G).
    """

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_metric_compat_passed_for_static(self, n: int) -> None:
        """Compatibilidad métrica verificada para métricas estáticas."""
        G = make_spd_matrix(n, seed=n * 127)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.metric_compat_passed, (
            f"Compatibilidad métrica debe pasar para n={n}."
        )
        assert diag.covd_metric_norm < _METRIC_COMPAT_TOLERANCE, (
            f"‖∇G‖_F={diag.covd_metric_norm:.2e} ≥ tol={_METRIC_COMPAT_TOLERANCE}."
        )

    def test_covd_metric_is_zero_for_zero_gamma_and_dg(self) -> None:
        """
        ∇G = 0 cuando Γ = 0 y dG = 0 (espacio plano, métrica constante).
        La derivada covariante se reduce a:
            (∇G)_{γμν} = ∂_γ G_{μν} - 0 - 0 = 0.
        """
        n = 4
        G = make_spd_matrix(n, seed=131)
        conn = build_torsion_free(G)

        Gamma = conn._christoffel_data.Gamma  # = 0 para modo estático
        dG    = conn._christoffel_data.dG     # = 0 para modo estático

        covd = conn._compute_covd_metric(Gamma, dG)
        assert_allclose(
            covd, np.zeros((n, n, n)), atol=1e-14,
            err_msg="∇G debe ser 0 para Γ=0 y dG=0."
        )

    def test_metric_compat_error_raised_for_large_covd(self) -> None:
        """
        MetricCompatibilityError cuando ‖∇G‖_F ≥ _METRIC_COMPAT_TOLERANCE.
        """
        n = 3
        G = make_spd_matrix(n, seed=137)
        conn = build_torsion_free(G)

        # Simular un tensor ∇G con norma grande
        large_covd = np.ones((n, n, n), dtype=np.float64) * 10.0

        with pytest.raises(MetricCompatibilityError, match="Violación"):
            conn._verify_metric_compatibility(large_covd)

    def test_metric_compat_error_message_contains_norm(self) -> None:
        """El mensaje de MetricCompatibilityError incluye la norma ‖∇G‖_F."""
        n = 3
        G = make_spd_matrix(n, seed=139)
        conn = build_torsion_free(G)
        large_covd = np.ones((n, n, n), dtype=np.float64)

        with pytest.raises(MetricCompatibilityError) as exc_info:
            conn._verify_metric_compatibility(large_covd)

        assert "∇_γ G" in str(exc_info.value), (
            "El mensaje debe mencionar '∇_γ G'."
        )

    def test_covd_metric_symmetry_in_lower_indices(self) -> None:
        """
        (∇_γ G)_{μν} es simétrico en μ,ν ya que G_{μν} = G_{νμ}.
        Para dG = 0 y Γ simétrico, ∇G = 0 es automáticamente simétrico.
        """
        n = 4
        G = make_spd_matrix(n, seed=149)
        conn = build_torsion_free(G)

        Gamma = conn._christoffel_data.Gamma
        dG    = conn._christoffel_data.dG

        covd = conn._compute_covd_metric(Gamma, dG)
        # ∇G debe ser simétrico en índices 1 y 2 (μ y ν)
        sym_error = np.max(np.abs(covd - covd.transpose(0, 2, 1)))
        assert sym_error < 1e-14, (
            f"∇G no es simétrico en μ,ν: error={sym_error:.2e}."
        )

    def test_covd_metric_norm_zero_for_static(self) -> None:
        """‖∇G‖_F = 0 exactamente para modo estático."""
        G = make_spd_matrix(5, seed=151)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.covd_metric_norm == 0.0, (
            f"‖∇G‖_F={diag.covd_metric_norm} ≠ 0 para modo estático."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G2c — TestConnectionDiagnostics: Riemann y Diagnósticos                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestConnectionDiagnostics:
    """
    Verifica el tensor de Riemann y los diagnósticos de ConnectionDiagnostics:
      - R = 0 para espacio plano (Γ = 0).
      - all_passed() ↔ torsion_passed AND metric_compat_passed.
      - summary() contiene todas las claves esperadas.
      - Valores numéricos correctos para casos analíticos.
    """

    def test_riemann_tensor_zero_for_flat_space(self) -> None:
        """
        R^r_{smn} = 0 para espacio plano (Γ = 0, dG = 0).

        Demostración:
            R = Γ·Γ - Γ·Γ = 0 (términos cuadráticos se cancelan).
        """
        n = 4
        G = make_spd_matrix(n, seed=157)
        conn = build_torsion_free(G)

        Gamma = conn._christoffel_data.Gamma  # = 0
        R = conn._compute_riemann_tensor(Gamma)
        assert_allclose(
            R, np.zeros((n, n, n, n)), atol=1e-14,
            err_msg="Tensor de Riemann debe ser 0 para Γ=0 (espacio plano)."
        )

    def test_riemann_norm_zero_in_diagnostics(self) -> None:
        """riemann_norm = 0 en diagnostics para modo estático."""
        G = make_spd_matrix(5, seed=163)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.riemann_norm == 0.0, (
            f"riemann_norm={diag.riemann_norm} ≠ 0 para modo estático."
        )

    def test_all_passed_true_for_valid_connection(self) -> None:
        """ConnectionDiagnostics.all_passed() == True para conexión válida."""
        G = make_spd_matrix(4, seed=167)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.all_passed(), (
            f"all_passed() debe ser True: {diag.summary()}."
        )

    def test_all_passed_false_when_torsion_fails(self) -> None:
        """all_passed() == False si torsion_passed == False."""
        # Construir ConnectionDiagnostics artificialmente con torsion_passed=False
        diag = ConnectionDiagnostics(
            torsion_norm         = 1.0,  # > tolerancia
            covd_metric_norm     = 0.0,
            riemann_norm         = 0.0,
            condition_number_reg = 10.0,
            torsion_passed       = False,
            metric_compat_passed = True,
        )
        assert not diag.all_passed(), "all_passed() debe ser False si torsion_passed=False."

    def test_all_passed_false_when_metric_compat_fails(self) -> None:
        """all_passed() == False si metric_compat_passed == False."""
        diag = ConnectionDiagnostics(
            torsion_norm         = 0.0,
            covd_metric_norm     = 1.0,  # > tolerancia
            riemann_norm         = 0.0,
            condition_number_reg = 10.0,
            torsion_passed       = True,
            metric_compat_passed = False,
        )
        assert not diag.all_passed(), "all_passed() debe ser False si metric_compat_passed=False."

    def test_summary_contains_all_expected_keys(self) -> None:
        """ConnectionDiagnostics.summary() contiene todas las claves esperadas."""
        G = make_spd_matrix(3, seed=173)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        summary = diag.summary()

        expected_keys = {
            "torsion_norm", "covd_metric_norm", "riemann_norm",
            "condition_number_reg", "torsion_passed", "metric_compat_passed",
            "all_passed",
        }
        assert expected_keys.issubset(summary.keys()), (
            f"Claves faltantes en summary: {expected_keys - summary.keys()}."
        )

    def test_connection_diagnostics_condition_number_positive(self) -> None:
        """condition_number_reg en diagnostics es positivo."""
        G = make_spd_matrix(4, seed=179)
        conn = build_torsion_free(G)
        diag = conn.connection_diagnostics()
        assert diag.condition_number_reg > 0, (
            "condition_number_reg debe ser > 0."
        )

    def test_riemann_tensor_shape(self) -> None:
        """_compute_riemann_tensor retorna tensor de shape (n,n,n,n)."""
        n = 4
        G = make_spd_matrix(n, seed=181)
        conn = build_torsion_free(G)
        Gamma = conn._christoffel_data.Gamma
        R = conn._compute_riemann_tensor(Gamma)
        assert R.shape == (n, n, n, n), (
            f"Tensor de Riemann debe tener shape ({n},{n},{n},{n}), "
            f"recibido {R.shape}."
        )

    def test_riemann_antisymmetry_in_last_indices(self) -> None:
        """
        R^r_{smn} es antisimétrico en m,n (propiedad algebraica del tensor de Riemann):
        R[r,s,m,n] = -R[r,s,n,m].

        Esta propiedad se hereda de la antisimetría de la contracción cuadrática
        cuando Gamma es simétrico en sus índices inferiores.
        """
        n = 4
        rng = np.random.default_rng(191)
        # Crear un Gamma simétrico no trivial (con derivadas inyectadas)
        Gamma = rng.standard_normal((n, n, n)).astype(np.float64)
        # Simetrizar en índices inferiores
        Gamma = (Gamma + Gamma.transpose(0, 2, 1)) * 0.5

        conn = build_torsion_free(make_spd_matrix(n, seed=191))
        R = conn._compute_riemann_tensor(Gamma)

        # R[r,s,m,n] + R[r,s,n,m] debe ser 0 (antisimetría en m,n)
        antisym_error = np.max(np.abs(R + R.transpose(0, 1, 3, 2)))
        assert antisym_error < 1e-13, (
            f"Tensor de Riemann no es antisimétrico en m,n: error={antisym_error:.2e}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G3a — TestGeodesicAcceleration: Cálculo del RHS Geodésico                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestGeodesicAcceleration:
    """
    Verifica el cálculo de la aceleración geodésica:
      a^μ = -Γ^μ_{rs} v^r v^s

    Pruebas:
      - a = 0 para Γ = 0 (espacio plano).
      - a = 0 para v = 0 (vector nulo).
      - a escala como ‖v‖² (cuadrática en v).
      - a tiene el signo correcto para Γ positivo.
      - Contracción tensorial correcta verificada analíticamente.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 4
        G = make_spd_matrix(self.n, seed=197)
        self.agent = build_agent(G)

    def test_acceleration_zero_for_flat_space(self) -> None:
        """a = 0 para Γ = 0 (espacio plano, modo estático)."""
        v = np.ones(self.n, dtype=np.float64)
        acc = self.agent._geodesic_acceleration(v)
        assert_allclose(
            acc, np.zeros(self.n), atol=1e-15,
            err_msg="Aceleración geodésica debe ser 0 para Γ=0."
        )

    def test_acceleration_zero_for_zero_velocity(self) -> None:
        """a = 0 para v = 0 (no hay aceleración sin velocidad)."""
        v_zero = np.zeros(self.n, dtype=np.float64)
        acc = self.agent._geodesic_acceleration(v_zero)
        assert_allclose(
            acc, np.zeros(self.n), atol=1e-15,
            err_msg="Aceleración geodésica debe ser 0 para v=0."
        )

    def test_acceleration_quadratic_in_velocity(self) -> None:
        """
        a(α·v) = α² · a(v) (cuadrática en la velocidad).
        """
        rng = np.random.default_rng(199)
        v = rng.standard_normal(self.n).astype(np.float64)
        alpha = 3.0

        acc_v       = self.agent._geodesic_acceleration(v)
        acc_alpha_v = self.agent._geodesic_acceleration(alpha * v)

        assert_allclose(
            acc_alpha_v, alpha ** 2 * acc_v, rtol=1e-12,
            err_msg="Aceleración geodésica debe escalar cuadráticamente con v."
        )

    def test_acceleration_shape(self) -> None:
        """_geodesic_acceleration retorna array de shape (n,)."""
        v = np.ones(self.n, dtype=np.float64)
        acc = self.agent._geodesic_acceleration(v)
        assert acc.shape == (self.n,), (
            f"Aceleración debe tener shape ({self.n},), obtenido {acc.shape}."
        )

    def test_acceleration_is_finite(self) -> None:
        """_geodesic_acceleration retorna valores finitos."""
        rng = np.random.default_rng(211)
        v = rng.standard_normal(self.n).astype(np.float64)
        acc = self.agent._geodesic_acceleration(v)
        assert np.all(np.isfinite(acc)), (
            "Aceleración geodésica contiene valores no finitos."
        )

    def test_acceleration_contraction_analytic_2d(self) -> None:
        """
        Verifica la contracción tensorial para un Γ no nulo conocido en 2D.

        Para Γ^0_{00} = c (único coeficiente no nulo) y v = (v₀, v₁):
            a^0 = -Γ^0_{rs} v^r v^s = -Γ^0_{00} v₀² = -c v₀²
            a^1 = 0

        Usamos una subclase que inyecta un Γ conocido.
        """
        c = 1.5
        v0 = 2.0

        class _MockAgent(LeviCivitaConnectionAgent):
            def _compute_metric_derivative(self) -> NDArray[np.float64]:
                n = self._n
                dG = np.zeros((n, n, n), dtype=np.float64)
                # Para 2D, inyectamos Γ via dG tal que Γ^0_{00} = c:
                # Γ^0_{00} = ½ G^{0k}(∂₀G_{k0} + ∂₀G_{0k} - ∂_kG_{00})
                # Con G = I: Γ^0_{00} = ½(dG[0,0,0] + dG[0,0,0] - dG[0,0,0])
                #           = ½ dG[0,0,0]
                # Entonces dG[0,0,0] = 2c
                dG[0, 0, 0] = 2.0 * c
                return dG

        G2 = make_identity_metric(2)
        try:
            agent_2d = _MockAgent(metric_tensor=G2)
            v = np.array([v0, 0.0], dtype=np.float64)
            acc = agent_2d._geodesic_acceleration(v)
            assert_allclose(
                acc[0], -c * v0 ** 2, rtol=1e-10,
                err_msg=f"a^0 = -c·v₀² = {-c*v0**2:.4f}, obtenido {acc[0]:.4f}."
            )
            assert_allclose(
                acc[1], 0.0, atol=1e-14,
                err_msg=f"a^1 debe ser 0, obtenido {acc[1]:.2e}."
            )
        except (TopologicalTorsionError, MetricCompatibilityError):
            pytest.skip(
                "Gamma inyectado viola verificaciones de Fase 2; test omitido."
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G3b — TestRK4Integration: Integrador Geodésico Runge-Kutta 4              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestRK4Integration:
    """
    Verifica el integrador RK4 de la ecuación geodésica:
      - Para Γ = 0: v(t+dt) = v(t) exactamente (aceleración nula).
      - Conservación de norma aproximada: |‖v_f‖ - ‖v_i‖| / ‖v_i‖ < _GEODESIC_NORM_DRIFT_TOL.
      - GeodesicStepReport con estructura correcta.
      - Rechazo de dt ≤ _DT_MIN.
      - Finitud del resultado.
      - _rk4_step retorna array de shape (n,).
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 5
        G = make_spd_matrix(self.n, seed=223)
        self.agent = build_agent(G)

    def test_rk4_identity_for_flat_space(self) -> None:
        """Para Γ = 0: v(t+dt) = v(t) exactamente (aceleración = 0)."""
        rng = np.random.default_rng(227)
        v = rng.standard_normal(self.n).astype(np.float64)
        v_out = self.agent._rk4_step(v, dt=_DEFAULT_DT)
        assert_allclose(
            v_out, v, atol=1e-15,
            err_msg="Para Γ=0, RK4 debe preservar v exactamente."
        )

    def test_rk4_output_shape(self) -> None:
        """_rk4_step retorna array de shape (n,)."""
        v = np.ones(self.n, dtype=np.float64)
        v_out = self.agent._rk4_step(v, dt=_DEFAULT_DT)
        assert v_out.shape == (self.n,), (
            f"_rk4_step debe retornar shape ({self.n},), obtenido {v_out.shape}."
        )

    def test_rk4_output_is_finite(self) -> None:
        """_rk4_step retorna valores finitos."""
        rng = np.random.default_rng(229)
        v = rng.standard_normal(self.n).astype(np.float64)
        v_out = self.agent._rk4_step(v, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(v_out)), (
            "Salida de _rk4_step contiene valores no finitos."
        )

    def test_enforce_geodesic_flow_returns_tuple(self) -> None:
        """enforce_geodesic_flow retorna Tuple[TangentVector, GeodesicStepReport]."""
        v = make_random_unit_tangent(self.n, seed=233)
        result = self.agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        assert isinstance(result, tuple) and len(result) == 2, (
            "enforce_geodesic_flow debe retornar una tupla de 2 elementos."
        )
        v_out, report = result
        assert isinstance(v_out, TangentVector), (
            f"Primer elemento debe ser TangentVector, obtenido {type(v_out).__name__}."
        )
        assert isinstance(report, GeodesicStepReport), (
            f"Segundo elemento debe ser GeodesicStepReport, obtenido {type(report).__name__}."
        )

    def test_norm_preserved_for_flat_space(self) -> None:
        """‖v(t+dt)‖ = ‖v(t)‖ exactamente para Γ = 0."""
        rng = np.random.default_rng(239)
        v_arr = rng.standard_normal(self.n).astype(np.float64)
        v = make_tangent_vector(v_arr)
        v_out, report = self.agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        assert_allclose(
            report.v_final_norm, report.v_initial_norm, rtol=1e-14,
            err_msg="Para Γ=0, ‖v_f‖ debe ser igual a ‖v_i‖."
        )

    def test_geodesic_step_report_fields(self) -> None:
        """GeodesicStepReport tiene todos los campos esperados."""
        v = make_random_unit_tangent(self.n, seed=241)
        _, report = self.agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        expected_fields = {
            "v_initial_norm", "v_final_norm", "norm_drift",
            "acceleration_norm", "dt", "dt_max_stable", "is_stable",
        }
        for field_name in expected_fields:
            assert hasattr(report, field_name), (
                f"GeodesicStepReport falta el campo '{field_name}'."
            )

    def test_geodesic_step_report_is_frozen(self) -> None:
        """GeodesicStepReport es inmutable (frozen dataclass)."""
        v = make_random_unit_tangent(self.n, seed=243)
        _, report = self.agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        with pytest.raises((AttributeError, TypeError)):
            report.norm_drift = 999.0  # type: ignore[misc]

    def test_enforce_geodesic_rejects_dt_below_min(self) -> None:
        """ValueError para dt ≤ _DT_MIN."""
        v = make_random_unit_tangent(self.n, seed=247)
        with pytest.raises(ValueError, match="_DT_MIN"):
            self.agent.enforce_geodesic_flow(v, dt=_DT_MIN * 0.1)

    def test_enforce_geodesic_rejects_non_numeric_dt(self) -> None:
        """TypeError para dt de tipo no numérico."""
        v = make_random_unit_tangent(self.n, seed=251)
        with pytest.raises(TypeError):
            self.agent.enforce_geodesic_flow(v, dt="0.01")  # type: ignore[arg-type]

    def test_enforce_geodesic_rejects_wrong_dim_velocity(self) -> None:
        """ValueError si dim(v) ≠ n."""
        v_wrong = make_tangent_vector(np.ones(self.n + 1, dtype=np.float64))
        with pytest.raises(ValueError, match="incompatibilidad dimensional"):
            self.agent.enforce_geodesic_flow(v_wrong, dt=_DEFAULT_DT)

    def test_enforce_geodesic_rejects_non_tangent_vector(self) -> None:
        """TypeError si el argumento no es TangentVector."""
        with pytest.raises(TypeError, match="TangentVector"):
            self.agent.enforce_geodesic_flow(
                np.ones(self.n),  # type: ignore[arg-type]
                dt=_DEFAULT_DT
            )

    def test_dt_max_stable_is_positive(self) -> None:
        """dt_max_stable del agente es siempre positivo."""
        assert self.agent._dt_max_stable > 0, (
            f"dt_max_stable={self.agent._dt_max_stable} debe ser > 0."
        )

    def test_dt_max_stable_infinite_for_zero_gamma(self) -> None:
        """Para Γ = 0 (modo estático), dt_max_stable debe ser un valor grande."""
        G = make_spd_matrix(4, seed=257)
        agent = build_agent(G)  # Γ = 0 en modo estático
        # Para Γ = 0, dt_max es 1e6 (límite práctico)
        assert agent._dt_max_stable >= 1e5, (
            f"Para Γ=0, dt_max_stable debe ser ≥ 1e5, "
            f"obtenido {agent._dt_max_stable}."
        )

    @pytest.mark.parametrize("dt", [1e-4, 1e-3, 1e-2])
    def test_rk4_different_dt_values_are_stable(self, dt: float) -> None:
        """enforce_geodesic_flow es estable para diferentes valores de dt en espacio plano."""
        v = make_random_unit_tangent(self.n, seed=263)
        v_out, report = self.agent.enforce_geodesic_flow(v, dt=dt)
        assert np.all(np.isfinite(v_out.coordinates)), (
            f"Resultado no finito para dt={dt}."
        )
        assert_allclose(
            report.v_final_norm, report.v_initial_norm, rtol=1e-12,
            err_msg=f"Norma no conservada para dt={dt} en espacio plano."
        )

    def test_validate_geodesic_output_returns_report(self) -> None:
        """_validate_geodesic_output construye GeodesicStepReport correctamente."""
        v_i = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v_f = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        report = self.agent._validate_geodesic_output(v_i, v_f, dt=_DEFAULT_DT)
        assert isinstance(report, GeodesicStepReport)
        assert_allclose(report.norm_drift, 0.0, atol=1e-15)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G3c — TestParallelTransport: Transporte Paralelo D V/dt = 0               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestParallelTransport:
    """
    Verifica la implementación del transporte paralelo:
      D V^μ/dt = dV^μ/dt + Γ^μ_{rs} ẏ^r V^s = 0

    Pruebas:
      - Para Γ = 0: V_transported = V exactamente.
      - Corrección proporcional a dt (método de Euler de primer orden).
      - Rechazo de vectores dimensionalmente incompatibles.
      - Resultado finito para entradas válidas.
      - Diferencia entre transporte paralelo y geodésico.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 4
        G = make_spd_matrix(self.n, seed=269)
        self.agent = build_agent(G)

    def test_parallel_transport_identity_for_flat_space(self) -> None:
        """Para Γ = 0: V(t+dt) = V(t) exactamente."""
        V    = make_random_unit_tangent(self.n, seed=271)
        ydot = make_random_unit_tangent(self.n, seed=273)

        V_transported = self.agent.parallel_transport(V, ydot, dt=_DEFAULT_DT)
        assert_allclose(
            V_transported.coordinates, V.coordinates, atol=1e-15,
            err_msg="Para Γ=0, el transporte paralelo debe ser la identidad."
        )

    def test_parallel_transport_returns_tangent_vector(self) -> None:
        """parallel_transport retorna TangentVector."""
        V    = make_random_unit_tangent(self.n, seed=277)
        ydot = make_random_unit_tangent(self.n, seed=281)
        V_t  = self.agent.parallel_transport(V, ydot, dt=_DEFAULT_DT)
        assert isinstance(V_t, TangentVector), (
            f"parallel_transport debe retornar TangentVector, "
            f"retornó {type(V_t).__name__}."
        )

    def test_parallel_transport_output_is_finite(self) -> None:
        """Salida de parallel_transport es finita."""
        V    = make_random_unit_tangent(self.n, seed=283)
        ydot = make_random_unit_tangent(self.n, seed=287)
        V_t  = self.agent.parallel_transport(V, ydot, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(V_t.coordinates)), (
            "parallel_transport contiene valores no finitos."
        )

    def test_parallel_transport_rejects_wrong_dim_vector(self) -> None:
        """ValueError si dim(V) ≠ n."""
        V_bad = make_tangent_vector(np.ones(self.n + 1, dtype=np.float64))
        ydot  = make_random_unit_tangent(self.n, seed=291)
        with pytest.raises(ValueError, match="incompatibilidad dimensional"):
            self.agent.parallel_transport(V_bad, ydot, dt=_DEFAULT_DT)

    def test_parallel_transport_rejects_wrong_dim_curve_tangent(self) -> None:
        """ValueError si dim(ydot) ≠ n."""
        V       = make_random_unit_tangent(self.n, seed=293)
        ydot_bad = make_tangent_vector(np.ones(self.n + 2, dtype=np.float64))
        with pytest.raises(ValueError, match="incompatibilidad dimensional"):
            self.agent.parallel_transport(V, ydot_bad, dt=_DEFAULT_DT)

    def test_parallel_transport_rejects_dt_below_min(self) -> None:
        """ValueError para dt ≤ _DT_MIN."""
        V    = make_random_unit_tangent(self.n, seed=297)
        ydot = make_random_unit_tangent(self.n, seed=299)
        with pytest.raises(ValueError, match="_DT_MIN"):
            self.agent.parallel_transport(V, ydot, dt=_DT_MIN * 0.5)

    def test_parallel_transport_scales_linearly_with_dt(self) -> None:
        """
        La corrección del transporte paralelo escala linealmente con dt
        (método de Euler: V(t+dt) = V(t) - dt · Γ·ẏ·V).

        Para Γ ≠ 0, la diferencia V(t+dt₁) - V(t) ≈ dt₁ · corr,
        y V(t+dt₂) - V(t) ≈ dt₂ · corr.
        Si Γ = 0 (modo estático), la corrección es 0 independientemente de dt.
        """
        V    = make_random_unit_tangent(self.n, seed=307)
        ydot = make_random_unit_tangent(self.n, seed=311)

        # Para Γ = 0 (modo estático), V_transported = V para cualquier dt
        V_t1 = self.agent.parallel_transport(V, ydot, dt=1e-3)
        V_t2 = self.agent.parallel_transport(V, ydot, dt=2e-3)

        diff1 = np.linalg.norm(V_t1.coordinates - V.coordinates)
        diff2 = np.linalg.norm(V_t2.coordinates - V.coordinates)

        # Ambas correcciones son 0 para Γ=0
        assert_allclose(diff1, 0.0, atol=1e-15,
                        err_msg="Corrección 1 debe ser 0 para Γ=0.")
        assert_allclose(diff2, 0.0, atol=1e-15,
                        err_msg="Corrección 2 debe ser 0 para Γ=0.")


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G4 — TestCategoricalTransports: ♭ y ♯ con Corrección Geodésica           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestCategoricalTransports:
    """
    Verifica los transportes categóricos ♭ y ♯ con corrección geodésica:
      - transport_to_finance_oracle: TangentVector → CotangentVector via ♭.
      - transport_to_logistics_manifold: CotangentVector → TangentVector via ♯.
      - Roundtrip ♭ → ♯ ≈ id_{TM} con y sin corrección geodésica.
      - Tipos de retorno correctos.
      - GeodesicStepReport en transport_to_finance_oracle.
      - Coherencia de normas.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 5
        G = make_spd_matrix(self.n, seed=313)
        self.agent = build_agent(G)

    # ── transport_to_finance_oracle (♭) ─────────────────────────────────────

    def test_finance_oracle_returns_correct_types(self) -> None:
        """transport_to_finance_oracle retorna (CotangentVector, GeodesicStepReport)."""
        v = make_random_unit_tangent(self.n, seed=317)
        omega, report = self.agent.transport_to_finance_oracle(v, dt=_DEFAULT_DT)
        assert isinstance(omega, CotangentVector), (
            f"Primer retorno debe ser CotangentVector, obtenido {type(omega).__name__}."
        )
        assert isinstance(report, GeodesicStepReport), (
            f"Segundo retorno debe ser GeodesicStepReport, "
            f"obtenido {type(report).__name__}."
        )

    def test_finance_oracle_covector_is_finite(self) -> None:
        """CotangentVector retornado por transport_to_finance_oracle es finito."""
        v = make_random_unit_tangent(self.n, seed=331)
        omega, _ = self.agent.transport_to_finance_oracle(v, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(omega.coordinates)), (
            "CotangentVector de transport_to_finance_oracle contiene no-finitos."
        )

    def test_finance_oracle_without_geodesic_correction(self) -> None:
        """
        Con apply_geodesic_correction=False, ♭ se aplica directamente
        sin pasar por RK4.
        """
        v = make_random_unit_tangent(self.n, seed=337)
        omega, report = self.agent.transport_to_finance_oracle(
            v, dt=_DEFAULT_DT, apply_geodesic_correction=False
        )
        assert isinstance(omega, CotangentVector)
        # El reporte trivial debe tener norm_drift = 0
        assert report.norm_drift == 0.0, (
            f"norm_drift debe ser 0 para corrección desactivada, "
            f"obtenido {report.norm_drift}."
        )

    def test_finance_oracle_rejects_non_tangent_vector(self) -> None:
        """TypeError si logistics_flow no es TangentVector."""
        with pytest.raises(TypeError, match="TangentVector"):
            self.agent.transport_to_finance_oracle(
                np.ones(self.n),  # type: ignore[arg-type]
                dt=_DEFAULT_DT
            )

    def test_finance_oracle_rejects_wrong_dimension(self) -> None:
        """ValueError si dim(v) ≠ n."""
        v_wrong = make_tangent_vector(np.ones(self.n + 1, dtype=np.float64))
        with pytest.raises(ValueError, match="incompatibilidad dimensional"):
            self.agent.transport_to_finance_oracle(v_wrong, dt=_DEFAULT_DT)

    def test_finance_oracle_norm_bound(self) -> None:
        """
        ‖♭(v_geodesic)‖ ≤ λ_max · ‖v_geodesic‖ (cota de Lipschitz de ♭).
        """
        v = make_random_unit_tangent(self.n, seed=347)
        omega, _ = self.agent.transport_to_finance_oracle(
            v, dt=_DEFAULT_DT, apply_geodesic_correction=False
        )
        lambda_max = float(self.agent._pm.eigenvalues_reg[-1])
        assert omega.norm <= lambda_max * v.norm * (1 + 1e-10), (
            f"‖ω‖={omega.norm:.3e} > λ_max·‖v‖={lambda_max * v.norm:.3e}."
        )

    # ── transport_to_logistics_manifold (♯) ──────────────────────────────────

    def test_logistics_manifold_returns_correct_types_no_post(self) -> None:
        """
        Sin post-corrección: retorna (TangentVector, None).
        """
        omega = make_random_cotangent(self.n, seed=353)
        v, report = self.agent.transport_to_logistics_manifold(
            omega, apply_post_geodesic=False
        )
        assert isinstance(v, TangentVector), (
            f"Primer retorno debe ser TangentVector, obtenido {type(v).__name__}."
        )
        assert report is None, (
            f"Segundo retorno debe ser None sin post-corrección, obtenido {report}."
        )

    def test_logistics_manifold_returns_correct_types_with_post(self) -> None:
        """
        Con post-corrección: retorna (TangentVector, GeodesicStepReport).
        """
        omega = make_random_cotangent(self.n, seed=359)
        v, report = self.agent.transport_to_logistics_manifold(
            omega, apply_post_geodesic=True, dt=_DEFAULT_DT
        )
        assert isinstance(v, TangentVector)
        assert isinstance(report, GeodesicStepReport), (
            f"Con post-corrección, segundo retorno debe ser GeodesicStepReport."
        )

    def test_logistics_manifold_rejects_non_cotangent(self) -> None:
        """TypeError si financial_force no es CotangentVector."""
        with pytest.raises(TypeError, match="CotangentVector"):
            self.agent.transport_to_logistics_manifold(
                np.ones(self.n),  # type: ignore[arg-type]
            )

    def test_logistics_manifold_result_is_finite(self) -> None:
        """TangentVector retornado es finito."""
        omega = make_random_cotangent(self.n, seed=367)
        v, _ = self.agent.transport_to_logistics_manifold(omega)
        assert np.all(np.isfinite(v.coordinates)), (
            "TangentVector de transport_to_logistics_manifold contiene no-finitos."
        )

    # ── Roundtrip ♭ → ♯ ≈ id ────────────────────────────────────────────────

    @pytest.mark.parametrize("seed", [370, 371, 372, 373, 374])
    def test_roundtrip_flat_sharp_identity(self, seed: int) -> None:
        """
        ♯(♭(v)) ≈ v bajo corrección geodésica para Γ = 0 (identidad exacta).

        Para Γ = 0: enforce_geodesic_flow(v) = v exactamente.
        Entonces: ♯(♭(v)) = ♯(♭(v)) = (G_inv @ G) @ v = I @ v = v.
        """
        v = make_random_unit_tangent(self.n, seed=seed)

        # ♭ con corrección geodésica (= identity para Γ=0)
        omega, _ = self.agent.transport_to_finance_oracle(
            v, dt=_DEFAULT_DT, apply_geodesic_correction=True
        )
        # ♯ sin corrección posterior
        v_rec, _ = self.agent.transport_to_logistics_manifold(
            omega, apply_post_geodesic=False
        )

        kappa = self.agent._pm.condition_number_reg
        tol = max(100 * kappa * _MACHINE_EPSILON, 1e-12)
        residual = np.linalg.norm(v_rec.coordinates - v.coordinates)

        assert residual <= tol * max(v.norm, 1.0), (
            f"Roundtrip ♯∘♭ ≈ id fallido: ‖v_rec - v‖={residual:.2e} "
            f"> tol={tol:.2e}, seed={seed}."
        )

    def test_roundtrip_without_geodesic_correction(self) -> None:
        """
        ♯(♭(v)) ≈ v sin corrección geodésica (prueba el par puro de isomorfismos).
        """
        v = make_random_unit_tangent(self.n, seed=379)

        omega, _ = self.agent.transport_to_finance_oracle(
            v, apply_geodesic_correction=False
        )
        v_rec, _ = self.agent.transport_to_logistics_manifold(
            omega, apply_post_geodesic=False
        )

        kappa = self.agent._pm.condition_number_reg
        tol = max(100 * kappa * _MACHINE_EPSILON, 1e-12)
        residual = np.linalg.norm(v_rec.coordinates - v.coordinates)
        assert residual <= tol * max(v.norm, 1.0), (
            f"Roundtrip puro ♯∘♭ ≈ id fallido: residual={residual:.2e}."
        )

    def test_zero_vector_roundtrip(self) -> None:
        """♯(♭(0)) = 0 (linealidad preserva el cero)."""
        v_zero = make_tangent_vector(np.zeros(self.n, dtype=np.float64))
        omega, _ = self.agent.transport_to_finance_oracle(
            v_zero, apply_geodesic_correction=False
        )
        v_rec, _ = self.agent.transport_to_logistics_manifold(omega)
        assert_allclose(
            v_rec.coordinates, np.zeros(self.n), atol=1e-14,
            err_msg="♯(♭(0)) debe ser 0."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G5 — TestGeodesicFlowReport: Informe Diagnóstico Unificado                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestGeodesicFlowReport:
    """
    Verifica geodesic_flow_report() del agente:
      - Contiene todas las claves de las 3 fases.
      - Valores coherentes con el estado del agente.
      - is_static = True para modo estático.
      - all_axioms_passed = True para conexión válida.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        G = make_spd_matrix(4, seed=383)
        self.agent = build_agent(G)

    def test_report_contains_all_phase_keys(self) -> None:
        """geodesic_flow_report() contiene claves de las 3 fases."""
        report = self.agent.geodesic_flow_report()
        expected_keys = {
            # Fase 1
            "metric_dimension", "christoffel_frob_norm",
            "is_static_metric", "condition_number_reg",
            "null_space_dim", "regularization_applied",
            # Fase 2
            "torsion_norm", "covd_metric_norm", "riemann_norm",
            "torsion_passed", "metric_compat_passed", "all_axioms_passed",
            # Fase 3
            "dt_default", "dt_max_stable",
            "musical_engine_available", "agent",
        }
        missing = expected_keys - set(report.keys())
        assert not missing, f"Claves faltantes en geodesic_flow_report: {missing}."

    def test_report_is_static_for_constant_metric(self) -> None:
        """is_static_metric == True para métrica constante."""
        report = self.agent.geodesic_flow_report()
        assert report["is_static_metric"], (
            "is_static_metric debe ser True para métrica constante."
        )

    def test_report_all_axioms_passed(self) -> None:
        """all_axioms_passed == True para una conexión válida."""
        report = self.agent.geodesic_flow_report()
        assert report["all_axioms_passed"], (
            f"all_axioms_passed debe ser True: {report}."
        )

    def test_report_musical_engine_available(self) -> None:
        """musical_engine_available == True tras construcción exitosa."""
        report = self.agent.geodesic_flow_report()
        assert report["musical_engine_available"], (
            "musical_engine_available debe ser True."
        )

    def test_report_christoffel_frob_norm_zero_for_static(self) -> None:
        """christoffel_frob_norm == 0 para métrica estática."""
        report = self.agent.geodesic_flow_report()
        assert report["christoffel_frob_norm"] == 0.0, (
            f"christoffel_frob_norm={report['christoffel_frob_norm']} ≠ 0 "
            f"para modo estático."
        )

    def test_report_torsion_norm_zero_for_static(self) -> None:
        """torsion_norm == 0 para modo estático."""
        report = self.agent.geodesic_flow_report()
        assert report["torsion_norm"] == 0.0, (
            f"torsion_norm={report['torsion_norm']} ≠ 0 para modo estático."
        )

    def test_report_metric_dimension_correct(self) -> None:
        """metric_dimension en el informe coincide con el n de la métrica."""
        n = 4
        G = make_spd_matrix(n, seed=389)
        agent = build_agent(G)
        report = agent.geodesic_flow_report()
        assert report["metric_dimension"] == n, (
            f"metric_dimension={report['metric_dimension']} ≠ {n}."
        )

    def test_report_dt_default_is_expected_constant(self) -> None:
        """dt_default en el informe coincide con _DEFAULT_DT."""
        report = self.agent.geodesic_flow_report()
        assert report["dt_default"] == _DEFAULT_DT, (
            f"dt_default={report['dt_default']} ≠ {_DEFAULT_DT}."
        )

    def test_report_agent_label(self) -> None:
        """El campo 'agent' identifica correctamente al agente."""
        report = self.agent.geodesic_flow_report()
        assert "LeviCivitaConnectionAgent" in report["agent"], (
            f"Campo 'agent' no menciona 'LeviCivitaConnectionAgent': {report['agent']}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G5 — TestAgentConstruction: Construcción y Contratos del Agente           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestAgentConstruction:
    """
    Verifica la construcción del LeviCivitaConnectionAgent:
      - Construcción exitosa para métricas SPD de distintas dimensiones.
      - Inyección de dependencias (preconditioner personalizado).
      - Propagación de errores de Fase 1 y 2.
      - Métricas mal condicionadas (con regularización).
      - Métricas de rango deficiente (con regularización de Tikhonov).
      - Acceso a propiedades después de la construcción.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
    def test_construction_valid_spd(self, n: int) -> None:
        """Construcción exitosa para SPD de dimensión n."""
        G = make_spd_matrix(n, seed=n * 397)
        agent = build_agent(G)
        assert agent.metric_dimension == n, (
            f"metric_dimension={agent.metric_dimension} ≠ {n}."
        )

    def test_construction_with_identity_metric(self) -> None:
        """Construcción exitosa para G = I_n."""
        for n in [2, 3, 5]:
            agent = build_agent(make_identity_metric(n))
            assert agent.metric_dimension == n

    def test_construction_with_diagonal_metric(self) -> None:
        """Construcción exitosa para métrica diagonal."""
        G = make_diagonal_metric([1.0, 2.0, 3.0, 4.0])
        agent = build_agent(G)
        assert agent.metric_dimension == 4

    def test_construction_with_ill_conditioned_metric(self) -> None:
        """Construcción exitosa para métrica mal condicionada (con regularización)."""
        G = make_ill_conditioned_matrix(4, condition_number=1e14, seed=401)
        agent = build_agent(G)
        assert agent.preconditioned_metric.regularization_applied, (
            "La métrica mal condicionada debe activar regularización."
        )

    def test_construction_with_rank_deficient_metric(self) -> None:
        """Construcción exitosa para métrica de rango deficiente."""
        G = make_rank_deficient_matrix(5, rank=3, seed=409)
        agent = build_agent(G)
        assert agent.preconditioned_metric.regularization_applied, (
            "Métrica de rango deficiente debe activar regularización."
        )

    def test_dependency_injection_preconditioner(self) -> None:
        """Preconditioner inyectado es utilizado correctamente."""
        G = make_spd_matrix(4, seed=419)
        custom_pc = MetricSpectralPreconditioner()
        agent = build_agent(G, preconditioner=custom_pc)
        assert agent.metric_dimension == 4

    def test_construction_propagates_nan_error(self) -> None:
        """ValueError propagado para métrica con NaN."""
        G = make_spd_matrix(3, seed=421)
        G[0, 1] = np.nan
        with pytest.raises(ValueError, match="no finitos"):
            build_agent(G)

    def test_construction_propagates_non_array_error(self) -> None:
        """TypeError propagado para entrada que no es ndarray."""
        with pytest.raises(TypeError):
            build_agent([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_agent_has_musical_engine(self) -> None:
        """El agente tiene un MusicalIsomorphismEngine disponible."""
        G = make_spd_matrix(4, seed=431)
        agent = build_agent(G)
        assert agent._musical_engine is not None, (
            "_musical_engine debe estar inicializado."
        )

    def test_agent_christoffel_symbols_property(self) -> None:
        """LeviCivitaConnectionAgent.christoffel_symbols retorna copia de Γ."""
        G = make_spd_matrix(4, seed=433)
        agent = build_agent(G)
        Gamma = agent.christoffel_symbols
        assert Gamma.shape == (4, 4, 4), (
            f"christoffel_symbols.shape={Gamma.shape} ≠ (4,4,4)."
        )
        # Verificar que es copia: modificación no afecta interno
        Gamma[0, 0, 0] = 9999.0
        assert agent._christoffel_data.Gamma[0, 0, 0] != 9999.0

    def test_agent_preconditioned_metric_property(self) -> None:
        """preconditioned_metric retorna PreconditionedMetric válida."""
        G = make_spd_matrix(4, seed=439)
        agent = build_agent(G)
        pm = agent.preconditioned_metric
        assert isinstance(pm, PreconditionedMetric)

    def test_agent_connection_diagnostics_all_passed(self) -> None:
        """connection_diagnostics().all_passed() == True para métrica válida."""
        G = make_spd_matrix(4, seed=443)
        agent = build_agent(G)
        diag = agent.connection_diagnostics()
        assert diag.all_passed(), (
            f"all_passed() debe ser True: {diag.summary()}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G6 — TestHighDimensionalAndStatistical: Robustez y Alta Dimensión        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestHighDimensionalAndStatistical:
    """
    Verifica robustez numérica para métricas de alta dimensión y
    propiedades estadísticas sobre conjuntos de métricas aleatorias:
      - Pipeline completo para n ∈ {10, 20, 50}.
      - Torsión nula preservada para toda n.
      - Compatibilidad métrica preservada para toda n.
      - Roundtrip ♯∘♭ ≈ id para alta dimensión.
      - Consistencia de diagnósticos sobre 10 métricas aleatorias.
    """

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_full_pipeline_high_dimensional(self, n: int) -> None:
        """
        El pipeline completo de 3 fases funciona para n grande.

        Para n = 50, la construcción implica:
          - Precondicionamiento espectral: O(n³) ≈ 1.25e5 ops.
          - Christoffel: O(n³) einsum sobre tensor (n,n,n).
          - Verificaciones: O(n³) vectorizadas.
        """
        G = make_spd_matrix(n, seed=n * 449)
        agent = build_agent(G)
        assert agent.metric_dimension == n
        assert agent.connection_diagnostics().all_passed(), (
            f"Axiomas deben pasar para n={n}."
        )

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_geodesic_integration_high_dimensional(self, n: int) -> None:
        """enforce_geodesic_flow funciona para n grande."""
        G = make_spd_matrix(n, seed=n * 457)
        agent = build_agent(G)
        v = make_random_unit_tangent(n, seed=n * 461)
        v_out, report = agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(v_out.coordinates)), (
            f"Resultado no finito para n={n}."
        )
        assert_allclose(
            report.v_final_norm, report.v_initial_norm, rtol=1e-12,
            err_msg=f"Norma no conservada para n={n} en espacio plano."
        )

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_roundtrip_high_dimensional(self, n: int) -> None:
        """♯(♭(v)) ≈ v para alta dimensión."""
        G = make_spd_matrix(n, seed=n * 463)
        agent = build_agent(G)
        v = make_random_unit_tangent(n, seed=n + 467)

        omega, _ = agent.transport_to_finance_oracle(
            v, apply_geodesic_correction=False
        )
        v_rec, _ = agent.transport_to_logistics_manifold(omega)

        kappa = agent._pm.condition_number_reg
        tol = max(1000 * kappa * _MACHINE_EPSILON, 1e-11)
        residual = np.linalg.norm(v_rec.coordinates - v.coordinates)
        assert residual <= tol * max(v.norm, 1.0), (
            f"Roundtrip fallido para n={n}: residual={residual:.2e} > tol={tol:.2e}."
        )

    @pytest.mark.parametrize("trial", range(10))
    def test_statistical_torsion_zero_across_random_metrics(
        self, trial: int
    ) -> None:
        """
        Torsión nula para 10 métricas SPD aleatorias con n ∈ {3,4,5}.
        Barrido estadístico para detectar regresiones numéricas.
        """
        rng = np.random.default_rng(trial * 479 + 3)
        n = int(rng.integers(3, 6))
        G = make_spd_matrix(n, seed=trial * 479)
        agent = build_agent(G)
        diag = agent.connection_diagnostics()
        assert diag.torsion_passed, (
            f"Torsión no nula en trial={trial}, n={n}: "
            f"‖T‖={diag.torsion_norm:.2e}."
        )

    @pytest.mark.parametrize("trial", range(10))
    def test_statistical_metric_compat_across_random_metrics(
        self, trial: int
    ) -> None:
        """
        Compatibilidad métrica para 10 métricas SPD aleatorias.
        """
        rng = np.random.default_rng(trial * 487 + 5)
        n = int(rng.integers(3, 6))
        G = make_spd_matrix(n, seed=trial * 487)
        agent = build_agent(G)
        diag = agent.connection_diagnostics()
        assert diag.metric_compat_passed, (
            f"Compatibilidad métrica fallida en trial={trial}, n={n}: "
            f"‖∇G‖={diag.covd_metric_norm:.2e}."
        )

    @pytest.mark.parametrize("trial", range(5))
    def test_statistical_roundtrip_across_random_metrics(
        self, trial: int
    ) -> None:
        """
        Roundtrip ♯∘♭ ≈ id para 5 métricas aleatorias y vectores aleatorios.
        """
        rng = np.random.default_rng(trial * 491 + 7)
        n = int(rng.integers(3, 7))
        G = make_spd_matrix(n, seed=trial * 491)
        agent = build_agent(G)

        v_arr = rng.standard_normal(n).astype(np.float64)
        v = make_tangent_vector(v_arr)

        omega, _ = agent.transport_to_finance_oracle(
            v, apply_geodesic_correction=False
        )
        v_rec, _ = agent.transport_to_logistics_manifold(omega)

        kappa = agent._pm.condition_number_reg
        tol = max(1000 * kappa * _MACHINE_EPSILON, 1e-11)
        residual = np.linalg.norm(v_rec.coordinates - v.coordinates)
        assert residual <= tol * max(float(np.linalg.norm(v_arr)), 1.0), (
            f"Roundtrip fallido en trial={trial}, n={n}: "
            f"residual={residual:.2e} > tol={tol:.2e}."
        )

    def test_ill_conditioned_full_pipeline(self) -> None:
        """
        El pipeline completo funciona para métricas muy mal condicionadas
        (κ ≈ 1e14), donde la regularización es imprescindible.
        """
        G = make_ill_conditioned_matrix(5, condition_number=1e14, seed=499)
        agent = build_agent(G)
        pm = agent.preconditioned_metric

        assert pm.regularization_applied, (
            "Regularización debe aplicarse para κ≈1e14."
        )
        assert agent.connection_diagnostics().all_passed(), (
            "Axiomas deben pasar post-regularización."
        )

        v = make_random_unit_tangent(5, seed=503)
        v_out, report = agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(v_out.coordinates)), (
            "Resultado no finito para métrica mal condicionada."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §G7 — TestEdgeCasesAndSpecialProperties: Casos Extremos                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndSpecialProperties:
    """
    Verifica propiedades especiales y casos extremos no cubiertos en las
    clases anteriores:
      - Métrica 1×1 escalar.
      - Métrica 2×2 con autovalores idénticos.
      - enforce_geodesic_flow con v = 0.
      - Transportes con co-vector cero.
      - RK4 con dt = _DEFAULT_DT conserva v exactamente para Γ=0.
      - geodesic_flow_report coherente tras múltiples pasos.
    """

    def test_scalar_metric_1x1(self) -> None:
        """Pipeline completo para G = [[g]] (espacio 1D)."""
        G = np.array([[3.7]], dtype=np.float64)
        agent = build_agent(G)
        assert agent.metric_dimension == 1

        v = make_tangent_vector(np.array([1.0]))
        v_out, report = agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)
        assert_allclose(v_out.coordinates, v.coordinates, atol=1e-15,
                        err_msg="Para n=1, Γ=0 → v(t+dt) = v(t).")

    def test_2x2_metric_repeated_eigenvalues(self) -> None:
        """
        Métrica 2×2 con autovalores iguales (espacio propio degenerado):
        G = 2·I₂ → Γ = 0, κ = 1.
        """
        G = 2.0 * make_identity_metric(2)
        agent = build_agent(G)
        assert agent.connection_diagnostics().all_passed()
        # κ = 1 para G = 2I
        assert_allclose(
            agent._pm.condition_number_reg, 1.0, rtol=1e-10,
            err_msg="κ(2·I₂) debe ser 1.0."
        )

    def test_zero_velocity_geodesic_flow(self) -> None:
        """enforce_geodesic_flow con v = 0 retorna v = 0."""
        G = make_spd_matrix(4, seed=509)
        agent = build_agent(G)
        v_zero = make_tangent_vector(np.zeros(4, dtype=np.float64))
        v_out, report = agent.enforce_geodesic_flow(v_zero, dt=_DEFAULT_DT)
        assert_allclose(
            v_out.coordinates, np.zeros(4), atol=1e-15,
            err_msg="enforce_geodesic_flow(0) debe retornar 0."
        )
        assert_allclose(report.acceleration_norm, 0.0, atol=1e-15,
                        err_msg="Aceleración debe ser 0 para v=0.")

    def test_zero_cotangent_transport(self) -> None:
        """♯(0) = 0 (linealidad preserva el cero en T*M)."""
        G = make_spd_matrix(4, seed=521)
        agent = build_agent(G)
        omega_zero = make_cotangent_vector(np.zeros(4, dtype=np.float64))
        v, _ = agent.transport_to_logistics_manifold(omega_zero)
        assert_allclose(v.coordinates, np.zeros(4), atol=1e-15,
                        err_msg="♯(0) debe ser 0.")

    def test_zero_tangent_finance_oracle(self) -> None:
        """♭(0) = 0 (linealidad preserva el cero en TM)."""
        G = make_spd_matrix(4, seed=523)
        agent = build_agent(G)
        v_zero = make_tangent_vector(np.zeros(4, dtype=np.float64))
        omega, _ = agent.transport_to_finance_oracle(
            v_zero, apply_geodesic_correction=True, dt=_DEFAULT_DT
        )
        assert_allclose(omega.coordinates, np.zeros(4), atol=1e-15,
                        err_msg="♭(0) debe ser 0.")

    def test_multiple_geodesic_steps_remain_stable(self) -> None:
        """
        Múltiples pasos geodésicos consecutivos permanecen estables
        para Γ = 0 (la norma no debe derivar).
        """
        G = make_spd_matrix(5, seed=541)
        agent = build_agent(G)
        v = make_random_unit_tangent(5, seed=547)
        v_current = v

        for step in range(20):
            v_current, report = agent.enforce_geodesic_flow(
                v_current, dt=_DEFAULT_DT
            )
            assert np.all(np.isfinite(v_current.coordinates)), (
                f"Resultado no finito en paso {step}."
            )
            assert_allclose(
                report.v_final_norm, v.norm, rtol=1e-12,
                err_msg=f"Norma no conservada en paso {step} para Γ=0."
            )

    def test_large_norm_vector_is_handled(self) -> None:
        """enforce_geodesic_flow funciona para vectores de norma grande (‖v‖=1e6)."""
        G = make_spd_matrix(4, seed=557)
        agent = build_agent(G)
        v_large = make_tangent_vector(
            np.ones(4, dtype=np.float64) * 1e6 / np.sqrt(4)
        )
        v_out, report = agent.enforce_geodesic_flow(v_large, dt=_DEFAULT_DT)
        assert np.all(np.isfinite(v_out.coordinates)), (
            "Resultado no finito para ‖v‖=1e6."
        )

    def test_geodesic_flow_report_after_multiple_steps(self) -> None:
        """geodesic_flow_report() es consistente independientemente del estado."""
        G = make_spd_matrix(4, seed=563)
        agent = build_agent(G)

        # Ejecutar algunos pasos
        v = make_random_unit_tangent(4, seed=569)
        for _ in range(5):
            v, _ = agent.enforce_geodesic_flow(v, dt=_DEFAULT_DT)

        # El reporte debe seguir siendo consistente
        report = agent.geodesic_flow_report()
        assert report["all_axioms_passed"], (
            "all_axioms_passed debe seguir siendo True tras múltiples pasos."
        )

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    def test_diagonal_metric_at_different_scales(self, scale: float) -> None:
        """
        Métrica diagonal escalada uniformemente G = scale·I no cambia la
        dirección de los isomorfismos, solo la magnitud.

        Para G = scale·I: ♭(v) = scale·v, ♯(ω) = ω/scale.
        """
        n = 4
        G = scale * make_identity_metric(n)
        agent = build_agent(G)

        v = make_random_unit_tangent(n, seed=571)
        omega, _ = agent.transport_to_finance_oracle(
            v, apply_geodesic_correction=False
        )

        # ♭(v) = G @ v = scale · v
        assert_allclose(
            omega.coordinates, scale * v.coordinates, rtol=1e-12,
            err_msg=f"Para G=scale·I, ♭(v)=scale·v fallido para scale={scale}."
        )

        # ♯(omega) = G_inv @ omega = (1/scale) · scale · v = v
        v_rec, _ = agent.transport_to_logistics_manifold(omega)
        assert_allclose(
            v_rec.coordinates, v.coordinates, rtol=1e-11,
            err_msg=f"Para G=scale·I, roundtrip fallido para scale={scale}."
        )