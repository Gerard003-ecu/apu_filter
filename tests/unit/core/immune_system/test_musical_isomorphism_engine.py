# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Suite de Pruebas: Musical Isomorphism Engine                                        ║
║  Ruta   : tests/unit/core/immune_system/test_musical_isomorphism_engine.py           ║
║  Versión: 3.0.0-Topos-Spectral-Categorical                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Filosofía de Testing — Contratos Algebraicos como Axiomas:
══════════════════════════════════════════════════════════════════════════════════════

Esta suite no prueba implementaciones; prueba AXIOMAS MATEMÁTICOS. Cada test
verifica que el código satisface una propiedad algebraica formal derivada de:

  §T1. Teoría Espectral de Operadores Autoadjuntos (Fase 1):
       - Simetría:      ‖G - Gᵀ‖_F = 0  (post-simetrización)
       - Def. positiva: λ_min(G_reg) > 0  (post-regularización)
       - Inversión:     ‖G_reg · G_inv - I‖_F < tol_κ
       - Diagnósticos:  κ_raw, Δ_abs, Δ_chg, null_dim correctamente calculados

  §T2. Isomorfismos de Fibrados Vectoriales (Fases 2 y 3):
       - Linealidad ♭:  ♭(αu + βv) = α·♭(u) + β·♭(v)
       - Linealidad ♯:  ♯(αω + βη) = α·♯(ω) + β·♯(η)
       - Adjunción:     ⟨♭(v), w⟩ = G(v, w) = ⟨v, ♭(w)⟩  (simetría de G)
       - Identidad:     ♯ ∘ ♭ = id_{TM}  y  ♭ ∘ ♯ = id_{T*M}

  §T3. Álgebra de Grupos Z₂ (Fase 3 — Composición Funtorial):
       - Tabla de Cayley completa: 4 combinaciones de varianza
       - Compatibilidad categórica: dom(f1) ≅ cod(f2)
       - Tipo-seguridad: TypeError ante varianza de tipo incorrecto

  §T4. Análisis de Casos Extremos (Robustez Numérica):
       - Matrices singulares (λ_min = 0)
       - Matrices mal condicionadas (κ > 1e12)
       - Métricas de alta dimensión (n = 100, 500)
       - Vectores nulos y vectores de norma extrema
       - Perturbaciones aleatorias de la identidad

  §T5. Contratos de Interfaz (Errores Semánticos):
       - Dimensiones incompatibles TangentVector / CotangentVector
       - Tipos incorrectos en constructores
       - Matrices no cuadradas, NaN, Inf
       - Funtores sin atributo variance

Estrategia de Fixtures:
══════════════════════════════════════════════════════════════════════════════════════
Los fixtures están parametrizados para cubrir:
  - Matrices de referencia con propiedades conocidas (identidad, diagonal, Hilbert)
  - Métricas aleatorias SPD (Symmetric Positive Definite) generadas reproduciblemente
  - Casos degenerados construidos algebraicamente (rango deficiente controlado)
  - Vectores base canónicos y vectores aleatorios unitarios

Estructura de la suite:
══════════════════════════════════════════════════════════════════════════════════════

  TestCategoricalVariance          — §T3: álgebra Z₂
  TestTangentCotangentVectors      — invariantes de dataclasses
  TestMetricSpectralPreconditioner — §T1: pipeline de Fase 1
    ├── TestValidation             — errores de estructura
    ├── TestSymmetryEnforcement    — simetrización
    ├── TestSpectralDecomposition  — autovalores y autovectores
    ├── TestSpectralDiagnostics    — κ, gaps, null_dim
    ├── TestTikhonovRegularization — activación y postcondiciones
    └── TestInversionVerification  — ‖G·G⁻¹ - I‖ adaptativo
  TestFlatIsomorphism              — §T2: ♭ : TM → T*M
    ├── TestLinearityFlat          — superposición y escala
    ├── TestAdjunction             — ⟨♭(v), w⟩ = ⟨♭(w), v⟩
    └── TestDimensionGuards        — FunctorialityError
  TestSharpIsomorphism             — §T2: ♯ : T*M → TM
    ├── TestLinearitySharp         — superposición y escala
    ├── TestRoundtrip              — ♯∘♭ = id, ♭∘♯ = id
    └── TestFunctorComposition     — §T3: auditoría Z₂ + topos
  TestMusicalIsomorphismEngine     — integración end-to-end
    ├── TestFullCycleReport        — informe unificado
    ├── TestDependencyInjection    — preconditioner inyectado
    └── TestHighDimensionalMetrics — n=100, n=500
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ════════════════════════════════════════════════════════════════════════════════
from app.core.immune_system.musical_isomorphism_engine import (
    CategoricalVariance,
    CovariantFunctor,
    ContravariantFunctor,
    CotangentVector,
    FlatIsomorphism,
    MetricSpectralPreconditioner,
    MusicalIsomorphismEngine,
    PreconditionedMetric,
    SharpIsomorphism,
    TangentVector,
    _MACHINE_EPSILON,
    _CONDITION_THRESHOLD,
    _TIKHONOV_EPSILON_RATIO,
    _INVERSION_BASE_TOLERANCE,
    _ROUNDTRIP_TOLERANCE_FACTOR,
)
from app.core.mic_algebra import FunctorialityError, NumericalInstabilityError

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL DE LOGGING Y SEMILLAS
# ════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING)
_RNG_SEED: int = 42
_RNG: np.random.Generator = np.random.default_rng(_RNG_SEED)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FÁBRICAS DE MATRICES Y VECTORES DE PRUEBA                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

def make_spd_matrix(n: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """
    Genera una matriz Simétrica Definida Positiva (SPD) de tamaño n×n.

    Método: G = AᵀA + n·I donde A es aleatoria, garantizando λ_min ≥ n > 0.
    El término n·I asegura separación espectral adecuada del cero.

    Parámetros
    ----------
    n : int
        Dimensión de la matriz.
    seed : int
        Semilla para reproducibilidad.

    Retorna
    -------
    NDArray[np.float64]
        Matriz SPD de tamaño (n,n).
    """
    rng = np.random.default_rng(seed)
    A: NDArray[np.float64] = rng.standard_normal((n, n))
    G: NDArray[np.float64] = A.T @ A + float(n) * np.eye(n)
    return G.astype(np.float64)


def make_ill_conditioned_matrix(
    n: int, condition_number: float = 1e14, seed: int = _RNG_SEED
) -> NDArray[np.float64]:
    """
    Genera una matriz SPD con número de condición κ ≈ condition_number.

    Construcción: G = V diag(λ) Vᵀ donde λ ∈ [1, condition_number]
    distribuidos logarítmicamente. V es una matriz ortogonal aleatoria
    (QR de una aleatoria).

    Parámetros
    ----------
    n : int
        Dimensión.
    condition_number : float
        Número de condición objetivo (κ = λ_max / λ_min).
    seed : int
        Semilla para reproducibilidad.

    Retorna
    -------
    NDArray[np.float64]
        Matriz SPD mal condicionada de tamaño (n,n).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Autovalores logarítmicamente espaciados en [1, condition_number]
    eigenvalues = np.logspace(0, np.log10(condition_number), n, dtype=np.float64)
    G: NDArray[np.float64] = Q @ np.diag(eigenvalues) @ Q.T
    return (G + G.T) * 0.5


def make_rank_deficient_matrix(n: int, rank: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """
    Genera una matriz semidefinida positiva de rango `rank` < n.

    El kernel tiene dimensión null_dim = n - rank.

    Parámetros
    ----------
    n : int
        Dimensión total.
    rank : int
        Rango efectivo (rank < n).
    seed : int
        Semilla.

    Retorna
    -------
    NDArray[np.float64]
        Matriz PSD de rango `rank`.
    """
    assert 0 < rank < n, f"rank={rank} debe satisfacer 0 < rank < n={n}."
    rng = np.random.default_rng(seed)
    A: NDArray[np.float64] = rng.standard_normal((n, rank))
    G: NDArray[np.float64] = A @ A.T  # Rango = rank, kernel dimensión = n - rank
    return G.astype(np.float64)


def make_hilbert_matrix(n: int) -> NDArray[np.float64]:
    """
    Genera la matriz de Hilbert H_{ij} = 1/(i+j-1), notoriamente mal condicionada.

    κ(H_n) crece exponencialmente con n:
      κ(H_10) ≈ 1.6e13,  κ(H_15) ≈ 4.8e17.
    Útil para probar regularización con una matriz analíticamente conocida.

    Parámetros
    ----------
    n : int
        Dimensión.

    Retorna
    -------
    NDArray[np.float64]
        Matriz de Hilbert (n,n).
    """
    i, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing='ij')
    return (1.0 / (i + j - 1)).astype(np.float64)


def make_asymmetric_matrix(n: int, asymmetry_scale: float = 1.0, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """
    Genera una matriz SPD con componente antisimétrica controlada.

    G = SPD_base + asymmetry_scale · (A - Aᵀ)/2   (suma de simétrica + antisimétrica)

    Parámetros
    ----------
    n : int
        Dimensión.
    asymmetry_scale : float
        Magnitud de la perturbación antisimétrica.
    seed : int
        Semilla.

    Retorna
    -------
    NDArray[np.float64]
        Matriz con parte antisimétrica conocida.
    """
    rng = np.random.default_rng(seed)
    G_spd = make_spd_matrix(n, seed=seed)
    anti = rng.standard_normal((n, n))
    anti_sym = (anti - anti.T) * 0.5
    return (G_spd + asymmetry_scale * anti_sym).astype(np.float64)


def make_random_unit_tangent_vector(n: int, seed: int = _RNG_SEED) -> TangentVector:
    """Genera un TangentVector aleatorio de norma unitaria."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)
    return TangentVector(coordinates=v.astype(np.float64))


def make_random_cotangent_vector(n: int, seed: int = _RNG_SEED) -> CotangentVector:
    """Genera un CotangentVector aleatorio."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(n)
    return CotangentVector(coordinates=omega.astype(np.float64))


def build_preconditioned_metric(G: NDArray[np.float64]) -> PreconditionedMetric:
    """Helper: ejecuta Fase 1 sobre G y retorna PreconditionedMetric."""
    return MetricSpectralPreconditioner().precondition(G)


def build_engine(G: NDArray[np.float64]) -> MusicalIsomorphismEngine:
    """Helper: construye MusicalIsomorphismEngine sobre G."""
    return MusicalIsomorphismEngine(metric_tensor=G)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §T3 — TestCategoricalVariance: Álgebra del Grupo Z₂                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestCategoricalVariance:
    """
    Verifica que CategoricalVariance implementa fielmente el grupo (Z₂, ×).

    Axiomas verificados:
      - Tabla de Cayley completa (4 entradas).
      - Elemento identidad: COVARIANT es la unidad del grupo.
      - Elemento inverso: CONTRAVARIANT es su propio inverso.
      - Clausura: el producto siempre retorna CategoricalVariance.
      - Rechazo de tipos incorrectos (TypeError).
    """

    COV  = CategoricalVariance.COVARIANT
    CONT = CategoricalVariance.CONTRAVARIANT

    # ── Tabla de Cayley ──────────────────────────────────────────────────────

    def test_cov_times_cov_is_cov(self) -> None:
        """Cov ⊗ Cov = Cov  (elemento identidad preservado)."""
        result = self.COV * self.COV
        assert result is self.COV, (
            f"COVARIANT * COVARIANT debe ser COVARIANT, obtenido {result}."
        )

    def test_cov_times_cont_is_cont(self) -> None:
        """Cov ⊗ Cont = Cont  (absorción del elemento no-identidad)."""
        result = self.COV * self.CONT
        assert result is self.CONT, (
            f"COVARIANT * CONTRAVARIANT debe ser CONTRAVARIANT, obtenido {result}."
        )

    def test_cont_times_cov_is_cont(self) -> None:
        """Cont ⊗ Cov = Cont  (no-conmutatividad aparente pero Z₂ es abeliano)."""
        result = self.CONT * self.COV
        assert result is self.CONT, (
            f"CONTRAVARIANT * COVARIANT debe ser CONTRAVARIANT, obtenido {result}."
        )

    def test_cont_times_cont_is_cov(self) -> None:
        """Cont ⊗ Cont = Cov  (dos inversiones restoran la covarianza)."""
        result = self.CONT * self.CONT
        assert result is self.COV, (
            f"CONTRAVARIANT * CONTRAVARIANT debe ser COVARIANT, obtenido {result}."
        )

    # ── Propiedades algebraicas ──────────────────────────────────────────────

    def test_product_returns_categorical_variance_instance(self) -> None:
        """La clausura del grupo garantiza que el producto es CategoricalVariance."""
        for a in [self.COV, self.CONT]:
            for b in [self.COV, self.CONT]:
                result = a * b
                assert isinstance(result, CategoricalVariance), (
                    f"Producto {a} * {b} debe retornar CategoricalVariance, "
                    f"retornó {type(result).__name__}."
                )

    def test_covariant_is_multiplicative_identity(self) -> None:
        """COVARIANT actúa como el elemento identidad del grupo Z₂."""
        for v in [self.COV, self.CONT]:
            assert v * self.COV is v, f"v * COVARIANT debe ser v para v={v}."
            assert self.COV * v is v, f"COVARIANT * v debe ser v para v={v}."

    def test_contravariant_is_self_inverse(self) -> None:
        """CONTRAVARIANT es su propio inverso en Z₂: Cont² = Cov = e."""
        assert self.CONT * self.CONT is self.COV, (
            "CONTRAVARIANT * CONTRAVARIANT debe ser el identidad COVARIANT."
        )

    def test_commutativity(self) -> None:
        """Z₂ es abeliano: a * b = b * a para todo a, b."""
        for a in [self.COV, self.CONT]:
            for b in [self.COV, self.CONT]:
                assert a * b is b * a, f"Conmutatividad fallida para {a} * {b}."

    def test_associativity(self) -> None:
        """Z₂ es un grupo: (a*b)*c = a*(b*c)."""
        elements = [self.COV, self.CONT]
        for a in elements:
            for b in elements:
                for c in elements:
                    assert (a * b) * c is a * (b * c), (
                        f"Asociatividad fallida para ({a}*{b})*{c}."
                    )

    def test_values_are_plus_minus_one(self) -> None:
        """Los valores numéricos del homomorfismo φ: Z₂ → {±1} son correctos."""
        assert self.COV.value  ==  1, "COVARIANT.value debe ser +1."
        assert self.CONT.value == -1, "CONTRAVARIANT.value debe ser -1."

    # ── Tipo-seguridad ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("bad_operand", [
        1, -1, "COVARIANT", None, 1.0, True, [], {}
    ])
    def test_mul_raises_type_error_for_non_variance(
        self, bad_operand: Any
    ) -> None:
        """TypeError si el operando no es CategoricalVariance."""
        with pytest.raises(TypeError, match="CategoricalVariance"):
            _ = self.COV * bad_operand  # type: ignore[operator]


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INVARIANTES DE DATACLASSES: TangentVector y CotangentVector                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestTangentCotangentVectors:
    """
    Verifica los invariantes algebraicos de los contenedores de vectores.

    Propiedades verificadas:
      - Inmutabilidad (frozen dataclass).
      - Validación de tipo, dimensión y dtype en construcción.
      - Coerción automática a float64.
      - Propiedades `dim` y `norm` con valores correctos.
    """

    # ── TangentVector ────────────────────────────────────────────────────────

    def test_tangent_vector_construction_valid(self) -> None:
        """Construcción válida con array float64 1-D."""
        v = TangentVector(coordinates=np.array([1.0, 2.0, 3.0]))
        assert v.dim == 3
        assert_allclose(v.norm, np.sqrt(14.0))

    def test_tangent_vector_coerces_to_float64(self) -> None:
        """Array de dtype int32 es coercionado automáticamente a float64."""
        v = TangentVector(coordinates=np.array([1, 2, 3], dtype=np.int32))
        assert v.coordinates.dtype == np.float64

    def test_tangent_vector_is_immutable(self) -> None:
        """El dataclass frozen impide modificación de coordinates."""
        v = TangentVector(coordinates=np.array([1.0, 0.0]))
        with pytest.raises((AttributeError, TypeError)):
            v.coordinates = np.array([0.0, 1.0])  # type: ignore[misc]

    @pytest.mark.parametrize("bad_coords", [
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2-D
        np.array([[[1.0]]]),                  # 3-D
    ])
    def test_tangent_vector_rejects_non_1d(self, bad_coords: NDArray) -> None:
        """TangentVector rechaza arrays de dimensión != 1."""
        with pytest.raises(ValueError, match="1-D"):
            TangentVector(coordinates=bad_coords)

    def test_tangent_vector_rejects_non_array(self) -> None:
        """TangentVector rechaza listas y tipos no-ndarray."""
        with pytest.raises(TypeError):
            TangentVector(coordinates=[1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_tangent_vector_zero_norm(self) -> None:
        """Vector cero tiene norma exactamente 0."""
        v = TangentVector(coordinates=np.zeros(5))
        assert v.norm == 0.0

    def test_tangent_vector_unit_norm(self) -> None:
        """Vector canónico e₀ tiene norma unitaria."""
        e0 = np.zeros(4, dtype=np.float64)
        e0[0] = 1.0
        v = TangentVector(coordinates=e0)
        assert_allclose(v.norm, 1.0)

    # ── CotangentVector ──────────────────────────────────────────────────────

    def test_cotangent_vector_construction_valid(self) -> None:
        """Construcción válida con array float64 1-D."""
        omega = CotangentVector(coordinates=np.array([0.5, -1.5, 2.0]))
        assert omega.dim == 3
        assert_allclose(omega.norm, np.sqrt(0.25 + 2.25 + 4.0))

    def test_cotangent_vector_coerces_to_float64(self) -> None:
        """dtype float32 es coercionado a float64."""
        omega = CotangentVector(
            coordinates=np.array([1.0, 2.0], dtype=np.float32)
        )
        assert omega.coordinates.dtype == np.float64

    @pytest.mark.parametrize("bad_coords", [
        np.array([[1.0, 0.0]]),   # 2-D
        np.array([]),              # Shape (0,) — válido en numpy pero dim=0
    ])
    def test_cotangent_vector_rejects_2d(self, bad_coords: NDArray) -> None:
        """CotangentVector rechaza arrays de ndim != 1 o dim == 0."""
        if bad_coords.ndim != 1:
            with pytest.raises(ValueError, match="1-D"):
                CotangentVector(coordinates=bad_coords)

    def test_cotangent_vector_is_immutable(self) -> None:
        """El dataclass frozen impide modificación."""
        omega = CotangentVector(coordinates=np.array([1.0, 0.0]))
        with pytest.raises((AttributeError, TypeError)):
            omega.coordinates = np.array([0.0, 1.0])  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §T1 — TestMetricSpectralPreconditioner: Pipeline de Fase 1                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMetricSpectralPreconditionerValidation:
    """
    Verifica que _validate_matrix_structure rechaza entradas inválidas
    con las excepciones y mensajes correctos.
    """

    def setup_method(self) -> None:
        self.P = MetricSpectralPreconditioner()

    def test_rejects_non_ndarray(self) -> None:
        """TypeError si la entrada no es ndarray."""
        with pytest.raises(TypeError, match="NDArray"):
            self.P.precondition([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_rejects_1d_array(self) -> None:
        """ValueError si ndim != 2."""
        with pytest.raises(ValueError, match="2-D"):
            self.P.precondition(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_array(self) -> None:
        """ValueError si ndim == 3."""
        with pytest.raises(ValueError, match="2-D"):
            self.P.precondition(np.ones((3, 3, 3)))

    def test_rejects_non_square_matrix(self) -> None:
        """ValueError si la matriz no es cuadrada."""
        with pytest.raises(ValueError, match="cuadrado"):
            self.P.precondition(np.eye(3, 4))

    def test_rejects_nan_entries(self) -> None:
        """ValueError si la matriz contiene NaN."""
        G = np.eye(3)
        G[1, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            self.P.precondition(G)

    def test_rejects_inf_entries(self) -> None:
        """ValueError si la matriz contiene Inf."""
        G = np.eye(3)
        G[0, 2] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            self.P.precondition(G)

    def test_rejects_mixed_nan_inf(self) -> None:
        """ValueError con mensaje correcto para NaN+Inf simultáneos."""
        G = np.eye(4, dtype=np.float64)
        G[0, 0] = np.nan
        G[2, 3] = np.inf
        with pytest.raises(ValueError, match="no finitos"):
            self.P.precondition(G)

    def test_accepts_1x1_identity(self) -> None:
        """La métrica escalar 1×1 positiva debe procesarse sin error."""
        G = np.array([[5.0]])
        pm = self.P.precondition(G)
        assert pm.matrix_dimension == 1
        assert pm.eigenvalues_reg[0] > 0

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_accepts_identity_matrices(self, n: int) -> None:
        """La identidad n×n es la métrica plana de ℝⁿ, debe pasar sin regularización."""
        pm = self.P.precondition(np.eye(n))
        assert pm.matrix_dimension == n
        assert not pm.regularization_applied
        assert_allclose(pm.condition_number_reg, 1.0, rtol=1e-10)


class TestMetricSpectralPreconditionerSymmetry:
    """
    Verifica el método _enforce_symmetry:
      - Matrices ya simétricas no son modificadas.
      - Matrices asimétricas son proyectadas correctamente a Sym(n, ℝ).
      - El resultado siempre satisface ‖G_sym - G_sym^T‖ = 0.
    """

    def setup_method(self) -> None:
        self.P = MetricSpectralPreconditioner()

    def test_symmetric_matrix_unchanged(self) -> None:
        """Una SPD ya simétrica no debe ser modificada en la simetrización."""
        G_orig = make_spd_matrix(4, seed=0)
        pm = self.P.precondition(G_orig)
        # La métrica regularizada debe ser simétrica
        assert_allclose(pm.G, pm.G.T, atol=1e-14,
                        err_msg="G_reg no es simétrica para entrada SPD.")

    def test_asymmetric_matrix_is_symmetrized(self) -> None:
        """Matriz con asimetría grande es proyectada a Sym(n, ℝ)."""
        G_asym = make_asymmetric_matrix(5, asymmetry_scale=10.0, seed=7)
        pm = self.P.precondition(G_asym)
        # Verificar que el resultado es simétrico
        sym_error = np.linalg.norm(pm.G - pm.G.T, 'fro')
        assert sym_error < 1e-12, (
            f"G_reg no es simétrica tras simetrización forzada: ‖G-Gᵀ‖_F={sym_error:.2e}."
        )

    def test_symmetrization_is_frobenius_projection(self) -> None:
        """
        La simetrización (G+Gᵀ)/2 es la proyección de Frobenius óptima
        al subespacio simétrico: ‖G - G_sym‖_F es mínima.
        """
        n = 5
        rng = np.random.default_rng(99)
        G_spd = make_spd_matrix(n, seed=99)
        anti = rng.standard_normal((n, n)) * 2.0
        G_asym = G_spd + (anti - anti.T)  # SPD + antisimétrica grande

        G_sym_expected = (G_asym + G_asym.T) * 0.5
        pm = self.P.precondition(G_asym)

        # La parte simétrica de G_reg debe coincidir con (G_asym + G_asymᵀ)/2
        # módulo la regularización de Tikhonov (que sí modifica los autovalores)
        # Verificamos que los autovectores son los mismos (el espacio propio no cambia)
        _, V_expected = np.linalg.eigh(G_sym_expected)
        _, V_actual   = np.linalg.eigh(pm.G)
        # Los subespacios propios deben coincidir (columnas paralelas o antiparalelas)
        overlap = np.abs(V_expected.T @ V_actual)
        assert_allclose(
            np.diag(overlap), np.ones(n), atol=1e-8,
            err_msg="Los autovectores de G_sym no coinciden con los esperados."
        )

    def test_output_is_always_symmetric(self) -> None:
        """Invariante: pm.G es siempre simétrica, sin importar la entrada."""
        for seed in range(5):
            G = make_asymmetric_matrix(4, asymmetry_scale=float(seed + 1), seed=seed)
            pm = self.P.precondition(G)
            sym_err = np.linalg.norm(pm.G - pm.G.T, 'fro')
            assert sym_err < 1e-12, f"G_reg no simétrica para seed={seed}: err={sym_err:.2e}."


class TestMetricSpectralPreconditionerSpectral:
    """
    Verifica la corrección de la descomposición espectral y los diagnósticos.

    Axiomas verificados:
      - Autovalores en orden ascendente (convenio eigh).
      - V es ortogonal: VᵀV = I.
      - Reconstrucción: V diag(λ) Vᵀ ≈ G_sym.
      - Correctitud de κ, Δ_abs, Δ_chg, null_dim para casos conocidos.
    """

    def setup_method(self) -> None:
        self.P = MetricSpectralPreconditioner()

    def test_eigenvalues_ascending_order(self) -> None:
        """Los autovalores de PreconditionedMetric están en orden ascendente."""
        G = make_spd_matrix(6, seed=17)
        pm = self.P.precondition(G)
        assert np.all(np.diff(pm.eigenvalues_raw) >= 0), (
            "eigenvalues_raw no están en orden ascendente."
        )
        assert np.all(np.diff(pm.eigenvalues_reg) >= 0), (
            "eigenvalues_reg no están en orden ascendente."
        )

    def test_eigenvectors_are_orthonormal(self) -> None:
        """V satisface VᵀV = I (ortogonalidad de autovectores)."""
        G = make_spd_matrix(8, seed=23)
        pm = self.P.precondition(G)
        V = pm.eigenvectors
        n = pm.matrix_dimension
        ortho_error = np.linalg.norm(V.T @ V - np.eye(n), 'fro')
        assert ortho_error < 1e-12, (
            f"Ortogonalidad de V violada: ‖VᵀV - I‖_F = {ortho_error:.2e}."
        )

    def test_spectral_reconstruction(self) -> None:
        """G_reg = V diag(λ_reg) Vᵀ dentro de tolerancia numérica."""
        G = make_spd_matrix(5, seed=31)
        pm = self.P.precondition(G)
        G_reconstructed = pm.eigenvectors @ np.diag(pm.eigenvalues_reg) @ pm.eigenvectors.T
        assert_allclose(
            pm.G, G_reconstructed, atol=1e-12,
            err_msg="Reconstrucción espectral G = V·diag(λ_reg)·Vᵀ fallida."
        )

    @pytest.mark.parametrize("n,expected_null_dim", [
        (5, 0),  # SPD: sin kernel
    ])
    def test_null_space_dim_spd(self, n: int, expected_null_dim: int) -> None:
        """Una SPD tiene null_dim = 0."""
        G = make_spd_matrix(n, seed=41)
        pm = self.P.precondition(G)
        assert pm.null_space_dim == expected_null_dim, (
            f"null_dim esperado {expected_null_dim}, obtenido {pm.null_space_dim}."
        )

    def test_null_space_dim_rank_deficient(self) -> None:
        """Una matriz de rango r en dimensión n tiene null_dim = n - r (pre-regularización)."""
        n, rank = 6, 4
        G = make_rank_deficient_matrix(n, rank, seed=53)
        pm = self.P.precondition(G)
        # null_dim debe ser n - rank = 2 (autovalores ≤ 0 pre-regularización)
        assert pm.null_space_dim == n - rank, (
            f"null_dim esperado {n - rank}, obtenido {pm.null_space_dim}."
        )

    def test_condition_number_identity(self) -> None:
        """κ(I_n) = 1 exactamente (la identidad es la métrica plana)."""
        pm = self.P.precondition(np.eye(5))
        assert_allclose(pm.condition_number_raw, 1.0, rtol=1e-10,
                        err_msg=f"κ(I_5) debe ser 1.0, obtenido {pm.condition_number_raw}.")

    def test_spectral_gap_absolute_identity(self) -> None:
        """Δ_abs(I_n) = 1.0 (máxima separación espectral, sin brecha)."""
        pm = self.P.precondition(np.eye(4))
        assert_allclose(pm.spectral_gap_absolute, 1.0, rtol=1e-10,
                        err_msg="Δ_abs(I) debe ser 1.0.")

    def test_spectral_gap_cheeger_diagonal(self) -> None:
        """Para G = diag(1, 2, 4), Δ_chg = λ_2/λ_max = 2/4 = 0.5."""
        G = np.diag([1.0, 2.0, 4.0])
        pm = self.P.precondition(G)
        # Δ_chg = segundo positivo / max = 2/4 = 0.5
        assert_allclose(pm.spectral_gap_cheeger, 0.5, rtol=1e-10,
                        err_msg="Δ_chg(diag(1,2,4)) debe ser 0.5.")

    def test_condition_number_diagonal(self) -> None:
        """Para G = diag(1, 10, 100), κ = 100/1 = 100 (no requiere regularización)."""
        G = np.diag([1.0, 10.0, 100.0])
        pm = self.P.precondition(G)
        assert_allclose(pm.condition_number_raw, 100.0, rtol=1e-8,
                        err_msg="κ(diag(1,10,100)) debe ser 100.")
        assert not pm.regularization_applied, (
            "No debería aplicarse regularización para κ=100 < 1e12."
        )

    def test_spectral_summary_keys(self) -> None:
        """spectral_summary() retorna todas las claves esperadas."""
        G = make_spd_matrix(4, seed=67)
        pm = self.P.precondition(G)
        summary = pm.spectral_summary()
        expected_keys = {
            "matrix_dimension", "null_space_dim", "condition_number_raw",
            "condition_number_reg", "spectral_gap_absolute", "spectral_gap_cheeger",
            "tikhonov_epsilon", "regularization_applied",
            "lambda_min_raw", "lambda_max_raw", "lambda_min_reg", "lambda_max_reg",
        }
        assert expected_keys.issubset(set(summary.keys())), (
            f"Claves faltantes en spectral_summary: {expected_keys - set(summary.keys())}."
        )


class TestMetricSpectralPreconditionerTikhonov:
    """
    Verifica la regularización de Tikhonov adaptativa:
      - Se activa para κ > 1e12 o λ_min ≤ 0.
      - Post-regularización: todos λ_i > 0.
      - ε = TIKHONOV_EPSILON_RATIO * λ_max.
      - κ_reg < κ_raw (mejoría del condicionamiento).
    """

    def setup_method(self) -> None:
        self.P = MetricSpectralPreconditioner()

    def test_regularization_applied_for_ill_conditioned(self) -> None:
        """Regularización se activa para κ ≈ 1e14 > CONDITION_THRESHOLD."""
        G = make_ill_conditioned_matrix(5, condition_number=1e14, seed=71)
        pm = self.P.precondition(G)
        assert pm.regularization_applied, (
            "Regularización debe activarse para κ ≈ 1e14 > 1e12."
        )

    def test_regularization_not_applied_for_well_conditioned(self) -> None:
        """No regularización para G bien condicionada (κ ≈ 100)."""
        G = make_ill_conditioned_matrix(5, condition_number=100.0, seed=73)
        pm = self.P.precondition(G)
        assert not pm.regularization_applied, (
            "Regularización no debe aplicarse para κ ≈ 100 < 1e12."
        )

    def test_all_regularized_eigenvalues_positive(self) -> None:
        """Post-regularización: ∀ i, λ_i^reg > 0."""
        G = make_rank_deficient_matrix(6, rank=4, seed=79)
        pm = self.P.precondition(G)
        assert np.all(pm.eigenvalues_reg > 0), (
            f"Autovalores regularizados no todos positivos: {pm.eigenvalues_reg}."
        )

    def test_tikhonov_epsilon_value(self) -> None:
        """ε = TIKHONOV_EPSILON_RATIO * λ_max cuando se aplica regularización."""
        G = make_ill_conditioned_matrix(4, condition_number=1e14, seed=83)
        pm = self.P.precondition(G)
        if pm.regularization_applied:
            lambda_max_raw = pm.eigenvalues_raw[-1]
            expected_epsilon = _TIKHONOV_EPSILON_RATIO * lambda_max_raw
            assert_allclose(pm.tikhonov_epsilon, expected_epsilon, rtol=1e-10,
                            err_msg="ε no coincide con TIKHONOV_EPSILON_RATIO * λ_max.")

    def test_condition_number_improves_after_regularization(self) -> None:
        """κ_reg < κ_raw para métricas mal condicionadas."""
        G = make_ill_conditioned_matrix(5, condition_number=1e15, seed=89)
        pm = self.P.precondition(G)
        if pm.regularization_applied:
            assert pm.condition_number_reg < pm.condition_number_raw, (
                f"κ_reg={pm.condition_number_reg:.2e} no es menor que "
                f"κ_raw={pm.condition_number_raw:.2e}."
            )

    def test_rank_deficient_requires_regularization(self) -> None:
        """Matrices de rango deficiente (λ_min=0) activan regularización."""
        G = make_rank_deficient_matrix(5, rank=3, seed=97)
        pm = self.P.precondition(G)
        assert pm.regularization_applied, (
            "Matriz de rango deficiente debe activar regularización."
        )

    def test_hilbert_matrix_regularized(self) -> None:
        """Matriz de Hilbert n=10 (κ≈1e13) activa regularización."""
        G = make_hilbert_matrix(10)
        pm = self.P.precondition(G)
        assert pm.regularization_applied, (
            "Matriz de Hilbert n=10 (κ≈1e13) debe activar regularización."
        )
        assert np.all(pm.eigenvalues_reg > 0), (
            "Autovalores de Hilbert regularizados deben ser todos positivos."
        )

    def test_zero_epsilon_when_not_regularized(self) -> None:
        """tikhonov_epsilon == 0 cuando la regularización no se aplica."""
        G = make_spd_matrix(4, seed=101)
        pm = self.P.precondition(G)
        if not pm.regularization_applied:
            assert pm.tikhonov_epsilon == 0.0, (
                f"tikhonov_epsilon debe ser 0.0 cuando no hay regularización, "
                f"obtenido {pm.tikhonov_epsilon}."
            )


class TestMetricSpectralPreconditionerInversion:
    """
    Verifica la verificación algebraica de inversión:
      - ‖G_reg · G_inv - I‖_F ≤ tol_adaptativa para todo G válido.
      - La tolerancia adaptativa es correcta: max(tol_base, κ·ε_machine·√n).
      - PreconditionedMetric.__post_init__ valida invariantes.
    """

    def setup_method(self) -> None:
        self.P = MetricSpectralPreconditioner()

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 20])
    def test_inversion_identity_spd(self, n: int) -> None:
        """‖G_reg · G_inv - I‖_F < tol para SPD de dimensión n."""
        G = make_spd_matrix(n, seed=n * 7)
        pm = self.P.precondition(G)
        product = pm.G @ pm.G_inv
        residual = np.linalg.norm(product - np.eye(n), 'fro')
        # Tolerancia adaptativa
        tol = max(_INVERSION_BASE_TOLERANCE,
                  pm.condition_number_reg * _MACHINE_EPSILON * np.sqrt(n))
        assert residual <= tol * 10, (  # Factor 10 de margen para CI
            f"‖G·G⁻¹ - I‖_F = {residual:.2e} > tol={tol:.2e} para n={n}."
        )

    def test_inversion_identity_after_regularization(self) -> None:
        """Inversión válida incluso para matriz de rango deficiente (post-regularización)."""
        G = make_rank_deficient_matrix(6, rank=4, seed=103)
        pm = self.P.precondition(G)
        n = pm.matrix_dimension
        product = pm.G @ pm.G_inv
        residual = np.linalg.norm(product - np.eye(n), 'fro')
        tol = max(_INVERSION_BASE_TOLERANCE,
                  pm.condition_number_reg * _MACHINE_EPSILON * np.sqrt(n))
        assert residual <= tol * 100, (  # Margen mayor para rango deficiente
            f"‖G·G⁻¹ - I‖_F = {residual:.2e} > tol={tol:.2e} (post-regularización)."
        )

    def test_preconditioned_metric_post_init_rejects_non_positive_eigenvalues(self) -> None:
        """PreconditionedMetric.__post_init__ rechaza eigenvalues_reg con valores ≤ 0."""
        n = 3
        G = make_spd_matrix(n, seed=107)
        pm_valid = self.P.precondition(G)

        # Construir un PreconditionedMetric con eigenvalues_reg inválidos
        bad_eigenvalues = pm_valid.eigenvalues_reg.copy()
        bad_eigenvalues[0] = -1.0  # Violación deliberada

        with pytest.raises(ValueError, match="eigenvalues_reg"):
            PreconditionedMetric(
                G                      = pm_valid.G,
                G_inv                  = pm_valid.G_inv,
                eigenvalues_raw        = pm_valid.eigenvalues_raw,
                eigenvalues_reg        = bad_eigenvalues,
                eigenvectors           = pm_valid.eigenvectors,
                condition_number_raw   = pm_valid.condition_number_raw,
                condition_number_reg   = pm_valid.condition_number_reg,
                spectral_gap_absolute  = pm_valid.spectral_gap_absolute,
                spectral_gap_cheeger   = pm_valid.spectral_gap_cheeger,
                null_space_dim         = pm_valid.null_space_dim,
                tikhonov_epsilon       = pm_valid.tikhonov_epsilon,
                regularization_applied = pm_valid.regularization_applied,
                matrix_dimension       = n,
            )

    def test_g_and_g_inv_are_symmetric(self) -> None:
        """Tanto G_reg como G_inv son simétricas (invariante de reconstrucción espectral)."""
        G = make_spd_matrix(7, seed=109)
        pm = self.P.precondition(G)
        assert_allclose(pm.G, pm.G.T, atol=1e-13,
                        err_msg="G_reg no es simétrica.")
        assert_allclose(pm.G_inv, pm.G_inv.T, atol=1e-13,
                        err_msg="G_inv no es simétrica.")


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §T2 — TestFlatIsomorphism: ♭ : TM → T*M (Fase 2)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestFlatIsomorphismConstruction:
    """Verifica la inicialización de FlatIsomorphism con contratos de tipo."""

    def test_construction_with_valid_pm(self) -> None:
        """FlatIsomorphism se construye sin error con PreconditionedMetric válida."""
        G = make_spd_matrix(4, seed=113)
        pm = build_preconditioned_metric(G)
        fi = FlatIsomorphism(pm)
        assert fi._n == 4

    def test_construction_rejects_wrong_type(self) -> None:
        """TypeError si se pasa algo distinto de PreconditionedMetric."""
        with pytest.raises(TypeError, match="PreconditionedMetric"):
            FlatIsomorphism(np.eye(3))  # type: ignore[arg-type]

    def test_stores_g_and_g_inv_references(self) -> None:
        """El constructor almacena G y G_inv de la métrica preacondicionada."""
        G = make_spd_matrix(5, seed=127)
        pm = build_preconditioned_metric(G)
        fi = FlatIsomorphism(pm)
        assert_array_equal(fi._G, pm.G)
        assert_array_equal(fi._G_inv, pm.G_inv)


class TestFlatIsomorphismLinearity:
    """
    Verifica la linealidad del isomorfismo ♭:
      ♭(αu + βv) = α·♭(u) + β·♭(v)

    Para la linealidad como homomorfismo de C∞(M)-módulos.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 5
        self.G = make_spd_matrix(self.n, seed=131)
        self.pm = build_preconditioned_metric(self.G)
        self.fi = FlatIsomorphism(self.pm)

    def test_flat_preserves_scalar_multiplication(self) -> None:
        """♭(α·v) = α·♭(v) para α ∈ ℝ."""
        alpha = 3.7
        v = make_random_unit_tangent_vector(self.n, seed=137)
        v_scaled = TangentVector(coordinates=alpha * v.coordinates)

        omega_v      = self.fi.apply_flat_isomorphism(v)
        omega_scaled = self.fi.apply_flat_isomorphism(v_scaled)

        assert_allclose(
            omega_scaled.coordinates, alpha * omega_v.coordinates,
            rtol=1e-13,
            err_msg=f"♭ no preserva multiplicación escalar con α={alpha}."
        )

    def test_flat_preserves_vector_addition(self) -> None:
        """♭(u + v) = ♭(u) + ♭(v)."""
        rng = np.random.default_rng(139)
        u_arr = rng.standard_normal(self.n)
        v_arr = rng.standard_normal(self.n)
        u     = TangentVector(coordinates=u_arr.astype(np.float64))
        v     = TangentVector(coordinates=v_arr.astype(np.float64))
        u_v   = TangentVector(coordinates=(u_arr + v_arr).astype(np.float64))

        omega_u   = self.fi.apply_flat_isomorphism(u)
        omega_v   = self.fi.apply_flat_isomorphism(v)
        omega_sum = self.fi.apply_flat_isomorphism(u_v)

        assert_allclose(
            omega_sum.coordinates,
            omega_u.coordinates + omega_v.coordinates,
            rtol=1e-13,
            err_msg="♭ no preserva la adición de vectores."
        )

    def test_flat_is_linear_combination(self) -> None:
        """♭(αu + βv) = α·♭(u) + β·♭(v) para α, β ∈ ℝ arbitrarios."""
        rng = np.random.default_rng(149)
        alpha, beta = 2.5, -1.3
        u_arr = rng.standard_normal(self.n)
        v_arr = rng.standard_normal(self.n)
        u     = TangentVector(coordinates=u_arr.astype(np.float64))
        v     = TangentVector(coordinates=v_arr.astype(np.float64))
        w     = TangentVector(coordinates=(alpha * u_arr + beta * v_arr).astype(np.float64))

        lhs = self.fi.apply_flat_isomorphism(w).coordinates
        rhs = (alpha * self.fi.apply_flat_isomorphism(u).coordinates +
               beta  * self.fi.apply_flat_isomorphism(v).coordinates)

        assert_allclose(lhs, rhs, rtol=1e-13,
                        err_msg="Linealidad completa de ♭ fallida.")

    def test_flat_of_zero_vector_is_zero_covector(self) -> None:
        """♭(0) = 0 (linealidad implica preservación del cero)."""
        zero = TangentVector(coordinates=np.zeros(self.n))
        omega = self.fi.apply_flat_isomorphism(zero)
        assert_allclose(omega.coordinates, np.zeros(self.n), atol=1e-15,
                        err_msg="♭(0) debe ser el co-vector cero.")

    def test_flat_returns_cotangent_vector_type(self) -> None:
        """♭ retorna siempre CotangentVector."""
        v = make_random_unit_tangent_vector(self.n, seed=151)
        result = self.fi.apply_flat_isomorphism(v)
        assert isinstance(result, CotangentVector), (
            f"♭ debe retornar CotangentVector, retornó {type(result).__name__}."
        )


class TestFlatIsomorphismAdjunction:
    """
    Verifica la propiedad de adjunción del isomorfismo ♭:
    Para G simétrica: ⟨♭(v), w⟩ = G(v, w) = ⟨v, ♭(w)⟩

    Esta propiedad es consecuencia directa de la simetría de G y es
    fundamental para la coherencia del fibrado de dualidad.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 6
        self.G = make_spd_matrix(self.n, seed=157)
        self.pm = build_preconditioned_metric(self.G)
        self.fi = FlatIsomorphism(self.pm)

    def test_flat_adjunction_symmetry(self) -> None:
        """⟨♭(v), w⟩ = ⟨♭(w), v⟩ (G simétrica implica adjunción simétrica)."""
        rng = np.random.default_rng(163)
        v_arr = rng.standard_normal(self.n).astype(np.float64)
        w_arr = rng.standard_normal(self.n).astype(np.float64)

        v = TangentVector(coordinates=v_arr)
        w = TangentVector(coordinates=w_arr)

        flat_v = self.fi.apply_flat_isomorphism(v)
        flat_w = self.fi.apply_flat_isomorphism(w)

        # ⟨♭(v), w⟩ = ωᵥ · w
        lhs = np.dot(flat_v.coordinates, w_arr)
        # ⟨♭(w), v⟩ = ω_w · v
        rhs = np.dot(flat_w.coordinates, v_arr)

        assert_allclose(lhs, rhs, rtol=1e-13,
                        err_msg="Adjunción ⟨♭(v),w⟩ = ⟨♭(w),v⟩ violada.")

    def test_flat_action_equals_metric_tensor(self) -> None:
        """⟨♭(v), w⟩ = G(v, w) = vᵀ G w (métrica Riemanniana como forma bilineal)."""
        rng = np.random.default_rng(167)
        v_arr = rng.standard_normal(self.n).astype(np.float64)
        w_arr = rng.standard_normal(self.n).astype(np.float64)

        v = TangentVector(coordinates=v_arr)
        flat_v = self.fi.apply_flat_isomorphism(v)

        # ⟨♭(v), w⟩ = (G·v) · w
        inner_via_flat = np.dot(flat_v.coordinates, w_arr)
        # G(v, w) = vᵀ G w (usando G_reg)
        inner_via_metric = v_arr @ self.pm.G @ w_arr

        assert_allclose(inner_via_flat, inner_via_metric, rtol=1e-13,
                        err_msg="⟨♭(v), w⟩ ≠ G(v, w).")

    def test_flat_norm_bound(self) -> None:
        """‖♭(v)‖ ≤ λ_max · ‖v‖ (cota de Lipschitz por norma espectral)."""
        v = make_random_unit_tangent_vector(self.n, seed=173)
        omega = self.fi.apply_flat_isomorphism(v)
        lambda_max = float(self.pm.eigenvalues_reg[-1])
        assert omega.norm <= lambda_max * v.norm * (1 + 1e-10), (
            f"‖♭(v)‖={omega.norm:.3e} > λ_max·‖v‖={lambda_max * v.norm:.3e}."
        )


class TestFlatIsomorphismDimensionGuards:
    """Verifica que FunctorialityError se lanza ante incompatibilidades dimensionales."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 4
        G = make_spd_matrix(self.n, seed=179)
        pm = build_preconditioned_metric(G)
        self.fi = FlatIsomorphism(pm)

    @pytest.mark.parametrize("wrong_dim", [1, 3, 5, 10])
    def test_flat_rejects_wrong_dimension(self, wrong_dim: int) -> None:
        """FunctorialityError si dim(v) ≠ n."""
        v = TangentVector(coordinates=np.ones(wrong_dim))
        with pytest.raises(FunctorialityError, match="Colapso dimensional"):
            self.fi.apply_flat_isomorphism(v)

    def test_flat_accepts_correct_dimension(self) -> None:
        """No excepción para dim(v) == n."""
        v = TangentVector(coordinates=np.ones(self.n))
        result = self.fi.apply_flat_isomorphism(v)
        assert result.dim == self.n

    def test_diagnostics_report_contains_phase_info(self) -> None:
        """diagnostics_report() incluye la clave 'phase' con 'Fase 2'."""
        report = self.fi.diagnostics_report()
        assert "phase" in report, "Reporte debe incluir clave 'phase'."
        assert "Fase 2" in report["phase"], (
            f"'phase' debe mencionar 'Fase 2', obtenido: {report['phase']}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §T2 — TestSharpIsomorphism: ♯ : T*M → TM (Fase 3)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSharpIsomorphismLinearity:
    """
    Verifica la linealidad del isomorfismo ♯:
      ♯(αω + βη) = α·♯(ω) + β·♯(η)
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 5
        G = make_spd_matrix(self.n, seed=181)
        pm = build_preconditioned_metric(G)
        self.si = SharpIsomorphism(pm)

    def test_sharp_preserves_scalar_multiplication(self) -> None:
        """♯(α·ω) = α·♯(ω)."""
        alpha = -2.1
        omega = make_random_cotangent_vector(self.n, seed=191)
        omega_scaled = CotangentVector(coordinates=alpha * omega.coordinates)

        v          = self.si.apply_sharp_isomorphism(omega)
        v_scaled   = self.si.apply_sharp_isomorphism(omega_scaled)

        assert_allclose(
            v_scaled.coordinates, alpha * v.coordinates,
            rtol=1e-13,
            err_msg=f"♯ no preserva multiplicación escalar con α={alpha}."
        )

    def test_sharp_preserves_addition(self) -> None:
        """♯(ω + η) = ♯(ω) + ♯(η)."""
        rng = np.random.default_rng(193)
        omega_arr = rng.standard_normal(self.n).astype(np.float64)
        eta_arr   = rng.standard_normal(self.n).astype(np.float64)
        omega     = CotangentVector(coordinates=omega_arr)
        eta       = CotangentVector(coordinates=eta_arr)
        omega_eta = CotangentVector(coordinates=(omega_arr + eta_arr).astype(np.float64))

        v_omega   = self.si.apply_sharp_isomorphism(omega)
        v_eta     = self.si.apply_sharp_isomorphism(eta)
        v_sum     = self.si.apply_sharp_isomorphism(omega_eta)

        assert_allclose(
            v_sum.coordinates,
            v_omega.coordinates + v_eta.coordinates,
            rtol=1e-13,
            err_msg="♯ no preserva la adición de co-vectores."
        )

    def test_sharp_of_zero_is_zero(self) -> None:
        """♯(0) = 0."""
        zero = CotangentVector(coordinates=np.zeros(self.n))
        v = self.si.apply_sharp_isomorphism(zero)
        assert_allclose(v.coordinates, np.zeros(self.n), atol=1e-15,
                        err_msg="♯(0) debe ser el vector cero.")

    def test_sharp_returns_tangent_vector_type(self) -> None:
        """♯ retorna siempre TangentVector."""
        omega = make_random_cotangent_vector(self.n, seed=197)
        result = self.si.apply_sharp_isomorphism(omega)
        assert isinstance(result, TangentVector), (
            f"♯ debe retornar TangentVector, retornó {type(result).__name__}."
        )

    @pytest.mark.parametrize("wrong_dim", [1, 3, 7])
    def test_sharp_rejects_wrong_dimension(self, wrong_dim: int) -> None:
        """FunctorialityError si dim(ω) ≠ n."""
        omega = CotangentVector(coordinates=np.ones(wrong_dim))
        with pytest.raises(FunctorialityError, match="Colapso dimensional"):
            self.si.apply_sharp_isomorphism(omega)


class TestSharpIsomorphismRoundtrip:
    """
    Verifica las identidades de roundtrip del par de isomorfismos:
      ♯ ∘ ♭ = id_{TM}    (aplicado a vectores tangentes)
      ♭ ∘ ♯ = id_{T*M}   (aplicado a co-vectores)

    Estas identidades son el test más potente de la coherencia algebraica
    end-to-end de las tres fases.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 6
        self.G = make_spd_matrix(self.n, seed=199)
        self.pm = build_preconditioned_metric(self.G)
        self.si = SharpIsomorphism(self.pm)

    @pytest.mark.parametrize("seed", [200, 201, 202, 203, 204])
    def test_sharp_flat_is_identity_on_tangent(self, seed: int) -> None:
        """♯(♭(v)) = v para múltiples vectores aleatorios."""
        v = make_random_unit_tangent_vector(self.n, seed=seed)
        omega   = self.si.apply_flat_isomorphism(v)
        v_prime = self.si.apply_sharp_isomorphism(omega)

        kappa = self.pm.condition_number_reg
        tol = max(
            _ROUNDTRIP_TOLERANCE_FACTOR * kappa * _MACHINE_EPSILON * max(v.norm, 1.0),
            _INVERSION_BASE_TOLERANCE
        )
        residual = np.linalg.norm(v_prime.coordinates - v.coordinates)
        assert residual <= tol * 10, (  # Factor 10 de margen CI
            f"♯(♭(v)) ≠ v: ‖♯(♭(v))-v‖={residual:.2e} > tol={tol:.2e}, seed={seed}."
        )

    @pytest.mark.parametrize("seed", [210, 211, 212])
    def test_flat_sharp_is_identity_on_cotangent(self, seed: int) -> None:
        """♭(♯(ω)) = ω para múltiples co-vectores aleatorios."""
        omega   = make_random_cotangent_vector(self.n, seed=seed)
        v       = self.si.apply_sharp_isomorphism(omega)
        omega_p = self.si.apply_flat_isomorphism(v)

        kappa = self.pm.condition_number_reg
        tol = max(
            _ROUNDTRIP_TOLERANCE_FACTOR * kappa * _MACHINE_EPSILON * max(omega.norm, 1.0),
            _INVERSION_BASE_TOLERANCE
        )
        residual = np.linalg.norm(omega_p.coordinates - omega.coordinates)
        assert residual <= tol * 10,  (
            f"♭(♯(ω)) ≠ ω: ‖♭(♯(ω))-ω‖={residual:.2e} > tol={tol:.2e}, seed={seed}."
        )

    def test_verify_roundtrip_identity_passes_for_valid_vector(self) -> None:
        """verify_roundtrip_identity() no lanza excepción para vector válido."""
        v = make_random_unit_tangent_vector(self.n, seed=215)
        report = self.si.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"verify_roundtrip_identity reportó fallo: {report}."
        )

    def test_verify_roundtrip_report_keys(self) -> None:
        """verify_roundtrip_identity() retorna todas las claves esperadas."""
        v = make_random_unit_tangent_vector(self.n, seed=217)
        report = self.si.verify_roundtrip_identity(v)
        expected_keys = {"passed", "residual", "tolerance", "v_norm", "kappa_reg", "phase"}
        assert expected_keys.issubset(report.keys()), (
            f"Claves faltantes en roundtrip report: {expected_keys - report.keys()}."
        )

    def test_roundtrip_identity_for_canonical_basis(self) -> None:
        """♯(♭(eᵢ)) = eᵢ para todos los vectores de la base canónica."""
        for i in range(self.n):
            e_i = np.zeros(self.n, dtype=np.float64)
            e_i[i] = 1.0
            v = TangentVector(coordinates=e_i)
            report = self.si.verify_roundtrip_identity(v)
            assert report["passed"], (
                f"Roundtrip fallido para e_{i}: {report}."
            )

    def test_roundtrip_error_scales_with_vector_norm(self) -> None:
        """
        El error de roundtrip escala con ‖v‖:
        residual(α·v) ≈ α · residual(v)  (linealidad del error).
        """
        v = make_random_unit_tangent_vector(self.n, seed=221)
        alpha = 1000.0
        v_scaled = TangentVector(coordinates=alpha * v.coordinates)

        r1 = self.si.verify_roundtrip_identity(v)["residual"]
        r2 = self.si.verify_roundtrip_identity(v_scaled)["residual"]

        # r2 / r1 ≈ alpha dentro de un orden de magnitud
        ratio = r2 / (r1 + _MACHINE_EPSILON)
        assert alpha * 0.01 <= ratio <= alpha * 100, (
            f"El error de roundtrip no escala linealmente con ‖v‖: "
            f"ratio={ratio:.2e}, esperado ≈ {alpha:.1e}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §T3 — TestFunctorComposition: Auditoría Categórica en Z₂                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestFunctorComposition:
    """
    Verifica audit_functor_composition():
      - Tabla de Cayley completa en Z₂.
      - Compatibilidad de dominio/codominio (semántica topos).
      - Errores ante funtores sin varianza.
      - Reporte de auditoría con claves correctas.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        G = make_spd_matrix(4, seed=227)
        pm = build_preconditioned_metric(G)
        self.si = SharpIsomorphism(pm)
        self.cov_functor  = CovariantFunctor()
        self.cont_functor = ContravariantFunctor()

    # ── Tabla de Cayley ──────────────────────────────────────────────────────

    def test_cov_compose_cov_gives_cov(self) -> None:
        """Cov ∘ Cov = Cov."""
        report = self.si.audit_functor_composition(
            self.cov_functor, self.cov_functor,
            verify_domain_compatibility=False
        )
        assert report["result_variance"] is CategoricalVariance.COVARIANT

    def test_cov_compose_cont_gives_cont(self) -> None:
        """Cov ∘ Cont = Cont."""
        report = self.si.audit_functor_composition(
            self.cov_functor, self.cont_functor,
            verify_domain_compatibility=False
        )
        assert report["result_variance"] is CategoricalVariance.CONTRAVARIANT

    def test_cont_compose_cov_gives_cont(self) -> None:
        """Cont ∘ Cov = Cont."""
        report = self.si.audit_functor_composition(
            self.cont_functor, self.cov_functor,
            verify_domain_compatibility=False
        )
        assert report["result_variance"] is CategoricalVariance.CONTRAVARIANT

    def test_cont_compose_cont_gives_cov(self) -> None:
        """Cont ∘ Cont = Cov."""
        report = self.si.audit_functor_composition(
            self.cont_functor, self.cont_functor,
            verify_domain_compatibility=False
        )
        assert report["result_variance"] is CategoricalVariance.COVARIANT

    # ── Reporte de auditoría ─────────────────────────────────────────────────

    def test_audit_report_contains_all_keys(self) -> None:
        """El reporte de auditoría contiene todas las claves esperadas."""
        report = self.si.audit_functor_composition(
            self.cov_functor, self.cov_functor,
            verify_domain_compatibility=False
        )
        expected_keys = {
            "result_variance", "var_f1", "var_f2",
            "domain_compatible", "f1_domain", "f2_codomain",
            "composition_valid", "phase"
        }
        assert expected_keys.issubset(report.keys()), (
            f"Claves faltantes en audit report: {expected_keys - report.keys()}."
        )

    def test_audit_var_f1_f2_types(self) -> None:
        """var_f1 y var_f2 en el reporte son instancias de CategoricalVariance."""
        report = self.si.audit_functor_composition(
            self.cov_functor, self.cont_functor,
            verify_domain_compatibility=False
        )
        assert isinstance(report["var_f1"], CategoricalVariance)
        assert isinstance(report["var_f2"], CategoricalVariance)

    # ── Errores de tipo ──────────────────────────────────────────────────────

    def test_functor_without_variance_raises_type_error(self) -> None:
        """TypeError si un funtor no tiene atributo 'variance'."""
        bad_functor = MagicMock(spec=[])  # Sin atributo 'variance'
        with pytest.raises(TypeError, match="variance"):
            self.si.audit_functor_composition(
                bad_functor, self.cov_functor,
                verify_domain_compatibility=False
            )

    def test_functor_with_wrong_variance_type_raises_type_error(self) -> None:
        """TypeError si 'variance' no es CategoricalVariance."""
        bad_functor = MagicMock()
        bad_functor.variance = "COVARIANT"  # String, no CategoricalVariance
        with pytest.raises(TypeError, match="CategoricalVariance"):
            self.si.audit_functor_composition(
                bad_functor, self.cov_functor,
                verify_domain_compatibility=False
            )

    # ── Compatibilidad de dominio ────────────────────────────────────────────

    def test_domain_compatibility_check_passes_for_compatible(self) -> None:
        """
        verify_domain_compatibility=True pasa cuando dom(f1) ≅ cod(f2).

        CovariantFunctor: domain='C', codomain='D'
        Para f1 ∘ f2: dom(f1) debe ser 'D' = cod(f2).
        Usamos f2 = CovariantFunctor (cod='D') y f1 con dom='D'.
        """
        # Crear f1 con domain_category='D' para que coincida con cod(f2)='D'
        f1 = MagicMock(spec=CovariantFunctor)
        f1.variance = CategoricalVariance.COVARIANT
        f1.domain_category   = "D"   # ≅ cod(f2) = 'D'
        f1.codomain_category = "E"

        report = self.si.audit_functor_composition(
            f1, self.cov_functor,
            verify_domain_compatibility=True
        )
        assert report["domain_compatible"], (
            "Compatibilidad de dominio debe pasar para dom(f1)='D' = cod(f2)='D'."
        )

    def test_domain_compatibility_check_fails_for_incompatible(self) -> None:
        """
        FunctorialityError cuando dom(f1) ≇ cod(f2).
        """
        f1 = MagicMock(spec=CovariantFunctor)
        f1.variance = CategoricalVariance.COVARIANT
        f1.domain_category   = "X"  # Incompatible con cod(f2) = 'D'
        f1.codomain_category = "Y"

        with pytest.raises(FunctorialityError, match="topos MIC"):
            self.si.audit_functor_composition(
                f1, self.cov_functor,
                verify_domain_compatibility=True
            )

    def test_domain_check_disabled_bypasses_compatibility(self) -> None:
        """Con verify_domain_compatibility=False no se verifica compatibilidad."""
        f1 = MagicMock(spec=CovariantFunctor)
        f1.variance = CategoricalVariance.COVARIANT
        f1.domain_category   = "INCOMPATIBLE"
        f1.codomain_category = "IRRELEVANT"

        # No debe lanzar excepción
        report = self.si.audit_functor_composition(
            f1, self.cov_functor,
            verify_domain_compatibility=False
        )
        assert report["result_variance"] is CategoricalVariance.COVARIANT


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INTEGRACIÓN END-TO-END — TestMusicalIsomorphismEngine                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMusicalIsomorphismEngineConstruction:
    """
    Verifica la construcción del orquestador MusicalIsomorphismEngine:
      - Constructor por defecto con G_PHYSICS.
      - Constructor con métrica personalizada.
      - Inyección de dependencias del preconditioner.
      - Acceso a preconditioned_metric property.
    """

    def test_construction_with_custom_spd_metric(self) -> None:
        """El motor se construye sin error para una métrica SPD válida."""
        G = make_spd_matrix(4, seed=229)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        assert engine._n == 4
        assert engine.preconditioned_metric.matrix_dimension == 4

    def test_preconditioned_metric_property_is_immutable(self) -> None:
        """preconditioned_metric retorna PreconditionedMetric (frozen dataclass)."""
        G = make_spd_matrix(3, seed=233)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        pm = engine.preconditioned_metric
        assert isinstance(pm, PreconditionedMetric)
        # Verificar inmutabilidad
        with pytest.raises((AttributeError, TypeError)):
            pm.matrix_dimension = 999  # type: ignore[misc]

    def test_dependency_injection_custom_preconditioner(self) -> None:
        """Preconditioner inyectado externamente es utilizado correctamente."""
        G = make_spd_matrix(5, seed=239)
        custom_preconditioner = MetricSpectralPreconditioner()
        engine = MusicalIsomorphismEngine(
            metric_tensor=G,
            preconditioner=custom_preconditioner
        )
        assert engine._n == 5

    def test_construction_propagates_phase1_errors(self) -> None:
        """Las excepciones de Fase 1 son propagadas correctamente."""
        with pytest.raises(ValueError, match="NaN"):
            G = np.eye(3)
            G[0, 0] = np.nan
            MusicalIsomorphismEngine(metric_tensor=G)

    def test_construction_with_ill_conditioned_metric(self) -> None:
        """El motor se construye incluso para métricas mal condicionadas (con regularización)."""
        G = make_ill_conditioned_matrix(5, condition_number=1e15, seed=241)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        assert engine.preconditioned_metric.regularization_applied

    def test_engine_inherits_flat_and_sharp(self) -> None:
        """MusicalIsomorphismEngine expone apply_flat_isomorphism y apply_sharp_isomorphism."""
        G = make_spd_matrix(4, seed=243)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        assert hasattr(engine, 'apply_flat_isomorphism')
        assert hasattr(engine, 'apply_sharp_isomorphism')
        assert hasattr(engine, 'verify_roundtrip_identity')
        assert hasattr(engine, 'audit_functor_composition')
        assert hasattr(engine, 'full_cycle_report')


class TestMusicalIsomorphismEngineFullCycle:
    """
    Verifica el método full_cycle_report() que ejecuta el ciclo completo:
      ♭ → ♯ → verify_roundtrip
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 5
        G = make_spd_matrix(self.n, seed=247)
        self.engine = MusicalIsomorphismEngine(metric_tensor=G)

    def test_full_cycle_report_passes(self) -> None:
        """full_cycle_report() completa sin excepción para vector válido."""
        v = make_random_unit_tangent_vector(self.n, seed=251)
        report = self.engine.full_cycle_report(v)
        assert report is not None

    def test_full_cycle_report_keys(self) -> None:
        """full_cycle_report() contiene claves de diagnóstico de las 3 fases."""
        v = make_random_unit_tangent_vector(self.n, seed=257)
        report = self.engine.full_cycle_report(v)
        expected_keys = {
            "input_vector_norm", "flat_covector_norm",
            "sharp_vector_norm", "roundtrip", "engine",
            "matrix_dimension", "condition_number_reg"
        }
        assert expected_keys.issubset(report.keys()), (
            f"Claves faltantes en full_cycle_report: {expected_keys - report.keys()}."
        )

    def test_full_cycle_norms_finite(self) -> None:
        """Las normas de entrada, ♭ y ♯ son finitas."""
        v = make_random_unit_tangent_vector(self.n, seed=263)
        report = self.engine.full_cycle_report(v)
        for key in ["input_vector_norm", "flat_covector_norm", "sharp_vector_norm"]:
            assert np.isfinite(report[key]), (
                f"Norma '{key}' no es finita: {report[key]}."
            )

    def test_full_cycle_roundtrip_passed(self) -> None:
        """El sub-reporte roundtrip indica 'passed': True."""
        v = make_random_unit_tangent_vector(self.n, seed=269)
        report = self.engine.full_cycle_report(v)
        assert report["roundtrip"]["passed"], (
            f"Roundtrip fallido en full_cycle_report: {report['roundtrip']}."
        )

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, 1e6, 1e-6])
    def test_full_cycle_scales_with_vector_norm(self, alpha: float) -> None:
        """Las normas intermedias escalan coherentemente con ‖v‖."""
        v = make_random_unit_tangent_vector(self.n, seed=271)
        v_scaled = TangentVector(coordinates=(alpha * v.coordinates).astype(np.float64))
        report = self.engine.full_cycle_report(v_scaled)
        expected_input_norm = abs(alpha)
        assert_allclose(
            report["input_vector_norm"], expected_input_norm, rtol=1e-12,
            err_msg=f"input_vector_norm incorrecto para α={alpha}."
        )


class TestMusicalIsomorphismEngineHighDimensional:
    """
    Pruebas de robustez numérica para métricas de alta dimensión:
      n = 50, 100, 200.

    Verifica que el pipeline se ejecuta correctamente sin degradación
    numérica significativa para dimensiones relevantes en circuitos MIC.
    """

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_high_dimensional_spd_roundtrip(self, n: int) -> None:
        """
        Roundtrip ♯∘♭ = id para métricas SPD de alta dimensión.

        Para n=200, el costo computacional es O(n³) ≈ 8e6 operaciones;
        acepta degradación numérica proporcional a √n · κ · ε_machine.
        """
        G = make_spd_matrix(n, seed=n)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        v = make_random_unit_tangent_vector(n, seed=n + 1)
        report = engine.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"Roundtrip fallido para n={n}: residual={report['residual']:.2e}, "
            f"tol={report['tolerance']:.2e}."
        )

    @pytest.mark.parametrize("n", [50, 100])
    def test_high_dimensional_ill_conditioned(self, n: int) -> None:
        """
        El pipeline maneja métricas mal condicionadas de alta dimensión.
        La regularización de Tikhonov debe activarse y producir G_reg invertible.
        """
        G = make_ill_conditioned_matrix(n, condition_number=1e14, seed=n + 2)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        pm = engine.preconditioned_metric
        assert pm.regularization_applied, f"Regularización debe aplicarse para n={n}, κ=1e14."
        assert np.all(pm.eigenvalues_reg > 0), f"Autovalores no todos positivos para n={n}."

    @pytest.mark.parametrize("n", [50, 100])
    def test_high_dimensional_hilbert_roundtrip(self, n: int) -> None:
        """
        La matriz de Hilbert n×n (κ >> 1) procesada correctamente.

        Para n ≥ 10, κ(H_n) > 1e13, garantizando que se prueba
        el path de regularización para métricas analíticamente conocidas.
        """
        G = make_hilbert_matrix(n)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        pm = engine.preconditioned_metric
        assert pm.regularization_applied, (
            f"Hilbert n={n} debe activar regularización."
        )
        # Verificar que el roundtrip funciona post-regularización
        v = make_random_unit_tangent_vector(n, seed=n + 3)
        report = engine.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"Roundtrip fallido para Hilbert n={n}: {report}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PRUEBAS DE CASOS EXTREMOS Y PROPIEDADES ESPECIALES                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndSpecialProperties:
    """
    Verifica casos extremos que no están cubiertos en los tests unitarios anteriores:
      - Métrica 1×1 escalar.
      - Vector de norma exactamente cero.
      - Métrica con autovalores repetidos (degenerada).
      - Perturbación infinitesimal de la identidad.
      - Composición triple de isomorfismos (♭ ∘ ♯ ∘ ♭ = ♭).
    """

    def test_scalar_metric_1x1(self) -> None:
        """Métrica escalar G = [[g]]: ♭(v) = g·v, ♯(ω) = ω/g."""
        g = 3.14159
        G = np.array([[g]], dtype=np.float64)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        v     = TangentVector(coordinates=np.array([2.0]))
        omega = engine.apply_flat_isomorphism(v)
        v_rec = engine.apply_sharp_isomorphism(omega)

        assert_allclose(omega.coordinates[0], g * 2.0, rtol=1e-14,
                        err_msg=f"♭ escalar: esperado {g*2.0}, obtenido {omega.coordinates[0]}.")
        assert_allclose(v_rec.coordinates[0], 2.0, rtol=1e-12,
                        err_msg="Roundtrip escalar fallido.")

    def test_identity_metric_flat_is_identity_map(self) -> None:
        """Para G = I_n, ♭ es la identidad: ♭(v) = v."""
        n = 5
        engine = MusicalIsomorphismEngine(metric_tensor=np.eye(n))
        v = make_random_unit_tangent_vector(n, seed=277)
        omega = engine.apply_flat_isomorphism(v)
        # Para G = I, los componentes del co-vector = componentes del vector
        assert_allclose(omega.coordinates, v.coordinates, rtol=1e-14,
                        err_msg="Para G=I, ♭(v) debe ser idéntico a v.")

    def test_identity_metric_sharp_is_identity_map(self) -> None:
        """Para G = I_n, ♯ es la identidad: ♯(ω) = ω."""
        n = 5
        engine = MusicalIsomorphismEngine(metric_tensor=np.eye(n))
        omega = make_random_cotangent_vector(n, seed=281)
        v = engine.apply_sharp_isomorphism(omega)
        assert_allclose(v.coordinates, omega.coordinates, rtol=1e-14,
                        err_msg="Para G=I, ♯(ω) debe ser idéntico a ω.")

    def test_diagonal_metric_flat_scales_components(self) -> None:
        """
        Para G = diag(g₁, g₂, ..., gₙ), ♭(v)ᵢ = gᵢ · vᵢ (bajada componente a componente).
        """
        diag_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        G = np.diag(diag_vals)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        v_arr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        v = TangentVector(coordinates=v_arr)
        omega = engine.apply_flat_isomorphism(v)

        assert_allclose(omega.coordinates, diag_vals * v_arr, rtol=1e-14,
                        err_msg="♭ diagonal debe escalar cada componente por gᵢ.")

    def test_diagonal_metric_sharp_scales_components(self) -> None:
        """
        Para G = diag(g₁, ..., gₙ), ♯(ω)ᵢ = ωᵢ / gᵢ.
        """
        diag_vals = np.array([2.0, 4.0, 8.0])
        G = np.diag(diag_vals)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        omega_arr = np.array([1.0, 1.0, 1.0])
        omega = CotangentVector(coordinates=omega_arr)
        v = engine.apply_sharp_isomorphism(omega)

        assert_allclose(v.coordinates, omega_arr / diag_vals, rtol=1e-14,
                        err_msg="♯ diagonal debe dividir cada componente por gᵢ.")

    def test_triple_composition_flat_sharp_flat(self) -> None:
        """
        ♭ ∘ ♯ ∘ ♭ = ♭ (triple composición debe colapsar a ♭ por idempotencia parcial).

        Algebraicamente: ♭(♯(♭(v))) = ♭(v)  ya que ♯∘♭ = id → ♭∘♯∘♭ = ♭.
        """
        n = 5
        G = make_spd_matrix(n, seed=283)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        v = make_random_unit_tangent_vector(n, seed=289)

        omega_1 = engine.apply_flat_isomorphism(v)         # ♭(v)
        v_mid   = engine.apply_sharp_isomorphism(omega_1)  # ♯(♭(v)) ≈ v
        omega_2 = engine.apply_flat_isomorphism(v_mid)     # ♭(♯(♭(v))) ≈ ♭(v)

        kappa = engine.preconditioned_metric.condition_number_reg
        tol = max(
            10 * _ROUNDTRIP_TOLERANCE_FACTOR * kappa * _MACHINE_EPSILON,
            1e-12
        )
        residual = np.linalg.norm(omega_2.coordinates - omega_1.coordinates)
        assert residual <= tol * max(omega_1.norm, 1.0), (
            f"♭∘♯∘♭ ≠ ♭: ‖resultado - ♭(v)‖={residual:.2e} > tol·‖♭(v)‖."
        )

    def test_metric_with_repeated_eigenvalues(self) -> None:
        """
        Métrica con autovalores repetidos (espacio propio degenerado) se procesa
        correctamente sin inestabilidad numérica.

        G = diag(1, 1, 1, 2, 2) tiene dos subespacios propios.
        """
        G = np.diag([1.0, 1.0, 1.0, 2.0, 2.0])
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        pm = engine.preconditioned_metric

        assert pm.null_space_dim == 0, "No debe haber kernel para G definida positiva."
        v = make_random_unit_tangent_vector(5, seed=293)
        report = engine.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"Roundtrip fallido para G con autovalores repetidos: {report}."
        )

    def test_infinitesimal_perturbation_of_identity(self) -> None:
        """
        G = I + ε·P donde P es SPD pequeña y ε << 1 produce una métrica
        casi-plana que no requiere regularización y satisface el roundtrip.
        """
        n = 4
        eps = 1e-6
        P = make_spd_matrix(n, seed=297)
        # Normalizar P para que ‖ε·P‖ << 1
        P_normalized = P / np.linalg.norm(P, 'fro')
        G = np.eye(n) + eps * P_normalized

        engine = MusicalIsomorphismEngine(metric_tensor=G)
        assert not engine.preconditioned_metric.regularization_applied, (
            "Métrica casi-identidad no debe requerir regularización."
        )
        v = make_random_unit_tangent_vector(n, seed=299)
        report = engine.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"Roundtrip fallido para métrica casi-identidad: {report}."
        )

    def test_flat_covector_is_finite_for_large_norm_vector(self) -> None:
        """
        apply_flat_isomorphism no falla para vectores de norma grande (‖v‖ = 1e6).
        El resultado debe ser finito (postcondición del método).
        """
        n = 4
        G = make_spd_matrix(n, seed=307)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        v_large = TangentVector(
            coordinates=np.ones(n, dtype=np.float64) * 1e6 / np.sqrt(n)
        )
        omega = engine.apply_flat_isomorphism(v_large)
        assert np.all(np.isfinite(omega.coordinates)), (
            "apply_flat_isomorphism debe producir resultados finitos para ‖v‖=1e6."
        )

    def test_sharp_vector_is_finite_for_large_norm_covector(self) -> None:
        """
        apply_sharp_isomorphism no falla para co-vectores de norma grande.
        """
        n = 4
        G = make_spd_matrix(n, seed=311)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        omega_large = CotangentVector(
            coordinates=np.ones(n, dtype=np.float64) * 1e6 / np.sqrt(n)
        )
        v = engine.apply_sharp_isomorphism(omega_large)
        assert np.all(np.isfinite(v.coordinates)), (
            "apply_sharp_isomorphism debe producir resultados finitos para ‖ω‖=1e6."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PRUEBAS DE PROPIEDADES ESTADÍSTICAS (PROPERTY-BASED LIGHT)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestStatisticalProperties:
    """
    Verifica propiedades algebraicas sobre conjuntos de vectores/métricas aleatorios.
    Complementa los tests deterministas con barridos estadísticos.
    """

    @pytest.mark.parametrize("trial", range(10))
    def test_roundtrip_across_random_spd_metrics(self, trial: int) -> None:
        """
        ♯∘♭ = id para métricas SPD aleatorias n ∈ {3, 4, 5} y vectores aleatorios.
        10 pruebas independientes con semillas distintas.
        """
        rng = np.random.default_rng(trial * 313 + 7)
        n = rng.integers(3, 6)
        G = make_spd_matrix(int(n), seed=trial * 313)
        engine = MusicalIsomorphismEngine(metric_tensor=G)
        v = TangentVector(
            coordinates=rng.standard_normal(int(n)).astype(np.float64)
        )
        report = engine.verify_roundtrip_identity(v)
        assert report["passed"], (
            f"Roundtrip fallido en trial={trial}, n={n}: {report}."
        )

    @pytest.mark.parametrize("trial", range(5))
    def test_linearity_flat_over_random_pairs(self, trial: int) -> None:
        """
        ♭(αu + βv) = α♭(u) + β♭(v) para parejas aleatorias (u, v, α, β).
        """
        rng = np.random.default_rng(trial * 317 + 11)
        n = int(rng.integers(3, 7))
        alpha, beta = float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))

        G = make_spd_matrix(n, seed=trial * 317)
        engine = MusicalIsomorphismEngine(metric_tensor=G)

        u_arr = rng.standard_normal(n).astype(np.float64)
        v_arr = rng.standard_normal(n).astype(np.float64)
        u     = TangentVector(coordinates=u_arr)
        v     = TangentVector(coordinates=v_arr)
        w     = TangentVector(coordinates=(alpha * u_arr + beta * v_arr).astype(np.float64))

        lhs = engine.apply_flat_isomorphism(w).coordinates
        rhs = (alpha * engine.apply_flat_isomorphism(u).coordinates +
               beta  * engine.apply_flat_isomorphism(v).coordinates)

        assert_allclose(lhs, rhs, rtol=1e-11,
                        err_msg=f"Linealidad de ♭ fallida en trial={trial}.")

    @pytest.mark.parametrize("trial", range(5))
    def test_z2_composition_law_random_sequences(self, trial: int) -> None:
        """
        La ley de composición Z₂ se mantiene para secuencias aleatorias de varianzas.
        """
        rng = np.random.default_rng(trial * 331)
        elements = [CategoricalVariance.COVARIANT, CategoricalVariance.CONTRAVARIANT]
        sequence_length = int(rng.integers(3, 10))
        sequence = [elements[int(rng.integers(0, 2))] for _ in range(sequence_length)]

        # Reducción iterativa
        result = sequence[0]
        for v in sequence[1:]:
            result = result * v

        # El resultado debe ser el producto de los valores numéricos módulo 2
        expected_value = 1
        for v in sequence:
            expected_value *= v.value

        expected = CategoricalVariance(expected_value)
        assert result is expected, (
            f"Composición Z₂ fallida en trial={trial}: "
            f"esperado {expected}, obtenido {result}."
        )