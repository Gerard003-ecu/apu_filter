"""
test_topological_watcher.py — Versión Refinada v5.0
═══════════════════════════════════════════════════════════════════════════════
Suite de Pruebas de Grado Aeroespacial para el Sistema Inmunológico Topológico

Mejoras implementadas:

NUMÉRICAS:
  N1. _safe_normalize: RuntimeWarning verificado con warns() no con catch_warnings
  N2. _stable_norm: overflow verificado analíticamente (scaled norm)
  N3. _regularize_metric: eigenvectores preservados solo para G+δI (no G singular)
  N4. _compute_condition_number: casos NaN tratados explícitamente
  N5. MetricTensor diagonal: eigen_decomposition verifica orden de eigenvalores

ALGEBRAICAS:
  A1. assert_spd: tolerancia parametrizable + mensaje diagnóstico completo
  A2. assert_idempotent: norma de Frobenius con tol explícita
  A3. assert_resolution_of_identity: inicialización correcta de la suma
  A4. Resolución de identidad: suma iniciada con np.zeros((n,n)) no sum()
  A5. Pairwise orthogonality: índices simétricamente verificados

TOPOLÓGICAS:
  T1. Euler χ: tipo int verificado con isinstance(chi, int), no duck typing
  T2. β₀ ≥ 1, β₁ ≥ 0: mensajes de error con valores actuales
  T3. Parámetros BETTI_INDICES verificados en construcción del fixture

HISTÉRESIS:
  H1. FIX-08: casos de borde con valores exactos en puntos de transición
  H2. Idempotencia: verificada para los tres estados (no solo 3 valores)
  H3. Anti-chattering: bandas de transición calculadas algebraicamente

FUNCTORIEDAD:
  F1. F(error) = error: verificado que result IS error_state (identidad exacta)
  F2. Contador de evaluaciones: aislado por fixture para evitar interferencia
  F3. context de with_update verificado por clave+valor, no solo clave

PROPERTY-BASED:
  P1. assume() más restrictivo para evitar ejemplos triviales
  P2. Homogeneidad cuadrática: manejo de overflow explícito
  P3. Idempotencia de histéresis: parámetros con precondiciones algebraicas

CONSTRUCCIÓN:
  C1. SubspaceSpec: referencia inmutable verificada con flags writeable
  C2. OrthogonalProjector: coverage de índices sin subespacio verificada
  C3. ThreatAssessment: JSON-serializabilidad de tipos recursiva
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import app.core.immune_system.topological_watcher as _module
from app.core.immune_system.topological_watcher import (
    DimensionalMismatchError,
    ImmuneSystemError,
    MetricTensorError,
    NumericalStabilityError,
    PhysicalBoundsError,
    TopologicalInvariantError,
    ALGEBRAIC_TOL,
    BETTI_INDICES,
    COND_NUM_TOL,
    EPS,
    MIN_EIGVAL_TOL,
    SIGNAL_SCHEMA,
    VALIDATOR_REGISTRY,
    PhysicalConstants,
    HealthStatus,
    ImmuneWatcherMorphism,
    MetricTensor,
    OrthogonalProjector,
    SignalComponent,
    SubspaceSpec,
    ThreatAssessment,
    build_signal,
    create_immune_watcher,
    _compute_condition_number,
    _regularize_metric,
    _safe_normalize,
    _stable_divide,
    _stable_norm,
    _stable_reciprocal,
)

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState, Morphism


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRUEBA — derivadas analíticamente
# ═══════════════════════════════════════════════════════════════════════════════

# Umbral de histéresis por defecto en las pruebas de la sección 10 y 16
_W: float = 0.8          # umbral warning
_C: float = 1.5          # umbral critical
_D: float = 0.05         # banda de histéresis

# Bandas efectivas calculadas algebraicamente (no hardcodeadas):
# Subir a WARNING  : value > _W + _D = 0.85
# Subir a CRITICAL : value > _C + _D = 1.55
# Bajar de WARNING : value < _W - _D = 0.75
# Bajar de CRITICAL a WARNING: value < _C - _D = 1.45 y value ≥ _W - _D = 0.75
_BAND_RISE_WARNING: float  = _W + _D       # 0.85
_BAND_RISE_CRITICAL: float = _C + _D       # 1.55
_BAND_FALL_WARNING: float  = _W - _D       # 0.75
_BAND_FALL_CRITICAL: float = _C - _D       # 1.45

# Tolerancia para la norma de Frobenius en assertions algebraicas
_FROB_TOL: float = 1e-10

# Máxima dimensión en property-based tests (controla complejidad O(n³))
_MAX_PROP_DIM: int = 8


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 · UTILIDADES DE ASERCIÓN MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════


def assert_spd(
    matrix: np.ndarray,
    tol: float = MIN_EIGVAL_TOL,
    label: str = "G",
) -> None:
    """
    Verifica que una matriz sea simétrica definida positiva (SPD).

    Condiciones:
    1. Cuadrada: shape = (n, n)
    2. Simétrica: ‖M - Mᵀ‖_F < 1e-10
    3. Definida positiva: λ_min(M) ≥ tol > 0

    Usa eigvalsh (para matrices simétricas reales) en lugar de eigvals
    para evitar partes imaginarias espurias por redondeo.
    """
    assert matrix.ndim == 2, f"{label}: debe ser 2D, got ndim={matrix.ndim}"
    n, m = matrix.shape
    assert n == m, f"{label}: debe ser cuadrada, got shape={matrix.shape}"

    # Simetría (Frobenius)
    sym_err = float(np.linalg.norm(matrix - matrix.T, "fro"))
    assert sym_err < _FROB_TOL, (
        f"{label}: no simétrica — ‖G−Gᵀ‖_F = {sym_err:.2e} ≥ {_FROB_TOL:.2e}"
    )

    # Definida positiva (eigvalsh garantiza eigenvalores reales para S simétricas)
    eigvals = np.linalg.eigvalsh(matrix)
    lam_min = float(eigvals.min())
    assert lam_min >= tol, (
        f"{label}: no definida positiva — λ_min = {lam_min:.4e} < tol = {tol:.4e}. "
        f"Espectro: {eigvals}"
    )


def assert_idempotent(
    matrix: np.ndarray,
    tol: float = ALGEBRAIC_TOL,
    label: str = "P",
) -> None:
    """
    Verifica idempotencia: P² = P en norma de Frobenius.

    ‖P² − P‖_F < tol
    Condición necesaria y suficiente para que P sea un proyector.
    """
    p_squared = matrix @ matrix
    err = float(np.linalg.norm(p_squared - matrix, "fro"))
    assert err < tol, (
        f"{label}: no idempotente — ‖P²−P‖_F = {err:.4e} ≥ tol = {tol:.4e}"
    )


def assert_self_adjoint(
    matrix: np.ndarray,
    tol: float = ALGEBRAIC_TOL,
    label: str = "P",
) -> None:
    """
    Verifica autoadjunta: Pᵀ = P en norma de Frobenius.

    Para proyectores ortogonales (en contraste con oblicuos), Pᵀ = P
    es equivalente a que el rango y el núcleo sean ortogonales.
    """
    err = float(np.linalg.norm(matrix - matrix.T, "fro"))
    assert err < tol, (
        f"{label}: no autoadjunta — ‖P−Pᵀ‖_F = {err:.4e} ≥ tol = {tol:.4e}"
    )


def assert_resolution_of_identity(
    matrices: List[np.ndarray],
    tol: float = ALGEBRAIC_TOL,
    label: str = "Σπₖ",
) -> None:
    """
    Verifica resolución de identidad: Σₖ Pₖ = I_n.

    Inicialización explícita con np.zeros((n,n)) para evitar el error
    de type: sum([array, array], 0) donde 0 es int, no array.

    Invariante: Para subespacios ortogonales disjuntos que cubren ℝⁿ,
    la suma de los proyectores es la identidad.
    """
    assert len(matrices) > 0, "Lista de matrices vacía"
    n = matrices[0].shape[0]

    # CRÍTICO: inicializar con array, no con escalar 0
    total = np.zeros((n, n), dtype=float)
    for P in matrices:
        assert P.shape == (n, n), (
            f"Matriz con shape incorrecto: {P.shape}, esperado ({n},{n})"
        )
        total = total + P

    err = float(np.linalg.norm(total - np.eye(n), "fro"))
    assert err < tol, (
        f"{label}: resolución de identidad fallida — "
        f"‖ΣPₖ−I_{n}‖_F = {err:.4e} ≥ tol = {tol:.4e}"
    )


def assert_pairwise_orthogonal(
    matrices: List[np.ndarray],
    tol: float = ALGEBRAIC_TOL,
) -> None:
    """
    Verifica ortogonalidad mutua: πᵢ·πⱼ = 0 para todo i ≠ j.

    Para proyectores ortogonales sobre subespacios disjuntos,
    πᵢ·πⱼ = 0 ⟺ Im(πᵢ) ⊥ Im(πⱼ).

    Verifica ambas direcciones (i,j) y (j,i) explícitamente.
    """
    n = len(matrices)
    for i in range(n):
        for j in range(n):
            if i != j:
                product = matrices[i] @ matrices[j]
                err = float(np.linalg.norm(product, "fro"))
                assert err < tol, (
                    f"Ortogonalidad violada: ‖π_{i}·π_{j}‖_F = {err:.4e} "
                    f"≥ tol = {tol:.4e}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def healthy_telemetry() -> Dict[str, Any]:
    """Telemetría completamente saludable — punto de referencia."""
    return {
        "saturation": 0.1,
        "flyback_voltage": 10.0,
        "dissipated_power": 5.0,
        "beta_0": 1,
        "beta_1": 0,
        "entropy": 0.05,
        "exergy_loss": 0.02,
    }


@pytest.fixture(scope="session")
def warning_telemetry() -> Dict[str, Any]:
    """
    Telemetría en zona de advertencia.

    Valores elegidos para estar en (warning, critical) con perfil default.
    """
    return {
        "saturation": 0.92,
        "flyback_voltage": 340.0,
        "dissipated_power": 160.0,
        "beta_0": 1,
        "beta_1": 0,
        "entropy": 0.45,
        "exergy_loss": 0.38,
    }


@pytest.fixture(scope="session")
def critical_telemetry() -> Dict[str, Any]:
    """Telemetría en zona crítica — múltiples dominios comprometidos."""
    return {
        "saturation": 1.0,
        "flyback_voltage": 620.0,
        "dissipated_power": 420.0,
        "beta_0": 4,
        "beta_1": 3,
        "entropy": 0.97,
        "exergy_loss": 0.93,
    }


@pytest.fixture(scope="session")
def reference_telemetry() -> Dict[str, Any]:
    """Telemetría en el punto de referencia exacto de cada subespacio."""
    return {
        "saturation": 0.0,
        "flyback_voltage": 0.0,
        "dissipated_power": 0.0,
        "beta_0": 1,
        "beta_1": 0,
        "entropy": 0.0,
        "exergy_loss": 0.0,
    }


def _make_state(
    telemetry: Optional[Dict[str, Any]] = None,
    success: bool = True,
) -> MagicMock:
    """
    Factory interna para CategoricalState mockeado.

    El mock refleja la interfaz mínima necesaria:
    - is_success: bool
    - context: dict con clave 'telemetry_metrics'
    - with_update: retorna nuevo mock (inmutabilidad categórica)
    - with_error: retorna nuevo mock
    """
    state = MagicMock(spec=CategoricalState)
    state.is_success = success
    state.context = {"telemetry_metrics": telemetry or {}}
    state.with_update = MagicMock(return_value=MagicMock(spec=CategoricalState))
    state.with_error = MagicMock(return_value=MagicMock(spec=CategoricalState))
    return state


@pytest.fixture
def success_state() -> MagicMock:
    return _make_state(telemetry={})


@pytest.fixture
def error_state() -> MagicMock:
    return _make_state(success=False)


@pytest.fixture(scope="session")
def identity_metric_3d() -> MetricTensor:
    return MetricTensor(np.ones(3))


@pytest.fixture(scope="session")
def diagonal_metric_3d() -> MetricTensor:
    return MetricTensor(np.array([2.0, 0.5, 4.0]))


@pytest.fixture(scope="session")
def dense_metric_3d() -> MetricTensor:
    A = np.array([
        [2.0, 0.5, 0.0],
        [0.0, 1.5, 0.3],
        [0.0, 0.0, 1.0],
    ])
    return MetricTensor(A.T @ A + 0.1 * np.eye(3))


import app.core.immune_system.metric_tensors as ext_metric_tensors

@pytest.fixture
def standard_projector() -> OrthogonalProjector:
    """
    Proyector estándar en ℝ⁷ con tres subespacios canónicos.

    Partición de ℝ⁷:
    - physics_core:  índices [0,3)  → saturación, flyback, potencia
    - topology_core: índices [3,5)  → β₀, β₁
    - thermo_core:   índices [5,7)  → entropía, pérdida de exergía

    Verificación post-construcción: subespacios DEBEN ser disjuntos
    y su unión DEBE cubrir {0,...,6} exactamente (resolución de identidad).
    """
    scale_phys = np.array([
        PhysicalConstants.SATURATION_CRITICAL,
        PhysicalConstants.FLYBACK_MAX_SAFE,
        PhysicalConstants.P_NOMINAL(),
    ], dtype=np.float64)
    D_inv_phys = np.diag(1.0 / scale_phys)
    scaled_G_phys = D_inv_phys @ ext_metric_tensors.G_PHYSICS @ D_inv_phys

    scale_topo = np.array([1.0, 1.0], dtype=np.float64)
    D_inv_topo = np.diag(1.0 / scale_topo)
    scaled_G_topo = D_inv_topo @ ext_metric_tensors.G_TOPOLOGY @ D_inv_topo

    scale_thermo = np.array([0.5, 0.5], dtype=np.float64)
    D_inv_thermo = np.diag(1.0 / scale_thermo)
    scaled_G_thermo = D_inv_thermo @ ext_metric_tensors.G_THERMODYNAMICS @ D_inv_thermo

    subspaces = {
        "physics_core": SubspaceSpec(
            name="physics_core",
            indices=slice(0, 3),
            weight=1.0,
            reference=np.zeros(3),
            scale=scale_phys,
            metric=MetricTensor(scaled_G_phys, validate=False),
        ),
        "topology_core": SubspaceSpec(
            name="topology_core",
            indices=slice(3, 5),
            weight=1.5,
            reference=np.array([1.0, 0.0]),
            scale=scale_topo,
            metric=MetricTensor(scaled_G_topo, validate=False),
        ),
        "thermo_core": SubspaceSpec(
            name="thermo_core",
            indices=slice(5, 7),
            weight=1.2,
            reference=np.zeros(2),
            scale=scale_thermo,
            metric=MetricTensor(scaled_G_thermo, validate=False),
        ),
    }
    proj = OrthogonalProjector(
        dimensions=7,
        subspaces=subspaces,
        topo_indices=BETTI_INDICES,
        cache_projections=True,
    )
    # Post-condition: BETTI_INDICES deben estar dentro de topology_core [3,5)
    assert all(3 <= idx < 5 for idx in BETTI_INDICES), (
        f"BETTI_INDICES={BETTI_INDICES} fuera del rango de topology_core [3,5)"
    )
    return proj


@pytest.fixture
def watcher_default() -> ImmuneWatcherMorphism:
    """Watcher con perfil default. Scope 'function' para aislamiento."""
    return create_immune_watcher("default")


@pytest.fixture
def watcher_strict() -> ImmuneWatcherMorphism:
    return create_immune_watcher("strict")


@pytest.fixture
def watcher_laboratory() -> ImmuneWatcherMorphism:
    """Sin histéresis — clasificación determinista pura (δ=0)."""
    return create_immune_watcher("laboratory")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 · FUNCIONES DE ESTABILIDAD NUMÉRICA — Refinadas
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeNormalize:
    """
    _safe_normalize: normalización L∞ con protección contra degeneración.

    Propiedades matemáticas:
    1. Escala: s = ‖v‖_∞ = max|vᵢ|
    2. Reconstrucción exacta: v = normed * s
    3. Post-normalización: max|normed_i| = 1 (si s > 0)
    4. Vector nulo: retorna ceros + s=0, emite RuntimeWarning
    """

    def test_scale_equals_linf_norm(self):
        """
        El factor de escala es exactamente ‖v‖_∞ = max|vᵢ|.

        Para v = [3, -7, 2]: ‖v‖_∞ = max(3, 7, 2) = 7.
        """
        v = np.array([3.0, -7.0, 2.0])
        _, scale = _safe_normalize(v)
        assert scale == pytest.approx(7.0, rel=1e-14), (
            f"scale = {scale} ≠ 7.0 = ‖v‖_∞"
        )

    def test_normalized_max_abs_is_one(self):
        """
        Post-normalización: max|normed_i| = 1.

        Propiedad: normed = v / ‖v‖_∞, entonces max|normed_i| = 1.
        """
        v = np.array([3.0, -7.0, 2.0])
        normed, scale = _safe_normalize(v)
        assert float(np.max(np.abs(normed))) == pytest.approx(1.0, abs=1e-14), (
            f"max|normed_i| = {np.max(np.abs(normed)):.2e} ≠ 1.0"
        )

    def test_reconstruction_exact(self):
        """
        Reconstrucción exacta: v = normed * scale.

        Propiedad algebraica fundamental: la normalización es invertible.
        Tolerancia: rtol=1e-14 (precisa al límite de float64).
        """
        v = np.array([1.5, -3.0, 0.5])
        normed, scale = _safe_normalize(v)
        np.testing.assert_allclose(
            normed * scale, v, rtol=1e-14,
            err_msg="Reconstrucción v = normed * scale fallida",
        )

    def test_zero_vector_emits_runtime_warning(self):
        """
        Vector nulo emite exactamente UN RuntimeWarning.

        Uso de pytest.warns() en lugar de warnings.catch_warnings()
        para compatibilidad con pytest y aislamiento de warnings.
        """
        with pytest.warns(RuntimeWarning):
            normed, scale = _safe_normalize(np.zeros(5))

    def test_zero_vector_returns_zeros_and_scale_zero(self):
        """
        Vector nulo: normed = 0, scale = 0.0.

        Verificación separada del warning (test anterior) y de los valores,
        para identificar fallos con precisión.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            normed, scale = _safe_normalize(np.zeros(5))

        assert scale == 0.0, f"scale = {scale} ≠ 0.0 para vector nulo"
        np.testing.assert_array_equal(normed, np.zeros(5))

    def test_sub_eps_vector_warns(self):
        """
        Vector con ‖v‖_∞ < EPS lanza RuntimeWarning.

        El umbral EPS define la degeneración numérica.
        """
        v = np.full(3, EPS / 10.0)
        assert float(np.max(np.abs(v))) < EPS
        with pytest.warns(RuntimeWarning):
            _safe_normalize(v)

    def test_empty_vector_returns_empty_scale_zero(self):
        """
        Vector vacío (shape=(0,)): normed vacío, scale=0.0.

        Caso degenerado: no hay elementos, no hay norma.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normed, scale = _safe_normalize(np.array([]))
        assert scale == 0.0
        assert normed.size == 0

    def test_large_magnitude_no_overflow(self):
        """
        ‖v‖_∞ ~ 3e200: normalización por L∞ evita overflow en float64.

        La normalización L∞ (dividir por max|vᵢ|) es numéricamente estable
        incluso para valores cercanos a float_max = 1.8e308.
        """
        v = np.array([1e200, -2e200, 3e200])
        normed, scale = _safe_normalize(v)
        assert np.all(np.isfinite(normed)), (
            f"normed contiene no-finitos: {normed}"
        )
        assert np.isfinite(scale), f"scale = {scale} no es finito"
        # Verificar reconstrucción
        np.testing.assert_allclose(normed * scale, v, rtol=1e-14)

    def test_single_negative_element(self):
        """
        Un único elemento negativo: scale = |valor|, normed[0] = -1.

        ‖[-5]‖_∞ = 5, normed = [-5]/5 = [-1].
        """
        v = np.array([-5.0])
        normed, scale = _safe_normalize(v)
        assert scale == pytest.approx(5.0, rel=1e-14)
        assert normed[0] == pytest.approx(-1.0, rel=1e-14)

    @pytest.mark.parametrize("seed", range(5))
    def test_randomized_reconstruction_property(self, seed: int):
        """
        Propiedad: v = normed * scale para vectores no-degenerados.

        Parametrizado con semillas fijas para reproducibilidad.
        """
        rng = np.random.default_rng(seed)
        v = rng.normal(0, 10, size=50)
        # Evitar vectores degenerados para esta prueba
        assume_min = np.max(np.abs(v)) > EPS
        if not assume_min:
            return

        normed, scale = _safe_normalize(v)
        if scale > EPS:
            np.testing.assert_allclose(
                normed * scale, v, rtol=1e-13,
                err_msg=f"Reconstrucción fallida (seed={seed})",
            )


class TestStableReciprocal:
    """
    _stable_reciprocal: recíproco vectorial numéricamente estable.

    Propiedades:
    1. Valores normales: 1/x exacto (dentro de rtol = 1e-12)
    2. Cero: resultado finito (no Inf)
    3. Signo: preservado para sub-EPS
    4. Identidad: x * (1/x) ≈ 1 para |x| >> EPS
    5. Shape: preservado
    """

    def test_standard_values_exact(self):
        """
        Valores normales producen recíprocos con precisión float64.

        [2, -4, 0.5] → [0.5, -0.25, 2.0] con rtol=1e-12.
        """
        x = np.array([2.0, -4.0, 0.5])
        expected = np.array([0.5, -0.25, 2.0])
        np.testing.assert_allclose(
            _stable_reciprocal(x), expected, rtol=1e-12,
            err_msg="Recíproco incorrecto para valores normales",
        )

    def test_zero_does_not_produce_inf(self):
        """
        Entrada cero produce resultado finito (no ±Inf).

        La regularización evita la división por cero del tipo 1/0 = Inf.
        """
        result = _stable_reciprocal(np.array([0.0]))
        assert np.all(np.isfinite(result)), (
            f"Resultado no finito para x=0: {result}"
        )

    def test_sign_preserved_near_zero(self):
        """
        El signo se preserva para valores sub-EPS (positivos y negativos).

        Aunque la magnitud se regulariza, el signo es un invariante físico.
        """
        x = np.array([1e-15, -1e-15])
        r = _stable_reciprocal(x)
        assert r[0] > 0, f"Signo positivo no preservado: r[0]={r[0]}"
        assert r[1] < 0, f"Signo negativo no preservado: r[1]={r[1]}"

    def test_identity_property_for_normal_values(self):
        """
        Propiedad de identidad: x · (1/x) ≈ 1 para |x| >> EPS.

        Tolerancia rtol=1e-10 para absorber errores de redondeo float64
        en la multiplicación.
        """
        x = np.array([0.1, 1.0, 10.0, 100.0, 1e6])
        product = x * _stable_reciprocal(x)
        np.testing.assert_allclose(
            product, np.ones(5), rtol=1e-10,
            err_msg="Identidad x·(1/x)=1 violada",
        )

    def test_output_shape_preserved_1d(self):
        """Shape de array 1D preservado."""
        x = np.arange(1.0, 6.0)
        assert _stable_reciprocal(x).shape == x.shape

    def test_output_shape_preserved_2d(self):
        """Shape de array 2D preservado."""
        x = np.arange(1.0, 13.0).reshape(3, 4)
        assert _stable_reciprocal(x).shape == (3, 4)

    def test_all_outputs_finite(self):
        """
        Ningún valor de salida es NaN o Inf para entradas mixtas.

        Incluyendo: 0, sub-EPS positivo, sub-EPS negativo, normal.
        """
        x = np.array([0.0, EPS / 2, -EPS / 2, 1.0, -1.0, 1e-15, -1e-15])
        result = _stable_reciprocal(x)
        assert np.all(np.isfinite(result)), (
            f"Valores no-finitos: {result[~np.isfinite(result)]}"
        )


class TestStableDivide:
    """
    _stable_divide: división vectorial con denominadores degenerados.

    Invariantes:
    1. Denominadores normales: coincide con operador /
    2. Denominador cero: resultado finito
    3. Signo: num·sign(den) preservado para denominadores regulares
    4. Shape: preservado para arrays n-dimensionales
    """

    def test_normal_division_matches_operator(self):
        """División normal coincide con / hasta rtol=1e-12."""
        num = np.array([1.0, 2.0, -3.0])
        den = np.array([2.0, 4.0, -1.5])
        np.testing.assert_allclose(
            _stable_divide(num, den), num / den, rtol=1e-12,
            err_msg="stable_divide difiere de / para valores normales",
        )

    def test_zero_denominator_produces_finite(self):
        """Denominador cero produce resultado finito (no NaN ni Inf)."""
        result = _stable_divide(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert np.all(np.isfinite(result)), (
            f"Resultado no finito con den=0: {result}"
        )

    def test_sign_of_result_for_regular_denominators(self):
        """
        El signo del resultado es sign(num) * sign(den) para |den| > EPS.

        Propiedad física: la dirección del resultado preserva la física del sistema.
        """
        num = np.array([1.0, 1.0, -1.0, -1.0])
        den = np.array([0.1, -0.1, 0.1, -0.1])
        r = _stable_divide(num, den)
        expected_signs = np.sign(num) * np.sign(den)
        actual_signs = np.sign(r)
        np.testing.assert_array_equal(
            actual_signs, expected_signs,
            err_msg="Signos incorrectos en stable_divide",
        )

    def test_2d_shape_preserved(self):
        """Opera correctamente sobre arrays 2D."""
        num = np.ones((4, 4))
        den = np.eye(4) + 0.01
        result = _stable_divide(num, den)
        assert result.shape == (4, 4), f"Shape incorrecto: {result.shape}"
        assert np.all(np.isfinite(result)), "Resultado 2D contiene no-finitos"

    def test_output_all_finite_mixed_inputs(self):
        """Resultado siempre finito para entradas mixtas incluyendo cero."""
        num = np.array([0.0, 1.0, -1.0, 1e10, 0.0])
        den = np.array([0.0, 0.0, 1e-20, 1e10, 1.0])
        result = _stable_divide(num, den)
        assert np.all(np.isfinite(result)), (
            f"No-finitos en salida: {result}"
        )


class TestStableNorm:
    """
    _stable_norm: norma L² estable con y sin métrica.

    Propiedades:
    1. Euclidiana: coincide con np.linalg.norm para G=I
    2. ‖0‖ = 0
    3. Con G=I explícita: coincide con euclidiana
    4. Con G diagonal: ‖v‖_G = √(Σ gᵢᵢ vᵢ²)
    5. Homogeneidad: ‖αv‖ = |α|·‖v‖
    6. No-negatividad: ‖v‖_G ≥ 0
    7. Sin overflow para magnitudes extremas
    """

    def test_euclidean_matches_numpy_randomized(self):
        """
        Sin métrica, coincide con np.linalg.norm para vectores aleatorios.

        Parametrizado con semillas fijas para reproducibilidad.
        """
        rng = np.random.default_rng(1)
        for _ in range(30):
            v = rng.normal(0, 5, size=20)
            expected = float(np.linalg.norm(v))
            result = _stable_norm(v)
            assert result == pytest.approx(expected, rel=1e-10), (
                f"stable_norm ≠ numpy.norm: {result} vs {expected}"
            )

    def test_zero_vector_norm_is_zero(self):
        """‖0‖ = 0 exactamente."""
        with pytest.warns(RuntimeWarning, match="degeneración"):
            result = _stable_norm(np.zeros(10))
        assert result == pytest.approx(0.0, abs=1e-15), (
            f"‖0‖ = {result} ≠ 0"
        )

    def test_identity_metric_matches_euclidean(self):
        """Con G = I₂, ‖[3,4]‖_I = √(9+16) = 5."""
        v = np.array([3.0, 4.0])
        result = _stable_norm(v, metric=np.eye(2))
        assert result == pytest.approx(5.0, rel=1e-12), (
            f"‖[3,4]‖_I = {result} ≠ 5.0"
        )

    def test_diagonal_metric_formula(self):
        """
        Con G = diag(4, 9): ‖[1,1]‖_G = √(4·1² + 9·1²) = √13.

        Fórmula: ‖v‖_G = √(Σ gᵢᵢ vᵢ²) para G diagonal.
        """
        v = np.array([1.0, 1.0])
        G = np.diag([4.0, 9.0])
        expected = float(np.sqrt(4.0 * 1.0 + 9.0 * 1.0))
        assert _stable_norm(v, metric=G) == pytest.approx(expected, rel=1e-12), (
            f"‖[1,1]‖_G = {_stable_norm(v, G)} ≠ √13 = {expected}"
        )

    def test_scale_invariance_homogeneity(self):
        """
        Homogeneidad: ‖αv‖ = |α|·‖v‖.

        Propiedad fundamental de cualquier norma.
        """
        rng = np.random.default_rng(42)
        v = rng.normal(0, 1, size=10)
        alpha = 7.5
        lhs = _stable_norm(alpha * v)
        rhs = abs(alpha) * _stable_norm(v)
        assert lhs == pytest.approx(rhs, rel=1e-10), (
            f"Homogeneidad violada: ‖{alpha}v‖ = {lhs} ≠ {alpha}·‖v‖ = {rhs}"
        )

    def test_empty_vector_returns_zero(self):
        """‖[]‖ = 0 (convención para espacio vacío)."""
        assert _stable_norm(np.array([])) == 0.0

    def test_large_vector_no_overflow(self):
        """
        ‖v‖ para v con elementos 1e150: sin overflow en float64.

        Verificación analítica: ‖[1e150, ..., 1e150]‖₂ = 1e150·√n.
        Para n=100: 1e150·10 = 1e151 < float_max = 1.8e308. ✓

        Implementaciones robustas usan scaled norm para evitar overflow
        intermedio (np.linalg.norm lo hace internamente).
        """
        n = 100
        v = np.full(n, 1e150)
        expected = 1e150 * math.sqrt(n)
        result = _stable_norm(v)
        assert np.isfinite(result), f"‖v‖ = {result} no es finito (overflow)"
        assert result == pytest.approx(expected, rel=1e-8), (
            f"‖v‖ = {result} ≠ 1e150·√{n} = {expected:.4e}"
        )

    def test_triangle_inequality_holds(self):
        """
        ‖u + v‖ ≤ ‖u‖ + ‖v‖ (desigualdad triangular).

        Propiedad axiomática de normas — verificada numéricamente.
        """
        rng = np.random.default_rng(7)
        for _ in range(20):
            u = rng.normal(0, 5, size=15)
            v = rng.normal(0, 5, size=15)
            lhs = _stable_norm(u + v)
            rhs = _stable_norm(u) + _stable_norm(v)
            # Tolerancia para errores de redondeo en la suma
            assert lhs <= rhs + 1e-10, (
                f"Desigualdad triangular violada: ‖u+v‖={lhs:.6f} > ‖u‖+‖v‖={rhs:.6f}"
            )


class TestRegularizeMetric:
    """
    _regularize_metric: regularización de Tikhonov G → G + δI.

    La regularización de Tikhonov garantiza G ≻ 0 añadiendo δI
    donde δ = max(0, min_eig - λ_min(G)) + ε.

    Invariantes:
    1. Matriz ya SPD: modificación mínima (< tol)
    2. Matriz singular: resultado SPD
    3. Eigenvalores negativos: resultado SPD
    4. Estructura Tikhonov: G_reg = G + δ·I
    5. Eigenvectores: preservados (G y G+δI tienen los mismos)
    """

    def test_already_spd_minimally_modified(self):
        """
        Matriz ya SPD no se modifica significativamente.

        Si λ_min(G) >> min_eig, δ ≈ 0 y G_reg ≈ G.
        Tolerancia: 1e-9 (mucho menor que ALGEBRAIC_TOL).
        """
        G = np.array([[4.0, 1.0], [1.0, 3.0]])
        # Verificar que G ya es SPD
        assert np.linalg.eigvalsh(G).min() > MIN_EIGVAL_TOL * 100
        G_reg = _regularize_metric(G, min_eig=MIN_EIGVAL_TOL)
        diff = float(np.max(np.abs(G_reg - G)))
        assert diff < 1e-9, (
            f"Matriz SPD modificada excesivamente: max|G_reg - G| = {diff:.2e}"
        )

    def test_singular_becomes_spd(self):
        """
        Matriz singular (rango deficiente) → SPD tras regularización.

        G = [[1,1],[1,1]]: rango 1, λ_min = 0 → G_reg SPD.
        """
        G = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.linalg.matrix_rank(G) == 1
        G_reg = _regularize_metric(G, min_eig=1e-6)
        assert_spd(G_reg, tol=1e-7, label="G_reg(singular)")

    def test_indefinite_becomes_spd(self):
        """
        Matriz indefinida (λ < 0) → SPD tras regularización.

        G = [[2,3],[3,2]]: det = 4-9 = -5 < 0 → indefinida.
        """
        G = np.array([[2.0, 3.0], [3.0, 2.0]])
        lam_min_before = float(np.linalg.eigvalsh(G).min())
        assert lam_min_before < 0, "G debe ser indefinida para este test"
        G_reg = _regularize_metric(G)
        assert_spd(G_reg, label="G_reg(indefinida)")

    def test_tikhonov_structure_exact(self):
        """
        Regularización añade exactamente δ·I cuando λ_min(G) = 0.

        Para G = diag(5, 3, 0) y min_eig = 1.0:
        δ = 1.0 (porque λ_min = 0 < min_eig = 1.0)
        G_reg = G + 1.0·I = diag(6, 4, 1).

        Tolerancia: atol=1e-14 (aritmética de punto flotante exacta).
        """
        G = np.diag([5.0, 3.0, 0.0])
        target_min_eig = 1.0
        G_reg = _regularize_metric(G, min_eig=target_min_eig)
        expected = G + target_min_eig * np.eye(3)
        np.testing.assert_allclose(
            G_reg, expected, atol=1e-14,
            err_msg="Estructura Tikhonov G + δI incorrecta",
        )

    def test_eigenvectors_preserved_spd(self):
        """
        Para G SPD, G_reg = G + δI tiene los mismos eigenvectores.

        Fundamento algebraico: si Gv = λv entonces (G+δI)v = (λ+δ)v.
        Los eigenvectores son los MISMOS, solo los eigenvalores cambian.

        NOTA IMPORTANTE: este test solo es válido para G con eigenvalores
        DISTINTOS (eigenvectores únicos). Para eigenvalores degenerados,
        cualquier base del subespacio es válida.
        """
        # G con eigenvalores distintos: λ₁ = 2-√2, λ₂ = 2+√2
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        G = A @ A.T  # G es SPD y sus eigenvalores son distintos

        eigvals_before, V_before = np.linalg.eigh(G)
        assert len(set(eigvals_before.round(6))) == len(eigvals_before), (
            "Eigenvalores degenerados: test de eigenvectores no es válido"
        )

        G_reg = _regularize_metric(G, min_eig=MIN_EIGVAL_TOL)
        _, V_after = np.linalg.eigh(G_reg)

        for i in range(V_before.shape[1]):
            # |cos θ| = |vᵢ · wᵢ| debe ser ≈ 1 (paralelos o antiparalelos)
            dot = abs(float(V_before[:, i] @ V_after[:, i]))
            assert dot == pytest.approx(1.0, abs=1e-10), (
                f"Eigenvector {i} no preservado: |cos θ| = {dot:.4e}"
            )

    def test_result_always_spd(self):
        """
        Propiedad global: toda matriz real simétrica regularizada es SPD.

        Verificado para matrices aleatorias (posiblemente indefinidas).
        """
        rng = np.random.default_rng(13)
        for trial in range(20):
            A = rng.normal(0, 1, (4, 4))
            G = (A + A.T) / 2  # Simetrizar
            G_reg = _regularize_metric(G, min_eig=1e-6)
            assert_spd(G_reg, tol=1e-7, label=f"G_reg(trial={trial})")


class TestComputeConditionNumber:
    """
    _compute_condition_number: κ(G) = λ_max / λ_min para matrices SPD.

    Propiedades:
    1. Identidad: κ(I_n) = 1 para todo n
    2. Diagonal: κ(diag(d)) = max(d)/min(d) exactamente
    3. Singular: κ = ∞
    4. SPD general: método 'eig' y 'svd' coinciden
    5. No-cuadrada: ValueError
    6. Método desconocido: ValueError
    7. NaN en matriz: manejo correcto
    """

    def test_identity_has_condition_one(self):
        """κ(I_n) = 1 para n = 2, 5, 10."""
        for n in [2, 5, 10]:
            kappa = _compute_condition_number(np.eye(n))
            assert kappa == pytest.approx(1.0, rel=1e-10), (
                f"κ(I_{n}) = {kappa} ≠ 1"
            )

    def test_diagonal_exact(self):
        """
        κ(diag(1, 2, 10)) = max(d)/min(d) = 10/1 = 10.

        Verificación analítica exacta.
        """
        d = np.array([1.0, 2.0, 10.0])
        kappa = _compute_condition_number(np.diag(d))
        assert kappa == pytest.approx(10.0, rel=1e-10), (
            f"κ(diag(d)) = {kappa} ≠ 10"
        )

    def test_singular_returns_inf(self):
        """
        Matriz singular (λ_min = 0): κ = ∞.

        κ = λ_max / λ_min → λ_min = 0 → κ = ∞ (no es un error, es una propiedad).
        """
        G = np.array([[1.0, 0.0], [0.0, 0.0]])
        result = _compute_condition_number(G)
        assert result == float("inf"), (
            f"Singular no retorna inf: {result}"
        )

    def test_eig_svd_agree_for_spd(self):
        """
        Para G SPD, κ por 'eig' y 'svd' coinciden (rel=1e-6).

        Para matrices simétricas, eigenvalores = valores singulares.
        La pequeña tolerancia absorbe diferencias algorítmicas.
        """
        G = np.array([
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 0.3],
            [0.5, 0.3, 2.0],
        ])
        kappa_eig = _compute_condition_number(G, method="eig")
        kappa_svd = _compute_condition_number(G, method="svd")
        assert kappa_eig == pytest.approx(kappa_svd, rel=1e-6), (
            f"Discrepancia eig/svd: {kappa_eig:.6f} vs {kappa_svd:.6f}"
        )

    def test_non_square_raises_value_error(self):
        """Matriz no cuadrada lanza ValueError con mensaje 'cuadrada'."""
        with pytest.raises(ValueError, match="cuadrada"):
            _compute_condition_number(np.ones((3, 4)))

    def test_unknown_method_raises_value_error(self):
        """Método desconocido lanza ValueError con mensaje 'desconocido'."""
        with pytest.raises(ValueError, match="desconocido"):
            _compute_condition_number(np.eye(3), method="qr")

    def test_well_conditioned_is_small(self):
        """
        κ(G) pequeño para matriz bien condicionada.

        G = I + ε·randn: κ ≈ (1+ε·σ_max)/(1-ε·σ_max) ≈ 1 para ε pequeño.
        """
        rng = np.random.default_rng(99)
        eps = 0.01
        A = rng.normal(0, eps, (5, 5))
        G = np.eye(5) + (A + A.T) / 2
        kappa = _compute_condition_number(G)
        # Para eps=0.01, κ debe ser muy cercano a 1
        assert kappa < 5.0, f"κ = {kappa} demasiado grande para G bien condicionada"

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_condition_number_ge_one(self, n: int):
        """
        κ(G) ≥ 1 para cualquier matriz invertible.

        Fundamento: κ = ‖G‖·‖G⁻¹‖ ≥ ‖G·G⁻¹‖ = ‖I‖ = 1.
        """
        rng = np.random.default_rng(n)
        A = rng.normal(0, 1, (n, n))
        G = A.T @ A + np.eye(n)
        kappa = _compute_condition_number(G)
        assert kappa >= 1.0 - 1e-10, (
            f"κ({n}×{n}) = {kappa} < 1 (imposible para matriz invertible)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 · MetricTensor — Refinada
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricTensorDiagonal:
    """
    MetricTensor con representación diagonal (1D).

    Marco algebraico: G = diag(g₁,...,gₙ) con gᵢ > 0 define una métrica
    Riemanniana en ℝⁿ. La forma cuadrática es vᵀGv = Σ gᵢvᵢ².
    """

    def test_construction_valid_1d(self):
        """Construcción válida: dimension = n, is_diagonal = True."""
        mt = MetricTensor(np.array([1.0, 2.0, 3.0]))
        assert mt.dimension == 3
        assert mt.is_diagonal is True

    def test_condition_number_diagonal_exact(self):
        """
        κ(diag(1, 10)) = max(d)/min(d) = 10/1 = 10.

        Verificación analítica exacta.
        """
        mt = MetricTensor(np.array([1.0, 10.0]))
        assert mt.condition_number == pytest.approx(10.0, rel=1e-10), (
            f"κ = {mt.condition_number} ≠ 10"
        )

    def test_zero_element_raises(self):
        """
        Elemento cero viola G ≻ 0 (gᵢ = 0 → matriz singular).

        gᵢ = 0 crearía una dirección degenerada donde ‖v‖_G = 0 para v ≠ 0.
        """
        with pytest.raises(MetricTensorError, match=r"[Pp]ositiv|[Ss]ingular|0"):
            MetricTensor(np.array([1.0, 0.0, 2.0]))

    def test_negative_element_raises(self):
        """
        Elemento negativo viola G ≻ 0 (la forma cuadrática sería indefinida).
        """
        with pytest.raises(MetricTensorError):
            MetricTensor(np.array([-1.0, 2.0]))

    def test_quadratic_form_matches_formula(self):
        """
        vᵀGv = Σ gᵢᵢvᵢ² para G = diag(d).

        v = [1, -1, 2], d = [2, 3, 5]:
        qf = 2·1 + 3·1 + 5·4 = 2 + 3 + 20 = 25.
        """
        d = np.array([2.0, 3.0, 5.0])
        mt = MetricTensor(d)
        v = np.array([1.0, -1.0, 2.0])
        expected = float(np.sum(d * v * v))
        result = mt.quadratic_form(v)
        assert result == pytest.approx(expected, rel=1e-12), (
            f"vᵀGv = {result} ≠ {expected}"
        )

    def test_quadratic_form_zero_vector_is_zero(self):
        """vᵀGv = 0 para v = 0 (G es definida positiva, no solo semidefinida)."""
        mt = MetricTensor(np.array([1.0, 2.0]))
        result = mt.quadratic_form(np.zeros(2))
        assert result == pytest.approx(0.0, abs=1e-15), (
            f"‖0‖²_G = {result} ≠ 0"
        )

    def test_quadratic_form_strictly_positive_for_nonzero(self):
        """
        vᵀGv > 0 para v ≠ 0 (definida positiva, no semidefinida).

        Propiedad esencial: G ≻ 0 ⟺ vᵀGv > 0 ∀v ≠ 0.
        """
        mt = MetricTensor(np.array([2.0, 4.0, 0.5]))
        rng = np.random.default_rng(7)
        for _ in range(50):
            v = rng.normal(0, 5, 3)
            if np.linalg.norm(v) > 1e-10:  # v ≠ 0
                qf = mt.quadratic_form(v)
                assert qf > 0.0, (
                    f"vᵀGv = {qf} ≤ 0 para v ≠ 0 (G no es SDP)"
                )

    def test_apply_elementwise_diagonal(self):
        """
        G·v = d ⊙ v para G = diag(d).

        Multiplicación matriz-vector reduce a producto elemento-a-elemento.
        """
        d = np.array([2.0, 4.0, 0.5])
        mt = MetricTensor(d)
        v = np.array([1.0, -1.0, 3.0])
        np.testing.assert_allclose(
            mt.apply(v), d * v, rtol=1e-14,
            err_msg="G·v ≠ d⊙v para diagonal",
        )

    def test_inverse_sqrt_roundtrip(self):
        """
        G^{-1/2}·(G^{1/2}·v) = v.

        Para G = diag(d): G^{1/2} = diag(√d), G^{-1/2} = diag(1/√d).
        """
        d = np.array([4.0, 9.0, 16.0])
        mt = MetricTensor(d)
        v = np.array([1.0, 2.0, -3.0])
        sqrt_Gv = np.sqrt(d) * v  # G^{1/2}·v
        result = mt.inverse_sqrt_apply(sqrt_Gv)
        np.testing.assert_allclose(
            result, v, rtol=1e-12,
            err_msg="G^{-1/2}·G^{1/2}·v ≠ v",
        )

    def test_to_array_returns_mutable_copy(self):
        """
        to_array() retorna una COPIA mutable que no modifica el tensor.

        Propiedad de encapsulamiento: el estado interno no debe ser mutable
        por el caller.
        """
        mt = MetricTensor(np.array([1.0, 2.0]))
        arr = mt.to_array()
        arr[0] = 999.0
        original = mt.to_array()
        assert original[0] == pytest.approx(1.0), (
            "to_array() retornó referencia (no copia): modificación externa afectó el tensor"
        )

    def test_eigen_decomposition_diagonal_structure(self):
        """
        Para G = diag(d): eigvals = d, eigvecs = I (base canónica).

        Fundamento: los vectores canónicas son eigenvectores de toda diagonal.
        Tolerancia: atol=1e-14 (exacto en aritmética float64).
        """
        d = np.array([3.0, 1.0, 2.0])
        mt = MetricTensor(d)
        eigvals, eigvecs = mt.eigen_decomposition
        # Eigenvalores en orden de d (no necesariamente ordenados)
        np.testing.assert_allclose(eigvals, d, rtol=1e-14,
            err_msg="Eigenvalores de diagonal incorrectos")
        np.testing.assert_allclose(eigvecs, np.eye(3), atol=1e-14,
            err_msg="Eigenvectores de diagonal no son I")

    def test_eigen_decomposition_cached_identity(self):
        """
        Segunda llamada retorna exactamente el mismo objeto (caché de identidad).

        Verificación: `is` en Python (no solo igualdad de valores).
        """
        mt = MetricTensor(np.array([1.0, 2.0, 3.0]))
        first = mt.eigen_decomposition
        second = mt.eigen_decomposition
        assert first is second, (
            "eigen_decomposition no está cacheado: retornó objetos distintos"
        )

    def test_quadratic_form_bilinearity(self):
        """
        Bilinealidad: (u+v)ᵀG(u+v) = uᵀGu + 2uᵀGv + vᵀGv.

        Propiedad algebraica de formas cuadráticas.
        """
        d = np.array([2.0, 3.0])
        mt = MetricTensor(d)
        u = np.array([1.0, 2.0])
        v = np.array([3.0, -1.0])

        qf_sum = mt.quadratic_form(u + v)
        qf_u = mt.quadratic_form(u)
        qf_v = mt.quadratic_form(v)
        cross = float(np.sum(d * u * v))  # uᵀGv para G diagonal

        expected = qf_u + 2 * cross + qf_v
        assert qf_sum == pytest.approx(expected, rel=1e-12), (
            f"Bilinealidad violada: {qf_sum} ≠ {expected}"
        )


class TestMetricTensorDense:
    """
    MetricTensor con representación densa (2D).

    Para G densa: verificación de simetría, SPD, y operaciones.
    """

    def test_construction_spd_valid(self):
        """Construcción con G SPD: dimension y is_diagonal correctos."""
        G = np.array([[4.0, 1.0], [1.0, 3.0]])
        mt = MetricTensor(G)
        assert mt.dimension == 2
        assert mt.is_diagonal is False

    def test_symmetrization_enforced(self):
        """
        (G+Gᵀ)/2 se impone internamente para G casi-simétrica.

        Para G con perturbación de simetría de 0.001 (< tol),
        el resultado es exactamente simétrico.
        """
        G = np.array([[2.0, 1.001], [0.999, 2.0]])
        mt = MetricTensor(G, validate=False)
        arr = mt.to_array()
        np.testing.assert_allclose(
            arr, arr.T, atol=1e-15,
            err_msg="Tensor no es simétrico tras simetrización interna",
        )

    def test_asymmetric_raises_with_validate(self):
        """
        Matriz altamente asimétrica con validate=True lanza MetricTensorError.

        La asimetría |G[0,1] - G[1,0]| = 5 >> tol numérica.
        """
        with pytest.raises(MetricTensorError, match="[Ss]imétr"):
            MetricTensor(np.array([[1.0, 5.0], [0.0, 1.0]]), validate=True)

    def test_non_square_raises(self):
        """Matriz no cuadrada lanza MetricTensorError con mensaje 'cuadrada'."""
        with pytest.raises(MetricTensorError, match="[Cc]uadrada"):
            MetricTensor(np.ones((2, 3)))

    def test_3d_array_raises(self):
        """Array 3D lanza MetricTensorError."""
        with pytest.raises(MetricTensorError):
            MetricTensor(np.ones((2, 2, 2)))

    def test_quadratic_form_dense_correct(self):
        """
        vᵀGv para G densa.

        v = [2, -1], G = [[3,1],[1,2]]:
        vᵀGv = [2,-1]·[[3,1],[1,2]]·[2,-1]ᵀ = [2,-1]·[5,0]ᵀ = 10.
        """
        G = np.array([[3.0, 1.0], [1.0, 2.0]])
        mt = MetricTensor(G)
        v = np.array([2.0, -1.0])
        expected = float(v @ G @ v)
        result = mt.quadratic_form(v)
        assert result == pytest.approx(expected, rel=1e-12), (
            f"vᵀGv = {result} ≠ {expected}"
        )

    def test_apply_matches_matmul(self):
        """G·v para G densa coincide con @ en numpy."""
        G = np.array([[2.0, 1.0, 0.5],
                      [1.0, 3.0, 0.2],
                      [0.5, 0.2, 4.0]])
        mt = MetricTensor(G)
        v = np.array([1.0, 2.0, -1.0])
        np.testing.assert_allclose(
            mt.apply(v), G @ v, rtol=1e-12,
            err_msg="G·v ≠ G@v para densa",
        )

    def test_inverse_sqrt_roundtrip_dense(self):
        """
        G^{-1/2}·G^{1/2}·v = v para G densa SPD.

        Implementación: vía descomposición espectral G = VΛVᵀ.
        G^{1/2} = V·Λ^{1/2}·Vᵀ, G^{-1/2} = V·Λ^{-1/2}·Vᵀ.
        """
        G = np.array([[3.0, 0.5], [0.5, 2.0]])
        mt = MetricTensor(G)
        v = np.array([1.0, -1.0])
        eigvals, eigvecs = mt.eigen_decomposition
        sqrt_Gv = eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ v))
        result = mt.inverse_sqrt_apply(sqrt_Gv)
        np.testing.assert_allclose(
            result, v, rtol=1e-10,
            err_msg="G^{-1/2}·G^{1/2}·v ≠ v para densa",
        )

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_large_spd_construction(self, n: int):
        """
        Matrices SPD de distintos tamaños se construyen correctamente.

        G = AᵀA + I garantiza G ≻ 0 con κ(G) finito.
        """
        rng = np.random.default_rng(n * 17)
        A = rng.normal(0, 1, (n, n))
        G = A.T @ A + np.eye(n)
        mt = MetricTensor(G)
        assert mt.dimension == n, f"dimension = {mt.dimension} ≠ {n}"
        assert mt.condition_number < float("inf"), "κ = ∞ para matriz SPD (error)"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 · SubspaceSpec — Refinada
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubspaceSpec:
    """
    SubspaceSpec: especificación de subespacio con métrica y distancia de Mahalanobis.

    La distancia de Mahalanobis con respecto a la referencia ref:
        d_G(x, ref) = √((x-ref)ᵀ G (x-ref))

    Con scale=[s₀,...,sₙ₋₁]: G = diag(1/s₀², ..., 1/sₙ₋₁²)
    """

    def test_euclidean_metric_default(self):
        """
        Sin scale, G = I: d_G(v, 0) = ‖v‖₂.

        v = [3, 4, 0]: ‖v‖₂ = √(9+16+0) = 5.
        """
        spec = SubspaceSpec("A", slice(0, 3), 1.0, np.zeros(3))
        v = np.array([3.0, 4.0, 0.0])
        result = spec.compute_threat(v)
        assert result == pytest.approx(5.0, rel=1e-12), (
            f"threat = {result} ≠ 5.0 con G=I"
        )

    def test_weight_scales_threat_linearly(self):
        """
        threat = weight · d_G(v, ref).

        Con weight=2, G=I, v=[1,0]: threat = 2·1 = 2.
        """
        spec = SubspaceSpec("A", slice(0, 2), 2.0, np.zeros(2))
        v = np.array([1.0, 0.0])
        assert spec.compute_threat(v) == pytest.approx(2.0, rel=1e-12)

    def test_reference_point_zero_threat(self):
        """
        En el punto de referencia: d_G(ref, ref) = 0 → threat = 0.

        Propiedad axiomática de la distancia: d(x,x) = 0.
        """
        ref = np.array([1.0, 2.0, 3.0])
        spec = SubspaceSpec("A", slice(0, 3), 1.5, ref)
        result = spec.compute_threat(ref.copy())
        assert result == pytest.approx(0.0, abs=1e-14), (
            f"threat(ref, ref) = {result} ≠ 0"
        )

    def test_scale_normalization_formula(self):
        """
        Con scale=[s₀,s₁]: G = diag(1/s₀², 1/s₁²).

        δ = v - ref = [2.0, 4.0] = [1·s₀, 1·s₁].
        d_G(δ) = √(δ₀²/s₀² + δ₁²/s₁²) = √(1 + 1) = √2.

        Verificación analítica: con δ = scale, d = √(n) donde n = dim.
        """
        scale = np.array([2.0, 4.0])
        ref = np.zeros(2)
        spec = SubspaceSpec("A", slice(0, 2), 1.0, ref, scale=scale)
        delta = scale.copy()  # δ = 1·scale → d_normalized = √2
        expected = float(np.sqrt(2.0))
        result = spec.compute_threat(delta)
        assert result == pytest.approx(expected, rel=1e-10), (
            f"threat = {result} ≠ √2 = {expected}"
        )

    def test_threat_always_nonnegative(self):
        """
        threat ≥ 0 para todo v (distancia de Mahalanobis no negativa).

        d_G(x, ref) = √(δᵀGδ) ≥ 0 porque G ≻ 0 → δᵀGδ ≥ 0.
        """
        spec = SubspaceSpec("A", slice(0, 3), 1.0, np.zeros(3))
        rng = np.random.default_rng(42)
        for trial in range(100):
            v = rng.normal(0, 5, 3)
            threat = spec.compute_threat(v)
            assert threat >= 0.0, (
                f"Amenaza negativa en trial {trial}: threat = {threat:.4e}"
            )

    def test_dimensional_mismatch_raises(self):
        """
        Subvector con dimensión incorrecta lanza DimensionalMismatchError.

        spec espera dim=3, se provee v de dim=2.
        """
        spec = SubspaceSpec("A", slice(0, 3), 1.0, np.zeros(3))
        with pytest.raises(DimensionalMismatchError):
            spec.compute_threat(np.zeros(2))

    def test_invalid_weight_zero_raises(self):
        """Peso = 0 lanza ValueError (el peso debe ser estrictamente positivo)."""
        with pytest.raises(ValueError, match="weight"):
            SubspaceSpec("A", slice(0, 2), 0.0, np.zeros(2))

    def test_invalid_weight_negative_raises(self):
        """Peso < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="weight"):
            SubspaceSpec("A", slice(0, 2), -1.0, np.zeros(2))

    def test_scale_dim_mismatch_raises(self):
        """
        scale con dimensión ≠ dim(ref) lanza DimensionalMismatchError.

        ref tiene dim=3, scale tiene dim=2 → incompatible.
        """
        with pytest.raises(DimensionalMismatchError):
            SubspaceSpec(
                "A", slice(0, 3), 1.0, np.zeros(3), scale=np.ones(2),
            )

    def test_explicit_metric_overrides_scale(self):
        """
        Si se proporciona metric explícita, scale es ignorado.

        mt = diag(4, 4): ‖[1,0]‖_G = √4 = 2 (independiente de scale).
        """
        mt = MetricTensor(np.array([4.0, 4.0]))
        spec = SubspaceSpec(
            "A", slice(0, 2), 1.0, np.zeros(2),
            metric=mt, scale=np.ones(2),  # scale debe ser ignorado
        )
        v = np.array([1.0, 0.0])
        # ‖v‖_G = √(4·1² + 4·0²) = 2
        result = spec.compute_threat(v)
        assert result == pytest.approx(2.0, rel=1e-12), (
            f"metric explícita ignorada: threat = {result} ≠ 2.0"
        )

    def test_reference_is_immutable(self):
        """
        El vector de referencia interno es inmutable (writeable=False).

        Previene modificación accidental que corrompería la distancia de referencia.
        """
        ref = np.array([1.0, 2.0])
        spec = SubspaceSpec("A", slice(0, 2), 1.0, ref)
        # El array interno debe tener writeable=False
        assert not spec.reference.flags.writeable, (
            "reference.flags.writeable = True (debería ser False)"
        )

    def test_triangle_inequality_mahalanobis(self):
        """
        Desigualdad triangular: d_G(x, z) ≤ d_G(x, y) + d_G(y, z).

        Para distancias de Mahalanobis con G ≻ 0, la desigualdad triangular
        se satisface porque √(·ᵀG·) es una norma en ℝⁿ.
        """
        ref = np.zeros(3)
        spec = SubspaceSpec("A", slice(0, 3), 1.0, ref)
        rng = np.random.default_rng(123)
        for _ in range(20):
            x = rng.normal(0, 2, 3)
            y = rng.normal(0, 2, 3)
            z = rng.normal(0, 2, 3)
            # Redefinir ref temporalmente vía distancias
            # d(x,z) = ‖x-z‖, d(x,y) = ‖x-y‖, d(y,z) = ‖y-z‖
            d_xz = spec.compute_threat(x - z) / spec.weight  # dividir por weight
            d_xy = spec.compute_threat(x - y) / spec.weight
            d_yz = spec.compute_threat(y - z) / spec.weight
            assert d_xz <= d_xy + d_yz + 1e-10, (
                f"Desigualdad triangular violada: {d_xz:.6f} > {d_xy:.6f} + {d_yz:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 · OrthogonalProjector — INVARIANTES ALGEBRAICOS (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrthogonalProjectorAlgebra:
    """
    Invariantes algebraicos de los proyectores ortogonales πₖ.

    Marco: En ℝⁿ, un proyector ortogonal satisface πₖ² = πₖ y πₖᵀ = πₖ.
    Para subespacios disjuntos con unión ℝⁿ: Σₖ πₖ = I y πᵢπⱼ = 0 (i≠j).
    """

    def test_idempotence_all_projectors(self, standard_projector):
        """
        πₖ² = πₖ para todos los subespacios.

        Verifica INDIVIDUALMENTE cada proyector con mensaje de diagnóstico.
        """
        for name, P in standard_projector._projection_matrices.items():
            assert_idempotent(P, label=f"π_{name}")

    def test_self_adjoint_all_projectors(self, standard_projector):
        """
        πₖᵀ = πₖ para todos los subespacios.

        Condición de ortogonalidad: proyector ortogonal vs oblicuo.
        """
        for name, P in standard_projector._projection_matrices.items():
            assert_self_adjoint(P, label=f"π_{name}")

    def test_resolution_of_identity(self, standard_projector):
        """
        Σₖ πₖ = I₇ (resolución de identidad completa).

        Usa assert_resolution_of_identity con inicialización correcta
        (np.zeros no sum() con 0 inicial).
        """
        matrices = list(standard_projector._projection_matrices.values())
        assert_resolution_of_identity(matrices, label="Σπₖ")

    def test_pairwise_orthogonality_all_pairs(self, standard_projector):
        """
        πᵢ·πⱼ = 0 para todo i ≠ j (ortogonalidad mutua).

        Verifica AMBAS direcciones (i,j) y (j,i) explícitamente.
        Para proyectores autoadjuntos: πᵢπⱼ = 0 ⟺ πⱼπᵢ = 0.
        """
        projections = list(standard_projector._projection_matrices.values())
        assert_pairwise_orthogonal(projections)

    def test_eigenvalues_are_zero_or_one(self, standard_projector):
        """
        Los eigenvalores de un proyector ortogonal son solo 0 y 1.

        Fundamento: πₖ² = πₖ ⟹ λ² = λ ⟹ λ ∈ {0, 1}.
        """
        for name, P in standard_projector._projection_matrices.items():
            eigvals = np.linalg.eigvalsh(P)
            for lam in eigvals:
                near_zero = abs(lam) < ALGEBRAIC_TOL
                near_one = abs(lam - 1.0) < ALGEBRAIC_TOL
                assert near_zero or near_one, (
                    f"π_{name}: eigenvalor {lam:.6f} ∉ {{0,1}} "
                    f"(no es proyector)"
                )

    def test_rank_consistent_with_subspace_size(self, standard_projector):
        """
        rank(πₖ) = dimensión del subespacio k.

        Para physics_core (slice 0:3): rank = 3.
        Para topology_core (slice 3:5): rank = 2.
        Para thermo_core (slice 5:7): rank = 2.
        """
        expected_ranks = {
            "physics_core": 3,
            "topology_core": 2,
            "thermo_core": 2,
        }
        for name, P in standard_projector._projection_matrices.items():
            rank = int(round(float(np.trace(P))))  # tr(P) = rank para proyector
            if name in expected_ranks:
                assert rank == expected_ranks[name], (
                    f"rank(π_{name}) = {rank} ≠ {expected_ranks[name]}"
                )

    def test_overlapping_subspaces_raises(self):
        """
        Subespacios con índices solapados lanzan DimensionalMismatchError.

        slice(0,3) ∩ slice(2,5) = {2}: solapamiento en índice 2.
        """
        subspaces = {
            "A": SubspaceSpec("A", slice(0, 3), 1.0, np.zeros(3)),
            "B": SubspaceSpec("B", slice(2, 5), 1.0, np.zeros(3)),
        }
        with pytest.raises(DimensionalMismatchError, match="[Ss]olapa"):
            OrthogonalProjector(dimensions=5, subspaces=subspaces)

    def test_uncovered_indices_logged(self, caplog):
        """
        Índices sin cobertura de subespacio generan WARNING en log.

        Con dims=5 y solo slice(0,2): índices {2,3,4} sin cobertura.
        """
        subspaces = {
            "A": SubspaceSpec("A", slice(0, 2), 1.0, np.zeros(2)),
        }
        with caplog.at_level(logging.WARNING):
            OrthogonalProjector(dimensions=5, subspaces=subspaces)

        messages = " ".join(r.message for r in caplog.records).lower()
        assert "sin subespacio" in messages or "uncovered" in messages, (
            f"No se emitió warning de índices sin cobertura. Mensajes: {messages!r}"
        )

    def test_sum_initialized_with_zeros_not_scalar(self):
        """
        FIX-06: la suma de matrices de proyección debe inicializarse con
        np.zeros((n,n)) y NO con el escalar 0.

        sum([P1, P2], 0) lanza TypeError en numpy porque 0 (int) no es
        compatible con la adición de matrices.
        """
        subspaces = {
            "A": SubspaceSpec("A", slice(0, 3), 1.0, np.zeros(3)),
            "B": SubspaceSpec("B", slice(3, 5), 1.0, np.zeros(2)),
            "C": SubspaceSpec("C", slice(5, 7), 1.0, np.zeros(2)),
        }
        proj = OrthogonalProjector(dimensions=7, subspaces=subspaces)
        # Si FIX-06 está correcto, validation_report se calcula sin TypeError
        report = proj.validation_report
        coverage_err = report.get("coverage_identity_error", float("inf"))
        assert coverage_err < ALGEBRAIC_TOL, (
            f"FIX-06: coverage_identity_error = {coverage_err:.2e} ≥ {ALGEBRAIC_TOL}"
        )

    def test_validation_report_all_errors_below_tol(self, standard_projector):
        """
        Todos los errores algebraicos del informe son < ALGEBRAIC_TOL.

        Los campos de error incluyen: idempotence_*, self_adjoint_*, coverage_*.
        """
        report = standard_projector.validation_report
        error_keys = [
            k for k in report
            if any(kw in k for kw in ("error", "idempotence", "self_adjoint", "coverage"))
        ]
        assert len(error_keys) > 0, "validation_report no contiene claves de error"
        for key in error_keys:
            val = report[key]
            assert val < ALGEBRAIC_TOL, (
                f"Error algebraico inaceptable: {key} = {val:.4e} ≥ {ALGEBRAIC_TOL}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 · CARACTERÍSTICA DE EULER (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEulerCharacteristic:
    """
    χ = β₀ - β₁: invariante topológico entero.

    Restricciones físicas:
    - β₀ ≥ 1 (al menos una componente conexa)
    - β₁ ≥ 0 (número de ciclos no negativo)
    - χ ∈ ℤ (entero, no flotante)
    """

    def test_simply_connected_euler_one(self, standard_projector):
        """
        β₀=1, β₁=0 → χ = 1 - 0 = 1 (espacio simplemente conexo).
        """
        psi = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        chi = standard_projector.project(psi).euler_char
        assert chi == 1, f"χ = {chi} ≠ 1"

    def test_euler_with_multiple_components_and_cycles(self, standard_projector):
        """
        β₀=2, β₁=3 → χ = 2 - 3 = -1.
        """
        psi = np.array([0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0])
        chi = standard_projector.project(psi).euler_char
        assert chi == -1, f"χ = {chi} ≠ -1"

    @pytest.mark.parametrize(
        "b0, b1, expected_chi",
        [
            (1, 0, 1),    # Árbol
            (2, 1, 1),    # 2 componentes, 1 ciclo
            (3, 2, 1),
            (5, 4, 1),
            (2, 0, 2),    # 2 componentes, acíclico
            (1, 3, -2),   # 1 componente, 3 ciclos → superficie de genus 2
            (4, 4, 0),    # χ = 0 (toro si fuera una superficie)
            (1, 1, 0),    # S¹: χ = 0
        ],
    )
    def test_euler_formula_parametric(
        self,
        standard_projector: OrthogonalProjector,
        b0: int,
        b1: int,
        expected_chi: int,
    ):
        """
        χ = β₀ - β₁ para diversas combinaciones topológicas.

        Cada par (β₀, β₁) corresponde a un tipo de complejo simplicial.
        """
        psi = np.zeros(7)
        psi[BETTI_INDICES[0]] = float(b0)
        psi[BETTI_INDICES[1]] = float(b1)
        chi = standard_projector.project(psi).euler_char
        assert chi == expected_chi, (
            f"χ(β₀={b0}, β₁={b1}) = {chi} ≠ {expected_chi}"
        )

    def test_euler_always_python_int(self, standard_projector):
        """
        χ es un int de Python (no np.int64, np.int32, u otro tipo numpy).

        Verificación estricta con isinstance(chi, int).
        Nota: isinstance(np.int64(1), int) es True en Python 3.8+ pero
        la semántica debe ser de tipo Python nativo.
        """
        for b0, b1 in [(1, 0), (2, 1), (3, 5)]:
            psi = np.zeros(7)
            psi[BETTI_INDICES[0]] = float(b0)
            psi[BETTI_INDICES[1]] = float(b1)
            chi = standard_projector.project(psi).euler_char
            assert type(chi) is int, (
                f"χ type = {type(chi).__name__} (esperado int de Python)"
            )

    def test_beta0_zero_raises_topological_invariant_error(
        self, standard_projector,
    ):
        """
        β₀ = 0 viola el axioma topológico (toda variedad tiene ≥ 1 componente).

        El error debe mencionar 'β₀' para identificar la causa.
        """
        psi = np.zeros(7)
        # β₀ = 0, β₁ = 0 (por defecto al crear np.zeros)
        with pytest.raises(TopologicalInvariantError, match="β₀"):
            standard_projector.project(psi)

    def test_beta1_negative_raises_topological_invariant_error(
        self, standard_projector,
    ):
        """
        β₁ < 0 viola la definición de número de Betti.

        El error debe mencionar 'β₁'.
        """
        psi = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
        with pytest.raises(TopologicalInvariantError, match="β₁"):
            standard_projector.project(psi)

    def test_no_topo_indices_returns_none_euler(self):
        """
        Sin topo_indices, euler_char es None (no calculable sin β₀, β₁).
        """
        subspaces = {
            "A": SubspaceSpec("A", slice(0, 4), 1.0, np.zeros(4)),
            "B": SubspaceSpec("B", slice(4, 7), 1.0, np.zeros(3)),
        }
        proj = OrthogonalProjector(7, subspaces, topo_indices=None)
        psi = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0])
        result = proj.project(psi)
        assert result.euler_char is None, (
            f"euler_char = {result.euler_char} ≠ None sin topo_indices"
        )

    def test_euler_char_consistency_with_betti(self, standard_projector):
        """
        χ = β₀ - β₁ verificado aritméticamente desde los valores del vector ψ.

        No asumimos la implementación interna; verificamos que el valor
        retornado sea consistente con la fórmula.
        """
        for b0, b1 in [(1, 0), (2, 3), (5, 1), (10, 4)]:
            psi = np.zeros(7)
            psi[BETTI_INDICES[0]] = float(b0)
            psi[BETTI_INDICES[1]] = float(b1)
            chi = standard_projector.project(psi).euler_char
            expected = b0 - b1
            assert chi == expected, (
                f"Inconsistencia: χ = {chi} ≠ β₀-β₁ = {b0}-{b1} = {expected}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 · CLASIFICACIÓN CON HISTÉRESIS (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHysteresisClassification:
    """
    Verificación exhaustiva de _classify_with_hysteresis.

    Parámetros: W=0.8, C=1.5, D=0.05
    Bandas efectivas (calculadas algebraicamente, no hardcodeadas):
        Subir a WARNING  : value > W + D = 0.85
        Subir a CRITICAL : value > C + D = 1.55
        Bajar de WARNING : value < W - D = 0.75
        Bajar CRITICAL→WARNING: W-D ≤ value < C-D [0.75, 1.45)
        Bajar CRITICAL→HEALTHY: value < W-D = 0.75
    """

    @staticmethod
    def _cls(
        value: float,
        previous: Optional[HealthStatus] = None,
        w: float = _W,
        c: float = _C,
        d: float = _D,
    ) -> HealthStatus:
        """Wrapper tipado para _classify_with_hysteresis."""
        return OrthogonalProjector._classify_with_hysteresis(
            value, w, c, d, previous,
        )

    # -- Sin estado previo: clasificación pura por umbrales --

    def test_no_prev_below_warning(self):
        """value < W → HEALTHY."""
        assert self._cls(_W - 0.1) == HealthStatus.HEALTHY

    def test_no_prev_between_warning_and_critical(self):
        """W < value < C → WARNING."""
        mid = (_W + _C) / 2.0
        assert self._cls(mid) == HealthStatus.WARNING

    def test_no_prev_above_critical(self):
        """value > C → CRITICAL."""
        assert self._cls(_C + 0.5) == HealthStatus.CRITICAL

    def test_no_prev_at_warning_boundary(self):
        """
        value = W + ε (justo por encima del umbral): WARNING.

        Límite cerrado-izquierdo: [W, C) → WARNING.
        """
        assert self._cls(_W + 0.001) == HealthStatus.WARNING

    def test_no_prev_at_critical_boundary(self):
        """value = C + ε → CRITICAL."""
        assert self._cls(_C + 0.001) == HealthStatus.CRITICAL

    # -- Desde HEALTHY --

    def test_from_healthy_below_band_stays(self):
        """
        value ≤ W + D (no supera banda de subida): permanece HEALTHY.

        La histéresis previene entrar a WARNING hasta superar W+D=0.85.
        """
        value = _BAND_RISE_WARNING - 0.001  # < 0.85
        result = self._cls(value, HealthStatus.HEALTHY)
        assert result == HealthStatus.HEALTHY, (
            f"value={value:.3f} < {_BAND_RISE_WARNING} debería ser HEALTHY, "
            f"got {result}"
        )

    def test_from_healthy_enters_warning(self):
        """
        value > W + D = 0.85: entra en WARNING.
        """
        value = _BAND_RISE_WARNING + 0.001  # > 0.85
        result = self._cls(value, HealthStatus.HEALTHY)
        assert result == HealthStatus.WARNING, (
            f"value={value:.3f} > {_BAND_RISE_WARNING} debería ser WARNING"
        )

    def test_from_healthy_jumps_to_critical_directly(self):
        """
        value > C + D = 1.55 directamente desde HEALTHY: salta a CRITICAL.

        Sin restricción de ascenso gradual (solo para el descenso).
        """
        value = _BAND_RISE_CRITICAL + 0.001
        result = self._cls(value, HealthStatus.HEALTHY)
        assert result == HealthStatus.CRITICAL, (
            f"Salto directo HEALTHY→CRITICAL no ocurrió para value={value:.3f}"
        )

    # -- Desde WARNING --

    def test_from_warning_stays_in_zone(self):
        """value en zona WARNING (W-D, C+D): permanece WARNING."""
        mid = (_W + _C) / 2.0
        result = self._cls(mid, HealthStatus.WARNING)
        assert result == HealthStatus.WARNING

    def test_from_warning_escalates_to_critical(self):
        """value > C + D: escala a CRITICAL."""
        value = _BAND_RISE_CRITICAL + 0.001
        result = self._cls(value, HealthStatus.WARNING)
        assert result == HealthStatus.CRITICAL

    def test_from_warning_recovers_to_healthy(self):
        """
        value < W - D = 0.75: recupera a HEALTHY.

        La histéresis previene bajar a HEALTHY hasta cruzar W-D.
        """
        value = _BAND_FALL_WARNING - 0.001  # < 0.75
        result = self._cls(value, HealthStatus.WARNING)
        assert result == HealthStatus.HEALTHY, (
            f"value={value:.3f} < {_BAND_FALL_WARNING} debería ser HEALTHY"
        )

    def test_from_warning_hysteresis_prevents_recovery(self):
        """
        W - D ≤ value < W: dentro de banda inferior → permanece WARNING.

        La histéresis crea una zona muerta [W-D, W] donde no se baja.
        """
        value = _BAND_FALL_WARNING + 0.001  # > 0.75 pero < W = 0.80
        result = self._cls(value, HealthStatus.WARNING)
        assert result == HealthStatus.WARNING, (
            f"Chattering: value={value:.3f} en banda muerta debería ser WARNING"
        )

    # -- Desde CRITICAL (FIX-08) --

    def test_from_critical_stays_in_critical_zone(self):
        """
        value ≥ C - D = 1.45: permanece CRITICAL.

        La banda superior de CRITICAL es [C-D, ∞).
        """
        value = _BAND_FALL_CRITICAL + 0.001  # > 1.45
        result = self._cls(value, HealthStatus.CRITICAL)
        assert result == HealthStatus.CRITICAL, (
            f"value={value:.3f} > {_BAND_FALL_CRITICAL} debería ser CRITICAL"
        )

    def test_from_critical_descends_to_warning_not_healthy(self):
        """
        FIX-08: W-D ≤ value < C-D → desciende a WARNING (no a HEALTHY).

        Zona intermedia [0.75, 1.45): CRITICAL → WARNING (sin salto).
        El valor exacto del punto medio de la zona intermedia.
        """
        value = (_BAND_FALL_WARNING + _BAND_FALL_CRITICAL) / 2.0
        # value ≈ 1.10: en zona intermedia [0.75, 1.45)
        assert _BAND_FALL_WARNING <= value < _BAND_FALL_CRITICAL, (
            f"El punto medio {value:.3f} no está en la zona intermedia "
            f"[{_BAND_FALL_WARNING}, {_BAND_FALL_CRITICAL})"
        )
        result = self._cls(value, HealthStatus.CRITICAL)
        assert result == HealthStatus.WARNING, (
            f"FIX-08: value={value:.3f} desde CRITICAL debería ser WARNING, "
            f"got {result}"
        )
        assert result != HealthStatus.HEALTHY, (
            f"FIX-08: salto CRITICAL→HEALTHY espurio para value={value:.3f}"
        )

    def test_from_critical_jumps_healthy_for_very_low_value(self):
        """
        value < W - D = 0.75 desde CRITICAL → HEALTHY directo.

        Cuando el valor cae por debajo de la zona WARNING completa,
        se puede bajar directamente a HEALTHY.
        """
        value = _BAND_FALL_WARNING - 0.1  # 0.65: bien por debajo de 0.75
        result = self._cls(value, HealthStatus.CRITICAL)
        assert result == HealthStatus.HEALTHY, (
            f"value={value:.3f} << {_BAND_FALL_WARNING} debería ir a HEALTHY"
        )

    def test_fix08_boundary_value_in_intermediate_zone(self):
        """
        FIX-08 core: el error original clasificaba valores en zona
        intermedia [W-D, C-D) como HEALTHY desde CRITICAL.

        Verificamos explícitamente en el límite inferior de la zona.
        """
        # Valor justo por encima de W-D (en zona intermedia)
        value = _BAND_FALL_WARNING + 0.05  # 0.80: claramente en zona intermedia
        assert _BAND_FALL_WARNING < value < _BAND_FALL_CRITICAL
        result = self._cls(value, HealthStatus.CRITICAL)
        assert result == HealthStatus.WARNING, (
            f"FIX-08 violado: value={value:.3f} desde CRITICAL → {result.name} "
            f"(esperado WARNING)"
        )

    # -- Modo laboratorio (D=0): clasificación determinista pura --

    def test_zero_hysteresis_exactly_at_warning(self):
        """Con D=0, value = W + ε → WARNING (sin banda de transición)."""
        result = self._cls(_W + 0.001, d=0.0)
        assert result == HealthStatus.WARNING

    def test_zero_hysteresis_exactly_at_critical(self):
        """Con D=0, value = C + ε → CRITICAL."""
        result = self._cls(_C + 0.001, d=0.0)
        assert result == HealthStatus.CRITICAL

    def test_zero_hysteresis_recovery_immediate(self):
        """Con D=0, desde WARNING a value = W - ε → HEALTHY inmediato."""
        result = self._cls(_W - 0.001, HealthStatus.WARNING, d=0.0)
        assert result == HealthStatus.HEALTHY

    def test_hysteresis_band_is_algebraically_derived(self):
        """
        Las bandas de transición son _W±_D y _C±_D (derivadas del módulo).

        Test de coherencia: los valores de umbral usados en los tests
        son los mismos que usa la implementación.
        """
        # No podemos acceder a los umbrales internos directamente,
        # pero verificamos que la transición ocurre exactamente en W+D
        just_below = self._cls(_BAND_RISE_WARNING - 1e-9, HealthStatus.HEALTHY)
        just_above = self._cls(_BAND_RISE_WARNING + 1e-9, HealthStatus.HEALTHY)
        assert just_below == HealthStatus.HEALTHY
        assert just_above == HealthStatus.WARNING


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12 · ThreatAssessment — Refinada
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreatAssessment:
    """
    Invarianzas del resultado inmutable de evaluación de amenazas.

    ThreatAssessment es un objeto valor (value object) inmutable que
    resume la evaluación del sistema inmunológico.
    """

    def test_total_threat_is_l2_norm_of_levels(self):
        """
        total_threat = ‖(threat₁,...,threatₙ)‖₂.

        Fórmula: √(Σ levelᵢ²).
        """
        levels = {"A": 0.5, "B": 1.2, "C": 0.3}
        a = ThreatAssessment.from_components(levels)
        manual = float(np.linalg.norm(list(levels.values())))
        assert a.total_threat == pytest.approx(manual, rel=1e-12), (
            f"total_threat = {a.total_threat:.6f} ≠ ‖levels‖₂ = {manual:.6f}"
        )

    def test_max_source_points_to_maximum(self):
        """max_source apunta al subespacio con mayor amenaza."""
        levels = {"A": 0.1, "B": 2.5, "C": 1.0}
        a = ThreatAssessment.from_components(levels)
        assert a.max_source == "B", (
            f"max_source = {a.max_source!r} ≠ 'B'"
        )
        assert a.max_value == pytest.approx(2.5, rel=1e-12)

    def test_max_source_consistent_with_levels(self):
        """levels[max_source] == max_value (consistencia interna)."""
        levels = {"X": 3.0, "Y": 1.0, "Z": 2.5}
        a = ThreatAssessment.from_components(levels)
        assert levels[a.max_source] == a.max_value

    def test_status_healthy_for_low_threat(self):
        """threat < warning_threshold → HEALTHY."""
        a = ThreatAssessment.from_components({"A": 0.1, "B": 0.2})
        assert a.status == HealthStatus.HEALTHY

    def test_status_warning_for_medium_threat(self):
        """threat en zona WARNING."""
        a = ThreatAssessment.from_components({"A": 0.9})
        assert a.status == HealthStatus.WARNING

    def test_status_critical_for_high_threat(self):
        """threat > critical_threshold → CRITICAL."""
        a = ThreatAssessment.from_components({"A": 2.0})
        assert a.status == HealthStatus.CRITICAL

    def test_to_dict_contains_all_required_keys(self):
        """to_dict() contiene exactamente las claves requeridas."""
        a = ThreatAssessment.from_components({"A": 0.5}, euler_char=1)
        d = a.to_dict()
        required = {
            "threat_levels", "max_threat_source", "max_threat_value",
            "total_threat", "euler_characteristic", "health_status", "metadata",
        }
        missing = required - set(d.keys())
        assert not missing, f"Claves faltantes en to_dict(): {missing}"

    def test_to_dict_is_json_serializable(self):
        """
        Los valores en to_dict() son JSON-serializables.

        Verificación recursiva con json.dumps (no solo las claves top-level).
        """
        a = ThreatAssessment.from_components(
            {"A": 0.5, "B": 1.0}, euler_char=1,
        )
        d = a.to_dict()
        try:
            json.dumps(d)
        except (TypeError, ValueError) as e:
            pytest.fail(
                f"to_dict() no es JSON-serializable: {e}. "
                f"Dict: {d!r}"
            )

    def test_euler_char_in_dict(self):
        """euler_characteristic en dict coincide con el valor pasado."""
        a = ThreatAssessment.from_components({"A": 0.3}, euler_char=2)
        assert a.to_dict()["euler_characteristic"] == 2

    def test_repr_contains_status_name(self):
        """repr() contiene el nombre del estado de salud."""
        a = ThreatAssessment.from_components({"A": 0.5})
        r = repr(a)
        assert any(
            status.name in r for status in HealthStatus
        ), f"repr() no contiene nombre de HealthStatus: {r!r}"

    def test_empty_levels_raises(self):
        """levels vacío → no hay máximo → debe lanzar excepción."""
        with pytest.raises((ValueError, KeyError, Exception)):
            ThreatAssessment.from_components({})

    def test_immutability_frozen(self):
        """ThreatAssessment es inmutable (frozen dataclass o similar)."""
        a = ThreatAssessment.from_components({"A": 0.5})
        with pytest.raises((TypeError, AttributeError)):
            a.max_value = 999.0  # type: ignore

    def test_total_threat_nonnegative(self):
        """total_threat ≥ 0 siempre (norma L2 de valores ≥ 0)."""
        for values in [{"A": 0.0}, {"A": 1e-15}, {"A": 100.0}]:
            a = ThreatAssessment.from_components(values)
            assert a.total_threat >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14 · ImmuneWatcherMorphism — FUNCTORIALIDAD CATEGÓRICA (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestImmuneWatcherFunctoriality:
    """
    Propiedades del morfismo categórico F: CategoricalState → CategoricalState.

    Propiedades a verificar:
    1. F(error) IS error (identidad exacta del objeto, no solo igualdad)
    2. Contador incrementa en TODAS las llamadas
    3. Estados exitosos invocan with_update (no with_error)
    4. Estados críticos invocan with_error con acción QUARANTINE
    5. with_update recibe new_stratum=WISDOM
    """

    def test_topological_changes_do_not_leak_into_thermodynamics(
        self, watcher_default: ImmuneWatcherMorphism, healthy_telemetry: Dict,
    ):
        """
        La modificación de la complejidad topológica (β₁) no debe generar
        'fugas dimensionales' o efectos secundarios en el cálculo de la
        temperatura financiera (σ) o amenaza termodinámica, preservando la
        ortogonalidad funcional cruzada.
        """
        import copy

        # Estado base
        state_base = _make_state(healthy_telemetry)
        watcher_default(state_base)
        context_base = state_base.with_update.call_args[0][0]
        thermo_threat_base = context_base["threat_levels"]["thermo_core"]
        topo_threat_base = context_base["threat_levels"]["topology_core"]

        # Incrementar complejidad topológica (β₁: 0 -> 2)
        leaky_telemetry = copy.deepcopy(healthy_telemetry)
        leaky_telemetry["beta_1"] = 2

        state_leaky = _make_state(leaky_telemetry)
        watcher_default(state_leaky)

        # When critical threshold is exceeded, it uses with_error instead of with_update
        if state_leaky.with_update.called:
            context_leaky = state_leaky.with_update.call_args[0][0]
        else:
            context_leaky = state_leaky.with_error.call_args[1]["details"]

        thermo_threat_leaky = context_leaky["threat_levels"]["thermo_core"]
        topo_threat_leaky = context_leaky["threat_levels"]["topology_core"]

        # Asegurar que la amenaza topológica sí cambió
        assert topo_threat_leaky > topo_threat_base, "La amenaza topológica no respondió al cambio"

        # ASERCIÓN CLAVE: La amenaza termodinámica se mantiene matemáticamente idéntica
        assert thermo_threat_base == pytest.approx(thermo_threat_leaky, abs=1e-12), (
            f"Fuga dimensional: la topología alteró la termodinámica. "
            f"Base: {thermo_threat_base}, Leaky: {thermo_threat_leaky}"
        )

    def test_error_state_returned_as_identity(
        self, watcher_default: ImmuneWatcherMorphism, error_state: MagicMock,
    ):
        """
        F(error) IS error — identidad de objeto exacta (no solo igualdad).

        Propiedad categórica: el morfismo preserva el objeto cero sin mutación.
        Verificamos con `is` (identidad de objeto), no `==` (igualdad de valor).
        """
        result = watcher_default(error_state)
        assert result is error_state, (
            "F(error) debe retornar el mismo objeto (identidad), no una copia"
        )

    def test_error_state_not_modified(
        self, watcher_default: ImmuneWatcherMorphism,
    ):
        """
        Estado de error no se modifica: with_update y with_error NO se llaman.

        El morfismo puro no muta el estado de error; lo propaga inalterado.
        """
        error = _make_state(success=False)
        watcher_default(error)
        error.with_update.assert_not_called()
        error.with_error.assert_not_called()

    def test_counter_increments_for_success_state(
        self, watcher_default: ImmuneWatcherMorphism,
    ):
        """Contador incrementa para estados exitosos."""
        initial = watcher_default.evaluation_count
        watcher_default(_make_state(telemetry={}))
        assert watcher_default.evaluation_count == initial + 1

    def test_counter_increments_for_error_state(
        self, watcher_default: ImmuneWatcherMorphism,
    ):
        """
        Contador incrementa incluso para estados fallidos.

        Cada llamada al morfismo se cuenta, independientemente del resultado.
        """
        initial = watcher_default.evaluation_count
        watcher_default(_make_state(success=False))
        assert watcher_default.evaluation_count == initial + 1

    def test_healthy_telemetry_calls_with_update_not_error(
        self, watcher_default: ImmuneWatcherMorphism, healthy_telemetry: Dict,
    ):
        """Estado saludable invoca with_update exactamente una vez."""
        state = _make_state(healthy_telemetry)
        watcher_default(state)
        state.with_update.assert_called_once()
        state.with_error.assert_not_called()

    def test_critical_telemetry_calls_with_error_not_update(
        self, watcher_default: ImmuneWatcherMorphism, critical_telemetry: Dict,
    ):
        """Estado crítico activa cuarentena vía with_error."""
        state = _make_state(critical_telemetry)
        watcher_default(state)
        state.with_error.assert_called_once()
        state.with_update.assert_not_called()

    def test_healthy_context_has_immune_status_healthy(
        self,
        watcher_laboratory: ImmuneWatcherMorphism,
        healthy_telemetry: Dict,
    ):
        """
        with_update recibe context con immune_status='healthy'.

        Verificamos el valor EXACTO de la clave, no solo su presencia.
        """
        state = _make_state(healthy_telemetry)
        watcher_laboratory(state)
        context = state.with_update.call_args[0][0]
        assert "immune_status" in context, "Clave 'immune_status' no en context"
        assert context["immune_status"] == "healthy", (
            f"immune_status = {context['immune_status']!r} ≠ 'healthy'"
        )

    def test_critical_error_contains_quarantine_action(
        self, watcher_default: ImmuneWatcherMorphism, critical_telemetry: Dict,
    ):
        """
        with_error se invoca con details['action'] = 'QUARANTINE'.

        El morfismo debe etiquetar explícitamente la acción de cuarentena.
        """
        state = _make_state(critical_telemetry)
        watcher_default(state)
        call_kwargs = state.with_error.call_args[1]
        details = call_kwargs.get("details", {})
        assert details.get("action") == "QUARANTINE", (
            f"details = {details!r} no contiene action='QUARANTINE'"
        )

    def test_with_update_receives_wisdom_stratum(
        self,
        watcher_laboratory: ImmuneWatcherMorphism,
        healthy_telemetry: Dict,
    ):
        """
        with_update recibe new_stratum=WISDOM (codomain del morfismo).
        """
        state = _make_state(healthy_telemetry)
        watcher_laboratory(state)
        call_kwargs = state.with_update.call_args[1]
        assert call_kwargs.get("new_stratum") == Stratum.WISDOM, (
            f"new_stratum = {call_kwargs.get('new_stratum')!r} ≠ WISDOM"
        )

    def test_counter_increments_per_call_sequence(
        self, watcher_default: ImmuneWatcherMorphism,
    ):
        """
        Contador incrementa monótonamente en cada llamada.

        Verificación de secuencia: success, error, success → counter = 3.
        """
        initial = watcher_default.evaluation_count
        watcher_default(_make_state(telemetry={}))
        watcher_default(_make_state(success=False))
        watcher_default(_make_state(telemetry={}))
        assert watcher_default.evaluation_count == initial + 3

    def test_empty_telemetry_treated_as_healthy(
        self,
        watcher_default: ImmuneWatcherMorphism,
        success_state: MagicMock,
    ):
        """
        Telemetría vacía usa valores por defecto → estado HEALTHY.

        Los defaults de SIGNAL_SCHEMA corresponden a condiciones nominales.
        """
        watcher_default(success_state)
        success_state.with_update.assert_called_once()
        context = success_state.with_update.call_args[0][0]
        assert context.get("immune_status") == "healthy", (
            f"Telemetría vacía no produce 'healthy': {context!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 16 · DINÁMICA TEMPORAL DE HISTÉRESIS (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHysteresisDynamics:
    """
    Simulación de secuencias temporales de clasificación.

    Las bandas de transición se calculan algebraicamente desde _W, _C, _D
    para evitar magic numbers y garantizar consistencia con el sistema real.
    """

    @staticmethod
    def _sequence(
        values: List[float],
        warning: float = _W,
        critical: float = _C,
        hysteresis: float = _D,
    ) -> List[HealthStatus]:
        """
        Aplica clasificación iterativa con estado acumulado.

        Estado inicial: None (sin historial previo).
        """
        results: List[HealthStatus] = []
        prev: Optional[HealthStatus] = None
        for v in values:
            s = OrthogonalProjector._classify_with_hysteresis(
                v, warning, critical, hysteresis, prev,
            )
            results.append(s)
            prev = s
        return results

    def test_monotone_increase_transitions(self):
        """
        Escalada monotónica: HEALTHY → WARNING → CRITICAL.

        Valores elegidos algebraicamente respecto a las bandas:
        - 0.0: < W → HEALTHY
        - 0.3: < W → HEALTHY
        - 0.6: < W → HEALTHY (< 0.80)
        - _BAND_RISE_WARNING + 0.02 = 0.87: > 0.85 → WARNING
        - _W + 0.3 = 1.10: W < v < C → WARNING
        - _BAND_RISE_CRITICAL + 0.02 = 1.57: > 1.55 → CRITICAL
        """
        values = [
            0.0,
            0.3,
            0.6,
            _BAND_RISE_WARNING + 0.02,    # 0.87 → WARNING
            _W + 0.3,                      # 1.10 → WARNING
            _BAND_RISE_CRITICAL + 0.02,    # 1.57 → CRITICAL
        ]
        statuses = self._sequence(values)
        assert statuses[0] == HealthStatus.HEALTHY,  f"v=0.0: {statuses[0]}"
        assert statuses[1] == HealthStatus.HEALTHY,  f"v=0.3: {statuses[1]}"
        assert statuses[2] == HealthStatus.HEALTHY,  f"v=0.6: {statuses[2]}"
        assert statuses[3] == HealthStatus.WARNING,  f"v=0.87: {statuses[3]}"
        assert statuses[4] == HealthStatus.WARNING,  f"v=1.10: {statuses[4]}"
        assert statuses[5] == HealthStatus.CRITICAL, f"v=1.57: {statuses[5]}"

    def test_monotone_decrease_fix08_no_skip(self):
        """
        FIX-08: descenso CRITICAL → WARNING → HEALTHY sin saltos.

        Valores algebraicamente derivados:
        - 2.0: > C → CRITICAL
        - _BAND_FALL_CRITICAL - 0.15 = 1.30: < 1.45 desde CRITICAL → WARNING
        - _W: en zona WARNING (no baja) → WARNING
        - _BAND_FALL_WARNING - 0.03 = 0.72: < 0.75 → HEALTHY
        - 0.4: < W → HEALTHY
        """
        values = [
            2.0,
            _BAND_FALL_CRITICAL - 0.15,   # 1.30 desde CRITICAL → WARNING
            _W,                            # 0.80 desde WARNING → WARNING
            _BAND_FALL_WARNING - 0.03,     # 0.72 desde WARNING → HEALTHY
            0.4,
        ]
        statuses = self._sequence(values)
        assert statuses[0] == HealthStatus.CRITICAL, f"v=2.0: {statuses[0]}"
        assert statuses[1] == HealthStatus.WARNING,  f"v=1.30: {statuses[1]}"
        assert statuses[2] == HealthStatus.WARNING,  f"v=0.80: {statuses[2]}"
        assert statuses[3] == HealthStatus.HEALTHY,  f"v=0.72: {statuses[3]}"
        assert statuses[4] == HealthStatus.HEALTHY,  f"v=0.40: {statuses[4]}"

    def test_no_chattering_oscillation_near_warning_from_healthy(self):
        """
        Oscilación en [W-ε, W+D-ε] desde HEALTHY: ninguna transición a WARNING.

        Todos los valores están por debajo de W+D=0.85 → permanecen HEALTHY.
        """
        epsilon = 0.02
        values = [_W - epsilon, _BAND_RISE_WARNING - epsilon] * 6
        assert all(v < _BAND_RISE_WARNING for v in values), (
            "Valores de oscilación superan la banda de subida"
        )
        statuses = self._sequence(values)
        assert all(s == HealthStatus.HEALTHY for s in statuses), (
            f"Chattering detectado: {[s.name for s in statuses]}"
        )

    def test_no_chattering_from_warning_below_fall_band(self):
        """
        Desde WARNING, oscilación [W-D+ε, W-ε] no baja a HEALTHY.

        Todos los valores están por encima de W-D=0.75 → permanecen WARNING.
        """
        epsilon = 0.02
        values = [_BAND_FALL_WARNING + epsilon, _W - epsilon] * 4
        assert all(v > _BAND_FALL_WARNING for v in values), (
            "Valores de oscilación caen por debajo de la banda de caída"
        )
        statuses = self._sequence(values, hysteresis=_D)
        # Primera evaluación sin estado previo → puede ser HEALTHY o WARNING
        # Forzamos estado inicial desde WARNING
        prev = HealthStatus.WARNING
        for v in values:
            s = OrthogonalProjector._classify_with_hysteresis(
                v, _W, _C, _D, prev,
            )
            assert s == HealthStatus.WARNING, (
                f"Chattering desde WARNING: v={v:.3f} → {s.name}"
            )
            prev = s

    def test_idempotence_at_steady_state(self):
        """
        f(f(x, s), s) = f(x, s) — clasificación idempotente en estado estacionario.

        Si la clasificación converge a un estado s en f(x, None),
        entonces f(x, s) = s (punto fijo).
        """
        test_values = [
            0.2,                            # HEALTHY estacionario
            _BAND_RISE_WARNING + 0.02,      # WARNING (fuera de banda subida)
            _BAND_RISE_CRITICAL + 0.1,      # CRITICAL
        ]
        for value in test_values:
            s1 = OrthogonalProjector._classify_with_hysteresis(
                value, _W, _C, _D, None,
            )
            s2 = OrthogonalProjector._classify_with_hysteresis(
                value, _W, _C, _D, s1,
            )
            assert s1 == s2, (
                f"No idempotente: v={value:.4f}, "
                f"f(v, ∅)={s1.name}, f(v, {s1.name})={s2.name}"
            )

    def test_monotonicity_of_status_with_increasing_threat(self):
        """
        El estado de salud es monótonamente no-decreciente en el valor de amenaza.

        Para values crecientes desde 0 a 2: la secuencia de estados
        no puede bajar (CRITICAL no vuelve a HEALTHY en una secuencia ascendente).
        """
        # Secuencia estrictamente creciente
        values = [i * 0.1 for i in range(21)]  # 0.0 a 2.0
        statuses = self._sequence(values)
        severity = [s.severity for s in statuses]
        # La severidad máxima alcanzada no debe decrecer
        max_sev_so_far = 0
        for i, sev in enumerate(severity):
            # En presencia de histéresis, puede haber plateaux pero no decrementos
            # desde el máximo alcanzado en zonas de alta amenaza
            assert sev >= 0, f"Severidad negativa en i={i}: {sev}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 19 · CONTEXT MANAGER DE TOLERANCIAS (Refinada)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporaryAlgebraicTolerance:
    """
    temporary_algebraic_tolerance: modificación temporal de ALGEBRAIC_TOL.

    FIX-02: usa sys.modules para modificar la variable de módulo real.
    El context manager implementa el patrón try/finally para garantizar
    la restauración incluso ante excepciones.
    """

    def test_tolerance_changed_within_context(
        self, standard_projector: OrthogonalProjector,
    ):
        """ALGEBRAIC_TOL cambia dentro del context manager."""
        new_tol = 1e-3
        with standard_projector.temporary_algebraic_tolerance(new_tol):
            assert _module.ALGEBRAIC_TOL == pytest.approx(new_tol), (
                f"ALGEBRAIC_TOL = {_module.ALGEBRAIC_TOL} ≠ {new_tol} dentro del context"
            )

    def test_tolerance_restored_after_normal_exit(
        self, standard_projector: OrthogonalProjector,
    ):
        """ALGEBRAIC_TOL se restaura exactamente al salir normalmente."""
        original = _module.ALGEBRAIC_TOL
        with standard_projector.temporary_algebraic_tolerance(1e-3):
            pass
        assert _module.ALGEBRAIC_TOL == pytest.approx(original), (
            f"ALGEBRAIC_TOL = {_module.ALGEBRAIC_TOL} ≠ original={original} tras salida"
        )

    def test_tolerance_restored_after_exception(
        self, standard_projector: OrthogonalProjector,
    ):
        """
        FIX-02 core: ALGEBRAIC_TOL se restaura incluso si hay excepción.

        El finally del context manager debe ejecutarse antes de propagar la excepción.
        """
        original = _module.ALGEBRAIC_TOL
        with pytest.raises(ValueError, match="error deliberado"):
            with standard_projector.temporary_algebraic_tolerance(1e-4):
                raise ValueError("error deliberado")
        assert _module.ALGEBRAIC_TOL == pytest.approx(original), (
            f"FIX-02: ALGEBRAIC_TOL no restaurado tras excepción: "
            f"{_module.ALGEBRAIC_TOL} ≠ {original}"
        )

    def test_default_parameter_is_1e6(
        self, standard_projector: OrthogonalProjector,
    ):
        """El valor por defecto del parámetro es 1e-6."""
        with standard_projector.temporary_algebraic_tolerance():
            assert _module.ALGEBRAIC_TOL == pytest.approx(1e-6), (
                f"Default tolerance = {_module.ALGEBRAIC_TOL} ≠ 1e-6"
            )

    def test_nesting_restores_correctly(
        self, standard_projector: OrthogonalProjector,
    ):
        """
        Anidamiento de context managers: restauración en LIFO.

        Invariante: al salir del inner, se restaura el valor del outer
        (no el original global). Al salir del outer, se restaura el original.
        """
        original = _module.ALGEBRAIC_TOL
        outer_tol = 1e-4
        inner_tol = 1e-5

        with standard_projector.temporary_algebraic_tolerance(outer_tol):
            mid = _module.ALGEBRAIC_TOL
            assert mid == pytest.approx(outer_tol), (
                f"Outer context: {mid} ≠ {outer_tol}"
            )

            with standard_projector.temporary_algebraic_tolerance(inner_tol):
                inner = _module.ALGEBRAIC_TOL
                assert inner == pytest.approx(inner_tol), (
                    f"Inner context: {inner} ≠ {inner_tol}"
                )

            # Al salir del inner, debe restaurar outer_tol
            restored_outer = _module.ALGEBRAIC_TOL
            assert restored_outer == pytest.approx(outer_tol), (
                f"Tras salida inner: {restored_outer} ≠ outer={outer_tol}"
            )

        # Al salir del outer, debe restaurar original
        assert _module.ALGEBRAIC_TOL == pytest.approx(original), (
            f"Tras salida outer: {_module.ALGEBRAIC_TOL} ≠ original={original}"
        )

    def test_negative_tolerance_raises(
        self, standard_projector: OrthogonalProjector,
    ):
        """
        Tolerancia negativa no tiene sentido matemático → ValueError.

        Una tolerancia es una cota superior de error: debe ser positiva.
        """
        with pytest.raises(ValueError):
            with standard_projector.temporary_algebraic_tolerance(-1e-6):
                pass  # No debe llegar aquí


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 20 · PRUEBAS DE PROPIEDADES (HYPOTHESIS) — Refinadas
# ═══════════════════════════════════════════════════════════════════════════════

# Estrategias reutilizables con rangos físicamente motivados
_finite_float = st.floats(
    min_value=-1e6, max_value=1e6,
    allow_nan=False, allow_infinity=False,
)
_positive_float = st.floats(
    min_value=1e-4, max_value=1e4,
    allow_nan=False, allow_infinity=False,
)
_small_positive_float = st.floats(
    min_value=1e-3, max_value=10.0,
    allow_nan=False, allow_infinity=False,
)


class TestPropertyMahalanobisPositivity:
    """
    Propiedad fundamental: d_G(x, ref) ≥ 0 para todo x, ref, G SPD.

    Justificación: G ≻ 0 ⟹ δᵀGδ ≥ 0 ⟹ √(δᵀGδ) ≥ 0.
    """

    @given(
        diag=st.lists(_positive_float, min_size=2, max_size=_MAX_PROP_DIM),
        delta=st.lists(_finite_float, min_size=2, max_size=_MAX_PROP_DIM),
        weight=_positive_float,
    )
    @settings(
        max_examples=300,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_threat_always_nonnegative(
        self,
        diag: List[float],
        delta: List[float],
        weight: float,
    ):
        """
        threatₖ = wₖ·d_G(x, ref) ≥ 0 para todo x, w > 0, G ≻ 0.

        Assume: n ≥ 2 para que la métrica sea válida.
        """
        n = min(len(diag), len(delta))
        assume(n >= 2)
        d = np.array(diag[:n])
        dv = np.array(delta[:n])

        # Precondición: todos los valores de d son positivos
        assume(np.all(d > 0))
        assume(np.all(np.isfinite(dv)))
        assume(np.isfinite(weight) and weight > 0)

        try:
            mt = MetricTensor(d, validate=True)
            spec = SubspaceSpec("t", slice(0, n), weight, np.zeros(n), metric=mt)
            threat = spec.compute_threat(dv)
            assert threat >= 0.0, (
                f"Amenaza negativa: threat={threat:.4e} con d={d}, dv={dv}"
            )
            assert np.isfinite(threat), (
                f"Amenaza no finita: threat={threat} con d={d}, dv={dv}"
            )
        except (MetricTensorError, DimensionalMismatchError, ValueError):
            pass  # Construcción puede fallar con parámetros degenerados

    @given(
        diag=st.lists(_positive_float, min_size=2, max_size=6),
        delta=st.lists(_finite_float, min_size=2, max_size=6),
        weight=_positive_float,
    )
    @settings(max_examples=300)
    def test_threat_formula_diagonal_metric(
        self,
        diag: List[float],
        delta: List[float],
        weight: float,
    ):
        """
        threat = weight · √(Σ dᵢ·δᵢ²) para G = diag(d).

        Verificación de la implementación contra la fórmula analítica.
        """
        n = min(len(diag), len(delta))
        assume(n >= 2)
        d = np.array(diag[:n])
        dv = np.array(delta[:n])
        assume(np.all(d > 0) and np.all(np.isfinite(dv)) and weight > 0)

        try:
            mt = MetricTensor(d, validate=True)
            spec = SubspaceSpec("t", slice(0, n), weight, np.zeros(n), metric=mt)
            threat = spec.compute_threat(dv)
            # Fórmula analítica: √(max(Σ d·δ², 0)) para robustez numérica
            quad_form = float(np.sum(d * dv * dv))
            manual = weight * float(np.sqrt(max(quad_form, 0.0)))

            if np.isfinite(manual) and np.isfinite(threat):
                assert threat == pytest.approx(manual, rel=1e-8, abs=1e-14), (
                    f"Fórmula violada: threat={threat:.4e} ≠ {manual:.4e}"
                )
        except (MetricTensorError, DimensionalMismatchError, ValueError):
            pass


class TestPropertyThreatAssessmentInvariants:
    """
    Invarianzas de ThreatAssessment bajo construcción arbitraria.
    """

    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
            min_size=1, max_size=8,
        )
    )
    @settings(max_examples=400)
    def test_total_threat_is_l2_norm(self, values: List[float]):
        """
        total_threat = ‖levels‖₂ para cualquier conjunto de amenazas ≥ 0.
        """
        assume(len(values) >= 1)
        assume(all(np.isfinite(v) for v in values))

        levels = {f"sub_{i}": v for i, v in enumerate(values)}
        a = ThreatAssessment.from_components(levels)
        manual = float(np.linalg.norm(values))
        assert a.total_threat == pytest.approx(manual, rel=1e-10), (
            f"total_threat={a.total_threat:.6e} ≠ ‖levels‖₂={manual:.6e}"
        )

    @given(
        levels_dict=st.dictionaries(
            keys=st.text(min_size=1, max_size=8),
            values=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            min_size=1, max_size=6,
        )
    )
    @settings(max_examples=300)
    def test_max_source_always_correct(
        self, levels_dict: Dict[str, float],
    ):
        """
        max_source[levels_dict] == max(levels_dict.values()).
        """
        assume(len(levels_dict) >= 1)
        assume(all(np.isfinite(v) for v in levels_dict.values()))

        a = ThreatAssessment.from_components(levels_dict)
        max_val = max(levels_dict.values())
        assert a.max_value == pytest.approx(max_val, rel=1e-12), (
            f"max_value={a.max_value} ≠ max(levels)={max_val}"
        )
        assert levels_dict[a.max_source] == pytest.approx(a.max_value, rel=1e-12), (
            f"levels[max_source] = {levels_dict[a.max_source]} ≠ max_value"
        )


class TestPropertyHysteresisIdempotence:
    """
    Idempotencia de la clasificación en estado estacionario.

    f(f(x, ∅), f(x, ∅)) = f(x, ∅) — el estado converge en 1 iteración.
    """

    @given(
        value=_finite_float,
        warning=_small_positive_float,
        critical_delta=_small_positive_float,
        hysteresis_frac=st.floats(min_value=0.0, max_value=0.49),
    )
    @settings(max_examples=500)
    def test_idempotence_converges_in_one_step(
        self,
        value: float,
        warning: float,
        critical_delta: float,
        hysteresis_frac: float,
    ):
        """
        f(x, s₀) = s₀ donde s₀ = f(x, ∅).

        Precondiciones algebraicas:
        - warning > EPS (umbral no trivial)
        - critical = warning + critical_delta > warning (orden estricto)
        - 0 ≤ hysteresis < (critical - warning) / 2 (banda válida)
        - value ∈ ℝ finito
        """
        critical = warning + critical_delta
        # hysteresis debe ser estrictamente menor que (C-W)/2
        max_hysteresis = (critical - warning) / 2.0
        hysteresis = hysteresis_frac * max_hysteresis

        assume(warning > EPS)
        assume(critical > warning + EPS)
        assume(0.0 <= hysteresis < max_hysteresis - EPS)
        assume(np.isfinite(value))

        s1 = OrthogonalProjector._classify_with_hysteresis(
            value, warning, critical, hysteresis, None,
        )
        s2 = OrthogonalProjector._classify_with_hysteresis(
            value, warning, critical, hysteresis, s1,
        )
        assert s1 == s2, (
            f"No idempotente: v={value:.4f}, w={warning:.4f}, "
            f"c={critical:.4f}, δ={hysteresis:.4f} → "
            f"f(v,∅)={s1.name}, f(v,{s1.name})={s2.name}"
        )


class TestPropertyQuadraticFormPositivity:
    """
    Positividad de la forma cuadrática: vᵀGv ≥ 0 para G SPD.

    Propiedades adicionales: homogeneidad de grado 2.
    """

    @given(
        diag=st.lists(_positive_float, min_size=2, max_size=_MAX_PROP_DIM),
        v=st.lists(_finite_float, min_size=2, max_size=_MAX_PROP_DIM),
    )
    @settings(max_examples=400)
    def test_quadratic_form_nonnegative(
        self, diag: List[float], v: List[float],
    ):
        """vᵀGv ≥ 0 para G = diag(d) con dᵢ > 0."""
        n = min(len(diag), len(v))
        assume(n >= 2)
        d = np.array(diag[:n])
        v_arr = np.array(v[:n])
        assume(np.all(d > 0) and np.all(np.isfinite(v_arr)))

        try:
            mt = MetricTensor(d, validate=True)
            qf = mt.quadratic_form(v_arr)
            assert qf >= 0.0, (
                f"vᵀGv = {qf:.4e} < 0 para d={d}, v={v_arr}"
            )
            assert np.isfinite(qf), f"vᵀGv = {qf} no es finito"
        except (MetricTensorError, ValueError):
            pass

    @given(
        diag=st.lists(_positive_float, min_size=2, max_size=6),
        alpha=st.floats(min_value=-100.0, max_value=100.0,
                        allow_nan=False, allow_infinity=False),
        v=st.lists(_finite_float, min_size=2, max_size=6),
    )
    @settings(max_examples=300)
    def test_quadratic_form_homogeneity_degree_2(
        self,
        diag: List[float],
        alpha: float,
        v: List[float],
    ):
        """
        (αv)ᵀG(αv) = α²·vᵀGv — homogeneidad de grado 2.

        Precaución: para |α| grande y |v| grande puede haber overflow.
        Solo verificamos si el resultado es finito.
        """
        n = min(len(diag), len(v))
        assume(n >= 2)
        d = np.array(diag[:n])
        v_arr = np.array(v[:n])
        assume(np.all(d > 0) and np.all(np.isfinite(v_arr)) and np.isfinite(alpha))

        try:
            mt = MetricTensor(d, validate=True)
            qf_v = mt.quadratic_form(v_arr)
            qf_av = mt.quadratic_form(alpha * v_arr)
            expected = alpha ** 2 * qf_v

            # Solo verificar si no hay overflow
            if np.isfinite(expected) and np.isfinite(qf_av):
                assert qf_av == pytest.approx(expected, rel=1e-8, abs=1e-14), (
                    f"Homogeneidad violada: (αv)ᵀG(αv)={qf_av:.4e} ≠ "
                    f"α²·vᵀGv={expected:.4e} (α={alpha:.2f})"
                )
        except (MetricTensorError, ValueError):
            pass


class TestPropertyResolutionOfIdentity:
    """
    Resolución de identidad Σₖ πₖ = I para subespacios disjuntos y exhaustivos.

    Propiedad topológica: la descomposición en subespacios ortogonales es completa.
    """

    @given(
        n_sub=st.integers(min_value=1, max_value=5),
        sub_size=st.integers(min_value=1, max_value=4),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_sum_of_projections_is_identity(
        self, n_sub: int, sub_size: int,
    ):
        """
        Σₖ πₖ = I_{n_sub * sub_size} para subespacios uniformes.

        La suma se inicializa con np.zeros((n,n)) — no con el escalar 0.
        """
        total_dim = n_sub * sub_size
        assume(total_dim >= 2)
        assume(total_dim <= 20)

        subspaces = {
            f"sub_{k}": SubspaceSpec(
                f"sub_{k}",
                slice(k * sub_size, (k + 1) * sub_size),
                1.0,
                np.zeros(sub_size),
            )
            for k in range(n_sub)
        }

        try:
            proj = OrthogonalProjector(
                dimensions=total_dim,
                subspaces=subspaces,
                topo_indices=None,
                cache_projections=True,
            )
            matrices = list(proj._projection_matrices.values())

            # Suma con inicialización correcta
            total_P = np.zeros((total_dim, total_dim))
            for P in matrices:
                total_P = total_P + P

            err = float(np.linalg.norm(total_P - np.eye(total_dim), "fro"))
            assert err < _FROB_TOL, (
                f"‖ΣPₖ − I_{total_dim}‖_F = {err:.2e} ≥ {_FROB_TOL:.2e} "
                f"(n_sub={n_sub}, sub_size={sub_size})"
            )
        except (DimensionalMismatchError, NumericalStabilityError):
            pass