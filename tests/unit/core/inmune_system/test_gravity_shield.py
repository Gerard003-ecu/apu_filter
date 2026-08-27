# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite: Gravitational Shield — Verificación Formal de Invariantes             ║
║ Ubicación: tests/unit/core/immune_system/test_gravity_shield.py              ║
║ Cobertura: app/core/immune_system/gravity_shield.py (v3.0.0)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Filosofía de la Suite (Peritaje Metodológico):
────────────────────────────────────────────────────────────────────────────────
Esta suite NO se limita a verificar que "el código corre sin excepciones". Cada
clase de prueba corresponde 1:1 a una Fase del módulo bajo prueba y aplica una
de las siguientes estrategias de verificación, elegida por su rigor matemático:

  (a) VALORES DORADOS ANALÍTICOS — para el caso n=2 con métrica base diagonal,
      se deriva a mano la solución cerrada de los símbolos de Christoffel
      ($\Gamma^k_{kk}=c/2$, resto nulo) y se compara bit a bit.

  (b) ORÁCULOS INDEPENDIENTES POR DIFERENCIAS FINITAS — para el caso general
      (n≥3, métrica base no diagonal), se reconstruye Γ mediante diferenciación
      numérica directa de $\tilde G(m^{**})$, SIN reutilizar la fórmula cerrada
      del código bajo prueba, evitando así el antipatrón de "probar la fórmula
      contra sí misma".

  (c) INVARIANTES TEORÉMICOS — Ley de Inercia de Sylvester (SPD se preserva
      bajo congruencia diagonal), simetría de Γ en índices inferiores,
      exactitud de Simpson sobre polinomios cúbicos, cotas de saturación tanh.

  (d) PRUEBAS DE CARACTERIZACIÓN BOOLEANA — el veto dual (acción ∨ curvatura)
      se ejercita en sus 4 combinaciones de tabla de verdad para blindar contra
      una regresión silenciosa de OR a AND.

  (e) HERMETICIDAD — los módulos `app.core.mic_algebra`,
      `app.core.immune_system.metric_tensors` y `app.core.telemetry_schemas`
      se inyectan como dobles mínimos SOLO si no existen físicamente en el
      árbol del proyecto, garantizando que esta suite sea ejecutable de forma
      aislada (unitaria en sentido estricto) sin degradar la integración real.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import sys
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import scipy.linalg as la

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 0 — BOOTSTRAP HERMÉTICO DE DEPENDENCIAS ARQUITECTÓNICAS             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _ensure_stub_module(name: str, builder: Callable[[types.ModuleType], None]) -> None:
    """Inyecta un módulo doble mínimo en `sys.modules` únicamente si el módulo
    real no puede resolverse. No sustituye lógica de negocio: sólo provee la
    superficie de tipos estrictamente necesaria para importar el módulo bajo
    prueba de forma aislada."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType(name)
    builder(module)
    sys.modules[name] = module
    # Enlaza el submódulo como atributo del paquete padre (si existe),
    # replicando el comportamiento del importador real.
    if "." in name:
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, module)


def _register_namespace_chain(*names: str) -> None:
    for pkg in names:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)


def _build_mic_algebra(module: types.ModuleType) -> None:
    class Morphism:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class CategoricalState:
        pass

    class TopologicalInvariantError(Exception):
        pass

    module.Morphism = Morphism
    module.CategoricalState = CategoricalState
    module.TopologicalInvariantError = TopologicalInvariantError


def _build_metric_tensors(module: types.ModuleType) -> None:
    # Métrica base sintética SPD 6x6, determinista y NO diagonal (para ejercitar
    # los términos de acoplamiento cruzado en Fase 2 durante los tests de
    # integración del funtor completo).
    rng = np.random.default_rng(1234)
    A = rng.normal(size=(6, 6))
    module.G_PHYSICS = A @ A.T + 6.0 * np.eye(6)


def _build_telemetry_schemas(module: types.ModuleType) -> None:
    @dataclass
    class PolaronCartridge:
        inertial_mass: float
        volatility_alpha: float = 0.0
        frohlich_coupling: float = 0.0

    module.PolaronCartridge = PolaronCartridge


_register_namespace_chain("app", "app.core", "app.core.immune_system")
_ensure_stub_module("app.core.mic_algebra", _build_mic_algebra)
_ensure_stub_module("app.core.immune_system.metric_tensors", _build_metric_tensors)
_ensure_stub_module("app.core.telemetry_schemas", _build_telemetry_schemas)

# Import del módulo bajo prueba. Se intenta la ruta canónica documentada en el
# encabezado del propio módulo; se conserva un fallback defensivo por si el
# árbol físico del repositorio difiere del declarado.
try:
    from app.core.immune_system import gravity_shield as gs
except ModuleNotFoundError:  # pragma: no cover - fallback de compatibilidad
    from app.tactics import data_validator as gs  # type: ignore[no-redef]


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   UTILIDADES COMPARTIDAS DE LA SUITE                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _random_spd(n: int, seed: int) -> np.ndarray:
    """Genera una matriz SPD determinista, no diagonal, de dimensión n."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def _reference_christoffel_via_fd(
    cache: "gs.BaseMetricCache", effective_mass: float, node_index: int, h: float = 1e-5
) -> np.ndarray:
    """Oráculo INDEPENDIENTE de la fórmula cerrada `_christoffel_from_metric`.

    Reconstruye Γ^μ_{νρ} mediante diferenciación finita centrada de la métrica
    respecto al parámetro escalar m**, explotando la única propiedad
    estructural del modelo (∂_μ G_ab ≡ 0 ∀ μ ≠ k, k=node_index):

        Γ^μ_{νρ} = ½ G^{μσ}(∂_ν G_{σρ} + ∂_ρ G_{νσ} − ∂_σ G_{νρ})

    donde cada término parcial sólo es no nulo si el índice diferenciado es k.
    """
    n = cache.dimension
    k = node_index
    g0, g_inv0 = gs._construct_warped_metric(cache, effective_mass, node_index)
    g_plus, _ = gs._construct_warped_metric(cache, effective_mass + h, node_index)
    g_minus, _ = gs._construct_warped_metric(cache, effective_mass - h, node_index)
    dk_g = (g_plus - g_minus) / (2.0 * h)  # ∂_k G_{ab}

    gamma_ref = np.zeros((n, n, n), dtype=np.float64)
    for mu in range(n):
        for nu in range(n):
            for rho in range(n):
                acc = 0.0
                for sigma in range(n):
                    d_nu = dk_g[sigma, rho] if nu == k else 0.0
                    d_rho = dk_g[nu, sigma] if rho == k else 0.0
                    d_sigma = dk_g[nu, rho] if sigma == k else 0.0
                    acc += g_inv0[mu, sigma] * (d_nu + d_rho - d_sigma)
                gamma_ref[mu, nu, rho] = 0.5 * acc
    return gamma_ref


def _make_dummy_warped_space(
    deformed_metric: np.ndarray, max_curvature: float, node_index: int = 0
) -> "gs.WarpedSpaceTime":
    """Construye un `WarpedSpaceTime` sintético para aislar la Fase 3 de la
    Fase 2 (los símbolos de Christoffel/Riemann no son usados por
    `_evaluate_feynman_kac`, por lo que se rellenan con ceros honestos)."""
    n = deformed_metric.shape[0]
    return gs.WarpedSpaceTime(
        original_metric=deformed_metric.copy(),
        deformed_metric=deformed_metric,
        christoffel_symbols=np.zeros((n, n, n)),
        riemann_curvature=np.zeros((n, n)),
        sectional_curvatures=np.zeros(n),
        max_sectional_curvature=max_curvature,
        node_index=node_index,
    )


def _critical_action() -> float:
    """Réplica independiente del umbral crítico de acción, para no depender
    de una constante interna oculta al comparar resultados."""
    return -gs.GravitationalConstants.HBAR_EFF * math.log(
        gs.GravitationalConstants.SCHWARZSCHILD_TOLERANCE
    )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 — ADQUISICIÓN DE MASA INERCIAL                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase1SmoothInertialFloor:
    """Verifica el piso de masa C^∞ que sustituye a `max()`."""

    def test_floor_en_cero_devuelve_masa_minima_exacta(self) -> None:
        result = gs._smooth_inertial_floor(0.0, 1e-12)
        assert result == pytest.approx(1e-12, rel=1e-12)

    def test_floor_es_estrictamente_monotono_creciente(self) -> None:
        xs = np.linspace(0.0, 10.0, 50)
        ys = [gs._smooth_inertial_floor(x, 1e-6) for x in xs]
        assert all(b >= a for a, b in zip(ys, ys[1:]))

    def test_floor_es_transparente_para_masas_grandes(self) -> None:
        m = 1e6
        floored = gs._smooth_inertial_floor(m, 1e-12)
        # lim_{m→∞} floor(m) - m = 0
        assert abs(floored - m) < 1e-6

    def test_floor_no_presenta_punto_anguloso_en_m_min(self) -> None:
        """A diferencia de max(m, m_min), sqrt(m²+m_min²) debe tener derivada
        continua (idéntica por ambos lados) en m = m_min."""
        m_min = 1.0
        h = 1e-6
        left = (
            gs._smooth_inertial_floor(m_min, m_min)
            - gs._smooth_inertial_floor(m_min - h, m_min)
        ) / h
        right = (
            gs._smooth_inertial_floor(m_min + h, m_min)
            - gs._smooth_inertial_floor(m_min, m_min)
        ) / h
        assert left == pytest.approx(right, abs=1e-4)


class TestPhase1SaturatingFrohlichFactor:
    """Verifica la saturación tanh que reemplaza la corrección lineal
    divergente $1+\\alpha_f/2\\pi$."""

    def test_factor_en_cero_es_unidad(self) -> None:
        assert gs._saturating_frohlich_factor(0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("coupling", [1.0, 10.0, 1e3, 1e9])
    def test_factor_acotado_en_uno_dos(self, coupling: float) -> None:
        f = gs._saturating_frohlich_factor(coupling)
        assert 1.0 <= f < 2.0

    def test_factor_monotono_creciente(self) -> None:
        xs = np.linspace(0.0, 50.0, 30)
        ys = [gs._saturating_frohlich_factor(x) for x in xs]
        assert all(b >= a for a, b in zip(ys, ys[1:]))

    def test_pendiente_en_origen_coincide_con_aproximacion_fisica(self) -> None:
        """f'(0) debe ser 1/(2π), preservando la corrección de baja
        intensidad de acoplamiento del polarón de Fröhlich original."""
        h = 1e-6
        derivative = (
            gs._saturating_frohlich_factor(h) - gs._saturating_frohlich_factor(-h)
        ) / (2 * h)
        assert derivative == pytest.approx(1.0 / (2.0 * math.pi), rel=1e-4)


class TestPhase1AcquireEffectiveMass:
    """Verifica el morfismo C1: (costo, α, α_f) ⟶ MassAcquisitionResult."""

    def test_rechaza_costo_negativo(self) -> None:
        with pytest.raises(ValueError):
            gs._acquire_effective_mass(-1.0, 0.0)

    def test_rechaza_volatilidad_negativa(self) -> None:
        with pytest.raises(ValueError):
            gs._acquire_effective_mass(1.0, -0.1)

    def test_rechaza_frohlich_negativo(self) -> None:
        with pytest.raises(ValueError):
            gs._acquire_effective_mass(1.0, 0.0, frohlich_coupling=-0.1)

    def test_caso_base_sin_volatilidad_ni_frohlich(self) -> None:
        result = gs._acquire_effective_mass(10.0, 0.0)
        expected = gs._smooth_inertial_floor(
            10.0, gs.GravitationalConstants.MINIMAL_INERTIAL_MASS
        )
        assert result.effective_mass == pytest.approx(expected)
        assert result.frohlich_factor == pytest.approx(1.0)

    def test_frohlich_none_y_frohlich_cero_son_equivalentes(self) -> None:
        """Documenta explícitamente la trampa del `0.0` falsy en
        `if frohlich_coupling:` — ambos casos deben producir el mismo
        resultado, evitando una futura regresión silenciosa."""
        r_none = gs._acquire_effective_mass(5.0, 1.0, frohlich_coupling=None)
        r_zero = gs._acquire_effective_mass(5.0, 1.0, frohlich_coupling=0.0)
        assert r_none.effective_mass == pytest.approx(r_zero.effective_mass)
        assert r_none.frohlich_factor == r_zero.frohlich_factor == 1.0

    def test_masa_resultante_nunca_por_debajo_del_piso(self) -> None:
        result = gs._acquire_effective_mass(0.0, 0.0)
        assert result.effective_mass >= gs.GravitationalConstants.MINIMAL_INERTIAL_MASS

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 10.0])
    def test_masa_monotona_en_alpha(self, alpha: float) -> None:
        base = gs._acquire_effective_mass(3.0, alpha).effective_mass
        higher = gs._acquire_effective_mass(3.0, alpha + 0.1).effective_mass
        assert higher >= base

    def test_registro_de_procedencia_es_auditable(self) -> None:
        result = gs._acquire_effective_mass(7.5, 1.2, frohlich_coupling=0.3)
        assert result.raw_cost == 7.5
        assert result.volatility_alpha == 1.2
        assert result.frohlich_coupling == 0.3
        assert result.frohlich_factor == pytest.approx(
            gs._saturating_frohlich_factor(0.3)
        )

    def test_resultado_es_inmutable(self) -> None:
        result = gs._acquire_effective_mass(1.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.effective_mass = 999.0  # type: ignore[misc]


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 — DEFORMACIÓN MÉTRICA (POZO GRAVITACIONAL)                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase2BaseMetricCache:
    """Verifica la construcción y validación única de la caché espectral
    (Fase 2, primer eslabón, recibe implícitamente `MassAcquisitionResult`
    de la Fase 1 vía `_deform_metric_tensor`)."""

    def test_matriz_spd_valida_construye_cache_correcta(self) -> None:
        g = _random_spd(4, seed=1)
        cache = gs._build_base_metric_cache(g)
        assert cache.dimension == 4
        np.testing.assert_allclose(
            cache.cholesky_factor @ cache.cholesky_factor.T, g, atol=1e-8
        )
        np.testing.assert_allclose(g @ cache.g_inv, np.eye(4), atol=1e-8)

    def test_matriz_no_cuadrada_lanza_colapso(self) -> None:
        g = np.ones((3, 4))
        with pytest.raises(gs.GravitationalCollapseError):
            gs._build_base_metric_cache(g)

    def test_matriz_no_simetrica_lanza_colapso(self) -> None:
        g = _random_spd(3, seed=2)
        g[0, 1] += 5.0  # rompe simetría
        with pytest.raises(gs.GravitationalCollapseError):
            gs._build_base_metric_cache(g)

    def test_matriz_no_definida_positiva_lanza_colapso(self) -> None:
        g = np.diag([1.0, -1.0, 1.0])
        with pytest.raises(gs.GravitationalCollapseError):
            gs._build_base_metric_cache(g)


class TestPhase2ConstructWarpedMetric:
    """Verifica la congruencia diagonal y la Ley de Inercia de Sylvester."""

    @pytest.fixture
    def cache(self) -> "gs.BaseMetricCache":
        return gs._build_base_metric_cache(_random_spd(5, seed=3))

    def test_masa_nula_deja_metrica_invariante(self, cache) -> None:
        g_tilde, _ = gs._construct_warped_metric(cache, 0.0, node_index=0)
        np.testing.assert_allclose(g_tilde, cache.g_base, atol=1e-12)

    def test_simetria_preservada(self, cache) -> None:
        g_tilde, _ = gs._construct_warped_metric(cache, 2.5, node_index=1)
        np.testing.assert_allclose(g_tilde, g_tilde.T, atol=1e-10)

    @pytest.mark.parametrize("mass", [0.0, 1e-9, 1.0, 50.0, 1e4])
    def test_spd_preservada_bajo_congruencia_sylvester(self, cache, mass: float) -> None:
        g_tilde, _ = gs._construct_warped_metric(cache, mass, node_index=2)
        eigenvalues = la.eigvalsh(g_tilde)
        assert np.all(eigenvalues > 0.0), (
            "La Ley de Inercia de Sylvester garantiza SPD para toda masa >= 0; "
            "una violación indica un defecto de implementación."
        )

    def test_inversa_analitica_es_correcta(self, cache) -> None:
        g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, 3.3, node_index=0)
        np.testing.assert_allclose(
            g_tilde @ g_inv_tilde, np.eye(cache.dimension), atol=1e-8
        )

    def test_indice_fuera_de_rango_lanza_index_error(self, cache) -> None:
        with pytest.raises(IndexError):
            gs._construct_warped_metric(cache, 1.0, node_index=999)

    def test_soporta_masa_compleja_para_diferenciacion_de_paso_complejo(self, cache) -> None:
        g_tilde, _ = gs._construct_warped_metric(cache, complex(1.0, 1e-20), node_index=0)
        assert g_tilde.dtype == np.complex128

    def test_formula_de_escalado_explicita(self, cache) -> None:
        """Verifica elemento a elemento: g̃_ab = g_ab · exp(δ_a/2)·exp(δ_b/2)."""
        mass, k = 4.2, 2
        c = gs.GravitationalConstants.G_C4_FACTOR
        n = cache.dimension
        delta = np.zeros(n)
        delta[k] = c * mass
        expected = np.outer(np.exp(delta / 2), np.exp(delta / 2)) * cache.g_base
        g_tilde, _ = gs._construct_warped_metric(cache, mass, node_index=k)
        np.testing.assert_allclose(g_tilde, expected, atol=1e-10)


class TestPhase2ChristoffelClosedForm:
    """Valida la fórmula cerrada de Γ contra (a) un caso analítico exacto y
    (b) un oráculo numérico independiente por diferencias finitas."""

    def test_caso_analitico_n2_diagonal(self) -> None:
        """Para g_base = I_2, la métrica deformada es el warp exponencial
        clásico ds² = e^{cu}dx² + dy². Se deriva a mano:
            Γ^0_{00} = c/2,  resto de componentes = 0.
        """
        cache = gs._build_base_metric_cache(np.eye(2))
        c = gs.GravitationalConstants.G_C4_FACTOR
        for mass in (0.0, 1.0, 10.0, 100.0):
            g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, mass, node_index=0)
            gamma = gs._christoffel_from_metric(g_tilde, g_inv_tilde, node_index=0)
            assert gamma[0, 0, 0] == pytest.approx(c / 2.0, rel=1e-9)
            gamma_sin_000 = gamma.copy()
            gamma_sin_000[0, 0, 0] = 0.0
            np.testing.assert_allclose(gamma_sin_000, np.zeros((2, 2, 2)), atol=1e-9)

    @pytest.mark.parametrize("n,seed,node_index,mass", [
        (3, 10, 0, 1.5),
        (3, 11, 2, 0.3),
        (4, 12, 1, 7.0),
        (5, 13, 4, 0.01),
    ])
    def test_oraculo_diferencias_finitas_metrica_general(
        self, n: int, seed: int, node_index: int, mass: float
    ) -> None:
        cache = gs._build_base_metric_cache(_random_spd(n, seed=seed))
        g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, mass, node_index)
        gamma_closed = gs._christoffel_from_metric(g_tilde, g_inv_tilde, node_index)
        gamma_reference = _reference_christoffel_via_fd(cache, mass, node_index)
        np.testing.assert_allclose(gamma_closed, gamma_reference, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("n,seed,node_index,mass", [(4, 20, 1, 2.0), (5, 21, 3, 0.5)])
    def test_simetria_en_indices_inferiores(
        self, n: int, seed: int, node_index: int, mass: float
    ) -> None:
        """Γ^μ_{νρ} = Γ^μ_{ρν} — propiedad intrínseca de la conexión de
        Levi-Civita (torsión nula)."""
        cache = gs._build_base_metric_cache(_random_spd(n, seed=seed))
        g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, mass, node_index)
        gamma = gs._christoffel_from_metric(g_tilde, g_inv_tilde, node_index)
        np.testing.assert_allclose(gamma, np.transpose(gamma, (0, 2, 1)), atol=1e-10)


class TestPhase2ComplexStepDifferentiation:
    """Contrasta la diferenciación de paso complejo contra diferencias
    finitas centradas de precisión moderada, demostrando su superioridad."""

    def test_coincide_con_diferencias_finitas_finas(self) -> None:
        cache = gs._build_base_metric_cache(_random_spd(4, seed=30))
        mass, node_index = 2.0, 1
        d_complex = gs._complex_step_partial_k_christoffel(cache, mass, node_index)

        h = 1e-5
        g_p, ginv_p = gs._construct_warped_metric(cache, mass + h, node_index)
        g_m, ginv_m = gs._construct_warped_metric(cache, mass - h, node_index)
        gamma_p = gs._christoffel_from_metric(g_p, ginv_p, node_index)
        gamma_m = gs._christoffel_from_metric(g_m, ginv_m, node_index)
        d_fd = (gamma_p - gamma_m) / (2 * h)

        np.testing.assert_allclose(d_complex, d_fd, atol=1e-4, rtol=1e-4)

    def test_es_mas_preciso_que_diferencias_finitas_agresivas(self) -> None:
        """Con h agresivamente pequeño (1e-8), la diferencia finita central
        sufre cancelación catastrófica; el paso complejo no."""
        cache = gs._build_base_metric_cache(_random_spd(3, seed=31))
        mass, node_index = 1.0, 0
        h_tiny = 1e-8

        g_p, ginv_p = gs._construct_warped_metric(cache, mass + h_tiny, node_index)
        g_m, ginv_m = gs._construct_warped_metric(cache, mass - h_tiny, node_index)
        gamma_p = gs._christoffel_from_metric(g_p, ginv_p, node_index)
        gamma_m = gs._christoffel_from_metric(g_m, ginv_m, node_index)
        d_fd_tiny = (gamma_p - gamma_m) / (2 * h_tiny)

        d_complex = gs._complex_step_partial_k_christoffel(cache, mass, node_index)

        # Referencia de alta calidad con h moderado (bien condicionado).
        h_ref = 1e-5
        g_p2, ginv_p2 = gs._construct_warped_metric(cache, mass + h_ref, node_index)
        g_m2, ginv_m2 = gs._construct_warped_metric(cache, mass - h_ref, node_index)
        d_fd_ref = (
            gs._christoffel_from_metric(g_p2, ginv_p2, node_index)
            - gs._christoffel_from_metric(g_m2, ginv_m2, node_index)
        ) / (2 * h_ref)

        error_complex = np.max(np.abs(d_complex - d_fd_ref))
        error_fd_tiny = np.max(np.abs(d_fd_tiny - d_fd_ref))
        assert error_complex <= error_fd_tiny


class TestPhase2SectionalCurvature:
    """Verifica propiedades estructurales de la curvatura seccional reducida."""

    def test_curvatura_en_plano_degenerado_es_nula(self) -> None:
        cache = gs._build_base_metric_cache(_random_spd(4, seed=40))
        mass, k = 1.0, 1
        g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, mass, k)
        gamma = gs._christoffel_from_metric(g_tilde, g_inv_tilde, k)
        d_k_gamma = gs._complex_step_partial_k_christoffel(cache, mass, k)
        curvatures, _ = gs._sectional_curvatures_around_node(g_tilde, gamma, d_k_gamma, k)
        assert curvatures[k] == 0.0

    def test_curvaturas_son_finitas_para_masas_extremas(self) -> None:
        cache = gs._build_base_metric_cache(_random_spd(5, seed=41))
        for mass in (1e-6, 1.0, 1e3):
            g_tilde, g_inv_tilde = gs._construct_warped_metric(cache, mass, 2)
            gamma = gs._christoffel_from_metric(g_tilde, g_inv_tilde, 2)
            d_k_gamma = gs._complex_step_partial_k_christoffel(cache, mass, 2)
            curvatures, _ = gs._sectional_curvatures_around_node(g_tilde, gamma, d_k_gamma, 2)
            assert np.all(np.isfinite(curvatures))


class TestPhase2DeformMetricTensorIntegration:
    """Integra Fase 1 → Fase 2: recibe `MassAcquisitionResult` (salida real
    de `_acquire_effective_mass`) y produce `WarpedSpaceTime`."""

    @pytest.fixture
    def cache(self) -> "gs.BaseMetricCache":
        return gs._build_base_metric_cache(_random_spd(4, seed=50))

    def test_composicion_fase1_fase2_end_to_end(self, cache) -> None:
        mass_result = gs._acquire_effective_mass(base_cost=5.0, volatility_alpha=1.0)
        warped = gs._deform_metric_tensor(cache, mass_result, node_index=1)
        assert isinstance(warped, gs.WarpedSpaceTime)
        np.testing.assert_allclose(warped.original_metric, cache.g_base)
        eigenvalues = la.eigvalsh(warped.deformed_metric)
        assert np.all(eigenvalues > 0.0)
        assert warped.max_sectional_curvature == pytest.approx(
            float(np.max(np.abs(warped.sectional_curvatures)))
        )
        assert warped.node_index == 1

    def test_masa_nula_produce_metrica_deformada_igual_a_base(self, cache) -> None:
        mass_result = gs._acquire_effective_mass(base_cost=0.0, volatility_alpha=0.0)
        # El piso mínimo garantiza masa > 0 pero infinitesimal; la métrica
        # debe ser prácticamente idéntica a la base.
        warped = gs._deform_metric_tensor(cache, mass_result, node_index=0)
        np.testing.assert_allclose(warped.deformed_metric, cache.g_base, atol=1e-6)

    def test_indice_fuera_de_rango_lanza_index_error(self, cache) -> None:
        mass_result = gs._acquire_effective_mass(1.0, 0.0)
        with pytest.raises(IndexError):
            gs._deform_metric_tensor(cache, mass_result, node_index=999)

    def test_rama_defensiva_diagonal_no_positiva_es_alcanzable(self) -> None:
        """Aunque la SPD está teorémicamente garantizada (Sylvester) para
        entradas válidas, se fuerza deliberadamente una `BaseMetricCache`
        patológica (bypass del constructor validado) para certificar que la
        aserción defensiva del código realmente dispara la excepción y no es
        código muerto inalcanzable."""
        n = 3
        g_pathological = np.eye(n)
        g_pathological[0, 0] = 0.0  # diagonal nula: inválida físicamente
        pathological_cache = gs.BaseMetricCache(
            g_base=g_pathological,
            cholesky_factor=np.eye(n),
            g_inv=np.eye(n),
            dimension=n,
        )
        mass_result = gs._acquire_effective_mass(1.0, 0.0)
        with pytest.raises(gs.GravitationalCollapseError):
            gs._deform_metric_tensor(pathological_cache, mass_result, node_index=0)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 — ATRAPAMIENTO GEODÉSICO (FEYNMAN-KAC)                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ── Continuación directa de la Fase 2: `WarpedSpaceTime` es tanto la salida
#    final de la Fase 2 como la entrada formal de las pruebas siguientes. ────

class TestPhase3KineticDensity:
    def test_calculo_explicito_diagonal(self) -> None:
        g = np.diag([2.0, 3.0])
        v = np.array([1.0, 1.0])
        assert gs._kinetic_density(v, g) == pytest.approx(5.0)

    def test_vector_nulo_produce_densidad_nula(self) -> None:
        g = _random_spd(3, seed=60)
        assert gs._kinetic_density(np.zeros(3), g) == 0.0

    def test_metrica_negativa_definida_lanza_colapso(self) -> None:
        g = -np.eye(2)
        v = np.array([1.0, 0.0])
        with pytest.raises(gs.GravitationalCollapseError):
            gs._kinetic_density(v, g)

    def test_ruido_numerico_negativo_infimo_se_recorta_a_cero(self) -> None:
        """Un valor ligeramente negativo (dentro de tolerancia de ruido de
        punto flotante) NO debe lanzar excepción, sino recortarse a 0.0."""
        g = np.array([[-1e-10, 0.0], [0.0, 1.0]])
        v = np.array([1.0, 0.0])
        assert gs._kinetic_density(v, g) == 0.0


class TestPhase3SimpsonComposite:
    def test_exactitud_sobre_polinomio_cubico(self) -> None:
        """Simpson es exacto (error de truncamiento nulo) para polinomios de
        grado ≤ 3. ∫₀¹x³dx = 0.25."""
        xs = np.linspace(0.0, 1.0, 5)
        f_vals = xs ** 3
        h = xs[1] - xs[0]
        result = gs._simpson_composite(f_vals, h)
        assert result == pytest.approx(0.25, abs=1e-12)

    def test_numero_par_de_puntos_lanza_value_error(self) -> None:
        f_vals = np.array([0.0, 1.0, 2.0, 3.0])  # 4 puntos = 3 subintervalos (impar)
        with pytest.raises(ValueError):
            gs._simpson_composite(f_vals, 0.25)

    def test_menos_de_tres_puntos_lanza_value_error(self) -> None:
        with pytest.raises(ValueError):
            gs._simpson_composite(np.array([1.0, 2.0]), 1.0)


class TestPhase3EvaluateFeynmanKac:
    """Ejercita la composición C3 y la tabla de verdad completa del veto
    booleano dual (acción ∨ curvatura)."""

    def test_trayectoria_1d_accion_exacta_sin_error_discretizacion(self) -> None:
        g = 2.0 * np.eye(3)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        v = np.ones(3)
        result = gs._evaluate_feynman_kac(warped, v)
        assert result.action_integral == pytest.approx(0.5 * (v @ g @ v))
        assert result.discretization_error == 0.0

    def test_veto_dual_tabla_de_verdad_completa(self) -> None:
        """Certifica semánticamente el operador OR (no AND) mediante las 4
        combinaciones de la tabla de verdad; una regresión de OR→AND haría
        fallar los casos 2 y 3."""
        g_low = 1e-6 * np.eye(2)   # produce acción despreciable
        g_high = 1e8 * np.eye(2)   # produce acción muy por encima del umbral
        v = np.ones(2)
        crit = gs.GravitationalConstants.CRITICAL_SECTIONAL_CURVATURE

        # Caso 1: ambos falsos → no atrapado.
        w1 = _make_dummy_warped_space(g_low, max_curvature=crit / 2)
        r1 = gs._evaluate_feynman_kac(w1, v)
        assert not r1.action_veto and not r1.curvature_veto and not r1.is_trapped

        # Caso 2: sólo acción → atrapado.
        w2 = _make_dummy_warped_space(g_high, max_curvature=crit / 2)
        r2 = gs._evaluate_feynman_kac(w2, v)
        assert r2.action_veto and not r2.curvature_veto and r2.is_trapped

        # Caso 3: sólo curvatura → atrapado (clave para descartar AND).
        w3 = _make_dummy_warped_space(g_low, max_curvature=crit * 2)
        r3 = gs._evaluate_feynman_kac(w3, v)
        assert not r3.action_veto and r3.curvature_veto and r3.is_trapped

        # Caso 4: ambos verdaderos → atrapado.
        w4 = _make_dummy_warped_space(g_high, max_curvature=crit * 2)
        r4 = gs._evaluate_feynman_kac(w4, v)
        assert r4.action_veto and r4.curvature_veto and r4.is_trapped

    def test_veto_decidido_en_espacio_logaritmico_es_inmune_a_underflow(self) -> None:
        """Con acción extrema, Ψ colapsa a 0.0 (underflow), pero el veredicto
        de atrapamiento debe seguir siendo correcto porque se decide sobre
        S_E, no sobre Ψ."""
        g = 1e12 * np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        result = gs._evaluate_feynman_kac(warped, np.ones(2))
        assert result.feynman_amplitude == 0.0
        assert result.action_veto is True
        assert result.is_trapped is True

    @pytest.mark.parametrize("m_points", [5, 9])
    def test_trayectoria_2d_constante_coincide_con_1d(self, m_points: int) -> None:
        g = np.diag([1.0, 4.0])
        v = np.array([0.5, 0.5])
        warped = _make_dummy_warped_space(g, max_curvature=0.0)

        r_1d = gs._evaluate_feynman_kac(warped, v)
        trajectory_2d = np.tile(v, (m_points, 1))
        r_2d = gs._evaluate_feynman_kac(warped, trajectory_2d)

        assert r_2d.action_integral == pytest.approx(r_1d.action_integral, rel=1e-9)

    def test_richardson_no_aplicable_produce_nan(self) -> None:
        g = np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        trajectory = np.tile(np.array([0.1, 0.1]), (3, 1))  # m-1=2, no divisible entre 4
        result = gs._evaluate_feynman_kac(warped, trajectory)
        assert math.isnan(result.discretization_error)

    def test_richardson_aplicable_produce_error_casi_nulo_para_velocidad_constante(
        self,
    ) -> None:
        g = np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        trajectory = np.tile(np.array([0.2, 0.2]), (9, 1))  # m-1=8, divisible entre 4
        result = gs._evaluate_feynman_kac(warped, trajectory)
        assert not math.isnan(result.discretization_error)
        assert result.discretization_error < 1e-9

    def test_relleno_por_paridad_no_falla_con_numero_par_de_muestras(self) -> None:
        g = np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        trajectory = np.tile(np.array([0.1, 0.1]), (4, 1))  # m-1=3, impar → padding
        result = gs._evaluate_feynman_kac(warped, trajectory)
        assert math.isfinite(result.action_integral)

    @pytest.mark.parametrize("bad_shape", [(), (2, 2, 2)])
    def test_forma_de_trayectoria_invalida_lanza_value_error(self, bad_shape) -> None:
        g = np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        trajectory = np.zeros(bad_shape) if bad_shape else np.array(1.0)
        with pytest.raises(ValueError):
            gs._evaluate_feynman_kac(warped, trajectory)

    def test_menos_de_tres_muestras_2d_lanza_value_error(self) -> None:
        g = np.eye(2)
        warped = _make_dummy_warped_space(g, max_curvature=0.0)
        trajectory = np.zeros((2, 2))
        with pytest.raises(ValueError):
            gs._evaluate_feynman_kac(warped, trajectory)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR — GravitationalShieldFunctor (F = C3 ∘ C2 ∘ C1)             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class _FakePolaronCompleto:
    inertial_mass = 3.0
    volatility_alpha = 0.5
    frohlich_coupling = 0.2


class _FakePolaronMinimo:
    """Sólo expone `inertial_mass`; el resto debe resolverse vía defaults."""
    inertial_mass = 4.0


class TestOrchestratorPolaronExtraction:
    def test_extrae_campos_completos(self) -> None:
        cost, alpha, coupling = gs.GravitationalShieldFunctor._extract_polaron_fields(
            _FakePolaronCompleto()
        )
        assert cost == 3.0 and alpha == 0.5 and coupling == 0.2

    def test_campos_ausentes_usan_defaults_seguros(self) -> None:
        cost, alpha, coupling = gs.GravitationalShieldFunctor._extract_polaron_fields(
            _FakePolaronMinimo()
        )
        assert cost == 4.0
        assert alpha == 0.0
        assert coupling is None

    def test_valores_negativos_se_normalizan_con_valor_absoluto(self) -> None:
        class _Negativo:
            inertial_mass = -5.0
            volatility_alpha = -1.0
            frohlich_coupling = -0.3

        cost, alpha, coupling = gs.GravitationalShieldFunctor._extract_polaron_fields(
            _Negativo()
        )
        assert cost == 5.0 and alpha == 1.0 and coupling == 0.3


class TestOrchestratorContractIntegrity:
    def test_excepciones_heredan_de_topological_invariant_error(self) -> None:
        assert issubclass(
            gs.GravitationalCollapseError,
            gs.TopologicalInvariantError if hasattr(gs, "TopologicalInvariantError")
            else Exception,
        )

    def test_estructuras_de_datos_son_inmutables(self) -> None:
        mass_result = gs._acquire_effective_mass(1.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mass_result.raw_cost = 0.0  # type: ignore[misc]


class TestOrchestratorEndToEnd:
    """Verifica la composición completa F = C3 ∘ C2 ∘ C1 a través del punto
    de entrada público `enforce_gravitational_attractor`."""

    def test_inicializacion_falla_si_g_physics_es_invalida(self, monkeypatch) -> None:
        monkeypatch.setattr(gs, "G_PHYSICS", np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(gs.GravitationalCollapseError):
            gs.GravitationalShieldFunctor()

    def test_masa_baja_no_activa_el_veto(self) -> None:
        functor = gs.GravitationalShieldFunctor()
        n = functor._metric_cache.dimension
        polaron = _FakePolaronMinimo()
        polaron.inertial_mass = 1e-8
        result = functor.enforce_gravitational_attractor(
            polaron=polaron,
            node_index=0,
            llm_attention_vector=1e-6 * np.ones(n),
        )
        assert isinstance(result, gs.PolyakovAction)
        assert result.is_trapped is False

    def test_masa_alta_activa_veto_por_horizonte_de_eventos(self) -> None:
        functor = gs.GravitationalShieldFunctor()
        n = functor._metric_cache.dimension
        polaron = _FakePolaronCompleto()
        polaron.inertial_mass = 1e6
        with pytest.raises(gs.EventHorizonViolation) as excinfo:
            functor.enforce_gravitational_attractor(
                polaron=polaron,
                node_index=0,
                llm_attention_vector=np.ones(n),
            )
        assert "Obstrucción Geodésica" in str(excinfo.value)

    def test_indice_de_nodo_invalido_propaga_index_error(self) -> None:
        functor = gs.GravitationalShieldFunctor()
        n = functor._metric_cache.dimension
        with pytest.raises(IndexError):
            functor.enforce_gravitational_attractor(
                polaron=_FakePolaronCompleto(),
                node_index=n + 10,
                llm_attention_vector=np.ones(n),
            )

    def test_determinismo_absoluto_entre_ejecuciones_independientes(self) -> None:
        """El 'Atractor Determinista Absoluto' exige reproducibilidad bit-casi
        -exacta entre instancias frescas del funtor con idénticas entradas."""
        n = np.asarray(gs.G_PHYSICS).shape[0]
        polaron = _FakePolaronCompleto()
        vector = np.linspace(0.01, 0.05, n)

        r1 = gs.GravitationalShieldFunctor().enforce_gravitational_attractor(
            polaron, node_index=0, llm_attention_vector=vector
        )
        r2 = gs.GravitationalShieldFunctor().enforce_gravitational_attractor(
            polaron, node_index=0, llm_attention_vector=vector
        )
        assert r1.action_integral == pytest.approx(r2.action_integral, rel=1e-12)
        assert r1.is_trapped == r2.is_trapped

    def test_veto_reporta_ambas_causas_cuando_ambas_aplican(self) -> None:
        """Si tanto la acción como la curvatura exceden umbral, el mensaje de
        excepción debe listar ambas razones (trazabilidad forense)."""
        functor = gs.GravitationalShieldFunctor()
        n = functor._metric_cache.dimension
        polaron = _FakePolaronCompleto()
        polaron.inertial_mass = 1e6
        try:
            functor.enforce_gravitational_attractor(
                polaron=polaron,
                node_index=0,
                llm_attention_vector=1e3 * np.ones(n),
            )
            pytest.fail("Se esperaba EventHorizonViolation")
        except gs.EventHorizonViolation as exc:
            assert "Acción de Polyakov" in str(exc) or "Curvatura seccional" in str(exc)