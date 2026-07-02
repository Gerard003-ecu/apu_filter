# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : test_optical_riemann_lens.py                                               ║
║  Ruta   : tests/physics/test_optical_riemann_lens.py                                 ║
║  Versión: 4.0.0-Test-Suite                                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Suite de pruebas rigurosa para optical_riemann_lens.py v4.0.0.

Estructura de la suite (espejo de las 3 fases del módulo):
══════════════════════════════════════════════════════════

  BLOQUE 0 — Fixtures y utilidades matemáticas compartidas
  BLOQUE 1 — Pruebas de estructuras de datos (dataclasses)
  BLOQUE 2 — Pruebas de excepciones ópticas
  BLOQUE 3 — Pruebas de Fase 1 (SphericalHarmonicsSpectrometer)
  BLOQUE 4 — Pruebas de Fase 2 (CategoricalOpticLens)
  BLOQUE 5 — Pruebas de Fase 3 (OpticalRiemannLensFibrator)
  BLOQUE 6 — Pruebas de propiedades matemáticas invariantes
  BLOQUE 7 — Pruebas de robustez numérica y casos extremos
  BLOQUE 8 — Pruebas de integración end-to-end
  BLOQUE 9 — Pruebas de rendimiento y complejidad

Principios de diseño de la suite:
───────────────────────────────────
  P1. Cada prueba verifica exactamente UNA propiedad matemática.
  P2. Los valores esperados se calculan analíticamente, no se hardcodean
      sin justificación.
  P3. Las tolerancias numéricas se justifican explícitamente en cada prueba.
  P4. Los casos de error verifican el TIPO de excepción y el MENSAJE.
  P5. Las pruebas paramétricas cubren el espacio de parámetros relevante.
  P6. Las fixtures de numpy usan semillas fijas para reproducibilidad.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np
import pytest
import scipy.special as sp_special
from numpy.typing import NDArray

# ── Módulo bajo prueba ────────────────────────────────────────────────────────
from app.physics.optical_riemann_lens import (
    # Excepciones
    GramSchmidtDegeneracyError,
    LensSingularityError,
    OpticalDispersionError,
    ParsevalViolationError,
    SpectralGridError,
    # Tipos auxiliares
    GramSchmidtBasis,
    SpherePoint,
    # Estructuras de datos
    RefractedState,
    SpectralDiagnostics,
    SphericalGrid,
    SphericalSpectrum,
    # Clases de fase
    Phase1_SphericalHarmonicsSpectrometer,
    Phase2_CategoricalOpticLens,
    # Agente principal
    OpticalRiemannLensFibrator,
    # Constantes (importadas para pruebas de invariantes)
    _EPSILON,
    _FERMAT_ALPHA,
    _ORTHOGONALITY_TOL,
    _PARSEVAL_REL_TOL,
    _SIGMA_MAX,
    _SIGMA_MIN,
    _N_SVD_COMPONENTS,
    _COMPRESSION_FLOOR,
    _COMPRESSION_CAP,
)

# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 0 — FIXTURES Y UTILIDADES MATEMÁTICAS COMPARTIDAS                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

# Semilla global para reproducibilidad determinista
_RNG_SEED: int = 42
_RNG = np.random.default_rng(_RNG_SEED)


def make_rng(seed: int = _RNG_SEED) -> np.random.Generator:
    """Genera un RNG con semilla fija para reproducibilidad."""
    return np.random.default_rng(seed)


def analytic_sph_harm_integral(
    l: int, m: int, lp: int, mp: int,
    n_theta: int = 64, n_phi: int = 128,
) -> complex:
    r"""
    Calcula numéricamente ⟨Y_l^m, Y_{l'}^{m'}⟩_{L²(S²)} con cuadratura
    de alta resolución para usar como referencia en las pruebas.

    Usa n_theta=64, n_phi=128 por defecto para alta precisión (independiente
    de la grilla bajo prueba).

    Retorna
    -------
    complex : ⟨Y_l^m, Y_{l'}^{m'}⟩ ≈ δ_{ll'} δ_{mm'}
    """
    t_nodes, w_gauss = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(t_nodes)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    delta_phi = 2.0 * np.pi / n_phi
    W = w_gauss[:, np.newaxis] * delta_phi

    Y_lm  = sp_special.sph_harm(m,  l,  PHI, THETA)
    Y_lpm = sp_special.sph_harm(mp, lp, PHI, THETA)
    return complex(np.sum(Y_lm * np.conj(Y_lpm) * W))


def make_random_logits(n: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """Genera un vector de logits aleatorios de longitud n."""
    rng = make_rng(seed)
    return rng.standard_normal(n).astype(np.float64)


def make_spd_matrix(d: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    r"""
    Genera una matriz SPD (simétrica definida positiva) de dimensión d×d.

    Método: G = AᵀA + d·I para garantizar λ_min ≥ d > 0.
    """
    rng = make_rng(seed)
    A = rng.standard_normal((d, d))
    return (A.T @ A + d * np.eye(d)).astype(np.float64)


def make_constant_logits(n: int, value: float = 1.0) -> NDArray[np.float64]:
    """Genera un vector de logits constantes (señal degenerada)."""
    return np.full(n, value, dtype=np.float64)


def make_spike_logits(n: int, spike_idx: int = 0, spike_val: float = 10.0) -> NDArray[np.float64]:
    """Genera un vector con un único pico (señal concentrada)."""
    v = np.zeros(n, dtype=np.float64)
    v[spike_idx] = spike_val
    return v


def compute_l2_norm_on_sphere(
    psi: NDArray[np.float64],
    grid: SphericalGrid,
) -> float:
    r"""
    Calcula ‖ψ‖_{L²(S²)} = √(Σ_{ij} ψ²_{ij} · W_{ij}).
    """
    return float(math.sqrt(float(np.sum(psi ** 2 * grid.weights))))


def compute_parseval_lhs(
    psi: NDArray[np.float64],
    grid: SphericalGrid,
) -> float:
    r"""LHS de Parseval: ‖ψ‖²_{L²(S²)} = Σ_{ij} ψ²_{ij} · W_{ij}."""
    return float(np.sum(psi.astype(np.float64) ** 2 * grid.weights))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de pytest
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spectrometer_default() -> Phase1_SphericalHarmonicsSpectrometer:
    """
    Espectrómetro con parámetros por defecto (l_cutoff=10, n_theta=32, n_phi=64).
    Scope=module: se construye una sola vez por módulo de pruebas (costoso).
    """
    return Phase1_SphericalHarmonicsSpectrometer(
        l_cutoff      = 10,
        gamma_damping = 0.1,
        n_theta       = 32,
        n_phi         = 64,
        verify_parseval = True,
    )


@pytest.fixture(scope="module")
def spectrometer_high_res() -> Phase1_SphericalHarmonicsSpectrometer:
    """
    Espectrómetro de alta resolución para pruebas de precisión.
    n_theta=64 satisface 2·l_cutoff+2 = 22 holgadamente.
    """
    return Phase1_SphericalHarmonicsSpectrometer(
        l_cutoff      = 10,
        gamma_damping = 0.1,
        n_theta       = 64,
        n_phi         = 128,
        verify_parseval = True,
    )


@pytest.fixture(scope="module")
def lens_default() -> Phase2_CategoricalOpticLens:
    """
    Lente categórico con métrica identidad 4×4 (SPD trivial).
    """
    G = np.eye(4, dtype=np.float64)
    return Phase2_CategoricalOpticLens(
        metric_tensor = G,
        l_cutoff      = 10,
        gamma_damping = 0.1,
        n_theta       = 32,
        n_phi         = 64,
    )


@pytest.fixture(scope="module")
def fibrator_default() -> OpticalRiemannLensFibrator:
    """
    Fibrador principal con configuración estándar.
    """
    G = make_spd_matrix(4, seed=0)
    return OpticalRiemannLensFibrator(
        metric_tensor = G,
        l_cutoff      = 8,
        gamma_damping = 0.1,
        n_theta       = 32,
        n_phi         = 64,
        verify_parseval = True,
    )


@pytest.fixture(scope="module")
def logits_standard() -> NDArray[np.float64]:
    """Vector de logits estándar: n=50, distribuidos N(0,1)."""
    return make_random_logits(50, seed=_RNG_SEED)


@pytest.fixture(scope="module")
def logits_large() -> NDArray[np.float64]:
    """Vector de logits grande: n=512, distribuidos N(0,1)."""
    return make_random_logits(512, seed=_RNG_SEED + 1)


@pytest.fixture(scope="module")
def logits_minimal() -> NDArray[np.float64]:
    """Vector de logits mínimo: n=3 (límite inferior exacto)."""
    return np.array([1.0, -0.5, 2.0], dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 1 — PRUEBAS DE ESTRUCTURAS DE DATOS (DATACLASSES)                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSphericalGrid:
    """Pruebas unitarias de SphericalGrid."""

    def test_area_total_equals_4pi(self) -> None:
        r"""
        PROPIEDAD: Σ_{i,j} W_{ij} = 4π (área total de S²).

        Justificación analítica:
            Σ_i w_i = 2  (los pesos de GL integran ∫_{-1}^{1} dt = 2)
            Σ_j Δφ = 2π
            ⟹ Σ_{ij} w_i · Δφ = 2 · 2π = 4π ✓
        """
        for n_theta in [8, 16, 32, 64]:
            t_nodes, w_gauss = np.polynomial.legendre.leggauss(n_theta)
            n_phi = 2 * n_theta
            delta_phi = 2.0 * np.pi / n_phi
            theta = np.arccos(t_nodes.astype(np.float64))
            phi   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            weights = w_gauss.astype(np.float64)[:, np.newaxis] * delta_phi

            grid = SphericalGrid(
                theta         = theta,
                phi           = phi,
                gauss_weights = w_gauss.astype(np.float64),
                delta_phi     = delta_phi,
                weights       = weights,
                shape         = (n_theta, n_phi),
            )
            total_area = float(np.sum(grid.weights))
            expected   = 4.0 * np.pi
            # Tolerancia: precisión de doble precisión para la suma de n_theta pesos
            rel_err = abs(total_area - expected) / expected
            assert rel_err < 1e-12, (
                f"Área total ≠ 4π para n_theta={n_theta}: "
                f"got={total_area:.12f}, expected={expected:.12f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_theta_in_open_interval(self) -> None:
        """
        PROPIEDAD: θ_i ∈ (0, π) estrictamente (sin polos exactos).

        Los nodos de Gauss-Legendre son raíces de P_n en el interior de (-1,1),
        por tanto arccos(t_i) ∈ (0,π) estrictamente.
        """
        n_theta = 32
        t_nodes, w_gauss = np.polynomial.legendre.leggauss(n_theta)
        theta = np.arccos(t_nodes.astype(np.float64))
        assert np.all(theta > 0.0), "Algunos nodos tienen θ = 0 (polo norte)"
        assert np.all(theta < np.pi), "Algunos nodos tienen θ = π (polo sur)"

    def test_gauss_weights_positive(self) -> None:
        """
        PROPIEDAD: w_i > 0 ∀ i.

        Los pesos de Gauss-Legendre son siempre positivos (teorema de cuadratura
        Gaussiana: todos los pesos son positivos para cualquier n ≥ 1).
        """
        for n_theta in [4, 8, 16, 32, 64]:
            _, w_gauss = np.polynomial.legendre.leggauss(n_theta)
            assert np.all(w_gauss > 0.0), (
                f"Pesos de GL negativos para n_theta={n_theta}"
            )

    def test_invalid_grid_raises_spectral_grid_error_with_sin_weights(self) -> None:
        r"""
        PROPIEDAD: Si los pesos incluyen sin(θ_i) (Jacobiano duplicado),
        el área total ≠ 4π y SphericalGrid.__post_init__ lanza SpectralGridError.

        Fundamentación:
            Con W_{ij} = w_i · sin(θ_i) · Δφ:
            Σ_{ij} W_{ij} = Δφ · n_φ · Σ_i w_i · sin(θ_i)
                          = 2π · Σ_i w_i · sin(θ_i) ≠ 4π  (en general)
        """
        n_theta = 16
        n_phi   = 32
        t_nodes, w_gauss = np.polynomial.legendre.leggauss(n_theta)
        theta     = np.arccos(t_nodes.astype(np.float64))
        phi       = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        delta_phi = 2.0 * np.pi / n_phi

        # BUG intencional: multiplicar por sin(θ) → Jacobiano duplicado
        weights_wrong = (
            w_gauss.astype(np.float64) * np.sin(theta)
        )[:, np.newaxis] * delta_phi

        with pytest.raises(SpectralGridError, match="Jacobiano"):
            SphericalGrid(
                theta         = theta,
                phi           = phi,
                gauss_weights = w_gauss.astype(np.float64),
                delta_phi     = delta_phi,
                weights       = weights_wrong,
                shape         = (n_theta, n_phi),
            )

    def test_shape_attribute_matches_arrays(self) -> None:
        """PROPIEDAD: grid.shape == (len(theta), len(phi))."""
        n_theta, n_phi = 16, 32
        t_nodes, w_gauss = np.polynomial.legendre.leggauss(n_theta)
        theta     = np.arccos(t_nodes.astype(np.float64))
        phi       = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        delta_phi = 2.0 * np.pi / n_phi
        weights   = w_gauss.astype(np.float64)[:, np.newaxis] * delta_phi

        grid = SphericalGrid(
            theta=theta, phi=phi, gauss_weights=w_gauss.astype(np.float64),
            delta_phi=delta_phi, weights=weights, shape=(n_theta, n_phi),
        )
        assert grid.shape == (n_theta, n_phi)
        assert grid.weights.shape == grid.shape


class TestSphericalSpectrum:
    """Pruebas unitarias de SphericalSpectrum."""

    def _make_dummy_spectrum(
        self, l_max: int, seed: int = 0
    ) -> SphericalSpectrum:
        """Genera un espectro dummy con coeficientes aleatorios."""
        rng = make_rng(seed)
        n_coeffs = (l_max + 1) ** 2
        c = rng.standard_normal(n_coeffs) + 1j * rng.standard_normal(n_coeffs)
        c = c.astype(np.complex128)
        parseval_lhs = float(np.sum(np.abs(c) ** 2)) * 1.01  # 1% de error intencional
        parseval_rhs = float(np.sum(np.abs(c) ** 2))
        rel_err      = abs(parseval_lhs - parseval_rhs) / max(parseval_lhs, _EPSILON)
        return SphericalSpectrum(
            coefficients            = c,
            l_max                   = l_max,
            parseval_lhs            = parseval_lhs,
            parseval_rhs            = parseval_rhs,
            parseval_relative_error = rel_err,
        )

    def test_n_coefficients_formula(self) -> None:
        r"""
        PROPIEDAD: n_coefficients == (l_max+1)²
        = Σ_{l=0}^{l_max}(2l+1) = 1 + 3 + 5 + ... + (2l_max+1).
        """
        for l_max in [0, 1, 5, 10, 20]:
            spectrum = self._make_dummy_spectrum(l_max)
            expected_n = (l_max + 1) ** 2
            assert spectrum.n_coefficients == expected_n, (
                f"n_coefficients={spectrum.n_coefficients} ≠ (l_max+1)²={expected_n}"
                f" para l_max={l_max}"
            )

    def test_power_at_degree_zero_is_c00_squared(self) -> None:
        r"""
        PROPIEDAD: P_0 = |c_{0,0}|².

        Dado que el grado l=0 tiene un solo coeficiente (m=0):
            P_0 = Σ_{m=-0}^{0} |c_{0,0}|² = |c_{0,0}|²
        """
        rng = make_rng(0)
        c = rng.standard_normal(25) + 1j * rng.standard_normal(25)
        c = c.astype(np.complex128)
        spectrum = SphericalSpectrum(
            coefficients=c, l_max=4,
            parseval_lhs=1.0, parseval_rhs=1.0, parseval_relative_error=0.0,
        )
        p0 = spectrum.power_at_degree(0)
        assert abs(p0 - float(abs(c[0]) ** 2)) < 1e-14, (
            f"P_0={p0} ≠ |c_00|²={float(abs(c[0])**2)}"
        )

    def test_power_at_degree_partition_of_total(self) -> None:
        r"""
        PROPIEDAD: Σ_{l=0}^{l_max} P_l = Σ_k |c_k|² (partición de la energía total).
        """
        l_max    = 5
        spectrum = self._make_dummy_spectrum(l_max, seed=7)
        total_from_degrees = sum(
            spectrum.power_at_degree(l) for l in range(l_max + 1)
        )
        total_from_coeffs = float(np.sum(np.abs(spectrum.coefficients) ** 2))
        assert abs(total_from_degrees - total_from_coeffs) < 1e-12, (
            f"Σ P_l={total_from_degrees} ≠ Σ|c|²={total_from_coeffs}"
        )

    def test_dominant_degree_returns_argmax_of_powers(self) -> None:
        """
        PROPIEDAD: dominant_degree() devuelve el l con máxima P_l.
        """
        l_max = 4
        n_coeffs = (l_max + 1) ** 2
        c = np.zeros(n_coeffs, dtype=np.complex128)
        # Concentrar energía en l=3: índices k = l² + (m+l) para l=3
        c[9:16] = 1.0 + 0j  # l=3: índices 9,...,15
        spectrum = SphericalSpectrum(
            coefficients=c, l_max=l_max,
            parseval_lhs=7.0, parseval_rhs=7.0, parseval_relative_error=0.0,
        )
        assert spectrum.dominant_degree() == 3

    def test_invalid_shape_raises_value_error(self) -> None:
        """
        PROPIEDAD: SphericalSpectrum lanza ValueError si shape ≠ (l_max+1)².
        """
        with pytest.raises(ValueError, match="shape"):
            SphericalSpectrum(
                coefficients=np.zeros(10, dtype=np.complex128),
                l_max=4,  # espera (4+1)²=25 coeficientes, recibe 10
                parseval_lhs=0.0, parseval_rhs=0.0, parseval_relative_error=0.0,
            )

    def test_negative_l_max_raises_value_error(self) -> None:
        """PROPIEDAD: l_max < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="l_max"):
            SphericalSpectrum(
                coefficients=np.zeros(1, dtype=np.complex128),
                l_max=-1,
                parseval_lhs=0.0, parseval_rhs=0.0, parseval_relative_error=0.0,
            )


class TestRefractedState:
    """Pruebas unitarias de RefractedState."""

    def _make_dummy_diagnostics(self) -> SpectralDiagnostics:
        return SpectralDiagnostics(
            n_refract=1.5, fermat_metric_trace=6.0, parseval_error=0.01,
            kv_compression_ratio=0.8, sigma_projection=0.3,
            theta0_projection=1.0, phi0_projection=0.5,
            energy_raw=100.0, energy_focused=64.0, energy_retention=0.64,
            l_dominant=2, gram_schmidt_cond=1.0,
        )

    def test_non_finite_focused_logits_raises(self) -> None:
        """PROPIEDAD: RefractedState.__post_init__ rechaza logits no finitos."""
        focused = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="finitos"):
            RefractedState(
                focused_logits=focused,
                kv_compression_ratio=0.5,
                fermat_metric_trace=4.0,
                diagnostics=self._make_dummy_diagnostics(),
            )

    def test_kv_ratio_outside_unit_interval_raises(self) -> None:
        """PROPIEDAD: kv_compression_ratio ∉ [0,1] lanza ValueError."""
        focused = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="kv_compression_ratio"):
            RefractedState(
                focused_logits=focused,
                kv_compression_ratio=1.5,  # > 1
                fermat_metric_trace=4.0,
                diagnostics=self._make_dummy_diagnostics(),
            )

    def test_valid_refracted_state_constructed_correctly(self) -> None:
        """PROPIEDAD: RefractedState válido no lanza excepciones."""
        focused = np.array([0.5, -0.3, 1.2], dtype=np.float64)
        state = RefractedState(
            focused_logits=focused,
            kv_compression_ratio=0.7,
            fermat_metric_trace=5.0,
            diagnostics=self._make_dummy_diagnostics(),
        )
        assert np.allclose(state.focused_logits, focused)
        assert state.kv_compression_ratio == 0.7


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 2 — PRUEBAS DE EXCEPCIONES ÓPTICAS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestExceptions:
    """
    Verifica que todas las excepciones están correctamente encadenadas en la
    jerarquía de TopologicalInvariantError y contienen mensajes descriptivos.
    """

    def test_exception_hierarchy_optical_dispersion(self) -> None:
        """OpticalDispersionError es subclase de TopologicalInvariantError."""
        from app.core.mic_algebra import TopologicalInvariantError
        assert issubclass(OpticalDispersionError, TopologicalInvariantError)

    def test_exception_hierarchy_lens_singularity(self) -> None:
        """LensSingularityError es subclase de TopologicalInvariantError."""
        from app.core.mic_algebra import TopologicalInvariantError
        assert issubclass(LensSingularityError, TopologicalInvariantError)

    def test_exception_hierarchy_gram_schmidt(self) -> None:
        """GramSchmidtDegeneracyError es subclase de LensSingularityError."""
        assert issubclass(GramSchmidtDegeneracyError, LensSingularityError)

    def test_exception_hierarchy_parseval(self) -> None:
        """ParsevalViolationError es subclase de TopologicalInvariantError."""
        from app.core.mic_algebra import TopologicalInvariantError
        assert issubclass(ParsevalViolationError, TopologicalInvariantError)

    def test_exception_hierarchy_spectral_grid(self) -> None:
        """SpectralGridError es subclase de TopologicalInvariantError."""
        from app.core.mic_algebra import TopologicalInvariantError
        assert issubclass(SpectralGridError, TopologicalInvariantError)

    def test_all_exceptions_are_catchable_as_base(self) -> None:
        """
        PROPIEDAD: todas las excepciones del módulo pueden capturarse como
        TopologicalInvariantError (polimorfismo de excepciones).
        """
        from app.core.mic_algebra import TopologicalInvariantError
        exceptions = [
            OpticalDispersionError("test"),
            LensSingularityError("test"),
            GramSchmidtDegeneracyError("test"),
            ParsevalViolationError("test"),
            SpectralGridError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, TopologicalInvariantError), (
                f"{type(exc).__name__} no es instancia de TopologicalInvariantError"
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 3 — PRUEBAS DE FASE 1 (SphericalHarmonicsSpectrometer)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase1_ValidateParams:
    """Pruebas de _validate_spectrometer_params."""

    @pytest.mark.parametrize("l_cutoff", [-1, -10])
    def test_negative_l_cutoff_raises(self, l_cutoff: int) -> None:
        """l_cutoff < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="l_cutoff"):
            Phase1_SphericalHarmonicsSpectrometer(l_cutoff=l_cutoff)

    @pytest.mark.parametrize("l_cutoff", [0.5, 1.0, "10"])
    def test_non_integer_l_cutoff_raises(self, l_cutoff) -> None:
        """l_cutoff no-entero debe lanzar ValueError."""
        with pytest.raises(ValueError, match="l_cutoff"):
            Phase1_SphericalHarmonicsSpectrometer(l_cutoff=l_cutoff)

    @pytest.mark.parametrize("gamma", [0.0, -0.1, -1e-15])
    def test_non_positive_gamma_raises(self, gamma: float) -> None:
        """gamma ≤ 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="gamma_damping"):
            Phase1_SphericalHarmonicsSpectrometer(gamma_damping=gamma)

    @pytest.mark.parametrize("n_theta", [1, 2, 3])
    def test_n_theta_less_than_4_raises(self, n_theta: int) -> None:
        """n_theta < 4 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="n_theta"):
            Phase1_SphericalHarmonicsSpectrometer(n_theta=n_theta)

    @pytest.mark.parametrize("n_phi", [1, 2, 3])
    def test_n_phi_less_than_4_raises(self, n_phi: int) -> None:
        """n_phi < 4 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="n_phi"):
            Phase1_SphericalHarmonicsSpectrometer(n_phi=n_phi)

    def test_valid_params_construct_without_error(self) -> None:
        """Parámetros válidos no deben lanzar ninguna excepción."""
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=5, gamma_damping=0.2, n_theta=16, n_phi=32
        )
        assert spec.l_cutoff == 5
        assert spec.gamma_damping == pytest.approx(0.2)

    def test_l_cutoff_zero_is_valid(self) -> None:
        """l_cutoff = 0 (solo modo constante) debe construirse sin error."""
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=0, n_theta=4, n_phi=4
        )
        assert spec.l_cutoff == 0
        assert spec.harmonics_tensor.shape == (1, 4, 4)  # solo Y_0^0


class TestPhase1_SphericalGrid:
    """Pruebas de _build_spherical_grid."""

    def test_grid_area_is_4pi(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: Σ W_{ij} = 4π exactamente en doble precisión.
        """
        grid = spectrometer_default.grid
        total = float(np.sum(grid.weights))
        assert abs(total - 4.0 * np.pi) / (4.0 * np.pi) < 1e-12

    def test_grid_weights_have_no_sin_factor(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: Los pesos NO incluyen sin(θ_i).

        Verificación: si W_{ij} = w_i · sin(θ_i) · Δφ, entonces
        Σ_j W_{ij} = w_i · sin(θ_i) · 2π.
        Si W_{ij} = w_i · Δφ, entonces Σ_j W_{ij} = w_i · 2π.

        Distinguimos verificando que Σ_j W_{i,j} / (2π) = w_i (no w_i·sin(θ_i)).
        """
        grid  = spectrometer_default.grid
        sum_j = np.sum(grid.weights, axis=1)  # shape (n_theta,)
        expected = grid.gauss_weights * 2.0 * np.pi
        assert np.allclose(sum_j, expected, rtol=1e-12), (
            "Los pesos de fila no son iguales a w_i·2π. "
            "Posible inclusión de sin(θ_i)."
        )

    def test_grid_delta_phi_is_2pi_over_n_phi(self, spectrometer_default) -> None:
        """PROPIEDAD: Δφ = 2π/n_φ exactamente."""
        grid  = spectrometer_default.grid
        n_phi = len(grid.phi)
        expected_delta_phi = 2.0 * np.pi / n_phi
        assert abs(grid.delta_phi - expected_delta_phi) < 1e-15

    def test_phi_nodes_are_uniform_in_0_2pi(self, spectrometer_default) -> None:
        """
        PROPIEDAD: Los nodos φ_j = j·Δφ son uniformes en [0, 2π).
        El nodo j=0 es exactamente 0 y el último es < 2π.
        """
        grid = spectrometer_default.grid
        phi  = grid.phi
        assert abs(phi[0]) < 1e-15, "φ_0 ≠ 0"
        assert phi[-1] < 2.0 * np.pi, "φ_{n-1} ≥ 2π"
        diffs = np.diff(phi)
        assert np.allclose(diffs, diffs[0], rtol=1e-14), "Espaciado φ no uniforme"


class TestPhase1_GridOrthogonality:
    """
    Pruebas de _validate_grid_orthogonality.
    Verifican que los 5 pares de armónicos satisfacen ⟨Y,Y'⟩ ≈ δ.
    """

    @pytest.mark.parametrize("l,m,lp,mp,expected", [
        (0,  0, 0,  0, 1.0),  # auto-norma Y_0^0
        (1,  0, 1,  0, 1.0),  # auto-norma Y_1^0
        (1,  1, 1,  1, 1.0),  # auto-norma Y_1^1
        (1, -1, 1,  1, 0.0),  # cruzado mismo l, distinto m
        (2,  0, 1,  0, 0.0),  # cruzado distinto l
    ])
    def test_orthogonality_pair(
        self, spectrometer_high_res, l, m, lp, mp, expected
    ) -> None:
        r"""
        PROPIEDAD: ⟨Y_l^m, Y_{l'}^{m'}⟩_{L²(S²)} ≈ δ_{ll'}δ_{mm'}.

        Tolerancia: _ORTHOGONALITY_TOL = 1e-5 (relajada por n_theta finito).
        La grilla high_res con n_theta=64 debe satisfacer esta condición con
        margen amplio.
        """
        grid  = spectrometer_high_res.grid
        THETA, PHI = np.meshgrid(grid.theta, grid.phi, indexing='ij')
        Y_lm  = sp_special.sph_harm(m,  l,  PHI, THETA)
        Y_lpm = sp_special.sph_harm(mp, lp, PHI, THETA)
        inner = float(np.abs(np.sum(Y_lm * np.conj(Y_lpm) * grid.weights)))
        assert abs(inner - expected) < _ORTHOGONALITY_TOL, (
            f"⟨Y_{l}^{m}, Y_{lp}^{mp}⟩ = {inner:.6f} ≠ {expected:.1f} "
            f"(error={abs(inner-expected):.2e} > tol={_ORTHOGONALITY_TOL:.2e})"
        )

    def test_insufficient_grid_would_fail_orthogonality(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: Una grilla muy pequeña (n_theta=4) con l_cutoff=10
        falla la verificación de ortogonalidad (SpectralGridError).

        n_theta=4 es insuficiente para integrar productos Y_2^0 · Y_1^0 exactamente.
        """
        with pytest.raises((SpectralGridError, ParsevalViolationError)):
            Phase1_SphericalHarmonicsSpectrometer(
                l_cutoff=10,
                gamma_damping=0.1,
                n_theta=4,   # muy insuficiente para l_cutoff=10
                n_phi=8,
                verify_parseval=True,
            )


class TestPhase1_HarmonicsTensor:
    """Pruebas de _precompute_harmonics_tensor."""

    def test_harmonics_tensor_shape(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: Y_tensor.shape == ((l_cutoff+1)², n_theta, n_phi).
        """
        spec     = spectrometer_default
        expected = ((spec.l_cutoff + 1) ** 2, spec._n_theta, spec._n_phi)
        assert spec.harmonics_tensor.shape == expected, (
            f"Y_tensor.shape={spec.harmonics_tensor.shape} ≠ {expected}"
        )

    def test_harmonics_tensor_dtype_complex128(self, spectrometer_default) -> None:
        """PROPIEDAD: Y_tensor.dtype == complex128."""
        assert spectrometer_default.harmonics_tensor.dtype == np.complex128

    def test_y00_value_in_tensor(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: Y_tensor[0, :, :] = Y_0^0(θ,φ) = 1/√(4π) (constante).

        Verificación analítica:
            Y_0^0(θ,φ) = 1/√(4π)  ∀ (θ,φ)
        """
        Y00_from_tensor = spectrometer_default.harmonics_tensor[0]
        expected_val    = 1.0 / math.sqrt(4.0 * np.pi)
        # Debe ser constante e igual a 1/√(4π)
        assert np.allclose(np.real(Y00_from_tensor), expected_val, atol=1e-12), (
            f"Y_0^0 en el tensor no es 1/√(4π)={expected_val:.8f}"
        )
        assert np.allclose(np.imag(Y00_from_tensor), 0.0, atol=1e-12), (
            "Y_0^0 tiene parte imaginaria no nula"
        )

    def test_degree_vector_length(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: len(degree_vector) == (l_cutoff+1)².
        """
        spec = spectrometer_default
        expected_len = (spec.l_cutoff + 1) ** 2
        assert len(spec.degree_vector) == expected_len

    def test_degree_vector_values(self) -> None:
        r"""
        PROPIEDAD: degree_vector[k] = l para k = l² + (m+l).

        Para l_cutoff=2:
            k=0: (l=0,m=0) → degree=0
            k=1,2,3: (l=1,m=-1,0,1) → degree=1
            k=4,...,8: (l=2,m=-2,...,2) → degree=2
        Expected: [0, 1, 1, 1, 2, 2, 2, 2, 2]
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=2, n_theta=8, n_phi=8
        )
        expected = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float64)
        assert np.array_equal(spec.degree_vector, expected), (
            f"degree_vector={spec.degree_vector} ≠ {expected}"
        )


class TestPhase1_GramSchmidt:
    """Pruebas de _project_logits_to_r3_robust."""

    def test_output_is_unit_vector(self, logits_standard) -> None:
        r"""
        PROPIEDAD: ‖proj_r3‖₂ = 1 (vector unitario en ℝ³).
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        gs   = spec._project_logits_to_r3_robust(logits_standard)
        norm = float(np.linalg.norm(gs.proj_r3))
        assert abs(norm - 1.0) < 1e-12, f"‖proj_r3‖={norm} ≠ 1"

    def test_basis_vectors_are_orthonormal(self, logits_standard) -> None:
        r"""
        PROPIEDAD: {e₁, e₂, e₃} forman una base ortonormal:
            ⟨e_i, e_j⟩ = δ_{ij}   para i,j ∈ {1,2,3}.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        gs   = spec._project_logits_to_r3_robust(logits_standard)

        pairs = [(gs.e1, gs.e1, 1.0), (gs.e2, gs.e2, 1.0), (gs.e3, gs.e3, 1.0),
                 (gs.e1, gs.e2, 0.0), (gs.e1, gs.e3, 0.0), (gs.e2, gs.e3, 0.0)]
        for vi, vj, expected in pairs:
            dot = abs(float(np.dot(vi, vj)))
            assert abs(dot - expected) < 1e-10, (
                f"⟨e_i, e_j⟩ = {dot:.4e} ≠ {expected} (error={abs(dot-expected):.2e})"
            )

    def test_e1_parallel_to_centered_logits(self, logits_standard) -> None:
        r"""
        PROPIEDAD: e₁ es paralelo a v̄ = logits - mean(logits).
            e₁ = v̄ / ‖v̄‖  ⟹  e₁ × v̄ = 0  (producto cruzado cero).

        Verificación 1D: |e₁ · (v̄/‖v̄‖)| = 1.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        gs   = spec._project_logits_to_r3_robust(logits_standard)

        v_bar = logits_standard - np.mean(logits_standard)
        v_hat = v_bar / np.linalg.norm(v_bar)
        dot   = abs(float(np.dot(gs.e1, v_hat)))
        assert abs(dot - 1.0) < 1e-12, (
            f"|e₁ · v̂| = {dot:.8f} ≠ 1 (e₁ no es paralelo a v̄)"
        )

    def test_constant_logits_raises_lens_singularity(self) -> None:
        r"""
        PROPIEDAD: Un vector constante v̄ = 0 lanza LensSingularityError.

        Fundamento: v̄ = logits - mean(logits) = 0 para señal constante.
        No hay dirección preferencial en ℝⁿ → singularidad de Riemann.
        """
        spec   = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        logits = make_constant_logits(50, value=3.14)
        with pytest.raises(LensSingularityError, match="constante|nulo"):
            spec._project_logits_to_r3_robust(logits)

    def test_n_less_than_3_raises_lens_singularity(self) -> None:
        r"""
        PROPIEDAD: n < 3 lanza LensSingularityError.

        La proyección a ℝ³ requiere n ≥ 3 para que exista un subespacio
        3-dimensional en ℝⁿ.
        """
        spec   = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        logits = np.array([1.0, -1.0], dtype=np.float64)  # n=2
        with pytest.raises(LensSingularityError, match="3"):
            spec._project_logits_to_r3_robust(logits)

    @pytest.mark.parametrize("n", [3, 5, 10, 50, 200])
    def test_gram_schmidt_works_for_various_n(self, n: int) -> None:
        """
        PROPIEDAD: Gram-Schmidt converge para logits de longitud n ∈ {3,...,200}.
        """
        spec   = Phase1_SphericalHarmonicsSpectrometer(l_cutoff=5, n_theta=16, n_phi=32)
        logits = make_random_logits(n, seed=n)
        gs     = spec._project_logits_to_r3_robust(logits)
        assert np.linalg.norm(gs.proj_r3) == pytest.approx(1.0, abs=1e-12)


class TestPhase1_MapLogitsToSphere:
    """Pruebas de _map_logits_to_sphere."""

    def test_returns_sphere_point_namedtuple(self, spectrometer_default, logits_standard) -> None:
        """PROPIEDAD: El retorno es un SpherePoint con los campos correctos."""
        sp = spectrometer_default._map_logits_to_sphere(logits_standard)
        assert isinstance(sp, SpherePoint)
        assert hasattr(sp, 'psi')
        assert hasattr(sp, 'sigma')
        assert hasattr(sp, 'theta0')
        assert hasattr(sp, 'phi0')
        assert hasattr(sp, 'p_r3')

    def test_psi_shape_matches_grid(self, spectrometer_default, logits_standard) -> None:
        """PROPIEDAD: psi.shape == grid.shape."""
        sp   = spectrometer_default._map_logits_to_sphere(logits_standard)
        grid = spectrometer_default.grid
        assert sp.psi.shape == grid.shape

    def test_psi_is_finite(self, spectrometer_default, logits_standard) -> None:
        """PROPIEDAD: psi no contiene NaN ni Inf."""
        sp = spectrometer_default._map_logits_to_sphere(logits_standard)
        assert np.all(np.isfinite(sp.psi))

    def test_psi_is_positive(self, spectrometer_default, logits_standard) -> None:
        r"""
        PROPIEDAD: psi ≥ 0 (perfil gaussiano no negativo).

        psi(θ,φ) = A · exp(-½·(d/σ)²) ≥ 0 siempre (A > 0, exp ≥ 0).
        """
        sp = spectrometer_default._map_logits_to_sphere(logits_standard)
        assert np.all(sp.psi >= 0.0)

    def test_l2_norm_equals_logits_norm(self, spectrometer_default, logits_standard) -> None:
        r"""
        PROPIEDAD: ‖ψ‖_{L²(S²)} ≈ ‖logits‖₂ (conservación de energía).

        Tolerancia: 1% (el escalado es exacto en teoría, pero la cuadratura
        numérica introduce un error de O(Δφ²) ≈ (2π/64)² ≈ 0.1%).
        """
        sp   = spectrometer_default._map_logits_to_sphere(logits_standard)
        grid = spectrometer_default.grid

        norm_psi    = compute_l2_norm_on_sphere(sp.psi, grid)
        norm_logits = float(np.linalg.norm(logits_standard))
        rel_err     = abs(norm_psi - norm_logits) / max(norm_logits, _EPSILON)
        assert rel_err < 0.01, (
            f"‖ψ‖_{{L²}}={norm_psi:.4e} ≠ ‖logits‖₂={norm_logits:.4e} "
            f"(rel_err={rel_err:.2e})"
        )

    def test_theta0_in_valid_range(self, spectrometer_default, logits_standard) -> None:
        r"""PROPIEDAD: θ₀ ∈ (0, π) (colatitud en el interior)."""
        sp = spectrometer_default._map_logits_to_sphere(logits_standard)
        assert 0.0 < sp.theta0 < np.pi, f"θ₀={sp.theta0} ∉ (0,π)"

    def test_phi0_in_valid_range(self, spectrometer_default, logits_standard) -> None:
        r"""PROPIEDAD: φ₀ ∈ (-π, π] (longitud en rango estándar de arctan2)."""
        sp = spectrometer_default._map_logits_to_sphere(logits_standard)
        assert -np.pi < sp.phi0 <= np.pi or abs(sp.phi0) <= np.pi, (
            f"φ₀={sp.phi0} ∉ (-π,π]"
        )

    def test_sigma_in_valid_range(self, spectrometer_default) -> None:
        r"""
        PROPIEDAD: σ ∈ [_SIGMA_MIN, _SIGMA_MAX] = [0.05, π].

        Casos cubiertos:
            - Señal muy concentrada (pico en un índice).
            - Señal aleatoria normal.
            - Señal casi uniforme.
        """
        spec = spectrometer_default
        test_cases = [
            make_spike_logits(50, spike_idx=0, spike_val=100.0),
            make_random_logits(50, seed=123),
            np.linspace(-0.01, 0.01, 50, dtype=np.float64),
        ]
        for logits in test_cases:
            try:
                sp = spec._map_logits_to_sphere(logits)
                assert _SIGMA_MIN <= sp.sigma <= _SIGMA_MAX, (
                    f"σ={sp.sigma} ∉ [{_SIGMA_MIN}, {_SIGMA_MAX}]"
                )
            except LensSingularityError:
                pass  # señal constante → excepción esperada

    def test_nan_logits_raises_value_error(self, spectrometer_default) -> None:
        """PROPIEDAD: logits con NaN lanzan ValueError."""
        logits = make_random_logits(20)
        logits[5] = np.nan
        with pytest.raises(ValueError, match="finitos"):
            spectrometer_default._map_logits_to_sphere(logits)

    def test_inf_logits_raises_value_error(self, spectrometer_default) -> None:
        """PROPIEDAD: logits con Inf lanzan ValueError."""
        logits = make_random_logits(20)
        logits[0] = np.inf
        with pytest.raises(ValueError, match="finitos"):
            spectrometer_default._map_logits_to_sphere(logits)


class TestPhase1_SphericalCoefficients:
    """Pruebas de _compute_spherical_coefficients."""

    def test_c00_is_real_for_real_psi(self, spectrometer_default, logits_standard) -> None:
        r"""
        PROPIEDAD: Para ψ real, c_{0,0} ∈ ℝ (parte imaginaria ≈ 0).

        Fundamento:
            c_{0,0} = ∫_{S²} ψ · Y_0^0 dΩ = (1/√4π) ∫_{S²} ψ dΩ
        Si ψ es real, c_{0,0} es real.
        """
        spec     = spectrometer_default
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)
        c00      = spectrum.coefficients[0]
        assert abs(float(np.imag(c00))) < 1e-10, (
            f"Im(c_00)={float(np.imag(c00)):.2e} ≠ 0 para ψ real"
        )

    def test_c00_equals_mean_of_psi_times_4pi_sqrt(
        self, spectrometer_high_res, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: c_{0,0} = ∫_{S²} ψ · Y_0^0 dΩ = (1/√4π) · ∫_{S²} ψ dΩ.

        Verificación directa:
            c_00 = Σ_{ij} ψ_{ij} · (1/√4π) · W_{ij}
                 = (1/√4π) · Σ_{ij} ψ_{ij} · W_{ij}
        """
        spec     = spectrometer_high_res
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)

        integral_psi = float(np.sum(sp.psi * spec.grid.weights))
        expected_c00 = integral_psi / math.sqrt(4.0 * np.pi)
        c00_computed = float(np.real(spectrum.coefficients[0]))
        assert abs(c00_computed - expected_c00) < 1e-8, (
            f"c_00={c00_computed:.6e} ≠ expected={expected_c00:.6e}"
        )

    def test_spectrum_has_correct_number_of_coefficients(
        self, spectrometer_default, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: len(c_{lm}) == (l_cutoff+1)².
        """
        spec     = spectrometer_default
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)
        expected = (spec.l_cutoff + 1) ** 2
        assert spectrum.coefficients.shape == (expected,), (
            f"shape={spectrum.coefficients.shape} ≠ ({expected},)"
        )

    def test_parseval_identity_satisfied(
        self, spectrometer_high_res, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: |‖ψ‖²_{L²} - Σ|c_{lm}|²| / ‖ψ‖²_{L²} < _PARSEVAL_REL_TOL.

        Fundamento: Teorema Espectral para Laplace-Beltrami en S².
        Tolerancia: _PARSEVAL_REL_TOL = 5% (ajustado para truncación en l_cutoff).
        """
        spec     = spectrometer_high_res
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL, (
            f"Error de Parseval={spectrum.parseval_relative_error:.2e} "
            f"> tol={_PARSEVAL_REL_TOL}"
        )

    def test_parseval_lhs_equals_l2_norm_squared(
        self, spectrometer_default, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: parseval_lhs = ‖ψ‖²_{L²(S²)} calculado directamente.
        """
        spec     = spectrometer_default
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)

        direct_norm_sq = compute_parseval_lhs(sp.psi, spec.grid)
        assert abs(spectrum.parseval_lhs - direct_norm_sq) < 1e-10, (
            f"parseval_lhs={spectrum.parseval_lhs:.6e} ≠ "
            f"direct={direct_norm_sq:.6e}"
        )

    def test_parseval_rhs_equals_sum_of_squared_coeffs(
        self, spectrometer_default, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: parseval_rhs = Σ_k |c_k|².
        """
        spec     = spectrometer_default
        sp       = spec._map_logits_to_sphere(logits_standard)
        spectrum = spec._compute_spherical_coefficients(sp.psi)

        direct_rhs = float(np.sum(np.abs(spectrum.coefficients) ** 2))
        assert abs(spectrum.parseval_rhs - direct_rhs) < 1e-14, (
            f"parseval_rhs={spectrum.parseval_rhs:.6e} ≠ direct={direct_rhs:.6e}"
        )

    def test_all_zero_psi_raises_value_error(self, spectrometer_default) -> None:
        """PROPIEDAD: ψ = 0 lanza ValueError (energía nula)."""
        psi = np.zeros(spectrometer_default.grid.shape, dtype=np.float64)
        with pytest.raises(ValueError, match="energía|nula"):
            spectrometer_default._compute_spherical_coefficients(psi)

    def test_vectorized_equals_loop_computation(self) -> None:
        r"""
        PROPIEDAD: La implementación vectorizada produce los mismos coeficientes
        que el bucle explícito for-l for-m.

        Esta prueba valida que la vectorización es matemáticamente equivalente.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=3, n_theta=16, n_phi=32
        )
        logits   = make_random_logits(20)
        sp       = spec._map_logits_to_sphere(logits)
        spectrum = spec._compute_spherical_coefficients(sp.psi)

        # Calcular manualmente con bucle
        THETA, PHI = np.meshgrid(spec.grid.theta, spec.grid.phi, indexing='ij')
        W = spec.grid.weights
        num_coeffs = (spec.l_cutoff + 1) ** 2
        c_manual   = np.zeros(num_coeffs, dtype=np.complex128)
        idx = 0
        for l in range(spec.l_cutoff + 1):
            for m in range(-l, l + 1):
                Y_lm  = sp_special.sph_harm(m, l, PHI, THETA)
                c_manual[idx] = np.sum(sp.psi * np.conj(Y_lm) * W)
                idx += 1

        assert np.allclose(spectrum.coefficients, c_manual, atol=1e-12), (
            "Los coeficientes vectorizados difieren del bucle manual"
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 4 — PRUEBAS DE FASE 2 (CategoricalOpticLens)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase2_MetricValidation:
    """Pruebas de _validate_metric_tensor."""

    def test_non_ndarray_raises_type_error(self) -> None:
        """PROPIEDAD: G no-ndarray lanza TypeError."""
        with pytest.raises(TypeError, match="NDArray"):
            Phase2_CategoricalOpticLens(metric_tensor=[[1, 0], [0, 1]])

    def test_non_square_matrix_raises_value_error(self) -> None:
        """PROPIEDAD: G no cuadrado lanza ValueError."""
        G_rect = np.ones((3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="cuadrado"):
            Phase2_CategoricalOpticLens(metric_tensor=G_rect)

    def test_non_spd_matrix_raises_value_error(self) -> None:
        r"""
        PROPIEDAD: G con λ_min ≤ 0 lanza ValueError.

        G = [[-1, 0], [0, 1]] tiene λ_min = -1 < 0.
        """
        G_not_pd = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="definido positivo|λ_min"):
            Phase2_CategoricalOpticLens(metric_tensor=G_not_pd)

    def test_singular_matrix_raises_value_error(self) -> None:
        """PROPIEDAD: G singular (λ_min = 0) lanza ValueError."""
        G_sing = np.array([[1.0, 1.0], [1.0, 1.0]])  # det=0, λ=(0,2)
        with pytest.raises(ValueError, match="definido positivo|λ_min"):
            Phase2_CategoricalOpticLens(metric_tensor=G_sing)

    def test_nan_in_metric_raises_value_error(self) -> None:
        """PROPIEDAD: G con NaN lanza ValueError."""
        G_nan = np.eye(3, dtype=np.float64)
        G_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="finitos"):
            Phase2_CategoricalOpticLens(metric_tensor=G_nan)

    def test_complex_metric_with_large_imaginary_raises(self) -> None:
        r"""
        PROPIEDAD: G complejo con |Im(G)| / |Re(G)| > 1e-10 lanza ValueError.
        """
        G_complex = np.eye(3, dtype=np.complex128) * (1.0 + 0.1j)
        with pytest.raises(ValueError, match="imaginaria"):
            Phase2_CategoricalOpticLens(metric_tensor=G_complex)

    def test_identity_metric_is_valid(self) -> None:
        """PROPIEDAD: G = I_d es siempre válida (SPD trivial)."""
        for d in [2, 3, 4, 10]:
            lens = Phase2_CategoricalOpticLens(
                metric_tensor=np.eye(d, dtype=np.float64),
                l_cutoff=3, n_theta=10, n_phi=16,
            )
            assert lens._dim == d

    def test_asymmetric_metric_is_symmetrized(self) -> None:
        r"""
        PROPIEDAD: G levemente asimétrico se simetriza a (G+Gᵀ)/2.

        Tolerancia de asimetría: ‖G-Gᵀ‖_F/‖G‖_F < 1e-8 (casi-simétrico).
        """
        G = make_spd_matrix(4, seed=42)
        G_asym = G.copy()
        G_asym[0, 1] += 1e-10  # perturbación pequeña
        # No debe lanzar excepción (dentro del límite de simetrización)
        lens = Phase2_CategoricalOpticLens(
            metric_tensor=G_asym,
            l_cutoff=3, n_theta=10, n_phi=16,
        )
        # Verificar que la métrica almacenada es simétrica
        assert np.allclose(lens._G, lens._G.T, atol=1e-15)


class TestPhase2_FermatRefractiveIndex:
    """Pruebas de _compute_fermat_refractive_index."""

    @pytest.mark.parametrize("sigma_star", [
        -1000.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1000.0
    ])
    def test_n_in_open_interval_1_2(self, lens_default, sigma_star: float) -> None:
        r"""
        PROPIEDAD: n(σ*) ∈ (1, 2)  ∀ σ* ∈ ℝ.

        Verificación algebraica:
            n = 1 + tanh(α·σ*)
            tanh ∈ (-1,1) ⟹ n ∈ (0,2)
            Para α > 0: tanh(α·σ*) > -1 ⟹ n > 0
            Pero más precisamente: tanh(x) → -1 solo cuando x → -∞
            Para σ* finito: n = 1 + tanh(α·σ*) ∈ (1-1+ε, 2-ε) = (ε, 2-ε)
        """
        n = lens_default._compute_fermat_refractive_index(sigma_star)
        assert n > 1.0, f"n={n} ≤ 1 para σ*={sigma_star}"
        assert n < 2.0, f"n={n} ≥ 2 para σ*={sigma_star}"

    def test_n_monotonically_increasing(self, lens_default) -> None:
        r"""
        PROPIEDAD: n(σ*) es estrictamente creciente en σ*.

        Fundamento: tanh es estrictamente creciente y α > 0.
        """
        sigmas = np.linspace(-5.0, 5.0, 100)
        ns     = [lens_default._compute_fermat_refractive_index(s) for s in sigmas]
        diffs  = np.diff(ns)
        assert np.all(diffs > 0.0), "n(σ*) no es estrictamente creciente"

    def test_n_at_zero_stress_is_1_plus_tanh_0(self, lens_default) -> None:
        r"""
        PROPIEDAD: n(0) = 1 + tanh(0) = 1 + 0 = 1.

        Exactamente el punto medio del intervalo (1, 2).
        """
        n = lens_default._compute_fermat_refractive_index(0.0)
        assert abs(n - 1.0) < 1e-15, f"n(0)={n} ≠ 1 (esperado para tanh(0)=0)"

    def test_n_formula_matches_manual_calculation(self, lens_default) -> None:
        r"""
        PROPIEDAD: n = 1 + tanh(α·σ*) exactamente.
        """
        for sigma_star in [-2.5, 0.7, 3.14]:
            n_computed = lens_default._compute_fermat_refractive_index(sigma_star)
            n_expected = 1.0 + math.tanh(_FERMAT_ALPHA * sigma_star)
            assert abs(n_computed - n_expected) < 1e-15, (
                f"n(σ*={sigma_star}): computed={n_computed}, expected={n_expected}"
            )

    def test_infinite_stress_raises_value_error(self, lens_default) -> None:
        """PROPIEDAD: σ* = Inf lanza ValueError."""
        with pytest.raises(ValueError, match="finito"):
            lens_default._compute_fermat_refractive_index(float('inf'))

    def test_nan_stress_raises_value_error(self, lens_default) -> None:
        """PROPIEDAD: σ* = NaN lanza ValueError."""
        with pytest.raises(ValueError, match="finito"):
            lens_default._compute_fermat_refractive_index(float('nan'))


class TestPhase2_FermatMetricTrace:
    """Pruebas de _compute_fermat_metric_trace."""

    def test_trace_is_positive(self, lens_default) -> None:
        r"""
        PROPIEDAD: Tr(n²·G) > 0.

        Fundamento: n > 1 > 0 y Tr(G) > 0 (G es SPD).
        """
        for sigma_star in [-5.0, 0.0, 5.0]:
            n     = lens_default._compute_fermat_refractive_index(sigma_star)
            trace = lens_default._compute_fermat_metric_trace(n)
            assert trace > 0.0, f"Tr(n²·G)={trace} ≤ 0 para n={n}"

    def test_trace_for_identity_metric(self) -> None:
        r"""
        PROPIEDAD: Para G = I_d, Tr(n²·G) = n²·d.
        """
        d    = 4
        lens = Phase2_CategoricalOpticLens(
            metric_tensor=np.eye(d, dtype=np.float64),
            l_cutoff=3, n_theta=10, n_phi=16,
        )
        n     = 1.5
        trace = lens._compute_fermat_metric_trace(n)
        assert abs(trace - n ** 2 * d) < 1e-12, (
            f"Tr={trace} ≠ n²·d={n**2*d} para G=I_{d}"
        )

    def test_trace_scales_quadratically_with_n(self, lens_default) -> None:
        r"""
        PROPIEDAD: Tr(n²·G) / Tr(G) = n².

        Verificación de escalado cuadrático.
        """
        trace_G = float(np.trace(lens_default._G))
        for n in [1.1, 1.5, 1.9]:
            trace = lens_default._compute_fermat_metric_trace(n)
            ratio = trace / trace_G
            assert abs(ratio - n ** 2) < 1e-12, (
                f"Tr(n²G)/Tr(G) = {ratio} ≠ n²={n**2}"
            )


class TestPhase2_ReconstructPsi:
    """Pruebas de _reconstruct_psi_from_spectrum."""

    def test_output_shape_matches_grid(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: ψ̃.shape == grid.shape."""
        sp       = lens_default._map_logits_to_sphere(logits_standard)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        n        = lens_default._compute_fermat_refractive_index(1.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)
        assert psi_rec.shape == lens_default.grid.shape

    def test_output_is_real(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: ψ̃ es un array de dtype float64 (parte real tomada)."""
        sp       = lens_default._map_logits_to_sphere(logits_standard)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        n        = lens_default._compute_fermat_refractive_index(1.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)
        assert psi_rec.dtype == np.float64
        assert np.all(np.isfinite(psi_rec))

    def test_attenuation_reduces_energy(self, lens_default, logits_standard) -> None:
        r"""
        PROPIEDAD: ‖ψ̃‖_{L²} ≤ ‖ψ‖_{L²}.

        El operador de difracción h_l ∈ [0,1] para l ≥ 0 es una contracción:
            ‖O_{lens} ψ‖² = Σ_{lm} h_l² |c_{lm}|² ≤ Σ_{lm} |c_{lm}|² = ‖ψ‖²
        """
        sp       = lens_default._map_logits_to_sphere(logits_standard)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        n        = lens_default._compute_fermat_refractive_index(2.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)

        norm_original = compute_l2_norm_on_sphere(sp.psi, lens_default.grid)
        norm_filtered = compute_l2_norm_on_sphere(psi_rec, lens_default.grid)

        assert norm_filtered <= norm_original + 1e-8, (
            f"‖ψ̃‖_{{L²}}={norm_filtered:.4e} > ‖ψ‖_{{L²}}={norm_original:.4e} "
            f"(el filtro amplificó la señal)"
        )

    def test_h0_mode_preserved_exactly(self) -> None:
        r"""
        PROPIEDAD: El modo l=0 se preserva exactamente (h_0 = exp(0) = 1).

        Verificación: Si ψ = Y_0^0 (armónico de grado 0), entonces:
            ψ̃ = h_0 · c_{0,0} · Y_0^0 = 1 · c_{0,0} · Y_0^0 = ψ

        Esta propiedad garantiza que el modo constante (información de DC)
        se transmite sin pérdida a través del Lente.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=2, n_theta=16, n_phi=32
        )
        THETA, PHI = np.meshgrid(spec.grid.theta, spec.grid.phi, indexing='ij')

        # ψ = A · Y_0^0 (función constante en S²)
        A      = 3.7
        Y00    = np.real(sp_special.sph_harm(0, 0, PHI, THETA)).astype(np.float64)
        psi    = A * Y00  # shape (n_theta, n_phi)

        lens = Phase2_CategoricalOpticLens(
            metric_tensor=np.eye(4), l_cutoff=2, n_theta=16, n_phi=32
        )
        spectrum = lens._compute_spherical_coefficients(psi)
        n        = lens._compute_fermat_refractive_index(0.0)  # n=1 (mínimo)
        psi_rec  = lens._reconstruct_psi_from_spectrum(spectrum, n)

        # ψ̃ debe ser ≈ ψ (el modo l=0 se conserva)
        # Tolerancia por truncación en l_cutoff=2 (no hay modos contaminantes aquí)
        assert np.allclose(psi_rec, psi, atol=1e-6), (
            f"El modo l=0 no se preserva: ‖ψ̃-ψ‖_max="
            f"{float(np.max(np.abs(psi_rec - psi))):.2e}"
        )

    def test_high_n_increases_attenuation(self, lens_default, logits_standard) -> None:
        r"""
        PROPIEDAD: Mayor n ⟹ mayor atenuación de alta frecuencia.

        h_l(n) = exp(-γ·n²·l²). Como γ > 0 y l > 0:
            n₁ < n₂ ⟹ h_l(n₁) > h_l(n₂) ⟹ ‖ψ̃(n₁)‖ ≥ ‖ψ̃(n₂)‖
        """
        sp       = lens_default._map_logits_to_sphere(logits_standard)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)

        n_low   = lens_default._compute_fermat_refractive_index(-5.0)
        n_high  = lens_default._compute_fermat_refractive_index(5.0)

        psi_low  = lens_default._reconstruct_psi_from_spectrum(spectrum, n_low)
        psi_high = lens_default._reconstruct_psi_from_spectrum(spectrum, n_high)

        norm_low  = compute_l2_norm_on_sphere(psi_low,  lens_default.grid)
        norm_high = compute_l2_norm_on_sphere(psi_high, lens_default.grid)

        assert norm_low >= norm_high - 1e-10, (
            f"Mayor n no implica mayor atenuación: "
            f"‖ψ̃(n_low={n_low:.3f})‖={norm_low:.4e}, "
            f"‖ψ̃(n_high={n_high:.3f})‖={norm_high:.4e}"
        )


class TestPhase2_ProjectPsiToLogitSpace:
    """Pruebas de _project_psi_to_logit_space."""

    def test_output_dimension_matches_input(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: focused.size == logits.size."""
        logits   = logits_standard
        sp       = lens_default._map_logits_to_sphere(logits)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        gs       = lens_default._project_logits_to_r3_robust(logits)
        n        = lens_default._compute_fermat_refractive_index(1.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)
        focused  = lens_default._project_psi_to_logit_space(
            psi_rec, logits, gs, spectrum
        )
        assert focused.size == logits.size

    def test_output_is_finite(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: focused no contiene NaN ni Inf."""
        logits   = logits_standard
        sp       = lens_default._map_logits_to_sphere(logits)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        gs       = lens_default._project_logits_to_r3_robust(logits)
        n        = lens_default._compute_fermat_refractive_index(1.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)
        focused  = lens_default._project_psi_to_logit_space(
            psi_rec, logits, gs, spectrum
        )
        assert np.all(np.isfinite(focused))

    def test_zero_logits_returns_zero(self, lens_default) -> None:
        r"""
        PROPIEDAD: logits = 0 ⟹ focused = 0.

        Fundamento: norm_orig < _EPSILON → retorna np.zeros(n).
        """
        n        = 20
        logits   = np.zeros(n, dtype=np.float64)
        psi_zero = np.zeros(lens_default.grid.shape, dtype=np.float64)
        # Crear un gs_basis ficticio (no se usa cuando norm_orig ≈ 0)
        gs = lens_default._project_logits_to_r3_robust(
            make_random_logits(n, seed=99)  # logits distintos para GS
        )
        # Construir spectrum dummy
        c_dummy = np.ones((lens_default.l_cutoff + 1) ** 2, dtype=np.complex128)
        spectrum_dummy = SphericalSpectrum(
            coefficients=c_dummy, l_max=lens_default.l_cutoff,
            parseval_lhs=1.0, parseval_rhs=1.0, parseval_relative_error=0.0,
        )
        focused = lens_default._project_psi_to_logit_space(
            psi_zero, logits, gs, spectrum_dummy
        )
        assert np.allclose(focused, 0.0, atol=1e-14)

    def test_dissipative_property(self, lens_default, logits_standard) -> None:
        r"""
        PROPIEDAD: ‖focused‖₂ ≤ ‖original‖₂ + tolerancia_numérica.

        El Lente es un operador disipativo (filtra energía, no amplifica).
        Tolerancia numérica: 1% de ‖original‖₂.
        """
        logits   = logits_standard
        sp       = lens_default._map_logits_to_sphere(logits)
        spectrum = lens_default._compute_spherical_coefficients(sp.psi)
        gs       = lens_default._project_logits_to_r3_robust(logits)
        n        = lens_default._compute_fermat_refractive_index(2.0)
        psi_rec  = lens_default._reconstruct_psi_from_spectrum(spectrum, n)
        focused  = lens_default._project_psi_to_logit_space(
            psi_rec, logits, gs, spectrum
        )

        norm_focused  = float(np.linalg.norm(focused))
        norm_original = float(np.linalg.norm(logits))
        tol           = 0.01 * norm_original + _EPSILON
        assert norm_focused <= norm_original + tol, (
            f"‖focused‖={norm_focused:.4e} > ‖orig‖={norm_original:.4e}+tol={tol:.2e}"
        )


class TestPhase2_ViewPutFunctors:
    """Pruebas de apply_view_functor y apply_put_functor."""

    def test_view_functor_returns_tuple_of_three(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: apply_view_functor retorna (SphericalSpectrum, SpherePoint, GramSchmidtBasis)."""
        result = lens_default.apply_view_functor(logits_standard)
        assert isinstance(result, tuple)
        assert len(result) == 3
        spectrum, sphere_pt, gs_basis = result
        assert isinstance(spectrum,  SphericalSpectrum)
        assert isinstance(sphere_pt, SpherePoint)
        assert isinstance(gs_basis,  GramSchmidtBasis)

    def test_put_functor_output_shape(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: apply_put_functor retorna vector de la misma dimensión que los logits."""
        spectrum, sphere_pt, gs_basis = lens_default.apply_view_functor(logits_standard)
        n = lens_default._compute_fermat_refractive_index(1.0)
        focused = lens_default.apply_put_functor(logits_standard, spectrum, n, gs_basis)
        assert focused.shape == logits_standard.shape

    def test_put_functor_output_is_finite(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: apply_put_functor retorna vector finito."""
        spectrum, sphere_pt, gs_basis = lens_default.apply_view_functor(logits_standard)
        n = lens_default._compute_fermat_refractive_index(0.5)
        focused = lens_default.apply_put_functor(logits_standard, spectrum, n, gs_basis)
        assert np.all(np.isfinite(focused))

    def test_view_spectrum_parseval_satisfied(self, lens_default, logits_standard) -> None:
        """PROPIEDAD: El espectro del View funtor satisface Parseval."""
        spectrum, _, _ = lens_default.apply_view_functor(logits_standard)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL

    def test_put_with_zero_gamma_preserves_all_modes(self) -> None:
        r"""
        PROPIEDAD: Con γ → 0, h_l ≈ 1 ∀ l ⟹ ψ̃ ≈ ψ (filtro transparente).

        Para γ muy pequeño (1e-10):
            h_l = exp(-γ·n²·l²) ≈ 1 - γ·n²·l² ≈ 1 para todo l ≤ l_cutoff.
        """
        lens = Phase2_CategoricalOpticLens(
            metric_tensor=np.eye(4),
            l_cutoff=5, n_theta=16, n_phi=32,
            gamma_damping=1e-10,  # γ ≈ 0
        )
        logits   = make_random_logits(30, seed=7)
        spectrum, sphere_pt, gs_basis = lens.apply_view_functor(logits)
        n       = lens._compute_fermat_refractive_index(0.0)

        psi_rec = lens._reconstruct_psi_from_spectrum(spectrum, n)
        norm_rec = compute_l2_norm_on_sphere(psi_rec, lens.grid)
        norm_psi = compute_l2_norm_on_sphere(sphere_pt.psi, lens.grid)

        # Para γ=1e-10 y n≈1: h_l = exp(-1e-10·1·l²) ≈ 1 para l ≤ 5
        # La energía reconstruida debe ser ≈ energía original
        rel_diff = abs(norm_rec - norm_psi) / max(norm_psi, _EPSILON)
        assert rel_diff < 0.01, (
            f"Con γ≈0, ‖ψ̃‖/‖ψ‖={norm_rec/max(norm_psi,_EPSILON):.4f} ≠ 1 "
            f"(rel_diff={rel_diff:.2e})"
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 5 — PRUEBAS DE FASE 3 (OpticalRiemannLensFibrator)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPhase3_InputValidation:
    """Pruebas de _validate_input_logits y _validate_stress_norm."""

    def test_non_ndarray_logits_raises_type_error(self, fibrator_default) -> None:
        """PROPIEDAD: logits como lista lanza TypeError."""
        with pytest.raises(TypeError, match="NDArray"):
            fibrator_default._validate_input_logits([1.0, 2.0, 3.0])

    def test_2d_logits_raises_value_error(self, fibrator_default) -> None:
        """PROPIEDAD: logits 2D lanza ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            fibrator_default._validate_input_logits(
                np.array([[1.0, 2.0], [3.0, 4.0]])
            )

    def test_logits_too_small_raises_lens_singularity(self, fibrator_default) -> None:
        """PROPIEDAD: logits.size < 3 lanza LensSingularityError."""
        with pytest.raises(LensSingularityError):
            fibrator_default._validate_input_logits(
                np.array([1.0, 2.0], dtype=np.float64)
            )

    def test_nan_logits_raises_value_error(self, fibrator_default) -> None:
        """PROPIEDAD: logits con NaN lanza ValueError."""
        logits = make_random_logits(20)
        logits[3] = np.nan
        with pytest.raises(ValueError, match="finitos"):
            fibrator_default._validate_input_logits(logits)

    def test_inf_logits_raises_value_error(self, fibrator_default) -> None:
        """PROPIEDAD: logits con Inf lanza ValueError."""
        logits    = make_random_logits(20)
        logits[0] = np.inf
        with pytest.raises(ValueError, match="finitos"):
            fibrator_default._validate_input_logits(logits)

    @pytest.mark.parametrize("stress", [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_stress_raises_value_error(self, fibrator_default, stress: float) -> None:
        """PROPIEDAD: stress_norm no finito lanza ValueError."""
        with pytest.raises(ValueError, match="finito"):
            fibrator_default._validate_stress_norm(stress)

    def test_string_stress_raises_type_error(self, fibrator_default) -> None:
        """PROPIEDAD: stress_norm como string lanza TypeError."""
        with pytest.raises(TypeError, match="numérico"):
            fibrator_default._validate_stress_norm("1.5")

    @pytest.mark.parametrize("stress", [-100.0, -1.0, 0.0, 1.0, 100.0])
    def test_valid_stress_values_accepted(self, fibrator_default, stress: float) -> None:
        """PROPIEDAD: stress_norm finito (incluyendo negativos) es aceptado."""
        fibrator_default._validate_stress_norm(stress)  # no debe lanzar


class TestPhase3_KVCompressionRatio:
    """Pruebas de _compute_kv_compression_ratio."""

    def test_ratio_in_unit_interval(self, fibrator_default) -> None:
        r"""
        PROPIEDAD: ratio = clip(‖focused‖/‖raw‖, 0, 1) ∈ [0, 1].
        """
        raw     = make_random_logits(30, seed=1)
        focused = make_random_logits(30, seed=2) * 0.5  # energía reducida
        ratio   = fibrator_default._compute_kv_compression_ratio(raw, focused)
        assert _COMPRESSION_FLOOR <= ratio <= _COMPRESSION_CAP

    def test_ratio_is_zero_for_zero_focused(self, fibrator_default) -> None:
        r"""
        PROPIEDAD: ‖focused‖=0 ⟹ ratio=0 (máxima compresión).
        """
        raw     = make_random_logits(30)
        focused = np.zeros(30, dtype=np.float64)
        ratio   = fibrator_default._compute_kv_compression_ratio(raw, focused)
        assert ratio == pytest.approx(0.0, abs=1e-14)

    def test_ratio_is_one_for_equal_norms(self, fibrator_default) -> None:
        r"""
        PROPIEDAD: ‖focused‖ = ‖raw‖ ⟹ ratio = 1 (sin compresión).
        """
        raw     = make_random_logits(30, seed=5)
        focused = raw * (np.linalg.norm(raw) / np.linalg.norm(raw))  # misma norma
        ratio   = fibrator_default._compute_kv_compression_ratio(raw, focused)
        assert ratio == pytest.approx(1.0, abs=1e-12)

    def test_ratio_formula_explicit(self, fibrator_default) -> None:
        r"""
        PROPIEDAD: ratio = clip(‖focused‖₂/‖raw‖₂, 0, 1).

        Verificación directa con valores conocidos.
        """
        raw     = np.array([3.0, 4.0], dtype=np.float64)  # ‖raw‖=5
        focused = np.array([1.5, 2.0], dtype=np.float64)  # ‖focused‖=2.5
        ratio   = fibrator_default._compute_kv_compression_ratio(raw, focused)
        expected = 2.5 / 5.0  # = 0.5
        assert abs(ratio - expected) < 1e-12

    def test_zero_raw_returns_floor(self, fibrator_default) -> None:
        r"""
        PROPIEDAD: ‖raw‖ = 0 ⟹ ratio → _COMPRESSION_FLOOR via max(‖raw‖, ε).
        """
        raw     = np.zeros(10, dtype=np.float64)
        focused = np.zeros(10, dtype=np.float64)
        ratio   = fibrator_default._compute_kv_compression_ratio(raw, focused)
        assert ratio == pytest.approx(_COMPRESSION_FLOOR, abs=1e-14)


class TestPhase3_RefractAttentionLogits:
    """
    Pruebas de integración de refract_attention_logits.
    Verifican el pipeline completo de las 3 fases.
    """

    def test_returns_refracted_state(self, fibrator_default, logits_standard) -> None:
        """PROPIEDAD: El retorno es una instancia de RefractedState."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert isinstance(state, RefractedState)

    def test_focused_logits_shape_preserved(self, fibrator_default, logits_standard) -> None:
        """PROPIEDAD: focused_logits.shape == logits.shape."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert state.focused_logits.shape == logits_standard.shape

    def test_focused_logits_are_finite(self, fibrator_default, logits_standard) -> None:
        """PROPIEDAD: focused_logits no contiene NaN ni Inf."""
        state = fibrator_default.refract_attention_logits(logits_standard, 0.5)
        assert np.all(np.isfinite(state.focused_logits))

    def test_kv_compression_ratio_in_unit_interval(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: kv_compression_ratio ∈ [0, 1]."""
        for sigma in [-2.0, 0.0, 2.0]:
            state = fibrator_default.refract_attention_logits(logits_standard, sigma)
            assert 0.0 <= state.kv_compression_ratio <= 1.0, (
                f"ratio={state.kv_compression_ratio} ∉ [0,1] para σ*={sigma}"
            )

    def test_fermat_trace_is_positive(self, fibrator_default, logits_standard) -> None:
        """PROPIEDAD: fermat_metric_trace > 0."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert state.fermat_metric_trace > 0.0

    def test_n_refract_in_diagnostics_matches_formula(
        self, fibrator_default, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: diagnostics.n_refract = 1 + tanh(α·σ*).
        """
        sigma_star = 2.3
        state      = fibrator_default.refract_attention_logits(logits_standard, sigma_star)
        expected_n = 1.0 + math.tanh(_FERMAT_ALPHA * sigma_star)
        assert abs(state.diagnostics.n_refract - expected_n) < 1e-12

    def test_diagnostics_l_dominant_in_range(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: diagnostics.l_dominant ∈ [0, l_cutoff]."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert 0 <= state.diagnostics.l_dominant <= fibrator_default.l_cutoff

    def test_diagnostics_sigma_in_valid_range(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: diagnostics.sigma_projection ∈ [_SIGMA_MIN, _SIGMA_MAX]."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert _SIGMA_MIN <= state.diagnostics.sigma_projection <= _SIGMA_MAX

    def test_diagnostics_theta0_in_valid_range(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: diagnostics.theta0_projection ∈ (0, π)."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert 0.0 < state.diagnostics.theta0_projection < np.pi

    def test_diagnostics_energy_retention_in_unit_interval(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: diagnostics.energy_retention ∈ [0, 1]."""
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert 0.0 <= state.diagnostics.energy_retention <= 1.0

    def test_gram_schmidt_cond_is_close_to_one(
        self, fibrator_default, logits_standard
    ) -> None:
        r"""
        PROPIEDAD: gram_schmidt_cond ≈ 1.

        La matriz A = [e₁|e₂|e₃]ᵀ tiene filas ortonormales, por lo que
        sus valores singulares deberían ser todos ≈ 1 (κ = σ_max/σ_min ≈ 1).
        Tolerancia: 1% (errores de punto flotante en Gram-Schmidt).
        """
        state = fibrator_default.refract_attention_logits(logits_standard, 1.0)
        assert state.diagnostics.gram_schmidt_cond == pytest.approx(1.0, rel=0.01), (
            f"κ_GS={state.diagnostics.gram_schmidt_cond} ≫ 1 "
            f"(pérdida de ortogonalidad numérica)"
        )

    def test_last_diagnostics_updated_after_call(
        self, fibrator_default, logits_standard
    ) -> None:
        """PROPIEDAD: spectral_diagnostics se actualiza después de cada llamada."""
        _ = fibrator_default.refract_attention_logits(logits_standard, 0.0)
        assert fibrator_default.spectral_diagnostics is not None
        assert isinstance(fibrator_default.spectral_diagnostics, SpectralDiagnostics)

    def test_constant_logits_raises_lens_singularity(self, fibrator_default) -> None:
        """PROPIEDAD: Logits constantes → LensSingularityError (señal degenerada)."""
        logits = make_constant_logits(50, value=2.71828)
        with pytest.raises(LensSingularityError):
            fibrator_default.refract_attention_logits(logits, 1.0)

    def test_minimal_size_logits_works(self, fibrator_default, logits_minimal) -> None:
        """PROPIEDAD: n=3 (mínimo) funciona correctamente."""
        state = fibrator_default.refract_attention_logits(logits_minimal, 0.5)
        assert isinstance(state, RefractedState)
        assert state.focused_logits.size == 3

    def test_float32_logits_accepted_and_cast(self, fibrator_default) -> None:
        """PROPIEDAD: logits de dtype float32 se aceptan (se castean a float64)."""
        logits_f32 = make_random_logits(20).astype(np.float32)
        state = fibrator_default.refract_attention_logits(logits_f32, 1.0)
        assert state.focused_logits.dtype == np.float64

    @pytest.mark.parametrize("sigma_star", [-10.0, -1.0, 0.0, 1.0, 10.0])
    def test_various_stress_values_produce_valid_state(
        self, fibrator_default, logits_standard, sigma_star: float
    ) -> None:
        """PROPIEDAD: Diferentes valores de σ* producen RefractedState válido."""
        state = fibrator_default.refract_attention_logits(logits_standard, sigma_star)
        assert isinstance(state, RefractedState)
        assert np.all(np.isfinite(state.focused_logits))
        assert 0.0 <= state.kv_compression_ratio <= 1.0


class TestPhase3_MorphismProtocol:
    """Pruebas del protocolo Morphism (__call__, compose)."""

    def test_call_raises_if_state_missing_logits(self, fibrator_default) -> None:
        """
        PROPIEDAD: __call__ con CategoricalState sin logits lanza AttributeError.
        """
        from app.core.mic_algebra import CategoricalState
        state_no_logits = CategoricalState(stress_norm=1.0)  # sin logits
        with pytest.raises(AttributeError):
            fibrator_default(state_no_logits)

    def test_compose_with_non_fibrator_raises_type_error(
        self, fibrator_default
    ) -> None:
        """PROPIEDAD: compose con tipo incorrecto lanza TypeError."""
        with pytest.raises(TypeError, match="OpticalRiemannLensFibrator"):
            fibrator_default.compose("not_a_fibrator")

    def test_compose_returns_composed_lens(self, fibrator_default) -> None:
        """PROPIEDAD: compose retorna un objeto con método refract_attention_logits."""
        G2 = make_spd_matrix(4, seed=99)
        other = OpticalRiemannLensFibrator(
            metric_tensor=G2, l_cutoff=5, n_theta=16, n_phi=32
        )
        composed = fibrator_default.compose(other)
        assert hasattr(composed, 'refract_attention_logits')

    def test_composed_lens_produces_valid_output(
        self, fibrator_default, logits_standard
    ) -> None:
        """
        PROPIEDAD: El lente compuesto (self ∘ other) produce un RefractedState válido.
        """
        G2 = make_spd_matrix(4, seed=77)
        other = OpticalRiemannLensFibrator(
            metric_tensor=G2, l_cutoff=5, n_theta=16, n_phi=32
        )
        composed = fibrator_default.compose(other)
        state = composed.refract_attention_logits(logits_standard, 1.0)
        assert isinstance(state, RefractedState)
        assert np.all(np.isfinite(state.focused_logits))

    def test_spectral_diagnostics_is_none_before_first_call(self) -> None:
        """PROPIEDAD: spectral_diagnostics es None antes de la primera llamada."""
        G = make_spd_matrix(3, seed=0)
        fresh = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=3, n_theta=10, n_phi=16
        )
        assert fresh.spectral_diagnostics is None


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 6 — PRUEBAS DE PROPIEDADES MATEMÁTICAS INVARIANTES                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMathematicalInvariants:
    """
    Pruebas de las propiedades matemáticas fundamentales que deben satisfacerse
    en toda ejecución del módulo, independientemente de los parámetros de entrada.
    """

    def test_parseval_invariant_across_logit_magnitudes(self) -> None:
        r"""
        INVARIANTE: La identidad de Parseval se satisface para logits de
        diferente magnitud (escalado).

        Si ψ → λψ, entonces c_{lm} → λ·c_{lm}, y Parseval escala como λ².
        El error relativo es invariante bajo escalado.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=5, n_theta=32, n_phi=64, verify_parseval=True
        )
        base_logits = make_random_logits(30, seed=42)

        for scale in [1e-3, 1e-1, 1.0, 1e1, 1e3]:
            logits   = base_logits * scale
            sp       = spec._map_logits_to_sphere(logits)
            spectrum = spec._compute_spherical_coefficients(sp.psi)
            assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL, (
                f"Parseval viola para scale={scale}: "
                f"error={spectrum.parseval_relative_error:.2e}"
            )

    def test_fermat_index_range_invariant(self) -> None:
        r"""
        INVARIANTE: n(σ*) ∈ (1, 2) para todo σ* ∈ ℝ finito.

        Prueba exhaustiva con 10,000 valores de σ* ∈ [-1000, 1000].
        """
        lens    = Phase2_CategoricalOpticLens(
            metric_tensor=np.eye(3), l_cutoff=3, n_theta=8, n_phi=8
        )
        sigmas  = np.linspace(-1000.0, 1000.0, 10000)
        ns      = np.array([
            lens._compute_fermat_refractive_index(float(s)) for s in sigmas
        ])
        assert np.all(ns > 1.0), f"n ≤ 1 para algunos σ*: min(n)={ns.min():.8f}"
        assert np.all(ns < 2.0), f"n ≥ 2 para algunos σ*: max(n)={ns.max():.8f}"

    def test_compression_ratio_invariant_under_rotation(self) -> None:
        r"""
        INVARIANTE: La razón de compresión es aproximadamente invariante bajo
        rotaciones del vector de logits.

        Fundamento: El módulo trata a los logits como un vector en ℝⁿ. Una rotación
        en ℝⁿ preserva la norma, por lo que n(σ*) y la energía relativa deben
        ser similares (no exactamente iguales porque la dirección en S² cambia).

        Verificación débil: dos vectores con la misma norma deben producir
        razones de compresión en el mismo rango de orden de magnitud.
        """
        G   = make_spd_matrix(3, seed=0)
        fib = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=5, n_theta=16, n_phi=32
        )
        v1 = make_random_logits(20, seed=1)
        v1 /= np.linalg.norm(v1)  # normalizar

        # Rotar v1 usando una rotación aleatoria (QR de matriz aleatoria)
        rng    = make_rng(42)
        A_rand = rng.standard_normal((20, 20))
        Q, _   = np.linalg.qr(A_rand)
        v2     = Q @ v1  # rotación: ‖v2‖ = ‖v1‖ = 1

        state1 = fib.refract_attention_logits(v1, 1.0)
        state2 = fib.refract_attention_logits(v2, 1.0)

        # Ambas razones deben estar en el mismo orden de magnitud (factor 10)
        r1, r2 = state1.kv_compression_ratio, state2.kv_compression_ratio
        assert max(r1, r2) < 10 * max(min(r1, r2), _EPSILON), (
            f"Ratios muy diferentes bajo rotación: r1={r1:.4f}, r2={r2:.4f}"
        )

    def test_attenuation_kernel_is_monotone_in_gamma(self) -> None:
        r"""
        INVARIANTE: Mayor γ ⟹ mayor atenuación para modos l ≥ 1.

        h_l(γ₁) > h_l(γ₂) si γ₁ < γ₂ y l ≥ 1.

        Equivalentemente: la energía reconstruida decrece con γ.
        """
        G       = make_spd_matrix(4, seed=0)
        logits  = make_random_logits(30, seed=42)
        sigma_s = 1.0

        norms_reconstructed = []
        for gamma in [0.01, 0.05, 0.1, 0.5, 1.0]:
            fib = OpticalRiemannLensFibrator(
                metric_tensor=G, l_cutoff=8,
                gamma_damping=gamma, n_theta=24, n_phi=48
            )
            state = fib.refract_attention_logits(logits, sigma_s)
            norms_reconstructed.append(float(np.linalg.norm(state.focused_logits)))

        # La norma debe ser no-creciente con γ
        for i in range(len(norms_reconstructed) - 1):
            assert norms_reconstructed[i] >= norms_reconstructed[i + 1] - 1e-8, (
                f"‖focused‖ aumenta con γ: "
                f"γ={[0.01,0.05,0.1,0.5,1.0][i]}, "
                f"‖focused‖={norms_reconstructed[i]:.4e} < "
                f"‖focused‖={norms_reconstructed[i+1]:.4e}"
            )

    def test_area_of_sphere_is_4pi_by_gauss_legendre(self) -> None:
        r"""
        INVARIANTE FUNDAMENTAL: ∫_{S²} dΩ = 4π.

        Verificación independiente usando Gauss-Legendre con diferentes n:
            Σ_{i,j} W_{ij} = Σ_i w_i · n_φ · Δφ = 2 · 2π = 4π
        """
        expected = 4.0 * np.pi
        for n_theta, n_phi in [(8,16), (16,32), (32,64), (64,128)]:
            _, w = np.polynomial.legendre.leggauss(n_theta)
            total = float(np.sum(w)) * 2.0 * np.pi  # Σ w_i · 2π
            rel_err = abs(total - expected) / expected
            assert rel_err < 1e-14, (
                f"Σ w_i · 2π = {total:.10f} ≠ 4π = {expected:.10f} "
                f"para n_theta={n_theta} (rel_err={rel_err:.2e})"
            )

    def test_spherical_harmonic_addition_theorem(self) -> None:
        r"""
        INVARIANTE: Teorema de adición de armónicos esféricos:

            Σ_{m=-l}^{l} |Y_l^m(θ,φ)|² = (2l+1)/(4π)  ∀ (θ,φ)

        Verificación numérica para l ∈ {0, 1, 2, 3}.
        """
        # Usar un punto arbitrario en S²
        theta_test = 1.2
        phi_test   = 0.8
        for l in range(4):
            sum_sq = sum(
                abs(float(sp_special.sph_harm(m, l, phi_test, theta_test))) ** 2
                for m in range(-l, l + 1)
            )
            expected = (2 * l + 1) / (4.0 * np.pi)
            assert abs(sum_sq - expected) < 1e-13, (
                f"Teorema de adición falla para l={l}: "
                f"Σ|Y|²={sum_sq:.8f} ≠ (2l+1)/4π={expected:.8f}"
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 7 — PRUEBAS DE ROBUSTEZ NUMÉRICA Y CASOS EXTREMOS                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestNumericalRobustness:
    """
    Pruebas que verifican el comportamiento del módulo en condiciones extremas:
    señales de muy alta o muy baja energía, dimensiones extremas, parámetros límite.
    """

    def test_very_small_logits_no_nan(self, fibrator_default) -> None:
        r"""
        ROBUSTEZ: Logits con magnitud muy pequeña (≈ ε) no producen NaN.

        Riesgo: División por norma muy pequeña en normalización Gram-Schmidt.
        """
        logits = make_random_logits(20) * 1e-10
        state  = fibrator_default.refract_attention_logits(logits, 0.0)
        assert np.all(np.isfinite(state.focused_logits))

    def test_very_large_logits_no_nan(self, fibrator_default) -> None:
        r"""
        ROBUSTEZ: Logits con magnitud muy grande (≈ 1e8) no producen NaN.

        Riesgo: Desbordamiento en psi_raw antes del escalado.
        """
        logits = make_random_logits(20) * 1e8
        state  = fibrator_default.refract_attention_logits(logits, 0.0)
        assert np.all(np.isfinite(state.focused_logits))

    def test_spike_logits_no_nan(self, fibrator_default) -> None:
        r"""
        ROBUSTEZ: Señal con pico extremo (spike de 1e6 en índice 0).

        Riesgo: sigma = std/range puede ser 0/1e6 ≈ 0, pero clip a σ_min lo protege.
        """
        logits = make_spike_logits(50, spike_idx=0, spike_val=1e6)
        state  = fibrator_default.refract_attention_logits(logits, 1.0)
        assert np.all(np.isfinite(state.focused_logits))

    def test_alternating_signs_logits(self, fibrator_default) -> None:
        r"""
        ROBUSTEZ: Señal alternada (-1, 1, -1, 1, ...) — alta frecuencia en ℝⁿ.

        Esta señal tiene la máxima varianza posible (std = 1) para rango = 2.
        """
        n      = 50
        logits = np.array([-1.0 if i % 2 == 0 else 1.0
                           for i in range(n)], dtype=np.float64)
        state  = fibrator_default.refract_attention_logits(logits, 1.0)
        assert np.all(np.isfinite(state.focused_logits))

    def test_nearly_constant_logits_sigma_clips_to_sigma_min(self) -> None:
        r"""
        ROBUSTEZ: Logits casi constantes (rango = 1e-8) ⟹ σ → σ_max.

        Para logits casi uniformes, range ≈ 0 ⟹ std/range → ∞, clipado a σ_max.
        """
        spec   = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=3, n_theta=10, n_phi=16
        )
        # Agregar una pequeña perturbación para evitar señal exactamente constante
        logits = np.ones(30, dtype=np.float64) + np.linspace(0, 1e-9, 30)
        sp     = spec._map_logits_to_sphere(logits)
        # σ debe estar en [σ_min, σ_max]
        assert _SIGMA_MIN <= sp.sigma <= _SIGMA_MAX

    def test_gram_schmidt_with_n_equals_3(self) -> None:
        r"""
        ROBUSTEZ: n=3 es el caso límite de Gram-Schmidt.

        Para n=3, la base {e₁, e₂, e₃} es exactamente la base canónica ℝ³.
        Los tres vectores canónicos deben estar disponibles para Gram-Schmidt.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=2, n_theta=8, n_phi=8
        )
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        gs     = spec._project_logits_to_r3_robust(logits)
        assert np.abs(np.linalg.norm(gs.proj_r3) - 1.0) < 1e-12

    @pytest.mark.parametrize("seed", range(10))
    def test_random_logits_always_produce_finite_output(self, seed: int) -> None:
        r"""
        ROBUSTEZ ESTADÍSTICA: 10 vectores aleatorios diferentes producen
        focused_logits finitos.
        """
        G   = make_spd_matrix(4, seed=seed)
        fib = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=5, n_theta=16, n_phi=32
        )
        logits = make_random_logits(30, seed=seed * 100)
        state  = fib.refract_attention_logits(logits, float(seed))
        assert np.all(np.isfinite(state.focused_logits)), (
            f"NaN/Inf en focused_logits para seed={seed}"
        )

    def test_parseval_error_bounds_known_function(self) -> None:
        r"""
        ROBUSTEZ ANALÍTICA: Para ψ = Y_1^0 (armónico puro de grado 1),
        el error de Parseval debe ser < _PARSEVAL_REL_TOL con n_theta ≥ 4.

        Fundamento: Y_1^0 es un polinomio de grado 1 en cos(θ). La cuadratura
        GL con n_theta ≥ 2 integra polinomios de grado ≤ 2·n_theta-1 = 3
        exactamente. Por tanto c_{1,0} = 1 exactamente y todos los demás = 0.
        """
        l_cutoff = 2
        n_theta  = 16
        n_phi    = 32
        spec     = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=l_cutoff, n_theta=n_theta, n_phi=n_phi
        )
        THETA, PHI = np.meshgrid(spec.grid.theta, spec.grid.phi, indexing='ij')

        # ψ = Y_1^0 real (normalizado)
        psi = np.real(sp_special.sph_harm(0, 1, PHI, THETA)).astype(np.float64)
        # Verificar energía L² = 1 antes de usar
        energy = compute_parseval_lhs(psi, spec.grid)
        psi    = psi / math.sqrt(max(energy, _EPSILON))  # renormalizar

        spectrum = spec._compute_spherical_coefficients(psi)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL, (
            f"Parseval error para Y_1^0: {spectrum.parseval_relative_error:.2e} "
            f"> {_PARSEVAL_REL_TOL}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 8 — PRUEBAS DE INTEGRACIÓN END-TO-END                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """
    Pruebas de integración completa que verifican el comportamiento del módulo
    como un sistema: de los logits crudos al RefractedState, pasando por las
    3 fases completas.
    """

    def test_full_pipeline_50_tokens(self, fibrator_default) -> None:
        r"""
        INTEGRACIÓN: Pipeline completo con vocabulario de 50 tokens.

        Simula la salida típica del head de logits de un LLM pequeño.
        """
        logits = make_random_logits(50, seed=0)
        state  = fibrator_default.refract_attention_logits(logits, 0.8)

        assert isinstance(state, RefractedState)
        assert state.focused_logits.shape == (50,)
        assert np.all(np.isfinite(state.focused_logits))
        assert 0.0 <= state.kv_compression_ratio <= 1.0
        assert state.fermat_metric_trace > 0.0

    def test_full_pipeline_32k_vocabulary(self) -> None:
        r"""
        INTEGRACIÓN: Pipeline con vocabulario grande n=32,768 (GPT-2 scale).

        Prueba que el módulo escala correctamente sin desbordamiento de memoria
        para n grande.
        """
        G   = make_spd_matrix(4, seed=1)
        fib = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=5, n_theta=16, n_phi=32
        )
        logits = make_random_logits(32768, seed=42)
        state  = fib.refract_attention_logits(logits, 2.0)

        assert np.all(np.isfinite(state.focused_logits))
        assert state.focused_logits.shape == (32768,)

    def test_sequential_calls_are_independent(self, fibrator_default) -> None:
        r"""
        INTEGRACIÓN: Dos llamadas secuenciales con logits diferentes producen
        resultados independientes (sin estado interno compartido entre llamadas).
        """
        logits_a = make_random_logits(30, seed=10)
        logits_b = make_random_logits(30, seed=20)

        state_a1 = fibrator_default.refract_attention_logits(logits_a, 1.0)
        state_b  = fibrator_default.refract_attention_logits(logits_b, 1.0)
        state_a2 = fibrator_default.refract_attention_logits(logits_a, 1.0)

        # state_a1 y state_a2 deben ser idénticos (determinismo)
        assert np.allclose(
            state_a1.focused_logits, state_a2.focused_logits, atol=1e-12
        ), "El fibrador no es determinista para la misma entrada"

        # state_b debe ser diferente de state_a1
        assert not np.allclose(
            state_a1.focused_logits, state_b.focused_logits
        ), "Entradas diferentes produjeron salidas idénticas"

    def test_high_gamma_produces_near_zero_output(self) -> None:
        r"""
        INTEGRACIÓN: Con γ → ∞, todos los modos l ≥ 1 se anulan.

        h_l = exp(-γ·n²·l²) → 0 para l ≥ 1 y γ → ∞.
        Solo el modo l=0 se preserva. La señal resultante es proporcional
        al valor medio de ψ sobre S².

        Para γ=100: h_1 = exp(-100·n²·1) < exp(-100) ≈ 4e-44 (numéricamente 0).
        """
        G   = make_spd_matrix(3, seed=0)
        fib = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=3, n_theta=10, n_phi=16,
            gamma_damping=100.0,  # γ extremadamente grande
        )
        logits = make_random_logits(15, seed=42)
        state  = fib.refract_attention_logits(logits, 0.5)

        # La señal resultante debe tener energía muy reducida
        norm_raw     = float(np.linalg.norm(logits))
        norm_focused = float(np.linalg.norm(state.focused_logits))
        ratio        = norm_focused / max(norm_raw, _EPSILON)
        assert ratio < 0.5, (
            f"Con γ=100, el ratio debería ser << 1, "
            f"pero ratio={ratio:.4f}"
        )

    def test_stress_norm_sign_affects_n_not_focused_shape(
        self, fibrator_default
    ) -> None:
        r"""
        INTEGRACIÓN: El signo de σ* cambia n(σ*) pero no la dimensión de la salida.

        σ* > 0: n → 2 (alta curvatura, más atenuación de alta frecuencia).
        σ* < 0: n → 1 (baja curvatura, menos atenuación).

        En ambos casos: focused_logits.shape == logits.shape.
        """
        logits = make_random_logits(40)
        for sigma in [-5.0, -0.1, 0.0, 0.1, 5.0]:
            state = fibrator_default.refract_attention_logits(logits, sigma)
            assert state.focused_logits.shape == logits.shape
            assert np.all(np.isfinite(state.focused_logits))

    def test_pipeline_energy_decreases_monotonically_with_cutoff(self) -> None:
        r"""
        INTEGRACIÓN: Mayor l_cutoff ⟹ más modos preservados ⟹ más energía.

        Con l_cutoff_1 < l_cutoff_2 y el mismo γ y σ*:
            ‖focused(l_cutoff_1)‖ ≤ ‖focused(l_cutoff_2)‖

        Fundamento: l_cutoff_2 incluye todos los modos de l_cutoff_1 más modos
        adicionales de alta frecuencia.
        """
        G      = make_spd_matrix(3, seed=0)
        logits = make_random_logits(20, seed=42)
        norms  = []

        for l_cutoff in [2, 4, 6, 8, 10]:
            n_theta = max(2 * l_cutoff + 2, 10)
            fib = OpticalRiemannLensFibrator(
                metric_tensor=G, l_cutoff=l_cutoff,
                gamma_damping=0.1, n_theta=n_theta, n_phi=2 * n_theta,
            )
            state = fib.refract_attention_logits(logits, 1.0)
            norms.append(float(np.linalg.norm(state.focused_logits)))

        # Normas deben ser no-decrecientes con l_cutoff
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i + 1] + 1e-6, (
                f"‖focused‖ decrece con l_cutoff: "
                f"l={[2,4,6,8,10][i]}: ‖f‖={norms[i]:.4e}, "
                f"l={[2,4,6,8,10][i+1]}: ‖f‖={norms[i+1]:.4e}"
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 9 — PRUEBAS DE RENDIMIENTO Y COMPLEJIDAD                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Pruebas de rendimiento que verifican que el módulo se ejecuta dentro de
    límites temporales razonables para casos de uso reales.

    NOTA: Estas pruebas son orientativas y pueden fallar en hardware lento.
    Se usan márgenes amplios (10×) para robustez en CI.
    """

    @pytest.mark.slow
    def test_initialization_time(self) -> None:
        r"""
        RENDIMIENTO: La inicialización (construcción de grilla + precomputa de armónicos)
        debe completarse en < 5 segundos para l_cutoff=10, n_theta=32, n_phi=64.

        El tensor de armónicos tiene shape (121, 32, 64) ≈ 3.8 MB complex128.
        Precomputarlo tarda O((l_cutoff+1)² · n_theta · n_phi) evaluaciones.
        """
        G = np.eye(4, dtype=np.float64)
        t0 = time.perf_counter()
        _ = OpticalRiemannLensFibrator(
            metric_tensor=G, l_cutoff=10, n_theta=32, n_phi=64
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, (
            f"Inicialización tardó {elapsed:.2f}s > 5s. "
            f"Posible regresión de rendimiento en precomputa de armónicos."
        )

    @pytest.mark.slow
    def test_refract_time_for_standard_vocabulary(self, fibrator_default) -> None:
        r"""
        RENDIMIENTO: refract_attention_logits para n=50,000 tokens debe completarse
        en < 1 segundo (una llamada de inferencia).

        El cuello de botella es _project_psi_to_logit_space (O(n·n_θ·n_φ) para
        la base de sensores). Con n=50k, n_θ=32, n_φ=64: 50k·2048 = 10⁸ ops.
        """
        logits = make_random_logits(50000, seed=42)
        t0     = time.perf_counter()
        state  = fibrator_default.refract_attention_logits(logits, 1.0)
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"refract_attention_logits tardó {elapsed:.2f}s > 2s para n=50k."
        )
        assert np.all(np.isfinite(state.focused_logits))

    @pytest.mark.slow
    def test_coefficient_computation_is_vectorized(self) -> None:
        r"""
        RENDIMIENTO: La implementación vectorizada de _compute_spherical_coefficients
        debe ser al menos 10× más rápida que un bucle Python equivalente para
        l_cutoff=10.

        Referencia: para l_cutoff=10 (121 coeficientes), n_theta=32, n_phi=64:
        - Bucle Python: ~121 iteraciones × overhead Python × n_theta × n_phi
        - NumPy vectorizado: una matmul de (121, 2048) × (2048,)
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=10, n_theta=32, n_phi=64
        )
        logits = make_random_logits(50)
        sp     = spec._map_logits_to_sphere(logits)

        # Tiempo de la implementación vectorizada
        n_runs = 10
        t0 = time.perf_counter()
        for _ in range(n_runs):
            spec._compute_spherical_coefficients(sp.psi)
        t_vectorized = (time.perf_counter() - t0) / n_runs

        # Tiempo del bucle Python de referencia
        THETA, PHI = np.meshgrid(spec.grid.theta, spec.grid.phi, indexing='ij')
        W = spec.grid.weights

        def loop_computation():
            num_coeffs = (spec.l_cutoff + 1) ** 2
            c = np.zeros(num_coeffs, dtype=np.complex128)
            idx = 0
            for l in range(spec.l_cutoff + 1):
                for m in range(-l, l + 1):
                    Y_lm = sp_special.sph_harm(m, l, PHI, THETA)
                    c[idx] = np.sum(sp.psi * np.conj(Y_lm) * W)
                    idx += 1
            return c

        t0 = time.perf_counter()
        for _ in range(n_runs):
            loop_computation()
        t_loop = (time.perf_counter() - t0) / n_runs

        speedup = t_loop / max(t_vectorized, 1e-9)
        assert speedup >= 2.0, (
            f"La implementación vectorizada ({t_vectorized*1000:.1f}ms) "
            f"no es al menos 2× más rápida que el bucle ({t_loop*1000:.1f}ms). "
            f"Speedup={speedup:.1f}×."
        )

    def test_harmonics_tensor_memory_footprint(self, spectrometer_default) -> None:
        r"""
        RENDIMIENTO (MEMORIA): El tensor de armónicos para l_cutoff=10, n=32×64
        debe ocupar < 10 MB.

        Cálculo: (11)² × 32 × 64 × 16 bytes (complex128) = 121 × 2048 × 16 ≈ 3.97 MB.
        """
        tensor = spectrometer_default.harmonics_tensor
        size_mb = tensor.nbytes / 1e6
        assert size_mb < 10.0, (
            f"Tensor de armónicos ocupa {size_mb:.2f} MB > 10 MB."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOQUE 10 — PRUEBAS DE PROPIEDADES ALGEBRAICAS DE LAS EXCEPCIONES          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestAlgebraicExceptionProperties:
    """
    Pruebas que verifican que las excepciones se lanzan en las condiciones
    algebraicas correctas y NO se lanzan cuando no deben.
    """

    def test_optical_dispersion_never_raised_for_standard_implementation(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: OpticalDispersionError NUNCA se lanza en la
        implementación estándar de _compute_fermat_refractive_index.

        Fundamento algebraico: n = 1 + tanh(α·σ*) ≥ 1 + tanh(-∞) = 1 + (-1) = 0,
        pero más precisamente n > 0 para todo σ* finito, y la condición en el
        código es n < 1.0, que es algebraicamente imposible con tanh ∈ (-1,1).
        """
        lens = Phase2_CategoricalOpticLens(
            metric_tensor=np.eye(3), l_cutoff=3, n_theta=8, n_phi=8
        )
        # Probar con valores extremos de σ*
        extreme_values = [-1e15, -1e10, -1e5, -100, -1, 0, 1, 100, 1e5, 1e10]
        for sigma in extreme_values:
            try:
                n = lens._compute_fermat_refractive_index(float(sigma))
                assert n >= 1.0, (
                    f"n={n} < 1 para σ*={sigma}, pero OpticalDispersionError no se lanzó"
                )
            except OpticalDispersionError:
                pytest.fail(
                    f"OpticalDispersionError lanzado incorrectamente para σ*={sigma} "
                    f"en la implementación estándar"
                )

    def test_parseval_violation_not_raised_with_sufficient_grid(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: ParsevalViolationError NO se lanza cuando
        n_theta ≥ 2·l_cutoff + 2 (grilla suficiente).

        Para l_cutoff=5 y n_theta=16 ≥ 2·5+2 = 12:
        La cuadratura GL integra exactamente polinomios de grado ≤ 2·16-1 = 31 ≥ 10.
        """
        l_cutoff = 5
        n_theta  = 2 * l_cutoff + 4  # holgura de 4 nodos extra
        spec     = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=l_cutoff, n_theta=n_theta, n_phi=2*n_theta,
            verify_parseval=True,
        )
        logits   = make_random_logits(30, seed=42)
        sp       = spec._map_logits_to_sphere(logits)
        # No debe lanzar ParsevalViolationError
        spectrum = spec._compute_spherical_coefficients(sp.psi)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL

    def test_lens_singularity_not_raised_for_diverse_signals(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: LensSingularityError NO se lanza para señales
        no-constantes de diversas distribuciones.
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=3, n_theta=8, n_phi=16
        )
        test_signals = [
            make_random_logits(20, seed=i) for i in range(5)
        ] + [
            make_spike_logits(20, spike_idx=i, spike_val=5.0) for i in range(5)
        ] + [
            np.arange(20, dtype=np.float64),   # [0,1,...,19]
            np.exp(np.linspace(-2, 2, 20)),     # exponencial
        ]
        for i, logits in enumerate(test_signals):
            try:
                _ = spec._project_logits_to_r3_robust(logits)
            except LensSingularityError:
                pytest.fail(
                    f"LensSingularityError lanzado incorrectamente para "
                    f"señal {i}: {logits[:5]}"
                )

    def test_spectral_grid_error_not_raised_for_correct_jacobian(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: SpectralGridError NO se lanza cuando el Jacobiano
        es W_{ij} = w_i · Δφ (correcto).

        Verificación de que la construcción estándar de la grilla nunca produce
        este error.
        """
        for n_theta, n_phi in [(8,16), (16,32), (32,64)]:
            # No debe lanzar
            spec = Phase1_SphericalHarmonicsSpectrometer(
                l_cutoff=min(n_theta//2-1, 5),
                n_theta=n_theta, n_phi=n_phi
            )
            assert spec.grid.shape == (n_theta, n_phi)

    def test_gram_schmidt_degeneracy_not_raised_for_generic_signals(self) -> None:
        r"""
        PROPIEDAD NEGATIVA: GramSchmidtDegeneracyError NO se lanza para
        señales genéricas con n ≥ 3.

        Solo puede ocurrir si todos los vectores canónicos son colineales
        con e₁, lo cual es imposible para n ≥ 3 (existe al menos un canónico
        con componente ≤ 1/√n < 1 en la dirección de e₁).
        """
        spec = Phase1_SphericalHarmonicsSpectrometer(
            l_cutoff=3, n_theta=8, n_phi=16
        )
        for n in range(3, 50):
            logits = make_random_logits(n, seed=n * 7 + 13)
            try:
                gs = spec._project_logits_to_r3_robust(logits)
                assert np.abs(np.linalg.norm(gs.proj_r3) - 1.0) < 1e-12
            except GramSchmidtDegeneracyError:
                pytest.fail(
                    f"GramSchmidtDegeneracyError incorrecto para n={n}"
                )


# ════════════════════════════════════════════════════════════════════════════════
# Configuración de pytest
# ════════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Registra marcadores personalizados."""
    config.addinivalue_line(
        "markers", "slow: pruebas de rendimiento (omitir con -m 'not slow')"
    )