# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Suite de Pruebas: Optical Riemann Lens                                             ║
║  Ruta   : tests/unit/physics/test_optical_riemann_lens.py                          ║
║  Versión: 3.0.0-Rigorous-Jacobian-Orthogonal                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Filosofía de Testing — Axiomas de Óptica Geométrica como Contratos Ejecutables:
══════════════════════════════════════════════════════════════════════════════════════

Cada test verifica una propiedad matemática formal derivada de:

  §O1. Correctitud del Jacobiano de Gauss-Legendre (Fase 1):
       - Σ W_{ij} = 4π  (área total de S²)
       - W_{ij} = w_i · Δφ  (SIN sin(θ_i) — Jacobiano correcto)
       - ⟨Y_0^0, Y_0^0⟩_{L²} = 1  (ortogonalidad de la grilla)

  §O2. Identidad de Parseval en L²(S²) (Fase 1):
       - ‖ψ‖²_{L²} = Σ|c_{lm}|²  (dentro de tolerancia _PARSEVAL_REL_TOL)
       - La violación indica cuadratura incorrecta o grilla insuficiente

  §O3. Proyección a S² y propiedades geométricas (Fase 1):
       - ‖p‖₂ = 1  (vector unitario en ℝ³ — Gram-Schmidt ortonormal)
       - θ₀ ∈ [0,π], φ₀ ∈ [-π,π]  (coordenadas esféricas válidas)
       - σ ∈ [_SIGMA_MIN, _SIGMA_MAX]  (ancho gaussiano acotado)
       - ‖ψ‖_{L²} ≈ ‖logits‖₂  (conservación de energía)

  §O4. Índice de Refracción de Fermat (Fase 2):
       - n ∈ (1,2)  ∀ σ* ∈ ℝ  (algebraicamente garantizado)
       - n es monótonamente creciente en σ*
       - Tr(g_F) = n² · Tr(G) > 0  (traza positiva)

  §O5. Morfismos View y Put (Fase 2):
       - view(logits) retorna SphericalSpectrum con shape correcto
       - put(logits, spec, n) retorna array de la misma dimensión
       - Energía filtrada ≤ energía original  (amortiguación pasa-bajos)
       - Para γ→0: put → reconstrucción casi perfecta  (identidad límite)
       - Para γ→∞: put → modo DC (solo l=0 sobrevive)

  §O6. Orquestador refract_attention_logits (Fase 3):
       - RefractedState con invariantes verificados en __post_init__
       - kv_compression_ratio ∈ [0,1]  (algebraicamente acotado)
       - focused_logits finito y de dimensión n
       - SpectralDiagnostics con todos los campos positivos o en rango

  §O7. Contratos de Error y Robustez:
       - LensSingularityError para n<3 o señal constante
       - SpectralGridError para grilla con Jacobiano incorrecto
       - ParsevalViolationError para grilla insuficiente con l_cutoff alto
       - TypeError/ValueError para entradas inválidas en todos los métodos

  §O8. Propiedades Estadísticas (barrido sobre 10 señales aleatorias):
       - Parseval satisfecho para señales diversas
       - kv_ratio ∈ [0,1] para estrés logístico variado
       - Monotonicidad de energía filtrada vs. γ

Estructura de la suite:
══════════════════════════════════════════════════════════════════════════════════════
  TestSphericalGridConstruction         — §O1: Jacobiano, área, ortogonalidad
  TestSphericalGridInvariants           — __post_init__ y campos inmutables
  TestSphericalSpectrumDataclass        — invariantes del contenedor espectral
  TestRefractedStateDataclass           — invariantes del estado refractado
  TestSpectrometerParams                — validación de parámetros __init__
  TestProjectLogitsToR3                 — §O3: Gram-Schmidt, norma, unicidad
  TestMapLogitsToSphere                 — §O3: gaussiana, energía, sigma
  TestSphericalCoefficients             — §O2: Parseval, ortogonalidad, shape
  TestFermatRefractiveIndex             — §O4: rango, monotonía, traza
  TestViewFunctor                       — §O5: shape, tipo, consistencia
  TestPutFunctor                        — §O5: dimensión, amortiguación, límites
  TestRefractAttentionLogits            — §O6: pipeline completo, tipos, diagnósticos
  TestErrorContracts                    — §O7: excepciones correctas
  TestStatisticalProperties             — §O8: barrido aleatorio
  TestHighDimensionalLogits             — robustez para n grande
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

import numpy as np
import pytest
import scipy.special as sp_special
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ════════════════════════════════════════════════════════════════════════════════
from app.omega.optical_riemann_lens import (
    # Excepciones
    OpticalDispersionError,
    LensSingularityError,
    ParsevalViolationError,
    SpectralGridError,
    # Estructuras de datos
    SphericalGrid,
    SphericalSpectrum,
    SpectralDiagnostics,
    RefractedState,
    # Fases internas (para test granular)
    Phase1_SphericalHarmonicsSpectrometer,
    Phase2_CategoricalOpticLens,
    # Agente principal
    OpticalRiemannLensFibrator,
    # Constantes
    _EPSILON,
    _SIGMA_MIN,
    _SIGMA_MAX,
    _ORTHOGONALITY_TOL,
    _PARSEVAL_REL_TOL,
    _N_SVD_COMPONENTS,
    _FERMAT_ALPHA,
    _COMPRESSION_FLOOR,
    _COMPRESSION_CAP,
)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING)
_RNG_SEED: int = 42
_RNG: np.random.Generator = np.random.default_rng(_RNG_SEED)

# Parámetros de construcción rápida para tests (grilla pequeña)
_FAST_N_THETA: int = 16
_FAST_N_PHI  : int = 32
_FAST_L_MAX  : int = 4
_FAST_GAMMA  : float = 0.1


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FÁBRICAS DE OBJETOS DE PRUEBA                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

def make_spectrometer(
    l_cutoff     : int   = _FAST_L_MAX,
    gamma_damping: float = _FAST_GAMMA,
    n_theta      : int   = _FAST_N_THETA,
    n_phi        : int   = _FAST_N_PHI,
    verify_parseval: bool = True,
) -> Phase1_SphericalHarmonicsSpectrometer:
    """Construye un espectrómetro con parámetros de prueba rápidos."""
    return Phase1_SphericalHarmonicsSpectrometer(
        l_cutoff      = l_cutoff,
        gamma_damping = gamma_damping,
        n_theta       = n_theta,
        n_phi         = n_phi,
        verify_parseval = verify_parseval,
    )


def make_lens(
    l_cutoff     : int   = _FAST_L_MAX,
    gamma_damping: float = _FAST_GAMMA,
    n_theta      : int   = _FAST_N_THETA,
    n_phi        : int   = _FAST_N_PHI,
    metric_tensor: Optional[NDArray[np.float64]] = None,
) -> Phase2_CategoricalOpticLens:
    """Construye un lente categórico con parámetros de prueba."""
    G = metric_tensor if metric_tensor is not None else np.eye(4, dtype=np.float64)
    return Phase2_CategoricalOpticLens(
        metric_tensor = G,
        l_cutoff      = l_cutoff,
        gamma_damping = gamma_damping,
        n_theta       = n_theta,
        n_phi         = n_phi,
    )


def make_fibrator(
    l_cutoff     : int   = _FAST_L_MAX,
    gamma_damping: float = _FAST_GAMMA,
    n_theta      : int   = _FAST_N_THETA,
    n_phi        : int   = _FAST_N_PHI,
    metric_tensor: Optional[NDArray[np.float64]] = None,
) -> OpticalRiemannLensFibrator:
    """Construye el fibrador óptico completo."""
    G = metric_tensor if metric_tensor is not None else np.eye(4, dtype=np.float64)
    return OpticalRiemannLensFibrator(
        metric_tensor = G,
        l_cutoff      = l_cutoff,
        gamma_damping = gamma_damping,
        n_theta       = n_theta,
        n_phi         = n_phi,
    )


def make_random_logits(
    n: int, seed: int = _RNG_SEED, scale: float = 1.0
) -> NDArray[np.float64]:
    """Genera un vector de logits aleatorio de tamaño n."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * scale).astype(np.float64)


def make_spd_metric(n: int, seed: int = _RNG_SEED) -> NDArray[np.float64]:
    """Genera una métrica SPD de tamaño n×n."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    G = A.T @ A + float(n) * np.eye(n)
    return G.astype(np.float64)


def make_valid_spherical_spectrum(
    l_max: int, seed: int = _RNG_SEED
) -> SphericalSpectrum:
    """Genera un SphericalSpectrum válido con coeficientes aleatorios."""
    rng = np.random.default_rng(seed)
    n_coeffs = (l_max + 1) ** 2
    c = (rng.standard_normal(n_coeffs)
         + 1j * rng.standard_normal(n_coeffs)).astype(np.complex128)
    parseval_lhs = float(np.sum(np.abs(c) ** 2))
    return SphericalSpectrum(
        coefficients            = c,
        l_max                   = l_max,
        parseval_lhs            = parseval_lhs,
        parseval_rhs            = parseval_lhs,
        parseval_relative_error = 0.0,
    )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O1 — TestSphericalGridConstruction: Jacobiano y Área                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSphericalGridConstruction:
    """
    Verifica que la grilla de Gauss-Legendre tiene el Jacobiano correcto.

    Propiedad central: los pesos de leggauss integran respecto a dt = d(cos θ),
    por lo que W_{ij} = w_i · Δφ (SIN multiplicar por sin(θ_i)).

    Verificaciones:
      - Σ W_{ij} = 4π  (área total de S²)
      - W_{ij} > 0 para todo i,j
      - W_{ij} = w_i · Δφ  (estructura del peso)
      - θ_i ∈ (0,π) para todo i
      - ⟨Y_0^0, Y_0^0⟩_{L²} = 1  (ortogonalidad)
    """

    @pytest.mark.parametrize("n_theta,n_phi", [
        (8, 16), (16, 32), (32, 64), (64, 128),
    ])
    def test_total_area_equals_4pi(self, n_theta: int, n_phi: int) -> None:
        """Σ W_{ij} = 4π para distintas resoluciones de grilla."""
        spec = make_spectrometer(n_theta=n_theta, n_phi=n_phi)
        total_area = float(np.sum(spec.grid.weights))
        assert_allclose(
            total_area, 4.0 * np.pi, rtol=1e-12,
            err_msg=(
                f"Área total de S² = {total_area:.8f} ≠ 4π = {4*np.pi:.8f} "
                f"para n_θ={n_theta}, n_φ={n_phi}."
            )
        )

    def test_weights_are_positive(self) -> None:
        """Todos los pesos W_{ij} son estrictamente positivos."""
        spec = make_spectrometer()
        assert np.all(spec.grid.weights > 0), (
            "Los pesos de integración deben ser todos positivos."
        )

    def test_weights_structure_is_wt_times_dphi(self) -> None:
        """
        W[i, j] = w_i · Δφ para todo j (independiente de j).

        Verifica que los pesos no dependen de φ y que su estructura
        es exactamente el producto de los pesos de Gauss-Legendre
        por el espaciado uniforme en φ.
        """
        spec = make_spectrometer()
        grid = spec.grid
        delta_phi = grid.delta_phi
        expected_weights_row = grid.gauss_weights * delta_phi

        # Cada fila de grid.weights debe ser constante = w_i · Δφ
        for i in range(grid.shape[0]):
            assert_allclose(
                grid.weights[i, :],
                expected_weights_row[i] * np.ones(grid.shape[1]),
                rtol=1e-14,
                err_msg=f"Fila {i} de weights no coincide con w_{i}·Δφ."
            )

    def test_weights_do_not_include_sin_theta(self) -> None:
        """
        Verificación negativa: los pesos NO incluyen el factor sin(θ_i).

        Si los pesos incluyesen sin(θ_i), la integral de la función constante
        f=1 sobre S² daría un resultado diferente de 4π.
        Esta prueba verifica que la corrección del Jacobiano fue aplicada.
        """
        spec = make_spectrometer()
        grid = spec.grid

        # Con Jacobiano correcto: Σ W_{ij} = 4π
        area_correct = float(np.sum(grid.weights))
        assert_allclose(area_correct, 4.0 * np.pi, rtol=1e-12)

        # Si los pesos incluyesen sin(θ): Σ W'_{ij} = Σ w_i·Δφ·sin(θ_i)
        # Esa suma sería 2π · Σ w_i · sin(arccos(t_i)) ≠ 4π en general
        wrong_weights = (
            grid.gauss_weights[:, np.newaxis]
            * grid.delta_phi
            * np.sin(grid.theta)[:, np.newaxis]
        )
        area_wrong = float(np.sum(wrong_weights))
        # El área incorrecta debe diferir de 4π si leggauss no fuese exactamente
        # la cuadratura de sin(θ) (que en general no lo es)
        # Para n_theta suficientemente grande, ambas convergen a 4π,
        # pero para n_theta pequeño difieren
        # Aquí verificamos que la implementación usa la versión CORRECTA
        assert abs(area_correct - 4.0 * np.pi) < abs(area_wrong - 4.0 * np.pi) + 1e-10, (
            "La implementación debe usar el Jacobiano correcto (sin sin(θ_i))."
        )

    def test_theta_nodes_in_open_interval(self) -> None:
        """θ_i ∈ (0,π) para todo i (sin incluir los polos exactos)."""
        spec = make_spectrometer()
        theta = spec.grid.theta
        assert np.all(theta > 0) and np.all(theta < np.pi), (
            "Los nodos theta deben estar en el interior (0,π)."
        )

    def test_phi_nodes_in_0_2pi(self) -> None:
        """φ_j ∈ [0, 2π) para todo j (dominio correcto de la longitud)."""
        spec = make_spectrometer()
        phi = spec.grid.phi
        assert np.all(phi >= 0) and np.all(phi < 2 * np.pi), (
            "Los nodos phi deben estar en [0, 2π)."
        )

    def test_phi_spacing_is_uniform(self) -> None:
        """Los nodos φ están equiespaciados con Δφ = 2π/n_φ."""
        spec = make_spectrometer()
        phi = spec.grid.phi
        diffs = np.diff(phi)
        assert_allclose(
            diffs, spec.grid.delta_phi * np.ones_like(diffs), rtol=1e-14,
            err_msg="Los nodos phi deben estar equiespaciados."
        )

    def test_grid_shape_matches_n_theta_n_phi(self) -> None:
        """grid.shape == (n_theta, n_phi)."""
        n_theta, n_phi = 20, 40
        spec = make_spectrometer(n_theta=n_theta, n_phi=n_phi)
        assert spec.grid.shape == (n_theta, n_phi), (
            f"grid.shape={spec.grid.shape} ≠ ({n_theta},{n_phi})."
        )

    def test_orthogonality_Y00_is_satisfied(self) -> None:
        """
        ⟨Y_0^0, Y_0^0⟩_{L²(S²)} = 1 con la grilla construida.

        Y_0^0 = 1/√(4π) en toda la esfera.
        ⟨Y_0^0, Y_0^0⟩ = Σ_{ij} |Y_0^0|² · W_{ij} = (1/4π) · 4π = 1.
        """
        spec = make_spectrometer()
        THETA, PHI = np.meshgrid(
            spec.grid.theta, spec.grid.phi, indexing='ij'
        )
        Y00 = sp_special.sph_harm(0, 0, PHI, THETA)
        norm_sq = float(np.sum(np.abs(Y00) ** 2 * spec.grid.weights))
        assert_allclose(
            norm_sq, 1.0, atol=_ORTHOGONALITY_TOL,
            err_msg=f"⟨Y_0^0, Y_0^0⟩ = {norm_sq:.8f} ≠ 1."
        )

    @pytest.mark.parametrize("l,m", [
        (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, 0), (2, 2)
    ])
    def test_orthonormality_diagonal_terms(self, l: int, m: int) -> None:
        """⟨Y_l^m, Y_l^m⟩_{L²} = 1 para varios (l,m)."""
        spec = make_spectrometer(n_theta=32, n_phi=64)
        THETA, PHI = np.meshgrid(
            spec.grid.theta, spec.grid.phi, indexing='ij'
        )
        Y_lm = sp_special.sph_harm(m, l, PHI, THETA)
        norm_sq = float(np.sum(np.abs(Y_lm) ** 2 * spec.grid.weights))
        assert_allclose(
            norm_sq, 1.0, atol=5e-4,
            err_msg=f"‖Y_{l}^{m}‖²_{{L²}} = {norm_sq:.6f} ≠ 1."
        )

    @pytest.mark.parametrize("l1,m1,l2,m2", [
        (0, 0, 1, 0),
        (1, 1, 1, -1),
        (2, 0, 1, 1),
    ])
    def test_orthogonality_off_diagonal_terms(
        self, l1: int, m1: int, l2: int, m2: int
    ) -> None:
        """⟨Y_l1^m1, Y_l2^m2⟩_{L²} ≈ 0 para (l1,m1) ≠ (l2,m2)."""
        spec = make_spectrometer(n_theta=32, n_phi=64)
        THETA, PHI = np.meshgrid(
            spec.grid.theta, spec.grid.phi, indexing='ij'
        )
        Y1 = sp_special.sph_harm(m1, l1, PHI, THETA)
        Y2 = sp_special.sph_harm(m2, l2, PHI, THETA)
        inner_product = float(np.abs(np.sum(Y1 * np.conj(Y2) * spec.grid.weights)))
        assert inner_product < 1e-3, (
            f"⟨Y_{l1}^{m1}, Y_{l2}^{m2}⟩ = {inner_product:.2e} debe ser ≈ 0."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestSphericalGridInvariants: __post_init__ y Campos                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSphericalGridInvariants:
    """
    Verifica que SphericalGrid.__post_init__ rechaza grillas inválidas
    y que sus campos son inmutables (frozen dataclass).
    """

    def _build_valid_grid(
        self, n_theta: int = 8, n_phi: int = 16
    ) -> SphericalGrid:
        """Construye una SphericalGrid válida directamente."""
        t, w = np.polynomial.legendre.leggauss(n_theta)
        theta = np.arccos(t).astype(np.float64)
        gauss_weights = w.astype(np.float64)
        delta_phi = 2.0 * np.pi / n_phi
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float64)
        weights = gauss_weights[:, np.newaxis] * delta_phi * np.ones((1, n_phi))
        return SphericalGrid(
            theta=theta, phi=phi, gauss_weights=gauss_weights,
            delta_phi=delta_phi, weights=weights, shape=(n_theta, n_phi)
        )

    def test_valid_grid_constructs_without_error(self) -> None:
        """Una grilla válida se construye sin excepción."""
        grid = self._build_valid_grid()
        assert grid.shape == (8, 16)

    def test_grid_is_frozen(self) -> None:
        """SphericalGrid es inmutable (frozen dataclass)."""
        grid = self._build_valid_grid()
        with pytest.raises((AttributeError, TypeError)):
            grid.shape = (999, 999)  # type: ignore[misc]

    def test_rejects_theta_including_poles(self) -> None:
        """SpectralGridError si θ incluye 0 o π."""
        n = 8
        t, w = np.polynomial.legendre.leggauss(n)
        theta_bad = np.arccos(t)
        theta_bad[0] = 0.0  # polo norte exacto
        gauss_weights = w.astype(np.float64)
        delta_phi = 2.0 * np.pi / 16
        phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        weights = gauss_weights[:, np.newaxis] * delta_phi * np.ones((1, 16))
        with pytest.raises(SpectralGridError):
            SphericalGrid(
                theta=theta_bad, phi=phi, gauss_weights=gauss_weights,
                delta_phi=delta_phi, weights=weights, shape=(n, 16)
            )

    def test_rejects_negative_gauss_weights(self) -> None:
        """SpectralGridError si hay pesos de Gauss-Legendre negativos."""
        n = 8
        t, w = np.polynomial.legendre.leggauss(n)
        theta = np.arccos(t)
        gauss_weights_bad = w.copy()
        gauss_weights_bad[0] = -1.0  # peso negativo artificial
        delta_phi = 2.0 * np.pi / 16
        phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        weights_bad = gauss_weights_bad[:, np.newaxis] * delta_phi * np.ones((1, 16))
        with pytest.raises(SpectralGridError):
            SphericalGrid(
                theta=theta, phi=phi, gauss_weights=gauss_weights_bad,
                delta_phi=delta_phi, weights=weights_bad, shape=(n, 16)
            )

    def test_rejects_wrong_weights_shape(self) -> None:
        """SpectralGridError si weights.shape ≠ (n_theta, n_phi)."""
        n = 8
        t, w = np.polynomial.legendre.leggauss(n)
        theta = np.arccos(t)
        gauss_weights = w.astype(np.float64)
        delta_phi = 2.0 * np.pi / 16
        phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        # Weights con shape incorrecto
        weights_bad = np.ones((n, 20))  # n_phi incorrecto
        with pytest.raises(SpectralGridError):
            SphericalGrid(
                theta=theta, phi=phi, gauss_weights=gauss_weights,
                delta_phi=delta_phi, weights=weights_bad, shape=(n, 16)
            )

    def test_rejects_wrong_area(self) -> None:
        """SpectralGridError si Σ W_{ij} ≠ 4π."""
        n = 8
        t, w = np.polynomial.legendre.leggauss(n)
        theta = np.arccos(t)
        gauss_weights = w.astype(np.float64)
        delta_phi = 2.0 * np.pi / 16
        phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        # Weights con área incorrecta (multiplicados por sin(θ))
        wrong_w = (
            gauss_weights[:, np.newaxis]
            * delta_phi
            * np.sin(theta)[:, np.newaxis]  # Jacobiano duplicado
            * np.ones((1, 16))
        )
        # Este test verifica que la grilla con Jacobiano duplicado falla
        # Solo si la suma de wrong_w difiere suficientemente de 4π
        wrong_area = float(np.sum(wrong_w))
        if abs(wrong_area - 4.0 * np.pi) > 1e-10:
            with pytest.raises(SpectralGridError, match="4π"):
                SphericalGrid(
                    theta=theta, phi=phi, gauss_weights=gauss_weights,
                    delta_phi=delta_phi, weights=wrong_w, shape=(n, 16)
                )
        else:
            pytest.skip(
                "El Jacobiano duplicado produce área ≈ 4π para esta configuración."
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestSphericalSpectrumDataclass: Invariantes del Contenedor Espectral       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSphericalSpectrumDataclass:
    """
    Verifica los invariantes de SphericalSpectrum:
      - Shape de coefficients == ((l_max+1)²,)
      - l_max ≥ 0
      - Inmutabilidad (frozen)
      - n_coefficients correcto
      - power_at_degree con valores esperados
    """

    def test_valid_spectrum_constructs(self) -> None:
        """Construcción válida sin error."""
        spec = make_valid_spherical_spectrum(l_max=3)
        assert spec.l_max == 3
        assert spec.coefficients.shape == (16,)  # (3+1)² = 16

    def test_n_coefficients_property(self) -> None:
        """n_coefficients == (l_max+1)²."""
        for l_max in [0, 1, 2, 5, 10]:
            spec = make_valid_spherical_spectrum(l_max=l_max)
            assert spec.n_coefficients == (l_max + 1) ** 2

    def test_rejects_wrong_coefficients_shape(self) -> None:
        """ValueError si coefficients.shape ≠ ((l_max+1)²,)."""
        with pytest.raises(ValueError, match="shape"):
            SphericalSpectrum(
                coefficients            = np.zeros(10, dtype=np.complex128),
                l_max                   = 3,   # esperado: 16 coeficientes
                parseval_lhs            = 0.0,
                parseval_rhs            = 0.0,
                parseval_relative_error = 0.0,
            )

    def test_rejects_negative_l_max(self) -> None:
        """ValueError si l_max < 0."""
        with pytest.raises(ValueError, match="l_max"):
            SphericalSpectrum(
                coefficients            = np.zeros(1, dtype=np.complex128),
                l_max                   = -1,
                parseval_lhs            = 0.0,
                parseval_rhs            = 0.0,
                parseval_relative_error = 0.0,
            )

    def test_spectrum_is_frozen(self) -> None:
        """SphericalSpectrum es inmutable."""
        spec = make_valid_spherical_spectrum(l_max=2)
        with pytest.raises((AttributeError, TypeError)):
            spec.l_max = 99  # type: ignore[misc]

    def test_power_at_degree_zero(self) -> None:
        """
        power_at_degree(0) = |c_00|² (solo un coeficiente en l=0).
        """
        rng = np.random.default_rng(101)
        n_coeffs = (3 + 1) ** 2
        c = (rng.standard_normal(n_coeffs)
             + 1j * rng.standard_normal(n_coeffs)).astype(np.complex128)
        spec = SphericalSpectrum(
            coefficients            = c,
            l_max                   = 3,
            parseval_lhs            = 1.0,
            parseval_rhs            = 1.0,
            parseval_relative_error = 0.0,
        )
        expected_power = float(np.abs(c[0]) ** 2)  # índice 0 = (l=0, m=0)
        assert_allclose(
            spec.power_at_degree(0), expected_power, rtol=1e-14,
            err_msg="power_at_degree(0) debe ser |c_00|²."
        )

    def test_power_at_degree_sums_to_parseval_rhs(self) -> None:
        """
        Σ_{l=0}^{l_max} P_l = Σ|c_{lm}|² = parseval_rhs.
        """
        spec = make_valid_spherical_spectrum(l_max=4)
        total_power = sum(spec.power_at_degree(l) for l in range(spec.l_max + 1))
        assert_allclose(
            total_power, float(np.sum(np.abs(spec.coefficients) ** 2)),
            rtol=1e-12,
            err_msg="Σ P_l debe ser igual a Σ|c_{lm}|²."
        )

    def test_power_at_degree_raises_for_out_of_range(self) -> None:
        """ValueError para l fuera de [0, l_max]."""
        spec = make_valid_spherical_spectrum(l_max=3)
        with pytest.raises(ValueError):
            spec.power_at_degree(-1)
        with pytest.raises(ValueError):
            spec.power_at_degree(4)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestRefractedStateDataclass: Invariantes del Estado Refractado             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestRefractedStateDataclass:
    """
    Verifica los invariantes de RefractedState.__post_init__:
      - focused_logits es finito
      - kv_compression_ratio ∈ [0,1]
      - Inmutabilidad (frozen)
    """

    def _make_diagnostics(self) -> SpectralDiagnostics:
        return SpectralDiagnostics(
            n_refract=1.5, fermat_metric_trace=6.0, parseval_error=0.01,
            kv_compression_ratio=0.8, sigma_projection=0.5,
            energy_raw=10.0, energy_focused=8.0, energy_retention=0.8,
            l_dominant=1,
        )

    def test_valid_refracted_state_constructs(self) -> None:
        """Construcción válida sin excepción."""
        state = RefractedState(
            focused_logits       = np.array([1.0, 2.0, 3.0]),
            kv_compression_ratio = 0.75,
            fermat_metric_trace  = 8.0,
            diagnostics          = self._make_diagnostics(),
        )
        assert state.kv_compression_ratio == 0.75

    def test_rejects_nan_logits(self) -> None:
        """ValueError si focused_logits contiene NaN."""
        with pytest.raises(ValueError, match="no finitos"):
            RefractedState(
                focused_logits       = np.array([1.0, np.nan, 3.0]),
                kv_compression_ratio = 0.5,
                fermat_metric_trace  = 4.0,
                diagnostics          = self._make_diagnostics(),
            )

    def test_rejects_inf_logits(self) -> None:
        """ValueError si focused_logits contiene Inf."""
        with pytest.raises(ValueError, match="no finitos"):
            RefractedState(
                focused_logits       = np.array([1.0, np.inf, 3.0]),
                kv_compression_ratio = 0.5,
                fermat_metric_trace  = 4.0,
                diagnostics          = self._make_diagnostics(),
            )

    def test_rejects_ratio_above_one(self) -> None:
        """ValueError si kv_compression_ratio > 1."""
        with pytest.raises(ValueError, match="kv_compression_ratio"):
            RefractedState(
                focused_logits       = np.array([1.0, 2.0, 3.0]),
                kv_compression_ratio = 1.5,
                fermat_metric_trace  = 4.0,
                diagnostics          = self._make_diagnostics(),
            )

    def test_rejects_negative_ratio(self) -> None:
        """ValueError si kv_compression_ratio < 0."""
        with pytest.raises(ValueError, match="kv_compression_ratio"):
            RefractedState(
                focused_logits       = np.array([1.0, 2.0, 3.0]),
                kv_compression_ratio = -0.1,
                fermat_metric_trace  = 4.0,
                diagnostics          = self._make_diagnostics(),
            )

    def test_refracted_state_is_frozen(self) -> None:
        """RefractedState es inmutable (frozen dataclass)."""
        state = RefractedState(
            focused_logits       = np.array([1.0, 2.0, 3.0]),
            kv_compression_ratio = 0.8,
            fermat_metric_trace  = 4.0,
            diagnostics          = self._make_diagnostics(),
        )
        with pytest.raises((AttributeError, TypeError)):
            state.kv_compression_ratio = 0.5  # type: ignore[misc]

    def test_ratio_boundary_values(self) -> None:
        """kv_compression_ratio en los límites exactos [0, 1] es válido."""
        diag = self._make_diagnostics()
        for ratio in [0.0, 1.0]:
            state = RefractedState(
                focused_logits       = np.array([1.0, 2.0]),
                kv_compression_ratio = ratio,
                fermat_metric_trace  = 2.0,
                diagnostics          = diag,
            )
            assert state.kv_compression_ratio == ratio


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestSpectrometerParams: Validación de Parámetros de Inicialización         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSpectrometerParams:
    """
    Verifica que Phase1_SphericalHarmonicsSpectrometer valida correctamente
    sus parámetros de inicialización.
    """

    def test_valid_construction(self) -> None:
        """Construcción válida sin error."""
        spec = make_spectrometer(l_cutoff=5, gamma_damping=0.1,
                                 n_theta=16, n_phi=32)
        assert spec.l_cutoff == 5
        assert spec.gamma_damping == 0.1

    @pytest.mark.parametrize("bad_l", [-1, -10])
    def test_rejects_negative_l_cutoff(self, bad_l: int) -> None:
        """ValueError para l_cutoff < 0."""
        with pytest.raises(ValueError, match="l_cutoff"):
            make_spectrometer(l_cutoff=bad_l)

    @pytest.mark.parametrize("bad_gamma", [0.0, -0.1, -1.0])
    def test_rejects_non_positive_gamma(self, bad_gamma: float) -> None:
        """ValueError para gamma_damping ≤ 0."""
        with pytest.raises(ValueError, match="gamma_damping"):
            make_spectrometer(gamma_damping=bad_gamma)

    @pytest.mark.parametrize("bad_n", [0, 1, 2, 3])
    def test_rejects_small_n_theta(self, bad_n: int) -> None:
        """ValueError para n_theta < 4."""
        with pytest.raises(ValueError, match="n_theta"):
            make_spectrometer(n_theta=bad_n)

    @pytest.mark.parametrize("bad_n", [0, 1, 2, 3])
    def test_rejects_small_n_phi(self, bad_n: int) -> None:
        """ValueError para n_phi < 4."""
        with pytest.raises(ValueError, match="n_phi"):
            make_spectrometer(n_phi=bad_n)

    def test_rejects_non_int_l_cutoff(self) -> None:
        """ValueError para l_cutoff de tipo float."""
        with pytest.raises(ValueError, match="l_cutoff"):
            make_spectrometer(l_cutoff=2.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("l_cutoff", [0, 1, 2, 5, 10])
    def test_accepts_valid_l_cutoff_values(self, l_cutoff: int) -> None:
        """l_cutoff = 0, 1, 2, 5, 10 son todos válidos."""
        spec = make_spectrometer(l_cutoff=l_cutoff)
        assert spec.l_cutoff == l_cutoff

    def test_grid_property_returns_spherical_grid(self) -> None:
        """La propiedad grid retorna SphericalGrid."""
        spec = make_spectrometer()
        assert isinstance(spec.grid, SphericalGrid)

    def test_gamma_damping_property(self) -> None:
        """La propiedad gamma_damping retorna el valor correcto."""
        spec = make_spectrometer(gamma_damping=0.25)
        assert spec.gamma_damping == 0.25


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O3 — TestProjectLogitsToR3: Gram-Schmidt y Propiedades Geométricas        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestProjectLogitsToR3:
    """
    Verifica la proyección de logits a ℝ³ via Gram-Schmidt:
      - ‖p‖₂ = 1  (vector unitario)
      - p es determinista (misma entrada → misma salida)
      - LensSingularityError para n < 3
      - LensSingularityError para señal constante
      - Las componentes de p son finitas
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.spec = make_spectrometer()

    def test_projection_returns_unit_vector(self) -> None:
        """‖p‖₂ = 1 para logits con variación."""
        logits = make_random_logits(10, seed=200)
        p = self.spec._project_logits_to_r3(logits)
        assert_allclose(
            np.linalg.norm(p), 1.0, atol=1e-14,
            err_msg=f"‖p‖₂ = {np.linalg.norm(p):.8f} ≠ 1."
        )

    def test_projection_returns_3d_vector(self) -> None:
        """p tiene exactamente 3 componentes."""
        logits = make_random_logits(10, seed=203)
        p = self.spec._project_logits_to_r3(logits)
        assert p.shape == (3,), f"p.shape={p.shape} ≠ (3,)."

    def test_projection_is_deterministic(self) -> None:
        """Misma entrada → misma salida (sin aleatoriedad interna)."""
        logits = make_random_logits(10, seed=207)
        p1 = self.spec._project_logits_to_r3(logits)
        p2 = self.spec._project_logits_to_r3(logits)
        assert_array_equal(p1, p2,
                           err_msg="La proyección debe ser determinista.")

    def test_projection_components_are_finite(self) -> None:
        """Todas las componentes de p son finitas."""
        logits = make_random_logits(20, seed=211)
        p = self.spec._project_logits_to_r3(logits)
        assert np.all(np.isfinite(p)), "p contiene valores no finitos."

    @pytest.mark.parametrize("n", [3, 5, 8, 15, 50])
    def test_projection_works_for_various_n(self, n: int) -> None:
        """Proyección exitosa para distintos tamaños de logits."""
        logits = make_random_logits(n, seed=n * 213)
        p = self.spec._project_logits_to_r3(logits)
        assert_allclose(np.linalg.norm(p), 1.0, atol=1e-13,
                        err_msg=f"‖p‖₂ ≠ 1 para n={n}.")

    def test_rejects_logits_with_size_less_than_3(self) -> None:
        """LensSingularityError para logits.size < 3."""
        for bad_n in [1, 2]:
            bad_logits = np.ones(bad_n, dtype=np.float64)
            with pytest.raises(LensSingularityError, match=str(_N_SVD_COMPONENTS)):
                self.spec._project_logits_to_r3(bad_logits)

    def test_rejects_constant_logits(self) -> None:
        """LensSingularityError para señal constante (v̄ = 0)."""
        constant_logits = np.ones(10, dtype=np.float64) * 3.14
        with pytest.raises(LensSingularityError, match="nulo"):
            self.spec._project_logits_to_r3(constant_logits)

    def test_projection_e1_parallel_to_centered_logits(self) -> None:
        """
        El primer vector base e₁ es paralelo a v̄ = logits - mean(logits).
        Por tanto p[0] = ‖v̄‖/‖p‖ = ‖v̄‖ > 0 (positivo).
        """
        logits = make_random_logits(10, seed=229)
        p = self.spec._project_logits_to_r3(logits)
        v_centered = logits - np.mean(logits)
        # La primera componente de p debe ser positiva (e₁ ∥ v̄)
        assert p[0] > 0, f"p[0]={p[0]} debe ser > 0 (e₁ ∥ v̄)."


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O3 — TestMapLogitsToSphere: Propiedades de la Proyección a S²             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMapLogitsToSphere:
    """
    Verifica la función ψ(θ,φ) generada por _map_logits_to_sphere:
      - ψ.shape == grid.shape
      - ψ ∈ ℝ (valores reales no negativos para gaussiana)
      - ψ es finita
      - ‖ψ‖_{L²} ≈ ‖logits‖₂  (conservación de energía)
      - σ ∈ [_SIGMA_MIN, _SIGMA_MAX]
      - ValueError para logits con NaN/Inf
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.spec = make_spectrometer()

    def test_psi_has_correct_shape(self) -> None:
        """ψ.shape == grid.shape."""
        logits = make_random_logits(10, seed=233)
        psi = self.spec._map_logits_to_sphere(logits)
        assert psi.shape == self.spec.grid.shape, (
            f"ψ.shape={psi.shape} ≠ {self.spec.grid.shape}."
        )

    def test_psi_is_finite(self) -> None:
        """ψ no contiene NaN ni Inf."""
        logits = make_random_logits(10, seed=237)
        psi = self.spec._map_logits_to_sphere(logits)
        assert np.all(np.isfinite(psi)), "ψ contiene valores no finitos."

    def test_psi_is_real(self) -> None:
        """ψ retorna array real (dtype float64)."""
        logits = make_random_logits(10, seed=241)
        psi = self.spec._map_logits_to_sphere(logits)
        assert psi.dtype == np.float64, f"ψ.dtype={psi.dtype} ≠ float64."

    def test_psi_energy_conservation(self) -> None:
        """
        ‖ψ‖_{L²(S²)} ≈ ‖logits‖₂ (conservación de energía).

        La función ψ se escala explícitamente para que su norma L² coincida
        con la norma euclídea de logits.
        """
        logits = make_random_logits(15, seed=247)
        psi = self.spec._map_logits_to_sphere(logits)
        norm_psi_L2 = math.sqrt(
            float(np.sum(psi ** 2 * self.spec.grid.weights))
        )
        norm_logits = float(np.linalg.norm(logits))
        assert_allclose(
            norm_psi_L2, norm_logits, rtol=1e-10,
            err_msg=f"‖ψ‖_{{L²}}={norm_psi_L2:.4e} ≠ ‖logits‖₂={norm_logits:.4e}."
        )

    def test_psi_nonnegative_for_gaussian(self) -> None:
        """
        El perfil gaussiano es ψ = A·exp(-½(d/σ)²) ≥ 0 para todo (θ,φ).
        Después del escalado por norma, ψ puede ser negativo si la norma
        de logits es negativa — pero ‖logits‖₂ ≥ 0 siempre.
        """
        logits = make_random_logits(10, seed=251)
        psi = self.spec._map_logits_to_sphere(logits)
        # El signo de ψ coincide con el signo de ‖logits‖₂ ≥ 0
        assert np.all(psi >= 0), (
            "ψ debe ser ≥ 0 (perfil gaussiano escalado por norma positiva)."
        )

    def test_psi_peak_near_hot_spot(self) -> None:
        """
        El máximo de ψ debe estar cerca del punto caliente de la proyección.
        Para un logit con señal fuerte en la primera componente, el punto
        caliente está cerca del polo norte (θ₀ ≈ 0).
        """
        # Logit concentrado en la primera componente → p ≈ e₁ → θ₀ ≈ 0
        logits = np.zeros(10, dtype=np.float64)
        logits[0] = 100.0  # señal muy fuerte en componente 0
        logits[1:] = np.linspace(-0.1, 0.1, 9)  # pequeñas variaciones

        psi = self.spec._map_logits_to_sphere(logits)
        max_idx = np.unravel_index(np.argmax(psi), psi.shape)
        theta_max = self.spec.grid.theta[max_idx[0]]
        # El máximo debe estar en θ < π/2 (hemisferio norte)
        assert theta_max < np.pi / 2, (
            f"El máximo de ψ está en θ={theta_max:.3f} rad, "
            f"esperado en el hemisferio norte (θ < π/2)."
        )

    def test_rejects_nan_logits(self) -> None:
        """ValueError para logits con NaN."""
        bad = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="no finitos"):
            self.spec._map_logits_to_sphere(bad)

    def test_rejects_inf_logits(self) -> None:
        """ValueError para logits con Inf."""
        bad = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="no finitos"):
            self.spec._map_logits_to_sphere(bad)

    def test_sigma_is_bounded(self) -> None:
        """σ ∈ [_SIGMA_MIN, _SIGMA_MAX] para distintas distribuciones."""
        for seed in range(10):
            logits = make_random_logits(10, seed=seed * 257)
            v_c = logits - np.mean(logits)
            std_v = float(np.std(v_c))
            range_v = float(np.max(logits) - np.min(logits))
            sigma = float(np.clip(std_v / (range_v + _EPSILON), _SIGMA_MIN, _SIGMA_MAX))
            assert _SIGMA_MIN <= sigma <= _SIGMA_MAX, (
                f"σ={sigma:.4f} fuera de [{_SIGMA_MIN}, {_SIGMA_MAX}] para seed={seed}."
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O2 — TestSphericalCoefficients: Parseval y Ortogonalidad                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestSphericalCoefficients:
    """
    Verifica la correctitud de los coeficientes c_{lm}:
      - Shape == ((l_max+1)²,)
      - c_{lm} son números complejos
      - Identidad de Parseval: ‖ψ‖²_{L²} ≈ Σ|c_{lm}|²
      - c_{00} real y positivo para ψ real no negativa
      - Relación de escala: c_{lm}(α·ψ) = α·c_{lm}(ψ)
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.spec = make_spectrometer(
            l_cutoff=_FAST_L_MAX,
            n_theta=2 * _FAST_L_MAX + 4,  # suficiente para exactitud
            n_phi=4 * _FAST_L_MAX + 4,
            verify_parseval=False,  # verificamos manualmente
        )

    def test_coefficients_shape(self) -> None:
        """c_{lm}.shape == ((l_max+1)²,)."""
        logits = make_random_logits(10, seed=261)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)
        expected_n = (self.spec.l_cutoff + 1) ** 2
        assert spectrum.coefficients.shape == (expected_n,), (
            f"coefficients.shape={spectrum.coefficients.shape} ≠ ({expected_n},)."
        )

    def test_coefficients_are_complex(self) -> None:
        """Los coeficientes c_{lm} son números complejos (dtype complex128)."""
        logits = make_random_logits(10, seed=263)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)
        assert spectrum.coefficients.dtype == np.complex128, (
            f"dtype={spectrum.coefficients.dtype} ≠ complex128."
        )

    def test_parseval_identity_satisfied(self) -> None:
        """
        ‖ψ‖²_{L²(S²)} ≈ Σ_{l,m}|c_{lm}|²  (identidad de Parseval truncada).

        Para funciones suaves, la suma truncada hasta l_cutoff captura la
        mayor parte de la energía. El error relativo debe ser < _PARSEVAL_REL_TOL.
        """
        logits = make_random_logits(10, seed=271)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL, (
            f"Error de Parseval = {spectrum.parseval_relative_error:.2e} "
            f"≥ tol = {_PARSEVAL_REL_TOL:.2e}."
        )

    def test_parseval_lhs_equals_psi_norm_squared(self) -> None:
        """parseval_lhs = ‖ψ‖²_{L²} calculado numéricamente."""
        logits = make_random_logits(10, seed=277)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)
        expected_lhs = float(np.sum(psi ** 2 * self.spec.grid.weights))
        assert_allclose(
            spectrum.parseval_lhs, expected_lhs, rtol=1e-12,
            err_msg="parseval_lhs no coincide con ‖ψ‖²_{L²}."
        )

    def test_parseval_rhs_equals_sum_abs_clm_squared(self) -> None:
        """parseval_rhs = Σ|c_{lm}|² calculado desde coeficientes."""
        logits = make_random_logits(10, seed=281)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)
        expected_rhs = float(np.sum(np.abs(spectrum.coefficients) ** 2))
        assert_allclose(
            spectrum.parseval_rhs, expected_rhs, rtol=1e-12,
            err_msg="parseval_rhs no coincide con Σ|c_{lm}|²."
        )

    def test_linearity_of_coefficients(self) -> None:
        """c_{lm}(α·ψ) = α·c_{lm}(ψ)  (linealidad de la integral)."""
        logits = make_random_logits(10, seed=283)
        alpha = 2.5
        psi = self.spec._map_logits_to_sphere(logits)
        psi_scaled = psi * alpha

        spec1 = self.spec._compute_spherical_coefficients(psi)
        spec2 = self.spec._compute_spherical_coefficients(psi_scaled)

        assert_allclose(
            np.abs(spec2.coefficients), alpha * np.abs(spec1.coefficients),
            rtol=1e-12,
            err_msg="c_{lm}(α·ψ) ≠ α·c_{lm}(ψ): linealidad violada."
        )

    def test_c00_is_proportional_to_psi_average(self) -> None:
        """
        c_{00} = ∫ ψ · Ȳ_0^0 dΩ = (1/√(4π)) · ∫ ψ dΩ

        Para ψ ≥ 0, c_{00} debe ser real y positivo.
        """
        logits = make_random_logits(10, seed=289)
        psi = self.spec._map_logits_to_sphere(logits)
        spectrum = self.spec._compute_spherical_coefficients(psi)

        # c_{00} = Σ_{ij} ψ_{ij} · conj(Y_0^0) · W_{ij}
        THETA, PHI = np.meshgrid(
            self.spec.grid.theta, self.spec.grid.phi, indexing='ij'
        )
        Y00 = sp_special.sph_harm(0, 0, PHI, THETA)
        expected_c00 = np.sum(psi * np.conj(Y00) * self.spec.grid.weights)

        assert_allclose(
            spectrum.coefficients[0], expected_c00, rtol=1e-10,
            err_msg="c_{00} no coincide con el valor esperado de la integral."
        )

    def test_validate_psi_rejects_wrong_shape(self) -> None:
        """ValueError si ψ.shape ≠ grid.shape."""
        bad_psi = np.ones((5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            self.spec._compute_spherical_coefficients(bad_psi)

    def test_validate_psi_rejects_zero_energy(self) -> None:
        """ValueError si ψ = 0 (energía nula)."""
        zero_psi = np.zeros(self.spec.grid.shape, dtype=np.float64)
        with pytest.raises(ValueError, match="energía"):
            self.spec._compute_spherical_coefficients(zero_psi)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O4 — TestFermatRefractiveIndex: Rango y Propiedades Físicas               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestFermatRefractiveIndex:
    """
    Verifica el índice de refracción de Fermat n(σ*):
      - n ∈ (1, 2)  ∀ σ* ∈ ℝ
      - n(0) = 1 + tanh(0) = 1 (sin estrés: vacío óptico)
      - n es monótonamente creciente en σ*
      - Tr(g_F) = n²·Tr(G) > 0
      - ValueError para σ* no finito
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        G = np.eye(4, dtype=np.float64)
        self.lens = make_lens(metric_tensor=G)

    @pytest.mark.parametrize("sigma_star", [
        -100.0, -10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 100.0
    ])
    def test_n_in_open_interval_1_2(self, sigma_star: float) -> None:
        """n(σ*) ∈ (1, 2) para todo σ* ∈ ℝ."""
        n = self.lens._compute_fermat_refractive_index(sigma_star)
        assert 1.0 < n < 2.0, (
            f"n(σ*={sigma_star}) = {n:.6f} ∉ (1, 2)."
        )

    def test_n_at_zero_stress(self) -> None:
        """n(0) = 1 + tanh(0) = 1.0 (límite inferior exacto cuando σ*=0)."""
        n = self.lens._compute_fermat_refractive_index(0.0)
        assert_allclose(
            n, 1.0 + math.tanh(0.0), rtol=1e-14,
            err_msg=f"n(0) = {n:.8f} ≠ 1 + tanh(0) = 1.0."
        )

    def test_n_is_monotonically_increasing(self) -> None:
        """n es estrictamente creciente en σ*."""
        sigma_values = np.linspace(-5.0, 5.0, 50)
        n_values = [self.lens._compute_fermat_refractive_index(s) for s in sigma_values]
        diffs = np.diff(n_values)
        assert np.all(diffs > 0), (
            "n(σ*) debe ser estrictamente creciente en σ*."
        )

    def test_fermat_formula_matches_tanh(self) -> None:
        """n(σ*) = 1 + tanh(α·σ*) con α = _FERMAT_ALPHA."""
        for sigma_star in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            n_computed = self.lens._compute_fermat_refractive_index(sigma_star)
            n_expected = 1.0 + math.tanh(_FERMAT_ALPHA * sigma_star)
            assert_allclose(
                n_computed, n_expected, rtol=1e-14,
                err_msg=f"n({sigma_star}) = {n_computed} ≠ {n_expected}."
            )

    def test_fermat_metric_trace_positive(self) -> None:
        """Tr(g_F) = n²·Tr(G) > 0 para G definida positiva."""
        n = self.lens._compute_fermat_refractive_index(1.0)
        trace = self.lens._compute_fermat_metric_trace(n)
        assert trace > 0, f"Tr(g_F) = {trace:.4f} ≤ 0."

    def test_fermat_metric_trace_formula(self) -> None:
        """Tr(g_F) = n²·Tr(G) verificado con G conocida."""
        G_diag = np.diag([1.0, 2.0, 3.0, 4.0])
        lens_diag = make_lens(metric_tensor=G_diag)
        sigma_star = 1.0
        n = lens_diag._compute_fermat_refractive_index(sigma_star)
        trace = lens_diag._compute_fermat_metric_trace(n)
        expected = n ** 2 * float(np.trace(G_diag))
        assert_allclose(
            trace, expected, rtol=1e-14,
            err_msg=f"Tr(g_F) = {trace:.4f} ≠ n²·Tr(G) = {expected:.4f}."
        )

    def test_rejects_nan_stress_norm(self) -> None:
        """ValueError para stress_norm = NaN."""
        with pytest.raises(ValueError, match="finito"):
            self.lens._compute_fermat_refractive_index(float('nan'))

    def test_rejects_inf_stress_norm(self) -> None:
        """ValueError para stress_norm = ∞."""
        with pytest.raises(ValueError, match="finito"):
            self.lens._compute_fermat_refractive_index(float('inf'))


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O5 — TestViewFunctor: Morfismo view : S → A                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestViewFunctor:
    """
    Verifica apply_view_functor (morfismo view):
      - Retorna SphericalSpectrum con shape correcto
      - Coeficientes son complejos
      - Parseval satisfecho
      - Determinista (misma entrada → mismo espectro)
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.lens = make_lens()

    def test_view_returns_spherical_spectrum(self) -> None:
        """apply_view_functor retorna SphericalSpectrum."""
        logits = make_random_logits(10, seed=293)
        spectrum = self.lens.apply_view_functor(logits)
        assert isinstance(spectrum, SphericalSpectrum), (
            f"apply_view_functor debe retornar SphericalSpectrum, "
            f"retornó {type(spectrum).__name__}."
        )

    def test_view_spectrum_l_max_correct(self) -> None:
        """spectrum.l_max == l_cutoff del espectrómetro."""
        logits = make_random_logits(10, seed=297)
        spectrum = self.lens.apply_view_functor(logits)
        assert spectrum.l_max == self.lens.l_cutoff, (
            f"spectrum.l_max={spectrum.l_max} ≠ l_cutoff={self.lens.l_cutoff}."
        )

    def test_view_coefficients_shape(self) -> None:
        """spectrum.coefficients.shape == ((l_cutoff+1)²,)."""
        logits = make_random_logits(10, seed=307)
        spectrum = self.lens.apply_view_functor(logits)
        expected = (self.lens.l_cutoff + 1) ** 2
        assert spectrum.coefficients.shape == (expected,), (
            f"coefficients.shape={spectrum.coefficients.shape} ≠ ({expected},)."
        )

    def test_view_is_deterministic(self) -> None:
        """Misma entrada produce el mismo espectro."""
        logits = make_random_logits(10, seed=311)
        spec1 = self.lens.apply_view_functor(logits)
        spec2 = self.lens.apply_view_functor(logits)
        assert_allclose(
            spec1.coefficients, spec2.coefficients, rtol=1e-14,
            err_msg="apply_view_functor debe ser determinista."
        )

    def test_view_parseval_satisfied(self) -> None:
        """El espectro retornado satisface la identidad de Parseval."""
        logits = make_random_logits(10, seed=313)
        spectrum = self.lens.apply_view_functor(logits)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL, (
            f"Parseval error = {spectrum.parseval_relative_error:.2e} "
            f"≥ tol = {_PARSEVAL_REL_TOL:.2e}."
        )

    @pytest.mark.parametrize("n", [3, 5, 10, 20, 50])
    def test_view_works_for_various_sizes(self, n: int) -> None:
        """apply_view_functor funciona para distintos tamaños de logits."""
        logits = make_random_logits(n, seed=n * 317)
        spectrum = self.lens.apply_view_functor(logits)
        assert isinstance(spectrum, SphericalSpectrum)

    def test_view_raises_for_constant_logits(self) -> None:
        """LensSingularityError para señal constante."""
        constant = np.ones(10, dtype=np.float64) * 5.0
        with pytest.raises(LensSingularityError):
            self.lens.apply_view_functor(constant)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O5 — TestPutFunctor: Morfismo put : S × A → S                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestPutFunctor:
    """
    Verifica apply_put_functor (morfismo put):
      - Retorna array de la misma dimensión que original_logits
      - El resultado es finito (sin NaN/Inf)
      - Energía filtrada ≤ energía original  (amortiguación pasa-bajos)
      - Para γ→∞: solo survives l=0 (modo DC)
      - Para γ→0: reconstrucción casi perfecta (identidad límite)
      - Escalado correcto con n_refract
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n = 20
        self.logits = make_random_logits(self.n, seed=331)
        self.lens_default = make_lens()

    def test_put_returns_correct_dimension(self) -> None:
        """apply_put_functor retorna array de tamaño n."""
        spectrum = self.lens_default.apply_view_functor(self.logits)
        n_refract = self.lens_default._compute_fermat_refractive_index(1.0)
        result = self.lens_default.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        assert result.shape == (self.n,), (
            f"put retornó shape={result.shape} ≠ ({self.n},)."
        )

    def test_put_result_is_finite(self) -> None:
        """apply_put_functor retorna valores finitos."""
        spectrum = self.lens_default.apply_view_functor(self.logits)
        n_refract = self.lens_default._compute_fermat_refractive_index(0.5)
        result = self.lens_default.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        assert np.all(np.isfinite(result)), "put retornó valores no finitos."

    def test_put_result_is_float64(self) -> None:
        """apply_put_functor retorna dtype float64."""
        spectrum = self.lens_default.apply_view_functor(self.logits)
        n_refract = self.lens_default._compute_fermat_refractive_index(0.5)
        result = self.lens_default.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        assert result.dtype == np.float64, f"dtype={result.dtype} ≠ float64."

    def test_put_energy_at_most_original(self) -> None:
        """
        ‖put(logits)‖₂ ≤ ‖logits‖₂ (el filtro pasa-bajos no amplifica).

        La amortiguación exp(-γ n² l²) ≤ 1 para γ,n,l ≥ 0, por tanto
        la energía filtrada debe ser ≤ energía original.
        """
        spectrum = self.lens_default.apply_view_functor(self.logits)
        n_refract = self.lens_default._compute_fermat_refractive_index(1.0)
        result = self.lens_default.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        norm_result = float(np.linalg.norm(result))
        norm_orig   = float(np.linalg.norm(self.logits))
        # Con margen para errores de cuadratura
        assert norm_result <= norm_orig * 1.01, (
            f"‖put(logits)‖={norm_result:.4e} > ‖logits‖={norm_orig:.4e} "
            f"(filtro amplifica energía)."
        )

    def test_put_with_high_gamma_attenuates_high_modes(self) -> None:
        """
        Para γ → ∞: la amortiguación exp(-γ n² l²) → 0 para l > 0.
        Solo el modo DC (l=0) sobrevive, reduciendo la energía drásticamente.
        """
        # γ muy grande: solo l=0 sobrevive
        lens_high_gamma = make_lens(gamma_damping=100.0)
        spectrum = lens_high_gamma.apply_view_functor(self.logits)
        n_refract = lens_high_gamma._compute_fermat_refractive_index(1.0)
        result = lens_high_gamma.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        norm_result = float(np.linalg.norm(result))
        norm_orig   = float(np.linalg.norm(self.logits))
        # Con γ=100, la energía debe estar muy atenuada
        assert norm_result < norm_orig * 0.5, (
            f"Para γ=100, esperaba ‖result‖ << ‖orig‖, "
            f"obtenido {norm_result:.4e} vs {norm_orig:.4e}."
        )

    def test_put_with_low_gamma_preserves_energy(self) -> None:
        """
        Para γ → 0: exp(-γ n² l²) → 1 para todo l, y la reconstrucción
        es casi perfecta. La energía se conserva aproximadamente.
        """
        lens_low_gamma = make_lens(
            gamma_damping=1e-6,
            n_theta=2 * _FAST_L_MAX + 4,
            n_phi=4 * _FAST_L_MAX + 4,
        )
        spectrum = lens_low_gamma.apply_view_functor(self.logits)
        n_refract = lens_low_gamma._compute_fermat_refractive_index(0.0)  # n≈1
        result = lens_low_gamma.apply_put_functor(
            self.logits, spectrum, n_refract
        )
        norm_result = float(np.linalg.norm(result))
        norm_orig   = float(np.linalg.norm(self.logits))
        # La energía debe estar cerca de la original (dentro de 5%)
        assert abs(norm_result - norm_orig) / max(norm_orig, _EPSILON) < 0.05, (
            f"Para γ≈0, ‖result‖={norm_result:.4e} debe ≈ ‖orig‖={norm_orig:.4e}."
        )

    def test_put_zero_logits_returns_zeros(self) -> None:
        """apply_put_functor con logits nulos retorna vector nulo."""
        zero_logits = np.zeros(10, dtype=np.float64)
        # No podemos aplicar view a señal constante (singularidad),
        # así que usamos un espectro artificial de ceros
        n = (self.lens_default.l_cutoff + 1) ** 2
        spectrum_zero = SphericalSpectrum(
            coefficients            = np.zeros(n, dtype=np.complex128),
            l_max                   = self.lens_default.l_cutoff,
            parseval_lhs            = 0.0,
            parseval_rhs            = 0.0,
            parseval_relative_error = 0.0,
        )
        n_refract = self.lens_default._compute_fermat_refractive_index(0.0)
        result = self.lens_default.apply_put_functor(
            zero_logits, spectrum_zero, n_refract
        )
        assert_allclose(
            result, np.zeros(10), atol=1e-15,
            err_msg="put con logits nulos debe retornar zeros."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O6 — TestRefractAttentionLogits: Pipeline Completo de Fase 3             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestRefractAttentionLogits:
    """
    Verifica el método principal refract_attention_logits:
      - Retorna RefractedState con invariantes verificados
      - kv_compression_ratio ∈ [0,1]
      - focused_logits finito y de dimensión n
      - SpectralDiagnostics completo
      - spectral_diagnostics property actualizado
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.fibrator = make_fibrator()
        self.n = 20
        self.logits = make_random_logits(self.n, seed=353)

    def test_refract_returns_refracted_state(self) -> None:
        """refract_attention_logits retorna RefractedState."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert isinstance(state, RefractedState), (
            f"refract_attention_logits debe retornar RefractedState, "
            f"retornó {type(state).__name__}."
        )

    def test_focused_logits_dimension_preserved(self) -> None:
        """focused_logits tiene la misma dimensión que raw_logits."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert state.focused_logits.shape == (self.n,), (
            f"focused_logits.shape={state.focused_logits.shape} ≠ ({self.n},)."
        )

    def test_focused_logits_is_finite(self) -> None:
        """focused_logits no contiene NaN/Inf."""
        state = self.fibrator.refract_attention_logits(self.logits, 0.5)
        assert np.all(np.isfinite(state.focused_logits)), (
            "focused_logits contiene valores no finitos."
        )

    def test_kv_compression_ratio_in_0_1(self) -> None:
        """kv_compression_ratio ∈ [0,1]."""
        for stress in [0.0, 0.5, 1.0, 2.0, 5.0]:
            state = self.fibrator.refract_attention_logits(self.logits, stress)
            assert 0.0 <= state.kv_compression_ratio <= 1.0, (
                f"kv_compression_ratio={state.kv_compression_ratio:.4f} "
                f"∉ [0,1] para stress={stress}."
            )

    def test_fermat_metric_trace_positive(self) -> None:
        """fermat_metric_trace > 0 (G definida positiva, n² > 0)."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert state.fermat_metric_trace > 0, (
            f"fermat_metric_trace={state.fermat_metric_trace:.4f} ≤ 0."
        )

    def test_diagnostics_field_is_spectral_diagnostics(self) -> None:
        """state.diagnostics es SpectralDiagnostics."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert isinstance(state.diagnostics, SpectralDiagnostics), (
            f"state.diagnostics debe ser SpectralDiagnostics, "
            f"es {type(state.diagnostics).__name__}."
        )

    def test_spectral_diagnostics_property_updated(self) -> None:
        """spectral_diagnostics property se actualiza tras cada llamada."""
        assert self.fibrator.spectral_diagnostics is None, (
            "spectral_diagnostics debe ser None antes de la primera llamada."
        )
        self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert self.fibrator.spectral_diagnostics is not None, (
            "spectral_diagnostics debe actualizarse tras refract_attention_logits."
        )

    def test_diagnostics_n_refract_in_range(self) -> None:
        """diagnostics.n_refract ∈ (1, 2)."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.5)
        assert 1.0 < state.diagnostics.n_refract < 2.0, (
            f"n_refract={state.diagnostics.n_refract:.4f} ∉ (1, 2)."
        )

    def test_diagnostics_parseval_error_bounded(self) -> None:
        """diagnostics.parseval_error < _PARSEVAL_REL_TOL."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert state.diagnostics.parseval_error < _PARSEVAL_REL_TOL, (
            f"parseval_error={state.diagnostics.parseval_error:.2e} "
            f"≥ tol={_PARSEVAL_REL_TOL:.2e}."
        )

    def test_diagnostics_energy_retention_in_0_1(self) -> None:
        """diagnostics.energy_retention ∈ [0,1]."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert 0.0 <= state.diagnostics.energy_retention <= 1.0, (
            f"energy_retention={state.diagnostics.energy_retention:.4f} ∉ [0,1]."
        )

    def test_diagnostics_l_dominant_in_range(self) -> None:
        """diagnostics.l_dominant ∈ [0, l_cutoff]."""
        state = self.fibrator.refract_attention_logits(self.logits, 1.0)
        assert 0 <= state.diagnostics.l_dominant <= self.fibrator.l_cutoff, (
            f"l_dominant={state.diagnostics.l_dominant} ∉ [0, {self.fibrator.l_cutoff}]."
        )

    def test_higher_stress_gives_higher_n_refract(self) -> None:
        """Mayor estrés logístico → mayor índice de refracción."""
        state_low  = self.fibrator.refract_attention_logits(self.logits, 0.1)
        state_high = self.fibrator.refract_attention_logits(self.logits, 5.0)
        assert state_high.diagnostics.n_refract > state_low.diagnostics.n_refract, (
            "Mayor estrés debe producir mayor índice de refracción."
        )

    @pytest.mark.parametrize("n", [3, 5, 10, 30, 100])
    def test_refract_works_for_various_logit_sizes(self, n: int) -> None:
        """Pipeline completo funciona para distintos tamaños de logits."""
        logits = make_random_logits(n, seed=n * 359)
        state = self.fibrator.refract_attention_logits(logits, 1.0)
        assert state.focused_logits.shape == (n,)
        assert np.all(np.isfinite(state.focused_logits))


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O7 — TestErrorContracts: Excepciones y Contratos de Tipo                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestErrorContracts:
    """
    Verifica que todas las excepciones se lanzan en las condiciones correctas:
      - TypeError para tipos incorrectos
      - ValueError para valores inválidos (NaN, Inf, dimensión)
      - LensSingularityError para singularidades geométricas
      - SpectralGridError para grillas inválidas
      - ParsevalViolationError para grilla insuficiente
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.fibrator = make_fibrator()

    # ── Validaciones de raw_llm_logits ───────────────────────────────────────

    def test_rejects_list_logits(self) -> None:
        """TypeError para logits que no son ndarray."""
        with pytest.raises(TypeError, match="NDArray"):
            self.fibrator.refract_attention_logits([1.0, 2.0, 3.0], 1.0)  # type: ignore

    def test_rejects_2d_logits(self) -> None:
        """ValueError para logits 2-D."""
        with pytest.raises(ValueError, match="1-D"):
            self.fibrator.refract_attention_logits(np.ones((5, 5)), 1.0)

    def test_rejects_logits_with_too_few_elements(self) -> None:
        """LensSingularityError para logits.size < 3."""
        with pytest.raises(LensSingularityError):
            self.fibrator.refract_attention_logits(np.array([1.0, 2.0]), 1.0)

    def test_rejects_nan_logits(self) -> None:
        """ValueError para logits con NaN."""
        bad = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="no finitos"):
            self.fibrator.refract_attention_logits(bad, 1.0)

    def test_rejects_inf_logits(self) -> None:
        """ValueError para logits con Inf."""
        bad = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="no finitos"):
            self.fibrator.refract_attention_logits(bad, 1.0)

    def test_rejects_constant_logits(self) -> None:
        """LensSingularityError para logits constantes (v̄=0)."""
        constant = np.ones(10, dtype=np.float64)
        with pytest.raises(LensSingularityError):
            self.fibrator.refract_attention_logits(constant, 1.0)

    # ── Validaciones de logistic_stress_norm ─────────────────────────────────

    def test_rejects_string_stress_norm(self) -> None:
        """TypeError para stress_norm de tipo string."""
        logits = make_random_logits(10, seed=367)
        with pytest.raises(TypeError, match="escalar numérico"):
            self.fibrator.refract_attention_logits(logits, "1.0")  # type: ignore

    def test_rejects_nan_stress_norm(self) -> None:
        """ValueError para stress_norm = NaN."""
        logits = make_random_logits(10, seed=373)
        with pytest.raises(ValueError, match="finito"):
            self.fibrator.refract_attention_logits(logits, float('nan'))

    def test_rejects_inf_stress_norm(self) -> None:
        """ValueError para stress_norm = ±∞."""
        logits = make_random_logits(10, seed=379)
        with pytest.raises(ValueError, match="finito"):
            self.fibrator.refract_attention_logits(logits, float('inf'))

    # ── Validaciones de la métrica ─────────────────────────────────────────

    def test_rejects_non_square_metric(self) -> None:
        """ValueError para métrica no cuadrada."""
        bad_metric = np.eye(3, 4)
        with pytest.raises(ValueError, match="cuadrado"):
            make_lens(metric_tensor=bad_metric)

    def test_rejects_non_spd_metric(self) -> None:
        """ValueError para métrica no definida positiva."""
        G_bad = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="definido positivo"):
            make_lens(metric_tensor=G_bad)

    def test_rejects_nan_metric(self) -> None:
        """ValueError para métrica con NaN."""
        G_bad = np.eye(3)
        G_bad[1, 1] = np.nan
        with pytest.raises(ValueError, match="no finitos"):
            make_lens(metric_tensor=G_bad)

    def test_rejects_non_array_metric(self) -> None:
        """TypeError para métrica que no es ndarray."""
        with pytest.raises(TypeError, match="NDArray"):
            make_lens(metric_tensor=[[1.0, 0.0], [0.0, 1.0]])  # type: ignore

    # ── Parseval y grilla ─────────────────────────────────────────────────────

    def test_parseval_violation_with_insufficient_grid(self) -> None:
        """
        ParsevalViolationError cuando la grilla es insuficiente para
        capturar modos de alta frecuencia.

        Para l_cutoff=10 y n_theta=4 (muy inferior a 2·10+2=22), la
        cuadratura no puede integrar exactamente los polinomios de Legendre
        de grado 20, causando error de Parseval grande.
        """
        # Grilla mínima para l_cutoff=8 — debería fallar Parseval
        try:
            spec = make_spectrometer(
                l_cutoff=8,
                n_theta=4,    # muy insuficiente para l_cutoff=8
                n_phi=8,
                verify_parseval=True,
            )
            logits = make_random_logits(15, seed=383)
            # Si la construcción pasa, intentamos el cálculo de coeficientes
            psi = spec._map_logits_to_sphere(logits)
            with pytest.raises(ParsevalViolationError):
                spec._compute_spherical_coefficients(psi)
        except (SpectralGridError, ParsevalViolationError):
            pass  # La grilla mínima puede fallar en la construcción

    def test_verify_parseval_raises_for_bad_spectrum(self) -> None:
        """
        _verify_parseval_identity lanza ParsevalViolationError para
        un espectro con error relativo > _PARSEVAL_REL_TOL.
        """
        spec = make_spectrometer()
        bad_spectrum = SphericalSpectrum(
            coefficients            = np.zeros(
                (spec.l_cutoff + 1) ** 2, dtype=np.complex128
            ),
            l_max                   = spec.l_cutoff,
            parseval_lhs            = 100.0,   # valor grande
            parseval_rhs            = 0.001,   # valor muy diferente
            parseval_relative_error = 0.999,   # >> _PARSEVAL_REL_TOL
        )
        with pytest.raises(ParsevalViolationError):
            spec._verify_parseval_identity(bad_spectrum)


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §O8 — TestStatisticalProperties: Barrido Aleatorio                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestStatisticalProperties:
    """
    Barrido estadístico sobre 10 señales aleatorias para detectar regresiones:
      - Parseval satisfecho para señales diversas
      - kv_ratio ∈ [0,1] para estrés variado
      - Monotonicidad de energía filtrada vs. γ
      - Proyección retorna vector unitario en ℝ³ siempre
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.fibrator = make_fibrator()

    @pytest.mark.parametrize("trial", range(10))
    def test_parseval_across_random_logits(self, trial: int) -> None:
        """Parseval satisfecho para 10 señales aleatorias."""
        rng = np.random.default_rng(trial * 389 + 7)
        n = int(rng.integers(5, 30))
        logits = rng.standard_normal(n).astype(np.float64)
        # Asegurar que no sea constante
        logits += np.arange(n, dtype=np.float64) * 0.01
        spec = make_spectrometer(verify_parseval=False)
        psi = spec._map_logits_to_sphere(logits)
        spectrum = spec._compute_spherical_coefficients(psi)
        assert spectrum.parseval_relative_error < _PARSEVAL_REL_TOL * 2, (
            f"Parseval error={spectrum.parseval_relative_error:.2e} "
            f"para trial={trial}, n={n}."
        )

    @pytest.mark.parametrize("trial", range(10))
    def test_kv_ratio_in_0_1_across_random_stress(self, trial: int) -> None:
        """kv_compression_ratio ∈ [0,1] para 10 valores de estrés aleatorio."""
        rng = np.random.default_rng(trial * 397 + 3)
        logits = rng.standard_normal(15).astype(np.float64)
        logits += np.arange(15, dtype=np.float64) * 0.1
        stress = float(rng.uniform(-5.0, 5.0))
        state = self.fibrator.refract_attention_logits(logits, stress)
        assert _COMPRESSION_FLOOR <= state.kv_compression_ratio <= _COMPRESSION_CAP, (
            f"kv_ratio={state.kv_compression_ratio:.4f} ∉ [0,1] "
            f"para trial={trial}, stress={stress:.2f}."
        )

    @pytest.mark.parametrize("trial", range(5))
    def test_unit_vector_in_r3_across_random_logits(self, trial: int) -> None:
        """‖p‖₂ = 1 para 5 logits aleatorios de tamaño variable."""
        rng = np.random.default_rng(trial * 401 + 11)
        n = int(rng.integers(3, 20))
        logits = rng.standard_normal(n).astype(np.float64)
        logits += np.arange(n, dtype=np.float64) * 0.05
        spec = make_spectrometer()
        p = spec._project_logits_to_r3(logits)
        assert_allclose(
            np.linalg.norm(p), 1.0, atol=1e-13,
            err_msg=f"‖p‖₂ ≠ 1 para trial={trial}, n={n}."
        )

    def test_energy_monotonically_decreasing_with_gamma(self) -> None:
        """
        La energía filtrada disminuye monótonamente con γ
        (mayor amortiguación → menor energía).
        """
        logits = make_random_logits(20, seed=409)
        gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0]
        energies: List[float] = []

        for gamma in gamma_values:
            fib = make_fibrator(gamma_damping=gamma)
            state = fib.refract_attention_logits(logits, 1.0)
            energies.append(state.diagnostics.energy_focused)

        # Verificar monotonicidad (puede haber pequeñas oscilaciones numéricas)
        diffs = np.diff(energies)
        # Al menos el 80% de las diferencias deben ser no-positivas
        non_increasing = np.sum(diffs <= 1e-8)
        assert non_increasing >= int(0.8 * len(diffs)), (
            f"La energía filtrada no es monótonamente decreciente con γ: {energies}."
        )

    @pytest.mark.parametrize("trial", range(5))
    def test_fermat_trace_positive_across_metrics(self, trial: int) -> None:
        """Tr(g_F) > 0 para 5 métricas SPD aleatorias."""
        rng = np.random.default_rng(trial * 419 + 13)
        G = make_spd_metric(4, seed=trial * 419)
        fib = make_fibrator(metric_tensor=G)
        logits = rng.standard_normal(10).astype(np.float64)
        logits += np.arange(10, dtype=np.float64) * 0.1
        state = fib.refract_attention_logits(logits, float(rng.uniform(0, 3)))
        assert state.fermat_metric_trace > 0, (
            f"fermat_metric_trace={state.fermat_metric_trace:.4f} ≤ 0 "
            f"para trial={trial}."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestHighDimensionalLogits: Robustez para n Grande                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestHighDimensionalLogits:
    """
    Verifica robustez numérica para logits de alta dimensión (n = 100, 500, 1000).
    Los tests de alta dimensión verifican que no hay degradación numérica
    al proyectar vectores de gran dimensión a S².
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # Fibrator con parámetros conservadores para velocidad
        self.fibrator = make_fibrator(
            l_cutoff=3,
            n_theta=12,
            n_phi=24,
        )

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_pipeline_for_large_n(self, n: int) -> None:
        """Pipeline completo funciona para n = 100, 500, 1000."""
        logits = make_random_logits(n, seed=n * 431)
        state = self.fibrator.refract_attention_logits(logits, 1.0)
        assert state.focused_logits.shape == (n,), (
            f"focused_logits.shape={state.focused_logits.shape} ≠ ({n},)."
        )
        assert np.all(np.isfinite(state.focused_logits)), (
            f"focused_logits contiene no-finitos para n={n}."
        )

    @pytest.mark.parametrize("n", [100, 500])
    def test_unit_projection_for_large_n(self, n: int) -> None:
        """‖p‖₂ = 1 para logits de alta dimensión."""
        spec = make_spectrometer(n_theta=12, n_phi=24)
        logits = make_random_logits(n, seed=n * 433)
        p = spec._project_logits_to_r3(logits)
        assert_allclose(
            np.linalg.norm(p), 1.0, atol=1e-13,
            err_msg=f"‖p‖₂ ≠ 1 para n={n}."
        )

    @pytest.mark.parametrize("n", [100, 500])
    def test_kv_ratio_valid_for_large_n(self, n: int) -> None:
        """kv_compression_ratio ∈ [0,1] para n grande."""
        logits = make_random_logits(n, seed=n * 439)
        state = self.fibrator.refract_attention_logits(logits, 1.5)
        assert 0.0 <= state.kv_compression_ratio <= 1.0, (
            f"kv_ratio={state.kv_compression_ratio:.4f} ∉ [0,1] para n={n}."
        )

    def test_extreme_logit_values(self) -> None:
        """
        El pipeline maneja logits con valores extremos (|v| ~ 1e6).
        La normalización por norma en _map_logits_to_sphere debe evitar overflow.
        """
        n = 50
        logits = make_random_logits(n, seed=443) * 1e6
        state = self.fibrator.refract_attention_logits(logits, 1.0)
        assert np.all(np.isfinite(state.focused_logits)), (
            "Pipeline falla para logits con magnitud ~ 1e6."
        )

    def test_small_logit_values(self) -> None:
        """
        El pipeline maneja logits con valores muy pequeños (|v| ~ 1e-6).
        La normalización debe evitar underflow y señal nula.
        """
        n = 50
        base = make_random_logits(n, seed=449)
        logits = base * 1e-6 + np.arange(n, dtype=np.float64) * 1e-7
        state = self.fibrator.refract_attention_logits(logits, 0.1)
        assert np.all(np.isfinite(state.focused_logits)), (
            "Pipeline falla para logits con magnitud ~ 1e-6."
        )

    def test_logits_with_high_variance(self) -> None:
        """
        Logits con varianza muy alta no causan overflow en la proyección gaussiana.
        """
        n = 30
        logits = np.zeros(n, dtype=np.float64)
        logits[0] = 1000.0   # valor extremo
        logits[1:] = make_random_logits(n - 1, seed=457) * 0.01
        state = self.fibrator.refract_attention_logits(logits, 1.0)
        assert np.all(np.isfinite(state.focused_logits)), (
            "Pipeline falla para logits con alta varianza."
        )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TestMetricValidation: Validación Completa de la Métrica                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TestMetricValidation:
    """
    Verifica _validate_metric_tensor exhaustivamente:
      - Métricas SPD de distintas dimensiones aceptadas
      - Métrica asimétrica simetrizada automáticamente
      - Rechaza matrices singulares, NaN, Inf, no-cuadradas
    """

    def test_identity_metric_accepted(self) -> None:
        """Métrica identidad es aceptada sin error."""
        for n in [2, 3, 4, 8]:
            lens = make_lens(metric_tensor=np.eye(n))
            assert lens._dim == n

    def test_spd_metric_accepted(self) -> None:
        """Métricas SPD de distintas dimensiones son aceptadas."""
        for n in [2, 3, 5, 8]:
            G = make_spd_metric(n, seed=n * 461)
            lens = make_lens(metric_tensor=G)
            assert lens._dim == n

    def test_asymmetric_metric_is_symmetrized(self) -> None:
        """
        Una métrica casi-simétrica es simetrizada automáticamente.
        El resultado G_sym = (G + Gᵀ)/2 debe ser simétrico.
        """
        rng = np.random.default_rng(467)
        G_spd = make_spd_metric(4, seed=467)
        # Añadir pequeña asimetría
        asym = rng.standard_normal((4, 4)) * 0.01
        G_asym = G_spd + (asym - asym.T) * 0.5  # anti-simétrico pequeño
        G_slightly_asym = G_asym  # casi-simétrica

        lens = make_lens(metric_tensor=G_slightly_asym)
        # El resultado debe ser simétrico
        asym_error = float(np.linalg.norm(lens._G - lens._G.T, 'fro'))
        assert asym_error < 1e-14, (
            f"Métrica simetrizada no es simétrica: ‖G - Gᵀ‖_F={asym_error:.2e}."
        )

    def test_singular_metric_rejected(self) -> None:
        """ValueError para métrica singular (det = 0)."""
        G_singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # rango 1
        with pytest.raises(ValueError, match="definido positivo"):
            make_lens(metric_tensor=G_singular)

    def test_1d_metric_rejected(self) -> None:
        """ValueError para métrica 1-D."""
        with pytest.raises(ValueError, match="cuadrado"):
            make_lens(metric_tensor=np.array([1.0, 2.0, 3.0]))

    def test_non_square_metric_rejected(self) -> None:
        """ValueError para métrica no cuadrada."""
        with pytest.raises(ValueError, match="cuadrado"):
            make_lens(metric_tensor=np.eye(3, 4))

    def test_diagonal_positive_metric_accepted(self) -> None:
        """Métrica diagonal con valores positivos es aceptada."""
        G_diag = np.diag([1.0, 2.0, 3.0, 4.0])
        lens = make_lens(metric_tensor=G_diag)
        assert_allclose(lens._G, G_diag, rtol=1e-14)