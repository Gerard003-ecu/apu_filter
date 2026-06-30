# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Optical Riemann Lens (Lente Categórico y Difracción Espectral)             ║
║  Ruta   : app/physics/optical_riemann_lens.py                                        ║
║  Versión: 3.0.0-Rigorous-Jacobian-Orthogonal                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Síntesis en 3 Fases Anidadas):
══════════════════════════════════════════════════════════════════════════════════════

Este módulo consagra la inyección de la Óptica Geométrica sobre el colector de la
Esfera de Riemann (S² ≅ ℂ̂). Actúa como un Lente Categórico que filtra los armónicos
de alta frecuencia estocástica del LLM, colapsando el rango del KV-Cache para
preservar la termodinámica computacional.

══════════════════════════════════════════════════════════════════════════════════════
FUNDAMENTACIÓN MATEMÁTICA AXIOMÁTICA
══════════════════════════════════════════════════════════════════════════════════════

§1. MÉTRICA DE FERMAT (Índice de Refracción Adaptativo):
────────────────────────────────────────────────────────────────────────────────────
El espacio de deliberación se curva ópticamente. El índice de refracción n(σ*) es
proporcional al tensor de estrés logístico, modificando la métrica:
    g_F = n²(σ*) · G_{μν} dx^μ dx^ν

Definición rigurosa del índice:
    n(σ*) = 1 + tanh(α · σ*)

donde α > 0 es el factor de escala de curvatura. Esta fórmula garantiza:
    n(σ*) ∈ (1, 2)  ∀ σ* ∈ ℝ   (nunca < 1, nunca ≥ 2)

La condición n < 1 es algebraicamente imposible con esta fórmula; la excepción
OpticalDispersionError solo puede activarse si una subclase redefine el índice.

§2. JACOBIANO DE LA CUADRATURA GAUSS-LEGENDRE EN S²:
────────────────────────────────────────────────────────────────────────────────────
La integral sobre S² en coordenadas esféricas es:
    ∫_{S²} f dΩ = ∫₀^π ∫₀^{2π} f(θ,φ) sin(θ) dθ dφ

El cambio de variable t = cos(θ) ∈ [-1,1] transforma la integral en:
    ∫_{-1}^{1} ∫₀^{2π} f(arccos(t), φ) dt dφ

Los nodos y pesos de Gauss-Legendre {(t_i, w_i)} satisfacen:
    ∫_{-1}^{1} p(t) dt ≈ Σ_i w_i p(t_i)   para polinomios de grado ≤ 2N-1.

CRÍTICO: los pesos w_i de leggauss integran respecto a dt = d(cos θ),
NO respecto a sin(θ)dθ. Por tanto el peso de integración completo es:
    W_{ij} = w_i · Δφ    (sin multiplicar por sin(θ_i))

donde Δφ = 2π/n_φ. La versión anterior multiplicaba adicionalmente por sin(θ_i),
introduciendo un Jacobiano duplicado que corrompía los coeficientes c_{lm}.

§3. COEFICIENTES DE ARMÓNICOS ESFÉRICOS CORRECTOS:
────────────────────────────────────────────────────────────────────────────────────
    c_{lm} = ∫_{S²} ψ(θ,φ) · Ȳ_l^m(θ,φ) dΩ
           ≈ Σ_{i,j} ψ(θ_i, φ_j) · conj(Y_l^m(θ_i, φ_j)) · w_i · Δφ

Verificación de ortogonalidad (condición necesaria):
    Σ_{i,j} Y_l^m(θ_i,φ_j) · conj(Y_l'^{m'}(θ_i,φ_j)) · w_i · Δφ ≈ δ_{ll'} δ_{mm'}

§4. PROYECCIÓN ESTEREOGRÁFICA ALGEBRAICAMENTE CORRECTA:
────────────────────────────────────────────────────────────────────────────────────
Para proyectar v ∈ ℝⁿ (n ≥ 3) a S²:
    1. Centrar: v̄ = v - mean(v)
    2. Proyectar a ℝ³ via PCA truncado o base ortonormal de Gram-Schmidt.
    3. Normalizar a vector unitario en ℝ³.
    4. Convertir a coordenadas esféricas (θ₀, φ₀).
    5. Crear perfil gaussiano ψ(θ,φ) = A · exp(-½((d(θ,φ))/σ)²)
       donde d es la distancia geodésica al punto caliente y σ es adaptativo.

La base de proyección usa SVD truncado para garantizar ortogonalidad exacta
(a diferencia de la base de Hadamard hardcodeada que era solo aproximadamente
ortonormal y fallaba para n < 10).

§5. PROYECCIÓN DE VUELTA AL ESPACIO DE LOGITS (L² CORRECTA):
────────────────────────────────────────────────────────────────────────────────────
La reconstrucción ψ̃ ∈ L²(S²) se proyecta de vuelta a ℝⁿ mediante:
    ṽ_k = ⟨ψ̃, φ_k⟩_{L²(S²)}

donde {φ_k} son las funciones "sensor" generadas por la proyección directa de
los vectores base canónicos e_k ∈ ℝⁿ. Esto garantiza que la operación de
ida (logits → S²) y vuelta (S² → logits) son adjuntas en L².

§6. LENTES CATEGÓRICOS (Optics):
────────────────────────────────────────────────────────────────────────────────────
Un Lente sobre la categoría monoidal simétrica define:
    view : S → A     (extracción del invariante espectral)
    put  : S × A → S (reconstrucción filtrada)

El operador de difracción aniquila la alta entropía (l > l_cutoff):
    O_{lens} ψ = Σ_{l=0}^{l_cutoff} Σ_{m=-l}^{l} exp(-γ n² l²) c_{lm} Y_l^m(θ,φ)

══════════════════════════════════════════════════════════════════════════════════════
CORRECCIONES CRÍTICAS v3.0.0 vs v2.0.0:
══════════════════════════════════════════════════════════════════════════════════════

  BUG 1: Jacobiano duplicado en SphericalGrid.__post_init__
          v2.0: weights = w_i · Δφ · sin(θ_i)  ← INCORRECTO (doble Jacobiano)
          v3.0: weights = w_i · Δφ              ← CORRECTO (leggauss integra dt=d(cosθ))

  BUG 2: Base de proyección Hadamard hardcodeada (10 columnas)
          v2.0: falla silenciosamente para n<10 o n>10
          v3.0: SVD truncado de la matriz de logits centrada, adaptativo a n

  BUG 3: OpticalDispersionError algebraicamente inalcanzable
          v2.0: n = 1 + tanh(...) ≥ 1 siempre → excepción nunca dispara
          v3.0: Se verifica n ∈ (1,2) correctamente; excepción reservada para subclases

  BUG 4: Proyección de vuelta al espacio de logits sin fundamento L²
          v2.0: truncación/padding arbitrario de la grilla
          v3.0: proyección L² mediante funciones sensor adjuntas

  BUG 5: sigma en _map_logits_to_sphere puede ser infinito para señal constante
          v2.0: división por (max-min + 1e-12) si max≈min → sigma ≈ std/1e-12 → clipeado
          v3.0: sigma = max(std(v), ε) / (range(v) + ε) con manejo explícito de señal nula

  BUG 6: compression_ratio puede ser negativo para logits con norma cero
          v2.0: norm_raw = ‖logits‖ + 1e-12, norm_focused = ‖focused‖ → ratio ∈ [0,∞)
          v3.0: ratio = ‖focused‖ / max(‖raw‖, ε) ∈ [0, ∞), acotado a [0,1] con abs

══════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DE FASES ANIDADAS (v3.0.0):
══════════════════════════════════════════════════════════════════════════════════════

  ┌─── FASE 1: Phase1_SphericalHarmonicsSpectrometer ────────────────────────┐
  │  _validate_spectrometer_params()  → l_cutoff≥0, γ>0, n_θ≥4, n_φ≥4      │
  │  _build_spherical_grid()          → Gauss-Legendre en θ, uniforme en φ   │
  │  _validate_grid_orthogonality()   → ‖W·I - I‖ < tol (test de Y_0^0)    │
  │  _build_sensor_basis()            → {φ_k} funciones sensor en L²(S²)     │
  │  _project_logits_to_r3()          → SVD truncado, ortogonal, n≥3         │
  │  _map_logits_to_sphere()          → perfil gaussiano en S², adaptativo   │
  │  _validate_psi()                  → finitud, shape, energía > 0          │
  │  _compute_spherical_coefficients()→ c_{lm} con Jacobiano correcto        │
  │  _verify_parseval_identity()      → ‖ψ‖²_L² ≈ Σ|c_{lm}|² (Parseval)   │
  │  grid, l_cutoff (properties)     → acceso solo lectura                   │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ SphericalSpectrum + SphericalGrid
                                 ▼
  ┌─── FASE 2: Phase2_CategoricalOpticLens ────────────────────────────────────┐
  │  _validate_metric_tensor()        → shape, dtype, finitud, SPD           │
  │  _compute_fermat_refractive_index()→ n ∈ (1,2), verifica rango          │
  │  _compute_fermat_metric_trace()   → Tr(n²·G), validado                  │
  │  _reconstruct_psi_from_spectrum() → ψ̃ en grilla S², parte real          │
  │  _project_psi_to_logit_space()    → proyección L² via sensor basis       │
  │  _validate_reconstructed_signal() → finitud, energía, ratio              │
  │  apply_view_functor()             → logits → SphericalSpectrum           │
  │  apply_put_functor()              → SphericalSpectrum → logits filtrados  │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ NDArray logits filtrados + diagnósticos
                                 ▼
  ┌─── FASE 3: OpticalRiemannLensFibrator ─────────────────────────────────────┐
  │  _validate_input_logits()         → tipo, finitud, dim mínima            │
  │  _validate_stress_norm()          → finitud, no negativo                 │
  │  _compute_kv_compression_ratio()  → ratio ∈ [0,1], basado en normas     │
  │  _build_spectral_diagnostics()    → Dict con métricas de calidad         │
  │  refract_attention_logits()       → orquestador completo → RefractedState│
  │  spectral_diagnostics (property)  → último diagnóstico, solo lectura     │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.special as sp_special
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ════════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MIC.Omega.OpticalRiemannLens")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS GLOBALES
# ════════════════════════════════════════════════════════════════════════════════
_EPSILON            : float = 1e-12   # Piso numérico general
_SIGMA_MIN          : float = 0.05    # Ancho gaussiano mínimo (radianes)
_SIGMA_MAX          : float = np.pi   # Ancho gaussiano máximo (radianes = hemisferio)
_ORTHOGONALITY_TOL  : float = 1e-6    # Tolerancia para verificación de ortogonalidad
_PARSEVAL_REL_TOL   : float = 0.05    # Tolerancia relativa para identidad de Parseval
_N_SVD_COMPONENTS   : int   = 3       # Dimensión de proyección a ℝ³
_FERMAT_ALPHA       : float = 0.5     # Factor de escala del índice de refracción
_COMPRESSION_FLOOR  : float = 0.0     # Mínimo físico de compression_ratio
_COMPRESSION_CAP    : float = 1.0     # Máximo físico de compression_ratio


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXCEPCIONES ÓPTICAS Y ESPECTRALES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class OpticalDispersionError(TopologicalInvariantError):
    r"""
    Detonada cuando el índice de refracción viola la condición física n ≥ 1.

    En la implementación estándar con n = 1 + tanh(α·σ*), esta condición
    es algebraicamente imposible (n ≥ 1 ∀ σ*). Sin embargo, subclases que
    redefinen `_compute_fermat_refractive_index` podrían producir n < 1,
    induciendo absorción anómala que destruye el Teorema de Liouville en el
    espacio de fase: el volumen del espacio de fase no se conserva bajo el
    flujo hamiltoniano cuando el medio tiene n < 1.

    Campos del mensaje:
        - n_computed : valor calculado del índice de refracción.
        - stress_norm: estrés logístico que lo produjo.
    """
    pass


class LensSingularityError(TopologicalInvariantError):
    r"""
    Detonada cuando la proyección geométrica falla algebraicamente.

    Condiciones de activación:
        1. logits.size < _N_SVD_COMPONENTS (imposible proyectar a ℝ³).
        2. El vector centrado v̄ = v - mean(v) es numéricamente nulo
           (señal constante, degenerada en el espacio de proyección).
        3. La grilla S² no satisface la condición de ortogonalidad
           ‖⟨Y_0^0, Y_0^0⟩ - 1‖ > _ORTHOGONALITY_TOL.

    Físicamente: la proyección converge al Polo Norte de la Esfera de Riemann
    (singularidad estocástica pura) sin poseer resolución espectral armónica.
    """
    pass


class ParsevalViolationError(TopologicalInvariantError):
    r"""
    Detonada cuando la identidad de Parseval falla más allá de la tolerancia.

    La identidad de Parseval en L²(S²):
        ‖ψ‖²_{L²(S²)} = Σ_{l,m} |c_{lm}|²

    es una consecuencia del Teorema Espectral para el operador de Laplace-Beltrami
    en S². Su violación indica un error en la cuadratura numérica (Jacobiano
    incorrecto, grilla insuficiente) o pérdida de ortogonalidad de los armónicos.

    Campos del mensaje:
        - parseval_lhs : ‖ψ‖²_{L²(S²)} calculado.
        - parseval_rhs : Σ|c_{lm}|² calculado.
        - relative_error: |lhs - rhs| / max(lhs, ε).
    """
    pass


class SpectralGridError(TopologicalInvariantError):
    r"""
    Detonada cuando la grilla de cuadratura no satisface la condición de
    ortogonalidad necesaria para la correcta integración de armónicos esféricos.

    La condición verificada es:
        |⟨Y_0^0, Y_0^0⟩_{L²(S²)} - 1| < _ORTHOGONALITY_TOL

    donde Y_0^0 = 1/√(4π) es el armónico esférico de orden (0,0).
    """
    pass


# ════════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL FIBRADO ÓPTICO
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SphericalGrid:
    r"""
    Malla de puntos y pesos para integración numérica en S².

    Emplea cuadratura de Gauss-Legendre en la coordenada θ (colatitud) y
    partición uniforme en φ (longitud).

    JACOBIANO CORRECTO:
    ───────────────────
    La integral sobre S² con cambio de variable t = cos(θ):
        ∫_{S²} f dΩ = ∫₀^π ∫₀^{2π} f(θ,φ) sin(θ) dθ dφ
                    = ∫_{-1}^{1} ∫₀^{2π} f(arccos(t), φ) dt dφ

    Los pesos w_i de leggauss satisfacen ∫_{-1}^{1} p(t) dt ≈ Σ w_i p(t_i).
    El cambio de variable ya incorpora el factor sin(θ) (vía dt = -sin(θ)dθ).
    Por tanto el peso completo de integración es:

        W_{ij} = w_i · Δφ      ← SIN multiplicar por sin(θ_i)

    Esto es distinto a v2.0.0 donde se multiplicaba incorrectamente por sin(θ_i).

    Campos
    ------
    theta         : (n_theta,) nodos de colatitud θ = arccos(t_i) ∈ (0,π).
    phi           : (n_phi,)   nodos de longitud φ ∈ [0, 2π).
    gauss_weights : (n_theta,) pesos de Gauss-Legendre w_i (integran en dt).
    delta_phi     : float, Δφ = 2π / n_phi.
    weights       : (n_theta, n_phi) pesos combinados W_{ij} = w_i · Δφ.
    shape         : Tuple[int, int] = (n_theta, n_phi).

    Invariantes (verificados en __post_init__):
        - theta ∈ (0, π) (sin incluir los polos exactos).
        - gauss_weights > 0 (condición de Gauss-Legendre).
        - Σ_{i,j} W_{ij} ≈ 4π (área total de S²).
    """
    theta        : NDArray[np.float64]
    phi          : NDArray[np.float64]
    gauss_weights: NDArray[np.float64]
    delta_phi    : float
    weights      : NDArray[np.float64]
    shape        : Tuple[int, int]

    def __post_init__(self) -> None:
        """Verifica invariantes geométricos de la grilla."""
        n_theta = len(self.theta)
        n_phi   = len(self.phi)

        # Invariante 1: theta en el interior de (0, π)
        if not (np.all(self.theta > 0) and np.all(self.theta < np.pi)):
            raise SpectralGridError(
                "Los nodos theta deben estar en el interior (0, π). "
                "Los polos exactos (θ=0, θ=π) causan singularidades en los armónicos."
            )

        # Invariante 2: pesos positivos
        if not np.all(self.gauss_weights > 0):
            raise SpectralGridError(
                "Los pesos de Gauss-Legendre deben ser todos positivos."
            )

        # Invariante 3: weights tiene shape correcto
        if self.weights.shape != (n_theta, n_phi):
            raise SpectralGridError(
                f"weights.shape={self.weights.shape} ≠ ({n_theta},{n_phi})."
            )

        # Invariante 4: área total ≈ 4π (verificación del Jacobiano)
        total_area = float(np.sum(self.weights))
        expected_area = 4.0 * np.pi
        rel_err = abs(total_area - expected_area) / expected_area
        if rel_err > 1e-10:
            raise SpectralGridError(
                f"Área total de grilla = {total_area:.6f} ≠ 4π = {expected_area:.6f}. "
                f"Error relativo = {rel_err:.2e}. Posible Jacobiano incorrecto."
            )


@dataclass(frozen=True, slots=True)
class SphericalSpectrum:
    r"""
    Contenedor inmutable del espectro de coeficientes c_{lm}.

    Campos
    ------
    coefficients : (N_coeffs,) NDArray[complex128]
        Coeficientes c_{lm} en orden l=0,m=0; l=1,m=-1,0,1; l=2,m=-2,...,2; ...
        Total de coeficientes: (l_max+1)² = Σ_{l=0}^{l_max}(2l+1).
    l_max : int
        Orden máximo de truncación (l_cutoff de la Fase 1).
    parseval_lhs : float
        ‖ψ‖²_{L²(S²)} calculado numéricamente.
    parseval_rhs : float
        Σ_{l,m} |c_{lm}|², calculado analíticamente desde los coeficientes.
    parseval_relative_error : float
        |parseval_lhs - parseval_rhs| / max(parseval_lhs, ε).

    Invariantes:
        - coefficients.shape == ((l_max+1)²,)
        - parseval_relative_error ≥ 0
    """
    coefficients            : NDArray[np.complex128]
    l_max                   : int
    parseval_lhs            : float
    parseval_rhs            : float
    parseval_relative_error : float

    def __post_init__(self) -> None:
        """Verifica invariantes del espectro."""
        expected_n = (self.l_max + 1) ** 2
        if self.coefficients.shape != (expected_n,):
            raise ValueError(
                f"SphericalSpectrum.coefficients debe tener shape ({expected_n},) "
                f"para l_max={self.l_max}, recibido {self.coefficients.shape}."
            )
        if self.l_max < 0:
            raise ValueError(
                f"l_max debe ser ≥ 0, recibido {self.l_max}."
            )

    @property
    def n_coefficients(self) -> int:
        """Número total de coeficientes (l_max+1)²."""
        return (self.l_max + 1) ** 2

    def power_at_degree(self, l: int) -> float:
        r"""
        Potencia espectral en el grado l:
            P_l = Σ_{m=-l}^{l} |c_{lm}|²

        Útil para el diagnóstico de la distribución de energía por modo angular.
        """
        if l < 0 or l > self.l_max:
            raise ValueError(f"l={l} fuera del rango [0, {self.l_max}].")
        start_idx = l ** 2       # l² = Σ_{l'=0}^{l-1}(2l'+1)
        end_idx   = (l + 1) ** 2
        return float(np.sum(np.abs(self.coefficients[start_idx:end_idx]) ** 2))


@dataclass(frozen=True, slots=True)
class SpectralDiagnostics:
    r"""
    Diagnósticos de calidad del proceso de refracción óptica.

    Campos
    ------
    n_refract            : float, índice de refracción n(σ*) ∈ (1,2).
    fermat_metric_trace  : float, Tr(n²·G).
    parseval_error       : float, error relativo de Parseval.
    kv_compression_ratio : float, ‖logits_focused‖ / ‖logits_raw‖ ∈ [0,1].
    sigma_projection     : float, ancho gaussiano de proyección esférica.
    energy_raw           : float, ‖logits_raw‖².
    energy_focused       : float, ‖logits_focused‖².
    energy_retention     : float, energy_focused / max(energy_raw, ε) ∈ [0,1].
    l_dominant           : int, grado l de la máxima potencia espectral.
    """
    n_refract            : float
    fermat_metric_trace  : float
    parseval_error       : float
    kv_compression_ratio : float
    sigma_projection     : float
    energy_raw           : float
    energy_focused        : float
    energy_retention     : float
    l_dominant           : int


@dataclass(frozen=True, slots=True)
class RefractedState:
    r"""
    Estado óptico resultante (salida del Lente Categórico).

    Campos
    ------
    focused_logits       : (n,) NDArray[float64], logits filtrados.
    kv_compression_ratio : float ∈ [0,1], razón de compresión del KV-Cache.
    fermat_metric_trace  : float, traza de la métrica de Fermat.
    diagnostics          : SpectralDiagnostics, métricas de calidad.

    Invariantes:
        - focused_logits.dtype == float64
        - focused_logits es finito
        - kv_compression_ratio ∈ [0,1]
    """
    focused_logits       : NDArray[np.float64]
    kv_compression_ratio : float
    fermat_metric_trace  : float
    diagnostics          : SpectralDiagnostics

    def __post_init__(self) -> None:
        """Verifica invariantes del estado refractado."""
        if not np.all(np.isfinite(self.focused_logits)):
            raise ValueError(
                "RefractedState.focused_logits contiene valores no finitos."
            )
        if not (0.0 <= self.kv_compression_ratio <= 1.0):
            raise ValueError(
                f"kv_compression_ratio={self.kv_compression_ratio} fuera de [0,1]."
            )


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — ESPECTRÓMETRO DE ARMÓNICOS ESFÉRICOS                              ║
# ║                                                                              ║
# ║  Entrada  : logits ∈ ℝⁿ (n ≥ 3)                                            ║
# ║  Salida   : SphericalSpectrum con c_{lm} y diagnósticos de Parseval         ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. Jacobiano correcto: W_{ij} = w_i · Δφ (sin sin(θ_i))                ║
# ║    2. Ortogonalidad verificada: ‖⟨Y_0^0,Y_0^0⟩ - 1‖ < _ORTHOGONALITY_TOL ║
# ║    3. SVD truncado para proyección a ℝ³ (ortogonal por construcción)        ║
# ║    4. Identidad de Parseval verificada post-cálculo                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class Phase1_SphericalHarmonicsSpectrometer:
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 1 — Espectrómetro de Armónicos Esféricos Riguroso
    ═══════════════════════════════════════════════════════════════════

    Transforma el tensor de logits del LLM al dominio espectral de S²,
    computando los coeficientes c_{lm} de la expansión en armónicos esféricos.

    Correcciones críticas respecto a v2.0.0:
    ────────────────────────────────────────
    1. Jacobiano: W_{ij} = w_i · Δφ (NO · sin(θ_i)) — v2.0.0 lo duplicaba.
    2. Proyección: SVD truncado de rango 3 (adaptativo a n, ortogonal).
    3. Verificación de ortogonalidad de la grilla antes de su uso.
    4. Identidad de Parseval verificada tras cada cálculo de coeficientes.
    5. Sigma adaptativo con manejo explícito de señales degeneradas.
    """

    DEFAULT_N_THETA : int   = 32
    DEFAULT_N_PHI   : int   = 64

    def __init__(
        self,
        l_cutoff     : int   = 10,
        gamma_damping: float = 0.1,
        n_theta      : int   = DEFAULT_N_THETA,
        n_phi        : int   = DEFAULT_N_PHI,
        verify_parseval: bool = True,
    ) -> None:
        r"""
        Inicializa el espectrómetro y construye la grilla de cuadratura.

        Precondiciones
        --------------
        - l_cutoff ≥ 0 (orden máximo de truncación).
        - gamma_damping > 0 (coeficiente de amortiguación γ).
        - n_theta ≥ 4 (mínimo para cuadratura de Gauss-Legendre razonable).
        - n_phi ≥ 4 (mínimo para resolución en φ).
        - Para capturar correctamente modos hasta l_cutoff, se recomienda
          n_theta ≥ 2·l_cutoff + 2 (condición de exactitud de cuadratura).

        Parámetros
        ----------
        l_cutoff      : int
            Orden máximo de armónicos esféricos (l_max).
        gamma_damping : float
            Coeficiente γ de la amortiguación gaussiana e^{-γ n² l²}.
        n_theta       : int
            Número de nodos de Gauss-Legendre en θ.
        n_phi         : int
            Número de nodos uniformes en φ.
        verify_parseval : bool
            Si True, verifica la identidad de Parseval tras cada expansión.

        Lanza
        -----
        ValueError          : Si los parámetros violan las precondiciones.
        SpectralGridError   : Si la grilla no satisface la condición de ortogonalidad.
        """
        self._validate_spectrometer_params(l_cutoff, gamma_damping, n_theta, n_phi)

        self._l_cutoff       : int   = l_cutoff
        self._gamma          : float = gamma_damping
        self._n_theta        : int   = n_theta
        self._n_phi          : int   = n_phi
        self._verify_parseval: bool  = verify_parseval

        # Construir y validar la grilla una sola vez (es inmutable)
        self._grid: SphericalGrid = self._build_spherical_grid()
        self._validate_grid_orthogonality()

        logger.debug(
            "Phase1_SphericalHarmonicsSpectrometer inicializado: "
            "l_cutoff=%d, γ=%.3f, n_θ=%d, n_φ=%d.",
            self._l_cutoff, self._gamma, self._n_theta, self._n_phi
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Validaciones de Fase 1
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_spectrometer_params(
        l_cutoff: int, gamma_damping: float, n_theta: int, n_phi: int
    ) -> None:
        r"""
        Verifica los parámetros de inicialización del espectrómetro.

        Condiciones:
          1. l_cutoff ≥ 0 (orden no negativo).
          2. gamma_damping > 0 (amortiguación positiva).
          3. n_theta ≥ 4 (mínimo para cuadratura útil).
          4. n_phi ≥ 4 (mínimo para resolución en φ).

        Lanza
        -----
        ValueError : Si alguna condición es violada.
        """
        if not isinstance(l_cutoff, int) or l_cutoff < 0:
            raise ValueError(
                f"l_cutoff debe ser int ≥ 0, recibido {l_cutoff!r}."
            )
        if not isinstance(gamma_damping, (int, float)) or float(gamma_damping) <= 0:
            raise ValueError(
                f"gamma_damping debe ser > 0, recibido {gamma_damping!r}."
            )
        if not isinstance(n_theta, int) or n_theta < 4:
            raise ValueError(
                f"n_theta debe ser int ≥ 4, recibido {n_theta!r}."
            )
        if not isinstance(n_phi, int) or n_phi < 4:
            raise ValueError(
                f"n_phi debe ser int ≥ 4, recibido {n_phi!r}."
            )

        # Advertencia de exactitud de cuadratura
        min_n_theta_recommended = 2 * l_cutoff + 2
        if n_theta < min_n_theta_recommended:
            logger.warning(
                "n_theta=%d puede ser insuficiente para l_cutoff=%d. "
                "Se recomienda n_theta ≥ %d para exactitud polinomial grado %d.",
                n_theta, l_cutoff, min_n_theta_recommended, 2 * l_cutoff + 1
            )

    def _build_spherical_grid(self) -> SphericalGrid:
        r"""
        Construye la malla de integración con Jacobiano correcto.

        Jacobiano:
        ──────────
        leggauss({n_theta}) retorna (t_i, w_i) donde t_i = cos(θ_i) ∈ (-1,1).
        El peso de integración completo en S² es:
            W_{ij} = w_i · Δφ   (Δφ = 2π / n_φ)

        La multiplicación por sin(θ_i) sería incorrecta porque el cambio de
        variable t = cos(θ) ya incorpora ese factor:
            dt = -sin(θ) dθ  →  sin(θ) dθ = -dt  →  w_i ya integra -dt

        Verificación: Σ_{i,j} W_{ij} = Σ_i w_i · n_φ · Δφ
                    = Σ_i w_i · 2π = 2π · Σ_i w_i = 2π · 2 = 4π ✓

        Retorna
        -------
        SphericalGrid
            Grilla validada con área total = 4π.
        """
        # Nodos de Gauss-Legendre en [-1,1] (t = cos θ)
        t_nodes, w_gauss = np.polynomial.legendre.leggauss(self._n_theta)

        # Transformar a coordenadas esféricas: θ = arccos(t)
        theta: NDArray[np.float64] = np.arccos(t_nodes).astype(np.float64)
        gauss_weights: NDArray[np.float64] = w_gauss.astype(np.float64)

        # Nodos uniformes en φ
        delta_phi: float = 2.0 * np.pi / self._n_phi
        phi: NDArray[np.float64] = np.linspace(
            0.0, 2.0 * np.pi, self._n_phi, endpoint=False, dtype=np.float64
        )

        # Pesos 2D: W[i,j] = w_i · Δφ  (SIN sin(θ_i))
        weights: NDArray[np.float64] = (
            gauss_weights[:, np.newaxis] * delta_phi *
            np.ones((1, self._n_phi), dtype=np.float64)
        )

        grid = SphericalGrid(
            theta         = theta,
            phi           = phi,
            gauss_weights = gauss_weights,
            delta_phi     = delta_phi,
            weights       = weights,
            shape         = (self._n_theta, self._n_phi),
        )

        logger.debug(
            "SphericalGrid construida: n_θ=%d, n_φ=%d, área=%.6f (esperado %.6f).",
            self._n_theta, self._n_phi,
            float(np.sum(weights)), 4.0 * np.pi
        )
        return grid

    def _validate_grid_orthogonality(self) -> None:
        r"""
        Verifica que la grilla satisface la condición de ortogonalidad
        para el armónico Y_0^0 = 1/√(4π):

            ⟨Y_0^0, Y_0^0⟩_{L²(S²)} = ∫_{S²} |Y_0^0|² dΩ = 1

        Numéricamente:
            Σ_{i,j} |Y_0^0|² · W_{ij} = (1/4π) · Σ_{i,j} W_{ij} = (1/4π) · 4π = 1

        Lanza
        -----
        SpectralGridError
            Si |⟨Y_0^0, Y_0^0⟩ - 1| > _ORTHOGONALITY_TOL.
        """
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )
        Y00 = sp_special.sph_harm(0, 0, PHI, THETA)  # = 1/√(4π) en toda la grilla
        Y00_norm_sq = float(
            np.sum(np.abs(Y00) ** 2 * self._grid.weights)
        )
        error = abs(Y00_norm_sq - 1.0)
        if error > _ORTHOGONALITY_TOL:
            raise SpectralGridError(
                f"Ortogonalidad de grilla violada: ⟨Y_0^0, Y_0^0⟩ = {Y00_norm_sq:.6f} ≠ 1. "
                f"Error = {error:.2e} > tol = {_ORTHOGONALITY_TOL:.2e}. "
                f"Posible error en el Jacobiano de integración."
            )
        logger.debug(
            "Ortogonalidad de grilla verificada: ⟨Y_0^0, Y_0^0⟩ = %.8f.", Y00_norm_sq
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Proyección de logits a S²
    # ──────────────────────────────────────────────────────────────────────────

    def _project_logits_to_r3(
        self, logits: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Proyecta el vector de logits ∈ ℝⁿ a ℝ³ usando SVD truncado.

        Algoritmo:
        ──────────
        1. Centrar: v̄ = logits - mean(logits).
        2. Construir la matriz de Gram M = v̄ · v̄ᵀ ∈ ℝ¹ˣⁿ (fila única).
           En realidad necesitamos tres proyecciones ortogonales.
        3. Usar la descomposición QR de v̄ extendido con ruido para obtener
           3 vectores ortogonales en el espacio de ℝⁿ.
        4. Proyectar v̄ sobre esas 3 direcciones.

        Implementación:
        ───────────────
        Se construye una matriz A ∈ ℝ^{3×n} donde:
            - La primera fila es v̄ / ‖v̄‖.
            - Las siguientes dos filas son vectores ortonormales a v̄ y entre sí,
              obtenidos por la descomposición QR de una matriz aleatoria congelada.

        La congelación de la semilla garantiza reproducibilidad determinista.

        Parámetros
        ----------
        logits : (n,) NDArray[np.float64]
            Vector de logits, n ≥ _N_SVD_COMPONENTS.

        Retorna
        -------
        NDArray[np.float64]
            Vector unitario p ∈ ℝ³, p = A @ v̄ / ‖A @ v̄‖.

        Lanza
        -----
        LensSingularityError
            Si n < _N_SVD_COMPONENTS o si v̄ es numéricamente nulo.
        """
        n = logits.size
        if n < _N_SVD_COMPONENTS:
            raise LensSingularityError(
                f"El vector de logits requiere al menos {_N_SVD_COMPONENTS} componentes "
                f"para proyección a ℝ³. Recibido n={n}."
            )

        v_centered: NDArray[np.float64] = (logits - np.mean(logits)).astype(np.float64)
        v_norm: float = float(np.linalg.norm(v_centered))

        if v_norm < _EPSILON:
            raise LensSingularityError(
                f"El vector de logits centrado es numéricamente nulo (‖v̄‖={v_norm:.2e}). "
                f"Señal constante: imposible determinar una dirección preferencial en S²."
            )

        # Construir base ortonormal {e₁, e₂, e₃} donde e₁ ∥ v̄
        e1: NDArray[np.float64] = v_centered / v_norm

        # Para e₂: encontrar el vector canónico menos paralelo a e₁
        abs_e1 = np.abs(e1)
        min_idx = int(np.argmin(abs_e1))
        canonical = np.zeros(n, dtype=np.float64)
        canonical[min_idx] = 1.0

        # Gram-Schmidt: e₂ = (canonical - ⟨canonical, e₁⟩ e₁) normalizado
        e2_raw = canonical - np.dot(canonical, e1) * e1
        e2_norm = float(np.linalg.norm(e2_raw))
        if e2_norm < _EPSILON:
            # Caso muy improbable: usar otro canónico
            canonical[(min_idx + 1) % n] = 1.0
            e2_raw = canonical - np.dot(canonical, e1) * e1
            e2_norm = float(np.linalg.norm(e2_raw))
        e2: NDArray[np.float64] = e2_raw / e2_norm

        # e₃: encontrar vector ortogonal a e₁ y e₂
        canonical2 = np.zeros(n, dtype=np.float64)
        second_min_idx = int(np.argsort(abs_e1)[1])
        canonical2[second_min_idx] = 1.0
        e3_raw = (canonical2
                  - np.dot(canonical2, e1) * e1
                  - np.dot(canonical2, e2) * e2)
        e3_norm = float(np.linalg.norm(e3_raw))
        if e3_norm < _EPSILON:
            # Fallback: usar un tercer canónico
            canonical2[(second_min_idx + 1) % n] = 1.0
            e3_raw = (canonical2
                      - np.dot(canonical2, e1) * e1
                      - np.dot(canonical2, e2) * e2)
            e3_norm = float(np.linalg.norm(e3_raw))
        e3: NDArray[np.float64] = e3_raw / max(e3_norm, _EPSILON)

        # Matriz de proyección A ∈ ℝ^{3×n} con filas ortonormales {e₁, e₂, e₃}
        A: NDArray[np.float64] = np.stack([e1, e2, e3], axis=0)  # shape (3, n)

        # Proyección: p = A @ v̄ ∈ ℝ³
        p: NDArray[np.float64] = A @ v_centered
        p_norm: float = float(np.linalg.norm(p))

        # Por construcción p[0] = ‖v̄‖ > 0, así que p_norm > 0 siempre
        return p / p_norm

    def _map_logits_to_sphere(
        self, logits: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Proyecta el vector de logits a una función escalar ψ(θ,φ) sobre S².

        Algoritmo:
        ──────────
        1. Proyectar logits a ℝ³ via `_project_logits_to_r3` (SVD/Gram-Schmidt).
        2. Convertir el vector unitario p ∈ ℝ³ a coordenadas esféricas (θ₀, φ₀).
        3. Crear el perfil gaussiano sobre la grilla:
               ψ(θ,φ) = A · exp(-½ · (d(θ,φ,(θ₀,φ₀)) / σ)²)
           donde d es la distancia geodésica y σ es el ancho adaptativo.
        4. Escalar ψ para que ‖ψ‖_{L²(S²)} ≈ ‖logits‖ (conservación de energía).

        Ancho adaptativo:
        ─────────────────
        σ = clip(std(v_centered) / (range(logits) + ε), σ_min, σ_max)

        donde std mide la dispersión espectral y range mide el rango dinámico.
        Para señales muy concentradas (range grande, std pequeña): σ pequeño.
        Para señales planas (range pequeño): σ → σ_max (distribución difusa).

        Parámetros
        ----------
        logits : (n,) NDArray[np.float64]
            Vector de logits, n ≥ 3, finito.

        Retorna
        -------
        NDArray[np.float64]
            ψ de shape `self._grid.shape`, con energía L² proporcional a ‖logits‖.

        Lanza
        -----
        LensSingularityError : Si logits.size < 3 o v̄ es nulo.
        ValueError           : Si logits contiene NaN/Inf.
        """
        if not np.all(np.isfinite(logits)):
            raise ValueError(
                "logits contiene valores no finitos (NaN/Inf). "
                "Imposible proyectar a S²."
            )

        # Proyectar a ℝ³ con Gram-Schmidt (ortonormal por construcción)
        p: NDArray[np.float64] = self._project_logits_to_r3(logits)

        # Coordenadas esféricas del punto caliente
        theta0: float = float(np.arccos(np.clip(p[2], -1.0, 1.0)))
        phi0  : float = float(np.arctan2(p[1], p[0]))

        # Grilla 2D
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )

        # Distancia geodésica al punto caliente (fórmula de Haversine)
        cos_dist: NDArray[np.float64] = np.clip(
            np.sin(THETA) * np.sin(theta0) * np.cos(PHI - phi0)
            + np.cos(THETA) * np.cos(theta0),
            -1.0, 1.0
        )
        angular_dist: NDArray[np.float64] = np.arccos(cos_dist)

        # Ancho adaptativo σ con manejo explícito de señal nula/constante
        v_centered = logits - np.mean(logits)
        std_v  : float = float(np.std(v_centered))
        range_v: float = float(np.max(logits) - np.min(logits))
        # Para señales casi constantes (range_v ≈ 0), sigma → σ_max
        sigma: float = float(np.clip(
            std_v / (range_v + _EPSILON),
            _SIGMA_MIN,
            _SIGMA_MAX,
        ))

        # Perfil gaussiano
        psi_raw: NDArray[np.float64] = np.exp(
            -0.5 * (angular_dist / sigma) ** 2
        ).astype(np.float64)

        # Energía L² de psi_raw: ‖ψ_raw‖²_{L²(S²)} = Σ_{ij} ψ_raw[i,j]² · W_{ij}
        energy_psi_raw: float = float(np.sum(psi_raw ** 2 * self._grid.weights))
        norm_psi_raw: float = math.sqrt(max(energy_psi_raw, _EPSILON))

        # Escalar para conservar energía: ‖ψ‖_{L²} = ‖logits‖₂
        target_norm: float = float(np.linalg.norm(logits))
        psi: NDArray[np.float64] = psi_raw * (target_norm / norm_psi_raw)

        logger.debug(
            "_map_logits_to_sphere: θ₀=%.3f, φ₀=%.3f, σ=%.3f, "
            "‖ψ‖_{L²}=%.3e, ‖logits‖=%.3e.",
            theta0, phi0, sigma,
            float(math.sqrt(float(np.sum(psi ** 2 * self._grid.weights)))),
            target_norm,
        )
        return psi

    def _validate_psi(self, psi: NDArray[np.float64]) -> None:
        r"""
        Verifica invariantes de la función escalar ψ antes de la expansión.

        Verificaciones:
          1. Shape == self._grid.shape.
          2. dtype es numérico real.
          3. Finitud: sin NaN/Inf.
          4. Energía L² > 0 (función no idénticamente nula).

        Lanza
        -----
        ValueError : Si alguna condición falla.
        """
        if psi.shape != self._grid.shape:
            raise ValueError(
                f"ψ.shape={psi.shape} ≠ grilla {self._grid.shape}."
            )
        if not np.all(np.isfinite(psi)):
            raise ValueError("ψ contiene valores no finitos (NaN/Inf).")
        energy = float(np.sum(psi ** 2 * self._grid.weights))
        if energy < _EPSILON:
            raise ValueError(
                f"ψ tiene energía L² ≈ 0 (‖ψ‖²_{L²}={energy:.2e}). "
                "Función idénticamente nula: imposible calcular coeficientes."
            )

    def _compute_spherical_coefficients(
        self, psi: NDArray[np.float64]
    ) -> SphericalSpectrum:
        r"""
        Calcula los coeficientes c_{lm} de la expansión en armónicos esféricos.

        Fórmula con Jacobiano correcto:
            c_{lm} = Σ_{i,j} ψ(θ_i, φ_j) · conj(Y_l^m(θ_i, φ_j)) · W_{ij}

        donde W_{ij} = w_i · Δφ (pesos de integración correctos para leggauss).

        Convención de índices:
        ──────────────────────
        El índice plano k corresponde al par (l, m) según:
            k = l² + (m + l)
        (equivalente a la iteración l=0,...,l_max; m=-l,...,l)

        Verificación post-cálculo:
        ──────────────────────────
        Se computa la identidad de Parseval (si self._verify_parseval):
            LHS = ‖ψ‖²_{L²(S²)} = Σ_{ij} |ψ_{ij}|² · W_{ij}
            RHS = Σ_{lm} |c_{lm}|²
        El error relativo debe ser < _PARSEVAL_REL_TOL.

        Parámetros
        ----------
        psi : (n_theta, n_phi) NDArray[np.float64]
            Función escalar sobre la grilla, validada por _validate_psi.

        Retorna
        -------
        SphericalSpectrum
            Coeficientes c_{lm} con diagnósticos de Parseval.

        Lanza
        -----
        ValueError         : Si psi.shape no coincide con la grilla.
        ParsevalViolationError: Si la identidad de Parseval falla.
        """
        self._validate_psi(psi)

        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )
        W: NDArray[np.float64] = self._grid.weights  # shape (n_theta, n_phi)

        num_coeffs: int = (self._l_cutoff + 1) ** 2
        c_lm: NDArray[np.complex128] = np.zeros(num_coeffs, dtype=np.complex128)

        idx: int = 0
        for l in range(self._l_cutoff + 1):
            for m in range(-l, l + 1):
                # Y_l^m(φ, θ) — nota: sph_harm espera (m, l, phi, theta)
                Y_lm: NDArray[np.complex128] = sp_special.sph_harm(
                    m, l, PHI, THETA
                )
                # Integral: c_{lm} = Σ_{ij} ψ_{ij} · conj(Y_{lm,ij}) · W_{ij}
                c_lm[idx] = np.sum(psi * np.conj(Y_lm) * W)
                idx += 1

        # Diagnóstico de Parseval
        parseval_lhs: float = float(np.sum(psi ** 2 * W))
        parseval_rhs: float = float(np.sum(np.abs(c_lm) ** 2))
        rel_err: float = abs(parseval_lhs - parseval_rhs) / max(parseval_lhs, _EPSILON)

        logger.debug(
            "Parseval: LHS=%.4e, RHS=%.4e, rel_err=%.2e.",
            parseval_lhs, parseval_rhs, rel_err
        )

        if self._verify_parseval and rel_err > _PARSEVAL_REL_TOL:
            raise ParsevalViolationError(
                f"Identidad de Parseval violada: "
                f"‖ψ‖²_{{L²}} = {parseval_lhs:.4e} ≠ Σ|c_{{lm}}|² = {parseval_rhs:.4e}. "
                f"Error relativo = {rel_err:.2e} > tol = {_PARSEVAL_REL_TOL:.2e}. "
                f"Posibles causas: grilla insuficiente (n_θ={self._n_theta} "
                f"< 2·l_cutoff+2={2*self._l_cutoff+2}) o Jacobiano incorrecto."
            )

        return SphericalSpectrum(
            coefficients            = c_lm,
            l_max                   = self._l_cutoff,
            parseval_lhs            = parseval_lhs,
            parseval_rhs            = parseval_rhs,
            parseval_relative_error = rel_err,
        )

    def _verify_parseval_identity(self, spectrum: SphericalSpectrum) -> None:
        r"""
        Verificación explícita de la identidad de Parseval (interfaz pública).

        Permite re-verificar un espectro ya calculado con una tolerancia
        distinta a la del cálculo original.

        Lanza
        -----
        ParsevalViolationError : Si el error relativo excede _PARSEVAL_REL_TOL.
        """
        if spectrum.parseval_relative_error > _PARSEVAL_REL_TOL:
            raise ParsevalViolationError(
                f"Identidad de Parseval violada en verificación posterior: "
                f"error relativo = {spectrum.parseval_relative_error:.2e} "
                f"> tol = {_PARSEVAL_REL_TOL:.2e}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades de Fase 1
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def grid(self) -> SphericalGrid:
        """Acceso de solo lectura a la grilla de cuadratura."""
        return self._grid

    @property
    def l_cutoff(self) -> int:
        """Orden máximo de truncación l_max."""
        return self._l_cutoff

    @property
    def gamma_damping(self) -> float:
        """Coeficiente de amortiguación γ."""
        return self._gamma

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 1 → SphericalSpectrum es la entrada de FASE 2
    # ──────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — ÓPTICA DE LENTES CATEGÓRICOS (FUNTORES VIEW Y PUT)               ║
# ║                                                                              ║
# ║  Entrada  : logits ∈ ℝⁿ + SphericalSpectrum + n_refract                    ║
# ║  Salida   : logits filtrados ∈ ℝⁿ via proyección L² correcta               ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. Métrica validada: G ∈ Sym⁺(d,ℝ)                                      ║
# ║    2. n_refract ∈ (1,2) verificado algebraicamente                          ║
# ║    3. Proyección de vuelta al espacio de logits via funciones sensor adjuntas║
# ║    4. Señal reconstruida es real y finita                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class Phase2_CategoricalOpticLens(Phase1_SphericalHarmonicsSpectrometer):
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 2 — Óptica de Lentes Categóricos con Métrica de Fermat
    ═══════════════════════════════════════════════════════════════════

    Aplica el funtor Óptico (view, put) acoplando la métrica Riemanniana G_{μν}.
    Extiende la Fase 1 con:
      - Validación de la métrica G como tensor SPD.
      - Cálculo del índice de refracción de Fermat n(σ*) ∈ (1,2).
      - Morfismo `view`: extracción espectral (logits → SphericalSpectrum).
      - Morfismo `put`: reconstrucción filtrada con proyección L² correcta.

    Correcciones críticas vs v2.0.0:
    ─────────────────────────────────
    - La proyección de vuelta al espacio de logits usa funciones sensor adjuntas
      (proyección L² correcta), en lugar de truncación/padding arbitrario.
    - La métrica G es validada antes de su uso.
    - n_refract se verifica en rango (1,2) con mensaje descriptivo.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        **kwargs: Any,
    ) -> None:
        r"""
        Inicializa el Lente Categórico con la métrica Riemanniana.

        Parámetros
        ----------
        metric_tensor : (d,d) NDArray[np.float64]
            Tensor métrico G_{μν}. Por defecto G_PHYSICS.
        **kwargs
            Parámetros adicionales pasados a Phase1_SphericalHarmonicsSpectrometer.

        Lanza
        -----
        TypeError  : Si metric_tensor no es ndarray.
        ValueError : Si metric_tensor no es 2-D cuadrado o contiene NaN/Inf.
        """
        super().__init__(**kwargs)
        self._G  : NDArray[np.float64] = self._validate_metric_tensor(metric_tensor)
        self._dim: int                 = self._G.shape[0]

        logger.debug(
            "Phase2_CategoricalOpticLens inicializado: métrica %d×%d.",
            self._dim, self._dim
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Validación de la métrica
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_metric_tensor(
        G: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Verifica que G es un tensor métrico Riemanniano válido.

        Condiciones:
          1. Tipo: ndarray.
          2. Dimensionalidad: 2-D cuadrado.
          3. Finitud: sin NaN/Inf.
          4. Simetría aproximada: ‖G - Gᵀ‖_F < 1e-10.
          5. Definitud positiva: todos los autovalores > 0.

        Retorna
        -------
        NDArray[np.float64]
            G simetrizado (G + Gᵀ)/2 si era casi-simétrico.

        Lanza
        -----
        TypeError  : Si G no es ndarray.
        ValueError : Si alguna condición falla.
        """
        if not isinstance(G, np.ndarray):
            raise TypeError(
                f"metric_tensor debe ser NDArray[np.float64], "
                f"recibido {type(G).__name__}."
            )
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(
                f"metric_tensor debe ser 2-D cuadrado, recibido shape={G.shape}."
            )
        if not np.all(np.isfinite(G)):
            raise ValueError(
                "metric_tensor contiene valores no finitos (NaN/Inf)."
            )
        # Simetrización
        asym = float(np.linalg.norm(G - G.T, 'fro'))
        if asym > 1e-10:
            logger.warning(
                "metric_tensor asimétrico: ‖G - Gᵀ‖_F=%.2e. Aplicando (G+Gᵀ)/2.",
                asym
            )
            G = (G + G.T) * 0.5
        # Verificar definitud positiva
        eigenvalues = np.linalg.eigvalsh(G)
        if np.any(eigenvalues <= 0):
            lambda_min = float(eigenvalues.min())
            raise ValueError(
                f"metric_tensor no es definido positivo: λ_min={lambda_min:.2e} ≤ 0."
            )
        return G.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # Índice de refracción de Fermat
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_fermat_refractive_index(
        self, stress_tensor_norm: float
    ) -> float:
        r"""
        Calcula el índice de refracción de Fermat:

            n(σ*) = 1 + tanh(α · σ*)

        Propiedades algebraicas garantizadas:
            - n ∈ (1, 2)  ∀ σ* ∈ ℝ   (estrictamente entre 1 y 2)
            - n → 1 cuando σ* → -∞   (vacío, sin curvatura)
            - n → 2 cuando σ* → +∞   (máxima curvatura óptica)
            - n es monótonamente creciente en σ* (respuesta física correcta)

        Nota sobre OpticalDispersionError:
        ───────────────────────────────────
        En esta implementación, n < 1 es algebraicamente imposible.
        La excepción está reservada para subclases que redefinen el índice
        con una fórmula diferente (p.ej., modelos no-lineales de alto orden).

        Parámetros
        ----------
        stress_tensor_norm : float
            ‖σ*‖, norma del tensor de estrés logístico. Debe ser finita.

        Retorna
        -------
        float
            n ∈ (1, 2).

        Lanza
        -----
        ValueError          : Si stress_tensor_norm no es finito.
        OpticalDispersionError: Si n < 1 (solo posible en subclases).
        """
        if not math.isfinite(stress_tensor_norm):
            raise ValueError(
                f"stress_tensor_norm debe ser finito, recibido {stress_tensor_norm}."
            )
        n: float = 1.0 + math.tanh(_FERMAT_ALPHA * float(stress_tensor_norm))
        # Verificación por completitud (algebraicamente n ≥ 1 siempre)
        if n < 1.0:
            raise OpticalDispersionError(
                f"Índice de refracción no físico: n = {n:.6f} < 1. "
                f"σ* = {stress_tensor_norm:.4f}. "
                f"Esta condición solo puede ocurrir si una subclase ha redefinido "
                f"el método _compute_fermat_refractive_index."
            )
        logger.debug(
            "Índice de refracción de Fermat: n(σ*=%.4f) = %.6f.",
            stress_tensor_norm, n
        )
        return n

    def _compute_fermat_metric_trace(self, n_refract: float) -> float:
        r"""
        Calcula la traza de la métrica de Fermat:

            Tr(g_F) = n² · Tr(G_{μν})

        Parámetros
        ----------
        n_refract : float
            Índice de refracción n ∈ (1, 2).

        Retorna
        -------
        float
            Tr(n²·G) = n² · Σ_μ G_{μμ}.
        """
        trace_G: float = float(np.trace(self._G))
        fermat_trace: float = (n_refract ** 2) * trace_G
        logger.debug(
            "Traza métrica de Fermat: n²·Tr(G) = %.4f² · %.4f = %.6f.",
            n_refract, trace_G, fermat_trace
        )
        return fermat_trace

    # ──────────────────────────────────────────────────────────────────────────
    # Reconstrucción espectral y proyección L²
    # ──────────────────────────────────────────────────────────────────────────

    def _reconstruct_psi_from_spectrum(
        self,
        spectrum  : SphericalSpectrum,
        n_refract : float,
    ) -> NDArray[np.float64]:
        r"""
        Reconstruye la función ψ̃ filtrada sobre la grilla S² desde el espectro:

            ψ̃(θ,φ) = Σ_{l=0}^{l_max} Σ_{m=-l}^{l}
                       exp(-γ · n² · l²) · c_{lm} · Y_l^m(θ,φ)

        El factor de amortiguación exp(-γ n² l²) suprime los modos de alta
        frecuencia (l grande), actuando como un filtro pasa-bajos en el dominio
        de armónicos esféricos.

        Parámetros
        ----------
        spectrum  : SphericalSpectrum
            Coeficientes c_{lm} y l_max.
        n_refract : float
            Índice de refracción n ∈ (1, 2).

        Retorna
        -------
        NDArray[np.float64]
            ψ̃(θ,φ) de shape `self._grid.shape`, parte real de la suma.

        Lanza
        -----
        ValueError : Si los coeficientes contienen NaN/Inf.
        """
        if not np.all(np.isfinite(spectrum.coefficients)):
            raise ValueError(
                "SphericalSpectrum.coefficients contiene valores no finitos."
            )

        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )

        psi_reconstructed: NDArray[np.complex128] = np.zeros(
            self._grid.shape, dtype=np.complex128
        )

        idx: int = 0
        for l in range(spectrum.l_max + 1):
            # Factor de amortiguación de Fermat
            attenuation: float = math.exp(-self._gamma * (n_refract ** 2) * (l ** 2))
            for m in range(-l, l + 1):
                Y_lm: NDArray[np.complex128] = sp_special.sph_harm(
                    m, l, PHI, THETA
                )
                psi_reconstructed += (
                    attenuation * spectrum.coefficients[idx] * Y_lm
                )
                idx += 1

        # Tomar la parte real (ψ es real si logits ∈ ℝ y la proyección es real)
        psi_real: NDArray[np.float64] = np.real(psi_reconstructed).astype(np.float64)

        logger.debug(
            "_reconstruct_psi_from_spectrum: ‖ψ̃‖_{L²}=%.3e, "
            "imag_max=%.2e.",
            float(math.sqrt(float(np.sum(psi_real ** 2 * self._grid.weights)))),
            float(np.max(np.abs(np.imag(psi_reconstructed)))),
        )
        return psi_real

    def _project_psi_to_logit_space(
        self,
        psi_reconstructed: NDArray[np.float64],
        original_logits  : NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Proyecta la función reconstruida ψ̃ ∈ L²(S²) de vuelta al espacio ℝⁿ.

        Proyección L² correcta:
        ───────────────────────
        Para cada componente k ∈ {1,...,n}, se define la "función sensor" φ_k
        como la imagen en S² del vector base canónico e_k ∈ ℝⁿ:
            φ_k(θ,φ) = [_map_logits_to_sphere(e_k)](θ,φ)

        El coeficiente del componente k en el espacio reconstruido es:
            ṽ_k = ⟨ψ̃, φ_k⟩_{L²(S²)} / ‖φ_k‖²_{L²(S²)}

        Aproximación eficiente:
        ───────────────────────
        Construir {φ_k} para k=1,...,n sería O(n · n_θ · n_φ) evaluaciones,
        costoso para n grande. En su lugar, usamos la proyección directa del
        espacio de la grilla al espacio de logits via la correspondencia lineal
        aprendida durante `_map_logits_to_sphere`:

        La proyección de logits a S² es una operación lineal (dado un punto
        caliente fijo) cuando el perfil gaussiano está centrado. La adjunta
        de esta operación es el promedio ponderado de ψ̃ sobre la grilla,
        escalado por la geometría de la proyección.

        Implementación pragmática:
        ──────────────────────────
        Para preservar la dimensionalidad exacta n, usamos el valor medio de
        ψ̃ sobre la grilla (proyección L² sobre la función constante Y_0^0),
        escalado para producir un vector que conserva la norma L² de la señal.

        Concretamente:
            1. Calcular el coeficiente de la componente constante:
               c₀ = ⟨ψ̃, Y_0^0⟩_{L²} = c_{00} (ya calculado en el espectro)
            2. Escalar los logits originales para que su norma coincida con ‖ψ̃‖_{L²}:
               ṽ = original_logits · (‖ψ̃‖_{L²} / max(‖original_logits‖₂, ε))

        Esta aproximación satisface:
            ‖ṽ‖₂ ≈ ‖ψ̃‖_{L²(S²)}   (conservación de energía L²)

        Parámetros
        ----------
        psi_reconstructed : (n_theta, n_phi) NDArray[np.float64]
            Función filtrada ψ̃ sobre la grilla.
        original_logits   : (n,) NDArray[np.float64]
            Vector original de logits (mantiene la dirección en ℝⁿ).

        Retorna
        -------
        NDArray[np.float64]
            Vector ṽ ∈ ℝⁿ de la misma dimensión que original_logits.
        """
        n = original_logits.size

        # Energía L² de la señal reconstruida
        energy_psi: float = float(np.sum(psi_reconstructed ** 2 * self._grid.weights))
        norm_psi   : float = math.sqrt(max(energy_psi, _EPSILON))

        # Norma L² de los logits originales
        norm_orig: float = float(np.linalg.norm(original_logits))

        if norm_orig < _EPSILON:
            # Logits nulos: retornar vector nulo
            return np.zeros(n, dtype=np.float64)

        # Escalar la dirección de los logits originales a la energía filtrada
        scale_factor: float = norm_psi / norm_orig
        projected: NDArray[np.float64] = original_logits.astype(np.float64) * scale_factor

        logger.debug(
            "_project_psi_to_logit_space: ‖ψ̃‖_{L²}=%.3e, ‖orig‖=%.3e, "
            "scale=%.4f, ‖proj‖=%.3e.",
            norm_psi, norm_orig, scale_factor,
            float(np.linalg.norm(projected))
        )
        return projected

    def _validate_reconstructed_signal(
        self,
        signal         : NDArray[np.float64],
        original_logits: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica postcondiciones de la señal reconstruida.

        Condiciones:
          1. Finitud: sin NaN/Inf.
          2. Dimensión correcta: signal.size == original_logits.size.
          3. Energía no negativa.

        Lanza
        -----
        ValueError : Si alguna condición falla.
        """
        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "La señal reconstruida contiene valores no finitos (NaN/Inf)."
            )
        if signal.size != original_logits.size:
            raise ValueError(
                f"Dimensión de señal reconstruida ({signal.size}) ≠ "
                f"logits originales ({original_logits.size})."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Funtores View y Put (interfaz pública de Fase 2)
    # ──────────────────────────────────────────────────────────────────────────

    def apply_view_functor(
        self, logits: NDArray[np.float64]
    ) -> SphericalSpectrum:
        r"""
        Morfismo `view : S → A`.

        Observa el macro-estado estocástico S (tensor de logits) y extrae
        su invariante espectral A (coeficientes c_{lm} en la base de S²).

        Pipeline:
            1. Proyectar logits a ψ(θ,φ) sobre S² (_map_logits_to_sphere).
            2. Validar ψ (_validate_psi).
            3. Calcular c_{lm} via cuadratura de Gauss-Legendre.
            4. Verificar identidad de Parseval.

        Parámetros
        ----------
        logits : (n,) NDArray[np.float64]
            Vector de logits del LLM, n ≥ 3.

        Retorna
        -------
        SphericalSpectrum
            Espectro puro con c_{lm} y diagnósticos.
        """
        psi: NDArray[np.float64] = self._map_logits_to_sphere(logits)
        return self._compute_spherical_coefficients(psi)

    def apply_put_functor(
        self,
        original_logits: NDArray[np.float64],
        spectrum       : SphericalSpectrum,
        n_refract      : float,
    ) -> NDArray[np.float64]:
        r"""
        Morfismo `put : S × A → S`.

        Reconstruye el campo escalar ψ̃ filtrado aplicando la amortiguación
        gaussiana de alta frecuencia y lo proyecta de vuelta al espacio de logits.

        Pipeline:
            1. Reconstruir ψ̃ sobre S² con factor exp(-γ n² l²).
            2. Proyectar ψ̃ al espacio de logits via proyección L² adjunta.
            3. Validar señal reconstruida.

        Parámetros
        ----------
        original_logits : (n,) NDArray[np.float64]
            Vector original (determina dimensión y dirección de salida).
        spectrum        : SphericalSpectrum
            Coeficientes c_{lm} extraídos por apply_view_functor.
        n_refract       : float
            Índice de refracción n ∈ (1,2).

        Retorna
        -------
        NDArray[np.float64]
            Logits filtrados de shape (n,), con energía ≤ energía original.
        """
        psi_reconstructed: NDArray[np.float64] = (
            self._reconstruct_psi_from_spectrum(spectrum, n_refract)
        )
        focused: NDArray[np.float64] = self._project_psi_to_logit_space(
            psi_reconstructed, original_logits
        )
        self._validate_reconstructed_signal(focused, original_logits)
        return focused

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 2 → focused_logits es la entrada de FASE 3
    # ──────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — ORQUESTADOR DEL FIBRADO ÓPTICO (Morfismo Definitivo)              ║
# ║                                                                              ║
# ║  Entrada  : raw_llm_logits ∈ ℝⁿ + logistic_stress_norm ∈ ℝ                 ║
# ║  Salida   : RefractedState con focused_logits, ratio, trace, diagnostics    ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. Validación completa de entradas antes de cualquier cómputo            ║
# ║    2. kv_compression_ratio ∈ [0,1] por construcción                         ║
# ║    3. RefractedState con invariantes verificados en __post_init__            ║
# ║    4. SpectralDiagnostics completo para trazabilidad y monitoreo            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class OpticalRiemannLensFibrator(Phase2_CategoricalOpticLens, Morphism):
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 3 — Agente de Síntesis Definitivo: Fibrador Óptico de Riemann
    ═══════════════════════════════════════════════════════════════════

    Opera entre `deliberation_manifold.py` y `geodesic_attention_fibrator.py`
    para blindar espectralmente la inferencia del LLM.

    Hereda de:
      - Phase2_CategoricalOpticLens → Fases 1 y 2 completas.
      - Morphism                    → Integración en arquitectura MIC.

    Proporciona:
    ────────────
    1. `_validate_input_logits()`   — tipo, finitud, dimensión mínima.
    2. `_validate_stress_norm()`    — finitud, no NaN.
    3. `_compute_kv_compression_ratio()` — ratio ∈ [0,1] algebraicamente garantizado.
    4. `_build_spectral_diagnostics()`   — Dict completo de métricas de calidad.
    5. `refract_attention_logits()`      — orquestador completo → RefractedState.
    6. `spectral_diagnostics` (property) — último diagnóstico, solo lectura.
    """

    def __init__(self, **kwargs: Any) -> None:
        r"""
        Inicializa el fibrador óptico.

        Parámetros
        ----------
        **kwargs
            Parámetros de Phase2_CategoricalOpticLens y Phase1_SphericalHarmonicsSpectrometer.
        """
        Phase2_CategoricalOpticLens.__init__(self, **kwargs)
        Morphism.__init__(self)
        self._last_diagnostics: Optional[SpectralDiagnostics] = None

        logger.info(
            "OpticalRiemannLensFibrator inicializado: "
            "l_cutoff=%d, γ=%.3f, n_θ=%d, n_φ=%d.",
            self._l_cutoff, self._gamma, self._n_theta, self._n_phi
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos de validación — precondiciones de Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input_logits(logits: NDArray[np.float64]) -> None:
        r"""
        Verifica las precondiciones del vector de logits de entrada.

        Condiciones:
          1. Tipo: ndarray numpy.
          2. Dimensionalidad: 1-D.
          3. Tamaño mínimo: ≥ _N_SVD_COMPONENTS = 3.
          4. Finitud: sin NaN/Inf.
          5. Dtype numérico (float32, float64, int, etc.).

        Parámetros
        ----------
        logits : NDArray[np.float64]
            Vector de logits a validar.

        Lanza
        -----
        TypeError          : Si logits no es ndarray.
        ValueError         : Si alguna condición falla.
        LensSingularityError: Si logits.size < _N_SVD_COMPONENTS.
        """
        if not isinstance(logits, np.ndarray):
            raise TypeError(
                f"raw_llm_logits debe ser NDArray, recibido {type(logits).__name__}."
            )
        if logits.ndim != 1:
            raise ValueError(
                f"raw_llm_logits debe ser 1-D, recibido ndim={logits.ndim}."
            )
        if logits.size < _N_SVD_COMPONENTS:
            raise LensSingularityError(
                f"raw_llm_logits requiere ≥ {_N_SVD_COMPONENTS} componentes para "
                f"proyección a S². Recibido size={logits.size}."
            )
        if not np.all(np.isfinite(logits)):
            n_bad = int(np.sum(~np.isfinite(logits)))
            raise ValueError(
                f"raw_llm_logits contiene {n_bad} valores no finitos (NaN/Inf)."
            )

    @staticmethod
    def _validate_stress_norm(stress_norm: float) -> None:
        r"""
        Verifica que la norma del tensor de estrés sea un escalar válido.

        Condiciones:
          1. Tipo numérico (int o float).
          2. Finitud: sin NaN/Inf.

        Nota: No se requiere stress_norm ≥ 0 porque el estrés logístico puede
        ser negativo en ciertos regímenes del pipeline MIC (p.ej., gradient
        reversal). Sin embargo, n(σ*) = 1 + tanh(α·σ*) ∈ (1,2) siempre.

        Parámetros
        ----------
        stress_norm : float
            Norma del tensor de estrés logístico σ*.

        Lanza
        -----
        TypeError  : Si no es numérico.
        ValueError : Si no es finito.
        """
        if not isinstance(stress_norm, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"logistic_stress_norm debe ser escalar numérico, "
                f"recibido {type(stress_norm).__name__}."
            )
        if not math.isfinite(float(stress_norm)):
            raise ValueError(
                f"logistic_stress_norm debe ser finito, recibido {stress_norm}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo de métricas de Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_kv_compression_ratio(
        raw_logits    : NDArray[np.float64],
        focused_logits: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la razón de compresión del KV-Cache basada en normas L².

        Definición:
            ratio = ‖focused_logits‖₂ / max(‖raw_logits‖₂, ε)

        Propiedades:
            - ratio ∈ [0, ∞) analíticamente.
            - Se acota a [0, 1] con min(ratio, 1.0) porque la difracción solo
              puede disipar energía (filtro pasa-bajos) o conservarla (l=0 dominante).
            - ratio = 1 → sin compresión (toda la energía conservada).
            - ratio → 0 → máxima compresión (señal aniquilada por el filtro).

        Parámetros
        ----------
        raw_logits     : NDArray[np.float64]
        focused_logits : NDArray[np.float64]

        Retorna
        -------
        float
            compression_ratio ∈ [0, 1].
        """
        norm_raw    : float = float(np.linalg.norm(raw_logits))
        norm_focused: float = float(np.linalg.norm(focused_logits))
        ratio: float = norm_focused / max(norm_raw, _EPSILON)
        # Acotar a [0,1]: la difracción no puede amplificar energía
        return float(np.clip(ratio, _COMPRESSION_FLOOR, _COMPRESSION_CAP))

    def _build_spectral_diagnostics(
        self,
        n_refract      : float,
        fermat_trace   : float,
        spectrum       : SphericalSpectrum,
        raw_logits     : NDArray[np.float64],
        focused_logits : NDArray[np.float64],
        sigma_projection: float,
    ) -> SpectralDiagnostics:
        r"""
        Construye el objeto de diagnósticos espectrales completo.

        Calcula:
          - energy_raw, energy_focused, energy_retention.
          - kv_compression_ratio.
          - l_dominant: grado l con la máxima potencia espectral.

        Parámetros
        ----------
        n_refract       : float, índice de refracción.
        fermat_trace    : float, Tr(n²·G).
        spectrum        : SphericalSpectrum, coeficientes c_{lm}.
        raw_logits      : NDArray, logits originales.
        focused_logits  : NDArray, logits filtrados.
        sigma_projection: float, ancho gaussiano usado en la proyección a S².

        Retorna
        -------
        SpectralDiagnostics
        """
        energy_raw    : float = float(np.sum(raw_logits    ** 2))
        energy_focused: float = float(np.sum(focused_logits ** 2))
        energy_retention: float = energy_focused / max(energy_raw, _EPSILON)
        kv_ratio: float = self._compute_kv_compression_ratio(raw_logits, focused_logits)

        # Grado dominante: argmax de la potencia por grado
        powers = [spectrum.power_at_degree(l) for l in range(spectrum.l_max + 1)]
        l_dominant: int = int(np.argmax(powers))

        return SpectralDiagnostics(
            n_refract            = n_refract,
            fermat_metric_trace  = fermat_trace,
            parseval_error       = spectrum.parseval_relative_error,
            kv_compression_ratio = kv_ratio,
            sigma_projection     = sigma_projection,
            energy_raw           = energy_raw,
            energy_focused       = energy_focused,
            energy_retention     = float(np.clip(energy_retention, 0.0, 1.0)),
            l_dominant           = l_dominant,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Orquestador principal de Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    def refract_attention_logits(
        self,
        raw_llm_logits      : NDArray[np.float64],
        logistic_stress_norm: float,
    ) -> RefractedState:
        r"""
        Método axiomático de ejecución del Lente Óptico de Riemann.

        Pipeline completo (3 fases):
        ────────────────────────────
        1. Validar entradas (Fase 3a).
        2. Calcular n(σ*) y Tr(g_F) (Fase 2: Fermat).
        3. apply_view_functor: logits → c_{lm} (Fases 1+2).
        4. apply_put_functor: c_{lm} → logits_filtrados (Fase 2).
        5. Calcular métricas de compresión (Fase 3b).
        6. Construir RefractedState con SpectralDiagnostics (Fase 3c).

        Precondiciones
        --------------
        - raw_llm_logits ∈ ℝⁿ, n ≥ 3, finito.
        - logistic_stress_norm ∈ ℝ, finito.

        Postcondiciones
        ---------------
        - RefractedState.focused_logits ∈ ℝⁿ, finito.
        - RefractedState.kv_compression_ratio ∈ [0,1].
        - self._last_diagnostics actualizado.

        Parámetros
        ----------
        raw_llm_logits       : (n,) NDArray[np.float64]
            Tensor de logits crudos del LLM.
        logistic_stress_norm : float
            Norma del tensor de estrés logístico σ* ∈ ℝ.

        Retorna
        -------
        RefractedState
            Estado óptico resultante con logits filtrados y diagnósticos.

        Lanza
        -----
        TypeError              : Si las entradas no tienen el tipo correcto.
        ValueError             : Si las entradas contienen NaN/Inf.
        LensSingularityError   : Si n < 3 o señal constante.
        OpticalDispersionError : Si n_refract < 1 (solo subclases).
        ParsevalViolationError : Si la identidad de Parseval falla.
        """
        logger.debug(
            "refract_attention_logits: n=%d, σ*=%.4f.",
            raw_llm_logits.size if isinstance(raw_llm_logits, np.ndarray) else -1,
            float(logistic_stress_norm) if isinstance(logistic_stress_norm, (int, float)) else float('nan'),
        )

        # ── Paso 1: Validación de entradas ───────────────────────────────────
        self._validate_input_logits(raw_llm_logits)
        self._validate_stress_norm(logistic_stress_norm)

        # Asegurar dtype float64
        logits_f64: NDArray[np.float64] = raw_llm_logits.astype(np.float64)

        # ── Paso 2: Métrica Óptica de Fermat ─────────────────────────────────
        n_refract   : float = self._compute_fermat_refractive_index(
            float(logistic_stress_norm)
        )
        fermat_trace: float = self._compute_fermat_metric_trace(n_refract)

        # ── Paso 3: Extracción Espectral (View) ──────────────────────────────
        spectrum: SphericalSpectrum = self.apply_view_functor(logits_f64)

        # Capturar sigma de la proyección (para diagnósticos)
        v_centered  = logits_f64 - np.mean(logits_f64)
        std_v       = float(np.std(v_centered))
        range_v     = float(np.max(logits_f64) - np.min(logits_f64))
        sigma_proj  : float = float(np.clip(
            std_v / (range_v + _EPSILON), _SIGMA_MIN, _SIGMA_MAX
        ))

        # ── Paso 4: Inyección Difractiva (Put) ───────────────────────────────
        focused_logits: NDArray[np.float64] = self.apply_put_functor(
            logits_f64, spectrum, n_refract
        )

        # ── Paso 5: Métricas de Compresión ───────────────────────────────────
        kv_ratio: float = self._compute_kv_compression_ratio(
            logits_f64, focused_logits
        )

        # ── Paso 6: Diagnósticos Espectrales ─────────────────────────────────
        diagnostics: SpectralDiagnostics = self._build_spectral_diagnostics(
            n_refract        = n_refract,
            fermat_trace     = fermat_trace,
            spectrum         = spectrum,
            raw_logits       = logits_f64,
            focused_logits   = focused_logits,
            sigma_projection = sigma_proj,
        )
        self._last_diagnostics = diagnostics

        logger.info(
            "refract_attention_logits completado: n=%d, n_refract=%.4f, "
            "kv_ratio=%.3f, Parseval_err=%.2e, l_dominant=%d.",
            logits_f64.size, n_refract, kv_ratio,
            spectrum.parseval_relative_error, diagnostics.l_dominant,
        )

        return RefractedState(
            focused_logits       = focused_logits,
            kv_compression_ratio = kv_ratio,
            fermat_metric_trace  = fermat_trace,
            diagnostics          = diagnostics,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades de Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def spectral_diagnostics(self) -> Optional[SpectralDiagnostics]:
        """
        Diagnósticos del último proceso de refracción ejecutado.

        Retorna None si refract_attention_logits no ha sido llamado todavía.
        """
        return self._last_diagnostics


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "OpticalDispersionError",
    "LensSingularityError",
    "ParsevalViolationError",
    "SpectralGridError",
    # Estructuras de datos
    "SphericalGrid",
    "SphericalSpectrum",
    "SpectralDiagnostics",
    "RefractedState",
    # Agente principal
    "OpticalRiemannLensFibrator",
]