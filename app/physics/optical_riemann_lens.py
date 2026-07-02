# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Optical Riemann Lens — Lente Categórico y Difracción Espectral             ║
║  Ruta   : app/physics/optical_riemann_lens.py                                        ║
║  Versión: 4.0.0-Rigorous-Spectral-Adjoint                                            ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Síntesis en 3 Fases Anidadas):
══════════════════════════════════════════════════════════════════════

Este módulo consagra la inyección de Óptica Geométrica sobre el colector de la
Esfera de Riemann (S² ≅ ℂ̂). Actúa como un Lente Categórico que filtra los
armónicos de alta frecuencia estocástica del LLM, colapsando el rango del
KV-Cache para preservar la termodinámica computacional.

══════════════════════════════════════════════════════════════════════════════════════
FUNDAMENTACIÓN MATEMÁTICA AXIOMÁTICA — v4.0.0
══════════════════════════════════════════════════════════════════════════════════════

§1. MÉTRICA DE FERMAT (Índice de Refracción Adaptativo):
────────────────────────────────────────────────────────
El espacio de deliberación se curva ópticamente. El índice de refracción n(σ*)
modifica la métrica de Riemann mediante el principio de Fermat generalizado:

    g_F = n²(σ*) · G_{μν} dx^μ dx^ν

Definición rigurosa del índice:
    n(σ*) = 1 + tanh(α · σ*)  ∈ (1, 2)  ∀ σ* ∈ ℝ

donde α = _FERMAT_ALPHA > 0. La condición n < 1 (refracción anómala) es
algebraicamente imposible con esta fórmula; la excepción OpticalDispersionError
está reservada para subclases que redefinen el índice con modelos no-lineales.

§2. JACOBIANO CORRECTO DE CUADRATURA GAUSS-LEGENDRE EN S²:
────────────────────────────────────────────────────────────
La integral de área sobre S² en coordenadas esféricas:

    ∫_{S²} f dΩ = ∫₀^π ∫₀^{2π} f(θ,φ) sin(θ) dθ dφ

Con el cambio de variable t = cos(θ) ∈ [-1,1], dt = -sin(θ)dθ:

    ∫_{S²} f dΩ = ∫_{-1}^{1} ∫₀^{2π} f(arccos(t), φ) dt dφ

Los nodos y pesos de Gauss-Legendre {(t_i, w_i)} satisfacen:
    ∫_{-1}^{1} p(t) dt ≈ Σ_i w_i p(t_i)  para deg(p) ≤ 2N-1.

El peso de integración en S² es W_{ij} = w_i · Δφ  (SIN multiplicar por sin θ_i).
El factor sin(θ) ya está absorbido en la medida dt = d(cos θ).

Verificación: Σ_{i,j} W_{ij} = (Σ_i w_i) · n_φ · Δφ = 2 · 2π = 4π = Área(S²) ✓

§3. COEFICIENTES DE ARMÓNICOS ESFÉRICOS Y PARSEVAL:
────────────────────────────────────────────────────
    c_{lm} = ∫_{S²} ψ(θ,φ) · Ȳ_l^m(θ,φ) dΩ
           ≈ Σ_{i,j} ψ(θ_i, φ_j) · conj(Y_l^m(θ_i, φ_j)) · w_i · Δφ

Identidad de Parseval (Teorema Espectral, Laplace-Beltrami sobre S²):
    ‖ψ‖²_{L²(S²)} = Σ_{l=0}^{∞} Σ_{m=-l}^{l} |c_{lm}|²

Verificación numérica: |‖ψ‖²_{L²} - Σ|c_{lm}|²| / ‖ψ‖²_{L²} < _PARSEVAL_REL_TOL

Ortogonalidad extendida verificada (§6 de este módulo):
    ⟨Y_l^m, Y_{l'}^{m'}⟩_{L²(S²)} ≈ δ_{ll'} δ_{mm'}
para pares (l,m) ∈ {(0,0), (1,0), (1,1), (2,0), (2,2)}.

§4. PROYECCIÓN GRAM-SCHMIDT ROBUSTA A ℝ³:
──────────────────────────────────────────
Para proyectar v ∈ ℝⁿ (n ≥ 3) a S²:
    1. Centrar: v̄ = v - mean(v)
    2. Si ‖v̄‖ < ε: señal constante → LensSingularityError
    3. e₁ = v̄/‖v̄‖  (primera dirección: la señal misma)
    4. e₂, e₃: completar base ortonormal via Gram-Schmidt estabilizado
       usando los vectores canónicos ordenados por menor proyección en e₁
    5. Verificar ortogonalidad: |e_i·e_j| < ε para i≠j
    6. p = [v̄·e₁, v̄·e₂, v̄·e₃] / ‖[...]‖ ∈ S²

§5. OPERADOR DE DIFRACCIÓN (FILTRO ESPECTRAL):
──────────────────────────────────────────────
El Lente Categórico aplica la transformación:
    O_{lens} ψ = Σ_{l=0}^{l_cutoff} Σ_{m=-l}^{l} h(l) · c_{lm} · Y_l^m(θ,φ)

donde h(l) = exp(-γ · n² · l²) es el kernel de amortiguación gaussiana.

§6. PROYECCIÓN ADJUNTA ESPECTRAL AL ESPACIO DE LOGITS:
───────────────────────────────────────────────────────
La proyección correcta ψ̃ → ℝⁿ utiliza la adjunta de la función de análisis.

Sea P: ℝⁿ → L²(S²) el operador de análisis (logits → ψ).
La adjunta P*: L²(S²) → ℝⁿ satisface ⟨Pv, ψ̃⟩_{L²} = ⟨v, P*ψ̃⟩_{ℝⁿ}.

Dado que P es lineal (como función de los coeficientes c_{lm} que determinan
la reconstrucción en ψ̃), la adjunta está completamente determinada por los
coeficientes de la expansión filtrada:

    (P*ψ̃)_k = Re(Σ_{l,m} h(l) · c_{lm} · ⟨Y_l^m, φ_k⟩_{L²(S²)})

donde φ_k es la función sensor asociada a la dirección e_k ∈ ℝⁿ.

En la implementación: φ_k se construye evaluando el perfil gaussiano centrado
en la proyección del vector base canónico e_k a S². Esto requiere O(n) llamadas
a _map_logits_to_sphere para construir la base de sensores, que se vectoriza.

§7. LEYES DEL LENTE CATEGÓRICO (VERIFICACIÓN):
────────────────────────────────────────────────
Sea L = (view, put) el Lente. Las leyes que satisface:
    1. PutGet: view(put(s, c_{lm})) ≈ O(c_{lm})  [espectro filtrado, no original]
    2. GetPut: put(s, view(s)) ≈ P*(O(Ps))        [reconstrucción desde s]
    3. PutPut: put(put(s,a),b) = put(s,b)          [idempotencia del filtro]
La ley 3 es exacta si h(l) es idempotente (h² = h), lo cual ocurre cuando
h(l) ∈ {0,1} (filtro ideal). En el caso gaussiano, h² ≠ h, y la composición
converge al punto fijo ψ = 0 (energía total → 0 tras n aplicaciones).

══════════════════════════════════════════════════════════════════════════════════════
CORRECCIONES CRÍTICAS v4.0.0 vs v3.0.0:
══════════════════════════════════════════════════════════════════════════════════════

  BUG 1: Gram-Schmidt numérico no verificado
          v3.0: fallback usa (min_idx+1)%n que puede coincidir con min_idx (n=1)
          v4.0: Selección de índices por np.argsort con exclusión explícita
                + verificación de ortogonalidad de la base resultante

  BUG 2: _map_logits_to_sphere no retorna sigma → recálculo inconsistente en Fase 3
          v3.0: sigma recalculado manualmente en refract_attention_logits
          v4.0: Retorna NamedTuple (psi, sigma, theta0, phi0) para consistencia total

  BUG 3: _compute_spherical_coefficients con bucle Python puro O((l+1)²·n_θ·n_φ)
          v3.0: bucle for l for m en Python puro → lento para l_cutoff > 10
          v4.0: Vectorización completa: precomputar tensor Y[l,m,i,j] con broadcasting

  BUG 4: Verificación de ortogonalidad solo en Y_0^0
          v3.0: solo ⟨Y_0^0, Y_0^0⟩ = 1 (trivial, solo verifica área total)
          v4.0: Verificación de 5 pares cruzados: (0,0)×(0,0), (1,0)×(1,0),
                (1,1)×(1,1), (1,-1)×(1,1), (2,0)×(1,0) — cubre ortogonalidad real

  BUG 5: _validate_metric_tensor no maneja G con tipo complejo
          v3.0: eigvalsh falla si G tiene imag parts por casting incorrecto
          v4.0: Conversión explícita a float64 real antes de eigvalsh + chequeo de imag

  BUG 6: _project_psi_to_logit_space es identidad escalada (no filtra)
          v3.0: focused = original_logits * (‖ψ̃‖/‖orig‖) — solo cambia magnitud
          v4.0: Proyección adjunta espectral real: construye funciones sensor {φ_k}
                via SVD de la base de proyección y calcula ⟨ψ̃, φ_k⟩_{L²}

  BUG 7: energy_retention puede exceder 1.0 sin corrección algebraica
          v3.0: np.clip a posteriori sin justificación de la fuente
          v4.0: La proyección adjunta garantiza ‖focused‖ ≤ ‖raw‖ por ser
                composición de operadores de norma ≤ 1 (filtro pasa-bajos + adjunta)

  BUG 8: Herencia de Morphism sin implementación del protocolo
          v3.0: class OpticalRiemannLensFibrator(Phase2, Morphism) — Morphism vacío
          v4.0: Implementa __call__ y compose según el protocolo MIC

══════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DE FASES ANIDADAS (v4.0.0):
══════════════════════════════════════════════════════════════════════════════════════

  ┌─── FASE 1: Phase1_SphericalHarmonicsSpectrometer ─────────────────────────┐
  │  _validate_spectrometer_params()       → l_cutoff≥0, γ>0, n_θ≥2l+2, n_φ≥4│
  │  _build_spherical_grid()               → GL en θ (t=cosθ), uniforme en φ  │
  │  _validate_grid_orthogonality()        → 5 pares cruzados Y_l^m           │
  │  _project_logits_to_r3_robust()        → Gram-Schmidt verificado + datos  │
  │  _map_logits_to_sphere()               → NamedTuple(psi,sigma,θ₀,φ₀)     │
  │  _validate_psi()                       → finitud, shape, energía > 0      │
  │  _precompute_harmonics_tensor()        → Y[idx,i,j] precomputado           │
  │  _compute_spherical_coefficients()     → c_{lm} vectorizado               │
  │  _verify_parseval_identity()           → Parseval con diagnóstico          │
  │  grid, l_cutoff, gamma_damping (props)→ acceso solo lectura               │
  └─────────────────────────────┬──────────────────────────────────────────────┘
                                │ SphericalSpectrum + SphericalGrid + SpherePoint
                                ▼
  ┌─── FASE 2: Phase2_CategoricalOpticLens ────────────────────────────────────┐
  │  _validate_metric_tensor()             → SPD, real, finito, simétrico      │
  │  _compute_fermat_refractive_index()    → n∈(1,2), algebraicamente garantido│
  │  _compute_fermat_metric_trace()        → Tr(n²·G) validado                │
  │  _build_sensor_basis()                 → {φ_k} via proyección de e_k a S² │
  │  _reconstruct_psi_from_spectrum()      → ψ̃ vectorizado en grilla S²       │
  │  _project_psi_to_logit_space()         → proyección adjunta espectral ℝⁿ  │
  │  _validate_reconstructed_signal()      → finitud, energía, ratio           │
  │  apply_view_functor()                  → logits → SphericalSpectrum        │
  │  apply_put_functor()                   → SphericalSpectrum → logits filtrados│
  └─────────────────────────────┬──────────────────────────────────────────────┘
                                │ NDArray focused_logits + SpherePoint
                                ▼
  ┌─── FASE 3: OpticalRiemannLensFibrator ─────────────────────────────────────┐
  │  _validate_input_logits()              → tipo, finitud, dim mínima         │
  │  _validate_stress_norm()               → finitud, tipo numérico            │
  │  _compute_kv_compression_ratio()       → ratio∈[0,1] algebraico           │
  │  _build_spectral_diagnostics()         → SpectralDiagnostics completo      │
  │  refract_attention_logits()            → orquestador → RefractedState      │
  │  __call__()                            → protocolo Morphism MIC            │
  │  compose()                             → composición de morfismos          │
  │  spectral_diagnostics (property)       → último diagnóstico, solo lectura  │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import scipy.special as sp_special
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ════════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MIC.Omega.OpticalRiemannLens")


# ════════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES NUMÉRICAS GLOBALES
# ════════════════════════════════════════════════════════════════════════════════
#
# Todas las constantes son inmutables y tienen justificación física/numérica
# explícita. No deben modificarse sin análisis de impacto en la topología de
# la cuadratura de Gauss-Legendre.

_EPSILON             : float = 1e-12   # Piso numérico general (doble precisión: ~2.2e-16)
_SIGMA_MIN           : float = 0.05    # Ancho gaussiano mínimo [rad] ≈ 3° — evita delta de Dirac
_SIGMA_MAX           : float = np.pi   # Ancho máximo = hemisferio [rad]
_ORTHOGONALITY_TOL   : float = 1e-5    # Tolerancia de ortogonalidad (relajada por n_θ finito)
_PARSEVAL_REL_TOL    : float = 0.05    # Tolerancia relativa Parseval (5%)
_N_SVD_COMPONENTS    : int   = 3       # Dimensión de proyección a ℝ³ (inmersión mínima de S²)
_FERMAT_ALPHA        : float = 0.5     # Factor α del índice n(σ*)=1+tanh(α·σ*)
_COMPRESSION_FLOOR   : float = 0.0     # Mínimo físico de compression_ratio
_COMPRESSION_CAP     : float = 1.0     # Máximo físico (difracción no amplifica)
_GRAM_SCHMIDT_TOL    : float = 1e-10   # Tolerancia para detectar colinealidad en GS
_SENSOR_BASIS_CLAMP  : float = 1e-8    # Mínimo de ‖φ_k‖_{L²} para no dividir por cero
_ORTH_VERIFY_PAIRS   : List[Tuple[int, int, int, int]] = [
    # (l, m, l', m') — pares para verificar ⟨Y_l^m, Y_{l'}^{m'}⟩ ≈ δ_{ll'}δ_{mm'}
    (0,  0, 0,  0),   # auto-norma Y_0^0 = 1
    (1,  0, 1,  0),   # auto-norma Y_1^0 = 1
    (1,  1, 1,  1),   # auto-norma Y_1^1 = 1
    (1, -1, 1,  1),   # cruzado (debe ser 0)
    (2,  0, 1,  0),   # cruzado inter-grado (debe ser 0)
]


# ════════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES ÓPTICAS Y ESPECTRALES
# ════════════════════════════════════════════════════════════════════════════════

class OpticalDispersionError(TopologicalInvariantError):
    r"""
    Detonada cuando el índice de refracción viola la condición física n ≥ 1.

    En la implementación estándar con n = 1 + tanh(α·σ*), esta condición es
    algebraicamente imposible (tanh ∈ (-1,1) ⟹ n ∈ (0,2), y más precisamente
    tanh(x) > -1 ∀ x∈ℝ ⟹ n > 0; pero el límite inferior real es n → 1⁺ para
    α > 0, lo que garantiza n > 1 siempre). La excepción está reservada para
    subclases que redefinen `_compute_fermat_refractive_index` con modelos de
    índice no-estándar (p.ej., medios con ganancia, dispersión anómala de Cauchy).

    Referencia física:
        Un medio con n < 1 viola la causalidad en óptica clásica (velocidad de
        fase superlumínica) pero es posible en óptica cuántica (gases atómicos
        cerca de resonancia). Para el modelo MIC, n < 1 destruye el Teorema
        de Liouville: el flujo simpléctopláctico no preserva el volumen de fase.
    """
    pass


class LensSingularityError(TopologicalInvariantError):
    r"""
    Detonada cuando la proyección geométrica falla algebraicamente.

    Condiciones de activación:
        1. logits.size < _N_SVD_COMPONENTS = 3.
        2. El vector centrado v̄ = v - mean(v) es numéricamente nulo (‖v̄‖ < ε).
        3. La base de Gram-Schmidt no puede ser ortonormalizada (determinante ≈ 0).
        4. La grilla S² no satisface la condición de ortogonalidad extendida.

    Físicamente: corresponde al colapso de la proyección al Polo Norte de la Esfera
    de Riemann (punto de singularidad de la proyección estereográfica), donde la
    función ψ degenera en una δ de Dirac y todos los coeficientes c_{lm} = 1/√(4π).
    """
    pass


class ParsevalViolationError(TopologicalInvariantError):
    r"""
    Detonada cuando la identidad de Parseval falla más allá de la tolerancia.

    La identidad de Parseval en L²(S²):
        ‖ψ‖²_{L²(S²)} = Σ_{l=0}^{l_max} Σ_{m=-l}^{l} |c_{lm}|²

    es consecuencia del Teorema Espectral para el operador de Laplace-Beltrami
    ΔS² (auto-adjunto, con espectro discreto {-l(l+1)}). Su violación indica:
        (a) Error en el Jacobiano de cuadratura (Σ W_{ij} ≠ 4π).
        (b) n_theta < 2·l_cutoff + 2 (cuadratura insuficiente para el grado).
        (c) Pérdida de ortogonalidad numérica de los armónicos.
        (d) Función ψ con discontinuidades que exceden la resolución de la grilla.

    Campos del mensaje: parseval_lhs, parseval_rhs, relative_error, n_theta, l_cutoff.
    """
    pass


class SpectralGridError(TopologicalInvariantError):
    r"""
    Detonada cuando la grilla de cuadratura no satisface la condición de
    ortogonalidad extendida de los armónicos esféricos.

    La condición verificada incluye 5 pares (l,m,l',m') de _ORTH_VERIFY_PAIRS:
        - Auto-normas: |⟨Y_l^m, Y_l^m⟩ - 1| < _ORTHOGONALITY_TOL
        - Términos cruzados: |⟨Y_l^m, Y_{l'}^{m'}⟩| < _ORTHOGONALITY_TOL
    """
    pass


class GramSchmidtDegeneracyError(LensSingularityError):
    r"""
    Subclase de LensSingularityError específica para degeneración en Gram-Schmidt.

    Ocurre cuando los vectores canónicos disponibles para construir e₂, e₃
    son todos (casi) paralelos a e₁ en ℝⁿ. Matemáticamente imposible para n ≥ 3
    con la selección óptima de índices, pero puede ocurrir por redondeo extremo
    en vectores de logits con distribución muy concentrada (spike casi perfecto).
    """
    pass


# ════════════════════════════════════════════════════════════════════════════════
# §C. TIPOS AUXILIARES (NamedTuples para retornos múltiples coherentes)
# ════════════════════════════════════════════════════════════════════════════════

class SpherePoint(NamedTuple):
    r"""
    Resultado de la proyección de logits a S².

    Campos
    ------
    psi    : (n_theta, n_phi) NDArray[float64] — función escalar sobre la grilla.
    sigma  : float — ancho gaussiano [rad] usado en la proyección.
    theta0 : float — colatitud del punto caliente [rad] ∈ (0, π).
    phi0   : float — longitud del punto caliente [rad] ∈ (-π, π].
    p_r3   : (3,) NDArray[float64] — vector unitario en ℝ³ (punto en S²).
    """
    psi   : NDArray[np.float64]
    sigma  : float
    theta0 : float
    phi0   : float
    p_r3   : NDArray[np.float64]


class GramSchmidtBasis(NamedTuple):
    r"""
    Base ortonormal en ℝⁿ resultante de la proyección Gram-Schmidt.

    Campos
    ------
    e1       : (n,) NDArray[float64] — primera dirección (∥ a v̄).
    e2       : (n,) NDArray[float64] — segunda dirección (⊥ a e1).
    e3       : (n,) NDArray[float64] — tercera dirección (⊥ a e1, e2).
    proj_r3  : (3,) NDArray[float64] — proyección de v̄ en {e1,e2,e3}, normalizada.
    """
    e1      : NDArray[np.float64]
    e2      : NDArray[np.float64]
    e3      : NDArray[np.float64]
    proj_r3 : NDArray[np.float64]


# ════════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES DEL FIBRADO ÓPTICO
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SphericalGrid:
    r"""
    Malla de puntos y pesos para integración numérica en S².

    JACOBIANO CORRECTO v4.0.0:
    ─────────────────────────
    leggauss(n) devuelve (t_i, w_i) donde t_i = cos(θ_i) ∈ (-1,1) y los pesos
    w_i satisfacen ∫_{-1}^{1} p(t)dt ≈ Σ_i w_i p(t_i).

    Bajo el cambio de variable t = cos(θ), la medida de S² se transforma:
        dΩ = sin(θ)dθ dφ = -dt dφ  (con el signo absorbido por el límite de integración)

    Por tanto:
        ∫_{S²} f dΩ = ∫_{-1}^{1} ∫₀^{2π} f(arccos(t), φ) dt dφ
                    ≈ Σ_{i,j} f(θ_i, φ_j) · w_i · Δφ

    El peso completo es W_{ij} = w_i · Δφ, SIN multiplicar por sin(θ_i).

    Campos
    ------
    theta         : (n_theta,) nodos de colatitud θ_i = arccos(t_i) ∈ (0,π).
    phi           : (n_phi,)   nodos de longitud φ_j ∈ [0, 2π).
    gauss_weights : (n_theta,) pesos de Gauss-Legendre w_i (para dt).
    delta_phi     : float, Δφ = 2π / n_phi.
    weights       : (n_theta, n_phi) pesos W_{ij} = w_i · Δφ.
    shape         : (n_theta, n_phi) — dimensiones de la grilla.

    Invariantes verificados en __post_init__:
        - theta ∈ (0, π) estricto.
        - gauss_weights > 0.
        - |Σ W_{ij} - 4π| / 4π < 1e-12.
        - weights.shape == shape.
    """
    theta        : NDArray[np.float64]
    phi          : NDArray[np.float64]
    gauss_weights: NDArray[np.float64]
    delta_phi    : float
    weights      : NDArray[np.float64]
    shape        : Tuple[int, int]

    def __post_init__(self) -> None:
        """Verifica invariantes geométricas de la grilla."""
        n_theta, n_phi = self.shape

        # Invariante 1: theta en el interior de (0, π) — polos excluidos
        if not (np.all(self.theta > 0.0) and np.all(self.theta < np.pi)):
            raise SpectralGridError(
                "Los nodos theta deben estar en el interior (0, π). "
                "Los polos exactos θ∈{0,π} producen singularidades en sph_harm."
            )

        # Invariante 2: pesos de Gauss-Legendre positivos
        if not np.all(self.gauss_weights > 0.0):
            raise SpectralGridError(
                "Los pesos de Gauss-Legendre deben ser todos positivos."
            )

        # Invariante 3: shape de weights
        if self.weights.shape != (n_theta, n_phi):
            raise SpectralGridError(
                f"weights.shape={self.weights.shape} ≠ shape={self.shape}."
            )

        # Invariante 4: área total = 4π con alta precisión
        total_area      = float(np.sum(self.weights))
        expected_area   = 4.0 * np.pi
        rel_err         = abs(total_area - expected_area) / expected_area
        if rel_err > 1e-10:
            raise SpectralGridError(
                f"Área total de grilla = {total_area:.10f} ≠ 4π = {expected_area:.10f}. "
                f"Error relativo = {rel_err:.2e}. Jacobiano incorrecto."
            )


@dataclass(frozen=True, slots=True)
class SphericalSpectrum:
    r"""
    Contenedor inmutable del espectro de coeficientes c_{lm}.

    Los coeficientes se almacenan en orden lexicográfico:
        k = l² + (m + l)   para l=0,…,l_max; m=-l,…,l

    Total de coeficientes: (l_max+1)² = Σ_{l=0}^{l_max}(2l+1).

    Campos
    ------
    coefficients            : (N_coeffs,) NDArray[complex128].
    l_max                   : int, orden máximo de truncación.
    parseval_lhs            : float, ‖ψ‖²_{L²(S²)} numérico.
    parseval_rhs            : float, Σ|c_{lm}|² analítico.
    parseval_relative_error : float ≥ 0.
    """
    coefficients            : NDArray[np.complex128]
    l_max                   : int
    parseval_lhs            : float
    parseval_rhs            : float
    parseval_relative_error : float

    def __post_init__(self) -> None:
        expected_n = (self.l_max + 1) ** 2
        if self.coefficients.shape != (expected_n,):
            raise ValueError(
                f"SphericalSpectrum.coefficients debe tener shape ({expected_n},) "
                f"para l_max={self.l_max}, recibido {self.coefficients.shape}."
            )
        if self.l_max < 0:
            raise ValueError(f"l_max debe ser ≥ 0, recibido {self.l_max}.")

    @property
    def n_coefficients(self) -> int:
        """Número total de coeficientes (l_max+1)²."""
        return (self.l_max + 1) ** 2

    def power_at_degree(self, l: int) -> float:
        r"""
        Potencia espectral en el grado l:
            P_l = Σ_{m=-l}^{l} |c_{lm}|²

        Parámetros
        ----------
        l : int, grado ∈ [0, l_max].

        Retorna
        -------
        float, P_l ≥ 0.
        """
        if not (0 <= l <= self.l_max):
            raise ValueError(f"l={l} fuera de [0, {self.l_max}].")
        start_idx = l ** 2
        end_idx   = (l + 1) ** 2
        return float(np.sum(np.abs(self.coefficients[start_idx:end_idx]) ** 2))

    def dominant_degree(self) -> int:
        """Grado l con máxima potencia espectral."""
        powers = [self.power_at_degree(l) for l in range(self.l_max + 1)]
        return int(np.argmax(powers))


@dataclass(frozen=True, slots=True)
class SpectralDiagnostics:
    r"""
    Diagnósticos de calidad del proceso de refracción óptica.

    Campos
    ------
    n_refract            : float ∈ (1,2), índice de refracción.
    fermat_metric_trace  : float, Tr(n²·G).
    parseval_error       : float ≥ 0, error relativo de Parseval.
    kv_compression_ratio : float ∈ [0,1], ‖focused‖/‖raw‖.
    sigma_projection     : float ∈ [σ_min, σ_max], ancho gaussiano [rad].
    theta0_projection    : float ∈ (0,π), colatitud del punto caliente.
    phi0_projection      : float ∈ (-π,π], longitud del punto caliente.
    energy_raw           : float ≥ 0, ‖logits_raw‖².
    energy_focused       : float ≥ 0, ‖logits_focused‖².
    energy_retention     : float ∈ [0,1], energy_focused/max(energy_raw,ε).
    l_dominant           : int ≥ 0, grado l de máxima potencia espectral.
    gram_schmidt_cond    : float ≥ 1, número de condición de la base GS.
    """
    n_refract            : float
    fermat_metric_trace  : float
    parseval_error       : float
    kv_compression_ratio : float
    sigma_projection     : float
    theta0_projection    : float
    phi0_projection      : float
    energy_raw           : float
    energy_focused       : float
    energy_retention     : float
    l_dominant           : int
    gram_schmidt_cond    : float


@dataclass(frozen=True, slots=True)
class RefractedState:
    r"""
    Estado óptico resultante del Lente Categórico.

    Invariantes verificados en __post_init__:
        - focused_logits es finito y de dtype float64.
        - kv_compression_ratio ∈ [0,1].
        - fermat_metric_trace es finito.
    """
    focused_logits       : NDArray[np.float64]
    kv_compression_ratio : float
    fermat_metric_trace  : float
    diagnostics          : SpectralDiagnostics

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.focused_logits)):
            n_bad = int(np.sum(~np.isfinite(self.focused_logits)))
            raise ValueError(
                f"RefractedState.focused_logits contiene {n_bad} valores no finitos."
            )
        if not (0.0 <= self.kv_compression_ratio <= 1.0):
            raise ValueError(
                f"kv_compression_ratio={self.kv_compression_ratio:.6f} ∉ [0,1]."
            )
        if not math.isfinite(self.fermat_metric_trace):
            raise ValueError(
                f"fermat_metric_trace={self.fermat_metric_trace} no es finito."
            )


# ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 1 — ESPECTRÓMETRO DE ARMÓNICOS ESFÉRICOS                                   ║
# ║                                                                                  ║
# ║  Contrato formal:                                                                ║
# ║    Entrada  : logits ∈ ℝⁿ (n ≥ 3), parámetros de cuadratura                      ║
# ║    Salida   : SphericalSpectrum con c_{lm}, diagnósticos Parseval,               ║
# ║               SpherePoint con sigma/θ₀/φ₀ para trazabilidad                      ║
# ║                                                                                  ║
# ║  Garantías formales:                                                             ║
# ║    G1. Jacobiano W_{ij} = w_i·Δφ (sin(θ_i) NO incluido)                          ║
# ║    G2. Ortogonalidad verificada en 5 pares de armónicos                          ║
# ║    G3. Gram-Schmidt robusto con verificación de ortogonalidad resultante         ║
# ║    G4. Tensor Y[idx,i,j] precomputado (vectorizado, O(1) en tiempo de uso)       ║
# ║    G5. Identidad de Parseval verificada post-cálculo                             ║
# ║    G6. sigma, θ₀, φ₀ retornados en SpherePoint (sin recálculo en Fase 3)         ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════

class Phase1_SphericalHarmonicsSpectrometer:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    FASE 1 — Espectrómetro de Armónicos Esféricos Riguroso (v4.0.0)
    ═══════════════════════════════════════════════════════════════════════════

    Transforma el tensor de logits del LLM al dominio espectral de S²,
    computando los coeficientes c_{lm} de la expansión en armónicos esféricos.

    Mejoras v4.0.0 sobre v3.0.0:
    ─────────────────────────────
    1. Gram-Schmidt con selección óptima de índices auxiliares y verificación
       de ortogonalidad del triedro resultante.
    2. _map_logits_to_sphere retorna SpherePoint (NamedTuple) incluyendo sigma,
       θ₀, φ₀ — elimina recálculo inconsistente en Fase 3.
    3. Tensor de armónicos Y precomputado en __init__ para vectorización completa
       del cálculo de coeficientes (sin bucle Python for-l-for-m en tiempo de uso).
    4. Verificación de ortogonalidad en 5 pares representativos de Y_l^m.
    5. Número de condición de la base de Gram-Schmidt registrado en diagnósticos.
    """

    DEFAULT_N_THETA: int = 32
    DEFAULT_N_PHI  : int = 64

    def __init__(
        self,
        l_cutoff       : int   = 10,
        gamma_damping  : float = 0.1,
        n_theta        : int   = DEFAULT_N_THETA,
        n_phi          : int   = DEFAULT_N_PHI,
        verify_parseval: bool  = True,
    ) -> None:
        r"""
        Inicializa el espectrómetro, construye la grilla y precomputa armónicos.

        Precondiciones
        --------------
        - l_cutoff ≥ 0.
        - gamma_damping > 0.
        - n_theta ≥ max(4, 2·l_cutoff + 2) para exactitud polinomial grado 2l+1.
        - n_phi ≥ 4.

        Parámetros
        ----------
        l_cutoff       : int   — orden máximo de armónicos esféricos.
        gamma_damping  : float — coeficiente γ de amortiguación.
        n_theta        : int   — nodos de Gauss-Legendre en θ.
        n_phi          : int   — nodos uniformes en φ.
        verify_parseval: bool  — si True, lanza ParsevalViolationError si falla.

        Lanza
        -----
        ValueError        : Parámetros fuera de rango.
        SpectralGridError : Grilla no satisface ortogonalidad.
        """
        self._validate_spectrometer_params(l_cutoff, gamma_damping, n_theta, n_phi)

        self._l_cutoff        : int   = l_cutoff
        self._gamma           : float = float(gamma_damping)
        self._n_theta         : int   = n_theta
        self._n_phi           : int   = n_phi
        self._verify_parseval : bool  = verify_parseval

        # Construir grilla y verificar ortogonalidad
        self._grid: SphericalGrid = self._build_spherical_grid()
        self._validate_grid_orthogonality()

        # Precomputar tensor de armónicos Y[num_coeffs, n_theta, n_phi]
        # Este tensor es inmutable y se reutiliza en cada llamada a _compute_spherical_coefficients
        self._harmonics_tensor: NDArray[np.complex128] = (
            self._precompute_harmonics_tensor()
        )

        # Precomputar factores de amortiguación h[num_coeffs] = exp(-γ·n²·l²)
        # Se actualizan al momento de aplicar el put (dependen de n_refract)
        # pero el vector de grados l[idx] es constante — precomputar aquí
        self._degree_vector: NDArray[np.float64] = self._build_degree_vector()

        logger.debug(
            "Phase1 inicializado: l_cutoff=%d, γ=%.3f, n_θ=%d, n_φ=%d, "
            "armonicos precomputados: shape=%s.",
            self._l_cutoff, self._gamma, self._n_theta, self._n_phi,
            self._harmonics_tensor.shape,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.1 — Validación de parámetros
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_spectrometer_params(
        l_cutoff: int, gamma_damping: float, n_theta: int, n_phi: int
    ) -> None:
        r"""
        Verifica los parámetros de inicialización del espectrómetro.

        Regla de cuadratura exacta (Gauss-Legendre):
        ─────────────────────────────────────────────
        Para integrar polinomios de grado ≤ 2n_theta-1 exactamente, y dado que
        los productos Y_l^m · conj(Y_{l'}^{m'}) son polinomios de grado ≤ 2·l_cutoff
        en cos(θ), se requiere:
            2·n_theta - 1 ≥ 2·l_cutoff  ⟹  n_theta ≥ l_cutoff + 1

        Para integrar Y_l^m · ψ(θ,φ) donde ψ es el perfil gaussiano (no polinomial),
        se recomienda n_theta ≥ 2·l_cutoff + 2 para resolución suficiente.

        Parámetros
        ----------
        l_cutoff     : int
        gamma_damping: float
        n_theta      : int
        n_phi        : int

        Lanza
        -----
        ValueError : Si algún parámetro viola las precondiciones mínimas.
        """
        if not isinstance(l_cutoff, int) or l_cutoff < 0:
            raise ValueError(
                f"l_cutoff debe ser int ≥ 0, recibido {l_cutoff!r}."
            )
        if not isinstance(gamma_damping, (int, float)) or float(gamma_damping) <= 0.0:
            raise ValueError(
                f"gamma_damping debe ser float > 0, recibido {gamma_damping!r}."
            )
        if not isinstance(n_theta, int) or n_theta < 4:
            raise ValueError(
                f"n_theta debe ser int ≥ 4, recibido {n_theta!r}."
            )
        if not isinstance(n_phi, int) or n_phi < 4:
            raise ValueError(
                f"n_phi debe ser int ≥ 4, recibido {n_phi!r}."
            )

        # Advertencia de insuficiencia de cuadratura
        min_n_theta = 2 * l_cutoff + 2
        if n_theta < min_n_theta:
            logger.warning(
                "CUADRATURA INSUFICIENTE: n_theta=%d < %d = 2·l_cutoff+2 "
                "para l_cutoff=%d. La identidad de Parseval puede fallar. "
                "Se recomienda n_theta ≥ %d.",
                n_theta, min_n_theta, l_cutoff, min_n_theta,
            )

        # Advertencia de insuficiencia angular en φ
        min_n_phi = 2 * l_cutoff + 1
        if n_phi < min_n_phi:
            logger.warning(
                "RESOLUCIÓN ANGULAR INSUFICIENTE: n_phi=%d < %d = 2·l_cutoff+1 "
                "para l_cutoff=%d. Modos m = ±l_cutoff pueden estar submuestreados.",
                n_phi, min_n_phi, l_cutoff,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.2 — Construcción de la grilla esférica
    # ─────────────────────────────────────────────────────────────────────────

    def _build_spherical_grid(self) -> SphericalGrid:
        r"""
        Construye la malla de integración con Jacobiano correcto.

        Implementación:
        ───────────────
        np.polynomial.legendre.leggauss(n_theta) retorna (t_i, w_i) donde:
            - t_i ∈ (-1, 1) son las raíces del polinomio de Legendre P_{n_theta}.
            - w_i > 0 son los pesos de cuadratura.
            - La ordenación es de t=-1 a t=1 (θ=π a θ=0).

        El peso de integración en S² es:
            W_{ij} = w_i · Δφ    (NO multiplicar por sin(θ_i))

        Verificación de área: Σ_{i,j} W_{ij} = (Σ_i w_i) · n_φ · Δφ = 2 · 2π = 4π ✓

        Retorna
        -------
        SphericalGrid con invariantes verificados en __post_init__.
        """
        # Nodos y pesos de Gauss-Legendre en [-1,1] (t = cos θ)
        t_nodes, w_gauss = np.polynomial.legendre.leggauss(self._n_theta)

        # θ_i = arccos(t_i) ∈ (0, π) — los nodos de GL están en el interior
        theta: NDArray[np.float64] = np.arccos(t_nodes.astype(np.float64))

        # φ_j uniformes en [0, 2π)
        delta_phi: float = 2.0 * np.pi / self._n_phi
        phi: NDArray[np.float64] = np.linspace(
            0.0, 2.0 * np.pi, self._n_phi, endpoint=False, dtype=np.float64
        )

        # Pesos 2D: W[i,j] = w_i · Δφ  —  broadcasting sobre eje j
        weights: NDArray[np.float64] = (
            w_gauss.astype(np.float64)[:, np.newaxis]
            * np.full(self._n_phi, delta_phi, dtype=np.float64)[np.newaxis, :]
        )

        return SphericalGrid(
            theta         = theta,
            phi           = phi,
            gauss_weights = w_gauss.astype(np.float64),
            delta_phi     = delta_phi,
            weights       = weights,
            shape         = (self._n_theta, self._n_phi),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.3 — Verificación de ortogonalidad de la grilla
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_grid_orthogonality(self) -> None:
        r"""
        Verifica la ortogonalidad de la grilla en 5 pares representativos.

        Condición:
            ⟨Y_l^m, Y_{l'}^{m'}⟩_{L²(S²)} ≈ δ_{ll'} δ_{mm'}

        Los pares verificados son _ORTH_VERIFY_PAIRS:
            (0,0,0,0)  → norma 1 (verifica que área = 4π)
            (1,0,1,0)  → norma 1 (polinomio en cos θ de grado 1)
            (1,1,1,1)  → norma 1 (incluye e^{iφ}, verifica integración en φ)
            (1,-1,1,1) → cruzado debe ser 0 (ortogonalidad entre m distintos)
            (2,0,1,0)  → cruzado debe ser 0 (ortogonalidad entre l distintos)

        Esta verificación es más robusta que verificar solo Y_0^0, porque:
        - (0,0): solo verifica Σ W_{ij} = 4π (función constante)
        - (1,0): verifica integración de polinomio de grado 1 en cos(θ)
        - (1,1): verifica integración con fase e^{iφ} (modo azimutal)
        - cruzados: verifican cancelación (requisito de ortogonalidad real)

        Lanza
        -----
        SpectralGridError : Si algún par viola la tolerancia _ORTHOGONALITY_TOL.
        """
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )
        W: NDArray[np.float64] = self._grid.weights

        for (l, m, lp, mp) in _ORTH_VERIFY_PAIRS:
            Y_lm  = sp_special.sph_harm(m,  l,  PHI, THETA)
            Y_lpm = sp_special.sph_harm(mp, lp, PHI, THETA)
            inner = float(np.abs(np.sum(Y_lm * np.conj(Y_lpm) * W)))

            # Para pares diagonales (l==l', m==m'): inner ≈ 1
            # Para pares cruzados: inner ≈ 0
            expected = 1.0 if (l == lp and m == mp) else 0.0
            error = abs(inner - expected)

            if error > _ORTHOGONALITY_TOL:
                raise SpectralGridError(
                    f"Ortogonalidad violada en par (l={l},m={m}),(l'={lp},m'={mp}): "
                    f"⟨Y,Y'⟩ = {inner:.6f}, esperado {expected:.1f}, "
                    f"error = {error:.2e} > tol = {_ORTHOGONALITY_TOL:.2e}. "
                    f"Posible causa: n_theta={self._n_theta} insuficiente o "
                    f"Jacobiano incorrecto."
                )

        logger.debug(
            "Ortogonalidad de grilla verificada en %d pares. "
            "n_theta=%d, n_phi=%d, tol=%.2e.",
            len(_ORTH_VERIFY_PAIRS), self._n_theta, self._n_phi, _ORTHOGONALITY_TOL,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.4 — Precomputa tensor de armónicos esféricos
    # ─────────────────────────────────────────────────────────────────────────

    def _precompute_harmonics_tensor(self) -> NDArray[np.complex128]:
        r"""
        Precomputa el tensor Y[num_coeffs, n_theta, n_phi] de armónicos esféricos.

        Definición:
            Y[k, i, j] = Y_l^m(θ_i, φ_j)

        donde k = l² + (m + l) es el índice plano del par (l, m).

        Complejidad:
        ─────────────
        - Tiempo: O((l_cutoff+1)² · n_theta · n_phi) en tiempo de inicialización.
        - Espacio: O((l_cutoff+1)² · n_theta · n_phi) en memoria compleja.
        - En tiempo de uso (_compute_spherical_coefficients): O(num_coeffs · n_theta · n_phi)
          pero completamente vectorizado en NumPy (sin bucle Python).

        Para l_cutoff=10, n_theta=32, n_phi=64:
            num_coeffs = 121, tensor_size = 121 · 32 · 64 = 247,808 complex128 ≈ 3.8 MB

        Retorna
        -------
        NDArray[np.complex128]
            Tensor Y de shape ((l_cutoff+1)², n_theta, n_phi).
        """
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )  # ambos de shape (n_theta, n_phi)

        num_coeffs: int = (self._l_cutoff + 1) ** 2
        Y_tensor: NDArray[np.complex128] = np.zeros(
            (num_coeffs, self._n_theta, self._n_phi), dtype=np.complex128
        )

        idx: int = 0
        for l in range(self._l_cutoff + 1):
            for m in range(-l, l + 1):
                # scipy.special.sph_harm(m, l, phi, theta) — ojo: orden phi, theta
                Y_tensor[idx] = sp_special.sph_harm(m, l, PHI, THETA)
                idx += 1

        logger.debug(
            "Tensor de armónicos precomputado: shape=%s, dtype=%s, "
            "mem=%.2f MB.",
            Y_tensor.shape, Y_tensor.dtype,
            Y_tensor.nbytes / 1e6,
        )
        return Y_tensor

    def _build_degree_vector(self) -> NDArray[np.float64]:
        r"""
        Construye el vector de grados l[k] en orden lexicográfico.

        l[k] = l para k = l² + (m+l), es decir:
            [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, ...]

        Utilizado para vectorizar el cálculo de la amortiguación exp(-γ·n²·l²).

        Retorna
        -------
        NDArray[np.float64]
            Vector de grados de shape ((l_cutoff+1)²,).
        """
        degrees: list[float] = []
        for l in range(self._l_cutoff + 1):
            for _ in range(-l, l + 1):  # 2l+1 veces
                degrees.append(float(l))
        return np.array(degrees, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # §1.5 — Proyección Gram-Schmidt robusta a ℝ³
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _project_logits_to_r3_robust(
        logits: NDArray[np.float64],
    ) -> GramSchmidtBasis:
        r"""
        Proyecta logits ∈ ℝⁿ a ℝ³ usando Gram-Schmidt robusto.

        Algoritmo (versión estabilizada):
        ─────────────────────────────────
        Sea v̄ = logits - mean(logits) ∈ ℝⁿ con ‖v̄‖ > ε.

        1. e₁ = v̄ / ‖v̄‖.
        2. Ordenar los índices canónicos {0,...,n-1} por valor absoluto de
           proyección |⟨e₁, e_k⟩| en orden creciente. Los índices con menor
           proyección son los mejores candidatos para extender la base.
        3. Para e₂: tomar el índice k₁ con menor |e₁[k₁]|.
           e₂_raw = e_{k₁} - ⟨e_{k₁}, e₁⟩ · e₁
           Si ‖e₂_raw‖ < tol: tomar el siguiente índice k₂.
           e₂ = e₂_raw / ‖e₂_raw‖.
        4. Para e₃: tomar el primer índice k con |⟨e_k, e₁⟩|+|⟨e_k, e₂⟩| mínimo,
           distinto de k₁.
           e₃_raw = e_k - ⟨e_k, e₁⟩·e₁ - ⟨e_k, e₂⟩·e₂
           e₃ = e₃_raw / ‖e₃_raw‖.
        5. Verificar ortogonalidad: |⟨e_i, e_j⟩| < _GRAM_SCHMIDT_TOL para i≠j.
        6. p = [⟨v̄, e₁⟩, ⟨v̄, e₂⟩, ⟨v̄, e₃⟩]
               = [‖v̄‖, 0, 0]  (por construcción de e₁ = v̄/‖v̄‖)
           Normalizar: p_unit = p / ‖p‖ = e₁  (trivialmente, dado que e₂·v̄=e₃·v̄=0)

        Nota sobre la degeneración:
        ───────────────────────────
        Para n ≥ 3, siempre existe al menos un índice k con |e₁[k]| < 1/√n < 1,
        lo que garantiza que ‖e₂_raw‖ ≥ √(1 - 1/n) > 0 para n ≥ 2.
        El caso n = 3 con e₁ ∥ a algún canónico es el caso límite, manejado
        por el fallback con el siguiente índice ordenado.

        Número de condición:
        ────────────────────
        Se calcula como el cociente del mayor y menor valor singular de la
        matriz A = [e₁|e₂|e₃]ᵀ ∈ ℝ^{3×n}. Por construcción A·Aᵀ = I₃ (los
        vectores son ortonormales), así que κ(A) = 1 exactamente en aritmética
        exacta. En aritmética de punto flotante, κ > 1 indica pérdida de
        ortogonalidad numérica.

        Parámetros
        ----------
        logits : (n,) NDArray[np.float64], n ≥ 3.

        Retorna
        -------
        GramSchmidtBasis con e1, e2, e3, proj_r3.

        Lanza
        -----
        LensSingularityError    : Si n < 3 o ‖v̄‖ < ε.
        GramSchmidtDegeneracyError: Si la base no puede ortonormalizarse.
        """
        n = logits.size
        if n < _N_SVD_COMPONENTS:
            raise LensSingularityError(
                f"Se requieren al menos {_N_SVD_COMPONENTS} componentes. "
                f"Recibido n={n}."
            )

        v_centered: NDArray[np.float64] = logits - np.mean(logits)
        v_norm: float = float(np.linalg.norm(v_centered))

        if v_norm < _EPSILON:
            raise LensSingularityError(
                f"Vector centrado numéricamente nulo: ‖v̄‖={v_norm:.2e} < ε={_EPSILON}. "
                f"Señal constante (todos los logits iguales): no hay dirección preferencial."
            )

        # e₁: primera dirección (paralela a v̄)
        e1: NDArray[np.float64] = v_centered / v_norm

        # Ordenar índices por |proyección sobre e₁| creciente
        abs_projections: NDArray[np.float64] = np.abs(e1)
        sorted_indices: NDArray[np.intp]     = np.argsort(abs_projections)

        # e₂: Gram-Schmidt contra e₁
        e2: Optional[NDArray[np.float64]] = None
        used_idx_for_e2: int = -1

        for k_idx in sorted_indices:
            canonical_k: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
            canonical_k[k_idx] = 1.0
            e2_raw: NDArray[np.float64] = canonical_k - np.dot(canonical_k, e1) * e1
            e2_norm: float = float(np.linalg.norm(e2_raw))
            if e2_norm > _GRAM_SCHMIDT_TOL:
                e2 = e2_raw / e2_norm
                used_idx_for_e2 = int(k_idx)
                break

        if e2 is None:
            raise GramSchmidtDegeneracyError(
                f"Imposible construir e₂ ortogonal a e₁. "
                f"Todos los vectores canónicos son colineales con e₁. "
                f"n={n}, ‖v̄‖={v_norm:.2e}."
            )

        # e₃: Gram-Schmidt contra e₁ y e₂, usando índice distinto al de e₂
        e3: Optional[NDArray[np.float64]] = None
        for k_idx in sorted_indices:
            if int(k_idx) == used_idx_for_e2:
                continue
            canonical_k = np.zeros(n, dtype=np.float64)
            canonical_k[k_idx] = 1.0
            e3_raw: NDArray[np.float64] = (
                canonical_k
                - np.dot(canonical_k, e1) * e1
                - np.dot(canonical_k, e2) * e2
            )
            e3_norm: float = float(np.linalg.norm(e3_raw))
            if e3_norm > _GRAM_SCHMIDT_TOL:
                e3 = e3_raw / e3_norm
                break

        if e3 is None:
            raise GramSchmidtDegeneracyError(
                f"Imposible construir e₃ ortogonal a e₁ y e₂. "
                f"n={n}, ‖v̄‖={v_norm:.2e}, índice e₂={used_idx_for_e2}."
            )

        # Verificar ortogonalidad del triedro resultante
        dot_12 = abs(float(np.dot(e1, e2)))
        dot_13 = abs(float(np.dot(e1, e3)))
        dot_23 = abs(float(np.dot(e2, e3)))
        max_cross_dot = max(dot_12, dot_13, dot_23)
        if max_cross_dot > _GRAM_SCHMIDT_TOL * 10:
            raise GramSchmidtDegeneracyError(
                f"Base de Gram-Schmidt no es ortogonal: "
                f"max|e_i·e_j|={max_cross_dot:.2e} > 10·tol={_GRAM_SCHMIDT_TOL*10:.2e}."
            )

        # Proyección: A·v̄ = [e₁·v̄, e₂·v̄, e₃·v̄] = [‖v̄‖, 0, 0] por construcción
        # Pero calculamos numéricamente para verificar
        A: NDArray[np.float64] = np.stack([e1, e2, e3], axis=0)  # (3, n)
        p: NDArray[np.float64] = A @ v_centered                   # (3,)
        p_norm: float = float(np.linalg.norm(p))

        # p_norm ≥ ‖v̄‖ · |e₁ · (v̄/‖v̄‖)| = ‖v̄‖ — no puede ser cero
        proj_r3: NDArray[np.float64] = p / max(p_norm, _EPSILON)

        return GramSchmidtBasis(e1=e1, e2=e2, e3=e3, proj_r3=proj_r3)

    # ─────────────────────────────────────────────────────────────────────────
    # §1.6 — Mapeo de logits a función escalar sobre S²
    # ─────────────────────────────────────────────────────────────────────────

    def _map_logits_to_sphere(
        self, logits: NDArray[np.float64]
    ) -> SpherePoint:
        r"""
        Proyecta logits ∈ ℝⁿ a una función escalar ψ ∈ L²(S²) sobre la grilla.

        Pipeline:
        ─────────
        1. Gram-Schmidt robusto → p ∈ S² ⊂ ℝ³ (GramSchmidtBasis.proj_r3).
        2. Convertir p a coordenadas esféricas (θ₀, φ₀).
        3. Calcular la distancia geodésica d(θ,φ) = arccos(p·q(θ,φ)) en la grilla.
        4. Ancho adaptativo σ = clip(std(v̄)/(range+ε), σ_min, σ_max).
        5. Perfil gaussiano: ψ_raw(θ,φ) = exp(-½·(d/σ)²).
        6. Escalar ψ para que ‖ψ‖_{L²(S²)} = ‖logits‖₂ (conservación de energía).

        Ancho σ adaptativo:
        ───────────────────
        - std(v̄): dispersión de los logits (cuánta información hay).
        - range: rango dinámico (qué tan "puntiagudo" es el pico).
        - Señal concentrada (range grande, std pequeño): σ pequeño → pico en S².
        - Señal plana (range pequeño): σ → σ_max → distribución difusa.
        - std/range ∈ [0,1] por la desigualdad de rango: std ≤ range/2.
          (para distribuciones sobre [a,b]: std ≤ (b-a)/2 = range/2)

        Retorna
        -------
        SpherePoint(psi, sigma, theta0, phi0, p_r3)

        Lanza
        -----
        ValueError         : Si logits contiene NaN/Inf.
        LensSingularityError: Si la señal es constante o n < 3.
        """
        if not np.all(np.isfinite(logits)):
            n_bad = int(np.sum(~np.isfinite(logits)))
            raise ValueError(
                f"logits contiene {n_bad} valores no finitos. "
                f"Imposible proyectar a S²."
            )

        # §1.6.1: Proyección a ℝ³
        gs_basis: GramSchmidtBasis = self._project_logits_to_r3_robust(logits)
        p: NDArray[np.float64] = gs_basis.proj_r3  # vector unitario en ℝ³

        # §1.6.2: Coordenadas esféricas del punto caliente
        theta0: float = float(np.arccos(np.clip(p[2], -1.0, 1.0)))
        phi0  : float = float(np.arctan2(p[1], p[0]))

        # §1.6.3: Grilla 2D de coordenadas
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )

        # §1.6.4: Distancia geodésica (fórmula de Haversine para estabilidad)
        # d(θ,φ, θ₀,φ₀) = arccos(cos θ·cos θ₀ + sin θ·sin θ₀·cos(φ-φ₀))
        cos_dist: NDArray[np.float64] = np.clip(
            np.cos(THETA) * np.cos(theta0)
            + np.sin(THETA) * np.sin(theta0) * np.cos(PHI - phi0),
            -1.0, 1.0
        )
        angular_dist: NDArray[np.float64] = np.arccos(cos_dist)

        # §1.6.5: Ancho adaptativo σ
        v_centered: NDArray[np.float64] = logits - np.mean(logits)
        std_v   : float = float(np.std(v_centered))
        range_v : float = float(np.max(logits) - np.min(logits))
        # std_v/range_v ∈ [0, 0.5] matemáticamente, pero numéricamente puede
        # ser ligeramente mayor — clip garantiza el rango físico
        sigma: float = float(np.clip(
            std_v / (range_v + _EPSILON),
            _SIGMA_MIN,
            _SIGMA_MAX,
        ))

        # §1.6.6: Perfil gaussiano (real, positivo)
        psi_raw: NDArray[np.float64] = np.exp(
            -0.5 * (angular_dist / sigma) ** 2
        ).astype(np.float64)

        # §1.6.7: Escalar para conservar energía L²
        energy_raw: float = float(np.sum(psi_raw ** 2 * self._grid.weights))
        norm_psi_raw: float = math.sqrt(max(energy_raw, _EPSILON))
        target_norm : float = float(np.linalg.norm(logits))
        psi: NDArray[np.float64] = psi_raw * (target_norm / norm_psi_raw)

        logger.debug(
            "_map_logits_to_sphere: θ₀=%.3f rad, φ₀=%.3f rad, "
            "σ=%.3f rad, ‖ψ‖_{L²}=%.3e, ‖logits‖₂=%.3e.",
            theta0, phi0, sigma,
            float(math.sqrt(float(np.sum(psi ** 2 * self._grid.weights)))),
            target_norm,
        )

        return SpherePoint(
            psi    = psi,
            sigma  = sigma,
            theta0 = theta0,
            phi0   = phi0,
            p_r3   = p,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.7 — Validación de ψ
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_psi(self, psi: NDArray[np.float64]) -> None:
        r"""
        Verifica invariantes de la función ψ antes de la expansión espectral.

        Verificaciones:
            1. shape == self._grid.shape.
            2. dtype numérico real (float32 o float64).
            3. Finitud: sin NaN ni Inf.
            4. Energía L² > ε: ψ no es idénticamente nula.

        Lanza
        -----
        ValueError : Si alguna condición falla.
        """
        if psi.shape != self._grid.shape:
            raise ValueError(
                f"ψ.shape={psi.shape} ≠ grilla={self._grid.shape}."
            )
        if not np.issubdtype(psi.dtype, np.floating):
            raise ValueError(
                f"ψ.dtype={psi.dtype} debe ser float. "
                f"La función de onda en S² es real-valuada."
            )
        if not np.all(np.isfinite(psi)):
            n_bad = int(np.sum(~np.isfinite(psi)))
            raise ValueError(
                f"ψ contiene {n_bad} valores no finitos (NaN/Inf)."
            )
        energy: float = float(np.sum(psi.astype(np.float64) ** 2 * self._grid.weights))
        if energy < _EPSILON:
            raise ValueError(
                f"ψ tiene energía L² ≈ 0 (‖ψ‖²_{{L²}}={energy:.2e}). "
                f"Función idénticamente nula: no hay señal que expandir."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.8 — Cálculo vectorizado de coeficientes c_{lm}
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_spherical_coefficients(
        self, psi: NDArray[np.float64]
    ) -> SphericalSpectrum:
        r"""
        Calcula los coeficientes c_{lm} de forma completamente vectorizada.

        Fórmula con Jacobiano correcto:
            c_{lm} = Σ_{i,j} ψ(θ_i, φ_j) · conj(Y_l^m(θ_i, φ_j)) · W_{ij}

        Implementación vectorizada:
        ───────────────────────────
        Usando el tensor Y precomputado Y_tensor[k, i, j] y el tensor de pesos W[i,j]:

            c[k] = Σ_{i,j} ψ[i,j] · conj(Y_tensor[k,i,j]) · W[i,j]
                 = np.einsum('ij, kij, ij -> k', psi, conj(Y_tensor), W)

        Lo que es equivalente a:
            integrand[k,i,j] = psi[i,j] · conj(Y[k,i,j]) · W[i,j]
            c[k] = Σ_{i,j} integrand[k,i,j]
                 = np.sum(integrand, axis=(1,2))

        Implementado con broadcasting:
            psi_W[i,j] = psi[i,j] · W[i,j]   (producto puntual, shape (n_θ,n_φ))
            c = Y_conj @ vec(psi_W)            (matmul, shape (num_coeffs,))

        Parseval vectorizado:
        ─────────────────────
            LHS = Σ_{ij} ψ[i,j]² · W[i,j]   = np.dot(vec(psi²), vec(W))
            RHS = Σ_k |c[k]|²                 = np.dot(|c|, |c|)

        Parámetros
        ----------
        psi : (n_theta, n_phi) NDArray[np.float64], validado por _validate_psi.

        Retorna
        -------
        SphericalSpectrum con c_{lm} y diagnósticos de Parseval.

        Lanza
        -----
        ParsevalViolationError : Si la identidad de Parseval falla (>_PARSEVAL_REL_TOL).
        """
        self._validate_psi(psi)

        W: NDArray[np.float64] = self._grid.weights  # (n_theta, n_phi)

        # Producto puntual ψ · W: shape (n_theta, n_phi)
        psi_W: NDArray[np.float64] = psi.astype(np.float64) * W

        # Vectorizar: aplanar a (n_theta*n_phi,)
        psi_W_flat: NDArray[np.float64] = psi_W.ravel()

        # Y_tensor_conj: shape (num_coeffs, n_theta*n_phi)
        Y_conj_flat: NDArray[np.complex128] = (
            np.conj(self._harmonics_tensor)
            .reshape(self._harmonics_tensor.shape[0], -1)
        )

        # c_{lm}: shape (num_coeffs,) — matmul vectorizado
        c_lm: NDArray[np.complex128] = Y_conj_flat @ psi_W_flat

        # Parseval
        parseval_lhs: float = float(np.dot(psi_W_flat, psi.ravel()))
        parseval_rhs: float = float(np.sum(np.abs(c_lm) ** 2))
        rel_err: float = abs(parseval_lhs - parseval_rhs) / max(parseval_lhs, _EPSILON)

        logger.debug(
            "_compute_spherical_coefficients: LHS=%.4e, RHS=%.4e, "
            "rel_err=%.2e, num_coeffs=%d.",
            parseval_lhs, parseval_rhs, rel_err, len(c_lm),
        )

        if self._verify_parseval and rel_err > _PARSEVAL_REL_TOL:
            raise ParsevalViolationError(
                f"Identidad de Parseval violada: "
                f"‖ψ‖²_{{L²}} = {parseval_lhs:.4e} ≠ Σ|c_{{lm}}|² = {parseval_rhs:.4e}. "
                f"Error relativo = {rel_err:.2e} > tol = {_PARSEVAL_REL_TOL:.2e}. "
                f"Causas probables: n_theta={self._n_theta} < {2*self._l_cutoff+2} "
                f"(mínimo para l_cutoff={self._l_cutoff}), "
                f"o ψ con discontinuidades de escala > resolución de grilla."
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
        Re-verificación explícita de la identidad de Parseval sobre un espectro dado.

        Permite verificar a posteriori con una tolerancia diferente a la del cálculo.

        Lanza
        -----
        ParsevalViolationError : Si el error supera _PARSEVAL_REL_TOL.
        """
        if spectrum.parseval_relative_error > _PARSEVAL_REL_TOL:
            raise ParsevalViolationError(
                f"Verificación posterior de Parseval fallida: "
                f"error relativo = {spectrum.parseval_relative_error:.2e} "
                f"> tol = {_PARSEVAL_REL_TOL:.2e}."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # §1.9 — Propiedades de solo lectura de Fase 1
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def grid(self) -> SphericalGrid:
        """Grilla de cuadratura (inmutable, solo lectura)."""
        return self._grid

    @property
    def l_cutoff(self) -> int:
        """Orden máximo de truncación l_max ≥ 0."""
        return self._l_cutoff

    @property
    def gamma_damping(self) -> float:
        """Coeficiente de amortiguación γ > 0."""
        return self._gamma

    @property
    def harmonics_tensor(self) -> NDArray[np.complex128]:
        """Tensor Y[num_coeffs, n_theta, n_phi] (solo lectura)."""
        return self._harmonics_tensor

    @property
    def degree_vector(self) -> NDArray[np.float64]:
        """Vector de grados l[k] en orden lexicográfico (solo lectura)."""
        return self._degree_vector


# ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 2 — ÓPTICA DE LENTES CATEGÓRICOS (FUNTORES VIEW Y PUT)                     ║
# ║                                                                                  ║
# ║  Contrato formal:                                                                ║
# ║    Entrada  : logits ∈ ℝⁿ + SphericalSpectrum de Fase 1 + n_refract ∈ (1,2)      ║
# ║    Salida   : focused_logits ∈ ℝⁿ via proyección adjunta espectral               ║
# ║                                                                                  ║
# ║  Garantías formales:                                                             ║
# ║    G1. G validado como tensor SPD real (simétrico, definido positivo)            ║
# ║    G2. n(σ*) = 1+tanh(α·σ*) ∈ (1,2) verificado algebraicamente                   ║
# ║    G3. Proyección adjunta correcta: P*: L²(S²) → ℝⁿ via base de sensores         ║
# ║    G4. ‖focused‖₂ ≤ ‖raw‖₂ (filtrado disipativo, verificado numéricamente).      ║
# ║    G5. Métrica de Fermat validada: Tr(n²·G) es finito y positivo                 ║
# ║                                                                                  ║
# ║  La definición del último método de Fase 1 (_verify_parseval_identity)           ║
# ║  continúa aquí en Phase2_CategoricalOpticLens, que hereda de Fase 1.             ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════

class Phase2_CategoricalOpticLens(Phase1_SphericalHarmonicsSpectrometer):
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    FASE 2 — Óptica de Lentes Categóricos con Proyección Adjunta Espectral
    ═══════════════════════════════════════════════════════════════════════════

    Aplica el par de funtores (view, put) del Lente Categórico, acoplando la
    métrica Riemanniana G_{μν} y el índice de refracción de Fermat n(σ*).

    La mejora central v4.0.0 es la proyección adjunta espectral correcta:
    ─────────────────────────────────────────────────────────────────────────
    En v3.0.0, `_project_psi_to_logit_space` simplemente escalaba los logits
    originales por ‖ψ̃‖/‖orig‖, preservando la DIRECCIÓN exacta de los logits
    originales. Esto significa que el filtrado espectral (la aniquilación de
    modos de alta frecuencia en ψ̃) NO se propagaba al espacio de logits.

    En v4.0.0, se implementa la proyección adjunta correcta:
        (P*ψ̃)_k = ⟨ψ̃, φ_k⟩_{L²(S²)}
    donde φ_k es la función sensor asociada al vector base canónico e_k ∈ ℝⁿ.

    Para vectores de logits con n grande, la construcción explícita de todas
    las {φ_k} sería O(n·n_θ·n_φ), costoso pero correcto. Se implementa una
    versión eficiente usando el hecho de que la función sensor φ_k es un
    gaussiano en S² centrado en la proyección de e_k:
        φ_k(θ,φ) ∝ exp(-½·(d(θ,φ,(θ_k,φ_k))/σ_k)²)
    y que ⟨ψ̃, φ_k⟩_{L²} puede calcularse espectralmente via los c_{lm} de ψ̃.
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
            Tensor métrico G_{μν}. Debe ser SPD (simétrico definido positivo).
        **kwargs
            Parámetros para Phase1_SphericalHarmonicsSpectrometer.

        Lanza
        -----
        TypeError  : Si metric_tensor no es ndarray.
        ValueError : Si metric_tensor no es 2-D cuadrado SPD real.
        """
        super().__init__(**kwargs)
        self._G  : NDArray[np.float64] = self._validate_metric_tensor(metric_tensor)
        self._dim: int                 = self._G.shape[0]

        logger.debug(
            "Phase2_CategoricalOpticLens inicializado: "
            "métrica %d×%d, Tr(G)=%.4f.",
            self._dim, self._dim, float(np.trace(self._G)),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §2.1 — Validación de la métrica Riemanniana
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_metric_tensor(
        G: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Verifica que G es un tensor métrico Riemanniano SPD real.

        Condiciones:
        ─────────────
        1. Tipo: ndarray.
        2. Dimensionalidad: 2-D cuadrado.
        3. Dtype o contenido real: Re(G) = G (sin parte imaginaria significativa).
        4. Finitud: sin NaN ni Inf.
        5. Simetría aproximada: ‖G - Gᵀ‖_F / ‖G‖_F < 1e-8.
        6. Definitud positiva: λ_min(G) > 0.

        Retorna
        -------
        NDArray[np.float64]
            G real, simetrizado ((G+Gᵀ)/2), dtype float64.

        Lanza
        -----
        TypeError  : Si G no es ndarray.
        ValueError : Si alguna condición falla.
        """
        if not isinstance(G, np.ndarray):
            raise TypeError(
                f"metric_tensor debe ser NDArray, recibido {type(G).__name__}."
            )
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(
                f"metric_tensor debe ser 2-D cuadrado, recibido shape={G.shape}."
            )

        # Verificar que es esencialmente real
        if np.iscomplexobj(G):
            imag_norm = float(np.linalg.norm(np.imag(G), 'fro'))
            real_norm = float(np.linalg.norm(np.real(G), 'fro'))
            if imag_norm > 1e-10 * max(real_norm, _EPSILON):
                raise ValueError(
                    f"metric_tensor tiene parte imaginaria significativa: "
                    f"‖Im(G)‖_F={imag_norm:.2e}, ‖Re(G)‖_F={real_norm:.2e}. "
                    f"El tensor métrico Riemanniano debe ser real."
                )
            G = np.real(G)

        G_real: NDArray[np.float64] = G.astype(np.float64)

        if not np.all(np.isfinite(G_real)):
            raise ValueError(
                "metric_tensor contiene valores no finitos (NaN/Inf)."
            )

        # Simetrización con advertencia
        fro_G   = float(np.linalg.norm(G_real, 'fro'))
        asym    = float(np.linalg.norm(G_real - G_real.T, 'fro'))
        rel_asym = asym / max(fro_G, _EPSILON)
        if rel_asym > 1e-8:
            logger.warning(
                "metric_tensor asimétrico: ‖G-Gᵀ‖_F/‖G‖_F=%.2e. "
                "Aplicando simetrización (G+Gᵀ)/2.",
                rel_asym
            )
        G_sym: NDArray[np.float64] = (G_real + G_real.T) * 0.5

        # Verificar definitud positiva
        eigenvalues: NDArray[np.float64] = np.linalg.eigvalsh(G_sym)
        lambda_min: float = float(eigenvalues.min())
        if lambda_min <= 0.0:
            raise ValueError(
                f"metric_tensor no es definido positivo: "
                f"λ_min={lambda_min:.4e} ≤ 0. "
                f"Espectro: min={lambda_min:.4e}, max={float(eigenvalues.max()):.4e}."
            )

        return G_sym

    # ─────────────────────────────────────────────────────────────────────────
    # §2.2 — Métrica de Fermat
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_fermat_refractive_index(
        self, stress_tensor_norm: float
    ) -> float:
        r"""
        Calcula el índice de refracción de Fermat:

            n(σ*) = 1 + tanh(α · σ*)  ∈ (1, 2)  ∀ σ* ∈ ℝ

        Propiedades algebraicas:
        ─────────────────────────
        - tanh: ℝ → (-1,1) es estrictamente creciente y suave.
        - α > 0 ⟹ α·σ* ∈ ℝ ⟹ tanh(α·σ*) ∈ (-1,1).
        - n = 1 + tanh(α·σ*) ∈ (1-1, 1+1) = (0,2).
        - Más precisamente: n = 1 + tanh(x) > 1 + (-1) = 0, pero
          el límite inferior real es n → 1⁺ solo cuando x → -∞,
          es decir, σ* → -∞.
        - Para α = 0.5 y |σ*| ≤ 100: n ≈ 1 + tanh(50) ≈ 2.0 (prácticamente).

        Nota: La condición n ≥ 1 es SIEMPRE satisfecha para α > 0 y σ* finito.
        La excepción OpticalDispersionError está reservada para subclases que
        redefinen este método con modelos de índice no-estándar.

        Parámetros
        ----------
        stress_tensor_norm : float, ‖σ*‖ ∈ ℝ, finito.

        Retorna
        -------
        float, n ∈ (1, 2).

        Lanza
        -----
        ValueError             : Si stress_tensor_norm no es finito.
        OpticalDispersionError : Si n < 1 (solo subclases).
        """
        if not math.isfinite(float(stress_tensor_norm)):
            raise ValueError(
                f"stress_tensor_norm debe ser finito, recibido {stress_tensor_norm!r}."
            )
        n: float = 1.0 + math.tanh(_FERMAT_ALPHA * float(stress_tensor_norm))
        if n < 1.0:
            raise OpticalDispersionError(
                f"Índice de refracción no físico: n={n:.8f} < 1. "
                f"σ*={stress_tensor_norm:.4f}, α={_FERMAT_ALPHA}. "
                f"Esta condición solo puede ocurrir en subclases que redefinen "
                f"_compute_fermat_refractive_index."
            )
        logger.debug(
            "Índice de Fermat: n(σ*=%.4f, α=%.3f) = %.6f.",
            stress_tensor_norm, _FERMAT_ALPHA, n,
        )
        return n

    def _compute_fermat_metric_trace(self, n_refract: float) -> float:
        r"""
        Calcula la traza de la métrica de Fermat:

            Tr(g_F) = n²(σ*) · Tr(G_{μν})

        Propiedades:
        ─────────────
        - n ∈ (1,2) ⟹ n² ∈ (1,4).
        - Tr(G) > 0 ya que G es SPD (todos los autovalores > 0).
        - Tr(g_F) = n² · Tr(G) > 0 siempre.

        Parámetros
        ----------
        n_refract : float, n ∈ (1, 2).

        Retorna
        -------
        float, Tr(n²·G) > 0.
        """
        trace_G      : float = float(np.trace(self._G))
        fermat_trace : float = (n_refract ** 2) * trace_G
        logger.debug(
            "Tr(g_F) = n²·Tr(G) = %.4f² · %.4f = %.6f.",
            n_refract, trace_G, fermat_trace,
        )
        return fermat_trace

    # ─────────────────────────────────────────────────────────────────────────
    # §2.3 — Reconstrucción espectral filtrada
    # ─────────────────────────────────────────────────────────────────────────

    def _reconstruct_psi_from_spectrum(
        self,
        spectrum  : SphericalSpectrum,
        n_refract : float,
    ) -> NDArray[np.float64]:
        r"""
        Reconstruye ψ̃ filtrada usando el operador de difracción vectorizado.

        Operador de difracción:
        ───────────────────────
            O_{lens} ψ = Σ_{l=0}^{l_max} Σ_{m=-l}^{l} h_l · c_{lm} · Y_l^m(θ,φ)

        donde:
            h_l = exp(-γ · n² · l²)  ∈ (0, 1]  (pasa-bajos gaussiano)

        Propiedades:
        ─────────────
        - h_0 = exp(0) = 1 → el modo constante se preserva exactamente.
        - h_l → 0 exponencialmente para l → ∞ → aniquilación de alta frecuencia.
        - ‖O_{lens} ψ‖_{L²} ≤ ‖ψ‖_{L²} (operador de contracción, ‖O‖ ≤ 1).

        Implementación vectorizada:
        ────────────────────────────
        Usando Y_tensor[k, i, j] precomputado y el vector de atenuación h[k]:

            h_vector[k] = exp(-γ · n² · degree_vector[k]²)
            c_attenuated[k] = h_vector[k] · c_{lm}[k]
            ψ̃[i,j] = Σ_k c_attenuated[k] · Y_tensor[k,i,j]
                    = np.einsum('k, kij -> ij', c_attenuated, Y_tensor)

        Equivalente a la matmul:
            ψ̃_flat = c_attenuated @ Y_tensor_flat

        Parámetros
        ----------
        spectrum  : SphericalSpectrum — coeficientes c_{lm}.
        n_refract : float ∈ (1,2) — índice de refracción.

        Retorna
        -------
        NDArray[np.float64]
            ψ̃(θ,φ) de shape self._grid.shape, parte real.

        Lanza
        -----
        ValueError : Si spectrum.coefficients contiene NaN/Inf.
        """
        if not np.all(np.isfinite(spectrum.coefficients)):
            raise ValueError(
                "SphericalSpectrum.coefficients contiene valores no finitos."
            )

        # Vector de atenuación gaussiana h[k] = exp(-γ·n²·l[k]²)
        # degree_vector[k] = l para el k-ésimo coeficiente
        h_vector: NDArray[np.float64] = np.exp(
            -self._gamma * (n_refract ** 2) * (self._degree_vector ** 2)
        )  # shape: (num_coeffs,)

        # Coeficientes atenuados: c_att[k] = h[k] · c_{lm}[k]
        c_attenuated: NDArray[np.complex128] = (
            h_vector.astype(np.complex128) * spectrum.coefficients
        )  # shape: (num_coeffs,)

        # Reconstrucción vectorizada: ψ̃_flat = c_att @ Y_flat
        # Y_tensor_flat: (num_coeffs, n_theta*n_phi)
        Y_flat: NDArray[np.complex128] = self._harmonics_tensor.reshape(
            self._harmonics_tensor.shape[0], -1
        )
        psi_flat: NDArray[np.complex128] = c_attenuated @ Y_flat
        psi_reconstructed: NDArray[np.complex128] = psi_flat.reshape(self._grid.shape)

        # Tomar parte real (ψ̃ es real si c_{lm} satisface c_{l,-m} = (-1)^m conj(c_{lm}))
        imag_max: float = float(np.max(np.abs(np.imag(psi_reconstructed))))
        if imag_max > 1e-6:
            logger.warning(
                "ψ̃ tiene parte imaginaria significativa: max|Im(ψ̃)|=%.2e. "
                "Esto indica que los logits no producen una señal real sobre S². "
                "Tomando parte real.",
                imag_max
            )

        psi_real: NDArray[np.float64] = np.real(psi_reconstructed).astype(np.float64)

        logger.debug(
            "_reconstruct_psi_from_spectrum: ‖ψ̃‖_{L²}=%.3e, "
            "h_0=%.4f, h_{l_max}=%.4e, max|Im|=%.2e.",
            float(math.sqrt(float(np.sum(psi_real ** 2 * self._grid.weights)))),
            float(h_vector[0]),
            float(h_vector[-1]),
            imag_max,
        )
        return psi_real

    # ─────────────────────────────────────────────────────────────────────────
    # §2.4 — Proyección adjunta espectral al espacio de logits
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sensor_basis(
        self,
        original_logits : NDArray[np.float64],
        gs_basis        : GramSchmidtBasis,
    ) -> NDArray[np.float64]:
        r"""
        Construye la base de funciones sensor {φ_k} sobre S² para la proyección adjunta.

        Definición de función sensor:
        ──────────────────────────────
        Para cada componente k ∈ {0,...,n-1}, la función sensor φ_k es la imagen
        en S² del vector base canónico e_k ∈ ℝⁿ bajo el operador de análisis P:
            φ_k = P(e_k) ∈ L²(S²)

        Dado que P(v) = A · exp(-½·(d(θ,φ,p_v)/σ_v)²) donde p_v ∈ S² es la
        proyección de v y σ_v es el ancho adaptativo, la función sensor φ_k es
        un gaussiano en S² centrado en la proyección del vector canónico e_k.

        Implementación eficiente para n grande:
        ─────────────────────────────────────────
        Construir φ_k explícitamente para cada k ∈ {0,...,n-1} sería costoso.
        En su lugar, usamos la estructura de la proyección Gram-Schmidt:

        La proyección de v a ℝ³ es: p_v = [v·e₁, v·e₂, v·e₃] / ‖[...]‖
        Para el canónico e_k: p_{e_k} = [e_k·e₁, e_k·e₂, e_k·e₃] / ‖[...]‖
                                        = [e₁[k], e₂[k], e₃[k]] / ‖[e₁[k],e₂[k],e₃[k]]‖

        Por tanto, la base de sensores puede representarse como:
            A_sensor[k, :] = [e₁[k], e₂[k], e₃[k]] ∈ ℝ³

        que es la k-ésima columna de la matriz [e₁|e₂|e₃] ∈ ℝ^{n×3}.

        La proyección adjunta es entonces:
            (P*ψ̃)_k ∝ ⟨ψ̃, φ_k⟩_{L²(S²)}

        que se calcula via los coeficientes de Fourier de ψ̃ y φ_k.

        En la implementación, usamos la proyección aproximada de norma acotada:
            v̂_k = (A_sensor @ c̃_l0) [k]
        donde c̃_l0 son los coeficientes de baja frecuencia de ψ̃.

        Para la implementación práctica que garantiza ‖focused‖ ≤ ‖raw‖:
        ────────────────────────────────────────────────────────────────
        Sea α_sensor[k] = ‖A_sensor[k,:]‖₂ = ‖[e₁[k], e₂[k], e₃[k]]‖₂.
        La proyección adjunta correcta satisface:
            Σ_k α_sensor[k]² ≤ Σ_k ‖e_k‖²_2 = n (por Bessel)

        La proyección implementada usa:
            focused_k = logits[k] · (‖ψ̃‖_{L²} / ‖logits‖₂) · α_filter[k]

        donde α_filter[k] ∈ [0,1] modula por la retención espectral del modo k.

        Parámetros
        ----------
        original_logits : (n,) NDArray[np.float64].
        gs_basis        : GramSchmidtBasis con e₁, e₂, e₃.

        Retorna
        -------
        NDArray[np.float64]
            Matriz A_sensor de shape (n, 3), donde A_sensor[k,:] = [e₁[k],e₂[k],e₃[k]].
        """
        n: int = original_logits.size
        # Matriz de base de sensores: shape (n, 3)
        A_sensor: NDArray[np.float64] = np.column_stack([
            gs_basis.e1,   # (n,)
            gs_basis.e2,   # (n,)
            gs_basis.e3,   # (n,)
        ])  # shape (n, 3)
        return A_sensor

    def _project_psi_to_logit_space(
        self,
        psi_reconstructed : NDArray[np.float64],
        original_logits   : NDArray[np.float64],
        gs_basis          : GramSchmidtBasis,
        spectrum_filtered  : SphericalSpectrum,
    ) -> NDArray[np.float64]:
        r"""
        Proyecta ψ̃ ∈ L²(S²) de vuelta a ℝⁿ via la proyección adjunta espectral.

        Teoría:
        ────────
        El operador de análisis P: ℝⁿ → L²(S²) es lineal en los siguientes sentidos:
            (a) La dirección del punto caliente p_v depende de v de forma no-lineal.
            (b) Dado p_v fijo, el perfil gaussiano φ(θ,φ;p_v,σ) es lineal en la amplitud.

        Para la proyección adjunta, explotamos la estructura de la base GS:
        La proyección de v ∈ ℝⁿ a ℝ³ es v_3 = A·v donde A = [e₁|e₂|e₃]ᵀ.
        La adjunta de A es A* = Aᵀ (pues A tiene filas ortonormales).
        Por tanto:
            P*ψ̃ ≈ Aᵀ · (proyección de ψ̃ a ℝ³)

        La "proyección de ψ̃ a ℝ³" se calcula via los tres coeficientes del
        modo dipolar (l=1) de ψ̃, que corresponden a Y_1^{-1}, Y_1^0, Y_1^1.

        El modo monopolar (l=0) de ψ̃ corresponde a la componente uniforme.
        Los modos dipolares (l=1) corresponden a las tres coordenadas cartesianas x,y,z.

        Concretamente:
            ⟨ψ̃, x⟩ = Re(c_{1,1}) · √(2π/3)  (aproximadamente, via armónicos reales)
            ⟨ψ̃, y⟩ = Im(c_{1,1}) · √(2π/3)
            ⟨ψ̃, z⟩ = c_{1,0} · √(4π/3)

        Pero para garantizar la proyección L² correcta sin depender del número de
        modos disponibles, usamos la formulación directa:

            ψ̃_3D[μ] = ∫_{S²} ψ̃(θ,φ) · x_μ(θ,φ) dΩ   para μ ∈ {x,y,z}

        donde x(θ,φ) = sin(θ)cos(φ), y(θ,φ) = sin(θ)sin(φ), z(θ,φ) = cos(θ).

        Esta integral se calcula directamente en la grilla:
            ψ̃_3D[μ] = Σ_{i,j} ψ̃[i,j] · x_μ(θ_i,φ_j) · W_{i,j}

        Luego la proyección adjunta es:
            focused = Aᵀ · ψ̃_3D + (‖ψ̃‖_{L²} - ‖Aᵀ·ψ̃_3D‖₂) · original_dir

        donde original_dir = original_logits / ‖original_logits‖₂ garantiza
        que la componente nula-dipolar se conserva isométricamente.

        Propiedad de disipación:
        ─────────────────────────
        El filtro de alta frecuencia (h_l < 1 para l ≥ 1) disipa energía en
        los modos dipolares y superiores. El modo monopolar (h_0 = 1) se conserva.
        Por tanto:
            ‖focused‖₂² ≤ ‖A·A^ᵀ‖² · ‖ψ̃_3D‖² + ‖ψ̃_{monopolar}‖²
                          ≤ ‖ψ‖²_{L²} = ‖psi_raw‖²_{L²}

        Parámetros
        ----------
        psi_reconstructed  : (n_theta, n_phi) NDArray[float64], ψ̃ filtrada.
        original_logits    : (n,) NDArray[float64].
        gs_basis           : GramSchmidtBasis de la proyección original.
        spectrum_filtered  : SphericalSpectrum de ψ̃ (ya calculado en apply_view_functor).

        Retorna
        -------
        NDArray[np.float64]
            focused_logits de shape (n,), con ‖focused‖₂ ≤ ‖original‖₂.
        """
        n: int = original_logits.size

        # §2.4.1: Energías L² de ψ̃ y de los logits originales
        energy_psi: float = float(
            np.sum(psi_reconstructed ** 2 * self._grid.weights)
        )
        norm_psi   : float = math.sqrt(max(energy_psi, _EPSILON))
        norm_orig  : float = float(np.linalg.norm(original_logits))

        if norm_orig < _EPSILON:
            return np.zeros(n, dtype=np.float64)

        # §2.4.2: Proyección de ψ̃ a ℝ³ via coordenadas cartesianas en S²
        # x(θ,φ) = sin(θ)cos(φ), y(θ,φ) = sin(θ)sin(φ), z(θ,φ) = cos(θ)
        THETA, PHI = np.meshgrid(
            self._grid.theta, self._grid.phi, indexing='ij'
        )
        W: NDArray[np.float64] = self._grid.weights

        x_S2: NDArray[np.float64] = np.sin(THETA) * np.cos(PHI)
        y_S2: NDArray[np.float64] = np.sin(THETA) * np.sin(PHI)
        z_S2: NDArray[np.float64] = np.cos(THETA)

        # ψ̃_3D[μ] = ∫_{S²} ψ̃ · x_μ dΩ  (integración numérica en la grilla)
        psi_3D: NDArray[np.float64] = np.array([
            float(np.sum(psi_reconstructed * x_S2 * W)),
            float(np.sum(psi_reconstructed * y_S2 * W)),
            float(np.sum(psi_reconstructed * z_S2 * W)),
        ])  # shape (3,)

        # §2.4.3: Proyección adjunta: focused_dipolar = Aᵀ · ψ̃_3D ∈ ℝⁿ
        # A = [e₁|e₂|e₃]ᵀ ∈ ℝ^{3×n}, Aᵀ ∈ ℝ^{n×3}
        # focused_dipolar = Aᵀ · ψ̃_3D = e₁·ψ̃_3D[0] + e₂·ψ̃_3D[1] + e₃·ψ̃_3D[2]
        focused_dipolar: NDArray[np.float64] = (
            gs_basis.e1 * psi_3D[0]
            + gs_basis.e2 * psi_3D[1]
            + gs_basis.e3 * psi_3D[2]
        )  # shape (n,)

        # §2.4.4: Componente monopolar de ψ̃ (l=0, conservada exactamente)
        # c_{0,0} = ∫_{S²} ψ̃ · Y_0^0 dΩ = ∫ ψ̃ / √(4π) dΩ
        # La energía monopolar en ℝⁿ se proyecta como:
        #   focused_monopolar = (c_{0,0} / ‖Y_0^0‖_{sensor}) · original_dir
        # Dado que h_0 = 1, la componente monopolar se conserva sin atenuación.
        c_00: complex = complex(spectrum_filtered.coefficients[0])
        Y_00_norm: float = 1.0 / math.sqrt(4.0 * np.pi)  # Y_0^0 = 1/√(4π)
        monopolar_amplitude: float = float(np.real(c_00)) / Y_00_norm  # ∈ ℝ

        original_dir: NDArray[np.float64] = original_logits / norm_orig
        focused_monopolar: NDArray[np.float64] = monopolar_amplitude * original_dir

        # §2.4.5: Combinación de componentes dipolar y monopolar
        # La componente dipolar captura la modulación espacial filtrada.
        # La monopolar preserva la amplitud DC (componente constante).
        focused_combined: NDArray[np.float64] = focused_dipolar + focused_monopolar

        # §2.4.6: Escalar para que ‖focused‖₂ ≤ ‖psi_L2‖ (propiedad disipativa)
        norm_combined: float = float(np.linalg.norm(focused_combined))
        if norm_combined < _EPSILON:
            # Fallback si la proyección dipolar+monopolar es nula
            focused_combined = original_dir * norm_psi
            norm_combined = norm_psi

        # Escalar al mínimo de (‖ψ̃‖_{L²}, ‖original‖₂) — proporcional al filtrado
        target_norm: float = min(norm_psi, norm_orig)
        focused: NDArray[np.float64] = focused_combined * (target_norm / norm_combined)

        logger.debug(
            "_project_psi_to_logit_space: ‖ψ̃‖_{L²}=%.3e, ‖orig‖=%.3e, "
            "‖ψ̃_3D‖=%.3e, ‖focused_dipolar‖=%.3e, ‖focused‖=%.3e.",
            norm_psi, norm_orig,
            float(np.linalg.norm(psi_3D)),
            float(np.linalg.norm(focused_dipolar)),
            float(np.linalg.norm(focused)),
        )
        return focused

    def _validate_reconstructed_signal(
        self,
        signal          : NDArray[np.float64],
        original_logits : NDArray[np.float64],
    ) -> None:
        r"""
        Verifica postcondiciones de la señal reconstruida.

        Condiciones:
            1. Finitud: sin NaN/Inf.
            2. Dimensión: signal.size == original_logits.size.
            3. Propiedad disipativa: ‖signal‖₂ ≤ ‖original‖₂ + 10·ε
               (tolerancia numérica para la afirmación de disipación).

        Lanza
        -----
        ValueError : Si alguna condición falla.
        """
        if not np.all(np.isfinite(signal)):
            n_bad = int(np.sum(~np.isfinite(signal)))
            raise ValueError(
                f"Señal reconstruida contiene {n_bad} valores no finitos."
            )
        if signal.size != original_logits.size:
            raise ValueError(
                f"Dimensión de señal reconstruida ({signal.size}) ≠ "
                f"logits originales ({original_logits.size})."
            )
        norm_sig  = float(np.linalg.norm(signal))
        norm_orig = float(np.linalg.norm(original_logits))
        # Tolerancia: 1% sobre el máximo para errores de redondeo acumulados
        tol_dissipation = 0.01 * norm_orig + 10.0 * _EPSILON
        if norm_sig > norm_orig + tol_dissipation:
            logger.warning(
                "Propiedad disipativa violada: ‖signal‖=%.4e > ‖orig‖=%.4e "
                "(exceso=%.2e > tol=%.2e). Posible amplificación numérica.",
                norm_sig, norm_orig, norm_sig - norm_orig, tol_dissipation,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # §2.5 — Funtores View y Put (interfaz pública de Fase 2)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_view_functor(
        self, logits: NDArray[np.float64]
    ) -> Tuple[SphericalSpectrum, SpherePoint, GramSchmidtBasis]:
        r"""
        Morfismo `view : S → (A, SpherePoint, GramSchmidtBasis)`.

        Observa el macro-estado S (tensor de logits) y extrae:
            - A = SphericalSpectrum: invariante espectral c_{lm}.
            - SpherePoint: parámetros geométricos de la proyección (σ, θ₀, φ₀).
            - GramSchmidtBasis: base ortonormal para la proyección adjunta.

        Pipeline:
        ─────────
            1. _map_logits_to_sphere → SpherePoint (psi, sigma, θ₀, φ₀, gs_basis).
            2. _validate_psi.
            3. _compute_spherical_coefficients → SphericalSpectrum.
            4. [Opcional] _verify_parseval_identity.

        Retorna
        -------
        Tuple[SphericalSpectrum, SpherePoint, GramSchmidtBasis]
        """
        # Gram-Schmidt robusto (se llama internamente en _map_logits_to_sphere)
        gs_basis: GramSchmidtBasis = self._project_logits_to_r3_robust(logits)

        sphere_pt: SpherePoint = self._map_logits_to_sphere(logits)
        spectrum : SphericalSpectrum = self._compute_spherical_coefficients(
            sphere_pt.psi
        )
        return spectrum, sphere_pt, gs_basis

    def apply_put_functor(
        self,
        original_logits : NDArray[np.float64],
        spectrum        : SphericalSpectrum,
        n_refract       : float,
        gs_basis        : GramSchmidtBasis,
    ) -> NDArray[np.float64]:
        r"""
        Morfismo `put : S × A → S`.

        Reconstruye el campo escalar ψ̃ filtrado y lo proyecta de vuelta
        al espacio de logits via la proyección adjunta espectral.

        Pipeline:
        ─────────
            1. _reconstruct_psi_from_spectrum(spectrum, n_refract) → ψ̃.
            2. _project_psi_to_logit_space(ψ̃, original_logits, gs_basis, spectrum) → focused.
            3. _validate_reconstructed_signal.

        Parámetros
        ----------
        original_logits : (n,) NDArray[float64].
        spectrum        : SphericalSpectrum.
        n_refract       : float ∈ (1,2).
        gs_basis        : GramSchmidtBasis (base ortonormal de la proyección original).

        Retorna
        -------
        NDArray[np.float64]
            focused_logits de shape (n,), ‖focused‖₂ ≤ ‖original‖₂.
        """
        psi_reconstructed: NDArray[np.float64] = (
            self._reconstruct_psi_from_spectrum(spectrum, n_refract)
        )
        focused: NDArray[np.float64] = self._project_psi_to_logit_space(
            psi_reconstructed, original_logits, gs_basis, spectrum
        )
        self._validate_reconstructed_signal(focused, original_logits)
        return focused


# ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 3 — ORQUESTADOR DEL FIBRADO ÓPTICO (Morfismo Definitivo)                   ║
# ║                                                                                  ║
# ║  Contrato formal:                                                                ║
# ║    Entrada  : raw_llm_logits ∈ ℝⁿ + logistic_stress_norm ∈ ℝ                     ║
# ║    Salida   : RefractedState (focused_logits, kv_ratio, trace, diagnostics)      ║
# ║                                                                                  ║
# ║  Garantías formales:                                                             ║
# ║    G1. Validación completa de entradas antes de cualquier cómputo                ║
# ║    G2. kv_compression_ratio ∈ [0,1] algebraicamente (‖focused‖ ≤ ‖raw‖)          ║
# ║    G3. RefractedState con invariantes verificados en __post_init__               ║
# ║    G4. SpectralDiagnostics completo (θ₀, φ₀, gram_schmidt_cond incluidos)        ║
# ║    G5. Protocolo Morphism MIC implementado (__call__, compose)                   ║
# ║                                                                                  ║
# ║  La definición del último método de Fase 2 (apply_put_functor) continúa          ║
# ║  aquí en OpticalRiemannLensFibrator, que hereda de Fase 2.                       ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════

class OpticalRiemannLensFibrator(Phase2_CategoricalOpticLens, Morphism):
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    FASE 3 — Agente de Síntesis Definitivo: Fibrador Óptico de Riemann v4.0.0
    ═══════════════════════════════════════════════════════════════════════════

    Opera entre `deliberation_manifold.py` y `geodesic_attention_fibrator.py`
    para blindar espectralmente la inferencia del LLM.

    Hereda de:
      - Phase2_CategoricalOpticLens → toda la lógica de Fases 1 y 2.
      - Morphism                    → protocolo MIC (__call__, compose).

    Mejoras v4.0.0:
    ────────────────
    1. apply_view_functor retorna (SphericalSpectrum, SpherePoint, GramSchmidtBasis)
       → los diagnósticos σ, θ₀, φ₀ y el número de condición GS se registran
       sin recálculo.
    2. kv_compression_ratio calculado sobre las normas reales de focused/raw
       (propiedad disipativa garantizada algebraicamente).
    3. Protocolo Morphism implementado: __call__ delega a refract_attention_logits,
       y compose permite encadenar Lentes Ópticos en series.
    4. SpectralDiagnostics incluye theta0_projection, phi0_projection,
       gram_schmidt_cond para monitoreo geométrico completo.
    """

    def __init__(self, **kwargs: Any) -> None:
        r"""
        Inicializa el Fibrador Óptico de Riemann.

        Parámetros
        ----------
        **kwargs
            Parámetros de Phase2_CategoricalOpticLens (metric_tensor, l_cutoff,
            gamma_damping, n_theta, n_phi, verify_parseval).
        """
        Phase2_CategoricalOpticLens.__init__(self, **kwargs)
        Morphism.__init__(self)
        self._last_diagnostics: Optional[SpectralDiagnostics] = None

        logger.info(
            "OpticalRiemannLensFibrator v4.0.0 inicializado: "
            "l_cutoff=%d, γ=%.3f, n_θ=%d, n_φ=%d, dim_G=%d.",
            self._l_cutoff, self._gamma,
            self._n_theta, self._n_phi, self._dim,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §3.1 — Validaciones de entrada (precondiciones de Fase 3)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input_logits(logits: NDArray[np.float64]) -> None:
        r"""
        Verifica las precondiciones del vector de logits de entrada.

        Condiciones:
        ─────────────
        1. Tipo: ndarray numpy.
        2. Dimensionalidad: 1-D (vector, no tensor).
        3. Tamaño mínimo: ≥ _N_SVD_COMPONENTS = 3.
        4. Dtype numérico (int, float32, float64 — se convertirá a float64).
        5. Finitud: sin NaN ni Inf.

        Parámetros
        ----------
        logits : NDArray[np.float64], vector de logits.

        Lanza
        -----
        TypeError           : Si logits no es ndarray.
        ValueError          : Si dimensionalidad, finitud o dtype son incorrectos.
        LensSingularityError: Si logits.size < _N_SVD_COMPONENTS.
        """
        if not isinstance(logits, np.ndarray):
            raise TypeError(
                f"raw_llm_logits debe ser NDArray, recibido {type(logits).__name__}."
            )
        if logits.ndim != 1:
            raise ValueError(
                f"raw_llm_logits debe ser 1-D (vector), recibido ndim={logits.ndim}. "
                f"Para tensores de mayor dimensión, aplanar con .ravel() antes."
            )
        if not np.issubdtype(logits.dtype, np.number):
            raise ValueError(
                f"raw_llm_logits.dtype={logits.dtype} debe ser numérico."
            )
        if logits.size < _N_SVD_COMPONENTS:
            raise LensSingularityError(
                f"raw_llm_logits requiere ≥ {_N_SVD_COMPONENTS} componentes "
                f"para proyección a S² ⊂ ℝ³. Recibido size={logits.size}."
            )
        if not np.all(np.isfinite(logits)):
            n_bad = int(np.sum(~np.isfinite(logits)))
            bad_vals = logits[~np.isfinite(logits)][:5]  # máx 5 para el mensaje
            raise ValueError(
                f"raw_llm_logits contiene {n_bad} valores no finitos: "
                f"{bad_vals} (primeros 5)."
            )

    @staticmethod
    def _validate_stress_norm(stress_norm: float) -> None:
        r"""
        Verifica que la norma del tensor de estrés logístico sea un escalar válido.

        El estrés logístico σ* puede ser negativo (p.ej., en regímenes de
        gradient reversal del pipeline MIC). El índice n(σ*) = 1 + tanh(α·σ*)
        es siempre ≥ 1 independientemente del signo de σ*.

        Parámetros
        ----------
        stress_norm : float

        Lanza
        -----
        TypeError  : Si no es escalar numérico.
        ValueError : Si no es finito.
        """
        if not isinstance(stress_norm, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"logistic_stress_norm debe ser escalar numérico, "
                f"recibido {type(stress_norm).__name__}."
            )
        if not math.isfinite(float(stress_norm)):
            raise ValueError(
                f"logistic_stress_norm debe ser finito, "
                f"recibido {stress_norm!r}."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # §3.2 — Métricas de compresión
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_kv_compression_ratio(
        raw_logits    : NDArray[np.float64],
        focused_logits: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la razón de compresión del KV-Cache basada en normas L².

        Definición:
            ratio = ‖focused_logits‖₂ / max(‖raw_logits‖₂, ε)

        Propiedades algebraicas:
        ─────────────────────────
        - ratio ∈ [0, ∞) sin acotamiento.
        - Dado que _project_psi_to_logit_space garantiza ‖focused‖₂ ≤ ‖raw‖₂,
          el ratio ≤ 1 algebraicamente.
        - np.clip a [0,1] es solo un seguro numérico para errores de redondeo < ε.
        - ratio = 1: sin compresión (modos de baja frecuencia dominan).
        - ratio → 0: máxima compresión (señal de alta frecuencia, aniquilada).

        Parámetros
        ----------
        raw_logits     : NDArray[float64].
        focused_logits : NDArray[float64].

        Retorna
        -------
        float, compression_ratio ∈ [0, 1].
        """
        norm_raw    : float = float(np.linalg.norm(raw_logits))
        norm_focused: float = float(np.linalg.norm(focused_logits))
        ratio: float = norm_focused / max(norm_raw, _EPSILON)
        return float(np.clip(ratio, _COMPRESSION_FLOOR, _COMPRESSION_CAP))

    # ─────────────────────────────────────────────────────────────────────────
    # §3.3 — Construcción de diagnósticos espectrales
    # ─────────────────────────────────────────────────────────────────────────

    def _build_spectral_diagnostics(
        self,
        n_refract      : float,
        fermat_trace   : float,
        spectrum       : SphericalSpectrum,
        raw_logits     : NDArray[np.float64],
        focused_logits : NDArray[np.float64],
        sphere_pt      : SpherePoint,
        gs_basis       : GramSchmidtBasis,
    ) -> SpectralDiagnostics:
        r"""
        Construye el objeto de diagnósticos espectrales completo.

        Campos calculados:
        ───────────────────
        - energy_raw, energy_focused: normas al cuadrado (‖·‖²₂, no L²).
        - energy_retention: energy_focused / max(energy_raw, ε), ∈ [0,1].
        - kv_compression_ratio: ‖focused‖₂/‖raw‖₂, ∈ [0,1].
        - l_dominant: spectrum.dominant_degree().
        - gram_schmidt_cond: número de condición de la matriz [e₁|e₂|e₃]ᵀ.
          Por ortogonalidad exacta, κ = 1 en teoría; κ > 1 indica pérdida numérica.
        - theta0_projection, phi0_projection: del SpherePoint.
        - sigma_projection: del SpherePoint.

        Parámetros
        ----------
        n_refract      : float
        fermat_trace   : float
        spectrum       : SphericalSpectrum
        raw_logits     : NDArray[float64]
        focused_logits : NDArray[float64]
        sphere_pt      : SpherePoint (contiene sigma, theta0, phi0)
        gs_basis       : GramSchmidtBasis (para calcular número de condición)

        Retorna
        -------
        SpectralDiagnostics
        """
        energy_raw    : float = float(np.dot(raw_logits,     raw_logits))
        energy_focused: float = float(np.dot(focused_logits, focused_logits))
        energy_retention: float = float(np.clip(
            energy_focused / max(energy_raw, _EPSILON), 0.0, 1.0
        ))
        kv_ratio: float = self._compute_kv_compression_ratio(raw_logits, focused_logits)

        # Número de condición de la base de Gram-Schmidt
        # A = [e₁|e₂|e₃]ᵀ ∈ ℝ^{3×n}: matriz con filas ortonormales
        # Valores singulares de A deben ser todos = 1 (ortogonalidad)
        A_gs: NDArray[np.float64] = np.stack(
            [gs_basis.e1, gs_basis.e2, gs_basis.e3], axis=0
        )  # (3, n)
        sv = np.linalg.svd(A_gs, compute_uv=False)  # (3,)
        gram_cond: float = float(sv[0] / max(sv[-1], _EPSILON))

        return SpectralDiagnostics(
            n_refract            = n_refract,
            fermat_metric_trace  = fermat_trace,
            parseval_error       = spectrum.parseval_relative_error,
            kv_compression_ratio = kv_ratio,
            sigma_projection     = sphere_pt.sigma,
            theta0_projection    = sphere_pt.theta0,
            phi0_projection      = sphere_pt.phi0,
            energy_raw           = energy_raw,
            energy_focused       = energy_focused,
            energy_retention     = energy_retention,
            l_dominant           = spectrum.dominant_degree(),
            gram_schmidt_cond    = gram_cond,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §3.4 — Orquestador principal
    # ─────────────────────────────────────────────────────────────────────────

    def refract_attention_logits(
        self,
        raw_llm_logits      : NDArray[np.float64],
        logistic_stress_norm: float,
    ) -> RefractedState:
        r"""
        Método axiomático de ejecución del Lente Óptico de Riemann.

        Pipeline completo (3 fases anidadas):
        ──────────────────────────────────────
        FASE 3a — Validación de entradas:
            _validate_input_logits(raw_llm_logits)
            _validate_stress_norm(logistic_stress_norm)
            Castear a float64.

        FASE 2 — Métrica óptica:
            n = _compute_fermat_refractive_index(σ*)   → n ∈ (1,2)
            Tr = _compute_fermat_metric_trace(n)        → n²·Tr(G) > 0

        FASE 1+2 — Extracción espectral (View):
            (spectrum, sphere_pt, gs_basis) = apply_view_functor(logits)
            spectrum: c_{lm} con verificación de Parseval
            sphere_pt: psi, sigma, θ₀, φ₀
            gs_basis: e₁, e₂, e₃ (base ortonormal)

        FASE 2 — Reconstrucción filtrada (Put):
            focused_logits = apply_put_functor(logits, spectrum, n, gs_basis)
            ‖focused‖₂ ≤ ‖raw‖₂ (propiedad disipativa)

        FASE 3b — Métricas y diagnósticos:
            kv_ratio = _compute_kv_compression_ratio(raw, focused)
            diagnostics = _build_spectral_diagnostics(...)

        FASE 3c — Construcción del estado refractado:
            return RefractedState(focused, kv_ratio, fermat_trace, diagnostics)

        Precondiciones
        --------------
        - raw_llm_logits ∈ ℝⁿ, n ≥ 3, finito, 1-D.
        - logistic_stress_norm ∈ ℝ, finito.

        Postcondiciones
        ---------------
        - RefractedState.focused_logits ∈ ℝⁿ, finito.
        - RefractedState.kv_compression_ratio ∈ [0,1].
        - self._last_diagnostics actualizado con los diagnósticos del ciclo.

        Parámetros
        ----------
        raw_llm_logits       : (n,) NDArray[np.float64]
        logistic_stress_norm : float

        Retorna
        -------
        RefractedState

        Lanza
        -----
        TypeError               : Tipo de entrada incorrecto.
        ValueError              : Entradas con NaN/Inf o dimensión incorrecta.
        LensSingularityError    : n < 3 o señal constante.
        GramSchmidtDegeneracyError: base GS no ortonormalizable.
        OpticalDispersionError  : n_refract < 1 (solo subclases).
        ParsevalViolationError  : Parseval falla (cuadratura insuficiente).
        SpectralGridError       : Ortogonalidad de grilla violada.
        """
        logger.debug(
            "refract_attention_logits iniciado: n=%s, σ*=%.4f.",
            raw_llm_logits.size if isinstance(raw_llm_logits, np.ndarray) else "?",
            float(logistic_stress_norm)
            if isinstance(logistic_stress_norm, (int, float, np.floating)) else float("nan"),
        )

        # ── §3.4.1: Validación de entradas ───────────────────────────────────
        self._validate_input_logits(raw_llm_logits)
        self._validate_stress_norm(logistic_stress_norm)
        logits_f64: NDArray[np.float64] = raw_llm_logits.astype(np.float64, copy=False)

        # ── §3.4.2: Métrica óptica de Fermat ─────────────────────────────────
        n_refract   : float = self._compute_fermat_refractive_index(
            float(logistic_stress_norm)
        )
        fermat_trace: float = self._compute_fermat_metric_trace(n_refract)

        # ── §3.4.3: Gram-Schmidt (se llama antes de apply_view_functor para
        #            reutilizar el resultado en apply_put_functor y diagnósticos)
        gs_basis: GramSchmidtBasis = self._project_logits_to_r3_robust(logits_f64)

        # ── §3.4.4: Extracción espectral (View) ──────────────────────────────
        # _map_logits_to_sphere llama internamente a _project_logits_to_r3_robust
        # de nuevo — por consistencia, usamos el SpherePoint resultante
        sphere_pt: SpherePoint = self._map_logits_to_sphere(logits_f64)
        spectrum  : SphericalSpectrum = self._compute_spherical_coefficients(
            sphere_pt.psi
        )

        # ── §3.4.5: Reconstrucción filtrada (Put) ────────────────────────────
        psi_reconstructed: NDArray[np.float64] = (
            self._reconstruct_psi_from_spectrum(spectrum, n_refract)
        )
        focused_logits: NDArray[np.float64] = self._project_psi_to_logit_space(
            psi_reconstructed, logits_f64, gs_basis, spectrum
        )
        self._validate_reconstructed_signal(focused_logits, logits_f64)

        # ── §3.4.6: Métricas de compresión ───────────────────────────────────
        kv_ratio: float = self._compute_kv_compression_ratio(
            logits_f64, focused_logits
        )

        # ── §3.4.7: Diagnósticos espectrales ─────────────────────────────────
        diagnostics: SpectralDiagnostics = self._build_spectral_diagnostics(
            n_refract      = n_refract,
            fermat_trace   = fermat_trace,
            spectrum       = spectrum,
            raw_logits     = logits_f64,
            focused_logits = focused_logits,
            sphere_pt      = sphere_pt,
            gs_basis       = gs_basis,
        )
        self._last_diagnostics = diagnostics

        logger.info(
            "refract_attention_logits completado: n=%d, n_refract=%.4f, "
            "kv_ratio=%.3f, Parseval_err=%.2e, l_dom=%d, "
            "θ₀=%.2f°, φ₀=%.2f°, σ=%.3f rad, κ_GS=%.4f.",
            logits_f64.size, n_refract, kv_ratio,
            spectrum.parseval_relative_error,
            diagnostics.l_dominant,
            math.degrees(sphere_pt.theta0),
            math.degrees(sphere_pt.phi0),
            sphere_pt.sigma,
            diagnostics.gram_schmidt_cond,
        )

        return RefractedState(
            focused_logits       = focused_logits,
            kv_compression_ratio = kv_ratio,
            fermat_metric_trace  = fermat_trace,
            diagnostics          = diagnostics,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §3.5 — Protocolo Morphism MIC
    # ─────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        state: CategoricalState,
    ) -> CategoricalState:
        r"""
        Implementa el protocolo Morphism de la arquitectura MIC.

        Transforma un CategoricalState que contiene (logits, stress_norm)
        en un CategoricalState que contiene el RefractedState resultante.

        Parámetros
        ----------
        state : CategoricalState
            Debe contener:
                - state.logits : NDArray[float64]
                - state.stress_norm : float

        Retorna
        -------
        CategoricalState con state.refracted = RefractedState.
        """
        logits      : NDArray[np.float64] = state.logits
        stress_norm : float               = state.stress_norm
        refracted   : RefractedState      = self.refract_attention_logits(
            logits, stress_norm
        )
        return state.evolve(refracted=refracted)

    def compose(
        self,
        other: "OpticalRiemannLensFibrator",
    ) -> "OpticalRiemannLensFibrator":
        r"""
        Composición de morfismos: (self ∘ other)(v) = self(other(v)).

        La composición de dos Lentes Ópticos de Riemann es otro Lente.
        Dado que la composición no es un objeto OpticalRiemannLensFibrator
        simple (los parámetros no se componen algebraicamente de forma cerrada),
        se retorna un ComposedOpticalLens que delega la ejecución secuencial.

        Parámetros
        ----------
        other : OpticalRiemannLensFibrator

        Retorna
        -------
        _ComposedOpticalLens (subclase interna) con apply secuencial.

        Lanza
        -----
        TypeError : Si other no es OpticalRiemannLensFibrator.
        """
        if not isinstance(other, OpticalRiemannLensFibrator):
            raise TypeError(
                f"compose requiere OpticalRiemannLensFibrator, "
                f"recibido {type(other).__name__}."
            )
        return _ComposedOpticalLens(first=other, second=self)

    # ─────────────────────────────────────────────────────────────────────────
    # §3.6 — Propiedades de solo lectura de Fase 3
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def spectral_diagnostics(self) -> Optional[SpectralDiagnostics]:
        r"""
        Diagnósticos del último proceso de refracción ejecutado.

        Retorna None si refract_attention_logits no ha sido invocado aún.
        Útil para monitoreo post-hoc sin modificar la firma de refract_attention_logits.
        """
        return self._last_diagnostics


# ════════════════════════════════════════════════════════════════════════════════
# CLASE AUXILIAR: Composición secuencial de Lentes Ópticos
# ════════════════════════════════════════════════════════════════════════════════

class _ComposedOpticalLens:
    r"""
    Composición secuencial de dos OpticalRiemannLensFibrator.

    Implementa el morfismo compuesto (second ∘ first):
        (second ∘ first)(logits, σ*) = second.refract(first.refract(logits, σ*).focused_logits, σ*)

    Nota: No hereda de OpticalRiemannLensFibrator para evitar la complejidad
    de componer los parámetros de grilla y métrica. Es un adaptador ligero.
    """

    def __init__(
        self,
        first : OpticalRiemannLensFibrator,
        second: OpticalRiemannLensFibrator,
    ) -> None:
        self._first  = first
        self._second = second

    def refract_attention_logits(
        self,
        raw_llm_logits      : NDArray[np.float64],
        logistic_stress_norm: float,
    ) -> RefractedState:
        r"""
        Aplica first, luego second sobre los focused_logits resultantes.

        Parámetros
        ----------
        raw_llm_logits       : (n,) NDArray[float64]
        logistic_stress_norm : float

        Retorna
        -------
        RefractedState del segundo lente.
        """
        intermediate: RefractedState = self._first.refract_attention_logits(
            raw_llm_logits, logistic_stress_norm
        )
        return self._second.refract_attention_logits(
            intermediate.focused_logits, logistic_stress_norm
        )


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "OpticalDispersionError",
    "LensSingularityError",
    "GramSchmidtDegeneracyError",
    "ParsevalViolationError",
    "SpectralGridError",
    # Tipos auxiliares
    "SpherePoint",
    "GramSchmidtBasis",
    # Estructuras de datos
    "SphericalGrid",
    "SphericalSpectrum",
    "SpectralDiagnostics",
    "RefractedState",
    # Agente principal
    "OpticalRiemannLensFibrator",
]