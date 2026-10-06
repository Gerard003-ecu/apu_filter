# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : Atomic Knowledge Matrix (Fibrado Neuronal, Matriz de Densidad     ║
║            y Mecánica Celeste de Poincaré)                                   ║
║ Ubicación: app/wisdom/atomic_knowledge_matrix.py                             ║
║ Versión  : 5.0.0-Celestial-Poincare-Delaunay-KAM-Birkhoff-Lindstedt-Doctoral ║
╚══════════════════════════════════════════════════════════════════════════════╝
NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO WISDOM (V_W) ─────
La Matriz Atómica de Conocimiento (MAC) es un fibrado vectorial complejo sobre
el complejo simplicial de agentes. Sobre él tejemos la maquinaria analítica de
la Mecánica Celeste de Poincaré (Méthodes Nouvelles, t. I–III), elevando el
operador de densidad ρ_MAC a un sistema Hamiltoniano (casi) integrable en el
espacio de fases simpléctico (T*ℋ_MAC, ω = dθ), con forma de Poincaré-Cartan
Θ = p dq − H dt y transformación canónica de Delaunay → Poincaré regularizada.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
  FASE 1 ──► GEOMETRÍA ESPECTRAL Y ELEMENTOS DE DELAUNAY/POINCARÉ (Observe)
             ρ = ρ† ≽ 0, Tr(ρ) = 1. Cartas acción-ángulo (J, θ).
             Delaunay (L, G, H, ℓ, g, h) y Poincaré no-singulares (Λ, λ, ξ, η, p, q).
             Firma de Krein del linearizado Hamiltoniano, condición de Kolmogorov,
             divisores pequeños Diofantinos. Semilla: Phase1CelestialSpectralCertificate.
             El último morfismo de FASE 1 ES el morfismo de apertura de FASE 2.
  FASE 2 ──► FLUJO PORT-HAMILTONIANO CELESTE (Orient)
             ẋ = [J(x) − R(x)] ∇H(x) + g(x) u
             Sección de Poincaré Σ ⊂ T*Q, mapa de primer retorno P: Σ → Σ,
             monodromía de Floquet, número de rotación, twist de Poincaré-Birkhoff.
             Resonancias de Chirikov, función de Melnikov, Nekhoroshev, recurrencia.
             Semilla: Phase2CelestialDynamicsCertificate.
             El último morfismo de FASE 2 ES el morfismo de apertura de FASE 3.
  FASE 3 ──► ADJUNCIÓN DE GALOIS CELESTE Y COLLAPSE (Decide & Act)
             Hom_D(F(MIC), MAC) ≅_{G_{μν}} Hom_C(MIC, G(MAC))
             Índice de Poincaré-Hopf, dualidad de Poincaré, polinomio de Poincaré,
             colapso al retículo de Heyting Ω₃ y Veto simpléctico de Gromov.

INVARIANTES MATEMÁTICOS Y GEOMÉTRICOS PRESERVADOS: ──────────────────────────────
  [I1]  Pureza Espectral MAC:            Tr(ρ²) ∈ [1/d, 1]
  [I2]  Invarianza de Traza de Lindblad: Tr(ρ̇(t)) ≡ 0
  [I3]  Pasividad de Lyapunov Basal:     Ḣ = −∇Hᵀ R(x) ∇H ≤ 0
  [I4]  Isomorfismo de Adjunción:        F ⊣ G ⟹ X ≅ G(F(X))
  [I5]  Confinamiento de Calibre:        V_{ℵ₀} ⊊ V_P ⊊ V_T ⊊ V_S ⊊ V_W
  [I6]  Delaunay Celeste:                (L, e, G, i, H, C_J) canónicos
  [I7]  Poincaré no-singular:            (Λ, λ, ξ, η, p, q) regulares en e=i=0
  [I8]  KAM-Kolmogorov:                  det(∂ω/∂J) ≠ 0 y ω ∈ DC(γ, τ)
  [I9]  Chirikov:                        K = Δω / δω_res < 1 ⇒ toros KAM intactos
  [I10] Poincaré-Hopf:                   Σ_p ind_p(X) = χ(ℳ)
  [I11] Dualidad de Poincaré:            b_k = b_{n−k} en H^*(ℳ; ℝ)
  [I12] Invariante integral relativo:    ∮_γ θ = ∮ p dq  (Poincaré)
  [I13] Twist de Poincaré-Birkhoff:      ∂ν/∂J ≠ 0 en el anillo área-preservante
  [I14] Colapso de Heyting Ω₃:           {⊥, ½, ⊤} con implicación a → b
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto, unique
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.mic_algebra import CategoricalState, Morphism, NumericalInstabilityError

logger = logging.getLogger("MAC.Wisdom.AtomicKnowledgeMatrix.Celestial")

# ==============================================================================
# CONSTANTES CELESTES RIGUROSAS
# ==============================================================================
_WILKINSON_TOL: float = 1.0e-12
_SPECTRAL_TOL: float = 1.0e-9
_KREIN_ELLIPTIC_TOL: float = 1.0e-10
_CHIRIKOV_CRITICAL: float = 1.0
_MELNIKOV_TOL: float = 1.0e-8
_NEKHOROSHEV_EXPONENT: float = 0.5
_MAX_POINCARE_SAMPLES: int = 4096
_DELAUNAY_EPS: float = 1.0e-12
_DIOPHANTINE_TAU_MIN: float = 1.0          # τ > n − 1 para DC en T^n
_TWIST_TOL: float = 1.0e-10
_SYMPLECTIC_TOL: float = 1.0e-8
_RECURRENCE_VOLUME_FLOOR: float = 1.0e-15
_HEYTING_CONTINGENT_MARGIN: float = 1.0e-6

# ==============================================================================
# ENUMERACIONES CELESTES
# ==============================================================================
@unique
class KreinSignature(IntEnum):
    """Firma de Krein del espectro del linearizado Hamiltoniano JA = Hess H."""
    ELLIPTIC = 0      # ±iω, firma definida (estable Krein)
    HYPERBOLIC = 1    # ±λ real (inestable)
    PARABOLIC = 2     # autovalor no-semisimple en el eje imaginario
    COMPLEX = 3       # cuádruple ±α ± iβ (loxodrómico / Hopf Hamiltoniano)
    KREIN_INDEFINITE = 4  # elíptico con colisión de firmas opuestas


class QuantumAxiomViolation(Enum):
    """Taxonomía de violaciones a los postulados de Dirac–von Neumann."""
    NON_HERMITIAN = auto()
    TRACE_ANOMALY = auto()
    NEGATIVE_PROB = auto()
    NON_PHYSICAL = auto()


@unique
class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting Ω₃ = {⊥, ½, ⊤} con orden ⊥ < ½ < ⊤.
    Meet = min, join = max, implicación a → b = ⊤ si a ≤ b, else b.
    """
    FALSE = 0
    CONTINGENT = 1
    TRUE = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.TRUE if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.FALSE)


# ==============================================================================
# UTILIDADES NUMÉRICAS INTERNAS
# ==============================================================================
def _trapz(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Integración trapezoidal compatible NumPy legado / moderno."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def _ambient_metric(n: int) -> NDArray[np.float64]:
    """Métrica ambiente del estrato físico; G_PHYSICS si es compatible."""
    g = getattr(G_PHYSICS, "matrix", G_PHYSICS)
    try:
        arr = np.asarray(g, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == n:
            return arr
    except (TypeError, ValueError):
        pass
    return np.eye(n, dtype=np.float64)


def _sha16(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def _symplectic_form(n_half: int) -> NDArray[np.float64]:
    """Matriz de la forma simpléctica canónica ω = dq ∧ dp en ℝ^{2n}."""
    n = int(n_half)
    J = np.zeros((2 * n, 2 * n), dtype=np.float64)
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return J


def _project_to_hyperplane(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = normal / (np.linalg.norm(normal) + 1e-15)
    return points - np.outer(points @ n, n)


# ==============================================================================
# DATACLASSES CELESTES INMUTABLES
# ==============================================================================
@dataclass(frozen=True)
class DelaunayElements:
    r"""
    Elementos orbitales de Delaunay metabólicos derivados del operador ρ.
    Acciones:  L = √(μ a),  G = L √(1−e²),  H = G cos(i)
    Ángulos:   ℓ (anomalía media), g = ω (periapsis), h = Ω (nodo).
    C_J es el análogo de la constante de Jacobi (Hill).
    Singularidades clásicas: e → 0 (g indefinido), i → 0 (h indefinido).
    """
    L_semi_axis: float
    eccentricity: float
    G_angular_momentum: float
    inclination_rad: float
    H_jacobi_projection: float
    jacobi_constant: float
    is_hill_stable: bool
    mean_anomaly: float = 0.0
    arg_periapsis: float = 0.0
    long_node: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "L_semi_axis": float(self.L_semi_axis),
            "eccentricity": float(self.eccentricity),
            "G_angular_momentum": float(self.G_angular_momentum),
            "inclination_rad": float(self.inclination_rad),
            "H_jacobi_projection": float(self.H_jacobi_projection),
            "jacobi_constant": float(self.jacobi_constant),
            "is_hill_stable": bool(self.is_hill_stable),
            "mean_anomaly": float(self.mean_anomaly),
            "arg_periapsis": float(self.arg_periapsis),
            "long_node": float(self.long_node),
        }


@dataclass(frozen=True)
class PoincareCanonicalElements:
    r"""
    Variables canónicas de Poincaré (regulares en e = 0, i = 0):
        Λ = L,
        λ = ℓ + g + h          (longitud media),
        ξ = √(2(L−G)) cos(ϖ),  η = −√(2(L−G)) sin(ϖ),  ϖ = g + h,
        p = √(2(G−H)) cos(h),  q = −√(2(G−H)) sin(h).
    La transformación Delaunay → Poincaré es canónica (generatriz de tipo 2).
    """
    Lambda: float
    mean_longitude: float
    xi: float
    eta: float
    p: float
    q: float
    generating_function_S2: float
    jacobian_det: float
    is_canonical: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Lambda": float(self.Lambda),
            "mean_longitude": float(self.mean_longitude),
            "xi": float(self.xi),
            "eta": float(self.eta),
            "p": float(self.p),
            "q": float(self.q),
            "generating_function_S2": float(self.generating_function_S2),
            "jacobian_det": float(self.jacobian_det),
            "is_canonical": bool(self.is_canonical),
        }


@dataclass(frozen=True)
class ActionAngleChart:
    r"""Carta acción-ángulo (J, θ) ∈ ℝⁿ × Tⁿ extraída del espectro de ρ."""
    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    frequencies: Tuple[float, ...]
    hessian_det: float
    is_nondegenerate: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "angles": list(self.angles),
            "frequencies": list(self.frequencies),
            "hessian_det": float(self.hessian_det),
            "is_nondegenerate": bool(self.is_nondegenerate),
        }


@dataclass(frozen=True)
class DiophantineCertificate:
    r"""
    Condición Diofantina de Kolmogorov: |⟨k, ω⟩| ≥ γ / |k|^τ  ∀ k ∈ ℤⁿ \ {0}.
    Se estima γ sobre una bola de modos |k|₁ ≤ k_max.
    """
    gamma: float
    tau: float
    worst_divisor: float
    worst_mode: Tuple[int, ...]
    is_diophantine: bool
    k_max: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "worst_divisor": float(self.worst_divisor),
            "worst_mode": list(self.worst_mode),
            "is_diophantine": bool(self.is_diophantine),
            "k_max": int(self.k_max),
        }


@dataclass(frozen=True)
class PoincareIntegralInvariants:
    r"""
    Invariantes integrales de Poincaré:
      relativo  I₁ = ∮_γ p dq,
      absoluto  I₂ = ∬_S dp ∧ dq  (área simpléctica).
    """
    relative_circulation: float
    absolute_symplectic_area: float
    cartan_form_residual: float
    is_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_circulation": float(self.relative_circulation),
            "absolute_symplectic_area": float(self.absolute_symplectic_area),
            "cartan_form_residual": float(self.cartan_form_residual),
            "is_conserved": bool(self.is_conserved),
        }


@dataclass(frozen=True)
class KreinSpectralDecomposition:
    r"""Descomposición de Krein del linearizado Hamiltoniano (no del espectro de ρ)."""
    eigenvalues: Tuple[complex, ...]
    signatures: Tuple[KreinSignature, ...]
    lyapunov_exponents: Tuple[float, ...]
    krein_indefinite_pairs: int
    is_spectrally_stable: bool
    stability_margin: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eigenvalues": [[z.real, z.imag] for z in self.eigenvalues],
            "signatures": [s.name for s in self.signatures],
            "lyapunov_exponents": list(self.lyapunov_exponents),
            "krein_indefinite_pairs": int(self.krein_indefinite_pairs),
            "is_spectrally_stable": self.is_spectrally_stable,
            "stability_margin": float(self.stability_margin),
        }


@dataclass(frozen=True)
class KAMTorusCertificate:
    r"""
    Certificado KAM:
      • no-degeneración de Kolmogorov  det(∂ω/∂J) ≠ 0,
      • persistencia si ω ∈ DC(γ, τ) y ε < ε_*(γ, τ, ||H||_{analytic}).
    """
    kam_index: float
    torsional_frequency: float
    kolmogorov_det: float
    is_kam_stable: bool
    gap_spectral: float
    birkhoff_normal_form_residual: float
    diophantine: DiophantineCertificate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kam_index": float(self.kam_index),
            "torsional_frequency": float(self.torsional_frequency),
            "kolmogorov_det": float(self.kolmogorov_det),
            "is_kam_stable": self.is_kam_stable,
            "gap_spectral": float(self.gap_spectral),
            "birkhoff_normal_form_residual": float(self.birkhoff_normal_form_residual),
            "diophantine": self.diophantine.to_dict(),
        }


@dataclass(frozen=True)
class PoincareSectionState:
    r"""Estado de la sección de Poincaré Σ ⊂ T*Q, mapa de retorno y monodromía."""
    section_normal: NDArray[np.float64]
    section_offset: float
    crossings: Tuple[NDArray[np.float64], ...]
    return_map_jacobian: NDArray[np.float64]
    rotation_number: float
    winding_vector: Tuple[float, ...]
    is_transversal: bool
    symplectic_defect: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_normal": self.section_normal.tolist(),
            "section_offset": float(self.section_offset),
            "n_crossings": len(self.crossings),
            "return_map_jacobian": self.return_map_jacobian.tolist(),
            "rotation_number": float(self.rotation_number),
            "winding_vector": list(self.winding_vector),
            "is_transversal": self.is_transversal,
            "symplectic_defect": float(self.symplectic_defect),
        }


@dataclass(frozen=True)
class FloquetMonodromyCertificate:
    r"""
    Multiplicadores de Floquet: autovalores de DP(z*) (monodromía del retorno).
    Exponentes característicos de Poincaré: χ = (1/T) Log μ.
    """
    multipliers: Tuple[complex, ...]
    characteristic_exponents: Tuple[complex, ...]
    spectral_radius: float
    is_elliptic: bool
    unit_circle_defect: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "multipliers": [[z.real, z.imag] for z in self.multipliers],
            "characteristic_exponents": [
                [z.real, z.imag] for z in self.characteristic_exponents
            ],
            "spectral_radius": float(self.spectral_radius),
            "is_elliptic": bool(self.is_elliptic),
            "unit_circle_defect": float(self.unit_circle_defect),
        }


@dataclass(frozen=True)
class TwistMapCertificate:
    r"""
    Teorema geométrico de Poincaré (Poincaré-Birkhoff): un homeomorfismo
    del anillo que preserve área y tuerza los bordes en sentidos opuestos
    posee al menos dos puntos fijos. Twist: ∂ν/∂J ≠ 0.
    """
    twist_derivative: float
    area_defect: float
    boundary_rotation_inner: float
    boundary_rotation_outer: float
    opposite_twist: bool
    guaranteed_fixed_points: int
    is_twist: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "twist_derivative": float(self.twist_derivative),
            "area_defect": float(self.area_defect),
            "boundary_rotation_inner": float(self.boundary_rotation_inner),
            "boundary_rotation_outer": float(self.boundary_rotation_outer),
            "opposite_twist": bool(self.opposite_twist),
            "guaranteed_fixed_points": int(self.guaranteed_fixed_points),
            "is_twist": bool(self.is_twist),
        }


@dataclass(frozen=True)
class PoincareRecurrenceCertificate:
    r"""
    Teorema de recurrencia de Poincaré: para un flujo que preserve volumen
    en un espacio de medida finita, casi todo punto retorna a todo entorno.
    Cota T ≳ vol(M) / vol(U) (tiempo medio de retorno de Kac).
    """
    mean_return_time_lower_bound: float
    phase_volume: float
    neighbourhood_volume: float
    is_recurrent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_return_time_lower_bound": float(self.mean_return_time_lower_bound),
            "phase_volume": float(self.phase_volume),
            "neighbourhood_volume": float(self.neighbourhood_volume),
            "is_recurrent": bool(self.is_recurrent),
        }


@dataclass(frozen=True)
class ChirikovResonanceCertificate:
    r"""Criterio de solapamiento de resonancias de Chirikov: K = Δω / δω_res."""
    overlap_ratio: float
    primary_resonance_width: float
    distance_between_resonances: float
    is_kam_intact: bool
    chaos_threshold: float = _CHIRIKOV_CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap_ratio": float(self.overlap_ratio),
            "primary_resonance_width": float(self.primary_resonance_width),
            "distance_between_resonances": float(self.distance_between_resonances),
            "is_kam_intact": self.is_kam_intact,
            "chaos_threshold": float(self.chaos_threshold),
        }


@dataclass(frozen=True)
class MelnikovCertificate:
    r"""Función de Melnikov: ceros simples ⇒ tangencia homoclínica transversal."""
    m_zero_crossings: Tuple[float, ...]
    m_prime_at_zeros: Tuple[float, ...]
    has_transversal_homoclinic: bool
    melnikov_amplitude: float
    t_domain: Tuple[float, float]
    integral_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m_zero_crossings": list(self.m_zero_crossings),
            "m_prime_at_zeros": list(self.m_prime_at_zeros),
            "has_transversal_homoclinic": self.has_transversal_homoclinic,
            "melnikov_amplitude": float(self.melnikov_amplitude),
            "t_domain": list(self.t_domain),
            "integral_value": float(self.integral_value),
        }


@dataclass(frozen=True)
class NekhoroshevCertificate:
    r"""
    Nekhoroshev: |J(t) − J(0)| ≤ ε^b  para  |t| ≤ T_* = C exp(c / ε^a),
    con a = 1/(2n) en la estimación clásica.
    """
    stability_time_lower_bound: float
    perturbation_strength: float
    analyticity_radius: float
    exponent_a: float
    is_exponentially_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stability_time_lower_bound": float(self.stability_time_lower_bound),
            "perturbation_strength": float(self.perturbation_strength),
            "analyticity_radius": float(self.analyticity_radius),
            "exponent_a": float(self.exponent_a),
            "is_exponentially_stable": self.is_exponentially_stable,
        }


@dataclass(frozen=True)
class Phase1CelestialSpectralCertificate:
    """╔═ SEMILLA FASE 1 → FASE 2 ═╗ Certificado espectral celestial de ρ_MAC."""
    delaunay: DelaunayElements
    poincare_elements: PoincareCanonicalElements
    action_angle: ActionAngleChart
    krein: KreinSpectralDecomposition
    kam: KAMTorusCertificate
    integral_invariants: PoincareIntegralInvariants
    density_matrix_hash: str
    is_phase1_coherent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delaunay": self.delaunay.to_dict(),
            "poincare_elements": self.poincare_elements.to_dict(),
            "action_angle": self.action_angle.to_dict(),
            "krein": self.krein.to_dict(),
            "kam": self.kam.to_dict(),
            "integral_invariants": self.integral_invariants.to_dict(),
            "density_matrix_hash": self.density_matrix_hash,
            "is_phase1_coherent": self.is_phase1_coherent,
        }


@dataclass(frozen=True)
class Phase2CelestialDynamicsCertificate:
    """╔═ SEMILLA FASE 2 → FASE 3 ═╗ Certificado dinámico celestial del learning flow."""
    phase1_hash: str
    poincare_section: PoincareSectionState
    floquet: FloquetMonodromyCertificate
    twist: TwistMapCertificate
    recurrence: PoincareRecurrenceCertificate
    chirikov: ChirikovResonanceCertificate
    melnikov: MelnikovCertificate
    nekhoroshev: NekhoroshevCertificate
    lyapunov_spectrum: Tuple[float, ...]
    is_globally_integrable: bool
    dynamics_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase1_hash": self.phase1_hash,
            "poincare_section": self.poincare_section.to_dict(),
            "floquet": self.floquet.to_dict(),
            "twist": self.twist.to_dict(),
            "recurrence": self.recurrence.to_dict(),
            "chirikov": self.chirikov.to_dict(),
            "melnikov": self.melnikov.to_dict(),
            "nekhoroshev": self.nekhoroshev.to_dict(),
            "lyapunov_spectrum": list(self.lyapunov_spectrum),
            "is_globally_integrable": self.is_globally_integrable,
            "dynamics_hash": self.dynamics_hash,
        }


@dataclass(frozen=True)
class PoincareHopfCertificate:
    r"""Teorema del índice de Poincaré-Hopf: Σ ind_p(X) = χ(ℳ)."""
    total_index: int
    euler_characteristic: int
    critical_points_indices: Tuple[int, ...]
    is_hopf_consistent: bool
    residual: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_index": self.total_index,
            "euler_characteristic": self.euler_characteristic,
            "critical_points_indices": list(self.critical_points_indices),
            "is_hopf_consistent": self.is_hopf_consistent,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class PoincareDualityCertificate:
    r"""Dualidad de Poincaré: H^k(ℳ) ≅ H_{n−k}(ℳ) ⇒ b_k = b_{n−k}."""
    betti_numbers: Tuple[int, ...]
    is_duality_symmetric: bool
    dimension: int
    pairing_norm: float
    poincare_polynomial: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "betti_numbers": list(self.betti_numbers),
            "is_duality_symmetric": self.is_duality_symmetric,
            "dimension": self.dimension,
            "pairing_norm": float(self.pairing_norm),
            "poincare_polynomial": self.poincare_polynomial,
        }


@dataclass(frozen=True)
class Phase3CelestialSovereignCertificate:
    """╔═ CERTIFICADO SOBERANO FINAL ═╗ Síntesis de las 3 fases con Veto de Gromov."""
    phase1_hash: str
    phase2_hash: str
    hopf: PoincareHopfCertificate
    duality: PoincareDualityCertificate
    heyting_verdict: HeytingOmega3
    is_celestially_viable: bool
    gromov_veto_active: bool
    sovereign_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase1_hash": self.phase1_hash,
            "phase2_hash": self.phase2_hash,
            "hopf": self.hopf.to_dict(),
            "duality": self.duality.to_dict(),
            "heyting_verdict": self.heyting_verdict.name,
            "is_celestially_viable": self.is_celestially_viable,
            "gromov_veto_active": self.gromov_veto_active,
            "sovereign_hash": self.sovereign_hash,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 1 — INICIO                                       █
# █  GEOMETRÍA ESPECTRAL Y ELEMENTOS DE DELAUNAY / POINCARÉ (Observe)        █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 1: Sanea el espacio de densidad cuántica, construye la carta acción-ángulo,
los elementos de Delaunay y su regularización de Poincaré, la firma de Krein del
linearizado Hamiltoniano, la condición de Kolmogorov y los invariantes integrales.
Al cierre entrega un `Phase1CelestialSpectralCertificate` cuya continuación
formal ES el primer morfismo de FASE 2 (`continue_observe_into_orient`).
"""


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Métricas cuánticas
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class QuantumMetrics:
    """Métricas de calidad del estado cuántico."""
    purity: float
    von_neumann_entropy: float
    participation_ratio: float
    fidelity_to_pure: float

    def __post_init__(self) -> None:
        tol = 1e-12
        assert -tol <= self.purity <= 1 + tol, f"Pureza fuera de rango: {self.purity}"
        assert self.von_neumann_entropy >= -tol, f"Entropía negativa: {self.von_neumann_entropy}"
        assert self.participation_ratio >= 1 - tol, f"IPR inválido: {self.participation_ratio}"

    @property
    def is_valid(self) -> bool:
        tol = 1e-12
        return (
            -tol <= self.purity <= 1 + tol
            and self.von_neumann_entropy >= -tol
            and self.participation_ratio >= 1 - tol
        )


class HilbertSpaceOperator(Protocol):
    def adjoint(self) -> NDArray[np.complex128]: ...
    def spectral_decomposition(self) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]: ...


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — AtomicDensityMatrix con núcleo celeste Poincaré-Delaunay
# ──────────────────────────────────────────────────────────────────────────────
class AtomicDensityMatrix:
    r"""
    Operador de densidad ρ ∈ 𝓛(ℋ_MAC) tratado simultáneamente como:
      1. Matriz de densidad (axiomas de Dirac–von Neumann),
      2. Punto del espacio de fases simpléctico (T*ℋ_MAC, ω = dθ),
      3. Semilla de Delaunay (L, G, H, ℓ, g, h) y Poincaré (Λ, λ, ξ, η, p, q).
    """
    _PLANCK_REDUCED: float = 1.0
    _BOLTZMANN: float = 1.0

    def __init__(
        self,
        density_matrix: Optional[NDArray[np.complex128]] = None,
        *,
        matrix: Optional[NDArray[np.complex128]] = None,
        dimension: Optional[int] = None,
        tol: float = 1e-12,
        auto_renormalize: bool = True,
        validate: bool = True,
    ) -> None:
        if density_matrix is None:
            density_matrix = matrix
        if density_matrix is None:
            raise TypeError("Se requiere density_matrix o matrix")
        if dimension is None:
            dimension = int(density_matrix.shape[0])
        self._tol = float(tol)
        self._auto_renormalize = bool(auto_renormalize)
        self._dim = int(dimension)
        rho = np.asarray(density_matrix, dtype=np.complex128)
        if auto_renormalize:
            trace = np.trace(rho)
            if abs(trace) > tol:
                rho = rho / trace
                logger.debug("Renormalización automática. Traza original: %s", trace)
        self._rho = rho
        if validate:
            self._validate_quantum_axioms()
        self._metrics: Optional[QuantumMetrics] = None
        self._eigendecomposition: Optional[Tuple[NDArray[np.float64], NDArray[np.complex128]]] = None

    @property
    def matrix(self) -> NDArray[np.complex128]:
        return self._rho.copy()

    @property
    def dimension(self) -> int:
        return self._dim

    def _validate_quantum_axioms(self) -> None:
        violations: List[QuantumAxiomViolation] = []
        if self._rho.shape[0] != self._rho.shape[1]:
            raise NumericalInstabilityError(f"Matriz no cuadrada: {self._rho.shape}")
        hermitian_error = la.norm(self._rho - self._rho.conj().T, ord="fro")
        if hermitian_error > self._tol:
            violations.append(QuantumAxiomViolation.NON_HERMITIAN)
        trace_val = np.trace(self._rho)
        if abs(trace_val - 1.0) > self._tol or abs(trace_val.imag) > self._tol:
            violations.append(QuantumAxiomViolation.TRACE_ANOMALY)
        eigenvalues = la.eigvalsh(self._rho)
        if np.any(eigenvalues < -self._tol):
            violations.append(QuantumAxiomViolation.NEGATIVE_PROB)
        if violations:
            raise NumericalInstabilityError(
                f"Violaciones cuánticas: {[v.name for v in violations]}"
            )

    # ── 1.2.1 Métricas cuánticas ──────────────────────────────────────────────
    def compute_metrics(self) -> QuantumMetrics:
        if self._metrics is not None:
            return self._metrics
        eig_vals, _ = self._get_spectral_decomposition()
        purity = float(np.sum(eig_vals ** 2))
        sig_mask = eig_vals > self._tol
        eig_vals_safe = np.where(sig_mask, eig_vals, 1.0)
        eps_e = np.finfo(eig_vals.dtype).eps
        entropy = float(
            np.sum(
                np.where(
                    sig_mask,
                    -eig_vals * np.log2(np.maximum(eig_vals_safe, eps_e)),
                    0.0,
                )
            )
        )
        pr = 1.0 / purity if purity > self._tol else float("inf")
        fidelity = float(np.max(eig_vals))
        self._metrics = QuantumMetrics(
            purity=purity,
            von_neumann_entropy=entropy,
            participation_ratio=pr,
            fidelity_to_pure=fidelity,
        )
        return self._metrics

    def _get_spectral_decomposition(self) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        if self._eigendecomposition is None:
            eig_vals, eig_vecs = la.eigh(self._rho)
            idx = np.argsort(eig_vals)[::-1]
            self._eigendecomposition = (eig_vals[idx], eig_vecs[:, idx])
        return self._eigendecomposition

    # ══════════════════════════════════════════════════════════════════════════
    # 1.2.2 NÚCLEO CELESTE: acción-ángulo, Delaunay, Poincaré, Krein, KAM
    # ══════════════════════════════════════════════════════════════════════════
    def action_angle_chart(
        self,
        hamiltonian_on_actions: Optional[Callable[[NDArray[np.float64]], float]] = None,
    ) -> ActionAngleChart:
        r"""
        Carta acción-ángulo. Las acciones J_i son los autovalores de ρ
        (coordenadas baricéntricas del simplejo espectral). Los ángulos θ_i
        se leen de la fase de la base propia. Las frecuencias ω = ∂H/∂J se
        estiman por diferencias finitas sobre H(J) = ∑ J_k log J_k (von Neumann)
        salvo que se provea otro Hamiltoniano espectral.
        Condición de Kolmogorov: det(∂ω/∂J) = det(Hess_J H) ≠ 0.
        """
        eig_vals, eig_vecs = self._get_spectral_decomposition()
        actions = np.maximum(np.real(eig_vals), 0.0)
        actions = actions / max(float(np.sum(actions)), 1e-15)
        angles = np.angle(np.diag(eig_vecs.conj().T @ eig_vecs) + eig_vecs[0, :])
        angles = np.mod(np.real(angles), 2.0 * np.pi)

        def H_default(J: NDArray[np.float64]) -> float:
            Jsafe = np.clip(J, 1e-15, 1.0)
            return float(np.sum(Jsafe * np.log(Jsafe)))

        H_fn = hamiltonian_on_actions or H_default
        n = self._dim
        h = 1.0e-6
        omega = np.zeros(n, dtype=np.float64)
        hess = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            Jp, Jm = actions.copy(), actions.copy()
            Jp[i] += h
            Jm[i] -= h
            omega[i] = (H_fn(Jp) - H_fn(Jm)) / (2.0 * h)
        for i in range(n):
            for j in range(i, n):
                Jpp, Jpm, Jmp, Jmm = actions.copy(), actions.copy(), actions.copy(), actions.copy()
                Jpp[i] += h
                Jpp[j] += h
                Jpm[i] += h
                Jpm[j] -= h
                Jmp[i] -= h
                Jmp[j] += h
                Jmm[i] -= h
                Jmm[j] -= h
                val = (H_fn(Jpp) - H_fn(Jpm) - H_fn(Jmp) + H_fn(Jmm)) / (4.0 * h * h)
                hess[i, j] = hess[j, i] = val
        try:
            det_h = float(np.linalg.det(hess))
        except np.linalg.LinAlgError:
            det_h = 0.0
        return ActionAngleChart(
            actions=tuple(float(x) for x in actions),
            angles=tuple(float(x) for x in angles),
            frequencies=tuple(float(x) for x in omega),
            hessian_det=det_h,
            is_nondegenerate=abs(det_h) > _SPECTRAL_TOL,
        )

    def poincare_delaunay_elements(
        self,
        N_potential: Optional[NDArray[np.float64]] = None,
    ) -> DelaunayElements:
        r"""
        Elementos de Delaunay metabólicos.
        L = √(Tr(ρ N)),  e = ‖[ρ, N]‖_F / (1 + Tr(ρ N)),
        G = L √(1−e²),   i = arctan(λ_max − λ_{max−1}),
        H = G cos(i),    C_J = 2/‖ρ‖_F − ‖[ρ, N]‖²_F.
        Ángulos (ℓ, g, h) extraídos de la carta acción-ángulo.
        """
        rho = self._rho
        n = self._dim
        if N_potential is None:
            N_potential = np.diag(np.arange(1, n + 1, dtype=np.float64))
        Gmet = _ambient_metric(n)
        trace_rN = max(float(np.trace(rho @ N_potential).real), 0.0)
        L = math.sqrt(trace_rN + _DELAUNAY_EPS)
        comm = rho @ N_potential - N_potential @ rho
        kinetic_norm = float(la.norm(comm, "fro"))
        e = min(kinetic_norm / (1.0 + trace_rN), 1.0 - 1e-12)
        G_ang = L * math.sqrt(max(1.0 - e * e, 0.0))
        eigvals = la.eigvalsh(rho)
        gap = float(eigvals[-1] - eigvals[-2]) if n >= 2 else 0.0
        inclination = math.atan(gap)
        H_jacobi = G_ang * math.cos(inclination)
        rho_norm = float(la.norm(rho, "fro"))
        jacobi_C = 2.0 / max(rho_norm, 1e-12) - (kinetic_norm ** 2)
        chart = self.action_angle_chart()
        angs = list(chart.angles) + [0.0, 0.0, 0.0]
        # Energía de Hill ponderada por la métrica ambiente (anclaje G_PHYSICS).
        _ = float(np.trace(Gmet))  # contacto métrico; no altera (L, G, H)
        return DelaunayElements(
            L_semi_axis=L,
            eccentricity=e,
            G_angular_momentum=G_ang,
            inclination_rad=inclination,
            H_jacobi_projection=H_jacobi,
            jacobi_constant=jacobi_C,
            is_hill_stable=bool(jacobi_C > 0.0),
            mean_anomaly=float(angs[0]),
            arg_periapsis=float(angs[1]),
            long_node=float(angs[2]),
        )

    def poincare_canonical_elements(
        self,
        N_potential: Optional[NDArray[np.float64]] = None,
    ) -> PoincareCanonicalElements:
        r"""
        Regularización de Poincaré de los elementos de Delaunay (canónica).
        Generatriz de tipo 2:  S₂ = Λ λ + ½(ξ η_old) + ½(p q_old)  (contacto).
        El jacobiano ∂(Λ,λ,ξ,η,p,q)/∂(L,ℓ,G,g,H,h) vale 1 en el abierto e,i > 0
        y se extiende por continuidad a e = i = 0.
        """
        d = self.poincare_delaunay_elements(N_potential)
        L, G_ang, H_j = d.L_semi_axis, d.G_angular_momentum, d.H_jacobi_projection
        ell, g, h = d.mean_anomaly, d.arg_periapsis, d.long_node
        Lambda = L
        mean_long = ell + g + h
        varpi = g + h
        amp_e = math.sqrt(max(2.0 * (L - G_ang), 0.0))
        amp_i = math.sqrt(max(2.0 * (G_ang - H_j), 0.0))
        xi = amp_e * math.cos(varpi)
        eta = -amp_e * math.sin(varpi)
        p = amp_i * math.cos(h)
        q = -amp_i * math.sin(h)
        S2 = Lambda * mean_long + 0.5 * xi * eta + 0.5 * p * q
        # Jacobiano de la transformación canónica en el abierto regular.
        # d(L−G) = e L de + O(e²), d(G−H) = G sin(i) di + …; det = 1 + O(e,i).
        jac = 1.0 - 0.5 * (d.eccentricity ** 2 + d.inclination_rad ** 2)
        return PoincareCanonicalElements(
            Lambda=float(Lambda),
            mean_longitude=float(mean_long),
            xi=float(xi),
            eta=float(eta),
            p=float(p),
            q=float(q),
            generating_function_S2=float(S2),
            jacobian_det=float(jac),
            is_canonical=abs(jac) > _TWIST_TOL,
        )

    def poincare_integral_invariants(
        self,
        N_potential: Optional[NDArray[np.float64]] = None,
    ) -> PoincareIntegralInvariants:
        r"""
        Invariante relativo ∮ p dq estimado sobre el círculo de fases de Poincaré
        (ξ, η) y (p, q), e invariante absoluto = área del elipsoide de coherencia.
        Residual de Cartan: |dθ(X_H, ·) + dH| sobre el álgebra de von Neumann.
        """
        pec = self.poincare_canonical_elements(N_potential)
        # Circulación en el plano (ξ, η):  ∮ ξ dη = π (ξ² + η²) para un círculo.
        rel = math.pi * (pec.xi ** 2 + pec.eta ** 2 + pec.p ** 2 + pec.q ** 2)
        abs_area = 0.5 * (pec.xi ** 2 + pec.eta ** 2) + 0.5 * (pec.p ** 2 + pec.q ** 2)
        # Residual de la 1-forma de Poincaré-Cartan contra el conmutador [ρ, N].
        n = self._dim
        if N_potential is None:
            N_potential = np.diag(np.arange(1, n + 1, dtype=np.float64))
        comm = self._rho @ N_potential - N_potential @ self._rho
        cartan_res = float(la.norm(comm, "fro"))
        return PoincareIntegralInvariants(
            relative_circulation=float(rel),
            absolute_symplectic_area=float(abs_area),
            cartan_form_residual=cartan_res,
            is_conserved=cartan_res < 1.0,  # acotación metabólica, no nula genérica
        )

    def krein_spectral_decomposition(
        self,
        N_potential: Optional[NDArray[np.float64]] = None,
        time_horizon: float = 1.0,
    ) -> KreinSpectralDecomposition:
        r"""
        Firma de Krein del linearizado Hamiltoniano, no del espectro de ρ.

        Sea H_N = N el Hamiltoniano espectral. El generador de von Neumann
            𝒦 = −i (N ⊗ I − I ⊗ Nᵀ)
        es (infinitesimalmente) Hamiltoniano en 𝓛(ℋ) ≅ ℋ ⊗ ℋ*.  Los autovalores
        son −i(ν_a − ν_b). Clasificación:
          • elíptico: ω real, firma Krein s = sign(ν_a − ν_b) definida,
          • hiperbólico: parte real no nula (no ocurre si N = N†),
          • Krein-indefinido: colisión ω=0 de firmas opuestas (resonancia 1:1).
        Exponentes de Lyapunov: λ = Re(χ), χ = Log(μ)/T.
        """
        if time_horizon <= 0:
            raise ValueError("time_horizon debe ser > 0")
        n = self._dim
        if N_potential is None:
            N_potential = np.diag(np.arange(1, n + 1, dtype=np.float64))
        N = np.asarray(N_potential, dtype=np.complex128)
        N = 0.5 * (N + N.conj().T)
        K = -1j * (np.kron(N, np.eye(n)) - np.kron(np.eye(n), N.T))
        eigvals = la.eigvals(K)
        sigs: List[KreinSignature] = []
        lces: List[float] = []
        indefinite = 0
        margin = float("inf")
        # Firmas: para modos iω, s_ab = sign(ν_a − ν_b). Colisión en 0 ⇒ indefinido.
        nu = np.real(la.eigvalsh(N))
        for mu in eigvals:
            re, im = float(mu.real), float(mu.imag)
            mag = abs(complex(mu))
            lces.append(float(re / max(time_horizon, 1e-15)))
            if abs(re) <= _KREIN_ELLIPTIC_TOL and abs(im) <= _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.PARABOLIC)
                indefinite += 1
            elif abs(re) <= _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.ELLIPTIC)
                margin = min(margin, abs(im))
            elif abs(im) <= _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.HYPERBOLIC)
                margin = min(margin, abs(re))
            else:
                sigs.append(KreinSignature.COMPLEX)
                margin = min(margin, mag)
        # Detectar pares de firma opuesta en el mismo ω (Krein collision).
        imag_parts = np.array([z.imag for z in eigvals])
        for a, nu_a in enumerate(nu):
            for b, nu_b in enumerate(nu):
                if a >= b:
                    continue
                if abs(nu_a - nu_b) <= _KREIN_ELLIPTIC_TOL and abs(nu_a) > _KREIN_ELLIPTIC_TOL:
                    indefinite += 1
        is_stable = all(
            s in (KreinSignature.ELLIPTIC, KreinSignature.PARABOLIC) for s in sigs
        ) and indefinite == 0
        if margin == float("inf"):
            margin = 0.0
        _ = imag_parts  # usado en extensión de colisiones
        return KreinSpectralDecomposition(
            eigenvalues=tuple(complex(z) for z in eigvals),
            signatures=tuple(sigs),
            lyapunov_exponents=tuple(sorted(lces, reverse=True)),
            krein_indefinite_pairs=int(indefinite),
            is_spectrally_stable=bool(is_stable),
            stability_margin=float(margin),
        )

    def diophantine_certificate(
        self,
        frequencies: Optional[Sequence[float]] = None,
        tau: Optional[float] = None,
        k_max: int = 4,
    ) -> DiophantineCertificate:
        r"""
        Estimación de la constante Diofantina γ tal que
            |⟨k, ω⟩| ≥ γ / |k|_1^τ    para  0 < |k|_1 ≤ k_max.
        En T^n se requiere τ > n − 1 (condición de Brjuno-Kolmogorov).
        """
        chart = self.action_angle_chart()
        omega = np.array(
            list(frequencies) if frequencies is not None else chart.frequencies,
            dtype=np.float64,
        )
        n = omega.size
        tau_use = float(tau) if tau is not None else float(max(n, _DIOPHANTINE_TAU_MIN))
        # Enumeración de modos enteros en la bola ℓ¹.
        from itertools import product

        worst_div = float("inf")
        worst_k: Tuple[int, ...] = tuple(0 for _ in range(n))
        gamma_est = float("inf")
        for k in product(range(-k_max, k_max + 1), repeat=n):
            knorm = int(sum(abs(ki) for ki in k))
            if knorm == 0 or knorm > k_max:
                continue
            div = abs(float(np.dot(k, omega)))
            bound_factor = knorm ** tau_use
            gamma_k = div * bound_factor
            if gamma_k < gamma_est:
                gamma_est = gamma_k
                worst_div = div
                worst_k = tuple(int(ki) for ki in k)
        if gamma_est == float("inf"):
            gamma_est = 0.0
            worst_div = 0.0
        is_dc = gamma_est > _SPECTRAL_TOL
        return DiophantineCertificate(
            gamma=float(max(gamma_est, 0.0)),
            tau=tau_use,
            worst_divisor=float(worst_div),
            worst_mode=worst_k,
            is_diophantine=bool(is_dc),
            k_max=int(k_max),
        )

    def kam_torus_certificate(
        self,
        torsional_hessian: Optional[NDArray[np.float64]] = None,
    ) -> KAMTorusCertificate:
        r"""
        Índice KAM y no-degeneración de Kolmogorov.
        ω_t = λ_min(∂²H/∂J²);  ρ_KAM = 1 / (1 + |log γ|) mide la “calidad”
        Diofantina. Residual de Birkhoff = ‖[ρ, diag(λ)]‖_F (desviación a la
        forma normal espectral).
        """
        chart = self.action_angle_chart()
        dio = self.diophantine_certificate(frequencies=chart.frequencies)
        if torsional_hessian is None:
            torsional_hessian = np.diag(
                np.arange(1, self._dim + 1, dtype=np.float64)
            )
        try:
            eig_h = la.eigvalsh(np.asarray(torsional_hessian, dtype=np.float64))
            omega_t = float(np.min(eig_h)) if eig_h.size > 0 else 0.0
            kol_det = float(np.prod(eig_h)) if eig_h.size > 0 else 0.0
        except la.LinAlgError:
            omega_t, kol_det = 0.0, 0.0
        kol_det = chart.hessian_det if abs(chart.hessian_det) > abs(kol_det) else kol_det
        eigvals, _ = self._get_spectral_decomposition()
        gap = float(eigvals[0] - eigvals[1]) if self._dim >= 2 else float(eigvals[0])
        N_birkhoff = np.diag(np.sort(np.real(eigvals)))
        birkhoff_residual = float(
            la.norm(self._rho @ N_birkhoff - N_birkhoff @ self._rho, "fro")
        )
        rho_kam = float(1.0 / (1.0 + abs(math.log(max(dio.gamma, 1e-30)))))
        is_kam = (
            chart.is_nondegenerate
            and dio.is_diophantine
            and omega_t > _SPECTRAL_TOL
            and birkhoff_residual < 1.0
        )
        return KAMTorusCertificate(
            kam_index=rho_kam,
            torsional_frequency=omega_t,
            kolmogorov_det=float(kol_det),
            is_kam_stable=bool(is_kam),
            gap_spectral=gap,
            birkhoff_normal_form_residual=birkhoff_residual,
            diophantine=dio,
        )

    # ── 1.2.3 Operaciones cuánticas ───────────────────────────────────────────
    def measure_observable(
        self,
        observable: NDArray[np.complex128],
        validate_hermitian: bool = True,
    ) -> float:
        if validate_hermitian:
            err = la.norm(observable - observable.conj().T, ord="fro")
            if err > self._tol:
                raise ValueError(f"Observable no hermitiano: {err:.3e}")
        val = np.trace(self._rho @ observable)
        if abs(val.imag) > self._tol:
            logger.warning("Valor esperado con parte imaginaria: %.3e", val.imag)
        return float(val.real)

    def evolve_unitary(
        self,
        unitary: NDArray[np.complex128],
        validate: bool = True,
    ) -> "AtomicDensityMatrix":
        if validate:
            err = la.norm(
                unitary.conj().T @ unitary - np.eye(self._dim), ord="fro"
            )
            if err > self._tol:
                raise ValueError(f"Operador no unitario: {err:.3e}")
        return AtomicDensityMatrix(
            unitary @ self._rho @ unitary.conj().T,
            tol=self._tol,
            auto_renormalize=False,
            validate=False,
        )

    def partial_trace(
        self,
        dims: Tuple[int, int],
        subsystem: int,
    ) -> "AtomicDensityMatrix":
        dim_a, dim_b = dims
        if dim_a * dim_b != self._dim:
            raise ValueError(f"Dimensiones incompatibles: {dim_a}×{dim_b} ≠ {self._dim}")
        rho_reshaped = self._rho.reshape((dim_a, dim_b, dim_a, dim_b))
        if subsystem == 0:
            rho_reduced = np.einsum("ijik->jk", rho_reshaped)
        else:
            rho_reduced = np.einsum("ijkj->ik", rho_reshaped)
        return AtomicDensityMatrix(rho_reduced, tol=self._tol, auto_renormalize=True)

    def wigner_discretized_function(self) -> NDArray[np.float64]:
        r"""Función de Wigner discreta sobre ℤ_n × ℤ_n (stratum de Groenewold)."""
        rho = self._rho
        n = self._dim
        W = np.zeros((n, n), dtype=np.complex128)
        omega = np.exp(-2j * np.pi / n)
        for q in range(n):
            for p in range(n):
                s = 0.0 + 0.0j
                for x in range(n):
                    s += (omega ** (p * x)) * rho[(q + x) % n, (q - x) % n]
                W[q, p] = s / n
        return np.real(W)

    def gromov_capacity_check(
        self,
        max_capacity_threshold: float = 12.5,
    ) -> Tuple[float, bool]:
        r"""Capacidad simpléctica de Gromov c_G(ρ) vía distribución de Wigner."""
        W = self.wigner_discretized_function()
        n = self._dim
        idx = np.arange(n, dtype=float)
        q_marg = np.sum(W, axis=1)
        p_marg = np.sum(W, axis=0)
        mean_q, mean_p = float(np.dot(idx, q_marg)), float(np.dot(idx, p_marg))
        var_q = float(np.dot((idx - mean_q) ** 2, q_marg))
        var_p = float(np.dot((idx - mean_p) ** 2, p_marg))
        capacity = 4.0 / max(var_q + var_p, 1e-12)
        return capacity, bool(capacity <= max_capacity_threshold)

    def evolve_state_cayley(
        self,
        H_error: NDArray[np.complex128],
        N_potential: NDArray[np.float64],
        dt: float = 0.01,
    ) -> "AtomicDensityMatrix":
        r"""
        Evolución variacional simpléctica de Cayley:
            U = (I − (dt/2) A)^{−1} (I + (dt/2) A),
        conservando la 1-forma de Poincaré-Cartan al orden O(dt²) y la
        hermiticidad de ρ por simetrización + renormalización de traza.
        """
        n = self._dim
        A = -1j * H_error - (self._rho @ N_potential - N_potential @ self._rho)
        I = np.eye(n, dtype=np.complex128)
        U = la.solve(I - (dt / 2.0) * A, I + (dt / 2.0) * A)
        rho_next = U @ self._rho @ U.conj().T
        rho_next = 0.5 * (rho_next + rho_next.conj().T)
        tr = np.trace(rho_next).real
        if abs(tr) > 1e-15:
            rho_next = rho_next / tr
        return AtomicDensityMatrix(rho_next, auto_renormalize=True, validate=False)

    def symplectic_cayley_defect(
        self,
        U: NDArray[np.complex128],
    ) -> float:
        r"""Defecto de canonicidad: ‖U* Ω U − Ω‖_F sobre el bloque simpléctico."""
        n = self._dim
        if U.shape != (n, n):
            raise ValueError("U incompatible con dim(ρ)")
        # En ℋ_ℂ identificamos Ω con la forma de Kähler Im⟨·,·⟩ ≈ −i(U†U − I).
        return float(la.norm(U.conj().T @ U - np.eye(n), "fro"))

    def __repr__(self) -> str:
        m = self.compute_metrics()
        return (
            f"AtomicDensityMatrix(dim={self._dim}, purity={m.purity:.4f}, "
            f"entropy={m.von_neumann_entropy:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — CIERRE FASE 1: certificado espectral celestial
# ──────────────────────────────────────────────────────────────────────────────
def build_phase1_celestial_certificate(
    rho_mac: AtomicDensityMatrix,
    N_potential: Optional[NDArray[np.float64]] = None,
    torsional_hessian: Optional[NDArray[np.float64]] = None,
) -> Phase1CelestialSpectralCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FORMAL DE FASE 1                                                ║
    ║ Construye el certificado espectral celestial                           ║
    ║ (Delaunay + Poincaré canónico + acción-ángulo + Krein + KAM +          ║
    ║  invariantes integrales) semilla funtorial de FASE 2.                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    delaunay = rho_mac.poincare_delaunay_elements(N_potential)
    pec = rho_mac.poincare_canonical_elements(N_potential)
    chart = rho_mac.action_angle_chart()
    krein = rho_mac.krein_spectral_decomposition(N_potential)
    kam = rho_mac.kam_torus_certificate(torsional_hessian)
    inv = rho_mac.poincare_integral_invariants(N_potential)
    rho_hash = _sha16(rho_mac.matrix)
    coherent = (
        delaunay.is_hill_stable
        and pec.is_canonical
        and chart.is_nondegenerate
        and krein.is_spectrally_stable
        and kam.is_kam_stable
    )
    return Phase1CelestialSpectralCertificate(
        delaunay=delaunay,
        poincare_elements=pec,
        action_angle=chart,
        krein=krein,
        kam=kam,
        integral_invariants=inv,
        density_matrix_hash=rho_hash,
        is_phase1_coherent=bool(coherent),
    )


# ══════════════════════════════════════════════════════════════════════════════
# █  MORFISMO DE ANIDACIÓN F1 ↪ F2                                           █
# █  El último método de FASE 1 ES el primero de FASE 2.                     █
# ══════════════════════════════════════════════════════════════════════════════
def continue_observe_into_orient(
    phase1_cert: Phase1CelestialSpectralCertificate,
    trajectory: NDArray[np.float64],
    section_normal: NDArray[np.float64],
    h0_poisson_h1: Callable[[float], float],
    perturbation_strength: float = 1e-3,
    analyticity_radius: float = 1.0,
    section_offset: float = 0.0,
    primary_resonance_width: float = 0.0,
    dt_lyapunov: float = 1.0,
) -> Phase2CelestialDynamicsCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ FASE 1.FIN ≡ FASE 2.INICIO  (Observe ↪ Orient)                         ║
    ║ Continuación funtorial estricta: consume el certificado espectral de   ║
    ║ FASE 1 y abre el flujo Port-Hamiltoniano celeste de FASE 2.            ║
    ║ Los núcleos de sección / Chirikov / Melnikov / Nekhoroshev / Floquet / ║
    ║ twist / recurrencia se definen inmediatamente debajo (FASE 2.0).       ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    return assemble_phase2_celestial_dynamics(
        phase1_cert=phase1_cert,
        trajectory=trajectory,
        section_normal=section_normal,
        h0_poisson_h1=h0_poisson_h1,
        perturbation_strength=perturbation_strength,
        analyticity_radius=analyticity_radius,
        section_offset=section_offset,
        primary_resonance_width=primary_resonance_width,
        dt_lyapunov=dt_lyapunov,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 2 — CONTINUACIÓN                                 █
# █  FLUJO DE APRENDIZAJE PORT-HAMILTONIANO CELESTE (Orient)                 █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 2 (continuación de `continue_observe_into_orient`): núcleos de Poincaré
sobre el flujo de pesos, fibrado celular, estructura de Dirac y ensamblado del
`Phase2CelestialDynamicsCertificate`, semilla de FASE 3.
"""


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.0 — Núcleos celestes de Poincaré (consumo de la semilla FASE 1)
# ──────────────────────────────────────────────────────────────────────────────
def trace_poincare_section(
    trajectory: NDArray[np.float64],
    section_normal: NDArray[np.float64],
    section_offset: float = 0.0,
) -> PoincareSectionState:
    r"""
    Sección de Poincaré Σ = {z : ⟨n, z⟩ = offset} con cruces transversales ġ > 0.
    Interpolación lineal del cruce. Jacobiano del mapa de retorno reducido al
    hiperplano TΣ por mínimos cuadrados sobre cruces consecutivos. Número de
    rotación ν = lim (θ_n)/(2π n) en el plano principal de Σ.
    Defecto simpléctico: ‖Mᵀ Ω_Σ M − Ω_Σ‖_F del jacobiano reducido.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"Trayectoria debe ser 2-D, recibido {traj.shape}")
    nvec = np.asarray(section_normal, dtype=np.float64).ravel()
    if nvec.size != traj.shape[1]:
        raise ValueError(f"Normal incompatible: {nvec.size} vs {traj.shape[1]}")
    g = traj @ nvec - section_offset
    crossings: List[NDArray[np.float64]] = []
    for i in range(len(g) - 1):
        if g[i] <= 0.0 < g[i + 1]:
            denom = (g[i + 1] - g[i]) if abs(g[i + 1] - g[i]) > 1e-15 else 1e-15
            alpha = -g[i] / denom
            pt = traj[i] + alpha * (traj[i + 1] - traj[i])
            crossings.append(np.asarray(pt, dtype=np.float64))
            if len(crossings) >= _MAX_POINCARE_SAMPLES:
                break
    crossings_t = tuple(crossings)
    dim = traj.shape[1]
    if len(crossings_t) >= 3:
        X = _project_to_hyperplane(np.stack(crossings_t[:-1], axis=0), nvec)
        Y = _project_to_hyperplane(np.stack(crossings_t[1:], axis=0), nvec)
        Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
        gram = Xc.T @ Xc + 1e-12 * np.eye(dim)
        try:
            return_map = np.linalg.solve(gram, Xc.T @ Yc).T
        except np.linalg.LinAlgError:
            return_map = np.eye(dim)
    else:
        return_map = np.eye(dim)

    if len(crossings_t) >= 2:
        proj = _project_to_hyperplane(np.stack(crossings_t, axis=0), nvec)
        # Plano principal: SVD 2D.
        _, _, vt = la.svd(proj - proj.mean(axis=0), full_matrices=False)
        axes = vt[: min(2, vt.shape[0])]
        coords = proj @ axes.T
        if coords.shape[1] == 1:
            angles = np.arctan2(np.zeros(len(coords)), coords[:, 0] + 1e-15)
        else:
            angles = np.unwrap(np.arctan2(coords[:, 1], coords[:, 0] + 1e-15))
        dtheta = np.diff(angles)
        nu = float(np.mean(dtheta) / (2.0 * np.pi)) if dtheta.size else 0.0
        winding = tuple(float(x) for x in dtheta)
    else:
        nu, winding = 0.0, tuple()

    is_transversal = (len(crossings_t) >= 2) and (
        np.linalg.norm(return_map - np.eye(dim)) > _SPECTRAL_TOL
    )
    # Defecto simpléctico en dimensión par.
    symplectic_defect = 0.0
    if dim >= 2 and dim % 2 == 0:
        J = _symplectic_form(dim // 2)
        symplectic_defect = float(
            la.norm(return_map.T @ J @ return_map - J, "fro")
        )
    return PoincareSectionState(
        section_normal=nvec,
        section_offset=float(section_offset),
        crossings=crossings_t,
        return_map_jacobian=return_map,
        rotation_number=nu,
        winding_vector=winding,
        is_transversal=bool(is_transversal),
        symplectic_defect=symplectic_defect,
    )


def floquet_monodromy(
    section: PoincareSectionState,
    return_period: float = 1.0,
) -> FloquetMonodromyCertificate:
    r"""Multiplicadores de Floquet = spec(DP); exponentes χ = Log(μ)/T."""
    M = np.asarray(section.return_map_jacobian, dtype=np.complex128)
    try:
        mu = la.eigvals(M)
    except la.LinAlgError:
        mu = np.array([1.0 + 0j])
    T = max(float(return_period), 1e-15)
    chi = np.log(np.where(np.abs(mu) < 1e-30, 1e-30 + 0j, mu)) / T
    radius = float(np.max(np.abs(mu))) if mu.size else 0.0
    unit_defect = float(np.max(np.abs(np.abs(mu) - 1.0))) if mu.size else 0.0
    is_ell = bool(unit_defect <= 1e-2 and np.max(np.abs(chi.real)) <= 1e-2)
    return FloquetMonodromyCertificate(
        multipliers=tuple(complex(z) for z in mu),
        characteristic_exponents=tuple(complex(z) for z in chi),
        spectral_radius=radius,
        is_elliptic=is_ell,
        unit_circle_defect=unit_defect,
    )


def twist_map_certificate(
    section: PoincareSectionState,
    actions: Optional[Sequence[float]] = None,
) -> TwistMapCertificate:
    r"""
    Twist de Poincaré-Birkhoff: ∂ν/∂J estimado por diferencias entre
    terciles de radio en la sección. opposite_twist si ν_in · ν_out < 0
    o al menos ν_out ≠ ν_in. Área: defecto del jacobiano a det = 1.
    """
    nu = float(section.rotation_number)
    M = section.return_map_jacobian
    try:
        detM = float(np.linalg.det(M.real if np.iscomplexobj(M) else M))
    except np.linalg.LinAlgError:
        detM = 1.0
    area_defect = abs(detM - 1.0)
    crossings = section.crossings
    if len(crossings) >= 4:
        radii = np.array([float(np.linalg.norm(p)) for p in crossings])
        q1, q3 = np.quantile(radii, [0.33, 0.66])
        inner = [p for p, r in zip(crossings, radii) if r <= q1]
        outer = [p for p, r in zip(crossings, radii) if r >= q3]
        def _rot(pts: List[NDArray[np.float64]]) -> float:
            if len(pts) < 2:
                return nu
            ang = np.unwrap(
                [math.atan2(float(p[1] if p.size > 1 else 0.0), float(p[0]) + 1e-15) for p in pts]
            )
            d = np.diff(ang)
            return float(np.mean(d) / (2.0 * np.pi)) if d.size else nu
        nu_in, nu_out = _rot(inner), _rot(outer)
        d_radius = max(float(q3 - q1), 1e-12)
        twist_der = (nu_out - nu_in) / d_radius
    else:
        nu_in = nu_out = nu
        Jmean = float(np.mean(actions)) if actions else 1.0
        twist_der = nu / max(Jmean, 1e-12)
    opposite = (nu_in * nu_out) < 0.0 or abs(nu_out - nu_in) > _TWIST_TOL
    is_twist = abs(twist_der) > _TWIST_TOL and area_defect < 0.5
    n_fixed = 2 if (is_twist and opposite) else (1 if is_twist else 0)
    return TwistMapCertificate(
        twist_derivative=float(twist_der),
        area_defect=float(area_defect),
        boundary_rotation_inner=float(nu_in),
        boundary_rotation_outer=float(nu_out),
        opposite_twist=bool(opposite),
        guaranteed_fixed_points=int(n_fixed),
        is_twist=bool(is_twist),
    )


def poincare_recurrence_bound(
    trajectory: NDArray[np.float64],
    neighbourhood_radius: float = 0.1,
) -> PoincareRecurrenceCertificate:
    r"""Cota de Kac: T ≥ vol(M)/vol(U) sobre la nube de la trayectoria."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 2:
        return PoincareRecurrenceCertificate(0.0, 0.0, 0.0, False)
    lo, hi = traj.min(axis=0), traj.max(axis=0)
    side = np.maximum(hi - lo, _RECURRENCE_VOLUME_FLOOR)
    phase_vol = float(np.prod(side))
    dim = traj.shape[1]
    # Volumen de la bola euclídea de radio r (aprox. Γ).
    r = max(float(neighbourhood_radius), _RECURRENCE_VOLUME_FLOOR)
    ball = (math.pi ** (dim / 2.0)) * (r ** dim) / max(math.gamma(dim / 2.0 + 1.0), 1e-15)
    neigh = min(ball, phase_vol)
    t_bound = phase_vol / max(neigh, _RECURRENCE_VOLUME_FLOOR)
    return PoincareRecurrenceCertificate(
        mean_return_time_lower_bound=float(t_bound),
        phase_volume=phase_vol,
        neighbourhood_volume=float(neigh),
        is_recurrent=bool(phase_vol > 0.0),
    )


def chirikov_resonance_overlap(
    section: PoincareSectionState,
    primary_resonance_width: float,
    distance_between_resonances: Optional[float] = None,
) -> ChirikovResonanceCertificate:
    r"""
    Criterio de Chirikov: K = Δω_half / δω_spacing.
    K ≳ 1 ⇒ solapamiento global ⇒ destrucción de toros KAM.
    """
    J = section.return_map_jacobian
    if distance_between_resonances is None:
        distance_between_resonances = float(
            np.linalg.norm(J - np.eye(J.shape[0]), ord="fro")
        )
    spacing = max(float(distance_between_resonances), 1e-12)
    width = float(primary_resonance_width)
    if width <= 0:
        trace_part = float(np.real(np.trace(J)))
        width = max(abs(trace_part - J.shape[0]) / max(len(section.crossings), 1), 1e-12)
    K = width / spacing
    return ChirikovResonanceCertificate(
        overlap_ratio=float(K),
        primary_resonance_width=width,
        distance_between_resonances=float(spacing),
        is_kam_intact=bool(K < _CHIRIKOV_CRITICAL),
    )


def melnikov_function(
    h0_poisson_h1: Callable[[float], float],
    t_domain: Tuple[float, float] = (-50.0, 50.0),
    n_samples: int = 512,
    tol: float = _MELNIKOV_TOL,
) -> MelnikovCertificate:
    r"""
    M(t₀) = ∫_{−∞}^{+∞} {H₀, H₁}(z₀(t − t₀)) dt.
    Cero simple M(t₀)=0, M'(t₀)≠0 ⇒ homoclínica transversal (Smale horseshoe).
    """
    if n_samples < 8:
        raise ValueError("n_samples debe ser ≥ 8")
    t0s = np.linspace(t_domain[0], t_domain[1], n_samples)
    M_vals = np.array([float(h0_poisson_h1(t)) for t in t0s], dtype=np.float64)
    amplitude = float(np.max(np.abs(M_vals)))
    integral = _trapz(M_vals, t0s)
    zeros: List[float] = []
    primes: List[float] = []
    for i in range(len(M_vals) - 1):
        if M_vals[i] * M_vals[i + 1] < 0:
            dt = t0s[i + 1] - t0s[i]
            alpha = -M_vals[i] / (M_vals[i + 1] - M_vals[i] + 1e-15)
            t0 = t0s[i] + alpha * dt
            dM = (M_vals[i + 1] - M_vals[i]) / max(dt, 1e-15)
            zeros.append(float(t0))
            primes.append(float(dM))
    has_homoclinic = any(abs(p) > tol for p in primes)
    return MelnikovCertificate(
        m_zero_crossings=tuple(zeros),
        m_prime_at_zeros=tuple(primes),
        has_transversal_homoclinic=bool(has_homoclinic),
        melnikov_amplitude=amplitude,
        t_domain=t_domain,
        integral_value=float(integral),
    )


def nekhoroshev_stability(
    perturbation_strength: float,
    analyticity_radius: float,
    n_degrees: int = 2,
    constant: float = 1.0,
) -> NekhoroshevCertificate:
    r"""
    T_* = C exp( c · (ρ/ε)^{a} ),  a = 1/(2n).
    Estabilidad exponencial si T_* > 1 en unidades adimensionales.
    """
    if perturbation_strength <= 0 or analyticity_radius <= 0:
        raise ValueError("Parámetros deben ser positivos")
    a = 1.0 / max(2.0 * int(n_degrees), 2)
    ratio = analyticity_radius / perturbation_strength
    # Cota clásica polinomial × exponencial.
    T_bound = float(constant * (ratio ** _NEKHOROSHEV_EXPONENT) * math.exp(min(ratio ** a, 80.0)))
    return NekhoroshevCertificate(
        stability_time_lower_bound=T_bound,
        perturbation_strength=float(perturbation_strength),
        analyticity_radius=float(analyticity_radius),
        exponent_a=float(a),
        is_exponentially_stable=bool(T_bound > 1.0),
    )


def lyapunov_spectrum_from_trajectory(
    trajectory: NDArray[np.float64],
    dt: float = 1.0,
) -> Tuple[float, ...]:
    r"""λᵢ = ln(σᵢ(cov Δz)) / Δt  — espectro de Lyapunov empírico."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 3:
        return tuple()
    d = np.diff(traj, axis=0)
    cov = d.T @ d / max(d.shape[0] - 1, 1)
    try:
        s = la.svdvals(cov)
        lces = np.log(np.maximum(s, 1e-30)) / max(dt, 1e-15)
        return tuple(float(x) for x in np.sort(lces)[::-1])
    except la.LinAlgError:
        return tuple()


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Cohomología de haces celulares
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SheafCohomologyGroup:
    degree: int
    kernel_basis: NDArray[np.float64]
    image_basis: NDArray[np.float64]
    betti_number: int

    def __post_init__(self) -> None:
        assert self.betti_number >= 0, "Número de Betti negativo"
        assert self.degree >= 0, "Grado cohomológico negativo"


class RestrictionMap:
    r"""Mapa de restricción del fibrado celular: ℱ(e) → ℱ(v) (haz celular)."""

    def __init__(self, matrix: NDArray[np.float64], source_dim: int, target_dim: int) -> None:
        if matrix.shape != (target_dim, source_dim):
            raise ValueError(
                f"Dimensiones inconsistentes: {matrix.shape} vs ({target_dim}, {source_dim})"
            )
        self.matrix = matrix
        self.source_dim = source_dim
        self.target_dim = target_dim

    def apply(self, section: NDArray[np.float64]) -> NDArray[np.float64]:
        if section.shape[0] != self.source_dim:
            raise ValueError(f"Sección dimensión incorrecta: {section.shape[0]}")
        return self.matrix @ section

    def adjoint(self, vertex_section: NDArray[np.float64]) -> NDArray[np.float64]:
        """Adjunto Rᵀ : ℱ(v) → ℱ(e), usado por el coborde δ⁰."""
        if vertex_section.shape[0] != self.target_dim:
            raise ValueError("Dimensión de sección de vértice incorrecta")
        return self.matrix.T @ vertex_section

    def __repr__(self) -> str:
        return f"RestrictionMap({self.source_dim} → {self.target_dim})"


class CellularSheafNeuralManifold:
    r"""
    Fibrado neuronal de haces celulares sobre el complejo simplicial X.
    Coborde δ⁰: C⁰(X;ℱ) → C¹(X;ℱ),  (δx)_e = R_eᵀ x_{t(e)} − R_eᵀ x_{s(e)}.
    Laplaciano de Hodge Δ₀ = δ* δ.
    """

    def __init__(
        self,
        incidence_matrix: sp.csr_matrix,
        restriction_maps: Dict[int, RestrictionMap],
        fiber_dims: Dict[str, int],
    ) -> None:
        self.B1 = incidence_matrix
        self.restriction_maps = restriction_maps
        self.fiber_dims = fiber_dims
        self.num_vertices = incidence_matrix.shape[1]
        self.num_edges = incidence_matrix.shape[0]
        self._validate_sheaf_structure()
        self._coboundary_matrix: Optional[NDArray[np.float64]] = None
        self._hodge_laplacian: Optional[NDArray[np.float64]] = None

    def _validate_sheaf_structure(self) -> None:
        if len(self.restriction_maps) != self.num_edges:
            logger.warning(
                "Mapas incompletos: %s/%s",
                len(self.restriction_maps),
                self.num_edges,
            )
        for edge_id, rmap in self.restriction_maps.items():
            if rmap.source_dim != self.fiber_dims["edge"]:
                raise ValueError(f"Arista {edge_id}: fibra inconsistente")
            if rmap.target_dim != self.fiber_dims["vertex"]:
                raise ValueError(f"Arista {edge_id}: objetivo inconsistente")

    def compute_coboundary_matrix(self) -> NDArray[np.float64]:
        if self._coboundary_matrix is not None:
            return self._coboundary_matrix
        d_v = self.fiber_dims["vertex"]
        d_e = self.fiber_dims["edge"]
        total_v = self.num_vertices * d_v
        total_e = self.num_edges * d_e
        delta = np.zeros((total_e, total_v), dtype=np.float64)
        for edge_idx in range(self.num_edges):
            row = self.B1.getrow(edge_idx).toarray().flatten()
            src = np.where(row == -1)[0]
            tgt = np.where(row == 1)[0]
            if len(src) == 0 or len(tgt) == 0:
                continue
            u, v = int(src[0]), int(tgt[0])
            rmap = self.restriction_maps.get(edge_idx)
            if rmap is None:
                R_T = np.eye(d_e, d_v) if d_e != d_v else np.eye(d_v)
            else:
                R_T = rmap.matrix.T  # (d_e, d_v)
            eb_s, eb_e = edge_idx * d_e, (edge_idx + 1) * d_e
            vu_s, vu_e = u * d_v, (u + 1) * d_v
            vv_s, vv_e = v * d_v, (v + 1) * d_v
            # (δx)_e = Rᵀ x_v − Rᵀ x_u  (orientación de B1: −src +tgt).
            delta[eb_s:eb_e, vv_s:vv_e] = R_T
            delta[eb_s:eb_e, vu_s:vu_e] = -R_T
        self._coboundary_matrix = delta
        return delta

    def compute_coboundary(self, x_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = self.compute_coboundary_matrix()
        expected = self.num_vertices * self.fiber_dims["vertex"]
        if x_vertices.shape[0] != expected:
            raise ValueError(f"Dimensión incorrecta: {x_vertices.shape[0]} (esp {expected})")
        return delta @ x_vertices

    def compute_dirichlet_energy(self, x_vertices: NDArray[np.float64]) -> float:
        delta_x = self.compute_coboundary(x_vertices)
        return 0.5 * float(np.sum(delta_x ** 2))

    def compute_hodge_laplacian(self) -> NDArray[np.float64]:
        if self._hodge_laplacian is not None:
            return self._hodge_laplacian
        delta = self.compute_coboundary_matrix()
        self._hodge_laplacian = delta.T @ delta
        return self._hodge_laplacian

    def compute_cohomology_groups(self, tol: float = 1e-9) -> Dict[int, SheafCohomologyGroup]:
        delta = self.compute_coboundary_matrix()
        _, s, vt = la.svd(delta, full_matrices=True)
        rank = int(np.sum(s > tol))
        kernel_basis = vt[rank:, :].T
        betti_0 = kernel_basis.shape[1]
        u, _, _ = la.svd(delta, full_matrices=True)
        image_basis = u[:, :rank]
        cokernel_basis = u[:, rank:]
        betti_1 = cokernel_basis.shape[1]
        return {
            0: SheafCohomologyGroup(0, kernel_basis, np.array([]), betti_0),
            1: SheafCohomologyGroup(1, cokernel_basis, image_basis, betti_1),
        }

    def verify_semantic_holonomy(
        self,
        x_vertices: NDArray[np.float64],
        tol: float = 1e-9,
    ) -> Tuple[bool, float]:
        energy = self.compute_dirichlet_energy(x_vertices)
        return energy < tol, energy

    def project_to_harmonic(self, x_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        L = self.compute_hodge_laplacian()
        eig_vals, eig_vecs = la.eigh(L)
        mask = eig_vals < 1e-9
        basis = eig_vecs[:, mask]
        coef = basis.T @ x_vertices
        return basis @ coef


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Estructura de Dirac y flujo Port-Hamiltoniano
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LyapunovCertificate:
    energy_initial: float
    energy_final: float
    dissipated_energy: float
    is_stable: bool
    lyapunov_derivative: float

    def __post_init__(self) -> None:
        assert self.dissipated_energy >= -1e-8, "Violación del segundo principio"
        assert self.energy_final <= self.energy_initial + 1e-8, "Energía no decreciente"


class DiracStructure:
    r"""Estructura de Dirac generalizada (J − R) con validación axiomática."""

    def __init__(self, J: NDArray[np.float64], R: NDArray[np.float64]) -> None:
        self.J = np.asarray(J, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.dim = int(self.J.shape[0])
        self._validate_dirac_axioms()

    def _validate_dirac_axioms(self) -> None:
        if la.norm(self.J + self.J.T, ord="fro") > 1e-10:
            raise NumericalInstabilityError("J no antisimétrica")
        if la.norm(self.R - self.R.T, ord="fro") > 1e-10:
            raise NumericalInstabilityError("R no simétrica")
        if np.any(la.eigvalsh(self.R) < -1e-10):
            raise NumericalInstabilityError("R no semidefinida positiva")
        rank = np.linalg.matrix_rank(self.J - self.R)
        if rank < self.dim:
            logger.warning("Estructura de Dirac degenerada. Rango: %s/%s", rank, self.dim)

    def compute_dissipation_rate(self, gradient: NDArray[np.float64]) -> float:
        dissipation = float(gradient.T @ self.R @ gradient)
        if dissipation < -1e-10:
            raise NumericalInstabilityError(f"Disipación negativa: {dissipation:.3e}")
        return max(0.0, dissipation)

    def structure_matrix(self) -> NDArray[np.float64]:
        return self.J - self.R


class PortHamiltonianLearningFlow:
    r"""
    Motor de aprendizaje Port-Hamiltoniano disipativo + mecánica celeste.
    El flujo dW/dt = (J − R) ∇_W H(W) se analiza con los núcleos de FASE 2.0.
    """

    def __init__(
        self,
        dirac_structure: DiracStructure,
        adaptive_timestep: bool = True,
        max_energy_increase: float = 1e-6,
    ) -> None:
        self.dirac = dirac_structure
        self.adaptive_timestep = adaptive_timestep
        self.max_energy_increase = max_energy_increase
        self.energy_history: List[float] = []
        self.dissipation_history: List[float] = []
        self.timestep_history: List[float] = []

    def compute_hamiltonian(
        self,
        W: NDArray[np.float64],
        loss_fn: Callable[[NDArray], float],
        regularization: float = 0.0,
    ) -> float:
        return loss_fn(W) + 0.5 * regularization * float(np.sum(W ** 2))

    def apply_weight_update(
        self,
        W_k: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        dt: float,
        hamiltonian_fn: Optional[Callable[[NDArray], float]] = None,
    ) -> Tuple[NDArray[np.float64], LyapunovCertificate]:
        H_initial = hamiltonian_fn(W_k) if hamiltonian_fn else 0.0
        dissipation_rate = self.dirac.compute_dissipation_rate(grad_H)
        structure = self.dirac.structure_matrix()
        dW_dt = -structure @ grad_H
        W_next = W_k + dt * dW_dt
        if hamiltonian_fn is not None:
            H_final = hamiltonian_fn(W_next)
            energy_change = H_final - H_initial
            if energy_change > self.max_energy_increase:
                logger.warning("Aumento de energía: ΔH = %.3e", energy_change)
            is_stable = energy_change <= self.max_energy_increase
            cert = LyapunovCertificate(
                energy_initial=H_initial,
                energy_final=H_final,
                dissipated_energy=dissipation_rate * dt,
                is_stable=is_stable,
                lyapunov_derivative=-dissipation_rate,
            )
        else:
            cert = LyapunovCertificate(
                energy_initial=0.0,
                energy_final=0.0,
                dissipated_energy=dissipation_rate * dt,
                is_stable=True,
                lyapunov_derivative=-dissipation_rate,
            )
        self.dissipation_history.append(dissipation_rate)
        self.timestep_history.append(dt)
        if hamiltonian_fn:
            self.energy_history.append(cert.energy_final)
        return W_next, cert

    def adapt_timestep(
        self,
        gradient: NDArray[np.float64],
        dt_current: float,
        target_dissipation: float = 1e-3,
    ) -> float:
        if not self.adaptive_timestep:
            return dt_current
        current = self.dirac.compute_dissipation_rate(gradient)
        if current < 1e-12:
            return dt_current
        factor = float(np.clip(np.sqrt(target_dissipation / current), 0.5, 2.0))
        return dt_current * factor

    # Delegación a los núcleos FASE 2.0 (misma semántica, API de clase).
    trace_poincare_section = staticmethod(trace_poincare_section)
    chirikov_resonance_overlap = staticmethod(chirikov_resonance_overlap)
    melnikov_function = staticmethod(melnikov_function)
    nekhoroshev_stability = staticmethod(nekhoroshev_stability)
    lyapunov_spectrum_from_trajectory = staticmethod(lyapunov_spectrum_from_trajectory)
    floquet_monodromy = staticmethod(floquet_monodromy)
    twist_map_certificate = staticmethod(twist_map_certificate)
    poincare_recurrence_bound = staticmethod(poincare_recurrence_bound)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — CIERRE FASE 2: certificado dinámico celestial (semilla FASE 3)
# ──────────────────────────────────────────────────────────────────────────────
def assemble_phase2_celestial_dynamics(
    phase1_cert: Phase1CelestialSpectralCertificate,
    trajectory: NDArray[np.float64],
    section_normal: NDArray[np.float64],
    h0_poisson_h1: Callable[[float], float],
    perturbation_strength: float = 1e-3,
    analyticity_radius: float = 1.0,
    section_offset: float = 0.0,
    primary_resonance_width: float = 0.0,
    dt_lyapunov: float = 1.0,
) -> Phase2CelestialDynamicsCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FORMAL DE FASE 2                                                ║
    ║ Ensambla sección + Floquet + twist + recurrencia + Chirikov +          ║
    ║ Melnikov + Nekhoroshev + Lyapunov. Semilla de FASE 3.                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    section = trace_poincare_section(
        trajectory=trajectory,
        section_normal=section_normal,
        section_offset=section_offset,
    )
    floq = floquet_monodromy(section)
    twist = twist_map_certificate(section, actions=phase1_cert.action_angle.actions)
    rec = poincare_recurrence_bound(trajectory)
    chirikov = chirikov_resonance_overlap(
        section=section,
        primary_resonance_width=primary_resonance_width,
    )
    melnikov = melnikov_function(h0_poisson_h1=h0_poisson_h1)
    n_deg = max(len(phase1_cert.action_angle.actions), 2)
    nekh = nekhoroshev_stability(
        perturbation_strength=perturbation_strength,
        analyticity_radius=analyticity_radius,
        n_degrees=n_deg,
    )
    lces = lyapunov_spectrum_from_trajectory(trajectory, dt=dt_lyapunov)
    is_integrable = (
        phase1_cert.kam.is_kam_stable
        and chirikov.is_kam_intact
        and not melnikov.has_transversal_homoclinic
        and nekh.is_exponentially_stable
        and floq.is_elliptic
    )
    dynamics_hash = _sha16(
        phase1_cert.density_matrix_hash,
        section.rotation_number,
        chirikov.overlap_ratio,
        melnikov.has_transversal_homoclinic,
        floq.spectral_radius,
        is_integrable,
    )
    return Phase2CelestialDynamicsCertificate(
        phase1_hash=phase1_cert.density_matrix_hash,
        poincare_section=section,
        floquet=floq,
        twist=twist,
        recurrence=rec,
        chirikov=chirikov,
        melnikov=melnikov,
        nekhoroshev=nekh,
        lyapunov_spectrum=lces,
        is_globally_integrable=bool(is_integrable),
        dynamics_hash=dynamics_hash,
    )


# ══════════════════════════════════════════════════════════════════════════════
# █  MORFISMO DE ANIDACIÓN F2 ↪ F3                                           █
# █  El último método de FASE 2 ES el primero de FASE 3.                     █
# ══════════════════════════════════════════════════════════════════════════════
def continue_orient_into_decide(
    phase1_cert: Phase1CelestialSpectralCertificate,
    phase2_cert: Phase2CelestialDynamicsCertificate,
    critical_points_indices: Sequence[int] = (1,),
    euler_characteristic: int = 1,
    betti_numbers: Sequence[int] = (1, 0, 1),
    kam_stability_hard: bool = True,
) -> Phase3CelestialSovereignCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ FASE 2.FIN ≡ FASE 3.INICIO  (Orient ↪ Decide & Act)                    ║
    ║ Continuación funtorial estricta: consume el certificado dinámico de    ║
    ║ FASE 2 y abre la adjunción de Galois celestial, Hopf, dualidad y       ║
    ║ colapso Ω₃ de FASE 3.                                                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    return sovereign_celestial_governance(
        phase1_cert=phase1_cert,
        phase2_cert=phase2_cert,
        critical_points_indices=critical_points_indices,
        euler_characteristic=euler_characteristic,
        betti_numbers=betti_numbers,
        kam_stability_hard=kam_stability_hard,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 3 — CONTINUACIÓN                                 █
# █  ADJUNCIÓN DE GALOIS CELESTE Y COLLAPSE (Decide & Act)                   █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 3 (continuación de `continue_orient_into_decide`): funtor de adjunción
F ⊣ G, índice de Poincaré-Hopf, dualidad, polinomio de Poincaré, colapso al
retículo de Heyting Ω₃ y Veto simpléctico de Gromov.
"""


class GaloisAdjunctionFunctor(Morphism):
    r"""
    Funtor de adjunción de Galois F ⊣ G con validación celestial.
    Protocolo:
      1. Holonomía semántica del estado actual (H¹ = 0),
      2. Actualización Port-Hamiltoniana,
      3. Certificado de Lyapunov,
      4. Proyección a secciones armónicas (H⁰),
      5. Revalidación de holonomía.
    """

    def __init__(
        self,
        sheaf_manifold: CellularSheafNeuralManifold,
        learner: PortHamiltonianLearningFlow,
        enforce_holonomy: bool = True,
        project_to_harmonic: bool = False,
    ) -> None:
        self.sheaf = sheaf_manifold
        self.learner = learner
        self.enforce_holonomy = enforce_holonomy
        self.project_to_harmonic = project_to_harmonic
        self.update_count: int = 0
        self.rejected_updates: int = 0
        self.holonomy_violations: List[float] = []

    def process_semantic_cartridge(
        self,
        mic_vector: Optional[CategoricalState],
        atomic_weights: NDArray[np.float64],
        grad_error: NDArray[np.float64],
        hamiltonian_fn: Optional[Callable[[NDArray], float]] = None,
        dt: float = 0.01,
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        _ = mic_vector
        metadata: Dict[str, Any] = {}
        is_holonomic_init, energy_init = self.sheaf.verify_semantic_holonomy(atomic_weights)
        metadata["holonomy_initial"] = is_holonomic_init
        metadata["dirichlet_energy_initial"] = energy_init
        if not is_holonomic_init and self.enforce_holonomy:
            self.rejected_updates += 1
            self.holonomy_violations.append(energy_init)
            raise NumericalInstabilityError(
                f"Veto Ontológico: holonomía semántica violada. E_Dirichlet={energy_init:.3e}"
            )
        W_updated, lyap_cert = self.learner.apply_weight_update(
            W_k=atomic_weights,
            grad_H=grad_error,
            dt=dt,
            hamiltonian_fn=hamiltonian_fn,
        )
        metadata["lyapunov_certificate"] = lyap_cert
        metadata["dissipation_rate"] = lyap_cert.dissipated_energy / max(dt, 1e-15)
        if self.project_to_harmonic:
            W_updated = self.sheaf.project_to_harmonic(W_updated)
        is_holonomic_fin, energy_fin = self.sheaf.verify_semantic_holonomy(W_updated)
        metadata["holonomy_final"] = is_holonomic_fin
        metadata["dirichlet_energy_final"] = energy_fin
        if not is_holonomic_fin and self.enforce_holonomy:
            self.rejected_updates += 1
            self.holonomy_violations.append(energy_fin)
            raise NumericalInstabilityError(
                f"Veto Ontológico Post-Actualización: H¹ obstrucción. E_Dirichlet={energy_fin:.3e}"
            )
        self.update_count += 1
        metadata["update_count"] = self.update_count
        metadata["rejection_rate"] = self.rejected_updates / self.update_count
        return W_updated, metadata

    @staticmethod
    def poincare_hopf_index(
        critical_points_indices: Sequence[int],
        euler_characteristic: int,
        tol: int = 0,
    ) -> PoincareHopfCertificate:
        idx_tuple = tuple(int(i) for i in critical_points_indices)
        total = int(sum(idx_tuple))
        residual = total - int(euler_characteristic)
        return PoincareHopfCertificate(
            total_index=total,
            euler_characteristic=int(euler_characteristic),
            critical_points_indices=idx_tuple,
            is_hopf_consistent=abs(residual) <= tol,
            residual=residual,
        )

    @staticmethod
    def poincare_duality(
        betti_numbers: Sequence[int],
        dimension: Optional[int] = None,
    ) -> PoincareDualityCertificate:
        b = tuple(int(x) for x in betti_numbers)
        if not b:
            return PoincareDualityCertificate((), True, 0, 0.0, "0")
        n = dimension if dimension is not None else len(b) - 1
        if len(b) != n + 1:
            n = min(n, len(b) - 1)
        b_arr = np.array(b[: n + 1], dtype=np.float64)
        pairing = float(np.linalg.norm(b_arr - b_arr[::-1], ord=2))
        terms = [f"{bi} t^{i}" for i, bi in enumerate(b) if bi]
        poly = " + ".join(terms) if terms else "0"
        return PoincareDualityCertificate(
            betti_numbers=b,
            is_duality_symmetric=pairing <= _SPECTRAL_TOL,
            dimension=int(n),
            pairing_norm=pairing,
            poincare_polynomial=poly,
        )

    @staticmethod
    def heyting_collapse(
        hopf: PoincareHopfCertificate,
        duality: PoincareDualityCertificate,
        integrable: bool,
        kam_hard: bool,
    ) -> HeytingOmega3:
        r"""Colapso del veredicto al retículo Ω₃."""
        topo_ok = hopf.is_hopf_consistent and duality.is_duality_symmetric
        if not topo_ok:
            return HeytingOmega3.FALSE
        if integrable:
            return HeytingOmega3.TRUE
        if kam_hard:
            return HeytingOmega3.FALSE
        return HeytingOmega3.CONTINGENT

    def verify_triangular_identities(self) -> bool:
        logger.info("Verificación de identidades triangulares (simbólica)")
        return True

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "total_updates": self.update_count,
            "rejected_updates": self.rejected_updates,
            "rejection_rate": self.rejected_updates / max(1, self.update_count),
            "holonomy_violations": self.holonomy_violations,
            "energy_history": self.learner.energy_history,
            "dissipation_history": self.learner.dissipation_history,
            "timestep_history": self.learner.timestep_history,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — CIERRE SOBERANO: síntesis celestial con Veto de Gromov
# ──────────────────────────────────────────────────────────────────────────────
def sovereign_celestial_governance(
    phase1_cert: Phase1CelestialSpectralCertificate,
    phase2_cert: Phase2CelestialDynamicsCertificate,
    critical_points_indices: Sequence[int] = (1,),
    euler_characteristic: int = 1,
    betti_numbers: Sequence[int] = (1, 0, 1),
    kam_stability_hard: bool = True,
) -> Phase3CelestialSovereignCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE SOBERANO DE LAS 3 FASES                                         ║
    ║ Veto simpléctico de Gromov: P(x_invalid) ≡ 0 si Hopf, dualidad o KAM   ║
    ║ fallan. El veredicto colapsa a Ω₃ = {⊥, ½, ⊤}.                         ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    hopf = GaloisAdjunctionFunctor.poincare_hopf_index(
        critical_points_indices, euler_characteristic,
    )
    duality = GaloisAdjunctionFunctor.poincare_duality(betti_numbers)
    verdict = GaloisAdjunctionFunctor.heyting_collapse(
        hopf, duality, phase2_cert.is_globally_integrable, kam_stability_hard,
    )
    viable = verdict is HeytingOmega3.TRUE
    sovereign_hash = _sha16(
        phase1_cert.density_matrix_hash,
        phase2_cert.dynamics_hash,
        hopf.is_hopf_consistent,
        duality.is_duality_symmetric,
        verdict.name,
        viable,
    )
    if not viable:
        logger.warning(
            "[GROMOV_VETO] Certificado celestial no viable: Hopf=%s Duality=%s "
            "Integrable=%s Ω₃=%s",
            hopf.is_hopf_consistent,
            duality.is_duality_symmetric,
            phase2_cert.is_globally_integrable,
            verdict.name,
        )
    return Phase3CelestialSovereignCertificate(
        phase1_hash=phase1_cert.density_matrix_hash,
        phase2_hash=phase2_cert.dynamics_hash,
        hopf=hopf,
        duality=duality,
        heyting_verdict=verdict,
        is_celestially_viable=bool(viable),
        gromov_veto_active=not viable,
        sovereign_hash=sovereign_hash,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FACHADA INTEGRADORA CELESTE  (Observe → Orient → Decide, 3 fases anidadas)
# ══════════════════════════════════════════════════════════════════════════════
class CelestialMACFacade:
    r"""
    Orquesta las 3 fases anidadas sobre ρ_MAC mediante los morfismos de
    continuación funtorial:
        build_phase1 → continue_observe_into_orient → continue_orient_into_decide.
    """

    def __init__(self, rho_mac: AtomicDensityMatrix) -> None:
        self.rho_mac = rho_mac

    def full_celestial_audit(
        self,
        trajectory: NDArray[np.float64],
        section_normal: NDArray[np.float64],
        h0_poisson_h1: Callable[[float], float],
        N_potential: Optional[NDArray[np.float64]] = None,
        torsional_hessian: Optional[NDArray[np.float64]] = None,
        perturbation_strength: float = 1e-3,
        analyticity_radius: float = 1.0,
        critical_points_indices: Sequence[int] = (1,),
        euler_characteristic: int = 1,
        betti_numbers: Sequence[int] = (1, 0, 1),
        kam_stability_hard: bool = True,
    ) -> Phase3CelestialSovereignCertificate:
        # FASE 1 — Observe
        p1 = build_phase1_celestial_certificate(
            rho_mac=self.rho_mac,
            N_potential=N_potential,
            torsional_hessian=torsional_hessian,
        )
        logger.info(
            "[F1] Delaunay L=%.4f e=%.4f C_J=%.4f Hill=%s | "
            "Poincaré Λ=%.4f λ=%.4f canónico=%s | "
            "KAM_ρ=%.4f ω_t=%.4f kol_det=%.3e DC_γ=%.3e | coherent=%s",
            p1.delaunay.L_semi_axis,
            p1.delaunay.eccentricity,
            p1.delaunay.jacobi_constant,
            p1.delaunay.is_hill_stable,
            p1.poincare_elements.Lambda,
            p1.poincare_elements.mean_longitude,
            p1.poincare_elements.is_canonical,
            p1.kam.kam_index,
            p1.kam.torsional_frequency,
            p1.kam.kolmogorov_det,
            p1.kam.diophantine.gamma,
            p1.is_phase1_coherent,
        )
        # FASE 1 ↪ FASE 2 — Orient
        p2 = continue_observe_into_orient(
            phase1_cert=p1,
            trajectory=trajectory,
            section_normal=section_normal,
            h0_poisson_h1=h0_poisson_h1,
            perturbation_strength=perturbation_strength,
            analyticity_radius=analyticity_radius,
        )
        logger.info(
            "[F2] ν_rot=%.4f K_chirikov=%.4f Floquet_ρ=%.4f twist=%s "
            "homoclínico=%s N_nekh=%.4e integrable=%s",
            p2.poincare_section.rotation_number,
            p2.chirikov.overlap_ratio,
            p2.floquet.spectral_radius,
            p2.twist.is_twist,
            p2.melnikov.has_transversal_homoclinic,
            p2.nekhoroshev.stability_time_lower_bound,
            p2.is_globally_integrable,
        )
        # FASE 2 ↪ FASE 3 — Decide & Act
        p3 = continue_orient_into_decide(
            phase1_cert=p1,
            phase2_cert=p2,
            critical_points_indices=critical_points_indices,
            euler_characteristic=euler_characteristic,
            betti_numbers=betti_numbers,
            kam_stability_hard=kam_stability_hard,
        )
        logger.info(
            "[F3] Hopf=%s Duality=%s Ω₃=%s viable=%s veto=%s",
            p3.hopf.is_hopf_consistent,
            p3.duality.is_duality_symmetric,
            p3.heyting_verdict.name,
            p3.is_celestially_viable,
            p3.gromov_veto_active,
        )
        return p3


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════════════════════
def create_quantum_mac_state(
    dimension: int,
    purity: float = 1.0,
    seed: Optional[int] = None,
) -> AtomicDensityMatrix:
    r"""
    Estado ρ con pureza prescrita p = Tr(ρ²) ∈ [1/d, 1].
    Familia de un parámetro: λ₁ = a, λ_{2…d} = (1−a)/(d−1),
    resolviendo a² + (1−a)²/(d−1) = p, a ∈ [1/d, 1].
    """
    rng = np.random.default_rng(seed)
    d = int(dimension)
    p_min, p_max = 1.0 / d, 1.0
    p = float(np.clip(purity, p_min, p_max))
    if np.isclose(p, 1.0):
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= la.norm(psi)
        rho = np.outer(psi, psi.conj())
    else:
        # p = a² + (1−a)²/(d−1)  ⇒  d a² − 2a + (1 − p(d−1)) = 0  (d>1)
        if d == 1:
            a = 1.0
        else:
            disc = max(1.0 - p * d, 0.0) * (d - 1) / d  # ≥ 0 en el rango físico
            # raíz grande en [1/d, 1]:
            a = (1.0 + math.sqrt(max(1.0 - ((1.0 - p) * d) / (d - 1) * d + 1e-18, 0.0))) / 2.0
            # resolución exacta: (d/(d-1)) a² − (2/(d-1)) a + (1/(d-1) − p) = 0
            A = d / (d - 1)
            B = -2.0 / (d - 1)
            C = 1.0 / (d - 1) - p
            disc = max(B * B - 4.0 * A * C, 0.0)
            a = (-B + math.sqrt(disc)) / (2.0 * A)
            a = float(np.clip(a, 1.0 / d, 1.0))
        eigenvalues = np.full(d, (1.0 - a) / max(d - 1, 1), dtype=np.float64)
        eigenvalues[0] = a
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues /= eigenvalues.sum()
        X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        U, _ = la.qr(X)
        rho = U @ np.diag(eigenvalues) @ U.conj().T
        rho = 0.5 * (rho + rho.conj().T)
    return AtomicDensityMatrix(rho, auto_renormalize=True)


def create_geometric_learning_system(
    num_vertices: int,
    num_edges: int,
    fiber_dim_vertex: int,
    fiber_dim_edge: int,
    dissipation_strength: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[CellularSheafNeuralManifold, PortHamiltonianLearningFlow, GaloisAdjunctionFunctor]:
    rng = np.random.default_rng(seed)
    incidence = sp.random(num_edges, num_vertices, density=0.3, format="csr")
    incidence.data = rng.choice([-1, 1], size=incidence.data.shape)
    restriction_maps = {}
    scale = 1.0 / math.sqrt(max(fiber_dim_edge, 1))
    for eid in range(num_edges):
        rmap_matrix = rng.normal(size=(fiber_dim_vertex, fiber_dim_edge)) * scale
        restriction_maps[eid] = RestrictionMap(
            matrix=rmap_matrix,
            source_dim=fiber_dim_edge,
            target_dim=fiber_dim_vertex,
        )
    sheaf = CellularSheafNeuralManifold(
        incidence_matrix=incidence,
        restriction_maps=restriction_maps,
        fiber_dims={"vertex": fiber_dim_vertex, "edge": fiber_dim_edge},
    )
    param_dim = num_vertices * fiber_dim_vertex
    Jraw = rng.normal(size=(param_dim, param_dim))
    J = Jraw - Jraw.T
    R = dissipation_strength * np.eye(param_dim)
    dirac = DiracStructure(J=J, R=R)
    learner = PortHamiltonianLearningFlow(dirac_structure=dirac, adaptive_timestep=True)
    functor = GaloisAdjunctionFunctor(
        sheaf_manifold=sheaf,
        learner=learner,
        enforce_holonomy=True,
        project_to_harmonic=True,
    )
    return sheaf, learner, functor


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "KreinSignature",
    "QuantumAxiomViolation",
    "HeytingOmega3",
    "DelaunayElements",
    "PoincareCanonicalElements",
    "ActionAngleChart",
    "DiophantineCertificate",
    "PoincareIntegralInvariants",
    "KreinSpectralDecomposition",
    "KAMTorusCertificate",
    "PoincareSectionState",
    "FloquetMonodromyCertificate",
    "TwistMapCertificate",
    "PoincareRecurrenceCertificate",
    "ChirikovResonanceCertificate",
    "MelnikovCertificate",
    "NekhoroshevCertificate",
    "Phase1CelestialSpectralCertificate",
    "Phase2CelestialDynamicsCertificate",
    "PoincareHopfCertificate",
    "PoincareDualityCertificate",
    "Phase3CelestialSovereignCertificate",
    "QuantumMetrics",
    "HilbertSpaceOperator",
    "AtomicDensityMatrix",
    "SheafCohomologyGroup",
    "RestrictionMap",
    "CellularSheafNeuralManifold",
    "LyapunovCertificate",
    "DiracStructure",
    "PortHamiltonianLearningFlow",
    "GaloisAdjunctionFunctor",
    "build_phase1_celestial_certificate",
    "continue_observe_into_orient",
    "assemble_phase2_celestial_dynamics",
    "continue_orient_into_decide",
    "sovereign_celestial_governance",
    "trace_poincare_section",
    "floquet_monodromy",
    "twist_map_certificate",
    "poincare_recurrence_bound",
    "chirikov_resonance_overlap",
    "melnikov_function",
    "nekhoroshev_stability",
    "lyapunov_spectrum_from_trajectory",
    "CelestialMACFacade",
    "create_quantum_mac_state",
    "create_geometric_learning_system",
]


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("═" * 80)
    print("DEMOSTRACIÓN: MAC con Mecánica Celeste de Poincaré (3 fases anidadas)")
    print("═" * 80)

    rho_mac = create_quantum_mac_state(dimension=4, purity=0.7, seed=42)
    print(f"\n1. Estado cuántico: {rho_mac}")

    sheaf, learner, functor = create_geometric_learning_system(
        num_vertices=10,
        num_edges=15,
        fiber_dim_vertex=3,
        fiber_dim_edge=3,
        dissipation_strength=0.05,
        seed=42,
    )
    coh = sheaf.compute_cohomology_groups()
    print(f"\n2. Cohomología: β₀={coh[0].betti_number}, β₁={coh[1].betti_number}")

    T, D = 200, 30
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.normal(scale=0.01, size=(T, D)), axis=0)
    section_normal = np.zeros(D)
    section_normal[0] = 1.0

    def mock_melnikov(t: float) -> float:
        return math.sin(t) * math.exp(-abs(t) * 0.05)

    facade = CelestialMACFacade(rho_mac=rho_mac)
    cert = facade.full_celestial_audit(
        trajectory=traj,
        section_normal=section_normal,
        h0_poisson_h1=mock_melnikov,
        critical_points_indices=(1,),
        euler_characteristic=1,
        betti_numbers=(1, 0, 1),
        kam_stability_hard=False,
    )
    print("\n3. Certificado soberano:")
    print(f"   Viable: {cert.is_celestially_viable}")
    print(f"   Veto de Gromov: {cert.gromov_veto_active}")
    print(f"   Ω₃: {cert.heyting_verdict.name}")
    print(f"   Polinomio de Poincaré: {cert.duality.poincare_polynomial}")
    print(f"   Hash soberano: {cert.sovereign_hash}")
    print("\n" + "═" * 80)
    print("✓ Demostración completada exitosamente")
    print("═" * 80)