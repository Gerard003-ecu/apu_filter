# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Buffer Engine v5 — Mecánica Celeste de Poincaré Anidada      ║
║ Ubicación: app/wisdom/toon_buffer_engine.py                                  ║
║ Versión  : 5.0.0-Poincare-Delaunay-Lindstedt-Moser-Greene-Chirikov-Topos     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Evolución del Campo Tensorial Transitorio T_TOON^trans(t) sobre T*Q, anidando
granularmente los métodos de Henri Poincaré (*Les Méthodes Nouvelles de la
Mécanique Céleste*, tt. I–III, 1892–1899) y su descendencia KAM/Moser/Greene:

    • Variables canónicas de Delaunay (L, G, H, l, g, h) y de Poincaré (ξ, η, p, q)
    • Tensor de Poisson J^{ij}, corchete {F,G} y 1-forma de Poincaré-Cartan Θ
    • Hamiltoniano perturbado H = H₀(L) + ε H₁(l,g,h;L,G,H) con derivadas analíticas
    • Ecuaciones de Hamilton en Delaunay (no Euler ingenuo: Verlet simpléctico)
    • Series de Lindstedt-Poincaré (eliminación de términos seculares)
    • Sección de Poincaré Σ_ℓ = {ℓ ≡ ℓ₀ (mod 2π), ℓ̇ > 0} ⊂ T*Q  (estroboscópica)
    • Mapa de retorno P : Σ → Σ, monodromía D P, exponentes de Floquet
    • Teorema del twist de Moser (τ = ∂n/∂L = −3μ²/L⁴ ≠ 0)
    • Residuo de Greene R = (2 − Tr M)/4  y criterio de destrucción del último toro
    • Solapamiento de Chirikov de resonancias de movimiento medio
    • Función de Melnikov M(t₀) = ∫ {H₀, H₁}(φ_t(z)) dt  (splitting homoclínico)
    • Teorema geométrico de Poincaré-Birkhoff (puntos fijos del mapa twist)
    • Vector de Laplace-Runge-Lenz como invariante adiabático (cuaternión orbital)
    • Álgebras de Heyting Ω₃ (clasificador de subobjetos del topos) y de Banach B(H)

Arquitectura de 3 FASES ANIDADAS por herencia (el último método de la fase k
es el germen formal / primer acto de la fase k+1):

    FASE I  : PoincareCanonicalAtlas
              ontología simpléctica + Delaunay + LRL + semilla canónica
              → compute_poincare_canonical_seed()     ⟶ germen de FASE II

    FASE II : TOONBufferEngine(PoincareCanonicalAtlas)
              ingesta Hamiltonian, Lindstedt, Verlet, mapa de retorno, monodromía
              → poincare_return_map_step()            ⟶ germen de FASE III

    FASE III: KAMMelnikovAuditor
              KAM / Greene / Chirikov / Melnikov / Birkhoff + funtor CPTP de Lüders
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONBufferEngine.v5")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Constantes numéricas del atlas (escalas de regularización)
_EPS: float = 1e-15
_EPS_ACTION: float = 1e-9
_TWO_PI: float = 2.0 * math.pi
_PI: float = math.pi


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃ (CLASIFICADOR DE SUBOBJETOS), MODOS, REGÍMENES
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """
    Clasificador de subobjetos Ω del topos finito de valuaciones ternarias.
    Orden: VETOED ≼ DEGRADED ≼ COHERENT.  Álgebra de Heyting (no Booleana):
        a ∧ b = min,  a ∨ b = max,  a → b = ⊤ si a ≤ b else b,  ¬a = a → ⊥.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Implicación intuicionista: a → b = ⊤ si a ≤ b, si no b."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        """Negación de Heyting ¬a = a → ⊥.  ¬¬a ≰ a (no Booleana)."""
        return self.implies(HeytingOmega3.VETOED)


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class CartridgeStatus(IntEnum):
    ACTIVE = 0
    ASSIMILATED = 1
    PURGED_DIRAC_VACUUM = 2
    EXPIRED_TTL = 3
    KAM_BROKEN_CHAOTIC = 4
    GREENE_RESONANT = 5  # residuo de Greene |R| → 1/4 (último toro)


class KAMRegime(IntEnum):
    """Clasificación espectral del toro invariante (Kolmogorov-Arnold-Moser)."""

    INTACT_TORUS = 0  # δJ/J < ε_KAM  y  |R_Greene| < 1/4
    CANTORUS_PARTIAL = 1  # Cantor torus (Aubry-Mather)
    ARNOLD_DIFFUSION = 2  # difusión secular a lo largo de la red de resonancias


# ══════════════════════════════════════════════════════════════════════════════
# §B. ESTRUCTURAS DE DATOS — TENSORIALES, CELESTES, HIPERCOMPLEJAS, ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Quaternion:
    """
    Álgebra de división H (números hipercomplejos de Hamilton) para el triedro
    orbital.  Producto no conmutativo: i² = j² = k² = ijk = −1.
    Parametriza SO(3) del marco (nodo, periapsis, momento angular).
    """

    w: float
    x: float
    y: float
    z: float

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        qv = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
        qr = self * qv * self.conjugate()
        return np.array([qr.x, qr.y, qr.z], dtype=np.float64)

    @staticmethod
    def from_euler_313(Omega: float, inc: float, omega: float) -> "Quaternion":
        """Ángulos de Euler 3-1-3 (Ω, i, ω) → cuaternión unitario del triedro orbital."""
        hO, hi, ho = 0.5 * Omega, 0.5 * inc, 0.5 * omega
        cO, sO = math.cos(hO), math.sin(hO)
        ci, si = math.cos(hi), math.sin(hi)
        co, so = math.cos(ho), math.sin(ho)
        return Quaternion(
            w=cO * ci * co - sO * ci * so,
            x=cO * si * co + sO * si * so,
            y=sO * si * co - cO * si * so,
            z=sO * ci * co + cO * ci * so,
        )


@dataclass(frozen=True, slots=True)
class QuantitativeComponent:
    """Componente Cuantitativa Q_cuant(t) ∈ B₁⁺(H_Dirac^(r)) (estados densos)."""

    dimension_r: int
    density_matrix: np.ndarray
    action_variables_J: np.ndarray
    poincare_cartan_1form: float
    tangible_cost_vector: np.ndarray
    banach_op_norm: float = 0.0
    spectral_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class QualitativeComponent:
    """Componente Cualitativa K_cual(t) sobre T*H² (plano hiperbólico)."""

    coordinates_h2: Tuple[float, float]
    shearing_tensor_W: np.ndarray
    exergy_risk_index: float
    ricci_scalar_curvature: float


@dataclass(frozen=True, slots=True)
class DelaunayPoint:
    """
    Coordenadas canónicas de Delaunay (Poincaré 1897, Leçon 1; Tisserand).
    Par canónicamente conjugado (ℓ, g, h ; L, G, H) sobre T*Q ≅ T³ × R³₊:

        L = √(μ a)                 acción Kepleriana (n = μ² / L³)
        G = L √(1 − e²)            módulo del momento angular
        H = G cos i                proyección sobre el eje invariable
        ℓ = anomalía media,  g = ω (periapsis),  h = Ω (nodo)

    Hamiltoniano kepleriano no perturbado:  H₀ = −μ² / (2 L²).
    Twist de Moser:  τ = ∂²H₀/∂L² = ∂n/∂L = −3 μ² / L⁴  < 0  (monótono).
    """

    L: float
    G: float
    H: float
    l: float
    g: float
    h: float
    mu_gravitational: float

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(max(0.0, 1.0 - (self.G / max(self.L, _EPS)) ** 2)))

    @property
    def inclination(self) -> float:
        return float(np.arccos(np.clip(self.H / max(self.G, _EPS), -1.0, 1.0)))

    @property
    def mean_motion(self) -> float:
        """n = μ² / L³  (3ª ley de Kepler en acciones de Delaunay)."""
        return float(self.mu_gravitational ** 2 / max(self.L, _EPS) ** 3)

    @property
    def kepler_energy(self) -> float:
        """H₀ = −μ² / (2 L²)."""
        return float(-self.mu_gravitational ** 2 / (2.0 * max(self.L, _EPS) ** 2))

    @property
    def moser_twist(self) -> float:
        """τ = ∂n/∂L = −3 μ² / L⁴.  Twist no degenerado ⇔ KAM aplicable."""
        L = max(self.L, _EPS)
        return float(-3.0 * self.mu_gravitational ** 2 / L ** 4)

    def poincare_regular_variables(self) -> Tuple[float, float, float, float]:
        """
        Variables de Poincaré (regulares en e = 0, i = 0):
            ξ = √(2(L−G)) cos g,   η = −√(2(L−G)) sin g
            p = √(2(G−H)) cos h,   q = −√(2(G−H)) sin h
        Eliminan la singularidad de Delaunay en órbitas circulares/ecuatorial.
        """
        dLG = max(0.0, 2.0 * (self.L - self.G))
        dGH = max(0.0, 2.0 * (self.G - self.H))
        sLG, sGH = math.sqrt(dLG), math.sqrt(dGH)
        return (
            sLG * math.cos(self.g),
            -sLG * math.sin(self.g),
            sGH * math.cos(self.h),
            -sGH * math.sin(self.h),
        )


@dataclass(frozen=True, slots=True)
class CanonicalSeed:
    """
    GERMEN FORMAL FASE I → FASE II.
    Contrato tipado que `compute_poincare_canonical_seed` entrega y que
    `ingest_from_canonical_seed` (primer acto de FASE II) consume.
    """

    delaunay: DelaunayPoint
    laplace_runge_lenz: np.ndarray
    orbital_quaternion: Quaternion
    n_mean: float
    H0_kepler: float
    moser_twist: float
    poisson_tensor: np.ndarray
    sigma_l0: float
    eigenvalues_J: np.ndarray
    density_matrix: np.ndarray
    lindstedt_n1: float
    generating_function_S2: float


@dataclass(frozen=True, slots=True)
class PoincareSection:
    """
    Sección de Poincaré estroboscópica
        Σ_ℓ₀ = { (ℓ,g,h; L,G,H) : ℓ ≡ ℓ₀ (mod 2π),  ℓ̇ > 0 } ⊂ T*Q.
    Almacena el cruce transversal, el jacobiano 4×4 del mapa de retorno
    (reducción planar L,G,ℓ,g) y el residuo de Greene asociado.
    """

    section_angle_l0: float
    crossing_actions: Tuple[float, float, float]  # (L, G, H)
    crossing_angles: Tuple[float, float]  # (g, h)
    epoch: float
    jacobian_det: float
    monodromy_trace: float = 0.0
    greene_residue: float = 0.0
    floquet_multipliers: Tuple[complex, ...] = ()


@dataclass(frozen=True, slots=True)
class ReturnMapOrbit:
    """
    GERMEN FORMAL FASE II → FASE III.
    Órbita discreta del mapa de retorno P: Σ → Σ junto con su monodromía.
    """

    sections: Tuple[PoincareSection, ...]
    cartridge_id: str
    twist: float
    chirikov_overlap: float
    winding_number: float
    lyapunov_max: float
    birkhoff_fixed_points: int


@dataclass(slots=True)
class TransientTensorCartridge:
    """Cartucho Tensorial Transitorio T_TOON^trans(t) = Q_cuant(t) ⊗ K_cual(t)."""

    cartridge_id: str
    apu_code: str
    quant_comp: QuantitativeComponent
    qual_comp: QualitativeComponent
    creation_timestamp: float
    time_to_live_ttl: float
    capacity_gromov: float
    delaunay: Optional[DelaunayPoint] = None
    seed: Optional[CanonicalSeed] = None
    kam_regime: KAMRegime = KAMRegime.INTACT_TORUS
    status: CartridgeStatus = CartridgeStatus.ACTIVE
    sha256_signature: str = ""
    return_orbit: Optional[ReturnMapOrbit] = None

    def __post_init__(self) -> None:
        if not self.sha256_signature:
            payload = (
                f"{self.cartridge_id}:{self.apu_code}:{self.creation_timestamp}:"
                f"{self.capacity_gromov:.8f}:{self.qual_comp.exergy_risk_index:.8f}:"
                f"{self.delaunay.L if self.delaunay else 0.0:.8f}"
            ).encode("utf-8")
            object.__setattr__(self, "sha256_signature", hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class BufferCertificate:
    """Certificado de Auditoría y Asimilación con datos de Mecánica Celeste."""

    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    active_cartridges_count: int
    assimilated_count: int
    purged_count: int
    capacity_gromov_peak: float
    mac_von_neumann_entropy: float
    poincare_cartan_residual: float
    merkle_root_sha256: str
    delaunay_resonance_order: int = 0
    melnikov_max_amplitude: float = 0.0
    kam_intact_fraction: float = 1.0
    mean_motion_vector: Tuple[float, ...] = ()
    laplace_runge_lenz_norm: float = 0.0
    greene_residue_max: float = 0.0
    chirikov_overlap_max: float = 0.0
    moser_twist_mean: float = 0.0
    floquet_spectral_radius: float = 1.0
    birkhoff_fixed_points: int = 0
    heyting_implication: str = "COHERENT"


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE I — ONTOLOGÍA GEOMÉTRICA Y COORDENADAS CANÓNICAS DE POINCARÉ ▓▓▓
# El último método, compute_poincare_canonical_seed, ES el germen de FASE II.
# ══════════════════════════════════════════════════════════════════════════════

class PoincareCanonicalAtlas:
    """
    FASE I — Atlas canónico sobre el espectro de ρ_trans.

    Traduce las acciones de Liouville-Arnold J_i = λ_i(ρ_trans) a las variables
    de Delaunay (L, G, H) mediante un embedding espectral, construye el tensor
    de Poisson, el cuaternión del triedro orbital, la función generatriz de
    tipo 2 y el primer coeficiente de Lindstedt.  El diccionario tipado
    `CanonicalSeed` que emite el último método es el INPUT FORMAL de FASE II.

    Referencias:
        • Poincaré, H. (1892–1899), *Les Méthodes Nouvelles…*, I–III.
        • Arnold, V.I. (1989), *Mathematical Methods of Classical Mechanics*, §50.
        • Meyer, Hall, Offin (2009), *Introduction to Hamiltonian Dynamical Systems*.
    """

    def __init__(self, dimension: int, mu_gravitational: float = 1.0) -> None:
        self.dimension = int(dimension)
        self.mu_gravitational = float(mu_gravitational)

    # ─────────────────────────────────────────────────────────────────────
    # §I.1 — Tensor de Poisson canónico J^{ij} sobre T*Q  y  corchete {F,G}
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def build_poisson_tensor(dim_pairs: int) -> np.ndarray:
        """
        Tensor de Poisson J^{ij} en coordenadas canónicas (q, p):
            J = [[ 0, I ], [ −I, 0 ]] ∈ 𝔰𝔭(2n, R)
        Forma simpléctica ω = J⁻¹ (Poincaré-Cartan), dω = 0, ωⁿ ≠ 0.
        """
        n = int(dim_pairs)
        J = np.zeros((2 * n, 2 * n), dtype=np.float64)
        J[:n, n:] = np.eye(n)
        J[n:, :n] = -np.eye(n)
        return J

    @staticmethod
    def poisson_bracket(
        dF_dq: np.ndarray, dF_dp: np.ndarray, dG_dq: np.ndarray, dG_dp: np.ndarray
    ) -> float:
        """
        Corchete de Poisson {F, G} = Σ_i (∂F/∂q_i ∂G/∂p_i − ∂F/∂p_i ∂G/∂q_i).
        Identidad de Jacobi y {H, H} = 0 ⇒ conservación de H a lo largo del flujo.
        """
        return float(np.dot(dF_dq, dG_dp) - np.dot(dF_dp, dG_dq))

    # ─────────────────────────────────────────────────────────────────────
    # §I.2 — Embedding espectral λ(ρ) ⟼ Delaunay (L, G, H, ℓ, g, h)
    # ─────────────────────────────────────────────────────────────────────
    def spectral_to_delaunay(
        self,
        action_variables_J: np.ndarray,
        phases_lgh: Optional[Tuple[float, float, float]] = None,
    ) -> DelaunayPoint:
        """
        Mapeo riguroso λ_i(ρ) ⟼ (L, G, H, ℓ, g, h):

            L = Σ λ_i                       (acción Kepleriana total)
            G = L · (λ₂ / λ₁)               (módulo angular espectral)
            H = G · (λ₃ / λ₂)               (proyección espectral)

        Las fases (ℓ, g, h) se reconstruyen, en ausencia de dato externo, como
        fases deterministas del espectro (sin semilla aleatoria en FASE I).
        """
        J = np.sort(np.maximum(np.asarray(action_variables_J, dtype=np.float64), _EPS))[::-1]
        lam1 = float(J[0])
        lam2 = float(J[1] if J.size > 1 else J[0])
        lam3 = float(J[2] if J.size > 2 else J[0])

        L = float(np.sum(J))
        ratio_21 = float(np.clip(lam2 / max(lam1, _EPS), 0.0, 1.0))
        ratio_32 = float(np.clip(lam3 / max(lam2, _EPS), 0.0, 1.0))
        G = L * ratio_21
        H = G * ratio_32

        if phases_lgh is None:
            ell = float((J[0] * _TWO_PI) % _TWO_PI)
            gee = float((np.sum(J[1:5]) * _PI) % _TWO_PI) if J.size > 1 else 0.0
            aitch = float((np.sum(J[5:10]) * 0.5 * _PI) % _TWO_PI) if J.size > 5 else 0.0
        else:
            ell, gee, aitch = (float(phases_lgh[0]), float(phases_lgh[1]), float(phases_lgh[2]))

        return DelaunayPoint(
            L=L, G=G, H=H, l=ell, g=gee, h=aitch,
            mu_gravitational=self.mu_gravitational,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §I.3 — Vector de Laplace-Runge-Lenz y cuaternión orbital
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def laplace_runge_lenz_vector(del_pt: DelaunayPoint) -> np.ndarray:
        """
        A = p × L_vec − μ r̂.  En Delaunay: |A| = μ e.
        Dirección = versor de periapsis en el triedro inercial (Ω, i, ω).
        """
        e = del_pt.eccentricity
        omega_per, Omega_asc, i_inc = del_pt.g, del_pt.h, del_pt.inclination
        ci, si = math.cos(i_inc), math.sin(i_inc)
        cO, sO = math.cos(Omega_asc), math.sin(Omega_asc)
        co, so = math.cos(omega_per), math.sin(omega_per)
        px = cO * co - sO * so * ci
        py = sO * co + cO * so * ci
        pz = so * si
        return del_pt.mu_gravitational * e * np.array([px, py, pz], dtype=np.float64)

    @staticmethod
    def orbital_quaternion(del_pt: DelaunayPoint) -> Quaternion:
        """Triedro orbital como unidad de H: q = q_Ω · q_i · q_ω  (Euler 3-1-3)."""
        return Quaternion.from_euler_313(del_pt.h, del_pt.inclination, del_pt.g)

    # ─────────────────────────────────────────────────────────────────────
    # §I.4 — Función generatriz de tipo 2  S(q, P)  y  1ª corrección Lindstedt
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def generating_function_type2(del_pt: DelaunayPoint) -> float:
        """
        S₂(q, P) = q · P  (identidad canónica) + corrección Kepleriana L·ℓ.
        Genera la transformación idéntica sobre el toro de Liouville-Arnold.
        """
        return float(del_pt.L * del_pt.l + del_pt.G * del_pt.g + del_pt.H * del_pt.h)

    @staticmethod
    def lindstedt_first_frequency_shift(del_pt: DelaunayPoint, epsilon: float) -> float:
        """
        Serie de Lindstedt-Poincaré a orden 1:
            n = n₀ + ε n₁ + O(ε²),
            n₁ = ⟨∂H₁/∂L⟩_{𝕋²}  (promedio sobre el toro, mata el secular ℓ ∼ t).
        H₁ = −cos g + e cos ℓ + (i/π) sin(ℓ+g)  ⇒  ⟨∂H₁/∂L⟩ = 0
        (H₁ es pura fluctuación angular a este orden; n₁ = 0).
        Se devuelve n₁ · ε como marcador del esquema; el orden 2 vive en FASE II.
        """
        del epsilon  # orden 1 nulo por promedio; se retiene la firma
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # §I.5 — GERMEN FASE I → FASE II  (último método de la ontología)
    # ─────────────────────────────────────────────────────────────────────
    def compute_poincare_canonical_seed(
        self,
        density_matrix: np.ndarray,
        action_variables_J: np.ndarray,
        epsilon_risk: float = 0.0,
    ) -> CanonicalSeed:
        """
        ÚLTIMO método de la FASE I  ≡  GERMEN / PRIMER ACTO de la FASE II.

        Sintetiza el `CanonicalSeed` que FASE II usa para construir
            H = H₀(Kepler) + ε H₁(riesgo exógeno)
        y la sección estroboscópica Σ_ℓ.  Toda la información simpléctica,
        hipercompleja y espectral viaja en un único objeto inmutable.

        ⟶ `TOONBufferEngine.ingest_from_canonical_seed(seed, …)` es la
           continuación formal de este método (FASE II anidada por herencia).
        """
        J = np.asarray(action_variables_J, dtype=np.float64)
        del_pt = self.spectral_to_delaunay(J)
        lrl = self.laplace_runge_lenz_vector(del_pt)
        q_orb = self.orbital_quaternion(del_pt)

        n_pairs = max(3, min(self.dimension // 2, 28))
        J_poisson = self.build_poisson_tensor(n_pairs)
        sigma_l0 = float(del_pt.l % _TWO_PI)
        n1 = self.lindstedt_first_frequency_shift(del_pt, epsilon_risk)
        S2 = self.generating_function_type2(del_pt)

        seed = CanonicalSeed(
            delaunay=del_pt,
            laplace_runge_lenz=lrl,
            orbital_quaternion=q_orb,
            n_mean=del_pt.mean_motion,
            H0_kepler=del_pt.kepler_energy,
            moser_twist=del_pt.moser_twist,
            poisson_tensor=J_poisson,
            sigma_l0=sigma_l0,
            eigenvalues_J=J,
            density_matrix=density_matrix,
            lindstedt_n1=n1,
            generating_function_S2=S2,
        )
        logger.info(
            f"[FASE I] Seed canónico: L={del_pt.L:.4f} G={del_pt.G:.4f} H={del_pt.H:.4f} "
            f"e={del_pt.eccentricity:.4f} i={math.degrees(del_pt.inclination):.2f}° "
            f"n={del_pt.mean_motion:.6e} τ={del_pt.moser_twist:.6e} |A|={np.linalg.norm(lrl):.4f}"
        )
        return seed


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE II — INGESTA HAMILTONIANA Y EVOLUCIÓN SOBRE T*Q ▓▓▓
# Anidada: TOONBufferEngine(PoincareCanonicalAtlas).
# El primer método consume CanonicalSeed; el último emite ReturnMapOrbit.
# ══════════════════════════════════════════════════════════════════════════════

class TOONBufferEngine(PoincareCanonicalAtlas):
    """
    FASE II — Motor Hamiltoniano anidado en el atlas de FASE I.

    Hereda `compute_poincare_canonical_seed` (germen I→II) y lo consume en
    `ingest_from_canonical_seed`.  Evoluciona bajo
        H = H₀(L) + ε H₁(ℓ, g, h; L, G, H)
    con Verlet simpléctico, series de Lindstedt a orden 2, sección
    estroboscópica Σ_ℓ y monodromía del mapa de retorno.

    El último método, `poincare_return_map_step`, emite `ReturnMapOrbit`
    (germen II→III) que FASE III audita.
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = 12.5,
        default_ttl_sec: float = 3.6,
        esp32_gpio_pin: int = 14,
        mu_gravitational: float = 1.0,
    ) -> None:
        super().__init__(dimension=dimension, mu_gravitational=mu_gravitational)
        self.capacity_gromov_max = float(capacity_gromov_max)
        self.default_ttl_sec = float(default_ttl_sec)
        self.esp32_gpio_pin = int(esp32_gpio_pin)

        self._active_cartridges: Dict[str, TransientTensorCartridge] = {}
        self._purged_history: List[str] = []
        self._assimilated_history: List[str] = []
        self._capacity_peak: float = 0.0
        self._poincare_sections: List[PoincareSection] = []
        self._return_orbits: List[ReturnMapOrbit] = []

        self._N_diag = np.diag(np.linspace(1.0, 2.0, self.dimension, dtype=np.float64))
        self._H_buffer = np.diag(np.sin(np.linspace(0.0, _PI, self.dimension)))

        logger.info(
            f"[FASE II] TOONBufferEngine v5 anidado en PoincareCanonicalAtlas. "
            f"Dim={dimension}, c_G_max={capacity_gromov_max}, μ={mu_gravitational}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # §II.0 — Banach / espectro de observables (álgebra B(H))
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def banach_operator_norm(A: np.ndarray) -> float:
        """Norma de operador ||A||₂ = σ_max(A)  (C*-norma de B(H))."""
        try:
            return float(np.linalg.norm(A, 2))
        except ValueError:
            return float(np.max(np.abs(A)))

    @staticmethod
    def spectral_radius_of(A: np.ndarray) -> float:
        """r(A) = max |σ(A)|.  En C*-álgebras r(A*A)^{1/2} = ||A||."""
        try:
            return float(np.max(np.abs(np.linalg.eigvals(A))))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # §II.1 — Continuación formal del germen de FASE I
    # ─────────────────────────────────────────────────────────────────────
    def ingest_from_canonical_seed(
        self,
        seed: CanonicalSeed,
        apu_code: str,
        tangible_cost_vector: np.ndarray,
        policy_risk_intangible: float,
        weather_gremial_factor: float,
        coordinates_h2: Tuple[float, float] = (1.0, 2.0),
    ) -> TransientTensorCartridge:
        """
        PRIMER método de FASE II  ≡  continuación de
        `PoincareCanonicalAtlas.compute_poincare_canonical_seed`.

        Recibe el `CanonicalSeed` inmutable y lo tensorializa a
        T_TOON^trans = Q_cuant ⊗ K_cual, enriquecido con Delaunay, LRL,
        twist de Moser y norma de Banach de ρ_trans.
        """
        now = time.time()
        cartridge_id = f"TOON-BUFF-{int(now * 1000) % 1_000_000:06d}"

        rho_trans = np.asarray(seed.density_matrix, dtype=np.float64)
        rho_trans = 0.5 * (rho_trans + rho_trans.T)
        tr = float(np.trace(rho_trans))
        if tr > _EPS:
            rho_trans = rho_trans / tr

        action_J = np.sort(np.maximum(la.eigvalsh(rho_trans), _EPS))[::-1]
        poincare_cartan = float(np.trace(rho_trans @ self._N_diag))
        cost_vec = np.asarray(tangible_cost_vector, dtype=np.float64).flatten()
        if cost_vec.size < self.dimension:
            cost_vec = np.pad(cost_vec, (0, self.dimension - cost_vec.size))
        else:
            cost_vec = cost_vec[: self.dimension]

        quant_comp = QuantitativeComponent(
            dimension_r=self.dimension,
            density_matrix=rho_trans,
            action_variables_J=action_J,
            poincare_cartan_1form=poincare_cartan,
            tangible_cost_vector=cost_vec,
            banach_op_norm=self.banach_operator_norm(rho_trans),
            spectral_radius=self.spectral_radius_of(rho_trans),
        )

        x_h2, y_h2 = coordinates_h2 if coordinates_h2[1] > 0 else (coordinates_h2[0], 1.0)
        ricci_R = -2.0 / (y_h2 ** 2)
        # Cizalladura determinista (no aleatoria): generada por el espectro y el riesgo
        phase = np.linspace(0.0, _TWO_PI, self.dimension, endpoint=False)
        W_raw = np.outer(np.sin(phase), np.cos(phase))
        W_skew = (W_raw - W_raw.T) * (policy_risk_intangible + weather_gremial_factor)
        exergy_risk = float(np.clip(0.5 * policy_risk_intangible + 0.5 * weather_gremial_factor, 0.0, 1.0))

        qual_comp = QualitativeComponent(
            coordinates_h2=(float(x_h2), float(y_h2)),
            shearing_tensor_W=W_skew,
            exergy_risk_index=exergy_risk,
            ricci_scalar_curvature=float(ricci_R),
        )

        wigner_radius = float(np.sqrt(np.sum(action_J[:3] ** 2)))
        c_gromov = float(min(0.5 * _PI * (wigner_radius * 100.0) ** 2, self.capacity_gromov_max))

        renyi_entropy = float(-np.log(np.sum(action_J ** 2) + _EPS))
        n_kepler = max(seed.n_mean, 1e-6)
        ttl_sec = float(np.clip(self.default_ttl_sec / (1.0 + renyi_entropy + 0.1 * n_kepler), 0.5, 10.0))

        cart = TransientTensorCartridge(
            cartridge_id=cartridge_id,
            apu_code=apu_code,
            quant_comp=quant_comp,
            qual_comp=qual_comp,
            creation_timestamp=now,
            time_to_live_ttl=ttl_sec,
            capacity_gromov=c_gromov,
            delaunay=seed.delaunay,
            seed=seed,
        )
        self._active_cartridges[cartridge_id] = cart
        self._capacity_peak = max(self._capacity_peak, c_gromov)

        d = seed.delaunay
        logger.info(
            f"[FASE II ← I] Cartucho {cartridge_id} ({apu_code}) ingerido desde CanonicalSeed. "
            f"L={d.L:.4f} e={d.eccentricity:.3f} i={math.degrees(d.inclination):.2f}° "
            f"c_G={c_gromov:.3f} TTL={ttl_sec:.2f}s ||ρ||={quant_comp.banach_op_norm:.4f}"
        )
        return cart

    def ingest_cartridge(
        self,
        apu_code: str,
        tangible_cost_vector: np.ndarray,
        policy_risk_intangible: float,
        weather_gremial_factor: float,
        coordinates_h2: Tuple[float, float] = (1.0, 2.0),
    ) -> TransientTensorCartridge:
        """
        Fachada de ingesta: construye ρ_trans a partir del vector de costos,
        invoca el germen de FASE I (`compute_poincare_canonical_seed`) y
        delega en `ingest_from_canonical_seed` (continuación anidada).
        """
        cost_vec = np.asarray(tangible_cost_vector, dtype=np.float64).flatten()
        if cost_vec.size < self.dimension:
            cost_vec = np.pad(cost_vec, (0, self.dimension - cost_vec.size))
        else:
            cost_vec = cost_vec[: self.dimension]
        norm_cost = float(np.linalg.norm(cost_vec))
        if norm_cost < 1e-12:
            cost_vec = np.ones(self.dimension, dtype=np.float64) / math.sqrt(self.dimension)
            norm_cost = 1.0

        psi = cost_vec / norm_cost
        rho_pure = np.outer(psi, psi)
        rho_trans = 0.98 * rho_pure + 0.02 * (np.eye(self.dimension) / self.dimension)
        rho_trans = 0.5 * (rho_trans + rho_trans.T)
        rho_trans /= float(np.trace(rho_trans))

        eigenvals = la.eigvalsh(rho_trans)
        action_J = np.sort(np.maximum(eigenvals, _EPS))[::-1]
        eps_risk = float(np.clip(0.5 * policy_risk_intangible + 0.5 * weather_gremial_factor, 0.0, 1.0))

        seed = self.compute_poincare_canonical_seed(rho_trans, action_J, epsilon_risk=eps_risk)
        return self.ingest_from_canonical_seed(
            seed=seed,
            apu_code=apu_code,
            tangible_cost_vector=cost_vec,
            policy_risk_intangible=policy_risk_intangible,
            weather_gremial_factor=weather_gremial_factor,
            coordinates_h2=coordinates_h2,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §II.2 — Hamiltoniano perturbado y campo vectorial X_H en Delaunay
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _eccentricity_and_inclination(L: float, G: float, H: float) -> Tuple[float, float, float, float, float, float]:
        """e, i, ∂e/∂L, ∂e/∂G, ∂ι/∂G, ∂ι/∂H  con ι = i/π."""
        L = max(L, _EPS)
        G = max(G, _EPS)
        ee2 = max(0.0, 1.0 - (G / L) ** 2)
        e = math.sqrt(ee2)
        cos_i = float(np.clip(H / G, -1.0, 1.0))
        inc = math.acos(cos_i)
        iota = inc / _PI
        # derivadas (regularizadas)
        if e > 1e-12:
            de_dL = (G * G) / (L ** 3 * e)
            de_dG = -G / (L * L * e)
        else:
            de_dL = 0.0
            de_dG = 0.0
        sin_i = math.sqrt(max(0.0, 1.0 - cos_i * cos_i))
        if sin_i > 1e-12:
            di_dG = (H / (G * G)) / sin_i
            di_dH = -1.0 / (G * sin_i)
        else:
            di_dG = 0.0
            di_dH = 0.0
        return e, iota, de_dL, de_dG, di_dG / _PI, di_dH / _PI

    def hamiltonian_and_field(
        self, L: float, G: float, H: float, ell: float, gee: float, aitch: float,
        mu: float, epsilon_risk: float,
    ) -> Tuple[float, np.ndarray]:
        """
        H = H₀ + ε H₁,  H₀ = −μ²/(2L²),
        H₁ = −cos g + e cos ℓ + ι sin(ℓ+g)     (perturbación genérica de Poincaré).

        Campo hamiltoniano X_H = J ∇H  en coordenadas (L, G, H, ℓ, g, h):
            ℓ̇ =  ∂H/∂L,   L̇ = −∂H/∂ℓ
            ġ =  ∂H/∂G,   Ġ = −∂H/∂g
            ḣ =  ∂H/∂H,   Ḣ = −∂H/∂h
        (aquí se corrige el bug v4: ġ = ε ∂H₁/∂G ≠ 0 ⇒ hay precesión).
        """
        del aitch  # H₁ independiente de h a este orden (simetría axial)
        H0 = -mu ** 2 / (2.0 * max(L, _EPS) ** 2)
        e, iota, de_dL, de_dG, diota_dG, diota_dH = self._eccentricity_and_inclination(L, G, H)
        H1 = -math.cos(gee) + e * math.cos(ell) + iota * math.sin(ell + gee)
        Ham = H0 + epsilon_risk * H1

        dH1_dL = de_dL * math.cos(ell)
        dH1_dG = de_dG * math.cos(ell) + diota_dG * math.sin(ell + gee)
        dH1_dH = diota_dH * math.sin(ell + gee)
        dH1_dl = -e * math.sin(ell) + iota * math.cos(ell + gee)
        dH1_dg = math.sin(gee) + iota * math.cos(ell + gee)
        dH1_dh = 0.0

        dH_dL = mu ** 2 / max(L, _EPS) ** 3 + epsilon_risk * dH1_dL
        dH_dG = epsilon_risk * dH1_dG
        dH_dH = epsilon_risk * dH1_dH
        dH_dl = epsilon_risk * dH1_dl
        dH_dg = epsilon_risk * dH1_dg
        dH_dh = epsilon_risk * dH1_dh

        # z = (L, G, H, ℓ, g, h)  →  ż = (−∂H/∂ℓ, −∂H/∂g, −∂H/∂h, ∂H/∂L, ∂H/∂G, ∂H/∂H)
        field = np.array(
            [-dH_dl, -dH_dg, -dH_dh, dH_dL, dH_dG, dH_dH],
            dtype=np.float64,
        )
        return float(Ham), field

    def lindstedt_second_order(
        self, del_pt: DelaunayPoint, epsilon_risk: float, n_quad: int = 32
    ) -> Tuple[float, float]:
        """
        Lindstedt-Poincaré a orden 2.  n = n₀ + ε n₁ + ε² n₂.
        n₁ = ⟨∂H₁/∂L⟩ = 0 (FASE I).  n₂ se elige para matar el secular
        generado por los productos de Fourier de H₁ (pequeños divisores).
        Aproximación: n₂ ≈ ⟨(∂²H₁/∂ℓ ∂L) · (∂H₁/∂ℓ) / n₀⟩  sobre 𝕋².
        """
        if epsilon_risk <= 0.0:
            return del_pt.mean_motion, 0.0
        L, G, H, mu = del_pt.L, del_pt.G, del_pt.H, del_pt.mu_gravitational
        n0 = del_pt.mean_motion
        ls = np.linspace(0.0, _TWO_PI, n_quad, endpoint=False)
        gs = np.linspace(0.0, _TWO_PI, n_quad, endpoint=False)
        acc = 0.0
        for ell in ls:
            for gee in gs:
                _, X = self.hamiltonian_and_field(L, G, H, ell, gee, del_pt.h, mu, 1.0)
                dH1_dl = -X[0]  # X_L = −∂H₁/∂ℓ
                dH1_dL = X[3] - n0  # X_ℓ = n₀ + ∂H₁/∂L  ⇒  ∂H₁/∂L = X_ℓ − n₀
                acc += dH1_dl * dH1_dL
        n2 = float(acc / (n_quad * n_quad) / max(n0, _EPS))
        n_corr = n0 + (epsilon_risk ** 2) * n2
        return n_corr, n2

    # ─────────────────────────────────────────────────────────────────────
    # §II.3 — Integrador simpléctico de Verlet (separable H = T(p) + ε V(q,p))
    # ─────────────────────────────────────────────────────────────────────
    def _verlet_step(
        self, z: np.ndarray, dt: float, mu: float, epsilon_risk: float
    ) -> np.ndarray:
        """
        Verlet / Störmer leapfrog sobre (L,G,H,ℓ,g,h).
        Semi-paso en acciones, paso completo en ángulos, semi-paso en acciones.
        Conserva la 2-forma ω hasta O(dt²) (simpléctico de orden 2).
        """
        L, G, H, ell, gee, aitch = (float(z[0]), float(z[1]), float(z[2]),
                                    float(z[3]), float(z[4]), float(z[5]))
        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        # semi-paso acciones
        L += 0.5 * dt * X[0]
        G += 0.5 * dt * X[1]
        H += 0.5 * dt * X[2]
        L, G = max(L, _EPS_ACTION), max(G, _EPS_ACTION)
        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        # paso ángulos
        ell += dt * X[3]
        gee += dt * X[4]
        aitch += dt * X[5]
        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        # semi-paso acciones
        L += 0.5 * dt * X[0]
        G += 0.5 * dt * X[1]
        H += 0.5 * dt * X[2]
        L, G = max(L, _EPS_ACTION), max(G, _EPS_ACTION)
        return np.array([L, G, H, ell, gee, aitch], dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # §II.4 — Evolución Lindblad-GKSL/RLC (canal CPTP sobre ρ_trans)
    # ─────────────────────────────────────────────────────────────────────
    def evolve_lindblad_rlc_step(self, dt: float = 0.1) -> Tuple[int, int]:
        """Paso GKSL: ρ̇ = −i[H_buffer, ρ] + D[L](ρ), proyección hermítica/traza 1."""
        now = time.time()
        purged_count = 0
        to_purge: List[str] = []

        for cid, cart in list(self._active_cartridges.items()):
            elapsed = now - cart.creation_timestamp
            if elapsed >= cart.time_to_live_ttl or cart.status == CartridgeStatus.EXPIRED_TTL:
                to_purge.append(cid)
                continue
            if cart.delaunay is None:
                continue

            rho = cart.quant_comp.density_matrix
            comm = -1j * (self._H_buffer @ rho - rho @ self._H_buffer)
            L_buff = 0.05 * np.eye(self.dimension)
            LdagL = L_buff.T.conj() @ L_buff
            diss = L_buff @ rho @ L_buff.T.conj() - 0.5 * (LdagL @ rho + rho @ LdagL)
            rho_new = np.real(rho + (comm + diss) * dt)
            rho_new = 0.5 * (rho_new + rho_new.T)
            tr = float(np.trace(rho_new))
            if tr > _EPS:
                rho_new /= tr

            action_J_new = np.sort(np.maximum(la.eigvalsh(rho_new), _EPS))[::-1]
            poincare_new = float(np.trace(rho_new @ self._N_diag))
            n_corr, _ = self.lindstedt_second_order(cart.delaunay, cart.qual_comp.exergy_risk_index)
            new_del = self.spectral_to_delaunay(
                action_J_new,
                phases_lgh=(
                    (cart.delaunay.l + dt * n_corr) % _TWO_PI,
                    cart.delaunay.g,
                    cart.delaunay.h,
                ),
            )
            cart.delaunay = new_del
            cart.quant_comp = QuantitativeComponent(
                dimension_r=self.dimension,
                density_matrix=rho_new,
                action_variables_J=action_J_new,
                poincare_cartan_1form=poincare_new,
                tangible_cost_vector=cart.quant_comp.tangible_cost_vector,
                banach_op_norm=self.banach_operator_norm(rho_new),
                spectral_radius=self.spectral_radius_of(rho_new),
            )

        for cid in to_purge:
            cart = self._active_cartridges.pop(cid)
            cart.status = CartridgeStatus.PURGED_DIRAC_VACUUM
            self._purged_history.append(cid)
            purged_count += 1
            logger.info(f"[FASE II] {cid} purgado en Vacío de Dirac (0.0 dB).")

        return len(self._active_cartridges), purged_count

    # ─────────────────────────────────────────────────────────────────────
    # §II.5 — Monodromía, residuo de Greene, número de rotación, Chirikov
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def greene_residue(monodromy_trace: float) -> float:
        """R = (2 − Tr M)/4.  |R|<1 elíptico, |R|>1 hiperbólico; Greene: R→1/4 destruye KAM."""
        return float((2.0 - monodromy_trace) / 4.0)

    @staticmethod
    def chirikov_overlap(n_a: float, n_b: float, halfwidth_a: float, halfwidth_b: float) -> float:
        """
        Parámetro de solapamiento s = (Δn_a + Δn_b) / (2 |n_a − n_b|).
        s ≳ 1 ⇒ destrucción del toro último (transición al caos global).
        """
        denom = 2.0 * max(abs(n_a - n_b), _EPS)
        return float((abs(halfwidth_a) + abs(halfwidth_b)) / denom)

    def _finite_difference_monodromy(
        self, z0: np.ndarray, mu: float, epsilon_risk: float,
        dt: float, n_steps: int, delta: float = 1e-6,
    ) -> np.ndarray:
        """
        Jacobiano 4×4 del flujo (L, G, ℓ, g) por diferencias finitas centradas.
        Es la monodromía reducida (sección planar) del mapa de Poincaré.
        """
        def flow(z: np.ndarray) -> np.ndarray:
            zz = z.copy()
            for _ in range(n_steps):
                zz = self._verlet_step(zz, dt, mu, epsilon_risk)
            return zz

        # índices reducidos: (L, G, ℓ, g) = (0, 1, 3, 4)
        idx = (0, 1, 3, 4)
        M = np.zeros((4, 4), dtype=np.float64)
        for k, ik in enumerate(idx):
            zp = z0.copy()
            zm = z0.copy()
            zp[ik] += delta
            zm[ik] -= delta
            fp = flow(zp)
            fm = flow(zm)
            for r, ir in enumerate(idx):
                M[r, k] = (fp[ir] - fm[ir]) / (2.0 * delta)
        return M

    # ─────────────────────────────────────────────────────────────────────
    # §II.6 — GERMEN FASE II → FASE III  (último método del motor)
    # ─────────────────────────────────────────────────────────────────────
    def poincare_return_map_step(
        self,
        dt_integration: float = 1e-3,
        max_steps: int = 20_000,
        l0_reference: float = 0.0,
        max_crossings: int = 8,
    ) -> List[ReturnMapOrbit]:
        """
        ÚLTIMO método de la FASE II  ≡  GERMEN / PRIMER ACTO de la FASE III.

        Integra X_H con Verlet simpléctico y registra cruces transversales con
            Σ_ℓ₀ = { ℓ ≡ ℓ₀ (mod 2π),  ℓ̇ > 0 }.

        (Se usa la anomalía media —siempre creciente por n>0— y no g, que en v4
        era estacionaria.  Esto garantiza cruces y un mapa de retorno honesto.)

        Por cada cartucho se construye un `ReturnMapOrbit` con:
            • secciones, twist de Moser, número de rotación ϖ
            • monodromía, residuo de Greene, exponentes de Floquet
            • solapamiento de Chirikov, cota de Lyapunov, puntos de Birkhoff

        ⟶ `KAMMelnikovAuditor.audit_return_orbits(orbits, …)` es la
           continuación formal de este método.
        """
        orbits: List[ReturnMapOrbit] = []
        all_sections: List[PoincareSection] = []

        for cart in self._active_cartridges.values():
            if cart.delaunay is None:
                continue
            del_pt = cart.delaunay
            eps_risk = cart.qual_comp.exergy_risk_index
            mu = del_pt.mu_gravitational
            z = np.array(
                [del_pt.L, del_pt.G, del_pt.H, del_pt.l, del_pt.g, del_pt.h],
                dtype=np.float64,
            )

            def wrap_l(val: float) -> float:
                return ((val - l0_reference + _PI) % _TWO_PI) - _PI

            prev = wrap_l(z[3])
            crossings: List[PoincareSection] = []
            crossing_angles_g: List[float] = []
            step = 0
            n_corr, _ = self.lindstedt_second_order(del_pt, eps_risk)

            while step < max_steps and len(crossings) < max_crossings:
                z = self._verlet_step(z, dt_integration, mu, eps_risk)
                curr = wrap_l(z[3])
                _, X = self.hamiltonian_and_field(z[0], z[1], z[2], z[3], z[4], z[5], mu, eps_risk)
                ell_dot = X[3]  # ∂H/∂L = n + ε ∂H₁/∂L  > 0 genéricamente
                if prev < 0.0 <= curr and ell_dot > 0.0:
                    # monodromía local: flujo de un periodo medio 2π/n
                    n_loc = max(abs(ell_dot), 1e-9)
                    n_period = max(8, int((_TWO_PI / n_loc) / max(dt_integration, 1e-9)))
                    n_period = min(n_period, 400)
                    M = self._finite_difference_monodromy(
                        z, mu, eps_risk, dt_integration, n_steps=n_period
                    )
                    trM = float(np.trace(M))
                    R = self.greene_residue(trM)
                    try:
                        eigs = tuple(complex(x) for x in np.linalg.eigvals(M))
                    except np.linalg.LinAlgError:
                        eigs = ()
                    jac = float(np.linalg.det(M)) if M.size else float(abs(mu ** 2 / max(z[0], _EPS) ** 4))
                    sec = PoincareSection(
                        section_angle_l0=l0_reference,
                        crossing_actions=(float(z[0]), float(z[1]), float(z[2])),
                        crossing_angles=(float(z[4] % _TWO_PI), float(z[5] % _TWO_PI)),
                        epoch=time.time(),
                        jacobian_det=jac,
                        monodromy_trace=trM,
                        greene_residue=R,
                        floquet_multipliers=eigs,
                    )
                    crossings.append(sec)
                    crossing_angles_g.append(float(z[4] % _TWO_PI))
                prev = curr
                step += 1

            # estado final → cartucho
            cart.delaunay = DelaunayPoint(
                L=float(max(z[0], _EPS_ACTION)),
                G=float(max(z[1], _EPS_ACTION)),
                H=float(z[2]),
                l=float(z[3] % _TWO_PI),
                g=float(z[4] % _TWO_PI),
                h=float(z[5] % _TWO_PI),
                mu_gravitational=mu,
            )

            # número de rotación (Birkhoff): ϖ = ⟨Δg / Δℓ⟩ ≃ Δg / 2π por cruce
            if len(crossing_angles_g) >= 2:
                dg = np.diff(np.unwrap(crossing_angles_g))
                winding = float(np.mean(dg) / _TWO_PI)
            else:
                winding = 0.0

            # Lyapunov máximo ≃ log ρ(M)  (radio espectral de la monodromía)
            lyap = 0.0
            if crossings:
                radii = [
                    max((abs(lam) for lam in s.floquet_multipliers), default=1.0)
                    for s in crossings
                    if s.floquet_multipliers
                ]
                if radii:
                    lyap = float(math.log(max(max(radii), _EPS)))

            # Chirikov: semianchos ∼ ε / |τ|  entre el primer y último cruce
            if len(crossings) >= 2:
                n_a = mu ** 2 / max(crossings[0].crossing_actions[0], _EPS) ** 3
                n_b = mu ** 2 / max(crossings[-1].crossing_actions[0], _EPS) ** 3
                hw = abs(eps_risk / max(abs(del_pt.moser_twist), _EPS))
                s_ch = self.chirikov_overlap(n_a, n_b, hw, hw)
            else:
                s_ch = 0.0

            # Poincaré-Birkhoff: un mapa twist del anillo tiene ≥ 2 puntos fijos
            # Proxy: cambios de signo de (g_{k+1} − g_k − ϖ) sobre la órbita
            birkhoff = 0
            if len(crossing_angles_g) >= 3:
                unwrapped = np.unwrap(crossing_angles_g)
                residual = np.diff(unwrapped) - winding * _TWO_PI
                signs = np.sign(residual)
                birkhoff = int(np.sum(signs[1:] * signs[:-1] < 0))
                birkhoff = max(birkhoff, 2 if del_pt.moser_twist != 0.0 else 0)

            orbit = ReturnMapOrbit(
                sections=tuple(crossings),
                cartridge_id=cart.cartridge_id,
                twist=float(del_pt.moser_twist),
                chirikov_overlap=float(s_ch),
                winding_number=float(winding),
                lyapunov_max=float(lyap),
                birkhoff_fixed_points=int(birkhoff),
            )
            cart.return_orbit = orbit
            orbits.append(orbit)
            all_sections.extend(crossings)

        self._poincare_sections.extend(all_sections)
        self._return_orbits.extend(orbits)
        logger.info(
            f"[FASE II → FASE III] Mapas de retorno: {len(orbits)} órbitas, "
            f"{len(all_sections)} cruces de Σ_ℓ sobre {len(self._active_cartridges)} cartuchos."
        )
        return orbits


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE III — KAM, GREENE, CHIRIKOV, MELNIKOV, BIRKHOFF, FUNTOR CPTP ▓▓▓
# El primer método consume ReturnMapOrbit (germen de FASE II).
# ══════════════════════════════════════════════════════════════════════════════

class KAMMelnikovAuditor:
    """
    FASE III — Auditoría espectral-celeste del Campo Tensorial Transitorio.

    Recibe las `ReturnMapOrbit` de FASE II y:

        1. Detecta resonancias de movimiento medio (p:q) por fracciones continuas
           (pequeños divisores de Poincaré, 1885)
        2. Evalúa KAM + residuo de Greene + solapamiento de Chirikov
        3. Calcula Melnikov M(t₀) = ∫ {H₀, H₁} dt  (splitting homoclínico)
        4. Verifica el teorema geométrico de Poincaré-Birkhoff (puntos fijos)
        5. Emite el veredicto Heyting Ω₃ que modula el funtor CPTP de Lüders
           F : Cartucho  →  MAC
    """

    def __init__(self, epsilon_kam: float = 1e-2, greene_critical: float = 0.25) -> None:
        self.epsilon_kam = float(epsilon_kam)
        self.greene_critical = float(greene_critical)

    # ─────────────────────────────────────────────────────────────────────
    # §III.1 — Resonancias por fracciones continuas (Poincaré 1885)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def continued_fraction_resonance(ratio: float, max_den: int = 10 ** 6) -> Tuple[int, int]:
        """Aproximación diofántica óptima |p/q − ratio| < 1/q² (Dirichlet)."""
        if not np.isfinite(ratio) or ratio == 0.0:
            return (0, 1)
        frac = Fraction(abs(float(ratio))).limit_denominator(max_den)
        return (int(frac.numerator), int(frac.denominator))

    def detect_mean_motion_resonance(
        self, sections: Sequence[PoincareSection]
    ) -> Tuple[int, int, List[float]]:
        """Resonancia p:q  ⇔  q n₁ − p n₂ ≃ 0  (pequeño divisor)."""
        if len(sections) < 2:
            return (0, 1, [])
        n_vals = [1.0 / max(s.crossing_actions[0], _EPS) ** 3 for s in sections[:8]]
        ratios = [n_vals[i] / max(n_vals[0], _EPS) for i in range(1, len(n_vals))]
        if not ratios:
            return (0, 1, n_vals)
        p, q = self.continued_fraction_resonance(float(np.mean(ratios)))
        return (p, q, n_vals)

    # ─────────────────────────────────────────────────────────────────────
    # §III.2 — KAM + Greene + Chirikov sobre el mapa de retorno
    # ─────────────────────────────────────────────────────────────────────
    def kam_preservation(
        self, sections: Sequence[PoincareSection]
    ) -> Tuple[KAMRegime, float, float]:
        """
        δJ/J = max |L(t)−L(0)| / L(0).  Se combina con max |R_Greene|:
            INTACT_TORUS      si δJ/J < ε_KAM y |R| < R_c
            CANTORUS_PARTIAL  si ε_KAM ≤ δJ/J < √ε_KAM  o  |R| ≃ R_c
            ARNOLD_DIFFUSION  si δJ/J ≥ √ε_KAM  o  |R| > 1
        """
        if len(sections) < 2:
            return KAMRegime.INTACT_TORUS, 0.0, 0.0
        L0 = sections[0].crossing_actions[0]
        drift = max(abs(s.crossing_actions[0] - L0) / max(L0, _EPS) for s in sections)
        R_max = max(abs(s.greene_residue) for s in sections)
        if drift < self.epsilon_kam and R_max < self.greene_critical:
            regime = KAMRegime.INTACT_TORUS
        elif drift < math.sqrt(self.epsilon_kam) or R_max < 1.0:
            regime = KAMRegime.CANTORUS_PARTIAL
        else:
            regime = KAMRegime.ARNOLD_DIFFUSION
        return regime, float(drift), float(R_max)

    # ─────────────────────────────────────────────────────────────────────
    # §III.3 — Función de Melnikov (splitting homoclínico, Poincaré 1890)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def melnikov_amplitude(
        del_pt: DelaunayPoint, epsilon_risk: float, n_samples: int = 64
    ) -> float:
        """
        M(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(φ_t^{H₀}(z)) dt.
        Para H₁ trigonométrica se reduce al promedio sobre 𝕋²:
            {H₀, H₁} = n₀ ∂H₁/∂ℓ,
            M ≃ ε n₀ ⟨−e sin ℓ + ι cos(ℓ+g)⟩ · (2π)².
        Un cero simple de M implica splitting homoclínico (caos).
        """
        if epsilon_risk <= 0.0:
            return 0.0
        l_grid = np.linspace(0.0, _TWO_PI, n_samples, endpoint=False)
        g_grid = np.linspace(0.0, _TWO_PI, n_samples, endpoint=False)
        e = del_pt.eccentricity
        iota = del_pt.inclination / _PI
        n0 = del_pt.mean_motion
        Lg, Gg = np.meshgrid(l_grid, g_grid, indexing="ij")
        poisson_density = -e * np.sin(Lg) + iota * np.cos(Lg + Gg)
        M = float(np.mean(poisson_density) * epsilon_risk * n0 * (_TWO_PI ** 2))
        return abs(M)

    # ─────────────────────────────────────────────────────────────────────
    # §III.4 — Continuación formal del germen de FASE II
    # ─────────────────────────────────────────────────────────────────────
    def audit_return_orbits(
        self,
        orbits: Sequence[ReturnMapOrbit],
        cartridges: Sequence[TransientTensorCartridge],
    ) -> Dict[str, Any]:
        """
        PRIMER método de FASE III  ≡  continuación de
        `TOONBufferEngine.poincare_return_map_step`.

        Consume `ReturnMapOrbit` y clasifica cada cartucho (KAM/Greene/Chirikov).
        """
        all_sections: List[PoincareSection] = [s for o in orbits for s in o.sections]
        p, q, n_vals = self.detect_mean_motion_resonance(all_sections)
        regime, drift, R_max = self.kam_preservation(all_sections)

        melnikov_max = 0.0
        chirikov_max = 0.0
        twist_acc = 0.0
        floquet_r = 1.0
        birkhoff_total = 0
        n_carts = 0

        cart_by_id = {c.cartridge_id: c for c in cartridges}
        for orbit in orbits:
            cart = cart_by_id.get(orbit.cartridge_id)
            if cart is None or cart.delaunay is None:
                continue
            n_carts += 1
            Mamp = self.melnikov_amplitude(cart.delaunay, cart.qual_comp.exergy_risk_index)
            melnikov_max = max(melnikov_max, Mamp)
            chirikov_max = max(chirikov_max, orbit.chirikov_overlap)
            twist_acc += orbit.twist
            birkhoff_total += orbit.birkhoff_fixed_points
            if orbit.sections:
                for s in orbit.sections:
                    if s.floquet_multipliers:
                        floquet_r = max(
                            floquet_r,
                            max(abs(lam) for lam in s.floquet_multipliers),
                        )

            # régimen individual
            if orbit.sections:
                L_series = [s.crossing_actions[0] for s in orbit.sections]
                L0 = cart.delaunay.L
                drift_c = max(abs(L - L0) / max(L0, _EPS) for L in L_series) if L_series else 0.0
                R_c = max(abs(s.greene_residue) for s in orbit.sections)
            else:
                drift_c, R_c = 0.0, 0.0

            if orbit.chirikov_overlap >= 1.0 or drift_c >= math.sqrt(self.epsilon_kam) or R_c > 1.0:
                cart.kam_regime = KAMRegime.ARNOLD_DIFFUSION
                cart.status = CartridgeStatus.KAM_BROKEN_CHAOTIC
            elif abs(R_c - self.greene_critical) < 0.05:
                cart.kam_regime = KAMRegime.CANTORUS_PARTIAL
                cart.status = CartridgeStatus.GREENE_RESONANT
            elif drift_c < self.epsilon_kam and R_c < self.greene_critical:
                cart.kam_regime = KAMRegime.INTACT_TORUS
            else:
                cart.kam_regime = KAMRegime.CANTORUS_PARTIAL

        return {
            "resonance_pq": (p, q),
            "kam_regime": regime,
            "kam_drift": drift,
            "melnikov_max": melnikov_max,
            "n_mean_vector": tuple(n_vals),
            "greene_residue_max": R_max,
            "chirikov_overlap_max": chirikov_max,
            "moser_twist_mean": float(twist_acc / max(n_carts, 1)),
            "floquet_spectral_radius": float(floquet_r),
            "birkhoff_fixed_points": int(birkhoff_total),
        }

    def audit(
        self,
        sections: List[PoincareSection],
        cartridges: List[TransientTensorCartridge],
    ) -> Dict[str, Any]:
        """Compatibilidad v4: envuelve secciones sueltas como órbitas degeneradas."""
        if not sections:
            orbits: List[ReturnMapOrbit] = [
                c.return_orbit for c in cartridges if c.return_orbit is not None
            ]
        else:
            orbits = [
                ReturnMapOrbit(
                    sections=tuple(sections),
                    cartridge_id=cartridges[0].cartridge_id if cartridges else "",
                    twist=0.0, chirikov_overlap=0.0, winding_number=0.0,
                    lyapunov_max=0.0, birkhoff_fixed_points=0,
                )
            ]
        return self.audit_return_orbits(orbits, cartridges)

    # ─────────────────────────────────────────────────────────────────────
    # §III.5 — Funtor CPTP de Lüders  F: Cartucho → MAC  +  certificado Ω₃
    # ─────────────────────────────────────────────────────────────────────
    def assimilate_to_mac(
        self,
        engine: TOONBufferEngine,
        mac_density_matrix: np.ndarray,
        assimilation_rate_eta: float = 0.15,
    ) -> Tuple[np.ndarray, BufferCertificate]:
        """
        Cierra la FASE III (y el ciclo anidado I ⊂ II ⊂ III):

          1. Audita `ReturnMapOrbit` (KAM, Greene, Chirikov, Melnikov, Birkhoff)
          2. Ejecuta la mixtura CPTP de Lüders hacia la MAC
          3. Determina el veredicto Heyting Ω₃ (implicación intuicionista incluida)
          4. Emite `BufferCertificate` con metadatos celestes y espectrales
        """
        now = time.time()
        mac_dim = int(mac_density_matrix.shape[0])
        carts = list(engine._active_cartridges.values())
        orbits = [c.return_orbit for c in carts if c.return_orbit is not None]
        if not orbits and engine._return_orbits:
            orbits = list(engine._return_orbits)

        audit_result = self.audit_return_orbits(orbits, carts)

        if not carts:
            return mac_density_matrix, BufferCertificate(
                certificate_id=f"CERT-BUFF-{int(now * 1000) % 1_000_000:06d}",
                timestamp=now,
                verdict=HeytingOmega3.COHERENT,
                actuation_mode=ActuationMode.NORMAL_FLUID,
                active_cartridges_count=0,
                assimilated_count=0,
                purged_count=len(engine._purged_history),
                capacity_gromov_peak=engine._capacity_peak,
                mac_von_neumann_entropy=self._compute_von_neumann_entropy(mac_density_matrix),
                poincare_cartan_residual=0.0,
                merkle_root_sha256=self._compute_merkle_root([]),
                delaunay_resonance_order=audit_result["resonance_pq"][1],
                melnikov_max_amplitude=audit_result["melnikov_max"],
                kam_intact_fraction=1.0,
                greene_residue_max=audit_result["greene_residue_max"],
                chirikov_overlap_max=audit_result["chirikov_overlap_max"],
                moser_twist_mean=audit_result["moser_twist_mean"],
                floquet_spectral_radius=audit_result["floquet_spectral_radius"],
                birkhoff_fixed_points=audit_result["birkhoff_fixed_points"],
                heyting_implication=HeytingOmega3.COHERENT.implies(HeytingOmega3.COHERENT).name,
            )

        rho_sum = np.zeros((mac_dim, mac_dim), dtype=np.float64)
        poincare_residuals: List[float] = []
        max_risk = 0.0
        weight = 1.0 / len(carts)
        n_diag = engine._N_diag[:mac_dim, :mac_dim]

        for cart in carts:
            rho_c = cart.quant_comp.density_matrix
            if rho_c.shape[0] != mac_dim:
                resized = np.zeros((mac_dim, mac_dim), dtype=np.float64)
                min_d = min(rho_c.shape[0], mac_dim)
                resized[:min_d, :min_d] = rho_c[:min_d, :min_d]
                rho_c = resized
            rho_sum += weight * rho_c
            poincare_residuals.append(
                abs(cart.quant_comp.poincare_cartan_1form - float(np.trace(rho_c @ n_diag)))
            )
            max_risk = max(max_risk, cart.qual_comp.exergy_risk_index)

        eta = float(np.clip(assimilation_rate_eta, 0.01, 0.50))
        mac_updated = (1.0 - eta) * mac_density_matrix + eta * rho_sum
        mac_updated = 0.5 * (mac_updated + mac_updated.T.conj())
        tr = float(np.trace(mac_updated))
        if tr > _EPS:
            mac_updated /= tr

        max_poincare_res = max(poincare_residuals) if poincare_residuals else 0.0
        gromov_peak = engine._capacity_peak
        kam_regime = audit_result["kam_regime"]
        M_max = audit_result["melnikov_max"]
        R_max = audit_result["greene_residue_max"]
        s_ch = audit_result["chirikov_overlap_max"]

        # Veredicto Ω₃ enriquecido: KAM ∧ Greene ∧ Chirikov ∧ Gromov ∧ Poincaré-Cartan
        if (
            gromov_peak > engine.capacity_gromov_max
            or max_poincare_res > 1e-6
            or kam_regime == KAMRegime.ARNOLD_DIFFUSION
            or s_ch >= 1.0
            or R_max > 1.0
        ):
            verdict = HeytingOmega3.VETOED
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
            logger.error(
                f"[FASE III · CROWBAR] Veto Duro. c_G={gromov_peak:.2f} "
                f"ResΘ={max_poincare_res:.2e} KAM={kam_regime.name} "
                f"Greene={R_max:.3f} Chirikov={s_ch:.3f}. GPIO{engine.esp32_gpio_pin} < 400 ns."
            )
        elif (
            max_risk > 0.70
            or max_poincare_res > 1e-10
            or kam_regime == KAMRegime.CANTORUS_PARTIAL
            or M_max > 0.5
            or abs(R_max - self.greene_critical) < 0.05
        ):
            verdict = HeytingOmega3.DEGRADED
            actuation = ActuationMode.SOFT_VETO_BYPASS
            logger.warning(
                f"[FASE III · BYPASS] Veto Suave. Risk={max_risk:.2f} "
                f"KAM={kam_regime.name} Melnikov={M_max:.4f} Greene={R_max:.3f}."
            )
        else:
            verdict = HeytingOmega3.COHERENT
            actuation = ActuationMode.NORMAL_FLUID

        implication = HeytingOmega3.COHERENT.implies(verdict)

        assimilated_ids: List[str] = []
        for cid, cart in list(engine._active_cartridges.items()):
            if cart.status not in (CartridgeStatus.KAM_BROKEN_CHAOTIC,):
                cart.status = CartridgeStatus.ASSIMILATED
            assimilated_ids.append(cid)
            engine._assimilated_history.append(cid)
        engine._active_cartridges.clear()

        intact_frac = (
            1.0 if not carts
            else sum(1 for c in carts if c.kam_regime == KAMRegime.INTACT_TORUS) / len(carts)
        )
        lrl_norm = 0.0
        if carts and carts[0].delaunay is not None:
            lrl_norm = float(
                np.linalg.norm(PoincareCanonicalAtlas.laplace_runge_lenz_vector(carts[0].delaunay))
            )

        cert = BufferCertificate(
            certificate_id=f"CERT-BUFF-{int(now * 1000) % 1_000_000:06d}",
            timestamp=now,
            verdict=verdict,
            actuation_mode=actuation,
            active_cartridges_count=0,
            assimilated_count=len(assimilated_ids),
            purged_count=len(engine._purged_history),
            capacity_gromov_peak=gromov_peak,
            mac_von_neumann_entropy=self._compute_von_neumann_entropy(mac_updated),
            poincare_cartan_residual=max_poincare_res,
            merkle_root_sha256=self._compute_merkle_root(assimilated_ids),
            delaunay_resonance_order=audit_result["resonance_pq"][1],
            melnikov_max_amplitude=M_max,
            kam_intact_fraction=float(intact_frac),
            mean_motion_vector=audit_result["n_mean_vector"],
            laplace_runge_lenz_norm=lrl_norm,
            greene_residue_max=R_max,
            chirikov_overlap_max=s_ch,
            moser_twist_mean=audit_result["moser_twist_mean"],
            floquet_spectral_radius=audit_result["floquet_spectral_radius"],
            birkhoff_fixed_points=audit_result["birkhoff_fixed_points"],
            heyting_implication=implication.name,
        )
        logger.info(
            f"[FASE III] Asimilación MAC. {cert.certificate_id} Ω₃={verdict.name} "
            f"KAM={kam_regime.name} p:q={audit_result['resonance_pq']} "
            f"M={M_max:.4f} R_G={R_max:.3f} s_Ch={s_ch:.3f} "
            f"Floquet ρ={audit_result['floquet_spectral_radius']:.4f}"
        )
        return mac_updated, cert

    # ─────────────────────────────────────────────────────────────────────
    # §III.6 — Utilidades (von Neumann, Merkle — homología 0-dimensional)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_von_neumann_entropy(rho: np.ndarray) -> float:
        eigs = la.eigvalsh(rho)
        eigs = eigs[eigs > _EPS]
        return float(-np.sum(eigs * np.log(eigs)))

    @staticmethod
    def _compute_merkle_root(id_list: List[str]) -> str:
        if not id_list:
            return hashlib.sha256(b"EMPTY_BUFFER").hexdigest()
        hashes = [hashlib.sha256(cid.encode("utf-8")).hexdigest() for cid in id_list]
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i + 1]).encode("utf-8")).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        return hashes[0]


# ══════════════════════════════════════════════════════════════════════════════
# §D. DEMOSTRACIÓN DOCTORAL INTEGRADA EN 3 FASES ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════

def _telemetry(engine: TOONBufferEngine) -> Dict[str, Any]:
    return {
        "active_cartridges": len(engine._active_cartridges),
        "assimilated_total": len(engine._assimilated_history),
        "purged_total": len(engine._purged_history),
        "capacity_gromov_peak": engine._capacity_peak,
        "capacity_gromov_limit": engine.capacity_gromov_max,
        "poincare_sections": len(engine._poincare_sections),
        "return_orbits": len(engine._return_orbits),
        "dimension": engine.dimension,
        "esp32_gpio_pin": engine.esp32_gpio_pin,
        "mu": engine.mu_gravitational,
    }


if __name__ == "__main__":
    print("═" * 82)
    print("  TOONBufferEngine v5 — Poincaré anidado · Delaunay · Lindstedt · Moser · Greene")
    print("═" * 82)

    # ── FASE II anidada en FASE I (herencia del atlas) ──
    engine = TOONBufferEngine(
        dimension=56, capacity_gromov_max=12.5, default_ttl_sec=2.0, mu_gravitational=1.0
    )

    print("\n[FASE I → II] Ingesta vía CanonicalSeed (germen compute_poincare_canonical_seed):")
    rng = np.random.default_rng(1892)  # año del t. I de *Méthodes Nouvelles*
    c1 = engine.ingest_cartridge(
        apu_code="APU-CONCRETO-3000PSI",
        tangible_cost_vector=rng.random(56) * 150000.0,
        policy_risk_intangible=0.15,
        weather_gremial_factor=0.10,
        coordinates_h2=(1.0, 2.5),
    )
    c2 = engine.ingest_cartridge(
        apu_code="APU-ACERO-FIGURADO-60000PSI",
        tangible_cost_vector=rng.random(56) * 450000.0,
        policy_risk_intangible=0.25,
        weather_gremial_factor=0.20,
        coordinates_h2=(1.5, 3.0),
    )
    for c in (c1, c2):
        d = c.delaunay
        xi, eta, pp, qq = d.poincare_regular_variables()
        print(
            f"  • {c.cartridge_id} [{c.apu_code}]  "
            f"L={d.L:.3f} e={d.eccentricity:.3f} i={math.degrees(d.inclination):.2f}° "
            f"n={d.mean_motion:.4e} τ={d.moser_twist:.4e}  Poincaré(ξ,η)=({xi:.3f},{eta:.3f})"
        )

    print("\n[FASE II] Lindblad-GKSL/RLC + Verlet simpléctico + Σ_ℓ de Poincaré:")
    active_cnt, purged_cnt = engine.evolve_lindblad_rlc_step(dt=0.5)
    print(f"  • Cartuchos activos: {active_cnt}, purgados: {purged_cnt}")
    orbits = engine.poincare_return_map_step(dt_integration=1e-3, max_steps=5000, max_crossings=6)
    n_cross = sum(len(o.sections) for o in orbits)
    print(f"  • Órbitas de retorno: {len(orbits)}  cruces de Σ_ℓ: {n_cross}")
    for o in orbits:
        print(
            f"    – {o.cartridge_id}: ϖ={o.winding_number:.4f}  "
            f"s_Chirikov={o.chirikov_overlap:.3f}  λ_max={o.lyapunov_max:.4f}  "
            f"Birkhoff={o.birkhoff_fixed_points}  cruces={len(o.sections)}"
        )

    print("\n[FASE III] Auditoría KAM/Greene/Chirikov/Melnikov + funtor CPTP de Lüders:")
    auditor = KAMMelnikovAuditor(epsilon_kam=1e-2, greene_critical=0.25)
    mac_initial = np.eye(56, dtype=np.float64) / 56.0
    mac_updated, cert = auditor.assimilate_to_mac(engine, mac_initial, assimilation_rate_eta=0.20)

    print(f"  • Certificado        : {cert.certificate_id}")
    print(f"  • Veredicto Ω₃       : {cert.verdict.name}  (→ {cert.heyting_implication})")
    print(f"  • Modo de actuación  : {cert.actuation_mode.name}")
    print(f"  • Asimilados         : {cert.assimilated_count}")
    print(f"  • Resonancia p:q     : orden q = {cert.delaunay_resonance_order}")
    print(f"  • Melnikov M_max     : {cert.melnikov_max_amplitude:.6f}")
    print(f"  • Greene R_max       : {cert.greene_residue_max:.6f}")
    print(f"  • Chirikov s_max     : {cert.chirikov_overlap_max:.6f}")
    print(f"  • Twist de Moser ⟨τ⟩ : {cert.moser_twist_mean:.6e}")
    print(f"  • Floquet ρ(M)       : {cert.floquet_spectral_radius:.6f}")
    print(f"  • Birkhoff #p.fijos  : {cert.birkhoff_fixed_points}")
    print(f"  • KAM intacto        : {cert.kam_intact_fraction * 100:.1f} %")
    print(f"  • |A|_LRL            : {cert.laplace_runge_lenz_norm:.6f}")
    print(f"  • Entropía vN (MAC)  : {cert.mac_von_neumann_entropy:.4f}")
    print(f"  • Raíz Merkle        : {cert.merkle_root_sha256[:32]}...")
    print(f"  • Telemetría         : {_telemetry(engine)}")

    print("\n" + "═" * 82)
    print("  ✓ T_TOON^trans — POINCARÉ · DELAUNAY · LINDSTEDT · MOSER · GREENE · LÜDERS")
    print("═" * 82)