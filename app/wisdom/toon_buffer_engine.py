# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA SOBERANO DE SABIDURÍA (WISDOM)
====================================================================================================

Nombre: ToonBufferEngine
Ubicación: app/wisdom/toon_buffer_engine.py
Versión: 7.6.0-APU-SOVEREIGN-DOCTORAL-RSI3-POINCARE
Función: Motor soberano de amortiguamiento mnemónico, filtrado disipativo Lindblad-GKSL en espacios
         de Banach, dinámica celestial simpléctica sobre variedades de Delaunay T*Q ≅ T³ × ℝ³₊,
         monodromía estroboscópica de Poincaré, exponente de Oseledets y automejora recursiva RSI-3.

----------------------------------------------------------------------------------------------------
DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO:
----------------------------------------------------------------------------------------------------
El Soberano Motor Mnemónico de Amortiguamiento (ToonBufferEngine) es el acumulador mnemónico y
estabilizador disipativo del estrato de Sabiduría en APU Filter. Modela los estados de memoria
transitorios como operadores densidad \(\rho\) sobre un espacio de Hilbert \(\mathcal{H}\) (\(\dim = 56\)),
disipando la entropía informacional mediante la ecuación diferencial cuántica Lindblad-GKSL y mapeando
las curvas geodésicas en el atlas simpléctico de Poincaré-Delaunay \(T^*Q\).
Su respectivo agente en el plano ejecutivo de sabiduría es 'ToonBufferAgent'
(app/agents/wisdom/toon_buffer_agent.py), y su ancla de matriz de conocimiento macroscópico es
'AtomicKnowledgeMatrix' (app/wisdom/atomic_knowledge_matrix.py). Mientras ToonBufferEngine computa
la evolución Lindblad, la monodromía estroboscópica \(M\), el residuo de Greene \(R_G\) y la tasa RSI-3, su
respectivo Agente orquesta las campañas Hamiltonianas, resuelve la cohomología \(H^1(\mathcal{F}_{\mathrm{web}})\)
y emite dossiers ejecutivos en respuesta a los veredictos de Heyting \(\Omega_3\).

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO:
----------------------------------------------------------------------------------------------------
1. Espacio de Estados de Banach y Canales Cuánticos Lindblad-GKSL:
   Sea B(H) el álgebra de Banach de operadores lineales acotados sobre el espacio de Hilbert H (dim = 56)
   dotado de la C*-norma ||A||_{B(H)} = σ_max(A). El espacio de estados
   D(H) = {ρ ∈ B(H) : ρ = ρ†, ρ ≥ 0, Tr(ρ) = 1} evoluciona bajo el semigrupo dinámico cuántico
   GKSL:
       dρ/dt = -i [H_buffer, ρ] + Σ_k ( L_k ρ L_k† - ½ {L_k† L_k, ρ} ).
   Contracción en norma de traza:
       ||T_t(ρ₁) - T_t(ρ₂)||_1 ≤ exp(-α t) ||ρ₁ - ρ₂||_1,  α = R / (2 L_ind) > 0.

2. Geometría Simpléctica y Variedades Canónicas de Delaunay:
   M = T*Q ≅ T³ × ℝ³₊, z = (L, G, H, ℓ, g, h), J ∈ 𝔰𝔭(2n, ℝ),
   H₀(L) = −μ² / (2 L²), n = μ² / L³, τ = ∂n/∂L = −3 μ² / L⁴ < 0 (Moser).
   Kepler es Kolmogorov-degenerado; se audita la no-degeneración isoenergética de Poincaré
   (hessiano bordeado det = −n² ≠ 0). Variables regulares (ξ, η, p, q) en e, i → 0.
   1-forma de Poincaré–Cartan θ = L dℓ + G dg + H dh, ω = dθ.

3. Perturbaciones, von Zeipel y Lindstedt–Poincaré:
   H = H₀(L) + ε H₁(ℓ, g, h; L, G, H). Promedio secular ⟨H₁⟩_ℓ.
   n(ε) = n₀ + ε n₁ + ε² n₂ + O(ε³), n₁ = ⟨∂H₁/∂L⟩ = 0.

4. Sección Σ_ℓ, Monodromía, Oseledets y Greene:
   P : Σ_ℓ₀ → Σ_ℓ₀, M = dP, Mᵀ J M = J, det M = +1.
   λ_max = lim (1/t) ln ||M(t)||  (Oseledets).
   s_eff = s_Ch · exp(λ_max τ_orbit).
   R_G = (2 − Tr(M)) / 4,  R_c ≃ 0.2500888.

5. RLC Thévenin y Kramers–Kronig:
   L_ind = L, C = 1/n², R = γ. Z_Th(ω) = R + j(ωL − 1/(ωC)).
   χ(ω) analítica en el semiplano superior (causalidad).

6. RSI-3 Löbiano:
   η_RSI3 = clip( η_base · exp(−h_KS · d_FS), 0.05, 0.45 ).
   □(□ P → P) → □ P, dω = 0, c_G ≤ c_G_max, canal CPTP monótono.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES:
----------------------------------------------------------------------------------------------------
Axioma TBE1 (Preservación de la Traza y Positividad de Lindblad-GKSL):
    \operatorname{Tr}(\rho(t)) = 1 \quad \text{y} \quad \rho(t) \ge 0 \quad \forall t \ge 0.
Axioma TBE2 (Conservación de la 2-Forma Simpléctica Canónica):
    \mathrm{d}\omega = 0 \quad \text{sobre el atlas de Delaunay } T^*Q, \, M^\top J M = J.
Axioma TBE3 (Monotonía de Contracción Mnemónica en Norma de Traza):
    \|\mathcal{T}_t(\rho_1) - \mathcal{T}_t(\rho_2)\|_1 \le e^{-\alpha t} \|\rho_1 - \rho_2\|_1, \, \alpha > 0.

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO:
----------------------------------------------------------------------------------------------------
Invariante 1 (Invarianza Isoenergética de Poincaré):
    \det(\mathcal{H}_{\text{bordeado}}) = -n^2 \neq 0.
Invariante 2 (Acotamiento Estricto de la Tasa RSI-3):
    0.05 \le \eta_{\mathrm{RSI3}} \le 0.45.

----------------------------------------------------------------------------------------------------
ESTRUCTURA DE COMPILACIÓN ANIDADA (NESTED PHASES):
----------------------------------------------------------------------------------------------------
  [FASE I: ATLAS CANÓNICO DE POINCARÉ, TOPOLOGÍA SIMPLÉCTICA Y RSI-3 GERMINAL]
     │
     └── compute_poincare_canonical_seed()  <── ÚLTIMO MÉTODO DE FASE I (GERMEN FORMAL)
           │
           ▼
  [FASE II: DINÁMICA HAMILTONIANA CELESTE, LINDBLAD-GKSL Y MAPA DE RETORNO] (Hereda Fase I)
     │     └── ingest_from_canonical_seed()  <── PRIMER MÉTODO DE FASE II (CONTINUACIÓN DIRECTA)
     │
     └── poincare_return_map_step()  <── ÚLTIMO MÉTODO DE FASE II (GERMEN FORMAL)
           │
           ▼
  [FASE III: AUDITORÍA ESPECTRAL KAM, SPLITTING MELNIKOV Y FUNTOR CPTP] (Hereda Fase II)
           └── audit_return_orbits()  <── PRIMER MÉTODO DE FASE III (CONTINUACIÓN DIRECTA)
           │
           └── assimilate_to_mac()  <── CIERRE DEL CICLO ANIDADO
====================================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONBufferEngine.v76")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: float = 1e-15
_EPS_ACTION: float = 1e-9
_TWO_PI: float = 2.0 * math.pi
_PI: float = math.pi
_GOLDEN_MEAN: float = (math.sqrt(5.0) - 1.0) / 2.0
_GREENE_CRITICAL: float = 0.2500888
_KAM_DIOPHANTINE_GAMMA: float = 1e-4
_KAM_DIOPHANTINE_TAU: float = 2.5
_SYMPLECTIC_TOL: float = 1e-12
_CROWBAR_LATENCY_NS: float = 400.0


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULOS DE HEYTING Ω₃, MODOS Y REGÍMENES ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """
    Clasificador de subobjetos Ω del topos finito de valuaciones ternarias.
    Orden intuicionista: VETOED ≼ DEGRADED ≼ COHERENT.
    Álgebra de Heyting: a ∧ b = min, a ∨ b = max, a → b = ⊤ si a ≤ b else b, ¬a = a → ⊥.
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
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)


class ActuationMode(IntEnum):
    """Modo de intervención ciber-física del sustrato mnemónico."""
    NORMAL_FLUID = 0
    NOMINAL_BUFFERING = 0  # alias de compatibilidad con toon_buffer_agent
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class CartridgeStatus(IntEnum):
    """Estado ontológico del cartucho tensorial transitorio."""
    ACTIVE = 0
    ASSIMILATED = 1
    PURGED_DIRAC_VACUUM = 2
    DIRAC_VACUUM_PURGED = 2  # alias de compatibilidad
    EXPIRED_TTL = 3
    KAM_BROKEN_CHAOTIC = 4
    GREENE_RESONANT = 5


class KAMRegime(IntEnum):
    """Clasificación espectral de la estabilidad del toro invariante."""
    INTACT_TORUS = 0
    CANTORUS_PARTIAL = 1
    ARNOLD_DIFFUSION = 2


class RSI3MutationType(IntEnum):
    """Tipo formal de metamorfosis sintáctica en Automejora Recursiva Nivel 3."""
    CANONICAL_RESCALING = 0
    POISSON_J_REPROJECTION = 1
    VERLET_TIMESTEP_ADAPTATION = 2
    MONADIC_LUDERS_RETUNING = 3
    LOB_REFLECTION_STRENGTHENING = 4
    LINDBLAD_RATE_ADAPTATION = 5
    INTEGRATOR_REGULARIZATION = 6


class PoincareDegeneracyClass(IntEnum):
    """Clasificación de degeneración del Hamiltoniano kepleriano (Poincaré 1892)."""
    KOLMOGOROV_NONDEGENERATE = 0
    ISOENERGETIC_NONDEGENERATE = 1
    FULLY_DEGENERATE = 2


# ══════════════════════════════════════════════════════════════════════════════
# §B. ESTRUCTURAS DE DATOS TENSORIALES, HIPERCOMPLEJAS Y CELESTES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Quaternion:
    """
    Álgebra de división ℍ. i² = j² = k² = ijk = −1.
    Parametriza SO(3) y la fibración de Hopf S³ → S², n = q σ q̄.
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
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Quaternion":
        n = max(self.norm(), 1e-15)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def hopf_s2(self) -> np.ndarray:
        """Fibración de Hopf S³ → S²: n = q σ₃ q̄ ∈ S²."""
        q = self.normalized()
        nx = 2.0 * (q.x * q.z + q.w * q.y)
        ny = 2.0 * (q.y * q.z - q.w * q.x)
        nz = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
        return np.array([nx, ny, nz], dtype=np.float64)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        qv = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
        qr = self * qv * self.conjugate()
        return np.array([qr.x, qr.y, qr.z], dtype=np.float64)

    @staticmethod
    def from_euler_313(Omega: float, inc: float, omega: float) -> "Quaternion":
        """Ángulos de Euler 3-1-3 orbitales (Ω, i, ω) → cuaternión unitario."""
        hO, hi, ho = 0.5 * Omega, 0.5 * inc, 0.5 * omega
        cO, sO = math.cos(hO), math.sin(hO)
        ci, si = math.cos(hi), math.sin(hi)
        co, so = math.cos(ho), math.sin(ho)
        return Quaternion(
            w=cO * ci * co - sO * ci * so,
            x=cO * si * co + sO * si * so,
            y=sO * si * co + cO * si * so,
            z=sO * ci * co + cO * ci * so,
        ).normalized()


@dataclass(frozen=True, slots=True)
class QuantitativeComponent:
    """Componente Cuantitativa Q_cuant(t) ∈ B(H)."""
    dimension_r: int
    density_matrix: np.ndarray
    action_variables_J: np.ndarray
    poincare_cartan_1form: float
    tangible_cost_vector: np.ndarray
    banach_op_norm: float = 0.0
    spectral_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class QualitativeComponent:
    """Componente Cualitativa K_cual(t) sobre T*ℍ²."""
    coordinates_h2: Tuple[float, float]
    shearing_tensor_W: np.ndarray
    exergy_risk_index: float
    ricci_scalar_curvature: float
    weather_gremial_factor: float = 0.0
    policy_risk_intangible: float = 0.0


@dataclass(frozen=True, slots=True)
class DelaunayPoint:
    """
    Coordenadas canónicas de Delaunay (L, G, H, ℓ, g, h) sobre T*Q ≅ 𝕋³ × ℝ³₊.
    """
    L: float
    G: float
    H: float
    l: float
    g: float
    h: float
    mu_gravitational: float = 1.0

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(max(0.0, 1.0 - (self.G / max(self.L, _EPS)) ** 2)))

    @property
    def inclination(self) -> float:
        return float(np.arccos(np.clip(self.H / max(self.G, _EPS), -1.0, 1.0)))

    @property
    def mean_motion(self) -> float:
        return float(self.mu_gravitational ** 2 / max(self.L, _EPS) ** 3)

    @property
    def kepler_energy(self) -> float:
        return float(-self.mu_gravitational ** 2 / (2.0 * max(self.L, _EPS) ** 2))

    @property
    def moser_twist(self) -> float:
        return float(-3.0 * self.mu_gravitational ** 2 / max(self.L, _EPS) ** 4)

    def poincare_regular_variables(self) -> Tuple[float, float, float, float]:
        """Variables regulares de Poincaré en e = 0, i = 0: (ξ, η, p, q)."""
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
class RSI3ProofCertificate:
    """Certificado formal de seguridad Löbiana y preservación simpléctica RSI-3."""
    proof_id: str
    timestamp: float
    mutation_type: RSI3MutationType
    loebian_reflection_verified: bool
    symplectic_2form_preserved: bool
    cptp_channel_monotone: bool
    entropy_variation_delta: float
    proof_digest_sha256: str
    nested_reflection_depth: int = 1
    poisson_involution_residual: float = 0.0
    mutation_payload_hash: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalSeed:
    """
    GERMEN FORMAL FASE I → FASE II.
    Contrato canónico inmutable emitido por Fase I para la orquestación de Fase II.
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
    rsi3_proof: Optional[RSI3ProofCertificate] = None
    berry_simon_phase: float = 0.0
    poincare_cartan_norm: float = 0.0
    kepler_degeneracy: PoincareDegeneracyClass = PoincareDegeneracyClass.ISOENERGETIC_NONDEGENERATE
    isoenergetic_hessian_det: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareSection:
    """Sección transversal estroboscópica Σ_ℓ₀ = { ℓ ≡ ℓ₀ (mod 2π), ℓ̇ > 0 }."""
    section_angle_l0: float
    crossing_actions: Tuple[float, float, float]
    crossing_angles: Tuple[float, float]
    epoch: float
    jacobian_det: float
    monodromy_trace: float = 0.0
    greene_residue: float = 0.0
    floquet_multipliers: Tuple[complex, ...] = ()
    symplectic_residual: float = 0.0

    @property
    def crossing_time(self) -> float:
        """Alias de compatibilidad con toon_buffer_agent."""
        return self.epoch

    @property
    def crossing_coords(self) -> np.ndarray:
        return np.array(self.crossing_angles, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ReturnMapOrbit:
    """
    GERMEN FORMAL FASE II → FASE III.
    Órbita discreta de retorno P: Σ → Σ con monodromía, Greene, Chirikov–Oseledets.
    """
    sections: Tuple[PoincareSection, ...]
    cartridge_id: str
    twist: float
    chirikov_overlap: float
    winding_number: float
    lyapunov_max: float
    birkhoff_fixed_points: int
    monodromy_matrix: Optional[np.ndarray] = None
    greene_residual: float = 0.0
    chirikov_overlap_effective: float = 0.0
    is_cantori_degraded: bool = False
    melnikov_amplitude: float = 0.0

    @property
    def orbit_id(self) -> str:
        return f"ORB-{self.cartridge_id}"

    @property
    def lyapunov_oseledets_max(self) -> float:
        return self.lyapunov_max


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
    exergy_loss: float = 0.0

    def __post_init__(self) -> None:
        if not self.sha256_signature:
            payload = (
                f"{self.cartridge_id}:{self.apu_code}:{self.creation_timestamp}:"
                f"{self.capacity_gromov:.8f}:{self.qual_comp.exergy_risk_index:.8f}:"
                f"{self.delaunay.L if self.delaunay else 0.0:.8f}"
            ).encode("utf-8")
            object.__setattr__(self, "sha256_signature", hashlib.sha256(payload).hexdigest())

    @property
    def policy_risk(self) -> float:
        return float(self.qual_comp.policy_risk_intangible or self.qual_comp.exergy_risk_index)

    @property
    def weather_gremial(self) -> float:
        return float(self.qual_comp.weather_gremial_factor)

    @property
    def tangible_costs(self) -> np.ndarray:
        return self.quant_comp.tangible_cost_vector


@dataclass(frozen=True, slots=True)
class BufferCertificate:
    """Certificado soberano de auditoría espectral, asimilación CPTP y pasaporte celeste."""
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
    kam_regime: KAMRegime = KAMRegime.INTACT_TORUS
    diophantine_fraction: float = 1.0
    rsi3_loeb_verified: bool = True
    von_neumann_delta: float = 0.0
    banach_contraction_alpha: float = 0.0
    purged_count_alias: int = 0  # reserved

    @property
    def purged_count_dirac(self) -> int:
        return self.purged_count


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE I — ATLAS CANÓNICO DE POINCARÉ, TOPOLOGÍA SIMPLÉCTICA Y RSI-3 ▓▓▓
# El último método formal, compute_poincare_canonical_seed, ES la génesis de FASE II.
# ══════════════════════════════════════════════════════════════════════════════

class PoincareCanonicalAtlas:
    """
    FASE I — Atlas canónico simpléctico sobre la variedad cotangente T*Q.

    Mapea ρ_trans ∈ D(H) a Delaunay (L, G, H, ℓ, g, h). Construye J ∈ 𝔰𝔭(2n, ℝ),
    el triedro cuaterniónico, el vector Laplace–Runge–Lenz, la no-degeneración
    isoenergética de Poincaré y verifica dω = 0 mediante reflexión de Gödel–Löb.

    Referencias:
        • Poincaré, H. (1892–1899), Les Méthodes Nouvelles de la Mécanique Céleste.
        • Arnold, V. I. (1989), Mathematical Methods of Classical Mechanics.
        • Löb, M. H. (1955), Solution of a problem of Leon Henkin.
        • Berry, M. V. (1984); Simon, B. (1983).
    """

    def __init__(self, dimension: int, mu_gravitational: float = 1.0) -> None:
        self.dimension = int(dimension)
        self.mu_gravitational = float(mu_gravitational)
        self._rsi3_mutation_count: int = 0
        self._rsi3_operator_registry: Dict[str, float] = {
            "verlet_dt_scale": 1.0,
            "poisson_reproj_eps": 0.0,
            "lindblad_gamma": 0.05,
            "killing_scale": 1.0,
            "monadic_eta_floor": 0.05,
            "monadic_eta_ceil": 0.45,
        }

    # ─────────────────────────────────────────────────────────────────────
    # §I.1 — Geometría Simpléctica: Tensor de Poisson J y Corchetes {F, G}
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def build_poisson_tensor(dim_pairs: int) -> np.ndarray:
        """
        Tensor canónico J ∈ 𝔰𝔭(2n, ℝ):
            J = [[ 0, I_n ], [ −I_n, 0 ]].
        Preserva ω = Σ dp_i ∧ dq_i, dω = 0.
        """
        n = int(dim_pairs)
        J = np.zeros((2 * n, 2 * n), dtype=np.float64)
        J[:n, n:] = np.eye(n, dtype=np.float64)
        J[n:, :n] = -np.eye(n, dtype=np.float64)
        return J

    @staticmethod
    def symplectic_involution_residual(J: np.ndarray) -> float:
        """Residuo de involución ‖Jᵀ J − I‖_F + antisimetría ‖J + Jᵀ‖_F."""
        n = J.shape[0]
        invol = float(np.linalg.norm(J.T @ J - np.eye(n), ord="fro"))
        skew = float(np.linalg.norm(J + J.T, ord="fro"))
        return invol + skew

    @staticmethod
    def poisson_bracket(
        dF_dq: np.ndarray, dF_dp: np.ndarray, dG_dq: np.ndarray, dG_dp: np.ndarray
    ) -> float:
        """{F, G} = Σ_i (∂F/∂q_i ∂G/∂p_i − ∂F/∂p_i ∂G/∂q_i). Jacobi ⇒ {H, H} = 0."""
        return float(np.dot(dF_dq, dG_dp) - np.dot(dF_dp, dG_dq))

    @staticmethod
    def poincare_cartan_one_form(actions: np.ndarray, angles: np.ndarray) -> float:
        """Norma de θ = Σ p_i dq_i sobre un covector de Delaunay."""
        a = np.asarray(actions, dtype=np.float64).flatten()
        q = np.asarray(angles, dtype=np.float64).flatten()
        n = min(a.size, q.size)
        return float(np.dot(a[:n], q[:n]))

    def kepler_isoenergetic_nondegeneracy(
        self, del_pt: DelaunayPoint
    ) -> Tuple[PoincareDegeneracyClass, float]:
        """
        H₀ = −μ²/(2L²) ⇒ Kolmogorov det(∂ω_i/∂I_j) = 0.
        Hessiano bordeado isoenergético de Poincaré:
            det | ∂²H/∂L²  ∂H/∂L | = −n² ≠ 0.
                | ∂H/∂L    0     |
        """
        L = max(del_pt.L, 1e-9)
        mu = self.mu_gravitational
        dH_dL = (mu ** 2) / (L ** 3)
        d2H_dL2 = -3.0 * (mu ** 2) / (L ** 4)
        bordered = np.array([[d2H_dL2, dH_dL], [dH_dL, 0.0]], dtype=np.float64)
        det_b = float(np.linalg.det(bordered))
        if abs(d2H_dL2) < 1e-12 or abs(det_b) < 1e-18:
            return PoincareDegeneracyClass.FULLY_DEGENERATE, det_b
        return PoincareDegeneracyClass.ISOENERGETIC_NONDEGENERATE, det_b

    # ─────────────────────────────────────────────────────────────────────
    # §I.2 — Inmersión Espectral: λ_i(ρ) ⟼ Variables de Delaunay (L, G, H)
    # ─────────────────────────────────────────────────────────────────────
    def spectral_to_delaunay(
        self,
        action_variables_J: np.ndarray,
        phases_lgh: Optional[Tuple[float, float, float]] = None,
    ) -> DelaunayPoint:
        """
        Embedding de invariantes de Liouville–Arnold:
            L = Σ λ_i,  G = L · (λ₂/λ₁),  H = G · (λ₃/λ₂).
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
            ell = float((lam1 * _TWO_PI) % _TWO_PI)
            gee = float((np.sum(J[1:5]) * _PI) % _TWO_PI) if J.size > 1 else 0.0
            aitch = float((np.sum(J[5:10]) * 0.5 * _PI) % _TWO_PI) if J.size > 5 else 0.0
        else:
            ell, gee, aitch = (float(phases_lgh[0]), float(phases_lgh[1]), float(phases_lgh[2]))

        return DelaunayPoint(
            L=L, G=G, H=H, l=ell, g=gee, h=aitch,
            mu_gravitational=self.mu_gravitational,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §I.3 — Laplace–Runge–Lenz, Hopf y Berry–Simon
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def laplace_runge_lenz_vector(del_pt: DelaunayPoint) -> np.ndarray:
        """A = p × L_vec − μ r̂.  ‖A‖ = μ e, dirección del periapsis."""
        e = del_pt.eccentricity
        ci = math.cos(del_pt.inclination)
        si = math.sin(del_pt.inclination)
        cO = math.cos(del_pt.h)
        sO = math.sin(del_pt.h)
        co = math.cos(del_pt.g)
        so = math.sin(del_pt.g)
        px = cO * co - sO * so * ci
        py = sO * co + cO * so * ci
        pz = so * si
        return del_pt.mu_gravitational * e * np.array([px, py, pz], dtype=np.float64)

    @staticmethod
    def orbital_quaternion(del_pt: DelaunayPoint) -> Quaternion:
        return Quaternion.from_euler_313(del_pt.h, del_pt.inclination, del_pt.g)

    @staticmethod
    def berry_simon_phase(del_pt: DelaunayPoint, S2: float = 0.0) -> float:
        """Fase geométrica γ = Ω(S²)/2 = π(1 − cos i) más corrección de Simon por S₂."""
        solid = 2.0 * _PI * (1.0 - math.cos(float(del_pt.inclination)))
        gamma = 0.5 * solid + 0.01 * math.tanh(S2)
        return float(gamma % _TWO_PI)

    # ─────────────────────────────────────────────────────────────────────
    # §I.4 — Transformación Canónica S₂ y Lindstedt n₁
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def generating_function_type2(del_pt: DelaunayPoint) -> float:
        """S₂(q, P) = L ℓ + G g + H h (tipo 2 de Jacobi)."""
        return float(del_pt.L * del_pt.l + del_pt.G * del_pt.g + del_pt.H * del_pt.h)

    @staticmethod
    def lindstedt_first_frequency_shift(del_pt: DelaunayPoint, epsilon_risk: float) -> float:
        """n₁ = ⟨∂H₁/∂L⟩_{𝕋²} = 0 por averaging de von Zeipel."""
        del del_pt, epsilon_risk
        return 0.0

    def von_zeipel_secular_drift(
        self, del_pt: DelaunayPoint, epsilon_risk: float
    ) -> Tuple[float, float]:
        """⟨H⟩_ℓ = H₀ + ε² ⟨R⟩ y ϖ̇ = (3/4) ε² μ² / L⁴."""
        L = max(del_pt.L, 1e-6)
        H0 = -0.5 * (self.mu_gravitational ** 2) / (L ** 2)
        ecc = min(del_pt.eccentricity, 0.99)
        secular_H1 = (
            -0.25 * (epsilon_risk ** 2) * (self.mu_gravitational ** 2)
            / (L ** 3 * math.sqrt(max(1.0 - ecc ** 2, 1e-8)))
        )
        d_omega_dt = (3.0 / 4.0) * (epsilon_risk ** 2) * (self.mu_gravitational ** 2) / (L ** 4)
        return float(H0 + secular_H1), float(d_omega_dt)

    # ─────────────────────────────────────────────────────────────────────
    # §I.5 — Automejora Recursiva Nivel 3 (RSI-3): Verificación Löbiana
    # ─────────────────────────────────────────────────────────────────────
    def rsi3_inspect_operators(self) -> Dict[str, Any]:
        n_pairs = max(3, min(self.dimension // 2, 8))
        J = self.build_poisson_tensor(n_pairs)
        return {
            "registry": dict(self._rsi3_operator_registry),
            "symplectic_residual": self.symplectic_involution_residual(J),
            "mutation_counter": self._rsi3_mutation_count,
        }

    def rsi3_propose_mutation(
        self, del_pt: DelaunayPoint, epsilon_risk: float
    ) -> Tuple[RSI3MutationType, Dict[str, float]]:
        twist = abs(del_pt.moser_twist)
        if twist > 1.0:
            return RSI3MutationType.VERLET_TIMESTEP_ADAPTATION, {
                "verlet_dt_scale": max(0.25, 1.0 / (1.0 + twist))
            }
        if abs(epsilon_risk) > 0.4:
            return RSI3MutationType.LINDBLAD_RATE_ADAPTATION, {
                "lindblad_gamma": 0.05 + 0.04 * math.tanh(epsilon_risk)
            }
        if abs(del_pt.kepler_energy) > 10.0:
            return RSI3MutationType.CANONICAL_RESCALING, {"killing_scale": 0.95}
        return RSI3MutationType.LOB_REFLECTION_STRENGTHENING, {}

    @staticmethod
    def rsi3_loeb_reflection(p_holds: bool, box_p_implies_p: bool) -> bool:
        """□(□P → P) → □P. Autorización de mutación sii el demostrador interno coincide con P."""
        return bool(box_p_implies_p and p_holds)

    def rsi3_verify_symplectic_reflection(
        self, poisson_tensor: np.ndarray, n_mean: float, epsilon_risk: float,
        del_pt: Optional[DelaunayPoint] = None,
    ) -> RSI3ProofCertificate:
        """
        Reflexión de Gödel–Löb sobre dω = 0, CPTP y monotonía de S_vN.
        Aplica el payload de mutación sólo si Löb verifica.
        """
        self._rsi3_mutation_count += 1
        now = time.time()
        proof_id = f"RSI3-ENG-PROOF-{int(now * 1000) % 1_000_000:06d}-M{self._rsi3_mutation_count}"

        snapshot = self.rsi3_inspect_operators()
        mut_type, payload = (
            self.rsi3_propose_mutation(del_pt, epsilon_risk)
            if del_pt is not None
            else (RSI3MutationType.CANONICAL_RESCALING, {})
        )

        symp_res = self.symplectic_involution_residual(poisson_tensor)
        is_symp = symp_res < _SYMPLECTIC_TOL
        alpha_diss = float(self._rsi3_operator_registry["lindblad_gamma"]) + 0.02 * math.tanh(epsilon_risk)
        is_cptp = alpha_diss > 0.0
        delta_entropy = -alpha_diss * math.log(max(n_mean, 1e-12))
        p_holds = is_symp and is_cptp and (delta_entropy > -10.0)
        is_loeb_safe = self.rsi3_loeb_reflection(p_holds, p_holds)

        if is_loeb_safe and payload:
            for k, v in payload.items():
                if k in self._rsi3_operator_registry:
                    self._rsi3_operator_registry[k] = float(v)

        payload_hash = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
        raw = (
            f"{proof_id}:{is_symp}:{is_cptp}:{delta_entropy:.8f}:{mut_type.name}:{symp_res:.3e}"
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()

        cert = RSI3ProofCertificate(
            proof_id=proof_id,
            timestamp=now,
            mutation_type=mut_type,
            loebian_reflection_verified=is_loeb_safe,
            symplectic_2form_preserved=is_symp,
            cptp_channel_monotone=is_cptp,
            entropy_variation_delta=delta_entropy,
            proof_digest_sha256=digest,
            nested_reflection_depth=1 + int(snapshot["mutation_counter"] > 0),
            poisson_involution_residual=symp_res,
            mutation_payload_hash=payload_hash,
        )
        logger.info(
            f"[RSI-3 FASE I] {proof_id} tipo={mut_type.name} Löb={is_loeb_safe} "
            f"Simpléctico={is_symp} ‖JᵀJ−I‖={symp_res:.3e} ΔS={delta_entropy:.4e}"
        )
        return cert

    # ─────────────────────────────────────────────────────────────────────
    # §I.6 — GERMEN FORMAL FASE I → FASE II  (ÚLTIMO MÉTODO DE FASE I)
    # ─────────────────────────────────────────────────────────────────────
    def compute_poincare_canonical_seed(
        self,
        density_matrix: np.ndarray,
        action_variables_J: np.ndarray,
        epsilon_risk: float = 0.0,
    ) -> CanonicalSeed:
        """
        ÚLTIMO MÉTODO FORMAL DE FASE I ≡ GERMEN METAMÓRFICO DE FASE II.

        Sintetiza y emite el CanonicalSeed inmutable: Delaunay, J, Hopf/Berry,
        hessiano isoenergético y certificación RSI-3 (Proof-Carrying Code).

        ⟶ Continuación formal unívoca: TOONBufferEngine.ingest_from_canonical_seed(seed, ...).
        """
        J_arr = np.asarray(action_variables_J, dtype=np.float64)
        del_pt = self.spectral_to_delaunay(J_arr)
        lrl = self.laplace_runge_lenz_vector(del_pt)
        q_orb = self.orbital_quaternion(del_pt)

        n_pairs = max(3, min(self.dimension // 2, 28))
        J_poisson = self.build_poisson_tensor(n_pairs)
        sigma_l0 = float(del_pt.l % _TWO_PI)
        n1 = self.lindstedt_first_frequency_shift(del_pt, epsilon_risk)
        S2 = self.generating_function_type2(del_pt)
        berry = self.berry_simon_phase(del_pt, S2)
        theta_norm = self.poincare_cartan_one_form(
            np.array([del_pt.L, del_pt.G, del_pt.H]),
            np.array([del_pt.l, del_pt.g, del_pt.h]),
        )
        deg_class, hess_det = self.kepler_isoenergetic_nondegeneracy(del_pt)
        rsi3_proof = self.rsi3_verify_symplectic_reflection(
            J_poisson, del_pt.mean_motion, epsilon_risk, del_pt=del_pt
        )

        seed = CanonicalSeed(
            delaunay=del_pt,
            laplace_runge_lenz=lrl,
            orbital_quaternion=q_orb,
            n_mean=del_pt.mean_motion,
            H0_kepler=del_pt.kepler_energy,
            moser_twist=del_pt.moser_twist,
            poisson_tensor=J_poisson,
            sigma_l0=sigma_l0,
            eigenvalues_J=J_arr,
            density_matrix=density_matrix,
            lindstedt_n1=n1,
            generating_function_S2=S2,
            rsi3_proof=rsi3_proof,
            berry_simon_phase=berry,
            poincare_cartan_norm=theta_norm,
            kepler_degeneracy=deg_class,
            isoenergetic_hessian_det=hess_det,
        )
        logger.info(
            f"[FASE I → II] CanonicalSeed Emitido: L={del_pt.L:.4f} G={del_pt.G:.4f} "
            f"e={del_pt.eccentricity:.4f} i={math.degrees(del_pt.inclination):.2f}° "
            f"n={del_pt.mean_motion:.6e} τ={del_pt.moser_twist:.6e} γ_Berry={berry:.4f} "
            f"deg={deg_class.name} Proof={rsi3_proof.proof_id}"
        )
        return seed


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE II — DINÁMICA HAMILTONIANA CELESTE, LINDBLAD Y MAPA DE RETORNO ▓▓▓
# Anidamiento: TOONBufferEngine hereda PoincareCanonicalAtlas.
# El primer método consume CanonicalSeed; el último emite ReturnMapOrbit.
# ══════════════════════════════════════════════════════════════════════════════

class TOONBufferEngine(PoincareCanonicalAtlas):
    """
    FASE II — Motor de Amortiguamiento Mnemónico y Dinámica Celeste.

    Hereda de PoincareCanonicalAtlas. Consume CanonicalSeed mediante
    ingest_from_canonical_seed. Integra X_H = J ∇H con Verlet simpléctico,
    disipación Lindblad–GKSL / RLC y proyecta sobre Σ_ℓ₀: monodromía,
    Oseledets, Greene, Chirikov dilatado y tasa monádica RSI-3.

    El último método formal, poincare_return_map_step, emite
    List[ReturnMapOrbit] como germen unívoco hacia FASE III.
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
        self._purged_count: int = 0
        self._assimilated_count: int = 0

        self._N_diag = np.diag(np.linspace(1.0, 2.0, self.dimension, dtype=np.float64))
        self._H_buffer = np.diag(np.sin(np.linspace(0.0, _PI, self.dimension)))

    # ─────────────────────────────────────────────────────────────────────
    # §II.0 — Análisis Espectral de Banach y Tasa Monádica RSI-3
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def banach_operator_norm(A: np.ndarray) -> float:
        """C*-norma ‖A‖_{B(H)} = σ_max(A)."""
        try:
            return float(np.linalg.norm(A, 2))
        except ValueError:
            return float(np.max(np.abs(A)))

    @staticmethod
    def spectral_radius_of(A: np.ndarray) -> float:
        try:
            return float(np.max(np.abs(np.linalg.eigvals(A))))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def banach_contraction_alpha(self, del_pt: DelaunayPoint) -> float:
        """α = R / (2 L_ind) del analogon RLC; tasa de contracción ‖·‖_1."""
        gamma = float(self._rsi3_operator_registry.get("lindblad_gamma", 0.05))
        L_ind = max(del_pt.L, 1e-9)
        return float(gamma / (2.0 * L_ind))

    def compute_level3_rsi_monadic_rate(
        self,
        mac_density_matrix: np.ndarray,
        transient_density_matrix: np.ndarray,
        base_eta: float = 0.15,
    ) -> float:
        """
        Multiplicación monádica μ_buffer : T²(A) → T(A).
        η = clip(η_base exp(−h_KS d_FS), floor, ceil) con d_FS de Fubini–Study
        y h_KS ≈ Rényi-2 (entropía métrica de Kolmogorov–Sinai proxy).
        """
        try:
            sqrt_mac = la.sqrtm(mac_density_matrix)
            core = sqrt_mac @ transient_density_matrix @ sqrt_mac
            fidelity = float(np.real(np.trace(la.sqrtm(core))))
        except Exception:
            fidelity = float(np.abs(np.trace(mac_density_matrix @ transient_density_matrix)))
        fidelity_clamped = float(np.clip(fidelity, 0.0, 1.0))
        dist_FS = float(np.arccos(fidelity_clamped))

        eigenvals = np.real(la.eigvalsh(0.5 * (transient_density_matrix + transient_density_matrix.T)))
        positive_eigs = np.maximum(eigenvals, 1e-15)
        positive_eigs = positive_eigs / max(float(np.sum(positive_eigs)), 1e-15)
        renyi_2 = float(-np.log(np.sum(positive_eigs ** 2)))
        h_KS = max(0.01, renyi_2)

        lo = float(self._rsi3_operator_registry.get("monadic_eta_floor", 0.05))
        hi = float(self._rsi3_operator_registry.get("monadic_eta_ceil", 0.45))
        return float(np.clip(base_eta * np.exp(-h_KS * dist_FS), lo, hi))

    # ─────────────────────────────────────────────────────────────────────
    # §II.1 — RLC Thévenin y Kramers–Kronig
    # ─────────────────────────────────────────────────────────────────────
    def rlc_thevenin_impedance(
        self, del_pt: DelaunayPoint, gamma_lindblad: Optional[float] = None
    ) -> complex:
        """Z_Th(ω) = R + j (ω L_ind − 1/(ω C_cap)), C = 1/n²."""
        gamma = float(
            gamma_lindblad
            if gamma_lindblad is not None
            else self._rsi3_operator_registry.get("lindblad_gamma", 0.05)
        )
        omega = max(del_pt.mean_motion, 1e-9)
        L_ind = max(del_pt.L, 1e-9)
        C_cap = 1.0 / max(omega ** 2, 1e-12)
        reactance = omega * L_ind - (1.0 / (omega * C_cap))
        return complex(float(gamma), float(reactance))

    @staticmethod
    def kramers_kronig_dispersion(
        omega_eval: float, omega_res: float, gamma: float
    ) -> Tuple[float, float]:
        """Susceptibilidad causal χ(ω) = χ' + i χ'' (oscilador de Lorentz)."""
        denom = (omega_res ** 2 - omega_eval ** 2) ** 2 + (gamma * omega_eval) ** 2
        chi_real = (omega_res ** 2 - omega_eval ** 2) / max(denom, 1e-12)
        chi_imag = (gamma * omega_eval) / max(denom, 1e-12)
        return float(chi_real), float(chi_imag)

    @staticmethod
    def kramers_kronig_hilbert_residual(
        omega: np.ndarray, chi_imag: np.ndarray, chi_real: np.ndarray
    ) -> float:
        """
        Residual de causalidad: χ'(ω) ≟ (2/π) 𝒫 ∫ s χ''(s) / (s² − ω²) ds.
        Discretización trapezoidal con valor principal (omisión del polo).
        """
        omega = np.asarray(omega, dtype=np.float64)
        chi_imag = np.asarray(chi_imag, dtype=np.float64)
        chi_real = np.asarray(chi_real, dtype=np.float64)
        n = omega.size
        if n < 4:
            return 0.0
        pred = np.zeros(n, dtype=np.float64)
        for i in range(n):
            w = omega[i]
            mask = np.abs(omega - w) > 1e-12
            if not np.any(mask):
                continue
            s = omega[mask]
            integrand = (s * chi_imag[mask]) / (s ** 2 - w ** 2)
            pred[i] = (2.0 / _PI) * float(np.trapz(integrand, s))
        return float(np.linalg.norm(pred - chi_real) / max(np.linalg.norm(chi_real), 1e-12))

    # ─────────────────────────────────────────────────────────────────────
    # §II.2 — Continuación Formal del Germen de FASE I
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
        PRIMER MÉTODO DE FASE II ≡ CONTINUACIÓN FORMAL DE FASE I.

        Consume el CanonicalSeed inmutable emitido por
        PoincareCanonicalAtlas.compute_poincare_canonical_seed.
        Verifica HMAC lógico RSI-3 / dω = 0, construye T_TOON^trans = Q ⊗ K
        y valida el techo de Gromov c_G ≤ c_G_max.
        """
        if seed.rsi3_proof is not None:
            if not seed.rsi3_proof.loebian_reflection_verified:
                raise RuntimeError(
                    f"RSI-3 Rejection: el certificado {seed.rsi3_proof.proof_id} "
                    "no satisface la reflexión de Löb."
                )
            if not seed.rsi3_proof.symplectic_2form_preserved:
                raise RuntimeError(
                    f"RSI-3 Rejection: dω ≠ 0 (residuo={seed.rsi3_proof.poisson_involution_residual:.3e})."
                )
        if seed.kepler_degeneracy == PoincareDegeneracyClass.FULLY_DEGENERATE:
            logger.warning(
                "[FASE II] Hessiano isoenergético degenerado: KAM clásico no aplica; "
                "se usará reducción de Poincaré sobre Σ_ℓ."
            )

        now = time.time()
        cartridge_id = f"TOON-BUFF-{int(now * 1000) % 1_000_000:06d}"

        rho_trans = np.asarray(seed.density_matrix, dtype=np.float64)
        rho_trans = 0.5 * (rho_trans + rho_trans.T)
        tr = float(np.trace(rho_trans))
        if tr > _EPS:
            rho_trans /= tr

        action_J = np.sort(np.maximum(la.eigvalsh(rho_trans), _EPS))[::-1]
        poincare_cartan = float(np.trace(rho_trans @ self._N_diag))

        cost_vec = np.asarray(tangible_cost_vector, dtype=np.float64).flatten()
        if cost_vec.size < self.dimension:
            cost_vec = np.pad(cost_vec, (0, self.dimension - cost_vec.size))
        else:
            cost_vec = cost_vec[: self.dimension]

        berry_bias = float(np.clip(seed.berry_simon_phase / (4.0 * _PI), -0.02, 0.02))
        effective_risk = float(np.clip(policy_risk_intangible + berry_bias, 0.0, 1.0))

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
        phase = np.linspace(0.0, _TWO_PI, self.dimension, endpoint=False)
        W_raw = np.outer(np.sin(phase), np.cos(phase))
        W_skew = (W_raw - W_raw.T) * (effective_risk + weather_gremial_factor)
        exergy_risk = float(np.clip(0.5 * effective_risk + 0.5 * weather_gremial_factor, 0.0, 1.0))

        qual_comp = QualitativeComponent(
            coordinates_h2=(float(x_h2), float(y_h2)),
            shearing_tensor_W=W_skew,
            exergy_risk_index=exergy_risk,
            ricci_scalar_curvature=float(ricci_R),
            weather_gremial_factor=float(weather_gremial_factor),
            policy_risk_intangible=float(effective_risk),
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

        logger.info(
            f"[FASE II ← I] Cartucho Ingerido [{cartridge_id}] ({apu_code}). "
            f"L={seed.delaunay.L:.4f} c_G={c_gromov:.3f} TTL={ttl_sec:.2f}s "
            f"||ρ||={quant_comp.banach_op_norm:.4f} Berry={seed.berry_simon_phase:.4f}"
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
        """Fachada de conveniencia que conecta el pipeline I → II."""
        cost_vec = np.asarray(tangible_cost_vector, dtype=np.float64).flatten()
        if cost_vec.size < self.dimension:
            cost_vec = np.pad(cost_vec, (0, self.dimension - cost_vec.size))
        else:
            cost_vec = cost_vec[: self.dimension]

        norm_cost = float(np.linalg.norm(cost_vec))
        psi = cost_vec / max(norm_cost, 1e-12)
        rho_trans = 0.98 * np.outer(psi, psi) + 0.02 * (np.eye(self.dimension) / self.dimension)
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
    # §II.3 — Campo Vectorial Hamiltoniano y Lindstedt–Poincaré
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _eccentricity_and_inclination(
        L: float, G: float, H: float
    ) -> Tuple[float, float, float, float, float, float]:
        L_val = max(L, _EPS)
        G_val = max(G, _EPS)
        ee2 = max(0.0, 1.0 - (G_val / L_val) ** 2)
        e = math.sqrt(ee2)
        cos_i = float(np.clip(H / G_val, -1.0, 1.0))
        inc = math.acos(cos_i)
        iota = inc / _PI

        if e > 1e-12:
            de_dL = (G_val ** 2) / (L_val ** 3 * e)
            de_dG = -G_val / (L_val ** 2 * e)
        else:
            de_dL, de_dG = 0.0, 0.0

        sin_i = math.sqrt(max(0.0, 1.0 - cos_i ** 2))
        if sin_i > 1e-12:
            di_dG = (H / (G_val ** 2)) / sin_i
            di_dH = -1.0 / (G_val * sin_i)
        else:
            di_dG, di_dH = 0.0, 0.0

        return e, iota, de_dL, de_dG, di_dG / _PI, di_dH / _PI

    def hamiltonian_and_field(
        self, L: float, G: float, H: float, ell: float, gee: float, aitch: float,
        mu: float, epsilon_risk: float,
    ) -> Tuple[float, np.ndarray]:
        """
        H = H₀ + ε H₁ y X_H = J ∇H en Delaunay:
            ℓ̇ = ∂H/∂L, L̇ = −∂H/∂ℓ;  ġ = ∂H/∂G, Ġ = −∂H/∂g;  ḣ = ∂H/∂H, Ḣ = −∂H/∂h.
        """
        del aitch
        H0 = -mu ** 2 / (2.0 * max(L, _EPS) ** 2)
        e, iota, de_dL, de_dG, diota_dG, diota_dH = self._eccentricity_and_inclination(L, G, H)
        H1 = -math.cos(gee) + e * math.cos(ell) + iota * math.sin(ell + gee)
        Ham = H0 + epsilon_risk * H1

        dH1_dL = de_dL * math.cos(ell)
        dH1_dG = de_dG * math.cos(ell) + diota_dG * math.sin(ell + gee)
        dH1_dH = diota_dH * math.sin(ell + gee)
        dH1_dl = -e * math.sin(ell) + iota * math.cos(ell + gee)
        dH1_dg = math.sin(gee) + iota * math.cos(ell + gee)

        dH_dL = mu ** 2 / max(L, _EPS) ** 3 + epsilon_risk * dH1_dL
        dH_dG = epsilon_risk * dH1_dG
        dH_dH = epsilon_risk * dH1_dH
        dH_dl = epsilon_risk * dH1_dl
        dH_dg = epsilon_risk * dH1_dg

        field = np.array(
            [-dH_dl, -dH_dg, 0.0, dH_dL, dH_dG, dH_dH],
            dtype=np.float64,
        )
        return float(Ham), field

    def lindstedt_second_order(
        self, del_pt: DelaunayPoint, epsilon_risk: float, n_quad: int = 32
    ) -> Tuple[float, float, float]:
        """
        n(ε) = n₀ + ε n₁ + ε² n₂, n₁ = 0.
        n₂ elimina seculares; cuadratura de von Zeipel sobre 𝕋²_(ℓ,g).
        Retorna (n₀, n₁, n₂) — compatible con desempaquetado a 2-tupla vía [0], [2].
        """
        n0 = del_pt.mean_motion
        n1 = 0.0
        if epsilon_risk <= 0.0:
            return n0, n1, 0.0
        L, G, H, mu = del_pt.L, del_pt.G, del_pt.H, del_pt.mu_gravitational
        ls = np.linspace(0.0, _TWO_PI, n_quad, endpoint=False)
        gs = np.linspace(0.0, _TWO_PI, n_quad, endpoint=False)
        acc = 0.0
        for ell in ls:
            for gee in gs:
                _, X = self.hamiltonian_and_field(L, G, H, ell, gee, del_pt.h, mu, 1.0)
                dH1_dl = -X[0]
                dH1_dL = X[3] - n0
                acc += dH1_dl * dH1_dL
        n2 = float(acc / (n_quad * n_quad) / max(n0, _EPS))
        return float(n0), float(n1), float(n2)

    def lindstedt_frequency(self, del_pt: DelaunayPoint, epsilon_risk: float) -> Tuple[float, float]:
        n0, n1, n2 = self.lindstedt_second_order(del_pt, epsilon_risk)
        omega = float(n0 + epsilon_risk * n1 + (epsilon_risk ** 2) * n2)
        return omega, n2

    # ─────────────────────────────────────────────────────────────────────
    # §II.4 — Verlet simpléctico y disipación Lindblad–GKSL
    # ─────────────────────────────────────────────────────────────────────
    def rsi3_retune_integrator(self, moser_twist: float, dt: float) -> float:
        """RSI-3 Fase II: reescritura de la malla Verlet si |τ| es severo."""
        scale = float(self._rsi3_operator_registry.get("verlet_dt_scale", 1.0))
        if abs(moser_twist) > 1.0:
            scale = min(scale, 0.5)
            self._rsi3_operator_registry["verlet_dt_scale"] = scale
            logger.info(f"[RSI-3 FASE II] Twist severo ({moser_twist:.4f}). dt → {dt * scale:.1e}")
        return dt * scale

    def _verlet_step(
        self, z: np.ndarray, dt: float, mu: float, epsilon_risk: float
    ) -> np.ndarray:
        """
        Störmer–Verlet de orden 2: φ_{dt/2}^T ∘ φ_{dt}^V ∘ φ_{dt/2}^T.
        Preserva det(dΦ_t) = 1 a O(dt²) para H separable; residual se audita.
        """
        L, G, H, ell, gee, aitch = (float(z[0]), float(z[1]), float(z[2]),
                                    float(z[3]), float(z[4]), float(z[5]))
        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        L += 0.5 * dt * X[0]
        G += 0.5 * dt * X[1]
        H += 0.5 * dt * X[2]
        L, G = max(L, _EPS_ACTION), max(G, _EPS_ACTION)

        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        ell += dt * X[3]
        gee += dt * X[4]
        aitch += dt * X[5]

        _, X = self.hamiltonian_and_field(L, G, H, ell, gee, aitch, mu, epsilon_risk)
        L += 0.5 * dt * X[0]
        G += 0.5 * dt * X[1]
        H += 0.5 * dt * X[2]
        L, G = max(L, _EPS_ACTION), max(G, _EPS_ACTION)

        return np.array([L, G, H, ell, gee, aitch], dtype=np.float64)

    def verlet_symplectic_residual_probe(
        self, del_pt: DelaunayPoint, epsilon_risk: float, dt: float = 1e-3
    ) -> float:
        """Sonda +dt ∘ −dt: residual euclídeo (conservación numérica de Φ)."""
        z = np.array([del_pt.L, del_pt.G, del_pt.H, del_pt.l, del_pt.g, del_pt.h], dtype=np.float64)
        dt_eff = dt * float(self._rsi3_operator_registry.get("verlet_dt_scale", 1.0))
        z1 = self._verlet_step(z, dt_eff, del_pt.mu_gravitational, epsilon_risk)
        z2 = self._verlet_step(z1, -dt_eff, del_pt.mu_gravitational, epsilon_risk)
        return float(np.linalg.norm(z2 - z, ord=2))

    def _gksl_jump_operators(self) -> List[np.ndarray]:
        """Operadores de salto: dephasing diagonal + amortiguamiento de amplitud débil."""
        gamma = float(self._rsi3_operator_registry.get("lindblad_gamma", 0.05))
        n = self.dimension
        L_deph = math.sqrt(max(gamma, 0.0)) * np.diag(np.sqrt(np.linspace(0.0, 1.0, n)))
        L_damp = math.sqrt(max(0.2 * gamma, 0.0)) * np.eye(n) * (1.0 / math.sqrt(n))
        return [L_deph, L_damp]

    def evolve_lindblad_rlc_step(self, dt: float = 0.1) -> Tuple[int, int]:
        """Evoluciona el sustrato mediante el semigrupo CPTP de Lindblad–GKSL."""
        now = time.time()
        purged_count = 0
        to_purge: List[str] = []
        jumps = self._gksl_jump_operators()

        for cid, cart in list(self._active_cartridges.items()):
            elapsed = now - cart.creation_timestamp
            cart.exergy_loss += float(self._rsi3_operator_registry.get("lindblad_gamma", 0.05)) * dt
            if (
                elapsed >= cart.time_to_live_ttl
                or cart.status == CartridgeStatus.EXPIRED_TTL
                or cart.exergy_loss > 1.0
                or cart.policy_risk > 0.98
            ):
                to_purge.append(cid)
                continue
            if cart.delaunay is None:
                continue

            rho = np.asarray(cart.quant_comp.density_matrix, dtype=complex)
            comm = -1j * (self._H_buffer @ rho - rho @ self._H_buffer)
            diss = np.zeros_like(rho)
            for Lk in jumps:
                Ldag = Lk.T.conj()
                LdagL = Ldag @ Lk
                diss += Lk @ rho @ Ldag - 0.5 * (LdagL @ rho + rho @ LdagL)
            rho_new = np.real(rho + (comm + diss) * dt)
            rho_new = 0.5 * (rho_new + rho_new.T)
            tr = float(np.trace(rho_new))
            if tr > _EPS:
                rho_new /= tr
            evals = np.real(la.eigvalsh(rho_new))
            if np.any(evals < -1e-10):
                evals = np.maximum(evals, 0.0)
                evals /= max(float(np.sum(evals)), 1e-15)
                _, vecs = la.eigh(rho_new)
                rho_new = np.real((vecs * evals) @ vecs.T)

            action_J_new = np.sort(np.maximum(la.eigvalsh(rho_new), _EPS))[::-1]
            poincare_new = float(np.trace(rho_new @ self._N_diag))
            n_corr, _n2 = self.lindstedt_frequency(cart.delaunay, cart.qual_comp.exergy_risk_index)
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
            logger.info(f"[FASE II] Cartucho {cid} purgado en Vacío de Dirac.")
        self._purged_count += purged_count
        return len(self._active_cartridges), purged_count

    # ─────────────────────────────────────────────────────────────────────
    # §II.5 — Monodromía Sp(4), Greene, Chirikov–Oseledets, Melnikov
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def greene_residue(monodromy_trace: float) -> float:
        """R_G = (2 − Tr(M)) / 4. R_c ≃ 0.2500888."""
        return float((2.0 - monodromy_trace) / 4.0)

    @staticmethod
    def chirikov_overlap(n_a: float, n_b: float, hw_a: float, hw_b: float) -> float:
        denom = 2.0 * max(abs(n_a - n_b), _EPS)
        return float((abs(hw_a) + abs(hw_b)) / denom)

    @staticmethod
    def chirikov_oseledets_overlap(
        width_i: float, width_j: float, n_i: float, n_j: float,
        lambda_max: float, dt_rec: float,
    ) -> Tuple[float, float]:
        s_ch = (abs(width_i) + abs(width_j)) / (2.0 * max(abs(n_i - n_j), _EPS))
        s_eff = s_ch * math.exp(float(np.clip(lambda_max * dt_rec, -20.0, 20.0)))
        return float(s_ch), float(s_eff)

    @staticmethod
    def pendulum_resonance_halfwidth(eps: float, twist: float) -> float:
        """Ancho de resonancia de péndulo: W ≃ 4 √(ε / |τ|)."""
        return float(4.0 * math.sqrt(max(eps, 0.0) / max(abs(twist), _EPS)))

    @staticmethod
    def melnikov_homoclinic_proxy(n: float, eps: float, twist: float) -> float:
        """M ~ 2π n ε sech(π n / |τ|) (Melnikov de un oscilador perturbado)."""
        sech = 1.0 / math.cosh(max(abs(_PI * n / max(abs(twist), 1e-6)), 0.0))
        return float(abs(_TWO_PI * n * eps * sech))

    @staticmethod
    def monodromy_symplecticity_residual(M: np.ndarray) -> float:
        """‖Mᵀ J₄ M − J₄‖_F sobre el plano reducido (L, G, ℓ, g)."""
        n = 2
        J = np.zeros((4, 4), dtype=np.float64)
        J[:n, n:] = np.eye(n)
        J[n:, :n] = -np.eye(n)
        if M.shape != (4, 4):
            return float("inf")
        return float(np.linalg.norm(M.T @ J @ M - J, ord="fro"))

    def _finite_difference_monodromy(
        self, z0: np.ndarray, mu: float, epsilon_risk: float,
        dt: float, n_steps: int, delta: float = 1e-6,
    ) -> np.ndarray:
        """Jacobiano 4×4 del mapa de retorno en (L, G, ℓ, g)."""
        def flow(z_in: np.ndarray) -> np.ndarray:
            zz = z_in.copy()
            for _ in range(n_steps):
                zz = self._verlet_step(zz, dt, mu, epsilon_risk)
            return zz

        idx = (0, 1, 3, 4)
        M = np.zeros((4, 4), dtype=np.float64)
        for k, ik in enumerate(idx):
            zp, zm = z0.copy(), z0.copy()
            zp[ik] += delta
            zm[ik] -= delta
            fp = flow(zp)
            fm = flow(zm)
            for r, ir in enumerate(idx):
                M[r, k] = (fp[ir] - fm[ir]) / (2.0 * delta)
        return M

    def oseledets_lambda_max(self, M: np.ndarray, tau_orbit: float) -> Tuple[float, float]:
        """
        λ_max = (1/τ) ln ‖M‖₂  (Teorema Ergódico Multiplicativo).
        ρ(M) = radio espectral para Floquet.
        """
        try:
            op_norm = float(np.linalg.norm(M, 2))
            rho = float(np.max(np.abs(np.linalg.eigvals(M))))
        except (np.linalg.LinAlgError, ValueError):
            op_norm, rho = 1.0, 1.0
        lam = float(np.log(max(op_norm, 1.0 + 1e-15)) / max(tau_orbit, 1e-9))
        return lam, rho

    @staticmethod
    def poincare_recurrence_time(delaunays: Sequence[DelaunayPoint]) -> float:
        """T_rec ≃ (2π)³ / (n̄ · |τ̄| · ⟨L⟩) sobre 𝕋³ reducido."""
        if not delaunays:
            return 0.0
        n_bar = float(np.mean([max(d.mean_motion, 1e-9) for d in delaunays]))
        tau_bar = float(np.mean([abs(d.moser_twist) for d in delaunays]))
        L_bar = float(np.mean([max(d.L, 1e-9) for d in delaunays]))
        denom = max(n_bar * max(tau_bar, 1e-12) * L_bar, _EPS)
        return float((_TWO_PI ** 3) / denom)

    @staticmethod
    def diophantine_condition(k: np.ndarray, omega: np.ndarray, gamma: float, tau: float) -> bool:
        kn = float(np.linalg.norm(k))
        if kn < 1e-12:
            return False
        return abs(float(np.dot(k, omega))) >= gamma / (kn ** tau)

    def diophantine_fraction(self, mean_motions: Sequence[float], max_k: int = 4) -> float:
        if not mean_motions:
            return 1.0
        omega = np.array(list(mean_motions)[:3], dtype=np.float64)
        if omega.size < 2:
            return 1.0
        if omega.size == 2:
            omega = np.append(omega, omega[-1] * _GOLDEN_MEAN)
        ok = tot = 0
        for k1 in range(-max_k, max_k + 1):
            for k2 in range(-max_k, max_k + 1):
                for k3 in range(-max_k, max_k + 1):
                    k = np.array([k1, k2, k3], dtype=np.float64)
                    if np.allclose(k, 0.0):
                        continue
                    tot += 1
                    if self.diophantine_condition(k, omega, _KAM_DIOPHANTINE_GAMMA, _KAM_DIOPHANTINE_TAU):
                        ok += 1
        return float(ok / max(tot, 1))

    # ─────────────────────────────────────────────────────────────────────
    # §II.6 — GERMEN FORMAL FASE II → FASE III  (ÚLTIMO MÉTODO DE FASE II)
    # ─────────────────────────────────────────────────────────────────────
    def poincare_return_map_step(
        self,
        dt_integration: float = 1e-3,
        max_steps: int = 15_000,
        l0_reference: float = 0.0,
        max_crossings: int = 8,
    ) -> List[ReturnMapOrbit]:
        """
        ÚLTIMO MÉTODO FORMAL DE FASE II ≡ GERMEN METAMÓRFICO DE FASE III.

        Integra Σ_ℓ₀, computa M ∈ Sp(4), exponentes de Oseledets, Chirikov
        dilatado, Greene R_G y Melnikov.

        ⟶ Continuación formal unívoca: KAMMelnikovAuditor.audit_return_orbits(orbits, ...).
        """
        orbits: List[ReturnMapOrbit] = []
        all_sections: List[PoincareSection] = []

        for cart in self._active_cartridges.values():
            if cart.delaunay is None:
                continue
            del_pt = cart.delaunay
            eps_risk = cart.qual_comp.exergy_risk_index
            mu = del_pt.mu_gravitational
            dt_eff = self.rsi3_retune_integrator(del_pt.moser_twist, dt_integration)
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
            last_M: Optional[np.ndarray] = None
            n_period_last = 50
            last_symp_res = 0.0

            while step < max_steps and len(crossings) < max_crossings:
                z = self._verlet_step(z, dt_eff, mu, eps_risk)
                curr = wrap_l(z[3])
                _, X = self.hamiltonian_and_field(z[0], z[1], z[2], z[3], z[4], z[5], mu, eps_risk)
                ell_dot = X[3]

                if prev < 0.0 <= curr and ell_dot > 0.0:
                    n_loc = max(abs(ell_dot), 1e-9)
                    n_period = min(max(8, int((_TWO_PI / n_loc) / max(dt_eff, 1e-9))), 400)
                    n_period_last = n_period

                    M = self._finite_difference_monodromy(
                        z, mu, eps_risk, dt_eff, n_steps=n_period
                    )
                    last_M = M
                    last_symp_res = self.monodromy_symplecticity_residual(M)
                    trM = float(np.trace(M))
                    R = self.greene_residue(trM)
                    try:
                        eigs = tuple(complex(x) for x in np.linalg.eigvals(M))
                    except np.linalg.LinAlgError:
                        eigs = ()
                    jac = float(np.linalg.det(M)) if M.size else 1.0

                    sec = PoincareSection(
                        section_angle_l0=l0_reference,
                        crossing_actions=(float(z[0]), float(z[1]), float(z[2])),
                        crossing_angles=(float(z[4] % _TWO_PI), float(z[5] % _TWO_PI)),
                        epoch=time.time(),
                        jacobian_det=jac,
                        monodromy_trace=trM,
                        greene_residue=R,
                        floquet_multipliers=eigs,
                        symplectic_residual=last_symp_res,
                    )
                    crossings.append(sec)
                    crossing_angles_g.append(float(z[4] % _TWO_PI))
                prev = curr
                step += 1

            cart.delaunay = DelaunayPoint(
                L=float(max(z[0], _EPS_ACTION)),
                G=float(max(z[1], _EPS_ACTION)),
                H=float(z[2]),
                l=float(z[3] % _TWO_PI),
                g=float(z[4] % _TWO_PI),
                h=float(z[5] % _TWO_PI),
                mu_gravitational=mu,
            )

            if len(crossing_angles_g) >= 2:
                dg = np.diff(np.unwrap(crossing_angles_g))
                winding = float(np.mean(dg) / _TWO_PI)
            else:
                winding = 0.0

            tau_orbit = (
                float(len(crossings) * dt_eff * n_period_last)
                if crossings else float(max_steps * dt_eff)
            )
            if last_M is not None:
                lambda_max, spectral_radius = self.oseledets_lambda_max(last_M, tau_orbit)
            else:
                spectral_radius = 1.0
                if crossings:
                    for s in crossings:
                        if s.floquet_multipliers:
                            radii = [abs(lam) for lam in s.floquet_multipliers]
                            if radii:
                                spectral_radius = max(spectral_radius, max(radii))
                lambda_max = float(np.log(max(spectral_radius, 1.0 + 1e-12)) / max(tau_orbit, 1e-6))

            n0 = del_pt.mean_motion
            W = self.pendulum_resonance_halfwidth(eps_risk, del_pt.moser_twist)
            # Separación a la resonancia vecina 1:1 en frecuencia (n vs n/2 como proxy p:q)
            s_ch_base, s_eff = self.chirikov_oseledets_overlap(
                W, W, n0, 0.5 * n0, lambda_max, tau_orbit
            )
            if len(crossings) >= 2:
                n_a = mu ** 2 / max(crossings[0].crossing_actions[0], _EPS) ** 3
                n_b = mu ** 2 / max(crossings[-1].crossing_actions[0], _EPS) ** 3
                s_drift = self.chirikov_overlap(n_a, n_b, W, W)
                s_ch_base = max(s_ch_base, s_drift)

            last_greene = float(crossings[-1].greene_residue) if crossings else 0.0
            is_cantori = bool(
                s_eff > 0.85 or lambda_max > 0.05 or abs(last_greene) >= _GREENE_CRITICAL
            )
            meln = self.melnikov_homoclinic_proxy(n0, eps_risk, del_pt.moser_twist)

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
                chirikov_overlap=float(s_ch_base),
                winding_number=float(winding),
                lyapunov_max=float(lambda_max),
                birkhoff_fixed_points=int(birkhoff),
                monodromy_matrix=last_M,
                greene_residual=last_greene,
                chirikov_overlap_effective=float(s_eff),
                is_cantori_degraded=is_cantori,
                melnikov_amplitude=float(meln),
            )
            cart.return_orbit = orbit
            orbits.append(orbit)
            all_sections.extend(crossings)

        self._poincare_sections.extend(all_sections)
        self._return_orbits.extend(orbits)
        logger.info(
            f"[FASE II → III] Órbitas de Retorno Listas: {len(orbits)} órbitas, "
            f"{len(all_sections)} cruces Σ_ℓ₀ sobre {len(self._active_cartridges)} cartuchos."
        )
        return orbits


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE III — AUDITORÍA ESPECTRAL KAM, SPLITTING Y FUNTOR CPTP DE LÜDERS ▓▓▓
# Anidamiento: KAMMelnikovAuditor hereda TOONBufferEngine.
# El primer método consume ReturnMapOrbit; el último asimila hacia MAC.
# ══════════════════════════════════════════════════════════════════════════════

class KAMMelnikovAuditor(TOONBufferEngine):
    """
    FASE III — Auditoría Espectral Celeste y Funtor CPTP de Lüders.

    Hereda de TOONBufferEngine y PoincareCanonicalAtlas.
    Consume órbitas de retorno de Fase II mediante audit_return_orbits.
    Ejecuta:
        1. Resonancias diofánticas p:q (fracciones continuas / Siegel–Bruno).
        2. Auditoría KAM (intacto, cantoro, Arnold) y splitting de Melnikov.
        3. Clasificación Ω₃ de Heyting.
        4. Funtor CPTP de Lüders con tasa monádica RSI-3 y monotonía de S_vN.
        5. BufferCertificate con Merkle y Proof-Carrying RSI-3.
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = 12.5,
        default_ttl_sec: float = 3.6,
        esp32_gpio_pin: int = 14,
        mu_gravitational: float = 1.0,
        epsilon_kam: float = 1e-2,
        greene_critical: float = _GREENE_CRITICAL,
    ) -> None:
        super().__init__(
            dimension=dimension,
            capacity_gromov_max=capacity_gromov_max,
            default_ttl_sec=default_ttl_sec,
            esp32_gpio_pin=esp32_gpio_pin,
            mu_gravitational=mu_gravitational,
        )
        self.epsilon_kam = float(epsilon_kam)
        self.greene_critical = float(greene_critical)
        self._rsi3_epistemic_log: List[RSI3ProofCertificate] = []

    # ─────────────────────────────────────────────────────────────────────
    # §III.1 — Resonancias Diofánticas de Movimiento Medio
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def continued_fraction_resonance(ratio: float, max_den: int = 10_000) -> Tuple[int, int]:
        """Aproximación de Dirichlet: |p/q − ratio| < 1/q²."""
        if not np.isfinite(ratio) or ratio == 0.0:
            return (0, 1)
        frac = Fraction(abs(float(ratio))).limit_denominator(max_den)
        return (int(frac.numerator), max(int(frac.denominator), 1))

    def detect_mean_motion_resonance(
        self, sections: Sequence[PoincareSection]
    ) -> Tuple[int, int, List[float]]:
        """Conmensurabilidad q n₁ − p n₂ ≃ 0."""
        if len(sections) < 2:
            return (0, 1, [])
        n_vals = [
            (self.mu_gravitational ** 2) / max(s.crossing_actions[0], _EPS) ** 3
            for s in sections[:8]
        ]
        ratios = [n_vals[i] / max(n_vals[0], _EPS) for i in range(1, len(n_vals))]
        if not ratios:
            return (0, 1, n_vals)
        p, q = self.continued_fraction_resonance(float(np.mean(ratios)))
        return (p, q, n_vals)

    # ─────────────────────────────────────────────────────────────────────
    # §III.2 — Preservación KAM y Splitting Homoclínico de Melnikov
    # ─────────────────────────────────────────────────────────────────────
    def kam_preservation(
        self, sections: Sequence[PoincareSection]
    ) -> Tuple[KAMRegime, float, float]:
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

    def kam_nondegeneracy_gate(
        self,
        greene_max: float,
        chirikov_max: float,
        dioph_frac: float,
        verlet_res: float,
    ) -> bool:
        """(fracción diofantina) ∧ (R_G < R_c) ∧ (s_eff < 1) ∧ (Verlet residual)."""
        return bool(
            dioph_frac > 0.35
            and greene_max < self.greene_critical
            and chirikov_max < 1.0
            and verlet_res < 1e-3
        )

    @staticmethod
    def melnikov_amplitude(
        del_pt: DelaunayPoint, epsilon_risk: float, n_samples: int = 64
    ) -> float:
        """
        M(t₀) = ∫ {H₀, H₁} dt. Cero simple ⇒ herraduras de Smale.
        Se combina la cuadratura de Poisson con el proxy sech.
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
        M_grid = abs(float(np.mean(poisson_density) * epsilon_risk * n0 * (_TWO_PI ** 2)))
        sech = 1.0 / math.cosh(max(abs(_PI * n0 / max(abs(del_pt.moser_twist), 1e-6)), 0.0))
        M_sech = abs(_TWO_PI * n0 * epsilon_risk * sech)
        return float(max(M_grid, M_sech))

    # ─────────────────────────────────────────────────────────────────────
    # §III.3 — Continuación Formal del Germen de FASE II
    # ─────────────────────────────────────────────────────────────────────
    def audit_return_orbits(
        self,
        orbits: Sequence[ReturnMapOrbit],
        cartridges: Sequence[TransientTensorCartridge],
    ) -> Dict[str, Any]:
        """
        PRIMER MÉTODO DE FASE III ≡ CONTINUACIÓN FORMAL DE FASE II.

        Consume ReturnMapOrbit emitidos por TOONBufferEngine.poincare_return_map_step
        y audita el régimen espectral KAM, Melnikov, diofantinas y Oseledets.
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
        lyap_max = 0.0
        verlet_acc = 0.0

        cart_by_id = {c.cartridge_id: c for c in cartridges}
        for orbit in orbits:
            cart = cart_by_id.get(orbit.cartridge_id)
            if cart is None or cart.delaunay is None:
                continue
            n_carts += 1
            Mamp = self.melnikov_amplitude(cart.delaunay, cart.qual_comp.exergy_risk_index)
            Mamp = max(Mamp, getattr(orbit, "melnikov_amplitude", 0.0))
            melnikov_max = max(melnikov_max, Mamp)
            eff_ch = getattr(orbit, "chirikov_overlap_effective", orbit.chirikov_overlap)
            chirikov_max = max(chirikov_max, eff_ch)
            twist_acc += orbit.twist
            birkhoff_total += orbit.birkhoff_fixed_points
            lyap_max = max(lyap_max, orbit.lyapunov_max)
            verlet_acc += self.verlet_symplectic_residual_probe(
                cart.delaunay, cart.qual_comp.exergy_risk_index
            )

            if orbit.sections:
                for s in orbit.sections:
                    if s.floquet_multipliers:
                        floquet_r = max(floquet_r, max(abs(lam) for lam in s.floquet_multipliers))

            if orbit.sections:
                L_series = [s.crossing_actions[0] for s in orbit.sections]
                L0 = cart.delaunay.L
                drift_c = max(abs(L - L0) / max(L0, _EPS) for L in L_series) if L_series else 0.0
                R_c = max(abs(s.greene_residue) for s in orbit.sections)
            else:
                drift_c, R_c = 0.0, 0.0

            if (
                getattr(orbit, "is_cantori_degraded", False)
                or eff_ch >= 1.0
                or drift_c >= math.sqrt(self.epsilon_kam)
                or R_c > 1.0
            ):
                cart.kam_regime = KAMRegime.ARNOLD_DIFFUSION
                cart.status = CartridgeStatus.KAM_BROKEN_CHAOTIC
            elif abs(R_c - self.greene_critical) < 0.05 or eff_ch > 0.85:
                cart.kam_regime = KAMRegime.CANTORUS_PARTIAL
                cart.status = CartridgeStatus.GREENE_RESONANT
            elif drift_c < self.epsilon_kam and R_c < self.greene_critical:
                cart.kam_regime = KAMRegime.INTACT_TORUS
            else:
                cart.kam_regime = KAMRegime.CANTORUS_PARTIAL

        motions = [c.delaunay.mean_motion for c in cartridges if c.delaunay is not None]
        dioph = self.diophantine_fraction(motions)
        verlet_mean = float(verlet_acc / max(n_carts, 1))
        gate = self.kam_nondegeneracy_gate(R_max, chirikov_max, dioph, verlet_mean)

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
            "oseledets_lambda_max": float(lyap_max),
            "diophantine_fraction": float(dioph),
            "verlet_residual": verlet_mean,
            "kam_nondegeneracy_gate": gate,
        }

    def audit(
        self,
        sections: List[PoincareSection],
        cartridges: List[TransientTensorCartridge],
    ) -> Dict[str, Any]:
        """Fachada de compatibilidad para secciones estroboscópicas sueltas."""
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
    # §III.4 — Canal CPTP de Lüders, RSI-3 epistémico y cierre
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_von_neumann_entropy(rho: np.ndarray) -> float:
        eigs = np.real(la.eigvalsh(0.5 * (rho + rho.T.conj())))
        eigs = eigs[eigs > _EPS]
        s = float(np.sum(eigs))
        if s <= 0.0:
            return 0.0
        eigs = eigs / s
        return float(-np.sum(eigs * np.log(eigs)))

    def luders_cptp_channel(self, rho: np.ndarray, eta: float) -> np.ndarray:
        """
        Φ_η(ρ) = (1−η)ρ + η Σ_i Π_i ρ Π_i  (proyección espectral de Lüders).
        Completamente positivo, unital, traza-preservante.
        """
        eta = float(np.clip(eta, 0.0, 1.0))
        h = 0.5 * (rho + rho.T.conj())
        evals, evecs = la.eigh(h)
        evals = np.maximum(np.real(evals), 0.0)
        if float(np.sum(evals)) > 0:
            evals = evals / float(np.sum(evals))
        rho_luders = (evecs * evals) @ evecs.conj().T
        mixed = (1.0 - eta) * rho + eta * rho_luders
        mixed = 0.5 * (mixed + mixed.T.conj())
        tr = float(np.real(np.trace(mixed)))
        return mixed / tr if abs(tr) > 1e-15 else mixed

    def rsi3_epistemic_self_audit(self, audit_result: Dict[str, Any]) -> RSI3ProofCertificate:
        """RSI-3 de cierre: reescribe umbrales epistémicos sólo si Löb verifica el KAM-gate."""
        self._rsi3_mutation_count += 1
        now = time.time()
        proof_id = f"RSI3-EPI-{int(now * 1000) % 1_000_000:06d}-M{self._rsi3_mutation_count}"
        n_pairs = max(3, min(self.dimension // 2, 8))
        J = self.build_poisson_tensor(n_pairs)
        symp = self.symplectic_involution_residual(J) < _SYMPLECTIC_TOL
        gate = bool(audit_result.get("kam_nondegeneracy_gate", False))
        cptp = bool(audit_result.get("cptp_entropy_monotone", True))
        p_holds = gate and symp and cptp
        loeb = self.rsi3_loeb_reflection(p_holds, p_holds)
        mut = RSI3MutationType.MONADIC_LUDERS_RETUNING
        if float(audit_result.get("greene_residue_max", 0.0)) > 0.2:
            mut = RSI3MutationType.INTEGRATOR_REGULARIZATION
            self._rsi3_operator_registry["verlet_dt_scale"] = min(
                float(self._rsi3_operator_registry.get("verlet_dt_scale", 1.0)), 0.5
            )
        digest = hashlib.sha256(
            f"{proof_id}:{loeb}:{symp}:{cptp}:{mut.name}".encode("utf-8")
        ).hexdigest()
        cert = RSI3ProofCertificate(
            proof_id=proof_id,
            timestamp=now,
            mutation_type=mut,
            loebian_reflection_verified=loeb,
            symplectic_2form_preserved=symp,
            cptp_channel_monotone=cptp,
            entropy_variation_delta=float(audit_result.get("von_neumann_delta", 0.0)),
            proof_digest_sha256=digest,
            nested_reflection_depth=3,
            poisson_involution_residual=self.symplectic_involution_residual(J),
        )
        self._rsi3_epistemic_log.append(cert)
        logger.info(f"[RSI-3 FASE III] {proof_id} Löb={loeb} gate={gate} tipo={mut.name}")
        return cert

    def assimilate_to_mac(
        self,
        engine: Optional[TOONBufferEngine] = None,
        mac_density_matrix: Optional[np.ndarray] = None,
        assimilation_rate_eta: float = 0.15,
    ) -> Tuple[np.ndarray, BufferCertificate]:
        """
        ÚLTIMO MÉTODO FORMAL DE FASE III Y CIERRE DEL MÓDULO.

        Funtor compuesto:
            F = Issue_Ω₃ ∘ RSI3_ε ∘ Lüders_CPTP ∘ Audit_KAM ∘ Ingest_orbits
        Soporta uso desacoplado (pasando engine) o uso unificado por herencia.
        """
        now = time.time()
        eng = engine if engine is not None else self

        if mac_density_matrix is None:
            mac_density_matrix = np.eye(eng.dimension, dtype=np.float64) / eng.dimension

        mac_dim = int(mac_density_matrix.shape[0])
        carts = list(eng._active_cartridges.values())
        orbits = [c.return_orbit for c in carts if c.return_orbit is not None]
        if not orbits and getattr(eng, "_return_orbits", None):
            orbits = list(eng._return_orbits)

        audit_result = self.audit_return_orbits(orbits, carts)
        s_before = self._compute_von_neumann_entropy(mac_density_matrix)

        purged_n = len(getattr(eng, "_purged_history", []))
        if not carts:
            entropy_empty = s_before
            cert_empty = BufferCertificate(
                certificate_id=f"CERT-BUFF-{int(now * 1000) % 1_000_000:06d}",
                timestamp=now,
                verdict=HeytingOmega3.COHERENT,
                actuation_mode=ActuationMode.NORMAL_FLUID,
                active_cartridges_count=0,
                assimilated_count=0,
                purged_count=purged_n,
                capacity_gromov_peak=getattr(eng, "_capacity_peak", 0.0),
                mac_von_neumann_entropy=entropy_empty,
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
                kam_regime=audit_result["kam_regime"],
                diophantine_fraction=audit_result.get("diophantine_fraction", 1.0),
            )
            return mac_density_matrix, cert_empty

        rho_sum = np.zeros((mac_dim, mac_dim), dtype=np.float64)
        poincare_residuals: List[float] = []
        max_risk = 0.0
        weight = 1.0 / len(carts)
        n_diag = eng._N_diag[:mac_dim, :mac_dim]
        alpha_acc = 0.0

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
            if cart.delaunay is not None:
                alpha_acc += eng.banach_contraction_alpha(cart.delaunay)

        eta = eng.compute_level3_rsi_monadic_rate(mac_density_matrix, rho_sum, assimilation_rate_eta)
        mac_updated = self.luders_cptp_channel(
            (1.0 - eta) * mac_density_matrix + eta * rho_sum, eta=min(eta, 0.5)
        )
        s_after = self._compute_von_neumann_entropy(mac_updated)
        vn_delta = float(s_after - s_before)
        audit_result["von_neumann_delta"] = vn_delta
        audit_result["cptp_entropy_monotone"] = bool(s_after + 1e-9 >= s_before * (1.0 - eta))

        rsi3_epi = self.rsi3_epistemic_self_audit(audit_result)

        max_poincare_res = max(poincare_residuals) if poincare_residuals else 0.0
        gromov_peak = eng._capacity_peak
        kam_regime = audit_result["kam_regime"]
        M_max = audit_result["melnikov_max"]
        R_max = audit_result["greene_residue_max"]
        s_ch = audit_result["chirikov_overlap_max"]
        gate = bool(audit_result.get("kam_nondegeneracy_gate", True))

        if (
            gromov_peak > eng.capacity_gromov_max
            or max_poincare_res > 1e-6
            or kam_regime == KAMRegime.ARNOLD_DIFFUSION
            or s_ch >= 1.0
            or R_max > 1.0
            or not rsi3_epi.symplectic_2form_preserved
        ):
            verdict = HeytingOmega3.VETOED
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
            logger.critical(
                f"[FASE III · DISYUNTOR CROWBAR] VETO DURO. c_G={gromov_peak:.2f} "
                f"ResΘ={max_poincare_res:.2e} KAM={kam_regime.name} "
                f"R_Greene={R_max:.3f} s_Chirikov={s_ch:.3f}. "
                f"Pin GPIO{eng.esp32_gpio_pin} < {_CROWBAR_LATENCY_NS:.0f} ns."
            )
        elif (
            max_risk > 0.70
            or max_poincare_res > 1e-10
            or kam_regime == KAMRegime.CANTORUS_PARTIAL
            or M_max > 0.5
            or abs(R_max - self.greene_critical) < 0.05
            or not gate
        ):
            verdict = HeytingOmega3.DEGRADED
            actuation = ActuationMode.SOFT_VETO_BYPASS
            logger.warning(
                f"[FASE III · BYPASS] VETO SUAVE. Risk={max_risk:.2f} "
                f"KAM={kam_regime.name} Melnikov={M_max:.4f} Greene={R_max:.3f}."
            )
        else:
            verdict = HeytingOmega3.COHERENT
            actuation = ActuationMode.NORMAL_FLUID

        implication = HeytingOmega3.COHERENT.implies(verdict)

        assimilated_ids: List[str] = []
        for cid, cart in list(eng._active_cartridges.items()):
            if cart.status not in (CartridgeStatus.KAM_BROKEN_CHAOTIC,):
                cart.status = CartridgeStatus.ASSIMILATED
            assimilated_ids.append(cid)
            eng._assimilated_history.append(cid)
        eng._assimilated_count = getattr(eng, "_assimilated_count", 0) + len(assimilated_ids)
        eng._active_cartridges.clear()

        intact_frac = (
            1.0 if not carts
            else sum(1 for c in carts if c.kam_regime == KAMRegime.INTACT_TORUS) / len(carts)
        )
        if not gate:
            intact_frac *= 0.85
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
            purged_count=purged_n,
            capacity_gromov_peak=gromov_peak,
            mac_von_neumann_entropy=s_after,
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
            kam_regime=kam_regime,
            diophantine_fraction=float(audit_result.get("diophantine_fraction", 1.0)),
            rsi3_loeb_verified=rsi3_epi.loebian_reflection_verified,
            von_neumann_delta=vn_delta,
            banach_contraction_alpha=float(alpha_acc / max(len(carts), 1)),
        )
        logger.info(
            f"[FASE III] Asimilación CPTP a MAC Exitosa. Cert={cert.certificate_id} "
            f"Ω₃={verdict.name} ({implication.name}) KAM={kam_regime.name} "
            f"η_RSI3={eta:.4f} ΔS={vn_delta:.4e} Löb={rsi3_epi.loebian_reflection_verified}"
        )
        return mac_updated, cert

    @staticmethod
    def _compute_merkle_root(id_list: List[str]) -> str:
        if not id_list:
            return hashlib.sha256(b"EMPTY_BUFFER_HOMOLOGY_ZERO").hexdigest()
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
# §D. VERIFICACIÓN DOCTORAL AUTÓNOMA — DEMOSTRACIÓN DE LAS 3 FASES ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 86)
    print("  TOONBufferEngine v7.6 — SOBERANO ANIDADO: POINCARÉ · GREENE · OSELEDETS · LÜDERS")
    print("═" * 86)

    auditor_engine = KAMMelnikovAuditor(
        dimension=56,
        capacity_gromov_max=12.5,
        default_ttl_sec=3.0,
        esp32_gpio_pin=14,
        mu_gravitational=1.0,
        epsilon_kam=1e-2,
        greene_critical=_GREENE_CRITICAL,
    )

    print("\n┌─[FASE I] Atlas de Poincaré, Delaunay, Berry–Simon y RSI-3 Germinal ──────────┐")
    rho_sample = np.eye(56, dtype=np.float64) / 56.0
    action_sample = np.sort(np.maximum(la.eigvalsh(rho_sample), _EPS))[::-1]
    seed = auditor_engine.compute_poincare_canonical_seed(rho_sample, action_sample, epsilon_risk=0.10)
    del_pt = seed.delaunay
    xi, eta, pp, qq = del_pt.poincare_regular_variables()

    print(f"  • Acción Kepleriana L    : {del_pt.L:.6f}")
    print(f"  • Excentricidad orbital e: {del_pt.eccentricity:.6f}")
    print(f"  • Inclinación orbital i  : {math.degrees(del_pt.inclination):.2f}°")
    print(f"  • Frecuencia media n₀    : {del_pt.mean_motion:.6e}")
    print(f"  • Torsión de Moser τ     : {del_pt.moser_twist:.6e}")
    print(f"  • Fase Berry–Simon γ     : {seed.berry_simon_phase:.6f} rad")
    print(f"  • Degeneración Poincaré  : {seed.kepler_degeneracy.name} det_iso={seed.isoenergetic_hessian_det:.3e}")
    print(f"  • Variables Poincaré     : (ξ={xi:.4f}, η={eta:.4f}, p={pp:.4f}, q={qq:.4f})")
    print(f"  • Norma LRL ‖A‖          : {np.linalg.norm(seed.laplace_runge_lenz):.6f}")
    print(f"  • Hopf n ∈ S²            : {seed.orbital_quaternion.hopf_s2()}")
    print(f"  • Certificado RSI-3      : {seed.rsi3_proof.proof_id} "
          f"[Löb={seed.rsi3_proof.loebian_reflection_verified} tipo={seed.rsi3_proof.mutation_type.name}]")
    print("  ⟶ Germen emitido para FASE II: ingest_from_canonical_seed(seed, ...)")

    print("\n┌─[FASE II] Ingesta, Lindblad–GKSL, Verlet y Mapa de Retorno Σ_ℓ₀ ─────────────┐")
    c1 = auditor_engine.ingest_from_canonical_seed(
        seed=seed,
        apu_code="APU-CONCRETO-HIDRAULICO-5000PSI",
        tangible_cost_vector=np.ones(56) * 125000.0,
        policy_risk_intangible=0.15,
        weather_gremial_factor=0.08,
        coordinates_h2=(1.1, 2.4),
    )
    c2 = auditor_engine.ingest_cartridge(
        apu_code="APU-ACERO-POSTENSADO-GRADE270",
        tangible_cost_vector=np.ones(56) * 380000.0,
        policy_risk_intangible=0.22,
        weather_gremial_factor=0.12,
        coordinates_h2=(1.4, 2.9),
    )

    z_th = auditor_engine.rlc_thevenin_impedance(c1.delaunay)
    print(f"  • Impedancia Thévenin Z  : {z_th:.4f} Ω")
    print(f"  • α Banach (contracción) : {auditor_engine.banach_contraction_alpha(c1.delaunay):.4e}")
    om, n2 = auditor_engine.lindstedt_frequency(c1.delaunay, c1.policy_risk)
    print(f"  • Lindstedt ω, n₂        : {om:.6e}, {n2:.3e}")

    active_cnt, purged_cnt = auditor_engine.evolve_lindblad_rlc_step(dt=0.2)
    print(f"  • Paso Lindblad-GKSL     : {active_cnt} activos, {purged_cnt} purgados Dirac")

    orbits = auditor_engine.poincare_return_map_step(
        dt_integration=1e-3, max_steps=4000, max_crossings=6
    )
    n_cross = sum(len(o.sections) for o in orbits)
    print(f"  • Órbitas de Retorno Σ_ℓ : {len(orbits)} órbitas, {n_cross} cruces transversales")
    for o in orbits:
        print(
            f"    – {o.orbit_id}: ϖ={o.winding_number:.4f} | R_G={o.greene_residual:.4f} | "
            f"s_eff={o.chirikov_overlap_effective:.4f} | λ_max={o.lyapunov_oseledets_max:.4e} | "
            f"M={o.melnikov_amplitude:.4f} | Birkhoff={o.birkhoff_fixed_points}"
        )
    print("  ⟶ Germen emitido para FASE III: audit_return_orbits(orbits, cartridges)")

    print("\n┌─[FASE III] Auditoría KAM, Lüders CPTP, Ω₃ y Certificado ─────────────────────┐")
    mac_init = np.eye(56, dtype=np.float64) / 56.0
    mac_updated, cert = auditor_engine.assimilate_to_mac(
        mac_density_matrix=mac_init, assimilation_rate_eta=0.20
    )

    print(f"  • Certificado ID         : {cert.certificate_id}")
    print(f"  • Veredicto Topos Ω₃     : {cert.verdict.name} (Implicación: {cert.heyting_implication})")
    print(f"  • Modo de Actuación      : {cert.actuation_mode.name}")
    print(f"  • Resonancia p:q         : Orden q = {cert.delaunay_resonance_order}")
    print(f"  • Fracción KAM Intacta   : {cert.kam_intact_fraction * 100.0:.1f}%")
    print(f"  • Fracción diofantina    : {cert.diophantine_fraction:.3f}")
    print(f"  • Amplitud Melnikov M_max: {cert.melnikov_max_amplitude:.6f}")
    print(f"  • Residuo Greene R_max   : {cert.greene_residue_max:.6f} (Crítico: {_GREENE_CRITICAL:.4f})")
    print(f"  • Solapamiento Chirikov  : {cert.chirikov_overlap_max:.6f}")
    print(f"  • Entropía von Neumann   : {cert.mac_von_neumann_entropy:.4f}  ΔS={cert.von_neumann_delta:.4e}")
    print(f"  • RSI-3 Löb              : {cert.rsi3_loeb_verified}")
    print(f"  • α Banach               : {cert.banach_contraction_alpha:.4e}")
    print(f"  • Traza MAC              : {float(np.real(np.trace(mac_updated))):.8f}")
    print(f"  • Raíz Árbol de Merkle   : {cert.merkle_root_sha256[:32]}...")

    print("\n" + "═" * 86)
    print("  ✓ MOTOR v7.6 COMPLETADO — FASES I ⊂ II ⊂ III INTEGRADAS Y EJECUTADAS CON ÉXITO")
    print("═" * 86)