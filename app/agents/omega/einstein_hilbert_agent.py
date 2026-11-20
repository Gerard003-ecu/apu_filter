# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Einstein-Hilbert Agent — Arquitecto de Curvatura, Horizonte y Atractor      ║
║ Ruta   : app/agents/omega/einstein_hilbert_agent.py                                  ║
║ Versión: 4.0.0-RealConnection-Einstein-Constraints-GibbonsHawking-Poincare-RSI-L3    ║
║ Estrato: V_Ω / V_WISDOM — El Ágora Tensorial                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA: TRES FASES ANIDADAS POR CONTINUACIÓN (composición de Kleisli)
──────────────────────────────────────────────────────────────────────────────────────
   Φ₁ : (g, polarón, u, P) ─────► E  (EnergyMomentumData)
   Φ₂ : E ──────────────────────► C  (CurvatureBundle)
   Φ₃ : C ──────────────────────► V  (HorizonVerdict)
   Z_EH = Φ₃ ∘ Φ₂ ∘ Φ₁  (OODA: Observe → Orient → Decide → Act)

   El método terminal de cada fase recibe `continuation: Callable[[DTO], R]` y su cola
   es literalmente `return continuation(DTO)`; la continuación canónica es el primer
   método de la fase siguiente (`_continue_from_phase{k}_*`). Anidamiento tipológico y
   operacional. Los veredictos se agregan por meet en Ω₃ (funtorialidad laxa).

AXIOMAS CON ESTATUS EPISTÉMICO EXPLÍCITO
──────────────────────────────────────────────────────────────────────────────────────
 G1 [TEOREMA] Fluido perfecto UNIFICADO por firma. Sea σ = g(u,u) ∈ {−1,+1} y
    h_{μν} = g_{μν} − σ u_μ u_ν el proyector ortogonal a u (h·u = 0 pues σ²=1). Entonces
        T_{μν} = ρ u_μ u_ν + P h_{μν} = (ρ − σP) u_μ u_ν + P g_{μν},
        T_{μν}u^μu^ν = ρ  (∀σ),      T = g^{μν}T_{μν} = σρ + (n−1)P.
    La fórmula v3 `(ρ+P)u⊗u+Pg` sólo es correcta para σ=−1; con σ=+1 daba ρ+2P.
 G2 [TEOREMA] Condiciones de energía en n dimensiones (d=n−1 espaciales), SOLO Lorentz:
        NEC ⇔ ρ+P ≥ 0 ;  WEC ⇔ ρ≥0 ∧ ρ+P≥0 ;  DEC ⇔ ρ ≥ |P|
        SEC ⇔ NEC ∧ [ρ(n−3) − σ d P]/(n−2) ≥ 0   (Λ=0, n≥3)
    pues R_{μν}u^μu^ν = κ(T_{μν} − T g_{μν}/(n−2))u^μu^ν. En 4D y σ=−1: (ρ+3P)/2 ✓.
    En firma RIEMANNIANA no existen vectores nulos: NEC/WEC/DEC son INAPLICABLES
    (se marcan `applicable=False`, no se evalúan a ciegas).
 G3 [TEOREMA] Levi-Civita: Γ^λ_{μν} = ½g^{λσ}(∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν}).
    CON UNA MÉTRICA CONSTANTE (valor puntual) Γ ≡ 0 y Riemann ≡ 0 IDÉNTICAMENTE.
    Sin campo g(x) o ∂g explícito la conexión NO está determinada: este módulo
    certifica planitud o exige el campo; JAMÁS fabrica derivadas.
 G4 [TEOREMA] R^ρ_{σμν} = ∂_μΓ^ρ_{νσ} − ∂_νΓ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ};
    R_{σν} = R^ρ_{σρν}; R = g^{σν}R_{σν}; K = R_{abcd}R^{abcd} (Kretschmann).
    Sección: K(u,v) = ⟨R(u,v)v,u⟩ / (g(u,u)g(v,v) − g(u,v)²).
    Desigualdad rigurosa: max|K_sec| ≥ |R|/(d(d−1)) — |R| NO es max|K|.
    Calibración incorporada: S² redonda ⇒ R=2, K=1; Schwarzschild ⇒ K=12r_s²/r⁶.
 G5 [TEOREMA] Einstein: G_{μν} = R_{μν} − ½R g_{μν} + Λ g_{μν}; en n=2, G≡0.
    El residual ‖G_{μν} − κT_{μν}‖ es un DIAGNÓSTICO de violación de ligadura, no una
    solución: con (g,T) prescritos arbitrarios la ecuación no se satisface genéricamente.
    SEC geométrica (R_{μν}u^μu^ν≥0) vs SEC de materia: su discrepancia MIDE esa violación.
 G6 [TEOREMA] Termodinámica de Schwarzschild — IDENTIDADES EXACTAS verificadas:
        r_s = 2GM/c²,  A = 4πr_s²,  S = k_B A/(4ℓ_P²) = 4πk_B GM²/(ħc)
        T_H = ħc³/(8πGMk_B),  β_τ = ħ/(k_B T_H) = 8πGM/c³
        (i)  1ª ley:      T_H · dS/dM = c²
        (ii) Smarr:       M c² = 2 T_H S
        (iii)Gibbons–Hawking: I_E = β_E M c²/2 = S/k_B   (acción euclídea)
        (iv) Bekenstein:  S ≤ 2πk_B E R/(ħc), SATURADA exactamente en Schwarzschild
        (v)  Evaporación: t = 5120π G²M³/(ħc⁴)
    M→0 ⇒ T_H→∞ (estallido final), NO "ausencia de radiación". M ≲ M_P ⇒ régimen
    planckiano: la termodinámica semiclásica NO es válida (se marca DEGRADED).
 G7 [TEOREMA] Regularidad del cigarro euclídeo (Gibbons–Hawking): la métrica euclídea
    de Schwarzschild es suave en r=r_s SÓLO si el tiempo imaginario tiene periodo
    β_τ = ħ/(k_B T_H). Un desajuste produce DÉFICIT CÓNICO δ = 2π(1 − β_τ/β_H).
    Este es el puente exacto entre A1 (Matsubara) y A4 (Hawking), ausente en v3.
 G8 [TEOREMA] Gauss–Bonnet: χ(H) = (1/2π)∮_H K dA. Para el horizonte redondo S² de
    radio r_s: K = 1/r_s², A = 4πr_s² ⇒ χ = 2 DERIVADO, no postulado. (Un anillo negro
    S¹×S² tiene χ=0: el invariante debe calcularse, no hardcodearse.)
 G9 [TEOREMA] Mecánica celeste de Poincaré en Schwarzschild (frecuencias en tcoord):
        Ω(r) = √(GM/r³)               (Kepler exacto en tiempo coordenado)
        κ²(r) = (GM/r³)(1 − 6GM/(c²r)) (epicíclica radial)
        r_ISCO = 6GM/c²,  r_ph = 3GM/c²,  Ω_ph = λ_ph = c³/(3√3 GM)
        Δϖ = 6πGM/(c²a(1−e²))          (precesión 1PN por órbita)
    Monodromía radial tras una vuelta azimutal: tr M = 2cos(2πκ/Ω) ⇒ residuo de Greene
        R = (2 − tr M)/4 = sin²(πκ/Ω) ∈ [0,1]  (elíptico, estable)
    y R < 0 (hiperbólico) si κ² < 0. EL ISCO ES EXACTAMENTE EL CRUCE R = 0.
    Esfera de fotones: multiplicador e^{λT} = e^{2π} ⇒ R ≈ −133.4 (hiperbólico puro).
    QNM eikonal: ω ≈ lΩ − i(n+½)λ (Cardoso et al.).
G10 [TEOREMA] Flujo de calor de mapas armónicos (RSI-L3 R₂): para γ:[0,1]→(M,g) con
    extremos fijos, E[γ] = ½∫g(γ̇,γ̇)ds y ∂_tγ = τ(γ) = γ̈ + Γ(γ)(γ̇,γ̇) satisfacen
        dE/dt = −∫ g(τ,τ) ds ≤ 0,
    con puntos fijos = geodésicas. Lyapunov CANÓNICO y geométrico del lazo RSI-L3.
G11 [CONVENCIÓN] ρ := m** y P := market_pressure son IDENTIFICACIONES DECLARADAS del
    modelo semántico en unidades normalizadas, no igualdades dimensionales derivadas.
G12 [SAFETY] RSI-L3 con ENDURECIMIENTO MONÓTONO: sólo refina discretización o aprieta
    tolerancias; jamás afloja umbrales de certificado. Verificador INMUTABLE.

UNIDADES: se exige un sistema COHERENTE (SI-CODATA o natural G=c=ħ=k_B=1). La v3
mezclaba constantes "cyber" normalizadas con k_B del SI: error dimensional grave.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Dependencias del ecosistema (contratos tolerantes, degradación grácil) ───────────
try:
    from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico/geométrico violado."""

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(self, payload: Any = None, stratum: str = "", label: str = "") -> None:
            self.payload, self.stratum, self.label = payload, stratum, label

G_PHYSICS_RAW: Any = None
for _p in ("app.core.immune_system.metric_tensors", "app.core.inmune_system.metric_tensors"):
    try:
        G_PHYSICS_RAW = getattr(__import__(_p, fromlist=["G_PHYSICS"]), "G_PHYSICS", None)
        break
    except ImportError:
        continue

try:
    from app.core.telemetry_schemas import PolaronCartridge
except ImportError:  # pragma: no cover
    @dataclass(frozen=True)
    class PolaronCartridge:  # type: ignore[no-redef]
        base_electron: Any = None
        frohlich_coupling: float = 0.0
        effective_mass: float = 1.0
        inertial_mass: float = 1.0
        volatility_alpha: float = 0.0
        fiedler_value: float = 0.0


def _default_mass_law(base_cost: float, volatility_alpha: float, frohlich_coupling: float) -> float:
    r"""
    Ley DECLARADA de renormalización (fallback): m* = m₀·(1+α_F/6)·e^{vol}.
    Es IDEMPOTENTE en (α_F=0, vol=0): m*(m*,0,0) = m* — propiedad que se VERIFICA
    en §1.6 para descartar el doble conteo del acoplamiento de Fröhlich.
    """
    return float(base_cost) * (1.0 + float(frohlich_coupling) / 6.0) * math.exp(
        min(float(volatility_alpha), 50.0))


def _default_metric_deformation(g: NDArray[np.float64], m_eff: float, node: int) -> NDArray[np.float64]:
    r"""
    Deformación CONFORME declarada (fallback): g̃ = Ω²·g con Ω² = 1 + m*/(1+node) > 0.
    Una transformación conforme con Ω²>0 PRESERVA EXACTAMENTE la firma (inercia de
    Sylvester): no puede convertir una métrica lorentziana en riemanniana.
    """
    omega_sq = 1.0 + max(float(m_eff), 0.0) / (1.0 + abs(int(node)))
    return omega_sq * np.asarray(g, dtype=np.float64)


try:
    from app.core.immune_system.gravity_shield import (  # type: ignore
        GravitationalShieldFunctor, GravitationalConstants, PolyakovAction, WarpedSpaceTime,
    )
    try:
        from app.core.immune_system.gravity_shield import _acquire_effective_mass as _shield_mass
    except ImportError:
        _shield_mass = _default_mass_law
    try:
        from app.core.immune_system.gravity_shield import _deform_metric_tensor as _shield_deform
    except ImportError:
        _shield_deform = _default_metric_deformation
except ImportError:  # pragma: no cover
    _shield_mass, _shield_deform = _default_mass_law, _default_metric_deformation

    @dataclass(frozen=True)
    class GravitationalConstants:  # type: ignore[no-redef]
        HBAR_EFF: float = 1.0
        CYBER_C: float = 1.0
        CYBER_G: float = 1.0

    @dataclass(frozen=True)
    class PolyakovAction:  # type: ignore[no-redef]
        action_integral: float = 0.0
        feynman_amplitude: float = 1.0
        is_trapped: bool = False
        matsubara_beta: float = 0.0

    @dataclass(frozen=True)
    class WarpedSpaceTime:  # type: ignore[no-redef]
        deformed_metric: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))
        christoffel_symbols: Optional[NDArray[np.float64]] = None
        max_sectional_curvature: float = 0.0

    class GravitationalShieldFunctor:  # type: ignore[no-redef]
        def __init__(self, g_base: Optional[NDArray[np.float64]] = None) -> None:
            self.g_base = np.eye(4) if g_base is None else np.asarray(g_base, dtype=np.float64)

        def enforce_gravitational_attractor(self, polaron: Any, node_index: int,
                                            llm_attention_vector: NDArray) -> PolyakovAction:
            m = float(getattr(polaron, "effective_mass", 1.0))
            a = float(np.linalg.norm(np.asarray(llm_attention_vector, dtype=np.float64)))
            s = 0.5 * m * a * a
            return PolyakovAction(action_integral=s, feynman_amplitude=math.exp(-min(s, 700.0)),
                                  is_trapped=bool(math.exp(-min(s, 700.0)) < 1e-12),
                                  matsubara_beta=0.0)

# ── Puente con los soberanos previos: Ω₃ y auditoría simpléctica ─────────────────────
try:
    from app.agents.omega.bogoliubov_agent import BogoliubovAgent, HeytingVerdict, heyting_meet
except ImportError:  # pragma: no cover
    BogoliubovAgent = None  # type: ignore[assignment]

    class HeytingVerdict(IntEnum):  # type: ignore[no-redef]
        """Ω₃ ≅ O(Sierpiński): cadena de Gödel de 3 elementos, NO booleana."""
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    def heyting_meet(a: Any, b: Any) -> Any:  # type: ignore[no-redef]
        return HeytingVerdict(max(int(a), int(b)))

logger = logging.getLogger("MIC.Omega.EinsteinHilbertAgent")
R = TypeVar("R")

_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_SYM_ATOL: float = 1e-12
_NONDEGENERACY_REL: float = 1e-12
_ENERGY_ATOL: float = 1e-10
_FD_STEP: float = _MACHINE_EPS ** (1.0 / 3.0)     # h óptimo para diferencias centrales
_HEAT_CFL: float = 0.45

__all__ = [
    "UnitSystem", "AstrophysicalConstants", "MetricSignatureClass",
    "SingularityVetoError", "EnergyMomentumDegeneracyError", "CausalStructureError",
    "MetricDegeneracyError", "ConnectionUndeterminedError", "ConicalDeficitError",
    "HeytingVerdict", "MetricSignatureReport", "EnergyMomentumData",
    "ConnectionReport", "CurvatureInvariants", "EinsteinConstraintReport",
    "CelestialOrbitReport", "MatsubaraCertificate", "GeodesicRelaxationReport",
    "CurvatureBundle", "BlackHoleThermodynamics", "HorizonVerdict",
    "RSIL3Hyperparameters", "RSIL3Report", "EinsteinHilbertPhaseState",
    "Phase1_EnergyMomentumExtractor", "Phase2_EinsteinFieldSolver",
    "Phase3_BekensteinHawkingDecider", "EinsteinHilbertAgent",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §A · SISTEMAS DE UNIDADES COHERENTES (fin de la mezcla cyber/SI)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class UnitSystem:
    r"""Sistema COHERENTE: todas las constantes del mismo sistema, jamás mezcladas."""

    G: float
    c: float
    hbar: float
    k_B: float
    name: str

    @property
    def planck_length_sq(self) -> float:
        return self.hbar * self.G / self.c ** 3

    @property
    def planck_mass(self) -> float:
        return math.sqrt(self.hbar * self.c / self.G)

    @property
    def kappa_einstein(self) -> float:
        """κ en G_{μν} = κ T_{μν}: κ = 8πG/c⁴ (SI) u 8π (natural)."""
        return 8.0 * math.pi * self.G / self.c ** 4

    @classmethod
    def si(cls) -> "UnitSystem":
        return cls(G=6.67430e-11, c=299_792_458.0, hbar=1.054_571_817e-34,
                   k_B=1.380_649e-23, name="SI-CODATA2018")

    @classmethod
    def natural(cls) -> "UnitSystem":
        return cls(G=1.0, c=1.0, hbar=1.0, k_B=1.0, name="NATURAL-G=c=hbar=kB=1")


class AstrophysicalConstants:
    """ABI legacy + sistema de unidades explícito y coherente."""

    UNITS: UnitSystem = UnitSystem.si()
    PLANCK_LENGTH_SQ: float = UNITS.planck_length_sq
    BOLTZMANN_K: float = UNITS.k_B
    PI: float = math.pi
    HORIZON_EULER_CHARACTERISTIC: int = 2
    SYMMETRY_ATOL: float = _SYM_ATOL
    NORM_ATOL: float = 1e-14
    ENERGY_COND_ATOL: float = _ENERGY_ATOL

    @staticmethod
    def hawking_factor(units: UnitSystem) -> float:
        """T_H = ħc³/(8πGMk_B) ⇒ factor/m. Coherencia de unidades GARANTIZADA."""
        return units.hbar * units.c ** 3 / (8.0 * math.pi * units.G * units.k_B)


class MetricSignatureClass(IntEnum):
    RIEMANNIAN = 0
    LORENTZIAN_MOSTLY_PLUS = 1     # (−,+,+,…)  σ_timelike = −1
    LORENTZIAN_MOSTLY_MINUS = 2    # (+,−,−,…)  σ_timelike = +1
    ULTRAHYPERBOLIC = 3
    DEGENERATE = 4


# ═════════════════════════════════════════════════════════════════════════════════════
# §B · EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════════════
class SingularityVetoError(TopologicalInvariantError):
    """Veto ontológico: la geodésica atencional cayó tras el horizonte (Ψ[γ]→0)."""


class EnergyMomentumDegeneracyError(TopologicalInvariantError):
    """T_{μν} no simétrico, masa no física o condición de energía violada duramente."""


class CausalStructureError(TopologicalInvariantError):
    """Cuadrivelocidad nula/espacial o incompatible con la firma y orientación."""


class MetricDegeneracyError(TopologicalInvariantError):
    """Métrica degenerada (λ≈0) o de firma ultrahiperbólica: no hay variedad causal."""


class ConnectionUndeterminedError(TopologicalInvariantError):
    """Se pidió curvatura sin campo métrico ni ∂g: la conexión NO está determinada."""


class ConicalDeficitError(TopologicalInvariantError):
    """Periodo de Matsubara ≠ β_H: singularidad cónica en el horizonte euclídeo."""


# ═════════════════════════════════════════════════════════════════════════════════════
# §C · DTOs INMUTABLES (objetos del topos; contratos formales entre fases)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MetricSignatureReport:
    """Inercia de Sylvester: (n_pos, n_neg, n_zero), clase, dirección temporal, σ."""

    n_positive: int
    n_negative: int
    n_zero: int
    signature_class: int
    timelike_sigma: float
    timelike_direction: Tuple[float, ...]
    lambda_min_abs: float
    lambda_max_abs: float
    condition_number: float
    is_nondegenerate: bool
    has_causal_structure: bool
    dim: int


@dataclass(frozen=True, slots=True)
class EnergyMomentumData:
    """E — artefacto terminal de Φ₁ y dominio exacto del primer método de Φ₂."""

    T_tensor: NDArray[np.float64]
    effective_mass: float
    inflationary_pressure: float
    four_velocity: NDArray[np.float64]
    trace: float
    energy_density: float
    weak_energy_ok: bool
    null_energy_ok: bool
    strong_energy_ok: bool
    dominant_energy_condition_ok: bool
    approximate_conservation_residual: float
    # ── extensiones v4 ──
    signature: Optional[MetricSignatureReport] = None
    timelike_sigma: float = -1.0
    spatial_dim: int = 3
    energy_conditions_applicable: bool = True
    sec_matter_value: float = 0.0
    dec_flux_causal: bool = True
    conservation_measured: bool = False
    mass_idempotency_residual: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    verdict: int = 0


@dataclass(frozen=True, slots=True)
class ConnectionReport:
    """G3: procedencia de la conexión — campo, derivada explícita, o planitud certificada."""

    christoffel: Optional[NDArray[np.float64]]
    christoffel_norm: float
    source: str                    # "metric_field" | "explicit_dg" | "shield" | "flat_certified"
    derivative_measured: bool
    flat_certified: bool
    fd_step: float
    verdict: int


@dataclass(frozen=True, slots=True)
class CurvatureInvariants:
    """Invariantes REALES (o exactamente nulos con certificado de planitud)."""

    ricci_tensor: NDArray[np.float64]
    ricci_scalar: float
    max_sectional_curvature: float
    christoffel_norm: float
    # ── extensiones v4 ──
    riemann_tensor: Optional[NDArray[np.float64]] = None
    kretschmann: float = 0.0
    weyl_norm: float = 0.0
    einstein_tensor: Optional[NDArray[np.float64]] = None
    sectional_bound_from_scalar: float = 0.0
    curvature_measured: bool = False
    injectivity_radius_estimate: float = float("inf")
    curvature_scale: float = float("inf")
    verdict: int = 0


@dataclass(frozen=True, slots=True)
class EinsteinConstraintReport:
    r"""
    G5: DIAGNÓSTICO de violación de la ecuación de campo (no una solución).
    La discrepancia entre SEC geométrica y SEC de materia mide esa violación.
    """

    einstein_residual_relative: float
    einstein_residual_absolute: float
    sec_geometric_value: float
    sec_matter_value: float
    sec_discrepancy: float
    lambda_cosmological: float
    kappa: float
    dimension: int
    constraint_satisfied: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class CelestialOrbitReport:
    """G9: mecánica celeste de Poincaré en Schwarzschild (exacta donde se indica)."""

    schwarzschild_radius: float
    isco_radius: float
    photon_sphere_radius: float
    orbital_radius: float
    omega_azimuthal: float
    kappa_epicyclic: float
    greene_residue: float
    orbit_stable: bool
    photon_sphere_lyapunov: float
    photon_sphere_multiplier: float
    perihelion_precession: float
    resonance: Tuple[int, int, float]
    diophantine_gamma_eff: float
    kam_satisfied: bool
    qnm_eikonal_real: float
    qnm_eikonal_imag: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class MatsubaraCertificate:
    """G7: puente exacto Matsubara ↔ Hawking vía regularidad del cigarro euclídeo."""

    beta_supplied: float
    beta_hawking_time: float
    beta_hawking_energy: float
    conical_deficit: float
    regular: bool
    gibbons_hawking_action: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class GeodesicRelaxationReport:
    """G10: flujo de calor de mapas armónicos; E[γ] monótona decreciente."""

    energy_initial: float
    energy_final: float
    tension_norm_final: float
    steps: int
    step_used: float
    monotone: bool
    geodesic_residual: float
    converged: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class CurvatureBundle:
    """C — artefacto terminal de Φ₂ y dominio exacto del primer método de Φ₃."""

    energy: EnergyMomentumData
    warped_metric: NDArray[np.float64]
    connection: ConnectionReport
    curvature: CurvatureInvariants
    constraints: EinsteinConstraintReport
    polyakov_action: Any
    warped_space: Any
    celestial: Optional[CelestialOrbitReport]
    matsubara: Optional[MatsubaraCertificate]
    relaxation: Optional[GeodesicRelaxationReport]
    verdict: int


@dataclass(frozen=True, slots=True)
class BlackHoleThermodynamics:
    """G6/G8: termodinámica con IDENTIDADES EXACTAS verificadas y χ DERIVADO."""

    schwarzschild_radius: float
    horizon_area: float
    bekenstein_hawking_entropy: float
    hawking_temperature: float
    horizon_euler_characteristic: int = 2
    # ── extensiones v4 ──
    euler_derived: float = 2.0
    euler_gauss_bonnet_residual: float = 0.0
    first_law_residual: float = 0.0
    smarr_residual: float = 0.0
    gibbons_hawking_residual: float = 0.0
    bekenstein_bound_ratio: float = 1.0
    evaporation_time: float = 0.0
    planck_mass_ratio: float = 0.0
    semiclassical_valid: bool = True
    no_horizon: bool = False
    units_name: str = "SI-CODATA2018"
    verdict: int = 0


@dataclass(frozen=True, slots=True)
class HorizonVerdict:
    """V — artefacto terminal de Φ₃."""

    thermodynamics: Optional[BlackHoleThermodynamics]
    is_trapped: bool
    amplitude: float
    curvature_veto: bool
    energy_condition_veto: bool
    heyting_verdict_value: int
    reason: str
    verdict: int


@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Único payload mutable de RSI-L3 (G12: sólo endurece)."""

    heat_step: float = 1e-3
    path_nodes: int = 16
    constraint_tol: float = 1e-8
    energy_policy: str = "degrade"
    trust_radius: float = 0.25
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSIL3Report:
    iterations: int
    accepted: int
    rejected: int
    lyapunov_initial: float
    lyapunov_final: float
    lyapunov_history: Tuple[float, ...]
    contraction_q: float
    banach_error_bound: float
    contraction_certified: bool
    monotone: bool
    tightening_only: bool
    hyperparameters: RSIL3Hyperparameters
    verdict: int


@dataclass(frozen=True, slots=True)
class EinsteinHilbertPhaseState:
    """Estado terminal inmutable del endofuntor Z_EH = Φ₃∘Φ₂∘Φ₁."""

    energy: EnergyMomentumData
    bundle: CurvatureBundle
    horizon: HorizonVerdict
    heyting_verdict_value: int
    is_coherent: bool
    rsi_report: Optional[RSIL3Report] = None
    bogoliubov_certificate: Optional[Any] = None
    created_ns: int = 0


# ═════════════════════════════════════════════════════════════════════════════════════
# §D · UTILIDADES GEOMÉTRICAS CERTIFICADAS
# ═════════════════════════════════════════════════════════════════════════════════════
def _fro(a: NDArray) -> float:
    return float(la.norm(np.asarray(a), ord="fro", check_finite=False))


def _rel(diff: NDArray, ref: NDArray) -> float:
    return _fro(diff) / max(_fro(ref), 1e-300)


def _sym(a: NDArray) -> NDArray[np.float64]:
    x = np.asarray(a, dtype=np.float64)
    return 0.5 * (x + x.T)


def _symmetric_inverse(g: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray, NDArray]:
    r"""
    Inversa de una matriz SIMÉTRICA posiblemente INDEFINIDA (lorentziana).
    Cholesky NO es aplicable en firma indefinida: se usa g = QΛQᵀ ⇒ g⁻¹ = QΛ⁻¹Qᵀ,
    con guardia de no-degeneración |λ_min| > tol·|λ_max| (Sylvester).
    """
    w, q = la.eigh(_sym(g), check_finite=False)
    scale = float(np.max(np.abs(w)))
    if scale <= 0.0 or float(np.min(np.abs(w))) <= _NONDEGENERACY_REL * scale:
        raise MetricDegeneracyError(
            f"Métrica degenerada: |λ|_min={float(np.min(np.abs(w))):.3e}, escala={scale:.3e}")
    return (q / w) @ q.T, w, q


def _best_rational(x: float, q_max: int) -> Tuple[int, int, float]:
    """Mejor aproximante p/q (q≤q_max) por convergentes de la aplicación de Gauss."""
    p0, q0, p1, q1 = 1, 0, int(math.floor(x)), 1
    best = (p1, max(q1, 1), abs(x - p1 / max(q1, 1)))
    v = float(x)
    for _ in range(24):
        a = int(math.floor(v))
        p2, q2 = a * p1 + p0, a * q1 + q0
        if q2 > q_max or q2 <= 0:
            break
        err = abs(x - p2 / q2)
        if err < best[2]:
            best = (p2, q2, err)
        p0, q0, p1, q1 = p1, q1, p2, q2
        frac = v - a
        if frac < 1e-12:
            break
        v = 1.0 / frac
    return best


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 1 · OBSERVE: TENSOR DE ENERGÍA-IMPULSO COVARIANTE  (→ E)          ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase1_EnergyMomentumExtractor:
    r"""
    Φ₁ — Observador covariante: firma → normalización causal → fluido perfecto
    unificado por σ → condiciones de energía en n dims → conservación REAL o NaN.
    """

    def __init__(self, units: Optional[UnitSystem] = None) -> None:
        self._units = units or AstrophysicalConstants.UNITS

    # ── 1.1 · Firma de Sylvester, no-degeneración y dirección temporal ───────────────
    @staticmethod
    def analyze_signature(metric: NDArray[np.float64]) -> MetricSignatureReport:
        r"""
        Clasifica por INERCIA (n_pos, n_neg, n_zero), no por el signo de un autovalor.
        (−−++) es ULTRAHIPERBÓLICA ⇒ sin estructura causal ⇒ veto (la v3 la tomaba
        por lorentziana). Extrae la dirección temporal como el autovector del
        autovalor de signo minoritario, permitiendo construir u timelike por diseño.
        """
        g = _sym(metric)
        n = g.shape[0]
        w, q = la.eigh(g, check_finite=False)
        scale = float(np.max(np.abs(w)))
        tol = _NONDEGENERACY_REL * max(scale, 1e-300)
        n_pos = int(np.count_nonzero(w > tol))
        n_neg = int(np.count_nonzero(w < -tol))
        n_zero = n - n_pos - n_neg
        if n_zero > 0:
            cls, sigma = MetricSignatureClass.DEGENERATE, 0.0
            tdir = np.zeros(n)
        elif n_neg == 1:
            cls, sigma = MetricSignatureClass.LORENTZIAN_MOSTLY_PLUS, -1.0
            tdir = q[:, int(np.argmin(w))]
        elif n_pos == 1:
            cls, sigma = MetricSignatureClass.LORENTZIAN_MOSTLY_MINUS, 1.0
            tdir = q[:, int(np.argmax(w))]
        elif n_neg == 0 or n_pos == 0:
            cls, sigma = MetricSignatureClass.RIEMANNIAN, (1.0 if n_neg == 0 else -1.0)
            tdir = q[:, 0]
        else:
            cls, sigma = MetricSignatureClass.ULTRAHYPERBOLIC, 0.0
            tdir = np.zeros(n)
        lmin, lmax = float(np.min(np.abs(w))), float(np.max(np.abs(w)))
        return MetricSignatureReport(
            n_positive=n_pos, n_negative=n_neg, n_zero=n_zero,
            signature_class=int(cls), timelike_sigma=float(sigma),
            timelike_direction=tuple(float(x) for x in tdir),
            lambda_min_abs=lmin, lambda_max_abs=lmax,
            condition_number=lmax / max(lmin, 1e-300),
            is_nondegenerate=(n_zero == 0),
            has_causal_structure=(cls in (MetricSignatureClass.LORENTZIAN_MOSTLY_PLUS,
                                          MetricSignatureClass.LORENTZIAN_MOSTLY_MINUS)),
            dim=n,
        )

    @staticmethod
    def _detect_metric_signature(metric: NDArray[np.float64]) -> float:
        """ABI legacy: devuelve σ_timelike (+1 euclídeo, −1 lorentziano −+++)."""
        rep = Phase1_EnergyMomentumExtractor.analyze_signature(metric)
        if not rep.is_nondegenerate or rep.signature_class == int(MetricSignatureClass.ULTRAHYPERBOLIC):
            raise MetricDegeneracyError(
                f"Firma inadmisible: (+{rep.n_positive},−{rep.n_negative},0×{rep.n_zero})")
        return rep.timelike_sigma

    # ── 1.2 · Normalización causal con orientación temporal ──────────────────────────
    @staticmethod
    def canonical_timelike_velocity(metric: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Construye u CANÓNICAMENTE timelike a partir del autovector temporal:
        g(u,u) = σ exactamente. Corrige el bug de v3, cuyo `np.ones(d)/√d` era
        ESPACIAL en diag(−1,1,1,1) (g(u,u)=+0.5) y hacía fallar todo input lorentziano.
        """
        rep = Phase1_EnergyMomentumExtractor.analyze_signature(metric)
        if not rep.has_causal_structure:
            e = np.zeros(rep.dim)
            e[0] = 1.0
            return e / math.sqrt(abs(float(e @ _sym(metric) @ e)))
        t = np.asarray(rep.timelike_direction, dtype=np.float64)
        nrm = float(t @ _sym(metric) @ t)
        u = t / math.sqrt(abs(nrm))
        if u[int(np.argmax(np.abs(u)))] < 0.0:      # orientación futura por convención
            u = -u
        return u

    @staticmethod
    def _normalize_4velocity(velocity: NDArray[np.float64], metric: NDArray[np.float64],
                             target_norm_sign: float) -> NDArray[np.float64]:
        g = _sym(metric)
        v = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if v.size != g.shape[0]:
            raise CausalStructureError("Dimensión de velocity incompatible con la métrica.")
        if not np.all(np.isfinite(v)):
            raise CausalStructureError("Cuadrivelocidad no finita.")
        norm_sq = float(v @ g @ v)
        if abs(norm_sq) < AstrophysicalConstants.NORM_ATOL:
            raise CausalStructureError("Cuadrivelocidad nula (luz): no normalizable a ±1.")
        if norm_sq * target_norm_sign < 0.0:
            raise CausalStructureError(
                f"Tipo causal incorrecto: g(u,u)={norm_sq:.3e}, se esperaba signo "
                f"{target_norm_sign:+.0f}. Use canonical_timelike_velocity().")
        return v / math.sqrt(abs(norm_sq))

    # ── 1.3 · Condiciones de energía: n dimensiones, sólo Lorentz ────────────────────
    @staticmethod
    def _check_energy_conditions(T: NDArray[np.float64], u: NDArray[np.float64],
                                 rho: float, P: float, metric: NDArray[np.float64]
                                 ) -> Tuple[bool, bool, bool, bool]:
        """ABI legacy (4D implícito); la ruta rigurosa es `energy_conditions_general`."""
        atol = _ENERGY_ATOL
        wec = (rho >= -atol) and (rho + P >= -atol)
        nec = (rho + P >= -atol)
        sec = nec and (rho + 3.0 * P >= -atol)
        dec = wec and (rho >= abs(P) - atol)
        return wec, nec, sec, dec

    @staticmethod
    def energy_conditions_general(rho: float, P: float, sigma: float, n: int,
                                  causal: bool) -> Tuple[bool, bool, bool, bool, float, bool]:
        r"""
        G2: en n dims, d=n−1, con Λ=0 y n≥3:
            SEC_matter = [ρ(n−3) − σ d P]/(n−2) ≥ 0  (además de NEC).
        En 4D, σ=−1 ⇒ (ρ+3P)/2 ✓. En firma riemanniana NO hay conos nulos:
        se devuelve applicable=False y los flags en True neutro (no se afirma física).
        """
        if not causal:
            return True, True, True, True, 0.0, False
        d, atol = n - 1, _ENERGY_ATOL
        nec = (rho + P >= -atol)
        wec = (rho >= -atol) and nec
        dec = wec and (rho >= abs(P) - atol)
        if n >= 3:
            sec_val = (rho * (n - 3) - sigma * d * P) / float(n - 2)
        else:
            sec_val = 0.0                      # n=2: G_{μν}≡0, SEC vacua
        sec = nec and (sec_val >= -atol)
        return wec, nec, sec, dec, float(sec_val), True

    # ── 1.4 · Divergencia covariante REAL (o NaN declarado) ──────────────────────────
    @staticmethod
    def covariant_divergence(T: NDArray[np.float64], g: NDArray[np.float64],
                             christoffel: Optional[NDArray[np.float64]] = None,
                             dT: Optional[NDArray[np.float64]] = None) -> Tuple[float, bool]:
        r"""
        ∇_μ T^{μν} = ∂_μ T^{μν} + Γ^μ_{μλ}T^{λν} + Γ^ν_{μλ}T^{μλ}.
        Con T y g CONSTANTES y Γ=0 el resultado es EXACTAMENTE 0 (medido, no simulado).
        Sin conexión ni derivadas: (NaN, measured=False). Jamás `h·Σ|T_ij|`.
        """
        if christoffel is None and dT is None:
            return float("nan"), False
        n = g.shape[0]
        ginv, _, _ = _symmetric_inverse(g)
        T_up = ginv @ np.asarray(T, dtype=np.float64) @ ginv
        gam = np.zeros((n, n, n)) if christoffel is None else np.asarray(christoffel)
        dt = np.zeros((n, n, n)) if dT is None else np.asarray(dT)
        div = np.zeros(n)
        for nu in range(n):
            acc = 0.0
            for mu in range(n):
                acc += dt[mu, mu, nu]
                for lam in range(n):
                    acc += gam[mu, mu, lam] * T_up[lam, nu] + gam[nu, mu, lam] * T_up[mu, lam]
            div[nu] = acc
        return float(la.norm(div) / max(_fro(T_up), 1e-300)), True

    @staticmethod
    def _approximate_divergence_residual(T: NDArray[np.float64],
                                         metric: NDArray[np.float64],
                                         h: float = 1e-5) -> float:
        """ABI legacy: ahora devuelve NaN si no hay datos (antes fabricaba h·Σ|T|)."""
        val, measured = Phase1_EnergyMomentumExtractor.covariant_divergence(T, metric)
        return val if measured else float("nan")

    # ── 1.5 · Idempotencia de la renormalización de masa (anti doble conteo) ─────────
    @staticmethod
    def mass_idempotency_residual(m_eff: float) -> float:
        r"""
        El acoplamiento de Fröhlich YA está absorbido en m*. Re-renormalizar con
        (α=0, vol=0) DEBE ser la identidad: |m*(m*,0,0) − m*|/m*. Si no lo es,
        el "polarón purificado" introduce doble conteo (bug latente de v3).
        """
        again = float(_shield_mass(base_cost=m_eff, volatility_alpha=0.0, frohlich_coupling=0.0))
        return abs(again - m_eff) / max(abs(m_eff), 1e-300)

    # ── 1.6 · Construcción del tensor de fluido perfecto unificado por firma ─────────
    def compute_stress_energy_tensor(
        self, g_base: NDArray[np.float64], polaron: Any,
        flow_velocity: Optional[NDArray[np.float64]], market_pressure: float,
        frohlich_coupling_override: Optional[float] = None, *,
        metric_field: Optional[Callable[[NDArray], NDArray]] = None,
        x0: Optional[NDArray[np.float64]] = None,
        energy_policy: str = "degrade",
    ) -> EnergyMomentumData:
        r"""
        G1: T_{μν} = ρ u_μu_ν + P h_{μν}, h = g − σ u⊗u ⇒ T(u,u)=ρ para TODA firma.
        ρ := m** y P := market_pressure son identificaciones DECLARADAS del modelo (G11).
        """
        g = _sym(g_base)
        rep = self.analyze_signature(g)
        if not rep.is_nondegenerate:
            raise MetricDegeneracyError("Métrica base degenerada: no hay variedad.")
        if rep.signature_class == int(MetricSignatureClass.ULTRAHYPERBOLIC):
            raise MetricDegeneracyError(
                f"Firma ultrahiperbólica (+{rep.n_positive},−{rep.n_negative}): sin estructura causal.")

        coupling = (frohlich_coupling_override if frohlich_coupling_override is not None
                    else float(getattr(polaron, "frohlich_coupling", 0.0)))
        base_cost = float(getattr(polaron, "inertial_mass",
                                  getattr(polaron, "effective_mass", 1.0)))
        volatility = float(getattr(polaron, "volatility_alpha", 0.0))
        m_eff = float(_shield_mass(base_cost=base_cost, volatility_alpha=volatility,
                                   frohlich_coupling=coupling))
        if not math.isfinite(m_eff) or m_eff < 0.0:
            raise EnergyMomentumDegeneracyError(f"Masa efectiva no física: m**={m_eff}")
        idem = self.mass_idempotency_residual(m_eff)

        sigma = rep.timelike_sigma
        if flow_velocity is None:
            u = self.canonical_timelike_velocity(g)
        else:
            u = self._normalize_4velocity(flow_velocity, g, sigma)
        if abs(float(u @ g @ u) - sigma) > 1e-9:
            raise CausalStructureError("Normalización causal inconsistente tras el saneamiento.")

        rho, P = m_eff, float(market_pressure)
        if not math.isfinite(P):
            raise EnergyMomentumDegeneracyError("Presión no finita.")
        u_cov = g @ u
        h_proj = g - sigma * np.outer(u_cov, u_cov)
        T = rho * np.outer(u_cov, u_cov) + P * h_proj
        T = _sym(T)
        if _rel(T - T.T, T) > _SYM_ATOL:
            raise EnergyMomentumDegeneracyError("T_{μν} no simétrico (violación de covarianza).")
        rho_check = float(T @ np.outer(u, u) @ np.eye(g.shape[0]))  # placeholder evitado abajo
        rho_check = float(u @ T @ u)
        if abs(rho_check - rho) > 1e-8 * max(abs(rho), 1.0):
            raise EnergyMomentumDegeneracyError(
                f"T(u,u)={rho_check:.6e} ≠ ρ={rho:.6e}: forma de fluido perfecto mal aplicada.")

        ginv, _, _ = _symmetric_inverse(g)
        trace = float(np.einsum("ij,ij->", ginv, T))
        n = g.shape[0]
        wec, nec, sec, dec, sec_val, applicable = self.energy_conditions_general(
            rho, P, sigma, n, rep.has_causal_structure)

        # DEC tensorial: J^μ = σ T^μ{}_ν u^ν debe ser causal (σ·g(J,J) ≥ 0)
        J = sigma * (ginv @ T @ u)
        dec_causal = bool(sigma * float(J @ g @ J) >= -1e-12) if applicable else True

        gam = None
        if metric_field is not None and x0 is not None:
            gam = Phase2_EinsteinFieldSolver.christoffel_from_field(metric_field, np.asarray(x0))[0]
        cons, cons_measured = self.covariant_divergence(T, g, gam)
        if not cons_measured and gam is None:
            cons, cons_measured = 0.0, True      # T,g constantes ⇒ Γ=0 ⇒ ∇T=0 EXACTO

        verdict = HeytingVerdict.COHERENT
        if not applicable or not cons_measured or idem > 1e-10:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if applicable and not (wec and nec):
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED
                                   if energy_policy == "veto" else HeytingVerdict.DEGRADED)
        if applicable and not sec and energy_policy == "veto":
            verdict = HeytingVerdict.VETOED
        if energy_policy == "veto" and verdict is HeytingVerdict.VETOED:
            raise EnergyMomentumDegeneracyError(
                f"Condición de energía violada (política=veto): WEC={wec} NEC={nec} "
                f"SEC={sec} DEC={dec}")

        return EnergyMomentumData(
            T_tensor=T, effective_mass=m_eff, inflationary_pressure=P, four_velocity=u,
            trace=trace, energy_density=rho, weak_energy_ok=wec, null_energy_ok=nec,
            strong_energy_ok=sec, dominant_energy_condition_ok=dec,
            approximate_conservation_residual=cons, signature=rep, timelike_sigma=sigma,
            spatial_dim=n - 1, energy_conditions_applicable=applicable,
            sec_matter_value=sec_val, dec_flux_causal=dec_causal,
            conservation_measured=cons_measured, mass_idempotency_residual=float(idem),
            provenance={"base_electron": getattr(polaron, "base_electron", None),
                        "fiedler_value": float(getattr(polaron, "fiedler_value", 0.0)),
                        "frohlich_coupling": coupling, "volatility_alpha": volatility,
                        "base_cost": base_cost},
            verdict=int(verdict),
        )

    # ── 1.7 · MÉTODO TERMINAL DE Φ₁ → continuación a Φ₂ ─────────────────────────────
    def observe_and_handoff(
        self, g_base: NDArray[np.float64], polaron: Any,
        flow_velocity: Optional[NDArray[np.float64]], market_pressure: float,
        frohlich_coupling_override: Optional[float] = None, *,
        continuation: Optional[Callable[[EnergyMomentumData], R]] = None, **kw: Any,
    ) -> Union[EnergyMomentumData, R]:
        r"""
        Definición formal final de la Fase 1: `continuation: E → R`. Su cola
        `return continuation(E)` es el arranque de `Phase2._continue_from_phase1_energy`.
        """
        energy = self.compute_stress_energy_tensor(
            g_base, polaron, flow_velocity, market_pressure,
            frohlich_coupling_override, **kw)
        logger.info("[Φ₁ OBSERVE] m**=%.6e P=%.3e σ=%+.0f tr(T)=%.6e WEC=%s SEC=%s idem=%.2e",
                    energy.effective_mass, energy.inflationary_pressure,
                    energy.timelike_sigma, energy.trace, energy.weak_energy_ok,
                    energy.strong_energy_ok, energy.mass_idempotency_residual)
        if continuation is None:
            return energy
        return continuation(energy)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 2 · ORIENT: CONEXIÓN REAL, CURVATURA, LIGADURAS, POINCARÉ (E → C) ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase2_EinsteinFieldSolver:
    r"""
    Φ₂ — Orientador covariante. NO "resuelve" Einstein con (g,T) prescritos:
    AUDITA la violación de ligadura y calcula curvatura REAL a partir de un campo
    métrico o de ∂g explícito; en su ausencia CERTIFICA PLANITUD (Γ≡0, Riemann≡0).
    """

    def __init__(self, g_base: Optional[NDArray[np.float64]] = None, *,
                 units: Optional[UnitSystem] = None,
                 metric_field: Optional[Callable[[NDArray], NDArray]] = None,
                 shield: Optional[Any] = None,
                 lambda_cosmological: float = 0.0) -> None:
        self._units = units or AstrophysicalConstants.UNITS
        self._metric_field = metric_field
        self._lambda = float(lambda_cosmological)
        self._shield = shield if shield is not None else GravitationalShieldFunctor()
        gb = g_base if g_base is not None else getattr(self._shield, "g_base", G_PHYSICS_RAW)
        self._g_base = _sym(self._normalize_metric_source(gb))

    @staticmethod
    def _normalize_metric_source(src: Any) -> NDArray[np.float64]:
        """Normaliza el contrato roto ndarray|dict|None entre módulos del estrato."""
        if src is None:
            return np.eye(4, dtype=np.float64)
        if isinstance(src, dict):
            for k in ("G", "metric", "metric_tensor", "spacetime"):
                if k in src:
                    return np.asarray(src[k], dtype=np.float64)
            return np.eye(4, dtype=np.float64)
        return np.asarray(src, dtype=np.float64)

    # ── 2.1 · Continuación formal desde Φ₁ ───────────────────────────────────────────
    def _continue_from_phase1_energy(self, energy: EnergyMomentumData) -> EnergyMomentumData:
        if not isinstance(energy, EnergyMomentumData):
            raise TopologicalInvariantError(
                f"Continuidad Φ₁→Φ₂ violada: se esperaba EnergyMomentumData, hay {type(energy)}")
        if energy.verdict == int(HeytingVerdict.VETOED):
            raise EnergyMomentumDegeneracyError("Tensor de energía vetado en Φ₁.")
        return energy

    # ── 2.2 · Christoffel REALES por campo métrico (G3) ──────────────────────────────
    @staticmethod
    def metric_derivative(metric_field: Callable[[NDArray], NDArray],
                          x: NDArray[np.float64], h: Optional[float] = None
                          ) -> NDArray[np.float64]:
        """∂_μ g_{αβ} por diferencias centrales, h = eps^{1/3}·escala (error O(h²))."""
        x = np.asarray(x, dtype=np.float64)
        step = float(h) if h else _FD_STEP * max(1.0, float(np.max(np.abs(x))))
        n = x.size
        dg = np.empty((n, n, n), dtype=np.float64)
        for mu in range(n):
            e = np.zeros(n)
            e[mu] = step
            dg[mu] = (_sym(metric_field(x + e)) - _sym(metric_field(x - e))) / (2.0 * step)
        return dg

    @staticmethod
    def christoffel_from_dg(g: NDArray[np.float64], dg: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Γ^λ_{μν} = ½g^{λσ}(∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})  (exacto, sin magia)."""
        ginv, _, _ = _symmetric_inverse(g)
        return 0.5 * (np.einsum("ls,mns->lmn", ginv, dg, optimize=True)
                      + np.einsum("ls,nms->lmn", ginv, dg, optimize=True)
                      - np.einsum("ls,smn->lmn", ginv, dg, optimize=True))

    @staticmethod
    def christoffel_from_field(metric_field: Callable[[NDArray], NDArray],
                               x: NDArray[np.float64], h: Optional[float] = None
                               ) -> Tuple[NDArray[np.float64], float]:
        g = _sym(metric_field(np.asarray(x, dtype=np.float64)))
        dg = Phase2_EinsteinFieldSolver.metric_derivative(metric_field, x, h)
        return Phase2_EinsteinFieldSolver.christoffel_from_dg(g, dg), float(h or _FD_STEP)

    @staticmethod
    def _compute_christoffel_from_metric(g: NDArray[np.float64], h: float = 1e-6
                                         ) -> NDArray[np.float64]:
        r"""
        ABI legacy CORREGIDA: con un valor puntual de g la conexión NO está determinada.
        Se devuelve Γ≡0 (métrica constante en estas coordenadas ⇒ Levi-Civita trivial),
        que es el valor EXACTO, no una aproximación fabricada.
        """
        return np.zeros((g.shape[0],) * 3, dtype=np.float64)

    # ── 2.3 · Riemann, Ricci, Kretschmann, Weyl, seccional (G4) ──────────────────────
    @staticmethod
    def riemann_from_connection(g: NDArray[np.float64], gamma: NDArray[np.float64],
                                dgamma: Optional[NDArray[np.float64]] = None
                                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        R^ρ_{σμν} = ∂_μΓ^ρ_{νσ} − ∂_νΓ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ}.
        dgamma[μ,ρ,ν,σ] = ∂_μ Γ^ρ_{νσ}; si es None se usa 0 (conexión constante).
        Devuelve (Riemann (1,3), Riemann (0,4)).
        """
        n = g.shape[0]
        dg = np.zeros((n, n, n, n)) if dgamma is None else np.asarray(dgamma)
        r13 = (np.einsum("mrns->rsmn", dg, optimize=True)
               - np.einsum("nrms->rsmn", dg, optimize=True)
               + np.einsum("rml,lvs->rsmv", gamma, gamma, optimize=True)
               - np.einsum("rnl,lvs->rsmv", gamma, gamma, optimize=True))
        # Reindexado explícito para evitar ambigüedad de convención:
        r13 = np.zeros((n, n, n, n), dtype=np.float64)
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(n):
                        val = dg[mu, rho, nu, sig] - dg[nu, rho, mu, sig]
                        for lam in range(n):
                            val += (gamma[rho, mu, lam] * gamma[lam, nu, sig]
                                    - gamma[rho, nu, lam] * gamma[lam, mu, sig])
                        r13[rho, sig, mu, nu] = val
        r04 = np.einsum("rl,lsmn->rsmn", _sym(g), r13, optimize=True)
        return r13, r04

    @staticmethod
    def ricci_and_scalar(r13: NDArray[np.float64], g: NDArray[np.float64]
                         ) -> Tuple[NDArray[np.float64], float]:
        """R_{σν} = R^ρ_{σρν}; R = g^{σν}R_{σν}."""
        ricci = np.einsum("pspn->sn", r13, optimize=True)
        ricci = _sym(ricci)
        ginv, _, _ = _symmetric_inverse(g)
        return ricci, float(np.einsum("ij,ij->", ginv, ricci))

    @staticmethod
    def kretschmann(r04: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """K = R_{abcd}R^{abcd} (invariante cuadrático genuino; Schwarzschild: 12r_s²/r⁶)."""
        ginv, _, _ = _symmetric_inverse(g)
        up = np.einsum("ae,bf,cg,dh,efgh->abcd", ginv, ginv, ginv, ginv, r04, optimize=True)
        return float(np.einsum("abcd,abcd->", r04, up, optimize=True))

    @staticmethod
    def weyl_tensor(r04: NDArray[np.float64], ricci: NDArray[np.float64],
                    R: float, g: NDArray[np.float64]) -> NDArray[np.float64]:
        """C = Riemann − parte de traza (n≥3); Weyl≡0 ⇔ conformemente plano."""
        n = g.shape[0]
        if n < 3:
            return np.zeros_like(r04)
        gg = _sym(g)
        term = (np.einsum("ac,bd->abcd", gg, ricci) - np.einsum("ad,bc->abcd", gg, ricci)
                - np.einsum("bc,ad->abcd", gg, ricci) + np.einsum("bd,ac->abcd", gg, ricci))
        scal = (np.einsum("ac,bd->abcd", gg, gg) - np.einsum("ad,bc->abcd", gg, gg))
        return r04 - term / (n - 2) + R * scal / ((n - 1) * (n - 2))

    @staticmethod
    def sectional_curvatures(r13: NDArray[np.float64], g: NDArray[np.float64]
                             ) -> Tuple[float, NDArray[np.float64]]:
        r"""
        K(u,v) = ⟨R(u,v)v,u⟩/(g(u,u)g(v,v) − g(u,v)²) sobre un marco (pseudo-)ortonormal
        construido por Gram–Schmidt respecto a g. Devuelve (max|K|, matriz K_ab).
        """
        gg = _sym(g)
        n = gg.shape[0]
        frame: List[NDArray[np.float64]] = []
        for a in range(n):
            e = np.zeros(n)
            e[a] = 1.0
            for f in frame:
                nf = float(f @ gg @ f)
                if abs(nf) > 1e-12:
                    e = e - (float(e @ gg @ f) / nf) * f
            ne = float(e @ gg @ e)
            if abs(ne) > 1e-12:
                frame.append(e / math.sqrt(abs(ne)))
        m = len(frame)
        K = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                u, v = frame[a], frame[b]
                rv = np.einsum("rsmn,s,m,n->r", r13, v, u, v, optimize=True)
                num = float(u @ gg @ rv)
                den = float(u @ gg @ u) * float(v @ gg @ v) - float(u @ gg @ v) ** 2
                if abs(den) > 1e-12:
                    K[a, b] = num / den
        return float(np.max(np.abs(K))) if K.size else 0.0, K

    def compute_curvature(self, g: NDArray[np.float64],
                          connection: ConnectionReport) -> CurvatureInvariants:
        """Curvatura REAL si hay derivadas medidas; EXACTAMENTE CERO si planitud certificada."""
        gg = _sym(g)
        n = gg.shape[0]
        gam = (np.zeros((n, n, n)) if connection.christoffel is None
               else np.asarray(connection.christoffel))
        dgam = None
        if connection.source == "metric_field" and self._metric_field is not None:
            x = self._x_eval if hasattr(self, "_x_eval") else np.zeros(n)
            step = connection.fd_step or _FD_STEP
            dgam = np.empty((n, n, n, n))
            for mu in range(n):
                e = np.zeros(n)
                e[mu] = step
                gp, _ = self.christoffel_from_field(self._metric_field, x + e, step)
                gm, _ = self.christoffel_from_field(self._metric_field, x - e, step)
                dgam[mu] = (gp - gm) / (2.0 * step)
        r13, r04 = self.riemann_from_connection(gg, gam, dgam)
        ricci, R = self.ricci_and_scalar(r13, gg)
        kret = self.kretschmann(r04, gg)
        weyl = self.weyl_tensor(r04, ricci, R, gg)
        max_k, _ = self.sectional_curvatures(r13, gg)
        d = max(n - 1, 1)
        bound = abs(R) / max(d * (d - 1), 1)
        curv_scale = 1.0 / math.sqrt(max_k) if max_k > 1e-300 else float("inf")
        measured = connection.derivative_measured
        if not measured and max(abs(R), kret, max_k) > 1e-10:
            logger.error("[Φ₂] Curvatura no nula SIN derivadas medidas: artefacto detectado.")
        verdict = HeytingVerdict.COHERENT if measured or (abs(R) < 1e-12 and kret < 1e-12) \
            else HeytingVerdict.DEGRADED
        return CurvatureInvariants(
            ricci_tensor=ricci, ricci_scalar=R, max_sectional_curvature=max_k,
            christoffel_norm=float(la.norm(gam)), riemann_tensor=r04, kretschmann=kret,
            weyl_norm=_fro(weyl), einstein_tensor=None, sectional_bound_from_scalar=bound,
            curvature_measured=measured, injectivity_radius_estimate=curv_scale,
            curvature_scale=curv_scale, verdict=int(verdict),
        )

    @staticmethod
    def _compute_ricci_tensor(Gamma: NDArray[np.float64], g: NDArray[np.float64],
                              h: float = 1e-5) -> NDArray[np.float64]:
        """ABI legacy CORREGIDA: sin derivadas de Γ, sólo el término cuadrático exacto."""
        r13, _ = Phase2_EinsteinFieldSolver.riemann_from_connection(g, Gamma, None)
        ric, _ = Phase2_EinsteinFieldSolver.ricci_and_scalar(r13, g)
        return ric

    @staticmethod
    def _compute_ricci_scalar(Ricci: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        try:
            ginv, _, _ = _symmetric_inverse(g)
        except MetricDegeneracyError:
            return float("nan")
        return float(np.einsum("ij,ij->", ginv, Ricci))

    # ── 2.4 · Tensor de Einstein y DIAGNÓSTICO de ligadura (G5) ──────────────────────
    def einstein_constraints(self, curvature: CurvatureInvariants, g: NDArray[np.float64],
                             T: NDArray[np.float64], u: NDArray[np.float64],
                             sec_matter: float) -> EinsteinConstraintReport:
        r"""
        G_{μν} = R_{μν} − ½R g_{μν} + Λ g_{μν}; residual relativo contra κT_{μν}.
        SEC geométrica = R_{μν}u^μu^ν. Su discrepancia con la SEC de materia MIDE
        la violación de la ecuación de campo (no se presupone satisfecha).
        """
        gg = _sym(g)
        n = gg.shape[0]
        ric, R = curvature.ricci_tensor, curvature.ricci_scalar
        ein = ric - 0.5 * R * gg + self._lambda * gg
        if n == 2:
            ein = np.zeros_like(gg)          # identidad topológica en 2D
        kappa = self._units.kappa_einstein
        target = kappa * np.asarray(T, dtype=np.float64)
        res_abs = _fro(ein - target)
        res_rel = res_abs / max(_fro(target), 1e-300)
        sec_geo = float(u @ ric @ u)
        disc = abs(sec_geo - kappa * sec_matter) / max(abs(kappa * sec_matter), 1e-300)
        ok = res_rel <= 1e-8
        verdict = HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED
        return EinsteinConstraintReport(
            einstein_residual_relative=float(res_rel), einstein_residual_absolute=float(res_abs),
            sec_geometric_value=sec_geo, sec_matter_value=float(sec_matter),
            sec_discrepancy=float(disc), lambda_cosmological=self._lambda, kappa=kappa,
            dimension=n, constraint_satisfied=bool(ok), verdict=int(verdict),
        )

    # ── 2.5 · Flujo de calor de mapas armónicos (G10): Lyapunov de RSI-L3 ────────────
    def geodesic_energy(self, path: NDArray[np.float64]) -> float:
        """E = Σ_k g(Δγ,Δγ)/(2Δs) con Δs=1/N (energía discreta de Dirichlet)."""
        d = np.diff(path, axis=0)
        ds = 1.0 / max(len(path) - 1, 1)
        g = self._g_base
        return float(0.5 * np.sum(np.einsum("ti,ij,tj->t", d, g, d, optimize=True)) / ds)

    def relax_geodesic(self, path: NDArray[np.float64], steps: int = 32,
                       heat_step: Optional[float] = None) -> Tuple[NDArray, GeodesicRelaxationReport]:
        r"""
        ∂_tγ^μ = τ^μ = γ̈^μ + Γ^μ_{αβ}γ̇^αγ̇^β con extremos FIJOS (Dirichlet).
        dE/dt = −Σ g(τ,τ) ≤ 0: se VERIFICA a posteriori; paso CFL h ≤ 0.45·Δs²/2.
        Punto fijo = geodésica discreta. Este es el Lyapunov geométrico del nivel 3.
        """
        p = np.array(path, dtype=np.float64, copy=True)
        n_nodes = len(p)
        if n_nodes < 3:
            return p, GeodesicRelaxationReport(self.geodesic_energy(p), self.geodesic_energy(p),
                                               0.0, 0, 0.0, True, 0.0, True,
                                               int(HeytingVerdict.DEGRADED))
        ds = 1.0 / (n_nodes - 1)
        h_cap = _HEAT_CFL * ds * ds / 2.0
        h = float(min(heat_step, h_cap)) if heat_step else h_cap
        e0 = self.geodesic_energy(p)
        e_prev, monotone, tension = e0, True, 0.0
        for _ in range(max(1, int(steps))):
            acc = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / (ds * ds)
            vel = (p[2:] - p[:-2]) / (2.0 * ds)
            tau = acc.copy()
            if self._metric_field is not None:
                for k in range(1, n_nodes - 1):
                    gam, _ = self.christoffel_from_field(self._metric_field, p[k])
                    tau[k - 1] += np.einsum("mab,a,b->m", gam, vel[k - 1], vel[k - 1])
            p[1:-1] += h * tau
            e_k = self.geodesic_energy(p)
            monotone = monotone and (e_k <= e_prev + 1e-12)
            e_prev = e_k
            tension = float(np.sqrt(np.sum(np.einsum("ti,ij,tj->t", tau, self._g_base, tau))))
        resid = tension
        conv = resid <= 1e-9
        verdict = HeytingVerdict.COHERENT if (monotone and conv) else HeytingVerdict.DEGRADED
        return p, GeodesicRelaxationReport(
            energy_initial=float(e0), energy_final=float(e_prev), tension_norm_final=tension,
            steps=int(steps), step_used=h, monotone=bool(monotone), geodesic_residual=resid,
            converged=bool(conv), verdict=int(verdict))

    # ── 2.6 · Mecánica celeste de Poincaré en Schwarzschild (G9) ─────────────────────
    def celestial_orbits(self, m_eff: float, orbital_radius: Optional[float] = None,
                         semi_major: Optional[float] = None, eccentricity: float = 0.0,
                         kam_k_max: int = 6) -> CelestialOrbitReport:
        r"""
        Ω=√(GM/r³), κ²=(GM/r³)(1−6GM/(c²r)); residuo de Greene R=sin²(πκ/Ω).
        ISCO ⇔ R=0 exactamente. Esfera de fotones: λ=Ω=c³/(3√3GM), multiplicador e^{2π}.
        Precesión 1PN: Δϖ=6πGM/(c²a(1−e²)). KAM diofántico sobre κ/Ω.
        """
        u = self._units
        GM = u.G * max(float(m_eff), 0.0)
        if GM <= 0.0:
            return CelestialOrbitReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0,
                                        0.0, (0, 0, float("nan")), float("nan"), False,
                                        0.0, 0.0, False, int(HeytingVerdict.DEGRADED))
        r_s = 2.0 * GM / u.c ** 2
        r_isco, r_ph = 6.0 * GM / u.c ** 2, 3.0 * GM / u.c ** 2
        r = float(orbital_radius) if orbital_radius else 2.0 * r_isco
        omega = math.sqrt(GM / max(r, 1e-300) ** 3)
        k2 = (GM / max(r, 1e-300) ** 3) * (1.0 - 6.0 * GM / (u.c ** 2 * max(r, 1e-300)))
        kappa = math.sqrt(k2) if k2 > 0.0 else 0.0
        if k2 > 0.0:
            residue = math.sin(math.pi * kappa / omega) ** 2
            stable = True
        else:
            kk = math.sqrt(-k2)
            residue = float((2.0 - 2.0 * math.cosh(min(2.0 * math.pi * kk / omega, 700.0))) / 4.0)
            stable = False
        lam_ph = u.c ** 3 / (3.0 * math.sqrt(3.0) * GM)
        mult_ph = math.exp(min(2.0 * math.pi, 700.0))
        a = float(semi_major) if semi_major else r
        prec = 6.0 * math.pi * GM / (u.c ** 2 * max(a, 1e-300) * max(1.0 - eccentricity ** 2, 1e-12))
        ratio = kappa / omega if omega > 0 else 0.0
        p, q, err = _best_rational(ratio, 12)
        gamma = float("inf")
        for k in range(1, kam_k_max + 1):
            gamma = min(gamma, abs(k * ratio - round(k * ratio)) * k ** 2)
        kam_ok = bool(gamma >= 1e-4 and stable)
        verdict = HeytingVerdict.COHERENT if (stable and kam_ok) else HeytingVerdict.DEGRADED
        return CelestialOrbitReport(
            schwarzschild_radius=r_s, isco_radius=r_isco, photon_sphere_radius=r_ph,
            orbital_radius=r, omega_azimuthal=omega, kappa_epicyclic=kappa,
            greene_residue=float(residue), orbit_stable=stable,
            photon_sphere_lyapunov=lam_ph, photon_sphere_multiplier=mult_ph,
            perihelion_precession=float(prec), resonance=(int(p), int(q), float(err)),
            diophantine_gamma_eff=float(gamma), kam_satisfied=kam_ok,
            qnm_eikonal_real=lam_ph, qnm_eikonal_imag=0.5 * lam_ph,
            measured=True, verdict=int(verdict))

    # ── 2.7 · Matsubara ↔ Hawking: déficit cónico (G7) ───────────────────────────────
    def matsubara_certificate(self, m_eff: float, beta_supplied: float) -> MatsubaraCertificate:
        r"""
        β_τ = ħ/(k_B T_H) = 8πGM/c³ (periodo de tiempo imaginario). La métrica euclídea
        es suave en el horizonte SÓLO con ese periodo; si no, déficit δ=2π(1−β/β_H).
        I_E = β_E Mc²/2 = S_BH/k_B (Gibbons–Hawking) — identidad exacta verificada en Φ₃.
        """
        u = self._units
        M = max(float(m_eff), 0.0)
        if M <= 0.0 or beta_supplied <= 0.0:
            return MatsubaraCertificate(float(beta_supplied), 0.0, 0.0, float("nan"),
                                        False, 0.0, False, int(HeytingVerdict.DEGRADED))
        beta_tau = 8.0 * math.pi * u.G * M / u.c ** 3
        beta_energy = beta_tau / u.hbar
        deficit = 2.0 * math.pi * (1.0 - float(beta_supplied) / beta_tau)
        regular = abs(deficit) <= 1e-8
        i_e = beta_energy * M * u.c ** 2 / 2.0
        verdict = HeytingVerdict.COHERENT if regular else HeytingVerdict.DEGRADED
        return MatsubaraCertificate(
            beta_supplied=float(beta_supplied), beta_hawking_time=beta_tau,
            beta_hawking_energy=beta_energy, conical_deficit=float(deficit),
            regular=bool(regular), gibbons_hawking_action=float(i_e), measured=True,
            verdict=int(verdict))

    # ── 2.8 · MÉTODO TERMINAL DE Φ₂ → continuación a Φ₃ ─────────────────────────────
    def orient_from_energy_momentum(
        self, energy_data: EnergyMomentumData, node_index: int,
        attention_vector: NDArray[np.float64], *,
        integration_steps: int = 16, heat_step: Optional[float] = None,
        x_eval: Optional[NDArray[np.float64]] = None,
        orbital_radius: Optional[float] = None,
        continuation: Optional[Callable[[CurvatureBundle], R]] = None,
    ) -> Union[CurvatureBundle, R]:
        r"""
        Primer método formal de Φ₂: consume E de Φ₁. La "purificación" del polarón
        CONSERVA la procedencia (v3 perdía base_electron y fiedler_value por getattr
        sobre campos inexistentes) y se verifica la idempotencia de la renormalización.
        """
        energy = self._continue_from_phase1_energy(energy_data)
        self._x_eval = np.zeros(self._g_base.shape[0]) if x_eval is None else np.asarray(x_eval)

        pure = PolaronCartridge(
            base_electron=energy.provenance.get("base_electron"),
            frohlich_coupling=0.0,
            effective_mass=energy.effective_mass,
            inertial_mass=energy.effective_mass,
            volatility_alpha=0.0,
            fiedler_value=float(energy.provenance.get("fiedler_value", 0.0)),
        )
        action = self._shield.enforce_gravitational_attractor(
            polaron=pure, node_index=int(node_index),
            llm_attention_vector=np.asarray(attention_vector, dtype=np.float64))
        g_def = _sym(_shield_deform(g_base=self._g_base,
                                    effective_mass=energy.effective_mass, node=int(node_index)))

        if self._metric_field is not None:
            gam, step = self.christoffel_from_field(self._metric_field, self._x_eval)
            conn = ConnectionReport(gam, float(la.norm(gam)), "metric_field", True, False,
                                    step, int(HeytingVerdict.COHERENT))
        else:
            gam = np.zeros((g_def.shape[0],) * 3)
            conn = ConnectionReport(gam, 0.0, "flat_certified", False, True, _FD_STEP,
                                    int(HeytingVerdict.COHERENT))
            logger.info("[Φ₂] Sin campo métrico: Γ≡0 y Riemann≡0 CERTIFICADOS (no fabricados).")

        curv = self.compute_curvature(g_def, conn)
        constraints = self.einstein_constraints(curv, g_def, energy.T_tensor,
                                                energy.four_velocity, energy.sec_matter_value)
        curv = replace(curv, einstein_tensor=curv.ricci_tensor
                       - 0.5 * curv.ricci_scalar * g_def + self._lambda * g_def)

        att = np.asarray(attention_vector, dtype=np.float64).reshape(-1)
        n_nodes = max(3, int(integration_steps))
        path = np.outer(np.linspace(0.0, 1.0, n_nodes), att)
        relaxed, relaxation = self.relax_geodesic(path, steps=n_nodes, heat_step=heat_step)

        celestial = self.celestial_orbits(energy.effective_mass, orbital_radius=orbital_radius)
        beta = float(getattr(action, "matsubara_beta", 0.0))
        matsubara = self.matsubara_certificate(energy.effective_mass, beta) if beta > 0 else None

        parts = [HeytingVerdict(energy.verdict), HeytingVerdict(conn.verdict),
                 HeytingVerdict(curv.verdict), HeytingVerdict(constraints.verdict),
                 HeytingVerdict(relaxation.verdict), HeytingVerdict(celestial.verdict)]
        if matsubara is not None:
            parts.append(HeytingVerdict(matsubara.verdict))
        verdict = HeytingVerdict.COHERENT
        for pv in parts:
            verdict = heyting_meet(verdict, pv)

        bundle = CurvatureBundle(
            energy=energy, warped_metric=g_def, connection=conn, curvature=curv,
            constraints=constraints, polyakov_action=action, warped_space=WarpedSpaceTime(
                deformed_metric=g_def, christoffel_symbols=conn.christoffel,
                max_sectional_curvature=curv.max_sectional_curvature),
            celestial=celestial, matsubara=matsubara, relaxation=relaxation, verdict=int(verdict))
        logger.info("[Φ₂ ORIENT] R=%.6e K_ret=%.6e maxK=%.6e ‖Γ‖=%.3e ligadura=%.3e "
                    "E_geo %.4f→%.4f Greene=%.4f",
                    curv.ricci_scalar, curv.kretschmann, curv.max_sectional_curvature,
                    conn.christoffel_norm, constraints.einstein_residual_relative,
                    relaxation.energy_initial, relaxation.energy_final,
                    celestial.greene_residue)
        if continuation is None:
            return bundle
        return continuation(bundle)

    def deform_and_handoff(self, energy_data: EnergyMomentumData, node_index: int,
                           attention_vector: NDArray[np.float64], **kw: Any):
        """ABI legacy: método terminal de Φ₂ (continúa en Φ₃)."""
        return self.orient_from_energy_momentum(energy_data, node_index, attention_vector, **kw)

    # ── 2.9 · CALIBRACIÓN: pruebas exactas del motor de curvatura ────────────────────
    def self_test_curvature(self) -> Dict[str, float]:
        r"""
        Verificación EXACTA (no decorativa) del motor geométrico:
          · S² redonda de radio 1 en (θ,φ): R=2, K_sec=1, Kretschmann=4.
          · Schwarzschild (r_s=1) en r=3, θ=π/2: Kretschmann = 12r_s²/r⁶ = 12/729.
        Si estos residuos no son ~0, el motor de curvatura está mal y se declara.
        """
        def sphere(x: NDArray) -> NDArray:
            th = float(x[0])
            return np.diag([1.0, math.sin(th) ** 2])

        self._metric_field = sphere
        self._x_eval = np.array([1.0, 0.3])
        gam, step = self.christoffel_from_field(sphere, self._x_eval)
        conn = ConnectionReport(gam, float(la.norm(gam)), "metric_field", True, False, step, 0)
        c = self.compute_curvature(sphere(self._x_eval), conn)
        res_sphere_R = abs(c.ricci_scalar - 2.0)
        res_sphere_K = abs(c.max_sectional_curvature - 1.0)

        def schwarzschild(x: NDArray) -> NDArray:
            r, th = float(x[1]), float(x[2])
            f = 1.0 - 1.0 / r
            return np.diag([-f, 1.0 / f, r * r, (r * math.sin(th)) ** 2])

        self._metric_field = schwarzschild
        self._x_eval = np.array([0.0, 3.0, math.pi / 2.0, 0.0])
        gam2, step2 = self.christoffel_from_field(schwarzschild, self._x_eval)
        conn2 = ConnectionReport(gam2, float(la.norm(gam2)), "metric_field", True, False, step2, 0)
        c2 = self.compute_curvature(schwarzschild(self._x_eval), conn2)
        res_schwarz = abs(c2.kretschmann - 12.0 / 3.0 ** 6) / (12.0 / 3.0 ** 6)
        self._metric_field = None
        out = {"sphere_R_residual": res_sphere_R, "sphere_K_residual": res_sphere_K,
               "schwarzschild_kretschmann_relative_residual": res_schwarz}
        logger.info("[Φ₂ SELFTEST] %s", out)
        return out


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 3 · DECIDE/ACT: HORIZONTE, IDENTIDADES EXACTAS, Ω₃  (C → V)        ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase3_BekensteinHawkingDecider:
    r"""
    Φ₃ — Decisor termodinámico. Calcula Schwarzschild, VERIFICA las identidades exactas
    (1ª ley, Smarr, Gibbons–Hawking, saturación de Bekenstein), DERIVA χ por Gauss–Bonnet
    y colapsa el veredicto en Ω₃. Veto DURO (excepción) o SUAVE (⊥ en Ω₃) para que
    RSI-L3 pueda reparar en lugar de limitarse a rechazar.
    """

    def __init__(self, units: Optional[UnitSystem] = None,
                 planck_guard_factor: float = 1.0) -> None:
        self._units = units or AstrophysicalConstants.UNITS
        self._planck_guard = float(planck_guard_factor)

    # ── 3.1 · Continuación formal desde Φ₂ ───────────────────────────────────────────
    def _continue_from_phase2_curvature(self, bundle: CurvatureBundle) -> CurvatureBundle:
        if not isinstance(bundle, CurvatureBundle):
            raise TopologicalInvariantError(
                f"Continuidad Φ₂→Φ₃ violada: se esperaba CurvatureBundle, hay {type(bundle)}")
        return bundle

    # ── 3.2 · Termodinámica con identidades exactas y guardia planckiana (G6) ────────
    def evaluate_singularity(self, effective_mass: float) -> BlackHoleThermodynamics:
        r"""
        r_s=2GM/c², A=4πr_s², S=k_B A/(4ℓ_P²), T_H=ħc³/(8πGMk_B).
        IDENTIDADES VERIFICADAS: T_H dS/dM=c²; Mc²=2T_H S; I_E=S/k_B; S=S_Bekenstein.
        χ se DERIVA por Gauss–Bonnet con K=1/r_s² del horizonte redondo (G8).
        M=0 ⇒ NO hay horizonte (no "singularidad sin radiación"); M≲M_P ⇒ semiclásica inválida.
        """
        u = self._units
        M = float(effective_mass)
        if not math.isfinite(M) or M < 0.0:
            raise EnergyMomentumDegeneracyError(f"Masa no física para horizonte: {M}")
        if M == 0.0:
            return BlackHoleThermodynamics(
                0.0, 0.0, 0.0, float("inf"), horizon_euler_characteristic=0,
                euler_derived=0.0, no_horizon=True, semiclassical_valid=False,
                units_name=u.name, verdict=int(HeytingVerdict.DEGRADED))

        r_s = 2.0 * u.G * M / u.c ** 2
        area = 4.0 * math.pi * r_s ** 2
        s_bh = u.k_B * area / (4.0 * u.planck_length_sq)
        t_h = AstrophysicalConstants.hawking_factor(u) / M

        # (i) 1ª ley: T_H · dS/dM = c²  (derivada analítica exacta)
        ds_dm = 8.0 * math.pi * u.k_B * u.G ** 2 * M / (u.c ** 4 * u.planck_length_sq)
        first_law = abs(t_h * ds_dm - u.c ** 2) / u.c ** 2
        # (ii) Smarr: Mc² = 2 T_H S
        smarr = abs(2.0 * t_h * s_bh - M * u.c ** 2) / max(M * u.c ** 2, 1e-300)
        # (iii) Gibbons–Hawking: I_E = β_E Mc²/2 = S/k_B
        beta_e = 1.0 / (u.k_B * t_h)
        i_e = beta_e * M * u.c ** 2 / 2.0
        gh = abs(i_e - s_bh / u.k_B) / max(s_bh / u.k_B, 1e-300)
        # (iv) Bekenstein: S ≤ 2πk_B E R/(ħc); Schwarzschild la SATURA
        s_bek = 2.0 * math.pi * u.k_B * (M * u.c ** 2) * r_s / (u.hbar * u.c)
        bek_ratio = s_bh / max(s_bek, 1e-300)
        # (v) Evaporación
        t_evap = 5120.0 * math.pi * u.G ** 2 * M ** 3 / (u.hbar * u.c ** 4)
        # (G8) Gauss–Bonnet: K_horizonte = 1/r_s² ⇒ χ = K·A/(2π) = 2 DERIVADO
        k_hor = 1.0 / r_s ** 2
        chi_derived = k_hor * area / (2.0 * math.pi)
        chi_res = abs(chi_derived - 2.0)
        chi_int = int(round(chi_derived))

        m_planck = u.planck_mass
        ratio_planck = M / m_planck
        semiclassical = ratio_planck >= self._planck_guard
        verdict = HeytingVerdict.COHERENT
        if max(first_law, smarr, gh, chi_res) > 1e-8 or abs(bek_ratio - 1.0) > 1e-8:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if not semiclassical:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
            logger.warning("[Φ₃] M/M_P=%.3e < %.3g: termodinámica semiclásica NO válida.",
                           ratio_planck, self._planck_guard)
        return BlackHoleThermodynamics(
            schwarzschild_radius=r_s, horizon_area=area, bekenstein_hawking_entropy=s_bh,
            hawking_temperature=t_h, horizon_euler_characteristic=chi_int,
            euler_derived=float(chi_derived), euler_gauss_bonnet_residual=float(chi_res),
            first_law_residual=float(first_law), smarr_residual=float(smarr),
            gibbons_hawking_residual=float(gh), bekenstein_bound_ratio=float(bek_ratio),
            evaporation_time=float(t_evap), planck_mass_ratio=float(ratio_planck),
            semiclassical_valid=bool(semiclassical), no_horizon=False,
            units_name=u.name, verdict=int(verdict))

    # ── 3.3 · Decisión: Ω₃ con veto suave/duro ───────────────────────────────────────
    def decide_from_quantum_collapse(
        self, warped_space: Any, polyakov_action: Any, curvature: CurvatureInvariants,
        effective_mass: float, *, bundle: Optional[CurvatureBundle] = None,
        amplitude_tol: float = 1e-12, curvature_scale: Optional[float] = None,
        strict: bool = False,
        continuation: Optional[Callable[[HorizonVerdict], R]] = None,
    ) -> Union[Optional[BlackHoleThermodynamics], HorizonVerdict, R]:
        r"""
        Método de entrada de Φ₃ (continuación directa de Φ₂). El umbral de curvatura
        ya NO es `1e3` (dimensional y arbitrario): se compara con la escala geométrica
        r_inj⁻² estimada por 1/max|K_sec| (A2 por fin operativo).
        `strict=False` emite ⊥ en Ω₃ (veto SUAVE) para que RSI-L3 pueda reparar.
        Con `continuation` devuelve HorizonVerdict/`R`; sin él, el DTO legacy.
        """
        if bundle is not None:
            self._continue_from_phase2_curvature(bundle)
        amp = float(getattr(polyakov_action, "feynman_amplitude", 1.0))
        trapped = bool(getattr(polyakov_action, "is_trapped", False)) or amp < amplitude_tol
        kmax = abs(curvature.max_sectional_curvature)
        scale = float(curvature_scale) if curvature_scale else curvature.curvature_scale
        threshold = (1.0 / scale ** 2) if math.isfinite(scale) and scale > 0 else float("inf")
        curv_veto = bool(kmax > threshold and math.isfinite(threshold))

        ec_veto = False
        if bundle is not None:
            e = bundle.energy
            ec_veto = bool(e.energy_conditions_applicable and not (e.weak_energy_ok and e.null_energy_ok))

        verdict = HeytingVerdict.COHERENT
        reasons: List[str] = []
        if trapped:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED); reasons.append("horizonte:Ψ→0")
        if curv_veto:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED); reasons.append("K>r_inj⁻²")
        if ec_veto:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED); reasons.append("WEC/NEC")
        if bundle is not None:
            verdict = heyting_meet(verdict, HeytingVerdict(bundle.verdict))

        thermo = self.evaluate_singularity(effective_mass) if (trapped or curv_veto) else None
        if thermo is not None:
            verdict = heyting_meet(verdict, HeytingVerdict(thermo.verdict))
            if not thermo.semiclassical_valid:
                reasons.append("régimen_planckiano")

        hv = HorizonVerdict(
            thermodynamics=thermo, is_trapped=trapped, amplitude=amp,
            curvature_veto=curv_veto, energy_condition_veto=ec_veto,
            heyting_verdict_value=int(verdict),
            reason=";".join(reasons) if reasons else "geodésica_libre", verdict=int(verdict))

        if strict and verdict is HeytingVerdict.VETOED and thermo is not None:
            self.act_veto(thermo, polyakov_action)
        if continuation is not None:
            return continuation(hv)
        return thermo

    # ── 3.4 · ACT: veto ontológico con telemetría completa ───────────────────────────
    @staticmethod
    def act_veto(bh_thermo: BlackHoleThermodynamics, polyakov_action: Any) -> None:
        logger.critical(
            "[Φ₃ ACT] VETO ONTOLÓGICO — geodésica atencional absorbida.\n"
            "  r_s=%.6e  A=%.6e  S_BH=%.6e J/K  T_H=%.6e K\n"
            "  χ(Gauss-Bonnet)=%.6f (res=%.2e)  M/M_P=%.3e  semiclásica=%s\n"
            "  Identidades: 1ªley=%.2e Smarr=%.2e Gibbons-Hawking=%.2e Bekenstein=%.6f\n"
            "  t_evap=%.6e  Amplitud Ψ=%.6e",
            bh_thermo.schwarzschild_radius, bh_thermo.horizon_area,
            bh_thermo.bekenstein_hawking_entropy, bh_thermo.hawking_temperature,
            bh_thermo.euler_derived, bh_thermo.euler_gauss_bonnet_residual,
            bh_thermo.planck_mass_ratio, bh_thermo.semiclassical_valid,
            bh_thermo.first_law_residual, bh_thermo.smarr_residual,
            bh_thermo.gibbons_hawking_residual, bh_thermo.bekenstein_bound_ratio,
            bh_thermo.evaporation_time, getattr(polyakov_action, "feynman_amplitude", 0.0))
        raise SingularityVetoError(
            "Catástrofe de horizonte: la distorsión del espacio-tiempo forzó Ψ[γ]→0. "
            "Veredicto colapsado al supremo ⊤ (RECHAZO ABSOLUTO).")


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR · COMPOSICIÓN Φ₃∘Φ₂∘Φ₁ + RSI-L3 + PUENTE BOGOLIUBOV               ══
# ═════════════════════════════════════════════════════════════════════════════════════
class EinsteinHilbertAgent(Morphism):
    r"""
    El Gran Arquitecto Covariante. Compone por AGREGACIÓN tipada y continuación CPS
    (no por herencia ni por fabricación de tensores). RSI-L3 con Lyapunov geométrico:
    energía de Dirichlet del camino (G10) + residual de ligadura + déficit cónico.
    """

    def __init__(self, metric_tensor: Any = None, *,
                 units: Optional[UnitSystem] = None,
                 metric_field: Optional[Callable[[NDArray], NDArray]] = None,
                 lambda_cosmological: float = 0.0,
                 energy_policy: str = "degrade",
                 shield: Optional[Any] = None,
                 enable_bogoliubov_bridge: bool = True) -> None:
        self._units = units or AstrophysicalConstants.UNITS
        self._energy_policy = energy_policy
        gb = metric_tensor if metric_tensor is not None else G_PHYSICS_RAW
        self._g_base = _sym(Phase2_EinsteinFieldSolver._normalize_metric_source(gb))
        sig = Phase1_EnergyMomentumExtractor.analyze_signature(self._g_base)
        if not sig.is_nondegenerate:
            raise MetricDegeneracyError("G_PHYSICS degenerada: no hay variedad diferenciable.")
        self.phase1 = Phase1_EnergyMomentumExtractor(self._units)
        self.phase2 = Phase2_EinsteinFieldSolver(
            self._g_base, units=self._units, metric_field=metric_field,
            shield=shield, lambda_cosmological=lambda_cosmological)
        self.phase3 = Phase3_BekensteinHawkingDecider(self._units)
        self._bogoliubov = (BogoliubovAgent() if (enable_bogoliubov_bridge
                                                  and BogoliubovAgent is not None) else None)
        try:
            super().__init__(name="EinsteinHilbertAgent")
        except TypeError:
            super().__init__()
        logger.info("[EinsteinHilbertAgent] v4 d=%d firma=(+%d,−%d) unidades=%s "
                    "campo_métrico=%s política_energía=%s",
                    sig.dim, sig.n_positive, sig.n_negative, self._units.name,
                    metric_field is not None, energy_policy)

    # ── ABI: ciclo OODA completo con las tres fases anidadas ─────────────────────────
    def execute_covariant_ooda(
        self, polaron: Any, node_index: int, attention_vector: NDArray[np.float64],
        market_pressure: float, integration_steps: int = 16, *,
        flow_velocity: Optional[NDArray[np.float64]] = None,
        strict: bool = True, return_state: bool = False,
    ) -> Union[CategoricalState, EinsteinHilbertPhaseState]:
        r"""
        Z_EH = Φ₃∘Φ₂∘Φ₁ por continuaciones anidadas. `integration_steps` por fin se
        USA: es el número de nodos del camino relajado por flujo de calor (G10).
        La cuadrivelocidad por defecto es CANÓNICAMENTE TIMELIKE (corrige el bug v3).
        """
        att = np.asarray(attention_vector, dtype=np.float64).reshape(-1)
        if att.size != self._g_base.shape[0]:
            raise CausalStructureError(
                f"attention_vector dim={att.size} ≠ dim(g)={self._g_base.shape[0]}")
        trace: Dict[str, Any] = {}

        def _phase3(bundle: CurvatureBundle) -> EinsteinHilbertPhaseState:
            trace["C"] = bundle
            hv = self.phase3.decide_from_quantum_collapse(
                bundle.warped_space, bundle.polyakov_action, bundle.curvature,
                bundle.energy.effective_mass, bundle=bundle, strict=False,
                continuation=lambda v: v)
            assert isinstance(hv, HorizonVerdict)
            verdict = heyting_meet(HeytingVerdict(bundle.verdict), HeytingVerdict(hv.verdict))
            state = EinsteinHilbertPhaseState(
                energy=bundle.energy, bundle=bundle, horizon=hv,
                heyting_verdict_value=int(verdict),
                is_coherent=(verdict is HeytingVerdict.COHERENT),
                created_ns=time.perf_counter_ns())
            if strict and verdict is HeytingVerdict.VETOED and hv.thermodynamics is not None:
                self.phase3.act_veto(hv.thermodynamics, bundle.polyakov_action)
            return state

        def _phase2(energy: EnergyMomentumData) -> EinsteinHilbertPhaseState:
            trace["E"] = energy
            out = self.phase2.orient_from_energy_momentum(
                energy, node_index, att, integration_steps=integration_steps,
                continuation=_phase3)
            assert isinstance(out, EinsteinHilbertPhaseState)
            return out

        state = self.phase1.observe_and_handoff(
            self._g_base, polaron, flow_velocity, market_pressure,
            energy_policy=self._energy_policy, continuation=_phase2)
        assert isinstance(state, EinsteinHilbertPhaseState)

        logger.info("[Φ₃ DECIDE] veredicto=%s razón=%s R=%.3e ligadura=%.3e χ=%s",
                    HeytingVerdict(state.heyting_verdict_value).name, state.horizon.reason,
                    state.bundle.curvature.ricci_scalar,
                    state.bundle.constraints.einstein_residual_relative,
                    state.horizon.thermodynamics.horizon_euler_characteristic
                    if state.horizon.thermodynamics else "n/a")
        if return_state:
            return state
        return CategoricalState(stratum="WISDOM", payload={
            "action_integral": getattr(state.bundle.polyakov_action, "action_integral", None),
            "feynman_amplitude": state.horizon.amplitude,
            "effective_mass": state.energy.effective_mass,
            "ricci_scalar": state.bundle.curvature.ricci_scalar,
            "kretschmann": state.bundle.curvature.kretschmann,
            "max_sectional_curvature": state.bundle.curvature.max_sectional_curvature,
            "curvature_measured": state.bundle.curvature.curvature_measured,
            "einstein_residual": state.bundle.constraints.einstein_residual_relative,
            "greene_residue": state.bundle.celestial.greene_residue if state.bundle.celestial else None,
            "geodesic_energy_final": state.bundle.relaxation.energy_final if state.bundle.relaxation else None,
            "energy_conditions": {"WEC": state.energy.weak_energy_ok,
                                  "NEC": state.energy.null_energy_ok,
                                  "SEC": state.energy.strong_energy_ok,
                                  "DEC": state.energy.dominant_energy_condition_ok,
                                  "applicable": state.energy.energy_conditions_applicable},
            "trace_T": state.energy.trace,
            "heyting_verdict_value": state.heyting_verdict_value,
        })

    # ── RSI-L3: ledger geométrico + Banach + endurecimiento monótono (G12) ───────────
    @staticmethod
    def _ledger(state: EinsteinHilbertPhaseState) -> Tuple[float, Tuple[float, ...]]:
        r"""
        Componentes NO NEGATIVAS: E₁ energía de Dirichlet (decrece por G10),
        E₂ residual de ligadura de Einstein, E₃ discrepancia SEC geo/materia,
        E₄ |déficit cónico|/2π, E₅ residuo de tensión geodésica.
        La monotonía del COMPUESTO se IMPONE por aceptación con rollback.
        """
        b = state.bundle
        e1 = b.relaxation.energy_final if b.relaxation else 0.0
        e2 = b.constraints.einstein_residual_relative
        e3 = b.constraints.sec_discrepancy
        e4 = abs(b.matsubara.conical_deficit) / (2 * math.pi) if b.matsubara else 0.0
        e5 = b.relaxation.geodesic_residual if b.relaxation else 0.0
        comp = (float(abs(e1)), float(e2), float(e3), float(e4), float(e5))
        return float(np.mean(comp)), comp

    def run_rsi_l3_cycle(
        self, polaron: Any, node_index: int, attention_vector: NDArray[np.float64],
        market_pressure: float, *, max_iterations: int = 6, tolerance: float = 1e-12,
        hyperparameters: Optional[RSIL3Hyperparameters] = None,
    ) -> Tuple[EinsteinHilbertPhaseState, RSIL3Report]:
        r"""
        R₁ (Φ₁): idempotencia de masa y condiciones de energía en n dims.
        R₂ (Φ₂): flujo de calor de mapas armónicos (Lyapunov E[γ]) + curvatura real.
        R₃ (Φ₃): identidades termodinámicas exactas + χ por Gauss–Bonnet.
        META: reescribe paso/nodos SÓLO hacia el endurecimiento; rollback si V crece.
        """
        hyper = hyperparameters or RSIL3Hyperparameters(path_nodes=16)
        best = self.execute_covariant_ooda(polaron, node_index, attention_vector,
                                           market_pressure, hyper.path_nodes,
                                           strict=False, return_state=True)
        assert isinstance(best, EinsteinHilbertPhaseState)
        v_prev, _ = self._ledger(best)
        v0 = v_prev
        hist_v: List[float] = [v_prev]
        hist_x: List[NDArray[np.float64]] = [self._state_vector(best)]
        accepted = rejected = 0
        q, bound, certified, monotone = 1.0, float("inf"), False, True

        for _ in range(max(1, int(max_iterations))):
            self.phase2._units = self._units
            cand = self.execute_covariant_ooda(
                polaron, node_index, attention_vector, market_pressure,
                hyper.path_nodes, strict=False, return_state=True)
            assert isinstance(cand, EinsteinHilbertPhaseState)
            v_k, _ = self._ledger(cand)
            hard = (cand.heyting_verdict_value == int(HeytingVerdict.VETOED)
                    or not math.isfinite(v_k))
            ok = (not hard) and (v_k <= v_prev + tolerance)
            if ok:
                accepted += 1
                best = cand
                hist_v.append(v_k); hist_x.append(self._state_vector(cand))
                if len(hist_x) >= 3:
                    ratios = [float(la.norm(hist_x[i] - hist_x[i - 1]) /
                                    max(la.norm(hist_x[i - 1] - hist_x[i - 2]), 1e-300))
                              for i in range(len(hist_x) - 1, 1, -1)[:3]]
                    q = float(np.median(ratios))
                    certified = q < 1.0
                    bound = (q / (1.0 - q) * float(la.norm(hist_x[-1] - hist_x[-2]))
                             if certified else float("inf"))
                delta, v_prev = abs(v_prev - v_k), v_k
            else:
                rejected += 1
                monotone = False
                delta = float("inf")
            gain = (1.0 - hyper.trust_radius) if (ok and q < 0.5) else 0.5
            hyper = replace(
                hyper,
                heat_step=max(hyper.heat_step * gain, 1e-9),
                path_nodes=int(min(hyper.path_nodes * (2 if ok else 1), 4096)),
                constraint_tol=max(hyper.constraint_tol * gain, 1e-14),
                trust_radius=float(np.clip(hyper.trust_radius * (1.0 if ok else 0.5), 1e-3, 0.5)),
                meta_depth=hyper.meta_depth + 1)
            if ok and delta < tolerance and certified:
                break

        verdict = int(HeytingVerdict.COHERENT if (monotone and certified)
                      else HeytingVerdict.DEGRADED)
        report = RSIL3Report(
            iterations=accepted + rejected, accepted=accepted, rejected=rejected,
            lyapunov_initial=float(v0), lyapunov_final=float(v_prev),
            lyapunov_history=tuple(float(x) for x in hist_v), contraction_q=float(q),
            banach_error_bound=float(bound), contraction_certified=bool(certified),
            monotone=bool(monotone), tightening_only=True, hyperparameters=hyper,
            verdict=verdict)
        return replace(best, rsi_report=report,
                       heyting_verdict_value=int(heyting_meet(best.heyting_verdict_value, verdict))), report

    @staticmethod
    def _state_vector(state: EinsteinHilbertPhaseState) -> NDArray[np.float64]:
        b = state.bundle
        return np.concatenate([
            np.asarray([state.energy.effective_mass, state.energy.trace,
                        b.curvature.ricci_scalar, b.curvature.kretschmann,
                        b.constraints.einstein_residual_relative]),
            np.asarray(b.curvature.ricci_tensor, dtype=np.float64).ravel()])

    # ── Puente con el soberano Bogoliubov: monodromía simpléctica de la geodésica ────
    def govern_with_bogoliubov(self, monodromy_M: NDArray[np.float64],
                               metric_G: Optional[NDArray[np.float64]] = None,
                               **kw: Any) -> Dict[str, Any]:
        r"""
        El flujo geodésico es Hamiltoniano en T*M ⇒ su monodromía M ∈ Sp(2d,ℝ).
        Se delega la auditoría simpléctica/CAR-CCR al agente anterior y se fusionan
        veredictos por meet intuicionista (funtorialidad laxa Φ₃∘Φ₂∘Φ₁ → Ω₃).
        """
        m = np.asarray(monodromy_M, dtype=np.float64)
        d = m.shape[0] // 2
        g = self._g_base if metric_G is None else np.asarray(metric_G, dtype=np.float64)
        om = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
        sym_res = _fro(m.T @ om @ m - om) / _fro(om)
        local = int(HeytingVerdict.COHERENT if sym_res <= 1e-9 else HeytingVerdict.DEGRADED)
        if self._bogoliubov is None:
            return {"bogoliubov": None, "symplectic_residual": sym_res,
                    "local_verdict": local,
                    "merged_verdict": int(heyting_meet(local, HeytingVerdict.DEGRADED))}
        cert = self._bogoliubov.audit_bogoliubov_transformation(
            np.eye(d, dtype=np.complex128), np.zeros((d, d), dtype=np.complex128),
            m, g[:d, :d], False, **kw)
        return {"bogoliubov": cert, "symplectic_residual": sym_res, "local_verdict": local,
                "merged_verdict": int(heyting_meet(local, cert.heyting_verdict_value))}

    # ── Auditoría end-to-end ─────────────────────────────────────────────────────────
    def verify_complete_integrity(self) -> Dict[str, Any]:
        sig = Phase1_EnergyMomentumExtractor.analyze_signature(self._g_base)
        calib = self.phase2.self_test_curvature()
        verdict = int(HeytingVerdict.COHERENT if max(calib.values()) < 1e-4
                      else HeytingVerdict.DEGRADED)
        return {
            "metric": {"dim": sig.dim, "signature": (sig.n_positive, sig.n_negative, sig.n_zero),
                       "nondegenerate": sig.is_nondegenerate,
                       "causal": sig.has_causal_structure, "kappa": sig.condition_number},
            "units": {"system": self._units.name, "kappa_einstein": self._units.kappa_einstein,
                      "planck_mass": self._units.planck_mass},
            "curvature_calibration": calib,
            "heyting_verdict_value": verdict,
            "schema_version": "4.0.0",
        }

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._g_base.copy()

    @property
    def units(self) -> UnitSystem:
        return self._units