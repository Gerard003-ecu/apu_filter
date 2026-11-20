# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Gravitational Shield — Atractor Determinista con Certificado de Gauge       ║
║ Ruta   : app/core/immune_system/gravity_shield.py                                    ║
║ Versión: 4.0.0-GaugeCertificate-IsotropicCurvature-Matsubara-Spohn-RSI-L3-PhD        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA: TRES FASES ANIDADAS POR CONTINUACIÓN (composición de Kleisli)
──────────────────────────────────────────────────────────────────────────────────────
   C₁ : (m*, α_v, α_f) ──► M  (MassAcquisitionResult)
   C₂ : M ───────────────► W  (WarpedSpaceTime)
   C₃ : W ───────────────► P  (PolyakovAction)
   Z_shield = C₃ ∘ C₂ ∘ C₁

   El método terminal de cada fase recibe `continuation: Callable[[DTO], R]` y su cola
   es literalmente `return continuation(DTO)`; la continuación canónica es el primer
   método de la fase siguiente (`_continue_from_phase{k}_*`). Los veredictos se agregan
   por meet en Ω₃ (funtorialidad laxa hacia el retículo de Heyting).

AXIOMAS CON ESTATUS EPISTÉMICO EXPLÍCITO
──────────────────────────────────────────────────────────────────────────────────────
 S1 [TEOREMA · GAUGE] LA DEFORMACIÓN LEGADA ES UN DIFEOMORFISMO, NO UN CAMPO GRAVITATORIO.
    Sea δ_a(x) = κ·[a=k]·x^k y Λ^{1/2} = diag(e^{δ_a/2}). Entonces
        g̃_ab = e^{(δ_a+δ_b)/2} G_ab = (φ*G)_ab,
        φ: x ↦ y,  y^k = (2/κ)e^{κ x^k/2},  y^a = x^a (a≠k),
        Dφ = Λ^{1/2},  det Dφ = e^{κ x^k/2} > 0.
    Si G es CONSTANTE (plana), por naturalidad del tensor de Riemann bajo pullback:
        Riemann(g̃) ≡ 0  IDÉNTICAMENTE.
    CONSECUENCIA: el veto por curvatura del modo legado es VACUO (sólo ve redondeo).
    Se CERTIFICA numéricamente y se DECLARA; no se disimula. Para curvatura real se
    ofrece el modo conforme isótropo (S2), que NO es un pullback de una métrica plana.
 S2 [TEOREMA] Modo conforme isótropo (curvatura genuina):
        g̃(x) = Ω²(r) G,  Ω²(r) = (1 + Gm/(2c²r))⁴,  r = ‖x − x_nodo‖.
    Para G plana en n dims, con ω = ln Ω:
        R̃_{μν} = −(n−2)(∂_μ∂_νω − ∂_μω∂_νω) − G_{μν}(Δω + (n−2)|∇ω|²)
        R̃ = Ω⁻²[−2(n−1)Δω − (n−1)(n−2)|∇ω|²]
    No es armónica ⇒ curvatura NO nula. (En n=3 y G=δ es la slice t=cte de
    Schwarzschild en coordenadas isótropas: R̃=0 por la ligadura hamiltoniana, pero
    R̃_{μν}≠0 y las curvaturas seccionales ≠0.) Weyl ≡ 0 (conformemente plana).
 S3 [TEOREMA] Congruencia diagonal ⇒ Sylvester: Λ^{1/2} real positiva ⇒ g̃ SPD.
    PERO en flotante: κ(g̃) ≤ κ(G)·κ(Λ) = κ(G)·e^{|δ_k|}. Se ACOTA y se veta si
    excede el presupuesto de exactitud. "No re-verificar SPD" sólo vale en aritmética
    exacta; aquí se certifica κ y se vigila el overflow de exp.
 S4 [TEOREMA] Christoffel del modo legado, forma cerrada EXACTA (verificada):
        Γ^μ_{νρ} = (κ/4)[ [ν=k](g̃^{μk}g̃_{kρ} + [ρ=k]δ^μ_ρ)
                        + [ρ=k](g̃^{μk}g̃_{kν} + [ν=k]δ^μ_ν)
                        − g̃^{μk}g̃_{νρ}([ν=k]+[ρ=k]) ]
    Verificación: G=I, n=2, k=0 ⇒ único Γ no nulo = Γ^0_{00} = κ/2 ⇒ Riemann=0 ✓ (S1).
 S5 [TEOREMA] CSMD (Squire–Trapp): ∂_k f = Im f(x+ih e_k)/h + O(h²), SIN cancelación
    catastrófica, válida SÓLO si toda la cadena es holomorfa. Se AUDITA comparando
    contra diferencias centrales; `np.real`, `abs`, `conj`, `norm` rompen holomorfía.
    Detección de complejidad con `np.iscomplexobj` (no `isinstance(...,complex)`,
    que descarta silenciosamente np.complex64).
 S6 [TEOREMA] Círculo de Matsubara y acción de Polyakov. Con τ = β·s, s∈[0,1]:
        S_E[γ] = ½∫₀^β g̃(dγ/dτ)² dτ = (1/β)·E[γ],  E[γ] = ½∫₀¹ g̃(dγ/ds)² ds.
    Regularidad del cigarro euclídeo (Gibbons–Hawking) exige
        β_τ = ħ/(k_B T_H) = 8πGM/c³.
    Cualquier otro β produce DÉFICIT CÓNICO δ_conf = 2π(1 − β/β_H).
    MONOTONÍA DECLARADA: S_E = E/β DECRECE con M (β crece). Si la semántica deseada
    es "más masa ⇒ más veto", el funcional correcto es la acción de Gibbons–Hawking
    I_E = β_E Mc²/2 = S_BH/k_B (crece como M²), NO la acción de camino de prueba.
    Se reportan AMBAS y se declara la dirección de monotonía.
 S7 [TEOREMA] Cuadratura cerrada de orden 4 SIN fabricar muestras:
    N impar ⇒ Simpson compuesto; N par ≥4 ⇒ Simpson en los primeros N−3 nodos +
    Simpson 3/8 en los últimos 3 intervalos. Richardson: err ≈ |S_h − S_2h|/15.
    N=1 ⇒ NO hay intervalo: 0.0 y regla "insufficient_data" (jamás f₀·h inventado).
 S8 [TEOREMA] Veto en espacio logarítmico: Ψ = e^{−S/ħ} < tol ⇔ S > −ħ ln tol.
    La comparación se hace sobre S (exacta aunque Ψ desborde por debajo).
    Umbral de curvatura GEOMÉTRICO: K_max > r_inj⁻², con r_inj estimada por
    1/√K_max (escala de curvatura) o por una escala de negocio declarada; nunca 1e3.
 S9 [CONVENCIÓN] Piso inercial C^∞: m_floor(m) = √(m² + m_min²). Propiedades:
    m_floor(0)=m_min, m_floor(m)−m → 0, C^∞ en ℝ≥0. CONSECUENCIA DECLARADA: el
    modelo NO tiene límite sin masa (todo nodo gravita). La re-entrada idempotente
    (`already_renormalized=True`) es OBLIGATORIA para evitar la deriva
    m_n = √(m₀² + n·m_min²) por doble aplicación del piso.
S10 [CONVENCIÓN] Fröhlich saturado f(α) = 1 + tanh(α/2π): f'(0)=1/2π (pendiente
    física de bajo acoplamiento preservada), f ∈ [1,2) (regularización UV).
S11 [SAFETY] RSI-L3 con ENDURECIMIENTO MONÓTONO: sólo refina cuadratura o aprieta
    tolerancias; jamás afloja umbrales. Verificador INMUTABLE. Veto SUAVE (⊥ en Ω₃)
    por defecto en el modo agente, para que la Fase 3 del soberano Einstein reciba
    el `PolyakovAction` y emita la telemetría de horizonte.

UNIDADES: sistema COHERENTE explícito (SI-CODATA o natural). El acoplamiento del
modo legado es ADIMENSIONAL (κ = 2Gm/(c²·r_node)); `exp(2Gm/c⁴)` era dimensionalmente
inválido y numéricamente inerte (~1.65e-11 con las constantes "cyber").
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Dependencias del ecosistema (contratos tolerantes, degradación grácil) ───────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico violado."""

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name

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
        inertial_mass: float = 1.0
        effective_mass: float = 1.0
        volatility_alpha: float = 0.0
        frohlich_coupling: Optional[float] = None
        fiedler_value: float = 0.0

# ── Ω₃ del soberano de cuasipartículas (cadena de Gödel de 3 elementos) ──────────────
try:
    from app.agents.omega.bogoliubov_agent import HeytingVerdict, heyting_meet
except ImportError:  # pragma: no cover
    class HeytingVerdict(IntEnum):  # type: ignore[no-redef]
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    def heyting_meet(a: Any, b: Any) -> Any:  # type: ignore[no-redef]
        return HeytingVerdict(max(int(a), int(b)))

logger = logging.getLogger("MIC.ImmuneSystem.GravityShield")
R = TypeVar("R")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CSMD_H: Final[float] = 1.0e-30
_FD_H: Final[float] = _MACHINE_EPS ** (1.0 / 3.0)
_LOG_EXP_CAP: Final[float] = 700.0
_SYM_ATOL: Final[float] = 1.0e-12
_NONDEGEN_REL: Final[float] = 1.0e-12
_HEAT_CFL: Final[float] = 0.45

__all__ = [
    "UnitSystem", "GravitationalConstants", "GravitationalCollapseError",
    "EventHorizonViolation", "DimensionalConsistencyError", "PolaronLike",
    "MassAcquisitionResult", "BaseMetricCache", "GaugeCertificate",
    "CurvatureReport", "MatsubaraCircle", "WarpedSpaceTime", "PolyakovAction",
    "GravitationalShieldState", "RSIL3Hyperparameters", "RSIL3Report",
    "Phase1_MassAcquisition", "Phase2_ConformalWarping", "Phase3_HeytingPolyakovDecider",
    "GravitationalShieldFunctor",
    # Aliases ABI (corrigen los NameError de la v3.1)
    "build_base_metric_cache", "_build_base_metric_cache",
    "kinetic_density", "_kinetic_density",
    "_acquire_effective_mass", "_deform_metric_tensor", "_evaluate_feynman_kac",
    "_smooth_inertial_floor", "_saturating_frohlich_factor",
    "_construct_warped_metric", "_christoffel_from_metric",
    "complex_step_partial_k_christoffel", "_sectional_curvatures_around_node",
    "_simpson_composite",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §A · UNIDADES COHERENTES Y CONSTANTES (fin de la mezcla cyber/SI y del expo dimensional)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class UnitSystem:
    r"""Sistema COHERENTE. El acoplamiento adimensional κ = 2Gm/(c²r) exige r explícita."""

    G: float
    c: float
    hbar: float
    k_B: float
    name: str

    @classmethod
    def si(cls) -> "UnitSystem":
        return cls(6.67430e-11, 299_792_458.0, 1.054_571_817e-34, 1.380_649e-23, "SI-CODATA2018")

    @classmethod
    def natural(cls) -> "UnitSystem":
        return cls(1.0, 1.0, 1.0, 1.0, "NATURAL")

    @classmethod
    def cyber_legacy(cls) -> "UnitSystem":
        r"""
        Unidades "cyber" de la v3.1 (G×10⁹, c/10⁶). Se conservan para reproducibilidad,
        pero se DECLARA que 2G/c⁴ ≈ 1.65e-11 ⇒ acoplamiento numéricamente inerte.
        """
        return cls(6.67430e-2, 2.99792e2, 1.05457e-3, 1.0, "CYBER-LEGACY-v3.1")

    @property
    def planck_length_sq(self) -> float:
        return self.hbar * self.G / self.c ** 3

    def dimensionless_coupling(self, mass: float, radius: float) -> float:
        """κ = 2Gm/(c²r) — ADIMENSIONAL (r_s/r). Sustituye al inválido exp(2Gm/c⁴)."""
        if radius <= 0.0:
            raise DimensionalConsistencyError("r_node > 0 requerido para un κ adimensional.")
        return 2.0 * self.G * float(mass) / (self.c ** 2 * float(radius))

    def hawking_temperature(self, mass: float) -> float:
        if mass <= 0.0:
            return float("inf")
        return self.hbar * self.c ** 3 / (8.0 * math.pi * self.G * mass * self.k_B)

    def matsubara_beta_time(self, mass: float) -> float:
        """β_τ = ħ/(k_B T_H) = 8πGM/c³ (unidades de tiempo). Regularidad del cigarro."""
        return 8.0 * math.pi * self.G * float(mass) / self.c ** 3


class GravitationalConstants:
    """ABI legacy + sistema de unidades explícito y umbrales GEOMÉTRICOS."""

    UNITS: UnitSystem = UnitSystem.si()
    CYBER_G: Final[float] = 6.67430e-2
    CYBER_C: Final[float] = 2.99792e2
    G_C4_FACTOR: Final[float] = (2.0 * CYBER_G) / (CYBER_C ** 4)   # ≈1.65e-11 (inerte)
    HBAR_EFF: Final[float] = 1.05457e-3
    SCHWARZSCHILD_TOLERANCE: Final[float] = 1e-9
    MINIMAL_INERTIAL_MASS: Final[float] = 1e-12
    CRITICAL_SECTIONAL_CURVATURE: Final[float] = 1.0e3             # legacy (dimensional)
    COMPLEX_STEP_H: Final[float] = _CSMD_H
    SYMMETRY_ATOL: Final[float] = _SYM_ATOL
    DEFAULT_NODE_RADIUS: Final[float] = 1.0
    MAX_CONDITION_NUMBER: Final[float] = 1.0e10


class GravitationalCollapseError(TopologicalInvariantError):
    """La deformación destruyó la SPD, desbordó κ, o una forma cuadrática perdió signo."""


class EventHorizonViolation(TopologicalInvariantError):
    """Veto DURO: acción o curvatura superan el umbral (combinador booleano OR)."""


class DimensionalConsistencyError(TopologicalInvariantError):
    """Exponente dimensional, escala de longitud ausente o unidades incoherentes."""


@runtime_checkable
class PolaronLike(Protocol):
    """Contrato mínimo. `frohlich_coupling` es OPCIONAL (la v3 lo exigía y luego lo trataba Optional)."""

    inertial_mass: float
    volatility_alpha: float


# ═════════════════════════════════════════════════════════════════════════════════════
# §B · DTOs INMUTABLES (objetos del topos; contratos formales entre fases)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MassAcquisitionResult:
    r"""
    M — salida terminal de C₁ y dominio exacto del primer método de C₂.
    `__float__` restaura la ABI con el agente Einstein (`float(_acquire_effective_mass(...))`).
    """

    raw_cost: float
    volatility_alpha: float
    frohlich_coupling: float
    frohlich_factor: float
    effective_mass: float
    floor_applied: bool = True
    already_renormalized: bool = False
    idempotency_residual: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.effective_mass)

    @classmethod
    def from_effective(cls, m_eff: float) -> "MassAcquisitionResult":
        """Re-entrada IDEMPOTENTE: no vuelve a aplicar el piso ni Fröhlich (S9)."""
        return cls(raw_cost=float(m_eff), volatility_alpha=0.0, frohlich_coupling=0.0,
                   frohlich_factor=1.0, effective_mass=float(m_eff), floor_applied=False,
                   already_renormalized=True, idempotency_residual=0.0,
                   provenance={"reentry": True})


@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    """Caché espectral SPD construida UNA vez (Cholesky + κ + presupuesto de exactitud)."""

    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    dimension: int
    condition_number: float = 1.0
    lambda_min: float = 1.0
    lambda_max: float = 1.0
    symmetry_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class GaugeCertificate:
    r"""
    S1: prueba de que la deformación legada es un pullback (⇒ Riemann≡0 si G plana).
    Se certifica numéricamente y se DECLARA la vacuidad del veto por curvatura.
    """

    is_pure_gauge: bool
    base_is_constant: bool
    diffeomorphism_det: float
    riemann_identically_zero: bool
    measured_max_abs_riemann: float
    curvature_veto_vacuous: bool
    statement: str


@dataclass(frozen=True, slots=True)
class CurvatureReport:
    """Curvatura REAL (o exactamente nula con certificado de gauge)."""

    christoffel: NDArray[np.float64]
    christoffel_norm: float
    sectional_curvatures: NDArray[np.float64]
    max_sectional_curvature: float
    sectional_is_lower_bound: bool
    riemann_reduced: NDArray[np.float64]
    riemann_full: Optional[NDArray[np.float64]]
    ricci_tensor: Optional[NDArray[np.float64]]
    ricci_scalar: float
    kretschmann: float
    curvature_measured: bool
    csmd_valid: bool
    csmd_vs_central_residual: float
    injectivity_scale: float
    verdict: int


@dataclass(frozen=True, slots=True)
class MatsubaraCircle:
    """S6: β derivado de Hawking, déficit cónico y dirección de monotonía declarada."""

    beta_supplied: float
    beta_hawking_time: float
    beta_hawking_energy: float
    conical_deficit: float
    regular: bool
    hawking_temperature: float
    gibbons_hawking_action: float
    bekenstein_hawking_entropy: float
    path_action_monotonicity: str
    mode: str
    verdict: int


@dataclass(frozen=True, slots=True)
class WarpedSpaceTime:
    """W — salida terminal de C₂ y dominio exacto del primer método de C₃."""

    original_metric: NDArray[np.float64]
    deformed_metric: NDArray[np.float64]
    christoffel_symbols: NDArray[np.float64]
    riemann_curvature: NDArray[np.float64]
    sectional_curvatures: NDArray[np.float64]
    max_sectional_curvature: float
    node_index: int
    # ── extensiones v4 ──
    deformed_inverse: Optional[NDArray[np.float64]] = None
    mode: str = "gauge_congruence"
    dimensionless_coupling: float = 0.0
    condition_number_bound: float = 1.0
    condition_number_measured: float = 1.0
    gauge: Optional[GaugeCertificate] = None
    curvature: Optional[CurvatureReport] = None
    metric_field: Optional[Callable[[NDArray], NDArray]] = None
    x_eval: Optional[NDArray[np.float64]] = None
    mass: Optional[MassAcquisitionResult] = None
    verdict: int = 0


@dataclass(frozen=True, slots=True)
class PolyakovAction:
    """P — salida terminal de C₃ (ABI v3.1 preservada + campos rigurosos)."""

    action_integral: float
    feynman_amplitude: float
    is_trapped: bool
    action_veto: bool
    curvature_veto: bool
    discretization_error: float
    # ── extensiones v4 ──
    dirichlet_energy: float = 0.0
    matsubara_beta: float = 1.0
    matsubara: Optional[MatsubaraCircle] = None
    gibbons_hawking_action: float = 0.0
    integration_rule: str = "simpson"
    n_samples_used: int = 0
    adaptive_refinements: int = 0
    critical_action: float = 0.0
    critical_curvature: float = 0.0
    curvature_veto_vacuous: bool = False
    heyting_verdict_value: int = 0
    reason: str = ""
    forensic_provenance: Dict[str, Any] = field(default_factory=dict)


GravitationalShieldState = PolyakovAction  # alias histórico


@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Único payload mutable de RSI-L3 (S11: sólo endurece)."""

    integration_steps: int = 16
    quadrature_tol: float = 1e-10
    amplitude_tol: float = GravitationalConstants.SCHWARZSCHILD_TOLERANCE
    curvature_scale: Optional[float] = None
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


# ═════════════════════════════════════════════════════════════════════════════════════
# §C · UTILIDADES NUMÉRICAS CERTIFICADAS
# ═════════════════════════════════════════════════════════════════════════════════════
def _fro(a: NDArray) -> float:
    return float(la.norm(np.asarray(a), ord="fro", check_finite=False))


def _sym(a: NDArray) -> NDArray[np.float64]:
    x = np.asarray(a, dtype=np.float64)
    return 0.5 * (x + x.T)


def _normalize_metric_source(src: Any) -> NDArray[np.float64]:
    """Normaliza el contrato roto ndarray|dict|None entre módulos del estrato."""
    if src is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(src, dict):
        for k in ("G", "metric", "metric_tensor", "spatial"):
            if k in src:
                return np.asarray(src[k], dtype=np.float64)
        return np.eye(4, dtype=np.float64)
    return np.asarray(src, dtype=np.float64)


def _spd_inverse(g: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inversa vía Cholesky (SPD garantizada); jamás np.linalg.inv en la ruta caliente."""
    L = la.cholesky(_sym(g), lower=True, check_finite=False)
    return la.cho_solve((L, True), np.eye(g.shape[0]), check_finite=False)


def _simpson_weights(n: int) -> NDArray[np.float64]:
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def _simpson_composite(f_vals: NDArray[np.float64], h: float) -> float:
    r"""
    ABI legacy CORREGIDA (el `raise` estaba sin indentar ⇒ IndentationError).
    Simpson compuesto clásico: exige N impar (intervalos pares). Exactitud O(h⁴).
    """
    f = np.asarray(f_vals, dtype=np.float64)
    n = int(f.size)
    if n < 3 or (n - 1) % 2 != 0:
        raise ValueError("Simpson compuesto requiere un número impar de puntos (intervalos pares).")
    return float((h / 3.0) * np.dot(_simpson_weights(n), f))


def _closed_newton_cotes(f: NDArray[np.float64], h: float) -> Tuple[float, str, int]:
    r"""
    S7: cuadratura cerrada de orden 4 SIN fabricar ni descartar muestras.
      N impar      → Simpson compuesto.
      N par, N≥4   → Simpson en los primeros N−3 nodos + 3/8 en los últimos 3 intervalos.
      N=2          → trapecio (orden 2, declarado).
      N≤1          → 0.0, "insufficient_data" (no existe intervalo).
    """
    f = np.asarray(f_vals, dtype=np.float64)
    n = int(f.size)
    if n == 0:
        return 0.0, "empty", 0
    if n == 1:
        return 0.0, "insufficient_data", 1
    if n == 2:
        return float(0.5 * h * (f[0] + f[1])), "trapezoid", 2
    if n % 2 == 1:
        return _simpson_composite(f, h), "simpson", n
    m = n - 3
    if m >= 3:
        left = _simpson_composite(f[:m], h)
        tail = (3.0 * h / 8.0) * (f[m - 1] + 3.0 * f[m] + 3.0 * f[m + 1] + f[m + 2])
        return float(left + tail), "simpson+3/8", n
    return float((3.0 * h / 8.0) * (f[0] + 3.0 * f[1] + 3.0 * f[2] + f[3])), "simpson3/8", n


def kinetic_density(velocity: NDArray[np.float64], metric: NDArray[np.float64]) -> float:
    r"""
    Densidad cinética g̃(v,v) ≥ 0 (Cauchy–Schwarz generalizada para g̃ SPD).
    Nota S5: usa `.real` ⇒ NO holomorfa; nunca debe entrar en una cadena CSMD.
    """
    value = float(np.real(np.asarray(velocity) @ np.asarray(metric) @ np.asarray(velocity)))
    if value < -1e-9:
        raise GravitationalCollapseError(
            "Forma cuadrática negativa: g̃ perdió la SPD en tiempo de evaluación.")
    return max(value, 0.0)


_kinetic_density = kinetic_density  # alias ABI: corrige el NameError de la v3.1


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 1 · ADQUISICIÓN DE MASA INERCIAL  (→ M)                           ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase1_MassAcquisition:
    r"""
    C₁ — Renormalización de Fröhlich saturada + piso C^∞ + RE-ENTRADA IDEMPOTENTE.
    La validación de dominio NO se bypassa con `abs()` (bug de la v3.1).
    """

    def __init__(self, units: Optional[UnitSystem] = None,
                 minimal_mass: float = GravitationalConstants.MINIMAL_INERTIAL_MASS,
                 *, rectify_negative: bool = False) -> None:
        self._units = units or GravitationalConstants.UNITS
        self._m_min = float(minimal_mass)
        self._rectify = bool(rectify_negative)

    # ── 1.1 · Piso inercial C^∞ (S9) ─────────────────────────────────────────────────
    def _smooth_inertial_floor(self, m_linear: float) -> float:
        r"""
        m_floor(m) = √(m² + m_min²). Propiedades CERTIFICADAS:
          m_floor(0) = m_min ; m_floor(m) − m = m_min²/(√(m²+m_min²)+m) → 0 ; C^∞.
        CONSECUENCIA DECLARADA: no existe límite sin masa (todo nodo gravita).
        """
        return math.sqrt(m_linear * m_linear + self._m_min * self._m_min)

    # ── 1.2 · Fröhlich saturado (S10) ────────────────────────────────────────────────
    @staticmethod
    def _saturating_frohlich_factor(coupling: float) -> float:
        r"""
        f(α) = 1 + tanh(α/2π). f'(0) = 1/(2π) (pendiente de bajo acoplamiento exacta),
        f ∈ [1,2) para α≥0 (regularización ultravioleta del polarón).
        """
        return 1.0 + math.tanh(float(coupling) / (2.0 * math.pi))

    # ── 1.3 · Certificado de idempotencia (anti doble conteo) ────────────────────────
    def idempotency_residual(self, m_eff: float) -> float:
        r"""
        |C₁(m**,0,0) − m**|/m**. Con el piso ACTIVO es ≠0 (√(m²+m_min²)) ⇒ la re-entrada
        debe usar `already_renormalized=True`. Se mide y se declara la deriva evitable
        m_n = √(m₀² + n·m_min²) que la v3.1 acumulaba en cada ciclo OODA.
        """
        again = self._smooth_inertial_floor(float(m_eff))
        return abs(again - float(m_eff)) / max(abs(float(m_eff)), 1e-300)

    # ── 1.4 · Adquisición con procedencia y validación NO eludida ────────────────────
    def acquire(
        self, base_cost: float, volatility_alpha: float,
        frohlich_coupling: Optional[float] = None, *,
        already_renormalized: bool = False,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> MassAcquisitionResult:
        r"""
        m** = √((m*(1+α_v/6))² + m_min²) · (1 + tanh(α_f/2π)).
        (El docstring I1 de la v3.1 decía α_f/6: contradicción con el código, corregida.)
        `already_renormalized=True` ⇒ identidad EXACTA (re-entrada idempotente, S9).
        """
        cost, alpha = float(base_cost), float(volatility_alpha)
        coup = 0.0 if frohlich_coupling is None else float(frohlich_coupling)
        if self._rectify:
            cost, alpha, coup = abs(cost), abs(alpha), abs(coup)
            logger.debug("[C₁] rectificación declarada de entradas negativas (abs).")
        if cost < 0.0 or alpha < 0.0 or coup < 0.0:
            raise ValueError(
                "costo, volatilidad y acoplamiento deben ser ≥ 0 "
                f"(recibido {cost}, {alpha}, {coup}); use rectify_negative=True si es deliberado.")
        if not all(math.isfinite(x) for x in (cost, alpha, coup)):
            raise ValueError("Entradas no finitas en la adquisición de masa.")

        if already_renormalized:
            return MassAcquisitionResult.from_effective(cost)

        m_linear = cost * (1.0 + alpha / 6.0)
        m_floored = self._smooth_inertial_floor(m_linear)
        f_froh = self._saturating_frohlich_factor(coup) if coup > 0.0 else 1.0
        m_eff = m_floored * f_froh
        if not math.isfinite(m_eff):
            raise GravitationalCollapseError(f"m** no finita: {m_eff}")
        idem = self.idempotency_residual(m_eff)
        logger.debug("[C₁] m**=%.6e (base=%.6e, α_v=%.4f, f_F=%.6f, idem_res=%.2e)",
                     m_eff, cost, alpha, f_froh, idem)
        return MassAcquisitionResult(
            raw_cost=cost, volatility_alpha=alpha, frohlich_coupling=coup,
            frohlich_factor=f_froh, effective_mass=float(m_eff), floor_applied=True,
            already_renormalized=False, idempotency_residual=float(idem),
            provenance=dict(provenance or {}),
        )

    # ── 1.5 · MÉTODO TERMINAL DE C₁ → continuación a C₂ ─────────────────────────────
    def acquire_and_handoff(
        self, base_cost: float, volatility_alpha: float,
        frohlich_coupling: Optional[float] = None, *,
        already_renormalized: bool = False,
        provenance: Optional[Dict[str, Any]] = None,
        continuation: Optional[Callable[[MassAcquisitionResult], R]] = None,
    ) -> Union[MassAcquisitionResult, R]:
        r"""
        Definición formal final de C₁: `continuation: M → R`. Su cola
        `return continuation(M)` es el arranque de `Phase2._continue_from_phase1_mass`.
        """
        mass = self.acquire(base_cost, volatility_alpha, frohlich_coupling,
                            already_renormalized=already_renormalized, provenance=provenance)
        if continuation is None:
            return mass
        return continuation(mass)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 2 · DEFORMACIÓN, CERTIFICADO DE GAUGE Y CURVATURA REAL  (M → W)   ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase2_ConformalWarping:
    r"""
    C₂ — Deformación con TRES modos y certificado de gauge:
      · "gauge_congruence"  : legado. S1 demuestra que es un pullback ⇒ Riemann≡0.
      · "isotropic_conformal": Ω²(r) = (1+Gm/(2c²r))⁴ ⇒ curvatura GENUINA (S2).
      · "custom_field"      : g(x) suministrada por el llamante.
    Christoffel por forma cerrada (legado, S4) o por campo métrico (CSMD auditado, S5).
    """

    def __init__(self, g_base: Optional[NDArray[np.float64]] = None, *,
                 units: Optional[UnitSystem] = None,
                 mode: str = "gauge_congruence",
                 node_radius: float = GravitationalConstants.DEFAULT_NODE_RADIUS,
                 node_position: Optional[NDArray[np.float64]] = None,
                 legacy_coupling: bool = False,
                 max_condition_number: float = GravitationalConstants.MAX_CONDITION_NUMBER,
                 compute_full_curvature: bool = True) -> None:
        if mode not in ("gauge_congruence", "isotropic_conformal", "custom_field"):
            raise ValueError(f"modo de deformación desconocido: {mode}")
        if node_radius <= 0.0:
            raise DimensionalConsistencyError("node_radius > 0 requerido (κ adimensional).")
        self._units = units or GravitationalConstants.UNITS
        self._mode = mode
        self._r_node = float(node_radius)
        self._node_position = node_position
        self._legacy_coupling = bool(legacy_coupling)
        self._kappa_max = float(max_condition_number)
        self._full_curv = bool(compute_full_curvature)
        self._cache = build_base_metric_cache(
            _normalize_metric_source(G_PHYSICS_RAW if g_base is None else g_base))
        if self._legacy_coupling:
            logger.warning(
                "[C₂] legacy_coupling=True: κ = 2Gm/c⁴ ≈ %.3e·m es DIMENSIONALMENTE "
                "INVÁLIDO y numéricamente inerte; se conserva sólo por reproducibilidad.",
                GravitationalConstants.G_C4_FACTOR)

    # ── 2.1 · Continuación formal desde C₁ ───────────────────────────────────────────
    def _continue_from_phase1_mass(self, mass: MassAcquisitionResult) -> MassAcquisitionResult:
        if not isinstance(mass, MassAcquisitionResult):
            raise TopologicalInvariantError(
                f"Continuidad C₁→C₂ violada: se esperaba MassAcquisitionResult, hay {type(mass)}")
        if not math.isfinite(mass.effective_mass) or mass.effective_mass < 0.0:
            raise GravitationalCollapseError("m** no física en el contrato C₁→C₂.")
        return mass

    # ── 2.2 · Caché base SPD con κ y presupuesto de exactitud ────────────────────────
    @staticmethod
    def build_cache(g_base: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Cholesky = verificación SPD + medio estable para G⁻¹ (cho_solve, no inv).
        Se SIMETRIZA una vez (la congruencia preservaría la asimetría residual).
        Ley de inercia de Sylvester: Λ^{1/2} real positiva ⇒ congruencia conserva SPD.
        """
        g = np.asarray(g_base, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] == 0:
            raise GravitationalCollapseError("G_PHYSICS debe ser cuadrada no vacía.")
        if not np.all(np.isfinite(g)):
            raise GravitationalCollapseError("G_PHYSICS con entradas no finitas.")
        sym_res = _fro(g - g.T) / max(_fro(g), 1e-300)
        if sym_res > _SYM_ATOL:
            raise GravitationalCollapseError(f"G_PHYSICS no simétrica: residual={sym_res:.3e}")
        gs = _sym(g)
        try:
            L = la.cholesky(gs, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise GravitationalCollapseError(f"G_PHYSICS no es SPD (Cholesky): {exc}") from exc
        w = la.eigvalsh(gs, check_finite=False)
        lmin, lmax = float(w[0]), float(w[-1])
        if lmin <= _NONDEGEN_REL * max(abs(lmax), 1e-300):
            raise GravitationalCollapseError(f"G_PHYSICS degenerada: λ_min={lmin:.3e}")
        g_inv = la.cho_solve((L, True), np.eye(gs.shape[0]), check_finite=False)
        kappa = lmax / lmin
        logger.info("[C₂·setup] caché SPD verified (Sylvester) d=%d κ=%.3e", gs.shape[0], kappa)
        return BaseMetricCache(gs, L, g_inv, int(gs.shape[0]), float(kappa), lmin, lmax, float(sym_res))

    # ── 2.3 · Congruencia legada y su CERTIFICADO DE GAUGE (S1) ──────────────────────
    def _construct_warped_metric(
        self, cache: BaseMetricCache, effective_mass: Union[float, complex], node_index: int
    ) -> Tuple[NDArray, NDArray, float]:
        r"""
        g̃ = Λ^{1/2} G Λ^{1/2}, g̃⁻¹ = Λ^{-1/2} G⁻¹ Λ^{-1/2} (analítica, sin resolver).
        δ_k = κ adimensional = 2Gm/(c²r) (o el legado 2Gm/c⁴ si legacy_coupling).
        Devuelve (g̃, g̃⁻¹, κ_dimless). Guardia de overflow de exp (S3).
        """
        n = cache.dimension
        if not (0 <= node_index < n):
            raise IndexError(f"node_index={node_index} fuera de [0,{n}).")
        if self._legacy_coupling:
            delta_k = GravitationalConstants.G_C4_FACTOR * effective_mass
        else:
            m_real = float(np.real(effective_mass))
            kappa_dim = self._units.dimensionless_coupling(m_real, self._r_node)
            delta_k = kappa_dim * (effective_mass / m_real if m_real != 0.0 else 1.0)
        if abs(float(np.real(delta_k))) > _LOG_EXP_CAP:
            raise GravitationalCollapseError(
                f"exp(δ/2) desbordaría: δ={np.real(delta_k):.3e}. Reduzca m o aumente r_node.")
        dtype = np.complex128 if np.iscomplexobj(effective_mass) else np.float64
        delta = np.zeros(n, dtype=dtype)
        delta[node_index] = delta_k
        lam_sqrt, lam_inv = np.exp(delta / 2.0), np.exp(-delta / 2.0)
        g_t = np.outer(lam_sqrt, lam_sqrt) * cache.g_base.astype(dtype)
        g_i = np.outer(lam_inv, lam_inv) * cache.g_inv.astype(dtype)
        return g_t, g_i, float(np.real(delta_k))

    def gauge_certificate(self, cache: BaseMetricCache, kappa_dim: float,
                          node_index: int, measured_riemann_max: float) -> GaugeCertificate:
        r"""
        S1: φ: y^k=(2/κ)e^{κx^k/2}, y^a=x^a ⇒ Dφ=Λ^{1/2}, det Dφ=e^{κx^k/2}>0.
        Con G constante, g̃=φ*G ⇒ Riemann(g̃)≡0. Se certifica y se declara la vacuidad.
        """
        base_const = self._mode == "gauge_congruence"
        det_j = math.exp(min(kappa_dim / 2.0, _LOG_EXP_CAP))
        pure = base_const
        zero = bool(pure and measured_riemann_max <= 1e-10)
        stmt = (
            "g̃ = φ*G con φ difeomorfismo (Dφ=Λ^{1/2}); G constante ⇒ Riemann≡0. "
            "El veto por curvatura es VACUO en este modo: use isotropic_conformal."
            if pure else
            "Deformación NO gauge: curvatura genuina medida a partir del campo métrico."
        )
        if pure:
            logger.warning("[C₂·GAUGE] %s (|Riemann|_max medido=%.3e)", stmt, measured_riemann_max)
        return GaugeCertificate(
            is_pure_gauge=pure, base_is_constant=base_const, diffeomorphism_det=float(det_j),
            riemann_identically_zero=zero, measured_max_abs_riemann=float(measured_riemann_max),
            curvature_veto_vacuous=bool(pure), statement=stmt,
        )

    # ── 2.4 · Christoffel del modo legado: forma cerrada EXACTA (S4) ─────────────────
    @staticmethod
    def _christoffel_from_metric(g_tilde: NDArray, g_inv_tilde: NDArray,
                                 node_index: int, kappa_dim: Optional[float] = None) -> NDArray:
        r"""
        Γ^μ_{νρ} = (κ/4)[[ν=k](g̃^{μk}g̃_{kρ}+[ρ=k]δ^μ_ρ)
                        +[ρ=k](g̃^{μk}g̃_{kν}+[ν=k]δ^μ_ν)
                        −g̃^{μk}g̃_{νρ}([ν=k]+[ρ=k])].
        Verificada: G=I,n=2,k=0 ⇒ único Γ=Γ^0_{00}=κ/2 ⇒ Riemann=0 (consistente con S1).
        `outer(e_k,e_k)` es correcto en esos slots porque [ρ=k]δ^μ_ρ = [μ=k][ρ=k].
        """
        n = g_tilde.shape[0]
        k = node_index
        c = GravitationalConstants.G_C4_FACTOR if kappa_dim is None else float(kappa_dim)
        gk_up, gk_down = g_inv_tilde[:, k], g_tilde[k, :]
        e_k = np.zeros(n, dtype=g_tilde.dtype)
        e_k[k] = 1.0
        base_term = np.outer(gk_up, gk_down) + np.outer(e_k, e_k)
        t1 = np.zeros((n, n, n), dtype=g_tilde.dtype); t1[:, k, :] = (c / 2.0) * base_term
        t2 = np.zeros((n, n, n), dtype=g_tilde.dtype); t2[:, :, k] = (c / 2.0) * base_term
        dsum = np.zeros((n, n), dtype=g_tilde.dtype)
        dsum[k, :] += 1.0
        dsum[:, k] += 1.0
        t3 = (c / 2.0) * np.einsum("m,vr->mvr", gk_up, dsum * g_tilde)
        return 0.5 * (t1 + t2 - t3)

    # ── 2.5 · CSMD con AUDITORÍA DE HOLOMORFÍA (S5) ──────────────────────────────────
    def complex_step_partial_k_christoffel(
        self, cache: BaseMetricCache, effective_mass: float, node_index: int,
        h: Optional[float] = None, *, kappa_dim: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        r"""
        ∂_k Γ = Im Γ(m+ih)/h + O(h²), SIN cancelación catastrófica (Squire–Trapp).
        Se AUDITA contra diferencias centrales: si discrepan, la cadena NO es holomorfa
        (p.ej. np.real/abs/conj/norm) y se usa el valor central, declarándolo.
        `np.iscomplexobj` (no isinstance(...,complex)) evita descartar np.complex64.
        """
        hh = float(h or _CSMD_H)
        g_c, gi_c, kd = self._construct_warped_metric(cache, complex(effective_mass, hh), node_index)
        gam_c = self._christoffel_from_metric(g_c, gi_c, node_index,
                                              kappa_dim if kappa_dim is not None else kd)
        csmd = np.asarray(gam_c.imag / hh, dtype=np.float64)
        hc = _FD_H * max(1.0, abs(float(effective_mass)))
        gp, gip, kdp = self._construct_warped_metric(cache, effective_mass + hc, node_index)
        gm, gim, kdm = self._construct_warped_metric(cache, effective_mass - hc, node_index)
        central = (self._christoffel_from_metric(gp, gip, node_index, kdp)
                   - self._christoffel_from_metric(gm, gim, node_index, kdm)) / (2.0 * hc)
        res = _fro(csmd - central) / max(_fro(central), 1e-300)
        valid = bool(res <= 1e-6)
        if not valid:
            logger.warning("[C₂·CSMD] cadena NO holomorfa (res=%.3e) ⇒ diferencias centrales.", res)
        return (csmd if valid else np.asarray(central, dtype=np.float64)), valid, float(res)

    # ── 2.6 · Curvatura seccional en planos con e_k (cota INFERIOR declarada) ────────
    @staticmethod
    def _sectional_curvatures_around_node(
        g_tilde: NDArray[np.float64], gamma: NDArray[np.float64],
        d_k_gamma: NDArray[np.float64], node_index: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        K(e_k,e_j) = R(e_k,e_j,e_j,e_k)/(g_kk g_jj − g_kj²), j≠k.
        (R(e_k,e_j)e_j)^ρ = ∂_kΓ^ρ_{jj} + Γ^ρ_{kλ}Γ^λ_{jj} − Γ^ρ_{jλ}Γ^λ_{kj}
        (∂_jΓ^ρ_{kj}≡0 pues g̃ no depende de x^j para j≠k).
        Es una COTA INFERIOR del máximo global: sólo cubre planos que contienen e_k.
        """
        n, k = g_tilde.shape[0], node_index
        d_dk = np.einsum("rjj->rj", d_k_gamma)
        d_gg = np.einsum("ljj->lj", gamma)
        gk = gamma[:, k, :]
        r_vec = d_dk + gk @ d_gg - np.einsum("rjl,lj->rj", gamma, gk)
        r_full = g_tilde[:, k] @ r_vec
        denom = g_tilde[k, k] * np.diagonal(g_tilde) - g_tilde[k, :] ** 2
        curv = np.zeros(n, dtype=np.float64)
        valid = np.abs(denom) > 1e-14
        valid[k] = False
        curv[valid] = np.real(r_full[valid]) / np.real(denom[valid])
        return curv, np.asarray(r_vec, dtype=np.float64)

    # ── 2.7 · Campo métrico conforme isótropo (S2) y engine de curvatura general ─────
    def _isotropic_field(self, mass: float, node_index: int) -> Callable[[NDArray], NDArray]:
        r"""
        g̃(x) = Ω²(r)G, Ω²(r) = (1 + Gm/(2c²r))⁴, r = √(Σ(x_i−x0_i)²).
        Se usa raíz HOLOMORFA (no norm/abs) para habilitar CSMD; r>0 en el punto de
        evaluación. NO es un pullback de una métrica plana ⇒ curvatura genuina.
        """
        u, G0 = self._units, self._cache.g_base
        n = self._cache.dimension
        x0 = (np.zeros(n) if self._node_position is None
              else np.asarray(self._node_position, dtype=np.float64))
        gm_over_2c2 = u.G * float(mass) / (2.0 * u.c ** 2)

        def field(x: NDArray) -> NDArray:
            z = np.asarray(x)
            d = z - x0.astype(z.dtype)
            r = np.sqrt(np.sum(d * d))
            if abs(complex(r)) < 1e-14:
                raise GravitationalCollapseError("r→0: singularidad isótropa del nodo.")
            omega_sq = (1.0 + gm_over_2c2 / r) ** 4
            return omega_sq * G0.astype(z.dtype)

        return field

    @staticmethod
    def _christoffel_of_field(field: Callable[[NDArray], NDArray], x: NDArray[np.float64],
                              h: float = _FD_H) -> NDArray[np.float64]:
        """Γ^λ_{μν} = ½g^{λσ}(∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν}) por centrales O(h²)."""
        n = x.size
        step = h * max(1.0, float(np.max(np.abs(x))))
        dg = np.empty((n, n, n), dtype=np.float64)
        for mu in range(n):
            e = np.zeros(n); e[mu] = step
            dg[mu] = (_sym(field(x + e)) - _sym(field(x - e))) / (2.0 * step)
        g = _sym(field(x))
        gi = _spd_inverse(g)
        return 0.5 * (np.einsum("ls,mns->lmn", gi, dg, optimize=True)
                      + np.einsum("ls,nms->lmn", gi, dg, optimize=True)
                      - np.einsum("ls,smn->lmn", gi, dg, optimize=True))

    @staticmethod
    def _riemann_from_gamma(gamma: NDArray[np.float64],
                            dgamma: Optional[NDArray[np.float64]] = None) -> Tuple[NDArray, NDArray]:
        """R^ρ_{σμν} = ∂_μΓ^ρ_{νσ} − ∂_νΓ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ}."""
        n = gamma.shape[0]
        dg = np.zeros((n, n, n, n)) if dgamma is None else np.asarray(dgamma)
        quad = (np.einsum("rml,lvs->rsmv", gamma, gamma, optimize=True)
                - np.einsum("rnl,lvs->rsmv", gamma, gamma, optimize=True))
        r13 = np.einsum("mrns->rsmn", dg) - np.einsum("nrms->rsmn", dg) + quad
        return r13, r13

    def _full_curvature(self, field: Callable[[NDArray], NDArray],
                        x: NDArray[np.float64]) -> Tuple[NDArray, NDArray, float, float, float]:
        r"""Riemann completo, Ricci, R, Kretschmann y max|K| sobre marco ortonormal."""
        n = x.size
        g = _sym(field(x))
        gam = self._christoffel_of_field(field, x)
        step = _FD_H * max(1.0, float(np.max(np.abs(x))))
        dgam = np.empty((n, n, n, n))
        for mu in range(n):
            e = np.zeros(n); e[mu] = step
            dgam[mu] = (self._christoffel_of_field(field, x + e)
                        - self._christoffel_of_field(field, x - e)) / (2.0 * step)
        r13, _ = self._riemann_from_gamma(gam, dgam)
        r04 = np.einsum("rl,lsmn->rsmn", g, r13, optimize=True)
        ricci = _sym(np.einsum("pspn->sn", r13, optimize=True))
        gi = _spd_inverse(g)
        R = float(np.einsum("ij,ij->", gi, ricci))
        up = np.einsum("ae,bf,cg,dh,efgh->abcd", gi, gi, gi, gi, r04, optimize=True)
        kret = float(np.einsum("abcd,abcd->", r04, up, optimize=True))
        frame: List[NDArray[np.float64]] = []
        for a in range(n):
            v = np.zeros(n); v[a] = 1.0
            for f in frame:
                nf = float(f @ g @ f)
                if abs(nf) > 1e-12:
                    v = v - (float(v @ g @ f) / nf) * f
            nv = float(v @ g @ v)
            if abs(nv) > 1e-12:
                frame.append(v / math.sqrt(abs(nv)))
        kmax = 0.0
        for a in range(len(frame)):
            for b in range(len(frame)):
                if a == b:
                    continue
                u_, v_ = frame[a], frame[b]
                rv = np.einsum("rsmn,s,m,n->r", r13, v_, u_, v_, optimize=True)
                den = float(u_ @ g @ u_) * float(v_ @ g @ v_) - float(u_ @ g @ v_) ** 2
                if abs(den) > 1e-12:
                    kmax = max(kmax, abs(float(u_ @ g @ rv) / den))
        return gam, ricci, R, kret, kmax

    # ── 2.8 · MÉTODO TERMINAL DE C₂ → continuación a C₃ ─────────────────────────────
    def warp(
        self, mass: MassAcquisitionResult, node_index: int, *,
        custom_field: Optional[Callable[[NDArray], NDArray]] = None,
        x_eval: Optional[NDArray[np.float64]] = None,
        continuation: Optional[Callable[[WarpedSpaceTime], R]] = None,
    ) -> Union[WarpedSpaceTime, R]:
        r"""
        Definición formal final de C₂: `continuation: W → R`, cuya cola es el arranque
        de `Phase3._continue_from_phase2_warp`. Emite el certificado de gauge (S1),
        la cota de κ (S3) y la curvatura real o exactamente nula.
        """
        m = self._continue_from_phase1_mass(mass)
        cache, n = self._cache, self._cache.dimension
        if not (0 <= node_index < n):
            raise IndexError(f"node_index={node_index} fuera de [0,{n}).")
        x = (np.zeros(n) if x_eval is None else np.asarray(x_eval, dtype=np.float64))
        if x_eval is None:
            x = x.copy(); x[node_index] = self._r_node

        if self._mode == "gauge_congruence":
            g_t, g_i, kd = self._construct_warped_metric(cache, m.effective_mass, node_index)
            if np.any(np.diagonal(g_t) <= 0.0):
                raise GravitationalCollapseError("diagonal no positiva tras congruencia (canario).")
            gam = self._christoffel_from_metric(g_t, g_i, node_index, kd)
            dkg, cs_ok, cs_res = self.complex_step_partial_k_christoffel(
                cache, m.effective_mass, node_index, kappa_dim=kd)
            curv, r_red = self._sectional_curvatures_around_node(g_t, gam, dkg, node_index)
            r13, _ = self._riemann_from_gamma(gam, None)
            rmeas = float(np.max(np.abs(r13))) if r13.size else 0.0
            gauge = self.gauge_certificate(cache, kd, node_index, rmeas)
            kmax = float(np.max(np.abs(curv))) if n > 1 else 0.0
            ricci, R, kret = None, 0.0, 0.0
            field = None
            kappa_bound = cache.condition_number * math.exp(min(abs(kd), _LOG_EXP_CAP))
            g_def, g_inv_def = np.real(g_t), np.real(g_i)
        else:
            field = (custom_field if (self._mode == "custom_field" and custom_field is not None)
                     else self._isotropic_field(m.effective_mass, node_index))
            g_def = _sym(field(x))
            g_inv_def = _spd_inverse(g_def)
            gam, ricci, R, kret, kmax = self._full_curvature(field, x)
            curv = np.full(n, kmax, dtype=np.float64)
            r_red = np.zeros((n, n))
            kd = self._units.dimensionless_coupling(m.effective_mass, self._r_node)
            gauge = self.gauge_certificate(cache, kd, node_index, max(abs(R), kret))
            cs_ok, cs_res = False, float("nan")
            kappa_bound = cache.condition_number * float(np.max(np.abs(g_def))) / max(
                float(np.min(np.abs(la.eigvalsh(g_def)))), 1e-300)
            field = field

        w = la.eigvalsh(_sym(g_def), check_finite=False)
        kappa_meas = float(w[-1] / max(w[0], 1e-300))
        if kappa_meas > self._kappa_max or kappa_bound > self._kappa_max:
            raise GravitationalCollapseError(
                f"κ(g̃) fuera del presupuesto de exactitud: medido={kappa_meas:.3e}, "
                f"cota={kappa_bound:.3e} > {self._kappa_max:.3e} (S3).")
        if float(w[0]) <= 0.0:
            raise GravitationalCollapseError("g̃ perdió la SPD en flotante (S3).")

        kscale = 1.0 / math.sqrt(kmax) if kmax > 1e-300 else float("inf")
        verdict = HeytingVerdict.COHERENT
        if gauge.curvature_veto_vacuous or not cs_ok:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        crep = CurvatureReport(
            christoffel=np.asarray(gam, dtype=np.float64),
            christoffel_norm=float(la.norm(np.real(gam))),
            sectional_curvatures=np.asarray(curv, dtype=np.float64),
            max_sectional_curvature=float(kmax), sectional_is_lower_bound=(self._mode == "gauge_congruence"),
            riemann_reduced=np.asarray(r_red, dtype=np.float64), riemann_full=None,
            ricci_tensor=ricci, ricci_scalar=float(R), kretschmann=float(kret),
            curvature_measured=(self._mode != "gauge_congruence"),
            csmd_valid=bool(cs_ok), csmd_vs_central_residual=float(cs_res),
            injectivity_scale=float(kscale), verdict=int(verdict),
        )
        warped = WarpedSpaceTime(
            original_metric=cache.g_base, deformed_metric=g_def,
            christoffel_symbols=crep.christoffel, riemann_curvature=crep.riemann_reduced,
            sectional_curvatures=crep.sectional_curvatures, max_sectional_curvature=float(kmax),
            node_index=int(node_index), deformed_inverse=g_inv_def, mode=self._mode,
            dimensionless_coupling=float(kd), condition_number_bound=float(kappa_bound),
            condition_number_measured=float(kappa_meas), gauge=gauge, curvature=crep,
            metric_field=field, x_eval=x, mass=m, verdict=int(verdict),
        )
        logger.info("[C₂] modo=%s κ_dim=%.3e K_max=%.3e R=%.3e K_ret=%.3e κ(g̃)=%.3e gauge=%s",
                    self._mode, kd, kmax, R, kret, kappa_meas, gauge.is_pure_gauge)
        if continuation is None:
            return warped
        return continuation(warped)

    # ── 2.9 · CALIBRACIÓN EXACTA del engine de curvatura ─────────────────────────────
    def self_test_curvature(self, radius: float = 2.0) -> Dict[str, float]:
        r"""
        S² redonda de radio a: K = 1/a², R = 2/a², Kretschmann = 4/a⁴ (EXACTOS).
        Y certificado de planitud del modo legado (S1): |Riemann| ≈ 0.
        Si estos residuos no son ~0, el engine está mal y se DECLARA.
        """
        def sphere(x: NDArray) -> NDArray:
            return np.diag([radius ** 2, (radius * math.sin(float(x[0]))) ** 2])

        gam, ricci, R, kret, kmax = self._full_curvature(sphere, np.array([1.0, 0.3]))
        out = {
            "sphere_K_relative_residual": abs(kmax - 1.0 / radius ** 2) / (1.0 / radius ** 2),
            "sphere_R_relative_residual": abs(R - 2.0 / radius ** 2) / (2.0 / radius ** 2),
            "sphere_kretschmann_relative_residual": abs(kret - 4.0 / radius ** 4) / (4.0 / radius ** 4),
        }
        cache2 = build_base_metric_cache(np.eye(2))
        gt, gi, kd = self._construct_warped_metric(cache2, 1.0, 0)
        g2 = self._christoffel_from_metric(gt, gi, 0, kd)
        r13, _ = self._riemann_from_gamma(g2, None)
        out["legacy_gauge_riemann_max"] = float(np.max(np.abs(r13)))
        logger.info("[C₂·SELFTEST] %s", out)
        return out


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 3 · POLYAKOV, MATSUBARA Y VETO EN Ω₃  (W → P)                     ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingPolyakovDecider:
    r"""
    C₃ — Acción de Polyakov sobre el círculo de Matsubara (S6), cuadratura cerrada
    sin fabricar muestras (S7), veto en espacio logarítmico (S8) y colapso en Ω₃.
    Veto SUAVE por defecto para que la Fase 3 del soberano Einstein reciba el DTO.
    """

    def __init__(self, units: Optional[UnitSystem] = None, *,
                 hbar_eff: Optional[float] = None,
                 amplitude_tol: float = GravitationalConstants.SCHWARZSCHILD_TOLERANCE,
                 matsubara_mode: str = "unit_interval",
                 curvature_scale: Optional[float] = None) -> None:
        if matsubara_mode not in ("unit_interval", "thermal_circle"):
            raise ValueError("matsubara_mode ∈ {'unit_interval','thermal_circle'}")
        self._units = units or GravitationalConstants.UNITS
        self._hbar = float(hbar_eff if hbar_eff is not None else self._units.hbar)
        if self._hbar <= 0.0:
            raise DimensionalConsistencyError("ħ_eff > 0 requerido.")
        self._tol = float(amplitude_tol)
        self._mode = matsubara_mode
        self._curv_scale = curvature_scale

    # ── 3.1 · Continuación formal desde C₂ ───────────────────────────────────────────
    def _continue_from_phase2_warp(self, warped: WarpedSpaceTime) -> WarpedSpaceTime:
        if not isinstance(warped, WarpedSpaceTime):
            raise TopologicalInvariantError(
                f"Continuidad C₂→C₃ violada: se esperaba WarpedSpaceTime, hay {type(warped)}")
        return warped

    # ── 3.2 · Círculo de Matsubara derivado de Hawking (S6) ──────────────────────────
    def matsubara_circle(self, mass: float, beta_supplied: Optional[float] = None) -> MatsubaraCircle:
        r"""
        β_τ = ħ/(k_B T_H) = 8πGM/c³. Déficit cónico δ=2π(1−β/β_H).
        I_E = β_E Mc²/2 = S_BH/k_B (Gibbons–Hawking, identidad exacta).
        MONOTONÍA DECLARADA: S_E = E/β decrece con M; I_E crece como M².
        """
        u = self._units
        M = float(mass)
        if M <= 0.0:
            return MatsubaraCircle(float(beta_supplied or 1.0), 0.0, 0.0, float("nan"),
                                   False, float("inf"), 0.0, 0.0,
                                   "indefinida (M=0)", self._mode, int(HeytingVerdict.DEGRADED))
        beta_tau = u.matsubara_beta_time(M)
        beta_e = beta_tau / u.hbar
        t_h = u.hawking_temperature(M)
        beta_used = float(beta_supplied) if beta_supplied else (
            beta_tau if self._mode == "thermal_circle" else 1.0)
        deficit = 2.0 * math.pi * (1.0 - beta_used / beta_tau) if beta_tau > 0 else float("nan")
        i_e = beta_e * M * u.c ** 2 / 2.0
        s_bh = u.k_B * (4.0 * math.pi * (2.0 * u.G * M / u.c ** 2) ** 2) / (4.0 * u.planck_length_sq)
        regular = bool(self._mode == "unit_interval" or abs(deficit) <= 1e-8)
        verdict = HeytingVerdict.COHERENT if regular else HeytingVerdict.DEGRADED
        return MatsubaraCircle(
            beta_supplied=beta_used, beta_hawking_time=beta_tau, beta_hawking_energy=beta_e,
            conical_deficit=float(deficit), regular=regular, hawking_temperature=float(t_h),
            gibbons_hawking_action=float(i_e), bekenstein_hawking_entropy=float(s_bh),
            path_action_monotonicity="S_E=E/β DECRECE con M; I_E=β_E Mc²/2 CRECE como M²",
            mode=self._mode, verdict=int(verdict),
        )

    # ── 3.3 · Energía de Dirichlet y cuadratura cerrada (S7) ─────────────────────────
    @staticmethod
    def dirichlet_energy(trajectory: NDArray[np.float64], metric: NDArray[np.float64]
                         ) -> Tuple[float, str, int, float]:
        r"""
        E[γ] = ½∫₀¹ g̃(dγ/ds)² ds. 1D ⇒ integrando constante, error EXACTAMENTE 0.
        2D ⇒ cuadratura cerrada de orden 4 sin duplicar ni descartar muestras, con
        estimación de Richardson |S_h−S_2h|/15 cuando la resolución lo permite.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 1:
            return 0.5 * kinetic_density(traj, metric), "exact_constant", 1, 0.0
        if traj.ndim != 2:
            raise ValueError("attention_trajectory debe ser 1D (velocidad) o 2D (muestras).")
        m = traj.shape[0]
        if m < 1:
            return 0.0, "empty", 0, float("nan")
        f = np.array([0.5 * kinetic_density(traj[i], metric) for i in range(m)], dtype=np.float64)
        h = 1.0 / max(m - 1, 1)
        energy, rule, used = _closed_newton_cotes(f, h)
        err = float("nan")
        if m >= 5 and (m - 1) % 2 == 0:
            coarse = f[::2]
            if coarse.size >= 3 and (coarse.size - 1) % 2 == 0:
                err = abs(energy - _simpson_composite(coarse, 2.0 * h)) / 15.0
        return float(energy), rule, int(used), float(err)

    def refine_trajectory(self, trajectory: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        r"""
        Uso REAL de `integration_steps` (parámetro muerto en v3.1): remuestreo lineal
        a `steps+1` nodos sobre [0,1] para habilitar Richardson y convergencia O(h⁴).
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 1 or traj.shape[0] < 2 or steps < 2:
            return traj
        s_old = np.linspace(0.0, 1.0, traj.shape[0])
        s_new = np.linspace(0.0, 1.0, int(steps) + 1)
        return np.column_stack([np.interp(s_new, s_old, traj[:, j]) for j in range(traj.shape[1])])

    # ── 3.4 · Relajación geodésica: Lyapunov de RSI-L3 ───────────────────────────────
    @staticmethod
    def relax_geodesic(trajectory: NDArray[np.float64], metric: NDArray[np.float64],
                       steps: int = 16) -> Tuple[NDArray[np.float64], float, float, bool]:
        r"""
        Flujo de calor de mapas armónicos con extremos FIJOS: ∂_tγ = γ̈ (Γ=0 en el
        modelo de velocidad constante) ⇒ dE/dt = −∫|τ|² ≤ 0. Lyapunov geométrico.
        Paso CFL h ≤ 0.45·Δs²/2 (estabilidad del Laplaciano discreto).
        """
        p = np.array(trajectory, dtype=np.float64, copy=True)
        if p.ndim != 2 or p.shape[0] < 3:
            e0, _, _, _ = Phase3_HeytingPolyakovDecider.dirichlet_energy(p, metric)
            return p, e0, e0, True
        ds = 1.0 / (p.shape[0] - 1)
        h = _HEAT_CFL * ds * ds / 2.0
        e_prev = float(np.sum(np.einsum("ti,ij,tj->t", np.diff(p, axis=0), metric,
                                        np.diff(p, axis=0))) / (2.0 * ds))
        e0, monotone = e_prev, True
        for _ in range(max(1, int(steps))):
            p[1:-1] += h * (p[2:] - 2.0 * p[1:-1] + p[:-2]) / (ds * ds)
            d = np.diff(p, axis=0)
            e_k = float(np.sum(np.einsum("ti,ij,tj->t", d, metric, d)) / (2.0 * ds))
            monotone = monotone and (e_k <= e_prev + 1e-12)
            e_prev = e_k
        return p, e0, e_prev, bool(monotone)

    # ── 3.5 · MÉTODO TERMINAL DE C₃ ─────────────────────────────────────────────────
    def evaluate(
        self, warped: WarpedSpaceTime, attention_trajectory: NDArray[np.float64],
        integration_steps: int = 16, *,
        beta_supplied: Optional[float] = None,
        curvature_scale: Optional[float] = None,
        strict: bool = False,
        continuation: Optional[Callable[[PolyakovAction], R]] = None,
    ) -> Union[PolyakovAction, R]:
        r"""
        Definición formal final de C₃. Veto dual con OR explícito:
            V_acción    ≡ S_E > −ħ ln(tol)        (comparación en espacio logarítmico)
            V_curvatura ≡ K_max > r_inj⁻²         (escala GEOMÉTRICA, no 1e3)
        `strict=False` (defecto) emite ⊥ en Ω₃ sin lanzar: el soberano Einstein necesita
        el DTO para emitir la telemetría de Bekenstein–Hawking (defecto #10 de la v3.1).
        """
        w = self._continue_from_phase2_warp(warped)
        mass = w.mass.effective_mass if w.mass is not None else 0.0
        traj = self.refine_trajectory(attention_trajectory, int(integration_steps))
        energy, rule, used, err = self.dirichlet_energy(traj, w.deformed_metric)
        circle = self.matsubara_circle(mass, beta_supplied)
        beta = circle.beta_supplied if self._mode == "thermal_circle" else 1.0
        action = float(energy / max(beta, 1e-300))
        if not math.isfinite(action):
            raise GravitationalCollapseError(f"Acción de Polyakov no finita: {action}")

        crit_action = -self._hbar * math.log(self._tol)
        action_veto = bool(action > crit_action)
        scale = (curvature_scale if curvature_scale is not None else self._curv_scale)
        if scale is None:
            kmax = w.max_sectional_curvature
            scale = 1.0 / math.sqrt(kmax) if kmax > 1e-300 else float("inf")
        crit_curv = (1.0 / float(scale) ** 2) if (math.isfinite(scale) and scale > 0) else float("inf")
        curv_veto = bool(w.max_sectional_curvature > crit_curv)
        vacuous = bool(w.gauge.curvature_veto_vacuous) if w.gauge is not None else False
        trapped = bool(action_veto or curv_veto)

        exponent = -action / self._hbar
        amplitude = math.exp(exponent) if exponent > -_LOG_EXP_CAP else 0.0

        verdict = HeytingVerdict(w.verdict)
        verdict = heyting_meet(verdict, HeytingVerdict(circle.verdict))
        if vacuous or not math.isfinite(err):
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if trapped:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        reasons: List[str] = []
        if action_veto:
            reasons.append(f"S_E={action:.4e}>S_crit={crit_action:.4e}")
        if curv_veto:
            reasons.append(f"K_max={w.max_sectional_curvature:.4e}>r_inj⁻²={crit_curv:.4e}")
        if vacuous:
            reasons.append("veto_curvatura_VACUO(modo gauge, Riemann≡0)")

        result = PolyakovAction(
            action_integral=float(action), feynman_amplitude=float(amplitude),
            is_trapped=trapped, action_veto=action_veto, curvature_veto=curv_veto,
            discretization_error=float(err), dirichlet_energy=float(energy),
            matsubara_beta=float(beta), matsubara=circle,
            gibbons_hawking_action=circle.gibbons_hawking_action,
            integration_rule=rule, n_samples_used=int(used), adaptive_refinements=0,
            critical_action=float(crit_action), critical_curvature=float(crit_curv),
            curvature_veto_vacuous=vacuous, heyting_verdict_value=int(verdict),
            reason=";".join(reasons) if reasons else "geodésica_libre",
            forensic_provenance={"mode": w.mode, "kappa_dim": w.dimensionless_coupling,
                                 "kappa_g_tilde": w.condition_number_measured,
                                 "pure_gauge": vacuous, "units": self._units.name},
        )
        logger.info("[C₃] S_E=%.6e (E=%.6e, β=%.3e) Ψ=%.6e veto=%s Ω₃=%s",
                    action, energy, beta, amplitude, trapped, HeytingVerdict(int(verdict)).name)
        if strict and trapped:
            raise EventHorizonViolation(
                f"Obstrucción geodésica en nodo {w.node_index}: {result.reason}. "
                "El LLM está forzado matemáticamente a absorber este costo.")
        if continuation is None:
            return result
        return continuation(result)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR · COMPOSICIÓN C₃∘C₂∘C₁ + RSI-L3 + PUENTES                         ══
# ═════════════════════════════════════════════════════════════════════════════════════
class GravitationalShieldFunctor(Morphism):
    r"""
    Funtor Endomórfico F: WISDOM → WISDOM. Composición por CONTINUACIÓN (no herencia).
    Mantiene la ABI v3.1 (`enforce_gravitational_attractor`) y corrige las dos rupturas
    de contrato con el agente Einstein mediante adaptadores de doble firma.
    """

    def __init__(self, g_base: Optional[NDArray[np.float64]] = None, *,
                 units: Optional[UnitSystem] = None,
                 mode: str = "gauge_congruence",
                 node_radius: float = GravitationalConstants.DEFAULT_NODE_RADIUS,
                 matsubara_mode: str = "unit_interval",
                 legacy_coupling: bool = False,
                 strict_default: bool = True,
                 rectify_negative: bool = False) -> None:
        self._units = units or GravitationalConstants.UNITS
        self._mode = mode
        self._strict_default = bool(strict_default)
        self.phase1 = Phase1_MassAcquisition(self._units, rectify_negative=rectify_negative)
        self.phase2 = Phase2_ConformalWarping(
            g_base, units=self._units, mode=mode, node_radius=node_radius,
            legacy_coupling=legacy_coupling)
        self.phase3 = Phase3_HeytingPolyakovDecider(self._units, matsubara_mode=matsubara_mode)
        self._metric_cache = self.phase2._cache
        try:
            super().__init__(name="GravitationalShieldFunctor")
        except TypeError:
            super().__init__()
        logger.info("[GravityShield] v4 d=%d modo=%s unidades=%s Matsubara=%s strict=%s",
                    self._metric_cache.dimension, mode, self._units.name,
                    matsubara_mode, strict_default)

    # ── ABI: extractor tipado SIN eludir la validación ───────────────────────────────
    @staticmethod
    def _extract_polaron_fields(polaron: Any) -> Tuple[float, float, Optional[float]]:
        r"""
        La v3.1 aplicaba `abs()` ⇒ los `raise ValueError` de adquisición eran
        INALCANZABLES. Ahora se pasan los valores crudos y la validación actúa
        (o se rectifica explícitamente con `rectify_negative=True`).
        """
        cost = float(getattr(polaron, "inertial_mass", getattr(polaron, "effective_mass", 0.0)))
        alpha = float(getattr(polaron, "volatility_alpha", 0.0))
        raw = getattr(polaron, "frohlich_coupling", None)
        return cost, alpha, (None if raw is None else float(raw))

    # ── ABI: ciclo completo con las tres fases anidadas por CPS ──────────────────────
    def enforce_gravitational_attractor(
        self, polaron: Any, node_index: int,
        llm_attention_vector: NDArray[np.float64],
        integration_steps: int = 16, *,
        strict: Optional[bool] = None,
        already_renormalized: Optional[bool] = None,
        beta_supplied: Optional[float] = None,
        return_state: bool = False,
    ) -> Union[PolyakovAction, Tuple[PolyakovAction, WarpedSpaceTime]]:
        r"""
        F = C₃∘C₂∘C₁ por continuaciones anidadas. `integration_steps` por fin se USA
        (remuestreo para Richardson). `already_renormalized` evita la deriva del piso;
        por defecto se detecta si el polarón trae masa ya renormalizada y α=α_f=0.
        """
        cost, alpha, coup = self._extract_polaron_fields(polaron)
        if already_renormalized is None:
            already_renormalized = bool(alpha == 0.0 and (coup or 0.0) == 0.0
                                        and getattr(polaron, "base_electron", None) is None
                                        and hasattr(polaron, "effective_mass"))
        strict_eff = self._strict_default if strict is None else bool(strict)
        trace: Dict[str, Any] = {}

        def _c3(warped: WarpedSpaceTime) -> Tuple[PolyakovAction, WarpedSpaceTime]:
            trace["W"] = warped
            action = self.phase3.evaluate(
                warped, llm_attention_vector, integration_steps,
                beta_supplied=beta_supplied, strict=strict_eff, continuation=lambda p: p)
            assert isinstance(action, PolyakovAction)
            return action, warped

        def _c2(mass: MassAcquisitionResult) -> Tuple[PolyakovAction, WarpedSpaceTime]:
            trace["M"] = mass
            out = self.phase2.warp(mass, int(node_index), continuation=_c3)
            assert isinstance(out, tuple)
            return out

        action, warped = self.phase1.acquire_and_handoff(
            cost, alpha, coup, already_renormalized=already_renormalized,
            provenance={"node_index": int(node_index)}, continuation=_c2)
        if action.is_trapped:
            logger.warning("[VETO] nodo %d: %s", node_index, action.reason)
        return (action, warped) if return_state else action

    # ── RSI-L3: ledger + Banach + endurecimiento monótono (S11) ──────────────────────
    @staticmethod
    def _ledger(action: PolyakovAction, warped: WarpedSpaceTime) -> Tuple[float, Tuple[float, ...]]:
        r"""
        Componentes NO NEGATIVAS: E₁ energía de Dirichlet (decrece bajo relajación),
        E₂ error de cuadratura relativo, E₃ |déficit cónico|/2π,
        E₄ exceso de acción sobre el umbral, E₅ κ(g̃)/κ_max.
        La monotonía del COMPUESTO se IMPONE por aceptación con rollback.
        """
        e1 = abs(action.dirichlet_energy)
        e2 = (abs(action.discretization_error) / max(abs(action.action_integral), 1e-300)
              if math.isfinite(action.discretization_error) else 0.0)
        e3 = (abs(action.matsubara.conical_deficit) / (2 * math.pi)
              if action.matsubara and math.isfinite(action.matsubara.conical_deficit) else 0.0)
        e4 = max(0.0, action.action_integral - action.critical_action)
        e5 = warped.condition_number_measured / GravitationalConstants.MAX_CONDITION_NUMBER
        comp = (float(e1), float(e2), float(e3), float(e4), float(e5))
        return float(np.mean(comp)), comp

    def run_rsi_l3_cycle(
        self, polaron: Any, node_index: int, attention_vector: NDArray[np.float64], *,
        max_iterations: int = 6, tolerance: float = 1e-12,
        hyperparameters: Optional[RSIL3Hyperparameters] = None,
    ) -> Tuple[PolyakovAction, RSIL3Report]:
        r"""
        R₁ (C₁): idempotencia de la renormalización. R₂ (C₂): cota de κ y curvatura real.
        R₃ (C₃): cuadratura cerrada + Richardson + relajación geodésica (Lyapunov E).
        META: sólo REFINA el remuestreo o APRIETA tolerancias; rollback si V crece.
        """
        hyper = hyperparameters or RSIL3Hyperparameters()
        best, warped = self.enforce_gravitational_attractor(
            polaron, node_index, attention_vector, hyper.integration_steps,
            strict=False, return_state=True)
        v_prev, _ = self._ledger(best, warped)
        v0 = v_prev
        hist: List[float] = [v_prev]
        accepted = rejected = 0
        q, bound, certified, monotone = 1.0, float("inf"), False, True

        for _ in range(max(1, int(max_iterations))):
            traj = self.phase3.refine_trajectory(attention_vector, hyper.integration_steps)
            relaxed, e0, e1, mono = self.phase3.relax_geodesic(
                traj, warped.deformed_metric, hyper.integration_steps)
            cand, w_c = self.enforce_gravitational_attractor(
                polaron, node_index, relaxed, hyper.integration_steps,
                strict=False, return_state=True)
            v_k, _ = self._ledger(cand, w_c)
            ok = math.isfinite(v_k) and (v_k <= v_prev + tolerance) and mono
            if ok:
                accepted += 1
                best, warped = cand, w_c
                hist.append(v_k)
                if len(hist) >= 3:
                    ratios = [abs(hist[i] - hist[i - 1]) / max(abs(hist[i - 1] - hist[i - 2]), 1e-300)
                              for i in range(len(hist) - 1, 1, -1)[:3]]
                    q = float(np.median(ratios))
                    certified = q < 1.0
                    bound = q / (1.0 - q) * abs(hist[-1] - hist[-2]) if certified else float("inf")
                delta, v_prev = abs(v_prev - v_k), v_k
            else:
                rejected += 1
                monotone = False
                delta = float("inf")
            gain = (1.0 - hyper.trust_radius) if (ok and q < 0.5) else 0.5
            hyper = replace(
                hyper,
                integration_steps=int(min(hyper.integration_steps * (2 if ok else 1), 4096)),
                quadrature_tol=max(hyper.quadrature_tol * gain, 1e-14),
                amplitude_tol=max(hyper.amplitude_tol * gain, 1e-15),
                trust_radius=float(np.clip(hyper.trust_radius * (1.0 if ok else 0.5), 1e-3, 0.5)),
                meta_depth=hyper.meta_depth + 1)
            if ok and delta < tolerance and certified:
                break

        verdict = int(HeytingVerdict.COHERENT if (monotone and certified) else HeytingVerdict.DEGRADED)
        report = RSIL3Report(
            iterations=accepted + rejected, accepted=accepted, rejected=rejected,
            lyapunov_initial=float(v0), lyapunov_final=float(v_prev),
            lyapunov_history=tuple(float(x) for x in hist), contraction_q=float(q),
            banach_error_bound=float(bound), contraction_certified=bool(certified),
            monotone=bool(monotone), tightening_only=True, hyperparameters=hyper, verdict=verdict)
        return replace(best, heyting_verdict_value=int(
            heyting_meet(best.heyting_verdict_value, verdict))), report

    # ── Puente con el soberano Einstein (import perezoso: evita ciclo) ───────────────
    def govern_with_einstein_hilbert(self, polaron: Any, node_index: int,
                                     attention_vector: NDArray[np.float64],
                                     market_pressure: float = 0.0) -> Dict[str, Any]:
        r"""
        Delega el ciclo OODA al agente superior con veto SUAVE (para que su Fase 3
        emita la termodinámica de horizonte) y fusiona veredictos por meet en Ω₃.
        """
        action, warped = self.enforce_gravitational_attractor(
            polaron, node_index, attention_vector, strict=False, return_state=True)
        local = int(action.heyting_verdict_value)
        try:
            from app.omega.einstein_hilbert_agent import EinsteinHilbertAgent  # perezoso
        except ImportError:
            return {"einstein": None, "local_verdict": local,
                    "merged_verdict": int(heyting_meet(local, HeytingVerdict.DEGRADED))}
        agent = EinsteinHilbertAgent(metric_tensor=warped.original_metric)
        state = agent.execute_covariant_ooda(polaron, node_index, attention_vector,
                                             market_pressure, strict=False, return_state=True)
        remote = int(getattr(state, "heyting_verdict_value", local))
        return {"einstein": state, "shield_action": action, "local_verdict": local,
                "merged_verdict": int(heyting_meet(local, remote))}

    def verify_complete_integrity(self) -> Dict[str, Any]:
        """Auditoría end-to-end: SPD, calibración de curvatura y certificado de gauge."""
        calib = self.phase2.self_test_curvature()
        gauge_ok = calib["legacy_gauge_riemann_max"] <= 1e-10
        curv_ok = max(calib["sphere_K_relative_residual"],
                      calib["sphere_R_relative_residual"],
                      calib["sphere_kretschmann_relative_residual"]) < 1e-4
        verdict = int(HeytingVerdict.COHERENT if (gauge_ok and curv_ok) else HeytingVerdict.DEGRADED)
        return {
            "metric_cache": {"dim": self._metric_cache.dimension,
                             "condition_number": self._metric_cache.condition_number,
                             "symmetry_residual": self._metric_cache.symmetry_residual},
            "units": {"system": self._units.name,
                      "legacy_coupling_inertness": GravitationalConstants.G_C4_FACTOR},
            "mode": self._mode,
            "curvature_calibration": calib,
            "gauge_flatness_certified": gauge_ok,
            "heyting_verdict_value": verdict,
            "schema_version": "4.0.0",
        }


# ═════════════════════════════════════════════════════════════════════════════════════
# §D · ADAPTADORES ABI (corrigen NameError y las dos rupturas de contrato con el agente)
# ═════════════════════════════════════════════════════════════════════════════════════
_DEFAULT_PHASE1 = Phase1_MassAcquisition()
_DEFAULT_PHASE2: Optional[Phase2_ConformalWarping] = None
_DEFAULT_PHASE3 = Phase3_HeytingPolyakovDecider()


def _default_phase2() -> Phase2_ConformalWarping:
    global _DEFAULT_PHASE2
    if _DEFAULT_PHASE2 is None:
        _DEFAULT_PHASE2 = Phase2_ConformalWarping()
    return _DEFAULT_PHASE2


def build_base_metric_cache(g_base: NDArray[np.float64]) -> BaseMetricCache:
    """Caché SPD canónica (Cholesky + κ). Simetriza una sola vez."""
    return Phase2_ConformalWarping.build_cache(g_base)


_build_base_metric_cache = build_base_metric_cache  # alias ABI: corrige el NameError


def _smooth_inertial_floor(m_linear: float,
                           m_min: float = GravitationalConstants.MINIMAL_INERTIAL_MASS) -> float:
    """ABI legacy: piso C^∞ (S9)."""
    return math.sqrt(m_linear * m_linear + m_min * m_min)


def _saturating_frohlich_factor(coupling: float) -> float:
    """ABI legacy: f(α)=1+tanh(α/2π) (S10)."""
    return Phase1_MassAcquisition._saturating_frohlich_factor(coupling)


def _acquire_effective_mass(base_cost: float, volatility_alpha: float,
                            frohlich_coupling: Optional[float] = None, *,
                            already_renormalized: bool = False) -> MassAcquisitionResult:
    r"""
    ABI legacy. Devuelve el DTO (correcto), pero `MassAcquisitionResult.__float__`
    restaura `float(_acquire_effective_mass(...))` del agente Einstein (ruptura #5).
    """
    return _DEFAULT_PHASE1.acquire(base_cost, volatility_alpha, frohlich_coupling,
                                   already_renormalized=already_renormalized)


def _construct_warped_metric(cache: BaseMetricCache, effective_mass: Union[float, complex],
                             node_index: int) -> Tuple[NDArray, NDArray]:
    """ABI legacy (2-tupla). Internamente se usa la 3-tupla con κ adimensional."""
    g_t, g_i, _ = _default_phase2()._construct_warped_metric(cache, effective_mass, node_index)
    return g_t, g_i


def _christoffel_from_metric(g_tilde: NDArray, g_inv_tilde: NDArray, node_index: int) -> NDArray:
    """ABI legacy: forma cerrada exacta (S4) con el κ legado."""
    return Phase2_ConformalWarping._christoffel_from_metric(g_tilde, g_inv_tilde, node_index)


def complex_step_partial_k_christoffel(cache: BaseMetricCache, effective_mass: float,
                                       node_index: int,
                                       h: Optional[float] = None) -> NDArray[np.float64]:
    """ABI legacy: CSMD auditado (S5); devuelve sólo el tensor."""
    out, _, _ = _default_phase2().complex_step_partial_k_christoffel(
        cache, effective_mass, node_index, h)
    return out


def _sectional_curvatures_around_node(g_tilde: NDArray, gamma: NDArray, d_k_gamma: NDArray,
                                      node_index: int) -> Tuple[NDArray, NDArray]:
    """ABI legacy: cota INFERIOR del máximo seccional (planos con e_k)."""
    return Phase2_ConformalWarping._sectional_curvatures_around_node(
        g_tilde, gamma, d_k_gamma, node_index)


def _deform_metric_tensor(cache: Optional[BaseMetricCache] = None,
                          mass_result: Optional[MassAcquisitionResult] = None,
                          node_index: int = 0, *,
                          g_base: Optional[NDArray[np.float64]] = None,
                          effective_mass: Optional[float] = None,
                          mode: Optional[str] = None) -> WarpedSpaceTime:
    r"""
    ADAPTADOR DE DOBLE FIRMA (corrige la ruptura #5 con el agente Einstein):
      · legado:  _deform_metric_tensor(cache, mass_result, node_index)
      · agente:  _deform_metric_tensor(g_base=..., effective_mass=..., node_index=...)
    La re-entrada usa `MassAcquisitionResult.from_effective` ⇒ IDEMPOTENTE (S9).
    """
    phase2 = _default_phase2() if mode is None else Phase2_ConformalWarping(g_base, mode=mode)
    cache_eff = cache if cache is not None else build_base_metric_cache(
        _normalize_metric_source(G_PHYSICS_RAW if g_base is None else g_base))
    if cache is not None:
        phase2._cache = cache_eff
    mass_eff = (mass_result if mass_result is not None
                else MassAcquisitionResult.from_effective(float(effective_mass or 0.0)))
    out = phase2.warp(mass_eff, int(node_index))
    assert isinstance(out, WarpedSpaceTime)
    return out


def _evaluate_feynman_kac(warped_space: WarpedSpaceTime,
                          attention_trajectory: NDArray[np.float64],
                          integration_steps: int = 16, *,
                          strict: bool = False) -> PolyakovAction:
    """ABI legacy: cuadratura cerrada + Matsubara + Ω₃, sin fabricar muestras (S7)."""
    out = _DEFAULT_PHASE3.evaluate(warped_space, attention_trajectory, integration_steps,
                                   strict=strict)
    assert isinstance(out, PolyakovAction)
    return out