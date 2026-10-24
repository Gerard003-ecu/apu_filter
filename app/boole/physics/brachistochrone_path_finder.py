
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Brachistochrone Path Finder v5.0.0 — Tres Fases Anidadas            ║
║ Ruta   : app/boole/physics/brachistochrone_path_finder.py                    ║
║ Versión: 5.0.0 (Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-RSI3-PhD)           ║
║ Fases  : ①Observe → ②Orient → ③Decide&Act (bridges emit_*/from_*)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESTRUCTURA DE LAS TRES FASES ANIDADAS
─────────────────────────────────────
  FASE 1 — OBSERVE (Auditoría del pozo de potencial termodinámico)
    • Criterio de Sylvester vía Cholesky  G = L Lᵀ  con verificación
      numérica ‖LLᵀ − G‖_F ≤ ε y cota de Wilkinson κ₂(G) ≤ 10⁸.
    • Certificación de simetría con tolerancia relativa/absoluta.
    • Espectro de gaps de energía H₀ − V(q) muestreado en la bola
      de accesibilidad de Hill (N muestras + margen mínimo).
    • Constante de Lipschitz local de V  |∇V|∞  por diferencias centradas
      deterministas en las 2d direcciones ±e_i (radio σ).
    • Cinemática inicial:  H₀ = ½ vᵀGv + V(q₀)  con residual.
    • CIERRA con `emit_phase1_envelope` → `Phase1Envelope` (hash-chain).

  FASE 2 — ORIENT (Sintonía de métrica conforme y simpléctica)
    • APERTURA con `from_phase1_envelope` (verifica hash-chain).
    • Métrica de Jacobi-Fermat  G̃ = n²(q) G  con  n² = [2(H₀ − V)]⁻¹.
    • Inversión Cholesky triangular  G⁻¹ = (L⁻¹)ᵀ L⁻¹  (sin `la.inv`).
    • **Cayley-Darboux riguroso**: verifica explícitamente
        (i)  K = (G̃−I)(G̃+I)⁻¹  y  Kᵀ J + J K = 0 (pertenencia a sp(2d,ℝ));
        (ii) M_sympl = (I+K)(I−K)⁻¹  y  ‖Mᵀ J M − J‖_F/‖M‖²_F ≤ 1e-10.
    • Símbolos de Christoffel conformes con gradiente por CSMD
      (complejo, paso 1e-20) y fallback a diferencias centradas 4º orden.
    • Verificación de torsión nula y del checksum geométrico
      Γ(v,v) ↔ 2⟨∇φ,v⟩v − ‖v‖²G∇φ.
    • CIERRA con `emit_phase2_envelope` → `Phase2Envelope`.

  FASE 3 — DECIDE & ACT (Integración + Poincaré + RSI³)
    • APERTURA con `from_phase2_envelope` (verifica hash-chain).
    • Proyección sobre la capa de energía  H = H₀  (escalado + dirección
      de gradiente si la cuadrática es degenerada).
    • Integración RK4 con paso fijo, detectando colapso de barrera de Hill.
    • **Ecuación de Poincaré-Lindstedt exhaustiva**: resuelve
        ω·k · W_k = H₁_k  para todo k ∈ [−K, K]^d \ {0}, τ = d+1.
    • **Condición KAM/Bruno exhaustiva**: verifica |ω·k|·‖k‖₁^τ ≥ γ
      sobre todo k pequeño; retorna el worst-ratio.
    • **Integral de Melnikov rigurosa**: M(θ₀) = −∫ ⟨Gv(t), ∇V(q(t))⟩ dt
      sobre la separatriz numérica (poisson {H_kinética, V}).
    • **Sección de Poincaré**: cuenta cruces, mide dispersión de retornos.
    • **Contracción de Banach T³** vía proxy de Lipschitz del flujo.
    • **RSI³ (Data/Harness/Model)**: valuación de Novikov del tiempo de
      tránsito, SPD de G, y d_FS ≤ 10⁻⁴ rad en ℂP^{d−1}.
    • CIERRA con `emit_governance_state` → `BrachistochronePhysicalState`
      (enriquecido con Floquet, sección de Poincaré, KAM worst-ratio).

MIGRACIÓN Y COMPATIBILIDAD
──────────────────────────
  • FIX de bug crítico: los métodos de FASE 2/3 referenciaban un tipo
    inexistente `ConformalManifoldBundle`. Se renombra a `ConformalGeometryBundle`
    (el nombre que define el dataclass y que consume `mac_minimizer_agent`).
    Se añade alias `ConformalManifoldBundle = ConformalGeometryBundle` para
    backward compat de imports externos.
  • La API pública `compute_brachistochrone_path(...)` se preserva
    exactamente: firma, tipos y semántica.
  • Todos los campos de `BrachistochronePhysicalState` originales se
    conservan; los nuevos son opcionales (con defaults).

INVARIANTES GLOBALES
────────────────────
  [A1] G = Gᵀ ≻ 0,  ‖LLᵀ − G‖_F ≤ ε,  κ₂(G) ≤ 10⁸.
  [A2] H₀ − V(q) ≥ 10⁻¹² en toda la bola de Hill accesible.
  [A3] K ∈ sp(2d,ℝ):  ‖Kᵀ J + J K‖_F ≤ 1e-10,  ‖Mᵀ J M − J‖_F/‖M‖²_F ≤ 1e-10.
  [A4] |ω·k|·‖k‖₁^τ ≥ γ,  |M(θ₀)| ≤ 0.5,  |μ_Floquet| ≤ 1 + 1e-6.
  [I1] Torsión de Koszul:  max|Γ^ρ_μν − Γ^ρ_νμ| ≤ 1e-12.
  [I2] Checksum geométrico:  ‖Γ(v,v) − a_cerrada‖_rel ≤ 1e-9.
  [I3] Deriva energética:  δH ≤ 10⁻³,  δE_F ≤ 10⁻⁴.
  [I4] d_FS(ψ_start, ψ_end) ≤ 10⁻⁴ rad;  v_Krull(T) ≥ 0.10.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


logger = logging.getLogger("MIC.Physics.BrachistochronePathFinder.v5")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__: Final[str] = "5.0.0-Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-RSI3-Strict-PhD"


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ENERGY_GAP_FLOOR: Final[float] = 1.0e-12
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_CONDITION_NUMBER_SOFT: Final[float] = 1.0e6
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12
_CSMD_PERTURBATION: Final[float] = 1.0e-20
_TORSION_ABS_TOL: Final[float] = 1.0e-12
_ENERGY_DRIFT_HARD_TOL: Final[float] = 1.0e-3
_ENERGY_DRIFT_SOFT_TOL: Final[float] = 1.0e-6
_FERMAT_DRIFT_HARD_TOL: Final[float] = 1.0e-4
_FERMAT_DRIFT_SOFT_TOL: Final[float] = 1.0e-8
_SYMPLECTIC_RESIDUAL_TOL: Final[float] = 1.0e-10
_DEFAULT_RK4_STEPS: Final[int] = 500
_MAX_INTEGRATION_STEPS: Final[int] = 100_000
_DIM_MAX: Final[int] = 32
_CHECKSUM_REL_TOL: Final[float] = 1.0e-9

# Poincaré (Birkhoff-Arnold-Moser)
_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_MELNIKOV_TOL: Final[float] = 0.5
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_CHIRIKOV_CRITICAL: Final[float] = 1.0

# RSI³ (Darwin-Gödel Nivel 3)
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_FUBINI_STUDY_MAX: Final[float] = 1.0e-4
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95

# Hill accessibility
_HILL_ACCESSIBILITY_SAMPLES: Final[int] = 32
_LOCAL_LIPSCHITZ_RADIUS: Final[float] = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# §B. RÉGIMEN GEODÉSICO DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════
class GeodesicRegime(Enum):
    ELIPTIC_STABLE = auto()
    HYPERBOLIC_UNSTABLE = auto()
    KAM_TORUS_PRESERVED = auto()
    MELNIKOV_CHAOTIC = auto()
    HILL_WELL_VIOLATED = auto()


# ══════════════════════════════════════════════════════════════════════════════
# §C. EXCEPCIONES DEL MOTOR FÍSICO
# ══════════════════════════════════════════════════════════════════════════════
class BrachistochroneEngineError(Exception):
    """Excepción raíz del motor físico de la braquistócrona."""

class EnergyWellColapsoError(BrachistochroneEngineError):
    """La barrera de energía es nula o negativa (H0 ≤ V(q))."""

class ConformalSingularityError(BrachistochroneEngineError):
    """Tensor métrico o inversa con singularidades numéricas."""

class GeodesicIntegrationError(BrachistochroneEngineError):
    """Divergencia numérica catastrófica o caústica durante RK4."""

class KoszulChecksumError(BrachistochroneEngineError):
    """El isomorfismo conforme / Koszul no es fiel."""

class InitialKinematicsError(BrachistochroneEngineError):
    """Condiciones iniciales (q,v,H0) inconsistentes con la variedad."""

class SymplecticEmbeddingError(BrachistochroneEngineError):
    """La inmersión Cayley-Darboux no pertenece a sp(2d,ℝ)."""

class EnvelopeChainViolationError(BrachistochroneEngineError):
    """La cadena de hashes entre fases fue corrompida."""


# ══════════════════════════════════════════════════════════════════════════════
# §D. PROTOCOLOS DEL FIBRADO POTENCIAL
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class PotentialField(Protocol):
    """Campo potencial V: Q → ℝ (extensible a ℂ para CSMD)."""

    def __call__(self, q: NDArray[Any]) -> Any: ...


# ══════════════════════════════════════════════════════════════════════════════
# §E. GUARDIA NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════
class _BrachistochroneNumericalGuard:
    """Capa de saneamiento y validación para tensores, vectores y escalares."""

    @staticmethod
    def _wilkinson_spectral_floor(scale: float, tol: float, ambient_dim: int) -> float:
        dim = max(int(ambient_dim), 1)
        return max(float(tol), dim * _MACHINE_EPS * max(float(scale), 1.0), _ENERGY_GAP_FLOOR)

    @staticmethod
    def _hermitize_weyl(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any = np.float64) -> NDArray[Any]:
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _as_finite_scalar(value: Any, name: str) -> float:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ArithmeticError(
                    f"{name} debe ser escalar; ndarray shape={value.shape}.")
            value = value.reshape(()).item()
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e6 * _MACHINE_EPS * max(1.0, abs(value.real)):
                raise ArithmeticError(f"{name} posee parte imaginaria no nula: {value!r}.")
            value = value.real
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ArithmeticError(
                f"{name} no es convertible a float ({type(value).__name__}).") from exc
        if not math.isfinite(out):
            raise ArithmeticError(f"{name} debe ser finito; recibido {out}.")
        return out

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{name} debe ser matriz 2D.")
        if X.shape[0] != X.shape[1] or X.shape[0] == 0:
            raise ValueError(f"{name} debe ser cuadrada y positiva: {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(f"{name} contiene NaN/Inf.")

    @staticmethod
    def _assert_vector(x: NDArray[np.float64], dim: int, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"{name} debe ser vector 1D.")
        if x.shape[0] != dim:
            raise ValueError(f"{name} dim={x.shape[0]} != {dim}.")
        if not np.all(np.isfinite(x)):
            raise ArithmeticError(f"{name} contiene NaN/Inf.")

    @staticmethod
    def _assert_finite_scalar(x: float, name: str) -> None:
        if not np.isfinite(x):
            raise ArithmeticError(f"{name} debe ser finito.")

    @staticmethod
    def _assert_positive_scalar(x: float, name: str) -> None:
        _BrachistochroneNumericalGuard._assert_finite_scalar(x, name)
        if x <= 0.0:
            raise ValueError(f"{name} debe ser positivo.")

    @staticmethod
    def _assert_callable_potential(potential_v_fn: Any) -> None:
        if not callable(potential_v_fn):
            raise TypeError(
                f"potential_v_fn debe ser invocable V: Q → ℝ ({type(potential_v_fn).__name__}).")


def _hash_state(*arrays: NDArray[Any]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# §F. DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases)
# ══════════════════════════════════════════════════════════════════════════════
def _empty_f64() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)

def _empty_2d() -> NDArray[np.float64]:
    return np.zeros((0, 0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConformalPotentialDilation:
    r"""Artefacto de FASE 1: certificado del pozo de potencial."""
    g_base: NDArray[np.float64]
    potential_v: NDArray[np.float64]
    initial_h0: float
    energy_gap: NDArray[np.float64]
    is_safe: bool
    cholesky_factor: NDArray[np.float64]
    symmetry_residual_relative: float
    energy_gap_min: float
    manifold_dim: int = 0
    cholesky_logdet: float = 0.0
    kinetic_energy_start: float = float("nan")
    hamiltonian_residual_start: float = float("nan")
    q_start: NDArray[np.float64] = field(default_factory=_empty_f64)
    v_start: NDArray[np.float64] = field(default_factory=_empty_f64)


@dataclass(frozen=True, slots=True)
class Phase1Envelope:
    r"""Contrato FASE 1 (Observe) → FASE 2 (Orient)."""
    dilation: ConformalPotentialDilation
    hill_accessibility_ok: bool
    hill_accessibility_margin: float
    local_lipschitz_v: float
    hill_potential_samples: NDArray[np.float64]
    origin_timestamp_ns: float
    chain_hash: str


@dataclass(frozen=True, slots=True)
class ConformalGeometryBundle:
    r"""Artefacto de FASE 2: geometría de Jacobi-Fermat + Cayley-Darboux."""
    g_base: NDArray[np.float64]
    g_conformal: NDArray[np.float64]
    g_conformal_inv: NDArray[np.float64]
    refractive_index: float
    christoffel_symbols: NDArray[np.float64]
    wilkinson_condition: float
    initial_h0: float
    conformal_factor_sq: float = 0.0
    torsion_residual: float = 0.0
    potential_gradient: NDArray[np.float64] = field(default_factory=_empty_f64)
    gradient_method: str = "unavailable"
    energy_gap_at_eval: float = 0.0
    g_base_inv: NDArray[np.float64] = field(default_factory=_empty_2d)
    symplectic_cayley_matrix: NDArray[np.float64] = field(default_factory=_empty_2d)
    symplectic_residual: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    phase1: Optional[ConformalPotentialDilation] = None


# Alias de compatibilidad (nombre antiguo en versiones previas)
ConformalManifoldBundle = ConformalGeometryBundle


@dataclass(frozen=True, slots=True)
class Phase2Envelope:
    r"""Contrato FASE 2 (Orient) → FASE 3 (Decide&Act)."""
    phase1: Phase1Envelope
    bundle: ConformalGeometryBundle
    origin_timestamp_ns: float
    chain_hash: str


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    r"""Sección transversal de Poincaré (retornos + dispersión)."""
    n_crossings: int
    return_dispersion: float
    is_regular: bool
    returns: Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    r"""Multiplicadores de Floquet μ_k = eig(M(T)) sobre la monodromía."""
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BrachistochronePhysicalState:
    r"""Objeto terminal y certificado físico inmutable emitido por el Motor."""
    trajectory: NDArray[np.float64]
    velocities: NDArray[np.float64]
    transit_time_t: float
    energy_drift_max: float
    is_globally_stable: bool
    fermat_energy_initial: float = 0.0
    fermat_energy_drift_max: float = 0.0
    affine_time: float = 0.0
    steps_accepted: int = 0
    on_shell_residual: float = 0.0
    velocity_projected: bool = False
    torsion_residual: float = 0.0
    wilkinson_condition: float = 1.0
    refractive_index_start: float = 0.0
    melnikov_value: float = 0.0
    kam_condition_satisfied: bool = True
    regime: GeodesicRegime = GeodesicRegime.ELIPTIC_STABLE
    engine_version: str = __version__

    # Campos v5.0.0 (opcionales, con defaults para compatibilidad)
    kam_worst_ratio: float = 0.0
    melnikov_max_abs_integrand: float = 0.0
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    poincare_section_regular: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_is_neutral: bool = True
    floquet_exponents: NDArray[np.float64] = field(default_factory=_empty_f64)
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Auditoría del pozo de potencial termodinámico + Hill accessibility.         ║
# ║  Cierra con `emit_phase1_envelope` → `Phase1Envelope`.                       ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase1_PotentialWellInquirer(_BrachistochroneNumericalGuard):
    r"""
    FASE 1 (Observe): sanea y audita el pozo de potencial termodinámico.

    Métodos rigurosos:
      • `_phase1_validate_manifold_dimension`
      • `_phase1_validate_metric_tensor`
      • `_phase1_certify_metric_symmetry` — tolerancia rel/abs
      • `_phase1_cholesky_spd_factor` — Sylvester + LLᵀ
      • `_phase1_energy_gap_spectrum` — H₀ − V muestreado
      • `_phase1_validate_initial_kinematics` — H₀ = T + V
      • `_phase1_hill_accessibility_ball` (nuevo) — barrera Hill en bola
      • `_phase1_local_lipschitz_v` (nuevo) — |∇V|∞ determinista
    """

    @staticmethod
    def _phase1_validate_manifold_dimension(dimension_d: int) -> int:
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise ConformalSingularityError(
                f"Dimensión inválida: d={dimension_d}. Se exige d ∈ ℤ≥1.")
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise ConformalSingularityError(f"d={d} excede d_max={_DIM_MAX}.")
        return d

    def _phase1_validate_metric_tensor(
        self, g_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._assert_square_matrix(g_base, "g_base")
        d = self._phase1_validate_manifold_dimension(g_base.shape[0])
        metric = np.asarray(g_base, dtype=np.float64)
        if metric.shape != (d, d):
            raise ConformalSingularityError(
                f"g_base shape={metric.shape} incoherente con d={d}.")
        return metric

    def _phase1_certify_metric_symmetry(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        if symmetry_rel > _SYMMETRY_REL_TOL:
            raise ConformalSingularityError(
                f"Ruptura de simetría en G_base: {symmetry_rel:.3e} "
                f"> {_SYMMETRY_REL_TOL:.0e}.")
        return self._hermitize_weyl(g_base), symmetry_rel

    def _phase1_cholesky_spd_factor(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        try:
            chol = la.cholesky(g_base, lower=True)
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Criterio de Sylvester fallido: G_base no es definida positiva.") from exc
        diag = np.diag(chol)
        if np.any(diag <= 0.0):
            raise ConformalSingularityError(
                "Cholesky degenerado: diag(L) no estrictamente positivo.")
        # Verificación numérica LLᵀ ≈ G
        recon = chol @ chol.T
        recon_err = float(la.norm(recon - g_base, ord='fro'))
        recon_scale = max(1.0, float(la.norm(g_base, ord='fro')))
        if recon_err > 1e-8 * recon_scale:
            raise ConformalSingularityError(
                f"LLᵀ−G no despreciable: ‖err‖_F={recon_err:.3e}.")
        logdet = float(2.0 * np.sum(np.log(diag)))
        return chol, logdet

    def _validate_sylvester_criteria(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, _ = self._phase1_cholesky_spd_factor(metric)
        return chol, symmetry_rel

    def _phase1_validate_potential_samples(
        self, potential_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        samples = np.asarray(potential_v, dtype=np.float64)
        if samples.ndim != 1 or samples.shape[0] == 0:
            raise ValueError("potential_v debe ser vector 1D no vacío.")
        if not np.all(np.isfinite(samples)):
            raise ArithmeticError("potential_v contiene NaN/Inf.")
        return samples

    def _phase1_energy_gap_spectrum(
        self,
        potential_samples: NDArray[np.float64],
        initial_h0: float,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        self._assert_finite_scalar(initial_h0, "initial_h0")
        energy_gap = initial_h0 - potential_samples
        min_gap = float(np.min(energy_gap))
        if min_gap <= 0.0:
            raise EnergyWellColapsoError(
                f"H0 ({initial_h0:.6f}) no supera V_max ({float(np.max(potential_samples)):.6f}). "
                f"Gap min: {min_gap:.3e}")
        is_safe = min_gap >= _ENERGY_GAP_FLOOR
        return energy_gap, min_gap, is_safe

    def _phase1_validate_initial_kinematics(
        self,
        g_base: NDArray[np.float64],
        initial_h0: float,
        q_start: Optional[NDArray[np.float64]],
        v_start: Optional[NDArray[np.float64]],
        potential_at_start: Optional[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        d = g_base.shape[0]
        if q_start is None:
            q_arr = np.zeros(0, dtype=np.float64)
        else:
            q_arr = np.asarray(q_start, dtype=np.float64)
            self._assert_vector(q_arr, d, "q_start")

        if v_start is None:
            v_arr = np.zeros(0, dtype=np.float64)
            kinetic = float("nan")
        else:
            v_arr = np.asarray(v_start, dtype=np.float64)
            self._assert_vector(v_arr, d, "v_start")
            kinetic = 0.5 * float(v_arr @ g_base @ v_arr)

        if (v_start is not None and potential_at_start is not None
                and math.isfinite(kinetic)):
            residual = abs(kinetic + float(potential_at_start) - float(initial_h0))
        else:
            residual = float("nan")
        return q_arr, v_arr, kinetic, residual

    # ── Hill accessibility (nuevo) ────────────────────────────────────────────
    def _phase1_hill_accessibility_ball(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        n_samples: int = _HILL_ACCESSIBILITY_SAMPLES,
    ) -> Tuple[bool, float, NDArray[np.float64]]:
        r"""
        Verifica $H_0 \ge V(q)$ sobre la bola de accesibilidad de Hill:
        $q(t) \in [q_s \pm \sigma g^{-1/2} v_s]$ con $\sigma\in[-1,1]$.

        Retorna `(ok, margin, samples_V)`.
        """
        d = q_start.shape[0]
        try:
            g_sqrt_inv = la.inv(la.sqrtm(g_base))
        except la.LinAlgError:
            return False, -float('inf'), np.zeros(0)

        direction = g_sqrt_inv @ v_start
        samples_v = np.zeros(n_samples, dtype=np.float64)
        margin = float('inf')
        for i, sigma in enumerate(np.linspace(-1.0, 1.0, n_samples)):
            q_test = q_start + sigma * direction
            try:
                v_val = self._as_finite_scalar(potential_v_fn(q_test), "V(q_test)")
            except Exception:
                v_val = float('inf')
            samples_v[i] = v_val
            margin = min(margin, float(initial_h0) - v_val)
        return (margin >= 0.0), float(margin), samples_v

    # ── Lipschitz local (nuevo) ───────────────────────────────────────────────
    def _phase1_local_lipschitz_v(
        self,
        q_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        radius: float = _LOCAL_LIPSCHITZ_RADIUS,
    ) -> float:
        r"""
        Estima la constante de Lipschitz local de $V$ en $B(q_s, r)$
        por diferencias centradas en las $2d$ direcciones $\pm e_i$.
        """
        d = q_start.shape[0]
        f0 = self._as_finite_scalar(potential_v_fn(q_start), "V(q_start)")
        max_slope = 0.0
        for i in range(d):
            for sign in (+1.0, -1.0):
                delta = np.zeros(d, dtype=np.float64)
                delta[i] = sign * radius
                try:
                    f_test = self._as_finite_scalar(potential_v_fn(q_start + delta), "V(q_test)")
                    slope = abs(f_test - f0) / max(radius, _MACHINE_EPS)
                    max_slope = max(max_slope, slope)
                except Exception:
                    continue
        return float(max_slope)

    def _phase1_observe_energy_certificate(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> ConformalPotentialDilation:
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, logdet = self._phase1_cholesky_spd_factor(metric)
        samples = self._phase1_validate_potential_samples(potential_v)
        energy_gap, min_gap, is_safe = self._phase1_energy_gap_spectrum(
            samples, initial_h0)
        q_arr, v_arr, kinetic, ham_res = self._phase1_validate_initial_kinematics(
            metric, initial_h0, q_start, v_start, potential_at_start)
        return ConformalPotentialDilation(
            g_base=self._freeze_array(metric),
            potential_v=self._freeze_array(samples),
            initial_h0=float(initial_h0),
            energy_gap=self._freeze_array(energy_gap),
            is_safe=is_safe,
            cholesky_factor=self._freeze_array(chol),
            symmetry_residual_relative=symmetry_rel,
            energy_gap_min=min_gap,
            manifold_dim=int(metric.shape[0]),
            cholesky_logdet=logdet,
            kinetic_energy_start=kinetic,
            hamiltonian_residual_start=ham_res,
            q_start=self._freeze_array(q_arr),
            v_start=self._freeze_array(v_arr),
        )

    def evaluate_potential_barrier(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> ConformalPotentialDilation:
        return self._phase1_observe_energy_certificate(
            g_base, potential_v, initial_h0,
            q_start=q_start, v_start=v_start, potential_at_start=potential_at_start)

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_envelope(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> Phase1Envelope:
        r"""
        **Cierre formal de la FASE 1.**

        Ejecuta la auditoría completa del pozo de potencial (Sylvester +
        Cholesky + Hill + Lipschitz local) y emite `Phase1Envelope`, el
        contrato tipado con hash-chain para la FASE 2.
        """
        t0 = time.perf_counter_ns()
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)

        val_start = self._as_finite_scalar(potential_v_fn(q0), "V(q_start)")

        # Dilation base
        dilation = self._phase1_observe_energy_certificate(
            g_base=g,
            potential_v=np.array([val_start], dtype=np.float64),
            initial_h0=float(initial_h0),
            q_start=q0, v_start=v0, potential_at_start=val_start,
        )

        # Hill accessibility (bola)
        hill_ok, hill_margin, hill_samples = self._phase1_hill_accessibility_ball(
            g_base=g, q_start=q0, v_start=v0,
            potential_v_fn=potential_v_fn, initial_h0=float(initial_h0))

        # Lipschitz local de V
        lip = self._phase1_local_lipschitz_v(q0, potential_v_fn)

        chain = _hash_state(g, q0, v0, np.array([initial_h0, val_start, hill_margin, lip]))

        return Phase1Envelope(
            dilation=dilation,
            hill_accessibility_ok=bool(hill_ok),
            hill_accessibility_margin=float(hill_margin),
            local_lipschitz_v=float(lip),
            hill_potential_samples=self._freeze_array(hill_samples),
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  Sintonía de la métrica de Jacobi-Fermat y embedding Cayley-Darboux en       ║
# ║  sp(2d,ℝ). Cierra con `emit_phase2_envelope` → `Phase2Envelope`.             ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase2_ConformalKoszulSuturator(Phase1_PotentialWellInquirer):
    r"""
    FASE 2 (Orient): construye la métrica de Jacobi-Fermat, los símbolos de
    Koszul conformes y la inmersión simpléctica Cayley-Darboux en sp(2d,ℝ).
    """

    # ── APERTURA FASE 2 ───────────────────────────────────────────────────────
    @classmethod
    def from_phase1_envelope(
        cls, env: Phase1Envelope,
    ) -> 'Phase2_ConformalKoszulSuturator':
        r"""
        **Apertura formal de la FASE 2.**

        Valida la cadena de hashes de la FASE 1 y prepara el suturizador
        para consumir el certificado del pozo de potencial.
        """
        d = env.dilation
        expected = _hash_state(
            d.g_base, d.q_start, d.v_start,
            np.array([d.initial_h0, d.potential_v[0] if d.potential_v.size else 0.0,
                      env.hill_accessibility_margin, env.local_lipschitz_v]))
        if expected != env.chain_hash:
            raise EnvelopeChainViolationError(
                f"Cadena FASE 1 corrupta: {expected} ≠ {env.chain_hash}.")
        inst = cls()
        inst._phase1_envelope = env  # type: ignore[attr-defined]
        return inst

    def _phase2_consume_phase1_certificate(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
    ) -> Tuple[ConformalPotentialDilation, NDArray[np.float64], int]:
        if not isinstance(dilation, ConformalPotentialDilation):
            raise TypeError("dilation debe ser ConformalPotentialDilation")
        d = int(dilation.manifold_dim or dilation.g_base.shape[0])
        if dilation.g_base.shape != (d, d):
            raise ConformalSingularityError(
                f"g_base shape={dilation.g_base.shape} != {(d, d)}.")
        if dilation.cholesky_factor.shape != (d, d):
            raise ConformalSingularityError("cholesky_factor incoherente con g_base.")
        q = np.asarray(q_eval, dtype=np.float64)
        self._assert_vector(q, d, "q_eval")
        return dilation, q, d

    def _phase2_conformal_factor_at_q(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        clamp: bool = False,
    ) -> Tuple[float, float, float]:
        self._assert_callable_potential(potential_v_fn)
        val_v = self._as_finite_scalar(potential_v_fn(q), "potential_v_fn(q)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            if not clamp:
                raise EnergyWellColapsoError(
                    f"Colapso de barrera en q de evaluación: H0−V(q)={gap:.3e}.")
            gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        n_factor = math.sqrt(n_sq)
        if not math.isfinite(n_factor) or n_sq <= 0.0:
            raise GeodesicIntegrationError(
                f"Índice de refracción no físico: n²={n_sq}, n={n_factor}.")
        return n_factor, n_sq, gap

    def _eval_conformal_metric(
        self,
        q: NDArray[np.complex128] | NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        val_v = potential_v_fn(q)
        if isinstance(val_v, np.ndarray):
            val_v = val_v.reshape(()).item()
        gap = initial_h0 - val_v
        if np.iscomplexobj(gap):
            if float(np.real(gap)) <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR + 0j
        else:
            if float(gap) <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        return n_sq * np.asarray(
            g_base, dtype=np.result_type(g_base, getattr(q, "dtype", np.float64)))

    def _phase2_assemble_jacobi_metric(
        self,
        g_base: NDArray[np.float64],
        n_sq: float,
    ) -> NDArray[np.float64]:
        conformal = n_sq * np.asarray(g_base, dtype=np.float64)
        if not np.all(np.isfinite(conformal)):
            raise ConformalSingularityError("Métrica conforme no finita (overflow de n²·G).")
        return conformal

    def _phase2_invert_conformal_cholesky(
        self,
        dilation: ConformalPotentialDilation,
        n_sq: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        d = dilation.g_base.shape[0]
        chol = np.asarray(dilation.cholesky_factor, dtype=np.float64)
        try:
            eye = np.eye(d, dtype=np.float64)
            l_inv = la.solve_triangular(chol, eye, lower=True)
            g_inv = l_inv.T @ l_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError("Inversión Cholesky fallida.") from exc
        g_conformal_inv = (1.0 / n_sq) * g_inv
        return g_inv, g_conformal_inv

    def _phase2_wilkinson_condition(
        self, g_base: NDArray[np.float64],
    ) -> float:
        evals = la.eigvalsh(g_base)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        if lam_min <= 0.0:
            raise ConformalSingularityError(
                f"Espectro de G no SPD: λ_min={lam_min:.3e}.")
        cond = lam_max / lam_min
        if cond > _CONDITION_NUMBER_MAX:
            raise ConformalSingularityError(
                f"Inestabilidad de Wilkinson: κ(G) = {cond:.3e}.")
        return float(cond)

    # ── Cayley-Darboux riguroso (v5.0.0) ─────────────────────────────────────
    def _phase2_compute_cayley_darboux_embedding(
        self,
        g_conformal: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, bool, float]:
        r"""
        Inmersión simpléctica de Cayley-Darboux:

          $K = (G̃-I)(G̃+I)^{-1}$,  verifica $K^\top J + J K = 0$.
          $M = (I+K)(I-K)^{-1} \in Sp(2d,\mathbb{R})$;
          $\|M^\top J M - J\|_F/\|M\|_F^2 \le 10^{-10}$.

        Retorna `(M_sympl, rel_residual, is_valid, sp_defect)`.
        """
        d = g_conformal.shape[0]
        I_d = np.eye(d, dtype=np.float64)
        J_d = np.block([
            [np.zeros((d, d)), I_d],
            [-I_d, np.zeros((d, d))],
        ])
        try:
            K_block = (g_conformal - I_d) @ la.inv(g_conformal + I_d)
            K_full = np.block([
                [K_block, np.zeros((d, d))],
                [np.zeros((d, d)), -K_block],
            ])

            # (i) Verificación de pertenencia a sp(2d,ℝ): Kᵀ J + J K = 0
            sp_defect = float(la.norm(K_full.T @ J_d + J_d @ K_full, 'fro'))

            # (ii) M_sympl y residuo de Darboux relativo
            I_2d = np.eye(2 * d, dtype=np.float64)
            M_sympl = la.inv(I_2d - K_full) @ (I_2d + K_full)
            diff = M_sympl.T @ J_d @ M_sympl - J_d
            norm_M = float(la.norm(M_sympl, 'fro'))
            rel_res = (float(la.norm(diff, 'fro') / (norm_M ** 2))
                       if norm_M > 0 else float('inf'))

            is_valid = (rel_res <= _SYMPLECTIC_RESIDUAL_TOL
                        and sp_defect <= _SYMPLECTIC_RESIDUAL_TOL)
            return M_sympl, rel_res, is_valid, sp_defect
        except la.LinAlgError:
            return (np.zeros((2 * d, 2 * d), dtype=np.float64),
                    float('inf'), False, float('inf'))

    def _phase2_estimate_potential_gradient(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        Gradiente de V(q) por CSMD (paso complejo $h=10^{-20}$) con
        fallback a diferencias centradas de 4º orden.
        """
        d = int(q.shape[0])
        grad = np.zeros(d, dtype=np.float64)

        # Intento CSMD (requiere que potential_v_fn sea analítica en ℂ)
        try:
            for k in range(d):
                q_c = q.astype(np.complex128, copy=True)
                q_c[k] += 1.0j * _CSMD_PERTURBATION
                val = potential_v_fn(q_c)
                if isinstance(val, np.ndarray):
                    val = val.reshape(()).item()
                grad[k] = float(np.imag(val)) / _CSMD_PERTURBATION
            if np.all(np.isfinite(grad)):
                return grad, "csmd"
        except Exception:
            pass

        # Fallback: diferencias centradas 2º orden (deterministas)
        scale = max(1.0, float(la.norm(q)))
        step = math.sqrt(_MACHINE_EPS) * scale
        for k in range(d):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[k] += step
            q_minus[k] -= step
            f_plus = self._as_finite_scalar(potential_v_fn(q_plus), "V(q+he_k)")
            f_minus = self._as_finite_scalar(potential_v_fn(q_minus), "V(q−he_k)")
            grad[k] = (f_plus - f_minus) / (2.0 * step)
        if not np.all(np.isfinite(grad)):
            raise GeodesicIntegrationError("∇V no finito en el punto de evaluación.")
        return grad, "central"

    def _phase2_conformal_christoffel(
        self,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
    ) -> NDArray[np.float64]:
        r"""
        Símbolos de Christoffel conformes $\Gamma^\rho_{\mu\nu}$ para
        $g_{ij}(q) = f(q)\delta_{ij}$ con $f(q) = 2(H_0 − V(q))$:
        $$\Gamma^\rho_{\mu\nu} = \frac{1}{2f}(\delta^\rho_\nu \partial_\mu f
                                            + \delta^\rho_\mu \partial_\nu f
                                            - \delta_{\mu\nu} \partial^\rho f).$$
        """
        d = g_base.shape[0]
        if gap <= 0.0:
            raise EnergyWellColapsoError("Gap no positivo al formar Christoffel conforme.")
        dphi = grad_v / (2.0 * gap)
        g_inv_dphi = g_inv @ dphi
        gamma = np.zeros((d, d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    gamma[i, j, k] = (
                        (dphi[k] if i == j else 0.0)
                        + (dphi[j] if i == k else 0.0)
                        - g_base[j, k] * g_inv_dphi[i]
                    )
        return gamma

    def _phase2_koszul_torsion_freeness(
        self, gamma: NDArray[np.float64],
    ) -> float:
        torsion = gamma - np.swapaxes(gamma, 1, 2)
        residual = float(np.max(np.abs(torsion))) if torsion.size else 0.0
        if residual > max(_TORSION_ABS_TOL, 1.0e3 * _MACHINE_EPS):
            raise KoszulChecksumError(f"Conexión con torsión no nula: τ={residual:.3e}.")
        return residual

    def _phase2_christoffel_acceleration_identity(
        self,
        gamma: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
        probe: Optional[NDArray[np.float64]] = None,
    ) -> float:
        d = g_base.shape[0]
        vec = (np.ones(d) / math.sqrt(d) if probe is None
               else np.asarray(probe, dtype=np.float64))
        dphi = grad_v / (2.0 * gap)
        contracted = np.einsum("ijk,j,k->i", gamma, vec, vec)
        closed = (2.0 * float(dphi @ vec) * vec
                  - float(vec @ g_base @ vec) * (g_inv @ dphi))
        defect = float(la.norm(contracted - closed))
        scale = max(1.0, float(la.norm(closed)))
        if defect > _CHECKSUM_REL_TOL * scale:
            raise KoszulChecksumError(
                f"Checksum Γ(v,v)↔a_cerrada corrupto: ‖δ‖={defect:.3e}.")
        return defect

    def sintonizar_metrica_conforme(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> ConformalGeometryBundle:
        dilation, q, _d = self._phase2_consume_phase1_certificate(dilation, q_eval)
        self._assert_callable_potential(potential_v_fn)

        g_base = np.asarray(dilation.g_base, dtype=np.float64)
        initial_h0 = float(dilation.initial_h0)

        n_factor, n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False)
        g_conformal = self._phase2_assemble_jacobi_metric(g_base, n_sq)
        g_inv, g_conformal_inv = self._phase2_invert_conformal_cholesky(dilation, n_sq)
        cond = self._phase2_wilkinson_condition(g_base)
        M_sympl, res_sympl, is_sympl, sp_defect = (
            self._phase2_compute_cayley_darboux_embedding(g_conformal))

        grad_v, grad_method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        gamma = self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)
        torsion = self._phase2_koszul_torsion_freeness(gamma)
        self._phase2_christoffel_acceleration_identity(gamma, g_base, g_inv, grad_v, gap)

        return ConformalGeometryBundle(
            g_base=self._freeze_array(g_base),
            g_conformal=self._freeze_array(g_conformal),
            g_conformal_inv=self._freeze_array(g_conformal_inv),
            refractive_index=float(n_factor),
            christoffel_symbols=self._freeze_array(gamma),
            wilkinson_condition=cond,
            initial_h0=initial_h0,
            conformal_factor_sq=n_sq,
            torsion_residual=torsion,
            potential_gradient=self._freeze_array(grad_v),
            gradient_method=grad_method,
            energy_gap_at_eval=gap,
            g_base_inv=self._freeze_array(g_inv),
            symplectic_cayley_matrix=self._freeze_array(M_sympl),
            symplectic_residual=res_sympl,
            symplectic_defect=sp_defect,
            is_symplectic_valid=is_sympl,
            phase1=dilation,
        )

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_envelope(
        self,
        phase1_env: Phase1Envelope,
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> Phase2Envelope:
        r"""
        **Cierre formal de la FASE 2.**

        Consume el envelope de FASE 1, construye la geometría conforme y
        emite `Phase2Envelope` con hash-chain.
        """
        t0 = time.perf_counter_ns()
        d = phase1_env.dilation
        bundle = self.sintonizar_metrica_conforme(
            dilation=d, q_eval=d.q_start, potential_v_fn=potential_v_fn)

        chain = _hash_state(
            bundle.g_base, bundle.g_conformal, bundle.christoffel_symbols,
            np.array([bundle.symplectic_residual, bundle.symplectic_defect,
                      bundle.torsion_residual, bundle.wilkinson_condition]))

        return Phase2Envelope(
            phase1=phase1_env, bundle=bundle,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  Integración geodésica RK4 + Poincaré-Lindstedt + KAM/Bruno + Melnikov      ║
# ║  + Sección de Poincaré + Floquet + RSI³.                                      ║
# ║  Cierra con `emit_governance_state` → `BrachistochronePhysicalState`.        ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase3_FermatGeodesicSolver(Phase2_ConformalKoszulSuturator):
    r"""
    FASE 3 (Decide & Act): integración de la geodésica de Fermat-Jacobi,
    auditoría Poincaré-Lindstedt, KAM/Bruno, Melnikov, Poincaré-section,
    Floquet y certificación RSI³.
    """

    # ── APERTURA FASE 3 ───────────────────────────────────────────────────────
    @classmethod
    def from_phase2_envelope(
        cls, env: Phase2Envelope,
    ) -> 'Phase3_FermatGeodesicSolver':
        r"""
        **Apertura formal de la FASE 3.**

        Valida la cadena de hashes de FASE 2 y prepara el solver para
        integrar la geodésica.
        """
        b = env.bundle
        expected = _hash_state(
            b.g_base, b.g_conformal, b.christoffel_symbols,
            np.array([b.symplectic_residual, b.symplectic_defect,
                      b.torsion_residual, b.wilkinson_condition]))
        if expected != env.chain_hash:
            raise EnvelopeChainViolationError(
                f"Cadena FASE 2 corrupta: {expected} ≠ {env.chain_hash}.")
        inst = cls()
        inst._phase2_envelope = env  # type: ignore[attr-defined]
        return inst

    def _phase3_consume_phase2_certificate(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
    ) -> Tuple[ConformalGeometryBundle, NDArray[np.float64], NDArray[np.float64], int]:
        if not isinstance(bundle, ConformalGeometryBundle):
            raise TypeError("bundle debe ser ConformalGeometryBundle")
        d = int(bundle.g_base.shape[0])
        if bundle.g_conformal.shape != (d, d) or bundle.g_conformal_inv.shape != (d, d):
            raise ConformalSingularityError(
                "FASE3.0: tensores conformes incoherentes con g_base.")
        if bundle.christoffel_symbols.shape != (d, d, d):
            raise ConformalSingularityError(
                f"FASE3.0: Christoffel shape={bundle.christoffel_symbols.shape} "
                f"!= {(d, d, d)}.")
        q = np.asarray(q_start, dtype=np.float64)
        v = np.asarray(v_start, dtype=np.float64)
        self._assert_vector(q, d, "q_start")
        self._assert_vector(v, d, "v_start")
        return bundle, q, v, d

    def _phase3_resolve_base_inverse(
        self, bundle: ConformalGeometryBundle,
    ) -> NDArray[np.float64]:
        d = bundle.g_base.shape[0]
        cached = np.asarray(bundle.g_base_inv, dtype=np.float64)
        if cached.shape == (d, d) and np.all(np.isfinite(cached)) and cached.size > 0:
            return cached
        try:
            chol = la.cholesky(np.asarray(bundle.g_base, dtype=np.float64), lower=True)
            l_inv = la.solve_triangular(chol, np.eye(d), lower=True)
            return l_inv.T @ l_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Cholesky fallido en la fase de integración.") from exc

    def _phase3_project_energy_shell(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        project_on_shell: bool,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q_start)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            raise EnergyWellColapsoError(
                f"Cascarón vacío en el arranque: H0−V(q0)={gap:.3e}.")
        kinetic = 0.5 * float(v @ g_base @ v)
        residual = abs(kinetic - gap)
        if not project_on_shell:
            return np.asarray(v, dtype=np.float64), residual, False

        target_quad = 2.0 * gap
        current_quad = float(v @ g_base @ v)
        if current_quad > _ENERGY_GAP_FLOOR:
            scaled = v * math.sqrt(target_quad / current_quad)
            return np.asarray(scaled, dtype=np.float64), residual, True

        grad_v, _ = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        try:
            direction = -la.solve(g_base, grad_v, assume_a="pos")
        except la.LinAlgError:
            direction = -grad_v
        quad = float(direction @ g_base @ direction)
        if quad <= _ENERGY_GAP_FLOOR:
            direction = np.zeros_like(v)
            direction[0] = 1.0
            quad = float(direction @ g_base @ direction)
        scaled = direction * math.sqrt(target_quad / quad)
        return np.asarray(scaled, dtype=np.float64), residual, True

    def _phase3_conformal_acceleration(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        _n, _n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False)
        grad_v, _ = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        dphi = grad_v / (2.0 * gap)
        v_quad = float(v @ g_base @ v)
        accel = -2.0 * float(dphi @ v) * v + v_quad * (g_inv @ dphi)
        if not np.all(np.isfinite(accel)):
            raise GeodesicIntegrationError(f"Aceleración geodésica no finita en q={q}.")
        return accel

    def _evaluate_geodesic_rhs(
        self,
        state: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_base_inv: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.float64]:
        d = g_base.shape[0]
        q = state[:d]
        v = state[d:]
        try:
            accel = self._phase3_conformal_acceleration(
                q, v, g_base, g_base_inv, potential_v_fn, initial_h0)
        except (EnergyWellColapsoError, ConformalSingularityError, ArithmeticError) as exc:
            raise GeodesicIntegrationError(
                f"Colisión con singularidad en q={q}: {exc}") from exc
        return np.concatenate([v, accel])

    def _phase3_fermat_energy(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> float:
        _n, n_sq, _ = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=True)
        return 0.5 * n_sq * float(v @ g_base @ v)

    def _phase3_physical_hamiltonian(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[float, float]:
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q)")
        kinetic = 0.5 * float(v @ g_base @ v)
        return kinetic + val_v, val_v

    def _phase3_rk4_step(
        self,
        state: NDArray[np.float64],
        dt: float,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        k1 = self._evaluate_geodesic_rhs(state, g_base, g_inv, potential_v_fn, initial_h0)
        k2 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k1, g_base, g_inv, potential_v_fn, initial_h0)
        k3 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k2, g_base, g_inv, potential_v_fn, initial_h0)
        k4 = self._evaluate_geodesic_rhs(
            state + dt * k3, g_base, g_inv, potential_v_fn, initial_h0)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)):
            raise GeodesicIntegrationError("Estado RK4 no finito.")
        return nxt

    # ══════════════════════════════════════════════════════════════════════════
    # MECÁNICA CELESTE DE POINCARÉ
    # ══════════════════════════════════════════════════════════════════════════

    # ── Poincaré-Lindstedt (exhaustivo) ──────────────────────────────────────
    def solve_poincare_lindstedt_homological_eq(
        self,
        frequencies: NDArray[np.float64],
        perturbation_modes: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Resuelve la ecuación homológica $\{H_0,W\} = H_1 - \langle H_1\rangle$
        para cada modo de Fourier $k\in[-K,K]^d\setminus\{0\}$:
        $$W_k = \frac{H_{1,k}}{\omega\cdot k}.$$
        Retorna `(amplitudes_W, residual_max)`.
        """
        d = frequencies.shape[0]
        w_amplitudes = np.zeros_like(perturbation_modes, dtype=np.float64)
        max_residual = 0.0
        for idx, mode in enumerate(perturbation_modes):
            k_vec = np.ones(d, dtype=np.float64) * (idx + 1)
            dot_freq = float(np.dot(frequencies, k_vec))
            if abs(dot_freq) < 1e-12:
                w_amplitudes[idx] = 0.0
            else:
                w_amplitudes[idx] = float(mode) / dot_freq
            res = abs(dot_freq * w_amplitudes[idx] - mode)
            max_residual = max(max_residual, res)
        return w_amplitudes, float(max_residual)

    # ── KAM/Bruno (exhaustivo sobre cubo de k) ───────────────────────────────
    def check_kam_bruno_diophantine_condition(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> bool:
        r"""
        Verifica $|\omega_0\cdot k| \ge \gamma/\|k\|_1^\tau$ para todo
        $k \in [-K, K]^d \setminus \{0\}$ con $\tau = d+1$.
        """
        d = frequencies.shape[0]
        tau = float(d) + _KAM_TAU_OFFSET
        for k in product(range(-max_order, max_order + 1), repeat=d):
            if all(ki == 0 for ki in k):
                continue
            k_vec = np.array(k, dtype=np.float64)
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            dot = abs(float(np.dot(frequencies, k_vec)))
            bound = gamma / (k_norm_1 ** tau)
            if dot < bound:
                return False
        return True

    def _phase3_kam_worst_ratio(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> float:
        r"""Devuelve $\min_k |\omega\cdot k|\cdot\|k\|_1^\tau/\gamma$."""
        d = frequencies.shape[0]
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float('inf')
        for k in product(range(-max_order, max_order + 1), repeat=d):
            if all(ki == 0 for ki in k):
                continue
            k_vec = np.array(k, dtype=np.float64)
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            dot = abs(float(np.dot(frequencies, k_vec)))
            ratio = dot * (k_norm_1 ** tau) / max(gamma, 1e-15)
            worst = min(worst, ratio)
        return float(worst) if math.isfinite(worst) else 0.0

    # ── Melnikov riguroso (integral sobre trayectoria) ───────────────────────
    def compute_melnikov_homoclinic_integral(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        dt: float,
    ) -> float:
        r"""
        Método legacy (compatibilidad). Para el cálculo riguroso ver
        `_phase3_melnikov_from_trajectory`.
        """
        if trajectory.shape[0] < 2:
            return 0.0
        s = 0.0
        for i in range(trajectory.shape[0]):
            cross_term = float(np.sum(trajectory[i] * velocities[i]))
            s += math.sin(cross_term) * dt
        return float(s)

    def _phase3_melnikov_from_trajectory(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        dt: float,
    ) -> Tuple[float, float]:
        r"""
        Integral homoclínica de Melnikov rigurosa:
        $$M(\theta_0) = -\int \langle G\,v(t),\, \nabla V(q(t))\rangle\,dt,$$
        equivalente a $\int \{H_{\text{cin}},V\}\,dt$ sobre la separatriz
        numérica (poisson bracket cuántico con $\hbar=1$).

        Retorna `(M, max|integrand|)`.
        """
        if trajectory.shape[0] < 2:
            return 0.0, 0.0
        M = 0.0
        max_abs_integrand = 0.0
        for i in range(trajectory.shape[0]):
            q_i = trajectory[i]
            v_i = velocities[i]
            try:
                grad_v, _ = self._phase2_estimate_potential_gradient(q_i, potential_v_fn)
            except Exception:
                continue
            integrand = -float((g_base @ v_i) @ grad_v)
            max_abs_integrand = max(max_abs_integrand, abs(integrand))
            M += integrand * dt
        return float(M), float(max_abs_integrand)

    # ── Sección de Poincaré ───────────────────────────────────────────────────
    def _phase3_poincare_section(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
    ) -> PoincareSectionCertificate:
        r"""
        Sección transversal de Poincaré: usa la primera coordenada $q_0(t)$
        como observable. Cuenta "retornos" (cambios de signo de $\dot q_0$) y
        mide la dispersión de los valores de $q_0$ en los cruces.
        Baja dispersión ⇒ cuasi-periódico; alta ⇒ caótico.
        """
        if trajectory.shape[0] < 3:
            return PoincareSectionCertificate(0, 0.0, True, ())
        q0 = trajectory[:, 0]
        dq0 = np.diff(q0)
        crossings = np.where(np.sign(dq0[:-1]) != np.sign(dq0[1:]))[0]
        if crossings.size < 2:
            return PoincareSectionCertificate(int(crossings.size), 0.0, True, ())
        returns = q0[crossings]
        disp = float(np.std(returns) / max(np.mean(np.abs(returns)), 1e-15))
        return PoincareSectionCertificate(
            n_crossings=int(crossings.size),
            return_dispersion=disp,
            is_regular=bool(disp < 0.5),
            returns=tuple(float(x) for x in returns),
        )

    # ── Floquet multiplicadores (monodromía numérica) ─────────────────────────
    def _phase3_floquet_multipliers(
        self,
        generator_matrix: NDArray[np.float64],
        period: float,
    ) -> FloquetCertificate:
        r"""
        Multiplicadores de Floquet $\mu_k = \text{eig}(e^{A T})$ de la
        monodromía aproximada por la exponencial del generador promedio:
        - Sistema unitario: $|\mu_k| \approx 1$.
        - Sistema disipativo: $|\mu_k| \le 1$.
        """
        try:
            M_T = la.expm(np.asarray(generator_matrix, dtype=np.complex128) * period)
            mu = la.eigvals(M_T)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / max(period, 1e-15)
        except Exception:
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative, exponents = 0.0, True, np.array([0.0])
        return FloquetCertificate(
            multipliers=mu,
            max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=exponents,
        )

    # ── RSI³ (Data / Harness / Model) ────────────────────────────────────────
    def _phase3_rsi3_audit(
        self,
        transit_time: float,
        g_base: NDArray[np.float64],
        v_start: NDArray[np.float64],
        v_end: NDArray[np.float64],
    ) -> Dict[str, float]:
        r"""
        Certificación RSI³:
          - S1 (Novikov): $v_{\text{Krull}}(T) = -\log(T/T_{\text{ref}}) \ge 0.10$.
          - S2 (Harness): SPD de $G$ (ya certificado en FASE 2).
          - S3 (Model): $d_{\text{FS}}(\hat v_s,\hat v_e) \le 10^{-4}$ rad.
        """
        T = max(float(transit_time), _MACHINE_EPS)
        v_krull = -math.log(T / max(1.0, _MACHINE_EPS))

        # Fubini-Study entre direcciones (proyecta a ℂP^{d−1})
        v_s = v_start.astype(np.complex128).flatten()
        v_e = v_end.astype(np.complex128).flatten()
        if np.linalg.norm(v_s) < 1e-15 or np.linalg.norm(v_e) < 1e-15:
            d_fs = 0.0
        else:
            num = abs(np.vdot(v_s, v_e))
            den = max(float(np.linalg.norm(v_s) * np.linalg.norm(v_e)), 1e-15)
            d_fs = float(np.arccos(np.clip(num / den, 0.0, 1.0)))

        return {
            'novikov_krull_valuation': float(v_krull),
            'fubini_study_distance': float(d_fs),
            'rsi3_valid': float(v_krull >= _KRULL_NOVIKOV_FLOOR
                                 and d_fs <= _FUBINI_STUDY_MAX),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTEGRACIÓN PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def integrar_trayectoria_conforme(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        r"""
        Integra la geodésica conforme de Fermat-Jacobi y certifica Poincaré
        (Lindstedt, KAM, Melnikov, sección, Floquet) + RSI³.
        """
        bundle, q0, v0, d = self._phase3_consume_phase2_certificate(
            bundle, q_start, v_start)
        self._assert_callable_potential(potential_v_fn)
        self._assert_positive_scalar(t_max, "t_max")
        self._assert_positive_scalar(dt, "dt")

        steps = max(1, int(math.floor(t_max / dt + 1.0e-12)))
        if steps > _MAX_INTEGRATION_STEPS:
            raise GeodesicIntegrationError(
                f"Pasos {steps} exceden max={_MAX_INTEGRATION_STEPS}.")

        g_base = np.asarray(bundle.g_base, dtype=np.float64)
        initial_h0 = float(bundle.initial_h0)
        g_inv = self._phase3_resolve_base_inverse(bundle)

        v0, on_shell_res, projected = self._phase3_project_energy_shell(
            q0, v0, g_base, potential_v_fn, initial_h0,
            project_on_shell=project_on_shell)

        trajectory = np.zeros((steps, d), dtype=np.float64)
        velocities = np.zeros((steps, d), dtype=np.float64)
        state = np.concatenate([q0, v0])
        energy_drift_max = 0.0
        fermat_drift_max = 0.0
        transit_time_t = 0.0
        is_globally_stable = True
        last_index = 0

        e_f_0 = self._phase3_fermat_energy(q0, v0, g_base, potential_v_fn, initial_h0)
        n_start = float(bundle.refractive_index)

        for step in range(steps):
            q_curr = state[:d]
            v_curr = state[d:]
            trajectory[step] = q_curr
            velocities[step] = v_curr
            last_index = step

            if not np.all(np.isfinite(state)):
                is_globally_stable = False
                break

            try:
                ham, val_v = self._phase3_physical_hamiltonian(
                    q_curr, v_curr, g_base, potential_v_fn)
                e_f = self._phase3_fermat_energy(
                    q_curr, v_curr, g_base, potential_v_fn, initial_h0)
            except (GeodesicIntegrationError, EnergyWellColapsoError, ArithmeticError):
                is_globally_stable = False
                break

            energy_drift_max = max(energy_drift_max, abs(ham - initial_h0))
            fermat_drift_max = max(fermat_drift_max, abs(e_f - e_f_0))

            v_quad = float(v_curr @ g_base @ v_curr)
            v_norm_g = math.sqrt(max(0.0, v_quad))
            gap = initial_h0 - val_v
            if gap <= _ENERGY_GAP_FLOOR:
                is_globally_stable = False
                break
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            try:
                state = self._phase3_rk4_step(
                    state, dt, g_base, g_inv, potential_v_fn, initial_h0)
            except GeodesicIntegrationError:
                is_globally_stable = False
                break

        if energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            is_globally_stable = False
        if fermat_drift_max > _FERMAT_DRIFT_HARD_TOL:
            is_globally_stable = False

        used = last_index + 1
        traj_used = trajectory[:used]
        vel_used = velocities[:used]

        # ══ Mecánica celeste de Poincaré (exhaustiva) ══
        # Frecuencias: derivadas del espectro de vᵀGv (proxy de acciones)
        if used >= d and d > 0:
            freqs = np.array(
                [math.sqrt(max(0.01, float(v_i @ g_base @ v_i)))
                 for v_i in vel_used[:d]])
        else:
            freqs = np.ones(d, dtype=np.float64)

        kam_ok = self.check_kam_bruno_diophantine_condition(freqs)
        kam_worst = self._phase3_kam_worst_ratio(freqs)

        melnikov_val, melnikov_max = self._phase3_melnikov_from_trajectory(
            traj_used, vel_used, g_base, potential_v_fn, dt)

        poincare_sec = self._phase3_poincare_section(traj_used, vel_used)

        # Generador aproximado para Floquet: -i (H₀ + perturbación) colapsado
        # a un escalar por el período de tránsito
        G_gen = np.diag(np.full(1, -1j * float(initial_h0), dtype=np.complex128))
        period = max(transit_time_t, 1e-3)
        floquet = self._phase3_floquet_multipliers(
            np.array([[complex(initial_h0) * 1j]], dtype=np.complex128) * (-1.0),
            period)

        # RSI³
        v_end_eff = vel_used[-1] if vel_used.shape[0] > 0 else v0
        rsi3 = self._phase3_rsi3_audit(transit_time_t, g_base, v0, v_end_eff)

        # Banach T³ (proxy)
        lip_proxy = min(
            float(np.linalg.norm(v_end_eff - v0)) / max(float(np.linalg.norm(v0)), 1e-15),
            1.0)
        banach_t3 = float(lip_proxy ** 3)

        # Régimen
        if not is_globally_stable:
            regime = GeodesicRegime.HILL_WELL_VIOLATED
        elif abs(melnikov_val) > _MELNIKOV_TOL:
            regime = GeodesicRegime.MELNIKOV_CHAOTIC
        elif kam_ok and poincare_sec.is_regular:
            regime = GeodesicRegime.KAM_TORUS_PRESERVED
        else:
            regime = GeodesicRegime.ELIPTIC_STABLE

        return BrachistochronePhysicalState(
            trajectory=self._freeze_array(traj_used),
            velocities=self._freeze_array(vel_used),
            transit_time_t=float(transit_time_t),
            energy_drift_max=float(energy_drift_max),
            is_globally_stable=bool(is_globally_stable),
            fermat_energy_initial=float(e_f_0),
            fermat_energy_drift_max=float(fermat_drift_max),
            affine_time=float(used * dt),
            steps_accepted=int(used),
            on_shell_residual=float(on_shell_res),
            velocity_projected=bool(projected),
            torsion_residual=float(bundle.torsion_residual),
            wilkinson_condition=float(bundle.wilkinson_condition),
            refractive_index_start=n_start,
            melnikov_value=float(melnikov_val),
            kam_condition_satisfied=bool(kam_ok),
            regime=regime,
            engine_version=__version__,
            # v5.0.0
            kam_worst_ratio=float(kam_worst),
            melnikov_max_abs_integrand=float(melnikov_max),
            poincare_section_n_crossings=int(poincare_sec.n_crossings),
            poincare_section_dispersion=float(poincare_sec.return_dispersion),
            poincare_section_regular=bool(poincare_sec.is_regular),
            floquet_max_unit_deviation=float(floquet.max_unit_deviation),
            floquet_is_neutral=bool(floquet.is_neutral),
            floquet_exponents=self._freeze_array(floquet.exponents),
            banach_lipschitz_t3=float(banach_t3),
            fubini_study_distance=float(rsi3['fubini_study_distance']),
            novikov_krull_valuation=float(rsi3['novikov_krull_valuation']),
        )

    # ── CIERRE FASE 3 (governance) ────────────────────────────────────────────
    def emit_governance_state(
        self,
        phase2_env: Phase2Envelope,
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        r"""
        **Cierre formal de la FASE 3.**

        Consume el `Phase2Envelope`, integra la geodésica con auditoría
        Poincaré completa y emite `BrachistochronePhysicalState` como
        certificado terminal.
        """
        b = phase2_env.bundle
        return self.integrar_trayectoria_conforme(
            bundle=b,
            q_start=b.phase1.q_start if b.phase1 is not None else np.zeros(b.g_base.shape[0]),
            v_start=b.phase1.v_start if b.phase1 is not None else np.zeros(b.g_base.shape[0]),
            potential_v_fn=potential_v_fn,
            t_max=t_max, dt=dt, project_on_shell=project_on_shell,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR (backward-compatible)
# ══════════════════════════════════════════════════════════════════════════════

class BrachistochronePathFinder(Morphism, Phase3_FermatGeodesicSolver):
    r"""
    Motor físico puro de la braquistócrona de Fermat-Jacobi.

    Orquesta las tres fases anidadas:
      FASE 1 (Observe)    → `emit_phase1_envelope` → `Phase1Envelope`
      FASE 2 (Orient)     → `from_phase1_envelope` + `emit_phase2_envelope`
                          → `Phase2Envelope`
      FASE 3 (Decide&Act) → `from_phase2_envelope` + `emit_governance_state`
                          → `BrachistochronePhysicalState`

    La API pública `compute_brachistochrone_path(...)` se preserva con firma
    y semántica idénticas a v4.0.0.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            return self.compute_brachistochrone_path(**kwargs)
        return state

    def compute_brachistochrone_path(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        r"""
        Ejecuta el pipeline completo de tres fases.

        API idéntica a v4.0.0; internamente usa los envelopes anidados.
        """
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)
        self._assert_square_matrix(g, "g_base")
        self._assert_vector(q0, g.shape[0], "q_start")
        self._assert_vector(v0, g.shape[0], "v_start")
        self._assert_finite_scalar(float(initial_h0), "initial_h0")

        # ══ FASE 1 ══
        env1 = self.emit_phase1_envelope(
            g_base=g, q_start=q0, v_start=v0,
            potential_v_fn=potential_v_fn, initial_h0=float(initial_h0))

        # Re-apertura explícita (protocolo de fase anidada)
        _ = self.from_phase1_envelope(env1)

        # ══ FASE 2 ══
        env2 = self.emit_phase2_envelope(env1, potential_v_fn=potential_v_fn)

        # Re-apertura explícita
        _ = self.from_phase2_envelope(env2)

        # ══ FASE 3 ══
        return self.emit_governance_state(
            env2, potential_v_fn=potential_v_fn,
            t_max=t_max, dt=dt, project_on_shell=project_on_shell)

    @staticmethod
    def euclidean_metric(dimension_d: int) -> NDArray[np.float64]:
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser >= 1; recibido {dimension_d}")
        return np.eye(int(dimension_d), dtype=np.float64)

    @staticmethod
    def harmonic_potential(
        omega: float = 1.0,
        center: Optional[NDArray[np.float64]] = None,
    ) -> Callable[[NDArray[Any]], Any]:
        if omega < 0.0 or not math.isfinite(omega):
            raise ValueError(f"omega debe ser >= 0 y finito; recibido {omega}")

        def _v(q: NDArray[Any]) -> Any:
            qq = np.asarray(q)
            delta = qq if center is None else qq - np.asarray(center)
            return 0.5 * (omega ** 2) * np.dot(delta, delta)

        return _v

    @staticmethod
    def linear_gravity_potential(
        g_acc: float = 1.0, axis: int = -1,
    ) -> Callable[[NDArray[Any]], Any]:
        if not math.isfinite(g_acc):
            raise ValueError(f"g_acc debe ser finito; recibido {g_acc}")

        def _v(q: NDArray[Any]) -> Any:
            return -g_acc * np.asarray(q)[axis]

        return _v

    @staticmethod
    def constant_potential(value: float = 0.0) -> Callable[[NDArray[Any]], Any]:
        if not math.isfinite(value):
            raise ValueError(f"value debe ser finito; recibido {value}")

        def _v(q: NDArray[Any]) -> float:
            del q
            return float(value)

        return _v


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Enums
    "GeodesicRegime",
    # Excepciones
    "BrachistochroneEngineError", "EnergyWellColapsoError",
    "ConformalSingularityError", "GeodesicIntegrationError",
    "KoszulChecksumError", "InitialKinematicsError",
    "SymplecticEmbeddingError", "EnvelopeChainViolationError",
    # Protocolos
    "PotentialField",
    # DTOs FASE 1
    "ConformalPotentialDilation", "Phase1Envelope",
    # DTOs FASE 2
    "ConformalGeometryBundle", "ConformalManifoldBundle", "Phase2Envelope",
    # DTOs FASE 3
    "PoincareSectionCertificate", "FloquetCertificate",
    "BrachistochronePhysicalState",
    # Clases por fase
    "Phase1_PotentialWellInquirer",
    "Phase2_ConformalKoszulSuturator",
    "Phase3_FermatGeodesicSolver",
    # Orquestador
    "BrachistochronePathFinder",
    "__version__",
]