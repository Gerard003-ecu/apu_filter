# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Eikonal Agent — Soberano de Fase de Fresnel, WKB y Monodromía Óptica        ║
║ Ruta   : app/agents/omega/eikonal_agent.py                                           ║
║ Versión: 4.0.0-WKB-Jacobi-Maslov-Floquet-KAM-Spectraplex-RSI-L3-Bogoliubov-Bridge    ║
║ Estrato: V_Ω / V_WISDOM — Meta-Funtor de Control del Fibrador de Lente Cuántico      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA: TRES FASES ANIDADAS POR CONTINUACIÓN (composición de Kleisli)
──────────────────────────────────────────────────────────────────────────────────────
   Φ₁ : (S_MAC, ρ) ─────────────► A  (ApertureModulationResult)
   Φ₂ : (A, ∇S, n) ─────────────► Σ  (EikonalSurfaceResult)
   Φ₃ : (Σ, γ̇, x) ─────────────► P  (FermatPathResult)
   Z_eikonal = Φ₃ ∘ Φ₂ ∘ Φ₁ : D(ℋ) × ℝ≥0 × ℝⁿ → EikonalPhaseState

   El método terminal de cada fase recibe `continuation: Callable[[DTO], R]` y su cola
   es literalmente `return continuation(DTO)`; la continuación canónica es el primer
   método de la fase siguiente (`_continue_from_phase{k}`). Anidamiento tipológico y
   operacional, no cosmético. Los veredictos se agregan por meet en Ω₃ (funtorialidad
   laxa hacia el retículo de Heyting), de modo que el veredicto compuesto es ⊑ al meet
   de los parciales.

AXIOMAS CON ESTATUS EPISTÉMICO EXPLÍCITO
──────────────────────────────────────────────────────────────────────────────────────
 E1 [TEOREMA] Eikonal = Hamilton–Jacobi de la óptica geométrica. Con convención
    "background" (G = métrica de fondo, n escalar):
        H(x,p) = ½ ( G^{μν}(x) p_μ p_ν − n²(x) ) = 0      (variedad característica)
    Con convención "optical" (G_opt = n²·g ya incorpora el índice):
        H = ½ ( G_opt^{μν} p_μ p_ν − 1 ) = 0.
    NUNCA se mezclan: el doble conteo de n es un error dimensional-lógico.
    El test es BILATERAL: |‖p‖²_{G⁻¹} − target| ≤ slack·target (la eikonal es una
    igualdad, no una desigualdad unilateral).
 E2 [TEOREMA] Transporte WKB de primer orden: 2∇S·∇A + A∇²S = 0 ⇒
        A(s) = A₀ √(J₀/J(s)),  J(s) = det X(s)  (determinante de van Vleck),
    donde (X;P) es el marco lagrangiano de campos de Jacobi con X(0)=0, P(0)=I.
    ∇²S = d/ds log J(s) (divergencia del congruencia de rayos).
 E3 [TEOREMA] Cáustica ⇔ punto conjugado ⇔ det X(s)=0. Índice de Maslov por la forma
    de cruce de Robbin–Salamon: en s₀ con X(s₀)u=0, Γ(u) = ⟨P(s₀)u, X'(s₀)u⟩;
    μ += signatura(Γ|_{ker X}). Verificación: oscilador armónico X=sin s, P=cos s ⇒
    en s=π: Γ=(−1)(−1)=+1 ⇒ μ+=1 ✓. Fase de Keller–Maslov: ψ ↦ ψ·e^{−iπμ/2}.
 E4 [TEOREMA] Índice de Morse (Poincaré–Morse): el índice de la segunda variación de
    la energía a lo largo de una geodésica equals el número de puntos conjugados
    contados con multiplicidad; coincide con el índice de Maslov del camino
    lagrangiano asociado. Puente exacto entre cáusticas ópticas y teoría de Morse.
 E5 [TEOREMA] El flujo de rayos es Hamiltoniano en T*M ⇒ simplectico. La ecuación
    variacional ż = J·D²H(z)·z tiene A = J D²H ∈ sp(2d); el punto medio implícito
    produce M ↦ (I−hA/2)⁻¹(I+hA/2)·M = transformada de Cayley ⇒ M ∈ Sp(2d,ℝ)
    EXACTAMENTE (Cayley aplica sp → Sp). Liouville: det M = +1.
 E6 [TEOREMA] Estabilidad de resonador (Poincaré–Floquet–Greene–Siegmann): para la
    matriz de ronda ABCD ∈ Sp(2,ℝ),
        estable ⇔ |tr M|/2 ≤ 1 ⇔ residuo de Greene R=(2−tr M)/4 ∈ [0,1]
                ⇔ multiplicadores de Floquet en el círculo unidad.
    Fase de Gouy por ronda: ν = arccos(tr M/2)/(2π). Resonancia ν≈p/q ⇒ degeneración
    de modos; ν noble (cola de fracción continua en 1) es la más robusta (MacKay).
 E7 [TEOREMA] Proyección euclídea sobre el espectralplex {ρ⪰0, Trρ=1}: por invariancia
    unitaria la proyección es espectral con proyección simplex de autovalores
    (Duchi et al. 2008). clip+renormalizar NO es óptima.
 E8 [TEOREMA] Simpson compuesto requiere cardinal IMPAR de nodos. Con cardinal par
    N≥4 se compone Simpson sobre los primeros N−3 nodos + Simpson 3/8 sobre los
    últimos 3 intervalos: O(h⁴) global, SIN descartar datos. Adaptativo con
    extrapolación de Richardson: error ≈ |S_izq+S_der−S|/15, valor corregido
    S_izq+S_der+(S_izq+S_der−S)/15.
 E9 [CONVENCIÓN] Ley de apertura l_cutoff = ⌊l_max·exp(−κ S_MAC/Tr ρ²)⌋ es una LEY DE
    CONTROL declarada (no un teorema). Se prueban sus propiedades: monotonía
    decreciente en S_MAC, rango [1,l_max], seguridad de subdesbordamiento.
E10 [CONVENCIÓN] Cota operacional de inversión: el presupuesto de κ se deriva de la
    exactitud requerida, κ ≤ tol_obj/(c·n·eps), no de un 1e12 arbitrario.
E11 [SAFETY] RSI-L3 con ENDURECIMIENTO MONÓTONO: la auto-mejora sólo puede REFINAR
    discretización o APRETAR tolerancias; jamás aflojar slack, κ_max ni umbrales de
    certificado. El núcleo verificador es INMUTABLE (evita el obstáculo löbiano).
E12 [TEOREMA] Validez WKB: ħ_eff‖∇n‖/n² ≪ 1 y ħ_eff|∇²S|/‖∇S‖² ≪ 1. En cáustica
    J→0 ⇒ ∇²S diverge ⇒ WKB falla localmente: se DECLARA y se remite a la asintótica
    uniforme de Airy, no se disimula con un número finito falso.

CONVENCIÓN DE ÍNDICES: p_μ = G_{μν} v^ν (la métrica baja el índice);
    H = ½ vᵀG v = ½ pᵀG⁻¹p; ẋ = ∂H/∂p = G⁻¹p; ṗ = −∂H/∂x = ½ vᵀ(∂_x G)v.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Dependencias del ecosistema (contratos tolerantes, degradación grácil) ───────────
try:
    from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico/algebraico violado."""

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(self, payload: Any = None, label: str = "") -> None:
            self.payload, self.label = payload, label

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name

# G_PHYSICS llega como ndarray (aquí) o como dict (módulo Fock): se NORMALIZA el tipo.
G_PHYSICS: Any = None
for _path in ("app.core.immune_system.metric_tensors",
              "app.core.inmune_system.metric_tensors"):
    try:
        _mod = __import__(_path, fromlist=["G_PHYSICS"])
        G_PHYSICS = getattr(_mod, "G_PHYSICS", None)
        break
    except ImportError:
        continue


def _normalize_metric_source(src: Any) -> NDArray[np.float64]:
    """Normaliza el contrato roto ndarray|dict|None → matriz real cuadrada."""
    if src is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(src, dict):
        for key in ("G", "metric", "metric_tensor", "spatial"):
            if key in src:
                return np.asarray(src[key], dtype=np.float64)
        return np.eye(4, dtype=np.float64)
    return np.asarray(src, dtype=np.float64)


try:
    from app.core.immune_system.musical_isomorphism_engine import MetricSpectralPreconditioner
except ImportError:  # pragma: no cover
    try:
        from app.core.inmune_system.musical_isomorphism_engine import MetricSpectralPreconditioner
    except ImportError:
        class MetricSpectralPreconditioner:  # type: ignore[no-redef]
            @dataclass
            class _PM:
                G_inv: NDArray[np.float64]
                condition_number: float

            def precondition(self, G: NDArray[np.float64]) -> "_PM":
                s = 0.5 * (G + G.T)
                ev = la.eigvalsh(s, check_finite=False)
                return self._PM(la.inv(s), float(ev[-1] / max(ev[0], 1e-300)))

try:
    from app.omega.optical_riemann_lens import OpticalRiemannLensFibrator, RefractedState
except ImportError:  # pragma: no cover
    @dataclass(frozen=True)
    class RefractedState:  # type: ignore[no-redef]
        refracted_logits: NDArray[np.float64]
        kv_compression_ratio: float
        refractive_index: float

    class OpticalRiemannLensFibrator:  # type: ignore[no-redef]
        def __init__(self, G: NDArray[np.float64]) -> None:
            self._G = np.asarray(G, dtype=np.float64)
            self._l_cutoff = 50

        def _compute_fermat_refractive_index(self, sigma: float) -> float:
            return 1.0 + float(np.tanh(0.5 * sigma))

        def refract_attention_logits(self, logits: NDArray[np.float64], sigma: float) -> RefractedState:
            n = self._compute_fermat_refractive_index(sigma)
            return RefractedState(
                refracted_logits=np.asarray(logits, dtype=np.float64) * n,
                kv_compression_ratio=max(0.0, 1.0 - 1.0 / max(self._l_cutoff, 1)),
                refractive_index=n,
            )

try:
    from app.omega.levi_civita_agent import LeviCivitaConnectionAgent, TangentVector
except ImportError:  # pragma: no cover
    @dataclass
    class TangentVector:  # type: ignore[no-redef]
        coordinates: NDArray[np.float64]

    class LeviCivitaConnectionAgent:  # type: ignore[no-redef]
        def __init__(self, G: NDArray[np.float64]) -> None:
            self._G = np.asarray(G, dtype=np.float64)

        def enforce_geodesic_flow(self, v: TangentVector, dt: float) -> TangentVector:
            return TangentVector(coordinates=np.asarray(v.coordinates, dtype=np.float64).copy())

        def geodesic_rhs(self, v_coords: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.zeros_like(np.asarray(v_coords, dtype=np.float64))

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

logger = logging.getLogger("MIC.Omega.EikonalAgent")
R = TypeVar("R")

# ── Constantes de rigor numérico ─────────────────────────────────────────────────────
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY: float = 32.0
_DEFAULT_KAPPA_MAX: float = 1.0e10
_DEFAULT_EIKONAL_SLACK: float = 0.1
_DEFAULT_CAVITY_TOL: float = 1.0e-9
_EPS_NEG: float = 1.0e-12
_SYM_ATOL: float = 1.0e-12
_HERM_ATOL: float = 1.0e-10
_LOG_EXP_FLOOR: float = -745.0          # exp(−745) > 0 en binary64
_HBAR_EFF_DEFAULT: float = 1.0
_SIMPSON_MAX_DEPTH: int = 24
_MIDPOINT_MAX_ITER: int = 64
_MASLOV_SVD_TOL: float = 1e-9

__all__ = [
    "QuantumPurityCollapseError", "EikonalSingularityError", "FermatOpticalDeviationError",
    "DimensionalMismatchError", "MetricSignatureError", "EikonalParameterError",
    "EikonalRefractionError", "HeytingVerdict", "SpectralDensityAudit",
    "ApertureModulationResult", "EikonalSurfaceResult", "JacobiMaslovReport",
    "WKBTransportReport", "CavityFloquetReport", "FermatPathResult", "EikonalPhaseState",
    "EikonalControlInput", "RSIL3Hyperparameters", "RSIL3Report",
    "ApertureModulatorPort", "EikonalResolverPort", "FermatAuditorPort", "EikonalAgent",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §0 · EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════════════
class QuantumPurityCollapseError(TopologicalInvariantError):
    """ρ no física: hermiticidad, PSD, traza o pureza fuera del espectralplex."""


class EikonalSingularityError(TopologicalInvariantError):
    """G⁻¹ no SPD, κ fuera de presupuesto, o variedad característica degenerada."""


class FermatOpticalDeviationError(TopologicalInvariantError):
    """Acción de Fermat no finita o residuo geodésico fuera de tolerancia."""


class DimensionalMismatchError(TopologicalInvariantError):
    """Dimensiones incompatibles entre ρ, G, ∇S, γ̇ o logits."""


class MetricSignatureError(TopologicalInvariantError):
    """G no simétrica o no definida positiva."""


class EikonalParameterError(TopologicalInvariantError):
    """Parámetro escalar de control fuera de rango admisible."""


class EikonalRefractionError(TopologicalInvariantError):
    """|‖∇S‖²_{G⁻¹} − target| > slack·target: frente de onda fuera de la variedad."""


# ═════════════════════════════════════════════════════════════════════════════════════
# §1 · UTILIDADES NUMÉRICAS CERTIFICADAS
# ═════════════════════════════════════════════════════════════════════════════════════
def _fro(a: NDArray) -> float:
    return float(la.norm(np.asarray(a), ord="fro", check_finite=False))


def _rel(diff: NDArray, ref: NDArray) -> float:
    """Residuo relativo de Frobenius (invariante de escala)."""
    return _fro(diff) / max(_fro(ref), 1e-300)


def _as_vec(name: str, v: Any, d: Optional[int] = None) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if d is not None and a.size != d:
        raise DimensionalMismatchError(f"{name}: dim {a.size} ≠ {d}")
    if not np.all(np.isfinite(a)):
        raise EikonalParameterError(f"{name}: entradas no finitas")
    return a


def _darboux(n: int) -> NDArray[np.float64]:
    """Ω = [[0,I],[−I,0]] ∈ ℝ^{2n×2n}; Ω⁻¹ = Ωᵀ = −Ω."""
    e, z = np.eye(n), np.zeros((n, n))
    om = np.block([[z, e], [-e, z]])
    om.flags.writeable = False
    return om


def _project_spectraplex(rho: NDArray) -> Tuple[NDArray, NDArray[np.float64], float]:
    r"""
    E7: proyección de Frobenius ÓPTIMA sobre {ρ⪰0, Trρ=1}. Por invariancia unitaria,
    ρ ↦ V diag(max(λ−θ,0)) V† con θ del algoritmo simplex (Duchi et al. 2008).
    Devuelve (ρ_proy, autovalores_proyectados, θ).
    """
    herm = 0.5 * (np.asarray(rho) + np.asarray(rho).conj().T)
    w, v = la.eigh(herm, check_finite=False)
    lam = np.sort(w.real)[::-1]
    css = np.cumsum(lam) - 1.0
    idx = np.arange(1, lam.size + 1)
    cond = lam - css / idx > 0.0
    k = int(np.max(np.nonzero(cond)[0])) + 1
    theta = float(css[k - 1] / k)
    wp = np.maximum(w.real - theta, 0.0)
    out = (v * wp) @ v.conj().T
    return 0.5 * (out + out.conj().T), wp, theta


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


def _simpson_weights(n: int) -> NDArray[np.float64]:
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def _composite_newton_cotes(f: NDArray[np.float64], h: float) -> Tuple[float, str, int]:
    r"""
    E8: cuadratura cerrada de orden 4 SIN descartar nodos.
      · N impar        → Simpson compuesto (1,4,2,…,4,1)·h/3.
      · N par, N≥4     → Simpson sobre los primeros N−3 nodos + Simpson 3/8 en los
                         últimos 3 intervalos: (3h/8)(f₀+3f₁+3f₂+f₃).
      · N=2            → trapecio (orden 2, declarado).
      · N=1            → NO hay intervalo: 0.0 y regla "insufficient_data"
                         (jamás se inventa f₀·dt).
    """
    n = int(f.size)
    if n == 0:
        return 0.0, "empty", 0
    if n == 1:
        return 0.0, "insufficient_data", 1
    if n == 2:
        return float(0.5 * h * (f[0] + f[1])), "trapezoid", 2
    if n % 2 == 1:
        return float((h / 3.0) * np.dot(_simpson_weights(n), f)), "simpson", n
    m = n - 3
    if m >= 3:
        left = (h / 3.0) * np.dot(_simpson_weights(m), f[:m])
        tail = (3.0 * h / 8.0) * (f[m - 1] + 3.0 * f[m] + 3.0 * f[m + 1] + f[m + 2])
        return float(left + tail), "simpson+3/8", n
    return float((3.0 * h / 8.0) * (f[0] + 3.0 * f[1] + 3.0 * f[2] + f[3])), "simpson3/8", n


# ═════════════════════════════════════════════════════════════════════════════════════
# §2 · DTOs INMUTABLES (objetos del topos; sólo escalares/tuplas donde es posible)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralDensityAudit:
    """Certificado espectral de ρ tras la proyección ÓPTIMA al espectralplex (E7)."""

    purity: float
    von_neumann_entropy: float
    eigenvalues_psd: Tuple[float, ...]
    negative_eigvals_pruned: int
    hermiticity_residual: float
    trace_before: float
    trace_after_pruning: float
    projection_distance: float          # ‖ρ_proy − ρ_h‖_F (coste de la reparación)
    simplex_shift: float
    rank: int
    min_eigenvalue: float
    entropy_bound_log_d: float
    is_physical: bool
    dim: int
    verdict: int


@dataclass(frozen=True, slots=True)
class ApertureModulationResult:
    """A — salida terminal de la Fase 1 y precondición formal de la Fase 2."""

    l_cutoff: int
    attenuation_factor: float
    spectral_certificate: SpectralDensityAudit
    s_mac: float
    kappa_coupling: float
    l_max: int
    law_monotone_certified: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class EikonalSurfaceResult:
    r"""
    Σ — salida terminal de la Fase 2. Test BILATERAL sobre la variedad característica
    H = ½(‖p‖²_{G⁻¹} − target) = 0, con target = n² (background) o 1 (optical).
    """

    phase_norm_sq: float
    phase_norm: float                   # norma REAL (corrige el nombre mentiroso)
    n_refract: float
    n_sq: float
    target: float
    convention: str
    eikonal_residual: float             # |‖p‖² − target|
    eikonal_relative_residual: float
    eikonal_threshold: float
    hamiltonian_constraint: float       # H = ½(‖p‖² − target)
    margin_sound: bool
    upper_bound_ok: bool
    kappa_G: float
    kappa_G_inv: float
    kappa_identity_defect: float        # |κ(G⁻¹)−κ(G)|/κ(G): degradación numérica
    inverse_residual: float
    inverse_tolerance: float
    phase_velocity_ratio: float         # 1/n (superlumínica si n<1: se DECLARA)
    enz_singular: bool
    dim: int
    verdict: int


@dataclass(frozen=True, slots=True)
class JacobiMaslovReport:
    """E2–E5: campos de Jacobi, van Vleck, puntos conjugados e índice de Maslov."""

    maslov_index: int
    conjugate_points: Tuple[float, ...]
    crossing_signatures: Tuple[int, ...]
    van_vleck_det_final: float
    min_abs_det: float
    caustic_detected: bool
    symplectic_residual: float
    determinant_drift: float
    hamiltonian_drift: float
    morse_index: int                    # E4: equals #puntos conjugados con multiplicidad
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class WKBTransportReport:
    """E2/E12: transporte de amplitud, fase de Keller–Maslov y validez semiclásica."""

    amplitude_initial: float
    amplitude_final: float
    amplitude_ratio: float
    maslov_phase: float                 # −π μ/2
    complex_phase_factor: Tuple[float, float]   # e^{−iπμ/2} = (cos, sin)
    laplacian_S: float                  # ∇²S = d/ds log J
    wkb_parameter_gradient: float       # ħ‖∇n‖/n²
    wkb_parameter_amplitude: float      # ħ|∇²S|/‖∇S‖²
    wkb_valid: bool
    caustic_regularization: str         # "none" | "airy_uniform_required"
    verdict: int


@dataclass(frozen=True, slots=True)
class CavityFloquetReport:
    """E6: monodromía de ronda, Floquet–Krein–Greene y KAM sobre la fase de Gouy."""

    trace: float
    greene_residue: float
    stability_parameter: float          # |tr|/2
    stable: bool
    multipliers_abs: Tuple[float, ...]
    krein_signatures: Tuple[int, ...]
    gouy_phase_per_roundtrip: float
    resonance: Tuple[int, int, float]   # (p,q,|ν−p/q|)
    diophantine_gamma_eff: float
    noble_nobility: float
    symplectic_residual: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class FermatPathResult:
    """P — salida terminal de la Fase 3 (ABI v3 preservada + campos rigurosos)."""

    path_velocities: NDArray[np.float64]
    fermat_action: float
    geodesic_deviation: float
    integration_rule: str
    n_refract: float
    n_steps_integrated: int
    # ── extensiones v4 ──
    positions: Optional[NDArray[np.float64]] = None
    action_dimensionless: float = 0.0    # A/ħ_eff (parámetro semiclásico)
    quadrature_error_estimate: float = float("nan")
    quadrature_converged: bool = False
    adaptive_refinements: int = 0
    grin_profile_used: bool = False
    hamiltonian_drift: float = 0.0
    symplectic_residual: float = 0.0
    jacobi: Optional[JacobiMaslovReport] = None
    wkb: Optional[WKBTransportReport] = None
    cavity: Optional[CavityFloquetReport] = None
    verdict: int = 0


@dataclass(frozen=True, slots=True)
class EikonalPhaseState:
    """Estado terminal inmutable del endofuntor Z_eikonal = Φ₃∘Φ₂∘Φ₁."""

    phase_gradient_norm: float          # ABI: contiene ‖∇S‖² (nombre histórico)
    fermat_action_integral: float
    dynamic_l_cutoff: int
    refracted_state: RefractedState
    geodesic_deviation: float = 0.0
    spectral_certificate: Optional[SpectralDensityAudit] = None
    eikonal_surface: Optional[EikonalSurfaceResult] = None
    fermat_path: Optional[FermatPathResult] = None
    aperture: Optional[ApertureModulationResult] = None
    # ── extensiones v4 ──
    phase_gradient_norm_true: float = 0.0
    maslov_index: int = 0
    wkb_valid: bool = False
    heyting_verdict_value: int = 0
    is_coherent: bool = False
    rsi_report: Optional["RSIL3Report"] = None
    bogoliubov_certificate: Optional[Any] = None
    created_ns: int = 0

    @property
    def verdict(self) -> HeytingVerdict:
        return HeytingVerdict(self.heyting_verdict_value)


@dataclass(frozen=True, slots=True)
class EikonalControlInput:
    r"""
    Entrada inmutable. `logits_dim` puede diferir de `dim(G)`: el espacio de
    vocabulario NO es la variedad semántica; se requiere un morfismo de proyección
    explícito (o identificación axiomática cuando las dimensiones coinciden).
    """

    raw_llm_logits: NDArray[np.float64]
    rho_llm: NDArray[np.complex128]
    s_mac_entropy: float
    logistic_stress_norm: float
    phase_gradient: NDArray[np.float64]
    path_velocities: NDArray[np.float64]
    use_geodesic_correction: bool = True
    cavity_tol: float = _DEFAULT_CAVITY_TOL
    # ── extensiones v4 ──
    positions: Optional[NDArray[np.float64]] = None
    refractive_index_profile: Optional[NDArray[np.float64]] = None
    dt: float = 1e-3
    hbar_eff: float = _HBAR_EFF_DEFAULT
    logits_projection: Optional[NDArray[np.float64]] = None
    cavity_roundtrip_matrix: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.s_mac_entropy < 0.0:
            raise EikonalParameterError("s_mac_entropy ≥ 0 requerido")
        if self.cavity_tol < 0.0:
            raise EikonalParameterError("cavity_tol ≥ 0 requerido")
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise EikonalParameterError("dt finito y > 0 requerido")
        if self.hbar_eff <= 0.0:
            raise EikonalParameterError("ħ_eff > 0 requerido")


@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Único payload mutable de RSI-L3. E11: sólo puede ENDURECER, nunca aflojar."""

    dt: float = 1e-3
    eikonal_slack: float = _DEFAULT_EIKONAL_SLACK
    kappa_max: float = _DEFAULT_KAPPA_MAX
    quadrature_tol: float = 1e-10
    trust_radius: float = 0.25
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSIL3Report:
    """Trazabilidad del lazo de nivel 3: ledger, Banach a posteriori, rollback."""

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
# §3 · PROTOCOLOS (puertos del topos)
# ═════════════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ApertureModulatorPort(Protocol):
    def modulate_aperture(self, s_mac: float, rho_llm: NDArray) -> ApertureModulationResult: ...


@runtime_checkable
class EikonalResolverPort(Protocol):
    def resolve_eikonal(self, phase_gradient: NDArray[np.float64], n_refract: float) -> EikonalSurfaceResult: ...


@runtime_checkable
class FermatAuditorPort(Protocol):
    def audit_path(self, path_velocities: NDArray[np.float64], n_refract: float, *,
                   use_geodesic_correction: bool, cavity_tol: float,
                   dt_fermat: float = 1e-3, dt_geo: float = 1.0) -> FermatPathResult: ...


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR CON TRES FASES ANIDADAS                                            ══
# ═════════════════════════════════════════════════════════════════════════════════════
class EikonalAgent(Morphism):
    r"""
    Morfismo Eikonal maestro. Compone por agregación tipada y continuación CPS:
      phase1 (diafragma espectral) → phase2 (variedad característica) →
      phase3 (Fermat, Jacobi/Maslov, WKB, cavidad de Poincaré).
    """

    # ─────────────────────────────────────────────────────────────────────────────────
    # ══ FASE 1 · DIAFRAGMA DINÁMICO CUÁNTICO  (ρ, S_MAC → A)                        ══
    # ─────────────────────────────────────────────────────────────────────────────────
    class Phase1_DynamicApertureModulator:
        r"""
        E9 (ley de control declarada):
            l_cutoff = clamp(⌊l_max·exp(−κ S_MAC / Tr ρ²)⌋, 1, l_max).
        Propiedades PROBADAS: monotonía decreciente en S_MAC, rango [1,l_max],
        cálculo en espacio logarítmico (inmune a subdesbordamiento).
        Antes: proyección ÓPTIMA al espectralplex (E7), no clip+renormalizar.
        """

        def __init__(self, l_max_absolute: int = 50, kappa_coupling: float = 1.0) -> None:
            if l_max_absolute < 1:
                raise EikonalParameterError("l_max_absolute ≥ 1 requerido")
            if kappa_coupling <= 0.0:
                raise EikonalParameterError("kappa_coupling > 0 requerido")
            self._l_max = int(l_max_absolute)
            self._kappa = float(kappa_coupling)

        # ── 1.1 · Auditoría espectral con proyección óptima y physicality real ───────
        def audit_density_matrix(self, rho: NDArray) -> SpectralDensityAudit:
            r"""
            Physicality NO vacua (corrige `purity ≥ 1e-12`):
              λ_min ≥ −tol ∧ |Tr−1| ≤ tol ∧ pureza ∈ [1/d−tol, 1+tol] ∧ S ∈ [0, log d]
              ∧ hermiticidad relativa ≤ _HERM_ATOL.
            S_vN con 0·log0 := 0 por continuidad (enmascarado exacto, sin floor sesgado).
            """
            a = np.asarray(rho)
            if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
                raise DimensionalMismatchError(f"ρ debe ser cuadrada no vacía; hay {a.shape}")
            if not np.all(np.isfinite(a)):
                raise QuantumPurityCollapseError("ρ con entradas no finitas")
            d = int(a.shape[0])
            a = a.astype(np.complex128, copy=False)
            herm_res = _rel(a - a.conj().T, a)
            if herm_res > _HERM_ATOL:
                raise QuantumPurityCollapseError(f"ρ no hermítica: residual={herm_res:.3e}")
            herm = 0.5 * (a + a.conj().T)
            tr_before = float(np.real(np.trace(herm)))
            w_raw = la.eigvalsh(herm, check_finite=False).real
            n_neg = int(np.count_nonzero(w_raw < -_EPS_NEG))
            proj, wp, theta = _project_spectraplex(herm)
            dist = _fro(proj - herm)
            tr_after = float(np.sum(wp))
            purity = float(np.sum(wp ** 2))
            mask = wp > 0.0
            s_vn = float(-np.sum(wp[mask] * np.log(wp[mask]))) if np.any(mask) else 0.0
            rank = int(np.count_nonzero(wp > 1e-14))
            log_d = math.log(d)
            tol = 64.0 * d * _MACHINE_EPS
            physical = bool(
                float(np.min(wp)) >= -tol
                and abs(tr_after - 1.0) <= 1e-9
                and (1.0 / d - tol) <= purity <= (1.0 + tol)
                and -tol <= s_vn <= log_d + tol
                and herm_res <= _HERM_ATOL
            )
            verdict = int(HeytingVerdict.COHERENT if physical and n_neg == 0
                          else HeytingVerdict.DEGRADED if physical
                          else HeytingVerdict.VETOED)
            if n_neg:
                logger.warning("[Fase1] ρ: %d autovalores negativos (λ_min=%.3e); "
                               "proyección óptima al espectralplex (θ=%.3e, ‖Δ‖_F=%.3e)",
                               n_neg, float(w_raw.min()), theta, dist)
            return SpectralDensityAudit(
                purity=purity, von_neumann_entropy=s_vn,
                eigenvalues_psd=tuple(float(x) for x in wp),
                negative_eigvals_pruned=n_neg, hermiticity_residual=float(herm_res),
                trace_before=tr_before, trace_after_pruning=tr_after,
                projection_distance=float(dist), simplex_shift=theta, rank=rank,
                min_eigenvalue=float(np.min(wp)), entropy_bound_log_d=log_d,
                is_physical=physical, dim=d, verdict=verdict,
            )

        # ── 1.2 · Ley de apertura: monotona, acotada, sin subdesbordamiento ──────────
        def _aperture_law(self, s_mac: float, purity: float) -> Tuple[int, float, bool]:
            """E9: evaluación en log-espacio; se certifica monotonía por construcción."""
            log_att = -(self._kappa * s_mac) / max(purity, 1e-300)
            att = math.exp(max(log_att, _LOG_EXP_FLOOR))
            l_cut = int(max(1, min(self._l_max, math.floor(self._l_max * att))))
            probe = math.exp(max(-(self._kappa * (s_mac + 1e-9)) / max(purity, 1e-300),
                                 _LOG_EXP_FLOOR))
            monotone = probe <= att + 1e-15
            return l_cut, float(att), bool(monotone)

        # ── 1.3 · MÉTODO TERMINAL DE LA FASE 1 → continuación a la Fase 2 ────────────
        def modulate_aperture(
            self, s_mac: float, rho_llm: NDArray,
            continuation: Optional[Callable[[ApertureModulationResult], R]] = None,
        ) -> Union[ApertureModulationResult, R]:
            r"""
            Definición formal final de la Fase 1: `continuation: A → R`. Su cola
            `return continuation(A)` es el arranque de `Phase2._continue_from_phase1_aperture`.
            """
            s = float(s_mac)
            if not math.isfinite(s) or s < 0.0:
                raise EikonalParameterError(f"s_mac finito y ≥ 0; hay {s}")
            audit = self.audit_density_matrix(rho_llm)
            if not audit.is_physical:
                raise QuantumPurityCollapseError(
                    f"ρ no física tras proyección óptima: pureza={audit.purity:.6f}, "
                    f"Tr={audit.trace_after_pruning:.12f}, λ_min={audit.min_eigenvalue:.3e}"
                )
            l_cut, att, mono = self._aperture_law(s, audit.purity)
            verdict = int(heyting_meet(audit.verdict,
                                       HeytingVerdict.COHERENT if mono else HeytingVerdict.DEGRADED))
            result = ApertureModulationResult(
                l_cutoff=l_cut, attenuation_factor=att, spectral_certificate=audit,
                s_mac=s, kappa_coupling=self._kappa, l_max=self._l_max,
                law_monotone_certified=mono, verdict=verdict,
            )
            logger.info("[Fase1] l_max=%d → l_cutoff=%d (κ=%.3f, S_MAC=%.4f, "
                        "pureza=%.6f, S_vN=%.4f, rango=%d)",
                        self._l_max, l_cut, self._kappa, s, audit.purity,
                        audit.von_neumann_entropy, audit.rank)
            if continuation is None:
                return result
            return continuation(result)

        def compute_dynamic_cutoff(self, s_mac: float, rho_llm: NDArray) -> int:
            """Alias retrocompatible v2/v3."""
            out = self.modulate_aperture(s_mac, rho_llm)
            assert isinstance(out, ApertureModulationResult)
            return out.l_cutoff

    # ─────────────────────────────────────────────────────────────────────────────────
    # ══ FASE 2 · VARIEDAD CARACTERÍSTICA EIKONAL  (A, ∇S, n → Σ)                    ══
    # ─────────────────────────────────────────────────────────────────────────────────
    class Phase2_EikonalSurfaceResolver:
        r"""
        E1: H(x,p) = ½(‖p‖²_{G⁻¹} − target) = 0, target = n² (background) | 1 (optical).
        Test BILATERAL relativo. E10: presupuesto de κ derivado de la exactitud pedida.
        Nunca se forma G⁻¹ en la ruta caliente: `cho_solve` (O(n²), mejor precisión).
        """

        def __init__(
            self, metric_tensor: NDArray[np.float64],
            kappa_max: float = _DEFAULT_KAPPA_MAX,
            eikonal_slack: float = _DEFAULT_EIKONAL_SLACK,
            *, convention: str = "background",
            accuracy_target: float = 1e-10,
            metric_field: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        ) -> None:
            if convention not in ("background", "optical"):
                raise EikonalParameterError("convention ∈ {'background','optical'}")
            if not (0.0 <= eikonal_slack < 1.0):
                raise EikonalParameterError("eikonal_slack ∈ [0,1)")
            self._convention = convention
            self._eikonal_slack = float(eikonal_slack)
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._metric_field = metric_field
            n = self._dim
            self._kappa_budget = float(accuracy_target / (_WILKINSON_SAFETY * n * _MACHINE_EPS))
            self._kappa_max = float(min(kappa_max, self._kappa_budget))
            (self._G_inv, self._kappa_G, self._kappa_G_inv,
             self._inv_residual, self._inv_tol, self._kappa_identity_defect) = (
                self._build_and_certify_inverse(self._G))
            self._cf = la.cho_factor(self._G, lower=True, check_finite=False)
            logger.debug("[Fase2] κ(G)=%.3e (presupuesto %.3e), inv_res=%.3e (tol %.3e), "
                         "convención=%s", self._kappa_G, self._kappa_max,
                         self._inv_residual, self._inv_tol, convention)

        # ── 2.1 · Continuación formal desde la Fase 1 ────────────────────────────────
        def _continue_from_phase1_aperture(self, aperture: ApertureModulationResult) -> ApertureModulationResult:
            if not isinstance(aperture, ApertureModulationResult):
                raise TopologicalInvariantError(
                    f"Continuidad Φ₁→Φ₂ violada: se esperaba ApertureModulationResult, hay {type(aperture)}")
            if aperture.spectral_certificate.verdict == int(HeytingVerdict.VETOED):
                raise QuantumPurityCollapseError("certificado espectral vetado en Φ₁")
            if aperture.l_cutoff < 1:
                raise EikonalParameterError("l_cutoff ≥ 1 requerido por el contrato de Φ₁")
            return aperture

        # ── 2.2 · Inversa por Cholesky con certificado de κ y residual honesto ───────
        def _build_and_certify_inverse(
            self, G: NDArray[np.float64]
        ) -> Tuple[NDArray[np.float64], float, float, float, float, float]:
            r"""
            G = LLᵀ ⇒ G⁻¹ = L⁻ᵀL⁻¹. Residual relativo ‖G·G⁻¹−I‖_F/√n (‖I‖_F=√n).
            Tolerancia SIN suelo arbitrario: tol = c·n·eps·κ(G).
            κ(G⁻¹)=κ(G) exactamente ⇒ lo que se audita es la DEGRADACIÓN numérica de
            esa identidad, no una segunda cota redundante. El check "bilateral" es
            redundante (transpuestas ⇒ misma norma F); se conserva sólo como canario
            de asimetría del inverso calculado.
            """
            n = G.shape[0]
            s = 0.5 * (G + G.T)
            ev = la.eigvalsh(s, check_finite=False)
            lmin, lmax = float(ev[0]), float(ev[-1])
            if lmin <= _MACHINE_EPS * max(abs(lmax), 1.0):
                raise EikonalSingularityError(f"G degenerada: λ_min={lmin:.3e}")
            kappa = lmax / lmin
            if kappa > self._kappa_max:
                raise EikonalSingularityError(
                    f"κ(G)={kappa:.3e} excede el presupuesto de exactitud {self._kappa_max:.3e}")
            try:
                L = la.cholesky(s, lower=True, check_finite=False)
                Y = la.solve_triangular(L, np.eye(n), lower=True, check_finite=False)
                g_inv = Y.T @ Y
            except la.LinAlgError:
                logger.warning("[Fase2] Cholesky falló; precondicionador espectral de respaldo")
                pm = MetricSpectralPreconditioner().precondition(s)
                g_inv = np.asarray(pm.G_inv, dtype=np.float64)
            asym = _rel(g_inv - g_inv.T, g_inv)
            g_inv = 0.5 * (g_inv + g_inv.T)
            ev_i = la.eigvalsh(g_inv, check_finite=False)
            if float(ev_i[0]) <= 0.0:
                raise EikonalSingularityError(f"G⁻¹ no SPD: λ_min={float(ev_i[0]):.3e}")
            kappa_i = float(ev_i[-1]) / float(ev_i[0])
            eye = np.eye(n)
            res = max(_fro(s @ g_inv - eye), _fro(g_inv @ s - eye)) / math.sqrt(n)
            tol = _WILKINSON_SAFETY * n * _MACHINE_EPS * kappa
            if res > tol or asym > 1e-10:
                raise EikonalSingularityError(
                    f"Inversión no certificada: residual={res:.3e} > tol={tol:.3e} (asym={asym:.3e})")
            return g_inv, float(kappa), float(kappa_i), float(res), float(tol), float(abs(kappa_i - kappa) / kappa)

        # ── 2.3 · Forma cuadrática SIN formar la inversa (ruta caliente) ─────────────
        def quadratic_form(self, p: NDArray[np.float64]) -> float:
            """pᵀG⁻¹p vía sustitución triangular doble: O(n²), numéricamente superior."""
            return float(p @ la.cho_solve(self._cf, p, check_finite=False))

        def metric_at(self, x: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
            """G(x) si se suministró campo métrico; si no, G constante (Γ≡0, declarado)."""
            if self._metric_field is None or x is None:
                return self._G
            gx = np.asarray(self._metric_field(np.asarray(x, dtype=np.float64)), dtype=np.float64)
            if gx.shape != self._G.shape:
                raise DimensionalMismatchError("metric_field devolvió una forma incompatible")
            return 0.5 * (gx + gx.T)

        # ── 2.4 · MÉTODO TERMINAL DE LA FASE 2 → continuación a la Fase 3 ────────────
        def resolve_eikonal(
            self, phase_gradient: NDArray[np.float64], n_refract: float,
            *, aperture: Optional[ApertureModulationResult] = None,
            continuation: Optional[Callable[[EikonalSurfaceResult], R]] = None,
        ) -> Union[EikonalSurfaceResult, R]:
            r"""
            Test BILATERAL sobre la variedad característica. n=0 (ENZ) se VETA:
            el cono característico colapsa y el certificado sería vacuo.
            n<1 ⇒ velocidad de fase > c: admisible en medios dispersivos, se DECLARA.
            """
            if aperture is not None:
                self._continue_from_phase1_aperture(aperture)
            p = _as_vec("phase_gradient", phase_gradient, self._dim)
            n_ref = float(n_refract)
            if not math.isfinite(n_ref) or n_ref <= 0.0:
                raise EikonalRefractionError(
                    f"n_refract debe ser > 0 (n=0 es ENZ singular); hay {n_ref}")
            norm_sq = self.quadratic_form(p)
            if not math.isfinite(norm_sq) or norm_sq < 0.0:
                raise EikonalSingularityError(f"‖∇S‖²_{{G⁻¹}} inválido: {norm_sq}")
            target = n_ref ** 2 if self._convention == "background" else 1.0
            residual = abs(norm_sq - target)
            rel_res = residual / max(target, 1e-300)
            threshold = self._eikonal_slack * target
            lower_ok = norm_sq >= target - threshold
            upper_ok = norm_sq <= target + threshold
            if not (lower_ok and upper_ok):
                raise EikonalRefractionError(
                    f"Eikonal fuera de la variedad característica: ‖∇S‖²={norm_sq:.6e}, "
                    f"target={target:.6e}, |Δ|={residual:.3e} > slack·target={threshold:.3e}")
            margin = rel_res <= 0.5 * self._eikonal_slack
            verdict = int(HeytingVerdict.COHERENT if margin
                          else HeytingVerdict.DEGRADED if n_ref >= 1.0
                          else HeytingVerdict.DEGRADED)
            result = EikonalSurfaceResult(
                phase_norm_sq=norm_sq, phase_norm=math.sqrt(max(norm_sq, 0.0)),
                n_refract=n_ref, n_sq=n_ref ** 2, target=float(target),
                convention=self._convention, eikonal_residual=float(residual),
                eikonal_relative_residual=float(rel_res), eikonal_threshold=float(threshold),
                hamiltonian_constraint=float(0.5 * (norm_sq - target)),
                margin_sound=bool(margin), upper_bound_ok=bool(upper_ok),
                kappa_G=self._kappa_G, kappa_G_inv=self._kappa_G_inv,
                kappa_identity_defect=self._kappa_identity_defect,
                inverse_residual=self._inv_residual, inverse_tolerance=self._inv_tol,
                phase_velocity_ratio=float(1.0 / n_ref), enz_singular=False,
                dim=self._dim, verdict=verdict,
            )
            logger.debug("[Fase2] H=%.3e, ‖∇S‖²=%.6e, target=%.6e, rel=%.3e, margen=%s",
                         result.hamiltonian_constraint, norm_sq, target, rel_res, margin)
            if continuation is None:
                return result
            return continuation(result)

        def resolve_eikonal_equation(self, phase_gradient: NDArray[np.float64], n_refract: float) -> float:
            """Alias v2/v3."""
            out = self.resolve_eikonal(phase_gradient, n_refract)
            assert isinstance(out, EikonalSurfaceResult)
            return out.phase_norm_sq

        @property
        def G_inv(self) -> NDArray[np.float64]:
            """ABI: copia de la inversa explícita (la ruta caliente usa cho_solve)."""
            return self._G_inv.copy()

    # ─────────────────────────────────────────────────────────────────────────────────
    # ══ FASE 3 · FERMAT, JACOBI/MASLOV, WKB Y CAVIDAD DE POINCARÉ  (Σ, γ̇ → P)       ══
    # ─────────────────────────────────────────────────────────────────────────────────
    class Phase3_FermatActionAuditor:
        r"""
        Flujo de rayos Hamiltoniano en T*M (E5), campos de Jacobi y Maslov (E2–E4),
        transporte WKB (E2/E12), cuadratura de Fermat sin pérdida de datos (E8) y
        auditoría de cavidad de Poincaré–Floquet–Greene (E6).
        """

        def __init__(
            self, metric_tensor: NDArray[np.float64],
            *, resolver: Optional["EikonalAgent.Phase2_EikonalSurfaceResolver"] = None,
            metric_field: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
            hbar_eff: float = _HBAR_EFF_DEFAULT,
        ) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._resolver = resolver
            self._metric_field = metric_field
            self._hbar = float(hbar_eff)
            self._levi_civita = LeviCivitaConnectionAgent(self._G)
            self._omega = _darboux(self._dim)
            self._constant_metric = metric_field is None
            if self._constant_metric:
                logger.debug("[Fase3] G constante ⇒ Γ^μ_αβ ≡ 0 en estas coordenadas: "
                             "las geodésicas son rectas y la corrección geodésica es vacua "
                             "por construcción (declarado, no simulado).")

        # ── 3.1 · Continuación formal desde la Fase 2 ────────────────────────────────
        def _continue_from_phase2_surface(self, surface: EikonalSurfaceResult) -> EikonalSurfaceResult:
            if not isinstance(surface, EikonalSurfaceResult):
                raise TopologicalInvariantError(
                    f"Continuidad Φ₂→Φ₃ violada: se esperaba EikonalSurfaceResult, hay {type(surface)}")
            if surface.verdict == int(HeytingVerdict.VETOED) or surface.enz_singular:
                raise EikonalRefractionError("superficie eikonal vetada en Φ₂")
            return surface

        # ── 3.2 · Métrica, Christoffel y RHS geodésico REAL ──────────────────────────
        def _metric_at(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            if self._metric_field is None:
                return self._G
            gx = np.asarray(self._metric_field(x), dtype=np.float64)
            return 0.5 * (gx + gx.T)

        def _metric_derivative(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            """∂_μ G_{αβ} por diferencias centrales, h = eps^{1/3}·escala (error O(h²))."""
            d = self._dim
            out = np.zeros((d, d, d), dtype=np.float64)
            if self._metric_field is None:
                return out
            h = (1e-16) ** (1.0 / 3.0) * max(1.0, float(np.max(np.abs(x))))
            for mu in range(d):
                e = np.zeros(d); e[mu] = h
                out[mu] = (self._metric_at(x + e) - self._metric_at(x - e)) / (2.0 * h)
            return out

        def christoffel(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            r"""Γ^μ_{αβ} = ½ G^{μν}(∂_α G_{νβ} + ∂_β G_{να} − ∂_ν G_{αβ})."""
            g = self._metric_at(x)
            dg = self._metric_derivative(x)
            g_inv = la.inv(g)
            return 0.5 * np.einsum("mn,anb,bna->mab", g_inv, dg, dg, optimize=True) * 0.0 + \
                   0.5 * (np.einsum("mn,anb->mab", g_inv, dg) +
                          np.einsum("mn,bna->mab", g_inv, dg) -
                          np.einsum("mn,nab->mab", g_inv, dg))

        def geodesic_rhs(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
            """dv^μ/dt = −Γ^μ_{αβ} v^α v^β (cero exacto si G es constante)."""
            gam = self.christoffel(x)
            return -np.einsum("mab,a,b->m", gam, v, v, optimize=True)

        # ── 3.3 · Hamiltoniano, gradiente y Hessiano en T*M ─────────────────────────
        def _ham_grad(self, z: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            d = self._dim
            x, p = z[:d], z[d:]
            g = self._metric_at(x)
            cf = la.cho_factor(g, lower=True, check_finite=False)
            v = la.cho_solve(cf, p, check_finite=False)
            h = 0.5 * float(p @ v)
            dg = self._metric_derivative(x)
            dh_dx = -0.5 * np.einsum("mab,a,b->m", dg, v, v, optimize=True)
            return h, np.concatenate([v, -dh_dx])

        def _ham_hessian(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
            """D²H por diferencias centrales del gradiente (exacta y constante si G lo es)."""
            n = 2 * self._dim
            h = 1e-6 * max(1.0, float(np.max(np.abs(z))))
            hes = np.empty((n, n), dtype=np.float64)
            for j in range(n):
                e = np.zeros(n); e[j] = h
                hes[:, j] = (self._ham_grad(z + e)[1] - self._ham_grad(z - e)[1]) / (2.0 * h)
            return 0.5 * (hes + hes.T)

        def _symplectic_step(
            self, z: NDArray[np.float64], h: float
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
            r"""
            E5: punto medio implícito z' = z + h·J∇H((z+z')/2) — simplectico, orden 2,
            reversible. La ecuación variacional lineal ż = A z con A = J·D²H ∈ sp(2d)
            se integra con la transformada de Cayley ⇒ M ∈ Sp(2d,ℝ) EXACTAMENTE.
            """
            d, n = self._dim, 2 * self._dim
            j = self._omega
            zk = z.copy()
            it, conv = 0, False
            for it in range(_MIDPOINT_MAX_ITER):
                mid = 0.5 * (z + zk)
                cand = z + h * (j @ self._ham_grad(mid)[1])
                if float(la.norm(cand - zk)) <= 1e-13 * max(1.0, float(la.norm(z))):
                    zk, conv = cand, True
                    break
                zk = cand
            a = j @ self._ham_hessian(0.5 * (z + zk))
            eye = np.eye(n)
            m_step = la.solve(eye - 0.5 * h * a, eye + 0.5 * h * a, check_finite=False)
            return zk, m_step, it + 1, conv

        def propagate_geodesic(
            self, x0: NDArray[np.float64], p0: NDArray[np.float64], n_steps: int, dt: float
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
            r"""
            Integra (x,p) y el marco lagrangiano M(s) con X(0)=0, P(0)=I (fuente puntual).
            Devuelve (posiciones, momentos, velocidades, deriva de H, residuo simpléctico).
            """
            d = self._dim
            z = np.concatenate([_as_vec("x0", x0, d), _as_vec("p0", p0, d)])
            m = np.eye(2 * d)
            xs, ps, vs = [z[:d].copy()], [z[d:].copy()], []
            h0 = self._ham_grad(z)[0]
            for _ in range(max(1, int(n_steps))):
                z, m_step, _, _ = self._symplectic_step(z, dt)
                m = m_step @ m
                xs.append(z[:d].copy()); ps.append(z[d:].copy())
            x_arr = np.asarray(xs); p_arr = np.asarray(ps)
            for x, p in zip(x_arr, p_arr):
                g = self._metric_at(x)
                vs.append(la.solve(g, p, assume_a="pos", check_finite=False))
            h_final = self._ham_grad(z)[0]
            drift = abs(h_final - h0) / max(abs(h0), 1e-300)
            sym = _fro(m.T @ self._omega @ m - self._omega) / _fro(self._omega)
            return x_arr, p_arr, np.asarray(vs), float(drift), float(sym)

        # ── 3.4 · Jacobi, van Vleck, puntos conjugados y Maslov (E2–E4) ──────────────
        def analyze_jacobi_maslov(
            self, positions: NDArray[np.float64], momenta: NDArray[np.float64], dt: float
        ) -> JacobiMaslovReport:
            r"""
            Marco lagrangiano con X(0)=0, P(0)=I. Cáustica ⇔ det X = 0.
            Forma de cruce (Robbin–Salamon): Γ(u) = ⟨P(s₀)u, X'(s₀)u⟩ sobre ker X;
            μ += signatura(Γ). Verificado con el oscilador armónico (Γ=+1 en s=π).
            Morse (E4): índice = nº de puntos conjugados con multiplicidad.
            """
            d = self._dim
            m = np.eye(2 * d)
            dets: List[float] = []
            times: List[float] = []
            crossings: List[float] = []
            signatures: List[int] = []
            z = np.concatenate([positions[0], momenta[0]])
            n_steps = len(positions) - 1
            for k in range(n_steps):
                z_next = np.concatenate([positions[k + 1], momenta[k + 1]])
                _, m_step, _, _ = self._symplectic_step(0.5 * (z + z_next), dt)
                m_prev = m
                m = m_step @ m
                x_prev, x_cur = m_prev[:d, d:], m[:d, d:]
                d_prev, d_cur = float(la.det(x_prev)), float(la.det(x_cur))
                dets.append(d_cur); times.append((k + 1) * dt)
                if k > 0 and np.sign(d_prev) != np.sign(d_cur) and d_prev != 0.0:
                    frac = abs(d_prev) / max(abs(d_prev) + abs(d_cur), 1e-300)
                    m_cross = m_prev + frac * (m - m_prev)
                    xc, pc = m_cross[:d, d:], m_cross[d:, d:]
                    _, sv, vh = la.svd(xc, check_finite=False)
                    ker = vh[sv <= _MASLOV_SVD_TOL * max(float(sv[0]), 1e-300)].conj().T
                    if ker.shape[1] == 0:
                        ker = vh[-1:].conj().T
                    xdot = (x_cur - x_prev) / max(dt, 1e-300)
                    form = ker.conj().T @ (pc.T @ xdot) @ ker
                    form = 0.5 * (form + form.conj().T)
                    ev = la.eigvalsh(form, check_finite=False)
                    sig = int(np.count_nonzero(ev > 1e-12) - np.count_nonzero(ev < -1e-12))
                    crossings.append(float(times[-1] - (1.0 - frac) * dt))
                    signatures.append(sig if sig != 0 else 1)
                z = z_next
            mu = int(sum(signatures))
            det_final = float(dets[-1]) if dets else 1.0
            min_abs = float(min(abs(x) for x in dets)) if dets else 1.0
            sym = _fro(m.T @ self._omega @ m - self._omega) / _fro(self._omega)
            _, det_m = la.slogdet(m)
            caustic = bool(crossings) or min_abs <= 1e-10
            verdict = int(HeytingVerdict.COHERENT if (sym <= 1e-8 and not caustic)
                          else HeytingVerdict.DEGRADED)
            return JacobiMaslovReport(
                maslov_index=mu, conjugate_points=tuple(crossings),
                crossing_signatures=tuple(signatures), van_vleck_det_final=det_final,
                min_abs_det=min_abs, caustic_detected=caustic,
                symplectic_residual=float(sym), determinant_drift=float(abs(math.exp(float(det_m)) - 1.0)),
                hamiltonian_drift=0.0, morse_index=int(sum(abs(s) for s in signatures)),
                measured=bool(dets), verdict=verdict,
            )

        # ── 3.5 · Transporte WKB y validez semiclásica (E2/E12) ──────────────────────
        def wkb_transport(
            self, jacobi: JacobiMaslovReport, phase_norm_sq: float, n_refract: float,
            *, n_gradient_norm: float = 0.0, det_initial: Optional[float] = None,
            amplitude_initial: float = 1.0,
        ) -> WKBTransportReport:
            r"""
            A(s) = A₀√(J₀/J(s)); ∇²S = d/ds log J ≈ Δlog J/Δs. En cáustica J→0:
            la amplitud DIVERGE (correcto físicamente) y se exige asintótica uniforme
            de Airy; NO se fabrica un valor finito. Fase de Keller–Maslov e^{−iπμ/2}.
            """
            j_final = abs(jacobi.van_vleck_det_final)
            j0 = abs(det_initial) if det_initial is not None else max(jacobi.min_abs_det, 1e-12)
            if j_final <= 1e-14:
                ratio = float("inf")
                amp = float("inf")
                lap = float("inf")
                reg = "airy_uniform_required"
            else:
                ratio = math.sqrt(max(j0, 1e-300) / j_final)
                amp = amplitude_initial * ratio
                lap = math.log(max(j_final, 1e-300) / max(j0, 1e-300))
                reg = "airy_uniform_required" if jacobi.caustic_detected else "none"
            mu = jacobi.maslov_index
            phase = -0.5 * math.pi * mu
            wkb_grad = self._hbar * n_gradient_norm / max(n_refract ** 2, 1e-300)
            wkb_amp = self._hbar * abs(lap) / max(phase_norm_sq, 1e-300)
            valid = bool(wkb_grad < 0.1 and wkb_amp < 0.1 and not jacobi.caustic_detected)
            verdict = int(HeytingVerdict.COHERENT if valid else HeytingVerdict.DEGRADED)
            return WKBTransportReport(
                amplitude_initial=float(amplitude_initial), amplitude_final=float(amp),
                amplitude_ratio=float(ratio), maslov_phase=float(phase),
                complex_phase_factor=(math.cos(phase), math.sin(phase)),
                laplacian_S=float(lap), wkb_parameter_gradient=float(wkb_grad),
                wkb_parameter_amplitude=float(wkb_amp), wkb_valid=valid,
                caustic_regularization=reg, verdict=verdict,
            )

        # ── 3.6 · Cuadratura de Fermat: GRIN, sin pérdida de nodos, Richardson ───────
        def _speed_norms(self, V: NDArray[np.float64], G: Optional[NDArray] = None) -> NDArray[np.float64]:
            g = self._G if G is None else G
            quad = np.einsum("ti,ij,tj->t", V, g, V, optimize=True)
            if np.any(quad < -1e-10):
                raise MetricSignatureError("‖v‖²_G < 0: G no es SPD en la práctica")
            return np.sqrt(np.maximum(quad, 0.0))

        def integrate_fermat(
            self, path_velocities: NDArray[np.float64], n_refract: float, dt: float = 1e-3,
            *, n_profile: Optional[Sequence[float]] = None,
            convention: str = "background",
        ) -> Tuple[float, str, int]:
            r"""
            A = ∫ n(x(s))‖γ̇‖_G ds  (convención "background");
            A = ∫ ‖γ̇‖_{G_opt} ds  (convención "optical": n YA está en la métrica).
            El perfil n(s) se integra DENTRO (medio GRIN). Sin descarte de nodos (E8).
            """
            v = np.asarray(path_velocities, dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != self._dim:
                raise DimensionalMismatchError(f"path_velocities debe ser (T,{self._dim}); hay {v.shape}")
            if dt <= 0.0 or not math.isfinite(dt):
                raise EikonalParameterError("dt finito y > 0 requerido")
            norms = self._speed_norms(v)
            if convention == "optical":
                integrand = norms
            elif n_profile is not None:
                prof = np.asarray(n_profile, dtype=np.float64).reshape(-1)
                if prof.size != norms.size:
                    raise DimensionalMismatchError("n_profile debe tener un valor por nodo")
                if np.any(prof <= 0.0):
                    raise EikonalRefractionError("n(s) ≤ 0: medio ENZ singular")
                integrand = prof * norms
            else:
                integrand = float(n_refract) * norms
            value, rule, used = _composite_newton_cotes(integrand, dt)
            if not math.isfinite(value):
                raise FermatOpticalDeviationError(f"Acción de Fermat no finita: {value}")
            return float(value), rule, used

        def adaptive_fermat(
            self, speed_fn: Callable[[float], float], a: float, b: float,
            tol: float = 1e-10, max_depth: int = _SIMPSON_MAX_DEPTH,
        ) -> Tuple[float, float, int, bool]:
            r"""
            Simpson adaptativo con extrapolación de Richardson (E8):
            err ≈ |S_izq+S_der−S|/15; valor corregido S_izq+S_der+(S_izq+S_der−S)/15.
            Devuelve (valor, estimación_de_error, evaluaciones, convergido).
            """
            evals = 0

            def simpson(l: float, r: float) -> Tuple[float, float, float, float]:
                nonlocal evals
                m = 0.5 * (l + r)
                fl, fm, fr = speed_fn(l), speed_fn(m), speed_fn(r)
                evals += 3
                return (r - l) / 6.0 * (fl + 4.0 * fm + fr), fl, fm, fr

            def rec(l: float, r: float, whole: float, fl: float, fm: float, fr: float,
                    eps: float, depth: int) -> Tuple[float, float, bool]:
                nonlocal evals
                m = 0.5 * (l + r)
                lm, rm = 0.5 * (l + m), 0.5 * (m + r)
                f1, f2 = speed_fn(lm), speed_fn(rm)
                evals += 2
                left = (m - l) / 6.0 * (fl + 4.0 * f1 + fm)
                right = (r - m) / 6.0 * (fm + 4.0 * f2 + fr)
                delta = left + right - whole
                if depth <= 0:
                    return left + right + delta / 15.0, abs(delta) / 15.0, False
                if abs(delta) <= 15.0 * eps:
                    return left + right + delta / 15.0, abs(delta) / 15.0, True
                vl, el, cl = rec(l, m, left, fl, f1, fm, eps / 2.0, depth - 1)
                vr, er, cr = rec(m, r, right, fm, f2, fr, eps / 2.0, depth - 1)
                return vl + vr, el + er, bool(cl and cr)

            whole, fl, fm, fr = simpson(a, b)
            val, err, conv = rec(a, b, whole, fl, fm, fr, float(tol), int(max_depth))
            return float(val), float(err), int(evals), bool(conv)

        # ── 3.7 · Cavidad de Poincaré: Floquet–Krein–Greene–KAM (E6) ─────────────────
        def analyze_cavity_monodromy(
            self, m_roundtrip: NDArray[np.float64], *, roundtrip_time: float = 1.0,
            kam_k_max: int = 6, resonance_order: int = 12,
        ) -> CavityFloquetReport:
            r"""
            M ∈ Sp(2d,ℝ). Para d=1 (ABCD): estable ⇔ |tr|/2 ≤ 1 ⇔ R=(2−tr)/4 ∈ [0,1]
            ⇔ multiplicadores en el círculo (Siegmann). Fase de Gouy ν=arccos(tr/2)/2π;
            resonancia ν≈p/q por fracciones continuas; γ_eff diofántico sobre ν.
            Signatura de Krein κ(v)=sgn(−i vᴴΩv) (vᴴΩv es imaginario puro).
            """
            m = np.asarray(m_roundtrip, dtype=np.float64)
            if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] % 2:
                raise DimensionalMismatchError("M de ronda debe ser 2d×2d")
            d = m.shape[0] // 2
            om = _darboux(d)
            sym = _fro(m.T @ om @ m - om) / _fro(om)
            ev, vec = la.eig(m, check_finite=False)
            abs_ev = np.abs(ev)
            tr = float(np.real(np.trace(m)))
            residue = float((2.0 - tr) / 4.0) if d == 1 else float("nan")
            stab_param = abs(tr) / 2.0 if d == 1 else float(np.max(abs_ev))
            stable = bool(np.all(np.abs(abs_ev - 1.0) <= 1e-6))
            krein: List[int] = []
            for i, lam in enumerate(ev):
                if abs(abs_ev[i] - 1.0) <= 1e-6:
                    v = vec[:, i]
                    val = float(np.real(-1j * (v.conj() @ om @ v)))
                    krein.append(1 if val > 0 else (-1 if val < 0 else 0))
            if d == 1 and abs(tr) <= 2.0:
                nu = math.acos(max(-1.0, min(1.0, tr / 2.0))) / (2.0 * math.pi)
            else:
                nu = float("nan")
            if math.isfinite(nu):
                p, q, err = _best_rational(nu, resonance_order)
                gamma = float("inf")
                for k in range(1, kam_k_max + 1):
                    gamma = min(gamma, abs(k * nu - round(k * nu)) * k ** 2)
                phi = 0.5 * (1.0 + math.sqrt(5.0))
                noble = 1.0 / (1.0 + abs(nu - (phi - 1.0)))
            else:
                p, q, err, gamma, noble = 0, 0, float("nan"), float("nan"), 0.0
            verdict = int(HeytingVerdict.COHERENT if (stable and sym <= 1e-8)
                          else HeytingVerdict.DEGRADED if stable
                          else HeytingVerdict.VETOED)
            return CavityFloquetReport(
                trace=tr, greene_residue=residue, stability_parameter=float(stab_param),
                stable=stable, multipliers_abs=tuple(float(x) for x in abs_ev),
                krein_signatures=tuple(krein), gouy_phase_per_roundtrip=float(nu),
                resonance=(int(p), int(q), float(err)), diophantine_gamma_eff=float(gamma),
                noble_nobility=float(noble), symplectic_residual=float(sym),
                measured=True, verdict=verdict,
            )

        # ── 3.8 · Compatibilidad ABI: trayectoria geodésica y residuo medio ──────────
        def enforce_geodesic_path(
            self, initial_velocity: TangentVector, n_steps: int, dt: float = 1.0,
            *, initial_position: Optional[NDArray[np.float64]] = None,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Residuo covariante REAL: ‖(v_{k+1}−v_k)/dt − a_geo‖_G con
            a_geo = −Γ(x_k) v_k v_k. Con G constante, Γ≡0 (declarado).
            """
            if n_steps < 1 or dt <= 0.0:
                raise EikonalParameterError("n_steps ≥ 1 y dt > 0 requeridos")
            d = self._dim
            v = _as_vec("initial_velocity", initial_velocity.coordinates, d)
            x = np.zeros(d) if initial_position is None else _as_vec("x0", initial_position, d)
            g0 = self._metric_at(x)
            p = g0 @ v
            xs, ps, vs, _, _ = self.propagate_geodesic(x, p, n_steps, dt)
            total = 0.0
            for k in range(n_steps):
                a_eff = (vs[k + 1] - vs[k]) / dt
                a_geo = self.geodesic_rhs(xs[k], vs[k])
                delta = a_eff - a_geo
                total += math.sqrt(max(float(delta @ (self._metric_at(xs[k]) @ delta)), 0.0))
            return vs, float(total / n_steps)

        # ── 3.9 · MÉTODO TERMINAL DE LA FASE 3 ───────────────────────────────────────
        def audit_path(
            self, path_velocities: NDArray[np.float64], n_refract: float, *,
            use_geodesic_correction: bool, cavity_tol: float,
            dt_fermat: float = 1e-3, dt_geo: Optional[float] = None,
            surface: Optional[EikonalSurfaceResult] = None,
            positions: Optional[NDArray[np.float64]] = None,
            n_profile: Optional[Sequence[float]] = None,
            cavity_roundtrip_matrix: Optional[NDArray[np.float64]] = None,
            hbar_eff: float = _HBAR_EFF_DEFAULT,
            quadrature_tol: float = 1e-10,
            continuation: Optional[Callable[[FermatPathResult], R]] = None,
        ) -> Union[FermatPathResult, R]:
            r"""
            Definición formal final de la Fase 3. Unifica la parametrización:
            `dt_geo = dt_fermat` por defecto (corrige la inconsistencia 1.0 vs 1e-3).
            Emite acción de Fermat, Jacobi/Maslov, transporte WKB y cavidad de Poincaré.
            """
            if surface is not None:
                self._continue_from_phase2_surface(surface)
            self._hbar = float(hbar_eff)
            v = np.asarray(path_velocities, dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != self._dim:
                raise DimensionalMismatchError(f"path_velocities debe ser (T,{self._dim})")
            dt_geo = float(dt_fermat if dt_geo is None else dt_geo)
            convention = surface.convention if surface is not None else "background"
            geo_dev, drift, sym = 0.0, 0.0, 0.0
            x_arr: Optional[NDArray[np.float64]] = None
            jac: Optional[JacobiMaslovReport] = None

            if use_geodesic_correction and v.shape[0] > 1:
                x0 = np.zeros(self._dim) if positions is None else np.asarray(positions[0], dtype=np.float64)
                p0 = self._metric_at(x0) @ v[0]
                x_arr, p_arr, v_geo, drift, sym = self.propagate_geodesic(
                    x0, p0, v.shape[0] - 1, dt_geo)
                total = 0.0
                for k in range(len(v_geo) - 1):
                    a_eff = (v_geo[k + 1] - v_geo[k]) / dt_geo
                    delta = a_eff - self.geodesic_rhs(x_arr[k], v_geo[k])
                    total += math.sqrt(max(float(delta @ (self._metric_at(x_arr[k]) @ delta)), 0.0))
                geo_dev = float(total / max(len(v_geo) - 1, 1))
                if geo_dev > cavity_tol:
                    raise FermatOpticalDeviationError(
                        f"Residuo geodésico covariante {geo_dev:.6e} > tol={cavity_tol:.6e}")
                v = v_geo
                jac = self.analyze_jacobi_maslov(x_arr, p_arr, dt_geo)
                jac = replace(jac, hamiltonian_drift=drift)
            elif positions is not None:
                x_arr = np.asarray(positions, dtype=np.float64)

            action, rule, used = self.integrate_fermat(
                v, n_refract, dt=dt_fermat, n_profile=n_profile, convention=convention)

            norms = self._speed_norms(v)
            prof = (np.ones_like(norms) if convention == "optical"
                    else np.asarray(n_profile, dtype=np.float64) if n_profile is not None
                    else np.full_like(norms, float(n_refract)))
            integrand = prof * norms
            total_time = max((len(integrand) - 1) * dt_fermat, 0.0)
            if len(integrand) >= 3 and total_time > 0.0:
                adapt, err, _, conv = self.adaptive_fermat(
                    lambda s: float(np.interp(s, np.arange(len(integrand)) * dt_fermat, integrand)),
                    0.0, total_time, tol=quadrature_tol)
                refinements = int(abs(adapt - action) > 1e-14)
            else:
                err, conv, refinements = float("nan"), False, 0

            n_grad = 0.0
            if n_profile is not None and len(prof) > 1:
                n_grad = float(np.max(np.abs(np.gradient(prof, dt_fermat))))
            wkb = (self.wkb_transport(jac, surface.phase_norm_sq if surface else float(np.max(norms) ** 2),
                                      n_refract, n_gradient_norm=n_grad)
                   if jac is not None else None)
            cav = (self.analyze_cavity_monodromy(cavity_roundtrip_matrix)
                   if cavity_roundtrip_matrix is not None else None)

            parts = [HeytingVerdict.COHERENT if geo_dev <= cavity_tol else HeytingVerdict.VETOED]
            if surface is not None:
                parts.append(HeytingVerdict(surface.verdict))
            if jac is not None:
                parts.append(HeytingVerdict(jac.verdict))
            if wkb is not None:
                parts.append(HeytingVerdict(wkb.verdict))
            if cav is not None:
                parts.append(HeytingVerdict(cav.verdict))
            if not conv:
                parts.append(HeytingVerdict.DEGRADED)
            verdict = HeytingVerdict.COHERENT
            for pv in parts:
                verdict = heyting_meet(verdict, pv)

            result = FermatPathResult(
                path_velocities=v, fermat_action=float(action), geodesic_deviation=geo_dev,
                integration_rule=rule, n_refract=float(n_refract), n_steps_integrated=used,
                positions=x_arr, action_dimensionless=float(action / max(hbar_eff, 1e-300)),
                quadrature_error_estimate=float(err), quadrature_converged=bool(conv),
                adaptive_refinements=refinements, grin_profile_used=n_profile is not None,
                hamiltonian_drift=float(drift), symplectic_residual=float(sym),
                jacobi=jac, wkb=wkb, cavity=cav, verdict=int(verdict),
            )
            logger.debug("[Fase3] A=%.6e (%s, %d nodos), R_geo=%.3e, μ=%d, WKB=%s",
                         action, rule, used, geo_dev,
                         jac.maslov_index if jac else 0, wkb.wkb_valid if wkb else False)
            if continuation is None:
                return result
            return continuation(result)

        def audit_fermat_action(self, path_velocities: NDArray[np.float64],
                                n_refract: float, dt: float = 1e-3) -> float:
            """Alias v2/v3."""
            action, _, _ = self.integrate_fermat(path_velocities, n_refract, dt=dt)
            return action

    # Alias histórico del docstring v3
    Phase3_FermatPathAuditor = Phase3_FermatActionAuditor

    # ─────────────────────────────────────────────────────────────────────────────────
    # §4 · CONSTRUCCIÓN DEL ORQUESTADOR (composición, no herencia)
    # ─────────────────────────────────────────────────────────────────────────────────
    def __init__(
        self, metric_tensor: Any = None, l_max_absolute: int = 50, kappa_coupling: float = 1.0,
        eikonal_slack: float = _DEFAULT_EIKONAL_SLACK, kappa_max: float = _DEFAULT_KAPPA_MAX,
        *, convention: str = "background", accuracy_target: float = 1e-10,
        metric_field: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        hbar_eff: float = _HBAR_EFF_DEFAULT, enable_bogoliubov_bridge: bool = True,
    ) -> None:
        if l_max_absolute < 1:
            raise EikonalParameterError(f"l_max_absolute ≥ 1; hay {l_max_absolute}")
        if kappa_coupling <= 0.0:
            raise EikonalParameterError(f"kappa_coupling > 0; hay {kappa_coupling}")
        if not (0.0 <= eikonal_slack < 1.0):
            raise EikonalParameterError(f"eikonal_slack ∈ [0,1); hay {eikonal_slack}")
        if kappa_max <= 1.0:
            raise EikonalParameterError(f"kappa_max > 1; hay {kappa_max}")
        g = _normalize_metric_source(G_PHYSICS if metric_tensor is None else metric_tensor)
        g = self._validate_metric_spd(g)
        self._G, self._dim = g, int(g.shape[0])
        self._eikonal_slack, self._kappa_max = float(eikonal_slack), float(kappa_max)
        self._convention, self._hbar = convention, float(hbar_eff)
        self.phase1 = EikonalAgent.Phase1_DynamicApertureModulator(l_max_absolute, kappa_coupling)
        self.phase2 = EikonalAgent.Phase2_EikonalSurfaceResolver(
            g, kappa_max, eikonal_slack, convention=convention,
            accuracy_target=accuracy_target, metric_field=metric_field)
        self.phase3 = EikonalAgent.Phase3_FermatActionAuditor(
            g, resolver=self.phase2, metric_field=metric_field, hbar_eff=hbar_eff)
        self._lens_fibrator = OpticalRiemannLensFibrator(g)
        self._bogoliubov = (BogoliubovAgent() if (enable_bogoliubov_bridge and BogoliubovAgent is not None) else None)
        try:
            super().__init__(name="EikonalAgent")
        except TypeError:
            super().__init__()
        logger.info("[EikonalAgent] v4: d=%d, l_max=%d, κ_coup=%.3f, slack=%.3f, "
                    "κ_max_efectivo=%.3e, convención=%s, ħ_eff=%.3g",
                    self._dim, l_max_absolute, kappa_coupling, eikonal_slack,
                    self.phase2._kappa_max, convention, hbar_eff)

    @staticmethod
    def _validate_metric_spd(g: NDArray[np.float64]) -> NDArray[np.float64]:
        if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] == 0:
            raise DimensionalMismatchError(f"G cuadrada no vacía requerida; hay {g.shape}")
        if not np.all(np.isfinite(g)):
            raise MetricSignatureError("G con entradas no finitas")
        if _rel(g - g.T, g) > _SYM_ATOL:
            raise MetricSignatureError(f"G no simétrica: {_rel(g - g.T, g):.3e}")
        s = 0.5 * (g + g.T)
        try:
            la.cholesky(s, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(f"G no SPD: {exc}") from exc
        return s

    # ── §5 · Adaptadores públicos del lente (sin tocar atributos privados ajenos) ────
    def _refractive_index(self, sigma: float) -> float:
        """Adaptador: usa el puerto público si existe; si no, la ley declarada."""
        for name in ("compute_fermat_refractive_index", "refractive_index"):
            fn = getattr(self._lens_fibrator, name, None)
            if callable(fn):
                return float(fn(sigma))
        fn = getattr(self._lens_fibrator, "_compute_fermat_refractive_index", None)
        if callable(fn):
            logger.debug("[Lens] usando puerto privado heredado (deuda de encapsulamiento)")
            return float(fn(sigma))
        return 1.0 + math.tanh(0.5 * float(sigma))

    def _inject_cutoff(self, l_cutoff: int) -> None:
        setter = getattr(self._lens_fibrator, "set_cutoff", None)
        if callable(setter):
            setter(int(l_cutoff))
            return
        if hasattr(self._lens_fibrator, "l_cutoff"):
            setattr(self._lens_fibrator, "l_cutoff", int(l_cutoff))
            return
        setattr(self._lens_fibrator, "_l_cutoff", int(l_cutoff))

    # ── §6 · INTERFAZ PÚBLICA: plegado Φ₃∘Φ₂∘Φ₁ por continuaciones anidadas ──────────
    def execute_optical_guidance(self, control: EikonalControlInput) -> EikonalPhaseState:
        r"""
        Pipeline canónico con anidamiento CPS real:
            Φ₁(A) ──continuation──► Φ₂(Σ) ──continuation──► Φ₃(P) ──► EikonalPhaseState
        """
        self._validate_control_input(control)
        n_ref = self._refractive_index(control.logistic_stress_norm)
        if n_ref <= 0.0:
            raise EikonalRefractionError(f"índice del lente no positivo (ENZ): n={n_ref}")
        if n_ref < 1.0:
            logger.warning("[Lens] n=%.4f < 1: velocidad de fase superlumínica "
                           "(admisible en medio dispersivo; declarado)", n_ref)
        trace: Dict[str, Any] = {}

        def _phase3_from_surface(surface: EikonalSurfaceResult) -> FermatPathResult:
            trace["surface"] = surface
            out = self.phase3.audit_path(
                control.path_velocities, n_ref,
                use_geodesic_correction=control.use_geodesic_correction,
                cavity_tol=control.cavity_tol, dt_fermat=control.dt,
                surface=surface, positions=control.positions,
                n_profile=control.refractive_index_profile,
                cavity_roundtrip_matrix=control.cavity_roundtrip_matrix,
                hbar_eff=control.hbar_eff)
            assert isinstance(out, FermatPathResult)
            return out

        def _phase2_from_aperture(aperture: ApertureModulationResult) -> EikonalPhaseState:
            trace["aperture"] = aperture
            surface_out = self.phase2.resolve_eikonal(
                control.phase_gradient, n_ref, aperture=aperture,
                continuation=_phase3_from_surface)
            assert isinstance(surface_out, FermatPathResult)
            self._inject_cutoff(aperture.l_cutoff)
            logits = self._project_logits(control)
            refracted = self._lens_fibrator.refract_attention_logits(logits, control.logistic_stress_norm)
            verdict = heyting_meet(heyting_meet(HeytingVerdict(aperture.verdict),
                                                HeytingVerdict(trace["surface"].verdict)),
                                   HeytingVerdict(surface_out.verdict))
            jac = surface_out.jacobi
            state = EikonalPhaseState(
                phase_gradient_norm=trace["surface"].phase_norm_sq,
                fermat_action_integral=surface_out.fermat_action,
                dynamic_l_cutoff=aperture.l_cutoff, refracted_state=refracted,
                geodesic_deviation=surface_out.geodesic_deviation,
                spectral_certificate=aperture.spectral_certificate,
                eikonal_surface=trace["surface"], fermat_path=surface_out, aperture=aperture,
                phase_gradient_norm_true=trace["surface"].phase_norm,
                maslov_index=jac.maslov_index if jac else 0,
                wkb_valid=bool(surface_out.wkb.wkb_valid) if surface_out.wkb else False,
                heyting_verdict_value=int(verdict),
                is_coherent=(verdict is HeytingVerdict.COHERENT),
                created_ns=time.perf_counter_ns(),
            )
            logger.info("[EikonalAgent] A=%.4f, l_cutoff=%d, ‖∇S‖=%.4f, μ=%d, "
                        "R_geo=%.2e, veredicto=%s", state.fermat_action_integral,
                        state.dynamic_l_cutoff, state.phase_gradient_norm_true,
                        state.maslov_index, state.geodesic_deviation, state.verdict.name)
            return state

        out = self.phase1.modulate_aperture(
            control.s_mac_entropy, control.rho_llm, continuation=_phase2_from_aperture)
        assert isinstance(out, EikonalPhaseState)
        return out

    def _project_logits(self, control: EikonalControlInput) -> NDArray[np.float64]:
        """Morfismo explícito ℝ^V → ℝ^d (o identidad axiomática si V=d)."""
        logits = np.asarray(control.raw_llm_logits, dtype=np.float64).reshape(-1)
        if control.logits_projection is not None:
            p = np.asarray(control.logits_projection, dtype=np.float64)
            if p.shape != (self._dim, logits.size):
                raise DimensionalMismatchError(f"proyección {p.shape} incompatible con (d,V)")
            return p @ logits
        if logits.size != self._dim:
            raise DimensionalMismatchError(
                f"logits dim={logits.size} ≠ dim(G)={self._dim}: se requiere "
                "`logits_projection` explícita (vocabulario ≠ variedad semántica)")
        return logits

    def _validate_control_input(self, control: EikonalControlInput) -> None:
        d = self._dim
        rho = np.asarray(control.rho_llm)
        if rho.shape != (d, d):
            raise DimensionalMismatchError(f"rho_llm {rho.shape} ≠ ({d},{d})")
        _as_vec("phase_gradient", control.phase_gradient, d)
        v = np.asarray(control.path_velocities, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != d:
            raise DimensionalMismatchError(f"path_velocities {v.shape} incompatible con d={d}")
        if not np.all(np.isfinite(v)):
            raise EikonalParameterError("path_velocities con entradas no finitas")
        if control.positions is not None:
            x = np.asarray(control.positions, dtype=np.float64)
            if x.shape != v.shape:
                raise DimensionalMismatchError("positions debe tener la misma forma que path_velocities")
        if not math.isfinite(float(control.logistic_stress_norm)):
            raise EikonalParameterError("logistic_stress_norm finito requerido")

    # ── §7 · RSI-L3: ledger + Banach + endurecimiento monótono (E11) ─────────────────
    @staticmethod
    def _ledger(state: EikonalPhaseState) -> Tuple[float, Tuple[float, ...]]:
        r"""
        Componentes NO NEGATIVAS, cada una decreciente bajo su propio refinamiento:
          E₁ residuo eikonal relativo · E₂ residuo geodésico escalado ·
          E₃ deriva hamiltoniana · E₄ error de cuadratura relativo ·
          E₅ distancia de proyección espectral.
        V = media ponderada. La monotonía del COMPUESTO se IMPONE por aceptación con
        rollback, no se presupone (honestidad operacional del nivel 3).
        """
        s, p, a = state.eikonal_surface, state.fermat_path, state.aperture
        e1 = s.eikonal_relative_residual if s else 0.0
        e2 = (p.geodesic_deviation if p else 0.0)
        e3 = (p.hamiltonian_drift if p else 0.0)
        e4 = (abs(p.quadrature_error_estimate) / max(abs(p.fermat_action), 1e-300)
              if p and math.isfinite(p.quadrature_error_estimate) else 0.0)
        e5 = a.spectral_certificate.projection_distance if a else 0.0
        comp = (float(e1), float(e2), float(e3), float(e4), float(e5))
        return float(np.mean(comp)), comp

    def run_rsi_l3_cycle(
        self, control: EikonalControlInput, max_iterations: int = 6, tolerance: float = 1e-12,
        hyperparameters: Optional[RSIL3Hyperparameters] = None,
    ) -> Tuple[EikonalPhaseState, RSIL3Report]:
        r"""
        Niveles: R₁ proyección espectral óptima (Φ₁), R₂ presupuesto de κ y forma
        cuadrática por Cholesky (Φ₂), R₃ refinamiento de cuadratura y paso (Φ₃).
        META: reescribe dt/tol/slack SÓLO hacia el endurecimiento (E11); acepta si
        V no crece y los invariantes duros se preservan; si no, rollback.
        Criterio de parada: ΔV < tol ∧ q̂ < 1 (Banach a posteriori).
        """
        hyper = hyperparameters or RSIL3Hyperparameters(dt=control.dt, eikonal_slack=self._eikonal_slack)
        state = self.execute_optical_guidance(control)
        v_prev, _ = self._ledger(state)
        v0 = v_prev
        hist_v: List[float] = [v_prev]
        hist_x: List[NDArray[np.float64]] = [self._state_vector(state)]
        best, best_ctrl, best_hyper = state, control, hyper
        accepted = rejected = 0
        q, bound, certified, monotone = 1.0, float("inf"), False, True

        for _ in range(max(1, int(max_iterations))):
            cand_ctrl = replace(best_ctrl, dt=hyper.dt)
            self.phase2._eikonal_slack = hyper.eikonal_slack
            try:
                cand = self.execute_optical_guidance(cand_ctrl)
            except TopologicalInvariantError as exc:
                logger.warning("[RSI-L3] candidato rechazado por invariante: %s", exc)
                rejected += 1
                monotone = False
                hyper = replace(hyper, dt=max(hyper.dt * 0.5, 1e-9),
                                quadrature_tol=max(hyper.quadrature_tol * 0.5, 1e-14),
                                trust_radius=max(hyper.trust_radius * 0.5, 1e-3),
                                meta_depth=hyper.meta_depth + 1)
                continue
            v_k, _ = self._ledger(cand)
            hard = (cand.heyting_verdict_value == int(HeytingVerdict.VETOED)
                    or not math.isfinite(v_k))
            ok = (not hard) and (v_k <= v_prev + tolerance)
            if ok:
                accepted += 1
                best, best_ctrl = cand, cand_ctrl
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
                dt=float(np.clip(hyper.dt * gain, 1e-9, best_ctrl.dt)),
                quadrature_tol=max(hyper.quadrature_tol * gain, 1e-14),
                eikonal_slack=max(hyper.eikonal_slack * (1.0 if ok else 0.5), 1e-6),
                trust_radius=float(np.clip(hyper.trust_radius * (1.0 if ok else 0.5), 1e-3, 0.5)),
                meta_depth=hyper.meta_depth + 1,
            )
            if ok and delta < tolerance and certified:
                break

        self.phase2._eikonal_slack = hyper.eikonal_slack
        verdict = int(HeytingVerdict.COHERENT if (monotone and certified) else HeytingVerdict.DEGRADED)
        report = RSIL3Report(
            iterations=accepted + rejected, accepted=accepted, rejected=rejected,
            lyapunov_initial=float(v0), lyapunov_final=float(v_prev),
            lyapunov_history=tuple(float(x) for x in hist_v), contraction_q=float(q),
            banach_error_bound=float(bound), contraction_certified=bool(certified),
            monotone=bool(monotone), tightening_only=True, hyperparameters=hyper, verdict=verdict,
        )
        return replace(best, rsi_report=report,
                       heyting_verdict_value=int(heyting_meet(best.heyting_verdict_value, verdict))), report

    @staticmethod
    def _state_vector(state: EikonalPhaseState) -> NDArray[np.float64]:
        p = state.fermat_path
        parts = [np.asarray([state.phase_gradient_norm, state.fermat_action_integral,
                             state.geodesic_deviation, float(state.dynamic_l_cutoff)])]
        if p is not None:
            parts.append(np.asarray(p.path_velocities, dtype=np.float64).ravel())
        return np.concatenate(parts)

    # ── §8 · Puente con el soberano Bogoliubov (Ω₃ + Liouville) ───────────────────────
    def govern_with_bogoliubov(
        self, ray_matrix: NDArray[np.float64], metric_G: Optional[NDArray[np.float64]] = None,
        U_matrix: Optional[NDArray] = None, V_matrix: Optional[NDArray] = None,
        is_fermionic: bool = False,
    ) -> Dict[str, Any]:
        r"""
        Delega la auditoría simpléctica/CAR-CCR al agente anterior. Si no se dan
        (U,V), se usa la transformación identidad (sin squeezing) para auditar sólo
        la monodromía de rayos M ∈ Sp(2d,ℝ) — que es exactamente el objeto que E5
        certifica aquí por Cayley.
        """
        m = np.asarray(ray_matrix, dtype=np.float64)
        d = m.shape[0] // 2
        g = self._G if metric_G is None else np.asarray(metric_G, dtype=np.float64)
        u = np.eye(d, dtype=np.complex128) if U_matrix is None else np.asarray(U_matrix, dtype=np.complex128)
        v = np.zeros((d, d), dtype=np.complex128) if V_matrix is None else np.asarray(V_matrix, dtype=np.complex128)
        local = int(self.phase3.analyze_cavity_monodromy(m).verdict)
        if self._bogoliubov is None:
            return {"bogoliubov": None, "local_verdict": local,
                    "merged_verdict": int(heyting_meet(local, HeytingVerdict.DEGRADED))}
        cert = self._bogoliubov.audit_bogoliubov_transformation(u, v, m, g[:d, :d], is_fermionic)
        return {"bogoliubov": cert, "local_verdict": local,
                "merged_verdict": int(heyting_meet(local, cert.heyting_verdict_value))}

    # ── §9 · forward/backward: adjuntos UNITARIOS reales (no forward=backward) ───────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Aplica el transporte WKB: ψ ↦ A·e^{iS/ħ_eff}·e^{−iπμ/2}·ψ.
        El factor de fase es unitario (módulo 1); la amplitud A es el transporte E2.
        """
        psi = np.asarray(state.payload, dtype=np.complex128).reshape(-1)
        if psi.size != self._dim:
            raise DimensionalMismatchError(f"payload dim={psi.size} ≠ {self._dim}")
        payload = getattr(state, "eikonal_context", None)
        s_action = float(payload.get("action", 0.0)) if isinstance(payload, dict) else 0.0
        mu = int(payload.get("maslov", 0)) if isinstance(payload, dict) else 0
        phase = np.exp(1j * (s_action / self._hbar - 0.5 * math.pi * mu))
        return CategoricalState(payload=phase * psi,
                                label=f"{getattr(state, 'label', '')}::eikonal_forward")

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Adjunto verdadero: U† = conj(U) para un factor de fase ⇒ U†U = I exactamente.
        (La v3 devolvía `forward`, un adjunto falso salvo que la fase fuera trivial.)
        """
        psi = np.asarray(state.payload, dtype=np.complex128).reshape(-1)
        if psi.size != self._dim:
            raise DimensionalMismatchError(f"payload dim={psi.size} ≠ {self._dim}")
        payload = getattr(state, "eikonal_context", None)
        s_action = float(payload.get("action", 0.0)) if isinstance(payload, dict) else 0.0
        mu = int(payload.get("maslov", 0)) if isinstance(payload, dict) else 0
        phase = np.exp(-1j * (s_action / self._hbar - 0.5 * math.pi * mu))
        return CategoricalState(payload=phase * psi,
                                label=f"{getattr(state, 'label', '')}::eikonal_backward")

    # ── §10 · Propiedades ABI ────────────────────────────────────────────────────────
    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._G.copy()

    @property
    def metric_inverse(self) -> NDArray[np.float64]:
        return self.phase2.G_inv

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def phase1_modulator(self) -> ApertureModulatorPort:
        return self.phase1

    @property
    def phase2_resolver(self) -> EikonalResolverPort:
        return self.phase2

    @property
    def phase3_auditor(self) -> FermatAuditorPort:
        return self.phase3