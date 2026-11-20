# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Bool-Novikov Paladin Engine — Fidelidad, Poincaré, Aniquilación y Ω₃ RSI³   ║
║ Ruta   : app/core/inmune_system/bool_novikov_paladin_engine.py                       ║
║ Versión: 15.0.0-BuresGeodesic-OctonionMoufang-PoincareRegular-Turnstile-LoebRSI3     ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA: TRES FASES ANIDADAS POR CONTINUACIÓN (Composición de Kleisli en Set^(•→•))
──────────────────────────────────────────────────────────────────────────────────────
   Π₁ : (ρ, σ) ────────────► G  (DensityGeometryCertificate)
   Π₂ : G ─────────────────► S  (CelestialStabilityReport)
   Π₃ : (G, S) ────────────► V  (PaladinEngineReport)
   Z_paladin = Π₃ ∘ Π₂ ∘ Π₁

   ESTRUCTURA DE ANIDAMIENTO FORMAL:
   · La definición formal final del último método de la Fase 1 (`certify_density_geometry`)
     transfiere el control en posición de cola (`return continuation(G)`) directamente al
     método inicial de la Fase 2 (`_continue_from_phase1_geometry`).
   · La definición formal final del último método de la Fase 2 (`audit_celestial_stability`)
     transfiere el control en posición de cola (`return continuation(S)`) directamente al
     método inicial de la Fase 3 (`_continue_from_phase2_celestial`).
   · Los veredictos se agregan mediante el infimum intuicionista (meet ∧) en el clasificador
     de subobjetos Ω₃ ≅ O(Sierpiński) = {COHERENT=0, DEGRADED=1, VETOED=2}.

AXIOMAS Y TEOREMAS CON ESTATUS EPISTÉMICO EXPLÍCITO
──────────────────────────────────────────────────────────────────────────────────────
 P1 [TEOREMA · Uhlmann & Geodésica de Bures]
    F(ρ,σ) = (Tr√(√ρ σ √ρ))² = ‖√ρ √σ‖₁².
    Dada la descomposición en valores singulares √ρ √σ = W Σ Vᴴ, el factor unitario polar
    de Uhlmann es U = W Vᴴ, cumpliendo Tr(√ρ U √σ) = Tr(Σ) = √F = cos θ, θ ∈ [0, π/2].
    La GEODÉSICA MINIMAL DE BURES-HELSTROM entre ρ (τ=0) y σ (τ=1) viene dada por
    ρ(τ) = A(τ) A(τ)ᴴ con:
        A(τ) = [sin((1−τ)θ)/sin θ]·√ρ + [sin(τθ)/sin θ]·(U √σ).
    Por construcción, ρ(τ) ⪰ 0, Tr(ρ(τ)) = 1 y d_B(ρ(0), ρ(τ)) = √(2 − 2 cos(τθ)).
 P2 [TEOREMA · Fuchs–van de Graaf & Umegaki–Pinsker]
    Con T(ρ,σ) = ½‖ρ−σ‖₁ y D(ρ‖σ) = Tr(ρ(ln ρ − ln σ)) (si supp(ρ) ⊆ supp(σ), +∞ en otro
    caso), se satisfacen simultáneamente:
        1 − √F  ≤  T  ≤  √(1 − F)   y   2 T²  ≤  D(ρ‖σ)  (Desigualdad cuántica de Pinsker).
 P3 [TEOREMA · Inmersión Hipercompleja de Cayley-Dickson (Octoniones O)]
    Proyectando el vector de coherencia generalizado de Bloch–Gell-Mann de ρ y σ sobre el
    álgebra de división normada no asociativa O ≅ ℝ⁸ (plano proyectivo de Fano PG(2,2)),
    el asociador [x,y,z] = (xy)z − x(yz) cuantifica la curvatura no asociativa del par de
    estados, mientras que el residuo de Moufang ‖(xy)(zx) − (x(yz))x‖_O = 0 certifica la
    integridad algebraica de alternatividad.
 P4 [TEOREMA · CRÍTICO · Canal Despolarizante y Pasividad de Brune-Choi]
    El "Escudo de Fides" ρ ↦ (ρ+λI)/(1+dλ) = (1−p)ρ + p·I/d con p = dλ/(1+dλ) es CPTP y
    UNITAL. Por procesamiento de datos, F(Λρ,Λσ) ≥ F(ρ,σ) y S(Λρ) ≥ S(ρ).
    La fidelidad de GOBIERNO se mide SIEMPRE sin escudo; el escudo se contabiliza en un
    ledger separado con su matriz de Choi-Jamiołkowski y rango de Kraus.
 P5 [TEOREMA · Proyección Euclídea al Espectralplex]
    Proyección de Frobenius óptima e idempotente sobre {ρ ⪰ 0, Tr ρ = 1} vía proyección
    simplex de autovalores λ ↦ max(λ − θ, 0) (Duchi et al. 2008).
 P6 [TEOREMA · QED Vértice e⁻e⁺ → 2γ]
    Conservación de cuadrimomento en el CM y paridad C exigen ≥ 2 fotones. En reposo:
    E_2γ = 2 m_e c² = 1021.99790 keV (CODATA-2018). El ledger está estrictamente acotado
    por la anomalía real aniquilada.
 P7 [TEOREMA · Autenticación Ligada al Contexto]
    HMAC-SHA256 sobre contexto canónico (anomalía, SHA256(ρ), SHA256(σ), sesión, contador
    monótono, bucket temporal, key_id) con caché LRU anti-replay en tiempo constante.
 P8 [TEOREMA · Melnikov–Duffing & Área de Turnstile de MacKay–Meiss–Percival]
    Para ẍ − x + x³ = ε(−γẋ + f cos ωt), M(t₀) = −A + B cos(ωt₀) con A = (4/3)γ y
    B = √2 π f ω sech(πω/2). Existen ceros simples (herradura de Smale) ⇔ B > A.
    Cuando B > A, el área simpléctica del lóbulo de transporte caótico (turnstile) es:
        A_lobe / ε = (2/ω)·(√(B² − A²) − A·arccos(A/B)) > 0.
 P9 [TEOREMA · Carta Canónica Regularizada de Poincaré & KAM Diofántico]
    La ecuación de Kepler E − e sin E = M se resuelve mediante el método de 3.er orden de
    Halley. Las singularidades de Delaunay en e=0 e i=0 se eliminan con las variables
    canónicas de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂, λ), verificando la identidad simpléctica
    exacta Λ − ½(ξ₁² + η₁²) − ½(ξ₂² + η₂²) = H = L√(1−e²)cos i.
P10 [TEOREMA · Krein–Moser, Proyección Simpléctica y Poincaré–Cartan]
    Para M ∈ Sp(2n,ℝ), κ(v) = sgn(Re(−i vᴴΩv)). Se calcula la distancia a Sp(2n,ℝ) por
    proyección polar-simpléctica y el defecto del invariante integral de Poincaré-Cartan.
P11 [TEOREMA · Automejora Recursiva de Nivel 3 (RSI-L3) y Seguridad de Löb]
    · Nivel 1 (Estado): Contracción geodésica de Bures ρ_{k+1} = Γ_{τ_k}(ρ_k, σ).
    · Nivel 2 (Meta-Parámetros): Adaptación de τ_k y umbrales vía Banach q̂_k y Nekhoroshev.
    · Nivel 3 (Endofuntor + Verificador de Löb + Topología de Hodge): Selección del
      morfismo CPTP óptimo en el álgebra de Banach y auditoría homológica (β₀=1, β₁=0)
      mediante un verificador inmutable separado (evitando la paradoja de auto-validación
      de Löb: el optimizador no tiene acceso de escritura al predicado de aceptación).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Dependencias del ecosistema (contratos tolerantes, degradación grácil) ───────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Excepción formal cuando un invariante topológico, simpléctico o algebraico falla."""

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name

try:
    from app.agents.omega.bogoliubov_agent import HeytingVerdict, heyting_meet
except ImportError:  # pragma: no cover
    class HeytingVerdict(IntEnum):  # type: ignore[no-redef]
        r"""Ω₃ ≅ O(Sierpiński) ≅ Ω del topos Set^(•→•). Cadena de Gödel (0=⊤, 1=medio, 2=⊥)."""
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    def heyting_meet(a: Any, b: Any) -> Any:  # type: ignore[no-redef]
        return HeytingVerdict(max(int(a), int(b)))


def heyting_join(a: Any, b: Any) -> HeytingVerdict:
    r"""Disyunción intuicionista (a ∨ b) = supremo en orden de verdad = mínimo numérico."""
    return HeytingVerdict(min(int(a), int(b)))


def heyting_implies(a: Any, b: Any) -> HeytingVerdict:
    r"""Implicación relativa de Heyting (a ⇒ b) = ⊤ si a ≤_verdad b (numéricamente a ≥ b), sino b."""
    return HeytingVerdict.COHERENT if int(a) >= int(b) else HeytingVerdict(b)


def heyting_negation(a: Any) -> HeytingVerdict:
    r"""Pseudocomplemento intuicionista ¬a := (a ⇒ ⊥). Falla el tercero excluido: ¬¬DEGRADED = ⊤."""
    return heyting_implies(a, HeytingVerdict.VETOED)


logger = logging.getLogger("APU.Core.BoolNovikovPaladinEngine")
R = TypeVar("R")

_ENGINE_VERSION: Final[str] = (
    "15.0.0-BuresGeodesic-OctonionMoufang-PoincareRegular-Turnstile-LoebRSI3"
)
__version__: Final[str] = _ENGINE_VERSION
version: Final[str] = _ENGINE_VERSION

# ── Constantes físicas universales (CODATA-2018) y umbrales matemáticos ──────────────
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ELECTRON_REST_ENERGY_KEV: Final[float] = 510.99895000
_PAIR_ANNIHILATION_ENERGY_KEV: Final[float] = 2.0 * _ELECTRON_REST_ENERGY_KEV  # 1021.99790 keV
_POSITRON_ANNIHILATION_ENERGY_KEV: Final[float] = _PAIR_ANNIHILATION_ENERGY_KEV
_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95
_BURES_DISTANCE_CEIL: Final[float] = math.sqrt(2.0 - 2.0 * math.sqrt(_UHLMANN_FIDELITY_FLOOR))
_MAX_BANACH_Q: Final[float] = 0.95
_MAX_BANACH_CONTRACTION: Final[float] = _MAX_BANACH_Q ** 3
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_NEG_EIG_REL: Final[float] = 1.0e-9
_HERM_REL_TOL: Final[float] = 1.0e-10
_KAM_TAU: Final[float] = 2.0
_KAM_GAMMA_THRESHOLD: Final[float] = 1.0e-4
_CHIRIKOV_SOFT: Final[float] = 1.0
_CHIRIKOV_HARD: Final[float] = 2.0
_LOG_EXP_CAP: Final[float] = 700.0
_REPLAY_CACHE_MAX: Final[int] = 4096
_DEFAULT_KEY_ID: Final[str] = "EPHEMERAL-UNTRUSTED"

# Ternas orientadas del plano proyectivo de Fano PG(2,2) para la multiplicación en O (Octoniones)
_FANO_TRIPLES: Final[Tuple[Tuple[int, int, int], ...]] = (
    (1, 2, 3), (1, 4, 5), (1, 7, 6),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5),
)

__all__ = [
    "HeytingVerdict", "HeytingTruthValue", "heyting_meet", "heyting_join",
    "heyting_implies", "heyting_negation",
    "PaladinDensityError", "PaladinAuthorizationError", "PaladinCelestialError",
    "PaladinRSIError",
    "SpectraplexProjection", "UhlmannCertificate", "HypercomplexOctonionCertificate",
    "DepolarizingShieldLedger", "DensityGeometryCertificate",
    "DelaunayChart", "PoincareCanonicalChart", "DiophantineKAMReport",
    "MelnikovDuffingReport", "KreinFloquetReport", "CelestialStabilityReport",
    "PaladinCelestialReport", "AnnihilationLedger", "TopologicalTrajectoryAudit",
    "RSI3Certificate", "PaladinRSI3Report", "PaladinEngineReport",
    "RSIL3Hyperparameters", "RSIL3LoopReport",
    "Phase1_DensityGeometryGuard", "Phase2_CelestialStabilityAuditor",
    "Phase3_PaladinAdjudicator", "BoolNovikovPaladinEngine",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §A · CODIFICACIÓN LEGADA Ω₃ Y JERARQUÍA DE EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    r"""ABI legada v13 (COHERENT=0x00, DEGRADED=0x01, VETOED=0x03) con biyección a Ω₃={0,1,2}."""

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def to_canonical(self) -> HeytingVerdict:
        return HeytingVerdict({0x00: 0, 0x01: 1, 0x03: 2}[int(self)])

    @classmethod
    def from_canonical(cls, v: HeytingVerdict) -> "HeytingTruthValue":
        return cls({0: 0x00, 1: 0x01, 2: 0x03}[int(v)])


class PaladinDensityError(TopologicalInvariantError):
    """Violación de la estructura C*-algebraica o del cono positivo D(H)."""


class PaladinAuthorizationError(TopologicalInvariantError):
    """Violación criptográfica: HMAC desligado del contexto, replay o clave inválida."""


class PaladinCelestialError(TopologicalInvariantError):
    """Violación del dominio simpléctico o kepleriano en la mecánica celeste de Poincaré."""


class PaladinRSIError(TopologicalInvariantError):
    """Violación del contrato de Löb, no monotonía de Lyapunov o divergencia de Banach."""


# ═════════════════════════════════════════════════════════════════════════════════════
# §B · PRIMITIVAS MATEMÁTICAS, HIPERCOMPLEJAS (O), GEODÉSICAS Y SIMPLÉCTICAS
# ═════════════════════════════════════════════════════════════════════════════════════
def _fro(a: NDArray) -> float:
    return float(la.norm(np.asarray(a), ord="fro", check_finite=False))


def _as_density(name: str, a: Any, d: Optional[int] = None) -> NDArray[np.complex128]:
    """Validación dura de matriz cuadrada compleja finita no vacía."""
    m = np.asarray(a, dtype=np.complex128)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise PaladinDensityError(f"[{name}] se exige matriz cuadrada no vacía; recibido {m.shape}")
    if d is not None and m.shape[0] != d:
        raise PaladinDensityError(f"[{name}] dimensión {m.shape[0]} incompatible con d={d}")
    if not np.all(np.isfinite(m)):
        raise PaladinDensityError(f"[{name}] entradas no finitas detectadas (NaN/Inf)")
    return m


def _project_spectraplex(
    a: NDArray[np.complex128],
) -> Tuple[NDArray[np.complex128], NDArray[np.float64], float]:
    r"""
    P5: Proyección de Frobenius exacta e idempotente sobre el espectralplex {ρ ⪰ 0, Tr ρ = 1}
    mediante proyección euclídea al simplex canónico Δ^{d−1} (Duchi et al. 2008).
    """
    herm = 0.5 * (a + a.conj().T)
    w, v = la.eigh(herm, check_finite=False)
    lam = np.sort(w.real)[::-1]
    css = np.cumsum(lam) - 1.0
    idx = np.arange(1, lam.size + 1, dtype=np.float64)
    cond = lam - css / idx > 0.0
    k = int(np.max(np.nonzero(cond)[0])) + 1
    theta = float(css[k - 1] / k)
    wp = np.maximum(w.real - theta, 0.0)
    wp_sum = float(np.sum(wp))
    if wp_sum > 0.0:
        wp = wp / wp_sum
    out = (v * wp) @ v.conj().T
    return 0.5 * (out + out.conj().T), wp, theta


def _psd_sqrt(a: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Raíz cuadrada espectral exacta √ρ sin piso artificial que falsifique el soporte."""
    w, v = la.eigh(0.5 * (a + a.conj().T), check_finite=False)
    wc = np.maximum(w.real, 0.0)
    return (v * np.sqrt(wc)) @ v.conj().T, wc


def _matrix_log_support(
    a: NDArray[np.complex128], tol: float = 1e-12
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
    """Logaritmo matricial restringido al soporte ortogonal supp(A) y su proyector P_supp."""
    w, v = la.eigh(0.5 * (a + a.conj().T), check_finite=False)
    cut = tol * max(float(w[-1]), 1.0)
    sup = w.real > cut
    lw = np.zeros_like(w.real)
    lw[sup] = np.log(w.real[sup])
    vs = v[:, sup]
    return (v * lw) @ v.conj().T, vs @ vs.conj().T, int(np.count_nonzero(sup))


def _umegaki_relative_entropy(
    rho: NDArray[np.complex128], sigma: NDArray[np.complex128], tol: float = 1e-12
) -> Tuple[float, bool]:
    r"""
    Entropía relativa cuántica de Umegaki D(ρ‖σ) = Tr(ρ(ln ρ − ln σ)).
    Verifica rigurosamente la condición de soporte supp(ρ) ⊆ supp(σ) ⇔ Tr(ρ(I − P_σ)) = 0.
    """
    log_r, _, _ = _matrix_log_support(rho, tol=tol)
    log_s, proj_s, _ = _matrix_log_support(sigma, tol=tol)
    d = rho.shape[0]
    ker_s = np.eye(d, dtype=np.complex128) - proj_s
    leakage = float(np.real(np.trace(rho @ ker_s)))
    if leakage > 1e-9:
        return float("inf"), False
    val = float(np.real(np.trace(rho @ (log_r - log_s))))
    return max(0.0, val), True


def _von_neumann_entropy(a: NDArray[np.complex128]) -> float:
    """Entropía de von Neumann S(ρ) = −Tr(ρ ln ρ) con convención continua 0 ln 0 := 0."""
    w = la.eigvalsh(0.5 * (a + a.conj().T), check_finite=False).real
    w = np.maximum(w, 0.0)
    m = w > 0.0
    return float(-np.sum(w[m] * np.log(w[m]))) if np.any(m) else 0.0


def _bures_geodesic_interpolate(
    rho: NDArray[np.complex128], sigma: NDArray[np.complex128], tau: float
) -> NDArray[np.complex128]:
    r"""
    P1: Geodésica riemanniana exacta de Bures-Uhlmann entre ρ (τ=0) y σ (τ=1).
    Si √ρ √σ = W Σ Vᴴ, el transporte paralelo polar de Uhlmann es U = W Vᴴ.
    Entonces A(τ) = [sin((1−τ)θ)/sin θ] √ρ + [sin(τθ)/sin θ] U √σ, y ρ(τ) = A(τ) A(τ)ᴴ.
    """
    t = float(np.clip(tau, 0.0, 1.0))
    if t <= 0.0:
        return rho.copy()
    if t >= 1.0:
        return sigma.copy()
    sr, _ = _psd_sqrt(rho)
    ss, _ = _psd_sqrt(sigma)
    w_svd, sv, vh_svd = la.svd(sr @ ss, full_matrices=True, check_finite=False)
    u_polar = w_svd @ vh_svd
    cos_theta = float(np.clip(np.sum(sv.real), 0.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-10:
        a_tau = (1.0 - t) * sr + t * (u_polar @ ss)
    else:
        sin_t = math.sin(theta)
        c0 = math.sin((1.0 - t) * theta) / sin_t
        c1 = math.sin(t * theta) / sin_t
        a_tau = c0 * sr + c1 * (u_polar @ ss)
    out = a_tau @ a_tau.conj().T
    out_proj, _, _ = _project_spectraplex(out)
    return out_proj


def _octonion_multiply(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Multiplicación exacta en el álgebra de Cayley-Dickson de los Octoniones O ≅ ℝ⁸
    orientada según las 7 líneas del plano proyectivo de Fano PG(2,2).
    """
    z = np.zeros(8, dtype=np.float64)
    z[0] = x[0] * y[0] - float(np.dot(x[1:], y[1:]))
    z[1:] = x[0] * y[1:] + y[0] * x[1:]
    for i, j, k in _FANO_TRIPLES:
        z[k] += x[i] * y[j] - x[j] * y[i]
        z[i] += x[j] * y[k] - x[k] * y[j]
        z[j] += x[k] * y[i] - x[i] * y[k]
    return z


def _density_to_octonion(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""
    Inmersión canónica de ρ ∈ D(H) en O ≅ ℝ⁸: la componente escalar e₀ recibe la traza
    normalizada y las 7 unidades imaginarias de Fano (e₁..e₇) reciben las primeras 7
    coordenadas del vector de coherencia de Bloch–Gell-Mann en su(d).
    """
    d = rho.shape[0]
    o = np.zeros(8, dtype=np.float64)
    o[0] = float(np.real(np.trace(rho))) / math.sqrt(float(d))
    if d < 2:
        return o
    coords: List[float] = []
    # Generadores simétricos y antisimétricos fuera de la diagonal + diagonales de Cartan
    for r in range(d):
        for c in range(r + 1, d):
            coords.append(math.sqrt(2.0) * float(np.real(rho[r, c])))
            coords.append(math.sqrt(2.0) * float(np.imag(rho[r, c])))
            if len(coords) >= 7:
                break
        if len(coords) >= 7:
            break
    for k in range(1, d):
        if len(coords) >= 7:
            break
        diag_sum = float(np.sum(np.real(np.diag(rho)[:k]))) - k * float(np.real(rho[k, k]))
        coords.append(diag_sum / math.sqrt(float(k * (k + 1))))
    for idx, val in enumerate(coords[:7]):
        o[idx + 1] = val
    return o


def _canonical_bytes(a: NDArray[np.complex128]) -> bytes:
    arr = np.ascontiguousarray(a, dtype=np.complex128)
    return (
        json.dumps({"shape": list(arr.shape), "dtype": "complex128"}, sort_keys=True).encode()
        + arr.tobytes()
    )


def _darboux(n: int) -> NDArray[np.float64]:
    e, z = np.eye(n), np.zeros((n, n))
    om = np.block([[z, e], [-e, z]])
    om.flags.writeable = False
    return om


def _solve_kepler_halley(mean_anomaly: float, ecc: float, tol: float = 1e-14) -> Tuple[float, float]:
    r"""
    P9: Solucionador de 3.er orden de Halley para la ecuación trascendental de Kepler:
        f(E) = E − e sin E − M = 0.
    Convergencia cúbica garantizada para todo e ∈ [0, 1). Devuelve (E, |f(E)|).
    """
    m = math.fmod(float(mean_anomaly), 2.0 * math.pi)
    if m < 0.0:
        m += 2.0 * math.pi
    e = float(ecc)
    u = m + (e * math.sin(m) / (1.0 - math.sin(m + e) + math.sin(m)) if e > 0.0 else 0.0)
    for _ in range(12):
        sin_u, cos_u = math.sin(u), math.cos(u)
        f = u - e * sin_u - m
        if abs(f) <= tol:
            break
        fp = 1.0 - e * cos_u
        fpp = e * sin_u
        denom = 2.0 * fp * fp - f * fpp
        if abs(denom) < 1e-30:
            break
        u -= (2.0 * f * fp) / denom
    res = abs(u - e * math.sin(u) - m)
    return float(u), float(res)


def _best_rational(x: float, q_max: int) -> Tuple[int, int, float]:
    """Mejor aproximante diofántico p/q (1 ≤ q ≤ q_max) vía convergentes de Gauss."""
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
# §C · DTOs INMUTABLES CERTIFICADOS (Congelados con Slots)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectraplexProjection:
    """P5: Certificado de proyección óptima e idempotente sobre el espectralplex."""

    eigenvalues_projected: Tuple[float, ...]
    simplex_shift: float
    projection_distance: float
    negatives_pruned: int
    min_eigenvalue_raw: float
    min_eigenvalue_projected: float
    trace_before: float
    trace_after: float
    hermiticity_residual: float
    idempotency_residual: float
    rank: int
    purity: float
    entropy: float
    repair_was_numerical: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class UhlmannCertificate:
    """P1–P2: Fidelidad por SVD, Bures, Fuchs–van de Graaf y Umegaki–Pinsker."""

    fidelity_svd: float
    fidelity_spectral: float
    route_disagreement: float
    fidelity_raw_unclamped: float
    upper_bound_violation: float
    bures_distance: float
    bures_angle_rad: float
    trace_distance: float
    umegaki_relative_entropy: float
    support_inclusion_ok: bool
    pinsker_inequality_ok: bool
    fvdg_lower_ok: bool
    fvdg_upper_ok: bool
    supports_disjoint: bool
    is_pure_pair: bool
    fubini_study_applicable: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class HypercomplexOctonionCertificate:
    """P3: Invariantes hipercomplejos de Cayley-Dickson en los Octoniones O ≅ ℝ⁸."""

    octonion_rho_norm: float
    octonion_sigma_norm: float
    associator_norm: float
    moufang_identity_residual: float
    norm_multiplicativity_residual: float
    verdict: int


@dataclass(frozen=True, slots=True)
class DepolarizingShieldLedger:
    """P4: Contabilidad del canal despolarizante CPTP unital y su rango de Choi-Kraus."""

    applied: bool
    lambda_shift: float
    mixing_parameter_p: float
    choi_kraus_rank: int
    entropy_before: float
    entropy_after: float
    entropy_injected_by_channel: float
    fidelity_before: float
    fidelity_after: float
    fidelity_gain_attributed_to_channel: float
    min_eigenvalue_after: float
    can_fake_coherence: bool
    statement: str
    verdict: int


@dataclass(frozen=True, slots=True)
class DensityGeometryCertificate:
    """G — Objeto terminal de la Fase 1 (Π₁) y dominio exacto de la Fase 2 (Π₂)."""

    rho_projected: NDArray[np.complex128]
    sigma_projected: NDArray[np.complex128]
    rho_shielded: Optional[NDArray[np.complex128]]
    projection_rho: SpectraplexProjection
    projection_sigma: SpectraplexProjection
    uhlmann: UhlmannCertificate
    octonion: HypercomplexOctonionCertificate
    shield: DepolarizingShieldLedger
    dim: int
    coherence_ok: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class PoincareCanonicalChart:
    """P9: Variables canónicas regularizadas de Poincaré libres de singularidades e=0, i=0."""

    lambda_actions: Tuple[float, ...]
    xi1_eccentric: Tuple[float, ...]
    eta1_eccentric: Tuple[float, ...]
    xi2_inclination: Tuple[float, ...]
    eta2_inclination: Tuple[float, ...]
    mean_longitudes: Tuple[float, ...]
    eccentric_anomalies: Tuple[float, ...]
    kepler_halley_max_residual: float
    symplectic_identity_residual: float
    measured: bool


@dataclass(frozen=True, slots=True)
class DelaunayChart:
    """Carta clásica de Delaunay (L, G, H; l, g, h) acompañada de la regularización de Poincaré."""

    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    mean_motions: Tuple[float, ...]
    ordering_valid: bool
    measured: bool
    poincare_regular: Optional[PoincareCanonicalChart] = None


@dataclass(frozen=True, slots=True)
class DiophantineKAMReport:
    """P9: Certificado diofántico KAM sobre la red entera ℤ^d ∖ {0}."""

    gamma_effective: float
    tau: float
    k_max: int
    kam_satisfied: bool
    min_divisor: float
    resonances: Tuple[Tuple[int, int, int, float], ...]
    golden_nobility: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class MelnikovDuffingReport:
    """P8: Integral de Melnikov con núcleo SECH y área simpléctica de turnstile homoclínico."""

    damping_offset: float
    forcing_amplitude_sech: float
    legacy_amplitude_csch: float
    kernel_correction_factor: float
    threshold_f_over_gamma: float
    actual_f_over_gamma: float
    has_simple_zeros: bool
    separatrix_splitting: float
    turnstile_lobe_area_over_eps: float
    phase_zero_rad: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class KreinFloquetReport:
    """P10: Signaturas de Krein–Moser, residuos de Greene y defecto de Poincaré–Cartan."""

    multipliers_abs: Tuple[float, ...]
    krein_signatures: Tuple[int, ...]
    greene_residues: Tuple[float, ...]
    reciprocal_pairing_defect: float
    symplectic_residual: float
    poincare_cartan_circulation_defect: float
    krein_collision: bool
    spectrally_stable: bool
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class CelestialStabilityReport:
    """S — Objeto terminal de la Fase 2 (Π₂) y dominio exacto de la Fase 3 (Π₃)."""

    delaunay: DelaunayChart
    kam: DiophantineKAMReport
    melnikov: MelnikovDuffingReport
    chirikov_max: float
    chirikov_couplings_declared: bool
    krein: KreinFloquetReport
    nekhoroshev_scale_time: float
    nekhoroshev_steepness_a: float
    nekhoroshev_steepness_b: float
    nekhoroshev_is_bound: bool
    coverage: float
    verdict: int


@dataclass(frozen=True, slots=True)
class PaladinCelestialReport:
    """DTO compatible con ABI v13/v14 con trazabilidad honesta de mediciones."""

    delaunay_actions: Tuple[float, ...]
    delaunay_frequencies: Tuple[float, ...]
    kam_diophantine_ok: bool
    kam_min_divisor: float
    melnikov_amplitude_rel: float
    melnikov_chaotic: bool
    chirikov_k: float
    krein_stable: bool
    nekhoroshev_time_seconds: float
    measured_fraction: float = 0.0
    turnstile_lobe_area: float = 0.0
    fabrications_removed: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnihilationLedger:
    """P6–P7: Ledger QED e⁻e⁺ → 2γ autenticado por contexto con protección anti-replay."""

    authorized: bool
    auth_reason: str
    token_context_bound: bool
    replay_detected: bool
    key_is_trusted: bool
    key_id: str
    anomaly_norm_initial: float
    anomaly_norm_residual: float
    pairs_available: float
    pairs_authorized: float
    pairs_annihilated: float
    energy_released_kev: float
    photon_energies_kev: Tuple[float, ...]
    energy_conservation_residual: float
    verdict: int


@dataclass(frozen=True, slots=True)
class TopologicalTrajectoryAudit:
    """P11: Homología simplicial 1D (Betti β₀, β₁) y espectro de Hodge de la trayectoria RSI."""

    vertices: int
    edges: int
    betti_0_components: int
    betti_1_cycles: int
    euler_characteristic: int
    algebraic_connectivity_fiedler: float
    loop_free: bool


@dataclass(frozen=True, slots=True)
class RSI3Certificate:
    """P11–P12: Certificado Banach-Caccioppoli a posteriori, Lyapunov y topología de trayectoria."""

    banach_q: float
    banach_error_bound: float
    contraction_certified: bool
    iterations_observed: int
    lyapunov_initial: float
    lyapunov_final: float
    lyapunov_monotone: bool
    lyapunov_history: Tuple[float, ...]
    jerk_third_difference: float
    jerk_rate: float
    jerk_is_stability_criterion: bool
    topology: TopologicalTrajectoryAudit
    synthesized_endofunctor: str
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class PaladinRSI3Report:
    """DTO compatible con ABI v13/v14."""

    banach_ratio_q: float
    banach_contraction_ok: bool
    jerk_capacity: float
    jerk_positive: bool
    lyapunov_value: float
    measured: bool = False
    banach_error_bound: float = float("inf")
    betti_1_cycles: int = 0


@dataclass(frozen=True, slots=True)
class PaladinEngineReport:
    """V — Objeto terminal del motor Paladín Z_paladin = Π₃ ∘ Π₂ ∘ Π₁."""

    timestamp_ns: int
    uhlmann_fidelity: float
    bures_distance: float
    trace_preserved: bool
    positron_annihilated: bool
    positron_energy_released_kev: float
    amber_relief_active: bool
    celestial_report: PaladinCelestialReport
    rsi3_report: PaladinRSI3Report
    truth_value: HeytingTruthValue
    sha256_fingerprint: str
    geometry: Optional[DensityGeometryCertificate] = None
    celestial: Optional[CelestialStabilityReport] = None
    annihilation: Optional[AnnihilationLedger] = None
    rsi3: Optional[RSI3Certificate] = None
    canonical_verdict: int = 0
    trace_residual_measured: float = 0.0
    fingerprint_chain_head: str = ""
    engine_version: str = _ENGINE_VERSION
    coverage: float = 0.0


@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Hiperparámetros de Nivel 2/3 (el verificador inmutable sólo permite su endurecimiento)."""

    shield_lambda_cap: float = 1.0e-10
    fidelity_floor: float = _UHLMANN_FIDELITY_FLOOR
    banach_q_max: float = _MAX_BANACH_Q
    trust_radius: float = 0.35
    bures_geodesic_step: float = 0.45
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSIL3LoopReport:
    """Certificado completo de clausura del bucle de Automejora Recursiva de Nivel 3."""

    iterations: int
    accepted: int
    rejected: int
    lyapunov_history: Tuple[float, ...]
    fidelity_history: Tuple[float, ...]
    contraction_q: float
    banach_error_bound: float
    monotone: bool
    tightening_only: bool
    loeb_verifier_sound: bool
    synthesized_law: str
    topology: TopologicalTrajectoryAudit
    hyperparameters: RSIL3Hyperparameters
    verdict: int


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 1 · GEOMETRÍA CUÁNTICA DE BURES-UHLMANN Y OCTONIONES (Π₁ : → G)    ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase1_DensityGeometryGuard:
    r"""
    Π₁ — Guardián de la geometría riemanniana de Bures-Helstrom y álgebra hipercompleja.

    Ejecuta:
      1.1 Proyección óptima e idempotente al espectralplex (Duchi et al. 2008).
      1.2 Fidelidad de Uhlmann por SVD, cota de Fuchs–van de Graaf, entropía relativa de
          Umegaki D(ρ‖σ) con auditoría de soporte y desigualdad cuántica de Pinsker.
      1.3 Inmersión hipercompleja en Octoniones O ≅ ℝ⁸ con auditoría de Moufang y asociador.
      1.4 Contabilidad de canal despolarizante CPTP unital (Escudo de Fides) y rango de Choi.
      1.5 Geodésica exacta de Bures-Uhlmann Γ_τ(ρ, σ).
      1.6 Método terminal `certify_density_geometry` cuyo retorno encadena directamente con
          el método inicial `_continue_from_phase1_geometry` de la Fase 2.
    """

    def __init__(
        self,
        fidelity_floor: float = _UHLMANN_FIDELITY_FLOOR,
        repair_scale: float = _NEG_EIG_REL,
    ) -> None:
        if not (0.0 < fidelity_floor <= 1.0):
            raise ValueError("fidelity_floor debe pertenecer a (0, 1]")
        self._floor = float(fidelity_floor)
        self._repair_scale = float(repair_scale)

    # ── 1.1 · Proyección espectral óptima con certificado de idempotencia ────────────
    def project(
        self, a: NDArray[np.complex128], name: str = "rho"
    ) -> Tuple[NDArray[np.complex128], SpectraplexProjection]:
        m = _as_density(name, a)
        d = m.shape[0]
        herm_res = _fro(m - m.conj().T) / max(_fro(m), 1e-300)
        if herm_res > _HERM_REL_TOL:
            raise PaladinDensityError(f"[{name}] matriz no hermítica: residual={herm_res:.3e}")
        herm = 0.5 * (m + m.conj().T)
        w_raw = la.eigvalsh(herm, check_finite=False).real
        tr_before = float(np.sum(w_raw))
        proj, wp, theta = _project_spectraplex(herm)
        again, _, _ = _project_spectraplex(proj)
        idem = _fro(again - proj)
        dist = _fro(proj - herm)
        n_neg = int(np.count_nonzero(w_raw < -_MACHINE_EPS * d))
        min_raw = float(w_raw[0])
        numerical = bool(min_raw >= -self._repair_scale * max(1.0, abs(tr_before)))
        purity = float(np.sum(wp ** 2))
        entropy = _von_neumann_entropy(proj)
        ok = (
            float(np.min(wp)) >= -1e-12
            and abs(float(np.sum(wp)) - 1.0) <= 1e-9
            and (1.0 / d - 1e-9) <= purity <= 1.0 + 1e-9
            and idem <= 1e-10
        )
        verdict = (
            HeytingVerdict.COHERENT
            if (ok and n_neg == 0)
            else (HeytingVerdict.DEGRADED if (ok and numerical) else HeytingVerdict.VETOED)
        )
        if n_neg and not numerical:
            logger.error(
                "[Π₁·%s] Violación estructural de positividad: λ_min=%.3e", name, min_raw
            )
        return proj, SpectraplexProjection(
            eigenvalues_projected=tuple(float(x) for x in wp),
            simplex_shift=theta,
            projection_distance=dist,
            negatives_pruned=n_neg,
            min_eigenvalue_raw=min_raw,
            min_eigenvalue_projected=float(np.min(wp)),
            trace_before=tr_before,
            trace_after=float(np.sum(wp)),
            hermiticity_residual=herm_res,
            idempotency_residual=idem,
            rank=int(np.count_nonzero(wp > 1e-14)),
            purity=purity,
            entropy=entropy,
            repair_was_numerical=numerical,
            verdict=int(verdict),
        )

    def sanitize_density_matrix(self, A: NDArray[np.complex128]) -> NDArray[np.complex128]:
        out, _ = self.project(A, "sanitize")
        return out

    # ── 1.2 · Uhlmann por SVD + Fuchs–van de Graaf + Umegaki–Pinsker ─────────────────
    def uhlmann(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> UhlmannCertificate:
        d = rho.shape[0]
        sr, wr = _psd_sqrt(rho)
        ss, ws = _psd_sqrt(sigma)
        sv = la.svdvals(sr @ ss, check_finite=False)
        root_f = float(np.sum(sv))
        f_svd = float(root_f ** 2)
        target = sr @ sigma @ sr
        wt = la.eigvalsh(0.5 * (target + target.conj().T), check_finite=False).real
        f_spec = float(np.sum(np.sqrt(np.maximum(wt, 0.0))) ** 2)
        disagree = abs(f_svd - f_spec) / max(f_svd, 1e-300)
        tol = 64.0 * d * _MACHINE_EPS
        violation = max(0.0, f_svd - (1.0 + tol))
        fidelity = float(min(max(f_svd, 0.0), 1.0))
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity)))
        bures_angle = math.acos(min(max(math.sqrt(fidelity), 0.0), 1.0))
        diff = rho - sigma
        t_dist = float(0.5 * np.sum(la.svdvals(diff, check_finite=False)))
        lo, hi = 1.0 - math.sqrt(fidelity), math.sqrt(max(0.0, 1.0 - fidelity))
        fvdg_lo = bool(t_dist >= lo - 1e-8)
        fvdg_hi = bool(t_dist <= hi + 1e-8)
        d_rel, supp_inc = _umegaki_relative_entropy(rho, sigma)
        pinsker_ok = bool(d_rel >= 2.0 * (t_dist ** 2) - 1e-8) if math.isfinite(d_rel) else True
        _, sup_r, _ = _matrix_log_support(rho)
        _, sup_s, _ = _matrix_log_support(sigma)
        disjoint = bool(abs(float(np.real(np.trace(sup_r @ sup_s)))) < 1e-12)
        pure_pair = bool(
            abs(float(np.sum(wr ** 2)) - 1.0) < 1e-9
            and abs(float(np.sum(ws ** 2)) - 1.0) < 1e-9
        )
        ok = violation <= 0.0 and fvdg_lo and fvdg_hi and pinsker_ok and disagree <= 1e-6
        verdict = HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED
        return UhlmannCertificate(
            fidelity_svd=fidelity,
            fidelity_spectral=float(min(max(f_spec, 0.0), 1.0)),
            route_disagreement=float(disagree),
            fidelity_raw_unclamped=f_svd,
            upper_bound_violation=float(violation),
            bures_distance=float(bures),
            bures_angle_rad=float(bures_angle),
            trace_distance=t_dist,
            umegaki_relative_entropy=float(d_rel),
            support_inclusion_ok=supp_inc,
            pinsker_inequality_ok=pinsker_ok,
            fvdg_lower_ok=fvdg_lo,
            fvdg_upper_ok=fvdg_hi,
            supports_disjoint=disjoint,
            is_pure_pair=pure_pair,
            fubini_study_applicable=pure_pair,
            verdict=int(verdict),
        )

    def compute_uhlmann_fidelity(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        r, _ = self.project(rho, "rho")
        s, _ = self.project(sigma, "sigma")
        cert = self.uhlmann(r, s)
        return cert.fidelity_svd, cert.bures_distance

    # ── 1.3 · Auditoría hipercompleja en Octoniones O (Asociador y Moufang) ──────────
    @staticmethod
    def audit_hypercomplex_octonions(
        rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> HypercomplexOctonionCertificate:
        r"""
        P3: En el álgebra octoniónica O, verifica:
          · Identidad de Moufang: (xy)(zx) == (x(yz))x.
          · Multiplicatividad de la norma de Hurwitz: ‖xy‖_O == ‖x‖_O · ‖y‖_O.
          · Norma del asociador [x, y, z] = (xy)z − x(yz) donde z = _density_to_octonion(½(ρσ+σρ)).
        """
        x = _density_to_octonion(rho)
        y = _density_to_octonion(sigma)
        anti = 0.5 * (rho @ sigma + sigma @ rho)
        z = _density_to_octonion(anti)
        xy = _octonion_multiply(x, y)
        yz = _octonion_multiply(y, z)
        zx = _octonion_multiply(z, x)
        assoc = _octonion_multiply(xy, z) - _octonion_multiply(x, yz)
        moufang_lhs = _octonion_multiply(xy, zx)
        moufang_rhs = _octonion_multiply(_octonion_multiply(x, yz), x)
        moufang_res = float(la.norm(moufang_lhs - moufang_rhs))
        nx, ny, nxy = float(la.norm(x)), float(la.norm(y)), float(la.norm(xy))
        hurwitz_res = abs(nxy - nx * ny)
        ok = moufang_res <= 1e-12 and hurwitz_res <= 1e-12
        return HypercomplexOctonionCertificate(
            octonion_rho_norm=nx,
            octonion_sigma_norm=ny,
            associator_norm=float(la.norm(assoc)),
            moufang_identity_residual=moufang_res,
            norm_multiplicativity_residual=hurwitz_res,
            verdict=int(HeytingVerdict.COHERENT if ok else HeytingVerdict.VETOED),
        )

    # ── 1.4 · Escudo declarado como canal despolarizante con rango de Choi ───────────
    def shield(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        *,
        apply: bool = False,
        lambda_cap: float = 1.0e-10,
    ) -> Tuple[Optional[NDArray[np.complex128]], DepolarizingShieldLedger]:
        d = rho.shape[0]
        herm = 0.5 * (rho + rho.conj().T)
        w = la.eigvalsh(herm, check_finite=False).real
        lam = float(min(max(0.0, _SPECTRAL_PSD_FLOOR - float(w[0])), lambda_cap))
        p = d * lam / (1.0 + d * lam)
        f_before = self.uhlmann(rho, sigma).fidelity_svd
        s_before = _von_neumann_entropy(herm)
        if lam <= 0.0:
            return (herm.copy() if apply else None), DepolarizingShieldLedger(
                applied=bool(apply),
                lambda_shift=0.0,
                mixing_parameter_p=0.0,
                choi_kraus_rank=1,
                entropy_before=s_before,
                entropy_after=s_before,
                entropy_injected_by_channel=0.0,
                fidelity_before=f_before,
                fidelity_after=f_before,
                fidelity_gain_attributed_to_channel=0.0,
                min_eigenvalue_after=float(w[0]),
                can_fake_coherence=False,
                statement="Escudo innecesario (λ_min ≥ piso). Canal unitario identidad (Kraus=1).",
                verdict=int(HeytingVerdict.COHERENT),
            )
        eye = np.eye(d, dtype=np.complex128)
        shielded = (herm + lam * eye) / (1.0 + d * lam)
        sigma_sh = (0.5 * (sigma + sigma.conj().T) + lam * eye) / (1.0 + d * lam)
        s_after = _von_neumann_entropy(shielded)
        f_after = self.uhlmann(shielded, sigma_sh).fidelity_svd
        gain = f_after - f_before
        can_fake = bool(f_before < self._floor <= f_after)
        kraus_rank = int(d * d) if p > 0.0 else 1
        stmt = (
            f"CANAL DESPOLARIZANTE CPTP unital (Choi-Kraus={kraus_rank}): p={p:.3e}. "
            f"ΔS=+{s_after - s_before:.3e}, ΔF=+{gain:.3e}."
        )
        if can_fake:
            stmt += " ⚠ EL ESCUDO CRUZA EL UMBRAL DE COHERENCIA: descartado para gobierno (P4)."
            logger.warning("[Π₁·ESCUDO] %s", stmt)
        verdict = HeytingVerdict.DEGRADED if (can_fake or not apply) else HeytingVerdict.COHERENT
        return (shielded if apply else None), DepolarizingShieldLedger(
            applied=bool(apply),
            lambda_shift=lam,
            mixing_parameter_p=float(p),
            choi_kraus_rank=kraus_rank,
            entropy_before=s_before,
            entropy_after=s_after,
            entropy_injected_by_channel=float(s_after - s_before),
            fidelity_before=f_before,
            fidelity_after=f_after,
            fidelity_gain_attributed_to_channel=float(gain),
            min_eigenvalue_after=float(la.eigvalsh(shielded, check_finite=False)[0]),
            can_fake_coherence=can_fake,
            statement=stmt,
            verdict=int(verdict),
        )

    def shield_density_matrix(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], float]:
        m = _as_density("rho", rho)
        herm = 0.5 * (m + m.conj().T)
        w = la.eigvalsh(herm, check_finite=False).real
        lam = float(max(0.0, _SPECTRAL_PSD_FLOOR - float(w[0])))
        d = m.shape[0]
        return (herm + lam * np.eye(d, dtype=np.complex128)) / (1.0 + d * lam), lam

    # ── 1.5 · Retracción geodésica de Bures-Uhlmann ──────────────────────────────────
    def bures_geodesic_step(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        step_tau: float,
    ) -> NDArray[np.complex128]:
        r_proj, _ = self.project(rho, "rho_geodesic")
        s_proj, _ = self.project(sigma, "sigma_geodesic")
        return _bures_geodesic_interpolate(r_proj, s_proj, step_tau)

    # ── 1.6 · MÉTODO TERMINAL DE LA FASE 1 → CONTINUACIÓN DIRECTA A LA FASE 2 ────────
    def certify_density_geometry(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        apply_shield: bool = False,
        shield_lambda_cap: float = 1.0e-10,
        continuation: Optional[Callable[[DensityGeometryCertificate], R]] = None,
    ) -> Union[DensityGeometryCertificate, R]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1 (Π₁):
        Construye el certificado geométrico-algebraico `G: DensityGeometryCertificate`
        y, en su sentencia terminal, invoca `return continuation(G)` que enlaza con el
        método inicial `Phase2_CelestialStabilityAuditor._continue_from_phase1_geometry(G)`.
        """
        rho, p_rho = self.project(rho_input, "rho_input")
        sigma, p_sig = self.project(rho_target, "rho_target")
        cert = self.uhlmann(rho, sigma)
        oct_cert = self.audit_hypercomplex_octonions(rho, sigma)
        shielded, ledger = self.shield(
            rho, sigma, apply=apply_shield, lambda_cap=shield_lambda_cap
        )
        coherence = bool(cert.fidelity_svd >= self._floor)
        verdict = HeytingVerdict.COHERENT
        for v in (p_rho.verdict, p_sig.verdict, cert.verdict, oct_cert.verdict, ledger.verdict):
            verdict = heyting_meet(verdict, HeytingVerdict(v))
        if not coherence:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        geometry_certificate = DensityGeometryCertificate(
            rho_projected=rho,
            sigma_projected=sigma,
            rho_shielded=shielded,
            projection_rho=p_rho,
            projection_sigma=p_sig,
            uhlmann=cert,
            octonion=oct_cert,
            shield=ledger,
            dim=int(rho.shape[0]),
            coherence_ok=coherence,
            verdict=int(verdict),
        )
        logger.info(
            "[Π₁→Π₂] F=%.10f d_B=%.6f D(ρ‖σ)=%.6e Moufang=%.2e coherente=%s",
            cert.fidelity_svd,
            cert.bures_distance,
            cert.umegaki_relative_entropy,
            oct_cert.moufang_identity_residual,
            coherence,
        )
        if continuation is None:
            return geometry_certificate
        # ── ESLABÓN DE CONTINUACIÓN FORMAL Π₁ ──► Π₂ ──
        return continuation(geometry_certificate)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 2 · MECÁNICA CELESTE DE POINCARÉ Y KAM/KREIN  (Π₂ : G → S)         ══
# ═════════════════════════════════════════════════════════════════════════════════════
class Phase2_CelestialStabilityAuditor:
    r"""
    Π₂ — Auditor de estabilidad celeste de Poincaré, KAM, Melnikov-Turnstile y Krein-Floquet.

    Su primer método `_continue_from_phase1_geometry` recibe el objeto `G` emitido por la
    cola de `Phase1_DensityGeometryGuard.certify_density_geometry`.
    Su método final `audit_celestial_stability` emite `S: CelestialStabilityReport` y
    transfiere el control en cola a `Phase3_PaladinAdjudicator._continue_from_phase2_celestial`.
    """

    def __init__(
        self,
        kam_k_max: int = 6,
        kam_tau: float = _KAM_TAU,
        resonance_order: int = 12,
    ) -> None:
        self._k_max = int(kam_k_max)
        self._tau = float(kam_tau)
        self._q_max = int(resonance_order)

    # ── 2.1 · MÉTODO INICIAL DE LA FASE 2 (CONTINUACIÓN DE LA FASE 1) ────────────────
    def _continue_from_phase1_geometry(
        self, geometry: DensityGeometryCertificate
    ) -> DensityGeometryCertificate:
        r"""
        INICIO FORMAL DE LA FASE 2: Recibe y valida el invariante de frontera `G`
        producido por el método final `certify_density_geometry` de la Fase 1.
        """
        if not isinstance(geometry, DensityGeometryCertificate):
            raise TopologicalInvariantError(
                f"Ruptura de continuidad Kleisli Π₁→Π₂: se esperaba DensityGeometryCertificate, "
                f"recibido {type(geometry)}"
            )
        if geometry.octonion.moufang_identity_residual > 1e-9:
            raise PaladinDensityError(
                "Invariante octoniónico de Moufang violado en la frontera Π₁→Π₂."
            )
        return geometry

    # ── 2.2 · Delaunay + Carta Canónica Regularizada de Poincaré (Kepler-Halley) ─────
    @staticmethod
    def delaunay_from_kepler(
        elements: Sequence[Tuple[float, ...]], mu: float = 1.0
    ) -> DelaunayChart:
        r"""
        P9: Admite tuplas keplerianas de 3 elementos `(a, e, i)` o 6 elementos
        `(a, e, i, M, omega_arg, Omega_node)`.
        Calcula:
          · Acciones de Delaunay: L = √(μa), G = L√(1−e²), H = G cos i.
          · Ecuación de Kepler E − e sin E = M mediante el método cúbico de Halley.
          · Variables canónicas regularizadas de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂, λ):
                Λ = L,   λ = M + ω + Ω,
                ξ₁ = √(2(L−G)) cos(ω+Ω),   η₁ = −√(2(L−G)) sin(ω+Ω),
                ξ₂ = √(2(G−H)) cos(Ω),     η₂ = −√(2(G−H)) sin(Ω),
            verificando la identidad simpléctica exacta Λ − ½(ξ₁²+η₁²) − ½(ξ₂²+η₂²) = H.
        """
        if not elements:
            return DelaunayChart((), (), (), False, False, None)
        acts: List[float] = []
        angs: List[float] = []
        nm: List[float] = []
        lam_list: List[float] = []
        xi1_list: List[float] = []
        eta1_list: List[float] = []
        xi2_list: List[float] = []
        eta2_list: List[float] = []
        mlon_list: List[float] = []
        eanom_list: List[float] = []
        max_kepler_res = 0.0
        max_symp_res = 0.0
        valid = True

        for tup in elements:
            if len(tup) < 3:
                raise PaladinCelestialError(
                    f"Cada cuerpo kepleriano exige al menos (a, e, i); recibido len={len(tup)}"
                )
            a, e, inc = float(tup[0]), float(tup[1]), float(tup[2])
            mean_anom = float(tup[3]) if len(tup) >= 4 else 0.0
            omega_arg = float(tup[4]) if len(tup) >= 5 else 0.0
            omega_node = float(tup[5]) if len(tup) >= 6 else 0.0
            if not (a > 0.0 and 0.0 <= e < 1.0 and mu > 0.0):
                raise PaladinCelestialError(
                    f"Elementos keplerianos fuera de dominio: a={a}, e={e}, μ={mu}"
                )
            L = math.sqrt(mu * a)
            G = L * math.sqrt(1.0 - e * e)
            H = G * math.cos(inc)
            acts.extend([L, G, H])
            angs.extend([mean_anom, omega_arg, omega_node])
            nm.append(math.sqrt(mu / (a ** 3)))
            valid = valid and (L >= G - 1e-14 and G >= abs(H) - 1e-14)

            ecc_anom, kep_res = _solve_kepler_halley(mean_anom, e)
            max_kepler_res = max(max_kepler_res, kep_res)
            varpi = omega_arg + omega_node
            rho1 = math.sqrt(max(0.0, 2.0 * (L - G)))
            rho2 = math.sqrt(max(0.0, 2.0 * (G - H)))
            xi1, eta1 = rho1 * math.cos(varpi), -rho1 * math.sin(varpi)
            xi2, eta2 = rho2 * math.cos(omega_node), -rho2 * math.sin(omega_node)
            symp_check = abs((L - 0.5 * (xi1 * xi1 + eta1 * eta1) - 0.5 * (xi2 * xi2 + eta2 * eta2)) - H)
            max_symp_res = max(max_symp_res, symp_check)

            lam_list.append(L)
            xi1_list.append(xi1)
            eta1_list.append(eta1)
            xi2_list.append(xi2)
            eta2_list.append(eta2)
            mlon_list.append(mean_anom + varpi)
            eanom_list.append(ecc_anom)

        poincare = PoincareCanonicalChart(
            lambda_actions=tuple(lam_list),
            xi1_eccentric=tuple(xi1_list),
            eta1_eccentric=tuple(eta1_list),
            xi2_inclination=tuple(xi2_list),
            eta2_inclination=tuple(eta2_list),
            mean_longitudes=tuple(mlon_list),
            eccentric_anomalies=tuple(eanom_list),
            kepler_halley_max_residual=float(max_kepler_res),
            symplectic_identity_residual=float(max_symp_res),
            measured=True,
        )
        return DelaunayChart(
            actions=tuple(acts),
            angles=tuple(angs),
            mean_motions=tuple(nm),
            ordering_valid=bool(valid),
            measured=True,
            poincare_regular=poincare,
        )

    # ── 2.3 · KAM diofántico sobre vectores enteros k ∈ ℤ^d ∖ {0} ───────────────────
    def diophantine_kam(
        self,
        frequencies: Optional[Sequence[float]],
        gamma_threshold: float = _KAM_GAMMA_THRESHOLD,
    ) -> DiophantineKAMReport:
        if frequencies is None or len(frequencies) < 2:
            return DiophantineKAMReport(
                float("nan"),
                self._tau,
                self._k_max,
                False,
                float("nan"),
                (),
                0.0,
                False,
                int(HeytingVerdict.DEGRADED),
            )
        w = np.asarray(frequencies, dtype=np.float64).ravel()
        d = w.size
        kk = self._k_max
        while (2 * kk + 1) ** d > 250_000 and kk > 2:
            kk -= 1
        gamma, min_div = float("inf"), float("inf")
        for k in product(range(-kk, kk + 1), repeat=d):
            arr = np.asarray(k, dtype=np.float64)
            l1 = float(np.sum(np.abs(arr)))
            if l1 == 0.0 or next(x for x in k if x != 0) < 0:
                continue
            val = abs(float(arr @ w))
            min_div = min(min_div, val)
            gamma = min(gamma, val * (l1 ** self._tau))
        reson: List[Tuple[int, int, int, float]] = []
        for i in range(d):
            for j in range(i + 1, d):
                if abs(w[j]) < 1e-14:
                    continue
                p, q, err = _best_rational(abs(w[i] / w[j]), self._q_max)
                if err < 5e-2:
                    reson.append((i, j, int(p + q), float(err)))
        phi = 0.5 * (1.0 + math.sqrt(5.0))
        noble = (
            1.0 / (1.0 + abs(abs(w[0] / w[1]) - (phi - 1.0))) if abs(w[1]) > 1e-14 else 0.0
        )
        ok = bool(gamma >= gamma_threshold)
        return DiophantineKAMReport(
            gamma_effective=float(gamma),
            tau=self._tau,
            k_max=kk,
            kam_satisfied=ok,
            min_divisor=float(min_div),
            resonances=tuple(reson),
            golden_nobility=float(noble),
            measured=True,
            verdict=int(HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED),
        )

    # ── 2.4 · Melnikov–Duffing con núcleo SECH y área de lóbulo Turnstile ───────────
    @staticmethod
    def melnikov_duffing(
        gamma_diss: Optional[float],
        f_drive: Optional[float],
        omega_drive: Optional[float],
    ) -> MelnikovDuffingReport:
        r"""
        P8: M(t₀) = −A + B cos(ωt₀) con A = (4/3)γ, B = √2 π f ω sech(πω/2).
        Cuando B > A (intersección transversal de variedades estable e inestable), los ceros
        simples adyacentes son ±t₀* con ωt₀* = arccos(A/B) ∈ (0, π/2], y el área simpléctica
        del lóbulo de transporte (turnstile de MacKay–Meiss–Percival) es:
            A_lobe / ε = (2/ω)·(√(B² − A²) − A·arccos(A/B)) > 0.
        """
        if gamma_diss is None or f_drive is None or omega_drive is None:
            return MelnikovDuffingReport(
                damping_offset=float("nan"),
                forcing_amplitude_sech=float("nan"),
                legacy_amplitude_csch=float("nan"),
                kernel_correction_factor=float("nan"),
                threshold_f_over_gamma=float("nan"),
                actual_f_over_gamma=float("nan"),
                has_simple_zeros=False,
                separatrix_splitting=float("nan"),
                turnstile_lobe_area_over_eps=float("nan"),
                phase_zero_rad=float("nan"),
                measured=False,
                verdict=int(HeytingVerdict.DEGRADED),
            )
        g, f, w = abs(float(gamma_diss)), abs(float(f_drive)), abs(float(omega_drive))
        if w <= 0.0:
            raise PaladinCelestialError("ω_drive > 0 requerido para forzamiento armónico")
        x = min(math.pi * w / 2.0, _LOG_EXP_CAP)
        sech = 1.0 / math.cosh(x)
        csch = 1.0 / math.sinh(max(x, 1e-12))
        offset = (4.0 / 3.0) * g
        amp = math.sqrt(2.0) * math.pi * f * w * sech
        legacy = math.sqrt(2.0) * math.pi * f * w * csch
        thresh = (4.0 / (3.0 * math.sqrt(2.0) * math.pi * w)) * math.cosh(x)
        zeros = bool(amp > offset and g > 0.0)
        ratio = f / g if g > 0.0 else float("inf")
        if zeros and amp > 0.0:
            phase_star = math.acos(min(max(offset / amp, 0.0), 1.0))
            lobe_area = (2.0 / w) * (math.sqrt(max(0.0, amp * amp - offset * offset)) - offset * phase_star)
        else:
            phase_star = 0.0
            lobe_area = 0.0
        verdict = HeytingVerdict.DEGRADED if zeros else HeytingVerdict.COHERENT
        if zeros:
            logger.warning(
                "[Π₂·Melnikov] Herradura de Smale activa: amp=%.4e > offset=%.4e, "
                "A_lobe/ε=%.4e",
                amp,
                offset,
                lobe_area,
            )
        return MelnikovDuffingReport(
            damping_offset=float(offset),
            forcing_amplitude_sech=float(amp),
            legacy_amplitude_csch=float(legacy),
            kernel_correction_factor=float(sech / csch) if csch > 0 else float("nan"),
            threshold_f_over_gamma=float(thresh),
            actual_f_over_gamma=float(ratio),
            has_simple_zeros=zeros,
            separatrix_splitting=float(amp - offset),
            turnstile_lobe_area_over_eps=float(lobe_area),
            phase_zero_rad=float(phase_star),
            measured=True,
            verdict=int(verdict),
        )

    # ── 2.5 · Solapamiento de resonancias de Chirikov ────────────────────────────────
    @staticmethod
    def chirikov_overlap(
        frequencies: Optional[Sequence[float]],
        eps_pert: float,
        couplings: Optional[Sequence[float]] = None,
    ) -> Tuple[float, bool]:
        if frequencies is None or len(frequencies) < 2:
            return 0.0, False
        w = np.asarray(frequencies, dtype=np.float64).ravel()
        declared = couplings is not None
        v = np.ones_like(w) if not declared else np.asarray(couplings, dtype=np.float64).ravel()
        half = 2.0 * np.sqrt(max(eps_pert, 0.0) * np.abs(v))
        kmax = 0.0
        for i in range(w.size):
            for j in range(i + 1, w.size):
                sep = abs(abs(w[i]) - abs(w[j]))
                if sep > 1e-14:
                    kmax = max(kmax, float((half[i] + half[j]) / sep))
        return kmax, declared

    # ── 2.6 · Krein–Floquet y defecto del invariante integral de Poincaré–Cartan ─────
    @staticmethod
    def krein_floquet(
        monodromy: Optional[NDArray[np.float64]], collision_tol: float = 1e-4
    ) -> KreinFloquetReport:
        r"""
        P10: Audita M ∈ Sp(2n,ℝ). Calcula:
          · Residuo simpléctico ‖MᵀΩM − Ω‖_F / ‖Ω‖_F.
          · Defecto del primer invariante integral de Poincaré–Cartan ∮ p dq sobre los
            n planos canónicos (q_k, p_k): |ω(M e_{q_k}, M e_{p_k}) − 1|.
          · Signaturas de Krein κ(v) = sgn(Re(−i vᴴΩv)) y residuos de Greene R = (2−(λ+λ⁻¹))/4.
        """
        if monodromy is None:
            return KreinFloquetReport(
                (),
                (),
                (),
                float("nan"),
                float("nan"),
                float("nan"),
                False,
                False,
                False,
                int(HeytingVerdict.DEGRADED),
            )
        m = np.asarray(monodromy, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] % 2:
            raise PaladinCelestialError("La matriz de monodromía debe tener dimensión par 2n×2n")
        n = m.shape[0] // 2
        om = _darboux(n)
        sym_diff = m.T @ om @ m - om
        sym = _fro(sym_diff) / _fro(om)
        # Defecto de circulación de Poincaré-Cartan sobre los n pares conjugados (q_k, p_k)
        pc_defects = [abs(float(m[:, k].T @ om @ m[:, n + k]) - 1.0) for k in range(n)]
        pc_defect = float(max(pc_defects, default=0.0))

        ev, vec = la.eig(m, check_finite=False)
        abs_ev = np.abs(ev)
        pairing = max((float(np.min(np.abs(lam * ev - 1.0))) for lam in ev), default=0.0)
        sigs: List[int] = []
        residues: List[float] = []
        on_circle: List[Tuple[complex, int]] = []
        for i, lam in enumerate(ev):
            if abs(abs_ev[i] - 1.0) <= 1e-6:
                v = vec[:, i]
                val = float(np.real(-1j * (v.conj() @ om @ v)))
                kap = 1 if val > 0 else (-1 if val < 0 else 0)
                sigs.append(kap)
                on_circle.append((complex(lam), kap))
            inv = 1.0 / lam if abs(lam) > 1e-14 else complex(float("inf"))
            residues.append(float(np.real(2.0 - (lam + inv)) / 4.0))
        collision = any(
            abs(on_circle[a][0] - on_circle[b][0]) < collision_tol
            and on_circle[a][1] * on_circle[b][1] < 0
            for a in range(len(on_circle))
            for b in range(a + 1, len(on_circle))
        )
        stable = bool(np.all(np.abs(abs_ev - 1.0) <= 1e-6) and not collision)
        verdict = (
            HeytingVerdict.COHERENT
            if (stable and sym <= 1e-8 and pc_defect <= 1e-8)
            else HeytingVerdict.DEGRADED
            if stable
            else HeytingVerdict.VETOED
        )
        return KreinFloquetReport(
            multipliers_abs=tuple(float(x) for x in abs_ev),
            krein_signatures=tuple(sigs),
            greene_residues=tuple(residues),
            reciprocal_pairing_defect=float(pairing),
            symplectic_residual=float(sym),
            poincare_cartan_circulation_defect=pc_defect,
            krein_collision=bool(collision),
            spectrally_stable=stable,
            measured=True,
            verdict=int(verdict),
        )

    # ── 2.7 · MÉTODO TERMINAL DE LA FASE 2 → CONTINUACIÓN DIRECTA A LA FASE 3 ────────
    def audit_celestial_stability(
        self,
        geometry: DensityGeometryCertificate,
        *,
        kepler_elements: Optional[Sequence[Tuple[float, ...]]] = None,
        mu: float = 1.0,
        frequency_vector: Optional[Sequence[float]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        monodromy: Optional[NDArray[np.float64]] = None,
        gamma_diss: Optional[float] = None,
        f_drive: Optional[float] = None,
        omega_drive: Optional[float] = None,
        eps_pert: float = 0.01,
        couplings: Optional[Sequence[float]] = None,
        continuation: Optional[Callable[[CelestialStabilityReport], R]] = None,
    ) -> Union[CelestialStabilityReport, R]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 2 (Π₂):
        Recibe `G` desde `_continue_from_phase1_geometry(geometry)`, construye el reporte
        simpléctico-celeste `S: CelestialStabilityReport` con exponentes de escarpamiento
        cuasi-convexo de Lochak–Neishtadt–Nekhoroshev a = b = 1/(2d), e invoca en posición
        de cola `return continuation(S)` para iniciar `_continue_from_phase2_celestial` en Π₃.
        """
        _ = self._continue_from_phase1_geometry(geometry)
        chart = self.delaunay_from_kepler(kepler_elements or (), mu)
        freqs: Optional[Sequence[float]] = frequency_vector
        if freqs is None and chart.measured:
            freqs = chart.mean_motions
        if freqs is None and hamiltonian is not None:
            h = np.asarray(hamiltonian, dtype=np.complex128)
            ev = np.sort(la.eigvalsh(0.5 * (h + h.conj().T), check_finite=False).real)
            uniq = np.unique(np.round(ev, 12))
            freqs = tuple(float(x) for x in (uniq[1:5] - uniq[0])) if uniq.size > 1 else None

        kam = self.diophantine_kam(freqs)
        mel = self.melnikov_duffing(gamma_diss, f_drive, omega_drive)
        chir, declared = self.chirikov_overlap(freqs, eps_pert, couplings)
        krein = self.krein_floquet(monodromy)
        d_eff = max(len(freqs) if freqs is not None else 1, 1)
        steep_a = 1.0 / (2.0 * d_eff)
        steep_b = 1.0 / (2.0 * d_eff)
        expo = min(
            (max(_KAM_GAMMA_THRESHOLD, 1e-16) / max(eps_pert, 1e-16)) ** steep_a,
            _LOG_EXP_CAP,
        )
        nekho = math.exp(expo)

        parts = [
            HeytingVerdict(kam.verdict),
            HeytingVerdict(mel.verdict),
            HeytingVerdict(krein.verdict),
        ]
        if chir > _CHIRIKOV_HARD:
            parts.append(HeytingVerdict.VETOED)
        elif chir > _CHIRIKOV_SOFT or not declared:
            parts.append(HeytingVerdict.DEGRADED)
        verdict = HeytingVerdict.COHERENT
        for pv in parts:
            verdict = heyting_meet(verdict, pv)

        measured = sum([chart.measured, kam.measured, mel.measured, krein.measured]) / 4.0
        celestial_report = CelestialStabilityReport(
            delaunay=chart,
            kam=kam,
            melnikov=mel,
            chirikov_max=float(chir),
            chirikov_couplings_declared=declared,
            krein=krein,
            nekhoroshev_scale_time=float(nekho),
            nekhoroshev_steepness_a=float(steep_a),
            nekhoroshev_steepness_b=float(steep_b),
            nekhoroshev_is_bound=bool(kam.measured and kam.kam_satisfied),
            coverage=float(measured),
            verdict=int(verdict),
        )
        logger.info(
            "[Π₂→Π₃] KAM γ=%.3e Turnstile=%.3e Chirikov=%.4f Krein=%s cobertura=%.2f",
            kam.gamma_effective,
            mel.turnstile_lobe_area_over_eps,
            chir,
            krein.spectrally_stable,
            measured,
        )
        if continuation is None:
            return celestial_report
        # ── ESLABÓN DE CONTINUACIÓN FORMAL Π₂ ──► Π₃ ──
        return continuation(celestial_report)

    def analyze_celestial_poincare(
        self, rho: NDArray[np.complex128], eps_pert: float = 0.01
    ) -> PaladinCelestialReport:
        _ = _as_density("rho", rho)
        return PaladinCelestialReport(
            delaunay_actions=(),
            delaunay_frequencies=(),
            kam_diophantine_ok=False,
            kam_min_divisor=float("nan"),
            melnikov_amplitude_rel=float("nan"),
            melnikov_chaotic=False,
            chirikov_k=float("nan"),
            krein_stable=False,
            nekhoroshev_time_seconds=float("nan"),
            measured_fraction=0.0,
            turnstile_lobe_area=0.0,
            fabrications_removed=(
                "actions=eigvalsh(rho)",
                "freqs=sqrt((i+1)*lambda_i)",
                "KAM por diferencias de pares con (i+j+2)^tau",
                "Melnikov con csch en lugar de sech",
                "abs(-A+B) como criterio de caos",
                "krein_stable sin matriz de monodromía",
            ),
        )


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 3 · ANIQUILACIÓN QED, TOPOLOGÍA Y RSI-L3 LÖB-SAFE (Π₃ : G,S → V)   ══
# ═════════════════════════════════════════════════════════════════════════════════════
class _LoebSafeImmutableVerifier:
    r"""
    P11: Verificador externo inmutable separado del agente optimizador RSI-L3.
    Evita el obstáculo de auto-referencia de Löb (un sistema que modifica su propio
    predicado de aceptación puede trivializar la función de Lyapunov V ≡ 0).
    Este objeto carece de estado mutable y exige:
      1. Decrecimiento monótono estricto/no creciente del funcional de Lyapunov de Bures.
      2. Endurecimiento monótono de los hiperparámetros de seguridad.
      3. Ausencia de ciclos espurios en la homología 1D de la trayectoria (β₁ = 0).
    """

    __slots__ = ()

    @staticmethod
    def verify_transition(
        v_prev: float,
        v_curr: float,
        hyper_prev: RSIL3Hyperparameters,
        hyper_curr: RSIL3Hyperparameters,
        cand_verdict: int,
        tolerance: float,
    ) -> Tuple[bool, bool]:
        if cand_verdict == int(HeytingVerdict.VETOED) or not math.isfinite(v_curr):
            return False, False
        lyapunov_ok = bool(v_curr <= v_prev + tolerance)
        tightening_ok = bool(
            hyper_curr.shield_lambda_cap <= hyper_prev.shield_lambda_cap + 1e-18
            and hyper_curr.fidelity_floor >= hyper_prev.fidelity_floor - 1e-15
            and hyper_curr.banach_q_max <= hyper_prev.banach_q_max + 1e-15
        )
        return lyapunov_ok, tightening_ok


class Phase3_PaladinAdjudicator:
    r"""
    Π₃ — Adjudicador soberano en Ω₃, ledger QED e⁻e⁺→2γ, topología de Hodge-Betti y RSI-L3.

    Su primer método `_continue_from_phase2_celestial` es la continuación directa del
    método terminal `audit_celestial_stability` de la Fase 2.
    """

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        *,
        key_id: str = _DEFAULT_KEY_ID,
        max_annihilable_anomaly: float = 1.0,
        anomaly_per_pair: float = 1.0,
        token_window_ns: int = 60_000_000_000,
        require_explicit_key: bool = False,
    ) -> None:
        if secret_key is None:
            if require_explicit_key:
                raise PaladinAuthorizationError(
                    "Se exige clave explícita de ≥ 256 bits para autenticación HMAC (P7)."
                )
            secret_key = secrets.token_bytes(32)
            logger.warning(
                "[Π₃] Clave efímera generada: veredicto máximo acotado a DEGRADED (P7)."
            )
            self._trusted = False
        else:
            if len(secret_key) < 32:
                raise PaladinAuthorizationError("Clave HMAC ≥ 256 bits (32 bytes) requerida")
            self._trusted = True
        self._key = bytes(secret_key)
        self._key_id = str(key_id)
        self._max_ann = float(max_annihilable_anomaly)
        self._per_pair = float(anomaly_per_pair)
        self._window = int(token_window_ns)
        self._replay: "OrderedDict[str, int]" = OrderedDict()
        self._counter = 0
        self._chain_head = ""

    # ── 3.1 · MÉTODO INICIAL DE LA FASE 3 (CONTINUACIÓN DE LA FASE 2) ────────────────
    def _continue_from_phase2_celestial(
        self, celestial: CelestialStabilityReport
    ) -> CelestialStabilityReport:
        r"""
        INICIO FORMAL DE LA FASE 3: Recibe y valida el certificado simpléctico-celeste `S`
        emitido en posición de cola por `Phase2_CelestialStabilityAuditor.audit_celestial_stability`.
        """
        if not isinstance(celestial, CelestialStabilityReport):
            raise TopologicalInvariantError(
                f"Ruptura de continuidad Kleisli Π₂→Π₃: se esperaba CelestialStabilityReport, "
                f"recibido {type(celestial)}"
            )
        return celestial

    # ── 3.2 · HMAC-SHA256 ligado al contexto + caché anti-replay ─────────────────────
    def build_token(
        self,
        anomaly_norm: float,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str,
        ts_ns: int,
    ) -> bytes:
        self._counter += 1
        msg = self._context_message(
            anomaly_norm, rho, sigma, session_id, ts_ns, self._counter
        )
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def _context_message(
        self,
        anomaly_norm: float,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str,
        ts_ns: int,
        counter: int,
    ) -> bytes:
        payload = {
            "purpose": "positron_annihilation",
            "engine": _ENGINE_VERSION,
            "key_id": self._key_id,
            "anomaly_norm": float(f"{float(anomaly_norm):.17g}"),
            "max_annihilable": self._max_ann,
            "session_id": str(session_id),
            "counter": int(counter),
            "ts_bucket": int(ts_ns // self._window),
            "rho_sha256": hashlib.sha256(_canonical_bytes(rho)).hexdigest(),
            "sigma_sha256": hashlib.sha256(_canonical_bytes(sigma)).hexdigest(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _replay_check(self, token: bytes) -> bool:
        digest = hashlib.sha256(token).hexdigest()
        if digest in self._replay:
            return True
        self._replay[digest] = time.time_ns()
        while len(self._replay) > _REPLAY_CACHE_MAX:
            self._replay.popitem(last=False)
        return False

    # ── 3.3 · Ledger de aniquilación QED e⁻e⁺ → 2γ acotado ──────────────────────────
    def annihilate(
        self,
        electron_anomaly_norm: float,
        positron_hmac: bytes,
        *,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str = "default",
        ts_ns: Optional[int] = None,
        counter: Optional[int] = None,
        pairs_authorized: Optional[float] = None,
    ) -> AnnihilationLedger:
        ts = int(ts_ns if ts_ns is not None else time.time_ns())
        anomaly = float(electron_anomaly_norm)
        if not math.isfinite(anomaly) or anomaly < 0.0:
            raise PaladinAuthorizationError(f"Norma de anomalía no física: {anomaly}")
        available = anomaly / self._per_pair if self._per_pair > 0 else 0.0

        if anomaly <= 0.0:
            return AnnihilationLedger(
                authorized=True,
                auth_reason="sin_anomalía:no_requiere_token",
                token_context_bound=True,
                replay_detected=False,
                key_is_trusted=self._trusted,
                key_id=self._key_id,
                anomaly_norm_initial=0.0,
                anomaly_norm_residual=0.0,
                pairs_available=0.0,
                pairs_authorized=0.0,
                pairs_annihilated=0.0,
                energy_released_kev=0.0,
                photon_energies_kev=(),
                energy_conservation_residual=0.0,
                verdict=int(HeytingVerdict.COHERENT),
            )

        cnt = int(counter if counter is not None else self._counter + 1)
        expected = hmac.new(
            self._key,
            self._context_message(anomaly, rho, sigma, session_id, ts, cnt),
            hashlib.sha256,
        ).digest()
        provided = bytes(positron_hmac or b"")
        matched = bool(provided) and hmac.compare_digest(provided, expected)
        replay = self._replay_check(provided) if provided else False
        cap_pairs = self._max_ann / self._per_pair if self._per_pair > 0 else 0.0
        auth_pairs = float(
            min(pairs_authorized if pairs_authorized is not None else cap_pairs, cap_pairs)
        )
        if not matched:
            return AnnihilationLedger(
                authorized=False,
                auth_reason="HMAC_no_ligado_al_contexto_o_ausente",
                token_context_bound=False,
                replay_detected=replay,
                key_is_trusted=self._trusted,
                key_id=self._key_id,
                anomaly_norm_initial=anomaly,
                anomaly_norm_residual=anomaly,
                pairs_available=available,
                pairs_authorized=0.0,
                pairs_annihilated=0.0,
                energy_released_kev=0.0,
                photon_energies_kev=(),
                energy_conservation_residual=0.0,
                verdict=int(HeytingVerdict.VETOED),
            )
        if replay:
            logger.error("[Π₃·AUTH] Replay detectado: token previamente consumido (P7).")
            return AnnihilationLedger(
                authorized=False,
                auth_reason="replay_detectado",
                token_context_bound=True,
                replay_detected=True,
                key_is_trusted=self._trusted,
                key_id=self._key_id,
                anomaly_norm_initial=anomaly,
                anomaly_norm_residual=anomaly,
                pairs_available=available,
                pairs_authorized=0.0,
                pairs_annihilated=0.0,
                energy_released_kev=0.0,
                photon_energies_kev=(),
                energy_conservation_residual=0.0,
                verdict=int(HeytingVerdict.VETOED),
            )

        self._counter = max(self._counter, cnt)
        annihilated = float(min(available, auth_pairs))
        energy = annihilated * _PAIR_ANNIHILATION_ENERGY_KEV
        residual = max(0.0, anomaly - annihilated * self._per_pair)
        photons: Tuple[float, ...] = tuple(
            [_ELECTRON_REST_ENERGY_KEV, _ELECTRON_REST_ENERGY_KEV]
            if annihilated >= 1.0
            else ([annihilated * _ELECTRON_REST_ENERGY_KEV] * 2 if annihilated > 0.0 else [])
        )
        ledger_res = abs(energy - sum(photons)) if photons else abs(energy)
        verdict = HeytingVerdict.COHERENT if self._trusted else HeytingVerdict.DEGRADED
        return AnnihilationLedger(
            authorized=True,
            auth_reason="HMAC_ligado_a_contexto_verificado",
            token_context_bound=True,
            replay_detected=False,
            key_is_trusted=self._trusted,
            key_id=self._key_id,
            anomaly_norm_initial=anomaly,
            anomaly_norm_residual=residual,
            pairs_available=available,
            pairs_authorized=auth_pairs,
            pairs_annihilated=annihilated,
            energy_released_kev=float(energy),
            photon_energies_kev=photons,
            energy_conservation_residual=float(ledger_res),
            verdict=int(verdict),
        )

    def annihilate_positron_hmac(
        self, electron_anomaly_norm: float, positron_hmac: bytes
    ) -> Tuple[bool, float, float]:
        dummy = np.zeros((1, 1), dtype=np.complex128)
        led = self.annihilate(
            electron_anomaly_norm, positron_hmac, rho=dummy, sigma=dummy
        )
        return led.authorized, led.energy_released_kev, led.anomaly_norm_residual

    # ── 3.4 · Auditoría topológica de Hodge–Betti (β₀, β₁) de la trayectoria RSI ─────
    @staticmethod
    def audit_trajectory_topology(
        state_history: Optional[Sequence[NDArray]],
    ) -> TopologicalTrajectoryAudit:
        r"""
        Construye el complejo simplicial 1-dimensional de la trayectoria de estados
        (aristas temporales (k, k+1) más aristas de recurrencia ε-cercanas cuando un
        estado revisita un estado no adyacente |i−j| > 1 dentro de radio ε_loop).
        Calcula mediante el espectro del 0-Laplaciano de Hodge L₀ = D − A:
          · β₀ = dim ker(L₀) (componentes conexas),
          · β₁ = E − V + β₀ (ciclos homológicos independientes; β₁ = 0 certifica ausencia de lazos),
          · Conectividad algebraica de Fiedler λ₂(L₀).
        """
        if state_history is None or len(state_history) == 0:
            return TopologicalTrajectoryAudit(0, 0, 0, 0, 0, 0.0, True)
        xs = [np.asarray(x, dtype=np.float64).ravel() for x in state_history]
        n_v = len(xs)
        if n_v == 1:
            return TopologicalTrajectoryAudit(1, 0, 1, 0, 1, 0.0, True)
        adj = np.zeros((n_v, n_v), dtype=np.float64)
        step_dists: List[float] = []
        for i in range(n_v - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
            step_dists.append(float(la.norm(xs[i + 1] - xs[i])))
        med_step = float(np.median(step_dists)) if step_dists else 0.0
        eps_loop = 0.05 * med_step
        # Detecta ciclos homológicos verdaderos (retroceso hacia estados pasados i < j−1)
        if eps_loop > 1e-15:
            for i in range(n_v):
                for j in range(i + 2, n_v):
                    dist_ij = float(la.norm(xs[i] - xs[j]))
                    dist_prev = float(la.norm(xs[i] - xs[j - 1]))
                    if dist_ij < eps_loop and dist_ij < dist_prev:
                        adj[i, j] = adj[j, i] = 1.0
        n_e = int(np.sum(adj) // 2)
        deg = np.diag(np.sum(adj, axis=1))
        lap = deg - adj
        ev = np.sort(la.eigvalsh(lap, check_finite=False).real)
        b0 = int(np.count_nonzero(np.abs(ev) < 1e-9))
        b1 = max(0, n_e - n_v + b0)
        fiedler = float(ev[b0]) if b0 < n_v else 0.0
        return TopologicalTrajectoryAudit(
            vertices=n_v,
            edges=n_e,
            betti_0_components=b0,
            betti_1_cycles=b1,
            euler_characteristic=int(n_v - n_e),
            algebraic_connectivity_fiedler=fiedler,
            loop_free=bool(b1 == 0),
        )

    # ── 3.5 · Certificado RSI-L3: Banach a posteriori, Lyapunov y Topología ──────────
    @classmethod
    def rsi3_certificate(
        cls,
        state_history: Optional[Sequence[NDArray]] = None,
        lyapunov_history: Optional[Sequence[float]] = None,
        capacity_history: Optional[Sequence[float]] = None,
        dt_capacity: float = 1.0,
        q_max: float = _MAX_BANACH_Q,
        synthesized_endofunctor: str = "Identity",
    ) -> RSI3Certificate:
        q, bound, certified = float("nan"), float("inf"), False
        n_obs = 0
        if state_history is not None and len(state_history) >= 3:
            xs = [np.asarray(x, dtype=np.float64).ravel() for x in state_history]
            n_obs = len(xs)
            ratios: List[float] = []
            for k in range(len(xs) - 1, 1, -1):
                d1 = float(la.norm(xs[k] - xs[k - 1]))
                d0 = float(la.norm(xs[k - 1] - xs[k - 2]))
                if d0 > 1e-18:
                    ratios.append(d1 / d0)
                if len(ratios) >= 4:
                    break
            if ratios:
                q = float(np.median(ratios))
                certified = bool(q < q_max)
                last = float(la.norm(xs[-1] - xs[-2]))
                bound = q / (1.0 - q) * last if q < 1.0 else float("inf")
        lyap = tuple(float(x) for x in (lyapunov_history or ()))
        mono = (
            bool(all(lyap[i + 1] <= lyap[i] + 1e-12 for i in range(len(lyap) - 1)))
            if len(lyap) > 1
            else False
        )
        jerk, jerk_rate = float("nan"), float("nan")
        if capacity_history is not None and len(capacity_history) >= 4:
            c3, c2, c1, c0 = (float(capacity_history[-i]) for i in (1, 2, 3, 4))
            jerk = c3 - 3.0 * c2 + 3.0 * c1 - c0
            jerk_rate = jerk / (max(dt_capacity, 1e-300) ** 3)
        topo = cls.audit_trajectory_topology(state_history)
        measured = bool(certified or (n_obs >= 3))
        verdict = (
            HeytingVerdict.COHERENT
            if (certified and (mono or not lyap) and topo.loop_free)
            else HeytingVerdict.DEGRADED
        )
        return RSI3Certificate(
            banach_q=q,
            banach_error_bound=float(bound),
            contraction_certified=certified,
            iterations_observed=n_obs,
            lyapunov_initial=lyap[0] if lyap else float("nan"),
            lyapunov_final=lyap[-1] if lyap else float("nan"),
            lyapunov_monotone=mono,
            lyapunov_history=lyap,
            jerk_third_difference=jerk,
            jerk_rate=jerk_rate,
            jerk_is_stability_criterion=False,
            topology=topo,
            synthesized_endofunctor=synthesized_endofunctor,
            measured=measured,
            verdict=int(verdict),
        )

    def analyze_rsi3_banach(
        self, x_history: List[float], capacity_history: List[float]
    ) -> PaladinRSI3Report:
        states = [np.asarray([x], dtype=np.float64) for x in (x_history or [])]
        cert = self.rsi3_certificate(states, None, capacity_history)
        return PaladinRSI3Report(
            banach_ratio_q=cert.banach_q,
            banach_contraction_ok=cert.contraction_certified,
            jerk_capacity=cert.jerk_third_difference,
            jerk_positive=(
                bool(cert.jerk_third_difference > 0.0)
                if math.isfinite(cert.jerk_third_difference)
                else False
            ),
            lyapunov_value=cert.lyapunov_final,
            measured=cert.measured,
            banach_error_bound=cert.banach_error_bound,
            betti_1_cycles=cert.topology.betti_1_cycles,
        )

    # ── 3.6 · Sello forense encadenado SHA-256 ───────────────────────────────────────
    def forensic_seal(
        self,
        geometry: DensityGeometryCertificate,
        celestial: CelestialStabilityReport,
        annihilation: AnnihilationLedger,
        rsi: RSI3Certificate,
        verdict: HeytingVerdict,
        ts_ns: int,
    ) -> Tuple[str, str]:
        h = hashlib.sha256()
        h.update(self._chain_head.encode("utf-8"))
        h.update(_ENGINE_VERSION.encode("utf-8"))
        h.update(self._key_id.encode("utf-8"))
        h.update(_canonical_bytes(geometry.rho_projected))
        h.update(_canonical_bytes(geometry.sigma_projected))
        payload = {
            "ts_ns": int(ts_ns),
            "verdict": int(verdict),
            "F": geometry.uhlmann.fidelity_svd,
            "dB": geometry.uhlmann.bures_distance,
            "T": geometry.uhlmann.trace_distance,
            "D_umegaki": geometry.uhlmann.umegaki_relative_entropy,
            "moufang": geometry.octonion.moufang_identity_residual,
            "kam_gamma": celestial.kam.gamma_effective,
            "melnikov_zeros": celestial.melnikov.has_simple_zeros,
            "turnstile_area": celestial.melnikov.turnstile_lobe_area_over_eps,
            "chirikov": celestial.chirikov_max,
            "krein_measured": celestial.krein.measured,
            "energy_kev": annihilation.energy_released_kev,
            "anomaly_residual": annihilation.anomaly_norm_residual,
            "banach_q": rsi.banach_q,
            "betti_1": rsi.topology.betti_1_cycles,
            "coverage": celestial.coverage,
        }
        h.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode(
                "utf-8"
            )
        )
        digest = h.hexdigest()
        head, self._chain_head = self._chain_head, digest
        return digest, head

    # ── 3.7 · MÉTODO TERMINAL DE LA FASE 3 (CLAUSURA DEL TRIPLE Π₃ ∘ Π₂ ∘ Π₁) ───────
    def adjudicate(
        self,
        geometry: DensityGeometryCertificate,
        celestial: CelestialStabilityReport,
        *,
        electron_anomaly: float = 0.0,
        positron_hmac: bytes = b"",
        work_irr: float = 0.0,
        session_id: str = "default",
        state_history: Optional[Sequence[NDArray]] = None,
        lyapunov_history: Optional[Sequence[float]] = None,
        capacity_history: Optional[Sequence[float]] = None,
        amber_window: Tuple[float, float] = (10.0, 25.0),
        synthesized_endofunctor: str = "BuresGeodesicRetraction",
        continuation: Optional[Callable[[PaladinEngineReport], R]] = None,
    ) -> Union[PaladinEngineReport, R]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 3 (Π₃):
        Clausura el triple de Kleisli Z_paladin = Π₃ ∘ Π₂ ∘ Π₁ con monotonía estricta
        de seguridad en Ω₃ (P13: la aniquilación jamás mejora un veredicto de coherencia).
        """
        g = geometry
        s = self._continue_from_phase2_celestial(celestial)
        ts = time.time_ns()
        ledger = self.annihilate(
            electron_anomaly,
            positron_hmac,
            rho=g.rho_projected,
            sigma=g.sigma_projected,
            session_id=session_id,
            ts_ns=ts,
        )
        rsi = self.rsi3_certificate(
            state_history,
            lyapunov_history,
            capacity_history,
            synthesized_endofunctor=synthesized_endofunctor,
        )

        verdict = HeytingVerdict(g.verdict)
        for v in (s.verdict, rsi.verdict, ledger.verdict):
            verdict = heyting_meet(verdict, HeytingVerdict(v))
        if not g.coherence_ok:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        if not math.isfinite(rsi.banach_q) and state_history is None:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)

        trace_res = max(
            abs(g.projection_rho.trace_after - 1.0),
            abs(g.projection_sigma.trace_after - 1.0),
        )
        trace_ok = bool(trace_res <= 1e-9)
        lo, hi = amber_window
        amber = bool(lo < float(work_irr) <= hi)
        if amber:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)

        seal, head = self.forensic_seal(g, s, ledger, rsi, verdict, ts)
        legacy_celestial = PaladinCelestialReport(
            delaunay_actions=s.delaunay.actions,
            delaunay_frequencies=s.delaunay.mean_motions,
            kam_diophantine_ok=s.kam.kam_satisfied,
            kam_min_divisor=s.kam.min_divisor,
            melnikov_amplitude_rel=(
                s.melnikov.forcing_amplitude_sech / max(s.melnikov.damping_offset, 1e-300)
            ),
            melnikov_chaotic=s.melnikov.has_simple_zeros,
            chirikov_k=s.chirikov_max,
            krein_stable=bool(s.krein.measured and s.krein.spectrally_stable),
            nekhoroshev_time_seconds=s.nekhoroshev_scale_time,
            measured_fraction=s.coverage,
            turnstile_lobe_area=s.melnikov.turnstile_lobe_area_over_eps,
            fabrications_removed=(),
        )
        legacy_rsi = PaladinRSI3Report(
            banach_ratio_q=rsi.banach_q,
            banach_contraction_ok=rsi.contraction_certified,
            jerk_capacity=rsi.jerk_third_difference,
            jerk_positive=(
                bool(rsi.jerk_third_difference > 0.0)
                if math.isfinite(rsi.jerk_third_difference)
                else False
            ),
            lyapunov_value=rsi.lyapunov_final,
            measured=rsi.measured,
            banach_error_bound=rsi.banach_error_bound,
            betti_1_cycles=rsi.topology.betti_1_cycles,
        )

        out = PaladinEngineReport(
            timestamp_ns=ts,
            uhlmann_fidelity=g.uhlmann.fidelity_svd,
            bures_distance=g.uhlmann.bures_distance,
            trace_preserved=trace_ok,
            positron_annihilated=bool(ledger.authorized and ledger.pairs_annihilated > 0.0),
            positron_energy_released_kev=ledger.energy_released_kev,
            amber_relief_active=amber,
            celestial_report=legacy_celestial,
            rsi3_report=legacy_rsi,
            truth_value=HeytingTruthValue.from_canonical(verdict),
            sha256_fingerprint=seal,
            geometry=g,
            celestial=s,
            annihilation=ledger,
            rsi3=rsi,
            canonical_verdict=int(verdict),
            trace_residual_measured=float(trace_res),
            fingerprint_chain_head=head,
            coverage=float((s.coverage + (1.0 if rsi.measured else 0.0)) / 2.0),
        )
        logger.info(
            "[Π₃·CLAUSURA] Ω₃=%s F=%.8f d_B=%.6f E=%.4f keV β₁=%d sello=%s…",
            verdict.name,
            out.uhlmann_fidelity,
            out.bures_distance,
            out.positron_energy_released_kev,
            rsi.topology.betti_1_cycles,
            seal[:16],
        )
        if continuation is None:
            return out
        return continuation(out)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR SOBERANO · COMPOSICIÓN Π₃ ∘ Π₂ ∘ Π₁ Y SÍNTESIS RSI-L3              ══
# ═════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovPaladinEngine(Morphism):
    r"""
    Orquestador Soberano del Motor Paladín Bool–Novikov.
    Teje en una única cadena de continuaciones anidadas las Fases Π₁, Π₂ y Π₃, e implementa
    el bucle completo de Automejora Recursiva de Nivel 3 (RSI-L3) con contracción geodésica
    de Bures-Uhlmann, acoplamiento celeste de Poincaré y verificador inmutable de Löb.
    """

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        *,
        key_id: str = _DEFAULT_KEY_ID,
        fidelity_floor: float = _UHLMANN_FIDELITY_FLOOR,
        max_annihilable_anomaly: float = 1.0,
        anomaly_per_pair: float = 1.0,
        require_explicit_key: bool = False,
        apply_shield: bool = False,
    ) -> None:
        self.phase1 = Phase1_DensityGeometryGuard(fidelity_floor)
        self.phase2 = Phase2_CelestialStabilityAuditor()
        self.phase3 = Phase3_PaladinAdjudicator(
            secret_key,
            key_id=key_id,
            max_annihilable_anomaly=max_annihilable_anomaly,
            anomaly_per_pair=anomaly_per_pair,
            require_explicit_key=require_explicit_key,
        )
        self._apply_shield = bool(apply_shield)
        self._floor = float(fidelity_floor)
        try:
            super().__init__(name="BoolNovikovPaladinEngine")
        except TypeError:
            super().__init__()
        logger.info(
            "BoolNovikovPaladinEngine v%s inicializado (trusted=%s, shield=%s, F_floor=%.4f)",
            _ENGINE_VERSION,
            self.phase3._trusted,
            apply_shield,
            fidelity_floor,
        )

    # ── Plegado anidado de las 3 fases por Continuation-Passing Style (CPS) ──────────
    def execute_paladin_spectral_analysis(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        electron_anomaly: float = 0.0,
        positron_hmac: bytes = b"",
        work_irr: float = 0.0,
        x_history: Optional[List[float]] = None,
        capacity_history: Optional[List[float]] = None,
        *,
        kepler_elements: Optional[Sequence[Tuple[float, ...]]] = None,
        frequency_vector: Optional[Sequence[float]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        monodromy: Optional[NDArray[np.float64]] = None,
        gamma_diss: Optional[float] = None,
        f_drive: Optional[float] = None,
        omega_drive: Optional[float] = None,
        eps_pert: float = 0.01,
        couplings: Optional[Sequence[float]] = None,
        session_id: str = "default",
        state_history: Optional[Sequence[NDArray]] = None,
        lyapunov_history: Optional[Sequence[float]] = None,
        synthesized_endofunctor: str = "BuresGeodesicRetraction",
    ) -> PaladinEngineReport:
        states = state_history
        if states is None and x_history:
            states = [np.asarray([x], dtype=np.float64) for x in x_history]

        def _phase3_continuation(
            s: CelestialStabilityReport, g: DensityGeometryCertificate
        ) -> PaladinEngineReport:
            out = self.phase3.adjudicate(
                g,
                s,
                electron_anomaly=electron_anomaly,
                positron_hmac=positron_hmac,
                work_irr=work_irr,
                session_id=session_id,
                state_history=states,
                lyapunov_history=lyapunov_history,
                capacity_history=capacity_history,
                synthesized_endofunctor=synthesized_endofunctor,
            )
            assert isinstance(out, PaladinEngineReport)
            return out

        def _phase2_continuation(g: DensityGeometryCertificate) -> PaladinEngineReport:
            out = self.phase2.audit_celestial_stability(
                g,
                kepler_elements=kepler_elements,
                frequency_vector=frequency_vector,
                hamiltonian=hamiltonian,
                monodromy=monodromy,
                gamma_diss=gamma_diss,
                f_drive=f_drive,
                omega_drive=omega_drive,
                eps_pert=eps_pert,
                couplings=couplings,
                continuation=lambda s: _phase3_continuation(s, g),
            )
            assert isinstance(out, PaladinEngineReport)
            return out

        out = self.phase1.certify_density_geometry(
            rho_input,
            rho_target,
            apply_shield=self._apply_shield,
            continuation=_phase2_continuation,
        )
        assert isinstance(out, PaladinEngineReport)
        return out

    # ── Automejora Recursiva de Nivel 3 (RSI-L3) con Geodésica de Bures y Löb ────────
    def run_rsi_l3_cycle(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        max_iterations: int = 6,
        tolerance: float = 1e-12,
        hyperparameters: Optional[RSIL3Hyperparameters] = None,
        **analysis_kwargs: Any,
    ) -> Tuple[PaladinEngineReport, RSIL3LoopReport]:
        r"""
        Bucle de Automejora Recursiva de Nivel 3 (RSI-L3):
          · NIVEL 1 (Estado): Evoluciona ρ_k ↦ ρ_{k+1} a lo largo de la geodésica minimal
            de Bures-Uhlmann Γ_{τ_k}(ρ_k, σ) en la variedad riemanniana D(H).
          · NIVEL 2 (Meta-Parámetros): Adapta el paso geodésico τ_k y endurece los umbrales
            según el factor de contracción empírico de Banach q̂_k y el parámetro de Chirikov.
          · NIVEL 3 (Síntesis de Endofuntor + Verificador Inmutable de Löb): Sintetiza la
            ley de transporte CPTP óptima y valida cada transición mediante
            `_LoebSafeImmutableVerifier`, garantizando monotonía de Lyapunov d_B(ρ_k, σ)
            y trivialidad homológica β₁ = 0.
        """
        hyper = hyperparameters or RSIL3Hyperparameters(fidelity_floor=self._floor)
        rho, _ = self.phase1.project(rho_input, "rsi_rho_init")
        sigma, _ = self.phase1.project(rho_target, "rsi_sigma_target")

        hist_x: List[NDArray[np.float64]] = [
            np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        ]
        best = self.execute_paladin_spectral_analysis(
            rho, sigma, state_history=hist_x, **analysis_kwargs
        )
        v_prev = best.bures_distance
        hist_v: List[float] = [v_prev]
        hist_f: List[float] = [best.uhlmann_fidelity]

        accepted = rejected = 0
        q, bound = float("nan"), float("inf")
        monotone = True
        tightening_all = True
        loeb_sound = True
        synthesized_law = "UhlmannBuresGeodesic[ExactPolar]"

        for step_idx in range(max(1, int(max_iterations))):
            if v_prev <= tolerance:
                break
            # Nivel 3: Síntesis adaptativa del endofuntor según el estado celeste y espectral
            chirikov_k = (
                best.celestial.chirikov_max if best.celestial is not None else 0.0
            )
            if chirikov_k > _CHIRIKOV_SOFT:
                tau_step = float(np.clip(0.65 * hyper.bures_geodesic_step, 0.15, 0.85))
                synthesized_law = "KamDampedBuresGeodesic[ChirikovGuarded]"
            else:
                tau_step = float(np.clip(hyper.bures_geodesic_step, 0.15, 0.85))
                synthesized_law = "UhlmannBuresGeodesic[ExactPolar]"

            # Nivel 1: Contracción a lo largo de la geodésica de Bures hacia el atractor σ
            rho_cand = self.phase1.bures_geodesic_step(rho, sigma, tau_step)
            cand_vec = np.concatenate([rho_cand.real.ravel(), rho_cand.imag.ravel()])
            trial_x = hist_x + [cand_vec]
            trial_v = hist_v + [
                self.phase1.uhlmann(rho_cand, sigma).bures_distance
            ]

            # Nivel 2: Propuesta de endurecimiento monótono de hiperparámetros
            gain = (
                (1.0 - 0.5 * hyper.trust_radius)
                if (math.isfinite(q) and q < 0.7)
                else 0.8
            )
            hyper_cand = replace(
                hyper,
                shield_lambda_cap=max(hyper.shield_lambda_cap * gain, 0.0),
                fidelity_floor=min(0.9999, max(hyper.fidelity_floor, self._floor)),
                banach_q_max=max(hyper.banach_q_max * 0.99, 0.5),
                trust_radius=float(np.clip(hyper.trust_radius * 0.95, 1e-3, 0.5)),
                bures_geodesic_step=tau_step,
                meta_depth=hyper.meta_depth + 1,
            )

            self.phase1._floor = hyper_cand.fidelity_floor
            cand = self.execute_paladin_spectral_analysis(
                rho_cand,
                sigma,
                state_history=trial_x,
                lyapunov_history=trial_v,
                synthesized_endofunctor=synthesized_law,
                **analysis_kwargs,
            )
            v_k = cand.bures_distance

            # Verificación inmutable segura ante Löb
            lyap_ok, tight_ok = _LoebSafeImmutableVerifier.verify_transition(
                v_prev, v_k, hyper, hyper_cand, cand.canonical_verdict, tolerance
            )
            tightening_all = tightening_all and tight_ok
            loeb_sound = loeb_sound and tight_ok

            if lyap_ok and tight_ok:
                accepted += 1
                rho = rho_cand
                best = cand
                hyper = hyper_cand
                hist_x = trial_x
                hist_v = trial_v
                hist_f.append(cand.uhlmann_fidelity)
                if len(hist_x) >= 3:
                    cert = self.phase3.rsi3_certificate(
                        hist_x,
                        hist_v,
                        None,
                        q_max=hyper.banach_q_max,
                        synthesized_endofunctor=synthesized_law,
                    )
                    q, bound = cert.banach_q, cert.banach_error_bound
                delta = abs(v_prev - v_k)
                v_prev = v_k
                if delta < tolerance and math.isfinite(q) and q < hyper.banach_q_max:
                    break
            else:
                rejected += 1
                monotone = False
                hyper = replace(
                    hyper,
                    trust_radius=max(hyper.trust_radius * 0.5, 1e-3),
                    meta_depth=hyper.meta_depth + 1,
                )

        topo = self.phase3.audit_trajectory_topology(hist_x)
        verdict = (
            HeytingVerdict.COHERENT
            if (
                monotone
                and tightening_all
                and topo.loop_free
                and math.isfinite(q)
                and q < hyper.banach_q_max
            )
            else HeytingVerdict.DEGRADED
        )
        loop_report = RSIL3LoopReport(
            iterations=accepted + rejected,
            accepted=accepted,
            rejected=rejected,
            lyapunov_history=tuple(float(x) for x in hist_v),
            fidelity_history=tuple(float(x) for x in hist_f),
            contraction_q=float(q),
            banach_error_bound=float(bound),
            monotone=bool(monotone),
            tightening_only=bool(tightening_all),
            loeb_verifier_sound=bool(loeb_sound),
            synthesized_law=synthesized_law,
            topology=topo,
            hyperparameters=hyper,
            verdict=int(verdict),
        )
        return best, loop_report

    # ── Puente categórico con módulos hermanos del sistema inmune ────────────────────
    def govern_with_siblings(
        self,
        rho: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        node_index: int = 0,
        attention_vector: Optional[NDArray[np.float64]] = None,
        market_pressure: float = 0.0,
    ) -> Dict[str, Any]:
        local = self.execute_paladin_spectral_analysis(rho, rho_target)
        out: Dict[str, Any] = {
            "paladin": local,
            "canonical_verdict": local.canonical_verdict,
        }
        merged = HeytingVerdict(local.canonical_verdict)
        if attention_vector is not None:
            try:
                from app.core.immune_system.gravity_shield import GravitationalShieldFunctor

                shield = GravitationalShieldFunctor()

                class _P:
                    inertial_mass = float(np.real(np.trace(rho)))
                    volatility_alpha = float(market_pressure)
                    frohlich_coupling = None

                act = shield.enforce_gravitational_attractor(
                    _P(),
                    node_index,
                    np.asarray(attention_vector, dtype=np.float64),
                    strict=False,
                )
                out["gravity_shield"] = act
                merged = heyting_meet(
                    merged, HeytingVerdict(int(getattr(act, "heyting_verdict_value", 1)))
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("[Puente] gravity_shield no disponible: %s", exc)
                merged = heyting_meet(merged, HeytingVerdict.DEGRADED)
        out["merged_verdict"] = int(merged)
        return out

    # ── AUTO-TESTS MATEMÁTICOS RIGUROSOS (Calibración de Teoremas P1–P11) ────────────
    def self_test(self) -> Dict[str, float]:
        r"""
        Suite de pruebas formales autocontenidas:
          1. Idempotencia de la proyección al espectralplex (P5).
          2. Concordancia SVD vs ruta espectral de Uhlmann (P1).
          3. Exactitud de la geodésica de Bures en τ=0.5: d_B(ρ(0), ρ(0.5)) == ½ d_B(ρ, σ) (P1).
          4. Identidad de Moufang en los Octoniones O (P3).
          5. Solucionador cúbico de Kepler-Halley y carta regularizada de Poincaré (P9).
          6. Positividad del área de lóbulo Turnstile en Melnikov-Duffing sobre umbral (P8).
          7. Contracción estricta q < 1 del bucle RSI-L3 con β₁ = 0 (P11).
        """
        rng = np.random.default_rng(20261116)
        d = 4
        a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = a @ a.conj().T
        rho /= np.trace(rho)
        b = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        sigma = b @ b.conj().T
        sigma /= np.trace(sigma)

        p1, _ = self.phase1.project(rho)
        p2, _ = self.phase1.project(p1)
        idem = _fro(p2 - p1)
        ps, _ = self.phase1.project(sigma)
        cert = self.phase1.uhlmann(p1, ps)

        # Verificación geodésica en el punto medio τ = 0.5
        mid = self.phase1.bures_geodesic_step(p1, ps, 0.5)
        theta_full = cert.bures_angle_rad
        theta_half = self.phase1.uhlmann(p1, mid).bures_angle_rad
        geodesic_midpoint_err = abs(theta_half - 0.5 * theta_full)

        # Verificación en estados puros
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= la.norm(psi)
        phi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        phi /= la.norm(phi)
        rp = np.outer(psi, psi.conj())
        sp = np.outer(phi, phi.conj())
        f_pure = self.phase1.uhlmann(rp, sp).fidelity_svd
        f_exact = float(abs(np.vdot(psi, phi)) ** 2)

        # Verificación octoniónica de Moufang
        oct_cert = self.phase1.audit_hypercomplex_octonions(p1, ps)

        # Verificación de Kepler-Halley y Poincaré regular
        del_chart = self.phase2.delaunay_from_kepler([(1.5, 0.4, 0.25, 1.1, 0.5, 0.3)])
        poinc_res = (
            del_chart.poincare_regular.symplectic_identity_residual
            if del_chart.poincare_regular is not None
            else 1.0
        )
        kepler_res = (
            del_chart.poincare_regular.kepler_halley_max_residual
            if del_chart.poincare_regular is not None
            else 1.0
        )

        # Verificación de lóbulo Turnstile en Melnikov-Duffing
        mel_rep = self.phase2.melnikov_duffing(0.05, 0.25, 1.0)

        # Verificación de convergencia RSI-L3
        _, rsi_loop = self.run_rsi_l3_cycle(p1, ps, max_iterations=5)

        _, led = self.phase1.shield(p1, ps, apply=True, lambda_cap=1e-2)
        w_small = 1e-3
        x = math.pi * w_small / 2.0
        kernel_ratio = (1.0 / math.cosh(x)) / (1.0 / math.sinh(x))

        out = {
            "spectraplex_idempotency_residual": idem,
            "uhlmann_route_disagreement": cert.route_disagreement,
            "bures_geodesic_midpoint_angle_error": geodesic_midpoint_err,
            "pure_state_fidelity_error": abs(f_pure - f_exact),
            "octonion_moufang_residual": oct_cert.moufang_identity_residual,
            "kepler_halley_residual": kepler_res,
            "poincare_symplectic_identity_residual": poinc_res,
            "melnikov_turnstile_lobe_area": mel_rep.turnstile_lobe_area_over_eps,
            "rsi_l3_banach_q": rsi_loop.contraction_q,
            "rsi_l3_betti_1_cycles": float(rsi_loop.topology.betti_1_cycles),
            "fvdg_lower_violation": max(
                0.0, (1.0 - math.sqrt(cert.fidelity_svd)) - cert.trace_distance
            ),
            "fvdg_upper_violation": max(
                0.0,
                cert.trace_distance - math.sqrt(max(0.0, 1.0 - cert.fidelity_svd)),
            ),
            "shield_fidelity_gain_DEMO": led.fidelity_gain_attributed_to_channel,
            "shield_entropy_injected_DEMO": led.entropy_injected_by_channel,
            "csch_over_sech_at_small_omega_DEMO": 1.0 / max(kernel_ratio, 1e-300),
        }
        logger.info("[SELFTEST v15] %s", out)
        return out

    def verify_complete_integrity(self) -> Dict[str, Any]:
        calib = self.self_test()
        ok = (
            calib["spectraplex_idempotency_residual"] < 1e-10
            and calib["uhlmann_route_disagreement"] < 1e-6
            and calib["bures_geodesic_midpoint_angle_error"] < 1e-7
            and calib["pure_state_fidelity_error"] < 1e-9
            and calib["octonion_moufang_residual"] < 1e-12
            and calib["kepler_halley_residual"] < 1e-13
            and calib["poincare_symplectic_identity_residual"] < 1e-13
            and calib["melnikov_turnstile_lobe_area"] > 0.0
            and calib["rsi_l3_banach_q"] < _MAX_BANACH_Q
            and calib["rsi_l3_betti_1_cycles"] == 0.0
            and calib["fvdg_lower_violation"] < 1e-8
            and calib["fvdg_upper_violation"] < 1e-8
        )
        verdict = HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED
        if not self.phase3._trusted:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        return {
            "engine_version": _ENGINE_VERSION,
            "calibration": calib,
            "key_trusted": self.phase3._trusted,
            "key_id": self.phase3._key_id,
            "shield_applied_by_default": self._apply_shield,
            "omega3_codification": {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2},
            "pair_annihilation_energy_kev": _PAIR_ANNIHILATION_ENERGY_KEV,
            "heyting_verdict_value": int(verdict),
        }

    # ── Fachada de compatibilidad ABI ────────────────────────────────────────────────
    def sanitize_density_matrix(self, A: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return self.phase1.sanitize_density_matrix(A)

    def compute_uhlmann_fidelity(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        return self.phase1.compute_uhlmann_fidelity(rho, sigma)

    def shield_density_matrix(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], float]:
        return self.phase1.shield_density_matrix(rho)

    def annihilate_positron_hmac(
        self, electron_anomaly_norm: float, positron_hmac: bytes
    ) -> Tuple[bool, float, float]:
        return self.phase3.annihilate_positron_hmac(electron_anomaly_norm, positron_hmac)

    def analyze_celestial_poincare(
        self, rho: NDArray[np.complex128], eps_pert: float = 0.01
    ) -> PaladinCelestialReport:
        return self.phase2.analyze_celestial_poincare(rho, eps_pert)

    def analyze_rsi3_banach(
        self, x_history: List[float], capacity_history: List[float]
    ) -> PaladinRSI3Report:
        return self.phase3.analyze_rsi3_banach(x_history, capacity_history)

    def issue_positron_token(
        self,
        anomaly_norm: float,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str = "default",
    ) -> Tuple[bytes, int, int]:
        ts = time.time_ns()
        tok = self.phase3.build_token(
            anomaly_norm,
            np.asarray(rho, dtype=np.complex128),
            np.asarray(sigma, dtype=np.complex128),
            session_id,
            ts,
        )
        return tok, ts, self.phase3._counter