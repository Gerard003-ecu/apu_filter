# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : ToonCartridgeBase + Mecánica Celeste de Poincaré + Automejora Nivel 3                 ║
║ Ubicación: app/wisdom/toon_cartridge_base.py                                                     ║
║ Versión  : 11.0.0-SOVEREIGN-TOON-POINCARE-KAM-L3-PhD                                             ║
║ Función  : 12 Partículas Fundamentales con Secciones de Poincaré, Tori KAM Invariantes           ║
║            (KAM disipativo BHT) y Operador de Automejora Recursiva Nivel 3                       ║
║            (Metacognición Metacognitiva con realimentación del mapa de retorno).                 ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.   ║
║   FASE 1 → Embedding Canónico (Action-Angle) + Invariante Integral de Poincaré-Cartan            ║
║   FASE 2 → Secciones de Poincaré + Twist de Poincaré-Birkhoff + KAM + Operadores L3              ║
║   FASE 3 → Integración, Auditoría KAM Topológica y Factoría Soberana                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.ToonCartridgeBase")

__version__: Final[str] = "11.0.0-SOVEREIGN-TOON-POINCARE-KAM-L3-PhD"

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, numéricas FPU y umbrales KAM / Poincaré
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_DEFAULT_HILBERT_DIM: Final[int] = 56
_ELECTRON_MASS_KEV: Final[float] = 511.0
_POSITRON_REST_ENERGY_KEV: Final[float] = 1022.0
_GAUGE_SECRET_KEY: Final[bytes] = b"APU_FILTER_TOON_CARTRIDGE_BASE_SECRET_KEY"
_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_MAX_K: Final[int] = 12
_KAM_LATTICE_MODES: Final[int] = 6
_L3_DEFAULT_ITERS: Final[int] = 8
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_TRANSVERSALITY_TOL: Final[float] = 1.0e-10
_RESONANCE_MAX_Q: Final[int] = 8
_RESONANCE_TOL: Final[float] = 0.02
_BHT_DISSIPATION_CEILING: Final[float] = 0.95


class ParticleCategory(Enum):
    """Categorización cuántica de las 12 partículas fundamentales."""

    FERMION_CAR = auto()  # 5 Fermiones Estructurales (Anticonmutación CAR)
    BOSON_CCR = auto()  # 6 Bosones de Gauge (Conmutación CCR)
    ANTIMATTER = auto()  # 1 Familia de Antimateria y Radiación (Aniquilación en Dirac)


def _safe_unit(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Normalización numéricamente estable; fallback e_0 si ||v|| ~ 0."""
    nrm = float(np.linalg.norm(vector))
    if nrm <= _WILKINSON_EPSILON:
        out = np.zeros_like(vector)
        out.reshape(-1)[0] = 1.0 + 0.0j
        return out
    return vector / nrm


def _is_near_resonance(rho: float, max_q: int = _RESONANCE_MAX_Q, tol: float = _RESONANCE_TOL) -> bool:
    """
    Aproximación de Farey: ρ ≈ p/q con q ≤ max_q.
    Resonancias racionales del mapa de Poincaré destruyen tori KAM (divisores pequeños).
    """
    if not math.isfinite(rho):
        return True
    frac = rho - math.floor(rho)
    for q in range(1, max_q + 1):
        p = round(frac * q)
        if abs(frac - (p / q)) < (tol / q):
            return True
    return False


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: FUNDACIÓN INMUTABLE + EMBEDDING ACTION-ANGLE                                            ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La ÚLTIMA definición de FASE 1 (ToonCartridge.to_action_angle_seed) retorna un objeto del      ║
# ║  tipo ActionAngleSeed, que es la PRIMERA definición consumida por FASE 2. La firma de retorno    ║
# ║  de to_action_angle_seed ES la firma de apertura de la clase ActionAngleSeed.__init__.          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class ToonCartridge:
    """
    Patrón Polimórfico Inmutable Base para las 12 partículas del Espacio de Fock.
    Inmutabilidad estricta vía (frozen=True, slots=True) → huella CPython optimizada.
    """

    cartridge_id: str
    category: ParticleCategory
    particle_name: str
    mass_energy_kev: float
    signature_hmac: str
    timestamp_utc: float
    payload_tuple: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def quantum_state_hash(self) -> str:
        """
        Firma SHA-256 determinista e inmutable O(1). Identifica unívocamente
        el estado en la Malla Agéntica sin desempaquetar el payload.
        """
        raw_str = (
            f"{self.cartridge_id}:{self.category.name}:{self.particle_name}:"
            f"{self.mass_energy_kev:.6f}:{self.signature_hmac}:{self.timestamp_utc:.6f}:"
            f"{self.payload_tuple}"
        )
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def to_hilbert_state_vector(
        self, dimension: int = _DEFAULT_HILBERT_DIM
    ) -> NDArray[np.complex128]:
        """
        Proyecta la partícula a un vector unitario |v⟩ ∈ S^(2N-1) ⊂ ℂ^N
        en el Espacio de Hilbert de la Matriz Atómica de Conocimiento (MAC).

        Semilla: 64 bits del hash cuántico → generador PCG64 (reproducible).
        """
        dim = max(4, int(dimension))
        seed_val = int(self.quantum_state_hash[:16], 16)
        rng = np.random.default_rng(seed_val)

        real_part = rng.normal(0.0, 1.0, size=(dim, 1))
        imag_part = rng.normal(0.0, 1.0, size=(dim, 1))
        v_vector = (real_part + 1j * imag_part).astype(np.complex128)
        return _safe_unit(v_vector)

    def verify_hmac_authenticity(self, secret_key: bytes = _GAUGE_SECRET_KEY) -> bool:
        """Verifica autenticidad HMAC-SHA256 (comparación en tiempo constante)."""
        seed = (
            f"{self.cartridge_id}:{self.particle_name}:"
            f"{self.mass_energy_kev:.6f}:{self.timestamp_utc:.6f}"
        )
        expected = hmac.new(secret_key, seed.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)

    def fock_occupation_hint(self, dimension: int = _DEFAULT_HILBERT_DIM) -> Tuple[float, ...]:
        """Números de ocupación n_k = |v_k|² del embedding de Fock (antes del mapa canónico)."""
        v = self.to_hilbert_state_vector(dimension).flatten()
        return tuple(float(x) for x in (np.abs(v) ** 2))

    def symplectic_norm(self, dimension: int = _DEFAULT_HILBERT_DIM) -> float:
        """
        Norma inducida por la forma simpléctica estándar ω = dq ∧ dp sobre ℂ^N ≅ T*ℝ^N.
        Para |v⟩ unitario, ||v||_ω² = ||v||² = 1 (identidad de polarización de Bargmann).
        """
        v = self.to_hilbert_state_vector(dimension).flatten()
        return float(np.real(np.vdot(v, v)))

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: EMERGENCIA DEL SEED CANÓNICO                                               ║
    # ║  Este es el ÚLTIMO método de FASE 1. Retorna un ActionAngleSeed — el PRIMER objeto           ║
    # ║  consumido por FASE 2. La firma del retorno continúa directamente en la definición de        ║
    # ║  la clase ActionAngleSeed.__init__ (ver bloque siguiente).                                   ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def to_action_angle_seed(
        self, dimension: int = _DEFAULT_HILBERT_DIM
    ) -> "ActionAngleSeed":
        r"""
        Transformación canónica (q, p) ↦ (J, θ) sobre el vector de Hilbert:

            a_k = v_k ∈ ℂ                         (amplitud coherente del modo k)
            J_k = |a_k|²                          (acción del modo k)
            θ_k = arg(a_k)                        (ángulo del modo k)
            ω_k = √(1+k) + (φ−1)/(1+k)            (frecuencias FPU ⊕ cascade áureo)

        La perturbación áurea rompe conmensurabilidades residuales de √(1+k) y
        mejora la constante diofantina γ (condición de Kolmogorov).

        Conservación: Σ_k J_k = ||v||² = 1  (invariante integral de Poincaré I₁).
        Energía canónica: H = Σ_k ω_k J_k   (Hamiltoniano del oscilador armónico).
        Forma simpléctica: dq ∧ dp = dθ ∧ dJ.

        Retorna el ActionAngleSeed que INICIA FASE 2.
        """
        dim = max(4, int(dimension))
        v = self.to_hilbert_state_vector(dim).flatten()
        J_vec = np.abs(v) ** 2
        theta_vec = np.angle(v)

        k_idx = np.arange(dim, dtype=np.float64)
        omega_vec = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)

        J_total = float(J_vec.sum())
        if J_total > _WILKINSON_EPSILON:
            J_vec = J_vec / J_total
            J_total = 1.0

        E_total = float((omega_vec * J_vec).sum())
        dominant = int(np.argmax(J_vec))

        return ActionAngleSeed(
            source_hash=self.quantum_state_hash,
            J_vector=tuple(float(x) for x in J_vec),
            theta_vector=tuple(float(x) for x in theta_vec),
            omega_vector=tuple(float(x) for x in omega_vec),
            J_total=J_total,
            E_total=E_total,
            theta_dominant=float(theta_vec[dominant]),
            omega_dominant=float(omega_vec[dominant]),
            dimension=dim,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: MECÁNICA CELESTE DE POINCARÉ + AUTOMEJORA RECURSIVA NIVEL 3                             ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La PRIMERA definición (ActionAngleSeed) consume el retorno de FASE 1.                          ║
# ║  La ÚLTIMA definición (RecursiveImprovementOperator.iterate_level_3) retorna un objeto del      ║
# ║  tipo ImprovementTrajectory, que es la PRIMERA definición de FASE 3.                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class ActionAngleSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRY: Continúa de FASE 1 (to_action_angle_seed).                                 │
    │  Semilla canónica (J_k, θ_k, ω_k) en el espacio de fases del oscilador armónico.            │
    │  Base de todos los análisis de Poincaré / KAM / L3 que siguen.                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_hash: str
    J_vector: Tuple[float, ...]
    theta_vector: Tuple[float, ...]
    omega_vector: Tuple[float, ...]
    J_total: float
    E_total: float
    theta_dominant: float
    omega_dominant: float
    dimension: int

    def hamiltonian(self) -> float:
        """H(J) = Σ_k ω_k J_k — energía del toro isócrono."""
        return float(self.E_total)

    def poincare_cartan_invariant(self) -> float:
        r"""
        Primer invariante integral de Poincaré-Cartan:
            I₁ = ∮_γ p dq = Σ_k J_k   (en coordenadas acción-ángulo).
        """
        return float(self.J_total)

    def frequency_ratios(self) -> Tuple[float, ...]:
        """Razones de enrollamiento ω_k / ω_0 (números de rotación relativos)."""
        if abs(self.omega_dominant) <= _WILKINSON_EPSILON:
            return tuple(0.0 for _ in self.omega_vector)
        return tuple(float(w / self.omega_vector[0]) for w in self.omega_vector)

    def to_canonical_pair(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Par canónico (q_k, p_k) de la transformación de Whittaker:
            q_k = √(2 J_k / ω_k) · cos(θ_k)
            p_k = −√(2 J_k ω_k) · sin(θ_k)

        Identidad hamiltoniana: (p_k² + ω_k² q_k²) / (2 ω_k) = J_k.  ✓
        """
        J = np.array(self.J_vector, dtype=np.float64)
        theta = np.array(self.theta_vector, dtype=np.float64)
        omega = np.array(self.omega_vector, dtype=np.float64)
        omega_safe = np.maximum(omega, _WILKINSON_EPSILON)
        J_safe = np.maximum(J, _WILKINSON_EPSILON)
        q = np.sqrt(2.0 * J_safe / omega_safe) * np.cos(theta)
        p = -np.sqrt(2.0 * J_safe * omega_safe) * np.sin(theta)
        return q, p

    def hamiltonian_identity_residue(self) -> float:
        """Residuo max_k |(p² + ω² q²)/(2ω) − J| — debe ser O(ε_máquina)."""
        q, p = self.to_canonical_pair()
        J = np.array(self.J_vector, dtype=np.float64)
        omega = np.maximum(np.array(self.omega_vector, dtype=np.float64), _WILKINSON_EPSILON)
        reconstructed = (p**2 + (omega**2) * (q**2)) / (2.0 * omega)
        return float(np.max(np.abs(reconstructed - J)))

    def to_improvement_vector(self) -> NDArray[np.complex128]:
        r"""Amplitud coherente a_k = √J_k · e^{i θ_k} → vector de fase complejo."""
        J = np.array(self.J_vector, dtype=np.float64)
        theta = np.array(self.theta_vector, dtype=np.float64)
        return (np.sqrt(np.maximum(J, _WILKINSON_EPSILON)) * np.exp(1j * theta)).astype(
            np.complex128
        )

    def winding_signature(self) -> float:
        """Invariante topológico ω_dom / Σ ω (número de enrollamiento dominante)."""
        omega_arr = np.array(self.omega_vector, dtype=np.float64)
        total = float(np.sum(omega_arr))
        return float(self.omega_dominant / total) if total > 0.0 else 0.0

    def generating_function_S(self) -> float:
        r"""
        Función generatriz de tipo 2, S(q, θ) ~ Σ_k J_k θ_k, evaluada en el seed.
        Genera la transformación canónica vía p = ∂S/∂q, J = ∂S/∂θ.
        """
        J = np.array(self.J_vector, dtype=np.float64)
        theta = np.array(self.theta_vector, dtype=np.float64)
        return float(np.dot(J, theta))


@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección de Poincaré transversal Σ = { z : g(z) = Re⟨n, z⟩ − c = 0 }.
    Requiere transversalidad ∇g · f ≠ 0 para que Σ corte el flujo limpiamente.
    """

    normal: Tuple[float, ...]
    offset: float
    label: str = "Σ"

    def g(self, z: NDArray[np.complex128]) -> float:
        r"""Función de sección g(z) = Re⟨n, z⟩ − c  (producto hermítico)."""
        n = np.array(self.normal, dtype=np.complex128)
        z_flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        n_flat = n.reshape(-1)
        m = min(n_flat.size, z_flat.size)
        return float(np.real(np.vdot(n_flat[:m], z_flat[:m])) - self.offset)

    def transversality(
        self, vector_field: NDArray[np.complex128], tol: float = _TRANSVERSALITY_TOL
    ) -> bool:
        """Verifica ∇g · f ≠ 0 (corte transversal del flujo por la sección)."""
        n = np.array(self.normal, dtype=np.complex128)
        f = np.asarray(vector_field, dtype=np.complex128).reshape(-1)
        n_flat = n.reshape(-1)
        m = min(n_flat.size, f.size)
        return abs(float(np.real(np.vdot(n_flat[:m], f[:m])))) > tol

    def oriented_flux(self, vector_field: NDArray[np.complex128]) -> float:
        """Flujo orientado ⟨n, f⟩: signo = sentido del cruce Σ⁺ → Σ⁻."""
        n = np.array(self.normal, dtype=np.complex128)
        f = np.asarray(vector_field, dtype=np.complex128).reshape(-1)
        m = min(n.size, f.size)
        return float(np.real(np.vdot(n.reshape(-1)[:m], f[:m])))


class PoincareReturnMap:
    r"""
    Mapa de retorno P : Σ → Σ inducido por el flujo de automejora L3.
    Análogo al análisis de Poincaré de sistemas celestes (problema restringido),
    aplicado aquí a la órbita discreta del operador de mejora.

    El número de rotación de Birkhoff
        ρ = lim_{n→∞} (θ_n − θ_0) / (2π n)
    clasifica la dinámica del retorno: racional → periódica; irracional → cuasiperiódica.
    """

    def __init__(self, section: PoincareSection) -> None:
        self.section = section

    def detect_crossings(
        self, trajectory: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        """
        Detecta cruces transversales: g(z_i) · g(z_{i+1}) < 0 → interpolación lineal
        en el parámetro afín, aceptada sólo si el campo discreto Δz es transversal a Σ.
        """
        crossings: List[NDArray[np.complex128]] = []
        if len(trajectory) < 2:
            return crossings
        g_vals = [self.section.g(z) for z in trajectory]
        for i in range(len(g_vals) - 1):
            gi, gj = g_vals[i], g_vals[i + 1]
            if gi * gj >= 0.0:
                continue
            denom = gi - gj
            if abs(denom) <= _WILKINSON_EPSILON:
                continue
            t = gi / denom
            t = float(np.clip(t, 0.0, 1.0))
            zi = np.asarray(trajectory[i], dtype=np.complex128)
            zj = np.asarray(trajectory[i + 1], dtype=np.complex128)
            flow = zj - zi
            if not self.section.transversality(flow):
                continue
            z_cross = (1.0 - t) * zi + t * zj
            crossings.append(z_cross)
        return crossings

    def apply(
        self, trajectory: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        """Composición del mapa de retorno sobre los cruces transversales detectados."""
        return self.detect_crossings(trajectory)

    @staticmethod
    def birkhoff_rotation_number(crossings: Sequence[NDArray[np.complex128]]) -> float:
        r"""
        Número de rotación de Birkhoff del mapa de Poincaré:
            ρ = (θ_N − θ_0) / (2π N),   θ = arg(Σ_k z_k).
        Se usa unwrap para eliminar saltos de rama de ±π.
        """
        if len(crossings) < 2:
            return 0.0
        raw = [float(np.angle(np.sum(np.asarray(z, dtype=np.complex128)))) for z in crossings]
        angles = np.unwrap(np.array(raw, dtype=np.float64))
        n_steps = len(angles) - 1
        if n_steps <= 0:
            return 0.0
        return float((angles[-1] - angles[0]) / (_TWO_PI * n_steps))

    def mean_return_phase(self, crossings: Sequence[NDArray[np.complex128]]) -> float:
        """Fase media de los cruces (centroide angular sobre Σ)."""
        if not crossings:
            return 0.0
        acc = 0.0 + 0.0j
        for z in crossings:
            acc += np.exp(1j * float(np.angle(np.sum(np.asarray(z, dtype=np.complex128)))))
        return float(np.angle(acc)) if abs(acc) > _WILKINSON_EPSILON else 0.0


class PoincareBirkhoffTwist:
    r"""
    Teorema geométrico de Poincaré (Poincaré-Birkhoff):
    un mapa que preserva área en el anillo, con condición de twist
        ∂(Δθ)/∂J ≠ 0,
    posee al menos dos puntos fijos.

    Aquí se estima el twist sobre una nube de acciones perturbadas del seed.
    """

    def __init__(self, seed: ActionAngleSeed, delta_J: float = 1.0e-4) -> None:
        self.seed = seed
        self.delta_J = float(delta_J)

    def twist_derivative(self) -> float:
        """Estimación por diferencias centrales de ∂θ_dom / ∂J_dom."""
        J = np.array(self.seed.J_vector, dtype=np.float64)
        theta = np.array(self.seed.theta_vector, dtype=np.float64)
        omega = np.array(self.seed.omega_vector, dtype=np.float64)
        dom = int(np.argmax(J))
        dJ = self.delta_J

        def advance(J_vec: NDArray[np.float64]) -> float:
            return float((theta[dom] + omega[dom] * J_vec[dom]) % _TWO_PI)

        J_plus = J.copy()
        J_minus = J.copy()
        J_plus[dom] = min(1.0, J[dom] + dJ)
        J_minus[dom] = max(_WILKINSON_EPSILON, J[dom] - dJ)
        d_theta = advance(J_plus) - advance(J_minus)
        d_theta = (d_theta + math.pi) % _TWO_PI - math.pi
        denom = float(J_plus[dom] - J_minus[dom])
        if abs(denom) <= _WILKINSON_EPSILON:
            return 0.0
        return float(d_theta / denom)

    def satisfies_twist(self, tol: float = 1.0e-12) -> bool:
        return abs(self.twist_derivative()) > tol


class BirkhoffNormalForm:
    r"""
    Forma normal local de Birkhoff en un punto fijo z*:
        I(z) = z* + M(z − z*) + O(|z − z*|²),   M = DI(z*)
    Clasificación espectral de la dinámica local del mapa discreto L3.
    """

    def __init__(self, fixed_point: NDArray[np.complex128], jacobian: NDArray[np.complex128]) -> None:
        self.fixed_point = np.asarray(fixed_point, dtype=np.complex128)
        self.jacobian = np.asarray(jacobian, dtype=np.complex128)

    def eigenvalues(self) -> NDArray[np.complex128]:
        if self.jacobian.size == 0:
            return np.array([], dtype=np.complex128)
        try:
            return np.linalg.eigvals(self.jacobian)
        except np.linalg.LinAlgError:
            return np.array([], dtype=np.complex128)

    def spectral_radius(self) -> float:
        eigs = self.eigenvalues()
        return float(np.max(np.abs(eigs))) if eigs.size else 0.0

    def classify(self) -> str:
        eigs = self.eigenvalues()
        if eigs.size == 0:
            return "DEGENERATE"
        moduli = np.abs(eigs)
        if np.all(moduli < 1.0 - 1e-8):
            return "ATTRACTING"
        if np.all(moduli > 1.0 + 1e-8):
            return "REPELLING"
        if np.all(np.abs(moduli - 1.0) < 1e-8):
            return "ELLIPTIC_CENTER"
        return "SADDLE_HYPERBOLIC"

    def lyapunov_exponent(self) -> float:
        rho = self.spectral_radius()
        return float(np.log(max(rho, _WILKINSON_EPSILON)))

    def twist_eigenvalue_condition(self) -> bool:
        """Condición elíptica no degenerada: ∃ λ con |λ|=1 y λ ≠ ±1 (no resonancia 1,2)."""
        eigs = self.eigenvalues()
        for lam in eigs:
            mod = abs(complex(lam))
            if abs(mod - 1.0) < 1e-8 and abs(lam - 1.0) > 1e-8 and abs(lam + 1.0) > 1e-8:
                return True
        return False


class KAMInvariantTorus:
    r"""
    Toro invariante 𝕋^n con vector de frecuencias ω ∈ ℝ^n.

    Condición diofantina (Kolmogorov):
        |⟨k, ω⟩| ≥ γ / |k|_1^τ ,   ∀ k ∈ ℤ^n \ {0},   τ > n − 1.

    En el régimen disipativo (mapa L3 contractivo) se aplica el criterio BHT
    (Broer–Huitema–Takens): el toro persiste como variedad invariante normalmente
    hiperbólica si γ > ||ε|| y la disipación no satura γ.
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies: Tuple[float, ...] = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)

    def _lattice_vectors(self, max_k: int, modes: int) -> Iterable[NDArray[np.float64]]:
        """
        Generadores de k ∈ ℤ^m \ {0} con m = min(n, modes):
          • ejes ±e_i
          • pares ±e_i ± e_j
          • cubo |k|_∞ ≤ 2 en los primeros min(m, 4) modos
        Evita explosión combinatoria en dim Hilbert 56.
        """
        m = max(1, min(self.n, modes))
        seen: Set[Tuple[int, ...]] = set()

        def emit(vec: NDArray[np.float64]) -> Iterable[NDArray[np.float64]]:
            key = tuple(int(x) for x in vec)
            if any(key) and key not in seen:
                seen.add(key)
                yield vec

        for i in range(m):
            for s in (-1, 1):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = s
                yield from emit(kvec)

        for i, j in itertools.combinations(range(m), 2):
            for si, sj in itertools.product((-1, 1), repeat=2):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = si
                kvec[j] = sj
                yield from emit(kvec)

        cube = min(m, 4)
        bound = min(2, max(1, max_k // 6))
        for coeffs in itertools.product(range(-bound, bound + 1), repeat=cube):
            if all(c == 0 for c in coeffs):
                continue
            kvec = np.zeros(m, dtype=np.float64)
            kvec[:cube] = np.array(coeffs, dtype=np.float64)
            yield from emit(kvec)

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: float = _DIOPHANTINE_TAU
    ) -> float:
        """γ_eff = min_k |⟨k, ω⟩| · |k|_1^τ  (constante diofantina efectiva)."""
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        gammas: List[float] = []
        for kvec in self._lattice_vectors(max_k=max_k, modes=m):
            inner = abs(float(np.dot(kvec, omega)))
            l1 = float(np.sum(np.abs(kvec)))
            if l1 <= 0.0:
                continue
            gammas.append(inner * (l1**tau))
        if not gammas:
            return 0.0
        return float(min(gammas))

    def small_divisors(self, max_k: int = _KAM_MAX_K) -> Tuple[float, ...]:
        """Los 8 divisores |⟨k, ω⟩| más pequeños (diagnóstico de resonancia)."""
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        divs = [abs(float(np.dot(kvec, omega))) for kvec in self._lattice_vectors(max_k, m)]
        divs = [d for d in divs if d > 0.0]
        divs.sort()
        return tuple(divs[:8])

    def is_kam_stable(self, perturbation_norm: float, gamma: Optional[float] = None) -> bool:
        """Estabilidad KAM hamiltoniana clásica: γ > ||perturbación|| ⇒ el toro sobrevive."""
        if gamma is None:
            gamma = self.diophantine_constant()
        return gamma > perturbation_norm

    def is_dissipative_kam_stable(
        self,
        perturbation_norm: float,
        dissipation: float,
        gamma: Optional[float] = None,
    ) -> bool:
        """
        Criterio BHT (KAM disipativo): el toro persiste como NHIM si
            γ > ||ε||  ∧  |disipación| < min(γ, 0.95).
        """
        if gamma is None:
            gamma = self.diophantine_constant()
        dissip = abs(float(dissipation))
        return (gamma > perturbation_norm) and (dissip < min(gamma, _BHT_DISSIPATION_CEILING))

    def winding_numbers(self, phases: Sequence[float]) -> Tuple[float, ...]:
        """
        Números de enrollamiento relativos μ_i = ω_i / Δφ_unwrapped.
        `np.unwrap` elimina discontinuidades de rama; si Δφ~0 se reporta el 0-vector.
        """
        if len(phases) < 2:
            return tuple(0.0 for _ in range(self.n))
        unwrapped = np.unwrap(np.array(phases, dtype=np.float64))
        delta_phi = float(unwrapped[-1] - unwrapped[0])
        if abs(delta_phi) < _WILKINSON_EPSILON:
            return tuple(0.0 for _ in range(self.n))
        return tuple(float(w / delta_phi) for w in self.frequencies)


class RecursiveImprovementOperator:
    r"""
    Operador de Automejora Recursiva I : ℳ → ℳ sobre el espacio de fase complejo ℳ ⊂ ℂ^N.

    Jerarquía metacognitiva:
      L0: I_0(s) = s                                 (identidad)
      L1: I_1(s) = normalize((1−θ)s + θ·s*)          (contracción al objetivo)
      L2: I_2 = I_1 con θ actualizado por gradiente espectral del error
      L3: I_3 = I_2 con meta-regla dη/dn adaptada por
            (i)  espectro local DI (Rayleigh de diferencias finitas)
            (ii) número de rotación de Poincaré del historial
            (iii) detección de resonancia p/q (divisores pequeños)
          = metacognición metacognitiva con realimentación celeste.

    La recursión de nivel 3 actúa sobre los *hiperparámetros* (θ, η) usando
    el mapa de retorno como observable de segundo orden: si ρ es resonante se
    incrementa la contracción (escapar del divisor pequeño); si ρ es diofantino
    se relaja (preservar el toro BHT).
    """

    def __init__(self, contraction: float = 0.5, meta_learning: float = 0.3) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self._last_rotation: float = 0.0
        self._resonance_hits: int = 0

    def apply_level_0(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.asarray(state, dtype=np.complex128).copy()

    def apply_level_1(
        self, state: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        theta = self.contraction
        new_state = (1.0 - theta) * state + theta * target
        return _safe_unit(new_state.astype(np.complex128))

    def apply_level_2(
        self,
        state: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """Adapta la contracción θ según el ratio de error residual reciente."""
        if len(history) >= 2:
            err = float(np.linalg.norm(state - target))
            prev_err = float(np.linalg.norm(history[-2] - target))
            if prev_err > _WILKINSON_EPSILON:
                ratio = err / prev_err
                self.contraction = float(
                    np.clip(self.contraction * (1.0 + 0.1 * (ratio - 1.0)), 0.05, 0.95)
                )
        return self.apply_level_1(state, target)

    def apply_level_3(
        self,
        state: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
        poincare_rotation: Optional[float] = None,
    ) -> NDArray[np.complex128]:
        """
        Meta-mejora L3:
          • espectro local (cociente de Rayleigh de Δz) → frena/acelera η
          • rotación de Poincaré → penaliza resonancias p/q (KAM breakdown)
          • dθ ← dθ + 0.05 (η − 1/2)  (meta-regla de segundo orden)
        """
        if len(history) >= 3:
            eigs = self._estimate_local_jacobian_spectrum(history[-3:])
            rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
            if rho > 1.0:
                self.meta_learning = float(np.clip(self.meta_learning * 0.90, 0.05, 0.90))
            elif 0.0 < rho < 0.5:
                self.meta_learning = float(np.clip(self.meta_learning * 1.05, 0.05, 0.90))

        if poincare_rotation is not None and math.isfinite(poincare_rotation):
            self._last_rotation = float(poincare_rotation)
            if _is_near_resonance(poincare_rotation):
                self._resonance_hits += 1
                self.contraction = float(np.clip(self.contraction * 1.08, 0.05, 0.95))
                self.meta_learning = float(np.clip(self.meta_learning * 0.92, 0.05, 0.90))
            else:
                self.contraction = float(np.clip(self.contraction * 0.98, 0.05, 0.95))

        self.contraction = float(
            np.clip(self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95)
        )
        return self.apply_level_2(state, target, history)

    @staticmethod
    def _estimate_local_jacobian_spectrum(
        recent: Sequence[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """
        Espectro local de DI vía cociente de Rayleigh:
            λ ≈ ⟨v₂, v₁⟩ / ⟨v₂, v₂⟩,   v₁ = z_{t}−z_{t−1}, v₂ = z_{t−1}−z_{t−2}.
        """
        if len(recent) < 3:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        v1 = np.asarray(recent[-1], dtype=np.complex128) - np.asarray(recent[-2], dtype=np.complex128)
        v2 = np.asarray(recent[-2], dtype=np.complex128) - np.asarray(recent[-3], dtype=np.complex128)
        denom = np.vdot(v2, v2)
        if abs(denom) < _WILKINSON_EPSILON:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        rayleigh = np.vdot(v2, v1) / denom
        return np.array([complex(rayleigh)], dtype=np.complex128)

    def _poincare_rotation_from_history(
        self, history: Sequence[NDArray[np.complex128]]
    ) -> Optional[float]:
        if len(history) < 4:
            return None
        dim = int(np.asarray(history[0]).reshape(-1).size)
        normal = tuple([1.0 / math.sqrt(dim)] * dim)
        section = PoincareSection(normal=normal, offset=0.0, label="Σ_L3")
        prm = PoincareReturnMap(section)
        crossings = prm.detect_crossings(list(history))
        if len(crossings) < 2:
            return None
        return PoincareReturnMap.birkhoff_rotation_number(crossings)

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (iterate_level_3).                                                ║
    # ║  ÚLTIMO método de FASE 2. Retorna ImprovementTrajectory → PRIMER objeto de FASE 3.          ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: ActionAngleSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> "ImprovementTrajectory":
        r"""
        Itera el operador L3 sobre el seed y devuelve la trayectoria completa con:
          • Estados complejos z_0 → z_T (T = max_iters)
          • Acciones J_t = Σ_k |z_{t,k}|²
          • Ángulos dominantes θ_t = arg(z_t)[dominante]  (unwrap-ready)
          • Exponente de Lyapunov λ (media de log-ratios de ||Δz||)
          • Constante diofantina γ y estabilidad KAM / BHT
          • Números de enrollamiento topológicos del toro
          • Número de rotación de Birkhoff del mapa de Poincaré
          • Cruces transversales y condición de twist Poincaré-Birkhoff
        """
        state = seed.to_improvement_vector().astype(np.complex128)
        dim = int(state.size)
        target = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        states_list: List[NDArray[np.complex128]] = [state.copy()]
        actions_list: List[float] = [float(np.sum(np.abs(state) ** 2))]
        angles_list: List[float] = [float(np.angle(state[int(np.argmax(np.abs(state)))]))]
        history: List[NDArray[np.complex128]] = []

        n_sec = np.ones(dim, dtype=np.float64) / math.sqrt(dim)
        live_section = PoincareSection(
            normal=tuple(float(x) for x in n_sec),
            offset=0.0,
            label="Σ_flow",
        )

        for _ in range(max_iters):
            history.append(state.copy())
            rho_p = self._poincare_rotation_from_history(history)
            state = self.apply_level_3(state, target, history, poincare_rotation=rho_p)
            states_list.append(state.copy())
            actions_list.append(float(np.sum(np.abs(state) ** 2)))
            dom = int(np.argmax(np.abs(state)))
            angles_list.append(float(np.angle(state[dom])))

        traj_arr = np.array(states_list, dtype=np.complex128)
        if len(traj_arr) >= 2:
            diffs = np.diff(traj_arr, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            valid = norms > _WILKINSON_EPSILON
            if int(np.sum(valid)) >= 2 and np.all(norms > _WILKINSON_EPSILON):
                log_ratios = np.log(norms[1:] / np.maximum(norms[:-1], _WILKINSON_EPSILON))
                finite = np.isfinite(log_ratios)
                lyapunov = float(np.mean(log_ratios[finite])) if np.any(finite) else 0.0
            elif int(np.sum(valid)) == 0:
                lyapunov = -20.0
            else:
                lyapunov = float(np.log(max(float(norms[-1]), _WILKINSON_EPSILON)))
            convergence = float(norms[-1]) if norms.size else 0.0
        else:
            lyapunov, convergence = 0.0, 0.0

        if not math.isfinite(lyapunov):
            lyapunov = -20.0 if lyapunov < 0 else 0.0

        torus = KAMInvariantTorus(seed.omega_vector)
        gamma = torus.diophantine_constant()
        dissipation = abs(self.contraction)
        kam_stable = torus.is_kam_stable(convergence, gamma)
        bht_stable = torus.is_dissipative_kam_stable(convergence, dissipation, gamma)
        winding = torus.winding_numbers(angles_list)
        spectral_radius = float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0

        prm = PoincareReturnMap(live_section)
        crossings = prm.apply(states_list)
        rotation = PoincareReturnMap.birkhoff_rotation_number(crossings) if len(crossings) >= 2 else 0.0
        twist = PoincareBirkhoffTwist(seed)
        twist_ok = twist.satisfies_twist()

        return ImprovementTrajectory(
            source_seed_hash=seed.source_hash,
            states=tuple(tuple(complex(x) for x in s) for s in states_list),
            actions=tuple(actions_list),
            angles=tuple(angles_list),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            spectral_radius=spectral_radius,
            is_kam_stable=kam_stable,
            is_bht_stable=bht_stable,
            kam_winding_numbers=winding,
            diophantine_gamma=gamma,
            poincare_rotation_number=rotation,
            poincare_crossing_count=len(crossings),
            twist_condition=twist_ok,
            hamiltonian_residue=seed.hamiltonian_identity_residue(),
            resonance_hits=self._resonance_hits,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: INTEGRACIÓN, AUDITORÍA KAM Y FACTORÍA SOBERANA                                          ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La PRIMERA definición (ImprovementTrajectory) consume iterate_level_3 de FASE 2.              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class ImprovementTrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRY: Continúa de iterate_level_3 (FASE 2).                                      │
    │  Trayectoria completa de la automejora L3 con análisis de Poincaré, twist y KAM/BHT.        │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_seed_hash: str
    states: Tuple[Tuple[complex, ...], ...]
    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    lyapunov_exponent: float
    convergence_residual: float
    spectral_radius: float
    is_kam_stable: bool
    is_bht_stable: bool
    kam_winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    poincare_rotation_number: float
    poincare_crossing_count: int
    twist_condition: bool
    hamiltonian_residue: float
    resonance_hits: int

    def to_poincare_section(
        self, normal: Optional[Tuple[float, ...]] = None
    ) -> PoincareSection:
        """
        Proyecta la trayectoria a una sección de Poincaré:
        el offset es la media de g a lo largo de la órbita (producto hermítico).
        """
        dim = len(self.states[0]) if self.states else 1
        if normal is None:
            normal = tuple([1.0 / math.sqrt(dim)] * dim)
        z_arr = np.array([np.array(s, dtype=np.complex128) for s in self.states])
        n_arr = np.array(normal, dtype=np.complex128)
        g_vals = np.real(z_arr @ np.conjugate(n_arr))
        offset = float(np.mean(g_vals)) if g_vals.size else 0.0
        return PoincareSection(
            normal=normal, offset=offset, label=f"Σ[{self.source_seed_hash[:8]}]"
        )

    def is_stable_fixed_point(self) -> bool:
        """Punto fijo estable ⇔ Lyapunov < 0 ∧ (toro KAM o NHIM BHT) sobrevive."""
        return self.lyapunov_exponent < 0.0 and (self.is_kam_stable or self.is_bht_stable)

    def topological_index(self) -> int:
        """
        Índice topológico grosero: round(ρ) del número de rotación de Poincaré.
        Distingue clases de homotopía del retorno sobre Σ.
        """
        if not math.isfinite(self.poincare_rotation_number):
            return 0
        return int(round(self.poincare_rotation_number))

    def kam_audit_vector(self) -> Tuple[float, float, float, int, int]:
        """Firma de auditoría (γ, λ, ρ, #cruces, resonancias) para exclusión topológica."""
        return (
            float(self.diophantine_gamma),
            float(self.lyapunov_exponent),
            float(self.poincare_rotation_number),
            int(self.poincare_crossing_count),
            int(self.resonance_hits),
        )


class PauliExclusionAuditorCAR:
    r"""
    Auditor de exclusión de Pauli {a_i, a_j†} = δ_ij I, enriquecido con:
      • Registro de tori KAM (exclusión topológica: dos fermiones no pueden
        compartir números de enrollamiento ni índice de Poincaré).
      • Registro de exponentes de Lyapunov (estabilidad dinámica).
      • Registro de números de rotación de Birkhoff (clase homotópica del retorno).
    """

    def __init__(self) -> None:
        self._occupied_fermion_hashes: Set[str] = set()
        self._kam_winding_registry: List[Tuple[float, ...]] = []
        self._lyapunov_registry: List[float] = []
        self._poincare_index_registry: List[int] = []

    def audit_and_register_fermion(self, cartridge: ToonCartridge) -> Tuple[bool, str]:
        """
        Auditoría O(1) (hash) + verificación topológica KAM/Poincaré (O(n_fermiones)).
        Veto si hay duplicación de hash, de firma de enrollamiento o de índice de retorno.
        """
        if cartridge.category != ParticleCategory.FERMION_CAR:
            return True, "NON_FERMION_PASSTHROUGH"

        state_hash = cartridge.quantum_state_hash
        if state_hash in self._occupied_fermion_hashes:
            logger.error("[PAULI VETO] Duplicación hash: %s", cartridge.particle_name)
            return False, "PAULI_EXCLUSION_VIOLATION_DUPLICATE_HASH"

        seed = cartridge.to_action_angle_seed()
        improver = RecursiveImprovementOperator()
        traj = improver.iterate_level_3(seed, max_iters=6)

        for existing in self._kam_winding_registry:
            if self._windings_too_close(traj.kam_winding_numbers, existing):
                logger.error(
                    "[PAULI VETO KAM] Duplicación topológica: %s",
                    cartridge.particle_name,
                )
                return False, "PAULI_EXCLUSION_VIOLATION_KAM_DUPLICATE"

        topo = traj.topological_index()
        if topo != 0 and topo in self._poincare_index_registry:
            if any(
                abs(a - b) < 1e-9
                for a, b in zip(
                    traj.kam_winding_numbers[:2],
                    self._kam_winding_registry[-1][:2] if self._kam_winding_registry else (0.0, 0.0),
                )
            ):
                logger.error(
                    "[PAULI VETO POINCARE] Índice de retorno duplicado: %s",
                    cartridge.particle_name,
                )
                return False, "PAULI_EXCLUSION_VIOLATION_POINCARE_INDEX"

        self._occupied_fermion_hashes.add(state_hash)
        self._kam_winding_registry.append(traj.kam_winding_numbers)
        self._lyapunov_registry.append(traj.lyapunov_exponent)
        self._poincare_index_registry.append(topo)
        return True, "FERMION_OCCUPIED_KAM_OK"

    @staticmethod
    def _windings_too_close(
        w1: Tuple[float, ...], w2: Tuple[float, ...], tol: float = 1e-12
    ) -> bool:
        n = min(len(w1), len(w2), 3)
        if n == 0:
            return False
        return all(abs(a - b) < tol for a, b in zip(w1[:n], w2[:n]))

    def clear(self) -> None:
        self._occupied_fermion_hashes.clear()
        self._kam_winding_registry.clear()
        self._lyapunov_registry.clear()
        self._poincare_index_registry.clear()


class ToonParticleFactory:
    """
    Factoría soberana: las energías base se modulan por la firma espectral
    del operador de automejora L3 y por la constante diofantina del seed
    (convergencia propia + persistencia KAM de cada trayectoria).
    """

    @staticmethod
    def _create_hmac(
        cartridge_id: str, particle_name: str, energy: float, timestamp: float, key: bytes
    ) -> str:
        seed = f"{cartridge_id}:{particle_name}:{energy:.6f}:{timestamp:.6f}"
        return hmac.new(key, seed.encode("utf-8"), hashlib.sha256).hexdigest()

    @classmethod
    def _sovereign_energy(cls, base_energy: float, seed_id: str, level: int = 3) -> float:
        """
        Modulación de la energía base por:
          1. tasa de contracción L3 (residuo espectral),
          2. constante diofantina γ de un mini-toro áureo derivado del seed.
        Cada seed_id produce una trayectoria distinta cuya tasa de convergencia
        y γ firman la energía resultante (automejora recursiva determinista).
        """
        h = hashlib.sha256(f"{seed_id}:{base_energy:.6f}:L{level}".encode()).digest()
        seed_val = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed_val)
        dim = 8
        state = rng.normal(0.0, 1.0, dim) + 1j * rng.normal(0.0, 1.0, dim)
        state = _safe_unit(state.astype(np.complex128))
        target = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        op = RecursiveImprovementOperator(contraction=0.5, meta_learning=0.3)
        history: List[NDArray[np.complex128]] = []
        for _ in range(max(1, int(level))):
            history.append(state.copy())
            rho_p = op._poincare_rotation_from_history(history)
            state = op.apply_level_3(state, target, history, poincare_rotation=rho_p)

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)
        gamma = KAMInvariantTorus(tuple(float(x) for x in omega)).diophantine_constant()
        gamma_mod = float(np.clip(math.log10(gamma + 1.0e-30) / 10.0, -0.3, 0.3))

        rate_mod = 0.0
        if len(history) >= 2:
            d1 = float(np.linalg.norm(history[-1] - history[-2]))
            d0 = (
                float(np.linalg.norm(history[-2] - history[-3]))
                if len(history) >= 3
                else d1
            )
            if d0 > _WILKINSON_EPSILON:
                rate = d1 / d0
                rate_mod = float(np.clip(rate - 0.5, -0.5, 0.5))

        return base_energy * (1.0 + 0.02 * rate_mod + 0.01 * gamma_mod)

    @classmethod
    def create_polaron(
        cls, cartridge_id: str, ballast_mass_tn: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "PolaronCartridge"
        energy = cls._sovereign_energy(150.0 + ballast_mass_tn, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.FERMION_CAR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("material", "concreto_5000psi_acero"),
                ("ballast_tn", str(ballast_mass_tn)),
            ),
        )

    @classmethod
    def create_torsion(
        cls, cartridge_id: str, torsion_angle_rad: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "TorsionCartridge"
        energy = cls._sovereign_energy(75.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.FERMION_CAR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("homological_torsion", "True"),
                ("angle_rad", f"{torsion_angle_rad:.6f}"),
            ),
        )

    @classmethod
    def create_electron(
        cls, cartridge_id: str, anomaly_cost_cop: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "ElectronCartridge"
        energy = _ELECTRON_MASS_KEV
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.FERMION_CAR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("anomaly_type", "sobrecosto_apu"),
                ("cost_cop", f"{anomaly_cost_cop:.2f}"),
            ),
        )

    @classmethod
    def create_proton(
        cls, cartridge_id: str, cement_volume_m3: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "ProtonCartridge"
        energy = 938272.0
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.FERMION_CAR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("stability", "LTI_BIBO_STABLE"),
                ("volume_m3", f"{cement_volume_m3:.2f}"),
            ),
        )

    @classmethod
    def create_householder(
        cls, cartridge_id: str, bim_revision: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "HouseholderReflectionFermion"
        energy = cls._sovereign_energy(500.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.FERMION_CAR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("reflection_type", "orthogonal_BIM_2026"),
                ("revision", bim_revision),
            ),
        )

    @classmethod
    def create_photon(
        cls, cartridge_id: str, secop_id: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "PhotonCartridge"
        energy = 0.0
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(("regulation", "SECOP_II_OPA"), ("secop_id", secop_id)),
        )

    @classmethod
    def create_riemannian_focal(
        cls, cartridge_id: str, eikonal_action: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "RiemannianFocalBoson"
        energy = cls._sovereign_energy(10.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("geodesic_path", "critical_route"),
                ("eikonal_action", f"{eikonal_action:.6f}"),
            ),
        )

    @classmethod
    def create_magnon(
        cls, cartridge_id: str, triangulation_veto: bool, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "MagnonCartridge"
        energy = cls._sovereign_energy(25.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(("anti_triangulation", str(triangulation_veto)),),
        )

    @classmethod
    def create_swelling_plasmon(
        cls, cartridge_id: str, wacc_spread_pct: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "SwellingPlasmonCartridge"
        energy = cls._sovereign_energy(120.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("wacc_volatility", "15.0_pct_protection"),
                ("spread", f"{wacc_spread_pct:.4f}"),
            ),
        )

    @classmethod
    def create_yielding_phonon(
        cls, cartridge_id: str, union_agreement: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "YieldingPhononCartridge"
        energy = cls._sovereign_energy(40.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("labor_agreement", "CUT_CTC_CGT"),
                ("union_code", union_agreement),
            ),
        )

    @classmethod
    def create_liquefaction_soliton(
        cls, cartridge_id: str, ethnic_consultation: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "LiquefactionSolitonCartridge"
        energy = cls._sovereign_energy(300.0, cartridge_id)
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.BOSON_CCR,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("geotechnics_license", "consult_ethnic_ok"),
                ("license_id", ethnic_consultation),
            ),
        )

    @classmethod
    def create_positron(
        cls, cartridge_id: str, auth_signature: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        now = time.time()
        pname = "PositronCartridge"
        energy = _POSITRON_REST_ENERGY_KEV
        sig = cls._create_hmac(cartridge_id, pname, energy, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=ParticleCategory.ANTIMATTER,
            particle_name=pname,
            mass_energy_kev=energy,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=(
                ("antimatter_type", "hmac_authorization"),
                ("auth_sig", auth_signature),
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — v11.0.0-SOVEREIGN-TOON-POINCARE-KAM-L3-PhD")
    print("  12 Partículas | Poincaré | Birkhoff-Twist | KAM/BHT | Automejora L3")
    print(banner)

    auditor = PauliExclusionAuditorCAR()

    p1 = ToonParticleFactory.create_polaron("POL-001", ballast_mass_tn=12.5)
    p2 = ToonParticleFactory.create_torsion("TOR-002", torsion_angle_rad=0.045)
    p3 = ToonParticleFactory.create_electron("ELE-003", anomaly_cost_cop=15000000.0)
    p4 = ToonParticleFactory.create_proton("PRO-004", cement_volume_m3=45.0)
    p5 = ToonParticleFactory.create_householder("HOU-005", bim_revision="REV-2026-v9")
    p6 = ToonParticleFactory.create_photon("PHO-006", secop_id="SECOP-2026-8891")
    p7 = ToonParticleFactory.create_riemannian_focal("RFB-007", eikonal_action=0.0012)
    p8 = ToonParticleFactory.create_magnon("MAG-008", triangulation_veto=True)
    p9 = ToonParticleFactory.create_swelling_plasmon("SWP-009", wacc_spread_pct=0.015)
    p10 = ToonParticleFactory.create_yielding_phonon("YPH-010", union_agreement="CCT-2026-CIVIL")
    p11 = ToonParticleFactory.create_liquefaction_soliton("LQS-011", ethnic_consultation="ACTA-ETHNIC-001")
    p12 = ToonParticleFactory.create_positron("POS-012", auth_signature="HMAC-AUTH-OK-2026")

    all_particles = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]

    print(f"\n  • Partículas instanciadas: {len(all_particles)}")
    print(
        f"  {'#':<3} {'Nombre':<32} {'Cat':<13} {'SHA':<14} {'Pauli':<28} "
        f"{'HMAC':<6} {'J':<7} {'λ':<9} {'KAM':<4} {'BHT':<4} {'ρ'}"
    )
    print("  " + "─" * 140)

    for i, p in enumerate(all_particles, 1):
        ok, msg = auditor.audit_and_register_fermion(p)
        hmac_ok = p.verify_hmac_authenticity()
        v_vec = p.to_hilbert_state_vector(dimension=56)
        _ = float(np.linalg.norm(v_vec))

        seed = p.to_action_angle_seed()
        improver = RecursiveImprovementOperator()
        traj = improver.iterate_level_3(seed, max_iters=6)

        section = traj.to_poincare_section()
        states_np = [np.array(s, dtype=np.complex128) for s in traj.states]
        prm = PoincareReturnMap(section)
        crossings = prm.apply(states_np)

        if len(states_np) >= 2:
            d_state = states_np[-1] - states_np[-2]
            delta = float(np.linalg.norm(d_state))
            scale = (traj.spectral_radius + 0.0j) if delta > _WILKINSON_EPSILON else (0.5 + 0.0j)
            jac = np.eye(len(states_np[-1]), dtype=np.complex128) * scale
            bnf = BirkhoffNormalForm(states_np[-1], jac)
            _dyn_class = bnf.classify()
        else:
            _dyn_class = "N/A"

        print(
            f"  [{i:02d}] {p.particle_name:<32} {p.category.name:<13} "
            f"{p.quantum_state_hash[:12]}... {msg:<28} "
            f"{str(hmac_ok):<6} {seed.J_total:<7.4f} {traj.lyapunov_exponent:<9.4f} "
            f"{'✓' if traj.is_kam_stable else '✗':<4} "
            f"{'✓' if traj.is_bht_stable else '✗':<4} "
            f"{traj.poincare_rotation_number:+.4f}"
        )

    print("\n  • Prueba de duplicación Pauli (p1 re-registered):")
    dup_ok, dup_msg = auditor.audit_and_register_fermion(p1)
    print(f"      Veto exitoso = {not dup_ok}  →  {dup_msg}")

    print("\n  • Análisis Poincaré/KAM/BHT detallado (fermión p1 = Polaron):")
    seed1 = p1.to_action_angle_seed()
    improver = RecursiveImprovementOperator()
    traj1 = improver.iterate_level_3(seed1, max_iters=8)
    section1 = traj1.to_poincare_section()
    states_np1 = [np.array(s, dtype=np.complex128) for s in traj1.states]
    prm1 = PoincareReturnMap(section1)
    crossings1 = prm1.apply(states_np1)
    torus1 = KAMInvariantTorus(seed1.omega_vector)
    gamma1 = torus1.diophantine_constant()
    twist1 = PoincareBirkhoffTwist(seed1)

    print(f"      Invariante Poincaré-Cartan I₁ = {seed1.poincare_cartan_invariant():.6f}")
    print(f"      Residuo identidad hamiltoniana = {seed1.hamiltonian_identity_residue():.3e}")
    print(f"      Norma simpléctica ||v||_ω² = {p1.symplectic_norm():.6f}")
    print(f"      Sección Σ: g(z)=Re⟨n,z⟩−c, offset={section1.offset:+.6e}")
    print(f"      Cruces transversales detectados: {len(crossings1)}")
    print(f"      Número de rotación de Birkhoff ρ = {traj1.poincare_rotation_number:+.6f}")
    print(f"      Twist Poincaré-Birkhoff ∂(Δθ)/∂J = {twist1.twist_derivative():+.6e}  (ok={twist1.satisfies_twist()})")
    print(f"      Constante diofantina γ = {gamma1:.6e}")
    print(f"      Divisores pequeños (4): {torus1.small_divisors()[:4]}")
    print(f"      Perturbación residual = {traj1.convergence_residual:.6e}")
    print(f"      Estabilidad KAM: γ > residual → {traj1.is_kam_stable}")
    print(f"      Estabilidad BHT (KAM disipativo) → {traj1.is_bht_stable}")
    print(f"      Exponente de Lyapunov λ = {traj1.lyapunov_exponent:+.6f}")
    print(f"      Hits de resonancia p/q (L3) = {traj1.resonance_hits}")
    print(
        "      Números de enrollamiento (primeros 3): "
        f"({traj1.kam_winding_numbers[0]:.6f}, "
        f"{traj1.kam_winding_numbers[1]:.6f}, "
        f"{traj1.kam_winding_numbers[2]:.6f})"
    )
    print(f"      Índice topológico de retorno = {traj1.topological_index()}")
    print(f"      Punto fijo estable: {traj1.is_stable_fixed_point()}")

    print("\n" + banner)
    print("  ✓ Todas las 3 fases anidadas se ejecutaron correctamente.")
    print("  ✓ FASE 1 → FASE 2: ToonCartridge.to_action_angle_seed → ActionAngleSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveImprovementOperator.iterate_level_3 → ImprovementTrajectory")
    print(banner)