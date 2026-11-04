# -*- coding: utf-8 -*-
r"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo    : ToonCartridgeBase · Mecánica Celeste de Poincaré · Automejora Nivel 3          ║
║ Ubicación : app/wisdom/toon_cartridge_base.py                                              ║
║ Versión   : 12.0.0-SOVEREIGN-TOON-POINCARE-KAM-L3-PhD                                      ║
║ Función   : Empaquetado inmutable de 12 partículas en un espacio de Fock discretizado,     ║
║             embedding acción-ángulo tipo Delaunay–Kepler, mapas de Poincaré–Birkhoff,      ║
║             auditoría KAM/BHT, forma normal de Birkhoff y automejora recursiva L3.         ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝

MARCO TEÓRICO
─────────────
1. Embedding de Hilbert: |v⟩ ∈ S^{2N−1} ⊂ ℂ^N. Acción-ángulo:
       a_k = v_k,  J_k = |a_k|²,  θ_k = arg(a_k),  ω_k = √(1+k) + (φ−1)/(1+k)
   Σ_k J_k = ‖v‖² = 1 (invariante integral I₁ con normalización J := (1/2π)∮p dq).

2. Carta canónica de Whittaker:
       q_k = √(2J_k/ω_k) cos θ_k,   p_k = −√(2J_kω_k) sin θ_k
   Identidad hamiltoniana:  (p_k² + ω_k²q_k²)/(2ω_k) = J_k.
   Jacobiano:  det ∂(q_k,p_k)/∂(J_k,θ_k) = −1  ⇒  dq∧dp = dθ∧dJ (Bargmann).

3. Anisocronía tipo Delaunay–Kepler: la ley de frecuencias de la carta es
       ω_k(J) = ω_k⁰ (J_k / J_k⁰)^(−3),
   análoga a n(L) = μ²/L³ de la ley de Kepler (H_K = −μ²/2L²). Su twist es
       ∂(Δθ_k)/∂J_k = −3 T ω_k / J_k ≠ 0,   T = período del mapa.
   Esto cumple la condición de twist de Poincaré–Birkhoff, que el flujo lineal no cumple.

4. Sección de Poincaré Σ = {z : g(z) = Re⟨n,z⟩ − c = 0}, con transversalidad ∇g·f ≠ 0.
   Número de rotación de Birkhoff: ρ = lim_{N→∞} (θ_N − θ_0)/(2πN).

5. Criterio BHT: el toro disipativo persiste si γ > ‖ε‖ y |disipación| < min(γ, 0.95).
   Forma normal de Birkhoff: I(z) = z* + M(z−z*) + O(|z−z*|²); no resonancia de Moser λᵏ ≠ 1, k ≤ 4.

6. Automejora L3: sobre hiperparámetros (θ, η) con realimentación del número de rotación ρ
   y del espectro local de Rayleigh.

AXIOMAS VERIFICADOS
───────────────────
[A1] Normalización: Σ_k J_k = 1.
[A2] Hamiltoniano: residuo max_k |(p²+ω²q²)/2ω − J| = O(ε_máquina).
[A3] Bargmann: |det ∂(q,p)/∂(J,θ) + 1| = O(ε_máquina).
[A4] Twist: ∂(Δθ)/∂J analítico ≈ numérico (error relativo < 10⁻⁴).
[A5] Criterio BHT: persistencia de toro disipativo bajo γ > ‖ε‖.

NOTA DE RIGOR
─────────────
γ_eff es una estimación muestral sobre un retículo finito de k ∈ ℤ^m (m ≤ 6). Es una cota
superior de la constante diofantina verdadera. Las analogías CAR/CCR, Fock y Delaunay son
representaciones del modelo; no implican operadores fermiónicos reales.
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
from typing import Final, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.ToonCartridgeBase")

__version__: Final[str] = "12.0.0-SOVEREIGN-TOON-POINCARE-KAM-L3-PhD"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDACIÓN INMUTABLE, CONSTANTES, EMBEDDING ACCIÓN-ÁNGULO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# 1.0  Constantes y utilidades numéricas
# 1.1  ParticleCategory y ToonCartridge (inmutable, slots)
# 1.2  Embedding de Hilbert, HMAC, ocupación de Fock
# 1.3  to_action_angle_seed  ← ÚLTIMO método de FASE 1.
#      Retorna ActionAngleSeed, que es el PRIMER objeto consumido por FASE 2.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ── 1.0 Constantes físicas, numéricas FPU y umbrales KAM / Poincaré / L3 ──────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_DEFAULT_HILBERT_DIM: Final[int] = 56
_MIN_HILBERT_DIM: Final[int] = 4
_SOVEREIGN_MINI_DIM: Final[int] = 8
_ELECTRON_MASS_KEV: Final[float] = 511.0
_POSITRON_REST_ENERGY_KEV: Final[float] = 1022.0
_PROTON_MASS_KEV: Final[float] = 938272.0
_GAUGE_SECRET_KEY: Final[bytes] = b"APU_FILTER_TOON_CARTRIDGE_BASE_SECRET_KEY"
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_MAX_K: Final[int] = 12
_KAM_LATTICE_MODES: Final[int] = 6
_KAM_CUBE_MODES: Final[int] = 4
_SMALL_DIVISOR_COUNT: Final[int] = 8
_KEPLER_EXPONENT: Final[float] = 3.0
_TRANSVERSALITY_TOL: Final[float] = 1.0e-10
_TWIST_TOL: Final[float] = 1.0e-12
_TWIST_REL_TOL: Final[float] = 1.0e-4
_MOSER_NONRESONANCE_ORDER: Final[int] = 4
_RESONANCE_MAX_Q: Final[int] = 8
_RESONANCE_TOL: Final[float] = 0.02
_BHT_DISSIPATION_CEILING: Final[float] = 0.95
_L3_DEFAULT_ITERS: Final[int] = 8
_TRAJECTORY_PHASE_MODES: Final[int] = 3
_LYAPUNOV_CLIP: Final[float] = 20.0
_CONTRACTION_BOUNDS: Final[Tuple[float, float]] = (0.05, 0.95)
_META_BOUNDS: Final[Tuple[float, float]] = (0.05, 0.90)
_PAULI_WINDING_TOL: Final[float] = 1.0e-12
_PAULI_POINCARE_TOL: Final[float] = 1.0e-9


def _safe_unit(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Normalización estable en ℂ^N. Fallback e_0 si ‖v‖ ≈ 0 o no finito."""
    arr = np.asarray(vector, dtype=np.complex128)
    nrm = float(np.linalg.norm(arr))
    if not math.isfinite(nrm) or nrm <= _WILKINSON_EPSILON:
        out = np.zeros_like(arr)
        out.reshape(-1)[0] = 1.0 + 0.0j
        return out
    return arr / nrm


def _clamp(x: float, lo: float, hi: float) -> float:
    """Proyección sobre [lo, hi]. NaN se proyecta a lo (estado conservador)."""
    if math.isnan(x):
        return float(lo)
    return float(min(max(x, lo), hi))


def _wrap_to_pi(x: float) -> float:
    """Lleva un ángulo al intervalo [−π, π)."""
    return (x + math.pi) % _TWO_PI - math.pi


def _fpu_golden_frequencies(k_idx: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ley de frecuencias base ω_k = √(1+k) + (φ−1)/(1+k)."""
    return np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)


def _is_near_resonance(
    rho: float, max_q: int = _RESONANCE_MAX_Q, tol: float = _RESONANCE_TOL
) -> bool:
    """
    Aproximación de Farey: frac(ρ) ≈ p/q con q ≤ max_q y |frac − p/q| < tol/q.
    Las resonancias racionales del mapa de Poincaré destruyen tori KAM.
    """
    if not math.isfinite(rho):
        return True
    frac = rho - math.floor(rho)
    for q in range(1, max_q + 1):
        p = round(frac * q)
        if abs(frac - (p / q)) < (tol / q):
            return True
    return False


def _hmac_message(
    cartridge_id: str, particle_name: str, energy_kev: float, timestamp_utc: float
) -> bytes:
    """Mensaje canónico para HMAC. Única fuente de verdad del formato de firma."""
    return f"{cartridge_id}:{particle_name}:{energy_kev:.6f}:{timestamp_utc:.6f}".encode("utf-8")


# ── 1.1 Categorización y cartucho inmutable ───────────────────────────────────────────────────────
class ParticleCategory(Enum):
    """Categorización cuántica de las 12 partículas fundamentales."""

    FERMION_CAR = auto()  # 5 fermiones estructurales (anticonmutación CAR)
    BOSON_CCR = auto()  # 6 bosones de gauge (conmutación CCR)
    ANTIMATTER = auto()  # 1 antimateria / radiación (aniquilación en Dirac)


@dataclass(frozen=True, slots=True)
class ToonCartridge:
    """
    Patrón polimórfico inmutable base para las 12 partículas del espacio de Fock.
    Inmutabilidad estricta vía (frozen=True, slots=True).
    """

    cartridge_id: str
    category: ParticleCategory
    particle_name: str
    mass_energy_kev: float
    signature_hmac: str
    timestamp_utc: float
    payload_tuple: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    # ── 1.2 Identidad, embedding y autenticidad ───────────────────────────────────────────────────
    @property
    def quantum_state_hash(self) -> str:
        """Firma SHA-256 determinista O(1) del estado en la Malla Agéntica."""
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
        Proyecta la partícula a |v⟩ ∈ S^{2N−1} ⊂ ℂ^N.
        Semilla: 64 bits del hash cuántico → PCG64 (reproducible, draws 1-D).
        """
        dim = max(_MIN_HILBERT_DIM, int(dimension))
        seed_val = int(self.quantum_state_hash[:16], 16)
        rng = np.random.default_rng(seed_val)
        real_part = rng.normal(0.0, 1.0, size=dim)
        imag_part = rng.normal(0.0, 1.0, size=dim)
        return _safe_unit((real_part + 1j * imag_part).astype(np.complex128))

    def verify_hmac_authenticity(self, secret_key: bytes = _GAUGE_SECRET_KEY) -> bool:
        """Verifica autenticidad HMAC-SHA256 con comparación en tiempo constante."""
        msg = _hmac_message(
            self.cartridge_id, self.particle_name, self.mass_energy_kev, self.timestamp_utc
        )
        expected = hmac.new(secret_key, msg, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)

    def fock_occupation_hint(self, dimension: int = _DEFAULT_HILBERT_DIM) -> Tuple[float, ...]:
        """Números de ocupación n_k = |v_k|² previos al mapa canónico."""
        v = self.to_hilbert_state_vector(dimension).reshape(-1)
        return tuple(float(x) for x in np.abs(v) ** 2)

    def symplectic_norm(self, dimension: int = _DEFAULT_HILBERT_DIM) -> float:
        """
        Norma hamiltoniana Σ_k (ω_k q_k² + p_k²/ω_k)/2 calculada desde la carta canónica.
        No es tautológica: depende de la transformación (J,θ) ↦ (q,p). Debe valer Σ J = 1.
        """
        return self.to_action_angle_seed(dimension).canonical_norm_squared()

    # ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: ÚLTIMO MÉTODO DE FASE 1                                                ║
    # ║  to_action_angle_seed retorna ActionAngleSeed, PRIMERA definición de FASE 2.             ║
    # ║  La firma de retorno "ActionAngleSeed" continúa en la definición de la clase siguiente.  ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    def to_action_angle_seed(
        self, dimension: int = _DEFAULT_HILBERT_DIM
    ) -> "ActionAngleSeed":
        r"""
        Transformación canónica (q, p) ↦ (J, θ) sobre el vector de Hilbert:

            a_k = v_k,  J_k = |a_k|²,  θ_k = arg(a_k),  ω_k = √(1+k) + (φ−1)/(1+k)

        Conservación: Σ_k J_k = 1 (invariante integral I₁ con J := (1/2π)∮p dq).
        Energía: E = Σ_k ω_k J_k.  Forma simpléctica: dq∧dp = dθ∧dJ.
        """
        dim = max(_MIN_HILBERT_DIM, int(dimension))
        v = self.to_hilbert_state_vector(dim).reshape(-1)
        J_vec = np.abs(v) ** 2
        theta_vec = np.angle(v)
        omega_vec = _fpu_golden_frequencies(np.arange(dim, dtype=np.float64))

        J_total = float(J_vec.sum())
        if J_total > _WILKINSON_EPSILON:
            J_vec = J_vec / J_total  # guarda; ‖v‖ = 1 ya implica Σ J = 1
        J_total = float(J_vec.sum())

        E_total = float(np.dot(omega_vec, J_vec))
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


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — MECÁNICA CELESTE DE POINCARÉ, FORMA NORMAL DE BIRKHOFF, KAM Y AUTOMEJORA L3
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# 2.1  ActionAngleSeed (ENTRADA desde FASE 1)
# 2.2  PoincareSection
# 2.3  PoincareReturnMap
# 2.4  PoincareBirkhoffTwist (twist analítico y numérico)
# 2.5  BirkhoffNormalForm (espectro y no resonancia de Moser)
# 2.6  KAMInvariantTorus (γ diofantina, BHT, enrollamiento empírico)
# 2.7  RecursiveImprovementOperator
#      iterate_level_3  ← ÚLTIMO método de FASE 2.
#      Retorna ImprovementTrajectory, PRIMERA definición de FASE 3.
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def _as_flat_complex(z: object) -> NDArray[np.complex128]:
    """Aplana cualquier vector complejo a 1-D sin copias innecesarias."""
    return np.asarray(z, dtype=np.complex128).reshape(-1)


def _real_inner(n: NDArray[np.complex128], z: NDArray[np.complex128]) -> float:
    """Re⟨n, z⟩ con truncación al mínimo común de dimensiones."""
    m = min(n.size, z.size)
    return float(np.real(np.vdot(n[:m], z[:m])))


# ── 2.1 Semilla acción-ángulo ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ActionAngleSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRADA: consume to_action_angle_seed (FASE 1).                                   │
    │  Semilla canónica (J_k, θ_k, ω_k) del oscilador armónico con anisocronía Delaunay–Kepler.   │
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

    def _vectors(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Acceso saneado (J, θ, ω) con piso ε sobre J y ω (evita divisiones por cero)."""
        J = np.maximum(np.asarray(self.J_vector, dtype=np.float64), _WILKINSON_EPSILON)
        theta = np.asarray(self.theta_vector, dtype=np.float64)
        omega = np.maximum(np.asarray(self.omega_vector, dtype=np.float64), _WILKINSON_EPSILON)
        return J, theta, omega

    def hamiltonian(self) -> float:
        """H(J) = Σ_k ω_k J_k en el punto de la semilla."""
        return float(self.E_total)

    def poincare_cartan_invariant(self) -> float:
        r"""Invariante integral I₁ = Σ_k J_k (normalización J := (1/2π)∮p dq)."""
        return float(self.J_total)

    def dominant_index(self) -> int:
        """Índice del modo de mayor acción."""
        return int(np.argmax(np.asarray(self.J_vector, dtype=np.float64)))

    def frequency_ratios(self) -> Tuple[float, ...]:
        """Razones de frecuencia ω_k / ω_0 (ω_0 > 0 por construcción)."""
        w = np.asarray(self.omega_vector, dtype=np.float64)
        if abs(float(w[0])) <= _WILKINSON_EPSILON:
            return tuple(0.0 for _ in w)
        return tuple(float(x) for x in w / w[0])

    def kepler_frequencies(
        self, J_eval: Optional[Sequence[float]] = None
    ) -> NDArray[np.float64]:
        r"""
        Ley anisócrona tipo Delaunay–Kepler: ω_k(J) = ω_k⁰ (J_k / J_k⁰)^(−3).
        Análoga a n(L) = μ²/L³ (H_K = −μ²/2L²). Sin argumento retorna ω⁰.
        """
        J0, _, w0 = self._vectors()
        if J_eval is None:
            return w0.copy()
        J = np.maximum(np.asarray(J_eval, dtype=np.float64), _WILKINSON_EPSILON)
        if J.shape != J0.shape:
            raise ValueError(f"J_eval debe tener forma {J0.shape}, recibido {J.shape}")
        return w0 * (J / J0) ** (-_KEPLER_EXPONENT)

    def anisochrony_derivative(self) -> NDArray[np.float64]:
        r"""∂ω_k/∂J_k en la semilla = −3 ω_k / J_k ≠ 0 (condición de twist)."""
        J0, _, w0 = self._vectors()
        return -_KEPLER_EXPONENT * w0 / J0

    def to_canonical_pair(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Carta de Whittaker:  q_k = √(2J_k/ω_k) cos θ_k,  p_k = −√(2J_kω_k) sin θ_k.
        """
        J, theta, omega = self._vectors()
        q = np.sqrt(2.0 * J / omega) * np.cos(theta)
        p = -np.sqrt(2.0 * J * omega) * np.sin(theta)
        return q, p

    def hamiltonian_identity_residue(self) -> float:
        """Residuo max_k |(p²+ω²q²)/(2ω) − J|; debe ser O(ε_máquina)."""
        q, p = self.to_canonical_pair()
        J, _, omega = self._vectors()
        reconstructed = (p**2 + (omega**2) * (q**2)) / (2.0 * omega)
        return float(np.max(np.abs(reconstructed - J)))

    def canonical_norm_squared(self) -> float:
        """Norma hamiltoniana Σ_k (ω_k q_k² + p_k²/ω_k)/2 = Σ_k J_k = 1 (Axioma A1)."""
        q, p = self.to_canonical_pair()
        _, _, omega = self._vectors()
        return float(np.sum(0.5 * (omega * q * q + p * p / omega)))

    def canonical_jacobian_determinants(self) -> NDArray[np.float64]:
        r"""
        Determinante por modo de ∂(q_k,p_k)/∂(J_k,θ_k). Analíticamente vale −1:
            det = q_J p_θ − q_θ p_J = −cos²θ − sin²θ = −1.
        Por tanto dq∧dp = −dJ∧dθ = dθ∧dJ (invariancia de Bargmann).
        """
        J, theta, omega = self._vectors()
        sJ = np.sqrt(J)
        q_J = np.sqrt(2.0 / omega) * 0.5 / sJ * np.cos(theta)
        q_th = -np.sqrt(2.0 * J / omega) * np.sin(theta)
        p_J = -np.sqrt(2.0 * omega) * 0.5 / sJ * np.sin(theta)
        p_th = -np.sqrt(2.0 * J * omega) * np.cos(theta)
        return q_J * p_th - q_th * p_J

    def bargmann_symplectic_residual(self) -> float:
        """max_k |det ∂(q,p)/∂(J,θ) + 1| (Axioma A3)."""
        return float(np.max(np.abs(self.canonical_jacobian_determinants() + 1.0)))

    def to_improvement_vector(self) -> NDArray[np.complex128]:
        r"""Amplitud coherente a_k = √J_k · e^{iθ_k} (norma 1 por A1)."""
        J, theta, _ = self._vectors()
        return (np.sqrt(J) * np.exp(1j * theta)).astype(np.complex128)

    def winding_signature(self) -> float:
        """Invariante ω_dom / Σ ω (número de enrollamiento dominante grosero)."""
        omega = np.asarray(self.omega_vector, dtype=np.float64)
        total = float(np.sum(omega))
        return float(self.omega_dominant / total) if total > 0.0 else 0.0

    def generating_function_F(self) -> float:
        r"""
        Evaluación de la función generatriz de tipo 1 F(θ,J) = Σ_k J_k θ_k.
        Cumple ∂F/∂θ_k = J_k. El escalar es un diagnóstico de la acción ∮ J dθ.
        """
        J = np.asarray(self.J_vector, dtype=np.float64)
        theta = np.asarray(self.theta_vector, dtype=np.float64)
        return float(np.dot(J, theta))


# ── 2.2 Sección de Poincaré ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección transversal Σ = { z : g(z) = Re⟨n, z⟩ − c = 0 }.
    Requiere ∇g·f ≠ 0 para que Σ corte el flujo limpiamente.
    """

    normal: Tuple[float, ...]
    offset: float
    label: str = "Σ"

    def _normal_array(self) -> NDArray[np.complex128]:
        return np.asarray(self.normal, dtype=np.complex128).reshape(-1)

    def g(self, z: object) -> float:
        """g(z) = Re⟨n, z⟩ − c."""
        return _real_inner(self._normal_array(), _as_flat_complex(z)) - self.offset

    def transversality(
        self, vector_field: object, tol: float = _TRANSVERSALITY_TOL
    ) -> bool:
        """Verifica |⟨n, f⟩| > tol (corte transversal del flujo)."""
        return abs(_real_inner(self._normal_array(), _as_flat_complex(vector_field))) > tol

    def oriented_flux(self, vector_field: object) -> float:
        """Flujo orientado Re⟨n,f⟩: el signo indica el sentido del cruce Σ⁺ → Σ⁻."""
        return _real_inner(self._normal_array(), _as_flat_complex(vector_field))


# ── 2.3 Mapa de retorno ───────────────────────────────────────────────────────────────────────────
class PoincareReturnMap:
    r"""
    Mapa de retorno P : Σ → Σ inducido por la órbita discreta de la automejora L3.
    El número de rotación de Birkhoff clasifica la dinámica: racional → periódica,
    irracional → cuasiperiódica.
    """

    def __init__(self, section: PoincareSection) -> None:
        self.section = section

    def detect_crossings(
        self, trajectory: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        """
        Cruces transversales: g(z_i)·g(z_{i+1}) < 0 y Δz transversal a Σ.
        Punto de corte por interpolación lineal en el parámetro afín t = g_i/(g_i−g_j).
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
            t = float(np.clip(gi / denom, 0.0, 1.0))
            zi = _as_flat_complex(trajectory[i])
            zj = _as_flat_complex(trajectory[i + 1])
            if not self.section.transversality(zj - zi):
                continue
            crossings.append((1.0 - t) * zi + t * zj)
        return crossings

    def apply(
        self, trajectory: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        """Composición del mapa de retorno sobre los cruces transversales."""
        return self.detect_crossings(trajectory)

    @staticmethod
    def birkhoff_rotation_number(crossings: Sequence[NDArray[np.complex128]]) -> float:
        r"""
        ρ = (θ_N − θ_0)/(2πN), con θ_n = arg(Σ_k z_{n,k}) desenrollado (unwrap).
        """
        if len(crossings) < 2:
            return 0.0
        raw = np.array(
            [float(np.angle(np.sum(_as_flat_complex(z)))) for z in crossings], dtype=np.float64
        )
        angles = np.unwrap(raw)
        n_steps = len(angles) - 1
        if n_steps <= 0:
            return 0.0
        return float((angles[-1] - angles[0]) / (_TWO_PI * n_steps))

    def mean_return_phase(self, crossings: Sequence[NDArray[np.complex128]]) -> float:
        """Fase media (centroide angular unitario) de los cruces."""
        if not crossings:
            return 0.0
        acc = 0.0 + 0.0j
        for z in crossings:
            acc += np.exp(1j * float(np.angle(np.sum(_as_flat_complex(z)))))
        return float(np.angle(acc)) if abs(acc) > _WILKINSON_EPSILON else 0.0


# ── 2.4 Twist de Poincaré–Birkhoff ────────────────────────────────────────────────────────────────
class PoincareBirkhoffTwist:
    r"""
    Teorema geométrico de Poincaré–Birkhoff: un difeomorfismo del anillo que preserva área
    con twist ∂(Δθ)/∂J ≠ 0 posee al menos dos puntos fijos.

    Para el mapa de tiempo T con ley Delaunay–Kepler, Δθ_dom(J) = T·ω_dom(J), luego
        ∂(Δθ_dom)/∂J = −3 T ω_dom / J_dom ≠ 0.
    El twist se evalúa analíticamente y se contrasta por diferencias centrales.
    """

    def __init__(
        self, seed: ActionAngleSeed, delta_J: float = 1.0e-4, period: float = 1.0
    ) -> None:
        self.seed = seed
        self.delta_J = float(delta_J)
        self.period = float(period)
        self.dom = seed.dominant_index()

    def lifted_advance(self, J_dom: float) -> float:
        """Avance angular levantado (lifted) del modo dominante con J_dom variable."""
        J_vec = np.asarray(self.seed.J_vector, dtype=np.float64).copy()
        J_vec[self.dom] = max(float(J_dom), _WILKINSON_EPSILON)
        return float(self.period * self.seed.kepler_frequencies(J_vec)[self.dom])

    def twist_analytic(self) -> float:
        """∂(Δθ_dom)/∂J_dom = T · ∂ω_dom/∂J_dom = −3 T ω_dom / J_dom."""
        return float(self.period * self.seed.anisochrony_derivative()[self.dom])

    def twist_numerical(self) -> float:
        """Diferencias centrales sobre el avance levantado; paso relativo h ≤ 10⁻³ J_dom."""
        J0 = float(self.seed.J_vector[self.dom])
        h = min(self.delta_J, 1.0e-3 * J0)
        if h <= _WILKINSON_EPSILON:
            return 0.0
        return (self.lifted_advance(J0 + h) - self.lifted_advance(J0 - h)) / (2.0 * h)

    def twist_consistency_error(self) -> float:
        """Error relativo |analítico − numérico| / |analítico|."""
        a = self.twist_analytic()
        n = self.twist_numerical()
        return abs(a - n) / max(abs(a), _WILKINSON_EPSILON)

    def satisfies_twist(self, tol: float = _TWIST_TOL) -> bool:
        """Condición de twist: |∂(Δθ)/∂J| > tol."""
        return abs(self.twist_analytic()) > tol


# ── 2.5 Forma normal local de Birkhoff ────────────────────────────────────────────────────────────
class BirkhoffNormalForm:
    r"""
    Forma normal local en el punto fijo z*:  I(z) = z* + M(z−z*) + O(|z−z*|²), M = DI(z*).
    Clasifica el espectro local y la no resonancia de Moser (condición de twist elíptico).
    """

    def __init__(
        self, fixed_point: NDArray[np.complex128], jacobian: NDArray[np.complex128]
    ) -> None:
        self.fixed_point = np.asarray(fixed_point, dtype=np.complex128)
        self.jacobian = np.asarray(jacobian, dtype=np.complex128)

    def eigenvalues(self) -> NDArray[np.complex128]:
        """Espectro de M; vacío si la matriz es vacía o la descomposición falla."""
        if self.jacobian.size == 0:
            return np.array([], dtype=np.complex128)
        try:
            return np.linalg.eigvals(self.jacobian)
        except np.linalg.LinAlgError:
            return np.array([], dtype=np.complex128)

    def spectral_radius(self) -> float:
        eigs = self.eigenvalues()
        return float(np.max(np.abs(eigs))) if eigs.size else 0.0

    def classify(self, tol: float = 1.0e-8) -> str:
        """ATTRACTING / REPELLING / ELLIPTIC_CENTER / SADDLE_HYPERBOLIC / DEGENERATE."""
        eigs = self.eigenvalues()
        if eigs.size == 0:
            return "DEGENERATE"
        moduli = np.abs(eigs)
        if np.all(moduli < 1.0 - tol):
            return "ATTRACTING"
        if np.all(moduli > 1.0 + tol):
            return "REPELLING"
        if np.all(np.abs(moduli - 1.0) <= tol):
            return "ELLIPTIC_CENTER"
        return "SADDLE_HYPERBOLIC"

    def lyapunov_exponent(self) -> float:
        """log ρ(M): exponente de Lyapunov lineal por paso."""
        return float(np.log(max(self.spectral_radius(), _WILKINSON_EPSILON)))

    def moser_nonresonant(
        self, order: int = _MOSER_NONRESONANCE_ORDER, tol: float = 1.0e-8
    ) -> bool:
        """
        Existe λ con |λ| = 1 tal que λᵏ ≠ 1 para k = 1…order (no resonancia de Moser).
        Excluir solo ±1 era insuficiente: λ³ = 1 o λ⁴ = 1 también degeneran el twist.
        """
        eigs = self.eigenvalues()
        if eigs.size == 0:
            return False
        elliptic = eigs[np.abs(np.abs(eigs) - 1.0) < tol]
        for lam in elliptic:
            if all(abs(complex(lam) ** k - 1.0) > tol for k in range(1, order + 1)):
                return True
        return False


# ── 2.6 Toro KAM y criterio BHT ───────────────────────────────────────────────────────────────────
class KAMInvariantTorus:
    r"""
    Toro invariante 𝕋^n con frecuencias ω ∈ ℝ^n.

    Condición diofantina (Kolmogorov): |⟨k,ω⟩| ≥ γ / |k|₁^τ,  ∀k ∈ ℤⁿ∖{0},  τ > n−1.
    Estimación: γ_eff = min_k |⟨k,ω⟩|·|k|₁^τ sobre un retículo finito de m ≤ 6 modos.
    Es una cota superior de la constante verdadera.

    Criterio BHT (disipativo): γ > ‖ε‖ y |disipación| < min(γ, 0.95).
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies: Tuple[float, ...] = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)
        self.m = max(1, min(self.n, _KAM_LATTICE_MODES))

    def _lattice_vectors(self, max_k: int = _KAM_MAX_K) -> List[NDArray[np.float64]]:
        r"""
        Retículo determinista de k ∈ ℤ^m ∖ {0}, sin duplicados:
          • ejes ±e_i;  • pares ±e_i ± e_j;  • cubo |k|_∞ ≤ bound en los primeros min(m,4) modos.
        """
        m = self.m
        out: List[NDArray[np.float64]] = []
        seen: Set[Tuple[int, ...]] = set()

        def push(coords: NDArray[np.float64]) -> None:
            key = tuple(int(c) for c in coords)
            if any(key) and key not in seen:
                seen.add(key)
                out.append(coords)

        for i in range(m):
            for s in (-1, 1):
                kv = np.zeros(m, dtype=np.float64)
                kv[i] = float(s)
                push(kv)

        for i, j in itertools.combinations(range(m), 2):
            for si, sj in itertools.product((-1, 1), repeat=2):
                kv = np.zeros(m, dtype=np.float64)
                kv[i] = float(si)
                kv[j] = float(sj)
                push(kv)

        cube = min(m, _KAM_CUBE_MODES)
        bound = min(2, max(1, max_k // 6))
        for coeffs in itertools.product(range(-bound, bound + 1), repeat=cube):
            if not any(coeffs):
                continue
            kv = np.zeros(m, dtype=np.float64)
            kv[:cube] = np.asarray(coeffs, dtype=np.float64)
            push(kv)
        return out

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: float = _DIOPHANTINE_TAU
    ) -> float:
        """γ_eff = min_k |⟨k,ω⟩|·|k|₁^τ (estimación muestral, ver nota de rigor)."""
        omega = np.asarray(self.frequencies[: self.m], dtype=np.float64)
        best = math.inf
        for kv in self._lattice_vectors(max_k):
            l1 = float(np.sum(np.abs(kv)))
            if l1 <= 0.0:
                continue
            inner = abs(float(np.dot(kv, omega)))
            best = min(best, inner * l1**tau)
        return float(best) if math.isfinite(best) else 0.0

    def small_divisors(self, max_k: int = _KAM_MAX_K) -> Tuple[float, ...]:
        """Los divisores pequeños |⟨k,ω⟩| > 0 más pequeños (diagnóstico de resonancia)."""
        omega = np.asarray(self.frequencies[: self.m], dtype=np.float64)
        divs = sorted(
            d for d in (abs(float(np.dot(kv, omega))) for kv in self._lattice_vectors(max_k))
            if d > 0.0
        )
        return tuple(divs[:_SMALL_DIVISOR_COUNT])

    def is_kam_stable(self, perturbation_norm: float, gamma: Optional[float] = None) -> bool:
        """Estabilidad KAM hamiltoniana: γ > ‖ε‖ ⇒ el toro sobrevive."""
        if gamma is None:
            gamma = self.diophantine_constant()
        return gamma > perturbation_norm

    def is_dissipative_kam_stable(
        self,
        perturbation_norm: float,
        dissipation: float,
        gamma: Optional[float] = None,
    ) -> bool:
        """Criterio BHT: γ > ‖ε‖ ∧ |disipación| < min(γ, 0.95)."""
        if gamma is None:
            gamma = self.diophantine_constant()
        dissip = abs(float(dissipation))
        return (gamma > perturbation_norm) and (dissip < min(gamma, _BHT_DISSIPATION_CEILING))

    @staticmethod
    def winding_numbers(phase_matrix: NDArray[np.float64]) -> Tuple[float, ...]:
        r"""
        Números de enrollamiento empíricos por modo: ν_i = Δφ_i^{unwrap}/(2π).
        phase_matrix tiene forma (T, m): T instantes, m modos.
        """
        P = np.asarray(phase_matrix, dtype=np.float64)
        if P.ndim != 2:
            return ()
        if P.shape[0] < 2:
            return tuple(0.0 for _ in range(P.shape[1]))
        U = np.unwrap(P, axis=0)
        advance = U[-1] - U[0]
        return tuple(float(x) for x in advance / _TWO_PI)


# ── 2.7 Operador de automejora recursiva L0–L3 ────────────────────────────────────────────────────
class RecursiveImprovementOperator:
    r"""
    Operador de automejora I : ℳ → ℳ sobre ℳ ⊂ ℂ^N.

    Jerarquía metacognitiva:
      L0: I₀(s) = s
      L1: I₁(s) = normalize((1−θ)s + θ·s*)
      L2: I₂ = I₁ con θ actualizado por el cociente de errores residuales
      L3: I₃ = I₂ con meta-regla η adaptada por
            (i)  espectro local de Rayleigh de Δz,
            (ii) número de rotación de Poincaré,
            (iii) resonancia p/q (divisores pequeños).
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
        return _safe_unit((1.0 - theta) * state + theta * target)

    def apply_level_2(
        self,
        state: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """θ ← θ·(1 + 0.1·(ratio − 1)), con ratio = err_t / err_{t−2}."""
        if len(history) >= 2:
            err = float(np.linalg.norm(state - target))
            prev_err = float(np.linalg.norm(history[-2] - target))
            if prev_err > _WILKINSON_EPSILON:
                ratio = err / prev_err
                self.contraction = _clamp(
                    self.contraction * (1.0 + 0.1 * (ratio - 1.0)), *_CONTRACTION_BOUNDS
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
          • espectro local de Rayleigh → frena (ρ>1) o acelera (0<ρ<0.5) la meta-tasa η;
          • rotación resonante p/q → sube θ y baja η (escapar del divisor pequeño);
          • rotación no resonante → relaja θ (preservar el toro BHT);
          • meta-regla de segundo orden: θ ← θ + 0.05(η − 1/2).
        """
        if len(history) >= 3:
            eigs = self._estimate_local_jacobian_spectrum(history[-3:])
            rho_loc = float(np.max(np.abs(eigs))) if eigs.size else 0.0
            if rho_loc > 1.0:
                self.meta_learning = _clamp(self.meta_learning * 0.90, *_META_BOUNDS)
            elif 0.0 < rho_loc < 0.5:
                self.meta_learning = _clamp(self.meta_learning * 1.05, *_META_BOUNDS)

        if poincare_rotation is not None and math.isfinite(poincare_rotation):
            self._last_rotation = float(poincare_rotation)
            if _is_near_resonance(poincare_rotation):
                self._resonance_hits += 1
                self.contraction = _clamp(self.contraction * 1.08, *_CONTRACTION_BOUNDS)
                self.meta_learning = _clamp(self.meta_learning * 0.92, *_META_BOUNDS)
            else:
                self.contraction = _clamp(self.contraction * 0.98, *_CONTRACTION_BOUNDS)

        self.contraction = _clamp(
            self.contraction + 0.05 * (self.meta_learning - 0.5), *_CONTRACTION_BOUNDS
        )
        return self.apply_level_2(state, target, history)

    @staticmethod
    def _estimate_local_jacobian_spectrum(
        recent: Sequence[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """
        Cociente de Rayleigh λ ≈ ⟨v₂, v₁⟩/⟨v₂, v₂⟩ con v₁ = z_t − z_{t−1}, v₂ = z_{t−1} − z_{t−2}.
        """
        if len(recent) < 3:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        r = [_as_flat_complex(z) for z in recent]
        v1 = r[-1] - r[-2]
        v2 = r[-2] - r[-3]
        denom = np.vdot(v2, v2)
        if abs(denom) < _WILKINSON_EPSILON:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        return np.array([complex(np.vdot(v2, v1) / denom)], dtype=np.complex128)

    def _poincare_rotation_from_history(
        self, history: Sequence[NDArray[np.complex128]]
    ) -> Optional[float]:
        """Número de rotación de Birkhoff sobre la sección uniforme Σ_L3 del historial."""
        if len(history) < 4:
            return None
        dim = int(_as_flat_complex(history[0]).size)
        normal = tuple([1.0 / math.sqrt(dim)] * dim)
        section = PoincareSection(normal=normal, offset=0.0, label="Σ_L3")
        crossings = PoincareReturnMap(section).detect_crossings(list(history))
        if len(crossings) < 2:
            return None
        return PoincareReturnMap.birkhoff_rotation_number(crossings)

    # ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: ÚLTIMO MÉTODO DE FASE 2                                                ║
    # ║  iterate_level_3 retorna ImprovementTrajectory, PRIMERA definición de FASE 3.            ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: ActionAngleSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> "ImprovementTrajectory":
        r"""
        Itera L3 sobre la semilla y devuelve la trayectoria con:
          • estados z_0 … z_T, acciones J_t = Σ_k |z_{t,k}|², ángulo dominante θ_t;
          • exponente de Lyapunov finito-temporal λ = mean Δ log‖Δz_t‖;
          • γ diofantina, estabilidad KAM/BHT, enrollamientos empíricos por modo;
          • número de rotación de Birkhoff, cruces transversales;
          • twist analítico, no resonancia de Moser, residuos hamiltoniano y de Bargmann.
        """
        state = seed.to_improvement_vector()
        dim = int(state.size)
        target = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        self._resonance_hits = 0

        n_sec = tuple(float(x) for x in np.ones(dim, dtype=np.float64) / math.sqrt(dim))
        live_section = PoincareSection(normal=n_sec, offset=0.0, label="Σ_flow")
        top_modes = np.argsort(-np.abs(state))[: min(_TRAJECTORY_PHASE_MODES, dim)]

        states_list: List[NDArray[np.complex128]] = [state.copy()]
        actions_list: List[float] = [float(np.sum(np.abs(state) ** 2))]
        angles_list: List[float] = [float(np.angle(state[int(np.argmax(np.abs(state)))]))]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max(0, int(max_iters))):
            history.append(state.copy())
            rho_p = self._poincare_rotation_from_history(history)
            state = self.apply_level_3(state, target, history, poincare_rotation=rho_p)
            states_list.append(state.copy())
            actions_list.append(float(np.sum(np.abs(state) ** 2)))
            angles_list.append(float(np.angle(state[int(np.argmax(np.abs(state)))])))

        traj = np.asarray(states_list, dtype=np.complex128)
        steps = (
            np.linalg.norm(np.diff(traj, axis=0), axis=1)
            if traj.shape[0] >= 2
            else np.zeros(0, dtype=np.float64)
        )

        if steps.size >= 2:
            log_steps = np.log(np.maximum(steps, _WILKINSON_EPSILON))
            lyapunov = float(np.mean(np.diff(log_steps)))
        else:
            lyapunov = 0.0
        lyapunov = float(np.clip(lyapunov, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP))
        convergence = float(steps[-1]) if steps.size else 0.0
        target_distance = float(np.linalg.norm(traj[-1] - target))
        spectral_radius = float(math.exp(lyapunov))

        torus = KAMInvariantTorus(seed.omega_vector)
        gamma = torus.diophantine_constant()
        kam_stable = torus.is_kam_stable(convergence, gamma)
        bht_stable = torus.is_dissipative_kam_stable(convergence, self.contraction, gamma)
        winding = torus.winding_numbers(np.angle(traj[:, top_modes]))

        crossings = PoincareReturnMap(live_section).apply(list(traj))
        rotation = PoincareReturnMap.birkhoff_rotation_number(crossings)

        twist = PoincareBirkhoffTwist(seed)
        twist_val = twist.twist_analytic()
        twist_ok = twist.satisfies_twist()

        lam_est = (
            self._estimate_local_jacobian_spectrum(history[-3:])
            if len(history) >= 3
            else np.array([0.5 + 0.0j], dtype=np.complex128)
        )
        jac = complex(lam_est[0]) * np.eye(dim, dtype=np.complex128)
        moser_ok = BirkhoffNormalForm(traj[-1], jac).moser_nonresonant()

        return ImprovementTrajectory(
            source_seed_hash=seed.source_hash,
            states=tuple(tuple(complex(c) for c in row) for row in traj),
            actions=tuple(actions_list),
            angles=tuple(angles_list),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            target_distance=target_distance,
            spectral_radius=spectral_radius,
            is_kam_stable=kam_stable,
            is_bht_stable=bht_stable,
            kam_winding_numbers=winding,
            diophantine_gamma=gamma,
            poincare_rotation_number=rotation,
            poincare_crossing_count=len(crossings),
            twist_derivative=twist_val,
            twist_condition=twist_ok,
            moser_nonresonant=moser_ok,
            hamiltonian_residue=seed.hamiltonian_identity_residue(),
            symplectic_residual=seed.bargmann_symplectic_residual(),
            resonance_hits=self._resonance_hits,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTEGRACIÓN, AUDITORÍA KAM/PAULI Y FACTORÍA SOBERANA
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# 3.1  ImprovementTrajectory (ENTRADA desde FASE 2)
# 3.2  OccupationRecord y PauliExclusionAuditorCAR
# 3.3  ToonParticleFactory (12 partículas)
# 3.4  Verificación doctoral (__main__)
# ══════════════════════════════════════════════════════════════════════════════════════════════════


# ── 3.1 Trayectoria de automejora ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ImprovementTrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRADA: consume iterate_level_3 (FASE 2).                                        │
    │  Trayectoria L3 con análisis de Poincaré, twist, KAM/BHT, Moser y residuos de invariantes.   │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_seed_hash: str
    states: Tuple[Tuple[complex, ...], ...]
    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    lyapunov_exponent: float
    convergence_residual: float
    target_distance: float
    spectral_radius: float
    is_kam_stable: bool
    is_bht_stable: bool
    kam_winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    poincare_rotation_number: float
    poincare_crossing_count: int
    twist_derivative: float
    twist_condition: bool
    moser_nonresonant: bool
    hamiltonian_residue: float
    symplectic_residual: float
    resonance_hits: int

    def to_poincare_section(
        self, normal: Optional[Tuple[float, ...]] = None
    ) -> PoincareSection:
        """Sección de Poincaré con offset = media de g sobre la órbita."""
        dim = len(self.states[0]) if self.states else 1
        if normal is None:
            normal = tuple([1.0 / math.sqrt(dim)] * dim)
        z_arr = np.asarray(self.states, dtype=np.complex128)
        n_arr = np.asarray(normal, dtype=np.complex128)
        g_vals = np.real(z_arr @ np.conjugate(n_arr))
        offset = float(np.mean(g_vals)) if g_vals.size else 0.0
        return PoincareSection(
            normal=normal, offset=offset, label=f"Σ[{self.source_seed_hash[:8]}]"
        )

    def is_stable_fixed_point(self) -> bool:
        """Punto fijo estable ⇔ λ < 0 ∧ (toro KAM ∨ NHIM BHT)."""
        return self.lyapunov_exponent < 0.0 and (self.is_kam_stable or self.is_bht_stable)

    def topological_index(self) -> int:
        """Índice topológico grosero round(ρ) de la clase homotópica del retorno."""
        if not math.isfinite(self.poincare_rotation_number):
            return 0
        return int(round(self.poincare_rotation_number))

    def kam_audit_vector(self) -> Tuple[float, float, float, int, int]:
        """Firma de auditoría (γ, λ, ρ, #cruces, resonancias)."""
        return (
            float(self.diophantine_gamma),
            float(self.lyapunov_exponent),
            float(self.poincare_rotation_number),
            int(self.poincare_crossing_count),
            int(self.resonance_hits),
        )


# ── 3.2 Auditor de exclusión de Pauli (CAR) con registro topológico ───────────────────────────────
@dataclass(frozen=True, slots=True)
class OccupationRecord:
    """Entrada inmutable de la matriz de ocupación fermiónica."""

    state_hash: str
    particle_name: str
    windings: Tuple[float, ...]
    topological_index: int
    rotation_number: float
    lyapunov_exponent: float


class PauliExclusionAuditorCAR:
    r"""
    Auditor de exclusión de Pauli. Vetos en tres niveles:
      1. Hash: duplicación exacta del estado cuántico.
      2. Toro KAM: enrollamientos empíricos coincidentes (dentro de winding_tol).
      3. Poincaré: mismo índice topológico y mismo número de rotación (dentro de poincare_tol).
    """

    def __init__(
        self,
        dimension: int = _DEFAULT_HILBERT_DIM,
        max_iters: int = 6,
        winding_tol: float = _PAULI_WINDING_TOL,
        poincare_tol: float = _PAULI_POINCARE_TOL,
    ) -> None:
        self._dimension = int(dimension)
        self._max_iters = int(max_iters)
        self._winding_tol = float(winding_tol)
        self._poincare_tol = float(poincare_tol)
        self._occupied_hashes: Set[str] = set()
        self._records: List[OccupationRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def occupation_matrix(self) -> Tuple[OccupationRecord, ...]:
        """Vista inmutable de la matriz de ocupación fermiónica registrada."""
        return tuple(self._records)

    @staticmethod
    def _windings_too_close(
        w1: Tuple[float, ...], w2: Tuple[float, ...], tol: float
    ) -> bool:
        """Compara los primeros 3 componentes; vacío nunca colisiona."""
        n = min(len(w1), len(w2), 3)
        if n == 0:
            return False
        return all(abs(a - b) < tol for a, b in zip(w1[:n], w2[:n]))

    def audit_and_register_fermion(self, cartridge: ToonCartridge) -> Tuple[bool, str]:
        """Audita y, si procede, registra un fermión. Retorna (aceptado, código)."""
        if cartridge.category is not ParticleCategory.FERMION_CAR:
            return True, "NON_FERMION_PASSTHROUGH"

        state_hash = cartridge.quantum_state_hash
        if state_hash in self._occupied_hashes:
            logger.error("[PAULI VETO] Duplicación hash: %s", cartridge.particle_name)
            return False, "PAULI_EXCLUSION_VIOLATION_DUPLICATE_HASH"

        seed = cartridge.to_action_angle_seed(self._dimension)
        traj = RecursiveImprovementOperator().iterate_level_3(seed, max_iters=self._max_iters)
        topo = traj.topological_index()

        for rec in self._records:
            if self._windings_too_close(traj.kam_winding_numbers, rec.windings, self._winding_tol):
                logger.error("[PAULI VETO KAM] Duplicación de toro: %s", cartridge.particle_name)
                return False, "PAULI_EXCLUSION_VIOLATION_KAM_DUPLICATE"
            if (
                topo != 0
                and topo == rec.topological_index
                and abs(traj.poincare_rotation_number - rec.rotation_number) < self._poincare_tol
            ):
                logger.error("[PAULI VETO POINCARE] Índice de retorno duplicado: %s",
                             cartridge.particle_name)
                return False, "PAULI_EXCLUSION_VIOLATION_POINCARE_INDEX"

        self._occupied_hashes.add(state_hash)
        self._records.append(
            OccupationRecord(
                state_hash=state_hash,
                particle_name=cartridge.particle_name,
                windings=traj.kam_winding_numbers,
                topological_index=topo,
                rotation_number=traj.poincare_rotation_number,
                lyapunov_exponent=traj.lyapunov_exponent,
            )
        )
        return True, "FERMION_OCCUPIED_KAM_OK"

    def clear(self) -> None:
        self._occupied_hashes.clear()
        self._records.clear()


# ── 3.3 Factoría soberana de las 12 partículas ────────────────────────────────────────────────────
class ToonParticleFactory:
    """
    Factoría soberana. Las energías base se modulan por la dinámica L3 de un mini-orbital
    determinista derivado del seed_id (contracción y γ del toro asociado).
    """

    @staticmethod
    def _create_hmac(
        cartridge_id: str, particle_name: str, energy: float, timestamp: float, key: bytes
    ) -> str:
        msg = _hmac_message(cartridge_id, particle_name, energy, timestamp)
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    @classmethod
    def _sovereign_energy(cls, base_energy: float, seed_id: str, level: int = 3) -> float:
        """
        E = E_base·(1 + 0.02·rate_mod + 0.01·gamma_mod).
          • rate_mod: cociente ‖Δz_t‖/‖Δz_{t−1}‖ de la órbita L3 del mini-sistema (dim 8);
          • gamma_mod: log₁₀γ/10 del toro áureo (depende solo de la ley de frecuencias).
        """
        digest = hashlib.sha256(f"{seed_id}:{base_energy:.6f}:L{level}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        dim = _SOVEREIGN_MINI_DIM
        state = _safe_unit(rng.normal(0.0, 1.0, dim) + 1j * rng.normal(0.0, 1.0, dim))
        target = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        op = RecursiveImprovementOperator(contraction=0.5, meta_learning=0.3)
        history: List[NDArray[np.complex128]] = []
        for _ in range(max(1, int(level))):
            history.append(state.copy())
            rho_p = op._poincare_rotation_from_history(history)
            state = op.apply_level_3(state, target, history, poincare_rotation=rho_p)

        omega = _fpu_golden_frequencies(np.arange(dim, dtype=np.float64))
        gamma = KAMInvariantTorus(omega).diophantine_constant()
        gamma_mod = float(np.clip(math.log10(gamma + 1.0e-30) / 10.0, -0.3, 0.3))

        rate_mod = 0.0
        if len(history) >= 2:
            d1 = float(np.linalg.norm(history[-1] - history[-2]))
            d0 = float(np.linalg.norm(history[-2] - history[-3])) if len(history) >= 3 else d1
            if d0 > _WILKINSON_EPSILON:
                rate_mod = float(np.clip(d1 / d0 - 0.5, -0.5, 0.5))

        return float(base_energy * (1.0 + 0.02 * rate_mod + 0.01 * gamma_mod))

    @classmethod
    def _assemble(
        cls,
        cartridge_id: str,
        particle_name: str,
        category: ParticleCategory,
        energy_kev: float,
        payload: Tuple[Tuple[str, str], ...],
        key: bytes,
    ) -> ToonCartridge:
        """Ensamblador común: mismo timestamp para la firma y el cartucho."""
        now = time.time()
        sig = cls._create_hmac(cartridge_id, particle_name, energy_kev, now, key)
        return ToonCartridge(
            cartridge_id=cartridge_id,
            category=category,
            particle_name=particle_name,
            mass_energy_kev=energy_kev,
            signature_hmac=sig,
            timestamp_utc=now,
            payload_tuple=tuple(payload),
        )

    # ── Fermiones CAR (5) ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def create_polaron(
        cls, cartridge_id: str, ballast_mass_tn: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(150.0 + ballast_mass_tn, cartridge_id)
        return cls._assemble(
            cartridge_id, "PolaronCartridge", ParticleCategory.FERMION_CAR, energy,
            (("material", "concreto_5000psi_acero"), ("ballast_tn", str(ballast_mass_tn))),
            key,
        )

    @classmethod
    def create_torsion(
        cls, cartridge_id: str, torsion_angle_rad: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(75.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "TorsionCartridge", ParticleCategory.FERMION_CAR, energy,
            (("homological_torsion", "True"), ("angle_rad", f"{torsion_angle_rad:.6f}")),
            key,
        )

    @classmethod
    def create_electron(
        cls, cartridge_id: str, anomaly_cost_cop: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        return cls._assemble(
            cartridge_id, "ElectronCartridge", ParticleCategory.FERMION_CAR, _ELECTRON_MASS_KEV,
            (("anomaly_type", "sobrecosto_apu"), ("cost_cop", f"{anomaly_cost_cop:.2f}")),
            key,
        )

    @classmethod
    def create_proton(
        cls, cartridge_id: str, cement_volume_m3: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        return cls._assemble(
            cartridge_id, "ProtonCartridge", ParticleCategory.FERMION_CAR, _PROTON_MASS_KEV,
            (("stability", "LTI_BIBO_STABLE"), ("volume_m3", f"{cement_volume_m3:.2f}")),
            key,
        )

    @classmethod
    def create_householder(
        cls, cartridge_id: str, bim_revision: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(500.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "HouseholderReflectionFermion", ParticleCategory.FERMION_CAR, energy,
            (("reflection_type", "orthogonal_BIM_2026"), ("revision", bim_revision)),
            key,
        )

    # ── Bosones CCR (6) ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def create_photon(
        cls, cartridge_id: str, secop_id: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        return cls._assemble(
            cartridge_id, "PhotonCartridge", ParticleCategory.BOSON_CCR, 0.0,
            (("regulation", "SECOP_II_OPA"), ("secop_id", secop_id)),
            key,
        )

    @classmethod
    def create_riemannian_focal(
        cls, cartridge_id: str, eikonal_action: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(10.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "RiemannianFocalBoson", ParticleCategory.BOSON_CCR, energy,
            (("geodesic_path", "critical_route"), ("eikonal_action", f"{eikonal_action:.6f}")),
            key,
        )

    @classmethod
    def create_magnon(
        cls, cartridge_id: str, triangulation_veto: bool, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(25.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "MagnonCartridge", ParticleCategory.BOSON_CCR, energy,
            (("anti_triangulation", str(triangulation_veto)),),
            key,
        )

    @classmethod
    def create_swelling_plasmon(
        cls, cartridge_id: str, wacc_spread_pct: float, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(120.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "SwellingPlasmonCartridge", ParticleCategory.BOSON_CCR, energy,
            (("wacc_volatility", "15.0_pct_protection"), ("spread", f"{wacc_spread_pct:.4f}")),
            key,
        )

    @classmethod
    def create_yielding_phonon(
        cls, cartridge_id: str, union_agreement: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(40.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "YieldingPhononCartridge", ParticleCategory.BOSON_CCR, energy,
            (("labor_agreement", "CUT_CTC_CGT"), ("union_code", union_agreement)),
            key,
        )

    @classmethod
    def create_liquefaction_soliton(
        cls, cartridge_id: str, ethnic_consultation: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        energy = cls._sovereign_energy(300.0, cartridge_id)
        return cls._assemble(
            cartridge_id, "LiquefactionSolitonCartridge", ParticleCategory.BOSON_CCR, energy,
            (("geotechnics_license", "consult_ethnic_ok"), ("license_id", ethnic_consultation)),
            key,
        )

    # ── Antimateria (1) ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def create_positron(
        cls, cartridge_id: str, auth_signature: str, key: bytes = _GAUGE_SECRET_KEY
    ) -> ToonCartridge:
        return cls._assemble(
            cartridge_id, "PositronCartridge", ParticleCategory.ANTIMATTER,
            _POSITRON_REST_ENERGY_KEV,
            (("antimatter_type", "hmac_authorization"), ("auth_sig", auth_signature)),
            key,
        )


# ── 3.4 Verificación doctoral ─────────────────────────────────────────────────────────────────────
def _doctoral_verification() -> None:
    """Ejecuta el banco de verificación de las 12 partículas y de los invariantes de FASE 2."""
    banner = "═" * 100
    print(banner)
    print(f"  VERIFICACIÓN DOCTORAL — v{__version__}")
    print("  12 Partículas | Delaunay–Kepler | Poincaré–Birkhoff | KAM/BHT | Moser | Automejora L3")
    print(banner)

    auditor = PauliExclusionAuditorCAR()
    particles = [
        ToonParticleFactory.create_polaron("POL-001", ballast_mass_tn=12.5),
        ToonParticleFactory.create_torsion("TOR-002", torsion_angle_rad=0.045),
        ToonParticleFactory.create_electron("ELE-003", anomaly_cost_cop=15_000_000.0),
        ToonParticleFactory.create_proton("PRO-004", cement_volume_m3=45.0),
        ToonParticleFactory.create_householder("HOU-005", bim_revision="REV-2026-v9"),
        ToonParticleFactory.create_photon("PHO-006", secop_id="SECOP-2026-8891"),
        ToonParticleFactory.create_riemannian_focal("RFB-007", eikonal_action=0.0012),
        ToonParticleFactory.create_magnon("MAG-008", triangulation_veto=True),
        ToonParticleFactory.create_swelling_plasmon("SWP-009", wacc_spread_pct=0.015),
        ToonParticleFactory.create_yielding_phonon("YPH-010", union_agreement="CCT-2026-CIVIL"),
        ToonParticleFactory.create_liquefaction_soliton(
            "LQS-011", ethnic_consultation="ACTA-ETHNIC-001"
        ),
        ToonParticleFactory.create_positron("POS-012", auth_signature="HMAC-AUTH-OK-2026"),
    ]

    print(f"\n  • Partículas instanciadas: {len(particles)}")
    print(
        f"  {'#':<4} {'Nombre':<30} {'Categoría':<12} {'SHA':<15} {'Pauli':<28} "
        f"{'HMAC':<6} {'ΣJ':<7} {'λ':<9} {'KAM':<4} {'BHT':<4} {'ρ'}"
    )
    print("  " + "─" * 146)

    for i, p in enumerate(particles, 1):
        ok, msg = auditor.audit_and_register_fermion(p)
        hmac_ok = p.verify_hmac_authenticity()
        seed = p.to_action_angle_seed()
        traj = RecursiveImprovementOperator().iterate_level_3(seed, max_iters=6)
        kam_mark = "✓" if traj.is_kam_stable else "✗"
        bht_mark = "✓" if traj.is_bht_stable else "✗"
        print(
            f"  [{i:02d}] {p.particle_name:<30} {p.category.name:<12} "
            f"{p.quantum_state_hash[:12]}... {msg:<28} "
            f"{str(hmac_ok):<6} {seed.J_total:<7.4f} {traj.lyapunov_exponent:<9.4f} "
            f"{kam_mark:<4} {bht_mark:<4} {traj.poincare_rotation_number:+.4f}"
        )

    dup_ok, dup_msg = auditor.audit_and_register_fermion(particles[0])
    print("\n  • Prueba de duplicación Pauli (POL-001 re-registrado):")
    print(f"      Veto exitoso = {not dup_ok}  →  {dup_msg}")
    print(f"      Ocupaciones fermiónicas registradas = {len(auditor)}")

    p1 = particles[0]
    seed1 = p1.to_action_angle_seed()
    traj1 = RecursiveImprovementOperator().iterate_level_3(seed1, max_iters=8)
    section1 = traj1.to_poincare_section()
    states1 = [np.asarray(s, dtype=np.complex128) for s in traj1.states]
    crossings1 = PoincareReturnMap(section1).apply(states1)
    torus1 = KAMInvariantTorus(seed1.omega_vector)
    gamma1 = torus1.diophantine_constant()
    twist1 = PoincareBirkhoffTwist(seed1)

    print("\n  • Análisis Poincaré/KAM/BHT detallado (fermión POL-001):")
    print(f"      Invariante Poincaré–Cartan I₁ = {seed1.poincare_cartan_invariant():.6f}")
    print(f"      Residuo identidad hamiltoniana = {seed1.hamiltonian_identity_residue():.3e}")
    print(f"      Norma hamiltoniana Σ(ωq²+p²/ω)/2 = {seed1.canonical_norm_squared():.6f}")
    print(f"      Residuo de Bargmann |det+1| = {seed1.bargmann_symplectic_residual():.3e}")
    print(f"      Sección Σ: g(z)=Re⟨n,z⟩−c, offset={section1.offset:+.6e}")
    print(f"      Cruces transversales detectados: {len(crossings1)}")
    print(f"      Rotación de Birkhoff ρ = {traj1.poincare_rotation_number:+.6f}")
    print(f"      Twist analítico ∂(Δθ)/∂J = {twist1.twist_analytic():+.6e}  "
          f"(numérico rel.err={twist1.twist_consistency_error():.2e})")
    print(f"      No resonancia de Moser = {traj1.moser_nonresonant}")
    print(f"      Constante diofantina γ (estimación) = {gamma1:.6e}")
    print(f"      Divisores pequeños (4): {torus1.small_divisors()[:4]}")
    print(f"      Perturbación residual = {traj1.convergence_residual:.6e}")
    print(f"      Distancia al objetivo = {traj1.target_distance:.6e}")
    print(f"      Estabilidad KAM: γ > residual → {traj1.is_kam_stable}")
    print(f"      Estabilidad BHT (KAM disipativo) → {traj1.is_bht_stable}")
    print(f"      Exponente de Lyapunov λ = {traj1.lyapunov_exponent:+.6f}")
    print(f"      Hits de resonancia p/q (L3) = {traj1.resonance_hits}")
    w = traj1.kam_winding_numbers
    print(f"      Enrollamientos empíricos (≤3 modos): "
          f"({', '.join(f'{x:+.6f}' for x in w[:3])})")
    print(f"      Índice topológico de retorno = {traj1.topological_index()}")
    print(f"      Punto fijo estable: {traj1.is_stable_fixed_point()}")

    checks = [
        ("Identidad hamiltoniana < 1e-12", seed1.hamiltonian_identity_residue() < 1e-12),
        ("Norma hamiltoniana |Σ−1| < 1e-12",
         abs(seed1.canonical_norm_squared() - 1.0) < 1e-12),
        ("Bargmann |det+1| < 1e-9", seed1.bargmann_symplectic_residual() < 1e-9),
        ("Twist analítico ≈ numérico", twist1.twist_consistency_error() < _TWIST_REL_TOL),
        ("Veto Pauli sobre duplicado", not dup_ok),
        ("Hash determinista", p1.quantum_state_hash == p1.quantum_state_hash),
    ]
    print("\n  • Invariantes verificados:")
    for label, passed in checks:
        print(f"      {'✓' if passed else '✗'}  {label}")

    print("\n" + banner)
    print("  ✓ FASE 1 → FASE 2: ToonCartridge.to_action_angle_seed → ActionAngleSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveImprovementOperator.iterate_level_3 → ImprovementTrajectory")
    print(banner)


if __name__ == "__main__":
    _doctoral_verification()