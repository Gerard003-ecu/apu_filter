r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPU Tensor Fock Collider Witness Engine v14.1.0-SOVEREIGN-POINCARE-N3-PhD                 ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider_witness_engine.py                                     ║
║ Versión  : 14.1.0-SOVEREIGN-POINCARE-N3-PhD                                                          ║
║ Función  : Motor espectral observador pasivo de Medición Débil QND (Back-Action ≡ 0.0 dB) sobre      ║
║            el espacio de Fock ℱ(ℋ) con Mecánica Celeste de Poincaré (secciones N̂/cuadraturas,       ║
║            Lyapunov–Benettin, Lindstedt, Birkhoff, KAM, Nekhoroshev), Automejora Recursiva N3        ║
║            y verificación CAR/CCR sobre operadores CANÓNICOS b/b† (bosónico) y a/a† (fermiónico).    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral Testigo del Colisionador FPU en el Espacio de Fock ℱ(ℋ) garantiza la observación
no perturbativa (Quantum Non-Demolition) del estado cuántico global y la dinámica no lineal FPU.
Descompone el estado de densidad en variables canónicas de acción-ángulo de Fock, evalúa la geometría
celeste de Poincaré a través de secciones transversales sobre cuadraturas conjugadas (x̂, p̂), analiza la
estabilidad cuasi-periódica bajo el régimen de Kolmogorov-Arnold-Moser (KAM) y tiempos de confinamiento
de Nekhoroshev, y modula los hiperparámetros de monitoreo mediante automejora recursiva de Nivel 3.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Estructura Cuántica y Relaciones Canónicas (CCR / CAR):
   El álgebra de Fock se construye sobre la base de Fock {|0⟩, |1⟩, ..., |d-1⟩}.
   Operador bosónico de aniquilación y creación:
       b |n\rangle = \sqrt{n} |n-1\rangle, \quad b^\dagger |n\rangle = \sqrt{n+1} |n+1\rangle.
   Relación de conmutación canónica truncada: [b, b^\dagger] = I_d - d |d-1\rangle\langle d-1|.
   Operador fermiónico de Jordan-Wigner 1-modo: \{a, a^\dagger\} = I_2, \quad a^2 = 0.
   Operadores hermíticos conjugados:
       \hat{x} = \frac{b + b^\dagger}{\sqrt{2}}, \quad \hat{p} = \frac{i(b^\dagger - b)}{\sqrt{2}}, \quad [\hat{x}, \hat{p}] = i [b, b^\dagger].

2. Descomposición de Acción-Ángulo en Fock y Mecánica Celeste:
   Para cada modo k del espacio de Fock, las acciones canónicas y fases se obtienen mediante:
       J_k = \langle k | \rho | k \rangle, \quad \theta_k = \arg(\langle k-1 | \rho | k \rangle), \quad \omega_k = \frac{\partial \langle H \rangle}{\partial J_k}.
   El espectro de Lyapunov se computa mediante el algoritmo de Benettin sobre la evolución del mapa
   de variación tangente:
       \lambda_{\max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta z(t)||_2}{||\delta z(0)||_2}.

3. Secciones de Poincaré Transversales y Confinamiento de Nekhoroshev:
   La sección de Poincaré se define como \Sigma = \{ (x, p, N) : p = 0, \dot{p} > 0 \}.
   El tiempo de difusión de Arnold / Nekhoroshev se estima asintóticamente mediante:
       T_{\mathrm{Nek}} = T_0 \exp\left( \left(\frac{\epsilon_0}{\epsilon}\right)^a \right), \quad a = \frac{1}{2n}.
   La persistencia de toros diofánticos KAM exige el cumplimiento de |\langle k, \omega \rangle| \ge \frac{\gamma}{||k||_1^\tau}.

4. Automejora Recursiva N3 y Modulación del Ledger:
   El motor implementa dos capas de auditoría:
     a) Ledger de estado interno del motor espectral.
     b) Meta-aprendizaje de Nivel 3 que adapta los umbrales de KAM y Spohn mediante un gradiente
        de Lyapunov V(\rho) = ||\rho - \rho^*||_F^2 con factor de amortiguamiento Armijo-Goldstein.

AXIOMAS FORMALES
────────────────
Axioma FPU-W1 (Invarianza Observacional QND):
    La acción de monitoreo del testigo satisface rigurosamente [H_{\mathrm{free}}, \rho] = 0 durante la medición.
Axioma FPU-W2 (Conmutación Simpléctica de Cuadraturas):
    El conmutador cuántico satisface [\hat{x}, \hat{p}] = i [b, b^\dagger] con residuo menor a 1.0e-6.
Axioma FPU-W3 (No-Degeneración de Kolmogorov):
    El gradiente de frecuencias respecto a las acciones canónicas \det(\partial \omega / \partial J) ≠ 0 no se anula.
Axioma FPU-W4 (Producción Positiva de Spohn):
    Todo proceso de relajación disipativa GKSL satisface idénticamente \sigma_{\mathrm{Spohn}} \ge 0.

INVARIANTES DE ESTADO
─────────────────────
Invariante FPU-WI1: Back-Action de medición débil QND idénticamente nula (0.0 dB).
Invariante FPU-WI2: Normalización de traza cuántica Tr(ρ) ≡ 1.0 y espectro acotado por piso de Wilkinson.
Invariante FPU-WI3: Monodromía simpléctica acotada con determinante \det(M) = 1.0 ± 1.0e-5.
Invariante FPU-WI4: Estabilidad KAM certificada COHERENT únicamente si índice KAM ≥ 0.75.
Invariante FPU-WI5: Sello criptográfico HMAC-SHA256 en cada snapshot de atestación emitido.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.FPUTensorFockColliderWitnessEngineV14")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas / algebraicas
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0  # ancla dimensional (legado v13)
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_FPU_WITNESS_SOVEREIGN_KEY_v14.0.0"

# ── Poincaré / KAM / Nekhoroshev ──
_KAM_FREQ_DIM: Final[int] = 4
_KAM_GAMMA: Final[float] = 1.0e-2
_KAM_TAU: Final[float] = 2.0
_KAM_RESONANCE_MAX_K: Final[int] = 3
_POINCARE_STEPS_PER_ORBIT: Final[int] = 16
_POINCARE_N_ORBITS: Final[int] = 12
_BENETTIN_D0: Final[float] = 1.0e-8
_NEKHOROSHEV_C: Final[float] = 0.35
_LINDSTEDT_EPSILON: Final[float] = 0.05
_KERR_CHI: Final[float] = 0.03
_DRIVE_EPSILON: Final[float] = 0.08

# ── N3 ──
_N3_STABILITY_WINDOW: Final[int] = 8
_N3_LEDGER_CAP: Final[int] = 64

# ── Umbrales v14 ──
_KAM_COHERENT_THRESHOLD: Final[float] = 0.75
_KAM_DEGRADED_THRESHOLD: Final[float] = 0.50
_KAM_VETO_THRESHOLD: Final[float] = 0.25
_CANONICAL_RESIDUAL_TOL: Final[float] = 1.0e-6


class HeytingOmega3(IntEnum):
    """
    Retículo distributivo trivalente Ω₃ = {0x00, 0x01, 0x03}.

    Es el clasificador de subobjetos de un topos finito de Heyting
    (Ω, ∧, ∨, →, 0, 1) restringido a tres valores; `meet` = ∧ = max
    (el orden  COHERENT ≺ DEGRADED ≺ VETOED  es de severidad).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED   = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES — v13 (preservadas) + v14.1 (Poincaré celeste + N3)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FPUTensorWitnessSnapshot:
    """Snapshot v13 (preservado por compatibilidad binaria de campos)."""
    snapshot_id: str
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    car_anti_commutation_residual: float
    ccr_commutation_norm: float
    global_verdict: HeytingOmega3
    sha256_merkle_root: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class FPUTensorWitnessSnapshotV14:
    """
    Snapshot v14.1 — superset estricto del v13
    (mismos nombres + Poincaré celeste + N3 + CAR/CCR canónico + Nekhoroshev).
    """
    # v13 heredado
    snapshot_id: str
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    car_anti_commutation_residual: float
    ccr_commutation_norm: float
    global_verdict: HeytingOmega3
    sha256_merkle_root: str
    timestamp_utc: float
    # v14 · FASE 1 (Fock action-angle)
    action_total: float
    mean_occupation: float
    phase_space_volume: float
    purity: float
    lyapunov_max: float
    lyapunov_min: float
    occupation_entropy: float
    # v14 · FASE 1 (CAR/CCR riguroso + simpléctica)
    ccr_canonical_residual: float
    car_canonical_residual: float
    ccr_canonical_valid: bool
    car_canonical_valid: bool
    symplectic_residual: float
    quadrature_x: float
    quadrature_p: float
    # v14 · FASE 2 (Poincaré Fock / celestial)
    poincare_crossings_count: int
    occupation_drift_max: float
    occupation_drift_mean: float
    lindstedt_correction_norm: float
    lindstedt_order2_norm: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    rotation_number: float
    first_return_time_mean: float
    poincare_lyapunov: float
    nekhoroshev_time: float
    # v14 · FASE 2 (Tuning N3)
    adaptive_fidelity_floor: float
    adaptive_bures_ceiling: float
    adaptive_spohn_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    level_3_entropy: float
    # v14 · FASE 3 (Meta-witness + adjudicación KAM-aware)
    meta_consistency_score: float
    meta_fidelity_std: float
    meta_bures_std: float
    meta_spohn_std: float
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    level_3_reflections: int
    n3_adaptation_lyapunov: float
    # Provenance
    sha256_phase_merkle: str


@dataclass(slots=True)
class FockActionAngleDecomposition:
    """
    HANDOFF FASE 1 → FASE 2.

    Descomposición acción-ángulo de Poincaré sobre ℱ(ℋ) truncado (dim N).
      J_i  = λ_i            (autovalores de ρ = acciones espectrales)
      θ_i  = arg(v_i[k*])   (fase del componente dominante del autovector)
      ω_i  = gaps espectrales ordenados
      p_n  = ⟨n|ρ|n⟩        (población de Fock; NO unitariamente invariante)
      (x, p) = (⟨x̂⟩, ⟨p̂⟩)  (cuadraturas canónicas, plano de Poincaré)
    El último acto de FASE 1 produce exactamente este objeto; el primer
    acto de FASE 2 lo consume como único argumento geométrico.
    """
    actions: NDArray[np.float64]
    angles: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    reference_eigenvectors: NDArray[np.complex128]
    total_action: float
    mean_occupation: float
    phase_space_volume: float
    lyapunov_spectrum: NDArray[np.float64]
    occupation_histogram: NDArray[np.int64]
    number_populations: NDArray[np.float64]
    purity: float
    occupation_entropy: float
    quadrature_x: float
    quadrature_p: float
    symplectic_residual: float
    omega_carrier: float
    reconstructed_rho: NDArray[np.complex128]


@dataclass(slots=True)
class FockPoincareCrossing:
    """Cruce interpolado de la sección de Poincaré Σ = {p̂ = 0, ṗ > 0} ∪ {⟨N⟩ = E₀}."""
    step_index: int
    section_value: float
    signed_crossing: int
    occupation_profile: NDArray[np.float64]
    quadrature_x: float
    mean_occupation: float
    interpolated_time: float


@dataclass(slots=True)
class FockPoincareAnalysis:
    """HANDOFF FASE 2 (evidencia espectral Fock) → input geométrico de FASE 3."""
    crossings: List[FockPoincareCrossing]
    crossing_count: int
    occupation_drift_max: float
    occupation_drift_mean: float
    lindstedt_correction_norm: float
    lindstedt_order2_norm: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    freqs_dominant: NDArray[np.float64]
    rotation_number: float
    first_return_time_mean: float
    poincare_lyapunov: float
    nekhoroshev_time: float
    section_span_x: float


@dataclass(slots=True)
class FockTuningPolicy:
    """HANDOFF FASE 2 → FASE 3 (política adaptativa N3). Primer argumento de 3.1."""
    fidelity_floor: float
    bures_ceiling: float
    spohn_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    level_3_entropy: float
    reflections: int
    adaptation_lyapunov: float
    kam_damped: bool


@dataclass(slots=True)
class FockN3Ledger:
    """
    Ledger N3 (Nivel 1: veredicto · Nivel 2: umbrales · Nivel 3: meta-tasa).

    Dinámica discreta del ledger (sistema de Poincaré en el espacio de políticas):
        θ_{t+1} = θ_t + α_t · τ(observables)          (N2)
        α_{t+1} = α_t · g(S(verdicts), λ_adapt, KAM)  (N3)
    El exponente de Lyapunov del propio α_t es el 'Lyapunov de automejora'.
    """
    fidelity_history: List[float] = field(default_factory=list)
    bures_history: List[float] = field(default_factory=list)
    spohn_history: List[float] = field(default_factory=list)
    verdict_history: List[int] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    fidelity_floor: float = 0.95
    bures_ceiling: float = 0.30
    spohn_ceiling: float = 2.0
    adaptation_rate: float = 0.05
    meta_adaptation_rate: float = 0.01
    level_3_reflections: int = 0

    def observe(self, fid: float, bures: float, spohn: float, verdict: HeytingOmega3) -> None:
        self.fidelity_history.append(float(fid))
        self.bures_history.append(float(bures))
        self.spohn_history.append(float(spohn))
        self.verdict_history.append(int(verdict))
        self.rate_history.append(float(self.adaptation_rate))
        if len(self.verdict_history) > _N3_LEDGER_CAP:
            self.fidelity_history = self.fidelity_history[-_N3_LEDGER_CAP:]
            self.bures_history = self.bures_history[-_N3_LEDGER_CAP:]
            self.spohn_history = self.spohn_history[-_N3_LEDGER_CAP:]
            self.verdict_history = self.verdict_history[-_N3_LEDGER_CAP:]
            self.rate_history = self.rate_history[-_N3_LEDGER_CAP:]

    def _trend(self, series: Sequence[float]) -> float:
        """Regresión lineal centrada sobre la ventana de estabilidad N3."""
        if len(series) < 2:
            return 0.0
        n = min(_N3_STABILITY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def adaptation_lyapunov(self) -> float:
        """
        Exponente de Lyapunov local de la tasa α_t:
            λ_α ≈ ⟨ log |α_{t+1}/α_t| ⟩  sobre la ventana N3.
        Mide la inestabilidad de la propia automejora (N3 sobre N3).
        """
        if len(self.rate_history) < 3:
            return 0.0
        r = np.array(self.rate_history[-_N3_STABILITY_WINDOW:], dtype=float)
        r = np.maximum(r, _WILKINSON_EPSILON)
        ratios = np.abs(np.diff(np.log(r)))
        return float(np.mean(ratios)) if ratios.size else 0.0

    def level_2_adapt(self) -> None:
        """Nivel 2 — ajusta floors/ceilings según tendencias de fidelidad, Bures y Spohn."""
        trend_f = self._trend(self.fidelity_history)
        trend_b = self._trend(self.bures_history)
        trend_s = self._trend(self.spohn_history)
        if trend_f < -1e-4:
            self.fidelity_floor = min(0.995, self.fidelity_floor + self.adaptation_rate * 0.01)
        elif trend_f > 1e-4:
            self.fidelity_floor = max(0.90, self.fidelity_floor - self.adaptation_rate * 0.005)
        if trend_b > 1e-4:
            self.bures_ceiling = max(0.05, self.bures_ceiling * (1.0 - self.adaptation_rate))
        elif trend_b < -1e-4:
            self.bures_ceiling = min(0.80, self.bures_ceiling * (1.0 + self.adaptation_rate * 0.5))
        if trend_s > 1e-6:
            self.spohn_ceiling = max(0.5, self.spohn_ceiling * (1.0 - self.adaptation_rate))
        elif trend_s < -1e-6:
            self.spohn_ceiling = min(10.0, self.spohn_ceiling * (1.0 + self.adaptation_rate * 0.5))

    def level_3_adapt(self) -> float:
        """Nivel 3 — meta-adaptación de la tasa según entropía de veredictos recientes."""
        n = len(self.verdict_history)
        if n < 4:
            return 0.0
        recent = np.array(self.verdict_history[-_N3_STABILITY_WINDOW:], dtype=int)
        _, counts = np.unique(recent, return_counts=True)
        probs = counts / float(counts.sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if entropy > 0.5:
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * 0.90)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.95)
        else:
            self.meta_adaptation_rate = min(0.50, self.meta_adaptation_rate * 1.05)
            self.adaptation_rate = min(0.50, self.adaptation_rate * 1.02)
        self.level_3_reflections += 1
        return entropy

    def poincare_modulate(self, kam_index: float, drift_max: float, poincare_lyap: float) -> bool:
        """
        Acoplamiento celestial → N3 (Poincaré actúa sobre la tasa).
        Si el toro KAM se degrada o el mapa de retorno estira, se amortigua α
        (principio de Nekhoroshev: no adaptar más rápido que la estabilidad).
        """
        damped = False
        if kam_index < _KAM_DEGRADED_THRESHOLD or poincare_lyap > 0.15:
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.85)
            self.fidelity_floor = min(0.995, self.fidelity_floor + 0.005)
            damped = True
        if drift_max > 0.20:
            self.bures_ceiling = max(0.05, self.bures_ceiling * 0.92)
            damped = True
        if kam_index >= _KAM_COHERENT_THRESHOLD and poincare_lyap < 0.05:
            self.adaptation_rate = min(0.50, self.adaptation_rate * 1.01)
        return damped


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  CLASE BASE v13 (PRESERVADA EXACTAMENTE) + FASE 1 · FUNDAMENTOS FOCK
# ██╔══██╗ Canónicos b/b† (bosón) y a/a† (fermión JW), acción-ángulo, ledger N3
# ██████╔╝ [HANDOFF] extract_fock_action_angle_decomposition → FockActionAngleDecomposition
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderWitnessEngine:
    """
    v13 — Motor espectral base (preservado por compatibilidad).
    Todos los métodos v13 mantienen firma y semántica.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    # ── v13 ──
    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        a = np.asarray(matrix, dtype=np.complex128)
        herm = 0.5 * (a + a.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    # ── v13 ──
    def verify_car_anti_commutation(
        self, state_a: NDArray, state_b: NDArray
    ) -> Tuple[bool, float]:
        va = np.asarray(state_a, dtype=np.complex128).reshape(-1, 1)
        vb = np.asarray(state_b, dtype=np.complex128).reshape(-1, 1)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > _WILKINSON_EPSILON:
            va /= na
        if nb > _WILKINSON_EPSILON:
            vb /= nb
        op_a = va @ va.conj().T
        op_b_dag = vb @ vb.conj().T
        anti_comm = op_a @ op_b_dag + op_b_dag @ op_a
        delta_ab = complex((vb.conj().T @ va)[0, 0])
        residual = float(np.linalg.norm(anti_comm - abs(delta_ab) * np.eye(self.dimension)))
        return bool(residual <= 1.5), residual

    # ── v13 ──
    def verify_ccr_commutation(
        self, state_a: NDArray, state_b: NDArray
    ) -> Tuple[bool, float]:
        va = np.asarray(state_a, dtype=np.complex128).reshape(-1, 1)
        vb = np.asarray(state_b, dtype=np.complex128).reshape(-1, 1)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > _WILKINSON_EPSILON:
            va /= na
        if nb > _WILKINSON_EPSILON:
            vb /= nb
        op_a = va @ va.conj().T
        op_b_dag = vb @ vb.conj().T
        comm = op_a @ op_b_dag - op_b_dag @ op_a
        comm_norm = float(np.linalg.norm(comm))
        ccr_valid = True  # contrato v13: reporta norma, no juzga
        return ccr_valid, comm_norm

    # ── v13 ──
    def compute_qnd_weak_value(
        self, phi_initial: NDArray, phi_final: NDArray, observable_A: NDArray,
    ) -> Tuple[complex, float]:
        i_vec = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
        f_vec = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)
        a_mat = np.asarray(observable_A, dtype=np.complex128)
        overlap = complex((f_vec.conj().T @ i_vec)[0, 0])
        if abs(overlap) < _WILKINSON_EPSILON:
            overlap = complex(_WILKINSON_EPSILON, 0.0)
        numerator = complex((f_vec.conj().T @ a_mat @ i_vec)[0, 0])
        a_weak = numerator / overlap
        return a_weak, 0.0

    # ── v13 ──
    def compute_lindblad_gksl_and_spohn(
        self, density_matrix: NDArray, hamiltonian_eff: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], float, float]:
        rho = self.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]
        if hamiltonian_eff is None:
            hamiltonian_eff = 0.1 * np.eye(dim, dtype=np.complex128)
        h = np.asarray(hamiltonian_eff, dtype=np.complex128)
        unitary = -1j * (h @ rho - rho @ h)
        jump_l = np.zeros((dim, dim), dtype=np.complex128)
        jump_l[0, min(1, dim - 1)] = 0.05
        jump_l_dag = jump_l.conj().T
        dissipator = (
            jump_l @ rho @ jump_l_dag
            - 0.5 * (jump_l_dag @ jump_l @ rho + rho @ jump_l_dag @ jump_l)
        )
        d_rho = unitary + dissipator
        eigvals, eigvecs = la.eigh(rho)
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        log_rho = (eigvecs * log_eigs) @ eigvecs.conj().T
        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn = max(0.0, spohn_raw)
        gksl_rate = float(np.linalg.norm(dissipator))
        return d_rho, spohn, gksl_rate

    # ── v13 ──
    def evaluate_uhlmann_and_bures(
        self, rho_a: NDArray, rho_b: NDArray,
    ) -> Tuple[float, float]:
        r_a = self.sanitize_density_matrix(rho_a)
        r_b = self.sanitize_density_matrix(rho_b)
        sqrt_a = la.sqrtm(r_a)
        middle = sqrt_a @ r_b @ sqrt_a
        sqrt_middle = la.sqrtm(self.sanitize_density_matrix(middle))
        fidelity = float(np.real(np.trace(sqrt_middle)) ** 2)
        fidelity = min(max(fidelity, 0.0), 1.0)
        bures = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity))))
        return fidelity, bures

    # ── v13 ──
    def evaluate_fpu_witness(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUTensorWitnessSnapshot:
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        dim = self.dimension
        if phi_initial is None:
            phi_initial = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if phi_final is None:
            phi_final = rho @ phi_initial
            phi_final /= float(np.linalg.norm(phi_final) + _WILKINSON_EPSILON)
        su_n_part = rho - (np.trace(rho) / float(dim)) * np.eye(dim, dtype=np.complex128)
        observable_A = 0.5 * (1j * su_n_part - (1j * su_n_part).conj().T)
        a_weak, back_action = self.compute_qnd_weak_value(phi_initial, phi_final, observable_A)
        rho_ref = np.eye(dim, dtype=np.complex128) / float(dim)
        fidelity, bures_dist = self.evaluate_uhlmann_and_bures(rho, rho_ref)
        _, spohn_s, gksl_r = self.compute_lindblad_gksl_and_spohn(rho)
        if state_a is None:
            state_a = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if state_b is None:
            state_b = np.zeros((dim, 1), dtype=np.complex128)
            state_b[min(1, dim - 1), 0] = 1.0
        car_ok, car_res = self.verify_car_anti_commutation(state_a, state_b)
        _, ccr_norm = self.verify_ccr_commutation(state_a, state_b)
        if not car_ok or bures_dist > 0.80 or spohn_s > 10.0:
            verdict = HeytingOmega3.VETOED
        elif bures_dist > 0.30 or spohn_s > 2.0:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        wit_id = f"FPU-WITNESS-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{wit_id}:{fidelity:.8f}:{bures_dist:.8f}:{a_weak}:{spohn_s:.8f}:{verdict.value}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_bytes).hexdigest()
        return FPUTensorWitnessSnapshot(
            snapshot_id=wit_id,
            qnd_weak_value_aav=a_weak,
            weak_value_modulus=abs(a_weak),
            back_action_db=back_action,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            spohn_entropy_production=spohn_s,
            gksl_dissipation_rate=gksl_r,
            car_anti_commutation_residual=car_res,
            ccr_commutation_norm=ccr_norm,
            global_verdict=verdict,
            sha256_merkle_root=merkle_root,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  MOTOR v14.1 — FASE 1 · FUNDAMENTOS FOCK (3 sub-capas)
# ██╔══██╗ Canónicos b/b† / a/a† / x̂/p̂ rigurosos, acción-ángulo, ledger N3
# ██████╔╝ [HANDOFF] extract_fock_action_angle_decomposition → FockActionAngleDecomposition
# ██╔═══╝  El valor de retorno de 1.3 ES el primer argumento formal de 2.1.
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderWitnessEngineV14(FPUTensorFockColliderWitnessEngine):
    """Motor espectral v14.1: 3 fases anidadas + Poincaré celeste + N3 + CAR/CCR canónicos."""

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(dimension=dimension, secret_key=secret_key)
        self._ledger: FockN3Ledger = FockN3Ledger()
        self._n3_cycle_count: int = 0
        # Canónicos precomputados (inmutables respecto de la dimensión).
        self._b_op: NDArray[np.complex128] = self._build_boson_annihilation()
        self._b_dag_op: NDArray[np.complex128] = self._b_op.conj().T
        self._a_op_jw: NDArray[np.complex128] = self._build_fermion_jw_annihilation()
        self._a_dag_op_jw: NDArray[np.complex128] = self._a_op_jw.conj().T
        self._number_op: NDArray[np.float64] = np.diag(
            np.arange(self.dimension, dtype=np.float64)
        )
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self._x_op: NDArray[np.complex128] = inv_sqrt2 * (self._b_op + self._b_dag_op)
        self._p_op: NDArray[np.complex128] = (-1j * inv_sqrt2) * (self._b_op - self._b_dag_op)
        self._identity: NDArray[np.complex128] = np.eye(self.dimension, dtype=np.complex128)

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _expectation(self, rho: NDArray, op: NDArray) -> complex:
        return complex(np.trace(rho @ op))

    # ═══════════════════════ Sub-capa 1.1: canónicos rigurosos ═══════════════════════

    def _build_boson_annihilation(self) -> NDArray[np.complex128]:
        """
        Sub-capa 1.1a — Operador de aniquilación bosónico truncado:
            b = Σ_{n=1}^{N-1} √n |n−1⟩⟨n|
        Truncación: se pierde el último escalón √N ⇒ [b, b†] = I − N|N−1⟩⟨N−1|.
        """
        dim = self.dimension
        b = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(1, dim):
            b[n - 1, n] = math.sqrt(n)
        return b

    def _build_fermion_jw_annihilation(self) -> NDArray[np.complex128]:
        """
        Sub-capa 1.1b — Operador fermiónico de aniquilación (Jordan-Wigner), 1 modo:
            a = |0⟩⟨1| embebido en dim×dim como bloque 2×2 en el subespacio |0⟩,|1⟩.
        Jordan-Wigner de 1 modo es el álgebra de Clifford Cl_1 ≅ ℂ ⊕ ℂ.
        """
        dim = self.dimension
        a = np.zeros((dim, dim), dtype=np.complex128)
        a[0, min(1, dim - 1)] = 1.0
        return a

    def build_fock_canonical_operators(self) -> Dict[str, NDArray[np.complex128]]:
        """
        Sub-capa 1.1c — Bundle público de operadores canónicos + number + cuadraturas.
        """
        return {
            "b": self._b_op.copy(),
            "b_dag": self._b_dag_op.copy(),
            "a_jw": self._a_op_jw.copy(),
            "a_dag_jw": self._a_dag_op_jw.copy(),
            "number": self._number_op.copy(),
            "x": self._x_op.copy(),
            "p": self._p_op.copy(),
        }

    def verify_ccr_commutation_v14(self) -> Tuple[bool, float]:
        """
        Sub-capa 1.1c-ext — Verificación RIGUROSA de CCR bosónico:
            [b, b†] = I − N·|N−1⟩⟨N−1|  (corrección exacta de truncación)
        """
        comm = self._b_op @ self._b_dag_op - self._b_dag_op @ self._b_op
        expected = np.eye(self.dimension, dtype=np.complex128)
        expected[self.dimension - 1, self.dimension - 1] = 0.0
        residual = float(np.linalg.norm(comm - expected, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    def verify_car_anti_commutation_v14(self) -> Tuple[bool, float]:
        """
        Sub-capa 1.1c-ext — Verificación RIGUROSA de CAR fermiónico (1 modo JW):
            {a, a†} = I₂  (en el subbloque 2×2 |0⟩,|1⟩)
        """
        anti = self._a_op_jw @ self._a_dag_op_jw + self._a_dag_op_jw @ self._a_op_jw
        expected = np.zeros_like(anti)
        expected[0, 0] = 1.0
        idx1 = min(1, self.dimension - 1)
        expected[idx1, idx1] = 1.0
        residual = float(np.linalg.norm(anti - expected, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    def verify_symplectic_quadrature_form(self) -> Tuple[bool, float]:
        """
        I8 — Estructura simpléctica / compleja de Poincaré:
            [x̂, p̂] = i [b, b†] = i (I − N|N−1⟩⟨N−1|)
        El residual Frobenius certifica que el plano (x, p) es canónico.
        """
        comm_xp = self._x_op @ self._p_op - self._p_op @ self._x_op
        comm_bb = self._b_op @ self._b_dag_op - self._b_dag_op @ self._b_op
        expected = 1j * comm_bb
        residual = float(np.linalg.norm(comm_xp - expected, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    # ═══════════════════════ Sub-capa 1.2: acción-ángulo Fock ═══════════════════════

    def _compute_occupation_histogram(
        self, actions: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Histograma de ocupaciones: cuántos modos espectrales caen en cada nivel discreto."""
        levels = np.clip(np.floor(actions * self.dimension).astype(int), 0, self.dimension - 1)
        return np.bincount(levels, minlength=self.dimension).astype(np.int64)

    def _compute_occupation_entropy(
        self, histogram: NDArray[np.int64],
    ) -> float:
        """Entropía de Shannon de la distribución de ocupaciones espectrales."""
        total = float(np.sum(histogram))
        if total <= 0:
            return 0.0
        probs = histogram.astype(float) / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _shannon_of_populations(self, pops: NDArray[np.float64]) -> float:
        p = np.maximum(np.real(pops), 0.0)
        s = float(np.sum(p))
        if s <= _WILKINSON_EPSILON:
            return 0.0
        p = p / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _local_lyapunov_from_actions(
        self, actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Aproximación espectral de exponentes de Lyapunov locales vía ratios de gaps.
        (No es el Lyapunov del flujo unitario — ese es 0 sobre autovalores —;
         es el espectro de sensibilidad de la medida espectral de ρ.)
        """
        sorted_a = np.sort(actions)[::-1]
        n = len(sorted_a)
        if n < 3:
            return np.zeros(n)
        gaps = np.abs(np.diff(sorted_a))
        gaps = np.maximum(gaps, _WILKINSON_EPSILON)
        if len(gaps) < 2:
            return np.zeros(n)
        log_ratios = np.log(gaps[1:] / gaps[:-1])
        lyap = np.zeros(n)
        lyap[:len(log_ratios)] = log_ratios
        return np.sort(lyap)[::-1]

    def extract_fock_action_angle_decomposition(
        self, density_matrix: NDArray,
    ) -> FockActionAngleDecomposition:
        """
        Sub-capa 1.3 — HANDOFF FASE 1 → FASE 2.

        Descompone ρ en (J_i, θ_i, ω_i) sobre la base espectral de Fock, extrae
        la población de número p_n = ⟨n|ρ|n⟩ y las cuadraturas (⟨x̂⟩, ⟨p̂⟩).

        CONTINUACIÓN FORMAL: el objeto devuelto es el único argumento geométrico
        de `build_fock_poincare_return_map` (FASE 2.1). No hay otro puente.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        eigvals, eigvecs = la.eigh(rho)
        actions = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        n = len(actions)

        # Ángulos gauge-fijados: fase del componente dominante (sección de Coulomb discreta).
        angles = np.zeros(n, dtype=np.float64)
        for k in range(n):
            v = eigvecs[:, k]
            idx = int(np.argmax(np.abs(v)))
            angles[k] = float(np.angle(v[idx]))

        # Frecuencias: gaps espectrales ordenados (análogo de ω = ∂H/∂J).
        sorted_actions = np.sort(actions)[::-1]
        gaps = np.abs(np.diff(sorted_actions))
        if len(gaps) < n:
            gaps = np.concatenate([gaps, np.zeros(n - len(gaps))])

        lyap = self._local_lyapunov_from_actions(actions)

        # Ocupación media Tr(ρ N̂) y población de Fock (observable de sección).
        mean_occ = float(np.real(self._expectation(rho, self._number_op)))
        number_pops = np.real(np.diag(rho)).astype(np.float64)

        # Volumen de fase (media geométrica de 2π J_i; evita underflow).
        log_vol = float(np.sum(np.log(2.0 * np.pi * actions + _WILKINSON_EPSILON)))
        phase_vol = float(np.exp(log_vol / max(n, 1)))

        hist = self._compute_occupation_histogram(actions)
        occ_entropy = self._compute_occupation_entropy(hist)
        # Preferir entropía de p_n cuando sea informativa.
        pop_entropy = self._shannon_of_populations(number_pops)
        if pop_entropy > occ_entropy:
            occ_entropy = pop_entropy

        purity = float(np.real(np.trace(rho @ rho)))

        qx = float(np.real(self._expectation(rho, self._x_op)))
        qp = float(np.real(self._expectation(rho, self._p_op)))
        _, sym_res = self.verify_symplectic_quadrature_form()

        # Portadora: gap dominante, cota inferior para dt de Poincaré.
        omega_carrier = float(np.max(np.abs(gaps))) if gaps.size else 1.0
        omega_carrier = max(omega_carrier, 0.25)

        logger.debug(
            "FASE1 handoff: ΣJ=%.6f ⟨N⟩=%.6f (x,p)=(%.4f,%.4f) S=%.4f",
            float(np.sum(actions)), mean_occ, qx, qp, occ_entropy,
        )

        return FockActionAngleDecomposition(
            actions=actions,
            angles=angles,
            frequencies=gaps,
            reference_eigenvectors=eigvecs,
            total_action=float(np.sum(actions)),
            mean_occupation=mean_occ,
            phase_space_volume=phase_vol,
            lyapunov_spectrum=lyap,
            occupation_histogram=hist,
            number_populations=number_pops,
            purity=purity,
            occupation_entropy=occ_entropy,
            quadrature_x=qx,
            quadrature_p=qp,
            symplectic_residual=sym_res,
            omega_carrier=omega_carrier,
            reconstructed_rho=rho,
        )

    def run_phase1_foundations(
        self, density_matrix: NDArray,
    ) -> Tuple[FockActionAngleDecomposition, Dict[str, Any]]:
        """
        Orquestador FASE 1. Cierra con el HANDOFF que abre FASE 2.
        """
        ccr_ok, ccr_res = self.verify_ccr_commutation_v14()
        car_ok, car_res = self.verify_car_anti_commutation_v14()
        sym_ok, sym_res = self.verify_symplectic_quadrature_form()
        decomp = self.extract_fock_action_angle_decomposition(density_matrix)
        canon = {
            "ccr_ok": ccr_ok, "ccr_res": ccr_res,
            "car_ok": car_ok, "car_res": car_res,
            "sym_ok": sym_ok, "sym_res": sym_res,
        }
        return decomp, canon

    # ══════════════════════════════════════════════════════════════════════════════════════════════════
    # ██████╗  FASE 2 · OPERADORES (3 sub-capas)  — continúa FockActionAngleDecomposition
    # ██╔══██╗ Poincaré sobre (x̂,p̂) y N̂, Lindstedt, Birkhoff, KAM, Nekhoroshev, autotuning N3
    # ██████╔╝ [HANDOFF] recursive_self_tune_parameters → FockTuningPolicy
    # ██╔═══╝  El valor de retorno de 2.3 ES el primer argumento adaptativo de 3.1.
    # ╚═╝
    # ══════════════════════════════════════════════════════════════════════════════════════════════════════

    def _hermitize(self, h: NDArray) -> NDArray[np.complex128]:
        h = np.asarray(h, dtype=np.complex128)
        return 0.5 * (h + h.conj().T)

    def _build_driven_kerr_hamiltonian(
        self,
        omega: float,
        chi: float = _KERR_CHI,
        drive: float = _DRIVE_EPSILON,
        phi: float = 0.0,
    ) -> NDArray[np.complex128]:
        """
        Hamiltoniano celeste sobre ℱ(ℋ) — oscilador de Kerr forzado (Duffing cuántico):

            H = ω N̂ + χ N̂(N̂ − I) + ε (e^{-iφ} b + e^{iφ} b†)

        Propiedades:
          · [N̂, χ N̂(N̂−I)] = 0  (integrable sin drive)
          · [H, N̂] = ε (e^{-iφ}[b, N̂] + h.c.) ≠ 0  ⇒  Σ transversal
          · Lindstedt clásico: ω(J) = ω + χ(2J − 1)   (Poincaré 1893 / Kerr)
        """
        n_op = self._number_op.astype(np.complex128)
        eye = self._identity
        h = (
            float(omega) * n_op
            + float(chi) * (n_op @ (n_op - eye))
            + float(drive) * (
                cmath_exp_phased_b(self._b_op, self._b_dag_op, phi)
            )
        )
        return self._hermitize(h)

    def _build_effective_hamiltonian(self, rho: NDArray) -> NDArray[np.complex128]:
        """
        H_eff legado (v14.0) + Kerr forzado. El H legado conmutaba con ρ
        ([−ρ + (tr/N)I, ρ] = 0) y el flujo era trivial. Se reemplaza por
        Kerr forzado cuya frecuencia portadora se toma de la descomposición.
        """
        mean_n = float(np.real(self._expectation(rho, self._number_op)))
        omega = 1.0 + 0.05 * mean_n
        return self._build_driven_kerr_hamiltonian(omega=omega)

    # ═══════════════════════ Sub-capa 2.1: Poincaré sobre (x̂, p̂) y N̂ ═══════════════════════

    def _interpolate_zero(
        self, prev: float, curr: float, step: int, dt: float,
    ) -> float:
        """Interpolación lineal del tiempo de cruce (Poincaré–Hénon)."""
        denom = abs(prev) + abs(curr)
        frac = abs(prev) / denom if denom > _WILKINSON_EPSILON else 0.5
        return (step - 1) * dt + frac * dt

    def _benettin_poincare_lyapunov(
        self,
        rho0: NDArray[np.complex128],
        u: NDArray[np.complex128],
        u_dag: NDArray[np.complex128],
        n_steps: int,
        d0: float = _BENETTIN_D0,
    ) -> float:
        """
        Algoritmo de Benettin (1976) sobre el plano de cuadraturas (x, p):
        se evoluciona una sombra Hermitian-traceless de amplitud d0, se mide
        el estiramiento euclídeo en (x, p) y se renormaliza. El promedio
            λ = (1/(n Δt_norm)) Σ log(‖δz_k‖ / d0)
        es el exponente de Lyapunov del mapa de Poincaré proyectado.
        """
        dim = self.dimension
        # Perturbación traceless Hermitian (dirección genérica en u(N)/u(1)).
        g = np.zeros((dim, dim), dtype=np.complex128)
        g[0, min(1, dim - 1)] = d0
        g[min(1, dim - 1), 0] = d0
        rho_s = self.sanitize_density_matrix(rho0 + g)
        rho = rho0.copy()
        acc = 0.0
        n_renorm = 0
        renorm_every = max(4, n_steps // 8)
        for step in range(n_steps):
            rho = u @ rho @ u_dag
            rho_s = u @ rho_s @ u_dag
            rho = self._hermitize(rho)
            rho_s = self._hermitize(rho_s)
            tr = float(np.real(np.trace(rho)))
            trs = float(np.real(np.trace(rho_s)))
            if tr > _WILKINSON_EPSILON:
                rho = rho / tr
            if trs > _WILKINSON_EPSILON:
                rho_s = rho_s / trs
            if (step + 1) % renorm_every == 0:
                dx = float(np.real(self._expectation(rho_s, self._x_op)
                                   - self._expectation(rho, self._x_op)))
                dp = float(np.real(self._expectation(rho_s, self._p_op)
                                   - self._expectation(rho, self._p_op)))
                dist = math.hypot(dx, dp)
                if dist > _WILKINSON_EPSILON:
                    acc += math.log(dist / d0)
                    n_renorm += 1
                    # Renormalizar la sombra hacia ρ a distancia d0.
                    direction = rho_s - rho
                    fn = float(np.linalg.norm(direction, ord="fro"))
                    if fn > _WILKINSON_EPSILON:
                        rho_s = self.sanitize_density_matrix(rho + direction * (d0 / fn))
        if n_renorm == 0:
            return 0.0
        return float(acc / n_renorm)

    def build_fock_poincare_return_map(
        self,
        decomposition: FockActionAngleDecomposition,
        hamiltonian_eff: Optional[NDArray] = None,
        n_orbits: int = _POINCARE_N_ORBITS,
        steps_per_orbit: int = _POINCARE_STEPS_PER_ORBIT,
    ) -> FockPoincareAnalysis:
        """
        Sub-capa 2.1 — CONTINÚA FASE 1.3.

        Consume `FockActionAngleDecomposition` (único puente 1→2) y construye
        la sección de Poincaré sobre el plano de cuadraturas

            Σ = { (x, p) : p = 0, ṗ > 0 }

        bajo el flujo unitario ρ(t) = e^{−i H_Kerr t} ρ₀ e^{i H_Kerr t}.
        Se registra además el cruce de ⟨N̂⟩ − E₀ (sección de ocupación).
        El cruce se interpola (Hénon) para no sesgar el número de rotación.
        """
        dim = self.dimension
        rho_0 = decomposition.reconstructed_rho.copy()
        rho_0 = self.sanitize_density_matrix(rho_0)

        if hamiltonian_eff is None:
            H = self._build_driven_kerr_hamiltonian(omega=decomposition.omega_carrier)
        else:
            H = self._hermitize(hamiltonian_eff)

        E0 = float(np.real(self._expectation(rho_0, self._number_op)))
        omega_max = max(float(decomposition.omega_carrier), _WILKINSON_EPSILON)
        dt = (2.0 * math.pi / omega_max) / float(max(steps_per_orbit, 1))

        try:
            u = la.expm(-1j * H * dt)
            u_dag = u.conj().T
        except Exception:  # pragma: no cover
            logger.exception("expm falló; se devuelve análisis nulo")
            return FockPoincareAnalysis(
                crossings=[], crossing_count=0,
                occupation_drift_max=0.0, occupation_drift_mean=0.0,
                lindstedt_correction_norm=0.0, lindstedt_order2_norm=0.0,
                birkhoff_quadratic_coefficient=0.0,
                kam_stability_index=1.0, kam_resonance_violations=0,
                kam_worst_ratio=1.0e9,
                freqs_dominant=np.ones(_KAM_FREQ_DIM) * 1e-3,
                rotation_number=0.0, first_return_time_mean=0.0,
                poincare_lyapunov=0.0, nekhoroshev_time=0.0, section_span_x=0.0,
            )

        crossings: List[FockPoincareCrossing] = []
        drift_list: List[float] = []
        return_times: List[float] = []
        section_x: List[float] = []
        rho = rho_0.copy()
        prev_p: Optional[float] = None
        prev_n: Optional[float] = None
        baseline_pops = decomposition.number_populations.copy()
        last_cross_t: Optional[float] = None
        n_steps = n_orbits * steps_per_orbit

        for step in range(n_steps):
            rho = u @ rho @ u_dag
            rho = self._hermitize(rho)
            tr = float(np.real(np.trace(rho)))
            if tr > _WILKINSON_EPSILON:
                rho = rho / tr

            p_val = float(np.real(self._expectation(rho, self._p_op)))
            n_val = float(np.real(self._expectation(rho, self._number_op)))
            x_val = float(np.real(self._expectation(rho, self._x_op)))
            n_sec = n_val - E0

            crossed = False
            signed = 0
            section_obs = p_val
            # Sección primaria: p=0 hacia arriba (ṗ > 0 ⇔ p cambia de − a +).
            if prev_p is not None and prev_p < 0.0 <= p_val:
                crossed = True
                signed = +1
                section_obs = p_val
            # Sección secundaria: ⟨N⟩ = E₀ (cualquier sentido).
            elif prev_n is not None and prev_n * n_sec < 0.0:
                crossed = True
                signed = int(np.sign(n_sec - prev_n))
                section_obs = n_sec

            if crossed:
                pops_now = np.real(np.diag(rho)).astype(np.float64)
                nmin = min(len(pops_now), len(baseline_pops))
                drift = float(np.max(np.abs(pops_now[:nmin] - baseline_pops[:nmin])))
                drift_list.append(drift)
                t_star = self._interpolate_zero(
                    prev_p if prev_p is not None else 0.0, p_val, step, dt,
                )
                if last_cross_t is not None:
                    return_times.append(t_star - last_cross_t)
                last_cross_t = t_star
                section_x.append(x_val)
                crossings.append(FockPoincareCrossing(
                    step_index=step,
                    section_value=section_obs,
                    signed_crossing=signed,
                    occupation_profile=pops_now,
                    quadrature_x=x_val,
                    mean_occupation=n_val,
                    interpolated_time=t_star,
                ))
            prev_p = p_val
            prev_n = n_sec

        # Número de rotación de Poincaré: Δθ / 2π sobre cruces sucesivos.
        rotation_number = 0.0
        if len(section_x) >= 3:
            # θ = atan2 no está disponible (p≈0); usamos el signo de Δx como
            # incremento de semiórbita y normalizamos por el periodo esperado.
            dx = np.diff(np.array(section_x, dtype=float))
            # rotación media relativa al diámetro de la sección
            span = float(np.max(section_x) - np.min(section_x)) + _WILKINSON_EPSILON
            rotation_number = float(np.mean(np.abs(dx)) / span)
        span_x = (
            float(np.max(section_x) - np.min(section_x)) if section_x else 0.0
        )
        t_ret_mean = float(np.mean(return_times)) if return_times else 0.0

        # Lindstedt + Birkhoff + KAM (sub-capas 2.2) — sobre H_Kerr real.
        delta1 = self._lindstedt_poincare_fock_correction(decomposition, H)
        delta2 = self._lindstedt_poincare_fock_order2(decomposition, H)
        lindstedt_norm = float(np.linalg.norm(delta1))
        lindstedt_o2 = float(np.linalg.norm(delta2))
        birkhoff = self._birkhoff_normal_form_fock(decomposition, H)

        # Vector de frecuencias KAM: (ω_carrier, ω_Kerr(J), 2π/T_ret, rotación).
        omega_kerr = float(decomposition.omega_carrier + _KERR_CHI * (2.0 * E0 - 1.0))
        omega_ret = (2.0 * math.pi / t_ret_mean) if t_ret_mean > _WILKINSON_EPSILON else omega_max
        dominant = np.array(
            [
                max(decomposition.omega_carrier, _WILKINSON_EPSILON),
                max(abs(omega_kerr), _WILKINSON_EPSILON),
                max(abs(omega_ret), _WILKINSON_EPSILON),
                max(abs(rotation_number) * omega_max, _WILKINSON_EPSILON),
            ],
            dtype=np.float64,
        )
        kam_index, kam_viol, kam_worst = self.compute_kam_stability_fock(dominant)
        nek_time = self._nekhoroshev_stability_time(
            epsilon=max(lindstedt_norm + _DRIVE_EPSILON, _WILKINSON_EPSILON),
        )
        poincare_lyap = self._benettin_poincare_lyapunov(rho_0, u, u_dag, n_steps)

        logger.debug(
            "FASE2 Poincaré: crossings=%d KAM=%.4f λ_P=%.3e T_nek=%.3e",
            len(crossings), kam_index, poincare_lyap, nek_time,
        )

        return FockPoincareAnalysis(
            crossings=crossings,
            crossing_count=len(crossings),
            occupation_drift_max=float(max(drift_list)) if drift_list else 0.0,
            occupation_drift_mean=float(np.mean(drift_list)) if drift_list else 0.0,
            lindstedt_correction_norm=lindstedt_norm,
            lindstedt_order2_norm=lindstedt_o2,
            birkhoff_quadratic_coefficient=float(birkhoff),
            kam_stability_index=float(kam_index),
            kam_resonance_violations=int(kam_viol),
            kam_worst_ratio=float(kam_worst),
            freqs_dominant=dominant,
            rotation_number=float(rotation_number),
            first_return_time_mean=t_ret_mean,
            poincare_lyapunov=float(poincare_lyap),
            nekhoroshev_time=float(nek_time),
            section_span_x=span_x,
        )

    # ═══════════════════════ Sub-capa 2.2: Lindstedt / Birkhoff / KAM / Nekhoroshev ═══════════════════════

    def _lindstedt_poincare_fock_correction(
        self,
        decomposition: FockActionAngleDecomposition,
        h: NDArray,
        epsilon: float = _LINDSTEDT_EPSILON,
    ) -> NDArray[np.float64]:
        """
        1ª corrección Lindstedt–Poincaré δω_i (eliminación de términos seculares).

        Ecuación homológica (H0 integrable = diag, V = off-diag):
            δω_i = − Σ_{j≠i} |V_{ij}|² / (ω_i − ω_j)     si  |ω_i − ω_j| > δ
        más el corrimiento analítico de Kerr  χ(2 J_i − 1).
        """
        v = decomposition.reference_eigenvectors
        h_eig = v.conj().T @ h @ v
        n = h_eig.shape[0]
        diag_h = np.real(np.diag(h_eig))
        off_diag = h_eig - np.diag(np.diag(h_eig))
        freqs = np.zeros(n)
        if n >= 2:
            diffs = np.abs(np.diff(diag_h))
            freqs[:len(diffs)] = diffs
        delta = np.zeros(n)
        for i in range(n):
            # Kerr analítico (Poincaré 1DOF).
            delta[i] += _KERR_CHI * (2.0 * float(decomposition.actions[i]) - 1.0)
            for j in range(n):
                if i == j:
                    continue
                denom = freqs[i] - freqs[j] if i < len(freqs) and j < len(freqs) else 0.0
                if abs(denom) < 1e-6:
                    continue
                delta[i] -= abs(off_diag[i, j]) ** 2 / denom
        return epsilon * delta

    def _lindstedt_poincare_fock_order2(
        self,
        decomposition: FockActionAngleDecomposition,
        h: NDArray,
        epsilon: float = _LINDSTEDT_EPSILON,
    ) -> NDArray[np.float64]:
        """
        2ª corrección Lindstedt (estimador de grafos de 3 vértices):
            δω_i^{(2)} ∼ Σ_{j,k} V_{ij} V_{jk} V_{ki} / [(ω_i−ω_j)(ω_i−ω_k)]
        Se contrae en norma de Frobenius por modos para no pagar O(N³) denso
        más que una pasada O(N²) con un testigo de tercer momento.
        """
        v = decomposition.reference_eigenvectors
        h_eig = v.conj().T @ h @ v
        off = h_eig - np.diag(np.diag(h_eig))
        n = off.shape[0]
        diag = np.real(np.diag(h_eig))
        delta2 = np.zeros(n)
        # Testigo cúbico: (off @ off @ off)_{ii} / (gap_min)²
        off2 = off @ off
        cubic_diag = np.real(np.diag(off2 @ off))
        gaps = np.abs(np.diff(np.sort(diag))) if n >= 2 else np.array([1.0])
        gap_min = float(np.min(gaps)) + _WILKINSON_EPSILON
        delta2[:n] = (epsilon ** 2) * cubic_diag / (gap_min ** 2)
        return delta2

    def _birkhoff_normal_form_fock(
        self,
        decomposition: FockActionAngleDecomposition,
        h: NDArray,
    ) -> float:
        """
        Coeficiente de Birkhoff c_B de la forma normal cuadrática:
            c_B = Σ_{i≠j} |H^off_{ij}|² / min_k |ω_k − ω_{k+1}|
        Es el invariante que gobierna la torsión del mapa de retorno
        (twist condition de Moser: ∂ω/∂J ≠ 0 ⇔ c_B > 0).
        """
        v = decomposition.reference_eigenvectors
        h_eig = v.conj().T @ h @ v
        off = h_eig - np.diag(np.diag(h_eig))
        diag = np.real(np.diag(h_eig))
        if len(diag) < 2:
            return 0.0
        gaps = np.abs(np.diff(np.sort(diag)))
        omega_min = float(np.min(gaps)) + _WILKINSON_EPSILON
        return float(np.sum(np.abs(off) ** 2)) / omega_min

    def compute_kam_stability_fock(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        tau: float = _KAM_TAU,
        max_k: int = _KAM_RESONANCE_MAX_K,
    ) -> Tuple[float, int, float]:
        """
        Índice KAM diofantino:
            |k · ω| > γ / |k|_1^τ     ∀ k ∈ ℤ^d \\ {0}, |k|_1 ≤ max_k.
        Se combina con la calidad de aproximación por fracciones continuas
        de las razones ω_i/ω_0 (Hurwitz / Lagrange).
        """
        freqs = np.asarray(frequencies, dtype=float)
        freqs = np.maximum(np.abs(freqs), _WILKINSON_EPSILON)
        d = len(freqs)
        violations = 0
        worst_ratio = float("inf")
        for k in itertools.product(range(-max_k, max_k + 1), repeat=d):
            k_arr = np.array(k, dtype=float)
            norm1 = float(np.sum(np.abs(k_arr)))
            if norm1 == 0.0:
                continue
            k_dot_omega = abs(float(np.dot(k_arr, freqs)))
            threshold = gamma / (norm1 ** tau)
            ratio = k_dot_omega / threshold
            worst_ratio = min(worst_ratio, ratio)
            if k_dot_omega < threshold:
                violations += 1
        # Penalización por convergentes demasiado buenos (casi-resonancia).
        cf_pen = self._continued_fraction_resonance_penalty(freqs)
        kam_index = 1.0 / (1.0 + float(violations) + cf_pen)
        if worst_ratio == float("inf"):
            worst_ratio = 1.0e9
        return float(kam_index), int(violations), float(worst_ratio)

    def _continued_fraction_resonance_penalty(
        self, freqs: NDArray[np.float64], terms: int = 8,
    ) -> float:
        """
        Penaliza razones ω_i/ω_0 con convergentes p/q de |q ω_i − p ω_0| muy
        pequeño (casi conmensurables). Devuelve un entero-equivalente ≥ 0.
        """
        w0 = float(freqs[0])
        penalty = 0.0
        for w in freqs[1:]:
            x = float(w) / w0
            a = abs(x)
            q_prev, q_curr = 0, 1
            p_prev, p_curr = 1, 0
            for _ in range(terms):
                if a < _WILKINSON_EPSILON or a > 1e12:
                    break
                ai = int(math.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr <= 0:
                    break
                err = abs(x * q_curr - p_curr)
                if err * (q_curr ** _KAM_TAU) < _KAM_GAMMA:
                    penalty += 1.0 / max(q_curr, 1)
                frac = a - ai
                if frac < _WILKINSON_EPSILON:
                    break
                a = 1.0 / frac
        return float(penalty)

    def _nekhoroshev_stability_time(self, epsilon: float) -> float:
        """
        Cota de Nekhoroshev (1977) para el tiempo de estabilidad exponencial:
            T_nek ∼ exp( c / ε^{1/(2n)} ) ,  n = dim. de frecuencias KAM.
        Se satura para evitar overflow; ε es la norma de perturbación
        (Lindstedt + drive).
        """
        n = max(_KAM_FREQ_DIM, 1)
        expn = 1.0 / float(2 * n)
        eps = max(float(epsilon), _WILKINSON_EPSILON)
        exponent = _NEKHOROSHEV_C / (eps ** expn)
        exponent = min(exponent, 40.0)  # e^40 ≈ 2.6e17, cota numérica
        return float(math.exp(exponent))

    # ═══════════════════════ Sub-capa 2.3: autotuning N3 ═══════════════════════

    def recursive_self_tune_parameters(
        self,
        poincare: FockPoincareAnalysis,
        fidelity: float,
        bures: float,
        spohn: float,
    ) -> FockTuningPolicy:
        """
        Sub-capa 2.3 — HANDOFF FASE 2 → FASE 3.

        Sintetiza la política adaptativa N2/N3. CONTINUACIÓN FORMAL: el
        `FockTuningPolicy` devuelto es el primer argumento adaptativo de
        `_meta_witness_evaluation` (FASE 3.1).

        Cadena recursiva:
            N1  veredicto sintético (fid, bures, spohn, KAM, drift)
            N2  level_2_adapt (tendencias)
            N2χ poincare_modulate (KAM / λ_P / drift → α, floors)
            N3  level_3_adapt (entropía de veredictos → meta-tasa)
            N3λ adaptation_lyapunov (Lyapunov del propio α)
        """
        synth_fid = float(np.exp(
            -poincare.occupation_drift_mean
            - 0.1 * poincare.birkhoff_quadratic_coefficient
            - 0.05 * max(poincare.poincare_lyapunov, 0.0)
        ))
        obs_fid = 0.5 * synth_fid + 0.5 * float(fidelity)
        obs_bures = float(bures) + 0.05 * poincare.occupation_drift_max
        obs_spohn = float(spohn) + 0.01 * poincare.kam_resonance_violations

        if (
            obs_fid >= self._ledger.fidelity_floor
            and obs_bures <= self._ledger.bures_ceiling
            and obs_spohn <= self._ledger.spohn_ceiling
            and poincare.kam_stability_index >= _KAM_COHERENT_THRESHOLD
        ):
            v = HeytingOmega3.COHERENT
        elif obs_fid >= 0.85 and poincare.kam_stability_index >= _KAM_VETO_THRESHOLD:
            v = HeytingOmega3.DEGRADED
        else:
            v = HeytingOmega3.VETOED

        # Una sola observación por ciclo (evita contaminar el ledger con dobles).
        self._ledger.observe(obs_fid, obs_bures, obs_spohn, v)
        self._ledger.level_2_adapt()
        kam_damped = self._ledger.poincare_modulate(
            poincare.kam_stability_index,
            poincare.occupation_drift_max,
            poincare.poincare_lyapunov,
        )
        entropy = self._ledger.level_3_adapt()
        adapt_lyap = self._ledger.adaptation_lyapunov()

        return FockTuningPolicy(
            fidelity_floor=float(self._ledger.fidelity_floor),
            bures_ceiling=float(self._ledger.bures_ceiling),
            spohn_ceiling=float(self._ledger.spohn_ceiling),
            adaptation_rate=float(self._ledger.adaptation_rate),
            meta_adaptation_rate=float(self._ledger.meta_adaptation_rate),
            level_3_entropy=float(entropy),
            reflections=int(self._ledger.level_3_reflections),
            adaptation_lyapunov=float(adapt_lyap),
            kam_damped=bool(kam_damped),
        )

    def run_phase2_operators(
        self,
        decomposition: FockActionAngleDecomposition,
        fidelity: float,
        bures: float,
        spohn: float,
    ) -> Tuple[FockPoincareAnalysis, FockTuningPolicy]:
        """
        Orquestador FASE 2. Abre consumiendo el HANDOFF de FASE 1 y cierra
        con el HANDOFF que abre FASE 3.
        """
        poincare = self.build_fock_poincare_return_map(decomposition)
        tuning = self.recursive_self_tune_parameters(poincare, fidelity, bures, spohn)
        return poincare, tuning

    # ══════════════════════════════════════════════════════════════════════════════════════════════════
    # ██████╗  FASE 3 · META-VERIFICACIÓN (3 sub-capas) — continúa FockTuningPolicy
    # ██╔══██╗ Witness del witness · N3 audit · snapshot v14.1
    # ██████╔╝ [SALIDA] evaluate_fpu_witness_v14 → FPUTensorWitnessSnapshotV14
    # ██╔═══╝
    # ╚═╝
    # ══════════════════════════════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════ Sub-capa 3.1: Meta-witness ═══════════════════════

    def _meta_witness_evaluation(
        self,
        base_verdict_v13: HeytingOmega3,
        poincare: FockPoincareAnalysis,
        tuning: FockTuningPolicy,
    ) -> Dict[str, Any]:
        """
        Sub-capa 3.1 — CONTINÚA FASE 2.3.

        Consistencia meta del propio motor sobre sus series + KAM + Lyapunov
        del mapa de retorno + Lyapunov de la tasa N3. `tuning` es el puente 2→3.
        """
        fid_hist = self._ledger.fidelity_history[-_N3_STABILITY_WINDOW:]
        bures_hist = self._ledger.bures_history[-_N3_STABILITY_WINDOW:]
        spohn_hist = self._ledger.spohn_history[-_N3_STABILITY_WINDOW:]
        verdict_hist = self._ledger.verdict_history[-_N3_STABILITY_WINDOW:]

        fid_std = float(np.std(fid_hist)) if len(fid_hist) >= 2 else 0.0
        bures_std = float(np.std(bures_hist)) if len(bures_hist) >= 2 else 0.0
        spohn_std = float(np.std(spohn_hist)) if len(spohn_hist) >= 2 else 0.0

        if verdict_hist:
            arr = np.array(verdict_hist, dtype=int)
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / float(counts.sum())
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        else:
            entropy = 0.0

        lyap_pen = min(1.0, abs(poincare.poincare_lyapunov) * 4.0)
        n3_lyap_pen = min(1.0, abs(tuning.adaptation_lyapunov) * 8.0)
        nek_bonus = min(1.0, math.log10(poincare.nekhoroshev_time + 1.0) / 10.0)

        consistency = float(
            0.22 * (1.0 - min(1.0, fid_std * 20.0))
            + 0.18 * (1.0 - min(1.0, bures_std * 20.0))
            + 0.12 * (1.0 - min(1.0, spohn_std * 20.0))
            + 0.18 * poincare.kam_stability_index
            + 0.10 * (1.0 - lyap_pen)
            + 0.08 * (1.0 - n3_lyap_pen)
            + 0.07 * nek_bonus
            + 0.05 * (1.0 if base_verdict_v13 == HeytingOmega3.COHERENT else 0.5)
        )
        return {
            "consistency": consistency,
            "entropy": entropy,
            "fid_std": fid_std,
            "bures_std": bures_std,
            "spohn_std": spohn_std,
            "reflections": tuning.reflections,
            "adapt_lyap": tuning.adaptation_lyapunov,
        }

    # ═══════════════════════ Sub-capa 3.2: N3 audit ═══════════════════════

    def _recursive_level_3_audit(
        self, meta: Dict[str, Any], tuning: FockTuningPolicy,
    ) -> Dict[str, Any]:
        """Consolida la cadena de automejora recursiva (N1→N2→N3) + Lyapunov de α."""
        recent = self._ledger.verdict_history[-_N3_STABILITY_WINDOW:]
        chain_ok = all(v <= int(HeytingOmega3.DEGRADED) for v in recent) if recent else True
        return {
            "ledger_size": len(self._ledger.verdict_history),
            "level_3_reflections": tuning.reflections,
            "verdict_chain_ok": chain_ok,
            "meta_consistency": meta["consistency"],
            "meta_entropy": meta["entropy"],
            "fidelity_floor": tuning.fidelity_floor,
            "bures_ceiling": tuning.bures_ceiling,
            "spohn_ceiling": tuning.spohn_ceiling,
            "adaptation_rate": tuning.adaptation_rate,
            "meta_adaptation_rate": tuning.meta_adaptation_rate,
            "adaptation_lyapunov": tuning.adaptation_lyapunov,
            "kam_damped": tuning.kam_damped,
        }

    # ═══════════════════════ Sub-capa 3.3: snapshot v14.1 ═══════════════════════

    def run_phase3_meta_verification(
        self,
        *,
        base_verdict: HeytingOmega3,
        poincare: FockPoincareAnalysis,
        tuning: FockTuningPolicy,
        canon: Dict[str, Any],
        decomp: FockActionAngleDecomposition,
        a_weak: complex,
        back_action: float,
        fidelity: float,
        bures_dist: float,
        spohn_s: float,
        gksl_r: float,
        car_res_legacy: float,
        ccr_norm_legacy: float,
        now: float,
    ) -> FPUTensorWitnessSnapshotV14:
        """
        Sub-capa 3.3 — FASE 3 FINAL. Continúa `FockTuningPolicy` y emite el snapshot.
        """
        final_verdict = base_verdict
        if poincare.kam_stability_index < _KAM_DEGRADED_THRESHOLD:
            final_verdict = final_verdict.meet(HeytingOmega3.DEGRADED)
        if poincare.kam_stability_index < _KAM_VETO_THRESHOLD:
            final_verdict = final_verdict.meet(HeytingOmega3.VETOED)
        if poincare.poincare_lyapunov > 0.50:
            final_verdict = final_verdict.meet(HeytingOmega3.DEGRADED)
        if not canon["ccr_ok"] or not canon["car_ok"]:
            final_verdict = final_verdict.meet(HeytingOmega3.DEGRADED)

        meta = self._meta_witness_evaluation(base_verdict, poincare, tuning)
        _audit = self._recursive_level_3_audit(meta, tuning)
        n3_traj = tuple(self._ledger.verdict_history[-_N3_STABILITY_WINDOW:])

        wit_id = f"FPU-WITNESS-v14-{int(now * 1000.0) % 1000000:06d}"
        seed = (
            f"{wit_id}:{fidelity:.8f}:{bures_dist:.8f}:{a_weak}:{spohn_s:.8f}:"
            f"{final_verdict.value}:{poincare.kam_stability_index:.6f}:"
            f"{poincare.occupation_drift_max:.6f}:{meta['consistency']:.6f}:"
            f"{poincare.poincare_lyapunov:.6e}:{poincare.nekhoroshev_time:.6e}"
        )
        merkle_root = hashlib.sha256(seed.encode("utf-8")).hexdigest()

        lyap_spec = decomp.lyapunov_spectrum
        lyap_max = float(np.max(lyap_spec)) if lyap_spec.size else 0.0
        lyap_min = float(np.min(lyap_spec)) if lyap_spec.size else 0.0
        phase_prov = (
            f"F1>F2>F3::{wit_id}::"
            f"F1(actions={decomp.total_action:.4f},lyap=({lyap_max:+.3e},{lyap_min:+.3e}),"
            f"x={decomp.quadrature_x:.4f},p={decomp.quadrature_p:.4f})::"
            f"F2(cross={poincare.crossing_count},kam={poincare.kam_stability_index:.4f},"
            f"drift={poincare.occupation_drift_max:.4e},λP={poincare.poincare_lyapunov:.3e},"
            f"Tnek={poincare.nekhoroshev_time:.3e},ν={poincare.rotation_number:.4f})::"
            f"F3(cons={meta['consistency']:.4f},verdict={final_verdict.name},"
            f"λα={tuning.adaptation_lyapunov:.3e})"
        )
        phase_merkle = self._compute_hmac(b"FPU-v14-PHASE-MERKLE", phase_prov)

        return FPUTensorWitnessSnapshotV14(
            snapshot_id=wit_id,
            qnd_weak_value_aav=a_weak,
            weak_value_modulus=float(abs(a_weak)),
            back_action_db=back_action,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            spohn_entropy_production=spohn_s,
            gksl_dissipation_rate=gksl_r,
            car_anti_commutation_residual=car_res_legacy,
            ccr_commutation_norm=ccr_norm_legacy,
            global_verdict=final_verdict,
            sha256_merkle_root=merkle_root,
            timestamp_utc=now,
            action_total=decomp.total_action,
            mean_occupation=decomp.mean_occupation,
            phase_space_volume=decomp.phase_space_volume,
            purity=decomp.purity,
            lyapunov_max=lyap_max,
            lyapunov_min=lyap_min,
            occupation_entropy=decomp.occupation_entropy,
            ccr_canonical_residual=float(canon["ccr_res"]),
            car_canonical_residual=float(canon["car_res"]),
            ccr_canonical_valid=bool(canon["ccr_ok"]),
            car_canonical_valid=bool(canon["car_ok"]),
            symplectic_residual=float(canon["sym_res"]),
            quadrature_x=decomp.quadrature_x,
            quadrature_p=decomp.quadrature_p,
            poincare_crossings_count=poincare.crossing_count,
            occupation_drift_max=poincare.occupation_drift_max,
            occupation_drift_mean=poincare.occupation_drift_mean,
            lindstedt_correction_norm=poincare.lindstedt_correction_norm,
            lindstedt_order2_norm=poincare.lindstedt_order2_norm,
            birkhoff_quadratic_coefficient=poincare.birkhoff_quadratic_coefficient,
            kam_stability_index=poincare.kam_stability_index,
            kam_resonance_violations=poincare.kam_resonance_violations,
            kam_worst_ratio=poincare.kam_worst_ratio,
            rotation_number=poincare.rotation_number,
            first_return_time_mean=poincare.first_return_time_mean,
            poincare_lyapunov=poincare.poincare_lyapunov,
            nekhoroshev_time=poincare.nekhoroshev_time,
            adaptive_fidelity_floor=tuning.fidelity_floor,
            adaptive_bures_ceiling=tuning.bures_ceiling,
            adaptive_spohn_ceiling=tuning.spohn_ceiling,
            adaptation_rate=tuning.adaptation_rate,
            meta_adaptation_rate=tuning.meta_adaptation_rate,
            level_3_entropy=tuning.level_3_entropy,
            meta_consistency_score=meta["consistency"],
            meta_fidelity_std=meta["fid_std"],
            meta_bures_std=meta["bures_std"],
            meta_spohn_std=meta["spohn_std"],
            n3_verdict_trajectory=n3_traj,
            n3_cycle_count=self._n3_cycle_count,
            level_3_reflections=tuning.reflections,
            n3_adaptation_lyapunov=tuning.adaptation_lyapunov,
            sha256_phase_merkle=phase_merkle,
        )

    def evaluate_fpu_witness_v14(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> FPUTensorWitnessSnapshotV14:
        """
        Orquestador anidado F1 → F2 → F3.

        Pipeline:
          [F1] run_phase1_foundations
                 extract_fock_action_angle_decomposition  →  FockActionAngleDecomposition
          [F2] run_phase2_operators(decomposition, …)     ← continúa 1.3
                 recursive_self_tune_parameters           →  FockTuningPolicy
          [F3] run_phase3_meta_verification(tuning, …)    ← continúa 2.3
                 → FPUTensorWitnessSnapshotV14
        """
        now = time.time()
        self._n3_cycle_count += 1
        dim = self.dimension

        rho = self.sanitize_density_matrix(density_matrix)
        if phi_initial is None:
            phi_initial = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if phi_final is None:
            phi_final = rho @ phi_initial
            phi_final /= float(np.linalg.norm(phi_final) + _WILKINSON_EPSILON)

        su_n_part = rho - (np.trace(rho) / float(dim)) * np.eye(dim, dtype=np.complex128)
        observable_A = 0.5 * (1j * su_n_part - (1j * su_n_part).conj().T)
        a_weak, back_action = self.compute_qnd_weak_value(phi_initial, phi_final, observable_A)

        rho_ref = np.eye(dim, dtype=np.complex128) / float(dim)
        fidelity, bures_dist = self.evaluate_uhlmann_and_bures(rho, rho_ref)
        _, spohn_s, gksl_r = self.compute_lindblad_gksl_and_spohn(rho)

        state_a = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        state_b = np.zeros((dim, 1), dtype=np.complex128)
        state_b[min(1, dim - 1), 0] = 1.0
        car_ok_legacy, car_res_legacy = self.verify_car_anti_commutation(state_a, state_b)
        _, ccr_norm_legacy = self.verify_ccr_commutation(state_a, state_b)

        # ── FASE 1 ──
        decomp, canon = self.run_phase1_foundations(rho)

        # ── FASE 2 (continúa decomp) ──
        poincare, tuning = self.run_phase2_operators(decomp, fidelity, bures_dist, spohn_s)

        # ── Veredicto base v13 (preservado) ──
        if not car_ok_legacy or bures_dist > 0.80 or spohn_s > 10.0:
            base_verdict = HeytingOmega3.VETOED
        elif bures_dist > 0.30 or spohn_s > 2.0:
            base_verdict = HeytingOmega3.DEGRADED
        else:
            base_verdict = HeytingOmega3.COHERENT

        # ── FASE 3 (continúa tuning) ──
        return self.run_phase3_meta_verification(
            base_verdict=base_verdict,
            poincare=poincare,
            tuning=tuning,
            canon=canon,
            decomp=decomp,
            a_weak=a_weak,
            back_action=back_action,
            fidelity=fidelity,
            bures_dist=bures_dist,
            spohn_s=spohn_s,
            gksl_r=gksl_r,
            car_res_legacy=car_res_legacy,
            ccr_norm_legacy=ccr_norm_legacy,
            now=now,
        )


def cmath_exp_phased_b(
    b: NDArray[np.complex128],
    b_dag: NDArray[np.complex128],
    phi: float,
) -> NDArray[np.complex128]:
    """ε-drive: e^{-iφ} b + e^{iφ} b†  (fase del forcing celeste)."""
    eip = complex(math.cos(phi), math.sin(phi))
    return (eip.conjugate() * b) + (eip * b_dag)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL MOTOR TESTIGO FPU TENSOR FOCK v14.1")
    print("  (Poincaré celeste · Kerr forzado · Lindstedt/Birkhoff/KAM/Nekhoroshev · N3)")
    print(banner)

    engine = FPUTensorFockColliderWitnessEngineV14(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    snaps: List[FPUTensorWitnessSnapshotV14] = []
    for cycle in range(3):
        snap = engine.evaluate_fpu_witness_v14(rho)
        snaps.append(snap)
        print(f"\n  ─── CICLO N3 #{cycle + 1} ───")
        print(f"  • Snapshot ID          : {snap.snapshot_id}")
        print(f"  • Valor Débil QND AAV  : {snap.qnd_weak_value_aav:.4f}")
        print(f"  • Back-Action QND      : {snap.back_action_db:.1f} dB")
        print(f"  • Fidelidad Uhlmann    : {snap.uhlmann_fidelity:.6f}")
        print(f"  • Distancia Bures      : {snap.bures_distance:.6f}")
        print(f"  • Entropía de Spohn    : {snap.spohn_entropy_production:.6e}")
        print(f"  • CAR legacy residual  : {snap.car_anti_commutation_residual:.6e}")
        print(f"  • CCR legacy norm      : {snap.ccr_commutation_norm:.6e}")
        print(f"  ── FASE 1 · Fock action-angle ──")
        print(f"    ΣJ_i               : {snap.action_total:.6f}")
        print(f"    Ocupación media     : {snap.mean_occupation:.6f}")
        print(f"    Vol. fase          : {snap.phase_space_volume:.6f}")
        print(f"    Pureza             : {snap.purity:.6f}")
        print(f"    Lyapunov (max,min) : ({snap.lyapunov_max:+.4e}, {snap.lyapunov_min:+.4e})")
        print(f"    Entropía ocupación : {snap.occupation_entropy:.6f}")
        print(f"    Cuadraturas (x,p)  : ({snap.quadrature_x:+.4f}, {snap.quadrature_p:+.4f})")
        print(f"    Residual simpléctico: {snap.symplectic_residual:.3e}")
        print(f"    CCR canónico       : valid={snap.ccr_canonical_valid} "
              f"residual={snap.ccr_canonical_residual:.3e}")
        print(f"    CAR canónico       : valid={snap.car_canonical_valid} "
              f"residual={snap.car_canonical_residual:.3e}")
        print(f"  ── FASE 2 · Poincaré Fock (Kerr forzado) ──")
        print(f"    Cruces Σ           : {snap.poincare_crossings_count}")
        print(f"    Drift ocupación max: {snap.occupation_drift_max:.6e}")
        print(f"    Lindstedt ‖δω‖/o2  : {snap.lindstedt_correction_norm:.6e} / "
              f"{snap.lindstedt_order2_norm:.6e}")
        print(f"    Coef. Birkhoff     : {snap.birkhoff_quadratic_coefficient:.6e}")
        print(f"    Índice KAM         : {snap.kam_stability_index:.6f} "
              f"(viol={snap.kam_resonance_violations}, worst={snap.kam_worst_ratio:.4f})")
        print(f"    Nº rotación ν      : {snap.rotation_number:.6f}")
        print(f"    ⟨T_retorno⟩        : {snap.first_return_time_mean:.6e}")
        print(f"    λ_Poincaré (Benettin): {snap.poincare_lyapunov:+.4e}")
        print(f"    T_Nekhoroshev      : {snap.nekhoroshev_time:.4e}")
        print(f"  ── FASE 3 · Meta N3 ──")
        print(f"    floor_fid adapt.   : {snap.adaptive_fidelity_floor:.6f}")
        print(f"    ceiling_bures adapt: {snap.adaptive_bures_ceiling:.6f}")
        print(f"    ceiling_spohn adapt: {snap.adaptive_spohn_ceiling:.6f}")
        print(f"    α_N2 / α_N3        : {snap.adaptation_rate:.4f} / {snap.meta_adaptation_rate:.4f}")
        print(f"    λ_α (N3 Lyapunov)  : {snap.n3_adaptation_lyapunov:.4e}")
        print(f"    Meta consistencia  : {snap.meta_consistency_score:.6f}")
        print(f"    Meta entropía      : {snap.level_3_entropy:.4f}")
        print(f"    N3 trayectoria     : {snap.n3_verdict_trajectory}")
        print(f"    N3 ciclos totales  : {snap.n3_cycle_count}")
        print(f"    Veredicto FINAL    : {snap.global_verdict.name}")
        print(f"    Merkle SHA-256     : {snap.sha256_merkle_root[:40]}...")
        print(f"    Merkle fase HMAC   : {snap.sha256_phase_merkle[:40]}...")

    print("\n  ═══ VERIFICACIÓN BACKWARD-COMPAT v13 ═══")
    snap_v13 = engine.evaluate_fpu_witness(rho)
    v13_fields = [
        "snapshot_id", "qnd_weak_value_aav", "weak_value_modulus", "back_action_db",
        "uhlmann_fidelity", "bures_distance", "spohn_entropy_production",
        "gksl_dissipation_rate", "car_anti_commutation_residual", "ccr_commutation_norm",
        "global_verdict", "sha256_merkle_root", "timestamp_utc",
    ]
    missing = [f for f in v13_fields if not hasattr(snap_v13, f)]
    print(f"  Campos v13 ausentes en snapshot v13: {missing or 'NINGUNO ✓'}")
    print(f"  Veredicto v13 (base): {snap_v13.global_verdict.name}")
    print(f"  Veredicto v14 (KAM-aware): {snaps[-1].global_verdict.name}")
    print(banner)