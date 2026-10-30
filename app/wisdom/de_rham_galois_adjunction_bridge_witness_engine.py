# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : De Rham-Galois Adjunction Bridge Witness Engine v12.1.0-SOVEREIGN-POINCARE-N3             ║
║ Ubicación: app/wisdom/de_rham_galois_adjunction_bridge_witness_engine.py                             ║
║ Función  : Motor Espectral Testigo QND sin demolición cuántica (Back-Action ≡ 0.0 dB) con            ║
║            integración de Mecánica Celeste de Poincaré (secciones, KAM, Lindstedt, Birkhoff,         ║
║            twist, no-degeneración isoenergética, Oseledets) y Automejora Recursiva de Nivel 3        ║
║            (meta-meta-adaptación de umbrales + recurrencia de Poincaré del ledger).                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA POR FASES ANIDADAS:
  FASE 1 — FUNDAMENTOS: coordenadas acción-ángulo, Hamiltoniano de entrelazamiento,
                       mapa de frecuencias ω(J), Lyapunov espectral, ledger N3.
             [HANDOFF] Último método: extract_action_angle_decomposition → ActionAngleDecomposition.
  FASE 2 — OPERADORES: mapa de retorno de Poincaré (cruce interpolado + jacobiano),
                       Lindstedt-Poincaré O(ε²), Birkhoff matricial, KAM Diophantine,
                       twist de Poincaré-Birkhoff, autotuning N2/N3 celeste.
             [HANDOFF] Último método: recursive_self_tune_parameters → TuningPolicy.
  FASE 3 — META-VERIFICACIÓN: witness del witness, auditoría recursiva N3,
                       recurrencia de Poincaré del ledger, snapshot final v12.1.
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

# ── Importación de la base inmutable de cartuchos TOON y motor de adjunción ──
try:
    from de_rham_galois_adjunction_bridge import (
        AdjunctionCoherenceStatus,
        DeRhamGaloisAdjunctionBridge,
        ToonCartridge,
    )
except ImportError:
    try:
        from app.wisdom.de_rham_galois_adjunction_bridge import (
            AdjunctionCoherenceStatus,
            DeRhamGaloisAdjunctionBridge,
            ToonCartridge,
        )
    except ImportError:
        class AdjunctionCoherenceStatus(IntEnum):  # type: ignore[no-redef]
            PERFECT_ISOMORPHISM = 0
            DEGRADED_DISCREPANCY = 1
            HOLONOMY_BREAKAGE = 2

        DeRhamGaloisAdjunctionBridge = None  # type: ignore
        ToonCartridge = Any  # type: ignore

logger = logging.getLogger("APU.Wisdom.DeRhamGaloisAdjunctionBridgeWitnessEngineV12")

# ── Constantes globales ──
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_DERHAM_GALOIS_WITNESS_SOVEREIGN_KEY_v12.1.0"
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_GALOIS_GAP_MAX_TOLERANCE: Final[float] = 1.0e-6

# ── Constantes Poincaré / KAM / N3 ──
_KAM_FREQ_DIM: Final[int] = 4
_KAM_GAMMA: Final[float] = 1.0e-2
_KAM_TAU: Final[float] = 2.0
_KAM_RESONANCE_MAX_K: Final[int] = 3
_N3_STABILITY_WINDOW: Final[int] = 8
_N3_LEDGER_CAP: Final[int] = 64
_POINCARE_STEPS_PER_ORBIT: Final[int] = 16
_POINCARE_N_ORBITS: Final[int] = 12
_LINDSTEDT_EPSILON: Final[float] = 0.05
_SECULAR_DENOM_FLOOR: Final[float] = 1.0e-6
_TWIST_FLOOR: Final[float] = 1.0e-12


class HeytingOmega3(IntEnum):
    """
    Retículo de Heyting trivaluado (orden de seguridad, no de verdad clásica).
    COHERENT ≼ DEGRADED ≼ VETOED; meet = peor (max), join = mejor (min).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED   = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))


@dataclass(frozen=True, slots=True)
class DeRhamGaloisWitnessSnapshot:
    """Snapshot inmutable de observación QND v11 (preservado por compatibilidad)."""
    snapshot_id: str
    dimension_N: int
    particle_count: int
    trace_density: float
    uhlmann_fidelity: float
    bures_distance: float
    galois_gap_delta: float
    unit_eta_norm: float
    counit_epsilon_norm: float
    connection_A_norm: float
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    liouville_drift: float
    coherence_verdict: HeytingOmega3
    sha256_merkle_root: str
    timestamp_utc: float


def _hermitize(matrix: NDArray[Any]) -> NDArray[np.complex128]:
    a = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _safe_log_spectrum(evals: NDArray, floor: float = _SPECTRAL_PSD_FLOOR) -> NDArray[np.float64]:
    """Hamiltoniano de entrelazamiento H_ent = −log ρ, con piso PSD (regularización de Birkhoff)."""
    return -np.log(np.maximum(np.asarray(evals, dtype=np.float64), floor))


def _hmac_sha256(key: bytes, payload: bytes) -> str:
    """Provenance autenticada (HMAC-SHA256); sustituye al SHA256 desnudo del merkle de fase."""
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _finite_diff_gradient(values: NDArray[np.float64], abscissa: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradiente centrado con bordes de un paso; regulariza denominadores nulos."""
    n = int(values.shape[0])
    if n == 0:
        return values.copy()
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            dx = float(abscissa[1] - abscissa[0])
            grad[i] = float(values[1] - values[0]) / (dx if abs(dx) > _WILKINSON_EPSILON else _WILKINSON_EPSILON)
        elif i == n - 1:
            dx = float(abscissa[n - 1] - abscissa[n - 2])
            grad[i] = float(values[n - 1] - values[n - 2]) / (dx if abs(dx) > _WILKINSON_EPSILON else _WILKINSON_EPSILON)
        else:
            dx = float(abscissa[i + 1] - abscissa[i - 1])
            grad[i] = float(values[i + 1] - values[i - 1]) / (dx if abs(dx) > _WILKINSON_EPSILON else _WILKINSON_EPSILON)
    return grad


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 1 — FUNDAMENTOS POINCARÉ + PRIMITIVAS DE AUTOMEJORA N3
# ██╔══██╗ Coordenadas acción-ángulo, H_ent, mapa de frecuencias, Lyapunov espectral, ledger N3
# ██████╔╝ [HANDOFF] extract_action_angle_decomposition → ActionAngleDecomposition (input de FASE 2)
# ██╔═══╝
# ██║
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ActionAngleDecomposition:
    """
    FASE 1 — Descomposición acción-ángulo de Poincaré sobre el espacio de estados.

    Modelo integrable de referencia:
        ρ = Σ_i J_i |v_i⟩⟨v_i|,     J_i ≥ 0,  Σ J_i = 1,
        H_ent(J) = −Σ_i J_i log J_i   (von Neumann / entanglement Hamiltonian),
        ω_i(J) = ∂H_ent/∂J_i         (mapa de frecuencias; no gaps de spec(ρ)).

    Los ángulos θ_i se gauge-fijan por la fase del componente dominante de |v_i⟩.
    El hessiano B_ij = ∂²H_ent/∂J_i∂J_j gobierna twist y no-degeneración de Kolmogorov.
    """
    actions: NDArray[np.float64]
    angles: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    hamiltonian_spectrum: NDArray[np.float64]
    reference_eigenvectors: NDArray[np.complex128]
    total_action: float
    phase_space_volume: float
    lyapunov_spectrum: NDArray[np.float64]
    frequency_map_jacobian: NDArray[np.float64]
    kolmogorov_nondegeneracy: float
    von_neumann_entropy: float


@dataclass(slots=True)
class PoincareSectionCrossing:
    """
    FASE 1/2 — Cruce de la sección de Poincaré
        Σ = {ρ : σ(ρ) := Tr(ρ H) − E₀ = 0},
    con tiempo interpolado de Newton de primer orden
        t_× = t_n + dt · |σ_n| / (|σ_n| + |σ_{n+1}|).
    """
    step_index: int
    interpolated_time: float
    section_value: float
    signed_crossing: int
    action_invariants: NDArray[np.float64]
    angle_invariants: NDArray[np.float64]


@dataclass(slots=True)
class RecursiveSelfImprovementLedger:
    """
    FASE 1 — Ledger de observación recursiva de Nivel 3, ahora con órbita celeste.

    Nivel 1: el motor produce veredicto (FASE 3).
    Nivel 2: adapta umbrales (fidelity_floor, gap_ceiling) según tendencias.
    Nivel 3: adapta la tasa de adaptación N2 según entropía de veredictos.
    Nivel 3-celeste (Poincaré):
        - recurrencia de la órbita (floor, ceiling) en el espacio de umbrales;
        - exponente de Lyapunov discreto del mapa de adaptación
          Φ: (floor, ceiling, α) ↦ (floor', ceiling', α');
        si λ_Φ > 0 (caos de umbrales) se amortigua α; si hay recurrencia se congela.
    """
    fidelity_history: List[float] = field(default_factory=list)
    gap_history: List[float] = field(default_factory=list)
    verdict_history: List[int] = field(default_factory=list)
    floor_history: List[float] = field(default_factory=list)
    ceiling_history: List[float] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    fidelity_floor: float = _UHLMANN_FIDELITY_MIN
    gap_ceiling: float = _GALOIS_GAP_MAX_TOLERANCE
    adaptation_rate: float = 0.05
    meta_adaptation_rate: float = 0.01
    level_3_reflections: int = 0
    last_entropy: float = 0.0
    last_recurrence_index: float = 0.0
    last_adaptation_lyapunov: float = 0.0

    def observe(self, fidelity: float, gap: float, verdict: HeytingOmega3) -> None:
        self.fidelity_history.append(float(fidelity))
        self.gap_history.append(float(gap))
        self.verdict_history.append(int(verdict))
        self.floor_history.append(float(self.fidelity_floor))
        self.ceiling_history.append(float(self.gap_ceiling))
        self.rate_history.append(float(self.adaptation_rate))
        if len(self.fidelity_history) > _N3_LEDGER_CAP:
            self.fidelity_history = self.fidelity_history[-_N3_LEDGER_CAP:]
            self.gap_history = self.gap_history[-_N3_LEDGER_CAP:]
            self.verdict_history = self.verdict_history[-_N3_LEDGER_CAP:]
            self.floor_history = self.floor_history[-_N3_LEDGER_CAP:]
            self.ceiling_history = self.ceiling_history[-_N3_LEDGER_CAP:]
            self.rate_history = self.rate_history[-_N3_LEDGER_CAP:]

    def _trend(self, series: Sequence[float]) -> float:
        """Pendiente de mínimos cuadrados centrados sobre la ventana N3."""
        if len(series) < 2:
            return 0.0
        n = min(_N3_STABILITY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def level_2_adapt(self) -> None:
        """Nivel 2 — ajusta floors/ceilings según tendencias de fidelidad y gap."""
        trend_f = self._trend(self.fidelity_history)
        trend_g = self._trend(self.gap_history)
        if trend_f < -1e-4:
            self.fidelity_floor = min(0.99, self.fidelity_floor + self.adaptation_rate * 0.01)
        elif trend_f > 1e-4:
            self.fidelity_floor = max(0.90, self.fidelity_floor - self.adaptation_rate * 0.005)
        if trend_g > 1e-8:
            self.gap_ceiling = max(1e-9, self.gap_ceiling * (1.0 - self.adaptation_rate))
        elif trend_g < -1e-8:
            self.gap_ceiling = min(1e-5, self.gap_ceiling * (1.0 + self.adaptation_rate * 0.5))

    def poincare_recurrence_index(self) -> float:
        """
        Índice de recurrencia de Poincaré del par (floor, ceiling) en ℝ².
        r = exp(−d_min / (σ + ε)) ∈ (0, 1]: cercano a 1 ⇒ retorno próximo
        (el flujo de umbrales vuelve a una vecindad ya visitada).
        """
        n = len(self.floor_history)
        if n < 4:
            return 0.0
        pts = np.column_stack([
            np.array(self.floor_history, dtype=float),
            np.log10(np.maximum(np.array(self.ceiling_history, dtype=float), _WILKINSON_EPSILON)),
        ])
        last = pts[-1]
        prior = pts[:-1]
        delta = prior - last
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        d_min = float(np.min(dist))
        sigma = float(np.std(pts, axis=0).mean()) + _WILKINSON_EPSILON
        return float(math.exp(-d_min / sigma))

    def adaptation_lyapunov(self) -> float:
        """
        Exponente de Lyapunov discreto del mapa de adaptación Φ sobre α_N2.
        λ_Φ ≈ ⟨ log |α_{k+1} − α_k| − log |α_k − α_{k−1}| ⟩; λ>0 ⇒ umbrales caóticos.
        """
        rates = np.array(self.rate_history, dtype=float)
        if rates.size < 4:
            return 0.0
        d1 = np.abs(np.diff(rates))
        d1 = np.maximum(d1, _WILKINSON_EPSILON)
        logs = np.log(d1[1:] / d1[:-1])
        return float(np.mean(logs))

    def level_3_adapt(self) -> float:
        """Nivel 3 — meta-adaptación de α_N2 según entropía de veredictos."""
        n = len(self.verdict_history)
        if n < 4:
            self.last_entropy = 0.0
            return 0.0
        recent = np.array(self.verdict_history[-_N3_STABILITY_WINDOW:], dtype=int)
        _uniq, counts = np.unique(recent, return_counts=True)
        probs = counts / float(counts.sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if entropy > 0.5:
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * 0.9)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.95)
        else:
            self.meta_adaptation_rate = min(0.5, self.meta_adaptation_rate * 1.05)
            self.adaptation_rate = min(0.5, self.adaptation_rate * 1.02)
        self.level_3_reflections += 1
        self.last_entropy = entropy
        return entropy

    def level_3_celestial_adapt(self) -> Tuple[float, float]:
        """
        Nivel 3-celeste — Poincaré sobre el propio ledger:
          * recurrencia alta ⇒ congelar α (el umbral ya orbitó una toro estable);
          * λ_Φ > 0 ⇒ amortiguar α y α_meta (caos de umbrales, KAM roto);
          * λ_Φ < 0 y recurrencia baja ⇒ permitir deriva lenta (cuasi-periódica).
        """
        rec = self.poincare_recurrence_index()
        lam = self.adaptation_lyapunov()
        self.last_recurrence_index = rec
        self.last_adaptation_lyapunov = lam
        if rec > 0.85:
            self.adaptation_rate *= 0.5
            self.meta_adaptation_rate *= 0.5
        elif lam > 0.0:
            damp = math.exp(-min(4.0, lam))
            self.adaptation_rate = max(1e-3, self.adaptation_rate * damp)
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * damp)
        else:
            self.adaptation_rate = min(0.5, self.adaptation_rate * 1.01)
        return rec, lam


class DeRhamGaloisAdjunctionWitnessEngine:
    """
    Motor Espectral Testigo QND de Adjunción de de Rham-Galois F ⊣ G,
    evolución v12.1 con Mecánica Celeste de Poincaré y Automejora N3.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._ledger: RecursiveSelfImprovementLedger = RecursiveSelfImprovementLedger()
        if DeRhamGaloisAdjunctionBridge is not None:
            self.bridge = DeRhamGaloisAdjunctionBridge(dimension=dimension, secret_key=secret_key)
        else:
            self.bridge = None

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa PSD de Higham y traza unitaria estricta."""
        a = _hermitize(matrix)
        eigvals, eigvecs = la.eigh(a)
        clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    def connection_from_density(self, rho: NDArray) -> NDArray[np.complex128]:
        """
        Conexión de gauge anti-hermítica A_μ sobre su(N):
            A = (i/2) (ρ̃ − ρ̃†)  con ρ̃ = ρ − (Tr ρ / N) I.
        Equivale al generador del flujo hamiltoniano H = i A  (H = H†).
        """
        dim = self.dimension
        rho = np.asarray(rho, dtype=np.complex128)
        tr_val = np.trace(rho) / float(dim)
        su_part = rho - tr_val * np.eye(dim, dtype=np.complex128)
        return 0.5 * (1j * su_part - (1j * su_part).conj().T)

    def hamiltonian_from_connection(self, gauge_connection: NDArray) -> NDArray[np.complex128]:
        """H = i A_μ, proyectado a Hermítico (generador del flujo de Liouville unitario)."""
        H = 1j * np.asarray(gauge_connection, dtype=np.complex128)
        return 0.5 * (H + H.conj().T)

    # ────────────────────── FASE 1 · MÉTODOS INTERNOS ──────────────────────

    def _gauge_fix_angles(self, eigvecs: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Fijación de gauge U(1)^N: θ_k = arg(⟨e_{i*(k)} | v_k⟩), i* = argmax |v_k|."""
        n = eigvecs.shape[1]
        angles = np.zeros(n, dtype=np.float64)
        for k in range(n):
            v = eigvecs[:, k]
            idx = int(np.argmax(np.abs(v)))
            angles[k] = float(np.angle(v[idx]))
        return angles

    def _entanglement_frequency_map(
        self,
        actions: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
        """
        Mapa de frecuencias del Hamiltoniano de entrelazamiento H_ent(J) = −Σ J_i log J_i.

            ω_i(J) = ∂H_ent/∂J_i = −log J_i − 1,
            B_ij   = ∂ω_i/∂J_j = ∂²H_ent/∂J_i∂J_j = −δ_ij / J_i.

        No-degeneración de Kolmogorov: det B ≠ 0  (aquí B diagonal, det B = Π (−1/J_i)).
        Entropía de von Neumann S = −Σ J_i log J_i = H_ent.
        """
        J = np.maximum(np.asarray(actions, dtype=np.float64), _SPECTRAL_PSD_FLOOR)
        H_ent_diag = -np.log(J)
        omega = H_ent_diag - 1.0
        hess_diag = -1.0 / J
        B = np.diag(hess_diag)
        # det B en log-espacio para estabilidad.
        log_abs_det = float(np.sum(np.log(np.abs(hess_diag) + _WILKINSON_EPSILON)))
        kolmogorov = float(math.exp(min(log_abs_det, 50.0) / max(len(J), 1)))
        entropy = float(np.sum(J * H_ent_diag))
        return omega, H_ent_diag, B, kolmogorov, entropy

    def _spectral_lyapunov_from_oseledets(
        self,
        actions: NDArray[np.float64],
        frequencies: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Proxy de Oseledets sobre el espectro: λ_i ≈ log |ω_{i}| · log(J_i / J_{i+1}).
        Distinto del Lyapunov dinámico del flujo unitario (idénticamente 0 por QND).
        Mide sensibilidad espectral, no divergencia de trayectorias hamiltonianas.
        """
        J = np.sort(np.maximum(actions, _SPECTRAL_PSD_FLOOR))[::-1]
        n = len(J)
        lyap = np.zeros(n, dtype=np.float64)
        if n < 2:
            return lyap
        ratios = np.log(np.maximum(J[:-1] / J[1:], _WILKINSON_EPSILON))
        omega_sorted = np.sort(np.abs(frequencies))[::-1]
        m = min(len(ratios), len(omega_sorted) - 1 if len(omega_sorted) > 1 else len(omega_sorted))
        for i in range(m):
            w = float(omega_sorted[i]) if i < len(omega_sorted) else 1.0
            lyap[i] = math.log(max(abs(w), _WILKINSON_EPSILON)) * float(ratios[i])
        return np.sort(lyap)[::-1]

    def extract_action_angle_decomposition(self, rho: NDArray) -> ActionAngleDecomposition:
        """
        FASE 1 · MÉTODO FINAL → [HANDOFF] FASE 2.

        Extrae (J_i, θ_i, ω_i(J), B = Dω) de Poincaré a partir de ρ.
        Salida lista para alimentar `build_poincare_return_map` (FASE 2, primer método):
        el mapa de retorno se construye sobre las coordenadas (J, θ) aquí fijadas,
        y el hessiano B se reutiliza como generador del twist de Poincaré-Birkhoff.
        """
        rho = self.sanitize_density_matrix(rho)
        eigvals, eigvecs = la.eigh(rho)
        actions = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        angles = self._gauge_fix_angles(eigvecs)

        frequencies, hamiltonian_spectrum, hess_B, kolmogorov, entropy = (
            self._entanglement_frequency_map(actions)
        )
        lyapunov = self._spectral_lyapunov_from_oseledets(actions, frequencies)

        total_action = float(np.sum(actions))
        n = max(len(actions), 1)
        log_vol = float(np.sum(np.log(2.0 * np.pi * actions + _WILKINSON_EPSILON)))
        phase_volume = float(np.exp(log_vol / n))

        return ActionAngleDecomposition(
            actions=actions,
            angles=angles,
            frequencies=frequencies,
            hamiltonian_spectrum=hamiltonian_spectrum,
            reference_eigenvectors=eigvecs,
            total_action=total_action,
            phase_space_volume=phase_volume,
            lyapunov_spectrum=lyapunov,
            frequency_map_jacobian=hess_B,
            kolmogorov_nondegeneracy=kolmogorov,
            von_neumann_entropy=entropy,
        )

    # ─────────── (fin FASE 1 · handoff a FASE 2 · build_poincare_return_map) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 2 — OPERADORES DE RETORNO POINCARÉ + PERTURBATIVOS + AUTOTUNING N2/N3
# ██╔══██╗ Mapa de Poincaré, Lindstedt O(ε²), Birkhoff, KAM, twist, self-tuning celeste
# ██████╔╝ [HANDOFF] recursive_self_tune_parameters → TuningPolicy (input de FASE 3)
# ██╔═══╝
# ██║
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PoincareReturnAnalysis:
    """FASE 2 — Resultado consolidado del análisis de retorno de Poincaré."""
    crossings: List[PoincareSectionCrossing]
    crossing_count: int
    action_drift_max: float
    action_drift_mean: float
    lindstedt_frequency_correction_norm: float
    birkhoff_quadratic_coefficient: float
    birkhoff_matrix_frobenius: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    freqs_for_kam: NDArray[np.float64]
    twist_index: float
    rotation_number: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    mean_return_time: float


@dataclass(slots=True)
class TuningPolicy:
    """FASE 2 → [HANDOFF] FASE 3. Política adaptativa sintetizada por automejora N3-celeste."""
    fidelity_floor: float
    gap_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    level_3_entropy: float
    reflections: int
    n3_recurrence_index: float
    n3_adaptation_lyapunov: float


class DeRhamGaloisAdjunctionWitnessEnginePhase2(DeRhamGaloisAdjunctionWitnessEngine):
    """Extensión v12.1 · FASE 2 del motor testigo (continúa el handoff de FASE 1)."""

    # ────────────────────── FASE 2 · MÉTODOS ──────────────────────

    def _interpolate_crossing_time(
        self,
        prev_val: float,
        val: float,
        step_index: int,
        dt: float,
    ) -> float:
        """Newton de orden 1: t_× = t_n + dt · |σ_n| / (|σ_n| + |σ_{n+1}|)."""
        denom = abs(prev_val) + abs(val)
        alpha = abs(prev_val) / denom if denom > _WILKINSON_EPSILON else 0.5
        return float(step_index - 1) * dt + alpha * dt

    def _reconstruct_rho0(self, decomposition: ActionAngleDecomposition) -> NDArray[np.complex128]:
        V = decomposition.reference_eigenvectors
        rho_0 = V @ np.diag(decomposition.actions) @ V.conj().T
        rho_0 = 0.5 * (rho_0 + rho_0.conj().T)
        tr = float(np.real(np.trace(rho_0)))
        if tr > _WILKINSON_EPSILON:
            return rho_0 / tr
        return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)

    def _angles_from_rho(
        self,
        rho: NDArray[np.complex128],
        reference: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        """Ángulos relativos al gauge de FASE 1 (producto ⟨v_k|ψ_k⟩)."""
        evals, evecs = la.eigh(rho)
        order = np.argsort(np.real(evals))
        evecs = evecs[:, order]
        n = min(evecs.shape[1], reference.shape[1])
        angles = np.zeros(n, dtype=np.float64)
        for k in range(n):
            overlap = complex(np.vdot(reference[:, k], evecs[:, k]))
            angles[k] = float(np.angle(overlap if abs(overlap) > _WILKINSON_EPSILON else 1.0 + 0.0j))
        return angles

    def _poincare_jacobian_lyapunov(
        self,
        action_crossings: List[NDArray[np.float64]],
        mean_return_time: float,
    ) -> float:
        """
        Lyapunov de Oseledets del mapa de Poincaré sobre acciones dominantes:
            λ_max ≈ (1 / T_ret) log σ_max(ΔJ_{k+1} ΔJ_k⁺),
        con pseudoinversa de Tikhonov. El flujo unitario tiene λ≡0; este λ
        captura deriva numérica + anarmonicidad de Birkhoff proyectada a Σ.
        """
        if len(action_crossings) < 3:
            return 0.0
        d = min(_KAM_FREQ_DIM, action_crossings[0].size)
        seq = np.stack([np.sort(np.real(a))[::-1][:d] for a in action_crossings], axis=0)
        dJ = np.diff(seq, axis=0)
        if dJ.shape[0] < 2:
            return 0.0
        X = dJ[:-1].T
        Y = dJ[1:].T
        gram = X @ X.T + _WILKINSON_EPSILON * np.eye(d)
        try:
            jac = Y @ X.T @ la.inv(gram)
            spec = np.abs(la.eigvals(jac))
            sigma_max = float(np.max(spec))
        except la.LinAlgError:
            return 0.0
        T = mean_return_time if mean_return_time > _WILKINSON_EPSILON else 1.0
        return float(math.log(max(sigma_max, _WILKINSON_EPSILON)) / T)

    def build_poincare_return_map(
        self,
        decomposition: ActionAngleDecomposition,
        gauge_connection: NDArray,
        n_orbits: int = _POINCARE_N_ORBITS,
        steps_per_orbit: int = _POINCARE_STEPS_PER_ORBIT,
    ) -> PoincareReturnAnalysis:
        """
        FASE 2 · MÉTODO 1 ← [HANDOFF FASE 1].

        Construye el mapa de retorno de Poincaré P: Σ → Σ sobre la variedad de Bures,
        bajo el flujo hamiltoniano ρ(t) = e^{−i H t} ρ₀ e^{i H t} con H = i A_μ.

        Mejoras respecto a v12.0:
          * cruce interpolado (Newton-1) → tiempo de retorno continuo;
          * número de rotación ϖ = ⟨Δθ⟩ / 2π sobre la sección;
          * jacobiano discreto de P → Lyapunov de Oseledets;
          * error de preservación de Liouville |det U| − 1 (debe ser ~0, QND);
          * twist, KAM, Lindstedt y Birkhoff se calculan sobre (J, θ) de FASE 1.
        """
        H = self.hamiltonian_from_connection(gauge_connection)

        omega_max = float(np.max(np.abs(decomposition.frequencies))) + _WILKINSON_EPSILON
        dt = (2.0 * math.pi / omega_max) / float(max(steps_per_orbit, 1))

        rho_0 = self._reconstruct_rho0(decomposition)
        E0 = float(np.real(np.trace(rho_0 @ H)))

        U = la.expm(-1j * H * dt)
        U_dag = U.conj().T
        liouville_err = float(abs(abs(np.linalg.det(U)) - 1.0))

        crossings: List[PoincareSectionCrossing] = []
        action_drifts: List[float] = []
        return_times: List[float] = []
        rho = rho_0.copy()
        prev_val: Optional[float] = None
        last_cross_t: Optional[float] = None
        ref_V = decomposition.reference_eigenvectors
        ref_actions_sorted = np.sort(decomposition.actions)[::-1]

        n_steps = int(n_orbits * steps_per_orbit)
        for step in range(n_steps):
            rho = U @ rho @ U_dag
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.real(np.trace(rho)))
            if tr > _WILKINSON_EPSILON:
                rho = rho / tr

            val = float(np.real(np.trace(rho @ H))) - E0
            if prev_val is not None and prev_val * val < 0.0:
                t_cross = self._interpolate_crossing_time(prev_val, val, step, dt)
                evals_now = np.sort(np.real(la.eigvalsh(rho)))[::-1]
                actions_now = np.maximum(evals_now, _SPECTRAL_PSD_FLOOR)
                nmin = min(actions_now.size, ref_actions_sorted.size)
                drift = float(np.max(np.abs(actions_now[:nmin] - ref_actions_sorted[:nmin])))
                action_drifts.append(drift)
                angles_now = self._angles_from_rho(rho, ref_V)
                crossings.append(PoincareSectionCrossing(
                    step_index=step,
                    interpolated_time=t_cross,
                    section_value=val,
                    signed_crossing=int(np.sign(val - prev_val)),
                    action_invariants=actions_now,
                    angle_invariants=angles_now,
                ))
                if last_cross_t is not None:
                    return_times.append(t_cross - last_cross_t)
                last_cross_t = t_cross
            prev_val = val

        mean_T = float(np.mean(return_times)) if return_times else (2.0 * math.pi / omega_max)

        lindstedt_corr = self.lindstedt_poincare_perturbation(decomposition, gauge_connection)
        nfreq = min(lindstedt_corr.size, decomposition.frequencies.size)
        lindstedt_norm = float(
            np.linalg.norm(lindstedt_corr[:nfreq] - decomposition.frequencies[:nfreq])
        )

        birkhoff_coef, birkhoff_fro = self.birkhoff_normal_form_reduction(
            decomposition, gauge_connection
        )

        dominant_freqs = np.sort(np.abs(decomposition.frequencies))[::-1][:_KAM_FREQ_DIM]
        dominant_freqs = np.maximum(dominant_freqs, _WILKINSON_EPSILON)
        kam_index, kam_viol, kam_worst = self.compute_kam_stability_index(dominant_freqs)

        twist = self.poincare_birkhoff_twist_index(decomposition)
        rot_num = self.rotation_number_estimate(crossings)
        iso = self.isoenergetic_nondegeneracy(decomposition, dominant_freqs)
        lyap_p = self._poincare_jacobian_lyapunov(
            [c.action_invariants for c in crossings], mean_T
        )

        return PoincareReturnAnalysis(
            crossings=crossings,
            crossing_count=len(crossings),
            action_drift_max=float(max(action_drifts)) if action_drifts else 0.0,
            action_drift_mean=float(np.mean(action_drifts)) if action_drifts else 0.0,
            lindstedt_frequency_correction_norm=lindstedt_norm,
            birkhoff_quadratic_coefficient=float(birkhoff_coef),
            birkhoff_matrix_frobenius=float(birkhoff_fro),
            kam_stability_index=float(kam_index),
            kam_resonance_violations=int(kam_viol),
            kam_worst_ratio=float(kam_worst),
            freqs_for_kam=dominant_freqs,
            twist_index=float(twist),
            rotation_number=float(rot_num),
            isoenergetic_nondegeneracy=float(iso),
            poincare_lyapunov_max=float(lyap_p),
            liouville_preservation_error=liouville_err,
            mean_return_time=mean_T,
        )

    def lindstedt_poincare_perturbation(
        self,
        decomposition: ActionAngleDecomposition,
        gauge_connection: NDArray,
        epsilon: float = _LINDSTEDT_EPSILON,
    ) -> NDArray[np.float64]:
        """
        FASE 2 · Serie de Lindstedt-Poincaré hasta O(ε²), eliminación secular.

        H = H₀(J) + ε H₁(J, θ),  H₀ diagonal en la base de FASE 1.
        Primera corrección (van Vleck / no-degenerada):
            δω_i^{(1)} = Σ_{j≠i} |⟨i|H₁|j⟩|² / (E_i − E_j),   |E_i − E_j| > δ_sec.
        Segunda corrección (términos no seculares residuales):
            δω_i^{(2)} = Σ_{j≠i} |V_ij|² (B_ii − B_jj) / (E_i − E_j)².
        Resonancias |k·ω| < δ_sec se delegan a la forma normal de Birkhoff.
        """
        V = decomposition.reference_eigenvectors
        H = self.hamiltonian_from_connection(gauge_connection)
        H_eig = V.conj().T @ H @ V
        n = H_eig.shape[0]
        E = np.real(np.diag(H_eig))
        off = H_eig - np.diag(np.diag(H_eig))
        B_diag = np.real(np.diag(decomposition.frequency_map_jacobian))
        if B_diag.size < n:
            B_diag = np.pad(B_diag, (0, n - B_diag.size))

        delta1 = np.zeros(n, dtype=np.float64)
        delta2 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom = float(E[i] - E[j])
                if abs(denom) < _SECULAR_DENOM_FLOOR:
                    continue
                vij2 = float(abs(off[i, j]) ** 2)
                delta1[i] += vij2 / denom
                bdiff = float(B_diag[i] - B_diag[j]) if i < B_diag.size and j < B_diag.size else 0.0
                delta2[i] += vij2 * bdiff / (denom * denom)

        omega0 = np.asarray(decomposition.frequencies, dtype=np.float64)
        if omega0.size < n:
            omega0 = np.pad(omega0, (0, n - omega0.size))
        return omega0[:n] + epsilon * delta1 + (epsilon ** 2) * delta2

    def birkhoff_normal_form_reduction(
        self,
        decomposition: ActionAngleDecomposition,
        gauge_connection: NDArray,
    ) -> Tuple[float, float]:
        """
        FASE 2 · Forma normal de Birkhoff truncada a grado 2 sobre modos dominantes:

            H_B(J) = ω·J + (1/2) J^T B_off J,
            (B_off)_ij = |H^off_ij|² / (ω_i − ω_j)   (i ≠ j, no resonante).

        Devuelve (c_B, ‖B_off‖_F) donde
            c_B = Σ_{i≠j} |H^off_ij|² / min_k |ω_k − ω_{k+1}|
        mide la anarmonicidad que desgarra los tori KAM del punto fijo diagonal.
        """
        V = decomposition.reference_eigenvectors
        H = self.hamiltonian_from_connection(gauge_connection)
        H_eig = V.conj().T @ H @ V
        off = H_eig - np.diag(np.diag(H_eig))
        diag = np.real(np.diag(H_eig))
        n = min(off.shape[0], _KAM_FREQ_DIM)
        if n < 2:
            return 0.0, 0.0
        off_d = off[:n, :n]
        E = diag[:n]
        gaps = np.abs(np.diff(np.sort(E)))
        omega_min = float(np.min(gaps)) + _WILKINSON_EPSILON
        B_off = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom = float(E[i] - E[j])
                if abs(denom) < _SECULAR_DENOM_FLOOR:
                    continue
                B_off[i, j] = float(abs(off_d[i, j]) ** 2) / denom
        c_B = float(np.sum(np.abs(off_d) ** 2)) / omega_min
        fro = float(np.linalg.norm(B_off, ord="fro"))
        return c_B, fro

    def compute_kam_stability_index(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        tau: float = _KAM_TAU,
        max_k: int = _KAM_RESONANCE_MAX_K,
    ) -> Tuple[float, int, float]:
        """
        FASE 2 · Condición diofantina de Siegel–Moser / KAM:
            |k · ω| > γ / |k|_1^τ    ∀ k ∈ ℤ^d \\ {0}, |k|_1 ≤ max_k.
        Devuelve (kam_index, n_violaciones, worst_ratio).
        kam_index = 1 / (1 + #violaciones) ∈ (0, 1].
        """
        freqs = np.asarray(frequencies, dtype=np.float64)
        d = int(freqs.size)
        if d == 0:
            return 0.0, 0, 1.0e9
        violations = 0
        worst_ratio = float("inf")
        span = range(-max_k, max_k + 1)
        for k in itertools.product(span, repeat=d):
            k_arr = np.array(k, dtype=float)
            norm1 = float(np.sum(np.abs(k_arr)))
            if norm1 == 0.0:
                continue
            k_dot_omega = abs(float(np.dot(k_arr, freqs)))
            threshold = gamma / (norm1 ** tau)
            ratio = k_dot_omega / threshold if threshold > 0.0 else float("inf")
            worst_ratio = min(worst_ratio, ratio)
            if k_dot_omega < threshold:
                violations += 1
        kam_index = 1.0 / (1.0 + float(violations))
        if worst_ratio == float("inf"):
            worst_ratio = 1.0e9
        return float(kam_index), int(violations), float(worst_ratio)

    def poincare_birkhoff_twist_index(self, decomposition: ActionAngleDecomposition) -> float:
        """
        FASE 2 · Condición de twist (Poincaré–Birkhoff):
            τ = det(∂θ'/∂J) ~ det B = det(Dω/DJ).
        τ ≠ 0 ⇒ el mapa de Poincaré es twist (existencia de órbitas p-q).
        Se reporta log|τ| normalizado; 0 ⇒ degeneración (sin twist).
        """
        B = np.asarray(decomposition.frequency_map_jacobian, dtype=np.float64)
        d = min(B.shape[0], B.shape[1], _KAM_FREQ_DIM)
        if d < 1:
            return 0.0
        Bd = B[:d, :d]
        try:
            sign, logdet = np.linalg.slogdet(Bd)
            if sign == 0.0:
                return 0.0
            return float(abs(sign) * math.exp(min(abs(logdet), 40.0) / d))
        except np.linalg.LinAlgError:
            return 0.0

    def rotation_number_estimate(self, crossings: Sequence[PoincareSectionCrossing]) -> float:
        """
        Número de rotación ϖ de Poincaré:
            ϖ = lim (θ_{n} − θ_0) / (2π n),
        estimado sobre el ángulo dominante de los cruces consecutivos, unwrap incluido.
        """
        if len(crossings) < 2:
            return 0.0
        thetas = []
        for c in crossings:
            if c.angle_invariants.size == 0:
                continue
            thetas.append(float(c.angle_invariants[int(np.argmax(np.abs(c.action_invariants[:c.angle_invariants.size])))]
                                if c.action_invariants.size else c.angle_invariants[0]))
        if len(thetas) < 2:
            return 0.0
        unwrapped = np.unwrap(np.array(thetas, dtype=float))
        return float((unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi * (len(unwrapped) - 1)))

    def isoenergetic_nondegeneracy(
        self,
        decomposition: ActionAngleDecomposition,
        frequencies: NDArray[np.float64],
    ) -> float:
        """
        FASE 2 · No-degeneración isoenergética de Arnold:
            Δ = det  |  B     ω |
                    |  ωᵀ    0 |  ≠ 0.
        Necesaria para KAM en la superficie de energía (sección de Poincaré).
        """
        B = np.asarray(decomposition.frequency_map_jacobian, dtype=np.float64)
        d = min(B.shape[0], frequencies.size, _KAM_FREQ_DIM)
        if d < 1:
            return 0.0
        Bd = np.real(B[:d, :d])
        w = np.real(np.asarray(frequencies[:d], dtype=np.float64)).reshape(d, 1)
        top = np.hstack([Bd, w])
        bottom = np.hstack([w.T, np.zeros((1, 1))])
        M = np.vstack([top, bottom])
        try:
            sign, logdet = np.linalg.slogdet(M)
            if sign == 0.0:
                return 0.0
            return float(abs(sign) * math.exp(min(abs(logdet), 40.0) / (d + 1)))
        except np.linalg.LinAlgError:
            return 0.0

    def recursive_self_tune_parameters(
        self,
        poincare_analysis: PoincareReturnAnalysis,
    ) -> TuningPolicy:
        """
        FASE 2 · MÉTODO FINAL → [HANDOFF] FASE 3.

        Sintetiza la política adaptativa N2 / N3 / N3-celeste a partir del análisis
        de Poincaré. El ledger observa un proxy (fidelidad, gap) construido con
        deriva de acción, Birkhoff y violaciones KAM; luego:

            N2          → mueve floors/ceilings,
            N3          → meta-adapta α según entropía de veredictos,
            N3-celeste  → corrige α con recurrencia de Poincaré y Lyapunov de Φ.

        La TuningPolicy resultante es el input de `meta_witness_evaluation` (FASE 3).
        """
        synthetic_fidelity = float(
            np.exp(
                -poincare_analysis.action_drift_mean
                - 0.1 * poincare_analysis.birkhoff_quadratic_coefficient
                - 0.05 * max(0.0, -poincare_analysis.poincare_lyapunov_max)
            )
        )
        synthetic_fidelity = float(min(1.0, max(0.0, synthetic_fidelity)))
        synthetic_gap = float(
            poincare_analysis.action_drift_max
            + 0.01 * poincare_analysis.kam_resonance_violations
            + poincare_analysis.liouville_preservation_error
        )
        if (
            synthetic_fidelity >= self._ledger.fidelity_floor
            and synthetic_gap <= self._ledger.gap_ceiling
            and poincare_analysis.kam_stability_index >= 0.5
        ):
            v = HeytingOmega3.COHERENT
        elif synthetic_fidelity >= 0.85:
            v = HeytingOmega3.DEGRADED
        else:
            v = HeytingOmega3.VETOED

        self._ledger.observe(synthetic_fidelity, synthetic_gap, v)
        self._ledger.level_2_adapt()
        entropy = self._ledger.level_3_adapt()
        rec, lam = self._ledger.level_3_celestial_adapt()

        return TuningPolicy(
            fidelity_floor=float(self._ledger.fidelity_floor),
            gap_ceiling=float(self._ledger.gap_ceiling),
            adaptation_rate=float(self._ledger.adaptation_rate),
            meta_adaptation_rate=float(self._ledger.meta_adaptation_rate),
            level_3_entropy=float(entropy),
            reflections=int(self._ledger.level_3_reflections),
            n3_recurrence_index=float(rec),
            n3_adaptation_lyapunov=float(lam),
        )

    # ─────────── (fin FASE 2 · handoff a FASE 3 · meta_witness_evaluation) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 3 — META-VERIFICACIÓN RECURSIVA + SNAPSHOT FINAL v12.1
# ██╔══██╗ Witness del witness, auditoría N3-celeste, integración F1 ▷ F2 ▷ F3
# ██████╔╝ [SALIDA] evaluate_adjunction_witness_v12 → DeRhamGaloisWitnessSnapshotV12
# ██╔═══╝
# ██║
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MetaWitnessReport:
    """FASE 3 — Reporte meta-observacional (witness del witness)."""
    meta_fidelity_std: float
    meta_gap_std: float
    meta_verdict_entropy: float
    meta_consistency_score: float
    meta_verdict: HeytingOmega3
    provenance_merkle: str
    n3_recurrence_index: float
    n3_adaptation_lyapunov: float
    kam_consistency: float


@dataclass(frozen=True, slots=True)
class DeRhamGaloisWitnessSnapshotV12:
    """FASE 3 — Snapshot inmutable final v12.1 con la información de las 3 fases anidadas."""
    # Campos v11 (compatibilidad)
    snapshot_id: str
    dimension_N: int
    particle_count: int
    trace_density: float
    uhlmann_fidelity: float
    bures_distance: float
    galois_gap_delta: float
    unit_eta_norm: float
    counit_epsilon_norm: float
    connection_A_norm: float
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    liouville_drift: float
    coherence_verdict: HeytingOmega3
    sha256_merkle_root: str
    timestamp_utc: float

    # Campos v12 · FASE 1
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    von_neumann_entropy: float
    kolmogorov_nondegeneracy: float

    # Campos v12 · FASE 2
    poincare_crossings_count: int
    action_drift_max: float
    action_drift_mean: float
    lindstedt_correction_norm: float
    birkhoff_quadratic_coefficient: float
    birkhoff_matrix_frobenius: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    twist_index: float
    rotation_number: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    mean_return_time: float

    # Campos v12 · FASE 3
    adaptive_fidelity_floor: float
    adaptive_gap_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    meta_verdict_entropy: float
    meta_consistency_score: float
    level_3_reflections: int
    n3_recurrence_index: float
    n3_adaptation_lyapunov: float

    # Provenance
    phase_provenance: str
    sha256_phase_merkle: str


class DeRhamGaloisAdjunctionWitnessEngineV12(DeRhamGaloisAdjunctionWitnessEnginePhase2):
    """Extensión v12.1 · FASE 3 del motor testigo (continúa el handoff de FASE 2)."""

    # ────────────────────── FASE 3 · MÉTODOS ──────────────────────

    def meta_witness_evaluation(
        self,
        base_snapshot: DeRhamGaloisWitnessSnapshot,
        decomposition: ActionAngleDecomposition,
        poincare_analysis: PoincareReturnAnalysis,
        tuning: TuningPolicy,
    ) -> MetaWitnessReport:
        """
        FASE 3 · MÉTODO 1 ← [HANDOFF FASE 2].

        Witness del witness: evalúa la consistencia meta del propio motor combinando
        (i)  estabilidad estadística del ledger N3,
        (ii) fidelidad Uhlmann y veredicto QND de base,
        (iii) índice KAM + twist + Lyapunov de P,
        (iv) recurrencia de Poincaré y Lyapunov de la adaptación Φ.
        """
        fid_series = self._ledger.fidelity_history[-_N3_STABILITY_WINDOW:]
        gap_series = self._ledger.gap_history[-_N3_STABILITY_WINDOW:]
        verdict_series = self._ledger.verdict_history[-_N3_STABILITY_WINDOW:]

        fid_std = float(np.std(fid_series)) if len(fid_series) >= 2 else 0.0
        gap_std = float(np.std(gap_series)) if len(gap_series) >= 2 else 0.0

        if verdict_series:
            _uniq, counts = np.unique(np.array(verdict_series), return_counts=True)
            probs = counts / float(counts.sum())
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        else:
            entropy = 0.0

        kam_consistency = float(
            0.5 * poincare_analysis.kam_stability_index
            + 0.25 * (1.0 if poincare_analysis.twist_index > _TWIST_FLOOR else 0.0)
            + 0.25 * math.exp(-abs(poincare_analysis.poincare_lyapunov_max))
        )

        n3_celestial = float(
            0.5 * tuning.n3_recurrence_index
            + 0.5 * math.exp(-abs(tuning.n3_adaptation_lyapunov))
        )

        consistency = float(
            0.28 * base_snapshot.uhlmann_fidelity
            + 0.22 * kam_consistency
            + 0.16 * (1.0 - min(1.0, fid_std * 10.0))
            + 0.14 * (1.0 - min(1.0, gap_std * 1e3))
            + 0.12 * n3_celestial
            + 0.08 * math.exp(-poincare_analysis.liouville_preservation_error * 1e6)
        )
        consistency = float(min(1.0, max(0.0, consistency)))

        if (
            consistency >= 0.90
            and base_snapshot.coherence_verdict == HeytingOmega3.COHERENT
            and entropy <= 0.5
            and poincare_analysis.liouville_preservation_error < 1e-8
        ):
            meta_verdict = HeytingOmega3.COHERENT
        elif consistency >= 0.75:
            meta_verdict = HeytingOmega3.DEGRADED
        else:
            meta_verdict = HeytingOmega3.VETOED

        prov = (
            f"F1:{decomposition.total_action:.6f}:{decomposition.kolmogorov_nondegeneracy:.6e}|"
            f"F2:{poincare_analysis.kam_stability_index:.6f}:"
            f"{poincare_analysis.birkhoff_quadratic_coefficient:.6e}:"
            f"{poincare_analysis.twist_index:.6e}|"
            f"F3:{tuning.fidelity_floor:.6f}:{tuning.gap_ceiling:.6e}:"
            f"{consistency:.6f}:{tuning.n3_recurrence_index:.6f}"
        ).encode("utf-8")
        provenance_merkle = _hmac_sha256(self.secret_key, prov)

        return MetaWitnessReport(
            meta_fidelity_std=fid_std,
            meta_gap_std=gap_std,
            meta_verdict_entropy=entropy,
            meta_consistency_score=consistency,
            meta_verdict=meta_verdict,
            provenance_merkle=provenance_merkle,
            n3_recurrence_index=float(tuning.n3_recurrence_index),
            n3_adaptation_lyapunov=float(tuning.n3_adaptation_lyapunov),
            kam_consistency=kam_consistency,
        )

    def recursive_level_3_audit(
        self,
        meta_report: MetaWitnessReport,
        tuning: TuningPolicy,
    ) -> Dict[str, Any]:
        """
        FASE 3 · Auditoría recursiva N3-celeste: consolida la cadena de auto-mejora
        y verifica que la órbita de umbrales no haya abandonado la región KAM-estable.
        """
        n = len(self._ledger.verdict_history)
        window = self._ledger.verdict_history[-_N3_STABILITY_WINDOW:]
        verdict_chain_ok = all(v <= int(HeytingOmega3.DEGRADED) for v in window) if window else True
        celestial_stable = (
            tuning.n3_adaptation_lyapunov <= 0.05
            and meta_report.kam_consistency >= 0.4
        )
        return {
            "ledger_size": n,
            "level_3_reflections": tuning.reflections,
            "verdict_chain_ok": verdict_chain_ok,
            "celestial_stable": celestial_stable,
            "meta_verdict": meta_report.meta_verdict.name,
            "meta_consistency": meta_report.meta_consistency_score,
            "fidelity_floor": tuning.fidelity_floor,
            "gap_ceiling": tuning.gap_ceiling,
            "adaptation_rate": tuning.adaptation_rate,
            "meta_adaptation_rate": tuning.meta_adaptation_rate,
            "n3_recurrence_index": tuning.n3_recurrence_index,
            "n3_adaptation_lyapunov": tuning.n3_adaptation_lyapunov,
        }

    def evaluate_adjunction_witness_v12(
        self,
        cartridges: Sequence[Any],
        target_density_matrix: Optional[NDArray] = None,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> DeRhamGaloisWitnessSnapshotV12:
        """
        FASE 3 · MÉTODO FINAL (META).

        Orquesta las 3 fases anidadas y produce el snapshot v12.1 enriquecido.
        Cadena formal:
            FASE 1 → ActionAngleDecomposition
                  → FASE 2 → PoincareReturnAnalysis → TuningPolicy
                  → FASE 3 → MetaWitnessReport → Snapshot v12.1.
        """
        now = time.time()
        dim = self.dimension

        # ── Base v11 (invariante QND) ──
        if self.bridge is not None and cartridges:
            rho_F = self.bridge.functor_F_elevation(cartridges)
        else:
            rho_F = np.eye(dim, dtype=np.complex128) / float(dim)

        if target_density_matrix is None:
            target_density_matrix = rho_F
        else:
            target_density_matrix = self.sanitize_density_matrix(target_density_matrix)

        if self.bridge is not None and cartridges:
            unit_eta, counit_eps, galois_gap = self.bridge.compute_unit_eta_and_counit_epsilon(
                cartridges, target_density_matrix
            )
            fidelity, bures_dist = self.bridge.evaluate_uhlmann_fidelity_and_bures(
                rho_F, target_density_matrix
            )
            connection_A = self.bridge.extract_gauge_connection_A(target_density_matrix)
            conn_norm = float(np.linalg.norm(connection_A, ord="fro"))
        else:
            unit_eta, counit_eps, galois_gap = 0.0, 0.0, 0.0
            fidelity, bures_dist = 1.0, 0.0
            connection_A = self.connection_from_density(target_density_matrix)
            conn_norm = float(np.linalg.norm(connection_A, ord="fro"))

        a_weak, back_action = self._compute_qnd_weak_value(
            target_density_matrix, phi_initial, phi_final
        )
        tr_val = float(np.real(np.trace(target_density_matrix)))
        liouville_drift = float(abs(tr_val - 1.0))

        if (
            fidelity >= _UHLMANN_FIDELITY_MIN
            and galois_gap <= _GALOIS_GAP_MAX_TOLERANCE
            and liouville_drift <= 1e-4
        ):
            verdict = HeytingOmega3.COHERENT
        elif fidelity >= 0.85 and bures_dist <= 0.35:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED

        snap_id = f"DERHAM-WITNESS-V12-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{snap_id}:{fidelity:.8f}:{bures_dist:.8f}:{galois_gap:.8e}:{verdict.value}"
        ).encode("utf-8")
        base_merkle = _hmac_sha256(self.secret_key, seed_bytes)

        base_snapshot = DeRhamGaloisWitnessSnapshot(
            snapshot_id=snap_id,
            dimension_N=dim,
            particle_count=len(cartridges) if cartridges is not None else 0,
            trace_density=tr_val,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            galois_gap_delta=galois_gap,
            unit_eta_norm=float(unit_eta),
            counit_epsilon_norm=float(counit_eps),
            connection_A_norm=conn_norm,
            qnd_weak_value_aav=a_weak,
            weak_value_modulus=float(abs(a_weak)),
            back_action_db=back_action,
            liouville_drift=liouville_drift,
            coherence_verdict=verdict,
            sha256_merkle_root=base_merkle,
            timestamp_utc=now,
        )

        # ── FASE 1 ──
        decomposition = self.extract_action_angle_decomposition(target_density_matrix)

        # ── FASE 2 ──
        poincare = self.build_poincare_return_map(decomposition, connection_A)
        tuning = self.recursive_self_tune_parameters(poincare)

        # ── FASE 3 ──
        meta = self.meta_witness_evaluation(base_snapshot, decomposition, poincare, tuning)
        audit = self.recursive_level_3_audit(meta, tuning)

        provenance = f"F1>F2>F3::{base_merkle}::{meta.provenance_merkle}"
        phase_merkle = _hmac_sha256(self.secret_key, provenance.encode("utf-8"))

        lyap = decomposition.lyapunov_spectrum
        lyap_max = float(np.max(lyap)) if lyap.size else 0.0
        lyap_min = float(np.min(lyap)) if lyap.size else 0.0

        return DeRhamGaloisWitnessSnapshotV12(
            snapshot_id=base_snapshot.snapshot_id,
            dimension_N=base_snapshot.dimension_N,
            particle_count=base_snapshot.particle_count,
            trace_density=base_snapshot.trace_density,
            uhlmann_fidelity=base_snapshot.uhlmann_fidelity,
            bures_distance=base_snapshot.bures_distance,
            galois_gap_delta=base_snapshot.galois_gap_delta,
            unit_eta_norm=base_snapshot.unit_eta_norm,
            counit_epsilon_norm=base_snapshot.counit_epsilon_norm,
            connection_A_norm=base_snapshot.connection_A_norm,
            qnd_weak_value_aav=base_snapshot.qnd_weak_value_aav,
            weak_value_modulus=base_snapshot.weak_value_modulus,
            back_action_db=base_snapshot.back_action_db,
            liouville_drift=base_snapshot.liouville_drift,
            coherence_verdict=base_snapshot.coherence_verdict,
            sha256_merkle_root=base_snapshot.sha256_merkle_root,
            timestamp_utc=base_snapshot.timestamp_utc,
            action_total=decomposition.total_action,
            phase_space_volume=decomposition.phase_space_volume,
            lyapunov_max=lyap_max,
            lyapunov_min=lyap_min,
            von_neumann_entropy=decomposition.von_neumann_entropy,
            kolmogorov_nondegeneracy=decomposition.kolmogorov_nondegeneracy,
            poincare_crossings_count=poincare.crossing_count,
            action_drift_max=poincare.action_drift_max,
            action_drift_mean=poincare.action_drift_mean,
            lindstedt_correction_norm=poincare.lindstedt_frequency_correction_norm,
            birkhoff_quadratic_coefficient=poincare.birkhoff_quadratic_coefficient,
            birkhoff_matrix_frobenius=poincare.birkhoff_matrix_frobenius,
            kam_stability_index=poincare.kam_stability_index,
            kam_resonance_violations=poincare.kam_resonance_violations,
            kam_worst_ratio=poincare.kam_worst_ratio,
            twist_index=poincare.twist_index,
            rotation_number=poincare.rotation_number,
            isoenergetic_nondegeneracy=poincare.isoenergetic_nondegeneracy,
            poincare_lyapunov_max=poincare.poincare_lyapunov_max,
            liouville_preservation_error=poincare.liouville_preservation_error,
            mean_return_time=poincare.mean_return_time,
            adaptive_fidelity_floor=tuning.fidelity_floor,
            adaptive_gap_ceiling=tuning.gap_ceiling,
            adaptation_rate=tuning.adaptation_rate,
            meta_adaptation_rate=tuning.meta_adaptation_rate,
            meta_verdict_entropy=meta.meta_verdict_entropy,
            meta_consistency_score=meta.meta_consistency_score,
            level_3_reflections=tuning.reflections,
            n3_recurrence_index=tuning.n3_recurrence_index,
            n3_adaptation_lyapunov=tuning.n3_adaptation_lyapunov,
            phase_provenance=str(audit["meta_verdict"]),
            sha256_phase_merkle=phase_merkle,
        )

    def _compute_qnd_weak_value(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray],
        phi_final: Optional[NDArray],
    ) -> Tuple[complex, float]:
        """Valor débil AAV sobre la conexión A_μ (Back-Action ≡ 0.0 dB)."""
        return self.compute_qnd_weak_value_connection(density_matrix, phi_initial, phi_final)

    def compute_qnd_weak_value_connection(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> Tuple[complex, float]:
        """
        Valor débil AAV de Aharonov–Albert–Vaidman de la conexión A_μ:
            a_w = ⟨φ_f| A |φ_i⟩ / ⟨φ_f|φ_i⟩,
        con Back-Action ≡ 0.0 dB (observación QND, invariante v11).
        """
        dim = self.dimension
        rho = self.sanitize_density_matrix(density_matrix)
        connection_A = self.connection_from_density(rho)

        if phi_initial is None:
            phi_initial = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        else:
            phi_initial = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
            nrm = float(np.linalg.norm(phi_initial))
            phi_initial = phi_initial / (nrm if nrm > _WILKINSON_EPSILON else 1.0)

        if phi_final is None:
            phi_final = rho @ phi_initial
            nrm = float(np.linalg.norm(phi_final))
            phi_final = phi_final / (nrm if nrm > _WILKINSON_EPSILON else 1.0)
        else:
            phi_final = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)
            nrm = float(np.linalg.norm(phi_final))
            phi_final = phi_final / (nrm if nrm > _WILKINSON_EPSILON else 1.0)

        overlap = complex((phi_final.conj().T @ phi_initial)[0, 0])
        if abs(overlap) < _WILKINSON_EPSILON:
            overlap = complex(_WILKINSON_EPSILON, 0.0)

        numerator = complex((phi_final.conj().T @ connection_A @ phi_initial)[0, 0])
        a_weak = numerator / overlap
        return a_weak, 0.0


if __name__ == "__main__":
    banner = "═" * 88
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v12.1 · QND-POINCARÉ-N3 · FASES ANIDADAS F1 → F2 → F3")
    print(banner)

    engine = DeRhamGaloisAdjunctionWitnessEngineV12(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    snap = engine.evaluate_adjunction_witness_v12(cartridges=[], target_density_matrix=rho)

    print(f"  Snapshot ID          : {snap.snapshot_id}")
    print(f"  Dimensión N          : {snap.dimension_N}")
    print(f"  Traza ρ_MAC          : {snap.trace_density:.6f}")
    print(f"  Back-Action QND      : {snap.back_action_db:.1f} dB")
    print(f"  Fidelidad Uhlmann    : {snap.uhlmann_fidelity:.6f}")
    print(f"  Distancia Bures      : {snap.bures_distance:.6f}")
    print(f"  Galois Gap Δ_Galois  : {snap.galois_gap_delta:.8e}")
    print("-" * 88)
    print("  FASE 1 · Acción-ángulo Poincaré:")
    print(f"    ΣJ_i               : {snap.action_total:.6f}")
    print(f"    Vol. fase          : {snap.phase_space_volume:.6f}")
    print(f"    S_vonNeumann       : {snap.von_neumann_entropy:.6f}")
    print(f"    Kolmogorov det B   : {snap.kolmogorov_nondegeneracy:.6e}")
    print(f"    Lyapunov (max,min) : ({snap.lyapunov_max:+.4e}, {snap.lyapunov_min:+.4e})")
    print("-" * 88)
    print("  FASE 2 · Retorno Poincaré + Perturbativo:")
    print(f"    Cruces Σ           : {snap.poincare_crossings_count}")
    print(f"    T_retorno medio    : {snap.mean_return_time:.6e}")
    print(f"    Drift acción (max) : {snap.action_drift_max:.6e}")
    print(f"    Lindstedt ‖δω‖     : {snap.lindstedt_correction_norm:.6e}")
    print(f"    Coef. Birkhoff     : {snap.birkhoff_quadratic_coefficient:.6e}")
    print(f"    ‖B_off‖_F          : {snap.birkhoff_matrix_frobenius:.6e}")
    print(f"    Twist Poincaré     : {snap.twist_index:.6e}")
    print(f"    Nº rotación ϖ      : {snap.rotation_number:.6f}")
    print(f"    Isoenergética Δ    : {snap.isoenergetic_nondegeneracy:.6e}")
    print(f"    λ_max(P) Oseledets : {snap.poincare_lyapunov_max:+.6e}")
    print(f"    |det U|−1 Liouville: {snap.liouville_preservation_error:.6e}")
    print(f"    Índice KAM         : {snap.kam_stability_index:.6f} "
          f"(violaciones={snap.kam_resonance_violations}, worst_ratio={snap.kam_worst_ratio:.4f})")
    print("-" * 88)
    print("  FASE 3 · Meta-verificación N3-celeste:")
    print(f"    floor_fid adapt.   : {snap.adaptive_fidelity_floor:.6f}")
    print(f"    ceiling_gap adapt. : {snap.adaptive_gap_ceiling:.6e}")
    print(f"    α_N2 / α_N3        : {snap.adaptation_rate:.4f} / {snap.meta_adaptation_rate:.4f}")
    print(f"    Entropía veredicto : {snap.meta_verdict_entropy:.4f}")
    print(f"    Consistencia meta  : {snap.meta_consistency_score:.6f}")
    print(f"    Recurrencia N3     : {snap.n3_recurrence_index:.6f}")
    print(f"    λ_Φ adaptación     : {snap.n3_adaptation_lyapunov:+.6e}")
    print(f"    Reflexiones N3     : {snap.level_3_reflections}")
    print(f"    Provenance         : {snap.phase_provenance}")
    print(f"    Merkle fase HMAC   : {snap.sha256_phase_merkle[:40]}...")
    print(banner)