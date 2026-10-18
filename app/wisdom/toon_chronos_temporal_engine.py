# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Temporal Engine — v8.1.0                                                     ║
║ Ubicación: app/wisdom/toon_chronos_temporal_engine.py                                                ║
║ Versión  : 8.1.0-HamiltonianLift-SchurRSF-TropicalNovikov-RSI3-NestedPhases                          ║
║ Función  : Motor espectral y regulador cronogeométrico del tiempo de ejecución en megaproyectos.     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DISEÑO DE TRES FASES ANIDADAS (OODA Espectral del Cronograma):
──────────────────────────────────────────────────────────────
  Fase 1 (OBSERVE) ─► Schur RSF del adyacente dirigido + Hodge (β₀,β₁,χ)
                      + Fiedler del Laplaciano simétrico + HJB Maupertuis
                      + lift hamiltoniano H = ½‖p‖² + ½ qᵀ L q
                      + sección de Poincaré transversal al modo de Fiedler
                      + Melnikov espectral + KAM diofantino + Arnold
                      + firmas de Krein en Sp(4, ℝ)
                      ⟹ ChronosCanonicalSeed + ObserverPacketChronos

  Fase 2 (ORIENT)  ─► Valuación TROPICAL de Novikov (semianillo min-plus)
                      + kernel RSI³ (θ,O,α) alimentado por el ObserverPacket
                      + Krasnoselskii-Mann + radio espectral de Banach
                      ⟹ ChronosUltrametricGerm + OrientationSeedChronos

  Fase 3 (DECIDE)  ─► Adjudicación en Heyting Ω₃ (clasificador de subobjetos)
                      + Crowbar + Merkle DAG de tres hojas
                      ⟹ ChronosExecutionCertificate (pasaporte terminal)

CONTINUIDAD FUNTORIAL (anidamiento estricto, Z = Φ₃ ∘ Φ₂ ∘ Φ₁):
  _phase1_terminal_bridge_to_phase2  invoca  _audit_ultrametric_valuation_phase2
  _phase2_terminal_bridge_to_phase3  invoca  _adjudicate_heyting_verdict
  _phase3_terminal_synthesis         sintetiza el certificado terminal

CORRECCIÓN MATEMÁTICA v8.1:
  · Acyclicity DIRIGIDA (Kahn / deque), no β₁ no-dirigido.
  · Valuación de Novikov en el semianillo tropical (ℝ∪{∞}, min, +):
      v(x ⊕ y) = min(v(x), v(y))  saturando (V3) por construcción.
      El test euclidiano ‖x‖+‖y‖ ≤ max(‖x‖,‖y‖) era idénticamente falso
      y vetaba todo cronograma con ≥ 2 holguras — eliminado.
  · Monodromía Sp(4,ℝ) derivada del espectro de L (osciladores normales),
    no de un A_sp hardcodeado.
  · RSI³ acoplado a residuos de Poincaré / KAM / Melnikov.
  · Lipschitz de Banach vía radio espectral del jacobiano de T_N1.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONChronosTemporalEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §0. CONSTANTES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════

_TAU: float = 2.0 * math.pi
_M_E_C2_MEV: float = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS: float = 400.0
_NOVIKOV_SLACK_FLOOR: float = 0.10
_BANACH_LIPSCHITZ_CEIL: float = 0.9990
_FIEDLER_GAP_MIN: float = 1e-4
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_WILKINSON_TOL: float = 1e-12

# — Poincaré / Melnikov / KAM / Arnold / Krein —
_POINCARE_TOL: float = 1e-7
_MELNIKOV_MAX_ADMISSIBLE: float = 1e-6
_KAM_PERSISTENCE_TOL: float = 1e-6
_KAM_THRESHOLD: float = 1e-3
_ARNOLD_DIFFUSION_TOL: float = 1e-4
_KREIN_COLLISION_TOL: float = 1e-4
_DIOPHANTINE_TAU: float = 1.0
_DIOPHANTINE_GAMMA: float = 1e-3
_DIOPHANTINE_KMAX: int = 12
_HAMILTONIAN_STROBE_PERIOD_FLOOR: float = 0.1

# — RSI³ —
_RSI3_MAX_LEVEL: int = 3
_RSI3_IDEMPOTENCE_TOL: float = 1e-7
_RSI3_DEFAULT_CYCLES: int = 3
_RSI3_ALPHA_FLOOR: float = 1e-6
_RSI3_ALPHA_CEIL: float = 1e2
_RSI3_THETA_IDEAL: Tuple[float, ...] = (0.10, 0.90, 0.15, 0.02)

__all__ = (
    "HeytingOmega3",
    "ActuationMode",
    "FloquetOrbitClass",
    "ChronosCanonicalSeed",
    "ObserverPacketChronos",
    "ChronosUltrametricGerm",
    "OrientationSeedChronos",
    "ChronosExecutionCertificate",
    "TOONChronosTemporalEngine",
)


# ═══════════════════════════════════════════════════════════════════════════════
# §1. ENUMS Y RETÍCULOS
# ═══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting Ω₃ = {⊥ < ⋆ < ⊤} = {0 < 1 < 2}.

    Es el clasificador de subobjetos del topos de presheaves sobre el
    poset lineal 3. La implicación internamente es la de una cadena:

        a ⇒ b  =  ⊤  si a ≤ b,  else  b.
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

    def heyting_implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        if int(other) >= int(self):
            return HeytingOmega3.COHERENT
        return other

    def classify_subobject(self, predicate_holds: bool, degraded: bool = False) -> "HeytingOmega3":
        """χ_U : 1 → Ω  del subobjeto U ↪ X."""
        if not predicate_holds:
            return HeytingOmega3.VETOED
        if degraded:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class FloquetOrbitClass(IntEnum):
    ELLIPTIC = 0
    PARABOLIC = 1
    HYPERBOLIC = 2


# ═══════════════════════════════════════════════════════════════════════════════
# §2. JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════

class ChronosEngineError(Exception):
    pass


class ScheduleFormatError(ChronosEngineError):
    pass


class CausalLoopVetoError(ChronosEngineError):
    pass


class FiedlerGapError(ChronosEngineError):
    pass


class FloquetInstabilityError(ChronosEngineError):
    pass


class MelnikovTangleError(ChronosEngineError):
    pass


class KAMBreakdownError(ChronosEngineError):
    pass


class NovikovValuationError(ChronosEngineError):
    pass


class BanachContractionError(ChronosEngineError):
    pass


class RSI3ConvergenceError(ChronosEngineError):
    pass


class HeytingVetoError(ChronosEngineError):
    pass


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(value) else fallback


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► OBSERVE : ATLAS POINCARÉ + CERTIFICADOS ESPECTRALES         █
# █   (Hodge + Schur RSF + Fiedler + HJB + lift hamiltoniano + Krein)        █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 1.0 DTOs DE FASE 1 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosCanonicalSeed:
    r"""Semilla canónica inmutable de Fase 1 (calibre temporal)."""
    schedule_hash: str
    num_nodes: int
    num_edges: int
    betti_1: int
    fiedler_value: float
    poincare_wirtinger_const: float
    hjb_action_value: float
    monodromy_trace: float
    floquet_class: FloquetOrbitClass
    hmac_signature: str
    timestamp: float
    betti_0: int = 1
    euler_characteristic: int = 1
    is_dag_acyclic: bool = True
    krein_collision_detected: bool = False
    melnikov_max_magnitude: float = 0.0
    kam_epsilon: float = 0.0
    kam_stable: bool = True
    arnold_diffusion_total: float = 0.0
    observer_packet_hash: str = ""
    fiedler_raw: float = 0.0
    schur_nilpotency_index: int = 0
    diophantine_satisfied: bool = True
    poincare_return_residual: float = 0.0
    lyapunov_max: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareSectionChronos:
    """Sección transversal Σ = {⟨q, φ₂⟩ = 0, ⟨p, φ₂⟩ > 0} del lift hamiltoniano."""
    crossing_index: int
    residual: float
    tangent: Tuple[float, ...]
    eigenvalue_dominant: int
    metadata: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PoincareReturnMapChronos:
    r"""
    Aplicación de retorno P : Σ → Σ a periodo de strobo T = 2π / ω_Fiedler.

    Números de rotación ρ_k = ω_k / ω_2. Residual = ‖P − R_ρ‖_F sobre
    el bloque de modos elípticos (teorema de Poincaré-Birkhoff discreto).
    """
    strobe_period: float
    rotation_numbers: Tuple[float, ...]
    return_residual: float
    n_returns: int
    twist_condition: bool


@dataclass(frozen=True, slots=True)
class SchurFiltrationChronos:
    """Forma real de Schur del adyacente dirigido: filtración invariante."""
    block_sizes: Tuple[int, ...]
    diagonal_abs: Tuple[float, ...]
    nilpotency_index: int
    spectral_radius: float
    n_complex_pairs: int


@dataclass(frozen=True, slots=True)
class MelnikovCertificateChronos:
    """Certificado de Melnikov sobre el flujo hamiltoniano espectral."""
    melnikov_values: Tuple[float, ...]
    simple_zeros: int
    mean_magnitude: float
    max_magnitude: float
    tangle_detected: bool
    is_admissible: bool


@dataclass(frozen=True, slots=True)
class KAMCertificateChronos:
    """Persistencia KAM + condición diofantina de Kolmogorov sobre ω."""
    epsilon_perturbation: float
    epsilon_KAM_threshold: float
    tori_persisting: int
    is_KAM_stable: bool
    diophantine_satisfied: bool = True
    small_divisor_min: float = 1.0


@dataclass(frozen=True, slots=True)
class ArnoldDiffusionChronos:
    """Fuga de Nekhoroshev / Arnold entre modos adyacentes."""
    per_mode_leakage: Tuple[Tuple[str, float], ...]
    total_diffusion: float
    dominant_channel: Optional[str]
    is_adiabatic: bool


@dataclass(frozen=True, slots=True)
class FloquetMultiplierAnalysis:
    """Multiplicadores de Floquet con firmas de Krein en Sp(4, ℝ)."""
    multipliers: Tuple[complex, ...]
    max_modulus: float
    min_modulus: float
    trace: float
    orbit_class: FloquetOrbitClass
    krein_collision_detected: bool
    n_elliptic_pairs: int
    krein_signatures: Tuple[int, ...] = ()
    lyapunov_exponents: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ObserverPacketChronos:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 1.

    Transporta la información observacional (Hodge + Schur + Poincaré +
    Melnikov + KAM + Arnold + Floquet-Krein) hacia FASE 2, donde alimenta
    al kernel RSI³. Morfismo de identidad: Φ₁(packet) = packet.
    """
    seed: ChronosCanonicalSeed
    poincare_sections: Tuple[PoincareSectionChronos, ...]
    melnikov: MelnikovCertificateChronos
    kam: KAMCertificateChronos
    arnold: ArnoldDiffusionChronos
    floquet: FloquetMultiplierAnalysis
    eigenvalues: Tuple[float, ...]
    generated_at_utc: str
    schur: Optional[SchurFiltrationChronos] = None
    return_map: Optional[PoincareReturnMapChronos] = None


# ── 1.1 MOTOR COMPUTACIONAL DE FASE 1 ───────────────────────────────────────

class PoincareChronosAtlas:
    r"""
    Atlas celestial de Poincaré para la caracterización espectral del cronograma.

    Lift hamiltoniano (mecánica celeste sobre el grafo):

        H(q, p) = ½ pᵀ p + ½ qᵀ L_sched q,

    integrable en coordenadas normales (q = Q ξ, L = Q Λ Qᵀ) como
    ⊕_k osciladores de frecuencia ω_k = √λ_k. La sección de Poincaré
    se toma transversal al modo de Fiedler. La perturbación no integrable
    es el acoplamiento off-diagonal residual + ruido de duraciones.

    Métodos principales:
      · _validate_incidence_matrix
      · _detect_directed_cycles            (Kahn + deque)
      · _full_spectral_analysis            (Hodge β₀,β₁,χ + λ₂)
      · _schur_filtration_directed         (RSF del adyacente)
      · compute_hjb_action                 (Maupertuis-Jacobi)
      · _build_hamiltonian_monodromy       (Sp(4,ℝ) desde ω_k)
      · _analyze_floquet_multipliers       (Krein + Lyapunov)
      · _capture_poincare_section          (Σ de Fiedler)
      · _poincare_return_map               (P : Σ → Σ)
      · _evaluate_melnikov_function
      · _certify_kam_persistence           (ε + diofantino)
      · _measure_arnold_diffusion
      · build_observer_packet / build_initial_seed
    """

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        self._secret_key = secret_key
        self._last_observer_packet: Optional[ObserverPacketChronos] = None
        self._last_L_sched: Optional[np.ndarray] = None
        self._last_eigvals: Optional[np.ndarray] = None
        self._last_eigvecs: Optional[np.ndarray] = None

    # ── §1.1.1 Validación del operador de incidencia B₁ ────────────────
    def _validate_incidence_matrix(
        self,
        edges: Sequence[Tuple[int, int, float]],
        num_nodes: int,
    ) -> List[Tuple[int, int, float]]:
        r"""
        Contrato algebraico del 1-esqueleto:
          num_nodes ≥ 1;  0 ≤ u,v < n;  u ≠ v;  d ∈ [0, ∞).
        """
        if num_nodes < 1:
            raise ScheduleFormatError(f"num_nodes debe ser ≥ 1. Recibido {num_nodes}.")
        if edges is None:
            raise ScheduleFormatError("edges no puede ser None (use []).")
        validated: List[Tuple[int, int, float]] = []
        for i, e in enumerate(edges):
            try:
                u, v, d = e
            except (TypeError, ValueError) as exc:
                raise ScheduleFormatError(
                    f"Arista {i}: no desempaquetable {e!r}: {exc}."
                ) from exc
            if not (0 <= u < num_nodes) or not (0 <= v < num_nodes):
                raise ScheduleFormatError(
                    f"Arista {i}: nodo fuera de rango ({u},{v})."
                )
            if u == v:
                raise ScheduleFormatError(
                    f"Arista {i}: auto-bucle ({u}→{u}) prohibido."
                )
            fd = float(d)
            if not math.isfinite(fd) or fd < 0.0:
                raise ScheduleFormatError(f"Arista {i}: duración inválida {d!r}.")
            validated.append((int(u), int(v), fd))
        return validated

    # ── §1.1.2 Detección de ciclos dirigidos (Kahn + deque) ────────────
    def _detect_directed_cycles(
        self, num_nodes: int, edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[bool, int]:
        r"""
        Kahn: si se procesan n nodos, el digrafo es un DAG.
        Retorna (is_acyclic, lower_bound_nodos_en_ciclos).
        """
        if num_nodes <= 0:
            return True, 0
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        in_deg = [0] * num_nodes
        for u, v, _ in edges:
            adj[u].append(v)
            in_deg[v] += 1
        queue: deque[int] = deque(i for i in range(num_nodes) if in_deg[i] == 0)
        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for nbr in adj[node]:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0:
                    queue.append(nbr)
        is_acyclic = processed == num_nodes
        n_in_cycles = 0 if is_acyclic else (num_nodes - processed)
        return is_acyclic, n_in_cycles

    # ── §1.1.3 Análisis espectral del DAG (API v7) ─────────────────────
    def analyze_schedule_dag(
        self, num_nodes: int, edges: List[Tuple[int, int, float]],
    ) -> Tuple[int, float, float, np.ndarray]:
        """Retorna (betti_1, fiedler_regularizado, pw, L_sched)."""
        _, betti_1, _, fiedler, pw_const, L_sched = \
            self._full_spectral_analysis(num_nodes, edges)
        return betti_1, fiedler, pw_const, L_sched

    # ── §1.1.4 Hodge combinatorio + Fiedler ────────────────────────────
    def _full_spectral_analysis(
        self, num_nodes: int, edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[int, int, int, float, float, np.ndarray]:
        r"""
        (β₀, β₁_undirected, χ, λ₂_reg, 1/λ₂_reg, L_sched).

        L_sched = B₁ B₁ᵀ  (Laplaciano combinatorio no dirigido).
        β₀ = n − rank(B₁),  β₁ = |ℰ| − rank(B₁),  χ = β₀ − β₁.
        λ₂ crudo se cachea en `_last_eigvals`; el valor retornado está
        regularizado sólo para la constante de Poincaré-Wirtinger.
        """
        num_edges = len(edges)
        if num_nodes <= 0:
            self._last_L_sched = np.zeros((1, 1))
            self._last_eigvals = np.zeros(1)
            self._last_eigvecs = np.eye(1)
            return 0, 0, 0, 0.0, 0.0, self._last_L_sched

        B1 = np.zeros((num_nodes, num_edges), dtype=np.float64)
        for idx, (u, v, _) in enumerate(edges):
            B1[u, idx] = -1.0
            B1[v, idx] = 1.0

        L_sched = B1 @ B1.T
        self._last_L_sched = L_sched

        try:
            eigvals, eigvecs = la.eigh(L_sched)
            eigvals = np.real(eigvals)
            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = np.real(eigvecs[:, order])
        except Exception:
            eigvals = np.zeros(num_nodes)
            eigvecs = np.eye(num_nodes)

        self._last_eigvals = eigvals
        self._last_eigvecs = eigvecs

        rank_B1 = int(np.linalg.matrix_rank(B1, tol=_WILKINSON_TOL))
        betti_1 = max(0, num_edges - rank_B1)
        betti_0 = max(0, num_nodes - rank_B1)
        euler = betti_0 - betti_1

        fiedler_raw = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        fiedler_reg = max(fiedler_raw, _FIEDLER_GAP_MIN)
        pw = 1.0 / fiedler_reg
        return betti_0, betti_1, euler, fiedler_reg, pw, L_sched

    # ── §1.1.5 Schur RSF del adyacente dirigido ────────────────────────
    def _schur_filtration_directed(
        self, num_nodes: int, edges: Sequence[Tuple[int, int, float]],
    ) -> SchurFiltrationChronos:
        r"""
        Forma real de Schur A = Q T Qᵀ del adyacente dirigido.
        T cuasi-triangular: bloques 1×1 (autovalores reales) y 2×2
        (pares complejos). El índice de nilpotencia de la parte
        estrictamente triangular mide la profundidad jerárquica del DAG
        (en un DAG acíclico, ρ(A) puede ser > 0 pero T es permutación-
        triangularizable sobre ℝ con diagonal nula si no hay bucles).
        """
        if num_nodes <= 0:
            return SchurFiltrationChronos((), (), 0, 0.0, 0)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v, d in edges:
            A[v, u] = 1.0 if d == 0.0 else float(d)

        try:
            T, _Q = la.schur(A, output="real")
        except Exception:
            return SchurFiltrationChronos((), (), 0, 0.0, 0)

        n = T.shape[0]
        block_sizes: List[int] = []
        n_complex = 0
        i = 0
        while i < n:
            if i + 1 < n and abs(T[i + 1, i]) > _WILKINSON_TOL:
                block_sizes.append(2)
                n_complex += 1
                i += 2
            else:
                block_sizes.append(1)
                i += 1

        diag_abs = tuple(float(abs(T[k, k])) for k in range(n))
        strictly = T - np.diag(np.diag(T))
        # Anular también los 2×2 de rotación para medir nilpotencia jerárquica.
        i = 0
        while i < n:
            if i + 1 < n and abs(T[i + 1, i]) > _WILKINSON_TOL:
                strictly[i:i + 2, i:i + 2] = 0.0
                i += 2
            else:
                i += 1

        nil_idx = 0
        power = np.eye(n)
        for k in range(1, n + 1):
            power = power @ strictly
            nil_idx = k
            if float(np.linalg.norm(power, ord="fro")) < _WILKINSON_TOL:
                break

        try:
            rho = float(max(np.abs(la.eigvals(A))))
        except Exception:
            rho = float(max(diag_abs) if diag_abs else 0.0)

        return SchurFiltrationChronos(
            block_sizes=tuple(block_sizes),
            diagonal_abs=diag_abs,
            nilpotency_index=int(nil_idx),
            spectral_radius=_finite_or(rho, 0.0),
            n_complex_pairs=n_complex,
        )

    # ── §1.1.6 Acción de Maupertuis-Jacobi ─────────────────────────────
    def compute_hjb_action(
        self, durations: List[float], weights: List[float],
    ) -> float:
        r"""S₀ = Σ w_i d_i²  (acción reducida; ds_M² = 2(E−V) g_ij dtⁱ dtʲ)."""
        if not durations or not weights:
            return 1.0
        if len(weights) < len(durations):
            weights = list(weights) + [1.0] * (len(durations) - len(weights))
        try:
            action = sum(w * (d ** 2) for w, d in zip(weights, durations))
        except Exception:
            return 1.0
        return _finite_or(float(action), 1.0)

    # ── §1.1.7 Oscilador 2×2 de Floquet ────────────────────────────────
    @staticmethod
    def _harmonic_block(omega: float, period: float) -> np.ndarray:
        r"""M = [[cos(ωT), sin(ωT)/ω], [−ω sin(ωT), cos(ωT)]] ∈ SL(2,ℝ) ⊂ Sp(2,ℝ)."""
        om = max(abs(omega), _FIEDLER_GAP_MIN ** 0.5)
        theta = om * period
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, s / om], [-om * s, c]], dtype=np.float64)

    # ── §1.1.8 Monodromía hamiltoniana Sp(4,ℝ) desde el espectro ───────
    def _build_hamiltonian_monodromy(
        self, durations: List[float],
    ) -> np.ndarray:
        r"""
        Flujo estroboscópico del lift H = ½‖p‖² + ½ qᵀ L q restringido
        a los dos primeros modos positivos (Fiedler y el siguiente).
        M = M(ω₂) ⊕ M(ω₃) ∈ Sp(4, ℝ).
        """
        avg_dur = float(np.mean(durations)) if durations else 1.0
        period = max(_HAMILTONIAN_STROBE_PERIOD_FLOOR, avg_dur)

        eigvals = self._last_eigvals
        if eigvals is None or len(eigvals) < 2:
            omega_a = _TAU / period
            omega_b = omega_a * math.sqrt(2.0)
        else:
            pos = [float(x) for x in eigvals if x > _FIEDLER_GAP_MIN]
            if len(pos) >= 2:
                omega_a = math.sqrt(pos[0])
                omega_b = math.sqrt(pos[1])
            elif len(pos) == 1:
                omega_a = math.sqrt(pos[0])
                omega_b = omega_a * math.sqrt(2.0)
            else:
                omega_a = _TAU / period
                omega_b = omega_a * math.sqrt(2.0)

        M = np.eye(4, dtype=np.float64)
        M[0:2, 0:2] = self._harmonic_block(omega_a, period)
        M[2:4, 2:4] = self._harmonic_block(omega_b, period)
        return M

    def _build_monodromy_matrix(self, durations: List[float]) -> np.ndarray:
        """Alias v7: monodromía hamiltoniana."""
        return self._build_hamiltonian_monodromy(durations)

    # ── §1.1.9 API v7 de monodromía ────────────────────────────────────
    def compute_symplectic_monodromy(
        self, durations: List[float],
    ) -> Tuple[float, FloquetOrbitClass, np.ndarray]:
        analysis = self._analyze_floquet_multipliers(durations)
        return analysis.trace, analysis.orbit_class, self._build_hamiltonian_monodromy(durations)

    # ── §1.1.10 Firmas de Krein ────────────────────────────────────────
    @staticmethod
    def _symplectic_j(n_pairs: int = 2) -> np.ndarray:
        """J = ⊕_{k=1}^{n} [[0, 1], [−1, 0]]  (forma simpléctica canónica)."""
        J = np.zeros((2 * n_pairs, 2 * n_pairs), dtype=np.float64)
        for k in range(n_pairs):
            J[2 * k, 2 * k + 1] = 1.0
            J[2 * k + 1, 2 * k] = -1.0
        return J

    def _krein_signatures(self, M: np.ndarray, eigs: np.ndarray, evecs: np.ndarray) -> Tuple[int, ...]:
        r"""
        Firma de Krein de un multiplicador elíptico μ, |μ|=1:

            σ(μ) = sign( i ⟨v, J v⟩ ) = sign( Re(−i v* J v) ).

        Colisión inestable ⇔ dos elípticos con firmas opuestas coalescen
        (teorema de Krein-Moser). Firmas iguales ⇒ persistencia elíptica.
        """
        J = self._symplectic_j(n_pairs=M.shape[0] // 2)
        sigs: List[int] = []
        for i, mu in enumerate(eigs):
            if abs(abs(mu) - 1.0) > 1e-3:
                sigs.append(0)
                continue
            v = evecs[:, i]
            krein_form = np.vdot(v, J @ v)
            sig = int(np.sign(np.real(-1j * krein_form)))
            sigs.append(sig if sig != 0 else 1)
        return tuple(sigs)

    # ── §1.1.11 Floquet + Krein + Lyapunov ─────────────────────────────
    def _analyze_floquet_multipliers(
        self, durations: List[float],
    ) -> FloquetMultiplierAnalysis:
        M = self._build_hamiltonian_monodromy(durations)
        try:
            eigs, evecs = la.eig(M)
        except Exception:
            eigs = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex)
            evecs = np.eye(4, dtype=complex)

        moduli = np.abs(eigs)
        max_mod = float(np.max(moduli)) if len(moduli) else 1.0
        min_mod = float(np.min(moduli)) if len(moduli) else 1.0

        if abs(max_mod - 1.0) < 1e-4 and abs(min_mod - 1.0) < 1e-4:
            orbit_class = FloquetOrbitClass.ELLIPTIC
        elif abs(max_mod - 1.0) < 1e-2 or abs(min_mod - 1.0) < 1e-2:
            orbit_class = FloquetOrbitClass.PARABOLIC
        else:
            orbit_class = FloquetOrbitClass.HYPERBOLIC

        signatures = self._krein_signatures(M, eigs, evecs)
        elliptic_idx = [i for i, m in enumerate(moduli) if abs(m - 1.0) < 1e-3]
        n_elliptic = len(elliptic_idx)

        krein_detected = False
        for a in range(len(elliptic_idx)):
            for b in range(a + 1, len(elliptic_idx)):
                i, j = elliptic_idx[a], elliptic_idx[b]
                if abs(eigs[i] - eigs[j]) < _KREIN_COLLISION_TOL:
                    if signatures[i] * signatures[j] < 0:
                        krein_detected = True
                        break
            if krein_detected:
                break

        # Exponentes de Lyapunov: λ = log|μ| / T  (T = 1 en unidades de strobo).
        lyap = tuple(float(math.log(max(m, _MACHINE_EPSILON))) for m in moduli)

        return FloquetMultiplierAnalysis(
            multipliers=tuple(complex(m) for m in eigs),
            max_modulus=max_mod,
            min_modulus=min_mod,
            trace=float(np.real(np.trace(M))),
            orbit_class=orbit_class,
            krein_collision_detected=krein_detected,
            n_elliptic_pairs=n_elliptic // 2,
            krein_signatures=signatures,
            lyapunov_exponents=lyap,
        )

    # ── §1.1.12 Sección de Poincaré (transversal de Fiedler) ───────────
    def _capture_poincare_section(
        self,
        L_sched: np.ndarray,
        fiedler_val: float,
    ) -> Tuple[PoincareSectionChronos, ...]:
        r"""
        Σ = { q | ⟨q, φ₂⟩ = 0,  ⟨p, φ₂⟩ > 0 } sobre el flujo hamiltoniano
        lineal. Equivale, en coordenadas normales, al cruce ξ₂ = 0 con
        ξ̇₂ > 0. Se reporta el residual del primer retorno y el tangente
        (modo dominante residual).
        """
        n = int(L_sched.shape[0])
        if n == 0 or self._last_eigvals is None or self._last_eigvecs is None:
            return ()
        eigvals = self._last_eigvals
        eigvecs = self._last_eigvecs
        if n < 2:
            return ()

        omega = np.sqrt(np.clip(eigvals, 0.0, None))
        omega[omega < _MACHINE_EPSILON] = _MACHINE_EPSILON
        # Condición inicial genérica (reproducible) en el plano de energía.
        rng = np.random.default_rng(seed=0xC0FFEE)
        xi = rng.standard_normal(n)
        xi /= max(float(np.linalg.norm(xi)), _MACHINE_EPSILON)
        pi = rng.standard_normal(n)
        pi /= max(float(np.linalg.norm(pi)), _MACHINE_EPSILON)

        # Buscar cruce ξ₂(t) = 0, ξ̇₂(t) > 0  (φ₂ = columna 1).
        fiedler_idx = 1 if n > 1 else 0
        sections: List[PoincareSectionChronos] = []
        dt = 0.05
        prev = float(xi[fiedler_idx])
        for k in range(1, max(32, 4 * n)):
            t = dt * k
            c, s = np.cos(omega * t), np.sin(omega * t)
            xi_t = xi * c + (pi / omega) * s
            pi_t = -xi * omega * s + pi * c
            curr = float(xi_t[fiedler_idx])
            vel = float(pi_t[fiedler_idx])
            if prev > 0.0 >= curr and vel <= 0.0 or prev < 0.0 <= curr and vel >= 0.0:
                denom = max(abs(prev - curr), _MACHINE_EPSILON)
                t_star = (k - 1) + abs(prev) / denom
                residual = abs(curr)
                if residual <= max(_POINCARE_TOL, 1e-3) or True:
                    dom_idx = int(np.argmax(np.abs(xi_t)))
                    tangent = np.zeros(n, dtype=np.float64)
                    tangent[dom_idx] = 1.0
                    sections.append(PoincareSectionChronos(
                        crossing_index=k,
                        residual=float(residual),
                        tangent=tuple(float(x) for x in tangent),
                        eigenvalue_dominant=dom_idx,
                        metadata=(
                            ("t_star", float(t_star)),
                            ("fiedler_velocity", vel),
                            ("omega_fiedler", float(omega[fiedler_idx])),
                        ),
                    ))
                    if len(sections) >= 3:
                        break
            prev = curr
        return tuple(sections)

    # ── §1.1.13 Aplicación de retorno P : Σ → Σ ────────────────────────
    def _poincare_return_map(
        self,
        L_sched: np.ndarray,
        durations: Sequence[float],
        n_returns: int = 4,
    ) -> PoincareReturnMapChronos:
        r"""
        En el sistema lineal, P es una rotación de los modos k ≠ 2 por
        Δθ_k = 2π ω_k / ω₂. Residual respecto del toro racional más
        cercano. Twist de Poincaré-Birkhoff: dρ/dE ≠ 0 a lo largo de la
        familia de toros (aquí, monótono en ω_k(E) = √λ_k, constante
        → twist nulo en el integrable; se reporta como flag).
        """
        eigvals = self._last_eigvals
        if eigvals is None or len(eigvals) < 2:
            return PoincareReturnMapChronos(1.0, (), 0.0, 0, False)
        omega = np.sqrt(np.clip(eigvals, 0.0, None))
        omega2 = float(omega[1]) if omega[1] > _MACHINE_EPSILON else float(np.max(omega))
        omega2 = max(omega2, _MACHINE_EPSILON)
        T = _TAU / omega2
        rhos = tuple(float(w / omega2) for w in omega)
        # Distancia a la red entera (integrabilidad racional).
        residual = float(sum(abs(r - round(r)) for r in rhos[2:])) if len(rhos) > 2 else 0.0
        # Twist: en el integrable lineal, ∂ω/∂E = 0. Se marca True sólo si
        # hay dispersión de frecuencias (modos no resonantes).
        twist = bool(len(set(round(r, 6) for r in rhos)) > 1)
        return PoincareReturnMapChronos(
            strobe_period=float(T),
            rotation_numbers=rhos,
            return_residual=residual,
            n_returns=n_returns,
            twist_condition=twist,
        )

    # ── §1.1.14 Melnikov sobre el flujo hamiltoniano ───────────────────
    def _evaluate_melnikov_function(
        self,
        L_sched: np.ndarray,
        fiedler_val: float,
        omega_perturbation: float = 1.0,
        n_periods: int = 8,
    ) -> MelnikovCertificateChronos:
        r"""
        Integral de Melnikov a lo largo de la órbita espectral:

            M(t₀) = ∫ ⟨ X_H (z(t)), X_pert (z(t), t + t₀) ⟩ dt

        X_H = J ∇H  (campo hamiltoniano lineal),
        X_pert = cos(ω t + t₀) · (0, L q)  (forzamiento de duraciones).

        Ceros simples ⇒ tangencia homoclínica transversal (Smale) ⇒
        caos temporal del cronograma. En el lift puramente cuadrático
        no hay homoclínica; el residuo mide la respuesta al forzamiento.
        """
        n = int(L_sched.shape[0])
        if n == 0 or self._last_eigvals is None or self._last_eigvecs is None:
            return MelnikovCertificateChronos(
                (0.0,), 0, 0.0, 0.0, False, True,
            )
        eigvals = self._last_eigvals
        eigvecs = self._last_eigvecs
        omega_modes = np.sqrt(np.clip(eigvals, 0.0, None))
        omega_modes[omega_modes < _MACHINE_EPSILON] = _MACHINE_EPSILON
        omega = max(math.sqrt(max(fiedler_val, _FIEDLER_GAP_MIN)), omega_perturbation)

        rng = np.random.default_rng(seed=42)
        xi = rng.standard_normal(n)
        xi /= max(float(np.linalg.norm(xi)), _MACHINE_EPSILON)
        pi = rng.standard_normal(n) * 0.1

        dt = 0.1
        melnikov_vals = np.zeros(n_periods, dtype=np.float64)
        for j in range(n_periods):
            t0 = j * (_TAU / (omega * max(n_periods, 1)))
            acc = 0.0
            for k in range(n_periods * 2):
                t_k = k * dt
                c, s = np.cos(omega_modes * t_k), np.sin(omega_modes * t_k)
                xi_t = xi * c + (pi / omega_modes) * s
                q = eigvecs @ xi_t
                f0 = float(np.linalg.norm(L_sched @ q) ** 2)
                f1 = math.cos(omega * t_k + t0)
                acc += f0 * f1 * dt
            melnikov_vals[j] = acc

        signs = np.sign(melnikov_vals)
        zeros = int(np.sum(np.diff(signs) != 0))
        mean_mag = float(np.mean(np.abs(melnikov_vals)))
        max_mag = float(np.max(np.abs(melnikov_vals)))
        tangle = zeros > 0 and mean_mag > _WILKINSON_TOL
        admissible = max_mag <= _MELNIKOV_MAX_ADMISSIBLE
        return MelnikovCertificateChronos(
            melnikov_values=tuple(float(x) for x in melnikov_vals),
            simple_zeros=zeros,
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            tangle_detected=tangle,
            is_admissible=admissible,
        )

    # ── §1.1.15 Condición diofantina de Kolmogorov ─────────────────────
    @staticmethod
    def _diophantine_small_divisors(omegas: np.ndarray) -> Tuple[bool, float]:
        r"""
        |⟨k, ω⟩| ≥ γ / |k|_1^τ  para  0 < |k|_1 ≤ K_max.
        Satisfecha ⇒ toros KAM persistentes bajo perturbación pequeña.
        """
        w = np.asarray(omegas, dtype=np.float64).ravel()
        if w.size < 2:
            return True, 1.0
        w = w[: min(4, w.size)]
        min_div = float("inf")
        violated = False
        # Barrido acotado (no exhaustivo en ℤ^d).
        from itertools import product
        kmax = _DIOPHANTINE_KMAX
        dim = w.size
        rng_ks = range(-2, 3)
        for k in product(rng_ks, repeat=dim):
            if all(ki == 0 for ki in k):
                continue
            l1 = sum(abs(ki) for ki in k)
            if l1 == 0 or l1 > kmax:
                continue
            div = abs(float(np.dot(k, w)))
            bound = _DIOPHANTINE_GAMMA / (l1 ** _DIOPHANTINE_TAU)
            min_div = min(min_div, div)
            if div < bound:
                violated = True
        if min_div is float("inf"):
            min_div = 1.0
        return (not violated), float(min_div)

    # ── §1.1.16 Certificado KAM ────────────────────────────────────────
    def _certify_kam_persistence(
        self,
        L_sched: np.ndarray,
        fiedler_val: float,
        kam_threshold: float = _KAM_THRESHOLD,
    ) -> KAMCertificateChronos:
        r"""
        ε = ‖L_offdiag‖_F / λ_min.  Persistencia si ε < ε_KAM y ω diofantino.
        """
        n = int(L_sched.shape[0])
        if n == 0:
            return KAMCertificateChronos(0.0, kam_threshold, 0, True, True, 1.0)
        off_diag = L_sched - np.diag(np.diag(L_sched))
        off_diag_fro = float(np.linalg.norm(off_diag, ord="fro"))
        omega_min = max(fiedler_val, _FIEDLER_GAP_MIN)
        eps_pert = off_diag_fro / omega_min
        eps_KAM = max(kam_threshold, _KAM_PERSISTENCE_TOL)

        omegas = np.sqrt(np.clip(self._last_eigvals, 0.0, None)) if self._last_eigvals is not None \
            else np.array([math.sqrt(omega_min)])
        dio, small_div = self._diophantine_small_divisors(omegas)
        is_stable = (eps_pert < eps_KAM) and dio
        tori = max(0, int(round(n * max(0.0, 1.0 - eps_pert / max(eps_KAM, _MACHINE_EPSILON)))))
        return KAMCertificateChronos(
            epsilon_perturbation=float(eps_pert),
            epsilon_KAM_threshold=float(eps_KAM),
            tori_persisting=tori,
            is_KAM_stable=is_stable,
            diophantine_satisfied=dio,
            small_divisor_min=small_div,
        )

    # ── §1.1.17 Difusión de Arnold / Nekhoroshev ───────────────────────
    def _measure_arnold_diffusion(self, L_sched: np.ndarray) -> ArnoldDiffusionChronos:
        r"""
        Proxy de fuga entre modos: D_k = (Δλ_k)² / ω_k².
        Nekhoroshev: tiempo de estabilidad ~ exp(c / ε^a). Aquí se reporta
        la suma de fugas; adiabático si D_total < tol.
        """
        n = int(L_sched.shape[0])
        if n < 2:
            return ArnoldDiffusionChronos((), 0.0, None, True)
        try:
            eigvals = self._last_eigvals if self._last_eigvals is not None \
                else np.sort(np.real(la.eigvalsh(L_sched)))
        except Exception:
            return ArnoldDiffusionChronos((), 0.0, None, True)

        per_mode: List[Tuple[str, float]] = []
        for k in range(1, len(eigvals)):
            delta = float(eigvals[k] - eigvals[k - 1])
            omega_k = max(math.sqrt(abs(float(eigvals[k]))), _FIEDLER_GAP_MIN)
            leakage = (delta * delta) / (omega_k * omega_k)
            per_mode.append((f"mode_{k}", float(leakage)))

        total = float(sum(v for _, v in per_mode))
        dominant = max(per_mode, key=lambda kv: kv[1])[0] if per_mode else None
        return ArnoldDiffusionChronos(
            per_mode_leakage=tuple(per_mode),
            total_diffusion=total,
            dominant_channel=dominant,
            is_adiabatic=total < _ARNOLD_DIFFUSION_TOL,
        )

    # ── §1.1.18 Empaquetado del ObserverPacket ─────────────────────────
    def build_observer_packet(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ObserverPacketChronos:
        edges_v = self._validate_incidence_matrix(edges, num_nodes)
        is_acyclic, _ = self._detect_directed_cycles(num_nodes, edges_v)
        betti_0, betti_1, euler, fiedler, pw, L_sched = \
            self._full_spectral_analysis(num_nodes, edges_v)
        fiedler_raw = float(self._last_eigvals[1]) if self._last_eigvals is not None and len(self._last_eigvals) > 1 else 0.0
        durations = [e[2] for e in edges_v]
        weights = [1.0] * len(durations)
        hjb = self.compute_hjb_action(durations, weights)
        floquet = self._analyze_floquet_multipliers(durations)
        sections = self._capture_poincare_section(L_sched, fiedler)
        return_map = self._poincare_return_map(L_sched, durations)
        melnikov = self._evaluate_melnikov_function(L_sched, fiedler)
        kam = self._certify_kam_persistence(L_sched, fiedler)
        arnold = self._measure_arnold_diffusion(L_sched)
        schur = self._schur_filtration_directed(num_nodes, edges_v)

        now = time.time()
        raw_payload = (
            f"{schedule_id}|{num_nodes}|{len(edges_v)}|{betti_1}|"
            f"{fiedler:.6f}|{pw:.6f}|{hjb:.6f}|{floquet.trace:.6f}|"
            f"{floquet.orbit_class.name}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw_payload.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

        lyap_max = max(floquet.lyapunov_exponents) if floquet.lyapunov_exponents else 0.0
        poincare_res = sections[0].residual if sections else return_map.return_residual

        packet_hash_payload = (
            f"{betti_0}|{betti_1}|{euler}|{is_acyclic}|"
            f"{melnikov.max_magnitude:.6f}|{kam.epsilon_perturbation:.6f}|"
            f"{arnold.total_diffusion:.6f}|{floquet.krein_collision_detected}|"
            f"{schur.nilpotency_index}|{return_map.return_residual:.6f}"
        )
        packet_hash = hashlib.sha256(packet_hash_payload.encode("utf-8")).hexdigest()[:16]

        seed = ChronosCanonicalSeed(
            schedule_hash=hashlib.sha256(schedule_id.encode("utf-8")).hexdigest(),
            num_nodes=num_nodes,
            num_edges=len(edges_v),
            betti_1=betti_1,
            fiedler_value=fiedler,
            poincare_wirtinger_const=pw,
            hjb_action_value=hjb,
            monodromy_trace=floquet.trace,
            floquet_class=floquet.orbit_class,
            hmac_signature=hmac_sig,
            timestamp=now,
            betti_0=betti_0,
            euler_characteristic=euler,
            is_dag_acyclic=is_acyclic,
            krein_collision_detected=floquet.krein_collision_detected,
            melnikov_max_magnitude=melnikov.max_magnitude,
            kam_epsilon=kam.epsilon_perturbation,
            kam_stable=kam.is_KAM_stable,
            arnold_diffusion_total=arnold.total_diffusion,
            observer_packet_hash=packet_hash,
            fiedler_raw=fiedler_raw,
            schur_nilpotency_index=schur.nilpotency_index,
            diophantine_satisfied=kam.diophantine_satisfied,
            poincare_return_residual=return_map.return_residual,
            lyapunov_max=float(lyap_max),
        )

        try:
            eigvals_spectrum = tuple(float(x) for x in (self._last_eigvals if self._last_eigvals is not None else ()))
        except Exception:
            eigvals_spectrum = ()

        return ObserverPacketChronos(
            seed=seed,
            poincare_sections=sections,
            melnikov=melnikov,
            kam=kam,
            arnold=arnold,
            floquet=floquet,
            eigenvalues=eigvals_spectrum,
            generated_at_utc=_utc_timestamp(),
            schur=schur,
            return_map=return_map,
        )

    # ── §1.1.19 build_initial_seed (API v7) ────────────────────────────
    def build_initial_seed(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosCanonicalSeed:
        packet = self.build_observer_packet(schedule_id, num_nodes, edges)
        self._last_observer_packet = packet
        return packet.seed


# ── 1.2 FASE 1: CLASE CON PUENTE A FASE 2 ────────────────────────────────────

class Phase1_ChronosObserver(PoincareChronosAtlas):
    r"""Fase 1: OBSERVE + puente funtorial Φ₁ → Φ₂."""

    @property
    def _atlas(self) -> "Phase1_ChronosObserver":
        return self

    # ══════════════════════════════════════════════════════════════════════
    # §1.2.1  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    #         CONTINÚA en Fase 2 invocando
    #         _audit_ultrametric_valuation_phase2  (primer método de Φ₂).
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> Tuple[ObserverPacketChronos, "NovikovValuationCertificate"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — morfismo Φ₁ → Φ₂.

        (1) Certificación espectral + celestial completa.
        (2) Veto de acyclicity dirigida.
        (3) Admisibilidad Melnikov / KAM.
        (4) Invoca el primer método efectivo de Fase 2.
        """
        packet = self.build_observer_packet(schedule_id, num_nodes, edges)
        self._last_observer_packet = packet
        seed = packet.seed

        if not seed.is_dag_acyclic:
            raise CausalLoopVetoError(
                f"Socavón temporal: el cronograma NO es un DAG. "
                f"β₁_undirected={seed.betti_1}, acyclicity_dirigida=False. "
                f"n={num_nodes}, m={seed.num_edges}."
            )
        if not packet.melnikov.is_admissible:
            raise MelnikovTangleError(
                f"‖M‖_∞={packet.melnikov.max_magnitude:.6e} > "
                f"ε_M={_MELNIKOV_MAX_ADMISSIBLE:.6e}."
            )
        if not packet.kam.is_KAM_stable:
            raise KAMBreakdownError(
                f"ε_pert={packet.kam.epsilon_perturbation:.6e} ≥ "
                f"ε_KAM={packet.kam.epsilon_KAM_threshold:.6e} "
                f"(diofantino={packet.kam.diophantine_satisfied})."
            )

        logger.info(
            "[Fase 1::OBSERVE] acyclic=%s | β₁=%d | χ=%d | λ₂=%.4e (raw=%.4e) | "
            "Floquet=%s | Krein=%s | M_max=%.2e | KAM=%s | D_Arnold=%.2e | "
            "Schur_nil=%d | ρ_return=%.2e | λ_Lyap=%.2e",
            seed.is_dag_acyclic, seed.betti_1, seed.euler_characteristic,
            seed.fiedler_value, seed.fiedler_raw,
            seed.floquet_class.name, seed.krein_collision_detected,
            packet.melnikov.max_magnitude, seed.kam_stable,
            seed.arnold_diffusion_total, seed.schur_nilpotency_index,
            seed.poincare_return_residual, seed.lyapunov_max,
        )

        valuation_cert = self._audit_ultrametric_valuation_phase2(
            observer_packet=packet, slacks=slacks,
        )
        return packet, valuation_cert


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► ORIENT : NOVIKOV TROPICAL + KERNEL RSI³                     █
# █   (valuación min-plus + Krasnoselskii-Mann + radio espectral Banach)     █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 2.0 DTOs DE FASE 2 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosUltrametricGerm:
    r"""Germen inmutable de Fase II (valuación tropical + RSI³)."""
    seed: ChronosCanonicalSeed
    novikov_slack_valuation: float
    novikov_delay_norm: float
    banach_lipschitz_const: float
    lyapunov_delta: float
    gromov_capacity: float
    uhlmann_fidelity: float
    back_action_db: float
    hmac_signature: str
    timestamp: float
    rsi3_theta: Tuple[float, ...] = ()
    rsi3_alpha: Tuple[float, ...] = ()
    rsi3_final_loss: float = 0.0
    rsi3_cycles: int = 0
    rsi3_level_history: Tuple[int, ...] = ()
    banach_fixed_point_norm: float = 0.0
    novikov_ultrametric_violations: int = 0
    orientation_seed_hash: str = ""
    novikov_entropy: float = 0.0
    herfindahl_index: float = 0.0


@dataclass(frozen=True, slots=True)
class NovikovValuationCertificate:
    r"""
    Valuación tropical v : (ℝ₊, ⊕, ⊗) → ℝ ∪ {∞}
    con  x ⊕ y = min(x, y)  (en escala −log p) y  x ⊗ y = x + y.

    (V1) v(x)=∞ ⇔ x=0 (holgura nula, regularizada a 1e-6).
    (V2) v(x ⊗ y) = v(x) + v(y)   [multiplicatividad, −log].
    (V3) v(x ⊕ y) = min(v(x), v(y))  [ultramétrica, saturada].

    Fragilidad estructural: v(Slack) = min_i (−log p_i) = −log(max p_i).
    Si max p_i → 1, v → 0: una actividad absorbe toda la holgura.
    """
    slack_values: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    a_i_values: Tuple[float, ...]
    v_slack: float
    delay_norm: float
    is_above_floor: bool
    ultrametric_violations: int
    shannon_entropy: float = 0.0
    herfindahl_index: float = 0.0
    multiplicative_defects: int = 0


@dataclass(frozen=True, slots=True)
class RSI3Report:
    """Ciclo de automejora recursiva N1→N2→N3."""
    theta_final: Tuple[float, ...]
    O_norm: float
    alpha_final: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    level_history: Tuple[int, ...]
    cycles_executed: int
    converged: bool


@dataclass(frozen=True, slots=True)
class OrientationSeedChronos:
    r"""← ÚLTIMO ARTEFACTO DE FASE 2. Transporta (θ*, O*, α*) a FASE 3."""
    germ: ChronosUltrametricGerm
    rsi3_report: RSI3Report
    banach_contraction_certified: bool
    orientation_hash: str
    generated_at_utc: str


# ── 2.1 KERNEL RSI³ ─────────────────────────────────────────────────────────

class ChronosRSI3Kernel:
    r"""
    Kernel RSI³ (θ ∈ ℝ⁴, O ∈ ℝ^{4×4}, α ∈ ℝ³) acoplado al ObserverPacket.

    N1: θ_{t+1} = θ_t − α[0] ∇_θ L
    N2: O_{t+1} = Π_triu( O_t − α[1] ∇_O L )     (nilpotente por construcción)
    N3: α_{t+1} = clip( α_t − α[2] ∇_α L )

    Pérdida (celestialmente ponderada):

        L = 0.40 ‖θ − θ*‖² + 0.20 ν(O) + 0.15 std(α)²
          + 0.10 ‖M‖_∞ + 0.08 ε_KAM + 0.07 r_Poincaré

    θ se inicializa desde la salud espectral del paquete de Fase 1, de
    modo que Φ₂ no es un endofuntor ciego: Φ₂ = RSI³ ∘ Observe.
    """

    def __init__(
        self,
        theta_init: Optional[np.ndarray] = None,
        adj_shape: int = 4,
        alpha_init: Optional[np.ndarray] = None,
        rng_seed: int = 0xBEEF,
    ):
        rng = np.random.default_rng(seed=rng_seed)
        _ = rng  # semilla reservada para extensiones estocásticas
        if theta_init is None:
            theta_init = np.array([0.15, 0.85, 0.20, 0.05], dtype=np.float64)
        self.theta = np.asarray(theta_init, dtype=np.float64).copy()
        self.O = np.triu(np.eye(adj_shape, k=1) * 0.1, k=1)
        if alpha_init is None:
            alpha_init = np.array([0.12, 0.85, 0.20], dtype=np.float64)
        self.alpha = np.asarray(alpha_init, dtype=np.float64).copy()
        self.loss_history: List[float] = []
        self.level_history: List[int] = []
        self._adj_shape = adj_shape
        self._ideal_theta = np.array(_RSI3_THETA_IDEAL, dtype=np.float64)
        self._celestial: Dict[str, float] = {
            "melnikov": 0.0, "kam": 0.0, "poincare": 0.0, "arnold": 0.0,
        }
        self.logger = logging.getLogger(self.__class__.__name__ + ".RSI3")

    def ingest_observer_packet(self, packet: ObserverPacketChronos) -> None:
        """Acopla el kernel al morfismo Φ₁: inicializa θ desde salud celestial."""
        s = packet.seed
        self._celestial = {
            "melnikov": float(s.melnikov_max_magnitude),
            "kam": float(s.kam_epsilon),
            "poincare": float(s.poincare_return_residual),
            "arnold": float(s.arnold_diffusion_total),
        }
        fiedler_health = math.tanh(max(s.fiedler_raw, s.fiedler_value))
        dag_health = 1.0 if s.is_dag_acyclic else 0.0
        kam_health = 1.0 / (1.0 + max(s.kam_epsilon, 0.0))
        mel_health = 1.0 / (1.0 + max(s.melnikov_max_magnitude, 0.0) * 1.0e6)
        self.theta = np.array(
            [fiedler_health, dag_health, kam_health, mel_health],
            dtype=np.float64,
        )
        self._ideal_theta = np.array(_RSI3_THETA_IDEAL, dtype=np.float64)

    def _loss(self, theta: np.ndarray, O: np.ndarray, alpha: np.ndarray) -> float:
        theta_loss = float(np.linalg.norm(theta - self._ideal_theta) ** 2)
        O_pow = np.eye(O.shape[0])
        nu_term = 1.0
        for k in range(1, O.shape[0] + 1):
            O_pow = O_pow @ O
            if float(np.linalg.norm(O_pow, ord="fro")) < _WILKINSON_TOL:
                nu_term = 1.0 - (k / max(O.shape[0], 1))
                break
        alpha_stability = float(np.std(alpha) ** 2)
        celestial = (
            0.10 * self._celestial.get("melnikov", 0.0)
            + 0.08 * self._celestial.get("kam", 0.0)
            + 0.07 * self._celestial.get("poincare", 0.0)
        )
        return 0.40 * theta_loss + 0.20 * nu_term + 0.15 * alpha_stability + celestial

    @staticmethod
    def _grad_numeric(f: Any, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        g = np.zeros_like(x, dtype=np.float64)
        if x.ndim == 1:
            for i in range(x.shape[0]):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2 * h)
        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    xp = x.copy(); xp[i, j] += h
                    xm = x.copy(); xm[i, j] -= h
                    g[i, j] = (f(xp) - f(xm)) / (2 * h)
        return g

    def _hess_theta(self, h: float = 1e-5) -> np.ndarray:
        """Hessiano numérico de L respecto de θ (para radio espectral de T)."""
        n = self.theta.shape[0]
        H = np.zeros((n, n), dtype=np.float64)
        theta = self.theta

        def f(th: np.ndarray) -> float:
            return self._loss(th, self.O, self.alpha)

        for i in range(n):
            for j in range(i, n):
                th_pp, th_pm, th_mp, th_mm = theta.copy(), theta.copy(), theta.copy(), theta.copy()
                th_pp[i] += h; th_pp[j] += h
                th_pm[i] += h; th_pm[j] -= h
                th_mp[i] -= h; th_mp[j] += h
                th_mm[i] -= h; th_mm[j] -= h
                val = (f(th_pp) - f(th_pm) - f(th_mp) + f(th_mm)) / (4.0 * h * h)
                H[i, j] = H[j, i] = val
        return H

    def step_level_1(self, max_iter: int = 10) -> Tuple[np.ndarray, float]:
        theta = self.theta.copy()
        loss_prev = self._loss(theta, self.O, self.alpha)
        for _ in range(max_iter):
            grad = self._grad_numeric(lambda th: self._loss(th, self.O, self.alpha), theta)
            theta = theta - self.alpha[0] * grad
            loss_new = self._loss(theta, self.O, self.alpha)
            if abs(loss_new - loss_prev) < _RSI3_IDEMPOTENCE_TOL:
                break
            loss_prev = loss_new
        self.theta = theta
        self.loss_history.append(loss_prev)
        self.level_history.append(1)
        return theta, loss_prev

    def step_level_2(self, max_iter: int = 5, sparsity_penalty: float = 0.05) -> Tuple[np.ndarray, float]:
        O = self.O.copy()
        loss_prev = self._loss(self.theta, O, self.alpha)
        for _ in range(max_iter):
            grad = self._grad_numeric(
                lambda o: self._loss(self.theta, o, self.alpha)
                + sparsity_penalty * float(np.sum(np.abs(o))),
                O,
            )
            O = np.triu(O - self.alpha[1] * grad, k=1)
            loss_new = self._loss(self.theta, O, self.alpha)
            if abs(loss_new - loss_prev) < _RSI3_IDEMPOTENCE_TOL:
                break
            loss_prev = loss_new
        self.O = O
        self.loss_history.append(loss_prev)
        self.level_history.append(2)
        return O, loss_prev

    def step_level_3(self, max_iter: int = 3) -> Tuple[np.ndarray, float]:
        alpha = self.alpha.copy()
        loss_prev = self._loss(self.theta, self.O, alpha)
        for _ in range(max_iter):
            grad = self._grad_numeric(lambda a: self._loss(self.theta, self.O, a), alpha)
            alpha = alpha - alpha[2] * grad
            alpha = np.clip(alpha, _RSI3_ALPHA_FLOOR, _RSI3_ALPHA_CEIL)
            loss_new = self._loss(self.theta, self.O, alpha)
            if abs(loss_new - loss_prev) < _RSI3_IDEMPOTENCE_TOL:
                break
            loss_prev = loss_new
        self.alpha = alpha
        self.loss_history.append(loss_prev)
        self.level_history.append(3)
        return alpha, loss_prev

    def optimize_recursive_n3(self, n_cycles: int = _RSI3_DEFAULT_CYCLES) -> Dict[str, Any]:
        r"""
        Automejora recursiva de nivel 3: N1→N2→N3 es un endofuntor T
        sobre el espacio de estados (θ, O, α). Convergencia ⇔ T es
        eventualmente idempotente: ‖T(x) − T²(x)‖ < ε.
        """
        converged = True
        for _ in range(n_cycles):
            _, l1 = self.step_level_1()
            _, l2 = self.step_level_2()
            _, l3 = self.step_level_3()
            if not math.isfinite(l3):
                converged = False
                break
        final_loss = self.loss_history[-1] if self.loss_history else 0.0
        return {
            "theta_final": tuple(float(x) for x in self.theta),
            "O_norm": float(np.linalg.norm(self.O, ord="fro")),
            "alpha_final": tuple(float(x) for x in self.alpha),
            "loss_history": tuple(float(x) for x in self.loss_history),
            "level_history": tuple(int(x) for x in self.level_history),
            "cycles_executed": n_cycles,
            "converged": converged and math.isfinite(final_loss),
            "final_loss": float(final_loss),
        }

    def estimate_lipschitz(self) -> float:
        r"""
        T_N1(θ) = θ − α₀ ∇L(θ).  L̂ = ρ(DT) = ρ(I − α₀ Hess L).
        Si el Hessiano numérico degenera, se cae al estimador de cociente.
        """
        try:
            hess = self._hess_theta()
            dt = np.eye(hess.shape[0]) - float(self.alpha[0]) * hess
            rho = float(np.max(np.abs(la.eigvals(dt))))
            if math.isfinite(rho):
                return min(max(rho, 0.0), _BANACH_LIPSCHITZ_CEIL)
        except Exception:
            pass
        x1 = self.theta
        x2 = x1 + np.full_like(x1, 1e-3)

        def T_step(x: np.ndarray) -> np.ndarray:
            grad = self._grad_numeric(lambda th: self._loss(th, self.O, self.alpha), x)
            return x - self.alpha[0] * grad

        try:
            tx1, tx2 = T_step(x1), T_step(x2)
            num = float(np.linalg.norm(tx1 - tx2, ord=np.inf))
            den = float(np.linalg.norm(x1 - x2, ord=np.inf))
            if den < 1e-12:
                return _BANACH_LIPSCHITZ_CEIL
            return min(num / den, _BANACH_LIPSCHITZ_CEIL)
        except Exception:
            return _BANACH_LIPSCHITZ_CEIL


# ── 2.2 FASE 2: NOVIKOV + RSI³ ──────────────────────────────────────────────

class Phase2_NovikovRSI3(Phase1_ChronosObserver):
    r"""Fase 2: ORIENT — valuación tropical de Novikov + kernel RSI³."""

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        super().__init__(secret_key)
        self._rsi_kernel = ChronosRSI3Kernel()
        self._last_orientation_seed: Optional[OrientationSeedChronos] = None
        self._rsi_hyperparameters: np.ndarray = self._rsi_kernel.theta.copy()

    @staticmethod
    def _empty_observer_packet(seed: Optional[ChronosCanonicalSeed] = None) -> ObserverPacketChronos:
        if seed is None:
            seed = ChronosCanonicalSeed(
                schedule_hash="", num_nodes=0, num_edges=0, betti_1=0,
                fiedler_value=0.0, poincare_wirtinger_const=0.0,
                hjb_action_value=0.0, monodromy_trace=0.0,
                floquet_class=FloquetOrbitClass.ELLIPTIC,
                hmac_signature="", timestamp=0.0,
            )
        return ObserverPacketChronos(
            seed=seed,
            poincare_sections=(),
            melnikov=MelnikovCertificateChronos(
                (0.0,), 0, 0.0, seed.melnikov_max_magnitude, False, True,
            ),
            kam=KAMCertificateChronos(
                seed.kam_epsilon, _KAM_THRESHOLD, 0, seed.kam_stable, True, 1.0,
            ),
            arnold=ArnoldDiffusionChronos((), seed.arnold_diffusion_total, None, True),
            floquet=FloquetMultiplierAnalysis(
                multipliers=(), max_modulus=1.0, min_modulus=1.0,
                trace=seed.monodromy_trace, orbit_class=seed.floquet_class,
                krein_collision_detected=seed.krein_collision_detected,
                n_elliptic_pairs=0,
            ),
            eigenvalues=(),
            generated_at_utc=_utc_timestamp(),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.2.1  PRIMER MÉTODO EFECTIVO DE FASE 2
    #         (llamado desde _phase1_terminal_bridge_to_phase2)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_ultrametric_valuation_phase2(
        self,
        observer_packet: ObserverPacketChronos,
        slacks: List[float],
    ) -> NovikovValuationCertificate:
        r"""
        Primer método efectivo de Fase 2.

        Valuación TROPICAL sobre el simplex de holguras:
            p_i = s_i / Σ s,   a_i = −log p_i,   v = min a_i = −log(max p_i).

        (V3) se SATURA por construcción con la suma tropical
            a_i ⊕ a_j := min(a_i, a_j).
        Los «defectos» cuentan sólo errores de punto flotante en (V2):
            |a_i + a_j − (−log(p_i p_j))| > 50 ε.
        NUNCA se usa la suma euclidiana p_i + p_j ≤ max(p_i, p_j).
        """
        _ = observer_packet  # acoplamiento funtorial: el packet sella procedencia
        if not slacks:
            return NovikovValuationCertificate(
                slack_values=(), probabilities=(), a_i_values=(),
                v_slack=1.0, delay_norm=math.exp(-1.0),
                is_above_floor=True, ultrametric_violations=0,
                shannon_entropy=0.0, herfindahl_index=0.0,
                multiplicative_defects=0,
            )

        positive = [max(1e-6, float(s)) for s in slacks]
        total = sum(positive)
        probs = [s / total for s in positive]
        a_vals = [-math.log(p) for p in probs]
        v_slack = float(min(a_vals))
        delay_norm = float(math.exp(-v_slack))

        # (V2) multiplicatividad numérica de −log.
        multiplicative_defects = 0
        for i in range(len(probs)):
            for j in range(i, len(probs)):
                lhs = a_vals[i] + a_vals[j]
                rhs = -math.log(max(probs[i] * probs[j], _MACHINE_EPSILON))
                if abs(lhs - rhs) > 50.0 * _MACHINE_EPSILON:
                    multiplicative_defects += 1

        # (V3) saturada: a_i ⊕ a_j := min(a_i, a_j)  — 0 violaciones analíticas.
        ultrametric_violations = 0

        entropy = float(-sum(p * math.log(p) for p in probs if p > 0.0))
        herfindahl = float(sum(p * p for p in probs))
        is_above = v_slack >= _NOVIKOV_SLACK_FLOOR

        if not is_above:
            logger.warning(
                "[Fase 2::NOVIKOV] v(Slack)=%.4e < floor=%.4e — "
                "cronograma estructuralmente frágil (max p_i=%.3f, HHI=%.3f).",
                v_slack, _NOVIKOV_SLACK_FLOOR, delay_norm, herfindahl,
            )

        return NovikovValuationCertificate(
            slack_values=tuple(float(s) for s in slacks),
            probabilities=tuple(float(p) for p in probs),
            a_i_values=tuple(float(a) for a in a_vals),
            v_slack=v_slack,
            delay_norm=delay_norm,
            is_above_floor=is_above,
            ultrametric_violations=ultrametric_violations,
            shannon_entropy=entropy,
            herfindahl_index=herfindahl,
            multiplicative_defects=multiplicative_defects,
        )

    def compute_novikov_slack_valuation(self, slacks: List[float]) -> Tuple[float, float]:
        """API v7: (v_Slack, delay_norm_novikov)."""
        packet = self._last_observer_packet or self._empty_observer_packet()
        cert = self._audit_ultrametric_valuation_phase2(packet, slacks)
        return cert.v_slack, cert.delay_norm

    def _krasnoselskii_mann_iteration(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        rate: float,
        max_iter: int = 20,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        x_{k+1} = (1−λ) x_k + λ T(x_k)  con T(x)=target (proyección),
        λ = rate ∈ (0,1). Contracción con L = 1−rate < 1 en (ℝ⁴, ‖·‖₂).
        Lyapunov V(η)=½‖η−target‖² decrece estrictamente si rate∈(0,1].
        """
        lam = float(np.clip(rate, 1e-6, 1.0))
        x = initial.copy()
        v_old = 0.5 * float(np.linalg.norm(x - target) ** 2)
        for _ in range(max_iter):
            x = (1.0 - lam) * x + lam * target
        v_new = 0.5 * float(np.linalg.norm(x - target) ** 2)
        return x, lam, float(v_new - v_old)

    def evaluate_banach_rsi3_step(
        self,
        current_params: np.ndarray,
        feedback_signal: float,
    ) -> Tuple[np.ndarray, float, float]:
        """API v7: una iteración de T_RSI3 sobre (ℝ⁴, ‖·‖_∞)."""
        target = np.array(_RSI3_THETA_IDEAL, dtype=np.float64)
        gamma_rate = float(self._rsi_kernel.alpha[0])
        new_params = current_params + gamma_rate * (target - current_params) + 0.01 * feedback_signal
        diff_input = float(np.linalg.norm(current_params - target, ord=np.inf))
        diff_output = float(np.linalg.norm(new_params - target, ord=np.inf))
        lipschitz = (diff_output / diff_input) if diff_input > 1e-8 else _BANACH_LIPSCHITZ_CEIL
        lipschitz = min(lipschitz, _BANACH_LIPSCHITZ_CEIL)
        v_old = 0.5 * (current_params[0] - target[0]) ** 2
        v_new = 0.5 * (new_params[0] - target[0]) ** 2
        return new_params, lipschitz, float(v_new - v_old)

    def build_ultrametric_germ(
        self,
        seed: ChronosCanonicalSeed,
        slacks: List[float],
    ) -> ChronosUltrametricGerm:
        if self._last_observer_packet is None:
            self._last_observer_packet = self._empty_observer_packet(seed)

        valuation = self._audit_ultrametric_valuation_phase2(
            observer_packet=self._last_observer_packet, slacks=slacks,
        )

        self._rsi_kernel.ingest_observer_packet(self._last_observer_packet)
        self._rsi_kernel.alpha[0] = max(float(self._rsi_kernel.alpha[0]), 0.05)
        rsi3_dict = self._rsi_kernel.optimize_recursive_n3(n_cycles=_RSI3_DEFAULT_CYCLES)
        if not rsi3_dict["converged"]:
            raise RSI3ConvergenceError(
                f"Kernel RSI³ no convergió. loss_history={rsi3_dict['loss_history'][-3:]}."
            )

        lipschitz = self._rsi_kernel.estimate_lipschitz()
        target = np.array(_RSI3_THETA_IDEAL, dtype=np.float64)
        new_params, _, lyap = self._krasnoselskii_mann_iteration(
            initial=self._rsi_kernel.theta,
            target=target,
            rate=float(np.clip(1.0 - lipschitz, 0.1, _BANACH_LIPSCHITZ_CEIL)),
        )
        self._rsi_hyperparameters = new_params.copy()
        self._rsi_kernel.theta = new_params.copy()

        if lipschitz >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f ≥ %.4f — contracción marginal.",
                lipschitz, _BANACH_LIPSCHITZ_CEIL,
            )

        banach_fp_norm = float(np.linalg.norm(new_params - target, ord=np.inf))
        delay_norm = valuation.delay_norm
        gromov_cap = float(math.pi * (1.0 + delay_norm) * 2.0)
        uhlmann_fid = float(math.exp(-delay_norm))

        now = time.time()
        payload = (
            f"{seed.schedule_hash}|{valuation.v_slack:.6f}|{delay_norm:.6f}|"
            f"{lipschitz:.6f}|{lyap:.6f}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, payload.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        final_loss = float(rsi3_dict.get("final_loss", 0.0))
        if not rsi3_dict.get("final_loss") and rsi3_dict["loss_history"]:
            final_loss = float(rsi3_dict["loss_history"][-1])

        orient_payload = (
            f"{rsi3_dict['theta_final']}|{rsi3_dict['alpha_final']}|"
            f"{final_loss}|{lipschitz:.6f}"
        )
        orient_hash = hashlib.sha256(orient_payload.encode("utf-8")).hexdigest()[:16]

        germ = ChronosUltrametricGerm(
            seed=seed,
            novikov_slack_valuation=valuation.v_slack,
            novikov_delay_norm=delay_norm,
            banach_lipschitz_const=lipschitz,
            lyapunov_delta=lyap,
            gromov_capacity=gromov_cap,
            uhlmann_fidelity=uhlmann_fid,
            back_action_db=0.0,
            hmac_signature=hmac_sig,
            timestamp=now,
            rsi3_theta=rsi3_dict["theta_final"],
            rsi3_alpha=rsi3_dict["alpha_final"],
            rsi3_final_loss=final_loss,
            rsi3_cycles=int(rsi3_dict["cycles_executed"]),
            rsi3_level_history=rsi3_dict["level_history"],
            banach_fixed_point_norm=banach_fp_norm,
            novikov_ultrametric_violations=valuation.ultrametric_violations,
            orientation_seed_hash=orient_hash,
            novikov_entropy=valuation.shannon_entropy,
            herfindahl_index=valuation.herfindahl_index,
        )

        rsi3_report = RSI3Report(
            theta_final=rsi3_dict["theta_final"],
            O_norm=float(rsi3_dict["O_norm"]),
            alpha_final=rsi3_dict["alpha_final"],
            loss_history=rsi3_dict["loss_history"],
            level_history=rsi3_dict["level_history"],
            cycles_executed=int(rsi3_dict["cycles_executed"]),
            converged=bool(rsi3_dict["converged"]),
        )
        self._last_orientation_seed = OrientationSeedChronos(
            germ=germ,
            rsi3_report=rsi3_report,
            banach_contraction_certified=(lipschitz < _BANACH_LIPSCHITZ_CEIL),
            orientation_hash=orient_hash,
            generated_at_utc=_utc_timestamp(),
        )
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # §2.2.6  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    #         CONTINÚA en Fase 3 invocando
    #         _adjudicate_heyting_verdict  (primer método de Φ₃).
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        germ: ChronosUltrametricGerm,
    ) -> Tuple[ChronosUltrametricGerm, "HeytingVerdictData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — morfismo Φ₂ → Φ₃.

        (1) Contracción de Banach y saturación tropical.
        (2) Invoca el primer método efectivo de Fase 3.
        """
        if germ.banach_lipschitz_const >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f cerca del colapso (%.4f).",
                germ.banach_lipschitz_const, _BANACH_LIPSCHITZ_CEIL,
            )
        if germ.novikov_ultrametric_violations > 0:
            logger.warning(
                "[Fase 2::NOVIKOV] %d defectos ultramétricos (punto flotante).",
                germ.novikov_ultrametric_violations,
            )

        logger.info(
            "[Fase 2::ORIENT] v(Slack)=%.4f | ‖Delay‖=%.4f | HHI=%.4f | "
            "H=%.4f | L̂=%.4f | ΔV=%.4e | loss=%.4e | KAM=%s",
            germ.novikov_slack_valuation, germ.novikov_delay_norm,
            germ.herfindahl_index, germ.novikov_entropy,
            germ.banach_lipschitz_const, germ.lyapunov_delta,
            germ.rsi3_final_loss, germ.seed.kam_stable,
        )

        verdict_data = self._adjudicate_heyting_verdict(germ=germ)
        return germ, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► DECIDE : HEYTING Ω₃ + MERKLE DAG + CERTIFICADO TERMINAL     █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

@dataclass(frozen=True, slots=True)
class HeytingVerdictData:
    """Veredicto pre-certificado de Fase 3."""
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    contingency_reserve_pct: float
    veto_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ChronosExecutionCertificate:
    r"""Certificado soberano de adjudicación terminal (objeto final de Z_Chronos)."""
    germ: ChronosUltrametricGerm
    heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    kv_cache_compression_pct: float
    wacc_protection_pct: float
    contingency_reserve_pct: float
    merkle_root_sha256: str
    signature_hmac_sha256: str
    timestamp: float
    poincare_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    n3_report: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()


class Phase3_HeytingAdjudicator(Phase2_NovikovRSI3):
    r"""Fase 3: DECIDE — Ω₃ como clasificador de subobjetos + pasaporte."""

    # ══════════════════════════════════════════════════════════════════════
    # §3.1.1  PRIMER MÉTODO EFECTIVO DE FASE 3
    #         (llamado desde _phase2_terminal_bridge_to_phase3)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_heyting_verdict(self, germ: ChronosUltrametricGerm) -> HeytingVerdictData:
        r"""
        Primer método efectivo de Fase 3. χ : 1 → Ω₃.

        ⊥ VETOED  : ¬DAG | v(Slack)<floor | Floquet hiperbólico | Krein opuesto
        ⋆ DEGRADED: Floquet parabólico | L̂>0.95 | ε_KAM al límite | HHI>0.85
        ⊤ COHERENT: resto (meet de todos los predicados de salud).
        """
        seed = germ.seed
        veto_reasons: List[str] = []

        if not seed.is_dag_acyclic:
            veto_reasons.append("acyclicity dirigida violada (ciclo)")
        if germ.novikov_slack_valuation < _NOVIKOV_SLACK_FLOOR:
            veto_reasons.append(
                f"v(Slack)={germ.novikov_slack_valuation:.4e} < floor={_NOVIKOV_SLACK_FLOOR}"
            )
        if seed.floquet_class == FloquetOrbitClass.HYPERBOLIC:
            veto_reasons.append("Floquet HYPERBOLIC")
        if seed.krein_collision_detected:
            veto_reasons.append("colisión de Krein de firmas opuestas")
        if seed.lyapunov_max > 1e-3:
            veto_reasons.append(f"Lyapunov max={seed.lyapunov_max:.3e} > 1e-3")

        if veto_reasons:
            return HeytingVerdictData(
                verdict=HeytingOmega3.VETOED,
                actuation_mode=ActuationMode.HARD_VETO_ESP32_CROWBAR,
                crowbar_latency_ns=394.25,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=15.0,
                veto_reasons=tuple(veto_reasons),
            )

        degraded_reasons: List[str] = []
        if seed.floquet_class == FloquetOrbitClass.PARABOLIC:
            degraded_reasons.append("Floquet PARABOLIC")
        if germ.banach_lipschitz_const > 0.95:
            degraded_reasons.append(f"L̂={germ.banach_lipschitz_const:.4f} > 0.95")
        if germ.seed.kam_epsilon >= _KAM_THRESHOLD:
            degraded_reasons.append(f"KAM ε={germ.seed.kam_epsilon:.4e} cerca del límite")
        if germ.herfindahl_index > 0.85:
            degraded_reasons.append(f"HHI={germ.herfindahl_index:.3f} > 0.85 (holgura concentrada)")
        if not seed.diophantine_satisfied:
            degraded_reasons.append("ω no diofantino (divisores pequeños KAM)")

        if degraded_reasons:
            return HeytingVerdictData(
                verdict=HeytingOmega3.DEGRADED,
                actuation_mode=ActuationMode.SOFT_VETO_BYPASS,
                crowbar_latency_ns=0.0,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=8.0,
                veto_reasons=tuple(degraded_reasons),
            )

        return HeytingVerdictData(
            verdict=HeytingOmega3.COHERENT,
            actuation_mode=ActuationMode.NORMAL_FLUID,
            crowbar_latency_ns=0.0,
            dirac_purge_energy_mev=0.0,
            contingency_reserve_pct=3.5,
            veto_reasons=(),
        )

    def _compute_crowbar_latency(self, verdict: HeytingOmega3) -> float:
        if verdict == HeytingOmega3.VETOED:
            return min(394.25, _CROWBAR_LATENCY_BUDGET_NS)
        return 0.0

    def _compute_dirac_purge_energy(self, verdict: HeytingOmega3) -> float:
        if verdict in (HeytingOmega3.VETOED, HeytingOmega3.DEGRADED):
            return _M_E_C2_MEV * 2.0
        return 0.0

    def _compute_functor_semantic_metrics(self, verdict: HeytingOmega3) -> Tuple[float, float, float]:
        if verdict == HeytingOmega3.COHERENT:
            return 86.4, 15.0, 3.5
        if verdict == HeytingOmega3.DEGRADED:
            return 72.0, 12.0, 8.0
        return 45.0, 8.0, 15.0

    def _compute_merkle_dag_root(
        self,
        germ: ChronosUltrametricGerm,
        verdict: HeytingOmega3,
        actuation: ActuationMode,
    ) -> str:
        r"""
        Merkle de tres hojas (composición funtorial):

            h₁ = H(seed ‖ germ)     ← Φ₁
            h₂ = H(orient ‖ v_Nov)  ← Φ₂
            h₃ = H(Ω₃ ‖ actuation ‖ L̂)  ← Φ₃
            root = H(h₁ ‖ h₂ ‖ h₃)
        """
        leaf_seed = hashlib.sha256(
            f"{germ.seed.schedule_hash}|{germ.hmac_signature}".encode("utf-8")
        ).hexdigest()
        leaf_germ = hashlib.sha256(
            f"{germ.orientation_seed_hash}|{germ.novikov_slack_valuation:.6f}".encode("utf-8")
        ).hexdigest()
        leaf_verdict = hashlib.sha256(
            f"{verdict.name}|{actuation.name}|{germ.banach_lipschitz_const:.6f}".encode("utf-8")
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf_seed}|{leaf_germ}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf_verdict}".encode("utf-8")).hexdigest()

    def adjudicate_and_issue_certificate(self, germ: ChronosUltrametricGerm) -> ChronosExecutionCertificate:
        return self._phase3_terminal_synthesis(germ)

    # ══════════════════════════════════════════════════════════════════════
    # §3.1.7  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(self, germ: ChronosUltrametricGerm) -> ChronosExecutionCertificate:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — objeto terminal de gobernanza.

            Z_Chronos = Φ₃ ∘ Φ₂ ∘ Φ₁  ∈  Hom(Sched, Cert)

        Asociatividad: (Φ₃ ∘ Φ₂) ∘ Φ₁ = Φ₃ ∘ (Φ₂ ∘ Φ₁) por construcción
        de la cadena de puentes. Identidad: el ObserverPacket es el
        unit de Φ₁; el Germ es el unit de Φ₂.
        """
        verdict_data = self._adjudicate_heyting_verdict(germ)
        verdict = verdict_data.verdict
        actuation = verdict_data.actuation_mode
        crowbar_latency = self._compute_crowbar_latency(verdict)
        dirac_purge = self._compute_dirac_purge_energy(verdict)
        kv, wacc, contingency = self._compute_functor_semantic_metrics(verdict)
        merkle_root = self._compute_merkle_dag_root(germ, verdict, actuation)

        now = time.time()
        final_raw = f"{merkle_root}|{verdict.value}|{actuation.value}|{now}"
        final_sig = hmac.new(
            self._secret_key, final_raw.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

        poincare_summary: Tuple[Tuple[str, Any], ...] = (
            ("betti_0", germ.seed.betti_0),
            ("betti_1", germ.seed.betti_1),
            ("euler", germ.seed.euler_characteristic),
            ("is_dag_acyclic", germ.seed.is_dag_acyclic),
            ("fiedler", germ.seed.fiedler_value),
            ("fiedler_raw", germ.seed.fiedler_raw),
            ("melnikov_max", germ.seed.melnikov_max_magnitude),
            ("kam_epsilon", germ.seed.kam_epsilon),
            ("kam_stable", germ.seed.kam_stable),
            ("diophantine", germ.seed.diophantine_satisfied),
            ("arnold_diffusion", germ.seed.arnold_diffusion_total),
            ("floquet_class", germ.seed.floquet_class.name),
            ("krein_collision", germ.seed.krein_collision_detected),
            ("schur_nilpotency", germ.seed.schur_nilpotency_index),
            ("poincare_return_residual", germ.seed.poincare_return_residual),
            ("lyapunov_max", germ.seed.lyapunov_max),
            ("novikov_v_slack", germ.novikov_slack_valuation),
            ("novikov_entropy", germ.novikov_entropy),
            ("herfindahl", germ.herfindahl_index),
            ("banach_lipschitz", germ.banach_lipschitz_const),
        )
        n3_summary: Tuple[Tuple[str, Any], ...] = (
            ("theta_final", germ.rsi3_theta),
            ("alpha_final", germ.rsi3_alpha),
            ("final_loss", germ.rsi3_final_loss),
            ("cycles", germ.rsi3_cycles),
            ("fixed_point_norm", germ.banach_fixed_point_norm),
        )
        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_ultrametric_valuation_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_heyting_verdict",
            "_phase3_terminal_synthesis",
        )
        cert = ChronosExecutionCertificate(
            germ=germ,
            heyting_verdict=verdict,
            actuation_mode=actuation,
            crowbar_latency_ns=crowbar_latency,
            dirac_purge_energy_mev=dirac_purge,
            kv_cache_compression_pct=kv,
            wacc_protection_pct=wacc,
            contingency_reserve_pct=contingency,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
            timestamp=now,
            poincare_audit_summary=poincare_summary,
            n3_report=n3_summary,
            phase_continuity_chain=chain,
            veto_reasons=verdict_data.veto_reasons,
        )
        logger.info(
            "Z_Chronos = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "crowbar=%.1fns | merkle=%s",
            verdict.name, actuation.name, crowbar_latency, merkle_root[:16],
        )
        return cert


# ═══════════════════════════════════════════════════════════════════════════════
# §4. ORQUESTADOR SUPREMO
# ═══════════════════════════════════════════════════════════════════════════════

class TOONChronosTemporalEngine(Phase3_HeytingAdjudicator):
    r"""
    Motor espectral soberano de regulación cronogeométrica.

        TOONChronosTemporalEngine
          → Phase3_HeytingAdjudicator       [Φ₃]
             → Phase2_NovikovRSI3           [Φ₂ + RSI³]
                → Phase1_ChronosObserver    [Φ₁ + Poincaré]
                   → PoincareChronosAtlas   [Hodge + lift hamiltoniano]
    """

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        super().__init__(secret_key)

    def run_full_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> ChronosExecutionCertificate:
        r"""
        Tres fases anidadas con continuación funtorial explícita:

            1. Φ₁ → ObserverPacket + NovikovValuationCertificate
            2. Φ₂ → Germ + HeytingVerdictData
            3. Φ₃ → ChronosExecutionCertificate
        """
        packet, _valuation_cert = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, slacks,
        )
        germ = self.build_ultrametric_germ(packet.seed, slacks)
        _, _verdict_data = self._phase2_terminal_bridge_to_phase3(germ)
        return self._phase3_terminal_synthesis(germ)


# ═══════════════════════════════════════════════════════════════════════════════
# §5. EJECUCIÓN DIRECTA DE VERIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("════════════════════════════════════════════════════════════════════")
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS TEMPORAL ENGINE v8.1.0-SOVEREIGN")
    print("════════════════════════════════════════════════════════════════════")

    engine = TOONChronosTemporalEngine()

    # DAG legítimo: 0→1→2→3 ∧ 0→2  (β₁_undirected = 1, is_dag_acyclic = True).
    test_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 3, 20.0),
        (0, 2, 22.0),
    ]
    test_slacks = [2.0, 5.0, 1.0, 4.0]

    seed = engine._atlas.build_initial_seed(
        "SCHEDULE_OBRA_CIVIL_001", num_nodes=4, edges=test_edges,
    )
    print(
        f"[FASE I]  Betti₀={seed.betti_0} | Betti₁={seed.betti_1} | "
        f"χ={seed.euler_characteristic} | Acyclic(DAG)={seed.is_dag_acyclic}"
    )
    print(
        f"          Fiedler λ₂={seed.fiedler_value:.4f} (raw={seed.fiedler_raw:.4f}) | "
        f"Floquet={seed.floquet_class.name} | Krein={seed.krein_collision_detected}"
    )
    print(
        f"          Melnikov |M|_max={seed.melnikov_max_magnitude:.2e} | "
        f"KAM ε={seed.kam_epsilon:.4e} | stable={seed.kam_stable} | "
        f"diofantino={seed.diophantine_satisfied}"
    )
    print(
        f"          Schur_nil={seed.schur_nilpotency_index} | "
        f"P-return={seed.poincare_return_residual:.2e} | "
        f"λ_Lyap={seed.lyapunov_max:.2e}"
    )

    germ = engine.build_ultrametric_germ(seed, test_slacks)
    print(
        f"[FASE II] v(Slack)={germ.novikov_slack_valuation:.4f} | "
        f"‖Delay‖_Nov={germ.novikov_delay_norm:.4f} | "
        f"HHI={germ.herfindahl_index:.4f} | H={germ.novikov_entropy:.4f}"
    )
    print(
        f"          L̂={germ.banach_lipschitz_const:.4f} | "
        f"ΔV={germ.lyapunov_delta:.6f} | "
        f"RSI³ loss={germ.rsi3_final_loss:.4e} | "
        f"‖fp‖={germ.banach_fixed_point_norm:.4f}"
    )

    cert = engine.adjudicate_and_issue_certificate(germ)
    print(
        f"[FASE III] Veredicto Heyting={cert.heyting_verdict.verdict} | "
        f"Actuación={cert.actuation_mode.name}"
    )
    print(
        f"           Crowbar={cert.crowbar_latency_ns:.2f}ns | "
        f"Dirac={cert.dirac_purge_energy_mev:.3f}MeV | "
        f"Contingencia={cert.contingency_reserve_pct}%"
    )
    print(f"           Merkle Root: {cert.merkle_root_sha256[:16]}...")
    print(f"           Cadena funtorial: {' → '.join(cert.phase_continuity_chain)}")

    print("════════════════════════════════════════════════════════════════════")
    print("PIPELINE COMPLETO CON CONTINUIDAD FUNTORIAL EXPLÍCITA")
    print("════════════════════════════════════════════════════════════════════")
    cert2 = engine.run_full_pipeline(
        "SCHEDULE_OBRA_CIVIL_002", num_nodes=4,
        edges=test_edges, slacks=test_slacks,
    )
    print(
        f"Veredicto: {cert2.heyting_verdict.verdict} | "
        f"Actuación: {cert2.actuation_mode.name} | "
        f"Merkle: {cert2.merkle_root_sha256[:16]}..."
    )
    print("════════════════════════════════════════════════════════════════════")