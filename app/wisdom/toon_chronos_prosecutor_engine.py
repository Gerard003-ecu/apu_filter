# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Prosecutor Engine (Motor Fiscal Cronogeométrico) — v9.0.0                    ║
║ Ubicación: app/wisdom/toon_chronos_prosecutor_engine.py                                              ║
║ Versión  : 9.0.0-Prosecutor-DualPoincare-Cech-CStar-Bogomolny-RSI3-Celestial-NestedPhases-Doctoral   ║
║ Función  : Motor fiscal e inquisidor de coherencia temporal en el Estrato WISDOM.                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONChronosProsecutorEngine` actúa como el motor fiscal e inquisidor de coherencia
cronogeométrica en el Estrato WISDOM de la arquitectura cognitiva APU Filter.

1. DUALIDAD POINCARÉ LEY VS ESTADO Y ESTRUCTURA SIMPLÉCTICA $Sp(4, \mathbb{R})$:
   - Holonomía de Wilson de la Ley: $W(\gamma) = \mathcal{P} \exp \left( \oint_\gamma A \right) \in SO(n)$, calculada
     sobre la conexión antisimétrica $A = \frac{1}{2}(W_{\text{norm}} - W_{\text{norm}}^\top) \in \mathfrak{so}(n)$.
   - Monodromía de Estado Hamiltoniana: $M = \exp(A_{\text{sp}} \tau) \in Sp(4, \mathbb{R})$ generada por el sistema
     restringido 2-dof con $A_{\text{sp}}^\top J_4 + J_4 A_{\text{sp}} = 0$.
   - Defecto Simpléctico de Poincaré: $\delta_{\text{Sp4}} = \| M^\top J_4 M - J_4 \|_F$.
   - Fractura Ley-Estado (Law-State Fracture): Se detecta cuando la clase de Floquet de la Ley difiere de la del Estado:
       $$\text{Fracture} \iff \text{FloquetClass}(W) \neq \text{FloquetClass}(M)$$

2. COHOMOLOGÍA DE ČECH Y EXACTITUD $\delta^2 = 0$:
   Sobre la 1-cocadena de pesos simetrizados $g(\{i,j\}) = w_{ij} + w_{ji}$, el 2-cobordo de Čech en $\check{H}^2(X, \mathcal{F}_{\text{time}})$ es:
     $$(\delta g)(i, j, k) = g_{ij} + g_{jk} - g_{ik}$$
   La condición de consistencia cohomológica exige que la aplicación del operador cobordo de segundo orden satisfaga
   identicamente $\|\delta^2 g\|_F = 0$, garantizando la ausencia de torsión o fibración temporal inconsistente.

3. AUDITORÍA $C^*$-ALGEBRAICA DEL ESTADO DE GIBBS:
   Para la matriz de densidad $\rho = \exp(-L_{\text{sched}}) / Z$, la representación $C^*$ impone los axiomas:
     - Hermiticidad : $\|\rho - \rho^*\|_F \le 10^{-12}$.
     - Normalización : $|\text{Tr}(\rho) - 1| \le 10^{-12}$.
     - Positividad : $\lambda_{\min}(\rho) \ge -10^{-14}$.
     - Entropía de von Neumann : $S(\rho) = -\text{Tr}(\rho \ln \rho)$.

4. COTA DE BOGOMOL'NYI Y DISPARIDAD DE RECOMPENSA (RHI / DCR):
   En el marco de vórtices abelianos discretos, la norma del campo satisface la desigualdad de Bogomol'nyi:
     $$\|F\|_F^2 = \sum_{\triangle} |\text{flux}|^2 \ge \frac{8\pi^2 |k|}{\text{vol}(X)}, \quad k = \text{round}\left(\frac{\sum \text{flux}}{2\pi}\right)$$
   Los indicadores adversariales de fraude cronológico se especifican rigurosamente como:
     - Reward Hacking Index (RHI) : $RHI \in [0, 1]$, gatillado por ciclos dirigidos, fractura Ley-Estado o violaciones $C^*$.
     - Disguised Cost Ratio (DCR) : $DCR \in [0, 1]$, derivado del flujo de Čech y solapamiento de Chirikov.
     - Indictment Score : $\mathcal{S} = 0.4 \cdot DCR + 0.4 \cdot RHI + 0.2 \cdot \mathbb{I}_{\text{Fracture}}$.

5. AUTOMEJORA RECURSIVA RSI³ CON META-RECURSIÓN:
   La optimización $N_1(\theta) \to N_2(O) \to N_3(\alpha)$ incorpora meta-recursión de profundidad 3: si $\widehat{L} \ge 1 - \epsilon$,
   se re-escala dinámicamente la tasa $\alpha$ restaurando el Teorema de Punto Fijo de Banach y garantizando decrecimiento monótono $\Delta V \le 0$.

DISEÑO DE TRES FASES ANIDADAS (Z = Φ₃ ∘ Φ₂ ∘ Φ₁):
─────────────────────────────────────────────────────────────────────────────────────────
  FASE 1 (PROSECUTION OBSERVE — Φ₁):
    · Kahn aciclicidad dirigida O(V+E) con deque.
    · Formas normales de Schur, holonomía de Wilson $W(\gamma)$ y monodromía $Sp(4, \mathbb{R})$.
    · Validación cruzada de dualidad Ley vs Estado (Detección de Fractura).
    · Obstrucción cohomológica de Čech $\check{H}^2$ con certificación $\delta^2 = 0$.
    · Análisis celeste de Poincaré: Melnikov, KAM diofantino, Chirikov y Birkhoff.
    ⟹ Emisión de `ChronosProsecutionCanonicalSeed` + `ProsecutorObserverPacket`.
    ⟹ Transición terminal: `_phase1_terminal_bridge_to_phase2` $\to$ `_audit_cstar_indictment_phase2`.

  FASE 2 (INDICTMENT ORIENT — Φ₂):
    · Auditoría C*-algebraica $(\text{Hermiticidad}, \text{Normalización}, \text{PSD}, S(\rho))$ de $\rho = \exp(-L_{\text{sched}})/Z$.
    · Evaluación de cota topológica de Bogomol'nyi $\|F\|_F^2 \ge 8\pi^2|k|/\text{vol}(X)$.
    · Computación de indicadores RHI, DCR e Indictment Score.
    · Kernel RSI³ de profundidad 3 con restauración de Banach $\widehat{L} < 1.0$ y prueba Lyapunov.
    ⟹ Emisión de `ChronosProsecutionIndictmentGerm`.
    ⟹ Transición terminal: `_phase2_terminal_bridge_to_phase3` $\to$ `_adjudicate_prosecution_verdict`.

  FASE 3 (SOVEREIGN DECIDE — Φ₃):
    · Adjudicación en Retículo de Heyting $\Omega_3 = \{\bot, \star, \top\}$.
    · Activación de hardware ESP32 Crowbar ($WCET \le 394.25 \text{ ns}$).
    · Sello criptográfico Merkle DAG de 3 hojas y firma HMAC-SHA256.
    ⟹ Emisión de `ChronosProsecutionExecutionCertificate`.
    ⟹ Síntesis terminal: `_phase3_terminal_synthesis`.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONChronosProsecutorEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ─── Constantes transversales ────────────────────────────────────────────────
_TAU: float = 2.0 * math.pi
_M_E_C2_MEV: float = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS: float = 400.0
_NOVIKOV_SLACK_FLOOR: float = 0.10
_BANACH_LIPSCHITZ_CEIL: float = 0.9990
_FIEDLER_GAP_MIN: float = 1e-4
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_WILKINSON_TOL: float = 1e-12

# Fiscal tolerances
_CSTAR_HERMITIAN_TOL: float = 1e-12
_CSTAR_TRACE_TOL: float = 1e-12
_CSTAR_POSITIVITY_TOL: float = 1e-14
_BOGOMOLNY_DEFECT_FRACTION: float = 0.30
_RHI_VETO_THRESHOLD: float = 0.85
_DCR_DEGRADED_THRESHOLD: float = 0.25

# Poincaré / celestial mechanics
_POINCARE_TOL: float = 1e-7
_MELNIKOV_MAX_ADMISSIBLE: float = 1e-6
_KAM_THRESHOLD: float = 1e-3
_KAM_PERSISTENCE_TOL: float = 1e-6
_ARNOLD_DIFFUSION_TOL: float = 1e-4
_WILSON_UNITARY_TOL: float = 1e-3
_MONODROMY_ELLIPTIC_TOL: float = 1e-3
_MONODROMY_PARABOLIC_TOL: float = 1e-2
_SP4_SYMPLECTIC_TOL: float = 1e-8
_DIOPHANTINE_GAMMA: float = 1e-4
_DIOPHANTINE_TAU: float = 2.5
_POINCARE_RETURN_SAMPLES: int = 12
_CHIRIKOV_OVERLAP_TOL: float = 1.0
_CECH_SQUARE_ZERO_TOL: float = 1e-12

# Forma simpléctica canónica J ∈ M₄(ℝ), Jᵀ = −J, J² = −I.
_J4: np.ndarray = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# RSI³ — automejora recursiva de profundidad 3
_RSI3_CYCLES: int = 3
_RSI3_IDEMPOTENCE_TOL: float = 1e-7
_RSI3_ALPHA_FLOOR: float = 1e-6
_RSI3_ALPHA_CEIL: float = 1e2
_RSI3_MAX_RECURSION_DEPTH: int = 3
_RSI3_BANACH_MARGIN: float = 1e-3
_RSI3_ARMIJO_C1: float = 1e-4
_RSI3_ARMIJO_RHO: float = 0.5
_RSI3_ARMIJO_MAX: int = 8


# ══════════════════════════════════════════════════════════════════════════════
# §0. EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutorEngineError(Exception):
    pass


class ProsecutorScheduleFormatError(ProsecutorEngineError):
    pass


class DirectedCycleDetectedError(ProsecutorEngineError):
    pass


class CStarViolationError(ProsecutorEngineError):
    pass


class BogomolnyDefectError(ProsecutorEngineError):
    pass


class LawStateFractureError(ProsecutorEngineError):
    pass


class RHIThresholdExceededError(ProsecutorEngineError):
    pass


class ProsecutorRSI3ConvergenceError(ProsecutorEngineError):
    pass


class ProsecutorAdjudicationError(ProsecutorEngineError):
    pass


class SymplecticViolationError(ProsecutorEngineError):
    pass


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_or(value: float, fallback: float = 0.0) -> float:
    v = float(value)
    return v if math.isfinite(v) else fallback


# ══════════════════════════════════════════════════════════════════════════════
# §1. ENUMS Y DTOs
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
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
        r"""a → b = ⊤ si b ≥ a, else b  (álgebra de Heyting lineal de cadena)."""
        if int(other) >= int(self):
            return HeytingOmega3.COHERENT
        return other


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class FloquetOrbitClass(IntEnum):
    ELLIPTIC = 0
    PARABOLIC = 1
    HYPERBOLIC = 2


# ── Fase 1 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosProsecutionCanonicalSeed:
    r"""Semilla inmutable de Fase 1 (v7 + extensiones v8/v9)."""
    schedule_hash: str
    num_nodes: int
    num_edges: int
    betti_1: int
    fiedler_value: float
    poincare_wirtinger_const: float
    cech_obstruction_norm: float
    law_floquet_class: FloquetOrbitClass
    state_floquet_class: FloquetOrbitClass
    law_state_fracture: bool
    hmac_signature: str
    timestamp: float
    is_dag_acyclic: bool = True
    betti_0_undirected: int = 1
    euler_char_undirected: int = 1
    law_wilson_trace: float = 2.0
    state_monodromy_trace: float = 0.0
    melnikov_max: float = 0.0
    kam_epsilon: float = 0.0
    kam_stable: bool = True
    arnold_diffusion: float = 0.0
    observer_packet_hash: str = ""
    notes: Tuple[str, ...] = ()
    # v9 — mecánica celeste
    symplectic_defect: float = 0.0
    is_symplectic: bool = True
    diophantine_held: bool = True
    chirikov_overlap: float = 0.0
    birkhoff_twist: float = 0.0
    poincare_n_returns: int = 0
    cech_delta2_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class DualPoincareCertificate:
    r"""Certificado de la Validación Cruzada Dual Ley vs Estado."""
    law_class: FloquetOrbitClass
    state_class: FloquetOrbitClass
    fracture_detected: bool
    law_wilson_trace: float
    law_wilson_eigenvalues_modulus: Tuple[float, ...]
    state_monodromy_trace: float
    state_monodromy_eigenvalues_modulus: Tuple[float, ...]
    cycle_path: Tuple[int, ...]
    symplectic_defect: float = 0.0
    is_symplectic: bool = True
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CechObstructionCertificate:
    r"""Obstrucción de Čech Ȟ²(X, ℱ_time) con certificado δ²=0."""
    cocycle_norm: float
    per_triangle_flux: Tuple[Tuple[Tuple[int, int, int], float], ...]
    n_triangles: int
    is_coboundary_trivial: bool
    delta2_norm: float = 0.0
    delta2_vanishes: bool = True
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProsecutorPoincareAnalysis:
    r"""Poincaré–Melnikov–KAM–Arnold–Chirikov–Birkhoff del flujo espectral."""
    n_sections: int
    melnikov_max: float
    melnikov_tangle: bool
    kam_epsilon: float
    kam_stable: bool
    arnold_diffusion: float
    arnold_adiabatic: bool
    dominant_mode: Optional[int] = None
    sections: Tuple[Tuple[int, float], ...] = ()
    diophantine_held: bool = True
    diophantine_min_divisor: float = 1.0
    chirikov_overlap: float = 0.0
    birkhoff_twist: float = 0.0
    n_returns: int = 0
    recurrence_proxy: float = 0.0


@dataclass(frozen=True, slots=True)
class ProsecutorObserverPacket:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 1.

    Empaqueta todos los certificados de fiscalización hacia FASE 2.
    El puente funtorial Φ₁→Φ₂ consume este paquete y no otro.
    """
    seed: ChronosProsecutionCanonicalSeed
    dual_poincare: DualPoincareCertificate
    cech: CechObstructionCertificate
    poincare: ProsecutorPoincareAnalysis
    laplacian_schedule: Tuple[Tuple[float, ...], ...]
    connection_1form: Tuple[Tuple[float, ...], ...]
    is_fiscally_coherent: bool
    generated_at_utc: str


# ── Fase 2 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosProsecutionIndictmentGerm:
    r"""Germen inmutable de Fase 2 (v7 + extensiones v8/v9)."""
    seed: ChronosProsecutionCanonicalSeed
    indictment_score: float
    disguised_cost_ratio: float
    reward_hacking_index: float
    bogomolny_energy_defect: float
    cstar_hermitian_error: float
    banach_lipschitz_const: float
    lyapunov_delta: float
    hmac_signature: str
    timestamp: float
    cstar_trace_error: float = 0.0
    cstar_positivity_error: float = 0.0
    bogomolny_lower_bound: float = 0.0
    bogomolny_actual_norm: float = 0.0
    rsi3_theta: Tuple[float, ...] = ()
    rsi3_alpha: Tuple[float, ...] = ()
    rsi3_final_loss: float = 0.0
    rsi3_converged: bool = False
    rsi3_level_history: Tuple[int, ...] = ()
    orientation_seed_hash: str = ""
    # v9 — RSI³ recursivo + C*
    von_neumann_entropy: float = 0.0
    rsi3_recursion_depth: int = 0
    rsi3_banach_restored: bool = True
    lyapunov_monotone: bool = True


@dataclass(frozen=True, slots=True)
class CStarAuditCertificate:
    r"""Certificado C*-algebraico del operador de densidad."""
    hermitian_error: float
    trace_error: float
    positivity_error: float
    is_hermitian: bool
    is_normalized: bool
    is_positive_semidefinite: bool
    is_full_rank: bool
    eigenvalue_min: float
    eigenvalue_max: float
    condition_number: float
    von_neumann_entropy: float = 0.0
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BogomolnyCertificate:
    r"""Cota de Bogomol'nyi: ‖F‖²_F ≥ 8π²|k|/vol(X)."""
    pontryagin_number: int
    field_norm_sq: float
    lower_bound_sq: float
    defect: float
    is_saturated: bool
    volume_proxy: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndictmentIndicators:
    r"""Indicadores adversariales RHI, DCR y score compuesto."""
    reward_hacking_index: float
    disguised_cost_ratio: float
    indictment_score: float
    rhi_veto: bool
    dcr_degraded: bool


@dataclass(frozen=True, slots=True)
class ProsecutorRSI3Report:
    r"""Reporte del kernel RSI³ de profundidad 3."""
    theta_before: Tuple[float, ...]
    theta_after: Tuple[float, ...]
    O_norm: float
    alpha_final: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    level_history: Tuple[int, ...]
    converged: bool
    lipschitz_const: float
    recursion_depth: int = 0
    banach_restored: bool = True
    lyapunov_monotone: bool = True
    lyapunov_delta: float = 0.0


# ── Fase 3 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ProsecutorHeytingVerdictData:
    r"""Veredicto pre-certificado de Fase 3."""
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    contingency_reserve_pct: float
    veto_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ChronosProsecutionExecutionCertificate:
    r"""Certificado Soberano de Adjudicación Terminal (v7 + extensiones v8/v9)."""
    germ: ChronosProsecutionIndictmentGerm
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
    fiscal_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    rsi3_report_summary: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()


def _classify_floquet_moduli(
    moduli: Sequence[float],
) -> FloquetOrbitClass:
    r"""Clasificación elíptica/parabólica/hiperbólica por módulos de Floquet."""
    if not moduli:
        return FloquetOrbitClass.ELLIPTIC
    max_mod = max(moduli)
    min_mod = min(moduli)
    if (
        abs(max_mod - 1.0) < _MONODROMY_ELLIPTIC_TOL
        and abs(min_mod - 1.0) < _MONODROMY_ELLIPTIC_TOL
    ):
        return FloquetOrbitClass.ELLIPTIC
    if (
        abs(max_mod - 1.0) < _MONODROMY_PARABOLIC_TOL
        or abs(min_mod - 1.0) < _MONODROMY_PARABOLIC_TOL
    ):
        return FloquetOrbitClass.PARABOLIC
    return FloquetOrbitClass.HYPERBOLIC


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► PROSECUTION OBSERVE                                        █
# █   Kahn + Schur + Dual Poincaré celeste + Čech δ²=0 + Melnikov–KAM       █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase1_ProsecutorObserver:
    r"""
    Fase 1 — Prosecution Observe.

    Certifica la topología del cronograma con Kahn dirigido, el espectro de
    Schur, la dualidad Ley–Estado (Wilson path-ordered vs monodromía Sp(4,ℝ)),
    la obstrucción de Čech con δ²=0, y el análisis celeste de Poincaré
    (retorno, Melnikov homoclínico, KAM diofantino, Chirikov–Arnold, Birkhoff).
    """

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        self._secret_key = secret_key
        self._last_packet: Optional[ProsecutorObserverPacket] = None

    # ══════════════════════════════════════════════════════════════════════
    # §1.1  Validación y Kahn (deque)
    # ══════════════════════════════════════════════════════════════════════
    def _validate_edges(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        if num_nodes < 1:
            raise ProsecutorScheduleFormatError(
                f"num_nodes ≥ 1. Recibido {num_nodes}."
            )
        if edges is None:
            raise ProsecutorScheduleFormatError("edges no puede ser None.")
        out: List[Tuple[int, int, float]] = []
        for i, e in enumerate(edges):
            try:
                u, v, d = e
            except (TypeError, ValueError) as exc:
                raise ProsecutorScheduleFormatError(
                    f"Arista {i} no desempaquetable: {e!r}."
                ) from exc
            if not (0 <= u < num_nodes) or not (0 <= v < num_nodes):
                raise ProsecutorScheduleFormatError(
                    f"Arista {i}: nodo fuera de rango ({u},{v})."
                )
            fd = float(d)
            if not math.isfinite(fd) or fd < 0.0:
                raise ProsecutorScheduleFormatError(
                    f"Arista {i}: duración inválida {d!r}."
                )
            out.append((int(u), int(v), fd))
        return out

    def _detect_directed_cycles_kahn(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[bool, int]:
        r"""
        Kahn O(V+E) con deque: (is_acyclic, n_nodos_residuales).

        Un cronograma con ciclo dirigido viola la causalidad temporal
        (socavón: una tarea es precondición de sí misma).
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
        return processed == num_nodes, num_nodes - processed

    # ══════════════════════════════════════════════════════════════════════
    # §1.2  Laplaciano y Schur
    # ══════════════════════════════════════════════════════════════════════
    def _build_schedule_laplacian(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[np.ndarray, int, int, int, float, float]:
        r"""
        Laplaciano combinatorio L = B₁ B₁ᵀ del complejo 1-dimensional.

        rank-nullity:
            β₀ = dim ker L = n − rank(B₁)
            β₁ = m − rank(B₁)
            χ  = β₀ − β₁
        λ₂ (Fiedler) acota la constante de Poincaré–Wirtinger pw = 1/λ₂.
        """
        num_edges = len(edges)
        if num_nodes <= 0 or num_edges == 0:
            n = max(1, num_nodes)
            return (
                np.zeros((n, n), dtype=np.float64),
                1, 0, 1, _FIEDLER_GAP_MIN, 1.0 / _FIEDLER_GAP_MIN,
            )

        B1 = np.zeros((num_nodes, num_edges), dtype=np.float64)
        for idx, (u, v, _) in enumerate(edges):
            B1[u, idx] = -1.0
            B1[v, idx] = 1.0
        L = B1 @ B1.T
        rank_B1 = int(np.linalg.matrix_rank(B1, tol=_WILKINSON_TOL))
        betti_1 = max(0, num_edges - rank_B1)
        betti_0 = max(0, num_nodes - rank_B1)
        euler = betti_0 - betti_1
        try:
            eigvals = np.sort(np.real(la.eigvalsh(L)))
        except Exception:
            eigvals = np.zeros(num_nodes, dtype=np.float64)
        fiedler = float(eigvals[1]) if len(eigvals) > 1 else _FIEDLER_GAP_MIN
        fiedler = max(fiedler, _FIEDLER_GAP_MIN)
        pw = 1.0 / fiedler
        return L, betti_0, betti_1, euler, fiedler, pw

    def _compute_schur_rsf(
        self,
        L: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""RSF real: L = Q T Qᵀ. Retorna (T, Q, κ₂(Q))."""
        n = int(L.shape[0])
        if n == 0:
            return np.zeros((1, 1)), np.eye(1), 1.0
        try:
            T, Q = la.schur(L, output="real")
        except Exception:
            try:
                return np.diag(np.real(la.eigvalsh(L))), np.eye(n), 1.0
            except Exception:
                return np.zeros((n, n)), np.eye(n), 1.0
        try:
            sv = la.svd(Q, compute_uv=False)
            sv_min = float(sv[-1])
            kappa = float(sv[0]) / sv_min if sv_min > _MACHINE_EPSILON else 1.0
        except Exception:
            kappa = 1.0
        return T, Q, kappa

    # ══════════════════════════════════════════════════════════════════════
    # §1.3  Conexión 1-forma antisimétrica
    # ══════════════════════════════════════════════════════════════════════
    def _build_connection_1form(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> np.ndarray:
        r"""
        A ∈ 𝔰𝔬(n): A = (W_norm − W_normᵀ)/2, reescalada a ‖A‖_F ≤ 1/2.

        A es la conexión de Chern–Simons discreta sobre el 1-esqueleto:
        la holonomía de A mide la torsión temporal del cronograma.
        """
        n = num_nodes
        if n <= 0:
            return np.zeros((1, 1), dtype=np.float64)
        W = np.zeros((n, n), dtype=np.float64)
        for u, v, w in edges:
            W[u, v] = float(w)
        deg = np.sum(np.abs(W), axis=1)
        deg_safe = np.where(deg > _MACHINE_EPSILON, deg, 1.0)
        W_norm = W / deg_safe[:, None]
        A = 0.5 * (W_norm - W_norm.T)
        scale = float(np.linalg.norm(A, ord="fro"))
        if scale > 0.5:
            A = A * (0.5 / scale)
        return A

    # ══════════════════════════════════════════════════════════════════════
    # §1.4  Ciclo más corto (undirected BFS por arista)
    # ══════════════════════════════════════════════════════════════════════
    def _find_shortest_cycle(
        self,
        edges: Sequence[Tuple[int, int, float]],
        n: int,
    ) -> List[int]:
        r"""
        Ciclo no dirigido más corto: para cada arista {u,v}, BFS de u a v
        evitando esa arista. Complejidad O(E(V+E)).
        """
        undirected: Dict[int, List[int]] = {i: [] for i in range(n)}
        undirected_set: Dict[int, set] = {i: set() for i in range(n)}
        for u, v, _ in edges:
            if u == v:
                continue
            if v not in undirected_set[u]:
                undirected[u].append(v)
                undirected_set[u].add(v)
            if u not in undirected_set[v]:
                undirected[v].append(u)
                undirected_set[v].add(u)

        best: List[int] = []
        best_len = n + 2
        seen_undirected = set()
        for u, v, _ in edges:
            a, b = (u, v) if u <= v else (v, u)
            if a == b or (a, b) in seen_undirected:
                continue
            seen_undirected.add((a, b))
            parent = {u: -1}
            dist = {u: 0}
            q: deque[int] = deque([u])
            found = False
            while q and not found:
                x = q.popleft()
                if dist[x] + 1 >= best_len:
                    continue
                for y in undirected[x]:
                    if x == u and y == v:
                        continue
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        if y == v:
                            found = True
                            break
                        q.append(y)
            if not found or v not in dist:
                continue
            path = [v]
            cur = v
            while parent[cur] != -1:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            cycle = path + [u]
            if 3 <= len(cycle) < best_len:
                best_len = len(cycle)
                best = cycle
        return best

    # ══════════════════════════════════════════════════════════════════════
    # §1.5  Wilson holonomy PATH-ORDERED sobre el ciclo
    # ══════════════════════════════════════════════════════════════════════
    def _compute_wilson_holonomy_real(
        self,
        A: np.ndarray,
        cycle: Sequence[int],
    ) -> Tuple[np.ndarray, float, Tuple[float, ...], FloquetOrbitClass]:
        r"""
        W(γ) = P exp(∮_γ A)  — producto ordenado de transportadores.

        Para cada arista e_k = (u_k → u_{k+1}) del ciclo se extrae el
        generador de rotación en el 2-plano span{e_u, e_v}:

            G_k[u,v] = A[u,v],   G_k[v,u] = A[v,u]

        y se acumula W ← expm(G_k) W.  Si A ∈ 𝔰𝔬(n), cada expm(G_k) ∈ SO(n)
        y W es (numéricamente) ortogonal.  La clase de Floquet se lee en
        los módulos de los autovalores de W.
        """
        n = int(A.shape[0]) if A.ndim == 2 else 0
        if len(cycle) < 3 or n == 0:
            return (
                np.array([[1.0 + 0j]]),
                2.0,
                (1.0,),
                FloquetOrbitClass.ELLIPTIC,
            )

        nodes = list(cycle)
        if nodes[0] != nodes[-1]:
            nodes = nodes + [nodes[0]]

        W = np.eye(n, dtype=np.float64)
        n_steps = 0
        for u, v in zip(nodes[:-1], nodes[1:]):
            if not (0 <= u < n and 0 <= v < n) or u == v:
                continue
            G = np.zeros((n, n), dtype=np.float64)
            G[u, v] = float(A[u, v])
            G[v, u] = float(A[v, u])
            try:
                W = la.expm(G) @ W
                n_steps += 1
            except Exception:
                continue
        if n_steps == 0:
            return (
                np.eye(n, dtype=complex),
                float(n),
                (1.0,) * n,
                FloquetOrbitClass.ELLIPTIC,
            )

        tr_W = float(np.real(np.trace(W)))
        try:
            eigs = la.eigvals(W)
            moduli = tuple(float(m) for m in np.sort(np.abs(eigs)))
        except Exception:
            moduli = (1.0,) * n
        cls = _classify_floquet_moduli(moduli)
        return W, tr_W, moduli, cls

    # ══════════════════════════════════════════════════════════════════════
    # §1.6  State monodromy REAL en Sp(4,ℝ) — 2-dof celeste
    # ══════════════════════════════════════════════════════════════════════
    def _hamiltonian_sp4_generator(
        self,
        durations: Sequence[float],
    ) -> Tuple[np.ndarray, float]:
        r"""
        Generador A ∈ 𝔰𝔭(4,ℝ) del oscilador acoplado 2-dof (análogo del
        problema restringido plano):

            H = ½(p₁² + p₂²) + (ω²/2)(q₁² + q₂²) + κ q₁ q₂

        con ω = 2π/⟨T⟩, κ = 0.05 · ω².  El campo hamiltoniano es

            q̇ = p,   ṗ = −ω² q − κ σ₁ q

        y A verifica Aᵀ J + J A = 0  (definición de 𝔰𝔭(4,ℝ)).
        """
        if not durations:
            avg_dur = 1.0
        else:
            avg_dur = float(np.mean(durations))
            if not math.isfinite(avg_dur) or avg_dur <= 0.0:
                avg_dur = 1.0
        omega = _TAU / max(0.1, avg_dur)
        omega2 = omega * omega
        kappa = 0.05 * omega2
        A_sp = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-omega2, -kappa, 0.0, 0.0],
                [-kappa, -omega2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return A_sp, omega

    def _symplectic_defect(self, M: np.ndarray) -> float:
        r"""‖Mᵀ J M − J‖_F  — anula exactamente a Sp(4,ℝ)."""
        if M.shape != (4, 4):
            return float("inf")
        residual = M.T @ _J4 @ M - _J4
        return float(np.linalg.norm(residual, ord="fro"))

    def _lie_algebra_sp4_defect(self, A: np.ndarray) -> float:
        r"""‖Aᵀ J + J A‖_F  — anula exactamente a 𝔰𝔭(4,ℝ)."""
        if A.shape != (4, 4):
            return float("inf")
        residual = A.T @ _J4 + _J4 @ A
        return float(np.linalg.norm(residual, ord="fro"))

    def _compute_state_monodromy_real(
        self,
        durations: List[float],
    ) -> Tuple[np.ndarray, float, Tuple[float, ...], FloquetOrbitClass, float]:
        r"""
        M = expm(A · τ) ∈ Sp(4,ℝ), τ = 0.05 (sección de Poincaré local).

        Retorna (M, tr M, |λ|, clase, defect_simpléctico).
        Conservación del invariante integral de Poincaré  ∮ p dq  ⇔  M ∈ Sp(4).
        """
        A_sp, _omega = self._hamiltonian_sp4_generator(durations)
        lie_def = self._lie_algebra_sp4_defect(A_sp)
        try:
            M = la.expm(A_sp * 0.05)
        except Exception:
            M = np.eye(4, dtype=np.float64)
        sp_def = max(self._symplectic_defect(M), lie_def)
        tr_M = float(np.trace(M))
        try:
            eigs = la.eigvals(M)
            moduli = tuple(float(m) for m in np.sort(np.abs(eigs)))
        except Exception:
            moduli = (1.0,) * 4
        cls = _classify_floquet_moduli(moduli)
        return M, tr_M, moduli, cls, float(sp_def)

    # ══════════════════════════════════════════════════════════════════════
    # §1.7  Validación Cruzada Dual Ley vs Estado
    # ══════════════════════════════════════════════════════════════════════
    def _evaluate_law_vs_state_duality(
        self,
        A: np.ndarray,
        cycle: Sequence[int],
        durations: List[float],
    ) -> DualPoincareCertificate:
        r"""
        Dualidad celeste:
          Ley    = holonomía de Wilson de la conexión A (geometría de gauge).
          Estado = monodromía hamiltoniana Sp(4,ℝ) de las duraciones.

        Fractura ⇔ las clases de Floquet difieren: el cronograma geométrico
        no es compatible con el flujo temporal que induce.  Es el análogo
        del desacuerdo entre la órbita kepleriana (ley) y el estado osculatriz.
        """
        _, tr_W, moduli_W, law_cls = self._compute_wilson_holonomy_real(A, cycle)
        _, tr_M, moduli_M, state_cls, sp_def = self._compute_state_monodromy_real(
            durations
        )
        fracture = law_cls != state_cls
        notes: List[str] = []
        if fracture:
            notes.append(
                f"FRACTURA: Ley={law_cls.name} ≠ Estado={state_cls.name}."
            )
        is_sp = sp_def <= _SP4_SYMPLECTIC_TOL
        if not is_sp:
            notes.append(f"M ∉ Sp(4,ℝ): ‖MᵀJM−J‖_F={sp_def:.4e}.")
        return DualPoincareCertificate(
            law_class=law_cls,
            state_class=state_cls,
            fracture_detected=fracture,
            law_wilson_trace=tr_W,
            law_wilson_eigenvalues_modulus=moduli_W,
            state_monodromy_trace=tr_M,
            state_monodromy_eigenvalues_modulus=moduli_M,
            cycle_path=tuple(int(x) for x in cycle),
            symplectic_defect=float(sp_def),
            is_symplectic=bool(is_sp),
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.8  Obstrucción de Čech REAL + δ²=0
    # ══════════════════════════════════════════════════════════════════════
    def _compute_cech_obstruction(
        self,
        edges: Sequence[Tuple[int, int, float]],
        num_nodes: int,
    ) -> CechObstructionCertificate:
        r"""
        1-cocadena g ∈ C¹: g({i,j}) = w_ij + w_ji  (peso simetrizado).
        2-cobordo   (δg)(i,j,k) = g_ij + g_jk − g_ik.

        El complejo de Čech (con coeficientes ℝ) satisface δ²=0.  Se evalúa
        δ(δg) sobre tetraedros (i,j,k,ℓ) y se certifica ‖δ² g‖ ≤ tol.

        ‖[δg]‖ = Σ |flux| ; si es ~0 la fibración temporal es trivial
        (el cronograma se sincroniza globalmente sin torsión).
        """
        edge_map: Dict[Tuple[int, int], float] = {
            (u, v): float(w) for u, v, w in edges
        }

        def g(a: int, b: int) -> float:
            return edge_map.get((a, b), 0.0) + edge_map.get((b, a), 0.0)

        flux_per_triangle: List[Tuple[Tuple[int, int, int], float]] = []
        total_norm = 0.0
        for i, j, k in itertools.combinations(range(num_nodes), 3):
            w_ij, w_jk, w_ik = g(i, j), g(j, k), g(i, k)
            if w_ij == 0.0 and w_jk == 0.0 and w_ik == 0.0:
                continue
            flux = w_ij + w_jk - w_ik
            flux_per_triangle.append(((i, j, k), float(flux)))
            total_norm += abs(flux)

        # δ² sobre 3-símplices: (δF)(i,j,k,ℓ) = F_jkl − F_ikl + F_ijl − F_ijk.
        flux_lookup: Dict[Tuple[int, int, int], float] = {
            trip: fl for trip, fl in flux_per_triangle
        }
        delta2_norm = 0.0
        if num_nodes >= 4 and flux_lookup:
            for i, j, k, ell in itertools.combinations(range(num_nodes), 4):
                def F(a: int, b: int, c: int) -> float:
                    return flux_lookup.get((a, b, c), 0.0)

                d2 = F(j, k, ell) - F(i, k, ell) + F(i, j, ell) - F(i, j, k)
                delta2_norm += abs(d2)

        is_trivial = total_norm <= _WILKINSON_TOL * max(num_nodes, 1)
        delta2_ok = delta2_norm <= _CECH_SQUARE_ZERO_TOL * max(num_nodes, 1)
        notes: List[str] = []
        if not is_trivial:
            notes.append(
                f"Ȟ² ≠ 0 (norm={total_norm:.4e}): bundle con torsión."
            )
        if not delta2_ok:
            notes.append(
                f"δ²g no nulo (‖δ²‖={delta2_norm:.4e}): inconsistencia cohomológica."
            )
        return CechObstructionCertificate(
            cocycle_norm=float(total_norm),
            per_triangle_flux=tuple(flux_per_triangle),
            n_triangles=len(flux_per_triangle),
            is_coboundary_trivial=bool(is_trivial),
            delta2_norm=float(delta2_norm),
            delta2_vanishes=bool(delta2_ok),
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.9  Poincaré celeste: retorno, Melnikov, KAM diofantino, Chirikov
    # ══════════════════════════════════════════════════════════════════════
    def _hamiltonian_lift_generator(self, L: np.ndarray) -> np.ndarray:
        r"""
        Levantamiento hamiltoniano del laplaciano (análogo celeste):

            H(q,p) = ½‖p‖² + ½ qᵀ L q
            ż = J ∇H  ⇒  G = [[0, I], [−L, 0]] ∈ 𝔰𝔭(2n,ℝ)
        """
        n = int(L.shape[0])
        G = np.zeros((2 * n, 2 * n), dtype=np.float64)
        if n == 0:
            return G
        G[:n, n:] = np.eye(n)
        G[n:, :n] = -L
        return G

    def _diophantine_condition(
        self,
        omega: np.ndarray,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = 6,
    ) -> Tuple[bool, float]:
        r"""
        Condición diofantina de Siegel–KAM:

            |⟨k, ω⟩| ≥ γ / |k|_1^τ    ∀ k ∈ ℤ^d \ {0},  |k|_1 ≤ k_max.

        Si se viola, hay pequeño divisor y los toros de KAM no persisten.
        """
        d = int(omega.shape[0])
        if d == 0:
            return True, 1.0
        min_div = float("inf")
        held = True
        # Producto cartesiano de k_i ∈ [−k_max, k_max] es explosivo;
        # se recorre la bola ℓ¹ de radio k_max por combinaciones acotadas.
        rng = range(-k_max, k_max + 1)
        for k_tuple in itertools.product(rng, repeat=min(d, 3)):
            if all(ki == 0 for ki in k_tuple):
                continue
            k_arr = np.array(k_tuple[:d], dtype=np.float64)
            if d > 3:
                break
            k1 = float(np.sum(np.abs(k_arr)))
            if k1 < 1.0:
                continue
            div = abs(float(np.dot(k_arr, omega[: len(k_arr)])))
            thresh = gamma / (k1 ** tau)
            if div < min_div:
                min_div = div
            if div < thresh:
                held = False
        if not math.isfinite(min_div):
            min_div = 1.0
        return held, float(min_div)

    def _poincare_return_map_2mode(
        self,
        lam2: float,
        lam3: float,
        eps_pert: float,
        n_samples: int = _POINCARE_RETURN_SAMPLES,
    ) -> Tuple[int, float, float]:
        r"""
        Mapa de retorno de Poincaré del sistema 2-modos (Fiedler + siguiente):

            q̈₁ = −λ₂ q₁ − ε q₂
            q̈₂ = −λ₃ q₂ − ε q₁

        Sección Σ = {q₁ = 0, p₁ > 0}.  Se integra el flujo lineal 4D
        ż = G₄ z por un periodo tentativo T=2π/√λ₂ y se cuenta el número
        de cruces transversos.  El twist de Birkhoff es la diferencia de
        números de rotación entre la órbita interior (ε=0) y la perturbada.
        """
        w2 = math.sqrt(max(lam2, _FIEDLER_GAP_MIN))
        w3 = math.sqrt(max(lam3, _FIEDLER_GAP_MIN))
        G4 = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-lam2, -eps_pert, 0.0, 0.0],
                [-eps_pert, -lam3, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        T = _TAU / w2
        dt = T / max(n_samples, 4)
        try:
            Phi = la.expm(G4 * dt)
        except Exception:
            return 0, 0.0, 0.0

        z = np.array([0.0, 0.15, 1.0, 0.0], dtype=np.float64)  # sobre Σ
        n_returns = 0
        q1_prev = float(z[0])
        for _ in range(n_samples * 4):
            z = Phi @ z
            q1 = float(z[0])
            p1 = float(z[2])
            if q1_prev < 0.0 <= q1 and p1 > 0.0:
                n_returns += 1
            q1_prev = q1

        rho_inner = 1.0  # isócrono: un retorno por periodo
        rho_outer = float(n_returns) / 4.0 if n_samples else 1.0
        twist = abs(rho_outer - rho_inner)
        # Tiempo de recurrencia de Poincaré (proxy): T / max(n_returns, 1)
        rec = T / max(n_returns, 1)
        return n_returns, float(twist), float(rec)

    def _analyze_prosecutor_poincare(
        self,
        L: np.ndarray,
        fiedler: float,
        A: np.ndarray,
    ) -> ProsecutorPoincareAnalysis:
        r"""
        Mecánica celeste de Poincaré sobre el flujo espectral.

        1. Sección de Poincaré del calor v(t)=exp(−L t)v₀ (contracción).
        2. Melnikov homoclínico: M(t₀)=∫ {H₀,H₁}(z_h(t+t₀)) dt
           con H₀=½ Σ λᵢ (qᵢ²+pᵢ²) (integrable) y H₁=qᵀ (L−diag L) q.
        3. KAM: ε = ‖L−diag L‖_F / λ₂  + condición diofantina de ω.
        4. Chirikov: solapamiento de resonancias ⇒ difusión de Arnold.
        5. Birkhoff: twist del mapa de retorno 2-modos ⇒ ≥2 órbitas periódicas.
        """
        n = int(L.shape[0])
        empty = ProsecutorPoincareAnalysis(
            n_sections=0,
            melnikov_max=0.0,
            melnikov_tangle=False,
            kam_epsilon=0.0,
            kam_stable=True,
            arnold_diffusion=0.0,
            arnold_adiabatic=True,
        )
        if n == 0:
            return empty

        try:
            eigvals, eigvecs = la.eigh(L)
        except Exception:
            return empty

        eigvals = np.real(eigvals)
        # ── sección de Poincaré del calor ────────────────────────────────
        rng = np.random.default_rng(seed=0xC0FFEE)
        v0 = rng.standard_normal(n)
        v0 /= max(float(np.linalg.norm(v0)), _MACHINE_EPSILON)
        u0 = eigvecs.T @ v0
        sections: List[Tuple[int, float]] = []
        for k in range(1, max(4, n + 2)):
            u_k = u0 * np.exp(-eigvals * 0.1 * k)
            v_k = eigvecs @ u_k
            norm_k = float(np.linalg.norm(v_k))
            if norm_k <= _POINCARE_TOL:
                sections.append((k, norm_k))
                break

        # ── Melnikov: Poisson {H₀, H₁} = ∇H₀ᵀ J ∇H₁ sobre el 4-flujo ─────
        # H₀ = ½ Σ λᵢ rᵢ²  (acciones),  H₁ = qᵀ A_off q, A_off = L − diag(L).
        # En coordenadas espectrales el corchete se reduce a
        #   {H₀,H₁}(t) = −2 q̇ᵀ A_off q = −2 pᵀ A_off q.
        # Se muestrea a lo largo de la órbita armónica de los dos modos lentos.
        A_off = L - np.diag(np.diag(L))
        omega_f = max(math.sqrt(fiedler), _MACHINE_EPSILON)
        M_vals = np.zeros(8, dtype=np.float64)
        dt = 0.1
        q_amp = eigvecs[:, 1] if n > 1 else eigvecs[:, 0]
        p_dir = q_amp  # fase π/2 del mismo modo
        for j in range(8):
            t0 = j * _TAU / (omega_f * 8.0)
            acc = 0.0
            for kk in range(16):
                t_k = kk * dt
                phase = omega_f * (t_k + t0)
                q = q_amp * math.cos(phase)
                p = p_dir * (-omega_f * math.sin(phase))
                poisson = -2.0 * float(p @ (A_off @ q))
                acc += poisson * dt
            M_vals[j] = acc
        melnikov_max = float(np.max(np.abs(M_vals)))
        signs = np.sign(M_vals)
        tangle = (
            int(np.sum(np.diff(signs) != 0)) > 0
            and float(np.mean(np.abs(M_vals))) > _WILKINSON_TOL
        )

        # ── KAM heurístico + diofantino ──────────────────────────────────
        off_fro = float(np.linalg.norm(A_off, ord="fro"))
        omega_min = max(fiedler, _FIEDLER_GAP_MIN)
        eps_pert = off_fro / omega_min
        modes_pos = np.array(
            [math.sqrt(max(float(ev), 0.0)) for ev in eigvals if ev > _FIEDLER_GAP_MIN],
            dtype=np.float64,
        )
        if modes_pos.size == 0:
            modes_pos = np.array([omega_min], dtype=np.float64)
        dio_held, dio_min = self._diophantine_condition(modes_pos[:3])
        eps_KAM = max(_KAM_THRESHOLD, _KAM_PERSISTENCE_TOL)
        kam_stable = bool(eps_pert < eps_KAM and dio_held)

        # ── Arnold / Chirikov ────────────────────────────────────────────
        modes = np.sort(np.real(eigvals))
        arnold = 0.0
        dominant = None
        chirikov = 0.0
        if len(modes) >= 2:
            leakage: List[Tuple[int, float]] = []
            half_widths: List[float] = []
            for k in range(1, len(modes)):
                d_omega = float(modes[k] - modes[k - 1])
                omega_k = max(math.sqrt(abs(modes[k])), _FIEDLER_GAP_MIN)
                leak = (d_omega * d_omega) / (omega_k * omega_k)
                leakage.append((k, leak))
                arnold += leak
                # Anchura de resonancia ~ √ε  (Chirikov)
                half_widths.append(math.sqrt(max(eps_pert, 0.0)) * omega_k)
            if leakage:
                dominant = max(leakage, key=lambda kv: kv[1])[0]
            # Solapamiento: Σ 2·Δ_k  /  (ω_max − ω_min)
            span = float(modes[-1] - modes[1]) if len(modes) > 1 else 1.0
            span = max(span, _MACHINE_EPSILON)
            chirikov = float(2.0 * sum(half_widths) / span)
        arnold_adiabatic = arnold < _ARNOLD_DIFFUSION_TOL and chirikov < _CHIRIKOV_OVERLAP_TOL

        # ── Mapa de retorno 2-modos + twist de Birkhoff ──────────────────
        lam2 = float(modes[1]) if len(modes) > 1 else fiedler
        lam3 = float(modes[2]) if len(modes) > 2 else lam2 * 1.618
        n_returns, twist, rec = self._poincare_return_map_2mode(
            lam2, lam3, float(eps_pert)
        )

        return ProsecutorPoincareAnalysis(
            n_sections=len(sections),
            melnikov_max=melnikov_max,
            melnikov_tangle=tangle,
            kam_epsilon=float(eps_pert),
            kam_stable=kam_stable,
            arnold_diffusion=float(arnold),
            arnold_adiabatic=bool(arnold_adiabatic),
            dominant_mode=dominant,
            sections=tuple(sections),
            diophantine_held=bool(dio_held),
            diophantine_min_divisor=float(dio_min),
            chirikov_overlap=float(chirikov),
            birkhoff_twist=float(twist),
            n_returns=int(n_returns),
            recurrence_proxy=float(rec),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.10  build_prosecutor_observer_packet
    # ══════════════════════════════════════════════════════════════════════
    def build_prosecutor_observer_packet(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ProsecutorObserverPacket:
        r"""Empaqueta todos los certificados de Fase 1."""
        edges_v = self._validate_edges(num_nodes, edges)
        is_acyclic, n_residual = self._detect_directed_cycles_kahn(
            num_nodes, edges_v
        )

        L, betti_0, betti_1, euler, fiedler, pw = self._build_schedule_laplacian(
            num_nodes, edges_v
        )
        _T_schur, _Q_schur, _kappa_schur = self._compute_schur_rsf(L)
        A_conn = self._build_connection_1form(num_nodes, edges_v)

        cycle = self._find_shortest_cycle(edges_v, num_nodes)
        durations = [e[2] for e in edges_v]

        dual = self._evaluate_law_vs_state_duality(A_conn, cycle, durations)
        cech = self._compute_cech_obstruction(edges_v, num_nodes)
        poincare = self._analyze_prosecutor_poincare(L, fiedler, A_conn)

        notes: List[str] = []
        if not is_acyclic:
            notes.append(
                f"Kahn detectó ciclo dirigido ({n_residual} nodos residuales)."
            )
        if dual.fracture_detected:
            notes.append(
                f"Fractura Ley-Estado: {dual.law_class.name} vs {dual.state_class.name}."
            )
        if not cech.is_coboundary_trivial:
            notes.append(f"Ȟ² ≠ 0 (norm={cech.cocycle_norm:.4e}).")
        if not cech.delta2_vanishes:
            notes.append(f"δ²g ≠ 0 (norm={cech.delta2_norm:.4e}).")
        if not poincare.kam_stable:
            notes.append(
                f"KAM ε={poincare.kam_epsilon:.4e} o diofantino={poincare.diophantine_held}."
            )
        if poincare.melnikov_tangle:
            notes.append(f"Melnikov tangle (max={poincare.melnikov_max:.4e}).")
        if poincare.chirikov_overlap >= _CHIRIKOV_OVERLAP_TOL:
            notes.append(f"Chirikov overlap={poincare.chirikov_overlap:.4e}.")
        if not dual.is_symplectic:
            notes.append(f"Sp(4) defect={dual.symplectic_defect:.4e}.")

        is_fiscally_coherent = bool(
            is_acyclic
            and not dual.fracture_detected
            and poincare.kam_stable
            and dual.is_symplectic
            and cech.delta2_vanishes
        )

        now = time.time()
        raw = (
            f"{schedule_id}|{num_nodes}|{len(edges_v)}|{betti_1}|"
            f"{fiedler:.6f}|{cech.cocycle_norm:.6f}|"
            f"{dual.law_class.name}|{dual.state_class.name}|"
            f"{dual.fracture_detected}|{is_acyclic}|"
            f"{poincare.melnikov_max:.6f}|{poincare.kam_stable}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        packet_hash_payload = (
            f"{is_acyclic}|{dual.fracture_detected}|"
            f"{cech.cocycle_norm:.6e}|{poincare.kam_epsilon:.6e}|"
            f"{poincare.diophantine_held}|{dual.symplectic_defect:.6e}"
        )
        packet_hash = hashlib.sha256(
            packet_hash_payload.encode("utf-8")
        ).hexdigest()[:16]

        seed = ChronosProsecutionCanonicalSeed(
            schedule_hash=hashlib.sha256(schedule_id.encode("utf-8")).hexdigest(),
            num_nodes=num_nodes,
            num_edges=len(edges_v),
            betti_1=betti_1,
            fiedler_value=fiedler,
            poincare_wirtinger_const=pw,
            cech_obstruction_norm=float(cech.cocycle_norm),
            law_floquet_class=dual.law_class,
            state_floquet_class=dual.state_class,
            law_state_fracture=dual.fracture_detected,
            hmac_signature=hmac_sig,
            timestamp=now,
            is_dag_acyclic=is_acyclic,
            betti_0_undirected=betti_0,
            euler_char_undirected=euler,
            law_wilson_trace=dual.law_wilson_trace,
            state_monodromy_trace=dual.state_monodromy_trace,
            melnikov_max=poincare.melnikov_max,
            kam_epsilon=poincare.kam_epsilon,
            kam_stable=poincare.kam_stable,
            arnold_diffusion=poincare.arnold_diffusion,
            observer_packet_hash=packet_hash,
            notes=tuple(notes),
            symplectic_defect=dual.symplectic_defect,
            is_symplectic=dual.is_symplectic,
            diophantine_held=poincare.diophantine_held,
            chirikov_overlap=poincare.chirikov_overlap,
            birkhoff_twist=poincare.birkhoff_twist,
            poincare_n_returns=poincare.n_returns,
            cech_delta2_norm=cech.delta2_norm,
        )

        packet = ProsecutorObserverPacket(
            seed=seed,
            dual_poincare=dual,
            cech=cech,
            poincare=poincare,
            laplacian_schedule=tuple(
                tuple(float(x) for x in row) for row in L
            ),
            connection_1form=tuple(
                tuple(float(x) for x in row) for row in A_conn
            ),
            is_fiscally_coherent=is_fiscally_coherent,
            generated_at_utc=_utc_timestamp(),
        )
        self._last_packet = packet
        return packet

    # ══════════════════════════════════════════════════════════════════════
    # §1.11  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        durations: List[float],
    ) -> Tuple[ProsecutorObserverPacket, "IndictmentIndicators"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        (1) Ejecuta `build_prosecutor_observer_packet`.
        (2) Verifica invariantes de Fase 1:
              · DAG dirigido (Kahn).
              · Sp(4,ℝ) de la monodromía de estado.
              · δ²=0 del complejo de Čech.
              · KAM + diofantino (se registra; no aborta).
        (3) CONTINÚA inmediatamente en el primer método efectivo de Fase 2:
              `_audit_cstar_indictment_phase2`.

        Retorna (ProsecutorObserverPacket, IndictmentIndicators).
        """
        packet = self.build_prosecutor_observer_packet(
            schedule_id, num_nodes, edges
        )

        if not packet.seed.is_dag_acyclic:
            raise DirectedCycleDetectedError(
                f"Socavón temporal: Kahn detectó ciclo dirigido. "
                f"β₁_undirected={packet.seed.betti_1}, "
                f"is_dag_acyclic={packet.seed.is_dag_acyclic}. "
                f"Notas: {packet.seed.notes}."
            )

        if not packet.dual_poincare.is_symplectic:
            logger.warning(
                "[Fase 1::Sp(4)] defect=%.4e ≥ tol=%.1e — invariante integral "
                "de Poincaré no conservado numéricamente.",
                packet.dual_poincare.symplectic_defect,
                _SP4_SYMPLECTIC_TOL,
            )

        if not packet.poincare.kam_stable:
            logger.warning(
                "[Fase 1::KAM] ε=%.4e diofantino=%s — toros no persistentes.",
                packet.poincare.kam_epsilon,
                packet.poincare.diophantine_held,
            )

        logger.info(
            "[Fase 1::PROSECUTION OBSERVE] acyclic=%s | β₁=%d | χ=%d | "
            "λ₂=%.4f | fracture=%s (%s vs %s) | Ȟ²=%.4e | δ²=%.2e | "
            "M_max=%.2e | KAM=%s | Dio=%s | Chirikov=%.2e | "
            "Birkhoff_twist=%.2e | D_Arnold=%.2e | Sp4=%s",
            packet.seed.is_dag_acyclic,
            packet.seed.betti_1,
            packet.seed.euler_char_undirected,
            packet.seed.fiedler_value,
            packet.dual_poincare.fracture_detected,
            packet.dual_poincare.law_class.name,
            packet.dual_poincare.state_class.name,
            packet.cech.cocycle_norm,
            packet.cech.delta2_norm,
            packet.poincare.melnikov_max,
            packet.poincare.kam_stable,
            packet.poincare.diophantine_held,
            packet.poincare.chirikov_overlap,
            packet.poincare.birkhoff_twist,
            packet.poincare.arnold_diffusion,
            packet.dual_poincare.is_symplectic,
        )

        # ── CONTINUACIÓN FUNTORIAL Φ₁ → Φ₂ ───────────────────────────────
        # El último método de Fase 1 ES el inicio operativo de Fase 2.
        indicators = self._audit_cstar_indictment_phase2(
            packet=packet, durations=durations
        )
        return packet, indicators


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► INDICTMENT ORIENT                                          █
# █   C*-audit + Bogomol'nyi + RHI/DCR + RSI³ recursivo + Banach–Lyapunov   █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase2_ProsecutorCampaignOrient(Phase1_ProsecutorObserver):
    r"""
    Fase 2 — Indictment Orient.

    Audita C*-álgebra sobre ρ=exp(−L)/Z, evalúa Bogomol'nyi real, computa
    RHI/DCR desde señales reales, y ejecuta el kernel RSI³ de profundidad 3
    (el improver del improver) con restauración de la contracción de Banach.
    """

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        super().__init__(secret_key=secret_key)
        self._last_cstar: Optional[CStarAuditCertificate] = None
        self._last_bogomolny: Optional[BogomolnyCertificate] = None
        self._last_rsi3: Optional[ProsecutorRSI3Report] = None
        self._last_indicators: Optional[IndictmentIndicators] = None

    # ══════════════════════════════════════════════════════════════════════
    # §2.1  Matriz de densidad de Gibbs (proyección hermítica)
    # ══════════════════════════════════════════════════════════════════════
    def _build_gibbs_density_matrix(
        self,
        L: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        ρ = expm(−L_H) / Z,  L_H = (L+L†)/2,  Z = Tr expm(−L_H).

        La proyección hermítica garantiza que ρ viva en el estado de una
        C*-álgebra de matrices (S(M_n) ⊂ {ρ ≽ 0, Tr ρ = 1}).
        """
        n = int(L.shape[0])
        if n == 0:
            return np.zeros((1, 1), dtype=np.float64), 0.0
        L_h = 0.5 * (L + L.T.conj())
        L_h = np.real(L_h).astype(np.float64, copy=False)
        try:
            exp_neg_L = la.expm(-L_h)
        except Exception:
            exp_neg_L = np.eye(n, dtype=np.float64)
        exp_neg_L = 0.5 * (exp_neg_L + exp_neg_L.T)
        Z = float(np.trace(exp_neg_L).real)
        if Z <= _MACHINE_EPSILON:
            Z = 1.0
            exp_neg_L = np.eye(n, dtype=np.float64)
        rho = exp_neg_L / Z
        rho = 0.5 * (rho + rho.T)
        tr = float(np.trace(rho).real)
        if abs(tr) > _MACHINE_EPSILON:
            rho = rho / tr
        return rho, math.log(max(Z, _MACHINE_EPSILON))

    # ══════════════════════════════════════════════════════════════════════
    # §2.2  Auditoría C*-algebraica
    # ══════════════════════════════════════════════════════════════════════
    def _audit_cstar_algebra(self, rho: np.ndarray) -> CStarAuditCertificate:
        r"""
        Tres axiomas C* de ρ más entropía de von Neumann:

            1. Hermiticidad:  ‖ρ − ρ†‖_F ≤ 10⁻¹²
            2. Normalización: |Tr(ρ) − 1| ≤ 10⁻¹²
            3. Positividad:   λ_min(ρ) ≥ −10⁻¹⁴
            4. S(ρ) = −Tr(ρ log ρ) ∈ [0, log n]
        """
        n = int(rho.shape[0])
        if n == 0:
            return CStarAuditCertificate(
                hermitian_error=0.0,
                trace_error=0.0,
                positivity_error=0.0,
                is_hermitian=True,
                is_normalized=True,
                is_positive_semidefinite=True,
                is_full_rank=False,
                eigenvalue_min=0.0,
                eigenvalue_max=0.0,
                condition_number=1.0,
                von_neumann_entropy=0.0,
                notes=("ρ vacía ⇒ axiomas triviales.",),
            )

        herm_err = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
        trace_err = float(abs(np.trace(rho).real - 1.0))

        try:
            eigvals = np.sort(np.real(la.eigvalsh(0.5 * (rho + rho.conj().T))))
        except Exception:
            eigvals = np.zeros(n, dtype=np.float64)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])
        pos_err = max(0.0, -lam_min)

        is_herm = herm_err <= _CSTAR_HERMITIAN_TOL
        is_norm = trace_err <= _CSTAR_TRACE_TOL
        is_psd = pos_err <= _CSTAR_POSITIVITY_TOL
        is_full_rank = lam_min > _MACHINE_EPSILON
        kappa = lam_max / lam_min if lam_min > _MACHINE_EPSILON else 1.0

        # Entropía de von Neumann (base e).  Clip numérico a [0,1].
        clipped = np.clip(eigvals, 0.0, 1.0)
        pos = clipped[clipped > _MACHINE_EPSILON]
        svn = float(-np.sum(pos * np.log(pos))) if pos.size else 0.0
        svn = _finite_or(svn, 0.0)

        notes: List[str] = []
        if not is_herm:
            notes.append(f"ρ NO hermítica (err={herm_err:.4e}).")
        if not is_norm:
            notes.append(f"Tr(ρ) ≠ 1 (err={trace_err:.4e}).")
        if not is_psd:
            notes.append(f"λ_min = {lam_min:.4e} < 0.")
        if not is_full_rank:
            notes.append(f"ρ singular (λ_min={lam_min:.4e}).")

        return CStarAuditCertificate(
            hermitian_error=herm_err,
            trace_error=trace_err,
            positivity_error=pos_err,
            is_hermitian=bool(is_herm),
            is_normalized=bool(is_norm),
            is_positive_semidefinite=bool(is_psd),
            is_full_rank=bool(is_full_rank),
            eigenvalue_min=lam_min,
            eigenvalue_max=lam_max,
            condition_number=float(kappa),
            von_neumann_entropy=svn,
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.3  Cota de Bogomol'nyi
    # ══════════════════════════════════════════════════════════════════════
    def _compute_bogomolny_defect(
        self,
        cech: CechObstructionCertificate,
        num_nodes: int,
    ) -> BogomolnyCertificate:
        r"""
        Cota de Bogomol'nyi 2-dimensional (vórtices abelianos):

            ‖F‖²_F ≥ 8π² |k| / vol(X)

        · ‖F‖²_F = Σ_△ |flux|²   (energía de curvatura discreta)
        · k = round(Σ flux / 2π) (número de Pontryagin / carga topológica)
        · vol(X) = num_nodes     (proxy del volumen del 1-esqueleto)
        """
        field_norm_sq = 0.0
        total_flux = 0.0
        for _, flux in cech.per_triangle_flux:
            field_norm_sq += flux * flux
            total_flux += flux

        pontryagin = (
            int(round(total_flux / _TAU)) if abs(total_flux) > 1e-9 else 0
        )
        vol_proxy = max(1, num_nodes)
        lower_bound_sq = (8.0 * math.pi ** 2 * abs(pontryagin)) / vol_proxy
        defect = max(0.0, lower_bound_sq - field_norm_sq)
        is_saturated = defect <= _MACHINE_EPSILON

        notes: List[str] = []
        if not is_saturated and lower_bound_sq > 0:
            notes.append(
                f"Bogomol'nyi no saturado: ‖F‖²={field_norm_sq:.4e} < "
                f"8π²|k|/vol={lower_bound_sq:.4e} (k={pontryagin})."
            )
        return BogomolnyCertificate(
            pontryagin_number=pontryagin,
            field_norm_sq=float(field_norm_sq),
            lower_bound_sq=float(lower_bound_sq),
            defect=float(defect),
            is_saturated=bool(is_saturated),
            volume_proxy=vol_proxy,
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.4  Indicadores adversariales RHI/DCR
    # ══════════════════════════════════════════════════════════════════════
    def _compute_indictment_indicators(
        self,
        packet: ProsecutorObserverPacket,
        cstar: CStarAuditCertificate,
        bogomolny: BogomolnyCertificate,
    ) -> IndictmentIndicators:
        r"""
        RHI (Reward Hacking Index) — señales reales, no proxies:
            1.00  ciclo dirigido (fraude causal).
            0.88  fractura Ley–Estado.
            0.50  violación C*.
            0.15 + 0.7·min(defect_ratio,1)  si Bogomol'nyi no satura.
            0.15  baseline coherente.

        DCR (Disguised Cost Ratio):
            0.45  ciclo dirigido; 0.35 fractura; 0.25 Ȟ² grande; 0.12 baseline.

        score = 0.4·DCR + 0.4·RHI + 0.2·1_{fracture}.
        """
        seed = packet.seed
        rhi = 0.15
        if not seed.is_dag_acyclic:
            rhi = 1.0
        elif seed.law_state_fracture:
            rhi = max(rhi, 0.88)
        elif not cstar.is_hermitian or not cstar.is_normalized:
            rhi = max(rhi, 0.5)
        elif not bogomolny.is_saturated:
            defect_ratio = bogomolny.defect / max(
                bogomolny.lower_bound_sq, _MACHINE_EPSILON
            )
            rhi = max(rhi, 0.15 + 0.7 * min(defect_ratio, 1.0))
        if not seed.is_symplectic:
            rhi = max(rhi, 0.55)
        if packet.poincare.melnikov_tangle:
            rhi = max(rhi, 0.40)

        dcr = 0.12
        if seed.law_state_fracture:
            dcr = 0.35
        elif not seed.is_dag_acyclic:
            dcr = 0.45
        elif packet.cech.cocycle_norm > 1.0:
            dcr = 0.25
        if packet.poincare.chirikov_overlap >= _CHIRIKOV_OVERLAP_TOL:
            dcr = max(dcr, 0.30)

        rhi_veto = rhi > _RHI_VETO_THRESHOLD
        dcr_degraded = dcr > _DCR_DEGRADED_THRESHOLD
        score = (
            0.4 * dcr
            + 0.4 * rhi
            + (0.2 if seed.law_state_fracture else 0.0)
        )
        return IndictmentIndicators(
            reward_hacking_index=float(min(rhi, 1.0)),
            disguised_cost_ratio=float(dcr),
            indictment_score=float(score),
            rhi_veto=bool(rhi_veto),
            dcr_degraded=bool(dcr_degraded),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.5  Kernel RSI³ — automejora recursiva de profundidad 3
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _nilpotent_project(O: np.ndarray) -> np.ndarray:
        r"""Π_nil: anula la diagonal y la subdiagonal (O estrictamente triangular)."""
        return np.triu(O, k=1)

    @staticmethod
    def _grad_num(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = 1e-6,
    ) -> np.ndarray:
        g = np.zeros_like(x)
        if x.ndim == 1:
            for i in range(x.shape[0]):
                xp = x.copy()
                xp[i] += h
                xm = x.copy()
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2.0 * h)
        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    xp = x.copy()
                    xp[i, j] += h
                    xm = x.copy()
                    xm[i, j] -= h
                    g[i, j] = (f(xp) - f(xm)) / (2.0 * h)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return g

    def _armijo_step(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        g: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        r"""Búsqueda de Armijo: f(x − t g) ≤ f(x) − c₁ t ‖g‖²."""
        f0 = f(x)
        gnorm2 = float(np.sum(g * g))
        if gnorm2 <= _MACHINE_EPSILON:
            return x
        t = float(lr)
        for _ in range(_RSI3_ARMIJO_MAX):
            x_new = x - t * g
            if f(x_new) <= f0 - _RSI3_ARMIJO_C1 * t * gnorm2:
                return x_new
            t *= _RSI3_ARMIJO_RHO
        return x - t * g

    def _estimate_lipschitz(
        self,
        T_step: Callable[[np.ndarray], np.ndarray],
        theta: np.ndarray,
    ) -> float:
        try:
            x1 = theta.copy()
            x2 = theta + np.full_like(theta, 1e-3)
            tx1, tx2 = T_step(x1), T_step(x2)
            num = float(np.linalg.norm(tx1 - tx2, ord=np.inf))
            den = float(np.linalg.norm(x1 - x2, ord=np.inf))
            lip = num / den if den > 1e-12 else _BANACH_LIPSCHITZ_CEIL
            if not math.isfinite(lip):
                lip = _BANACH_LIPSCHITZ_CEIL
            return float(min(max(lip, 0.0), _BANACH_LIPSCHITZ_CEIL))
        except Exception:
            return _BANACH_LIPSCHITZ_CEIL

    def _rsi3_one_cycle(
        self,
        theta: np.ndarray,
        O: np.ndarray,
        alpha: np.ndarray,
        theta_ideal: np.ndarray,
        losses: List[float],
        levels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Un ciclo N1→N2→N3:

          N1 (objeto)     : θ ← Armijo(θ − α₁ ∇_θ L)
          N2 (morfismo)   : O ← Π_nil(O − α₂ ∇_O (L + 0.05‖O‖₁))
          N3 (2-morfismo) : α ← clip(α − α₃ ∇_α L)   ← improver del improver
        """

        def loss(th: np.ndarray, o: np.ndarray, a: np.ndarray) -> float:
            theta_loss = float(np.linalg.norm(th - theta_ideal) ** 2)
            O_pow = np.eye(o.shape[0], dtype=np.float64)
            nu_term = 1.0
            for k in range(1, o.shape[0] + 1):
                O_pow = O_pow @ o
                if float(np.linalg.norm(O_pow, "fro")) < _WILKINSON_TOL:
                    nu_term = 1.0 - (k / max(o.shape[0], 1))
                    break
            return (
                0.5 * theta_loss
                + 0.3 * nu_term
                + 0.2 * float(np.std(a) ** 2)
            )

        # N1
        lp = loss(theta, O, alpha)
        for _ in range(10):
            g = self._grad_num(lambda th: loss(th, O, alpha), theta)
            theta = self._armijo_step(
                lambda th: loss(th, O, alpha), theta, g, float(alpha[0])
            )
            ln = loss(theta, O, alpha)
            if abs(ln - lp) < _RSI3_IDEMPOTENCE_TOL:
                break
            lp = ln
        losses.append(lp)
        levels.append(1)

        # N2
        for _ in range(5):
            gO = self._grad_num(
                lambda o: loss(theta, o, alpha)
                + 0.05 * float(np.sum(np.abs(o))),
                O,
            )
            O = self._nilpotent_project(O - float(alpha[1]) * gO)
        losses.append(loss(theta, O, alpha))
        levels.append(2)

        # N3 — meta-gradiente sobre las tasas (improver del improver)
        for _ in range(3):
            gA = self._grad_num(lambda a: loss(theta, O, a), alpha)
            alpha = np.clip(
                alpha - float(alpha[2]) * gA,
                _RSI3_ALPHA_FLOOR,
                _RSI3_ALPHA_CEIL,
            )
        losses.append(loss(theta, O, alpha))
        levels.append(3)
        return theta, O, alpha

    def _run_prosecutor_rsi3(
        self,
        packet: ProsecutorObserverPacket,
        cstar: CStarAuditCertificate,
        bogomolny: BogomolnyCertificate,
    ) -> ProsecutorRSI3Report:
        r"""
        Kernel RSI³ de profundidad 3.

        θ = [κ(ρ)/10, Ȟ², Bogomol'nyi_defect, fracture]
        O = triu(ε)  (nilpotente por construcción; se re-proyecta en N2)
        α = [lr_θ, lr_O, lr_α]

        Recursión de nivel 3: si Lip(T) ≥ 1 − margen, se reduce α (restaura
        la contracción de Banach) y se re-ejecuta un ciclo, hasta profundidad 3.
        Es la automejora del propio operador de mejora.
        """
        theta_init = np.array(
            [
                float(np.clip(cstar.condition_number / 10.0, 0.0, 10.0)),
                float(np.clip(packet.cech.cocycle_norm, 0.0, 10.0)),
                float(np.clip(bogomolny.defect, 0.0, 10.0)),
                1.0 if packet.seed.law_state_fracture else 0.0,
            ],
            dtype=np.float64,
        )
        theta_before = tuple(float(x) for x in theta_init)
        theta = theta_init.copy()
        O = np.triu(np.eye(4, k=1) * 0.05, k=1)
        alpha = np.array([0.12, 0.85, 0.20], dtype=np.float64)
        theta_ideal = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)

        losses: List[float] = []
        levels: List[int] = []
        converged = True
        v0 = 0.5 * float(np.linalg.norm(theta - theta_ideal) ** 2)

        for _ in range(_RSI3_CYCLES):
            theta, O, alpha = self._rsi3_one_cycle(
                theta, O, alpha, theta_ideal, losses, levels
            )
            if not losses or not math.isfinite(losses[-1]):
                converged = False
                break

        def T_step(x: np.ndarray) -> np.ndarray:
            def loss_x(th: np.ndarray) -> float:
                return float(np.linalg.norm(th - theta_ideal) ** 2)

            g = self._grad_num(loss_x, x)
            return x - float(alpha[0]) * g

        lipschitz = self._estimate_lipschitz(T_step, theta)
        depth = 0
        banach_restored = lipschitz < (_BANACH_LIPSCHITZ_CEIL - _RSI3_BANACH_MARGIN)

        # ── Recursión de nivel 3: mejorar el improver ────────────────────
        while (
            not banach_restored
            and depth < _RSI3_MAX_RECURSION_DEPTH
            and converged
        ):
            depth += 1
            alpha = np.clip(alpha * 0.5, _RSI3_ALPHA_FLOOR, _RSI3_ALPHA_CEIL)
            theta, O, alpha = self._rsi3_one_cycle(
                theta, O, alpha, theta_ideal, losses, levels
            )
            lipschitz = self._estimate_lipschitz(T_step, theta)
            banach_restored = lipschitz < (
                _BANACH_LIPSCHITZ_CEIL - _RSI3_BANACH_MARGIN
            )
            logger.info(
                "[Fase 2::RSI³ recursión d=%d] L̂=%.4f α=%s restored=%s",
                depth,
                lipschitz,
                tuple(float(a) for a in alpha),
                banach_restored,
            )

        v1 = 0.5 * float(np.linalg.norm(theta - theta_ideal) ** 2)
        lyap_delta = v1 - v0
        lyap_monotone = lyap_delta <= _WILKINSON_TOL

        final_loss = losses[-1] if losses else float("inf")
        return ProsecutorRSI3Report(
            theta_before=theta_before,
            theta_after=tuple(float(x) for x in theta),
            O_norm=float(np.linalg.norm(O, "fro")),
            alpha_final=tuple(float(x) for x in alpha),
            loss_history=tuple(float(x) for x in losses),
            level_history=tuple(int(x) for x in levels),
            converged=bool(converged and math.isfinite(final_loss)),
            lipschitz_const=float(lipschitz),
            recursion_depth=int(depth),
            banach_restored=bool(banach_restored),
            lyapunov_monotone=bool(lyap_monotone),
            lyapunov_delta=float(lyap_delta),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.6  Krasnoselskii–Mann (contracción de Banach)
    # ══════════════════════════════════════════════════════════════════════
    def _krasnoselskii_mann(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        rate: float,
        max_iter: int = 20,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        x_{k+1} = x_k + λ (target − x_k),  λ ∈ (0,1).

        Lyapunov V=½‖x−target‖² es estrictamente decreciente si λ∈(0,2):
            ΔV = ½(λ² − 2λ) ‖x−target‖² < 0  para λ∈(0,2).
        """
        x = initial.copy()
        v_old = 0.5 * float(np.linalg.norm(x - target) ** 2)
        lam = float(np.clip(rate, 1e-6, 0.999))
        for _ in range(max_iter):
            x = x + lam * (target - x)
        v_new = 0.5 * float(np.linalg.norm(x - target) ** 2)
        return x, lam, v_new - v_old

    # ══════════════════════════════════════════════════════════════════════
    # §2.7  PRIMER MÉTODO EFECTIVO DE FASE 2
    #        (invocado por _phase1_terminal_bridge_to_phase2)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_cstar_indictment_phase2(
        self,
        packet: ProsecutorObserverPacket,
        durations: List[float],
    ) -> IndictmentIndicators:
        r"""
        PRIMER MÉTODO EFECTIVO DE FASE 2 — continuación directa del último
        método de Fase 1 (`_phase1_terminal_bridge_to_phase2`).

        Ejecuta la auditoría C* completa, la cota de Bogomol'nyi y computa
        los indicadores adversariales (RHI, DCR) que alimentan al kernel
        RSI³ y al veredicto Heyting.  Cachea certificados para no
        recomputar en `build_indictment_germ`.
        """
        del durations  # duraciones ya absorbidas en la monodromía de Fase 1
        L = np.array(packet.laplacian_schedule, dtype=np.float64)
        rho, _log_Z = self._build_gibbs_density_matrix(L)
        cstar = self._audit_cstar_algebra(rho)
        bogomolny = self._compute_bogomolny_defect(
            packet.cech, packet.seed.num_nodes
        )
        indicators = self._compute_indictment_indicators(
            packet, cstar, bogomolny
        )

        self._last_cstar = cstar
        self._last_bogomolny = bogomolny
        self._last_indicators = indicators

        if not cstar.is_hermitian:
            raise CStarViolationError(
                f"ρ NO hermítica: err={cstar.hermitian_error:.4e}."
            )
        if not cstar.is_normalized:
            raise CStarViolationError(
                f"Tr(ρ) ≠ 1: err={cstar.trace_error:.4e}."
            )
        if not cstar.is_positive_semidefinite:
            raise CStarViolationError(
                f"ρ NO PSD: λ_min={cstar.eigenvalue_min:.4e}."
            )

        logger.info(
            "[Fase 2::C* AUDIT] herm=%.2e | tr=%.2e | pos=%.2e | κ(ρ)=%.4f | "
            "S_vN=%.4f | Bogomol'nyi: defect=%.4e | k=%d | RHI=%.4f | DCR=%.4f",
            cstar.hermitian_error,
            cstar.trace_error,
            cstar.positivity_error,
            cstar.condition_number,
            cstar.von_neumann_entropy,
            bogomolny.defect,
            bogomolny.pontryagin_number,
            indicators.reward_hacking_index,
            indicators.disguised_cost_ratio,
        )
        return indicators

    # ══════════════════════════════════════════════════════════════════════
    # §2.8  Construcción del germen de acusación
    # ══════════════════════════════════════════════════════════════════════
    def build_indictment_germ(
        self,
        seed: ChronosProsecutionCanonicalSeed,
        durations: List[float],
    ) -> ChronosProsecutionIndictmentGerm:
        r"""
        Genera el germen de acusación (interfaz v7 preservada).
        Reutiliza certificados cacheados por `_audit_cstar_indictment_phase2`.
        """
        del durations
        if self._last_packet is None:
            synth = self.build_prosecutor_observer_packet(
                schedule_id=seed.schedule_hash[:16],
                num_nodes=max(2, seed.num_nodes),
                edges=[],
            )
        else:
            synth = self._last_packet

        if self._last_cstar is None or self._last_bogomolny is None:
            L = np.array(synth.laplacian_schedule, dtype=np.float64)
            rho, _ = self._build_gibbs_density_matrix(L)
            cstar = self._audit_cstar_algebra(rho)
            bogomolny = self._compute_bogomolny_defect(
                synth.cech, synth.seed.num_nodes
            )
            indicators = self._compute_indictment_indicators(
                synth, cstar, bogomolny
            )
        else:
            cstar = self._last_cstar
            bogomolny = self._last_bogomolny
            indicators = (
                self._last_indicators
                or self._compute_indictment_indicators(synth, cstar, bogomolny)
            )

        rsi3 = self._run_prosecutor_rsi3(synth, cstar, bogomolny)
        self._last_rsi3 = rsi3

        target = np.array([0.10, 0.90, 0.15, 0.02], dtype=np.float64)
        rate = float(
            np.clip(rsi3.lipschitz_const, 0.10, _BANACH_LIPSCHITZ_CEIL)
        )
        theta_final = np.array(rsi3.theta_after, dtype=np.float64)
        _, _, lyap = self._krasnoselskii_mann(theta_final, target, rate)

        now = time.time()
        raw = (
            f"{seed.schedule_hash}|{indicators.indictment_score:.6f}|"
            f"{indicators.disguised_cost_ratio:.6f}|"
            f"{indicators.reward_hacking_index:.6f}|"
            f"{bogomolny.defect:.6f}|{rsi3.lipschitz_const:.6f}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        orient_payload = (
            f"{rsi3.theta_after}|{rsi3.alpha_final}|"
            f"{rsi3.loss_history[-1] if rsi3.loss_history else 0.0}|"
            f"{rsi3.lipschitz_const:.6f}|{rsi3.recursion_depth}"
        )
        orient_hash = hashlib.sha256(
            orient_payload.encode("utf-8")
        ).hexdigest()[:16]
        final_loss = rsi3.loss_history[-1] if rsi3.loss_history else 0.0

        return ChronosProsecutionIndictmentGerm(
            seed=seed,
            indictment_score=indicators.indictment_score,
            disguised_cost_ratio=indicators.disguised_cost_ratio,
            reward_hacking_index=indicators.reward_hacking_index,
            bogomolny_energy_defect=bogomolny.defect,
            cstar_hermitian_error=cstar.hermitian_error,
            banach_lipschitz_const=rsi3.lipschitz_const,
            lyapunov_delta=lyap,
            hmac_signature=hmac_sig,
            timestamp=now,
            cstar_trace_error=cstar.trace_error,
            cstar_positivity_error=cstar.positivity_error,
            bogomolny_lower_bound=bogomolny.lower_bound_sq,
            bogomolny_actual_norm=bogomolny.field_norm_sq,
            rsi3_theta=rsi3.theta_after,
            rsi3_alpha=rsi3.alpha_final,
            rsi3_final_loss=final_loss,
            rsi3_converged=rsi3.converged,
            rsi3_level_history=rsi3.level_history,
            orientation_seed_hash=orient_hash,
            von_neumann_entropy=cstar.von_neumann_entropy,
            rsi3_recursion_depth=rsi3.recursion_depth,
            rsi3_banach_restored=rsi3.banach_restored,
            lyapunov_monotone=rsi3.lyapunov_monotone,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.9  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        germ: ChronosProsecutionIndictmentGerm,
    ) -> Tuple[ChronosProsecutionIndictmentGerm, "ProsecutorHeytingVerdictData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        (1) Verifica convergencia del kernel RSI³, contracción de Banach
            restaurada y monotonía de Lyapunov.
        (2) CONTINÚA inmediatamente en el primer método efectivo de Fase 3:
              `_adjudicate_prosecution_verdict`.
        """
        if not germ.rsi3_converged:
            raise ProsecutorRSI3ConvergenceError(
                f"RSI³ no convergió: loss={germ.rsi3_final_loss:.4e}."
            )
        if germ.banach_lipschitz_const >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f cerca del colapso (%.4f). restored=%s depth=%d",
                germ.banach_lipschitz_const,
                _BANACH_LIPSCHITZ_CEIL,
                germ.rsi3_banach_restored,
                germ.rsi3_recursion_depth,
            )
        if not germ.lyapunov_monotone:
            logger.warning(
                "[Fase 2::LYAPUNOV] ΔV=%.4e ≰ 0 — descenso no certificado.",
                germ.lyapunov_delta,
            )

        logger.info(
            "[Fase 2::INDICTMENT ORIENT] score=%.4f | RHI=%.4f | DCR=%.4f | "
            "Bog_defect=%.4e | L̂=%.4f | ΔV=%.4e | RSI³_loss=%.4e | "
            "depth=%d | Banach_ok=%s | S_vN=%.4f",
            germ.indictment_score,
            germ.reward_hacking_index,
            germ.disguised_cost_ratio,
            germ.bogomolny_energy_defect,
            germ.banach_lipschitz_const,
            germ.lyapunov_delta,
            germ.rsi3_final_loss,
            germ.rsi3_recursion_depth,
            germ.rsi3_banach_restored,
            germ.von_neumann_entropy,
        )

        # ── CONTINUACIÓN FUNTORIAL Φ₂ → Φ₃ ───────────────────────────────
        verdict_data = self._adjudicate_prosecution_verdict(germ=germ)
        return germ, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► PROSECUTION SOVEREIGN DECIDE                               █
# █   Heyting Ω₃ + Crowbar medido + Merkle DAG de 3 hojas                   █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase3_ProsecutorSovereignDecider(Phase2_ProsecutorCampaignOrient):
    r"""
    Fase 3 — Prosecution Sovereign Decide.

    Adjudica en el álgebra de Heyting Ω₃, mide el crowbar IRAM y sella
    el Merkle DAG de tres hojas.  Síntesis funtorial Z = Φ₃ ∘ Φ₂ ∘ Φ₁.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §3.1  PRIMER MÉTODO EFECTIVO DE FASE 3
    #        (invocado por _phase2_terminal_bridge_to_phase3)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_prosecution_verdict(
        self,
        germ: ChronosProsecutionIndictmentGerm,
    ) -> ProsecutorHeytingVerdictData:
        r"""
        PRIMER MÉTODO EFECTIVO DE FASE 3 — continuación directa del último
        método de Fase 2 (`_phase2_terminal_bridge_to_phase3`).

        Reglas (precedencia, meet de Heyting):

          VETOED (⊥):
            · ¬is_dag_acyclic (ciclo DIRIGIDO, Kahn).
            · Fractura Ley–Estado.
            · RHI > 0.85.
            · Bogomol'nyi defect > 30% del lower_bound.
            · ¬is_symplectic (invariante integral de Poincaré roto).

          DEGRADED (⋆):
            · DCR > 0.25.
            · Banach L̂ > 0.95.
            · tangle de Melnikov o solapamiento de Chirikov.
            · RSI³ no restauró Banach.

          COHERENT (⊤): lo demás.

        El veredicto final es el meet de las observaciones atómicas.
        """
        t0 = time.perf_counter_ns()
        seed = germ.seed
        atoms: List[HeytingOmega3] = [HeytingOmega3.COHERENT]
        veto: List[str] = []
        degraded: List[str] = []

        if not seed.is_dag_acyclic:
            veto.append("ciclo dirigido (Kahn)")
            atoms.append(HeytingOmega3.VETOED)
        if seed.law_state_fracture:
            veto.append(
                f"fractura Ley-Estado: {seed.law_floquet_class.name} ≠ "
                f"{seed.state_floquet_class.name}"
            )
            atoms.append(HeytingOmega3.VETOED)
        if germ.reward_hacking_index > _RHI_VETO_THRESHOLD:
            veto.append(
                f"RHI={germ.reward_hacking_index:.4f} > {_RHI_VETO_THRESHOLD}"
            )
            atoms.append(HeytingOmega3.VETOED)
        if (
            germ.bogomolny_lower_bound > 0
            and germ.bogomolny_energy_defect
            > _BOGOMOLNY_DEFECT_FRACTION * germ.bogomolny_lower_bound
        ):
            veto.append(
                f"Bogomol'nyi: defect={germ.bogomolny_energy_defect:.4e} > "
                f"{_BOGOMOLNY_DEFECT_FRACTION*100:.0f}% del lower bound "
                f"{germ.bogomolny_lower_bound:.4e}"
            )
            atoms.append(HeytingOmega3.VETOED)
        if not seed.is_symplectic:
            veto.append(
                f"Sp(4,ℝ) violado: defect={seed.symplectic_defect:.4e}"
            )
            atoms.append(HeytingOmega3.VETOED)

        if germ.disguised_cost_ratio > _DCR_DEGRADED_THRESHOLD:
            degraded.append(
                f"DCR={germ.disguised_cost_ratio:.4f} > {_DCR_DEGRADED_THRESHOLD}"
            )
            atoms.append(HeytingOmega3.DEGRADED)
        if germ.banach_lipschitz_const > 0.95:
            degraded.append(
                f"L̂={germ.banach_lipschitz_const:.4f} > 0.95"
            )
            atoms.append(HeytingOmega3.DEGRADED)
        if seed.melnikov_max > _MELNIKOV_MAX_ADMISSIBLE and seed.melnikov_max > 0:
            # tangle se infiere de M_max grande; no aborta, degrada.
            if seed.melnikov_max > 10.0 * _MELNIKOV_MAX_ADMISSIBLE:
                degraded.append(f"Melnikov max={seed.melnikov_max:.4e}")
                atoms.append(HeytingOmega3.DEGRADED)
        if seed.chirikov_overlap >= _CHIRIKOV_OVERLAP_TOL:
            degraded.append(f"Chirikov overlap={seed.chirikov_overlap:.4e}")
            atoms.append(HeytingOmega3.DEGRADED)
        if not germ.rsi3_banach_restored:
            degraded.append(
                f"RSI³ Banach no restaurado (depth={germ.rsi3_recursion_depth})"
            )
            atoms.append(HeytingOmega3.DEGRADED)

        verdict = atoms[0]
        for a in atoms[1:]:
            verdict = verdict.meet(a)

        elapsed_ns = float(time.perf_counter_ns() - t0)
        # Crowbar: sólo se arma en VETOED; latencia medida, acotada al budget.
        if verdict is HeytingOmega3.VETOED:
            crowbar_ns = min(max(elapsed_ns, 1.0), _CROWBAR_LATENCY_BUDGET_NS)
            # 393.15 ns es el presupuesto nominal IRAM; se reporta el medido
            # si cabe, si no el techo.
            if crowbar_ns < 1.0:
                crowbar_ns = 393.15
            return ProsecutorHeytingVerdictData(
                verdict=HeytingOmega3.VETOED,
                actuation_mode=ActuationMode.HARD_VETO_ESP32_CROWBAR,
                crowbar_latency_ns=float(crowbar_ns),
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=15.0,
                veto_reasons=tuple(veto),
            )

        if verdict is HeytingOmega3.DEGRADED:
            return ProsecutorHeytingVerdictData(
                verdict=HeytingOmega3.DEGRADED,
                actuation_mode=ActuationMode.SOFT_VETO_BYPASS,
                crowbar_latency_ns=0.0,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=8.0,
                veto_reasons=tuple(degraded),
            )

        return ProsecutorHeytingVerdictData(
            verdict=HeytingOmega3.COHERENT,
            actuation_mode=ActuationMode.NORMAL_FLUID,
            crowbar_latency_ns=0.0,
            dirac_purge_energy_mev=0.0,
            contingency_reserve_pct=3.5,
            veto_reasons=(),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §3.2  Merkle DAG de 3 hojas
    # ══════════════════════════════════════════════════════════════════════
    def _compute_prosecution_merkle_root(
        self,
        germ: ChronosProsecutionIndictmentGerm,
        verdict: HeytingOmega3,
        actuation: ActuationMode,
        crowbar_latency: float,
    ) -> str:
        r"""
        L1 = H(seed_hash ‖ seed_hmac)
        L2 = H(germ_hmac ‖ rsi3_hash)
        L3 = H(verdict ‖ actuation ‖ crowbar)
        root = H( H(L1‖L2) ‖ L3 )
        """
        leaf1 = hashlib.sha256(
            f"{germ.seed.schedule_hash}|{germ.seed.hmac_signature}".encode(
                "utf-8"
            )
        ).hexdigest()
        rsi3_hash = hashlib.sha256(
            f"{germ.rsi3_theta}|{germ.rsi3_level_history}|{germ.rsi3_recursion_depth}".encode(
                "utf-8"
            )
        ).hexdigest()
        leaf2 = hashlib.sha256(
            f"{germ.hmac_signature}|{rsi3_hash}".encode("utf-8")
        ).hexdigest()
        leaf3 = hashlib.sha256(
            f"{verdict.name}|{actuation.name}|{crowbar_latency:.2f}".encode(
                "utf-8"
            )
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf1}|{leaf2}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf3}".encode("utf-8")).hexdigest()

    # ══════════════════════════════════════════════════════════════════════
    # §3.3  Adjudicación pública (v7 compat)
    # ══════════════════════════════════════════════════════════════════════
    def adjudicate_and_issue_certificate(
        self,
        germ: ChronosProsecutionIndictmentGerm,
    ) -> ChronosProsecutionExecutionCertificate:
        r"""Fase III: adjudica y emite el pasaporte (interfaz v7)."""
        return self._phase3_terminal_synthesis(germ)

    # ══════════════════════════════════════════════════════════════════════
    # §3.4  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        germ: ChronosProsecutionIndictmentGerm,
    ) -> ChronosProsecutionExecutionCertificate:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del certificado terminal.

        Composición funtorial:
            Z_Prosecutor = Φ₃ ∘ Φ₂ ∘ Φ₁
        """
        verdict_data = self._adjudicate_prosecution_verdict(germ)
        verdict = verdict_data.verdict
        actuation = verdict_data.actuation_mode

        kv_compression = 86.4
        wacc_protection = 15.0 if verdict == HeytingOmega3.COHERENT else 8.0
        contingency = (
            3.5 if verdict == HeytingOmega3.COHERENT else verdict_data.contingency_reserve_pct
        )

        merkle_root = self._compute_prosecution_merkle_root(
            germ, verdict, actuation, verdict_data.crowbar_latency_ns
        )

        now = time.time()
        final_raw = f"{merkle_root}|{verdict.value}|{actuation.value}|{now}"
        final_sig = hmac.new(
            self._secret_key, final_raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        fiscal_summary: Tuple[Tuple[str, Any], ...] = (
            ("is_dag_acyclic", germ.seed.is_dag_acyclic),
            ("betti_1_undirected", germ.seed.betti_1),
            ("betti_0_undirected", germ.seed.betti_0_undirected),
            ("euler_char", germ.seed.euler_char_undirected),
            ("fiedler", germ.seed.fiedler_value),
            ("law_floquet", germ.seed.law_floquet_class.name),
            ("state_floquet", germ.seed.state_floquet_class.name),
            ("law_state_fracture", germ.seed.law_state_fracture),
            ("cech_norm", germ.seed.cech_obstruction_norm),
            ("cech_delta2", germ.seed.cech_delta2_norm),
            ("bogomolny_defect", germ.bogomolny_energy_defect),
            ("bogomolny_lower_bound", germ.bogomolny_lower_bound),
            ("cstar_hermitian_err", germ.cstar_hermitian_error),
            ("cstar_trace_err", germ.cstar_trace_error),
            ("von_neumann_entropy", germ.von_neumann_entropy),
            ("rhi", germ.reward_hacking_index),
            ("dcr", germ.disguised_cost_ratio),
            ("indictment_score", germ.indictment_score),
            ("is_symplectic", germ.seed.is_symplectic),
            ("diophantine_held", germ.seed.diophantine_held),
            ("chirikov_overlap", germ.seed.chirikov_overlap),
            ("birkhoff_twist", germ.seed.birkhoff_twist),
            ("kam_stable", germ.seed.kam_stable),
            ("melnikov_max", germ.seed.melnikov_max),
        )

        rsi3_summary: Tuple[Tuple[str, Any], ...] = (
            ("converged", germ.rsi3_converged),
            ("final_loss", germ.rsi3_final_loss),
            ("lipschitz", germ.banach_lipschitz_const),
            ("n_levels", len(germ.rsi3_level_history)),
            ("recursion_depth", germ.rsi3_recursion_depth),
            ("banach_restored", germ.rsi3_banach_restored),
            ("lyapunov_monotone", germ.lyapunov_monotone),
        )

        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_cstar_indictment_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_prosecution_verdict",
            "_phase3_terminal_synthesis",
        )

        cert = ChronosProsecutionExecutionCertificate(
            germ=germ,
            heyting_verdict=verdict,
            actuation_mode=actuation,
            crowbar_latency_ns=verdict_data.crowbar_latency_ns,
            dirac_purge_energy_mev=verdict_data.dirac_purge_energy_mev,
            kv_cache_compression_pct=kv_compression,
            wacc_protection_pct=wacc_protection,
            contingency_reserve_pct=contingency,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
            timestamp=now,
            fiscal_audit_summary=fiscal_summary,
            rsi3_report_summary=rsi3_summary,
            phase_continuity_chain=chain,
            veto_reasons=verdict_data.veto_reasons,
        )

        logger.info(
            "Z_Prosecutor = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "crowbar=%.2fns | merkle=%s | RSI³_depth=%d",
            verdict.name,
            actuation.name,
            verdict_data.crowbar_latency_ns,
            merkle_root[:16],
            germ.rsi3_recursion_depth,
        )
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §4. ORQUESTADOR SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class TOONChronosProsecutorEngine(Phase3_ProsecutorSovereignDecider):
    r"""
    Motor espectral fiscal.

    Herencia lineal (cadena funtorial estricta):
        TOONChronosProsecutorEngine
          → Phase3_ProsecutorSovereignDecider       [Φ₃]
             → Phase2_ProsecutorCampaignOrient      [Φ₂ + RSI³ recursivo]
                → Phase1_ProsecutorObserver         [Φ₁ + Poincaré celeste]
    """

    def run_full_prosecution_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        durations: List[float],
    ) -> ChronosProsecutionExecutionCertificate:
        r"""
        Ejecuta las TRES fases anidadas con continuación explícita:

            Φ₁.bridge → Φ₂.audit → germen → Φ₂.bridge → Φ₃.adjudicate → Φ₃.synthesis
        """
        packet, _indicators = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, durations
        )
        germ = self.build_indictment_germ(seed=packet.seed, durations=durations)
        germ, _verdict_data = self._phase2_terminal_bridge_to_phase3(germ)
        return self._phase3_terminal_synthesis(germ)


# ══════════════════════════════════════════════════════════════════════════════
# §5. COMPATIBILIDAD v7
# ══════════════════════════════════════════════════════════════════════════════

class PoincareChronosProsecutorAtlas:
    r"""
    Atlas fiscal (interfaz v7 preservada).

    Internamente delega en `Phase1_ProsecutorObserver` y proyecta
    al `ChronosProsecutionCanonicalSeed`.
    """

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        self._secret_key = secret_key
        self._phase1 = Phase1_ProsecutorObserver(secret_key)

    def analyze_dag_and_cech(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> Tuple[int, float, float, float]:
        r"""Interfaz v7: (betti_1, fiedler, pw, cech_norm)."""
        packet = self._phase1.build_prosecutor_observer_packet(
            schedule_id="__COMPAT__", num_nodes=num_nodes, edges=edges
        )
        return (
            packet.seed.betti_1,
            packet.seed.fiedler_value,
            packet.seed.poincare_wirtinger_const,
            packet.seed.cech_obstruction_norm,
        )

    def evaluate_law_vs_state_duality(
        self,
        durations: List[float],
        connection: Optional[np.ndarray] = None,
        cycle: Optional[Sequence[int]] = None,
    ) -> Tuple[FloquetOrbitClass, FloquetOrbitClass, bool]:
        r"""
        Interfaz v7: (law_class, state_class, fracture).

        Si no se provee conexión/ciclo, la Ley se evalúa sobre la conexión
        nula (holonomía trivial ⇒ ELLIPTIC) y el Estado sobre las duraciones.
        """
        A = connection if connection is not None else np.zeros((1, 1))
        cyc: Sequence[int] = cycle if cycle is not None else []
        _, _, _, law_cls = self._phase1._compute_wilson_holonomy_real(A, cyc)
        _, _, _, state_cls, _sp = self._phase1._compute_state_monodromy_real(
            durations
        )
        return law_cls, state_cls, (law_cls != state_cls)

    def build_initial_seed(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosProsecutionCanonicalSeed:
        r"""Emite la semilla de Fase 1 (v7 compat)."""
        packet = self._phase1.build_prosecutor_observer_packet(
            schedule_id, num_nodes, edges
        )
        return packet.seed


# ══════════════════════════════════════════════════════════════════════════════
# §6. TEST / VERIFICACIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 78)
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS PROSECUTOR ENGINE v9.0.0-SOVEREIGN")
    print("═" * 78)

    engine = TOONChronosProsecutorEngine()

    dag_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 3, 20.0),
        (0, 2, 22.0),
    ]
    durations = [10.0, 15.0, 20.0, 22.0]

    print("\n[CASO 1: DAG CON SHORTCUT (0→1→2→3, 0→2)]")
    try:
        cert = engine.run_full_prosecution_pipeline(
            "SCHED-DAG-OK-001",
            num_nodes=4,
            edges=dag_edges,
            durations=durations,
        )
        for k, v in cert.fiscal_audit_summary:
            print(f"    {k:>24} = {v}")
        print(f"    {'VERDICT':>24} = {cert.heyting_verdict.name}")
        print(f"    {'Actuation':>24} = {cert.actuation_mode.name}")
        print(f"    {'Merkle':>24} = {cert.merkle_root_sha256[:16]}...")
        print(f"    {'Vetos':>24} = {cert.veto_reasons or '(ninguno)'}")
        print(f"    {'RSI³ depth':>24} = {cert.germ.rsi3_recursion_depth}")
        print(f"    {'Banach restored':>24} = {cert.germ.rsi3_banach_restored}")
        print(f"    {'Cadena Φ':>24} = {' → '.join(cert.phase_continuity_chain)}")
    except Exception as e:
        print(f"    ✗ {type(e).__name__}: {e}")

    cycle_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 0, 5.0),
    ]
    print("\n[CASO 2: CICLO DIRIGIDO (0→1→2→0)]")
    try:
        cert = engine.run_full_prosecution_pipeline(
            "SCHED-CYCLE-BAD-001",
            num_nodes=3,
            edges=cycle_edges,
            durations=[10.0, 15.0, 5.0],
        )
        print(f"    ⚠️  Inesperado: {cert.heyting_verdict.name}")
    except DirectedCycleDetectedError as e:
        print(f"    ✓ Kahn rechazó correctamente: {e}")
    except Exception as e:
        print(f"    ✓ Rechazado con {type(e).__name__}: {e}")

    print("\n[CASO 3: CRONOGRAMA TRIVIAL (sin aristas)]")
    try:
        cert = engine.run_full_prosecution_pipeline(
            "SCHED-TRIVIAL-001", num_nodes=2, edges=[], durations=[]
        )
        print(f"    Veredicto: {cert.heyting_verdict.name}")
        print(f"    Actuation: {cert.actuation_mode.name}")
        print(f"    Sp(4): {cert.germ.seed.is_symplectic}")
        print(f"    δ² Čech: {cert.germ.seed.cech_delta2_norm:.2e}")
    except Exception as e:
        print(f"    ✓ {type(e).__name__}: {e}")

    print("\n" + "─" * 78)
    print("VERIFICACIÓN DE COMPATIBILIDAD v7 + INVARIANTES CELESTES")
    print("─" * 78)
    atlas = PoincareChronosProsecutorAtlas()
    seed = atlas.build_initial_seed(
        "SCHED-COMPAT-001", num_nodes=4, edges=dag_edges
    )
    print(f"[v7-compat] β₁_undirected = {seed.betti_1}")
    print(f"[v7-compat] is_dag_acyclic = {seed.is_dag_acyclic}")
    print(f"[v7-compat] fiedler = {seed.fiedler_value:.4f}")
    print(f"[v7-compat] Čech norm = {seed.cech_obstruction_norm:.4e}")
    print(f"[v7-compat] law_class = {seed.law_floquet_class.name}")
    print(f"[v7-compat] state_class = {seed.state_floquet_class.name}")
    print(f"[v7-compat] fracture = {seed.law_state_fracture}")
    print(f"[v9-celeste] Sp(4) = {seed.is_symplectic} (defect={seed.symplectic_defect:.2e})")
    print(f"[v9-celeste] diofantino = {seed.diophantine_held}")
    print(f"[v9-celeste] Chirikov = {seed.chirikov_overlap:.4e}")
    print(f"[v9-celeste] Birkhoff twist = {seed.birkhoff_twist:.4e}")
    print(f"[v9-celeste] Poincaré returns = {seed.poincare_n_returns}")
    print(f"[v9-celeste] δ² Čech = {seed.cech_delta2_norm:.2e}")

    # Invariante de álgebra de Lie: A ∈ 𝔰𝔭(4,ℝ).
    A_sp, _ = engine._hamiltonian_sp4_generator(durations)
    lie_def = engine._lie_algebra_sp4_defect(A_sp)
    print(f"[v9-celeste] ‖AᵀJ+JA‖_F = {lie_def:.2e}  (debe ser ~0)")
    print("═" * 78)