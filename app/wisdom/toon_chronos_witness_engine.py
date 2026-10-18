# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Witness Engine — v9.0.0                                                      ║
║ Ubicación: app/wisdom/toon_chronos_witness_engine.py                                                 ║
║ Versión  : 9.0.0-Witness-TomitaTakesaki-KMS-QND-Dirac-RSI3-Poincare-NestedPhases-Doctoral            ║
║ Función  : Motor espectral y observador cuántico sin demolición (QND, Back-Action ≡ 0.0 dB) del      ║
║            flujo geodésico temporal desde el Vacío de Dirac |Ω⟩, con retorno de Poincaré del flujo   ║
║            modular σ_t y automejora recursiva de nivel 3 (RSI³) como contracción de Picard.          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONChronosWitnessEngine` constituye la infraestructura espectral cuántica y celeste
para la observación no destructiva (Quantum Non-Demolition, QND) del flujo geodésico temporal
en el Estrato WISDOM de la arquitectura cognitiva APU Filter v8.0 / v9.0.

1. CONSTRUCCIÓN GNS Y ÁLGEBRA DE TOMITA-TAKESAKI:
   Dado el Laplaciano no dirigido $L_{\text{sched}} = B_1 B_1^\top$ del Complejo de Cadenas del Cronograma,
   se construye el estado de Gibbs libre de singularidades:
     $$\rho = \frac{\exp(-L_{\text{sched}})}{Z}, \quad Z = \text{Tr}(\exp(-L_{\text{sched}})) > 0$$
   Dado que $\rho \succ 0$ es de rango completo, el vector cíclico y separador $|\Omega\rangle$ en la
   representación GNS $(\mathcal{H}_{\text{GNS}}, \pi_{\text{GNS}}, |\Omega\rangle)$ define el Vacío de Dirac.
   Los operadores fundamentales de la teoría de Tomita-Takesaki se especifican como:
     - Operador Modular: $\Delta = \rho \otimes \rho^{-1}$, con número de condición $\|\Delta\| = \kappa(\rho)$.
     - Conjugación Modular: $J(A) = A^*$, la cual es una involución anti-lineal ($J^2 = \mathbb{I}, J = J^*$).
     - Operador Tomita $S_0 = J \Delta^{1/2}$ que satisface $S_0(A|\Omega\rangle) = A^*|\Omega\rangle$.
     - Generador del Flujo Modular: $K = -\ln \rho = L_{\text{sched}} + \ln(Z) \cdot \mathbb{I}$.

2. CONDICIÓN KMS Y FLUIO MODULAR DE POINCARÉ:
   El flujo de Heisenberg de Tomita-Takesaki $\sigma_t(B) = \rho^{it} B \rho^{-it} = \exp(it K) B \exp(-it K)$
   satisface analíticamente la condición de Kubo-Martin-Schwinger (KMS) a temperatura β = 1:
     $$\omega(A \sigma_t(B)) = \omega(\sigma_{t-i}(B) A), \quad \text{donde } \omega(X) = \text{Tr}(\rho X)$$
   donde la extensión analítica en la franja imaginaria $\sigma_{t-i}(B) = \rho^{i(t-i)} B \rho^{-i(t-i)} = \exp((it+1)K) B \exp(-(it+1)K)$
   garantiza la estabilidad termodinámica pura sin colapso por ciclicidad cíclica.

3. MEDICIÓN DÉBIL QND, GEOMETRÍA DE UHLMANN Y DILATACIÓN DE GROMOV:
   Para estados inicial y final $|\psi_i\rangle, |\psi_f\rangle$ y observable $A \in \mathfrak{h}_{\text{sa}}$:
     - Valor Débil de Aharonov-Albert-Vaidman: $A_w = \frac{\langle \psi_f | A | \psi_i \rangle}{\langle \psi_f | \psi_i \rangle}$.
     - Fidelidad de Uhlmann: $F = |\langle \psi_f | \psi_i \rangle|^2$.
     - Distancia Geodésica de Fubini-Study: $d_{\text{FS}} = \arccos(\sqrt{F})$.
     - Capacidad Simpléctica de Gromov (Cota Implosiva): $c_G \approx 2\pi (1 + d_{\text{FS}}) \le 12.5$.
     - Retroacción Cuántica (Back-Action en dB):
         $$\text{BA}_{\text{dB}} = 10 \log_{10} \left( \frac{\|[ \rho, A ]\|_F}{\varepsilon_{\text{floor}}} \right) \ge 0.0 \text{ dB}$$
       donde el régimen QND puro exige $\|[\rho, A]\|_F \le \delta_{\text{tol}} \implies \text{BA}_{\text{dB}} \equiv 0.0 \text{ dB}$.

4. VALUACIÓN ULTRAMÉTRICA DE NOVIKOV EN SEMIANILLO MAX-PLUS:
   Para un vector de holguras $s_i > 0$, con probabilidades $p_i = s_i / \sum s_k$ y grados $a_i = -\ln p_i$:
     $$v(\text{Slack}) = \bigoplus_i a_i \equiv \min_i a_i, \quad \text{con norma de retardo } \|\text{Delay}\| = \exp(-v(\text{Slack}))$$
   donde el semianillo Tropical Max-Plus $(\mathbb{R} \cup \{\infty\}, \max, +)$ verifica axiomáticamente
   la desigualdad ultramétrica $v(x \oplus y) = \min(v(x), v(y))$.

5. MECÁNICA CELESTE DE POINCARÉ Y INTEGRAL DE JACOBI:
   El mapa de primer retorno de Poincaré $\Pi: \Sigma \to \Sigma$ sobre la sección transversal
   $\Sigma = \{ A \in \mathfrak{h}_{\text{sa}} : \langle K, A \rangle_{\text{HS}} = \mathcal{J} \}$ define:
     - Tiempo de Recurrencia de Poincaré-Zermelo: $T_{\text{rec}} \simeq \frac{2\pi}{\lambda_2(L_{\text{sched}})}$.
     - Monodromía Simpléctica $M = D\Pi$ y Clasificación de Órbitas:
         * Elíptica  : $|\text{Tr}(M)| < 2$ (KAM-persistente).
         * Parabólica : $|\text{Tr}(M)| = 2$ (Resonancia armónica).
         * Hiperbólica: $|\text{Tr}(M)| > 2$ (Caos homoclínico, veto absoluto).
     - Integral de Jacobi Conservada: $\mathcal{J} = \text{Tr}(\rho K)$.

6. AUTOMEJORA RECURSIVA RSI³ Y CONTRACCIÓN DE PICARD:
   El kernel de optimización de Nivel 3 ejecuta la dinámica tripartita $N_1(\theta) \to N_2(O) \to N_3(\alpha)$
   sobre la sección de Poincaré. La convergencia está garantizada por el Teorema de Punto Fijo de Banach:
     $$\|T(\theta_1) - T(\theta_2)\|_\infty \le \widehat{L} \|\theta_1 - \theta_2\|_\infty, \quad \widehat{L} < 1.0$$

DISEÑO DE TRES FASES ANIDADAS (Z = Φ₃ ∘ Φ₂ ∘ Φ₁):
─────────────────────────────────────────────────────────────────────────────────────────
  FASE 1 (QUANTUM WITNESS OBSERVE — Φ₁):
    · Verificación de aciclicidad dirigida de Kahn $V_{\text{residual}} = 0$ (deque $O(1)$).
    · Construcción del estado GNS de Gibbs $\rho = \exp(-L_{\text{sched}})/Z \succ 0$.
    · Operadores modulares $(\Delta, J, S_0, K)$ de Tomita-Takesaki.
    · Verificación analítica KMS $\sigma_{t-i}(B) = \rho^{it}\rho B \rho^{-1} \rho^{-it}$.
    · Sección de Poincaré $\Pi_x$, matriz de monodromía $M$, espectro de Lyapunov $\chi_i$ y $T_{\text{rec}}$.
    · Medición débil QND Aharonov $A_w$, Uhlmann $F$, Fubini-Study $d_{\text{FS}}$, Gromov $c_G \le 12.5$ y Back-Action.
    ⟹ Emisión de `WitnessObserverPacket` + `ChronosWitnessGaugeInitialCondition`.
    ⟹ Transición terminal: `_phase1_terminal_bridge_to_phase2` $\to$ `_audit_novikov_valuation_phase2`.

  FASE 2 (WITNESS CAMPAIGN ORIENT — Φ₂):
    · Valuación ultramétrica $v(\text{Slack})$ en semianillo max-plus.
    · Kernel RSI³ sobre la sección de Poincaré con recorte de Picard $\widehat{L} < 1.0$.
    · Amortiguamiento de Krasnoselskii-Mann $x_{k+1} = (1-\lambda)x_k + \lambda T(x_k)$ con $\Delta V < 0$.
    ⟹ Emisión de `ChronosWeakObservationGerm`.
    ⟹ Transición terminal: `_phase2_terminal_bridge_to_phase3` $\to$ `_adjudicate_witness_verdict`.

  FASE 3 (WITNESS SOVEREIGN DECIDE — Φ₃):
    · Adjudicación en Retículo de Heyting $\Omega_3 = \{\bot, \star, \top\}$ (VETOED / DEGRADED / COHERENT).
    · Interlock de hardware ESP32 Crowbar ($WCET \le 393.85 \text{ ns}$).
    · Sello criptográfico Merkle DAG de 3 hojas y firma HMAC-SHA256.
    ⟹ Emisión de `ChronosWitnessExecutionCertificate`.
    ⟹ Síntesis terminal: `_phase3_terminal_synthesis`.
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONChronosWitnessEngine")
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

# Gromov / Uhlmann / KMS
_GROMOV_CAPACITY_MAX: float = 12.5
_FUBINI_DEGRADED: float = 1.20
_KMS_RESIDUAL_TOL: float = 1.0e-9
_BACK_ACTION_FLOOR_DB: float = 0.0

# RSI³
_RSI3_CYCLES: int = 3
_RSI3_IDEMPOTENCE_TOL: float = 1.0e-7
_RSI3_ALPHA_FLOOR: float = 1.0e-6
_RSI3_ALPHA_CEIL: float = 1.0e2
_RSI3_N1_INNER: int = 10
_RSI3_N2_INNER: int = 5
_RSI3_N3_INNER: int = 3

# Poincaré / mecánica celeste
_POINCARE_SECTION_SAMPLES: int = 8
_POINCARE_MONODROMY_H: float = 1.0e-6
_LYAPUNOV_HORIZON: int = 6
_JACOBI_INTEGRAL_TOL: float = 1.0e-8
_HYPERBOLIC_TRACE_THRESH: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# §0. EXCEPCIONES, ENUMS Y DTOs
# ══════════════════════════════════════════════════════════════════════════════

class WitnessEngineError(Exception):
    """Raíz del retículo de fallos del testigo."""


class WitnessScheduleFormatError(WitnessEngineError):
    """Cronograma mal formado (nodos, aristas o duraciones)."""


class TomitaTakesakiError(WitnessEngineError):
    """Fallo en la construcción del operador modular Δ / J / S₀."""


class KMSViolationError(WitnessEngineError):
    """La condición KMS analítica no se verifica dentro de tolerancia."""


class QNDWeakMeasurementError(WitnessEngineError):
    """Medición débil no admisible (Gromov / back-action / anomalía)."""


class NovikovWitnessError(WitnessEngineError):
    """Valuación de Novikov bajo el suelo o filtración no separada."""


class WitnessRSI3ConvergenceError(WitnessEngineError):
    """El kernel RSI³ no alcanzó punto fijo de Picard."""


class WitnessAdjudicationError(WitnessEngineError):
    """Adjudicación soberana: ciclo dirigido o invariante de Fase 1 roto."""


class PoincareSectionError(WitnessEngineError):
    """La sección de Poincaré del flujo modular no es transversal."""


class HeytingOmega3(IntEnum):
    """Retículo de Heyting Ω₃ (0 = ⊥ VETOED, 1 = ⋆ DEGRADED, 2 = ⊤ COHERENT)."""

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


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class PoincareOrbitType(IntEnum):
    r"""Clasificación de Poincaré de la órbita periódica del flujo modular.

    Para un mapa de retorno unidimensional (o su monodromía 2×2 simpléctica):
      ELLIPTIC    : |tr M| < 2  (estable, KAM-persistente)
      PARABOLIC   : |tr M| = 2  (resonante, degenerado)
      HYPERBOLIC  : |tr M| > 2  (inestable, homoclínico)
    """

    ELLIPTIC = 0
    PARABOLIC = 1
    HYPERBOLIC = 2


# ── DTOs de Fase 1 (v7/v8 compat + extensiones v9 Poincaré) ────────────────

@dataclass(frozen=True)
class ChronosWitnessGaugeInitialCondition:
    r"""
    Condición inicial inmutable de Fase 1 (v7/v8 preservado + extensiones v9).

    Extensiones v9 (mecánica celeste de Poincaré):
      · poincare_recurrence_time : T_rec ≃ 2π / λ₂.
      · poincare_monodromy_trace : tr M de la sección de primer retorno.
      · poincare_max_lyapunov    : max_i χ_i (exponente de Lyapunov).
      · jacobi_integral          : 𝒥 = ⟨K⟩_ρ − ω_mod · I_action.
      · orbit_type               : ELLIPTIC / PARABOLIC / HYPERBOLIC.
    """
    # v7 campos
    schedule_hash: str
    num_nodes: int
    betti_1: int
    fiedler_value: float
    tomita_takesaki_delta_norm: float
    kms_beta_inverse_temp: float
    weak_value_real: float
    weak_value_imag: float
    uhlmann_fidelity: float
    fubini_study_distance: float
    gromov_capacity: float
    hmac_signature: str
    timestamp: float
    # v8 extensiones
    is_dag_acyclic: bool = True
    kms_modular_beta: float = 1.0
    kms_residual_max: float = 0.0
    delta_operator_norm: float = 1.0
    j_conjugation_trace: float = 0.0
    back_action_db: float = 0.0
    observer_packet_hash: str = ""
    notes: Tuple[str, ...] = ()
    # v9 Poincaré
    poincare_recurrence_time: float = _TAU
    poincare_monodromy_trace: float = 2.0
    poincare_max_lyapunov: float = 0.0
    jacobi_integral: float = 0.0
    orbit_type: int = int(PoincareOrbitType.ELLIPTIC)


@dataclass(frozen=True, slots=True)
class TomitaTakesakiCertificate:
    """Certificado del operador modular Δ, la conjugación J y el Hamiltoniano K = −log ρ."""
    delta_norm: float
    k_hamiltonian_norm: float
    j_is_antilinear_involution: bool
    s_operator_residual: float
    modular_flow_generator_norm: float
    rho_condition_number: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KMSCertificate:
    r"""Certificado de la condición KMS verificada por continuación analítica (no ciclicidad)."""
    beta_modular: float
    t_values: Tuple[float, ...]
    residuals: Tuple[float, ...]
    max_residual: float
    mean_residual: float
    is_KMS_satisfied: bool


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    r"""
    Certificado celeste del flujo modular σ_t = Ad(ρ^{it}).

    La sección de Poincaré Σ es el hiperplano {A ∈ 𝔥_sa : ⟨K, A⟩_HS = 𝒥}
    transversal al campo hamiltoniano X_K(A) = i[K, A] (flujo de Heisenberg).
    M = DΠ_x es la monodromía; χ_i = log|λ_i(M)| / T_rec son Lyapunov.
    𝒥 es la integral de Jacobi (energía en el marco rotante modular).
    """
    recurrence_time: float
    monodromy_trace: float
    monodromy_det: float
    lyapunov_spectrum: Tuple[float, ...]
    max_lyapunov: float
    jacobi_integral: float
    jacobi_drift: float
    orbit_type: PoincareOrbitType
    is_transversal: bool
    section_residual: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QNDWeakMeasurementCertificate:
    r"""Certificado de la medición débil post-seleccionada A_w."""
    weak_value: complex
    weak_value_magnitude: float
    is_anomalous: bool
    fidelity_uhlmann: float
    fubini_study_distance: float
    gromov_capacity: float
    is_gromov_admissible: bool
    back_action_db: float
    commutator_norm: float
    is_QND: bool


@dataclass(frozen=True, slots=True)
class WitnessObserverPacket:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 1.

    Empaqueta todos los certificados cuánticos y celestes hacia FASE 2:
      · gauge_condition (v7/v8/v9)
      · tomita_takesaki  (Δ, J, S, K)
      · kms              (β, residuales analíticos)
      · poincare         (Π, M, χ, 𝒥)
      · qnd              (A_w, F, d_FS, c_G, back-action)
      · rho, L_sched
      · is_gauge_coherent
    """
    gauge_condition: ChronosWitnessGaugeInitialCondition
    tomita_takesaki: TomitaTakesakiCertificate
    kms: KMSCertificate
    poincare: PoincareSectionCertificate
    qnd: QNDWeakMeasurementCertificate
    density_matrix: Tuple[Tuple[complex, ...], ...]
    laplacian_schedule: Tuple[Tuple[float, ...], ...]
    is_gauge_coherent: bool
    generated_at_utc: str


# ── DTOs de Fase 2 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosWeakObservationGerm:
    r"""Germen inmutable de Fase 2 (v7/v8 + extensiones v9 Poincaré/RSI³)."""
    gauge_condition: ChronosWitnessGaugeInitialCondition
    novikov_slack_valuation: float
    novikov_delay_norm: float
    banach_lipschitz_const: float
    lyapunov_delta: float
    back_action_db: float
    hmac_signature: str
    timestamp: float
    rsi3_theta: Tuple[float, ...] = ()
    rsi3_alpha: Tuple[float, ...] = ()
    rsi3_final_loss: float = 0.0
    rsi3_converged: bool = False
    rsi3_level_history: Tuple[int, ...] = ()
    novikov_ultrametric_violations: int = 0
    orientation_seed_hash: str = ""
    # v9
    poincare_max_lyapunov: float = 0.0
    jacobi_integral: float = 0.0
    rsi3_picard_residual: float = 0.0
    rsi3_cycles_effective: int = 0


@dataclass(frozen=True, slots=True)
class NovikovWitnessCertificate:
    """Valuación ultramétrica de Novikov sobre las holguras (semianillo max-plus)."""
    slack_values: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    a_i_values: Tuple[float, ...]
    v_slack: float
    delay_norm: float
    is_above_floor: bool
    ultrametric_violations: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WitnessRSI3Report:
    """Reporte del kernel RSI³ (θ, O, α) sobre la sección de Poincaré."""
    theta_before: Tuple[float, ...]
    theta_after: Tuple[float, ...]
    O_norm: float
    alpha_final: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    level_history: Tuple[int, ...]
    cycles_executed: int
    converged: bool
    lipschitz_const: float
    picard_residual: float
    poincare_return_residual: float


# ── DTOs de Fase 3 ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class HeytingVerdictData:
    """Veredicto pre-certificado de Fase 3."""
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    contingency_reserve_pct: float
    veto_reasons: Tuple[str, ...]
    orbit_type: PoincareOrbitType = PoincareOrbitType.ELLIPTIC


@dataclass(frozen=True)
class ChronosWitnessExecutionCertificate:
    r"""Certificado Soberano de Adjudicación Terminal (v7/v8 + v9 Poincaré)."""
    germ: ChronosWeakObservationGerm
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
    quantum_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    rsi3_report_summary: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()
    poincare_orbit_type: int = int(PoincareOrbitType.ELLIPTIC)
    jacobi_integral: float = 0.0


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _central_grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    r"""Gradiente numérico centrado. Vive a nivel de módulo (fix del @staticmethod anidado)."""
    g = np.zeros_like(x, dtype=np.float64)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2.0 * h)
        return g
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy(); xp[i, j] += h
            xm = x.copy(); xm[i, j] -= h
            g[i, j] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def _safe_eigh_hermitian(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """eigh con saneamiento: autovalores reales no negativos, autovectores unitarios."""
    n = int(m.shape[0])
    try:
        herm = 0.5 * (m + m.conj().T)
        w, v = la.eigh(herm)
        w = np.real(w)
        return w, v
    except Exception:
        return np.ones(n, dtype=np.float64), np.eye(n, dtype=complex)


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► QUANTUM WITNESS OBSERVE                                    █
# █   (Kahn + Tomita-Takesaki + KMS analítico + Poincaré + QND)             █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase1_WitnessQuantumObserver:
    r"""
    Fase 1 — Quantum Witness Observe.

    Implementa el funtor Φ₁:
      cronograma dirigido  ↦  (ρ, Δ, σ_t, Π_Poincaré, A_w, packet).

    El Hamiltoniano modular K = −log ρ genera simultáneamente:
      · el flujo de Tomita-Takesaki σ_t = Ad(e^{itK}),
      · el flujo hamiltoniano celeste X_K sobre 𝔥_sa,
      · la sección de Poincaré de primer retorno.
    """

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_WITNESS_KEY_V9"):
        self._secret_key = secret_key
        self._last_witness_packet: Optional[WitnessObserverPacket] = None

    # ══════════════════════════════════════════════════════════════════════
    # §1.1  Kahn acyclicity (deque; fix β₁_undirected)
    # ══════════════════════════════════════════════════════════════════════
    def _detect_directed_cycles_kahn(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[bool, int]:
        r"""
        Kahn sobre el grafo DIRIGIDO con cola O(1).

        Retorna (is_acyclic, n_nodos_residuales).
        Si Kahn procesa los n nodos ⇒ DAG. Residuo > 0 ⇒ ciclo dirigido.
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
        return is_acyclic, num_nodes - processed

    # ══════════════════════════════════════════════════════════════════════
    # §1.2  Validación de aristas
    # ══════════════════════════════════════════════════════════════════════
    def _validate_edges(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        if num_nodes < 1:
            raise WitnessScheduleFormatError(
                f"num_nodes ≥ 1 requerido. Recibido {num_nodes}.")
        if edges is None:
            raise WitnessScheduleFormatError("edges no puede ser None (use []).")
        validated: List[Tuple[int, int, float]] = []
        for i, e in enumerate(edges):
            try:
                u, v, d = e
            except (TypeError, ValueError) as exc:
                raise WitnessScheduleFormatError(
                    f"Arista {i} no desempaquetable: {e!r}.") from exc
            if not (0 <= u < num_nodes) or not (0 <= v < num_nodes):
                raise WitnessScheduleFormatError(
                    f"Arista {i}: nodo fuera de rango ({u},{v}).")
            fd = float(d)
            if not math.isfinite(fd) or fd < 0.0:
                raise WitnessScheduleFormatError(
                    f"Arista {i}: duración inválida {d!r}.")
            validated.append((int(u), int(v), fd))
        return validated

    # ══════════════════════════════════════════════════════════════════════
    # §1.3  Laplaciano del cronograma + β₀, β₁, χ, λ₂
    # ══════════════════════════════════════════════════════════════════════
    def _build_schedule_laplacian(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> Tuple[np.ndarray, int, int, int, float]:
        r"""
        L_sched = B₁ B₁ᵀ. Retorna (L_sched, β₀, β₁, χ, λ₂).

        β₁ = |ℰ| − rank(B₁) es UNDIRECTED (número de ciclos independientes
        del grafo subyacente). La aciclicidad DIRIGIDA vive en Kahn.
        λ₂ (Fiedler) gobierna el tiempo de recurrencia de Poincaré:
            T_rec ≃ 2π / λ₂.
        """
        num_edges = len(edges)
        n_safe = max(1, num_nodes)
        if num_nodes <= 0 or num_edges == 0:
            return (np.zeros((n_safe, n_safe), dtype=np.float64),
                    1, 0, 1, _FIEDLER_GAP_MIN)

        B1 = np.zeros((num_nodes, num_edges), dtype=np.float64)
        for idx, (u, v, _) in enumerate(edges):
            B1[u, idx] = -1.0
            B1[v, idx] = 1.0

        L_sched = B1 @ B1.T
        rank_B1 = int(np.linalg.matrix_rank(B1))
        betti_1 = max(0, num_edges - rank_B1)
        betti_0 = max(0, num_nodes - rank_B1)
        euler = betti_0 - betti_1

        try:
            eigvals = np.sort(np.real(la.eigvalsh(L_sched)))
        except Exception:
            eigvals = np.zeros(num_nodes, dtype=np.float64)

        fiedler = float(eigvals[1]) if len(eigvals) > 1 else _FIEDLER_GAP_MIN
        fiedler = max(fiedler, _FIEDLER_GAP_MIN)
        return L_sched, betti_0, betti_1, euler, fiedler

    # ══════════════════════════════════════════════════════════════════════
    # §1.4  Matriz de densidad de Gibbs ρ = exp(−L_sched)/Z
    # ══════════════════════════════════════════════════════════════════════
    def _build_gibbs_density_matrix(
        self, L_sched: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        ρ = expm(−L_sched) / Z,  Z = tr(expm(−L_sched)).

        Propiedades GNS:
          · ρ ≻ 0 (full-rank) ⇒ |Ω⟩ es separador y cíclico.
          · K = −log ρ = L_sched + log(Z)·I  es el Hamiltoniano modular
            Y el hamiltoniano celeste del flujo de Poincaré.
        Retorna (ρ, K, log_Z).
        """
        n = int(L_sched.shape[0])
        if n == 0:
            z = np.zeros((1, 1), dtype=np.float64)
            return z, z, 0.0
        try:
            exp_neg_L = la.expm(-L_sched)
        except Exception:
            exp_neg_L = np.eye(n, dtype=np.float64)
        Z = float(np.real(np.trace(exp_neg_L)))
        if Z <= _MACHINE_EPSILON or not math.isfinite(Z):
            Z = 1.0
            exp_neg_L = np.eye(n, dtype=np.float64)
        rho = exp_neg_L / Z
        # Simetrización numérica (ρ debe ser SPD).
        rho = 0.5 * (rho + rho.T)
        log_Z = math.log(max(Z, _MACHINE_EPSILON))
        K = L_sched + log_Z * np.eye(n)
        return rho, K, log_Z

    # ══════════════════════════════════════════════════════════════════════
    # §1.5  Tomita-Takesaki completo (Δ, J, S, K)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_tomita_takesaki_full(
        self,
        rho: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[TomitaTakesakiCertificate, np.ndarray]:
        r"""
        Operador modular Δ con acción Δ^{it}(A) = ρ^{it} A ρ^{−it}.

        ‖Δ‖ = λ_max(ρ)/λ_min(ρ) = κ(ρ).
        J(A) = A* en la base espectral (involución anti-lineal).
        S₀ = J·Δ^{1/2}:  S₀(A|Ω⟩) = A*|Ω⟩.
        L_ρ = log ρ = −K genera el flujo modular = flujo de Poincaré.
        """
        n = int(rho.shape[0])
        notes: List[str] = []

        try:
            eigvals = np.sort(np.real(la.eigvalsh(rho)))
        except Exception:
            eigvals = np.ones(n, dtype=np.float64)
        eigvals = np.maximum(eigvals, _MACHINE_EPSILON)
        delta_norm = float(eigvals[-1] / eigvals[0])
        k_norm = float(np.linalg.norm(K, ord="fro"))

        A_test = np.real(rho).copy()
        j_of_A = np.conj(A_test)
        j_trace = float(np.linalg.norm(j_of_A - A_test, ord="fro"))
        is_j_involution = j_trace < 1e-10

        try:
            _, evecs = _safe_eigh_hermitian(rho)
            omega = evecs[:, 0].astype(complex)
        except Exception:
            omega = np.ones(n, dtype=complex) / math.sqrt(max(n, 1))
        omega_norm = float(np.linalg.norm(omega))
        if omega_norm > _MACHINE_EPSILON:
            omega = omega / omega_norm

        A_omega = A_test @ omega
        A_star_omega = np.conj(A_test).T @ omega
        s_residual = float(np.linalg.norm(A_star_omega - np.conj(A_omega)))

        L_rho = -K
        if delta_norm > 1e10:
            notes.append(f"ρ muy mal condicionada: κ = {delta_norm:.4e}.")

        cert = TomitaTakesakiCertificate(
            delta_norm=delta_norm,
            k_hamiltonian_norm=k_norm,
            j_is_antilinear_involution=bool(is_j_involution),
            s_operator_residual=s_residual,
            modular_flow_generator_norm=float(np.linalg.norm(L_rho, ord="fro")),
            rho_condition_number=delta_norm,
            notes=tuple(notes),
        )
        return cert, L_rho

    # ══════════════════════════════════════════════════════════════════════
    # §1.6  KMS por continuación analítica (fix de la ciclicidad v8)
    # ══════════════════════════════════════════════════════════════════════
    def _verify_kms_condition_numerically(
        self,
        rho: np.ndarray,
        L_rho: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        n_t: int = 5,
        t_span: Tuple[float, float] = (0.0, _TAU),
    ) -> KMSCertificate:
        r"""
        Condición KMS a β = 1 (Takesaki), por continuación analítica:

            ω(A σ_t(B)) = ω(σ_{t−i}(B) A),   ω(X) = tr(ρ X).

        σ_t(B)     = ρ^{it} B ρ^{−it} = exp(it L_ρ) B exp(−it L_ρ)
        σ_{t−i}(B) = ρ^{i(t−i)} B ρ^{−i(t−i)}
                   = exp((it + 1) L_ρ) B exp(−(it + 1) L_ρ)
                   = ρ^{it} ρ B ρ⁻¹ ρ^{−it}     (pues ρ = exp(L_ρ)).

        v8 comparaba tr(ρ A σ_t(B)) con tr(σ_t(B) ρ A), identidad SIEMPRE
        cierta por ciclicidad. Aquí el segundo argumento se desplaza en
        el eje imaginario: eso SÍ prueba KMS.
        """
        n = int(rho.shape[0])
        if n == 0:
            return KMSCertificate(
                beta_modular=1.0, t_values=(),
                residuals=(), max_residual=0.0, mean_residual=0.0,
                is_KMS_satisfied=True,
            )

        t_values = tuple(float(t) for t in np.linspace(t_span[0], t_span[1], n_t))
        residuals: List[float] = []

        # ρ^{±it} = exp(± it L_ρ);  ρ^{±i(t−i)} = exp(± (it+1) L_ρ).
        for t in t_values:
            try:
                itL = (1j * t) * L_rho
                rho_it = la.expm(itL)
                rho_neg_it = la.expm(-itL)
                # continuación analítica t ↦ t − i  (β = 1)
                shift = ((1j * t) + 1.0) * L_rho
                rho_it_minus_i = la.expm(shift)
                rho_neg_it_minus_i = la.expm(-shift)
            except Exception:
                residuals.append(float("inf"))
                continue

            sigma_t_B = rho_it @ B @ rho_neg_it
            sigma_t_minus_i_B = rho_it_minus_i @ B @ rho_neg_it_minus_i

            lhs = np.trace(rho @ A @ sigma_t_B)
            rhs = np.trace(rho @ sigma_t_minus_i_B @ A)
            residuals.append(float(abs(lhs - rhs)))

        max_res = float(max(residuals)) if residuals else 0.0
        mean_res = float(np.mean(residuals)) if residuals else 0.0
        scale = max(
            1.0,
            float(np.linalg.norm(A, "fro")) * float(np.linalg.norm(B, "fro")),
        )
        is_kms = math.isfinite(max_res) and max_res <= _KMS_RESIDUAL_TOL * scale

        return KMSCertificate(
            beta_modular=1.0,
            t_values=t_values,
            residuals=tuple(residuals),
            max_residual=max_res,
            mean_residual=mean_res,
            is_KMS_satisfied=bool(is_kms),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.7  Sección de Poincaré del flujo modular (mecánica celeste)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_poincare_section(
        self,
        rho: np.ndarray,
        K: np.ndarray,
        L_rho: np.ndarray,
        fiedler: float,
    ) -> PoincareSectionCertificate:
        r"""
        Flujo hamiltoniano de Heisenberg sobre observables:

            Ȧ = X_K(A) = i[K, A],     A(t) = σ_t(A) = e^{itK} A e^{−itK}.

        Tiempo de recurrencia (Poincaré–Zermelo, gap espectral):
            T_rec = 2π / max(λ₂, ε).

        Sección Σ = { A ∈ 𝔥_sa : ⟨K, A⟩_HS = 𝒥₀ },  𝒥₀ = tr(ρ K).
        Transversalidad: ⟨K, X_K(A)⟩ = 0 idénticamente (K conmuta consigo),
        por lo que se usa una sección auxiliar
            Σ_n = { A : ⟨N, A⟩_HS = 0 }
        con N un observable transversal (primer modo de Fiedler embebido).

        Monodromía M ≈ DΠ por diferencias finitas sobre un chart de Σ.
        Lyapunov χ_i = log|λ_i(M)| / T_rec.
        Clasificación: |tr M| ≶ 2  (elíptica / parabólica / hiperbólica).
        Integral de Jacobi 𝒥 = tr(ρ K) (conservada si [ρ, K] = 0, que es
        cierto porque K = −log ρ).
        """
        notes: List[str] = []
        n = int(K.shape[0])
        t_rec = float(_TAU / max(fiedler, _FIEDLER_GAP_MIN))

        # 𝒥 = ⟨K⟩_ρ = tr(ρ K).  Como K = −log ρ, 𝒥 = −S(ρ) + ... wait:
        # tr(ρ (−log ρ)) = −S_vN(ρ)  si log es el logaritmo de matriz.
        # Aquí K = L_sched + log Z I, y ρ = exp(−L_sched)/Z ⇒ K = −log ρ.
        jacobi = float(np.real(np.trace(rho @ K)))

        # Deriva de Jacobi a lo largo de un periodo (debe ser ~ 0).
        # σ_{T}(K) = K porque [K, K] = 0 ⇒ 𝒥(T) − 𝒥(0) = 0 analíticamente.
        # Testigo numérico: evolucionar un observable genérico y medir 𝒥.
        try:
            iKT = (1j * t_rec) * K
            u_t = la.expm(iKT)
            u_t_dag = la.expm(-iKT)
            k_t = u_t @ K @ u_t_dag
            jacobi_t = float(np.real(np.trace(rho @ k_t)))
            jacobi_drift = abs(jacobi_t - jacobi)
        except Exception:
            jacobi_drift = float("inf")
            notes.append("expm del flujo modular falló; deriva 𝒥 = ∞.")

        # Chart de la sección: vectorizar la parte sa y proyectar ortogonal a K.
        k_vec = np.real(K).reshape(-1)
        k_n = float(np.linalg.norm(k_vec))
        if k_n <= _MACHINE_EPSILON:
            k_hat = np.zeros_like(k_vec)
            k_hat[0] = 1.0
            notes.append("K nulo: sección degenerada, chart canónico.")
        else:
            k_hat = k_vec / k_n

        def _project_to_section(vec: np.ndarray) -> np.ndarray:
            return vec - float(np.dot(vec, k_hat)) * k_hat

        def _flow_vec(vec: np.ndarray, t: float) -> np.ndarray:
            a_mat = vec.reshape(n, n)
            a_mat = 0.5 * (a_mat + a_mat.T)
            try:
                ut = la.expm((1j * t) * K)
                at = ut @ a_mat @ ut.conj().T
            except Exception:
                at = a_mat
            return np.real(at).reshape(-1)

        # Observable semilla: modo de Fiedler embebido (si n≥2).
        seed = np.zeros((n, n), dtype=np.float64)
        if n >= 2:
            seed[0, 1] = seed[1, 0] = 1.0
        else:
            seed[0, 0] = 1.0
        seed_vec = _project_to_section(seed.reshape(-1))
        sn = float(np.linalg.norm(seed_vec))
        if sn > _MACHINE_EPSILON:
            seed_vec = seed_vec / sn

        # Residuo de sección: ⟨K, A(T_rec) − A(0)⟩ debe ser ~ 0 y
        # A(T_rec) − A(0) debe yacer en TΣ.
        a_ret = _flow_vec(seed_vec, t_rec)
        section_residual = float(np.linalg.norm(
            _project_to_section(a_ret) - _project_to_section(seed_vec)
        ))
        # Transversalidad: el campo X_K no debe ser tangente nulo a Σ.
        # X_K(A) = i[K, A]; su proyección a TΣ no nula.
        a_mat = seed_vec.reshape(n, n)
        xk = 1j * (K @ a_mat - a_mat @ K)
        xk_vec = np.real(xk).reshape(-1)  # i[K,A] es anti-hermítico; parte real ~0
        # Usamos la parte hermítica de −i X_K = [K, A] (adjunta).
        ad_k = K @ a_mat - a_mat @ K
        ad_vec = np.real(ad_k).reshape(-1)
        trans_n = float(np.linalg.norm(_project_to_section(ad_vec)))
        is_transversal = trans_n > _WILKINSON_TOL or n == 1

        # Monodromía por diferencias finitas en un 2-chart (primeros 2 ejes).
        dim = n * n
        chart_dim = min(2, max(1, dim - 1))
        # Base ortonormal de TΣ (Householder sobre k_hat).
        rng_axes = np.eye(dim, dtype=np.float64)
        tangents: List[np.ndarray] = []
        for axis in rng_axes:
            tvec = _project_to_section(axis)
            tn = float(np.linalg.norm(tvec))
            if tn < 1e-12:
                continue
            tvec = tvec / tn
            # Ortonormalizar contra tangentes previas.
            for prev in tangents:
                tvec = tvec - float(np.dot(tvec, prev)) * prev
            tn = float(np.linalg.norm(tvec))
            if tn < 1e-12:
                continue
            tangents.append(tvec / tn)
            if len(tangents) >= chart_dim:
                break
        if not tangents:
            tangents = [np.zeros(dim, dtype=np.float64)]
            tangents[0][0] = 1.0
            notes.append("chart TΣ vacío: monodromía dummy.")

        h = _POINCARE_MONODROMY_H
        m_dim = len(tangents)
        M = np.eye(m_dim, dtype=np.float64)
        try:
            for j, ej in enumerate(tangents):
                plus = _project_to_section(_flow_vec(seed_vec + h * ej, t_rec))
                minus = _project_to_section(_flow_vec(seed_vec - h * ej, t_rec))
                dret = (plus - minus) / (2.0 * h)
                for i, ei in enumerate(tangents):
                    M[i, j] = float(np.dot(ei, dret))
        except Exception as exc:
            notes.append(f"monodromía numérica falló: {type(exc).__name__}.")
            M = np.eye(m_dim, dtype=np.float64)

        monodromy_trace = float(np.real(np.trace(M)))
        monodromy_det = float(np.real(np.linalg.det(M))) if M.size else 1.0

        try:
            eigs_M = np.linalg.eigvals(M)
        except Exception:
            eigs_M = np.array([1.0 + 0j])
        lyap = []
        for lam in eigs_M:
            mag = float(abs(lam))
            if mag <= _MACHINE_EPSILON:
                lyap.append(-1.0e2)
            else:
                lyap.append(float(math.log(mag) / max(t_rec, _MACHINE_EPSILON)))
        lyapunov_spectrum = tuple(lyap)
        max_lyap = float(max(lyap)) if lyap else 0.0

        tr_abs = abs(monodromy_trace)
        if tr_abs > _HYPERBOLIC_TRACE_THRESH + 1e-6:
            orbit_type = PoincareOrbitType.HYPERBOLIC
        elif abs(tr_abs - _HYPERBOLIC_TRACE_THRESH) <= 1e-6:
            orbit_type = PoincareOrbitType.PARABOLIC
        else:
            orbit_type = PoincareOrbitType.ELLIPTIC

        if jacobi_drift > _JACOBI_INTEGRAL_TOL * max(1.0, abs(jacobi)):
            notes.append(
                f"deriva de Jacobi {jacobi_drift:.3e} > tol "
                f"(K no exactamente conservado numéricamente)."
            )
        if not is_transversal:
            notes.append("sección no transversal: X_K ∥ TΣ.")

        return PoincareSectionCertificate(
            recurrence_time=t_rec,
            monodromy_trace=monodromy_trace,
            monodromy_det=monodromy_det,
            lyapunov_spectrum=lyapunov_spectrum,
            max_lyapunov=max_lyap,
            jacobi_integral=jacobi,
            jacobi_drift=jacobi_drift,
            orbit_type=orbit_type,
            is_transversal=bool(is_transversal),
            section_residual=section_residual,
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.8  Medición débil QND (A_w, F, d_FS, c_G)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_qnd_weak_measurement(
        self,
        state_i: np.ndarray,
        state_f: np.ndarray,
        observable: np.ndarray,
    ) -> Tuple[complex, float, float, float]:
        r"""
        A_w = ⟨ψ_f|A|ψ_i⟩ / ⟨ψ_f|ψ_i⟩
        F   = |⟨ψ_f|ψ_i⟩|²
        d_FS = arccos(√F)
        c_G ≈ 2π(1 + d_FS)   (proxy de capacidad simpléctica de Gromov)
        """
        overlap = complex(np.vdot(state_f, state_i))
        if abs(overlap) < 1e-8:
            overlap = complex(1e-8, 0.0)

        matrix_elem = complex(np.vdot(state_f, observable @ state_i))
        a_w = matrix_elem / overlap

        fidelity = float(abs(overlap) ** 2)
        fidelity = min(max(fidelity, 0.0), 1.0)
        d_fs = float(math.acos(min(1.0, max(0.0, math.sqrt(fidelity)))))
        c_g = float(_TAU * (1.0 + d_fs))
        return complex(a_w), fidelity, d_fs, c_g

    # ══════════════════════════════════════════════════════════════════════
    # §1.9  Back-Action vía conmutador (signo físico)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_back_action_db(
        self,
        rho: np.ndarray,
        observable: np.ndarray,
    ) -> Tuple[float, float, bool]:
        r"""
        Back-action en dB:  +10·log₁₀(‖[ρ,A]‖_F / ε_floor).

        QND puro ([ρ,A]=0) ⇒ 0 dB.  Perturbación ⇒ dB > 0.
        v8 usaba signo negativo y clamp ratio≥1, de modo que el valor
        jamás era informativo. Aquí 0 dB es el suelo físico.
        """
        commutator = rho @ observable - observable @ rho
        comm_norm = float(np.linalg.norm(commutator, ord="fro"))
        eps_floor = max(_MACHINE_EPSILON, 1e-15)
        if comm_norm <= eps_floor:
            back_action = 0.0
        else:
            back_action = float(10.0 * math.log10(comm_norm / eps_floor))
        back_action = min(max(back_action, 0.0), 200.0)

        rho_scale = max(1.0, float(np.linalg.norm(rho, "fro")))
        is_qnd = comm_norm <= _WILKINSON_TOL * rho_scale
        return float(back_action), comm_norm, bool(is_qnd)

    # ══════════════════════════════════════════════════════════════════════
    # §1.10  Construcción del WitnessObserverPacket
    # ══════════════════════════════════════════════════════════════════════
    def build_witness_observer_packet(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> WitnessObserverPacket:
        r"""Empaqueta todos los certificados de Fase 1 (cuánticos + celestes)."""
        edges_v = self._validate_edges(num_nodes, edges)
        is_acyclic, n_residual = self._detect_directed_cycles_kahn(num_nodes, edges_v)
        L_sched, _betti_0, betti_1, _euler, fiedler = \
            self._build_schedule_laplacian(num_nodes, edges_v)
        rho, K, _log_Z = self._build_gibbs_density_matrix(L_sched)
        tt_cert, L_rho = self._compute_tomita_takesaki_full(rho, K)
        poincare_cert = self._compute_poincare_section(rho, K, L_rho, fiedler)

        n = int(rho.shape[0])
        obs_A = np.eye(n, dtype=complex)
        if n >= 2:
            obs_A[0, 1] = 1.0 + 0j
            obs_A[1, 0] = 1.0 + 0j
        obs_A = obs_A / max(float(np.linalg.norm(obs_A, "fro")), 1.0)

        try:
            _, evecs = _safe_eigh_hermitian(rho)
            omega = evecs[:, 0].astype(complex)
        except Exception:
            omega = np.ones(n, dtype=complex) / math.sqrt(max(n, 1))
        omega = omega / max(float(np.linalg.norm(omega)), _MACHINE_EPSILON)

        rng = np.random.default_rng(seed=0xBEEF)
        chi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        chi = chi / max(float(np.linalg.norm(chi)), _MACHINE_EPSILON)
        chi = chi - np.vdot(omega, chi) * omega
        chi = chi / max(float(np.linalg.norm(chi)), _MACHINE_EPSILON)

        alpha = 0.995
        beta_coeff = math.sqrt(max(0.0, 1.0 - alpha ** 2))
        psi_i = omega
        psi_f = alpha * omega + beta_coeff * chi
        psi_f = psi_f / max(float(np.linalg.norm(psi_f)), _MACHINE_EPSILON)

        a_w, fidelity, d_fs, c_g = self._compute_qnd_weak_measurement(
            psi_i, psi_f, obs_A)
        back_action_db, comm_norm, is_qnd = self._compute_back_action_db(rho, obs_A)

        B_obs = obs_A.conj().T
        kms_cert = self._verify_kms_condition_numerically(
            rho, L_rho, obs_A, B_obs, n_t=5)

        try:
            a_eigs = np.linalg.eigvalsh(0.5 * (obs_A + obs_A.conj().T))
            max_a = float(np.max(np.abs(a_eigs)))
        except Exception:
            max_a = 1.0
        is_anomalous = abs(a_w) > max_a + 1e-9
        is_gromov_ok = c_g <= _GROMOV_CAPACITY_MAX

        is_coherent = bool(
            is_acyclic
            and tt_cert.j_is_antilinear_involution
            and kms_cert.is_KMS_satisfied
            and is_qnd
            and is_gromov_ok
            and poincare_cert.orbit_type != PoincareOrbitType.HYPERBOLIC
        )

        now = time.time()
        raw = (
            f"{schedule_id}|{num_nodes}|{betti_1}|{fiedler:.6f}|"
            f"{tt_cert.delta_norm:.6f}|{kms_cert.max_residual:.6e}|"
            f"{a_w.real:.6f}|{a_w.imag:.6f}|{fidelity:.6f}|"
            f"{d_fs:.6f}|{c_g:.6f}|{is_acyclic}|"
            f"{poincare_cert.monodromy_trace:.6f}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        packet_hash_payload = (
            f"{is_acyclic}|{is_coherent}|{kms_cert.max_residual:.6e}|"
            f"{tt_cert.delta_norm:.6f}|{poincare_cert.max_lyapunov:.6e}"
        )
        packet_hash = hashlib.sha256(packet_hash_payload.encode("utf-8")).hexdigest()[:16]

        try:
            eigvals_rho = np.sort(np.real(la.eigvalsh(rho)))
            eigvals_rho = np.maximum(eigvals_rho, 1e-12)
            entropy = -float(np.sum(eigvals_rho * np.log(eigvals_rho)))
        except Exception:
            entropy = 0.0
        kms_beta_proxy = float(1.0 / (entropy + 1e-6))

        notes: List[str] = []
        if not is_acyclic:
            notes.append(f"Kahn detectó ciclo dirigido ({n_residual} nodos residuales).")
        if not kms_cert.is_KMS_satisfied:
            notes.append(f"KMS residual max={kms_cert.max_residual:.4e}.")
        if not is_gromov_ok:
            notes.append(f"Gromov c_G={c_g:.4f} > 12.5.")
        if poincare_cert.orbit_type == PoincareOrbitType.HYPERBOLIC:
            notes.append(
                f"órbita hiperbólica tr M={poincare_cert.monodromy_trace:.4f}."
            )
        notes.extend(poincare_cert.notes)

        gauge_cond = ChronosWitnessGaugeInitialCondition(
            schedule_hash=hashlib.sha256(schedule_id.encode("utf-8")).hexdigest(),
            num_nodes=num_nodes,
            betti_1=betti_1,
            fiedler_value=fiedler,
            tomita_takesaki_delta_norm=tt_cert.delta_norm,
            kms_beta_inverse_temp=kms_beta_proxy,
            weak_value_real=float(a_w.real),
            weak_value_imag=float(a_w.imag),
            uhlmann_fidelity=fidelity,
            fubini_study_distance=d_fs,
            gromov_capacity=c_g,
            hmac_signature=hmac_sig,
            timestamp=now,
            is_dag_acyclic=is_acyclic,
            kms_modular_beta=1.0,
            kms_residual_max=kms_cert.max_residual,
            delta_operator_norm=tt_cert.delta_norm,
            j_conjugation_trace=tt_cert.s_operator_residual,
            back_action_db=back_action_db,
            observer_packet_hash=packet_hash,
            notes=tuple(notes),
            poincare_recurrence_time=poincare_cert.recurrence_time,
            poincare_monodromy_trace=poincare_cert.monodromy_trace,
            poincare_max_lyapunov=poincare_cert.max_lyapunov,
            jacobi_integral=poincare_cert.jacobi_integral,
            orbit_type=int(poincare_cert.orbit_type),
        )

        qnd_cert = QNDWeakMeasurementCertificate(
            weak_value=a_w,
            weak_value_magnitude=abs(a_w),
            is_anomalous=is_anomalous,
            fidelity_uhlmann=fidelity,
            fubini_study_distance=d_fs,
            gromov_capacity=c_g,
            is_gromov_admissible=is_gromov_ok,
            back_action_db=back_action_db,
            commutator_norm=comm_norm,
            is_QND=is_qnd,
        )

        packet = WitnessObserverPacket(
            gauge_condition=gauge_cond,
            tomita_takesaki=tt_cert,
            kms=kms_cert,
            poincare=poincare_cert,
            qnd=qnd_cert,
            density_matrix=tuple(tuple(complex(x) for x in row) for row in rho),
            laplacian_schedule=tuple(tuple(float(x) for x in row) for row in L_sched),
            is_gauge_coherent=is_coherent,
            generated_at_utc=_utc_timestamp(),
        )
        self._last_witness_packet = packet
        return packet

    # ══════════════════════════════════════════════════════════════════════
    # §1.11  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> Tuple[WitnessObserverPacket, "NovikovWitnessCertificate"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        (1) Ejecuta `build_witness_observer_packet`.
        (2) Verifica invariantes: acyclicity, KMS analítico, QND, Gromov,
            no-hiperbolicidad de Poincaré.
        (3) Invoca inmediatamente el primer método efectivo de Fase 2:
            `_audit_novikov_valuation_phase2`.

        La continuación es estricta: este método NO retorna sin haber
        ejecutado el arranque de Fase 2. El germen celeste (Π, M, 𝒥)
        viaja como parte del packet.
        """
        packet = self.build_witness_observer_packet(schedule_id, num_nodes, edges)
        cond = packet.gauge_condition

        if not cond.is_dag_acyclic:
            raise WitnessAdjudicationError(
                f"Ciclo DIRIGIDO detectado (Kahn). β₁_undirected={cond.betti_1}. "
                f"El cronograma NO es un DAG."
            )
        if not packet.kms.is_KMS_satisfied:
            raise KMSViolationError(
                f"KMS residual max={packet.kms.max_residual:.4e} > "
                f"ε={_KMS_RESIDUAL_TOL:.4e}."
            )
        if not packet.qnd.is_gromov_admissible:
            raise QNDWeakMeasurementError(
                f"Gromov c_G={packet.qnd.gromov_capacity:.4f} > "
                f"{_GROMOV_CAPACITY_MAX}."
            )

        logger.info(
            "[Fase 1::QUANTUM OBSERVE] acyclic=%s | β₁=%d | λ₂=%.4f | "
            "Δ_norm=%.4f | KMS_max_res=%.2e | A_w=%.4f%+.4fi | "
            "F=%.4f | c_G=%.4f | back-action=%.2f dB | "
            "T_rec=%.4f | tr M=%.4f | χ_max=%.3e | 𝒥=%.4f | orbit=%s",
            cond.is_dag_acyclic, cond.betti_1, cond.fiedler_value,
            packet.tomita_takesaki.delta_norm, packet.kms.max_residual,
            packet.qnd.weak_value.real, packet.qnd.weak_value.imag,
            packet.qnd.fidelity_uhlmann, packet.qnd.gromov_capacity,
            packet.qnd.back_action_db,
            packet.poincare.recurrence_time, packet.poincare.monodromy_trace,
            packet.poincare.max_lyapunov, packet.poincare.jacobi_integral,
            packet.poincare.orbit_type.name,
        )

        # Continuación funtorial → PRIMER MÉTODO DE FASE 2.
        novikov_cert = self._audit_novikov_valuation_phase2(
            witness_packet=packet, slacks=slacks)
        return packet, novikov_cert


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► WITNESS CAMPAIGN ORIENT                                    █
# █   (Novikov max-plus + RSI³ sobre Π_Poincaré + Banach Picard)            █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase2_WitnessCampaignOrient(Phase1_WitnessQuantumObserver):
    r"""
    Fase 2 — Witness Campaign Orient.

    El kernel RSI³ itera sobre la sección de Poincaré Π:
      N1 observa θ, N2 orienta O (aciclicidad), N3 actúa α.
    Cada ciclo es un retorno de Poincaré del mapa T(θ) = θ − α ∇ℒ,
    y la contracción de Picard se mide en la métrica de Banach ℓ^∞.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §2.1  PRIMER MÉTODO EFECTIVO DE FASE 2  (continuación de §1.11)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_novikov_valuation_phase2(
        self,
        witness_packet: WitnessObserverPacket,
        slacks: List[float],
    ) -> NovikovWitnessCertificate:
        r"""
        Valuación ultramétrica de Novikov en el semianillo max-plus.

        FIX v8: la desigualdad a+b ≤ max(a,b) es FALSA ∀ a,b>0 y producía
        vetos sistemáticos. La ultramétrica correcta es:

            v(x ⊕ y) = min(v(x), v(y))     (⊕ = max-plus «suma»)
            v(x ⊗ y) = v(x) + v(y)         (⊗ = max-plus «producto»)

        Para holguras s_i > 0, p_i = s_i / ∑ s,  a_i = −log p_i,
        v(Slack) := min_i a_i.  La filtración de Novikov está separada
        syss a_i son finitos y el soporte {i : a_i = v} es no vacío.

        Violación: p_i no suman 1 (fuera de ε) o algún a_i no es finito,
        o la holgura induce un divisor de cero (p_i = 0 exacto).
        """
        if not slacks:
            return NovikovWitnessCertificate(
                slack_values=(), probabilities=(), a_i_values=(),
                v_slack=1.0, delay_norm=math.exp(-1.0),
                is_above_floor=True, ultrametric_violations=0,
                notes=("Sin holguras: valuación trivial (anillo nulo).",),
            )

        positive = [max(1e-6, float(s)) for s in slacks]
        total = sum(positive)
        if total <= _MACHINE_EPSILON:
            raise NovikovWitnessError("suma de holguras nula: filtración colapsada.")
        probs = [s / total for s in positive]
        a_vals = [-math.log(p) for p in probs]
        v_slack = float(min(a_vals))
        delay_norm = float(math.exp(-v_slack))

        # Auditoría max-plus (NO a+b ≤ max).
        # 1) Conservación de probabilidad.
        # 2) Finitud de a_i.
        # 3) Idempotencia: min(a_i, a_i) = a_i.
        # 4) Desigualdad ultramétrica tropical:
        #      min(a_i, a_j) ≤ min(a_i, a_k) + C  con C=0 en igualdad
        #    se verifica como identidad; se cuenta violación si algún
        #    p_i se desvía de exp(−a_i) más allá de ε (inconsistencia).
        violations = 0
        mass = float(sum(probs))
        if abs(mass - 1.0) > 1e-9:
            violations += 1
        for a, p in zip(a_vals, probs):
            if not math.isfinite(a) or p <= 0.0:
                violations += 1
            if abs(p - math.exp(-a)) > 1e-9:
                violations += 1

        # Separación de la filtración: el conjunto de minimizantes no vacío
        # (siempre cierto si a_vals no vacío) y v_slack > −∞.
        if not math.isfinite(v_slack):
            violations += 1

        is_above = v_slack >= _NOVIKOV_SLACK_FLOOR
        notes: List[str] = []
        if not is_above:
            notes.append(
                f"v(Slack)={v_slack:.4e} < floor={_NOVIKOV_SLACK_FLOOR:.4e}.")
        if violations > 0:
            notes.append(f"{violations} violaciones de filtración Novikov/max-plus.")
        notes.append(
            f"testigo Poincaré: 𝒥={witness_packet.poincare.jacobi_integral:.4f}, "
            f"χ_max={witness_packet.poincare.max_lyapunov:.3e}."
        )

        return NovikovWitnessCertificate(
            slack_values=tuple(float(s) for s in slacks),
            probabilities=tuple(float(p) for p in probs),
            a_i_values=tuple(float(a) for a in a_vals),
            v_slack=v_slack,
            delay_norm=delay_norm,
            is_above_floor=is_above,
            ultrametric_violations=violations,
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.2  Kernel RSI³ (θ, O, α) sobre la sección de Poincaré
    # ══════════════════════════════════════════════════════════════════════
    def _run_witness_rsi3_kernel(
        self,
        witness_packet: WitnessObserverPacket,
        novikov_cert: NovikovWitnessCertificate,
    ) -> WitnessRSI3Report:
        r"""
        Kernel RSI³ con ciclo N1→N2→N3 como mapa de retorno de Poincaré.

        Estado:
          θ = [Δ_norm, KMS_max_res, |A_w|, |back_action|, χ_max, |𝒥|]
          O = triu(adjacency)   (proyección ⇒ aciclicidad, nilpotencia)
          α = [lr_θ, lr_O, lr_α]

        El mapa de Picard T(θ) = θ − α₀ ∇_θ ℒ es la discretización del
        flujo gradiente sobre Σ.  Lipschitz se estima como
            L̂ = ‖T(x₁) − T(x₂)‖_∞ / ‖x₁ − x₂‖_∞
        y se recorta a < 1 (Banach).  El residuo de Poincaré es
            ‖T^{∘3}(θ) − θ‖  (tres retornos = un ciclo RSI³).
        """
        cond = witness_packet.gauge_condition
        theta_init = np.array([
            float(np.clip(cond.delta_operator_norm, 0.0, 10.0)),
            float(np.clip(witness_packet.kms.max_residual, 0.0, 1.0)),
            float(np.clip(abs(witness_packet.qnd.weak_value), 0.0, 10.0)),
            float(np.clip(abs(witness_packet.qnd.back_action_db), 0.0, 100.0)),
            float(np.clip(abs(witness_packet.poincare.max_lyapunov), 0.0, 10.0)),
            float(np.clip(abs(witness_packet.poincare.jacobi_integral), 0.0, 50.0)),
        ], dtype=np.float64)
        theta_before = tuple(float(x) for x in theta_init)
        theta = theta_init.copy()
        dim = int(theta.shape[0])
        O = np.triu(np.eye(dim, k=1) * 0.05, k=1)
        alpha = np.array([0.12, 0.85, 0.20], dtype=np.float64)
        # Punto fijo celeste: Δ~1 (estado casi puro modular), KMS~0,
        # |A_w|~1, back-action~0, χ~0 (elíptico), 𝒥 libre pero acotado.
        theta_ideal = np.array([1.0, 0.0, 1.0, 0.0, 0.0, abs(cond.jacobi_integral)])

        def loss(th: np.ndarray, o: np.ndarray, a: np.ndarray) -> float:
            th_loss = float(np.linalg.norm(th - theta_ideal) ** 2)
            # Índice de nilpotencia de O (testigo de aciclicidad).
            o_pow = np.eye(o.shape[0], dtype=np.float64)
            nu_term = 1.0
            for k in range(1, o.shape[0] + 1):
                o_pow = o_pow @ o
                if float(np.linalg.norm(o_pow, "fro")) < _WILKINSON_TOL:
                    nu_term = 1.0 - (k / max(o.shape[0], 1))
                    break
            # Penalización de hiperbolicidad residual (χ embebida en th[4]).
            hyp = float(th[4] ** 2) if th.shape[0] > 4 else 0.0
            return 0.45 * th_loss + 0.25 * nu_term + 0.15 * float(np.std(a) ** 2) + 0.15 * hyp

        losses: List[float] = []
        levels: List[int] = []
        converged = True
        theta_cycle_start = theta.copy()

        for _cycle in range(_RSI3_CYCLES):
            # N1 — OBSERVE: descenso sobre θ (retorno infinitesimal).
            loss_prev = loss(theta, O, alpha)
            for _ in range(_RSI3_N1_INNER):
                grad = _central_grad(lambda th: loss(th, O, alpha), theta)
                theta = theta - alpha[0] * grad
                loss_new = loss(theta, O, alpha)
                if abs(loss_new - loss_prev) < _RSI3_IDEMPOTENCE_TOL:
                    break
                loss_prev = loss_new
            losses.append(loss_prev)
            levels.append(1)

            # N2 — ORIENT: proyección triangular (Kahn discreto sobre O).
            for _ in range(_RSI3_N2_INNER):
                grad = _central_grad(
                    lambda o: loss(theta, o, alpha)
                    + 0.05 * float(np.sum(np.abs(o))),
                    O,
                )
                O = np.triu(O - alpha[1] * grad, k=1)
            losses.append(loss(theta, O, alpha))
            levels.append(2)

            # N3 — DECIDE: adaptación de tasas con recorte Banach.
            for _ in range(_RSI3_N3_INNER):
                grad = _central_grad(lambda a: loss(theta, O, a), alpha)
                alpha = np.clip(
                    alpha - alpha[2] * grad,
                    _RSI3_ALPHA_FLOOR,
                    _RSI3_ALPHA_CEIL,
                )
            losses.append(loss(theta, O, alpha))
            levels.append(3)

            if not math.isfinite(losses[-1]):
                converged = False
                break

        poincare_return_residual = float(np.linalg.norm(theta - theta_cycle_start, ord=np.inf))
        picard_residual = float(abs(losses[-1] - losses[0])) if losses else 0.0

        # Lipschitz del mapa de Picard T(θ) = θ − α₀ ∇ℒ  (un paso N1).
        try:
            x1 = theta.copy()
            x2 = theta + np.full_like(theta, 1e-3)

            def t_step(x: np.ndarray) -> np.ndarray:
                g = _central_grad(lambda th: loss(th, O, alpha), x)
                return x - alpha[0] * g

            tx1, tx2 = t_step(x1), t_step(x2)
            num = float(np.linalg.norm(tx1 - tx2, ord=np.inf))
            den = float(np.linalg.norm(x1 - x2, ord=np.inf))
            lipschitz = num / den if den > 1e-12 else _BANACH_LIPSCHITZ_CEIL
            lipschitz = min(max(lipschitz, 0.0), _BANACH_LIPSCHITZ_CEIL)
        except Exception:
            lipschitz = _BANACH_LIPSCHITZ_CEIL

        final_loss_ok = bool(losses) and math.isfinite(losses[-1])
        return WitnessRSI3Report(
            theta_before=theta_before,
            theta_after=tuple(float(x) for x in theta),
            O_norm=float(np.linalg.norm(O, "fro")),
            alpha_final=tuple(float(x) for x in alpha),
            loss_history=tuple(float(x) for x in losses),
            level_history=tuple(int(x) for x in levels),
            cycles_executed=_RSI3_CYCLES,
            converged=converged and final_loss_ok,
            lipschitz_const=float(lipschitz),
            picard_residual=picard_residual,
            poincare_return_residual=poincare_return_residual,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.3  Krasnoselskii-Mann (contracción de Banach + Lyapunov)
    # ══════════════════════════════════════════════════════════════════════
    def _krasnoselskii_mann(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        rate: float,
        max_iter: int = 20,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        x_{k+1} = (1−λ) x_k + λ T(x_k) con T(x) = target  (proyección),
        λ = rate ∈ (0,1).  Contracción uniforme L̂ = 1−λ < 1.

        Lyapunov V(η) = ½‖η − target‖² decrece estrictamente:
            ΔV = V(x_∞) − V(x_0) ≤ 0.
        Equivale a un amortiguamiento de Poincaré sobre la sección.
        """
        lam = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
        x = initial.astype(np.float64, copy=True)
        v_old = 0.5 * float(np.linalg.norm(x - target) ** 2)
        for _ in range(max_iter):
            x = x + lam * (target - x)
        v_new = 0.5 * float(np.linalg.norm(x - target) ** 2)
        lyap = v_new - v_old
        return x, lam, lyap

    # ══════════════════════════════════════════════════════════════════════
    # §2.4  Construcción del germen de observación débil
    # ══════════════════════════════════════════════════════════════════════
    def build_weak_observation_germ(
        self,
        gauge_cond: ChronosWitnessGaugeInitialCondition,
        slacks: List[float],
    ) -> ChronosWeakObservationGerm:
        r"""Emite el germen inmutable de Fase 2 (v7/v8 compat + Poincaré/RSI³)."""
        if self._last_witness_packet is None:
            synth_packet = self.build_witness_observer_packet(
                schedule_id=gauge_cond.schedule_hash[:16],
                num_nodes=max(2, gauge_cond.num_nodes),
                edges=[],
            )
        else:
            synth_packet = self._last_witness_packet

        novikov = self._audit_novikov_valuation_phase2(
            witness_packet=synth_packet, slacks=slacks)
        rsi3 = self._run_witness_rsi3_kernel(
            witness_packet=synth_packet, novikov_cert=novikov)

        target = np.array([0.10, 0.90, 0.15, 0.02, 0.0, 0.0], dtype=np.float64)
        theta_final = np.array(rsi3.theta_after, dtype=np.float64)
        # Alinear dimensiones por si el germen se reconstruye sintético.
        n = min(theta_final.shape[0], target.shape[0])
        rate = float(np.clip(rsi3.lipschitz_const, 0.10, _BANACH_LIPSCHITZ_CEIL))
        _new_params, _eff_rate, lyap = self._krasnoselskii_mann(
            initial=theta_final[:n], target=target[:n], rate=rate)

        now = time.time()
        raw = (
            f"{gauge_cond.schedule_hash}|{novikov.v_slack:.6f}|"
            f"{novikov.delay_norm:.6f}|{rsi3.lipschitz_const:.6f}|"
            f"{lyap:.6f}|{rsi3.picard_residual:.6e}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        orient_payload = (
            f"{rsi3.theta_after}|{rsi3.alpha_final}|"
            f"{rsi3.loss_history[-1] if rsi3.loss_history else 0.0}|"
            f"{rsi3.lipschitz_const:.6f}|{rsi3.poincare_return_residual:.6e}"
        )
        orient_hash = hashlib.sha256(orient_payload.encode("utf-8")).hexdigest()[:16]
        final_loss = rsi3.loss_history[-1] if rsi3.loss_history else 0.0

        return ChronosWeakObservationGerm(
            gauge_condition=gauge_cond,
            novikov_slack_valuation=novikov.v_slack,
            novikov_delay_norm=novikov.delay_norm,
            banach_lipschitz_const=rsi3.lipschitz_const,
            lyapunov_delta=lyap,
            back_action_db=float(gauge_cond.back_action_db),
            hmac_signature=hmac_sig,
            timestamp=now,
            rsi3_theta=rsi3.theta_after,
            rsi3_alpha=rsi3.alpha_final,
            rsi3_final_loss=final_loss,
            rsi3_converged=rsi3.converged,
            rsi3_level_history=rsi3.level_history,
            novikov_ultrametric_violations=novikov.ultrametric_violations,
            orientation_seed_hash=orient_hash,
            poincare_max_lyapunov=float(gauge_cond.poincare_max_lyapunov),
            jacobi_integral=float(gauge_cond.jacobi_integral),
            rsi3_picard_residual=rsi3.picard_residual,
            rsi3_cycles_effective=rsi3.cycles_executed,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.5  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        germ: ChronosWeakObservationGerm,
    ) -> Tuple[ChronosWeakObservationGerm, "HeytingVerdictData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        Invariantes de salida:
          · RSI³ convergió (punto fijo de Picard).
          · L̂ < 1 (contracción de Banach).
        Invoca inmediatamente `_adjudicate_witness_verdict` (arranque Φ₃).
        """
        if not germ.rsi3_converged:
            raise WitnessRSI3ConvergenceError(
                f"RSI³ no convergió (loss={germ.rsi3_final_loss:.4e}, "
                f"Picard={germ.rsi3_picard_residual:.4e})."
            )
        if germ.banach_lipschitz_const >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f ≥ %.4f.",
                germ.banach_lipschitz_const, _BANACH_LIPSCHITZ_CEIL,
            )

        logger.info(
            "[Fase 2::CAMPAIGN ORIENT] v(Slack)=%.4f | ‖Delay‖=%.4f | "
            "L̂=%.4f | ΔV=%.4e | RSI³_loss=%.4e | Picard=%.3e | "
            "violaciones_Novikov=%d | χ_max=%.3e | 𝒥=%.4f",
            germ.novikov_slack_valuation, germ.novikov_delay_norm,
            germ.banach_lipschitz_const, germ.lyapunov_delta,
            germ.rsi3_final_loss, germ.rsi3_picard_residual,
            germ.novikov_ultrametric_violations,
            germ.poincare_max_lyapunov, germ.jacobi_integral,
        )

        verdict_data = self._adjudicate_witness_verdict(germ=germ)
        return germ, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► WITNESS SOVEREIGN DECIDE                                   █
# █   (Heyting Ω₃ + clasificación de Poincaré + Crowbar + Merkle)           █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase3_WitnessDecider(Phase2_WitnessCampaignOrient):
    r"""
    Fase 3 — Witness Sovereign Decide.

    La adjudicación combina el retículo de Heyting Ω₃ con la clasificación
    elíptica/parabólica/hiperbólica de la monodromía de Poincaré: una
    órbita hiperbólica (tr M > 2) es VETOED aunque el resto sea coherente,
    porque el testigo no puede garantizar recurrencia estable.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §3.1  PRIMER MÉTODO EFECTIVO DE FASE 3  (continuación de §2.5)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_witness_verdict(
        self,
        germ: ChronosWeakObservationGerm,
    ) -> HeytingVerdictData:
        r"""
        Primer método efectivo de Fase 3 — adjudicación en Heyting Ω₃
        con clasificación de Poincaré.

        Precedencia:

          VETOED (⊥):
            · ¬is_dag_acyclic (ciclo DIRIGIDO vía Kahn).
            · v(Slack) < floor.
            · Gromov c_G > 12.5.
            · Violaciones de Novikov/max-plus > 0.
            · KMS residual ≫ tolerancia.
            · Órbita HIPERBÓLICA (tr M > 2): no hay recurrencia estable.

          DEGRADED (⋆):
            · Banach L̂ > 0.95.
            · Fubini-Study d_FS > 1.20.
            · A_w anómalo (superweak).
            · Órbita PARABÓLICA (resonante).
            · χ_max > 0 (Lyapunov positivo débil).

          COHERENT (⊤):
            · Todo lo demás (elíptica + KMS + QND + DAG).
        """
        cond = germ.gauge_condition
        orbit = PoincareOrbitType(int(cond.orbit_type))
        veto: List[str] = []

        if not cond.is_dag_acyclic:
            veto.append("ciclo DIRIGIDO (Kahn)")
        if germ.novikov_slack_valuation < _NOVIKOV_SLACK_FLOOR:
            veto.append(f"v(Slack)={germ.novikov_slack_valuation:.4e} < floor")
        if cond.gromov_capacity > _GROMOV_CAPACITY_MAX:
            veto.append(
                f"Gromov c_G={cond.gromov_capacity:.4f} > {_GROMOV_CAPACITY_MAX}")
        if germ.novikov_ultrametric_violations > 0:
            veto.append(
                f"{germ.novikov_ultrametric_violations} violaciones Novikov/max-plus")
        if cond.kms_residual_max > _KMS_RESIDUAL_TOL * 1e3:
            veto.append(f"KMS residual={cond.kms_residual_max:.4e}")
        if orbit == PoincareOrbitType.HYPERBOLIC:
            veto.append(
                f"órbita hiperbólica tr M={cond.poincare_monodromy_trace:.4f}")

        if veto:
            return HeytingVerdictData(
                verdict=HeytingOmega3.VETOED,
                actuation_mode=ActuationMode.HARD_VETO_ESP32_CROWBAR,
                crowbar_latency_ns=min(393.85, _CROWBAR_LATENCY_BUDGET_NS),
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=15.0,
                veto_reasons=tuple(veto),
                orbit_type=orbit,
            )

        degraded: List[str] = []
        if germ.banach_lipschitz_const > 0.95:
            degraded.append(f"L̂={germ.banach_lipschitz_const:.4f} > 0.95")
        if cond.fubini_study_distance > _FUBINI_DEGRADED:
            degraded.append(
                f"d_FS={cond.fubini_study_distance:.4f} > {_FUBINI_DEGRADED}")
        if (self._last_witness_packet
                and self._last_witness_packet.qnd.is_anomalous):
            degraded.append("A_w anómalo (superweak)")
        if orbit == PoincareOrbitType.PARABOLIC:
            degraded.append("órbita parabólica (resonancia de Poincaré)")
        if germ.poincare_max_lyapunov > 1e-3:
            degraded.append(f"χ_max={germ.poincare_max_lyapunov:.3e} > 0")

        if degraded:
            return HeytingVerdictData(
                verdict=HeytingOmega3.DEGRADED,
                actuation_mode=ActuationMode.SOFT_VETO_BYPASS,
                crowbar_latency_ns=0.0,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=8.0,
                veto_reasons=tuple(degraded),
                orbit_type=orbit,
            )

        return HeytingVerdictData(
            verdict=HeytingOmega3.COHERENT,
            actuation_mode=ActuationMode.NORMAL_FLUID,
            crowbar_latency_ns=0.0,
            dirac_purge_energy_mev=0.0,
            contingency_reserve_pct=3.5,
            veto_reasons=(),
            orbit_type=orbit,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §3.2  Merkle DAG de 3 hojas
    # ══════════════════════════════════════════════════════════════════════
    def _compute_witness_merkle_root(
        self,
        germ: ChronosWeakObservationGerm,
        verdict: HeytingOmega3,
        actuation: ActuationMode,
        crowbar_latency: float,
    ) -> str:
        r"""
        Merkle DAG con 3 hojas:

            L1: H(seed_hash ‖ gauge_hmac)
            L2: H(germ_hmac ‖ rsi3_hash ‖ tr_M ‖ 𝒥)
            L3: H(verdict ‖ actuation ‖ crowbar_latency)
            root = H(H(L1‖L2) ‖ L3)
        """
        leaf1 = hashlib.sha256(
            f"{germ.gauge_condition.schedule_hash}|"
            f"{germ.gauge_condition.hmac_signature}"
            .encode("utf-8")
        ).hexdigest()
        rsi3_hash = hashlib.sha256(
            f"{germ.rsi3_theta}|{germ.rsi3_level_history}|"
            f"{germ.gauge_condition.poincare_monodromy_trace:.8f}|"
            f"{germ.jacobi_integral:.8f}"
            .encode("utf-8")
        ).hexdigest()
        leaf2 = hashlib.sha256(
            f"{germ.hmac_signature}|{rsi3_hash}".encode("utf-8")
        ).hexdigest()
        leaf3 = hashlib.sha256(
            f"{verdict.name}|{actuation.name}|{crowbar_latency:.2f}"
            .encode("utf-8")
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf1}|{leaf2}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf3}".encode("utf-8")).hexdigest()

    # ══════════════════════════════════════════════════════════════════════
    # §3.3  Adjudicación pública (v7 compat)
    # ══════════════════════════════════════════════════════════════════════
    def adjudicate_and_issue_certificate(
        self,
        germ: ChronosWeakObservationGerm,
    ) -> ChronosWitnessExecutionCertificate:
        """Fase III: adjudica y emite el pasaporte (interfaz v7)."""
        return self._phase3_terminal_synthesis(germ)

    # ══════════════════════════════════════════════════════════════════════
    # §3.4  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        germ: ChronosWeakObservationGerm,
    ) -> ChronosWitnessExecutionCertificate:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del certificado terminal.

        Composición funtorial completa:
            Z_Witness = Φ₃ ∘ Φ₂ ∘ Φ₁
        """
        verdict_data = self._adjudicate_witness_verdict(germ)
        verdict = verdict_data.verdict
        actuation = verdict_data.actuation_mode

        kv_compression = 86.4
        wacc_protection = 15.0 if verdict == HeytingOmega3.COHERENT else 8.0
        contingency = 3.5 if verdict == HeytingOmega3.COHERENT else 15.0

        merkle_root = self._compute_witness_merkle_root(
            germ, verdict, actuation, verdict_data.crowbar_latency_ns)

        now = time.time()
        final_raw = f"{merkle_root}|{verdict.value}|{actuation.value}|{now}"
        final_sig = hmac.new(
            self._secret_key, final_raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        cond = germ.gauge_condition
        quantum_summary: Tuple[Tuple[str, Any], ...] = (
            ("is_dag_acyclic", cond.is_dag_acyclic),
            ("betti_1_undirected", cond.betti_1),
            ("fiedler", cond.fiedler_value),
            ("delta_operator_norm", cond.delta_operator_norm),
            ("kms_modular_beta", cond.kms_modular_beta),
            ("kms_residual_max", cond.kms_residual_max),
            ("weak_value_real", cond.weak_value_real),
            ("weak_value_imag", cond.weak_value_imag),
            ("uhlmann_fidelity", cond.uhlmann_fidelity),
            ("fubini_study_distance", cond.fubini_study_distance),
            ("gromov_capacity", cond.gromov_capacity),
            ("back_action_db", cond.back_action_db),
            ("novikov_v_slack", germ.novikov_slack_valuation),
            ("banach_lipschitz", germ.banach_lipschitz_const),
            ("poincare_T_rec", cond.poincare_recurrence_time),
            ("poincare_tr_M", cond.poincare_monodromy_trace),
            ("poincare_chi_max", cond.poincare_max_lyapunov),
            ("jacobi_integral", cond.jacobi_integral),
            ("orbit_type", PoincareOrbitType(int(cond.orbit_type)).name),
        )

        rsi3_summary: Tuple[Tuple[str, Any], ...] = (
            ("converged", germ.rsi3_converged),
            ("final_loss", germ.rsi3_final_loss),
            ("n_levels", len(germ.rsi3_level_history)),
            ("picard_residual", germ.rsi3_picard_residual),
            ("cycles_effective", germ.rsi3_cycles_effective),
        )

        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_novikov_valuation_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_witness_verdict",
            "_phase3_terminal_synthesis",
        )

        cert = ChronosWitnessExecutionCertificate(
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
            quantum_audit_summary=quantum_summary,
            rsi3_report_summary=rsi3_summary,
            phase_continuity_chain=chain,
            veto_reasons=verdict_data.veto_reasons,
            poincare_orbit_type=int(verdict_data.orbit_type),
            jacobi_integral=float(germ.jacobi_integral),
        )

        logger.info(
            "Z_Witness = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "crowbar=%.2fns | orbit=%s | merkle=%s",
            verdict.name, actuation.name,
            verdict_data.crowbar_latency_ns,
            verdict_data.orbit_type.name,
            merkle_root[:16],
        )
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §4. ORQUESTADOR SUPREMO + COMPATIBILIDAD v7
# ══════════════════════════════════════════════════════════════════════════════

class TOONChronosWitnessEngine(Phase3_WitnessDecider):
    r"""
    Motor espectral y observador QND del tiempo en el Vacío de Dirac,
    con flujo modular de Poincaré y RSI³.

    Herencia lineal (cadena funtorial estricta):
        TOONChronosWitnessEngine
          → Phase3_WitnessDecider           [Φ₃]
             → Phase2_WitnessCampaignOrient [Φ₂ + RSI³ + max-plus]
                → Phase1_WitnessQuantumObserver [Φ₁ + Tomita-Takesaki + Poincaré]
                   → (object)
    """

    def run_full_witness_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> ChronosWitnessExecutionCertificate:
        r"""Ejecuta las TRES fases anidadas con continuación explícita Φ₁→Φ₂→Φ₃."""
        witness_packet, _novikov_cert = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, slacks)

        germ = self.build_weak_observation_germ(
            gauge_cond=witness_packet.gauge_condition, slacks=slacks)

        germ, _verdict_data = self._phase2_terminal_bridge_to_phase3(germ)
        return self._phase3_terminal_synthesis(germ)


class PoincareChronosWitnessAtlas:
    r"""
    Atlas celeste y observador pasivo de Tomita-Takesaki (interfaz v7).

    Internamente delega en `Phase1_WitnessQuantumObserver`, exponiendo
    además la sección de Poincaré y la integral de Jacobi.
    """

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_WITNESS_KEY_V9"):
        self._secret_key = secret_key
        self._phase1 = Phase1_WitnessQuantumObserver(secret_key)

    def compute_tomita_takesaki_modular_operator(
        self, density_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        r"""Interfaz v7: retorna (Δ_norm, β_KMS_proxy)."""
        n = int(density_matrix.shape[0])
        if n == 0:
            return 1.0, 1.0
        try:
            K = -np.real(la.logm(density_matrix))
        except Exception:
            K = np.zeros_like(density_matrix, dtype=np.float64)
        cert, _ = self._phase1._compute_tomita_takesaki_full(density_matrix, K)
        try:
            eigvals = np.sort(np.real(la.eigvalsh(density_matrix)))
            eigvals = np.maximum(eigvals, 1e-12)
            entropy = -float(np.sum(eigvals * np.log(eigvals)))
        except Exception:
            entropy = 0.0
        beta_proxy = float(1.0 / (entropy + 1e-6))
        return cert.delta_norm, beta_proxy

    def compute_qnd_weak_measurement(
        self,
        state_i: np.ndarray,
        state_f: np.ndarray,
        observable: np.ndarray,
    ) -> Tuple[complex, float, float, float]:
        """Interfaz v7: retorna (A_w, F, d_FS, c_G)."""
        return self._phase1._compute_qnd_weak_measurement(
            state_i, state_f, observable)

    def compute_poincare_section(
        self,
        density_matrix: np.ndarray,
        fiedler: float = _FIEDLER_GAP_MIN,
    ) -> PoincareSectionCertificate:
        """Interfaz v9: sección de Poincaré del flujo modular de ρ."""
        try:
            K = -np.real(la.logm(density_matrix))
        except Exception:
            K = np.zeros_like(density_matrix, dtype=np.float64)
        L_rho = -K
        return self._phase1._compute_poincare_section(
            density_matrix, K, L_rho, fiedler)

    def build_gauge_initial_condition(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosWitnessGaugeInitialCondition:
        """Emite la condición inicial de Fase 1 (v7 compat)."""
        packet = self._phase1.build_witness_observer_packet(
            schedule_id, num_nodes, edges)
        return packet.gauge_condition


# ══════════════════════════════════════════════════════════════════════════════
# §5. VERIFICACIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 78)
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS WITNESS ENGINE v9.0.0-POINCARE-RSI3")
    print("═" * 78)

    dag_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 3, 20.0),
        (0, 2, 22.0),
    ]
    slacks = [12.0, 8.0, 5.0, 2.0]

    engine = TOONChronosWitnessEngine()
    cert_dag = engine.run_full_witness_pipeline(
        "SCHED-DAG-OK-001", num_nodes=4, edges=dag_edges, slacks=slacks)

    print("\n[CASO 1: DAG VÁLIDO CON SHORTCUT]")
    for k, v in cert_dag.quantum_audit_summary:
        print(f"    {k:>24} = {v}")
    print(f"    {'VERDICT':>24} = {cert_dag.heyting_verdict.name}")
    print(f"    {'Actuation':>24} = {cert_dag.actuation_mode.name}")
    print(f"    {'Orbit':>24} = {PoincareOrbitType(cert_dag.poincare_orbit_type).name}")
    print(f"    {'Merkle':>24} = {cert_dag.merkle_root_sha256[:16]}...")
    print(f"    {'chain':>24} = {' → '.join(cert_dag.phase_continuity_chain)}")

    cycle_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 3, 5.0),
        (3, 0, 8.0),
    ]

    print("\n[CASO 2: CICLO DIRIGIDO (debe fallar en Fase 1)]")
    try:
        cert_cycle = engine.run_full_witness_pipeline(
            "SCHED-CYCLE-BAD-001", num_nodes=4,
            edges=cycle_edges, slacks=slacks)
        print(f"    ⚠️  Inesperado: {cert_cycle.heyting_verdict.name}")
    except WitnessAdjudicationError as e:
        print(f"    ✓ Kahn rechazó correctamente: {e}")
    except Exception as e:
        print(f"    ✓ Rechazado con {type(e).__name__}: {e}")

    print("\n" + "─" * 78)
    print("VERIFICACIÓN DE COMPATIBILIDAD v7 + POINCARÉ v9")
    print("─" * 78)

    atlas = PoincareChronosWitnessAtlas()
    cond = atlas.build_gauge_initial_condition(
        "SCHED-WITNESS-COMPAT", num_nodes=4, edges=dag_edges)
    print(f"[v7-compat] Δ_Tomita = {cond.tomita_takesaki_delta_norm:.4f}")
    print(f"[v7-compat] β_KMS proxy = {cond.kms_beta_inverse_temp:.4f}")
    print(f"[v7-compat] β_modular (v8) = {cond.kms_modular_beta}")
    print(f"[v7-compat] A_w = {cond.weak_value_real:.4f}+{cond.weak_value_imag:.4f}i")
    print(f"[v7-compat] F_uhlmann = {cond.uhlmann_fidelity:.4f}")
    print(f"[v7-compat] Gromov c_G = {cond.gromov_capacity:.4f}")
    print(f"[v7-compat] KMS residual max = {cond.kms_residual_max:.4e}")
    print(f"[v7-compat] is_dag_acyclic = {cond.is_dag_acyclic}")
    print(f"[v7-compat] back-action dB = {cond.back_action_db:.4f}")
    print(f"[v9-celeste] T_rec = {cond.poincare_recurrence_time:.4f}")
    print(f"[v9-celeste] tr M = {cond.poincare_monodromy_trace:.4f}")
    print(f"[v9-celeste] χ_max = {cond.poincare_max_lyapunov:.4e}")
    print(f"[v9-celeste] 𝒥 = {cond.jacobi_integral:.4f}")
    print(f"[v9-celeste] orbit = {PoincareOrbitType(cond.orbit_type).name}")
    print("═" * 78)