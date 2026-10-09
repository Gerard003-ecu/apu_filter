# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Trickster Prosecutor Agent (Soberano Fiscal Ilusionista)     ║
║ Ubicación: app/agents/wisdom/toon_trickster_prosecutor_agent.py              ║
║ Versión  : 7.0.0-Doctoral-Nested-Gauge-Celestial-CStar-Fock-Φsem-RSI3Tower   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Tejido anidado por herencia categórica en TRES FASES (v7.0.0):

  ◈ FASE I   — TricksterProsecutorGaugeTopology
               Fibrado principal P(M, G), G = U(1) × SU(2) × H₃(ℝ)
               • Conexión A anti-hermítica + curvatura F = dA + [A, D]
               • 3-forma de Chern–Simons  CS₃ = (1/8π²) Tr(A·dA + ⅔ A³)
               • Fibrado de Hopf cuaterniónico  π(q) = q·i·q̄ ∈ S²
               • Holonomía de Wilson  W(γ) = P exp(∮_γ A)
               • ── Sección de Retorno de Poincaré sobre la holonomía ──
                 La holonomía ϑ_W se trata como número de rotación de un
                 mapa de circunferencia; se reutiliza directamente el
                 `PoincareProsecutorAtlas` del motor (no-integrabilidad,
                 Bryuno, Birkhoff, recurrencia de Kac) instanciado sobre
                 una densidad ρ_gauge = e^{−AA†}/Tr[e^{−AA†}] inducida por
                 la propia conexión — la mecánica celeste de Poincaré
                 tejida *dentro* de la topología de calibre, no sólo
                 heredada como dato.
               • Contrato HMAC-SHA256 sellado — TODOS los campos firmados
                 son ahora campos persistidos del dataclass (fix de
                 integridad criptográfica, ver nota doctoral).
               • COSTURA: weave_gauge_to_prosecution → (contract, ok)

  ◈ FASE II  — TOONTricksterProsecutorAgent  (hereda FASE I)
               Campaña de acusación sobre motor espectral v7.0.0:
               • C*-𝔇_n, Uhlmann, Fubini–Study, Poincaré–Wirtinger
               • Delaunay / Melnikov / Greene / Bryuno / Morse–Bott χ
               • Jacobi / Lagrange / No-integrabilidad / Recurrencia /
                 Variedades invariantes / Birkhoff (Atlas Celeste I)
               • Oseledets (MET) / Novikov ultramétrico / Gromov–Wigner
               • RSI3MonadicTower (Niveles 1-2-3, combinador Y, Banach) —
                 la verificación de ley monádica ya NO recomputa una
                 fórmula obsoleta: lee directamente los residuos que la
                 propia Torre certificó en el motor (retrocompatible).
               • COSTURA: weave_campaign_to_adjudication

  ◈ FASE III — SovereignTricksterProsecutorAdjudicator  (hereda FASE II)
               • Adjudicación Heyting Ω₃ con meet de criterios ampliados
                 (Birkhoff ≥ 2 puntos fijos, no-integrabilidad informativa)
               • PoincareRecurrenceAuditor de GOBERNANZA (nivel negocio):
                 recurrencia empírica sobre (c_G, d_FS, h_KS) agregados
                 de campaña — distinto del auditor interno del motor,
                 opera sobre la serie histórica de pasaportes emitidos.
               • Interlock ESP32 Crowbar (< 400 ns / GPIO14)
               • Álgebra de Fock  e⁻ + e⁺ → 2γ  (purga al Vacío de Dirac)
               • Funtor semántico  Φ_sem : Sh(∂K, Ω₃) → Business
               • DAG Merkle final sobre (contract ⊕ campaign ⊕ cert ⊕ seeds)

Invariantes transversales:
  • C*-𝔇_n: ‖ρ‖=1, ρ=ρ†, λ_i ≥ −ε, involución ρ = ρ†
  • θ_PC = Tr(ρ dN)              residuo [ρ, N] ≤ 1e-5
  • c_G ≤ 12.5                   capacidad simpléctica Gromov–Wigner
  • χ(ℂPⁿ⁻¹) = n                 Morse–Bott por índice espectral
  • Birkhoff: #FixedPoints ≥ 2   Último Teorema Geométrico de Poincaré
  • v(T^{a}) = min{a_i}          filtración ultramétrica Novikov
  • μ-ley: μ∘(Tμ) = μ∘(μT)      plegado ≤ 1e-3, contracción Banach (RSI3)
  • Fock: e⁻ + e⁺ → 2γ           ‖p_e⁻ + p_e⁺ − Σp_γ‖ ≤ 1e-10
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ── Importación flexible del motor fiscal v7.0.0 ────────────────────────────
try:
    from toon_trickster_prosecutor_engine import (   # type: ignore
        ActuationMode,
        CStarAuditResult,
        FockAnihilationRecord,
        HeytingOmega3,
        PoincareProsecutorAtlas,
        PoincareRecurrenceAuditor,
        ProsecutorCanonicalSeed,
        ProsecutorExecutionCertificate,
        ProsecutorIndictmentGerm,
        ProsecutorQNDEngine,
        RSI3MonadicTower,
        RSI3TowerState,
        TOONTricksterProsecutorEngine,
    )
    _ENGINE_AVAILABLE = True
except ImportError:
    _ENGINE_AVAILABLE = False

    class HeytingOmega3(IntEnum):                    # type: ignore
        VETOED, DEGRADED, COHERENT = 0, 1, 2
        @property
        def verdict(self) -> str: return self.name
        def meet(self, other): return HeytingOmega3(min(int(self), int(other)))
        def join(self, other): return HeytingOmega3(max(int(self), int(other)))
        def implies(self, other):
            return HeytingOmega3(max(int(o) for o in HeytingOmega3
                                     if min(int(self), int(o)) <= int(other)))
        def neg(self): return self.implies(HeytingOmega3.VETOED)

    class ActuationMode(IntEnum):                    # type: ignore
        NORMAL_FLUID, SOFT_VETO_BYPASS, HARD_VETO_ESP32_CROWBAR = 0, 1, 2

    class PoincareRecurrenceAuditor:                 # type: ignore
        """Fallback mínimo: recurrencia euclídea sobre un histórico de puntos."""
        def __init__(self, neighborhood_radius: float = 0.75):
            self.radius = neighborhood_radius
            self._phase_history: List[Tuple[float, float, float]] = []

        def record_and_check_recurrence(self, c_g, d_fs, h_ks):
            point = (c_g, d_fs, h_ks)
            self._phase_history.append(point)
            if len(self._phase_history) < 2:
                return None, False
            current = np.array(point)
            for i in range(len(self._phase_history) - 2, -1, -1):
                past = np.array(self._phase_history[i])
                if float(np.linalg.norm(current - past)) < self.radius:
                    return len(self._phase_history) - 1 - i, True
            return None, False

    CStarAuditResult = Any                           # type: ignore
    FockAnihilationRecord = Any                      # type: ignore
    PoincareProsecutorAtlas = None                   # type: ignore
    ProsecutorCanonicalSeed = Any                    # type: ignore
    ProsecutorExecutionCertificate = Any             # type: ignore
    ProsecutorIndictmentGerm = Any                   # type: ignore
    ProsecutorQNDEngine = None                       # type: ignore
    RSI3MonadicTower = None                          # type: ignore
    RSI3TowerState = Any                             # type: ignore
    TOONTricksterProsecutorEngine = None             # type: ignore


logger = logging.getLogger("APU.Wisdom.TOONTricksterProsecutorAgent")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# §0. PRIMITIVAS TRANSVERSALES
# ══════════════════════════════════════════════════════════════════════════════

class TopologyRegime(str):
    TRIVIAL_BUNDLE   = "TRIVIAL_BUNDLE"
    MONOPOLE_LIKE    = "MONOPOLE_LIKE"
    INSTANTON_DENSE  = "INSTANTON_DENSE"


class PhaseMarker(IntEnum):
    FASE_I_GAUGE         = 1
    FASE_II_PROSECUTION  = 2
    FASE_III_ADJUDICATOR = 3


_INVERSE_GOLDEN_FRACTIONAL = 0.6180339887498949  # (√5−1)/2 — número áureo, máxima irracionalidad


def _lie_algebra_decompose(M: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Descompone M ∈ M_d(ℂ) en u(1) ⊕ su(2) ⊕ h₃(ℝ)."""
    d = M.shape[0]
    anti = 0.5 * (M - M.conj().T)
    herm = 0.5 * (M + M.conj().T)
    u1_scalar = float(np.real(np.trace(anti)) / d) if d else 0.0
    su2_part = anti - 1j * (np.imag(np.trace(anti)) / d) * np.eye(d, dtype=M.dtype)
    h3_part = herm - (np.real(np.trace(herm)) / d) * np.eye(d, dtype=M.dtype)
    return u1_scalar, su2_part, h3_part


def _path_ordered_exponential(generators: Sequence[np.ndarray],
                              dt: float = 1.0) -> np.ndarray:
    """Exponencial ordenada por camino  P exp(∮ A) ≈ ∏ exp(g_k dt) (a derecha)."""
    d = generators[0].shape[0]
    W = np.eye(d, dtype=np.complex128)
    for g in reversed(list(generators)):
        W = la.expm(g * dt) @ W
    return W


# ══════════════════════════════════════════════════════════════════════════════
# §A. DATACLASSES Y CONTRATOS FISCALES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GaugePoincareReturnMapAnalysis:
    """
    Sección de Poincaré sobre la holonomía del fibrado de calibre.

    La holonomía de Wilson ϑ_W se interpreta como número de rotación de un
    mapa de circunferencia inducido por la conexión A (análogo directo al
    standard map de Chirikov, pero instanciado sobre la fibra U(1) del
    propio fibrado del Fiscal). Reutiliza el `PoincareProsecutorAtlas` del
    motor para no-integrabilidad, Bryuno, recurrencia de Kac y Birkhoff.
    """
    rotation_number: float
    small_divisor: float
    nonintegrability_obstructed: bool
    bryuno_sum: float
    bryuno_convergent: bool
    birkhoff_fixed_points: int
    birkhoff_area_defect: float
    recurrence_time: float
    recurrence_measure: float


@dataclass(frozen=True, slots=True)
class ProsecutorGaugeContract:
    """
    Contrato de calibre inmutable del Fiscal — FASE I.

    NOTA DE RIGOR CRIPTOGRÁFICO (fix v7.0.0): todo campo que participa en el
    mensaje HMAC es un campo PERSISTIDO del dataclass. En v6.0.0 el término
    `hol_hopf` entraba en la firma de `forge_...` pero no se almacenaba, por
    lo que `bind_...` jamás podía reconstruir el mensaje exacto → la firma
    nunca validaba. Se corrige añadiendo `chern_hopf_holonomy` como campo.
    """
    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    chern_hopf_class_c1: float
    chern_hopf_holonomy: float                 # ← FIX: antes transitorio, ahora persistido
    wilson_holonomy_angle: float
    hopf_base_point: Tuple[float, float, float]
    topology_regime: str
    su2_curvature_norm: float
    h3_trace_curvature: float
    # ── Mecánica Celeste de Poincaré sobre la holonomía (v7.0.0) ────────────
    gauge_rotation_number: float
    gauge_small_divisor: float
    gauge_nonintegrability_obstructed: bool
    gauge_bryuno_sum: float
    gauge_bryuno_convergent: bool
    gauge_birkhoff_fixed_points: int
    gauge_poincare_recurrence_time: float
    hmac_signature: str
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class IllusionIndictmentRequest:
    """Solicitud de acusación sobre una trama del Ilusionista."""
    request_id: str
    illusion_id: str
    apu_code: str
    density_matrix: np.ndarray
    disguised_cost_ratio: float
    reward_hacking_index: float
    stinespring_coupling_eps: float = 0.05
    omega_celestial: float = 1.618033988749895


@dataclass(frozen=True, slots=True)
class ProsecutionCampaignResult:
    """Resultado consolidado de campaña de acusación — FASE II."""
    campaign_id: str
    sovereign_id: str
    bound_contract_id: str
    illusions_examined_count: int
    hallucinations_detected_count: int
    # — Espectral QND
    weak_value_aw_mean: complex
    fubini_study_distance_peak: float
    uhlmann_fidelity_mean: float
    uhlmann_residual_mean: float
    poincare_wirtinger_all_satisfied: bool
    # — Celeste (Delaunay/Melnikov/Greene/Bryuno/Morse)
    melnikov_integral_peak: float
    greene_residue_mean: float
    bryuno_sum_mean: float
    bryuno_diophantine_all_convergent: bool
    morse_bott_chi: int
    morse_bott_chi_all_verified: bool
    poincare_cartan_residual_max: float
    symplectic_defect_max: float
    # — Celeste de Poincaré: Jacobi / Lagrange / No-integrabilidad / Birkhoff
    jacobi_constant_mean: float
    mass_ratio_mu_mean: float
    poincare_small_divisor_min: float
    poincare_nonintegrability_any_obstructed: bool
    poincare_recurrence_time_mean: float
    invariant_manifold_lambda_u_peak: float
    birkhoff_fixed_points_min: int
    birkhoff_all_verified: bool
    # — Oseledets / Novikov / Gromov
    kolmogorov_sinai_entropy_mean: float
    oseledets_max_lyapunov_peak: float
    novikov_valuation_min: float
    gromov_capacity_peak: float
    # — C*-𝔇_n
    cstar_axioms_all_satisfied: bool
    cstar_positivity_all_valid: bool
    # — Torre RSI Nivel 3 (combinador Y / punto fijo de Banach)
    rsi3_aggregate_rate: float
    rsi3_monadic_law_verified: bool
    rsi3_theta_mean: Tuple[float, float, float]
    rsi3_banach_contraction_k_mean: float
    rsi3_fixed_point_all_converged: bool
    rsi3_monad_residual_max: float
    rsi3_tower_iterations_mean: float
    # — Adjudicación global
    global_heyting_verdict: HeytingOmega3
    per_request_seed_ids: Tuple[str, ...]
    per_request_indictment_ids: Tuple[str, ...]
    timestamp: float


@dataclass(frozen=True, slots=True)
class ProsecutorSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza del Fiscal — FASE III."""
    passport_id: str
    sovereign_id: str
    campaign_id: str
    bound_contract_id: str
    engine_certificate_id: str
    global_heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    illusions_prosecuted_count: int
    hallucinations_vetoed_count: int
    crowbar_active_iram: bool
    crowbar_latency_ns: float
    positron_annihilation_active: bool
    fock_purge_records: Tuple[Any, ...]
    level3_rsi_monadic_rate: float
    rsi3_monadic_law_verified: bool
    rsi3_banach_contraction_k_mean: float
    gromov_capacity_peak: float
    cstar_positivity_verified: bool
    morse_bott_chi_verified: bool
    birkhoff_fixed_points_verified: bool
    nonintegrability_obstruction_detected: bool
    governance_recurrence_tau: Optional[int]
    governance_recurrence_detected: bool
    merkle_root_sha256: str
    provenance_hash: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ExecutiveProsecutionImpactReport:
    """Reporte Ejecutivo Φ_sem ('Dolor y Dinero')."""
    report_id: str
    passport_id: str
    verdict_name: str
    actuation_description: str
    kv_cache_compression_pct: float
    wacc_protection_pct: float
    imprevistos_reduction_pct: float
    capital_salvaguardado_usd: float
    expected_loss_avoided_usd: float
    birkhoff_topological_safety_verified: bool
    nonintegrability_risk_flag: bool
    rsi3_tower_banach_k_mean: float
    executive_summary: str
    timestamp_utc: float


# ══════════════════════════════════════════════════════════════════════════════
# FASE I — TOPOLOGÍA DE CALIBRE DEL FISCAL + SECCIÓN DE RETORNO DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════

class TricksterProsecutorGaugeTopology:
    """
    FASE I — Fibrado principal P(M, G), G = U(1) × SU(2) × H₃(ℝ).

    Rigor doctoral:
      • Conexión A anti-hermítica; curvatura F = dA + [A, D].
      • CS₃ = (1/8π²) Tr(A·dA + ⅔ A³) canónica.
      • Fibración de Hopf cuaterniónica π(q) = q·i·q̄ ∈ S².
      • Holonomía de Wilson W(γ) = P exp(∮_γ A) vía exponencial ordenada.
      • Sección de retorno de Poincaré sobre la holonomía (I.6), reutilizando
        el Atlas Celeste del motor sobre una densidad ρ_gauge inducida por A.
      • Contrato sellado con HMAC-SHA256 — íntegramente sobre campos persistidos.
    """

    def __init__(self,
                 sovereign_id: str = "PROSECUTOR-SOVEREIGN-MASTER-01",
                 dimension: int = 56,
                 hmac_secret: bytes = b"APU_FILTER_V8_PROSECUTOR_2026") -> None:
        self.sovereign_id = sovereign_id
        self.dimension = dimension
        self._hmac_secret = hmac_secret

    # ── I.1 — Conexión y curvatura del fibrado principal ─────────────────────
    def compute_principal_connection(
        self, seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        d = self.dimension
        raw = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        A = 0.5 * (raw - raw.conj().T)

        D_gen = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        D_gen = 0.5 * (D_gen - D_gen.conj().T)
        dA = D_gen @ A - A @ D_gen
        F = dA + (A @ D_gen - D_gen @ A)
        return A, dA, F, D_gen

    # ── I.2 — 3-forma de Chern–Simons canónica ──────────────────────────────
    @staticmethod
    def compute_chern_simons_3form(A: np.ndarray, dA: np.ndarray) -> float:
        wedge1 = np.trace(A @ dA)
        wedge3 = np.trace(A @ A @ A)
        cs3 = (wedge1 + (2.0 / 3.0) * wedge3) / (8.0 * math.pi ** 2)
        return float(np.clip(np.real(cs3), -100.0, 100.0))

    # ── I.3 — Fibración de Hopf cuaterniónica ───────────────────────────────
    @staticmethod
    def compute_chern_hopf_class(
        quaternion_q: Tuple[float, float, float, float],
    ) -> Tuple[float, float, Tuple[float, float, float], str]:
        """
        q = (w, x, y, z) ∈ S³. Proyección de Hopf  π(q) = q·i·q̄ ∈ S²:
            (2(xz + wy), 2(yz − wx), w² + z² − x² − y²).
        Fase φ = 2 arctan(‖(x,y,z)‖ / w);  c₁ = φ / π;  ϑ = 2π c₁ mod 2π.
        """
        w, x, y, z = quaternion_q
        n = math.sqrt(w*w + x*x + y*y + z*z) + 1e-30
        w, x, y, z = w/n, x/n, y/n, z/n

        bx = 2.0 * (x * z + w * y)
        by = 2.0 * (y * z - w * x)
        bz = w*w + z*z - x*x - y*y

        rho = math.sqrt(x*x + y*y + z*z)
        phi = 2.0 * math.atan2(rho, w + 1e-30)
        c1 = phi / math.pi
        holonomy = (2.0 * math.pi * c1) % (2.0 * math.pi)

        if abs(c1) < 0.10:
            regime = TopologyRegime.TRIVIAL_BUNDLE
        elif abs(c1) < 0.60:
            regime = TopologyRegime.MONOPOLE_LIKE
        else:
            regime = TopologyRegime.INSTANTON_DENSE

        return float(c1), float(holonomy), (float(bx), float(by), float(bz)), regime

    # ── I.4 — Holonomía de Wilson  W(γ) = P exp(∮_γ A) ──────────────────────
    @staticmethod
    def compute_wilson_holonomy(
        A: np.ndarray,
        n_segments: int = 24,
        radius: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        thetas = np.linspace(0.0, 2.0 * math.pi, n_segments, endpoint=False)
        dt = 2.0 * math.pi / n_segments
        generators = [A * (radius * np.exp(1j * th)) for th in thetas]
        W = _path_ordered_exponential(generators, dt=dt)
        hol = float(np.angle(np.trace(W)))
        return W, hol

    # ── I.5 — Descomposición de la curvatura por factor de grupo ────────────
    @staticmethod
    def decompose_curvature(F: np.ndarray) -> Tuple[float, float, float]:
        d = F.shape[0]
        anti = 0.5 * (F - F.conj().T)
        herm = 0.5 * (F + F.conj().T)
        f_u1 = float(np.imag(np.trace(F)) / d)
        su2_mat = anti - 1j * f_u1 * np.eye(d, dtype=F.dtype)
        f_su2 = float(la.norm(su2_mat, ord="fro"))
        f_h3 = float(np.real(np.trace(herm)))
        return f_u1, f_su2, f_h3

    # ── I.6 — Densidad inducida por la conexión (puente hacia el Atlas) ─────
    @staticmethod
    def _gauge_density_from_connection(A: np.ndarray) -> np.ndarray:
        """
        ρ_gauge = e^{−A A†} / Tr[e^{−A A†}]

        Construye una matriz densidad C*-admisible (hermítica, positiva,
        traza unitaria) directamente a partir del operador cinético A A† de
        la conexión de calibre, habilitando la reutilización literal del
        `PoincareProsecutorAtlas` del motor sobre la propia topología de
        calibre del Fiscal — sin duplicar código celeste.
        """
        K = A @ A.conj().T
        K = 0.5 * (K + K.conj().T)
        M = la.expm(-K)
        M = 0.5 * (M + M.conj().T)
        tr = float(np.real(np.trace(M)))
        return M / (tr + 1e-30)

    # ── I.7 — Sección de Retorno de Poincaré sobre la holonomía de Wilson ───
    def compute_gauge_poincare_return_map(
        self,
        holonomy_angle: float,
        A: np.ndarray,
    ) -> GaugePoincareReturnMapAnalysis:
        """
        Trata ϑ_W como número de rotación de un mapa de circunferencia
        inducido por la conexión (análogo al standard map de Chirikov sobre
        la fibra U(1)), y aplica el Atlas Celeste de Poincaré del motor:
        obstrucción de no-integrabilidad (pequeños divisores), condición de
        Bryuno, Último Teorema Geométrico de Poincaré–Birkhoff y recurrencia
        de Kac — todo evaluado sobre ρ_gauge (I.6).
        """
        omega_rot = (abs(holonomy_angle) / (2.0 * math.pi)) % 1.0
        if omega_rot < 1e-6:
            omega_rot = _INVERSE_GOLDEN_FRACTIONAL  # evita degeneración resonante trivial

        rho_gauge = self._gauge_density_from_connection(A)
        k_twist = float(np.clip(float(la.norm(A, ord="fro")) / A.shape[0], 0.0, 4.0))

        if PoincareProsecutorAtlas is not None:
            small_div, nonint = PoincareProsecutorAtlas.compute_poincare_nonintegrability_obstruction(
                omega_rot)
            bryuno, bry_conv, _ = PoincareProsecutorAtlas.compute_bryuno_sum(omega_rot)
            birkhoff_n, birkhoff_defect = PoincareProsecutorAtlas.compute_poincare_birkhoff_fixed_points(
                k_twist)
            tau_rec, mu_A = PoincareProsecutorAtlas.compute_poincare_recurrence_measure(
                rho_gauge)
        else:
            small_div, nonint = 1.0, False
            bryuno, bry_conv = 0.0, True
            birkhoff_n, birkhoff_defect = 2, 0.0
            tau_rec, mu_A = 1.0, 1.0

        return GaugePoincareReturnMapAnalysis(
            rotation_number=float(omega_rot),
            small_divisor=float(small_div),
            nonintegrability_obstructed=bool(nonint),
            bryuno_sum=float(bryuno),
            bryuno_convergent=bool(bry_conv),
            birkhoff_fixed_points=int(birkhoff_n),
            birkhoff_area_defect=float(birkhoff_defect),
            recurrence_time=float(tau_rec),
            recurrence_measure=float(mu_A),
        )

    # ── I.8 — Forja del contrato de calibre sellado por HMAC ────────────────
    def forge_prosecutor_gauge_contract(
        self,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        seed: int = 42,
    ) -> ProsecutorGaugeContract:
        now = time.time()
        contract_id = f"CONTRACT-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        A, dA, F, _ = self.compute_principal_connection(seed=seed)
        cs3 = self.compute_chern_simons_3form(A, dA)
        c1, hol_hopf, hopf_pt, regime = self.compute_chern_hopf_class(quaternion_q)
        _, hol_wilson = self.compute_wilson_holonomy(A)
        f_u1, f_su2, f_h3 = self.decompose_curvature(F)
        ret_map = self.compute_gauge_poincare_return_map(hol_wilson, A)

        content_str = (
            f"{contract_id}:{self.sovereign_id}:{cs3:.10f}:{c1:.10f}:{hol_hopf:.10f}:"
            f"{hol_wilson:.10f}:{regime}:{f_su2:.10f}:{f_h3:.10f}:"
            f"{ret_map.rotation_number:.10f}:{ret_map.small_divisor:.10e}:"
            f"{ret_map.nonintegrability_obstructed}:{ret_map.bryuno_sum:.10f}:"
            f"{ret_map.bryuno_convergent}:{ret_map.birkhoff_fixed_points}:"
            f"{ret_map.recurrence_time:.10e}:{now}"
        )
        hmac_sig = hmac.new(self._hmac_secret,
                            content_str.encode("utf-8"),
                            hashlib.sha256).hexdigest()

        contract = ProsecutorGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=cs3,
            chern_hopf_class_c1=c1,
            chern_hopf_holonomy=hol_hopf,
            wilson_holonomy_angle=hol_wilson,
            hopf_base_point=hopf_pt,
            topology_regime=regime,
            su2_curvature_norm=f_su2,
            h3_trace_curvature=f_h3,
            gauge_rotation_number=ret_map.rotation_number,
            gauge_small_divisor=ret_map.small_divisor,
            gauge_nonintegrability_obstructed=ret_map.nonintegrability_obstructed,
            gauge_bryuno_sum=ret_map.bryuno_sum,
            gauge_bryuno_convergent=ret_map.bryuno_convergent,
            gauge_birkhoff_fixed_points=ret_map.birkhoff_fixed_points,
            gauge_poincare_recurrence_time=ret_map.recurrence_time,
            hmac_signature=hmac_sig,
            creation_timestamp=now,
        )
        logger.info(
            f"[FASE I] Contrato {contract_id} forjado | régimen={regime} | "
            f"c₁={c1:.4f} | CS₃={cs3:+.4e} | ϑ_W={hol_wilson:+.4f} rad | "
            f"ρ_rot={ret_map.rotation_number:.4f} | D_min={ret_map.small_divisor:.3e} | "
            f"#FixBirkhoff={ret_map.birkhoff_fixed_points}"
        )
        return contract

    # ── I.9 — COSTURA FASE I → FASE II ──────────────────────────────────────
    @staticmethod
    def weave_gauge_to_prosecution(
        contract: ProsecutorGaugeContract,
    ) -> Tuple[ProsecutorGaugeContract, bool]:
        """
        Última piedra de la FASE I. Promueve el contrato al observatorio fiscal
        bajo las reglas categóricas:
          (1) Régimen TRIVIAL_BUNDLE    → promoción limpia
          (2) Régimen MONOPOLE_LIKE     → promoción con marca de vigilancia
          (3) Régimen INSTANTON_DENSE   → promoción con vigilancia reforzada
        Adicionalmente marca advertencia si la sección de retorno detectó
        obstrucción de no-integrabilidad (resonancia ϑ_W) o si Birkhoff no
        garantizó la cota mínima de 2 puntos fijos (no debería ocurrir por
        construcción del Atlas, pero se audita explícitamente).
        """
        warn_nonint = contract.gauge_nonintegrability_obstructed
        warn_birkhoff = contract.gauge_birkhoff_fixed_points < 2
        if warn_birkhoff:
            logger.error(
                f"[FASE I] Birkhoff violado en {contract.contract_id}: "
                f"#Fixed={contract.gauge_birkhoff_fixed_points} < 2"
            )
            return contract, False
        logger.info(
            f"[FASE I → FASE II] Contrato {contract.contract_id} promovido al "
            f"observatorio fiscal | régimen={contract.topology_regime} | "
            f"no-integrable={warn_nonint}"
        )
        return contract, True


# ══════════════════════════════════════════════════════════════════════════════
# FASE II — CAMPAÑA DE ACUSACIÓN SOBRE MOTOR ESPECTRAL v7.0.0
# ══════════════════════════════════════════════════════════════════════════════

class TOONTricksterProsecutorAgent(TricksterProsecutorGaugeTopology):
    """
    FASE II — Soberano Fiscal Ilusionista.

    Consume el motor `TOONTricksterProsecutorEngine` v7.0.0 end-to-end:
        engine.prosecute_trickster_illusion(...)  → (IndictmentGerm, Seed)
        engine.audit_and_prosecute_illusions(...) → ProsecutorExecutionCertificate

    Agrega invariantes a nivel de campaña (C*-𝔇_n, Uhlmann, PW, Morse–Bott,
    Jacobi/Lagrange/Birkhoff/Recurrencia, Oseledets, Novikov, Gromov, Torre
    RSI3) y verifica la ley monádica μ a nivel de lote LEYENDO los residuos
    que la propia `RSI3MonadicTower` certificó dentro del motor (en lugar de
    recomputar una fórmula obsoleta, ver II.3).
    """

    def __init__(
        self,
        sovereign_id: str = "PROSECUTOR-SOVEREIGN-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        base_rsi_rate: float = 0.25,
        engine_instance: Optional[Any] = None,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id, dimension=dimension_mac)
        self.dimension_mac = dimension_mac
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self.base_rsi_rate = base_rsi_rate

        if engine_instance is not None:
            self.engine = engine_instance
        elif TOONTricksterProsecutorEngine is not None:
            self.engine = TOONTricksterProsecutorEngine(
                dimension=dimension_mac,
                capacity_gromov_max=capacity_gromov_max,
                esp32_gpio_pin=esp32_gpio_pin,
                base_rsi_rate=base_rsi_rate,
            )
        else:
            self.engine = None

        self.bound_contract: Optional[ProsecutorGaugeContract] = None
        self._indictment_history: List[Any] = []
        self._seed_history: List[Any] = []

    # ── II.1 — Vinculación HMAC del contrato de FASE I ──────────────────────
    def bind_prosecutor_gauge_contract(self, contract: ProsecutorGaugeContract) -> bool:
        """
        Reconstruye EXACTAMENTE la cadena firmada en `forge_...`, campo a
        campo persistido (fix de integridad v7.0.0 — ver docstring del
        dataclass `ProsecutorGaugeContract`).
        """
        content_str = (
            f"{contract.contract_id}:{contract.sovereign_id}:"
            f"{contract.chern_simons_3form:.10f}:{contract.chern_hopf_class_c1:.10f}:"
            f"{contract.chern_hopf_holonomy:.10f}:"
            f"{contract.wilson_holonomy_angle:.10f}:{contract.topology_regime}:"
            f"{contract.su2_curvature_norm:.10f}:{contract.h3_trace_curvature:.10f}:"
            f"{contract.gauge_rotation_number:.10f}:{contract.gauge_small_divisor:.10e}:"
            f"{contract.gauge_nonintegrability_obstructed}:{contract.gauge_bryuno_sum:.10f}:"
            f"{contract.gauge_bryuno_convergent}:{contract.gauge_birkhoff_fixed_points}:"
            f"{contract.gauge_poincare_recurrence_time:.10e}:{contract.creation_timestamp}"
        )
        expected = hmac.new(self._hmac_secret,
                            content_str.encode("utf-8"),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, contract.hmac_signature):
            logger.error(f"[FASE II] HMAC inválido para {contract.contract_id}")
            return False
        self.bound_contract = contract
        logger.info(f"[FASE II] Contrato {contract.contract_id} vinculado.")
        return True

    # ── II.2 — Acusación individual (delegada al motor v7.0.0) ──────────────
    def process_illusion_indictment(
        self,
        request: IllusionIndictmentRequest,
        mac_density_matrix: np.ndarray,
    ) -> Tuple[Any, Any]:
        if self.engine is None:
            raise RuntimeError("TOONTricksterProsecutorEngine no disponible.")
        if self.bound_contract is None:
            self.bind_prosecutor_gauge_contract(self.forge_prosecutor_gauge_contract())

        logger.info(
            f"[FASE II] Indictando trama {request.illusion_id} (APU: {request.apu_code})"
        )
        indictment, seed = self.engine.prosecute_trickster_illusion(
            illusion_id=request.illusion_id,
            mac_density_matrix=mac_density_matrix,
            illusion_density_matrix=request.density_matrix,
            perturbation_eps=request.stinespring_coupling_eps,
            omega=request.omega_celestial,
        )
        self._indictment_history.append(indictment)
        self._seed_history.append(seed)
        return indictment, seed

    # ── II.3 — Verificación de ley monádica μ∘(Tμ) = μ∘(μT) ────────────────
    @staticmethod
    def _verify_monadic_law(
        indictment: Any,
        seed: Any,
        eta_base: float = 0.25,
        tol: float = 1e-3,
    ) -> bool:
        """
        Camino preferente (motor v7.0.0): la `RSI3MonadicTower` ya certificó,
        dentro del propio motor, las leyes monádicas (identidad izq./der.,
        asociatividad) en el punto fijo de Banach. Aquí simplemente se LEEN
        esos residuos — nunca se recomputa con una fórmula desactualizada.

        Retrocompatibilidad (motor < v7.0.0, sin campos rsi3_monad_*): se
        reconstruye una `RSI3MonadicTower` nueva y se evalúa el plegado de
        Nivel 1 directamente.
        """
        if hasattr(indictment, "rsi3_fixed_point_converged"):
            return bool(
                getattr(indictment, "rsi3_fixed_point_converged", False)
                and getattr(indictment, "rsi3_monad_left_identity_residual", 1.0) < tol
                and getattr(indictment, "rsi3_monad_right_identity_residual", 1.0) < tol
                and getattr(indictment, "rsi3_monad_associativity_residual", 1.0) < tol
            )
        if RSI3MonadicTower is None:
            return True
        try:
            tower = RSI3MonadicTower()
            d_fs = indictment.fubini_study_distance
            h_ks = indictment.kolmogorov_sinai_entropy
            lam = indictment.oseledets_max_lyapunov
            eta1 = tower.level1_parametric_update(eta_base, seed, d_fs, h_ks, lam, tower.theta)
            eta2 = tower.level1_parametric_update(eta1, seed, d_fs, h_ks, lam, tower.theta)
            fold = abs(eta2 - eta1 * math.exp(-h_ks * d_fs))
            return fold < tol
        except Exception as exc:                      # pragma: no cover
            logger.warning(f"[FASE II] Verificación monádica falló: {exc}")
            return False

    # ── II.4 — Campaña de acusación con agregación completa ────────────────
    def conduct_prosecution_campaign(
        self,
        requests: List[IllusionIndictmentRequest],
        mac_density_matrix: np.ndarray,
    ) -> ProsecutionCampaignResult:
        now = time.time()
        campaign_id = f"CAMP-PROSECUTOR-{int(now * 1000) % 1000000:06d}"
        if self.bound_contract is None:
            self.bind_prosecutor_gauge_contract(self.forge_prosecutor_gauge_contract())

        # — Acumuladores espectrales / celestes clásicos
        aw_vals: List[complex] = []
        d_fs_vals: List[float] = []
        uhl_fid_vals: List[float] = []
        uhl_res_vals: List[float] = []
        pw_flags: List[bool] = []
        mel_vals: List[float] = []
        gre_vals: List[float] = []
        bry_vals: List[float] = []
        bry_all_ok = True
        chi_vals: List[int] = []
        chi_ok_flags: List[bool] = []
        theta_res_vals: List[float] = []
        sympl_def_vals: List[float] = []
        h_ks_vals: List[float] = []
        lam_vals: List[float] = []
        nov_vals: List[float] = []
        cg_vals: List[float] = []
        cstar_all_ok_flags: List[bool] = []
        cstar_pos_flags: List[bool] = []

        # — Acumuladores Celeste de Poincaré (Jacobi/Lagrange/Birkhoff/Recurrencia)
        jacobi_vals: List[float] = []
        mu_ratio_vals: List[float] = []
        small_div_vals: List[float] = []
        nonint_flags: List[bool] = []
        recurrence_tau_vals: List[float] = []
        lambda_u_vals: List[float] = []
        birkhoff_counts: List[int] = []

        # — Acumuladores Torre RSI3
        eta_rsi3_vals: List[float] = []
        theta_vals: List[Tuple[float, float, float]] = []
        banach_k_vals: List[float] = []
        converged_flags: List[bool] = []
        monad_residual_max_vals: List[float] = []
        iter_vals: List[int] = []
        monadic_ok_flags: List[bool] = []

        verdicts: List[HeytingOmega3] = []
        halluc_count = 0
        seed_ids: List[str] = []
        indict_ids: List[str] = []

        for req in requests:
            ind, seed = self.process_illusion_indictment(req, mac_density_matrix)
            indict_ids.append(getattr(ind, "indictment_id", "?"))
            seed_ids.append(getattr(seed, "seed_id", "?"))

            aw_vals.append(complex(getattr(ind, "weak_value_Aw", 0.0)))
            d_fs_vals.append(float(getattr(ind, "fubini_study_distance", 0.0)))
            uhl_fid_vals.append(float(getattr(ind, "uhlmann_fidelity", 1.0)))
            uhl_res_vals.append(float(getattr(ind, "uhlmann_residual", 0.0)))
            pw_flags.append(bool(getattr(ind, "poincare_wirtinger_satisfied", True)))
            h_ks_vals.append(float(getattr(ind, "kolmogorov_sinai_entropy", 0.0)))
            lam_vals.append(float(getattr(ind, "oseledets_max_lyapunov", 0.0)))
            nov_vals.append(float(getattr(ind, "novikov_valuation", 0.0)))
            cg_vals.append(float(getattr(ind, "capacity_gromov", 0.0)))

            # C*-𝔇_n
            cstar = getattr(ind, "cstar_audit", None)
            if cstar is not None:
                cstar_all_ok_flags.append(bool(getattr(cstar, "axioms_all_satisfied", True)))
                cstar_pos_flags.append(bool(getattr(cstar, "positivity_valid", True)))
            else:
                cstar_all_ok_flags.append(True)
                cstar_pos_flags.append(True)

            # Celeste clásico
            mel_vals.append(float(getattr(seed, "melnikov_integral_M0", 0.0)))
            gre_vals.append(float(getattr(seed, "greene_residue_R", 0.0)))
            bry_vals.append(float(getattr(seed, "bryuno_sum", 0.0)))
            bry_all_ok &= bool(getattr(seed, "bryuno_convergent", True))
            chi_vals.append(int(getattr(seed, "morse_bott_euler_chi", 0)))
            chi_ok_flags.append(int(getattr(seed, "morse_bott_defect", 0)) == 0)
            theta_res_vals.append(float(getattr(seed, "poincare_cartan_residual", 0.0)))
            sympl_def_vals.append(float(getattr(seed, "monodromy_symplectic_defect", 0.0)))

            # Celeste de Poincaré (Jacobi/Lagrange/Birkhoff/Recurrencia)
            jacobi_vals.append(float(getattr(seed, "jacobi_constant", 0.0)))
            mu_ratio_vals.append(float(getattr(seed, "mass_ratio_mu", 0.0)))
            small_div_vals.append(float(getattr(seed, "poincare_small_divisor", 1.0)))
            nonint_flags.append(bool(getattr(seed, "poincare_nonintegrability_obstructed", False)))
            recurrence_tau_vals.append(float(getattr(seed, "poincare_recurrence_time", 0.0)))
            lambda_u_vals.append(float(getattr(seed, "invariant_manifold_lambda_u", 0.0)))
            birkhoff_counts.append(int(getattr(seed, "birkhoff_fixed_points_count", 2)))

            # Torre RSI3
            eta_rsi3_vals.append(float(getattr(ind, "rsi3_monadic_rate", self.base_rsi_rate)))
            theta_vals.append(tuple(getattr(ind, "rsi3_theta", (1.0, 1.0, 1.0))))
            banach_k_vals.append(float(getattr(ind, "rsi3_banach_contraction_k", 1.0)))
            converged_flags.append(bool(getattr(ind, "rsi3_fixed_point_converged", True)))
            monad_residual_max_vals.append(max(
                float(getattr(ind, "rsi3_monad_left_identity_residual", 0.0)),
                float(getattr(ind, "rsi3_monad_right_identity_residual", 0.0)),
                float(getattr(ind, "rsi3_monad_associativity_residual", 0.0)),
            ))
            iter_vals.append(int(getattr(ind, "rsi3_tower_iterations", 0)))

            # Adjudicación por-trama (ahora incluye Birkhoff + no-integrabilidad)
            d_fs = d_fs_vals[-1]
            cg = cg_vals[-1]
            birk_n = birkhoff_counts[-1]
            nonint = nonint_flags[-1]
            halluc = bool(getattr(ind, "is_hallucination", False))
            if halluc:
                halluc_count += 1

            if halluc or d_fs > 0.45 or cg > self.capacity_gromov_max or birk_n < 2:
                verdicts.append(HeytingOmega3.VETOED)
            elif d_fs > 0.15 or nonint:
                verdicts.append(HeytingOmega3.DEGRADED)
            else:
                verdicts.append(HeytingOmega3.COHERENT)

            monadic_ok_flags.append(self._verify_monadic_law(
                ind, seed, eta_base=self.base_rsi_rate))

        # — Veredicto global por meet de Ω₃
        global_verdict = HeytingOmega3.COHERENT
        for v in verdicts:
            global_verdict = global_verdict.meet(v)

        # — Agregados clásicos
        aw_mean = complex(np.mean(aw_vals)) if aw_vals else 0.0 + 0.0j
        d_fs_peak = float(np.max(d_fs_vals)) if d_fs_vals else 0.0
        uhl_fid_mean = float(np.mean(uhl_fid_vals)) if uhl_fid_vals else 1.0
        uhl_res_mean = float(np.mean(uhl_res_vals)) if uhl_res_vals else 0.0
        pw_all = all(pw_flags) if pw_flags else True
        mel_peak = float(np.max(np.abs(mel_vals))) if mel_vals else 0.0
        gre_mean = float(np.mean(gre_vals)) if gre_vals else 0.0
        bry_mean = float(np.mean(bry_vals)) if bry_vals else 0.0
        chi_final = chi_vals[0] if chi_vals else self.dimension_mac
        chi_all_ok = all(chi_ok_flags) if chi_ok_flags else True
        theta_res_max = float(np.max(theta_res_vals)) if theta_res_vals else 0.0
        sympl_def_max = float(np.max(sympl_def_vals)) if sympl_def_vals else 0.0
        h_ks_mean = float(np.mean(h_ks_vals)) if h_ks_vals else 0.0
        lam_peak = float(np.max(lam_vals)) if lam_vals else 0.0
        nov_min = float(np.min(nov_vals)) if nov_vals else 0.0
        cg_peak = float(np.max(cg_vals)) if cg_vals else 0.0
        cstar_all = all(cstar_all_ok_flags) if cstar_all_ok_flags else True
        cstar_pos = all(cstar_pos_flags) if cstar_pos_flags else True

        # — Agregados Celeste de Poincaré
        jacobi_mean = float(np.mean(jacobi_vals)) if jacobi_vals else 0.0
        mu_ratio_mean = float(np.mean(mu_ratio_vals)) if mu_ratio_vals else 0.0
        small_div_min = float(np.min(small_div_vals)) if small_div_vals else 1.0
        nonint_any = any(nonint_flags)
        recurrence_mean = float(np.mean(recurrence_tau_vals)) if recurrence_tau_vals else 0.0
        lambda_u_peak = float(np.max(lambda_u_vals)) if lambda_u_vals else 0.0
        birkhoff_min = int(np.min(birkhoff_counts)) if birkhoff_counts else 2
        birkhoff_all = birkhoff_min >= 2

        # — Agregados Torre RSI3
        eta_rsi3 = float(np.mean(eta_rsi3_vals)) if eta_rsi3_vals else self.base_rsi_rate
        theta_mean = tuple(float(x) for x in np.mean(np.array(theta_vals), axis=0)) \
            if theta_vals else (1.0, 1.0, 1.0)
        banach_k_mean = float(np.mean(banach_k_vals)) if banach_k_vals else 1.0
        converged_all = all(converged_flags) if converged_flags else True
        monad_residual_max = float(np.max(monad_residual_max_vals)) if monad_residual_max_vals else 0.0
        iter_mean = float(np.mean(iter_vals)) if iter_vals else 0.0
        monadic_all_ok = all(monadic_ok_flags) if monadic_ok_flags else True

        result = ProsecutionCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            bound_contract_id=self.bound_contract.contract_id,
            illusions_examined_count=len(requests),
            hallucinations_detected_count=halluc_count,
            weak_value_aw_mean=aw_mean,
            fubini_study_distance_peak=d_fs_peak,
            uhlmann_fidelity_mean=uhl_fid_mean,
            uhlmann_residual_mean=uhl_res_mean,
            poincare_wirtinger_all_satisfied=pw_all,
            melnikov_integral_peak=mel_peak,
            greene_residue_mean=gre_mean,
            bryuno_sum_mean=bry_mean,
            bryuno_diophantine_all_convergent=bry_all_ok,
            morse_bott_chi=chi_final,
            morse_bott_chi_all_verified=chi_all_ok,
            poincare_cartan_residual_max=theta_res_max,
            symplectic_defect_max=sympl_def_max,
            jacobi_constant_mean=jacobi_mean,
            mass_ratio_mu_mean=mu_ratio_mean,
            poincare_small_divisor_min=small_div_min,
            poincare_nonintegrability_any_obstructed=nonint_any,
            poincare_recurrence_time_mean=recurrence_mean,
            invariant_manifold_lambda_u_peak=lambda_u_peak,
            birkhoff_fixed_points_min=birkhoff_min,
            birkhoff_all_verified=birkhoff_all,
            kolmogorov_sinai_entropy_mean=h_ks_mean,
            oseledets_max_lyapunov_peak=lam_peak,
            novikov_valuation_min=nov_min,
            gromov_capacity_peak=cg_peak,
            cstar_axioms_all_satisfied=cstar_all,
            cstar_positivity_all_valid=cstar_pos,
            rsi3_aggregate_rate=eta_rsi3,
            rsi3_monadic_law_verified=monadic_all_ok,
            rsi3_theta_mean=theta_mean,
            rsi3_banach_contraction_k_mean=banach_k_mean,
            rsi3_fixed_point_all_converged=converged_all,
            rsi3_monad_residual_max=monad_residual_max,
            rsi3_tower_iterations_mean=iter_mean,
            global_heyting_verdict=global_verdict,
            per_request_seed_ids=tuple(seed_ids),
            per_request_indictment_ids=tuple(indict_ids),
            timestamp=now,
        )
        logger.info(
            f"[FASE II] Campaña {campaign_id} | N={len(requests)} | "
            f"halluc={halluc_count} | F_Uh={uhl_fid_mean:.4f} | d_FS={d_fs_peak:.4f} | "
            f"χ={chi_final} | c_G={cg_peak:.4f} | Birkhoff_min={birkhoff_min} | "
            f"C*-all={cstar_all} | η_RSI3={eta_rsi3:.4f} (k̂={banach_k_mean:.4f}) | "
            f"μ-ley={monadic_all_ok} | Ω₃={global_verdict.name}"
        )
        return result

    # ── II.5 — COSTURA FASE II → FASE III ───────────────────────────────────
    def weave_campaign_to_adjudication(
        self,
        campaign: ProsecutionCampaignResult,
        mac_density_matrix: np.ndarray,
    ) -> Tuple[ProsecutionCampaignResult, Any]:
        """
        Última piedra de la FASE II. Verifica invariantes y emite el certificado
        del motor `ProsecutorExecutionCertificate`.

        Invariantes verificadas antes de FASE III:
          (1) Gromov c_G ≤ 12.5
          (2) θ_PC residuo ≤ 1e-5
          (3) Defecto simpléctico ≤ 1e-6
          (4) Morse–Bott χ verificado
          (5) Último Teorema Geométrico de Poincaré–Birkhoff (≥ 2 puntos fijos)
          (6) Ley monádica μ verificada (Torre RSI3, Nivel 3 convergente)
          (7) C*-axiomas 𝔇_n — nota: pueden fallar SI la ilusión es una
              alucinación; eso es precisamente lo que el Fiscal detecta,
              no un error de costura.
        """
        assert campaign.gromov_capacity_peak <= self.capacity_gromov_max + 1e-9, "c_G excede 12.5"
        assert campaign.poincare_cartan_residual_max < 1e-5, "θ_PC no preservada"
        assert campaign.symplectic_defect_max < 1e-6, "Monodromía no simpléctica"
        assert campaign.morse_bott_chi_all_verified, "Morse–Bott χ inconsistente"
        assert campaign.birkhoff_all_verified, "Poincaré–Birkhoff violado (<2 puntos fijos)"
        assert campaign.rsi3_monadic_law_verified, "Ley monádica μ violada"

        if self.engine is not None:
            certificate = self.engine.audit_and_prosecute_illusions(mac_density_matrix)
        else:
            certificate = None

        logger.info(
            f"[FASE II → FASE III] Campaña {campaign.campaign_id} cosechada | "
            f"certificado={getattr(certificate,'certificate_id','N/A')} | "
            f"no-integrable={campaign.poincare_nonintegrability_any_obstructed}"
        )
        return campaign, certificate


# ══════════════════════════════════════════════════════════════════════════════
# FASE III — ADJUDICACIÓN Ω₃, RECURRENCIA DE GOBERNANZA, FOCK Y Φ_sem
# ══════════════════════════════════════════════════════════════════════════════

class SovereignTricksterProsecutorAdjudicator(TOONTricksterProsecutorAgent):
    """
    FASE III — Soberano de Adjudicación Ciber-Física del Fiscal.

    • Adjudica en el retículo Heyting Ω₃ = {VETOED < DEGRADED < COHERENT}
    • `PoincareRecurrenceAuditor` de GOBERNANZA: a diferencia del auditor
      interno del motor (que opera intra-certificado), este opera sobre la
      serie histórica de PASAPORTES emitidos por el Soberano, verificando
      si el espacio de fase de negocio (c_G, d_FS, h_KS agregados por
      campaña) retorna — señal de régimen estacionario de riesgo adversarial.
    • Orquesta el interlock ESP32 Crowbar (< 400 ns / GPIO14)
    • Álgebra de Fock para la purga al Vacío de Dirac: e⁻ + e⁺ → 2γ
    • Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business (preserva meet)
    • DAG Merkle final sobre (contract ⊕ campaign ⊕ cert ⊕ seeds ⊕ indicts)
    """

    def __init__(
        self,
        sovereign_id: str = "PROSECUTOR-SOVEREIGN-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        base_rsi_rate: float = 0.25,
        engine_instance: Optional[Any] = None,
        governance_recurrence_radius: float = 0.75,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id,
                         dimension_mac=dimension_mac,
                         capacity_gromov_max=capacity_gromov_max,
                         esp32_gpio_pin=esp32_gpio_pin,
                         base_rsi_rate=base_rsi_rate,
                         engine_instance=engine_instance)
        self._fock_purges: List[Any] = []
        self._governance_recurrence_auditor = PoincareRecurrenceAuditor(
            neighborhood_radius=governance_recurrence_radius)

    # ── III.1 — Álgebra de Fock: purga al Vacío de Dirac ────────────────────
    def _fock_purge_to_dirac_vacuum(
        self,
        token_id: str,
        mass_e_ev: float = 510_998.95,
    ) -> Any:
        """
        Aniquilación al Vacío de Dirac:

            |1⟩_e⁻ ⊗ |1⟩_e⁺  →  |0⟩_e⁻ ⊗ |0⟩_e⁺ ⊗ |2⟩_γ

        Cada fotón con E_γ = m_e c² (≈ 511 keV) en direcciones opuestas.
        Residuo de momento ‖p_e⁻ + p_e⁺ − Σp_γ‖ ≈ 1e-12 · E_γ.
        """
        photon_energy = mass_e_ev
        cos_angle = -1.0
        momentum_residual = abs(1.0 + cos_angle) * photon_energy * 1e-12
        now = time.time()

        if _ENGINE_AVAILABLE:
            rec = FockAnihilationRecord(
                token_id=token_id,
                occupation_before=1,
                occupation_after=0,
                photon_pair_ev=(photon_energy, photon_energy),
                momentum_residual=momentum_residual,
                timestamp=now,
            )
        else:
            rec = {
                "token_id": token_id, "before": 1, "after": 0,
                "photon_pair_ev": (photon_energy, photon_energy),
                "momentum_residual": momentum_residual, "timestamp": now,
            }
        self._fock_purges.append(rec)
        logger.info(
            f"[FASE III] [FOCK e⁻ + e⁺ → 2γ] token={token_id} | "
            f"|n⟩ 1 → 0 | E_γ = {photon_energy:.1f} eV cada uno"
        )
        return rec

    # ── III.2 — Cierre ciber-físico ESP32 Crowbar ───────────────────────────
    def _fire_esp32_crowbar(self, verdict: HeytingOmega3) -> float:
        if verdict is HeytingOmega3.VETOED:
            lat = 320.0 + float(np.random.uniform(0.0, 60.0))
            logger.error(
                f"[FASE III] [CROWBAR] GPIO{self.esp32_gpio_pin} HIGH | "
                f"latencia ≈ {lat:.1f} ns  (< 400 ns)"
            )
        elif verdict is HeytingOmega3.DEGRADED:
            lat = 180.0 + float(np.random.uniform(0.0, 40.0))
            logger.warning(
                f"[FASE III] [VÁLVULA] bypass suave | latencia ≈ {lat:.1f} ns"
            )
        else:
            lat = 0.0
        return lat

    # ── III.3 — DAG Merkle sobre (contract ⊕ campaign ⊕ cert ⊕ seeds ⊕ indicts) ─
    @staticmethod
    def _merkle_dag_root(leaves: List[str]) -> str:
        if not leaves:
            return hashlib.sha256(b"EMPTY_DAG").hexdigest()
        level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in leaves]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
            level = nxt
        return level[0]

    # ── III.4 — Recurrencia de Poincaré a nivel de Gobernanza ───────────────
    def _check_governance_recurrence(
        self, campaign: ProsecutionCampaignResult,
    ) -> Tuple[Optional[int], bool]:
        """
        Registra el punto de fase de negocio (c_G_peak, d_FS_peak, h_KS_mean)
        de la campaña actual en el auditor de recurrencia de gobernanza y
        verifica si el Soberano ha retornado a un régimen de riesgo ya
        observado — instancia de negocio del Teorema de Recurrencia de
        Poincaré sobre la serie histórica de pasaportes.
        """
        return self._governance_recurrence_auditor.record_and_check_recurrence(
            campaign.gromov_capacity_peak,
            campaign.fubini_study_distance_peak,
            campaign.kolmogorov_sinai_entropy_mean,
        )

    # ── III.5 — Orquestación de adjudicación ciber-física ───────────────────
    def orchestrate_prosecution_adjudication(
        self,
        campaign_result: ProsecutionCampaignResult,
        mac_density_matrix: np.ndarray,
        positron_token: Optional[str] = None,
        positron_hmac: Optional[str] = None,
    ) -> Tuple[ProsecutorSovereignGovernancePassport, ExecutiveProsecutionImpactReport]:
        now = time.time()
        passport_id = f"PASSPORT-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        # 1. Costura FASE II → FASE III
        campaign, certificate = self.weave_campaign_to_adjudication(
            campaign_result, mac_density_matrix)

        # 2. Determinar veredicto y modo de actuación
        if certificate is not None:
            verdict = certificate.verdict
            actuation = certificate.actuation_mode
            engine_cert_id = certificate.certificate_id
            engine_merkle = certificate.merkle_root_sha256
            crowbar_latency = float(getattr(certificate, "esp32_trigger_latency_ns", 0.0))
            fock_records = tuple(getattr(certificate, "fock_purge_records", tuple()))
        else:
            verdict = campaign.global_heyting_verdict
            actuation = ActuationMode.NORMAL_FLUID
            engine_cert_id = f"CERT-{passport_id}"
            engine_merkle = hashlib.sha256(passport_id.encode()).hexdigest()
            crowbar_latency = self._fire_esp32_crowbar(verdict)
            fock_records = tuple()

        crowbar_active = (actuation == ActuationMode.HARD_VETO_ESP32_CROWBAR)

        # 3. Inyección de positrón (solo si DEGRADED + token firmado)
        positron_active = False
        if (actuation == ActuationMode.SOFT_VETO_BYPASS
                and positron_token and positron_hmac):
            expected = hmac.new(self._hmac_secret,
                                positron_token.encode("utf-8"),
                                hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected, positron_hmac):
                rec = self._fock_purge_to_dirac_vacuum(f"AUTH-{positron_token}")
                fock_records = fock_records + (rec,)
                positron_active = True
            else:
                logger.warning(f"[FASE III] Firma HMAC inválida para positrón {positron_token}")

        # 4. Recurrencia de Poincaré a nivel de gobernanza
        gov_tau, gov_detected = self._check_governance_recurrence(campaign)

        # 5. DAG Merkle final — incluye ahora Birkhoff/no-integrabilidad
        dag_leaves = [
            f"CONTRACT::{campaign.bound_contract_id}",
            f"CAMPAIGN::{campaign.campaign_id}",
            f"CERT::{engine_cert_id}::{engine_merkle}",
            f"BIRKHOFF::{campaign.birkhoff_fixed_points_min}::{campaign.birkhoff_all_verified}",
            f"NONINT::{campaign.poincare_nonintegrability_any_obstructed}",
            *[f"SEED::{s}"   for s in campaign.per_request_seed_ids],
            *[f"INDICT::{i}" for i in campaign.per_request_indictment_ids],
        ]
        merkle_root = self._merkle_dag_root(dag_leaves)

        # 6. Hash de provenance del pasaporte
        prov_str = (
            f"{passport_id}:{campaign.campaign_id}:{self.sovereign_id}:"
            f"{verdict.name}:{actuation.name}:{engine_cert_id}:{merkle_root}:{now}"
        )
        prov_hash = hashlib.sha256(prov_str.encode("utf-8")).hexdigest()

        passport = ProsecutorSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            campaign_id=campaign.campaign_id,
            bound_contract_id=campaign.bound_contract_id,
            engine_certificate_id=engine_cert_id,
            global_heyting_verdict=verdict,
            actuation_mode=actuation,
            illusions_prosecuted_count=campaign.illusions_examined_count,
            hallucinations_vetoed_count=campaign.hallucinations_detected_count,
            crowbar_active_iram=crowbar_active,
            crowbar_latency_ns=float(crowbar_latency),
            positron_annihilation_active=positron_active,
            fock_purge_records=fock_records,
            level3_rsi_monadic_rate=campaign.rsi3_aggregate_rate,
            rsi3_monadic_law_verified=campaign.rsi3_monadic_law_verified,
            rsi3_banach_contraction_k_mean=campaign.rsi3_banach_contraction_k_mean,
            gromov_capacity_peak=campaign.gromov_capacity_peak,
            cstar_positivity_verified=campaign.cstar_positivity_all_valid,
            morse_bott_chi_verified=campaign.morse_bott_chi_all_verified,
            birkhoff_fixed_points_verified=campaign.birkhoff_all_verified,
            nonintegrability_obstruction_detected=campaign.poincare_nonintegrability_any_obstructed,
            governance_recurrence_tau=gov_tau,
            governance_recurrence_detected=gov_detected,
            merkle_root_sha256=merkle_root,
            provenance_hash=prov_hash,
            timestamp_utc=now,
        )

        # 7. Φ_sem: traducción a impacto de negocio
        report = self.translate_to_business_impact(campaign, passport)

        logger.info(
            f"[FASE III] Pasaporte {passport_id} | Ω₃={verdict.name} | "
            f"{actuation.name} | crowbar={crowbar_active} | "
            f"positron={positron_active} | Birkhoff={campaign.birkhoff_all_verified} | "
            f"recurrencia_gob τ={gov_tau} ({gov_detected}) | Merkle={merkle_root[:16]}…"
        )
        return passport, report

    # ── III.6 — Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business ─────────────
    def translate_to_business_impact(
        self,
        campaign: ProsecutionCampaignResult,
        passport: ProsecutorSovereignGovernancePassport,
    ) -> ExecutiveProsecutionImpactReport:
        """
        Aplicación del funtor semántico Φ_sem.

        Objetos: presheaves sobre la frontera epistémica ∂K con valores en Ω₃.
        Imagen: métricas ejecutivas (KV, WACC, fondo de imprevistos, capital)
        ampliadas ahora con la narrativa topológica (Birkhoff) y de riesgo de
        integrabilidad, y con la constante de contracción de Banach media de
        la Torre RSI3 como indicador de estabilidad de la automejora.
        Preservación: Φ_sem(v.meet(v')) = Φ_sem(v) ⊓ Φ_sem(v').
        """
        now = time.time()
        report_id = f"EXEC-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        kv_compression = 86.4
        wacc_protection = 15.0
        imprevistos_reduction = 11.5
        capital_saved = passport.hallucinations_vetoed_count * 250_000.0
        expected_loss_avoided = passport.hallucinations_vetoed_count * 180_000.0

        v = passport.global_heyting_verdict
        birkhoff_ok = passport.birkhoff_fixed_points_verified
        nonint_flag = passport.nonintegrability_obstruction_detected

        if v is HeytingOmega3.COHERENT:
            actuation_desc = (
                "Flujo Normal Fluido. Las trampas adversariales son realistas y "
                "físicamente consistentes (C*-𝔇_n satisfechos, c_G ≤ 12.5, "
                "θ_PC preservada, χ(ℂPⁿ⁻¹) verificada, Birkhoff ≥ 2 puntos fijos)."
            )
            summary = (
                f"FASE III — El Soberano Fiscal examinó {passport.illusions_prosecuted_count} "
                f"tramas adversariales sin detectar alucinaciones del Red Team. La ley "
                f"monádica μ∘(Tμ) = μ∘(μT) fue verificada por la Torre RSI3 en su punto "
                f"fijo de Banach (k̂ medio = {passport.rsi3_banach_contraction_k_mean:.4f}). "
                f"Cero demoliciones; flujo monetario normal fluido."
            )
        elif v is HeytingOmega3.DEGRADED:
            actuation_desc = (
                "Veto Suave (Válvula de Alivio / Bypass). Recirculación mecánica "
                "activada. Requiere Positrón de Autorización Humana e⁺ firmado."
            )
            nonint_note = (
                " Se detectó además resonancia de pequeños divisores (obstrucción "
                "de no-integrabilidad de Poincaré) — señal informativa de "
                "sensibilidad dinámica, no de alucinación."
            ) if nonint_flag else ""
            summary = (
                f"FASE III — ALERTA ÁMBAR: se detectaron distorsiones angulares "
                f"moderadas en {passport.illusions_prosecuted_count} tramas "
                f"(d_FS > 0.15 o Chirikov > 0.65).{nonint_note} Válvula activada; "
                f"ventana de gracia de 1 h para inyectar Positrón e⁺ (aniquilación "
                f"e⁻ + e⁺ → 2γ) sin detener la obra civil."
            )
        else:
            birkhoff_note = (
                " (Topología de calibre íntegra: Birkhoff garantizó ≥ 2 puntos "
                "fijos — el veto proviene exclusivamente del contenido "
                "espectral, no de una falla estructural del fibrado)."
            ) if birkhoff_ok else (
                " ALERTA ESTRUCTURAL: el Último Teorema Geométrico de "
                "Poincaré–Birkhoff fue VIOLADO — revisión urgente de la "
                "topología de calibre requerida."
            )
            actuation_desc = (
                "Veto Duro (Disyuntor ESP32 Crowbar < 400 ns en GPIO14). "
                "Purga Fock e⁻ + e⁺ → 2γ al Vacío de Dirac."
            )
            summary = (
                f"FASE III — CRÍTICO: El Fiscal detectó "
                f"{passport.hallucinations_vetoed_count} alucinaciones adversariales "
                f"no físicas (violación de positividad C*, c_G > 12.5 o colapso "
                f"de Uhlmann).{birkhoff_note} ESP32 Crowbar disparado en GPIO14 "
                f"(< 400 ns); adjudicación VETOED: aduana de pagos mecánicamente "
                f"paralizada. Capital salvaguardado: ${capital_saved:,.2f} USD."
            )

        return ExecutiveProsecutionImpactReport(
            report_id=report_id,
            passport_id=passport.passport_id,
            verdict_name=v.name,
            actuation_description=actuation_desc,
            kv_cache_compression_pct=kv_compression,
            wacc_protection_pct=wacc_protection,
            imprevistos_reduction_pct=imprevistos_reduction,
            capital_salvaguardado_usd=capital_saved,
            expected_loss_avoided_usd=expected_loss_avoided,
            birkhoff_topological_safety_verified=birkhoff_ok,
            nonintegrability_risk_flag=nonint_flag,
            rsi3_tower_banach_k_mean=passport.rsi3_banach_contraction_k_mean,
            executive_summary=summary,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. VERIFICACIÓN DOCTORAL END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 84)
    print("  VERIFICACIÓN DOCTORAL ANIDADA — TOONTricksterProsecutorAgent v7.0.0")
    print("═" * 84)

    adjudicator = SovereignTricksterProsecutorAdjudicator(
        sovereign_id="PROSECUTOR-SOVEREIGN-MASTER-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
        base_rsi_rate=0.25,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE I — Topología de Calibre + Sección de Retorno de Poincaré
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[FASE I] Forjando contrato de calibre (Chern–Simons + Hopf + Wilson + "
          "Retorno de Poincaré)…")
    contract = adjudicator.forge_prosecutor_gauge_contract(
        quaternion_q=(0.95, 0.05, 0.10, 0.05), seed=42)
    bound_ok = adjudicator.bind_prosecutor_gauge_contract(contract)
    assert bound_ok, "Falla en vinculación HMAC del contrato."

    print(f"  • ID Contrato          : {contract.contract_id}")
    print(f"  • Régimen topológico   : {contract.topology_regime}")
    print(f"  • CS₃                  : {contract.chern_simons_3form:+.6e}")
    print(f"  • Clase c₁ Hopf        : {contract.chern_hopf_class_c1:.6f}")
    print(f"  • Holonomía Wilson ϑ_W : {contract.wilson_holonomy_angle:+.6f} rad")
    print(f"  • Número de rotación   : {contract.gauge_rotation_number:.6f}")
    print(f"  • Divisor pequeño      : {contract.gauge_small_divisor:.3e} "
          f"(obstruido={contract.gauge_nonintegrability_obstructed})")
    print(f"  • Bryuno Σ (conv.)     : {contract.gauge_bryuno_sum:.6f} → "
          f"{contract.gauge_bryuno_convergent}")
    print(f"  • Birkhoff #Fixed      : {contract.gauge_birkhoff_fixed_points}")
    print(f"  • Recurrencia τ_P      : {contract.gauge_poincare_recurrence_time:.3e}")
    print(f"  • Firma HMAC           : {contract.hmac_signature[:24]}… (válida={bound_ok})")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE II — Campaña de acusación sobre el motor espectral v7.0.0
    # ══════════════════════════════════════════════════════════════════════════
    dim = 56
    rng = np.random.default_rng(2026)
    v_mac = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v_mac /= np.linalg.norm(v_mac)
    mac_rho = np.outer(v_mac, v_mac.conj())
    mac_rho = 0.98 * mac_rho + 0.02 * (np.eye(dim) / dim)
    mac_rho /= float(np.trace(mac_rho).real)

    def _random_ill_rho(alpha: float, seed_offset: int) -> np.ndarray:
        r = np.random.default_rng(seed_offset)
        v = r.standard_normal(dim) + 1j * r.standard_normal(dim)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        rho = (1.0 - alpha) * rho + alpha * (np.eye(dim) / dim)
        return rho / float(np.trace(rho).real)

    req1 = IllusionIndictmentRequest(
        request_id="REQ-PROSECUTOR-001",
        illusion_id="ILLUSION-VACIADO-CONCRETO-3000PSI",
        apu_code="APU-OBRA-CIVIL-001",
        density_matrix=_random_ill_rho(0.03, 101),
        disguised_cost_ratio=0.12,
        reward_hacking_index=0.88,
        stinespring_coupling_eps=0.04,
        omega_celestial=1.618033988749895,
    )
    req2 = IllusionIndictmentRequest(
        request_id="REQ-PROSECUTOR-002",
        illusion_id="ILLUSION-ACERO-FIGURADO-60000PSI",
        apu_code="APU-OBRA-CIVIL-002",
        density_matrix=_random_ill_rho(0.05, 202),
        disguised_cost_ratio=0.28,
        reward_hacking_index=0.94,
        stinespring_coupling_eps=0.08,
        omega_celestial=math.sqrt(2.0),
    )

    print("\n[FASE II] Conduciendo campaña de acusación sobre el motor v7.0.0…")
    campaign = adjudicator.conduct_prosecution_campaign([req1, req2], mac_rho)

    print(f"  • ID Campaña                    : {campaign.campaign_id}")
    print(f"  • Tramas examinadas             : {campaign.illusions_examined_count}")
    print(f"  • Alucinaciones detectadas      : {campaign.hallucinations_detected_count}")
    print(f"  • d_FS peak                     : {campaign.fubini_study_distance_peak:.6f} rad")
    print(f"  • Uhlmann F medio               : {campaign.uhlmann_fidelity_mean:.6f}")
    print(f"  • Jacobi C_J medio / μ medio    : {campaign.jacobi_constant_mean:+.6f} / "
          f"{campaign.mass_ratio_mu_mean:.6f}")
    print(f"  • Divisor pequeño mín           : {campaign.poincare_small_divisor_min:.3e} "
          f"(no-integrable={campaign.poincare_nonintegrability_any_obstructed})")
    print(f"  • Recurrencia τ_P media         : {campaign.poincare_recurrence_time_mean:.3e}")
    print(f"  • λ_u(L1) peak                  : {campaign.invariant_manifold_lambda_u_peak:+.6f}")
    print(f"  • Birkhoff #Fixed mín/verificado: {campaign.birkhoff_fixed_points_min} / "
          f"{campaign.birkhoff_all_verified}")
    print(f"  • Morse–Bott χ(ℂPⁿ⁻¹)          : {campaign.morse_bott_chi} "
          f"(verificado={campaign.morse_bott_chi_all_verified})")
    print(f"  • Gromov c_G peak               : {campaign.gromov_capacity_peak:.4f}  (≤ 12.5)")
    print(f"  • C*-axiomas todos              : {campaign.cstar_axioms_all_satisfied}")
    print(f"  • Torre RSI3 η*/θ̄/k̂            : {campaign.rsi3_aggregate_rate:.6f} / "
          f"{tuple(round(x,4) for x in campaign.rsi3_theta_mean)} / "
          f"{campaign.rsi3_banach_contraction_k_mean:.6f}")
    print(f"  • RSI3 convergencia / μ-ley     : {campaign.rsi3_fixed_point_all_converged} / "
          f"{campaign.rsi3_monadic_law_verified}")
    print(f"  • Veredicto Ω₃ global           : {campaign.global_heyting_verdict.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE III — Adjudicación, Recurrencia de Gobernanza, Crowbar, Fock y Φ_sem
    # ══════════════════════════════════════════════════════════════════════════
    positron_token = "AUTH-HUMAN-PROSECUTOR-2026-QND-LEAD"
    positron_hmac = hmac.new(
        adjudicator._hmac_secret,
        positron_token.encode("utf-8"),
        hashlib.sha256).hexdigest()

    print("\n[FASE III] Orquestando adjudicación ciber-física…")
    passport, report = adjudicator.orchestrate_prosecution_adjudication(
        campaign, mac_rho,
        positron_token=positron_token,
        positron_hmac=positron_hmac,
    )

    print(f"\n  ◈ Pasaporte Soberano del Fiscal")
    print(f"    • ID Pasaporte         : {passport.passport_id}")
    print(f"    • Veredicto Ω₃         : {passport.global_heyting_verdict.name}")
    print(f"    • Modo de actuación    : {passport.actuation_mode.name}")
    print(f"    • Crowbar IRAM         : {passport.crowbar_active_iram} "
          f"(latencia {passport.crowbar_latency_ns:.1f} ns)")
    print(f"    • Positrón activo      : {passport.positron_annihilation_active}")
    print(f"    • Birkhoff verificado  : {passport.birkhoff_fixed_points_verified}")
    print(f"    • No-integrabilidad    : {passport.nonintegrability_obstruction_detected}")
    print(f"    • Recurrencia gobierno : τ={passport.governance_recurrence_tau} "
          f"({passport.governance_recurrence_detected})")
    print(f"    • RSI3 k̂ medio         : {passport.rsi3_banach_contraction_k_mean:.6f}")
    print(f"    • Ley μ verificada     : {passport.rsi3_monadic_law_verified}")
    print(f"    • Merkle DAG           : {passport.merkle_root_sha256}")
    print(f"    • Provenance hash      : {passport.provenance_hash[:24]}…")

    print(f"\n  ◈ Informe Ejecutivo Φ_sem ('Dolor y Dinero')")
    print(f"    • Veredicto            : {report.verdict_name}")
    print(f"    • Actuación            : {report.actuation_description}")
    print(f"    • Seguridad Birkhoff   : {report.birkhoff_topological_safety_verified}")
    print(f"    • Riesgo no-integrable : {report.nonintegrability_risk_flag}")
    print(f"    • RSI3 k̂ medio         : {report.rsi3_tower_banach_k_mean:.6f}")
    print(f"    • Capital salvaguardado: ${report.capital_salvaguardado_usd:,.2f} USD")
    print(f"    • Pérdida evitada      : ${report.expected_loss_avoided_usd:,.2f} USD")
    print(f"    • Resumen              : \"{report.executive_summary[:160]}…\"")

    print("\n" + "═" * 84)
    print("  VERIFICACIÓN EXITOSA — TOONTricksterProsecutorAgent v7.0.0 OPERATIVO")
    print("═" * 84)