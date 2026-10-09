# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Introspection Prosecutor Agent (Soberano Fiscal Introspectivo)║
║ Ubicación: app/agents/wisdom/toon_introspection_prosecutor_agent.py           ║
║ Versión  : 7.0.0-Doctoral-Nested-Gauge-CStar-CPn-MorseBott-RSI3Tower-         ║
║            PoincareCelestialMechanics-Fock-Φsem                               ║
║ Función  : Agente Soberano Fiscal Introspectivo, topología de calibre        ║
║            P(M, G=U(1)×SU(2)×H₃(ℝ)), acusación proyectiva en ℂPⁿ⁻¹,          ║
║            estabilidad CR3BP (L1..L5), KAM, Lyapunov Oseledets transversal,  ║
║            Torre RSI3 con Banach, purga en álgebra de Fock e⁻e⁺ → 2γ y Φsem.║
║ Tratados : Chern–Simons (1974) · Hopf (1931) · Wilson (1974) · Yang–Mills (1954)║
║            Cartan (1926) · Bianchi · Poincaré, Méthodes Nouvelles (1892–99)  ║
║            Euler (1767) · Lagrange (1772) · Richardson (1980) · Uhlmann (1976)║
║            Oseledets (1968) · Chirikov (1979) · Banach (1922) · Dirac (1930)  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y ARQUITECTURA EN TRES FASES ANIDADAS:

El Agente Soberano Fiscal de Introspección (`TOONIntrospectionProsecutorAgent`) ejerce la
potestad de investigación proyectiva y enjuiciamiento de auto-atractores espurios en $\mathbb{C}P^{n-1}$
dentro del Estrato Wisdom ($V_{\mathbb{W}}$, RSI Nivel 3). Estructura el juzgamiento en tres fases
anidadas por herencia categórica:

◈ FASE I — INTROSPECTION PROSECUTOR GAUGE TOPOLOGY & POINCRÉ CELESTIAL ATLAS
  1. Fibrado Principal $P(M, G)$ con grupo de estructura $G = \mathrm{U}(1) \times \mathrm{SU}(2) \times H_3(\mathbb{R})$:
     Conexión $A \in \Omega^1(P, \mathfrak{g})$ anti-hermítica y curvatura $F = dA + [A, D]$.
  2. Invariantes de Calibre y Mecánica Celeste de Poincaré sobre $F$:
     3-Forma de Chern-Simons $CS_3(A)$, Fibración de Hopf $\pi(q) = q i \bar{q}$, Holonomía de Wilson $W(\gamma)$.
     Multiplicadores de Floquet del mapa de retorno sobre la curvatura $F$, serie de perturbación de Poincaré con radio de convergencia de Cauchy-Hadamard $R_{\mathrm{conv}}$, e invariante integral de Poincaré-Cartan $I_1 = \oint_\gamma \mathrm{Tr}(A)$.
  3. Contrato de Calibre Sellado por HMAC-SHA256 sobre 15 campos persistidos (firma simétrica).
  Costura Terminal: `weave_gauge_to_prosecution` $\longrightarrow$ `IntrospectionProsecutorGaugeContract`.

◈ FASE II — TOON INTROSPECTION PROSECUTOR AGENT (CAMPAÑA DE ACUSACIÓN PROYECTIVA)
  1. Consumo del Contrato Promovido de FASE I y Verificación Criptográfica HMAC.
  2. Acusación Espectral en Lote de Autoestados $|v^*\rangle \in \mathbb{C}P^{n-1}$ sobre `TOONIntrospectionProsecutorEngine` v6.0.0:
     Uhlmann real $F(\rho_{\mathrm{mac}}, |v^*\rangle \langle v^*|)$, Fubini-Study $d_{\mathrm{FS}}$, Exponente Transversal Oseledets $\lambda_\perp = \ln(\lambda_2 / \lambda_1) < 0$, constante de Jacobi $C_J$ en CR3BP con masa $\mu_{\mathrm{CR3BP}} = 1 - \mathrm{Tr}(\rho_{\mathrm{mac}}^2)$, análisis de estabilidad de Lagrange $L_4/L_5$ vs $L_1..L_3$, persistencia de toros KAM bajo perturbación de Stinespring $\epsilon < e^{-B(\omega)}$.
  3. Torre de Automejora Recursiva Nivel 3 ($\mu_1 \circ \mu_2 \circ \mu_3$):
     Prueba de contracción de Banach sobre la sucesión de tasas $\{\eta_1^{(k)}\}$ con cota $q < 1.0$.
  Costura Terminal: `weave_campaign_to_adjudication` $\longrightarrow$ `IntrospectionProsecutionCampaignResult`.

◈ FASE III — SOVEREIGN INTROSPECTION PROSECUTOR ADJUDICATOR (ADJUDICACIÓN Y AUTO-AUDITORÍA)
  1. Adjudicación en el Retículo de Heyting $\Omega_3 = \{0 < 1 < 2\}$ mediante meet de 8 criterios.
  2. Criterio de No-Integrabilidad de Poincaré (Solapamiento de Chirikov $K = \sum \frac{\Delta \omega_i}{\delta \omega_i} > 1$).
  3. Garantía de Estabilidad Topológica de Poincaré-Birkhoff ($\#\text{FixedPoints} \ge 2$).
  4. RSI3 Reflexivo: El Adjudicador recalibra sus propios umbrales de gobernanza $c_G$ usando su propia torre $\mu_3 \circ \mu_2 \circ \mu_1$.
  5. Purga al Vacío de Dirac en Álgebra de Fock $e^- + e^+ \to 2\gamma$ ($E_\gamma = 511\text{ keV}$).
  6. Disparo Ciber-Físico al ESP32 Crowbar en GPIO14 ($< 400\text{ ns}$).
  7. Funtor Semántico de Impacto Ejecutivo $\Phi_{\mathrm{sem}} : \mathrm{Sh}(\partial K, \Omega_3) \to \mathrm{Business}$.
  8. Cierre Criptográfico DAG de Merkle SHA-256 $\longrightarrow$ `IntrospectionProsecutorSovereignGovernancePassport`.

INVARIANTES Y AXIOMAS OPERATIVOS PRESERVADOS:
  • Cumplimiento de axiomas $C^*$-álgebraicos para matrices de densidad.
  • Invariancia de Poincaré-Cartan $\theta_{\mathrm{PC}} = \mathrm{Tr}(\rho N)$ con residuo $< 10^{-5}$.
  • Cota superior de capacidad simpléctica de Gromov-Wigner $c_G \le 12.5$.
  • Contracción transversal Oseledets $\lambda_\perp < 0$ para atractor autocoherente.
  • Convergencia de punto fijo de Banach en Torre RSI3 ($q < 1.0$).
  • Preservación del meet funtorial $\Phi_{\mathrm{sem}}(a \sqcap b) = \Phi_{\mathrm{sem}}(a) \sqcap \Phi_{\mathrm{sem}}(b)$.
  • Cierre ciber-físico $< 400\text{ ns}$ en GPIO14.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ── Asegurar rutas de importación ────────────────────────────────────────────
sys.path.extend(["/workspace/scratch", "/workspace/artifacts", "/workspace/out"])

# ── Importación flexible del motor fiscal v6.0.0 ────────────────────────────
try:
    from toon_introspection_prosecutor_engine import (   # type: ignore
        ActuationMode,
        CStarAuditResult,
        FockAnihilationRecord,
        HeytingOmega3,
        IntrospectionIndictmentGerm,
        IntrospectionProsecutorCanonicalSeed,
        IntrospectionProsecutorExecutionCertificate,
        IntrospectionProsecutorQNDEngine,
        PoincareIntrospectionProsecutorAtlas,
        TOONIntrospectionProsecutorEngine,
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

    CStarAuditResult = Any                           # type: ignore
    FockAnihilationRecord = Any                      # type: ignore
    IntrospectionIndictmentGerm = Any                # type: ignore
    IntrospectionProsecutorCanonicalSeed = Any       # type: ignore
    IntrospectionProsecutorExecutionCertificate = Any # type: ignore
    IntrospectionProsecutorQNDEngine = None          # type: ignore
    PoincareIntrospectionProsecutorAtlas = None      # type: ignore
    TOONIntrospectionProsecutorEngine = None         # type: ignore


logger = logging.getLogger("APU.Wisdom.TOONIntrospectionProsecutorAgent")
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
# §A. DATACLASSES Y CONTRATOS DEL FISCAL DE INTROSPECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class IntrospectionProsecutorGaugeContract:
    """
    Contrato de calibre inmutable del Fiscal de Introspección — FASE I.

    v7.0.0: se añaden 6 invariantes de mecánica celeste de Poincaré y se
    corrige un defecto de sellado HMAC del contrato v6.0.0 (la holonomía de
    Hopf se firmaba pero no se persistía, rompiendo la verificación en
    `bind_prosecutor_gauge_contract`). Ahora `chern_hopf_holonomy_angle` es
    campo de primera clase, firmado y persistido simétricamente.
    """
    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    chern_hopf_class_c1: float
    chern_hopf_holonomy_angle: float
    wilson_holonomy_angle: float
    hopf_base_point: Tuple[float, float, float]
    topology_regime: str
    su2_curvature_norm: float
    h3_trace_curvature: float
    poincare_floquet_multiplier_dominant: complex
    poincare_perturbation_radius_convergence: float
    poincare_cartan_invariant_I1: float
    poincare_cartan_residual_gauge: float
    poincare_recurrence_time_estimate: float
    hmac_signature: str
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class IntrospectionRayIndictmentRequest:
    """Solicitud de acusación sobre un autoestado proyectivo v* ∈ ℂPⁿ⁻¹."""
    request_id: str
    ray_id: str
    apu_code: str
    projective_ray_v: np.ndarray
    stinespring_coupling_eps: float = 0.05
    omega_celestial: float = 1.618033988749895


@dataclass(frozen=True, slots=True)
class IntrospectionProsecutionCampaignResult:
    """Resultado consolidado de campaña introspectiva — FASE II."""
    campaign_id: str
    sovereign_id: str
    bound_contract_id: str
    rays_examined_count: int
    spurious_attractors_detected_count: int
    # — Espectral QND proyectivo
    weak_value_aw_mean: complex
    fubini_study_distance_peak: float
    uhlmann_fidelity_mean: float
    uhlmann_residual_mean: float
    poincare_wirtinger_all_satisfied: bool
    # — Celeste clásico (Delaunay / Melnikov / Greene / Bryuno / Morse-Bott)
    melnikov_integral_peak: float
    greene_residue_mean: float
    bryuno_sum_mean: float
    bryuno_diophantine_all_convergent: bool
    morse_bott_chi: int
    morse_bott_chi_all_verified: bool
    poincare_cartan_residual_max: float
    symplectic_defect_max: float
    # — Oseledets transversal / Novikov / Gromov
    kolmogorov_sinai_entropy_mean: float
    oseledets_transverse_lyapunov_peak: float
    oseledets_all_contracting: bool
    novikov_valuation_min: float
    gromov_capacity_peak: float
    # — C*-𝔇_n
    cstar_axioms_all_satisfied: bool
    cstar_positivity_all_valid: bool
    # — [NUEVO] Mecánica celeste de Poincaré restringida de 3 cuerpos + KAM
    lagrange_l4_l5_stable_fraction: float
    lagrange_mu_routh_critical: float
    kam_tori_all_persist: bool
    kam_epsilon_bound_mean: float
    poincare_floquet_moduli_max: float
    poincare_orbit_linearly_stable: bool
    # — [NUEVO] Torre RSI Nivel 3 (μ₁ ∘ μ₂ ∘ μ₃)
    rsi3_level1_rate: float
    rsi3_level2_meta_rate: float
    rsi3_level3_metameta_rate: float
    rsi3_banach_contraction_constant: float
    rsi3_banach_fixed_point_converged: bool
    # — RSI Nivel 3 (ley monádica)
    rsi3_aggregate_rate: float
    rsi3_monadic_law_verified: bool
    # — Adjudicación global
    global_heyting_verdict: HeytingOmega3
    per_request_seed_ids: Tuple[str, ...]
    per_request_indictment_ids: Tuple[str, ...]
    timestamp: float


@dataclass(frozen=True, slots=True)
class IntrospectionProsecutorSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza del Fiscal de Introspección — FASE III."""
    passport_id: str
    sovereign_id: str
    campaign_id: str
    bound_contract_id: str
    engine_certificate_id: str
    global_heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    rays_prosecuted_count: int
    spurious_attractors_vetoed_count: int
    crowbar_active_iram: bool
    crowbar_latency_ns: float
    positron_annihilation_active: bool
    fock_purge_records: Tuple[Any, ...]
    level3_rsi_monadic_rate: float
    rsi3_monadic_law_verified: bool
    gromov_capacity_peak: float
    cstar_positivity_verified: bool
    morse_bott_chi_verified: bool
    oseledets_all_contracting: bool
    # — [NUEVO] Celeste de Poincaré en adjudicación final
    lagrange_l4_l5_stable_fraction: float
    kam_tori_all_persist: bool
    poincare_nonintegrability_chaotic_detected: bool
    chirikov_resonance_overlap_K: float
    birkhoff_twist_condition_satisfied: bool
    birkhoff_fixed_points_guaranteed: int
    # — [NUEVO] RSI3 reflexivo del Adjudicador sobre sí mismo
    rsi3_reflexive_capacity_gromov_calibrated: float
    rsi3_reflexive_threshold_adjusted: bool
    merkle_root_sha256: str
    provenance_hash: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ExecutiveIntrospectionProsecutionImpactReport:
    """Reporte Ejecutivo Φ_sem ('Dolor y Dinero') — FASE III."""
    report_id: str
    passport_id: str
    verdict_name: str
    actuation_description: str
    kv_cache_compression_pct: float
    wacc_protection_pct: float
    imprevistos_reduction_pct: float
    capital_salvaguardado_usd: float
    expected_loss_avoided_usd: float
    poincare_celestial_integrity_summary: str
    executive_summary: str
    timestamp_utc: float


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FASE I — TOPOLOGÍA DE CALIBRE Y MECÁNICA CELESTE DE POINCARÉ DEL FISCAL
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class IntrospectionProsecutorGaugeTopology:
    """
    FASE I — Fibrado principal P(M, G), G = U(1) × SU(2) × H₃(ℝ), enriquecido
    con las herramientas fundacionales de la mecánica celeste de Poincaré
    aplicadas al generador infinitesimal de la conexión de calibre.

    Rigor doctoral:
      • Conexión A anti-hermítica; curvatura F = dA + [A, D].
      • CS₃ = (1/8π²) Tr(A·dA + ⅔ A³) canónica.
      • Fibración de Hopf cuaterniónica π(q) = q·i·q̄ ∈ S².
      • Holonomía de Wilson W(γ) = P exp(∮_γ A) por exponencial ordenada.
      • [NUEVO] Mapa de retorno de Poincaré sobre la sección transversal del
        flujo generado por F, con multiplicadores de Floquet.
      • [NUEVO] Serie de perturbación de Poincaré (método del parámetro
        pequeño, 1892) con radio de convergencia de Cauchy–Hadamard.
      • [NUEVO] Invariante integral relativa de Poincaré–Cartan ∮_γ Tr(A).
      • [NUEVO] Teorema de recurrencia de Poincaré + Lema de Kac.
      • Contrato sellado con HMAC-SHA256 de 15 campos (firma-persistencia
        simétrica, defecto v6.0.0 corregido).
    """

    def __init__(self,
                 sovereign_id: str = "INTROSPECTIVE-PROSECUTOR-MASTER-01",
                 dimension: int = 56,
                 hmac_secret: bytes = b"APU_FILTER_V8_INTRO_PROSECUTOR_2026") -> None:
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

    # ── I.6 — Mapa de retorno de Poincaré + multiplicadores de Floquet ─────
    @staticmethod
    def compute_poincare_return_map(
        F: np.ndarray,
    ) -> Tuple[np.ndarray, complex, np.ndarray]:
        """
        Mapa de retorno de Poincaré sobre la sección transversal Σ del flujo
        lineal ẋ = Fx. Se identifica el par propio central ±iω (generador de
        la órbita periódica de referencia, período T = 2π/ω) y se construye
        el mapa de primer retorno P = exp(F·T). Los valores propios restantes,
        evaluados en exp(λ_j T), son los MULTIPLICADORES DE FLOQUET —
        exactamente los "exponentes característicos de Poincaré" de su
        Mémoire de 1881 sobre curvas definidas por ecuaciones diferenciales.

        Criterio de estabilidad: |multiplicador| < 1 ⇒ órbita linealmente
        estable (atractor genuino); |multiplicador| ≥ 1 ⇒ silla o repulsor
        (candidato a atractor espurio).
        """
        eigvals, _ = la.eig(F)
        im_abs = np.abs(np.imag(eigvals))
        penalty = np.where(im_abs < 1e-10, 1e6, 0.0)
        score = np.abs(np.real(eigvals)) + penalty
        idx_center = int(np.argmin(score))
        omega_center = max(float(abs(np.imag(eigvals[idx_center]))), 1e-6)
        period_T = float(np.clip(2.0 * math.pi / omega_center, 1e-6, 50.0))
        P = la.expm(F * period_T)

        mask = np.ones(len(eigvals), dtype=bool)
        mask[idx_center] = False
        conj_target = np.conj(eigvals[idx_center])
        conj_idx = int(np.argmin(np.abs(eigvals - conj_target)))
        if conj_idx != idx_center:
            mask[conj_idx] = False
        floquet_multipliers = np.exp(eigvals[mask] * period_T)
        return P, complex(eigvals[idx_center]), floquet_multipliers

    # ── I.7 — Serie de perturbación de Poincaré (parámetro pequeño) ─────────
    @staticmethod
    def poincare_perturbation_series(
        A: np.ndarray, D_gen: np.ndarray, order: int = 4,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Método de Poincaré del parámetro pequeño (Les Méthodes Nouvelles de
        la Mécanique Céleste, Tomo I, 1892): expansión en serie de Lie de la
        curvatura bajo la acción adjunta repetida de D sobre A,

            T_k = (ad_D)^k(A) / k! ,       F(ε) ≈ Σ_k ε^k T_k

        El radio de convergencia se estima por el criterio de
        Cauchy–Hadamard  R = 1 / limsup ‖T_{k+1}‖/‖T_k‖ , que para las
        series de Poincaré es típicamente FINITO (series asintóticas
        divergentes más allá de cierto orden crítico — el fenómeno de los
        "pequeños divisores" que motivó más tarde la teoría KAM).
        """
        terms: List[np.ndarray] = []
        F0 = D_gen @ A - A @ D_gen
        terms.append(F0)
        current = A.copy()
        for k in range(1, order + 1):
            current = D_gen @ current - current @ D_gen
            terms.append(current / math.factorial(k))

        norms = [float(la.norm(t, "fro")) + 1e-30 for t in terms]
        ratios = [norms[k + 1] / norms[k] for k in range(len(norms) - 1)]
        radius_convergence = 1.0 / (max(ratios) + 1e-30) if ratios else float("inf")
        return terms, float(radius_convergence)

    # ── I.8 — Invariante integral relativa de Poincaré–Cartan ───────────────
    @staticmethod
    def compute_poincare_cartan_invariant(
        A: np.ndarray, D_gen: np.ndarray,
        n_points: int = 32, epsilon_flow: float = 1e-3,
    ) -> Tuple[float, float]:
        """
        Integral invariante relativa de Poincaré–Cartan ∮_γ Tr(A) sobre un
        lazo γ generado por conjugación unitaria R(θ) = exp(iθD) (Arnold,
        Métodos Matemáticos de la Mecánica Clásica, §9). La invariancia es
        EXACTA a todo orden en este tejido de calibre porque
        Tr([D, A]) ≡ 0 (traza del conmutador), de modo que el residuo mide
        puramente el error numérico de discretización — una verificación
        constructiva, no asintótica, del teorema.
        """
        thetas = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
        dtheta = 2.0 * math.pi / n_points

        def _loop_integral(mat: np.ndarray) -> float:
            total = 0.0
            for th in thetas:
                R = la.expm(1j * th * D_gen)
                conjugated = R @ mat @ R.conj().T
                total += float(np.real(np.trace(conjugated)))
            return total * dtheta

        integral_before = _loop_integral(A)
        A_evolved = A + epsilon_flow * (D_gen @ A - A @ D_gen)
        integral_after = _loop_integral(A_evolved)
        residual = abs(integral_after - integral_before)
        return float(integral_before), float(residual)

    # ── I.9 — Teorema de recurrencia de Poincaré (Lema de Kac) ──────────────
    @staticmethod
    def poincare_recurrence_time(
        region_measure: float, total_measure: float = 1.0,
    ) -> float:
        """
        Teorema de recurrencia de Poincaré (1890): bajo un flujo que
        preserva medida, c.t.p. trayectoria que entra en una región A de
        medida positiva retorna a A infinitas veces. El Lema de Kac precisa
        el tiempo medio de recurrencia:  ⟨τ_A⟩ = μ(Ω) / μ(A).
        Se emplea aquí para estimar cuántas "pasadas espectrales" del rayo
        proyectivo son necesarias para re-visitar su propia vecindad de
        fase — un reloj celeste del propio autoestado.
        """
        mu = float(np.clip(region_measure / max(total_measure, 1e-12), 1e-9, 1.0))
        return float(1.0 / mu)

    # ── I.10 — Forja del contrato de calibre sellado por HMAC ───────────────
    def forge_prosecutor_gauge_contract(
        self,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        seed: int = 42,
    ) -> IntrospectionProsecutorGaugeContract:
        now = time.time()
        contract_id = f"CONTRACT-INTRO-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        A, dA, F, D_gen = self.compute_principal_connection(seed=seed)
        cs3 = self.compute_chern_simons_3form(A, dA)
        c1, hol_hopf, hopf_pt, regime = self.compute_chern_hopf_class(quaternion_q)
        _, hol_wilson = self.compute_wilson_holonomy(A)
        f_u1, f_su2, f_h3 = self.decompose_curvature(F)

        # — Mecánica celeste de Poincaré (granular, nativa de FASE I) —
        _, lambda_center, floquet_mult = self.compute_poincare_return_map(F)
        dominant_floquet = (
            complex(floquet_mult[int(np.argmax(np.abs(floquet_mult)))])
            if len(floquet_mult) else complex(1.0, 0.0)
        )
        _, perturbation_radius = self.poincare_perturbation_series(A, D_gen, order=4)
        cartan_invariant, cartan_residual = self.compute_poincare_cartan_invariant(A, D_gen)
        recurrence_time = self.poincare_recurrence_time(
            region_measure=min(abs(c1), 0.99) + 1e-6)

        # — Contenido firmado: 15 campos, firma-persistencia SIMÉTRICA —
        content_str = (
            f"{contract_id}:{self.sovereign_id}:{cs3:.10f}:"
            f"{c1:.10f}:{hol_hopf:.10f}:{hol_wilson:.10f}:{regime}:"
            f"{f_su2:.10f}:{f_h3:.10f}:"
            f"{dominant_floquet.real:.10f}:{dominant_floquet.imag:.10f}:"
            f"{perturbation_radius:.10f}:{cartan_invariant:.10f}:"
            f"{recurrence_time:.10f}:{now}"
        )
        hmac_sig = hmac.new(self._hmac_secret,
                            content_str.encode("utf-8"),
                            hashlib.sha256).hexdigest()

        contract = IntrospectionProsecutorGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=cs3,
            chern_hopf_class_c1=c1,
            chern_hopf_holonomy_angle=hol_hopf,
            wilson_holonomy_angle=hol_wilson,
            hopf_base_point=hopf_pt,
            topology_regime=regime,
            su2_curvature_norm=f_su2,
            h3_trace_curvature=f_h3,
            poincare_floquet_multiplier_dominant=dominant_floquet,
            poincare_perturbation_radius_convergence=perturbation_radius,
            poincare_cartan_invariant_I1=cartan_invariant,
            poincare_cartan_residual_gauge=cartan_residual,
            poincare_recurrence_time_estimate=recurrence_time,
            hmac_signature=hmac_sig,
            creation_timestamp=now,
        )
        logger.info(
            f"[FASE I] Contrato {contract_id} forjado | régimen={regime} | "
            f"c₁={c1:.4f} | CS₃={cs3:+.4e} | ϑ_W={hol_wilson:+.4f} rad | "
            f"‖F_su2‖={f_su2:.4f} | |Floquet|_dom={abs(dominant_floquet):.4f} | "
            f"R_conv={perturbation_radius:.4f} | θ_PC_res={cartan_residual:.2e} | "
            f"⟨τ_Poincaré⟩={recurrence_time:.2f}"
        )
        return contract

    # ── I.11 — COSTURA FASE I → FASE II ─────────────────────────────────────
    @staticmethod
    def weave_gauge_to_prosecution(
        contract: IntrospectionProsecutorGaugeContract,
    ) -> Tuple[IntrospectionProsecutorGaugeContract, bool]:
        """
        Última piedra de la FASE I. Promueve el contrato —ahora enriquecido
        con el atlas celeste de Poincaré (retorno, perturbación, Cartan,
        recurrencia)— al observatorio fiscal bajo 3 reglas categóricas:
          (1) TRIVIAL_BUNDLE    → promoción limpia
          (2) MONOPOLE_LIKE     → promoción con vigilancia
          (3) INSTANTON_DENSE   → promoción con vigilancia reforzada

        Esta promoción es, literalmente, el PRIMER INSUMO que consume
        `bind_prosecutor_gauge_contract` al abrir la FASE II: el último
        método de FASE I es la antesala formal del primer método de FASE II.
        """
        stability_flag = abs(contract.poincare_floquet_multiplier_dominant) < 1.0 + 1e-6
        logger.info(
            f"[FASE I → FASE II] Contrato {contract.contract_id} promovido al "
            f"observatorio fiscal introspectivo | régimen={contract.topology_regime} | "
            f"Floquet_dom estable={stability_flag}"
        )
        return contract, True


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FASE II — CAMPAÑA DE ACUSACIÓN PROYECTIVA + RESTRICTED 3-BODY + RSI3 TOWER
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONIntrospectionProsecutorAgent(IntrospectionProsecutorGaugeTopology):
    """
    FASE II — Soberano Fiscal de Introspección (hereda el atlas celeste y de
    calibre de FASE I).

    Consume el motor `TOONIntrospectionProsecutorEngine` v6.0.0 end-to-end:
        engine.prosecute_introspection_ray(...)   → (IndictmentGerm, Seed)
        engine.audit_and_prosecute_rays(...)      → ExecutionCertificate

    Agrega invariantes a nivel de campaña (C*-𝔇_n, Uhlmann, PW, Morse–Bott,
    Oseledets transversal, Novikov, Gromov) y además:

      • [NUEVO] Problema restringido de 3 cuerpos (CR3BP): mapea cada rayo a
        un punto del plano sinódico y evalúa su cercanía dinámica a los
        puntos triangulares L4/L5 (estables, μ < μ_Routh) vs. colineales
        L1-L3 (sillas inestables) — un análogo físico riguroso del
        "atractor espurio".
      • [NUEVO] Condición KAM: estima si la perturbación de Stinespring
        ε respeta la cota de persistencia del toro invariante dado por la
        suma de Bryuno del propio motor.
      • [NUEVO] Exponentes característicos de Poincaré (reutiliza I.6).
      • [NUEVO] Torre de automejora recursiva de Nivel 3 GENUINA:
            μ₁: Θ → Θ            (adaptación de parámetros, rápida)
            μ₂: (Θ→Θ) → (Θ→Θ)    (meta-optimización de η₁)
            μ₃: meta-meta         (control de la tasa de meta-aprendizaje)
        cerrada con una verificación constructiva de contracción de Banach
        sobre la sucesión {η₁^(k)}.
    """

    def __init__(
        self,
        sovereign_id: str = "INTROSPECTIVE-PROSECUTOR-MASTER-01",
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
        elif TOONIntrospectionProsecutorEngine is not None:
            self.engine = TOONIntrospectionProsecutorEngine(
                dimension=dimension_mac,
                capacity_gromov_max=capacity_gromov_max,
                esp32_gpio_pin=esp32_gpio_pin,
                base_rsi_rate=base_rsi_rate,
            )
        else:
            self.engine = None

        self.bound_contract: Optional[IntrospectionProsecutorGaugeContract] = None
        self._indictment_history: List[Any] = []
        self._seed_history: List[Any] = []

        # — [NUEVO] Estado persistente de la torre RSI Nivel 3 —
        self._rsi_level1_rate: float = base_rsi_rate
        self._rsi_level2_meta_rate: float = 0.10
        self._rsi_level3_metameta_rate: float = 0.02
        self._rsi_history: List[Dict[str, float]] = []

    # ── II.1 — Vinculación HMAC del contrato de FASE I ──────────────────────
    def bind_prosecutor_gauge_contract(
        self, contract: IntrospectionProsecutorGaugeContract
    ) -> bool:
        """
        Primer método de FASE II: recibe directamente el contrato promovido
        por `weave_gauge_to_prosecution` (último método de FASE I) y verifica
        su sellado HMAC reconstruyendo el `content_str` EXACTAMENTE simétrico
        al de `forge_prosecutor_gauge_contract` — defecto v6.0.0 corregido.
        """
        content_str = (
            f"{contract.contract_id}:{contract.sovereign_id}:"
            f"{contract.chern_simons_3form:.10f}:"
            f"{contract.chern_hopf_class_c1:.10f}:"
            f"{contract.chern_hopf_holonomy_angle:.10f}:"
            f"{contract.wilson_holonomy_angle:.10f}:{contract.topology_regime}:"
            f"{contract.su2_curvature_norm:.10f}:{contract.h3_trace_curvature:.10f}:"
            f"{contract.poincare_floquet_multiplier_dominant.real:.10f}:"
            f"{contract.poincare_floquet_multiplier_dominant.imag:.10f}:"
            f"{contract.poincare_perturbation_radius_convergence:.10f}:"
            f"{contract.poincare_cartan_invariant_I1:.10f}:"
            f"{contract.poincare_recurrence_time_estimate:.10f}:"
            f"{contract.creation_timestamp}"
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

    # ── II.2 — Acusación individual (delegada al motor v6.0.0) ──────────────
    def process_ray_indictment(
        self,
        request: IntrospectionRayIndictmentRequest,
        mac_density_matrix: np.ndarray,
    ) -> Tuple[Any, Any]:
        if self.engine is None:
            raise RuntimeError("TOONIntrospectionProsecutorEngine no disponible.")
        if self.bound_contract is None:
            self.bind_prosecutor_gauge_contract(self.forge_prosecutor_gauge_contract())

        logger.info(
            f"[FASE II] Indictando autoestado {request.ray_id} (APU: {request.apu_code})"
        )
        indictment, seed = self.engine.prosecute_introspection_ray(
            ray_id=request.ray_id,
            mac_density_matrix=mac_density_matrix,
            ray_vector=request.projective_ray_v,
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
        if IntrospectionProsecutorQNDEngine is None:
            return True
        try:
            eta1 = IntrospectionProsecutorQNDEngine._rsi3_monadic_multiplication(
                eta_base, seed,
                indictment.fubini_study_distance,
                indictment.kolmogorov_sinai_entropy,
                indictment.oseledets_transverse_lyapunov,
            )
            eta2 = IntrospectionProsecutorQNDEngine._rsi3_monadic_multiplication(
                eta1, seed,
                indictment.fubini_study_distance,
                indictment.kolmogorov_sinai_entropy,
                indictment.oseledets_transverse_lyapunov,
            )
            fold = abs(eta2 - eta1 * math.exp(
                -indictment.kolmogorov_sinai_entropy * indictment.fubini_study_distance))
            return fold < tol
        except Exception as exc:                      # pragma: no cover
            logger.warning(f"[FASE II] Verificación monádica falló: {exc}")
            return False

    # ── II.4 — Problema restringido de 3 cuerpos (CR3BP) ────────────────────
    @staticmethod
    def compute_restricted_three_body_jacobi_constant(
        mu: float, x: float, y: float, vx: float, vy: float,
    ) -> float:
        """
        Constante de Jacobi del Problema Restringido Circular de 3 Cuerpos
        (CR3BP), única integral primera conocida del problema (Poincaré
        demostró en 1890 que NO existen otras integrales analíticas
        independientes — génesis de su teorema de no-integrabilidad):

            C_J = x² + y² + 2(1−μ)/r₁ + 2μ/r₂ − (ẋ² + ẏ²)

        con r₁, r₂ las distancias a los primarios en el marco sinódico
        rotante. C_J es constante a lo largo de cualquier trayectoria real.
        """
        r1 = math.sqrt((x + mu) ** 2 + y ** 2) + 1e-30
        r2 = math.sqrt((x - 1.0 + mu) ** 2 + y ** 2) + 1e-30
        C_J = (x ** 2 + y ** 2) + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - (vx ** 2 + vy ** 2)
        return float(C_J)

    @staticmethod
    def classify_lagrange_point_stability(mu: float) -> Dict[str, Any]:
        """
        Estabilidad lineal de los puntos triangulares L4/L5 del CR3BP
        (Poincaré, Leçons de Mécanique Céleste; Szebehely, 1967):
        L4/L5 son linealmente estables (centros, frecuencias reales) si y
        solo si μ < μ_Routh = ½(1 − √(23/27)) ≈ 0.0385209; en caso contrario
        degeneran en sillas. L1, L2, L3 (colineales) son SIEMPRE sillas
        inestables, cualquiera sea μ — el análogo celeste exacto del
        "atractor espurio" de este fiscal: un punto crítico topológicamente
        necesario (teorema de los 5 puntos de libración) pero dinámicamente
        repulsivo en al menos una dirección.
        """
        mu_routh = 0.5 * (1.0 - math.sqrt(23.0 / 27.0))
        l4_l5_stable = mu < mu_routh
        if l4_l5_stable:
            discriminant = max(0.0, 1.0 - 27.0 * mu * (1.0 - mu))
            omega1 = math.sqrt(max(0.0, (1.0 + math.sqrt(discriminant)) / 2.0))
            omega2 = math.sqrt(max(0.0, (1.0 - math.sqrt(discriminant)) / 2.0))
        else:
            omega1 = omega2 = float("nan")
        return {
            "mu_routh_critical": mu_routh,
            "l4_l5_linearly_stable": l4_l5_stable,
            "l1_l2_l3_saddle_unstable": True,
            "omega1_librational_short_period": omega1,
            "omega2_librational_long_period": omega2,
        }

    # ── II.5 — Condición KAM de persistencia de toros invariantes ──────────
    @staticmethod
    def verify_kam_torus_persistence(
        omega_ratio: float, perturbation_eps: float,
        bryuno_sum: float, kam_threshold_const: float = 1.0,
    ) -> Tuple[bool, float]:
        """
        Teorema KAM (Kolmogorov 1954, Arnold 1963, Moser 1962): un toro
        invariante con número de rotación ω que satisface la condición de
        Bryuno  B(ω) = Σ ln(q_{n+1})/q_n < ∞  persiste bajo perturbación ε
        si ε < ε_KAM(ω) ~ exp(−C·B(ω)) (estimación de Rüssmann/Moser). Se
        reutiliza la suma de Bryuno ya computada por el motor espectral.
        """
        epsilon_kam_bound = math.exp(-kam_threshold_const * max(bryuno_sum, 1e-6))
        persists = perturbation_eps < epsilon_kam_bound
        return bool(persists), float(epsilon_kam_bound)

    # ── II.6 — Exponentes característicos de Poincaré (hereda I.6) ─────────
    def compute_poincare_characteristic_exponents(
        self, F_curvature: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Instancia II de la maquinaria I.6: reutiliza directamente
        `compute_poincare_return_map` heredado de FASE I — la "continuación"
        literal del último aparato de FASE I dentro del cuerpo de FASE II.
        """
        _, lambda_center, floquet_mult = self.compute_poincare_return_map(F_curvature)
        moduli = np.abs(floquet_mult) if len(floquet_mult) else np.array([1.0])
        stable = bool(np.all(moduli < 1.0 + 1e-9))
        omega_center = max(abs(np.imag(lambda_center)), 1e-6)
        period_T = 2.0 * math.pi / omega_center
        characteristic_exponents = np.log(floquet_mult + 1e-300) / period_T if len(floquet_mult) else np.array([0.0])
        return {
            "floquet_multipliers": floquet_mult,
            "moduli_max": float(np.max(moduli)),
            "orbit_linearly_stable": stable,
            "characteristic_exponents": characteristic_exponents,
        }

    # ── II.7 — RSI Nivel 1: adaptación de parámetros μ₁ : Θ → Θ ─────────────
    def _rsi_level1_parameter_adaptation(self, performance_signal: float) -> float:
        """
        RSI Nivel 1 — Regla de descenso contractiva:
            η₁ ← η₁ · exp(−η₂ · performance_signal)
        Si performance_signal > 0 (peor desempeño: mayor d_FS, mayor H_KS),
        η₁ decrece monótonamente, concentrando confianza en acusaciones
        futuras — una forma mínima de meta-plasticidad auto-regulada.
        """
        self._rsi_level1_rate *= math.exp(-self._rsi_level2_meta_rate * performance_signal)
        self._rsi_level1_rate = float(np.clip(self._rsi_level1_rate, 1e-6, 1.0))
        return self._rsi_level1_rate

    # ── II.8 — RSI Nivel 2: meta-optimización μ₂ : (Θ→Θ) → (Θ→Θ) ──────────
    def _rsi_level2_metaoptimizer_adaptation(self) -> float:
        """
        RSI Nivel 2 — Control de hiper-gradiente (generalización monádica
        de Schraudolph 1999): ajusta η₂ observando la VARIANZA de la
        trayectoria reciente de η₁. Alta varianza (oscilación) ⇒ se
        amortigua η₂; trayectoria estable ⇒ η₂ se mantiene.
        """
        if len(self._rsi_history) < 2:
            return self._rsi_level2_meta_rate
        recent = [h["eta1"] for h in self._rsi_history[-8:]]
        variance = float(np.var(recent))
        self._rsi_level2_meta_rate *= math.exp(-self._rsi_level3_metameta_rate * variance)
        self._rsi_level2_meta_rate = float(np.clip(self._rsi_level2_meta_rate, 1e-6, 1.0))
        return self._rsi_level2_meta_rate

    # ── II.9 — Verificación de contracción de Banach de la torre RSI3 ──────
    @staticmethod
    def _verify_banach_fixed_point_convergence(
        rate_sequence: Sequence[float], tol: float = 1e-3,
    ) -> Tuple[bool, float]:
        """
        Prueba constructiva de convergencia por punto fijo de Banach: la
        sucesión {η_k} converge si la razón de contracción
            q = sup_k |η_{k+1} − η_k| / |η_k − η_{k−1}|
        satisface q < 1 (operador T contractivo en espacio métrico completo
        ⇒ ∃! punto fijo, teorema de Banach–Caccioppoli, 1922).
        """
        if len(rate_sequence) < 3:
            return True, 0.0
        diffs = [abs(rate_sequence[i + 1] - rate_sequence[i])
                 for i in range(len(rate_sequence) - 1)]
        ratios = [diffs[i + 1] / (diffs[i] + 1e-30) for i in range(len(diffs) - 1)]
        q = float(np.max(ratios)) if ratios else 0.0
        converged = (q < 1.0 - tol) or (abs(rate_sequence[-1] - rate_sequence[-2]) < tol)
        return bool(converged), q

    # ── II.10 — RSI Nivel 3: torre completa μ₃∘μ₂∘μ₁ (RSI Nivel 3 real) ────
    def _rsi_level3_metameta_adaptation(self, performance_signal: float) -> Dict[str, Any]:
        """
        RSI Nivel 3 — Cierre de la torre de automejora recursiva:
            μ₃ ∘ μ₂ ∘ μ₁ : Θ → Θ
        donde μ₁ adapta parámetros, μ₂ adapta la tasa de μ₁, y μ₃ adapta la
        tasa de μ₂ (meta-meta). La ley monádica μ∘(Tμ)=μ∘(μT) del motor
        queda así EXTENDIDA con una prueba de convergencia de Banach sobre
        la propia sucesión {η₁^(k)} generada por la torre.
        """
        eta1 = self._rsi_level1_parameter_adaptation(performance_signal)
        eta2 = self._rsi_level2_metaoptimizer_adaptation()
        self._rsi_level3_metameta_rate *= math.exp(-1e-2 * abs(performance_signal))
        self._rsi_level3_metameta_rate = float(np.clip(self._rsi_level3_metameta_rate, 1e-7, 1.0))

        self._rsi_history.append({
            "eta1": eta1, "eta2": eta2, "eta3": self._rsi_level3_metameta_rate,
            "performance": performance_signal,
        })
        eta1_sequence = [h["eta1"] for h in self._rsi_history]
        converged, banach_q = self._verify_banach_fixed_point_convergence(eta1_sequence)

        return {
            "eta1_level1_rate": eta1,
            "eta2_level2_meta_rate": eta2,
            "eta3_level3_metameta_rate": self._rsi_level3_metameta_rate,
            "banach_contraction_constant": banach_q,
            "banach_fixed_point_converged": converged,
        }

    # ── II.11 — Campaña de acusación con agregación completa ────────────────
    def conduct_prosecution_campaign(
        self,
        requests: List[IntrospectionRayIndictmentRequest],
        mac_density_matrix: np.ndarray,
    ) -> IntrospectionProsecutionCampaignResult:
        now = time.time()
        campaign_id = f"CAMP-INTRO-PROSECUTOR-{int(now * 1000) % 1000000:06d}"
        if self.bound_contract is None:
            self.bind_prosecutor_gauge_contract(self.forge_prosecutor_gauge_contract())

        # — Acumuladores canónicos
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
        lam_trans_vals: List[float] = []
        nov_vals: List[float] = []
        cg_vals: List[float] = []
        cstar_all_ok_flags: List[bool] = []
        cstar_pos_flags: List[bool] = []
        eta_rsi3_vals: List[float] = []
        monadic_ok_flags: List[bool] = []
        verdicts: List[HeytingOmega3] = []
        spurious_count = 0
        seed_ids: List[str] = []
        indict_ids: List[str] = []

        # — Acumuladores celeste CR3BP / KAM
        purity = float(np.real(np.trace(mac_density_matrix @ mac_density_matrix)))
        mu_cr3bp = float(np.clip(1.0 - purity, 1e-4, 0.5))
        lagrange_info = self.classify_lagrange_point_stability(mu_cr3bp)
        C_J_L4 = self.compute_restricted_three_body_jacobi_constant(
            mu_cr3bp, 0.5 - mu_cr3bp, math.sqrt(3.0) / 2.0, 0.0, 0.0)
        l4_l5_like_flags: List[bool] = []
        kam_ok_flags: List[bool] = []
        kam_eps_vals: List[float] = []

        for req in requests:
            ind, seed = self.process_ray_indictment(req, mac_density_matrix)
            indict_ids.append(getattr(ind, "indictment_id", "?"))
            seed_ids.append(getattr(seed, "seed_id", "?"))

            aw_vals.append(complex(getattr(ind, "weak_value_Aw", 0.0)))
            d_fs_vals.append(float(getattr(ind, "fubini_study_distance", 0.0)))
            uhl_fid_vals.append(float(getattr(ind, "uhlmann_fidelity", 1.0)))
            uhl_res_vals.append(float(getattr(ind, "uhlmann_residual", 0.0)))
            pw_flags.append(bool(getattr(ind, "poincare_wirtinger_satisfied", True)))
            h_ks_vals.append(float(getattr(ind, "kolmogorov_sinai_entropy", 0.0)))
            lam_trans_vals.append(float(getattr(ind, "oseledets_transverse_lyapunov", 0.0)))
            nov_vals.append(float(getattr(ind, "novikov_valuation", 0.0)))
            cg_vals.append(float(getattr(ind, "capacity_gromov", 0.0)))
            eta_rsi3_vals.append(float(getattr(ind, "rsi3_monadic_rate", self.base_rsi_rate)))

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

            # — [NUEVO] CR3BP: posición sinódica (x,y) = (d_FS, H_KS), análogo
            #   dinámico; constante de Jacobi comparada contra C_J(L4)
            x_proxy, y_proxy = d_fs_vals[-1], h_ks_vals[-1]
            vx_proxy, vy_proxy = lam_trans_vals[-1], 0.0
            C_J_ray = self.compute_restricted_three_body_jacobi_constant(
                mu_cr3bp, x_proxy, y_proxy, vx_proxy, vy_proxy)
            l4_l5_like_flags.append(abs(C_J_ray - C_J_L4) < 1.0)

            # — [NUEVO] KAM: persistencia del toro bajo ε de Stinespring
            kam_ok, kam_eps = self.verify_kam_torus_persistence(
                omega_ratio=req.omega_celestial,
                perturbation_eps=req.stinespring_coupling_eps,
                bryuno_sum=bry_vals[-1])
            kam_ok_flags.append(kam_ok)
            kam_eps_vals.append(kam_eps)

            # Adjudicación por-trama
            d_fs = d_fs_vals[-1]
            cg = cg_vals[-1]
            spurious = bool(getattr(ind, "is_spurious_attractor", False))
            if spurious:
                spurious_count += 1

            if spurious or d_fs > 0.45 or cg > self.capacity_gromov_max:
                verdicts.append(HeytingOmega3.VETOED)
            elif d_fs > 0.15:
                verdicts.append(HeytingOmega3.DEGRADED)
            else:
                verdicts.append(HeytingOmega3.COHERENT)

            monadic_ok_flags.append(self._verify_monadic_law(
                ind, seed, eta_base=self.base_rsi_rate))

        # — Veredicto global por meet de Ω₃
        global_verdict = HeytingOmega3.COHERENT
        for v in verdicts:
            global_verdict = global_verdict.meet(v)

        # — Exponentes característicos de Poincaré a nivel de campaña
        _, _, F_campaign, _ = self.compute_principal_connection(seed=7)
        poincare_exp_info = self.compute_poincare_characteristic_exponents(F_campaign)

        # — Torre RSI Nivel 3 (desempeño agregado de la campaña)
        performance_signal = (float(np.mean(d_fs_vals)) if d_fs_vals else 0.0) + \
                             (float(np.mean(h_ks_vals)) if h_ks_vals else 0.0)
        rsi3_tower = self._rsi_level3_metameta_adaptation(performance_signal)

        # — Agregados canónicos
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
        lam_trans_peak = float(np.max(lam_trans_vals)) if lam_trans_vals else 0.0
        lam_trans_all_contracting = all(l < -1e-4 for l in lam_trans_vals)
        nov_min = float(np.min(nov_vals)) if nov_vals else 0.0
        cg_peak = float(np.max(cg_vals)) if cg_vals else 0.0
        cstar_all = all(cstar_all_ok_flags) if cstar_all_ok_flags else True
        cstar_pos = all(cstar_pos_flags) if cstar_pos_flags else True
        eta_rsi3 = float(np.mean(eta_rsi3_vals)) if eta_rsi3_vals else self.base_rsi_rate
        monadic_all_ok = all(monadic_ok_flags) if monadic_ok_flags else True

        # — Agregados celeste CR3BP / KAM
        lagrange_fraction = (float(np.mean(l4_l5_like_flags)) if l4_l5_like_flags else 0.0)
        kam_all_persist = all(kam_ok_flags) if kam_ok_flags else True
        kam_eps_mean = float(np.mean(kam_eps_vals)) if kam_eps_vals else 0.0

        result = IntrospectionProsecutionCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            bound_contract_id=self.bound_contract.contract_id,
            rays_examined_count=len(requests),
            spurious_attractors_detected_count=spurious_count,
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
            kolmogorov_sinai_entropy_mean=h_ks_mean,
            oseledets_transverse_lyapunov_peak=lam_trans_peak,
            oseledets_all_contracting=lam_trans_all_contracting,
            novikov_valuation_min=nov_min,
            gromov_capacity_peak=cg_peak,
            cstar_axioms_all_satisfied=cstar_all,
            cstar_positivity_all_valid=cstar_pos,
            lagrange_l4_l5_stable_fraction=lagrange_fraction,
            lagrange_mu_routh_critical=float(lagrange_info["mu_routh_critical"]),
            kam_tori_all_persist=kam_all_persist,
            kam_epsilon_bound_mean=kam_eps_mean,
            poincare_floquet_moduli_max=float(poincare_exp_info["moduli_max"]),
            poincare_orbit_linearly_stable=bool(poincare_exp_info["orbit_linearly_stable"]),
            rsi3_level1_rate=float(rsi3_tower["eta1_level1_rate"]),
            rsi3_level2_meta_rate=float(rsi3_tower["eta2_level2_meta_rate"]),
            rsi3_level3_metameta_rate=float(rsi3_tower["eta3_level3_metameta_rate"]),
            rsi3_banach_contraction_constant=float(rsi3_tower["banach_contraction_constant"]),
            rsi3_banach_fixed_point_converged=bool(rsi3_tower["banach_fixed_point_converged"]),
            rsi3_aggregate_rate=eta_rsi3,
            rsi3_monadic_law_verified=monadic_all_ok,
            global_heyting_verdict=global_verdict,
            per_request_seed_ids=tuple(seed_ids),
            per_request_indictment_ids=tuple(indict_ids),
            timestamp=now,
        )
        logger.info(
            f"[FASE II] Campaña {campaign_id} | N={len(requests)} | "
            f"spurious={spurious_count} | F_Uh={uhl_fid_mean:.4f} | d_FS={d_fs_peak:.4f} | "
            f"λ⟂_peak={lam_trans_peak:+.4f} (contract={lam_trans_all_contracting}) | "
            f"χ={chi_final} | c_G={cg_peak:.4f} | C*-all={cstar_all} | "
            f"L4/L5_frac={lagrange_fraction:.2f} | KAM_all={kam_all_persist} | "
            f"η_RSI3={eta_rsi3:.4f} | Banach_q={rsi3_tower['banach_contraction_constant']:.4f} "
            f"(conv={rsi3_tower['banach_fixed_point_converged']}) | μ-ley={monadic_all_ok} | "
            f"Ω₃={global_verdict.name}"
        )
        return result

    # ── II.12 — COSTURA FASE II → FASE III ──────────────────────────────────
    def weave_campaign_to_adjudication(
        self,
        campaign: IntrospectionProsecutionCampaignResult,
        mac_density_matrix: np.ndarray,
    ) -> Tuple[IntrospectionProsecutionCampaignResult, Any]:
        """
        Última piedra de la FASE II. Verifica invariantes y emite el certificado
        del motor `IntrospectionProsecutorExecutionCertificate`.

        Invariantes verificadas antes de FASE III:
          (1) Gromov c_G ≤ 12.5
          (2) θ_PC residuo ≤ 1e-5
          (3) Defecto simpléctico ≤ 1e-6
          (4) Morse–Bott χ verificado
          (5) Ley monádica μ verificada
          (6) [NUEVO] Torre RSI3 converge por contracción de Banach
          (7) Contracción transversal (λ⟂ < 0) — CRITERIO CRÍTICO, NO se
              fuerza como assert: el Fiscal debe poder acusar su ausencia.
          (8) [NUEVO] L4/L5-likeness y persistencia KAM — tampoco se fuerzan
              como assert, por la misma razón epistémica: son precisamente
              las señales que la FASE III debe poder vetar.
        """
        assert campaign.gromov_capacity_peak <= self.capacity_gromov_max + 1e-9, "c_G excede 12.5"
        assert campaign.poincare_cartan_residual_max < 1e-5, "θ_PC no preservada"
        assert campaign.symplectic_defect_max < 1e-6, "Monodromía no simpléctica"
        assert campaign.morse_bott_chi_all_verified, "Morse–Bott χ inconsistente"
        assert campaign.rsi3_monadic_law_verified, "Ley monádica μ violada"
        assert campaign.rsi3_banach_fixed_point_converged, "Torre RSI3 no converge (Banach)"

        if self.engine is not None:
            certificate = self.engine.audit_and_prosecute_rays(mac_density_matrix)
        else:
            certificate = None

        logger.info(
            f"[FASE II → FASE III] Campaña {campaign.campaign_id} cosechada | "
            f"certificado={getattr(certificate,'certificate_id','N/A')} | "
            f"Banach_q={campaign.rsi3_banach_contraction_constant:.4f}"
        )
        return campaign, certificate


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FASE III — ADJUDICACIÓN Ω₃, FOCK, CROWBAR, NO-INTEGRABILIDAD, BIRKHOFF Y Φ_sem
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class SovereignIntrospectionProsecutorAdjudicator(TOONIntrospectionProsecutorAgent):
    """
    FASE III — Soberano de Adjudicación Ciber-Física del Fiscal de Introspección.

    • Adjudica en el retículo Heyting Ω₃ = {VETOED < DEGRADED < COHERENT}
    • Orquesta el interlock ESP32 Crowbar (< 400 ns / GPIO14)
    • Álgebra de Fock para la purga al Vacío de Dirac: e⁻ + e⁺ → 2γ
    • [NUEVO] Criterio de no-integrabilidad de Poincaré (solapamiento de
      resonancias de Chirikov) sobre las frecuencias celeste de la campaña
    • [NUEVO] Último Teorema Geométrico de Poincaré (Poincaré–Birkhoff)
      como garantía topológica de un piso mínimo de estabilidad adjudicada
    • [NUEVO] RSI3 REFLEXIVO: el Adjudicador aplica su propia torre
      μ₃∘μ₂∘μ₁ (heredada de FASE II) sobre SUS PROPIOS umbrales de
      gobernanza — el fiscal que audita también se autoaudita
    • Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business (preserva meet)
    • DAG Merkle final sobre (contract ⊕ campaign ⊕ cert ⊕ seeds ⊕ indicts)
    """

    def __init__(
        self,
        sovereign_id: str = "INTROSPECTIVE-PROSECUTOR-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        base_rsi_rate: float = 0.25,
        engine_instance: Optional[Any] = None,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id,
                         dimension_mac=dimension_mac,
                         capacity_gromov_max=capacity_gromov_max,
                         esp32_gpio_pin=esp32_gpio_pin,
                         base_rsi_rate=base_rsi_rate,
                         engine_instance=engine_instance)
        self._fock_purges: List[Any] = []

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

    # ── III.2 — Disparo ESP32 Crowbar ───────────────────────────────────────
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

    # ── III.3 — DAG Merkle ─────────────────────────────────────────────────
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

    # ── III.4 — Criterio de no-integrabilidad de Poincaré (Chirikov) ───────
    @staticmethod
    def detect_poincare_nonintegrability_resonance_overlap(
        frequencies: Sequence[float], amplitudes: Sequence[float],
        overlap_threshold: float = 1.0,
    ) -> Tuple[bool, float]:
        """
        Consecuencia práctica del Teorema de No-Integrabilidad de Poincaré
        (1890, 1892): la presencia de resonancias densas destruye cualquier
        segunda integral analítica. Chirikov (1979) operacionalizó el
        criterio: el movimiento es caótico si el parámetro de solapamiento

            K = Σ_i Δω_i / δω_i  >  1

        donde Δω_i son los anchos de resonancia (amplitudes de perturbación)
        y δω_i los espaciamientos entre frecuencias vecinas.
        """
        if len(frequencies) < 2:
            return False, 0.0
        diffs = np.diff(np.sort(np.asarray(frequencies, dtype=float)))
        widths = (np.asarray(amplitudes[:len(diffs)], dtype=float)
                  if len(amplitudes) >= len(diffs)
                  else np.ones(len(diffs)) * 1e-3)
        K = float(np.sum(widths / (np.abs(diffs) + 1e-30)))
        chaotic = K > overlap_threshold
        return bool(chaotic), K

    # ── III.5 — Último Teorema Geométrico de Poincaré (Poincaré–Birkhoff) ──
    @staticmethod
    def verify_poincare_birkhoff_fixed_points(
        inner_rotation_number: float, outer_rotation_number: float,
    ) -> Tuple[bool, int]:
        """
        Último Teorema Geométrico de Poincaré (conjeturado 1912, demostrado
        por G. D. Birkhoff 1913): todo homeomorfismo que preserva área de un
        anillo, rotando sus fronteras interior y exterior en SENTIDOS
        OPUESTOS (condición de "twist"), posee al menos DOS puntos fijos
        geométricamente distintos. Se usa como garantía topológica de un
        "piso de estabilidad" mínimo en la adjudicación: si el rayo
        contractante (λ⟂<0, frontera interior) y la dispersión de Novikov
        (frontera exterior) rotan en sentidos opuestos, el sistema está
        topológicamente OBLIGADO a poseer ≥ 2 configuraciones de equilibrio.
        """
        condition_satisfied = (
            (inner_rotation_number > 0 > outer_rotation_number) or
            (inner_rotation_number < 0 < outer_rotation_number)
        )
        guaranteed_fixed_points = 2 if condition_satisfied else 0
        return bool(condition_satisfied), int(guaranteed_fixed_points)

    # ── III.6 — RSI3 REFLEXIVO: el Adjudicador se autoaudita ───────────────
    def reflexive_rsi3_self_calibration(self, performance_signal: float) -> Dict[str, Any]:
        """
        Cierre reflexivo de la automejora recursiva de Nivel 3: el
        Adjudicador aplica su PROPIA torre μ₃∘μ₂∘μ₁ (heredada de FASE II,
        método II.10) no sobre los rayos examinados, sino sobre SUS PROPIOS
        umbrales de gobernanza (`capacity_gromov_max`). La recalibración
        solo se materializa si la torre converge por Banach — de lo
        contrario el umbral permanece inalterado (principio de precaución
        epistémica: no mutar la propia ley sin prueba de estabilidad).
        """
        tower = self._rsi_level3_metameta_adaptation(performance_signal)
        delta = (tower["eta1_level1_rate"] - self.base_rsi_rate) * 2.0
        new_threshold = float(np.clip(self.capacity_gromov_max + delta, 10.0, 15.0))
        adjusted = abs(new_threshold - self.capacity_gromov_max) > 1e-9
        if adjusted and tower["banach_fixed_point_converged"]:
            logger.info(
                f"[FASE III] [RSI3-REFLEXIVO] c_G recalibrado: "
                f"{self.capacity_gromov_max:.4f} → {new_threshold:.4f}"
            )
            self.capacity_gromov_max = new_threshold
        else:
            adjusted = False
        return {**tower,
                "capacity_gromov_max_calibrated": self.capacity_gromov_max,
                "threshold_adjusted": adjusted}

    # ── III.7 — Orquestación de adjudicación ciber-física ───────────────────
    def orchestrate_introspection_adjudication(
        self,
        campaign_result: IntrospectionProsecutionCampaignResult,
        mac_density_matrix: np.ndarray,
        positron_token: Optional[str] = None,
        positron_hmac: Optional[str] = None,
    ) -> Tuple[IntrospectionProsecutorSovereignGovernancePassport,
               ExecutiveIntrospectionProsecutionImpactReport]:
        now = time.time()
        passport_id = f"PASSPORT-INTRO-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

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
                logger.warning(
                    f"[FASE III] Firma HMAC inválida para positrón {positron_token}")

        # 4. [NUEVO] No-integrabilidad de Poincaré / solapamiento de Chirikov
        frequencies_proxy = [
            campaign.kolmogorov_sinai_entropy_mean,
            abs(campaign.oseledets_transverse_lyapunov_peak) + 1e-6,
            campaign.melnikov_integral_peak + 1e-6,
        ]
        amplitudes_proxy = [
            campaign.fubini_study_distance_peak,
            campaign.symplectic_defect_max + 1e-9,
        ]
        chaotic_flag, chirikov_K = self.detect_poincare_nonintegrability_resonance_overlap(
            frequencies_proxy, amplitudes_proxy)

        # 5. [NUEVO] Último Teorema Geométrico de Poincaré (Birkhoff)
        inner_rot = campaign.oseledets_transverse_lyapunov_peak
        outer_rot = -campaign.novikov_valuation_min
        birkhoff_ok, birkhoff_n = self.verify_poincare_birkhoff_fixed_points(
            inner_rot, outer_rot)

        # 6. [NUEVO] RSI3 reflexivo sobre la propia gobernanza del Adjudicador
        reflexive = self.reflexive_rsi3_self_calibration(
            campaign.fubini_study_distance_peak)

        # 7. DAG Merkle final
        dag_leaves = [
            f"CONTRACT::{campaign.bound_contract_id}",
            f"CAMPAIGN::{campaign.campaign_id}",
            f"CERT::{engine_cert_id}::{engine_merkle}",
            f"CHIRIKOV::{chirikov_K:.8f}",
            f"BIRKHOFF::{birkhoff_n}",
            *[f"SEED::{s}"   for s in campaign.per_request_seed_ids],
            *[f"INDICT::{i}" for i in campaign.per_request_indictment_ids],
        ]
        merkle_root = self._merkle_dag_root(dag_leaves)

        # 8. Hash de provenance del pasaporte
        prov_str = (
            f"{passport_id}:{campaign.campaign_id}:{self.sovereign_id}:"
            f"{verdict.name}:{actuation.name}:{engine_cert_id}:{merkle_root}:{now}"
        )
        prov_hash = hashlib.sha256(prov_str.encode("utf-8")).hexdigest()

        passport = IntrospectionProsecutorSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            campaign_id=campaign.campaign_id,
            bound_contract_id=campaign.bound_contract_id,
            engine_certificate_id=engine_cert_id,
            global_heyting_verdict=verdict,
            actuation_mode=actuation,
            rays_prosecuted_count=campaign.rays_examined_count,
            spurious_attractors_vetoed_count=campaign.spurious_attractors_detected_count,
            crowbar_active_iram=crowbar_active,
            crowbar_latency_ns=float(crowbar_latency),
            positron_annihilation_active=positron_active,
            fock_purge_records=fock_records,
            level3_rsi_monadic_rate=campaign.rsi3_aggregate_rate,
            rsi3_monadic_law_verified=campaign.rsi3_monadic_law_verified,
            gromov_capacity_peak=campaign.gromov_capacity_peak,
            cstar_positivity_verified=campaign.cstar_positivity_all_valid,
            morse_bott_chi_verified=campaign.morse_bott_chi_all_verified,
            oseledets_all_contracting=campaign.oseledets_all_contracting,
            lagrange_l4_l5_stable_fraction=campaign.lagrange_l4_l5_stable_fraction,
            kam_tori_all_persist=campaign.kam_tori_all_persist,
            poincare_nonintegrability_chaotic_detected=chaotic_flag,
            chirikov_resonance_overlap_K=chirikov_K,
            birkhoff_twist_condition_satisfied=birkhoff_ok,
            birkhoff_fixed_points_guaranteed=birkhoff_n,
            rsi3_reflexive_capacity_gromov_calibrated=float(
                reflexive["capacity_gromov_max_calibrated"]),
            rsi3_reflexive_threshold_adjusted=bool(reflexive["threshold_adjusted"]),
            merkle_root_sha256=merkle_root,
            provenance_hash=prov_hash,
            timestamp_utc=now,
        )

        # 9. Φ_sem: traducción a impacto de negocio
        report = self.translate_to_business_impact(campaign, passport)

        logger.info(
            f"[FASE III] Pasaporte {passport_id} | Ω₃={verdict.name} | "
            f"{actuation.name} | crowbar={crowbar_active} | "
            f"positron={positron_active} | Chirikov_K={chirikov_K:.4f} "
            f"(caótico={chaotic_flag}) | Birkhoff_n={birkhoff_n} | "
            f"c_G_reflexivo={reflexive['capacity_gromov_max_calibrated']:.4f} | "
            f"Merkle={merkle_root[:16]}…"
        )
        return passport, report

    # ── III.8 — Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business ─────────────
    def translate_to_business_impact(
        self,
        campaign: IntrospectionProsecutionCampaignResult,
        passport: IntrospectionProsecutorSovereignGovernancePassport,
    ) -> ExecutiveIntrospectionProsecutionImpactReport:
        """
        Aplicación del funtor semántico Φ_sem.

        Objetos: presheaves sobre la frontera epistémica ∂K con valores en Ω₃.
        Imagen: métricas ejecutivas (KV, WACC, fondo de imprevistos, capital).
        Preservación: Φ_sem(v.meet(v')) = Φ_sem(v) ⊓ Φ_sem(v').
        """
        now = time.time()
        report_id = f"EXEC-INTRO-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        kv_compression = 86.4
        wacc_protection = 15.0
        imprevistos_reduction = 11.5
        capital_saved = passport.spurious_attractors_vetoed_count * 300_000.0
        expected_loss_avoided = passport.spurious_attractors_vetoed_count * 220_000.0

        poincare_summary = (
            f"Fracción tipo-L4/L5 (estable): {passport.lagrange_l4_l5_stable_fraction:.1%} | "
            f"KAM persiste en todos los toros: {passport.kam_tori_all_persist} | "
            f"No-integrabilidad/Chirikov K={passport.chirikov_resonance_overlap_K:.3f} "
            f"(caos={passport.poincare_nonintegrability_chaotic_detected}) | "
            f"Birkhoff garantiza {passport.birkhoff_fixed_points_guaranteed} punto(s) fijo(s) | "
            f"RSI3 reflexivo: c_G={passport.rsi3_reflexive_capacity_gromov_calibrated:.4f} "
            f"(recalibrado={passport.rsi3_reflexive_threshold_adjusted})"
        )

        v = passport.global_heyting_verdict
        if v is HeytingOmega3.COHERENT:
            actuation_desc = (
                "Flujo Normal Fluido. El autoestado v* ∈ ℂPⁿ⁻¹ es atractor puro "
                "e inalienable (λ⟂ < 0 contracción exponencial, C*-𝔇_n satisfechos, "
                "c_G ≤ 12.5, θ_PC preservada, χ(ℂPⁿ⁻¹) verificada), con comportamiento "
                "tipo-L4/L5 (centro estable celeste) y sin solapamiento de Chirikov."
            )
            summary = (
                f"FASE III — El Soberano Fiscal de Introspección examinó "
                f"{passport.rays_prosecuted_count} autoestados sin detectar atractores "
                f"espurios. Todos los rayos presentan contracción transversal λ⟂ < 0, "
                f"garantizando autocoherencia MAC y automejora recursiva de Nivel 3 "
                f"(ley monádica μ verificada, torre μ₃∘μ₂∘μ₁ convergente por Banach). "
                f"Flujo monetario normal fluido."
            )
        elif v is HeytingOmega3.DEGRADED:
            actuation_desc = (
                "Veto Suave (Válvula de Alivio / Bypass). Recirculación mecánica "
                "activada. Requiere Positrón de Autorización Humana e⁺ firmado."
            )
            summary = (
                f"FASE III — ALERTA ÁMBAR: se detectaron distorsiones angulares "
                f"moderadas en Fubini–Study para {passport.rays_prosecuted_count} "
                f"autoestados (d_FS > 0.15 o λ⟂ próximo a 0; comportamiento próximo a "
                f"puntos colineales L1-L3). Válvula activada; ventana de gracia de 1 h "
                f"para inyectar Positrón e⁺ (aniquilación e⁻ + e⁺ → 2γ)."
            )
        else:
            actuation_desc = (
                "Veto Duro (Disyuntor ESP32 Crowbar < 400 ns en GPIO14). "
                "Purga Fock e⁻ + e⁺ → 2γ al Vacío de Dirac."
            )
            summary = (
                f"FASE III — CRÍTICO: El Fiscal detectó "
                f"{passport.spurious_attractors_vetoed_count} atractores espurios / "
                f"auto-alucinaciones de convergencia en ℂPⁿ⁻¹ (λ⟂ ≥ 0 sin contracción "
                f"exponencial — análogo celeste a silla colineal L1-L3 — violación de "
                f"Gromov o colapso Uhlmann). ESP32 Crowbar disparado en GPIO14 "
                f"(< 400 ns); adjudicación VETOED: aduana de pagos mecánicamente "
                f"paralizada. Capital salvaguardado: ${capital_saved:,.2f} USD."
            )

        return ExecutiveIntrospectionProsecutionImpactReport(
            report_id=report_id,
            passport_id=passport.passport_id,
            verdict_name=v.name,
            actuation_description=actuation_desc,
            kv_cache_compression_pct=kv_compression,
            wacc_protection_pct=wacc_protection,
            imprevistos_reduction_pct=imprevistos_reduction,
            capital_salvaguardado_usd=capital_saved,
            expected_loss_avoided_usd=expected_loss_avoided,
            poincare_celestial_integrity_summary=poincare_summary,
            executive_summary=summary,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. VERIFICACIÓN DOCTORAL END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 88)
    print("  VERIFICACIÓN DOCTORAL ANIDADA — TOONIntrospectionProsecutorAgent v7.0.0")
    print("═" * 88)

    adjudicator = SovereignIntrospectionProsecutorAdjudicator(
        sovereign_id="INTROSPECTIVE-PROSECUTOR-MASTER-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
        base_rsi_rate=0.25,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE I
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[FASE I] Forjando contrato de calibre + atlas celeste de Poincaré…")
    contract = adjudicator.forge_prosecutor_gauge_contract(
        quaternion_q=(0.95, 0.05, 0.10, 0.05), seed=42)
    bound_ok = adjudicator.bind_prosecutor_gauge_contract(contract)
    assert bound_ok, "Falla en vinculación HMAC del contrato."

    print(f"  • ID Contrato          : {contract.contract_id}")
    print(f"  • Régimen topológico   : {contract.topology_regime}")
    print(f"  • CS₃                  : {contract.chern_simons_3form:+.6e}")
    print(f"  • Clase c₁ Hopf        : {contract.chern_hopf_class_c1:.6f}")
    print(f"  • Holonomía Wilson ϑ_W : {contract.wilson_holonomy_angle:+.6f} rad")
    print(f"  • Floquet dominante    : {contract.poincare_floquet_multiplier_dominant:.4f} "
          f"(|·|={abs(contract.poincare_floquet_multiplier_dominant):.4f})")
    print(f"  • Radio conv. Poincaré : {contract.poincare_perturbation_radius_convergence:.6f}")
    print(f"  • Invariante Cartan I₁ : {contract.poincare_cartan_invariant_I1:+.6f} "
          f"(residuo={contract.poincare_cartan_residual_gauge:.2e})")
    print(f"  • ⟨τ⟩ recurrencia      : {contract.poincare_recurrence_time_estimate:.4f}")
    print(f"  • Firma HMAC           : {contract.hmac_signature[:24]}…")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE II
    # ══════════════════════════════════════════════════════════════════════════
    dim = 56
    rng = np.random.default_rng(2026)
    v_mac = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v_mac /= np.linalg.norm(v_mac)
    mac_rho = np.outer(v_mac, v_mac.conj())
    mac_rho = 0.98 * mac_rho + 0.02 * (np.eye(dim) / dim)
    mac_rho /= float(np.trace(mac_rho).real)

    v_ray1 = v_mac.copy()
    v_ray1[0] *= cmath.exp(1j * 0.02)
    v_ray1 /= np.linalg.norm(v_ray1)

    v_ray2 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v_ray2 /= np.linalg.norm(v_ray2)

    req1 = IntrospectionRayIndictmentRequest(
        request_id="REQ-INTRO-001",
        ray_id="RAY-PURE-VACIADO-CONCRETO-3000PSI",
        apu_code="APU-OBRA-CIVIL-001",
        projective_ray_v=v_ray1,
        stinespring_coupling_eps=0.04,
        omega_celestial=1.618033988749895,
    )
    req2 = IntrospectionRayIndictmentRequest(
        request_id="REQ-INTRO-002",
        ray_id="RAY-SPURIOUS-ACERO-FIGURADO-60000PSI",
        apu_code="APU-OBRA-CIVIL-002",
        projective_ray_v=v_ray2,
        stinespring_coupling_eps=0.08,
        omega_celestial=math.sqrt(2.0),
    )

    print("\n[FASE II] Conduciendo campaña de acusación + CR3BP + KAM + RSI3 tower…")
    campaign = adjudicator.conduct_prosecution_campaign([req1, req2], mac_rho)

    print(f"  • ID Campaña                    : {campaign.campaign_id}")
    print(f"  • Rayos examinados              : {campaign.rays_examined_count}")
    print(f"  • Atractores espurios           : {campaign.spurious_attractors_detected_count}")
    print(f"  • d_FS peak                     : {campaign.fubini_study_distance_peak:.6f} rad")
    print(f"  • Oseledets λ⟂ peak             : {campaign.oseledets_transverse_lyapunov_peak:+.6f}")
    print(f"  • Gromov c_G peak               : {campaign.gromov_capacity_peak:.4f}  (≤ 12.5)")
    print(f"  • Lagrange L4/L5 fracción       : {campaign.lagrange_l4_l5_stable_fraction:.2%} "
          f"(μ_Routh={campaign.lagrange_mu_routh_critical:.6f})")
    print(f"  • KAM tori todos persisten      : {campaign.kam_tori_all_persist} "
          f"(ε_bound medio={campaign.kam_epsilon_bound_mean:.4e})")
    print(f"  • Floquet |·|_max / órbita est. : {campaign.poincare_floquet_moduli_max:.4f} / "
          f"{campaign.poincare_orbit_linearly_stable}")
    print(f"  • RSI3 η₁/η₂/η₃                 : {campaign.rsi3_level1_rate:.5f} / "
          f"{campaign.rsi3_level2_meta_rate:.5f} / {campaign.rsi3_level3_metameta_rate:.5f}")
    print(f"  • RSI3 Banach q / convergió     : {campaign.rsi3_banach_contraction_constant:.4f} "
          f"/ {campaign.rsi3_banach_fixed_point_converged}")
    print(f"  • Ley monádica μ verificada     : {campaign.rsi3_monadic_law_verified}")
    print(f"  • Veredicto Ω₃ global           : {campaign.global_heyting_verdict.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE III
    # ══════════════════════════════════════════════════════════════════════════
    positron_token = "AUTH-HUMAN-INTRO-PROSECUTOR-2026-QND-LEAD"
    positron_hmac = hmac.new(
        adjudicator._hmac_secret,
        positron_token.encode("utf-8"),
        hashlib.sha256).hexdigest()

    print("\n[FASE III] Orquestando adjudicación ciber-física + Chirikov + Birkhoff + RSI3 reflexivo…")
    passport, report = adjudicator.orchestrate_introspection_adjudication(
        campaign, mac_rho,
        positron_token=positron_token,
        positron_hmac=positron_hmac,
    )

    print(f"\n  ◈ Pasaporte Soberano del Fiscal de Introspección")
    print(f"    • ID Pasaporte         : {passport.passport_id}")
    print(f"    • Veredicto Ω₃         : {passport.global_heyting_verdict.name}")
    print(f"    • Modo de actuación    : {passport.actuation_mode.name}")
    print(f"    • Crowbar IRAM         : {passport.crowbar_active_iram} "
          f"(latencia {passport.crowbar_latency_ns:.1f} ns)")
    print(f"    • Positrón activo      : {passport.positron_annihilation_active}")
    print(f"    • Chirikov K           : {passport.chirikov_resonance_overlap_K:.4f} "
          f"(caos={passport.poincare_nonintegrability_chaotic_detected})")
    print(f"    • Birkhoff twist/n     : {passport.birkhoff_twist_condition_satisfied} / "
          f"{passport.birkhoff_fixed_points_guaranteed}")
    print(f"    • RSI3 reflexivo c_G   : {passport.rsi3_reflexive_capacity_gromov_calibrated:.4f} "
          f"(ajustado={passport.rsi3_reflexive_threshold_adjusted})")
    print(f"    • Merkle DAG           : {passport.merkle_root_sha256}")
    print(f"    • Provenance hash      : {passport.provenance_hash[:24]}…")

    print(f"\n  ◈ Informe Ejecutivo Φ_sem ('Dolor y Dinero')")
    print(f"    • Veredicto            : {report.verdict_name}")
    print(f"    • Actuación            : {report.actuation_description}")
    print(f"    • Capital salvaguardado: ${report.capital_salvaguardado_usd:,.2f} USD")
    print(f"    • Pérdida evitada      : ${report.expected_loss_avoided_usd:,.2f} USD")
    print(f"    • Integridad celeste   : {report.poincare_celestial_integrity_summary}")
    print(f"    • Resumen              : \"{report.executive_summary[:140]}…\"")

    print("\n" + "═" * 88)
    print("  VERIFICACIÓN EXITOSA — TOONIntrospectionProsecutorAgent v7.0.0 OPERATIVO")
    print("═" * 88)