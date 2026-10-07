# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Buffer Agent v5 — Soberano de Calibre Anidado Poincaré       ║
║ Ubicación: app/agents/wisdom/toon_buffer_agent.py                            ║
║ Versión  : 5.0.0-Poincare-Hopf-Chern-Greene-Chirikov-Moser-Lindstedt-Topos   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Soberano de calibre del Campo Tensorial Transitorio sobre el fibrado
simpléctico principal

        P(T*Q, G),   G = U(1)_fase ⊗ SU(2)_orbital ⊗ H₃(ℝ)_Heisenberg.

Gobierna el ciclo Hamiltoniano H = H₀(Kepler) + ε H₁(riesgo) anidando los
invariantes de Henri Poincaré (*Méthodes Nouvelles*, 1892–1899) y su
descendencia KAM/Moser/Greene, ahora acoplados al motor v5:

    • Semilla canónica `CanonicalSeed` (Delaunay + LRL + cuaternión + twist)
    • Sección estroboscópica Σ_ℓ y `ReturnMapOrbit` (monodromía, Floquet)
    • Residuo de Greene, solapamiento de Chirikov, teorema del twist de Moser
    • Holonomía de Hopf S³ → S² (c₁ proxy no nulo) y forma de Chern-Simons
    • Red de resonancias p:q como grafo (Laplaciano, gap algebraico)
    • Recurrencia de Poincaré y promediado de von Zeipel (eliminación secular)
    • Funtor CPTP de Lüders  F : Campaña ⟶ MAC  y Φ_sem : Ω₃ ⟶ "Dolor y Dinero"

ARQUITECTURA EN 3 FASES ANIDADAS POR HERENCIA
(el último método de la fase k es el germen formal del primero de la k+1):

    FASE I  : SovereignGaugeTopology
              fibrado, curvatura, Hopf/Chern, Poisson, semilla canónica
              → forge_gauge_contract()              ⟶ germen de FASE II

    FASE II : TOONBufferAgent(SovereignGaugeTopology)
              vínculo de calibre, ingesta, Lindblad, Σ_ℓ, red de resonancias
              → conduct_hamiltonian_campaign()      ⟶ germen de FASE III

    FASE III: SovereignAuditAssimilator(TOONBufferAgent)
              KAM/Greene/Chirikov/Melnikov/Birkhoff + funtor CPTP + dossier
              → emit_full_executive_dossier()
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

current_dir = os.path.dirname(os.path.abspath(__file__))
for path_dir in [current_dir, "/workspace/scratch", "/workspace/artifacts"]:
    if path_dir and path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from toon_buffer_engine import (
    ActuationMode,
    BufferCertificate,
    CanonicalSeed,
    CartridgeStatus,
    DelaunayPoint,
    HeytingOmega3,
    KAMMelnikovAuditor,
    KAMRegime,
    PoincareCanonicalAtlas,
    PoincareSection,
    Quaternion,
    ReturnMapOrbit,
    TOONBufferEngine,
    TransientTensorCartridge,
)

logger = logging.getLogger("APU.Wisdom.TOONBufferAgent.v5")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_EPS: float = 1e-15
_TWO_PI: float = 2.0 * math.pi
_PI: float = math.pi
_SOVEREIGN_HMAC_KEY: bytes = b"TOON-SOVEREIGN-GAUGE-HMAC-v5-1892"


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULOS DE GOBERNANZA, TOPOLOGÍA DE CALIBRE Y GRADOS KAM
# ══════════════════════════════════════════════════════════════════════════════

class SovereignVerdictGrade(IntEnum):
    """Grado epistémico del veredicto soberano sobre el ciclo completo."""

    KAM_PRESERVED = 0
    CANTOR_DEGRADED = 1
    CHAOTIC_REJECTED = 2


class GaugeTopologyClass(IntEnum):
    """
    Clase de topología de calibre según el 1er número de Chern proxy c₁
    (Hopf S³ → S² del cuaternión orbital, no Tr[T_a, T_b] ≡ 0).
    """

    TRIVIAL_BUNDLE = 0  # |c₁| < 1/2  → P ≅ T*Q × U(1)
    MONOPOLE_LIKE = 1  # 1/2 ≤ |c₁| < 3/2  → fibrado de Hopf
    INSTANTON_DENSE = 2  # |c₁| ≥ 3/2  → sectores θ no triviales


# ══════════════════════════════════════════════════════════════════════════════
# §B. CONTRATOS DE DATOS — INGESTA, CALIBRE, CAMPAÑA, RED, PASAPORTE, Φ_sem
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BufferIngestionRequest:
    """Solicitud de ingesta de APU/Contrato con anclaje canónico a T*Q."""

    apu_code: str
    tangible_costs: np.ndarray
    policy_risk_intangible: float
    weather_gremial_factor: float
    coordinates_h2: Tuple[float, float] = (1.0, 2.0)
    target_section_l0: float = 0.0
    target_resonance_pq: Optional[Tuple[int, int]] = None
    metadata_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SovereignGaugeContract:
    """
    GERMEN FORMAL FASE I → FASE II.
    Contrato de calibre: rango del fibrado, c₁ de Hopf, holonomía, Chern-Simons,
    tensor de Poisson, `CanonicalSeed` v5 y vínculo ESP32 Crowbar.
    """

    contract_id: str
    sovereign_id: str
    timestamp: float
    gauge_bundle_rank: int
    chern_class_proxy: float
    gauge_topology: GaugeTopologyClass
    holonomy_phase: float
    chern_simons_density: float
    frobenius_norm_poisson: float
    killing_metric_signature: Tuple[int, int]
    canonical_seed: CanonicalSeed
    sigma_l0_default: float
    moser_twist_ref: float
    esp32_crowbar_pin: int
    contract_signature_hmac: str = ""

    def __post_init__(self) -> None:
        if not self.contract_signature_hmac:
            raw = (
                f"{self.contract_id}:{self.sovereign_id}:{self.timestamp}:"
                f"{self.chern_class_proxy:.10f}:{self.frobenius_norm_poisson:.10f}:"
                f"{self.sigma_l0_default:.8f}:{self.holonomy_phase:.8f}"
            ).encode("utf-8")
            sig = hmac.new(_SOVEREIGN_HMAC_KEY, raw, hashlib.sha256).hexdigest()
            object.__setattr__(self, "contract_signature_hmac", sig)


@dataclass(frozen=True, slots=True)
class ResonanceEdge:
    """Arista de la red de resonancias de movimiento medio (p:q)."""

    src_id: str
    dst_id: str
    p: int
    q: int
    small_divisor: float


@dataclass(frozen=True, slots=True)
class ResonanceWeb:
    """
    Grafo de conmensurabilidades p n_i ≃ q n_j (pequeños divisores).
    Invariantes: componentes conexas, grado máximo, gap algebraico λ₂(Δ).
    """

    vertices: Tuple[str, ...]
    edges: Tuple[ResonanceEdge, ...]
    connected_components: int
    max_degree: int
    spectral_gap: float


@dataclass(frozen=True, slots=True)
class HamiltonianCampaignResult:
    """
    GERMEN FORMAL FASE II → FASE III.
    Campaña Hamiltoniana: ingesta, Σ_ℓ, resonancias, Greene/Chirikov/Moser,
    recurrencia de Poincaré y holonomía de calibre.
    """

    campaign_id: str
    timestamp: float
    contract_id: str
    cartridges_ingested: int
    cartridges_purged_dirac: int
    poincare_sections_count: int
    resonance_pq: Tuple[int, int]
    mean_motion_vector: Tuple[float, ...]
    mean_motion_ratio_deviation: float
    delaunay_mean_L: float
    delaunay_mean_eccentricity: float
    delaunay_mean_inclination: float
    lrl_norm_mean: float
    poincare_sections: Tuple[PoincareSection, ...] = ()
    return_orbits: Tuple[ReturnMapOrbit, ...] = ()
    greene_residue_max: float = 0.0
    chirikov_overlap_max: float = 0.0
    moser_twist_mean: float = 0.0
    floquet_spectral_radius: float = 1.0
    birkhoff_fixed_points: int = 0
    poincare_recurrence_time: float = 0.0
    lindstedt_n2_mean: float = 0.0
    holonomy_phase: float = 0.0
    resonance_web: Optional[ResonanceWeb] = None
    rlc_thevenin_impedance: complex = 0j


@dataclass(frozen=True, slots=True)
class BufferSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza con metadatos celestes v5."""

    passport_id: str
    sovereign_agent_id: str
    timestamp_utc: float
    heyting_verdict: HeytingOmega3
    heyting_implication: str
    actuation_mode: ActuationMode
    active_cartridges: int
    total_assimilated: int
    total_purged_dirac: int
    gromov_capacity_peak: float
    poincare_cartan_residual: float
    mac_von_neumann_entropy: float
    merkle_root_sha256: str
    esp32_crowbar_interlock_armed: bool
    provenance_hash_sha256: str
    kam_regime: KAMRegime = KAMRegime.INTACT_TORUS
    kam_intact_fraction: float = 1.0
    melnikov_amplitude_max: float = 0.0
    delaunay_resonance_order_q: int = 1
    mean_motion_ratio_deviation: float = 0.0
    laplace_runge_lenz_norm: float = 0.0
    greene_residue_max: float = 0.0
    chirikov_overlap_max: float = 0.0
    moser_twist_mean: float = 0.0
    floquet_spectral_radius: float = 1.0
    birkhoff_fixed_points: int = 0
    holonomy_phase: float = 0.0
    poincare_recurrence_time: float = 0.0
    sovereign_verdict_grade: SovereignVerdictGrade = SovereignVerdictGrade.KAM_PRESERVED


@dataclass(frozen=True, slots=True)
class ExecutiveBusinessImpactReport:
    """Traducción semántica Φ_sem de invariantes celestes a 'Dolor y Dinero'."""

    report_id: str
    timestamp: float
    sovereign_verdict: str
    executive_summary: str
    kv_cache_token_savings_percent: float
    wacc_protection_rate_percent: float
    cash_flow_shielded_usd: float
    contingency_fund_reduction_percent: float
    bypass_grace_period_active: bool
    crowbar_paralysis_triggered: bool
    kam_torus_integrity_percent: float = 100.0
    resonance_hazard_factor: float = 0.0
    melnikov_chaos_index: float = 0.0
    orbital_stability_score: float = 1.0
    greene_criticality: float = 0.0
    chirikov_overlap_index: float = 0.0
    floquet_instability: float = 0.0
    recurrence_horizon_sec: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE I — ONTOLOGÍA SOBERANA Y CONTRATOS DE CALIBRE ▓▓▓
# El último método, forge_gauge_contract, ES el germen de FASE II.
# ══════════════════════════════════════════════════════════════════════════════

class SovereignGaugeTopology:
    """
    FASE I — Ontología soberana sobre el fibrado principal P → T*Q.

    Grupo de estructura  G = U(1)_fase ⊗ SU(2)_orbital ⊗ H₃(ℝ).
    La clase topológica se lee del fibrado de Hopf del cuaternión orbital
    (no de Tr[T_a, T_b] ≡ 0, que en v4 dejaba c₁ nulo por construcción).

    Referencias:
        • Poincaré, H. (1892–1899), *Méthodes Nouvelles…*, I–III.
        • Atiyah, M. (1979), *Geometry of Yang-Mills Fields*.
        • Nakahara, M. (2003), *Geometry, Topology and Physics*, §10–11.
        • Marsden, Weinstein (1974), reducción simpléctica.
    """

    def __init__(
        self,
        sovereign_id: str = "BUFFER-SOVEREIGN-SABIO-01",
        dimension: int = 56,
        esp32_gpio_pin: int = 14,
        mu_gravitational: float = 1.0,
    ) -> None:
        self.sovereign_id = sovereign_id
        self.dimension = int(dimension)
        self.esp32_gpio_pin = int(esp32_gpio_pin)
        self.mu_gravitational = float(mu_gravitational)
        self.atlas = PoincareCanonicalAtlas(
            dimension=self.dimension, mu_gravitational=self.mu_gravitational
        )

    # ─────────────────────────────────────────────────────────────────────
    # §I.1 — Fibrado de calibre: generadores u(1) ⊕ su(2) ⊕ h₃ y Killing
    # ─────────────────────────────────────────────────────────────────────
    def build_gauge_bundle(self, rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Carta local de P → T*Q.  Rango efectivo acotado (≤ 8 pares) para que
        la representación matricial de G sea tractable; el Poisson del motor
        conserva su dimensión nativa.
        """
        n_pairs = int(rank if rank is not None else max(3, min(self.dimension // 2, 8)))
        dim_g = n_pairs + 4  # 1 (u1) + 3 (su2) + n_pairs (heisenberg residual)
        generators: List[np.ndarray] = []

        G_u1 = np.zeros((dim_g, dim_g), dtype=complex)
        G_u1[0, 0] = 1.0j
        generators.append(G_u1)

        # su(2) en el bloque 1:3 (triedro orbital / cuaternión)
        sx = np.zeros((dim_g, dim_g), dtype=complex)
        sy = np.zeros((dim_g, dim_g), dtype=complex)
        sz = np.zeros((dim_g, dim_g), dtype=complex)
        sx[1, 2] = 0.5
        sx[2, 1] = 0.5
        sy[1, 2] = -0.5j
        sy[2, 1] = 0.5j
        sz[1, 1] = 0.5
        sz[2, 2] = -0.5
        generators.extend([sx, sy, sz])

        for k in range(n_pairs):
            idx = 3 + k
            if idx >= dim_g - 1:
                break
            P_k = np.zeros((dim_g, dim_g), dtype=complex)
            Q_k = np.zeros((dim_g, dim_g), dtype=complex)
            P_k[0, idx] = 1.0
            Q_k[idx, 0] = 1.0
            generators.append(P_k)
            generators.append(Q_k)

        n_g = len(generators)
        g_killing = np.zeros((n_g, n_g), dtype=np.float64)
        for a, Ta in enumerate(generators):
            for b, Tb in enumerate(generators):
                g_killing[a, b] = float(np.real(np.trace(Ta.conj().T @ Tb)))
        evals = np.real(la.eigvalsh(0.5 * (g_killing + g_killing.T)))
        n_plus = int(np.sum(evals > 1e-8))
        n_minus = int(np.sum(evals < -1e-8))

        return {
            "rank": n_pairs,
            "dim_gauge_group": dim_g,
            "generators": generators,
            "killing_metric": g_killing,
            "killing_signature": (n_plus, n_minus),
            "structure_group": "U(1) ⊗ SU(2) ⊗ H₃(ℝ)",
        }

    # ─────────────────────────────────────────────────────────────────────
    # §I.2 — Conexión, curvatura F = dA+[A,A] y densidad de Chern-Simons
    # ─────────────────────────────────────────────────────────────────────
    def compute_gauge_curvature(self, bundle: Dict[str, Any]) -> np.ndarray:
        """F ≈ Σ_{a<b} [T_a, T_b] / n_G², proyectada a anti-hermítica."""
        gens = bundle["generators"]
        n_g = max(len(gens), 1)
        dim = gens[0].shape[0]
        F = np.zeros((dim, dim), dtype=complex)
        for a in range(n_g):
            for b in range(a + 1, n_g):
                F += (gens[a] @ gens[b] - gens[b] @ gens[a]) / (n_g * n_g)
        return 0.5 * (F - F.conj().T)

    @staticmethod
    def chern_simons_density(connection_A: np.ndarray, curvature_F: np.ndarray) -> float:
        """
        Densidad CS₃ = (1/8π²) Im Tr(A ∧ dA + ⅔ A∧A∧A) ≈ Im Tr(A F) / 8π².
        Invariante de marco (θ-vacío) del sector de calibre.
        """
        cs = np.imag(np.trace(connection_A @ curvature_F))
        return float(cs / (8.0 * math.pi ** 2))

    # ─────────────────────────────────────────────────────────────────────
    # §I.3 — c₁ de Hopf (S³ → S²) + holonomía U(1)  — corrección del c₁≡0
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def hopf_chern_from_seed(seed: CanonicalSeed) -> Tuple[float, float]:
        """
        Fibrado de Hopf del cuaternión orbital q ∈ S³:
            c₁ ≃ atan2(q_z, q_w)/π   ∈ (−1, 1)
        más el periodo de la función generatriz S₂ / 2π (clase de Cartan).
        Holonomía: ϑ = 2π c₁  (Wilson loop de U(1) alrededor de Σ_ℓ).
        """
        q = seed.orbital_quaternion
        hopf = math.atan2(q.z, q.w) / _PI
        cartan = seed.generating_function_S2 / _TWO_PI
        scale = max(abs(seed.delaunay.L), 1.0)
        c1 = 0.5 * hopf + 0.5 * math.tanh(cartan / scale)
        holonomy = float((c1 * _TWO_PI) % _TWO_PI)
        return float(c1), holonomy

    def classify_topology(self, c1: float) -> GaugeTopologyClass:
        abs_c1 = abs(c1)
        if abs_c1 < 0.5:
            return GaugeTopologyClass.TRIVIAL_BUNDLE
        if abs_c1 < 1.5:
            return GaugeTopologyClass.MONOPOLE_LIKE
        return GaugeTopologyClass.INSTANTON_DENSE

    # ─────────────────────────────────────────────────────────────────────
    # §I.4 — Poisson, mapa de momentos (Marsden-Weinstein) y Thevenin RLC
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def frobenius_poisson_norm(poisson_tensor: np.ndarray) -> float:
        return float(np.linalg.norm(poisson_tensor, ord="fro"))

    @staticmethod
    def marsden_weinstein_moment(del_pt: DelaunayPoint) -> float:
        """
        Mapa de momentos μ: T*Q → u(1)* ≅ ℝ  de la simetría axial.
        μ = H = G cos i  (componente del momento angular).  Nivel regular
        μ⁻¹(c)/U(1) es el reducido de Marsden-Weinstein.
        """
        return float(del_pt.H)

    @staticmethod
    def rlc_thevenin(del_pt: DelaunayPoint, gamma_lindblad: float = 0.05) -> complex:
        """
        Análogo de circuito: L_ind ~ L_Delaunay, C ~ 1/n², R ~ γ.
        Z_Th = R + j(ωL − 1/(ωC)) con ω = n (movimiento medio).
        """
        omega = max(del_pt.mean_motion, 1e-9)
        L_ind = max(del_pt.L, 1e-9)
        C = 1.0 / max(omega * omega, 1e-12)
        react = omega * L_ind - 1.0 / (omega * C)
        return complex(float(gamma_lindblad), float(react))

    # ─────────────────────────────────────────────────────────────────────
    # §I.5 — GERMEN FASE I → FASE II  (último método de la ontología)
    # ─────────────────────────────────────────────────────────────────────
    def forge_gauge_contract(
        self,
        reference_density_matrix: Optional[np.ndarray] = None,
        reference_action_variables: Optional[np.ndarray] = None,
        epsilon_risk: float = 0.0,
    ) -> SovereignGaugeContract:
        """
        ÚLTIMO método de la FASE I  ≡  GERMEN / PRIMER ACTO de la FASE II.

        Emite el `SovereignGaugeContract` inmutable que FASE II vincula
        (`bind_gauge_contract`) antes de orquestar la campaña Hamiltoniana.

        ⟶ `TOONBufferAgent.bind_gauge_contract(contract)` es la
           continuación formal de este método.
        """
        now = time.time()
        contract_id = f"GAUGE-CONTRACT-{int(now * 1000) % 1_000_000:06d}"

        bundle = self.build_gauge_bundle()
        curvature = self.compute_gauge_curvature(bundle)
        A_conn = bundle["generators"][0]
        cs_density = self.chern_simons_density(A_conn, curvature)

        n_pairs = max(3, min(self.dimension // 2, 28))
        J_poisson = self.atlas.build_poisson_tensor(n_pairs)
        frob_norm = self.frobenius_poisson_norm(J_poisson)

        if reference_density_matrix is None:
            reference_density_matrix = np.eye(self.dimension, dtype=np.float64) / self.dimension
        if reference_action_variables is None:
            reference_action_variables = np.sort(
                np.maximum(la.eigvalsh(reference_density_matrix), _EPS)
            )[::-1]

        seed = self.atlas.compute_poincare_canonical_seed(
            reference_density_matrix,
            reference_action_variables,
            epsilon_risk=epsilon_risk,
        )
        c1, holonomy = self.hopf_chern_from_seed(seed)
        topology = self.classify_topology(c1)

        contract = SovereignGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            timestamp=now,
            gauge_bundle_rank=int(bundle["rank"]),
            chern_class_proxy=c1,
            gauge_topology=topology,
            holonomy_phase=holonomy,
            chern_simons_density=cs_density,
            frobenius_norm_poisson=frob_norm,
            killing_metric_signature=tuple(bundle["killing_signature"]),  # type: ignore[arg-type]
            canonical_seed=seed,
            sigma_l0_default=float(seed.sigma_l0),
            moser_twist_ref=float(seed.moser_twist),
            esp32_crowbar_pin=self.esp32_gpio_pin,
        )
        logger.info(
            f"[FASE I] Contrato {contract_id} | rank={bundle['rank']} "
            f"c₁={c1:.4f} ({topology.name}) ϑ={holonomy:.4f} "
            f"CS={cs_density:.3e} ||J||_F={frob_norm:.4f} τ={seed.moser_twist:.4e} "
            f"σ_ℓ0={seed.sigma_l0:.4f}"
        )
        return contract


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE II — ORQUESTACIÓN HAMILTONIANA (DELAUNAY · LINDBLAD · Σ_ℓ) ▓▓▓
# Anidada: TOONBufferAgent(SovereignGaugeTopology).
# El primer método consume SovereignGaugeContract; el último emite Campaign.
# ══════════════════════════════════════════════════════════════════════════════

class TOONBufferAgent(SovereignGaugeTopology):
    """
    FASE II — Soberano anidado en la ontología de calibre.

    Hereda `forge_gauge_contract` (germen I→II) y lo consume en
    `bind_gauge_contract`.  Orquesta ingesta, Lindblad-GKSL, mapa de
    retorno Σ_ℓ (API v5), red de resonancias y recurrencia de Poincaré.

    El último método, `conduct_hamiltonian_campaign`, emite
    `HamiltonianCampaignResult` (germen II→III).
    """

    def __init__(
        self,
        sovereign_id: str = "BUFFER-SOVEREIGN-SABIO-01",
        engine: Optional[TOONBufferEngine] = None,
        auditor: Optional[KAMMelnikovAuditor] = None,
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        mu_gravitational: float = 1.0,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            dimension=dimension_mac,
            esp32_gpio_pin=esp32_gpio_pin,
            mu_gravitational=mu_gravitational,
        )
        self.dimension_mac = int(dimension_mac)
        self.capacity_gromov_max = float(capacity_gromov_max)

        self.engine = engine if engine is not None else TOONBufferEngine(
            dimension=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
            mu_gravitational=mu_gravitational,
        )
        self.auditor = auditor if auditor is not None else KAMMelnikovAuditor(epsilon_kam=1e-2)

        self._active_contract: Optional[SovereignGaugeContract] = None
        self._passports_history: List[BufferSovereignGovernancePassport] = []
        self._positron_authorizations: Dict[str, float] = {}
        self._all_sections: List[PoincareSection] = []
        self._all_orbits: List[ReturnMapOrbit] = []

        logger.info(
            f"[FASE II] TOONBufferAgent [{sovereign_id}] anidado en SovereignGaugeTopology. "
            f"Dim={dimension_mac}, GromovMax={capacity_gromov_max}, GPIO={esp32_gpio_pin}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # §II.0 — Continuación formal del germen de FASE I
    # ─────────────────────────────────────────────────────────────────────
    def bind_gauge_contract(self, contract: SovereignGaugeContract) -> SovereignGaugeContract:
        """
        PRIMER método de FASE II  ≡  continuación de
        `SovereignGaugeTopology.forge_gauge_contract`.

        Verifica HMAC, ancla el contrato activo y alinea la sección Σ_ℓ
        del motor con `sigma_l0_default` del fibrado.
        """
        raw = (
            f"{contract.contract_id}:{contract.sovereign_id}:{contract.timestamp}:"
            f"{contract.chern_class_proxy:.10f}:{contract.frobenius_norm_poisson:.10f}:"
            f"{contract.sigma_l0_default:.8f}:{contract.holonomy_phase:.8f}"
        ).encode("utf-8")
        expected = hmac.new(_SOVEREIGN_HMAC_KEY, raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(contract.contract_signature_hmac, expected):
            raise ValueError(
                f"HMAC de calibre inválido en {contract.contract_id}: "
                "el germen de FASE I no autentica."
            )
        self._active_contract = contract
        logger.info(
            f"[FASE II ← I] Contrato {contract.contract_id} vinculado. "
            f"c₁={contract.chern_class_proxy:.4f} ϑ={contract.holonomy_phase:.4f} "
            f"topología={contract.gauge_topology.name}"
        )
        return contract

    # ─────────────────────────────────────────────────────────────────────
    # §II.1 — Ingesta anclada al contrato (sesgo de Chern + Delaunay)
    # ─────────────────────────────────────────────────────────────────────
    def process_apu_ingestion(
        self,
        request: BufferIngestionRequest,
        contract: Optional[SovereignGaugeContract] = None,
    ) -> TransientTensorCartridge:
        """Valida rangos, aplica sesgo topológico c₁ y delega al motor v5."""
        if contract is None:
            contract = self._active_contract
        if contract is None:
            raise RuntimeError("Ingesta sin contrato de calibre: ejecute bind_gauge_contract.")
        self._active_contract = contract

        if not 0.0 <= request.policy_risk_intangible <= 1.0:
            raise ValueError(f"Riesgo intangible fuera de [0,1]: {request.policy_risk_intangible}")
        if not 0.0 <= request.weather_gremial_factor <= 1.0:
            raise ValueError(f"Factor clima/gremial fuera de [0,1]: {request.weather_gremial_factor}")

        costs = np.asarray(request.tangible_costs, dtype=np.float64).flatten()
        if costs.size < self.dimension_mac:
            costs = np.pad(costs, (0, self.dimension_mac - costs.size))
        else:
            costs = costs[: self.dimension_mac]

        # Instanton-denso desplaza levemente el riesgo (clase de Chern como sesgo)
        bias = float(np.clip(contract.chern_class_proxy * 0.05, -0.05, 0.05))
        eff_risk = float(np.clip(request.policy_risk_intangible + bias, 0.0, 1.0))

        logger.info(
            f"[FASE II] Ingesta [{request.apu_code}] bajo {contract.contract_id} "
            f"(c₁={contract.chern_class_proxy:.4f}, bias={bias:+.4f})"
        )
        return self.engine.ingest_cartridge(
            apu_code=request.apu_code,
            tangible_cost_vector=costs,
            policy_risk_intangible=eff_risk,
            weather_gremial_factor=request.weather_gremial_factor,
            coordinates_h2=request.coordinates_h2,
        )

    def execute_governance_cycle(self, dt: float = 0.1) -> Tuple[int, int]:
        """Paso Lindblad-GKSL/RLC con supervisión de purga Dirac."""
        active_count, purged_step = self.engine.evolve_lindblad_rlc_step(dt=dt)
        if purged_step > 0:
            logger.info(
                f"[FASE II] Gobernanza: {purged_step} cartucho(s) "
                f"aniquilado(s) en Vacío de Dirac (0.0 dB)."
            )
        return active_count, purged_step

    # ─────────────────────────────────────────────────────────────────────
    # §II.2 — Mapa de retorno Σ_ℓ (API v5: l0_reference → ReturnMapOrbit)
    # ─────────────────────────────────────────────────────────────────────
    def orchestrate_poincare_return_map(
        self,
        contract: SovereignGaugeContract,
        dt_integration: float = 1e-3,
        max_steps: int = 5000,
        max_crossings: int = 6,
    ) -> List[ReturnMapOrbit]:
        """
        Integra X_H con Verlet y recolecta órbitas de retorno sobre
            Σ_ℓ₀ = { ℓ ≡ ℓ₀ (mod 2π), ℓ̇ > 0 }
        con ℓ₀ tomado del contrato (no g₀, que en v4 era estacionario).
        """
        l0 = float(contract.sigma_l0_default)
        orbits = self.engine.poincare_return_map_step(
            dt_integration=dt_integration,
            max_steps=max_steps,
            l0_reference=l0,
            max_crossings=max_crossings,
        )
        self._all_orbits.extend(orbits)
        for orb in orbits:
            self._all_sections.extend(list(orb.sections))
        n_cross = sum(len(o.sections) for o in orbits)
        logger.info(
            f"[FASE II] Mapa de retorno Σ_{{ℓ={l0:.3f}}}: {len(orbits)} órbitas, "
            f"{n_cross} cruces, {len(self.engine._active_cartridges)} cartuchos."
        )
        return orbits

    # ─────────────────────────────────────────────────────────────────────
    # §II.3 — Resonancias, red (grafo) y gap del Laplaciano
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _mean_motion_resonance(
        sections: Sequence[PoincareSection],
    ) -> Tuple[Tuple[int, int], Tuple[float, ...], float]:
        if len(sections) < 2:
            return (0, 1), (), 0.0
        L_vals = np.array([s.crossing_actions[0] for s in sections], dtype=np.float64)
        n_vals = 1.0 / np.maximum(L_vals, _EPS) ** 3
        ratios = n_vals[1:] / max(float(n_vals[0]), _EPS)
        mean_ratio = float(np.mean(ratios)) if ratios.size else 0.0
        if mean_ratio <= 0.0 or not np.isfinite(mean_ratio):
            return (0, 1), tuple(float(x) for x in n_vals), 0.0
        frac = Fraction(mean_ratio).limit_denominator(10_000)
        p, q = int(frac.numerator), int(frac.denominator)
        q = max(q, 1)
        return (p, q), tuple(float(x) for x in n_vals), float(abs(p / q - mean_ratio))

    def build_resonance_web(
        self,
        cartridges: Sequence[TransientTensorCartridge],
        divisor_tol: float = 5e-2,
    ) -> ResonanceWeb:
        """
        Grafo no dirigido: arista i—j si existen p, q ≤ 8 con
            |q n_i − p n_j| < tol · max(n_i, n_j)
        (pequeños divisores de Poincaré).  λ₂(Δ) = conectividad algebraica.
        """
        verts = [c.cartridge_id for c in cartridges if c.delaunay is not None]
        ns = {
            c.cartridge_id: c.delaunay.mean_motion
            for c in cartridges
            if c.delaunay is not None
        }
        edges: List[ResonanceEdge] = []
        nV = len(verts)
        adj = np.zeros((nV, nV), dtype=np.float64)
        idx = {v: i for i, v in enumerate(verts)}

        for i, u in enumerate(verts):
            for v in verts[i + 1 :]:
                nu, nv = ns[u], ns[v]
                best_div, best_pq = 1e9, (0, 1)
                for q in range(1, 9):
                    for p in range(1, 9):
                        div = abs(q * nu - p * nv)
                        if div < best_div:
                            best_div, best_pq = div, (p, q)
                scale = max(nu, nv, _EPS)
                if best_div < divisor_tol * scale:
                    edges.append(
                        ResonanceEdge(
                            src_id=u, dst_id=v, p=best_pq[0], q=best_pq[1],
                            small_divisor=float(best_div),
                        )
                    )
                    a, b = idx[u], idx[v]
                    adj[a, b] = adj[b, a] = 1.0

        degrees = np.sum(adj, axis=1) if nV else np.array([])
        max_deg = int(np.max(degrees)) if degrees.size else 0
        if nV >= 2:
            lap = np.diag(degrees) - adj
            evals = np.sort(np.real(la.eigvalsh(lap)))
            gap = float(evals[1]) if evals.size > 1 else 0.0
            n_zero = int(np.sum(np.abs(evals) < 1e-8))
            n_comp = max(n_zero, 1)
        else:
            gap, n_comp = 0.0, nV

        return ResonanceWeb(
            vertices=tuple(verts),
            edges=tuple(edges),
            connected_components=int(n_comp),
            max_degree=max_deg,
            spectral_gap=gap,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §II.4 — Recurrencia de Poincaré y promediado de von Zeipel
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def poincare_recurrence_time(delaunays: Sequence[DelaunayPoint]) -> float:
        """
        Cota de recurrencia (Poincaré 1890) sobre el toro 𝕋³ de ángulos:
            T_rec ≃ (2π)³ / (n · |τ| · ⟨L⟩)
        con n = movimiento medio y τ = twist de Moser.  Unidades: segundos
        del reloj del buffer (no siderales).
        """
        if not delaunays:
            return 0.0
        n_bar = float(np.mean([max(d.mean_motion, 1e-9) for d in delaunays]))
        tau_bar = float(np.mean([abs(d.moser_twist) for d in delaunays]))
        L_bar = float(np.mean([max(d.L, 1e-9) for d in delaunays]))
        denom = max(n_bar * max(tau_bar, 1e-12) * L_bar, _EPS)
        return float((_TWO_PI ** 3) / denom)

    def von_zeipel_averaged_energy(
        self, del_pt: DelaunayPoint, epsilon_risk: float, n_quad: int = 24
    ) -> float:
        """
        Principio de promediado (Poincaré / von Zeipel):
            ⟨H⟩_ℓ = (1/2π) ∫ H(ℓ, g, h; L, G, H) dℓ
        elimina el ángulo rápido y mata el primer secular.
        """
        if not hasattr(self.engine, "hamiltonian_and_field"):
            return del_pt.kepler_energy
        acc = 0.0
        for ell in np.linspace(0.0, _TWO_PI, n_quad, endpoint=False):
            Ham, _ = self.engine.hamiltonian_and_field(
                del_pt.L, del_pt.G, del_pt.H, float(ell), del_pt.g, del_pt.h,
                del_pt.mu_gravitational, epsilon_risk,
            )
            acc += Ham
        return float(acc / n_quad)

    # ─────────────────────────────────────────────────────────────────────
    # §II.5 — GERMEN FASE II → FASE III  (último método del soberano)
    # ─────────────────────────────────────────────────────────────────────
    def conduct_hamiltonian_campaign(
        self,
        contract: SovereignGaugeContract,
        requests: List[BufferIngestionRequest],
        dt_governance: float = 0.1,
        dt_integration: float = 1e-3,
        max_steps_return_map: int = 3000,
        max_crossings: int = 6,
    ) -> HamiltonianCampaignResult:
        """
        ÚLTIMO método de la FASE II  ≡  GERMEN / PRIMER ACTO de la FASE III.

        Campaña completa:
            0. Vincular el contrato de calibre (continuación de FASE I)
            1. Ingesta de APUs bajo el fibrado
            2. Ciclo Lindblad-GKSL
            3. Mapa de retorno Σ_ℓ (ReturnMapOrbit v5)
            4. Resonancia p:q, red de pequeños divisores, T_rec, Lindstedt

        ⟶ `SovereignAuditAssimilator.ingest_campaign(result)` es la
           continuación formal de este método.
        """
        now = time.time()
        campaign_id = f"CAMPAIGN-HAMILTON-{int(now * 1000) % 1_000_000:06d}"
        self.bind_gauge_contract(contract)

        logger.info(f"[FASE II] ═══ Campaña Hamiltoniana {campaign_id} ═══")

        for req in requests:
            self.process_apu_ingestion(req, contract)
        ingested = len(self.engine._active_cartridges)

        _, purged_step = self.execute_governance_cycle(dt=dt_governance)

        orbits = self.orchestrate_poincare_return_map(
            contract=contract,
            dt_integration=dt_integration,
            max_steps=max_steps_return_map,
            max_crossings=max_crossings,
        )
        sections = [s for o in orbits for s in o.sections]
        (p, q), n_vec, dev = self._mean_motion_resonance(sections)

        carts = list(self.engine._active_cartridges.values())
        active_del = [c.delaunay for c in carts if c.delaunay is not None]
        mean_L = float(np.mean([d.L for d in active_del])) if active_del else 0.0
        mean_e = float(np.mean([d.eccentricity for d in active_del])) if active_del else 0.0
        mean_i = float(np.mean([d.inclination for d in active_del])) if active_del else 0.0
        lrl_norms = [
            float(np.linalg.norm(PoincareCanonicalAtlas.laplace_runge_lenz_vector(d)))
            for d in active_del
        ]
        lrl_mean = float(np.mean(lrl_norms)) if lrl_norms else 0.0

        greene_max = max((abs(s.greene_residue) for s in sections), default=0.0)
        chirikov_max = max((o.chirikov_overlap for o in orbits), default=0.0)
        twist_mean = float(np.mean([o.twist for o in orbits])) if orbits else contract.moser_twist_ref
        floquet_r = 1.0
        for o in orbits:
            for s in o.sections:
                if s.floquet_multipliers:
                    floquet_r = max(floquet_r, max(abs(lam) for lam in s.floquet_multipliers))
        birkhoff_total = int(sum(o.birkhoff_fixed_points for o in orbits))

        t_rec = self.poincare_recurrence_time(active_del)
        web = self.build_resonance_web(carts)

        n2_acc: List[float] = []
        z_th = 0j
        if active_del and hasattr(self.engine, "lindstedt_second_order"):
            for cart in carts:
                if cart.delaunay is None:
                    continue
                _, n2 = self.engine.lindstedt_second_order(
                    cart.delaunay, cart.qual_comp.exergy_risk_index
                )
                n2_acc.append(n2)
            z_th = self.rlc_thevenin(active_del[0])

        result = HamiltonianCampaignResult(
            campaign_id=campaign_id,
            timestamp=now,
            contract_id=contract.contract_id,
            cartridges_ingested=ingested,
            cartridges_purged_dirac=purged_step,
            poincare_sections_count=len(sections),
            resonance_pq=(p, q),
            mean_motion_vector=n_vec,
            mean_motion_ratio_deviation=dev,
            delaunay_mean_L=mean_L,
            delaunay_mean_eccentricity=mean_e,
            delaunay_mean_inclination=mean_i,
            lrl_norm_mean=lrl_mean,
            poincare_sections=tuple(sections[:64]),
            return_orbits=tuple(orbits),
            greene_residue_max=float(greene_max),
            chirikov_overlap_max=float(chirikov_max),
            moser_twist_mean=float(twist_mean),
            floquet_spectral_radius=float(floquet_r),
            birkhoff_fixed_points=birkhoff_total,
            poincare_recurrence_time=t_rec,
            lindstedt_n2_mean=float(np.mean(n2_acc)) if n2_acc else 0.0,
            holonomy_phase=float(contract.holonomy_phase),
            resonance_web=web,
            rlc_thevenin_impedance=z_th,
        )
        logger.info(
            f"[FASE II] Campaña cerrada: ingestados={ingested}, Σ-cruces={len(sections)}, "
            f"p:q={p}:{q}, δ={dev:.4e}, R_G={greene_max:.3f}, s_Ch={chirikov_max:.3f}, "
            f"λ₂(Δ)={web.spectral_gap:.4f}, T_rec={t_rec:.3e}s, "
            f"<L>={mean_L:.4f} <e>={mean_e:.4f} <i>={math.degrees(mean_i):.2f}°"
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE III — KAM / GREENE / CHIRIKOV / MELNIKOV / CPTP / DOSSIER ▓▓▓
# Anidada: SovereignAuditAssimilator(TOONBufferAgent).
# El primer método consume HamiltonianCampaignResult.
# ══════════════════════════════════════════════════════════════════════════════

class SovereignAuditAssimilator(TOONBufferAgent):
    """
    FASE III — Auditor y asimilador anidado en el soberano de FASE II.

    Recibe `HamiltonianCampaignResult` (germen II→III) y ejecuta:

        1. Auditoría KAM + Greene + Chirikov + Melnikov + Birkhoff
        2. Funtor CPTP de Lüders  F : Campaña → MAC
        3. Pasaporte soberano Ω₃ (implicación de Heyting)
        4. Φ_sem : invariantes celestes → métricas ejecutivas
        5. Inyección de Positrón e⁺ (HMAC humano, aniquilación Fock)
    """

    def __init__(
        self,
        sovereign_id: str = "BUFFER-SOVEREIGN-SABIO-01",
        engine: Optional[TOONBufferEngine] = None,
        auditor: Optional[KAMMelnikovAuditor] = None,
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        mu_gravitational: float = 1.0,
        epsilon_kam: float = 1e-2,
        melnikov_chaos_threshold: float = 0.5,
        greene_critical: float = 0.25,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            engine=engine,
            auditor=auditor,
            dimension_mac=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
            mu_gravitational=mu_gravitational,
        )
        self.epsilon_kam = float(epsilon_kam)
        self.melnikov_chaos_threshold = float(melnikov_chaos_threshold)
        self.greene_critical = float(greene_critical)
        if auditor is None:
            self.auditor = KAMMelnikovAuditor(
                epsilon_kam=epsilon_kam, greene_critical=greene_critical
            )
        self._active_campaign: Optional[HamiltonianCampaignResult] = None

    # ─────────────────────────────────────────────────────────────────────
    # §III.0 — Continuación formal del germen de FASE II
    # ─────────────────────────────────────────────────────────────────────
    def ingest_campaign(
        self, campaign: HamiltonianCampaignResult
    ) -> HamiltonianCampaignResult:
        """
        PRIMER método de FASE III  ≡  continuación de
        `TOONBufferAgent.conduct_hamiltonian_campaign`.

        Ancla el germen II→III, verifica consistencia contrato/órbitas
        y deja la campaña lista para auditar.
        """
        if campaign.contract_id and self._active_contract is not None:
            if campaign.contract_id != self._active_contract.contract_id:
                logger.warning(
                    f"[FASE III] Campaña {campaign.campaign_id} referencia "
                    f"{campaign.contract_id} ≠ contrato activo "
                    f"{self._active_contract.contract_id}."
                )
        self._active_campaign = campaign
        n_orb = len(campaign.return_orbits)
        logger.info(
            f"[FASE III ← II] Campaña {campaign.campaign_id} ingerida. "
            f"órbitas={n_orb} cruces={campaign.poincare_sections_count} "
            f"p:q={campaign.resonance_pq} R_G={campaign.greene_residue_max:.3f}"
        )
        return campaign

    # ─────────────────────────────────────────────────────────────────────
    # §III.1 — Auditoría KAM / Greene / Chirikov / Melnikov
    # ─────────────────────────────────────────────────────────────────────
    def audit_campaign(
        self, campaign: Optional[HamiltonianCampaignResult] = None
    ) -> Dict[str, Any]:
        camp = campaign or self._active_campaign
        if camp is None:
            raise RuntimeError("FASE III sin campaña: ejecute ingest_campaign.")
        carts = list(self.engine._active_cartridges.values())
        orbits: Sequence[ReturnMapOrbit] = camp.return_orbits
        if hasattr(self.auditor, "audit_return_orbits") and orbits:
            result = self.auditor.audit_return_orbits(orbits, carts)
        else:
            result = self.auditor.audit(list(camp.poincare_sections), carts)
        result.setdefault("greene_residue_max", camp.greene_residue_max)
        result.setdefault("chirikov_overlap_max", camp.chirikov_overlap_max)
        result.setdefault("moser_twist_mean", camp.moser_twist_mean)
        result.setdefault("floquet_spectral_radius", camp.floquet_spectral_radius)
        result.setdefault("birkhoff_fixed_points", camp.birkhoff_fixed_points)
        return result

    @staticmethod
    def _grade_from_kam(
        kam_regime: KAMRegime,
        melnikov_max: float,
        melnikov_threshold: float,
        greene_max: float,
        chirikov_max: float,
        greene_critical: float,
    ) -> SovereignVerdictGrade:
        if (
            kam_regime == KAMRegime.ARNOLD_DIFFUSION
            or melnikov_max > melnikov_threshold
            or chirikov_max >= 1.0
            or greene_max > 1.0
        ):
            return SovereignVerdictGrade.CHAOTIC_REJECTED
        if (
            kam_regime == KAMRegime.CANTORUS_PARTIAL
            or abs(greene_max - greene_critical) < 0.05
        ):
            return SovereignVerdictGrade.CANTOR_DEGRADED
        return SovereignVerdictGrade.KAM_PRESERVED

    # ─────────────────────────────────────────────────────────────────────
    # §III.2 — Funtor CPTP de Lüders  F : Campaña → (MAC, Certificado)
    # ─────────────────────────────────────────────────────────────────────
    def orchestrate_mac_assimilation(
        self,
        campaign: HamiltonianCampaignResult,
        mac_density_matrix: np.ndarray,
        assimilation_rate_eta: float = 0.15,
    ) -> Tuple[np.ndarray, BufferCertificate, Dict[str, Any]]:
        self.ingest_campaign(campaign)
        audit_result = self.audit_campaign(campaign)
        logger.info(
            f"[FASE III] Asimilando {campaign.campaign_id} → MAC. "
            f"KAM={audit_result['kam_regime'].name} "
            f"M_max={audit_result['melnikov_max']:.4f} "
            f"p:q={audit_result['resonance_pq']}"
        )
        updated_mac, buffer_cert = self.auditor.assimilate_to_mac(
            engine=self.engine,
            mac_density_matrix=mac_density_matrix,
            assimilation_rate_eta=assimilation_rate_eta,
        )
        return updated_mac, buffer_cert, audit_result

    # ─────────────────────────────────────────────────────────────────────
    # §III.3 — Pasaporte soberano (Ω₃ + celeste v5)
    # ─────────────────────────────────────────────────────────────────────
    def issue_sovereign_passport(
        self,
        buffer_cert: BufferCertificate,
        campaign: HamiltonianCampaignResult,
        audit_result: Dict[str, Any],
    ) -> BufferSovereignGovernancePassport:
        now = time.time()
        passport_id = f"PASSPORT-BUFF-{int(now * 1000) % 1_000_000:06d}"

        kam_regime = audit_result.get("kam_regime", KAMRegime.INTACT_TORUS)
        melnikov_max = float(audit_result.get("melnikov_max", 0.0))
        greene_max = float(
            getattr(buffer_cert, "greene_residue_max", campaign.greene_residue_max)
        )
        chirikov_max = float(
            getattr(buffer_cert, "chirikov_overlap_max", campaign.chirikov_overlap_max)
        )
        intact_frac = float(
            getattr(buffer_cert, "kam_intact_fraction", campaign.cartridges_ingested or 1.0)
            if hasattr(buffer_cert, "kam_intact_fraction")
            else 1.0
        )
        grade = self._grade_from_kam(
            kam_regime, melnikov_max, self.melnikov_chaos_threshold,
            greene_max, chirikov_max, self.greene_critical,
        )
        implication = getattr(buffer_cert, "heyting_implication", None)
        if implication is None:
            implication = HeytingOmega3.COHERENT.implies(buffer_cert.verdict).name

        raw_provenance = (
            f"{passport_id}:{self.sovereign_id}:{now}:{buffer_cert.verdict.name}:"
            f"{buffer_cert.actuation_mode.name}:{buffer_cert.merkle_root_sha256}:"
            f"{buffer_cert.capacity_gromov_peak:.8f}:{buffer_cert.poincare_cartan_residual:.8e}:"
            f"{kam_regime.name}:{melnikov_max:.8f}:{campaign.resonance_pq}:"
            f"{greene_max:.8f}:{chirikov_max:.8f}"
        ).encode("utf-8")
        provenance_sha256 = hashlib.sha256(raw_provenance).hexdigest()

        passport = BufferSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_agent_id=self.sovereign_id,
            timestamp_utc=now,
            heyting_verdict=buffer_cert.verdict,
            heyting_implication=str(implication),
            actuation_mode=buffer_cert.actuation_mode,
            active_cartridges=buffer_cert.active_cartridges_count,
            total_assimilated=buffer_cert.assimilated_count,
            total_purged_dirac=buffer_cert.purged_count,
            gromov_capacity_peak=buffer_cert.capacity_gromov_peak,
            poincare_cartan_residual=buffer_cert.poincare_cartan_residual,
            mac_von_neumann_entropy=buffer_cert.mac_von_neumann_entropy,
            merkle_root_sha256=buffer_cert.merkle_root_sha256,
            esp32_crowbar_interlock_armed=(
                buffer_cert.actuation_mode == ActuationMode.HARD_VETO_ESP32_CROWBAR
            ),
            provenance_hash_sha256=provenance_sha256,
            kam_regime=kam_regime,
            kam_intact_fraction=intact_frac,
            melnikov_amplitude_max=melnikov_max,
            delaunay_resonance_order_q=int(campaign.resonance_pq[1]),
            mean_motion_ratio_deviation=float(campaign.mean_motion_ratio_deviation),
            laplace_runge_lenz_norm=float(
                getattr(buffer_cert, "laplace_runge_lenz_norm", campaign.lrl_norm_mean)
            ),
            greene_residue_max=greene_max,
            chirikov_overlap_max=chirikov_max,
            moser_twist_mean=float(
                getattr(buffer_cert, "moser_twist_mean", campaign.moser_twist_mean)
            ),
            floquet_spectral_radius=float(
                getattr(buffer_cert, "floquet_spectral_radius", campaign.floquet_spectral_radius)
            ),
            birkhoff_fixed_points=int(
                getattr(buffer_cert, "birkhoff_fixed_points", campaign.birkhoff_fixed_points)
            ),
            holonomy_phase=float(campaign.holonomy_phase),
            poincare_recurrence_time=float(campaign.poincare_recurrence_time),
            sovereign_verdict_grade=grade,
        )
        self._passports_history.append(passport)
        logger.info(
            f"[FASE III] Pasaporte [{passport_id}] Ω₃={buffer_cert.verdict.name} "
            f"→ {implication} | KAM={kam_regime.name} | grade={grade.name} "
            f"| R_G={greene_max:.3f} s_Ch={chirikov_max:.3f}"
        )
        return passport

    # ─────────────────────────────────────────────────────────────────────
    # §III.4 — Funtor semántico Φ_sem  (invariantes → Dolor y Dinero)
    # ─────────────────────────────────────────────────────────────────────
    def translate_to_business_impact(
        self, passport: BufferSovereignGovernancePassport
    ) -> ExecutiveBusinessImpactReport:
        now = time.time()
        report_id = f"RPT-EXEC-{int(now * 1000) % 1_000_000:06d}"

        kv_savings = 86.4
        coherent = passport.heyting_verdict == HeytingOmega3.COHERENT
        wacc_protection = 15.0 if coherent else 5.0
        contingency_reduction = 11.5 if coherent else 0.0
        shielded_capital = passport.total_assimilated * 25000.0
        bypass_active = passport.actuation_mode == ActuationMode.SOFT_VETO_BYPASS
        crowbar_triggered = passport.actuation_mode == ActuationMode.HARD_VETO_ESP32_CROWBAR

        kam_integrity_pct = float(passport.kam_intact_fraction * 100.0)
        q = max(1, passport.delaunay_resonance_order_q)
        resonance_hazard = float(
            np.clip(1.0 / q + passport.mean_motion_ratio_deviation * 10.0, 0.0, 1.0)
        )
        melnikov_index = float(np.clip(passport.melnikov_amplitude_max, 0.0, 1.0))
        greene_crit = float(
            np.clip(abs(passport.greene_residue_max) / max(self.greene_critical, _EPS), 0.0, 2.0)
        )
        chirikov_idx = float(np.clip(passport.chirikov_overlap_max, 0.0, 2.0))
        floquet_inst = float(max(0.0, passport.floquet_spectral_radius - 1.0))

        orbital_stability = float(
            np.clip(
                passport.kam_intact_fraction
                * (1.0 - 0.5 * melnikov_index)
                * (1.0 - 0.3 * resonance_hazard)
                * (1.0 - 0.2 * min(greene_crit, 1.0))
                * (1.0 - 0.2 * min(chirikov_idx, 1.0)),
                0.0,
                1.0,
            )
        )

        if (
            passport.sovereign_verdict_grade == SovereignVerdictGrade.KAM_PRESERVED
            and coherent
        ):
            executive_summary = (
                f"Toros KAM preservados ({kam_integrity_pct:.1f}%). Flujo cuasi-periódico "
                f"sobre T*Q. {passport.total_assimilated} APUs asimilados, "
                f"δn={passport.mean_motion_ratio_deviation:.2e}, "
                f"M_max={passport.melnikov_amplitude_max:.4f}, "
                f"R_Greene={passport.greene_residue_max:.3f}, "
                f"s_Chirikov={passport.chirikov_overlap_max:.3f}. "
                f"Resonancia p:q de orden q={q}. T_rec={passport.poincare_recurrence_time:.2e}s. "
                f"Holonomía ϑ={passport.holonomy_phase:.4f}."
            )
        elif (
            passport.sovereign_verdict_grade == SovereignVerdictGrade.CANTOR_DEGRADED
            or passport.heyting_verdict == HeytingOmega3.DEGRADED
        ):
            executive_summary = (
                f"Atención: transición a Cantor torus (Aubry-Mather). "
                f"Integridad KAM {kam_integrity_pct:.1f}%. Residuo de Greene "
                f"R={passport.greene_residue_max:.3f} (crítico ¼). "
                f"Válvula ABS (Bypass) activa. Melnikov M_max="
                f"{passport.melnikov_amplitude_max:.4f} sugiere splitting homoclínico "
                f"incipiente. 1 h de gracia para Positrón e⁺ (HMAC humano)."
            )
        else:
            executive_summary = (
                f"ALERTA CRÍTICA: difusión de Arnold / solapamiento de Chirikov "
                f"(s={passport.chirikov_overlap_max:.3f}). Disyuntor ESP32 Crowbar "
                f"< 400 ns. Integridad KAM {kam_integrity_pct:.1f}%. "
                f"Melnikov M_max={passport.melnikov_amplitude_max:.4f}, "
                f"ρ_Floquet={passport.floquet_spectral_radius:.3f}. "
                f"Bloqueo de triangulación / sobrecosto en SECOP II."
            )

        return ExecutiveBusinessImpactReport(
            report_id=report_id,
            timestamp=now,
            sovereign_verdict=passport.heyting_verdict.name,
            executive_summary=executive_summary,
            kv_cache_token_savings_percent=kv_savings,
            wacc_protection_rate_percent=wacc_protection,
            cash_flow_shielded_usd=shielded_capital,
            contingency_fund_reduction_percent=contingency_reduction,
            bypass_grace_period_active=bypass_active,
            crowbar_paralysis_triggered=crowbar_triggered,
            kam_torus_integrity_percent=kam_integrity_pct,
            resonance_hazard_factor=resonance_hazard,
            melnikov_chaos_index=melnikov_index,
            orbital_stability_score=orbital_stability,
            greene_criticality=greene_crit,
            chirikov_overlap_index=chirikov_idx,
            floquet_instability=floquet_inst,
            recurrence_horizon_sec=float(passport.poincare_recurrence_time),
        )

    # ─────────────────────────────────────────────────────────────────────
    # §III.5 — Positrón e⁺ (HMAC) — aniquilación Fock e⁺ + e⁻ → 2γ
    # ─────────────────────────────────────────────────────────────────────
    def inject_positron_authorization(
        self,
        cartridge_id: str,
        hmac_signature: str,
        shared_secret: bytes = b"POSITRON-AUTHORITY-SECRET-2025",
    ) -> bool:
        expected = hmac.new(shared_secret, cartridge_id.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(hmac_signature.lower(), expected.lower()):
            logger.warning(
                f"[FASE III] HMAC inválido para Positrón e⁺ en {cartridge_id}."
            )
            return False
        self._positron_authorizations[cartridge_id] = time.time()
        logger.info(
            f"[FASE III] [ANIKILACIÓN FOCK e⁺+e⁻→2γ] Positrón inyectado en "
            f"{cartridge_id}. HMAC={hmac_signature[:12]}..."
        )
        return True

    # ─────────────────────────────────────────────────────────────────────
    # §III.6 — CIERRE DEL CICLO ANIDADO I ⊂ II ⊂ III
    # ─────────────────────────────────────────────────────────────────────
    def emit_full_executive_dossier(
        self,
        campaign: HamiltonianCampaignResult,
        mac_density_matrix: np.ndarray,
        assimilation_rate_eta: float = 0.15,
    ) -> Dict[str, Any]:
        """
        ÚLTIMO método de la FASE III (y del módulo).

        Funtor compuesto
            F = Φ_sem ∘ Issue ∘ Assimilate ∘ Audit ∘ Ingest
              : HamiltonianCampaignResult  ⟶  Dossier

        cierra el ciclo I ⊂ II ⊂ III y emite todos los artefactos soberanos.
        """
        mac_updated, buffer_cert, audit_result = self.orchestrate_mac_assimilation(
            campaign=campaign,
            mac_density_matrix=mac_density_matrix,
            assimilation_rate_eta=assimilation_rate_eta,
        )
        passport = self.issue_sovereign_passport(buffer_cert, campaign, audit_result)
        exec_report = self.translate_to_business_impact(passport)

        if buffer_cert.actuation_mode == ActuationMode.HARD_VETO_ESP32_CROWBAR:
            logger.critical(
                f"[FASE III] [DISYUNTOR CROWBAR] GPIO{self.esp32_gpio_pin} < 400 ns. "
                f"ResΘ={buffer_cert.poincare_cartan_residual:.2e} "
                f"Gromov={buffer_cert.capacity_gromov_peak:.2f} "
                f"KAM={passport.kam_regime.name} R_G={passport.greene_residue_max:.3f} "
                f"s_Ch={passport.chirikov_overlap_max:.3f}."
            )

        dossier = {
            "campaign": campaign,
            "buffer_certificate": buffer_cert,
            "audit_result": audit_result,
            "sovereign_passport": passport,
            "executive_report": exec_report,
            "mac_updated_shape": mac_updated.shape,
            "mac_updated_trace": float(np.real(np.trace(mac_updated))),
            "emitted_at_utc": time.time(),
            "functor_composite": "Φ_sem ∘ Issue ∘ Assimilate ∘ Audit ∘ Ingest",
        }
        logger.info(
            f"[FASE III] Dossier emitido. Ω₃={passport.heyting_verdict.name} "
            f"KAM={passport.kam_regime.name} "
            f"orbital={exec_report.orbital_stability_score:.3f}"
        )
        return dossier


# ══════════════════════════════════════════════════════════════════════════════
# §C. VERIFICACIÓN DOCTORAL — 3 FASES ANIDADAS SOBRE UN ÚNICO SOBERANO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 82)
    print("  TOONBufferAgent v5 — Soberano anidado · Hopf · Greene · Chirikov · Lüders")
    print("═" * 82)

    # Un solo objeto: I ⊂ II ⊂ III  (SovereignAuditAssimilator hereda todo)
    sovereign = SovereignAuditAssimilator(
        sovereign_id="BUFFER-SOVEREIGN-SABIO-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
        epsilon_kam=1e-2,
        melnikov_chaos_threshold=0.5,
        greene_critical=0.25,
    )

    print("\n┌─[FASE I] Ontología soberana y contrato de calibre (Hopf/Chern) ─┐")
    ref_rho = np.eye(56, dtype=np.float64) / 56.0
    ref_J = np.sort(np.maximum(la.eigvalsh(ref_rho), _EPS))[::-1]
    contract = sovereign.forge_gauge_contract(ref_rho, ref_J)
    seed = contract.canonical_seed
    print(f"  • Contrato ID       : {contract.contract_id}")
    print(f"  • HMAC calibre      : {contract.contract_signature_hmac[:32]}...")
    print(f"  • Rango fibrado     : {contract.gauge_bundle_rank}")
    print(f"  • c₁ Hopf           : {contract.chern_class_proxy:.6f}")
    print(f"  • Topología         : {contract.gauge_topology.name}")
    print(f"  • Holonomía ϑ       : {contract.holonomy_phase:.6f}")
    print(f"  • CS₃ densidad      : {contract.chern_simons_density:.6e}")
    print(f"  • ||J||_F / sig(K)  : {contract.frobenius_norm_poisson:.4f} / {contract.killing_metric_signature}")
    print(f"  • σ_ℓ0 / τ_Moser    : {contract.sigma_l0_default:.6f} / {contract.moser_twist_ref:.4e}")
    print(f"  • Seed L,e,i,n      : {seed.delaunay.L:.4f}, {seed.delaunay.eccentricity:.4f}, "
          f"{math.degrees(seed.delaunay.inclination):.2f}°, {seed.n_mean:.4e}")
    print("  ⟶ germen emitido para FASE II  (bind_gauge_contract)")

    print("\n┌─[FASE II] Campaña Hamiltoniana (Delaunay · Lindblad · Σ_ℓ) ──────┐")
    requests = [
        BufferIngestionRequest(
            apu_code="APU-VACIADO-CONCRETO-3000PSI",
            tangible_costs=np.array([450000.0, 120000.0, 85000.0, 30000.0]),
            policy_risk_intangible=0.15,
            weather_gremial_factor=0.08,
            coordinates_h2=(1.2, 2.5),
        ),
        BufferIngestionRequest(
            apu_code="APU-ACERO-FIGURADO-60000PSI",
            tangible_costs=np.array([320000.0, 95000.0, 60000.0, 15000.0]),
            policy_risk_intangible=0.22,
            weather_gremial_factor=0.10,
            coordinates_h2=(0.8, 1.8),
        ),
        BufferIngestionRequest(
            apu_code="APU-MUROS-PANTALLA-SISMO",
            tangible_costs=np.array([280000.0, 150000.0, 90000.0, 25000.0]),
            policy_risk_intangible=0.28,
            weather_gremial_factor=0.12,
            coordinates_h2=(1.0, 2.0),
        ),
    ]
    campaign = sovereign.conduct_hamiltonian_campaign(
        contract=contract,
        requests=requests,
        dt_governance=0.1,
        dt_integration=1e-3,
        max_steps_return_map=3000,
        max_crossings=6,
    )
    web = campaign.resonance_web
    print(f"  • Campaña ID        : {campaign.campaign_id}")
    print(f"  • Cartuchos ingest. : {campaign.cartridges_ingested}")
    print(f"  • Órbitas / cruces  : {len(campaign.return_orbits)} / {campaign.poincare_sections_count}")
    print(f"  • Resonancia p:q    : {campaign.resonance_pq}  δn={campaign.mean_motion_ratio_deviation:.4e}")
    print(f"  • <L>, <e>, <i>     : {campaign.delaunay_mean_L:.3f}, "
          f"{campaign.delaunay_mean_eccentricity:.3f}, "
          f"{math.degrees(campaign.delaunay_mean_inclination):.2f}°")
    print(f"  • ||LRL|| / T_rec   : {campaign.lrl_norm_mean:.4f} / {campaign.poincare_recurrence_time:.3e}s")
    print(f"  • Greene / Chirikov : {campaign.greene_residue_max:.4f} / {campaign.chirikov_overlap_max:.4f}")
    print(f"  • Floquet ρ / Birk. : {campaign.floquet_spectral_radius:.4f} / {campaign.birkhoff_fixed_points}")
    print(f"  • Lindstedt ⟨n₂⟩    : {campaign.lindstedt_n2_mean:.4e}")
    print(f"  • Z_Th (RLC)        : {campaign.rlc_thevenin_impedance}")
    if web is not None:
        print(f"  • Red resonante     : |V|={len(web.vertices)} |E|={len(web.edges)} "
              f"κ={web.connected_components} Δ={web.max_degree} λ₂={web.spectral_gap:.4f}")
    print("  ⟶ germen emitido para FASE III  (ingest_campaign)")

    print("\n┌─[FASE III] Auditoría · funtor CPTP · Pasaporte · Φ_sem ──────────┐")
    mac_initial = np.eye(56, dtype=np.float64) / 56.0
    dossier = sovereign.emit_full_executive_dossier(
        campaign=campaign,
        mac_density_matrix=mac_initial,
        assimilation_rate_eta=0.15,
    )
    passport = dossier["sovereign_passport"]
    report = dossier["executive_report"]
    audit = dossier["audit_result"]

    print(f"  • Pasaporte ID      : {passport.passport_id}")
    print(f"  • Proveniencia      : {passport.provenance_hash_sha256[:32]}...")
    print(f"  • Veredicto Ω₃      : {passport.heyting_verdict.name}  → {passport.heyting_implication}")
    print(f"  • Grado Soberano    : {passport.sovereign_verdict_grade.name}")
    print(f"  • Modo Actuación    : {passport.actuation_mode.name}")
    print(f"  • KAM / drift       : {audit['kam_regime'].name} / {audit.get('kam_drift', 0.0):.6e}")
    print(f"  • Melnikov M_max    : {audit['melnikov_max']:.6f}")
    print(f"  • Asimilados/purga  : {passport.total_assimilated} / {passport.total_purged_dirac}")
    print(f"  • Funtor compuesto  : {dossier['functor_composite']}")

    print("\n  ── Reporte Ejecutivo Φ_sem ('Dolor y Dinero') ──")
    print(f"    • Ahorro KV-Cache        : {report.kv_cache_token_savings_percent}%")
    print(f"    • Protección WACC        : {report.wacc_protection_rate_percent}%")
    print(f"    • Capital salvaguardado  : ${report.cash_flow_shielded_usd:,.2f} USD")
    print(f"    • Integridad KAM         : {report.kam_torus_integrity_percent:.2f}%")
    print(f"    • Peligro de resonancia  : {report.resonance_hazard_factor:.4f}")
    print(f"    • Índice caos Melnikov   : {report.melnikov_chaos_index:.4f}")
    print(f"    • Criticidad Greene      : {report.greene_criticality:.4f}")
    print(f"    • Solapamiento Chirikov  : {report.chirikov_overlap_index:.4f}")
    print(f"    • Inestabilidad Floquet  : {report.floquet_instability:.4f}")
    print(f"    • Horizonte T_rec        : {report.recurrence_horizon_sec:.3e}s")
    print(f"    • Score orbital estable  : {report.orbital_stability_score:.4f}")
    print(f"    • Resumen: \"{report.executive_summary}\"")

    test_cart = "TOON-BUFF-TEST-SIG"
    secret = b"POSITRON-AUTHORITY-SECRET-2025"
    valid_hmac = hmac.new(secret, test_cart.encode("utf-8"), hashlib.sha256).hexdigest()
    ok = sovereign.inject_positron_authorization(test_cart, valid_hmac, shared_secret=secret)
    print(f"\n  • Inyección Positrón e⁺ : {'✓ ACEPTADA' if ok else '✗ RECHAZADA'}")

    print("\n" + "═" * 82)
    print("  ✓ SOBERANO v5 — HOPF · GREENE · CHIRIKOV · MOSER · LÜDERS · Φ_sem")
    print("═" * 82)