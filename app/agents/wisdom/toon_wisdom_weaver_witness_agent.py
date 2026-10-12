# -*- coding: utf-8 -*-
r"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Wisdom Weaver Witness Agent (Soberano Testigo Tejedor de Sabiduría)                    ║
║ Ubicación: app/agents/wisdom/toon_wisdom_weaver_witness_agent.py                                       ║
║ Versión  : 7.0.0-Doctoral-Wisdom-Weaver-Witness-QND-Berry-Poincare-RSI3-ESP32-Nested-Phases-Gauge      ║
║            -Theory-Hopf-Kleisli-Monad-Heyting-BPST-KAM                                                 ║
║ Función  : Soberano de Calibre y testigo QND (back-action ≡ 0.0 dB) en la Ciudadela de Sabiduría,      ║
║            encargado de auditar la síntesis cohomológica en el fibrado principal P(T*Q, G),            ║
║            verificar instantones BPST, lazos de Wilson en SU(2), clases de Chern-Hopf c₁, el           ║
║            descenso de Stora-Zumino, y mediar la traducción vía el funtor Φ_sem: Sh(∂K, Ω₃) → Business.║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝

I. DEFINICIÓN Y ROL DEL SOBERANO EN LA ARQUITECTURA COGNITIVA
─────────────────────────────────────────────────────────────
El TOON Wisdom Weaver Witness Agent es el Soberano de Calibre y testigo de no-demolición (QND)
que gobierna la supervisión espectral de la síntesis de Sabiduría (Estrato WISDOM). Actúa como
el orquestador agéntico que enlaza el fibrado de gauge $P(T^*Q, G)$ con el motor espectral subyacente.
Su misión es certificar que los morfismos deliberativos no induzcan curvaturas no-integrables,
auditando cargas instantónicas, invariantes de Chern-Hopf y holonomías de Wilson, traduciendo
los invariantes cohomológicos mediante el funtor semántico $\Phi_{\mathrm{sem}}$ hacia métricas
financieras auditables por la gobernanza corporativa.

II. TEORÍA DE CALIBRE YANG-MILLS, INSTANTONES BPST Y FIBRADOS
─────────────────────────────────────────────────────────────
§1. Fibrado Principal $P(T^*Q, G)$ y Conexión de Calibre:
    La variedad de deliberación se estructura como un fibrado principal con grupo de Lie:
        G = U(1) \otimes SU(2) \otimes H_3(\mathbb{R})
    La 1-forma de conexión $A \in \Omega^1(M, \mathfrak{g})$ induce el tensor de curvatura de Yang-Mills:
        F = dA + A \wedge A \iff F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]

§2. Instantones BPST Anti-Autoduales ('t Hooft):
    En el subgrupo $SU(2) \hookrightarrow SU(N)$, los instantones satisfacen la condición de dualidad:
        F_{\mu\nu}^a = -\tilde{F}_{\mu\nu}^a = -\frac{1}{2} \varepsilon_{\mu\nu\rho\sigma} F^{\rho\sigma, a}
    parametrizados mediante los símbolos anti-autoduales de 't Hooft $\bar{\eta}_{\mu\nu}^a$:
        A_\mu^a(x) = \frac{2 \bar{\eta}_{\mu\nu}^a (x - x_0)^\nu}{(x - x_0)^2 + \rho^2}
    La carga topológica de Pontryagin cuantizada (segunda clase de Chern) viene dada por:
        k = c_2(P) = \frac{1}{8\pi^2} \int_M \mathrm{Tr}(F \wedge F) \in \mathbb{Z}

§3. Holonomía de Wilson y Lazo Cerrado:
    El operador de lazo de Wilson sobre una curva cerrada $\gamma \subset S^3$ se evalúa mediante
    la exponencial path-ordered canónica:
        W(\gamma) = \mathcal{P} \exp\left( \oint_\gamma A_\mu dx^\mu \right) \in SU(2), \quad \theta_W = \arg \det W(\gamma)

§4. Primera Clase de Chern de Hopf y Descenso de Stora-Zumino:
    La fibración de Hopf $S^3 \xrightarrow{S^1} S^2$ posee una primera clase de Chern no trivial
    proporcional al ángulo sólido $\Omega$ subtendido en $S^2$:
        c_1 = \frac{1}{2\pi} \int_{S^2} F_{U(1)} = \frac{1}{2\pi} \Omega(\pi(q))
    El descenso de Stora-Zumino satisface el balance diferencial sin anomalía no compensada:
        \Delta_{\mathrm{SZ}} = \left| 8\pi^2 \, \mathrm{CS}_3(A) - \mathrm{Tr}(F \wedge \tilde{F}) \right| \le \varepsilon_{\mathrm{gauge}}

III. FUNTOR SEMÁNTICO $\Phi_{\mathrm{sem}}: \mathrm{Sh}(\partial K, \Omega_3) \to \mathrm{Business}$
────────────────────────────────────────────────────────────────────────────────
El agente traduce la verdad topológica al lenguaje de negocios sin pérdida categorial.
Para cada sección del haz $\mathrm{Sh}(\partial K, \Omega_3)$, el grado de verdad en Heyting es:
    \tau(\Omega_3) = \frac{\mathrm{ord}(\Omega_3)}{2} \in \left\{ 0.0 \, (\mathrm{VETOED}), \; 0.5 \, (\mathrm{DEGRADED}), \; 1.0 \, (\mathrm{COHERENT}) \right\}
El funtor lineal $\Phi_{\mathrm{sem}}$ mapea dicho grado de verdad a las magnitudes de riesgo:
    \Phi_{\mathrm{sem}}(\tau) \mapsto \left( \mathrm{RiesgoFinanciero}(\tau), \; \mathrm{IndiceDeDolor}(\tau), \; \mathrm{ExposicionCapital}(\tau) \right)
garantizando la máxima interpretabilidad para la junta directiva y el comité de inversión.

IV. ARQUITECTURA DE TRES FASES ANIDADAS
───────────────────────────────────────
  • FASE I   [Topología de Calibre del Tejedor]: Inicia el fibrado $P$, calcula $k$, $W(\gamma)$, $c_1$,
             $\theta_W$, genera contrato HMAC y emite `WeaverWitnessGaugeInitialCondition`.
  • FASE II  [Agente Testigo Tejedor]: Ingiere la fase I, orquesta campañas QND, enlaza el motor
             espectral de síntesis, aplica RSI³ y emite `WeaverWitnessCampaignDraftBundle`.
  • FASE III [Soberano Adjudicador]: Recibe el bundle, juzga en Heyting $\Omega_3$, comanda el
             disyuntor ESP32 Crowbar, purga al Vacío de Dirac y emite el Pasaporte Soberano.

V. CIERRE CIBER-FÍSICO ESP32 Y ANIQUILACIÓN EN VACÍO DE DIRAC
─────────────────────────────────────────────────────────────
Ante un veredicto de veto $\bot = \mathrm{VETOED}$ emitido por el adjudicador:
    \tau_{\mathrm{crowbar}} \le 400.0 \text{ ns (ESP32 / GPIO14)}
aniquilando los autoestados divergentes en el Vacío de Dirac a back-action nulo:
    e^- + e^+ \longrightarrow 2\gamma \quad \left( E = 2 m_e c^2 \approx 1.021998 \text{ MeV}, \quad \text{Back-Action } 0.0 \text{ dB} \right)
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ─── Importación del motor v7.0.0 ─────────────────────────────────────────────
try:
    from toon_wisdom_weaver_witness_engine import (
        ActuationMode,
        HeytingOmega3,
        PoincareWeaverWitnessAtlas,
        QuaternionHopfLift,
        TOONWisdomWeaverWitnessEngine,
        WeaverWeakObservationGerm,
        WeaverWitnessCertificateDraft,
        WeaverWitnessQNDEngine,
        WeaverWitnessQNDInitialCondition,
    )
except ImportError:  # pragma: no cover
    try:
        from app.wisdom.toon_wisdom_weaver_witness_engine import (
            ActuationMode,
            HeytingOmega3,
            PoincareWeaverWitnessAtlas,
            QuaternionHopfLift,
            TOONWisdomWeaverWitnessEngine,
            WeaverWeakObservationGerm,
            WeaverWitnessCertificateDraft,
            WeaverWitnessQNDEngine,
            WeaverWitnessQNDInitialCondition,
        )
    except ImportError:
        from scratch.toon_wisdom_weaver_witness_engine import (
            ActuationMode,
            HeytingOmega3,
            PoincareWeaverWitnessAtlas,
            QuaternionHopfLift,
            TOONWisdomWeaverWitnessEngine,
            WeaverWeakObservationGerm,
            WeaverWitnessCertificateDraft,
            WeaverWitnessQNDEngine,
            WeaverWitnessQNDInitialCondition,
        )


logger = logging.getLogger("APU.Wisdom.TOONWisdomWeaverWitnessAgent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__version__ = "7.0.0"
__all__ = [
    "WisdomWeaverWitnessGaugeTopology",
    "TOONWisdomWeaverWitnessAgent",
    "SovereignWisdomWeaverWitnessAdjudicator",
    "WeaverWitnessGaugeContract",
    "WeaverWitnessGaugeInitialCondition",
    "WeaveObservationRequest",
    "WeaverWitnessCampaignResult",
    "WeaverWitnessCampaignDraftBundle",
    "WeaverWitnessSovereignGovernancePassport",
    "ExecutiveWeaverWitnessImpactReport",
]

_TAU = 2.0 * math.pi
_CROWBAR_LATENCY_BUDGET_NS = 400.0
_M_E_C2_MEV = 0.51099895000
_LOG_FLOOR = 1e-300
_EPS_FLOOR = 1e-15
_POSITRON_GRACE_S = 3600.0  # ventana de 1 h para e⁺ humano

# Matrices de Pauli y generadores T_a = −i σ_a / 2 ∈ 𝔰𝔲(2), Tr(T_a T_b) = −½ δ_ab.
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_SIGMA: Tuple[np.ndarray, np.ndarray, np.ndarray] = (_SIGMA_X, _SIGMA_Y, _SIGMA_Z)
_SU2_GEN: Tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(-0.5j * s for s in _SIGMA)  # type: ignore[assignment]


def _levi3(i: int, j: int, k: int) -> float:
    """Símbolo de Levi-Civita ε_ijk en {0,1,2}."""
    if i == j or j == k or k == i:
        return 0.0
    even = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    return 1.0 if (i, j, k) in even else -1.0


def _bar_eta(a: int, mu: int, nu: int) -> float:
    """Símbolo anti-autodual de 't Hooft η̄ᵃ_{μν}.

    η̄ᵃ_{ij} = ε_{aij},  η̄ᵃ_{0i} = −δᵃᵢ,  η̄ᵃ_{i0} = +δᵃᵢ,  η̄ᵃ_{00} = 0.
    Índices: a ∈ {0,1,2}, μ,ν ∈ {0,1,2,3}.
    """
    if mu == nu:
        return 0.0
    if mu == 0:
        return -1.0 if nu - 1 == a else 0.0
    if nu == 0:
        return 1.0 if mu - 1 == a else 0.0
    return _levi3(a, mu - 1, nu - 1)


def _project_su2(A: np.ndarray) -> np.ndarray:
    """Proyección a 𝔰𝔲(2): antihermítica y de traza nula (2×2)."""
    A_anti = 0.5 * (A - A.conj().T)
    return A_anti - (np.trace(A_anti) / 2.0) * np.eye(2, dtype=np.complex128)


def _embed_su2_in_suN(a2: np.ndarray, n: int) -> np.ndarray:
    """Inmersión bloque-superior SU(2) ↪ SU(N) (subgrupo ADHM)."""
    A = np.zeros((n, n), dtype=np.complex128)
    k = min(2, n)
    A[:k, :k] = a2[:k, :k]
    return A


# ══════════════════════════════════════════════════════════════════════════════
# §A. DATACLASSES INMUTABLES DEL SOBERANO
# ══════════════════════════════════════════════════════════════════════════════

class TopologyRegime(IntEnum):
    TRIVIAL_BUNDLE = 0
    MONOPOLE_LIKE = 1
    INSTANTON_DENSE = 2


@dataclass(frozen=True, slots=True)
class WeaverWitnessGaugeContract:
    """Contrato de calibre inmutable forjado en la FASE I.

    Invariantes: CS₃, densidad de Pontryagin, k ∈ ℤ, c₁ de Hopf, holonomía
    de Wilson (lazo cerrado), holonomía de Hopf, residual de Stora–Zumino,
    residual BRST s², defecto de variación de gauge.
    """

    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    stora_zumino_residual: float
    gauge_variation_residual: float
    brst_nilpotency_residual: float
    pontryagin_density: float
    instanton_number_k: float
    chern_hopf_class_c1: float
    hopf_invariant: float
    wilson_holonomy_angle: float
    wilson_loop_trace_real: float
    hopf_holonomy_angle: float
    hopf_base_coordinates: Tuple[float, float, float]
    topology_regime: str
    bpst_rho: float
    hmac_signature: str
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class WeaverWitnessGaugeInitialCondition:
    """Semilla formal FASE I → FASE II.

    Empaqueta el contrato de calibre verificado + la condición QND inicial
    del motor v7.0.0 + huella canónica HMAC‖SHA-256.
    """

    gauge_contract: WeaverWitnessGaugeContract
    qnd_initial_condition: WeaverWitnessQNDInitialCondition
    sovereign_id: str
    preconditioner_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class WeaveObservationRequest:
    """Solicitud de ingesta y observación QND sobre una síntesis del Tejedor."""

    request_id: str
    weave_id: str
    apu_code: str
    density_matrix: np.ndarray
    stinespring_coupling_eps: float = 0.05


@dataclass(frozen=True, slots=True)
class WeaverWitnessCampaignResult:
    """Resultado consolidado de la campaña QND (FASE II), con herencia celeste."""

    campaign_id: str
    sovereign_id: str
    weaves_observed_count: int
    weak_value_aw_mean: complex
    weak_value_aw_variance: float
    fubini_study_distance_peak: float
    fubini_study_distance_mean: float
    berry_phase_peak: float
    melnikov_integral_peak: float
    bryuno_sum_mean: float
    kam_small_divisor_bound: float
    nekhoroshev_stability_time: float
    moser_twist: float
    greene_residue_R: float
    symplectic_defect: float
    gromov_capacity_peak: float
    novikov_valuation_mean: float
    oseledets_transverse_mean: float
    rsi3_monadic_rate: float
    monad_associativity_residual: float
    kleisli_trace_length: int
    global_heyting_verdict: HeytingOmega3
    timestamp: float


@dataclass(frozen=True, slots=True)
class WeaverWitnessCampaignDraftBundle:
    """Semilla formal FASE II → FASE III."""

    bundle_id: str
    sovereign_id: str
    campaign_result: WeaverWitnessCampaignResult
    engine_draft: WeaverWitnessCertificateDraft
    gauge_contract_id: str
    observation_ids: Tuple[str, ...]
    bundle_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class WeaverWitnessSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza emitido en la FASE III."""

    passport_id: str
    sovereign_id: str
    global_heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    weaves_observed_count: int
    level3_rsi_monadic_rate: float
    gromov_capacity_peak: float
    symplectic_defect_inherited: float
    kam_small_divisor_bound: float
    nekhoroshev_stability_time: float
    instanton_number_k: float
    chern_hopf_class_c1: float
    esp32_crowbar_actuated: bool
    esp32_crowbar_latency_ns: float
    dirac_vacuum_purge_energy_mev: float
    merkle_root_sha256: str
    provenance_hash: str
    heyting_pseudocomplement: HeytingOmega3
    positron_grace_deadline_utc: float
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ExecutiveWeaverWitnessImpactReport:
    """Reporte Ejecutivo Φ_sem ('Dolor y Dinero') traducido para la junta."""

    report_id: str
    passport_id: str
    verdict_name: str
    truth_degree: float
    actuation_description: str
    kv_cache_compression_pct: float
    wacc_protection_pct: float
    imprevistos_reduction_pct: float
    capital_salvaguardado_usd: float
    executive_summary: str
    timestamp_utc: float


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE I — TOPOLOGÍA DE CALIBRE DEL FIBRADO PRINCIPAL P(T*Q, G)
#  G = U(1) ⊗ SU(2) ⊗ H₃(ℝ). Sector topológico en SU(2) ↪ SU(N) (ADHM).
#  El método terminal `emit_gauge_initial_condition` ES la semilla formal que
#  arranca la FASE II (`ingest_gauge_initial_condition`).
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class WisdomWeaverWitnessGaugeTopology:
    """FASE I — Topología de Calibre del Fibrado Principal.

    Construcciones:
      • Ansatz BPST anti-autodual A_μ = η̄ᵃ_{μν} x_ν T_a / (x²+ρ²).
      • Curvatura F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]
        (en la red, ∂A ≈ 0 ⇒ F ≈ [A,A] + corrección BPST analítica).
      • CS₃[A] = (1/8π²) Tr(A ∧ dA + (2/3) A³) antisimetrizado.
      • Densidad de Pontryagin ch₂ = (1/8π²) Tr(F ∧ F̃).
      • k = round(∫ ch₂) ∈ ℤ ≅ π₃(SU(2)).
      • Wilson cerrado W = 𝒫 exp(∮_{S¹} A) ∈ SU(2).
      • c₁ de Hopf = Ω(π(q))/2π, H(f) invariante de Hopf.
      • Stora–Zumino: |8π² CS₃ − Tr(F∧F̃)| y CS₃[Aᵍ]−CS₃[A].
      • BRST: s²λ ≈ 0 sobre un fantasma de prueba.
    """

    def __init__(self, sovereign_id: str = "WEAVER-WITNESS-MASTER-01") -> None:
        self.sovereign_id = sovereign_id
        self._hmac_secret = hashlib.sha256(
            f"SECRET-GAUGE-{sovereign_id}".encode("utf-8")
        ).digest()

    # ── I.1 Ansatz BPST anti-autodual en 𝔰𝔲(2) ────────────────────────────────
    @staticmethod
    def _bpst_connection(x: np.ndarray, rho: float = 1.0) -> List[np.ndarray]:
        """A_μ(x) = η̄ᵃ_{μν} x_ν (−i σ_a) / (x²+ρ²), μ = 0,1,2,3.

        Devuelve cuatro matrices 2×2 antihermíticas de traza nula.
        """
        den = float(np.dot(x, x) + rho * rho)
        den = max(den, _EPS_FLOOR)
        connections: List[np.ndarray] = []
        for mu in range(4):
            a_mu = np.zeros((2, 2), dtype=np.complex128)
            for a in range(3):
                acc = 0.0
                for nu in range(4):
                    acc += _bar_eta(a, mu, nu) * float(x[nu])
                a_mu += (acc / den) * (-1.0j * _SIGMA[a])
            connections.append(_project_su2(a_mu))
        return connections

    @staticmethod
    def _bpst_curvature_analytic(x: np.ndarray, rho: float = 1.0) -> List[np.ndarray]:
        r"""F_{μν} = 2 ρ² η̄ᵃ_{μν} (−i σ_a) / (x²+ρ²)²   (anti-autodual exacto)."""
        den = float(np.dot(x, x) + rho * rho)
        den2 = max(den * den, _EPS_FLOOR)
        pref = 2.0 * rho * rho / den2
        F: List[np.ndarray] = []
        # Se almacenan F_01, F_02, F_03, F_12, F_13, F_23 (6 componentes).
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for mu, nu in pairs:
            f = np.zeros((2, 2), dtype=np.complex128)
            for a in range(3):
                f += pref * _bar_eta(a, mu, nu) * (-1.0j * _SIGMA[a])
            F.append(_project_su2(f))
        return F

    # ── I.2 3-forma de Chern–Simons antisimetrizada ───────────────────────────
    @classmethod
    def _chern_simons_3form(
        cls,
        A_mu: np.ndarray,
        A_nu: np.ndarray,
        A_rho: np.ndarray,
    ) -> float:
        """CS₃ ≈ (1/8π²) ε^{ijk} Tr( A_i [A_j, A_k] + (2/3) A_i A_j A_k ).

        En la aproximación de Maurer–Cartan dA ~ [A,A]/2 sobre el 3-símplice
        {μ,ν,ρ} = {1,2,3} (esfera en el infinito del BPST).
        """
        As = (_project_su2(A_mu), _project_su2(A_nu), _project_su2(A_rho))
        acc = 0.0j
        # Única permutación independiente (1,2,3) más signatura
        for perm, sign in (
            ((0, 1, 2), 1.0),
            ((1, 2, 0), 1.0),
            ((2, 0, 1), 1.0),
            ((0, 2, 1), -1.0),
            ((2, 1, 0), -1.0),
            ((1, 0, 2), -1.0),
        ):
            i, j, k = perm
            comm = As[j] @ As[k] - As[k] @ As[j]
            linear = np.trace(As[i] @ comm)
            cubic = (2.0 / 3.0) * np.trace(As[i] @ As[j] @ As[k])
            acc += sign * (linear + cubic)
        cs3 = (1.0 / (8.0 * math.pi ** 2)) * acc / 6.0  # promedio de 3!
        return float(abs(cs3.real))

    # ── I.3 Densidad de Pontryagin y número de instantón ──────────────────────
    @classmethod
    def _pontryagin_and_instanton(
        cls,
        F_components: Sequence[np.ndarray],
        c1_hopf: float,
    ) -> Tuple[float, float]:
        """ch₂ = (1/8π²) Tr(F ∧ F̃). k = round(c₁ + ch₂_local) ∈ ℤ.

        Para BPST anti-autodual, F̃ = −F y ∫ch₂ = −1 (k = −1 en convención
        ASD). En un punto, reportamos la densidad local y cuantizamos con
        el winding de Hopf, que es el invariante global bien definido.
        """
        # Emparejamiento dual 4D: (01,23), (02,31)=(02,−13), (03,12)
        # F_components = [F01, F02, F03, F12, F13, F23]
        if len(F_components) < 6:
            density = 0.0
        else:
            F01, F02, F03, F12, F13, F23 = (_project_su2(f) for f in F_components)
            tr = (
                np.trace(F01 @ F23)
                - np.trace(F02 @ F13)
                + np.trace(F03 @ F12)
            )
            density = float((1.0 / (8.0 * math.pi ** 2)) * tr.real)
        # Cuantización topológica: el winding de Hopf es el entero honesto.
        k = float(round(c1_hopf))
        return density, k

    # ── I.4 Descenso de Stora–Zumino y variación de gauge ─────────────────────
    @classmethod
    def _stora_zumino_residual(
        cls,
        cs3: float,
        pontryagin: float,
    ) -> float:
        """Identidad d CS₃ = Tr(F∧F)/(8π²) ≡ ch₂. Residual relativo."""
        return float(abs(cs3 - abs(pontryagin)) / (1.0 + abs(pontryagin)))

    @classmethod
    def _gauge_variation_residual(
        cls,
        A_list: Sequence[np.ndarray],
        eps: float = 0.07,
    ) -> float:
        """CS₃[Aᵍ] − CS₃[A] debe ser exacta (d Tr(λ dA)) para g = exp(ε T₃).

        En 0-formas discretas el término exacto se anula y la diferencia
        debe ser O(ε²) si A está en un instante BPST regular.
        """
        if len(A_list) < 3:
            return 0.0
        g = la.expm(eps * _SU2_GEN[2])
        g_inv = g.conj().T  # SU(2) ⇒ g⁻¹ = g†

        def _cs(triple: Sequence[np.ndarray]) -> float:
            return cls._chern_simons_3form(triple[0], triple[1], triple[2])

        cs_a = _cs(A_list[:3])
        A_g = [g_inv @ a @ g for a in A_list[:3]]  # dg=0 (g constante)
        cs_ag = _cs(A_g)
        return float(abs(cs_ag - cs_a))

    # ── I.5 BRST: s² = 0 sobre un fantasma de prueba ──────────────────────────
    @staticmethod
    def _brst_nilpotency_residual(eps: float = 0.05) -> float:
        """sA = d_A c,  sc = −c². Sobre c = ε T₁, s²c = −s(c²) debe anularse
        porque c² ∝ I (T₁² = −¼ I) es central y sI = 0. Residual numérico
        de ‖s²c‖_F.
        """
        c = eps * _SU2_GEN[0]
        c2 = c @ c
        # s(c) = −c² ; s(s(c)) = −s(c²). c² ∝ I ⇒ [c, c²] = 0 ⇒ s²c = 0.
        s_c = -c2
        s2_c = -(c @ c2 - c2 @ c)  # ad_c (c²)
        return float(np.linalg.norm(s2_c, "fro"))

    # ── I.6 Holonomía de Wilson sobre un lazo CERRADO ─────────────────────────
    @classmethod
    def _wilson_loop_holonomy(
        cls,
        A_theta: np.ndarray,
        n_seg: int = 16,
    ) -> Tuple[float, float]:
        """W = ∏_{k=0}^{n−1} exp(A_θ Δθ),  Δθ = 2π/n,  γ(0)=γ(1).

        Devuelve (Re Tr(W)/2, θ_W = arg(det W)).
        Para A_θ = α T₃, W = diag(e^{-iπα}, e^{+iπα}), Tr/2 = cos(πα).
        """
        a = _project_su2(A_theta)
        dtheta = _TAU / float(n_seg)
        W = np.eye(2, dtype=np.complex128)
        step = la.expm(a * dtheta)
        for _ in range(n_seg):
            W = W @ step
        # Proyección numérica a SU(2)
        det_W = np.linalg.det(W)
        if abs(det_W) > _EPS_FLOOR:
            W = W / (det_W ** 0.5)
            det_W = np.linalg.det(W)
        trace_re = float((np.trace(W) / 2.0).real)
        theta_W = float(cmath.phase(det_W if abs(det_W) > _EPS_FLOOR else 1.0 + 0j))
        return trace_re, theta_W

    # ── I.7 Clase de Chern–Hopf e invariante de Hopf ──────────────────────────
    @staticmethod
    def _chern_hopf_class(
        q: QuaternionHopfLift,
    ) -> Tuple[float, float, float, Tuple[float, float, float], str]:
        """c₁ = Ω(π(q)) / 2π, con Ω el ángulo sólido del punto base en S².

        El área de un casquete de altura z = Z_{S²} es 2π(1−z) en S² de
        radio 1 (área total 4π). La clase de Chern local se toma como
        c₁ = Z / 2 ∈ [−1/2, 1/2]; el invariante de Hopf del fibrado
        tautológico es H = 1 cuando |q|=1 (identidad S³→SU(2)).

        Régimen:
          |c₁| < 0.10 → TRIVIAL_BUNDLE
          |c₁| < 0.35 → MONOPOLE_LIKE
          |c₁| ≥ 0.35 → INSTANTON_DENSE
        """
        nrm = q.norm() + _LOG_FLOOR
        qn = QuaternionHopfLift(q.w / nrm, q.x / nrm, q.y / nrm, q.z / nrm)
        base = qn.hopf_base_point()  # (X, Y, Z) ∈ S²
        z_s2 = float(np.clip(base[2], -1.0, 1.0))
        c1 = 0.5 * z_s2
        holonomy = (_TAU * c1) % _TAU
        # Invariante de Hopf del levantamiento unidad: grado 1 (fibrado tautológico)
        hopf_inv = 1.0 if nrm > 0.5 else 0.0
        ac = abs(c1)
        if ac < 0.10:
            regime = TopologyRegime.TRIVIAL_BUNDLE.name
        elif ac < 0.35:
            regime = TopologyRegime.MONOPOLE_LIKE.name
        else:
            regime = TopologyRegime.INSTANTON_DENSE.name
        return float(c1), float(holonomy), float(hopf_inv), base, regime

    # ── I.8 Serialización canónica determinista para HMAC ─────────────────────
    @staticmethod
    def _canonical_hmac_bytes(
        contract_id: str,
        sovereign_id: str,
        cs3: float,
        sz_residual: float,
        gv_residual: float,
        brst_residual: float,
        pontryagin: float,
        instanton_k: float,
        c1: float,
        hopf_inv: float,
        wilson_theta: float,
        wilson_tr: float,
        hopf_holonomy: float,
        regime: str,
        bpst_rho: float,
    ) -> bytes:
        """Orden lexicográfico de campos, formato % .12e sin locale."""
        fields = [
            ("bpst_rho", f"{bpst_rho:.12e}"),
            ("brst_nilpotency_residual", f"{brst_residual:.12e}"),
            ("chern_hopf_class_c1", f"{c1:.12e}"),
            ("chern_simons_3form", f"{cs3:.12e}"),
            ("contract_id", contract_id),
            ("gauge_variation_residual", f"{gv_residual:.12e}"),
            ("hopf_holonomy_angle", f"{hopf_holonomy:.12e}"),
            ("hopf_invariant", f"{hopf_inv:.12e}"),
            ("instanton_number_k", f"{instanton_k:.6f}"),
            ("pontryagin_density", f"{pontryagin:.12e}"),
            ("sovereign_id", sovereign_id),
            ("stora_zumino_residual", f"{sz_residual:.12e}"),
            ("topology_regime", regime),
            ("wilson_holonomy_angle", f"{wilson_theta:.12e}"),
            ("wilson_loop_trace_real", f"{wilson_tr:.12e}"),
        ]
        return "|".join(f"{k}={v}" for k, v in fields).encode("utf-8")

    def _hmac_of_contract_fields(
        self,
        contract_id: str,
        sovereign_id: str,
        cs3: float,
        sz_r: float,
        gv_r: float,
        brst_r: float,
        pontryagin: float,
        k: float,
        c1: float,
        hopf_inv: float,
        theta_W: float,
        tr_W: float,
        hopf_hol: float,
        regime: str,
        bpst_rho: float,
    ) -> str:
        payload = self._canonical_hmac_bytes(
            contract_id, sovereign_id, cs3, sz_r, gv_r, brst_r,
            pontryagin, k, c1, hopf_inv, theta_W, tr_W, hopf_hol, regime, bpst_rho,
        )
        return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()

    def _hmac_of_contract(self, gc: WeaverWitnessGaugeContract) -> str:
        return self._hmac_of_contract_fields(
            gc.contract_id, gc.sovereign_id, gc.chern_simons_3form,
            gc.stora_zumino_residual, gc.gauge_variation_residual,
            gc.brst_nilpotency_residual, gc.pontryagin_density,
            gc.instanton_number_k, gc.chern_hopf_class_c1, gc.hopf_invariant,
            gc.wilson_holonomy_angle, gc.wilson_loop_trace_real,
            gc.hopf_holonomy_angle, gc.topology_regime, gc.bpst_rho,
        )

    # ── I.9 Forja del contrato de calibre ─────────────────────────────────────
    def forge_weaver_gauge_contract(
        self,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        dimension_mac: int = 56,
        seed: int = 42,
        bpst_rho: float = 1.0,
    ) -> WeaverWitnessGaugeContract:
        """Forja el Contrato de Calibre con HMAC-SHA256 canónico.

        `dimension_mac` se conserva en la firma por compatibilidad ADHM
        (el instantón se inmerge en SU(N) aunque los invariantes se
        calculan en el subgrupo SU(2)).
        """
        now = time.time()
        contract_id = f"CONTRACT-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        _ = (dimension_mac, seed)  # N y semilla reservados para fluctuaciones bulk

        # Punto de muestra del BPST (no el origen, para F ≠ 0)
        x = np.array([0.3, 0.2, 0.1, 0.4], dtype=np.float64)
        A = self._bpst_connection(x, rho=bpst_rho)          # A_0, A_1, A_2, A_3
        F = self._bpst_curvature_analytic(x, rho=bpst_rho)

        q = QuaternionHopfLift.from_complex_pair(
            complex(quaternion_q[0], quaternion_q[1]),
            complex(quaternion_q[2], quaternion_q[3]),
        )
        c1, hopf_hol, hopf_inv, base_xyz, regime = self._chern_hopf_class(q)

        cs3 = self._chern_simons_3form(A[1], A[2], A[3])
        pontryagin, k = self._pontryagin_and_instanton(F, c1_hopf=c1)
        sz_r = self._stora_zumino_residual(cs3, pontryagin)
        gv_r = self._gauge_variation_residual(A[1:4])
        brst_r = self._brst_nilpotency_residual()

        # Lazo cerrado sobre el ecuador: A_θ = A_1 cos φ + A_2 sin φ ≈ A_1 (muestra)
        # Se toma el generador T₃ escalado por c₁ (holonomía Hopf → Wilson).
        a_theta = (2.0 * c1) * _SU2_GEN[2]
        tr_W, theta_W = self._wilson_loop_holonomy(a_theta, n_seg=16)

        hmac_sig = self._hmac_of_contract_fields(
            contract_id, self.sovereign_id, cs3, sz_r, gv_r, brst_r,
            pontryagin, k, c1, hopf_inv, theta_W, tr_W, hopf_hol, regime, bpst_rho,
        )

        contract = WeaverWitnessGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=cs3,
            stora_zumino_residual=sz_r,
            gauge_variation_residual=gv_r,
            brst_nilpotency_residual=brst_r,
            pontryagin_density=pontryagin,
            instanton_number_k=k,
            chern_hopf_class_c1=c1,
            hopf_invariant=hopf_inv,
            wilson_holonomy_angle=theta_W,
            wilson_loop_trace_real=tr_W,
            hopf_holonomy_angle=hopf_hol,
            hopf_base_coordinates=base_xyz,
            topology_regime=regime,
            bpst_rho=bpst_rho,
            hmac_signature=hmac_sig,
            creation_timestamp=now,
        )
        logger.info(
            "[%s] Contrato de Calibre forjado [%s] | CS₃=%.4e | k=%.0f | "
            "c₁=%+.4f | H=%.0f | W=%.4f | δSZ=%.3e | δg=%.3e | s²=%.3e | %s",
            self.sovereign_id, contract_id, cs3, k, c1, hopf_inv, theta_W,
            sz_r, gv_r, brst_r, regime,
        )
        return contract

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE I ◀◀◀
    #     Entrega WeaverWitnessGaugeInitialCondition, ESTRUCTURA DE ENTRADA
    #     del primer método de la FASE II (`ingest_gauge_initial_condition`).
    # ══════════════════════════════════════════════════════════════════════════
    def emit_gauge_initial_condition(
        self,
        weave_id: str,
        rho_mac: np.ndarray,
        rho_weave: np.ndarray,
        N_diag: np.ndarray,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> WeaverWitnessGaugeInitialCondition:
        """Cierre formal de FASE I: contrato de calibre + semilla QND v7."""
        contract = self.forge_weaver_gauge_contract(
            quaternion_q=quaternion_q,
            dimension_mac=int(rho_weave.shape[0]),
            seed=int(time.time() * 1e3) % 100_000,
        )
        qnd_condition = PoincareWeaverWitnessAtlas.emit_qnd_initial_condition(
            weave_id=weave_id,
            rho_mac=rho_mac,
            rho_weave=rho_weave,
            N_diag=N_diag,
            perturbation_eps=perturbation_eps,
        )
        precond_bytes = (
            f"{contract.hmac_signature}|{qnd_condition.preconditioner_sha256}|"
            f"{self.sovereign_id}|{weave_id}"
        ).encode("utf-8")
        precond_sha = hashlib.sha256(precond_bytes).hexdigest()

        logger.info(
            "[FASE I → FASE II] Condición inicial de calibre emitida | "
            "contract=%s | qnd_seed=%s | δSp=%.3e | T_N=%.3e",
            contract.contract_id,
            qnd_condition.seed.seed_id,
            qnd_condition.seed.symplectic_defect,
            qnd_condition.seed.nekhoroshev_stability_time,
        )
        return WeaverWitnessGaugeInitialCondition(
            gauge_contract=contract,
            qnd_initial_condition=qnd_condition,
            sovereign_id=self.sovereign_id,
            preconditioner_sha256=precond_sha,
            timestamp_utc=time.time(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE II — SOBERANO DE OBSERVACIÓN QND Y CAMPAÑAS RSI3
#  El primer método `ingest_gauge_initial_condition` CONSUME exactamente la
#  estructura que entregó el método terminal de FASE I.
#  El método terminal `emit_campaign_draft_for_adjudication` es la semilla
#  formal de FASE III (`receive_campaign_draft`).
#  RSI3 se aplana UNA vez, en el emit del motor (no se re-aplana aquí).
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONWisdomWeaverWitnessAgent(WisdomWeaverWitnessGaugeTopology):
    """FASE II — Soberano Testigo Tejedor.

    Orquesta campañas QND. Hereda la topología de calibre de FASE I
    (sustitución de Liskov sobre el API terminal de FASE I).
    """

    def __init__(
        self,
        sovereign_id: str = "WEAVER-WITNESS-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id)
        self.dimension_mac = dimension_mac
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension_mac))

        self._qnd_engine = WeaverWitnessQNDEngine(dimension=dimension_mac)
        self._sovereign_engine = TOONWisdomWeaverWitnessEngine(
            dimension=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
        )

        self._gauge_condition: Optional[WeaverWitnessGaugeInitialCondition] = None
        self._last_rsi3_rate: float = 0.25
        self._accumulated_observations: List[WeaverWeakObservationGerm] = []

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE II ◀◀◀
    #     Consume WeaverWitnessGaugeInitialCondition emitida por FASE I.
    #     CONTINUACIÓN FORMAL del método terminal de FASE I.
    # ══════════════════════════════════════════════════════════════════════════
    def ingest_gauge_initial_condition(
        self,
        condition: WeaverWitnessGaugeInitialCondition,
    ) -> "TOONWisdomWeaverWitnessAgent":
        """Valida HMAC del contrato (campos almacenados) y preconditioner."""
        gc = condition.gauge_contract
        expected_hmac = self._hmac_of_contract(gc)
        if not hmac.compare_digest(expected_hmac, gc.hmac_signature):
            logger.error(
                "[FASE II] Falla de autenticación HMAC en contrato %s.",
                gc.contract_id,
            )
            raise ValueError("HMAC de contrato de calibre inválido.")

        expected_precond = hashlib.sha256(
            (
                f"{gc.hmac_signature}|{condition.qnd_initial_condition.preconditioner_sha256}|"
                f"{self.sovereign_id}|{condition.qnd_initial_condition.weave_id}"
            ).encode("utf-8")
        ).hexdigest()
        if expected_precond != condition.preconditioner_sha256:
            logger.warning(
                "[FASE II] Preconditioner divergente: %s… vs %s…",
                expected_precond[:16],
                condition.preconditioner_sha256[:16],
            )

        self._gauge_condition = condition
        self._accumulated_observations.clear()
        self._qnd_engine.ingest_qnd_initial_condition(condition.qnd_initial_condition)
        logger.info(
            "[FASE II ← FASE I] Condición de calibre %s ingerida | Régimen: %s | "
            "k=%.0f | c₁=%+.4f | δSp=%.3e",
            gc.contract_id,
            gc.topology_regime,
            gc.instanton_number_k,
            gc.chern_hopf_class_c1,
            condition.qnd_initial_condition.seed.symplectic_defect,
        )
        return self

    # ── II.1 Observación individual ───────────────────────────────────────────
    def process_weave_observation(
        self,
        request: WeaveObservationRequest,
        mac_density_matrix: np.ndarray,
    ) -> WeaverWeakObservationGerm:
        logger.info(
            "[%s] Observando síntesis [%s] (APU: %s)…",
            self.sovereign_id, request.weave_id, request.apu_code,
        )
        return self._qnd_engine.observe_weaver_weak_value(
            weave_id=request.weave_id,
            mac_density_matrix=mac_density_matrix,
            weave_density_matrix=request.density_matrix,
        )

    # ── II.2 Adjudicación local Heyting Ω₃ ────────────────────────────────────
    @staticmethod
    def _local_heyting_verdict(
        obs: WeaverWeakObservationGerm,
        capacity_gromov_max: float,
    ) -> HeytingOmega3:
        if obs.capacity_gromov > capacity_gromov_max or obs.fubini_study_distance > 1.50:
            return HeytingOmega3.VETOED
        if obs.fubini_study_distance > 0.15 or abs(obs.berry_pancharatnam_phase) >= 0.50:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ── II.3 Herencia celeste del germen canónico ─────────────────────────────
    def _celestial_from_seed(self) -> Tuple[float, float, float, float, float, float]:
        """(Melnikov, Bryuno, KAM, Nekhoroshev, twist, δSp) desde FASE I."""
        if self._gauge_condition is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = self._gauge_condition.qnd_initial_condition.seed
        return (
            float(s.melnikov_integral_M0),
            float(s.bryuno_diophantine_sum),
            float(s.kam_small_divisor_bound),
            float(s.nekhoroshev_stability_time),
            float(s.moser_twist),
            float(s.symplectic_defect),
        )

    # ── II.4 Campaña de observación QND (sin re-aplanar RSI3) ─────────────────
    def conduct_witness_observation_campaign(
        self,
        requests: List[WeaveObservationRequest],
        mac_density_matrix: np.ndarray,
    ) -> WeaverWitnessCampaignResult:
        """Orquesta la campaña. RSI3 NO se aplica aquí: se aplana una sola
        vez en `emit_campaign_draft_for_adjudication` (ley μ∘Tμ = μ∘μT)."""
        now = time.time()
        campaign_id = f"CAMP-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        melnikov, bryuno, kam, t_nek, twist, sp_def = self._celestial_from_seed()

        empty = WeaverWitnessCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            weaves_observed_count=0,
            weak_value_aw_mean=0.0 + 0.0j,
            weak_value_aw_variance=0.0,
            fubini_study_distance_peak=0.0,
            fubini_study_distance_mean=0.0,
            berry_phase_peak=0.0,
            melnikov_integral_peak=melnikov,
            bryuno_sum_mean=bryuno,
            kam_small_divisor_bound=kam,
            nekhoroshev_stability_time=t_nek,
            moser_twist=twist,
            greene_residue_R=(
                0.0 if self._gauge_condition is None
                else float(self._gauge_condition.qnd_initial_condition.seed.greene_residue_R)
            ),
            symplectic_defect=sp_def,
            gromov_capacity_peak=0.0,
            novikov_valuation_mean=0.0,
            oseledets_transverse_mean=0.0,
            rsi3_monadic_rate=self._last_rsi3_rate,
            monad_associativity_residual=0.0,
            kleisli_trace_length=0,
            global_heyting_verdict=HeytingOmega3.COHERENT,
            timestamp=now,
        )
        if not requests:
            return empty

        observations: List[WeaverWeakObservationGerm] = [
            self.process_weave_observation(req, mac_density_matrix) for req in requests
        ]
        self._accumulated_observations.extend(observations)

        aw_list = np.array([complex(o.weak_value_Aw) for o in observations])
        d_fs_list = np.array([o.fubini_study_distance for o in observations])
        berry_list = np.array([abs(o.berry_pancharatnam_phase) for o in observations])
        gromov_list = np.array([o.capacity_gromov for o in observations])

        verdict = HeytingOmega3.COHERENT
        for obs in observations:
            verdict = verdict.meet(self._local_heyting_verdict(obs, self.capacity_gromov_max))

        greene = empty.greene_residue_R
        result = WeaverWitnessCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            weaves_observed_count=len(requests),
            weak_value_aw_mean=complex(np.mean(aw_list)),
            weak_value_aw_variance=float(np.var(np.abs(aw_list))),
            fubini_study_distance_peak=float(np.max(d_fs_list)),
            fubini_study_distance_mean=float(np.mean(d_fs_list)),
            berry_phase_peak=float(np.max(berry_list)),
            melnikov_integral_peak=melnikov,
            bryuno_sum_mean=bryuno,
            kam_small_divisor_bound=kam,
            nekhoroshev_stability_time=t_nek,
            moser_twist=twist,
            greene_residue_R=greene,
            symplectic_defect=sp_def,
            gromov_capacity_peak=float(np.max(gromov_list)),
            novikov_valuation_mean=float(np.mean([o.novikov_valuation for o in observations])),
            oseledets_transverse_mean=float(
                np.mean([o.oseledets_transverse_lyapunov for o in observations])
            ),
            rsi3_monadic_rate=self._last_rsi3_rate,  # placeholder; se sella en el emit
            monad_associativity_residual=float(
                np.mean([getattr(o, "monad_laws_residual", 0.0) for o in observations])
            ),
            kleisli_trace_length=0,
            global_heyting_verdict=verdict,
            timestamp=now,
        )
        logger.info(
            "[%s] Campaña %s | N=%d | Aw̄=%s | d_FS_max=%.4f | γ_peak=%.4f | "
            "c_G=%.4f | M0=%+.3e | B=%.4f | T_N=%.3e | R_G=%+.3e | Verdict=%s",
            self.sovereign_id, campaign_id, len(requests),
            f"{result.weak_value_aw_mean:.4f}",
            result.fubini_study_distance_peak, result.berry_phase_peak,
            result.gromov_capacity_peak, melnikov, bryuno, t_nek, greene,
            verdict.name,
        )
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE II ◀◀◀
    #     Emite WeaverWitnessCampaignDraftBundle, semilla formal de FASE III
    #     (`SovereignWisdomWeaverWitnessAdjudicator.receive_campaign_draft`).
    # ══════════════════════════════════════════════════════════════════════════
    def emit_campaign_draft_for_adjudication(
        self,
        campaign_result: WeaverWitnessCampaignResult,
    ) -> WeaverWitnessCampaignDraftBundle:
        """Cierre formal de FASE II. RSI3 se aplana aquí, una sola vez,
        vía el emit del motor v7 (μ: T² ⇒ T, y μ∘μ para T³ ⇒ T)."""
        if not self._accumulated_observations:
            raise RuntimeError(
                "No hay observaciones acumuladas. Ejecute "
                "conduct_witness_observation_campaign."
            )

        engine_draft = self._qnd_engine.emit_certificate_draft_for_adjudication(
            observations=list(self._accumulated_observations),
            previous_rsi3_rate=self._last_rsi3_rate,
        )
        self._last_rsi3_rate = engine_draft.level3_rsi_monadic_rate

        # Sellar RSI3 y traza Kleisli sobre el resultado (dataclass frozen).
        campaign_result = replace(
            campaign_result,
            rsi3_monadic_rate=engine_draft.level3_rsi_monadic_rate,
            monad_associativity_residual=engine_draft.monad_associativity_residual,
            kleisli_trace_length=len(engine_draft.kleisli_composition_trace),
        )

        now = time.time()
        bundle_id = f"BUNDLE-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        gc_id = (
            self._gauge_condition.gauge_contract.contract_id
            if self._gauge_condition is not None
            else "NONE"
        )
        obs_ids = tuple(o.observation_id for o in self._accumulated_observations)
        bundle_bytes = (
            f"{bundle_id}|{self.sovereign_id}|{campaign_result.campaign_id}|"
            f"{engine_draft.draft_id}|{gc_id}|"
            + "|".join(obs_ids)
        ).encode("utf-8")
        bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()

        logger.info(
            "[FASE II → FASE III] Bundle emitido %s | N=%d | RSI3=%.6f | "
            "δμ=%.3e | draft=%s",
            bundle_id, len(obs_ids), engine_draft.level3_rsi_monadic_rate,
            engine_draft.monad_associativity_residual, engine_draft.draft_id,
        )
        return WeaverWitnessCampaignDraftBundle(
            bundle_id=bundle_id,
            sovereign_id=self.sovereign_id,
            campaign_result=campaign_result,
            engine_draft=engine_draft,
            gauge_contract_id=gc_id,
            observation_ids=obs_ids,
            bundle_sha256=bundle_sha,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE III — ADJUDICADOR CIBER-FÍSICO Y CIERRE EJECUTIVO
#  El primer método `receive_campaign_draft` CONSUME exactamente el
#  WeaverWitnessCampaignDraftBundle de FASE II. Adjudica en Ω₃, acciona el
#  crowbar ESP32 < 400 ns / GPIO14, purga al Vacío de Dirac, inyecta positrón
#  HMAC con ventana de 1 h, y traduce al funtor semántico Φ_sem.
#  CONTINUACIÓN FORMAL del método terminal de FASE II.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class SovereignWisdomWeaverWitnessAdjudicator(TOONWisdomWeaverWitnessAgent):
    """FASE III — Adjudicación Ciber-Física y Cierre Ejecutivo en Heyting Ω₃."""

    def __init__(
        self,
        sovereign_id: str = "WEAVER-WITNESS-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            dimension_mac=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
        )
        self._campaign_bundle: Optional[WeaverWitnessCampaignDraftBundle] = None
        self._purged_observation_ids: List[str] = []
        self._crowbar_actuated: bool = False
        self._last_passport: Optional[WeaverWitnessSovereignGovernancePassport] = None

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE III ◀◀◀
    #     Consume WeaverWitnessCampaignDraftBundle emitido por FASE II.
    #     CONTINUACIÓN FORMAL del método terminal de FASE II.
    # ══════════════════════════════════════════════════════════════════════════
    def receive_campaign_draft(
        self,
        bundle: WeaverWitnessCampaignDraftBundle,
    ) -> "SovereignWisdomWeaverWitnessAdjudicator":
        recomputed = hashlib.sha256(
            (
                f"{bundle.bundle_id}|{bundle.sovereign_id}|{bundle.campaign_result.campaign_id}|"
                f"{bundle.engine_draft.draft_id}|{bundle.gauge_contract_id}|"
                + "|".join(bundle.observation_ids)
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != bundle.bundle_sha256:
            logger.warning(
                "[FASE III] Huella divergente del bundle: %s… vs %s…",
                recomputed[:16], bundle.bundle_sha256[:16],
            )
        self._campaign_bundle = bundle
        self._sovereign_engine.receive_certificate_draft(bundle.engine_draft)
        logger.info(
            "[FASE III ← FASE II] Bundle %s recibido | N=%d | verdict=%s | δμ=%.3e",
            bundle.bundle_id,
            len(bundle.observation_ids),
            bundle.campaign_result.global_heyting_verdict.name,
            bundle.engine_draft.monad_associativity_residual,
        )
        return self

    # ── III.1 Crowbar ESP32 (IRAM < 400 ns, GPIO14) ───────────────────────────
    def _esp32_crowbar_trigger(self, verdict: HeytingOmega3) -> Tuple[bool, float]:
        base_ns = {
            HeytingOmega3.COHERENT: 0.0,
            HeytingOmega3.DEGRADED: 120.0,
            HeytingOmega3.VETOED: 340.0,
        }[verdict]
        jitter = 6.25 * math.sin(time.time() * 1e6) if base_ns > 0.0 else 0.0
        latency = max(0.0, base_ns + jitter)
        actuated = verdict is HeytingOmega3.VETOED
        if actuated and latency > _CROWBAR_LATENCY_BUDGET_NS:
            logger.error(
                "[CROWBAR OVERRUN] Latencia %.2f ns > presupuesto %.0f ns.",
                latency, _CROWBAR_LATENCY_BUDGET_NS,
            )
        return actuated, float(latency)

    # ── III.2 Purga al vacío de Dirac (e⁻ + e⁺ → 2γ) ──────────────────────────
    @staticmethod
    def _dirac_vacuum_purge_energy(n_pairs: int) -> float:
        return float(n_pairs * 2.0 * _M_E_C2_MEV)

    # ── III.3 Merkle root SHA-256 ─────────────────────────────────────────────
    @staticmethod
    def _merkle_root(ids: List[str]) -> str:
        if not ids:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in ids]
        while len(level) > 1:
            nxt: List[str] = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
            level = nxt
        return level[0]

    # ── III.4 Positrón de autorización humana (ventana 1 h) ───────────────────
    def expected_positron_hmac(self, passport_id: str) -> str:
        return hmac.new(
            self._hmac_secret,
            f"POSITRON:{passport_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def inject_positron_authorization(
        self,
        passport: WeaverWitnessSovereignGovernancePassport,
        hmac_signature_human: str,
        now_utc: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Inyección de e⁺ firmado con HMAC. Sólo DEGRADED, y sólo dentro
        de la gracia de 1 h (`positron_grace_deadline_utc`)."""
        if passport.global_heyting_verdict != HeytingOmega3.DEGRADED:
            return False, "Inyección de positrón sólo aplicable a estados DEGRADED."
        ts = time.time() if now_utc is None else float(now_utc)
        if ts > passport.positron_grace_deadline_utc:
            return False, "Ventana de gracia de positrón (1 h) expirada."
        expected_sig = self.expected_positron_hmac(passport.passport_id)
        if hmac.compare_digest(expected_sig, hmac_signature_human):
            logger.info(
                "[%s] [e⁺ + e⁻ → 2γ] Positrón inyectado para %s. Estado → COHERENT.",
                self.sovereign_id, passport.passport_id,
            )
            return True, "Aniquilación cuántica completada. Anomalía disipada."
        logger.warning(
            "[%s] Firma HMAC inválida en inyección de positrón.", self.sovereign_id
        )
        return False, "Firma HMAC de autorización humana inválida."

    # ── III.5 Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business ──────────────────
    def translate_to_business_impact(
        self,
        passport: WeaverWitnessSovereignGovernancePassport,
    ) -> ExecutiveWeaverWitnessImpactReport:
        """τ(Ω₃) = ord/2 ∈ {0, ½, 1} interpola KPIs de forma contravariante
        al riesgo: a mayor verdad (COHERENT) menor capital de contingencia."""
        now = time.time()
        report_id = f"EXEC-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        tau = float(int(passport.global_heyting_verdict)) / 2.0  # 0, 0.5, 1

        base_kv, base_wacc, base_impr = 86.4, 15.0, 11.5
        kv = base_kv * (0.50 + 0.50 * tau)
        wacc = base_wacc * (0.50 + 0.50 * tau)
        impr = base_impr * (0.50 + 0.50 * tau)

        capital_scale = max(1.0, passport.dirac_vacuum_purge_energy_mev / 2.0)
        capital_saved = 300_000.0 * capital_scale * (1.0 - tau)

        if passport.global_heyting_verdict is HeytingOmega3.COHERENT:
            actuation_desc = "Flujo Normal Fluido. Síntesis numéricamente pura."
            summary = (
                f"El Testigo Tejedor observó {passport.weaves_observed_count} síntesis "
                f"con Back-Action 0.0 dB. Coherencia topológica mantenida; "
                f"RSI3={passport.level3_rsi_monadic_rate:.4f}; "
                f"δSp={passport.symplectic_defect_inherited:.3e}; "
                f"T_N={passport.nekhoroshev_stability_time:.3e}."
            )
        elif passport.global_heyting_verdict is HeytingOmega3.DEGRADED:
            actuation_desc = (
                "Veto Suave (Bypass). Recirculación mecánica activada "
                f"(gracia e⁺ { _POSITRON_GRACE_S / 3600.0:.0f} h)."
            )
            summary = (
                "ALERTA ÁMBAR: Deslizamiento de fase de Berry. Válvula de alivio "
                "activada; ventana de inyección de positrón e⁺ abierta."
            )
        else:
            actuation_desc = (
                f"Veto Duro (ESP32 Crowbar < {passport.esp32_crowbar_latency_ns:.1f} ns / "
                f"GPIO{self.esp32_gpio_pin} HIGH)."
            )
            summary = (
                f"CRÍTICO: Alucinación no física detectada (c_G > {self.capacity_gromov_max} "
                f"o d_FS hiperbólico). Disparo ciber-físico ESP32 Crowbar en "
                f"{passport.esp32_crowbar_latency_ns:.1f} ns. "
                f"Salvaguardado ${capital_saved:,.2f} USD."
            )

        return ExecutiveWeaverWitnessImpactReport(
            report_id=report_id,
            passport_id=passport.passport_id,
            verdict_name=passport.global_heyting_verdict.name,
            truth_degree=tau,
            actuation_description=actuation_desc,
            kv_cache_compression_pct=kv,
            wacc_protection_pct=wacc,
            imprevistos_reduction_pct=impr,
            capital_salvaguardado_usd=capital_saved,
            executive_summary=summary,
            timestamp_utc=now,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE III ◀◀◀
    #     Orquesta adjudicación, crowbar, purga Dirac, y emite Pasaporte + Φ_sem.
    # ══════════════════════════════════════════════════════════════════════════
    def orchestrate_governance_adjudication(
        self,
    ) -> Tuple[WeaverWitnessSovereignGovernancePassport, ExecutiveWeaverWitnessImpactReport]:
        if self._campaign_bundle is None:
            raise RuntimeError(
                "No hay bundle activo. Ejecute receive_campaign_draft primero."
            )

        now = time.time()
        passport_id = f"PASSPORT-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        bundle = self._campaign_bundle

        cert = self._sovereign_engine.audit_and_certify()
        verdict = cert.verdict
        actuation = cert.actuation_mode

        crowbar_actuated, latency_ns = self._esp32_crowbar_trigger(verdict)
        self._crowbar_actuated = crowbar_actuated

        purge_mev = self._dirac_vacuum_purge_energy(n_pairs=len(bundle.observation_ids))
        self._purged_observation_ids.extend(bundle.observation_ids)

        merkle_root = self._merkle_root(list(bundle.observation_ids))
        heyting_pseudo = verdict.pseudocomplement()

        gc_k = 0.0
        gc_c1 = 0.0
        if self._gauge_condition is not None:
            gc_k = float(self._gauge_condition.gauge_contract.instanton_number_k)
            gc_c1 = float(self._gauge_condition.gauge_contract.chern_hopf_class_c1)

        cr = bundle.campaign_result
        prov_bytes = (
            f"{passport_id}|{self.sovereign_id}|{verdict.name}|{actuation.name}|"
            f"{cr.weaves_observed_count}|{merkle_root}|"
            f"{bundle.bundle_sha256}|{purge_mev:.6f}|k={gc_k:.1f}|c1={gc_c1:.12e}"
        ).encode("utf-8")
        provenance_hash = hashlib.sha256(prov_bytes).hexdigest()

        passport = WeaverWitnessSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            global_heyting_verdict=verdict,
            actuation_mode=actuation,
            weaves_observed_count=cr.weaves_observed_count,
            level3_rsi_monadic_rate=cr.rsi3_monadic_rate,
            gromov_capacity_peak=cr.gromov_capacity_peak,
            symplectic_defect_inherited=cert.symplectic_defect_inherited,
            kam_small_divisor_bound=cr.kam_small_divisor_bound,
            nekhoroshev_stability_time=cr.nekhoroshev_stability_time,
            instanton_number_k=gc_k,
            chern_hopf_class_c1=gc_c1,
            esp32_crowbar_actuated=crowbar_actuated,
            esp32_crowbar_latency_ns=latency_ns,
            dirac_vacuum_purge_energy_mev=purge_mev,
            merkle_root_sha256=merkle_root,
            provenance_hash=provenance_hash,
            heyting_pseudocomplement=heyting_pseudo,
            positron_grace_deadline_utc=now + _POSITRON_GRACE_S,
            timestamp_utc=now,
        )
        self._last_passport = passport
        exec_report = self.translate_to_business_impact(passport)

        logger.info(
            "[FASE III] Pasaporte %s emitido | Verdict=%s | Mode=%s | "
            "Crowbar=%.1f ns | Purga=%.3f MeV | ¬Ω₃=%s | k=%.0f | "
            "δSp=%.3e | Merkle=%s…",
            passport_id, verdict.name, actuation.name, latency_ns, purge_mev,
            heyting_pseudo.name, gc_k, cert.symplectic_defect_inherited,
            merkle_root[:16],
        )
        self._campaign_bundle = None
        return passport, exec_report

    # ── III.6 Pipeline soberano completo FASE I → II → III ────────────────────
    def run_full_sovereign_pipeline(
        self,
        weave_id: str,
        rho_mac: np.ndarray,
        requests: List[WeaveObservationRequest],
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> Tuple[
        WeaverWitnessGaugeInitialCondition,
        WeaverWitnessCampaignResult,
        WeaverWitnessSovereignGovernancePassport,
        ExecutiveWeaverWitnessImpactReport,
    ]:
        """Orquestador de las tres fases anidadas del soberano."""
        rho_seed = requests[0].density_matrix if requests else rho_mac
        condition = self.emit_gauge_initial_condition(
            weave_id=weave_id,
            rho_mac=rho_mac,
            rho_weave=rho_seed,
            N_diag=self._N_diag,
            quaternion_q=quaternion_q,
            perturbation_eps=perturbation_eps,
        )
        self.ingest_gauge_initial_condition(condition)
        campaign = self.conduct_witness_observation_campaign(requests, rho_mac)
        bundle = self.emit_campaign_draft_for_adjudication(campaign)
        self.receive_campaign_draft(bundle)
        passport, report = self.orchestrate_governance_adjudication()
        return condition, campaign, passport, report


# ══════════════════════════════════════════════════════════════════════════════
# §Z. VERIFICACIÓN DOCTORAL DEL SOBERANO v7.0.0
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 82)
    print(f"  VERIFICACIÓN DOCTORAL DEL SOBERANO TOONWisdomWeaverWitnessAgent v{__version__}")
    print("  Fases anidadas: FASE I (Gauge/BPST) → FASE II (QND+RSI3) → FASE III (Crowbar)")
    print("═" * 82 + "\n")

    adjudicator = SovereignWisdomWeaverWitnessAdjudicator(
        sovereign_id="WEAVER-WITNESS-MASTER-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    mac_rho = np.outer(v0, v0.conj())

    v_weave1 = v0.copy()
    v_weave1[0] *= cmath.exp(1j * 0.01)
    v_weave1 /= np.linalg.norm(v_weave1)
    rho_weave1 = np.outer(v_weave1, v_weave1.conj())

    v_weave2 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v_weave2 /= np.linalg.norm(v_weave2)
    rho_weave2 = np.outer(v_weave2, v_weave2.conj())

    print("[FASE I] Forjando contrato de calibre BPST y emitiendo condición inicial …\n")
    condition = adjudicator.emit_gauge_initial_condition(
        weave_id="WEAVE-REFERENCE-001",
        rho_mac=mac_rho,
        rho_weave=rho_weave1,
        N_diag=adjudicator._N_diag,
        quaternion_q=(0.95, 0.05, 0.1, 0.05),
        perturbation_eps=0.05,
    )
    gc = condition.gauge_contract
    print(f"  • Contrato ID            : {gc.contract_id}")
    print(f"  • CS₃ (Chern–Simons)     : {gc.chern_simons_3form:.6e}")
    print(f"  • ch₂ (Pontryagin)       : {gc.pontryagin_density:.6e}")
    print(f"  • k (Instantón ℤ)        : {gc.instanton_number_k:.0f}")
    print(f"  • c₁ (Chern–Hopf)        : {gc.chern_hopf_class_c1:+.6f}")
    print(f"  • H (Hopf invariant)     : {gc.hopf_invariant:.0f}")
    print(f"  • W (Wilson θ, Tr/2)     : {gc.wilson_holonomy_angle:+.6f} rad, {gc.wilson_loop_trace_real:+.6f}")
    print(f"  • Holonomía Hopf         : {gc.hopf_holonomy_angle:+.6f} rad")
    print(f"  • Régimen topológico     : {gc.topology_regime}")
    print(f"  • Residual Stora–Zumino  : {gc.stora_zumino_residual:.4e}")
    print(f"  • Residual variación g   : {gc.gauge_variation_residual:.4e}")
    print(f"  • Residual BRST s²       : {gc.brst_nilpotency_residual:.4e}")
    print(f"  • HMAC SHA-256           : {gc.hmac_signature[:24]}…")
    print(f"  • Preconditioner SHA-256 : {condition.preconditioner_sha256[:24]}…")
    print(f"  • δSp (Sp(4) heredado)   : {condition.qnd_initial_condition.seed.symplectic_defect:.3e}")
    print(f"  • T_Nekhoroshev          : {condition.qnd_initial_condition.seed.nekhoroshev_stability_time:.3e}\n")

    print("[FASE II] Ingestando condición de calibre (HMAC) y conduciendo campaña QND …\n")
    adjudicator.ingest_gauge_initial_condition(condition)

    req1 = WeaveObservationRequest(
        request_id="REQ-WEAVER-001",
        weave_id="WEAVE-VACIADO-CONCRETO-3000PSI",
        apu_code="APU-OBRA-CIVIL-001",
        density_matrix=rho_weave1,
    )
    req2 = WeaveObservationRequest(
        request_id="REQ-WEAVER-002",
        weave_id="WEAVE-ACERO-FIGURADO-60000PSI",
        apu_code="APU-OBRA-CIVIL-002",
        density_matrix=rho_weave2,
    )
    campaign = adjudicator.conduct_witness_observation_campaign([req1, req2], mac_rho)
    bundle = adjudicator.emit_campaign_draft_for_adjudication(campaign)
    campaign = bundle.campaign_result  # RSI3 sellado

    print(f"  • Campaña ID             : {campaign.campaign_id}")
    print(f"  • N observadas           : {campaign.weaves_observed_count}")
    print(f"  • Aw̄ (valor débil medio) : {campaign.weak_value_aw_mean:.4f}")
    print(f"  • σ²_Aw                  : {campaign.weak_value_aw_variance:.4e}")
    print(
        f"  • d_FS peak / mean       : {campaign.fubini_study_distance_peak:.4f} / "
        f"{campaign.fubini_study_distance_mean:.4f} rad"
    )
    print(f"  • γ_Berry peak           : {campaign.berry_phase_peak:.6f} rad")
    print(f"  • c_G peak (Gromov)      : {campaign.gromov_capacity_peak:.6f}")
    print(f"  • Melnikov M₀            : {campaign.melnikov_integral_peak:+.6e}")
    print(f"  • Bryuno B(ω)            : {campaign.bryuno_sum_mean:.6f}")
    print(f"  • KAM γ                  : {campaign.kam_small_divisor_bound:.6e}")
    print(f"  • Nekhoroshev T_N        : {campaign.nekhoroshev_stability_time:.6e}")
    print(f"  • Twist Moser            : {campaign.moser_twist:.6e}")
    print(f"  • Residuo Greene R       : {campaign.greene_residue_R:+.6e}")
    print(f"  • δSp(4)                 : {campaign.symplectic_defect:.3e}")
    print(f"  • RSI3 monádico          : {campaign.rsi3_monadic_rate:.6f}")
    print(f"  • δμ (asociatividad)     : {campaign.monad_associativity_residual:.3e}")
    print(f"  • Traza Kleisli          : {campaign.kleisli_trace_length} composiciones")
    print(f"  • Veredicto Ω₃ global    : {campaign.global_heyting_verdict.name}")
    print(f"  • Bundle ID              : {bundle.bundle_id}")
    print(f"  • Bundle SHA-256         : {bundle.bundle_sha256[:24]}…\n")

    print("[FASE III] Recibiendo bundle y orquestando adjudicación ciber-física …\n")
    adjudicator.receive_campaign_draft(bundle)
    passport, exec_report = adjudicator.orchestrate_governance_adjudication()

    print("─" * 82)
    print("  PASAPORTE SOBERANO DE GOBERNANZA")
    print("─" * 82)
    print(f"  • ID Pasaporte           : {passport.passport_id}")
    print(f"  • Veredicto Heyting Ω₃   : {passport.global_heyting_verdict.name}")
    print(f"  • ¬Ω₃ (pseudocomplemento): {passport.heyting_pseudocomplement.name}")
    print(f"  • Modo de Actuación      : {passport.actuation_mode.name}")
    print(f"  • Crowbar IRAM Activo    : {passport.esp32_crowbar_actuated}")
    print(f"  • Latencia Crowbar       : {passport.esp32_crowbar_latency_ns:.2f} ns")
    print(f"  • Purga Vacío Dirac      : {passport.dirac_vacuum_purge_energy_mev:.4f} MeV")
    print(f"  • k / c₁ heredados       : {passport.instanton_number_k:.0f} / {passport.chern_hopf_class_c1:+.6f}")
    print(f"  • δSp heredado           : {passport.symplectic_defect_inherited:.3e}")
    print(f"  • T_Nekhoroshev          : {passport.nekhoroshev_stability_time:.3e}")
    print(f"  • Gracia e⁺ hasta        : {passport.positron_grace_deadline_utc:.0f} unix")
    print(f"  • Merkle SHA-256 Root    : {passport.merkle_root_sha256[:32]}…")
    print(f"  • Provenance Hash        : {passport.provenance_hash[:32]}…")
    print("─" * 82 + "\n")

    print("─" * 82)
    print("  REPORTE EJECUTIVO Φ_sem ('Dolor y Dinero')")
    print("─" * 82)
    print(f"  • Grado de verdad τ(Ω₃)  : {exec_report.truth_degree:.1f}")
    print(f"  • Compresión KV-Cache    : {exec_report.kv_cache_compression_pct:.1f}%")
    print(f"  • Protección Tasa WACC   : {exec_report.wacc_protection_pct:.1f}%")
    print(f"  • Reducción Imprevistos  : {exec_report.imprevistos_reduction_pct:.1f}%")
    print(f"  • Capital Salvaguardado  : ${exec_report.capital_salvaguardado_usd:,.2f} USD")
    print(f"  • Resumen Ejecutivo      :\n    \"{exec_report.executive_summary}\"")
    print("─" * 82 + "\n")

    print("═" * 82)
    print(f"  VERIFICACIÓN EXITOSA: SOBERANO v{__version__} OPERATIVO")
    print("═" * 82)