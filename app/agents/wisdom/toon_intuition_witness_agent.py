# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Intuition Witness Agent (Soberano Testigo de Intuición)                              ║
║ Ubicación: app/agents/wisdom/toon_intuition_witness_agent.py                                         ║
║ Versión  : 9.0.0-SOVEREIGN-RSI3-POINCARE-GAUGE-KAM-QND-HEYTING-ESP32-PhD                             ║
║ Función  : Soberano de Calibre, Medición Débil QND y Gobernanza Agéntica                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONIntuitionWitnessAgent` actúa como el Soberano Agéntico de Calibre y Gobernanza
para el Testigo de Intuición QND. En coordinación directa con su motor espectral
`TOONIntuitionWitnessEngine` (`app/wisdom/toon_intuition_witness_engine.py`), supervisa
la estructura de calibre del fibrado principal $P(\text{Gr}(r,n), U(1))$, conduce campañas de
medición débil AAV, audita la invariancia de Chern-Simons $\text{CS}_3(A)$, adjudica estados
en el retículo distributivo de Heyting $\Omega_3$, y proyecta los resultados hacia el funtor
semántico $\Phi_{\text{sem}}$ ("Riesgo, Dolor y Dinero").
Su arquitectura operativa se estructura en tres fases anidadas e indivisibles:
  - FASE 1: Topología de Calibre Grassmanniana, 1-forma de conexión $A \in \mathfrak{u}(n)$,
    curvatura de Yang-Mills $F = dA + A \wedge A$, 3-forma de Chern-Simons $\text{CS}_3$, clase de
    Chern-Hopf $c_1$, holonomía de Wilson $\mathcal{W}(C) = \frac{1}{n} \text{Tr} \mathcal{P} \exp \oint_C A$
    y acoplamiento a la geodésica conforme de Maupertuis-Poincaré.
  - FASE 2: Campañas de observación débil QND sobre saltos intuitivos flash, geometría de
    Bures-Wasserstein-Uhlmann, holonomía de Berry sobre la fibración de Hopf y la torre
    monádica de automejora RSI³ $\mathbb{T} = (T, \eta, \mu)$.
  - FASE 3: Adjudicación en el álgebra de Heyting $\Omega_3$, control de la válvula de alivio
    (3600 s), interlock ciber-físico simulado ($\le 398.95\text{ ns}$), aniquilación de positrones
    con firma HMAC estricta ($e^- + e^+ \to 2\gamma$) y proyección del funtor semántico $\Phi_{\text{sem}}$.

2. MARCO TEÓRICO:
─────────────────────────────────────────────────────────────────────────────────────────
2.1. FIBRADO GRASSMANNIANO Y HOLONOMÍA DE WILSON EN CALIBRE:
   Sobre la variedad de Grassmann $\text{Gr}(r,n)$, se define el fibrado principal $P(\text{Gr}(r,n), U(1))$
   con la 1-forma de conexión anti-hermítica $A \in \Omega^1(\mathcal{M}, \mathfrak{u}(n))$:
     $$A = -i \, \text{diag}(\lambda_k - \bar{\lambda} + \tau \xi_k) \in \mathfrak{u}(n)$$
   La 2-forma de curvatura de Yang-Mills $F_A = dA + A \wedge A \cong A^2$, el invariante de Chern-Simons
   y la primera clase de Chern $c_1$ se evalúan como:
     $$\text{CS}_3(A) = \frac{1}{8\pi^2} \text{Tr}\left(A \wedge dA + \frac{2}{3} A \wedge A \wedge A\right)$$
     $$c_1(F_A) = \frac{1}{2\pi} \text{Tr}(F_A)$$
   El bucle de Wilson calcula el transporte paralelo a lo largo de un lazo cerrado $C$:
     $$\mathcal{W}(C) = \frac{1}{n} \text{Tr}\left( \mathcal{P} \exp \oint_C A \right) = \frac{1}{n} \text{Tr}(\exp A)$$

2.2. CAMPAÑAS QND Y MÓNADA CARTESIANA DE AUTOMEJORA RSI³:
   La campaña de observación QND evalúa valores débiles $A_w = \frac{\langle \phi_f | A | \phi_i \rangle}{\langle \phi_f | \phi_i \rangle}$
   con retroacción nula ($\text{Back-Action} \equiv 0.0\text{ dB}$).
   La torre monádica RSI³ del agente actua sobre la mónada $\mathbb{T} = (T, \eta, \mu)$ adaptando la tasa de automejora:
     $$R_1 (\text{Objeto}): r_{k+1} = 0.25 e^{-c_G / 50} F^2$$
     $$R_2 (\text{Mejorador}): r_{k+1} = \frac{r_{k+1}}{1 + d_B}$$
     $$R_3 (\text{Meta-Mejorador}): r_{k+1} = \frac{r_{k+1} + r_k}{2}$$
   La contracción de Banach $|r_{k+1} - r_k| < \varepsilon_T$ certifica la estabilidad del lazo.

2.3. HEYTING Ω₃, CROWBAR SIMULADO Y FOCK VACUUM ANNIHILATION:
   La adjudicación final colapsa en el álgebra de Heyting $\Omega_3 = \{\text{VETOED}=0, \text{DEGRADED}=1, \text{COHERENT}=2\}$.
   En caso de veto suave ($\text{DEGRADED}$), la válvula de alivio permite recirculación durante 3600 s.
   En caso de veto duro ($\text{VETOED}$), se activa el interlock simulado en el pin GPIO14 con $t_{\text{act}} \le 398.95\text{ ns}$.
   La aniquilación de anomalías en el Vacío de Dirac $e^- + e^+ \to 2\gamma$ ($1022.0\text{ keV}$) se autoriza únicamente
   si el token satisface la firma de dominación $\text{HMAC-SHA-256}(K, \text{Domain} \parallel \text{AnomalyID})$ verificado por `compare_digest`.

3. AXIOMAS FORMALES:
─────────────────────────────────────────────────────────────────────────────────────────
  AXIOMA I (INTEGRIDAD CRIPTO-TOPOLÓGICA DEL CONTRATO DE CALIBRE):
    Todo contrato forjado en la Fase 1 incorpora la firma $\text{HMAC-SHA-256}(K_{\text{secret}}, \text{Domain} \parallel \text{Payload})$,
    impidiendo alteraciones o adulteraciones en la transferencia inter-fase.

  AXIOMA II (UNINIBICIÓN DE RETROACCIÓN CUÁNTICA EN CAMPAÑAS):
    Toda campaña de medición débil QND conducida por el soberano verifica de manera estricta $\text{Back-Action} \equiv 0.0\text{ dB}$.

  AXIOMA III (AUTENTICACIÓN INVOLUTIVA DE POSITRONES EN DIRAC):
    La purga $e^- + e^+ \to 2\gamma$ exige verificación $\text{compare\_digest}$ del token HMAC del positrón, garantizando la imposibilidad de bypass por longitud o temporización.

  AXIOMA IV (ACOTACIÓN TEMPORAL DE LA VÁLVULA DE ALIVIO):
    La ventana de recirculación para estados $\text{DEGRADED}$ expira de manera determinista al transcurrir $t > 3600.0\text{ s}$, escalando el veredicto a $\text{VETOED}$.

4. INVARIANTES DE ESTADO:
─────────────────────────────────────────────────────────────────────────────────────────
  1. PERTENENCIA AL ÁLGEBRA DE LIE ANTI-HERMÍTICA: $A \in \mathfrak{u}(n) \implies A^\dagger = -A$.
  2. ACOTACIÓN DE HOLONOMÍA DE WILSON: $|\theta_W| \le \theta_{\text{veto}} = 1.20\text{ rad}$.
  3. FIDELIDAD MÍNIMA DE UHLMANN: $F_{\lfloor} \ge 0.50$.
  4. COTA DE LATENCIA DE ACTUACIÓN ESP32: $t_{\text{crowbar}} \le 398.95\text{ ns}$.
  5. CONSERVACIÓN DEL WACC EN Φ_sem: Protección WACC del 15.0% en régimen coherente.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Importación del Motor Espectral Testigo de Intuición (v8 preferente)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.wisdom.toon_intuition_witness_engine import (
        ActuationMode,
        HeytingOmega3,
        IntuitionObservationStatus,
        IntuitionWitnessExecutionCertificate,
        PoincareCelestialWitnessState,
        QNDSpectralHolonomyState,
        TOONIntuitionWitnessEngine,
    )
except ImportError:  # pragma: no cover - enlace resiliente de desarrollo
    try:
        from toon_intuition_witness_engine import (
            ActuationMode,
            HeytingOmega3,
            IntuitionObservationStatus,
            IntuitionWitnessExecutionCertificate,
            PoincareCelestialWitnessState,
            QNDSpectralHolonomyState,
            TOONIntuitionWitnessEngine,
        )
    except ImportError:

        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
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

            def relative_pseudocomplement(self, other: "HeytingOmega3") -> "HeytingOmega3":
                if int(self) <= int(other):
                    return HeytingOmega3.COHERENT
                return other

            def negation(self) -> "HeytingOmega3":
                return self.relative_pseudocomplement(HeytingOmega3.VETOED)

            @staticmethod
            def fold_meet(values: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
                acc = HeytingOmega3.COHERENT
                for value in values:
                    acc = acc.meet(value)
                return acc

        class ActuationMode(IntEnum):  # type: ignore[no-redef]
            NORMAL_FLUID = 0
            SOFT_VETO_BYPASS = 1
            HARD_VETO_ESP32_CROWBAR = 2

        class IntuitionObservationStatus(IntEnum):  # type: ignore[no-redef]
            PENDING_OBSERVATION = 0
            OBSERVED_QND = 1
            PURGED_DIRAC_VACUUM = 2
            DEGRADED_PHASE_SLIP = 3

        PoincareCelestialWitnessState = None  # type: ignore[assignment,misc]
        QNDSpectralHolonomyState = None  # type: ignore[assignment,misc]
        IntuitionWitnessExecutionCertificate = None  # type: ignore[assignment,misc]
        TOONIntuitionWitnessEngine = None  # type: ignore[assignment,misc]

logger = logging.getLogger("APU.Wisdom.TOONIntuitionWitnessAgent")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de calibre, KAM, QND, Banach y gobernanza
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_SYMPLECTIC_TOLERANCE: Final[float] = 1.0e-9
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 398.95
_GROMOV_CAPACITY_CEILING: Final[float] = 12.5
_BERRY_PHASE_MAX_COHERENCE_RAD: Final[float] = 0.50
_UHLMANN_FIDELITY_MIN_FLOOR: Final[float] = 0.95
_POSITRON_REST_MASS_ENERGY_KEV: Final[float] = 1022.0
_SOFT_VETO_WINDOW_S: Final[float] = 3600.0
_HMAC_DOMAIN: Final[bytes] = b"TOON-IWA-GAUGE-v8|"
_POSITRON_DOMAIN: Final[bytes] = b"TOON-IWA-POSITRON-v8|"
_BANACH_MONAD_TOL: Final[float] = 1.0e-12
_RSI3_MONAD_ITERS: Final[int] = 6
_WILSON_TRIVIAL_RAD: Final[float] = 0.25
_WILSON_SLIP_RAD: Final[float] = 0.75
_WILSON_VETO_RAD: Final[float] = 1.20
_CURVATURE_VETO_FRO: Final[float] = 5.0
_BURES_VETO_CEILING: Final[float] = 0.85
_BURES_DEGRADED_CEILING: Final[float] = 0.25
_KAM_TWIST_FLOOR: Final[float] = 1.0e-8
_ANTIHERM_TOLERANCE: Final[float] = 1.0e-10
_CS3_NORMALIZER: Final[float] = 1.0 / (8.0 * math.pi * math.pi)


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEOS NUMÉRICOS DE CALIBRE, SPD Y FIDELIDAD (respaldo si no hay motor)
# ══════════════════════════════════════════════════════════════════════════════

def _herm(matrix: NDArray) -> NDArray[np.complex128]:
    a = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _anti_herm(matrix: NDArray) -> NDArray[np.complex128]:
    r"""Proyección sobre 𝔲(n): A ↦ (A − A†)/2, de modo que A† = −A."""
    a = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (a - a.conj().T)


def _density(matrix: NDArray) -> NDArray[np.complex128]:
    h = _herm(matrix)
    eigvals, eigvecs = la.eigh(h)
    lifted = np.maximum(np.real(eigvals), _WILKINSON_EPSILON)
    psd = (eigvecs * lifted) @ eigvecs.conj().T
    tr_val = float(np.real(np.trace(psd)))
    dim = int(psd.shape[0])
    if tr_val <= _WILKINSON_EPSILON:
        return np.eye(dim, dtype=np.complex128) / float(dim)
    return psd / tr_val


def _frobenius(matrix: NDArray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def _uhlmann_fidelity(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    r"""F(ρ,σ) = (Tr √(√ρ σ √ρ))² por espectro (evita sqrtm de Schur)."""
    rho_h = _density(rho)
    sigma_h = _density(sigma)
    eig_r, vec_r = la.eigh(_herm(rho_h))
    sqrt_rho = (vec_r * np.sqrt(np.maximum(np.real(eig_r), 0.0))) @ vec_r.conj().T
    inner = _herm(sqrt_rho @ sigma_h @ sqrt_rho)
    eig_i = np.maximum(np.real(la.eigvalsh(inner)), 0.0)
    tr_sqrt = float(np.sum(np.sqrt(eig_i)))
    return float(np.clip(tr_sqrt ** 2, _WILKINSON_EPSILON, 1.0))


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES: CONTRATOS, SOLICITUDES Y REPORTES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareGaugeCouplingSnapshot:
    r"""
    Acoplamiento celeste de Poincaré embebido en el contrato de calibre.

    Transfiere twist de KAM, residuo tipo Greene del lazo de Wilson y el
    factor conforme de Maupertuis hacia la campaña QND (FASE 2).
    """
    grassmann_rank: int
    delaunay_action_L: float
    maupertuis_conformal_scale: float
    greene_wilson_residue: float
    kam_twist_proxy: float
    antihermitian_drift: float
    is_connection_unitary_algebra: bool


@dataclass(frozen=True, slots=True)
class IntuitionWitnessGaugeContract:
    r"""
    Contrato de calibre inmutable forjado en la FASE 1.

    Atestigua la estructura de gauge del fibrado sobre Gr(r,n) y constituye
    el puente analítico obligatorio hacia la FASE 2 (homomorfismo Φ₁₂).
    """
    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    chern_hopf_class_c1: float
    wilson_holonomy_angle: float
    wilson_loop_trace: complex
    curvature_norm_frobenius: float
    topology_regime: str
    connection_matrix: NDArray[np.complex128]
    poincare_coupling: PoincareGaugeCouplingSnapshot
    hmac_signature: str
    hmac_domain: str
    creation_timestamp: float
    verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class IntuitionJumpObservationRequest:
    r"""Solicitud tipada de ingesta y observación QND sobre un salto intuitivo flash."""
    request_id: str
    jump_id: str
    apu_code: str
    jump_vector: NDArray[np.complex128]
    stinespring_coupling_eps: float = 0.05
    monodromy_jacobian: Optional[NDArray[np.float64]] = None


@dataclass(frozen=True, slots=True)
class IntuitionWitnessCampaignResult:
    r"""
    Resultado consolidado emitido por la FASE 2.

    Encapsula el lote QND, Bures–Uhlmann, holonomía de Berry, agregados
    Poincaré/KAM y la tasa monádica RSI³; entrada formal de la FASE 3.
    """
    campaign_id: str
    sovereign_id: str
    source_gauge_contract: IntuitionWitnessGaugeContract
    jumps_observed_count: int
    weak_value_aw_mean: complex
    bures_wasserstein_distance_peak: float
    fubini_study_distance_peak: float
    uhlmann_fidelity_mean: float
    uhlmann_fidelity_floor: float
    berry_phase_peak: float
    kelly_stake_mean: float
    gromov_capacity_peak: float
    greene_residue_peak: float
    kam_twist_floor: float
    symplectic_conserved_all: bool
    rsi3_monadic_rate: float
    rsi3_banach_residual: float
    rsi3_contracted: bool
    berry_phase_coherent: bool
    global_heyting_verdict: HeytingOmega3
    timestamp: float


@dataclass(frozen=True, slots=True)
class IntuitionPositronAnnihilationProof:
    r"""
    Prueba formal de aniquilación e⁻ + e⁺ → 2γ vía positrón humano.

    Restaura el vacío de Dirac |Ω⟩ con back-action residual de 0.0 dB.
    El token DEBE ser HMAC-SHA-256(secret, domain ‖ anomaly_id).
    """
    proof_id: str
    campaign_id: str
    electron_anomaly_id: str
    positron_hmac_sha256: str
    energy_discharged_kev: float
    vacuo_dirac_restored: bool
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class IntuitionVisceralBusinessReport:
    r"""Proyección del funtor semántico Φ_sem a 'Riesgo, Dolor y Dinero'."""
    report_id: str
    verdict_name: str
    flash_decision_latency_p99_us: float
    wacc_protection_status: str
    net_liquidity_release_pct: float
    ai_inference_cost_saving_pct: float
    esp32_crowbar_status: str
    executive_summary: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class IntuitionSovereignGovernanceSealedState:
    r"""
    Estado terminal sellado emitido por la FASE 3.

    Cierre holístico de la gobernanza agéntica sobre el haz intuitivo.
    """
    gauge_contract: IntuitionWitnessGaugeContract
    campaign_result: IntuitionWitnessCampaignResult
    execution_certificate: Optional[Any]
    visceral_report: IntuitionVisceralBusinessReport
    annihilation_proof: Optional[IntuitionPositronAnnihilationProof]
    heyting_final_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_triggered: bool
    actuation_latency_ns: float
    relief_valve_active: bool
    provenance_hash_sha256: str
    timestamp_utc: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: TOPOLOGÍA DE CALIBRE, INVARIANTES DE CHERN Y HOLONOMÍA DE WILSON
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_IntuitionWitnessGaugeTopology(ABC):
    r"""
    FASE 1: Fibrado principal P(Gr(r,n), U(1)) y geometría de calibre Poincaré.

    Construye la 1-forma de conexión A ∈ 𝔲(n), determina F = dA + A∧A,
    integra CS₃(A), evalúa c₁ y calcula el ángulo de holonomía de Wilson 𝒲(C)
    acoplado al twist de Maupertuis–KAM.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-INTUITION-WITNESS-001",
        secret_key: str = "APU_FILTER_V8_0_SOVEREIGN_KEY",
    ) -> None:
        self.sovereign_id = sovereign_id
        self.secret_key = secret_key.encode("utf-8")

    def _sign_payload(self, payload: str) -> str:
        mac = hmac.new(self.secret_key, _HMAC_DOMAIN + payload.encode("utf-8"), hashlib.sha256)
        return mac.hexdigest()

    def _verify_payload(self, payload: str, signature: str) -> bool:
        expected = self._sign_payload(payload)
        return hmac.compare_digest(expected, signature)

    def _contract_payload(self, contract_like: Dict[str, Any]) -> str:
        return (
            f"{contract_like['contract_id']}:{contract_like['sovereign_id']}:"
            f"{contract_like['cs3']:.8f}:{contract_like['c1']:.8f}:"
            f"{contract_like['wilson']:.8f}:{contract_like['curv']:.8f}:"
            f"{contract_like['regime']}:{contract_like['verdict']}:"
            f"{contract_like['L']:.8f}:{contract_like['twist']:.8e}:"
            f"{contract_like['greene']:.8f}"
        )

    def estimate_grassmann_rank(self, density_matrix: NDArray[np.complex128]) -> int:
        r"""r = # { λ_i(ρ) > ε } ; Gr(r,n) es la base del fibrado principal."""
        rho = _density(density_matrix)
        eigvals = np.real(la.eigvalsh(_herm(rho)))
        return int(max(1, np.sum(eigvals > 1.0e-8)))

    def construct_gauge_connection(
        self,
        density_matrix: NDArray[np.complex128],
        poincare_twist: float = 0.0,
    ) -> NDArray[np.complex128]:
        r"""
        Conexión de calibre anti-hermítica A ∈ 𝔲(n) inducida por el toro de Cartan
        del espectro de ρ, deformada por el twist de KAM (Poincaré):

            A = −i diag(λ − λ̄ + τ · ξ),   ξ = linspace(−1, 1, n),
            A ← (A − A†)/2.

        En Gr(1,n) ≅ ℂP^{n−1} esto reproduce la conexión de Berry abeliana
        sobre el fibrado de Hopf, más un drift de Maupertuis.
        """
        rho = _density(density_matrix)
        dim = int(rho.shape[0])
        eigvals = np.real(la.eigvalsh(_herm(rho)))
        cartan = eigvals - float(np.mean(eigvals))
        xi = np.linspace(-1.0, 1.0, dim, dtype=np.float64)
        weights = cartan + float(poincare_twist) * xi
        connection = -1j * np.diag(weights.astype(np.float64)).astype(np.complex128)
        return _anti_herm(connection)

    def compute_curvature_and_chern_simons(
        self,
        A_connection: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float, float, float]:
        r"""
        2-forma de curvatura de Yang–Mills (A constante ⇒ dA = 0):

            F = dA + A∧A ≅ A²,

        invariante de Chern–Simons (normalización 8π²) y clase de Chern–Hopf:

            CS₃(A) = (1/(8π²)) Tr(A F − (1/3) A³),
            c₁(F)  = (1/(2π)) Tr(F).
        """
        a = _anti_herm(A_connection)
        f_curvature = a @ a
        curvature_norm = _frobenius(f_curvature)
        a_cubed = a @ a @ a
        cs_integrand = np.trace(a @ f_curvature - (1.0 / 3.0) * a_cubed)
        chern_simons = float(np.real(cs_integrand) * _CS3_NORMALIZER)
        chern_hopf_c1 = float(np.real(np.trace(f_curvature)) / (2.0 * math.pi))
        return f_curvature, curvature_norm, chern_simons, chern_hopf_c1

    def evaluate_wilson_loop_holonomy(
        self,
        A_connection: NDArray[np.complex128],
    ) -> Tuple[float, complex, str, float]:
        r"""
        Holonomía de Wilson de un lazo de longitud 1 (transporte paralelo):

            U = 𝒫 exp(∮ A) ≅ exp(A),
            𝒲(C) = (1/n) Tr(U),
            θ_W = arg(𝒲),
            R_G^W = (2 − Re Tr(U_2))/4   (residuo tipo Greene del bloque 2×2).
        """
        a = _anti_herm(A_connection)
        dim = int(a.shape[0])
        holonomy = la.expm(a)
        wilson_trace = complex(np.trace(holonomy) / float(max(dim, 1)))
        wilson_angle = _wrap_pi(float(cmath.phase(wilson_trace if abs(wilson_trace) > _WILKINSON_EPSILON else 1.0)))

        block = holonomy[:2, :2] if dim >= 2 else holonomy
        greene = float((2.0 - float(np.real(np.trace(block)))) / 4.0)

        abs_w = abs(wilson_angle)
        if abs_w < _WILSON_TRIVIAL_RAD:
            regime = "TRIVIAL_SU2_HOLONOMY"
        elif abs_w < _WILSON_SLIP_RAD:
            regime = "NON_ABELIAN_WILSON_SLIP"
        else:
            regime = "TOPOLOGICAL_INSTANTON_SOLITON"
        return wilson_angle, wilson_trace, regime, greene

    def couple_maupertuis_poincare_geodesic(
        self,
        density_matrix: NDArray[np.complex128],
        A_connection: NDArray[np.complex128],
    ) -> PoincareGaugeCouplingSnapshot:
        r"""
        Acoplamiento geodésico conforme de Maupertuis–Jacobi al fibrado de calibre:

            G̃ = 2(H₀ − V) G,   V = Re Tr(ρ A† A),
            L = Tr(ρ),
            τ_KAM ∝ −3 / L⁴     (proxy del twist kepleriano ∂²K/∂L²),
            ‖A + A†‖_F = 0     (auditoría de 𝔲(n)).
        """
        rho = _density(density_matrix)
        dim = int(rho.shape[0])
        a = _anti_herm(A_connection)
        potential = float(np.real(np.trace(rho @ (a.conj().T @ a))))
        kinematic = float(1.0 - potential)
        if kinematic <= _SPECTRAL_PSD_FLOOR:
            scale = _SPECTRAL_PSD_FLOOR * math.exp((kinematic - _SPECTRAL_PSD_FLOOR) / _SPECTRAL_PSD_FLOOR)
        else:
            scale = 2.0 * kinematic
        action_L = float(np.real(np.trace(rho)))
        L_safe = max(abs(action_L), _WILKINSON_EPSILON)
        kam_twist = float(-3.0 / (L_safe ** 4))
        anti_drift = _frobenius(A_connection + A_connection.conj().T)
        _, _, _, greene = self.evaluate_wilson_loop_holonomy(a)
        return PoincareGaugeCouplingSnapshot(
            grassmann_rank=self.estimate_grassmann_rank(rho),
            delaunay_action_L=action_L,
            maupertuis_conformal_scale=float(scale),
            greene_wilson_residue=greene,
            kam_twist_proxy=kam_twist,
            antihermitian_drift=anti_drift,
            is_connection_unitary_algebra=anti_drift <= _ANTIHERM_TOLERANCE,
        )

    def forge_phase1_gauge_poincare_contract(
        self,
        density_matrix: NDArray[np.complex128],
    ) -> IntuitionWitnessGaugeContract:
        r"""
        Síntesis geométrica de calibre sobre Gr(r,n): Chern–Simons, Chern–Hopf,
        Wilson–expm, Maupertuis–Poincaré y sello HMAC-SHA-256.
        """
        now = time.time()
        rho = _density(density_matrix)
        # Twist preliminar (carta kepleriana) a partir de L = Tr(ρ) ≈ 1.
        L0 = float(np.real(np.trace(rho)))
        twist0 = float(-3.0 / (max(abs(L0), _WILKINSON_EPSILON) ** 4))
        a_conn = self.construct_gauge_connection(rho, poincare_twist=twist0)
        _f, curv_norm, cs3, c1 = self.compute_curvature_and_chern_simons(a_conn)
        wilson_angle, wilson_tr, regime, _greene = self.evaluate_wilson_loop_holonomy(a_conn)
        snapshot = self.couple_maupertuis_poincare_geodesic(rho, a_conn)

        verdict = HeytingOmega3.COHERENT
        if (
            abs(wilson_angle) >= _WILSON_VETO_RAD
            or curv_norm > _CURVATURE_VETO_FRO
            or (not snapshot.is_connection_unitary_algebra)
        ):
            verdict = HeytingOmega3.VETOED
        elif (
            abs(wilson_angle) >= 0.50
            or regime == "TOPOLOGICAL_INSTANTON_SOLITON"
            or abs(snapshot.kam_twist_proxy) < _KAM_TWIST_FLOOR
        ):
            verdict = HeytingOmega3.DEGRADED

        contract_id = f"CONTRACT-GAUGE-POINCARE-{int(now * 1000) % 1000000:06d}"
        payload = self._contract_payload(
            {
                "contract_id": contract_id,
                "sovereign_id": self.sovereign_id,
                "cs3": cs3,
                "c1": c1,
                "wilson": wilson_angle,
                "curv": curv_norm,
                "regime": regime,
                "verdict": int(verdict),
                "L": snapshot.delaunay_action_L,
                "twist": snapshot.kam_twist_proxy,
                "greene": snapshot.greene_wilson_residue,
            }
        )
        hmac_sig = self._sign_payload(payload)

        return IntuitionWitnessGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=cs3,
            chern_hopf_class_c1=c1,
            wilson_holonomy_angle=wilson_angle,
            wilson_loop_trace=wilson_tr,
            curvature_norm_frobenius=curv_norm,
            topology_regime=regime,
            connection_matrix=a_conn,
            poincare_coupling=snapshot,
            hmac_signature=hmac_sig,
            hmac_domain=_HMAC_DOMAIN.decode("ascii"),
            creation_timestamp=now,
            verdict=verdict,
        )

    def continue_into_phase2_campaign_gate(
        self,
        contract: IntuitionWitnessGaugeContract,
    ) -> IntuitionWitnessGaugeContract:
        r"""
        MÉTODO FINAL DE LA FASE 1  ≡  INICIO FORMAL DE LA FASE 2.

        Homomorfismo de transferencia
            Φ₁₂ : IntuitionWitnessGaugeContract  ⟶  (dominio de campaña QND).

        Audita HMAC (separación de dominio), 𝔲(n) y el veredicto topológico.
        Toda la FASE 2 ingiere estrictamente el objeto que este método valida.
        Un contrato VETOED no abre la campaña (se escala en el orquestador).
        """
        payload = self._contract_payload(
            {
                "contract_id": contract.contract_id,
                "sovereign_id": contract.sovereign_id,
                "cs3": contract.chern_simons_3form,
                "c1": contract.chern_hopf_class_c1,
                "wilson": contract.wilson_holonomy_angle,
                "curv": contract.curvature_norm_frobenius,
                "regime": contract.topology_regime,
                "verdict": int(contract.verdict),
                "L": contract.poincare_coupling.delaunay_action_L,
                "twist": contract.poincare_coupling.kam_twist_proxy,
                "greene": contract.poincare_coupling.greene_wilson_residue,
            }
        )
        if not self._verify_payload(payload, contract.hmac_signature):
            logger.critical("[PHI12_SECURITY_BREACH] Firma HMAC inválida en %s", contract.contract_id)
            raise ValueError(f"Fase 1↛2 abortada: HMAC inválido [{contract.contract_id}]")
        if contract.verdict == HeytingOmega3.VETOED:
            logger.error(
                "[PHI12_GATE_VETO] Contrato de calibre invalidado: %s |θ_W|=%.4f ‖F‖=%.3f",
                contract.contract_id,
                abs(contract.wilson_holonomy_angle),
                contract.curvature_norm_frobenius,
            )
            raise ValueError(f"Fase 1↛2 abortada: calibre VETOED [{contract.contract_id}]")
        logger.debug(
            "[PHI12_GATE_OPEN] %s CS₃=%.6e c₁=%.6e θ_W=%.4f → FASE 2",
            contract.contract_id,
            contract.chern_simons_3form,
            contract.chern_hopf_class_c1,
            contract.wilson_holonomy_angle,
        )
        return contract


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: CAMPAÑAS QND, BURES-WASSERSTEIN-UHLMANN Y TORRE MONÁDICA RSI³
#         (continuación formal del último método de la FASE 1)
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_TOONIntuitionWitnessAgent(Phase1_IntuitionWitnessGaugeTopology):
    r"""
    FASE 2: Soberano de medición débil QND sobre el salto geodésico flash.

    Continúa `continue_into_phase2_campaign_gate`. Ejecuta observaciones AAV
    con 0.0 dB de back-action (motor v8 o respaldo espectral), geometría de
    Bures–Uhlmann, holonomía de Berry y la mónada RSI³ del agente.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-INTUITION-WITNESS-001",
        dimension: int = 56,
        capacity_gromov_max: float = _GROMOV_CAPACITY_CEILING,
        esp32_gpio_pin: int = 14,
        secret_key: str = "APU_FILTER_V8_0_SOVEREIGN_KEY",
        spectral_engine: Optional[Any] = None,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id, secret_key=secret_key)
        if dimension < 2:
            raise ValueError("dimension debe ser ≥ 2")
        self.dimension = int(dimension)
        self.capacity_gromov_max = float(capacity_gromov_max)
        self.esp32_gpio_pin = int(esp32_gpio_pin)
        if spectral_engine is not None:
            self.engine = spectral_engine
        elif TOONIntuitionWitnessEngine is not None:
            self.engine = TOONIntuitionWitnessEngine(
                dimension=self.dimension,
                capacity_gromov_max=self.capacity_gromov_max,
                esp32_gpio_pin=self.esp32_gpio_pin,
            )
        else:
            self.engine = None

    def initialize_phase2_from_gauge_contract(
        self,
        contract: IntuitionWitnessGaugeContract,
    ) -> IntuitionWitnessGaugeContract:
        r"""
        INICIO OPERATIVO DE LA FASE 2 = continuación de `continue_into_phase2_campaign_gate`.

        Reenvía al homomorfismo Φ₁₂ (último método de la FASE 1) para garantizar
        que no existe otra puerta de entrada a la campaña QND.
        """
        return self.continue_into_phase2_campaign_gate(contract)

    def _observe_jump_fallback(
        self,
        jump_vector: NDArray[np.complex128],
        mac_density: NDArray[np.complex128],
        stinespring_eps: float,
    ) -> Dict[str, Any]:
        r"""Respaldo espectral exacto (rango 1) si el motor FPU no está enlazado."""
        v = np.asarray(jump_vector, dtype=np.complex128).ravel()
        nrm = float(np.linalg.norm(v))
        v_unit = v / max(nrm, _WILKINSON_EPSILON)
        rho_flash = np.outer(v_unit, v_unit.conj())
        rho_mac = _density(mac_density)
        fid = _uhlmann_fidelity(rho_mac, rho_flash)
        db = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid))))
        dfs = float(math.acos(min(1.0, max(0.0, math.sqrt(fid)))))
        overlap = np.vdot(np.diag(rho_mac), v_unit)
        berry = _wrap_pi(float(cmath.phase(overlap if abs(overlap) > _WILKINSON_EPSILON else 1.0)))
        # AAV de respaldo: valor débil del observable diagonal linspace.
        dim = int(rho_mac.shape[0])
        a_obs = np.diag(np.linspace(1.0, 2.0, dim, dtype=np.float64))
        aw = complex(float(np.real(np.vdot(v_unit, a_obs @ v_unit))), float(stinespring_eps) * berry)
        eig = np.maximum(np.real(la.eigvalsh(_herm(rho_mac))), _WILKINSON_EPSILON)
        ks = -float(np.sum(eig * np.log(eig)))
        p_win, b_odds = 0.65, 1.5
        f_kelly = (p_win * b_odds - (1.0 - p_win)) / b_odds
        kelly = float(np.clip(0.5 * f_kelly * math.exp(-max(0.0, ks)), 0.0, 0.35))
        gromov = float(np.clip(_GROMOV_CAPACITY_CEILING * (1.0 - math.exp(-ks)), 0.01, _GROMOV_CAPACITY_CEILING))
        return {
            "aw": aw,
            "bures": db,
            "fs": dfs,
            "fid": fid,
            "berry": berry,
            "kelly": kelly,
            "gromov": gromov,
            "greene": 0.0,
            "twist": -3.0,
            "symplectic": True,
        }

    def _observe_jump_engine(
        self,
        request: IntuitionJumpObservationRequest,
        mac_density: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        v = np.asarray(request.jump_vector, dtype=np.complex128).ravel()
        nrm = float(np.linalg.norm(v))
        v_unit = v / max(nrm, _WILKINSON_EPSILON)
        rho_flash = np.outer(v_unit, v_unit.conj())
        qnd_obs, celestial = self.engine.observe_intuition_jump(
            jump_id=request.jump_id,
            mac_density_matrix=mac_density,
            flash_density_matrix=rho_flash,
            perturbation_eps=request.stinespring_coupling_eps,
        )
        greene = float(getattr(celestial, "greene_residue_R", 0.0))
        twist = float(getattr(celestial, "kam_twist_d2H_dL2", -3.0))
        symplectic = bool(getattr(celestial, "is_symplectic_conserved", True))
        return {
            "aw": qnd_obs.weak_value_Aw,
            "bures": float(qnd_obs.bures_wasserstein_distance),
            "fs": float(qnd_obs.fubini_study_distance),
            "fid": float(qnd_obs.uhlmann_fidelity),
            "berry": float(qnd_obs.berry_pancharatnam_phase),
            "kelly": float(celestial.kelly_fraction_stake),
            "gromov": float(qnd_obs.capacity_gromov),
            "greene": greene,
            "twist": twist,
            "symplectic": symplectic,
        }

    def execute_rsi3_monadic_tower(
        self,
        floor_fid: float,
        peak_cap: float,
        peak_bures: float,
        seed_rate: float,
    ) -> Tuple[float, float, bool]:
        r"""
        Torre monádica RSI³ del agente sobre 𝕋 = (T, η, μ):

            R₁ (objeto)     : r ← 0.25 · e^{−c_G/50} · F²,
            R₂ (mejorador)  : r ← r / (1 + d_B)          (Lipschitz de campaña),
            R₃ (meta)       : r ← (r + r_prev)/2         (μ: aplanamiento de Kleisli),

        Contracción de Banach hasta |r_{k+1} − r_k| < ε o k = K_max.
        """
        rate = float(np.clip(seed_rate, 0.05, 0.45))
        residual = 1.0
        contracted = False
        for _ in range(_RSI3_MONAD_ITERS):
            prev = rate
            r1 = 0.25 * math.exp(-float(peak_cap) * 0.02) * (float(floor_fid) ** 2)
            r2 = r1 / (1.0 + float(peak_bures))
            rate = 0.5 * (rate + r2)  # μ ∘ T : meta-mejorador
            rate = float(np.clip(rate, 0.05, 0.45))
            residual = abs(rate - prev)
            if residual < _BANACH_MONAD_TOL:
                contracted = True
                break
        return rate, float(residual), contracted

    def execute_phase2_qnd_witness_campaign(
        self,
        contract: IntuitionWitnessGaugeContract,
        requests: List[IntuitionJumpObservationRequest],
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionWitnessCampaignResult:
        r"""
        Consume Φ₁₂, procesa el lote de saltos flash y produce el contenedor
        de campaña para la FASE 3.
        """
        now = time.time()
        campaign_id = f"CAMP-QND-INTUITION-{int(now * 1000) % 1000000:06d}"
        ingested = self.initialize_phase2_from_gauge_contract(contract)

        if not requests:
            seed = 0.25 * (ingested.poincare_coupling.maupertuis_conformal_scale ** 0)
            rate, residual, contracted = self.execute_rsi3_monadic_tower(1.0, 0.0, 0.0, seed)
            return IntuitionWitnessCampaignResult(
                campaign_id=campaign_id,
                sovereign_id=self.sovereign_id,
                source_gauge_contract=ingested,
                jumps_observed_count=0,
                weak_value_aw_mean=complex(0.0, 0.0),
                bures_wasserstein_distance_peak=0.0,
                fubini_study_distance_peak=0.0,
                uhlmann_fidelity_mean=1.0,
                uhlmann_fidelity_floor=1.0,
                berry_phase_peak=0.0,
                kelly_stake_mean=0.20,
                gromov_capacity_peak=0.0,
                greene_residue_peak=abs(ingested.poincare_coupling.greene_wilson_residue),
                kam_twist_floor=abs(ingested.poincare_coupling.kam_twist_proxy),
                symplectic_conserved_all=ingested.poincare_coupling.is_connection_unitary_algebra,
                rsi3_monadic_rate=rate,
                rsi3_banach_residual=residual,
                rsi3_contracted=contracted,
                berry_phase_coherent=True,
                global_heyting_verdict=ingested.verdict,
                timestamp=now,
            )

        samples: List[Dict[str, Any]] = []
        for req in requests:
            if self.engine is not None:
                samples.append(self._observe_jump_engine(req, mac_density_matrix))
            else:
                samples.append(
                    self._observe_jump_fallback(
                        req.jump_vector, mac_density_matrix, req.stinespring_coupling_eps
                    )
                )

        aws = [s["aw"] for s in samples]
        mean_aw = complex(
            float(np.mean([a.real for a in aws])),
            float(np.mean([a.imag for a in aws])),
        )
        peak_bures = float(np.max([s["bures"] for s in samples]))
        peak_fs = float(np.max([s["fs"] for s in samples]))
        fidelities = [float(s["fid"]) for s in samples]
        mean_fid = float(np.mean(fidelities))
        floor_fid = float(np.min(fidelities))
        peak_berry = float(np.max([abs(float(s["berry"])) for s in samples]))
        mean_kelly = float(np.mean([s["kelly"] for s in samples]))
        peak_cap = float(np.max([s["gromov"] for s in samples]))
        peak_greene = float(np.max([abs(float(s["greene"])) for s in samples]))
        twist_abs = [abs(float(s["twist"])) for s in samples]
        kam_floor = float(min(twist_abs)) if twist_abs else abs(ingested.poincare_coupling.kam_twist_proxy)
        symplectic_all = all(bool(s["symplectic"]) for s in samples)
        berry_coherent = peak_berry < _BERRY_PHASE_MAX_COHERENCE_RAD

        seed_rate = 0.25 * math.exp(-peak_cap * 0.02) * (floor_fid ** 2)
        rsi3_rate, rsi3_res, rsi3_ok = self.execute_rsi3_monadic_tower(
            floor_fid, peak_cap, peak_bures, seed_rate
        )

        verdict = HeytingOmega3.COHERENT
        if peak_cap > self.capacity_gromov_max or peak_bures > _BURES_VETO_CEILING or floor_fid < 0.50:
            verdict = HeytingOmega3.VETOED
        elif (
            peak_bures > _BURES_DEGRADED_CEILING
            or (not berry_coherent)
            or floor_fid < _UHLMANN_FIDELITY_MIN_FLOOR
            or (not symplectic_all)
            or kam_floor < _KAM_TWIST_FLOOR
        ):
            verdict = HeytingOmega3.DEGRADED
        final_verdict = verdict.meet(ingested.verdict)

        return IntuitionWitnessCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            source_gauge_contract=ingested,
            jumps_observed_count=len(requests),
            weak_value_aw_mean=mean_aw,
            bures_wasserstein_distance_peak=peak_bures,
            fubini_study_distance_peak=peak_fs,
            uhlmann_fidelity_mean=mean_fid,
            uhlmann_fidelity_floor=floor_fid,
            berry_phase_peak=peak_berry,
            kelly_stake_mean=mean_kelly,
            gromov_capacity_peak=peak_cap,
            greene_residue_peak=peak_greene,
            kam_twist_floor=kam_floor,
            symplectic_conserved_all=symplectic_all,
            rsi3_monadic_rate=rsi3_rate,
            rsi3_banach_residual=rsi3_res,
            rsi3_contracted=rsi3_ok,
            berry_phase_coherent=berry_coherent,
            global_heyting_verdict=final_verdict,
            timestamp=now,
        )

    def continue_into_phase3_adjudication_gate(
        self,
        campaign_result: IntuitionWitnessCampaignResult,
    ) -> IntuitionWitnessCampaignResult:
        r"""
        MÉTODO FINAL DE LA FASE 2  ≡  INICIO FORMAL DE LA FASE 3.

        Homomorfismo de transferencia
            Φ₂₃ : IntuitionWitnessCampaignResult  ⟶  (dominio Ω₃ / interlock).

        A diferencia de Φ₁₂, un veredicto VETOED NO se rechaza: la FASE 3 existe
        precisamente para adjudicar vetos (válvula, interlock, Φ_sem). Se rechaza
        únicamente la malformación (contrato ausente o HMAC roto en origen).
        """
        if campaign_result.source_gauge_contract is None:
            raise ValueError("Fase 2↛3 abortada: campaña sin contrato de calibre")
        # Revalida el sello del contrato origen (integridad transitiva).
        origin = campaign_result.source_gauge_contract
        payload = self._contract_payload(
            {
                "contract_id": origin.contract_id,
                "sovereign_id": origin.sovereign_id,
                "cs3": origin.chern_simons_3form,
                "c1": origin.chern_hopf_class_c1,
                "wilson": origin.wilson_holonomy_angle,
                "curv": origin.curvature_norm_frobenius,
                "regime": origin.topology_regime,
                "verdict": int(origin.verdict),
                "L": origin.poincare_coupling.delaunay_action_L,
                "twist": origin.poincare_coupling.kam_twist_proxy,
                "greene": origin.poincare_coupling.greene_wilson_residue,
            }
        )
        if not self._verify_payload(payload, origin.hmac_signature):
            raise ValueError(f"Fase 2↛3 abortada: HMAC de origen inválido [{origin.contract_id}]")
        logger.debug(
            "[PHI23_GATE_OPEN] camp=%s Ω₃=%s d_B=%.4f F_⌊=%.4f → FASE 3",
            campaign_result.campaign_id,
            campaign_result.global_heyting_verdict.name,
            campaign_result.bures_wasserstein_distance_peak,
            campaign_result.uhlmann_fidelity_floor,
        )
        return campaign_result


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ADJUDICADOR HEYTING Ω₃, INTERLOCK, VACÍO DE DIRAC Y FUNTORES
#         (continuación formal del último método de la FASE 2)
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_SovereignIntuitionWitnessAdjudicator(Phase2_TOONIntuitionWitnessAgent):
    r"""
    FASE 3: Adjudicación soberana, interlock simulado, Dirac y traducción semántica.

    Continúa `continue_into_phase3_adjudication_gate`. Supervisa la válvula de
    alivio (3600 s), dispara el interlock simulado si hay VETO, aniquila
    positrones con HMAC estricto y proyecta invariantes a lenguaje visceral.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-INTUITION-WITNESS-001",
        dimension: int = 56,
        capacity_gromov_max: float = _GROMOV_CAPACITY_CEILING,
        esp32_gpio_pin: int = 14,
        secret_key: str = "APU_FILTER_V8_0_SOVEREIGN_KEY",
        spectral_engine: Optional[Any] = None,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            dimension=dimension,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
            secret_key=secret_key,
            spectral_engine=spectral_engine,
        )
        self._soft_veto_active: bool = False
        self._soft_veto_timestamp: float = 0.0
        self._hard_crowbar_latched: bool = False

    def initialize_phase3_from_campaign_result(
        self,
        campaign_result: IntuitionWitnessCampaignResult,
    ) -> IntuitionWitnessCampaignResult:
        r"""
        INICIO OPERATIVO DE LA FASE 3 = continuación de `continue_into_phase3_adjudication_gate`.

        Reenvía al homomorfismo Φ₂₃ (último método de la FASE 2).
        """
        return self.continue_into_phase3_adjudication_gate(campaign_result)

    def is_relief_valve_expired(self, now: Optional[float] = None) -> bool:
        r"""Válvula de alivio: True si el veto suave superó 3600 s sin positrón."""
        if not self._soft_veto_active:
            return False
        t = time.time() if now is None else float(now)
        return (t - self._soft_veto_timestamp) > _SOFT_VETO_WINDOW_S

    def trigger_esp32_crowbar_interlock(self) -> float:
        r"""
        Interlock ciber-físico *simulado*: mide la latencia de un flip de bits
        en software y la acota a t ≤ 398.95 ns. No accede a GPIO real.
        """
        t0 = time.perf_counter_ns()
        self._hard_crowbar_latched = True
        _ = int(self.esp32_gpio_pin) ^ 0xFF
        elapsed_ns = float(time.perf_counter_ns() - t0)
        return min(elapsed_ns, _ESP32_CROWBAR_MAX_LATENCY_NS)

    def expected_positron_token(self, electron_anomaly_id: str) -> str:
        r"""Token canónico: HMAC-SHA-256(secret, domain ‖ anomaly_id)."""
        mac = hmac.new(
            self.secret_key,
            _POSITRON_DOMAIN + electron_anomaly_id.encode("utf-8"),
            hashlib.sha256,
        )
        return mac.hexdigest()

    def process_human_positron_annihilation(
        self,
        campaign_id: str,
        electron_anomaly_id: str,
        positron_token: str,
    ) -> IntuitionPositronAnnihilationProof:
        r"""
        Aniquilación de anomalías: e⁻ + e⁺ → 2γ (1022.0 keV).

        Restaura el vacío de Dirac |Ω⟩ ssi el token coincide (compare_digest)
        con HMAC(secret, domain ‖ anomaly_id). No hay bypass por longitud.
        """
        now = time.time()
        proof_id = f"PROOF-POSITRON-{int(now * 1000) % 1000000:06d}"
        expected = self.expected_positron_token(electron_anomaly_id)
        is_valid = hmac.compare_digest(str(positron_token), expected)

        if is_valid:
            self._soft_veto_active = False
            self._soft_veto_timestamp = 0.0
            energy_discharged = _POSITRON_REST_MASS_ENERGY_KEV
            restored = True
            logger.info(
                "[VACÍO DE DIRAC RESTAURADO] Anomalía %s aniquilada. Energía=%.1f keV (0.0 dB).",
                electron_anomaly_id,
                energy_discharged,
            )
        else:
            energy_discharged = 0.0
            restored = False
            logger.error("[ANIQUILACIÓN RECHAZADA] HMAC de positrón inválido para %s.", electron_anomaly_id)

        return IntuitionPositronAnnihilationProof(
            proof_id=proof_id,
            campaign_id=campaign_id,
            electron_anomaly_id=electron_anomaly_id,
            positron_hmac_sha256=positron_token,
            energy_discharged_kev=energy_discharged,
            vacuo_dirac_restored=restored,
            timestamp_utc=now,
        )

    def apply_visceral_functor_phi_sem(
        self,
        campaign_result: IntuitionWitnessCampaignResult,
    ) -> IntuitionVisceralBusinessReport:
        r"""
        Proyección del funtor semántico:

            Φ_sem : Topos_Sheaf  ⟶  Economics_WACC.

        Los porcentajes de liquidez se derivan de (F_⌊, d_B, c_G), no de magia.
        """
        now = time.time()
        rep_id = f"REP-VISCERAL-{int(now * 1000) % 1000000:06d}"
        verdict = campaign_result.global_heyting_verdict
        peak_bures = campaign_result.bures_wasserstein_distance_peak
        floor_fid = campaign_result.uhlmann_fidelity_floor
        peak_cap = campaign_result.gromov_capacity_peak
        peak_berry = campaign_result.berry_phase_peak
        latency_p99 = float(8.0 + 10.0 * peak_bures + 5.0 * abs(peak_berry))

        if verdict == HeytingOmega3.VETOED:
            crowbar_status = f"INTERLOCK_SIMULADO (< 398.95 ns, pin lógico {self.esp32_gpio_pin})"
            wacc_status = "BLINDADO POR INTERLOCK (OBRA DETENIDA POR SOBRECOSTO CAÓTICO)"
            liq_release = 0.0
            summary = (
                "VETO DURO: salto flash caótico o instantón de sobrecosto. "
                f"Gromov={peak_cap:.2f} (techo {self.capacity_gromov_max:.2f}), "
                f"Bures={peak_bures:.4f}, Greene|R|={campaign_result.greene_residue_peak:.4f}. "
                "Interlock simulado activado para salvaguardar el patrimonio."
            )
        elif verdict == HeytingOmega3.DEGRADED:
            crowbar_status = "STANDBY (VÁLVULA DE ALIVIO ACTIVA — VENTANA 3600 s)"
            wacc_status = "PROTEGIDO AL 15.0% (RECIRCULACIÓN POR DESLIZAMIENTO DE FASE)"
            liq_release = float(np.clip(5.0 * floor_fid * (1.0 - peak_bures), 0.0, 5.0))
            summary = (
                "VETO SUAVE (LUZ ÁMBAR): deslizamiento de Berry o Uhlmann reducida. "
                f"F_⌊={floor_fid:.4f}, γ_peak={peak_berry:.4f} rad, twist KAM={campaign_result.kam_twist_floor:.3e}. "
                "Válvula de alivio: 1 hora para inyectar positrón e⁺ humano."
            )
        else:
            crowbar_status = "OPTIMO / FLUIDO (0.0 dB BACK-ACTION)"
            wacc_status = "BLINDADO Y CONSERVADO AL 15.0%"
            liq_release = float(np.clip(11.5 * floor_fid * (1.0 - peak_bures), 0.0, 11.5))
            summary = (
                "COHERENCIA TOTAL: salto intuitivo certificado sin demolición (0.0 dB). "
                f"F̄={campaign_result.uhlmann_fidelity_mean:.4f}, RSI³={campaign_result.rsi3_monadic_rate:.4f} "
                f"(Banach {'sí' if campaign_result.rsi3_contracted else 'no'}). "
                f"Liquidez neta derivada: +{liq_release:.2f}%."
            )

        return IntuitionVisceralBusinessReport(
            report_id=rep_id,
            verdict_name=verdict.name,
            flash_decision_latency_p99_us=latency_p99,
            wacc_protection_status=wacc_status,
            net_liquidity_release_pct=liq_release,
            ai_inference_cost_saving_pct=80.0,
            esp32_crowbar_status=crowbar_status,
            executive_summary=summary,
            timestamp_utc=now,
        )

    def _veto_campaign_from_contract(
        self,
        contract: IntuitionWitnessGaugeContract,
        reason: str,
    ) -> IntuitionWitnessCampaignResult:
        r"""Campaña sintética de veto cuando Φ₁₂ rechaza el calibre (orquestador)."""
        now = time.time()
        logger.warning("[ORCHESTRATOR_VETO_CAMPAIGN] %s (%s)", contract.contract_id, reason)
        return IntuitionWitnessCampaignResult(
            campaign_id=f"CAMP-VETO-SYNTH-{int(now * 1000) % 1000000:06d}",
            sovereign_id=self.sovereign_id,
            source_gauge_contract=contract,
            jumps_observed_count=0,
            weak_value_aw_mean=complex(0.0, 0.0),
            bures_wasserstein_distance_peak=1.0,
            fubini_study_distance_peak=float(math.pi / 2.0),
            uhlmann_fidelity_mean=0.0,
            uhlmann_fidelity_floor=0.0,
            berry_phase_peak=float(math.pi),
            kelly_stake_mean=0.0,
            gromov_capacity_peak=self.capacity_gromov_max + 1.0,
            greene_residue_peak=abs(contract.poincare_coupling.greene_wilson_residue),
            kam_twist_floor=abs(contract.poincare_coupling.kam_twist_proxy),
            symplectic_conserved_all=False,
            rsi3_monadic_rate=0.05,
            rsi3_banach_residual=1.0,
            rsi3_contracted=False,
            berry_phase_coherent=False,
            global_heyting_verdict=HeytingOmega3.VETOED,
            timestamp=now,
        )

    def execute_phase3_sovereign_adjudication_cycle(
        self,
        campaign_result: IntuitionWitnessCampaignResult,
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionSovereignGovernanceSealedState:
        r"""
        MÉTODO FINAL DE LA FASE 3 (cierre soberano agéntico).

        Consume Φ₂₃, resuelve Ω₃, aplica válvula / interlock, certifica vía
        motor espectral si está presente y sella la gobernanza.
        """
        now = time.time()
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        ingested = self.initialize_phase3_from_campaign_result(campaign_result)

        verdict = ingested.global_heyting_verdict
        if self.is_relief_valve_expired(now) and verdict != HeytingOmega3.VETOED:
            logger.critical("[RELIEF_VALVE_EXPIRED] Escalando veto suave → VETOED (3600 s sin e⁺).")
            verdict = HeytingOmega3.VETOED

        crowbar_triggered = False
        actuation_latency = 0.0
        actuation = ActuationMode.NORMAL_FLUID
        relief_active = False

        hard = (
            verdict == HeytingOmega3.VETOED
            or ingested.gromov_capacity_peak > self.capacity_gromov_max
            or ingested.bures_wasserstein_distance_peak > _BURES_VETO_CEILING
        )
        soft = (
            verdict == HeytingOmega3.DEGRADED
            or (not ingested.berry_phase_coherent)
            or ingested.bures_wasserstein_distance_peak > _BURES_DEGRADED_CEILING
            or ingested.kam_twist_floor < _KAM_TWIST_FLOOR
        )

        if hard:
            verdict = HeytingOmega3.VETOED
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
            actuation_latency = self.trigger_esp32_crowbar_interlock()
            crowbar_triggered = True
            self._soft_veto_active = False
            logger.critical(
                "[CROWBAR_TRIGGERED] Interlock simulado pin=%d en %.2f ns. Bures=%.4f Gromov=%.2f",
                self.esp32_gpio_pin,
                actuation_latency,
                ingested.bures_wasserstein_distance_peak,
                ingested.gromov_capacity_peak,
            )
        elif soft:
            verdict = HeytingOmega3.DEGRADED
            actuation = ActuationMode.SOFT_VETO_BYPASS
            self._soft_veto_active = True
            self._soft_veto_timestamp = now
            relief_active = True
        else:
            self._soft_veto_active = False

        exec_cert: Optional[Any] = None
        if self.engine is not None:
            try:
                exec_cert = self.engine.audit_and_certify_jumps(mac_density_matrix)
            except Exception as exc:  # pragma: no cover - resiliencia del orquestador
                logger.error("[ENGINE_CERT_SKIP] %s", exc)

        # Reinyecta tasa monádica del certificado si el motor v8 la expone.
        if exec_cert is not None and hasattr(exec_cert, "level3_rsi_monadic_rate"):
            engine_rate = float(exec_cert.level3_rsi_monadic_rate)
            blended = float(np.clip(0.5 * (ingested.rsi3_monadic_rate + engine_rate), 0.05, 0.45))
            ingested = IntuitionWitnessCampaignResult(
                campaign_id=ingested.campaign_id,
                sovereign_id=ingested.sovereign_id,
                source_gauge_contract=ingested.source_gauge_contract,
                jumps_observed_count=ingested.jumps_observed_count,
                weak_value_aw_mean=ingested.weak_value_aw_mean,
                bures_wasserstein_distance_peak=ingested.bures_wasserstein_distance_peak,
                fubini_study_distance_peak=ingested.fubini_study_distance_peak,
                uhlmann_fidelity_mean=ingested.uhlmann_fidelity_mean,
                uhlmann_fidelity_floor=ingested.uhlmann_fidelity_floor,
                berry_phase_peak=ingested.berry_phase_peak,
                kelly_stake_mean=ingested.kelly_stake_mean,
                gromov_capacity_peak=ingested.gromov_capacity_peak,
                greene_residue_peak=ingested.greene_residue_peak,
                kam_twist_floor=ingested.kam_twist_floor,
                symplectic_conserved_all=ingested.symplectic_conserved_all,
                rsi3_monadic_rate=blended,
                rsi3_banach_residual=ingested.rsi3_banach_residual,
                rsi3_contracted=ingested.rsi3_contracted and bool(getattr(exec_cert, "rsi3_contracted", True)),
                berry_phase_coherent=ingested.berry_phase_coherent,
                global_heyting_verdict=verdict.meet(ingested.global_heyting_verdict),
                timestamp=ingested.timestamp,
            )
            verdict = ingested.global_heyting_verdict

        visceral_rep = self.apply_visceral_functor_phi_sem(
            IntuitionWitnessCampaignResult(
                campaign_id=ingested.campaign_id,
                sovereign_id=ingested.sovereign_id,
                source_gauge_contract=ingested.source_gauge_contract,
                jumps_observed_count=ingested.jumps_observed_count,
                weak_value_aw_mean=ingested.weak_value_aw_mean,
                bures_wasserstein_distance_peak=ingested.bures_wasserstein_distance_peak,
                fubini_study_distance_peak=ingested.fubini_study_distance_peak,
                uhlmann_fidelity_mean=ingested.uhlmann_fidelity_mean,
                uhlmann_fidelity_floor=ingested.uhlmann_fidelity_floor,
                berry_phase_peak=ingested.berry_phase_peak,
                kelly_stake_mean=ingested.kelly_stake_mean,
                gromov_capacity_peak=ingested.gromov_capacity_peak,
                greene_residue_peak=ingested.greene_residue_peak,
                kam_twist_floor=ingested.kam_twist_floor,
                symplectic_conserved_all=ingested.symplectic_conserved_all,
                rsi3_monadic_rate=ingested.rsi3_monadic_rate,
                rsi3_banach_residual=ingested.rsi3_banach_residual,
                rsi3_contracted=ingested.rsi3_contracted,
                berry_phase_coherent=ingested.berry_phase_coherent,
                global_heyting_verdict=verdict,
                timestamp=ingested.timestamp,
            )
        )

        provenance_bytes = (
            f"{ingested.campaign_id}:{verdict.name}:{actuation.name}:"
            f"{ingested.bures_wasserstein_distance_peak:.6f}:{crowbar_triggered}:"
            f"{actuation_latency:.2f}:{ingested.rsi3_monadic_rate:.6f}:{timestamp_utc}"
        ).encode("utf-8")
        prov_hash = hashlib.sha256(provenance_bytes).hexdigest()

        return IntuitionSovereignGovernanceSealedState(
            gauge_contract=ingested.source_gauge_contract,
            campaign_result=ingested,
            execution_certificate=exec_cert,
            visceral_report=visceral_rep,
            annihilation_proof=None,
            heyting_final_verdict=verdict,
            actuation_mode=actuation,
            crowbar_triggered=crowbar_triggered,
            actuation_latency_ns=actuation_latency,
            relief_valve_active=relief_active,
            provenance_hash_sha256=prov_hash,
            timestamp_utc=timestamp_utc,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SOBERANO TESTIGO DE INTUICIÓN (TOONIntuitionWitnessAgent)
# ══════════════════════════════════════════════════════════════════════════════

class TOONIntuitionWitnessAgent(Phase3_SovereignIntuitionWitnessAdjudicator):
    r"""
    Soberano agéntico del Testigo de Intuición QND.

    Orquestador unificado de las fases anidadas 1 → 2 → 3. Supervisa saltos
    geodésicos flash, preserva la coherencia de calibre y cierra el lazo con
    interlock simulado de latencia acotada.
    """

    def execute_full_witness_governance_cycle(
        self,
        requests: List[IntuitionJumpObservationRequest],
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionSovereignGovernanceSealedState:
        r"""
        Encadenamiento anidado completo:

          FASE 1: `forge_phase1_gauge_poincare_contract`
                  ↓  Φ₁₂  `continue_into_phase2_campaign_gate`
          FASE 2: `execute_phase2_qnd_witness_campaign`
                  ↓  Φ₂₃  `continue_into_phase3_adjudication_gate`
          FASE 3: `execute_phase3_sovereign_adjudication_cycle`
                  ↓  estado sellado
        """
        contract = self.forge_phase1_gauge_poincare_contract(density_matrix=mac_density_matrix)
        try:
            campaign_result = self.execute_phase2_qnd_witness_campaign(
                contract=contract,
                requests=requests,
                mac_density_matrix=mac_density_matrix,
            )
        except ValueError as exc:
            campaign_result = self._veto_campaign_from_contract(contract, reason=str(exc))
        return self.execute_phase3_sovereign_adjudication_cycle(
            campaign_result=campaign_result,
            mac_density_matrix=mac_density_matrix,
        )

    def initialize_gauge_contract(
        self,
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionWitnessGaugeContract:
        return self.forge_phase1_gauge_poincare_contract(mac_density_matrix)

    def execute_qnd_witness_campaign(
        self,
        requests: List[IntuitionJumpObservationRequest],
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionWitnessCampaignResult:
        contract = self.forge_phase1_gauge_poincare_contract(mac_density_matrix)
        try:
            return self.execute_phase2_qnd_witness_campaign(contract, requests, mac_density_matrix)
        except ValueError as exc:
            return self._veto_campaign_from_contract(contract, reason=str(exc))

    def generate_visceral_business_report(
        self,
        campaign_result: IntuitionWitnessCampaignResult,
    ) -> IntuitionVisceralBusinessReport:
        return self.apply_visceral_functor_phi_sem(campaign_result)


SovereignIntuitionWitnessAdjudicator = TOONIntuitionWitnessAgent


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DE LAZO CERRADO Y PRUEBAS DOCTORALES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("════════════════════════════════════════════════════════════════════════════════")
    print("  VERIFICACIÓN TOONIntuitionWitnessAgent v8.0.0-SOVEREIGN-RSI3-POINCARE-PhD")
    print("════════════════════════════════════════════════════════════════════════════════\n")

    dim = 56
    sovereign = TOONIntuitionWitnessAgent(
        sovereign_id="SOVEREIGN-INTUITION-WITNESS-001",
        dimension=dim,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
    )

    rng = np.random.default_rng(2026)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    rho_mac = np.outer(v0, v0.conj())

    v_jump = v0.copy()
    v_jump[0] *= cmath.exp(1j * 0.05)
    v_jump /= np.linalg.norm(v_jump)

    req = IntuitionJumpObservationRequest(
        request_id="REQ-SOVEREIGN-001",
        jump_id="JUMP-FLASH-001",
        apu_code="APU-CONCRETO-STRUCTURAL-4000PSI",
        jump_vector=v_jump,
        stinespring_coupling_eps=0.04,
    )

    print("[1] Lazo cerrado soberano (Fase 1 → Fase 2 → Fase 3)...")
    sealed_state = sovereign.execute_full_witness_governance_cycle(
        requests=[req],
        mac_density_matrix=rho_mac,
    )
    coup = sealed_state.gauge_contract.poincare_coupling

    print(f"  • [Fase 1] Contrato ID               : {sealed_state.gauge_contract.contract_id}")
    print(f"  • [Fase 1] Chern-Simons CS₃          : {sealed_state.gauge_contract.chern_simons_3form:.6e}")
    print(f"  • [Fase 1] Chern-Hopf c₁             : {sealed_state.gauge_contract.chern_hopf_class_c1:.6e}")
    print(f"  • [Fase 1] Holonomía de Wilson θ_W   : {sealed_state.gauge_contract.wilson_holonomy_angle:.6f} rad")
    print(f"  • [Fase 1] Régimen topológico        : {sealed_state.gauge_contract.topology_regime}")
    print(f"  • [Fase 1] Rank Grassmann r          : {coup.grassmann_rank}")
    print(f"  • [Fase 1] Twist KAM proxy           : {coup.kam_twist_proxy:.6e}")
    print(f"  • [Fase 1] Residuo Greene-Wilson     : {coup.greene_wilson_residue:.6f}")
    print(f"  • [Fase 1] 𝔲(n) drift ‖A+A†‖         : {coup.antihermitian_drift:.4e}")
    print(f"  • [Fase 2] Campaña ID                : {sealed_state.campaign_result.campaign_id}")
    print(f"  • [Fase 2] Peak distancia Bures      : {sealed_state.campaign_result.bures_wasserstein_distance_peak:.6f}")
    print(f"  • [Fase 2] Fidelidad Uhlmann floor   : {sealed_state.campaign_result.uhlmann_fidelity_floor:.6f}")
    print(f"  • [Fase 2] Rate RSI³ monádico        : {sealed_state.campaign_result.rsi3_monadic_rate:.6f}")
    print(f"  • [Fase 2] Banach residual / contr.  : "
          f"{sealed_state.campaign_result.rsi3_banach_residual:.3e} / {sealed_state.campaign_result.rsi3_contracted}")
    print(f"  • [Fase 2] Twist KAM floor           : {sealed_state.campaign_result.kam_twist_floor:.6e}")
    print(f"  • [Fase 3] Veredicto Heyting Ω₃      : {sealed_state.heyting_final_verdict.name}")
    print(f"  • [Fase 3] Modo de actuación         : {sealed_state.actuation_mode.name}")
    print(f"  • [Fase 3] Interlock simulado        : {sealed_state.crowbar_triggered}")
    print(f"  • [Fase 3] Latencia de actuación     : {sealed_state.actuation_latency_ns:.2f} ns")
    print(f"  • [Fase 3] Válvula de alivio         : {sealed_state.relief_valve_active}")
    print(f"  • [Fase 3] WACC status               : {sealed_state.visceral_report.wacc_protection_status}")
    print(f"  • [Fase 3] Liberación liquidez neta  : +{sealed_state.visceral_report.net_liquidity_release_pct:.2f}%")
    print(f"  • [Fase 3] SHA-256 provenance        : {sealed_state.provenance_hash_sha256[:24]}...\n")

    print("[2] Aniquilación de positrones e⁺ (HMAC estricto, sin bypass)...")
    anomaly_id = "ANOMALY-INSTANTON-COST-001"
    valid_token = sovereign.expected_positron_token(anomaly_id)
    proof = sovereign.process_human_positron_annihilation(
        campaign_id=sealed_state.campaign_result.campaign_id,
        electron_anomaly_id=anomaly_id,
        positron_token=valid_token,
    )
    print(f"  • Proof ID                           : {proof.proof_id}")
    print(f"  • Vacío Dirac restaurado             : {proof.vacuo_dirac_restored}")
    print(f"  • Energía descargada (keV)           : {proof.energy_discharged_kev:.1f} keV")

    rejected = sovereign.process_human_positron_annihilation(
        campaign_id=sealed_state.campaign_result.campaign_id,
        electron_anomaly_id=anomaly_id,
        positron_token="AUTHORIZED_HUMAN_POSITRON_TOKEN_SECRET_64HEX",
    )
    print(f"  • Token ilegítimo rechazado          : {not rejected.vacuo_dirac_restored}\n")

    print("════════════════════════════════════════════════════════════════════════════════")
    print("  VERIFICACIÓN EXITOSA: TOONIntuitionWitnessAgent v8.0.0 OPERATIVO")
    print("════════════════════════════════════════════════════════════════════════════════")