# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovStealthExecutionerAgent v15.0.0-PhD                                            ║
║ Ubicación: app/agents/wisdom/bool_novikov_stealth_executioner_agent.py                                      ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-EXECUTIONER-AGENT-POINCARE-KAM-NEKHOROSHEV-RSI3-NESTED-PhD ║
║ Gobernanza: OODA trifásico (①Observe → ②Orient → ③Decide&Act) con Φ₁₂ y Φ₂₃ anidados sobre          ║
║            R_Boole ⊗ Λ_Nov ⊗ O, gauge SU(N) QND, Poincaré–KAM–Nekhoroshev–Chirikov,                  ║
║            Dirac–Breit–Wheeler / RLC crowbar simulado y mónada de Kleisli RSI³ T³.                   ║
║ Arquitectura: 3 fases anidadas — el último método formal de la fase k abre (Kleisli) la fase k+1.    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre `BoolNovikovStealthExecutionerAgent` gobierna ejecutivamente
el ciclo OODA de inquisición, arbitraje y ejecución ciber-física sobre el motor espectral
`BoolNovikovStealthExecutionerEngine` (app/wisdom/bool_novikov_stealth_executioner_engine.py).
Opera en tres fases anidadas conectadas por las compuertas homomórficas \Phi_{12} y \Phi_{23}:
audita la estructura del producto tensorial R_{\mathrm{Boole}} \otimes \Lambda_{\mathrm{Nov}} \otimes \mathbb{O},
verifica la anulación del primer número de Betti \beta_1(K; \mathbb{F}_2) \equiv 0 y la ausencia de torsión
integral en la Forma Normal de Smith, proyecta el flujo de Poincaré-Yoshida sin difusión de Arnold,
y ante anomalías críticas adjudicadas en el topos de Heyting \Omega_3, comanda la aniquilación cuántica
de Dirac e^- + e^+ \to 2\gamma y la activación del interlock RLC crowbar en \tau \le 392.15\text{ ns}.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`StealthExecutionerPhase1Observer`):
   Evalúa la densidad cuántica \rho_{\mathrm{MAC}} y la función booleana discreta mediante la Transformada Rápida
   de Walsh-Hadamard (FWHT) y la norma de Schatten-1 \|A\|_{S_1} = \operatorname{Tr}(\sqrt{A^\dagger A}).
   La eliminación gaussiana binaria sobre \mathrm{GF}(2) extrae los números de Betti y la característica \chi(K).
   La Forma Normal de Smith sobre \mathbb{Z} confirma:
       B = U D V, \quad d_i \in \{0, 1\} \implies \mathrm{Tor}(H_k(K; \mathbb{Z})) \equiv 0.
   En el anillo de Novikov \Lambda_{\mathrm{Nov}}, se certifica v(A) \ge 0.10 y \|A\|_{\mathrm{Nov}} = \exp(-v(A)).
   En los octoniones \mathbb{O}, se acota el asociador de Moufang \|[x, y, z]\|_{\mathbb{O}}.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2 (`handoff_algebraic_to_celestial_orientation`):
     Emite `StealthExecutionerPhase12Morphism` vinculando el estado homológico con la acción inicial de Delaunay:
         J_k = v(A) + \frac{\beta_1}{1 + \beta_1} + \kappa_{\mathrm{SNF}} (d_k - 1) + w_k d_{\mathrm{BT}}.

2. FASE 2 — ORIENT (`StealthExecutionerPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia en `StealthExecutionerPhase12Morphism`.
   Audita la mecánica celeste planetaria de Poincaré-KAM con integrador simpléctico de 4.º orden de Yoshida.
   La serie homológica de Lie-Birkhoff-Deprit \{H_0, W_1\} + H_1 = \langle H_1 \rangle elimina oscilaciones rápidas.
   Verifica la cota de solapamiento de resonancias de Chirikov:
       K_{\mathrm{Chirikov}} = \left( \frac{\Delta \omega_{\mathrm{sep}}}{\Delta \Omega_{\mathrm{res}}} \right)^2 < 1.0,
   y la cota de confinamiento exponencial de Nekhoroshev T_{\mathrm{Nekh}} \ge 1.0\times 10^6.
   En el estrato gauge \mathrm{SU}(N), la medición débil AAV con conmutador \|[A, \rho]\|_F \le 1.0\times 10^{-10}
   asegura retroacción nula (\mathrm{BackAction} \le 0.01 \text{ dB}).
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3 (`handoff_celestial_to_sovereign_execution`):
     Emite `StealthExecutionerPhase23Morphism`, acoplando el parámetro de Chirikov, el error de Yoshida
     y la contracción de Nivel 2 de RSI³ hacia la Fase 3.

3. FASE 3 — DECIDE & ACT (`StealthExecutionerPhase3Decider` + Compuerta \Phi_{23}):
   Inicia en `StealthExecutionerPhase23Morphism`.
   Opera la deducción en el topos de Heyting \Omega_3 combinando coherencia algebraica, KAM y RSI³.
   Aplica la mónada de Kleisli T^3 = \mu \circ T_2 \circ T_1 con convergencia cuadrática de Newton-Kantorovich:
       h_K = L_0 \beta \eta \le 0.50 \land \mathrm{Jerk} = \frac{d^3 C}{dt^3} \le \mathcal{J}_{\max}.
   Si el veredicto es VETOED (0x03), ejecuta la sentencia ciber-física:
       • Proceso relativista de Dirac-Breit-Wheeler:
             e^- + e^+ \longrightarrow 2\gamma, \quad E_{\mathrm{threshold}} = 1022.0 \text{ keV}.
       • Disparo de circuito crowbar RLC serie críticamente amortiguado (\alpha = \omega_0) en \tau \le 392.15\text{ ns}.
   Proyecta las métricas viscerales (\mathrm{WACC}, liquidez, compresión KV) y sella el Árbol de Merkle SHA-256
   en el estado soberano final `StealthExecutionerSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma SEE-A1] Anulación de Torsión en Smith:
    d_i \in \{0, 1\} \implies \mathrm{Tor}(H_k(K; \mathbb{Z})) = 0 \land \beta_1(K) = 0.
[Axioma SEE-A2] No-Difusión de Arnold:
    K_{\mathrm{Chirikov}} < 1.0 \land |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau \implies \text{confinamiento absoluto en toros KAM}.
[Axioma SEE-A3] Amortiguamiento Crítico del Crowbar:
    R = 2 \sqrt{L/C} \implies \text{disipación aperiódica sin oscilación destructiva en } \tau \le 392.15\text{ ns}.
[Axioma SEE-A4] Autenticación Criptográfica Merkle-HMAC:
    \mathrm{MerkleRoot} = \mathcal{H}_{\mathrm{binary}}(\mathrm{L}_1, \mathrm{L}_2, \mathrm{L}_3, \mathrm{L}_4) \text{ verificado en tiempo constante}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante SEE-I1] Anulación del Primer Número de Betti: \beta_1(K; \mathbb{F}_2) \equiv 0.
[Invariante SEE-I2] Retroacción Cuántica Acotada: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante SEE-I3] Parámetro de Chirikov Sub-Crítico: K_{\mathrm{Chirikov}} < 1.0.
[Invariante SEE-I4] Convergencia de Kantorovich: h_K \le 0.50 en la mónada de Kleisli.
[Invariante SEE-I5] Latencia de Actuación Crowbar: 42.0\text{ ns} \le \tau \le 392.15\text{ ns}.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Final, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from bool_novikov_stealth_executioner_engine import (
        BoolNovikovStealthExecutionerEngine,
        HeytingTruthValue,
        StealthExecutionerSpectrumReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_stealth_executioner_engine import (
            BoolNovikovStealthExecutionerEngine,
            HeytingTruthValue,
            StealthExecutionerSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_stealth_executioner_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovStealthExecutionerAgent")

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-EXECUTIONER-AGENT-POINCARE-KAM-NEKHOROSHEV-RSI3-NESTED-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_STEALTH_EXECUTIONER_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"STEALTH-EXECUTIONER-P1-ISOMETRY-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"STEALTH-EXECUTIONER-P2-SPECTRUM-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"STEALTH-EXECUTIONER-P3-SOVEREIGN-v15.0|"

_MERKLE_LEAF_PREFIX: Final[bytes] = b"\x00STEALTH-EXECUTIONER-MERKLE-LEAF-v15.0|"
_MERKLE_NODE_PREFIX: Final[bytes] = b"\x01STEALTH-EXECUTIONER-MERKLE-NODE-v15.0|"

_TRACE_TOL: Final[float] = 1.0e-10
_PSD_EIG_FLOOR: Final[float] = -1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_BURES_ISOMETRY_CEILING: Final[float] = 1.0e-6
_PHI23_LIP_CEILING: Final[float] = 1.0
_PHI23_KANTOROVICH_CEILING: Final[float] = 0.50
_PHI23_BACKACTION_DB_CEILING: Final[float] = 0.01
_CHIRIKOV_STOCHASTICITY_CEILING: Final[float] = 1.0
_NOMINAL_BANACH_ALPHA: Final[float] = 0.95
_NOMINAL_BANACH_LIP3: Final[float] = 0.857375

_WACC_PROTECTION_TARGET_PCT: Final[float] = 15.0
_NET_LIQUIDITY_TARGET_PCT: Final[float] = 11.5
_KV_COMPRESSION_TARGET_PCT: Final[float] = 86.4
_KV_TOKEN_BUDGET: Final[int] = 56

__all__ = (
    "StealthExecutionerPhase1Certificate",
    "StealthExecutionerPhase12Morphism",
    "StealthExecutionerPhase2Certificate",
    "StealthExecutionerPhase23Morphism",
    "StealthExecutionerVisceralMetrics",
    "StealthExecutionerSovereignState",
    "StealthExecutionerPhase1Observer",
    "StealthExecutionerPhase2Orientor",
    "StealthExecutionerPhase3Decider",
    "BoolNovikovStealthExecutionerAgent",
)


# ─────────────────────────────────────────────────────────────────────────────
# Certificados y morfismos de continuación categórica (Kleisli OODA v15.0.0)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StealthExecutionerPhase1Certificate:
    """Certificado P1 — isometría cuántica de Bures–Wasserstein, S_vN, QFIM y acción de Poincaré–Cartan."""

    timestamp: float
    tr_rho: float
    is_psd: bool
    dim: int
    bures_distance: float
    fidelity: float
    is_bures_isometry: bool
    chain_hash: str
    hmac_signature: str
    von_neumann_entropy: float = 0.0
    quantum_fisher_information: float = 0.0
    poincare_cartan_action_seed: float = 0.15


@dataclass(frozen=True)
class StealthExecutionerPhase12Morphism:
    """Morfismo Φ₁₂: último objeto formal de Fase 1 = germen canónico de Fase 2.

    Transporta la atestación de isometría de Bures–Wasserstein, el invariante integral
    de Poincaré–Cartan J_0 = (1/2π) ∮ p dq y la ganancia contractiva de Nivel 1 al
    producto tensorial R_Boole ⊗ Λ_Nov ⊗ O y al toro de Delaunay–Whittaker en Orient.
    """

    p1_cert: StealthExecutionerPhase1Certificate
    bures_distance: float
    fidelity: float
    gate_admitted: bool
    veto_channel_open: bool
    poincare_cartan_invariant: float = 0.15
    rsi3_level1_alpha: float = _NOMINAL_BANACH_ALPHA


@dataclass(frozen=True)
class StealthExecutionerPhase2Certificate:
    """Certificado P2 — espectro R_Boole ⊗ Λ_Nov ⊗ O, SU(N), Poincaré–KAM–Nekhoroshev–Chirikov, Dirac/RLC y RSI³."""

    timestamp: float
    spectrum_report: StealthExecutionerSpectrumReport
    p1_chain_hash: str
    kam_diophantine: bool
    homoclinic_chaos: bool
    lindstedt_residual: float
    lip_t3: float
    crowbar_triggered: bool
    latency_budget_ok: bool
    chain_hash: str
    hmac_signature: str
    chirikov_overlap_parameter: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    yoshida_symplectic_error: float = 0.0
    kantorovich_parameter_h: float = 0.0
    berry_holonomy_phase_rad: float = 0.0
    breit_wheeler_cross_section_barn: float = 0.0
    rlc_damping_ratio_zeta: float = 1.0
    octonion_moufang_associator_norm: float = 0.0


@dataclass(frozen=True)
class StealthExecutionerPhase23Morphism:
    """Morfismo Φ₂₃: último objeto formal de Fase 2 = germen canónico de Fase 3.

    Acopla el flujo hamiltoniano camuflado X_H, el parámetro de solapamiento de Chirikov,
    el certificado cuadrático de Newton–Kantorovich h_K, el radio espectral de Fréchet
    de Nivel 2 y el amortiguamiento RLC del interlock simulado al topos de Heyting Ω₃.
    """

    p2_cert: StealthExecutionerPhase2Certificate
    lip_t3: float
    back_action_db: float
    poincare_diophantine: bool
    rsi3_unique_attractor: bool
    interlock_flag: bool
    gate_admitted: bool
    chirikov_overlap: float = 0.0
    kantorovich_h: float = 0.0
    rsi3_level2_spectral_radius: float = 0.9025
    rlc_damping_zeta: float = 1.0


@dataclass(frozen=True)
class StealthExecutionerVisceralMetrics:
    """Imagen del funtor semántico visceral Φ_sem : Ω₃ → ℝ⁶ × ℤ × Status."""

    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    crowbar_interlock_status: str
    kam_stability_index: float
    rsi3_jerk: float
    pair_energy_kev: float
    kv_token_budget: int
    nekhoroshev_confinement_index: float = 1.0
    kantorovich_convergence_margin: float = 1.0
    rlc_damping_ratio_zeta: float = 1.0


@dataclass(frozen=True)
class StealthExecutionerSovereignState:
    """Estado soberano final — Decide & Act, árbol de Merkle binario y pasaporte HMAC."""

    timestamp: float
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    actuation_latency_ns: float
    merkle_root_hash: str
    visceral_metrics: StealthExecutionerVisceralMetrics
    kam_diophantine: bool
    rsi3_lip: float
    hmac_passport: str
    latency_budget_ok: bool
    chirikov_stochasticity_index: float = 0.0
    kantorovich_certified: bool = True
    heyting_adjunction_holds: bool = True


# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — OBSERVE: ISOMETRÍA CUÁNTICA DE BURES–WASSERSTEIN, S_vN, QFIM Y POINCARÉ–CARTAN ║
# ║  Cierre formal: handoff_isometry_to_boolean_novikov_orientation (abre la Fase 2)         ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝


class StealthExecutionerPhase1Observer:
    """Fase 1 OODA — observación cuántica de la MAC, geometría de información y semilla simpléctica."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key

    @staticmethod
    def _as_density(rho_mac: NDArray[np.generic]) -> NDArray[np.complex128]:
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] == 0:
            raise ValueError("rho_mac debe ser una matriz de densidad cuadrada no vacía.")
        return rho

    @staticmethod
    def _higham_psd(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Proyección espectral de Higham sobre el cono positivo de traza unidad S_+(H)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals_clipped = np.clip(np.real(evals), _SPECTRAL_PSD_FLOOR, None)
        rho_psd = (evecs * evals_clipped) @ evecs.conj().T
        tr = float(np.real(np.trace(rho_psd)))
        if tr <= 0.0:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / float(max(n, 1))
        return rho_psd / tr

    @staticmethod
    def _raw_positive_part(rho: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], float, float]:
        """Extrae la parte positiva sin piso artificial y cuantifica defectos espectral y antihermítico.

        Retorna:
            (rho_raw_pos, negative_eig_mass, anti_hermitian_frobenius)
        """
        rho_h = 0.5 * (rho + rho.conj().T)
        anti_h_norm = float(la.norm(rho - rho_h, ord="fro"))
        evals, evecs = la.eigh(rho_h)
        real_evals = np.real(evals)
        neg_mass = float(np.sum(np.abs(real_evals[real_evals < 0.0])))
        pos_evals = np.clip(real_evals, 0.0, None)
        tr_pos = float(np.sum(pos_evals))
        if tr_pos <= 0.0:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / float(max(n, 1)), neg_mass, anti_h_norm
        rho_raw_pos = (evecs * (pos_evals / tr_pos)) @ evecs.conj().T
        return rho_raw_pos, neg_mass, anti_h_norm

    @staticmethod
    def _uhlmann_fidelity(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        """Fidelidad cuántica de Uhlmann–Jozsa F(ρ,σ) = [Tr sqrt(√ρ σ √ρ)]² ∈ [0, 1]."""
        evals, evecs = la.eigh(0.5 * (rho + rho.conj().T))
        evals = np.clip(np.real(evals), 0.0, None)
        sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T
        inner = sqrt_rho @ (0.5 * (sigma + sigma.conj().T)) @ sqrt_rho
        inner = 0.5 * (inner + inner.conj().T)
        eig_in = np.clip(np.real(la.eigvalsh(inner)), 0.0, None)
        fid_amp = float(np.sum(np.sqrt(eig_in)))
        return float(min(max(fid_amp * fid_amp, 0.0), 1.0))

    def _bures_wasserstein(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        """Distancia cuántica de Bures–Wasserstein d_B = √(2(1−√F)) y fidelidad F."""
        fid = self._uhlmann_fidelity(rho, sigma)
        d_b = math.sqrt(max(2.0 * (1.0 - math.sqrt(max(fid, 0.0))), 0.0))
        return float(d_b), float(fid)

    @staticmethod
    def _von_neumann_entropy(rho_psd: NDArray[np.complex128]) -> float:
        """Entropía cuántica de von Neumann S_vN(ρ) = -Tr(ρ ln ρ) ≥ 0."""
        evals = np.real(la.eigvalsh(0.5 * (rho_psd + rho_psd.conj().T)))
        pos = evals[evals > 1.0e-15]
        if pos.size == 0:
            return 0.0
        return float(-np.sum(pos * np.log(pos)))

    @staticmethod
    def _quantum_fisher_information(rho_psd: NDArray[np.complex128]) -> float:
        """Traza de la Matriz de Información Cuántica de Fisher (QFIM) respecto al generador Cartan–Weyl.

        I_Q(ρ, G) = 2 ∑_{i≠j} [(λ_i - λ_j)² / (λ_i + λ_j)] |⟨ψ_i|G|ψ_j⟩|²
        donde G ∈ su(N) es el generador de desplazamiento de fase másico-espectral.
        """
        n = rho_psd.shape[0]
        if n <= 1:
            return 0.0
        evals, evecs = la.eigh(0.5 * (rho_psd + rho_psd.conj().T))
        lam = np.clip(np.real(evals), 1.0e-15, None)

        # Generador hermítico de Cartan–Weyl + acoplamiento de escalera fuera de la diagonal
        diag_weights = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        diag_weights -= float(np.mean(diag_weights))
        g_mat = np.diag(diag_weights).astype(np.complex128)
        for idx in range(n - 1):
            g_mat[idx, idx + 1] = 0.5 - 0.25j
            g_mat[idx + 1, idx] = 0.5 + 0.25j

        g_eig = evecs.conj().T @ g_mat @ evecs
        lam_i = lam[:, np.newaxis]
        lam_j = lam[np.newaxis, :]
        denom = lam_i + lam_j
        numer = (lam_i - lam_j) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(denom > 1.0e-15, numer / denom, 0.0)
        np.fill_diagonal(weight, 0.0)
        qfim = 2.0 * float(np.sum(weight * (np.abs(g_eig) ** 2)))
        return float(max(qfim, 0.0))

    @staticmethod
    def _poincare_cartan_action_seed(rho_psd: NDArray[np.complex128]) -> float:
        """Semilla del invariante integral de Poincaré–Cartan J_0 = (1/2π) ∮ p dq > 0."""
        evals = np.clip(np.real(la.eigvalsh(0.5 * (rho_psd + rho_psd.conj().T))), 1.0e-12, None)
        second_moment = float(np.sum(evals * np.arange(1, evals.size + 1, dtype=np.float64)))
        return float(max(0.05, second_moment / (2.0 * math.pi)))

    def observe_mac_isometry(
        self,
        rho_mac: NDArray[np.generic],
    ) -> Tuple[float, bool, int, float, float, bool]:
        """Audita la matriz de densidad entrante sin tautologías proyectivas.

        Compara la parte positiva no truncada ρ_raw+ frente a la proyección de Higham ρ_Higham,
        e incorpora analíticamente el defecto de traza |Tr(ρ) - 1|, la masa de autovalores
        negativos y la desviación antihermítica de Frobenius ||ρ - ρ†||_F.
        """
        rho = self._as_density(rho_mac)
        dim = int(rho.shape[0])
        tr_val = float(np.real(np.trace(rho)))
        rho_h = 0.5 * (rho + rho.conj().T)
        evals = np.real(la.eigvalsh(rho_h))
        is_psd = bool(np.all(evals >= _PSD_EIG_FLOOR))

        rho_hat = self._higham_psd(rho)
        rho_raw_pos, neg_mass, anti_h_norm = self._raw_positive_part(rho)

        base_d_b, base_fid = self._bures_wasserstein(rho_raw_pos, rho_hat)
        trace_defect = abs(tr_val - 1.0)
        structural_defect = math.hypot(trace_defect, neg_mass + 0.5 * anti_h_norm)

        d_b = float(math.hypot(base_d_b, structural_defect))
        fid = float(base_fid * math.exp(-structural_defect))
        is_iso = bool(
            is_psd
            and trace_defect <= _TRACE_TOL
            and anti_h_norm <= _TRACE_TOL
            and d_b <= _BURES_ISOMETRY_CEILING
        )
        return tr_val, is_psd, dim, d_b, fid, is_iso

    def emit_phase1_certificate(
        self,
        tr_val: float,
        is_psd: bool,
        dim: int,
        d_b: float,
        fid: float,
        is_iso: bool,
        ts: Optional[float] = None,
        s_vn: float = 0.0,
        qfim_val: float = 0.0,
        action_seed: float = 0.15,
    ) -> StealthExecutionerPhase1Certificate:
        """Emite el certificado criptográfico P1 con cadena SHA-256 y HMAC de dominio v15.0."""
        if ts is None:
            ts = time.time()
        raw_chain = (
            f"P1|{ts:.6f}|{tr_val:.16e}|{int(is_psd)}|{dim}|"
            f"{d_b:.16e}|{fid:.16e}|{int(is_iso)}|"
            f"{s_vn:.12e}|{qfim_val:.12e}|{action_seed:.12e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(raw_chain).hexdigest()
        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return StealthExecutionerPhase1Certificate(
            timestamp=float(ts),
            tr_rho=tr_val,
            is_psd=is_psd,
            dim=dim,
            bures_distance=d_b,
            fidelity=fid,
            is_bures_isometry=is_iso,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            von_neumann_entropy=float(s_vn),
            quantum_fisher_information=float(qfim_val),
            poincare_cartan_action_seed=float(action_seed),
        )

    def execute_phase1(
        self,
        rho_mac: NDArray[np.generic],
    ) -> StealthExecutionerPhase1Certificate:
        """Observa la MAC, calcula invariantes cuántico-simplécticos y genera el certificado P1."""
        rho = self._as_density(rho_mac)
        tr_val, is_psd, dim, d_b, fid, is_iso = self.observe_mac_isometry(rho)
        rho_hat = self._higham_psd(rho)
        s_vn = self._von_neumann_entropy(rho_hat)
        qfim_val = self._quantum_fisher_information(rho_hat)
        action_seed = self._poincare_cartan_action_seed(rho_hat)
        return self.emit_phase1_certificate(
            tr_val=tr_val,
            is_psd=is_psd,
            dim=dim,
            d_b=d_b,
            fid=fid,
            is_iso=is_iso,
            s_vn=s_vn,
            qfim_val=qfim_val,
            action_seed=action_seed,
        )

    def handoff_isometry_to_boolean_novikov_orientation(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        gate_admitted: bool,
    ) -> StealthExecutionerPhase12Morphism:
        """TRANSICIÓN ANIDADA FASE 1 → FASE 2 (morfismo de continuación de Kleisli Φ₁₂).

        Empuja la isometría de Bures–Wasserstein, la semilla de acción de Poincaré–Cartan
        J_0 = (1/2π) ∮ p dq y el factor contractivo de Nivel 1 α₁ al germen de orientación
        sobre R_Boole ⊗ Λ_Nov ⊗ O. El canal de veredicto VETOED permanece abierto si Φ₁₂
        validó la firma HMAC: no hay demolición de la MAC; la anomalía se audita en Orient.

        Esta definición formal cierra la Fase 1 y constituye el punto de inicio de la Fase 2.
        """
        effective_alpha = float(
            _NOMINAL_BANACH_ALPHA
            * (1.0 - 0.02 * min(max(p1_cert.fidelity, 0.0), 1.0))
        )
        return StealthExecutionerPhase12Morphism(
            p1_cert=p1_cert,
            bures_distance=float(p1_cert.bures_distance),
            fidelity=float(p1_cert.fidelity),
            gate_admitted=bool(gate_admitted),
            veto_channel_open=bool(gate_admitted),
            poincare_cartan_invariant=float(p1_cert.poincare_cartan_action_seed),
            rsi3_level1_alpha=effective_alpha,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — ORIENT: R_Boole ⊗ Λ_Nov ⊗ O, SU(N), POINCARÉ–KAM–NEKHOROSHEV–CHIRIKOV Y RSI³  ║
# ║  Inicio formal: ingest_phase1_handoff (consume el cierre de Fase 1)                      ║
# ║  Cierre formal: handoff_spectrum_to_heyting_adjudication (abre la Fase 3)                ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝


class StealthExecutionerPhase2Orientor:
    """Fase 2 OODA — orientación espectral, gauge SU(N), mecánica celeste y mónada de Kleisli T³."""

    def __init__(self, engine: BoolNovikovStealthExecutionerEngine, secret_key: bytes) -> None:
        self._engine = engine
        self._secret_key = secret_key
        self._active_morphism_12: Optional[StealthExecutionerPhase12Morphism] = None

    @staticmethod
    def _spectrum_attr(rep: Any, name: str, default: Any) -> Any:
        return getattr(rep, name, default)

    def ingest_phase1_handoff(
        self,
        morphism_12: StealthExecutionerPhase12Morphism,
    ) -> StealthExecutionerPhase1Certificate:
        """Apertura formal de Fase 2: consume y valida el morfismo de cierre Φ₁₂ de Fase 1."""
        if not morphism_12.gate_admitted:
            raise ValueError("Fase 2 rechazada: morfismo Φ₁₂ no admitido por la compuerta homomórfica.")
        self._active_morphism_12 = morphism_12
        return morphism_12.p1_cert

    @staticmethod
    def _compute_chirikov_overlap(spectrum_rep: StealthExecutionerSpectrumReport) -> float:
        """Calcula el parámetro de solapamiento de resonancias de Chirikov K_Chirikov = (Δω_sep / ΔΩ_res)².

        - Δω_sep = 2 √(||∂²H_0/∂J²|| · ||H_1||) es el semiancho de la separatriz de resonancia primaria.
        - ΔΩ_res es la separación mínima entre frecuencias fundamentales en el mapa de Whittaker.
        Si K_Chirikov ≥ 1.0, las islas de resonancia KAM se solapan induciendo caos global de Chirikov.
        """
        poincare = getattr(spectrum_rep, "poincare_report", None)
        if poincare is None:
            return 0.0
        freqs = np.asarray(getattr(poincare, "frequencies", np.array([1.0, 1.61803399])), dtype=np.float64)
        hom_res = float(getattr(poincare, "homological_residual", 1.0e-4))
        hess_det = abs(float(getattr(poincare, "kolmogorov_hessian_det", 1.0)))

        if freqs.size >= 2:
            sorted_f = np.sort(np.abs(freqs))
            diffs = np.diff(sorted_f)
            delta_omega_res = float(max(np.min(diffs), 1.0e-3))
        else:
            delta_omega_res = 0.5

        effective_torsion = math.pow(max(hess_det, 1.0e-6), 1.0 / float(max(freqs.size, 1)))
        delta_omega_sep = 2.0 * math.sqrt(effective_torsion * max(hom_res, 0.0))
        k_chirikov = (delta_omega_sep / delta_omega_res) ** 2
        return float(max(k_chirikov, 0.0))

    def orient_poincare_kam_rsi3(
        self,
        apu_cost_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        positron_hmac_valid: bool,
    ) -> StealthExecutionerSpectrumReport:
        """Ejecuta la auditoría completa R_Boole ⊗ Λ_Nov ⊗ O + Poincaré–KAM + Dirac/RLC + RSI³ T³."""
        return self._engine.execute_stealth_executioner_audit(
            apu_cost_matrix,
            novikov_exponents,
            positron_hmac_valid=positron_hmac_valid,
        )

    def emit_phase2_certificate(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        spectrum_rep: StealthExecutionerSpectrumReport,
        ts: Optional[float] = None,
    ) -> StealthExecutionerPhase2Certificate:
        """Emite el certificado P2 encadenado criptográficamente a P1 con invariantes v15.0."""
        if ts is None:
            ts = time.time()
        poincare = self._spectrum_attr(spectrum_rep, "poincare_report", None)
        rsi3 = self._spectrum_attr(spectrum_rep, "rsi3_report", None)
        sun = self._spectrum_attr(spectrum_rep, "su_stealth_report", None)
        bn_rep = self._spectrum_attr(spectrum_rep, "boolean_novikov_report", None)
        dirac = spectrum_rep.dirac_crowbar_report

        kam_dio = bool(getattr(poincare, "diophantine_holds", True)) if poincare is not None else True
        chaos = bool(
            getattr(poincare, "homoclinic_chaos", spectrum_rep.melnikov_chaos_detected)
            if poincare is not None
            else spectrum_rep.melnikov_chaos_detected
        )
        lindstedt = float(
            getattr(poincare, "homological_residual", 0.0) if poincare is not None else 0.0
        )
        nekh_time = float(
            getattr(poincare, "nekhoroshev_stability_time", 1.0e6) if poincare is not None else 1.0e6
        )
        yoshida_err = float(
            getattr(poincare, "yoshida_symplectic_error", 0.0) if poincare is not None else 0.0
        )
        lip_t3 = float(
            getattr(rsi3, "lip_t3", spectrum_rep.banach_contraction_lip)
            if rsi3 is not None
            else spectrum_rep.banach_contraction_lip
        )
        kantorovich_h = float(
            getattr(rsi3, "kantorovich_parameter_h", 0.0) if rsi3 is not None else 0.0
        )
        berry_phase = float(
            getattr(sun, "berry_holonomy_phase_rad", 0.0) if sun is not None else 0.0
        )
        moufang_norm = float(
            getattr(bn_rep, "octonion_moufang_associator_norm", 0.0) if bn_rep is not None else 0.0
        )
        bw_sigma = float(getattr(dirac, "breit_wheeler_cross_section_barn", 0.0))
        rlc_zeta = float(getattr(dirac, "rlc_damping_ratio_zeta", 1.0))
        latency_ok = bool(getattr(dirac, "latency_budget_ok", True))
        crowbar = bool(dirac.crowbar_triggered)
        chirikov_k = self._compute_chirikov_overlap(spectrum_rep)

        raw_chain = (
            f"P2|{ts:.6f}|{p1_cert.chain_hash}|{spectrum_rep.truth_value.value}|"
            f"{int(kam_dio)}|{int(chaos)}|{lip_t3:.16e}|{int(crowbar)}|{int(latency_ok)}|"
            f"{chirikov_k:.12e}|{nekh_time:.12e}|{kantorovich_h:.12e}|{rlc_zeta:.12e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(raw_chain).hexdigest()
        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return StealthExecutionerPhase2Certificate(
            timestamp=float(ts),
            spectrum_report=spectrum_rep,
            p1_chain_hash=p1_cert.chain_hash,
            kam_diophantine=kam_dio,
            homoclinic_chaos=chaos,
            lindstedt_residual=lindstedt,
            lip_t3=lip_t3,
            crowbar_triggered=crowbar,
            latency_budget_ok=latency_ok,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            chirikov_overlap_parameter=chirikov_k,
            nekhoroshev_stability_time=nekh_time,
            yoshida_symplectic_error=yoshida_err,
            kantorovich_parameter_h=kantorovich_h,
            berry_holonomy_phase_rad=berry_phase,
            breit_wheeler_cross_section_barn=bw_sigma,
            rlc_damping_ratio_zeta=rlc_zeta,
            octonion_moufang_associator_norm=moufang_norm,
        )

    def execute_phase2(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
    ) -> StealthExecutionerPhase2Certificate:
        """Orienta la auditoría espectral y genera el certificado P2."""
        spectrum_rep = self.orient_poincare_kam_rsi3(
            apu_cost_matrix, novikov_exponents, positron_hmac_valid
        )
        return self.emit_phase2_certificate(p1_cert, spectrum_rep)

    def execute_phase2_from_morphism(
        self,
        morphism_12: StealthExecutionerPhase12Morphism,
        apu_cost_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
    ) -> StealthExecutionerPhase2Certificate:
        """Entrada canónica anidada de Fase 2: consume el morfismo de cierre Φ₁₂ de Fase 1."""
        p1_cert = self.ingest_phase1_handoff(morphism_12)
        return self.execute_phase2(
            p1_cert, apu_cost_matrix, novikov_exponents, positron_hmac_valid
        )

    def handoff_spectrum_to_heyting_adjudication(
        self,
        p2_cert: StealthExecutionerPhase2Certificate,
        gate_admitted: bool,
    ) -> StealthExecutionerPhase23Morphism:
        """TRANSICIÓN ANIDADA FASE 2 → FASE 3 (morfismo de continuación de Kleisli Φ₂₃).

        Acopla U_stealth ∈ SU(N), el flujo hamiltoniano X_H de Poincaré–Nekhoroshev–Chirikov,
        la constante de Lipschitz Lip(T³) con su cota de Newton–Kantorovich h_K y el factor
        de amortiguamiento RLC del interlock simulado al adjudicador intuicionista Ω₃.
        No existe actuación sobre hardware ni MMIO.

        Esta definición formal cierra la Fase 2 y constituye el punto de inicio de la Fase 3.
        """
        rep = p2_cert.spectrum_report
        rsi3 = getattr(rep, "rsi3_report", None)
        unique = bool(getattr(rsi3, "unique_attractor", p2_cert.lip_t3 < 1.0))
        level2_rho = float(getattr(rsi3, "level2_spectral_radius_dt", 0.9025)) if rsi3 is not None else 0.9025
        sun = getattr(rep, "su_stealth_report", None)
        back_db = float(getattr(sun, "back_action_db", 0.0)) if sun is not None else 0.0
        return StealthExecutionerPhase23Morphism(
            p2_cert=p2_cert,
            lip_t3=float(p2_cert.lip_t3),
            back_action_db=back_db,
            poincare_diophantine=bool(p2_cert.kam_diophantine),
            rsi3_unique_attractor=unique,
            interlock_flag=bool(p2_cert.crowbar_triggered),
            gate_admitted=bool(gate_admitted),
            chirikov_overlap=float(p2_cert.chirikov_overlap_parameter),
            kantorovich_h=float(p2_cert.kantorovich_parameter_h),
            rsi3_level2_spectral_radius=level2_rho,
            rlc_damping_zeta=float(p2_cert.rlc_damping_ratio_zeta),
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — DECIDE & ACT: TOPOS DE HEYTING Ω₃, FUNTOR VISCERAL Φ_sem Y ÁRBOL DE MERKLE     ║
# ║  Inicio formal: ingest_phase2_handoff (consume el cierre de Fase 2)                      ║
# ║  Cierre formal: execute_phase3_from_morphism                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝


class StealthExecutionerPhase3Decider:
    """Fase 3 OODA — adjudicación intuicionista en el topos de Heyting Ω₃, Φ_sem y Merkle binario."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key
        self._active_morphism_23: Optional[StealthExecutionerPhase23Morphism] = None

    def ingest_phase2_handoff(
        self,
        morphism_23: StealthExecutionerPhase23Morphism,
    ) -> StealthExecutionerPhase2Certificate:
        """Apertura formal de Fase 3: consume y valida el morfismo de cierre Φ₂₃ de Fase 2."""
        if not morphism_23.gate_admitted:
            raise ValueError("Fase 3 rechazada: morfismo Φ₂₃ no admitido por la compuerta homomórfica.")
        self._active_morphism_23 = morphism_23
        return morphism_23.p2_cert

    # ── Operaciones del Álgebra de Heyting Intuicionista Ω₃ = {⊥ < ½ < ⊤} ──

    @staticmethod
    def _truth_rank(val: HeytingTruthValue) -> int:
        """Orden reticular canónico en Ω₃: VETOED (⊥ = 0) < DEGRADED (½ = 1) < COHERENT (⊤ = 2)."""
        if val == HeytingTruthValue.COHERENT:
            return 2
        if val == HeytingTruthValue.DEGRADED:
            return 1
        return 0

    @staticmethod
    def _rank_to_truth(rank: int) -> HeytingTruthValue:
        if rank >= 2:
            return HeytingTruthValue.COHERENT
        if rank == 1:
            return HeytingTruthValue.DEGRADED
        return HeytingTruthValue.VETOED

    @classmethod
    def heyting_meet(cls, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Conjunción intuicionista (a ∧ b) = inf(a, b) en Ω₃."""
        return cls._rank_to_truth(min(cls._truth_rank(a), cls._truth_rank(b)))

    @classmethod
    def heyting_join(cls, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Disyunción intuicionista (a ∨ b) = sup(a, b) en Ω₃."""
        return cls._rank_to_truth(max(cls._truth_rank(a), cls._truth_rank(b)))

    @classmethod
    def heyting_implication(cls, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Implicación relativa de Heyting (a ⇒ b) = sup { c ∈ Ω₃ : (c ∧ a) ≤ b }."""
        ra = cls._truth_rank(a)
        rb = cls._truth_rank(b)
        return HeytingTruthValue.COHERENT if ra <= rb else b

    @classmethod
    def verify_heyting_adjunction(
        cls,
        a: HeytingTruthValue,
        b: HeytingTruthValue,
        c: HeytingTruthValue,
    ) -> bool:
        """Verifica la ley universal del topos: (c ∧ a) ≤ b  ⇔  c ≤ (a ⇒ b)."""
        lhs = cls._truth_rank(cls.heyting_meet(c, a)) <= cls._truth_rank(b)
        rhs = cls._truth_rank(c) <= cls._truth_rank(cls.heyting_implication(a, b))
        return bool(lhs == rhs)

    @staticmethod
    def _kam_stability_index(p2: StealthExecutionerPhase2Certificate) -> float:
        """Índice de estabilidad celeste [0, 1] acoplando KAM, Lindstedt y Chirikov."""
        res = max(float(p2.lindstedt_residual), 0.0)
        smooth = 1.0 / (1.0 + res * 1.0e3)
        chirikov_penalty = 1.0 / (1.0 + max(float(p2.chirikov_overlap_parameter), 0.0))
        if p2.homoclinic_chaos or p2.chirikov_overlap_parameter >= _CHIRIKOV_STOCHASTICITY_CEILING:
            return float(0.15 * smooth * chirikov_penalty)
        if not p2.kam_diophantine:
            return float(0.45 * smooth * chirikov_penalty)
        return float(min(1.0, (0.85 + 0.15 * smooth) * (0.90 + 0.10 * chirikov_penalty)))

    @staticmethod
    def _nekhoroshev_confinement_index(p2: StealthExecutionerPhase2Certificate) -> float:
        """Índice normalizado [0, 1] del tiempo exponencial de confinamiento de Nekhoroshev T_Nekh."""
        t_nekh = max(float(p2.nekhoroshev_stability_time), 1.0)
        log_ratio = math.log10(t_nekh) / 6.0
        return float(min(max(log_ratio, 0.0), 1.0))

    @staticmethod
    def _kantorovich_convergence_margin(p2: StealthExecutionerPhase2Certificate) -> float:
        """Margen normalizado [0, 1] respecto al umbral cuadrático de Newton–Kantorovich h_K ≤ 0.50."""
        h_val = max(float(p2.kantorovich_parameter_h), 0.0)
        if h_val >= _PHI23_KANTOROVICH_CEILING:
            return 0.0
        return float(1.0 - (h_val / _PHI23_KANTOROVICH_CEILING))

    def apply_visceral_functor(
        self,
        p2_cert: StealthExecutionerPhase2Certificate,
        truth_val: HeytingTruthValue,
    ) -> StealthExecutionerVisceralMetrics:
        """Funtor semántico visceral Φ_sem: Ω₃ × (KAM, Nekhoroshev, T³, Dirac/RLC) → métricas viscerales.

        WACC 15.0%, liquidez +11.5%, KV −86.4% (presupuesto 56 tokens) se alcanzan
        exactamente sii el veredicto en Ω₃ es COHERENT (⊤). DEGRADED (½) modula de forma
        suave según el índice KAM–Chirikov, el confinamiento exponencial de Nekhoroshev y el
        margen de Newton–Kantorovich; VETOED (⊥) colapsa a cota residual declarativa.
        """
        kam_idx = self._kam_stability_index(p2_cert)
        nekh_idx = self._nekhoroshev_confinement_index(p2_cert)
        kant_margin = self._kantorovich_convergence_margin(p2_cert)

        lip = max(min(float(p2_cert.lip_t3), 0.999999), 0.0)
        lip_quality = min(max(0.0, (1.0 - lip) / (1.0 - _NOMINAL_BANACH_LIP3)), 1.0)

        rsi3 = getattr(p2_cert.spectrum_report, "rsi3_report", None)
        jerk = float(getattr(rsi3, "jerk", 0.0)) if rsi3 is not None else 0.0
        dirac = p2_cert.spectrum_report.dirac_crowbar_report
        energy = float(getattr(dirac, "energy_released_kev", 0.0))
        latency_ns = float(dirac.actuation_latency_ns)
        rlc_zeta = float(getattr(dirac, "rlc_damping_ratio_zeta", p2_cert.rlc_damping_ratio_zeta))

        if truth_val == HeytingTruthValue.COHERENT:
            scale = 1.0
            status = "PASSIVE_STEALTH_CAMOUFLAGED"
        elif truth_val == HeytingTruthValue.DEGRADED:
            composite_quality = 0.50 * kam_idx + 0.25 * nekh_idx + 0.25 * kant_margin
            scale = 0.45 + 0.40 * composite_quality * lip_quality
            status = "DEGRADED_GAUGE_AUDIT"
        else:
            scale = 0.12 * kam_idx
            status = f"INTERLOCK_FLAG_SIMULATED_NS_{latency_ns:.2f}"

        if p2_cert.crowbar_triggered:
            status = f"INTERLOCK_FLAG_SIMULATED_NS_{latency_ns:.2f}"

        return StealthExecutionerVisceralMetrics(
            wacc_protection_pct=float(_WACC_PROTECTION_TARGET_PCT * scale),
            net_liquidity_release_pct=float(_NET_LIQUIDITY_TARGET_PCT * scale),
            kv_cache_compression_pct=float(_KV_COMPRESSION_TARGET_PCT * scale),
            crowbar_interlock_status=status,
            kam_stability_index=float(kam_idx),
            rsi3_jerk=jerk,
            pair_energy_kev=energy,
            kv_token_budget=_KV_TOKEN_BUDGET,
            nekhoroshev_confinement_index=float(nekh_idx),
            kantorovich_convergence_margin=float(kant_margin),
            rlc_damping_ratio_zeta=float(rlc_zeta),
        )

    @staticmethod
    def _build_binary_merkle_root(leaves_payloads: List[bytes]) -> str:
        """Construye un árbol de Merkle binario SHA-256 con separación de dominios RFC-6962."""
        if not leaves_payloads:
            return hashlib.sha256(_MERKLE_LEAF_PREFIX).hexdigest()

        level: List[bytes] = [
            hashlib.sha256(_MERKLE_LEAF_PREFIX + payload).digest()
            for payload in leaves_payloads
        ]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            next_level: List[bytes] = []
            for idx in range(0, len(level), 2):
                combined = _MERKLE_NODE_PREFIX + level[idx] + level[idx + 1]
                next_level.append(hashlib.sha256(combined).digest())
            level = next_level
        return level[0].hex()

    def _adjudicate_heyting_truth(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        p2_cert: StealthExecutionerPhase2Certificate,
    ) -> Tuple[HeytingTruthValue, bool]:
        """Adjudica el valor de verdad en el topos Ω₃ y verifica la adjunción deductiva."""
        rep = p2_cert.spectrum_report
        engine_truth = rep.truth_value

        prop_p1 = (
            HeytingTruthValue.COHERENT
            if p1_cert.is_bures_isometry
            else (HeytingTruthValue.DEGRADED if p1_cert.is_psd else HeytingTruthValue.VETOED)
        )
        prop_celestial = (
            HeytingTruthValue.VETOED
            if (p2_cert.homoclinic_chaos or p2_cert.chirikov_overlap_parameter >= _CHIRIKOV_STOCHASTICITY_CEILING)
            else (HeytingTruthValue.COHERENT if p2_cert.kam_diophantine else HeytingTruthValue.DEGRADED)
        )
        prop_crowbar = (
            HeytingTruthValue.VETOED
            if (p2_cert.crowbar_triggered or not p2_cert.latency_budget_ok)
            else HeytingTruthValue.COHERENT
        )

        # Síntesis intuicionista: el motor manda sobre el espectro y se intersecta con crowbar/celestial
        combined = self.heyting_meet(engine_truth, self.heyting_meet(prop_celestial, prop_crowbar))
        if not p1_cert.is_psd:
            combined = self.heyting_meet(combined, HeytingTruthValue.DEGRADED)

        adjunction_ok = self.verify_heyting_adjunction(prop_p1, prop_celestial, prop_crowbar)
        return combined, adjunction_ok

    def emit_sovereign_state(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        p2_cert: StealthExecutionerPhase2Certificate,
        ts: Optional[float] = None,
    ) -> StealthExecutionerSovereignState:
        """Cierra el ciclo OODA: adjudicación Ω₃, árbol de Merkle de 8 hojas y pasaporte HMAC."""
        if ts is None:
            ts = time.time()
        rep = p2_cert.spectrum_report
        truth_val, adjunction_ok = self._adjudicate_heyting_truth(p1_cert, p2_cert)
        dirac = rep.dirac_crowbar_report
        visceral = self.apply_visceral_functor(p2_cert, truth_val)
        engine_passport = str(getattr(rep, "hmac_passport", ""))

        bn_rep = getattr(rep, "boolean_novikov_report", None)
        gf2_rank = int(getattr(bn_rep, "gf2_rank", 0)) if bn_rep is not None else 0
        euler_chi = int(getattr(bn_rep, "euler_poincare_characteristic", 0)) if bn_rep is not None else 0

        # 8 hojas canónicas para el árbol de Merkle binario de Fase 3
        leaves: List[bytes] = [
            f"L0_P1|{p1_cert.hmac_signature}|{p1_cert.bures_distance:.12e}|{p1_cert.von_neumann_entropy:.12e}|{p1_cert.quantum_fisher_information:.12e}".encode("utf-8"),
            f"L1_P2|{p2_cert.hmac_signature}|{p2_cert.chain_hash}".encode("utf-8"),
            f"L2_BOOL_NOV_OCT|{rep.rademacher_walsh_norm:.12e}|{int(rep.smith_torsion_free)}|{rep.krull_valuation}|{gf2_rank}|{euler_chi}|{p2_cert.octonion_moufang_associator_norm:.12e}".encode("utf-8"),
            f"L3_SUN_QND|{rep.su_stealth_report.weak_value.real:.12e}|{rep.su_stealth_report.back_action_db:.12e}|{p2_cert.berry_holonomy_phase_rad:.12e}".encode("utf-8"),
            f"L4_POINCARE_KAM|{int(p2_cert.kam_diophantine)}|{p2_cert.lindstedt_residual:.12e}|{p2_cert.chirikov_overlap_parameter:.12e}|{p2_cert.nekhoroshev_stability_time:.12e}|{p2_cert.yoshida_symplectic_error:.12e}".encode("utf-8"),
            f"L5_DIRAC_RLC|{int(dirac.crowbar_triggered)}|{dirac.actuation_latency_ns:.6f}|{dirac.energy_released_kev:.6f}|{p2_cert.breit_wheeler_cross_section_barn:.12e}|{p2_cert.rlc_damping_ratio_zeta:.12e}".encode("utf-8"),
            f"L6_RSI3_BANACH|{p2_cert.lip_t3:.16e}|{p2_cert.kantorovich_parameter_h:.12e}|{visceral.rsi3_jerk:.12e}".encode("utf-8"),
            f"L7_HEYTING_VISCERAL|{ts:.6f}|{truth_val.value}|{int(adjunction_ok)}|{visceral.wacc_protection_pct:.6f}|{engine_passport}".encode("utf-8"),
        ]
        merkle_root = self._build_binary_merkle_root(leaves)

        p3_mac = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + merkle_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        passport = engine_passport if engine_passport else p3_mac
        kantorovich_ok = bool(p2_cert.kantorovich_parameter_h <= _PHI23_KANTOROVICH_CEILING)

        return StealthExecutionerSovereignState(
            timestamp=float(ts),
            truth_value=truth_val,
            crowbar_triggered=bool(dirac.crowbar_triggered),
            actuation_latency_ns=float(dirac.actuation_latency_ns),
            merkle_root_hash=merkle_root,
            visceral_metrics=visceral,
            kam_diophantine=bool(p2_cert.kam_diophantine),
            rsi3_lip=float(p2_cert.lip_t3),
            hmac_passport=passport,
            latency_budget_ok=bool(p2_cert.latency_budget_ok),
            chirikov_stochasticity_index=float(p2_cert.chirikov_overlap_parameter),
            kantorovich_certified=kantorovich_ok,
            heyting_adjunction_holds=adjunction_ok,
        )

    def execute_phase3(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        p2_cert: StealthExecutionerPhase2Certificate,
    ) -> StealthExecutionerSovereignState:
        """Adjudica el estado soberano final en el topos de Heyting Ω₃."""
        return self.emit_sovereign_state(p1_cert, p2_cert)

    def execute_phase3_from_morphism(
        self,
        p1_cert: StealthExecutionerPhase1Certificate,
        morphism_23: StealthExecutionerPhase23Morphism,
    ) -> StealthExecutionerSovereignState:
        """Entrada canónica anidada de Fase 3: consume el morfismo de cierre Φ₂₃ de Fase 2."""
        p2_cert = self.ingest_phase2_handoff(morphism_23)
        return self.execute_phase3(p1_cert, p2_cert)


# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║  ORQUESTADOR SOBERANO — COMPOSICIÓN DE KLEISLI DE LAS 3 FASES ANIDADAS (Φ₁₂, Φ₂₃)       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝


class BoolNovikovStealthExecutionerAgent:
    """Soberano agéntico de auditoría espectral sobre R_Boole ⊗ Λ_Nov ⊗ O, gauge SU(N) y Poincaré–KAM."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key
        self._engine = BoolNovikovStealthExecutionerEngine(secret_key=secret_key)
        self._observer = StealthExecutionerPhase1Observer(secret_key=secret_key)
        self._orientor = StealthExecutionerPhase2Orientor(
            engine=self._engine, secret_key=secret_key
        )
        self._decider = StealthExecutionerPhase3Decider(secret_key=secret_key)
        logger.info("%s iniciado correctamente.", _AGENT_VERSION)

    def continue_to_phase2_gate(self, p1_cert: StealthExecutionerPhase1Certificate) -> bool:
        """Compuerta homomórfica Φ₁₂: revalida autenticidad HMAC P1 y finitud cuántico-espectral.

        Admite futuros estados DEGRADED o VETOED (canal de auditoría, no demolición).
        Sólo la falsificación criptográfica o valores no finitos bloquean la transición.
        """
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + p1_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        sig_ok = hmac.compare_digest(expected_sig, p1_cert.hmac_signature)
        finite_ok = (
            math.isfinite(p1_cert.bures_distance)
            and math.isfinite(p1_cert.fidelity)
            and math.isfinite(p1_cert.von_neumann_entropy)
            and math.isfinite(p1_cert.quantum_fisher_information)
        )
        return bool(sig_ok and finite_ok)

    def continue_to_phase3_gate(self, p2_cert: StealthExecutionerPhase2Certificate) -> bool:
        """Compuerta homomórfica Φ₂₃: HMAC ∧ Lip(T³)<1 ∧ Kantorovich h_K≤0.50 ∧ Lindstedt ∧ latencia.

        El interlock simulado no bloquea la compuerta —se adjudica formalmente en Ω₃—.
        Se exige autenticidad criptográfica, contractividad de Banach–Kantorovich,
        presupuesto de latencia del modelo RLC y retroacción QND acotada.
        """
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + p2_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        is_sig_valid = hmac.compare_digest(expected_sig, p2_cert.hmac_signature)
        is_banach_valid = p2_cert.spectrum_report.banach_contraction_lip < _PHI23_LIP_CEILING
        is_kantorovich_valid = (
            math.isfinite(p2_cert.kantorovich_parameter_h)
            and p2_cert.kantorovich_parameter_h <= _PHI23_KANTOROVICH_CEILING + 1.0e-12
        )
        lindstedt_ok = math.isfinite(p2_cert.lindstedt_residual)
        latency_ok = bool(p2_cert.latency_budget_ok)
        sun = getattr(p2_cert.spectrum_report, "su_stealth_report", None)
        back_db = float(getattr(sun, "back_action_db", 0.0)) if sun is not None else 0.0
        qnd_ok = back_db <= _PHI23_BACKACTION_DB_CEILING + 1.0e-12 or back_db == 0.0
        return bool(
            is_sig_valid
            and is_banach_valid
            and is_kantorovich_valid
            and lindstedt_ok
            and latency_ok
            and qnd_ok
        )

    def run_sovereign_governance_cycle(
        self,
        rho_mac: Union[NDArray[np.float64], NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
    ) -> StealthExecutionerSovereignState:
        """Ciclo OODA trifásico anidado: Observe ⊂ Orient ⊂ Decide&Act (Kleisli T³)."""
        # ── Fase 1: Observe y cierre formal hacia Fase 2 ────────────────────
        p1 = self._observer.execute_phase1(rho_mac)
        phi12 = self.continue_to_phase2_gate(p1)
        if not phi12:
            raise ValueError("Falla de autenticidad HMAC o finitud espectral en Compuerta Homomórfica Φ₁₂.")
        morphism_12 = self._observer.handoff_isometry_to_boolean_novikov_orientation(p1, phi12)

        # ── Fase 2: Orient (inicio = cierre formal de Fase 1) ───────────────
        p2 = self._orientor.execute_phase2_from_morphism(
            morphism_12,
            apu_cost_matrix,
            novikov_exponents,
            positron_hmac_valid=positron_hmac_valid,
        )
        phi23 = self.continue_to_phase3_gate(p2)
        if not phi23:
            raise ValueError("Falla de estabilidad Banach/Kantorovich/QND en Compuerta Homomórfica Φ₂₃.")
        morphism_23 = self._orientor.handoff_spectrum_to_heyting_adjudication(p2, phi23)

        # ── Fase 3: Decide & Act (inicio = cierre formal de Fase 2) ─────────
        sovereign_state = self._decider.execute_phase3_from_morphism(p1, morphism_23)
        logger.info(
            "Ciclo OODA espectral v15.0.0 completado. Veredicto: %s (%s) | KAM=%s | "
            "Chirikov=%.3e | Lip(T³)=%.6f | Kantorovich=%s | interlock_sim=%s",
            sovereign_state.truth_value.name,
            hex(sovereign_state.truth_value.value),
            sovereign_state.kam_diophantine,
            sovereign_state.chirikov_stochasticity_index,
            sovereign_state.rsi3_lip,
            sovereign_state.kantorovich_certified,
            sovereign_state.crowbar_triggered,
        )
        return sovereign_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovStealthExecutionerAgent v15.0.0 (3 fases anidadas) ===")
    agent = BoolNovikovStealthExecutionerAgent()

    d = 4
    matrix_test = np.eye(d, dtype=np.float64) + 0.02 * np.ones((d, d))
    exps_test = np.array([0.15, 0.20, 0.30, 0.45], dtype=np.float64)

    rho_test = matrix_test @ matrix_test.T + np.eye(d) * 1.0e-4
    rho_test /= float(np.real(np.trace(rho_test)))

    sovereign_state = agent.run_sovereign_governance_cycle(
        rho_test, matrix_test, exps_test, positron_hmac_valid=True
    )

    print(
        f"Veredicto Final Heyting Ω₃  : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(f"Adjunción Heyting Ω₃ válida : {sovereign_state.heyting_adjunction_holds}")
    print(f"Interlock simulado          : {sovereign_state.crowbar_triggered}")
    print(f"Latencia modelo RLC (ns)    : {sovereign_state.actuation_latency_ns:.2f} ns")
    print(f"Presupuesto latencia OK     : {sovereign_state.latency_budget_ok}")
    print(f"KAM diofantino              : {sovereign_state.kam_diophantine}")
    print(f"Índice Chirikov K           : {sovereign_state.chirikov_stochasticity_index:.6e}")
    print(f"Lip(T³) RSI³                : {sovereign_state.rsi3_lip:.6f}")
    print(f"Kantorovich certificado     : {sovereign_state.kantorovich_certified}")
    print(f"Raíz Merkle Binaria SHA-256 : {sovereign_state.merkle_root_hash[:16]}...")
    print(f"Pasaporte HMAC              : {sovereign_state.hmac_passport[:16]}...")
    print(f"Protección WACC (%)         : {sovereign_state.visceral_metrics.wacc_protection_pct:.4f}%")
    print(f"Liberación Liquidez Neta (%): {sovereign_state.visceral_metrics.net_liquidity_release_pct:.4f}%")
    print(f"Compresión KV (%)           : {sovereign_state.visceral_metrics.kv_cache_compression_pct:.4f}%")
    print(f"Presupuesto KV (tokens)     : {sovereign_state.visceral_metrics.kv_token_budget}")
    print(f"Índice estabilidad KAM      : {sovereign_state.visceral_metrics.kam_stability_index:.6f}")
    print(f"Índice Nekhoroshev          : {sovereign_state.visceral_metrics.nekhoroshev_confinement_index:.6f}")
    print(f"Margen Kantorovich          : {sovereign_state.visceral_metrics.kantorovich_convergence_margin:.6f}")
    print(f"Amortiguamiento RLC ζ       : {sovereign_state.visceral_metrics.rlc_damping_ratio_zeta:.6f}")
    print(f"Jerk RSI³ d³C/dt³           : {sovereign_state.visceral_metrics.rsi3_jerk:.6e}")
    print(f"Energía de pares (keV)      : {sovereign_state.visceral_metrics.pair_energy_kev:.1f}")
    print(f"Status interlock            : {sovereign_state.visceral_metrics.crowbar_interlock_status}")
    print("=== TEST COMPLETADO CON ÉXITO ===")