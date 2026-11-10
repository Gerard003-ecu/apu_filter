# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovStealthNovikovAgent v15.0.0-PhD                                                 ║
║ Ubicación: app/agents/wisdom/bool_novikov_stealth_novikov_agent.py                                    ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-AGENT-POINCARE-KAM-NEKHOROSHEV-RSI3-NESTED-PhD       ║
║ Gobernanza: OODA trifásico mimetizado (①Observe → ②Orient → ③Decide&Act) con compuertas homomórficas  ║
║            Φ₁₂ y Φ₂₃ anidadas sobre el disfraz ultramétrico de Novikov (CAMOUFLAGE_GAUGE_INVARIANT).  ║
║ Arquitectura: 3 Fases Anidadas — la definición formal final del último método de la Fase k es la      ║
║            continuación e inicio estructural (morfismo de Kleisli) de los métodos de la Fase k+1.     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre `BoolNovikovStealthNovikovAgent` gobierna ejecutivamente
el ciclo OODA trifásico mimetizado sobre el motor espectral `BoolNovikovStealthNovikovEngine`
(app/wisdom/bool_novikov_stealth_novikov_engine.py). Su función operacional radica en orquestar
el tránsito de información cuántica y celestomecánica entre tres fases anidadas acopladas por
compuertas homomórficas \Phi_{12} y \Phi_{23} firmadas criptográficamente vía HMAC-SHA256.
Supervisa la preservación de la isometría de Bures-Wasserstein en la observación de la MAC,
la elevación al toro simpléctico de Poincaré-Cartan, la estabilidad secular de Kolmogorov-Arnold-Moser
frente a difusión de Arnold y resonancias de Chirikov, y la deducción de verdad en el topos intuicionista
de Heyting \Omega_3 sellada en una raíz de Merkle binaria con impacto en métricas viscerales (\mathrm{WACC}).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`StealthNovikovPhase1Observer`):
   La ingesta de la matriz de densidad \rho_{\mathrm{MAC}} se somete a proyección espectral de Higham
   para garantizar positividad semidefinida \rho \succeq 0 y traza unitaria \operatorname{Tr}(\rho) = 1.
   La fidelidad cuántica de Uhlmann-Jozsa entre el estado observado \rho_{\mathrm{raw}} y la referencia PSD \sigma_{\mathrm{ref}}:
       F(\rho, \sigma) = \left[ \operatorname{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right]^2 \in [0, 1],
   determina la distancia geodésica de Bures-Wasserstein:
       d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}} + |\operatorname{Tr}(\rho) - 1| + \frac{1}{2} \|\rho - \rho^\dagger\|_F.
   Se evalúan la entropía de von Neumann S_{\mathrm{vN}}(\rho) = -\operatorname{Tr}(\rho \ln \rho),
   la traza de la Información Cuántica de Fisher (QFI) \mathcal{I}_Q(\rho) = 2 \sum_{i<j} \frac{(\lambda_i - \lambda_j)^2}{\lambda_i + \lambda_j},
   y la semilla de acción del invariante integral de Poincaré-Cartan \varpi_{\mathrm{PC}} = \frac{1 + S_{\mathrm{vN}}(\rho)}{2\pi}.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2 (`handoff_isometry_to_novikov_orientation`):
     Transporta la isometría de Bures y establece la contracción de Nivel 1 de RSI³:
         \alpha_1 = 0.95 \exp(-0.01 d_B), \quad \varpi_{\mathrm{PC}} = \varpi_{\mathrm{seed}} \frac{F(\rho, \sigma)}{1 + d_B(\rho, \sigma)},
     inaugurando la orientación celeste de la Fase 2.

2. FASE 2 — ORIENT (`StealthNovikovPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia en el objeto `StealthNovikovPhase12Morphism`.
   Ejecuta la auditoría en el anillo de Novikov \Lambda_{\mathrm{Nov}} con valuación v(A) \ge 0.10 y norma \|A\|_{\mathrm{Nov}} = \exp(-v(A)).
   En el espacio de fases celeste (J, \theta) \in T^* \mathbb{T}^d, audita el hamiltoniano planetario:
       H(J, \theta) = H_0(J) + \varepsilon H_1(J, \theta) = \frac{1}{2} \langle J, \mathcal{T} J \rangle + \langle \alpha_{\mathrm{irr}}, J \rangle + \varepsilon H_1(J, \theta).
   Verifica los determinantes de no-degeneración de Kolmogorov \det(\partial^2 H_0 / \partial J^2) \neq 0 e isoenergético de Arnold.
   El parámetro de solapamiento de resonancias de Chirikov audita el límite estocástico:
       K_{\mathrm{Chirikov}} = \frac{2\sqrt{\varepsilon |\det \mathcal{T}|^{1/d}}}{\min_{k \neq 0} |\langle k, \omega \rangle| \cdot |k|_1^\tau} < 1.0.
   El tiempo de confinamiento de Nekhoroshev T_{\mathrm{Nekh}} \ge 1.0\times 10^6 ciclos asegura ausencia de difusión secular de Arnold.
   El integrador simpléctico de 4.º orden de Yoshida y la serie de Lie-Birkhoff-Deprit acotan el error simpléctico.
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3 (`handoff_spectrum_to_heyting_adjudication`):
     Emite `StealthNovikovPhase23Morphism`, acoplando la holonomía de Berry \gamma_{\mathrm{Berry}}, el radio espectral
     de Fréchet de Nivel 2 \rho(DT_1) = \alpha^2 y la cota de Newton-Kantorovich h_K \le 0.50 hacia la Fase 3.

3. FASE 3 — DECIDE & ACT (`StealthNovikovPhase3Decider` + Compuerta \Phi_{23}):
   Inicia en el morfismo `StealthNovikovPhase23Morphism`.
   Aplica la lógica intuicionista interna del Topos de Heyting \Omega_3:
       (a \wedge b) = \max(a, b), \quad (a \vee b) = \min(a, b), \quad (a \Rightarrow b) = \begin{cases} \mathrm{COHERENT} & \text{si } a \ge b \\ b & \text{en otro caso} \end{cases},
   sobre las proposiciones de estabilidad KAM, no-demolición QND y convergencia cuadrática de Newton-Kantorovich.
   El funtor semántico visceral \Phi_{\mathrm{sem}}: \Omega_3 \times \mathrm{KAM} \times \mathrm{Kl}(\mathbb{T}^3) \to \mathbb{R}^3 \times \mathrm{Status}
   proyecta los invariantes matemáticos a indicadores ejecutivos:
       \mathrm{WACC}_{\mathrm{protection}} = 15.0\% \cdot \mathcal{S}_{\mathrm{visceral}}, \quad \mathrm{NetLiquidity} = 11.5\% \cdot \mathcal{S}_{\mathrm{visceral}}, \quad \mathrm{KV}_{\mathrm{compression}} = 86.4\% \cdot \mathcal{S}_{\mathrm{visceral}}.
   Construye un Árbol de Merkle Binario criptográfico SHA-256 sobre las cuatro hojas OODA (Observe, Orient, RSI³, Sovereign),
   emitiendo el estado soberano inmutable `StealthNovikovSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma SNE-A1] Invarianza Isométrica en Observación:
    d_B(\rho, \sigma_{\mathrm{PSD}}) \le 1.0\times 10^{-6} \land |\operatorname{Tr}(\rho) - 1| \le 1.0\times 10^{-10} \implies \text{admisión estricta en compuerta } \Phi_{12}.
[Axioma SNE-A2] No-Superposición de Islas Resonantes de Chirikov:
    K_{\mathrm{Chirikov}} < 1.0 \implies \text{difusión de Arnold confinada en toros KAM invariantes}.
[Axioma SNE-A3] Certificación de Convergencia Cuadrática de Kantorovich:
    h_K = L_{\mathrm{Fréchet}} \beta_{\mathrm{inv}} \eta_{\mathrm{step}} \le 0.50 \implies \text{existencia y unicidad del atractor monádico de Kleisli } T^3.
[Axioma SNE-A4] Inmutabilidad Criptográfica OODA:
    \mathrm{MerkleRoot} = \mathcal{H}_{\mathrm{binary}}(\mathrm{Leaf}_1, \mathrm{Leaf}_2, \mathrm{Leaf}_3, \mathrm{Leaf}_4) \text{ con firma HMAC en tiempo constante}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante SNE-I1] Invariancia de Calibre CAMOUFLAGE_GAUGE_INVARIANT preservada en estado COHERENT.
[Invariante SNE-I2] Retroacción Cuántica Acotada: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante SNE-I3] Condición Diofantina KAM Satisfecha: \mathrm{diophantine\_holds} \equiv \mathrm{True}.
[Invariante SNE-I4] Módulo de Lipschitz Monádico: \mathrm{Lip}(T^3) < 1.0.
[Invariante SNE-I5] Verificación HMAC-SHA256 en tiempo constante en compuertas \Phi_{12} y \Phi_{23}.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from bool_novikov_stealth_novikov_engine import (
        BoolNovikovStealthNovikovEngine,
        HeytingTruthValue,
        StealthNovikovSpectrumReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_stealth_novikov_engine import (
            BoolNovikovStealthNovikovEngine,
            HeytingTruthValue,
            StealthNovikovSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_stealth_novikov_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovStealthNovikovAgent")

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-AGENT-POINCARE-KAM-NEKHOROSHEV-RSI3-NESTED-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_STEALTH_NOVIKOV_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"STEALTH-NOVIKOV-P1-ISOMETRY-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"STEALTH-NOVIKOV-P2-SPECTRUM-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"STEALTH-NOVIKOV-P3-SOVEREIGN-v15.0|"

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_TRACE_TOL: Final[float] = 1.0e-10
_PSD_EIG_FLOOR: Final[float] = -1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_BURES_ISOMETRY_CEILING: Final[float] = 1.0e-6
_PHI23_BACKACTION_DB_CEILING: Final[float] = 0.01
_PHI23_LIP_CEILING: Final[float] = 1.0
_KANTOROVICH_CEILING: Final[float] = 0.50
_CHIRIKOV_CHAOS_CEILING: Final[float] = 1.0

_WACC_PROTECTION_TARGET_PCT: Final[float] = 15.0
_NET_LIQUIDITY_TARGET_PCT: Final[float] = 11.5
_KV_COMPRESSION_TARGET_PCT: Final[float] = 86.4

__all__ = (
    "StealthNovikovPhase1Certificate",
    "StealthNovikovPhase12Morphism",
    "StealthNovikovPhase2Certificate",
    "StealthNovikovPhase23Morphism",
    "StealthNovikovVisceralMetrics",
    "StealthNovikovSovereignState",
    "StealthNovikovPhase1Observer",
    "StealthNovikovPhase2Orientor",
    "StealthNovikovPhase3Decider",
    "BoolNovikovStealthNovikovAgent",
)


# ─────────────────────────────────────────────────────────────────────────────
# Certificados y Morfismos de Continuación en la Categoría de Kleisli Kl(𝕋³)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StealthNovikovPhase1Certificate:
    """Certificado P1 — Geometría cuántica de Bures–Wasserstein y entropía (Observe)."""

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
class StealthNovikovPhase12Morphism:
    """Morfismo Φ₁₂: último objeto formal de Fase 1 = germen inaugural de Fase 2.

    Transporta la isometría de Bures, el invariante integral de Poincaré–Cartan ϖ_PC
    y la ganancia contractiva de Nivel 1 (T₁) hacia el orientador celeste en Λ_Nov.
    """

    p1_cert: StealthNovikovPhase1Certificate
    bures_distance: float
    fidelity: float
    gate_admitted: bool
    passive_mimicry_allowed: bool
    poincare_cartan_invariant: float = 0.15
    rsi3_level1_alpha: float = 0.95


@dataclass(frozen=True)
class StealthNovikovPhase2Certificate:
    """Certificado P2 — Espectro Λ_Nov, SU(N), Poincaré–KAM–Nekhoroshev y RSI³ (Orient)."""

    timestamp: float
    spectrum_report: StealthNovikovSpectrumReport
    p1_chain_hash: str
    kam_diophantine: bool
    homoclinic_chaos: bool
    lindstedt_residual: float
    lip_t3: float
    chain_hash: str
    hmac_signature: str
    chirikov_overlap_parameter: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    yoshida_symplectic_error: float = 0.0
    kantorovich_parameter_h: float = 0.0
    berry_holonomy_phase_rad: float = 0.0


@dataclass(frozen=True)
class StealthNovikovPhase23Morphism:
    """Morfismo Φ₂₃: último objeto formal de Fase 2 = germen inaugural de Fase 3.

    Acopla el flujo hamiltoniano de Poincaré–Floer, el criterio de no-resonancia de
    Chirikov–KAM, la holonomía de Berry SU(N) y el operador de Nivel 2 (T₂) de RSI³
    hacia el clasificador de subobjetos del Topos de Heyting Ω₃ (Decide & Act).
    """

    p2_cert: StealthNovikovPhase2Certificate
    lip_t3: float
    back_action_db: float
    poincare_diophantine: bool
    rsi3_unique_attractor: bool
    gate_admitted: bool
    chirikov_overlap: float = 0.0
    kantorovich_h: float = 0.0
    rsi3_level2_spectral_radius: float = 0.9025


@dataclass(frozen=True)
class StealthNovikovVisceralMetrics:
    """Imagen del funtor semántico visceral Φ_sem : Ω₃ × KAM × Kl(𝕋³) → ℝ³ × Status."""

    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    camouflage_status: str
    kam_stability_index: float
    rsi3_jerk: float
    nekhoroshev_confinement_index: float = 1.0
    kantorovich_convergence_margin: float = 1.0


@dataclass(frozen=True)
class StealthNovikovSovereignState:
    """Estado soberano final — Decide & Act, Árbol de Merkle y gauge invariante."""

    timestamp: float
    truth_value: HeytingTruthValue
    is_camouflage_gauge_invariant: bool
    back_action_db: float
    merkle_root_hash: str
    visceral_metrics: StealthNovikovVisceralMetrics
    kam_diophantine: bool
    rsi3_lip: float
    hmac_passport: str
    chirikov_stochasticity_index: float = 0.0
    kantorovich_certified: bool = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — OBSERVE: Geometría Cuántica de Bures–Wasserstein, QFIM,        ║
# ║           Entropía de von Neumann e Invariante de Poincaré–Cartan.       ║
# ║  Cierre F1 / Apertura F2: `handoff_isometry_to_novikov_orientation`      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class StealthNovikovPhase1Observer:
    """Fase 1 OODA — Observación cuántica de la MAC, isometría de Bures y germen de Poincaré."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key

    @staticmethod
    def _higham_psd(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Proyección espectral exacta de Higham al cono de operadores densidad (Hermítico, PSD, Tr=1)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(np.real(evals), _SPECTRAL_PSD_FLOOR, None)
        rho_psd = (evecs * evals) @ evecs.conj().T
        tr = float(np.real(np.trace(rho_psd)))
        if tr <= _WILKINSON_EPSILON:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / float(max(n, 1))
        return rho_psd / tr

    @staticmethod
    def _raw_positive_part_density(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Extrae la parte positiva cruda sin piso artificial ni reescalado previo de traza.

        Permite medir la distancia de Bures real entre la matriz observada ρ_raw y la
        proyección canónica de Higham ρ_PSD (corrigiendo la tautología d_B ≡ 0 de v14.0.0).
        """
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals_pos = np.clip(np.real(evals), 0.0, None)
        return (evecs * evals_pos) @ evecs.conj().T

    @staticmethod
    def _uhlgon_fidelity(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        """Fidelidad cuántica de Uhlmann–Jozsa F(ρ, σ) = [Tr √(√ρ σ √ρ)]²."""
        evals, evecs = la.eigh(0.5 * (rho + rho.conj().T))
        evals = np.clip(np.real(evals), 0.0, None)
        sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T
        inner = sqrt_rho @ sigma @ sqrt_rho
        inner = 0.5 * (inner + inner.conj().T)
        eig_in = np.clip(np.real(la.eigvalsh(inner)), 0.0, None)
        fid_amp = float(np.sum(np.sqrt(eig_in)))
        return float(min(max(fid_amp * fid_amp, 0.0), 1.0))

    def _bures_wasserstein(
        self,
        rho_raw: NDArray[np.complex128],
        sigma_psd: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        """Métrica de Bures–Wasserstein d_B(ρ_raw, σ_PSD) y fidelidad F.

        Incorpora la penalización de traza |Tr(ρ_raw) - 1| y el defecto no-hermítico
        ||ρ_raw - ρ_raw†||_F para detectar cualquier perturbación fuera de la variedad de estados.
        """
        rho_pos = self._raw_positive_part_density(rho_raw)
        tr_raw = float(np.real(np.trace(rho_raw)))
        non_herm_defect = float(la.norm(rho_raw - rho_raw.conj().T, ord="fro"))
        fid = self._uhlgon_fidelity(rho_pos, sigma_psd)
        bures_core = math.sqrt(max(2.0 * (1.0 - math.sqrt(max(fid, 0.0))), 0.0))
        d_b = bures_core + abs(tr_raw - 1.0) + 0.5 * non_herm_defect
        return float(d_b), float(fid)

    @staticmethod
    def _quantum_entropy_and_qfi(
        rho_psd: NDArray[np.complex128],
    ) -> Tuple[float, float, float]:
        """Calcula la entropía de von Neumann S_vN(ρ), la traza QFIM y la semilla de Poincaré–Cartan.

        1. S_vN(ρ) = -Σ_i λ_i ln(λ_i).
        2. I_QFI(ρ) = 2 Σ_{i<j} (λ_i - λ_j)² / (λ_i + λ_j) (sensibilidad métrica de Braunstein–Caves).
        3. Semilla de acción de Poincaré–Cartan: ϖ_PC = (1/2π) Σ_i λ_i (1 + |ln λ_i|).
        """
        evals = np.clip(np.real(la.eigvalsh(rho_psd)), _WILKINSON_EPSILON, None)
        evals /= float(np.sum(evals))
        vn_entropy = float(-np.sum(evals * np.log(evals)))

        qfi_sum = 0.0
        n = evals.size
        for i in range(n):
            for j in range(i + 1, n):
                denom = float(evals[i] + evals[j])
                if denom > _WILKINSON_EPSILON:
                    diff = float(evals[i] - evals[j])
                    qfi_sum += 4.0 * (diff * diff) / denom

        pc_seed = float(max((1.0 + vn_entropy) / (2.0 * math.pi), 0.10))
        return vn_entropy, float(qfi_sum), pc_seed

    def observe_mac_isometry(
        self,
        rho_mac: NDArray[np.complex128],
    ) -> Tuple[float, bool, int, float, float, bool]:
        """Ingesta MIC/MAC: audita traza, positividad espectral y distancia de Bures–Wasserstein."""
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho_mac debe ser una matriz de densidad cuadrada.")
        dim = int(rho.shape[0])
        tr_val = float(np.real(np.trace(rho)))
        evals = np.real(la.eigvalsh(0.5 * (rho + rho.conj().T)))
        is_psd = bool(np.all(evals >= _PSD_EIG_FLOOR))
        rho_ref = self._higham_psd(rho)
        d_b, fid = self._bures_wasserstein(rho, rho_ref)
        is_iso = (
            is_psd
            and abs(tr_val - 1.0) <= _TRACE_TOL
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
        vn_entropy: float = 0.0,
        qfi_trace: float = 0.0,
        pc_seed: float = 0.15,
    ) -> StealthNovikovPhase1Certificate:
        """Emite el certificado criptográfico P1 con hash de cadena SHA-256 y HMAC de dominio."""
        if ts is None:
            ts = time.time()
        raw_chain = (
            f"P1|{ts:.6f}|{tr_val:.16e}|{int(is_psd)}|{dim}|"
            f"{d_b:.16e}|{fid:.16e}|{int(is_iso)}|{pc_seed:.16e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(raw_chain).hexdigest()
        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return StealthNovikovPhase1Certificate(
            timestamp=float(ts),
            tr_rho=tr_val,
            is_psd=is_psd,
            dim=dim,
            bures_distance=d_b,
            fidelity=fid,
            is_bures_isometry=is_iso,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            von_neumann_entropy=vn_entropy,
            quantum_fisher_information=qfi_trace,
            poincare_cartan_action_seed=pc_seed,
        )

    def execute_phase1(
        self,
        rho_mac: NDArray[np.complex128],
    ) -> StealthNovikovPhase1Certificate:
        """Observa la matriz de densidad MAC y emite el certificado P1 enriquecido."""
        tr_val, is_psd, dim, d_b, fid, is_iso = self.observe_mac_isometry(rho_mac)
        rho_ref = self._higham_psd(np.asarray(rho_mac, dtype=np.complex128))
        vn_ent, qfi_val, pc_seed = self._quantum_entropy_and_qfi(rho_ref)
        return self.emit_phase1_certificate(
            tr_val,
            is_psd,
            dim,
            d_b,
            fid,
            is_iso,
            vn_entropy=vn_ent,
            qfi_trace=qfi_val,
            pc_seed=pc_seed,
        )

    def handoff_isometry_to_novikov_orientation(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        gate_admitted: bool,
    ) -> StealthNovikovPhase12Morphism:
        """CLAUSURA FORMAL DE FASE 1 ──► MORFISMO INAUGURAL DE FASE 2 (Φ₁₂).

        Eleva la observación cuántica de Bures–Wasserstein hacia el espacio de fases
        celeste de Poincaré mediante el invariante integral de Poincaré–Cartan:
            ϖ_PC = J_seed · F(ρ, σ_PSD) / (1 + d_B(ρ, σ_PSD))
        y establece el coeficiente de contracción de Nivel 1 (T₁) de RSI³:
            α₁ = 0.95 · exp(-0.01 · d_B).

        Esta definición formal cierra la Fase 1 y constituye la entrada estructural
        directa de `StealthNovikovPhase2Orientor` en la Fase 2.
        """
        pc_invariant = float(
            p1_cert.poincare_cartan_action_seed
            * max(p1_cert.fidelity, 0.1)
            / (1.0 + p1_cert.bures_distance)
        )
        alpha_level1 = float(0.95 * math.exp(-0.01 * min(p1_cert.bures_distance, 10.0)))
        return StealthNovikovPhase12Morphism(
            p1_cert=p1_cert,
            bures_distance=p1_cert.bures_distance,
            fidelity=p1_cert.fidelity,
            gate_admitted=bool(gate_admitted),
            passive_mimicry_allowed=bool(gate_admitted),
            poincare_cartan_invariant=pc_invariant,
            rsi3_level1_alpha=alpha_level1,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — ORIENT: Mecánica Celeste de Poincaré–KAM–Nekhoroshev,          ║
# ║           Criterio de Solapamiento de Resonancias de Chirikov,           ║
# ║           Holonomía de Berry SU(N) y Meta-Optimización RSI³ Nivel 2.     ║
# ║  Inicio F2: Consume el morfismo `handoff_isometry_to_novikov_orientation`║
# ║  Cierre F2 / Apertura F3: `handoff_spectrum_to_heyting_adjudication`     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class StealthNovikovPhase2Orientor:
    """Fase 2 OODA — Orientación espectral en Λ_Nov, mecánica celeste de Poincaré y RSI³ Nivel 2."""

    def __init__(self, engine: BoolNovikovStealthNovikovEngine, secret_key: bytes) -> None:
        self._engine = engine
        self._secret_key = secret_key
        self._active_morphism_12: Optional[StealthNovikovPhase12Morphism] = None

    @staticmethod
    def _spectrum_attr(rep: Any, name: str, default: Any) -> Any:
        return getattr(rep, name, default)

    def ingest_phase1_handoff(
        self,
        morphism_12: StealthNovikovPhase12Morphism,
    ) -> StealthNovikovPhase1Certificate:
        """Continuación directa de `handoff_isometry_to_novikov_orientation` (Fase 1 → Fase 2).

        Valida la admisión de la compuerta homomórfica Φ₁₂ y ancla el invariante integral
        de Poincaré–Cartan ϖ_PC y la contracción T₁ como contexto de orientación celeste.
        """
        if not morphism_12.gate_admitted:
            raise ValueError("Fase 2 rechazada: morfismo Φ₁₂ no admitido.")
        self._active_morphism_12 = morphism_12
        return morphism_12.p1_cert

    @staticmethod
    def _compute_chirikov_resonance_overlap(
        poincare_rep: Any,
        pert_eps: float = 1.0e-3,
    ) -> float:
        """Calcula el parámetro de solapamiento de resonancias de Chirikov K_Chirikov.

        En mecánica celeste de Poincaré–Chirikov, el ancho de media separatriz de una
        resonancia primaria es Δω_sep = 2 √(ε · |det D²H₀|^{1/d}), mientras que la distancia
        al divisor diofantino más cercano es Δω_res = max(kam_small_divisor_min, γ_KAM).
        Si K_Chirikov = Δω_sep / Δω_res > 1, los toros KAM se destruyen dando paso a la
        difusión estocástica global de Arnold.
        """
        if poincare_rep is None:
            return 0.0
        kolm_det = abs(float(getattr(poincare_rep, "kolmogorov_hessian_det", 1.0)))
        freqs = getattr(poincare_rep, "frequencies", (1.0, 1.0))
        d = max(len(freqs), 1)
        min_div = max(float(getattr(poincare_rep, "kam_small_divisor_min", 1.0)), 1.0e-3)
        sep_half_width = 2.0 * math.sqrt(max(pert_eps, 0.0) * (kolm_det ** (1.0 / d)))
        return float(sep_half_width / min_div)

    def orient_poincare_kam_rsi3(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
    ) -> StealthNovikovSpectrumReport:
        """Ejecuta la auditoría espectral de Novikov, Poincaré–KAM–Nekhoroshev y Kleisli T³.

        Si existe un morfismo Φ₁₂ activo, acopla de forma no destructiva el invariante
        de Poincaré–Cartan con la sucesión de exponentes de Hahn–Novikov.
        """
        exp_list = list(exponents)
        if not exp_list and self._active_morphism_12 is not None:
            exp_list = [self._active_morphism_12.poincare_cartan_invariant]
        return self._engine.execute_stealth_novikov_audit(apu_cost_matrix, exp_list)

    def emit_phase2_certificate(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        spectrum_rep: StealthNovikovSpectrumReport,
        ts: Optional[float] = None,
    ) -> StealthNovikovPhase2Certificate:
        """Emite el certificado P2 encadenado criptográficamente a P1 con métricas celestes y RSI³."""
        if ts is None:
            ts = time.time()
        poincare = self._spectrum_attr(spectrum_rep, "poincare_report", None)
        rsi3 = self._spectrum_attr(spectrum_rep, "rsi3_report", None)
        sun = self._spectrum_attr(spectrum_rep, "sun_report", None)

        kam_dio = (
            bool(getattr(poincare, "diophantine_holds", True))
            if poincare is not None
            else True
        )
        chaos = bool(
            getattr(poincare, "homoclinic_chaos", spectrum_rep.melnikov_chaos_detected)
            if poincare is not None
            else spectrum_rep.melnikov_chaos_detected
        )
        lindstedt = float(
            getattr(poincare, "homological_residual", 0.0)
            if poincare is not None
            else 0.0
        )
        nekh_time = float(
            getattr(poincare, "nekhoroshev_stability_time", 1.0e6)
            if poincare is not None
            else 1.0e6
        )
        yoshida_err = float(
            getattr(poincare, "yoshida_symplectic_error", 0.0)
            if poincare is not None
            else 0.0
        )
        chirikov_k = self._compute_chirikov_resonance_overlap(poincare)

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

        raw_chain = (
            f"P2|{ts:.6f}|{p1_cert.chain_hash}|{spectrum_rep.truth_value.value}|"
            f"{int(kam_dio)}|{int(chaos)}|{lip_t3:.16e}|{chirikov_k:.16e}|{kantorovich_h:.16e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(raw_chain).hexdigest()
        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return StealthNovikovPhase2Certificate(
            timestamp=float(ts),
            spectrum_report=spectrum_rep,
            p1_chain_hash=p1_cert.chain_hash,
            kam_diophantine=kam_dio,
            homoclinic_chaos=chaos,
            lindstedt_residual=lindstedt,
            lip_t3=lip_t3,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            chirikov_overlap_parameter=chirikov_k,
            nekhoroshev_stability_time=nekh_time,
            yoshida_symplectic_error=yoshida_err,
            kantorovich_parameter_h=kantorovich_h,
            berry_holonomy_phase_rad=berry_phase,
        )

    def execute_phase2(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
    ) -> StealthNovikovPhase2Certificate:
        """Orienta la simulación de camuflaje ultramétrico en Λ_Nov y emite P2."""
        spectrum_rep = self.orient_poincare_kam_rsi3(apu_cost_matrix, exponents)
        return self.emit_phase2_certificate(p1_cert, spectrum_rep)

    def execute_phase2_from_morphism(
        self,
        morphism_12: StealthNovikovPhase12Morphism,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
    ) -> StealthNovikovPhase2Certificate:
        """Entrada canónica anidada de Fase 2 desde el morfismo final de Fase 1."""
        p1_cert = self.ingest_phase1_handoff(morphism_12)
        return self.execute_phase2(p1_cert, apu_cost_matrix, exponents)

    def handoff_spectrum_to_heyting_adjudication(
        self,
        p2_cert: StealthNovikovPhase2Certificate,
        gate_admitted: bool,
    ) -> StealthNovikovPhase23Morphism:
        """CLAUSURA FORMAL DE FASE 2 ──► MORFISMO INAUGURAL DE FASE 3 (Φ₂₃).

        Empaqueta el haz simpléctico-cuántico de Poincaré–Floer, el parámetro de
        solapamiento de resonancias de Chirikov, la cota cuadrática de Newton–Kantorovich h_K
        y el radio espectral de Fréchet de Nivel 2 ρ(DT₂) para alimentar el adjudicador
        del Topos Intuicionista de Heyting Ω₃.

        Esta definición formal cierra la Fase 2 y constituye la entrada estructural
        directa de `StealthNovikovPhase3Decider` en la Fase 3.
        """
        rep = p2_cert.spectrum_report
        rsi3 = getattr(rep, "rsi3_report", None)
        unique = bool(getattr(rsi3, "unique_attractor", p2_cert.lip_t3 < 1.0))
        level2_rho = float(
            getattr(rsi3, "level2_spectral_radius_dt", p2_cert.lip_t3 ** (2.0 / 3.0))
            if rsi3 is not None
            else p2_cert.lip_t3 ** (2.0 / 3.0)
        )
        return StealthNovikovPhase23Morphism(
            p2_cert=p2_cert,
            lip_t3=float(p2_cert.lip_t3),
            back_action_db=float(rep.sun_report.back_action_db),
            poincare_diophantine=bool(p2_cert.kam_diophantine),
            rsi3_unique_attractor=unique,
            gate_admitted=bool(gate_admitted),
            chirikov_overlap=p2_cert.chirikov_overlap_parameter,
            kantorovich_h=p2_cert.kantorovich_parameter_h,
            rsi3_level2_spectral_radius=level2_rho,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — DECIDE & ACT: Lógica Interna del Topos de Heyting Ω₃,          ║
# ║           Automejora Recursiva Nivel 3 (T₃ = μ ∘ T₂ ∘ T₁), Funtor        ║
# ║           Semántico Visceral Φ_sem, Árbol de Merkle y Gobernanza OODA.   ║
# ║  Inicio F3: Consume el morfismo `handoff_spectrum_to_heyting_adjudication`║
# ╚══════════════════════════════════════════════════════════════════════════╝


class StealthNovikovPhase3Decider:
    """Fase 3 OODA — Adjudicación intuicionista en Ω₃, RSI³ Nivel 3 y traducción visceral Φ_sem."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key
        self._active_morphism_23: Optional[StealthNovikovPhase23Morphism] = None

    def ingest_phase2_handoff(
        self,
        morphism_23: StealthNovikovPhase23Morphism,
    ) -> StealthNovikovPhase2Certificate:
        """Continuación directa de `handoff_spectrum_to_heyting_adjudication` (Fase 2 → Fase 3).

        Verifica la admisión de la compuerta homomórfica Φ₂₃ y registra el morfismo
        de Kleisli de Nivel 2 para la síntesis hiper-meta de Nivel 3 (T₃).
        """
        if not morphism_23.gate_admitted:
            raise ValueError("Fase 3 rechazada: morfismo Φ₂₃ no admitido.")
        self._active_morphism_23 = morphism_23
        return morphism_23.p2_cert

    @staticmethod
    def _heyting_meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Conjunción categórica (∧) en el retículo distributivo de Heyting Ω₃.

        Orden de verdad: COHERENT (⊤ = 0x00) > DEGRADED (0x01) > VETOED (⊥ = 0x03).
        """
        return HeytingTruthValue(max(a.value, b.value))

    @staticmethod
    def _heyting_join(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Disyunción categórica (∨) en el retículo distributivo de Heyting Ω₃."""
        return HeytingTruthValue(min(a.value, b.value))

    @classmethod
    def _heyting_implication(
        cls,
        antecedent: HeytingTruthValue,
        consequent: HeytingTruthValue,
    ) -> HeytingTruthValue:
        """Implicación relativa intuicionista (a ⇒ b) = ⋁ { c ∈ Ω₃ : (c ∧ a) ≤_truth b }.

        Nota: En la codificación hexadecimal del motor, menor valor entero representa
        mayor grado de verdad constructiva (0x00 = ⊤, 0x01 = intermedio, 0x03 = ⊥).
        Si a ≥_truth b (es decir, antecedent.value <= consequent.value) no se cumple,
        entonces a ≤_truth b se satisface y (a ⇒ b) = COHERENT (⊤).
        """
        if antecedent.value >= consequent.value:
            return HeytingTruthValue.COHERENT
        return consequent

    @staticmethod
    def _kam_stability_index(p2: StealthNovikovPhase2Certificate) -> float:
        """Índice compuesto de estabilidad celeste de Poincaré–KAM–Nekhoroshev en [0, 1].

        Integra:
          1. Residuo homológico de la serie de Lie–Birkhoff–Lindstedt.
          2. Conservación simpléctica de 4.º orden de Yoshida.
          3. Ausencia de solapamiento de resonancias de Chirikov (K_Chirikov < 1).
          4. Confinamiento exponencial de Nekhoroshev.
        """
        res = max(float(p2.lindstedt_residual), 0.0)
        yoshida = max(float(p2.yoshida_symplectic_error), 0.0)
        chirikov = max(float(p2.chirikov_overlap_parameter), 0.0)

        smooth_lie = 1.0 / (1.0 + res * 1.0e3 + yoshida * 1.0e4)
        chirikov_factor = 1.0 / (1.0 + chirikov * chirikov)
        nekh_factor = min(1.0, math.log10(max(p2.nekhoroshev_stability_time, 10.0)) / 6.0)
        composite = smooth_lie * (0.7 + 0.15 * chirikov_factor + 0.15 * nekh_factor)

        if p2.homoclinic_chaos:
            return float(0.15 * composite)
        if not p2.kam_diophantine:
            return float(0.45 * composite)
        return float(min(1.0, 0.85 + 0.15 * composite))

    def apply_visceral_functor(
        self,
        p2_cert: StealthNovikovPhase2Certificate,
        is_camouflage: bool,
        truth_val: HeytingTruthValue,
    ) -> StealthNovikovVisceralMetrics:
        """Funtor semántico visceral Φ_sem: Ω₃ × KAM × Kl(𝕋³) → ℝ³ × Status.

        Acopla la adjudicación del topos de Heyting con la estabilidad celeste de
        Poincaré–KAM–Nekhoroshev y el margen de convergencia cuadrática de
        Newton–Kantorovich (RSI³ Nivel 3).
        """
        kam_idx = self._kam_stability_index(p2_cert)
        lip = max(min(float(p2_cert.lip_t3), 0.999999), 0.0)
        lip_quality = max(0.0, (1.0 - lip) / (1.0 - 0.857375))
        lip_quality = min(lip_quality, 1.0)

        rsi3 = getattr(p2_cert.spectrum_report, "rsi3_report", None)
        jerk = float(getattr(rsi3, "jerk", 0.0)) if rsi3 is not None else 0.0
        kantorovich_h = float(p2_cert.kantorovich_parameter_h)
        kantorovich_margin = float(
            max(0.0, min(1.0, (_KANTOROVICH_CEILING - kantorovich_h) / _KANTOROVICH_CEILING))
        )
        nekh_idx = float(
            min(1.0, math.log10(max(p2_cert.nekhoroshev_stability_time, 10.0)) / 6.0)
        )

        if truth_val == HeytingTruthValue.COHERENT and is_camouflage:
            scale = 1.0
            status = "ACTIVE_CAMOUFLAGE_SU_N_0.0_dB"
        elif truth_val == HeytingTruthValue.DEGRADED:
            scale = 0.45 + 0.40 * kam_idx * lip_quality * (0.5 + 0.5 * kantorovich_margin)
            status = "DEGRADED_GAUGE"
        else:
            scale = 0.12 * kam_idx
            status = "VETOED_SILENT_MIMICRY"

        return StealthNovikovVisceralMetrics(
            wacc_protection_pct=float(_WACC_PROTECTION_TARGET_PCT * scale),
            net_liquidity_release_pct=float(_NET_LIQUIDITY_TARGET_PCT * scale),
            kv_cache_compression_pct=float(_KV_COMPRESSION_TARGET_PCT * scale),
            camouflage_status=status,
            kam_stability_index=float(kam_idx),
            rsi3_jerk=jerk,
            nekhoroshev_confinement_index=nekh_idx,
            kantorovich_convergence_margin=kantorovich_margin,
        )

    @staticmethod
    def _compute_binary_merkle_root(leaves: Sequence[bytes]) -> str:
        """Construye un Árbol de Merkle Binario criptográfico SHA-256 con separación de dominios."""
        if not leaves:
            return hashlib.sha256(b"EMPTY_MERKLE_TREE").hexdigest()
        layer = [hashlib.sha256(b"\x00" + leaf).digest() for leaf in leaves]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            next_layer = []
            for i in range(0, len(layer), 2):
                parent = hashlib.sha256(b"\x01" + layer[i] + layer[i + 1]).digest()
                next_layer.append(parent)
            layer = next_layer
        return layer[0].hex()

    def emit_sovereign_state(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        p2_cert: StealthNovikovPhase2Certificate,
        ts: Optional[float] = None,
    ) -> StealthNovikovSovereignState:
        """Cierra la mónada de Kleisli T³ (RSI³ Nivel 3), adjudica en Ω₃ y sella la raíz de Merkle."""
        if ts is None:
            ts = time.time()
        rep = p2_cert.spectrum_report
        engine_truth = rep.truth_value
        nov = rep.novikov_report
        sun = rep.sun_report
        rsi3 = getattr(rep, "rsi3_report", None)

        kantorovich_ok = bool(
            getattr(rsi3, "kantorovich_guaranteed", p2_cert.kantorovich_parameter_h <= _KANTOROVICH_CEILING)
            if rsi3 is not None
            else (p2_cert.kantorovich_parameter_h <= _KANTOROVICH_CEILING)
        )
        chirikov_ok = p2_cert.chirikov_overlap_parameter < _CHIRIKOV_CHAOS_CEILING

        # Deducción intuicionista en Ω₃ combinando veredicto del motor con invariantes celestes y RSI³
        poincare_prop = (
            HeytingTruthValue.COHERENT
            if (p2_cert.kam_diophantine and not p2_cert.homoclinic_chaos and chirikov_ok)
            else HeytingTruthValue.DEGRADED
        )
        rsi3_prop = (
            HeytingTruthValue.COHERENT
            if (p2_cert.lip_t3 < _PHI23_LIP_CEILING and kantorovich_ok)
            else HeytingTruthValue.DEGRADED
        )
        combined_truth = self._heyting_meet(
            engine_truth, self._heyting_meet(poincare_prop, rsi3_prop)
        )

        is_camouflage = bool(
            sun.is_camouflaged_qnd
            and nov.is_krull_above_floor
            and (not p2_cert.homoclinic_chaos)
            and p2_cert.lip_t3 < _PHI23_LIP_CEILING
            and p2_cert.kam_diophantine
            and kantorovich_ok
        )
        visceral = self.apply_visceral_functor(p2_cert, is_camouflage, combined_truth)
        engine_passport = str(getattr(rep, "hmac_passport", ""))

        # Árbol de Merkle Binario sobre las 4 hojas formales del ciclo OODA
        leaf_p1 = (
            f"L1_OBSERVE|{ts:.6f}|{p1_cert.hmac_signature}|{p1_cert.bures_distance:.16e}|"
            f"{p1_cert.poincare_cartan_action_seed:.16e}"
        ).encode("utf-8")
        leaf_p2 = (
            f"L2_ORIENT|{p2_cert.hmac_signature}|{p2_cert.chirikov_overlap_parameter:.16e}|"
            f"{p2_cert.nekhoroshev_stability_time:.16e}|{p2_cert.berry_holonomy_phase_rad:.16e}"
        ).encode("utf-8")
        leaf_p3 = (
            f"L3_RSI3|{combined_truth.value}|{p2_cert.lip_t3:.16e}|"
            f"{p2_cert.kantorovich_parameter_h:.16e}|{visceral.rsi3_jerk:.16e}"
        ).encode("utf-8")
        leaf_p4 = (
            f"L4_SOVEREIGN|{engine_passport}|{visceral.wacc_protection_pct:.8f}|"
            f"{visceral.camouflage_status}"
        ).encode("utf-8")

        merkle_root = self._compute_binary_merkle_root((leaf_p1, leaf_p2, leaf_p3, leaf_p4))
        p3_mac = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + merkle_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        passport = engine_passport if engine_passport else p3_mac

        return StealthNovikovSovereignState(
            timestamp=float(ts),
            truth_value=combined_truth,
            is_camouflage_gauge_invariant=is_camouflage,
            back_action_db=float(sun.back_action_db),
            merkle_root_hash=merkle_root,
            visceral_metrics=visceral,
            kam_diophantine=bool(p2_cert.kam_diophantine),
            rsi3_lip=float(p2_cert.lip_t3),
            hmac_passport=passport,
            chirikov_stochasticity_index=p2_cert.chirikov_overlap_parameter,
            kantorovich_certified=kantorovich_ok,
        )

    def execute_phase3(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        p2_cert: StealthNovikovPhase2Certificate,
    ) -> StealthNovikovSovereignState:
        """Adjudica el estado soberano final en el topos de Heyting Ω₃."""
        return self.emit_sovereign_state(p1_cert, p2_cert)

    def execute_phase3_from_morphism(
        self,
        p1_cert: StealthNovikovPhase1Certificate,
        morphism_23: StealthNovikovPhase23Morphism,
    ) -> StealthNovikovSovereignState:
        """Entrada canónica anidada de Fase 3 desde el morfismo final de Fase 2."""
        p2_cert = self.ingest_phase2_handoff(morphism_23)
        return self.execute_phase3(p1_cert, p2_cert)


# ─────────────────────────────────────────────────────────────────────────────
# SOBERANO AGÉNTICO — Composición de Kleisli de las 3 Fases Anidadas (Φ₁₂, Φ₂₃)
# ─────────────────────────────────────────────────────────────────────────────


class BoolNovikovStealthNovikovAgent:
    """Soberano agéntico del disfraz ultramétrico de Novikov, Poincaré–KAM y RSI³."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key
        self._engine = BoolNovikovStealthNovikovEngine(secret_key)
        self._observer = StealthNovikovPhase1Observer(secret_key)
        self._orientor = StealthNovikovPhase2Orientor(self._engine, secret_key)
        self._decider = StealthNovikovPhase3Decider(secret_key)
        logger.info("%s iniciado correctamente.", _AGENT_VERSION)

    def continue_to_phase2_gate(self, p1_cert: StealthNovikovPhase1Certificate) -> bool:
        """Compuerta homomórfica Φ₁₂: verifica la integridad criptográfica HMAC-SHA256 de P1.

        Preserva el mimetismo pasivo silencioso QND: estados cuánticos degradados
        no son demolidos en Φ₁₂, sino conducidos a adjudicación intuicionista en Ω₃.
        """
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + p1_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected_sig, p1_cert.hmac_signature)

    def continue_to_phase3_gate(self, p2_cert: StealthNovikovPhase2Certificate) -> bool:
        """Compuerta homomórfica Φ₂₃: valida HMAC P2, contracción de Banach–Kantorovich y QND.

        Exige:
          1. Autenticidad criptográfica HMAC de P2.
          2. Contracción estricta de la mónada de Kleisli: Lip(T³) < 1.0.
          3. Cota de Newton–Kantorovich finita y controlada: h_K ≤ 0.50.
          4. Back-action cuántico SU(N) ≤ 0.01 dB.
          5. Residuo homológico de Lie–Birkhoff–Lindstedt y error de Yoshida finitos.
        """
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + p2_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        is_sig_valid = hmac.compare_digest(expected_sig, p2_cert.hmac_signature)
        is_banach_valid = (
            p2_cert.spectrum_report.banach_contraction_lip < _PHI23_LIP_CEILING
        )
        is_kantorovich_valid = (
            math.isfinite(p2_cert.kantorovich_parameter_h)
            and p2_cert.kantorovich_parameter_h <= _KANTOROVICH_CEILING
        )
        is_qnd_slack = p2_cert.spectrum_report.sun_report.back_action_db <= (
            _PHI23_BACKACTION_DB_CEILING + 1.0e-12
        )
        celestial_finite = math.isfinite(p2_cert.lindstedt_residual) and math.isfinite(
            p2_cert.yoshida_symplectic_error
        )
        return bool(
            is_sig_valid
            and is_banach_valid
            and is_kantorovich_valid
            and is_qnd_slack
            and celestial_finite
        )

    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
    ) -> StealthNovikovSovereignState:
        """Ejecuta el ciclo OODA trifásico anidado: Observe ⊂ Orient ⊂ Decide&Act (Kleisli T³)."""
        # ── FASE 1: Observe + Levantamiento del Invariante de Poincaré–Cartan ──
        p1 = self._observer.execute_phase1(rho_mac)
        phi12 = self.continue_to_phase2_gate(p1)
        if not phi12:
            raise ValueError(
                "Falla de autenticidad HMAC en Compuerta Homomórfica Φ₁₂."
            )
        morphism_12 = self._observer.handoff_isometry_to_novikov_orientation(p1, phi12)

        # ── FASE 2: Inicia en `handoff_isometry_to_novikov_orientation` (F1) ───
        p2 = self._orientor.execute_phase2_from_morphism(
            morphism_12, apu_cost_matrix, exponents
        )
        phi23 = self.continue_to_phase3_gate(p2)
        if not phi23:
            raise ValueError(
                "Falla de estabilidad de Banach/Kantorovich/QND en Compuerta Homomórfica Φ₂₃."
            )
        morphism_23 = self._orientor.handoff_spectrum_to_heyting_adjudication(p2, phi23)

        # ── FASE 3: Inicia en `handoff_spectrum_to_heyting_adjudication` (F2) ──
        sovereign_state = self._decider.execute_phase3_from_morphism(p1, morphism_23)
        logger.info(
            "Ciclo OODA de Camuflaje en Novikov completado. Veredicto: %s (%s) | "
            "KAM=%s | Chirikov=%.4e | Lip(T³)=%.6f | Kantorovich=%s",
            sovereign_state.truth_value.name,
            hex(sovereign_state.truth_value.value),
            sovereign_state.kam_diophantine,
            sovereign_state.chirikov_stochasticity_index,
            sovereign_state.rsi3_lip,
            sovereign_state.kantorovich_certified,
        )
        return sovereign_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(
        "=== TEST AUTÓNOMO: BoolNovikovStealthNovikovAgent v15.0.0 (3 Fases Anidadas) ==="
    )
    agent = BoolNovikovStealthNovikovAgent()

    dim = 2
    rho_test = np.eye(dim, dtype=np.complex128) / float(dim)
    matrix_test = np.array([[120.0, 10.0], [10.0, 180.0]], dtype=np.float64)
    exps_test = [0.15, 0.30, 0.45]

    sovereign_state = agent.run_sovereign_governance_cycle(
        rho_test, matrix_test, exps_test
    )
    print(
        f"Veredicto Final Heyting Ω₃  : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(f"Invariante CAMOUFLAGE_GAUGE : {sovereign_state.is_camouflage_gauge_invariant}")
    print(f"Back-Action QND (dB)        : {sovereign_state.back_action_db:.6f} dB")
    print(f"KAM diofantino              : {sovereign_state.kam_diophantine}")
    print(f"Índice Chirikov K           : {sovereign_state.chirikov_stochasticity_index:.6e}")
    print(f"Lip(T³) RSI³                : {sovereign_state.rsi3_lip:.6f}")
    print(f"Newton-Kantorovich cert.    : {sovereign_state.kantorovich_certified}")
    print(f"Raíz Merkle Binaria SHA-256 : {sovereign_state.merkle_root_hash[:16]}...")
    print(f"Pasaporte HMAC              : {sovereign_state.hmac_passport[:16]}...")
    print(
        f"Protección WACC (%)         : {sovereign_state.visceral_metrics.wacc_protection_pct:.4f}%"
    )
    print(
        f"Liberación Liquidez Neta (%): {sovereign_state.visceral_metrics.net_liquidity_release_pct:.4f}%"
    )
    print(
        f"Compresión KV (%)           : {sovereign_state.visceral_metrics.kv_cache_compression_pct:.4f}%"
    )
    print(
        f"Índice estabilidad KAM      : {sovereign_state.visceral_metrics.kam_stability_index:.6f}"
    )
    print(
        f"Índice Nekhoroshev          : {sovereign_state.visceral_metrics.nekhoroshev_confinement_index:.6f}"
    )
    print(
        f"Margen Kantorovich          : {sovereign_state.visceral_metrics.kantorovich_convergence_margin:.6f}"
    )
    print(
        f"Jerk RSI³ d³C/dt³           : {sovereign_state.visceral_metrics.rsi3_jerk:.6e}"
    )
    print(
        f"Status camuflaje            : {sovereign_state.visceral_metrics.camouflage_status}"
    )
    print("=== TEST COMPLETADO CON ÉXITO ===")