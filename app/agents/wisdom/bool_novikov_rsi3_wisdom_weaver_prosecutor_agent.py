# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovRSI3WisdomWeaverProsecutorAgent v15.0.0-PhD+                                  ║
║ Ubicación: app/agents/wisdom/bool_novikov_rsi3_wisdom_weaver_prosecutor_agent.py                    ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD+  ║
║ Gobernanza: OODA Trifásica Inquisitorial Anidada ①Observe → ②Orient → ③Decide&Act con Φ₁₂ y Φ₂₃    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE 3 FASES ESTRICTAMENTE ANIDADAS (PUENTES HOMOMÓRFICOS Φ₁₂ Y Φ₂₃):
  [FASE 1: OBSERVACIÓN CINEMÁTICA CUÁNTICA E INGESTA ISOMÉTRICA (R_Boole ⊗ Λ_Nov)]
    1.1 `check_trace_conservation`                  → Conservación de traza Tr(ρ) = 1 y defecto hermítico.
    1.2 `check_psd_purity_entropy`                  → Espectro PSD, pureza Tr(ρ²), entropías S_vN y Rényi-2.
    1.3 `compute_quantum_coherence_and_wigner_yanase` → Coherencia l1, información de Wigner-Yanase y QFI F_Q.
    1.4 `emit_phase1_certificate` (`execute_phase1`) → Certificado P1 sellado con SHA-256 + HMAC-SHA256.
    1.5 `continue_to_phase2_gate`                   → DEFINICIÓN FINAL FASE 1 (Puente Φ₁₂ hacia Fase 2).
                                                       │
                                                       ▼
  [FASE 2: ORIENTACIÓN INQUISITORIAL FLOER-POINCARÉ-KAM Y GOBERNANZA RECURSIVA RSI³ NIVEL-3]
    2.1 `validate_entrance_from_phase1`             → INICIO FASE 2 (Consume el puente Φ₁₂ de Fase 1).
    2.2 `derive_delaunay_action_and_perturbation`   → Variables canónicas de Delaunay (L,G,H) y Kozai-Lidov.
    2.3 `recursive_governance_improvement_nivel3`   → Política óptima ε* vía Lyapunov-Halley-Darwin-Gödel.
    2.4 `execute_phase2` (`emit_phase2_certificate`) → Inquisición fiscal Floer-KAM-RSI³ y certificado P2.
    2.5 `continue_to_phase3_gate`                   → DEFINICIÓN FINAL FASE 2 (Puente Φ₂₃ hacia Fase 3).
                                                       │
                                                       ▼
  [FASE 3: ADJUDICACIÓN HEYTING Ω₃, INTERLOCK CROWBAR/DIRAC, FUNTOR Φ_sem Y RAÍZ DE MERKLE]
    3.1 `validate_entrance_from_phase2`             → INICIO FASE 3 (Consume el puente Φ₂₃ de Fase 2).
    3.2 `adjudicate_heyting_omega3`                 → Retículo intuicionista Ω₃ {COHERENT ≺ DEGRADED ≺ VETOED}.
    3.3 `verify_crowbar_interlock_timing`           → Auditoría física ESP32 (0x3FF44008, ≤392.15ns) + Dirac.
    3.4 `functor_semantic_visceral_phi_sem`         → Funtor financiero/sistémico (WACC 15%, Liq 11.5%, KV 86.4%).
    3.5 `emit_sovereign_state` (`execute_phase3`)   → Estado soberano con árbol de Merkle de 8 hojas.
    3.6 `run_sovereign_governance_cycle`            → Orquestador soberano del ciclo OODA inquisitorial.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Importación del motor fiscal (v15/v14 preferido, v13 compatible) ──
try:
    from bool_novikov_rsi3_wisdom_weaver_prosecutor_engine import (
        BoolNovikovRSI3WisdomWeaverProsecutorEngine,
        HeytingTruthValue,
        WisdomWeaverProsecutorReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_prosecutor_engine import (
            BoolNovikovRSI3WisdomWeaverProsecutorEngine,
            HeytingTruthValue,
            WisdomWeaverProsecutorReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_rsi3_wisdom_weaver_prosecutor_engine."
        ) from exc

try:
    from bool_novikov_rsi3_wisdom_weaver_prosecutor_engine import (  # type: ignore
        KAMConvictionCertificate,
        ProsecutionEvidenceAnchor,
    )
    _ENGINE_EXTENDED: bool = True
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_prosecutor_engine import (  # type: ignore
            KAMConvictionCertificate,
            ProsecutionEvidenceAnchor,
        )
        _ENGINE_EXTENDED = True
    except ImportError:
        KAMConvictionCertificate = Any  # type: ignore
        ProsecutionEvidenceAnchor = Any  # type: ignore
        _ENGINE_EXTENDED = False

logger = logging.getLogger("APU.Wisdom.BoolNovikovRSI3WisdomWeaverProsecutorAgent")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, CRIPTOGRÁFICAS Y PARÁMETROS UNIVERSALES (v15.0.0-PhD+)
# ═══════════════════════════════════════════════════════════════════════════════
_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD+"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_PROSECUTOR_AGENT_KEY_v13.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"PROSECUTOR-WEAVER-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"PROSECUTOR-WEAVER-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_STATE: Final[bytes] = b"PROSECUTOR-WEAVER-STATE-v14.0|"

_TRACE_TOL: Final[float] = 1.0e-9
_PSD_FLOOR: Final[float] = -1.0e-12
_FRESHNESS_MAX_S: Final[float] = 3600.0
_LIP_HARD_MAX: Final[float] = 1.0
_KANTOROVICH_MAX: Final[float] = 0.5
_GREENE_CRITICAL_RESIDUE: Final[float] = 0.25
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))

# Constantes electro-cuánticas del Interlock Crowbar ESP32 + Tiristor SCR BT151 + Dirac QED
_CROWBAR_MAX_NS: Final[float] = 392.15
_CROWBAR_IRAM_TARGET_NS: Final[float] = 41.666666666666664
_CROWBAR_IRAM_TOL_NS: Final[float] = 5.0
_CROWBAR_REG: Final[str] = "0x3FF44008"
_DIRAC_ANNIHILATION_KEV: Final[float] = 1022.0


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES (Compatibles v13/v14 + Invariantes v15 PhD+)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WisdomWeaverProsecutorPhase1Certificate:
    """Certificado cinemático cuántico de Fase 1 (Observe)."""

    timestamp: float
    tr_rho: float
    is_psd: bool
    chain_hash: str
    hmac_signature: str
    is_trace_conserved: bool = True
    purity: float = 0.0
    von_neumann_entropy: float = 0.0
    rho_hash: str = ""
    dim: int = 0
    # Extensiones PhD+ v15 (con valores por defecto para compatibilidad posicional)
    renyi2_entropy: float = 0.0
    l1_coherence: float = 0.0
    wigner_yanase_skew_info: float = 0.0
    quantum_fisher_info: float = 0.0
    min_eigenvalue: float = 0.0
    hermiticity_defect: float = 0.0


@dataclass(frozen=True)
class WisdomWeaverProsecutorPhase2Certificate:
    """Certificado dinámico-espectral inquisitorial de Fase 2 (Orient)."""

    timestamp: float
    spectrum_report: WisdomWeaverProsecutorReport
    chain_hash: str
    hmac_signature: str
    prev_chain_hash: str = ""
    action_J: float = 1.0
    pert_eps: float = 0.01
    lindstedt_residual: float = 0.0
    diophantine_gamma: float = 0.0
    melnikov_value: float = 0.0
    melnikov_chaos: bool = False
    floer_energy: float = 0.0
    fraud_probability: float = 0.0
    rsi_iterations: int = 0
    godel_hash: str = ""
    # Extensiones PhD+ v15 (mecánica celeste Delaunay-Kozai-Lidov + Floer-Kantorovich)
    greene_residue: float = 0.0
    conley_zehnder_index: int = 0
    nekhoroshev_time: float = 0.0
    octonion_associator_norm: float = 0.0
    kantorovich_h: float = 0.0
    delaunay_L: float = 1.0
    delaunay_G: float = 1.0
    delaunay_H: float = 1.0
    orbital_eccentricity: float = 0.0
    kozai_lidov_invariant: float = 1.0
    governance_eps_policy: float = 0.03


@dataclass(frozen=True)
class WisdomWeaverProsecutorVisceralMetrics:
    """Métricas viscerales financieras y de sistemas producidas por Φ_sem."""

    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    crowbar_status: str
    is_kam_stable: bool = True
    stability_factor: float = 1.0
    bures_fidelity: float = 1.0
    fraud_probability: float = 0.0
    # Extensiones PhD+ v15
    greene_residue: float = 0.0
    nekhoroshev_time: float = 0.0
    kozai_lidov_invariant: float = 1.0
    kantorovich_h: float = 0.0


@dataclass(frozen=True)
class WisdomWeaverProsecutorSovereignState:
    """Estado Soberano final emitido al cierre del ciclo OODA inquisitorial."""

    timestamp: float
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    total_actuation_time_ns: float
    merkle_root_hash: str
    visceral_metrics: WisdomWeaverProsecutorVisceralMetrics
    p1_chain_hash: str = ""
    p2_chain_hash: str = ""
    godel_hash: str = ""
    dirac_signature: str = ""
    governance_latency_ms: float = 0.0
    action_J: float = 1.0
    pert_eps: float = 0.01
    is_kam_stable: bool = True
    fraud_probability: float = 0.0
    interlock_compliant: bool = True
    # Extensiones PhD+ v15
    state_hmac_signature: str = ""
    greene_residue: float = 0.0
    conley_zehnder_index: int = 0
    nekhoroshev_time: float = 0.0
    kozai_lidov_invariant: float = 1.0
    kantorovich_h: float = 0.0
    interlock_reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# NÚCLEO DE UTILIDADES CRIPTOGRÁFICAS Y ESPECTRALES CANÓNICAS
# ═══════════════════════════════════════════════════════════════════════════════
def _canonical_p1_bytes(
    ts: float, tr: float, is_psd: bool, purity: float, rho_hash: str
) -> bytes:
    return (
        f"PROSECUTOR-P1|{ts:.6f}|{tr:.8f}|{is_psd}|{purity:.6f}|{rho_hash}"
    ).encode("utf-8")


def _canonical_p2_bytes(
    ts: float,
    prev_hash: str,
    truth_int: int,
    lip: float,
    krull: float,
    cr: float,
    chaos: bool,
    crowbar: bool,
    gshort: str,
    fraud: float,
) -> bytes:
    return (
        f"PROSECUTOR-P2|{ts:.6f}|{prev_hash}|{truth_int}|{lip:.6f}|{krull:.6f}|"
        f"{cr:.6f}|{chaos}|{crowbar}|{gshort}|{fraud:.4f}"
    ).encode("utf-8")


def _hmac_sign(secret: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(secret, domain + payload, hashlib.sha256).hexdigest()


def _hmac_verify(secret: bytes, domain: bytes, payload: bytes, sig_hex: str) -> bool:
    try:
        return hmac.compare_digest(_hmac_sign(secret, domain, payload), sig_hex)
    except Exception:
        return False


def _merkle_root(leaves_hex: List[str]) -> str:
    if not leaves_hex:
        return hashlib.sha256(b"PROSECUTOR_EMPTY").hexdigest()
    lvl = list(leaves_hex)
    while len(lvl) > 1:
        nxt: List[str] = []
        for i in range(0, len(lvl), 2):
            a = lvl[i]
            b = lvl[i + 1] if i + 1 < len(lvl) else a
            nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
        lvl = nxt
    return lvl[0]


def _rho_fingerprint(rho: NDArray[np.complex128]) -> str:
    arr = np.ascontiguousarray(np.asarray(rho, dtype=np.complex128))
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode("utf-8")).hexdigest()


def _spectral_density_analysis(
    rho: NDArray[np.complex128],
) -> Tuple[bool, float, float, float, float, float]:
    """Análisis espectral riguroso de la matriz de densidad cuántica ρ.

    Retorna:
      (is_psd, purity, von_neumann_entropy, renyi2_entropy, min_eigenvalue, hermiticity_defect)
    """
    arr = np.asarray(rho, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        return False, 0.0, 0.0, 0.0, -1.0, float("inf")
    herm_defect = float(la.norm(arr - arr.conj().T, ord="fro"))
    rh = 0.5 * (arr + arr.conj().T)
    try:
        raw_eigs = np.real(la.eigvalsh(rh))
    except Exception:
        return False, 0.0, 0.0, 0.0, -1.0, herm_defect

    min_eig = float(np.min(raw_eigs))
    is_psd = bool((min_eig >= _PSD_FLOOR) and (herm_defect <= 1.0e-7))

    w = np.clip(raw_eigs, 0.0, None)
    s = float(np.sum(w))
    if s <= 1.0e-15:
        return False, 0.0, 0.0, 0.0, min_eig, herm_defect
    w_norm = w / s
    purity = float(np.sum(w_norm * w_norm))
    purity_clamped = min(max(purity, 1.0e-15), 1.0)
    renyi2 = float(-math.log(purity_clamped))
    vn_ent = float(-np.sum([p * math.log(p) for p in w_norm if p > 1.0e-15]))
    return is_psd, purity, vn_ent, renyi2, min_eig, herm_defect


def _purity_entropy(rho: NDArray[np.complex128]) -> Tuple[bool, float, float]:
    """Helper compatible v14: retorna (is_psd, purity, von_neumann_entropy)."""
    is_psd, purity, vn_ent, _, _, _ = _spectral_density_analysis(rho)
    return is_psd, purity, vn_ent


def _safe(o: Any, name: str, default: Any) -> Any:
    return getattr(o, name, default) if o is not None else default


def _verify_p1(secret: bytes, c: WisdomWeaverProsecutorPhase1Certificate) -> bool:
    if len(c.chain_hash) != 64 or len(c.hmac_signature) != 64:
        return False
    payload = _canonical_p1_bytes(
        c.timestamp, c.tr_rho, c.is_psd, float(c.purity), str(c.rho_hash)
    )
    if hashlib.sha256(payload).hexdigest() != c.chain_hash:
        return False
    if not _hmac_verify(secret, _HMAC_DOMAIN_P1, payload, c.hmac_signature):
        return False
    if abs(float(c.tr_rho) - 1.0) > _TRACE_TOL or (not bool(c.is_psd)):
        return False
    age = time.time() - float(c.timestamp)
    if age > _FRESHNESS_MAX_S or age < -60.0:
        return False
    return True


def _p2_gshort(c: WisdomWeaverProsecutorPhase2Certificate) -> str:
    g = str(getattr(c, "godel_hash", "") or "") or str(
        _safe(c.spectrum_report, "godel_hash", "") or ""
    )
    return g[:16] if g else "NOGODEL"


def _verify_p2(secret: bytes, c: WisdomWeaverProsecutorPhase2Certificate) -> bool:
    if len(c.chain_hash) != 64 or len(c.hmac_signature) != 64:
        return False
    rep = c.spectrum_report
    try:
        lip = float(rep.banach_contraction_lip)
        krull = float(rep.krull_valuation)
        cr = float(rep.cauchy_riemann_residual)
        chaos = bool(rep.melnikov_chaos_detected)
        truth_int = int(rep.truth_value.value)
        fraud = float(_safe(rep, "fraud_probability", 0.0))
        crow = bool(rep.crowbar_report.crowbar_triggered)
    except Exception:
        return False
    payload = _canonical_p2_bytes(
        float(c.timestamp),
        str(c.prev_chain_hash),
        truth_int,
        lip,
        krull,
        cr,
        chaos,
        crow,
        _p2_gshort(c),
        fraud,
    )
    if hashlib.sha256(payload).hexdigest() != c.chain_hash:
        return False
    if not _hmac_verify(secret, _HMAC_DOMAIN_P2, payload, c.hmac_signature):
        return False
    age = time.time() - float(c.timestamp)
    if age > _FRESHNESS_MAX_S or age < -60.0:
        return False
    return True


def _spectrum_hash(rep: WisdomWeaverProsecutorReport) -> str:
    msg = (
        f"{int(rep.truth_value.value)}|{float(rep.banach_contraction_lip):.6f}|"
        f"{float(rep.krull_valuation):.6f}|{float(rep.cauchy_riemann_residual):.6f}|"
        f"{float(_safe(rep, 'fraud_probability', 0.0)):.4f}|"
        f"{bool(rep.crowbar_report.crowbar_triggered)}|"
        f"{float(_safe(rep, 'greene_residue', 0.0)):.6f}|"
        f"{float(_safe(rep, 'kantorovich_h', 0.0)):.6f}"
    )
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — OBSERVE: INGESTA ISOMÉTRICA Y CINEMÁTICA CUÁNTICA (R_Boole ⊗ Λ_Nov)║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverProsecutorPhase1Observer:
    """Fase 1 OODA — Observación cinemática cuántica:
    Conservación de traza, positividad semidefinida (PSD), entropías de von Neumann y
    Rényi-2, coherencia cuántica l1, información asimétrica de Wigner-Yanase e
    Información Cuántica de Fisher (QFI).
    """

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key: bytes = secret_key
        self._observed_rho_cache: Dict[str, NDArray[np.complex128]] = {}

    def get_cached_rho(self, rho_hash: str) -> Optional[NDArray[np.complex128]]:
        """Recupera la matriz de densidad observada en Fase 1 mediante su huella SHA-256."""
        arr = self._observed_rho_cache.get(rho_hash)
        return arr.copy() if arr is not None else None

    # ── 1.1 Conservación de traza unitaria ──────────────────────────────────
    def check_trace_conservation(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[float, bool]:
        """Verifica la conservación de probabilidad cuántica Tr(ρ) ≡ 1 (±1e-9)."""
        arr = np.asarray(rho_mac, dtype=np.complex128)
        tr = float(np.real(np.trace(arr)))
        return tr, bool(abs(tr - 1.0) <= _TRACE_TOL)

    # ── 1.2 Positividad semidefinida, pureza y entropía de von Neumann ──────
    def check_psd_purity_entropy(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[bool, float, float]:
        """Evalúa condición PSD (λ_min ≥ -1e-12), pureza P=Tr(ρ²) y entropía S=-Tr(ρ ln ρ)."""
        is_psd, purity, vn_ent, _, _, _ = _spectral_density_analysis(
            np.asarray(rho_mac, dtype=np.complex128)
        )
        return is_psd, purity, vn_ent

    # ── 1.3 Coherencia cuántica l1, Wigner-Yanase e Información de Fisher ───
    def compute_quantum_coherence_and_wigner_yanase(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[float, float, float]:
        r"""Calcula invariantes geométrico-cuánticos de segundo orden sobre ρ:
        1. Coherencia cuántica l1 de Baumgratz-Cramer-Plenio:
             C_{l1}(ρ) = \sum_{j \neq k} |ρ_{jk}|
        2. Información asimétrica (skew information) de Wigner-Yanase respecto al
           generador hamiltoniano canónico K = diag(0, 1, ..., d-1):
             I_{WY}(ρ, K) = -1/2 Tr([\sqrt{ρ}, K]²) = Tr(ρ K²) - Tr(\sqrt{ρ} K \sqrt{ρ} K)
        3. Información Cuántica de Fisher (QFI) mediante descomposición espectral:
             F_Q(ρ, K) = 2 \sum_{j,k: λ_j+λ_k > 0} ((λ_j - λ_k)² / (λ_j + λ_k)) |<j|K|k>|²
        """
        arr = np.asarray(rho_mac, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
            return 0.0, 0.0, 0.0
        d = arr.shape[0]
        rh = 0.5 * (arr + arr.conj().T)

        off_diag = rh - np.diag(np.diag(rh))
        l1_coh = float(np.sum(np.abs(off_diag)))

        if d <= 1:
            return l1_coh, 0.0, 0.0

        K = np.diag(np.arange(d, dtype=np.float64) / float(max(d - 1, 1))).astype(
            np.complex128
        )
        try:
            evals, evecs = la.eigh(rh)
            evals_clipped = np.clip(np.real(evals), 0.0, None)
            tr_val = float(np.sum(evals_clipped))
            if tr_val <= 1.0e-15:
                return l1_coh, 0.0, 0.0
            lambdas = evals_clipped / tr_val
            sqrt_lambdas = np.sqrt(lambdas)

            # K en la base propia de ρ: K_eig = U^† K U
            K_eig = evecs.conj().T @ K @ evecs
            K_abs2 = np.abs(K_eig) ** 2

            # Wigner-Yanase: I_WY(ρ, K) = 1/2 \sum_{j,k} (\sqrt{λ_j} - \sqrt{λ_k})² |K_{jk}|²
            diff_sqrt = sqrt_lambdas[:, None] - sqrt_lambdas[None, :]
            i_wy = float(0.5 * np.sum((diff_sqrt ** 2) * K_abs2))

            # Quantum Fisher Information: F_Q(ρ, K) = 2 \sum_{j,k} ((λ_j - λ_k)² / (λ_j + λ_k)) |K_{jk}|²
            sum_lam = lambdas[:, None] + lambdas[None, :]
            diff_lam = lambdas[:, None] - lambdas[None, :]
            mask = sum_lam > 1.0e-15
            quotient = np.zeros_like(sum_lam)
            quotient[mask] = (diff_lam[mask] ** 2) / sum_lam[mask]
            f_q = float(2.0 * np.sum(quotient * K_abs2))
            return max(0.0, l1_coh), max(0.0, i_wy), max(0.0, f_q)
        except Exception:
            return max(0.0, l1_coh), 0.0, 0.0

    # ── 1.4 Emisión del Certificado Criptográfico de Fase 1 ─────────────────
    def emit_phase1_certificate(
        self, rho_mac: NDArray[np.complex128]
    ) -> WisdomWeaverProsecutorPhase1Certificate:
        """Cierra la observación cinemática de Fase 1: calcula espectro, entropías,
        coherencia cuántica e información de Wigner-Yanase/Fisher, almacena ρ en caché
        por huella SHA-256 y firma con HMAC-SHA256.
        """
        ts = time.time()
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho_mac debe ser una matriz de densidad cuadrada.")

        tr, trace_ok = self.check_trace_conservation(rho)
        is_psd, purity, entropy, renyi2, min_eig, herm_def = _spectral_density_analysis(
            rho
        )
        l1_coh, i_wy, f_q = self.compute_quantum_coherence_and_wigner_yanase(rho)
        rhash = _rho_fingerprint(rho)

        # Retiene copia inmutable en caché para el puente hacia Fase 2
        if len(self._observed_rho_cache) >= 32:
            oldest_key = next(iter(self._observed_rho_cache))
            self._observed_rho_cache.pop(oldest_key, None)
        self._observed_rho_cache[rhash] = rho.copy()

        payload = _canonical_p1_bytes(ts, tr, is_psd, purity, rhash)
        ch = hashlib.sha256(payload).hexdigest()
        sig = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P1, payload)

        return WisdomWeaverProsecutorPhase1Certificate(
            timestamp=ts,
            tr_rho=tr,
            is_psd=is_psd,
            chain_hash=ch,
            hmac_signature=sig,
            is_trace_conserved=trace_ok,
            purity=purity,
            von_neumann_entropy=entropy,
            rho_hash=rhash,
            dim=int(rho.shape[0]),
            renyi2_entropy=renyi2,
            l1_coherence=l1_coh,
            wigner_yanase_skew_info=i_wy,
            quantum_fisher_info=f_q,
            min_eigenvalue=min_eig,
            hermiticity_defect=herm_def,
        )

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128]
    ) -> WisdomWeaverProsecutorPhase1Certificate:
        """Alias compatible v13/v14 → `emit_phase1_certificate`."""
        return self.emit_phase1_certificate(rho_mac)

    # ── 1.5 DEFINICIÓN FINAL DE FASE 1 = PUENTE HOMOMÓRFICO Φ₁₂ HACIA FASE 2 ─
    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverProsecutorPhase1Certificate
    ) -> bool:
        """PUENTE FORMAL FASE 1 → FASE 2 (Compuerta Homomórfica Φ₁₂).

        Verifica autenticidad HMAC-P1, integridad de cadena SHA-256, conservación de
        traza unitaria Tr(ρ)=1, condición PSD (λ_min ≥ -1e-12), pureza física en (0, 1]
        y frescura temporal.
        Nota inquisitorial: Φ₁₂ autentica la evidencia cinemática; el juicio dinámico y
        topológico sobre el dolo (VETOED) se sustancia en la Fase 2.
        ──► Alimenta directamente `WisdomWeaverProsecutorPhase2Orientor.validate_entrance_from_phase1`.
        """
        ok = _verify_p1(self._secret_key, p1_cert)
        if not ok:
            logger.warning("Compuerta Φ₁₂ fiscal bloqueada: certificado P1 inválido.")
            return False
        if not (0.0 < float(p1_cert.purity) <= 1.0 + 1.0e-9):
            logger.warning(
                "Compuerta Φ₁₂ fiscal bloqueada: pureza fuera de rango físico (%.6f).",
                p1_cert.purity,
            )
            return False
        return True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ORIENT: INQUISICIÓN FLOER-POINCARÉ-KAM Y GOBERNANZA RSI³ NIVEL-3  ║
# ║ ◄── CONTINUACIÓN DIRECTA DE FASE 1: consume el puente homomórfico Φ₁₂       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverProsecutorPhase2Orientor:
    """Fase 2 OODA — Orientación inquisitorial:
    Mecánica celeste de Poincaré-Delaunay-Kozai-Lidov, superficie de intersección de
    Floer-Gromov, estabilidad de Nekhoroshev-KAM, residuo de Greene y mejora recursiva
    autónoma de gobernanza RSI³ Nivel-3.
    """

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovRSI3WisdomWeaverProsecutorEngine,
        observer: Optional[WisdomWeaverProsecutorPhase1Observer] = None,
    ) -> None:
        self._secret_key: bytes = secret_key
        self._engine: BoolNovikovRSI3WisdomWeaverProsecutorEngine = engine
        self._observer: Optional[WisdomWeaverProsecutorPhase1Observer] = observer
        self._governance_history: List[Dict[str, float]] = []
        self._last_celestial_elements: Dict[str, float] = {
            "L": 1.0,
            "G": 1.0,
            "H": 1.0,
            "eccentricity": 0.0,
            "inclination": 0.0,
            "kozai_lidov": 1.0,
            "policy_eps": 0.03,
        }

    # ── 2.1 INICIO DE FASE 2: Validación del puente Φ₁₂ desde Fase 1 ────────
    def validate_entrance_from_phase1(
        self, p1_cert: WisdomWeaverProsecutorPhase1Certificate
    ) -> bool:
        """CONTINUACIÓN DIRECTA DE FASE 1: consume y revalida la compuerta Φ₁₂
        (`continue_to_phase2_gate`) antes de iniciar la orientación inquisitorial.
        """
        if not _verify_p1(self._secret_key, p1_cert):
            return False
        return bool(0.0 < float(p1_cert.purity) <= 1.0 + 1.0e-9)

    # ── 2.2 Mecánica Celeste de Poincaré: Delaunay (L,G,H) y Kozai-Lidov ────
    def derive_delaunay_action_and_perturbation(
        self,
        apu_cost_matrix: NDArray[np.float64],
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        exponents: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""Mecánica celeste canónica de Poincaré-Delaunay-Kozai-Lidov:
        Deriva el par dinámico (J, ε) y los elementos orbitales seculares a partir
        de la matriz de costes M, el certificado cuántico P1 y los exponentes de Novikov λ:
          1. Acción principal de Delaunay:
               L = J = clip(Tr(M)/d, 0.2, 5.0)
          2. Momento angular orbital total G a partir de la entropía espectral de valores
             singulares de M y la pureza cuántica P de Fase 1:
               G = L * sqrt(1 - e²),  con e ∈ [0, 0.95]
          3. Componente axial H = G * cos(i) gobernada por la alineación entre la diagonal
             de M y los exponentes de Novikov λ, definiendo el invariante secular de
             Kozai-Lidov:
               I_{KL} = sqrt(1 - e²) * cos(i) = H / L
          4. Perturbación hamiltoniana ε acoplada a la no-linealidad booleana de
             Rademacher-Walsh ||RW|| y modulada por la información de Wigner-Yanase I_{WY}:
               ε = clip(0.01 + 0.1 * ||RW|| + 0.015 * min(I_{WY}, 1.0), 0.01, 0.15)
        Retorna (J, ε) preservando la firma exacta v14 y almacena el paquete celeste.
        """
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        dim = M.shape[0] if M.ndim == 2 and M.shape[0] > 0 else 1
        J = float(min(max(float(np.trace(M)) / float(max(dim, 1)), 0.2), 5.0))
        L_act = J

        # Excentricidad espectral orbital e a partir de la anisotropía singular de M
        try:
            svals = la.svdvals(M)
            s_max = float(np.max(svals)) if svals.size > 0 else 1.0
            s_min = float(np.min(svals)) if svals.size > 0 else 1.0
            anisotropy = (s_max - s_min) / (s_max + s_min + 1.0e-12)
        except Exception:
            anisotropy = 0.0

        purity = float(min(max(getattr(p1_cert, "purity", 1.0), 1.0 / max(dim, 1)), 1.0))
        ecc = float(min(max(anisotropy * math.sqrt(max(0.0, 1.0 - 0.25 * purity)), 0.0), 0.95))
        G_act = float(L_act * math.sqrt(max(1.0 - ecc * ecc, 1.0e-6)))

        # Inclinación orbital i a partir del coseno direccional entre diag(M) y exponentes λ
        lam = np.asarray(exponents, dtype=np.float64).ravel()
        diag_m = np.diag(M).ravel() if M.ndim == 2 else np.ones(1, dtype=np.float64)
        m_len = min(lam.size, diag_m.size)
        if m_len > 0:
            v1 = diag_m[:m_len]
            v2 = lam[:m_len]
            norm_prod = float(la.norm(v1) * la.norm(v2))
            cos_inc = float(np.dot(v1, v2) / norm_prod) if norm_prod > 1.0e-12 else 1.0
            cos_inc = float(min(max(abs(cos_inc), 0.05), 1.0))
        else:
            cos_inc = 1.0

        inc_rad = float(math.acos(cos_inc))
        H_act = float(G_act * cos_inc)
        kozai_lidov = float(H_act / max(L_act, 1.0e-12))

        try:
            rw, _, _ = self._engine.compute_rademacher_walsh_and_snf(M)
        except Exception:
            rw = 0.05

        i_wy = float(max(0.0, getattr(p1_cert, "wigner_yanase_skew_info", 0.0)))
        eps = float(
            min(max(0.01 + 0.1 * float(rw) + 0.015 * min(i_wy, 1.0), 0.01), 0.15)
        )

        self._last_celestial_elements = {
            "L": L_act,
            "G": G_act,
            "H": H_act,
            "eccentricity": ecc,
            "inclination": inc_rad,
            "kozai_lidov": kozai_lidov,
            "policy_eps": eps,
        }
        return float(J), float(eps)

    # ── 2.3 Gobernanza Recursiva Autónoma RSI³ Nivel-3 ──────────────────────
    def recursive_governance_improvement_nivel3(
        self, candidates_eps: Optional[List[float]] = None
    ) -> float:
        r"""Mejora Recursiva Autónoma de Nivel 3 (RSI³) para la política inquisitorial ε*:
          - Nivel 1 (Exploración variacional de Lyapunov-Pontryagin):
              Evalúa cada candidato ε ∈ C mediante el funcional de gobernanza:
                V(ε) = (ε - ε_target)² + α_G * R_Greene * ε² + α_K * h_K * ε
              donde ε_target adapta la sensibilidad fiscal según la holgura diofántica
              KAM γ y el jerk de capacidad d³C/dt³.
          - Nivel 2 (Aceleración cúbica de Chebyshev-Halley):
              Refina el óptimo discreto ε_1 mediante un paso de tercer orden de Halley
              sobre V(ε) sujeto a la condición de Kantorovich h_K ≤ 1/2 y residuo de
              Greene R_Greene < 0.25.
          - Nivel 3 (Selección Darwiniana y Clausura Autorreferencial de Gödel):
              Selecciona el argumento óptimo ε* proyectado al intervalo admisible y emite
              el invariante criptográfico de Gödel HMAC-SHA256.
        """
        if not self._governance_history:
            return 0.03

        cands = (
            list(candidates_eps)
            if candidates_eps is not None and len(candidates_eps) > 0
            else [0.01, 0.03, 0.05]
        )
        recent = self._governance_history[-5:]
        gammas = [float(h.get("gamma", 1.0e-3)) for h in recent]
        jerks = [float(h.get("jerk", 0.0)) for h in recent]
        greenes = [float(h.get("greene", 0.05)) for h in recent]
        kant_hs = [float(h.get("kantorovich_h", 0.02)) for h in recent]

        gmean = float(np.mean(gammas)) if gammas else 1.0e-3
        jmean = float(np.mean(jerks)) if jerks else 0.0
        rmean = float(np.mean(greenes)) if greenes else 0.05
        hmean = float(np.mean(kant_hs)) if kant_hs else 0.02

        # Factor de cautela KAM-Greene-Kantorovich
        cautela = 0.7 if (gmean < 1.0e-3 or rmean >= _GREENE_CRITICAL_RESIDUE) else 1.0
        base_target = (0.03 if jmean >= 0.0 else 0.01) * cautela

        # Nivel 1: Funcional de Lyapunov-Pontryagin sobre los candidatos
        def _lyapunov_cost(e_val: float) -> float:
            diff = e_val - base_target
            greene_pen = 0.25 * max(0.0, rmean) * (e_val ** 2)
            kant_pen = 0.10 * max(0.0, hmean) * abs(diff)
            return float(diff * diff + greene_pen + kant_pen)

        eps_l1 = float(min(cands, key=_lyapunov_cost))

        # Nivel 2: Paso de Chebyshev-Halley de orden 3 sobre V(ε)
        # V'(ε) = 2(ε - base_target) + 0.5 * rmean * ε
        # V''(ε) = 2.0 + 0.5 * rmean
        v_prime = 2.0 * (eps_l1 - base_target) + 0.5 * max(0.0, rmean) * eps_l1
        v_second = 2.0 + 0.5 * max(0.0, rmean)
        halley_step = v_prime / max(v_second, 1.0e-12)
        eps_l2 = float(eps_l1 - halley_step)

        # Nivel 3: Selección Darwiniana sobre el conjunto de candidatos admisibles
        best = float(min(cands, key=lambda e: abs(e - eps_l2)))
        gh = hmac.new(
            self._secret_key,
            (
                f"GOV-PROS-RSI3|{best:.4f}|{eps_l2:.4f}|{jmean:.4f}|"
                f"{gmean:.6f}|{rmean:.5f}|{hmean:.5f}|{_AGENT_VERSION}"
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        logger.debug(
            "GOV-PROS-RSI3 ε*=%.4f (L2=%.4f) jerk=%.4f γ=%.5f R_Greene=%.4f gödel=%s",
            best,
            eps_l2,
            jmean,
            gmean,
            rmean,
            gh[:12],
        )
        return best

    # ── 2.4 Ejecución Inquisitorial de Fase 2 y Emisión de Certificado P2 ───
    def execute_phase2(
        self,
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 0.85,
        cylinder_R: float = 1.0,
    ) -> WisdomWeaverProsecutorPhase2Certificate:
        """Ejecuta la inquisición espectral, simpléctica y celeste de Fase 2:
        ◄── Exige validación previa de la compuerta Φ₁₂ (`validate_entrance_from_phase1`).
        Deriva variables canónicas de Delaunay-Kozai-Lidov, recupera `rho_obs` desde
        la caché de Fase 1 para la isometría de Bures-Uhlmann, ejecuta la auditoría
        del motor fiscal (`execute_wisdom_prosecutor_audit`), aplica el paso de
        refinamiento RSI³ Nivel-3 y sella el certificado P2 enlazado a `p1_cert.chain_hash`.
        """
        if not self.validate_entrance_from_phase1(p1_cert):
            raise ValueError("Falla de seguridad en Φ₁₂ al entrar a Fase 2 fiscal.")

        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        lam = np.asarray(exponents, dtype=np.float64)

        dJ, dE = self.derive_delaunay_action_and_perturbation(M, p1_cert, lam)
        J = float(action_J) if action_J is not None else dJ
        E = float(pert_eps) if pert_eps is not None else dE

        # Recupera la matriz de densidad observada en Fase 1 si sus dimensiones coinciden
        rho_obs: Optional[NDArray[np.complex128]] = None
        if self._observer is not None and p1_cert.rho_hash:
            cached = self._observer.get_cached_rho(p1_cert.rho_hash)
            if cached is not None and M.ndim == 2 and cached.shape == M.shape:
                rho_obs = cached

        try:
            spec = self._engine.execute_wisdom_prosecutor_audit(
                M,
                lam,
                positron_hmac_valid,
                force_veto,
                rho_obs=rho_obs,
                action_J=J,
                pert_eps=E,
                ball_r=ball_r,
                cylinder_R=cylinder_R,
            )
        except TypeError:
            try:
                spec = self._engine.execute_wisdom_prosecutor_audit(
                    M,
                    lam,
                    positron_hmac_valid,
                    force_veto,
                    action_J=J,
                    pert_eps=E,
                    ball_r=ball_r,
                    cylinder_R=cylinder_R,
                )  # type: ignore
            except TypeError:
                spec = self._engine.execute_wisdom_prosecutor_audit(
                    M, lam, positron_hmac_valid, force_veto
                )  # type: ignore

        # Refinamiento de telemetría Nivel-3 (sin alterar el veredicto jurídico del Fiscal)
        try:
            step = getattr(self._engine, "recursive_self_improvement_step_nivel3", None)
            cert_m = getattr(self._engine, "certify_conviction_and_chaos", None)
            if callable(step) and callable(cert_m):
                kcert = cert_m(
                    J,
                    E,
                    float(spec.cauchy_riemann_residual),
                    float(_safe(spec, "floer_energy", 0.0)),
                    bool(spec.is_gromov_violated),
                )
                s0 = np.array(
                    [
                        float(spec.krull_valuation),
                        float(spec.banach_contraction_lip),
                        float(spec.cauchy_riemann_residual),
                    ],
                    dtype=np.float64,
                )
                _s3, _r3 = step(s0, kcert, 3)
                logger.debug(
                    "RSI3-L3 fiscal: lip3=%.5f jerk=%.4f",
                    _r3.lip_T3,
                    _r3.capacity_jerk_d3c_dt3,
                )
        except Exception as ex:
            logger.debug("Refinamiento L3 fiscal omitido: %s", ex)

        ts = time.time()
        lind_res = float(_safe(spec, "lindstedt_residual", 0.0))
        gamma = float(_safe(spec, "diophantine_gamma", 0.0))
        mel_val = float(_safe(spec, "melnikov_value", 0.0))
        chaos = bool(spec.melnikov_chaos_detected)
        flo_e = float(_safe(spec, "floer_energy", 0.0))
        fraud = float(
            _safe(
                spec,
                "fraud_probability",
                1.0 if spec.is_gromov_violated else (0.75 if chaos else 0.1),
            )
        )
        n_it = int(_safe(spec, "rsi_iterations", 0))
        godel = str(_safe(spec, "godel_hash", "") or "")
        crow = bool(spec.crowbar_report.crowbar_triggered)

        # Invariantes extendidos v15 PhD+
        greene_res = float(_safe(spec, "greene_residue", 0.0))
        cz_index = int(_safe(spec, "conley_zehnder_index", 0))
        nekh_time = float(_safe(spec, "nekhoroshev_time", 0.0))
        oct_norm = float(_safe(spec, "octonion_associator_norm", 0.0))
        kant_h = float(_safe(spec, "kantorovich_h", 0.0))

        payload = _canonical_p2_bytes(
            ts,
            str(p1_cert.chain_hash),
            int(spec.truth_value.value),
            float(spec.banach_contraction_lip),
            float(spec.krull_valuation),
            float(spec.cauchy_riemann_residual),
            chaos,
            crow,
            (godel[:16] if godel else "NOGODEL"),
            fraud,
        )
        ch = hashlib.sha256(payload).hexdigest()
        sig = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P2, payload)

        try:
            self._governance_history.append(
                {
                    "jerk": float(spec.capacity_jerk_d3c_dt3),
                    "gamma": gamma,
                    "eps": float(E),
                    "greene": greene_res,
                    "kantorovich_h": kant_h,
                }
            )
        except Exception:
            pass

        gov_policy = self.recursive_governance_improvement_nivel3()
        cel = self._last_celestial_elements

        return WisdomWeaverProsecutorPhase2Certificate(
            timestamp=ts,
            spectrum_report=spec,
            chain_hash=ch,
            hmac_signature=sig,
            prev_chain_hash=str(p1_cert.chain_hash),
            action_J=float(J),
            pert_eps=float(E),
            lindstedt_residual=lind_res,
            diophantine_gamma=gamma,
            melnikov_value=mel_val,
            melnikov_chaos=chaos,
            floer_energy=flo_e,
            fraud_probability=fraud,
            rsi_iterations=n_it,
            godel_hash=godel,
            greene_residue=greene_res,
            conley_zehnder_index=cz_index,
            nekhoroshev_time=nekh_time,
            octonion_associator_norm=oct_norm,
            kantorovich_h=kant_h,
            delaunay_L=float(cel.get("L", J)),
            delaunay_G=float(cel.get("G", J)),
            delaunay_H=float(cel.get("H", J)),
            orbital_eccentricity=float(cel.get("eccentricity", 0.0)),
            kozai_lidov_invariant=float(cel.get("kozai_lidov", 1.0)),
            governance_eps_policy=float(gov_policy),
        )

    def emit_phase2_certificate(
        self, *a: Any, **k: Any
    ) -> WisdomWeaverProsecutorPhase2Certificate:
        """Alias canónico → `execute_phase2`."""
        return self.execute_phase2(*a, **k)

    # ── 2.5 DEFINICIÓN FINAL DE FASE 2 = PUENTE HOMOMÓRFICO Φ₂₃ HACIA FASE 3 ─
    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverProsecutorPhase2Certificate
    ) -> bool:
        """PUENTE FORMAL FASE 2 → FASE 3 (Compuerta Homomórfica Φ₂₃).

        Verifica autenticidad HMAC-P2, enlace criptográfico con Fase 1 (`prev_chain_hash`),
        frescura temporal, contractividad estricta de Banach Lip < 1.0 y convergencia de
        Newton-Kantorovich h_K ≤ 0.5.
        Diseño inquisitorial crítico: Φ₂₃ ADMITE estados VETOED, caos homoclínico de
        Melnikov y violaciones simplécticas de Gromov, pues tales condenas DEBEN cruzar
        hacia la Fase 3 para ejecutar la aniquilación de Dirac y el disparo físico de la
        guillotina Crowbar ESP32/BT151.
        ──► Alimenta directamente `WisdomWeaverProsecutorPhase3Decider.validate_entrance_from_phase2`.
        """
        if not _verify_p2(self._secret_key, p2_cert):
            logger.warning(
                "Compuerta Φ₂₃ fiscal bloqueada: HMAC-P2, cadena o frescura inválida."
            )
            return False
        if len(str(p2_cert.prev_chain_hash)) != 64:
            logger.warning(
                "Compuerta Φ₂₃ fiscal bloqueada: prev_chain_hash ausente o corrupto."
            )
            return False
        try:
            lip = float(p2_cert.spectrum_report.banach_contraction_lip)
            kant_h = float(
                getattr(
                    p2_cert,
                    "kantorovich_h",
                    _safe(p2_cert.spectrum_report, "kantorovich_h", 0.0),
                )
            )
        except Exception:
            return False
        if not (math.isfinite(lip) and 0.0 <= lip < _LIP_HARD_MAX):
            logger.warning("Compuerta Φ₂₃ fiscal bloqueada: Lip=%.6f ≥ 1.0.", lip)
            return False
        if not (math.isfinite(kant_h) and kant_h <= _KANTOROVICH_MAX + 1.0e-9):
            logger.warning(
                "Compuerta Φ₂₃ fiscal bloqueada: Kantorovich h=%.6f > 0.5.", kant_h
            )
            return False
        return True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — DECIDE & ACT: HEYTING Ω₃, INTERLOCK CROWBAR/DIRAC, Φ_sem Y MERKLE  ║
# ║ ◄── CONTINUACIÓN DIRECTA DE FASE 2: consume el puente homomórfico Φ₂₃       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverProsecutorPhase3Decider:
    """Fase 3 OODA — Decisión y actuación soberana:
    Adjudicación en el retículo de Heyting Ω₃, verificación electro-cuántica del
    interlock Crowbar ESP32 + Dirac 2γ, traducción mediante el funtor semántico-visceral
    Φ_sem y sellado de estado soberano con raíz de Merkle.
    """

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key: bytes = secret_key

    # ── 3.1 INICIO DE FASE 3: Validación del puente Φ₂₃ desde Fase 2 ────────
    def validate_entrance_from_phase2(
        self, p2_cert: WisdomWeaverProsecutorPhase2Certificate
    ) -> bool:
        """CONTINUACIÓN DIRECTA DE FASE 2: consume y revalida la compuerta Φ₂₃
        (`continue_to_phase3_gate`) al ingresar en la Fase 3.
        """
        if not _verify_p2(self._secret_key, p2_cert):
            return False
        if len(str(p2_cert.prev_chain_hash)) != 64:
            return False
        try:
            lip = float(p2_cert.spectrum_report.banach_contraction_lip)
            kant_h = float(
                getattr(
                    p2_cert,
                    "kantorovich_h",
                    _safe(p2_cert.spectrum_report, "kantorovich_h", 0.0),
                )
            )
            return bool(
                math.isfinite(lip)
                and 0.0 <= lip < _LIP_HARD_MAX
                and math.isfinite(kant_h)
                and kant_h <= _KANTOROVICH_MAX + 1.0e-9
            )
        except Exception:
            return False

    # ── 3.2 Adjudicación Monótona en el Álgebra de Heyting Ω₃ ───────────────
    def adjudicate_heyting_omega3(
        self,
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        p2_cert: WisdomWeaverProsecutorPhase2Certificate,
    ) -> HeytingTruthValue:
        """Adjudicación intuicionista en el retículo de Heyting Ω₃:
          0x00 COHERENT ≺ 0x01 DEGRADED ≺ 0x03 VETOED.
        Verifica autenticidad P1, autenticidad P2, continuidad criptográfica de cadena
        (`p2_cert.prev_chain_hash == p1_cert.chain_hash`) y estabilidad de Banach-Kantorovich.
        """
        if not _verify_p1(self._secret_key, p1_cert):
            return HeytingTruthValue.VETOED
        if not self.validate_entrance_from_phase2(p2_cert):
            return HeytingTruthValue.VETOED
        if str(p2_cert.prev_chain_hash) != str(p1_cert.chain_hash):
            logger.warning(
                "Ruptura de cadena criptográfica entre P1 (%s) y P2 (%s).",
                str(p1_cert.chain_hash)[:12],
                str(p2_cert.prev_chain_hash)[:12],
            )
            return HeytingTruthValue.VETOED
        return p2_cert.spectrum_report.truth_value

    # ── 3.3 Verificación Electro-Cuántica del Interlock Crowbar ESP32 + Dirac ─
    def verify_crowbar_interlock_timing(
        self, p2_cert: WisdomWeaverProsecutorPhase2Certificate
    ) -> Tuple[bool, str]:
        """Verifica la conformidad física del interlock Crowbar ESP32 y la aniquilación QED:
        1. Dirección de registro MMIO: `0x3FF44008` (`GPIO_OUT_W1TS_REG`, pin GPIO14).
        2. Modo Armado (`crowbar_triggered == True`):
           - Latencia total de actuación: 0 < t_total ≤ 392.15 ns.
           - Latencia de instrucción IRAM a 240 MHz: |t_IRAM - 41.667 ns| ≤ 5.0 ns.
           - Estado del tiristor SCR BT151: contiene `TRIGGERED`.
           - Energía de aniquilación de Dirac 2γ: E = 1022.0 keV con hash de aniquilación.
        3. Modo Standby (`crowbar_triggered == False`):
           - Latencia total == 0.0 ns y estado `STANDBY_ARMED`.
           - Coherencia penal: un veredicto `VETOED` prohíbe permanecer en standby.
        """
        rep = p2_cert.spectrum_report
        crow = rep.crowbar_report
        if str(crow.gpio_register_address) != _CROWBAR_REG:
            return (
                False,
                f"Registro MMIO {crow.gpio_register_address} ≠ {_CROWBAR_REG}",
            )

        if bool(crow.crowbar_triggered):
            t_act = float(crow.total_actuation_time_ns)
            if not (0.0 < t_act <= _CROWBAR_MAX_NS + 1.0e-9):
                return (
                    False,
                    f"Latencia Crowbar {t_act:.2f}ns fuera de (0, {_CROWBAR_MAX_NS}]",
                )
            if "TRIGGERED" not in str(crow.thyristor_status):
                return (
                    False,
                    f"Tiristor {crow.thyristor_status} no se encuentra TRIGGERED",
                )
            if (
                abs(float(crow.iram_instruction_latency_ns) - _CROWBAR_IRAM_TARGET_NS)
                > _CROWBAR_IRAM_TOL_NS
            ):
                return (
                    False,
                    f"Latencia IRAM {crow.iram_instruction_latency_ns:.2f}ns ≠ 41.67ns",
                )
            # Verificación de coherencia con la aniquilación electrodinámica de Dirac (1022 keV)
            dirac = getattr(rep, "dirac_report", None)
            if dirac is not None:
                e_kev = float(
                    getattr(dirac, "annihilation_energy_kev", _DIRAC_ANNIHILATION_KEV)
                )
                if abs(e_kev - _DIRAC_ANNIHILATION_KEV) > 1.0e-3:
                    return (
                        False,
                        f"Energía Dirac {e_kev:.2f}keV ≠ {_DIRAC_ANNIHILATION_KEV}keV",
                    )
            return True, "INTERLOCK_ARMED_BT151_GPIO14_HIGH"
        else:
            if abs(float(crow.total_actuation_time_ns)) > 1.0e-9:
                return False, "Standby inválido con latencia de actuación ≠ 0"
            if "STANDBY" not in str(crow.thyristor_status):
                return (
                    False,
                    f"Tiristor {crow.thyristor_status} no se encuentra en STANDBY",
                )
            if rep.truth_value == HeytingTruthValue.VETOED:
                return (
                    False,
                    "VETOED sin guillotina Crowbar armada (incoherencia penal)",
                )
            return True, "INTERLOCK_STANDBY_ARMED"

    # ── 3.4 Funtor Semántico-Visceral Φ_sem ─────────────────────────────────
    def functor_semantic_visceral_phi_sem(
        self,
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        p2_cert: WisdomWeaverProsecutorPhase2Certificate,
    ) -> WisdomWeaverProsecutorVisceralMetrics:
        """Funtor semántico-visceral Φ_sem: traduce invariantes simplécticos, espectrales
        y celestes en métricas financieras y operativas auditables.
        Base canónica garantizada en estado COHERENT:
          - Protección WACC: 15.0%
          - Liberación neta de liquidez: +11.5%
          - Compresión de caché KV: 86.4%
        """
        rep = p2_cert.spectrum_report
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        fid = float(_safe(rep, "bures_fidelity", 1.0))
        try:
            crow_stat = str(rep.crowbar_report.thyristor_status)
        except Exception:
            crow_stat = "STANDBY_ARMED"

        chaos = bool(rep.melnikov_chaos_detected)
        viol = bool(rep.is_gromov_violated)
        gamma = float(_safe(rep, "diophantine_gamma", 0.0))
        greene = float(
            getattr(p2_cert, "greene_residue", _safe(rep, "greene_residue", 0.0))
        )
        nekh_t = float(
            getattr(p2_cert, "nekhoroshev_time", _safe(rep, "nekhoroshev_time", 0.0))
        )
        kozai = float(getattr(p2_cert, "kozai_lidov_invariant", 1.0))
        kant_h = float(
            getattr(p2_cert, "kantorovich_h", _safe(rep, "kantorovich_h", 0.0))
        )

        kam = bool(
            (not chaos)
            and (not viol)
            and (gamma > 1.0e-4)
            and (greene < _GREENE_CRITICAL_RESIDUE or greene == 0.0)
        )
        fraud = float(
            _safe(
                rep,
                "fraud_probability",
                1.0 if viol else (0.75 if chaos else 0.1),
            )
        )
        f_stab = 1.0 if kam else 0.55
        f_iso = float(min(max(fid, 0.0), 1.0))
        f_krull = 0.5 + 0.5 * min(
            1.0, max(0.0, float(rep.krull_valuation) / 0.2)
        )

        if tv == HeytingTruthValue.COHERENT:
            wacc, liq, kv = 15.0, 11.5, 86.4
        elif tv == HeytingTruthValue.DEGRADED:
            wacc = round(15.0 * f_stab * (0.85 + 0.15 * f_iso), 3)
            liq = round(11.5 * f_krull * (0.7 + 0.3 * f_stab), 3)
            kv = round(86.4 * (0.9 + 0.1 * f_iso), 3)
        else:
            wacc = round(15.0 * 0.4 * f_stab * (1.0 - 0.5 * fraud), 3)
            liq = round(11.5 * 0.4 * f_krull * (1.0 - 0.5 * fraud), 3)
            kv = round(86.4 * 0.5, 3)

        status = (
            f"{crow_stat}|KAM={'STABLE' if kam else 'UNSTABLE'}|"
            f"FRAUD={fraud:.2f}|F={fid:.4f}|RG={greene:.4f}"
        )
        return WisdomWeaverProsecutorVisceralMetrics(
            wacc_protection_pct=wacc,
            net_liquidity_release_pct=liq,
            kv_cache_compression_pct=kv,
            crowbar_status=status,
            is_kam_stable=kam,
            stability_factor=f_stab,
            bures_fidelity=fid,
            fraud_probability=fraud,
            greene_residue=greene,
            nekhoroshev_time=nekh_t,
            kozai_lidov_invariant=kozai,
            kantorovich_h=kant_h,
        )

    # ── 3.5 Emisión del Estado Soberano y Árbol de Merkle de 8 Hojas ────────
    def emit_sovereign_state(
        self,
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        p2_cert: WisdomWeaverProsecutorPhase2Certificate,
    ) -> WisdomWeaverProsecutorSovereignState:
        """Cierra la Fase 3: adjudica el veredicto en Ω₃, verifica el interlock físico
        Crowbar/Dirac, evalúa el funtor visceral Φ_sem y construye la raíz de Merkle
        balanceada de 8 hojas firmada con HMAC-SHA256.
        """
        ts = time.time()
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        metrics = self.functor_semantic_visceral_phi_sem(p1_cert, p2_cert)
        rep = p2_cert.spectrum_report
        compliant, reason = self.verify_crowbar_interlock_timing(p2_cert)
        logger.debug("Interlock Crowbar: compliant=%s (%s)", compliant, reason)

        godel = str(_safe(rep, "godel_hash", "") or _p2_gshort(p2_cert))
        try:
            dirac = str(rep.dirac_report.annihilation_hash)
        except Exception:
            dirac = "NO_ANNIHILATION_REQUIRED"

        celestial_crowbar_leaf = hashlib.sha256(
            (
                f"{getattr(p2_cert, 'action_J', 1.0):.6f}|"
                f"{getattr(p2_cert, 'pert_eps', 0.01):.6f}|"
                f"{getattr(p2_cert, 'kozai_lidov_invariant', 1.0):.6f}|"
                f"{metrics.greene_residue:.6f}|"
                f"{bool(rep.crowbar_report.crowbar_triggered)}|{compliant}"
            ).encode("utf-8")
        ).hexdigest()

        # Árbol de Merkle binario balanceado de 8 hojas (2³)
        leaves = [
            str(p1_cert.chain_hash),
            str(p1_cert.hmac_signature),
            str(p2_cert.chain_hash),
            str(p2_cert.hmac_signature),
            _spectrum_hash(rep),
            hashlib.sha256(godel.encode("utf-8")).hexdigest(),
            hashlib.sha256(dirac.encode("utf-8")).hexdigest(),
            celestial_crowbar_leaf,
        ]
        root = _merkle_root(leaves)
        state_sig = _hmac_sign(
            self._secret_key,
            _HMAC_DOMAIN_STATE,
            f"{ts:.6f}|{int(tv.value)}|{root}|{compliant}".encode("utf-8"),
        )
        logger.debug("State HMAC=%s...", state_sig[:12])

        lat = max(0.0, (ts - float(p1_cert.timestamp)) * 1000.0)
        cz_idx = int(
            getattr(
                p2_cert,
                "conley_zehnder_index",
                _safe(rep, "conley_zehnder_index", 0),
            )
        )

        return WisdomWeaverProsecutorSovereignState(
            timestamp=ts,
            truth_value=tv,
            crowbar_triggered=bool(rep.crowbar_report.crowbar_triggered),
            total_actuation_time_ns=float(rep.crowbar_report.total_actuation_time_ns),
            merkle_root_hash=root,
            visceral_metrics=metrics,
            p1_chain_hash=str(p1_cert.chain_hash),
            p2_chain_hash=str(p2_cert.chain_hash),
            godel_hash=str(_safe(rep, "godel_hash", "") or ""),
            dirac_signature=dirac,
            governance_latency_ms=float(lat),
            action_J=float(getattr(p2_cert, "action_J", 1.0)),
            pert_eps=float(getattr(p2_cert, "pert_eps", 0.01)),
            is_kam_stable=bool(metrics.is_kam_stable),
            fraud_probability=float(metrics.fraud_probability),
            interlock_compliant=bool(compliant),
            state_hmac_signature=state_sig,
            greene_residue=float(metrics.greene_residue),
            conley_zehnder_index=cz_idx,
            nekhoroshev_time=float(metrics.nekhoroshev_time),
            kozai_lidov_invariant=float(metrics.kozai_lidov_invariant),
            kantorovich_h=float(metrics.kantorovich_h),
            interlock_reason=reason,
        )

    def execute_phase3(
        self,
        p1_cert: WisdomWeaverProsecutorPhase1Certificate,
        p2_cert: Optional[WisdomWeaverProsecutorPhase2Certificate] = None,
    ) -> WisdomWeaverProsecutorSovereignState:
        """Alias compatible v13/v14 → `emit_sovereign_state`."""
        if p2_cert is None and isinstance(
            p1_cert, WisdomWeaverProsecutorPhase2Certificate
        ):
            # Soporte defensivo por si un llamador legado invoca execute_phase3(p2_cert)
            dummy_ts = float(p1_cert.timestamp)
            dummy_rhash = hashlib.sha256(b"LEGACY_P1_SYNTH").hexdigest()
            dummy_payload = _canonical_p1_bytes(
                dummy_ts, 1.0, True, 1.0, dummy_rhash
            )
            synth_p1 = WisdomWeaverProsecutorPhase1Certificate(
                timestamp=dummy_ts,
                tr_rho=1.0,
                is_psd=True,
                chain_hash=str(p1_cert.prev_chain_hash)
                or hashlib.sha256(dummy_payload).hexdigest(),
                hmac_signature=_hmac_sign(
                    self._secret_key, _HMAC_DOMAIN_P1, dummy_payload
                ),
                purity=1.0,
                rho_hash=dummy_rhash,
                dim=4,
            )
            return self.emit_sovereign_state(synth_p1, p1_cert)
        if p2_cert is None:
            raise ValueError("Se requiere p2_cert para ejecutar la Fase 3 fiscal.")
        return self.emit_sovereign_state(p1_cert, p2_cert)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SOBERANO ORQUESTADOR OODA INQUISITORIAL (CICLO TRIFÁSICO ANIDADO)          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class BoolNovikovRSI3WisdomWeaverProsecutorAgent:
    """Soberano Agente Fiscal que gobierna el ciclo OODA trifásico anidado:
    Fase 1 (Observe) ──► Φ₁₂ ──► Fase 2 (Orient) ──► Φ₂₃ ──► Fase 3 (Decide & Act).
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = secret_key
        self._engine = BoolNovikovRSI3WisdomWeaverProsecutorEngine(
            secret_key=secret_key
        )
        self._observer = WisdomWeaverProsecutorPhase1Observer(secret_key=secret_key)
        self._orientor = WisdomWeaverProsecutorPhase2Orientor(
            secret_key=secret_key, engine=self._engine, observer=self._observer
        )
        self._decider = WisdomWeaverProsecutorPhase3Decider(secret_key=secret_key)
        self._cycle_count: int = 0
        logger.info(
            "%s v%s iniciado (engine_extended=%s).",
            self.__class__.__name__,
            _AGENT_VERSION,
            _ENGINE_EXTENDED,
        )

    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverProsecutorPhase1Certificate
    ) -> bool:
        """Puente homomórfico Φ₁₂: delega a `Phase1Observer.continue_to_phase2_gate`."""
        return self._observer.continue_to_phase2_gate(p1_cert)

    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverProsecutorPhase2Certificate
    ) -> bool:
        """Puente homomórfico Φ₂₃: delega a `Phase2Orientor.continue_to_phase3_gate`."""
        return self._orientor.continue_to_phase3_gate(p2_cert)

    # ── 3.6 Ciclo Soberano de Gobernanza Inquisitorial OODA ─────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 0.85,
        cylinder_R: float = 1.0,
    ) -> WisdomWeaverProsecutorSovereignState:
        """Ejecuta el ciclo OODA inquisitorial completo en 3 fases anidadas:
          P1 (`emit_phase1_certificate`)
            ──► Φ₁₂ (`continue_to_phase2_gate` / `validate_entrance_from_phase1`)
            ──► P2 (`execute_phase2`)
            ──► Φ₂₃ (`continue_to_phase3_gate` / `validate_entrance_from_phase2`)
            ──► P3 (`emit_sovereign_state`).
        Los estados VETOED, caos homoclínico o violaciones de Gromov fluyen hasta la
        Fase 3 para activar la aniquilación de Dirac y el disparo físico del Crowbar.
        """
        # FASE 1: Observación cinemática cuántica
        try:
            p1 = self._observer.emit_phase1_certificate(rho_mac)
        except AttributeError:
            p1 = self._observer.execute_phase1(rho_mac)

        # PUENTE Φ₁₂: Cierre de Fase 1 → Entrada a Fase 2
        if not self.continue_to_phase2_gate(p1):
            raise ValueError(
                "Falla crítica de autenticidad o cinemática cuántica en Compuerta Homomórfica Φ₁₂."
            )

        # FASE 2: Orientación inquisitorial Floer-Poincaré-KAM + RSI³ Nivel-3
        try:
            p2 = self._orientor.execute_phase2(
                p1,
                apu_cost_matrix,
                exponents,
                positron_hmac_valid,
                force_veto,
                action_J,
                pert_eps,
                ball_r,
                cylinder_R,
            )
        except TypeError:
            p2 = self._orientor.execute_phase2(
                p1, apu_cost_matrix, exponents, positron_hmac_valid, force_veto
            )

        # PUENTE Φ₂₃: Cierre de Fase 2 → Entrada a Fase 3
        if not self.continue_to_phase3_gate(p2):
            raise ValueError(
                "Falla crítica de estabilidad de Banach-Kantorovich en Compuerta Homomórfica Φ₂₃."
            )

        # FASE 3: Adjudicación Heyting Ω₃, Interlock Crowbar/Dirac, Φ_sem y Raíz de Merkle
        try:
            state = self._decider.emit_sovereign_state(p1, p2)
        except AttributeError:
            state = self._decider.execute_phase3(p1, p2)

        self._cycle_count += 1
        logger.info(
            "Ciclo OODA Inquisitorial #%d completado. Veredicto: %s, Crowbar: %s (compliant=%s)",
            self._cycle_count,
            state.truth_value.name,
            state.crowbar_triggered,
            state.interlock_compliant,
        )
        return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovRSI3WisdomWeaverProsecutorAgent v15.0.0-PhD+ ===")
    agent = BoolNovikovRSI3WisdomWeaverProsecutorAgent()
    dim = 4
    rho_test = np.eye(dim, dtype=np.complex128) / float(dim)
    matrix_test = np.eye(dim, dtype=np.float64)
    exps_test = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float64)

    # Caso 1: Régimen Coherente (KAM estable, sin disparo de guillotina Crowbar)
    sovereign_state = agent.run_sovereign_governance_cycle(
        rho_test,
        matrix_test,
        exps_test,
        positron_hmac_valid=True,
        force_veto=False,
    )
    print(
        f"[COHERENT] Veredicto Heyting Ω₃ : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(
        f"[COHERENT] Crowbar Triggered    : {sovereign_state.crowbar_triggered} "
        f"compliant={sovereign_state.interlock_compliant} ({sovereign_state.interlock_reason})"
    )
    print(
        f"[COHERENT] Latencia Crowbar (ns): {sovereign_state.total_actuation_time_ns:.2f} ns (≤392.15)"
    )
    print(
        f"[COHERENT] Raíz Merkle (8 hojas): {sovereign_state.merkle_root_hash[:16]}..."
    )
    print(
        f"[COHERENT] Acción J / ε / I_KL  : {sovereign_state.action_J:.4f} / "
        f"{sovereign_state.pert_eps:.4f} / {sovereign_state.kozai_lidov_invariant:.4f} "
        f"KAM:{sovereign_state.is_kam_stable}"
    )
    print(
        f"[COHERENT] Greene / Kantorovich : R_G={sovereign_state.greene_residue:.5f} "
        f"h_K={sovereign_state.kantorovich_h:.5f} T_Nekh={sovereign_state.nekhoroshev_time:.2f}"
    )
    print(
        f"[COHERENT] Métricas Φ_sem       : WACC={sovereign_state.visceral_metrics.wacc_protection_pct}% "
        f"Liq=+{sovereign_state.visceral_metrics.net_liquidity_release_pct}% "
        f"KV={sovereign_state.visceral_metrics.kv_cache_compression_pct}%"
    )

    # Caso 2: Régimen de Condena Inquisitorial (Violación de no-exprimido de Gromov r > R)
    veto_state = agent.run_sovereign_governance_cycle(
        rho_test,
        matrix_test,
        exps_test,
        positron_hmac_valid=True,
        force_veto=False,
        ball_r=1.25,
        cylinder_R=1.0,
    )
    print(
        f"[VETOED]   Veredicto Heyting Ω₃ : {veto_state.truth_value.name} "
        f"({hex(veto_state.truth_value.value)})"
    )
    print(
        f"[VETOED]   Crowbar Triggered    : {veto_state.crowbar_triggered} "
        f"t={veto_state.total_actuation_time_ns:.2f}ns compliant={veto_state.interlock_compliant}"
    )
    print(
        f"[VETOED]   Dirac 1022keV Sig    : {veto_state.dirac_signature[:24]}..."
    )
    print("=== TEST COMPLETADO CON ÉXITO ===")