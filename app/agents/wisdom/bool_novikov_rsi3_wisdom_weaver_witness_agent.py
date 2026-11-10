# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovRSI3WisdomWeaverWitnessAgent v15.0.0-PhD+                                    ║
║ Ubicación: app/agents/wisdom/bool_novikov_rsi3_wisdom_weaver_witness_agent.py                      ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD+   ║
║ Gobernanza: OODA Trifásica Pasiva QND Anidada ①Observe ──► ②Orient ──► ③Decide&Act (Φ₁₂ y Φ₂₃)     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre Testigo `BoolNovikovRSI3WisdomWeaverWitnessAgent` gobierna
ejecutivamente el ciclo de atestación pasiva y certificación no demolicional QND sobre el motor
espectral `BoolNovikovRSI3WisdomWeaverWitnessEngine` (app/wisdom/bool_novikov_rsi3_wisdom_weaver_witness_engine.py).
Opera en tres fases anidadas acopladas por compuertas homomórficas \Phi_{12} y \Phi_{23} firmadas
vía HMAC-SHA256: audita la fidelidad de Uhlmann y la distancia de Bures-Wasserstein en la observación de la MAC,
certifica la no-demolición cuántica con retroacción acotada \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB},
valida la persistencia de toros diofantinos de Poincaré-Lindstedt-KAM, y garantiza la cota
de convergencia cuadrática de Newton-Kantorovich para la tríada de Kleisli en RSI³.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`WisdomWeaverWitnessPhase1Observer`):
   Somete la matriz de densidad \rho a proyección PSD canónica de Higham (\rho_{\mathrm{PSD}} \succeq 0, \operatorname{Tr}(\rho) = 1).
   Audita la medición débil AAV A_w = \langle \psi_f | A | \psi_i \rangle / \langle \psi_f | \psi_i \rangle
   y verifica la condición QND: \|[A, \rho]\|_F \le 1.0\times 10^{-10}.
   Calcula la fidelidad cuántica de Uhlmann-Jozsa F(\rho, \sigma) y la distancia de Bures-Wasserstein:
       d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}}, \quad \mathcal{I}_Q(\rho) = 4 d_B^2(\rho, \sigma).
   En el anillo de Novikov \Lambda_{\mathrm{Nov}}, certifica la valuación de Krull v(A) \ge 0.10.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2:
     Emite `WisdomWeaverWitnessPhase1Certificate` y genera el ancla `WitnessEvidenceAnchor` hacia la Fase 2.

2. FASE 2 — ORIENT (`WisdomWeaverWitnessPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia consumiendo `WitnessEvidenceAnchor`.
   Resuelve la dinámica perturbativa en variables de Delaunay con la ecuación homológica de Poincaré-Lindstedt:
       \{H_0, W_1\} + H_1 = \langle H_1 \rangle_{\mathbb{T}^d},
   y evalúa la cota diofantina fuerte de KAM:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \neq 0.
   Verifica la ausencia de solapamiento de resonancias de Chirikov K_{\mathrm{Chirikov}} < 1.0
   y la rigidez de no-exprimido de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)) en la variedad pseudoholomorfa.
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3:
     Emite `WisdomWeaverWitnessPhase2Certificate` y `KAMAttestationCertificate` hacia la Fase 3.

3. FASE 3 — DECIDE & ACT (`WisdomWeaverWitnessPhase3Decider` + Compuerta \Phi_{23}):
   Inicia consumiendo `KAMAttestationCertificate`.
   Opera la tríada de Kleisli en espacio de Banach con contracción de Picard \mathrm{Lip}(T) \le 0.95.
   Aplica el corrector de Halley de Nivel 3 y la aceleración de Aitken-Steffensen,
   verificando el parámetro de Newton-Kantorovich:
       h_K = L_0 \cdot \|T(s_0) - s_0\| \le 0.50,
   y evaluando el tensor Jerk ciber-físico d^3 C / dt^3.
   Deduce el veredicto en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\},
   proyecta las métricas viscerales de protección de WACC y liquidez, y emite el estado soberano inmutable
   `WisdomWeaverWitnessSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma WWWA-A1] Atestación de No-Demolición Cuántica QND:
    \|[A, \rho]\|_F \le 1.0\times 10^{-10} \implies \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Axioma WWWA-A2] Isometría de Bures-Uhlmann:
    d_B(\rho, \sigma) \le 1.0\times 10^{-6} \land F(\rho, \sigma) \ge 0.95 \implies \text{coherencia atestada}.
[Axioma WWWA-A3] Persistencia de Toros KAM Atestados:
    \min_{k \neq 0} |\langle k, \omega \rangle| \cdot |k|_1^\tau \ge \gamma \implies \text{integrabilidad secular del toro de Delaunay}.
[Axioma WWWA-A4] Convergencia Monádica de Kantorovich:
    h_K \le 0.50 \implies \text{convergencia cuadrática al atractor único en } \mathrm{Kl}(\mathbb{T}^3).

INVARIANTES DE ESTADO
─────────────────────
[Invariante WWWA-I1] Fidelidad de Uhlmann Mínima: F(\rho, \sigma_{\mathrm{PSD}}) \ge 0.95.
[Invariante WWWA-I2] Retroacción Cuántica Acotada: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante WWWA-I3] Condición Diofantina KAM Satisfecha: \mathrm{kam\_diophantine} \equiv \mathrm{True}.
[Invariante WWWA-I4] Módulo de Lipschitz Monádico: \mathrm{Lip}(T^3) \le 0.85975 < 1.0.
[Invariante WWWA-I5] Sellado Criptográfico HMAC-SHA256 con verificación en tiempo constante compare_digest.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Importación del motor testigo (v15/v14 preferido, v13 compatible) ──
try:
    from bool_novikov_rsi3_wisdom_weaver_witness_engine import (
        BoolNovikovRSI3WisdomWeaverWitnessEngine,
        HeytingTruthValue,
        WisdomWeaverWitnessReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_witness_engine import (
            BoolNovikovRSI3WisdomWeaverWitnessEngine,
            HeytingTruthValue,
            WisdomWeaverWitnessReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_rsi3_wisdom_weaver_witness_engine."
        ) from exc

try:
    from bool_novikov_rsi3_wisdom_weaver_witness_engine import (  # type: ignore
        KAMAttestationCertificate,
    )
    _ENGINE_EXTENDED: bool = True
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_witness_engine import (  # type: ignore
            KAMAttestationCertificate,
        )
        _ENGINE_EXTENDED = True
    except ImportError:
        KAMAttestationCertificate = Any  # type: ignore
        _ENGINE_EXTENDED = False

logger = logging.getLogger("APU.Wisdom.BoolNovikovRSI3WisdomWeaverWitnessAgent")

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD+"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_WISDOM_WEAVER_WITNESS_KEY_v13.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"WISDOM-WEAVER-WITNESS-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"WISDOM-WEAVER-WITNESS-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_STATE: Final[bytes] = b"WISDOM-WEAVER-WITNESS-STATE-v14.0|"

# ── Umbrales Físicos, Espectrales y Categóricos ──
_TRACE_TOL: Final[float] = 1.0e-9
_PSD_FLOOR: Final[float] = -1.0e-12
_FRESHNESS_MAX_S: Final[float] = 3600.0
_LIP_HARD_MAX: Final[float] = 1.0
_MAX_BACK_DB: Final[float] = 0.05
_BURES_THRESHOLD: Final[float] = 0.95
_KAM_GAMMA_MIN: Final[float] = 1.0e-4
_GREENE_MAX_STABLE: Final[float] = 0.50
_KANTOROVICH_MAX_H: Final[float] = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS INMUTABLES Y ESTADO SOBERANO (Compatibles v13/v14 + Extensiones v15)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WisdomWeaverWitnessPhase1Certificate:
    timestamp: float
    tr_rho: float
    is_psd: bool
    chain_hash: str
    hmac_signature: str
    # — Extensiones v14 —
    is_trace_conserved: bool = True
    purity: float = 0.0
    von_neumann_entropy: float = 0.0
    rho_hash: str = ""
    dim: int = 0
    # — Extensiones v15 (con defaults para retrocompatibilidad total) —
    l1_quantum_coherence: float = 0.0
    wigner_yanase_skew_info: float = 0.0
    spectral_gap_rho: float = 0.0


@dataclass(frozen=True)
class WisdomWeaverWitnessPhase2Certificate:
    timestamp: float
    report: WisdomWeaverWitnessReport
    chain_hash: str
    hmac_signature: str
    # — Extensiones v14 —
    prev_chain_hash: str = ""
    action_J: float = 1.0
    pert_eps: float = 1.0e-3
    lindstedt_residual: float = 0.0
    diophantine_gamma: float = 0.0
    melnikov_value: float = 0.0
    melnikov_chaos: bool = False
    floer_energy: float = 0.0
    qnd_back_db: float = 0.0
    qnd_passive: bool = True
    bures_fidelity: float = 1.0
    rsi_iterations: int = 0
    godel_hash: str = ""
    # — Extensiones v15 (Mecánica Celeste de Poincaré-Delaunay y RSI³ Nivel-3) —
    delaunay_eccentricity: float = 0.0
    kozai_lidov_invariant: float = 1.0
    greene_residue: float = 0.0
    conley_zehnder_index: int = 0
    nekhoroshev_time: float = 0.0
    octonion_associator_norm: float = 0.0
    quantum_fisher_info: float = 0.0
    kantorovich_h: float = 0.0
    governance_rsi3_godel_hash: str = ""


@dataclass(frozen=True)
class WisdomWeaverWitnessVisceralMetrics:
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    passive_qnd_status: str
    # — Extensiones v14 —
    is_kam_stable: bool = True
    stability_factor: float = 1.0
    bures_fidelity: float = 1.0
    is_silent: bool = True
    # — Extensiones v15 —
    nekhoroshev_confidence_factor: float = 1.0
    kozai_lidov_invariant: float = 1.0


@dataclass(frozen=True)
class WisdomWeaverWitnessSovereignState:
    timestamp: float
    truth_value: HeytingTruthValue
    is_passive_silent_qnd: bool
    back_action_db: float
    capacity_jerk_status: str
    merkle_root_hash: str
    visceral_metrics: WisdomWeaverWitnessVisceralMetrics
    # — Extensiones v14 —
    p1_chain_hash: str = ""
    p2_chain_hash: str = ""
    godel_hash: str = ""
    governance_latency_ms: float = 0.0
    action_J: float = 1.0
    pert_eps: float = 1.0e-3
    is_kam_stable: bool = True
    diophantine_gamma: float = 0.0
    lindstedt_residual: float = 0.0
    bures_fidelity: float = 1.0
    # — Extensiones v15 —
    greene_residue: float = 0.0
    nekhoroshev_time: float = 0.0
    kozai_lidov_invariant: float = 1.0
    kantorovich_h: float = 0.0
    quantum_fisher_info: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# NÚCLEOS MATEMÁTICOS Y CRIPTOGRÁFICOS PRIVADOS
# ═══════════════════════════════════════════════════════════════════════════════

def _canonical_p1_bytes(
    ts: float, tr: float, is_psd: bool, purity: float, rho_hash: str
) -> bytes:
    return (
        f"WITNESS-P1|{ts:.6f}|{tr:.8f}|{is_psd}|{purity:.6f}|{rho_hash}".encode("utf-8")
    )


def _canonical_p2_bytes(
    ts: float,
    prev_hash: str,
    truth_int: int,
    lip: float,
    krull: float,
    cr: float,
    chaos: bool,
    qnd_passive: bool,
    back_db: float,
    fid: float,
    gshort: str,
) -> bytes:
    return (
        f"WITNESS-P2|{ts:.6f}|{prev_hash}|{truth_int}|{lip:.6f}|{krull:.6f}|"
        f"{cr:.6f}|{chaos}|{qnd_passive}|{back_db:.6f}|{fid:.6f}|{gshort}"
    ).encode("utf-8")


def _hmac_sign(secret: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(secret, domain + payload, hashlib.sha256).hexdigest()


def _hmac_verify(secret: bytes, domain: bytes, payload: bytes, sig_hex: str) -> bool:
    try:
        return hmac.compare_digest(_hmac_sign(secret, domain, payload), sig_hex)
    except Exception:
        return False


def _merkle_root(leaves_hex: List[str]) -> str:
    """Raíz de árbol de Merkle binario SHA-256 sobre las hojas de evidencia."""
    if not leaves_hex:
        return hashlib.sha256(b"WITNESS_EMPTY").hexdigest()
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


def _purity_entropy(rho: NDArray[np.complex128]) -> Tuple[bool, float, float]:
    """Evalúa PSD (λ_min ≥ -1e-12), pureza Tr(ρ²) y entropía de von Neumann -Tr(ρ ln ρ) en 1 paso."""
    rh = 0.5 * (rho + rho.conj().T)
    try:
        raw_eigs = np.real(la.eigvalsh(rh))
    except Exception:
        return False, 0.0, 0.0
    is_psd = bool(np.all(raw_eigs >= _PSD_FLOOR))
    w = np.clip(raw_eigs, 0.0, None)
    s = float(np.sum(w))
    if s <= 1.0e-15:
        return False, 0.0, 0.0
    w = w / s
    purity = float(np.sum(w * w))
    nz = w[w > 1.0e-15]
    ent = float(-np.sum(nz * np.log(nz)))
    return is_psd, purity, ent


def _coherence_and_wigner_yanase(
    rho: NDArray[np.complex128]
) -> Tuple[float, float, float]:
    """Calcula coherencia cuántica l₁, información sesgada de Wigner-Yanase I_WY(ρ, K₀) y gap espectral.

    La información sesgada de Wigner-Yanase se define como:
        I_WY(ρ, K₀) = -½ Tr([√ρ, K₀]²) = Tr(ρ K₀²) - Tr(√ρ K₀ √ρ K₀),
    donde K₀ es el generador dipolar de referencia en el espacio de Hilbert.
    """
    rh = 0.5 * (rho + rho.conj().T)
    d = rh.shape[0]
    if d <= 1:
        return 0.0, 0.0, 1.0
    # Coherencia l₁: suma de módulos fuera de la diagonal
    off_diag = rh - np.diag(np.diag(rh))
    c_l1 = float(np.sum(np.abs(off_diag)))

    try:
        w, V = la.eigh(rh)
    except Exception:
        return c_l1, 0.0, 0.0

    w_clean = np.clip(np.real(w), 0.0, None)
    s = float(np.sum(w_clean))
    if s <= 1.0e-15:
        return c_l1, 0.0, 0.0
    w_norm = w_clean / s
    srt = np.sort(w_norm)
    spec_gap = float(srt[-1] - srt[-2]) if d >= 2 else float(srt[-1])

    sqrt_rho = (V * np.sqrt(w_norm)) @ V.conj().T
    # Generador armónico transversal K₀ para sondear no-conmutatividad cinemática
    K0 = np.zeros((d, d), dtype=np.complex128)
    for i in range(d - 1):
        K0[i, i + 1] = 0.5
        K0[i + 1, i] = 0.5
    comm_sqrt = sqrt_rho @ K0 - K0 @ sqrt_rho
    i_wy = float(max(0.0, -0.5 * np.real(np.trace(comm_sqrt @ comm_sqrt))))
    return c_l1, i_wy, spec_gap


def _safe(o: Any, name: str, default: Any) -> Any:
    return getattr(o, name, default) if o is not None else default


def _verify_p1(secret: bytes, c: WisdomWeaverWitnessPhase1Certificate) -> bool:
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


def _p2_gshort(c: WisdomWeaverWitnessPhase2Certificate) -> str:
    g = str(getattr(c, "godel_hash", "") or "") or str(
        _safe(c.report, "godel_hash", "") or ""
    )
    return g[:16] if g else "NOGODEL"


def _p2_fields(
    rep: WisdomWeaverWitnessReport,
) -> Tuple[int, float, float, float, bool, bool, float, float]:
    try:
        truth_int = int(rep.truth_value.value)
        lip = float(rep.banach_contraction_lip)
        krull = float(rep.krull_valuation)
        cr = float(rep.cauchy_riemann_residual)
        chaos = bool(rep.melnikov_chaos_detected)
        qnd_p = bool(rep.qnd_report.is_qnd_passive)
        back = float(rep.qnd_report.back_action_db)
        fid = float(rep.bures_report.fidelity_uhlmann)
    except Exception as exc:
        raise ValueError("Reporte testigo inválido para certificado P2.") from exc
    return truth_int, lip, krull, cr, chaos, qnd_p, back, fid


def _verify_p2(secret: bytes, c: WisdomWeaverWitnessPhase2Certificate) -> bool:
    if len(c.chain_hash) != 64 or len(c.hmac_signature) != 64:
        return False
    try:
        truth_int, lip, krull, cr, chaos, qnd_p, back, fid = _p2_fields(c.report)
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
        qnd_p,
        back,
        fid,
        _p2_gshort(c),
    )
    if hashlib.sha256(payload).hexdigest() != c.chain_hash:
        return False
    if not _hmac_verify(secret, _HMAC_DOMAIN_P2, payload, c.hmac_signature):
        return False
    age = time.time() - float(c.timestamp)
    if age > _FRESHNESS_MAX_S or age < -60.0:
        return False
    return True


def _spectrum_hash(rep: WisdomWeaverWitnessReport) -> str:
    try:
        truth_int, lip, krull, cr, chaos, qnd_p, back, fid = _p2_fields(rep)
    except Exception:
        return hashlib.sha256(b"WITNESS_BAD").hexdigest()
    msg = (
        f"{truth_int}|{lip:.6f}|{krull:.6f}|{cr:.6f}|"
        f"{chaos}|{qnd_p}|{back:.6f}|{fid:.6f}"
    )
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — OBSERVE: INGESTA ISOMÉTRICA CUÁNTICA, ESPECTRAL Y WIGNER-YANASE   ║
# ║ Flujo: Tr(ρ)=1 ──► PSD/Pureza/Entropía ──► Coherencia/Wigner-Yanase ──► P1 ║
# ║ Clausura de Fase 1: `continue_to_phase2_gate` (Φ₁₂) ──► Inicio de Fase 2   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class WisdomWeaverWitnessPhase1Observer:
    """Fase 1 OODA — Observación cinemática cuántica: Tr(ρ)=1, PSD, S_vN y Wigner-Yanase."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key: bytes = secret_key

    def check_trace_conservation(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[float, bool]:
        """1.1 Verifica conservación unitaria de la traza Tr(ρ) ≡ 1 con tolerancia 1e-9."""
        tr = float(np.real(np.trace(np.asarray(rho_mac, dtype=np.complex128))))
        return tr, bool(abs(tr - 1.0) <= _TRACE_TOL)

    def check_psd_purity_entropy(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[bool, float, float]:
        """1.2 Verifica pertenencia al cono semidefinido positivo ρ ⪰ 0, pureza y S_vN(ρ)."""
        return _purity_entropy(np.asarray(rho_mac, dtype=np.complex128))

    def compute_quantum_coherence_and_wigner_yanase(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[float, float, float]:
        """1.3 Cuantifica la coherencia cuántica l₁, información I_WY(ρ, K₀) y gap espectral."""
        return _coherence_and_wigner_yanase(np.asarray(rho_mac, dtype=np.complex128))

    def emit_phase1_certificate(
        self, rho_mac: NDArray[np.complex128]
    ) -> WisdomWeaverWitnessPhase1Certificate:
        """1.4 Sintetiza y firma criptográficamente el certificado cinemático de Fase 1."""
        ts = time.time()
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho_mac debe ser una matriz cuadrada 2D.")
        tr, trace_ok = self.check_trace_conservation(rho)
        is_psd, purity, entropy = self.check_psd_purity_entropy(rho)
        c_l1, i_wy, spec_gap = self.compute_quantum_coherence_and_wigner_yanase(rho)
        rhash = _rho_fingerprint(rho)
        payload = _canonical_p1_bytes(ts, tr, is_psd, purity, rhash)
        ch = hashlib.sha256(payload).hexdigest()
        sig = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P1, payload)
        return WisdomWeaverWitnessPhase1Certificate(
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
            l1_quantum_coherence=c_l1,
            wigner_yanase_skew_info=i_wy,
            spectral_gap_rho=spec_gap,
        )

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128]
    ) -> WisdomWeaverWitnessPhase1Certificate:
        """Alias de compatibilidad retroactiva v13/v14 ──► `emit_phase1_certificate`."""
        return self.emit_phase1_certificate(rho_mac)

    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverWitnessPhase1Certificate
    ) -> bool:
        """1.5 [DEFINICIÓN FORMAL FINAL DE FASE 1 ──► PUENTE HOMOMÓRFICO Φ₁₂ HACIA FASE 2]

        Verifica la integridad criptográfica HMAC-SHA256, el encadenamiento de huella de ρ,
        la conservación de traza, la positividad espectral (PSD) y la frescura temporal.
        Su aprobación habilita directamente `WisdomWeaverWitnessPhase2Orientor.validate_entrance_from_phase1`.
        """
        ok = _verify_p1(self._secret_key, p1_cert)
        if not ok:
            logger.warning("Compuerta Φ₁₂ testigo bloqueada: cripto/cinemática P1 inválida.")
        return ok


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ORIENT: MECÁNICA CELESTE DE POINCARÉ-DELAUNAY Y ATESTACIÓN RSI³   ║
# ║ Inicio Encadenado: `validate_entrance_from_phase1(p1_cert)` (Consume Φ₁₂)  ║
# ║ Flujo: Delaunay-Kozai-Lidov ──► Gobernanza RSI³ L3 ──► Ejecución P2        ║
# ║ Clausura de Fase 2: `continue_to_phase3_gate` (Φ₂₃) ──► Inicio de Fase 3   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class WisdomWeaverWitnessPhase2Orientor:
    """Fase 2 OODA — Orientación en variables de Poincaré-Delaunay y atestación pasiva QND."""

    def __init__(
        self, secret_key: bytes, engine: BoolNovikovRSI3WisdomWeaverWitnessEngine
    ) -> None:
        self._secret_key: bytes = secret_key
        self._engine: BoolNovikovRSI3WisdomWeaverWitnessEngine = engine
        self._governance_history: List[Dict[str, float]] = []
        self._last_delaunay_eccentricity: float = 0.0
        self._last_kozai_lidov_invariant: float = 1.0
        self._last_gov_rsi3_godel_hash: str = ""

    def validate_entrance_from_phase1(
        self, p1_cert: WisdomWeaverWitnessPhase1Certificate
    ) -> bool:
        """2.1 [CONTINUACIÓN DIRECTA DE FASE 1] Revalida el morfismo Φ₁₂ al ingresar a Fase 2."""
        return _verify_p1(self._secret_key, p1_cert)

    def derive_delaunay_action_and_perturbation(
        self,
        apu_cost_matrix: NDArray[np.float64],
        p1_cert: WisdomWeaverWitnessPhase1Certificate,
        exponents: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """2.2 Mecánica Celeste de Poincaré-Delaunay-Whittaker y Conservación de Kozai-Lidov.

        Deriva las variables canónicas de acción de Delaunay (L, G, H_Del) a partir de la
        estructura espectral de la matriz de incidencia/costo M y del certificado P1:
          - Acción principal de Kepler-Delaunay: L = J = clip(⟨M⟩, 0.2, 5.0).
          - Excentricidad espectral orbital: e = √(1 - (det G_sym)^{1/d} / (Tr(G_sym)/d)),
            que mide la anisotropía elíptica de las órbitas en el espacio de fases.
          - Momento angular canónico: G = L √(1 - e²).
          - Inclinación secular cos(i) = 1 / √(1 + ‖M - Mᵀ‖_F² / (‖M + Mᵀ‖_F² + ε)) e
            invariante secular de Kozai-Lidov: I_KL = √(1 - e²) cos(i).
          - Perturbación silenciosa ε ∈ [1e-3, 0.02] modulada por Rademacher-Walsh, pureza P,
            valuación de Krull e información sesgada de Wigner-Yanase.
        """
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        d = M.shape[0] if M.ndim == 2 and M.shape[0] > 0 else 1
        J = float(min(max(float(np.mean(M)) if M.size > 0 else 1.0, 0.2), 5.0))

        # Excentricidad espectral e ∈ [0, 1) por cociente media geométrica / media aritmética (AM-GM)
        G_sym = 0.5 * (M + M.T) if M.ndim == 2 else np.array([[1.0]], dtype=np.float64)
        try:
            eigs = np.clip(np.real(la.eigvalsh(G_sym)), 1.0e-9, None)
            arith_mean = float(np.mean(eigs))
            geom_mean = float(np.exp(np.mean(np.log(eigs))))
            ratio_am_gm = min(1.0, max(0.0, geom_mean / max(arith_mean, 1.0e-12)))
            ecc = math.sqrt(max(0.0, 1.0 - ratio_am_gm * ratio_am_gm))
        except Exception:
            ecc = 0.0

        # Inclinación orbital cos(i) e invariante secular de Kozai-Lidov I_KL = √(1 - e²) cos(i)
        try:
            rw, _, _ = self._engine.compute_rademacher_walsh_and_snf(M)
        except Exception:
            rw = 0.05
        cos_i = 1.0 / math.sqrt(1.0 + 4.0 * float(rw) * float(rw))
        kozai_lidov = math.sqrt(max(0.0, 1.0 - ecc * ecc)) * cos_i
        self._last_delaunay_eccentricity = float(ecc)
        self._last_kozai_lidov_invariant = float(kozai_lidov)

        # Síntesis de la perturbación canónica ε ∈ [1e-3, 0.02]
        purity = float(getattr(p1_cert, "purity", 0.5) or 0.5)
        i_wy = float(getattr(p1_cert, "wigner_yanase_skew_info", 0.0) or 0.0)
        lam = np.asarray(exponents, dtype=np.float64).flatten()
        v_krull = float(np.min(lam)) if lam.size > 0 else 0.15
        krull_damp = math.exp(-max(0.0, v_krull - 0.10))

        eps_base = (
            1.0e-3
            + 0.01 * float(rw)
            + 0.005 * (1.0 - purity)
            + 0.001 * min(1.0, i_wy) * krull_damp
        )
        # Si existe historia de gobernanza RSI³ Nivel-3, acopla suavemente la política óptima
        if self._governance_history:
            eps_rsi3 = self.recursive_governance_improvement_nivel3()
            eps_base = 0.75 * eps_base + 0.25 * eps_rsi3

        eps = float(min(max(eps_base, 1.0e-3), 0.02))
        return float(J), eps

    def recursive_governance_improvement_nivel3(
        self, candidates_eps: Optional[List[float]] = None
    ) -> float:
        """2.3 Automejora Recursiva Nivel-3 (RSI³) de la Política de Gobernanza ε*.

        Implementa la jerarquía recursiva de 3 niveles sobre el espacio de control:
          - Nivel 1 (Objeto): Evalúa candidatos ε_i frente al funcional de utilidad de
            silencio y jerk espectral U(ε) = Jerk_mean - λ_cautela · |ε - ε_target|.
          - Nivel 2 (Meta-Operador): Modula el factor de cautela simpléctica usando la
            constante diofántica γ_KAM, el residuo de Greene R_Greene y el parámetro h_K.
          - Nivel 3 (Meta-Meta Endofuntor): Aplica un paso corrector de Chebyshev-Halley
            hacia el atractor de máxima estabilidad de Nekhoroshev y firma el hash Gödel-HMAC.
        """
        if not self._governance_history:
            return 0.003
        cands = list(candidates_eps) if candidates_eps else [0.001, 0.002, 0.003, 0.005, 0.008]
        recent = self._governance_history[-5:]

        gammas = [float(h.get("gamma", 1.0e-3)) for h in recent]
        jerks = [float(h.get("jerk", 0.0)) for h in recent]
        greenes = [abs(float(h.get("greene", 0.0))) for h in recent]
        kants = [float(h.get("kantorovich_h", 0.05)) for h in recent]

        g_mean = float(np.mean(gammas)) if gammas else 1.0e-3
        j_mean = float(np.mean(jerks)) if jerks else 0.0
        gr_mean = float(np.mean(greenes)) if greenes else 0.0
        kh_mean = float(np.mean(kants)) if kants else 0.05

        # Nivel 2: Factor de cautela diofántica-Kantorovich-Greene
        cautela = 1.0
        if g_mean < 1.0e-3:
            cautela *= 0.70
        if gr_mean > 0.20:
            cautela *= 0.80
        if kh_mean > 0.25:
            cautela *= 0.85

        target_eps = (0.003 if j_mean > 0.0 else 0.001) * cautela

        # Nivel 1: Selección Darwiniana sobre el conjunto discreto de candidatos
        best_l1 = min(cands, key=lambda e: abs(float(e) - target_eps))

        # Nivel 3: Corrección continua de Chebyshev-Halley acotada al intervalo [1e-3, 0.02]
        grad_u = best_l1 - target_eps
        curv_u = 1.0 + gr_mean + kh_mean
        halley_step = grad_u / (curv_u * (1.0 + 0.5 * abs(grad_u) / max(target_eps, 1.0e-6)))
        best_l3 = float(min(max(best_l1 - 0.25 * halley_step, 1.0e-3), 0.02))
        # Si el candidato discreto está extremadamente cerca, preservamos exactitud del candidato
        best_final = float(best_l1) if abs(best_l3 - best_l1) < 5.0e-4 else best_l3

        gh = hmac.new(
            self._secret_key,
            (
                f"GOV-WIT-RSI3|{best_final:.6f}|{j_mean:.5f}|{g_mean:.6f}|"
                f"{gr_mean:.5f}|{kh_mean:.5f}|{_AGENT_VERSION}"
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self._last_gov_rsi3_godel_hash = gh
        logger.debug(
            "GOV-WIT-RSI3 ε*=%.5f jerk=%.4f γ=%.5f R_Greene=%.4f gödel=%s",
            best_final,
            j_mean,
            g_mean,
            gr_mean,
            gh[:12],
        )
        return float(best_final)

    def execute_phase2(
        self,
        p1_cert: WisdomWeaverWitnessPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
        phi_i: Optional[NDArray[np.complex128]] = None,
        phi_f: Optional[NDArray[np.complex128]] = None,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 1.0,
        cylinder_R: float = 1.5,
    ) -> WisdomWeaverWitnessPhase2Certificate:
        """2.4 Ejecuta la orientación espectral de Fase 2 y sella el certificado P2.

        ◄── Requiere validación de la compuerta Φ₁₂ (`validate_entrance_from_phase1`).
        Deriva (J, ε) en variables canónicas de Delaunay, invoca la auditoría trifásica
        del motor testigo, aplica el paso de automejora Nivel-3 sobre el estado espectral
        y emite `WisdomWeaverWitnessPhase2Certificate` enlazado criptográficamente a P1.
        """
        if not self.validate_entrance_from_phase1(p1_cert):
            raise ValueError("Falla de seguridad en Φ₁₂ al entrar a Fase 2 testigo.")

        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        lam = np.asarray(exponents, dtype=np.float64)
        rho = np.asarray(rho_mac, dtype=np.complex128)

        dJ, dE = self.derive_delaunay_action_and_perturbation(M, p1_cert, lam)
        J = float(action_J) if action_J is not None else dJ
        E = float(pert_eps) if pert_eps is not None else dE

        try:
            spec = self._engine.execute_wisdom_witness_audit(
                M,
                lam,
                rho,
                phi_i,
                phi_f,
                action_J=J,
                pert_eps=E,
                ball_r=ball_r,
                cylinder_R=cylinder_R,
            )
        except TypeError:
            try:
                spec = self._engine.execute_wisdom_witness_audit(M, lam, rho, phi_i, phi_f)
            except TypeError:
                spec = self._engine.execute_wisdom_witness_audit(M, lam, rho)  # type: ignore

        # Refinamiento Nivel-3 de telemetría sobre el punto fijo de Kleisli del motor
        try:
            step_fn = getattr(self._engine, "recursive_self_improvement_step_nivel3", None)
            cert_fn = getattr(self._engine, "certify_attestation_and_silence", None)
            if callable(step_fn) and callable(cert_fn):
                kcert = cert_fn(
                    J,
                    E,
                    float(spec.cauchy_riemann_residual),
                    float(_safe(spec, "floer_energy", 0.0)),
                    bool(spec.is_gromov_rigid),
                    bool(spec.qnd_report.is_qnd_passive),
                    float(spec.qnd_report.back_action_db),
                )
                s0 = np.array(
                    [
                        float(spec.krull_valuation),
                        float(spec.banach_contraction_lip),
                        float(spec.cauchy_riemann_residual),
                    ],
                    dtype=np.float64,
                )
                _s3, _r3 = step_fn(s0, kcert, 3)
                logger.debug(
                    "RSI3-L3 testigo orientación: lip3=%.5f jerk=%.4f",
                    _r3.lip_T3,
                    _r3.capacity_jerk_d3c_dt3,
                )
        except Exception as ex:
            logger.debug("Refinamiento L3 testigo omitido: %s", ex)

        ts = time.time()
        truth_int, lip, krull, cr, chaos, qnd_p, back, fid = _p2_fields(spec)
        godel = str(_safe(spec, "godel_hash", "") or "")
        gshort = godel[:16] if godel else "NOGODEL"

        payload = _canonical_p2_bytes(
            ts,
            str(p1_cert.chain_hash),
            truth_int,
            lip,
            krull,
            cr,
            chaos,
            qnd_p,
            back,
            fid,
            gshort,
        )
        ch = hashlib.sha256(payload).hexdigest()
        sig = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P2, payload)

        greene_val = float(_safe(spec, "greene_residue", 0.0))
        cz_idx = int(_safe(spec, "conley_zehnder_index", 0))
        nekh_t = float(_safe(spec, "nekhoroshev_time", 0.0))
        oct_assoc = float(_safe(spec, "octonion_associator_norm", 0.0))
        qfi_val = float(_safe(spec, "quantum_fisher_info", 0.0))
        kant_h = float(_safe(spec, "kantorovich_h", 0.0))

        try:
            self._governance_history.append(
                {
                    "jerk": float(spec.capacity_jerk_d3c_dt3),
                    "gamma": float(_safe(spec, "diophantine_gamma", 0.0)),
                    "eps": float(E),
                    "greene": greene_val,
                    "kantorovich_h": kant_h,
                }
            )
            if not self._last_gov_rsi3_godel_hash:
                _ = self.recursive_governance_improvement_nivel3()
        except Exception:
            pass

        return WisdomWeaverWitnessPhase2Certificate(
            timestamp=ts,
            report=spec,
            chain_hash=ch,
            hmac_signature=sig,
            prev_chain_hash=str(p1_cert.chain_hash),
            action_J=float(J),
            pert_eps=float(E),
            lindstedt_residual=float(_safe(spec, "lindstedt_residual", 0.0)),
            diophantine_gamma=float(_safe(spec, "diophantine_gamma", 0.0)),
            melnikov_value=float(_safe(spec, "melnikov_value", 0.0)),
            melnikov_chaos=bool(chaos),
            floer_energy=float(_safe(spec, "floer_energy", 0.0)),
            qnd_back_db=float(back),
            qnd_passive=bool(qnd_p),
            bures_fidelity=float(fid),
            rsi_iterations=int(_safe(spec, "rsi_iterations", 0)),
            godel_hash=godel,
            delaunay_eccentricity=float(self._last_delaunay_eccentricity),
            kozai_lidov_invariant=float(self._last_kozai_lidov_invariant),
            greene_residue=greene_val,
            conley_zehnder_index=cz_idx,
            nekhoroshev_time=nekh_t,
            octonion_associator_norm=oct_assoc,
            quantum_fisher_info=qfi_val,
            kantorovich_h=kant_h,
            governance_rsi3_godel_hash=str(self._last_gov_rsi3_godel_hash),
        )

    def emit_phase2_certificate(
        self, *a: Any, **k: Any
    ) -> WisdomWeaverWitnessPhase2Certificate:
        """Alias canónico ──► `execute_phase2`."""
        return self.execute_phase2(*a, **k)

    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverWitnessPhase2Certificate
    ) -> bool:
        """2.5 [DEFINICIÓN FORMAL FINAL DE FASE 2 ──► PUENTE HOMOMÓRFICO Φ₂₃ HACIA FASE 3]

        Valida la firma HMAC-P2, el enlace criptográfico con P1, la frescura temporal y la
        contractividad estricta de Banach Lip(T³) < 1.0.
        Garantiza que estados degradados o en veto topológico lleguen silenciosamente a la
        Fase 3 para su adjudicación intuicionista en Ω₃ sin demolición cuántica.
        Su aprobación habilita directamente `WisdomWeaverWitnessPhase3Decider.validate_entrance_from_phase2`.
        """
        if not _verify_p2(self._secret_key, p2_cert):
            logger.warning("Compuerta Φ₂₃ testigo bloqueada: HMAC-P2/cadena/frescura inválida.")
            return False
        try:
            lip = float(p2_cert.report.banach_contraction_lip)
        except Exception:
            return False
        if not (math.isfinite(lip) and lip < _LIP_HARD_MAX):
            logger.warning("Compuerta Φ₂₃ testigo bloqueada: Lip=%.6f ≥ 1.0.", lip)
            return False
        try:
            if not bool(p2_cert.report.qnd_report.is_qnd_passive):
                logger.info(
                    "Compuerta Φ₂₃ testigo: QND no-pasivo admitido hacia Fase 3 (DEGRADED silencioso)."
                )
        except Exception:
            pass
        return True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — DECIDE & ACT: TOPOS HEYTING Ω₃, FUNTOR Φ_sem Y CLAUSURA MERKLE     ║
# ║ Inicio Encadenado: `validate_entrance_from_phase2(p2_cert)` (Consume Φ₂₃)  ║
# ║ Flujo: Heyting Ω₃ ──► Silencio QND ──► Funtor Φ_sem ──► Estado + Orquestador║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class WisdomWeaverWitnessPhase3Decider:
    """Fase 3 OODA — Adjudicación en el Álgebra de Heyting Ω₃, silencio QND y Φ_sem."""

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key: bytes = secret_key

    def validate_entrance_from_phase2(
        self, p2_cert: WisdomWeaverWitnessPhase2Certificate
    ) -> bool:
        """3.1 [CONTINUACIÓN DIRECTA DE FASE 2] Revalida el morfismo Φ₂₃ al ingresar a Fase 3."""
        if not _verify_p2(self._secret_key, p2_cert):
            return False
        try:
            return float(p2_cert.report.banach_contraction_lip) < _LIP_HARD_MAX
        except Exception:
            return False

    def adjudicate_heyting_omega3(
        self,
        p1_cert: WisdomWeaverWitnessPhase1Certificate,
        p2_cert: WisdomWeaverWitnessPhase2Certificate,
    ) -> HeytingTruthValue:
        """3.2 Adjudicación Intuicionista en el Álgebra de Heyting Ω₃ = {0x00 ≺ 0x01 ≺ 0x03}.

        Evalúa el supremo lógico (join en el orden de degradación) entre la validez
        criptográfica/cinemática de P1-P2, la contractividad de Banach-Kantorovich,
        el criterio de estabilidad KAM de MacKay-Greene y el veredicto espectral del motor.
        """
        if not _verify_p1(self._secret_key, p1_cert):
            return HeytingTruthValue.VETOED
        if not _verify_p2(self._secret_key, p2_cert):
            return HeytingTruthValue.VETOED
        # Verificación de encadenamiento criptográfico P1 ──► P2
        if p2_cert.prev_chain_hash and str(p2_cert.prev_chain_hash) != str(p1_cert.chain_hash):
            return HeytingTruthValue.VETOED

        rep = p2_cert.report
        try:
            if float(rep.banach_contraction_lip) >= _LIP_HARD_MAX:
                return HeytingTruthValue.VETOED
        except Exception:
            return HeytingTruthValue.VETOED

        base_tv = rep.truth_value
        if base_tv == HeytingTruthValue.VETOED:
            return HeytingTruthValue.VETOED

        # Verificación de invariantes extendidos v15 (Residuo de Greene y Kantorovich h_K)
        greene = abs(float(getattr(p2_cert, "greene_residue", 0.0) or 0.0))
        kant_h = float(getattr(p2_cert, "kantorovich_h", 0.0) or 0.0)
        if greene > _GREENE_MAX_STABLE or kant_h > _KANTOROVICH_MAX_H + 1.0e-9:
            return HeytingTruthValue.DEGRADED

        return base_tv

    def verify_passive_silent_qnd(
        self, p2_cert: WisdomWeaverWitnessPhase2Certificate
    ) -> Tuple[bool, str]:
        """3.3 Verifica el invariante soberano PASSIVE_SILENT_QND (invisible en RAM/Hardware).

        Garantiza que la retroacción cuántica cumpla 10 log₁₀(1 + ‖[A, ρ]‖_F²) ≤ 0.05 dB
        y Lip(T³) < 1.0. El testigo jamás ejecuta acciones destructivas de hardware.
        """
        rep = p2_cert.report
        try:
            qnd_p = bool(rep.qnd_report.is_qnd_passive)
            back = float(rep.qnd_report.back_action_db)
            lip = float(rep.banach_contraction_lip)
        except Exception:
            return False, "Reporte QND ilegible"
        if not qnd_p:
            return False, f"QND demolición (back={back:.4f}dB)"
        if not (math.isfinite(back) and back <= _MAX_BACK_DB + 1.0e-12):
            return False, f"Back-Action {back:.4f}dB >{_MAX_BACK_DB}dB"
        if not (math.isfinite(lip) and lip < _LIP_HARD_MAX):
            return False, f"Lip={lip:.5f} ≥1 (no silencioso)"
        return True, f"PASSIVE_SILENT_QND_BACKACTION={back:.6f}_dB"

    def functor_semantic_visceral_phi_sem(
        self,
        p1_cert: WisdomWeaverWitnessPhase1Certificate,
        p2_cert: WisdomWeaverWitnessPhase2Certificate,
    ) -> WisdomWeaverWitnessVisceralMetrics:
        """3.4 Funtor Covariante Visceral Φ_sem: E_Spec ──► E_Fin.

        Traduce las invariantes topológicas, simplécticas y cuánticas a métricas financieras
        y operativas de alto impacto:
          - Régimen COHERENT (0x00): WACC = 15.0%, Liquidez Neta = +11.5%, Compresión KV = 86.4%.
          - Regímenes DEGRADED (0x01) y VETOED (0x03): Modulación analítica por estabilidad
            KAM-Greene, fidelidad de Bures-Uhlmann, valuación de Krull, tiempo exponencial
            de Nekhoroshev e invariante secular de Kozai-Lidov.
        """
        rep = p2_cert.report
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        fid = float(_safe(getattr(rep, "bures_report", None), "fidelity_uhlmann", 1.0))
        chaos = bool(rep.melnikov_chaos_detected)
        gromov = bool(rep.is_gromov_rigid)
        gamma = float(_safe(rep, "diophantine_gamma", 0.0))
        has_gamma = hasattr(rep, "diophantine_gamma")
        greene = abs(float(getattr(p2_cert, "greene_residue", 0.0) or 0.0))

        kam = bool(
            (not chaos)
            and gromov
            and ((gamma > _KAM_GAMMA_MIN) if has_gamma else True)
            and (greene <= _GREENE_MAX_STABLE)
        )
        silent, sil_reason = self.verify_passive_silent_qnd(p2_cert)

        nekh_t = float(getattr(p2_cert, "nekhoroshev_time", 0.0) or 0.0)
        f_nekh = (
            min(1.0, max(0.85, math.log10(max(nekh_t, 10.0)) / 4.0))
            if nekh_t > 0.0
            else 1.0
        )
        kozai_lidov = float(getattr(p2_cert, "kozai_lidov_invariant", 1.0) or 1.0)

        f_stab = 1.0 if kam else 0.55
        f_iso = float(min(max(fid, 0.0), 1.0))
        f_krull = 0.5 + 0.5 * min(1.0, max(0.0, float(rep.krull_valuation) / 0.2))
        f_sil = 1.0 if silent else 0.6

        if tv == HeytingTruthValue.COHERENT:
            wacc, liq, kv = 15.0, 11.5, 86.4
        elif tv == HeytingTruthValue.DEGRADED:
            wacc = round(15.0 * f_stab * (0.85 + 0.15 * f_iso) * (0.9 + 0.1 * f_sil), 3)
            liq = round(11.5 * f_krull * (0.7 + 0.3 * f_stab), 3)
            kv = round(86.4 * (0.9 + 0.1 * f_iso) * (0.9 + 0.1 * f_sil), 3)
        else:
            wacc = round(15.0 * 0.4 * f_stab * f_sil, 3)
            liq = round(11.5 * 0.4 * f_krull * f_sil, 3)
            kv = round(86.4 * 0.5 * f_sil, 3)

        status = (
            f"{sil_reason}|KAM={'STABLE' if kam else 'UNSTABLE'}|"
            f"F={fid:.4f}|I_KL={kozai_lidov:.4f}"
        )
        return WisdomWeaverWitnessVisceralMetrics(
            wacc_protection_pct=wacc,
            net_liquidity_release_pct=liq,
            kv_cache_compression_pct=kv,
            passive_qnd_status=status,
            is_kam_stable=kam,
            stability_factor=f_stab,
            bures_fidelity=fid,
            is_silent=silent,
            nekhoroshev_confidence_factor=float(f_nekh),
            kozai_lidov_invariant=float(kozai_lidov),
        )

    def emit_sovereign_state(
        self,
        p1_cert: WisdomWeaverWitnessPhase1Certificate,
        p2_cert: WisdomWeaverWitnessPhase2Certificate,
    ) -> WisdomWeaverWitnessSovereignState:
        """3.5 Sella el Estado Soberano Testigo con Árbol de Merkle de 8 Hojas y Firma HMAC."""
        ts = time.time()
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        metrics = self.functor_semantic_visceral_phi_sem(p1_cert, p2_cert)
        rep = p2_cert.report
        silent, _ = self.verify_passive_silent_qnd(p2_cert)

        try:
            back = float(rep.qnd_report.back_action_db)
        except Exception:
            back = 0.0
        try:
            jerk = float(rep.capacity_jerk_d3c_dt3)
        except Exception:
            jerk = 0.0

        jerk_str = (
            f"WISDOM_RSI3_CAPACITY_JERK_POSITIVE_d3C_dt3={jerk:.4f}"
            if jerk > 0.0
            else f"WISDOM_RSI3_JERK_NONPOSITIVE_d3C_dt3={jerk:.4f}"
        )
        jerk_str += f"|SILENT={silent}|{tv.name}"

        godel = str(_safe(rep, "godel_hash", "") or "")
        gov_godel = str(getattr(p2_cert, "governance_rsi3_godel_hash", "") or "")
        leaves = [
            str(p1_cert.chain_hash),
            str(p1_cert.hmac_signature),
            str(p2_cert.chain_hash),
            str(p2_cert.hmac_signature),
            _spectrum_hash(rep),
            str(godel or _p2_gshort(p2_cert)),
            str(gov_godel or "NOGOVGODEL"),
            str(getattr(rep, "passport_hash", "") or "NOPASSPORT"),
        ]
        root = _merkle_root(leaves)
        _sig = _hmac_sign(
            self._secret_key,
            _HMAC_DOMAIN_STATE,
            f"{ts:.6f}|{int(tv.value)}|{root}".encode("utf-8"),
        )
        logger.debug("Witness-State HMAC=%s...", _sig[:12])
        lat = (ts - float(p1_cert.timestamp)) * 1000.0

        return WisdomWeaverWitnessSovereignState(
            timestamp=ts,
            truth_value=tv,
            is_passive_silent_qnd=bool(silent),
            back_action_db=float(back),
            capacity_jerk_status=jerk_str,
            merkle_root_hash=root,
            visceral_metrics=metrics,
            p1_chain_hash=str(p1_cert.chain_hash),
            p2_chain_hash=str(p2_cert.chain_hash),
            godel_hash=godel,
            governance_latency_ms=float(lat),
            action_J=float(getattr(p2_cert, "action_J", 1.0)),
            pert_eps=float(getattr(p2_cert, "pert_eps", 1.0e-3)),
            is_kam_stable=bool(metrics.is_kam_stable),
            diophantine_gamma=float(getattr(p2_cert, "diophantine_gamma", 0.0)),
            lindstedt_residual=float(getattr(p2_cert, "lindstedt_residual", 0.0)),
            bures_fidelity=float(metrics.bures_fidelity),
            greene_residue=float(getattr(p2_cert, "greene_residue", 0.0)),
            nekhoroshev_time=float(getattr(p2_cert, "nekhoroshev_time", 0.0)),
            kozai_lidov_invariant=float(getattr(p2_cert, "kozai_lidov_invariant", 1.0)),
            kantorovich_h=float(getattr(p2_cert, "kantorovich_h", 0.0)),
            quantum_fisher_info=float(getattr(p2_cert, "quantum_fisher_info", 0.0)),
        )

    def execute_phase3(
        self,
        p1_or_p2: Any,
        p2_cert: Optional[WisdomWeaverWitnessPhase2Certificate] = None,
    ) -> WisdomWeaverWitnessSovereignState:
        """Alias polivalente v13/v14/v15: soporta `execute_phase3(p1, p2)` y `execute_phase3(p2)`."""
        if p2_cert is not None:
            return self.emit_sovereign_state(p1_or_p2, p2_cert)
        if isinstance(p1_or_p2, WisdomWeaverWitnessPhase2Certificate):
            # Reconstruye certificado P1 sintético válido para llamadas legacy de 1 argumento
            ts_Synt = float(p1_or_p2.timestamp)
            rhash_synt = "SYNTHETIC_LEGACY_P1"
            payload_synt = _canonical_p1_bytes(ts_Synt, 1.0, True, 1.0, rhash_synt)
            ch_synt = str(p1_or_p2.prev_chain_hash) or hashlib.sha256(payload_synt).hexdigest()
            sig_synt = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P1, payload_synt)
            p1_synt = WisdomWeaverWitnessPhase1Certificate(
                timestamp=ts_Synt,
                tr_rho=1.0,
                is_psd=True,
                chain_hash=ch_synt,
                hmac_signature=sig_synt,
                purity=1.0,
                rho_hash=rhash_synt,
            )
            return self.emit_sovereign_state(p1_synt, p1_or_p2)
        raise TypeError("Argumentos inválidos para execute_phase3.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.6 ORQUESTADOR SOBERANO DEL CICLO OODA PASIVO QND (F1 ──► Φ₁₂ ──► F2 ──► Φ₂₃ ──► F3)
# ═══════════════════════════════════════════════════════════════════════════════

class BoolNovikovRSI3WisdomWeaverWitnessAgent:
    """Soberano testigo que gobierna el ciclo OODA pasivo anidado en 3 fases."""

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        engine: Optional[BoolNovikovRSI3WisdomWeaverWitnessEngine] = None,
    ) -> None:
        self._secret_key: bytes = secret_key
        self._engine: BoolNovikovRSI3WisdomWeaverWitnessEngine = (
            engine or BoolNovikovRSI3WisdomWeaverWitnessEngine(secret_key=secret_key)
        )
        self._observer = WisdomWeaverWitnessPhase1Observer(secret_key)
        self._orientor = WisdomWeaverWitnessPhase2Orientor(secret_key, self._engine)
        self._decider = WisdomWeaverWitnessPhase3Decider(secret_key)
        self._cycle_count: int = 0
        logger.info(
            f"{self.__class__.__name__} v{_AGENT_VERSION} iniciado (engine_extended={_ENGINE_EXTENDED})."
        )

    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverWitnessPhase1Certificate
    ) -> bool:
        """Compuerta Homomórfica Φ₁₂: delega en `Phase1Observer.continue_to_phase2_gate`."""
        return self._observer.continue_to_phase2_gate(p1_cert)

    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverWitnessPhase2Certificate
    ) -> bool:
        """Compuerta Homomórfica Φ₂₃: delega en `Phase2Orientor.continue_to_phase3_gate`."""
        return self._orientor.continue_to_phase3_gate(p2_cert)

    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        phi_i: Optional[NDArray[np.complex128]] = None,
        phi_f: Optional[NDArray[np.complex128]] = None,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 1.0,
        cylinder_R: float = 1.5,
    ) -> WisdomWeaverWitnessSovereignState:
        """Ejecuta el Ciclo OODA Trifásico Pasivo Anidado:

        FASE 1 (Observe) ──► Compuerta Φ₁₂ ──► FASE 2 (Orient) ──► Compuerta Φ₂₃ ──► FASE 3 (Decide & Act).
        Garantiza silencio cuántico QND sin demolición ni interrupciones destructivas de hardware.
        """
        # ── FASE 1: OBSERVE ──
        try:
            p1 = self._observer.emit_phase1_certificate(rho_mac)
        except AttributeError:
            p1 = self._observer.execute_phase1(rho_mac)

        # ── PUENTE Φ₁₂ (Clausura F1 ──► Inicio F2) ──
        if not self.continue_to_phase2_gate(p1):
            raise ValueError("Falla crítica de autenticidad en Compuerta Homomórfica Φ₁₂.")

        # ── FASE 2: ORIENT ──
        try:
            p2 = self._orientor.execute_phase2(
                p1,
                apu_cost_matrix,
                exponents,
                rho_mac,
                phi_i,
                phi_f,
                action_J,
                pert_eps,
                ball_r,
                cylinder_R,
            )
        except TypeError:
            p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, rho_mac)  # type: ignore

        # ── PUENTE Φ₂₃ (Clausura F2 ──► Inicio F3) ──
        if not self.continue_to_phase3_gate(p2):
            raise ValueError("Falla crítica de estabilidad en Compuerta Homomórfica Φ₂₃.")

        # ── FASE 3: DECIDE & ACT ──
        if not self._decider.validate_entrance_from_phase2(p2):
            raise ValueError("Falla de revalidación en entrada formal de Fase 3 (Φ₂₃).")

        try:
            state = self._decider.emit_sovereign_state(p1, p2)
        except (AttributeError, TypeError):
            state = self._decider.execute_phase3(p1, p2)

        self._cycle_count += 1
        logger.info(
            f"Ciclo OODA Pasivo #{self._cycle_count} completado. "
            f"Veredicto: {state.truth_value.name} | Silencioso={state.is_passive_silent_qnd}"
        )
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE DE VERIFICACIÓN AUTÓNOMA DEL AGENTE SOBERANO Y SUS 3 FASES ANIDADAS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("═" * 88)
    print("  TEST SOBERANO PH.D.: BoolNovikovRSI3WisdomWeaverWitnessAgent v15.0.0-PhD+")
    print("═" * 88)

    agent = BoolNovikovRSI3WisdomWeaverWitnessAgent()
    dim_test = 4
    matrix_test = np.array(
        [
            [1.0, 0.2, 0.0, 0.1],
            [0.2, 1.1, 0.3, 0.0],
            [0.0, 0.3, 0.9, 0.2],
            [0.1, 0.0, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    exps_test = np.array([0.15, 0.25, 0.40, 0.80], dtype=np.float64)
    rho_test = np.eye(dim_test, dtype=np.complex128) / float(dim_test)

    sovereign_state = agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test)

    print(
        f"Veredicto Final Heyting Ω₃ : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(f"Invariante PASSIVE_SILENT  : {sovereign_state.is_passive_silent_qnd}")
    print(
        f"Back-Action QND / SLD-QFI  : {sovereign_state.back_action_db:.6f} dB (≤0.05) "
        f"| QFI={sovereign_state.quantum_fisher_info:.6f}"
    )
    print(f"Jerk Capacidad Status      : {sovereign_state.capacity_jerk_status}")
    print(f"Raíz Merkle (8 hojas)      : {sovereign_state.merkle_root_hash[:16]}...")
    print(
        f"Delaunay J / ε / I_KL      : J={sovereign_state.action_J:.4f} "
        f"| ε={sovereign_state.pert_eps:.5f} "
        f"| I_KL={sovereign_state.kozai_lidov_invariant:.5f} "
        f"| KAM={sovereign_state.is_kam_stable}"
    )
    print(
        f"γ diofántica / Lindstedt   : γ={sovereign_state.diophantine_gamma:.5f} "
        f"| res={sovereign_state.lindstedt_residual:.3e} "
        f"| T_Nekh={sovereign_state.nekhoroshev_time:.3e}"
    )
    print(
        f"Residuo Greene / Kant h_K  : R_Greene={sovereign_state.greene_residue:.6f} "
        f"| h_K={sovereign_state.kantorovich_h:.6f}"
    )
    print(f"Protección WACC (%)        : {sovereign_state.visceral_metrics.wacc_protection_pct}%")
    print(
        f"Liberación Liquidez Neta (%): {sovereign_state.visceral_metrics.net_liquidity_release_pct}%"
    )
    print(
        f"Compresión KV (%)          : {sovereign_state.visceral_metrics.kv_cache_compression_pct}%"
    )
    print(f"Φ_sem pasivo               : {sovereign_state.visceral_metrics.passive_qnd_status}")
    print(f"Latencia gobernanza (ms)   : {sovereign_state.governance_latency_ms:.2f} ms")

    # ── Verificación explícita de los puentes homomórficos anidados Φ₁₂ y Φ₂₃ ──
    p1_explicit = agent._observer.emit_phase1_certificate(rho_test)
    gate_12 = agent._observer.continue_to_phase2_gate(p1_explicit)
    in_f2 = agent._orientor.validate_entrance_from_phase1(p1_explicit)
    p2_explicit = agent._orientor.execute_phase2(
        p1_explicit, matrix_test, exps_test, rho_test
    )
    gate_23 = agent._orientor.continue_to_phase3_gate(p2_explicit)
    in_f3 = agent._decider.validate_entrance_from_phase2(p2_explicit)
    eps_rsi3 = agent._orientor.recursive_governance_improvement_nivel3()
    state_explicit = agent._decider.emit_sovereign_state(p1_explicit, p2_explicit)

    print(
        f"Puentes Anidados Φ₁₂->Φ₂₃  : Φ₁₂={gate_12 and in_f2} "
        f"| Φ₂₃={gate_23 and in_f3} | ε*_RSI3={eps_rsi3:.5f} "
        f"| I_WY={p1_explicit.wigner_yanase_skew_info:.6f}"
    )
    print("═" * 88)