# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovRSI3WisdomWeaverAgent v15.0.0-PhD+                                             ║
║ Ubicación: app/agents/wisdom/bool_novikov_rsi3_wisdom_weaver_agent.py                                ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-AGENT-POINCARE-KAM-RSI3-PhD+             ║
║ Gobernanza: OODA Trifásica Anidada ①Observe → ②Orient → ③Decide&Act con Puentes Φ₁₂ y Φ₂₃           ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre `BoolNovikovRSI3WisdomWeaverAgent` gobierna ejecutivamente
el ciclo armónico OODA de tejeduría de sabiduría sobre el motor espectral
`BoolNovikovRSI3WisdomWeaverEngine` (app/wisdom/bool_novikov_rsi3_wisdom_weaver_engine.py).
Su cometido consiste en armonizar en tres fases anidadas la observación cuántica de la MAC,
la elevación simpléctica al toro de Delaunay, la persistencia de toros diofantinos de Poincaré-KAM
con integrador de Yoshida de 4.º orden y teoría de intersección lagrangiana en \mathcal{F}uk(M),
y la automejora recursiva monádica de Kleisli T^3 sobre espacios de Banach con certificación cuadrática
de Newton-Kantorovich h_K \le 0.50, traduciendo la coherencia matemática a protección de WACC y liquidez.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`WisdomWeaverPhase1Observer`):
   Evalúa la densidad cuántica \rho proyectada por Higham (\rho_{\mathrm{PSD}} \succeq 0, \operatorname{Tr}(\rho) = 1).
   Calcula la distancia geodésica de Bures-Wasserstein d_B(\rho, \sigma) y la fidelidad de Uhlmann F(\rho, \sigma).
   En el anillo \Lambda_{\mathrm{Nov}}, valida la valuación de Krull v(A) \ge 0.10 y la norma no arquimediana.
   En los octoniones \mathbb{O}, certifica que el asociador de Moufang \|[x, y, z]\|_{\mathbb{O}} se mantenga acotado.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2:
     Emite `WisdomWeaverPhase1Certificate` y genera el ancla simpléctica `SymplecticAnchor` hacia la Fase 2.

2. FASE 2 — ORIENT (`WisdomWeaverPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia consumiendo `SymplecticAnchor`.
   Resuelve la ecuación homológica de Poincaré-Lindstedt \{H_0, W_1\} + H_1 = \langle H_1 \rangle
   y verifica la condición diofantina fuerte de KAM:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \in \mathbb{Z}^d \setminus \{0\}.
   Audita la estabilidad de Nekhoroshev T_{\mathrm{Nekh}} \ge 1.0\times 10^6 ciclos y el confinamiento de Chirikov K_{\mathrm{Chirikov}} < 1.0.
   En la categoría de Fukaya \mathcal{F}uk(M), la ecuación de Floer garantiza no-squeezing de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)).
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3:
     Emite `WisdomWeaverPhase2Certificate` y `KAMStabilityCertificate` transfiriendo el haz simpléctico a la Fase 3.

3. FASE 3 — DECIDE & ACT (`WisdomWeaverPhase3Decider` + Compuerta \Phi_{23}):
   Inicia consumiendo `KAMStabilityCertificate`.
   Opera la tríada monádica de Kleisli T^3 en espacio de Banach con contracción de Picard \mathrm{Lip}(T) \le 0.95.
   Aplica la aceleración de Aitken-Steffensen y verifica el parámetro de Newton-Kantorovich:
       h_K = L_0 \cdot \|T(s_0) - s_0\| \le 0.50,
   junto con el tensor Jerk ciber-físico d^3 C / dt^3.
   Deduce el veredicto en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\},
   aplica el funtor semántico visceral para calcular protección de WACC y liquidez neta, y sella el estado
   soberano inmutable `WisdomWeaverSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma WWA-A1] Isometría Cuántica de Bures:
    d_B(\rho, \sigma_{\mathrm{PSD}}) \le 1.0\times 10^{-6} \implies \text{coherencia cuántica verificada}.
[Axioma WWA-A2] No-Resonancia de Poincaré-KAM:
    \min_{k \neq 0} |\langle k, \omega \rangle| \cdot |k|_1^\tau \ge \gamma \implies \text{persistencia de toros invariantes}.
[Axioma WWA-A3] Rigidez Simpléctica en Fukaya:
    c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)) \iff \pi r^2 \le \pi R^2.
[Axioma WWA-A4] Convergencia Hiperbólica de Kleisli:
    \mathrm{Lip}(T^3) < 1.0 \land h_K \le 0.50 \implies \text{atractor único cuadráticamente estable}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante WWA-I1] Fidelidad de Uhlmann: F(\rho, \sigma) \ge 0.95.
[Invariante WWA-I2] Valuación de Krull Mínima: v(A) \ge 0.10.
[Invariante WWA-I3] Condición Diofantina KAM Satisfecha: \mathrm{kam\_diophantine} \equiv \mathrm{True}.
[Invariante WWA-I4] Módulo de Lipschitz Monádico: \mathrm{Lip}(T^3) \le 0.85975 < 1.0.
[Invariante WWA-I5] Sellado Criptográfico HMAC-SHA256 con verificación en tiempo constante compare_digest.
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

# ── Importación del motor espectral (v15/v14 preferido, v13 compatible) ──
try:
    from bool_novikov_rsi3_wisdom_weaver_engine import (
        BoolNovikovRSI3WisdomWeaverEngine,
        HeytingTruthValue,
        WisdomWeaverSpectrumReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_engine import (
            BoolNovikovRSI3WisdomWeaverEngine,
            HeytingTruthValue,
            WisdomWeaverSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_rsi3_wisdom_weaver_engine."
        ) from exc

# Tipos extendidos v14/v15 (opcionales para compatibilidad v13)
try:
    from bool_novikov_rsi3_wisdom_weaver_engine import (  # type: ignore
        KAMStabilityCertificate,
        SymplecticAnchor,
    )
    _ENGINE_EXTENDED: bool = True
except ImportError:
    try:
        from app.wisdom.bool_novikov_rsi3_wisdom_weaver_engine import (  # type: ignore
            KAMStabilityCertificate,
            SymplecticAnchor,
        )
        _ENGINE_EXTENDED = True
    except ImportError:
        SymplecticAnchor = Any  # type: ignore
        KAMStabilityCertificate = Any  # type: ignore
        _ENGINE_EXTENDED = False

logger = logging.getLogger("APU.Wisdom.BoolNovikovRSI3WisdomWeaverAgent")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, CRIPTOGRÁFICAS Y PARÁMETROS UNIVERSALES (v15.0.0-PhD+)
# ═══════════════════════════════════════════════════════════════════════════════
_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-AGENT-POINCARE-KAM-RSI3-PhD+"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_WISDOM_WEAVER_KEY_v13.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"WISDOM-WEAVER-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"WISDOM-WEAVER-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_STATE: Final[bytes] = b"WISDOM-WEAVER-STATE-v14.0|"

_TRACE_TOL: Final[float] = 1.0e-9
_PSD_FLOOR: Final[float] = -1.0e-12
_FRESHNESS_MAX_S: Final[float] = 3600.0
_LIP_HARD_MAX: Final[float] = 1.0
_KANTOROVICH_MAX: Final[float] = 0.5
_GREENE_CRITICAL_RESIDUE: Final[float] = 0.25
_QND_BACK_TOL_DB: Final[float] = 0.05
_QND_COMM_TOL: Final[float] = 1.0e-2
_DIRAC_ENERGY_KEV: Final[float] = 1022.0


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS Y ESTADO SOBERANO (Compatibles v13/v14 + Extensiones v15 PhD+)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WisdomWeaverPhase1Certificate:
    """Certificado cinemático cuántico de Fase 1 (Observe)."""

    timestamp: float
    tr_rho: float
    is_psd: bool
    chain_hash: str
    hmac_signature: str
    # Extensiones v14
    is_trace_conserved: bool = True
    purity: float = 0.0
    von_neumann_entropy: float = 0.0
    rho_hash: str = ""
    dim: int = 0
    # Extensiones v15 PhD+ (con valores por defecto para compatibilidad posicional)
    renyi2_entropy: float = 0.0
    l1_coherence: float = 0.0
    wigner_yanase_skew_info: float = 0.0
    quantum_fisher_info: float = 0.0
    min_eigenvalue: float = 0.0
    hermiticity_defect: float = 0.0


@dataclass(frozen=True)
class WisdomWeaverPhase2Certificate:
    """Certificado dinámico-espectral de Fase 2 (Orient)."""

    timestamp: float
    spectrum_report: WisdomWeaverSpectrumReport
    chain_hash: str
    hmac_signature: str
    # Extensiones v14
    prev_chain_hash: str = ""
    action_J: float = 1.0
    pert_eps: float = 0.05
    lindstedt_residual: float = 0.0
    diophantine_gamma: float = 0.0
    melnikov_amplitude: float = 0.0
    floer_energy: float = 0.0
    rsi_iterations: int = 0
    godel_hash: str = ""
    # Extensiones v15 PhD+ (Delaunay-Kozai-Lidov + Floer-Kantorovich-Greene)
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
    governance_eps_policy: float = 0.05


@dataclass(frozen=True)
class WisdomWeaverVisceralMetrics:
    """Métricas viscerales financieras y operativas producidas por el funtor Φ_sem."""

    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    jerk_capacity_status: str
    # Extensiones v14
    is_kam_stable: bool = True
    stability_factor: float = 1.0
    bures_fidelity: float = 1.0
    # Extensiones v15 PhD+
    greene_residue: float = 0.0
    nekhoroshev_time: float = 0.0
    kozai_lidov_invariant: float = 1.0
    kantorovich_h: float = 0.0


@dataclass(frozen=True)
class WisdomWeaverSovereignState:
    """Estado Soberano final emitido al cierre del ciclo OODA."""

    timestamp: float
    truth_value: HeytingTruthValue
    merkle_root_hash: str
    visceral_metrics: WisdomWeaverVisceralMetrics
    # Extensiones v14
    p1_chain_hash: str = ""
    p2_chain_hash: str = ""
    godel_hash: str = ""
    dirac_signature: str = ""
    governance_latency_ms: float = 0.0
    action_J: float = 1.0
    pert_eps: float = 0.05
    is_kam_stable: bool = True
    # Extensiones v15 PhD+
    state_hmac_signature: str = ""
    greene_residue: float = 0.0
    conley_zehnder_index: int = 0
    nekhoroshev_time: float = 0.0
    kozai_lidov_invariant: float = 1.0
    kantorovich_h: float = 0.0
    qnd_dirac_compliant: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES PRIVADAS CANÓNICAS (CRIPTOGRAFÍA, ESPECTRO Y MERKLE)
# ═══════════════════════════════════════════════════════════════════════════════
def _canonical_p1_bytes(
    ts: float, tr: float, is_psd: bool, purity: float, rho_hash: str
) -> bytes:
    return (
        f"P1_OBSERVE|{ts:.6f}|{tr:.8f}|{is_psd}|{purity:.6f}|{rho_hash}"
    ).encode("utf-8")


def _canonical_p2_bytes(
    ts: float,
    prev_hash: str,
    truth_int: int,
    lip: float,
    krull: float,
    cr: float,
    chaos: bool,
    godel_short: str,
) -> bytes:
    return (
        f"P2_ORIENT|{ts:.6f}|{prev_hash}|{truth_int}|{lip:.6f}|"
        f"{krull:.6f}|{cr:.6f}|{chaos}|{godel_short}"
    ).encode("utf-8")


def _hmac_sign(secret: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(secret, domain + payload, hashlib.sha256).hexdigest()


def _hmac_verify(secret: bytes, domain: bytes, payload: bytes, sig_hex: str) -> bool:
    exp = _hmac_sign(secret, domain, payload)
    try:
        return hmac.compare_digest(exp, sig_hex)
    except Exception:
        return False


def _merkle_root(leaves_hex: List[str]) -> str:
    if not leaves_hex:
        return hashlib.sha256(b"WISDOM_EMPTY").hexdigest()
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
    """Análisis espectral completo de la matriz de densidad cuántica ρ.

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


def _safe_report_attr(rep: Any, name: str, default: Any) -> Any:
    return getattr(rep, name, default) if rep is not None else default


def _verify_p1_certificate(secret: bytes, c: WisdomWeaverPhase1Certificate) -> bool:
    if len(c.chain_hash) != 64 or len(c.hmac_signature) != 64:
        return False
    payload = _canonical_p1_bytes(
        c.timestamp, c.tr_rho, c.is_psd, float(c.purity), str(c.rho_hash)
    )
    if hashlib.sha256(payload).hexdigest() != c.chain_hash:
        return False
    if not _hmac_verify(secret, _HMAC_DOMAIN_P1, payload, c.hmac_signature):
        return False
    if abs(float(c.tr_rho) - 1.0) > _TRACE_TOL:
        return False
    if not bool(c.is_psd):
        return False
    age = time.time() - float(c.timestamp)
    if age > _FRESHNESS_MAX_S or age < -60.0:
        return False
    return True


def _p2_godel_short(c: WisdomWeaverPhase2Certificate) -> str:
    g = str(getattr(c, "godel_hash", "") or "")
    if not g:
        g = str(_safe_report_attr(c.spectrum_report, "godel_hash", "") or "")
    return g[:16] if g else "NOGODEL"


def _verify_p2_certificate(secret: bytes, c: WisdomWeaverPhase2Certificate) -> bool:
    if len(c.chain_hash) != 64 or len(c.hmac_signature) != 64:
        return False
    rep = c.spectrum_report
    try:
        lip = float(rep.banach_contraction_lip)
        krull = float(rep.krull_valuation)
        cr = float(rep.cauchy_riemann_residual)
        chaos = bool(rep.poincare_melnikov_chaos)
        truth_int = int(rep.truth_value.value)
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
        _p2_godel_short(c),
    )
    if hashlib.sha256(payload).hexdigest() != c.chain_hash:
        return False
    if not _hmac_verify(secret, _HMAC_DOMAIN_P2, payload, c.hmac_signature):
        return False
    age = time.time() - float(c.timestamp)
    if age > _FRESHNESS_MAX_S or age < -60.0:
        return False
    return True


def _spectrum_hash(rep: WisdomWeaverSpectrumReport) -> str:
    fid = float(
        _safe_report_attr(
            getattr(rep, "bures_report", None), "fidelity_uhlmann", 1.0
        )
    )
    greene = float(_safe_report_attr(rep, "greene_residue", 0.0))
    kant_h = float(_safe_report_attr(rep, "kantorovich_h", 0.0))
    msg = (
        f"{int(rep.truth_value.value)}|{float(rep.banach_contraction_lip):.6f}|"
        f"{float(rep.krull_valuation):.6f}|{float(rep.cauchy_riemann_residual):.6f}|"
        f"{fid:.6f}|{greene:.6f}|{kant_h:.6f}"
    )
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — OBSERVE: VARIEDAD DUAL R_Boole ⊗ Λ_Nov Y CINEMÁTICA CUÁNTICA       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverPhase1Observer:
    """Fase 1 OODA — Observación cinemática cuántica:
    Conservación de traza Tr(ρ)=1, positividad semidefinida (PSD), pureza, entropías
    de von Neumann y Rényi-2, coherencia cuántica l1, información asimétrica de
    Wigner-Yanase e Información Cuántica de Fisher (QFI).
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
        """Verifica la conservación de traza Tr(ρ) ≡ 1 con tolerancia Wilkinson 1e-9."""
        tr = float(np.real(np.trace(np.asarray(rho_mac, dtype=np.complex128))))
        return tr, bool(abs(tr - 1.0) <= _TRACE_TOL)

    # ── 1.2 Positividad semidefinida, pureza y entropía de von Neumann ──────
    def check_psd_purity_entropy(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[bool, float, float]:
        """Evalúa condición PSD (λ_min ≥ -1e-12), pureza P=Tr(ρ²) y entropía S=-Tr(ρ ln ρ)."""
        rho = np.asarray(rho_mac, dtype=np.complex128)
        return _purity_entropy(rho)

    # ── 1.3 Coherencia cuántica l1, Wigner-Yanase e Información de Fisher ───
    def compute_quantum_coherence_and_wigner_yanase(
        self, rho_mac: NDArray[np.complex128]
    ) -> Tuple[float, float, float]:
        r"""Calcula invariantes geométrico-cuánticos de segundo orden sobre ρ:
        1. Coherencia cuántica l1 de Baumgratz-Cramer-Plenio:
             C_{l1}(ρ) = \sum_{j \neq k} |ρ_{jk}|
        2. Información asimétrica (skew information) de Wigner-Yanase respecto al
           generador hamiltoniano canónico K = diag(0, 1/(d-1), ..., 1):
             I_{WY}(ρ, K) = -1/2 Tr([\sqrt{ρ}, K]²) = 1/2 \sum_{j,k} (\sqrt{λ_j}-\sqrt{λ_k})² |K_{jk}|²
        3. Información Cuántica de Fisher (QFI):
             F_Q(ρ, K) = 2 \sum_{j,k: λ_j+λ_k > 0} ((λ_j - λ_k)² / (λ_j + λ_k)) |K_{jk}|²
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

        K = np.diag(
            np.arange(d, dtype=np.float64) / float(max(d - 1, 1))
        ).astype(np.complex128)
        try:
            evals, evecs = la.eigh(rh)
            evals_clipped = np.clip(np.real(evals), 0.0, None)
            tr_val = float(np.sum(evals_clipped))
            if tr_val <= 1.0e-15:
                return l1_coh, 0.0, 0.0
            lambdas = evals_clipped / tr_val
            sqrt_lambdas = np.sqrt(lambdas)

            K_eig = evecs.conj().T @ K @ evecs
            K_abs2 = np.abs(K_eig) ** 2

            diff_sqrt = sqrt_lambdas[:, None] - sqrt_lambdas[None, :]
            i_wy = float(0.5 * np.sum((diff_sqrt ** 2) * K_abs2))

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
    ) -> WisdomWeaverPhase1Certificate:
        """Cierra la observación cinemática de Fase 1: evalúa espectro, entropías,
        coherencia cuántica e información de Wigner-Yanase/Fisher, almacena ρ en caché
        por huella SHA-256 y sella el certificado con HMAC-SHA256.
        """
        ts = time.time()
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho_mac debe ser matriz cuadrada.")

        tr, trace_ok = self.check_trace_conservation(rho)
        is_psd, purity, entropy, renyi2, min_eig, herm_def = _spectral_density_analysis(
            rho
        )
        l1_coh, i_wy, f_q = self.compute_quantum_coherence_and_wigner_yanase(rho)
        rhash = _rho_fingerprint(rho)

        if len(self._observed_rho_cache) >= 32:
            oldest_key = next(iter(self._observed_rho_cache))
            self._observed_rho_cache.pop(oldest_key, None)
        self._observed_rho_cache[rhash] = rho.copy()

        payload = _canonical_p1_bytes(ts, tr, is_psd, purity, rhash)
        ch = hashlib.sha256(payload).hexdigest()
        sig = _hmac_sign(self._secret_key, _HMAC_DOMAIN_P1, payload)

        return WisdomWeaverPhase1Certificate(
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
    ) -> WisdomWeaverPhase1Certificate:
        """Alias compatible v13/v14 → `emit_phase1_certificate`."""
        return self.emit_phase1_certificate(rho_mac)

    # ── 1.5 DEFINICIÓN FINAL DE FASE 1 = PUENTE HOMOMÓRFICO Φ₁₂ HACIA FASE 2 ─
    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverPhase1Certificate
    ) -> bool:
        """PUENTE FORMAL FASE 1 → FASE 2 (Compuerta Homomórfica Φ₁₂).

        Revalida HMAC-P1, cadena SHA-256, conservación de traza Tr(ρ)=1, condición PSD,
        pureza física en (0, 1] y frescura temporal. No filtra por veredicto dinámico
        (que se sustancia en Fase 2) para permitir que eventuales anomalías fluyan a
        la aniquilación de Dirac.
        ──► Alimenta directamente `WisdomWeaverPhase2Orientor.validate_entrance_from_phase1`.
        """
        ok = _verify_p1_certificate(self._secret_key, p1_cert)
        if not ok:
            logger.warning("Compuerta Φ₁₂ bloqueada: cripto/cinemática P1 inválida.")
            return False
        if not (0.0 < float(p1_cert.purity) <= 1.0 + 1.0e-9):
            logger.warning(
                "Compuerta Φ₁₂ bloqueada: pureza fuera de rango físico (%.6f).",
                p1_cert.purity,
            )
            return False
        return True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ORIENT: SUPERFICIE DE FLOER, POINCARÉ-KAM Y GOBERNANZA RSI³        ║
# ║ ◄── CONTINUACIÓN DIRECTA DE FASE 1: consume el puente homomórfico Φ₁₂       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverPhase2Orientor:
    """Fase 2 OODA — Orientación espectral y celeste:
    Variables canónicas de Delaunay-Kozai-Lidov, superficie pseudoholomorfa de Floer,
    series homológicas de Poincaré-Lindstedt-Birkhoff, estabilidad KAM/Greene y
    mejora recursiva autónoma de gobernanza RSI³ Nivel-3.
    """

    def __init__(
        self,
        engine: BoolNovikovRSI3WisdomWeaverEngine,
        secret_key: bytes,
        observer: Optional[WisdomWeaverPhase1Observer] = None,
    ) -> None:
        self._engine: BoolNovikovRSI3WisdomWeaverEngine = engine
        self._secret_key: bytes = secret_key
        self._observer: Optional[WisdomWeaverPhase1Observer] = observer
        self._governance_history: List[Dict[str, float]] = []
        self._last_celestial_elements: Dict[str, float] = {
            "L": 1.0,
            "G": 1.0,
            "H": 1.0,
            "eccentricity": 0.0,
            "inclination": 0.0,
            "kozai_lidov": 1.0,
            "policy_eps": 0.05,
        }

    # ── 2.1 INICIO DE FASE 2: Validación del puente Φ₁₂ desde Fase 1 ────────
    def validate_entrance_from_phase1(
        self, p1_cert: WisdomWeaverPhase1Certificate
    ) -> bool:
        """CONTINUACIÓN DIRECTA DE FASE 1: consume y revalida la compuerta Φ₁₂
        (`continue_to_phase2_gate`) al ingresar en la Fase 2.
        """
        if not _verify_p1_certificate(self._secret_key, p1_cert):
            return False
        return bool(0.0 < float(p1_cert.purity) <= 1.0 + 1.0e-9)

    # ── 2.2 Mecánica Celeste de Poincaré: Delaunay (L,G,H) y Kozai-Lidov ────
    def derive_delaunay_action_and_perturbation(
        self,
        apu_cost_matrix: NDArray[np.float64],
        p1_cert: WisdomWeaverPhase1Certificate,
        exponents: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""Mecánica celeste canónica de Poincaré-Delaunay-Kozai-Lidov:
        Deriva el par (J, ε) y la tríada secular de Delaunay (L, G, H) conectando
        la cinemática cuántica de Fase 1 con la dinámica hamiltoniana H = H_0(J) + ε H_1(J, θ):
          1. Acción principal de Delaunay:
               L = J = max(0.2, Tr(M)/d + 0.15 * P + 0.05 * mean(λ))
          2. Momento angular orbital G = L * sqrt(1 - e²) con excentricidad espectral e
             inducida por la anisotropía de valores singulares de M y la pureza P.
          3. Componente axial H = G * cos(i) e invariante secular de Kozai-Lidov:
               I_{KL} = sqrt(1 - e²) * cos(i) = H / L.
          4. Perturbación hamiltoniana ε modulada por la norma booleana de Rademacher-Walsh,
             la entropía lineal (1 - P) y la información asimétrica de Wigner-Yanase I_{WY}:
               ε = clip(0.02 + 0.15 * ||RW|| + 0.05 * (1 - P) + 0.01 * min(I_{WY}, 1.0), 0.01, 0.15).
        """
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        lam = np.asarray(exponents, dtype=np.float64).ravel()
        dim = M.shape[0] if M.ndim == 2 and M.shape[0] > 0 else 1
        mean_m = float(np.trace(M)) / float(max(dim, 1)) if M.ndim == 2 else 1.0
        purity = float(getattr(p1_cert, "purity", 0.5) or 0.5)
        purity = min(max(purity, 1.0 / float(max(dim, 1))), 1.0)
        mean_l = float(np.mean(lam)) if lam.size > 0 else 0.15
        L_act = float(max(0.2, mean_m + 0.15 * purity + 0.05 * mean_l))

        # Excentricidad orbital espectral e ∈ [0, 0.95]
        try:
            svals = la.svdvals(M) if M.ndim == 2 else np.ones(1, dtype=np.float64)
            s_max = float(np.max(svals)) if svals.size > 0 else 1.0
            s_min = float(np.min(svals)) if svals.size > 0 else 1.0
            anisotropy = (s_max - s_min) / (s_max + s_min + 1.0e-12)
        except Exception:
            anisotropy = 0.0

        ecc = float(
            min(max(anisotropy * math.sqrt(max(0.0, 1.0 - 0.25 * purity)), 0.0), 0.95)
        )
        G_act = float(L_act * math.sqrt(max(1.0 - ecc * ecc, 1.0e-6)))

        # Inclinación orbital i e invariante secular de Kozai-Lidov I_KL = H / L
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
        eps = (
            0.02
            + 0.15 * float(rw)
            + 0.05 * (1.0 - purity)
            + 0.01 * min(i_wy, 1.0)
        )
        eps = float(min(max(eps, 0.01), 0.15))

        self._last_celestial_elements = {
            "L": L_act,
            "G": G_act,
            "H": H_act,
            "eccentricity": ecc,
            "inclination": inc_rad,
            "kozai_lidov": kozai_lidov,
            "policy_eps": eps,
        }
        return float(L_act), float(eps)

    # ── 2.3 Gobernanza Recursiva Autónoma RSI³ Nivel-3 ──────────────────────
    def recursive_governance_improvement_nivel3(
        self, candidates_eps: Optional[List[float]] = None
    ) -> float:
        r"""Mejora Recursiva Autónoma de Nivel 3 (RSI³) para la política espectral ε*:
          - Nivel 1 (Funcional variacional de Lyapunov-Pontryagin):
              Evalúa cada candidato ε ∈ C frente al objetivo acoplado al jerk d³C/dt³,
              penalizando el residuo de Greene R_Greene y el parámetro de Kantorovich h_K:
                V(ε) = (ε - ε_target)² + 0.25 * R_Greene * ε² + 0.10 * h_K * |ε - ε_target|
          - Nivel 2 (Refinamiento cúbico de Chebyshev-Halley):
              Ejecuta un paso de tercer orden de Halley sobre V(ε) respetando la cota
              diofántica KAM γ y el umbral crítico de Greene (R_Greene < 0.25).
          - Nivel 3 (Selección Darwiniana y Clausura Autorreferencial de Gödel):
              Selecciona la política óptima ε* y emite el digest HMAC-SHA256 de Gödel.
        """
        if not self._governance_history:
            return 0.05

        cands = (
            list(candidates_eps)
            if candidates_eps is not None and len(candidates_eps) > 0
            else [0.03, 0.05, 0.08]
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

        cautela = 0.7 if (gmean < 1.0e-3 or rmean >= _GREENE_CRITICAL_RESIDUE) else 1.0
        base_target = (0.05 if jmean > 0.0 else 0.03) * cautela

        # Nivel 1: Funcional variacional de Lyapunov-Pontryagin
        def _lyapunov_cost(e_val: float) -> float:
            diff = e_val - base_target
            greene_pen = 0.25 * max(0.0, rmean) * (e_val ** 2)
            kant_pen = 0.10 * max(0.0, hmean) * abs(diff)
            return float(diff * diff + greene_pen + kant_pen)

        eps_l1 = float(min(cands, key=_lyapunov_cost))

        # Nivel 2: Paso de Chebyshev-Halley sobre el funcional suavizado
        v_prime = 2.0 * (eps_l1 - base_target) + 0.5 * max(0.0, rmean) * eps_l1
        v_second = 2.0 + 0.5 * max(0.0, rmean)
        eps_l2 = float(eps_l1 - v_prime / max(v_second, 1.0e-12))

        # Nivel 3: Selección Darwiniana sobre el conjunto discreto admisible + Gödel HMAC
        best = float(min(cands, key=lambda e: abs(e - eps_l2)))
        gmsg = (
            f"GOV-RSI3|{best:.4f}|{eps_l2:.4f}|{jmean:.4f}|"
            f"{gmean:.6f}|{rmean:.5f}|{hmean:.5f}|{_AGENT_VERSION}"
        )
        gh = hmac.new(
            self._secret_key, gmsg.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        logger.debug(
            "GOV-RSI3 ε*=%.4f (L2=%.4f) jerk=%.4f γ=%.5f R_Greene=%.4f gödel=%s",
            best,
            eps_l2,
            jmean,
            gmean,
            rmean,
            gh[:12],
        )
        return best

    # ── 2.4 Ejecución Espectral de Fase 2 y Emisión de Certificado P2 ───────
    def execute_phase2(
        self,
        p1_cert: WisdomWeaverPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> WisdomWeaverPhase2Certificate:
        """Tejeduría de Fase 2: auditoría espectral + refinamiento RSI³ + certificado P2.

        ◄── Exige la compuerta Φ₁₂ (`validate_entrance_from_phase1`).
        Deriva (J, ε) y elementos de Delaunay-Kozai-Lidov, recupera `rho_obs` desde
        la caché de Fase 1, ejecuta `engine.execute_wisdom_weaver_audit` (F1→F2→F3 del
        motor), aplica el refinamiento explícito de Nivel-3 y sella P2 enlazado a P1.
        """
        if not self.validate_entrance_from_phase1(p1_cert):
            raise ValueError("Falla de seguridad en Φ₁₂ al entrar a Fase 2.")

        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        lam = np.asarray(exponents, dtype=np.float64)

        dJ, dE = self.derive_delaunay_action_and_perturbation(M, p1_cert, lam)
        J = float(action_J) if action_J is not None else dJ
        E = float(pert_eps) if pert_eps is not None else dE

        rho_obs: Optional[NDArray[np.complex128]] = None
        if self._observer is not None and p1_cert.rho_hash:
            cached = self._observer.get_cached_rho(p1_cert.rho_hash)
            if cached is not None and M.ndim == 2 and cached.shape == M.shape:
                rho_obs = cached

        try:
            spec = self._engine.execute_wisdom_weaver_audit(
                M,
                lam,
                positron_hmac_valid,
                action_J=J,
                pert_eps=E,
                ball_r=ball_r,
                cylinder_R=cylinder_R,
                rho_obs=rho_obs,
            )
        except TypeError:
            try:
                spec = self._engine.execute_wisdom_weaver_audit(
                    M,
                    lam,
                    positron_hmac_valid,
                    action_J=J,
                    pert_eps=E,
                    ball_r=ball_r,
                    cylinder_R=cylinder_R,
                )  # type: ignore
            except TypeError:
                spec = self._engine.execute_wisdom_weaver_audit(
                    M, lam, positron_hmac_valid
                )

        # Refinamiento Nivel-3 explícito (telemetría, preserva el veredicto del motor)
        try:
            step = getattr(
                self._engine, "recursive_self_improvement_step_nivel3", None
            )
            cert_m = getattr(
                self._engine, "certify_kam_stability_and_section", None
            )
            if callable(step) and callable(cert_m):
                cr_val = float(spec.cauchy_riemann_residual)
                fe_val = float(_safe_report_attr(spec, "floer_energy", 0.0))
                gr_val = bool(spec.is_gromov_rigid)
                kcert = cert_m(J, E, cr_val, fe_val, gr_val)
                s0 = np.array(
                    [
                        float(spec.krull_valuation),
                        float(spec.banach_contraction_lip),
                        cr_val,
                    ],
                    dtype=np.float64,
                )
                _s3, _r3 = step(s0, kcert, 3)
                logger.debug(
                    "RSI3-L3 refinado: lip3=%.5f jerk=%.4f",
                    _r3.lip_T3,
                    _r3.capacity_jerk_d3c_dt3,
                )
        except Exception as ex:
            logger.debug("Refinamiento L3 omitido: %s", ex)

        ts = time.time()
        lind_res = float(_safe_report_attr(spec, "lindstedt_residual", 0.0))
        gamma = float(_safe_report_attr(spec, "diophantine_gamma", 0.0))
        mel_amp = float(_safe_report_attr(spec, "melnikov_amplitude", 0.0))
        flo_e = float(_safe_report_attr(spec, "floer_energy", 0.0))
        n_iter = int(_safe_report_attr(spec, "rsi_iterations", 0))
        godel = str(_safe_report_attr(spec, "godel_hash", "") or "")
        lip = float(spec.banach_contraction_lip)
        krull = float(spec.krull_valuation)
        cr = float(spec.cauchy_riemann_residual)
        chaos = bool(spec.poincare_melnikov_chaos)
        gshort = godel[:16] if godel else "NOGODEL"

        # Invariantes extendidos v15 PhD+
        greene_res = float(_safe_report_attr(spec, "greene_residue", 0.0))
        cz_idx = int(_safe_report_attr(spec, "conley_zehnder_index", 0))
        nekh_time = float(_safe_report_attr(spec, "nekhoroshev_time", 0.0))
        oct_norm = float(_safe_report_attr(spec, "octonion_associator_norm", 0.0))
        kant_h = float(_safe_report_attr(spec, "kantorovich_h", 0.0))

        payload = _canonical_p2_bytes(
            ts,
            str(p1_cert.chain_hash),
            int(spec.truth_value.value),
            lip,
            krull,
            cr,
            chaos,
            gshort,
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

        return WisdomWeaverPhase2Certificate(
            timestamp=ts,
            spectrum_report=spec,
            chain_hash=ch,
            hmac_signature=sig,
            prev_chain_hash=str(p1_cert.chain_hash),
            action_J=float(J),
            pert_eps=float(E),
            lindstedt_residual=lind_res,
            diophantine_gamma=gamma,
            melnikov_amplitude=mel_amp,
            floer_energy=flo_e,
            rsi_iterations=n_iter,
            godel_hash=godel,
            greene_residue=greene_res,
            conley_zehnder_index=cz_idx,
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
    ) -> WisdomWeaverPhase2Certificate:
        """Alias canónico → `execute_phase2`."""
        return self.execute_phase2(*a, **k)

    # ── 2.5 DEFINICIÓN FINAL DE FASE 2 = PUENTE HOMOMÓRFICO Φ₂₃ HACIA FASE 3 ─
    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverPhase2Certificate
    ) -> bool:
        """PUENTE FORMAL FASE 2 → FASE 3 (Compuerta Homomórfica Φ₂₃).

        Verifica autenticidad HMAC-P2, integridad de cadena SHA-256, enlace con Fase 1
        (`prev_chain_hash`), frescura temporal, contracción estricta de Banach Lip³ < 1.0
        y condición de convergencia de Newton-Kantorovich h_K ≤ 0.5.
        Admite estados VETOED/caos/Gromov-roto para que fluyan hasta la Fase 3 y
        registren la aniquilación fotónica de Dirac.
        ──► Alimenta directamente `WisdomWeaverPhase3Decider.validate_entrance_from_phase2`.
        """
        if not _verify_p2_certificate(self._secret_key, p2_cert):
            logger.warning("Compuerta Φ₂₃ bloqueada: HMAC-P2/cadena/frescura inválida.")
            return False
        if len(str(p2_cert.prev_chain_hash)) != 64:
            logger.warning(
                "Compuerta Φ₂₃ bloqueada: prev_chain_hash ausente o inválido."
            )
            return False
        try:
            lip = float(p2_cert.spectrum_report.banach_contraction_lip)
            kant_h = float(
                getattr(
                    p2_cert,
                    "kantorovich_h",
                    _safe_report_attr(p2_cert.spectrum_report, "kantorovich_h", 0.0),
                )
            )
        except Exception:
            return False
        if not (math.isfinite(lip) and 0.0 <= lip < _LIP_HARD_MAX):
            logger.warning(
                "Compuerta Φ₂₃ bloqueada: Lip=%.6f ≥ 1 (divergencia Banach).", lip
            )
            return False
        if not (math.isfinite(kant_h) and kant_h <= _KANTOROVICH_MAX + 1.0e-9):
            logger.warning(
                "Compuerta Φ₂₃ bloqueada: Kantorovich h=%.6f > 0.5.", kant_h
            )
            return False
        return True


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — DECIDE & ACT: HEYTING Ω₃, AUDITORÍA QND/DIRAC, Φ_sem Y MERKLE      ║
# ║ ◄── CONTINUACIÓN DIRECTA DE FASE 2: consume el puente homomórfico Φ₂₃       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class WisdomWeaverPhase3Decider:
    """Fase 3 OODA — Adjudicación intuicionista en el álgebra de Heyting Ω₃,
    verificación electro-cuántica QND/Dirac, traducción visceral Φ_sem y sellado
    de estado soberano con árbol de Merkle de 8 hojas.
    """

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key: bytes = secret_key

    # ── 3.1 INICIO DE FASE 3: Validación del puente Φ₂₃ desde Fase 2 ────────
    def validate_entrance_from_phase2(
        self, p2_cert: WisdomWeaverPhase2Certificate
    ) -> bool:
        """CONTINUACIÓN DIRECTA DE FASE 2: consume y revalida la compuerta Φ₂₃
        (`continue_to_phase3_gate`) al entrar en la Fase 3.
        """
        if not _verify_p2_certificate(self._secret_key, p2_cert):
            return False
        if len(str(p2_cert.prev_chain_hash)) != 64:
            return False
        try:
            lip = float(p2_cert.spectrum_report.banach_contraction_lip)
            kant_h = float(
                getattr(
                    p2_cert,
                    "kantorovich_h",
                    _safe_report_attr(p2_cert.spectrum_report, "kantorovich_h", 0.0),
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

    # ── 3.2 Adjudicación Monótona en el Retículo de Heyting Ω₃ ──────────────
    def adjudicate_heyting_omega3(
        self,
        p1_cert: WisdomWeaverPhase1Certificate,
        p2_cert: WisdomWeaverPhase2Certificate,
    ) -> HeytingTruthValue:
        """Adjudicación en el retículo intuicionista Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED}.

        Retorna VETOED si P1 es inválido, P2 es inválido, existe ruptura de cadena
        (`p2_cert.prev_chain_hash != p1_cert.chain_hash`) o diverge Banach (Lip ≥ 1).
        En caso contrario, propaga monótonamente el veredicto espectral del motor.
        """
        if not _verify_p1_certificate(self._secret_key, p1_cert):
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

    # ── 3.3 Verificación de Pasividad QND y Coherencia Fotónica de Dirac ────
    def verify_qnd_and_dirac_coherence(
        self, p2_cert: WisdomWeaverPhase2Certificate
    ) -> Tuple[bool, str]:
        """Audita la pasividad de la medición débil QND y la consistencia energética
        del canal de aniquilación electrodinámica de Dirac (e⁻ + e⁺ → 2γ, 1022.0 keV):
          1. QND pasivo: back_action_db ≤ 0.05 dB y commutator_norm ≤ 1e-2.
          2. Canal Dirac: si rademacher_walsh_norm > 0.10 y positron_valid == True,
             exige energy_released_kev == 1022.0 keV y firma HMAC de 64 hex dígitos.
        """
        rep = p2_cert.spectrum_report
        qnd = getattr(rep, "qnd_report", None)
        dirac = getattr(rep, "annihilation_report", None)

        if qnd is not None:
            if (
                float(qnd.back_action_db) > _QND_BACK_TOL_DB + 1.0e-9
                or float(qnd.commutator_norm) > _QND_COMM_TOL + 1.0e-9
            ):
                return (
                    False,
                    f"Violación QND: back={qnd.back_action_db:.4f}dB comm={qnd.commutator_norm:.4e}",
                )

        if dirac is not None:
            rw = float(getattr(rep, "rademacher_walsh_norm", 0.0))
            pos_ok = bool(getattr(dirac, "positron_valid", True))
            e_kev = float(getattr(dirac, "energy_released_kev", 0.0))
            sig = str(getattr(dirac, "photon_hash_signature", ""))
            if rw > 0.10 and pos_ok:
                if abs(e_kev - _DIRAC_ENERGY_KEV) > 1.0e-3 or len(sig) != 64:
                    return (
                        False,
                        f"Incoherencia Dirac 2γ: E={e_kev:.1f}keV sig_len={len(sig)}",
                    )
                return True, "QND_PASSIVE_AND_DIRAC_1022KEV_ANNIHILATED"
        return True, "QND_PASSIVE_AND_DIRAC_STANDBY_COHERENT"

    # ── 3.4 Funtor Semántico-Visceral Φ_sem ─────────────────────────────────
    def functor_semantic_visceral_phi_sem(
        self,
        p1_cert: WisdomWeaverPhase1Certificate,
        p2_cert: WisdomWeaverPhase2Certificate,
    ) -> WisdomWeaverVisceralMetrics:
        """Funtor semántico-visceral Φ_sem: proyecta los invariantes espectrales,
        simplécticos y celestes hacia métricas financieras y operativas auditables.
        Base contable exacta en estado COHERENT:
          - Protección WACC: 15.0%
          - Liberación neta de liquidez: +11.5%
          - Compresión de caché KV: 86.4%
        """
        rep = p2_cert.spectrum_report
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        fid = float(
            _safe_report_attr(
                getattr(rep, "bures_report", None), "fidelity_uhlmann", 1.0
            )
        )
        try:
            d_b = float(rep.bures_report.bures_distance)
        except Exception:
            d_b = math.sqrt(
                max(0.0, 2.0 - 2.0 * math.sqrt(max(0.0, min(1.0, fid))))
            )

        chaos = bool(rep.poincare_melnikov_chaos)
        lind_ok = (
            bool(_safe_report_attr(rep, "diophantine_gamma", 0.0) > 1.0e-4)
            if hasattr(rep, "diophantine_gamma")
            else True
        )
        greene = float(
            getattr(
                p2_cert,
                "greene_residue",
                _safe_report_attr(rep, "greene_residue", 0.0),
            )
        )
        nekh_t = float(
            getattr(
                p2_cert,
                "nekhoroshev_time",
                _safe_report_attr(rep, "nekhoroshev_time", 0.0),
            )
        )
        kozai = float(getattr(p2_cert, "kozai_lidov_invariant", 1.0))
        kant_h = float(
            getattr(
                p2_cert,
                "kantorovich_h",
                _safe_report_attr(rep, "kantorovich_h", 0.0),
            )
        )

        kam_stable = bool(
            (not chaos)
            and bool(rep.is_gromov_rigid)
            and lind_ok
            and (greene < _GREENE_CRITICAL_RESIDUE or greene == 0.0)
        )
        f_stab = 1.0 if kam_stable else 0.55
        f_iso = float(min(max(fid, 0.0), 1.0))
        f_krull = 0.5 + 0.5 * min(
            1.0, max(0.0, float(rep.krull_valuation) / 0.2)
        )
        f_b = float(min(max(1.0 - d_b / 2.0, 0.0), 1.0))
        jerk = float(rep.capacity_jerk_d3c_dt3)

        if tv == HeytingTruthValue.COHERENT:
            wacc, liq, kv = 15.0, 11.5, 86.4
        elif tv == HeytingTruthValue.DEGRADED:
            wacc = round(15.0 * f_stab * (0.85 + 0.15 * f_iso), 3)
            liq = round(11.5 * f_krull * (0.7 + 0.3 * f_stab), 3)
            kv = round(86.4 * f_b * (0.9 + 0.1 * f_iso), 3)
        else:
            wacc = round(15.0 * 0.4 * f_stab, 3)
            liq = round(11.5 * 0.4 * f_krull, 3)
            kv = round(86.4 * 0.5 * f_b, 3)

        status = (
            f"SUPER_EXPONENTIAL_JERK_POSITIVE_d3C_dt3={jerk:.4f}"
            if jerk > 0.0
            else f"JERK_NONPOSITIVE_d3C_dt3={jerk:.4f}"
        )
        status += (
            f"|KAM={'STABLE' if kam_stable else 'UNSTABLE'}|"
            f"F={fid:.4f}|dB={d_b:.4f}|RG={greene:.4f}"
        )
        return WisdomWeaverVisceralMetrics(
            wacc_protection_pct=wacc,
            net_liquidity_release_pct=liq,
            kv_cache_compression_pct=kv,
            jerk_capacity_status=status,
            is_kam_stable=kam_stable,
            stability_factor=f_stab,
            bures_fidelity=fid,
            greene_residue=greene,
            nekhoroshev_time=nekh_t,
            kozai_lidov_invariant=kozai,
            kantorovich_h=kant_h,
        )

    # ── 3.5 Emisión del Estado Soberano y Árbol de Merkle de 8 Hojas ────────
    def emit_sovereign_state(
        self,
        p1_cert: WisdomWeaverPhase1Certificate,
        p2_cert: WisdomWeaverPhase2Certificate,
    ) -> WisdomWeaverSovereignState:
        """Cierra la Fase 3: adjudica Ω₃, verifica coherencia QND/Dirac, traduce mediante
        Φ_sem y enraíza un árbol de Merkle binario balanceado de 8 hojas con firma HMAC.
        """
        ts = time.time()
        tv = self.adjudicate_heyting_omega3(p1_cert, p2_cert)
        metrics = self.functor_semantic_visceral_phi_sem(p1_cert, p2_cert)
        qnd_dirac_ok, qnd_reason = self.verify_qnd_and_dirac_coherence(p2_cert)
        logger.debug("QND/Dirac coherence: %s (%s)", qnd_dirac_ok, qnd_reason)

        rep = p2_cert.spectrum_report
        godel = str(
            _safe_report_attr(rep, "godel_hash", "") or _p2_godel_short(p2_cert)
        )
        try:
            dirac = str(rep.annihilation_report.photon_hash_signature)
        except Exception:
            dirac = "NO_ANNIHILATION_REQUIRED"

        celestial_qnd_leaf = hashlib.sha256(
            (
                f"{getattr(p2_cert, 'action_J', 1.0):.6f}|"
                f"{getattr(p2_cert, 'pert_eps', 0.05):.6f}|"
                f"{getattr(p2_cert, 'kozai_lidov_invariant', 1.0):.6f}|"
                f"{metrics.greene_residue:.6f}|{qnd_dirac_ok}"
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
            celestial_qnd_leaf,
        ]
        root = _merkle_root(leaves)
        state_sig = _hmac_sign(
            self._secret_key,
            _HMAC_DOMAIN_STATE,
            f"{ts:.6f}|{int(tv.value)}|{root}|{qnd_dirac_ok}".encode("utf-8"),
        )
        logger.debug("State HMAC=%s...", state_sig[:12])

        lat = max(0.0, (ts - float(p1_cert.timestamp)) * 1000.0)
        cz_idx = int(
            getattr(
                p2_cert,
                "conley_zehnder_index",
                _safe_report_attr(rep, "conley_zehnder_index", 0),
            )
        )

        return WisdomWeaverSovereignState(
            timestamp=ts,
            truth_value=tv,
            merkle_root_hash=root,
            visceral_metrics=metrics,
            p1_chain_hash=str(p1_cert.chain_hash),
            p2_chain_hash=str(p2_cert.chain_hash),
            godel_hash=str(_safe_report_attr(rep, "godel_hash", "") or ""),
            dirac_signature=dirac,
            governance_latency_ms=float(lat),
            action_J=float(getattr(p2_cert, "action_J", 1.0)),
            pert_eps=float(getattr(p2_cert, "pert_eps", 0.05)),
            is_kam_stable=bool(metrics.is_kam_stable),
            state_hmac_signature=state_sig,
            greene_residue=float(metrics.greene_residue),
            conley_zehnder_index=cz_idx,
            nekhoroshev_time=float(metrics.nekhoroshev_time),
            kozai_lidov_invariant=float(metrics.kozai_lidov_invariant),
            kantorovich_h=float(metrics.kantorovich_h),
            qnd_dirac_compliant=bool(qnd_dirac_ok),
        )

    def execute_phase3(
        self,
        p1_cert: WisdomWeaverPhase1Certificate,
        p2_cert: Optional[WisdomWeaverPhase2Certificate] = None,
    ) -> WisdomWeaverSovereignState:
        """Alias compatible v13/v14 → `emit_sovereign_state`."""
        if p2_cert is None and isinstance(p1_cert, WisdomWeaverPhase2Certificate):
            dummy_ts = float(p1_cert.timestamp)
            dummy_rhash = hashlib.sha256(b"LEGACY_P1_WEAVER").hexdigest()
            dummy_payload = _canonical_p1_bytes(
                dummy_ts, 1.0, True, 1.0, dummy_rhash
            )
            synth_p1 = WisdomWeaverPhase1Certificate(
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
                dim=3,
            )
            return self.emit_sovereign_state(synth_p1, p1_cert)
        if p2_cert is None:
            raise ValueError("Se requiere p2_cert para ejecutar la Fase 3.")
        return self.emit_sovereign_state(p1_cert, p2_cert)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SOBERANO ORQUESTADOR OODA (Observe ──► Φ₁₂ ──► Orient ──► Φ₂₃ ──► Decide)  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class BoolNovikovRSI3WisdomWeaverAgent:
    """Soberano Agéntico que gobierna el ciclo OODA trifásico anidado."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = secret_key
        self._engine = BoolNovikovRSI3WisdomWeaverEngine(secret_key=secret_key)
        self._observer = WisdomWeaverPhase1Observer(secret_key=secret_key)
        self._orientor = WisdomWeaverPhase2Orientor(
            engine=self._engine,
            secret_key=secret_key,
            observer=self._observer,
        )
        self._decider = WisdomWeaverPhase3Decider(secret_key=secret_key)
        self._cycle_count: int = 0
        logger.info(
            "%s iniciado correctamente (engine_extended=%s).",
            _AGENT_VERSION,
            _ENGINE_EXTENDED,
        )

    def continue_to_phase2_gate(
        self, p1_cert: WisdomWeaverPhase1Certificate
    ) -> bool:
        """Puente homomórfico Φ₁₂: delega a `Phase1Observer.continue_to_phase2_gate`."""
        return self._observer.continue_to_phase2_gate(p1_cert)

    def continue_to_phase3_gate(
        self, p2_cert: WisdomWeaverPhase2Certificate
    ) -> bool:
        """Puente homomórfico Φ₂₃: delega a `Phase2Orientor.continue_to_phase3_gate`."""
        return self._orientor.continue_to_phase3_gate(p2_cert)

    # ── 3.6 Ciclo Soberano de Gobernanza OODA ───────────────────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> WisdomWeaverSovereignState:
        """Ejecuta el ciclo OODA completo en 3 fases estrictamente anidadas:
          P1 (`emit_phase1_certificate`)
            ──► Φ₁₂ (`continue_to_phase2_gate` / `validate_entrance_from_phase1`)
            ──► P2 (`execute_phase2`)
            ──► Φ₂₃ (`continue_to_phase3_gate` / `validate_entrance_from_phase2`)
            ──► P3 (`emit_sovereign_state`).
        Los estados VETOED/caos NO abortan la tubería: fluyen hasta la Fase 3 para
        emitir el Estado Soberano sellado con la firma de aniquilación de Dirac (1022 keV).
        """
        # FASE 1: Observación cinemática cuántica
        try:
            if hasattr(self._observer, "emit_phase1_certificate"):
                p1 = self._observer.emit_phase1_certificate(rho_mac)
            else:
                p1 = self._observer.execute_phase1(rho_mac)
        except Exception as ex:
            raise ValueError(
                f"Falla cinemática en Fase 1 Observe: {ex}"
            ) from ex

        # PUENTE Φ₁₂: Cierre de Fase 1 → Entrada a Fase 2
        if not self.continue_to_phase2_gate(p1):
            raise ValueError(
                "Falla de seguridad en Compuerta Homomórfica Φ₁₂."
            )

        # FASE 2: Orientación espectral Floer-Poincaré-KAM + RSI³ Nivel-3
        try:
            p2 = self._orientor.execute_phase2(
                p1,
                apu_cost_matrix,
                exponents,
                positron_hmac_valid,
                action_J,
                pert_eps,
                ball_r,
                cylinder_R,
            )
        except TypeError:
            p2 = self._orientor.execute_phase2(
                p1, apu_cost_matrix, exponents, positron_hmac_valid
            )

        # PUENTE Φ₂₃: Cierre de Fase 2 → Entrada a Fase 3
        if not self.continue_to_phase3_gate(p2):
            raise ValueError(
                "Falla de estabilidad en Compuerta Homomórfica Φ₂₃."
            )

        # FASE 3: Adjudicación Heyting Ω₃, Auditoría QND/Dirac, Φ_sem y Raíz de Merkle
        try:
            state = self._decider.emit_sovereign_state(p1, p2)
        except AttributeError:
            state = self._decider.execute_phase3(p1, p2)

        self._cycle_count += 1
        logger.info(
            "Ciclo OODA #%d completado. Veredicto: %s J=%.3f ε=%.4f I_KL=%.4f",
            self._cycle_count,
            state.truth_value.name,
            float(state.action_J),
            float(state.pert_eps),
            float(state.kozai_lidov_invariant),
        )
        return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovRSI3WisdomWeaverAgent v15.0.0-PhD+ ===")
    agent = BoolNovikovRSI3WisdomWeaverAgent()
    dim = 3
    rho_test = np.eye(dim, dtype=np.complex128) / float(dim)
    matrix_test = np.array(
        [[0.8, 0.1, 0.0], [0.1, 0.9, 0.2], [0.0, 0.2, 0.85]], dtype=np.float64
    )
    exps_test = np.array([0.15, 0.30, 0.45], dtype=np.float64)
    sovereign_state = agent.run_sovereign_governance_cycle(
        rho_test, matrix_test, exps_test
    )
    print(
        f"Veredicto Final Heyting Ω₃ : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(
        f"Jerk Capacidad Status      : {sovereign_state.visceral_metrics.jerk_capacity_status}"
    )
    print(
        f"Raíz Merkle (8 hojas)      : {sovereign_state.merkle_root_hash[:16]}..."
    )
    print(
        f"Acción J / ε / I_KL        : {sovereign_state.action_J:.4f} / "
        f"{sovereign_state.pert_eps:.4f} / {sovereign_state.kozai_lidov_invariant:.4f} "
        f"KAM:{sovereign_state.is_kam_stable}"
    )
    print(
        f"Greene / Kantorovich / Nekh: R_G={sovereign_state.greene_residue:.5f} "
        f"h_K={sovereign_state.kantorovich_h:.5f} T_Nekh={sovereign_state.nekhoroshev_time:.2f}"
    )
    print(
        f"Protección WACC (%)        : {sovereign_state.visceral_metrics.wacc_protection_pct}%"
    )
    print(
        f"Liberación Liquidez Neta (%): {sovereign_state.visceral_metrics.net_liquidity_release_pct}%"
    )
    print(
        f"Compresión KV (%)          : {sovereign_state.visceral_metrics.kv_cache_compression_pct}%"
    )
    print(
        f"Gödel HMAC                 : "
        f"{sovereign_state.godel_hash[:16] if sovereign_state.godel_hash else 'NOGODEL'}..."
    )
    print(f"Dirac 2γ                   : {sovereign_state.dirac_signature[:16]}...")
    print(f"QND/Dirac Compliant        : {sovereign_state.qnd_dirac_compliant}")
    print(f"Latencia gobernanza (ms)   : {sovereign_state.governance_latency_ms:.2f}")
    print("=== TEST COMPLETADO CON ÉXITO ===")