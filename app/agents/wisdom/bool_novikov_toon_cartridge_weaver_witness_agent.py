# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovToonCartridgeWeaverWitnessAgent v14.0.0-PhD                                    ║
║ Ubicación: app/agents/wisdom/bool_novikov_toon_cartridge_weaver_witness_agent.py                     ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD   ║
║ Gobernanza: OODA Trifásica Pasiva QND (①Observe ➔ Φ₁₂ ➔ ②Orient ➔ Φ₂₃ ➔ ③Decide&Act) con compuertas   ║
║            homomórficas, automejora RSI³ de la política y monitor orbital de Poincaré.               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (herencia; el ÚLTIMO método de la fase k produce el objeto que el
PRIMER método de la fase k+1 consume):

  FASE 1 · OBSERVE   diagnóstico de ρ → sonda de isometría de Bures → huellas → coordenadas de sección
      └─ emit_phase1_certificate()  ──► P1 (HMAC + hash de cadena)
  FASE 2 · ORIENT    └─ continue_to_phase2_gate() (Φ₁₂: autenticidad+frescura; ADMITE VETOED)
      enlace de entradas (anti-TOCTOU) → motor v14 → diagnósticos re-derivables → veredicto de Fase 2
      └─ emit_phase2_certificate()  ──► P2 (encadenado a P1, sella la política vigente)
  FASE 3 · DECIDE&ACT └─ continue_to_phase3_gate() (Φ₂₃: autenticidad + Banach + Back-Action ≤ τₙ)
      funtor Φ_sem monótono → meet de Heyting → órbita de Poincaré → RSI³ de la política → Merkle
      └─ emit_sovereign_state()  ──► Estado soberano (raíz Merkle + firma HMAC)
      └─ (fachada) run_sovereign_governance_cycle()

MARCO MATEMÁTICO
 • Isometría de Bures: d_B(UρU†, UσU†) = d_B(ρ,σ) para U unitaria (Haar, semilla fija); F = (Σ√eig(√ρσ√ρ))².
 • Sección de Poincaré Σ = {frontera de ciclo OODA}; z_n ∈ ℝ⁹ (4 coordenadas de Observe + 5 de Orient).
   Aplicación de retorno P: z_n ↦ z_{n+1}.  Con δ_n = ‖z_{n+1} − z_n‖:  λ ≈ ⟨ln(δ_{n+1}/δ_n)⟩ (Lyapunov).
   Función de Lyapunov V = ‖z‖²; deriva secular = pendiente MCO de V / ⟨V⟩; recurrencia: min_k‖z_n − z_k‖ ≤ ε.
 • RSI³ (Darwin–Gödel) sobre la tolerancia τ de Back-Action, acotada por la constitución
   τ_floor ≤ τ ≤ 0.01 dB (la mutación solo puede ENDURECER):
     Nivel 1: T_η(τ) = τ + η(τ* − τ),  Lip(T_η) = 1 − η < 1,  Lip(T³) = (1 − η)³  (Banach).
     Nivel 2: η ← η + β(η_max − η)      (aceptado solo si el error de seguimiento decrece estrictamente).
     Nivel 3: β ← 1.5β (éxito) | β/2 (rechazo).
   Aceptación Gödel por *replay*: todo ciclo admitido del historial debe seguir admitido bajo τ'.
   Capacidad C_k = 1 − e₀(1−η)^k  ⇒  Δ³C = e₀η³ ≥ 0 (jerk no negativo, forma cerrada).
 • Funtor semántico Φ_sem: Ω₃ → ([0,1], ≤) monótono: VETOED ↦ 0 ≤ DEGRADED ↦ ½ ≤ COHERENT ↦ 1.
 • Merkle: hoja = H(0x00‖label‖0x1f‖value), nodo = H(0x01‖L‖R), nodo impar promovido (sin duplicación).
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Importación del motor espectral (v14) ─────────────────────────────────────────────────────────
try:
    from app.wisdom import bool_novikov_toon_cartridge_weaver_witness_engine as _engine_mod
except ImportError:
    try:
        import bool_novikov_toon_cartridge_weaver_witness_engine as _engine_mod  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError("No se pudo importar bool_novikov_toon_cartridge_weaver_witness_engine.") from exc

BoolNovikovToonCartridgeWeaverWitnessEngine = _engine_mod.BoolNovikovToonCartridgeWeaverWitnessEngine
HeytingTruthValue = _engine_mod.HeytingTruthValue
ToonCartridgeWeaverWitnessSpectrumReport = _engine_mod.ToonCartridgeWeaverWitnessSpectrumReport

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverWitnessAgent")

_AGENT_VERSION: Final[str] = "14.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD"
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_TOON_WITNESS_KEY_v13.0.0"  # heredada: solo desarrollo
_SECRET_ENV: Final[str] = "APU_WITNESS_AGENT_SECRET"
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-WITNESS-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-WITNESS-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_SOV: Final[bytes] = b"TOON-WITNESS-P3-SOVEREIGN-v14.0|"
_HMAC_DOMAIN_POLICY: Final[bytes] = b"TOON-WITNESS-RSI3-POLICY-v14.0|"

# ── Fase 1 ───────────────────────────────────────────────────────────────────────────────────────
_EPS: Final[float] = 1.0e-15
_TRACE_TOL: Final[float] = 1.0e-8
_HERMITICITY_TOL: Final[float] = 1.0e-9
_PSD_TOL: Final[float] = 1.0e-10
_ISOMETRY_TOL: Final[float] = 1.0e-8
_UNITARY_TOL: Final[float] = 1.0e-10
_PROBE_MIXING: Final[float] = 0.05
_PROBE_SEED: Final[int] = 0x5EED
# ── Fase 2 ───────────────────────────────────────────────────────────────────────────────────────
_CERT_MAX_AGE_S: Final[float] = 300.0
_CERT_FUTURE_SKEW_S: Final[float] = 5.0
_LIP_CONSISTENCY_TOL: Final[float] = 1.0e-9
_FIDELITY_SCALE: Final[float] = 0.05          # 1 − F_mín (F ≥ 0.95)
_CR_TOL: Final[float] = 1.0e-4
_MELNIKOV_CLIP: Final[float] = 10.0
# ── Fase 3 ───────────────────────────────────────────────────────────────────────────────────────
_BACKACTION_CONSTITUTIONAL_DB: Final[float] = 0.01   # techo constitucional (no mutable)
_BACKACTION_FLOOR_DB: Final[float] = 1.0e-9          # ≥ equivalente en dB del umbral QND del motor (1e-10 relativo)
_ETA_MIN: Final[float] = 1.0e-3
_ETA_MAX: Final[float] = 0.95
_BETA_MIN: Final[float] = 1.0e-3
_BETA_MAX: Final[float] = 1.0
_RSI_SAFETY_FACTOR: Final[float] = 4.0
_HISTORY_MAXLEN: Final[int] = 512
_ORBIT_WINDOW: Final[int] = 32
_ORBIT_MIN_SECTIONS: Final[int] = 4
_LYAPUNOV_TOL: Final[float] = 0.05
_LYAPUNOV_FLOOR: Final[float] = 1.0e-12
_LYAPUNOV_CAP: Final[float] = 50.0
_SECULAR_DRIFT_TOL: Final[float] = 0.05
_RECURRENCE_EPS: Final[float] = 1.0e-3


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES, POLÍTICA Y COMPUERTAS
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class WitnessGateError(ValueError):
    """Falla genérica de compuerta homomórfica (subclase de ValueError por compatibilidad v13)."""


class WitnessAuthenticityError(WitnessGateError):
    """Manipulación, firma/hash inválido, eslabón roto, enlace de entradas roto o certificado obsoleto."""


class WitnessStabilityError(WitnessGateError):
    """Violación física: cota de Banach o Back-Action fuera de política."""


@dataclass(frozen=True)
class GateDecision:
    """Decisión de una compuerta. `authentic=False` ⇒ manipulación; `admitted=False` ⇒ no continúa."""
    gate: str
    admitted: bool
    authentic: bool
    reasons: Tuple[str, ...]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GovernancePolicy:
    """Política de gobernanza. Mutables por RSI³: backaction_tol_db, eta, beta. El resto es constitucional."""
    backaction_tol_db: float = _BACKACTION_CONSTITUTIONAL_DB
    lip_ceiling: float = 1.0
    certificate_max_age_s: float = _CERT_MAX_AGE_S
    wacc_nominal_pct: float = 15.0
    liquidity_nominal_pct: float = 11.5
    eta: float = 0.25
    beta: float = 0.25

    def __post_init__(self) -> None:
        if not (_BACKACTION_FLOOR_DB <= self.backaction_tol_db <= _BACKACTION_CONSTITUTIONAL_DB):
            raise ValueError("backaction_tol_db fuera del intervalo constitucional.")
        if not (0.0 < self.lip_ceiling <= 1.0):
            raise ValueError("lip_ceiling debe estar en (0, 1].")
        if self.certificate_max_age_s <= 0.0:
            raise ValueError("certificate_max_age_s debe ser positivo.")
        if not (_ETA_MIN <= self.eta <= _ETA_MAX and _BETA_MIN <= self.beta <= _BETA_MAX):
            raise ValueError("eta/beta fuera de dominio.")

    def digest(self) -> str:
        return hashlib.sha256(_encode(dataclasses.asdict(self))).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES PURAS
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _canon(obj: Any) -> Any:
    """Canonicalización determinista (floats a 9 cifras) para digests reproducibles."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canon(dataclasses.asdict(obj))
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return f"{float(obj):.9e}"
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_canon(v) for v in obj.tolist()]
    return str(obj)


def _encode(body: Any) -> bytes:
    return json.dumps(_canon(body), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _g(obj: Any, path: str, default: Any) -> Any:
    """getattr anidado tolerante (fail-closed: el llamador pasa el valor por defecto adverso)."""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _sat(x: float, cap: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        return cap
    return min(max(x, 0.0), cap)


def _meet(*values: Any) -> Any:
    """Ínfimo en Ω₃: la severidad (0x00 < 0x01 < 0x03) crece hacia VETOED ⇒ meet = máximo de severidad."""
    return HeytingTruthValue(max(int(v) for v in values))


def _hermitize(m: NDArray[Any]) -> NDArray[Any]:
    return 0.5 * (m + m.conj().T)


def _project_to_density(rho: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Proyección a {ρ = ρ† ≥ 0, Tr ρ = 1}; devuelve (ρ̂, espectro normalizado)."""
    w, v = np.linalg.eigh(_hermitize(rho))
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s < _EPS:
        raise ValueError("ρ degenerada: traza espectral nula.")
    w = w / s
    return (v * w) @ v.conj().T, w


def _uhlmann_fidelity(r: NDArray[np.complex128], s: NDArray[np.complex128]) -> float:
    """F(ρ,σ) = (Σ√eig(√ρ σ √ρ))² con raíz PSD espectral (sin parte imaginaria espuria)."""
    w, v = np.linalg.eigh(_hermitize(r))
    sr = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T
    ev = np.clip(np.linalg.eigvalsh(_hermitize(sr @ s @ sr)), 0.0, None)
    return float(np.clip(np.sum(np.sqrt(ev)) ** 2, 0.0, 1.0))


def _haar_unitary(d: int, seed: int) -> NDArray[np.complex128]:
    """Unitaria de Haar (Mezzadri: QR con corrección de fases) con semilla fija ⇒ sonda reproducible."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    q, r = np.linalg.qr(g)
    dg = np.diag(r)
    return q * (dg / np.abs(dg))[None, :]


def _fingerprint(arr: NDArray[Any], kind: str) -> str:
    """SHA-256 canónico (little-endian, contiguo, con forma) de un arreglo."""
    a = np.ascontiguousarray(arr, dtype="<c16" if kind == "c" else "<f8")
    h = hashlib.sha256()
    h.update(kind.encode("ascii"))
    h.update(repr(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


# ── Árbol de Merkle (estilo RFC 6962) ─────────────────────────────────────────────────────────────
def _merkle_leaf(label: str, value: str) -> bytes:
    return hashlib.sha256(b"\x00" + label.encode("utf-8") + b"\x1f" + value.encode("utf-8")).digest()


def _merkle_levels(items: Sequence[Tuple[str, str]]) -> List[List[bytes]]:
    if not items:
        raise ValueError("Merkle: conjunto de hojas vacío.")
    levels: List[List[bytes]] = [[_merkle_leaf(lab, val) for lab, val in items]]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = [hashlib.sha256(b"\x01" + cur[i] + cur[i + 1]).digest() for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])  # promoción (sin duplicación: evita colisiones de segunda preimagen)
        levels.append(nxt)
    return levels


def _merkle_root(items: Sequence[Tuple[str, str]]) -> str:
    return _merkle_levels(items)[-1][0].hex()


def _merkle_proof(items: Sequence[Tuple[str, str]], index: int) -> Tuple[Tuple[str, bool], ...]:
    """Prueba de inclusión: pasos (hermano_hex, hermano_a_la_derecha)."""
    levels = _merkle_levels(items)
    if not 0 <= index < len(items):
        raise IndexError("Merkle: índice de hoja fuera de rango.")
    proof: List[Tuple[str, bool]] = []
    idx = index
    for level in levels[:-1]:
        if idx % 2 == 0:
            if idx + 1 < len(level):
                proof.append((level[idx + 1].hex(), True))
        else:
            proof.append((level[idx - 1].hex(), False))
        idx //= 2
    return tuple(proof)


def _merkle_verify(root_hex: str, label: str, value: str, proof: Sequence[Tuple[str, bool]]) -> bool:
    h = _merkle_leaf(label, value)
    try:
        for sib_hex, sib_right in proof:
            sib = bytes.fromhex(sib_hex)
            h = hashlib.sha256(b"\x01" + (h + sib if sib_right else sib + h)).digest()
    except ValueError:
        return False
    return hmac.compare_digest(h.hex(), root_hex)


# ── Funtor semántico Φ_sem: Ω₃ → ([0,1], ≤) ───────────────────────────────────────────────────────
_SEMANTIC_CREDIT: Final[Dict[Any, float]] = {
    HeytingTruthValue.COHERENT: 1.0,
    HeytingTruthValue.DEGRADED: 0.5,
    HeytingTruthValue.VETOED: 0.0,
}


def _functor_is_monotone() -> bool:
    """⊥ = VETOED ≤ DEGRADED ≤ ⊤ = COHERENT  ⇒  crédito no decreciente (preserva el orden)."""
    order = sorted(_SEMANTIC_CREDIT, key=lambda t: -int(t))
    credits = [_SEMANTIC_CREDIT[t] for t in order]
    return all(a <= b for a, b in zip(credits, credits[1:]))


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES Y CERTIFICADOS — FASE 1
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DensityDiagnostics:
    """Diagnóstico completo de la MAC (ρ) tal como llega al observador."""
    dim: int
    trace_real: float
    trace_defect: float
    hermiticity_defect: float
    lambda_min: float
    lambda_max: float
    purity: float
    entropy_nats: float
    effective_rank: int
    sanitization_defect: float
    is_hermitian: bool
    is_psd: bool
    is_normalized: bool


@dataclass(frozen=True)
class BuresIsometryProbe:
    """Invariancia de la métrica de Bures bajo conjugación unitaria (isometría)."""
    mixing_epsilon: float
    fidelity_reference: float
    fidelity_rotated: float
    bures_reference: float
    isometry_defect: float
    unitary_defect: float
    trace_defect_rotated: float
    isometry_ok: bool


@dataclass(frozen=True)
class ToonWeaverWitnessPhase1Certificate:
    """Certificado P1 emitido en la Fase 1 Observe del Testigo TOON."""
    timestamp: float
    tr_rho: float
    is_psd: bool
    chain_hash: str
    hmac_signature: str
    diagnostics: DensityDiagnostics
    isometry_probe: BuresIsometryProbe
    rho_fingerprint: str
    cost_fingerprint: str
    exponents_fingerprint: str
    verdict: Any  # HeytingTruthValue
    orbital_coordinates: Tuple[float, ...]
    version: str


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class ToonWeaverWitnessPhase1Observer:
    """FASE 1 OODA — Observación de la variedad dual: diagnóstico, isometría de Bures, huellas y sello P1."""

    def __init__(self, secret_key: Optional[bytes] = None, policy: Optional[GovernancePolicy] = None) -> None:
        self._secret_key: bytes = self._resolve_secret(secret_key)
        self._policy: GovernancePolicy = policy or GovernancePolicy()

    # ── infraestructura ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_secret(secret_key: Optional[bytes]) -> bytes:
        if secret_key is not None:
            if not isinstance(secret_key, (bytes, bytearray)) or len(secret_key) == 0:
                raise ValueError("secret_key debe ser un bytes no vacío.")
            return bytes(secret_key)
        env = os.environ.get(_SECRET_ENV)
        if env:
            return env.encode("utf-8")
        logger.warning("%s ausente: se usa la clave heredada (solo desarrollo).", _SECRET_ENV)
        return _DEFAULT_SECRET_KEY

    def _seal(self, domain: bytes, body: Dict[str, Any]) -> Tuple[str, str]:
        """(hash de cadena público, firma HMAC con separación de dominio) sobre el cuerpo canónico."""
        raw = _encode(body)
        return hashlib.sha256(raw).hexdigest(), hmac.new(self._secret_key, domain + raw, hashlib.sha256).hexdigest()

    # ── 1.1 validadores ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_density_input(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        arr = np.asarray(rho, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
            raise ValueError("ρ debe ser una matriz cuadrada no vacía.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("ρ contiene valores no finitos.")
        return arr

    @staticmethod
    def _validate_optional_cost(cost: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
        if cost is None:
            return None
        arr = np.asarray(cost, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
            raise ValueError("apu_cost_matrix debe ser cuadrada (≥2) y finita.")
        return arr

    @staticmethod
    def _validate_optional_exponents(exps: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
        if exps is None:
            return None
        arr = np.asarray(exps, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("exponents debe ser un vector no vacío y finito.")
        return arr

    # ── 1.2 diagnóstico de ρ ──────────────────────────────────────────────────────────────────────
    def compute_density_diagnostics(self, rho_mac: NDArray[np.complex128]) -> DensityDiagnostics:
        """Hermiticidad (relativa), PSD sobre la parte hermítica, traza, pureza, entropía de von Neumann."""
        arr = self._validate_density_input(rho_mac)
        herm_def = float(np.linalg.norm(arr - arr.conj().T) / (np.linalg.norm(arr) + _EPS))
        w_raw = np.linalg.eigvalsh(_hermitize(arr))
        trace = complex(np.trace(arr))
        trace_def = float(abs(trace - 1.0))
        rho_hat, p = _project_to_density(arr)
        nz = p[p > _EPS]
        return DensityDiagnostics(
            dim=int(arr.shape[0]),
            trace_real=float(trace.real),
            trace_defect=trace_def,
            hermiticity_defect=herm_def,
            lambda_min=float(w_raw[0]),
            lambda_max=float(w_raw[-1]),
            purity=float(np.sum(p ** 2)),
            entropy_nats=float(-np.sum(nz * np.log(nz))),
            effective_rank=int(np.sum(p > 1.0e-12)),
            sanitization_defect=float(np.linalg.norm(rho_hat - arr) / (np.linalg.norm(arr) + _EPS)),
            is_hermitian=bool(herm_def <= _HERMITICITY_TOL),
            is_psd=bool(w_raw[0] >= -_PSD_TOL),
            is_normalized=bool(trace_def <= _TRACE_TOL),
        )

    # ── 1.3 sonda de isometría de Bures ───────────────────────────────────────────────────────────
    def probe_bures_isometry(self, rho_mac: NDArray[np.complex128]) -> BuresIsometryProbe:
        """Verifica d_B(UρU†, UσU†) = d_B(ρ,σ) con σ = (1−ε)ρ̂ + εI/d y U de Haar: audita la numérica de Uhlmann."""
        arr = self._validate_density_input(rho_mac)
        rho_hat, _ = _project_to_density(arr)
        d = rho_hat.shape[0]
        eps = _PROBE_MIXING
        sigma = (1.0 - eps) * rho_hat + eps * np.eye(d, dtype=np.complex128) / d
        u = _haar_unitary(d, _PROBE_SEED)
        rot_r = u @ rho_hat @ u.conj().T
        rot_s = u @ sigma @ u.conj().T
        f_ref = _uhlmann_fidelity(rho_hat, sigma)
        f_rot = _uhlmann_fidelity(rot_r, rot_s)
        unitary_def = float(np.linalg.norm(u.conj().T @ u - np.eye(d)))
        tr_def = float(abs(np.trace(rot_r) - np.trace(rho_hat)))
        iso_def = float(abs(f_rot - f_ref))
        return BuresIsometryProbe(
            mixing_epsilon=eps,
            fidelity_reference=f_ref,
            fidelity_rotated=f_rot,
            bures_reference=float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f_ref)))),
            isometry_defect=iso_def,
            unitary_defect=unitary_def,
            trace_defect_rotated=tr_def,
            isometry_ok=bool(iso_def <= _ISOMETRY_TOL and unitary_def <= _UNITARY_TOL and tr_def <= _TRACE_TOL),
        )

    # ── 1.4 huellas de entrada (anti-TOCTOU) ──────────────────────────────────────────────────────
    def fingerprint_observation(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: Optional[NDArray[np.float64]] = None,
        exponents: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[str, str, str]:
        """(huella ρ, huella costo, huella exponentes); '-' si el objeto no se observó."""
        rho = self._validate_density_input(rho_mac)
        cost = self._validate_optional_cost(apu_cost_matrix)
        exps = self._validate_optional_exponents(exponents)
        return (
            _fingerprint(rho, "c"),
            "-" if cost is None else _fingerprint(cost, "f"),
            "-" if exps is None else _fingerprint(exps, "f"),
        )

    # ── 1.5 coordenadas de la sección de Poincaré (parte Observe) ─────────────────────────────────
    def observe_orbital_coordinates(self, diag: DensityDiagnostics, probe: BuresIsometryProbe) -> Tuple[float, ...]:
        """z[0:4] = log1p(defecto/tolerancia) de traza, hermiticidad, positividad e isometría (escala-libre)."""
        return (
            math.log1p(_sat(diag.trace_defect / _TRACE_TOL, 1.0e6)),
            math.log1p(_sat(diag.hermiticity_defect / _HERMITICITY_TOL, 1.0e6)),
            math.log1p(_sat(max(0.0, -diag.lambda_min) / _PSD_TOL, 1.0e6)),
            math.log1p(_sat(probe.isometry_defect / _ISOMETRY_TOL, 1.0e6)),
        )

    # ── 1.6 veredicto de Fase 1 ───────────────────────────────────────────────────────────────────
    def classify_phase1(self, diag: DensityDiagnostics, probe: BuresIsometryProbe) -> Any:
        if not (diag.is_hermitian and diag.is_psd and probe.isometry_ok):
            return HeytingTruthValue.VETOED
        if not diag.is_normalized:
            return HeytingTruthValue.DEGRADED  # el motor re-normaliza, pero la entrada no era un estado
        return HeytingTruthValue.COHERENT

    # ── 1.7 cuerpo firmado y verificación de P1 (usado por Φ₁₂) ──────────────────────────────────
    @staticmethod
    def _phase1_body(
        timestamp: float, tr_rho: float, is_psd: bool, diag: DensityDiagnostics, probe: BuresIsometryProbe,
        fps: Tuple[str, str, str], verdict: Any, orbit: Tuple[float, ...], version: str,
    ) -> Dict[str, Any]:
        return {
            "v": version, "ts": f"{timestamp:.6f}", "tr": tr_rho, "psd": is_psd, "diag": diag, "probe": probe,
            "fp": list(fps), "verdict": int(verdict), "orbit": list(orbit),
        }

    def _verify_phase1_certificate(
        self, p1: ToonWeaverWitnessPhase1Certificate, max_age_s: float, now: Optional[float] = None,
    ) -> GateDecision:
        now = time.time() if now is None else now
        auth: List[str] = []
        reasons: List[str] = []
        notes: List[str] = []
        body = self._phase1_body(
            p1.timestamp, p1.tr_rho, p1.is_psd, p1.diagnostics, p1.isometry_probe,
            (p1.rho_fingerprint, p1.cost_fingerprint, p1.exponents_fingerprint), p1.verdict,
            p1.orbital_coordinates, p1.version,
        )
        chain, sig = self._seal(_HMAC_DOMAIN_P1, body)
        if not hmac.compare_digest(chain, p1.chain_hash):
            auth.append("P1: hash de cadena no coincide con el contenido")
        if not hmac.compare_digest(sig, p1.hmac_signature):
            auth.append("P1: firma HMAC inválida")
        if p1.version != _AGENT_VERSION:
            auth.append(f"P1: versión {p1.version!r} ≠ {_AGENT_VERSION!r}")
        age = now - p1.timestamp
        if age > max_age_s or age < -_CERT_FUTURE_SKEW_S:
            reasons.append(f"P1: certificado fuera de ventana de frescura (edad {age:.1f} s)")
        if p1.verdict == HeytingTruthValue.VETOED:
            notes.append("P1 VETOED admitido: atestación pasiva continúa (el veto se propaga por meet)")
        if auth:
            logger.error("Compuerta Φ₁₂: %s", "; ".join(auth))
        return GateDecision("Φ₁₂", not (auth or reasons), not auth, tuple(auth + reasons), tuple(notes))

    # ── 1.8 (ÚLTIMO MÉTODO DE FASE 1) Sello → entrada de Φ₁₂ / Fase 2 ─────────────────────────────
    def emit_phase1_certificate(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: Optional[NDArray[np.float64]] = None,
        exponents: Optional[NDArray[np.float64]] = None,
    ) -> ToonWeaverWitnessPhase1Certificate:
        """Observa (ρ, S, exponentes) y emite P1 firmado: es el objeto que la compuerta Φ₁₂ revalida."""
        diag = self.compute_density_diagnostics(rho_mac)
        probe = self.probe_bures_isometry(rho_mac)
        fps = self.fingerprint_observation(rho_mac, apu_cost_matrix, exponents)
        orbit = self.observe_orbital_coordinates(diag, probe)
        verdict = self.classify_phase1(diag, probe)
        ts = time.time()
        body = self._phase1_body(ts, diag.trace_real, diag.is_psd, diag, probe, fps, verdict, orbit, _AGENT_VERSION)
        chain, sig = self._seal(_HMAC_DOMAIN_P1, body)
        return ToonWeaverWitnessPhase1Certificate(
            timestamp=ts, tr_rho=diag.trace_real, is_psd=diag.is_psd, chain_hash=chain, hmac_signature=sig,
            diagnostics=diag, isometry_probe=probe, rho_fingerprint=fps[0], cost_fingerprint=fps[1],
            exponents_fingerprint=fps[2], verdict=verdict, orbital_coordinates=orbit, version=_AGENT_VERSION,
        )

    execute_phase1 = emit_phase1_certificate  # alias de compatibilidad v13


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES Y CERTIFICADOS — FASE 2
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OrientDiagnostics:
    """Diagnósticos re-derivables del reporte del motor (la compuerta Φ₂₃ los recomputa y compara)."""
    passport_verified: bool
    back_action_db: float
    back_action_tolerance_db: float
    back_action_ok: bool
    qnd_passive: bool
    lip_certified: float
    lip_empirical: float
    lip_consistent: bool
    banach_ok: bool
    fidelity: float
    melnikov_ratio: float
    melnikov_chaos: bool
    kam_stable: bool
    cr_residual: float
    gromov_rigid: bool
    engine_truth: Any  # HeytingTruthValue
    engine_phase_verdicts: Tuple[int, ...]
    toon_tokens: int
    kv_cache_compression_pct: float
    orbital_coordinates: Tuple[float, ...]


@dataclass(frozen=True)
class ToonWeaverWitnessPhase2Certificate:
    """Certificado P2 emitido en la Fase 2 Orient del Testigo TOON."""
    timestamp: float
    spectrum_report: ToonCartridgeWeaverWitnessSpectrumReport
    chain_hash: str
    hmac_signature: str
    parent_chain_hash: str
    diagnostics: OrientDiagnostics
    verdict: Any  # HeytingTruthValue
    policy_digest: str
    version: str


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class ToonWeaverWitnessPhase2Orientor(ToonWeaverWitnessPhase1Observer):
    """FASE 2 OODA — (hereda Fase 1) Φ₁₂ → enlace de entradas → motor v14 → diagnósticos → sello P2."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovToonCartridgeWeaverWitnessEngine] = None,
        policy: Optional[GovernancePolicy] = None,
    ) -> None:
        super().__init__(secret_key, policy)
        self._engine = engine or BoolNovikovToonCartridgeWeaverWitnessEngine(secret_key=self._secret_key)

    # ── 2.0 (PRIMER MÉTODO DE FASE 2) Compuerta Φ₁₂: continuación del certificado P1 ──────────────
    def continue_to_phase2_gate(
        self, p1: ToonWeaverWitnessPhase1Certificate, now: Optional[float] = None,
    ) -> GateDecision:
        """Φ₁₂: revalida hash de cadena + HMAC + versión + frescura de P1. ADMITE veredictos VETOED."""
        return self._verify_phase1_certificate(p1, self._policy.certificate_max_age_s, now)

    # ── 2.1 enlace de entradas (anti-TOCTOU) ──────────────────────────────────────────────────────
    def verify_input_binding(
        self,
        p1: ToonWeaverWitnessPhase1Certificate,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
    ) -> GateDecision:
        """La ρ (y, si fue observado, S y los exponentes) auditada por el motor debe ser la observada en P1."""
        fps = self.fingerprint_observation(rho_mac, apu_cost_matrix, exponents)
        reasons: List[str] = []
        if not hmac.compare_digest(fps[0], p1.rho_fingerprint):
            reasons.append("Enlace: la ρ auditada difiere de la ρ observada en P1")
        if p1.cost_fingerprint != "-" and not hmac.compare_digest(fps[1], p1.cost_fingerprint):
            reasons.append("Enlace: la matriz de costo auditada difiere de la observada en P1")
        if p1.exponents_fingerprint != "-" and not hmac.compare_digest(fps[2], p1.exponents_fingerprint):
            reasons.append("Enlace: los exponentes auditados difieren de los observados en P1")
        return GateDecision("Φ₁₂.binding", not reasons, not reasons, tuple(reasons))

    # ── 2.2 diagnósticos re-derivables del reporte del motor ──────────────────────────────────────
    def derive_orient_diagnostics(self, report: ToonCartridgeWeaverWitnessSpectrumReport) -> OrientDiagnostics:
        """Función determinista (reporte, política) → diagnósticos. Faltantes ⇒ valor adverso (fail-closed)."""
        verify = getattr(self._engine, "verify_passport", None)
        passport_ok = bool(verify(report)) if callable(verify) else False
        tol = self._policy.backaction_tol_db
        ceiling = self._policy.lip_ceiling

        ba = float(_g(report, "qnd_report.back_action_db", math.inf))
        lip_cert = float(_g(report, "banach_contraction_lip", math.inf))
        lip_emp = float(_g(report, "rsi3_report.empirical_lip_t3", math.inf))
        fid = float(_g(report, "bures_report.fidelity_uhlmann", 0.0))
        mel = float(_g(report, "kam_report.melnikov_ratio", math.inf))
        cr = float(_g(report, "pseudoholomorphic_report.cauchy_riemann_residual", math.inf))
        engine_truth = HeytingTruthValue(int(_g(report, "truth_value", HeytingTruthValue.VETOED)))

        coords = (
            math.log1p(_sat(ba / _BACKACTION_CONSTITUTIONAL_DB, 1.0e6)),
            _sat(lip_cert, 2.0),
            _sat(mel, _MELNIKOV_CLIP),
            _sat((1.0 - fid) / _FIDELITY_SCALE, 20.0),
            math.log1p(_sat(cr / _CR_TOL, 1.0e6)),
        )
        return OrientDiagnostics(
            passport_verified=passport_ok,
            back_action_db=ba,
            back_action_tolerance_db=tol,
            back_action_ok=bool(math.isfinite(ba) and ba <= tol + 1.0e-12),
            qnd_passive=bool(_g(report, "qnd_report.is_qnd_passive", False)),
            lip_certified=lip_cert,
            lip_empirical=lip_emp,
            lip_consistent=bool(math.isfinite(lip_emp) and math.isfinite(lip_cert)
                                and lip_emp <= lip_cert + _LIP_CONSISTENCY_TOL),
            banach_ok=bool(lip_cert < ceiling and lip_emp < ceiling),
            fidelity=fid,
            melnikov_ratio=mel,
            melnikov_chaos=bool(_g(report, "kam_report.melnikov_chaos_detected", True)),
            kam_stable=bool(_g(report, "kam_report.kam_stable", False)),
            cr_residual=cr,
            gromov_rigid=bool(_g(report, "pseudoholomorphic_report.is_gromov_rigid", False)),
            engine_truth=engine_truth,
            engine_phase_verdicts=tuple(int(v) for v in _g(report, "phase_verdicts", ())),
            toon_tokens=int(_g(report, "toon_token_count", 0)),
            kv_cache_compression_pct=float(_g(report, "kv_cache_compression_pct", 0.0)),
            orbital_coordinates=coords,
        )

    # ── 2.3 veredicto de Fase 2 ───────────────────────────────────────────────────────────────────
    def classify_phase2(self, diag: OrientDiagnostics) -> Any:
        """meet(veredicto del motor, VETOED si el pasaporte falla, la cota de Banach falla o Lip_emp excede el certificado)."""
        v = diag.engine_truth
        if not (diag.passport_verified and diag.banach_ok and diag.lip_consistent):
            v = _meet(v, HeytingTruthValue.VETOED)
        return v

    # ── 2.4 cuerpo firmado de P2 (usado por emit y por Φ₂₃) ───────────────────────────────────────
    @staticmethod
    def _phase2_body(
        timestamp: float, parent: str, report: ToonCartridgeWeaverWitnessSpectrumReport,
        diag: OrientDiagnostics, verdict: Any, policy_digest: str, version: str,
    ) -> Dict[str, Any]:
        return {
            "v": version, "ts": f"{timestamp:.6f}", "parent": parent,
            "passport": str(_g(report, "passport_hash", "")), "ledger": str(_g(report, "ledger_digest", "")),
            "diag": diag, "verdict": int(verdict), "policy": policy_digest,
        }

    # ── 2.5 (ÚLTIMO MÉTODO DE FASE 2) Sello → entrada de Φ₂₃ / Fase 3 ─────────────────────────────
    def emit_phase2_certificate(
        self,
        p1: ToonWeaverWitnessPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
        witness_operator: Optional[NDArray[np.complex128]] = None,
        raw_token_count: int = 412,
        toon_token_count: int = 56,
    ) -> ToonWeaverWitnessPhase2Certificate:
        """Orienta la simulación espectral (motor v14) y emite P2 encadenado a P1 y a la política vigente."""
        binding = self.verify_input_binding(p1, rho_mac, apu_cost_matrix, exponents)
        if not binding.admitted:
            raise WitnessAuthenticityError("; ".join(binding.reasons))
        report = self._engine.execute_toon_cartridge_weaver_witness_audit(
            apu_cost_matrix=apu_cost_matrix,
            exponents=exponents,
            rho_mac_input=rho_mac,
            witness_operator=witness_operator,
            raw_token_count=raw_token_count,
            toon_token_count=toon_token_count,
        )
        diag = self.derive_orient_diagnostics(report)
        verdict = self.classify_phase2(diag)
        ts = time.time()
        pol_digest = self._policy.digest()
        body = self._phase2_body(ts, p1.chain_hash, report, diag, verdict, pol_digest, _AGENT_VERSION)
        chain, sig = self._seal(_HMAC_DOMAIN_P2, body)
        return ToonWeaverWitnessPhase2Certificate(
            timestamp=ts, spectrum_report=report, chain_hash=chain, hmac_signature=sig,
            parent_chain_hash=p1.chain_hash, diagnostics=diag, verdict=verdict,
            policy_digest=pol_digest, version=_AGENT_VERSION,
        )

    execute_phase2 = emit_phase2_certificate  # alias (firma ampliada respecto de v13)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# MONITOR ORBITAL DE POINCARÉ (mecánica celeste aplicada a la dinámica de ciclos OODA)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PoincareOrbitReport:
    """Aplicación de retorno de Poincaré sobre las secciones z_n ∈ ℝ⁹ de los ciclos de gobernanza."""
    cycle_index: int
    window_len: int
    sufficient_data: bool
    return_distance: float
    return_time_cycles: int
    recurrent: bool
    lyapunov_exponent: float
    lyapunov_function: float
    secular_drift_rel: float
    orbit_stable: bool
    verdict: Any  # HeytingTruthValue


class PoincareOrbitMonitor:
    """Sección de Poincaré, recurrencia, exponente de Lyapunov y deriva secular de V = ‖z‖².

    Honestidad de alcance: con ventanas finitas, λ es un estimador (razón de pasos sucesivos del retorno);
    no sustituye una demostración de estabilidad. Datos insuficientes ⇒ no penaliza (se marca)."""

    def __init__(self, window: int = _ORBIT_WINDOW) -> None:
        self._window = int(window)
        self._sections: Deque[NDArray[np.float64]] = deque(maxlen=self._window)
        self._count = 0

    def reset(self) -> None:
        self._sections.clear()
        self._count = 0

    @staticmethod
    def _lyapunov(arr: NDArray[np.float64]) -> float:
        steps = np.maximum(np.linalg.norm(np.diff(arr, axis=0), axis=1), _LYAPUNOV_FLOOR)
        lam = float(np.mean(np.log(steps[1:] / steps[:-1])))
        return float(np.clip(lam, -_LYAPUNOV_CAP, _LYAPUNOV_CAP))

    @staticmethod
    def _secular_drift(v: NDArray[np.float64]) -> float:
        t = np.arange(v.size, dtype=np.float64)
        t -= t.mean()
        slope = float(np.sum(t * (v - v.mean())) / np.sum(t * t))
        return slope / max(float(v.mean()), 1.0e-12)

    def push_and_assess(self, section: Sequence[float]) -> PoincareOrbitReport:
        z = np.asarray(section, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(z)):
            raise ValueError("Sección de Poincaré con valores no finitos.")
        if self._sections and self._sections[0].shape != z.shape:
            raise ValueError("Dimensión de la sección inconsistente con la órbita registrada.")

        if self._sections:
            dists = np.array([float(np.linalg.norm(z - s)) for s in self._sections])
            k = int(np.argmin(dists))
            ret_dist, ret_time = float(dists[k]), len(self._sections) - k
        else:
            ret_dist, ret_time = math.inf, 0

        self._sections.append(z)
        index = self._count
        self._count += 1
        arr = np.stack(self._sections)
        n = arr.shape[0]
        sufficient = n >= _ORBIT_MIN_SECTIONS
        lam = self._lyapunov(arr) if sufficient else 0.0
        v = np.sum(arr ** 2, axis=1)
        drift = self._secular_drift(v) if n >= 3 else 0.0
        stable = (not sufficient) or (lam <= _LYAPUNOV_TOL and drift <= _SECULAR_DRIFT_TOL)
        return PoincareOrbitReport(
            cycle_index=index, window_len=n, sufficient_data=sufficient,
            return_distance=ret_dist, return_time_cycles=ret_time, recurrent=bool(ret_dist <= _RECURRENCE_EPS),
            lyapunov_exponent=lam, lyapunov_function=float(v[-1]), secular_drift_rel=float(drift),
            orbit_stable=bool(stable),
            verdict=HeytingTruthValue.COHERENT if stable else HeytingTruthValue.DEGRADED,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES — FASE 3
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PolicyEvolutionReport:
    """Reporte RSI³ (Darwin–Gödel con replay) sobre la tolerancia de Back-Action τ."""
    evolved: bool
    reason: str
    tau_before: float
    tau_after: float
    tau_target: float
    eta_before: float
    eta_after: float
    beta_before: float
    beta_after: float
    level1_accepted: bool
    level2_accepted: bool
    replay_ok: bool
    tracking_error_before: float
    tracking_error_after: float
    lip_t1: float
    lip_t3: float
    contraction_ok: bool
    capacity_jerk: float
    capacity_third_difference: float
    jerk_nonnegative: bool
    history_len: int
    policy_digest_before: str
    policy_digest_after: str
    ledger_digest: str


@dataclass(frozen=True)
class ToonWeaverWitnessVisceralMetrics:
    """Métricas traducidas por el Funtor Semántico Visceral Φ_sem."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    toon_token_count: int
    qnd_status: str
    semantic_credit: float = 0.0
    functor_monotone: bool = True


@dataclass(frozen=True)
class ToonWeaverWitnessSovereignState:
    """Estado soberano final emitido en la Fase 3 Decide & Act del Testigo TOON."""
    timestamp: float
    truth_value: Any  # HeytingTruthValue
    is_passive_silent_qnd: bool
    back_action_db: float
    capacity_jerk_status: str
    merkle_root_hash: str
    visceral_metrics: ToonWeaverWitnessVisceralMetrics
    # ── extensiones v14 ──
    cycle_index: int
    phase_verdicts: Tuple[Any, ...]
    veto_reasons: Tuple[str, ...]
    gate12: GateDecision
    gate23: GateDecision
    orbit_report: PoincareOrbitReport
    policy_report: PolicyEvolutionReport
    merkle_leaf_items: Tuple[Tuple[str, str], ...]
    sovereign_signature: str


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class ToonWeaverWitnessPhase3Decider(ToonWeaverWitnessPhase2Orientor):
    """FASE 3 OODA — (hereda Fases 1 y 2) Φ₂₃ → Φ_sem → Heyting → Poincaré → RSI³ → Merkle."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovToonCartridgeWeaverWitnessEngine] = None,
        policy: Optional[GovernancePolicy] = None,
        strict_gates: bool = True,
        history_maxlen: int = _HISTORY_MAXLEN,
    ) -> None:
        super().__init__(secret_key, engine, policy)
        self._strict_gates = bool(strict_gates)
        self._monitor = PoincareOrbitMonitor()
        self._admitted_backactions: Deque[float] = deque(maxlen=int(history_maxlen))
        self._cycle_index = 0
        self._lock = threading.RLock()
        self._policy_ledger_tip = self._seal(_HMAC_DOMAIN_POLICY, {"genesis": _AGENT_VERSION})[1]

    # ── 3.0 (PRIMER MÉTODO DE FASE 3) Compuerta Φ₂₃: continuación del certificado P2 ──────────────
    def continue_to_phase3_gate(
        self,
        p2: ToonWeaverWitnessPhase2Certificate,
        p1: Optional[ToonWeaverWitnessPhase1Certificate] = None,
        now: Optional[float] = None,
    ) -> GateDecision:
        """Φ₂₃: (i) autenticidad: hash+HMAC+versión+eslabón P1→P2+pasaporte+re-derivación de diagnósticos;
        (ii) frescura; (iii) física: Lip_cert < 1, Lip_emp < 1, Lip_emp ≤ Lip_cert y Back-Action ≤ τₙ."""
        now = time.time() if now is None else now
        auth: List[str] = []
        reasons: List[str] = []
        notes: List[str] = []

        body = self._phase2_body(p2.timestamp, p2.parent_chain_hash, p2.spectrum_report, p2.diagnostics,
                                 p2.verdict, p2.policy_digest, p2.version)
        chain, sig = self._seal(_HMAC_DOMAIN_P2, body)
        if not hmac.compare_digest(chain, p2.chain_hash):
            auth.append("P2: hash de cadena no coincide con el contenido")
        if not hmac.compare_digest(sig, p2.hmac_signature):
            auth.append("P2: firma HMAC inválida")
        if p2.version != _AGENT_VERSION:
            auth.append(f"P2: versión {p2.version!r} ≠ {_AGENT_VERSION!r}")
        if p1 is not None and not hmac.compare_digest(p2.parent_chain_hash, p1.chain_hash):
            auth.append("P2: eslabón padre no corresponde a P1")

        if not hmac.compare_digest(p2.policy_digest, self._policy.digest()):
            reasons.append("P2: la política cambió entre la emisión y la compuerta")
        else:
            derived = self.derive_orient_diagnostics(p2.spectrum_report)
            if _canon(derived) != _canon(p2.diagnostics):
                auth.append("P2: los diagnósticos no son re-derivables del reporte del motor")
        d = p2.diagnostics
        if not d.passport_verified:
            auth.append("P2: pasaporte del motor no verificable")

        age = now - p2.timestamp
        if age > self._policy.certificate_max_age_s or age < -_CERT_FUTURE_SKEW_S:
            reasons.append(f"P2: certificado fuera de ventana de frescura (edad {age:.1f} s)")

        ceiling = self._policy.lip_ceiling
        if not d.banach_ok:
            reasons.append(f"Banach: Lip_cert={d.lip_certified:.6f}, Lip_emp={d.lip_empirical:.6f} ≮ {ceiling}")
        if not d.lip_consistent:
            reasons.append("Banach: Lip_emp excede el certificado de la RSI³ del motor")
        if not (math.isfinite(d.back_action_db) and d.back_action_db <= self._policy.backaction_tol_db + 1.0e-12):
            reasons.append(f"QND: Back-Action {d.back_action_db:.6f} dB > τ={self._policy.backaction_tol_db:.3e} dB")
        if p2.verdict == HeytingTruthValue.VETOED:
            notes.append("P2 VETOED: se adjudica sin interrumpir la atestación")

        if auth:
            logger.error("Compuerta Φ₂₃: %s", "; ".join(auth))
        if reasons:
            logger.warning("Compuerta Φ₂₃ (física/política): %s", "; ".join(reasons))
        return GateDecision("Φ₂₃", not (auth or reasons), not auth, tuple(auth + reasons), tuple(notes))

    # ── 3.1 Funtor Semántico Visceral Φ_sem (monótono) ────────────────────────────────────────────
    def apply_semantic_functor(
        self, truth: Any, report: ToonCartridgeWeaverWitnessSpectrumReport, diag: OrientDiagnostics,
    ) -> ToonWeaverWitnessVisceralMetrics:
        """Φ_sem: la protección WACC y la liquidez nominales se escalan por el crédito Φ(truth) ∈ {0, ½, 1};
        la compresión KV es una medición (no se escala)."""
        credit = _SEMANTIC_CREDIT[HeytingTruthValue(int(truth))]
        passive = bool(diag.qnd_passive and diag.back_action_ok)
        status = (f"PASSIVE_SILENT_QND_{diag.back_action_db:.6f}dB" if passive
                  else f"DEGRADED_BACKACTION_{diag.back_action_db:.6f}dB")
        return ToonWeaverWitnessVisceralMetrics(
            wacc_protection_pct=float(self._policy.wacc_nominal_pct * credit),
            net_liquidity_release_pct=float(self._policy.liquidity_nominal_pct * credit),
            kv_cache_compression_pct=float(diag.kv_cache_compression_pct),
            toon_token_count=int(diag.toon_tokens),
            qnd_status=status,
            semantic_credit=float(credit),
            functor_monotone=_functor_is_monotone(),
        )

    # ── 3.2 Adjudicación intuicionista (meet en Ω₃) ───────────────────────────────────────────────
    def adjudicate_heyting_truth(
        self,
        p1: ToonWeaverWitnessPhase1Certificate,
        p2: ToonWeaverWitnessPhase2Certificate,
        gate23: GateDecision,
        orbit: PoincareOrbitReport,
    ) -> Tuple[Any, Tuple[Any, ...], Tuple[str, ...]]:
        """truth = P1 ∧ P2 ∧ Φ₂₃ ∧ Órbita; devuelve (veredicto, (v₁, v₂, v_gate, v_orbit), razones)."""
        reasons: List[str] = list(gate23.reasons)
        if p1.verdict != HeytingTruthValue.COHERENT:
            d = p1.diagnostics
            reasons.append(
                f"Fase 1 {p1.verdict.name}: herm={d.is_hermitian}, psd={d.is_psd}, "
                f"traza_norm={d.is_normalized}, isometría={p1.isometry_probe.isometry_ok}")
        reasons.extend(f"Motor: {r}" for r in _g(p2.spectrum_report, "veto_reasons", ()))
        if not orbit.orbit_stable:
            reasons.append(f"Poincaré: λ={orbit.lyapunov_exponent:.3f}, deriva secular={orbit.secular_drift_rel:.3f}")
        v_gate = HeytingTruthValue.COHERENT if gate23.admitted else HeytingTruthValue.VETOED
        verdicts = (p1.verdict, p2.verdict, v_gate, orbit.verdict)
        return _meet(*verdicts), verdicts, tuple(reasons)

    # ── 3.3 Órbita de Poincaré ────────────────────────────────────────────────────────────────────
    def assess_poincare_orbit(
        self, p1: ToonWeaverWitnessPhase1Certificate, p2: ToonWeaverWitnessPhase2Certificate,
    ) -> PoincareOrbitReport:
        """Sección z_n = (coords. Observe[4] ‖ coords. Orient[5]) ∈ ℝ⁹ ⇒ retorno, Lyapunov y deriva secular."""
        return self._monitor.push_and_assess(tuple(p1.orbital_coordinates) + tuple(p2.diagnostics.orbital_coordinates))

    # ── 3.4 RSI³ de la política (Banach–Darwin–Gödel con replay) ──────────────────────────────────
    def evolve_policy_rsi3(
        self, truth: Any, back_action_db: float, admitted_cycle: bool = True,
    ) -> PolicyEvolutionReport:
        """Nivel 1: T_η(τ)=τ+η(τ*−τ). Nivel 2: η←η+β(η_max−η). Nivel 3: β←1.5β | β/2.
        Reglas: solo se endurece (τ' ≤ τ); τ* = clip(4·máx(historial admitido), τ_floor, 0.01 dB);
        aceptación por replay (ningún ciclo admitido previo quedaría vetado) y contracción estricta."""
        pol = self._policy
        digest_before = pol.digest()
        if (admitted_cycle and truth != HeytingTruthValue.VETOED
                and math.isfinite(back_action_db) and back_action_db >= 0.0):
            self._admitted_backactions.append(float(back_action_db))
        hist = tuple(self._admitted_backactions)
        tau, eta, beta = pol.backaction_tol_db, pol.eta, pol.beta

        if not hist:
            return PolicyEvolutionReport(
                evolved=False, reason="sin historial admitido: no hay evidencia para mutar la política",
                tau_before=tau, tau_after=tau, tau_target=tau, eta_before=eta, eta_after=eta,
                beta_before=beta, beta_after=beta, level1_accepted=False, level2_accepted=False, replay_ok=True,
                tracking_error_before=0.0, tracking_error_after=0.0, lip_t1=1.0 - eta, lip_t3=(1.0 - eta) ** 3,
                contraction_ok=True, capacity_jerk=0.0, capacity_third_difference=0.0, jerk_nonnegative=True,
                history_len=0, policy_digest_before=digest_before, policy_digest_after=digest_before,
                ledger_digest=self._policy_ledger_tip,
            )

        q = max(hist)
        target = min(max(_RSI_SAFETY_FACTOR * q, _BACKACTION_FLOOR_DB), _BACKACTION_CONSTITUTIONAL_DB)

        def step(eta_: float) -> float:
            if tau <= target:
                return tau                                  # prohibido relajar la política
            return max(min(tau, tau + eta_ * (target - tau)), target)

        def replay(candidate: float) -> bool:
            return all(h <= candidate for h in hist)

        err_before = abs(tau - target)
        eta_cand = min(_ETA_MAX, eta + beta * (_ETA_MAX - eta))
        tau_cur, tau_mut = step(eta), step(eta_cand)
        lvl2 = bool(eta_cand > eta and replay(tau_mut) and tau_mut < tau
                    and abs(tau_mut - target) < abs(tau_cur - target) * (1.0 - 1.0e-12))
        if lvl2:
            eta_new, tau_new, beta_new, lvl1 = eta_cand, tau_mut, min(_BETA_MAX, 1.5 * beta), True
        else:
            beta_new = max(_BETA_MIN, 0.5 * beta)
            eta_new = eta
            lvl1 = bool(replay(tau_cur) and tau_cur < tau)
            tau_new = tau_cur if lvl1 else tau

        err_after = abs(tau_new - target)
        lip_t1 = 1.0 - eta_new
        moved = tau_new < tau
        contraction = bool((not moved) or err_after <= lip_t1 * err_before * (1.0 + 1.0e-9) + 1.0e-18)
        replay_ok = replay(tau_new)

        # Jerk de capacidad: C_k = 1 − e₀(1−η)^k ⇒ Δ³C₀ = e₀η³ (forma cerrada) vs. tercera diferencia numérica
        jerk = float(err_before * eta_new ** 3)
        cap = 1.0 - err_before * (1.0 - eta_new) ** np.arange(4, dtype=np.float64)
        third = float(np.diff(cap, 3)[0])
        jerk_ok = bool(jerk >= 0.0 and abs(third - jerk) <= 1.0e-12 * (1.0 + err_before))

        if lvl2:
            reason = "nivel 2 aceptado (η mutado) y nivel 1 aplicado: la política se endurece"
        elif lvl1:
            reason = "nivel 2 rechazado; nivel 1 aplicado con η vigente"
        else:
            reason = "punto fijo o sin holgura (solo se permite endurecer; replay protege ciclos admitidos)"

        if lvl1 or lvl2 or beta_new != beta:
            self._policy = dataclasses.replace(pol, backaction_tol_db=tau_new, eta=eta_new, beta=beta_new)
        digest_after = self._policy.digest()
        self._policy_ledger_tip = self._seal(_HMAC_DOMAIN_POLICY, {
            "prev": self._policy_ledger_tip, "tau": tau_new, "eta": eta_new, "beta": beta_new,
            "target": target, "lvl1": lvl1, "lvl2": lvl2, "before": digest_before, "after": digest_after,
        })[1]
        return PolicyEvolutionReport(
            evolved=bool(lvl1 or lvl2), reason=reason, tau_before=tau, tau_after=tau_new, tau_target=target,
            eta_before=eta, eta_after=eta_new, beta_before=beta, beta_after=beta_new,
            level1_accepted=lvl1, level2_accepted=lvl2, replay_ok=replay_ok,
            tracking_error_before=err_before, tracking_error_after=err_after,
            lip_t1=lip_t1, lip_t3=lip_t1 ** 3, contraction_ok=contraction,
            capacity_jerk=jerk, capacity_third_difference=third, jerk_nonnegative=jerk_ok,
            history_len=len(hist), policy_digest_before=digest_before, policy_digest_after=digest_after,
            ledger_digest=self._policy_ledger_tip,
        )

    # ── 3.5 (ÚLTIMO MÉTODO DE FASE 3) Estado soberano con raíz Merkle ─────────────────────────────
    def emit_sovereign_state(
        self,
        p1: ToonWeaverWitnessPhase1Certificate,
        p2: ToonWeaverWitnessPhase2Certificate,
        gate12: Optional[GateDecision] = None,
        gate23: Optional[GateDecision] = None,
    ) -> ToonWeaverWitnessSovereignState:
        """Adjudica en Heyting Ω₃, evalúa la órbita, evoluciona la política y sella el estado (Merkle + HMAC)."""
        with self._lock:
            g12 = gate12 or self.continue_to_phase2_gate(p1)
            g23 = gate23 or self.continue_to_phase3_gate(p2, p1)
            if not (g12.authentic and g23.authentic):
                raise WitnessAuthenticityError("; ".join(g12.reasons + g23.reasons))
            if not g12.admitted:
                raise WitnessGateError("; ".join(g12.reasons))

            spec, d = p2.spectrum_report, p2.diagnostics
            orbit = self.assess_poincare_orbit(p1, p2)
            truth, verdicts, reasons = self.adjudicate_heyting_truth(p1, p2, g23, orbit)
            passive = bool(d.qnd_passive and d.back_action_ok and g23.admitted)
            metrics = self.apply_semantic_functor(truth, spec, d)
            policy_report = self.evolve_policy_rsi3(truth, d.back_action_db, admitted_cycle=g23.admitted)
            cycle = self._cycle_index
            self._cycle_index += 1

            engine_jerk = float(_g(spec, "capacity_jerk_d3c_dt3", 0.0))
            sign = "POSITIVE" if engine_jerk > 0.0 else ("NULL" if engine_jerk == 0.0 else "NEGATIVE")
            jerk_status = (f"TOON_WEAVER_WITNESS_JERK_{sign}_d3C_dt3={engine_jerk:.4f}"
                           f"|POLICY_d3C_dt3={policy_report.capacity_jerk:.3e}")

            items: Tuple[Tuple[str, str], ...] = (
                ("p1.chain", p1.chain_hash),
                ("p2.chain", p2.chain_hash),
                ("engine.passport", str(_g(spec, "passport_hash", ""))),
                ("engine.ledger", str(_g(spec, "ledger_digest", ""))),
                ("truth", hex(int(truth))),
                ("verdicts", ",".join(hex(int(v)) for v in verdicts)),
                ("back_action_db", f"{d.back_action_db:.9e}"),
                ("kv_cache_pct", f"{d.kv_cache_compression_pct:.6f}"),
                ("lip_emp", f"{d.lip_empirical:.9e}"),
                ("orbit.lyapunov", f"{orbit.lyapunov_exponent:.9e}"),
                ("orbit.return_distance", f"{orbit.return_distance:.9e}"),
                ("policy.before", policy_report.policy_digest_before),
                ("policy.after", policy_report.policy_digest_after),
                ("rsi3.ledger", policy_report.ledger_digest),
                ("cycle", str(cycle)),
            )
            root = _merkle_root(items)
            signature = hmac.new(self._secret_key, _HMAC_DOMAIN_SOV + root.encode("ascii"), hashlib.sha256).hexdigest()
            logger.info("Ciclo %d completado. Veredicto: %s", cycle, truth.name)
            return ToonWeaverWitnessSovereignState(
                timestamp=time.time(), truth_value=truth, is_passive_silent_qnd=passive,
                back_action_db=float(d.back_action_db), capacity_jerk_status=jerk_status,
                merkle_root_hash=root, visceral_metrics=metrics, cycle_index=cycle, phase_verdicts=verdicts,
                veto_reasons=reasons, gate12=g12, gate23=g23, orbit_report=orbit, policy_report=policy_report,
                merkle_leaf_items=items, sovereign_signature=signature,
            )

    def execute_phase3(
        self, p2: ToonWeaverWitnessPhase2Certificate, p1: ToonWeaverWitnessPhase1Certificate,
    ) -> ToonWeaverWitnessSovereignState:
        """Alias v13 (ahora exige P1 para la sección de Poincaré y el eslabón de la cadena)."""
        return self.emit_sovereign_state(p1, p2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO (fachada del ciclo OODA completo)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonCartridgeWeaverWitnessAgent(ToonWeaverWitnessPhase3Decider):
    """Soberano Agéntico Testigo QND de Tejeduría Sináptica TOON sobre el Fibrado Dual Boole-Novikov."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovToonCartridgeWeaverWitnessEngine] = None,
        policy: Optional[GovernancePolicy] = None,
        strict_gates: bool = True,
    ) -> None:
        super().__init__(secret_key=secret_key, engine=engine, policy=policy, strict_gates=strict_gates)
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    @property
    def policy(self) -> GovernancePolicy:
        return self._policy

    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        witness_operator: Optional[NDArray[np.complex128]] = None,
        raw_token_count: int = 412,
        toon_token_count: int = 56,
    ) -> ToonWeaverWitnessSovereignState:
        """OODA trifásico pasivo QND. Manipulación ⇒ siempre excepción; falla física ⇒ excepción si
        `strict_gates`, o estado VETOED con razones si no (la atestación pasiva no se pierde)."""
        with self._lock:
            p1 = self.emit_phase1_certificate(rho_mac, apu_cost_matrix, exponents)           # ① Observe
            g12 = self.continue_to_phase2_gate(p1)                                            # Φ₁₂
            if not g12.admitted:
                exc = WitnessAuthenticityError if not g12.authentic else WitnessGateError
                raise exc("Compuerta Φ₁₂: " + "; ".join(g12.reasons))

            p2 = self.emit_phase2_certificate(p1, apu_cost_matrix, exponents, rho_mac,        # ② Orient
                                              witness_operator, raw_token_count, toon_token_count)
            g23 = self.continue_to_phase3_gate(p2, p1)                                        # Φ₂₃
            if not g23.authentic:
                raise WitnessAuthenticityError("Compuerta Φ₂₃: " + "; ".join(g23.reasons))
            if not g23.admitted and self._strict_gates:
                raise WitnessStabilityError("Compuerta Φ₂₃: " + "; ".join(g23.reasons))

            return self.emit_sovereign_state(p1, p2, g12, g23)                                # ③ Decide & Act

    # ── verificación posterior del estado ─────────────────────────────────────────────────────────
    def verify_sovereign_state(self, state: ToonWeaverWitnessSovereignState) -> bool:
        """Recomputa raíz Merkle y firma HMAC en tiempo constante."""
        root = _merkle_root(state.merkle_leaf_items)
        sig = hmac.new(self._secret_key, _HMAC_DOMAIN_SOV + root.encode("ascii"), hashlib.sha256).hexdigest()
        return bool(hmac.compare_digest(root, state.merkle_root_hash)
                    and hmac.compare_digest(sig, state.sovereign_signature))

    @staticmethod
    def prove_leaf(state: ToonWeaverWitnessSovereignState, label: str) -> Tuple[Tuple[str, bool], ...]:
        labels = [lab for lab, _ in state.merkle_leaf_items]
        if label not in labels:
            raise KeyError(label)
        return _merkle_proof(state.merkle_leaf_items, labels.index(label))

    @staticmethod
    def verify_leaf_proof(root_hex: str, label: str, value: str, proof: Sequence[Tuple[str, bool]]) -> bool:
        return _merkle_verify(root_hex, label, value, proof)

    def reset_policy(self, policy: Optional[GovernancePolicy] = None) -> None:
        """Restablece política, historial admitido, libro mayor RSI³ y órbita (p. ej. tras un cambio de régimen)."""
        with self._lock:
            self._policy = policy or GovernancePolicy()
            self._admitted_backactions.clear()
            self._monitor.reset()
            self._policy_ledger_tip = self._seal(_HMAC_DOMAIN_POLICY, {"genesis": _AGENT_VERSION})[1]


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# COMPUERTAS COMO FUNCIONES DE MÓDULO (compatibilidad v13)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def continue_to_phase2_gate(
    p1: ToonWeaverWitnessPhase1Certificate, secret_key: bytes, max_age_s: Optional[float] = None,
) -> bool:
    """Compuerta Homomórfica Φ₁₂ — valida hash, firma HMAC y frescura de P1 (admite VETOED)."""
    observer = ToonWeaverWitnessPhase1Observer(secret_key)
    age = _CERT_MAX_AGE_S if max_age_s is None else max_age_s
    return observer._verify_phase1_certificate(p1, age).admitted


def continue_to_phase3_gate(
    p2: ToonWeaverWitnessPhase2Certificate,
    secret_key: bytes,
    p1: Optional[ToonWeaverWitnessPhase1Certificate] = None,
    policy: Optional[GovernancePolicy] = None,
    engine: Optional[BoolNovikovToonCartridgeWeaverWitnessEngine] = None,
) -> bool:
    """Compuerta Homomórfica Φ₂₃ — firma P2, eslabón P1, Banach (Lip_emp < 1) y Back-Action ≤ τₙ.
    Requiere el mismo `secret_key` con que el motor firmó el pasaporte."""
    decider = ToonWeaverWitnessPhase3Decider(secret_key=secret_key, engine=engine, policy=policy)
    return decider.continue_to_phase3_gate(p2, p1).admitted


__all__ = [
    "WitnessGateError", "WitnessAuthenticityError", "WitnessStabilityError", "GateDecision", "GovernancePolicy",
    "DensityDiagnostics", "BuresIsometryProbe", "ToonWeaverWitnessPhase1Certificate", "OrientDiagnostics",
    "ToonWeaverWitnessPhase2Certificate", "PoincareOrbitReport", "PoincareOrbitMonitor", "PolicyEvolutionReport",
    "ToonWeaverWitnessVisceralMetrics", "ToonWeaverWitnessSovereignState",
    "ToonWeaverWitnessPhase1Observer", "ToonWeaverWitnessPhase2Orientor", "ToonWeaverWitnessPhase3Decider",
    "BoolNovikovToonCartridgeWeaverWitnessAgent", "continue_to_phase2_gate", "continue_to_phase3_gate",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverWitnessAgent v14 ===")
    KEY = b"unit-test-key-not-for-production"

    # ── Auto-verificaciones puras ────────────────────────────────────────────────────────────────
    leaves = (("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5"))
    mroot = _merkle_root(leaves)
    for i, (lab, val) in enumerate(leaves):
        assert _merkle_verify(mroot, lab, val, _merkle_proof(leaves, i)), "Merkle: inclusión fallida"
    assert not _merkle_verify(mroot, "a", "manipulado", _merkle_proof(leaves, 0)), "Merkle: no detecta manipulación"
    assert _functor_is_monotone(), "Φ_sem no es monótono"

    mon = PoincareOrbitMonitor()
    for _ in range(8):
        rep_fix = mon.push_and_assess((0.1,) * 9)               # punto fijo: estable y recurrente
    assert rep_fix.orbit_stable and rep_fix.recurrent
    mon2 = PoincareOrbitMonitor()
    for k in range(8):
        rep_div = mon2.push_and_assess((0.01 * 2.0 ** k,) + (0.0,) * 8)   # divergencia exponencial (λ = ln 2)
    assert (not rep_div.orbit_stable) and rep_div.lyapunov_exponent > _LYAPUNOV_TOL

    # ── RSI³ determinista de la política ─────────────────────────────────────────────────────────
    rsi_agent = BoolNovikovToonCartridgeWeaverWitnessAgent(secret_key=KEY)
    tau_prev = rsi_agent.policy.backaction_tol_db
    for _ in range(12):
        rr = rsi_agent.evolve_policy_rsi3(HeytingTruthValue.COHERENT, 0.0)
        assert rr.replay_ok and rr.contraction_ok and rr.lip_t3 < 1.0 and rr.jerk_nonnegative
        assert _BACKACTION_FLOOR_DB <= rr.tau_after <= tau_prev, "la política solo puede endurecerse"
        assert _ETA_MIN <= rr.eta_after <= _ETA_MAX and _BETA_MIN <= rr.beta_after <= _BETA_MAX
        tau_prev = rr.tau_after
    assert rsi_agent.policy.backaction_tol_db < _BACKACTION_CONSTITUTIONAL_DB

    # ── Ciclo OODA completo ──────────────────────────────────────────────────────────────────────
    agent = BoolNovikovToonCartridgeWeaverWitnessAgent(secret_key=KEY)
    dim = 4
    matrix_test = np.array([
        [1.0, 0.2, 0.0, 0.1],
        [0.2, 1.1, 0.3, 0.0],
        [0.0, 0.3, 0.9, 0.2],
        [0.1, 0.0, 0.2, 1.0],
    ], dtype=np.float64)
    exps_test = np.array([0.15, 0.28, 0.45], dtype=np.float64)
    raw_rho = np.eye(dim, dtype=np.complex128) + 0.02 * np.ones((dim, dim), dtype=np.complex128)
    rho_test = raw_rho / float(np.real(np.trace(raw_rho)))

    state = agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test)
    assert agent.verify_sovereign_state(state), "estado soberano no verificable"
    proof = agent.prove_leaf(state, "truth")
    assert agent.verify_leaf_proof(state.merkle_root_hash, "truth", dict(state.merkle_leaf_items)["truth"], proof)

    print(f"Veredicto Final Heyting Ω₃ : {state.truth_value.name} ({hex(state.truth_value.value)}) "
          f"fases={[v.name for v in state.phase_verdicts]}")
    for reason in state.veto_reasons:
        print(f"   · {reason}")
    print(f"Invariante PASSIVE_SILENT  : {state.is_passive_silent_qnd}")
    print(f"Back-Action QND (dB)       : {state.back_action_db:.6f} dB")
    print(f"Compresión KV-Cache (%)    : {state.visceral_metrics.kv_cache_compression_pct:.1f}% "
          f"({state.visceral_metrics.toon_token_count} tokens)")
    print(f"Φ_sem crédito / WACC / Liq : {state.visceral_metrics.semantic_credit:.1f} / "
          f"{state.visceral_metrics.wacc_protection_pct}% / {state.visceral_metrics.net_liquidity_release_pct}%")
    print(f"Órbita Poincaré (λ, V)     : {state.orbit_report.lyapunov_exponent:.4f}, "
          f"{state.orbit_report.lyapunov_function:.4f}  estable={state.orbit_report.orbit_stable}")
    print(f"RSI³ política τ: {state.policy_report.tau_before:.3e} → {state.policy_report.tau_after:.3e} dB  "
          f"(η={state.policy_report.eta_after:.3f}, β={state.policy_report.beta_after:.3f}, "
          f"Lip(T³)={state.policy_report.lip_t3:.4f})")
    print(f"Estado de jerk             : {state.capacity_jerk_status}")
    print(f"Raíz Merkle SHA-256        : {state.merkle_root_hash[:16]}...  firma={state.sovereign_signature[:16]}...")

    for _ in range(4):  # varios ciclos: la órbita acumula secciones y la política puede endurecerse
        agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test)

    # ── Contrastes negativos ─────────────────────────────────────────────────────────────────────
    p1 = agent.emit_phase1_certificate(rho_test, matrix_test, exps_test)
    assert agent.continue_to_phase2_gate(p1).admitted
    forged_p1 = dataclasses.replace(p1, tr_rho=0.5)                       # 1) P1 manipulado
    assert not agent.continue_to_phase2_gate(forged_p1).authentic
    assert not continue_to_phase2_gate(forged_p1, KEY)

    other = rho_test.copy()                                               # 2) TOCTOU: ρ distinta de la observada
    other[0, 1] += 0.01
    other[1, 0] += 0.01
    try:
        agent.emit_phase2_certificate(p1, matrix_test, exps_test, other)
        raise AssertionError("TOCTOU no detectado")
    except WitnessAuthenticityError:
        pass

    p2 = agent.emit_phase2_certificate(p1, matrix_test, exps_test, rho_test)
    assert agent.continue_to_phase3_gate(p2, p1).authentic
    forged_p2 = dataclasses.replace(p2, parent_chain_hash="0" * 64)       # 3) eslabón P1→P2 roto
    assert not agent.continue_to_phase3_gate(forged_p2, p1).authentic

    bad_witness = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)     # 4) testigo no conmutante ⇒ back-action > τ
    try:
        agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test, witness_operator=bad_witness)
        raise AssertionError("Φ₂₃ estricta debió rechazar el testigo no conmutante")
    except WitnessStabilityError:
        pass
    lenient = BoolNovikovToonCartridgeWeaverWitnessAgent(secret_key=KEY, strict_gates=False)
    bad_state = lenient.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test, witness_operator=bad_witness)
    assert (not bad_state.is_passive_silent_qnd) and bad_state.truth_value != HeytingTruthValue.COHERENT
    assert bad_state.visceral_metrics.wacc_protection_pct < 15.0 or bad_state.truth_value == HeytingTruthValue.DEGRADED
    print("=== TEST COMPLETADO CON ÉXITO ===")