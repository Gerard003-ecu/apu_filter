# -*- coding: utf-8 -*-
r"""
BoolNovikovToonBufferProsecutorAgent — v15.0.0-PhD
Ubicación : app/agents/wisdom/toon_buffer_prosecutor_agent.py
Versión   : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-BUFFER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD
Motor     : app/wisdom/toon_buffer_prosecutor_engine.py (v15.0.0)
Gobernanza: OODA trifásica inquisitorial
            Observe (Fase 1) ➔ Φ₁₂ ➔ Orient (Fase 2) ➔ Φ₂₃ ➔ Decide & Act (Fase 3)

JERARQUÍA DE FASES (herencia anidada estricta, sin composición lateral)
  ToonBufferProsecutorPhase1Observer
    ⊂ ToonBufferProsecutorPhase2Orientor
      ⊂ ToonBufferProsecutorPhase3Decider
        ⊂ BoolNovikovToonBufferProsecutorAgent

ENLACES FORMALES DIRECTOS ENTRE FASES
  Último método de Fase 1 : execute_phase1(...) -> ToonBufferProsecutorPhase1Certificate
      ➔ Primer método de Fase 2 : continue_to_phase2_gate(p1, ...) [Compuerta Homomórfica Φ₁₂]
  Último método de Fase 2 : execute_phase2(...) -> ToonBufferProsecutorPhase2Certificate
      ➔ Primer método de Fase 3 : continue_to_phase3_gate(p2)      [Compuerta Homomórfica Φ₂₃]
  Último método de Fase 3 : execute_phase3(p2) -> ToonBufferProsecutorSovereignState

FUNDAMENTACIÓN FÍSICO-MATEMÁTICA INTEGRADA (v15.0.0-PhD)
  1. FASE 1 — OBSERVE (Óptica Cuántica, Teoría Espectral y Circuitos de Brune):
     • Ingesta sin normalización silenciosa de la matriz de densidad ρ_MAC ∈ ℂ^{N×N},
       de la Matriz de Interacción Central (MIC) M ∈ ℝ^{N×N}, del espectro de Lyapunov
       λ ∈ ℝ^k y de la matriz de incidencia topológica opcional B ∈ ℤ^{E×V}.
     • Cuantificación de hermiticidad ‖ρ − ρ†‖_F, positividad semidefinida λ_min(H(ρ)),
       entropía cuántica de von Neumann S(ρ) = −Tr(ρ ln ρ), pureza de Hilbert-Schmidt
       γ(ρ) = Tr(ρ†ρ) e información asimétrica cuántica de Wigner-Yanase
       I_WY(ρ, H_MIC) = −½ Tr([√ρ, H_MIC]²) respecto al hamiltoniano simetrizado
       H_MIC = ½(M + Mᵀ).
     • Auditoría eléctrica de pasividad de Brune λ_min(½(M + Mᵀ)) y cota espectral
       de discos de Gershgorin max_i (|M_ii| + ∑_{j≠i} |M_ij|).
     • Sellado criptográfico SHA-256 canónico y HMAC-SHA256 con separación de dominio.

  2. FASE 2 — ORIENT (Mecánica Celeste de Poincaré-KAM, Electrodinámica de Dirac y RSI³):
     • Compuerta Φ₁₂ fail-closed: recalcula el digest canónico de P1, valida el sello
       HMAC, exige densidad cuántica físicamente admisible y verifica la ligadura
       criptográfica exacta de MIC, exponentes, ρ_MAC e incidencia.
     • Invoca la auditoría inquisitorial completa del motor v15.0.0:
       - Aniquilación cuántica de Dirac e⁻e⁺ en Cl_{1,3}(ℂ) y física del tiristor
         crowbar (dI/dt pico e integral de fusión de Joule ∫i²dt).
       - Homología entera de Smith, brecha espectral de Hodge, energía booleana
         de Rademacher-Walsh-Parseval y valuación en el anillo de Novikov.
       - Dinámica de fluidos de golpe de ariete de Zhukovsky ΔP_Zhuk = ρ₀ c_s Δv
         y cota de entropía métrica de Pesin h_KS ≤ ∑ max(0, λ_i).
       - Mecánica celeste de Poincaré-KAM: coordenadas canónicas no singulares
         de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂), defecto del jacobiano simpléctico
         ‖Jᵀ Ω J − Ω‖_F, determinantes de torsión de Kolmogorov y Arnol'd
         isoenergético, serie de Lie de Deprit-Hori, tiempo exponencial de
         estabilidad de Nekhoroshev, deriva del invariante integral relativo de
         Poincaré-Cartan ∮ p dq (integrador simpléctico de Yoshida de 4º orden)
         y traza de monodromía elíptica de Floquet-Broucke.
       - Invariante celeste de Tisserand-Poincaré C_T = 1/L² + 2 ω_p H_Delaunay
         e integral vertical de momento angular de Kozai-Lidov L_z = H_Delaunay / L.
       - Automejora Recursiva de Nivel 3 (RSI³): punto fijo de Picard (Nivel 1),
         meta-endofuntor cuadrático de Schulz-Picard Lip_L2 = Lip_L1² (Nivel 2) y
         meta-meta-gobernador de coherencia de mónada de Kleisli + guardia de
         reflexión de Löb-Tarski (Nivel 3).

  3. FASE 3 — DECIDE & ACT (Topos de Heyting Ω₃, Gavificación de Lawvere-Tierney y Merkle RFC 6962):
     • Compuerta Φ₂₃ fail-closed: verifica digest y HMAC de P2, re-deriva de forma
       independiente el veredicto en el álgebra de Heyting Ω₃ desde las cláusulas
       selladas, audita la coherencia del disparo crowbar de Dirac y ejecuta una
       verificación independiente de solidez meta-recursiva RSI³ (Löb-Tarski +
       leyes de Kleisli) e invariancia simpléctica de Poincaré-Cartan.
     • Aplica el operador modal de gavificación de Lawvere-Tierney j_LT: Ω₃ → Ω₃,
       construye el estado soberano inmutable y calcula tanto la raíz de Merkle
       RFC 6962 como las pruebas criptográficas de inclusión para las hojas
       (P1, P2, Estado Soberano).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from app.wisdom.toon_buffer_prosecutor_engine import (
        BoolNovikovToonBufferProsecutorEngine,
        HeytingTruthValue,
        ToonBufferProsecutorSpectrumReport,
    )
except ImportError:
    try:
        from toon_buffer_prosecutor_engine import (
            BoolNovikovToonBufferProsecutorEngine,
            HeytingTruthValue,
            ToonBufferProsecutorSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar toon_buffer_prosecutor_engine desde 'app.wisdom' "
            "ni desde la raíz del módulo."
        ) from exc


logger = logging.getLogger("APU.Wisdom.BoolNovikovToonBufferProsecutorAgent")

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-BUFFER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_TOON_BUFFER_PROSECUTOR_AGENT_KEY_v14.0.0"
_MIN_SECRET_KEY_BYTES: Final[int] = 16
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-BUFFER-PROSECUTOR-AGENT-P1-OBSERVE-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-BUFFER-PROSECUTOR-AGENT-P2-ORIENT-v15.0|"
_MERKLE_LEAF_PREFIX: Final[bytes] = b"\x00"
_MERKLE_NODE_PREFIX: Final[bytes] = b"\x01"

_HERMITIAN_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_TRACE_TOL: Final[float] = 1.0e-9
_KLEISLI_COHERENCE_TOL: Final[float] = 1.0e-9
_SYMPLECTIC_INVARIANT_TOL: Final[float] = 1.0e-9

_ESP32_CROWBAR_BUDGET_NS: Final[float] = 392.15
_CROWBAR_PROVENANCE: Final[str] = "DECLARED_HARDWARE_SPEC_NOT_MEASURED"
_VISCERAL_PROVENANCE: Final[str] = "DECLARED_POLICY_NOT_MEASURED"
_HARDWARE_ACTUATION_EMITTED: Final[bool] = False
_PERTURBER_MEAN_MOTION_OMEGA: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)  # Razón áurea diofántica

_DEFAULT_STAGE_FACTORS: Final[Tuple[float, float, float]] = (0.95, 0.95, 0.95)
_DEFAULT_QUANT_SCALE: Final[float] = 1000.0
_DEFAULT_BUFFER_CAPACITY: Final[float] = 100.0

Clock = Callable[[], float]


# ════════════════════════════════════════════════════════════════════════════
# Excepciones, tipos de certificados y estado soberano
# ════════════════════════════════════════════════════════════════════════════

class ToonBufferProsecutorGateError(ValueError):
    """Rechazo fail-closed en una compuerta homomórfica inquisitorial (Φ₁₂ o Φ₂₃)."""


@dataclass(frozen=True)
class ToonBufferProsecutorPhase1Certificate:
    """
    Certificado P1 (Fase 1 — Observe).
    Sella la observación cuántica, espectral, eléctrica y topológica de MIC,
    exponentes, ρ_MAC y matriz de incidencia opcional sin normalización silenciosa.
    """
    timestamp: float
    dim: int
    mic_sha256: str
    exponents_sha256: str
    density_sha256: str
    tr_rho: float
    tr_imag_abs: float
    hermitian_residual: float
    min_eigenvalue: float
    is_hermitian: bool
    is_psd: bool
    is_unit_trace: bool
    payload_digest: str
    hmac_signature: str
    von_neumann_entropy: float = 0.0
    hilbert_schmidt_purity: float = 1.0
    wigner_yanase_skew_info: float = 0.0
    brune_passivity_min_eig: float = 1.0
    gershgorin_spectral_bound: float = 1.0
    incidence_sha256: str = "NONE"

    @property
    def is_valid_density(self) -> bool:
        """ρ es densidad cuántica válida si es hermítica, PSD y de traza unitaria."""
        return self.is_hermitian and self.is_psd and self.is_unit_trace


@dataclass(frozen=True)
class ToonBufferProsecutorPhase2Certificate:
    """
    Certificado P2 (Fase 2 — Orient).
    Sella el veredicto del motor inquisitorial, sus cláusulas de Heyting,
    los invariantes electrodinámicos de Dirac, la mecánica celeste de
    Poincaré-KAM y el certificado de Automejora Recursiva de Nivel 3 (RSI³).
    Los campos sellados son la única fuente de decisión de la Fase 3.
    `spectrum_report` se adjunta como referencia informativa no sellada.
    """
    timestamp: float
    p1_payload_digest: str
    mic_sha256: str
    exponents_sha256: str
    density_sha256: str
    truth_value: HeytingTruthValue
    veto_failures: Tuple[str, ...]
    degrade_failures: Tuple[str, ...]
    crowbar_triggered: bool
    photons_emitted: int
    oseledets_lambda: float
    water_hammer_overflow: bool
    banach_lip: float
    krull_valuation: float
    gromov_rigid: bool
    melnikov_chaos: bool
    smith_torsion_free: bool
    passport_hash: str
    payload_digest: str
    hmac_signature: str
    spectrum_report: ToonBufferProsecutorSpectrumReport = field(compare=False, repr=False)
    incidence_sha256: str = "NONE"
    # Invariantes electrodinámicos de Dirac y tiristor crowbar
    clifford_anticommutator_defect: float = 0.0
    chiral_gamma5_involutory_defect: float = 0.0
    mandelstam_s_kev2: float = 0.0
    crowbar_peak_di_dt_a_per_us: float = 0.0
    crowbar_joule_integral_a2s: float = 0.0
    # Invariantes de fluidos y topología algebraica de Hodge-Smith
    zhukovsky_surge_pressure: float = 0.0
    pesin_entropy_bound: float = 0.0
    torsion_subgroup_order: int = 1
    euler_characteristic: int = 1
    hodge_spectral_gap: float = 0.0
    # Mecánica celeste de Poincaré-KAM
    symplectic_jacobian_defect: float = 0.0
    poincare_cartan_drift: float = 0.0
    kolmogorov_torsion_det: float = 1.0
    arnold_isoenergetic_det: float = -1.0
    nekhoroshev_stability_time: float = 1.0
    floquet_monodromy_trace: float = 0.0
    is_floquet_elliptic_stable: bool = True
    tisserand_poincare_invariant: float = 0.0
    kozai_lidov_lz_invariant: float = 0.0
    # Automejora Recursiva de Nivel 3 (RSI³)
    rsi_depth_reached: int = 3
    rsi_meta_preconditioned_lip: float = 0.0
    rsi_banach_aposteriori_error: float = 0.0
    rsi_lyapunov_descent_verified: bool = True
    rsi_kleisli_associativity_defect: float = 0.0
    rsi_kleisli_unit_defect: float = 0.0
    rsi_loeb_soundness_verified: bool = True
    rsi_accepted: bool = True


@dataclass(frozen=True)
class ToonBufferVisceralPolicy:
    """Política visceral declarada para el Funtor Semántico Φ_sem. No son mediciones."""
    wacc_protection_pct: float = 15.0
    net_liquidity_release_pct: float = 11.5
    kv_cache_delta_pct: float = -86.4

    def __post_init__(self) -> None:
        for name in ("wacc_protection_pct", "net_liquidity_release_pct", "kv_cache_delta_pct"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} debe ser un número finito.")


@dataclass(frozen=True)
class ToonBufferVisceralMetrics:
    """Métricas traducidas por el Funtor Semántico Visceral Φ_sem."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_delta_pct: float
    water_hammer_status: str
    provenance: str


@dataclass(frozen=True)
class ToonBufferProsecutorSovereignState:
    """Estado soberano final emitido por la Fase 3 (Decide & Act)."""
    timestamp: float
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_budget_ns: float
    crowbar_latency_provenance: str
    hardware_actuation_emitted: bool
    state_digest: str
    merkle_root_hash: str
    visceral_metrics: ToonBufferVisceralMetrics
    merkle_inclusion_proofs: Tuple[Tuple[Tuple[str, str], ...], ...] = ()
    rsi3_loeb_soundness_ok: bool = True
    kleisli_coherence_ok: bool = True
    poincare_symplectic_invariant_ok: bool = True
    lawvere_tierney_truth: HeytingTruthValue = HeytingTruthValue.COHERENT

    @property
    def crowbar_latency_ns(self) -> float:
        """Alias de compatibilidad v13. Es un presupuesto declarado, NO una latencia medida."""
        return self.crowbar_latency_budget_ns


# ════════════════════════════════════════════════════════════════════════════
# Utilidades criptográficas, canónicas, de álgebra de Heyting y Merkle RFC 6962
# ════════════════════════════════════════════════════════════════════════════

def _validate_secret_key(secret_key: object) -> bytes:
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key debe ser bytes.")
    key = bytes(secret_key)
    if len(key) < _MIN_SECRET_KEY_BYTES:
        raise ValueError(f"secret_key debe tener al menos {_MIN_SECRET_KEY_BYTES} bytes.")
    if key == _DEFAULT_SECRET_KEY:
        warnings.warn(
            "Se usa la clave por defecto del agente; no es apta para producción.",
            RuntimeWarning,
            stacklevel=3,
        )
    return key


def _validate_clock(clock: object) -> Clock:
    if not callable(clock):
        raise TypeError("clock debe ser invocable y devolver un flotante.")
    return clock  # type: ignore[return-value]


def _as_real_float64(value: object, name: str) -> NDArray[np.float64]:
    """Convierte a float64 sin descartar parte imaginaria en silencio."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} no es un arreglo válido.") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} debe ser real; se recibió un arreglo complejo.")
    try:
        return raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} no es numérico.") from exc


def _str_equal(expected: str, observed: object) -> bool:
    """Comparación de cadenas en tiempo constante, tolerante a tipos forjados."""
    if not isinstance(observed, str):
        return False
    return hmac.compare_digest(
        expected.encode("utf-8", "surrogatepass"),
        observed.encode("utf-8", "surrogatepass"),
    )


def _jsafe(value: Any) -> Any:
    """Vuelve serializables NaN e infinitos sin perder información."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, dict):
        return {k: _jsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsafe(v) for v in value]
    return value


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """JSON canónico: claves ordenadas, ASCII, sin NaN no finitos literales."""
    return json.dumps(
        _jsafe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(arr: NDArray[Any]) -> str:
    """Digest criptográfico de forma, tipo y bytes little-endian explícitos."""
    dtype = "<c16" if np.iscomplexobj(arr) else "<f8"
    a = np.ascontiguousarray(arr, dtype=dtype)
    h = hashlib.sha256()
    h.update(f"{dtype}|{a.shape}|".encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _incidence_digest(incidence_matrix: Optional[Any]) -> str:
    """Digest criptográfico de la matriz de incidencia entera opcional o 'NONE'."""
    if incidence_matrix is None:
        return "NONE"
    raw = _as_real_float64(incidence_matrix, "incidence_matrix")
    if raw.ndim != 2 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("incidence_matrix debe ser una matriz 2-D finita y no vacía.")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1e-9):
        raise ValueError("incidence_matrix debe contener entradas enteras.")
    arr_i64 = np.ascontiguousarray(rounded, dtype="<i8")
    h = hashlib.sha256()
    h.update(f"<i8|{arr_i64.shape}|".encode("ascii"))
    h.update(arr_i64.tobytes())
    return h.hexdigest()


def _hmac_hex(key: bytes, domain: bytes, digest_hex: str) -> str:
    return hmac.new(key, domain + digest_hex.encode("ascii"), hashlib.sha256).hexdigest()


def _verify_seal(key: bytes, domain: bytes, digest_hex: str, signature: object) -> bool:
    return _str_equal(_hmac_hex(key, domain, digest_hex), signature)


def _rank(truth: HeytingTruthValue) -> int:
    """Rango en la cadena distributiva acotada COHERENT (2) ≻ DEGRADED (1) ≻ VETOED (0)."""
    return 2 - int(truth)


def _meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
    """Ínfimo (∧) en la cadena de Heyting Ω₃: el valor de menor rango."""
    return a if _rank(a) <= _rank(b) else b


def _merkle_root(leaf_digests: Tuple[str, ...]) -> str:
    """Árbol de Merkle RFC 6962: prefijo 0x00 para hojas y 0x01 para nodos internos."""
    if not leaf_digests:
        raise ValueError("El árbol de Merkle requiere al menos una hoja.")
    level = [hashlib.sha256(_MERKLE_LEAF_PREFIX + bytes.fromhex(d)).digest() for d in leaf_digests]
    while len(level) > 1:
        parents = [
            hashlib.sha256(_MERKLE_NODE_PREFIX + level[i] + level[i + 1]).digest()
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2 == 1:
            parents.append(level[-1])
        level = parents
    return level[0].hex()


def _merkle_inclusion_proofs(
    leaf_digests: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """
    Construye las pruebas de inclusión RFC 6962 para cada hoja del árbol de Merkle.
    Cada paso de prueba es un par `(side, sibling_hex)` donde `side` ∈ {'L', 'R'}
    indica si el nodo hermano se concatena a la izquierda o a la derecha.
    """
    if not leaf_digests:
        raise ValueError("Se requiere al menos una hoja para generar pruebas de inclusión.")
    n = len(leaf_digests)
    proofs: List[List[Tuple[str, str]]] = [[] for _ in range(n)]
    level: List[bytes] = [
        hashlib.sha256(_MERKLE_LEAF_PREFIX + bytes.fromhex(d)).digest()
        for d in leaf_digests
    ]
    indices: List[int] = list(range(n))

    while len(level) > 1:
        next_level: List[bytes] = []
        next_indices: List[int] = [0] * n
        for i in range(0, len(level) - 1, 2):
            left_node = level[i]
            right_node = level[i + 1]
            parent = hashlib.sha256(_MERKLE_NODE_PREFIX + left_node + right_node).digest()
            parent_pos = len(next_level)
            next_level.append(parent)
            for orig_idx in range(n):
                if indices[orig_idx] == i:
                    proofs[orig_idx].append(("R", right_node.hex()))
                    next_indices[orig_idx] = parent_pos
                elif indices[orig_idx] == i + 1:
                    proofs[orig_idx].append(("L", left_node.hex()))
                    next_indices[orig_idx] = parent_pos
        if len(level) % 2 == 1:
            last_pos = len(level) - 1
            parent_pos = len(next_level)
            next_level.append(level[last_pos])
            for orig_idx in range(n):
                if indices[orig_idx] == last_pos:
                    next_indices[orig_idx] = parent_pos
        level = next_level
        indices = next_indices

    return tuple(tuple(p) for p in proofs)


def verify_merkle_inclusion_proof(
    leaf_digest: str,
    proof: Tuple[Tuple[str, str], ...],
    expected_root: str,
) -> bool:
    """Verifica criptográficamente una prueba de inclusión RFC 6962 en tiempo constante."""
    try:
        node = hashlib.sha256(_MERKLE_LEAF_PREFIX + bytes.fromhex(leaf_digest)).digest()
        for side, sibling_hex in proof:
            sibling = bytes.fromhex(sibling_hex)
            if side == "L":
                node = hashlib.sha256(_MERKLE_NODE_PREFIX + sibling + node).digest()
            elif side == "R":
                node = hashlib.sha256(_MERKLE_NODE_PREFIX + node + sibling).digest()
            else:
                return False
        return _str_equal(node.hex(), expected_root)
    except (ValueError, TypeError):
        return False


# ════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE (Óptica Cuántica, Espectro de Wigner-Yanase y Pasividad de Brune)
# ════════════════════════════════════════════════════════════════════════════

class ToonBufferProsecutorPhase1Observer:
    """
    FASE 1 — OBSERVE. Métodos en orden estricto de definición:
      1. __init__                   valida clave criptográfica y reloj inyectado.
      2. _ingest_cost_matrix        (static) valida MIC cuadrada, real, finita y no vacía.
      3. _ingest_exponents          (static) valida exponentes 1-D reales, finitos y no vacíos.
      4. _ingest_density            (static) valida ρ_MAC de forma (dim, dim) sin normalizar.
      5. _observe_density           (static) mide hermiticidad, PSD, traza, entropía de von Neumann,
                                    pureza de Hilbert-Schmidt e información de Wigner-Yanase.
      6. _observe_mic_passivity     (static) mide pasividad de impedancia de Brune y cota de Gershgorin.
      7. _p1_payload                (static) estructura canónica sellable del certificado P1.
      8. _p1_digest                 (classmethod) digest SHA-256 canónico de _p1_payload.
      9. _read_clock                lectura validada del reloj inyectado.
      10. execute_phase1            sella y emite ToonBufferProsecutorPhase1Certificate.
                                    ÚLTIMO método de Fase 1 -> entrada directa de Fase 2 (Φ₁₂).
    """

    def __init__(self, secret_key: bytes, clock: Clock = time.time) -> None:
        self._secret_key: bytes = _validate_secret_key(secret_key)
        self._clock: Clock = _validate_clock(clock)

    @staticmethod
    def _ingest_cost_matrix(apu_cost_matrix: object) -> NDArray[np.float64]:
        mic = _as_real_float64(apu_cost_matrix, "La MIC")
        if mic.ndim != 2 or mic.shape[0] != mic.shape[1] or mic.shape[0] == 0:
            raise ValueError("La MIC debe ser una matriz cuadrada, bidimensional y no vacía.")
        if not np.all(np.isfinite(mic)):
            raise ValueError("La MIC contiene valores no finitos.")
        return mic

    @staticmethod
    def _ingest_exponents(exponents: object) -> NDArray[np.float64]:
        exps = _as_real_float64(exponents, "Los exponentes")
        if exps.ndim != 1 or exps.size == 0:
            raise ValueError("Los exponentes deben ser un vector 1-D real y no vacío.")
        if not np.all(np.isfinite(exps)):
            raise ValueError("Los exponentes contienen valores no finitos.")
        return exps

    @staticmethod
    def _ingest_density(rho_mac: object, dim: int) -> NDArray[np.complex128]:
        try:
            rho = np.asarray(rho_mac, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise ValueError("ρ_MAC no es numérica.") from exc
        if rho.shape != (dim, dim):
            raise ValueError(f"ρ_MAC debe tener forma ({dim}, {dim}); se recibió {rho.shape}.")
        if not np.all(np.isfinite(rho)):
            raise ValueError("ρ_MAC contiene valores no finitos.")
        return rho

    @staticmethod
    def _observe_density(
        rho: NDArray[np.complex128],
        mic: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        """
        Observa la matriz de densidad cuántica ρ_MAC sin corregirla ni normalizarla:
          • Residuo hermítico ‖ρ − ρ†‖_F y autovalor mínimo λ_min(½(ρ + ρ†)).
          • Entropía cuántica de von Neumann S(ρ) = −∑ λ_i ln(λ_i).
          • Pureza de Hilbert-Schmidt γ(ρ) = Tr(ρ† ρ).
          • Información asimétrica cuántica de Wigner-Yanase:
                I_WY(ρ, H_MIC) = −½ Tr([√ρ, H_MIC]²) ≥ 0
            donde H_MIC = ½(M + Mᵀ) es la parte hermítica observable de la MIC.
        """
        trace = complex(np.trace(rho))
        residual = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
        herm_part = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = np.linalg.eigh(herm_part)
        min_eig = float(eigvals[0])

        clipped_eigs = np.maximum(eigvals, 0.0)
        positive_mask = clipped_eigs > 1.0e-15
        if np.any(positive_mask):
            p_pos = clipped_eigs[positive_mask]
            vn_entropy = float(-np.sum(p_pos * np.log(p_pos)))
        else:
            vn_entropy = 0.0

        purity = float(np.real(np.trace(rho.conj().T @ rho)))

        if mic is not None and mic.shape == rho.shape:
            h_obs = 0.5 * (mic + mic.T).astype(np.complex128)
            sqrt_rho = (eigvecs * np.sqrt(clipped_eigs)) @ eigvecs.conj().T
            comm = sqrt_rho @ h_obs - h_obs @ sqrt_rho
            wy_skew = float(max(0.0, -0.5 * np.real(np.trace(comm @ comm))))
        else:
            wy_skew = 0.0

        return {
            "tr_rho": float(trace.real),
            "tr_imag_abs": float(abs(trace.imag)),
            "hermitian_residual": residual,
            "min_eigenvalue": min_eig,
            "is_hermitian": residual <= _HERMITIAN_TOL,
            "is_psd": min_eig >= -_PSD_TOL,
            "is_unit_trace": bool(
                abs(trace.imag) <= _TRACE_TOL and abs(trace.real - 1.0) <= _TRACE_TOL
            ),
            "von_neumann_entropy": vn_entropy,
            "hilbert_schmidt_purity": purity,
            "wigner_yanase_skew_info": wy_skew,
        }

    @staticmethod
    def _observe_mic_passivity(mic: NDArray[np.float64]) -> Dict[str, float]:
        """
        Evalúa la pasividad eléctrica de síntesis de Brune y la cota de discos
        de Gershgorin sobre la Matriz de Interacción Central M:
          • λ_min(½(M + Mᵀ)): margen de disipación óhmica positiva-real.
          • R_Gershgorin = max_i (|M_ii| + ∑_{j≠i} |M_ij|).
        """
        sym_part = 0.5 * (mic + mic.T)
        brune_min_eig = float(np.linalg.eigvalsh(sym_part)[0])
        abs_mic = np.abs(mic)
        row_sums = np.sum(abs_mic, axis=1)
        gershgorin_bound = float(np.max(row_sums))
        return {
            "brune_passivity_min_eig": brune_min_eig,
            "gershgorin_spectral_bound": gershgorin_bound,
        }

    @staticmethod
    def _p1_payload(cert: ToonBufferProsecutorPhase1Certificate) -> Dict[str, Any]:
        return {
            "version": _AGENT_VERSION,
            "timestamp": cert.timestamp,
            "dim": cert.dim,
            "mic_sha256": cert.mic_sha256,
            "exponents_sha256": cert.exponents_sha256,
            "density_sha256": cert.density_sha256,
            "incidence_sha256": cert.incidence_sha256,
            "tr_rho": cert.tr_rho,
            "tr_imag_abs": cert.tr_imag_abs,
            "hermitian_residual": cert.hermitian_residual,
            "min_eigenvalue": cert.min_eigenvalue,
            "is_hermitian": bool(cert.is_hermitian),
            "is_psd": bool(cert.is_psd),
            "is_unit_trace": bool(cert.is_unit_trace),
            "von_neumann_entropy": cert.von_neumann_entropy,
            "hilbert_schmidt_purity": cert.hilbert_schmidt_purity,
            "wigner_yanase_skew_info": cert.wigner_yanase_skew_info,
            "brune_passivity_min_eig": cert.brune_passivity_min_eig,
            "gershgorin_spectral_bound": cert.gershgorin_spectral_bound,
        }

    @classmethod
    def _p1_digest(cls, cert: ToonBufferProsecutorPhase1Certificate) -> str:
        return _sha256_hex(_canonical_bytes(cls._p1_payload(cert)))

    def _read_clock(self) -> float:
        t = float(self._clock())
        if not math.isfinite(t):
            raise ValueError("El reloj inyectado debe devolver un flotante finito.")
        return t

    def execute_phase1(
        self,
        rho_mac: object,
        apu_cost_matrix: object,
        exponents: object,
        incidence_matrix: Optional[Any] = None,
    ) -> ToonBufferProsecutorPhase1Certificate:
        """
        ÚLTIMO método de la Fase 1 (Observe).
        Sella criptográficamente el certificado `ToonBufferProsecutorPhase1Certificate`
        que constituye la entrada directa de `continue_to_phase2_gate` (Compuerta Φ₁₂).
        """
        mic = self._ingest_cost_matrix(apu_cost_matrix)
        exps = self._ingest_exponents(exponents)
        dim = int(mic.shape[0])
        rho = self._ingest_density(rho_mac, dim)
        inc_sha = _incidence_digest(incidence_matrix)
        obs = self._observe_density(rho, mic=mic)
        elec = self._observe_mic_passivity(mic)

        draft = ToonBufferProsecutorPhase1Certificate(
            timestamp=self._read_clock(),
            dim=dim,
            mic_sha256=_array_digest(mic),
            exponents_sha256=_array_digest(exps),
            density_sha256=_array_digest(rho),
            incidence_sha256=inc_sha,
            tr_rho=obs["tr_rho"],
            tr_imag_abs=obs["tr_imag_abs"],
            hermitian_residual=obs["hermitian_residual"],
            min_eigenvalue=obs["min_eigenvalue"],
            is_hermitian=obs["is_hermitian"],
            is_psd=obs["is_psd"],
            is_unit_trace=obs["is_unit_trace"],
            von_neumann_entropy=obs["von_neumann_entropy"],
            hilbert_schmidt_purity=obs["hilbert_schmidt_purity"],
            wigner_yanase_skew_info=obs["wigner_yanase_skew_info"],
            brune_passivity_min_eig=elec["brune_passivity_min_eig"],
            gershgorin_spectral_bound=elec["gershgorin_spectral_bound"],
            payload_digest="",
            hmac_signature="",
        )
        digest = self._p1_digest(draft)
        signature = _hmac_hex(self._secret_key, _HMAC_DOMAIN_P1, digest)
        cert = replace(draft, payload_digest=digest, hmac_signature=signature)
        logger.info(
            "Fase 1 sellada: dim=%d, valid_density=%s, S_vN=%.4f, I_WY=%.4e",
            cert.dim,
            cert.is_valid_density,
            cert.von_neumann_entropy,
            cert.wigner_yanase_skew_info,
        )
        return cert


# ════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT (Mecánica Celeste de Poincaré-KAM, Electrodinámica de Dirac y RSI³)
# ════════════════════════════════════════════════════════════════════════════

class ToonBufferProsecutorPhase2Orientor(ToonBufferProsecutorPhase1Observer):
    """
    FASE 2 — ORIENT. Hereda directamente de ToonBufferProsecutorPhase1Observer.
    Métodos en orden estricto de definición:
      1. __init__                       inicializa Fase 1 y construye o inyecta el motor v15.0.0.
      2. continue_to_phase2_gate        Compuerta Homomórfica Φ₁₂. PRIMER método de Fase 2,
                                        continuación directa de `execute_phase1`.
      3. _compute_poincare_celestial_invariants (static) calcula el invariante de Tisserand-Poincaré
                                        C_T y la integral vertical de Kozai-Lidov L_z.
      4. _extract_p2_fields             (classmethod) proyecta todos los invariantes auditados
                                        del reporte del motor hacia campos sellables de P2.
      5. _p2_payload                    (static) estructura canónica sellable de P2.
      6. _p2_digest                     (classmethod) digest SHA-256 canónico de _p2_payload.
      7. execute_phase2                 invoca Φ₁₂, ejecuta la auditoría inquisitorial del motor y
                                        sella `ToonBufferProsecutorPhase2Certificate`.
                                        ÚLTIMO método de Fase 2 -> entrada directa de Fase 3 (Φ₂₃).
    """

    def __init__(
        self,
        secret_key: bytes,
        engine: Optional[BoolNovikovToonBufferProsecutorEngine] = None,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(secret_key=secret_key, clock=clock)
        self._engine: BoolNovikovToonBufferProsecutorEngine = (
            engine
            if engine is not None
            else BoolNovikovToonBufferProsecutorEngine(secret_key=self._secret_key)
        )

    def continue_to_phase2_gate(
        self,
        p1: ToonBufferProsecutorPhase1Certificate,
        apu_cost_matrix: object,
        exponents: object,
        rho_mac: object,
        incidence_matrix: Optional[Any] = None,
    ) -> None:
        """
        PRIMER método de la Fase 2 — Compuerta Homomórfica Φ₁₂ (continuación directa de `execute_phase1`).
        Verifica en modo fail-closed:
          1. Tipado e integridad estructural del certificado P1.
          2. Recomputación exacta del digest SHA-256 canónico sobre todos los campos de P1.
          3. Autenticidad de la firma HMAC-SHA256 bajo `_HMAC_DOMAIN_P1`.
          4. Validez física cuántica de ρ_MAC (hermítica, PSD, traza unitaria,
             entropía de von Neumann no negativa, pureza acotada e información de Wigner-Yanase finita).
          5. Ligadura criptográfica exacta de MIC, exponentes, ρ_MAC y matriz de incidencia.
        """
        if not isinstance(p1, ToonBufferProsecutorPhase1Certificate):
            raise ToonBufferProsecutorGateError("Φ₁₂: el objeto no es un certificado P1.")
        try:
            recomputed = self._p1_digest(p1)
        except (TypeError, ValueError) as exc:
            raise ToonBufferProsecutorGateError("Φ₁₂: campos P1 malformados.") from exc
        if not _str_equal(recomputed, p1.payload_digest):
            raise ToonBufferProsecutorGateError(
                "Φ₁₂: el digest P1 no coincide con sus campos (manipulación detectada)."
            )
        if not _verify_seal(self._secret_key, _HMAC_DOMAIN_P1, recomputed, p1.hmac_signature):
            raise ToonBufferProsecutorGateError("Φ₁₂: firma HMAC P1 inválida.")
        if not p1.is_valid_density:
            raise ToonBufferProsecutorGateError(
                "Φ₁₂: ρ_MAC no es densidad válida (hermítica, PSD, traza 1)."
            )
        if (
            not math.isfinite(p1.von_neumann_entropy)
            or p1.von_neumann_entropy < -1.0e-12
            or not math.isfinite(p1.hilbert_schmidt_purity)
            or p1.hilbert_schmidt_purity <= 0.0
            or p1.hilbert_schmidt_purity > 1.0 + 1.0e-9
            or not math.isfinite(p1.wigner_yanase_skew_info)
            or p1.wigner_yanase_skew_info < -1.0e-12
        ):
            raise ToonBufferProsecutorGateError(
                "Φ₁₂: invariantes cuánticos espectrales P1 fuera del cono físicamente admisible."
            )
        try:
            mic = self._ingest_cost_matrix(apu_cost_matrix)
            exps = self._ingest_exponents(exponents)
            rho = self._ingest_density(rho_mac, int(mic.shape[0]))
            inc_sha = _incidence_digest(incidence_matrix)
        except ValueError as exc:
            raise ToonBufferProsecutorGateError(f"Φ₁₂: entrada no admitida ({exc}).") from exc
        if p1.dim != mic.shape[0]:
            raise ToonBufferProsecutorGateError("Φ₁₂: la dimensión no coincide con P1.")
        if not _str_equal(p1.mic_sha256, _array_digest(mic)):
            raise ToonBufferProsecutorGateError("Φ₁₂: la MIC no coincide con la atestación P1.")
        if not _str_equal(p1.exponents_sha256, _array_digest(exps)):
            raise ToonBufferProsecutorGateError("Φ₁₂: los exponentes no coinciden con P1.")
        if not _str_equal(p1.density_sha256, _array_digest(rho)):
            raise ToonBufferProsecutorGateError("Φ₁₂: ρ_MAC no coincide con la atestación P1.")
        if p1.incidence_sha256 != "NONE" and incidence_matrix is not None:
            if not _str_equal(p1.incidence_sha256, inc_sha):
                raise ToonBufferProsecutorGateError(
                    "Φ₁₂: la matriz de incidencia no coincide con la atestación P1."
                )

    @staticmethod
    def _compute_poincare_celestial_invariants(
        delaunay_actions: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """
        Calcula los invariantes seculares de la mecánica celeste de Poincaré a partir
        de las variables canónicas de acción de Delaunay (L, G, H_Delaunay):
          1. Parámetro invariante de Tisserand-Poincaré en el problema restringido:
                 C_T = 1 / L² + 2 · ω_p · H_Delaunay
             donde L = √(μ a) y H_Delaunay = G cos(i) = √(μ a (1 − e²)) cos(i).
          2. Integral vertical de momento angular de Kozai-Lidov:
                 L_z = H_Delaunay / L = √(1 − e²) · cos(i) ∈ [−1, 1].
        """
        l_act, _g_act, h_act = delaunay_actions
        l_safe = max(float(l_act), 1.0e-12)
        c_tisserand = float((1.0 / (l_safe * l_safe)) + 2.0 * _PERTURBER_MEAN_MOTION_OMEGA * float(h_act))
        kozai_lidov_lz = float(np.clip(float(h_act) / l_safe, -1.0, 1.0))
        return c_tisserand, kozai_lidov_lz

    @classmethod
    def _extract_p2_fields(
        cls,
        report: ToonBufferProsecutorSpectrumReport,
        incidence_sha256: str,
    ) -> Dict[str, Any]:
        """
        Extrae todos los campos sellables del reporte espectral del motor v15.0.0,
        incluyendo los invariantes electrodinámicos de Dirac, fluidos de Zhukovsky,
        topología de Hodge-Smith, mecánica celeste de Poincaré-KAM y RSI³.
        """
        clauses = report.clauses
        dirac = report.dirac_report
        buf = report.buffer_report
        alg = report.algebra_report
        kam = report.kam_report
        rsi = report.rsi_certificate

        c_tisserand, kozai_lz = cls._compute_poincare_celestial_invariants(kam.delaunay_actions)

        return {
            "truth_value": HeytingTruthValue(report.truth_value),
            "veto_failures": tuple(str(x) for x in clauses.veto_failures),
            "degrade_failures": tuple(str(x) for x in clauses.degrade_failures),
            "crowbar_triggered": bool(dirac.crowbar_triggered),
            "photons_emitted": int(dirac.photons_emitted),
            "oseledets_lambda": float(buf.oseledets_lyapunov_lambda),
            "water_hammer_overflow": bool(buf.water_hammer_overflow),
            "banach_lip": float(report.banach_contraction_lip),
            "krull_valuation": float(report.krull_valuation),
            "gromov_rigid": not bool(report.is_gromov_violated),
            "melnikov_chaos": bool(report.melnikov_chaos_detected),
            "smith_torsion_free": bool(report.smith_torsion_free),
            "passport_hash": str(report.passport_hash),
            "incidence_sha256": str(incidence_sha256),
            # Dirac & Crowbar
            "clifford_anticommutator_defect": float(dirac.clifford_anticommutator_defect),
            "chiral_gamma5_involutory_defect": float(dirac.chiral_gamma5_involutory_defect),
            "mandelstam_s_kev2": float(dirac.mandelstam_s_kev2),
            "crowbar_peak_di_dt_a_per_us": float(dirac.crowbar_peak_di_dt_a_per_us),
            "crowbar_joule_integral_a2s": float(dirac.crowbar_joule_integral_a2s),
            # Fluidos & Topología de Hodge-Smith
            "zhukovsky_surge_pressure": float(buf.zhukovsky_surge_pressure),
            "pesin_entropy_bound": float(buf.pesin_entropy_bound),
            "torsion_subgroup_order": int(alg.torsion_subgroup_order),
            "euler_characteristic": int(alg.euler_characteristic),
            "hodge_spectral_gap": float(alg.hodge_spectral_gap),
            # Mecánica Celeste de Poincaré-KAM
            "symplectic_jacobian_defect": float(kam.symplectic_jacobian_defect),
            "poincare_cartan_drift": float(kam.poincare_cartan_invariant_drift),
            "kolmogorov_torsion_det": float(kam.kolmogorov_torsion_det),
            "arnold_isoenergetic_det": float(kam.arnold_isoenergetic_det),
            "nekhoroshev_stability_time": float(kam.nekhoroshev_stability_time),
            "floquet_monodromy_trace": float(kam.floquet_monodromy_trace),
            "is_floquet_elliptic_stable": bool(kam.is_floquet_elliptic_stable),
            "tisserand_poincare_invariant": float(c_tisserand),
            "kozai_lidov_lz_invariant": float(kozai_lz),
            # Automejora Recursiva de Nivel 3 (RSI³)
            "rsi_depth_reached": int(rsi.depth_reached),
            "rsi_meta_preconditioned_lip": float(rsi.lip_level2_meta),
            "rsi_banach_aposteriori_error": float(rsi.banach_aposteriori_error),
            "rsi_lyapunov_descent_verified": bool(rsi.lyapunov_descent_verified),
            "rsi_kleisli_associativity_defect": float(rsi.kleisli_associativity_defect),
            "rsi_kleisli_unit_defect": float(rsi.kleisli_unit_defect),
            "rsi_loeb_soundness_verified": bool(rsi.loeb_soundness_verified),
            "rsi_accepted": bool(rsi.accepted),
        }

    @staticmethod
    def _p2_payload(cert: ToonBufferProsecutorPhase2Certificate) -> Dict[str, Any]:
        return {
            "version": _AGENT_VERSION,
            "timestamp": cert.timestamp,
            "p1_payload_digest": cert.p1_payload_digest,
            "mic_sha256": cert.mic_sha256,
            "exponents_sha256": cert.exponents_sha256,
            "density_sha256": cert.density_sha256,
            "incidence_sha256": cert.incidence_sha256,
            "truth_value_code": int(cert.truth_value),
            "veto_failures": list(cert.veto_failures),
            "degrade_failures": list(cert.degrade_failures),
            "crowbar_triggered": bool(cert.crowbar_triggered),
            "photons_emitted": int(cert.photons_emitted),
            "oseledets_lambda": float(cert.oseledets_lambda),
            "water_hammer_overflow": bool(cert.water_hammer_overflow),
            "banach_lip": float(cert.banach_lip),
            "krull_valuation": float(cert.krull_valuation),
            "gromov_rigid": bool(cert.gromov_rigid),
            "melnikov_chaos": bool(cert.melnikov_chaos),
            "smith_torsion_free": bool(cert.smith_torsion_free),
            "passport_hash": cert.passport_hash,
            "clifford_anticommutator_defect": float(cert.clifford_anticommutator_defect),
            "chiral_gamma5_involutory_defect": float(cert.chiral_gamma5_involutory_defect),
            "mandelstam_s_kev2": float(cert.mandelstam_s_kev2),
            "crowbar_peak_di_dt_a_per_us": float(cert.crowbar_peak_di_dt_a_per_us),
            "crowbar_joule_integral_a2s": float(cert.crowbar_joule_integral_a2s),
            "zhukovsky_surge_pressure": float(cert.zhukovsky_surge_pressure),
            "pesin_entropy_bound": float(cert.pesin_entropy_bound),
            "torsion_subgroup_order": int(cert.torsion_subgroup_order),
            "euler_characteristic": int(cert.euler_characteristic),
            "hodge_spectral_gap": float(cert.hodge_spectral_gap),
            "symplectic_jacobian_defect": float(cert.symplectic_jacobian_defect),
            "poincare_cartan_drift": float(cert.poincare_cartan_drift),
            "kolmogorov_torsion_det": float(cert.kolmogorov_torsion_det),
            "arnold_isoenergetic_det": float(cert.arnold_isoenergetic_det),
            "nekhoroshev_stability_time": float(cert.nekhoroshev_stability_time),
            "floquet_monodromy_trace": float(cert.floquet_monodromy_trace),
            "is_floquet_elliptic_stable": bool(cert.is_floquet_elliptic_stable),
            "tisserand_poincare_invariant": float(cert.tisserand_poincare_invariant),
            "kozai_lidov_lz_invariant": float(cert.kozai_lidov_lz_invariant),
            "rsi_depth_reached": int(cert.rsi_depth_reached),
            "rsi_meta_preconditioned_lip": float(cert.rsi_meta_preconditioned_lip),
            "rsi_banach_aposteriori_error": float(cert.rsi_banach_aposteriori_error),
            "rsi_lyapunov_descent_verified": bool(cert.rsi_lyapunov_descent_verified),
            "rsi_kleisli_associativity_defect": float(cert.rsi_kleisli_associativity_defect),
            "rsi_kleisli_unit_defect": float(cert.rsi_kleisli_unit_defect),
            "rsi_loeb_soundness_verified": bool(cert.rsi_loeb_soundness_verified),
            "rsi_accepted": bool(cert.rsi_accepted),
        }

    @classmethod
    def _p2_digest(cls, cert: ToonBufferProsecutorPhase2Certificate) -> str:
        return _sha256_hex(_canonical_bytes(cls._p2_payload(cert)))

    def execute_phase2(
        self,
        p1: ToonBufferProsecutorPhase1Certificate,
        apu_cost_matrix: object,
        exponents: object,
        rho_mac: object,
        positron_token: Optional[str] = None,
        force_veto: bool = False,
        incidence_matrix: Optional[Any] = None,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
        buffer_capacity: float = _DEFAULT_BUFFER_CAPACITY,
        stage_factors: Tuple[float, float, float] = _DEFAULT_STAGE_FACTORS,
    ) -> ToonBufferProsecutorPhase2Certificate:
        """
        ÚLTIMO método de la Fase 2 (Orient).
        Ejecuta internamente la compuerta Φ₁₂ (`continue_to_phase2_gate`), invoca la
        auditoría inquisitorial completa del motor y sella criptográficamente el
        certificado `ToonBufferProsecutorPhase2Certificate` que alimenta directamente
        a `continue_to_phase3_gate` (Compuerta Φ₂₃).
        """
        self.continue_to_phase2_gate(
            p1, apu_cost_matrix, exponents, rho_mac, incidence_matrix=incidence_matrix
        )
        if positron_token is not None and not isinstance(positron_token, str):
            raise TypeError("positron_token debe ser str o None.")
        if not isinstance(force_veto, bool):
            raise TypeError("force_veto debe ser bool.")

        mic = self._ingest_cost_matrix(apu_cost_matrix)
        exps = self._ingest_exponents(exponents)
        self._ingest_density(rho_mac, int(mic.shape[0]))
        inc_sha = (
            _incidence_digest(incidence_matrix)
            if incidence_matrix is not None
            else p1.incidence_sha256
        )

        report = self._engine.execute_toon_buffer_prosecutor_audit(
            apu_cost_matrix=mic,
            exponents=exps,
            positron_token=positron_token,
            force_veto=force_veto,
            incidence_matrix=incidence_matrix,
            quant_scale=quant_scale,
            buffer_capacity=buffer_capacity,
            stage_factors=stage_factors,
        )

        draft = ToonBufferProsecutorPhase2Certificate(
            timestamp=self._read_clock(),
            p1_payload_digest=p1.payload_digest,
            mic_sha256=p1.mic_sha256,
            exponents_sha256=p1.exponents_sha256,
            density_sha256=p1.density_sha256,
            **self._extract_p2_fields(report, incidence_sha256=inc_sha),
            payload_digest="",
            hmac_signature="",
            spectrum_report=report,
        )
        digest = self._p2_digest(draft)
        signature = _hmac_hex(self._secret_key, _HMAC_DOMAIN_P2, digest)
        cert = replace(draft, payload_digest=digest, hmac_signature=signature)
        logger.info(
            "Fase 2 sellada: veredicto=%s, crowbar=%s, C_T=%.4f, RSI3_accepted=%s",
            cert.truth_value.name,
            cert.crowbar_triggered,
            cert.tisserand_poincare_invariant,
            cert.rsi_accepted,
        )
        return cert


# ════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT (Topos de Heyting Ω₃, Auditoría RSI³ y Merkle RFC 6962)
# ════════════════════════════════════════════════════════════════════════════

class ToonBufferProsecutorPhase3Decider(ToonBufferProsecutorPhase2Orientor):
    """
    FASE 3 — DECIDE & ACT. Hereda directamente de ToonBufferProsecutorPhase2Orientor.
    Métodos en orden estricto de definición:
      1. __init__                               recibe la política visceral declarada.
      2. continue_to_phase3_gate                Compuerta Homomórfica Φ₂₃. PRIMER método de Fase 3,
                                                continuación directa de `execute_phase2`.
      3. _derive_verdict_from_clauses           (static) re-derivación independiente en el álgebra de Heyting Ω₃.
      4. _audit_rsi3_meta_soundness             (static) verificación independiente de reflexión de Löb-Tarski
                                                y leyes de coherencia de la mónada de Kleisli.
      5. _audit_poincare_symplectic_soundness   (static) verificación independiente de estructura simpléctica
                                                e invariantes celestes de Poincaré-Cartan.
      6. _build_crowbar_actuation               (static) solicitud de disparo crowbar con presupuesto declarado.
      7. _build_visceral_metrics                traducción por el Funtor Semántico Visceral Φ_sem.
      8. _state_payload                         (static) estructura canónica sellable del estado soberano.
      9. execute_phase3                         invoca Φ₂₃, aplica gavificación de Lawvere-Tierney y
                                                emite `ToonBufferProsecutorSovereignState` con pruebas RFC 6962.
                                                ÚLTIMO método de Fase 3.
    """

    def __init__(
        self,
        secret_key: bytes,
        engine: Optional[BoolNovikovToonBufferProsecutorEngine] = None,
        policy: Optional[ToonBufferVisceralPolicy] = None,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(secret_key=secret_key, engine=engine, clock=clock)
        if policy is not None and not isinstance(policy, ToonBufferVisceralPolicy):
            raise TypeError("policy debe ser ToonBufferVisceralPolicy o None.")
        self._policy: ToonBufferVisceralPolicy = (
            policy if policy is not None else ToonBufferVisceralPolicy()
        )

    def continue_to_phase3_gate(self, p2: ToonBufferProsecutorPhase2Certificate) -> bool:
        """
        PRIMER método de la Fase 3 — Compuerta Homomórfica Φ₂₃ (continuación directa de `execute_phase2`).
        Verifica en modo fail-closed:
          1. Tipado e integridad criptográfica del certificado P2 (digest SHA-256 + HMAC-SHA256).
          2. Re-derivación independiente del veredicto en el álgebra de Heyting Ω₃ desde las
             listas selladas `(veto_failures, degrade_failures)`.
          3. Coherencia de aniquilación de Dirac: todo disparo `crowbar_triggered` exige `VETOED`.
          4. Auditoría meta-recursiva RSI³: verifica que no exista aceptación espuria si falla la
             contracción de Banach (NO bloquea por `banach_lip >= 1.0`, pues esa cláusula ya es
             `DEGRADED` en el motor, pero exige que la guardia de Löb-Tarski haya rechazado la
             promoción y que la falla conste en `degrade_failures`).
          5. Conservación del jacobiano simpléctico y del invariante integral de Poincaré-Cartan.
        """
        if not isinstance(p2, ToonBufferProsecutorPhase2Certificate):
            raise ToonBufferProsecutorGateError("Φ₂₃: el objeto no es un certificado P2.")
        try:
            recomputed = self._p2_digest(p2)
            derived = self._derive_verdict_from_clauses(p2.veto_failures, p2.degrade_failures)
            sealed_code = int(p2.truth_value)
        except (TypeError, ValueError) as exc:
            raise ToonBufferProsecutorGateError("Φ₂₃: campos P2 malformados.") from exc
        if not _str_equal(recomputed, p2.payload_digest):
            raise ToonBufferProsecutorGateError(
                "Φ₂₃: el digest P2 no coincide con sus campos (manipulación detectada)."
            )
        if not _verify_seal(self._secret_key, _HMAC_DOMAIN_P2, recomputed, p2.hmac_signature):
            raise ToonBufferProsecutorGateError("Φ₂₃: firma HMAC P2 inválida.")
        if int(derived) != sealed_code:
            raise ToonBufferProsecutorGateError(
                "Φ₂₃: el veredicto sellado no coincide con la re-derivación desde las cláusulas."
            )
        if p2.crowbar_triggered and sealed_code != int(HeytingTruthValue.VETOED):
            raise ToonBufferProsecutorGateError(
                "Φ₂₃: disparo Dirac registrado sin veredicto VETOED (incoherencia)."
            )
        if not self._audit_rsi3_meta_soundness(p2):
            raise ToonBufferProsecutorGateError(
                "Φ₂₃: violación de solidez meta-recursiva RSI³ (incoherencia de Löb-Tarski o Kleisli)."
            )
        if not self._audit_poincare_symplectic_soundness(p2):
            raise ToonBufferProsecutorGateError(
                "Φ₂₃: violación de estructura simpléctica o invariante de Poincaré-Cartan en P2."
            )
        return True

    @staticmethod
    def _derive_verdict_from_clauses(
        veto_failures: Tuple[str, ...],
        degrade_failures: Tuple[str, ...],
    ) -> HeytingTruthValue:
        """Ínfimo (∧) en el álgebra de Heyting Ω₃ de los grupos de cláusulas fallidas."""
        verdict = HeytingTruthValue.COHERENT
        if veto_failures:
            verdict = _meet(verdict, HeytingTruthValue.VETOED)
        if degrade_failures:
            verdict = _meet(verdict, HeytingTruthValue.DEGRADED)
        return verdict

    @staticmethod
    def _audit_rsi3_meta_soundness(p2: ToonBufferProsecutorPhase2Certificate) -> bool:
        """
        Audita independientemente la solidez del gobernador RSI³ sellado en P2:
          • Las leyes de asociatividad y unidad de la mónada de Kleisli deben cumplirse
            dentro de `_KLEISLI_COHERENCE_TOL`.
          • La guardia de reflexión de Löb-Tarski nunca puede ser violada (`rsi_loeb_soundness_verified`).
          • Si `banach_lip < 1.0`, el precondicionador cuadrático de Schulz-Picard (Nivel 2)
            debe satisfacer `rsi_meta_preconditioned_lip <= banach_lip + 1e-9`.
          • Si `banach_lip >= 1.0`, NO se bloquea la compuerta Φ₂₃ (es una degradación legítima),
            pero se verifica que el gobernador RSI³ haya ejecutado rollback (`not p2.rsi_accepted`)
            y que `"banach_kleisli_contraction"` esté registrada en `p2.degrade_failures`.
        """
        if p2.rsi_depth_reached < 1:
            return False
        if not p2.rsi_loeb_soundness_verified:
            return False
        if (
            not math.isfinite(p2.rsi_kleisli_associativity_defect)
            or p2.rsi_kleisli_associativity_defect > _KLEISLI_COHERENCE_TOL
            or not math.isfinite(p2.rsi_kleisli_unit_defect)
            or p2.rsi_kleisli_unit_defect > _KLEISLI_COHERENCE_TOL
        ):
            return False
        if p2.banach_lip < 1.0:
            if p2.rsi_meta_preconditioned_lip > p2.banach_lip + 1.0e-9:
                return False
        else:
            if p2.rsi_accepted:
                return False
            if "banach_kleisli_contraction" not in p2.degrade_failures:
                return False
        return True

    @staticmethod
    def _audit_poincare_symplectic_soundness(p2: ToonBufferProsecutorPhase2Certificate) -> bool:
        """
        Audita independientemente la exactitud simpléctica de las variables canónicas
        de Poincaré y la conservación del invariante integral relativo de Poincaré-Cartan.
        """
        if (
            not math.isfinite(p2.symplectic_jacobian_defect)
            or p2.symplectic_jacobian_defect > _SYMPLECTIC_INVARIANT_TOL
            or not math.isfinite(p2.poincare_cartan_drift)
            or p2.poincare_cartan_drift > _SYMPLECTIC_INVARIANT_TOL
            or not math.isfinite(p2.tisserand_poincare_invariant)
            or not math.isfinite(p2.kozai_lidov_lz_invariant)
            or abs(p2.kozai_lidov_lz_invariant) > 1.0 + 1.0e-9
        ):
            return False
        return True

    @staticmethod
    def _build_crowbar_actuation(
        p2: ToonBufferProsecutorPhase2Certificate,
    ) -> Tuple[bool, float, str]:
        """Política conservada: todo veredicto VETOED solicita el crowbar con presupuesto declarado."""
        triggered = HeytingTruthValue(int(p2.truth_value)) == HeytingTruthValue.VETOED
        budget = _ESP32_CROWBAR_BUDGET_NS if triggered else 0.0
        return triggered, budget, _CROWBAR_PROVENANCE

    def _build_visceral_metrics(
        self,
        p2: ToonBufferProsecutorPhase2Certificate,
    ) -> ToonBufferVisceralMetrics:
        overflow = bool(p2.water_hammer_overflow)
        lam = float(p2.oseledets_lambda)
        return ToonBufferVisceralMetrics(
            wacc_protection_pct=float(self._policy.wacc_protection_pct),
            net_liquidity_release_pct=float(self._policy.net_liquidity_release_pct),
            kv_cache_delta_pct=float(self._policy.kv_cache_delta_pct),
            water_hammer_status=f"OVERFLOW={overflow}_LAMBDA={lam:.4f}",
            provenance=_VISCERAL_PROVENANCE,
        )

    @staticmethod
    def _state_payload(
        timestamp: float,
        truth_code: int,
        lawvere_tierney_code: int,
        crowbar_triggered: bool,
        crowbar_latency_budget_ns: float,
        crowbar_latency_provenance: str,
        hardware_actuation_emitted: bool,
        rsi3_loeb_soundness_ok: bool,
        kleisli_coherence_ok: bool,
        poincare_symplectic_invariant_ok: bool,
        p2_payload_digest: str,
        visceral: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "version": _AGENT_VERSION,
            "timestamp": timestamp,
            "truth_value_code": truth_code,
            "lawvere_tierney_truth_code": lawvere_tierney_code,
            "crowbar_triggered": crowbar_triggered,
            "crowbar_latency_budget_ns": crowbar_latency_budget_ns,
            "crowbar_latency_provenance": crowbar_latency_provenance,
            "hardware_actuation_emitted": hardware_actuation_emitted,
            "rsi3_loeb_soundness_ok": rsi3_loeb_soundness_ok,
            "kleisli_coherence_ok": kleisli_coherence_ok,
            "poincare_symplectic_invariant_ok": poincare_symplectic_invariant_ok,
            "p2_payload_digest": p2_payload_digest,
            "visceral": visceral,
        }

    def execute_phase3(
        self,
        p2: ToonBufferProsecutorPhase2Certificate,
    ) -> ToonBufferProsecutorSovereignState:
        """
        ÚLTIMO método de la Fase 3 (Decide & Act).
        Ejecuta internamente la compuerta Φ₂₃ (`continue_to_phase3_gate`), evalúa la
        gavificación modal de Lawvere-Tierney j_LT(Ω₃) y emite el estado soberano
        inmutable `ToonBufferProsecutorSovereignState` respaldado por un árbol de Merkle
        RFC 6962 con pruebas criptográficas de inclusión verificadas.
        """
        self.continue_to_phase3_gate(p2)
        timestamp = self._read_clock()
        truth = HeytingTruthValue(int(p2.truth_value))
        stabilize_degraded = bool(
            p2.rsi_accepted
            and p2.rsi_loeb_soundness_verified
            and p2.is_floquet_elliptic_stable
            and not p2.crowbar_triggered
        )
        lt_truth = truth.lawvere_tierney_sheafify(stabilize_degraded=stabilize_degraded)

        crowbar, budget, provenance = self._build_crowbar_actuation(p2)
        metrics = self._build_visceral_metrics(p2)

        kleisli_ok = bool(
            p2.rsi_kleisli_associativity_defect <= _KLEISLI_COHERENCE_TOL
            and p2.rsi_kleisli_unit_defect <= _KLEISLI_COHERENCE_TOL
        )
        loeb_ok = bool(p2.rsi_loeb_soundness_verified)
        symplectic_ok = self._audit_poincare_symplectic_soundness(p2)

        payload = self._state_payload(
            timestamp=timestamp,
            truth_code=int(truth),
            lawvere_tierney_code=int(lt_truth),
            crowbar_triggered=crowbar,
            crowbar_latency_budget_ns=budget,
            crowbar_latency_provenance=provenance,
            hardware_actuation_emitted=_HARDWARE_ACTUATION_EMITTED,
            rsi3_loeb_soundness_ok=loeb_ok,
            kleisli_coherence_ok=kleisli_ok,
            poincare_symplectic_invariant_ok=symplectic_ok,
            p2_payload_digest=p2.payload_digest,
            visceral=asdict(metrics),
        )
        state_digest = _sha256_hex(_canonical_bytes(payload))
        leaves = (p2.p1_payload_digest, p2.payload_digest, state_digest)
        merkle_root = _merkle_root(leaves)
        inclusion_proofs = _merkle_inclusion_proofs(leaves)

        logger.info(
            "Fase 3 emitida: veredicto=%s, j_LT=%s, crowbar=%s, merkle=%s...",
            truth.name,
            lt_truth.name,
            crowbar,
            merkle_root[:12],
        )
        return ToonBufferProsecutorSovereignState(
            timestamp=timestamp,
            truth_value=truth,
            crowbar_triggered=crowbar,
            crowbar_latency_budget_ns=budget,
            crowbar_latency_provenance=provenance,
            hardware_actuation_emitted=_HARDWARE_ACTUATION_EMITTED,
            state_digest=state_digest,
            merkle_root_hash=merkle_root,
            visceral_metrics=metrics,
            merkle_inclusion_proofs=inclusion_proofs,
            rsi3_loeb_soundness_ok=loeb_ok,
            kleisli_coherence_ok=kleisli_ok,
            poincare_symplectic_invariant_ok=symplectic_ok,
            lawvere_tierney_truth=lt_truth,
        )


# ════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO INQUISIDOR (Orquestador de la Jerarquía Trifásica)
# ════════════════════════════════════════════════════════════════════════════

class BoolNovikovToonBufferProsecutorAgent(ToonBufferProsecutorPhase3Decider):
    """
    Soberano Agéntico Fiscal Inquisidor del TOON Buffer (v15.0.0-PhD).
    Hereda en cadena lineal las tres fases (`Phase1Observer ⊂ Phase2Orientor ⊂ Phase3Decider`).
    Su única responsabilidad propia es la emisión del token de operador humano
    para simulación/pruebas y el ciclo soberano de orquestación OODA.
    """

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        engine: Optional[BoolNovikovToonBufferProsecutorEngine] = None,
        policy: Optional[ToonBufferVisceralPolicy] = None,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(secret_key=secret_key, engine=engine, policy=policy, clock=clock)
        logger.info("%s v%s iniciado.", self.__class__.__name__, _AGENT_VERSION)

    def issue_operator_authorization(self, apu_cost_matrix: object) -> str:
        """
        Simula al operador humano (Positron) vía el motor inquisitorial.
        En producción el token lo emite un operador humano y nunca se registra en logs.
        """
        mic = self._ingest_cost_matrix(apu_cost_matrix)
        return self._engine.issue_human_positron_authorization(mic)

    def run_sovereign_governance_cycle(
        self,
        apu_cost_matrix: object,
        exponents: object,
        rho_mac_input: Optional[Any] = None,
        positron_token: Optional[str] = None,
        force_veto: bool = False,
        incidence_matrix: Optional[Any] = None,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
        buffer_capacity: float = _DEFAULT_BUFFER_CAPACITY,
        stage_factors: Tuple[float, float, float] = _DEFAULT_STAGE_FACTORS,
    ) -> ToonBufferProsecutorSovereignState:
        """Ejecuta el ciclo inquisitorial completo: Observe ➔ Φ₁₂ ➔ Orient ➔ Φ₂₃ ➔ Decide & Act."""
        mic = self._ingest_cost_matrix(apu_cost_matrix)
        dim = int(mic.shape[0])
        rho_input: Any = (
            np.eye(dim, dtype=np.complex128) / float(dim)
            if rho_mac_input is None
            else rho_mac_input
        )
        p1 = self.execute_phase1(
            rho_input,
            mic,
            exponents,
            incidence_matrix=incidence_matrix,
        )
        p2 = self.execute_phase2(
            p1,
            mic,
            exponents,
            rho_input,
            positron_token=positron_token,
            force_veto=force_veto,
            incidence_matrix=incidence_matrix,
            quant_scale=quant_scale,
            buffer_capacity=buffer_capacity,
            stage_factors=stage_factors,
        )
        return self.execute_phase3(p2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _TEST_TIME: Final[float] = 1700000000.0

    def _fixed_clock() -> float:
        return _TEST_TIME

    matrix_test = np.eye(4, dtype=np.float64) + 0.02 * np.ones((4, 4), dtype=np.float64)
    exps_test = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    path_incidence = np.array(
        [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]],
        dtype=np.int64,
    )

    agent = BoolNovikovToonBufferProsecutorAgent(clock=_fixed_clock)
    token = agent.issue_operator_authorization(matrix_test)

    # Escenario A: con token válido del operador.
    state_auth = agent.run_sovereign_governance_cycle(
        matrix_test, exps_test, positron_token=token, incidence_matrix=path_incidence,
    )
    # Escenario B: sin token. Debe ser fail-closed (VETOED + crowbar_triggered).
    state_unauth = agent.run_sovereign_governance_cycle(
        matrix_test, exps_test, incidence_matrix=path_incidence,
    )
    # Escenario C: repetición idéntica. Debe reproducir exactamente la raíz Merkle.
    state_replay = agent.run_sovereign_governance_cycle(
        matrix_test, exps_test, positron_token=token, incidence_matrix=path_incidence,
    )
    if state_replay.merkle_root_hash != state_auth.merkle_root_hash:
        raise RuntimeError("Reproducibilidad rota: la raíz Merkle difiere en la repetición.")

    # Verificación criptográfica de las 3 pruebas de inclusión de Merkle RFC 6962.
    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    p1_test = agent.execute_phase1(
        rho_test, matrix_test, exps_test, incidence_matrix=path_incidence
    )
    p2_test = agent.execute_phase2(
        p1_test, matrix_test, exps_test, rho_test,
        positron_token=token, incidence_matrix=path_incidence,
    )
    leaf_digests_test = (p1_test.payload_digest, p2_test.payload_digest, state_auth.state_digest)
    for idx, leaf_hex in enumerate(leaf_digests_test):
        if not verify_merkle_inclusion_proof(
            leaf_hex, state_auth.merkle_inclusion_proofs[idx], state_auth.merkle_root_hash
        ):
            raise RuntimeError(f"Prueba de inclusión de Merkle RFC 6962 inválida en hoja {idx}.")

    # Sabotaje 1: P1 con campo alterado. Φ₁₂ debe rechazar.
    try:
        agent.continue_to_phase2_gate(
            replace(p1_test, is_psd=False), matrix_test, exps_test, rho_test
        )
    except ToonBufferProsecutorGateError as exc:
        print(f"Sabotaje P1 detectado        : {exc}")
    else:
        raise RuntimeError("Sabotaje P1 no detectado.")

    # Sabotaje 2: P2 con veredicto falsificado. Φ₂₃ debe rechazar.
    forged_p2 = replace(p2_test, truth_value=HeytingTruthValue.VETOED)
    try:
        agent.continue_to_phase3_gate(forged_p2)
    except ToonBufferProsecutorGateError as exc:
        print(f"Sabotaje P2 detectado        : {exc}")
    else:
        raise RuntimeError("Sabotaje P2 no detectado.")

    print("=== TEST AUTÓNOMO: BoolNovikovToonBufferProsecutorAgent v15.0.0-PhD ===")
    print(f"[A] Con token  — Veredicto Ω₃ : {state_auth.truth_value.name} ({hex(int(state_auth.truth_value))})")
    print(f"[A] Gavificación j_LT(Ω₃)     : {state_auth.lawvere_tierney_truth.name}")
    print(f"[A] Crowbar                   : {state_auth.crowbar_triggered}")
    print(f"[A] Presupuesto crowbar (ns)  : {state_auth.crowbar_latency_budget_ns} ({state_auth.crowbar_latency_provenance})")
    print(f"[A] Entropía von Neumann (P1) : {p1_test.von_neumann_entropy:.6f} nats")
    print(f"[A] Info Wigner-Yanase (P1)   : {p1_test.wigner_yanase_skew_info:.6e}")
    print(f"[A] Invariante Tisserand C_T  : {p2_test.tisserand_poincare_invariant:.6f}")
    print(f"[A] Integral Kozai-Lidov L_z  : {p2_test.kozai_lidov_lz_invariant:.6f}")
    print(f"[A] Solidez RSI³ Löb / Kleisli: {state_auth.rsi3_loeb_soundness_ok} / {state_auth.kleisli_coherence_ok}")
    print(f"[A] Invariancia Simpléctica   : {state_auth.poincare_symplectic_invariant_ok}")
    print(f"[A] Status Golpe de Bélier    : {state_auth.visceral_metrics.water_hammer_status}")
    print(f"[A] Raíz Merkle RFC 6962      : {state_auth.merkle_root_hash[:16]}... (3 pruebas verificadas)")
    print(f"[B] Sin token  — Veredicto Ω₃ : {state_unauth.truth_value.name} ({hex(int(state_unauth.truth_value))})")
    print(f"[B] Crowbar                   : {state_unauth.crowbar_triggered}")
    print(f"[C] Reproducible              : {state_replay.merkle_root_hash == state_auth.merkle_root_hash}")
    print(f"Actuación hardware emitida    : {state_auth.hardware_actuation_emitted}")
    print(f"Protección WACC (%)           : {state_auth.visceral_metrics.wacc_protection_pct}")
    print(f"Liquidez Neta (%)             : {state_auth.visceral_metrics.net_liquidity_release_pct}")
    print(f"Δ KV cache (%)                : {state_auth.visceral_metrics.kv_cache_delta_pct}")
    print("=== TEST COMPLETADO CON ÉXITO ===")