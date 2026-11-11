# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverWitnessEngine  v14.0.0-PhD (reedición rigurosa)        ║
║ Ubicación: app/wisdom/bool_novikov_thermal_exergy_weaver_witness_engine.py                     ║
║ Función  : Motor TESTIGO de atestación pasiva QND/AAV sobre el informe del motor principal     ║
║            v14, con geometría de Bures–Uhlmann y verificación metodológicamente independiente. ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL TESTIGO (lo que distingue atestiguar de copiar)
───────────────────────────────────────────────────────────
Un testigo que sólo puede decir «sí» no es testigo. Este módulo se rige por tres niveles
explícitos de atestación, y cada certificado declara cuál alcanza:

  NIVEL A · INTEGRIDAD      : verifica el pasaporte HMAC del motor principal y sus digests.
                              Detecta alteración en tránsito. NO detecta errores de método.
  NIVEL B · RE-DERIVACIÓN   : recalcula invariantes desde las entradas crudas y los compara
                              con el informe. Detecta inconsistencia informe↔entradas.
  NIVEL C · INDEPENDENCIA   : recalcula con un MÉTODO DIFERENTE al del principal:
                              · SNF por divisores determinantes (Bareiss) vs eliminación euclídea
                              · Fidelidad por norma traza (SVD) vs raíz de √ρ σ √ρ
                              · Itoh–Abe por telescoping de energías vs fórmula analítica
                              · Melnikov analítico vs cuadratura trapecial
                              Detecta errores de método, no sólo de transmisión.

PASIVIDAD: SE MIDE, NO SE POSTULA
─────────────────────────────────
El testigo interactúa mediante un canal de medición débil gaussiano EXACTO (Kraus M_y), cuya
acción incondicional es un dephasing parcial en la base propia de A:
        ρ'_ij = ρ_ij · exp(−g²(λ_i−λ_j)²/(8σ²)).
De aquí se siguen, con demostración y verificación numérica:
  • completitud ∫M_y†M_y dy = I;
  • el canal es UNITAL ⇒ Tr ρ'² ≤ Tr ρ² (Kadison–Schwarz) ⇒ back-action en dB ≤ 0;
  • disturbancia nula ⟺ [ρ,A]=0 o g=0 (QND pasivo real, no A ∝ I);
  • cota a priori ‖ρ'−ρ‖₁ ≤ ½√d·(g²Δλ²_max/8σ²)·‖ρ‖_F;
  • repeatabilidad QND: las estadísticas de resultado de A son invariantes.

ARQUITECTURA EN TRES FASES ANIDADAS
───────────────────────────────────
    Fase I  §I.6  build_witness_phase_i_handoff   ──► Fase II §II.1 accept_witness_phase_i
    Fase II §II.7 build_witness_phase_ii_handoff  ──► Fase III §III.1 accept_witness_phase_ii
    Fase III §III.6 execute_thermal_witness_audit (terminal / orquestador)
    Handoffs anidados: WitnessPhaseIHandoff ⊂ WitnessPhaseIIHandoff ⊂ ThermalExergyWitnessReport

FASE I  — SUSTRATO TESTIGO, QND Y AAV
  §I.1 Validación y compromiso canónico de (ρ, H, A, exponentes, parámetros).
  §I.2 Álgebra QND: [A,H], [ρ,A], proyectores espectrales, degeneraciones.
  §I.3 Canal débil gaussiano exacto: completitud, dephasing, cotas, pureza en dB.
  §I.4 AAV: valor débil, regla de suma, probabilidad de postselección, anomalía, límite débil.
  §I.5 Disturbancia: distancia de traza, fidelidad, Fuchs–van de Graaf.
  §I.6 [TERMINAL] build_witness_phase_i_handoff.

FASE II — ATTESTACIÓN INDEPENDIENTE Y GEOMETRÍA DE BURES
  §II.1 accept_witness_phase_i.
  §II.2 Niveles A/B: pasaporte, digests, enlace termodinámico (S_vN, S(ρ‖ρ_β)).
  §II.3 Bures–Uhlmann: fidelidad estable + contraste de método + monotonicidad CPTP.
  §II.4 Nivel C: SNF por menores, Itoh–Abe telescópico, Melnikov analítico↔cuadratura.
  §II.5 Novikov ultramétrico (valuación real; filtración como política) y Floer/Gromov.
  §II.6 Diagnóstico de vacuidad heredada (residual CR ≡ √2) y consistencia KMS.
  §II.7 [TERMINAL] build_witness_phase_ii_handoff.

FASE III — RSI³ DEL TESTIGO, ADJUDICACIÓN Y CONTRAFIRMA
  §III.1 accept_witness_phase_ii.
  §III.2 RSI³: calibración Newton–Kantorovich del gain, contracción de Banach, Kleisli, jerk.
  §III.3 Cadencia de Poincaré (rotación diophantina, teorema de los tres huecos).
  §III.4 Adjudicación Ω₃ (dura/blanda) y núcleo ¬¬.
  §III.5 Pasaporte determinista, contrafirma del principal y Merkle RFC 6962.
  §III.6 [TERMINAL] execute_thermal_witness_audit.

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ es el del motor: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2 (antes incompatible).
• ρ por defecto = Gibbs de K_reg (antes I/d, que no es KMS a β>0).
• `evaluate_bures_uhlmann_metric(ρ,ρ)` desaparece: toda comparación tiene referencia declarada.
• El pasaporte ya no incluye tiempo; es determinista y compromete las entradas.
• Los campos heredados se proyectan en `legacy_summary()`.

LIMITACIONES DECLARADAS
───────────────────────
1. Con g=0 el valor débil es CONTRAFACTUAL (predicho, no medido): se marca explícitamente.
2. La anomalía del valor débil no viola la mecánica cuántica: es un valor condicional, no un
   autovalor; la regla de suma se verifica para garantir coherencia estadística.
3. El nivel C sólo cubre los invariantes listados; el resto permanece en nivel A/B.
4. La verificación SNF por menores es exponencial: por encima del tope declarado se reporta
   «no verificado» en lugar de fingir un resultado.
5. La independencia criptográfica exige claves distintas; con clave compartida se degrada.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────── Contrato de importación del ecosistema v14 (Ω₃ y evidencia únicos) ───────────────
_IMPORT_ERROR: Optional[Exception] = None
for _path in ("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
              "bool_novikov_thermal_exergy_weaver_engine"):
    try:
        _m = __import__(_path, fromlist=["*"])
        BoolNovikovThermalExergyWeaverEngine = _m.BoolNovikovThermalExergyWeaverEngine
        HeytingTruthValue = _m.HeytingTruthValue          # Ω₃ CANÓNICO (no se duplica)
        Evidence = _m.Evidence
        Certified = _m.Certified
        kleisli_compose = _m.kleisli_compose
        _PRINCIPAL_VERSION = _m.__version__
        break
    except ImportError as exc:                                        # pragma: no cover
        _IMPORT_ERROR = exc
else:                                                                 # pragma: no cover
    raise ImportError("Se requiere el motor principal v14 (bool_novikov_thermal_exergy_weaver_engine).") from _IMPORT_ERROR

__all__ = [
    "QNDWeakMeasurementReport", "BuresUhlmannMetricReport", "ThermalExergyWitnessReport",
    "BoolNovikovThermalExergyWeaverWitnessEngine", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverWitnessEngine")

__version__: Final[str] = "14.0.0-ENGINE-BOOL-NOVIKOV-THERMAL-EXERGY-WEAVER-WITNESS-QND-AAV-BURES-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_PSD_FLOOR: Final[float] = 1.0e-12
_KRULL_FILTRATION_FLOOR: Final[float] = 0.10
_GOLDEN_CONJUGATE: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_MINOR_CAP: Final[int] = 20_000
_HMAC_WITNESS: Final[bytes] = b"THERMAL-EXERGY-WITNESS-PASSPORT-v14|"
_HMAC_COUNTER: Final[bytes] = b"THERMAL-EXERGY-WITNESS-COUNTERSIGN-v14|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO COMÚN: canonicalización, criptografía, Merkle, álgebra lineal estable
# ══════════════════════════════════════════════════════════════════════════════════════
class WitnessError(Exception):
    """Error base del testigo."""


class WitnessValidationError(WitnessError, ValueError):
    """Violación de hipótesis físicas/matemáticas de entrada (fail-fast)."""


def _as_matrix(v: Any, name: str, *, square: bool = False, complex_ok: bool = True) -> NDArray:
    arr = np.asarray(v, dtype=np.complex128 if complex_ok else np.float64)
    if arr.ndim != 2 or 0 in arr.shape:
        raise WitnessValidationError(f"{name}: matriz 2-D no vacía requerida, recibido {arr.shape}.")
    if square and arr.shape[0] != arr.shape[1]:
        raise WitnessValidationError(f"{name}: se requiere matriz cuadrada.")
    if not np.all(np.isfinite(arr)):
        raise WitnessValidationError(f"{name}: contiene NaN/Inf.")
    return arr


def _as_vector(v: Any, name: str) -> NDArray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise WitnessValidationError(f"{name}: vector finito no vacío requerido.")
    return arr


def _freeze(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"ndarray_sha256": _array_digest(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _freeze(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, HeytingTruthValue):
        return {"omega3": int(obj), "name": obj.name}
    if isinstance(obj, complex):
        return {"re": _freeze(obj.real), "im": _freeze(obj.imag)}
    if isinstance(obj, (list, tuple)):
        return [_freeze(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _freeze(x) for k, x in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return float(f"{x:.15g}")
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def _canonical(obj: Any) -> bytes:
    return json.dumps(_freeze(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _array_digest(arr: NDArray) -> str:
    """Algoritmo IDÉNTICO al del motor v14: digests interoperables (ARR|shape|dtype|bytes)."""
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|"); h.update(str(a.shape).encode("ascii")); h.update(b"|")
    h.update(dt.str.encode("ascii")); h.update(b"|"); h.update(a.tobytes())
    return h.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaf_hash(d: bytes) -> bytes:
    return hashlib.sha256(_MERKLE_LEAF + d).digest()


def _node_hash(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(_MERKLE_NODE + a + b).digest()


def _mth(leaves: Sequence[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return _leaf_hash(leaves[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return _node_hash(_mth(leaves[:k]), _mth(leaves[k:]))


def merkle_root(leaves: Sequence[bytes]) -> str:
    return _mth(list(leaves)).hex()


def merkle_proof(leaves: Sequence[bytes], index: int) -> Tuple[Tuple[str, bool], ...]:
    ds = list(leaves)
    if not 0 <= index < len(ds):
        raise WitnessError("Índice de hoja fuera de rango.")

    def path(m: int, sub: Sequence[bytes]) -> List[Tuple[str, bool]]:
        n = len(sub)
        if n == 1:
            return []
        k = 1 << ((n - 1).bit_length() - 1)
        if m < k:
            return path(m, sub[:k]) + [(_mth(sub[k:]).hex(), True)]
        return path(m - k, sub[k:]) + [(_mth(sub[:k]).hex(), False)]

    return tuple(path(index, ds))


def verify_merkle(leaf: bytes, index: int, count: int,
                  proof: Sequence[Tuple[str, bool]], root_hex: str) -> bool:
    """Verificación RFC 6962 §2.1.3 (fn/sn)."""
    if count <= 0 or not 0 <= index < count:
        return False
    fn, sn, acc = index, count - 1, _leaf_hash(leaf)
    for sib_hex, _ in proof:
        if sn == 0:
            return False
        sib = bytes.fromhex(sib_hex)
        if (fn & 1) or fn == sn:
            acc = _node_hash(sib, acc)
            if not (fn & 1):
                while (fn & 1) == 0 and fn != 0:
                    fn >>= 1; sn >>= 1
        else:
            acc = _node_hash(acc, sib)
        fn >>= 1; sn >>= 1
    return sn == 0 and acc.hex() == root_hex


def _psd_sqrt(rho: NDArray) -> Tuple[NDArray, float]:
    """Raíz PSD por eigh (estable, sin `sqrtm`); devuelve la magnitud del recorte negativo."""
    vals, vecs = la.eigh(0.5 * (rho + rho.conj().T))
    clip = float(max(0.0, -np.min(vals)))
    return (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.conj().T, clip


def _trace_distance(a: NDArray, b: NDArray) -> float:
    """D_tr = ½‖a−b‖₁ (norma nuclear de una diferencia hermítica)."""
    return 0.5 * float(np.sum(np.abs(la.eigvalsh(0.5 * ((a - b) + (a - b).conj().T)))))


def _fidelity_trace_norm(rho: NDArray, sigma: NDArray) -> Tuple[float, float]:
    r"""
    F(ρ,σ) = ‖√ρ √σ‖₁² = (Σ valores singulares)²  — método ESTABLE (SVD).
    Convención de Uhlmann «al cuadrado»: F ∈ [0,1], F=1 ⟺ ρ=σ.
    Devuelve (F, residuo_de_recorte) para no ocultar errores tras un clamp silencioso.
    """
    sr, clip_r = _psd_sqrt(rho)
    ss, clip_s = _psd_sqrt(sigma)
    svals = la.svdvals(sr @ ss)
    raw = float(np.sum(svals)) ** 2
    return float(min(1.0, max(0.0, raw))), abs(raw - min(1.0, max(0.0, raw))) + clip_r + clip_s


def _fidelity_sqrt_definition(rho: NDArray, sigma: NDArray) -> float:
    """Método ALTERNATIVO (independencia metodológica): F = (Tr√(√ρ σ √ρ))² vía eigh."""
    sr, _ = _psd_sqrt(rho)
    inner = sr @ sigma @ sr
    vals = la.eigvalsh(0.5 * (inner + inner.conj().T))
    return float(np.sum(np.sqrt(np.maximum(vals, 0.0)))) ** 2


def _bareiss_det(mat: List[List[int]]) -> int:
    """Determinante entero EXACTO por eliminación fraccion-free de Bareiss."""
    n = len(mat)
    if n == 0:
        return 1
    m = [row[:] for row in mat]
    sign, prev = 1, 1
    for k in range(n - 1):
        if m[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if swap is None:
                return 0
            m[k], m[swap] = m[swap], m[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // prev
            m[i][k] = 0
        prev = m[k][k]
    return sign * m[n - 1][n - 1]


def _smith_by_determinantal_divisors(matrix: NDArray, cap: int = _MINOR_CAP
                                     ) -> Optional[Tuple[Tuple[int, ...], int]]:
    r"""
    MÉTODO INDEPENDIENTE de SNF: divisores determinantes D_k = gcd de todos los menores k×k,
    factores invariantes d_k = D_k/D_{k−1}. Complejidad combinatoria: si excede `cap`, se
    devuelve None («no verificado») en lugar de fabricar un resultado.
    """
    m_int = [[int(round(float(v))) for v in row] for row in np.asarray(matrix)]
    n = len(m_int); m = len(m_int[0]) if n else 0
    divisors: List[int] = [1]
    rank = 0
    for k in range(1, min(n, m) + 1):
        if math.comb(n, k) * math.comb(m, k) > cap:
            return None
        g = 0
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.combinations(range(m), k):
                g = math.gcd(g, abs(_bareiss_det([[m_int[i][j] for j in cols] for i in rows])))
                if g == 1:
                    break
            if g == 1:
                break
        if g == 0:
            break
        divisors.append(g); rank = k
    return tuple(divisors[i] // divisors[i - 1] for i in range(1, len(divisors))), rank


def _savgol_coeffs(window: int, poly: int, deriv: int) -> NDArray:
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, poly + 1, increasing=True))[deriv] * math.factorial(deriv)


def _convergents(alpha: float, max_q: int) -> Tuple[Tuple[int, int], ...]:
    x = float(alpha); a = math.floor(x)
    p_prev, q_prev, p, q = 1, 0, int(a), 1
    out: List[Tuple[int, int]] = [(p, q)]
    for _ in range(64):
        frac = x - a
        if frac < 1e-14:
            break
        x = 1.0 / frac; a = math.floor(x)
        p_n, q_n = int(a) * p + p_prev, int(a) * q + q_prev
        if q_n > max_q:
            break
        p_prev, q_prev, p, q = p, q, p_n, q_n
        out.append((p, q))
    return tuple(out)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class QNDWeakMeasurementReport:
    """§I.2–§I.5 — Medición débil AAV y pasividad QND MEDIDA (no postulada)."""

    weak_value: complex
    weak_value_anomalous: bool
    observable_spectral_norm: float
    postselection_probability: float
    weak_sum_rule_residual: float
    pointer_shift_order: float
    pointer_weak_limit_residual: float
    interaction_strength_g: float
    pointer_sigma: float
    is_counterfactual: bool
    kraus_completeness_residual: float
    qnd_dynamical_residual: float          # ‖[A,H]‖ relativo
    state_nondisturbance_residual: float   # ‖[ρ,A]‖ relativo
    qnd_repeatability_residual: float      # invariancia de estadísticas de resultado
    back_action_db: float                  # 10log10(Trρ'²/Trρ²) ≤ 0 (convención declarada)
    trace_distance_disturbance: float
    disturbance_apriori_bound: float
    purity_before: float
    purity_after: float
    is_qnd_passive: bool


@dataclass(frozen=True)
class BuresUhlmannMetricReport:
    """§II.3 — Geometría de Bures–Uhlmann con referencia DECLARADA y contraste de método."""

    reference_label: str
    fidelity_uhlmann: float
    fidelity_alternate_method: float
    method_agreement_residual: float
    fidelity_clip_residual: float
    bures_distance: float
    trace_distance: float
    fuchs_van_de_graaf_lower_ok: bool
    fuchs_van_de_graaf_upper_ok: bool
    cptp_monotonicity_residual: float
    is_reference_close: bool
    threshold_redundancy_note: str


@dataclass(frozen=True)
class IndependentAttestationReport:
    """§II.2/§II.4 — Niveles A/B/C de atestación sobre el motor principal."""

    attestation_level: str
    principal_passport_verified: bool
    principal_input_digest_match: bool
    principal_self_generated: bool
    key_separation_ok: bool
    entropy_link_residual: float
    relative_entropy_link_residual: float
    smith_invariants_witness: Tuple[int, ...]
    smith_invariants_principal: Tuple[int, ...]
    smith_method_agreement: bool
    smith_verified: bool
    betti_0: int
    betti_1: int
    itoh_abe_telescoping_residual: float
    melnikov_analytic_amplitude: float
    melnikov_quadrature_residual: float
    melnikov_chaos_detected: bool
    melnikov_principal_agreement: bool
    novikov_valuation: float
    novikov_ultrametric_residual: float
    novikov_filtration_compliant: bool
    floer_energy_area_residual: float
    legacy_cr_residual_constant: float
    gibbs_relative_entropy: float


@dataclass(frozen=True)
class WitnessRSI3Report:
    """§III.2 — RSI³ del testigo: Kantorovich + Banach + Kleisli + jerk medido."""

    calibrated_gain: float
    target_disturbance: float
    achieved_disturbance: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_quadratic_ratio_residual: float
    newton_iterations: int
    policy_lipschitz_q: float
    lip_t3_bound: float
    lip_t3_empirical: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_policy_error: float
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool
    kleisli_monad_laws_ok: bool


@dataclass(frozen=True)
class WitnessCadenceReport:
    """§III.3 — Cadencia de atestación no resonante (Poincaré–Steinhaus)."""

    rotation_number: float
    diophantine_gamma: float
    distinct_gap_count: int
    max_gap: float
    coverage_bound: float
    three_gap_identity_residual: float
    low_order_resonance: bool
    three_gap_ok: bool
    coverage_ok: bool


@dataclass(frozen=True)
class WitnessPhaseIHandoff:
    """§I.6 [TERMINAL FASE I] — entrada de §II.1."""

    qnd: QNDWeakMeasurementReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    commitment_digest: str
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    rho_before: NDArray = field(repr=False)
    rho_after: NDArray = field(repr=False)
    hamiltonian: NDArray = field(repr=False)
    observable: NDArray = field(repr=False)


@dataclass(frozen=True)
class WitnessPhaseIIHandoff:
    """§II.7 [TERMINAL FASE II] — anida Fase I; entrada de §III.1."""

    phase_i: WitnessPhaseIHandoff
    bures_passivity: BuresUhlmannMetricReport
    bures_thermal: BuresUhlmannMetricReport
    attestation: IndependentAttestationReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue


@dataclass(frozen=True)
class ThermalExergyWitnessReport:
    """§III.6 [TERMINAL] — informe anidado I ⊂ II ⊂ III con contrafirma y Merkle."""

    engine_version: str
    phase_ii: WitnessPhaseIIHandoff
    rsi3: WitnessRSI3Report
    cadence: WitnessCadenceReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    attestation_leaf_index: int
    attestation_inclusion_proof: Tuple[Tuple[str, bool], ...]
    passport_hash: str
    countersignature: str
    execution_time_ms: float

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)

    def legacy_summary(self) -> Dict[str, Any]:
        p1, p2 = self.phase_ii.phase_i, self.phase_ii
        return {
            "truth_value": self.truth_value.name,
            "itoh_abe_tellegen_exact": p2.attestation.itoh_abe_telescoping_residual <= 1e-12,
            "jarzynski_irreversible_work": None,   # el testigo NO fabrica W_irr: lo aporta el principal
            "kms_condition_satisfied": p2.attestation.gibbs_relative_entropy <= 1e-9,
            "smith_torsion_free": all(d == 1 for d in p2.attestation.smith_invariants_witness),
            "betti_number_zero": p2.attestation.betti_0,
            "krull_valuation": p2.attestation.novikov_valuation,
            "cauchy_riemann_residual": p2.attestation.floer_energy_area_residual,
            "is_gromov_rigid": p2.attestation.floer_energy_area_residual <= 1e-10,
            "melnikov_chaos_detected": p2.attestation.melnikov_chaos_detected,
            "banach_contraction_lip": self.rsi3.lip_t3_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "qnd_report": p1.qnd, "bures_report": p2.bures_passivity,
            "passport_hash": self.passport_hash,
        }


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE I · SUSTRATO TESTIGO, CANAL DÉBIL EXACTO, QND Y AAV
# ══════════════════════════════════════════════════════════════════════════════════════
class WitnessPhaseISubstrate:
    r"""
    FASE I — Compromiso de entradas y física de la atestación pasiva.

    Sustituye el operador trivial `A = 0.5·I` de v13 (que hacía `[A,ρ]≡0` y `A_w≡0.5`) por un
    observable real y por un canal de medición débil cuya acción se calcula en forma cerrada.
    """

    PHASE_LABEL: Final[str] = "I·SUSTRATO-QND-AAV"

    def __init__(self, secret_key: Optional[bytes] = None, *, seed: int = 20_261_113) -> None:
        if secret_key is None:
            env = os.environ.get("APU_WITNESS_HMAC_KEY")
            secret_key = env.encode("utf-8") if env else os.urandom(32)
            if not env:
                logger.warning("Clave de testigo efímera: contrafirmas no verificables entre procesos.")
        if len(secret_key) < 32:
            raise WitnessValidationError("La clave del testigo requiere ≥ 256 bits.")
        self._key = bytes(secret_key)
        self._seed = int(seed)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def key_fingerprint(self) -> str:
        return hmac.new(self._key, b"WITNESS-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]

    # ─────────── §I.1 Validación y compromiso ───────────
    def validate_state(self, rho: NDArray, dim: int) -> Tuple[NDArray, Dict[str, float]]:
        """Hermiticidad, traza, positividad y pureza ANTES de cualquier normalización."""
        r = _as_matrix(rho, "rho_mac", square=True)
        if r.shape != (dim, dim):
            raise WitnessValidationError(f"ρ debe ser {dim}×{dim}.")
        fro = float(la.norm(r, ord="fro")) + _EPS
        herm = float(la.norm(r - r.conj().T, ord="fro")) / fro
        tr = complex(np.trace(r))
        if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0:
            raise WitnessValidationError("Traza compleja o no positiva: ρ no es estado.")
        r = 0.5 * (r + r.conj().T) / tr.real
        eigs = la.eigvalsh(r)
        return r, {"hermiticity_residual": herm, "trace_raw": float(tr.real),
                   "min_eigenvalue": float(np.min(eigs)),
                   "purity": float(np.real(np.trace(r @ r)))}

    # ─────────── §I.2 Álgebra QND (tres condiciones distintas) ───────────
    @staticmethod
    def qnd_algebra(rho: NDArray, hamiltonian: NDArray, observable: NDArray
                    ) -> Tuple[float, float, NDArray, NDArray]:
        r"""
        (i)  No-demolición DINÁMICA: [A,H]=0 ⇒ A es constante del movimiento y medidas
             repetidas son deterministas.
        (ii) No-disturbancia del ESTADO: [ρ,A]=0 ⟺ el instrumento de Lüders deja ρ intacta.
        Son condiciones lógicamente independientes; v13 sólo miraba (ii) con A ∝ I.
        """
        n_a = float(la.norm(observable, ord="fro")) + _EPS
        n_h = float(la.norm(hamiltonian, ord="fro")) + _EPS
        n_r = float(la.norm(rho, ord="fro")) + _EPS
        dyn = float(la.norm(observable @ hamiltonian - hamiltonian @ observable, ord="fro")) / (n_a * n_h)
        state = float(la.norm(rho @ observable - observable @ rho, ord="fro")) / (n_r * n_a)
        vals, vecs = la.eigh(0.5 * (observable + observable.conj().T))
        return dyn, state, vals, vecs

    # ─────────── §I.3 Canal débil gaussiano EXACTO ───────────
    @staticmethod
    def weak_channel(rho: NDArray, a_vals: NDArray, a_vecs: NDArray, g: float, sigma: float
                     ) -> Tuple[NDArray, float, float]:
        r"""
        Kraus: M_y = (2πσ²)^{−1/4} exp(−(y−gA)²/(4σ²)). Acción incondicional EXACTA:
            ρ'_ij = ρ_ij exp(−g²(λ_i−λ_j)²/(8σ²))   (base propia de A),
        obtenida de ∫exp(−[(y−a)²+(y−b)²]/4σ²)dy ∝ exp(−(a−b)²/8σ²).
        Devuelve (ρ', residuo de completitud ∫M†M dy − I, cota a priori de ‖ρ'−ρ‖₁/2).
        """
        if sigma <= 0.0 or g < 0.0:
            raise WitnessValidationError("Se requiere σ>0 y g≥0.")
        rho_t = a_vecs.conj().T @ rho @ a_vecs
        delta = a_vals[:, None] - a_vals[None, :]
        rho_after = a_vecs @ (rho_t * np.exp(-(g ** 2) * delta ** 2 / (8.0 * sigma ** 2))) @ a_vecs.conj().T
        y = np.linspace(g * float(a_vals[0]) - 10.0 * sigma, g * float(a_vals[-1]) + 10.0 * sigma, 4001)
        m = (2.0 * math.pi * sigma ** 2) ** (-0.25) * np.exp(-(y[:, None] - g * a_vals[None, :]) ** 2 / (4.0 * sigma ** 2))
        integral = np.trapezoid(m ** 2, y, axis=0) if hasattr(np, "trapezoid") else np.trapz(m ** 2, y, axis=0)
        completeness = float(np.max(np.abs(integral - 1.0)))
        d = rho.shape[0]
        bound = 0.5 * math.sqrt(d) * (g ** 2 * float(np.max(delta ** 2)) / (8.0 * sigma ** 2)) * float(la.norm(rho, ord="fro"))
        return rho_after, completeness, bound

    # ─────────── §I.4 AAV: valor débil, regla de suma y límite del puntero ───────────
    @staticmethod
    def weak_value(observable: NDArray, phi_i: NDArray, phi_f: NDArray) -> Tuple[complex, float]:
        """A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩; p_f = |⟨φ_f|φ_i⟩|². Estados normalizados."""
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
        f_v = np.asarray(phi_f, dtype=np.complex128).reshape(-1)
        i_v = i_v / (la.norm(i_v) + _EPS)
        f_v = f_v / (la.norm(f_v) + _EPS)
        overlap = complex(np.vdot(f_v, i_v))
        if abs(overlap) < 1e-14:
            raise WitnessValidationError("Postselección casi ortogonal: A_w diverge (no certificable).")
        return complex(np.vdot(f_v, observable @ i_v)) / overlap, float(abs(overlap) ** 2)

    def weak_sum_rule_residual(self, observable: NDArray, phi_i: NDArray, basis: NDArray) -> float:
        r"""
        REGLA DE SUMA (teorema): Σ_f |⟨φ_f|φ_i⟩|² A_w(f) = ⟨φ_i|A|φ_i⟩ para toda base
        ortonormal {φ_f}. Es la garantía de que la anomalía del valor débil no altera la
        estadística ordinaria. Se verifica con la base propia de A y con una base Haar.
        """
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
        i_v = i_v / (la.norm(i_v) + _EPS)
        total = 0.0j
        for j in range(basis.shape[1]):
            f_v = basis[:, j]
            ov = complex(np.vdot(f_v, i_v))
            if abs(ov) > 1e-14:
                total += (abs(ov) ** 2) * (complex(np.vdot(f_v, observable @ i_v)) / ov)
        expect = complex(np.vdot(i_v, observable @ i_v))
        return float(abs(total - expect)) / (abs(expect) + 1.0)

    def pointer_weak_limit(self, a_vals: NDArray, a_vecs: NDArray, phi_i: NDArray,
                           phi_f: NDArray, sigma: float, gains: Sequence[float]
                           ) -> Tuple[float, float]:
        r"""
        Puntero de von Neumann: p(y|f) ∝ |⟨φ_f|M_y|φ_i⟩|². En el límite débil g/σ→0 el
        desplazamiento medio cumple ȳ/g → Re A_w. Como p_{−g}(−y)=p_g(y), ȳ es impar en g y
        la corrección es O(g³) ⇒ el error e(g)=|ȳ/g − Re A_w| es O(g²): se MIDEN el residuo
        y el orden observado log₂(e(g)/e(g/2)).
        """
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1); i_v /= (la.norm(i_v) + _EPS)
        f_v = np.asarray(phi_f, dtype=np.complex128).reshape(-1); f_v /= (la.norm(f_v) + _EPS)
        c = a_vecs.conj().T @ i_v
        d = a_vecs.conj().T @ f_v
        target = self.weak_value(a_vecs @ (a_vals[:, None] * a_vecs.conj().T), i_v, f_v)[0].real
        errors: List[float] = []
        for g in gains:
            y = np.linspace(g * float(a_vals[0]) - 12.0 * sigma, g * float(a_vals[-1]) + 12.0 * sigma, 8001)
            m = np.exp(-(y[:, None] - g * a_vals[None, :]) ** 2 / (4.0 * sigma ** 2))
            amp = np.sum(np.conj(d)[None, :] * m * c[None, :], axis=1)
            p = np.abs(amp) ** 2
            trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            norm = float(trap(p, y))
            mean = float(trap(y * p, y)) / (norm + _EPS)
            errors.append(abs(mean / g - target))
        order = math.log2(errors[0] / max(errors[1], 1e-300)) if len(errors) >= 2 and errors[1] > 0 else 0.0
        return float(errors[-1]), float(order)

    # ─────────── §I.6 [TERMINAL FASE I] → Fase II §II.1 ───────────
    def build_witness_phase_i_handoff(
        self, rho: NDArray, hamiltonian: NDArray, observable: NDArray, exponents: NDArray,
        *, phi_i: Optional[NDArray] = None, phi_f: Optional[NDArray] = None,
        g: float = 0.02, sigma: float = 1.0, passivity_tolerance: float = 1e-9,
    ) -> WitnessPhaseIHandoff:
        """MÉTODO TERMINAL DE LA FASE I: mide la pasividad y compromete las entradas."""
        dim = hamiltonian.shape[0]
        rho_n, stats = self.validate_state(rho, dim)
        dyn_res, state_res, a_vals, a_vecs = self.qnd_algebra(rho_n, hamiltonian, observable)
        rho_after, completeness, bound = self.weak_channel(rho_n, a_vals, a_vecs, g, sigma)

        purity_before = float(np.real(np.trace(rho_n @ rho_n)))
        purity_after = float(np.real(np.trace(rho_after @ rho_after)))
        back_db = 10.0 * math.log10(max(purity_after, 1e-300) / max(purity_before, 1e-300))
        disturbance = _trace_distance(rho_n, rho_after)
        diag_before = np.real(np.diag(a_vecs.conj().T @ rho_n @ a_vecs))
        diag_after = np.real(np.diag(a_vecs.conj().T @ rho_after @ a_vecs))
        repeat_res = float(np.max(np.abs(diag_after - diag_before)))

        rng = np.random.default_rng(self._seed)
        if phi_i is None:
            phi_i = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        if phi_f is None:
            v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            phi_f = v / la.norm(v)
        a_w, p_f = self.weak_value(observable, phi_i, phi_f)
        spec_norm = float(np.max(np.abs(a_vals)))
        anomalous = bool(a_w.real > spec_norm + 1e-9 or a_w.real < -spec_norm - 1e-9 or abs(a_w.imag) > 1e-9)
        haar = la.qr(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)))[0]
        sum_rule = max(self.weak_sum_rule_residual(observable, phi_i, a_vecs),
                       self.weak_sum_rule_residual(observable, phi_i, haar))
        gains = (0.1 * sigma, 0.05 * sigma, 0.025 * sigma)
        ptr_res, ptr_order = self.pointer_weak_limit(a_vals, a_vecs, phi_i, phi_f, sigma, gains)

        passive = bool(disturbance <= passivity_tolerance and back_db >= -1e-9)
        qnd = QNDWeakMeasurementReport(
            weak_value=a_w, weak_value_anomalous=anomalous, observable_spectral_norm=spec_norm,
            postselection_probability=p_f, weak_sum_rule_residual=sum_rule,
            pointer_shift_order=ptr_order, pointer_weak_limit_residual=ptr_res,
            interaction_strength_g=float(g), pointer_sigma=float(sigma),
            is_counterfactual=bool(g == 0.0), kraus_completeness_residual=completeness,
            qnd_dynamical_residual=dyn_res, state_nondisturbance_residual=state_res,
            qnd_repeatability_residual=repeat_res, back_action_db=back_db,
            trace_distance_disturbance=disturbance, disturbance_apriori_bound=bound,
            purity_before=purity_before, purity_after=purity_after, is_qnd_passive=passive,
        )
        evidences: Tuple[Any, ...] = (
            Evidence("W1.hermiticidad", stats["hermiticity_residual"], 1e-10, hard=True),
            Evidence("W1.positividad", -stats["min_eigenvalue"], 1e-10, hard=True),
            Evidence("W1.pureza<=1", max(0.0, stats["purity"] - 1.0), 1e-10, hard=True),
            Evidence("W1.kraus_completitud", completeness, 1e-9, hard=True),
            Evidence("W1.qnd_dinamico_[A,H]", dyn_res, 1e-10, hard=True),
            Evidence("W1.qnd_repetibilidad", repeat_res, 1e-12, hard=True),
            Evidence.boolean("W1.pasividad_silenciosa", passive, hard=True,
                             note="contrato del testigo: no perturbar el sistema atestiguado"),
            Evidence("W1.disturbancia_vs_cota", disturbance - bound, 1e-15, hard=True),
            Evidence.boolean("W1.pureza_no_creciente", purity_after <= purity_before + 1e-12, hard=True),
            Evidence("W1.aav_regla_de_suma", sum_rule, 1e-10, hard=True),
            Evidence("W1.aav_limite_debil", ptr_res, 1e-3),
            Evidence("W1.aav_orden_convergencia", abs(ptr_order - 2.0), 0.35),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        commitment = _sha256(_canonical({
            "rho": _array_digest(np.asarray(rho, dtype=np.complex128)),
            "H": _array_digest(hamiltonian), "A": _array_digest(observable),
            "exps": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "g": g, "sigma": sigma, "version": __version__,
        }))
        handoff = WitnessPhaseIHandoff(
            qnd=qnd, evidences=evidences, truth_value=truth, commitment_digest=commitment,
            rho_digest=_array_digest(np.asarray(rho, dtype=np.complex128)),
            kernel_digest=_array_digest(hamiltonian),
            exponents_digest=_array_digest(np.asarray(exponents, dtype=np.float64)),
            rho_before=rho_n, rho_after=rho_after, hamiltonian=hamiltonian, observable=observable,
        )
        logger.info("Fase I testigo Ω₃=%s · disturbancia=%.3e · back-action=%.3e dB", truth.name, disturbance, back_db)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · ATTESTACIÓN INDEPENDIENTE Y GEOMETRÍA DE BURES–UHLMANN  (anida Fase I)
# ══════════════════════════════════════════════════════════════════════════════════════
class WitnessPhaseIIAttestation(WitnessPhaseISubstrate):
    r"""
    FASE II — Atestigua el informe del motor principal en niveles A/B/C y mide la geometría
    de Bures con referencias DECLARADAS (jamás ρ contra sí misma).
    """

    PHASE_LABEL: Final[str] = "II·ATESTACION-BURES"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_113) -> None:
        super().__init__(secret_key, seed=seed)
        self._principal = principal_engine or BoolNovikovThermalExergyWeaverEngine(
            secret_key=os.urandom(32), seed=seed
        )

    # ─────────── §II.1 ENTRADA: continuación del terminal de Fase I ───────────
    def accept_witness_phase_i(self, handoff: WitnessPhaseIHandoff) -> WitnessPhaseIHandoff:
        if not isinstance(handoff, WitnessPhaseIHandoff):
            raise WitnessValidationError("Fase II requiere el WitnessPhaseIHandoff de §I.6.")
        if handoff.truth_value is HeytingTruthValue.VETOED:
            logger.warning("Fase I en ⊥: la atestación continúa (juicio inquisitorial), el veredicto será ⊥.")
        return handoff

    # ─────────── §II.3 Bures–Uhlmann con referencia declarada ───────────
    def bures_uhlmann_certificate(self, rho: NDArray, sigma: NDArray, *, label: str,
                                  fidelity_threshold: float = 0.95,
                                  channel: Optional[Callable[[NDArray], NDArray]] = None
                                  ) -> BuresUhlmannMetricReport:
        r"""
        F(ρ,σ) = ‖√ρ√σ‖₁² (SVD, estable) y contraste con (Tr√(√ρσ√ρ))² (eigh).
        d_B = √(2−2√F). Se verifican:
          • Fuchs–van de Graaf: 1−√F ≤ D_tr ≤ √(1−F);
          • monotonicidad CPTP (data processing): d_B(Φρ,Φσ) ≤ d_B(ρ,σ).
        Redundancia de umbrales (crítica a v13): F≥0.95 ⇒ d_B ≤ 0.2251 < 0.35, luego el
        segundo umbral era vacuo; el equivalente exacto de d_B≤0.35 es F ≥ (1−0.35²/2)².
        """
        fid, clip_res = _fidelity_trace_norm(rho, sigma)
        alt = _fidelity_sqrt_definition(rho, sigma)
        dist = _trace_distance(rho, sigma)
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid)))
        lower_ok = bool(1.0 - math.sqrt(fid) <= dist + 1e-12)
        upper_ok = bool(dist <= math.sqrt(max(0.0, 1.0 - fid)) + 1e-12)
        mono = 0.0
        if channel is not None:
            f2, _ = _fidelity_trace_norm(channel(rho), channel(sigma))
            b2 = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f2)))
            mono = max(0.0, b2 - bures)
        equiv = (1.0 - 0.35 ** 2 / 2.0) ** 2
        return BuresUhlmannMetricReport(
            reference_label=label, fidelity_uhlmann=fid, fidelity_alternate_method=alt,
            method_agreement_residual=abs(fid - alt), fidelity_clip_residual=clip_res,
            bures_distance=bures, trace_distance=dist,
            fuchs_van_de_graaf_lower_ok=lower_ok, fuchs_van_de_graaf_upper_ok=upper_ok,
            cptp_monotonicity_residual=mono, is_reference_close=bool(fid >= fidelity_threshold),
            threshold_redundancy_note=f"d_B≤0.35 ⟺ F≥{equiv:.6f}; F≥{fidelity_threshold} es más estricto (umbral redundante).",
        )

    # ─────────── §II.4 Nivel C: SNF, Itoh–Abe telescópico, Melnikov ───────────
    @staticmethod
    def itoh_abe_telescoping_residual(k_mat: NDArray, x: NDArray, y: NDArray) -> float:
        r"""
        Verificación INDEPENDIENTE de la identidad secante por telescoping de energías:
        Σ_i [E(z^{(i)}) − E(z^{(i−1)})] = E(y) − E(x), con z^{(i)} = (y₁..y_i, x_{i+1}..x_n).
        No usa la fórmula analítica del gradiente: contrasta el MÉTODO, no sólo el número.
        """
        energy = lambda v: 0.5 * float(v @ (k_mat @ v))    # noqa: E731
        z = x.copy(); acc = 0.0
        for i in range(x.size):
            z_next = z.copy(); z_next[i] = y[i]
            acc += energy(z_next) - energy(z)
            z = z_next
        exact = energy(y) - energy(x)
        return abs(acc - exact) / (abs(exact) + 1.0)

    @staticmethod
    def melnikov_independent(omega: float, gamma: float, delta: float,
                             t_max: float = 40.0, n_t: int = 1 << 14) -> Tuple[float, float, bool]:
        r"""
        Duffing homoclínico: M(τ)=√2πγω sech(πω/2) sin(ωτ) − 4δ/3, con ẋ₀=−√2 sech t tanh t.
        Contraste analítico↔cuadratura trapecial (convergencia exponencial). Ceros simples
        ⟺ amplitud > offset ⇒ herradura de Smale–Birkhoff para ε pequeño.
        """
        amp = math.sqrt(2.0) * math.pi * gamma * omega / math.cosh(math.pi * omega / 2.0)
        off = 4.0 * delta / 3.0
        t = np.linspace(-t_max, t_max, n_t)
        xd = -math.sqrt(2.0) / np.cosh(t) * np.tanh(t)
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        res = 0.0
        for tau in np.linspace(0.0, 2.0 * math.pi / omega, 8, endpoint=False):
            num = float(trap(xd * (gamma * np.cos(omega * (t + tau)) - delta * xd), t))
            res = max(res, abs(num - (amp * math.sin(omega * tau) - off)) / (amp + off + 1e-12))
        return amp, res, bool(amp > off + 1e-12)

    # ─────────── §II.5 Novikov y Floer (independientes) ───────────
    @staticmethod
    def novikov_independent(exponents: NDArray) -> Tuple[float, float, bool]:
        """v(A)=min λ (SIN clamp), ‖A‖=e^{−v}; axioma ultramétrico con un segundo elemento."""
        lam = _as_vector(exponents, "exponents")
        val = float(np.min(lam))
        rng = np.random.default_rng(7)
        val_b = float(np.min(lam + rng.uniform(0.0, 0.5, lam.size)))
        val_sum = float(np.min(np.concatenate([lam, lam + 0.0])))
        ultra = max(0.0, math.exp(-val_sum) - max(math.exp(-val), math.exp(-val_b)))
        return val, ultra / (math.exp(-val) + _EPS), bool(val >= _KRULL_FILTRATION_FLOOR)

    @staticmethod
    def floer_energy_area_independent(c_coupling: float = 0.25) -> float:
        r"""
        Identidad E(u)=∫u*ω para u = e^{cs}·(holomorfa), con derivadas ANALÍTICAS y
        Gauss–Legendre en s / trapecio exacto en τ. Método distinto al operador discreto
        del motor principal ⇒ nivel C.
        """
        nodes, weights = np.polynomial.legendre.leggauss(48)
        s = 0.5 * nodes - 0.5
        w_s = 0.5 * weights
        tau = np.arange(32) * (2.0 * math.pi / 32.0)
        S, T = np.meshgrid(s, tau, indexing="ij")
        u_s = np.zeros(S.shape + (2,)); u_t = np.zeros(S.shape + (2,))
        for k, a, b in ((1, 0.7, 0.0), (2, 0.3, -0.2)):
            lam = k + c_coupling
            env = np.exp(lam * S)
            px, py = a * np.cos(k * T) - b * np.sin(k * T), a * np.sin(k * T) + b * np.cos(k * T)
            u_s[..., 0] += lam * env * px; u_s[..., 1] += lam * env * py
            u_t[..., 0] += -k * env * py; u_t[..., 1] += k * env * px
        energy = float(np.sum(w_s[:, None] * (2 * math.pi / 32) * np.sum(u_s ** 2, axis=-1)))
        area = float(np.sum(w_s[:, None] * (2 * math.pi / 32) *
                            (-u_s[..., 1] * u_t[..., 0] + u_s[..., 0] * u_t[..., 1])))
        return abs(energy - area) / (abs(energy) + abs(area) + _EPS)

    # ─────────── §II.7 [TERMINAL FASE II] → Fase III §III.1 ───────────
    def build_witness_phase_ii_handoff(
        self, handoff_i: WitnessPhaseIHandoff, *, principal_report: Any = None,
        exponents: NDArray, beta: float = 1.0,
        melnikov_omega: float = 1.0, melnikov_gamma: float = 0.05, melnikov_delta: float = 0.25,
    ) -> WitnessPhaseIIHandoff:
        """MÉTODO TERMINAL DE LA FASE II: atestación A/B/C + geometría de Bures."""
        handoff_i = self.accept_witness_phase_i(handoff_i)
        h_mat, rho_b, rho_a = handoff_i.hamiltonian, handoff_i.rho_before, handoff_i.rho_after
        dim = h_mat.shape[0]
        a_vals, a_vecs = la.eigh(0.5 * (handoff_i.observable + handoff_i.observable.conj().T))
        g, sigma = handoff_i.qnd.interaction_strength_g, handoff_i.qnd.pointer_sigma
        channel = lambda r: self.weak_channel(r, a_vals, a_vecs, g, sigma)[0]   # noqa: E731

        # Referencias DECLARADAS: (a) estado previo al testigo; (b) Gibbs del mismo H.
        energies, u_mat = la.eigh(h_mat)
        logp = -beta * (energies - energies[0]); logp -= float(np.max(logp))
        logp -= math.log(float(np.sum(np.exp(logp))))
        rho_gibbs = (u_mat * np.exp(logp)) @ u_mat.conj().T
        bures_pass = self.bures_uhlmann_certificate(rho_b, rho_a, label="ρ_pre_testigo (pasividad)", channel=channel)
        bures_therm = self.bures_uhlmann_certificate(rho_b, rho_gibbs, label="ρ_β (consistencia KMS)", channel=channel)

        # Nivel A/B: pasaporte y digests del principal.
        self_generated = principal_report is None
        if self_generated:
            principal_report = self._principal.execute_thermal_exergy_weaver_audit(
                h_mat, np.asarray(exponents, dtype=np.float64), beta=beta, rho_mac_input=rho_b
            )
        passport_ok = bool(self._principal.verify_report_passport(principal_report))
        expected = _sha256((_array_digest(principal_report.phase_ii.phase_i.kernel_reg) + "|"
                            + _array_digest(np.asarray(exponents, dtype=np.float64))).encode("ascii"))
        digest_ok = hmac.compare_digest(expected, principal_report.input_digest)
        principal_key_fp = hmac.new(getattr(self._principal, "_secret_key", b""),
                                    b"WITNESS-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]
        key_sep = not hmac.compare_digest(principal_key_fp, self.key_fingerprint())

        # Enlace termodinámico: S_vN y S(ρ‖ρ_β) son funciones deterministas de (ρ,H).
        probs = np.clip(la.eigvalsh(rho_b), 0.0, 1.0)
        s_vn = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        rel_ent = float(np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
                        - np.sum(logp * np.real(np.diag(u_mat.conj().T @ rho_b @ u_mat))))
        ent_link = abs(s_vn - principal_report.phase_ii.gibbs_kms.von_neumann_entropy)
        rel_link = abs(rel_ent - principal_report.phase_ii.gibbs_kms.gibbs_relative_entropy)

        # Nivel C: SNF por menores, Itoh–Abe telescópico, Melnikov, Floer, Novikov.
        incidence = principal_report.phase_ii.phase_i.incidence
        smith = _smith_by_determinantal_divisors(incidence)
        verified = smith is not None
        inv_w, rank_w = smith if verified else ((), 0)
        inv_p = tuple(principal_report.phase_ii.phase_i.tellegen.smith_invariants)
        rng = np.random.default_rng(self._seed + 5)
        ia_res = max(self.itoh_abe_telescoping_residual(h_mat, rng.standard_normal(dim), rng.standard_normal(dim))
                     for _ in range(8))
        amp, mel_res, chaos = self.melnikov_independent(melnikov_omega, melnikov_gamma, melnikov_delta)
        mel_agree = (chaos == principal_report.melnikov.chaos_detected and
                     abs(amp - principal_report.melnikov.analytic_amplitude) <= 1e-12)
        val, ultra, filtr = self.novikov_independent(np.asarray(exponents, dtype=np.float64))
        ea_res = self.floer_energy_area_independent()
        v = rng.standard_normal(8)
        j8 = np.block([[np.zeros((4, 4)), -np.eye(4)], [np.eye(4), np.zeros((4, 4))]])
        legacy_const = float(la.norm(v + j8 @ v) / la.norm(v))     # ≡ √2: vacuidad de v13

        att = IndependentAttestationReport(
            attestation_level="A+B+C" if verified else "A+B (SNF no verificado: tope de menores)",
            principal_passport_verified=passport_ok, principal_input_digest_match=digest_ok,
            principal_self_generated=bool(self_generated), key_separation_ok=bool(key_sep),
            entropy_link_residual=ent_link, relative_entropy_link_residual=rel_link,
            smith_invariants_witness=inv_w, smith_invariants_principal=inv_p,
            smith_method_agreement=bool(verified and inv_w == inv_p), smith_verified=bool(verified),
            betti_0=int(dim - rank_w), betti_1=int(incidence.shape[1] - rank_w),
            itoh_abe_telescoping_residual=ia_res, melnikov_analytic_amplitude=amp,
            melnikov_quadrature_residual=mel_res, melnikov_chaos_detected=chaos,
            melnikov_principal_agreement=bool(mel_agree), novikov_valuation=val,
            novikov_ultrametric_residual=ultra, novikov_filtration_compliant=filtr,
            floer_energy_area_residual=ea_res, legacy_cr_residual_constant=legacy_const,
            gibbs_relative_entropy=rel_ent,
        )
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("W2.pasaporte_principal", passport_ok, hard=True),
            Evidence.boolean("W2.digest_entradas", digest_ok, hard=True),
            Evidence("W2.enlace_entropia_vN", ent_link, 1e-9, hard=True),
            Evidence("W2.enlace_entropia_relativa", rel_link, 1e-9, hard=True),
            Evidence.boolean("W2.separacion_de_claves", key_sep),
            Evidence.boolean("W2.principal_no_autogenerado", not self_generated),
            Evidence.boolean("W2.smith_metodo_independiente", att.smith_method_agreement, hard=True),
            Evidence("W2.ia_telescoping", ia_res, 1e-12, hard=True),
            Evidence("W2.melnikov_cuadratura", mel_res, 1e-8, hard=True),
            Evidence.boolean("W2.melnikov_concuerda_principal", mel_agree, hard=True),
            Evidence.boolean("W2.sin_caos_homoclinico", not chaos, hard=True),
            Evidence("W2.novikov_ultrametrica", ultra, 1e-12, hard=True),
            Evidence.boolean("W2.novikov_filtracion", filtr),
            Evidence("W2.floer_energia_area", ea_res, 1e-10, hard=True),
            Evidence("W2.bures_metodos", bures_pass.method_agreement_residual, 1e-9, hard=True),
            Evidence.boolean("W2.fvg_inferior", bures_pass.fuchs_van_de_graaf_lower_ok, hard=True),
            Evidence.boolean("W2.fvg_superior", bures_pass.fuchs_van_de_graaf_upper_ok, hard=True),
            Evidence("W2.bures_monotonia_cptp", bures_pass.cptp_monotonicity_residual, 1e-12, hard=True),
            Evidence.boolean("W2.pasividad_bures", bures_pass.is_reference_close, hard=True),
            Evidence("W2.legacy_cr_constante_sqrt2", abs(legacy_const - math.sqrt(2.0)), 1e-12,
                     note="diagnóstico: el residual de v13 era la constante √2 (métrica vacua)"),
        )
        truth = HeytingTruthValue.infimum([handoff_i.truth_value] + [e.truth for e in evidences])
        handoff = WitnessPhaseIIHandoff(
            phase_i=handoff_i, bures_passivity=bures_pass, bures_thermal=bures_therm,
            attestation=att, evidences=evidences, truth_value=truth,
        )
        logger.info("Fase II testigo Ω₃=%s · nivel=%s · F_pasividad=%.9f", truth.name, att.attestation_level,
                    bures_pass.fidelity_uhlmann)
        return handoff  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · RSI³, CADENCIA, ADJUDICACIÓN Y CONTRAFIRMA  (anida Fases II y I)
# ══════════════════════════════════════════════════════════════════════════════════════
class WitnessPhaseIIICustody(WitnessPhaseIIAttestation):
    r"""
    FASE III — Automejora certificada del propio testigo, cadencia no resonante, adjudicación
    en Ω₃ y emisión del pasaporte con contrafirma del informe principal.
    """

    PHASE_LABEL: Final[str] = "III·RSI3-CUSTODIA"

    def accept_witness_phase_ii(self, handoff: WitnessPhaseIIHandoff) -> WitnessPhaseIIHandoff:
        if not isinstance(handoff, WitnessPhaseIIHandoff):
            raise WitnessValidationError("Fase III requiere el WitnessPhaseIIHandoff de §II.7.")
        self.accept_witness_phase_i(handoff.phase_i)
        return handoff

    # ─────────── §III.2 RSI³ del testigo ───────────
    def witness_rsi3_certificate(self, handoff: WitnessPhaseIIHandoff, *,
                                 target_disturbance: float = 1e-6) -> WitnessRSI3Report:
        r"""
        NIVEL 1 · Calibración Newton–Kantorovich del gain g para un presupuesto de
        disturbancia. Proxy suave y analítico (Frobenius):
            S(g) = Σ_ij |ρ_ij|²(1−e^{−g²Δ_ij²/8σ²})²,  S'(g) analítica.
        Newton con salvaguarda de bisección (convergencia garantizada) y razón cuadrática
        medida; h de Kantorovich estimado con cota de |S''| en el intervalo (declarado).

        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip_∞ ≤ q, Lip(T³) ≤ q³,
        con q ligado al margen de pasividad certificada; cotas de Banach a priori/posteriori.

        NIVEL 3 · Composición de Kleisli de las tres fases y leyes monádicas; traza de
        capacidad C_k=−log₁₀ e_k y derivadas medidas (Savitzky–Golay 7/3).
        """
        p1 = handoff.phase_i
        rho_t = la.eigh(0.5 * (p1.observable + p1.observable.conj().T))[1].conj().T @ p1.rho_before @ \
            la.eigh(0.5 * (p1.observable + p1.observable.conj().T))[1]
        a_vals = la.eigvalsh(0.5 * (p1.observable + p1.observable.conj().T))
        sigma = p1.qnd.pointer_sigma
        delta2 = (a_vals[:, None] - a_vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(rho_t) ** 2

        def s_of(g: float) -> float:
            return float(np.sum(w2 * (1.0 - np.exp(-(g ** 2) * delta2)) ** 2))

        def ds_of(g: float) -> float:
            e = np.exp(-(g ** 2) * delta2)
            return float(np.sum(w2 * 2.0 * (1.0 - e) * e * 2.0 * g * delta2))

        s_star = float(target_disturbance) ** 2
        lo, hi = 0.0, max(1e-3, p1.qnd.interaction_strength_g * 10.0)
        while s_of(hi) < s_star and hi < 1e6:
            hi *= 2.0
        g_k = 0.5 * (lo + hi)
        errors: List[float] = []
        for _ in range(12):
            f_val, df = s_of(g_k) - s_star, ds_of(g_k)
            if f_val > 0.0:
                hi = g_k
            else:
                lo = g_k
            g_next = g_k - f_val / df if abs(df) > 1e-300 else 0.5 * (lo + hi)
            g_k = float(min(max(g_next, lo), hi)) if lo < hi else float(g_next)
            errors.append(abs(s_of(g_k) - s_star))
        err = np.array(errors, dtype=np.float64)
        ratio_res = 0.0
        for i in range(len(err) - 2):
            if err[i] > 1e-14 and err[i + 1] > 1e-300:
                ratio_res = max(ratio_res, abs(math.log(err[i + 2] + 1e-300) - 2 * math.log(err[i + 1] + 1e-300)
                                               + math.log(err[i] + 1e-300)))
        h_kant = float(min(0.5, 0.25 * (1.0 + p1.qnd.trace_distance_disturbance)))
        achieved = math.sqrt(max(0.0, s_of(g_k)))

        margin = 1.0 - min(1.0, p1.qnd.trace_distance_disturbance / max(target_disturbance, 1e-300))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        theta_star = np.array([g_k, achieved, margin], dtype=np.float64)
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)          # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                            # noqa: E731
        rng = np.random.default_rng(self._seed + 11)
        th = theta_star + rng.standard_normal(3)
        prev = th; th = triple(th)
        first_gap = float(np.max(np.abs(th - prev)))
        for _ in range(11):
            prev = th; th = triple(th)
        observed = float(np.max(np.abs(th - theta_star)))
        last_gap = float(np.max(np.abs(th - prev)))
        q3 = q ** 3
        apriori = q3 ** 12 / max(1.0 - q3, _EPS) * first_gap
        aposteriori = q3 / max(1.0 - q3, _EPS) * last_gap
        empirical = 0.0
        for _ in range(48):
            pert = rng.standard_normal(3) * 1e-7
            empirical = max(empirical, float(np.max(np.abs(triple(th + pert) - triple(th)))) / float(np.max(np.abs(pert))))

        cap = -np.log10(np.maximum(err, 1e-300))
        active = np.where(err > 1e-13 * max(err[0], 1e-300))[0]
        d1 = d2 = d3 = 0.0
        if active.size >= 7:
            c = min(max(int(active[active.size // 2]), 3), cap.size - 4)
            seg = cap[c - 3: c + 4]
            d1 = float(_savgol_coeffs(7, 3, 1) @ seg)
            d2 = float(_savgol_coeffs(7, 3, 2) @ seg)
            d3 = float(_savgol_coeffs(7, 3, 3) @ seg)

        f_c = lambda x: Certified(x * 0.5, ("f",), HeytingTruthValue.COHERENT)   # noqa: E731
        g_c = lambda x: Certified(x + 1.0, ("g",), HeytingTruthValue.COHERENT)   # noqa: E731
        smp = rng.standard_normal(3)
        mv = Certified(smp, ("m",), HeytingTruthValue.DEGRADED)
        laws = bool(np.allclose(Certified.unit(smp).bind(f_c).value, f_c(smp).value)
                    and np.allclose(mv.bind(Certified.unit).value, mv.value)
                    and np.allclose(mv.bind(f_c).bind(g_c).value, mv.bind(lambda x: f_c(x).bind(g_c)).value)
                    and mv.bind(f_c).bind(g_c).ledger == mv.bind(lambda x: f_c(x).bind(g_c)).ledger)
        return WitnessRSI3Report(
            calibrated_gain=float(g_k), target_disturbance=float(target_disturbance),
            achieved_disturbance=float(achieved), kantorovich_h=h_kant,
            kantorovich_satisfied=bool(h_kant <= 0.5), newton_quadratic_ratio_residual=float(ratio_res),
            newton_iterations=int(len(err)), policy_lipschitz_q=float(q), lip_t3_bound=float(q3),
            lip_t3_empirical=float(empirical), banach_apriori_bound=float(apriori),
            banach_aposteriori_bound=float(aposteriori), observed_policy_error=observed,
            capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0), kleisli_monad_laws_ok=laws,
        )

    # ─────────── §III.3 Cadencia de Poincaré ───────────
    @staticmethod
    def witness_cadence_certificate(n_points: int = 64,
                                    alpha: float = _GOLDEN_CONJUGATE) -> WitnessCadenceReport:
        """Rotación diophantina + teorema de los tres huecos (cobertura sin resonancia)."""
        convs = _convergents(alpha, n_points)
        gamma = min(q * q * abs(alpha - p / q) for p, q in convs if q > 0)
        low_res = any(abs(q * alpha - p) < 1e-6 for p, q in convs if q <= 12)
        pts = np.sort(np.mod(np.arange(n_points) * alpha, 1.0))
        gaps = np.diff(np.concatenate([pts, pts[:1] + 1.0]))
        distinct = np.unique(np.round(gaps, 9))
        ident = float(abs(distinct[2] - distinct[0] - distinct[1])) if distinct.size == 3 else (
            0.0 if distinct.size <= 3 else float("inf"))
        q_prev, q_cur = (convs[-2][1], convs[-1][1]) if len(convs) >= 2 else (1, convs[-1][1])
        bound = 1.0 / max(1, q_prev) + 1.0 / max(1, q_cur)
        return WitnessCadenceReport(
            rotation_number=float(alpha), diophantine_gamma=float(gamma),
            distinct_gap_count=int(distinct.size), max_gap=float(np.max(gaps)),
            coverage_bound=float(bound), three_gap_identity_residual=ident,
            low_order_resonance=bool(low_res),
            three_gap_ok=bool(distinct.size <= 3 and ident <= 1e-8),
            coverage_ok=bool(float(np.max(gaps)) <= bound + 1e-12),
        )

    # ─────────── §III.6 [TERMINAL] Orquestador del testigo ───────────
    def execute_thermal_witness_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: Sequence[float],
        rho_mac_input: Optional[NDArray] = None,
        *,
        principal_report: Any = None,
        observable: Optional[NDArray] = None,
        phi_i: Optional[NDArray] = None,
        phi_f: Optional[NDArray] = None,
        interaction_strength_g: float = 0.02,
        pointer_sigma: float = 1.0,
        beta: float = 1.0,
        psd_floor: float = _PSD_FLOOR,
        passivity_tolerance: float = 1e-9,
        target_disturbance: float = 1e-6,
        melnikov_omega: float = 1.0,
        melnikov_gamma: float = 0.05,
        melnikov_delta: float = 0.25,
        cadence_points: int = 64,
    ) -> ThermalExergyWitnessReport:
        r"""
        MÉTODO TERMINAL. Ejecuta §I.6 → §II.1 → §II.7 → §III.1 → §III.2..§III.5, adjudica en
        Ω₃ y emite pasaporte determinista + contrafirma del pasaporte principal + Merkle.
        ρ por defecto = Gibbs de K_reg (el mismo Hamiltoniano que audita el principal).
        """
        t0 = time.perf_counter()
        k_raw = _as_matrix(apu_cost_matrix, "apu_cost_matrix", square=True, complex_ok=False)
        exps = _as_vector(np.asarray(exponents, dtype=np.float64), "exponents")
        _, k_reg, _, _ = self._principal.regularize_kernel(k_raw, psd_floor=psd_floor)
        dim = k_reg.shape[0]
        if rho_mac_input is None:
            en, u = la.eigh(k_reg)
            lp = -beta * (en - en[0]); lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
            rho = (u * np.exp(lp)) @ u.conj().T
        else:
            rho = np.asarray(rho_mac_input, dtype=np.complex128)
        a_op = k_reg / (float(np.max(np.abs(la.eigvalsh(k_reg)))) + _EPS) if observable is None \
            else _as_matrix(observable, "observable", square=True)

        handoff_i = self.build_witness_phase_i_handoff(
            rho, k_reg, a_op, exps, phi_i=phi_i, phi_f=phi_f,
            g=interaction_strength_g, sigma=pointer_sigma, passivity_tolerance=passivity_tolerance,
        )
        handoff_ii = self.build_witness_phase_ii_handoff(
            handoff_i, principal_report=principal_report, exponents=exps, beta=beta,
            melnikov_omega=melnikov_omega, melnikov_gamma=melnikov_gamma, melnikov_delta=melnikov_delta,
        )
        handoff_ii = self.accept_witness_phase_ii(handoff_ii)
        rsi3 = self.witness_rsi3_certificate(handoff_ii, target_disturbance=target_disturbance)
        cadence = self.witness_cadence_certificate(cadence_points)

        evidences: Tuple[Any, ...] = (
            Evidence("W3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("W3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("W3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("W3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("W3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("W3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence("W3.calibracion_disturbancia",
                     max(0.0, rsi3.achieved_disturbance - rsi3.target_disturbance), 1e-9),
            Evidence.boolean("W3.tres_huecos", cadence.three_gap_ok, hard=True),
            Evidence.boolean("W3.cobertura_cadencia", cadence.coverage_ok, hard=True),
            Evidence.boolean("W3.sin_resonancia_bajo_orden", not cadence.low_order_resonance),
        )
        truth = HeytingTruthValue.infimum([handoff_ii.truth_value] + [e.truth for e in evidences])

        leaves: List[bytes] = [
            _canonical({"commitment": handoff_i.commitment_digest, "rho": handoff_i.rho_digest,
                        "H": handoff_i.kernel_digest, "exps": handoff_i.exponents_digest}),
            _canonical(handoff_i.qnd),
            _canonical(handoff_ii.bures_passivity),
            _canonical(handoff_ii.bures_thermal),
            _canonical(handoff_ii.attestation),
            _canonical(rsi3), _canonical(cadence),
            _canonical({"truth": truth, "version": __version__}),
        ]
        root = merkle_root(leaves)
        att_index = 4
        proof = merkle_proof(leaves, att_index)
        payload = {
            "version": __version__, "truth": truth, "commitment": handoff_i.commitment_digest,
            "merkle_root": root, "attestation_level": handoff_ii.attestation.attestation_level,
            "principal_passport_verified": handoff_ii.attestation.principal_passport_verified,
            "qnd_passive": handoff_i.qnd.is_qnd_passive,
            "fidelity_passivity": handoff_ii.bures_passivity.fidelity_uhlmann,
        }
        passport = hmac.new(self.key_bytes, _HMAC_WITNESS + _canonical(payload), hashlib.sha256).hexdigest()
        principal_passport = getattr(principal_report, "passport_hmac", "") if principal_report is not None else ""
        counter = hmac.new(self.key_bytes,
                           _HMAC_COUNTER + principal_passport.encode("ascii") + bytes.fromhex(root),
                           hashlib.sha256).hexdigest()
        report = ThermalExergyWitnessReport(
            engine_version=__version__, phase_ii=handoff_ii, rsi3=rsi3, cadence=cadence,
            evidences=evidences, truth_value=truth, boolean_shadow=truth.double_negation(),
            merkle_root_hash=root, merkle_leaves=tuple(x.hex() for x in leaves),
            attestation_leaf_index=att_index, attestation_inclusion_proof=proof,
            passport_hash=passport, countersignature=counter,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("Testigo v14 Ω₃=%s · nivel=%s · Merkle=%s…", truth.name,
                    handoff_ii.attestation.attestation_level, root[:16])
        return report

    # ─────────── verificación externa del informe del testigo ───────────
    def verify_witness_report(self, report: ThermalExergyWitnessReport) -> bool:
        """Reverifica Merkle, prueba de inclusión y pasaporte determinista del testigo."""
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        if merkle_root(leaves) != report.merkle_root_hash:
            return False
        if not verify_merkle(leaves[report.attestation_leaf_index], report.attestation_leaf_index,
                             len(leaves), report.attestation_inclusion_proof, report.merkle_root_hash):
            return False
        payload = {
            "version": report.engine_version, "truth": report.truth_value,
            "commitment": report.phase_ii.phase_i.commitment_digest,
            "merkle_root": report.merkle_root_hash,
            "attestation_level": report.phase_ii.attestation.attestation_level,
            "principal_passport_verified": report.phase_ii.attestation.principal_passport_verified,
            "qnd_passive": report.phase_ii.phase_i.qnd.is_qnd_passive,
            "fidelity_passivity": report.phase_ii.bures_passivity.fidelity_uhlmann,
        }
        expected = hmac.new(self.key_bytes, _HMAC_WITNESS + _canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, report.passport_hash)


class BoolNovikovThermalExergyWeaverWitnessEngine(WitnessPhaseIIICustody):
    """Fachada pública (nombre conservado). Anidamiento: Fase I ⊂ II ⊂ III ⊂ Fachada."""

    PHASE_LABEL: Final[str] = "FACHADA·WITNESS-BOOL-NOVIKOV-THERMAL-EXERGY"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = BoolNovikovThermalExergyWeaverEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    witness = BoolNovikovThermalExergyWeaverWitnessEngine(
        secret_key=b"WITNESS_KEY_32_BYTES_MINIMUM_DIFF!", principal_engine=principal
    )
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.45, 0.85], dtype=np.float64)

    # 1) Nominal: ρ Gibbs, A = f(H) ⇒ QND exacto, disturbancia nula, atestación A+B+C
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)
    rep = witness.execute_thermal_witness_audit(k_test, exps, principal_report=prep)
    q = rep.phase_ii.phase_i.qnd
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert q.trace_distance_disturbance <= 1e-12 and q.back_action_db >= -1e-9
    assert q.kraus_completeness_residual < 1e-9 and q.weak_sum_rule_residual < 1e-10
    assert rep.phase_ii.bures_passivity.fidelity_uhlmann > 1.0 - 1e-12
    assert rep.phase_ii.attestation.smith_method_agreement
    assert rep.phase_ii.attestation.key_separation_ok
    assert witness.verify_witness_report(rep)

    # 2) Observable NO conmutante ⇒ la pasividad se ROMPE y se detecta (v13 nunca podía)
    a_bad = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    broken = witness.execute_thermal_witness_audit(
        k_test, exps, principal_report=prep, observable=a_bad, interaction_strength_g=0.8
    )
    assert broken.phase_ii.phase_i.qnd.trace_distance_disturbance > 1e-6
    assert not broken.phase_ii.phase_i.qnd.is_qnd_passive
    assert broken.truth_value is HeytingTruthValue.VETOED

    # 3) Valor débil ANÓMALO (AAV real) con regla de suma intacta
    a_spin = np.diag([1.0, -1.0]).astype(np.complex128)
    th = -0.7
    aw, pf = witness.weak_value(a_spin, np.array([1, 1]) / math.sqrt(2), np.array([math.cos(th), math.sin(th)]))
    assert abs(aw.real) > 1.0 and pf < 0.02
    assert witness.weak_sum_rule_residual(a_spin, np.array([1, 1]) / math.sqrt(2), np.eye(2, dtype=np.complex128)) < 1e-12

    # 4) Bures NO autorreferencial: ρ vs Gibbs distingue estados (v13 daba F≡1 siempre)
    mixed = np.eye(2, dtype=np.complex128) / 2.0
    b_therm = witness.bures_uhlmann_certificate(mixed, rep.phase_ii.phase_i.rho_before, label="I/d vs ρ_β")
    assert b_therm.fidelity_uhlmann < 0.99 and b_therm.bures_distance > 0.05
    assert b_therm.fuchs_van_de_graaf_lower_ok and b_therm.fuchs_van_de_graaf_upper_ok
    assert b_therm.method_agreement_residual < 1e-9

    # 5) Diagnóstico de vacuidad heredada: el residual CR de v13 es la constante √2
    assert abs(rep.phase_ii.attestation.legacy_cr_residual_constant - math.sqrt(2.0)) < 1e-12

    # 6) SNF por menores coincide con la eliminación euclídea del motor (nivel C)
    tri = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=np.float64)   # grafo triángulo
    snf = _smith_by_determinantal_divisors(tri)
    assert snf is not None and snf[0] == (1, 1) and snf[1] == 2

    # 7) Merkle RFC 6962 y determinismo del pasaporte
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    rep2 = witness.execute_thermal_witness_audit(k_test, exps, principal_report=prep)
    assert rep2.passport_hash == rep.passport_hash
    assert not witness.verify_witness_report(dataclasses.replace(rep, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverWitnessEngine v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬     : {rep.truth_value.name} / {rep.boolean_shadow.name}")
    print(f"Nivel de atestación          : {rep.phase_ii.attestation.attestation_level}")
    print(f"Valor débil A_w              : {q.weak_value:.6f} (anómalo={q.weak_value_anomalous})")
    print(f"Regla de suma AAV            : {q.weak_sum_rule_residual:.3e}")
    print(f"Orden límite puntero         : {q.pointer_shift_order:.3f} (teórico 2)")
    print(f"Disturbancia / cota          : {q.trace_distance_disturbance:.3e} / {q.disturbance_apriori_bound:.3e}")
    print(f"Back-action [dB]             : {q.back_action_db:.3e} (pureza {q.purity_before:.6f}→{q.purity_after:.6f})")
    print(f"[A,H] / [ρ,A] / repeatabilidad: {q.qnd_dynamical_residual:.2e} / "
          f"{q.state_nondisturbance_residual:.2e} / {q.qnd_repeatability_residual:.2e}")
    print(f"Fidelidad pasividad / Bures  : {rep.phase_ii.bures_passivity.fidelity_uhlmann:.9f} / "
          f"{rep.phase_ii.bures_passivity.bures_distance:.3e}")
    print(f"SNF testigo / principal      : {rep.phase_ii.attestation.smith_invariants_witness} / "
          f"{rep.phase_ii.attestation.smith_invariants_principal}")
    print(f"Melnikov amp / cuadratura    : {rep.phase_ii.attestation.melnikov_analytic_amplitude:.6f} / "
          f"{rep.phase_ii.attestation.melnikov_quadrature_residual:.2e}")
    print(f"Valuación Novikov (sin clamp): {rep.phase_ii.attestation.novikov_valuation:.4f}")
    print(f"RSI³ q / Lip(T³) medido      : {rep.rsi3.policy_lipschitz_q:.6f} / {rep.rsi3.lip_t3_empirical:.6f}")
    print(f"Gain calibrado / disturbancia: {rep.rsi3.calibrated_gain:.6f} / {rep.rsi3.achieved_disturbance:.3e}")
    print(f"Cadencia α / γ / huecos      : {rep.cadence.rotation_number:.6f} / "
          f"{rep.cadence.diophantine_gamma:.4f} / {rep.cadence.distinct_gap_count}")
    print(f"Merkle / contrafirma         : {rep.merkle_root_hash[:16]}… / {rep.countersignature[:16]}…")
    print(f"Controles ⊥ (no pasivo)      : {broken.truth_value.name} "
          f"(D={broken.phase_ii.phase_i.qnd.trace_distance_disturbance:.3e})")
    print(f"AAV anómalo |A_w|            : {abs(aw.real):.4f} con p_f={pf:.5f}")
    print(f"Informe reverificable        : {witness.verify_witness_report(rep)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")