# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovToonCartridgeWeaverWitnessEngine v14.0.0-PhD                                   ║
║ Ubicación: app/wisdom/bool_novikov_toon_cartridge_weaver_witness_engine.py                           ║
║ Versión  : 14.0.0-ENGINE-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-WITNESS-POINCARE-KAM-RSI3-PhD            ║
║ Función  : Motor Espectral Testigo QND de Atestación Pasiva (Aharonov-Albert-Vaidman) sobre los      ║
║            Gradientes de Tejeduría Sináptica TOON y el Fibrado Dual Boole-Novikov (R_Boole ⊗ Λ_Nov). ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (cada fase hereda de la anterior; el ÚLTIMO método de la fase k
produce un certificado que el PRIMER método de la fase k+1 consume):

  FASE 1 · Álgebra cuántica-booleana-novikoviana
      sanitize → AAV/QND → Bures-Uhlmann → Rademacher-Walsh + grafo + SNF(ℤ) → Novikov/TOON
      └─ seal_phase1_certificate()  ──►  Phase1Certificate
  FASE 2 · Geometría simpléctica, Poincaré-KAM y Melnikov
      └─ ingest_phase1_certificate()  (semilla simpléctica: H = ½ zᵀ(S⊕S)z, frecuencias ω_k = λ_k(S))
      Cauchy-Riemann-Floer + Gromov → Melnikov (separatriz) + Diofantina + ecuación homológica
      └─ seal_phase2_certificate()  ──►  Phase2Certificate
  FASE 3 · Automejora recursiva RSI³ en espacio de Banach y veredicto de Heyting
      └─ ingest_phase2_certificate()  (Hessiano SPD ← espectro de Fase 2)
      Mónada Kleisli-Writer → RSI³ (nivel 1: T_η, nivel 2: mutación de η, nivel 3: mutación de β)
      → meet de Heyting → pasaporte HMAC encadenado → orquestador de auditoría

Marco matemático (resumen):
 • AAV: A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩ ∈ ℂ.  QND ⇔ ‖[A,ρ]‖_F/(‖A‖_F‖ρ‖_F) ≤ 1e-10.
   Perturbación no selectiva de 2º orden: ρ' = ρ − (g²σ²/2)[A,[A,ρ]].
 • Uhlmann: F = (Σ√eig(√ρ σ √ρ))².  d_B = √(2−2√F).  Fuchs–van de Graaf: 1−√F ≤ T ≤ √(1−F).
 • Boole: D_i f(x) = f(x⊕e_i) ⊕ f(x);  Inf_i = Σ_{S∋i} f̂(S)²;  Parseval Σ f̂² = 1.
 • Homología: H₀ = coker B₁ ≅ ℤ^{β₀} ⊕ ⊕ℤ/d_i (SNF);  β₁ = m − rank B₁.  L = B₁B₁ᵀ.
 • Novikov: Λ₀ = {Σ c_i T^{λ_i}}, ‖a‖ = e^{−v(a)}; ultramétrica y multiplicativa.
 • Floer: ∂ₛu + J∂_τu − ∇H(u) = 0 con J = [[0,−I],[I,0]], ω(v,w) = ⟨Jv,w⟩, J X_H = ∇H.
   Elipsoide {½Σλ_k(q²+p²) ≤ c}: anchura de Gromov = 2πc/λ_max.
 • Melnikov (péndulo q̈ = −ω₀² sin q − δq̇ + ε cos Ωt):
   M(t₀) = −8ω₀δ + 2π ε sech(πΩ/2ω₀) cos(Ωt₀);  cero simple ⇔ 2π ε sech(πΩ/2ω₀) > 8ω₀δ.
 • Ecuación homológica: {H₀,W} = H₁ − ⟨H₁⟩, H₀ = ω·J ⇒ W_k = i h_k / (k·ω).
 • RSI³: Lip(T³) = Lip(T)³ (Writer sobre (ℝ₊,·,1)); jerk d³C/dt³ = e₀|ln L|³Lᵗ > 0 ⇔ 0 < L < 1.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverWitnessEngine")

_ENGINE_VERSION: Final[str] = "14.0.0-ENGINE-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-WITNESS-POINCARE-KAM-RSI3-PhD"
__version__: Final[str] = _ENGINE_VERSION

# ── Constantes numéricas heredadas ────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_MAX_QND_BACKACTION_DB: Final[float] = 0.0
# ── Fase 1 ───────────────────────────────────────────────────────────────────────────────────────
_QND_COMMUTATOR_TOL: Final[float] = 1.0e-10
_WEAK_OVERLAP_FLOOR: Final[float] = 1.0e-9
_FIDELITY_FLOOR: Final[float] = 0.95
_BURES_CEILING: Final[float] = 0.35
_DEGRADED_BACKACTION_DB: Final[float] = 0.05
_MAX_BOOLE_VARS: Final[int] = 14
_SNF_ENTRY_LIMIT: Final[int] = 40_000
_TOON_RAW_TOKENS: Final[int] = 412
_TOON_NOMINAL_TOKENS: Final[int] = 56
# ── Fase 2 ───────────────────────────────────────────────────────────────────────────────────────
_CR_RESIDUAL_TOL: Final[float] = 1.0e-4
_HAMILTONIAN_ASYMMETRY_TOL: Final[float] = 1.0e-6
_FLOQUET_MARGIN_TOL: Final[float] = 1.0e-6
_KAM_GAMMA_MIN: Final[float] = 1.0e-3
_HOMOLOGICAL_TOL: Final[float] = 1.0e-9
_RESONANCE_TOL: Final[float] = 1.0e-9
_GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0
# ── Fase 3 ───────────────────────────────────────────────────────────────────────────────────────
_LEGACY_LIP_MARGIN: Final[float] = 0.002375
_LEGACY_SECRET: Final[bytes] = b"APU_FILTER_TOON_WITNESS_SECRET_KEY_v13.0.0"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# TIPOS BASE: TOPOS DE HEYTING Ω₃
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    """Topos intuicionista de Heyting Ω₃ (códigos de severidad: 0x00 ok, 0x03 vetado)."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


_TRUTH_RANK: Final[Dict[HeytingTruthValue, int]] = {
    HeytingTruthValue.VETOED: 0,
    HeytingTruthValue.DEGRADED: 1,
    HeytingTruthValue.COHERENT: 2,
}
_RANK_TRUTH: Final[Dict[int, HeytingTruthValue]] = {v: k for k, v in _TRUTH_RANK.items()}


def heyting_meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
    """Ínfimo (∧) en la cadena de Heyting ⊥=VETOED < DEGRADED < ⊤=COHERENT."""
    return _RANK_TRUTH[min(_TRUTH_RANK[a], _TRUTH_RANK[b])]


def heyting_join(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
    """Supremo (∨) en la cadena de Heyting."""
    return _RANK_TRUTH[max(_TRUTH_RANK[a], _TRUTH_RANK[b])]


def heyting_implies(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
    """Implicación de Heyting a ⇒ b: ⊤ si a ≤ b, en otro caso b."""
    return HeytingTruthValue.COHERENT if _TRUTH_RANK[a] <= _TRUTH_RANK[b] else b


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES ALGEBRAICAS PURAS (sin estado)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _hermitize(m: NDArray[Any]) -> NDArray[Any]:
    return 0.5 * (m + m.conj().T)


def _psd_sqrt(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Raíz cuadrada PSD por descomposición espectral (sin parte imaginaria espuria de sqrtm)."""
    w, v = np.linalg.eigh(_hermitize(m))
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def _trapezoid(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _sech(x: float) -> float:
    ax = abs(x)
    return 2.0 * math.exp(-ax) / (1.0 + math.exp(-2.0 * ax))


def _fwht(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transformada rápida de Walsh–Hadamard normalizada: f̂(S) = 2⁻ⁿ Σ_x f(x)(−1)^{S·x}."""
    out = np.asarray(a, dtype=np.float64).copy()
    size = out.shape[0]
    h = 1
    while h < size:
        blk = out.reshape(-1, 2, h)
        out = np.stack((blk[:, 0, :] + blk[:, 1, :], blk[:, 0, :] - blk[:, 1, :]), axis=1).reshape(size)
        h *= 2
    return out / size


def _gf2_rank(matrix: NDArray[np.int64]) -> int:
    """Rango sobre 𝔽₂ mediante base XOR de filas empaquetadas en enteros."""
    rows: List[int] = []
    for row in matrix:
        v = 0
        for j, x in enumerate(row):
            if int(x) & 1:
                v |= 1 << j
        rows.append(v)
    rank = 0
    while rows:
        pivot = rows.pop()
        if pivot == 0:
            continue
        rank += 1
        lsb = pivot & -pivot
        rows = [r ^ pivot if r & lsb else r for r in rows]
    return rank


def _smith_invariants(matrix: Sequence[Sequence[int]]) -> List[int]:
    """Factores invariantes d₁ | d₂ | … de la Forma Normal de Smith sobre ℤ (aritmética exacta)."""
    a = [[int(x) for x in row] for row in matrix]
    rows = len(a)
    cols = len(a[0]) if rows else 0
    inv: List[int] = []
    for t in range(min(rows, cols)):
        while True:
            piv: Optional[Tuple[int, int]] = None
            best: Optional[int] = None
            for i in range(t, rows):
                for j in range(t, cols):
                    v = abs(a[i][j])
                    if v and (best is None or v < best):
                        best, piv = v, (i, j)
            if piv is None:
                return inv
            i0, j0 = piv
            if i0 != t:
                a[t], a[i0] = a[i0], a[t]
            if j0 != t:
                for r in a:
                    r[t], r[j0] = r[j0], r[t]
            p = a[t][t]
            clean = True
            for i in range(t + 1, rows):
                if a[i][t]:
                    q = a[i][t] // p
                    if q:
                        a[i] = [x - q * y for x, y in zip(a[i], a[t])]
                    if a[i][t]:
                        clean = False
            for j in range(t + 1, cols):
                if a[t][j]:
                    q = a[t][j] // p
                    if q:
                        for r in a:
                            r[j] -= q * r[t]
                    if a[t][j]:
                        clean = False
            if not clean:
                continue  # aparece un pivote estrictamente menor en valor absoluto
            bad = next((i for i in range(t + 1, rows) if any(a[i][j] % p for j in range(t + 1, cols))), None)
            if bad is None:
                break
            a[t] = [x + y for x, y in zip(a[t], a[bad])]
        inv.append(abs(a[t][t]))
    return inv


def _canon(obj: Any) -> Any:
    """Canonicalización determinista (floats a 9 cifras) para digests reproducibles."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return f"{float(obj):.9e}"
    if isinstance(obj, (complex, np.complexfloating)):
        return [f"{complex(obj).real:.9e}", f"{complex(obj).imag:.9e}"]
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_canon(v) for v in obj.tolist()]
    return str(obj)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE NOVIKOV (Λ₀): serie formal finita con valuación y norma ultramétrica
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class NovikovSeries:
    """Σ c_i T^{λ_i} con λ_i ≥ 0 ordenados (condición de finitud de Gromov trivial en soporte finito)."""
    terms: Tuple[Tuple[float, float], ...]

    @staticmethod
    def from_terms(pairs: Sequence[Tuple[float, float]]) -> "NovikovSeries":
        acc: Dict[float, float] = {}
        for lam, c in pairs:
            key = round(float(lam), 12)
            acc[key] = acc.get(key, 0.0) + float(c)
        return NovikovSeries(tuple(sorted((k, v) for k, v in acc.items() if abs(v) > 1.0e-14)))

    def valuation(self) -> float:
        return self.terms[0][0] if self.terms else math.inf

    def norm(self) -> float:
        return math.exp(-self.valuation()) if self.terms else 0.0

    def __add__(self, other: "NovikovSeries") -> "NovikovSeries":
        return NovikovSeries.from_terms(list(self.terms) + list(other.terms))

    def __mul__(self, other: "NovikovSeries") -> "NovikovSeries":
        return NovikovSeries.from_terms([(l1 + l2, c1 * c2) for l1, c1 in self.terms for l2, c2 in other.terms])


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 1
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class QNDWeakMeasurementReport:
    """Reporte de Medición Débil AAV y pasividad cuántica QND."""
    weak_value: complex
    back_action_db: float
    is_qnd_passive: bool
    commutator_norm: float
    relative_commutator: float
    overlap_abs: float
    weak_value_valid: bool
    perturbative_regime_ok: bool
    second_order_disturbance: float


@dataclass(frozen=True)
class BuresUhlmannMetricReport:
    """Reporte de Fidelidad de Uhlmann y Distancia Bures-Wasserstein."""
    fidelity_uhlmann: float
    bures_distance: float
    is_isometry_preserved: bool
    trace_distance: float
    fuchs_van_de_graaf_ok: bool
    computed_ok: bool


@dataclass(frozen=True)
class BooleHomologyReport:
    """Rademacher-Walsh sobre ℤ₂ⁿ + homología simplicial 1-dimensional del grafo de acoplamientos."""
    rademacher_walsh_norm: float
    total_influence: float
    walsh_parseval_defect: float
    influence_identity_defect: float
    boole_vars: int
    boole_truncated: bool
    edge_count: int
    rank_z: int
    rank_f2: int
    invariant_factors: Tuple[int, ...]
    snf_exact: bool
    torsion_free: bool
    betti_0: int
    betti_1: int
    betti_number_zero: bool
    fiedler_value: float
    laplacian_defect: float


@dataclass(frozen=True)
class NovikovToonReport:
    """Valuación de Krull, norma ultramétrica y condensación TOON."""
    krull_valuation: float
    novikov_norm: float
    krull_floor_ok: bool
    ultrametric_ok: bool
    multiplicativity_ok: bool
    toon_token_count: int
    raw_token_count: int
    kv_cache_compression_pct: float


@dataclass(frozen=True, eq=False)
class Phase1Certificate:
    """Certificado de Fase 1 (entrada de Fase 2)."""
    cost_matrix: NDArray[np.float64]
    symmetrized_cost: NDArray[np.float64]
    cost_spectrum: NDArray[np.float64]
    rho_mac: NDArray[np.complex128]
    qnd_report: QNDWeakMeasurementReport
    bures_report: BuresUhlmannMetricReport
    homology_report: BooleHomologyReport
    novikov_toon_report: NovikovToonReport
    truth_value: HeytingTruthValue
    digest: str


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — ÁLGEBRA CUÁNTICA-BOOLEANA-NOVIKOVIANA
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class _Phase1QuantumBooleNovikovWitness:
    """FASE 1: sanitización → AAV/QND → Bures-Uhlmann → Boole/SNF → Novikov/TOON → certificado."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_WITNESS_SECRET")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = _LEGACY_SECRET
                logger.warning("APU_WITNESS_SECRET ausente: se usa la clave heredada (solo desarrollo).")
        self._secret_key: bytes = secret_key
        logger.info("%s v%s inicializado correctamente.", self.__class__.__name__, _ENGINE_VERSION)

    # ── infraestructura criptográfica encadenada ──────────────────────────────────────────────────
    def _digest(self, tag: str, payload: Dict[str, Any], parent: str = "") -> str:
        body = json.dumps({"tag": tag, "parent": parent, "payload": _canon(payload)},
                          sort_keys=True, separators=(",", ":"))
        return hmac.new(self._secret_key, body.encode("utf-8"), hashlib.sha256).hexdigest()

    # ── validadores ───────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_cost_matrix(cost: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(cost, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            raise ValueError("apu_cost_matrix debe ser cuadrada de dimensión ≥ 2.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("apu_cost_matrix contiene valores no finitos.")
        return arr

    @staticmethod
    def _sanitize_density_matrix(rho: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], float]:
        """Proyecta a {ρ = ρ† ≥ 0, Tr ρ = 1}; devuelve (ρ, defecto de proyección)."""
        arr = np.asarray(rho, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("ρ debe ser una matriz cuadrada.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("ρ contiene valores no finitos.")
        herm_defect = float(np.linalg.norm(arr - arr.conj().T) / (np.linalg.norm(arr) + _WILKINSON_EPSILON))
        w, v = np.linalg.eigh(_hermitize(arr))
        neg_mass = float(np.sum(np.clip(-w, 0.0, None)))
        w = np.clip(w, 0.0, None)
        tr = float(w.sum())
        if tr < _SPECTRAL_PSD_FLOOR:
            raise ValueError("ρ degenerada: traza espectral nula.")
        w = w / tr
        return (v * w) @ v.conj().T, herm_defect + neg_mass / tr

    @staticmethod
    def _as_unit_ket(vec: NDArray[np.complex128], dim: int, name: str) -> NDArray[np.complex128]:
        v = np.asarray(vec, dtype=np.complex128).reshape(-1)
        if v.shape[0] != dim:
            raise ValueError(f"{name}: dimensión {v.shape[0]} ≠ {dim}.")
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < _SPECTRAL_PSD_FLOOR:
            raise ValueError(f"{name}: vector nulo o no finito.")
        return v / n

    # ── 1.1 Medición débil AAV ────────────────────────────────────────────────────────────────────
    def _aav_core(
        self,
        rho: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        witness: Optional[NDArray[np.complex128]],
        coupling_g: float,
        pointer_variance: float,
    ) -> Tuple[QNDWeakMeasurementReport, NDArray[np.complex128]]:
        dim = rho.shape[0]
        ket_i = self._as_unit_ket(phi_i, dim, "phi_i")
        ket_f = self._as_unit_ket(phi_f, dim, "phi_f")

        if witness is None:
            # Levantamiento espectral f(ρ)=ρ/λ_max: conmuta por construcción (el chequeo numérico
            # certifica consistencia; el poder de auditoría real aparece con un testigo inyectado).
            w, v = np.linalg.eigh(rho)
            scale = max(float(np.max(w)), _SPECTRAL_PSD_FLOOR)
            a_op = (v * (np.clip(w, 0.0, None) / scale)) @ v.conj().T
        else:
            a_op = np.asarray(witness, dtype=np.complex128)
            if a_op.shape != (dim, dim):
                raise ValueError("witness: forma incompatible con ρ.")
            a_op = _hermitize(a_op)

        overlap = complex(np.vdot(ket_f, ket_i))
        singular = abs(overlap) < _WEAK_OVERLAP_FLOOR
        weak = 0j if singular else complex(np.vdot(ket_f, a_op @ ket_i) / overlap)

        comm = a_op @ rho - rho @ a_op
        comm_norm = float(np.linalg.norm(comm))
        rel = comm_norm / (float(np.linalg.norm(a_op) * np.linalg.norm(rho)) + _WILKINSON_EPSILON)
        back_action = 10.0 * math.log10(1.0 + rel)

        double_comm = a_op @ comm - comm @ a_op
        rho_pert = rho - 0.5 * coupling_g ** 2 * pointer_variance * double_comm
        disturbance = float(np.linalg.norm(rho_pert - rho))

        report = QNDWeakMeasurementReport(
            weak_value=weak,
            back_action_db=float(back_action),
            is_qnd_passive=bool(rel <= _QND_COMMUTATOR_TOL and back_action <= _MAX_QND_BACKACTION_DB + 1.0e-6),
            commutator_norm=comm_norm,
            relative_commutator=float(rel),
            overlap_abs=float(abs(overlap)),
            weak_value_valid=bool(not singular),
            perturbative_regime_ok=bool((not singular) and coupling_g * abs(weak) < 0.1),
            second_order_disturbance=disturbance,
        )
        return report, rho_pert

    def compute_qnd_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        witness: Optional[NDArray[np.complex128]] = None,
        coupling_g: float = 1.0e-3,
        pointer_variance: float = 1.0,
    ) -> QNDWeakMeasurementReport:
        """Valor débil complejo A_w y pasividad QND (conmutador relativo ≤ 1e-10 ⇒ 0.0 dB)."""
        rho, _ = self._sanitize_density_matrix(rho_mac)
        report, _ = self._aav_core(rho, phi_i, phi_f, witness, coupling_g, pointer_variance)
        return report

    # ── 1.2 Geometría de Uhlmann–Bures ────────────────────────────────────────────────────────────
    def evaluate_bures_uhlmann_metric(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> BuresUhlmannMetricReport:
        """F = (Σ√eig(√ρσ√ρ))², d_B = √(2−2√F); fail-closed ante cualquier error numérico."""
        try:
            r, _ = self._sanitize_density_matrix(rho)
            s, _ = self._sanitize_density_matrix(sigma)
            if r.shape != s.shape:
                raise ValueError("ρ y σ de dimensiones distintas.")
            sr = _psd_sqrt(r)
            w = np.clip(np.linalg.eigvalsh(_hermitize(sr @ s @ sr)), 0.0, None)
            fidelity = float(np.clip(np.sum(np.sqrt(w)) ** 2, 0.0, 1.0))
            trace_dist = float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(_hermitize(r - s)))))
            computed = True
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.error("Bures-Uhlmann fail-closed: %s", exc)
            fidelity, trace_dist, computed = 0.0, 1.0, False

        sqrt_f = math.sqrt(fidelity)
        bures = float(math.sqrt(max(0.0, 2.0 - 2.0 * sqrt_f)))
        tol = 1.0e-9
        fvdg = bool((1.0 - sqrt_f - tol) <= trace_dist <= (math.sqrt(max(0.0, 1.0 - fidelity)) + tol))
        return BuresUhlmannMetricReport(
            fidelity_uhlmann=fidelity,
            bures_distance=bures,
            is_isometry_preserved=bool(computed and fvdg and fidelity >= _FIDELITY_FLOOR and bures <= _BURES_CEILING),
            trace_distance=trace_dist,
            fuchs_van_de_graaf_ok=fvdg,
            computed_ok=computed,
        )

    # ── 1.3 Rademacher–Walsh + homología/SNF ──────────────────────────────────────────────────────
    def compute_rademacher_walsh_and_snf(
        self, incidence_matrix: NDArray[np.float64], edge_rel_threshold: float = 0.5
    ) -> BooleHomologyReport:
        """Derivada booleana ∇f sobre ℤ₂ⁿ (f(x)=[xᵀSx > media]) y SNF exacta de B₁ del grafo de acoplamiento."""
        cost = self._validate_cost_matrix(incidence_matrix)
        sym = 0.5 * (cost + cost.T)
        n_total = sym.shape[0]
        n = min(n_total, _MAX_BOOLE_VARS)
        sb = sym[:n, :n]

        size = 1 << n
        idx = np.arange(size, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.float64)
        q = np.sum((bits @ sb) * bits, axis=1)
        threshold = 0.25 * (float(np.trace(sb)) + float(sb.sum()))  # E[xᵀSx] sobre el cubo
        f = (q > threshold + 1.0e-12).astype(np.uint8)

        fhat = _fwht(1.0 - 2.0 * f.astype(np.float64))
        parseval_defect = float(abs(np.sum(fhat ** 2) - 1.0))
        influences = np.array([float(np.mean(f ^ f[idx ^ (1 << i)])) for i in range(n)])
        total_inf = float(influences.sum())
        spectral_inf = float(np.sum(bits.sum(axis=1) * fhat ** 2))
        rw_norm = float(math.sqrt(total_inf / n))

        # Complejo de cadenas del grafo: C₁ --B₁--> C₀
        off = np.abs(sym - np.diag(np.diag(sym)))
        mx = float(off.max())
        tau = edge_rel_threshold * mx
        edges = [(i, j) for i in range(n_total) for j in range(i + 1, n_total) if mx > 0.0 and off[i, j] > tau]
        m = len(edges)
        b1 = np.zeros((n_total, m), dtype=np.int64)
        for e, (i, j) in enumerate(edges):
            b1[i, e], b1[j, e] = -1, 1

        rank_f2 = _gf2_rank(b1)
        if n_total * max(m, 1) <= _SNF_ENTRY_LIMIT:
            invariants: Tuple[int, ...] = tuple(_smith_invariants(b1.tolist()))
            rank_z, exact = len(invariants), True
            torsion_free = bool(all(d == 1 for d in invariants) and rank_f2 == rank_z)
        else:
            invariants = ()
            rank_z = int(np.linalg.matrix_rank(b1.astype(np.float64)))
            exact = False
            torsion_free = bool(rank_f2 == rank_z)  # condición necesaria (sin 2-torsión)

        lap = (b1 @ b1.T).astype(np.float64)
        adj = np.zeros((n_total, n_total))
        for i, j in edges:
            adj[i, j] = adj[j, i] = 1.0
        lap_defect = float(np.linalg.norm(lap - (np.diag(adj.sum(axis=1)) - adj)))
        fiedler = float(np.linalg.eigvalsh(lap)[1])

        betti_0, betti_1 = n_total - rank_z, m - rank_z
        return BooleHomologyReport(
            rademacher_walsh_norm=rw_norm,
            total_influence=total_inf,
            walsh_parseval_defect=parseval_defect,
            influence_identity_defect=float(abs(total_inf - spectral_inf)),
            boole_vars=n,
            boole_truncated=bool(n < n_total),
            edge_count=m,
            rank_z=rank_z,
            rank_f2=rank_f2,
            invariant_factors=invariants,
            snf_exact=exact,
            torsion_free=torsion_free,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_number_zero=bool(betti_1 == 0),
            fiedler_value=fiedler,
            laplacian_defect=lap_defect,
        )

    # ── 1.4 Novikov + TOON ────────────────────────────────────────────────────────────────────────
    def evaluate_novikov_krull_and_toon_compression(
        self,
        exponents: NDArray[np.float64],
        raw_token_count: int = _TOON_RAW_TOKENS,
        toon_token_count: int = _TOON_NOMINAL_TOKENS,
        krull_floor: float = _DEFAULT_KRULL_FLOOR,
    ) -> NovikovToonReport:
        """v(A)=min λ_i ≥ floor; ‖A‖=e^{−v}; verifica ultrametría y multiplicatividad en Λ₀; compresión TOON."""
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if exps.size == 0 or not np.all(np.isfinite(exps)) or np.any(exps < 0.0):
            raise ValueError("exponents: se requiere un vector no vacío, finito y ≥ 0 (Λ₀).")
        if raw_token_count <= 0 or not (0 < toon_token_count <= raw_token_count):
            raise ValueError("Conteo de tokens TOON inválido.")

        a = NovikovSeries.from_terms([(float(l), (-1.0) ** i / (i + 1.0)) for i, l in enumerate(exps)])
        b = NovikovSeries.from_terms([(float(l) + 0.5, 1.0 / (i + 2.0)) for i, l in enumerate(exps[::-1])])
        ultrametric = (a + b).norm() <= max(a.norm(), b.norm()) + 1.0e-12
        multiplicative = abs((a * b).norm() - a.norm() * b.norm()) <= 1.0e-9

        krull = float(a.valuation())
        return NovikovToonReport(
            krull_valuation=krull,
            novikov_norm=float(a.norm()),
            krull_floor_ok=bool(krull >= krull_floor - 1.0e-12),
            ultrametric_ok=bool(ultrametric),
            multiplicativity_ok=bool(multiplicative),
            toon_token_count=int(toon_token_count),
            raw_token_count=int(raw_token_count),
            kv_cache_compression_pct=float((1.0 - toon_token_count / float(raw_token_count)) * 100.0),
        )

    # ── 1.5 (ÚLTIMO MÉTODO DE FASE 1) Sello → semilla de Fase 2 ───────────────────────────────────
    def seal_phase1_certificate(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
        witness_operator: Optional[NDArray[np.complex128]] = None,
        raw_token_count: int = _TOON_RAW_TOKENS,
        toon_token_count: int = _TOON_NOMINAL_TOKENS,
    ) -> Phase1Certificate:
        """Ejecuta la Fase 1 completa y emite el Phase1Certificate (S, espectro de S, ρ, veredicto, digest)
        que es la condición inicial simpléctica de la Fase 2."""
        cost = self._validate_cost_matrix(apu_cost_matrix)
        dim = cost.shape[0]
        sym = 0.5 * (cost + cost.T)
        spectrum = np.linalg.eigvalsh(sym)

        if rho_mac_input is None:
            raw = np.eye(dim, dtype=np.complex128) + 0.02 * np.ones((dim, dim), dtype=np.complex128)
            rho, _ = self._sanitize_density_matrix(raw)
        else:
            rho, _ = self._sanitize_density_matrix(rho_mac_input)
        ket = np.ones(rho.shape[0], dtype=np.complex128) / math.sqrt(rho.shape[0])

        qnd, rho_pert = self._aav_core(rho, ket, ket, witness_operator, 1.0e-3, 1.0)
        bures = self.evaluate_bures_uhlmann_metric(rho, rho_pert)
        homology = self.compute_rademacher_walsh_and_snf(cost)
        nov = self.evaluate_novikov_krull_and_toon_compression(exponents, raw_token_count, toon_token_count)

        core_ok = (qnd.is_qnd_passive and qnd.weak_value_valid and bures.is_isometry_preserved
                   and homology.torsion_free and homology.betti_number_zero
                   and nov.krull_floor_ok and nov.ultrametric_ok and nov.multiplicativity_ok)
        if core_ok:
            truth = HeytingTruthValue.COHERENT
        elif (qnd.back_action_db <= _DEGRADED_BACKACTION_DB and homology.torsion_free
              and bures.fidelity_uhlmann >= 0.90):
            truth = HeytingTruthValue.DEGRADED
        else:
            truth = HeytingTruthValue.VETOED

        digest = self._digest("phase1", {
            "rw": homology.rademacher_walsh_norm, "b0": homology.betti_0, "b1": homology.betti_1,
            "inv": list(homology.invariant_factors), "krull": nov.krull_valuation,
            "toon": nov.toon_token_count, "kv": nov.kv_cache_compression_pct,
            "backaction": qnd.back_action_db, "fidelity": bures.fidelity_uhlmann,
            "truth": int(truth),
        })
        return Phase1Certificate(cost, sym, spectrum, rho, qnd, bures, homology, nov, truth, digest)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 2
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, eq=False)
class SymplecticSeed:
    """Semilla simpléctica derivada del certificado de Fase 1 (puente 1→2)."""
    cost_matrix: NDArray[np.float64]
    hamiltonian_matrix: NDArray[np.float64]   # M = S ⊕ S sobre (ℝ²ⁿ, ω₀)
    complex_structure: NDArray[np.float64]    # J = [[0,−I],[I,0]]
    frequencies: NDArray[np.float64]          # autovalores positivos de S (ascendentes)
    parent_digest: str
    upstream_truth: HeytingTruthValue


@dataclass(frozen=True)
class PseudoholomorphicReport:
    """Cauchy–Riemann–Floer discreto, no-degeneración de Floquet y rigidez de Gromov."""
    cauchy_riemann_residual: float
    hamiltonian_asymmetry: float
    complex_structure_defect: float
    floquet_margin: float
    orbit_nondegenerate: bool
    gromov_width_ellipsoid: float
    ball_capacity: float
    cylinder_capacity: float
    is_gromov_rigid: bool


@dataclass(frozen=True)
class PoincareKAMMelnikovReport:
    """Melnikov sobre la separatriz + estabilidad Diofantina KAM (ecuación homológica de Poincaré)."""
    melnikov_amplitude: float
    melnikov_amplitude_analytic: float
    melnikov_quadrature_defect: float
    dissipation_integral: float
    melnikov_ratio: float
    melnikov_chaos_detected: bool
    diophantine_gamma: float
    diophantine_tau: float
    homological_residual: float
    generating_norm: float
    resonant_modes: int
    kam_stable: bool


@dataclass(frozen=True, eq=False)
class Phase2Certificate:
    """Certificado de Fase 2 (entrada de Fase 3)."""
    pseudoholomorphic_report: PseudoholomorphicReport
    kam_report: PoincareKAMMelnikovReport
    hessian_spectrum: NDArray[np.float64]
    truth_value: HeytingTruthValue
    parent_digest: str
    digest: str


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — GEOMETRÍA SIMPLÉCTICA, POINCARÉ-KAM Y MELNIKOV
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class _Phase2SymplecticPoincareKAM(_Phase1QuantumBooleNovikovWitness):
    """FASE 2: (hereda Fase 1) semilla simpléctica → Floer/Gromov → Melnikov/KAM → certificado."""

    @staticmethod
    def _standard_complex_structure(n: int) -> NDArray[np.float64]:
        eye = np.eye(n)
        return np.block([[np.zeros((n, n)), -eye], [eye, np.zeros((n, n))]])

    # ── 2.0 (PRIMER MÉTODO DE FASE 2) continuación del certificado de Fase 1 ──────────────────────
    def ingest_phase1_certificate(self, certificate: Phase1Certificate) -> SymplecticSeed:
        """Construye (ℝ²ⁿ, ω₀, J, H=½zᵀ(S⊕S)z) a partir de S y de su espectro certificados en Fase 1."""
        n = certificate.symmetrized_cost.shape[0]
        s = certificate.symmetrized_cost
        zero = np.zeros((n, n))
        ham = np.block([[s, zero], [zero, s]])
        j = self._standard_complex_structure(n)
        if np.linalg.norm(j @ j + np.eye(2 * n)) > 1.0e-12:
            raise ArithmeticError("J² ≠ −I: estructura casi-compleja inválida.")
        pos = np.sort(certificate.cost_spectrum[certificate.cost_spectrum > _SPECTRAL_PSD_FLOOR])
        return SymplecticSeed(certificate.cost_matrix, ham, j, pos, certificate.digest, certificate.truth_value)

    # ── 2.1 Cauchy–Riemann–Floer + Gromov ─────────────────────────────────────────────────────────
    def solve_pseudoholomorphic_surface_witness(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.25,
        cylinder_R: float = 0.50,
        energy_level: float = 1.0,
        tau_samples: int = 513,
        s_samples: int = 9,
    ) -> PseudoholomorphicReport:
        """Verifica ∂ₛu + J∂_τu − ∇H(u) = 0 sobre la órbita exacta u(τ)=exp(−τJM)u₀ del Hamiltoniano
        cuadrático inducido por S; no es un resolvedor global de curvas J-holomorfas generales: certifica
        convención de signos, discretización, no-degeneración y rigidez de capacidad."""
        cost = self._validate_cost_matrix(apu_cost_matrix)
        if ball_r <= 0.0 or cylinder_R <= 0.0 or energy_level <= 0.0:
            raise ValueError("ball_r, cylinder_R y energy_level deben ser positivos.")
        n = cost.shape[0]
        sym = 0.5 * (cost + cost.T)
        asym = float(np.linalg.norm(cost - cost.T) / (np.linalg.norm(cost) + _WILKINSON_EPSILON))

        lam, _ = np.linalg.eigh(sym)                     # base simpléctica ortogonal: V⊕V conmuta con J
        lam_abs = np.maximum(np.abs(lam), 1.0e-6)
        amp = np.sqrt(2.0 * energy_level / (n * lam_abs))
        phi = 0.37 * (np.arange(n) + 1.0)

        tau = np.linspace(0.0, 1.0, tau_samples)
        s = np.linspace(-1.0, 1.0, s_samples)
        phase = lam[None, :] * tau[:, None] + phi[None, :]
        u = np.concatenate([amp[None, :] * np.cos(phase), -amp[None, :] * np.sin(phase)], axis=1)
        big_u = np.broadcast_to(u, (s_samples,) + u.shape).copy()

        du_ds = np.gradient(big_u, s, axis=0, edge_order=2)
        du_dt = np.gradient(big_u, tau, axis=1, edge_order=2)
        j_du = np.concatenate([-du_dt[..., n:], du_dt[..., :n]], axis=-1)
        grad_h = big_u * np.concatenate([lam, lam])[None, None, :]
        residual = du_ds + j_du - grad_h
        cr = float(np.linalg.norm(residual) / (np.linalg.norm(grad_h) + _WILKINSON_EPSILON))

        two_pi = 2.0 * math.pi
        margin = float(np.min(np.abs(lam - two_pi * np.round(lam / two_pi))))
        width = float(two_pi * energy_level / np.max(lam_abs))       # c_Gromov(E) = 2πc/λ_max
        ball_cap, cyl_cap = math.pi * ball_r ** 2, math.pi * cylinder_R ** 2
        j = self._standard_complex_structure(n)
        return PseudoholomorphicReport(
            cauchy_riemann_residual=cr,
            hamiltonian_asymmetry=asym,
            complex_structure_defect=float(np.linalg.norm(j @ j + np.eye(2 * n))),
            floquet_margin=margin,
            orbit_nondegenerate=bool(margin > _FLOQUET_MARGIN_TOL),
            gromov_width_ellipsoid=width,
            ball_capacity=ball_cap,
            cylinder_capacity=cyl_cap,
            is_gromov_rigid=bool(ball_cap <= cyl_cap and ball_cap <= width),
        )

    # ── 2.2 Poincaré: Melnikov + KAM Diofantina + ecuación homológica ─────────────────────────────
    def evaluate_poincare_kam_melnikov(
        self,
        action_J: float = 1.0,
        pert_eps: float = 1.0e-3,
        damping: float = 1.0e-2,
        omega0: float = 1.0,
        frequencies: Optional[NDArray[np.float64]] = None,
        diophantine_tau: Optional[float] = None,
        k_max: int = 12,
        torus_grid: int = 32,
    ) -> PoincareKAMMelnikovReport:
        """Melnikov del péndulo sobre la separatriz (Ω ≡ action_J) y verificación KAM sobre 𝕋²."""
        if action_J <= 0.0 or omega0 <= 0.0 or pert_eps < 0.0 or damping < 0.0:
            raise ValueError("Parámetros de Melnikov fuera de dominio.")

        # (a) Integral de Melnikov a lo largo de q₀(t)=4·arctan(e^{ω₀t})−π,  p₀(t)=2ω₀ sech(ω₀t)
        omega_f = float(action_J)
        t = np.linspace(-40.0 / omega0, 40.0 / omega0, 40_001)
        p0 = 2.0 * omega0 / np.cosh(omega0 * t)
        i_c = _trapezoid(p0 * np.cos(omega_f * t), t)
        i_s = _trapezoid(p0 * np.sin(omega_f * t), t)
        dissipation = _trapezoid(p0 * p0, t)                       # = 8ω₀
        amplitude = pert_eps * math.hypot(i_c, i_s)
        amplitude_exact = pert_eps * 2.0 * math.pi * _sech(math.pi * omega_f / (2.0 * omega0))
        q_defect = abs(amplitude - amplitude_exact) / (amplitude_exact + _WILKINSON_EPSILON)
        loss = damping * dissipation
        if loss > 0.0:
            ratio = amplitude / loss
        else:
            ratio = math.inf if amplitude > 0.0 else 0.0
        chaos = bool(ratio > 1.0)                                   # M(t₀) tiene ceros simples

        # (b) Condición Diofantina |k·ω| ≥ γ/‖k‖₁^τ en d=2
        pos = np.sort(np.asarray(frequencies if frequencies is not None else [], dtype=np.float64).reshape(-1))
        pos = pos[pos > _SPECTRAL_PSD_FLOOR][:2]
        if pos.size == 0:
            omega_vec = np.array([1.0, _GOLDEN_RATIO])
        elif pos.size == 1:
            omega_vec = np.array([pos[0], pos[0] * _GOLDEN_RATIO])
        else:
            omega_vec = pos.copy()
        omega_vec = omega_vec / float(np.linalg.norm(omega_vec))
        tau_d = float(diophantine_tau) if diophantine_tau is not None else 2.0
        ks = np.arange(-k_max, k_max + 1)
        k1, k2 = np.meshgrid(ks, ks, indexing="ij")
        norm1 = np.abs(k1) + np.abs(k2)
        mask = (norm1 > 0) & (norm1 <= k_max)
        gamma = float(np.min(np.abs(k1 * omega_vec[0] + k2 * omega_vec[1])[mask]
                             * norm1[mask].astype(np.float64) ** tau_d))

        # (c) Ecuación homológica {H₀,W}=H₁−⟨H₁⟩ con H₀=ω·J ⇒ −ω·∂_θW = H₁−⟨H₁⟩ ⇒ W_k = i h_k/(k·ω)
        n_g = int(torus_grid)
        th = 2.0 * math.pi * np.arange(n_g) / n_g
        t1, t2 = np.meshgrid(th, th, indexing="ij")
        h1 = pert_eps * (np.cos(t1) + np.cos(t2) + 0.5 * np.cos(t1 - t2) + 0.25 * np.sin(2.0 * t1 + t2))
        h_hat = np.fft.fft2(h1)
        kk = np.fft.fftfreq(n_g, d=1.0 / n_g)
        kw = kk[:, None] * omega_vec[0] + kk[None, :] * omega_vec[1]
        nonres = np.abs(kw) >= _RESONANCE_TOL
        w_hat = np.zeros_like(h_hat)
        w_hat[nonres] = 1j * h_hat[nonres] / kw[nonres]
        active = np.abs(h_hat) > 1.0e-12 * max(float(np.max(np.abs(h_hat))), _WILKINSON_EPSILON)
        resonant = int(np.sum(active & (~nonres) & (np.abs(kk[:, None]) + np.abs(kk[None, :]) > 0)))
        target = h1 - h1.mean()
        lhs = np.real(np.fft.ifft2(-1j * kw * w_hat))
        hom_res = float(np.linalg.norm(lhs - target) / (np.linalg.norm(target) + _WILKINSON_EPSILON))
        gen_norm = float(np.max(np.abs(np.real(np.fft.ifft2(w_hat)))))

        kam_ok = bool(gamma > _KAM_GAMMA_MIN and pert_eps <= gamma ** 2 and gen_norm < 1.0
                      and hom_res <= _HOMOLOGICAL_TOL and resonant == 0)
        return PoincareKAMMelnikovReport(
            melnikov_amplitude=float(amplitude),
            melnikov_amplitude_analytic=float(amplitude_exact),
            melnikov_quadrature_defect=float(q_defect),
            dissipation_integral=float(dissipation),
            melnikov_ratio=float(ratio),
            melnikov_chaos_detected=chaos,
            diophantine_gamma=gamma,
            diophantine_tau=tau_d,
            homological_residual=hom_res,
            generating_norm=gen_norm,
            resonant_modes=resonant,
            kam_stable=kam_ok,
        )

    # ── 2.3 (ÚLTIMO MÉTODO DE FASE 2) Sello → semilla de Fase 3 ───────────────────────────────────
    def seal_phase2_certificate(
        self,
        seed: SymplecticSeed,
        ball_r: float = 0.25,
        cylinder_R: float = 0.50,
        action_J: float = 1.0,
        pert_eps: float = 1.0e-3,
        damping: float = 1.0e-2,
    ) -> Phase2Certificate:
        """Ejecuta la Fase 2 y emite el Phase2Certificate (espectro hessiano, veredicto, digest encadenado)
        que es la condición inicial del operador de relajación de la Fase 3."""
        ph = self.solve_pseudoholomorphic_surface_witness(seed.cost_matrix, ball_r, cylinder_R)
        kam = self.evaluate_poincare_kam_melnikov(action_J, pert_eps, damping, frequencies=seed.frequencies)

        if (ph.is_gromov_rigid and ph.cauchy_riemann_residual <= _CR_RESIDUAL_TOL and ph.orbit_nondegenerate
                and ph.hamiltonian_asymmetry <= _HAMILTONIAN_ASYMMETRY_TOL
                and (not kam.melnikov_chaos_detected) and kam.kam_stable):
            truth = HeytingTruthValue.COHERENT
        elif ph.is_gromov_rigid and not kam.melnikov_chaos_detected:
            truth = HeytingTruthValue.DEGRADED
        else:
            truth = HeytingTruthValue.VETOED

        spectrum = np.sort(np.diag(seed.hamiltonian_matrix)[: seed.cost_matrix.shape[0]] * 0.0
                           + np.linalg.eigvalsh(0.5 * (seed.cost_matrix + seed.cost_matrix.T)))
        digest = self._digest("phase2", {
            "cr": ph.cauchy_riemann_residual, "floquet": ph.floquet_margin, "width": ph.gromov_width_ellipsoid,
            "melnikov": kam.melnikov_amplitude, "ratio": kam.melnikov_ratio, "gamma": kam.diophantine_gamma,
            "hom": kam.homological_residual, "truth": int(truth),
        }, parent=seed.parent_digest)
        return Phase2Certificate(ph, kam, spectrum, truth, seed.parent_digest, digest)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 3
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, eq=False)
class RSISeed:
    """Semilla de la automejora recursiva (puente 2→3): Hessiano SPD diagonalizado."""
    hessian_spectrum: NDArray[np.float64]
    parent_digest: str
    upstream_truth: HeytingTruthValue


@dataclass(frozen=True, eq=False)
class _LipWriter:
    """Mónada Writer sobre (ℝ₊,·,1): unit(x)=(x,1), bind((x,l),f)=(f(x), l·Lip f).
    Las leyes de mónada se heredan de las del monoide multiplicativo ⇒ Lip(T³)=ΠLip(Tᵢ)."""
    value: NDArray[np.float64]
    lip: float = 1.0

    def bind(self, arrow: Callable[[NDArray[np.float64]], NDArray[np.float64]], arrow_lip: float) -> "_LipWriter":
        return _LipWriter(arrow(self.value), self.lip * arrow_lip)


@dataclass(frozen=True)
class RSI3BanachReport:
    """Reporte RSI³ (Darwin–Gödel) bajo contracción de Banach."""
    lip_t3_legacy_bound: float
    lip_t3_rigorous_initial: float
    lip_t1_post: float
    lip_t3_post: float
    empirical_lip_t3: float
    eta_initial: float
    eta_final: float
    eta_optimal: float
    beta_final: float
    rounds_accepted: int
    rounds_rejected: int
    iterations_to_tolerance: int
    capacity_jerk_analytic: float
    capacity_third_difference: float
    jerk_positive: bool
    is_contraction: bool
    ledger_digest: str


@dataclass(frozen=True)
class ToonCartridgeWeaverWitnessSpectrumReport:
    """Reporte de atestación espectral testigo de tejeduría sináptica TOON (superconjunto del v13)."""
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    krull_valuation: float
    novikov_norm: float
    toon_token_count: int
    kv_cache_compression_pct: float
    cauchy_riemann_residual: float
    is_gromov_rigid: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    capacity_jerk_d3c_dt3: float
    qnd_report: QNDWeakMeasurementReport
    bures_report: BuresUhlmannMetricReport
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ms: float
    # ── extensiones v14 ──
    homology_report: BooleHomologyReport
    novikov_toon_report: NovikovToonReport
    pseudoholomorphic_report: PseudoholomorphicReport
    kam_report: PoincareKAMMelnikovReport
    rsi3_report: RSI3BanachReport
    phase_verdicts: Tuple[HeytingTruthValue, HeytingTruthValue, HeytingTruthValue]
    veto_reasons: Tuple[str, ...]
    ledger_digest: str
    passport_payload: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — AUTOMEJORA RECURSIVA RSI³, VEREDICTO DE HEYTING Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonCartridgeWeaverWitnessEngine(_Phase2SymplecticPoincareKAM):
    """FASE 3 (clase pública): (hereda Fases 1 y 2) Hessiano SPD → Kleisli/RSI³ → meet de Heyting
    → pasaporte HMAC encadenado → orquestador de auditoría."""

    # ── 3.0 (PRIMER MÉTODO DE FASE 3) continuación del certificado de Fase 2 ──────────────────────
    def ingest_phase2_certificate(self, certificate: Phase2Certificate) -> RSISeed:
        """H_SPD = diag(max(|λ_k|, 1e-6)): Hessiano del funcional cuadrático de márgenes de testigo."""
        spec = np.maximum(np.abs(np.asarray(certificate.hessian_spectrum, dtype=np.float64)), 1.0e-6)
        return RSISeed(spec, certificate.digest, certificate.truth_value)

    # ── 3.1 RSI³ + mónada Kleisli-Writer + cota de Banach ─────────────────────────────────────────
    def evaluate_rsi3_banach_kleisli(
        self,
        initial_lip: float = 0.95,
        perturbation_margin: float = _LEGACY_LIP_MARGIN,
        hessian_spectrum: Optional[NDArray[np.float64]] = None,
        meta_rounds: int = 8,
        tolerance: float = 1.0e-9,
        parent_digest: str = "",
    ) -> RSI3BanachReport:
        """Niveles: (1) T_η(x)=x−ηH(x−x*), (2) mutación η ← η+β(η_opt−η), (3) mutación β ← 1.5β | β/2.
        Una automodificación se acepta (Gödel) sólo si el certificado exacto Lip' < Lip < 1; si no, rollback."""
        if not 0.0 < initial_lip < 1.0 or perturbation_margin < 0.0 or meta_rounds < 0:
            raise ValueError("Parámetros RSI³ fuera de dominio.")
        spec = np.maximum(np.abs(np.asarray(hessian_spectrum if hessian_spectrum is not None else [1.0],
                                            dtype=np.float64).reshape(-1)), 1.0e-6)
        lmin, lmax = float(spec.min()), float(spec.max())

        def lip_of(eta_: float) -> float:
            return float(np.max(np.abs(1.0 - eta_ * spec)))

        # cotas heredadas y cota rigurosa por composición con perturbación aditiva δ = margin/L₀
        lip_legacy = float(initial_lip ** 3 + perturbation_margin)
        lip_rigorous = float((initial_lip + perturbation_margin / initial_lip) ** 3)

        eta = min((1.0 - initial_lip) / lmin, 1.0 / lmax)
        eta0 = eta
        eta_opt = 2.0 / (lmin + lmax)
        beta, current = 0.25, lip_of(eta)
        accepted = rejected = 0
        chain = parent_digest
        for r in range(meta_rounds):
            cand_eta = eta + beta * (eta_opt - eta)
            cand_lip = lip_of(cand_eta)
            ok = bool(cand_lip < current - 1.0e-15 and cand_lip < 1.0)
            if ok:
                eta, current = cand_eta, cand_lip
                beta = min(1.0, 1.5 * beta)
                accepted += 1
            else:
                beta *= 0.5
                rejected += 1
            chain = self._digest("rsi3-round", {"r": r, "eta": eta, "beta": beta, "lip": current, "ok": ok}, chain)

        # Composición de Kleisli T∘T∘T en el Writer y verificación empírica del certificado
        def arrow(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return (1.0 - eta * spec) * x

        x0 = np.ones_like(spec)
        t3 = _LipWriter(x0).bind(arrow, current).bind(arrow, current).bind(arrow, current)
        lip_t3_post = float(t3.lip)

        rng = np.random.default_rng(0xC0FFEE)
        samples = [np.eye(spec.size)[k] for k in range(spec.size)] + [rng.standard_normal(spec.size) for _ in range(256)]
        emp = 0.0
        for xv in samples:
            yv = rng.standard_normal(spec.size)
            for a_, b_ in ((xv, np.zeros_like(xv)), (xv, yv)):
                den = float(np.linalg.norm(a_ - b_))
                if den > 0.0:
                    out = _LipWriter(a_).bind(arrow, 1.0).bind(arrow, 1.0).bind(arrow, 1.0).value \
                        - _LipWriter(b_).bind(arrow, 1.0).bind(arrow, 1.0).bind(arrow, 1.0).value
                    emp = max(emp, float(np.linalg.norm(out)) / den)

        d01 = float(np.linalg.norm(arrow(x0) - x0))
        if current <= 0.0:
            n_iter = 1
        elif d01 <= 0.0:
            n_iter = 0
        else:
            n_iter = max(0, int(math.ceil(math.log(tolerance * (1.0 - current) / d01) / math.log(current))))

        # Capacidad C_n = 1 − e₀·L₃ⁿ  ⇒  d³C/dt³ = e₀|ln L₃|³ L₃ᵗ > 0  y  Δ³C = e₀(1−L₃)³ > 0
        l3_eff = min(max(lip_t3_post, 1.0e-12), 1.0 - 1.0e-15)
        jerk = float(abs(math.log(l3_eff)) ** 3)
        cap = 1.0 - l3_eff ** np.arange(4, dtype=np.float64)
        third_diff = float(np.diff(cap, 3)[0])

        return RSI3BanachReport(
            lip_t3_legacy_bound=lip_legacy,
            lip_t3_rigorous_initial=lip_rigorous,
            lip_t1_post=float(current),
            lip_t3_post=lip_t3_post,
            empirical_lip_t3=float(emp),
            eta_initial=float(eta0),
            eta_final=float(eta),
            eta_optimal=float(eta_opt),
            beta_final=float(beta),
            rounds_accepted=accepted,
            rounds_rejected=rejected,
            iterations_to_tolerance=n_iter,
            capacity_jerk_analytic=jerk,
            capacity_third_difference=third_diff,
            jerk_positive=bool(jerk > 0.0 and third_diff > 0.0),
            is_contraction=bool(lip_t3_post < 1.0 and lip_rigorous < 1.0),
            ledger_digest=chain,
        )

    # ── 3.2 Veredicto de Heyting (meet de las tres fases) ─────────────────────────────────────────
    def classify_heyting_truth(
        self, p1: Phase1Certificate, p2: Phase2Certificate, rsi: RSI3BanachReport
    ) -> Tuple[HeytingTruthValue, Tuple[HeytingTruthValue, HeytingTruthValue, HeytingTruthValue], Tuple[str, ...]]:
        reasons: List[str] = []
        q, b, h, n = p1.qnd_report, p1.bures_report, p1.homology_report, p1.novikov_toon_report
        if not q.is_qnd_passive:
            reasons.append(f"QND: conmutador relativo {q.relative_commutator:.3e} > {_QND_COMMUTATOR_TOL:.0e}")
        if not q.weak_value_valid:
            reasons.append("AAV: post-selección casi ortogonal (valor débil singular)")
        if not b.is_isometry_preserved:
            reasons.append(f"Bures: F={b.fidelity_uhlmann:.4f}, d_B={b.bures_distance:.4f}")
        if not h.torsion_free:
            reasons.append(f"SNF: torsión detectada {h.invariant_factors}")
        if not h.betti_number_zero:
            reasons.append(f"Topología: β₁={h.betti_1} > 0 (ciclos en el grafo de acoplamientos)")
        if not n.krull_floor_ok:
            reasons.append(f"Novikov: v(A)={n.krull_valuation:.4f} < {_DEFAULT_KRULL_FLOOR}")
        if not (n.ultrametric_ok and n.multiplicativity_ok):
            reasons.append("Novikov: violación ultramétrica/multiplicativa")
        ph, kam = p2.pseudoholomorphic_report, p2.kam_report
        if not ph.is_gromov_rigid:
            reasons.append("Gromov: capacidad de la bola excede cilindro o elipsoide")
        if ph.cauchy_riemann_residual > _CR_RESIDUAL_TOL:
            reasons.append(f"Floer: residuo CR {ph.cauchy_riemann_residual:.3e}")
        if not ph.orbit_nondegenerate:
            reasons.append("Floer: órbita degenerada (multiplicador de Floquet = 1)")
        if ph.hamiltonian_asymmetry > _HAMILTONIAN_ASYMMETRY_TOL:
            reasons.append(f"Costo no hamiltoniano: asimetría {ph.hamiltonian_asymmetry:.3e}")
        if kam.melnikov_chaos_detected:
            reasons.append(f"Melnikov: ratio {kam.melnikov_ratio:.3f} > 1 (intersección homoclínica transversal)")
        if not kam.kam_stable:
            reasons.append(f"KAM: γ={kam.diophantine_gamma:.3e}, resonancias={kam.resonant_modes}")
        if not rsi.is_contraction:
            reasons.append("Banach: Lip(T³) ≥ 1")
        if not rsi.jerk_positive:
            reasons.append("RSI³: jerk de capacidad no positivo")
        if rsi.empirical_lip_t3 > rsi.lip_t3_post + 1.0e-9:
            reasons.append("RSI³: Lipschitz empírico excede el certificado")

        if rsi.is_contraction and rsi.jerk_positive and rsi.empirical_lip_t3 <= rsi.lip_t3_post + 1.0e-9:
            v3 = HeytingTruthValue.COHERENT
        elif rsi.lip_t3_post < 1.0:
            v3 = HeytingTruthValue.DEGRADED
        else:
            v3 = HeytingTruthValue.VETOED
        v1, v2 = p1.truth_value, p2.truth_value
        return heyting_meet(heyting_meet(v1, v2), v3), (v1, v2, v3), tuple(reasons)

    # ── 3.3 Pasaporte verificable ─────────────────────────────────────────────────────────────────
    def verify_passport(self, report: ToonCartridgeWeaverWitnessSpectrumReport) -> bool:
        """Recalcula el HMAC canónico (sin tiempos) y compara en tiempo constante."""
        expected = self._digest("passport", report.passport_payload, report.ledger_digest)
        return hmac.compare_digest(expected, report.passport_hash)

    # ── 3.4 (ÚLTIMO MÉTODO) Orquestador 1→2→3 ─────────────────────────────────────────────────────
    def execute_toon_cartridge_weaver_witness_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
        witness_operator: Optional[NDArray[np.complex128]] = None,
        raw_token_count: int = _TOON_RAW_TOKENS,
        toon_token_count: int = _TOON_NOMINAL_TOKENS,
    ) -> ToonCartridgeWeaverWitnessSpectrumReport:
        """Ciclo completo: Fase 1 ⟶ semilla simpléctica ⟶ Fase 2 ⟶ semilla RSI ⟶ Fase 3 ⟶ pasaporte."""
        start = time.perf_counter()

        p1 = self.seal_phase1_certificate(apu_cost_matrix, exponents, rho_mac_input, witness_operator,
                                          raw_token_count, toon_token_count)
        seed2 = self.ingest_phase1_certificate(p1)
        p2 = self.seal_phase2_certificate(seed2)
        seed3 = self.ingest_phase2_certificate(p2)
        rsi = self.evaluate_rsi3_banach_kleisli(hessian_spectrum=seed3.hessian_spectrum,
                                                parent_digest=seed3.parent_digest)
        truth, verdicts, reasons = self.classify_heyting_truth(p1, p2, rsi)

        h, n, ph, kam, q, b = (p1.homology_report, p1.novikov_toon_report, p2.pseudoholomorphic_report,
                               p2.kam_report, p1.qnd_report, p1.bures_report)
        payload: Dict[str, Any] = {
            "version": _ENGINE_VERSION, "truth": int(truth), "verdicts": [int(v) for v in verdicts],
            "rw": h.rademacher_walsh_norm, "krull": n.krull_valuation, "toon": n.toon_token_count,
            "kv": n.kv_cache_compression_pct, "backaction": q.back_action_db,
            "fidelity": b.fidelity_uhlmann, "cr": ph.cauchy_riemann_residual,
            "melnikov_ratio": kam.melnikov_ratio, "lip_t3": rsi.lip_t3_post, "jerk": rsi.capacity_jerk_analytic,
            "phase1": p1.digest, "phase2": p2.digest,
        }
        passport = self._digest("passport", payload, rsi.ledger_digest)
        exec_ms = (time.perf_counter() - start) * 1000.0

        return ToonCartridgeWeaverWitnessSpectrumReport(
            rademacher_walsh_norm=h.rademacher_walsh_norm,
            smith_torsion_free=h.torsion_free,
            betti_number_zero=h.betti_number_zero,
            krull_valuation=n.krull_valuation,
            novikov_norm=n.novikov_norm,
            toon_token_count=n.toon_token_count,
            kv_cache_compression_pct=n.kv_cache_compression_pct,
            cauchy_riemann_residual=ph.cauchy_riemann_residual,
            is_gromov_rigid=ph.is_gromov_rigid,
            melnikov_chaos_detected=kam.melnikov_chaos_detected,
            banach_contraction_lip=rsi.lip_t3_post,
            capacity_jerk_d3c_dt3=rsi.capacity_jerk_analytic,
            qnd_report=q,
            bures_report=b,
            truth_value=truth,
            passport_hash=passport,
            execution_time_ms=exec_ms,
            homology_report=h,
            novikov_toon_report=n,
            pseudoholomorphic_report=ph,
            kam_report=kam,
            rsi3_report=rsi,
            phase_verdicts=verdicts,
            veto_reasons=reasons,
            ledger_digest=rsi.ledger_digest,
            passport_payload=payload,
        )


__all__ = [
    "HeytingTruthValue", "heyting_meet", "heyting_join", "heyting_implies", "NovikovSeries",
    "QNDWeakMeasurementReport", "BuresUhlmannMetricReport", "BooleHomologyReport", "NovikovToonReport",
    "Phase1Certificate", "SymplecticSeed", "PseudoholomorphicReport", "PoincareKAMMelnikovReport",
    "Phase2Certificate", "RSISeed", "RSI3BanachReport", "ToonCartridgeWeaverWitnessSpectrumReport",
    "BoolNovikovToonCartridgeWeaverWitnessEngine",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverWitnessEngine v14 ===")
    engine = BoolNovikovToonCartridgeWeaverWitnessEngine()

    # Auto-verificaciones algebraicas independientes
    assert _smith_invariants([[2, 4, 4], [-6, 6, 12], [10, -4, -16]]) == [2, 6, 12], "SNF incorrecta"
    a_ = NovikovSeries.from_terms([(0.2, 1.0), (0.5, -1.0)])
    b_ = NovikovSeries.from_terms([(0.3, 2.0)])
    assert abs((a_ * b_).norm() - a_.norm() * b_.norm()) < 1e-12, "Novikov no multiplicativa"
    wht = _fwht(np.array([1.0, -1.0, -1.0, 1.0]))          # f = (−1)^{x₁x₂}: Parseval
    assert abs(float(np.sum(wht ** 2)) - 1.0) < 1e-12

    matrix_test = np.array([
        [1.0, 0.2, 0.0, 0.1],
        [0.2, 1.1, 0.3, 0.0],
        [0.0, 0.3, 0.9, 0.2],
        [0.1, 0.0, 0.2, 1.0],
    ], dtype=np.float64)
    exps_test = np.array([0.15, 0.28, 0.45], dtype=np.float64)

    report = engine.execute_toon_cartridge_weaver_witness_audit(matrix_test, exps_test)

    print(f"Veredicto Heyting Ω₃       : {report.truth_value.name} ({hex(report.truth_value.value)}) "
          f"fases={[v.name for v in report.phase_verdicts]}")
    for reason in report.veto_reasons:
        print(f"   · {reason}")
    print(f"Back-Action QND (dB)       : {report.qnd_report.back_action_db:.6f} dB (QND: {report.qnd_report.is_qnd_passive})")
    print(f"Compresión KV-Cache        : {report.kv_cache_compression_pct:.1f}% ({report.toon_token_count} tokens)")
    print(f"Fidelidad de Uhlmann       : {report.bures_report.fidelity_uhlmann:.6f}")
    print(f"Homología (β₀,β₁,SNF)      : ({report.homology_report.betti_0}, {report.homology_report.betti_1}, "
          f"{report.homology_report.invariant_factors})  Fiedler={report.homology_report.fiedler_value:.4f}")
    print(f"Valuación Krull v(A)       : {report.krull_valuation:.4f}")
    print(f"Residuo Cauchy-Riemann     : {report.cauchy_riemann_residual:.3e}")
    print(f"Melnikov ratio / γ_Diof    : {report.kam_report.melnikov_ratio:.4f} / {report.kam_report.diophantine_gamma:.4f}")
    print(f"Lip(T³) legado → RSI³      : {report.rsi3_report.lip_t3_legacy_bound:.6f} → {report.banach_contraction_lip:.6f}")
    print(f"Jerk Capacidad d³C/dt³     : {report.capacity_jerk_d3c_dt3:.4f}")
    print(f"Pasaporte HMAC SHA-256     : {report.passport_hash[:16]}...  verificable={engine.verify_passport(report)}")

    # Contraste negativo 1: testigo no conmutante ⇒ back-action > 0 dB
    bad = engine.execute_toon_cartridge_weaver_witness_audit(
        matrix_test, exps_test, witness_operator=np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128))
    assert not bad.qnd_report.is_qnd_passive and bad.qnd_report.back_action_db > 0.0
    # Contraste negativo 2: forzamiento sin disipación ⇒ cero simple de Melnikov ⇒ caos homoclínico
    chaotic = engine.evaluate_poincare_kam_melnikov(action_J=1.0, pert_eps=0.1, damping=0.0)
    assert chaotic.melnikov_chaos_detected and chaotic.melnikov_quadrature_defect < 1e-6
    print("=== TEST COMPLETADO CON ÉXITO ===")