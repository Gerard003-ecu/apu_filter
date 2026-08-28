# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Audit Satellites (Satélites de Auditoría de Frontera Homológica)    ║
║ Ruta   : app/core/audit_satellites.py                                        ║
║ Versión: 2.1.0-Doctoral-SNF-Choi-CHSH-Heyting-Godel-Secure                   ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y EXERGÉTICA:                                            ║
║ Este módulo implementa satélites de auditoría para vigilar la frontera       ║
║ de-confinada ∂M de la fortaleza de APU Filter.                               ║
║                                                                              ║
║ El módulo queda organizado en tres fases anidadas (inclusión de kernels      ║
║ por herencia, funtorial en el sentido de                                     ║
║   Phase1 ↪ Phase2 ↪ Phase3 ↪ AuditSatellites):                               ║
║   FASE 1: Aduana homológica sobre ℤ y Smith Normal Form exacta.              ║
║   FASE 2: Aduana cuántica de Choi–Jamiołkowski (CP / TP / hermiticidad).     ║
║   FASE 3: Aduana Bell–CHSH, retículo de Heyting Ω₃, actuación y certificado. ║
║                                                                              ║
║ Morfismos de fase (continuación formal):                                     ║
║   audit_boundary_homology    : ∂ ↦ HomologyReport                            ║
║   audit_choi_with_homology   : (HomologyReport, C_ℰ) ↦ CausalConsolidation   ║
║   prepare_causal_consolidation: (Homology, Choi) ↦ CausalConsolidation       ║
║   audit_bell_with_causality  : (CausalConsolidation, E) ↦ HeytingDecision    ║
║   act_on_heyting_decision    : HeytingDecision ↦ AuditActuationReport        ║
║                                                                              ║
║ El último método de la Fase 1 prepara `HomologyReport`, objeto inicial de    ║
║ la Fase 2. El último método de la Fase 2 prepara `CausalConsolidation`,      ║
║ objeto inicial de la Fase 3.                                                 ║
║                                                                              ║
║ Dual-control intuicionista: el veredicto final es el ínfimo de Gödel de      ║
║ las tres aduanas. COHERENT sólo se afirma si las tres lo demuestran.         ║
║                                                                              ║
║   Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED}  ≅  {1, 1/2, 0} ⊂ [0, 1]              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Final, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Physics.AuditSatellites")

__version__: Final[str] = "2.1.0-Doctoral-SNF-Choi-CHSH-Heyting-Godel-Secure"


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS Y COTAS FÍSICAS
# ════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

_CLASSICAL_CHSH_BOUND: Final[float] = 2.0
_TSIRELSON_BOUND: Final[float] = 2.0 * math.sqrt(2.0)
_ALGEBRAIC_CHSH_BOUND: Final[float] = 4.0
_CORRELATION_ABS_MAX: Final[float] = 1.0

_DEFAULT_SNF_MAX_ITERATIONS: Final[int] = 5000
_DEFAULT_SNF_MAX_ORDER: Final[int] = 256
_DEFAULT_INTEGER_TOLERANCE: Final[float] = 1.0e-12
_FLOAT64_EXACT_INT_MAX: Final[float] = float(2**53)

_DEFAULT_MAX_BATCH: Final[int] = 4096

# Ω₃ = {0, 1/2, 1} ⊂ [0, 1], álgebra de Heyting finita de Gödel–Dummett.
_OMEGA3_FALSE: Final[float] = 0.0
_OMEGA3_MIDDLE: Final[float] = 0.5
_OMEGA3_TRUE: Final[float] = 1.0

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_VERDICT_ORDER: Final[Dict[str, int]] = {
    _VERDICT_COHERENT: 0,
    _VERDICT_DEGRADED: 1,
    _VERDICT_VETOED: 2,
}

_TRUTH_VALUES: Final[Dict[str, float]] = {
    _VERDICT_COHERENT: _OMEGA3_TRUE,
    _VERDICT_DEGRADED: _OMEGA3_MIDDLE,
    _VERDICT_VETOED: _OMEGA3_FALSE,
}

_AUDIT_PHASE: Final[str] = "G_AUDIT_SATELLITES_SUTURATED"


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS, ENTERAS Y RETICULARES PURAS
# ════════════════════════════════════════════════════════════════════════════


def _now_utc_iso() -> str:
    """Instante UTC canónico para trazabilidad."""
    return datetime.now(timezone.utc).isoformat()


def _unique_reasons(reasons: Iterable[str]) -> Tuple[str, ...]:
    """Deduplica razones preservando el orden de primera aparición."""
    seen = set()
    ordered: List[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    """Huella canónica SHA-256 de un mapeo JSON-serializable."""
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _heyting_meet(left: float, right: float) -> float:
    r"""Ínfimo de Gödel en \(\Omega_3\): \(a \wedge b = \min(a, b)\)."""
    return float(min(left, right))


def _heyting_implies(antecedent: float, consequent: float) -> float:
    r"""
    Implicación intuicionista de Gödel:

    \[
        a \to b
        =
        \begin{cases}
            1 & \text{si } a \le b, \\
            b & \text{en otro caso.}
        \end{cases}
    \]
    """
    if antecedent <= consequent + 8.0 * _MACHINE_EPS:
        return _OMEGA3_TRUE
    return float(consequent)


def _geometric_mean(values: Sequence[float]) -> float:
    """Media geométrica de márgenes en (0, 1], con piso numérico."""
    if not values:
        return 1.0
    acc = 0.0
    for item in values:
        if not math.isfinite(item) or item <= 0.0:
            return 0.0
        acc += math.log(max(item, _MACHINE_EPS))
    return float(math.exp(acc / float(len(values))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    r"""Cociente protegido en la compactificación \(\mathbb{R} \cup \{\infty\}\)."""
    num = float(numerator)
    den = float(denominator)
    if not math.isfinite(num):
        return math.inf
    if not math.isfinite(den) or den == 0.0:
        return 1.0 if num == 0.0 else math.inf
    ratio = num / den
    return float(ratio) if math.isfinite(ratio) else math.inf


def ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    r"""
    Algoritmo extendido de Euclides sobre el dominio de ideales principales
    \(\mathbb{Z}\).

    Retorna \((g, x, y)\) tales que
    \[
        a x + b y = g = \gcd(a,b) \ge 0.
    \]

    La implementación usa enteros de Python (precisión arbitraria) para
    evitar overflow de int64. Los cofactores de Bézout heredan el signo
    de \(a\) y \(b\).
    """
    a0 = int(a)
    b0 = int(b)

    aa = abs(a0)
    bb = abs(b0)

    old_r, r = aa, bb
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    g = int(old_r)
    x = int(old_s)
    y = int(old_t)

    if a0 < 0:
        x = -x
    if b0 < 0:
        y = -y

    return g, x, y


def _lcm_nonneg(a: int, b: int) -> int:
    r"""Mínimo común múltiplo en \(\mathbb{N}_0\). Convención: \(\mathrm{lcm}(0,b)=0\)."""
    aa = abs(int(a))
    bb = abs(int(b))
    if aa == 0 or bb == 0:
        return 0
    return (aa // math.gcd(aa, bb)) * bb


def _invariant_factors_from_diagonal(diag: Sequence[int]) -> Tuple[int, ...]:
    r"""
    Completa la forma de factores invariantes a partir de una diagonal.

    Si \(\operatorname{diag}(a_1,\ldots,a_r)\) es la parte no nula, el barrido
    adyacente \(\gcd/\mathrm{lcm}\) produce \(d_1 \mid d_2 \mid \cdots \mid d_r\)
    con
    \[
        \mathbb{Z}^r / \langle a_1 e_1,\ldots,a_r e_r\rangle
        \;\simeq\;
        \bigoplus_{i=1}^{r} \mathbb{Z}/d_i\mathbb{Z}
    \]
    (los \(d_i=1\) se conservan para no perder rango; la torsión es \(d_i>1\)).
    """
    nonzero = [abs(int(v)) for v in diag if int(v) != 0]
    if not nonzero:
        return ()

    changed = True
    guard = 0
    limit = max(8 * len(nonzero) * len(nonzero), 32)
    while changed and guard < limit:
        changed = False
        guard += 1
        for i in range(len(nonzero) - 1):
            g = math.gcd(nonzero[i], nonzero[i + 1])
            l = _lcm_nonneg(nonzero[i], nonzero[i + 1])
            if nonzero[i] != g or nonzero[i + 1] != l:
                nonzero[i] = g
                nonzero[i + 1] = l
                changed = True

    return tuple(int(v) for v in nonzero)


def _divisibility_chain_ok(factors: Sequence[int]) -> bool:
    r"""Verifica \(d_i \mid d_{i+1}\) sobre factores no nulos."""
    prev = None
    for raw in factors:
        value = abs(int(raw))
        if value == 0:
            continue
        if prev is not None and (value % prev != 0):
            return False
        prev = value
    return True


# ════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SmithNormalFormResult:
    r"""
    Resultado exacto (o parcialmente exacto) de la SNF sobre \(\mathbb{Z}\).

    \[
        S = U A V = \operatorname{diag}(d_1,\ldots,d_r,0,\ldots,0),
        \qquad
        U,V \in \mathrm{GL}(\mathbb{Z}).
    \]
    """

    diagonal: Tuple[int, ...]
    invariant_factors: Tuple[int, ...]
    rank: int
    nrows: int
    ncols: int
    is_complete: bool
    is_divisibility_chain: bool
    iterations: int
    off_diagonal_residue: int
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HomologyReport:
    r"""
    Resultado de la Fase 1: auditoría homológica sobre \(\mathbb{Z}\).

    La frontera es coherente si la SNF es completa y no produce factores
    invariantes mayores que 1:
    \[
        \operatorname{Tor}\bigl(H_\bullet(\partial K;\mathbb{Z})\bigr) = 0.
    \]

    Los números de Betti inferiores se estiman por
    \(\beta \ge n_{\mathrm{cols}} - \operatorname{rk}(\partial)\) (nulidad)
    y \(\operatorname{rk}_{\mathrm{coker}}^{\mathrm{libre}} = n_{\mathrm{rows}} - \operatorname{rk}(\partial)\).
    """

    is_coherent: bool
    torsion_coefficients: Tuple[int, ...]
    smith_diagonal: Tuple[int, ...]
    matrix_rank: int
    reasons: Tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    invariant_factors: Tuple[int, ...] = ()
    snf_complete: bool = True
    divisibility_chain_ok: bool = True
    nrows: int = 0
    ncols: int = 0
    nullity: int = 0
    free_cokernel_rank: int = 0
    snf_iterations: int = 0
    has_torsion: bool = False


@dataclass(frozen=True, slots=True)
class ChoiReport:
    r"""
    Resultado de la Fase 2: auditoría CPTP del canal cuántico de inyección.

    Isomorfismo de Choi–Jamiołkowski. Convención (C-order):
    \[
        C[i d + p,\; j d + q]
        =
        \langle p\rvert\,
        \mathcal{E}(\lvert i\rangle\langle j\rvert)
        \,\lvert q\rangle,
    \]
    de modo que \(C \cong C_4[i,p,j,q]\) y
    \[
        \operatorname{Tr}_2(C)_{\,ij}
        =
        \sum_p C_4[i,p,j,p].
    \]

    CP: \(C_{\mathcal{E}} \succeq 0\).
    TP: \(\operatorname{Tr}_2(C_{\mathcal{E}}) = I_d\).
    HP: \(C_{\mathcal{E}}^\dagger = C_{\mathcal{E}}\).
    Unitalidad (dual TP): \(\operatorname{Tr}_1(C_{\mathcal{E}}) = I_d\).
    """

    is_cptp: bool
    is_cp: bool
    is_tp: bool
    is_hermitian: bool

    min_eigenvalue: float
    tp_residual: float
    trace_residual: float
    hermitian_residual: float

    cp_tolerance: float
    tp_tolerance: float
    hermitian_tolerance: float

    reasons: Tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    dimension: int = 0
    kraus_rank: int = 0
    max_eigenvalue: float = 0.0
    choi_trace: float = 0.0
    unital_residual: float = 0.0
    is_unital: bool = False
    hermitian_relative_residual: float = 0.0
    nuclear_norm: float = 0.0
    frobenius_norm: float = 0.0
    spectral_gap: float = 0.0


@dataclass(frozen=True, slots=True)
class BellReport:
    r"""
    Resultado de la Fase 3: auditoría Bell–CHSH contra colusión monopólica.

    El correlador CHSH canónico es
    \[
        \mathcal{B}
        =
        \max_{\varepsilon \in \{\pm 1\}^4}
        \bigl|
            \varepsilon_{11} E_{11}
            + \varepsilon_{12} E_{12}
            + \varepsilon_{21} E_{21}
            + \varepsilon_{22} E_{22}
        \bigr|
    \]
    restringido a un número impar de signos negativos (cuatro patrones
    equivalentes a «un único minus»).

    Cota clásica (local-realista): \(|\mathcal{B}| \le 2\).
    Cota de Tsirelson (cuántica): \(|\mathcal{B}| \le 2\sqrt{2}\).
    Cota algebraica (no-signalling PR): \(|\mathcal{B}| \le 4\).
    """

    is_coherent: bool
    is_classical: bool
    chsh_value: float
    classical_bound: float = _CLASSICAL_CHSH_BOUND
    tsirelson_bound: float = _TSIRELSON_BOUND
    reasons: Tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    algebraic_bound: float = _ALGEBRAIC_CHSH_BOUND
    correlations: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    correlations_physical: bool = True
    chsh_canonical: float = 0.0
    classical_margin: float = 0.0
    tsirelson_margin: float = 0.0


@dataclass(frozen=True, slots=True)
class CausalConsolidation:
    r"""
    Consolidación de Fase 1 y Fase 2.

    Este objeto es el insumo formal de la Fase 3: el germen causal
    \(\partial M \times \mathcal{E}\) que la aduana Bell fiscaliza.
    """

    homology: HomologyReport
    choi: ChoiReport
    reasons: Tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    is_causally_admissible: bool = False


@dataclass(frozen=True, slots=True)
class HeytingDecisionReport:
    r"""
    Decidido final en el retículo de Heyting \(\Omega_3\).

    El valor de verdad reticular es el ínfimo de Gödel de las proposiciones
    atómicas de las tres aduanas (dual-control homológico–cuántico–Bell).
    """

    verdict: str
    truth_value: float

    homology: HomologyReport
    choi: ChoiReport
    bell: BellReport

    reasons: Tuple[str, ...] = ()
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    truth_continuous: float = 1.0
    heyting_implies_coherent: float = 1.0
    atomic_propositions: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class AuditActuationReport:
    r"""
    Reporte de actuación ciber-física del crowbar perimetral.

    Política fail-closed: VETOED ⇒ interlock; cualquier otro veredicto
    inhibe el disyuntor. La latencia se recorta al intervalo IRAM.
    """

    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    reason: str
    timestamp_utc: str = field(default_factory=_now_utc_iso)


@dataclass(frozen=True, slots=True)
class AuditSatelliteCertificate:
    r"""
    Certificado inmutable de la aduana homológica–cuántica–Bell.

    Conserva los campos del diccionario histórico y añade no-repudio:

      - ``decision_sha256`` : huella isomorfa (sin reloj).
      - ``digital_signature_sha256`` : huella de no-repudio (con reloj).
    """

    phase: str
    heyting_verdict: str
    smith_torsion_coefficients: Tuple[int, ...]
    choi_minimum_eigenvalue: float
    choi_trace_preserving_residual: float
    bell_chsh_parameter: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    is_homology_coherent: bool = False
    smith_diagonal: Tuple[int, ...] = ()
    smith_matrix_rank: int = 0
    choi_is_cptp: bool = False
    choi_is_cp: bool = False
    choi_is_tp: bool = False
    choi_is_hermitian: bool = False
    choi_trace_residual: float = math.inf
    choi_hermitian_residual: float = math.inf
    bell_is_classical: bool = False
    bell_classical_bound: float = _CLASSICAL_CHSH_BOUND
    bell_tsirelson_bound: float = _TSIRELSON_BOUND
    heyting_truth_value: float = 0.0
    gpio_pin: str = ""
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    reasons: Tuple[str, ...] = ()
    decision_sha256: str = ""
    invariant_factors: Tuple[int, ...] = ()
    choi_kraus_rank: int = 0
    bell_tsirelson_margin: float = 0.0
    version: str = __version__


# ════════════════════════════════════════════════════════════════════════════
# FASE 1 — ADUANA HOMOLÓGICA Y SMITH NORMAL FORM SOBRE ℤ
# ════════════════════════════════════════════════════════════════════════════


class Phase1HomologicalAuditKernel:
    r"""
    FASE 1: Auditoría homológica discreta.

    Responsabilidades:
      1. Validar matrices enteras (sin pérdida de precisión de float64).
      2. Calcular Smith Normal Form exacta sobre \(\mathbb{Z}\) con
         operaciones unimodulares (Bézout / \(\mathrm{GL}(n,\mathbb{Z})\)).
      3. Completar factores invariantes \(d_1 \mid \cdots \mid d_r\).
      4. Detectar torsión homológica \(\operatorname{Tor}(H_\bullet)\).
      5. Emitir `HomologyReport` hacia la Fase 2.

    El último método de esta fase, `audit_boundary_homology`, es el
    morfismo \(\mathrm{id}_{\partial K}\) que continúa en
    `Phase2QuantumCausalAuditKernel.audit_choi_with_homology`.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        rng_seed: Optional[int] = None,
        snf_max_iterations: int = _DEFAULT_SNF_MAX_ITERATIONS,
        integer_tolerance: float = _DEFAULT_INTEGER_TOLERANCE,
        snf_max_order: int = _DEFAULT_SNF_MAX_ORDER,
    ) -> None:
        if int(dimension_n) <= 0:
            raise ValueError(
                "La dimensión de la frontera debe ser estrictamente positiva."
            )

        self._n = int(dimension_n)
        self._safety_margin = self._finite_float(
            safety_margin, "safety_margin", _MACHINE_EPS
        )
        self._rng = np.random.default_rng(rng_seed)
        self._rng_seed = rng_seed

        self._snf_max_iterations = int(max(1, snf_max_iterations))
        self._integer_tol = self._finite_float(
            integer_tolerance, "integer_tolerance", 0.0
        )
        max_order = int(snf_max_order)
        if max_order <= 0:
            raise ValueError("snf_max_order debe ser entero positivo.")
        self._snf_max_order = max_order

    @staticmethod
    def _finite_float(
        value: float,
        name: str,
        minimum: Optional[float] = None,
        *,
        clamp: bool = True,
    ) -> float:
        """Coacciona a float finito y aplica piso opcional."""
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"{name} debe ser finito.")
        if minimum is not None and x < minimum:
            if clamp:
                return float(minimum)
            raise ValueError(f"{name} debe ser ≥ {minimum}.")
        return x

    @staticmethod
    def _now_utc_iso() -> str:
        """Instante UTC canónico."""
        return _now_utc_iso()

    def _to_integer_matrix(self, A: Any, name: str) -> List[List[int]]:
        r"""
        Valida y convierte una matriz a enteros exactos de Python.

        Rutas:
          - enteros nativos / ``dtype=object``: conversión directa;
          - complejos con parte imaginaria nula (tol);
          - floats a distancia \(\le \tau\) de un entero, con
            \(|x| \le 2^{53}\) (mantissa exacta de float64).

        Rechaza valores no finitos, no enteros o de orden excesivo.
        """
        arr = np.asarray(A)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")

        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        if rows > self._snf_max_order or cols > self._snf_max_order:
            raise ValueError(
                f"{name} excede snf_max_order={self._snf_max_order} "
                f"(forma {rows}×{cols})."
            )

        if rows == 0 or cols == 0:
            return [[int(0) for _ in range(cols)] for _ in range(rows)]

        if arr.dtype == object or np.issubdtype(arr.dtype, np.integer):
            out: List[List[int]] = []
            for i in range(rows):
                row: List[int] = []
                for j in range(cols):
                    try:
                        row.append(int(arr[i, j]))
                    except Exception as exc:
                        raise ValueError(
                            f"{name}[{i},{j}] no es entero exacto."
                        ) from exc
                out.append(row)
            return out

        if np.iscomplexobj(arr):
            imag = np.asarray(arr.imag, dtype=np.float64)
            if not np.all(np.isfinite(imag)):
                raise ValueError(f"{name} contiene imaginarios no finitos.")
            if float(np.max(np.abs(imag))) > self._integer_tol:
                raise ValueError(f"{name} tiene parte imaginaria no nula.")
            work = np.asarray(arr.real, dtype=np.float64)
        else:
            try:
                work = arr.astype(np.float64, copy=False)
            except Exception as exc:
                raise ValueError(f"{name} no pudo convertirse a numérico.") from exc

        if not np.all(np.isfinite(work)):
            raise ValueError(f"{name} contiene valores no finitos.")

        if float(np.max(np.abs(work))) > _FLOAT64_EXACT_INT_MAX:
            raise ValueError(
                f"{name} tiene entradas fuera de la mantissa exacta de float64 "
                f"(≤ 2^53); suministre enteros nativos."
            )

        rounded = np.rint(work)
        if not np.allclose(work, rounded, rtol=0.0, atol=self._integer_tol):
            raise ValueError(f"{name} debe ser una matriz entera.")

        return [[int(x) for x in row] for row in rounded.tolist()]

    @staticmethod
    def _list_to_object_array(M: List[List[int]]) -> np.ndarray:
        """Convierte una lista de listas de enteros a ndarray dtype=object."""
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        arr = np.empty((rows, cols), dtype=object)
        for i, row in enumerate(M):
            for j, val in enumerate(row):
                arr[i, j] = int(val)
        return arr

    @staticmethod
    def _swap_rows(M: List[List[int]], i: int, j: int) -> None:
        if i == j:
            return
        M[i], M[j] = M[j], M[i]

    @staticmethod
    def _swap_cols(M: List[List[int]], i: int, j: int) -> None:
        if i == j:
            return
        for row in M:
            row[i], row[j] = row[j], row[i]

    @staticmethod
    def _negate_row(M: List[List[int]], i: int) -> None:
        M[i] = [-x for x in M[i]]

    @staticmethod
    def _row_subtract_scaled(
        M: List[List[int]], target: int, source: int, scale: int
    ) -> None:
        if scale == 0 or target == source:
            return
        M[target] = [a - scale * b for a, b in zip(M[target], M[source])]

    @staticmethod
    def _col_subtract_scaled(
        M: List[List[int]], target: int, source: int, scale: int
    ) -> None:
        if scale == 0 or target == source:
            return
        for i in range(len(M)):
            M[i][target] -= scale * M[i][source]

    @staticmethod
    def _row_gcd_combine(M: List[List[int]], k: int, i: int) -> None:
        r"""
        Combinación unimodular de filas para reemplazar el pivote por gcd.

        Si \(a = M_{k,k}\), \(b = M_{i,k}\) y \(g = \gcd(a,b) = xa + yb\),
        entonces
        \[
            R_k \leftarrow x R_k + y R_i,
            \qquad
            R_i \leftarrow -\tfrac{b}{g} R_k^{\mathrm{old}}
                         + \tfrac{a}{g} R_i^{\mathrm{old}}.
        \]
        El determinante de la transformación \(2\times 2\) es \(1\), luego
        pertenece a \(\mathrm{SL}(2,\mathbb{Z})\). La segunda fila anula
        \(M_{i,k}\).
        """
        a = M[k][k]
        b = M[i][k]
        g, x, y = ext_gcd(a, b)
        if g == 0:
            return

        old_k = M[k][:]
        old_i = M[i][:]
        n_cols = len(old_k)
        M[k] = [x * old_k[t] + y * old_i[t] for t in range(n_cols)]
        M[i] = [
            -(b // g) * old_k[t] + (a // g) * old_i[t] for t in range(n_cols)
        ]

    @staticmethod
    def _col_gcd_combine(M: List[List[int]], k: int, j: int) -> None:
        r"""
        Combinación unimodular de columnas, dual de `_row_gcd_combine`.
        """
        rows = len(M)
        a = M[k][k]
        b = M[k][j]
        g, x, y = ext_gcd(a, b)
        if g == 0:
            return

        old_k = [M[r][k] for r in range(rows)]
        old_j = [M[r][j] for r in range(rows)]
        for r in range(rows):
            M[r][k] = x * old_k[r] + y * old_j[r]
            M[r][j] = -(b // g) * old_k[r] + (a // g) * old_j[r]

    @staticmethod
    def _off_diagonal_residue(M: List[List[int]]) -> int:
        """Suma de valores absolutos fuera de la diagonal."""
        residue = 0
        rows = len(M)
        cols = len(M[0]) if rows else 0
        for i in range(rows):
            for j in range(cols):
                if i != j:
                    residue += abs(int(M[i][j]))
        return int(residue)

    def compute_smith_normal_form_detailed(
        self,
        A: np.ndarray,
    ) -> Tuple[np.ndarray, SmithNormalFormResult]:
        r"""
        Calcula una Smith Normal Form exacta de una matriz entera sobre
        \(\mathbb{Z}\):
        \[
            S = U A V = \operatorname{diag}(d_1, d_2, \ldots, d_r, 0, \ldots, 0).
        \]

        Operaciones unimodulares:
          - intercambio de filas/columnas;
          - suma de múltiplos enteros de filas/columnas;
          - combinaciones gcd mediante `ext_gcd`.

        Tras la reducción, se extraen factores invariantes por barrido
        \(\gcd/\mathrm{lcm}\) adyacente (equivalencia sobre \(\mathbb{Z}\)).
        """
        M = self._to_integer_matrix(A, "boundary_matrix_integer")
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        reasons: List[str] = []

        if rows == 0 or cols == 0:
            result = SmithNormalFormResult(
                diagonal=(),
                invariant_factors=(),
                rank=0,
                nrows=rows,
                ncols=cols,
                is_complete=True,
                is_divisibility_chain=True,
                iterations=0,
                off_diagonal_residue=0,
                reasons=("empty_boundary_matrix",),
            )
            return self._list_to_object_array(M), result

        iterations = 0
        max_iterations = self._snf_max_iterations
        incomplete = False

        for k in range(min(rows, cols)):
            while True:
                iterations += 1
                if iterations > max_iterations:
                    logger.warning(
                        "Smith Normal Form excedió el límite seguro de iteraciones. "
                        "Se devuelve una forma diagonal parcial."
                    )
                    reasons.append("snf_iteration_limit")
                    incomplete = True
                    break

                pivot_val = None
                pr = pc = k
                for i in range(k, rows):
                    for j in range(k, cols):
                        val = M[i][j]
                        if val != 0:
                            if pivot_val is None or abs(val) < abs(pivot_val):
                                pivot_val = val
                                pr = i
                                pc = j

                if pivot_val is None:
                    break

                if pr != k:
                    self._swap_rows(M, k, pr)
                if pc != k:
                    self._swap_cols(M, k, pc)
                if M[k][k] < 0:
                    self._negate_row(M, k)

                pivot = M[k][k]
                if pivot == 0:
                    break

                changed = False

                for i in range(k + 1, rows):
                    val = M[i][k]
                    if val == 0:
                        continue
                    if val % pivot == 0:
                        self._row_subtract_scaled(M, i, k, val // pivot)
                    else:
                        self._row_gcd_combine(M, k, i)
                        if M[k][k] < 0:
                            self._negate_row(M, k)
                        changed = True
                        break
                if changed:
                    continue

                for j in range(k + 1, cols):
                    val = M[k][j]
                    if val == 0:
                        continue
                    if val % pivot == 0:
                        self._col_subtract_scaled(M, j, k, val // pivot)
                    else:
                        self._col_gcd_combine(M, k, j)
                        if M[k][k] < 0:
                            self._negate_row(M, k)
                        changed = True
                        break
                if changed:
                    continue

                if M[k][k] < 0:
                    self._negate_row(M, k)
                pivot = M[k][k]

                divisibility_changed = False
                if pivot != 0:
                    for i in range(k + 1, rows):
                        for j in range(k + 1, cols):
                            if M[i][j] % pivot != 0:
                                M[k] = [M[k][t] + M[i][t] for t in range(cols)]
                                divisibility_changed = True
                                break
                        if divisibility_changed:
                            break
                if divisibility_changed:
                    continue
                break

            if incomplete:
                break

        for i in range(min(rows, cols)):
            if M[i][i] < 0:
                self._negate_row(M, i)

        residue = self._off_diagonal_residue(M)
        if residue != 0:
            incomplete = True
            reasons.append("snf_off_diagonal_residue")

        diag = tuple(int(M[i][i]) for i in range(min(rows, cols)))
        factors = _invariant_factors_from_diagonal(diag)
        chain_ok = _divisibility_chain_ok(factors)
        if not chain_ok:
            reasons.append("snf_divisibility_chain_broken")
            incomplete = True

        rank = sum(1 for x in diag if int(x) != 0)
        is_complete = (not incomplete) and residue == 0

        result = SmithNormalFormResult(
            diagonal=diag,
            invariant_factors=factors,
            rank=int(rank),
            nrows=rows,
            ncols=cols,
            is_complete=bool(is_complete),
            is_divisibility_chain=bool(chain_ok),
            iterations=int(iterations),
            off_diagonal_residue=int(residue),
            reasons=_unique_reasons(reasons),
        )
        return self._list_to_object_array(M), result

    def compute_smith_normal_form(self, A: np.ndarray) -> np.ndarray:
        r"""
        API compatible: devuelve \(S\) como ``ndarray`` dtype=object.
        """
        matrix, _ = self.compute_smith_normal_form_detailed(A)
        return matrix

    def verify_boundary_torsion(
        self,
        boundary_matrix_integer: np.ndarray,
    ) -> Tuple[bool, List[int]]:
        r"""
        API de compatibilidad con la versión original.

        Audita que la homología de frontera esté libre de torsión:
        \[
            \operatorname{Tor}(H_k(\partial K;\mathbb{Z})) = 0.
        \]

        Devuelve:
          - `is_coherent`
          - `torsion_coefficients`
        """
        report = self.audit_boundary_homology(boundary_matrix_integer)
        return report.is_coherent, list(report.torsion_coefficients)

    def audit_boundary_homology(
        self, boundary_matrix_integer: np.ndarray
    ) -> HomologyReport:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1.

        Calcula la SNF, extrae factores invariantes y coeficientes de
        torsión, y estima nulidad / rango libre del conúcleo.

        Este método devuelve `HomologyReport`, objeto que continúa
        formalmente en la Fase 2 mediante

        \[
            \texttt{audit\_choi\_with\_homology}
            :
            \mathsf{HomologyReport}
            \times
            C_{\mathcal{E}}
            \longrightarrow
            \mathsf{CausalConsolidation}.
        \]
        """
        try:
            _S, snf = self.compute_smith_normal_form_detailed(
                boundary_matrix_integer
            )
            torsion = tuple(int(x) for x in snf.invariant_factors if int(x) > 1)
            has_torsion = len(torsion) > 0
            reasons = list(snf.reasons)
            if has_torsion:
                reasons.append("smith_torsion_detected")
            if not snf.is_complete:
                reasons.append("snf_incomplete")

            is_coherent = (not has_torsion) and bool(snf.is_complete)

            nullity = max(int(snf.ncols) - int(snf.rank), 0)
            free_coker = max(int(snf.nrows) - int(snf.rank), 0)

            return HomologyReport(
                is_coherent=bool(is_coherent),
                torsion_coefficients=torsion,
                smith_diagonal=snf.diagonal,
                matrix_rank=int(snf.rank),
                reasons=_unique_reasons(reasons),
                invariant_factors=snf.invariant_factors,
                snf_complete=bool(snf.is_complete),
                divisibility_chain_ok=bool(snf.is_divisibility_chain),
                nrows=int(snf.nrows),
                ncols=int(snf.ncols),
                nullity=int(nullity),
                free_cokernel_rank=int(free_coker),
                snf_iterations=int(snf.iterations),
                has_torsion=bool(has_torsion),
            )

        except Exception as exc:
            return HomologyReport(
                is_coherent=False,
                torsion_coefficients=(),
                smith_diagonal=(),
                matrix_rank=0,
                reasons=_unique_reasons(
                    (f"snf_failure: {type(exc).__name__}: {exc}",)
                ),
                invariant_factors=(),
                snf_complete=False,
                divisibility_chain_ok=False,
                nrows=0,
                ncols=0,
                nullity=0,
                free_cokernel_rank=0,
                snf_iterations=0,
                has_torsion=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# FASE 2 — ADUANA CUÁNTICA DE CHOI–JAMIOŁKOWSKI (CPTP)
# ════════════════════════════════════════════════════════════════════════════


class Phase2QuantumCausalAuditKernel(Phase1HomologicalAuditKernel):
    r"""
    FASE 2: Auditoría cuántico-causal.

    El primer método, `audit_choi_with_homology`, consume el
    `HomologyReport` producido por `audit_boundary_homology` (Fase 1).

    Responsabilidades:
      1. Recibir `HomologyReport` desde la Fase 1.
      2. Validar la matriz de Choi (forma \(d^2\times d^2\), finitud).
      3. Verificar hermiticidad (HP), CP y TP con cotas de Weyl.
      4. Estimar rango de Kraus, unitalidad y brecha espectral.
      5. Consolidar homología + causalidad cuántica en `CausalConsolidation`.

    El último método, `prepare_causal_consolidation`, entrega
    `CausalConsolidation`, insumo directo de
    `Phase3BellHeytingActuationKernel.audit_bell_with_causality`.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        choi_cp_veto_tolerance: float = 1.0e-4,
        choi_tp_veto_tolerance: float = 1.0e-4,
        choi_hermitian_veto_tolerance: float = 1.0e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(dimension_n, safety_margin, **kwargs)

        self._choi_cp_veto_tol = self._finite_float(
            choi_cp_veto_tolerance,
            "choi_cp_veto_tolerance",
            0.0,
        )
        self._choi_tp_veto_tol = self._finite_float(
            choi_tp_veto_tolerance,
            "choi_tp_veto_tolerance",
            0.0,
        )
        self._choi_herm_veto_tol = self._finite_float(
            choi_hermitian_veto_tolerance,
            "choi_hermitian_veto_tolerance",
            0.0,
        )

    def _fail_choi_report(self, reason: str) -> ChoiReport:
        """Reporte fail-safe para Choi inválido (fail-closed)."""
        return ChoiReport(
            is_cptp=False,
            is_cp=False,
            is_tp=False,
            is_hermitian=False,
            min_eigenvalue=-math.inf,
            tp_residual=math.inf,
            trace_residual=math.inf,
            hermitian_residual=math.inf,
            cp_tolerance=float(self._choi_cp_veto_tol),
            tp_tolerance=float(self._choi_tp_veto_tol),
            hermitian_tolerance=float(self._choi_herm_veto_tol),
            reasons=_unique_reasons((reason,)),
            dimension=self._n,
            kraus_rank=0,
            max_eigenvalue=math.inf,
            choi_trace=math.nan,
            unital_residual=math.inf,
            is_unital=False,
            hermitian_relative_residual=math.inf,
            nuclear_norm=math.inf,
            frobenius_norm=math.inf,
            spectral_gap=0.0,
        )

    @staticmethod
    def _weyl_tolerance(scale: float, dimension: int, extra: float = 100.0) -> float:
        """Piso de Weyl \(\tau \ge \varepsilon\, n_{\mathrm{eff}} \|C\|\)."""
        n_eff = max(1, int(dimension))
        return max(1.0e-12, extra * _MACHINE_EPS * max(scale, 1.0) * float(n_eff))

    def audit_choi_cptp(self, Choi: np.ndarray) -> ChoiReport:
        r"""
        Auditoría CPTP rigurosa de la matriz de Choi.

        CP:
        \[
            C_{\mathcal{E}} \succeq 0.
        \]

        TP (traza parcial sobre el sistema de salida, conv. documentada):
        \[
            \operatorname{Tr}_2(C_{\mathcal{E}}) = I_d.
        \]

        HP / hermiticidad:
        \[
            \|C - C^\dagger\|_F \le \tau_H.
        \]

        Normalización:
        \[
            \operatorname{Tr}(C_{\mathcal{E}}) = d
            \qquad\text{(consecuencia de TP)}.
        \]

        El rango de Kraus es el rango numérico de \(C_{\mathcal{E}}\).
        La unitalidad \(\operatorname{Tr}_1(C)=I\) se reporta pero no veta
        por sí sola (el canal físico exigido es TP, no necesariamente unital).
        """
        d = self._n
        expected_shape = (d * d, d * d)
        Choi_arr = np.array(Choi, dtype=np.complex128, copy=True, order="C")

        if Choi_arr.shape != expected_shape:
            raise ValueError(
                f"La matriz de Choi debe tener dimensión {expected_shape} "
                f"para dimension_n={d}."
            )

        if not np.all(np.isfinite(Choi_arr)):
            return self._fail_choi_report("choi_contains_non_finite_values")

        norm_c = float(la.norm(Choi_arr, ord="fro"))
        scale = max(1.0, norm_c)
        dim_choi = int(d * d)

        herm_tol = self._weyl_tolerance(scale, dim_choi)
        herm_diff = float(la.norm(Choi_arr - Choi_arr.conj().T, ord="fro"))
        herm_rel = _safe_ratio(herm_diff, norm_c + _MACHINE_EPS)

        C = 0.5 * (Choi_arr + Choi_arr.conj().T)

        try:
            eigs = la.eigvalsh(C, check_finite=True)
            eigs = np.asarray(eigs, dtype=np.float64)
            min_eig = float(np.min(eigs)) if eigs.size else 0.0
            max_eig = float(np.max(eigs)) if eigs.size else 0.0
        except Exception as exc:
            return self._fail_choi_report(
                f"choi_eigh_failure: {type(exc).__name__}: {exc}"
            )

        cp_tol = self._weyl_tolerance(scale, dim_choi)
        is_cp = min_eig >= -cp_tol

        try:
            C4 = C.reshape(d, d, d, d)
            tr_2 = np.trace(C4, axis1=1, axis2=3)
            tp_residual = float(la.norm(tr_2 - np.eye(d, dtype=np.complex128), ord="fro"))
            tr_1 = np.trace(C4, axis1=0, axis2=2)
            unital_residual = float(
                la.norm(tr_1 - np.eye(d, dtype=np.complex128), ord="fro")
            )
        except Exception as exc:
            return self._fail_choi_report(
                f"choi_partial_trace_failure: {type(exc).__name__}: {exc}"
            )

        tp_tol = max(1.0e-10, self._weyl_tolerance(scale, d))
        is_tp = math.isfinite(tp_residual) and tp_residual <= tp_tol
        is_unital = math.isfinite(unital_residual) and unital_residual <= tp_tol

        trace_val = float(np.real(np.trace(C)))
        trace_residual = float(abs(trace_val - float(d)))
        is_hermitian = herm_diff <= herm_tol
        is_cptp = bool(is_cp and is_tp and is_hermitian)

        eig_tol = max(cp_tol, 100.0 * _MACHINE_EPS * max(abs(max_eig), 1.0) * dim_choi)
        kraus_rank = int(np.count_nonzero(eigs > eig_tol))
        nuclear = float(np.sum(np.abs(eigs))) if eigs.size else 0.0

        positive = eigs[eigs > eig_tol]
        if positive.size >= 2 and max_eig > 0.0:
            spectral_gap = float(positive[0] / max(abs(max_eig), _MACHINE_EPS))
        elif positive.size == 1 and max_eig > 0.0:
            spectral_gap = 1.0
        else:
            spectral_gap = 0.0

        reasons: List[str] = []
        if not is_hermitian:
            reasons.append("choi_not_hermitian")
        if not is_cp:
            reasons.append("choi_not_completely_positive")
        if not is_tp:
            reasons.append("choi_not_trace_preserving")
        if trace_residual > tp_tol:
            reasons.append("choi_trace_not_normalized")
        if not is_unital:
            reasons.append("choi_not_unital")
        if herm_rel > 1.0e-2:
            reasons.append("choi_hermitian_relative_residual_large")

        return ChoiReport(
            is_cptp=is_cptp,
            is_cp=is_cp,
            is_tp=is_tp,
            is_hermitian=is_hermitian,
            min_eigenvalue=min_eig,
            tp_residual=tp_residual,
            trace_residual=trace_residual,
            hermitian_residual=herm_diff,
            cp_tolerance=cp_tol,
            tp_tolerance=tp_tol,
            hermitian_tolerance=herm_tol,
            reasons=_unique_reasons(reasons),
            dimension=d,
            kraus_rank=kraus_rank,
            max_eigenvalue=max_eig,
            choi_trace=trace_val,
            unital_residual=unital_residual,
            is_unital=is_unital,
            hermitian_relative_residual=float(herm_rel),
            nuclear_norm=nuclear,
            frobenius_norm=float(norm_c),
            spectral_gap=float(np.clip(spectral_gap, 0.0, 1.0)),
        )

    def verify_choi_cptp(self, Choi: np.ndarray) -> Tuple[bool, float, float]:
        r"""
        API de compatibilidad con la versión original.

        Devuelve:
          - `is_cptp`
          - `min_eigenvalue`
          - `tp_residual`
        """
        report = self.audit_choi_cptp(Choi)
        return report.is_cptp, report.min_eigenvalue, report.tp_residual

    def audit_choi_with_homology(
        self,
        homology_report: HomologyReport,
        Choi_matrix: np.ndarray,
    ) -> CausalConsolidation:
        r"""
        PRIMER MÉTODO DE LA FASE 2.

        Continuación formal de `audit_boundary_homology` (Fase 1).

        Consume `HomologyReport` y audita la matriz de Choi. El resultado
        se consolida en `CausalConsolidation` mediante el último método
        de esta fase.
        """
        try:
            choi_report = self.audit_choi_cptp(Choi_matrix)
        except Exception as exc:
            choi_report = self._fail_choi_report(
                f"choi_audit_failure: {type(exc).__name__}: {exc}"
            )
        return self.prepare_causal_consolidation(homology_report, choi_report)

    def prepare_causal_consolidation(
        self,
        homology_report: HomologyReport,
        choi_report: ChoiReport,
    ) -> CausalConsolidation:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2.

        Consolida `HomologyReport` y `ChoiReport` en `CausalConsolidation`,
        objeto que continúa formalmente en la Fase 3 mediante

        \[
            \texttt{audit\_bell\_with\_causality}
            :
            \mathsf{CausalConsolidation}
            \times
            (E_{11},E_{12},E_{21},E_{22})
            \longrightarrow
            \mathsf{HeytingDecisionReport}.
        \]
        """
        reasons = _unique_reasons(
            list(homology_report.reasons) + list(choi_report.reasons)
        )
        admissible = bool(homology_report.is_coherent and choi_report.is_cptp)
        return CausalConsolidation(
            homology=homology_report,
            choi=choi_report,
            reasons=reasons,
            is_causally_admissible=admissible,
        )


# ════════════════════════════════════════════════════════════════════════════
# FASE 3 — BELL–CHSH, HEYTING Ω₃ Y ACTUACIÓN CIBER-FÍSICA
# ════════════════════════════════════════════════════════════════════════════


class Phase3BellHeytingActuationKernel(Phase2QuantumCausalAuditKernel):
    r"""
    FASE 3: Bell–CHSH, Heyting y actuación.

    El primer método, `audit_bell_with_causality`, consume el
    `CausalConsolidation` producido por `prepare_causal_consolidation`
    (Fase 2).

    Responsabilidades:
      1. Recibir `CausalConsolidation` desde la Fase 2.
      2. Auditar correlaciones Bell–CHSH (cuatro patrones de signo,
         cotas clásica / Tsirelson / algebraica, fisicalidad \(|E|\le 1\)).
      3. Consolidar veredicto final en \(\Omega_3\) por ínfimo de Gödel.
      4. Activar crowbar perimetral si el veredicto es VETOED (fail-closed).
      5. Emitir diccionario compatible y certificado firmado SHA-256.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        crowbar_base_latency_ns: float = _CROWBAR_IRAM_LATENCY_NS,
        crowbar_jitter_ns: float = 3.5,
        crowbar_min_latency_ns: float = 380.0,
        crowbar_max_latency_ns: float = 420.0,
        gpio_pin: str = "GPIO14",
        async_timeout_s: float = 5.0,
        max_batch: int = _DEFAULT_MAX_BATCH,
        **kwargs: Any,
    ) -> None:
        super().__init__(dimension_n, safety_margin, **kwargs)

        self._crowbar_base_latency_ns = self._finite_float(
            crowbar_base_latency_ns,
            "crowbar_base_latency_ns",
            0.0,
        )
        self._crowbar_jitter_ns = self._finite_float(
            crowbar_jitter_ns,
            "crowbar_jitter_ns",
            0.0,
        )
        self._crowbar_min_latency_ns = self._finite_float(
            crowbar_min_latency_ns,
            "crowbar_min_latency_ns",
            0.0,
        )
        self._crowbar_max_latency_ns = self._finite_float(
            crowbar_max_latency_ns,
            "crowbar_max_latency_ns",
            self._crowbar_min_latency_ns,
        )
        if self._crowbar_max_latency_ns < self._crowbar_min_latency_ns:
            self._crowbar_max_latency_ns = self._crowbar_min_latency_ns

        self._gpio_pin = str(gpio_pin)
        self._async_timeout_s = self._finite_float(
            async_timeout_s,
            "async_timeout_s",
            0.0,
        )
        max_batch_n = int(max_batch)
        if max_batch_n <= 0:
            raise ValueError("max_batch debe ser entero positivo.")
        self._max_batch = max_batch_n

    @staticmethod
    def _chsh_canonical(e11: float, e12: float, e21: float, e22: float) -> float:
        r"""Patrón canónico \(|E_{11}+E_{12}+E_{21}-E_{22}|\)."""
        return float(abs(e11 + e12 + e21 - e22))

    @staticmethod
    def _chsh_max_sign_patterns(
        e11: float, e12: float, e21: float, e22: float
    ) -> float:
        r"""
        Máximo CHSH sobre los cuatro patrones con un único minus:

        \[
            |E_{11}+E_{12}+E_{21}-E_{22}|,\;
            |E_{11}+E_{12}-E_{21}+E_{22}|,\;
            |E_{11}-E_{12}+E_{21}+E_{22}|,\;
            |-E_{11}+E_{12}+E_{21}+E_{22}|.
        \]
        """
        patterns = (
            abs(e11 + e12 + e21 - e22),
            abs(e11 + e12 - e21 + e22),
            abs(e11 - e12 + e21 + e22),
            abs(-e11 + e12 + e21 + e22),
        )
        return float(max(patterns))

    def audit_bell_chsh(
        self,
        E_11: float,
        E_12: float,
        E_21: float,
        E_22: float,
    ) -> BellReport:
        r"""
        Audita correlaciones no locales Bell–CHSH.

        Fisicalidad: \(|E_{ij}| \le 1\).
        Coherencia cuántica: \(\mathcal{B} \le 2\sqrt{2}\).
        Clasicidad: \(\mathcal{B} \le 2\).
        Super-cuántico: \(2\sqrt{2} < \mathcal{B} \le 4\) (veto).
        """
        try:
            e11 = float(E_11)
            e12 = float(E_12)
            e21 = float(E_21)
            e22 = float(E_22)
        except Exception:
            return BellReport(
                is_coherent=False,
                is_classical=False,
                chsh_value=math.inf,
                reasons=("bell_correlations_not_numeric",),
                correlations_physical=False,
            )

        if not all(math.isfinite(x) for x in (e11, e12, e21, e22)):
            return BellReport(
                is_coherent=False,
                is_classical=False,
                chsh_value=math.inf,
                reasons=("bell_correlations_not_finite",),
                correlations=(e11, e12, e21, e22),
                correlations_physical=False,
            )

        tol = max(1.0e-12, 100.0 * _MACHINE_EPS)
        physical = all(abs(x) <= _CORRELATION_ABS_MAX + tol for x in (e11, e12, e21, e22))

        chsh_can = self._chsh_canonical(e11, e12, e21, e22)
        chsh_val = self._chsh_max_sign_patterns(e11, e12, e21, e22)

        is_tsirelson = chsh_val <= _TSIRELSON_BOUND + tol
        is_classical = chsh_val <= _CLASSICAL_CHSH_BOUND + tol

        reasons: List[str] = []
        if not physical:
            reasons.append("bell_correlations_outside_minus_one_one")
        if chsh_val > _ALGEBRAIC_CHSH_BOUND + tol:
            reasons.append("bell_chsh_above_algebraic_bound")
        if not is_tsirelson:
            reasons.append("bell_chsh_above_tsirelson_bound")
        elif not is_classical:
            reasons.append("bell_chsh_above_classical_bound")

        is_coherent = bool(is_tsirelson and physical)

        return BellReport(
            is_coherent=is_coherent,
            is_classical=bool(is_classical and physical),
            chsh_value=chsh_val,
            reasons=_unique_reasons(reasons),
            correlations=(e11, e12, e21, e22),
            correlations_physical=bool(physical),
            chsh_canonical=chsh_can,
            classical_margin=float(_CLASSICAL_CHSH_BOUND - chsh_val),
            tsirelson_margin=float(_TSIRELSON_BOUND - chsh_val),
        )

    def verify_bell_chsh(
        self,
        E_11: float,
        E_12: float,
        E_21: float,
        E_22: float,
    ) -> Tuple[bool, float]:
        r"""
        API de compatibilidad con la versión original.

        Devuelve:
          - `is_coherent` respecto a la cota de Tsirelson (y fisicalidad).
          - `chsh_value` (máximo sobre patrones de signo).
        """
        report = self.audit_bell_chsh(E_11, E_12, E_21, E_22)
        return report.is_coherent, report.chsh_value

    def audit_bell_with_causality(
        self,
        consolidation: CausalConsolidation,
        bell_correlations: Tuple[float, float, float, float],
    ) -> HeytingDecisionReport:
        r"""
        PRIMER MÉTODO DE LA FASE 3.

        Continuación formal de `prepare_causal_consolidation` (Fase 2).

        Consume `CausalConsolidation` y audita Bell–CHSH. El veredicto
        \(\Omega_3\) se decide por ínfimo de Gödel de las tres aduanas.
        """
        try:
            if len(bell_correlations) != 4:
                raise ValueError("bell_correlations debe tener exactamente 4 valores.")
            E_11, E_12, E_21, E_22 = bell_correlations
            bell_report = self.audit_bell_chsh(E_11, E_12, E_21, E_22)
        except Exception as exc:
            bell_report = BellReport(
                is_coherent=False,
                is_classical=False,
                chsh_value=math.inf,
                reasons=_unique_reasons(
                    (f"bell_audit_failure: {type(exc).__name__}: {exc}",)
                ),
                correlations_physical=False,
            )
        return self._decide_heyting(consolidation, bell_report)

    def _atomic_heyting_propositions(
        self,
        hom: HomologyReport,
        choi: ChoiReport,
        bell: BellReport,
    ) -> List[Tuple[str, float]]:
        r"""
        Proposiciones atómicas de las tres aduanas en \(\Omega_3\).

        Un átomo 0 veta; un átomo 1/2 degrada; 1 es regular. El veredicto
        global es el ínfimo de Gödel de esta familia.
        """
        atoms: List[Tuple[str, float]] = []

        atoms.append(
            ("homology", _OMEGA3_TRUE if hom.is_coherent else _OMEGA3_FALSE)
        )
        if not hom.snf_complete:
            atoms.append(("snf_complete", _OMEGA3_FALSE))
        else:
            atoms.append(("snf_complete", _OMEGA3_TRUE))

        if choi.min_eigenvalue < -self._choi_cp_veto_tol:
            atoms.append(("choi_cp", _OMEGA3_FALSE))
        elif choi.min_eigenvalue < -choi.cp_tolerance:
            atoms.append(("choi_cp", _OMEGA3_MIDDLE))
        else:
            atoms.append(("choi_cp", _OMEGA3_TRUE))

        if choi.tp_residual > self._choi_tp_veto_tol:
            atoms.append(("choi_tp", _OMEGA3_FALSE))
        elif choi.tp_residual > choi.tp_tolerance:
            atoms.append(("choi_tp", _OMEGA3_MIDDLE))
        else:
            atoms.append(("choi_tp", _OMEGA3_TRUE))

        if choi.hermitian_residual > self._choi_herm_veto_tol:
            atoms.append(("choi_hermitian", _OMEGA3_FALSE))
        elif not choi.is_hermitian:
            atoms.append(("choi_hermitian", _OMEGA3_MIDDLE))
        else:
            atoms.append(("choi_hermitian", _OMEGA3_TRUE))

        if not choi.is_cptp:
            atoms.append(("choi_cptp", _OMEGA3_MIDDLE))
        else:
            atoms.append(("choi_cptp", _OMEGA3_TRUE))

        if not bell.correlations_physical or not bell.is_coherent:
            atoms.append(("bell_tsirelson", _OMEGA3_FALSE))
        else:
            atoms.append(("bell_tsirelson", _OMEGA3_TRUE))

        if bell.is_classical:
            atoms.append(("bell_classical", _OMEGA3_TRUE))
        elif bell.is_coherent:
            atoms.append(("bell_classical", _OMEGA3_MIDDLE))
        else:
            atoms.append(("bell_classical", _OMEGA3_FALSE))

        return atoms

    def _decide_heyting(
        self,
        consolidation: CausalConsolidation,
        bell_report: BellReport,
    ) -> HeytingDecisionReport:
        """Consolida las tres aduanas en el retículo de Heyting \(\Omega_3\)."""
        hom = consolidation.homology
        choi = consolidation.choi
        bell = bell_report

        reasons: List[str] = list(consolidation.reasons) + list(bell.reasons)
        atoms = self._atomic_heyting_propositions(hom, choi, bell)

        truth_lattice = _OMEGA3_TRUE
        for name, value in atoms:
            truth_lattice = _heyting_meet(truth_lattice, value)
            if value == _OMEGA3_FALSE:
                reasons.append(f"atom_veto:{name}")
            elif value == _OMEGA3_MIDDLE:
                reasons.append(f"atom_degraded:{name}")

        if truth_lattice == _OMEGA3_FALSE:
            verdict = _VERDICT_VETOED
        elif truth_lattice == _OMEGA3_MIDDLE:
            verdict = _VERDICT_DEGRADED
        else:
            verdict = _VERDICT_COHERENT

        margins: List[float] = []
        if math.isfinite(choi.min_eigenvalue):
            margins.append(
                float(
                    np.clip(
                        (choi.min_eigenvalue + self._choi_cp_veto_tol)
                        / max(self._choi_cp_veto_tol, _MACHINE_EPS),
                        0.0,
                        1.0,
                    )
                )
            )
        else:
            margins.append(0.0)

        if math.isfinite(choi.tp_residual):
            margins.append(
                float(
                    np.clip(
                        1.0 - choi.tp_residual / max(self._choi_tp_veto_tol, _MACHINE_EPS),
                        0.0,
                        1.0,
                    )
                )
            )
        else:
            margins.append(0.0)

        if math.isfinite(bell.chsh_value):
            margins.append(
                float(
                    np.clip(
                        1.0
                        - bell.chsh_value
                        / max(_TSIRELSON_BOUND, _MACHINE_EPS),
                        0.0,
                        1.0,
                    )
                )
            )
        else:
            margins.append(0.0)

        geo = _geometric_mean(margins)
        if verdict == _VERDICT_VETOED:
            truth_continuous = 0.0
        elif verdict == _VERDICT_DEGRADED:
            truth_continuous = 0.5 * geo
        else:
            truth_continuous = 0.5 + 0.5 * geo

        if not reasons:
            reasons.append("boundary_regular")

        return HeytingDecisionReport(
            verdict=verdict,
            truth_value=_TRUTH_VALUES[verdict],
            homology=hom,
            choi=choi,
            bell=bell,
            reasons=_unique_reasons(reasons),
            truth_continuous=float(truth_continuous),
            heyting_implies_coherent=_heyting_implies(
                _TRUTH_VALUES[verdict], _OMEGA3_TRUE
            ),
            atomic_propositions=tuple(atoms),
        )

    def _fire_crowbar(self, reason: str) -> AuditActuationReport:
        """Dispara el crowbar simulado con jitter gaussiano acotado."""
        if self._crowbar_jitter_ns > 0.0:
            jitter = float(self._rng.normal(0.0, self._crowbar_jitter_ns))
        else:
            jitter = 0.0

        latency = self._crowbar_base_latency_ns + jitter
        latency = float(
            np.clip(
                latency,
                self._crowbar_min_latency_ns,
                self._crowbar_max_latency_ns,
            )
        )

        logger.critical(
            "¡RUPTURA DE LA COHERENCIA CAUSAL EN LA FRONTERA! "
            "Satélite de auditoría detectó intrusión semántica o colusión monopólica. "
            "Disyuntor perimetral BT151 [%s] gatillado en IRAM. Latencia: %.2f ns. "
            "Razón: %s",
            self._gpio_pin,
            latency,
            reason,
        )

        return AuditActuationReport(
            interlock_fired=True,
            actuation_latency_ns=latency,
            gpio_pin=self._gpio_pin,
            reason=reason,
        )

    def _no_actuation(self, reason: str) -> AuditActuationReport:
        """Inhibe el interlock físico."""
        return AuditActuationReport(
            interlock_fired=False,
            actuation_latency_ns=0.0,
            gpio_pin=self._gpio_pin,
            reason=reason,
        )

    def act_on_heyting_decision(
        self, decision: HeytingDecisionReport
    ) -> AuditActuationReport:
        """Actúa físicamente según el veredicto de Heyting (fail-closed)."""
        if decision.verdict == _VERDICT_VETOED:
            reason = ";".join(decision.reasons)[:200]
            return self._fire_crowbar(reason)
        return self._no_actuation("boundary_not_vetoed")

    def _decision_fingerprint(
        self,
        decision: HeytingDecisionReport,
        actuation: AuditActuationReport,
    ) -> str:
        """Huella isomorfa (sin reloj) del par (decisión, actuación lógica)."""
        return _stable_sha256(
            {
                "verdict": decision.verdict,
                "truth_value": decision.truth_value,
                "torsion": list(decision.homology.torsion_coefficients),
                "smith_diagonal": list(decision.homology.smith_diagonal),
                "invariant_factors": list(decision.homology.invariant_factors),
                "smith_rank": decision.homology.matrix_rank,
                "choi_min_eig": decision.choi.min_eigenvalue,
                "choi_tp": decision.choi.tp_residual,
                "choi_herm": decision.choi.hermitian_residual,
                "choi_kraus_rank": decision.choi.kraus_rank,
                "chsh": decision.bell.chsh_value,
                "bell_classical": decision.bell.is_classical,
                "interlock_fired": bool(actuation.interlock_fired),
                "gpio_pin": self._gpio_pin,
                "dimension": self._n,
                "reasons": list(decision.reasons),
            }
        )

    def synthesize_audit_certificate(
        self,
        decision: HeytingDecisionReport,
        actuation: AuditActuationReport,
    ) -> AuditSatelliteCertificate:
        """Emite el certificado de aduana (isomorfo + no-repudio)."""
        timestamp_utc = self._now_utc_iso()
        decision_sha = self._decision_fingerprint(decision, actuation)
        signature = _stable_sha256(
            {
                "decision_sha256": decision_sha,
                "timestamp_utc": timestamp_utc,
                "actuation_latency_ns": float(actuation.actuation_latency_ns),
                "phase": _AUDIT_PHASE,
                "version": __version__,
            }
        )

        if decision.verdict == _VERDICT_DEGRADED:
            logger.warning(
                "Aduana degradada. Homología=%s | Choi CPTP=%s | CHSH=%.6f | "
                "Razones=%s",
                decision.homology.is_coherent,
                decision.choi.is_cptp,
                decision.bell.chsh_value,
                ";".join(decision.reasons),
            )
        elif decision.verdict == _VERDICT_COHERENT:
            logger.info(
                "Aduana coherente. rk(∂)=%d | λ_min(C)=%.6e | CHSH=%.6f",
                decision.homology.matrix_rank,
                decision.choi.min_eigenvalue,
                decision.bell.chsh_value,
            )

        return AuditSatelliteCertificate(
            phase=_AUDIT_PHASE,
            heyting_verdict=decision.verdict,
            smith_torsion_coefficients=decision.homology.torsion_coefficients,
            choi_minimum_eigenvalue=float(decision.choi.min_eigenvalue),
            choi_trace_preserving_residual=float(decision.choi.tp_residual),
            bell_chsh_parameter=float(decision.bell.chsh_value),
            hardware_interlock_fired=bool(actuation.interlock_fired),
            actuation_latency_ns=float(actuation.actuation_latency_ns),
            digital_signature_sha256=signature,
            is_homology_coherent=bool(decision.homology.is_coherent),
            smith_diagonal=decision.homology.smith_diagonal,
            smith_matrix_rank=int(decision.homology.matrix_rank),
            choi_is_cptp=bool(decision.choi.is_cptp),
            choi_is_cp=bool(decision.choi.is_cp),
            choi_is_tp=bool(decision.choi.is_tp),
            choi_is_hermitian=bool(decision.choi.is_hermitian),
            choi_trace_residual=float(decision.choi.trace_residual),
            choi_hermitian_residual=float(decision.choi.hermitian_residual),
            bell_is_classical=bool(decision.bell.is_classical),
            bell_classical_bound=float(decision.bell.classical_bound),
            bell_tsirelson_bound=float(decision.bell.tsirelson_bound),
            heyting_truth_value=float(decision.truth_value),
            gpio_pin=actuation.gpio_pin,
            timestamp_utc=timestamp_utc,
            reasons=decision.reasons,
            decision_sha256=decision_sha,
            invariant_factors=decision.homology.invariant_factors,
            choi_kraus_rank=int(decision.choi.kraus_rank),
            bell_tsirelson_margin=float(decision.bell.tsirelson_margin),
        )

    def verify_audit_certificate(
        self, certificate: AuditSatelliteCertificate
    ) -> bool:
        """Verifica la huella isomorfa reconstruible de un certificado."""
        try:
            expected = _stable_sha256(
                {
                    "verdict": certificate.heyting_verdict,
                    "truth_value": certificate.heyting_truth_value,
                    "torsion": list(certificate.smith_torsion_coefficients),
                    "smith_diagonal": list(certificate.smith_diagonal),
                    "invariant_factors": list(certificate.invariant_factors),
                    "smith_rank": certificate.smith_matrix_rank,
                    "choi_min_eig": certificate.choi_minimum_eigenvalue,
                    "choi_tp": certificate.choi_trace_preserving_residual,
                    "choi_herm": certificate.choi_hermitian_residual,
                    "choi_kraus_rank": certificate.choi_kraus_rank,
                    "chsh": certificate.bell_chsh_parameter,
                    "bell_classical": certificate.bell_is_classical,
                    "interlock_fired": bool(certificate.hardware_interlock_fired),
                    "gpio_pin": certificate.gpio_pin or self._gpio_pin,
                    "dimension": self._n,
                    "reasons": list(certificate.reasons),
                }
            )
            return certificate.decision_sha256 == expected
        except Exception:
            return False

    def _build_audit_dict(
        self,
        decision: HeytingDecisionReport,
        actuation: AuditActuationReport,
        certificate: AuditSatelliteCertificate,
    ) -> Dict[str, Any]:
        """Construye el diccionario de auditoría compatible y extendido."""
        return {
            "heyting_verdict": decision.verdict,
            "smith_torsion_coefficients": list(decision.homology.torsion_coefficients),
            "choi_minimum_eigenvalue": decision.choi.min_eigenvalue,
            "choi_trace_preserving_residual": decision.choi.tp_residual,
            "bell_chsh_parameter": decision.bell.chsh_value,
            "hardware_interlock_fired": actuation.interlock_fired,
            "actuation_latency_ns": actuation.actuation_latency_ns,
            "is_homology_coherent": decision.homology.is_coherent,
            "smith_diagonal": list(decision.homology.smith_diagonal),
            "smith_matrix_rank": decision.homology.matrix_rank,
            "invariant_factors": list(decision.homology.invariant_factors),
            "snf_complete": decision.homology.snf_complete,
            "homology_nullity": decision.homology.nullity,
            "homology_free_cokernel_rank": decision.homology.free_cokernel_rank,
            "choi_is_cptp": decision.choi.is_cptp,
            "choi_is_cp": decision.choi.is_cp,
            "choi_is_tp": decision.choi.is_tp,
            "choi_is_hermitian": decision.choi.is_hermitian,
            "choi_is_unital": decision.choi.is_unital,
            "choi_trace_residual": decision.choi.trace_residual,
            "choi_hermitian_residual": decision.choi.hermitian_residual,
            "choi_unital_residual": decision.choi.unital_residual,
            "choi_kraus_rank": decision.choi.kraus_rank,
            "choi_spectral_gap": decision.choi.spectral_gap,
            "bell_is_classical": decision.bell.is_classical,
            "bell_classical_bound": decision.bell.classical_bound,
            "bell_tsirelson_bound": decision.bell.tsirelson_bound,
            "bell_algebraic_bound": decision.bell.algebraic_bound,
            "bell_tsirelson_margin": decision.bell.tsirelson_margin,
            "bell_correlations_physical": decision.bell.correlations_physical,
            "heyting_truth_value": decision.truth_value,
            "heyting_truth_continuous": decision.truth_continuous,
            "gpio_pin": actuation.gpio_pin,
            "timestamp_utc": certificate.timestamp_utc,
            "reasons": list(decision.reasons),
            "decision_sha256": certificate.decision_sha256,
            "digital_signature_sha256": certificate.digital_signature_sha256,
            "phase": certificate.phase,
            "version": __version__,
            "dimension_n": self._n,
        }

    def execute_audit_cycle(
        self,
        boundary_matrix: np.ndarray,
        Choi_matrix: np.ndarray,
        bell_correlations: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        r"""
        API síncrona principal compatible con la versión original.

        Ejecuta las tres aduanas anidadas:
        \[
            \text{Fase 1}
            \xrightarrow{\texttt{audit\_boundary\_homology}}
            \text{Fase 2}
            \xrightarrow{\texttt{audit\_choi\_with\_homology}}
            \texttt{prepare\_causal\_consolidation}
            \xrightarrow{\texttt{audit\_bell\_with\_causality}}
            \text{Fase 3}.
        \]

        Devuelve un diccionario compatible con la versión original, extendido
        con metadatos de auditoría doctoral y huellas SHA-256.
        """
        try:
            homology_report = self.audit_boundary_homology(boundary_matrix)
            consolidation = self.audit_choi_with_homology(
                homology_report,
                Choi_matrix,
            )
            decision = self.audit_bell_with_causality(
                consolidation,
                bell_correlations,
            )
            actuation = self.act_on_heyting_decision(decision)
            certificate = self.synthesize_audit_certificate(decision, actuation)
            return self._build_audit_dict(decision, actuation, certificate)

        except Exception as exc:
            logger.exception(
                "Fallo inesperado en el ciclo de auditoría; activando VETO fail-safe."
            )
            actuation = self._fire_crowbar(
                f"audit_exception: {type(exc).__name__}: {exc}"
            )
            return {
                "heyting_verdict": _VERDICT_VETOED,
                "smith_torsion_coefficients": [],
                "choi_minimum_eigenvalue": -math.inf,
                "choi_trace_preserving_residual": math.inf,
                "bell_chsh_parameter": math.inf,
                "hardware_interlock_fired": actuation.interlock_fired,
                "actuation_latency_ns": actuation.actuation_latency_ns,
                "is_homology_coherent": False,
                "smith_diagonal": [],
                "smith_matrix_rank": 0,
                "invariant_factors": [],
                "snf_complete": False,
                "homology_nullity": 0,
                "homology_free_cokernel_rank": 0,
                "choi_is_cptp": False,
                "choi_is_cp": False,
                "choi_is_tp": False,
                "choi_is_hermitian": False,
                "choi_is_unital": False,
                "choi_trace_residual": math.inf,
                "choi_hermitian_residual": math.inf,
                "choi_unital_residual": math.inf,
                "choi_kraus_rank": 0,
                "choi_spectral_gap": 0.0,
                "bell_is_classical": False,
                "bell_classical_bound": _CLASSICAL_CHSH_BOUND,
                "bell_tsirelson_bound": _TSIRELSON_BOUND,
                "bell_algebraic_bound": _ALGEBRAIC_CHSH_BOUND,
                "bell_tsirelson_margin": -math.inf,
                "bell_correlations_physical": False,
                "heyting_truth_value": 0.0,
                "heyting_truth_continuous": 0.0,
                "gpio_pin": actuation.gpio_pin,
                "timestamp_utc": actuation.timestamp_utc,
                "reasons": [f"audit_exception: {type(exc).__name__}: {exc}"],
                "decision_sha256": "",
                "digital_signature_sha256": "",
                "phase": _AUDIT_PHASE,
                "version": __version__,
                "dimension_n": self._n,
            }

    def execute_audit_cycle_certified(
        self,
        boundary_matrix: np.ndarray,
        Choi_matrix: np.ndarray,
        bell_correlations: Tuple[float, float, float, float],
    ) -> AuditSatelliteCertificate:
        """Ciclo completo que devuelve el certificado congelado."""
        payload = self.execute_audit_cycle(
            boundary_matrix, Choi_matrix, bell_correlations
        )
        return AuditSatelliteCertificate(
            phase=str(payload.get("phase", _AUDIT_PHASE)),
            heyting_verdict=str(payload["heyting_verdict"]),
            smith_torsion_coefficients=tuple(payload["smith_torsion_coefficients"]),
            choi_minimum_eigenvalue=float(payload["choi_minimum_eigenvalue"]),
            choi_trace_preserving_residual=float(
                payload["choi_trace_preserving_residual"]
            ),
            bell_chsh_parameter=float(payload["bell_chsh_parameter"]),
            hardware_interlock_fired=bool(payload["hardware_interlock_fired"]),
            actuation_latency_ns=float(payload["actuation_latency_ns"]),
            digital_signature_sha256=str(payload.get("digital_signature_sha256", "")),
            is_homology_coherent=bool(payload.get("is_homology_coherent", False)),
            smith_diagonal=tuple(payload.get("smith_diagonal", ())),
            smith_matrix_rank=int(payload.get("smith_matrix_rank", 0)),
            choi_is_cptp=bool(payload.get("choi_is_cptp", False)),
            choi_is_cp=bool(payload.get("choi_is_cp", False)),
            choi_is_tp=bool(payload.get("choi_is_tp", False)),
            choi_is_hermitian=bool(payload.get("choi_is_hermitian", False)),
            choi_trace_residual=float(payload.get("choi_trace_residual", math.inf)),
            choi_hermitian_residual=float(
                payload.get("choi_hermitian_residual", math.inf)
            ),
            bell_is_classical=bool(payload.get("bell_is_classical", False)),
            bell_classical_bound=float(
                payload.get("bell_classical_bound", _CLASSICAL_CHSH_BOUND)
            ),
            bell_tsirelson_bound=float(
                payload.get("bell_tsirelson_bound", _TSIRELSON_BOUND)
            ),
            heyting_truth_value=float(payload.get("heyting_truth_value", 0.0)),
            gpio_pin=str(payload.get("gpio_pin", self._gpio_pin)),
            timestamp_utc=str(payload.get("timestamp_utc", self._now_utc_iso())),
            reasons=tuple(payload.get("reasons", ())),
            decision_sha256=str(payload.get("decision_sha256", "")),
            invariant_factors=tuple(payload.get("invariant_factors", ())),
            choi_kraus_rank=int(payload.get("choi_kraus_rank", 0)),
            bell_tsirelson_margin=float(payload.get("bell_tsirelson_margin", 0.0)),
        )

    async def execute_audit_cycle_async(
        self,
        boundary_matrix: np.ndarray,
        Choi_matrix: np.ndarray,
        bell_correlations: Tuple[float, float, float, float],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Versión asíncrona del ciclo de auditoría."""
        worker = asyncio.to_thread(
            self.execute_audit_cycle,
            boundary_matrix,
            Choi_matrix,
            bell_correlations,
        )
        timeout = self._async_timeout_s if timeout_s is None else float(timeout_s)
        if timeout > 0.0:
            return await asyncio.wait_for(worker, timeout=timeout)
        return await worker

    def _normalize_async_request(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Extrae y valida las claves admitidas de una petición asíncrona."""
        allowed = {
            "boundary_matrix",
            "Choi_matrix",
            "bell_correlations",
            "timeout_s",
        }
        unknown = set(request.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Claves no admitidas en request asíncrono: {sorted(unknown)}"
            )
        if "boundary_matrix" not in request or "Choi_matrix" not in request:
            raise ValueError("Cada request requiere 'boundary_matrix' y 'Choi_matrix'.")
        if "bell_correlations" not in request:
            raise ValueError("Cada request requiere 'bell_correlations'.")
        return {
            "boundary_matrix": request["boundary_matrix"],
            "Choi_matrix": request["Choi_matrix"],
            "bell_correlations": request["bell_correlations"],
            "timeout_s": request.get("timeout_s"),
        }

    async def execute_audit_cycle_batch_async(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        max_concurrency: int = 8,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """Ejecuta múltiples auditorías asíncronas con semáforo de concurrencia."""
        materialized = list(requests)
        if len(materialized) > self._max_batch:
            raise ValueError(
                f"El lote ({len(materialized)}) excede max_batch={self._max_batch}."
            )

        concurrency = max(int(max_concurrency), 1)
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(request: Mapping[str, Any]) -> Dict[str, Any]:
            params = self._normalize_async_request(request)
            async with semaphore:
                return await self.execute_audit_cycle_async(**params)

        tasks = [_run(request) for request in materialized]
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))


# ════════════════════════════════════════════════════════════════════════════
# CLASE PÚBLICA FINAL
# ════════════════════════════════════════════════════════════════════════════


class AuditSatellites(Phase3BellHeytingActuationKernel):
    r"""
    Aduana de de Rham–Fukaya–Boole y Auditor de Estructuras Simpliciales.

    Clase pública final que hereda las tres fases anidadas:
      - Fase 1: Homología / Smith Normal Form (`audit_boundary_homology`).
      - Fase 2: Choi–Jamiołkowski CPTP (`audit_choi_with_homology`,
        `prepare_causal_consolidation`).
      - Fase 3: Bell–CHSH / Heyting / Actuación (`audit_bell_with_causality`).

    Ejemplo
    -------
    >>> auditor = AuditSatellites(dimension_n=2)
    >>> result = auditor.execute_audit_cycle(
    ...     boundary_matrix=np.array([[1, 0], [0, 1]], dtype=int),
    ...     Choi_matrix=np.eye(4) / 2.0,
    ...     bell_correlations=(0.5, 0.5, 0.5, -0.5),
    ... )
    >>> result["heyting_verdict"]
    """


__all__ = [
    "ext_gcd",
    "SmithNormalFormResult",
    "HomologyReport",
    "ChoiReport",
    "BellReport",
    "CausalConsolidation",
    "HeytingDecisionReport",
    "AuditActuationReport",
    "AuditSatelliteCertificate",
    "Phase1HomologicalAuditKernel",
    "Phase2QuantumCausalAuditKernel",
    "Phase3BellHeytingActuationKernel",
    "AuditSatellites",
]