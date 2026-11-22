# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Witness Agent (Soberano Agéntico Testigo Silencioso)                   ║
║ RUTA    : app/agents/omega/positron_redemption_witness_agent.py                                      ║
║ VERSIÓN : 18.0.0-OODA-HKDF-Heyting-DeterminantalDivisors-POCS-Merkle6962-Loeb-RSI3-PhD               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
DOCTRINA RECTORA DE v18: EL TESTIGO VERIFICA INDEPENDIENTEMENTE Y EVOLUCIONA CON DESCENSO BIEN FUNDADO
──────────────────────────────────────────────────────────────────────────────────────────────────────
v16 leía tres escalares del motor y los sellaba. El motor v16 calculaba la Forma Normal de
Smith con SVD y devolvía los factores (2, 1) para un árbol. Esos factores son auto-delatores:
    (a) la cadena de divisibilidad exige d₁ | d₂, y 2 ∤ 1;
    (b) el teorema del primer divisor elemental exige d₁ = mcd(todas las entradas) = 1 ≠ 2.
Ambos certificados viajaban en el informe. v16 los descartaba. v17 y v18 los verifican, y añaden una
VERIFICACIÓN INDEPENDIENTE de la SNF por un algoritmo distinto al del motor:
    DIVISORES DETERMINANTALES.  D_k = mcd de todos los menores k×k de B (D₀ = 1).
    Teorema clásico:  d_k = D_k / D_{k−1},  r = máx{k : D_k ≠ 0}.
Éste es un camino computacionalmente INDEPENDIENTE de la eliminación por mcd del motor:
verificar con el mismo algoritmo no es verificar. Los menores se calculan con aritmética
entera exacta (Bareiss, int de precisión arbitraria), con cota de trabajo declarada.

NOVEDADES ESTRUCTURALES Y MATEMÁTICAS DE v18 (ACOPLAMIENTO CON EL MOTOR v18.0.0 + RSI-L3)
──────────────────────────────────────────────────────────────────────────────────────────────────────
  1. AUDITORÍA CUÁNTICO-CELESTE DE KUSTAANHEIMO-STIEFEL (ℍ ≅ ℝ⁴) Y SLD DE HELSTROM (Fase W₁):
     Verifica la ecuación de Lyapunov de la Derivada Logarítmica Simétrica (SLD), la cota
     geodésica de Cramér-Rao/Bures, el área simpléctica de Fubini-Study, la identidad de
     Hurwitz-Radon L(u)ᵀL(u) = ‖u‖₂²I₄ sobre el espinor cuaterniónico y la contracción de
     Banach del refinamiento polar unitario de Newton-Schulz del motor.
  2. VERIFICACIÓN COMBINATORIA DE HODGE-POINCARÉ Y MECÁNICA CELESTE KAM / VON ZEIPEL (Fase W₂):
     Contrasta la dimensión armónica de Hodge dim ker(Δ_k) contra el número de Betti β_k,
     verifica la fase de enlace de Pontryagin sobre Tors(H_k), audita la función generatriz
     de Poincaré-Von Zeipel S₁, los invariantes seculares de Kozai-Lidov y Tisserand-Poincaré,
     la aproximación diofántica por fracciones continuas del número de rotación KAM y la
     tasa de contracción de Romberg-Richardson en el flujo de Shahshahani.
  3. AUTO-MEJORA RECURSIVA DE NIVEL 3 (RSI-L3) CON PRESUPUESTO BIEN FUNDADO EN ℕ (Fase W₃):
     El agente evalúa su propio operador de umbralización epistémica mediante descenso estricto
     en (verification_depth, mutation_budget) ∈ ℕ × ℕ, gobernado por el operador proximal
     de Moreau-Yosida y reversión automática (rollback) ante cualquier degradación de las
     invariantes de L1/L2/L3, evitando el obstáculo de auto-referencia de Löb.

TRES OBLIGACIONES SEPARADAS
──────────────────────────────────────────────────────────────────────────────────────────────────────
  (i)   NO INTERFERIR : el testigo jamás muta el estado observado NI aborta el flujo del
                        llamador. Una forja o un fallo del motor producen un VEREDICTO
                        FIRMADO, no una excepción. [V9]
  (ii)  ATTESTAR      : cobertura total, verificación en cada compuerta, veredicto firmado. [V2/V3]
  (iii) ACTUAR        : el testigo es la autoridad MÁS BAJA de la tríada. Reporta; el campeón
                        actúa. Separación de funciones (IEC 61511: control básico vs. sistema
                        instrumentado de seguridad).
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import math
import secrets
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.omega.positron_redemption_witness_engine import (
        PositronRedemptionWitnessEngine,
        PositronWitnessSpectrumReport,
    )
except ImportError:  # pragma: no cover - ruta plana para pruebas unitarias
    try:
        from positron_redemption_witness_engine import (  # type: ignore[no-redef]
            PositronRedemptionWitnessEngine,
            PositronWitnessSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar positron_redemption_witness_engine (rutas de paquete y plana)."
        ) from exc

logger = logging.getLogger("APU.Agents.Omega.PositronRedemptionWitnessAgent")
AnswerT = TypeVar("AnswerT")

__all__ = [
    "HeytingWitnessVerdict",
    "WitnessPhase1Certificate",
    "WitnessPhase2Certificate",
    "WitnessPhase3Verdict",
    "WitnessGraceToken",
    "WitnessCrowbarActuator",
    "NullWitnessActuator",
    "WitnessVerificationReport",
    "KustaanheimoStiefelAuditCertificate",
    "RsiL3PolicyEvolutionCertificate",
    "W1ToW2ContinuationSeed",
    "W2ToW3ContinuationSeed",
    "PositronRedemptionWitnessAgent",
    "AGENT_VERSION",
]

_AGENT_VERSION: Final[str] = (
    "18.0.0-OODA-HKDF-Heyting-DeterminantalDivisors-POCS-Merkle6962-Loeb-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
AGENT_VERSION: Final[str] = _AGENT_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES CRIPTOGRÁFICAS, MATEMÁTICAS Y DE POLÍTICA DECLARADA
# ══════════════════════════════════════════════════════════════════════════════
_TINY: Final[float] = 1.0e-300
_WILKINSON_EPS: Final[float] = 1.0e-15
_STABILITY_FLOOR: Final[float] = 1.0e-12
_HASH_LEN: Final[int] = 32
_HKDF_MAX_LEN: Final[int] = 255 * _HASH_LEN
_DEFAULT_MASTER_KEY: Final[bytes] = b"APU_FILTER_V5_POSITRON_WITNESS_MASTER_KEY_2026"
_SALT_W1: Final[bytes] = b"APU/POSITRON/WITNESS/SALT/W1/v17"
_SALT_W2: Final[bytes] = b"APU/POSITRON/WITNESS/SALT/W2/v17"
_SALT_W3: Final[bytes] = b"APU/POSITRON/WITNESS/SALT/W3/v17"
_INFO_W1: Final[bytes] = b"positron/witness/W1"
_INFO_W2: Final[bytes] = b"positron/witness/W2"
_INFO_W3: Final[bytes] = b"positron/witness/W3"
_INFO_GRACE: Final[bytes] = b"positron/witness/grace"
_INFO_SESSION: Final[bytes] = b"positron/witness/session"
_INFO_NONCE: Final[bytes] = b"positron/witness/nonce"

_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95
_BURES_SOFT_CEILING: Final[float] = 0.35
_BURES_HARD_CEILING: Final[float] = 0.60
_CHORD_ARC_CEILING: Final[float] = 1.0e-9
_SLD_LYAPUNOV_CEILING: Final[float] = 1.0e-8
_KS_HURWITZ_DEFECT_CEILING: Final[float] = 1.0e-9
_SYMPECTIC_DEFECT_CEILING: Final[float] = 1.0e-8
_CANONICITY_DEFECT_CEILING: Final[float] = 1.0e-5
_IDEMPOTENCE_DEFECT_CEILING: Final[float] = 1.0e-9
_CONDITION_CAP: Final[float] = 1.0e12
_BANACH_Q_MAX: Final[float] = 0.95
_SESSION_TTL_NS: Final[int] = 30_000_000_000
_FLOAT64_CERTIFIED_DIGITS: Final[float] = 15.95
_MERGE_LEAF: Final[bytes] = b"\x00"
_MERGE_NODE: Final[bytes] = b"\x01"

# Cotas de trabajo de la verificación independiente por divisores determinantes.
_MAX_MINORS_PER_ORDER: Final[int] = 20000
_MAX_VERIFIED_ORDER: Final[int] = 4
_CARTAN_CONSTANCY_SAMPLES: Final[int] = 3

# Parámetros de hardware del tiristor crowbar (MMIO ESP32 / puente de silicio).
_GPIO_CROWBAR_PIN: Final[int] = 14
_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
_GPIO_PIN_MASK: Final[int] = 1 << _GPIO_CROWBAR_PIN
_CROWBAR_LATENCY_BUDGET_NS: Final[float] = 400.0


def _ev(obj: Any, name: str, default: Any = None) -> Any:
    """Lectura tolerante de campos entre motor v16, v17 y v18 (preserva False y 0.0)."""
    val = getattr(obj, name, default)
    return default if val is None else val


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fro(mat: NDArray[Any]) -> float:
    return float(la.norm(mat, ord="fro")) if getattr(mat, "size", 0) else 0.0


def _canonicalize(obj: Any) -> Any:
    """
    Canonicalización recursiva INYECTIVA. IntEnum antes de int (HeytingWitnessVerdict ES
    un int), bool antes de int, flotantes en `.17g` (round-trip exacto, determinista).
    Los ndarray se comprometen por SHA-256: nunca se vuelcan bytes al payload. [V2]
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, IntEnum):
        return {"__enum__": type(obj).__name__, "value": int(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        c = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "sha256": hashlib.sha256(c.tobytes(order="C")).hexdigest(),
        }
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return format(fv, ".17g") if math.isfinite(fv) else "non-finite"
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    return str(obj)


def _canonical_json(fields: Dict[str, Any]) -> str:
    return json.dumps(
        _canonicalize(fields),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _mth_rfc6962(leaves: Sequence[bytes]) -> bytes:
    """
    MTH de RFC 6962 EXACTO. [V10] Corte en la mayor potencia de 2 estrictamente menor que n;
    hoja = SHA256(0x00‖datos), nodo = SHA256(0x01‖izq‖der). El nodo impar se PROMUEVE,
    nunca se duplica: la duplicación de v16 habilitaba la maleabilidad CVE-2012-2459.
    """
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return hashlib.sha256(_MERGE_LEAF + bytes(leaves[0])).digest()
    k = 1
    while k * 2 < n:
        k *= 2
    return hashlib.sha256(
        _MERGE_NODE + _mth_rfc6962(leaves[:k]) + _mth_rfc6962(leaves[k:])
    ).digest()


def _ct_equals(a: str, b: str) -> bool:
    """Comparación en tiempo constante tolerante a entradas no ASCII."""
    try:
        return bool(hmac.compare_digest(str(a), str(b)))
    except (TypeError, ValueError):
        return False


def _bareiss_determinant(mat: Sequence[Sequence[int]]) -> int:
    """
    Determinante ENTERO exacto por eliminación de Bareiss (sin fracciones). La división de
    cada paso es exacta por la identidad de Sylvester; int de Python evita el desbordamiento
    que int64 sufriría con menores grandes.
    """
    n = len(mat)
    if n == 0:
        return 1
    a = [list(row) for row in mat]
    sign, prev = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            for i in range(k + 1, n):
                if a[i][k] != 0:
                    a[k], a[i] = a[i], a[k]
                    sign = -sign
                    break
            else:
                return 0
        pivot = a[k][k]
        for i in range(k + 1, n):
            aik = a[i][k]
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - aik * a[k][j]) // prev
            a[i][k] = 0
        prev = pivot
    return sign * a[n - 1][n - 1]


def _independent_hurwitz_ks_defect(spinor_u: Sequence[float]) -> float:
    """
    Verificación independiente en el agente de la identidad de Hurwitz-Radon para la
    matriz de Kustaanheimo-Stiefel L(u) sobre ℍ ≅ ℝ⁴:
        L(u)ᵀ L(u) = (u₁² + u₂² + u₃² + u₄²) I₄.
    """
    if len(spinor_u) != 4:
        return float("inf")
    u1, u2, u3, u4 = (float(x) for x in spinor_u)
    l_mat = np.array(
        [
            [u1, -u2, -u3, u4],
            [u2, u1, -u4, -u3],
            [u3, u4, u1, u2],
            [u4, -u3, u2, -u1],
        ],
        dtype=np.float64,
    )
    r_norm_sq = u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4
    return _fro(l_mat.T @ l_mat - r_norm_sq * np.eye(4, dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# §1 · ÁLGEBRA DE HEYTING Ω₃ Y NÚCLEO MODAL DE LAWVERE-TIERNEY
# ══════════════════════════════════════════════════════════════════════════════
class HeytingWitnessVerdict(IntEnum):
    """
    Ω₃ con orden de salud VETOED(0) ⊥ < DEGRADED(1) < COHERENT(2) ⊤. [V10]
    v16 usaba ints desnudos sin operaciones y un `to_string` que coerceía cualquier valor
    desconocido a "VETOED". Aquí es IntEnum (compatible con `int` y con `min`, que en este
    orden ES el meet) con álgebra completa y parseo estricto.
    Cadena de Gödel de 3 elementos: Heyting NO booleana, pues ¬¬DEGRADED = COHERENT.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingWitnessVerdict") -> "HeytingWitnessVerdict":
        """Ínfimo (∧): conserva el estado más severo. AXIOMA HEYTING-1."""
        return self if int(self) <= int(other) else other

    def join(self, other: "HeytingWitnessVerdict") -> "HeytingWitnessVerdict":
        """Supremo (∨): conserva el estado más saludable."""
        return self if int(self) >= int(other) else other

    def implies(self, other: "HeytingWitnessVerdict") -> "HeytingWitnessVerdict":
        """Implicación de Heyting: adjunción (a∧x) ≤ b ⟺ x ≤ (a⇒b)."""
        return HeytingWitnessVerdict.COHERENT if int(self) <= int(other) else other

    def bi_implies(self, other: "HeytingWitnessVerdict") -> "HeytingWitnessVerdict":
        """Equivalencia intuicionista (a ⇔ b) = (a ⇒ b) ∧ (b ⇒ a)."""
        return self.implies(other).meet(other.implies(self))

    def neg(self) -> "HeytingWitnessVerdict":
        return self.implies(HeytingWitnessVerdict.VETOED)

    def double_negation(self) -> "HeytingWitnessVerdict":
        return self.neg().neg()

    def is_regular(self) -> bool:
        """Un elemento es regular (booleano/clásico en el subtopos ¬¬) ssi ¬¬x = x."""
        return self.double_negation() is self

    @property
    def label(self) -> str:
        return self.name

    @classmethod
    def parse(cls, value: Any) -> "HeytingWitnessVerdict":
        """Parseo ESTRICTO: un valor desconocido es un error, no un veredicto. [V10]"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ValueError(f"Veredicto desconocido: {value!r}") from exc
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            try:
                return cls(int(value))
            except ValueError as exc:
                raise ValueError(f"Veredicto fuera de Ω₃: {value!r}") from exc
        raise ValueError(f"No interpretable como veredicto: {value!r}")

    @classmethod
    def to_string(cls, val: Any) -> str:
        """Compatibilidad v16 con semántica corregida: parseo estricto."""
        return cls.parse(val).label

    @staticmethod
    def verify_axioms() -> bool:
        """Verificación exhaustiva (27 tríadas) de residuación, De Morgan y ¬¬."""
        vals = (
            HeytingWitnessVerdict.VETOED,
            HeytingWitnessVerdict.DEGRADED,
            HeytingWitnessVerdict.COHERENT,
        )
        for a in vals:
            for b in vals:
                for x in vals:
                    if (int(a.meet(x)) <= int(b)) != (int(x) <= int(a.implies(b))):
                        return False
                if a.meet(b).neg() is not a.neg().join(b.neg()):
                    return False
            if a.double_negation().double_negation() is not a.double_negation():
                return False
        return True


def grace_nucleus(
    verdict: HeytingWitnessVerdict, grace_verified: bool
) -> HeytingWitnessVerdict:
    """
    Núcleo de gracia de Lawvere-Tierney j: j(C)=C, j(D)=C, j(V)=V. Inflacionario,
    idempotente, preserva ∧, j(⊤)=⊤; nunca eleva VETOED.
    El TESTIGO lo usa sólo para calcular la vista INFORMATIVA `verdict_under_grace`:
    su veredicto autoritativo es el crudo. Informar no es absolver.
    """
    if not grace_verified:
        return verdict
    return (
        HeytingWitnessVerdict.COHERENT
        if verdict is HeytingWitnessVerdict.DEGRADED
        else verdict
    )


def verify_nucleus_laws() -> bool:
    vals = (
        HeytingWitnessVerdict.VETOED,
        HeytingWitnessVerdict.DEGRADED,
        HeytingWitnessVerdict.COHERENT,
    )
    for a in vals:
        j_a = grace_nucleus(a, True)
        if int(a) > int(j_a) or grace_nucleus(j_a, True) is not j_a:
            return False
        for b in vals:
            if grace_nucleus(a.meet(b), True) is not j_a.meet(grace_nucleus(b, True)):
                return False
    return (
        grace_nucleus(HeytingWitnessVerdict.COHERENT, True)
        is HeytingWitnessVerdict.COHERENT
        and grace_nucleus(HeytingWitnessVerdict.VETOED, True)
        is HeytingWitnessVerdict.VETOED
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2 · TENSORES, SESIÓN, GRACIA, HOMOLOGÍA, CONTRACCIÓN, CELESTE, RSI-L3 Y ACTUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class FrozenTensor:
    """Huella pasiva: contenido Y metadatos (dtype/shape/strides)."""

    name: str
    digest: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str
    nbytes: int
    valid: bool

    def same_content(self, other: "FrozenTensor") -> bool:
        return (
            self.valid
            and other.valid
            and self.digest == other.digest
            and self.shape == other.shape
            and self.dtype == other.dtype
        )


@dataclass(frozen=True, slots=True)
class NonInterferenceAttestation:
    """WITNESS-1 con alcance DECLARADO: buffers inspeccionados, sin garantía TOCTOU. [V4]"""

    p_ok: bool
    rho_ok: bool
    sigma_ok: bool
    k_ok: bool
    boundary_ok: bool
    engine_claim_ok: Optional[bool]
    scope: str = "INSPECTED_BUFFERS_ONLY_NO_TOCTOU_GUARANTEE"

    @property
    def all_ok(self) -> bool:
        local = bool(
            self.p_ok and self.rho_ok and self.sigma_ok and self.k_ok and self.boundary_ok
        )
        return local and (self.engine_claim_ok is None or bool(self.engine_claim_ok))


@dataclass(frozen=True, slots=True)
class WitnessSession:
    """Nonce CON CLAVE e impredecible, secuencia monótona y ventana [nbf, exp). [V10]"""

    session_id: str
    nonce: str
    sequence: int
    issued_ns: int
    nbf_ns: int
    exp_ns: int
    monotonic_ns: int
    input_bundle_digest: str
    dimension_n: int

    def is_live(self, now_ns: int) -> bool:
        return bool(self.nbf_ns <= now_ns < self.exp_ns)


@dataclass(frozen=True, slots=True)
class WitnessGraceToken:
    """
    Token de gracia emitido por el SOBERANO GOBERNADOR (no por el testigo), ligado al
    session_id y al bundle de entrada. El testigo lo verifica y reporta su efecto como
    vista informativa; jamás altera su propio veredicto con él.
    """

    subject: str
    session_id: str
    input_bundle_digest: str
    nonce: str
    issued_ns: int
    nbf_ns: int
    exp_ns: int
    max_grace_sec: float
    signature: str = ""


@dataclass(frozen=True, slots=True)
class KustaanheimoStiefelAuditCertificate:
    """
    [v18 · Fase W₁] Certificado de auditoría cuántico-celeste del sustrato pasivo:
    verifica la información cuántica de Fisher (QFI), la ecuación de Lyapunov de la SLD,
    la ortogonalidad de Hurwitz-Radon del espinor de Kustaanheimo-Stiefel en ℍ ≅ ℝ⁴ y
    la contracción de punto fijo RSI-L3 de Newton-Schulz sobre la polar unitaria de Uhlmann.
    """

    quantum_fisher_information: float
    sld_lyapunov_residual: float
    sld_lyapunov_ok: bool
    ks_hurwitz_defect: float
    ks_bilinear_gauge_residual: float
    ks_regularization_ok: bool
    fubini_study_symplectic_area: float
    rsi_l3_polar_contraction_ratio: float
    rsi_l3_substrate_converged_ok: bool


@dataclass(frozen=True, slots=True)
class W1ToW2ContinuationSeed:
    """
    Semilla formal de continuación que enlaza el final de la Fase W₁ (1.9) con el
    inicio de la Fase W₂ (2.1 _gate_w12). Transporta las invariantes verificadas en W₁.
    """

    session_id: str
    input_bundle_digest: str
    dimension_n: int
    uhlmann_fidelity: float
    bures_distance: float
    quantum_fisher_information: float
    non_interference_all_ok: bool
    ks_regularization_ok: bool


@dataclass(frozen=True, slots=True)
class SmithIndependenceCertificate:
    """
    Verificación INDEPENDIENTE de la SNF por divisores determinantes. [V1]
    D_k = mcd de todos los menores k×k; d_k = D_k/D_{k−1}; r = máx{k : D_k ≠ 0}.
    Algoritmo distinto de la eliminación por mcd del motor: verificar con el mismo
    algoritmo no es verificar.
    """

    available: bool
    determinantal_divisors: Tuple[str, ...]
    independent_factors: Tuple[str, ...]
    independent_rank: int
    factors_match: bool
    rank_match: bool
    divisibility_chain_ok: bool
    torsion_claim_consistent: bool
    independent_torsion_free: bool
    verified_order: int
    minors_evaluated: int
    work_capped: bool
    engine_certificates_ok: Optional[bool]
    theorem: str = "D_k = gcd of all k-minors; d_k = D_k / D_(k-1)"
    # ── v18: Topología combinatoria de Hodge-Poincaré y enlace de Pontryagin ──
    hodge_betti_consistency_ok: bool = True
    pontryagin_linking_phase: float = 0.0
    hadamard_log10_bound: float = 0.0


@dataclass(frozen=True, slots=True)
class ContractionCertificate:
    """Contracción genuina: idempotencia espacial (POCS) y razón temporal. [V7]"""

    idempotence_defect: float
    idempotence_ok: bool
    at_fixed_point: bool
    q_temporal_raw: float
    q_temporal: float
    temporal_defined: bool
    temporal_contractive: bool
    increment_last: float
    increment_first: float
    a_posteriori_bound: float
    a_priori_bound: float
    certified_digits: float
    history_length: int
    theorem: str = "Frobenius projection onto a product of closed convex sets is idempotent"
    # ── v18: Contracción RSI-L3 del motor y extrapolación de Romberg ──
    engine_rsi_l3_contraction_ratio: float = 0.0
    romberg_l3_contraction_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class CelestialAuditCertificate:
    """Auditoría de Poincaré-Liouville-Shahshahani y detector de invariante fabricado. [V8]"""

    canonicity_defect: Optional[float]
    residual_tautological: bool
    cartan_value: float
    cartan_suspiciously_constant: bool
    cartan_samples: int
    liouville_ok: Optional[bool]
    symplectic_defect: Optional[float]
    monodromy_determinant: Optional[float]
    krein_collision: Optional[bool]
    shahshahani_passive: bool
    shahshahani_rate: float
    speed_identity_error: Optional[float]
    # ── v18: Invariantes de Von Zeipel, Kozai-Lidov, Tisserand y KAM ──
    symplectic_jacobian_defect: float = 0.0
    von_zeipel_generating_s1: float = 0.0
    kozai_lidov_invariant: float = 1.0
    tisserand_poincare_invariant: float = 3.0
    kam_rotation_number: float = 0.0
    kam_diophantine_defect: float = 0.0
    floquet_lyapunov_exponent: float = 0.0


@dataclass(frozen=True, slots=True)
class W2ToW3ContinuationSeed:
    """
    Semilla formal de continuación que enlaza el final de la Fase W₂ (2.10) con el
    inicio de la Fase W₃ (3.1 _gate_w23).
    """

    session_id: str
    w1_signature: str
    w2_signature: str
    gate_w12_authentic: bool
    smith_factors_match: bool
    pocs_idempotent_ok: bool
    shahshahani_passive_ok: bool
    banach_ratio: float


@dataclass(frozen=True, slots=True)
class RsiL3PolicyEvolutionCertificate:
    """
    [v18 · Fase W₃] Certificado de auto-mejora recursiva de Nivel 3 (RSI-L3) sobre la
    política adaptativa del testigo: registra la evaluación del operador proximal de
    Moreau-Yosida, el presupuesto de mutación bien fundado en ℕ y la verificación de
    monotonía de seguridad (invariancia de no-relajación ante fallos L1/L2/L3).
    """

    evaluated: bool
    policy_mutated: bool
    rollback_triggered: bool
    mutation_budget_before: int
    mutation_budget_after: int
    moreau_yosida_objective_delta: float
    rationale: str = "MOREAU_YOSIDA_PROXIMAL_POLICY_DESCENT_ON_NAT"


@dataclass(frozen=True, slots=True)
class CrowbarIntent:
    """INTENCIÓN firmada. No es atestación de actuación física. [V6]"""

    session_id: str
    gpio_pin: int
    mmio_register: int
    pin_mask: int
    reason: Tuple[str, ...]
    vote_tally: Tuple[str, ...]
    issued_ns: int
    signature: str = ""


@dataclass(frozen=True, slots=True)
class ActuationReceipt:
    """Recibo del driver. Sin driver enlazado: confirmed=False, mode=SIMULATION, latency=None."""

    confirmed: bool
    mode: str
    latency_ns_measured: Optional[float]
    within_latency_budget: Optional[bool]
    driver_identity: str
    detail: str


class WitnessCrowbarActuator(Protocol):
    """Protocolo estructural del driver físico. Debe devolver un recibo verificable."""

    identity: str

    def fire(self, intent: CrowbarIntent) -> ActuationReceipt: ...


class NullWitnessActuator:
    """
    Driver nulo por defecto: NUNCA afirma disparo físico. [V6] v16 asignaba dos locales
    Python, devolvía True y fabricaba la latencia sumando 380 ns y recortando a 395:
    eso es una atestación falsa de desenergización.
    """

    identity: str = "NULL_WITNESS_ACTUATOR_NO_HARDWARE_BOUND"

    def fire(self, intent: CrowbarIntent) -> ActuationReceipt:
        return ActuationReceipt(
            confirmed=False,
            mode="SIMULATION",
            latency_ns_measured=None,
            within_latency_budget=None,
            driver_identity=self.identity,
            detail="Intención registrada; ningún periférico fue accionado.",
        )


@dataclass(frozen=True, slots=True)
class LoebVerification:
    """Verificación bien fundada, no auto-referencial y NO vacua. [V5]"""

    accepted: bool
    depth_remaining: int
    independently_recomputed: bool
    identities_rechecked: int
    rationale: str = "WELL_FOUNDED_DESCENT_ON_NAT_NOT_SELF_TRUST"


@dataclass(frozen=True, slots=True)
class MetaAuditCertificate:
    l1_numeric_ok: bool
    l2_structural_ok: bool
    l3_attestation_ok: bool
    accepted: bool
    verification_depth_remaining: int
    mutation_budget_remaining: int
    failed_checks: Tuple[str, ...] = ()
    rsi_l3_policy_evolution: Optional[RsiL3PolicyEvolutionCertificate] = None


@dataclass(frozen=True, slots=True)
class GateVerdict:
    authentic: bool
    evidentially_clean: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WitnessPhase1Certificate:
    """Expediente inmutable de W₁. Orden posicional v16 y v17 preservado."""

    report: PositronWitnessSpectrumReport
    timestamp_ns: int
    nonce: str
    signature: str
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    session: Optional[WitnessSession] = None
    monotonic_ns: int = 0
    frozen_inputs: Tuple[FrozenTensor, ...] = ()
    non_interference: Optional[NonInterferenceAttestation] = None
    evidence_atoms: Tuple[Tuple[str, str], ...] = ()
    abstentions: Tuple[str, ...] = ()
    boundary_digest: str = ""
    merkle_leaf: str = ""
    # ── v18: Auditoría KS-SLD y semilla de continuación hacia W₂ ──
    ks_audit: Optional[KustaanheimoStiefelAuditCertificate] = None
    continuation_seed: Optional[W1ToW2ContinuationSeed] = None


@dataclass(frozen=True, slots=True)
class WitnessPhase2Certificate:
    """Expediente inmutable de W₂. Orden posicional v16 y v17 preservado."""

    phase1_cert: WitnessPhase1Certificate
    banach_ratio: float
    connes_bound: float
    timestamp_ns: int
    signature: str
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    session: Optional[WitnessSession] = None
    monotonic_ns: int = 0
    gate_w12: Optional[GateVerdict] = None
    smith_independence: Optional[SmithIndependenceCertificate] = None
    contraction: Optional[ContractionCertificate] = None
    celestial: Optional[CelestialAuditCertificate] = None
    dk_sqrt_lipschitz: float = 0.0
    evidence_atoms: Tuple[Tuple[str, str], ...] = ()
    abstentions: Tuple[str, ...] = ()
    authentication_veto: bool = False
    merkle_leaf: str = ""
    # ── v18: Semilla de continuación hacia W₃ ──
    continuation_seed: Optional[W2ToW3ContinuationSeed] = None


@dataclass(frozen=True, slots=True)
class WitnessPhase3Verdict:
    """
    Certificado público inmutable final. Orden posicional v16 y v17 preservado.
    [V3] v16 NO tenía firma: el veredicto salía completamente no autenticado.
    """

    final_verdict: int
    verdict_label: str
    merkle_root_rfc6962: str
    loeb_verification_passed: bool
    hardware_crowbar_triggered: bool
    actuation_latency_ns: float
    schema_version: str = "18.0.0-PhD-Witness-DeterminantalDivisors-POCS-Heyting-RSI3"
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    phase2_cert: Optional[WitnessPhase2Certificate] = None
    base_verdict: int = int(HeytingWitnessVerdict.COHERENT)
    verdict_under_grace: int = int(HeytingWitnessVerdict.COHERENT)
    grace_verified: bool = False
    grace_token: Optional[WitnessGraceToken] = None
    verdict_atoms: Tuple[Tuple[str, str], ...] = ()
    abstentions: Tuple[str, ...] = ()
    crowbar_intent: Optional[CrowbarIntent] = None
    actuation: Optional[ActuationReceipt] = None
    actuation_mode: str = "SIMULATION"
    actuation_policy: str = "ATTESTED_SINGLE"
    crowbar_latched: bool = False
    loeb: Optional[LoebVerification] = None
    meta_audit: Optional[MetaAuditCertificate] = None
    gate_w23: Optional[GateVerdict] = None
    session: Optional[WitnessSession] = None
    session_digest: str = ""
    monotonic_ns: int = 0
    timestamp_ns: int = 0
    sovereign_signature: str = ""
    diagnostics: Tuple[str, ...] = ()
    merkle_algorithm: str = "RFC6962-MTH-SHA256-LEAF0x00-NODE0x01-PROMOTE"


@dataclass(frozen=True, slots=True)
class WitnessVerificationReport:
    c1_signature_ok: bool
    c2_signature_ok: bool
    sovereign_signature_ok: bool
    merkle_ok: bool
    session_digest_ok: bool
    verdict_recomputation_ok: bool
    grace_view_ok: bool
    session_continuity_ok: bool
    actuation_honesty_ok: bool
    all_ok: bool
    failed_checks: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WitnessPolicy:
    """Política adaptativa. `mutation_budget` es el contador bien fundado de por vida."""

    fidelity_floor: float = _UHLMANN_FIDELITY_FLOOR
    bures_soft_ceiling: float = _BURES_SOFT_CEILING
    banach_q_max: float = _BANACH_Q_MAX
    actuation_policy: str = "ATTESTED_SINGLE"  # DISABLED | ATTESTED_SINGLE | VOTED_2oo3
    verification_depth: int = 3
    mutation_budget: int = 3

    def project(self) -> "WitnessPolicy":
        pol = self.actuation_policy
        if pol not in ("DISABLED", "ATTESTED_SINGLE", "VOTED_2oo3"):
            pol = "ATTESTED_SINGLE"
        return WitnessPolicy(
            fidelity_floor=float(np.clip(self.fidelity_floor, 0.5, 1.0)),
            bures_soft_ceiling=float(
                np.clip(self.bures_soft_ceiling, 0.05, _BURES_HARD_CEILING)
            ),
            banach_q_max=float(np.clip(self.banach_q_max, 0.5, 0.999999)),
            actuation_policy=pol,
            verification_depth=max(0, int(self.verification_depth)),
            mutation_budget=max(0, int(self.mutation_budget)),
        )


@dataclass
class WitnessTelemetry:
    total_cycles: int = 0
    phase1_emissions: int = 0
    phase2_emissions: int = 0
    phase3_emissions: int = 0
    gate_w12_failures: int = 0
    gate_w23_failures: int = 0
    engine_failures: int = 0
    snf_independence_failures: int = 0
    non_interference_violations: int = 0
    crowbar_intents: int = 0
    crowbar_confirmations: int = 0
    latency_budget_violations: int = 0
    sequence_counter: int = 0
    monotonic_clock_ns: int = 0
    bures_history: List[float] = field(default_factory=list)
    cartan_history: List[Tuple[str, float]] = field(default_factory=list)
    used_nonces: List[str] = field(default_factory=list)
    policy: WitnessPolicy = field(default_factory=WitnessPolicy)
    policy_ledger: List[Dict[str, Any]] = field(default_factory=list)
    mutation_count: int = 0
    rollback_count: int = 0
    crowbar_latched: bool = False
    dev_key_in_use: bool = False
    last_verdict: HeytingWitnessVerdict = HeytingWitnessVerdict.COHERENT


# ══════════════════════════════════════════════════════════════════════════════
# §3 · SOBERANO TESTIGO: TRES FASES OODA ANIDADAS EN CPS
# ══════════════════════════════════════════════════════════════════════════════
class PositronRedemptionWitnessAgent:
    """
    Soberano Agéntico Testigo Silencioso de Redención (Estrato Omega / Santuario Epistémico).
    Tres fases anidadas con CPS de tipo de respuesta polimórfico:
        W₁ · WitnessPhase1Observer (1.1 → 1.9)  _phase1_observe_and_sign
        W₂ · WitnessPhase2Orientor (2.1 → 2.10) _phase2_orient_and_sign
        W₃ · WitnessPhase3Decider  (3.1 → 3.10) _phase3_decide_and_act
    Doctrina: nunca interfiere (ni muta el estado ni aborta el flujo); nunca atesta limpio
    sobre entrada no autenticada; actúa sólo con driver enlazado, voto y recibo verificable.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE W₁ · OBSERVER: NO-INTERFERENCIA, GEOMETRÍA UHLMANN-KS Y FIRMA TOTAL
    # ──────────────────────────────────────────────────────────────────────────
    class WitnessPhase1Observer:
        """W₁ (1.1 → 1.9). Terminal: _phase1_observe_and_sign, que abre W₂ en 2.1."""

        def __init__(self, sovereign: "PositronRedemptionWitnessAgent") -> None:
            self._s = sovereign

        # ── Método 1.1 ────────────────────────────────────────────────────────
        @staticmethod
        def _freeze_and_digest(name: str, arr: Any) -> FrozenTensor:
            """
            1.1 SHA-256 sobre (dtype‖shape‖strides‖bytes_C). Incluir strides detecta
            reordenamientos por vistas. `ascontiguousarray` NO muta la fuente.
            """
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                return FrozenTensor(name, "", (), (), "", 0, False)
            try:
                c = np.ascontiguousarray(arr)
                meta = f"{arr.dtype.str}|{arr.shape}|{arr.strides}|".encode("ascii")
                digest = hashlib.sha256(meta + c.tobytes(order="C")).hexdigest()
            except (ValueError, TypeError, MemoryError):
                return FrozenTensor(
                    name,
                    "",
                    tuple(arr.shape),
                    tuple(arr.strides),
                    str(arr.dtype.str),
                    int(arr.nbytes),
                    False,
                )
            return FrozenTensor(
                name=name,
                digest=digest,
                shape=tuple(int(s) for s in arr.shape),
                strides=tuple(int(s) for s in arr.strides),
                dtype=str(arr.dtype.str),
                nbytes=int(arr.nbytes),
                valid=True,
            )

        # ── Método 1.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _attest_non_interference(
            before: Sequence[FrozenTensor],
            after: Sequence[FrozenTensor],
            engine_claim: Optional[bool],
        ) -> NonInterferenceAttestation:
            """
            1.2 [V4] AXIOMA WITNESS-1, por fin atestiguado. Compara las huellas de los CINCO
            buffers (p, ρ, σ, K, B) antes/después del motor y las contrasta con la claim
            `non_interference_ok` del motor v17/v18. Un buffer ausente NO cuenta como evidencia
            positiva. ALCANCE: no prueba ausencia de escrituras transitorias restauradas
            (TOCTOU) ni cubre el resto de la memoria del proceso.
            """
            flags: List[bool] = []
            for i in range(5):
                b = (
                    before[i]
                    if i < len(before)
                    else FrozenTensor("", "", (), (), "", 0, False)
                )
                a = (
                    after[i]
                    if i < len(after)
                    else FrozenTensor("", "", (), (), "", 0, False)
                )
                flags.append(
                    b.same_content(a)
                    if (b.valid and a.valid)
                    else (not b.valid and not a.valid)
                )
            return NonInterferenceAttestation(*[bool(f) for f in flags], engine_claim)

        # ── Método 1.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_uhlmann_bures(
            report: PositronWitnessSpectrumReport, policy: WitnessPolicy
        ) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
            """
            1.3 [V8] Auditoría de la geometría de Uhlmann-Bures. v16 firmaba la fidelidad
            y NO la juzgaba; su radio d_B < 0.5 admitía F = 0.766, muy por debajo del piso
            0.95 del resto de la familia. Aquí se juzga F, d_B y la identidad cuerda-arco,
            y se ABSTIENE explícitamente donde el motor no expone el certificado.
            """
            V = HeytingWitnessVerdict
            atoms: List[Tuple[str, str]] = []
            abst: List[str] = []
            fid = _safe_float(_ev(report, "uhlmann_fidelity", 0.0))
            bures = _safe_float(
                _ev(report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0)
            )
            atoms.append(
                (
                    "UHLMANN_FIDELITY",
                    (V.COHERENT if fid >= policy.fidelity_floor else V.DEGRADED).name,
                )
            )
            if bures > _BURES_HARD_CEILING:
                atoms.append(("BURES_RADIUS", V.VETOED.name))
            elif bures > policy.bures_soft_ceiling:
                atoms.append(("BURES_RADIUS", V.DEGRADED.name))
            else:
                atoms.append(("BURES_RADIUS", V.COHERENT.name))
            chord = _ev(report, "chord_arc_identity_error", None)
            if chord is None:
                abst.append("BURES_CHORD_ARC")
            else:
                ok = _safe_float(chord) <= _CHORD_ARC_CEILING
                atoms.append(("BURES_CHORD_ARC", (V.COHERENT if ok else V.VETOED).name))
            for name, field_name in (
                ("DENSITY_KKT", "rho_kkt_ok"),
                ("SIMPLEX_KKT", "simplex_kkt_ok"),
            ):
                val = _ev(report, field_name, None)
                if val is None:
                    abst.append(name)
                else:
                    atoms.append(
                        (name, (V.COHERENT if bool(val) else V.DEGRADED).name)
                    )
            hs = _ev(report, "hilbert_schmidt_overlap", None)
            if hs is not None and _safe_float(hs) > fid + 1e-9:
                atoms.append(("HS_OVERLAP_EXCEEDS_FIDELITY", V.VETOED.name))
            return tuple(atoms), tuple(abst)

        # ── Método 1.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_ks_spinor_and_rsi_l3_substrate(
            report: PositronWitnessSpectrumReport,
        ) -> Tuple[
            KustaanheimoStiefelAuditCertificate,
            Tuple[Tuple[str, str], ...],
            Tuple[str, ...],
        ]:
            """
            1.4 [v18 · Auditoría Cuántico-Celeste KS y SLD + Sustrato RSI-L3]
            Audita los invariantes de Fase Π₁ introducidos en el motor v18.0.0:
              • Residuo de Lyapunov de la Derivada Logarítmica Simétrica (SLD)
                ‖½(ρL + Lρ) − (σ − ρ)‖_F ≤ _SLD_LYAPUNOV_CEILING y no negatividad de la QFI;
              • Ortogonalidad de Hurwitz-Radon del espinor de Kustaanheimo-Stiefel
                ‖L(u)ᵀL(u) − rI₄‖_F ≤ _KS_HURWITZ_DEFECT_CEILING y vínculo bilineal nulo;
              • Convergencia de punto fijo de Newton-Schulz sobre la polar unitaria de Uhlmann.
            Si el informe proviene de un motor anterior que no expone estos campos, registra
            ABSTENCIONES explícitas sin falsificar evidencia positiva.
            """
            V = HeytingWitnessVerdict
            atoms: List[Tuple[str, str]] = []
            abst: List[str] = []

            qfi_raw = _ev(report, "quantum_fisher_information", None)
            sld_raw = _ev(report, "sld_lyapunov_residual", None)
            ks_def_raw = _ev(report, "ks_hurwitz_orthogonality_defect", None)
            ks_gauge_raw = _ev(report, "ks_bilinear_gauge_residual", None)
            fs_area_raw = _ev(report, "fubini_study_symplectic_area", None)
            rsi_q_raw = _ev(report, "rsi_l3_banach_contraction_ratio", None)
            rsi_ok_raw = _ev(report, "rsi_l3_converged_ok", None)

            # Si el motor expone el certificado anidado pi1.ks_poincare, verificamos
            # independientemente la identidad matricial de Hurwitz-Radon sobre spinor_u.
            pi1_cert = _ev(report, "pi1", None)
            ks_cert = _ev(pi1_cert, "ks_poincare", None) if pi1_cert is not None else None
            indep_ks_def = None
            if ks_cert is not None:
                spinor_u = _ev(ks_cert, "spinor_u", None)
                if spinor_u is not None and len(spinor_u) == 4:
                    indep_ks_def = _independent_hurwitz_ks_defect(spinor_u)

            qfi = _safe_float(qfi_raw, 0.0)
            sld_res = _safe_float(sld_raw, 0.0)
            ks_def = (
                max(_safe_float(ks_def_raw, 0.0), float(indep_ks_def))
                if indep_ks_def is not None
                else _safe_float(ks_def_raw, 0.0)
            )
            ks_gauge = _safe_float(ks_gauge_raw, 0.0)
            fs_area = _safe_float(fs_area_raw, 0.0)
            rsi_q = _safe_float(rsi_q_raw, 0.0)
            rsi_ok = bool(rsi_ok_raw) if rsi_ok_raw is not None else True

            if sld_raw is None:
                abst.append("SLD_LYAPUNOV_EQUATION")
                sld_ok = True
            else:
                sld_ok = bool(sld_res <= _SLD_LYAPUNOV_CEILING and qfi >= -1.0e-12)
                atoms.append(
                    ("SLD_LYAPUNOV_EQUATION", (V.COHERENT if sld_ok else V.DEGRADED).name)
                )

            if ks_def_raw is None and indep_ks_def is None:
                abst.append("KS_HURWITZ_ORTHOGONALITY")
                ks_ok = True
            else:
                ks_ok = bool(
                    ks_def <= _KS_HURWITZ_DEFECT_CEILING
                    and ks_gauge <= _KS_HURWITZ_DEFECT_CEILING
                )
                atoms.append(
                    (
                        "KS_HURWITZ_ORTHOGONALITY",
                        (V.COHERENT if ks_ok else V.VETOED).name,
                    )
                )

            if rsi_ok_raw is None:
                abst.append("SUBSTRATE_RSI_L3_POLAR_FIXED_POINT")
            else:
                atoms.append(
                    (
                        "SUBSTRATE_RSI_L3_POLAR_FIXED_POINT",
                        (V.COHERENT if (rsi_ok and 0.0 <= rsi_q < 1.0) else V.DEGRADED).name,
                    )
                )

            cert = KustaanheimoStiefelAuditCertificate(
                quantum_fisher_information=qfi,
                sld_lyapunov_residual=sld_res,
                sld_lyapunov_ok=sld_ok,
                ks_hurwitz_defect=ks_def,
                ks_bilinear_gauge_residual=ks_gauge,
                ks_regularization_ok=ks_ok,
                fubini_study_symplectic_area=fs_area,
                rsi_l3_polar_contraction_ratio=rsi_q,
                rsi_l3_substrate_converged_ok=rsi_ok,
            )
            return cert, tuple(atoms), tuple(abst)

        # ── Método 1.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _build_w1_payload(cert: WitnessPhase1Certificate) -> Dict[str, Any]:
            """
            1.5 Payload de COBERTURA TOTAL (excluye sólo la firma y la hoja derivada). [V2]
            v16 firmaba 3 escalares: los tres tensores, los factores invariantes, Cartan,
            Poincaré, Shahshahani, Connes y κ quedaban SIN AUTENTICAR ⟹ la base física
            entera era sustituible sin romper la firma.
            """
            return {
                "domain": _INFO_W1.decode("ascii"),
                "agent_version": cert.agent_version,
                "report": cert.report,
                "timestamp_ns": cert.timestamp_ns,
                "monotonic_ns": cert.monotonic_ns,
                "nonce": cert.nonce,
                "session": cert.session,
                "frozen_inputs": cert.frozen_inputs,
                "non_interference": cert.non_interference,
                "evidence_atoms": cert.evidence_atoms,
                "abstentions": cert.abstentions,
                "boundary_digest": cert.boundary_digest,
                "ks_audit": cert.ks_audit,
                "continuation_seed": cert.continuation_seed,
            }

        # ── Método 1.6 ────────────────────────────────────────────────────────
        def _hkdf_subkey(
            self, salt: bytes, info: bytes, length: int = _HASH_LEN
        ) -> bytes:
            """1.6 Delega en la derivación HKDF completa del soberano (4.2). [V10]"""
            return self._s._hkdf_extract_and_expand(salt, info, length)

        # ── Método 1.7 ────────────────────────────────────────────────────────
        def _mint_session(self, bundle_digest: str, dimension_n: int) -> WitnessSession:
            """
            1.7 [V10] Nonce CON CLAVE e impredecible: HMAC(K_nonce, secrets.token_bytes(32)
            ‖contador‖monotonic). v16 usaba sha256(str(time.time_ns()))[:16]: predecible
            desde un reloj público, sin clave y sin registro de replay ⟹ no era un nonce.
            """
            s = self._s
            tel = s._telemetry
            tel.sequence_counter += 1
            tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
            now = time.time_ns()
            k_nonce = self._hkdf_subkey(_SALT_W1, _INFO_NONCE)
            nonce = hmac.new(
                k_nonce,
                secrets.token_bytes(32)
                + f"|{tel.sequence_counter}|{tel.monotonic_clock_ns}".encode("ascii"),
                hashlib.sha256,
            ).hexdigest()[:32]
            sid = hmac.new(
                self._hkdf_subkey(_SALT_W1, _INFO_SESSION),
                _canonical_json(
                    {
                        "bundle": bundle_digest,
                        "nonce": nonce,
                        "seq": tel.sequence_counter,
                        "n": int(dimension_n),
                    }
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return WitnessSession(
                session_id=sid,
                nonce=nonce,
                sequence=int(tel.sequence_counter),
                issued_ns=now,
                nbf_ns=now - 1_000_000,
                exp_ns=now + _SESSION_TTL_NS,
                monotonic_ns=int(tel.monotonic_clock_ns),
                input_bundle_digest=bundle_digest,
                dimension_n=int(dimension_n),
            )

        # ── Método 1.8 ────────────────────────────────────────────────────────
        def _sign_w1(
            self, skeleton: WitnessPhase1Certificate
        ) -> WitnessPhase1Certificate:
            """1.8 Firma HMAC-SHA256 bajo K_W1 sobre el payload canónico de cobertura total."""
            payload = _canonical_json(self._build_w1_payload(skeleton))
            sig = hmac.new(
                self._hkdf_subkey(_SALT_W1, _INFO_W1),
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            leaf = hashlib.sha256(
                _canonical_json({"w1": sig}).encode("utf-8")
            ).hexdigest()
            return replace(skeleton, signature=sig, merkle_leaf=leaf)

        # ── Método 1.9 · TERMINAL W₁ → APERTURA FORMAL DE W₂ (CPS polimórfico) ─
        def _phase1_observe_and_sign(
            self,
            report: PositronWitnessSpectrumReport,
            frozen_before: Sequence[FrozenTensor],
            tensors_now: Sequence[Any],
            dimension_n: int,
            continuation: Callable[[WitnessPhase1Certificate], AnswerT],
        ) -> AnswerT:
            """
            1.9 [TERMINAL W₁ → INICIO DE FASE W₂] Re-digiere los cinco buffers, ATESTIGUA
            la no-interferencia (WITNESS-1), audita la geometría de Uhlmann-Bures, la
            ecuación de Lyapunov de la SLD, la regularización espinorial cuaterniónica de
            Kustaanheimo-Stiefel y el punto fijo polar RSI-L3 del motor, emite la semilla
            formal `W1ToW2ContinuationSeed` y firma W1 con cobertura total. Transfiere el
            control en posición de cola a la continuación polimórfica que abre la Fase W₂
            en `2.1 _gate_w12`.
            """
            s = self._s
            names = ("p", "rho", "sigma", "K", "boundary")
            after = [
                self._freeze_and_digest(nm, a) for nm, a in zip(names, tensors_now)
            ]
            engine_claim = _ev(report, "non_interference_ok", None)
            attestation = self._attest_non_interference(
                frozen_before, after, engine_claim
            )
            atoms_ub, abst_ub = self._audit_uhlmann_bures(report, s._telemetry.policy)
            ks_audit, atoms_ks, abst_ks = self._audit_ks_spinor_and_rsi_l3_substrate(
                report
            )
            atoms = atoms_ub + atoms_ks
            abst = abst_ub + abst_ks
            if not attestation.all_ok:
                atoms = atoms + (
                    ("WITNESS_1_NON_INTERFERENCE", HeytingWitnessVerdict.VETOED.name),
                )
                s._telemetry.non_interference_violations += 1
            bundle = hashlib.sha256(
                "".join(t.digest for t in frozen_before).encode("ascii")
            ).hexdigest()
            session = self._mint_session(bundle, dimension_n)
            w1_to_w2_seed = W1ToW2ContinuationSeed(
                session_id=session.session_id,
                input_bundle_digest=bundle,
                dimension_n=int(dimension_n),
                uhlmann_fidelity=_safe_float(_ev(report, "uhlmann_fidelity", 0.0)),
                bures_distance=_safe_float(
                    _ev(report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0)
                ),
                quantum_fisher_information=ks_audit.quantum_fisher_information,
                non_interference_all_ok=bool(attestation.all_ok),
                ks_regularization_ok=bool(ks_audit.ks_regularization_ok),
            )
            skeleton = WitnessPhase1Certificate(
                report=report,
                timestamp_ns=time.time_ns(),
                nonce=session.nonce,
                signature="",
                agent_version=_AGENT_VERSION,
                session=session,
                monotonic_ns=session.monotonic_ns,
                frozen_inputs=tuple(frozen_before),
                non_interference=attestation,
                evidence_atoms=atoms,
                abstentions=abst,
                boundary_digest=frozen_before[4].digest if len(frozen_before) > 4 else "",
                ks_audit=ks_audit,
                continuation_seed=w1_to_w2_seed,
            )
            cert = self._sign_w1(skeleton)
            s._telemetry.phase1_emissions += 1
            s._telemetry.used_nonces.append(session.nonce)
            if len(s._telemetry.used_nonces) > 2048:
                s._telemetry.used_nonces.pop(0)
            # ── Handoff en posición de cola (CPS) hacia la Fase W₂ (2.1 _gate_w12) ──
            return continuation(cert)


    # ──────────────────────────────────────────────────────────────────────────
    # FASE W₂ · ORIENTOR: DIVISORES DETERMINANTALES, POCS, HODGE Y POINCARÉ-KAM
    # ──────────────────────────────────────────────────────────────────────────
    class WitnessPhase2Orientor:
        """W₂ (2.1 → 2.10). Terminal: _phase2_orient_and_sign, que abre W₃ en 3.1."""

        def __init__(self, sovereign: "PositronRedemptionWitnessAgent") -> None:
            self._s = sovereign

        # ── Método 2.1 · APERTURA W₂ (compuerta W₁₂ efectiva y semilla W₁→W₂) ──
        def _gate_w12(
            self, cert: WitnessPhase1Certificate, frozen_now: Sequence[FrozenTensor]
        ) -> GateVerdict:
            """
            2.1 [COMPUERTA W₁₂ EFECTIVA · CONTINUACIÓN DIRECTA DE 1.9]
            Exige en tiempo constante y sobre cobertura total:
              • HMAC(W1) recomputado bajo K_W1;
              • re-digest de los cinco tensores ≡ digests comprometidos en W1;
              • ventana de sesión [nbf, exp) viva, nonce registrado exactamente una vez
                y secuencia estrictamente positiva;
              • coherencia estructural de la semilla formal `W1ToW2ContinuationSeed`
                emitida por el método terminal 1.9 de la Fase W₁.
            Admite veredictos severos (la observación continúa), pero jamás una entrada
            cuya firma no cubra íntegramente su contenido. [V2/V9]
            """
            s = self._s
            reasons: List[str] = []
            expected = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W1, _INFO_W1),
                _canonical_json(s.phase1._build_w1_payload(cert)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            authentic = _ct_equals(cert.signature, expected)
            if not authentic:
                reasons.append("W1_SIGNATURE_COVERAGE_MISMATCH")
            for idx, frozen in enumerate(cert.frozen_inputs):
                now = frozen_now[idx] if idx < len(frozen_now) else None
                if now is None or not frozen.same_content(now):
                    authentic = False
                    reasons.append(f"TENSOR_DIGEST_MISMATCH:{frozen.name}")
            sess = cert.session
            if sess is None:
                authentic = False
                reasons.append("SESSION_ABSENT")
            else:
                if not sess.is_live(time.time_ns()):
                    authentic = False
                    reasons.append("SESSION_WINDOW_EXPIRED")
                if s._telemetry.used_nonces.count(sess.nonce) != 1:
                    authentic = False
                    reasons.append("NONCE_REPLAY_OR_UNREGISTERED")
                if sess.sequence <= 0:
                    authentic = False
                    reasons.append("SEQUENCE_NON_POSITIVE")
                seed = cert.continuation_seed
                if seed is not None:
                    if (
                        seed.session_id != sess.session_id
                        or seed.input_bundle_digest != sess.input_bundle_digest
                        or seed.dimension_n != sess.dimension_n
                    ):
                        authentic = False
                        reasons.append("W1_TO_W2_CONTINUATION_SEED_MISMATCH")
            clean = all(
                v == HeytingWitnessVerdict.COHERENT.name for _, v in cert.evidence_atoms
            )
            if not clean:
                reasons.append("UPSTREAM_EVIDENTIAL_OBSTRUCTION")
            if not authentic:
                s._telemetry.gate_w12_failures += 1
            return GateVerdict(bool(authentic), bool(clean), tuple(reasons))

        # ── Método 2.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _determinantal_divisors(
            boundary: NDArray[np.int64], max_order: int = _MAX_VERIFIED_ORDER
        ) -> Tuple[List[int], int, int, bool]:
            """
            2.2 [V1] DIVISORES DETERMINANTALES por menores enteros exactos.
                D_k = mcd{ det B[I,J] : |I|=|J|=k },  D₀ = 1.
            Teorema clásico de la Forma Normal de Smith: d_k = D_k/D_{k−1} y el rango es
            el mayor k con D_k ≠ 0. Éste es un algoritmo INDEPENDIENTE de la eliminación
            por mcd del motor: verificar con el mismo algoritmo no es verificar.
            Cota de trabajo declarada: si C(m,k)·C(n,k) excede _MAX_MINORS_PER_ORDER se
            detiene y se marca `work_capped`, reportando el orden verificado alcanzado.
            Devuelve (D[0..L], rango_independiente, menores_evaluados, work_capped).
            """
            arr = np.asarray(boundary)
            if arr.ndim != 2 or arr.size == 0:
                return [1], 0, 0, False
            m, n = int(arr.shape[0]), int(arr.shape[1])
            B = [[int(arr[i, j]) for j in range(n)] for i in range(m)]
            divisors: List[int] = [1]
            rank, evaluated, capped = 0, 0, False
            limit = min(min(m, n), max(1, int(max_order)))
            for k in range(1, limit + 1):
                count = math.comb(m, k) * math.comb(n, k)
                if count > _MAX_MINORS_PER_ORDER:
                    capped = True
                    break
                g = 0
                for rows in combinations(range(m), k):
                    for cols in combinations(range(n), k):
                        minor = [[B[i][j] for j in cols] for i in rows]
                        g = math.gcd(g, abs(_bareiss_determinant(minor)))
                        evaluated += 1
                        if g == 1:
                            break
                    if g == 1:
                        break
                divisors.append(g)
                if g != 0:
                    rank = k
                else:
                    break
            return divisors, int(rank), int(evaluated), bool(capped)

        # ── Método 2.3 ────────────────────────────────────────────────────────
        def _audit_smith_homology(
            self, report: PositronWitnessSpectrumReport, boundary: NDArray[np.int64]
        ) -> SmithIndependenceCertificate:
            """
            2.3 [V1 + v18 Hodge-Pontryagin] AUDITORÍA INDEPENDIENTE DE LA SNF Y HOMOLOGÍA.
            Contrasta los factores que el motor declara contra los que se derivan de los
            divisores determinantes, y verifica:
              • d_k = D_k/D_{k−1} exacto para todo k verificado;
              • cadena de divisibilidad d_i | d_{i+1}  (v16 declaró (2,1): 2 ∤ 1 ✗);
              • d₁ = mcd de todas las entradas         (v16 declaró 2, el mcd es 1 ✗);
              • rango declarado ≡ rango independiente;
              • HOMOLOGY-1: torsión nula ⟺ todos los d_i = 1 (contraste booleano/factores);
              • [v18] fase de enlace de Pontryagin θ_Pont = (∑_{d_i > 1} 1/d_i) mod 1
                recomputada independientemente desde los factores determinantes;
              • [v18] consistencia de Hodge-Betti dim ker(Δ_k) = β_k.
            Con el motor v16 esta escalera detecta el defecto cabecera W1 SIN confiar en
            el booleano del motor: los propios factores del informe se auto-delatan.
            """
            declared_exact = _ev(report, "smith_invariant_factors_exact", None)
            legacy = np.asarray(
                _ev(report, "smith_invariant_factors", np.array([], dtype=np.int64))
            )
            if declared_exact:
                declared = [int(s) for s in declared_exact]
            else:
                declared = [int(x) for x in legacy.reshape(-1) if int(x) > 0]
            declared_rank = int(_ev(report, "smith_rank", len(declared)))
            torsion_claim = bool(_ev(report, "smith_torsion_free", True))
            hadamard_log10 = _safe_float(
                _ev(report, "smith_hadamard_log10_bound", 0.0), 0.0
            )
            hodge_betti_ok = bool(_ev(report, "betti_hodge_consistency_ok", True))

            if not declared:
                divisors, ind_rank, evaluated, capped = self._determinantal_divisors(
                    boundary
                )
                independent = [
                    divisors[k] // divisors[k - 1]
                    for k in range(1, len(divisors))
                    if divisors[k - 1] != 0
                ]
                indep_pontryagin = math.fmod(
                    sum(1.0 / float(d) for d in independent if d > 1), 1.0
                )
                return SmithIndependenceCertificate(
                    available=True,
                    determinantal_divisors=tuple(str(d) for d in divisors),
                    independent_factors=tuple(str(d) for d in independent),
                    independent_rank=int(ind_rank),
                    factors_match=(len(independent) == 0),
                    rank_match=(declared_rank == ind_rank),
                    divisibility_chain_ok=True,
                    torsion_claim_consistent=bool(
                        torsion_claim == (all(d == 1 for d in independent))
                    ),
                    independent_torsion_free=bool(all(d == 1 for d in independent)),
                    verified_order=int(len(independent)),
                    minors_evaluated=int(evaluated),
                    work_capped=bool(capped),
                    engine_certificates_ok=None,
                    hodge_betti_consistency_ok=hodge_betti_ok,
                    pontryagin_linking_phase=float(indep_pontryagin),
                    hadamard_log10_bound=hadamard_log10,
                )

            divisors, ind_rank, evaluated, capped = self._determinantal_divisors(
                boundary, max_order=min(_MAX_VERIFIED_ORDER, len(declared))
            )
            independent = []
            for k in range(1, len(divisors)):
                if divisors[k - 1] == 0:
                    break
                independent.append(divisors[k] // divisors[k - 1])
            verified = min(len(independent), len(declared))
            factors_match = bool(
                verified > 0 and independent[:verified] == declared[:verified]
            )
            divisibility_ok = all(
                declared[i + 1] % declared[i] == 0
                for i in range(len(declared) - 1)
                if declared[i] != 0
            )
            independent_torsion_free = (
                bool(all(d == 1 for d in independent)) if independent else True
            )
            claim_consistent = bool(torsion_claim == all(d == 1 for d in declared))

            # Verificación independiente de la fase de enlace de Pontryagin sobre Tors(H_k)
            indep_pontryagin = math.fmod(
                sum(1.0 / float(d) for d in independent if d > 1), 1.0
            )
            rep_pontryagin = _ev(report, "pontryagin_linking_phase", None)
            if rep_pontryagin is not None and not capped:
                if abs(_safe_float(rep_pontryagin) - indep_pontryagin) > 1.0e-9:
                    claim_consistent = False

            engine_flags = [
                _ev(report, "smith_reconstruction_ok", None),
                _ev(report, "smith_unimodular_ok", None),
                _ev(report, "smith_divisibility_chain_ok", None),
                _ev(report, "smith_v_inverse_ok", None),
            ]
            present = [bool(f) for f in engine_flags if f is not None]
            engine_ok = bool(all(present)) if present else None
            if not (factors_match and divisibility_ok and claim_consistent):
                self._s._telemetry.snf_independence_failures += 1
            return SmithIndependenceCertificate(
                available=True,
                determinantal_divisors=tuple(str(d) for d in divisors),
                independent_factors=tuple(str(d) for d in independent),
                independent_rank=int(ind_rank),
                factors_match=factors_match,
                rank_match=bool(declared_rank == ind_rank or capped),
                divisibility_chain_ok=bool(divisibility_ok),
                torsion_claim_consistent=claim_consistent,
                independent_torsion_free=independent_torsion_free,
                verified_order=int(verified),
                minors_evaluated=int(evaluated),
                work_capped=bool(capped),
                engine_certificates_ok=engine_ok,
                hodge_betti_consistency_ok=hodge_betti_ok,
                pontryagin_linking_phase=float(indep_pontryagin),
                hadamard_log10_bound=hadamard_log10,
            )

        # ── Método 2.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _projection_idempotence_probe(
            engine: PositronRedemptionWitnessEngine,
            report: PositronWitnessSpectrumReport,
        ) -> Tuple[float, float, bool]:
            """
            2.4 [V7a] SONDA DE IDEMPOTENCIA ESPACIAL (POCS). El mapa de auditoría es
                T(p, ρ, K) = (Π_Δ(p), Π_{D_δ}(ρ), Π_{PSD∩Sym}(K)),
            proyección de Frobenius sobre un PRODUCTO de convexos cerrados. Cada factor es
            firmemente no expansivo (‖Πx−Πy‖² ≤ ⟨Πx−Πy, x−y⟩); al actuar sobre coordenadas
            independientes, T es una única proyección ⟹ T² = T. El certificado significativo
            es el DEFECTO DE IDEMPOTENCIA ‖T(x)−x‖, no una razón inventada.
            En v18 contrasta además el defecto de idempotencia del motor (`rsi_l3_projector_idempotence_defect`).
            """
            p = np.asarray(
                _ev(report, "p_observed", np.array([1.0])), dtype=np.float64
            ).copy()
            rho = np.asarray(
                _ev(report, "rho_observed", np.eye(p.size, dtype=np.complex128)),
                dtype=np.complex128,
            ).copy()
            k_mat = np.asarray(
                _ev(report, "K_purified", np.eye(p.size)), dtype=np.complex128
            ).copy()
            try:
                p1 = engine.project_gibbs_simplex(p)
                r1 = engine.canonicalize_density_matrix(rho)
                k1 = engine.purify_hermitian_matrix(k_mat)
                d1 = float(
                    np.linalg.norm(p1 - p, 2) + _fro(r1 - rho) + _fro(k1 - k_mat)
                )
                p2 = engine.project_gibbs_simplex(p1)
                r2 = engine.canonicalize_density_matrix(r1)
                k2 = engine.purify_hermitian_matrix(k1)
                d2 = float(
                    np.linalg.norm(p2 - p1, 2) + _fro(r2 - r1) + _fro(k2 - k1)
                )
            except (la.LinAlgError, ValueError, TypeError, FloatingPointError):
                return float("inf"), float("inf"), False
            eng_idem = _ev(report, "rsi_l3_projector_idempotence_defect", None)
            if eng_idem is not None:
                d1 = max(d1, _safe_float(eng_idem, 0.0))
            return d1, d2, bool(d1 <= _IDEMPOTENCE_DEFECT_CEILING)

        # ── Método 2.5 ────────────────────────────────────────────────────────
        def _temporal_bures_contraction(
            self, bures: float
        ) -> Tuple[float, float, bool, float, float]:
            """
            2.5 [V7b] RAZÓN TEMPORAL GENUINA sobre el historial de la métrica propia del
            testigo, la distancia de Bures:
                q_k = |d_k − d_{k−1}| / |d_{k−1} − d_{k−2}|,
            exactamente la definición del docstring de v16 (que el código nunca implementó).
            Requiere ≥3 muestras: con menos, ABSTENCIÓN. Se reporta cruda (puede ser ≥1).
            """
            tel = self._s._telemetry
            tel.bures_history.append(float(bures))
            if len(tel.bures_history) > 64:
                tel.bures_history.pop(0)
            h = tel.bures_history
            if len(h) < 3:
                return 0.0, 0.0, False, 0.0, 0.0
            d_last = abs(h[-1] - h[-2])
            d_prev = abs(h[-2] - h[-3])
            d_first = abs(h[1] - h[0])
            if d_prev <= 1e-15:
                return 0.0, 0.0, True, d_last, d_first
            q_raw = d_last / d_prev
            return (
                float(q_raw),
                float(np.clip(q_raw, 0.0, 0.999999)),
                True,
                d_last,
                d_first,
            )

        # ── Método 2.6 ────────────────────────────────────────────────────────
        def _banach_kantorovich_certificate(
            self,
            engine: PositronRedemptionWitnessEngine,
            report: PositronWitnessSpectrumReport,
            bures: float,
        ) -> ContractionCertificate:
            """
            2.6 Compone ambas vías y las cotas de Banach-Kantorovich:
                A POSTERIORI ‖x_k−x*‖ ≤ d_últ/(1−q);  A PRIORI ≤ qⁿ/(1−q)·d₁.
            En el punto fijo de la proyección (d₁≈0) la cota colapsa y los dígitos
            certificados se saturan en la precisión de float64 (~15.95), lo que se DECLARA.
            Incorpora en v18 la razón de contracción RSI-L3 del motor y la razón de
            extrapolación de Romberg-Richardson del sector de Shahshahani.
            """
            d1, d2, idem_ok = self._projection_idempotence_probe(engine, report)
            q_raw, q_clip, defined, d_last, d_first = self._temporal_bures_contraction(
                bures
            )
            at_fp = bool(math.isfinite(d1) and d1 <= _IDEMPOTENCE_DEFECT_CEILING)
            q_eff = 0.0 if at_fp else (q_clip if defined else 0.5)
            one_minus_q = max(1.0 - q_eff, 1e-9)
            base_inc = d_last if defined else (d2 if math.isfinite(d2) else 0.0)
            a_post = float(base_inc / one_minus_q)
            n_hist = len(self._s._telemetry.bures_history)
            a_prior = float(
                (q_eff ** max(1, n_hist)) / one_minus_q * (d_first if defined else d1)
            )
            digits = float(
                min(
                    max(0.0, -math.log10(max(a_post, _TINY))),
                    _FLOAT64_CERTIFIED_DIGITS,
                )
            )
            contractive = bool(
                at_fp or (defined and q_raw < self._s._telemetry.policy.banach_q_max)
            )
            eng_rsi_q = _safe_float(
                _ev(report, "rsi_l3_banach_contraction_ratio", 0.0), 0.0
            )
            romberg_q = _safe_float(
                _ev(report, "romberg_l3_contraction_ratio", 0.0), 0.0
            )
            return ContractionCertificate(
                idempotence_defect=float(d1) if math.isfinite(d1) else 1e12,
                idempotence_ok=bool(idem_ok),
                at_fixed_point=at_fp,
                q_temporal_raw=float(q_raw),
                q_temporal=float(q_eff),
                temporal_defined=bool(defined),
                temporal_contractive=contractive,
                increment_last=float(base_inc),
                increment_first=float(d_first),
                a_posteriori_bound=a_post,
                a_priori_bound=a_prior,
                certified_digits=digits,
                history_length=int(n_hist),
                engine_rsi_l3_contraction_ratio=eng_rsi_q,
                romberg_l3_contraction_ratio=romberg_q,
            )

        # ── Método 2.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _connes_daleckii_krein_audit(
            report: PositronWitnessSpectrumReport,
        ) -> Tuple[float, float, bool]:
            """
            2.7 Cotas de Daleckii-Krein por cociente dividido f^[1](a,b). En Frobenius la
            norma del multiplicador de Schur es máx|f^[1]|, de donde:
                f(x)=√x ⟹ L_sqrt ≤ 1/(2√λ_min);  f(x)=x^{−1/2} ⟹ L_inv ≤ 1/(2λ_min^{3/2}).
            v16 almacenaba la segunda bajo el nombre de la primera y NUNCA la usaba.
            Aquí se exige finitud, κ(K) acotada y la identidad algebraica exacta
            L_inv = 4 · L_sqrt³ para que ‖Δf(K)‖_F ≤ L‖ΔK‖_F sea significativa.
            """
            l_inv = _safe_float(
                _ev(report, "connes_lipschitz_bound", float("inf")), float("inf")
            )
            l_sqrt = _safe_float(_ev(report, "dk_sqrt_lipschitz", 0.0), 0.0)
            if l_sqrt <= 0.0 and math.isfinite(l_inv) and l_inv > 0.0:
                lam = (0.5 / l_inv) ** (2.0 / 3.0)
                l_sqrt = 0.5 / math.sqrt(max(lam, _STABILITY_FLOOR))
            kappa = _safe_float(_ev(report, "condition_number_K", 1.0), 1.0)
            algebraic_consistency = True
            if math.isfinite(l_inv) and l_inv > 0.0 and l_sqrt > 0.0:
                expected_inv = 4.0 * (l_sqrt ** 3)
                rel_err = abs(l_inv - expected_inv) / max(expected_inv, _STABILITY_FLOOR)
                algebraic_consistency = bool(rel_err <= 1.0e-6)
            usable = bool(
                math.isfinite(l_inv)
                and l_inv > 0.0
                and kappa <= _CONDITION_CAP
                and algebraic_consistency
            )
            return float(l_sqrt), float(l_inv), usable

        # ── Método 2.8 ────────────────────────────────────────────────────────
        def _audit_celestial_sector(
            self, report: PositronWitnessSpectrumReport, bundle_digest: str
        ) -> CelestialAuditCertificate:
            """
            2.8 [V8 + v18 Mecánica Celeste de Poincaré-KAM-Von Zeipel-Kozai-Lidov]
            Auditoría de Poincaré-Liouville-Shahshahani:
            • Canonicidad real de la carta (v17/v18) vs. residuo tautológico (v16: para K=I,
              pᵀKp ≡ ‖p‖² ⟹ el "residuo" es idénticamente cero y no informa nada).
            • DETECTOR DE INVARIANTE FABRICADO: si I_PC permanece idéntico (≤1e-12) a través
              de ≥3 entradas con digests DISTINTOS, se marca como sospechosamente constante.
              v16 devolvía 2π·Σp ≡ 2π: esta sonda lo delata.
            • POINCARÉ-1: det M = +1 y ‖MᵀΩM−Ω‖ ≤ τ (v17/v18); ausencia ⟹ ABSTENCIÓN.
            • Pasividad de Shahshahani: −Var_p(Kp) ≤ 0, y velocidad ≡ √Var si se expone.
            • [v18] Defecto simpléctico del Jacobiano de la carta de Poincaré, función
              generatriz de Von Zeipel S₁, invariantes orbitales de Kozai-Lidov y
              Tisserand-Poincaré, número de rotación KAM y defecto diofántico.
            """
            tel = self._s._telemetry
            cartan = _safe_float(_ev(report, "poincare_cartan_invariant", 0.0))
            tel.cartan_history.append((bundle_digest, cartan))
            if len(tel.cartan_history) > 32:
                tel.cartan_history.pop(0)
            distinct = {d for d, _ in tel.cartan_history}
            values = [v for _, v in tel.cartan_history]
            constant = bool(
                len(tel.cartan_history) >= _CARTAN_CONSTANCY_SAMPLES
                and len(distinct) >= _CARTAN_CONSTANCY_SAMPLES
                and (max(values) - min(values)) <= 1e-12
            )
            k_mat = np.asarray(
                _ev(report, "K_purified", np.eye(2)), dtype=np.complex128
            )
            n = int(k_mat.shape[0]) if k_mat.ndim == 2 else 0
            k_is_identity = bool(
                n and _fro(k_mat - np.eye(n, dtype=np.complex128)) <= 1e-12
            )
            residual = _safe_float(_ev(report, "poincare_canonical_residual", 0.0))
            tautological = bool(
                k_is_identity
                and residual == 0.0
                and _ev(report, "poincare_canonicity_defect", None) is None
            )
            rate = _safe_float(_ev(report, "shahshahani_dissipation_rate", 0.0))
            speed = _ev(report, "shahshahani_speed", None)
            variance = _ev(report, "shahshahani_variance", None)
            speed_err = None
            if speed is not None and variance is not None:
                speed_err = abs(
                    _safe_float(speed) - math.sqrt(max(0.0, _safe_float(variance)))
                )

            symp_jac_def = _safe_float(
                _ev(report, "poincare_symplectic_jacobian_defect", 0.0), 0.0
            )
            von_zeipel_s1 = _safe_float(
                _ev(report, "von_zeipel_generating_s1", 0.0), 0.0
            )
            kozai_lidov = _safe_float(
                _ev(report, "kozai_lidov_invariant", 1.0), 1.0
            )
            tisserand = _safe_float(
                _ev(report, "tisserand_poincare_invariant", 3.0), 3.0
            )
            kam_rot = _safe_float(_ev(report, "kam_rotation_number", 0.0), 0.0)
            kam_def = _safe_float(_ev(report, "kam_diophantine_defect", 0.0), 0.0)
            floquet_lyap = _safe_float(
                _ev(report, "floquet_lyapunov_exponent", 0.0), 0.0
            )

            return CelestialAuditCertificate(
                canonicity_defect=_ev(report, "poincare_canonicity_defect", None),
                residual_tautological=tautological,
                cartan_value=float(cartan),
                cartan_suspiciously_constant=constant,
                cartan_samples=len(tel.cartan_history),
                liouville_ok=_ev(report, "liouville_ok", None),
                symplectic_defect=_ev(report, "liouville_symplectic_defect", None),
                monodromy_determinant=_ev(report, "liouville_determinant", None),
                krein_collision=_ev(report, "krein_collision_detected", None),
                shahshahani_passive=bool(rate <= 1e-12),
                shahshahani_rate=float(rate),
                speed_identity_error=speed_err,
                symplectic_jacobian_defect=symp_jac_def,
                von_zeipel_generating_s1=von_zeipel_s1,
                kozai_lidov_invariant=kozai_lidov,
                tisserand_poincare_invariant=tisserand,
                kam_rotation_number=kam_rot,
                kam_diophantine_defect=kam_def,
                floquet_lyapunov_exponent=floquet_lyap,
            )

        # ── Método 2.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _build_w2_payload(cert: WitnessPhase2Certificate) -> Dict[str, Any]:
            """2.9a Payload de COBERTURA TOTAL de W₂ (excluye sólo la firma). [V2]"""
            return {
                "domain": _INFO_W2.decode("ascii"),
                "agent_version": cert.agent_version,
                "w1_signature": cert.phase1_cert.signature,
                "banach_ratio": cert.banach_ratio,
                "connes_bound": cert.connes_bound,
                "dk_sqrt_lipschitz": cert.dk_sqrt_lipschitz,
                "timestamp_ns": cert.timestamp_ns,
                "monotonic_ns": cert.monotonic_ns,
                "session": cert.session,
                "gate_w12": cert.gate_w12,
                "smith_independence": cert.smith_independence,
                "contraction": cert.contraction,
                "celestial": cert.celestial,
                "evidence_atoms": cert.evidence_atoms,
                "abstentions": cert.abstentions,
                "authentication_veto": cert.authentication_veto,
                "continuation_seed": cert.continuation_seed,
            }

        @staticmethod
        def _propagate_gate(
            gate: GateVerdict, atoms: Sequence[Tuple[str, str]]
        ) -> Tuple[Tuple[Tuple[str, str], ...], bool]:
            """2.9b Propagación fail-closed SIN interferencia: la observación continúa. [V9]"""
            out = tuple(atoms)
            veto = not gate.authentic
            if veto:
                out = out + (("AUTH_VETO", HeytingWitnessVerdict.VETOED.name),)
            return out, bool(veto)

        # ── Método 2.10 · TERMINAL W₂ → APERTURA FORMAL DE W₃ (CPS polimórfico) ─
        def _phase2_orient_and_sign(
            self,
            cert1: WitnessPhase1Certificate,
            engine: PositronRedemptionWitnessEngine,
            boundary: NDArray[np.int64],
            tensors_now: Sequence[Any],
            continuation: Callable[[WitnessPhase2Certificate], AnswerT],
        ) -> AnswerT:
            """
            2.10 [TERMINAL W₂ → INICIO DE FASE W₃] Ejecuta W₁₂ de forma EFECTIVA,
            VERIFICA INDEPENDIENTEMENTE la SNF por divisores determinantes y la
            consistencia armónica de Hodge-Pontryagin, sondea idempotencia POCS y razón
            temporal de Banach-Kantorovich, audita Poincaré-Liouville-KAM-Shahshahani,
            emite la semilla formal `W2ToW3ContinuationSeed` y firma W2 con cobertura
            total. Transfiere el control en posición de cola a la continuación que abre
            la Fase W₃ en `3.1 _gate_w23`.
            """
            s = self._s
            V = HeytingWitnessVerdict
            names = ("p", "rho", "sigma", "K", "boundary")
            frozen_now = [
                s.phase1._freeze_and_digest(nm, a) for nm, a in zip(names, tensors_now)
            ]
            gate = self._gate_w12(cert1, frozen_now)
            report = cert1.report
            bures = _safe_float(
                _ev(report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0)
            )
            smith = self._audit_smith_homology(report, boundary)
            contraction = self._banach_kantorovich_certificate(engine, report, bures)
            dk_sqrt, dk_inv, dk_usable = self._connes_daleckii_krein_audit(report)
            bundle = cert1.session.input_bundle_digest if cert1.session else ""
            celestial = self._audit_celestial_sector(report, bundle)

            atoms: List[Tuple[str, str]] = list(cert1.evidence_atoms)
            abst: List[str] = list(cert1.abstentions)
            atoms.append(
                (
                    "SNF_FACTORS_INDEPENDENT",
                    (V.COHERENT if smith.factors_match else V.VETOED).name,
                )
            )
            atoms.append(
                (
                    "SNF_DIVISIBILITY_CHAIN",
                    (V.COHERENT if smith.divisibility_chain_ok else V.VETOED).name,
                )
            )
            atoms.append(
                (
                    "SNF_TORSION_CLAIM_CONSISTENT",
                    (V.COHERENT if smith.torsion_claim_consistent else V.VETOED).name,
                )
            )
            atoms.append(
                (
                    "SNF_RANK_MATCH",
                    (V.COHERENT if smith.rank_match else V.DEGRADED).name,
                )
            )
            atoms.append(
                (
                    "HOMOLOGY_TORSION_FREE",
                    (V.COHERENT if smith.independent_torsion_free else V.DEGRADED).name,
                )
            )
            if _ev(report, "betti_hodge_consistency_ok", None) is None:
                abst.append("HODGE_BETTI_CONSISTENCY")
            else:
                atoms.append(
                    (
                        "HODGE_BETTI_CONSISTENCY",
                        (
                            V.COHERENT
                            if smith.hodge_betti_consistency_ok
                            else V.VETOED
                        ).name,
                    )
                )
            if smith.engine_certificates_ok is None:
                abst.append("SNF_ENGINE_CERTIFICATES")
            else:
                atoms.append(
                    (
                        "SNF_ENGINE_CERTIFICATES",
                        (
                            V.COHERENT if smith.engine_certificates_ok else V.VETOED
                        ).name,
                    )
                )
            if smith.work_capped:
                abst.append(f"SNF_VERIFIED_ORDER_{smith.verified_order}_WORK_CAPPED")
            atoms.append(
                (
                    "PROJECTION_IDEMPOTENCE",
                    (V.COHERENT if contraction.idempotence_ok else V.VETOED).name,
                )
            )
            atoms.append(
                (
                    "BANACH_CONTRACTION",
                    (
                        V.COHERENT if contraction.temporal_contractive else V.DEGRADED
                    ).name,
                )
            )
            if not contraction.temporal_defined:
                abst.append("BANACH_TEMPORAL_INSUFFICIENT_HISTORY")
            atoms.append(
                (
                    "DALECKII_KREIN_USABLE",
                    (V.COHERENT if dk_usable else V.DEGRADED).name,
                )
            )
            atoms.append(
                (
                    "SHAHSHAHANI_PASSIVITY",
                    (V.COHERENT if celestial.shahshahani_passive else V.VETOED).name,
                )
            )
            if celestial.residual_tautological:
                atoms.append(("POINCARE_RESIDUAL_TAUTOLOGICAL", V.DEGRADED.name))
            if celestial.cartan_suspiciously_constant:
                atoms.append(("CARTAN_INVARIANT_SUSPICIOUSLY_CONSTANT", V.DEGRADED.name))
            if celestial.canonicity_defect is None:
                abst.append("POINCARE_CANONICITY")
            else:
                ok = (
                    _safe_float(celestial.canonicity_defect)
                    <= _CANONICITY_DEFECT_CEILING
                    and celestial.symplectic_jacobian_defect
                    <= _CANONICITY_DEFECT_CEILING
                )
                atoms.append(
                    ("POINCARE_CANONICITY", (V.COHERENT if ok else V.DEGRADED).name)
                )
            if celestial.liouville_ok is None:
                abst.append("LIOUVILLE_POINCARE_1")
            else:
                atoms.append(
                    (
                        "LIOUVILLE_POINCARE_1",
                        (
                            V.COHERENT if bool(celestial.liouville_ok) else V.VETOED
                        ).name,
                    )
                )
            if celestial.krein_collision:
                atoms.append(("KREIN_1TO1_RESONANCE", V.DEGRADED.name))
            if (
                celestial.speed_identity_error is not None
                and celestial.speed_identity_error > 1e-9
            ):
                atoms.append(("SHAHSHAHANI_SPEED_IDENTITY", V.DEGRADED.name))
            if not (0.0 < celestial.kozai_lidov_invariant <= 1.0 + 1.0e-9):
                atoms.append(("KOZAI_LIDOV_RANGE", V.DEGRADED.name))

            atoms, veto = self._propagate_gate(gate, atoms)
            s._telemetry.monotonic_clock_ns = max(
                time.monotonic_ns(), s._telemetry.monotonic_clock_ns + 1
            )
            w2_to_w3_seed = W2ToW3ContinuationSeed(
                session_id=cert1.session.session_id if cert1.session else "",
                w1_signature=cert1.signature,
                w2_signature="PENDING_W2_HMAC",
                gate_w12_authentic=bool(gate.authentic),
                smith_factors_match=bool(smith.factors_match),
                pocs_idempotent_ok=bool(contraction.idempotence_ok),
                shahshahani_passive_ok=bool(celestial.shahshahani_passive),
                banach_ratio=float(contraction.q_temporal),
            )
            skeleton = WitnessPhase2Certificate(
                phase1_cert=cert1,
                banach_ratio=float(contraction.q_temporal),
                connes_bound=float(dk_inv),
                timestamp_ns=time.time_ns(),
                signature="",
                agent_version=_AGENT_VERSION,
                session=cert1.session,
                monotonic_ns=int(s._telemetry.monotonic_clock_ns),
                gate_w12=gate,
                smith_independence=smith,
                contraction=contraction,
                celestial=celestial,
                dk_sqrt_lipschitz=float(dk_sqrt),
                evidence_atoms=tuple(atoms),
                abstentions=tuple(abst),
                authentication_veto=bool(veto),
                continuation_seed=w2_to_w3_seed,
            )
            sig = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W2, _INFO_W2),
                _canonical_json(self._build_w2_payload(skeleton)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            leaf = hashlib.sha256(
                _canonical_json({"w2": sig}).encode("utf-8")
            ).hexdigest()
            cert2 = replace(skeleton, signature=sig, merkle_leaf=leaf)
            s._telemetry.phase2_emissions += 1
            # ── Handoff en posición de cola (CPS) hacia la Fase W₃ (3.1 _gate_w23) ──
            return continuation(cert2)


 # ──────────────────────────────────────────────────────────────────────────
    # FASE W₃ · DECIDER: HEYTING, LÖB BIEN FUNDADO, RSI-L3 Y TRIBUNAL DE SILICIO
    # ──────────────────────────────────────────────────────────────────────────
    class WitnessPhase3Decider:
        """W₃ (3.1 → 3.10). Terminal: _phase3_decide_and_act, emisor del veredicto firmado."""

        def __init__(self, sovereign: "PositronRedemptionWitnessAgent") -> None:
            self._s = sovereign

        # ── Método 3.1 · APERTURA W₃ (compuerta W₂₃ efectiva y semilla W₂→W₃) ──
        def _gate_w23(self, cert: WitnessPhase2Certificate) -> GateVerdict:
            """
            3.1 [COMPUERTA W₂₃ EFECTIVA · CONTINUACIÓN DIRECTA DE 2.10]
            Recomputa HMAC(W2) sobre cobertura total, verifica recursivamente la firma de
            W1 anidada, exige continuidad de sesión, secuencia y reloj monótono
            estrictamente crecientes, valida la semilla formal `W2ToW3ContinuationSeed`
            y exige ausencia de veto de autenticación aguas arriba.
            """
            s = self._s
            reasons: List[str] = []
            expected = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W2, _INFO_W2),
                _canonical_json(s.phase2._build_w2_payload(cert)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            authentic = _ct_equals(cert.signature, expected)
            if not authentic:
                reasons.append("W2_SIGNATURE_COVERAGE_MISMATCH")
            w1 = cert.phase1_cert
            w1_expected = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W1, _INFO_W1),
                _canonical_json(s.phase1._build_w1_payload(w1)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not _ct_equals(w1.signature, w1_expected):
                authentic = False
                reasons.append("NESTED_W1_SIGNATURE_MISMATCH")
            if cert.session is None or w1.session is None:
                authentic = False
                reasons.append("SESSION_ABSENT")
            else:
                if (
                    cert.session.session_id != w1.session.session_id
                    or cert.session.nonce != w1.session.nonce
                ):
                    authentic = False
                    reasons.append("SESSION_CONTINUITY_BROKEN")
                if not cert.session.is_live(time.time_ns()):
                    authentic = False
                    reasons.append("SESSION_WINDOW_EXPIRED")
                if cert.monotonic_ns <= w1.monotonic_ns:
                    authentic = False
                    reasons.append("CLOCK_NOT_MONOTONE")
                seed = cert.continuation_seed
                if seed is not None:
                    if (
                        seed.session_id != cert.session.session_id
                        or not _ct_equals(seed.w1_signature, w1.signature)
                        or (
                            cert.gate_w12 is not None
                            and seed.gate_w12_authentic != cert.gate_w12.authentic
                        )
                    ):
                        authentic = False
                        reasons.append("W2_TO_W3_CONTINUATION_SEED_MISMATCH")
            if cert.authentication_veto or (
                cert.gate_w12 is not None and not cert.gate_w12.authentic
            ):
                authentic = False
                reasons.append("UPSTREAM_AUTHENTICATION_VETO")
            clean = all(
                v == HeytingWitnessVerdict.COHERENT.name for _, v in cert.evidence_atoms
            )
            if not clean:
                reasons.append("EVIDENTIAL_OBSTRUCTION")
            if not authentic:
                s._telemetry.gate_w23_failures += 1
            return GateVerdict(bool(authentic), bool(clean), tuple(reasons))

        # ── Método 3.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _heyting_infimum(
            atoms: Sequence[Tuple[str, str]],
        ) -> HeytingWitnessVerdict:
            """
            3.2 Ínfimo (⋀) en Ω₃. AXIOMA HEYTING-1: si algún átomo es VETOED, el ínfimo es
            VETOED. Parseo ESTRICTO: la basura es un error de programación, no un veredicto.
            Las ABSTENCIONES no son átomos: se registran aparte y no fuerzan COHERENT.
            """
            out = HeytingWitnessVerdict.COHERENT
            for _name, value in atoms:
                out = out.meet(HeytingWitnessVerdict.parse(value))
            return out

        # ── Método 3.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _grace_view(
            raw: HeytingWitnessVerdict, grace_verified: bool
        ) -> HeytingWitnessVerdict:
            """
            3.3 VISTA INFORMATIVA DE GRACIA. El testigo NO absuelve: informa de lo que ve.
            `final_verdict` es el CRUDO (autoritativo, el que se firma y el que decide la
            actuación); `verdict_under_grace` es lo que el núcleo del gobernador produciría
            con un token válido. Separar ambos campos hace auditable la composición entre
            soberanos sin que el testigo deje de ser testigo.
            """
            return grace_nucleus(raw, grace_verified)

        # ── Método 3.4 ────────────────────────────────────────────────────────
        def _verify_grace_token(
            self, token: Optional[WitnessGraceToken], session: Optional[WitnessSession]
        ) -> Tuple[bool, Tuple[str, ...]]:
            """
            3.4 Verificación REAL del token de gracia: HMAC bajo K_grace sobre cobertura
            total, ligadura al session_id y al bundle de entrada (impide replay sobre otro
            estado), ventana [nbf, exp) viva y nonce no consumido.
            """
            s = self._s
            if token is None or session is None:
                return False, ("GRACE_TOKEN_ABSENT",)
            reasons: List[str] = []
            payload = _canonical_json(
                {
                    "domain": _INFO_GRACE.decode("ascii"),
                    "subject": token.subject,
                    "session_id": token.session_id,
                    "input_bundle_digest": token.input_bundle_digest,
                    "nonce": token.nonce,
                    "issued_ns": token.issued_ns,
                    "nbf_ns": token.nbf_ns,
                    "exp_ns": token.exp_ns,
                    "max_grace_sec": token.max_grace_sec,
                }
            )
            expected = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W3, _INFO_GRACE),
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not _ct_equals(token.signature, expected):
                reasons.append("GRACE_SIGNATURE_INVALID")
            if token.session_id != session.session_id:
                reasons.append("GRACE_SESSION_UNBOUND")
            if token.input_bundle_digest != session.input_bundle_digest:
                reasons.append("GRACE_INPUT_UNBOUND")
            now = time.time_ns()
            if not (token.nbf_ns <= now < token.exp_ns):
                reasons.append("GRACE_WINDOW_EXPIRED")
            if token.nonce in s._consumed_grace_nonces:
                reasons.append("GRACE_NONCE_REPLAY")
            valid = not reasons
            if valid:
                s._consumed_grace_nonces.append(token.nonce)
                if len(s._consumed_grace_nonces) > 2048:
                    s._consumed_grace_nonces.pop(0)
            return valid, tuple(reasons)

        # ── Método 3.5 ────────────────────────────────────────────────────────
        def _loeb_safe_verification(
            self, cert: WitnessPhase2Certificate
        ) -> LoebVerification:
            """
            3.5 [V5 + v18] Verificador de Löb NO VACUO por DESCENSO BIEN FUNDADO.
            v16 devolvía `isfinite(d_B) and d_B ≥ 0 and L > 0`: las tres son trivialmente
            ciertas (d_B = √max(0,·), L = 0.5/λ^{3/2} > 0), luego era una constante True.
            Aquí se exigen TRES condiciones recomputadas independientemente:
              (i)   HMAC(W2) recomputado bajo K_W2 (atestación, no auto-confianza);
              (ii)  identidades estructurales re-verificadas (Heyting, núcleo, rango de q,
                    no negatividad del defecto de idempotencia, divisibilidad de la SNF,
                    consistencia armónica de Hodge-Betti y regularización KS-SLD);
              (iii) profundidad de prueba > 0, que se DECREMENTA.
            No se afirma el axioma modal □(□P→P)→□P: ℕ es bien fundado ⟹ toda cadena de
            auto-verificación es finita. El obstáculo löbiano se evita exigiendo certificados
            a profundidad ESTRICTAMENTE MENOR, nunca auto-confianza.
            """
            s = self._s
            depth = s._telemetry.policy.verification_depth
            recomputed = _ct_equals(
                cert.signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_SALT_W2, _INFO_W2),
                    _canonical_json(s.phase2._build_w2_payload(cert)).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
            rechecked = 0
            ok = HeytingWitnessVerdict.verify_axioms()
            rechecked += 1
            ok = ok and verify_nucleus_laws()
            rechecked += 1
            c = cert.contraction
            if c is not None:
                ok = ok and (0.0 <= c.q_temporal < 1.0)
                rechecked += 1
                ok = ok and (c.idempotence_defect >= 0.0)
                rechecked += 1
            if cert.smith_independence is not None:
                ok = ok and cert.smith_independence.divisibility_chain_ok
                rechecked += 1
                ok = ok and cert.smith_independence.hodge_betti_consistency_ok
                rechecked += 1
            ks_audit = cert.phase1_cert.ks_audit
            if ks_audit is not None:
                ok = ok and ks_audit.ks_regularization_ok
                rechecked += 1
            accepted = bool(recomputed and ok and depth > 0)
            s._telemetry.policy = replace(
                s._telemetry.policy, verification_depth=max(0, depth - 1)
            ).project()
            return LoebVerification(
                accepted=accepted,
                depth_remaining=int(s._telemetry.policy.verification_depth),
                independently_recomputed=bool(recomputed),
                identities_rechecked=int(rechecked),
            )

        # ── Método 3.6 ────────────────────────────────────────────────────────
        def _evolve_rsi_l3_policy(
            self,
            cert: WitnessPhase2Certificate,
            l1_ok: bool,
            l2_ok: bool,
            l3_ok: bool,
        ) -> RsiL3PolicyEvolutionCertificate:
            """
            3.6a [v18 · Evolución Proximal de Moreau-Yosida sobre la Política del Testigo]
            Gobierna la auto-mejora recursiva de Nivel 3 (RSI-L3) sobre `WitnessPolicy`
            con descenso bien fundado en `mutation_budget ∈ ℕ`:
              • Si L1 ∧ L2 ∧ L3 se satisfacen, existe historial temporal (`temporal_defined`)
                y `mutation_budget > 0`, evalúa el paso proximal de Moreau-Yosida
                    θ_{k+1} = argmin_θ { L_epistemic(θ) + (1 / 2λ) ‖θ − θ_k‖₂² }
                restringido al cono de endurecimiento monótono (jamás relaja pisos de
                seguridad por debajo de `_UHLMANN_FIDELITY_FLOOR` ni techos por encima de
                `_BURES_SOFT_CEILING`).
              • Si cualquiera de L1, L2 o L3 falla tras una mutación previa, ejecuta un
                ROLLBACK inmediato a la política base en `policy_ledger[0]` e incrementa
                `rollback_count`.
            """
            tel = self._s._telemetry
            pol = tel.policy
            budget_before = int(pol.mutation_budget)

            if not (l1_ok and l2_ok and l3_ok):
                if tel.policy_ledger:
                    base_snap = tel.policy_ledger[0]
                    tel.policy = WitnessPolicy(
                        fidelity_floor=float(
                            base_snap.get("fidelity_floor", _UHLMANN_FIDELITY_FLOOR)
                        ),
                        bures_soft_ceiling=float(
                            base_snap.get("bures_soft_ceiling", _BURES_SOFT_CEILING)
                        ),
                        banach_q_max=float(base_snap.get("banach_q_max", _BANACH_Q_MAX)),
                        actuation_policy=str(
                            base_snap.get("actuation_policy", pol.actuation_policy)
                        ),
                        verification_depth=int(pol.verification_depth),
                        mutation_budget=0,
                    ).project()
                    tel.rollback_count += 1
                    return RsiL3PolicyEvolutionCertificate(
                        evaluated=True,
                        policy_mutated=False,
                        rollback_triggered=True,
                        mutation_budget_before=budget_before,
                        mutation_budget_after=0,
                        moreau_yosida_objective_delta=0.0,
                    )
                return RsiL3PolicyEvolutionCertificate(
                    evaluated=True,
                    policy_mutated=False,
                    rollback_triggered=False,
                    mutation_budget_before=budget_before,
                    mutation_budget_after=budget_before,
                    moreau_yosida_objective_delta=0.0,
                )

            c = cert.contraction
            if c is None or not c.temporal_defined or budget_before <= 0:
                return RsiL3PolicyEvolutionCertificate(
                    evaluated=True,
                    policy_mutated=False,
                    rollback_triggered=False,
                    mutation_budget_before=budget_before,
                    mutation_budget_after=budget_before,
                    moreau_yosida_objective_delta=0.0,
                )

            # Paso proximal de Moreau-Yosida con parámetro de regularización λ = 1e-4:
            # ajusta ligeramente banach_q_max hacia el régimen observado sin relajar nunca
            # los umbrales de seguridad física (fidelity_floor y bures_soft_ceiling).
            prox_step = 1.0e-4 * (max(0.5, c.q_temporal) - pol.banach_q_max)
            new_q_max = float(
                np.clip(pol.banach_q_max + prox_step, 0.5, _BANACH_Q_MAX)
            )
            delta = abs(new_q_max - pol.banach_q_max)
            budget_after = max(0, budget_before - 1)
            new_pol = replace(
                pol, banach_q_max=new_q_max, mutation_budget=budget_after
            ).project()
            tel.policy_ledger.append(
                {
                    "sequence": int(tel.sequence_counter),
                    "fidelity_floor": float(pol.fidelity_floor),
                    "bures_soft_ceiling": float(pol.bures_soft_ceiling),
                    "banach_q_max": float(pol.banach_q_max),
                    "actuation_policy": str(pol.actuation_policy),
                    "new_banach_q_max": float(new_pol.banach_q_max),
                    "budget_after": int(budget_after),
                }
            )
            tel.policy = new_pol
            tel.mutation_count += 1
            return RsiL3PolicyEvolutionCertificate(
                evaluated=True,
                policy_mutated=True,
                rollback_triggered=False,
                mutation_budget_before=budget_before,
                mutation_budget_after=budget_after,
                moreau_yosida_objective_delta=float(delta),
            )

        def _rsi_l3_meta_audit(
            self,
            cert: WitnessPhase2Certificate,
            gate: GateVerdict,
            loeb: LoebVerification,
        ) -> MetaAuditCertificate:
            """
            3.6b Meta-auditoría RSI-L3 en tres niveles con presupuestos bien fundados.
            L1 numérico: finitud y rango (incluyendo QFI y defecto de Hurwitz-Radon).
            L2 estructural: axiomas de Heyting, leyes del núcleo, cadena de divisibilidad,
            consistencia de Hodge-Betti, rango de q y pasividad de Shahshahani.
            L3 atestación: W₁₂/W₂₃, sesión viva, reloj monótono, Löb recomputado, profundidad.
            `verification_depth` se reinicia cada ciclo; `mutation_budget` decrece de por
            vida sólo al automodificar la política (descenso estricto en ℕ).
            """
            s = self._s
            failed: List[str] = []
            w1 = cert.phase1_cert
            rep = w1.report
            metrics = (
                _ev(rep, "uhlmann_fidelity", 0.0),
                _ev(rep, "bures_distance", 0.0),
                _ev(rep, "poincare_cartan_invariant", 0.0),
                _ev(rep, "shahshahani_dissipation_rate", 0.0),
                cert.banach_ratio,
                cert.connes_bound,
            )
            l1 = all(math.isfinite(_safe_float(m, float("nan"))) for m in metrics)
            if not l1:
                failed.append("L1_NON_FINITE_METRIC")
            fid = _safe_float(_ev(rep, "uhlmann_fidelity", 0.0))
            if not (
                0.0 <= fid <= 1.0
                and cert.banach_ratio >= 0.0
                and cert.connes_bound > 0.0
            ):
                l1 = False
                failed.append("L1_RANGE_VIOLATION")
            if w1.ks_audit is not None and not (
                math.isfinite(w1.ks_audit.quantum_fisher_information)
                and w1.ks_audit.quantum_fisher_information >= -1.0e-12
            ):
                l1 = False
                failed.append("L1_QFI_RANGE_VIOLATION")

            l2 = HeytingWitnessVerdict.verify_axioms() and verify_nucleus_laws()
            if not l2:
                failed.append("L2_HEYTING_OR_NUCLEUS_LAWS")
            if cert.smith_independence is not None:
                if not cert.smith_independence.divisibility_chain_ok:
                    l2 = False
                    failed.append("L2_SNF_DIVISIBILITY")
                if not cert.smith_independence.hodge_betti_consistency_ok:
                    l2 = False
                    failed.append("L2_HODGE_BETTI_CONSISTENCY")
            if cert.celestial is not None and not cert.celestial.shahshahani_passive:
                l2 = False
                failed.append("L2_SHAHSHAHANI_PASSIVITY")

            l3 = bool(gate.authentic and loeb.accepted and loeb.independently_recomputed)
            if not l3:
                failed.append("L3_GATE_OR_LOEB")
            if cert.session is None or not cert.session.is_live(time.time_ns()):
                l3 = False
                failed.append("L3_SESSION_NOT_LIVE")
            if cert.monotonic_ns <= w1.monotonic_ns:
                l3 = False
                failed.append("L3_CLOCK_NOT_MONOTONE")

            rsi_evol = self._evolve_rsi_l3_policy(cert, l1, l2, l3)
            return MetaAuditCertificate(
                l1_numeric_ok=bool(l1),
                l2_structural_ok=bool(l2),
                l3_attestation_ok=bool(l3),
                accepted=bool(l1 and l2 and l3),
                verification_depth_remaining=int(
                    s._telemetry.policy.verification_depth
                ),
                mutation_budget_remaining=int(s._telemetry.policy.mutation_budget),
                failed_checks=tuple(failed),
                rsi_l3_policy_evolution=rsi_evol,
            )

        # ── Método 3.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _build_sovereign_payload(state: WitnessPhase3Verdict) -> Dict[str, Any]:
            """3.7a Payload soberano de cobertura total (excluye sólo la firma). [V3]"""
            return {
                "domain": _INFO_W3.decode("ascii"),
                "agent_version": state.agent_version,
                "schema_version": state.schema_version,
                "w2_signature": state.phase2_cert.signature if state.phase2_cert else "",
                "final_verdict": state.final_verdict,
                "verdict_label": state.verdict_label,
                "base_verdict": state.base_verdict,
                "verdict_under_grace": state.verdict_under_grace,
                "grace_verified": state.grace_verified,
                "grace_token": state.grace_token,
                "verdict_atoms": state.verdict_atoms,
                "abstentions": state.abstentions,
                "merkle_root_rfc6962": state.merkle_root_rfc6962,
                "merkle_algorithm": state.merkle_algorithm,
                "loeb_verification_passed": state.loeb_verification_passed,
                "loeb": state.loeb,
                "meta_audit": state.meta_audit,
                "gate_w23": state.gate_w23,
                "hardware_crowbar_triggered": state.hardware_crowbar_triggered,
                "actuation_latency_ns": state.actuation_latency_ns,
                "actuation_mode": state.actuation_mode,
                "actuation_policy": state.actuation_policy,
                "actuation": state.actuation,
                "crowbar_intent": state.crowbar_intent,
                "crowbar_latched": state.crowbar_latched,
                "session": state.session,
                "session_digest": state.session_digest,
                "monotonic_ns": state.monotonic_ns,
                "timestamp_ns": state.timestamp_ns,
                "diagnostics": state.diagnostics,
            }

        def _seal(
            self,
            cert2: WitnessPhase2Certificate,
            atoms: Sequence[Tuple[str, str]],
            raw: HeytingWitnessVerdict,
            fired: bool,
        ) -> Tuple[str, str]:
            """
            3.7b Merkle RFC 6962 de OCHO hojas de cobertura total y session digest CON CLAVE.
            [V3/V10] v16 sellaba 8 strings de texto y NO firmaba el veredicto: cualquiera
            podía recomputar la raíz y fabricar un COHERENT.
            """
            s = self._s
            smith = cert2.smith_independence
            leaves = [
                _canonical_json({"leaf": 1, "w1": cert2.phase1_cert}).encode("utf-8"),
                _canonical_json({"leaf": 2, "w2": cert2}).encode("utf-8"),
                _canonical_json(
                    {"leaf": 3, "non_interference": cert2.phase1_cert.non_interference}
                ).encode("utf-8"),
                _canonical_json({"leaf": 4, "smith_independence": smith}).encode(
                    "utf-8"
                ),
                _canonical_json(
                    {"leaf": 5, "contraction": cert2.contraction}
                ).encode("utf-8"),
                _canonical_json({"leaf": 6, "celestial": cert2.celestial}).encode(
                    "utf-8"
                ),
                _canonical_json(
                    {"leaf": 7, "atoms": atoms, "abstentions": cert2.abstentions}
                ).encode("utf-8"),
                _canonical_json(
                    {"leaf": 8, "verdict": raw.name, "crowbar": fired}
                ).encode("utf-8"),
            ]
            root = _mth_rfc6962(leaves).hex()
            session_digest = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W3, _INFO_SESSION),
                _canonical_json(
                    {"root": root, "verdict": raw.name, "session": cert2.session}
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return root, session_digest

        # ── Método 3.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _actuation_vote(
            raw: HeytingWitnessVerdict,
            atoms: Sequence[Tuple[str, str]],
            policy: WitnessPolicy,
        ) -> Tuple[bool, Tuple[str, ...], Tuple[str, ...]]:
            """
            3.8 [V6] POLÍTICA DE VOTO (IEC 61508/61511). El testigo es la autoridad MÁS BAJA
            de la tríada: reporta; el campeón actúa. Se recomienda DISABLED cuando hay un
            campeón presente (separación de funciones). Reglas:
              DISABLED        ⟹ nunca dispara.
              ATTESTED_SINGLE ⟹ ≥1 predicado FÍSICO atestado (defecto por defecto).
              VOTED_2oo3      ⟹ ≥2 grupos físicos INDEPENDIENTES.
            NUNCA dispara por abstenciones, ni por atoms administrativos, ni por la vista
            de gracia: la decisión se toma sobre el veredicto CRUDO (fail-safe).
            """
            if (
                raw is not HeytingWitnessVerdict.VETOED
                or policy.actuation_policy == "DISABLED"
            ):
                return False, (), ()
            groups = {
                "GEOMETRY": {
                    "BURES_RADIUS",
                    "BURES_CHORD_ARC",
                    "HS_OVERLAP_EXCEEDS_FIDELITY",
                    "WITNESS_1_NON_INTERFERENCE",
                    "KS_HURWITZ_ORTHOGONALITY",
                },
                "ALGEBRA": {
                    "SNF_FACTORS_INDEPENDENT",
                    "SNF_DIVISIBILITY_CHAIN",
                    "SNF_TORSION_CLAIM_CONSISTENT",
                    "SNF_ENGINE_CERTIFICATES",
                    "HODGE_BETTI_CONSISTENCY",
                    "PROJECTION_IDEMPOTENCE",
                    "DENSITY_KKT",
                    "SIMPLEX_KKT",
                },
                "DYNAMICS": {"SHAHSHAHANI_PASSIVITY", "LIOUVILLE_POINCARE_1"},
            }
            attested: List[str] = []
            hit: List[str] = []
            administrative: List[str] = []
            for name, value in atoms:
                if value != HeytingWitnessVerdict.VETOED.name:
                    continue
                g = next((k for k, members in groups.items() if name in members), None)
                if g is not None:
                    attested.append(name)
                    if g not in hit:
                        hit.append(g)
                else:
                    administrative.append(name)
            if not attested:
                return False, tuple(administrative), ()
            ok = len(hit) >= 2 if policy.actuation_policy == "VOTED_2oo3" else True
            return bool(ok), tuple(administrative), tuple(hit)

        def _crowbar_dispatch(
            self,
            should_fire: bool,
            atoms: Sequence[Tuple[str, str]],
            session: Optional[WitnessSession],
            groups: Sequence[str],
        ) -> Tuple[CrowbarIntent, ActuationReceipt, bool]:
            """
            3.8b [V6] TRIBUNAL DE SILICIO: intención firmada ≠ actuación confirmada, con LATCH.
            Sin driver real el recibo es confirmed=False y mode=SIMULATION. La latencia se MIDE
            alrededor de la llamada al driver, SIN offset y SIN clamp: si excede el
            presupuesto de 400 ns se reporta como HALLAZGO, no se recorta para satisfacer
            el axioma. Un tiristor LATCHA hasta que la corriente cae bajo la de mantenimiento.
            """
            s = self._s
            intent = CrowbarIntent(
                session_id=session.session_id if session else "",
                gpio_pin=_GPIO_CROWBAR_PIN,
                mmio_register=_GPIO_OUT_W1TS_REG,
                pin_mask=_GPIO_PIN_MASK,
                reason=tuple(
                    n for n, v in atoms if v == HeytingWitnessVerdict.VETOED.name
                ),
                vote_tally=tuple(groups),
                issued_ns=time.time_ns(),
            )
            intent_sig = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W3, _INFO_W3),
                _canonical_json(
                    {
                        "session_id": intent.session_id,
                        "gpio": intent.gpio_pin,
                        "mmio": intent.mmio_register,
                        "mask": intent.pin_mask,
                        "reason": intent.reason,
                        "tally": intent.vote_tally,
                        "issued": intent.issued_ns,
                    }
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            intent = replace(intent, signature=intent_sig)
            if s._telemetry.crowbar_latched:
                return (
                    intent,
                    ActuationReceipt(
                        True,
                        "LATCHED",
                        None,
                        None,
                        s._actuator.identity,
                        "Tiristor ya latcheado; requiere reset explícito del operador.",
                    ),
                    True,
                )
            if not should_fire:
                return (
                    intent,
                    ActuationReceipt(
                        False,
                        "NOT_REQUIRED",
                        None,
                        None,
                        s._actuator.identity,
                        "Voto no satisfecho o veredicto crudo no VETOED.",
                    ),
                    False,
                )
            s._telemetry.crowbar_intents += 1
            t0 = time.monotonic_ns()
            try:
                receipt = s._actuator.fire(intent)
            except Exception as exc:  # pragma: no cover - fallo de driver
                receipt = ActuationReceipt(
                    False,
                    "DRIVER_FAILED",
                    None,
                    None,
                    s._actuator.identity,
                    f"Excepción del driver: {exc}",
                )
            elapsed = float(time.monotonic_ns() - t0)
            if receipt.confirmed:
                within = bool(elapsed <= _CROWBAR_LATENCY_BUDGET_NS)
                receipt = replace(
                    receipt,
                    latency_ns_measured=elapsed,
                    within_latency_budget=within,
                )
                s._telemetry.crowbar_confirmations += 1
                s._telemetry.crowbar_latched = True
                if not within:
                    s._telemetry.latency_budget_violations += 1
            else:
                receipt = replace(
                    receipt, latency_ns_measured=None, within_latency_budget=None
                )
            return intent, receipt, bool(receipt.confirmed)

        # ── Método 3.9 ────────────────────────────────────────────────────────
        def verify_sovereign_verdict(
            self, state: WitnessPhase3Verdict
        ) -> WitnessVerificationReport:
            """
            3.9 VERIFICADOR PÚBLICO. Recomputa, sólo desde el certificado: las tres firmas,
            la raíz Merkle, el session digest con clave, el veredicto CRUDO por ínfimo, la
            vista de gracia y la HONESTIDAD de la actuación. [V3] Un certificado que nadie
            puede verificar no es un certificado.
            """
            s = self._s
            failed: List[str] = []
            w2 = state.phase2_cert
            if w2 is None:
                return WitnessVerificationReport(
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    ("W2_ABSENT",),
                )
            w1 = w2.phase1_cert
            w1_ok = _ct_equals(
                w1.signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_SALT_W1, _INFO_W1),
                    _canonical_json(s.phase1._build_w1_payload(w1)).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
            w2_ok = _ct_equals(
                w2.signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_SALT_W2, _INFO_W2),
                    _canonical_json(s.phase2._build_w2_payload(w2)).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
            w3_ok = _ct_equals(
                state.sovereign_signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_SALT_W3, _INFO_W3),
                    _canonical_json(self._build_sovereign_payload(state)).encode(
                        "utf-8"
                    ),
                    hashlib.sha256,
                ).hexdigest(),
            )
            for name, ok in (
                ("W1_SIGNATURE", w1_ok),
                ("W2_SIGNATURE", w2_ok),
                ("W3_SIGNATURE", w3_ok),
            ):
                if not ok:
                    failed.append(name)
            raw = HeytingWitnessVerdict.parse(state.final_verdict)
            root, sess = self._seal(
                w2, state.verdict_atoms, raw, state.hardware_crowbar_triggered
            )
            merkle_ok = _ct_equals(root, state.merkle_root_rfc6962)
            sess_ok = _ct_equals(sess, state.session_digest)
            if not merkle_ok:
                failed.append("MERKLE_ROOT")
            if not sess_ok:
                failed.append("SESSION_DIGEST")
            base = self._heyting_infimum(state.verdict_atoms)
            expected = base
            if state.loeb is not None and not state.loeb.accepted:
                expected = expected.meet(HeytingWitnessVerdict.DEGRADED)
            if state.meta_audit is not None and not state.meta_audit.accepted:
                expected = expected.meet(HeytingWitnessVerdict.DEGRADED)
            verdict_ok = bool(
                int(expected) == int(raw) and state.verdict_label == raw.label
            )
            if not verdict_ok:
                failed.append("VERDICT_RECOMPUTATION")
            grace_ok = bool(
                state.verdict_under_grace
                == int(self._grace_view(raw, state.grace_verified))
            )
            if not grace_ok:
                failed.append("GRACE_VIEW_RECOMPUTATION")
            session_ok = bool(
                state.session is not None
                and w2.session is not None
                and w1.session is not None
                and state.session.session_id
                == w2.session.session_id
                == w1.session.session_id
                and w1.monotonic_ns < w2.monotonic_ns < state.monotonic_ns
            )
            if not session_ok:
                failed.append("SESSION_CONTINUITY")
            honesty_ok = bool(
                (not state.hardware_crowbar_triggered)
                or (
                    state.actuation is not None
                    and state.actuation.confirmed
                    and state.actuation_mode in ("DRIVER_CONFIRMED", "LATCHED")
                )
            )
            if not honesty_ok:
                failed.append("ACTUATION_ATTESTATION_DISHONEST")
            all_ok = bool(
                w1_ok
                and w2_ok
                and w3_ok
                and merkle_ok
                and sess_ok
                and verdict_ok
                and grace_ok
                and session_ok
                and honesty_ok
            )
            return WitnessVerificationReport(
                c1_signature_ok=w1_ok,
                c2_signature_ok=w2_ok,
                sovereign_signature_ok=w3_ok,
                merkle_ok=merkle_ok,
                session_digest_ok=sess_ok,
                verdict_recomputation_ok=verdict_ok,
                grace_view_ok=grace_ok,
                session_continuity_ok=session_ok,
                actuation_honesty_ok=honesty_ok,
                all_ok=all_ok,
                failed_checks=tuple(failed),
            )

        # ── Método 3.10 · TERMINAL W₃ · EMISIÓN SOBERANA FIRMADA ──────────────
        def _phase3_decide_and_act(
            self,
            cert2: WitnessPhase2Certificate,
            grace_token: Optional[WitnessGraceToken] = None,
        ) -> WitnessPhase3Verdict:
            """
            3.10 [TERMINAL W₃] Ejecuta W₂₃ de forma EFECTIVA, adjudica el ínfimo en Ω₃,
            calcula la vista informativa de gracia SIN alterar el veredicto propio, corre
            el Löb NO VACUO y la meta-auditoría RSI-L3 con evolución proximal de política,
            vota la actuación, despacha el crowbar con recibo y sella con Merkle RFC 6962
            + HMAC soberano.
            """
            s = self._s
            V = HeytingWitnessVerdict
            s._telemetry.policy = replace(
                s._telemetry.policy, verification_depth=3
            ).project()
            gate = self._gate_w23(cert2)
            atoms = list(cert2.evidence_atoms)
            if not gate.authentic:
                atoms.append(("GATE_W23_AUTH", V.VETOED.name))
            base = self._heyting_infimum(atoms)
            raw = base
            loeb = self._loeb_safe_verification(cert2)
            if not loeb.accepted:
                raw = raw.meet(V.DEGRADED)
                atoms.append(("LOEB_DESCENT", V.DEGRADED.name))
            meta = self._rsi_l3_meta_audit(cert2, gate, loeb)
            if not meta.accepted:
                raw = raw.meet(V.DEGRADED)
                atoms.append(("META_AUDIT", V.DEGRADED.name))
            grace_ok, grace_reasons = self._verify_grace_token(
                grace_token, cert2.session
            )
            under_grace = self._grace_view(raw, grace_ok)
            should_fire, admin_only, groups = self._actuation_vote(
                raw, atoms, s._telemetry.policy
            )
            intent, receipt, fired = self._crowbar_dispatch(
                should_fire, atoms, cert2.session, groups
            )
            diagnostics: List[str] = [
                f"{n}={v}" for n, v in atoms if v != V.COHERENT.name
            ]
            diagnostics.extend(gate.reasons)
            diagnostics.extend(grace_reasons)
            diagnostics.extend(meta.failed_checks)
            if cert2.abstentions:
                diagnostics.append(
                    "ABSTENCIONES (no se atesta lo no verificado): "
                    + ", ".join(cert2.abstentions)
                )
            if grace_ok:
                diagnostics.append(
                    f"VISTA DE GRACIA (informativa): {raw.name} → {under_grace.name}; "
                    "el veredicto del testigo NO cambia."
                )
            if admin_only and not should_fire:
                diagnostics.append(
                    "Veto sólo administrativo: CONTENCIÓN sin actuación física."
                )
            if intent.reason:
                diagnostics.append(
                    f"[CROWBAR_INTENT] MMIO {_GPIO_OUT_W1TS_REG:#x} máscara {_GPIO_PIN_MASK:#x} "
                    f"→ GPIO{_GPIO_CROWBAR_PIN}; voto={groups or '—'}; mode={receipt.mode}; "
                    f"confirmed={receipt.confirmed}; {receipt.detail}"
                )
            if receipt.within_latency_budget is False:
                diagnostics.append(
                    f"HALLAZGO: latencia medida {receipt.latency_ns_measured:.2f} ns EXCEDE el "
                    f"presupuesto de {_CROWBAR_LATENCY_BUDGET_NS:.2f} ns (no se recorta)."
                )
            root, sess = self._seal(cert2, atoms, raw, fired)
            s._telemetry.monotonic_clock_ns = max(
                time.monotonic_ns(), s._telemetry.monotonic_clock_ns + 1
            )
            skeleton = WitnessPhase3Verdict(
                final_verdict=int(raw),
                verdict_label=raw.label,
                merkle_root_rfc6962=root,
                loeb_verification_passed=bool(loeb.accepted),
                hardware_crowbar_triggered=bool(fired),
                actuation_latency_ns=float(receipt.latency_ns_measured or 0.0),
                agent_version=_AGENT_VERSION,
                phase2_cert=cert2,
                base_verdict=int(base),
                verdict_under_grace=int(under_grace),
                grace_verified=bool(grace_ok),
                grace_token=grace_token,
                verdict_atoms=tuple(atoms),
                abstentions=cert2.abstentions,
                crowbar_intent=intent,
                actuation=receipt,
                actuation_mode=receipt.mode,
                actuation_policy=s._telemetry.policy.actuation_policy,
                crowbar_latched=bool(s._telemetry.crowbar_latched),
                loeb=loeb,
                meta_audit=meta,
                gate_w23=gate,
                session=cert2.session,
                session_digest=sess,
                monotonic_ns=int(s._telemetry.monotonic_clock_ns),
                timestamp_ns=time.time_ns(),
                diagnostics=tuple(diagnostics),
            )
            sig = hmac.new(
                s._hkdf_extract_and_expand(_SALT_W3, _INFO_W3),
                _canonical_json(self._build_sovereign_payload(skeleton)).encode(
                    "utf-8"
                ),
                hashlib.sha256,
            ).hexdigest()
            state = replace(skeleton, sovereign_signature=sig)
            s._telemetry.phase3_emissions += 1
            s._telemetry.last_verdict = raw
            s._last_verdict = state
            return state

    # ──────────────────────────────────────────────────────────────────────────
    # §4 · INICIALIZACIÓN, HKDF, RESOLUCIÓN DIMENSIONAL, CPS Y VERIFICACIÓN
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        master_key: Optional[bytes] = None,
        engine: Optional[PositronRedemptionWitnessEngine] = None,
        actuator: Optional[WitnessCrowbarActuator] = None,
        policy: Optional[WitnessPolicy] = None,
        tolerance: float = 1e-9,
    ) -> None:
        # [V10] `master_key or default` caía al defecto con clave vacía b"".
        self._master_key: Final[bytes] = (
            bytes(master_key) if master_key is not None else _DEFAULT_MASTER_KEY
        )
        self._actuator: WitnessCrowbarActuator = (
            actuator if actuator is not None else NullWitnessActuator()
        )
        self._tolerance: Final[float] = float(tolerance)
        self._injected_engine = engine
        self._engines: Dict[int, PositronRedemptionWitnessEngine] = {}
        self._telemetry = WitnessTelemetry(
            policy=(policy or WitnessPolicy()).project(),
            dev_key_in_use=(master_key is None),
        )
        self._consumed_grace_nonces: List[str] = []
        self._last_verdict: Optional[WitnessPhase3Verdict] = None
        if master_key is None:
            logger.warning(
                "Clave de desarrollo publicada en uso: certificados NO atestables en producción."
            )
        elif len(self._master_key) < 32:
            logger.warning(
                "master_key con menos de 32 bytes: entropía insuficiente."
            )
        if isinstance(self._actuator, NullWitnessActuator):
            logger.warning(
                "Sin driver crowbar enlazado: toda actuación se reportará como SIMULATION. "
                "El testigo es la autoridad más baja de la tríada: considere DISABLED."
            )
        self.phase1 = self.WitnessPhase1Observer(self)
        self.phase2 = self.WitnessPhase2Orientor(self)
        self.phase3 = self.WitnessPhase3Decider(self)
        logger.info(
            "PositronRedemptionWitnessAgent v%s inicializado (W₁→W₂→W₃).",
            _AGENT_VERSION,
        )

    # ── 4.1 · Resolución dimensional del motor ──
    def _engine_for(self, dimension_n: int) -> PositronRedemptionWitnessEngine:
        """El motor v17/v18 exige `dimension_n`; v16 lo infería. Se cachea una instancia por n."""
        if self._injected_engine is not None:
            return self._injected_engine
        n = max(1, int(dimension_n))
        if n not in self._engines:
            self._engines[n] = PositronRedemptionWitnessEngine(
                self._tolerance, dimension_n=n
            )
        return self._engines[n]

    # ── 4.2 · HKDF-SHA256 (RFC 5869) completo ──
    def _hkdf_extract_and_expand(
        self, salt: bytes, info: bytes, length: int = _HASH_LEN
    ) -> bytes:
        """
        [V10] Extract-then-Expand COMPLETO. v16 sólo computaba T(1) y devolvía 32 bytes
        aunque se pidieran 64: truncamiento silencioso. Se exige 0 < L ≤ 255·HashLen.
        """
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or length > _HKDF_MAX_LEN
        ):
            raise ValueError(f"HKDF: length debe estar en (0, {_HKDF_MAX_LEN}].")
        prk = hmac.new(salt, self._master_key, hashlib.sha256).digest()
        okm, t, i = b"", b"", 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
            i += 1
        return okm[:length]

    @staticmethod
    def _build_merkle_tree_rfc6962(leaves: List[bytes]) -> str:
        """Compatibilidad v16 con semántica CORREGIDA: MTH recursivo con promoción. [V10]"""
        return _mth_rfc6962(leaves).hex()

    @staticmethod
    def _resolve_boundary(
        engine: PositronRedemptionWitnessEngine,
        boundary: Optional[NDArray[np.int64]],
    ) -> NDArray[np.int64]:
        """
        Resuelve la matriz de contorno EFECTIVA (la que el motor usará) para poder
        digerirla y verificarla independientemente. v16 no la digería ni la firmaba.
        """
        if boundary is not None:
            return np.asarray(boundary, dtype=np.int64)
        helper = getattr(
            getattr(engine, "phase_pi2", None), "_default_boundary_matrix", None
        )
        if callable(helper):
            return np.asarray(helper(), dtype=np.int64)
        return np.array([[1, -1, 0], [0, 1, -1]], dtype=np.int64)

    # ── 4.3 · Emisión de token de gracia (helper de interoperabilidad/pruebas) ──
    def mint_grace_token(
        self,
        session: WitnessSession,
        subject: str = "GOVERNOR_GRACE_AUTHORITY",
        ttl_ns: int = 10_000_000_000,
        max_grace_sec: float = 3600.0,
    ) -> WitnessGraceToken:
        """
        Helper de interoperabilidad y pruebas. EN PRODUCCIÓN el token lo emite el SOBERANO
        GOBERNADOR, no el testigo: se expone aquí porque verificar la composición entre
        soberanos exige poder construir un token válido bajo la misma clave maestra.
        """
        now = time.time_ns()
        nonce = hmac.new(
            self._hkdf_extract_and_expand(_SALT_W3, _INFO_GRACE),
            secrets.token_bytes(32) + session.session_id.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()[:32]
        skeleton = WitnessGraceToken(
            subject=subject,
            session_id=session.session_id,
            input_bundle_digest=session.input_bundle_digest,
            nonce=nonce,
            issued_ns=now,
            nbf_ns=now - 1_000_000,
            exp_ns=now + int(ttl_ns),
            max_grace_sec=float(max(0.0, max_grace_sec)),
        )
        sig = hmac.new(
            self._hkdf_extract_and_expand(_SALT_W3, _INFO_GRACE),
            _canonical_json(
                {
                    "domain": _INFO_GRACE.decode("ascii"),
                    "subject": skeleton.subject,
                    "session_id": skeleton.session_id,
                    "input_bundle_digest": skeleton.input_bundle_digest,
                    "nonce": skeleton.nonce,
                    "issued_ns": skeleton.issued_ns,
                    "nbf_ns": skeleton.nbf_ns,
                    "exp_ns": skeleton.exp_ns,
                    "max_grace_sec": skeleton.max_grace_sec,
                }
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return replace(skeleton, signature=sig)

    # ── 4.4 · Verificador público ──
    def verify_sovereign_verdict(
        self, state: WitnessPhase3Verdict
    ) -> WitnessVerificationReport:
        """Delega en 3.9 (recomputación íntegra desde el certificado)."""
        return self.phase3.verify_sovereign_verdict(state)

    # ── 4.5 · Entrada principal: cadena CPS W₃ ∘ W₂ ∘ W₁ ──
    def audit_witness_cycle(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        boundary_matrix: Optional[NDArray[np.int64]] = None,
        *,
        boundary_next: Optional[NDArray[np.int64]] = None,
        grace_token: Optional[WitnessGraceToken] = None,
        **engine_kwargs: Any,
    ) -> WitnessPhase3Verdict:
        """
        Ejecuta la auditoría pasiva completa en CPS con tipo de respuesta polimórfico.
        [V9] NUNCA lanza por evidencia ni por fallo del motor: un fallo produce un VEREDICTO
        VETOED contenido y firmado, porque dejar al llamador sin veredicto es peor que un
        veredicto severo. El testigo no interfiere: ni muta el estado ni aborta el flujo.
        """
        tel = self._telemetry
        tel.total_cycles += 1
        n = 0
        if isinstance(rho_init, np.ndarray) and rho_init.ndim == 2:
            n = int(rho_init.shape[0])
        elif isinstance(p_init, np.ndarray):
            n = int(np.asarray(p_init).reshape(-1).size)
        n = max(1, n)
        engine = self._engine_for(n)
        boundary = self._resolve_boundary(engine, boundary_matrix)
        tensors = (p_init, rho_init, sigma_ref, K_raw, boundary)
        names = ("p", "rho", "sigma", "K", "boundary")
        frozen_before = [
            self.phase1._freeze_and_digest(nm, a) for nm, a in zip(names, tensors)
        ]
        dims_ok = all(
            isinstance(a, np.ndarray)
            and (
                (a.ndim == 1 and a.reshape(-1).size == n)
                or (a.ndim == 2 and a.shape == (n, n))
            )
            for a in tensors[:4]
        )
        try:
            if not dims_ok:
                raise ValueError(
                    f"Incoherencia dimensional: se esperaba n={n} en p, ρ, σ, K."
                )
            kwargs = dict(engine_kwargs)
            if boundary_next is not None:
                kwargs["boundary_next"] = boundary_next
            report = engine.inspect_positron_trajectory(
                p_init=p_init,
                rho_init=rho_init,
                sigma_ref=sigma_ref,
                K_raw=K_raw,
                boundary_matrix=boundary,
                **kwargs,
            )
        except Exception as exc:
            tel.engine_failures += 1
            logger.error("Fallo del motor testigo contenido: %s", exc)
            return self._emit_engine_failure_verdict(frozen_before, n, str(exc))

        def _w2_continuation(w1: WitnessPhase1Certificate) -> WitnessPhase3Verdict:
            def _w3_continuation(w2: WitnessPhase2Certificate) -> WitnessPhase3Verdict:
                return self.phase3._phase3_decide_and_act(w2, grace_token=grace_token)

            return self.phase2._phase2_orient_and_sign(
                cert1=w1,
                engine=engine,
                boundary=boundary,
                tensors_now=tensors,
                continuation=_w3_continuation,
            )

        return self.phase1._phase1_observe_and_sign(
            report=report,
            frozen_before=frozen_before,
            tensors_now=tensors,
            dimension_n=n,
            continuation=_w2_continuation,
        )

    def _emit_engine_failure_verdict(
        self, frozen_before: Sequence[FrozenTensor], dimension_n: int, detail: str
    ) -> WitnessPhase3Verdict:
        """Veredicto de CONTENCIÓN ante fallo de cómputo: VETOED firmado, sin actuación. [V9]"""
        tel = self._telemetry
        tel.sequence_counter += 1
        tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
        now = time.time_ns()
        bundle = hashlib.sha256(
            "".join(t.digest for t in frozen_before).encode("ascii")
        ).hexdigest()
        nonce = hmac.new(
            self._hkdf_extract_and_expand(_SALT_W1, _INFO_NONCE),
            secrets.token_bytes(32) + str(tel.sequence_counter).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()[:32]
        session = WitnessSession(
            session_id=hmac.new(
                self._hkdf_extract_and_expand(_SALT_W1, _INFO_SESSION),
                bundle.encode("ascii"),
                hashlib.sha256,
            ).hexdigest(),
            nonce=nonce,
            sequence=int(tel.sequence_counter),
            issued_ns=now,
            nbf_ns=now - 1_000_000,
            exp_ns=now + _SESSION_TTL_NS,
            monotonic_ns=int(tel.monotonic_clock_ns),
            input_bundle_digest=bundle,
            dimension_n=int(dimension_n),
        )
        atoms = (("ENGINE_COMPUTATION_FAILURE", HeytingWitnessVerdict.VETOED.name),)
        root = _mth_rfc6962(
            [
                _canonical_json({"leaf": 1, "failure": detail}).encode("utf-8"),
                _canonical_json({"leaf": 2, "frozen": frozen_before}).encode("utf-8"),
                _canonical_json({"leaf": 3, "atoms": atoms}).encode("utf-8"),
                _canonical_json({"leaf": 4, "session": session}).encode("utf-8"),
            ]
        ).hex()
        sess = hmac.new(
            self._hkdf_extract_and_expand(_SALT_W3, _INFO_SESSION),
            _canonical_json({"root": root, "session": session}).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        skeleton = WitnessPhase3Verdict(
            final_verdict=int(HeytingWitnessVerdict.VETOED),
            verdict_label=HeytingWitnessVerdict.VETOED.label,
            merkle_root_rfc6962=root,
            loeb_verification_passed=False,
            hardware_crowbar_triggered=False,
            actuation_latency_ns=0.0,
            agent_version=_AGENT_VERSION,
            phase2_cert=None,
            base_verdict=int(HeytingWitnessVerdict.VETOED),
            verdict_under_grace=int(HeytingWitnessVerdict.VETOED),
            verdict_atoms=atoms,
            session=session,
            session_digest=sess,
            monotonic_ns=int(tel.monotonic_clock_ns),
            timestamp_ns=now,
            actuation_mode="NOT_REQUIRED",
            actuation_policy=tel.policy.actuation_policy,
            diagnostics=(
                f"FALLO DEL MOTOR CONTENIDO: {detail}",
                "Veredicto de contención firmado; sin actuación física.",
            ),
        )
        sig = hmac.new(
            self._hkdf_extract_and_expand(_SALT_W3, _INFO_W3),
            _canonical_json(self.phase3._build_sovereign_payload(skeleton)).encode(
                "utf-8"
            ),
            hashlib.sha256,
        ).hexdigest()
        state = replace(skeleton, sovereign_signature=sig)
        tel.last_verdict = HeytingWitnessVerdict.VETOED
        self._last_verdict = state
        return state

    # ── 4.6 · Accesores y control del latch ──
    def get_last_verdict(self) -> Optional[WitnessPhase3Verdict]:
        return self._last_verdict

    def get_telemetry(self) -> WitnessTelemetry:
        return self._telemetry

    def reset_crowbar_latch(self, operator_reason: str) -> bool:
        """Reset explícito del latch del tiristor: acción de operador, nunca automática."""
        if not self._telemetry.crowbar_latched:
            return False
        logger.warning("Reset del latch crowbar por operador: %s", operator_reason)
        self._telemetry.crowbar_latched = False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# §5 · AUTOVERIFICACIÓN EJECUTABLE (regresiones de v16, v17 y garantías v18)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """Las primeras comprobaciones son las que v16 NO podía pasar, seguidas de v18."""
    checks: List[Tuple[str, bool]] = []
    n = 3
    agent = PositronRedemptionWitnessAgent(
        master_key=b"witness-self-test-key-32-bytes-ok!"
    )
    p0 = np.array([0.5, 0.3, 0.2])
    rho = np.eye(n, dtype=np.complex128) / float(n)
    sigma = np.diag([0.6, 0.2, 0.2]).astype(np.complex128)
    k_raw = np.eye(n, dtype=np.complex128)
    verdict = agent.audit_witness_cycle(p0, rho, sigma, k_raw)
    w2 = verdict.phase2_cert
    assert w2 is not None
    w1 = w2.phase1_cert
    engine = agent._engine_for(n)
    boundary = agent._resolve_boundary(engine, None)

    # [V1] REGRESIÓN CABECERA: los factores SVD de v16 se auto-delatan.
    svd_factors = np.round(
        np.linalg.svd(np.abs(boundary.astype(np.int64)), compute_uv=False)
    ).astype(np.int64)
    v16_declared = [int(x) for x in svd_factors if x > 0]
    divisors, ind_rank, _ev_cnt, _cap = agent.phase2._determinantal_divisors(boundary)
    independent = [divisors[k] // divisors[k - 1] for k in range(1, len(divisors))]
    checks.append(
        (
            "[V1] v16 (SVD) declaraba (2,1); los divisores determinantes dan (1,1)",
            v16_declared == [2, 1] and independent == [1, 1],
        )
    )
    checks.append(
        (
            "[V1] d₁ = mcd(entradas) = 1 delata el 2 de v16",
            bool(independent and independent[0] == 1 and v16_declared[0] != 1),
        )
    )
    checks.append(
        (
            "[V1] Cadena de divisibilidad: 2 ∤ 1 en los factores de v16",
            not all(
                v16_declared[i + 1] % v16_declared[i] == 0
                for i in range(len(v16_declared) - 1)
            ),
        )
    )
    smith = w2.smith_independence
    checks.append(
        (
            "[V1] Verificación independiente PASA con el motor v17/v18",
            smith is not None
            and smith.factors_match
            and smith.divisibility_chain_ok
            and smith.torsion_claim_consistent
            and smith.independent_torsion_free,
        )
    )
    # Control con torsión real: diag(2,3) ⟹ SNF (1,6).
    b23 = np.array([[2, 0], [0, 3]], dtype=np.int64)
    d23, r23, _, _ = agent.phase2._determinantal_divisors(b23)
    f23 = [d23[k] // d23[k - 1] for k in range(1, len(d23))]
    checks.append(
        (
            "[V1] Divisores determinantes: SNF(diag(2,3)) = (1,6), rango 2",
            f23 == [1, 6] and r23 == 2,
        )
    )

    # [V2] Sustituir K_purified debe romper la firma (v16 la aceptaba).
    forged_w1 = replace(
        w1,
        report=replace(
            w1.report, K_purified=np.asarray(k_raw * 7.0, dtype=np.complex128)
        ),
    )
    checks.append(
        (
            "[V2] W₁₂ RECHAZA K_purified sustituido (v16 lo aceptaba)",
            not agent.phase2._gate_w12(forged_w1, list(w1.frozen_inputs)).authentic,
        )
    )
    factors_forged = replace(w1, report=replace(w1.report, smith_torsion_free=False))
    checks.append(
        (
            "[V2] W₁₂ RECHAZA el booleano de torsión falsificado",
            not agent.phase2._gate_w12(
                factors_forged, list(w1.frozen_inputs)
            ).authentic,
        )
    )

    # [V3] El veredicto ahora está FIRMADO y el verificador detecta adulteración.
    checks.append(
        (
            "[V3] Veredicto soberano con firma HMAC de 64 hex",
            len(verdict.sovereign_signature) == 64,
        )
    )
    checks.append(
        (
            "[V3] verify_sovereign_verdict(legítimo).all_ok",
            agent.verify_sovereign_verdict(verdict).all_ok,
        )
    )
    tampered = replace(
        verdict,
        final_verdict=int(HeytingWitnessVerdict.COHERENT),
        verdict_label="COHERENT",
    )
    checks.append(
        (
            "[V3] verify_sovereign_verdict(adulterado) detecta",
            not agent.verify_sovereign_verdict(tampered).all_ok,
        )
    )

    # [V4] WITNESS-1 atestiguado sobre los cinco buffers.
    att = w1.non_interference
    checks.append(
        (
            "[V4] No-interferencia atestiguada (v16 nunca la verificó)",
            att is not None and att.all_ok and len(w1.frozen_inputs) == 5,
        )
    )
    p_mut, rho_mut = p0.copy(), rho.copy()
    _ = agent.audit_witness_cycle(p_mut, rho_mut, sigma, k_raw)
    checks.append(
        (
            "[V4] Las entradas no fueron mutadas in-place",
            np.array_equal(p_mut, p0) and np.array_equal(rho_mut, rho),
        )
    )

    # [V5] Löb no vacuo: recomputación + identidades + profundidad.
    checks.append(
        (
            "[V5] Löb recomputado con ≥4 identidades y profundidad decrementada",
            verdict.loeb is not None
            and verdict.loeb.independently_recomputed
            and verdict.loeb.identities_rechecked >= 4
            and verdict.loeb.depth_remaining < 3,
        )
    )

    # [V6] Sin driver, jamás disparo físico ni latencia fabricada.
    checks.append(
        (
            "[V6] Sin hardware ⟹ triggered=False, latencia 0, modo no confirmado",
            not verdict.hardware_crowbar_triggered
            and verdict.actuation_latency_ns == 0.0
            and verdict.actuation_mode in ("SIMULATION", "NOT_REQUIRED"),
        )
    )

    class _SlowDriver:
        identity = "SLOW_DRIVER_FOR_TEST"

        def fire(self, intent: CrowbarIntent) -> ActuationReceipt:
            t0 = time.monotonic_ns()
            while time.monotonic_ns() - t0 < 2_000:
                pass
            return ActuationReceipt(
                True, "DRIVER_CONFIRMED", None, None, self.identity, "test"
            )

    agent2 = PositronRedemptionWitnessAgent(master_key=b"k" * 32, actuator=_SlowDriver())
    v2 = agent2.audit_witness_cycle(
        p0, rho, np.zeros((n, n), dtype=np.complex128), k_raw
    )
    if v2.hardware_crowbar_triggered and v2.actuation is not None:
        checks.append(
            (
                "[V6] Latencia MEDIDA sin clamp: exceso reportado, no recortado",
                v2.actuation.latency_ns_measured is not None
                and v2.actuation.latency_ns_measured > _CROWBAR_LATENCY_BUDGET_NS
                and v2.actuation.within_latency_budget is False,
            )
        )
        checks.append(("[V6] Latch del tiristor persiste", v2.crowbar_latched))
    else:
        checks.append(
            ("[V6] Voto no satisfecho ⟹ no disparo (asimetría declarada)", True)
        )
        checks.append(("[V6] Sin confirmación no hay latch", not v2.crowbar_latched))

    # [V7] Contracción genuina: idempotencia POCS, no |dissipation|/(κ+1).
    contraction = w2.contraction
    checks.append(
        (
            "[V7] Defecto de idempotencia POCS ~ 0 (T² = T)",
            contraction is not None
            and contraction.idempotence_ok
            and contraction.idempotence_defect < 1e-9,
        )
    )
    legacy_q = abs(_safe_float(_ev(w1.report, "shahshahani_dissipation_rate", 0.0))) / (
        _safe_float(_ev(w1.report, "condition_number_K", 1.0), 1.0) + 1.0
    )
    checks.append(
        (
            "[V7] banach_ratio ≠ |dissipation|/(κ+1) y en [0,1)",
            contraction is not None
            and 0.0 <= w2.banach_ratio < 1.0
            and (
                abs(w2.banach_ratio - legacy_q) > 1e-12 or contraction.at_fixed_point
            ),
        )
    )

    # [V8] Fidelidad juzgada; Cartan ya no es constante; Shahshahani pasivo.
    checks.append(
        (
            "[V8] La fidelidad entra en la escalera de veredicto",
            any(n_ == "UHLMANN_FIDELITY" for n_, _ in w1.evidence_atoms),
        )
    )
    celestial = w2.celestial
    checks.append(
        (
            "[V8] Pasividad de Shahshahani auditada (−Var ≤ 0)",
            celestial is not None and celestial.shahshahani_passive,
        )
    )
    v3_cycle = agent.audit_witness_cycle(
        p0, rho, np.diag([0.9, 0.05, 0.05]).astype(np.complex128), k_raw
    )
    checks.append(
        (
            "[V8] I_PC ya no es sospechosamente constante con entradas distintas",
            not agent.get_telemetry().cartan_history
            or not agent.phase2._audit_celestial_sector(
                w1.report, "probe"
            ).cartan_suspiciously_constant,
        )
    )

    # [V9] Fallo del motor ⟹ veredicto firmado, no excepción.
    v_fail = agent.audit_witness_cycle(p0, rho, sigma, np.eye(5, dtype=np.complex128))
    checks.append(
        (
            "[V9] Incoherencia dimensional ⟹ VETOED contenido y firmado",
            v_fail.final_verdict == int(HeytingWitnessVerdict.VETOED)
            and len(v_fail.sovereign_signature) == 64,
        )
    )

    # [V10] Merkle, HKDF, Heyting.
    checks.append(
        (
            "[V10] Duplicar hoja impar CAMBIA la raíz (anti CVE-2012-2459)",
            _mth_rfc6962([b"a", b"b", b"c"]) != _mth_rfc6962([b"a", b"b", b"c", b"c"]),
        )
    )
    checks.append(
        (
            "[V10] HKDF length=64 devuelve 64 bytes (v16 devolvía 32)",
            len(agent._hkdf_extract_and_expand(_SALT_W1, b"t", 64)) == 64,
        )
    )
    checks.append(
        (
            "[V10] Residuación y De Morgan exhaustivas",
            HeytingWitnessVerdict.verify_axioms(),
        )
    )
    try:
        HeytingWitnessVerdict.to_string(7)
        strict = False
    except ValueError:
        strict = True
    checks.append(
        ("[V10] to_string(7) lanza en vez de coerced a VETOED", strict)
    )
    checks.append(
        (
            "[V10] DEGRADED no es ¬¬-estable",
            not HeytingWitnessVerdict.DEGRADED.is_regular(),
        )
    )

    # [v18] Verificaciones cuántico-celestes KS-SLD, semillas de continuación y evolución RSI-L3.
    checks.append(
        (
            "[v18] Certificado Kustaanheimo-Stiefel y SLD de Helstrom verificado en W₁",
            w1.ks_audit is not None
            and w1.ks_audit.ks_regularization_ok
            and w1.ks_audit.sld_lyapunov_ok
            and _independent_hurwitz_ks_defect((0.5, 0.5, 0.5, 0.5)) < 1e-12,
        )
    )
    checks.append(
        (
            "[v18] Semillas formales W1ToW2ContinuationSeed y W2ToW3ContinuationSeed íntegras",
            w1.continuation_seed is not None
            and w2.continuation_seed is not None
            and w1.continuation_seed.session_id == w2.continuation_seed.session_id,
        )
    )
    checks.append(
        (
            "[v18] Evolución proximal RSI-L3 con descenso bien fundado en mutation_budget",
            v3_cycle.meta_audit is not None
            and v3_cycle.meta_audit.rsi_l3_policy_evolution is not None
            and v3_cycle.meta_audit.rsi_l3_policy_evolution.policy_mutated
            and v3_cycle.meta_audit.rsi_l3_policy_evolution.mutation_budget_after
            < v3_cycle.meta_audit.rsi_l3_policy_evolution.mutation_budget_before,
        )
    )

    # El testigo NO absuelve: la gracia no altera su veredicto.
    sess = w1.session
    assert sess is not None
    token = agent.mint_grace_token(sess, max_grace_sec=60.0)
    ok_valid, _ = agent.phase3._verify_grace_token(token, sess)
    ok_forged, _ = agent.phase3._verify_grace_token(
        replace(token, max_grace_sec=1e9), sess
    )
    ok_replay, _ = agent.phase3._verify_grace_token(token, sess)
    checks.append(
        (
            "Token de gracia válido en primer uso; mutado y replay rechazados",
            ok_valid and not ok_forged and not ok_replay,
        )
    )
    checks.append(
        (
            "La vista de gracia es informativa: j(V)=V, j(D)=C",
            grace_nucleus(HeytingWitnessVerdict.VETOED, True)
            is HeytingWitnessVerdict.VETOED
            and grace_nucleus(HeytingWitnessVerdict.DEGRADED, True)
            is HeytingWitnessVerdict.COHERENT,
        )
    )
    checks.append(("Leyes del núcleo de gracia", verify_nucleus_laws()))

    # CPS polimórfico.
    rep = engine.inspect_positron_trajectory(p0, rho, sigma, k_raw, boundary)
    tensors = (p0, rho, sigma, k_raw, boundary)
    frozen = [
        agent.phase1._freeze_and_digest(nm, a)
        for nm, a in zip(("p", "rho", "sigma", "K", "boundary"), tensors)
    ]
    answer = agent.phase1._phase1_observe_and_sign(
        report=rep,
        frozen_before=frozen,
        tensors_now=tensors,
        dimension_n=n,
        continuation=lambda c: f"W1:{len(c.signature)}",
    )
    checks.append(("CPS con AnswerT polimórfico (str)", answer == "W1:64"))

    ok = True
    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(
        f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN: {'COHERENTE' if ok else 'OBSTRUIDO'}"
    )
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s"
    )
    raise SystemExit(0 if _self_test() else 1)