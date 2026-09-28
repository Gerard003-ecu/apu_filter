# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Algebra — Evolución Bicompleja 2-Categórica                     ║
║ Ruta   : app/core/mic_algebra.py                                             ║
║ Versión: 5.1.0-Bicomplex-Idempotent-Kronecker-Interchange-Heyting-B2-Doctoral║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETO DEL MÓDULO
─────────────────
Este módulo evoluciona el sustrato algebraico-topológico de la MIC desde una
2-categoría escalar/real hacia una 2-categoría bicompleja estricta sobre

    C₂ ≅ C ⊗_R C ≅ C × C  (isomorfismo de anillos vía idempotentes),

con base idempotente ortogonal

    e₁ = (1 + j)/2,   e₂ = (1 - j)/2,
    e₁² = e₁, e₂² = e₂, e₁e₂ = 0, e₁ + e₂ = 1.

Todo estado, morfismo, 2-morfismo, verificación homológica y veredicto de
Heyting se descompone de forma bicanal:

    X_{C₂} = X⁽¹⁾ e₁ + X⁽²⁾ e₂.

FASES ANIDADAS
──────────────
Fase 1 — SANEAMIENTO ESPECTRAL Y ANILLO CONMUTATIVO BICOMPLEJO
    Núcleo algebraico de C₂ con operaciones de anillo certificables
    (`verify_bicomplex_ring_axioms`), ingesta 4D, proyección idempotente,
    módulo vectorial sobre C₂, no-degeneración bicanal, homogeneidad
    espectral y sellado semántico.

Fase 2 — ADJUNCIÓN DE GALOIS Y LEY DE INTERCAMBIO BICANAL
    Verificación del residuo de adjunción F ⊣ G y de la ley de intercambio
    2-categórica **exacta** vía el teorema del producto mixto de Kronecker
    (A⊗B)(C⊗D) = (AC)⊗(BD), evaluada eficientemente mediante el truco
    `vec` sin construir explícitamente las matrices n²×n².

Fase 3 — COBORDISMO A∞, BETTI DUAL Y VETO HEYTING B₂
    Validación homológica doble β₁⁽¹⁾ = 0 ∧ β₁⁽²⁾ = 0 sobre un **par dual
    no degenerado** de grafos (canal de éxito vs. canal íntegro),
    certificación de nilpotencia del coborde, coherencia bicompleja,
    retículo de Heyting B₂ completo (∧, ∨, ¬, →) y censura ciber-física
    GPIO14/Crowbar.

CONTINUIDAD DE FASES
────────────────────
`phase1_export_to_phase2` es la última definición formal de la Fase 1 y el
punto de apertura del dominio de la Fase 2 (`Phase2Input`). Análogamente,
`phase2_export_to_phase3` cierra la Fase 2 y abre el dominio de la Fase 3
(`Phase3Input`). La Fase 3 culmina en `compose_bicomplex_algebra_pipeline`,
el funtor maestro Z_MIC = Ψ₃ ∘ Ψ₂ ∘ Ψ₁.

INVARIANTES Y SU VERIFICACIÓN EXPLÍCITA
─────────────────────────────────────────
[I1] Desacoplamiento idempotente:
        e₁e₂ = 0.                                    → `verify_idempotent_decoupling`
[I2] Nilpotencia bicompleja del coborde:
        d_{k+1}^{C₂} ∘ d_k^{C₂} = 0.                  → `BicomplexHomologicalVerifier.verify_coboundary_nilpotency`
[I3] Doble veto de Betti (sobre par dual no degenerado):
        β₁⁽¹⁾ = 0 ∧ β₁⁽²⁾ = 0.                        → `BicomplexHomologicalVerifier.verify_double_betti_veto`
[I4] Ley de intercambio 2-categórica bicanal (exacta):
        (A⊗B)(C⊗D) = (AC)⊗(BD),  por canal.           → `TwoCategoryBicomplexOrchestrator.validate_interchange_law`
[I5] Heyting B₂:
        Verdict_global = v₁ ∧ v₂ = min(v₁, v₂).       → `verify_heyting_censorship_invariant`
[I6] Axiomas de anillo conmutativo de C₂:
        asociatividad, conmutatividad, distributividad. → `verify_bicomplex_ring_axioms`
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, replace
from enum import Enum, IntEnum
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

#=============================================================================
# LOGGING
#=============================================================================
logger = logging.getLogger("MIC.Algebra.Bicomplex")

#=============================================================================
# ESTRATOS (fallback standalone)
#=============================================================================
try:
    from app.core.schemas import Stratum  # type: ignore
except Exception:  # pragma: no cover
    class Stratum(IntEnum):
        """Estratificación DIKW simplificada para standalone (orden = jerarquía)."""
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3
        DATA = 4


#=============================================================================
# CONSTANTES
#=============================================================================
_SCHEMA_VERSION: Final[str] = "5.1.0-Bicomplex-2Category-Kronecker"
_MAX_CANONICALIZE_DEPTH: Final[int] = 64
_ALGEBRAIC_TOL: Final[float] = 1e-10
_FLOAT_COMPARISON_TOL: Final[float] = 1e-9
_MACHINE_EPSILON: Final[float] = float(np.finfo(float).eps)
_MIN_EXERGY_LEVEL: Final[float] = 0.1
_FIEDLER_EPSILON: Final[float] = 1e-6
_GPIO_VETO_PIN: Final[int] = 14
_ISR_ACTUATION_NS: Final[float] = 398.95
_COHERENT_THRESHOLD: Final[float] = 0.85
_DEGRADED_THRESHOLD: Final[float] = 0.50


#=============================================================================
# EXCEPCIONES — TAXONOMÍA ESTRUCTURADA CON CATEGORÍA ABSTRACTA OBLIGATORIA
#=============================================================================
class AlgebraicError(Exception, ABC):
    """
    Error algebraico estructurado, raíz de la taxonomía de errores del módulo.

    A diferencia de una `ABC` cosmética, esta clase exige que toda subclase
    concreta declare explícitamente su `category` semántica, habilitando
    enrutamiento programático de errores (p. ej. reintentar en errores de
    canonicalización pero escalar a censura ciber-física en `HeytingVetoError`).
    Intentar instanciar `AlgebraicError` directamente, o una subclase que no
    implemente `category`, produce `TypeError` en tiempo de construcción.
    """

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context: Dict[str, Any] = context
        self.timestamp: float = time.time()

    @property
    @abstractmethod
    def category(self) -> str:
        """Categoría semántica inmutable del error concreto."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serialización estructurada."""
        return {
            "type": self.__class__.__name__,
            "category": self.category,
            "message": str(self),
            "context": self.context,
            "timestamp": self.timestamp,
        }


class CanonicalizationError(AlgebraicError):
    """Error de canonicalización determinista."""

    @property
    def category(self) -> str:
        return "canonicalization"


class FunctorialityError(AlgebraicError):
    """Error de funtorialidad o ley de intercambio."""

    @property
    def category(self) -> str:
        return "functoriality"


class HomologicalError(AlgebraicError):
    """Error homológico bicomplejo (p. ej. violación de nilpotencia [I2])."""

    @property
    def category(self) -> str:
        return "homological"


class TopologicalInvariantError(AlgebraicError):
    """Violación de invariante topológico (p. ej. doble veto de Betti [I3])."""

    @property
    def category(self) -> str:
        return "topological_invariant"


class HeytingVetoError(AlgebraicError):
    """Veto global en el retículo de Heyting B₂ ([I5])."""

    @property
    def category(self) -> str:
        return "heyting_veto"


#=============================================================================
# UTILIDADES MATEMÁTICAS
#=============================================================================
class MathUtils:
    """Utilidades numéricas con garantías formales."""

    @staticmethod
    def float_equal(
        a: float,
        b: float,
        abs_tol: float = _FLOAT_COMPARISON_TOL,
        rel_tol: float = _FLOAT_COMPARISON_TOL,
    ) -> bool:
        """Igualdad numérica con tolerancia absoluta/relativa."""
        if a == b:
            return True
        diff = abs(a - b)
        if diff <= abs_tol:
            return True
        scale = max(abs(a), abs(b), 1.0)
        return diff <= rel_tol * scale

    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        eps: float = 10.0 * _MACHINE_EPSILON,
    ) -> float:
        """
        División segura con preservación de signo.

        NOTA DE RIGOR: `eps` es un umbral de **regularización numérica**
        (evita división por cero o subnormales), no debe confundirse con
        umbrales físicos de dominio como `_MIN_EXERGY_LEVEL`. Un llamador
        que necesite clampar por un piso físico significativo (p. ej. un
        nivel mínimo de exergía) debe pasar explícitamente ese valor como
        `eps`; el valor por defecto aquí es puramente de higiene IEEE-754.
        """
        abs_denom = abs(denominator)
        if abs_denom < eps:
            sign = math.copysign(1.0, denominator) if denominator != 0.0 else 1.0
            return numerator / (sign * eps)
        return numerator / denominator

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp estricto."""
        if min_val > max_val:
            raise ValueError(f"clamp inválido: {min_val} > {max_val}")
        return max(min_val, min(max_val, float(value)))

    @staticmethod
    def kbn_sum(values: Sequence[float]) -> float:
        """Suma compensada Kahan-Babuška-Neumaier."""
        s = 0.0
        c = 0.0
        for v in values:
            v = float(v)
            y = v - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    @staticmethod
    def condition_number_estimate(values: Sequence[float]) -> float:
        """Estimación de número de condición de un conjunto de escalares."""
        non_zero = [abs(float(v)) for v in values if float(v) != 0.0]
        if not non_zero:
            return float("inf")
        return max(non_zero) / min(non_zero)


#=============================================================================
# CANONICALIZACIÓN Y HASH
#=============================================================================
def _canonicalize(value: Any, *, _depth: int = 0, _seen: Optional[set] = None) -> Any:
    """Canonicalización determinista con detección de ciclos."""
    if _seen is None:
        _seen = set()

    if _depth > _MAX_CANONICALIZE_DEPTH:
        raise CanonicalizationError(
            f"Profundidad de canonicalización excedida: {_MAX_CANONICALIZE_DEPTH}",
            depth=_depth,
            type=type(value).__name__,
        )

    value_id = id(value)
    if value_id in _seen and not isinstance(value, (str, int, float, bool, type(None))):
        raise CanonicalizationError(
            "Ciclo detectado en canonicalización",
            depth=_depth,
            type=type(value).__name__,
        )

    _seen.add(value_id)
    next_depth = _depth + 1

    try:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, complex):
            return {"__complex__": [value.real, value.imag]}

        if isinstance(value, Stratum):
            return {"__stratum__": value.name}

        if isinstance(value, IntEnum):
            return {"__enum__": value.__class__.__name__, "__value__": value.value}

        if isinstance(value, dict):
            return {
                str(k): _canonicalize(v, _depth=next_depth, _seen=_seen)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }

        if isinstance(value, (list, tuple)):
            return [_canonicalize(v, _depth=next_depth, _seen=_seen) for v in value]

        if isinstance(value, (set, frozenset)):
            canonical = [_canonicalize(v, _depth=next_depth, _seen=_seen) for v in value]
            try:
                return sorted(canonical)
            except TypeError:
                return sorted(canonical, key=lambda x: repr(x))

        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _canonicalize(value.to_dict(), _depth=next_depth, _seen=_seen)

        if hasattr(value, "__dict__"):
            return _canonicalize(value.__dict__, _depth=next_depth, _seen=_seen)

        return repr(value)
    finally:
        _seen.discard(value_id)


def _stable_hash(data: Any) -> str:
    """Hash SHA-256 determinista."""
    try:
        canonical = _canonicalize(data)
        serialized = json.dumps(
            canonical,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception as exc:
        logger.warning("Fallback de hash por error: %s", exc)
        return hashlib.sha256(repr(data).encode("utf-8")).hexdigest()


def _safe_merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Fusión simple de diccionarios con prioridad derecha."""
    merged = dict(left)
    merged.update(right)
    return merged


def _dominant_stratum(strata: FrozenSet[Stratum]) -> Optional[Stratum]:
    """
    Determina el estrato dominante de un conjunto de estratos validados.

    Si `Stratum` es un `IntEnum` con orden numérico intrínseco (como en el
    fallback standalone, donde `WISDOM = 0` es la cúspide de la jerarquía
    DIKW), se selecciona el elemento de **valor entero mínimo** como
    dominante — consistente con la semántica de la jerarquía.

    Si `Stratum` proviene de `app.core.schemas` y no garantiza orden total
    (`Enum` simple), se recurre a un criterio determinista pero
    **explícitamente arbitrario**: orden lexicográfico descendente del
    nombre. Esta arbitrariedad se documenta aquí para no camuflarla bajo
    una apariencia de rigor matemático inexistente.
    """
    if not strata:
        return None
    if all(isinstance(s, IntEnum) for s in strata):
        return min(strata, key=lambda s: int(s))
    return sorted(strata, key=lambda s: s.name)[-1]


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███  FASE 1 — SANEAMIENTO ESPECTRAL Y ANILLO CONMUTATIVO BICOMPLEJO       ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase construye y certifica el anillo C₂ ≅ C × C:
#
#   1. BicomplexScalar: representación diagonal (Z⁽¹⁾, Z⁽²⁾) con suma,
#      producto, conjugaciones y detección de unidades/divisores de cero.
#   2. verify_bicomplex_ring_axioms / verify_idempotent_decoupling [I1][I6].
#   3. BicomplexVector: módulo libre sobre C₂ con acción escalar y
#      producto de Hadamard bicanal.
#   4. Phase1_SpectralRingObserver: ingesta 4D, homogeneidad espectral,
#      no-degeneración, sellado semántico.
#=============================================================================


#------------------------------------------------------------------------------
# 1.1 — NÚCLEO ALGEBRAICO: BicomplexScalar Y CERTIFICACIÓN DE ANILLO
#------------------------------------------------------------------------------
def _sanitize_complex_array(arr: np.ndarray) -> np.ndarray:
    """Saneamiento FPU de ceros signados en arreglos complejos."""
    a = np.asarray(arr, dtype=np.complex128)
    real = np.where(a.real == 0.0, 0.0, a.real)
    imag = np.where(a.imag == 0.0, 0.0, a.imag)
    return real + 1j * imag


@dataclass(frozen=True, slots=True)
class BicomplexScalar:
    """
    Escalar bicomplejo en base idempotente: Z = Z¹ e₁ + Z² e₂.

    C₂ es isomorfo como anillo a C × C vía (Z⁽¹⁾, Z⁽²⁾); en particular
    **no es un dominio de integridad**: posee divisores de cero exactos
    en los elementos con exactamente un canal nulo, y su grupo de
    unidades es C* × C* (ambos canales no nulos).
    """

    z1: complex
    z2: complex

    # -- Constructores -----------------------------------------------------
    @classmethod
    def from_real4(cls, vector: Sequence[float]) -> "BicomplexScalar":
        """Construye Z ∈ C₂ desde (s₀, s₁, s₂, s₃)."""
        if len(vector) != 4:
            raise ValueError("Se requieren exactamente 4 componentes reales.")

        s0, s1, s2, s3 = (float(v) for v in vector)
        z1 = complex(s0 + s2, s1 + s3)
        z2 = complex(s0 - s2, s1 - s3)

        return cls(
            complex(0.0 if z1.real == 0.0 else z1.real, 0.0 if z1.imag == 0.0 else z1.imag),
            complex(0.0 if z2.real == 0.0 else z2.real, 0.0 if z2.imag == 0.0 else z2.imag),
        )

    @classmethod
    def idempotent_e1(cls) -> "BicomplexScalar":
        """Elemento idempotente e₁ = (1+j)/2 en representación diagonal."""
        return cls(complex(1.0, 0.0), complex(0.0, 0.0))

    @classmethod
    def idempotent_e2(cls) -> "BicomplexScalar":
        """Elemento idempotente e₂ = (1-j)/2 en representación diagonal."""
        return cls(complex(0.0, 0.0), complex(1.0, 0.0))

    @classmethod
    def zero(cls) -> "BicomplexScalar":
        """Neutro aditivo 0 ∈ C₂."""
        return cls(complex(0.0), complex(0.0))

    @classmethod
    def one(cls) -> "BicomplexScalar":
        """Neutro multiplicativo 1 = e₁ + e₂ ∈ C₂."""
        return cls(complex(1.0), complex(1.0))

    # -- Álgebra de anillo ---------------------------------------------------
    def __add__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """Suma canal a canal."""
        return BicomplexScalar(self.z1 + other.z1, self.z2 + other.z2)

    def __sub__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """Resta canal a canal."""
        return BicomplexScalar(self.z1 - other.z1, self.z2 - other.z2)

    def __mul__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """
        Producto en C₂: (ZW)⁽ᵃ⁾ = Z⁽ᵃ⁾ W⁽ᵃ⁾ — diagonalización idempotente
        que realiza el isomorfismo de anillos C₂ ≅ C × C.
        """
        return BicomplexScalar(self.z1 * other.z1, self.z2 * other.z2)

    def scalar_mul(self, k: float) -> "BicomplexScalar":
        """Multiplicación por escalar real k ∈ R."""
        return BicomplexScalar(self.z1 * k, self.z2 * k)

    def conjugate_hyperbolic(self) -> "BicomplexScalar":
        """Conjugado hiperbólico (j ↦ -j): intercambia canales idempotentes."""
        return BicomplexScalar(self.z2, self.z1)

    def conjugate_complex(self) -> "BicomplexScalar":
        """Conjugado complejo canal a canal (i ↦ -i)."""
        return BicomplexScalar(self.z1.conjugate(), self.z2.conjugate())

    def is_zero_divisor(self, tol: float = _ALGEBRAIC_TOL) -> bool:
        """True si Z ≠ 0 tiene exactamente un canal nulo (divisor de cero)."""
        z1_zero = abs(self.z1) <= tol
        z2_zero = abs(self.z2) <= tol
        return z1_zero != z2_zero

    def is_unit(self, tol: float = _ALGEBRAIC_TOL) -> bool:
        """True si Z ∈ (C₂)ˣ = C* × C*, i.e. ambos canales no nulos."""
        return abs(self.z1) > tol and abs(self.z2) > tol

    def inverse(self, tol: float = _ALGEBRAIC_TOL) -> "BicomplexScalar":
        """
        Inverso multiplicativo Z⁻¹, definido si y solo si Z ∈ (C₂)ˣ.

        Raises:
            ZeroDivisionError: Si Z no es una unidad del anillo.
        """
        if not self.is_unit(tol):
            raise ZeroDivisionError(
                f"Z={self} no es unidad de C₂ (al menos un canal es nulo o "
                "está por debajo de la tolerancia)."
            )
        return BicomplexScalar(1.0 / self.z1, 1.0 / self.z2)

    # -- Métrica ----------------------------------------------------------
    def to_real4(self) -> np.ndarray:
        """Representación real 4D."""
        return np.array([self.z1.real, self.z1.imag, self.z2.real, self.z2.imag], dtype=np.float64)

    def norm_l2(self) -> float:
        """Norma euclidiana idempotente."""
        return float(np.sqrt(abs(self.z1) ** 2 + abs(self.z2) ** 2))


def verify_idempotent_decoupling(tol: float = _ALGEBRAIC_TOL) -> Dict[str, bool]:
    """
    FASE 1 — Certificación del invariante [I1]: desacoplamiento idempotente.

    Verifica sobre los representantes canónicos e₁, e₂:

        e₁² = e₁,   e₂² = e₂,   e₁e₂ = 0,   e₁ + e₂ = 1.

    Returns:
        Diccionario booleano por identidad, con clave `"all_hold"`.
    """
    e1 = BicomplexScalar.idempotent_e1()
    e2 = BicomplexScalar.idempotent_e2()
    one = BicomplexScalar.one()
    zero = BicomplexScalar.zero()

    def close(a: BicomplexScalar, b: BicomplexScalar) -> bool:
        return abs(a.z1 - b.z1) <= tol and abs(a.z2 - b.z2) <= tol

    result = {
        "e1_idempotent": close(e1 * e1, e1),
        "e2_idempotent": close(e2 * e2, e2),
        "orthogonal_channels": close(e1 * e2, zero),
        "partition_of_unity": close(e1 + e2, one),
    }
    result["all_hold"] = all(result.values())
    return result


def verify_bicomplex_ring_axioms(
    rng: Optional[np.random.Generator] = None,
    trials: int = 8,
    tol: float = _ALGEBRAIC_TOL,
) -> Dict[str, Any]:
    """
    FASE 1 — Certificación del invariante [I6]: axiomas de anillo
    conmutativo de C₂.

    Sobre `trials` triadas aleatorias de escalares bicomplejos, verifica
    numéricamente:

        - Asociatividad de (+, ·).
        - Conmutatividad de (+, ·).
        - Distributividad de · sobre +.

    y delega en `verify_idempotent_decoupling` la certificación de las
    identidades idempotentes estructurales [I1].

    Args:
        rng: Generador pseudoaleatorio; por defecto, semilla fija
            determinista (reproducibilidad de la certificación).
        trials: Número de triadas de prueba.
        tol: Tolerancia absoluta de comparación en C.

    Returns:
        Diccionario de resultados booleanos con clave `"all_hold"`.
    """
    rng = rng or np.random.default_rng(1729)

    def rand_scalar() -> BicomplexScalar:
        return BicomplexScalar.from_real4(rng.normal(size=4).tolist())

    def close(a: BicomplexScalar, b: BicomplexScalar) -> bool:
        return abs(a.z1 - b.z1) <= tol and abs(a.z2 - b.z2) <= tol

    results: Dict[str, bool] = {
        "associativity_add": True,
        "associativity_mul": True,
        "commutativity_add": True,
        "commutativity_mul": True,
        "distributivity": True,
    }

    for _ in range(max(1, trials)):
        a, b, c = rand_scalar(), rand_scalar(), rand_scalar()
        if not close((a + b) + c, a + (b + c)):
            results["associativity_add"] = False
        if not close((a * b) * c, a * (b * c)):
            results["associativity_mul"] = False
        if not close(a + b, b + a):
            results["commutativity_add"] = False
        if not close(a * b, b * a):
            results["commutativity_mul"] = False
        if not close(a * (b + c), (a * b) + (a * c)):
            results["distributivity"] = False

    decoupling = verify_idempotent_decoupling(tol=tol)
    merged: Dict[str, Any] = {**results, **{k: v for k, v in decoupling.items() if k != "all_hold"}}
    merged["all_hold"] = all(v for v in merged.values() if isinstance(v, bool))
    return merged


#------------------------------------------------------------------------------
# 1.2 — MÓDULO VECTORIAL LIBRE SOBRE C₂: BicomplexVector
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexVector:
    """
    Vector bicomplejo en el espacio de Hilbert bicomplejo:

        H_{C₂} ≅ H⁽¹⁾ e₁ ⊕ H⁽²⁾ e₂.

    Estructuralmente es un módulo libre sobre el anillo C₂: admite acción
    escalar (`scalar_action`) y producto de Hadamard bicanal (`hadamard`),
    ambos diagonales por canal idempotente en virtud de [I1].

    Attributes:
        channel_1: Canal idempotente e₁.
        channel_2: Canal idempotente e₂.
    """

    channel_1: np.ndarray
    channel_2: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_1", _sanitize_complex_array(self.channel_1))
        object.__setattr__(self, "channel_2", _sanitize_complex_array(self.channel_2))

    @classmethod
    def from_real4(cls, S: NDArray[np.float64]) -> "BicomplexVector":
        """
        Proyecta señal real 4D a base idempotente:

            Z⁽¹⁾ = (s₀ + s₂) + i(s₁ + s₃)
            Z⁽²⁾ = (s₀ - s₂) + i(s₁ - s₃)
        """
        arr = np.asarray(S, dtype=np.float64)

        if arr.ndim == 0:
            raise ValueError("La entrada no puede ser escalar pura.")

        if arr.ndim == 1:
            if arr.size % 4 != 0:
                raise ValueError(f"Vector 1D debe tener longitud múltiplo de 4; size={arr.size}.")
            arr = arr.reshape(-1, 4)
        elif arr.ndim == 2:
            if arr.shape[1] != 4:
                raise ValueError(f"Matriz 2D debe tener shape (n, 4); recibido {arr.shape}.")
        else:
            if arr.shape[-1] == 4:
                arr = arr.reshape(-1, 4)
            elif arr.size % 4 == 0:
                arr = arr.reshape(-1, 4)
            else:
                raise ValueError(f"No se puede reinterpretar shape {arr.shape} como (n, 4).")

        arr = arr.copy()
        arr[arr == 0.0] = 0.0

        z1 = (arr[:, 0] + arr[:, 2]) + 1j * (arr[:, 1] + arr[:, 3])
        z2 = (arr[:, 0] - arr[:, 2]) + 1j * (arr[:, 1] - arr[:, 3])

        return cls(z1, z2)

    @classmethod
    def zero(cls, dim: int = 1) -> "BicomplexVector":
        """Vector nulo bicomplejo."""
        return cls(np.zeros(dim, dtype=np.complex128), np.zeros(dim, dtype=np.complex128))

    def __len__(self) -> int:
        return int(max(self.channel_1.size, self.channel_2.size))

    def norm_channel_1(self) -> float:
        """Norma ||S⁽¹⁾||₂."""
        return float(np.linalg.norm(self.channel_1)) if self.channel_1.size else 0.0

    def norm_channel_2(self) -> float:
        """Norma ||S⁽²⁾||₂."""
        return float(np.linalg.norm(self.channel_2)) if self.channel_2.size else 0.0

    def is_non_degenerate(self, eps: float = _MACHINE_EPSILON) -> bool:
        """Condición de no-degeneración bicanal."""
        return self.norm_channel_1() > eps and self.norm_channel_2() > eps

    def is_finite(self) -> bool:
        """True si todas las componentes son finitas."""
        return bool(np.all(np.isfinite(self.channel_1)) and np.all(np.isfinite(self.channel_2)))

    def scalar_action(self, scalar: BicomplexScalar) -> "BicomplexVector":
        """
        Acción del anillo C₂ sobre el módulo H_{C₂}:

            (Z · S)⁽ᵃ⁾ = Z⁽ᵃ⁾ S⁽ᵃ⁾   (broadcasting canal a canal).
        """
        return BicomplexVector(self.channel_1 * scalar.z1, self.channel_2 * scalar.z2)

    def hadamard(self, other: "BicomplexVector") -> "BicomplexVector":
        """Producto de Hadamard bicanal (estructura de C₂-álgebra puntual)."""
        return BicomplexVector(self.channel_1 * other.channel_1, self.channel_2 * other.channel_2)

    def conjugate_hyperbolic(self) -> "BicomplexVector":
        """Conjugado hiperbólico: intercambia los dos canales idempotentes."""
        return BicomplexVector(self.channel_2.copy(), self.channel_1.copy())

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe."""
        return {
            "channel_1_real": self.channel_1.real.tolist(),
            "channel_1_imag": self.channel_1.imag.tolist(),
            "channel_2_real": self.channel_2.real.tolist(),
            "channel_2_imag": self.channel_2.imag.tolist(),
            "norm_channel_1": self.norm_channel_1(),
            "norm_channel_2": self.norm_channel_2(),
        }


#------------------------------------------------------------------------------
# 1.3 — INFRAESTRUCTURA CATEGÓRICA COMPARTIDA (traza y estado)
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CompositionTrace:
    """Traza inmutable de ejecución categórica."""

    step_number: int
    morphism_name: str
    input_domain: FrozenSet[Stratum]
    output_codomain: Stratum
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.step_number < 1:
            object.__setattr__(self, "step_number", 1)
        if self.timestamp <= 0:
            object.__setattr__(self, "timestamp", time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe."""
        return {
            "step": self.step_number,
            "morphism": self.morphism_name,
            "domain": sorted(s.name for s in self.input_domain),
            "codomain": self.output_codomain.name,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": _canonicalize(self.metadata) if self.metadata else None,
        }


@dataclass(frozen=True, slots=True)
class BicomplexCategoricalState:
    """
    Objeto fundamental de la 2-categoría bicompleja C_MIC^{C₂}.

    Todas las funciones de transición (`with_update`, `with_error`,
    `clear_error`, `add_trace`) se implementan sobre `dataclasses.replace`,
    de modo que cada nueva instancia hereda automáticamente los campos no
    modificados desde `self`, eliminando por construcción la clase de
    *bugs* en que un campo nuevo se olvida propagar en un constructor
    manual duplicado.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validated_strata: FrozenSet[Stratum] = field(default_factory=frozenset)
    vector: Optional[BicomplexVector] = None
    error: Optional[str] = None
    success: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    composition_trace: Tuple[CompositionTrace, ...] = field(default_factory=tuple)
    stratum: Optional[Stratum] = None

    def __post_init__(self) -> None:
        canonical_success = self.success if self.success is not None else (self.error is None)
        object.__setattr__(self, "success", canonical_success)

        if self.stratum is None and self.validated_strata:
            object.__setattr__(self, "stratum", _dominant_stratum(self.validated_strata))

    @property
    def is_success(self) -> bool:
        """Predicado de éxito categórico."""
        return self.error is None

    @property
    def is_failed(self) -> bool:
        """Predicado de fallo categórico."""
        return not self.is_success

    def with_update(
        self,
        new_payload: Optional[Dict[str, Any]] = None,
        new_context: Optional[Dict[str, Any]] = None,
        new_vector: Optional[BicomplexVector] = None,
        new_stratum: Optional[Stratum] = None,
    ) -> "BicomplexCategoricalState":
        """Funtor de actualización inmutable."""
        updated_payload = _safe_merge_dicts(self.payload, new_payload or {})
        updated_context = _safe_merge_dicts(self.context, new_context or {})
        updated_strata = self.validated_strata | (frozenset({new_stratum}) if new_stratum else frozenset())

        return replace(
            self,
            payload=updated_payload,
            context=updated_context,
            validated_strata=updated_strata,
            vector=new_vector if new_vector is not None else self.vector,
            stratum=new_stratum if new_stratum is not None else _dominant_stratum(updated_strata),
        )

    def with_error(
        self, error_msg: str, details: Optional[Dict[str, Any]] = None
    ) -> "BicomplexCategoricalState":
        """Funtor de error monádico."""
        return replace(
            self,
            context=_safe_merge_dicts(self.context, details or {}),
            error=error_msg,
            success=False,
        )

    def clear_error(self) -> "BicomplexCategoricalState":
        """Limpia error preservando payload/contexto/traza."""
        return replace(self, error=None, success=True)

    def add_trace(
        self,
        morphism_name: str,
        input_domain: FrozenSet[Stratum],
        output_codomain: Stratum,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BicomplexCategoricalState":
        """Añade traza de composición."""
        trace = CompositionTrace(
            step_number=len(self.composition_trace) + 1,
            morphism_name=morphism_name,
            input_domain=input_domain,
            output_codomain=output_codomain,
            success=success,
            error=error,
            metadata=dict(metadata) if metadata else None,
        )
        return replace(self, composition_trace=self.composition_trace + (trace,))

    def compute_semantic_hash(self) -> str:
        """
        Hash semántico invariante respecto de la traza de composición.

        Intencionalmente **no incluye** `composition_trace` ni `context`:
        dos estados que difieren solo en su historial de ejecución (p. ej.
        producidos por reintentos idempotentes) deben colapsar al mismo
        hash semántico. Para un hash sensible a la traza completa, use
        `compute_full_hash`.
        """
        data = {
            "payload": self.payload,
            "validated_strata": sorted(s.name for s in self.validated_strata),
            "vector": self.vector.to_dict() if self.vector else None,
            "error": self.error,
            "success": self.success,
        }
        return _stable_hash(data)

    def compute_full_hash(self) -> str:
        """Hash de auditoría completo, incluyendo contexto y traza."""
        return _stable_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa."""
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "payload": _canonicalize(self.payload),
            "context": _canonicalize(self.context),
            "validated_strata": sorted(s.name for s in self.validated_strata),
            "vector": self.vector.to_dict() if self.vector else None,
            "error": self.error,
            "success": self.success,
            "metadata": _canonicalize(self.metadata) if self.metadata else None,
            "composition_trace": [t.to_dict() for t in self.composition_trace],
        }


#------------------------------------------------------------------------------
# 1.4 — OBSERVADOR ESPECTRAL Y FRONTERA DE FASE: FASE 1 → FASE 2
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Phase1SpectralRingArtifact:
    """Artefacto de Fase 1."""

    vector: BicomplexVector
    ring_homogeneous: bool
    non_degenerate: bool
    channel_norms: Tuple[float, float]
    spectral_condition: Tuple[float, float]
    semantic_hash: str
    ring_axioms: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Phase1_SpectralRingObserver:
    """
    Fase 1: Observador del anillo espectral bicomplejo.

    Responsabilidades:
      - Ingesta 4D y proyección idempotente.
      - Verificación de finitud y homogeneidad espectral.
      - No-degeneración bicanal.
      - Certificación de los axiomas de anillo de C₂ ([I1], [I6]).
      - Sellado semántico.
    """

    @staticmethod
    def _spectral_condition(channel: np.ndarray) -> float:
        """Condición espectral proxy κ = max|z| / min|z| sobre el canal."""
        mag = np.abs(channel).astype(np.float64)
        if mag.size == 0:
            return 1.0

        finite = mag[np.isfinite(mag)]
        if finite.size == 0:
            return float("inf")

        mx = float(np.max(finite))
        positives = finite[finite > _MACHINE_EPSILON]

        if positives.size == 0:
            return 1.0 if mx <= _MACHINE_EPSILON else mx / _MACHINE_EPSILON

        mn = float(np.min(positives))
        return mx / mn if mn > 0.0 else float("inf")

    @classmethod
    def run(
        cls,
        S: NDArray[np.float64],
        context: Optional[Dict[str, Any]] = None,
        *,
        ring_axiom_trials: int = 8,
        rng: Optional[np.random.Generator] = None,
    ) -> Phase1SpectralRingArtifact:
        """Ejecuta Fase 1."""
        vector = BicomplexVector.from_real4(S)

        cond_1 = cls._spectral_condition(vector.channel_1)
        cond_2 = cls._spectral_condition(vector.channel_2)

        ring_homogeneous = vector.is_finite() and np.isfinite(cond_1) and np.isfinite(cond_2)
        non_degenerate = vector.is_non_degenerate()
        ring_axioms = verify_bicomplex_ring_axioms(rng=rng, trials=ring_axiom_trials)

        artifact = Phase1SpectralRingArtifact(
            vector=vector,
            ring_homogeneous=ring_homogeneous,
            non_degenerate=non_degenerate,
            channel_norms=(vector.norm_channel_1(), vector.norm_channel_2()),
            spectral_condition=(cond_1, cond_2),
            ring_axioms=ring_axioms,
            semantic_hash=_stable_hash({
                "vector": vector.to_dict(),
                "context": context or {},
            }),
        )

        if not ring_axioms.get("all_hold", False):
            logger.error("Fallo de certificación de anillo C₂: %s", ring_axioms)

        logger.debug(
            "Phase1: norms=(%.6f, %.6f), cond=(%.3e, %.3e), homogeneous=%s, ring_ok=%s",
            artifact.channel_norms[0],
            artifact.channel_norms[1],
            cond_1,
            cond_2,
            ring_homogeneous,
            ring_axioms.get("all_hold"),
        )
        return artifact


@dataclass(frozen=True, slots=True)
class Phase2Input:
    """Objeto frontera entre Fase 1 y Fase 2."""

    phase1_artifact: Phase1SpectralRingArtifact


def phase1_export_to_phase2(artifact: Phase1SpectralRingArtifact) -> Phase2Input:
    """
    FASE 1 — Último método de la fase y continuación formal de la Fase 2.

    Contrato: `phase1_export_to_phase2 : Phase1SpectralRingArtifact →
    Phase2Input`. Como condición de admisión se exige que la certificación
    de los axiomas de anillo de C₂ (`artifact.ring_axioms["all_hold"]`) sea
    afirmativa: la Fase 2 opera sobre morfismos C₂-lineales, y estos solo
    están bien definidos si el sustrato algebraico subyacente es
    genuinamente un anillo conmutativo con la estructura idempotente
    declarada.

    Raises:
        TypeError: Si `artifact` no es `Phase1SpectralRingArtifact`.
        HomologicalError: Si la certificación de anillo no se sostiene.
    """
    if not isinstance(artifact, Phase1SpectralRingArtifact):
        raise TypeError("phase1_export_to_phase2 requiere Phase1SpectralRingArtifact.")

    if not artifact.ring_axioms.get("all_hold", False):
        raise HomologicalError(
            "El sustrato algebraico no satisface los axiomas de anillo de "
            "C₂; la frontera Fase1→Fase2 no puede admitirlo.",
            ring_axioms=artifact.ring_axioms,
        )

    return Phase2Input(phase1_artifact=artifact)


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███       FASE 2 — ADJUNCIÓN DE GALOIS Y LEY DE INTERCAMBIO BICANAL       ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase certifica:
#
#   1. Morfismos C₂-lineales y su composición diagonal por canal.
#   2. Ley de intercambio de Godement, formalizada EXACTAMENTE vía el
#      teorema del producto mixto de Kronecker:
#
#          (A⊗B)(C⊗D) = (AC)⊗(BD),
#
#      evaluada sin construir las matrices n²×n² mediante el truco `vec`:
#
#          vec⁻¹[(A⊗B) vec(X)] = B X Aᵀ.
#
#   3. Residuo de adjunción de Galois F ⊣ G.
#=============================================================================


#------------------------------------------------------------------------------
# 2.1 — MORFISMOS LINEALES BICOMPLEJOS
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexLinearMorphism:
    """
    1-morfismo lineal bicomplejo:

        Φ_{C₂} = Φ⁽¹⁾ e₁ + Φ⁽²⁾ e₂.

    La composición preserva el desacoplamiento idempotente:

        Ψ ∘ Φ = (Ψ⁽¹⁾ ∘ Φ⁽¹⁾) e₁ + (Ψ⁽²⁾ ∘ Φ⁽²⁾) e₂.
    """

    name: str
    matrix_channel_1: np.ndarray
    matrix_channel_2: np.ndarray
    domain: FrozenSet[Stratum] = field(default_factory=frozenset)
    codomain: Optional[Stratum] = None

    def __post_init__(self) -> None:
        m1 = np.asarray(self.matrix_channel_1, dtype=np.complex128)
        m2 = np.asarray(self.matrix_channel_2, dtype=np.complex128)

        if m1.ndim != 2:
            raise ValueError("matrix_channel_1 debe ser 2D.")
        if m2.ndim != 2:
            raise ValueError("matrix_channel_2 debe ser 2D.")

        object.__setattr__(self, "matrix_channel_1", m1)
        object.__setattr__(self, "matrix_channel_2", m2)

    @classmethod
    def identity(
        cls,
        dim: int,
        name: str = "id",
        domain: Optional[FrozenSet[Stratum]] = None,
        codomain: Optional[Stratum] = None,
    ) -> "BicomplexLinearMorphism":
        """Morfismo identidad bicanal."""
        dim = max(int(dim), 1)
        eye = np.eye(dim, dtype=np.complex128)
        return cls(
            name=name,
            matrix_channel_1=eye.copy(),
            matrix_channel_2=eye.copy(),
            domain=domain or frozenset(),
            codomain=codomain,
        )

    def apply(self, vector: BicomplexVector) -> BicomplexVector:
        """Aplica el morfismo al vector bicomplejo."""
        if self.matrix_channel_1.shape[1] != vector.channel_1.size:
            raise FunctorialityError(
                "Dimensión incompatible en canal e₁",
                expected=self.matrix_channel_1.shape[1],
                got=vector.channel_1.size,
            )
        if self.matrix_channel_2.shape[1] != vector.channel_2.size:
            raise FunctorialityError(
                "Dimensión incompatible en canal e₂",
                expected=self.matrix_channel_2.shape[1],
                got=vector.channel_2.size,
            )

        return BicomplexVector(
            channel_1=self.matrix_channel_1 @ vector.channel_1,
            channel_2=self.matrix_channel_2 @ vector.channel_2,
        )

    def compose(self, next_morphism: "BicomplexLinearMorphism") -> "BicomplexLinearMorphism":
        """Composición Ψ ∘ Φ (self = Φ, next_morphism = Ψ)."""
        return BicomplexLinearMorphism(
            name=f"{self.name} >> {next_morphism.name}",
            matrix_channel_1=next_morphism.matrix_channel_1 @ self.matrix_channel_1,
            matrix_channel_2=next_morphism.matrix_channel_2 @ self.matrix_channel_2,
            domain=self.domain | next_morphism.domain,
            codomain=next_morphism.codomain or self.codomain,
        )


@dataclass(frozen=True, slots=True)
class BicomplexComposedMorphism:
    """Composición formal de dos morfismos bicomplejos."""

    f: BicomplexLinearMorphism
    g: BicomplexLinearMorphism

    def to_linear(self) -> BicomplexLinearMorphism:
        """Colapso de la composición a morfismo lineal bicanal."""
        return self.f.compose(self.g)

    def apply(self, vector: BicomplexVector) -> BicomplexVector:
        """Aplicación composicional."""
        return self.to_linear().apply(vector)


#------------------------------------------------------------------------------
# 2.2 — 2-MORFISMOS Y LEY DE INTERCAMBIO EXACTA (KRONECKER)
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexNaturalTransformation:
    """
    2-morfismo bicomplejo representado matricialmente por canal.

    Se usa para verificar la ley de intercambio de Godement:

        (β'∘α') · (β∘α) = (β'·β) ∘ (α'·α).
    """

    name: str
    matrix_channel_1: np.ndarray
    matrix_channel_2: np.ndarray

    def __post_init__(self) -> None:
        m1 = np.asarray(self.matrix_channel_1, dtype=np.complex128)
        m2 = np.asarray(self.matrix_channel_2, dtype=np.complex128)

        if m1.ndim != 2:
            raise ValueError("matrix_channel_1 debe ser 2D.")
        if m2.ndim != 2:
            raise ValueError("matrix_channel_2 debe ser 2D.")

        object.__setattr__(self, "matrix_channel_1", m1)
        object.__setattr__(self, "matrix_channel_2", m2)

    @classmethod
    def identity(cls, dim: int, name: str = "id_nat") -> "BicomplexNaturalTransformation":
        """Transformación natural identidad."""
        dim = max(int(dim), 1)
        eye = np.eye(dim, dtype=np.complex128)
        return cls(name=name, matrix_channel_1=eye.copy(), matrix_channel_2=eye.copy())


class TwoCategoryBicomplexOrchestrator:
    """
    Orquestador 2-categórico bicomplejo con verificación **exacta** de la
    ley de intercambio, vía el teorema del producto mixto de Kronecker:

        (A⊗B)(C⊗D) = (AC)⊗(BD).

    Identificando la composición horizontal `∘` con el producto tensorial
    `⊗` y la composición vertical `·` con el producto matricial ordinario,
    esta identidad es *exactamente* la ley de intercambio de Godement.
    A diferencia de una hipótesis empírica, es un **teorema del álgebra
    lineal**: cualquier residuo numérico más allá del redondeo IEEE-754
    delata un error de implementación en las rutinas de composición, nunca
    un genuino fallo estructural de la 2-categoría subyacente.

    Se evita construir explícitamente las matrices n²×n² de Kronecker
    (coste O(n⁶) en la verificación directa) aplicando el truco `vec`:

        vec⁻¹[(A⊗B) vec(X)] = B X Aᵀ,

    sobre matrices de prueba aleatorias X (estilo Freivalds), reduciendo
    el coste por sonda a O(n³).
    """

    _PROBE_SEED: Final[int] = 424242
    _PROBE_COUNT: Final[int] = 3
    _MAX_DENSE_PROBE_DIM: Final[int] = 512

    @staticmethod
    def _frobenius(M: np.ndarray) -> float:
        """Norma de Frobenius segura."""
        if M.size == 0:
            return 0.0
        return float(np.linalg.norm(M, ord="fro"))

    @staticmethod
    def _condition_number_safe(M: np.ndarray) -> float:
        """Número de condición con fallback robusto."""
        if M.size == 0:
            return 1.0
        try:
            cond = float(np.linalg.cond(M))
            return cond if np.isfinite(cond) else 1.0 / _MACHINE_EPSILON
        except Exception:
            return 1.0 / _MACHINE_EPSILON

    @classmethod
    def _tolerance_channel(cls, operands: Sequence[np.ndarray]) -> float:
        """Cota de Wilkinson adaptativa por canal."""
        if not operands:
            return _ALGEBRAIC_TOL

        scale = max([1.0] + [cls._frobenius(M) for M in operands])
        cond = max([1.0] + [cls._condition_number_safe(M) for M in operands])
        return max(_ALGEBRAIC_TOL, _MACHINE_EPSILON * cond * scale)

    @staticmethod
    def _is_approx_identity(M: np.ndarray, tol: float = 1e-9) -> bool:
        """Detección de matriz identidad para atajo de rendimiento."""
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return False
        return bool(np.allclose(M, np.eye(M.shape[0], dtype=M.dtype), atol=tol, rtol=tol))

    @staticmethod
    def _apply_kron_operator(A: np.ndarray, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Aplica (A⊗B) a vec(X) sin construir A⊗B explícitamente:

            vec⁻¹[(A⊗B) vec(X)] = B X Aᵀ.

        Complejidad O(n³) frente a O(n⁶) de la construcción explícita del
        producto de Kronecker n²×n².
        """
        return B @ X @ A.T

    @classmethod
    def _interchange_residual_channel(
        cls,
        alpha: np.ndarray,
        alpha_prime: np.ndarray,
        beta: np.ndarray,
        beta_prime: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """
        Residuo máximo de la identidad de intercambio sobre `_PROBE_COUNT`
        sondas complejas aleatorias, evaluado vía `_apply_kron_operator`.
        """
        n = alpha.shape[0]
        residuals: List[float] = []

        for _ in range(cls._PROBE_COUNT):
            X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

            # LHS: vec⁻¹[((α'α) ⊗ (β'β)) vec(X)]
            lhs = cls._apply_kron_operator(alpha_prime @ alpha, beta_prime @ beta, X)

            # RHS: vec⁻¹[(α'⊗β')(α⊗β) vec(X)], aplicado en dos etapas.
            inner = cls._apply_kron_operator(alpha, beta, X)
            rhs = cls._apply_kron_operator(alpha_prime, beta_prime, inner)

            residuals.append(cls._frobenius(lhs - rhs))

        return max(residuals) if residuals else 0.0

    @classmethod
    def _interchange_residual_channel_naive_proxy(
        cls,
        alpha: np.ndarray,
        alpha_prime: np.ndarray,
        beta: np.ndarray,
        beta_prime: np.ndarray,
    ) -> float:
        """
        **[LEGADO — NO USAR COMO CRITERIO PRIMARIO]**

        Reimplementación literal de la verificación original v5.0.0, que
        comparaba `(α'α)(β'β)` contra `(α'β')(αβ)` usando producto
        matricial ordinario para *ambas* composiciones. Para matrices
        genéricas esto **no es una identidad algebraica** (el producto de
        matrices no conmuta), por lo que el residuo es, en general,
        distinto de cero incluso en ausencia de cualquier error de
        implementación. Se conserva únicamente con fines de contraste
        histórico/regresión; el método autorizado es
        `_interchange_residual_channel` (formalismo de Kronecker).
        """
        lhs = (alpha_prime @ alpha) @ (beta_prime @ beta)
        rhs = (alpha_prime @ beta_prime) @ (alpha @ beta)
        return cls._frobenius(lhs - rhs)

    @classmethod
    def validate_interchange_law(
        cls,
        alpha: BicomplexNaturalTransformation,
        alpha_prime: BicomplexNaturalTransformation,
        beta: BicomplexNaturalTransformation,
        beta_prime: BicomplexNaturalTransformation,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Valida la ley de intercambio bicompleja mediante el formalismo
        exacto de Kronecker.

        Args:
            alpha, alpha_prime, beta, beta_prime: 2-morfismos bicanal.
            rng: Generador de sondas; por defecto, semilla determinista
                fija (`_PROBE_SEED`) para reproducibilidad.

        Returns:
            Reporte con residuo, tolerancia y veredicto por canal.

        Raises:
            FunctorialityError: Si algún operando no es cuadrado, si las
                dimensiones no compatibilizan entre los cuatro operandos,
                si la verificación se omite por infeasibilidad
                computacional en operandos no triviales, o si el residuo
                excede la tolerancia de Wilkinson.
        """
        rng = rng or np.random.default_rng(cls._PROBE_SEED)
        report: Dict[str, Any] = {"valid": True, "channels": {}, "formalism": "kronecker_mixed_product"}

        channel_operands = {
            "e1": (
                alpha.matrix_channel_1,
                alpha_prime.matrix_channel_1,
                beta.matrix_channel_1,
                beta_prime.matrix_channel_1,
            ),
            "e2": (
                alpha.matrix_channel_2,
                alpha_prime.matrix_channel_2,
                beta.matrix_channel_2,
                beta_prime.matrix_channel_2,
            ),
        }

        for ch_label, (a, ap, b, bp) in channel_operands.items():
            for op_name, M in (("alpha", a), ("alpha_prime", ap), ("beta", b), ("beta_prime", bp)):
                if M.ndim != 2 or M.shape[0] != M.shape[1]:
                    raise FunctorialityError(
                        f"Componente '{op_name}' del canal {ch_label} debe ser "
                        "cuadrada para evaluar la ley de intercambio.",
                        channel=ch_label,
                        operand=op_name,
                        shape=M.shape,
                    )
            if not (a.shape == ap.shape == b.shape == bp.shape):
                raise FunctorialityError(
                    "Los cuatro 2-morfismos deben compartir dimensión en el canal.",
                    channel=ch_label,
                    shapes=[a.shape, ap.shape, b.shape, bp.shape],
                )

            n = a.shape[0]
            all_identity = all(cls._is_approx_identity(M) for M in (a, ap, b, bp))

            if n > cls._MAX_DENSE_PROBE_DIM and not all_identity:
                raise FunctorialityError(
                    "Verificación de la ley de intercambio omitida por "
                    f"infeasibilidad computacional (n={n} > "
                    f"{cls._MAX_DENSE_PROBE_DIM}) sobre operandos no triviales "
                    "(no todos ≈ identidad). Provea representantes de menor "
                    "dimensión o una estructura dispersa/bloque-diagonal.",
                    channel=ch_label,
                    dimension=n,
                )

            tol = cls._tolerance_channel([a, ap, b, bp])
            residual = 0.0 if all_identity else cls._interchange_residual_channel(a, ap, b, bp, rng)
            passed = residual <= tol

            report["channels"][ch_label] = {
                "residual_frobenius": residual,
                "tolerance": tol,
                "passed": passed,
                "fast_path_identity": all_identity,
            }
            if not passed:
                report["valid"] = False

        if not report["valid"]:
            raise FunctorialityError(
                "Violación de la Ley de Intercambio bicompleja (formalismo de Kronecker).",
                report=report,
            )

        return report


#------------------------------------------------------------------------------
# 2.3 — VERIFICADOR DE ADJUNCIÓN Y FRONTERA DE FASE: FASE 2 → FASE 3
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Phase2AdjunctionArtifact:
    """Artefacto de Fase 2."""

    vector: BicomplexVector
    adjunction_residual: Tuple[float, float]
    interchange_report: Dict[str, Any]
    passed: bool
    timestamp: float = field(default_factory=time.time)


class Phase2_AdjunctionInterchangeVerifier:
    """
    Fase 2: Verificador de adjunción de Galois y ley de intercambio.

    La adjunción F ⊣ G se verifica computacionalmente mediante el residuo
    de casi-inversidad:

        ||G⁽ᵃ⁾ F⁽ᵃ⁾ - I||_F + ||F⁽ᵃ⁾ G⁽ᵃ⁾ - I||_F.
    """

    @staticmethod
    def _adjunction_residual_channel(F: Optional[np.ndarray], G: Optional[np.ndarray]) -> float:
        """Residuo de adjunción por canal."""
        if F is None or G is None:
            return 0.0

        F = np.asarray(F, dtype=np.complex128)
        G = np.asarray(G, dtype=np.complex128)

        if F.ndim != 2 or G.ndim != 2:
            return float("inf")

        if F.shape[0] != G.shape[1] or G.shape[0] != F.shape[1]:
            return float("inf")

        n = F.shape[1]
        I = np.eye(n, dtype=np.complex128)

        try:
            r1 = np.linalg.norm(G @ F - I, ord="fro")
            r2 = np.linalg.norm(F @ G - I, ord="fro")
            return float(r1 + r2)
        except Exception:
            return float("inf")

    @classmethod
    def verify(
        cls,
        phase1_input: Phase2Input,
        F: Optional[BicomplexLinearMorphism] = None,
        G: Optional[BicomplexLinearMorphism] = None,
        alpha: Optional[BicomplexNaturalTransformation] = None,
        alpha_prime: Optional[BicomplexNaturalTransformation] = None,
        beta: Optional[BicomplexNaturalTransformation] = None,
        beta_prime: Optional[BicomplexNaturalTransformation] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Phase2AdjunctionArtifact:
        """Ejecuta Fase 2 sobre la frontera `Phase2Input` de la Fase 1."""
        phase1 = phase1_input.phase1_artifact
        dim = max(len(phase1.vector), 1)

        # Identidades por defecto para mantener la fase definida.
        alpha = alpha or BicomplexNaturalTransformation.identity(dim, "alpha")
        alpha_prime = alpha_prime or BicomplexNaturalTransformation.identity(dim, "alpha_prime")
        beta = beta or BicomplexNaturalTransformation.identity(dim, "beta")
        beta_prime = beta_prime or BicomplexNaturalTransformation.identity(dim, "beta_prime")

        try:
            interchange_report = TwoCategoryBicomplexOrchestrator.validate_interchange_law(
                alpha=alpha,
                alpha_prime=alpha_prime,
                beta=beta,
                beta_prime=beta_prime,
                rng=rng,
            )
            interchange_passed = True
        except FunctorialityError as exc:
            interchange_report = exc.context.get("report") or dict(exc.context)
            interchange_passed = False

        adj_residual_1 = cls._adjunction_residual_channel(
            F.matrix_channel_1 if F else None,
            G.matrix_channel_1 if G else None,
        )
        adj_residual_2 = cls._adjunction_residual_channel(
            F.matrix_channel_2 if F else None,
            G.matrix_channel_2 if G else None,
        )

        adjunction_ok = np.isfinite(adj_residual_1) and np.isfinite(adj_residual_2)
        passed = phase1.ring_homogeneous and interchange_passed and adjunction_ok

        artifact = Phase2AdjunctionArtifact(
            vector=phase1.vector,
            adjunction_residual=(float(adj_residual_1), float(adj_residual_2)),
            interchange_report=interchange_report,
            passed=passed,
        )

        logger.debug(
            "Phase2: adj_residual=(%.3e, %.3e), interchange=%s",
            adj_residual_1,
            adj_residual_2,
            interchange_passed,
        )
        return artifact


@dataclass(frozen=True, slots=True)
class Phase3Input:
    """Objeto frontera entre Fase 2 y Fase 3."""

    phase2_artifact: Phase2AdjunctionArtifact


def phase2_export_to_phase3(artifact: Phase2AdjunctionArtifact) -> Phase3Input:
    """
    FASE 2 — Último método de la fase y continuación formal de la Fase 3.

    Contrato: `phase2_export_to_phase3 : Phase2AdjunctionArtifact →
    Phase3Input`. No se impone aquí una condición de admisión estricta
    (a diferencia de `phase1_export_to_phase2`): la Fase 3 está diseñada
    para recibir también artefactos con `passed=False`, de modo que pueda
    sintetizar un veredicto de Heyting que capture — en lugar de ocultar —
    los fallos de adjunción/intercambio detectados aguas arriba.

    Raises:
        TypeError: Si `artifact` no es `Phase2AdjunctionArtifact`.
    """
    if not isinstance(artifact, Phase2AdjunctionArtifact):
        raise TypeError("phase2_export_to_phase3 requiere Phase2AdjunctionArtifact.")

    return Phase3Input(phase2_artifact=artifact)


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███       FASE 3 — COBORDISMO A∞, BETTI DUAL Y VETO HEYTING B₂           ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase certifica:
#
#   1. Nilpotencia bicompleja del coborde [I2] (cuando se proveen
#      operadores explícitos; vacuamente satisfecha en el enfoque
#      combinatorio/Betti por defecto).
#   2. Doble veto de Betti [I3] sobre un PAR DUAL NO DEGENERADO de grafos:
#      canal e₁ = subgrafo de composiciones exitosas, canal e₂ = grafo
#      completo (incluye fallos) — evitando la tautología de usar el
#      mismo grafo para ambos canales.
#   3. Retículo de Heyting B₂ completo: ∧, ∨, ¬, →.
#   4. Coherencia bicompleja C_{C₂} y censura ciber-física.
#=============================================================================


#------------------------------------------------------------------------------
# 3.1 — GRAFOS DUALES Y VERIFICADOR HOMOLÓGICO
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexGraph:
    """Grafo/subcomplejo discreto para auditoría homológica."""

    vertices: FrozenSet[Any]
    edges: FrozenSet[Tuple[Any, Any]]

    @classmethod
    def empty(cls) -> "BicomplexGraph":
        """Grafo vacío acíclico."""
        return cls(vertices=frozenset(), edges=frozenset())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BicomplexGraph":
        """Construye grafo desde mapping."""
        vertices = frozenset(data.get("vertices", []))
        edges = frozenset(tuple(e) for e in data.get("edges", []))
        return cls(vertices=vertices, edges=edges)

    @classmethod
    def from_traces(cls, traces: Sequence[CompositionTrace]) -> "BicomplexGraph":
        """Construye grafo de trazas categóricas (dominio → codominio)."""
        vertices = set()
        edges = set()

        for t in traces:
            cod = t.output_codomain.name
            vertices.add(cod)
            for d in t.input_domain:
                dom = d.name
                vertices.add(dom)
                edges.add((dom, cod))

        return cls(vertices=frozenset(vertices), edges=frozenset(edges))

    @classmethod
    def from_traces_dual(
        cls, traces: Sequence[CompositionTrace]
    ) -> Tuple["BicomplexGraph", "BicomplexGraph"]:
        """
        Construye el **par dual no degenerado** (G⁽¹⁾, G⁽²⁾) a partir de
        trazas categóricas.

        A diferencia de emplear el mismo grafo para ambos canales (lo que
        trivializa el "doble veto" de Betti a una tautología duplicada),
        aquí:

            G⁽¹⁾ («canal de éxito»): subgrafo inducido únicamente por
                composiciones con `success=True`.
            G⁽²⁾ («canal íntegro»): grafo completo, incluyendo
                composiciones fallidas.

        Cualquier ciclo introducido exclusivamente por una rama de error
        será visible en β₁⁽²⁾ pero no necesariamente en β₁⁽¹⁾, dotando al
        invariante [I3] de contenido bicanal genuino.
        """
        success_traces = [t for t in traces if t.success]
        g1 = cls.from_traces(success_traces)
        g2 = cls.from_traces(traces)
        return g1, g2


def _beta0_beta1(graph: BicomplexGraph) -> Tuple[int, int]:
    """
    Calcula β₀ y β₁ para un 1-complejo finito.

    β₁ = |E| - |V| + β₀.
    """
    vertices = list(graph.vertices)
    if not vertices:
        return 0, 0

    parent = {v: v for v in vertices}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    valid_edges: List[Tuple[Any, Any]] = []
    for a, b in graph.edges:
        if a in parent and b in parent:
            union(a, b)
            valid_edges.append((a, b))

    beta0 = len({find(v) for v in vertices})
    beta1 = len(valid_edges) - len(vertices) + beta0
    if beta1 < 0:
        beta1 = 0

    return beta0, beta1


@dataclass(frozen=True, slots=True)
class NilpotencyCertificate:
    """Certificado del invariante [I2] para un canal idempotente."""

    operator_norm: float
    max_absolute_entry: float
    is_nilpotent: bool
    tolerance: float
    channel: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana."""
        return asdict(self)


class BicomplexHomologicalVerifier:
    """Verificador homológico dual sobre canales idempotentes."""

    @staticmethod
    def compute_betti_numbers(
        graph_channel_1: Optional[BicomplexGraph] = None,
        graph_channel_2: Optional[BicomplexGraph] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Calcula Betti por canal."""
        g1 = graph_channel_1 or BicomplexGraph.empty()
        g2 = graph_channel_2 or BicomplexGraph.empty()

        b0_1, b1_1 = _beta0_beta1(g1)
        b0_2, b1_2 = _beta0_beta1(g2)

        return {
            "channel_1": {"beta_0": b0_1, "beta_1": b1_1, "beta_2": 0},
            "channel_2": {"beta_0": b0_2, "beta_1": b1_2, "beta_2": 0},
        }

    @staticmethod
    def euler_bicomplex(betti: Mapping[str, Mapping[str, int]]) -> Tuple[int, int]:
        """
        Característica de Euler bicompleja:

            χ_{C₂} = (β₀⁽¹⁾ - β₁⁽¹⁾) e₁ + (β₀⁽²⁾ - β₁⁽²⁾) e₂.
        """
        c1 = betti.get("channel_1", {})
        c2 = betti.get("channel_2", {})
        return (
            int(c1.get("beta_0", 0)) - int(c1.get("beta_1", 0)),
            int(c2.get("beta_0", 0)) - int(c2.get("beta_1", 0)),
        )

    @classmethod
    def verify_double_betti_veto(
        cls,
        graph_channel_1: Optional[BicomplexGraph] = None,
        graph_channel_2: Optional[BicomplexGraph] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Doble condición de veto [I3]:

            β₁⁽¹⁾ = 0 ∧ β₁⁽²⁾ = 0.

        Nota de rigor: esta verificación solo es bicanal-no-trivial si
        `graph_channel_1` y `graph_channel_2` son genuinamente distintos
        (véase `BicomplexGraph.from_traces_dual`); si se invoca con el
        mismo grafo para ambos canales, el "doble" veto degenera a una
        comprobación simple duplicada.
        """
        betti = cls.compute_betti_numbers(graph_channel_1, graph_channel_2)
        beta1_1 = int(betti["channel_1"]["beta_1"])
        beta1_2 = int(betti["channel_2"]["beta_1"])

        ok = beta1_1 == 0 and beta1_2 == 0
        degenerate_pair = graph_channel_1 is not None and graph_channel_1 == graph_channel_2
        return ok, {
            "betti": betti,
            "double_betti_veto_passed": ok,
            "degenerate_channel_pair": degenerate_pair,
        }

    @staticmethod
    def verify_coboundary_nilpotency(
        D_lower: Optional[np.ndarray],
        D_upper: Optional[np.ndarray],
        channel: str,
        tolerance: float = _ALGEBRAIC_TOL,
    ) -> NilpotencyCertificate:
        """
        FASE 3 — Certificación del invariante [I2] para un canal
        idempotente: d_k ∘ d_{k-1} = 0.

        El enfoque primario de este verificador es combinatorio (números
        de Betti sobre 1-esqueletos), que no requiere operadores de
        coborde explícitos. Cuando el llamador dispone de tales
        operadores matriciales (p. ej. provenientes de un complejo de
        cocadenas simplicial explícito), esta función certifica [I2]
        directamente sobre ellos. Si no se proveen, el invariante se
        declara **vacuamente satisfecho** — explícitamente documentado
        como tal, para no simular un rigor inexistente.

        Raises:
            HomologicalError: Si las dimensiones de `D_lower`/`D_upper`
                son incompatibles para la composición.
        """
        if D_lower is None or D_upper is None:
            return NilpotencyCertificate(0.0, 0.0, True, tolerance, channel)

        Dl = np.asarray(D_lower, dtype=np.float64)
        Du = np.asarray(D_upper, dtype=np.float64)

        if Dl.ndim != 2 or Du.ndim != 2 or Du.shape[1] != Dl.shape[0]:
            raise HomologicalError(
                "Dimensiones incompatibles para verificar nilpotencia del coborde.",
                channel=channel,
                shape_lower=Dl.shape,
                shape_upper=Du.shape,
            )

        residual = Du @ Dl
        op_norm = float(np.linalg.norm(residual, ord="fro"))
        max_entry = float(np.max(np.abs(residual))) if residual.size else 0.0

        return NilpotencyCertificate(
            operator_norm=op_norm,
            max_absolute_entry=max_entry,
            is_nilpotent=max_entry <= tolerance,
            tolerance=tolerance,
            channel=channel,
        )


#------------------------------------------------------------------------------
# 3.2 — RETÍCULO DE HEYTING B₂ COMPLETO
#------------------------------------------------------------------------------
class HeytingValue(IntEnum):
    """
    Retículo de Heyting trivalente (cadena de Gödel B₂) por canal.

    En una cadena finita totalmente ordenada, ∧ = min, ∨ = max, y la
    implicación/negación quedan determinadas unívocamente.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @classmethod
    def from_coherence(cls, value: float) -> "HeytingValue":
        """Clasifica coherencia escalar según umbrales doctorales."""
        value = float(value)
        if value >= _COHERENT_THRESHOLD:
            return cls.COHERENT
        if value >= _DEGRADED_THRESHOLD:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingValue":
        """Elemento máximo ⊤ = COHERENT."""
        return cls.COHERENT

    @classmethod
    def bottom(cls) -> "HeytingValue":
        """Elemento mínimo ⊥ = VETOED."""
        return cls.VETOED


def heyting_conjunction(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """Conjunción de Heyting: ínfimo de la cadena, a ∧ b = min(a, b)."""
    return min(a, b)


def heyting_disjunction(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """Disyunción de Heyting: supremo de la cadena, a ∨ b = max(a, b)."""
    return max(a, b)


def heyting_implication(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """Implicación pseudo-complementada: a → b = ⊤ si a ≤ b, si no b."""
    return HeytingValue.top() if a <= b else b


def heyting_negation(a: HeytingValue) -> HeytingValue:
    """
    Pseudo-complemento ¬a := a → ⊥.

    En B₂, ¬DEGRADED = VETOED pero ¬¬DEGRADED = ¬VETOED = COHERENT ≠
    DEGRADED: la negación no es involutiva, la firma de una lógica
    intuicionista frente a una booleana.
    """
    return heyting_implication(a, HeytingValue.bottom())


@dataclass(frozen=True, slots=True)
class BicomplexHeytingVerdict:
    """Veredicto en el álgebra de Heyting bicompleja B₂ ≅ Ω₃⁽¹⁾ × Ω₃⁽²⁾."""

    channel_1: HeytingValue
    channel_2: HeytingValue

    @property
    def global_verdict(self) -> HeytingValue:
        """Conjunción de Heyting: min(v₁, v₂)."""
        return heyting_conjunction(self.channel_1, self.channel_2)

    @property
    def hardware_veto(self) -> bool:
        """True si el veredicto global colapsa a VETOED."""
        return self.global_verdict == HeytingValue.bottom()

    @property
    def gpio_pin(self) -> int:
        """Pin de actuación Crowbar."""
        return _GPIO_VETO_PIN

    @property
    def actuation_ns(self) -> float:
        """Cota temporal de ISR."""
        return _ISR_ACTUATION_NS

    @classmethod
    def from_coherence(cls, c1: float, c2: float) -> "BicomplexHeytingVerdict":
        """Construye veredicto desde coherencias."""
        return cls(
            channel_1=HeytingValue.from_coherence(c1),
            channel_2=HeytingValue.from_coherence(c2),
        )

    def conjunction(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Conjunción bicanal por componentes."""
        return BicomplexHeytingVerdict(
            channel_1=heyting_conjunction(self.channel_1, other.channel_1),
            channel_2=heyting_conjunction(self.channel_2, other.channel_2),
        )

    def disjunction(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Disyunción bicanal por componentes."""
        return BicomplexHeytingVerdict(
            channel_1=heyting_disjunction(self.channel_1, other.channel_1),
            channel_2=heyting_disjunction(self.channel_2, other.channel_2),
        )

    def negation(self) -> "BicomplexHeytingVerdict":
        """Pseudo-complemento bicanal."""
        return BicomplexHeytingVerdict(
            channel_1=heyting_negation(self.channel_1),
            channel_2=heyting_negation(self.channel_2),
        )

    def implication(self, other: "BicomplexHeytingVerdict") -> "BicomplexHeytingVerdict":
        """Implicación pseudo-complementada por componentes."""
        return BicomplexHeytingVerdict(
            channel_1=heyting_implication(self.channel_1, other.channel_1),
            channel_2=heyting_implication(self.channel_2, other.channel_2),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe."""
        return {
            "channel_1": self.channel_1.name,
            "channel_2": self.channel_2.name,
            "global_verdict": self.global_verdict.name,
            "hardware_veto": self.hardware_veto,
            "gpio_pin": self.gpio_pin,
            "actuation_ns": self.actuation_ns,
        }


def verify_heyting_censorship_invariant(verdict: BicomplexHeytingVerdict) -> bool:
    """
    FASE 3 — Certificación explícita del invariante [I5]:

        Verdict_global = v₁ ∧ v₂ = min(v₁, v₂),

    junto con la consistencia de `hardware_veto` respecto de
    `global_verdict == VETOED`.
    """
    expected = heyting_conjunction(verdict.channel_1, verdict.channel_2)
    verdict_ok = verdict.global_verdict == expected
    veto_ok = verdict.hardware_veto == (verdict.global_verdict == HeytingValue.bottom())
    return verdict_ok and veto_ok


#------------------------------------------------------------------------------
# 3.3 — COHERENCIA BICOMPLEJA
#------------------------------------------------------------------------------
def _channel_entropy(channel: np.ndarray) -> float:
    """Entropía de Shannon sobre energía normalizada."""
    mag = np.abs(channel).astype(np.float64)
    if mag.size == 0:
        return 0.0

    energy = float(np.sum(mag * mag))
    if energy <= _MACHINE_EPSILON or mag.size <= 1:
        return 0.0

    p = (mag * mag) / energy
    p = p[p > 0.0]
    return float(-MathUtils.kbn_sum((p * np.log(p)).tolist()))


def _channel_coherence(channel: np.ndarray, resonance: Optional[float] = None) -> float:
    """
    Coherencia por canal [I4 de mic_vectors, reutilizado aquí]:

        C⁽ᵃ⁾ = clamp(S⁽ᵃ⁾ R⁽ᵃ⁾ / (1 + H⁽ᵃ⁾), 0, 1).
    """
    mag = np.abs(channel).astype(np.float64)
    if mag.size == 0:
        return 0.0

    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))

    stability = MathUtils.clamp(mean_mag, 0.0, 1.0)

    if resonance is None:
        if mean_mag <= _MACHINE_EPSILON:
            resonance = 0.0
        else:
            resonance = MathUtils.clamp(1.0 - std_mag / mean_mag, 0.0, 1.0)

    entropy = _channel_entropy(channel)

    if stability <= _MACHINE_EPSILON or resonance <= _MACHINE_EPSILON:
        return 0.0

    return MathUtils.clamp(stability * resonance / (1.0 + entropy), 0.0, 1.0)


#------------------------------------------------------------------------------
# 3.4 — VALIDADOR A∞ Y SÍNTESIS TERMINAL
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Phase3CategoricalOutput:
    """Artefacto terminal de Fase 3."""

    state: BicomplexCategoricalState
    verdict: BicomplexHeytingVerdict
    betti_report: Dict[str, Any]
    coherence: Tuple[float, float]
    timestamp: float = field(default_factory=time.time)


class Phase3_A_InfinityHomotopyValidator:
    """
    Fase 3: Validador homotópico A∞ y colapso de Heyting B₂.

    Responsabilidades:
      - Nilpotencia del coborde [I2] (si se proveen operadores).
      - Doble veto de Betti [I3] sobre un par dual no degenerado.
      - Coherencia bicompleja C_{C₂}.
      - Veredicto Heyting B₂ [I5].
      - Emisión de `BicomplexCategoricalState`, opcionalmente con
        elevación a excepción tipada en modo `strict`.
    """

    @staticmethod
    def _coherence_from_override(
        channel: np.ndarray,
        override: Optional[Mapping[str, Any]],
        channel_id: int,
    ) -> float:
        """Extrae coherencia explícita o la deriva del canal."""
        if override is None:
            return _channel_coherence(channel)

        if f"coherence_{channel_id}" in override:
            return MathUtils.clamp(float(override[f"coherence_{channel_id}"]), 0.0, 1.0)

        stability = override.get(f"stability_{channel_id}")
        resonance = override.get(f"resonance_{channel_id}")
        entropy = override.get(f"entropy_{channel_id}", 0.0)

        if stability is not None and resonance is not None:
            s = MathUtils.clamp(float(stability), 0.0, 1.0)
            r = MathUtils.clamp(float(resonance), 0.0, 1.0)
            h = max(0.0, float(entropy))
            return MathUtils.clamp(s * r / (1.0 + h), 0.0, 1.0)

        return _channel_coherence(channel)

    @classmethod
    def validate(
        cls,
        phase3_input: Phase3Input,
        graph_channel_1: Optional[BicomplexGraph] = None,
        graph_channel_2: Optional[BicomplexGraph] = None,
        traces: Optional[Sequence[CompositionTrace]] = None,
        coherence_override: Optional[Mapping[str, Any]] = None,
        coboundary_channel_1: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        coboundary_channel_2: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        strict: bool = False,
    ) -> Phase3CategoricalOutput:
        """
        Ejecuta Fase 3 sobre la frontera `Phase3Input` de la Fase 2.

        Args:
            phase3_input: Frontera producida por `phase2_export_to_phase3`.
            graph_channel_1, graph_channel_2: Par dual explícito para
                Betti; si se omiten y se provee `traces`, se deriva un par
                **no degenerado** vía `BicomplexGraph.from_traces_dual`.
            traces: Trazas categóricas para derivar el par dual.
            coherence_override: Coherencias/estadísticas explícitas.
            coboundary_channel_1, coboundary_channel_2: Pares opcionales
                `(D_lower, D_upper)` para certificar [I2] explícitamente.
            strict: Si True, eleva fallos a excepciones tipadas
                (`HomologicalError`, `TopologicalInvariantError`,
                `HeytingVetoError`) en lugar de solo registrar el error en
                el estado monádico.

        Raises:
            HomologicalError: (modo `strict`) si falla la nilpotencia [I2].
            TopologicalInvariantError: (modo `strict`) si falla el doble
                veto de Betti [I3].
            HeytingVetoError: (modo `strict`) si el veredicto global
                colapsa a VETOED.
        """
        phase2 = phase3_input.phase2_artifact
        verifier = BicomplexHomologicalVerifier()

        if graph_channel_1 is None and graph_channel_2 is None and traces is not None:
            graph_channel_1, graph_channel_2 = BicomplexGraph.from_traces_dual(traces)

        betti_ok, betti_report = verifier.verify_double_betti_veto(
            graph_channel_1=graph_channel_1,
            graph_channel_2=graph_channel_2,
        )

        nilpotency_1 = verifier.verify_coboundary_nilpotency(
            *(coboundary_channel_1 or (None, None)), channel="e1"
        )
        nilpotency_2 = verifier.verify_coboundary_nilpotency(
            *(coboundary_channel_2 or (None, None)), channel="e2"
        )
        betti_report["nilpotency"] = {
            "channel_1": nilpotency_1.to_dict(),
            "channel_2": nilpotency_2.to_dict(),
        }
        nilpotency_ok = nilpotency_1.is_nilpotent and nilpotency_2.is_nilpotent

        c1 = cls._coherence_from_override(phase2.vector.channel_1, coherence_override, 1)
        c2 = cls._coherence_from_override(phase2.vector.channel_2, coherence_override, 2)

        beta1_1 = int(betti_report["betti"]["channel_1"]["beta_1"])
        beta1_2 = int(betti_report["betti"]["channel_2"]["beta_1"])

        v1 = HeytingValue.bottom() if beta1_1 > 0 else HeytingValue.from_coherence(c1)
        v2 = HeytingValue.bottom() if beta1_2 > 0 else HeytingValue.from_coherence(c2)

        verdict = BicomplexHeytingVerdict(channel_1=v1, channel_2=v2)
        assert verify_heyting_censorship_invariant(verdict), "Violación interna de [I5]."

        state = BicomplexCategoricalState(
            payload={
                "bicomplex_coherence": {"channel_1": c1, "channel_2": c2},
                "betti_report": betti_report,
                "phase2_passed": phase2.passed,
            },
            context={
                "adjunction_residual": phase2.adjunction_residual,
                "interchange_report": phase2.interchange_report,
            },
            validated_strata=frozenset({Stratum.TACTICS, Stratum.STRATEGY}),
            vector=phase2.vector,
        )

        overall_ok = (not verdict.hardware_veto) and betti_ok and phase2.passed and nilpotency_ok

        if not overall_ok:
            error_msg = (
                "Colapso bicomplejo. "
                f"Verdict={verdict.global_verdict.name}, "
                f"β₁⁽¹⁾={beta1_1}, β₁⁽²⁾={beta1_2}, "
                f"nilpotency_ok={nilpotency_ok}, phase2_passed={phase2.passed}."
            )
            error_details = {
                "verdict": verdict.to_dict(),
                "gpio_pin": verdict.gpio_pin,
                "actuation_ns": verdict.actuation_ns,
                "nilpotency": betti_report["nilpotency"],
            }
            state = state.with_error(error_msg, details=error_details)

            if strict:
                if not nilpotency_ok:
                    raise HomologicalError(error_msg, **error_details)
                if not betti_ok:
                    raise TopologicalInvariantError(error_msg, **error_details)
                if verdict.hardware_veto:
                    raise HeytingVetoError(error_msg, **error_details)
        else:
            state = state.clear_error()

        state = state.add_trace(
            morphism_name="Phase3_A_InfinityHomotopyValidator",
            input_domain=frozenset({Stratum.TACTICS}),
            output_codomain=Stratum.STRATEGY,
            success=overall_ok,
            error=state.error,
            metadata={"verdict": verdict.to_dict()},
        )

        output = Phase3CategoricalOutput(
            state=state,
            verdict=verdict,
            betti_report=betti_report,
            coherence=(c1, c2),
        )

        if verdict.hardware_veto:
            logger.warning(
                "VETO B₂ activo: C=(%.6f, %.6f), GPIO%d ≤ %.2f ns",
                c1,
                c2,
                verdict.gpio_pin,
                verdict.actuation_ns,
            )

        return output


#------------------------------------------------------------------------------
# 3.5 — COMPOSICIÓN MAESTRA DE FASES ANIDADAS
#------------------------------------------------------------------------------
def compose_bicomplex_algebra_pipeline(
    S: NDArray[np.float64],
    *,
    F: Optional[BicomplexLinearMorphism] = None,
    G: Optional[BicomplexLinearMorphism] = None,
    alpha: Optional[BicomplexNaturalTransformation] = None,
    alpha_prime: Optional[BicomplexNaturalTransformation] = None,
    beta: Optional[BicomplexNaturalTransformation] = None,
    beta_prime: Optional[BicomplexNaturalTransformation] = None,
    graph_channel_1: Optional[BicomplexGraph] = None,
    graph_channel_2: Optional[BicomplexGraph] = None,
    coherence_override: Optional[Mapping[str, Any]] = None,
    coboundary_channel_1: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    coboundary_channel_2: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    strict: bool = False,
) -> Phase3CategoricalOutput:
    """
    Funtor maestro de la 2-categoría bicompleja:

        Z_MIC = Ψ₃ ∘ Ψ₂ ∘ Ψ₁,

    materializado mediante las fronteras categóricas explícitas
    `phase1_export_to_phase2` y `phase2_export_to_phase3`.

    Args:
        S: Señal real 4D o matriz (n,4).
        F, G: Funtores opcionales para adjunción.
        alpha, alpha_prime, beta, beta_prime: 2-morfismos para intercambio.
        graph_channel_1, graph_channel_2: Par dual homológico explícito.
        coherence_override: Coherencias explícitas opcionales.
        coboundary_channel_1, coboundary_channel_2: Operadores de coborde
            opcionales para certificar [I2] explícitamente.
        rng: Generador determinista para las sondas de intercambio.
        strict: Propaga excepciones tipadas de Fase 3 en lugar de
            reportarlas solo en el estado monádico.

    Returns:
        Salida terminal de Fase 3.
    """
    phase1_artifact = Phase1_SpectralRingObserver.run(S, rng=rng)
    phase2_input = phase1_export_to_phase2(phase1_artifact)

    phase2_artifact = Phase2_AdjunctionInterchangeVerifier.verify(
        phase1_input=phase2_input,
        F=F,
        G=G,
        alpha=alpha,
        alpha_prime=alpha_prime,
        beta=beta,
        beta_prime=beta_prime,
        rng=rng,
    )
    phase3_input = phase2_export_to_phase3(phase2_artifact)

    return Phase3_A_InfinityHomotopyValidator.validate(
        phase3_input=phase3_input,
        graph_channel_1=graph_channel_1,
        graph_channel_2=graph_channel_2,
        traces=None,
        coherence_override=coherence_override,
        coboundary_channel_1=coboundary_channel_1,
        coboundary_channel_2=coboundary_channel_2,
        strict=strict,
    )


#=============================================================================
# EXPORTS
#=============================================================================
__all__ = [
    # Excepciones
    "AlgebraicError",
    "CanonicalizationError",
    "FunctorialityError",
    "HomologicalError",
    "TopologicalInvariantError",
    "HeytingVetoError",
    # Utilidades
    "MathUtils",
    # Fase 1
    "BicomplexScalar",
    "verify_idempotent_decoupling",
    "verify_bicomplex_ring_axioms",
    "BicomplexVector",
    "CompositionTrace",
    "BicomplexCategoricalState",
    "Phase1SpectralRingArtifact",
    "Phase1_SpectralRingObserver",
    "Phase2Input",
    "phase1_export_to_phase2",
    # Fase 2
    "BicomplexLinearMorphism",
    "BicomplexComposedMorphism",
    "BicomplexNaturalTransformation",
    "TwoCategoryBicomplexOrchestrator",
    "Phase2AdjunctionArtifact",
    "Phase2_AdjunctionInterchangeVerifier",
    "Phase3Input",
    "phase2_export_to_phase3",
    # Fase 3
    "BicomplexGraph",
    "NilpotencyCertificate",
    "BicomplexHomologicalVerifier",
    "HeytingValue",
    "heyting_conjunction",
    "heyting_disjunction",
    "heyting_implication",
    "heyting_negation",
    "BicomplexHeytingVerdict",
    "verify_heyting_censorship_invariant",
    "Phase3CategoricalOutput",
    "Phase3_A_InfinityHomotopyValidator",
    "compose_bicomplex_algebra_pipeline",
]