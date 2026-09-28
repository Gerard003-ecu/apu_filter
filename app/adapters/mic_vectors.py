# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Vectors — Evolución Bicompleja                                  ║
║ Ruta   : app/adapters/mic_vectors.py                                         ║
║ Versión: 5.1.0-Bicomplex-Heyting-Hodge-MayerVietoris-Doctoral-Certified      ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETO DEL MÓDULO
─────────────────
Este módulo materializa la transición categorial y espectral de la MIC desde
cocadenas reales/escalares hacia la teoría de cocadenas simpliciales bicomplejas

    C^k(K; C₂) = C^k(K; C) ⊗_R C₂ ≅ C^k(K; C) e₁ ⊕ C^k(K; C) e₂,

donde C₂ = C ⊗_R C es el álgebra conmutativa de números bicomplejos, con
unidad hiperbólica j² = +1, y base idempotente

    e₁ = (1 + j)/2,   e₂ = (1 - j)/2,
    e₁² = e₁, e₂² = e₂, e₁e₂ = 0, e₁ + e₂ = 1.

C₂ **no** es un dominio de integridad: posee divisores de cero exactamente
en los elementos Z tales que Z⁽¹⁾ = 0 xor Z⁽²⁾ = 0. Esta propiedad se explota
para diagonalizar todo operador C₂-lineal en dos operadores C-lineales
independientes, uno por canal idempotente — el principio estructural que
gobierna las tres fases de este módulo.

El módulo queda organizado en TRES FASES ANIDADAS:

    FASE 1 — PROYECCIÓN BICOMPLEJA Y SANEAMIENTO DIMENSIONAL (Observe)
        Núcleo algebraico C₂ con operaciones certificables, ingesta 4D,
        mapeo Z = z₁ + z₂ j, proyección idempotente, saneamiento FPU de
        ceros signados, certificación bilateral de Banach y sellado/
        verificación criptográfica HMAC-SHA256.

    FASE 2 — ANÁLISIS HOMOLÓGICO Y ESPECTRAL BICOMPLEJO (Orient)
        Auditoría Mayer-Vietoris dual, Laplaciano de Hodge discreto
        bicomplejo con saneamiento espectral PSD, certificación de
        nilpotencia del coborde, certificación de la descomposición de
        Hodge-Kodaira, conectividad de Fiedler dual, suma compensada KBN
        y síntesis de un índice de defecto topológico por canal.

    FASE 3 — SÍNTESIS DE COHERENCIA, HEYTING Y CENSURA CIBER-FÍSICA (Decide/Act)
        Retículo de Heyting B₂ completo (conjunción, disyunción,
        implicación, pseudo-complemento), índice de coherencia bicomplejo
        C_{C₂} penalizado por defectos topológicos heredados de Fase 2,
        veto dual, verificación explícita de invariantes, emisión de
        certificados y adaptación a VectorResult.

CONTINUIDAD DE FASES
────────────────────
La última definición formal de la FASE 1, `phase1_export_to_phase2`, constituye
la frontera categórica exacta cuya imagen es el dominio canónico de la FASE 2.
Análogamente, la última definición formal de la FASE 2, `phase2_export_to_phase3`,
es la frontera cuya imagen (`Phase3Input`) es el dominio canónico de la FASE 3,
y es **efectivamente consumida** por `vector_compute_bicomplex_coherence_index`
(a diferencia de versiones previas, donde esta frontera era observacionalmente
inerte).

INVARIANTES PRESERVADOS Y CERTIFICADOS
───────────────────────────────────────
[I1] Nilpotencia del coborde:
        d_{k+1} ∘ d_k = 0.                          → `verify_coboundary_nilpotency`
[I2] Exactitud Mayer-Vietoris dual:
        Δβ₁⁽¹⁾ = 0 ∧ Δβ₁⁽²⁾ = 0.                    → `vector_audit_bicomplex_mayer_vietoris`
[I3] Hodge-Kodaira discreto:
        C^k = im(d_{k-1}) ⊕ im(d_k^T) ⊕ ker(L_k).   → `verify_hodge_kodaira_decomposition`
[I4] Coherencia continua por canal:
        C⁽ᵃ⁾ = clamp(S⁽ᵃ⁾ R⁽ᵃ⁾ / (1 + H⁽ᵃ⁾), 0, 1).  → `verify_coherence_formula_invariant`
[I5] Censura de Heyting bicompleja:
        Verdict_global = v₁ ∧ v₂ = min(v₁, v₂).      → `verify_heyting_censorship_invariant`
[I6] Diagonalización multiplicativa idempotente:
        (Z · W)⁽ᵃ⁾ = Z⁽ᵃ⁾ · W⁽ᵃ⁾,  a ∈ {1, 2}.       → `BicomplexScalar.__mul__`
[I7] Sanidad espectral PSD:
        λ_i(L_k) ∈ (-ε, 0) ⇒ λ_i(L_k) ← 0.           → `_clip_psd_spectrum`
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field, asdict
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

logger = logging.getLogger("mic_vectors.bicomplex")

#=============================================================================
# COMPATIBILIDAD DE ESTRATOS
#=============================================================================
try:
    from app.core.schemas import Stratum  # type: ignore
except Exception:  # pragma: no cover - fallback standalone
    class Stratum(Enum):
        """Estratos funcionales de la arquitectura MIC/MAC."""
        PHYSICS = "physics"
        TACTICS = "tactics"
        STRATEGY = "strategy"


#=============================================================================
# CONSTANTES MAESTRAS
#=============================================================================
class BicomplexConstants:
    """Constantes matemáticas, metrológicas y ciber-físicas del módulo."""

    EPSILON: Final[float] = 1e-12
    HARMONIC_EPSILON: Final[float] = 1e-10
    FIEDLER_EPSILON: Final[float] = 1e-6

    COHERENT_THRESHOLD: Final[float] = 0.85
    DEGRADED_THRESHOLD: Final[float] = 0.50

    GPIO_VETO_PIN: Final[int] = 14
    ISR_ACTUATION_NS: Final[float] = 398.95

    HMAC_KEY_ENV: Final[str] = "MIC_BICOMPLEX_HMAC_KEY"
    DEFAULT_HMAC_KEY: Final[bytes] = b"MIC-BICOMPLEX-DEFAULT-KEY-CHANGE-IN-PRODUCTION"

    # Cotas de la equivalencia de normas ℓ¹/ℓ² en R^d, d = 4:
    #     1 ≤ ||x||_1 / ||x||_2 ≤ √d.
    BANACH_LOWER_BOUND: Final[float] = 1.0
    BANACH_UPPER_BOUND: Final[float] = float(np.sqrt(4.0))


class VectorResultStatus(Enum):
    """Estados canónicos de un VectorResult."""

    SUCCESS = "success"
    PHYSICS_ERROR = "physics_error"
    LOGIC_ERROR = "logic_error"
    TOPOLOGY_ERROR = "topology_error"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_ERROR = "dependency_error"


#=============================================================================
# MÉTRICAS Y RESULTADOS COMPATIBLES CON LA MIC ORIGINAL
#=============================================================================
@dataclass(frozen=True, slots=True)
class VectorMetrics:
    """
    Métricas inmutables de ejecución.

    Attributes:
        processing_time_ms: Tiempo total de procesamiento, ≥ 0.
        memory_usage_mb: Memoria de referencia (0.0 en modo standalone).
        topological_coherence: Coherencia topológica C ∈ [0, 1].
        algebraic_integrity: Integridad algebraica I ∈ [0, 1].
    """

    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    topological_coherence: float = 1.0
    algebraic_integrity: float = 1.0

    def __post_init__(self) -> None:
        if self.processing_time_ms < 0.0:
            object.__setattr__(self, "processing_time_ms", 0.0)
        if self.memory_usage_mb < 0.0:
            object.__setattr__(self, "memory_usage_mb", 0.0)

        tc = max(0.0, min(1.0, float(self.topological_coherence)))
        ai = max(0.0, min(1.0, float(self.algebraic_integrity)))
        object.__setattr__(self, "topological_coherence", tc)
        object.__setattr__(self, "algebraic_integrity", ai)

    def combine(self, other: "VectorMetrics") -> "VectorMetrics":
        """
        Composición de métricas por principio de cuello de botella.

        - Tiempos se suman (composición secuencial de procesos).
        - Memoria toma el máximo (uso pico, no acumulativo).
        - Coherencia e integridad toman el mínimo (el eslabón más débil
          domina la certificación global, en analogía con la norma del
          supremo en Banach C(X)).
        """
        return VectorMetrics(
            processing_time_ms=self.processing_time_ms + other.processing_time_ms,
            memory_usage_mb=max(self.memory_usage_mb, other.memory_usage_mb),
            topological_coherence=min(self.topological_coherence, other.topological_coherence),
            algebraic_integrity=min(self.algebraic_integrity, other.algebraic_integrity),
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialización plana."""
        return asdict(self)


VectorResult = Dict[str, Any]


def _build_result(
    *,
    success: bool,
    stratum: Stratum,
    status: VectorResultStatus,
    metrics: Optional[VectorMetrics] = None,
    error: Optional[str] = None,
    **payload: Any,
) -> VectorResult:
    """
    Constructor canónico de `VectorResult`.

    Garantiza el esquema mínimo:
        {success, stratum, status, metrics, [error], **payload}
    """
    result: VectorResult = {
        "success": success,
        "stratum": stratum.name if hasattr(stratum, "name") else str(stratum),
        "status": status.value,
        "metrics": (metrics or VectorMetrics()).to_dict(),
    }
    if error is not None:
        result["error"] = error
    result.update(payload)
    return result


def _build_success(
    stratum: Stratum,
    metrics: VectorMetrics,
    **payload: Any,
) -> VectorResult:
    """Constructor canónico de éxito."""
    return _build_result(
        success=True,
        stratum=stratum,
        status=VectorResultStatus.SUCCESS,
        metrics=metrics,
        **payload,
    )


def _build_error(
    *,
    stratum: Stratum,
    status: VectorResultStatus,
    error: str,
    metrics: Optional[VectorMetrics] = None,
    **payload: Any,
) -> VectorResult:
    """Constructor canónico de error."""
    return _build_result(
        success=False,
        stratum=stratum,
        status=status,
        metrics=metrics,
        error=error,
        **payload,
    )


#=============================================================================
# UTILIDADES NUMÉRICAS Y METROLÓGICAS
#=============================================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp numérico estricto sobre el cuerpo real."""
    return max(lo, min(hi, float(x)))


def _sanitize_float(x: float) -> float:
    """
    Saneamiento FPU de ceros signados.

    Transforma -0.0 → +0.0, preservando el resto de la mantisa IEEE-754.
    Es indispensable antes de cualquier hash criptográfico, pues -0.0 y +0.0
    son bit-patterns distintos aunque matemáticamente idénticos, lo que
    rompería la determinación del sello HMAC frente a entradas equivalentes.
    """
    x = float(x)
    if x == 0.0:
        return 0.0
    return x


def _sanitize_complex(z: complex) -> complex:
    """Saneamiento FPU de parte real e imaginaria."""
    return complex(_sanitize_float(z.real), _sanitize_float(z.imag))


def _kbn_sum(values: Sequence[float]) -> float:
    """
    Sumación compensada de Kahan-Babuška-Neumaier (KBN).

    Aniquila la deriva secular de Wilkinson en acumulaciones flotantes,
    manteniendo el error residual en O(ε) independientemente de n, frente
    al O(n·ε) de la sumación ingenua.
    """
    s = 0.0
    c = 0.0
    for v in values:
        v = float(v)
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


def _hmac_sha256(payload: np.ndarray) -> str:
    """
    Sello criptográfico HMAC-SHA256 sobre el payload canónico.

    La clave se toma de `MIC_BICOMPLEX_HMAC_KEY` si existe; en caso contrario
    se usa una clave de desarrollo y se emite una advertencia. En producción
    debe inyectarse una clave gestionada por secreto de entorno o HSM.
    """
    env_key = os.environ.get(BicomplexConstants.HMAC_KEY_ENV, "").encode("utf-8")
    if not env_key:
        logger.warning(
            "Variable de entorno '%s' no definida; usando clave HMAC de "
            "desarrollo. NO USAR en producción.",
            BicomplexConstants.HMAC_KEY_ENV,
        )
    key = env_key or BicomplexConstants.DEFAULT_HMAC_KEY
    return hmac.new(key, payload.astype(np.float64).tobytes(), hashlib.sha256).hexdigest()


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███  FASE 1 — PROYECCIÓN BICOMPLEJA Y SANEAMIENTO DIMENSIONAL (Observe)  ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase consagra el morfismo de ingestión:
#
#     Φ₁ : R⁴ → C₂,
#     Φ₁(s₀, s₁, s₂, s₃) = (s₀ + i s₁) + (s₂ + i s₃) j.
#
# En base idempotente:
#
#     Z⁽¹⁾ = (s₀ + s₂) + i(s₁ + s₃),
#     Z⁽²⁾ = (s₀ - s₂) + i(s₁ - s₃).
#
# La fase verifica además la equivalencia bilateral de normas de Banach:
#
#     1 ≤ ||Z||₁ / ||Z||₂ ≤ √d,   d = 4,
#
# y certifica formalmente que la representación diagonal (z1, z2) satisface
# las identidades algebraicas de la base idempotente de C₂.
#=============================================================================


#------------------------------------------------------------------------------
# 1.1 — NÚCLEO ALGEBRAICO BICOMPLEJO C₂
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexScalar:
    """
    Número bicomplejo representado en la base idempotente.

    Se almacena como par de componentes complejas independientes:

        Z = Z⁽¹⁾ e₁ + Z⁽²⁾ e₂.

    Esta representación diagonaliza el álgebra: suma, producto, y las dos
    conjugaciones naturales de C₂ (respecto de i y respecto de j) actúan
    canal a canal, evitando la propagación de divisores de cero entre
    canales — la firma algebraica distintiva de C₂ frente a C.

    Attributes:
        z1: Componente en el canal idempotente e₁.
        z2: Componente en el canal idempotente e₂.
    """

    z1: complex
    z2: complex

    # -- Construcción -------------------------------------------------------
    @classmethod
    def from_real4(cls, vector: Sequence[float]) -> "BicomplexScalar":
        """
        Construye Z ∈ C₂ desde un vector real 4D.

        Args:
            vector: Secuencia (s₀, s₁, s₂, s₃).

        Returns:
            BicomplexScalar con proyecciones idempotentes saneadas.

        Raises:
            ValueError: Si el vector no tiene exactamente 4 componentes.
        """
        if len(vector) != 4:
            raise ValueError(f"Se requiere vector 4D; recibido len={len(vector)}.")

        s0, s1, s2, s3 = (_sanitize_float(v) for v in vector)

        z1 = complex(s0 + s2, s1 + s3)
        z2 = complex(s0 - s2, s1 - s3)

        return cls(_sanitize_complex(z1), _sanitize_complex(z2))

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
        """Elemento neutro aditivo 0 ∈ C₂."""
        return cls(complex(0.0), complex(0.0))

    @classmethod
    def one(cls) -> "BicomplexScalar":
        """Elemento neutro multiplicativo 1 = e₁ + e₂ ∈ C₂."""
        return cls(complex(1.0), complex(1.0))

    # -- Álgebra --------------------------------------------------------------
    def __add__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """Suma en C₂, diagonal por canal idempotente."""
        return BicomplexScalar(
            _sanitize_complex(self.z1 + other.z1),
            _sanitize_complex(self.z2 + other.z2),
        )

    def __sub__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """Resta en C₂, diagonal por canal idempotente."""
        return BicomplexScalar(
            _sanitize_complex(self.z1 - other.z1),
            _sanitize_complex(self.z2 - other.z2),
        )

    def __mul__(self, other: "BicomplexScalar") -> "BicomplexScalar":
        """
        Producto en C₂.

        Por diagonalización idempotente el producto se factoriza canal a
        canal — este es precisamente el invariante [I6] del módulo:

            (Z W)⁽ᵃ⁾ = Z⁽ᵃ⁾ W⁽ᵃ⁾,   a ∈ {1, 2}.

        En consecuencia C₂ ≅ C × C como anillo (isomorfismo de Chinese
        Remainder Theorem inducido por los idempotentes ortogonales).
        """
        return BicomplexScalar(
            _sanitize_complex(self.z1 * other.z1),
            _sanitize_complex(self.z2 * other.z2),
        )

    def scalar_mul(self, k: float) -> "BicomplexScalar":
        """Multiplicación por escalar real k ∈ R."""
        return BicomplexScalar(
            _sanitize_complex(self.z1 * k),
            _sanitize_complex(self.z2 * k),
        )

    def conjugate_hyperbolic(self) -> "BicomplexScalar":
        """
        Conjugado hiperbólico (involución j ↦ -j).

        Intercambia los canales idempotentes: (Z⁽¹⁾, Z⁽²⁾) ↦ (Z⁽²⁾, Z⁽¹⁾).
        Su punto fijo es exactamente el subálgebra real-compleja diagonal
        Z⁽¹⁾ = Z⁽²⁾.
        """
        return BicomplexScalar(self.z2, self.z1)

    def conjugate_complex(self) -> "BicomplexScalar":
        """Conjugado complejo canal a canal (involución i ↦ -i)."""
        return BicomplexScalar(self.z1.conjugate(), self.z2.conjugate())

    def is_zero_divisor(self, tol: float = BicomplexConstants.EPSILON) -> bool:
        """
        Detecta divisores de cero del álgebra C₂.

        C₂ ≅ C ⊕ C no es un dominio de integridad: Z ≠ 0 es divisor de cero
        si y solo si exactamente uno de sus canales idempotentes se anula
        (equivalentemente, Z pertenece al ideal e₁C₂ o e₂C₂ pero no a
        ambos ni a ninguno).

        Returns:
            True si Z es un divisor de cero no trivial.
        """
        z1_zero = abs(self.z1) <= tol
        z2_zero = abs(self.z2) <= tol
        return z1_zero != z2_zero  # XOR: exactamente un canal nulo

    # -- Métrica ----------------------------------------------------------
    def to_real4(self) -> np.ndarray:
        """
        Reconstruye la representación real 4D canónica.

        Returns:
            array [Re Z⁽¹⁾, Im Z⁽¹⁾, Re Z⁽²⁾, Im Z⁽²⁾].
        """
        return np.array(
            [self.z1.real, self.z1.imag, self.z2.real, self.z2.imag],
            dtype=np.float64,
        )

    def norm_l2(self) -> float:
        """
        Norma euclidiana inducida por la base idempotente.

            ||Z||₂ = sqrt(|Z⁽¹⁾|² + |Z⁽²⁾|²).
        """
        return float(np.sqrt(abs(self.z1) ** 2 + abs(self.z2) ** 2))

    def norm_l1(self) -> float:
        """
        Norma ℓ₁ sobre la representación real 4D.

            ||Z||₁ = Σ |coord_i|.
        """
        arr = self.to_real4()
        return float(np.sum(np.abs(arr)))

    def banach_ratio(self) -> float:
        """
        Cociente de equivalencia de normas de Banach ℓ₁/ℓ₂.

        Para el vector nulo se define convencionalmente como 1.0 (límite
        de la razón cuando ambas normas tienden a cero al mismo orden).
        """
        l2 = self.norm_l2()
        if l2 <= BicomplexConstants.EPSILON:
            return 1.0
        return self.norm_l1() / l2


def verify_idempotent_algebra_identities(
    tol: float = BicomplexConstants.EPSILON,
) -> Dict[str, bool]:
    """
    FASE 1 — Certificación formal de las identidades idempotentes de C₂.

    Verifica numéricamente, sobre los representantes canónicos e₁, e₂, 1, 0
    de `BicomplexScalar`, las identidades estructurales de la base
    idempotente que definen el isomorfismo C₂ ≅ C ⊕ C:

        e₁² = e₁,   e₂² = e₂,   e₁ e₂ = 0,   e₁ + e₂ = 1.

    Esta función opera como *test de coherencia interna del álgebra*: si
    alguna identidad falla, la representación diagonal (z1, z2) empleada en
    todo el módulo dejaría de ser una realización fiel de C₂.

    Args:
        tol: Tolerancia absoluta de comparación en C.

    Returns:
        Diccionario con el resultado booleano de cada identidad y una
        clave `"all_hold"` con la conjunción de todas.
    """
    e1 = BicomplexScalar.idempotent_e1()
    e2 = BicomplexScalar.idempotent_e2()
    one = BicomplexScalar.one()
    zero = BicomplexScalar.zero()

    def close(a: BicomplexScalar, b: BicomplexScalar) -> bool:
        return abs(a.z1 - b.z1) <= tol and abs(a.z2 - b.z2) <= tol

    results = {
        "e1_idempotent": close(e1 * e1, e1),
        "e2_idempotent": close(e2 * e2, e2),
        "orthogonal": close(e1 * e2, zero),
        "partition_of_unity": close(e1 + e2, one),
    }
    results["all_hold"] = all(results.values())
    return results


#------------------------------------------------------------------------------
# 1.2 — CERTIFICACIÓN BILATERAL DE BANACH
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BanachCertificate:
    """
    Certificado metrológico de equivalencia bilateral de normas.

    Verifica la doble cota finito-dimensional (consecuencia directa de
    Cauchy-Schwarz en R^d, d = 4):

        1 ≤ ||Z||₁ / ||Z||₂ ≤ √d.

    La cota inferior es trivial pero su verificación explícita documenta
    que ninguna componente degenerada (p. ej. NaN o desbordamiento) ha
    corrompido la relación de orden entre normas.
    """

    l1_norm: float
    l2_norm: float
    ratio: float
    lower_bound: float
    upper_bound: float
    is_valid: bool

    @classmethod
    def from_scalar(cls, value: BicomplexScalar) -> "BanachCertificate":
        """
        Construye certificado bilateral para un escalar bicomplejo.

        Args:
            value: Escalar bicomplejo.

        Returns:
            Certificado inmutable de Banach.
        """
        l1 = value.norm_l1()
        l2 = value.norm_l2()
        ratio = value.banach_ratio()

        lower = BicomplexConstants.BANACH_LOWER_BOUND
        upper = BicomplexConstants.BANACH_UPPER_BOUND

        degenerate = l2 <= BicomplexConstants.EPSILON
        within_bounds = (
            ratio >= lower - BicomplexConstants.EPSILON
            and ratio <= upper + BicomplexConstants.EPSILON
        )

        return cls(
            l1_norm=l1,
            l2_norm=l2,
            ratio=ratio,
            lower_bound=lower,
            upper_bound=upper,
            is_valid=degenerate or within_bounds,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana."""
        return asdict(self)


#------------------------------------------------------------------------------
# 1.3 — INGESTA, ESTADO INMUTABLE Y SELLADO CRIPTOGRÁFICO
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexCochainState:
    """
    Estado inmutable de una cochain bicompleja discretizada.

    Representa una colección finita de escalares bicomplejos asociados a
    símplices, junto con sus certificados de Banach y sello HMAC-SHA256.

    Attributes:
        values: Tupla de escalares bicomplejos.
        dimension: Dimensión del espacio de ingestión (4).
        banach_certificates: Certificados por símplice.
        hmac_sha256: Sello criptográfico del payload saneado.
        created_at: Instante monotónico de construcción.
        metadata: Metadatos abiertos de trazabilidad (incluye la
            certificación de identidades algebraicas de C₂).
    """

    values: Tuple[BicomplexScalar, ...]
    dimension: int
    banach_certificates: Tuple[BanachCertificate, ...]
    hmac_sha256: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def channel_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae los dos canales idempotentes como arreglos complejos.

        Returns:
            (Z⁽¹⁾, Z⁽²⁾), ambos con dtype complex128.
        """
        z1 = np.asarray([v.z1 for v in self.values], dtype=np.complex128)
        z2 = np.asarray([v.z2 for v in self.values], dtype=np.complex128)
        return z1, z2

    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa para VectorResult."""
        return {
            "dimension": self.dimension,
            "cardinality": len(self.values),
            "values_real4": [v.to_real4().tolist() for v in self.values],
            "banach_certificates": [c.to_dict() for c in self.banach_certificates],
            "hmac_sha256": self.hmac_sha256,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


def vector_parse_bicomplex_structure(S: NDArray[np.float64]) -> BicomplexCochainState:
    """
    FASE 1 — Método canónico de ingestión bicompleja.

    Mapea una señal transaccional 4D a una cochain bicompleja:

        S = (s_purpose, s_confidence, s_constraints, s_risk) ∈ R⁴
        ↦ Z = (s₀ + i s₁) + (s₂ + i s₃) j ∈ C₂.

    El método realiza:
      1. Normalización dimensional de la entrada.
      2. Saneamiento FPU de ceros signados.
      3. Proyección idempotente Z⁽¹⁾, Z⁽²⁾.
      4. Certificación bilateral de Banach ℓ₁/ℓ₂.
      5. Certificación de las identidades algebraicas de C₂.
      6. Sellado criptográfico HMAC-SHA256.

    Args:
        S: Arreglo real de forma (4,), (n, 4) o compatible.

    Returns:
        Estado inmutable `BicomplexCochainState`.

    Raises:
        ValueError: Si la dimensionalidad no es múltiplo de 4.
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

    # Saneamiento FPU: -0.0 → +0.0.
    arr = arr.copy()
    arr[arr == 0.0] = 0.0

    values: List[BicomplexScalar] = []
    certs: List[BanachCertificate] = []

    for row in arr:
        z = BicomplexScalar.from_real4(row)
        values.append(z)
        certs.append(BanachCertificate.from_scalar(z))

    algebra_certificate = verify_idempotent_algebra_identities()

    state = BicomplexCochainState(
        values=tuple(values),
        dimension=4,
        banach_certificates=tuple(certs),
        hmac_sha256=_hmac_sha256(arr),
        created_at=time.monotonic(),
        metadata={
            "shape_in": tuple(arr.shape),
            "simplex_count": arr.shape[0],
            "algebra": "C2",
            "idempotent_basis": True,
            "algebra_identities": algebra_certificate,
        },
    )

    if not algebra_certificate["all_hold"]:
        logger.error("Fallo de certificación algebraica de C₂: %s", algebra_certificate)

    logger.debug(
        "Fase 1 completada: %d símplices bicomplejos, HMAC=%s",
        len(state),
        state.hmac_sha256[:12],
    )
    return state


def verify_bicomplex_cochain_hmac(
    state: BicomplexCochainState,
    original_array: NDArray[np.float64],
) -> bool:
    """
    FASE 1 — Verificación de integridad criptográfica HMAC-SHA256.

    Recalcula el sello sobre `original_array` (sometida al mismo saneamiento
    de ceros signados y reshape a (n, 4) que la ingestión original) y lo
    compara en **tiempo constante** contra `state.hmac_sha256`, detectando
    manipulación posterior a la ingestión.

    Args:
        state: Estado bicomplejo previamente sellado.
        original_array: Arreglo real bruto que originó el estado.

    Returns:
        True si el sello es válido; False en caso de discrepancia.
    """
    arr = np.asarray(original_array, dtype=np.float64).copy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 4)
    arr[arr == 0.0] = 0.0
    recomputed = _hmac_sha256(arr)
    return hmac.compare_digest(recomputed, state.hmac_sha256)


#------------------------------------------------------------------------------
# 1.4 — FRONTERA DE FASE: FASE 1 → FASE 2
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Phase2Input:
    """
    Objeto frontera entre Fase 1 y Fase 2.

    La salida de Fase 1 se convierte aquí en entrada canónica de la fase
    homológica/espectral. Este objeto puede portar, además del estado de
    cochain, fronteras discretas opcionales para operadores de Hodge.

    Attributes:
        cochain_state: Estado bicomplejo saneado y sellado.
        boundary_1: Operador de frontera opcional del canal 1.
        boundary_2: Operador de frontera opcional del canal 2.
    """

    cochain_state: BicomplexCochainState
    boundary_1: Optional[np.ndarray] = None
    boundary_2: Optional[np.ndarray] = None


def phase1_export_to_phase2(state: BicomplexCochainState) -> Phase2Input:
    """
    FASE 1 — Último método de la fase y continuación formal de la Fase 2.

    Este morfismo de cierre/apertura preserva el estado inmutable producido
    por `vector_parse_bicomplex_structure` y lo eleva al dominio de la Fase 2.

    Su contrato es:

        phase1_export_to_phase2 : BicomplexCochainState → Phase2Input

    y la Fase 2 debe consumir exclusivamente esta imagen canónica o
    estructuras equivalentes especificadas en esta misma capa. Como
    condición de admisión, se exige que la certificación algebraica de C₂
    registrada en `state.metadata["algebra_identities"]` sea afirmativa;
    de lo contrario el estado no puede considerarse una realización fiel
    del álgebra sobre la que operará toda la Fase 2.

    Args:
        state: Estado bicomplejo saneado y sellado.

    Returns:
        Entrada canónica para la Fase 2.

    Raises:
        TypeError: Si `state` no es `BicomplexCochainState`.
        ValueError: Si la certificación algebraica interna no se sostiene.
    """
    if not isinstance(state, BicomplexCochainState):
        raise TypeError("phase1_export_to_phase2 requiere BicomplexCochainState.")

    algebra_ok = state.metadata.get("algebra_identities", {}).get("all_hold", True)
    if not algebra_ok:
        raise ValueError(
            "El estado bicomplejo no satisface las identidades idempotentes "
            "de C₂; la frontera Fase1→Fase2 no puede admitirlo."
        )

    return Phase2Input(cochain_state=state)


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███       FASE 2 — ANÁLISIS HOMOLÓGICO Y ESPECTRAL BICOMPLEJO (Orient)   ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase implementa:
#
#   1. Mayer-Vietoris dual sobre canales idempotentes:
#
#        Δβ₁⁽¹⁾ = β₁⁽¹⁾(A∪B) - [β₁⁽¹⁾(A) + β₁⁽¹⁾(B) - β₁⁽¹⁾(A∩B)] = 0
#        Δβ₁⁽²⁾ = β₁⁽²⁾(A∪B) - [β₁⁽²⁾(A) + β₁⁽²⁾(B) - β₁⁽²⁾(A∩B)] = 0
#
#   2. Laplaciano de Hodge discreto bicomplejo:
#
#        L_k^{C₂} = L_k⁽¹⁾ e₁ + L_k⁽²⁾ e₂,
#
#      con certificación de nilpotencia [I1], saneamiento espectral PSD
#      [I7] y certificación de la descomposición de Hodge-Kodaira [I3].
#
#   3. Conectividad de Fiedler dual, sumación KBN y síntesis de un índice
#      de defecto topológico por canal, inyectado como frontera hacia la
#      Fase 3.
#=============================================================================


#------------------------------------------------------------------------------
# 2.1 — SUBCOMPLEJOS SIMPLICIALES Y AUDITORÍA MAYER-VIETORIS DUAL
#------------------------------------------------------------------------------
def _normalize_edge(edge: Sequence[int]) -> Tuple[int, int]:
    """
    Normaliza una arista no dirigida como tupla ordenada (a, b), a < b.

    Un 1-símplice simplicial requiere exactamente dos vértices distintos;
    los self-loops se rechazan por ser degeneraciones no admitidas por la
    definición estándar de complejo simplicial abstracto.

    Raises:
        ValueError: Si la arista no tiene exactamente dos extremos, o si
            es un self-loop (a == b).
    """
    if len(edge) != 2:
        raise ValueError(f"Arista inválida: {edge!r}")
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        raise ValueError(f"Arista degenerada (self-loop) no admitida: {edge!r}")
    return (a, b) if a <= b else (b, a)


def _optional_edge_set(raw: Any) -> Optional[FrozenSet[Tuple[int, int]]]:
    """Convierte una colección opcional de aristas en frozenset normalizado."""
    if raw is None:
        return None
    return frozenset(_normalize_edge(e) for e in raw)


@dataclass(frozen=True, slots=True)
class SubcomplexSpec:
    """
    Subcomplejo simplicial discreto, especializado a 1-esqueleto auditable.

    Attributes:
        vertices: Conjunto de vértices.
        edges: Aristas topológicas base.
        edges_channel_1: Aristas efectivas del canal idempotente e₁.
        edges_channel_2: Aristas efectivas del canal idempotente e₂.
        metadata: Metadatos de trazabilidad.
    """

    vertices: FrozenSet[int]
    edges: FrozenSet[Tuple[int, int]]
    edges_channel_1: Optional[FrozenSet[Tuple[int, int]]] = None
    edges_channel_2: Optional[FrozenSet[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SubcomplexSpec":
        """
        Construye un subcomplejo desde un mapping.

        Claves soportadas:
            - vertices: iterable de enteros.
            - edges: iterable de pares.
            - edges_channel_1: iterable opcional de pares.
            - edges_channel_2: iterable opcional de pares.
            - metadata: mapping opcional.
        """
        vertices = {int(v) for v in data.get("vertices", [])}
        edges = frozenset(_normalize_edge(e) for e in data.get("edges", []))

        # Clausura incidente: toda arista induce sus vértices.
        for a, b in edges:
            vertices.add(a)
            vertices.add(b)

        return cls(
            vertices=frozenset(vertices),
            edges=edges,
            edges_channel_1=_optional_edge_set(data.get("edges_channel_1")),
            edges_channel_2=_optional_edge_set(data.get("edges_channel_2")),
            metadata=dict(data.get("metadata", {})),
        )


def _beta0_beta1(
    vertices: FrozenSet[int],
    edges: FrozenSet[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Calcula β₀ y β₁ para un 1-complejo finito.

    Usa union-find para β₀ y la fórmula euleriana:

        β₁ = |E| - |V| + β₀.

    Returns:
        (beta_0, beta_1).
    """
    if not vertices:
        return 0, 0

    parent: Dict[int, int] = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    valid_edges: List[Tuple[int, int]] = []
    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)
            valid_edges.append((a, b))

    beta0 = len({find(v) for v in vertices})
    beta1 = len(valid_edges) - len(vertices) + beta0

    if beta1 < 0:
        beta1 = 0

    return beta0, beta1


def _effective_edges(spec: SubcomplexSpec, channel: int) -> FrozenSet[Tuple[int, int]]:
    """Devuelve las aristas efectivas de un canal idempotente."""
    if channel == 1 and spec.edges_channel_1 is not None:
        return spec.edges_channel_1
    if channel == 2 and spec.edges_channel_2 is not None:
        return spec.edges_channel_2
    return spec.edges


def _filter_edges_to_vertices(
    edges: FrozenSet[Tuple[int, int]],
    vertices: FrozenSet[int],
) -> FrozenSet[Tuple[int, int]]:
    """Restringe aristas al conjunto de vértices dado."""
    return frozenset(e for e in edges if e[0] in vertices and e[1] in vertices)


def _union_subcomplex(A: SubcomplexSpec, B: SubcomplexSpec) -> SubcomplexSpec:
    """Construye A ∪ B con canales idempotentes efectivos."""
    vertices = A.vertices | B.vertices

    edges = A.edges | B.edges
    e1 = _effective_edges(A, 1) | _effective_edges(B, 1)
    e2 = _effective_edges(A, 2) | _effective_edges(B, 2)

    return SubcomplexSpec(
        vertices=vertices,
        edges=edges,
        edges_channel_1=e1,
        edges_channel_2=e2,
        metadata={"origin": "union"},
    )


def _intersection_subcomplex(A: SubcomplexSpec, B: SubcomplexSpec) -> SubcomplexSpec:
    """Construye A ∩ B con canales idempotentes efectivos."""
    vertices = A.vertices & B.vertices

    edges = _filter_edges_to_vertices(A.edges & B.edges, vertices)
    e1 = _filter_edges_to_vertices(_effective_edges(A, 1) & _effective_edges(B, 1), vertices)
    e2 = _filter_edges_to_vertices(_effective_edges(A, 2) & _effective_edges(B, 2), vertices)

    return SubcomplexSpec(
        vertices=vertices,
        edges=edges,
        edges_channel_1=e1,
        edges_channel_2=e2,
        metadata={"origin": "intersection"},
    )


def _betti1_channel(spec: SubcomplexSpec, channel: int) -> int:
    """Calcula β₁ del canal idempotente especificado."""
    edges = _effective_edges(spec, channel)
    _, beta1 = _beta0_beta1(spec.vertices, edges)
    return beta1


@dataclass(frozen=True, slots=True)
class MayerVietorisBicomplexReport:
    """
    Reporte inmutable de auditoría Mayer-Vietoris bicompleja.

    Attributes:
        delta_beta_1_channel_1: Δβ₁⁽¹⁾.
        delta_beta_1_channel_2: Δβ₁⁽²⁾.
        exact: True si ambos residuos son nulos.
        beta_A: Betti por canal de A.
        beta_B: Betti por canal de B.
        beta_union: Betti por canal de A∪B.
        beta_intersection: Betti por canal de A∩B.
    """

    delta_beta_1_channel_1: int
    delta_beta_1_channel_2: int
    exact: bool
    beta_A: Dict[str, int]
    beta_B: Dict[str, int]
    beta_union: Dict[str, int]
    beta_intersection: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana."""
        return asdict(self)


def vector_audit_bicomplex_mayer_vietoris(
    A: Union[SubcomplexSpec, Mapping[str, Any]],
    B: Union[SubcomplexSpec, Mapping[str, Any]],
) -> MayerVietorisBicomplexReport:
    """
    FASE 2 — Auditoría Mayer-Vietoris dual [I2].

    Verifica la exactitud homológica de la fusión de dos subcomplejos en los
    dos canales idempotentes:

        Δβ₁⁽¹⁾ = 0 ∧ Δβ₁⁽²⁾ = 0.

    Si alguno de los residuos es distinto de cero, se detecta un ciclo
    fantasma, una triangulación parásita o una ruptura de la secuencia exacta.

    Args:
        A: Subcomplejo A, bien como `SubcomplexSpec` o mapping.
        B: Subcomplejo B, bien como `SubcomplexSpec` o mapping.

    Returns:
        `MayerVietorisBicomplexReport`.

    Raises:
        TypeError: Si A o B no son convertibles a SubcomplexSpec.
    """
    if isinstance(A, Mapping):
        A = SubcomplexSpec.from_mapping(A)
    if isinstance(B, Mapping):
        B = SubcomplexSpec.from_mapping(B)

    if not isinstance(A, SubcomplexSpec) or not isinstance(B, SubcomplexSpec):
        raise TypeError("A y B deben ser SubcomplexSpec o Mapping compatible.")

    U = _union_subcomplex(A, B)
    I = _intersection_subcomplex(A, B)

    beta_a_1 = _betti1_channel(A, 1)
    beta_a_2 = _betti1_channel(A, 2)

    beta_b_1 = _betti1_channel(B, 1)
    beta_b_2 = _betti1_channel(B, 2)

    beta_u_1 = _betti1_channel(U, 1)
    beta_u_2 = _betti1_channel(U, 2)

    beta_i_1 = _betti1_channel(I, 1)
    beta_i_2 = _betti1_channel(I, 2)

    delta_1 = beta_u_1 - (beta_a_1 + beta_b_1 - beta_i_1)
    delta_2 = beta_u_2 - (beta_a_2 + beta_b_2 - beta_i_2)

    report = MayerVietorisBicomplexReport(
        delta_beta_1_channel_1=int(delta_1),
        delta_beta_1_channel_2=int(delta_2),
        exact=(delta_1 == 0 and delta_2 == 0),
        beta_A={"channel_1": beta_a_1, "channel_2": beta_a_2},
        beta_B={"channel_1": beta_b_1, "channel_2": beta_b_2},
        beta_union={"channel_1": beta_u_1, "channel_2": beta_u_2},
        beta_intersection={"channel_1": beta_i_1, "channel_2": beta_i_2},
    )

    if not report.exact:
        logger.warning(
            "Mayer-Vietoris dual no exacto: Δβ₁⁽¹⁾=%d, Δβ₁⁽²⁾=%d",
            report.delta_beta_1_channel_1,
            report.delta_beta_1_channel_2,
        )
    else:
        logger.debug("Mayer-Vietoris dual exacto.")

    return report


#------------------------------------------------------------------------------
# 2.2 — HODGE LAPLACIANO BICOMPLEJO, CERTIFICACIÓN ESPECTRAL Y KODAIRA
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexHodgeCochain:
    """
    Cochain discreta con operadores de frontera para Hodge bicomplejo.

    Attributes:
        D_up_1: Operador ascendente d_k del canal 1, shape (m, n).
        D_down_1: Operador descendente d_{k-1} del canal 1, shape (n, p).
        D_up_2: Operador ascendente del canal 2. Si es None, usa D_up_1.
        D_down_2: Operador descendente del canal 2. Si es None, usa D_down_1.
        L0_1: Laplaciano de vértices opcional canal 1 para Fiedler.
        L0_2: Laplaciano de vértices opcional canal 2 para Fiedler.
        metadata: Metadatos.
    """

    D_up_1: np.ndarray
    D_down_1: Optional[np.ndarray] = None
    D_up_2: Optional[np.ndarray] = None
    D_down_2: Optional[np.ndarray] = None
    L0_1: Optional[np.ndarray] = None
    L0_2: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BicomplexHodgeMetric:
    """
    Métrica constitutiva bicompleja para el Laplaciano de Hodge.

    Se descompone en pesos `up` y `down` por canal, permitiendo la construcción

        L_k⁽ᵃ⁾ = (d_k⁽ᵃ⁾)ᵀ W_up⁽ᵃ⁾ d_k⁽ᵃ⁾
               + d_{k-1}⁽ᵃ⁾ (W_down⁽ᵃ⁾)⁻¹ (d_{k-1}⁽ᵃ⁾)ᵀ.
    """

    W_up_1: np.ndarray
    W_down_1: np.ndarray
    W_up_2: np.ndarray
    W_down_2: np.ndarray


def _as_square_matrix(M: Any, size: int, name: str) -> np.ndarray:
    """
    Convierte `M` en matriz cuadrada `size × size`.

    Reglas:
      - None → identidad.
      - escalar (incluyendo arrays 0-d) → escalar × identidad.
      - vector 1D → diagonal.
      - matriz 2D → validación y simetrización.
    """
    if size == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if M is None:
        return np.eye(size, dtype=np.float64)

    if np.isscalar(M) or (isinstance(M, np.ndarray) and M.ndim == 0):
        return float(M) * np.eye(size, dtype=np.float64)

    arr = np.asarray(M, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size != size:
            raise ValueError(f"{name}: vector de tamaño {arr.size} no compatible con size={size}.")
        return np.diag(arr)

    if arr.ndim == 2:
        if arr.shape != (size, size):
            raise ValueError(f"{name}: shape {arr.shape} no compatible con {(size, size)}.")
        return 0.5 * (arr + arr.T)

    raise ValueError(f"{name}: tipo métrico no soportado.")


def _prepare_hodge_metric(
    cochain: BicomplexHodgeCochain,
    W_metric: Union[BicomplexHodgeMetric, float, np.ndarray, Tuple[Any, ...]],
) -> BicomplexHodgeMetric:
    """
    Prepara la métrica constitutiva bicompleja desde formas múltiples.

    Formatos aceptados:
      - BicomplexHodgeMetric.
      - escalar / matriz única aplicada a todos los bloques.
      - tupla (up, down) aplicada a ambos canales.
      - tupla (up1, down1, up2, down2).
    """
    D_up_1 = np.asarray(cochain.D_up_1, dtype=np.float64)
    D_down_1 = None if cochain.D_down_1 is None else np.asarray(cochain.D_down_1, dtype=np.float64)
    D_up_2 = D_up_1 if cochain.D_up_2 is None else np.asarray(cochain.D_up_2, dtype=np.float64)
    D_down_2 = D_down_1 if cochain.D_down_2 is None else np.asarray(cochain.D_down_2, dtype=np.float64)

    n_up_1 = D_up_1.shape[0]
    n_down_1 = 0 if D_down_1 is None else D_down_1.shape[1]

    n_up_2 = D_up_2.shape[0]
    n_down_2 = 0 if D_down_2 is None else D_down_2.shape[1]

    if isinstance(W_metric, BicomplexHodgeMetric):
        return BicomplexHodgeMetric(
            W_up_1=_as_square_matrix(W_metric.W_up_1, n_up_1, "W_up_1"),
            W_down_1=_as_square_matrix(W_metric.W_down_1, n_down_1, "W_down_1"),
            W_up_2=_as_square_matrix(W_metric.W_up_2, n_up_2, "W_up_2"),
            W_down_2=_as_square_matrix(W_metric.W_down_2, n_down_2, "W_down_2"),
        )

    if isinstance(W_metric, tuple):
        if len(W_metric) == 2:
            up1, down1 = W_metric
            up2, down2 = up1, down1
        elif len(W_metric) == 4:
            up1, down1, up2, down2 = W_metric
        else:
            raise ValueError("Tupla métrica debe tener longitud 2 o 4.")
    else:
        up1 = down1 = up2 = down2 = W_metric

    return BicomplexHodgeMetric(
        W_up_1=_as_square_matrix(up1, n_up_1, "W_up_1"),
        W_down_1=_as_square_matrix(down1, n_down_1, "W_down_1"),
        W_up_2=_as_square_matrix(up2, n_up_2, "W_up_2"),
        W_down_2=_as_square_matrix(down2, n_down_2, "W_down_2"),
    )


def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Simetriza una matriz cuadrada: (M + Mᵀ)/2."""
    return 0.5 * (M + M.T)


def _assemble_laplacian(
    D_up: np.ndarray,
    W_up: np.ndarray,
    D_down: Optional[np.ndarray],
    W_down: np.ndarray,
) -> np.ndarray:
    """
    Ensambla el Laplaciano de Hodge discreto de un canal.

        L = D_upᵀ W_up D_up + D_down W_down⁻¹ D_downᵀ.

    Ambos sumandos son semidefinidos positivos (formas cuadráticas
    congruentes con W_up, W_down⁻¹ ⪰ 0), de modo que L ⪰ 0 por
    construcción — la base del invariante espectral [I7]. Si D_down es
    vacío, se omite el segundo término (caso de cohomología de grado 0).
    """
    if D_up.ndim != 2:
        raise ValueError("D_up debe ser matriz 2D.")

    L = D_up.T @ W_up @ D_up

    if D_down is not None and D_down.size > 0 and D_down.shape[1] > 0:
        W_down_inv = np.linalg.pinv(W_down)
        L = L + D_down @ W_down_inv @ D_down.T

    return _symmetrize(L)


def _eigvalsh_safe(M: np.ndarray) -> np.ndarray:
    """Autovalores simétricos con manejo de matrices vacías."""
    if M.size == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.eigvalsh(_symmetrize(M))


def _clip_psd_spectrum(
    eigvals: np.ndarray,
    tol: float = BicomplexConstants.HARMONIC_EPSILON,
) -> np.ndarray:
    """
    Saneamiento espectral PSD — invariante [I7].

    El Laplaciano de Hodge es semidefinido positivo por construcción
    (`_assemble_laplacian`); autovalores negativos de magnitud menor que
    `tol` son artefactos de redondeo de `eigvalsh` y se recortan a cero.
    Autovalores negativos de magnitud mayor indicarían una corrupción real
    de la métrica constitutiva y se preservan intencionalmente para su
    detección aguas abajo.

    Args:
        eigvals: Espectro bruto de un Laplaciano teóricamente PSD.
        tol: Umbral de tolerancia de redondeo.

    Returns:
        Espectro saneado.
    """
    return np.where((eigvals < 0.0) & (eigvals > -tol), 0.0, eigvals)


def _fiedler_value(L: np.ndarray) -> float:
    """
    Valor de Fiedler λ₂.

    Para grafos triviales de un nodo se devuelve 1.0, indicando conectividad
    degenerada pero no bloqueante.
    """
    if L.size == 0 or L.shape[0] < 2:
        return 1.0

    eigvals = np.linalg.eigvalsh(_symmetrize(L))
    if eigvals.size < 2:
        return float(eigvals[0]) if eigvals.size == 1 else 1.0

    return float(eigvals[1])


def _harmonic_dimension(eigvals: np.ndarray) -> int:
    """Dimensión del núcleo espectral dentro de la tolerancia armónica."""
    if eigvals.size == 0:
        return 0
    return int(np.count_nonzero(eigvals <= BicomplexConstants.HARMONIC_EPSILON))


@dataclass(frozen=True, slots=True)
class NilpotencyCertificate:
    """
    Certificado del invariante [I1]: nilpotencia del operador de coborde.

        d_k ∘ d_{k-1} = 0.

    Attributes:
        operator_norm: ||d_k d_{k-1}||_F (norma de Frobenius del residuo).
        max_absolute_entry: max |entrada| del residuo.
        is_nilpotent: True si el residuo es nulo dentro de tolerancia.
        tolerance: Tolerancia utilizada.
    """

    operator_norm: float
    max_absolute_entry: float
    is_nilpotent: bool
    tolerance: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana."""
        return asdict(self)


def verify_coboundary_nilpotency(
    D_lower: np.ndarray,
    D_upper: np.ndarray,
    tolerance: float = BicomplexConstants.EPSILON,
) -> NilpotencyCertificate:
    """
    FASE 2 — Certificación del invariante [I1] de nilpotencia del coborde.

    Dados d_{k-1}: C^{k-1} → C^k (shape (n, p)) y d_k: C^k → C^{k+1}
    (shape (m, n)), calcula el residuo

        R = d_k ∘ d_{k-1} = D_upper @ D_lower  (shape (m, p))

    y verifica ||R||_F ≈ 0, condición necesaria y suficiente en el caso
    simplicial estándar para que el par (d_{k-1}, d_k) defina un complejo
    de cocadenas válido cuya cohomología esté bien definida.

    Args:
        D_lower: Operador de coborde d_{k-1}, shape (n, p).
        D_upper: Operador de coborde d_k, shape (m, n).
        tolerance: Cota de tolerancia para el máximo absoluto del residuo.

    Returns:
        `NilpotencyCertificate`.

    Raises:
        ValueError: Si las dimensiones son incompatibles para la composición.
    """
    Dl = np.asarray(D_lower, dtype=np.float64)
    Du = np.asarray(D_upper, dtype=np.float64)

    if Dl.ndim != 2 or Du.ndim != 2:
        raise ValueError("D_lower y D_upper deben ser matrices 2D.")
    if Du.shape[1] != Dl.shape[0]:
        raise ValueError(
            f"Dimensiones incompatibles para D_upper @ D_lower: "
            f"{Du.shape} @ {Dl.shape}."
        )

    residual = Du @ Dl
    op_norm = float(np.linalg.norm(residual, ord="fro"))
    max_entry = float(np.max(np.abs(residual))) if residual.size > 0 else 0.0

    return NilpotencyCertificate(
        operator_norm=op_norm,
        max_absolute_entry=max_entry,
        is_nilpotent=max_entry <= tolerance,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class HodgeKodairaCertificate:
    """
    Certificado del invariante [I3]: descomposición de Hodge-Kodaira discreta.

        C^k = im(d_{k-1}) ⊕ im(d_k^T) ⊕ ker(L_k).

    Verificado por conteo dimensional:

        n = rank(d_{k-1}) + rank(d_k) + dim(ker L_k).

    Attributes:
        ambient_dimension: n = dim C^k.
        rank_down: rango de d_{k-1}.
        rank_up: rango de d_k.
        harmonic_dimension: dim ker(L_k) según tolerancia armónica.
        dimension_residual: n - (rank_down + rank_up + harmonic_dimension).
        is_consistent: True si el residuo dimensional es exactamente 0.
    """

    ambient_dimension: int
    rank_down: int
    rank_up: int
    harmonic_dimension: int
    dimension_residual: int
    is_consistent: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana."""
        return asdict(self)


def verify_hodge_kodaira_decomposition(
    D_up: np.ndarray,
    D_down: Optional[np.ndarray],
    eigenvalues: np.ndarray,
    tol: float = BicomplexConstants.HARMONIC_EPSILON,
) -> HodgeKodairaCertificate:
    """
    FASE 2 — Certificación del invariante [I3] por conteo dimensional.

    Args:
        D_up: Operador ascendente d_k, shape (m, n).
        D_down: Operador descendente d_{k-1}, shape (n, p), o None.
        eigenvalues: Espectro ya saneado (PSD) de L_k, tamaño n.
        tol: Tolerancia armónica para la dimensión del núcleo.

    Returns:
        `HodgeKodairaCertificate`.
    """
    Dup = np.asarray(D_up, dtype=np.float64)
    n = Dup.shape[1] if Dup.ndim == 2 else 0

    rank_up = int(np.linalg.matrix_rank(Dup)) if Dup.size > 0 else 0

    if D_down is not None and np.asarray(D_down).size > 0:
        Ddown = np.asarray(D_down, dtype=np.float64)
        rank_down = int(np.linalg.matrix_rank(Ddown))
    else:
        rank_down = 0

    harmonic_dim = int(np.count_nonzero(np.asarray(eigenvalues) <= tol))
    residual = n - (rank_down + rank_up + harmonic_dim)

    return HodgeKodairaCertificate(
        ambient_dimension=n,
        rank_down=rank_down,
        rank_up=rank_up,
        harmonic_dimension=harmonic_dim,
        dimension_residual=residual,
        is_consistent=(residual == 0),
    )


@dataclass(frozen=True, slots=True)
class BicomplexHodgeReport:
    """
    Reporte espectral bicomplejo.

    Attributes:
        L1_channel_1: Laplaciano canal 1.
        L1_channel_2: Laplaciano canal 2.
        eigenvalues_channel_1: Espectro saneado (PSD) canal 1.
        eigenvalues_channel_2: Espectro saneado (PSD) canal 2.
        fiedler_channel_1: λ₂ dual canal 1.
        fiedler_channel_2: λ₂ dual canal 2.
        harmonic_dimension_channel_1: dim ker L₁ canal 1.
        harmonic_dimension_channel_2: dim ker L₁ canal 2.
        trace_kbn_channel_1: Traza espectral KBN canal 1.
        trace_kbn_channel_2: Traza espectral KBN canal 2.
        connected_dual: Conectividad Fiedler dual aprobada.
        nilpotency_channel_1: Certificado [I1] canal 1, si D_down disponible.
        nilpotency_channel_2: Certificado [I1] canal 2, si D_down disponible.
        hodge_kodaira_channel_1: Certificado [I3] canal 1.
        hodge_kodaira_channel_2: Certificado [I3] canal 2.
    """

    L1_channel_1: np.ndarray
    L1_channel_2: np.ndarray
    eigenvalues_channel_1: np.ndarray
    eigenvalues_channel_2: np.ndarray
    fiedler_channel_1: float
    fiedler_channel_2: float
    harmonic_dimension_channel_1: int
    harmonic_dimension_channel_2: int
    trace_kbn_channel_1: float
    trace_kbn_channel_2: float
    connected_dual: bool
    nilpotency_channel_1: Optional[NilpotencyCertificate] = None
    nilpotency_channel_2: Optional[NilpotencyCertificate] = None
    hodge_kodaira_channel_1: Optional[HodgeKodairaCertificate] = None
    hodge_kodaira_channel_2: Optional[HodgeKodairaCertificate] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe."""
        return {
            "L1_channel_1_shape": list(self.L1_channel_1.shape),
            "L1_channel_2_shape": list(self.L1_channel_2.shape),
            "eigenvalues_channel_1": self.eigenvalues_channel_1.tolist(),
            "eigenvalues_channel_2": self.eigenvalues_channel_2.tolist(),
            "fiedler_channel_1": self.fiedler_channel_1,
            "fiedler_channel_2": self.fiedler_channel_2,
            "harmonic_dimension_channel_1": self.harmonic_dimension_channel_1,
            "harmonic_dimension_channel_2": self.harmonic_dimension_channel_2,
            "trace_kbn_channel_1": self.trace_kbn_channel_1,
            "trace_kbn_channel_2": self.trace_kbn_channel_2,
            "connected_dual": self.connected_dual,
            "nilpotency_channel_1": (
                self.nilpotency_channel_1.to_dict() if self.nilpotency_channel_1 else None
            ),
            "nilpotency_channel_2": (
                self.nilpotency_channel_2.to_dict() if self.nilpotency_channel_2 else None
            ),
            "hodge_kodaira_channel_1": (
                self.hodge_kodaira_channel_1.to_dict() if self.hodge_kodaira_channel_1 else None
            ),
            "hodge_kodaira_channel_2": (
                self.hodge_kodaira_channel_2.to_dict() if self.hodge_kodaira_channel_2 else None
            ),
        }


def vector_bicomplex_hodge_laplacian(
    cochain: BicomplexHodgeCochain,
    W_metric: Union[BicomplexHodgeMetric, float, np.ndarray, Tuple[Any, ...]],
) -> BicomplexHodgeReport:
    """
    FASE 2 — Laplaciano de Hodge discreto bicomplejo con certificación
    espectral completa.

    Construye

        L_k^{C₂} = L_k⁽¹⁾ e₁ + L_k⁽²⁾ e₂

    y extrae invariantes espectrales por canal:
      - autovalores simétricos saneados PSD [I7];
      - dimensión armónica ker(L);
      - conectividad de Fiedler dual;
      - traza espectral con suma KBN;
      - certificado de nilpotencia del coborde [I1] (si D_down disponible);
      - certificado de la descomposición de Hodge-Kodaira [I3].

    Args:
        cochain: Operadores de frontera por canal.
        W_metric: Métrica constitutiva. Puede ser escalar, matriz, tupla o
                  `BicomplexHodgeMetric`.

    Returns:
        `BicomplexHodgeReport`.

    Raises:
        ValueError: Si dimensiones de operadores y métricas no compatibilizan.
    """
    D_up_1 = np.asarray(cochain.D_up_1, dtype=np.float64)
    D_down_1 = None if cochain.D_down_1 is None else np.asarray(cochain.D_down_1, dtype=np.float64)

    D_up_2 = D_up_1 if cochain.D_up_2 is None else np.asarray(cochain.D_up_2, dtype=np.float64)
    D_down_2 = D_down_1 if cochain.D_down_2 is None else np.asarray(cochain.D_down_2, dtype=np.float64)

    if D_up_1.ndim != 2:
        raise ValueError("D_up_1 debe ser 2D.")
    if D_up_2.ndim != 2:
        raise ValueError("D_up_2 debe ser 2D.")
    if D_down_1 is not None and D_down_1.ndim != 2:
        raise ValueError("D_down_1 debe ser 2D o None.")
    if D_down_2 is not None and D_down_2.ndim != 2:
        raise ValueError("D_down_2 debe ser 2D o None.")

    metric = _prepare_hodge_metric(cochain, W_metric)

    L1_1 = _assemble_laplacian(D_up_1, metric.W_up_1, D_down_1, metric.W_down_1)
    L1_2 = _assemble_laplacian(D_up_2, metric.W_up_2, D_down_2, metric.W_down_2)

    eig1 = _clip_psd_spectrum(_eigvalsh_safe(L1_1))
    eig2 = _clip_psd_spectrum(_eigvalsh_safe(L1_2))

    L0_1 = None if cochain.L0_1 is None else np.asarray(cochain.L0_1, dtype=np.float64)
    L0_2 = None if cochain.L0_2 is None else np.asarray(cochain.L0_2, dtype=np.float64)

    fiedler1 = _fiedler_value(L0_1 if L0_1 is not None else L1_1)
    fiedler2 = _fiedler_value(L0_2 if L0_2 is not None else L1_2)

    trace1 = _kbn_sum(eig1.tolist())
    trace2 = _kbn_sum(eig2.tolist())

    connected_dual = (
        fiedler1 >= BicomplexConstants.FIEDLER_EPSILON
        and fiedler2 >= BicomplexConstants.FIEDLER_EPSILON
    )

    nilpotency_1: Optional[NilpotencyCertificate] = None
    nilpotency_2: Optional[NilpotencyCertificate] = None
    if D_down_1 is not None and D_down_1.size > 0:
        nilpotency_1 = verify_coboundary_nilpotency(D_down_1, D_up_1)
    if D_down_2 is not None and D_down_2.size > 0:
        nilpotency_2 = verify_coboundary_nilpotency(D_down_2, D_up_2)

    hodge_kodaira_1 = verify_hodge_kodaira_decomposition(D_up_1, D_down_1, eig1)
    hodge_kodaira_2 = verify_hodge_kodaira_decomposition(D_up_2, D_down_2, eig2)

    report = BicomplexHodgeReport(
        L1_channel_1=L1_1,
        L1_channel_2=L1_2,
        eigenvalues_channel_1=eig1,
        eigenvalues_channel_2=eig2,
        fiedler_channel_1=fiedler1,
        fiedler_channel_2=fiedler2,
        harmonic_dimension_channel_1=_harmonic_dimension(eig1),
        harmonic_dimension_channel_2=_harmonic_dimension(eig2),
        trace_kbn_channel_1=trace1,
        trace_kbn_channel_2=trace2,
        connected_dual=connected_dual,
        nilpotency_channel_1=nilpotency_1,
        nilpotency_channel_2=nilpotency_2,
        hodge_kodaira_channel_1=hodge_kodaira_1,
        hodge_kodaira_channel_2=hodge_kodaira_2,
    )

    if nilpotency_1 is not None and not nilpotency_1.is_nilpotent:
        logger.error("Violación de [I1] en canal 1: ||d_k d_{k-1}||_max=%.3e", nilpotency_1.max_absolute_entry)
    if nilpotency_2 is not None and not nilpotency_2.is_nilpotent:
        logger.error("Violación de [I1] en canal 2: ||d_k d_{k-1}||_max=%.3e", nilpotency_2.max_absolute_entry)

    logger.debug(
        "Hodge bicomplejo: Fiedler=(%.3e, %.3e), harmonic_dim=(%d, %d)",
        report.fiedler_channel_1,
        report.fiedler_channel_2,
        report.harmonic_dimension_channel_1,
        report.harmonic_dimension_channel_2,
    )
    return report


#------------------------------------------------------------------------------
# 2.3 — DEFECTOS TOPOLÓGICOS Y FRONTERA DE FASE: FASE 2 → FASE 3
#------------------------------------------------------------------------------
def compute_topological_defect_index(
    mv_report: Optional[MayerVietorisBicomplexReport],
    hodge_report: Optional[BicomplexHodgeReport],
) -> Tuple[float, float]:
    """
    FASE 2 — Índice de defecto topológico por canal idempotente.

    Sintetiza en un escalar no negativo por canal las anomalías detectadas
    en los dos análisis de esta fase:

        D⁽ᵃ⁾ = |Δβ₁⁽ᵃ⁾| + dim_H⁽ᵃ⁾ · 𝟙[fiedler⁽ᵃ⁾ < ε_Fiedler],

    donde el primer término penaliza rupturas de exactitud Mayer-Vietoris
    y el segundo penaliza dimensión armónica excedentaria en componentes
    desconectadas (Fiedler nulo, indicando fragmentación no anticipada del
    complejo). Este índice es exactamente el mecanismo por el cual la
    Fase 3 hereda información homológica/espectral: se inyecta como
    penalización entrópica aditiva en la fórmula de coherencia [I4].

    Args:
        mv_report: Auditoría Mayer-Vietoris dual, opcional.
        hodge_report: Reporte espectral de Hodge, opcional.

    Returns:
        (D⁽¹⁾, D⁽²⁾), ambos ≥ 0. Retorna (0.0, 0.0) si ambos reportes son None.
    """
    d1 = 0.0
    d2 = 0.0

    if mv_report is not None:
        d1 += abs(mv_report.delta_beta_1_channel_1)
        d2 += abs(mv_report.delta_beta_1_channel_2)

    if hodge_report is not None:
        disconnected_1 = hodge_report.fiedler_channel_1 < BicomplexConstants.FIEDLER_EPSILON
        disconnected_2 = hodge_report.fiedler_channel_2 < BicomplexConstants.FIEDLER_EPSILON
        if disconnected_1:
            d1 += hodge_report.harmonic_dimension_channel_1
        if disconnected_2:
            d2 += hodge_report.harmonic_dimension_channel_2

    return float(d1), float(d2)


@dataclass(frozen=True, slots=True)
class Phase3Input:
    """
    Objeto frontera entre Fase 2 y Fase 3.

    Reúne el estado bicomplejo, la auditoría Mayer-Vietoris dual y el reporte
    de Hodge para que la Fase 3 sintetice coherencia y veredicto de Heyting.
    A diferencia de una frontera puramente pasiva, este objeto es el
    **dominio efectivo** de `vector_compute_bicomplex_coherence_index`: sus
    campos `mayer_vietoris_report` y `hodge_report` se traducen en el
    índice de defecto topológico de `compute_topological_defect_index`,
    que penaliza aditivamente la entropía por canal en la Fase 3.

    Attributes:
        cochain_state: Estado bicomplejo original.
        mayer_vietoris_report: Auditoría Mayer-Vietoris dual, opcional.
        hodge_report: Reporte espectral de Hodge, opcional.
    """

    cochain_state: Optional[BicomplexCochainState] = None
    mayer_vietoris_report: Optional[MayerVietorisBicomplexReport] = None
    hodge_report: Optional[BicomplexHodgeReport] = None


def phase2_export_to_phase3(
    *,
    cochain_state: Optional[BicomplexCochainState] = None,
    mayer_vietoris_report: Optional[MayerVietorisBicomplexReport] = None,
    hodge_report: Optional[BicomplexHodgeReport] = None,
) -> Phase3Input:
    """
    FASE 2 — Último método de la fase y continuación formal de la Fase 3.

    Este morfismo de cierre/apertura toma los invariantes homológicos y
    espectrales producidos en la Fase 2 y los eleva al dominio de decisión
    ciber-física de la Fase 3.

    Contrato:

        phase2_export_to_phase3 :
            (BicomplexCochainState?, MayerVietorisBicomplexReport?, BicomplexHodgeReport?)
            → Phase3Input.

    La imagen de este morfismo es consumida **completa** por
    `vector_compute_bicomplex_coherence_index` cuando se le pasa un
    `Phase3Input`: la información de `mayer_vietoris_report` y
    `hodge_report`, lejos de ser meramente decorativa, participa
    activamente en la fórmula de coherencia vía
    `compute_topological_defect_index`.

    Args:
        cochain_state: Estado bicomplejo original, opcional.
        mayer_vietoris_report: Auditoría Mayer-Vietoris dual, opcional.
        hodge_report: Reporte espectral de Hodge, opcional.

    Returns:
        Entrada canónica para la Fase 3.
    """
    return Phase3Input(
        cochain_state=cochain_state,
        mayer_vietoris_report=mayer_vietoris_report,
        hodge_report=hodge_report,
    )


#=============================================================================
# ████████████████████████████████████████████████████████████████████████████
# ███ FASE 3 — COHERENCIA, HEYTING BICOMPLEJO Y CENSURA CIBER-FÍSICA (Act) ███
# ████████████████████████████████████████████████████████████████████████████
#=============================================================================
# Esta fase implementa el índice de coherencia bicomplejo:
#
#     C_{C₂} = C⁽¹⁾ e₁ + C⁽²⁾ e₂,
#
#     C⁽ᵃ⁾ = clamp(S⁽ᵃ⁾ R⁽ᵃ⁾ / (1 + H⁽ᵃ⁾), 0, 1),
#
# donde H⁽ᵃ⁾ = H_espectral⁽ᵃ⁾ + D⁽ᵃ⁾ incorpora el índice de defecto
# topológico heredado de la Fase 2 (frontera efectivamente consumida).
#
# Luego aplica el retículo de Heyting B₂ (con negación, disyunción,
# implicación y conjunción explícitas):
#
#     v_a = COHERENT  si C⁽ᵃ⁾ ≥ 0.85
#           DEGRADED  si 0.50 ≤ C⁽ᵃ⁾ < 0.85
#           VETOED    si C⁽ᵃ⁾ < 0.50
#
# y el veto global por conjunción:
#
#     Verdict_global = v₁ ∧ v₂ = min(v₁, v₂).
#=============================================================================


#------------------------------------------------------------------------------
# 3.1 — RETÍCULO DE HEYTING B₂
#------------------------------------------------------------------------------
class HeytingValue(IntEnum):
    """
    Retículo de Heyting de tres valores totalmente ordenado (cadena de
    Gödel B₂), el ejemplo canónico de álgebra de Heyting no booleana.

    Orden intuicionista:
        VETOED < DEGRADED < COHERENT.

    En una cadena finita, el retículo de Heyting está automáticamente bien
    definido con:
        a ∧ b = min(a, b),   a ∨ b = max(a, b),
        a → b = ⊤ si a ≤ b, b en otro caso.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @classmethod
    def from_coherence(cls, value: float) -> "HeytingValue":
        """Clasifica una coherencia escalar según umbrales doctorales."""
        if value >= BicomplexConstants.COHERENT_THRESHOLD:
            return cls.COHERENT
        if value >= BicomplexConstants.DEGRADED_THRESHOLD:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingValue":
        """Elemento máximo (verdadero) del retículo: ⊤ = COHERENT."""
        return cls.COHERENT

    @classmethod
    def bottom(cls) -> "HeytingValue":
        """Elemento mínimo (falso) del retículo: ⊥ = VETOED."""
        return cls.VETOED

    @property
    def label(self) -> str:
        """Etiqueta canónica."""
        return self.name


def heyting_implication(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """
    Implicación pseudo-complementada de Heyting.

        a → b = ⊤  si a ≤ b,
                b  si a > b.

    Es el mayor elemento c tal que a ∧ c ≤ b (adjunto a derecha de ∧),
    garantizado por ser B₂ una cadena finita y por tanto un retículo
    completo y distributivo.
    """
    return HeytingValue.top() if a <= b else b


def heyting_conjunction(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """Conjunción de Heyting como mínimo del retículo (ínfimo de cadena)."""
    return min(a, b)


def heyting_disjunction(a: HeytingValue, b: HeytingValue) -> HeytingValue:
    """Disyunción de Heyting como máximo del retículo (supremo de cadena)."""
    return max(a, b)


def heyting_negation(a: HeytingValue) -> HeytingValue:
    """
    Pseudo-complemento de Heyting.

        ¬a := a → ⊥.

    En B₂ de tres valores esto produce ¬VETOED = COHERENT y
    ¬DEGRADED = ¬COHERENT = VETOED. Nótese que ¬¬DEGRADED = ¬VETOED =
    COHERENT ≠ DEGRADED: la negación **no es involutiva**, la firma
    algebraica que distingue una lógica intuicionista de una booleana
    (donde ¬¬a = a siempre).
    """
    return heyting_implication(a, HeytingValue.bottom())


#------------------------------------------------------------------------------
# 3.2 — ESTADO Y FÓRMULA DE COHERENCIA
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexCoherenceState:
    """
    Estado de coherencia por canal idempotente.

    Attributes:
        stability_1: S⁽¹⁾ ∈ [0,1].
        resonance_1: R⁽¹⁾ ∈ [0,1].
        entropy_1: H⁽¹⁾ ≥ 0.
        stability_2: S⁽²⁾ ∈ [0,1].
        resonance_2: R⁽²⁾ ∈ [0,1].
        entropy_2: H⁽²⁾ ≥ 0.
    """

    stability_1: float
    resonance_1: float
    entropy_1: float
    stability_2: float
    resonance_2: float
    entropy_2: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BicomplexCoherenceState":
        """
        Construye estado desde mapping con aliases robustos.

        Claves principales:
            stability_1, resonance_1, entropy_1,
            stability_2, resonance_2, entropy_2.
        """
        def get(keys: Sequence[str], default: float) -> float:
            for k in keys:
                if k in data:
                    return float(data[k])
            return default

        return cls(
            stability_1=get(("stability_1", "stability_channel_1", "S1", "S_1"), 0.0),
            resonance_1=get(("resonance_1", "resonance_channel_1", "R1", "R_1"), 0.0),
            entropy_1=get(("entropy_1", "entropy_channel_1", "H1", "H_1"), 0.0),
            stability_2=get(("stability_2", "stability_channel_2", "S2", "S_2"), 0.0),
            resonance_2=get(("resonance_2", "resonance_channel_2", "R2", "R_2"), 0.0),
            entropy_2=get(("entropy_2", "entropy_channel_2", "H2", "H_2"), 0.0),
        )

    @classmethod
    def from_cochain_state(cls, state: BicomplexCochainState) -> "BicomplexCoherenceState":
        """
        Deriva un estado de coherencia desde la cochain bicompleja pura,
        sin información topológica adicional.

        Definiciones proxy rigurosas:
          - estabilidad: magnitud media del canal, acotada a [0,1];
          - resonancia: 1 - coeficiente de variación, acotada a [0,1];
          - entropía: entropía de Shannon de la distribución de energía
            espectral |z_i|² normalizada.
        """
        z1, z2 = state.channel_arrays()
        s1, r1, h1 = _channel_statistics(z1)
        s2, r2, h2 = _channel_statistics(z2)
        return cls(
            stability_1=s1,
            resonance_1=r1,
            entropy_1=h1,
            stability_2=s2,
            resonance_2=r2,
            entropy_2=h2,
        )

    @classmethod
    def from_phase3_input(cls, phase3_input: Phase3Input) -> "BicomplexCoherenceState":
        """
        Deriva el estado de coherencia desde la frontera canónica Fase2→3.

        Este es el constructor que **cierra el ciclo categórico** del
        módulo: si `phase3_input` porta reportes de Mayer-Vietoris y/o
        Hodge, su índice de defecto topológico D⁽ᵃ⁾
        (`compute_topological_defect_index`) se añade aditivamente a la
        entropía espectral H⁽ᵃ⁾, penalizando la coherencia final:

            H⁽ᵃ⁾_efectiva = H⁽ᵃ⁾_espectral + D⁽ᵃ⁾.

        De este modo, una ruptura de exactitud Mayer-Vietoris o una
        desconexión armónica detectada en la Fase 2 se propaga
        cuantitativamente — no solo cualitativamente — hasta el veredicto
        de Heyting de la Fase 3.

        Args:
            phase3_input: Frontera producida por `phase2_export_to_phase3`.

        Returns:
            Estado de coherencia con entropía aumentada por defectos.

        Raises:
            ValueError: Si `phase3_input.cochain_state` es None.
        """
        if phase3_input.cochain_state is None:
            raise ValueError(
                "Phase3Input.cochain_state es requerido para derivar coherencia."
            )

        base = cls.from_cochain_state(phase3_input.cochain_state)
        d1, d2 = compute_topological_defect_index(
            phase3_input.mayer_vietoris_report, phase3_input.hodge_report
        )
        return cls(
            stability_1=base.stability_1,
            resonance_1=base.resonance_1,
            entropy_1=base.entropy_1 + d1,
            stability_2=base.stability_2,
            resonance_2=base.resonance_2,
            entropy_2=base.entropy_2 + d2,
        )


def _channel_statistics(z: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula (S, R, H) para un canal complejo.

    Definiciones:
        S = clamp(mean(|z_i|), 0, 1)                         (estabilidad),
        R = clamp(1 - std(|z_i|)/mean(|z_i|), 0, 1)          (resonancia),
        H = -Σ p_i ln p_i,  p_i = |z_i|² / Σ_j |z_j|²        (entropía
            de Shannon de la distribución de energía espectral).

    Args:
        z: Vector complejo del canal idempotente.

    Returns:
        (stability, resonance, entropy).
    """
    mag = np.abs(z).astype(np.float64)

    if mag.size == 0:
        return 0.0, 0.0, 0.0

    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))

    stability = _clamp(mean_mag, 0.0, 1.0)

    if mean_mag <= BicomplexConstants.EPSILON:
        resonance = 0.0
    else:
        resonance = _clamp(1.0 - std_mag / mean_mag, 0.0, 1.0)

    energy = float(np.sum(mag * mag))
    if energy <= BicomplexConstants.EPSILON or mag.size <= 1:
        entropy = 0.0
    else:
        p = (mag * mag) / energy
        p = p[p > 0.0]
        entropy = float(-_kbn_sum((p * np.log(p)).tolist()))

    return stability, resonance, entropy


def _coherence_formula(stability: float, resonance: float, entropy: float) -> float:
    """
    Fórmula de coherencia continua por canal — invariante [I4].

        C = clamp(S R / (1 + H), 0, 1).

    Monotonía: C es creciente en S y R, decreciente en H; el denominador
    (1 + H) garantiza que C → 0 cuando H → ∞ sin necesidad de truncamiento
    artificial, y C = S R cuando el sistema es perfectamente armónico
    (H = 0).
    """
    stability = max(0.0, min(1.0, float(stability)))
    resonance = max(0.0, min(1.0, float(resonance)))
    entropy = max(0.0, float(entropy))

    if stability <= BicomplexConstants.EPSILON or resonance <= BicomplexConstants.EPSILON:
        return 0.0

    return _clamp(stability * resonance / (1.0 + entropy), 0.0, 1.0)


#------------------------------------------------------------------------------
# 3.3 — ÍNDICE DE COHERENCIA, CENSURA CIBER-FÍSICA Y VERIFICACIÓN DE INVARIANTES
#------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BicomplexCoherenceCertificate:
    """
    Certificado final de coherencia y censura bicompleja.

    Attributes:
        coherence_channel_1: C⁽¹⁾.
        coherence_channel_2: C⁽²⁾.
        verdict_channel_1: v₁ ∈ HeytingValue.
        verdict_channel_2: v₂ ∈ HeytingValue.
        global_verdict: v₁ ∧ v₂.
        hardware_veto: True si el veredicto global es VETOED.
        gpio_pin: Pin de actuación ciber-física.
        actuation_ns: Cota temporal de ISR.
        implication_1_to_2: v₁ → v₂.
        implication_2_to_1: v₂ → v₁.
        topological_defect_channel_1: D⁽¹⁾ heredado de Fase 2 (0.0 si no
            disponible).
        topological_defect_channel_2: D⁽²⁾ heredado de Fase 2 (0.0 si no
            disponible).
    """

    coherence_channel_1: float
    coherence_channel_2: float
    verdict_channel_1: HeytingValue
    verdict_channel_2: HeytingValue
    global_verdict: HeytingValue
    hardware_veto: bool
    gpio_pin: int
    actuation_ns: float
    implication_1_to_2: HeytingValue
    implication_2_to_1: HeytingValue
    topological_defect_channel_1: float = 0.0
    topological_defect_channel_2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialización JSON-safe."""
        return {
            "coherence_channel_1": self.coherence_channel_1,
            "coherence_channel_2": self.coherence_channel_2,
            "verdict_channel_1": self.verdict_channel_1.name,
            "verdict_channel_2": self.verdict_channel_2.name,
            "global_verdict": self.global_verdict.name,
            "hardware_veto": self.hardware_veto,
            "gpio_pin": self.gpio_pin,
            "actuation_ns": self.actuation_ns,
            "implication_1_to_2": self.implication_1_to_2.name,
            "implication_2_to_1": self.implication_2_to_1.name,
            "topological_defect_channel_1": self.topological_defect_channel_1,
            "topological_defect_channel_2": self.topological_defect_channel_2,
        }


def vector_compute_bicomplex_coherence_index(
    state: Union[
        BicomplexCoherenceState,
        BicomplexCochainState,
        Phase3Input,
        Mapping[str, Any],
    ],
) -> BicomplexCoherenceCertificate:
    """
    FASE 3 — Índice de coherencia bicomplejo y filtro de Heyting.

    Calcula:

        C_{C₂} = C⁽¹⁾ e₁ + C⁽²⁾ e₂,

    donde

        C⁽ᵃ⁾ = clamp(S⁽ᵃ⁾ R⁽ᵃ⁾ / (1 + H⁽ᵃ⁾), 0, 1).

    Luego clasifica cada canal en el retículo de Heyting B₂ y emite el
    veredicto global por conjunción:

        Verdict_global = v₁ ∧ v₂.

    Si el veredicto global es VETOED, se activa la bandera de censura
    ciber-física sobre GPIO14, con cota de actuación ISR ≤ 398.95 ns.

    **Frontera canónica**: cuando `state` es un `Phase3Input` (la imagen
    exacta de `phase2_export_to_phase3`), la entropía H⁽ᵃ⁾ se aumenta con
    el índice de defecto topológico D⁽ᵃ⁾ heredado de la Fase 2
    (`BicomplexCoherenceState.from_phase3_input`), cerrando así el ciclo
    categórico completo Fase1 → Fase2 → Fase3.

    Args:
        state: Estado de coherencia explícito, mapping, cochain bicompleja
            pura, o `Phase3Input` (recomendado para trazabilidad completa).

    Returns:
        `BicomplexCoherenceCertificate`.

    Raises:
        TypeError: Si el estado no es convertible.
    """
    defect_1, defect_2 = 0.0, 0.0

    if isinstance(state, BicomplexCoherenceState):
        coherence_state = state
    elif isinstance(state, Phase3Input):
        coherence_state = BicomplexCoherenceState.from_phase3_input(state)
        defect_1, defect_2 = compute_topological_defect_index(
            state.mayer_vietoris_report, state.hodge_report
        )
    elif isinstance(state, BicomplexCochainState):
        coherence_state = BicomplexCoherenceState.from_cochain_state(state)
    elif isinstance(state, Mapping):
        coherence_state = BicomplexCoherenceState.from_mapping(state)
    else:
        raise TypeError(
            "state debe ser BicomplexCoherenceState, BicomplexCochainState, "
            "Phase3Input o Mapping."
        )

    c1 = _coherence_formula(
        coherence_state.stability_1,
        coherence_state.resonance_1,
        coherence_state.entropy_1,
    )
    c2 = _coherence_formula(
        coherence_state.stability_2,
        coherence_state.resonance_2,
        coherence_state.entropy_2,
    )

    v1 = HeytingValue.from_coherence(c1)
    v2 = HeytingValue.from_coherence(c2)
    global_verdict = heyting_conjunction(v1, v2)

    certificate = BicomplexCoherenceCertificate(
        coherence_channel_1=c1,
        coherence_channel_2=c2,
        verdict_channel_1=v1,
        verdict_channel_2=v2,
        global_verdict=global_verdict,
        hardware_veto=(global_verdict == HeytingValue.bottom()),
        gpio_pin=BicomplexConstants.GPIO_VETO_PIN,
        actuation_ns=BicomplexConstants.ISR_ACTUATION_NS,
        implication_1_to_2=heyting_implication(v1, v2),
        implication_2_to_1=heyting_implication(v2, v1),
        topological_defect_channel_1=defect_1,
        topological_defect_channel_2=defect_2,
    )

    if certificate.hardware_veto:
        logger.warning(
            "VETO bicomplejo activo: C1=%.6f, C2=%.6f, D1=%.3f, D2=%.3f, GPIO%d trigger.",
            c1,
            c2,
            defect_1,
            defect_2,
            certificate.gpio_pin,
        )
    else:
        logger.debug(
            "Coherencia bicompleja: C1=%.6f, C2=%.6f, global=%s",
            c1,
            c2,
            certificate.global_verdict.name,
        )

    return certificate


def verify_coherence_formula_invariant(
    state: BicomplexCoherenceState,
    certificate: BicomplexCoherenceCertificate,
    tol: float = BicomplexConstants.EPSILON,
) -> bool:
    """
    FASE 3 — Certificación explícita del invariante [I4].

    Recalcula independientemente C⁽¹⁾, C⁽²⁾ a partir de `state` mediante
    `_coherence_formula` y verifica su coincidencia con los valores
    reportados en `certificate`, dentro de tolerancia.

    Args:
        state: Estado de coherencia utilizado para generar el certificado.
        certificate: Certificado a auditar.
        tol: Tolerancia absoluta de comparación.

    Returns:
        True si ambos canales satisfacen [I4] dentro de tolerancia.
    """
    c1 = _coherence_formula(state.stability_1, state.resonance_1, state.entropy_1)
    c2 = _coherence_formula(state.stability_2, state.resonance_2, state.entropy_2)
    return (
        abs(c1 - certificate.coherence_channel_1) <= tol
        and abs(c2 - certificate.coherence_channel_2) <= tol
    )


def verify_heyting_censorship_invariant(
    certificate: BicomplexCoherenceCertificate,
) -> bool:
    """
    FASE 3 — Certificación explícita del invariante [I5].

        Verdict_global = v₁ ∧ v₂ = min(v₁, v₂).

    Args:
        certificate: Certificado de coherencia a auditar.

    Returns:
        True si el veredicto global almacenado coincide con la conjunción
        recalculada de los veredictos por canal, y si `hardware_veto` es
        consistente con `global_verdict == VETOED`.
    """
    expected_verdict = heyting_conjunction(
        certificate.verdict_channel_1, certificate.verdict_channel_2
    )
    verdict_ok = certificate.global_verdict == expected_verdict
    veto_ok = certificate.hardware_veto == (certificate.global_verdict == HeytingValue.bottom())
    return verdict_ok and veto_ok


#------------------------------------------------------------------------------
# 3.4 — ADAPTADORES A VectorResult Y COMPOSICIÓN DE FASES ANIDADAS
#------------------------------------------------------------------------------
def _elapsed_ms(start: float) -> float:
    """Convierte tiempo monotónico a milisegundos."""
    return (time.monotonic() - start) * 1000.0


def vector_parse_bicomplex_structure_result(S: NDArray[np.float64]) -> VectorResult:
    """
    Adaptador VectorResult para la Fase 1.

    Devuelve un `VectorResult` compatible con la MIC original, encapsulando
    el estado bicomplejo puro.
    """
    start = time.monotonic()
    try:
        state = vector_parse_bicomplex_structure(S)
        valid_count = sum(1 for c in state.banach_certificates if c.is_valid)
        integrity = valid_count / max(len(state.banach_certificates), 1)

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            memory_usage_mb=0.0,
            topological_coherence=1.0,
            algebraic_integrity=integrity,
        )
        return _build_success(
            stratum=Stratum.TACTICS,
            metrics=metrics,
            bicomplex_state=state.to_dict(),
        )
    except Exception as exc:
        logger.error("Error en vector_parse_bicomplex_structure_result: %s", exc, exc_info=True)
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.VALIDATION_ERROR,
            error=str(exc),
            metrics=VectorMetrics(processing_time_ms=_elapsed_ms(start)),
        )


def vector_audit_bicomplex_mayer_vietoris_result(
    A: Union[SubcomplexSpec, Mapping[str, Any]],
    B: Union[SubcomplexSpec, Mapping[str, Any]],
) -> VectorResult:
    """Adaptador VectorResult para auditoría Mayer-Vietoris dual."""
    start = time.monotonic()
    try:
        report = vector_audit_bicomplex_mayer_vietoris(A, B)
        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            topological_coherence=1.0 if report.exact else 0.0,
            algebraic_integrity=1.0 if report.exact else 0.0,
        )

        if not report.exact:
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=(
                    f"Mayer-Vietoris dual no exacto: "
                    f"Δβ₁⁽¹⁾={report.delta_beta_1_channel_1}, "
                    f"Δβ₁⁽²⁾={report.delta_beta_1_channel_2}."
                ),
                metrics=metrics,
                mayer_vietoris_report=report.to_dict(),
            )

        return _build_success(
            stratum=Stratum.TACTICS,
            metrics=metrics,
            mayer_vietoris_report=report.to_dict(),
        )
    except Exception as exc:
        logger.error("Error en vector_audit_bicomplex_mayer_vietoris_result: %s", exc, exc_info=True)
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.LOGIC_ERROR,
            error=str(exc),
            metrics=VectorMetrics(processing_time_ms=_elapsed_ms(start)),
        )


def vector_bicomplex_hodge_laplacian_result(
    cochain: BicomplexHodgeCochain,
    W_metric: Union[BicomplexHodgeMetric, float, np.ndarray, Tuple[Any, ...]],
) -> VectorResult:
    """Adaptador VectorResult para Hodge bicomplejo."""
    start = time.monotonic()
    try:
        report = vector_bicomplex_hodge_laplacian(cochain, W_metric)
        coherence = 1.0 if report.connected_dual else 0.0
        integrity = 1.0 / (
            1.0
            + report.harmonic_dimension_channel_1
            + report.harmonic_dimension_channel_2
        )

        nilpotency_ok = all(
            cert.is_nilpotent
            for cert in (report.nilpotency_channel_1, report.nilpotency_channel_2)
            if cert is not None
        )
        kodaira_ok = all(
            cert.is_consistent
            for cert in (report.hodge_kodaira_channel_1, report.hodge_kodaira_channel_2)
            if cert is not None
        )

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            topological_coherence=coherence,
            algebraic_integrity=integrity if (nilpotency_ok and kodaira_ok) else 0.0,
        )

        if not (nilpotency_ok and kodaira_ok):
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error="Violación de invariantes espectrales [I1]/[I3] en el Laplaciano de Hodge.",
                metrics=metrics,
                hodge_report=report.to_dict(),
            )

        return _build_success(
            stratum=Stratum.TACTICS,
            metrics=metrics,
            hodge_report=report.to_dict(),
        )
    except Exception as exc:
        logger.error("Error en vector_bicomplex_hodge_laplacian_result: %s", exc, exc_info=True)
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=str(exc),
            metrics=VectorMetrics(processing_time_ms=_elapsed_ms(start)),
        )


def vector_compute_bicomplex_coherence_index_result(
    state: Union[
        BicomplexCoherenceState,
        BicomplexCochainState,
        Phase3Input,
        Mapping[str, Any],
    ],
) -> VectorResult:
    """Adaptador VectorResult para coherencia y Heyting B₂."""
    start = time.monotonic()
    try:
        certificate = vector_compute_bicomplex_coherence_index(state)

        topological_coherence = min(
            certificate.coherence_channel_1,
            certificate.coherence_channel_2,
        )
        algebraic_integrity = (
            certificate.coherence_channel_1 + certificate.coherence_channel_2
        ) / 2.0

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            topological_coherence=topological_coherence,
            algebraic_integrity=algebraic_integrity,
        )

        if certificate.hardware_veto:
            return _build_error(
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=(
                    "Veto bicomplejo global: el sistema colapsa a VETOED; "
                    "se ordena actuación ciber-física."
                ),
                metrics=metrics,
                coherence_certificate=certificate.to_dict(),
            )

        return _build_success(
            stratum=Stratum.STRATEGY,
            metrics=metrics,
            coherence_certificate=certificate.to_dict(),
        )
    except Exception as exc:
        logger.error("Error en vector_compute_bicomplex_coherence_index_result: %s", exc, exc_info=True)
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=str(exc),
            metrics=VectorMetrics(processing_time_ms=_elapsed_ms(start)),
        )


def compose_bicomplex_nested_pipeline(
    S: NDArray[np.float64],
    *,
    subcomplex_a: Optional[Union[SubcomplexSpec, Mapping[str, Any]]] = None,
    subcomplex_b: Optional[Union[SubcomplexSpec, Mapping[str, Any]]] = None,
    hodge_cochain: Optional[BicomplexHodgeCochain] = None,
    W_metric: Optional[Union[BicomplexHodgeMetric, float, np.ndarray, Tuple[Any, ...]]] = None,
    coherence_state: Optional[Union[BicomplexCoherenceState, Mapping[str, Any]]] = None,
) -> VectorResult:
    """
    Composición maestra de las tres fases anidadas.

    Flujo categórico:

        S ──Φ₁──► BicomplexCochainState
          ──∂₂──► MayerVietoris / Hodge
          ──Φ₃──► BicomplexCoherenceCertificate

    La composición está definida sólo cuando la imagen de cada fase
    pertenece al dominio de la fase siguiente. Esta función preserva esa
    condición al usar los morfismos frontera `phase1_export_to_phase2` y
    `phase2_export_to_phase3`, y — a diferencia de composiciones previas —
    **alimenta el `Phase3Input` completo** (no solo su `cochain_state`) a
    `vector_compute_bicomplex_coherence_index`, de modo que los defectos
    topológicos detectados en Fase 2 penalizan efectivamente la coherencia
    de Fase 3.

    Args:
        S: Señal 4D de entrada.
        subcomplex_a: Subcomplejo A opcional para Mayer-Vietoris.
        subcomplex_b: Subcomplejo B opcional para Mayer-Vietoris.
        hodge_cochain: Cochain opcional para Hodge.
        W_metric: Métrica opcional para Hodge.
        coherence_state: Estado explícito opcional de coherencia (si se
            provee, sustituye por completo la derivación automática desde
            `Phase3Input`, incluyendo la penalización por defectos).

    Returns:
        `VectorResult` agregado con trazabilidad de fases.
    """
    start = time.monotonic()
    phase_outputs: Dict[str, Any] = {}

    try:
        # ── Fase 1 ────────────────────────────────────────────────────────
        cochain_state = vector_parse_bicomplex_structure(S)
        phase2_input = phase1_export_to_phase2(cochain_state)
        phase_outputs["phase1"] = phase2_input.cochain_state.to_dict()

        # ── Fase 2 ────────────────────────────────────────────────────────
        mv_report: Optional[MayerVietorisBicomplexReport] = None
        if subcomplex_a is not None and subcomplex_b is not None:
            mv_report = vector_audit_bicomplex_mayer_vietoris(subcomplex_a, subcomplex_b)
            phase_outputs["phase2_mayer_vietoris"] = mv_report.to_dict()

        hodge_report: Optional[BicomplexHodgeReport] = None
        if hodge_cochain is not None and W_metric is not None:
            hodge_report = vector_bicomplex_hodge_laplacian(hodge_cochain, W_metric)
            phase_outputs["phase2_hodge"] = hodge_report.to_dict()

        phase3_input = phase2_export_to_phase3(
            cochain_state=phase2_input.cochain_state,
            mayer_vietoris_report=mv_report,
            hodge_report=hodge_report,
        )

        # ── Fase 3 ────────────────────────────────────────────────────────
        coherence_obj: Union[BicomplexCoherenceState, Mapping[str, Any], Phase3Input]
        coherence_obj = phase3_input if coherence_state is None else coherence_state

        certificate = vector_compute_bicomplex_coherence_index(coherence_obj)
        phase_outputs["phase3"] = certificate.to_dict()

        # ── Síntesis de éxito ─────────────────────────────────────────────
        topology_ok = True
        if mv_report is not None and not mv_report.exact:
            topology_ok = False
        if hodge_report is not None and not hodge_report.connected_dual:
            topology_ok = False

        success = topology_ok and certificate.global_verdict != HeytingValue.bottom()

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            topological_coherence=min(
                certificate.coherence_channel_1,
                certificate.coherence_channel_2,
            ),
            algebraic_integrity=(
                certificate.coherence_channel_1 + certificate.coherence_channel_2
            )
            / 2.0,
        )

        if not success:
            return _build_error(
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=(
                    "Pipeline bicomplejo rechazado: topología no exacta o "
                    "veredicto global VETOED."
                ),
                metrics=metrics,
                phases=phase_outputs,
            )

        return _build_success(
            stratum=Stratum.STRATEGY,
            metrics=metrics,
            phases=phase_outputs,
        )

    except Exception as exc:
        logger.error("Error en compose_bicomplex_nested_pipeline: %s", exc, exc_info=True)
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=str(exc),
            metrics=VectorMetrics(processing_time_ms=_elapsed_ms(start)),
            phases=phase_outputs,
        )


__all__ = [
    # Constantes y resultados
    "BicomplexConstants",
    "VectorResultStatus",
    "VectorMetrics",
    "VectorResult",
    # Fase 1
    "BicomplexScalar",
    "verify_idempotent_algebra_identities",
    "BanachCertificate",
    "BicomplexCochainState",
    "vector_parse_bicomplex_structure",
    "verify_bicomplex_cochain_hmac",
    "Phase2Input",
    "phase1_export_to_phase2",
    # Fase 2
    "SubcomplexSpec",
    "MayerVietorisBicomplexReport",
    "vector_audit_bicomplex_mayer_vietoris",
    "BicomplexHodgeCochain",
    "BicomplexHodgeMetric",
    "NilpotencyCertificate",
    "verify_coboundary_nilpotency",
    "HodgeKodairaCertificate",
    "verify_hodge_kodaira_decomposition",
    "BicomplexHodgeReport",
    "vector_bicomplex_hodge_laplacian",
    "compute_topological_defect_index",
    "Phase3Input",
    "phase2_export_to_phase3",
    # Fase 3
    "HeytingValue",
    "heyting_implication",
    "heyting_conjunction",
    "heyting_disjunction",
    "heyting_negation",
    "BicomplexCoherenceState",
    "BicomplexCoherenceCertificate",
    "vector_compute_bicomplex_coherence_index",
    "verify_coherence_formula_invariant",
    "verify_heyting_censorship_invariant",
    # Adaptadores y pipeline
    "vector_parse_bicomplex_structure_result",
    "vector_audit_bicomplex_mayer_vietoris_result",
    "vector_bicomplex_hodge_laplacian_result",
    "vector_compute_bicomplex_coherence_index_result",
    "compose_bicomplex_nested_pipeline",
]