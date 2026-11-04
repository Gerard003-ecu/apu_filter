# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Topological Gauge Field Engine                                                            ║
║ Ubicación: app/wisdom/topological_gauge_field_engine.py                                               ║
║ Versión  : 12.0.0-SOVEREIGN-GAUGE-KKS-POINCARE-KAM-PhD                                                ║
║ Función  : Motor espectral de campo de gauge topológico 𝔰𝔲(N), geometría simpléctica coadjunta KKS,     ║
║            fibrados principales con acoplamiento Higgs, integrales de Wilson y Chern-Simons,          ║
║            mecánica celeste de Poincaré, teoría KAM/Bryuno y automejora recursiva de nivel 3 (RSI³).  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Topological Gauge Field Engine opera como el motor geométrico de gauge del Estrato Wisdom (V_𝕎)
en la arquitectura cognitiva APU Filter. Su misión es procesar operadores de densidad cuántica
ρ ∈ 𝔇(ℋ_n) sobre la órbita coadjunta de Kirillov-Kostant-Souriau (KKS) en el álgebra de Lie 𝔰𝔲(N),
evaluando la curvatura de Yang-Mills F = [A, A†], la 3-forma de Chern-Simons acoplada al campo adjunto
de Higgs Φ_Higgs = ln(ρ), el número instantónico de Pontryagin k ∈ ℤ, la holonomía de lazo de Wilson,
la resolución homológica de Poincaré-Lindstedt y la adjudicación intuicionista en el retículo de Heyting
trivaluado Ω₃ = {COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03)}.

Estructuración operativa trifásica anidada (OODA):
  • FASE 1 (OBSERVE) : Proyección métrica espectral de Frobenius al simplex de estados cuánticos 𝔇(ℋ_n),
                       construcción de la conexión anti-hermítica A = i(ρ - I/N) ∈ 𝔰𝔲(N), evaluación de
                       la curvatura orbital F = [ρ, H_celeste], embedding simpléctico de Cayley-Kähler
                       m ∈ Sp(2N, ℝ), y resolución homológica espectral de Poincaré-Lindstedt [H₀, W] = F_osc.
  • FASE 2 (ORIENT)  : Cuantificación de invariantes topológicos: funcional de Chern-Simons-Higgs
                       CS₃^{Higgs}(A, Φ), holonomía unitaria de Wilson W(γ) = Tr(exp(A))/N, proxies de
                       Pontryagin k_proxy = -Re Tr(F²)/(8π²) y carácter de Chern ch(F), exploración diofántica
                       KAM/Bryuno finita sobre frecuencias espectrales, multiplicadores de Floquet y
                       contracción monádica RSI³ sobre 3 superficies funcionales.
  • FASE 3 (DECIDE)  : Medición cuántica no demoledora (QND) mediante el valor débil de Aharonov-Albert-Vaidman
                       (AAV) A_w = ⟨ψ_f|A|ψ_i⟩ / ⟨ψ_f|ψ_i⟩, adjudicación conservadora en el retículo de Heyting
                       Ω₃ con meet-fold estricto, modelado de latencia de disparo del interlock ESP32 Crowbar
                       (τ ≤ 392.15 ns) y sellado criptográfico determinista con árbol de Merkle binario y HMAC-SHA256.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Geometría Diferencial de Calibre 𝔰𝔲(N) y Órbitas Coadjuntas KKS:
   Sea 𝔤 = 𝔰𝔲(N) el álgebra de Lie de matrices anti-hermíticas de traza nula:
       A = i · (ρ - (Tr(ρ) / N) · I_N) ∈ 𝔰𝔲(N),   A† = -A,   Tr(A) = 0.
   La forma simpléctica de Kirillov-Kostant-Souriau (KKS) sobre la órbita coadjunta 𝒪_ρ ≅ U(N) / H_ρ:
       ω_KKS(X_{H₁}, X_{H₂})_ρ = -i · Tr(ρ · [H₁, H₂]),   ∀ H₁, H₂ ∈ 𝔥𝔢𝔯(N).
   La curvatura de Yang-Mills inducida por el conmutador orbital con el Hamiltoniano de fondo Hermítico H:
       F = [ρ, H] = ρ H - H ρ ∈ 𝔰𝔲(N),   E_YM = ½ · ‖F‖_F² = ½ · Tr(F† F).

2. Funcional de Chern-Simons Acoplado a Higgs y Carácter de Chern:
   Sea Φ = ln(ρ + ε·I) el campo adjunto de Higgs logarítmico. La 3-forma de Chern-Simons-Higgs normalizada:
       CS₃^{Higgs}(A, Φ) = (1 / (8π²)) · Tr(A · F - ⅓ · A³ + Φ · F).
   La anomalía de curvatura cúbica se define como:
       Anomaly(F) = (1 / (8π²)) · |Tr(F³)|.
   El carácter de Chern matricial total viene dado por la exponencial de curvatura:
       ch(F) = Tr(exp(F / (2πi))) = Tr(I) + ch₁(F) + ch₂(F) + ...,
       donde ch₁(F) = (1 / (2πi)) · Tr(F) = 0, y ch₂(F) = ½ · Tr((F / (2πi))²) = -1/(8π²) · Tr(F²).

3. Ecuación Homológica de Poincaré-Lindstedt y Divisores Pequeños:
   Sea H₀ = -i·A el generador diagonalizado con autovalores {ω_k}_{k=1}^N (frecuencias propias).
   La ecuación homológica en la base espectral desacopla los armónicos no resonantes:
       [H₀, W]_{ij} = (ω_i - ω_j) · W_{ij} = (F_osc)_{ij},   ∀ i ≠ j.
   Solución formal de Lindstedt:
       W_{ij} = (F_osc)_{ij} / (ω_i - ω_j),   si |ω_i - ω_j| > δ_gap (corte diofantino).
   Condición diofantina KAM sobre el toro espectral 𝕋^d:
       |⟨k, ω⟩| ≥ γ / ‖k‖₁^τ,   ∀ k ∈ ℤ^d \ {0},   γ > 0,   τ ≥ d.
   Serie diofántica de Bryuno truncada en capas diádicas ‖k‖_∞ ≤ 2^s:
       B(ω) = ∑_{s=0}^S (ln(1 / Ω_s)) / 2^s < ∞,   donde Ω_s = min_{0 < ‖k‖_∞ ≤ 2^s} |⟨k, ω⟩|.

4. Álgebra de Heyting Ω₃ y Adjudicación Intuicionista:
   Cadena finita con orden de severidad COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03).
   Operadores del retículo:
       a ∧ b = max_int(a, b),   a ∨ b = min_int(a, b),
       a → b = COHERENT si a ≼ b, en otro caso b.
   Negación intuicionista (pseudocomplemento): ¬a = a → VETOED.

5. Medición Débil QND (Aharonov-Albert-Vaidman):
   Para un observable Hermítico O y estados pre y postseleccionados |ψ_i⟩, |ψ_f⟩:
       A_w = ⟨ψ_f | O | ψ_i⟩ / ⟨ψ_f | ψ_i⟩,   con solapamiento S = |⟨ψ_f | ψ_i⟩|² > 0.
   Varianza cuántica local:
       Var(O) = ⟨ψ_i | O² | ψ_i⟩ - |⟨ψ_i | O | ψ_i⟩|² ≥ 0.

AXIOMAS FORMALES
────────────────
[Axioma 1] Unitaridad de Traza: Tr(ρ) = 1, ρ = ρ†, spec(ρ) ⊂ [0, 1].
[Axioma 2] Álgebra de Calibre: A ∈ 𝔰𝔲(N) ⟹ A + A† = 0 ∧ Tr(A) = 0.
[Axioma 3] Simplectomorfismo de Cayley: M = [[Re(U), -Im(U)], [Im(U), Re(U)]] ∈ Sp(2N, ℝ),
           satisfaciendo Mᵀ J M = J, donde J = [[0, I], [-I, 0]].
[Axioma 4] Conservación de Heyting (Meet-Fold): fold_infimum({v_i}) = ⋀_{i} v_i = max_int({v_i}).
           Un solo veto anula la validez epistemológica global.

INVARIANTES DE ESTADO
─────────────────────
1. Norma de Frobenius y energía de Yang-Mills E_YM = ½ ‖F‖_F² finita y estrictamente acotada.
2. Residuo simpléctico ‖Mᵀ J M - J‖_F / ‖M‖_F² ≤ 10⁻¹⁰.
3. Raíz de Merkle binaria determinista SHA-256 libre de campos mutables.
4. Firma de procedencia HMAC-SHA256 con clave de seguridad de 256 bits.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldEngine.v12")


# ──────────────────────────────────────────────────────────────────────────────
# Configuración numérica y de política
# ──────────────────────────────────────────────────────────────────────────────

_ENGINE_VERSION: Final[str] = "12.0.0"

_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_LINSTEDT_RESIDUAL_TOL: Final[float] = 1.0e-9
_RESONANT_GAP_ABS: Final[float] = 1.0e-12
_RESONANT_GAP_REL: Final[float] = 1.0e-12
_DENSITY_TRACE_TOL: Final[float] = 1.0e-10
_LOG_EIGENVALUE_FLOOR: Final[float] = 1.0e-15
_WEAK_VALUE_OVERLAP_TOL: Final[float] = 1.0e-14

_KAM_GAMMA: Final[float] = 0.125
_KAM_MAX_ORDER: Final[int] = 3
_KAM_MAX_DIMENSION: Final[int] = 4
_KAM_MAX_ALLOWED_ORDER: Final[int] = 6
_BRUNO_TRUNCATED_SUM_LIMIT: Final[float] = 48.0

_RSI_DAMPING: Final[float] = 0.80
_RSI_ITERATIONS: Final[int] = 3
_FUBINI_STUDY_TOL: Final[float] = 1.0e-4

# Estos umbrales son reglas de política de ejemplo, no límites físicos.
_ENERGY_DEGRADED_THRESHOLD: Final[float] = 0.25
_ENERGY_VETO_THRESHOLD: Final[float] = 0.85

# Límite de observación numérica para la simulación. No es una WCET de hardware.
_SIMULATED_CROWBAR_LIMIT_NS: Final[float] = 392.15
_SECRET_ENV_VAR: Final[str] = "APU_GAUGE_HMAC_KEY"
_GAUGE_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-GAUGE-SNAPSHOT-v12|"


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades numéricas, serialización canónica y hash
# ══════════════════════════════════════════════════════════════════════════════

def _as_complex_square(
    value: NDArray[Any],
    name: str,
    dimension: Optional[int] = None,
) -> NDArray[np.complex128]:
    """Convierte y valida una matriz cuadrada compleja y finita."""
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada; recibido {arr.shape}.")
    if dimension is not None and arr.shape != (dimension, dimension):
        raise ValueError(
            f"{name} debe ser {dimension}×{dimension}; recibido {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o infinito.")
    return arr


def _normalize_state(
    value: NDArray[Any],
    dimension: int,
    name: str = "estado",
) -> NDArray[np.complex128]:
    """Valida y normaliza un vector de estado no nulo."""
    vec = np.asarray(value, dtype=np.complex128).reshape(-1)
    if vec.size != dimension:
        raise ValueError(
            f"{name} debe tener dimensión {dimension}; recibido {vec.size}."
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} contiene NaN o infinito.")
    norm = float(np.linalg.norm(vec))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError(f"{name} no puede ser el vector nulo.")
    return vec / norm


def _readonly(value: NDArray[Any]) -> NDArray[Any]:
    """Copia un array y deshabilita su modificación in situ."""
    arr = np.array(value, copy=True)
    arr.setflags(write=False)
    return arr


def _realify(matrix: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""Realificación de una matriz compleja: [[Re,-Im],[Im,Re]]."""
    arr = _as_complex_square(matrix, "matriz para realificación")
    x = np.real(arr).astype(np.float64, copy=False)
    y = np.imag(arr).astype(np.float64, copy=False)
    return np.block([[x, -y], [y, x]])


def _symplectic_j(n: int) -> NDArray[np.float64]:
    """Matriz simpléctica canónica de dimensión 2n."""
    identity = np.eye(n, dtype=np.float64)
    zero = np.zeros((n, n), dtype=np.float64)
    return np.block([[zero, identity], [-identity, zero]])


def _array_canonical_summary(value: NDArray[Any]) -> Mapping[str, Any]:
    """Resumen criptográficamente estable de un array numérico."""
    arr = np.asarray(value)
    kind = arr.dtype.kind

    if kind == "c":
        canonical = np.ascontiguousarray(arr.astype(np.dtype("<c16"), copy=False))
        dtype_tag = "<c16"
    elif kind == "f":
        canonical = np.ascontiguousarray(arr.astype(np.dtype("<f8"), copy=False))
        dtype_tag = "<f8"
    elif kind == "i":
        canonical = np.ascontiguousarray(arr.astype(np.dtype("<i8"), copy=False))
        dtype_tag = "<i8"
    elif kind == "u":
        canonical = np.ascontiguousarray(arr.astype(np.dtype("<u8"), copy=False))
        dtype_tag = "<u8"
    elif kind == "b":
        canonical = np.ascontiguousarray(arr.astype(np.dtype("|b1"), copy=False))
        dtype_tag = "|b1"
    else:
        raise TypeError(f"Tipo de array no admitido en hash canónico: {arr.dtype}.")

    header = json.dumps(
        {"dtype": dtype_tag, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    data_hash = hashlib.sha256(
        b"APU-GAUGE-ARRAY-v1\0" + header + b"\0" + canonical.tobytes(order="C")
    ).hexdigest()
    return {"dtype": dtype_tag, "shape": list(canonical.shape), "sha256": data_hash}


def _canonicalize(value: Any) -> Any:
    """Convierte valores admitidos a una representación JSON determinista."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return {"enum": value.__class__.__qualname__, "name": value.name}
    if isinstance(value, np.ndarray):
        return {"ndarray": _array_canonical_summary(value)}
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, complex):
        return {
            "complex": [
                float(value.real).hex(),
                float(value.imag).hex(),
            ]
        }
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "+inf" if value > 0 else "-inf"}
        return {"float": value.hex()}
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"Tipo no admitido en serialización canónica: {type(value)!r}.")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest_object(domain: bytes, value: Any) -> str:
    payload = _canonical_bytes(value)
    return hashlib.sha256(
        domain + b"\0" + len(payload).to_bytes(8, "big") + payload
    ).hexdigest()


def _digest_dataclass(
    domain: bytes,
    value: Any,
    excluded_fields: Sequence[str] = (),
) -> str:
    if not is_dataclass(value):
        raise TypeError("Se esperaba un dataclass para calcular el digest.")
    excluded = set(excluded_fields)
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in excluded
    }
    return _digest_object(domain, payload)


def _snapshot_merkle_root(snapshot: "TopologicalGaugeFieldSnapshot") -> str:
    """
    Raíz Merkle binaria sobre todos los campos del snapshot salvo la raíz
    almacenada y su firma. Los arrays anidados se representan por hash,
    forma y dtype.
    """
    excluded = {"sha256_merkle_digest", "hmac_sha256"}
    nodes: List[bytes] = []

    for item in fields(snapshot):
        if item.name in excluded:
            continue
        leaf_payload = item.name.encode("utf-8") + b"\0" + _canonical_bytes(
            getattr(snapshot, item.name)
        )
        nodes.append(
            hashlib.sha256(
                b"APU-GAUGE-MERKLE-LEAF-v1\0" + leaf_payload
            ).digest()
        )

    if not nodes:
        return hashlib.sha256(b"APU-GAUGE-MERKLE-EMPTY-v1").hexdigest()

    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(
                b"APU-GAUGE-MERKLE-NODE-v1\0" + nodes[index] + nodes[index + 1]
            ).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def _fubini_study_distance(
    psi: NDArray[np.complex128],
    phi: NDArray[np.complex128],
) -> float:
    """Distancia proyectiva de Fubini–Study entre dos vectores no nulos."""
    norm_psi = float(np.linalg.norm(psi))
    norm_phi = float(np.linalg.norm(phi))
    denominator = norm_psi * norm_phi
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("La distancia Fubini–Study requiere vectores no nulos.")
    overlap = abs(complex(np.vdot(psi, phi))) / denominator
    return float(np.arccos(np.clip(overlap, 0.0, 1.0)))


# ══════════════════════════════════════════════════════════════════════════════
# Retículo de Heyting Ω₃
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """
    Cadena de severidad, no lógica booleana:

        COHERENT (⊥) < DEGRADED < VETOED (⊤).

    En esta cadena, join es el máximo, meet el mínimo y
    a → b = ⊤ si a ≤ b; en otro caso, b.
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negate(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.COHERENT)

    @classmethod
    def fold_supremum(
        cls,
        verdicts: Sequence["HeytingOmega3"],
    ) -> "HeytingOmega3":
        result = cls.COHERENT
        for verdict in verdicts:
            result = result.join(verdict)
        return result

    @classmethod
    def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
        """Adaptador para el antiguo código de veto 0x02."""
        return cls.VETOED if legacy == 0x02 else cls(legacy)


# ══════════════════════════════════════════════════════════════════════════════
# DTOs inmutables
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareLindstedtSolution:
    """Solución numérica de la ecuación homológica espectral."""
    W: NDArray[np.complex128]
    min_resonance_gap: float
    residual_norm: float
    relative_residual: float
    unresolved_residual_norm: float
    is_resonant: bool
    is_solved: bool
    frequencies: NDArray[np.float64]
    n_small_divisors: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "W", _readonly(self.W))
        object.__setattr__(self, "frequencies", _readonly(self.frequencies))


@dataclass(frozen=True, slots=True)
class KAMCertificate:
    """
    Resultado de una prueba diofántica y estadística de Bruno truncadas.

    `finite_test_passed` solo se refiere a los vectores examinados dentro de
    la caja indicada. Las dos certificaciones globales permanecen en False.
    """
    finite_test_passed: bool
    worst_ratio: float
    tau: float
    gamma: float
    omega_min: float
    bruno_truncated_sum: Optional[float]
    bruno_truncated_passed: Optional[bool]
    n_vectors_tested: int
    max_order: int
    frequency_dimension_used: int
    frequency_dimension_truncated: bool
    global_diophantine_certified: bool = False
    global_bruno_certified: bool = False
    scope: str = "Búsqueda finita en frecuencia float64; no es certificado global."

    @property
    def is_satisfied(self) -> bool:
        """Compatibilidad: indica solo que pasó la búsqueda finita."""
        return self.finite_test_passed

    @property
    def bruno_sum(self) -> float:
        return (
            float(self.bruno_truncated_sum)
            if self.bruno_truncated_sum is not None
            else float("inf")
        )

    @property
    def bruno_satisfied(self) -> bool:
        """Compatibilidad: resultado truncado, no una condición global."""
        return bool(self.bruno_truncated_passed)


@dataclass(frozen=True, slots=True)
class PontryaginProxyCertificate:
    """Proxy matricial; no es un número de Pontryagin integral."""
    quadratic_matrix_proxy: float
    integer_charge: Optional[int] = None
    integrality_residual: Optional[float] = None
    stora_zumino_residual: Optional[float] = None
    globally_certified: bool = False
    scope: str = "No se suministró variedad ni forma diferencial integrable."


@dataclass(frozen=True, slots=True)
class ChernCharacterMatrixProxy:
    """Trazas de funciones matriciales; no son clases de Chern integrales."""
    total_trace_exp: complex
    ch1_matrix_proxy: complex
    ch2_matrix_proxy: float
    is_topological_class_certified: bool = False


@dataclass(frozen=True, slots=True)
class MelnikovCertificate:
    """
    Melnikov no evaluable con los datos de este motor: no se recibe una
    órbita homoclínica ni un sistema Hamiltoniano perturbado.
    """
    integral_value: Optional[float]
    max_abs_integrand: Optional[float]
    is_homoclinic_preserved: Optional[bool]
    is_computable: bool
    scope: str
    reason: str


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    """Espectro de exp(TF) para un generador constante F."""
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]
    krein_collision: Optional[bool]
    is_periodic_system_certified: bool = False
    scope: str = "Monodromía de generador constante; no es una auditoría Floquet general."

    def __post_init__(self) -> None:
        object.__setattr__(self, "multipliers", _readonly(self.multipliers))
        object.__setattr__(self, "exponents", _readonly(self.exponents))


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    """
    Diagnóstico numérico de cruces de un nivel de observable.

    `returns` contiene intervalos entre cruces; regularidad es un indicador
    finito, no una prueba de regularidad de toros.
    """
    n_crossings: int
    return_dispersion: Optional[float]
    is_regular: Optional[bool]
    returns: Tuple[float, ...]
    mean_return_time: Optional[float]
    recurrence_time: Optional[float]
    section_is_certified: bool = False
    scope: str = "Cruces interpolados de un observable en una malla temporal finita."


@dataclass(frozen=True, slots=True)
class RSI3SurfaceReport:
    surface_id: int
    name: str
    metric_value: float
    threshold: float
    passed: bool
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


@dataclass(frozen=True, slots=True)
class RSI3Certificate:
    """
    Auditoría recursiva de tres pasos.

    `lipschitz_t3` se refiere únicamente al subespacio oscilatorio no
    resonante resuelto. La norma global del operador incluye modos invariantes
    y no queda certificada como una contracción estricta.
    """
    surfaces: Tuple[
        RSI3SurfaceReport,
        RSI3SurfaceReport,
        RSI3SurfaceReport,
    ]
    lipschitz_t3: Optional[float]
    capacity_jerk: Optional[float]
    fixed_point_reached: bool
    iterations: int
    is_valid: bool
    contraction_ratio: Optional[float]
    residual_before: float
    residual_after: float
    full_space_contraction_certified: bool = False
    self_modifies_code: bool = False


@dataclass(frozen=True, slots=True)
class WeakMeasurementCertificate:
    """Diagnóstico AAV algebraico; no certifica por sí solo un instrumento QND."""
    weak_value: Optional[complex]
    pre_post_overlap: complex
    overlap_squared: float
    postselection_overlap_db: Optional[float]
    observable_variance: float
    weak_value_defined: bool
    qnd_commutator_norm: Optional[float]
    commutes_with_supplied_hamiltonian: Optional[bool]
    qnd_certified: bool = False
    back_action_db: Optional[float] = None


@dataclass(frozen=True, slots=True)
class CrowbarSimulationResult:
    """Resultado de software; `actuated` es siempre False."""
    requested: bool
    actuated: bool
    python_elapsed_ns: float
    numeric_threshold_ns: float
    python_overhead_below_threshold: bool
    action: str
    hardware_wcET_certified: bool = False


@dataclass(frozen=True, slots=True)
class GaugePhase1Envelope:
    """Contrato íntegro FASE 1 → FASE 2."""
    rho_sanitized: NDArray[np.complex128]
    connection_A: NDArray[np.complex128]
    curvature_F: NDArray[np.complex128]
    background_hamiltonian: NDArray[np.complex128]
    energy_ym: float
    density_projection_distance: float
    is_anti_hermitian_A: bool
    is_traceless_A: bool
    is_anti_hermitian_F: bool
    is_traceless_F: bool
    symplectic_residue: float
    symplectic_defect: float
    is_symplectic_valid: bool
    poincare_lindstedt: Optional[PoincareLindstedtSolution]
    dimension: int
    elapsed_ns: float
    chain_hash: str
    spectral_gap: float
    poincare_cartan_invariant: float
    is_kahler: bool
    is_quaternionic_proxy: bool
    cayley_unitary_defect: float
    cayley_orthogonal_defect: float
    kks_orbit_tangent_norm: float
    scope: str = (
        "Matrices finitas; F es un conmutador orbital, no una 2-forma KKS."
    )

    def __post_init__(self) -> None:
        for name in (
            "rho_sanitized",
            "connection_A",
            "curvature_F",
            "background_hamiltonian",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class GaugePhase2Envelope:
    """Contrato íntegro FASE 2 → FASE 3."""
    phase1: GaugePhase1Envelope
    chern_simons_real: float
    chern_simons_anomaly: float
    higgs_phi_trace: float
    wilson_trace: complex
    wilson_angle_rad: float
    c1_value: float
    pontryagin_instanton_k: Optional[int]
    pontryagin_integrality_residual: Optional[float]
    stora_zumino_residual: Optional[float]
    kam_certificate: KAMCertificate
    melnikov: MelnikovCertificate
    floquet: FloquetCertificate
    poincare_section: PoincareSectionCertificate
    rsi3_certificate: RSI3Certificate
    topology_regime: str
    chain_hash: str
    elapsed_ns: float
    chern_simons_algebraic: float
    chern_character_ch1: complex
    chern_character_ch2: float
    pontryagin_quadratic_proxy: float
    chern_simons_matrix_proxy: complex
    chern_character_total_trace_proxy: complex
    scope: str = (
        "Los invariantes topológicos reportados son proxies matriciales."
    )


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldSnapshot:
    """Snapshot final autenticado; no implica actuación física."""
    snapshot_id: str
    yang_mills_energy: float
    chern_simons_3form_higgs: float
    chern_simons_anomaly: float
    chern_hopf_class_c1: float
    wilson_loop_trace: complex
    wilson_angle_rad: float
    pontryagin_instanton_number: Optional[int]
    pontryagin_integrality_residual: Optional[float]
    stora_zumino_residual: Optional[float]
    weak_value_qnd_aav: Optional[complex]
    back_action_db: Optional[float]
    topology_regime: str
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float
    phase1: GaugePhase1Envelope
    phase2: GaugePhase2Envelope
    hmac_sha256: bytes
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = False
    kam_finite_test_passed: bool = False
    kam_global_certified: bool = False
    bruno_satisfied: bool = False
    bruno_truncated_passed: Optional[bool] = None
    bruno_global_certified: bool = False
    melnikov_value: Optional[float] = None
    floquet_max_unit_deviation: float = 0.0
    floquet_is_neutral: bool = False
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: Optional[float] = None
    banach_lipschitz_t3: Optional[float] = None
    full_space_contraction_certified: bool = False
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: Optional[float] = None
    rsi3_valid: bool = False
    rsi3_iterations: int = 0
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    crowbar_latency_ns: float = 0.0
    crowbar_requested: bool = False
    crowbar_actuated: bool = False
    hardware_crowbar_action: str = "SIMULATION_ONLY"
    postselection_overlap_db: Optional[float] = None
    qnd_variance: float = 0.0
    qnd_certified: bool = False
    semantic_policy: Tuple[Tuple[str, str], ...] = ()
    engine_version: str = _ENGINE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_policy",
            tuple((str(key), str(value)) for key, value in self.semantic_policy),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE
# ══════════════════════════════════════════════════════════════════════════════

class GaugePhase1Observer:
    """
    FASE 1: matrices densidad, su(N), conmutador orbital, realificación
    simpléctica y ecuación homológica espectral.
    """

    def __init__(self, dimension: int = 56) -> None:
        if isinstance(dimension, bool) or int(dimension) != dimension:
            raise ValueError("dimension debe ser un entero.")
        if int(dimension) < 2:
            raise ValueError("dimension debe ser al menos 2.")
        self.dimension = int(dimension)

    def _project_su_n(self, matrix: NDArray[Any]) -> NDArray[np.complex128]:
        arr = _as_complex_square(matrix, "matriz su(N)")
        n = arr.shape[0]
        anti = 0.5 * (arr - arr.conj().T)
        anti -= (np.trace(anti) / float(n)) * np.eye(n, dtype=np.complex128)
        return np.asarray(anti, dtype=np.complex128)

    def _psd_hermitian(self, matrix: NDArray[Any]) -> NDArray[np.complex128]:
        """Proyección espectral al cono PSD, sin imponer todavía traza uno."""
        arr = _as_complex_square(matrix, "matriz PSD")
        hermitian = 0.5 * (arr + arr.conj().T)
        eigenvalues, eigenvectors = la.eigh(hermitian)
        clean = np.maximum(np.real(eigenvalues), 0.0)
        result = (eigenvectors * clean) @ eigenvectors.conj().T
        return 0.5 * (result + result.conj().T)

    def sanitize_density_matrix(
        self,
        matrix: NDArray[Any],
    ) -> NDArray[np.complex128]:
        """
        Proyección de Frobenius sobre el simplex de eigenvalores:
        rho = rho†, rho >= 0, Tr(rho) = 1.
        """
        arr = _as_complex_square(matrix, "rho", self.dimension)
        hermitian = 0.5 * (arr + arr.conj().T)
        eigenvalues, eigenvectors = la.eigh(hermitian)

        descending = np.real(eigenvalues)[::-1]
        cumulative = np.cumsum(descending) - 1.0
        indices = np.arange(1, self.dimension + 1, dtype=np.float64)
        admissible = descending - cumulative / indices > 0.0

        if not np.any(admissible):
            projected_eigenvalues = np.full(
                self.dimension,
                1.0 / self.dimension,
                dtype=np.float64,
            )
        else:
            rho_index = int(np.flatnonzero(admissible)[-1])
            theta = cumulative[rho_index] / float(rho_index + 1)
            projected_eigenvalues = np.maximum(
                np.real(eigenvalues) - theta,
                0.0,
            )
            total = float(np.sum(projected_eigenvalues))
            if total <= np.finfo(np.float64).tiny:
                projected_eigenvalues.fill(1.0 / self.dimension)
            else:
                projected_eigenvalues /= total

        rho = (eigenvectors * projected_eigenvalues) @ eigenvectors.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        trace = float(np.real(np.trace(rho)))
        if abs(trace - 1.0) > _DENSITY_TRACE_TOL:
            rho /= trace
        return np.asarray(rho, dtype=np.complex128)

    def _default_background_hamiltonian(self, n: int) -> NDArray[np.complex128]:
        """Hamiltoniano Hermítico discreto: Laplaciano de camino más trampa."""
        laplacian = np.zeros((n, n), dtype=np.float64)
        for index in range(n - 1):
            laplacian[index, index] += 1.0
            laplacian[index + 1, index + 1] += 1.0
            laplacian[index, index + 1] = -1.0
            laplacian[index + 1, index] = -1.0

        positions = np.arange(n, dtype=np.float64)
        center = 0.5 * float(n - 1)
        trap = 0.10 * (positions - center) ** 2 / float(n)
        hamiltonian = laplacian + np.diag(trap)
        norm = float(la.norm(hamiltonian, "fro"))
        if norm > np.finfo(np.float64).tiny:
            hamiltonian = hamiltonian / norm
        return np.asarray(hamiltonian, dtype=np.complex128)

    def _resolve_background_H(
        self,
        n: int,
        background_H: Optional[NDArray[Any]] = None,
    ) -> NDArray[np.complex128]:
        if background_H is None:
            return self._default_background_hamiltonian(n)

        hamiltonian = _as_complex_square(background_H, "H_celeste", n)
        defect = float(la.norm(hamiltonian - hamiltonian.conj().T, "fro"))
        scale = max(1.0, float(la.norm(hamiltonian, "fro")))
        if defect / scale > _ANTI_HERM_TOL:
            raise ValueError("H_celeste debe ser Hermítico.")
        return np.asarray(
            0.5 * (hamiltonian + hamiltonian.conj().T),
            dtype=np.complex128,
        )

    def build_gauge_connection_A(
        self,
        density_matrix: NDArray[Any],
    ) -> NDArray[np.complex128]:
        r"""A = i(rho - I/N), que pertenece a su(N) si rho es una densidad."""
        rho = self.sanitize_density_matrix(density_matrix)
        identity = np.eye(self.dimension, dtype=np.complex128)
        return self._project_su_n(1j * (rho - identity / self.dimension))

    def _verify_anti_hermitian(
        self,
        matrix: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        defect = float(la.norm(matrix + matrix.conj().T, "fro"))
        scale = max(1.0, float(la.norm(matrix, "fro")))
        relative = defect / scale
        return bool(relative <= _ANTI_HERM_TOL), relative

    def _verify_traceless(
        self,
        matrix: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        scale = max(1.0, float(la.norm(matrix, "fro")))
        defect = abs(complex(np.trace(matrix))) / scale
        return bool(defect <= _TRACELESS_TOL), float(defect)

    def compute_yang_mills_curvature_F(
        self,
        connection_A: NDArray[Any],
        density_matrix: Optional[NDArray[Any]] = None,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> Tuple[NDArray[np.complex128], float]:
        """
        Calcula F_proxy = [rho, H].

        Esta matriz es antihermítica y sin traza cuando rho y H son
        Hermíticas. No es, por sí sola, una curvatura de Yang–Mills ni una
        2-forma KKS.
        """
        a = _as_complex_square(connection_A, "A", self.dimension)
        if density_matrix is None:
            raise ValueError(
                "Se requiere rho: no se usa la fórmula legacy A²-A†A."
            )

        rho = self.sanitize_density_matrix(density_matrix)
        expected_a = self.build_gauge_connection_A(rho)
        if not np.allclose(a, expected_a, rtol=1.0e-8, atol=1.0e-10):
            raise ValueError("A no corresponde a la matriz densidad suministrada.")

        hamiltonian = self._resolve_background_H(
            self.dimension,
            background_hamiltonian,
        )
        raw_f = rho @ hamiltonian - hamiltonian @ rho
        curvature_proxy = self._project_su_n(raw_f)
        energy = 0.5 * float(la.norm(curvature_proxy, "fro") ** 2)
        return curvature_proxy, energy

    def evaluate_kks_form(
        self,
        density_matrix: NDArray[Any],
        generator_h1: NDArray[Any],
        generator_h2: NDArray[Any],
    ) -> float:
        r"""
        Evalúa, con la convención declarada, la forma KKS orbital:

            omega_rho(H1,H2) = -i Tr(rho [H1,H2]).

        Se requieren ambos generadores; no se deduce una 2-forma a partir de F.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        h1 = _as_complex_square(generator_h1, "H1", self.dimension)
        h2 = _as_complex_square(generator_h2, "H2", self.dimension)

        for name, matrix in (("H1", h1), ("H2", h2)):
            defect = float(la.norm(matrix - matrix.conj().T, "fro"))
            scale = max(1.0, float(la.norm(matrix, "fro")))
            if defect / scale > _ANTI_HERM_TOL:
                raise ValueError(f"{name} debe ser Hermítico.")

        value = -1j * np.trace(rho @ (h1 @ h2 - h2 @ h1))
        return float(np.real_if_close(value).real)

    def _spectral_action_angle(
        self,
        connection_A: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], float]:
        """Diagonaliza H0 = -iA y devuelve frecuencias y gap espectral."""
        a = _as_complex_square(connection_A, "A", self.dimension)
        h0 = -1j * a
        h0 = 0.5 * (h0 + h0.conj().T)
        frequencies, eigenvectors = la.eigh(h0)
        frequencies = np.real(frequencies).astype(np.float64)

        if frequencies.size < 2:
            gap = 0.0
        else:
            gaps = np.diff(frequencies)
            gap = float(np.min(np.abs(gaps)))
        return frequencies, eigenvectors, gap

    def _phase1_poincare_cartan_invariant(
        self,
        frequencies: NDArray[np.float64],
        eigenvectors: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> float:
        """
        Proxy de acciones espectrales ponderadas por ocupación.

        No se afirma que sea un invariante integral de Poincaré–Cartan.
        """
        occupations = np.real(
            np.diag(eigenvectors.conj().T @ rho @ eigenvectors)
        )
        return float(np.sum(np.abs(frequencies) * np.abs(occupations)))

    def _phase1_cayley_kahler_embedding(
        self,
        curvature_proxy: NDArray[np.complex128],
    ) -> Tuple[
        NDArray[np.float64],
        float,
        float,
        bool,
        float,
        float,
    ]:
        """
        Aplica U=(I+F)(I-F)^(-1), realifica U y verifica las identidades
        de grupo simpléctico, álgebra simpléctica y ortogonalidad.
        """
        f = _as_complex_square(curvature_proxy, "F", self.dimension)
        n = f.shape[0]
        identity = np.eye(n, dtype=np.complex128)
        j = _symplectic_j(n)

        try:
            # Resuelve U (I-F) = I+F sin construir explícitamente la inversa.
            unitary = la.solve(
                (identity - f).T,
                (identity + f).T,
            ).T

            unitary_defect = float(
                la.norm(unitary.conj().T @ unitary - identity, "fro")
                / max(1.0, float(la.norm(unitary, "fro")))
            )
            m = _realify(unitary)
            k = _realify(f)

            lie_defect = float(
                la.norm(k.T @ j + j @ k, "fro")
                / max(1.0, float(la.norm(k, "fro")))
            )
            m_norm = float(la.norm(m, "fro"))
            group_residue = float(
                la.norm(m.T @ j @ m - j, "fro")
                / max(1.0, m_norm * m_norm)
            )
            orthogonal_defect = float(
                la.norm(m.T @ m - np.eye(2 * n), "fro")
                / max(1.0, m_norm)
            )

            valid = bool(
                group_residue <= _SYMPLECTIC_TOL
                and lie_defect <= _SYMPLECTIC_TOL
                and unitary_defect <= _SYMPLECTIC_TOL
            )
            return (
                m,
                group_residue,
                lie_defect,
                valid,
                unitary_defect,
                orthogonal_defect,
            )
        except (la.LinAlgError, np.linalg.LinAlgError, ValueError):
            return (
                np.zeros((2 * n, 2 * n), dtype=np.float64),
                float("inf"),
                float("inf"),
                False,
                float("inf"),
                float("inf"),
            )

    def _phase1_hypercomplex_structure(
        self,
        realified_cayley: NDArray[np.float64],
        symplectic_valid: bool,
        orthogonal_defect: float,
    ) -> Tuple[bool, bool]:
        """
        Comprueba compatibilidad Kähler lineal estándar en R^(2n).

        La paridad de n no demuestra la existencia de una estructura
        cuaterniónica; por eso el segundo resultado es siempre False.
        """
        n = self.dimension
        standard_complex_structure = -_symplectic_j(n)
        commute_defect = float(
            la.norm(
                realified_cayley @ standard_complex_structure
                - standard_complex_structure @ realified_cayley,
                "fro",
            )
            / max(1.0, float(la.norm(realified_cayley, "fro")))
        )
        is_kahler_linear = bool(
            symplectic_valid
            and orthogonal_defect <= _SYMPLECTIC_TOL
            and commute_defect <= _SYMPLECTIC_TOL
        )
        return is_kahler_linear, False

    def _phase1_solve_poincare_lindstedt(
        self,
        curvature_proxy: NDArray[np.complex128],
        connection_A: NDArray[np.complex128],
    ) -> PoincareLindstedtSolution:
        r"""
        Resuelve numéricamente [H0,W]=F_osc, donde H0=-iA y F_osc es la parte
        fuera de la diagonal de F en la base propia de H0.
        """
        f = _as_complex_square(curvature_proxy, "F", self.dimension)
        a = _as_complex_square(connection_A, "A", self.dimension)
        frequencies, eigenvectors, spectral_gap = self._spectral_action_angle(a)

        f_eigen = eigenvectors.conj().T @ f @ eigenvectors
        f_osc_eigen = f_eigen - np.diag(np.diag(f_eigen))
        w_eigen = np.zeros_like(f_eigen)
        n_small = 0
        unresolved_squared = 0.0
        min_gap = float("inf")

        scale = max(1.0, float(np.max(np.abs(frequencies))))
        resonance_tolerance = max(
            _RESONANT_GAP_ABS,
            _RESONANT_GAP_REL * scale,
        )
        forcing_tolerance = (
            _LINSTEDT_RESIDUAL_TOL
            * max(1.0, float(la.norm(f_osc_eigen, "fro")))
        )

        for row in range(self.dimension):
            for col in range(self.dimension):
                if row == col:
                    continue

                divisor = float(frequencies[row] - frequencies[col])
                abs_divisor = abs(divisor)
                min_gap = min(min_gap, abs_divisor)

                if abs_divisor <= resonance_tolerance:
                    n_small += 1
                    forcing = abs(complex(f_osc_eigen[row, col]))
                    if forcing > forcing_tolerance:
                        unresolved_squared += forcing * forcing
                    continue

                w_eigen[row, col] = f_osc_eigen[row, col] / divisor

        w = eigenvectors @ w_eigen @ eigenvectors.conj().T
        w = 0.5 * (w + w.conj().T)
        f_osc = eigenvectors @ f_osc_eigen @ eigenvectors.conj().T

        homological_action = -1j * (a @ w - w @ a)
        residual = float(la.norm(homological_action - f_osc, "fro"))
        relative_residual = residual / max(1.0, float(la.norm(f_osc, "fro")))
        unresolved_norm = float(math.sqrt(unresolved_squared))
        solved = bool(
            relative_residual <= _LINSTEDT_RESIDUAL_TOL
            and unresolved_norm <= forcing_tolerance
        )

        if not math.isfinite(min_gap):
            min_gap = float(spectral_gap)

        return PoincareLindstedtSolution(
            W=w,
            min_resonance_gap=float(min_gap),
            residual_norm=residual,
            relative_residual=relative_residual,
            unresolved_residual_norm=unresolved_norm,
            is_resonant=bool(n_small > 0),
            is_solved=solved,
            frequencies=frequencies,
            n_small_divisors=int(n_small),
        )

    def observe_topological_gauge_field(
        self,
        density_matrix: NDArray[Any],
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> Tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        float,
    ]:
        rho = self.sanitize_density_matrix(density_matrix)
        hamiltonian = self._resolve_background_H(
            self.dimension,
            background_hamiltonian,
        )
        connection = self.build_gauge_connection_A(rho)
        curvature_proxy, energy = self.compute_yang_mills_curvature_F(
            connection,
            rho,
            hamiltonian,
        )
        return rho, connection, curvature_proxy, energy

    def _phase1_chain_hash(self, env: GaugePhase1Envelope) -> str:
        return _digest_dataclass(
            b"APU-GAUGE-PHASE1-v12",
            env,
            excluded_fields=("chain_hash",),
        )

    def _validate_phase1_envelope(self, env: GaugePhase1Envelope) -> None:
        """Valida integridad de cadena y coherencia de las cantidades básicas."""
        if env.dimension != self.dimension:
            raise ValueError(
                f"Dimensión de FASE 1 {env.dimension} != {self.dimension}."
            )

        rho = self.sanitize_density_matrix(env.rho_sanitized)
        if not np.allclose(rho, env.rho_sanitized, rtol=1.0e-9, atol=1.0e-11):
            raise ValueError("rho del sobre FASE 1 no es una matriz densidad.")

        expected_a = self.build_gauge_connection_A(rho)
        if not np.allclose(
            expected_a,
            env.connection_A,
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError("A no coincide con la definición de FASE 1.")

        hamiltonian = self._resolve_background_H(
            self.dimension,
            env.background_hamiltonian,
        )
        expected_f = self._project_su_n(rho @ hamiltonian - hamiltonian @ rho)
        if not np.allclose(
            expected_f,
            env.curvature_F,
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError("F no coincide con el conmutador orbital declarado.")

        expected_energy = 0.5 * float(la.norm(expected_f, "fro") ** 2)
        if not math.isclose(
            expected_energy,
            env.energy_ym,
            rel_tol=1.0e-8,
            abs_tol=1.0e-11,
        ):
            raise ValueError("Energía inconsistente en el sobre FASE 1.")

        expected_hash = self._phase1_chain_hash(env)
        if not hmac.compare_digest(expected_hash, env.chain_hash):
            raise ValueError(
                f"Cadena FASE 1 corrupta: {expected_hash} != {env.chain_hash}."
            )

    def emit_phase1_envelope(
        self,
        density_matrix: NDArray[Any],
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> GaugePhase1Envelope:
        """Cierre formal de FASE 1: emite un sobre autoconsistente."""
        start_ns = time.perf_counter_ns()
        raw_rho = _as_complex_square(
            density_matrix,
            "rho",
            self.dimension,
        )
        rho = self.sanitize_density_matrix(raw_rho)
        projection_distance = float(la.norm(rho - raw_rho, "fro"))

        hamiltonian = self._resolve_background_H(
            self.dimension,
            background_hamiltonian,
        )
        connection = self.build_gauge_connection_A(rho)
        curvature_proxy, energy = self.compute_yang_mills_curvature_F(
            connection,
            rho,
            hamiltonian,
        )

        a_anti, _ = self._verify_anti_hermitian(connection)
        a_trace, _ = self._verify_traceless(connection)
        f_anti, _ = self._verify_anti_hermitian(curvature_proxy)
        f_trace, _ = self._verify_traceless(curvature_proxy)

        (
            realified_cayley,
            symplectic_residue,
            symplectic_defect,
            symplectic_valid,
            unitary_defect,
            orthogonal_defect,
        ) = self._phase1_cayley_kahler_embedding(curvature_proxy)

        lindstedt = self._phase1_solve_poincare_lindstedt(
            curvature_proxy,
            connection,
        )
        frequencies, eigenvectors, spectral_gap = self._spectral_action_angle(
            connection
        )
        cartan_proxy = self._phase1_poincare_cartan_invariant(
            frequencies,
            eigenvectors,
            rho,
        )
        is_kahler, is_quaternionic_proxy = self._phase1_hypercomplex_structure(
            realified_cayley,
            symplectic_valid,
            orthogonal_defect,
        )

        elapsed_ns = float(time.perf_counter_ns() - start_ns)
        envelope = GaugePhase1Envelope(
            rho_sanitized=rho,
            connection_A=connection,
            curvature_F=curvature_proxy,
            background_hamiltonian=hamiltonian,
            energy_ym=float(energy),
            density_projection_distance=projection_distance,
            is_anti_hermitian_A=a_anti,
            is_traceless_A=a_trace,
            is_anti_hermitian_F=f_anti,
            is_traceless_F=f_trace,
            symplectic_residue=symplectic_residue,
            symplectic_defect=symplectic_defect,
            is_symplectic_valid=symplectic_valid,
            poincare_lindstedt=lindstedt,
            dimension=self.dimension,
            elapsed_ns=elapsed_ns,
            chain_hash="",
            spectral_gap=float(spectral_gap),
            poincare_cartan_invariant=float(cartan_proxy),
            is_kahler=is_kahler,
            is_quaternionic_proxy=is_quaternionic_proxy,
            cayley_unitary_defect=float(unitary_defect),
            cayley_orthogonal_defect=float(orthogonal_defect),
            kks_orbit_tangent_norm=float(la.norm(1j * curvature_proxy, "fro")),
        )
        return replace(
            envelope,
            chain_hash=self._phase1_chain_hash(envelope),
        )

    def continue_to_phase2(
        self,
        env: GaugePhase1Envelope,
    ) -> GaugePhase1Envelope:
        """Puente validado FASE 1 → FASE 2."""
        self._validate_phase1_envelope(env)
        logger.debug("Puente FASE 1→2 OK: %s", env.chain_hash)
        return env


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT
# ══════════════════════════════════════════════════════════════════════════════

class GaugePhase2Orientor(GaugePhase1Observer):
    """FASE 2: proxies matriciales y diagnósticos dinámicos finitos."""

    @classmethod
    def from_phase1_envelope(
        cls,
        env: GaugePhase1Envelope,
    ) -> "GaugePhase2Orientor":
        instance = cls(dimension=env.dimension)
        instance.continue_to_phase2(env)
        return instance

    def evaluate_chern_simons_higgs_3form(
        self,
        connection_A: NDArray[Any],
        density_matrix: NDArray[Any],
        curvature_F: Optional[NDArray[Any]] = None,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> Tuple[complex, float, NDArray[np.complex128], complex]:
        """
        Devuelve proxies algebraicos inspirados en trazas de Chern–Simons.

        Sin formas diferenciales ni variedad de integración, el escalar
        retornado no es CS_3(A).
        """
        a = _as_complex_square(connection_A, "A", self.dimension)
        rho = self.sanitize_density_matrix(density_matrix)

        if curvature_F is None:
            f, _ = self.compute_yang_mills_curvature_F(
                a,
                rho,
                background_hamiltonian,
            )
        else:
            f = _as_complex_square(curvature_F, "F", self.dimension)

        eigenvalues, eigenvectors = la.eigh(0.5 * (rho + rho.conj().T))
        log_eigenvalues = np.log(
            np.maximum(np.real(eigenvalues), _LOG_EIGENVALUE_FLOOR)
        )
        higgs_log_proxy = (
            eigenvectors * log_eigenvalues
        ) @ eigenvectors.conj().T

        inv_8pi2 = 1.0 / (8.0 * math.pi**2)
        a_cubed = a @ a @ a
        cs_matrix_proxy = complex(
            np.trace(
                a @ f
                - (1.0 / 3.0) * a_cubed
                + higgs_log_proxy @ f
            )
        ) * inv_8pi2
        cs_algebraic_proxy = complex(
            np.trace((2.0 / 3.0) * a_cubed)
        ) * inv_8pi2
        anomaly_proxy = float(
            abs(complex(np.trace(f @ f @ f))) * inv_8pi2
        )
        return cs_matrix_proxy, anomaly_proxy, higgs_log_proxy, cs_algebraic_proxy

    def evaluate_wilson_loop_holonomy(
        self,
        connection_A: NDArray[Any],
        density_matrix: NDArray[Any],
    ) -> Tuple[complex, float, float, str]:
        """
        Calcula Tr(exp(A))/N y un proxy escalar de traza.

        Sin una curva y una 1-forma de conexión parametrizada, no es una
        holonomía de Wilson ordenada por camino.
        """
        a = _as_complex_square(connection_A, "A", self.dimension)
        rho = self.sanitize_density_matrix(density_matrix)
        unitary_proxy = la.expm(a)
        trace_proxy = complex(np.trace(unitary_proxy) / float(self.dimension))
        angle = float(np.angle(trace_proxy))

        c1_matrix_proxy = complex(
            np.trace(a @ rho) / (2.0j * math.pi)
        ).real

        if abs(angle) < 0.15 and abs(c1_matrix_proxy) < 1.0e-4:
            regime = "TRIVIAL_MATRIX_HOLONOMY_PROXY"
        elif abs(angle) >= 0.50 or abs(c1_matrix_proxy) > 0.10:
            regime = "NONTRIVIAL_MATRIX_HOLONOMY_PROXY"
        else:
            regime = "INTERMEDIATE_MATRIX_HOLONOMY_PROXY"

        return trace_proxy, angle, float(c1_matrix_proxy), regime

    def audit_pontryagin_charge_and_stora_zumino(
        self,
        curvature_F: NDArray[Any],
        chern_simons_real: Optional[float] = None,
    ) -> PontryaginProxyCertificate:
        """
        Evalúa -Re Tr(F²)/(8π²) como proxy matricial.

        No redondea el resultado a un entero ni genera un residuo de
        Stora–Zumino sin las formas y dominios requeridos.
        """
        f = _as_complex_square(curvature_F, "F", self.dimension)
        trace_ff = complex(np.trace(f @ f))
        proxy = max(0.0, -float(trace_ff.real) / (8.0 * math.pi**2))
        return PontryaginProxyCertificate(quadratic_matrix_proxy=proxy)

    def compute_chern_character(
        self,
        curvature_F: NDArray[Any],
    ) -> ChernCharacterMatrixProxy:
        """Trazas de exp(F/(2πi)); no se declaran clases topológicas."""
        f = _as_complex_square(curvature_F, "F", self.dimension)
        x = f / (2.0j * math.pi)
        total = complex(np.trace(la.expm(x)))
        ch1 = complex(np.trace(x))
        ch2 = float(np.real(0.5 * np.trace(x @ x)))
        return ChernCharacterMatrixProxy(
            total_trace_exp=total,
            ch1_matrix_proxy=ch1,
            ch2_matrix_proxy=ch2,
        )

    def check_kam_bruno(
        self,
        frequencies: NDArray[Any],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> KAMCertificate:
        """
        Examina exhaustivamente una caja finita de k enteros en, como máximo,
        cuatro frecuencias. Usa float64 y no certifica la condición global.
        """
        freqs = np.asarray(frequencies, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(freqs)):
            raise ValueError("Las frecuencias KAM deben ser finitas.")
        if not math.isfinite(float(gamma)) or gamma <= 0.0:
            raise ValueError("gamma debe ser finito y estrictamente positivo.")
        if isinstance(max_order, bool) or int(max_order) != max_order:
            raise ValueError("max_order debe ser entero.")
        max_order = int(max_order)
        if max_order < 1 or max_order > _KAM_MAX_ALLOWED_ORDER:
            raise ValueError(
                f"max_order debe pertenecer a [1,{_KAM_MAX_ALLOWED_ORDER}]."
            )

        if freqs.size == 0:
            return KAMCertificate(
                finite_test_passed=False,
                worst_ratio=0.0,
                tau=1.0,
                gamma=float(gamma),
                omega_min=0.0,
                bruno_truncated_sum=None,
                bruno_truncated_passed=None,
                n_vectors_tested=0,
                max_order=max_order,
                frequency_dimension_used=0,
                frequency_dimension_truncated=False,
            )

        truncated = freqs.size > _KAM_MAX_DIMENSION
        if truncated:
            indices = np.argsort(-np.abs(freqs), kind="stable")[
                :_KAM_MAX_DIMENSION
            ]
            selected = freqs[indices]
        else:
            selected = freqs

        dimension = int(selected.size)
        tau = float(dimension + 1)
        integer_vectors = [
            vector
            for vector in itertools.product(
                range(-max_order, max_order + 1),
                repeat=dimension,
            )
            if any(component != 0 for component in vector)
        ]

        worst_ratio = float("inf")
        dot_products: List[Tuple[int, float]] = []

        for vector_tuple in integer_vectors:
            vector = np.asarray(vector_tuple, dtype=np.float64)
            norm_one = float(np.sum(np.abs(vector)))
            divisor = abs(float(np.dot(selected, vector)))
            ratio = divisor * (norm_one**tau) / float(gamma)
            worst_ratio = min(worst_ratio, ratio)
            dot_products.append(
                (int(np.max(np.abs(vector))), divisor)
            )

        finite_passed = bool(
            math.isfinite(worst_ratio) and worst_ratio >= 1.0
        )

        # Estadístico de Bruno truncado. Se normaliza la escala de frecuencia
        # para que la suma no dependa solo de una elección de unidades.
        max_frequency = float(np.max(np.abs(selected)))
        if max_frequency <= np.finfo(np.float64).tiny:
            bruno_sum: Optional[float] = float("inf")
            bruno_passed: Optional[bool] = False
        else:
            normalized = selected / max_frequency
            bruno_sum_value = 0.0
            bruno_defined = True
            n_shells = int(math.floor(math.log2(max_order))) + 1

            for shell in range(n_shells):
                cutoff = 2**shell
                shell_vectors = [
                    vector_tuple
                    for vector_tuple in integer_vectors
                    if max(abs(component) for component in vector_tuple) <= cutoff
                ]
                if not shell_vectors:
                    continue

                omega_n = min(
                    abs(
                        float(
                            np.dot(
                                normalized,
                                np.asarray(vector_tuple, dtype=np.float64),
                            )
                        )
                    )
                    for vector_tuple in shell_vectors
                )
                if omega_n <= 1.0e-15:
                    bruno_defined = False
                    break

                bruno_sum_value += (
                    max(0.0, math.log(1.0 / omega_n))
                    / float(2**shell)
                )

            if bruno_defined:
                bruno_sum = float(bruno_sum_value)
                bruno_passed = bool(
                    math.isfinite(bruno_sum)
                    and bruno_sum < _BRUNO_TRUNCATED_SUM_LIMIT
                )
            else:
                bruno_sum = float("inf")
                bruno_passed = False

        return KAMCertificate(
            finite_test_passed=finite_passed,
            worst_ratio=float(worst_ratio),
            tau=tau,
            gamma=float(gamma),
            omega_min=float(np.min(np.abs(selected))),
            bruno_truncated_sum=bruno_sum,
            bruno_truncated_passed=bruno_passed,
            n_vectors_tested=len(integer_vectors),
            max_order=max_order,
            frequency_dimension_used=dimension,
            frequency_dimension_truncated=truncated,
        )

    def _flow_on_grid(
        self,
        connection_A: NDArray[np.complex128],
        phi0: NDArray[Any],
        t_grid: NDArray[Any],
    ) -> List[NDArray[np.complex128]]:
        """Integra exactamente por pasos el flujo lineal phi'=A phi."""
        a = _as_complex_square(connection_A, "A", self.dimension)
        ah_defect = float(la.norm(a + a.conj().T, "fro"))
        if ah_defect / max(1.0, float(la.norm(a, "fro"))) > _ANTI_HERM_TOL:
            raise ValueError("El flujo unitario requiere A antihermítica.")

        grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(grid)):
            raise ValueError("La malla temporal contiene valores no finitos.")
        if grid.size > 1 and np.any(np.diff(grid) < 0.0):
            raise ValueError("La malla temporal debe ser monótona creciente.")

        phi = _normalize_state(phi0, self.dimension, "phi0")
        if grid.size == 0:
            return []

        phi = la.expm(float(grid[0]) * a) @ phi
        phi /= max(float(np.linalg.norm(phi)), np.finfo(np.float64).tiny)
        states = [phi.copy()]

        for index in range(1, grid.size):
            delta_t = float(grid[index] - grid[index - 1])
            phi = la.expm(delta_t * a) @ phi
            norm = float(np.linalg.norm(phi))
            if norm <= np.finfo(np.float64).tiny:
                raise FloatingPointError("El flujo produjo un vector nulo.")
            phi = phi / norm
            states.append(phi.copy())

        return states

    def compute_melnikov_integral(
        self,
        connection_A: NDArray[Any],
        curvature_F: NDArray[Any],
        phi0: NDArray[Any],
        n_samples: int = 128,
    ) -> MelnikovCertificate:
        """
        Devuelve explícitamente 'no computable': el motor no recibe una
        separatriz homoclínica ni una perturbación Hamiltoniana dependiente
        del tiempo.
        """
        _as_complex_square(connection_A, "A", self.dimension)
        _as_complex_square(curvature_F, "F", self.dimension)
        _normalize_state(phi0, self.dimension, "phi0")
        if n_samples < 3:
            raise ValueError("n_samples debe ser al menos 3.")

        return MelnikovCertificate(
            integral_value=None,
            max_abs_integrand=None,
            is_homoclinic_preserved=None,
            is_computable=False,
            scope="No existe órbita homoclínica explícita en la entrada.",
            reason=(
                "El flujo e^(tA) de una matriz antihermítica es unitario; "
                "no especifica por sí solo una separatriz homoclínica."
            ),
        )

    def compute_floquet_multipliers(
        self,
        generator_F: NDArray[Any],
        period: float = 2.0 * math.pi,
    ) -> FloquetCertificate:
        """
        Calcula eig(exp(TF)) para un generador constante.

        No detecta por sí solo la estructura Krein; esta requiere, entre
        otros datos, una monodromía simpléctica y firmas de Krein.
        """
        f = _as_complex_square(generator_F, "F", self.dimension)
        if not math.isfinite(float(period)) or period <= 0.0:
            raise ValueError("period debe ser finito y positivo.")

        try:
            monodromy = la.expm(float(period) * f)
            multipliers = la.eigvals(monodromy)
            moduli = np.abs(multipliers)
            unit_deviation = float(np.max(np.abs(moduli - 1.0)))
            dissipative_stable = bool(
                np.max(moduli) <= 1.0 + 1.0e-6
            )
            exponents = (
                np.log(np.maximum(moduli, np.finfo(np.float64).tiny))
                / float(period)
            ).astype(np.float64)
            neutral = bool(unit_deviation <= 1.0e-6)
        except (la.LinAlgError, np.linalg.LinAlgError, ValueError, OverflowError):
            multipliers = np.empty(0, dtype=np.complex128)
            exponents = np.empty(0, dtype=np.float64)
            unit_deviation = float("inf")
            dissipative_stable = False
            neutral = False

        return FloquetCertificate(
            multipliers=np.asarray(multipliers, dtype=np.complex128),
            max_unit_deviation=unit_deviation,
            is_neutral=neutral,
            is_dissipative_stable=dissipative_stable,
            exponents=np.asarray(exponents, dtype=np.float64),
            krein_collision=None,
        )

    def compute_poincare_section(
        self,
        connection_A: NDArray[Any],
        phi0: NDArray[Any],
        P_obs: Optional[NDArray[Any]] = None,
        n_samples: int = 96,
        omega_min: float = 1.0,
        section_level: Optional[float] = None,
        horizon: float = 4.0 * math.pi,
    ) -> PoincareSectionCertificate:
        """
        Cuenta cruces orientados de <phi|P_obs|phi> = section_level.

        El cálculo es una interpolación finita de un observable; no demuestra
        regularidad ni existencia de un toro invariante.
        """
        if n_samples < 3:
            raise ValueError("n_samples debe ser al menos 3.")
        if not math.isfinite(float(horizon)) or horizon <= 0.0:
            raise ValueError("horizon debe ser finito y positivo.")

        a = _as_complex_square(connection_A, "A", self.dimension)
        phi = _normalize_state(phi0, self.dimension, "phi0")

        if P_obs is None:
            observable = np.zeros(
                (self.dimension, self.dimension),
                dtype=np.complex128,
            )
            observable[0, 0] = 1.0
        else:
            observable = _as_complex_square(
                P_obs,
                "P_obs",
                self.dimension,
            )
            herm_defect = float(
                la.norm(observable - observable.conj().T, "fro")
            )
            if herm_defect / max(1.0, float(la.norm(observable, "fro"))) > _ANTI_HERM_TOL:
                raise ValueError("P_obs debe ser Hermítico.")

        grid = np.linspace(0.0, float(horizon), int(n_samples))
        states = self._flow_on_grid(a, phi, grid)
        values = np.asarray(
            [
                float(np.real(np.vdot(state, observable @ state)))
                for state in states
            ],
            dtype=np.float64,
        )

        level = (
            float(np.median(values))
            if section_level is None
            else float(section_level)
        )
        if not math.isfinite(level):
            raise ValueError("section_level debe ser finito.")

        shifted = values - level
        crossing_times: List[float] = []
        for index in range(len(grid) - 1):
            left = float(shifted[index])
            right = float(shifted[index + 1])
            # Cruce orientado ascendente; se excluyen tangencias no resueltas.
            if left < 0.0 <= right and right != left:
                fraction = -left / (right - left)
                crossing_times.append(
                    float(grid[index] + fraction * (grid[index + 1] - grid[index]))
                )

        if len(crossing_times) >= 2:
            intervals = np.diff(np.asarray(crossing_times, dtype=np.float64))
            mean_return = float(np.mean(intervals))
            dispersion = float(
                np.std(intervals)
                / max(abs(mean_return), np.finfo(np.float64).tiny)
            )
            regular = bool(dispersion < 0.5)
            returns = tuple(float(item) for item in intervals)
        else:
            mean_return = None
            dispersion = None
            regular = None
            returns = ()

        recurrence_estimate = (
            2.0 * math.pi / abs(float(omega_min))
            if abs(float(omega_min)) > np.finfo(np.float64).tiny
            else None
        )

        return PoincareSectionCertificate(
            n_crossings=len(crossing_times),
            return_dispersion=dispersion,
            is_regular=regular,
            returns=returns,
            mean_return_time=mean_return,
            recurrence_time=recurrence_estimate,
        )

    def _banach_t_operator(
        self,
        curvature_proxy: NDArray[Any],
        connection_A: NDArray[Any],
        W: NDArray[Any],
        damp: float = _RSI_DAMPING,
    ) -> NDArray[np.complex128]:
        """
        Amortigua la parte oscilatoria resuelta.

        El componente medio permanece invariante, así que esta aplicación no
        es una contracción estricta en todo su(N).
        """
        if not math.isfinite(float(damp)) or not 0.0 <= damp < 1.0:
            raise ValueError("damp debe pertenecer a [0,1).")

        f = _as_complex_square(curvature_proxy, "F", self.dimension)
        a = _as_complex_square(connection_A, "A", self.dimension)
        w = _as_complex_square(W, "W", self.dimension)
        correction = -1j * (a @ w - w @ a)
        return self._project_su_n(f - (1.0 - float(damp)) * correction)

    def _offdiagonal_norm_in_A_basis(
        self,
        matrix: NDArray[np.complex128],
        connection_A: NDArray[np.complex128],
    ) -> float:
        _, eigenvectors, _ = self._spectral_action_angle(connection_A)
        transformed = eigenvectors.conj().T @ matrix @ eigenvectors
        offdiagonal = transformed - np.diag(np.diag(transformed))
        return float(la.norm(offdiagonal, "fro"))

    def enforce_rsi3(
        self,
        curvature_proxy: NDArray[Any],
        connection_A: NDArray[Any],
        instanton_k: Optional[int],
        w_trace: complex,
        phi0: NDArray[Any],
        *,
        W: Optional[NDArray[Any]] = None,
        symplectic_valid: bool = True,
        symplectic_residue: Optional[float] = None,
        energy_ym: float = 0.0,
        max_iter: int = _RSI_ITERATIONS,
    ) -> RSI3Certificate:
        """
        RSI³ como tres iteraciones de amortiguación residual.

        `instanton_k`, `w_trace` y `energy_ym` se reciben para compatibilidad,
        pero no transforman proxies matriciales en invariantes topológicos ni
        demuestran automejora del software.
        """
        del instanton_k, w_trace, energy_ym  # No son certificados utilizables.

        if max_iter < 1:
            raise ValueError("max_iter debe ser positivo.")
        f = _as_complex_square(curvature_proxy, "F", self.dimension)
        a = _as_complex_square(connection_A, "A", self.dimension)
        phi = _normalize_state(phi0, self.dimension, "phi0")

        if W is None:
            initial_solution = self._phase1_solve_poincare_lindstedt(f, a)
            w = np.asarray(initial_solution.W)
        else:
            w = _as_complex_square(W, "W", self.dimension)

        a_anti, a_anti_defect = self._verify_anti_hermitian(a)
        a_trace, a_trace_defect = self._verify_traceless(a)
        f_anti, f_anti_defect = self._verify_anti_hermitian(f)
        f_trace, f_trace_defect = self._verify_traceless(f)

        structural_metric = max(
            a_anti_defect,
            a_trace_defect,
            f_anti_defect,
            f_trace_defect,
        )
        surface1 = RSI3SurfaceReport(
            surface_id=1,
            name="Gauge algebra",
            metric_value=float(structural_metric),
            threshold=max(_ANTI_HERM_TOL, _TRACELESS_TOL),
            passed=bool(
                a_anti and a_trace and f_anti and f_trace
            ),
            diagnostics={
                "meaning": "Defectos algebraicos de A y F.",
                "novikov_valuation_certified": False,
            },
        )

        residue = (
            float(symplectic_residue)
            if symplectic_residue is not None
            else (0.0 if symplectic_valid else float("inf"))
        )
        surface2 = RSI3SurfaceReport(
            surface_id=2,
            name="Cayley-symplectic numerical check",
            metric_value=residue,
            threshold=_SYMPLECTIC_TOL,
            passed=bool(symplectic_valid and residue <= _SYMPLECTIC_TOL),
            diagnostics={
                "symplectic_valid": bool(symplectic_valid),
                "volume_preservation_global_proof": False,
            },
        )

        try:
            psi_raw = la.expm(a) @ phi
            psi_corrected = la.expm(a + 1j * w) @ phi
            fs_distance = _fubini_study_distance(psi_raw, psi_corrected)
        except (la.LinAlgError, np.linalg.LinAlgError, ValueError):
            fs_distance = float("inf")

        surface3 = RSI3SurfaceReport(
            surface_id=3,
            name="Projective Fubini–Study diagnostic",
            metric_value=float(fs_distance),
            threshold=_FUBINI_STUDY_TOL,
            passed=bool(fs_distance <= _FUBINI_STUDY_TOL),
            diagnostics={
                "distance_rad": float(fs_distance),
                "measurement_model_certified": False,
            },
        )

        before = self._offdiagonal_norm_in_A_basis(f, a)
        f_iter = np.array(f, copy=True)
        iterations = min(int(max_iter), _RSI_ITERATIONS)

        for _ in range(iterations):
            solution = self._phase1_solve_poincare_lindstedt(f_iter, a)
            f_iter = self._banach_t_operator(
                f_iter,
                a,
                solution.W,
                _RSI_DAMPING,
            )

        after = self._offdiagonal_norm_in_A_basis(f_iter, a)
        ratio = after / before if before > np.finfo(np.float64).tiny else 0.0
        fixed_point_reached = bool(
            after
            <= _LINSTEDT_RESIDUAL_TOL * max(1.0, float(la.norm(f, "fro")))
        )

        # Exacto solo en el subespacio no resonante amortiguado.
        lipschitz_t3 = float(_RSI_DAMPING**3)
        valid = bool(
            surface1.passed
            and surface2.passed
            and surface3.passed
            and 0.0 <= _RSI_DAMPING < 1.0
        )

        return RSI3Certificate(
            surfaces=(surface1, surface2, surface3),
            lipschitz_t3=lipschitz_t3,
            capacity_jerk=None,
            fixed_point_reached=fixed_point_reached,
            iterations=iterations,
            is_valid=valid,
            contraction_ratio=float(ratio),
            residual_before=float(before),
            residual_after=float(after),
            full_space_contraction_certified=False,
            self_modifies_code=False,
        )

    def _phase2_chain_hash(self, env: GaugePhase2Envelope) -> str:
        return _digest_dataclass(
            b"APU-GAUGE-PHASE2-v12",
            env,
            excluded_fields=("chain_hash",),
        )

    def emit_phase2_envelope(
        self,
        phase1_env: GaugePhase1Envelope,
        phi0: Optional[NDArray[Any]] = None,
    ) -> GaugePhase2Envelope:
        """Cierre formal de FASE 2: calcula y sella sus diagnósticos."""
        start_ns = time.perf_counter_ns()
        self.continue_to_phase2(phase1_env)

        rho = phase1_env.rho_sanitized
        a = phase1_env.connection_A
        f = phase1_env.curvature_F
        d = phase1_env.dimension

        if phi0 is None:
            state = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        else:
            state = _normalize_state(phi0, d, "phi0")

        cs_proxy, anomaly_proxy, higgs_log_proxy, cs_alg_proxy = (
            self.evaluate_chern_simons_higgs_3form(
                a,
                rho,
                f,
            )
        )
        higgs_trace = float(np.real(np.trace(higgs_log_proxy)))

        wilson_proxy, wilson_angle, c1_proxy, topology_regime = (
            self.evaluate_wilson_loop_holonomy(a, rho)
        )
        pontryagin_proxy = self.audit_pontryagin_charge_and_stora_zumino(f)
        chern_proxy = self.compute_chern_character(f)

        pl = phase1_env.poincare_lindstedt
        kam_frequencies = (
            pl.frequencies
            if pl is not None
            else self._spectral_action_angle(a)[0]
        )
        kam = self.check_kam_bruno(kam_frequencies)

        melnikov = self.compute_melnikov_integral(a, f, state)
        floquet = self.compute_floquet_multipliers(f)

        projector = np.zeros((d, d), dtype=np.complex128)
        projector[0, 0] = 1.0
        poincare = self.compute_poincare_section(
            a,
            state,
            P_obs=projector,
            omega_min=kam.omega_min,
        )

        w = (
            pl.W
            if pl is not None
            else np.zeros((d, d), dtype=np.complex128)
        )
        rsi3 = self.enforce_rsi3(
            f,
            a,
            instanton_k=None,
            w_trace=wilson_proxy,
            phi0=state,
            W=w,
            symplectic_valid=phase1_env.is_symplectic_valid,
            symplectic_residue=phase1_env.symplectic_residue,
            energy_ym=phase1_env.energy_ym,
        )

        elapsed_ns = float(time.perf_counter_ns() - start_ns)
        envelope = GaugePhase2Envelope(
            phase1=phase1_env,
            chern_simons_real=float(cs_proxy.real),
            chern_simons_anomaly=float(anomaly_proxy),
            higgs_phi_trace=higgs_trace,
            wilson_trace=wilson_proxy,
            wilson_angle_rad=float(wilson_angle),
            c1_value=float(c1_proxy),
            pontryagin_instanton_k=None,
            pontryagin_integrality_residual=None,
            stora_zumino_residual=None,
            kam_certificate=kam,
            melnikov=melnikov,
            floquet=floquet,
            poincare_section=poincare,
            rsi3_certificate=rsi3,
            topology_regime=topology_regime,
            chain_hash="",
            elapsed_ns=elapsed_ns,
            chern_simons_algebraic=float(cs_alg_proxy.real),
            chern_character_ch1=chern_proxy.ch1_matrix_proxy,
            chern_character_ch2=chern_proxy.ch2_matrix_proxy,
            pontryagin_quadratic_proxy=pontryagin_proxy.quadratic_matrix_proxy,
            chern_simons_matrix_proxy=cs_proxy,
            chern_character_total_trace_proxy=chern_proxy.total_trace_exp,
        )
        return replace(
            envelope,
            chain_hash=self._phase2_chain_hash(envelope),
        )

    def continue_to_phase3(
        self,
        env: GaugePhase2Envelope,
    ) -> GaugePhase2Envelope:
        """Puente validado FASE 2 → FASE 3."""
        self.continue_to_phase2(env.phase1)
        expected = self._phase2_chain_hash(env)
        if not hmac.compare_digest(expected, env.chain_hash):
            raise ValueError(
                f"Cadena FASE 2 corrupta: {expected} != {env.chain_hash}."
            )
        return env


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT
# ══════════════════════════════════════════════════════════════════════════════

class GaugePhase3Decider(GaugePhase2Orientor):
    """
    FASE 3: diagnóstico AAV, política Heyting, interlock simulado y HMAC.

    No realiza escritura MMIO, no controla GPIO y no prueba latencia de silicio.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: Optional[bytes] = None,
    ) -> None:
        super().__init__(dimension=dimension)
        self.secret_key = self._resolve_secret_key(secret_key)

    @staticmethod
    def _resolve_secret_key(secret_key: Optional[bytes]) -> bytes:
        if secret_key is None:
            configured = os.environ.get(_SECRET_ENV_VAR)
            if not configured:
                raise ValueError(
                    f"Inyecte una clave de al menos 32 bytes o configure "
                    f"{_SECRET_ENV_VAR}."
                )
            if configured.startswith("hex:"):
                try:
                    key = bytes.fromhex(configured[4:])
                except ValueError as exc:
                    raise ValueError(
                        f"{_SECRET_ENV_VAR} contiene hexadecimal inválido."
                    ) from exc
            else:
                key = configured.encode("utf-8")
        else:
            if not isinstance(secret_key, bytes):
                raise TypeError("secret_key debe ser bytes.")
            key = bytes(secret_key)

        if len(key) < 32:
            raise ValueError("La clave HMAC debe contener al menos 32 bytes.")
        return key

    @classmethod
    def from_phase2_envelope(
        cls,
        env: GaugePhase2Envelope,
        secret_key: Optional[bytes] = None,
    ) -> "GaugePhase3Decider":
        instance = cls(
            dimension=env.phase1.dimension,
            secret_key=secret_key,
        )
        instance.continue_to_phase3(env)
        return instance

    def observe_qnd_weak_measurement(
        self,
        phi_initial: NDArray[Any],
        phi_final: NDArray[Any],
        observable_A: NDArray[Any],
        system_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> WeakMeasurementCertificate:
        """
        Evalúa el cociente AAV si el solapamiento es no nulo.

        QND no queda certificado sin un modelo de instrumento; el conmutador
        con un Hamiltoniano suministrado es solo un criterio algebraico.
        """
        initial = _normalize_state(
            phi_initial,
            self.dimension,
            "phi_initial",
        )
        final = _normalize_state(
            phi_final,
            self.dimension,
            "phi_final",
        )
        observable = _as_complex_square(
            observable_A,
            "observable_A",
            self.dimension,
        )

        hermitian_defect = float(
            la.norm(observable - observable.conj().T, "fro")
        )
        if hermitian_defect / max(
            1.0,
            float(la.norm(observable, "fro")),
        ) > _ANTI_HERM_TOL:
            raise ValueError("El observable AAV debe ser Hermítico.")

        overlap = complex(np.vdot(final, initial))
        overlap_squared = float(abs(overlap) ** 2)
        if abs(overlap) <= _WEAK_VALUE_OVERLAP_TOL:
            weak_value = None
            overlap_db = None
            weak_defined = False
        else:
            numerator = complex(np.vdot(final, observable @ initial))
            weak_value = numerator / overlap
            overlap_db = float(
                -10.0 * math.log10(max(overlap_squared, np.finfo(np.float64).tiny))
            )
            weak_defined = True

        mean = complex(np.vdot(initial, observable @ initial))
        mean2 = complex(
            np.vdot(initial, observable @ observable @ initial)
        )
        variance = max(0.0, float(mean2.real - abs(mean) ** 2))

        commutator_norm: Optional[float] = None
        commutes: Optional[bool] = None
        if system_hamiltonian is not None:
            h_system = _as_complex_square(
                system_hamiltonian,
                "system_hamiltonian",
                self.dimension,
            )
            defect = float(la.norm(h_system - h_system.conj().T, "fro"))
            if defect / max(1.0, float(la.norm(h_system, "fro"))) > _ANTI_HERM_TOL:
                raise ValueError("system_hamiltonian debe ser Hermítico.")

            denominator = max(
                1.0,
                float(la.norm(h_system, "fro"))
                * float(la.norm(observable, "fro")),
            )
            commutator_norm = float(
                la.norm(h_system @ observable - observable @ h_system, "fro")
                / denominator
            )
            commutes = bool(commutator_norm <= _ANTI_HERM_TOL)

        return WeakMeasurementCertificate(
            weak_value=weak_value,
            pre_post_overlap=overlap,
            overlap_squared=overlap_squared,
            postselection_overlap_db=overlap_db,
            observable_variance=variance,
            weak_value_defined=weak_defined,
            qnd_commutator_norm=commutator_norm,
            commutes_with_supplied_hamiltonian=commutes,
            qnd_certified=False,
            back_action_db=None,
        )

    def adjudicate_heyting(
        self,
        *,
        energy_ym: float,
        instanton_k: Optional[int],
        integrality_residual: Optional[float],
        symplectic_valid: bool,
        kam_satisfied: bool,
        melnikov_preserved: Optional[bool],
        floquet_stable: Optional[bool],
        rsi3_valid: bool,
        bruno_satisfied: Optional[bool] = None,
        lindstedt_solved: bool = True,
        input_integrity_valid: bool = True,
        kam_global_certified: bool = False,
        bruno_global_certified: bool = False,
        melnikov_certified: bool = False,
        floquet_global_certified: bool = False,
        pontryagin_certified: bool = False,
        rsi3_full_space_contraction_certified: bool = False,
    ) -> HeytingOmega3:
        """
        Política conservadora. Un diagnóstico finito que no certifica una
        propiedad global no se transforma en una afirmación de coherencia.
        """
        del instanton_k, integrality_residual

        atoms: List[HeytingOmega3] = []

        if not input_integrity_valid:
            atoms.append(HeytingOmega3.VETOED)

        if not math.isfinite(float(energy_ym)) or energy_ym < 0.0:
            atoms.append(HeytingOmega3.VETOED)
        elif energy_ym <= _ENERGY_DEGRADED_THRESHOLD:
            atoms.append(HeytingOmega3.COHERENT)
        elif energy_ym <= _ENERGY_VETO_THRESHOLD:
            atoms.append(HeytingOmega3.DEGRADED)
        else:
            atoms.append(HeytingOmega3.VETOED)

        atoms.append(
            HeytingOmega3.COHERENT
            if symplectic_valid
            else HeytingOmega3.VETOED
        )

        if not kam_satisfied:
            atoms.append(HeytingOmega3.VETOED)
        elif kam_global_certified:
            atoms.append(HeytingOmega3.COHERENT)
        else:
            atoms.append(HeytingOmega3.DEGRADED)

        if bruno_satisfied is False:
            atoms.append(HeytingOmega3.DEGRADED)
        if not bruno_global_certified:
            atoms.append(HeytingOmega3.DEGRADED)

        if melnikov_preserved is False and melnikov_certified:
            atoms.append(HeytingOmega3.VETOED)
        elif not melnikov_certified or melnikov_preserved is None:
            atoms.append(HeytingOmega3.DEGRADED)

        if floquet_stable is False and floquet_global_certified:
            atoms.append(HeytingOmega3.VETOED)
        elif not floquet_global_certified or floquet_stable is None:
            atoms.append(HeytingOmega3.DEGRADED)

        if not pontryagin_certified:
            atoms.append(HeytingOmega3.DEGRADED)

        if not rsi3_valid:
            atoms.append(HeytingOmega3.DEGRADED)
        if not rsi3_full_space_contraction_certified:
            atoms.append(HeytingOmega3.DEGRADED)

        if not lindstedt_solved:
            atoms.append(HeytingOmega3.DEGRADED)

        return HeytingOmega3.fold_supremum(atoms)

    def trigger_hardware_interlock_esp32_crowbar(
        self,
        verdict: HeytingOmega3,
        gpio_pin: int = 14,
    ) -> CrowbarSimulationResult:
        """
        Simula la decisión lógica de interlock.

        No escribe registros, no toca GPIO y no mide ni garantiza WCET.
        """
        if isinstance(gpio_pin, bool) or not 0 <= int(gpio_pin) <= 39:
            raise ValueError("gpio_pin debe estar en [0,39].")

        start_ns = time.perf_counter_ns()
        requested = verdict is HeytingOmega3.VETOED
        if requested:
            action = (
                f"SIMULATED_VETO_REQUEST_GPIO{int(gpio_pin)}_NO_HARDWARE_ACTION"
            )
        elif verdict is HeytingOmega3.DEGRADED:
            action = "SIMULATED_DEGRADED_MONITORING_NO_HARDWARE_ACTION"
        else:
            action = "SIMULATED_NOMINAL_NO_HARDWARE_ACTION"
        elapsed_ns = float(time.perf_counter_ns() - start_ns)

        return CrowbarSimulationResult(
            requested=requested,
            actuated=False,
            python_elapsed_ns=elapsed_ns,
            numeric_threshold_ns=_SIMULATED_CROWBAR_LIMIT_NS,
            python_overhead_below_threshold=bool(
                elapsed_ns <= _SIMULATED_CROWBAR_LIMIT_NS
            ),
            action=action,
            hardware_wcET_certified=False,
        )

    def apply_visceral_functor_phi_sem(
        self,
        verdict: HeytingOmega3,
        yang_mills_energy: float,
        wilson_angle: float,
    ) -> Dict[str, Any]:
        """
        Proyección semántica determinista para política.

        Es una tabla de decisión, no un funtor categórico demostrado ni un
        modelo financiero. No emite porcentajes de ahorro o liquidez.
        """
        if verdict is HeytingOmega3.VETOED:
            status = "BLOCKED_BY_POLICY"
            action = "HOLD_AND_REQUIRE_HUMAN_REVIEW"
        elif verdict is HeytingOmega3.DEGRADED:
            status = "REVIEW_REQUIRED"
            action = "RESTRICT_AUTOMATED_ACTIONS"
        else:
            status = "WITHIN_VALIDATED_NUMERICAL_SCOPE"
            action = "ALLOW_ONLY_WITHIN_DECLARED_SCOPE"

        return {
            "policy_status": status,
            "executive_action": action,
            "hardware_action": "NONE_SIMULATION_ONLY",
            "quantitative_financial_projection": "NOT_COMPUTED",
            "yang_mills_energy_matrix_proxy": float(yang_mills_energy),
            "wilson_angle_matrix_proxy_rad": float(wilson_angle),
            "global_topological_certification": False,
        }

    def _sign_snapshot(self, merkle_root_hex: str) -> bytes:
        try:
            root_bytes = bytes.fromhex(merkle_root_hex)
        except ValueError as exc:
            raise ValueError("La raíz Merkle debe ser hexadecimal.") from exc
        message = _GAUGE_HMAC_DOMAIN + root_bytes
        return hmac.new(self.secret_key, message, hashlib.sha256).digest()

    def verify_snapshot_hmac(
        self,
        snapshot: TopologicalGaugeFieldSnapshot,
    ) -> bool:
        """Verifica cadena de fase, raíz Merkle y HMAC del snapshot."""
        try:
            self.continue_to_phase3(snapshot.phase2)
            if (
                snapshot.phase1.chain_hash
                != snapshot.phase2.phase1.chain_hash
            ):
                return False

            computed_root = _snapshot_merkle_root(snapshot)
            if not hmac.compare_digest(
                computed_root,
                snapshot.sha256_merkle_digest,
            ):
                return False

            expected = self._sign_snapshot(computed_root)
            return hmac.compare_digest(expected, snapshot.hmac_sha256)
        except (ValueError, TypeError, la.LinAlgError):
            return False

    def emit_governance_state(
        self,
        phase2_env: GaugePhase2Envelope,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> Tuple[TopologicalGaugeFieldSnapshot, Dict[str, Any]]:
        """Cierre de FASE 3: decide, simula y sella un snapshot."""
        self.continue_to_phase3(phase2_env)
        now = time.time()
        phase1 = phase2_env.phase1
        dimension = phase1.dimension

        if phi_initial is None:
            initial = np.ones(dimension, dtype=np.complex128) / math.sqrt(dimension)
        else:
            initial = _normalize_state(phi_initial, dimension, "phi_initial")

        if phi_final is None:
            # Postselección neutral por defecto; no se afirma que sea un
            # protocolo de medición física.
            final = initial.copy()
        else:
            final = _normalize_state(phi_final, dimension, "phi_final")

        hermitian_observable = -1j * phase1.connection_A
        hermitian_observable = 0.5 * (
            hermitian_observable + hermitian_observable.conj().T
        )
        weak = self.observe_qnd_weak_measurement(
            initial,
            final,
            hermitian_observable,
        )

        kam = phase2_env.kam_certificate
        melnikov = phase2_env.melnikov
        floquet = phase2_env.floquet
        rsi3 = phase2_env.rsi3_certificate
        lindstedt = phase1.poincare_lindstedt

        verdict = self.adjudicate_heyting(
            energy_ym=phase1.energy_ym,
            instanton_k=None,
            integrality_residual=None,
            symplectic_valid=phase1.is_symplectic_valid,
            kam_satisfied=kam.finite_test_passed,
            melnikov_preserved=melnikov.is_homoclinic_preserved,
            floquet_stable=(
                floquet.is_neutral or floquet.is_dissipative_stable
            ),
            rsi3_valid=rsi3.is_valid,
            bruno_satisfied=kam.bruno_truncated_passed,
            lindstedt_solved=(
                bool(lindstedt.is_solved)
                if lindstedt is not None
                else False
            ),
            kam_global_certified=kam.global_diophantine_certified,
            bruno_global_certified=kam.global_bruno_certified,
            melnikov_certified=melnikov.is_computable,
            floquet_global_certified=floquet.is_periodic_system_certified,
            pontryagin_certified=False,
            rsi3_full_space_contraction_certified=(
                rsi3.full_space_contraction_certified
            ),
        )

        interlock = self.trigger_hardware_interlock_esp32_crowbar(verdict)
        policy = self.apply_visceral_functor_phi_sem(
            verdict,
            phase1.energy_ym,
            phase2_env.wilson_angle_rad,
        )
        semantic_policy = tuple(
            sorted((str(key), str(value)) for key, value in policy.items())
        )

        snapshot = TopologicalGaugeFieldSnapshot(
            snapshot_id=f"GAUGE-SNAP-{uuid.uuid4().hex}",
            yang_mills_energy=float(phase1.energy_ym),
            chern_simons_3form_higgs=float(phase2_env.chern_simons_real),
            chern_simons_anomaly=float(phase2_env.chern_simons_anomaly),
            chern_hopf_class_c1=float(phase2_env.c1_value),
            wilson_loop_trace=complex(phase2_env.wilson_trace),
            wilson_angle_rad=float(phase2_env.wilson_angle_rad),
            pontryagin_instanton_number=None,
            pontryagin_integrality_residual=None,
            stora_zumino_residual=None,
            weak_value_qnd_aav=weak.weak_value,
            back_action_db=None,
            topology_regime=phase2_env.topology_regime,
            global_verdict=verdict,
            sha256_merkle_digest="",
            timestamp_utc=float(now),
            phase1=phase1,
            phase2=phase2_env,
            hmac_sha256=b"",
            kam_worst_ratio=float(kam.worst_ratio),
            kam_satisfied=bool(kam.global_diophantine_certified),
            kam_finite_test_passed=bool(kam.finite_test_passed),
            kam_global_certified=bool(kam.global_diophantine_certified),
            bruno_satisfied=bool(kam.global_bruno_certified),
            bruno_truncated_passed=kam.bruno_truncated_passed,
            bruno_global_certified=bool(kam.global_bruno_certified),
            melnikov_value=melnikov.integral_value,
            floquet_max_unit_deviation=float(
                floquet.max_unit_deviation
            ),
            floquet_is_neutral=bool(floquet.is_neutral),
            poincare_section_n_crossings=phase2_env.poincare_section.n_crossings,
            poincare_section_dispersion=(
                phase2_env.poincare_section.return_dispersion
            ),
            banach_lipschitz_t3=rsi3.lipschitz_t3,
            full_space_contraction_certified=(
                rsi3.full_space_contraction_certified
            ),
            fubini_study_distance=float(
                rsi3.surfaces[2].metric_value
            ),
            novikov_krull_valuation=None,
            rsi3_valid=bool(rsi3.is_valid),
            rsi3_iterations=int(rsi3.iterations),
            symplectic_residue=float(phase1.symplectic_residue),
            symplectic_defect=float(phase1.symplectic_defect),
            crowbar_latency_ns=float(interlock.python_elapsed_ns),
            crowbar_requested=bool(interlock.requested),
            crowbar_actuated=False,
            hardware_crowbar_action=interlock.action,
            postselection_overlap_db=weak.postselection_overlap_db,
            qnd_variance=float(weak.observable_variance),
            qnd_certified=False,
            semantic_policy=semantic_policy,
            engine_version=_ENGINE_VERSION,
        )

        merkle_root = _snapshot_merkle_root(snapshot)
        snapshot = replace(snapshot, sha256_merkle_digest=merkle_root)
        signature = self._sign_snapshot(merkle_root)
        snapshot = replace(snapshot, hmac_sha256=signature)

        return snapshot, policy


# ══════════════════════════════════════════════════════════════════════════════
# Fachada — orquestación de las tres fases
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldEngine(GaugePhase3Decider):
    """
    Fachada pública.

        FASE 1: emit_phase1_envelope → continue_to_phase2
        FASE 2: emit_phase2_envelope → continue_to_phase3
        FASE 3: emit_governance_state

    Para firmar snapshots exige una clave de al menos 32 bytes, inyectada o
    configurada mediante APU_GAUGE_HMAC_KEY.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: Optional[bytes] = None,
    ) -> None:
        super().__init__(
            dimension=dimension,
            secret_key=secret_key,
        )
        logger.info(
            "TopologicalGaugeFieldEngine v%s dim=%d",
            _ENGINE_VERSION,
            self.dimension,
        )

    def evaluate_topological_gauge_field(
        self,
        density_matrix: NDArray[Any],
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> Tuple[TopologicalGaugeFieldSnapshot, Dict[str, Any]]:
        """Orquesta Observe → Orient → Decide&Act."""
        phase1 = self.emit_phase1_envelope(
            density_matrix,
            background_hamiltonian=background_hamiltonian,
        )
        phase1 = self.continue_to_phase2(phase1)

        phi0 = (
            np.asarray(phi_initial, dtype=np.complex128).reshape(-1)
            if phi_initial is not None
            else None
        )
        phase2 = self.emit_phase2_envelope(phase1, phi0=phi0)
        phase2 = self.continue_to_phase3(phase2)

        return self.emit_governance_state(
            phase2,
            phi_initial=phi_initial,
            phi_final=phi_final,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Demostración reproducible — no constituye una suite de pruebas
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print("  DEMOSTRACIÓN — Topological Gauge Field Engine v12.0.0")
    print(banner)

    dimension = 16
    demo_key = hashlib.sha256(
        b"DEVELOPMENT-ONLY-DEMO-KEY-NOT-FOR-PRODUCTION"
    ).digest()
    engine = TopologicalGaugeFieldEngine(
        dimension=dimension,
        secret_key=demo_key,
    )

    rng = np.random.default_rng(2026)
    g = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    density_sample = g @ g.conj().T
    density_sample /= np.trace(density_sample)

    snapshot, policy = engine.evaluate_topological_gauge_field(density_sample)

    print(f"  Snapshot ID              : {snapshot.snapshot_id}")
    print(f"  Energía matricial        : {snapshot.yang_mills_energy:.6e}")
    print(f"  Proxy Chern–Simons       : {snapshot.chern_simons_3form_higgs:.6e}")
    print(f"  Proxy Pontryagin          : {snapshot.phase2.pontryagin_quadratic_proxy:.6e}")
    print(f"  Carga instantónica       : {snapshot.pontryagin_instanton_number}")
    print(f"  KAM finito pasó          : {snapshot.kam_finite_test_passed}")
    print(f"  KAM global certificado   : {snapshot.kam_global_certified}")
    print(f"  Melnikov computable      : {snapshot.phase2.melnikov.is_computable}")
    print(f"  Krein certificado        : {snapshot.phase2.floquet.krein_collision}")
    print(f"  Contracción en subespacio: {snapshot.banach_lipschitz_t3}")
    print(f"  Contracción global       : {snapshot.full_space_contraction_certified}")
    print(f"  Veredicto Heyting        : {snapshot.global_verdict.name}")
    print(f"  Crowbar solicitado       : {snapshot.crowbar_requested}")
    print(f"  Hardware actuado         : {snapshot.crowbar_actuated}")
    print(f"  WCET hardware certificado: {snapshot.phase2.scope == 'hardware_wcET_certified'}")
    print(f"  HMAC válido              : {engine.verify_snapshot_hmac(snapshot)}")
    print(f"  Política semántica       : {policy}")
    print(banner)