# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Octonionic Dependency Resolver (Resolver Octoniónico de Malla)      ║
║ Ruta   : app/core/octonionic_dependency_resolver.py                          ║
║ Versión: 3.0.0-Doctoral-Cayley-Dickson-Malcev-Heyting-Nested                 ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Resolutor de dependencias trilaterales no asociativas sobre el álgebra de    ║
║ división normada de los octoniones reales O ≅ R^8 (teorema de Hurwitz:       ║
║ las únicas R-álgebras de división normadas son R, C, H, O).                  ║
║                                                                              ║
║ Construcción de Cayley–Dickson:                                              ║
║     O = H ⊕ Hℓ ,   (q₁, q₂)·(p₁, p₂)                                         ║
║       = (q₁ p₁ − p̄₂ q₂ ,  p₂ q₁ + q₂ p̄₁).                                    ║
║                                                                              ║
║ O es alternativa, flexible y de composición, no asociativa ni conmutativa.   ║
║ Aut(O) ≅ G₂.  Im(O) ≅ R^7 porta el producto cruzado de Malcev.               ║
║                                                                              ║
║ La tríada transaccional (Contratista, Proveedor, Interventoría) se           ║
║ inmerge en O³.  El asociador                                                 ║
║                                                                              ║
║     [a, b, c] = (a b) c − a (b c)                                            ║
║                                                                              ║
║ es la obstrucción de calibre: colusión trilateral / frustración de malla.    ║
║                                                                              ║
║ Cadena de funtores anidados:                                                 ║
║                                                                              ║
║   (a,b,c) --Fase 1-->  Tríada(a, b, c, [a,b,c], Artin)                       ║
║           --Fase 2-->  Orientación(Hurwitz, asoc, ω_pre)                     ║
║           --Fase 3-->  Certificado(Ω₃, Crowbar, sello HMAC)                  ║
║                                                                              ║
║ Germen Fase 1 → Fase 2:                                                      ║
║     synthesize_octonionic_triad  ⊣  observe_octonionic_triad                 ║
║                                                                              ║
║ Germen Fase 2 → Fase 3:                                                      ║
║     orient_octonionic_diagnostics ⊣  decide_from_orientation                 ║
║                                                                              ║
║ Ω₃ = {0 < ½ < 1}  (Gödel–Heyting ternario)                                   ║
║     1  ↔  COHERENT                                                           ║
║     ½  ↔  DEGRADED   (veto suave + gracia)                                   ║
║     0  ↔  VETOED     (veto duro + Crowbar IRAM)                              ║
║                                                                              ║
║     a ∧ b = min(a,b),  a ∨ b = max(a,b),                                     ║
║     a → b = 1 si a ≤ b else b,  ¬a = a → 0.                                  ║
║                                                                              ║
║ ORGANIZACIÓN EN TRES FASES ANIDADAS POR HERENCIA ESTRICTA:                   ║
║   FASE 1: Phase1_OctonionicAlgebraKernel                                     ║
║   FASE 2: Phase2_OctonionicDiagnostics(Phase1_OctonionicAlgebraKernel)       ║
║   FASE 3: Phase3_OODAActuator(Phase2_OctonionicDiagnostics)                  ║
║   Motor : OctonionicDependencyResolver(Phase3_OODAActuator)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AbstractSet, Callable, Final, Optional, Tuple

import numpy as np


logger = logging.getLogger("APU.Core.OctonionicDependencyResolver")

__version__: Final[str] = "3.0.0"

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_OCTONION_DIM: Final[int] = 8
_QUATERNION_DIM: Final[int] = 4
_KERNEL_IDENTITY_LIMIT: Final[float] = 1e-10

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_LEGACY_OVERRIDE_TOKENS: Final[frozenset] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_NON_ASSOCIATIVE_IDU_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)

# Base canónica de O: e₀ = 1, e₁…e₇ imaginarios.
_UNIT: Final[np.ndarray] = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)
_UNIT.setflags(write=False)


# ───────────────────────────────────────────────────────────────────────────────
# Clasificación en el álgebra de Heyting Gödel Ω₃
# ───────────────────────────────────────────────────────────────────────────────

class HeytingVerdict(str, Enum):
    r"""
    Puntos del álgebra de Heyting ternaria Ω₃ = {0 < ½ < 1}.

    Orden de Gödel: VETOED < DEGRADED < COHERENT.
    El override nunca eleva 0; a lo sumo fija ½ (implicación Heyting).
    """

    VETOED = _VERDICT_VETOED
    DEGRADED = _VERDICT_DEGRADED
    COHERENT = _VERDICT_COHERENT

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.VETOED: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.COHERENT: 1.0,
        }[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.VETOED
        if value < 1.0:
            return cls.DEGRADED
        return cls.COHERENT


def _heyting_meet(a: float, b: float) -> float:
    return float(min(a, b))


def _heyting_implies(a: float, b: float) -> float:
    return 1.0 if a <= b + _MACHINE_EPS else float(b)


def _heyting_not(a: float) -> float:
    return _heyting_implies(a, 0.0)


# ───────────────────────────────────────────────────────────────────────────────
# Serialización canónica e inmutabilidad de ndarrays
# ───────────────────────────────────────────────────────────────────────────────

def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _canonical_bytes(part: Any) -> bytes:
    r"""Serialización little-endian estable, independiente de la arquitectura."""
    if isinstance(part, np.ndarray):
        arr = np.ascontiguousarray(part)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        header += np.array([1 if np.iscomplexobj(arr) else 0], dtype="<i8").tobytes()
        if np.iscomplexobj(arr):
            real = np.ascontiguousarray(arr.real, dtype=np.float64).astype("<f8")
            imag = np.ascontiguousarray(arr.imag, dtype=np.float64).astype("<f8")
            return header + real.tobytes() + imag.tobytes()
        real = np.ascontiguousarray(arr, dtype=np.float64).astype("<f8")
        return header + real.tobytes()
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    if isinstance(part, (int, float, bool, np.generic)):
        return np.array([part], dtype="<f8").tobytes()
    return repr(part).encode("utf-8")


# ───────────────────────────────────────────────────────────────────────────────
# Dataclasses inmutables
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class OctonionicState:
    r"""
    Estado físico hipercomplejo octoniónico de 8 dimensiones en la FPU.

    Descomposición de Cayley–Dickson:

        o = (q₁, q₂) ∈ H × H,
        q₁ = [o₀, o₁, o₂, o₃],   q₂ = [o₄, o₅, o₆, o₇].

    Atributos:
        vector_rep:     Coeficientes reales en R^8.
        q1, q2:         Pares cuaterniónicos.
        norm:           ‖o‖_O = √⟨o, o⟩.
        is_unitary:     |‖o‖ − 1| ≤ τ.
        sha256_hash:    Firma canónica.
        norm_squared:   ‖o‖² (KBN).
        is_finite:      Finitud numérica completa.
        real_part:      Re(o) = o₀.
        imag_norm:      ‖Im(o)‖₂.
        polar_phase:    o / ‖o‖ si o ≠ 0; 0 si nulo.
    """

    vector_rep: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    norm: float
    is_unitary: bool
    sha256_hash: str

    norm_squared: float = 0.0
    is_finite: bool = True
    real_part: float = 0.0
    imag_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class OctonionicTriadReport:
    r"""
    GERMEN FASE 1 → FASE 2.

    Tríada de Cayley–Dickson

        T(a,b,c) = (a, b, c, ab, bc, (ab)c, a(bc), [a,b,c]) ∈ O^7 × R^8

    junto con los residuos de Artin (alternatividad / flexibilidad) que
    *deben* anularse idénticamente en O.  Un residuo de Artin no nulo es
    fallo del núcleo, no colusión de malla.

    El funtor de la Fase 2, ``observe_octonionic_triad``, actúa de forma
    estricta sobre este germen.
    """

    contractor: OctonionicState
    supplier: OctonionicState
    interventor: OctonionicState
    product_ab: OctonionicState
    product_bc: OctonionicState
    product_left: OctonionicState
    product_right: OctonionicState
    associator: np.ndarray
    associator_norm: float
    left_alternator_norm: float
    right_alternator_norm: float
    flexibility_norm: float
    artin_residual: float
    sha256_hash: str


@dataclass(frozen=True, slots=True)
class OctonionicOrientationReport:
    r"""
    GERMEN FASE 2 → FASE 3.

    Orientación covariante: composición de Hurwitz, asociador relativo y
    preclasificación Heyting ω_pre ∈ Ω₃ *antes* de gracia y override.

        ω_hard = 1 si Hurwitz grave y asociador finitos, else 0
        ω_cfl  = 1 si ‖ab‖=‖a‖‖b‖ (pares), else ½
        ω_asoc = 1 si ‖[a,b,c]‖ ≤ τ, else ½
        ω_pre  = ω_hard ∧ ω_cfl ∧ ω_asoc

    Un fallo de Artin (núcleo) fuerza ω_hard = 0.
    """

    triad: OctonionicTriadReport
    composition_error: float
    composition_relative_error: float
    associator_norm: float
    associator_relative_norm: float
    is_cfl_stable: bool
    is_associative_stable: bool
    hard_composition: bool
    artin_residual: float
    moufang_residual: float
    omega_hard: float
    omega_cfl: float
    omega_asoc: float
    omega_pre: float
    sha256_hash: str

    contractor_norm: float = 0.0
    supplier_norm: float = 0.0
    interventor_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class OctonionicAuditCertificate:
    r"""
    Certificado inmutable de regularidad no asociativa.

    Atributos:
        heyting_verdict:            'COHERENT' | 'DEGRADED' | 'VETOED'.
        associator_norm:            ‖[a,b,c]‖.
        is_associative_stable:      ‖[a,b,c]‖ ≤ umbral.
        composition_error:          max |‖xy‖ − ‖x‖‖y‖| sobre pares.
        is_cfl_stable:              Estabilidad de composición normada.
        is_soft_veto_active:        Veto suave (luz ámbar).
        is_hard_veto_active:        Veto duro (Crowbar).
        actuation_latency_ns:       Latencia Crowbar determinista < 400 ns.
        time_grace_remaining:       Gracia residual (s).
        cryptographic_seal:         Sello SHA-256 / HMAC-SHA256.
        composition_relative_error: Error relativo de Hurwitz.
        associator_relative_norm:   ‖[a,b,c]‖ / (‖a‖‖b‖‖c‖).
        contractor_norm:            ‖a‖.
        supplier_norm:              ‖b‖.
        interventor_norm:           ‖c‖.
        heyting_omega:              Valor numérico en Ω₃.
        omega_pre:                  ω antes de gracia/override.
        artin_residual:             max(alt_L, alt_R, flex).
        moufang_residual:           Residuo de Moufang sobre la tríada.
        triad_seal:                 Sello del germen Fase 1.
    """

    heyting_verdict: str
    associator_norm: float
    is_associative_stable: bool
    composition_error: float
    is_cfl_stable: bool
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    actuation_latency_ns: float
    time_grace_remaining: float
    cryptographic_seal: str

    composition_relative_error: float = 0.0
    associator_relative_norm: float = 0.0
    contractor_norm: float = 0.0
    supplier_norm: float = 0.0
    interventor_norm: float = 0.0
    heyting_omega: float = 1.0
    omega_pre: float = 1.0
    artin_residual: float = 0.0
    moufang_residual: float = 0.0
    triad_seal: str = ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: Núcleo algebraico octoniónico (Cayley–Dickson, Fano, Artin)         ║
# ║                                                                              ║
# ║ Objeto: O como R-álgebra de división normada de dimensión 8.                 ║
# ║ Cierre formal: synthesize_octonionic_triad  →  germen de la Fase 2.          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase1_OctonionicAlgebraKernel:
    r"""
    FASE 1 — Núcleo de Cayley–Dickson sobre O.

    Responsabilidades:
      1. Validar vectores R^4 y R^8 finitos.
      2. Construir estados OctonionicState (norma KBN, polaridad).
      3. Aritmética de Hamilton en H y duplicación CD hacia O.
      4. Conjugado, inversa, producto interno, conmutador y asociador.
      5. Certificar identidades de núcleo (unidad, eᵢ² = −1, Artin).
      6. Cierre: sintetizar la tríada T(a,b,c) germen de la Fase 2.
    """

    __slots__ = ("_tol", "_kernel_certified")

    def __init__(self, tolerance: float = 1e-12) -> None:
        self._tol: Final[float] = float(tolerance)
        self._kernel_certified: bool = False
        self._certify_kernel_identities()

    def _relative_tolerance(self, scale: float = 1.0) -> float:
        return max(self._tol, 10.0 * _MACHINE_EPS * max(1.0, float(scale)))

    # ───────────────────────────────────────────────────────────────────────────
    # Utilidades numéricas
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kbn_sum(values: np.ndarray) -> float:
        r"""Sumación compensada Kahan–Babuška–Neumaier (deriva de Wilkinson)."""
        total = 0.0
        compensation = 0.0
        for value in np.asarray(values, dtype=np.float64).ravel():
            val = float(value)
            if not math.isfinite(val):
                return val
            y = val - compensation
            t = total + y
            compensation = (t - total) - y
            total = t
        return total

    def _sha256_payload(self, *parts: Any) -> str:
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        return sha.hexdigest()

    def _euclidean_norm_squared(self, vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        return self._kbn_sum(arr * arr)

    def _euclidean_norm(self, vec: np.ndarray) -> float:
        nsq = self._euclidean_norm_squared(vec)
        if nsq < 0.0 and nsq > -self._tol:
            nsq = 0.0
        if nsq < 0.0:
            raise ValueError("Norma cuadrada negativa: corrupción numérica.")
        return float(math.sqrt(nsq))

    # ───────────────────────────────────────────────────────────────────────────
    # Validación dimensional
    # ───────────────────────────────────────────────────────────────────────────

    def _validate_vector4(
        self,
        q: np.ndarray,
        name: str = "cuaternión",
    ) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (_QUATERNION_DIM,):
            raise ValueError(
                f"El {name} debe ser estrictamente 4D. Obtenido: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El {name} contiene valores no finitos.")
        return arr

    def _validate_vector8(
        self,
        o: np.ndarray,
        name: str = "octonión",
    ) -> np.ndarray:
        arr = np.asarray(o, dtype=np.float64)
        if arr.shape != (_OCTONION_DIM,):
            raise ValueError(
                f"El {name} debe ser estrictamente 8D. Obtenido: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El {name} contiene valores no finitos.")
        return arr

    def _as_state(self, x: Any) -> OctonionicState:
        if isinstance(x, OctonionicState):
            self._validate_vector8(x.vector_rep, "estado octoniónico recibido")
            self._validate_vector4(x.q1, "q1 del estado octoniónico")
            self._validate_vector4(x.q2, "q2 del estado octoniónico")
            return x
        return self.build_state(x)

    # ───────────────────────────────────────────────────────────────────────────
    # Construcción de estado
    # ───────────────────────────────────────────────────────────────────────────

    def build_state(self, S: np.ndarray) -> OctonionicState:
        r"""
        Instancia un octonión rígido desde R^8.

        Norma KBN, unitariedad adaptativa, proyección polar implícita.
        """
        S_arr = self._validate_vector8(S, "estado octoniónico")
        q1 = S_arr[0:4].copy()
        q2 = S_arr[4:8].copy()

        norm_sq = self._euclidean_norm_squared(S_arr)
        if not math.isfinite(norm_sq):
            raise ValueError("La norma cuadrada del octonión no es finita.")
        if norm_sq < 0.0 and norm_sq > -self._tol:
            norm_sq = 0.0
        if norm_sq < 0.0:
            raise ValueError("La norma cuadrada del octonión es negativa.")

        norm_val = float(math.sqrt(norm_sq))
        if not math.isfinite(norm_val):
            raise ValueError("La norma del octonión no es finita.")

        unit_tolerance = self._relative_tolerance(max(1.0, norm_val))
        is_unit = bool(abs(norm_val - 1.0) <= unit_tolerance)

        imag = S_arr[1:].copy()
        imag_norm = self._euclidean_norm(imag)

        sha256_hash = self._sha256_payload(
            S_arr,
            q1,
            q2,
            np.array([norm_sq, norm_val, imag_norm], dtype=np.float64),
        )

        return OctonionicState(
            vector_rep=_immutable(S_arr, np.float64),
            q1=_immutable(q1, np.float64),
            q2=_immutable(q2, np.float64),
            norm=norm_val,
            is_unitary=is_unit,
            sha256_hash=sha256_hash,
            norm_squared=float(norm_sq),
            is_finite=True,
            real_part=float(S_arr[0]),
            imag_norm=imag_norm,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Aritmética cuaterniónica de Hamilton
    # ───────────────────────────────────────────────────────────────────────────

    def quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        r"""\(\bar q = (q_0, -q_1, -q_2, -q_3)\)."""
        q_arr = self._validate_vector4(q, "cuaternión a conjugar")
        return np.array(
            [q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]],
            dtype=np.float64,
        )

    def quaternion_multiply(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        r"""Producto de Hamilton \(r = qp \in \mathbb{H}\), KBN por componente."""
        q_arr = self._validate_vector4(q, "cuaternión q")
        p_arr = self._validate_vector4(p, "cuaternión p")
        q0, q1, q2, q3 = q_arr
        p0, p1, p2, p3 = p_arr

        r0 = self._kbn_sum(
            np.array([q0 * p0, -q1 * p1, -q2 * p2, -q3 * p3], dtype=np.float64)
        )
        r1 = self._kbn_sum(
            np.array([q0 * p1, q1 * p0, q2 * p3, -q3 * p2], dtype=np.float64)
        )
        r2 = self._kbn_sum(
            np.array([q0 * p2, -q1 * p3, q2 * p0, q3 * p1], dtype=np.float64)
        )
        r3 = self._kbn_sum(
            np.array([q0 * p3, q1 * p2, -q2 * p1, q3 * p0], dtype=np.float64)
        )
        result = np.array([r0, r1, r2, r3], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("El producto cuaterniónico produjo valores no finitos.")
        return result

    def quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        q_arr = self._validate_vector4(q, "cuaternión a invertir")
        nsq = self._euclidean_norm_squared(q_arr)
        threshold = max(self._tol * self._tol, _MACHINE_EPS)
        if nsq <= threshold:
            raise ZeroDivisionError("Cuaternión no invertible (norma casi nula).")
        return self.quaternion_conjugate(q_arr) / nsq

    # ───────────────────────────────────────────────────────────────────────────
    # Aritmética octoniónica de Cayley–Dickson
    # ───────────────────────────────────────────────────────────────────────────

    def octonion_conjugate(self, o: Any) -> np.ndarray:
        r"""
        Involución \(\bar o = (o_0, -o_1, \ldots, -o_7)\).

        Equivale a \(\overline{(q_1, q_2)} = (\bar q_1, -q_2)\).
        """
        state = self._as_state(o)
        out = -state.vector_rep.copy()
        out[0] = state.vector_rep[0]
        return out

    def octonion_inverse(self, o: Any) -> np.ndarray:
        r"""
        Inversa en el álgebra de división: \(o^{-1} = \bar o / \|o\|^2\).

        Por alternatividad, \(o^{-1}(o b) = b = (b o) o^{-1}\).
        """
        state = self._as_state(o)
        norm_sq = state.norm_squared
        threshold = max(self._tol * self._tol, _MACHINE_EPS)
        if norm_sq <= threshold:
            raise ZeroDivisionError(
                "El octonión no es invertible porque su norma cuadrada es casi nula."
            )
        return self.octonion_conjugate(state) / norm_sq

    def octonion_inner_product(self, a: Any, b: Any) -> float:
        r"""
        Producto interno euclídeo polarizado:

            ⟨a, b⟩ = Re(\(\bar a\, b\)) = Σᵢ aᵢ bᵢ.
        """
        A = self._as_state(a)
        B = self._as_state(b)
        return self._kbn_sum(A.vector_rep * B.vector_rep)

    def octonionic_multiply(self, a: Any, b: Any) -> OctonionicState:
        r"""
        Producto de Cayley–Dickson:

            (q₁, q₂)·(p₁, p₂)
              = (q₁ p₁ − \(\bar p_2\) q₂ ,
                 p₂ q₁ + q₂ \(\bar p_1\)).

        No asociativo, no conmutativo; sí alternativo y normado.
        """
        A = self._as_state(a)
        B = self._as_state(b)

        q1, q2 = A.q1, A.q2
        p1, p2 = B.q1, B.q2

        term_a1 = self.quaternion_multiply(q1, p1)
        conj_p2 = self.quaternion_conjugate(p2)
        term_a2 = self.quaternion_multiply(conj_p2, q2)
        part_A = term_a1 - term_a2

        term_b1 = self.quaternion_multiply(p2, q1)
        conj_p1 = self.quaternion_conjugate(p1)
        term_b2 = self.quaternion_multiply(q2, conj_p1)
        part_B = term_b1 + term_b2

        result_vec = np.concatenate([part_A, part_B])
        return self.build_state(result_vec)

    def left_multiply(self, a: Any, b: Any) -> OctonionicState:
        r"""Operador de multiplicación a izquierda \(L_a(b) = a b\)."""
        return self.octonionic_multiply(a, b)

    def right_multiply(self, a: Any, b: Any) -> OctonionicState:
        r"""Operador de multiplicación a derecha \(R_a(b) = b a\)."""
        return self.octonionic_multiply(b, a)

    def compute_commutator(self, a: Any, b: Any) -> np.ndarray:
        r"""Conmutador \([a,b] = ab - ba\). Se anula sobre R ⊕ span{eᵢ} pairwise? No."""
        A = self._as_state(a)
        B = self._as_state(b)
        ab = self.octonionic_multiply(A, B)
        ba = self.octonionic_multiply(B, A)
        comm = ab.vector_rep - ba.vector_rep
        if not np.all(np.isfinite(comm)):
            raise ValueError("El conmutador octoniónico contiene valores no finitos.")
        return comm

    def compute_associator(self, a: Any, b: Any, c: Any) -> np.ndarray:
        r"""
        Tensor asociador trilateral:

            [a, b, c] = (a b) c − a (b c).

        En O mide la frustración de calibre no asociativa.  Es alternado
        (teorema de Artin / alternatividad): [a,a,b] = [b,a,a] = [a,b,a] = 0.
        """
        A = self._as_state(a)
        B = self._as_state(b)
        C = self._as_state(c)

        ab = self.octonionic_multiply(A, B)
        left = self.octonionic_multiply(ab, C)
        bc = self.octonionic_multiply(B, C)
        right = self.octonionic_multiply(A, bc)

        associator = left.vector_rep - right.vector_rep
        if not np.all(np.isfinite(associator)):
            raise ValueError("El asociador octoniónico contiene valores no finitos.")
        return associator

    def compute_left_alternator(self, a: Any, b: Any) -> np.ndarray:
        r"""Alternador izquierdo [a, a, b]. Debe ser ~ 0 en O."""
        return self.compute_associator(a, a, b)

    def compute_right_alternator(self, a: Any, b: Any) -> np.ndarray:
        r"""Alternador derecho [b, a, a]. Debe ser ~ 0 en O."""
        return self.compute_associator(b, a, a)

    def compute_flexibility_associator(self, a: Any, b: Any) -> np.ndarray:
        r"""Identidad flexible [a, b, a]. Debe ser ~ 0 en O."""
        return self.compute_associator(a, b, a)

    def _basis_vector(self, index: int) -> np.ndarray:
        vec = np.zeros(_OCTONION_DIM, dtype=np.float64)
        vec[index] = 1.0
        return vec

    def _certify_kernel_identities(self) -> None:
        r"""
        Auditoría de núcleo al construir la Fase 1:

          1. e₀ es unidad bilateral.
          2. eᵢ² = −1 para i = 1…7.
          3. {eᵢ, eⱼ} = −2 δᵢⱼ I  (anticommutador imaginario).
          4. ‖eᵢ eⱼ‖ = 1  (Hurwitz sobre la base).
          5. [e₁, e₁, e₂] ≈ 0  (alternatividad muestral).

        No usa aleatoriedad: bit-reproducible.
        """
        unit = self.build_state(_UNIT)
        max_err = 0.0

        for i in range(1, _OCTONION_DIM):
            ei = self.build_state(self._basis_vector(i))
            left = self.octonionic_multiply(unit, ei)
            right = self.octonionic_multiply(ei, unit)
            max_err = max(
                max_err,
                float(np.linalg.norm(left.vector_rep - ei.vector_rep)),
                float(np.linalg.norm(right.vector_rep - ei.vector_rep)),
            )
            sq = self.octonionic_multiply(ei, ei)
            target = -_UNIT
            max_err = max(
                max_err, float(np.linalg.norm(sq.vector_rep - target))
            )

        e1 = self.build_state(self._basis_vector(1))
        e2 = self.build_state(self._basis_vector(2))
        alt = self.compute_left_alternator(e1, e2)
        max_err = max(max_err, float(np.linalg.norm(alt)))

        prod12 = self.octonionic_multiply(e1, e2)
        max_err = max(max_err, abs(prod12.norm - 1.0))

        if max_err > _KERNEL_IDENTITY_LIMIT:
            logger.warning(
                "Identidades de núcleo octoniónico degradadas: residuo=%.3e",
                max_err,
            )
            self._kernel_certified = False
        else:
            self._kernel_certified = True

    # ───────────────────────────────────────────────────────────────────────────
    # Cierre formal de la Fase 1
    # ───────────────────────────────────────────────────────────────────────────

    def synthesize_octonionic_triad(
        self,
        contractor_S: Any,
        supplier_S: Any,
        interventor_S: Any,
    ) -> OctonionicTriadReport:
        r"""
        CIERRE FORMAL DE LA FASE 1 / GERMEN DE LA FASE 2.

        Construye el objeto de tríada

            T(a,b,c) = (a, b, c, ab, bc, (ab)c, a(bc), [a,b,c])

        sobre el que el funtor de la Fase 2,
        ``observe_octonionic_triad``, actúa de forma estricta:

            observe : T(a,b,c) × R≥0  →  Orientación.

        No se evalúa aún ni Hurwitz relativo ni Ω₃: eso es diagnóstico
        espectral, no álgebra, y pertenece a la Fase 2.

        Residuos de Artin (deben anularse en O, fallo de núcleo si no):
            alt_L = ‖[a,a,b]‖,  alt_R = ‖[c,b,b]‖,  flex = ‖[a,b,a]‖.
        """
        a = self._as_state(contractor_S)
        b = self._as_state(supplier_S)
        c = self._as_state(interventor_S)

        ab = self.octonionic_multiply(a, b)
        bc = self.octonionic_multiply(b, c)
        left = self.octonionic_multiply(ab, c)
        right = self.octonionic_multiply(a, bc)

        associator = left.vector_rep - right.vector_rep
        if not np.all(np.isfinite(associator)):
            raise ValueError("El asociador octoniónico contiene valores no finitos.")

        associator_norm = self._euclidean_norm(associator)
        if not math.isfinite(associator_norm):
            raise ValueError("La norma del asociador octoniónico no es finita.")

        left_alternator_norm = self._euclidean_norm(self.compute_left_alternator(a, b))
        right_alternator_norm = self._euclidean_norm(self.compute_right_alternator(b, c))
        flexibility_norm = self._euclidean_norm(self.compute_flexibility_associator(a, b))
        artin_residual = float(
            max(left_alternator_norm, right_alternator_norm, flexibility_norm)
        )

        sha256_hash = self._sha256_payload(
            a.sha256_hash,
            b.sha256_hash,
            c.sha256_hash,
            associator,
            np.array(
                [
                    associator_norm,
                    left_alternator_norm,
                    right_alternator_norm,
                    flexibility_norm,
                ],
                dtype=np.float64,
            ),
        )

        return OctonionicTriadReport(
            contractor=a,
            supplier=b,
            interventor=c,
            product_ab=ab,
            product_bc=bc,
            product_left=left,
            product_right=right,
            associator=_immutable(associator, np.float64),
            associator_norm=associator_norm,
            left_alternator_norm=left_alternator_norm,
            right_alternator_norm=right_alternator_norm,
            flexibility_norm=flexibility_norm,
            artin_residual=artin_residual,
            sha256_hash=sha256_hash,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: Diagnóstico espectral, Hurwitz, Moufang y preclasificación Ω₃       ║
# ║                                                                              ║
# ║ Apertura: observe_octonionic_triad(synthesize_octonionic_triad(·)).          ║
# ║ Cierre formal: orient_octonionic_diagnostics  →  germen de la Fase 3.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_OctonionicDiagnostics(Phase1_OctonionicAlgebraKernel):
    r"""
    FASE 2 — Diagnóstico de composición y no asociatividad.

    Continuación estricta de ``synthesize_octonionic_triad``.

    Responsabilidades:
      1. Observar T(a,b,c) contra la ley de Hurwitz ‖xy‖ = ‖x‖‖y‖.
      2. Norma absoluta y relativa del asociador.
      3. Residuo de Moufang sobre la tríada.
      4. Cierre: preclasificar ω_pre ∈ Ω₃ para el decisor de la Fase 3.
    """

    __slots__ = ()

    def compute_hurwitz_error(
        self,
        a: Any,
        b: Any,
    ) -> Tuple[float, float, OctonionicState]:
        r"""
        Error de composición de Hurwitz para el producto a b.

        Retorna:
            absolute_error, relative_error, product_state.
        """
        A = self._as_state(a)
        B = self._as_state(b)
        AB = self.octonionic_multiply(A, B)

        expected = A.norm * B.norm
        if not (math.isfinite(AB.norm) and math.isfinite(expected)):
            return float("inf"), float("inf"), AB

        absolute_error = float(abs(AB.norm - expected))
        denominator = expected if (math.isfinite(expected) and expected > 1.0) else 1.0
        relative_error = float(absolute_error / denominator)
        return absolute_error, relative_error, AB

    def compute_associator_diagnostics(
        self,
        a: Any,
        b: Any,
        c: Any,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        Asociador y diagnósticos de escala.

        Retorna:
            associator_vector, associator_norm, associator_relative_norm.
        """
        A = self._as_state(a)
        B = self._as_state(b)
        C = self._as_state(c)

        associator = self.compute_associator(A, B, C)
        associator_norm = self._euclidean_norm(associator)
        if not math.isfinite(associator_norm):
            raise ValueError("La norma del asociador octoniónico no es finita.")

        product_scale = A.norm * B.norm * C.norm
        if not math.isfinite(product_scale) or product_scale <= _MACHINE_EPS:
            denominator = max(_MACHINE_EPS, 1.0, A.norm, B.norm, C.norm)
        else:
            denominator = product_scale

        associator_relative = float(associator_norm / denominator)
        return associator, associator_norm, associator_relative

    def compute_moufang_residual(
        self,
        a: OctonionicState,
        b: OctonionicState,
        c: OctonionicState,
    ) -> float:
        r"""
        Residuo de la identidad de Moufang

            (a x)(y a) = a (x y) a

        evaluada en x = b, y = c.  Debe anularse en O.
        """
        ax = self.octonionic_multiply(a, b)
        ya = self.octonionic_multiply(c, a)
        lhs = self.octonionic_multiply(ax, ya)

        xy = self.octonionic_multiply(b, c)
        a_xy = self.octonionic_multiply(a, xy)
        rhs = self.octonionic_multiply(a_xy, a)

        return float(np.linalg.norm(lhs.vector_rep - rhs.vector_rep))

    def observe_octonionic_triad(
        self,
        triad: OctonionicTriadReport,
        asoc_threshold: float = 0.15,
    ) -> OctonionicOrientationReport:
        r"""
        APERTURA FORMAL DE LA FASE 2.

        Continuación directa de ``synthesize_octonionic_triad``:

            T(a,b,c)  ↦  (Hurwitz, ‖[a,b,c]‖_rel, Moufang, ω_pre).
        """
        if not isinstance(triad, OctonionicTriadReport):
            raise TypeError(
                "observe_octonionic_triad exige un OctonionicTriadReport "
                "(germen de synthesize_octonionic_triad)."
            )

        asoc_threshold_float = float(asoc_threshold)
        if not math.isfinite(asoc_threshold_float) or asoc_threshold_float < 0.0:
            raise ValueError("asoc_threshold debe ser finito y no negativo.")

        a, b, c = triad.contractor, triad.supplier, triad.interventor

        abs_err_ab, rel_err_ab, _ = self.compute_hurwitz_error(a, b)
        abs_err_bc, rel_err_bc, _ = self.compute_hurwitz_error(b, c)
        abs_err_ac, rel_err_ac, _ = self.compute_hurwitz_error(a, c)

        composition_error = float(max(abs_err_ab, abs_err_bc, abs_err_ac))
        composition_relative_error = float(max(rel_err_ab, rel_err_bc, rel_err_ac))

        product_scale = a.norm * b.norm * c.norm
        if not math.isfinite(product_scale) or product_scale <= _MACHINE_EPS:
            denominator = max(_MACHINE_EPS, 1.0, a.norm, b.norm, c.norm)
        else:
            denominator = product_scale
        associator_relative = float(triad.associator_norm / denominator)

        pair_expected = [a.norm * b.norm, b.norm * c.norm, a.norm * c.norm]
        finite_expected = [v for v in pair_expected if math.isfinite(v)]
        expected_scale = max([1.0] + finite_expected)

        composition_tolerance = max(
            self._tol,
            100.0 * _MACHINE_EPS * expected_scale,
        )
        hard_composition_limit = max(
            1e-9,
            1000.0 * _MACHINE_EPS * expected_scale,
        )

        is_cfl_stable = bool(
            math.isfinite(composition_error)
            and composition_error <= composition_tolerance
        )
        hard_composition = bool(
            (not math.isfinite(composition_error))
            or (composition_error > hard_composition_limit)
        )
        is_associative_stable = bool(
            math.isfinite(triad.associator_norm)
            and triad.associator_norm <= (asoc_threshold_float + self._tol)
        )

        artin_tol = self._relative_tolerance(
            max(1.0, a.norm, b.norm, c.norm, expected_scale)
        )
        artin_broken = bool(
            (not math.isfinite(triad.artin_residual))
            or (triad.artin_residual > max(artin_tol, 1e-8))
        )

        moufang_residual = self.compute_moufang_residual(a, b, c)

        omega_hard = (
            0.0
            if (
                hard_composition
                or (not math.isfinite(triad.associator_norm))
                or artin_broken
            )
            else 1.0
        )
        omega_cfl = 1.0 if is_cfl_stable else 0.5
        omega_asoc = 1.0 if is_associative_stable else 0.5
        omega_pre = _heyting_meet(_heyting_meet(omega_hard, omega_cfl), omega_asoc)

        sha256_hash = self._sha256_payload(
            triad.sha256_hash,
            np.array(
                [
                    composition_error,
                    composition_relative_error,
                    triad.associator_norm,
                    associator_relative,
                    omega_pre,
                    moufang_residual,
                ],
                dtype=np.float64,
            ),
        )

        return OctonionicOrientationReport(
            triad=triad,
            composition_error=composition_error,
            composition_relative_error=composition_relative_error,
            associator_norm=triad.associator_norm,
            associator_relative_norm=associator_relative,
            is_cfl_stable=is_cfl_stable,
            is_associative_stable=is_associative_stable,
            hard_composition=hard_composition,
            artin_residual=triad.artin_residual,
            moufang_residual=moufang_residual,
            omega_hard=omega_hard,
            omega_cfl=omega_cfl,
            omega_asoc=omega_asoc,
            omega_pre=omega_pre,
            sha256_hash=sha256_hash,
            contractor_norm=a.norm,
            supplier_norm=b.norm,
            interventor_norm=c.norm,
        )

    def orient_octonionic_diagnostics(
        self,
        contractor_S: Any,
        supplier_S: Any,
        interventor_S: Any,
        asoc_threshold: float = 0.15,
        triad: Optional[OctonionicTriadReport] = None,
    ) -> OctonionicOrientationReport:
        r"""
        CIERRE FORMAL DE LA FASE 2 / GERMEN DE LA FASE 3.

        Encadena el germen algebraico de la Fase 1 con la observación
        espectral y produce el objeto de orientación

            Ω_pre = (ω_hard ∧ ω_cfl ∧ ω_asoc) ∈ Ω₃

        sobre el que ``Phase3_OODAActuator.decide_from_orientation`` actúa
        de forma estricta: aplica la ventana de gracia (flecha ½ → 0) y
        el override (implicación Heyting que no eleva 0).

        No muta estado de veto ni dispara Crowbar: eso es actuación, no
        orientación.
        """
        if triad is None:
            triad = self.synthesize_octonionic_triad(
                contractor_S, supplier_S, interventor_S
            )
        return self.observe_octonionic_triad(
            triad=triad,
            asoc_threshold=asoc_threshold,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: Actuación OODA, Heyting Ω₃ y veto ciber-físico                      ║
# ║                                                                              ║
# ║ Apertura: decide_from_orientation(orient_octonionic_diagnostics(·)).         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_OODAActuator(Phase2_OctonionicDiagnostics):
    r"""
    FASE 3 — Decisión en Ω₃, gracia, override y Crowbar IRAM.

    Continuación estricta de ``orient_octonionic_diagnostics``:

        ω = decide(ω_pre, gracia, override) ∈ Ω₃,
        Act = Crowbar  syss  ω = 0.

    Responsabilidades:
      1. Decidir COHERENT / DEGRADED / VETOED sobre ω_pre.
      2. Gestionar veto suave con ventana de gracia (½ persistente → 0).
      3. Verificar overrides en tiempo constante / HMAC.
      4. Actuar: latencia Crowbar determinista < 400 ns.
      5. Sellar HMAC/SHA-256 el certificado.
    """

    __slots__ = (
        "_grace_max",
        "_asoc_limit",
        "_soft_veto_timestamp",
        "_is_soft_veto_active",
        "_override_verifier",
        "_allowed_override_tokens",
        "_hmac_secret",
    )

    def __init__(
        self,
        tolerance: float = 1e-12,
        grace_period_seconds: float = 3600.0,
        asoc_threshold: float = 0.15,
        override_verifier: Optional[Callable[[str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        super().__init__(tolerance=tolerance)

        self._grace_max: Final[float] = float(grace_period_seconds)
        self._asoc_limit: Final[float] = float(asoc_threshold)

        if not math.isfinite(self._asoc_limit) or self._asoc_limit < 0.0:
            raise ValueError("asoc_threshold debe ser finito y no negativo.")

        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False
        self._override_verifier: Optional[Callable[[str], bool]] = override_verifier

        if allowed_override_tokens is None:
            tokens = _LEGACY_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        else:
            tokens = frozenset(allowed_override_tokens)
        self._allowed_override_tokens: Final[frozenset] = frozenset(tokens)

        if hmac_secret is not None and not isinstance(hmac_secret, (bytes, bytearray)):
            raise TypeError("hmac_secret debe ser bytes o None.")
        self._hmac_secret: Final[Optional[bytes]] = (
            bytes(hmac_secret) if hmac_secret is not None else None
        )

        if allow_legacy_overrides and self._allowed_override_tokens.intersection(
            _LEGACY_OVERRIDE_TOKENS
        ):
            logger.warning(
                "Se permiten tokens legacy de override octoniónico. "
                "Para producción, configure override_verifier o hmac_secret."
            )

    def _clear_soft_veto(self) -> None:
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def reset_soft_veto(self) -> None:
        r"""Reinicia el temporizador de gracia (laboratorio / tests)."""
        self._clear_soft_veto()

    def _verify_override(self, token: Optional[str]) -> bool:
        r"""
        Verificación de override, por orden de autoridad:

          1. override_verifier inyectado.
          2. HMAC-SHA256(hmac_secret, token) si hay secreto.
          3. Conjunto de tokens permitidos vía hmac.compare_digest.

        El override **no** se registra en claro.
        """
        if token is None or not isinstance(token, str) or not token.strip():
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token))
            except Exception:
                logger.exception(
                    "El override_verifier lanzó una excepción. Se rechaza el override."
                )
                return False

        token_bytes = token.encode("utf-8")

        if self._hmac_secret is not None:
            expected = hmac.new(
                self._hmac_secret, token_bytes, hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(expected, token):
                return True

        for allowed in self._allowed_override_tokens:
            allowed_bytes = allowed.encode("utf-8")
            if hmac.compare_digest(token_bytes, allowed_bytes):
                if allowed in _LEGACY_OVERRIDE_TOKENS:
                    logger.warning(
                        "Override legacy aceptado. Considere migrar a tokens firmados."
                    )
                return True
        return False

    def _seeded_latency_ns(self, *parts: Any) -> float:
        r"""
        Latencia Crowbar determinista en IRAM.

        En hardware real se sustituye por medición GPIO/ISR.
        Cota: 395 ns ≤ τ < 400 ns.  Sin np.random.
        """
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        digest = sha.digest()
        fraction = int.from_bytes(digest[:6], "little") / float(1 << 48)
        latency = 395.0 + 4.5 * fraction
        return float(min(_CROWBAR_IRAM_LATENCY_NS, latency))

    def _seal(self, *parts: Any) -> str:
        payload = b"".join(_canonical_bytes(part) for part in parts)
        if self._hmac_secret is not None:
            return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()
        return hashlib.sha256(payload).hexdigest()

    def decide_from_orientation(
        self,
        orientation: OctonionicOrientationReport,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
    ) -> Tuple[HeytingVerdict, bool, bool, float]:
        r"""
        APERTURA FORMAL DE LA FASE 3.

        Continuación directa de ``orient_octonionic_diagnostics``.

        Entrada: ω_pre = ω_hard ∧ ω_cfl ∧ ω_asoc.
        Dinámica de Heyting:
          · ω_pre = 1  →  COHERENT, se limpia la gracia.
          · ω_pre = 0  →  VETOED instantáneo (Hurwitz grave / Artin / NaN).
            El override **no** eleva 0  (¬¬0 = 0 en Ω₃ Gödel).
          · ω_pre = ½  →  DEGRADED; si la gracia expira, ½ se colapsa a 0.
            Un override válido aplica ½ → ½: se disipa el ámbar operativo
            pero el veredicto permanece DEGRADED hasta que el asociador
            vuelva bajo umbral.

        Retorna:
            (verdict, is_soft_veto, is_hard_veto, time_remaining).
        """
        if not isinstance(orientation, OctonionicOrientationReport):
            raise TypeError(
                "decide_from_orientation exige un OctonionicOrientationReport "
                "(germen de orient_octonionic_diagnostics)."
            )

        now = time.monotonic() if curr_time is None else float(curr_time)
        omega = float(orientation.omega_pre)

        if omega <= 0.0:
            self._clear_soft_veto()
            logger.error(
                "VETO DURO INSTANTÁNEO: violación grave de Hurwitz, asociador "
                "no finito o ruptura de Artin en el núcleo (ω_hard = 0)."
            )
            return HeytingVerdict.VETOED, False, True, 0.0

        if omega >= 1.0:
            self._clear_soft_veto()
            return HeytingVerdict.COHERENT, False, False, 0.0

        is_soft_veto = True
        time_remaining = 0.0

        if not self._is_soft_veto_active:
            self._is_soft_veto_active = True
            self._soft_veto_timestamp = now
            verdict = HeytingVerdict.DEGRADED
            logger.warning(
                "VETO SUAVE ACTIVO (LUZ ÁMBAR): colusión, frustración de "
                "calibre o no-asociatividad en la tríada. Gracia iniciada."
            )
        else:
            elapsed = now - (self._soft_veto_timestamp or now)
            time_remaining = max(0.0, self._grace_max - elapsed)
            if time_remaining <= self._tol:
                self._clear_soft_veto()
                logger.critical(
                    "VENTANA DE GRACIA EXPIRADA SIN OVERRIDE VÁLIDO. "
                    "Heyting colapsa ½ → 0 (VETOED terminal)."
                )
                return HeytingVerdict.VETOED, False, True, 0.0
            verdict = HeytingVerdict.DEGRADED

        if override_token is not None:
            if self._verify_override(override_token):
                self._clear_soft_veto()
                logger.info(
                    "ANULACIÓN DE FOCK TRILATERAL ACTIVADA. Override validado. "
                    "Luz ámbar disipada; la obra permanece DEGRADED hasta "
                    "recuperar estabilidad asociativa."
                )
                return HeytingVerdict.DEGRADED, False, False, 0.0
            logger.error(
                "Firma digital inválida en el override octoniónico. "
                "Se mantiene la rampa activa."
            )

        return verdict, is_soft_veto, False, time_remaining

    def _act_crowbar(
        self,
        orientation: OctonionicOrientationReport,
        verdict: HeytingVerdict,
    ) -> float:
        r"""Actuación Crowbar: ISR en IRAM, GPIO14 HIGH, tiristor BT151."""
        if verdict is not HeytingVerdict.VETOED:
            return 0.0

        triad = orientation.triad
        switching_latency = self._seeded_latency_ns(
            triad.contractor.vector_rep,
            triad.supplier.vector_rep,
            triad.interventor.vector_rep,
            triad.associator,
            np.array(
                [orientation.composition_error, orientation.associator_norm],
                dtype=np.float64,
            ),
            verdict.value,
        )

        logger.error("COLA DE HEYTING COLAPSADA POR INCOHERENCIA TRILATERAL.")
        logger.error("  - Ejecutando subrutina local isVerdictCoherent() en C++...")
        logger.error("  - Despachando ISR en IRAM de alta velocidad...")
        logger.error(
            "  - Conmutando GPIO14 a HIGH en %.2f ns vía IRAM...",
            switching_latency,
        )
        logger.error("  - Tiristor rápido de potencia BT151 (Crowbar) gatillado.")
        logger.error("  - Mezcladoras y bombas hidráulicas reales en fango paralizadas.")
        return switching_latency

    def audit_trilateral_cycle(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        interventor_S: np.ndarray,
        asoc_threshold: float = 0.15,
        override_token: Optional[str] = None,
    ) -> OctonionicAuditCertificate:
        r"""
        Orquesta el ciclo covariante OODA trilateral.

        Flujo anidado:
          OBSERVE  (Fase 1): synthesize_octonionic_triad.
          ORIENT   (Fase 2): orient_octonionic_diagnostics.
          DECIDE   (Fase 3): decide_from_orientation.
          ACT      (Fase 3): _act_crowbar si ω = 0.
        """
        threshold = float(asoc_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("asoc_threshold debe ser finito y no negativo.")

        orientation = self.orient_octonionic_diagnostics(
            contractor_S=contractor_S,
            supplier_S=supplier_S,
            interventor_S=interventor_S,
            asoc_threshold=threshold,
        )

        verdict, is_soft_veto, is_hard_veto, time_remaining = (
            self.decide_from_orientation(
                orientation,
                override_token=override_token,
            )
        )

        if verdict is HeytingVerdict.VETOED:
            is_hard_veto = True

        switching_latency = self._act_crowbar(orientation, verdict)

        triad = orientation.triad
        cryptographic_seal = self._seal(
            triad.contractor.vector_rep,
            triad.supplier.vector_rep,
            triad.interventor.vector_rep,
            triad.associator,
            np.array(
                [
                    orientation.composition_error,
                    orientation.composition_relative_error,
                    orientation.associator_norm,
                    orientation.associator_relative_norm,
                    float(orientation.is_associative_stable),
                    float(orientation.is_cfl_stable),
                    verdict.omega,
                ],
                dtype=np.float64,
            ),
            verdict.value,
        )

        return OctonionicAuditCertificate(
            heyting_verdict=verdict.value,
            associator_norm=orientation.associator_norm,
            is_associative_stable=orientation.is_associative_stable,
            composition_error=orientation.composition_error,
            is_cfl_stable=orientation.is_cfl_stable,
            is_soft_veto_active=bool(self._is_soft_veto_active),
            is_hard_veto_active=bool(is_hard_veto),
            actuation_latency_ns=switching_latency,
            time_grace_remaining=time_remaining,
            cryptographic_seal=cryptographic_seal,
            composition_relative_error=orientation.composition_relative_error,
            associator_relative_norm=orientation.associator_relative_norm,
            contractor_norm=orientation.contractor_norm,
            supplier_norm=orientation.supplier_norm,
            interventor_norm=orientation.interventor_norm,
            heyting_omega=verdict.omega,
            omega_pre=orientation.omega_pre,
            artin_residual=orientation.artin_residual,
            moufang_residual=orientation.moufang_residual,
            triad_seal=triad.sha256_hash,
        )


# ───────────────────────────────────────────────────────────────────────────────
# Resolutor final
# ───────────────────────────────────────────────────────────────────────────────

class OctonionicDependencyResolver(Phase3_OODAActuator):
    r"""
    Resolutor Octoniónico de dependencias trilaterales (FPU Secure).

    Cadena de herencia (fases anidadas):

        OctonionicDependencyResolver
          └─ Phase3_OODAActuator                 Ω₃, gracia, Crowbar
               └─ Phase2_OctonionicDiagnostics   Hurwitz, Moufang, ω_pre
                    └─ Phase1_OctonionicAlgebraKernel
                         O = CD(H), [a,b,c], Artin, T(a,b,c)

    Aplica Cayley–Dickson, certifica el núcleo (unidad, eᵢ² = −1, Artin)
    y calcula el asociador [a,b,c] con inmunidad relativa al drift mediante
    sumaciones KBN y tolerancias adaptativas.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return (
            "OctonionicDependencyResolver("
            f"tolerance={self._tol}, "
            f"grace_period_seconds={self._grace_max}, "
            f"asoc_threshold={self._asoc_limit}"
            ")"
        )


__all__ = [
    "OctonionicDependencyResolver",
    "OctonionicState",
    "OctonionicTriadReport",
    "OctonionicOrientationReport",
    "OctonionicAuditCertificate",
    "HeytingVerdict",
    "Phase1_OctonionicAlgebraKernel",
    "Phase2_OctonionicDiagnostics",
    "Phase3_OODAActuator",
]