# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : OCTONIONIC DEPENDENCY RESOLVER (RESOLVER OCTONIÓNICO DE MALLA)      ║
║ RUTA   : app/core/octonionic_dependency_resolver.py                          ║
║ NIVEL  : Doctorado en Ciencias Matemáticas, Física Teórica y Computación     ║
║ VERSIÓN: 4.0.0-Doctoral-Cayley-Dickson-Malcev-G2-Hodge-RC-PureKernel-Nested3 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TRATADO DE FUNDAMENTACIÓN MATEMÁTICA, FÍSICA Y TOPOLÓGICA TRILATERAL:        ║
║                                                                              ║
║ 1. ÁLGEBRA DE OCTONIONES \mathbb{O} Y TEOREMA DE HURWITZ (1898):             ║
║    \mathbb{O} = \mathbb{H} \oplus \mathbb{H}\ell,                            ║
║      (q_1, q_2)(p_1, p_2) = (q_1 p_1 - \overline{p}_2 q_2,\                  ║
║                              p_2 q_1 + q_2 \overline{p}_1).                  ║
║    A DIFERENCIA de los pathiones (dim 32, donde la ruptura de alternatividad ║
║    y Moufang es ESTRUCTURALMENTE ESPERADA), en dim=8 las siguientes          ║
║    identidades son EXACTAS por el Teorema de Hurwitz y el Teorema de Artin:  ║
║      - Composición normada: \|xy\| = \|x\|\|y\| (EXACTA).                    ║
║      - Alternatividad: [x,x,y] = [y,x,x] = 0 (EXACTA, Artin).                ║
║      - Flexibilidad: [x,y,x] = 0 (EXACTA, Albert).                           ║
║      - Moufang (3 identidades): EXACTAS.                                     ║
║    Por tanto, su ruptura catastrófica en la FPU es un CERTIFICADO DE BUG     ║
║    estructural del núcleo, NO un régimen físico esperado — se cablean        ║
║    excepciones dedicadas (`OctonionicCompositionError`,                      ║
║    `OctonionicArtinFailureError`) análogas al enforcement de Albert en el    ║
║    motor pationiónico 32D.                                                   ║
║                                                                              ║
║ 2. NÚCLEO ALGEBRAICO PURO SEPARADO (ARQUITECTURA DE LA TRILOGÍA):             ║
║    Se extrae `OctonionicAlgebraKernel` — clase ESTÁTICA de funciones puras   ║
║    (sin estado, sin I/O), análoga a `CayleyDicksonAlgebra32` del motor 32D — ║
║    separando la matemática pura de la metrología con estado de la Fase 1.    ║
║    La consistencia G_2/Fano se CERTIFICA EN TIEMPO DE IMPORT sobre la TABLA  ║
║    COMPLETA 8×8 (64 entradas: identidad, autocuadrados e_i^2=-1, y las 7     ║
║    líneas de Fano con sus 6 permutaciones cíclicas/anticonmutativas cada     ║
║    una), abortando con `OctonionicFanoStructureError` ante cualquier         ║
║    inconsistencia — mismo espíritu que el axioma dimensional del motor.      ║
║                                                                              ║
║ 3. GEOMETRÍA DEL GRUPO DE LIE EXCEPCIONAL G_2 Y PLANO DE FANO:               ║
║    G_2 = \mathrm{Aut}(\mathbb{O}), \dim_{\mathbb{R}} G_2 = 14.               ║
║      e_i e_j = -\delta_{ij} e_0 + \sum_{k=1}^7 \psi_{ijk} e_k,               ║
║    con \psi_{ijk} totalmente antisimétrico (3-forma asociativa \phi de G_2). ║
║                                                                              ║
║ 4. ÁLGEBRA DE MALCEV Y PRODUCTO CRUZADO EN \operatorname{Im}(\mathbb{O}):    ║
║      u \times v = \operatorname{Im}(uv) = \tfrac{1}{2}[u, v], \quad          ║
║      J(u,v,w) = [[u,v],w]+[[v,w],u]+[[w,u],v] = -6[u,v,w].                   ║
║    Normalizado a grado homogéneo 0 para independizar la escala:              ║
║      \delta_{\mathrm{Malcev}} = \|J + 6[u,v,w]\|_2 / (\|u\|\|v\|\|w\|)^{?}.  ║
║                                                                              ║
║ 5. TOPOLOGÍA ESPECTRAL DE HODGE-LAPLACE Y EXERGÍA DE DIRICHLET EN K_3:        ║
║    \Delta_2 (2-símplex), 1-esqueleto K_3. Gram COMPENSADO de Neumaier/       ║
║    Shewchuk (evita cancelación catastrófica en el producto interno coseno).  ║
║      \lambda_2(L),\ R_K = 3(1/\lambda_2 + 1/\lambda_3),\                     ║
║      \mathcal{E}_D = \operatorname{Tr}(P^T L P).                             ║
║                                                                              ║
║ 6. GOBERNANZA PURA EN \Omega_3 (MÓNADA DE ESTADO EXPLÍCITA):                  ║
║    `OctonionicGraceState` es un value-type inmutable que sustituye la        ║
║    mutación de instancia previa, en el mismo espíritu del                    ║
║    `PathionicGraceState` del agente soberano — decisión pura de Kleisli:     ║
║      \mathrm{decide}: \mathrm{Orientation} \times \Gamma \to (V, \Gamma').   ║
║    El interlock Crowbar BT151 emplea el MISMO modelo físico RC-descompuesto  ║
║    (ISR Xtensa + propagación GPIO/APB + carga de compuerta) del agente,      ║
║    por coherencia doctrinal de toda la trilogía pationiónica.                ║
║                                                                              ║
║ 7. ARQUITECTURA EN TRES FASES ANIDADAS FUNCTORIALES (OODA EN \Omega_3):      ║
║    Phase1_OctonionicAlgebraKernel (Observe):                                 ║
║      Morfismo terminal: synthesize_octonionic_triad \to OctonionicTriadReport║
║    Phase2_OctonionicDiagnostics (Orient, hereda Phase1):                     ║
║      Morfismo terminal: orient_octonionic_diagnostics                        ║
║                         \to OctonionicOrientationReport.                     ║
║    Phase3_OODAActuator (Decide + Act, hereda Phase2):                        ║
║      Morfismo terminal: audit_trilateral_cycle \to OctonionicAuditCertificate║
║    Fachada Soberana: OctonionicDependencyResolver = Phase3_OODAActuator.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import AbstractSet, Any, Callable, Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

__version__: Final[str] = (
    "4.0.0-Doctoral-Cayley-Dickson-Malcev-G2-Hodge-RC-PureKernel-Nested3"
)

logger = logging.getLogger("APU.Core.OctonionicDependencyResolver")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_OCTONION_DIM: Final[int] = 8
_QUATERNION_DIM: Final[int] = 4
_KERNEL_IDENTITY_LIMIT: Final[float] = 1e-10
_BANACH_O8_UPPER_BOUND: Final[float] = float(math.sqrt(float(_OCTONION_DIM)))
_BANACH_L1_LINF_BOUND: Final[float] = float(_OCTONION_DIM)
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_LEGACY_OVERRIDE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_NON_ASSOCIATIVE_IDU_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)

_UNIT: Final[np.ndarray] = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)
_UNIT.setflags(write=False)

# Líneas orientadas del Plano de Fano PG(2,2): e_i * e_j = e_k (cíclico).
_FANO_LINES: Final[Tuple[Tuple[int, int, int], ...]] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)

# --- Parámetros físicos del modelo RC descompuesto del interlock Crowbar BT151 ---
_CROWBAR_GPIO: Final[str] = "GPIO14"
_CROWBAR_DEVICE: Final[str] = "BT151-800R"
_CROWBAR_IRAM_LATENCY_NS_LIMIT: Final[float] = 400.0
_ISR_CYCLE_TIME_NS: Final[float] = 1.0e9 / 240.0e6
_ISR_NOMINAL_CYCLES: Final[float] = 18.0
_ISR_CYCLE_JITTER: Final[float] = 2.5
_APB_CYCLE_TIME_NS: Final[float] = 1.0e9 / 80.0e6
_GPIO_PAD_SLEW_NS_RANGE: Final[Tuple[float, float]] = (5.0, 10.0)
_CROWBAR_R_GK_OHM: Final[float] = 220.0
_CROWBAR_C_GK_FARAD: Final[float] = 1.0e-9
_CROWBAR_V_GT_VOLTS: Final[float] = 0.8
_CROWBAR_V_SUPPLY_NOMINAL: Final[float] = 3.3
_CROWBAR_V_SUPPLY_TOLERANCE: Final[float] = 0.03


# ═══════════════════════════════════════════════════════════════════════════════
# §A. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS Y FÍSICO-NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class OctonionicEngineError(Exception):
    r"""Excepción raíz para violaciones métricas, topológicas o algebraicas en \mathbb{O}."""


class OctonionicDimensionError(OctonionicEngineError):
    r"""Detonada ante dimensiones incompatibles con \mathbb{R}^8 o \mathbb{R}^4."""


class OctonionicNumericalSingularityError(OctonionicEngineError):
    r"""Detonada ante singularidades no finitas (NaN, Inf) o desbordamiento en FPU."""


class OctonionicCompositionError(OctonionicEngineError):
    r"""
    Detonada cuando la relación de Hurwitz \|xy\| = \|x\|\|y\| —EXACTA en dim=8—
    colapsa catastróficamente. A diferencia del motor pationiónico (dim 32,
    donde esta ruptura es estructuralmente esperada), aquí certifica un BUG.
    """


class OctonionicArtinFailureError(OctonionicEngineError):
    r"""
    Detonada si la alternatividad, flexibilidad o Moufang —EXACTAS en dim=8
    por el Teorema de Artin— se violan catastróficamente en el núcleo analítico.
    """


class OctonionicFanoStructureError(OctonionicEngineError):
    r"""
    Detonada en tiempo de import si la tabla de multiplicación 8×8 generada por
    duplicación de Cayley-Dickson no coincide con la estructura proyectiva del
    Plano de Fano asumida (`_FANO_LINES`). Fallo fundacional análogo al axioma
    dimensional del motor pationiónico 32D.
    """


class OctonionicHeytingAxiomError(RuntimeError):
    r"""Detonada si \Omega_3 no satisface sus axiomas de álgebra de Heyting lineal (Gödel-Dummett)."""


# ═══════════════════════════════════════════════════════════════════════════════
# §B. METROLOGÍA NUMÉRICA, CANONICALIZACIÓN Y SUMACIÓN EXACTA (SHEWCHUK)
# ═══════════════════════════════════════════════════════════════════════════════
def _log_norm(norm: float) -> float:
    """Logaritmo neperiano seguro: x \\le 0 \\mapsto -\\infty."""
    if norm <= 0.0 or not math.isfinite(norm):
        return -math.inf
    return math.log(norm)


def _safe_exp(log_val: float) -> float:
    """Exponencial acotada numéricamente dentro del rango de doble precisión IEEE-754."""
    if log_val == -math.inf:
        return 0.0
    if log_val == math.inf or log_val >= _LOG_DBL_MAX:
        return math.inf
    if not math.isfinite(log_val) or log_val <= _LOG_DBL_TINY:
        return 0.0
    return float(math.exp(log_val))


def _canonical_bytes(part: Any) -> bytes:
    r"""Serialización determinista Little-Endian IEEE-754 de escalares y tensores."""
    if isinstance(part, np.ndarray):
        arr = np.ascontiguousarray(part, dtype=np.float64)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        return header + np.asarray(arr, dtype="<f8").tobytes(order="C")
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    if isinstance(part, (int, float, bool, np.generic)):
        x = float(part)
        if math.isnan(x):
            return b"\x7fNAN\x00\x00\x00"
        if math.isinf(x):
            return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
        if x == 0.0:
            return struct.pack("<d", 0.0)
        return struct.pack("<d", x)
    return repr(part).encode("utf-8")


def _frame_bytes(tag: bytes, payload: bytes) -> bytes:
    r"""
    Codificación canónica con marcado de longitud (length-prefixed framing),
    eliminando la ambigüedad de concatenación de una construcción Merkle-Damgård
    ingenua. Formato: u64(len(tag)) || tag || u64(len(payload)) || payload.
    """
    return struct.pack("<Q", len(tag)) + tag + struct.pack("<Q", len(payload)) + payload


def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Clona y congela un tensor para asegurar inmutabilidad estricta en RAM."""
    out = np.array(array, dtype=dtype or np.float64, copy=True, order="C")
    out.setflags(write=False)
    return out


class KBNSummationKernel:
    r"""
    Núcleo de sumación de alta fidelidad. Delega en el algoritmo de destilación
    de Shewchuk (1997) —`math.fsum`—, con cota de redondeo correcto
    (\le 1 ulp), estrictamente superior al Kahan-Babuška-Neumaier clásico
    (O(n\varepsilon)). Se conserva el nombre histórico por estabilidad de API.
    """

    __slots__ = ()

    @staticmethod
    def sum(values: np.ndarray) -> float:
        r"""Sumación exacta (redondeo correcto) de Shewchuk."""
        flat = np.asarray(values, dtype=np.float64).ravel()
        if flat.size == 0:
            return 0.0
        if not np.all(np.isfinite(flat)):
            raise OctonionicNumericalSingularityError(
                "Detección de singularidad no finita durante la sumación exacta de Shewchuk."
            )
        try:
            result = math.fsum(flat.tolist())
        except (OverflowError, ValueError) as exc:
            raise OctonionicNumericalSingularityError(
                "Desbordamiento irrecuperable en la destilación de Shewchuk."
            ) from exc
        if not math.isfinite(result):
            raise OctonionicNumericalSingularityError("Resultado no finito en sumación exacta.")
        return float(result)

    @staticmethod
    def dot(x: np.ndarray, y: np.ndarray) -> float:
        r"""Producto interno exacto \\langle x, y \\rangle = \\sum_i x_i y_i."""
        arr_x = np.ravel(np.asarray(x, dtype=np.float64))
        arr_y = np.ravel(np.asarray(y, dtype=np.float64))
        if arr_x.shape != arr_y.shape:
            raise OctonionicDimensionError("Discrepancia dimensional en producto interno.")
        return KBNSummationKernel.sum(arr_x * arr_y)

    @staticmethod
    def norm(values: np.ndarray) -> float:
        r"""Norma euclídea \\ell_2 con pre-escalado de módulo máximo (inmune a over/underflow)."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if not np.all(np.isfinite(arr)):
            raise OctonionicNumericalSingularityError("Norma no finita sobre componentes NaN/Inf.")
        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            return 0.0
        scaled = arr / max_abs
        sum_sq = KBNSummationKernel.sum(scaled * scaled)
        if sum_sq < 0.0:
            sum_sq = 0.0 if sum_sq >= -10.0 * _MACHINE_EPS else (_ for _ in ()).throw(
                OctonionicNumericalSingularityError("Residuo negativo no conforme en norma exacta.")
            )
        return float(max_abs * math.sqrt(sum_sq))

    @staticmethod
    def pairwise_gram(matrix: np.ndarray) -> np.ndarray:
        r"""
        Matriz de Gram G_{ij} = \\langle \\mathrm{fila}_i, \\mathrm{fila}_j \\rangle
        vía productos internos exactos, evitando cancelación catastrófica del
        producto matricial denso `A @ A.T` sobre nubes de baja cardinalidad (n=3).
        """
        mat = np.asarray(matrix, dtype=np.float64)
        n = mat.shape[0]
        gram = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            gram[i, i] = KBNSummationKernel.dot(mat[i], mat[i])
            for j in range(i + 1, n):
                val = KBNSummationKernel.dot(mat[i], mat[j])
                gram[i, j] = val
                gram[j, i] = val
        return gram


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ÁLGEBRA DE HEYTING \Omega_3 CON AUTO-VERIFICACIÓN AXIOMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════
class HeytingVerdict(str, Enum):
    r"""
    Álgebra de Gödel-Dummett ternaria \Omega_3 = \{\bot \prec \tfrac12 \prec \top\}.
    Certificada exhaustivamente en tiempo de import (`_verify_heyting_omega3_axioms`).
    """

    VETOED = _VERDICT_VETOED
    DEGRADED = _VERDICT_DEGRADED
    COHERENT = _VERDICT_COHERENT

    @property
    def lattice_order(self) -> int:
        return {"VETOED": 0, "DEGRADED": 1, "COHERENT": 2}[self.value]

    @property
    def omega(self) -> float:
        return {"VETOED": 0.0, "DEGRADED": 0.5, "COHERENT": 1.0}[self.value]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0 + _MACHINE_EPS:
            return cls.VETOED
        if value < 1.0 - _MACHINE_EPS:
            return cls.DEGRADED
        return cls.COHERENT

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Ínfimo: a \\wedge b = \\min(a,b)."""
        return self if self.lattice_order <= other.lattice_order else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Supremo: a \\vee b = \\max(a,b)."""
        return self if self.lattice_order >= other.lattice_order else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Pseudocomplemento relativo: a \\to b = \\top si a \\le b, si no b."""
        return HeytingVerdict.COHERENT if self.lattice_order <= other.lattice_order else other

    def neg(self) -> "HeytingVerdict":
        r"""Negación intuicionista: \\neg a = a \\to \\bot."""
        return self.implies(HeytingVerdict.VETOED)


def _heyting_meet_all(predicates: Iterable[HeytingVerdict]) -> HeytingVerdict:
    """Meet de una secuencia de veredictos; elemento neutro: COHERENT."""
    acc = HeytingVerdict.COHERENT
    for val in predicates:
        acc = acc.meet(val)
        if acc is HeytingVerdict.VETOED:
            break
    return acc


def _verify_heyting_omega3_axioms() -> None:
    r"""
    Certificación axiomática exhaustiva de \Omega_3 (conmutatividad, asociatividad,
    absorción, prelinealidad de Gödel y residuación de Heyting). Ejecutada en
    tiempo de import; lanza `OctonionicHeytingAxiomError` ante cualquier violación.
    """
    elems: Tuple[HeytingVerdict, ...] = tuple(HeytingVerdict)
    top, bot = HeytingVerdict.COHERENT, HeytingVerdict.VETOED

    for a in elems:
        if a.meet(bot) is not bot or a.join(top) is not top:
            raise OctonionicHeytingAxiomError(f"Violación de cotas del retículo en a={a!r}.")
        if a.meet(a) is not a or a.join(a) is not a:
            raise OctonionicHeytingAxiomError(f"Violación de idempotencia en a={a!r}.")
        if a.meet(top) is not a or a.join(bot) is not a:
            raise OctonionicHeytingAxiomError(f"Violación de identidad neutra en a={a!r}.")

    for a in elems:
        for b in elems:
            if a.meet(b) is not b.meet(a) or a.join(b) is not b.join(a):
                raise OctonionicHeytingAxiomError("Violación de conmutatividad.")
            if a.meet(a.join(b)) is not a or a.join(a.meet(b)) is not a:
                raise OctonionicHeytingAxiomError("Violación de absorción.")
            if a.implies(b).join(b.implies(a)) is not top:
                raise OctonionicHeytingAxiomError("Violación de prelinealidad de Gödel.")

    for a in elems:
        for b in elems:
            for c in elems:
                if a.meet(b).meet(c) is not a.meet(b.meet(c)):
                    raise OctonionicHeytingAxiomError("Violación de asociatividad de meet.")
                lhs = c.lattice_order <= a.implies(b).lattice_order
                rhs = a.meet(c).lattice_order <= b.lattice_order
                if lhs != rhs:
                    raise OctonionicHeytingAxiomError("Violación de residuación de Heyting.")

    logger.debug("Axiomática de \u03a9_3 (Heyting-G\u00f6del-Dummett) verificada exhaustivamente.")


_verify_heyting_omega3_axioms()


# ═══════════════════════════════════════════════════════════════════════════════
# §D. NÚCLEO ALGEBRAICO PURO Y ESTÁTICO DE OCTONIONES (SIN ESTADO)
# ═══════════════════════════════════════════════════════════════════════════════
class OctonionicAlgebraKernel:
    r"""
    Implementación PURA, ESTÁTICA y SIN ESTADO del álgebra de octoniones
    \mathbb{O}, análoga arquitectónicamente a `CayleyDicksonAlgebra32` del
    motor pationiónico 32D. Separa la matemática exacta de la metrología con
    estado de la Fase 1 (`Phase1_OctonionicAlgebraKernel`), permitiendo su
    certificación fundacional en tiempo de import sin requerir instanciación.
    """

    __slots__ = ()

    @staticmethod
    def _validate4(q: Sequence[float], name: str) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (_QUATERNION_DIM,):
            raise OctonionicDimensionError(f"{name} debe residir en R^4. Forma={arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise OctonionicNumericalSingularityError(f"{name} contiene valores no finitos.")
        return arr

    @staticmethod
    def _validate8(o: Sequence[float], name: str) -> np.ndarray:
        arr = np.asarray(o, dtype=np.float64)
        if arr.shape != (_OCTONION_DIM,):
            raise OctonionicDimensionError(f"{name} debe residir en R^8. Forma={arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise OctonionicNumericalSingularityError(f"{name} contiene valores no finitos.")
        return arr

    @classmethod
    def quaternion_conjugate(cls, q: Sequence[float]) -> np.ndarray:
        r"""Involución en \\mathbb{H}: \\overline{(q_0,q_1,q_2,q_3)} = (q_0,-q_1,-q_2,-q_3)."""
        arr = cls._validate4(q, "cuaternión")
        return np.array([arr[0], -arr[1], -arr[2], -arr[3]], dtype=np.float64)

    @classmethod
    def quaternion_multiply(cls, q: Sequence[float], p: Sequence[float]) -> np.ndarray:
        r"""Producto de Hamilton q p \\in \\mathbb{H} con sumación exacta por componente."""
        q0, q1, q2, q3 = cls._validate4(q, "q")
        p0, p1, p2, p3 = cls._validate4(p, "p")
        r0 = KBNSummationKernel.sum(np.array([q0 * p0, -q1 * p1, -q2 * p2, -q3 * p3]))
        r1 = KBNSummationKernel.sum(np.array([q0 * p1, q1 * p0, q2 * p3, -q3 * p2]))
        r2 = KBNSummationKernel.sum(np.array([q0 * p2, -q1 * p3, q2 * p0, q3 * p1]))
        r3 = KBNSummationKernel.sum(np.array([q0 * p3, q1 * p2, -q2 * p1, q3 * p0]))
        return np.array([r0, r1, r2, r3], dtype=np.float64)

    @classmethod
    def octonion_conjugate(cls, o: Sequence[float]) -> np.ndarray:
        r"""Involución canónica: \\overline{o} = (o_0, -o_1, \\dots, -o_7)."""
        arr = cls._validate8(o, "o")
        out = -arr.copy()
        out[0] = arr[0]
        return out

    @classmethod
    def octonion_multiply(cls, a: Sequence[float], b: Sequence[float]) -> np.ndarray:
        r"""
        Producto de Cayley-Dickson:
          (q_1,q_2)(p_1,p_2) = (q_1 p_1 - \\overline{p}_2 q_2,\\ p_2 q_1 + q_2 \\overline{p}_1).
        """
        arr_a = cls._validate8(a, "a")
        arr_b = cls._validate8(b, "b")
        q1, q2 = arr_a[0:4], arr_a[4:8]
        p1, p2 = arr_b[0:4], arr_b[4:8]

        part_a = cls.quaternion_multiply(q1, p1) - cls.quaternion_multiply(
            cls.quaternion_conjugate(p2), q2
        )
        part_b = cls.quaternion_multiply(p2, q1) + cls.quaternion_multiply(
            q2, cls.quaternion_conjugate(p1)
        )
        result = np.concatenate([part_a, part_b])
        if not np.all(np.isfinite(result)):
            raise OctonionicNumericalSingularityError("Producto octoniónico divergió (NaN/Inf).")
        return result

    @classmethod
    def octonion_inverse(cls, o: Sequence[float], norm_sq: Optional[float] = None) -> np.ndarray:
        r"""Inversa en el álgebra de división: o^{-1} = \\overline{o} / \\|o\\|^2."""
        arr = cls._validate8(o, "o")
        n_sq = float(KBNSummationKernel.dot(arr, arr)) if norm_sq is None else norm_sq
        if n_sq <= _WILKINSON_FLOOR:
            raise ZeroDivisionError("El octonión no es invertible (norma casi nula).")
        return cls.octonion_conjugate(arr) / n_sq

    @classmethod
    def commutator(cls, a: Sequence[float], b: Sequence[float]) -> np.ndarray:
        r"""Conmutador de Lie [a,b] = ab - ba."""
        return cls.octonion_multiply(a, b) - cls.octonion_multiply(b, a)

    @classmethod
    def associator(cls, a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> np.ndarray:
        r"""Asociador trilateral [a,b,c] = (ab)c - a(bc)."""
        left = cls.octonion_multiply(cls.octonion_multiply(a, b), c)
        right = cls.octonion_multiply(a, cls.octonion_multiply(b, c))
        return left - right

    @classmethod
    def malcev_cross(cls, u: Sequence[float], v: Sequence[float]) -> np.ndarray:
        r"""Producto cruzado de Malcev: u \\times v = \\operatorname{Im}(uv) = \\tfrac12[u,v]."""
        pure_u = cls._validate8(u, "u").copy()
        pure_u[0] = 0.0
        pure_v = cls._validate8(v, "v").copy()
        pure_v[0] = 0.0
        prod = cls.octonion_multiply(pure_u, pure_v)
        prod[0] = 0.0
        return prod

    @classmethod
    def moufang_defects(
        cls,
        a: Sequence[float],
        b: Sequence[float],
        c: Sequence[float],
        ab: Optional[np.ndarray] = None,
        bc: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        r"""
        Calcula los TRES defectos de las identidades de Moufang (EXACTAS en
        \\mathbb{O}, a diferencia de \\mathbb{P} donde se rompen):
          Central: (ab)(ca) = a(bc)a.
          Izquierda: ((ab)a)c = a(b(ac)).
          Derecha: ((ba)c)a = b(a(ca)).
        Reutiliza `ab`, `bc` si se proveen (evita recómputo entre fases).
        """
        ab_v = cls.octonion_multiply(a, b) if ab is None else ab
        bc_v = cls.octonion_multiply(b, c) if bc is None else bc
        ca_v = cls.octonion_multiply(c, a)
        ac_v = cls.octonion_multiply(a, c)
        ba_v = cls.octonion_multiply(b, a)

        lhs_central = cls.octonion_multiply(ab_v, ca_v)
        rhs_central = cls.octonion_multiply(cls.octonion_multiply(a, bc_v), a)
        central_defect = KBNSummationKernel.norm(lhs_central - rhs_central)

        lhs_left = cls.octonion_multiply(cls.octonion_multiply(ab_v, a), c)
        rhs_left = cls.octonion_multiply(a, cls.octonion_multiply(b, ac_v))
        left_defect = KBNSummationKernel.norm(lhs_left - rhs_left)

        lhs_right = cls.octonion_multiply(cls.octonion_multiply(ba_v, c), a)
        rhs_right = cls.octonion_multiply(b, cls.octonion_multiply(a, ca_v))
        right_defect = KBNSummationKernel.norm(lhs_right - rhs_right)

        return left_defect, right_defect, central_defect


def _certify_fano_and_g2_structure_at_import() -> None:
    r"""
    CERTIFICACIÓN FUNDACIONAL EN TIEMPO DE IMPORT de la tabla de multiplicación
    completa 8×8 (64 entradas) generada por `OctonionicAlgebraKernel` contra la
    estructura proyectiva del Plano de Fano (`_FANO_LINES`):
      1. Identidad: e_0 e_0 = e_0; e_0 e_i = e_i e_0 = e_i.
      2. Autocuadrados imaginarios: e_i e_i = -e_0, i=1..7 (alternatividad).
      3. Las 7 líneas de Fano CON sus 6 permutaciones cíclicas/anticonmutativas:
         e_i e_j = e_k, e_j e_i = -e_k, e_j e_k = e_i, e_k e_j = -e_i,
         e_k e_i = e_j, e_i e_k = -e_j.
    Fallo fundacional análogo al axioma dimensional del motor 32D: aborta la
    carga del módulo con `OctonionicFanoStructureError` si la tabla generada
    por Cayley-Dickson no coincide EXACTAMENTE (salvo ruido IEEE-754) con la
    geometría de Fano asumida.
    """

    def basis(i: int) -> np.ndarray:
        v = np.zeros(8, dtype=np.float64)
        v[i] = 1.0
        return v

    e = [basis(i) for i in range(8)]
    max_err = 0.0

    def check(lhs: np.ndarray, rhs: np.ndarray) -> None:
        nonlocal max_err
        err = float(np.max(np.abs(lhs - rhs)))
        max_err = max(max_err, err)

    check(OctonionicAlgebraKernel.octonion_multiply(e[0], e[0]), e[0])
    for i in range(1, 8):
        check(OctonionicAlgebraKernel.octonion_multiply(e[0], e[i]), e[i])
        check(OctonionicAlgebraKernel.octonion_multiply(e[i], e[0]), e[i])
        check(OctonionicAlgebraKernel.octonion_multiply(e[i], e[i]), -e[0])

    for (i, j, k) in _FANO_LINES:
        check(OctonionicAlgebraKernel.octonion_multiply(e[i], e[j]), e[k])
        check(OctonionicAlgebraKernel.octonion_multiply(e[j], e[i]), -e[k])
        check(OctonionicAlgebraKernel.octonion_multiply(e[j], e[k]), e[i])
        check(OctonionicAlgebraKernel.octonion_multiply(e[k], e[j]), -e[i])
        check(OctonionicAlgebraKernel.octonion_multiply(e[k], e[i]), e[j])
        check(OctonionicAlgebraKernel.octonion_multiply(e[i], e[k]), -e[j])

    if max_err > _KERNEL_IDENTITY_LIMIT:
        raise OctonionicFanoStructureError(
            f"Inconsistencia fundacional G_2/Fano: residuo máximo={max_err:.3e} excede "
            f"el límite {_KERNEL_IDENTITY_LIMIT:.3e}. La tabla de multiplicación generada "
            "por duplicación de Cayley-Dickson NO coincide con la geometría de Fano asumida."
        )
    logger.debug("Estructura G_2/Fano (tabla 8x8 completa) certificada. Residuo máximo=%.3e", max_err)


_certify_fano_and_g2_structure_at_import()


# ═══════════════════════════════════════════════════════════════════════════════
# §E. DTOs INMUTABLES DEL ESPACIO DE FASES OCTONIÓNICO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class OctonionicThresholds:
    r"""Fronteras metrológicas y tolerancias analíticas inmutables para \mathbb{O}."""

    absolute_tolerance: float = 1e-12
    asoc_threshold: float = 0.15
    hurwitz_tolerance: float = 1e-9
    artin_tolerance: float = 1e-8
    moufang_tolerance: float = 1e-8
    fiedler_min: float = 1e-4
    kirchhoff_max: float = 1e6
    dirichlet_max: float = 1e5
    condition_number_limit: float = 1e12
    # Umbrales de CERTIFICADO DE BUG (catastrófico, muchos órdenes de magnitud
    # por encima del ruido IEEE-754 esperado); distintos de los umbrales
    # diagnósticos anteriores, que alimentan la clasificación Ω_3.
    catastrophic_factor: float = 1.0e5
    catastrophic_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            val = float(getattr(self, field))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"El umbral {field} debe ser finito y estrictamente positivo.")


@dataclass(frozen=True, slots=True)
class BanachRegularityReport:
    r"""
    Certificado inmutable de regularidad de Banach en \mathbb{R}^8, con
    VERIFICACIÓN EFECTIVA de la cota demostrada 1 \le \kappa_B(x) \le \sqrt{8}.
    """

    norm_1: float
    norm_2: float
    norm_inf: float
    ratio_12: float
    ratio_2inf: float
    ratio_1inf: float
    banach_distortion: float
    kappa_lower_bound: float
    kappa_upper_bound: float
    is_within_theoretical_bounds: bool


@dataclass(frozen=True, slots=True)
class OctonionicState:
    r"""Estado hipercomplejo inmutable de un elemento O \in \mathbb{O} (8D)."""

    vector_rep: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    norm: float
    norm_squared: float
    real_part: float
    imag_norm: float
    is_unitary: bool
    banach_report: BanachRegularityReport
    sha256_hash: str


@dataclass(frozen=True, slots=True)
class OctonionicTriadReport:
    r"""GERMEN FASE 1 \longrightarrow FASE 2. Tríada algebraica de Cayley-Dickson, Artin y Malcev."""

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
    artin_relative_residual: float
    jacobiator_norm: float
    malcev_residual: float
    malcev_relative_residual: float
    fano_3form_value: float
    sha256_hash: str


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FASE 1 — OBSERVE (METROLOGÍA CON ESTADO SOBRE EL NÚCLEO PURO)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_OctonionicAlgebraKernel:
    r"""
    FASE 1: Observe.
    Categoría Functorial: (\mathbb{R}^8)^3 \longrightarrow \mathbf{OctonionicTriadReport}.

    Responsabilidades axiomáticas:
      1. Ingesta, saneamiento estricto y congelamiento inmutable de tensores 8D,
         DELEGANDO la aritmética pura a `OctonionicAlgebraKernel` (certificada
         en tiempo de import).
      2. Certificación de regularidad de Banach sobre \mathbb{R}^8, incluyendo
         la VERIFICACIÓN EFECTIVA de la cota \kappa_B \in [1, \sqrt{8}].
      3. Auditoría del Teorema de Artin (alternadores y flexibilidad) y del
         álgebra de Malcev (Jacobiator), NORMALIZADOS a grado homogéneo 0 y
         con ENFORCEMENT DURO (`OctonionicArtinFailureError`) ante violación
         catastrófica — pues estas identidades son EXACTAS en dim=8.

    Morfismo Terminal: `synthesize_octonionic_triad`.
    """

    __slots__ = ("_tol", "_thresholds", "_strict_axioms")

    def __init__(self, tolerance: float = 1e-12, strict_axioms: bool = True) -> None:
        self._tol: Final[float] = float(tolerance)
        self._thresholds: Final[OctonionicThresholds] = OctonionicThresholds(
            absolute_tolerance=self._tol
        )
        self._strict_axioms: Final[bool] = bool(strict_axioms)

    def _relative_tolerance(self, scale: float = 1.0) -> float:
        return max(self._tol, 10.0 * _MACHINE_EPS * max(1.0, float(scale)))

    def _sha256_payload(self, *parts: Any) -> str:
        sha = hashlib.sha256()
        for idx, part in enumerate(parts):
            sha.update(_frame_bytes(f"P{idx}".encode("ascii"), _canonical_bytes(part)))
        return sha.hexdigest()

    def certify_banach_regularity_8d(self, arr: np.ndarray) -> BanachRegularityReport:
        r"""
        Audita las desigualdades de equivalencia de normas en \mathbb{R}^8 Y
        la cota DEMOSTRADA de distorsión convexa 1 \le \kappa_B(x) \le \sqrt{8}
        (prueba: x_i^2 \le |x_i|\|x\|_\infty \Rightarrow \kappa_B \ge 1;
        Cauchy-Schwarz \|x\|_1 \le \sqrt{8}\|x\|_2 y \|x\|_\infty \le \|x\|_2
        \Rightarrow \kappa_B \le \sqrt{8}).
        """
        n1 = KBNSummationKernel.sum(np.abs(arr))
        n2 = KBNSummationKernel.norm(arr)
        ninf = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0

        if not (math.isfinite(n1) and math.isfinite(n2) and math.isfinite(ninf)):
            raise OctonionicNumericalSingularityError("Norma no finita en regularidad de Banach.")

        zero_tol = max(self._tol, _WILKINSON_FLOOR)
        slack = 10.0 * _MACHINE_EPS

        if n2 <= zero_tol:
            r12, r2inf, r1inf, kappa = 1.0, 1.0, 1.0, 1.0
            is_valid = True
        else:
            denom_inf = max(ninf, zero_tol)
            r12 = n1 / n2
            r2inf = n2 / denom_inf
            r1inf = n1 / denom_inf
            kappa = (n1 * ninf) / (n2 * n2)

            ratio_valid = (
                r12 + slack >= 1.0 and r12 <= _BANACH_O8_UPPER_BOUND * (1.0 + slack)
                and r2inf + slack >= 1.0 and r2inf <= _BANACH_O8_UPPER_BOUND * (1.0 + slack)
                and r1inf + slack >= 1.0 and r1inf <= _BANACH_L1_LINF_BOUND * (1.0 + slack)
            )
            kappa_valid = kappa + slack >= 1.0 and kappa <= _BANACH_O8_UPPER_BOUND * (1.0 + slack)
            is_valid = ratio_valid and kappa_valid

        return BanachRegularityReport(
            norm_1=n1, norm_2=n2, norm_inf=ninf,
            ratio_12=r12, ratio_2inf=r2inf, ratio_1inf=r1inf,
            banach_distortion=kappa,
            kappa_lower_bound=1.0, kappa_upper_bound=_BANACH_O8_UPPER_BOUND,
            is_within_theoretical_bounds=is_valid,
        )

    def build_state(self, S: Sequence[float]) -> OctonionicState:
        """Construye un estado octoniónico inmutable con metrología exacta y certificación de Banach."""
        arr = OctonionicAlgebraKernel._validate8(S, "estado octoniónico")
        q1, q2 = arr[0:4].copy(), arr[4:8].copy()

        norm_val = KBNSummationKernel.norm(arr)
        norm_sq = norm_val * norm_val if norm_val < 1e150 else math.inf
        unit_tol = self._relative_tolerance(max(1.0, norm_val))
        is_unit = bool(abs(norm_val - 1.0) <= unit_tol)

        real_part = float(arr[0])
        imag_norm = KBNSummationKernel.norm(arr[1:])
        banach_rep = self.certify_banach_regularity_8d(arr)

        sha_hash = self._sha256_payload(arr, q1, q2, np.array([norm_val, norm_sq, imag_norm]))

        return OctonionicState(
            vector_rep=_immutable(arr), q1=_immutable(q1), q2=_immutable(q2),
            norm=norm_val, norm_squared=norm_sq, real_part=real_part, imag_norm=imag_norm,
            is_unitary=is_unit, banach_report=banach_rep, sha256_hash=sha_hash,
        )

    def _as_state(self, x: Any) -> OctonionicState:
        return x if isinstance(x, OctonionicState) else self.build_state(x)

    def octonionic_multiply(self, a: Any, b: Any) -> OctonionicState:
        """Envoltorio con estado sobre `OctonionicAlgebraKernel.octonion_multiply`."""
        st_a, st_b = self._as_state(a), self._as_state(b)
        return self.build_state(
            OctonionicAlgebraKernel.octonion_multiply(st_a.vector_rep, st_b.vector_rep)
        )

    def _relative_defect(self, defect_norm: float, *state_norms_with_degree: Tuple[float, int]) -> float:
        r"""
        Normalización homogénea de grado 0: \delta = \|D\|_2 / \prod_i \|x_i\|^{d_i}.
        Calculada en el dominio logarítmico para robustez frente a
        sub/sobredesbordamiento en escalas extremas.
        """
        if not math.isfinite(defect_norm):
            return math.inf
        log_scale = sum(deg * _log_norm(max(n, _WILKINSON_FLOOR)) for n, deg in state_norms_with_degree)
        return _safe_exp(_log_norm(defect_norm) - log_scale)

    def _enforce_axiom(
        self, defect: float, bound: float, label: str, exc_type: type
    ) -> None:
        r"""
        Enforcement estilo Albert: distingue ruido IEEE-754 esperado (silencioso)
        de violación catastrófica (>= `catastrophic_factor` veces la tolerancia,
        con piso absoluto `catastrophic_floor`), que certifica un BUG del núcleo
        dado que la identidad auditada es EXACTA en \mathbb{O}.
        """
        if not self._strict_axioms:
            return
        thr = self._thresholds
        if defect > thr.catastrophic_factor * bound and defect > thr.catastrophic_floor:
            raise exc_type(
                f"Violación catastrófica de «{label}»: defecto={defect:.6e} excede en más de "
                f"{thr.catastrophic_factor:.0e}x la tolerancia nominal ({bound:.3e}). Esta "
                "identidad es EXACTA en dim=8 (Teorema de Artin/Hurwitz); su ruptura grosera "
                "certifica un defecto estructural del núcleo de multiplicación."
            )

    def malcev_jacobiator(
        self, u: Sequence[float], v: Sequence[float], w: Sequence[float]
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        Verifica J(u,v,w) = [[u,v],w]+[[v,w],u]+[[w,u],v] = -6[u,v,w] en
        \operatorname{Im}(\mathbb{O}). Retorna (J, \|residuo\|_2, \delta_{\text{rel}}).
        """
        u_p = OctonionicAlgebraKernel._validate8(u, "u").copy(); u_p[0] = 0.0
        v_p = OctonionicAlgebraKernel._validate8(v, "v").copy(); v_p[0] = 0.0
        w_p = OctonionicAlgebraKernel._validate8(w, "w").copy(); w_p[0] = 0.0

        c_uv = OctonionicAlgebraKernel.commutator(u_p, v_p)
        c_vw = OctonionicAlgebraKernel.commutator(v_p, w_p)
        c_wu = OctonionicAlgebraKernel.commutator(w_p, u_p)

        jacobiator = (
            OctonionicAlgebraKernel.commutator(c_uv, w_p)
            + OctonionicAlgebraKernel.commutator(c_vw, u_p)
            + OctonionicAlgebraKernel.commutator(c_wu, v_p)
        )
        assoc = OctonionicAlgebraKernel.associator(u_p, v_p, w_p)
        residual = KBNSummationKernel.norm(jacobiator + 6.0 * assoc)

        n_u = KBNSummationKernel.norm(u_p)
        n_v = KBNSummationKernel.norm(v_p)
        n_w = KBNSummationKernel.norm(w_p)
        relative = self._relative_defect(residual, (n_u, 1), (n_v, 1), (n_w, 1))

        return jacobiator, residual, relative

    def fano_associative_3form(self, a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        r"""3-forma asociativa de G_2: \phi(a,b,c) = \langle a, b \times c \rangle."""
        arr_a = OctonionicAlgebraKernel._validate8(a, "a")
        return KBNSummationKernel.dot(arr_a, OctonionicAlgebraKernel.malcev_cross(b, c))

    def synthesize_octonionic_triad(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
    ) -> OctonionicTriadReport:
        r"""
        MORFISMO TERMINAL DE LA FASE 1 / GERMEN DE LA FASE 2.
        Construye la tríada completa T(a,b,c) en \mathbb{O}^7 \times \mathbb{R}^8,
        con ENFORCEMENT DURO de las identidades exactas de Artin/Albert.
        """
        a, b, c = self._as_state(contractor_S), self._as_state(supplier_S), self._as_state(interventor_S)

        ab = self.octonionic_multiply(a, b)
        bc = self.octonionic_multiply(b, c)
        left = self.octonionic_multiply(ab, c)
        right = self.octonionic_multiply(a, bc)

        associator = left.vector_rep - right.vector_rep
        assoc_norm = KBNSummationKernel.norm(associator)

        alt_left = KBNSummationKernel.norm(
            OctonionicAlgebraKernel.associator(a.vector_rep, a.vector_rep, b.vector_rep)
        )
        alt_right = KBNSummationKernel.norm(
            OctonionicAlgebraKernel.associator(b.vector_rep, c.vector_rep, c.vector_rep)
        )
        flex_norm = KBNSummationKernel.norm(
            OctonionicAlgebraKernel.associator(a.vector_rep, b.vector_rep, a.vector_rep)
        )
        artin_res_raw = max(alt_left, alt_right, flex_norm)
        # Normalización homogénea: [x,x,y] y [x,y,x] son de grado (2 en x, 1 en y).
        artin_rel = max(
            self._relative_defect(alt_left, (a.norm, 2), (b.norm, 1)),
            self._relative_defect(alt_right, (b.norm, 1), (c.norm, 2)),
            self._relative_defect(flex_norm, (a.norm, 2), (b.norm, 1)),
        )
        self._enforce_axiom(
            artin_res_raw, self._thresholds.artin_tolerance,
            "Teorema de Artin (alternatividad/flexibilidad)", OctonionicArtinFailureError,
        )

        jacobiator, malcev_res, malcev_rel = self.malcev_jacobiator(
            a.vector_rep, b.vector_rep, c.vector_rep
        )
        jacobi_norm = KBNSummationKernel.norm(jacobiator)
        fano_val = self.fano_associative_3form(a.vector_rep, b.vector_rep, c.vector_rep)

        sha_hash = self._sha256_payload(
            a.sha256_hash, b.sha256_hash, c.sha256_hash, associator,
            np.array([assoc_norm, artin_res_raw, malcev_res, fano_val]),
        )

        return OctonionicTriadReport(
            contractor=a, supplier=b, interventor=c,
            product_ab=ab, product_bc=bc, product_left=left, product_right=right,
            associator=_immutable(associator), associator_norm=assoc_norm,
            left_alternator_norm=alt_left, right_alternator_norm=alt_right, flexibility_norm=flex_norm,
            artin_residual=artin_res_raw, artin_relative_residual=artin_rel,
            jacobiator_norm=jacobi_norm, malcev_residual=malcev_res, malcev_relative_residual=malcev_rel,
            fano_3form_value=fano_val, sha256_hash=sha_hash,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §G. FASE 2 — ORIENT: HURWITZ, MOUFANG COMPLETO, HODGE-LAPLACE, PREDICADOS Ω_3
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class OctonionicOrientationReport:
    r"""
    GERMEN FASE 2 \longrightarrow FASE 3. Orientación física, Hurwitz, las TRES
    identidades de Moufang, geometría espectral de Hodge sobre K_3, y
    preclasificación auditable en \Omega_3 (predicados nombrados + razones).
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
    artin_relative_residual: float
    moufang_left_defect: float
    moufang_right_defect: float
    moufang_central_defect: float
    moufang_relative_max: float
    laplacian_connectivity: float
    laplacian_spectral_gap: float
    kirchhoff_index: float
    dirichlet_exergy: float
    omega_pre: HeytingVerdict
    conjuncts: Tuple[Tuple[str, str], ...]
    reasons: Tuple[str, ...]
    sha256_hash: str
    contractor_norm: float
    supplier_norm: float
    interventor_norm: float


class Phase2_OctonionicDiagnostics(Phase1_OctonionicAlgebraKernel):
    r"""
    FASE 2: Orient.
    Hereda ontológicamente de la Fase 1.

    Categoría Functorial:
      \mathbf{OctonionicTriadReport} \times \mathcal{P}_{\mathrm{thresholds}}
      \longrightarrow \mathbf{OctonionicOrientationReport}.

    Responsabilidades axiomáticas:
      1. Morfismo de inicio `continue_from_triad_report`: consume el objeto
         terminal de la Fase 1, REUTILIZANDO `triad.product_ab`/`product_bc`
         para evitar recómputo de productos ya resueltos (Hurwitz(a,b),
         Hurwitz(b,c), Moufang), calculando solo el producto fresco `c·a`.
      2. Auditoría de Hurwitz \delta_H(x,y)=|\|xy\|-\|x\|\|y\|| con ENFORCEMENT
         DURO (`OctonionicCompositionError`) ante ruptura catastrófica —EXACTA
         en dim=8.
      3. Las TRES identidades de Moufang (izquierda, derecha, central),
         normalizadas a grado homogéneo 0.
      4. Diagnóstico espectral de Hodge-Laplace sobre K_3 usando Gram
         COMPENSADO de Shewchuk (`KBNSummationKernel.pairwise_gram`).
      5. Preclasificación Ω_3 mediante ARQUITECTURA DE PREDICADOS NOMBRADOS
         con razones auditables (reemplaza la cadena opaca de `_heyting_meet`
         anidados de v3.1.0).

    Morfismo Terminal: `orient_octonionic_diagnostics`.
    """

    __slots__ = ()

    def continue_from_triad_report(
        self,
        triad: OctonionicTriadReport,
        asoc_threshold: float = 0.15,
    ) -> OctonionicOrientationReport:
        r"""MORFISMO DE CONTINUACIÓN DE LA FASE 2. Enlace formal con el final de la Fase 1."""
        return self.observe_octonionic_triad(triad=triad, asoc_threshold=asoc_threshold)

    def compute_hurwitz_error(
        self,
        a: OctonionicState,
        b: OctonionicState,
        precomputed_product: Optional[OctonionicState] = None,
    ) -> Tuple[float, float, OctonionicState]:
        r"""
        Mide \delta_H = |\|ab\| - \|a\|\|b\||, reutilizando `precomputed_product`
        si se provee (evita recómputo de productos ya resueltos en la Fase 1).
        """
        ab = precomputed_product if precomputed_product is not None else self.octonionic_multiply(a, b)
        expected = a.norm * b.norm
        abs_err = abs(ab.norm - expected)
        rel_err = abs_err / max(expected, _WILKINSON_FLOOR)
        return float(abs_err), float(rel_err), ab

    def _diagnose_hodge_laplacian_k3(
        self, states: Tuple[OctonionicState, OctonionicState, OctonionicState]
    ) -> Tuple[float, float, float, float]:
        r"""
        Auditoría espectral de Hodge sobre K_3 mediante Gram COMPENSADO de
        Shewchuk (evita cancelación catastrófica del producto coseno denso).
        Retorna (\lambda_2, \text{gap}, R_K, \mathcal{E}_D).
        """
        coords = np.vstack([s.vector_rep for s in states])
        norms = np.array([s.norm for s in states], dtype=np.float64)

        if np.any(norms <= self._thresholds.absolute_tolerance):
            return 0.0, 0.0, math.inf, 0.0

        normed = coords / norms[:, np.newaxis]
        cosine_matrix = np.clip(KBNSummationKernel.pairwise_gram(normed), -1.0, 1.0)
        weights = 0.5 * (1.0 + cosine_matrix)
        np.fill_diagonal(weights, 0.0)

        degrees = np.sum(weights, axis=1)
        laplacian = np.diag(degrees) - weights
        laplacian = 0.5 * (laplacian + laplacian.T)

        try:
            eigvals = la.eigvalsh(laplacian, check_finite=True)
        except la.LinAlgError:
            eigvals = np.linalg.eigvalsh(laplacian)

        eigvals = np.sort(np.maximum(np.real(eigvals), 0.0))
        fiedler = float(eigvals[1]) if eigvals.size >= 2 else 0.0
        gap = float(eigvals[1] - eigvals[0]) if eigvals.size >= 2 else 0.0

        kirchhoff = (
            float(3.0 * (1.0 / eigvals[1] + 1.0 / eigvals[2]))
            if fiedler > self._thresholds.fiedler_min
            else math.inf
        )
        dirichlet = max(0.0, float(np.trace(coords.T @ laplacian @ coords)))
        return fiedler, gap, kirchhoff, dirichlet

    def _evaluate_heyting_predicates(
        self,
        triad: OctonionicTriadReport,
        comp_err: float,
        hard_comp: bool,
        asoc_stable: bool,
        moufang_max: float,
        fiedler: float,
        kirchhoff: float,
        dirichlet: float,
    ) -> Tuple[Tuple[str, HeytingVerdict, str], ...]:
        r"""
        Evalúa los predicados locales nombrados en \Omega_3, con razones
        auditables — arquitectura consistente con el agente soberano.
        """
        thr = self._thresholds

        if hard_comp or not math.isfinite(comp_err):
            p_hurwitz, r_hurwitz = HeytingVerdict.VETOED, "Ruptura catastrófica de composición de Hurwitz."
        elif comp_err > thr.hurwitz_tolerance:
            p_hurwitz, r_hurwitz = HeytingVerdict.DEGRADED, "Deriva de Hurwitz por encima de la tolerancia nominal."
        else:
            p_hurwitz, r_hurwitz = HeytingVerdict.COHERENT, ""

        if triad.artin_residual > thr.artin_tolerance:
            p_artin, r_artin = HeytingVerdict.VETOED, "Ruptura de Artin (alternatividad/flexibilidad) más allá del ruido esperado."
        else:
            p_artin, r_artin = HeytingVerdict.COHERENT, ""

        if not math.isfinite(triad.associator_norm):
            p_asoc, r_asoc = HeytingVerdict.VETOED, "Asociador trilateral no finito: singularidad matemática."
        elif not asoc_stable:
            p_asoc, r_asoc = HeytingVerdict.DEGRADED, "Curvatura de calibre (asociador) fuera de la banda nominal."
        else:
            p_asoc, r_asoc = HeytingVerdict.COHERENT, ""

        if moufang_max > thr.moufang_tolerance:
            p_moufang, r_moufang = HeytingVerdict.DEGRADED, "Identidades de Moufang degradadas por deriva FPU."
        else:
            p_moufang, r_moufang = HeytingVerdict.COHERENT, ""

        if fiedler < thr.fiedler_min:
            p_hodge, r_hodge = HeytingVerdict.DEGRADED, "Estrangulamiento espectral de Fiedler en K_3."
        else:
            p_hodge, r_hodge = HeytingVerdict.COHERENT, ""

        if kirchhoff > thr.kirchhoff_max or dirichlet > thr.dirichlet_max:
            p_exergy, r_exergy = HeytingVerdict.DEGRADED, "Disipación exergética o resistencia de Kirchhoff anómalas."
        else:
            p_exergy, r_exergy = HeytingVerdict.COHERENT, ""

        return (
            ("hurwitz_composition", p_hurwitz, r_hurwitz),
            ("artin_alternativity_flexibility", p_artin, r_artin),
            ("trilateral_associator", p_asoc, r_asoc),
            ("moufang_identities", p_moufang, r_moufang),
            ("hodge_fiedler_connectivity", p_hodge, r_hodge),
            ("kirchhoff_dirichlet_exergy", p_exergy, r_exergy),
        )

    def observe_octonionic_triad(
        self,
        triad: OctonionicTriadReport,
        asoc_threshold: float = 0.15,
    ) -> OctonionicOrientationReport:
        """Orquesta la observación analítica sobre la tríada, sin recomputar productos base."""
        a, b, c = triad.contractor, triad.supplier, triad.interventor

        err_ab, rel_ab, _ = self.compute_hurwitz_error(a, b, precomputed_product=triad.product_ab)
        err_bc, rel_bc, _ = self.compute_hurwitz_error(b, c, precomputed_product=triad.product_bc)
        err_ca, rel_ca, ca_state = self.compute_hurwitz_error(c, a)  # producto fresco necesario

        comp_err = max(err_ab, err_bc, err_ca)
        comp_rel = max(rel_ab, rel_bc, rel_ca)
        hard_comp = bool(comp_err > 1e-6 or not math.isfinite(comp_err))

        self._enforce_axiom(
            comp_err, self._thresholds.hurwitz_tolerance,
            "Ley de composición de Hurwitz", OctonionicCompositionError,
        )

        denom = a.norm * b.norm * c.norm
        asoc_rel = triad.associator_norm / max(denom, _WILKINSON_FLOOR)
        is_asoc_stable = bool(triad.associator_norm <= asoc_threshold + self._tol)

        moufang_left, moufang_right, moufang_central = OctonionicAlgebraKernel.moufang_defects(
            a.vector_rep, b.vector_rep, c.vector_rep,
            ab=triad.product_ab.vector_rep, bc=triad.product_bc.vector_rep,
        )
        moufang_scale = [(a.norm, 2), (b.norm, 1), (c.norm, 1)]
        moufang_rel_max = max(
            self._relative_defect(moufang_left, *moufang_scale),
            self._relative_defect(moufang_right, *moufang_scale),
            self._relative_defect(moufang_central, *moufang_scale),
        )
        moufang_raw_max = max(moufang_left, moufang_right, moufang_central)
        self._enforce_axiom(
            moufang_raw_max, self._thresholds.moufang_tolerance,
            "Identidades de Moufang", OctonionicArtinFailureError,
        )

        fiedler, gap, kirchhoff, dirichlet = self._diagnose_hodge_laplacian_k3((a, b, c))
        is_cfl_stable = bool(comp_err <= self._thresholds.hurwitz_tolerance)

        predicates = self._evaluate_heyting_predicates(
            triad, comp_err, hard_comp, is_asoc_stable, moufang_raw_max, fiedler, kirchhoff, dirichlet
        )
        omega_pre = _heyting_meet_all(p[1] for p in predicates)
        conjuncts = tuple((name, val.value) for name, val, _ in predicates)
        reasons = tuple(r for _, _, r in predicates if r)

        sha_hash = self._sha256_payload(
            triad.sha256_hash,
            np.array([comp_err, asoc_rel, moufang_raw_max, fiedler, kirchhoff, dirichlet, omega_pre.omega]),
        )

        return OctonionicOrientationReport(
            triad=triad,
            composition_error=comp_err, composition_relative_error=comp_rel,
            associator_norm=triad.associator_norm, associator_relative_norm=asoc_rel,
            is_cfl_stable=is_cfl_stable, is_associative_stable=is_asoc_stable, hard_composition=hard_comp,
            artin_residual=triad.artin_residual, artin_relative_residual=triad.artin_relative_residual,
            moufang_left_defect=moufang_left, moufang_right_defect=moufang_right,
            moufang_central_defect=moufang_central, moufang_relative_max=moufang_rel_max,
            laplacian_connectivity=fiedler, laplacian_spectral_gap=gap,
            kirchhoff_index=kirchhoff, dirichlet_exergy=dirichlet,
            omega_pre=omega_pre, conjuncts=conjuncts, reasons=reasons,
            sha256_hash=sha_hash,
            contractor_norm=a.norm, supplier_norm=b.norm, interventor_norm=c.norm,
        )

    def orient_octonionic_diagnostics(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
        asoc_threshold: float = 0.15,
        triad: Optional[OctonionicTriadReport] = None,
    ) -> OctonionicOrientationReport:
        r"""
        MORFISMO TERMINAL DE LA FASE 2 / GERMEN DE LA FASE 3.
        Firma: (\mathbb{R}^8)^3 \longrightarrow \mathbf{OctonionicOrientationReport}.
        """
        if triad is None:
            triad = self.synthesize_octonionic_triad(contractor_S, supplier_S, interventor_S)
        return self.continue_from_triad_report(triad=triad, asoc_threshold=asoc_threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# §H. FASE 3 — DECIDE + ACT: Ω_3 PURO, HMAC ATADO A CONTEXTO, CROWBAR RC, SELLO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class OctonionicGraceState:
    r"""
    Value-type inmutable que porta el estado modal de la ventana de gracia
    \Gamma(t) como MÓNADA DE ESTADO EXPLÍCITA, eliminando la mutación de
    instancia previa (inseguro en concurrencia) — análogo a
    `PathionicGraceState` del agente soberano.
    """

    is_active: bool = False
    activated_at_monotonic: Optional[float] = None

    def with_activation(self, t_now: float) -> "OctonionicGraceState":
        return OctonionicGraceState(is_active=True, activated_at_monotonic=float(t_now))

    def cleared(self) -> "OctonionicGraceState":
        return OctonionicGraceState(is_active=False, activated_at_monotonic=None)

    def elapsed_since(self, t_now: float) -> float:
        if not self.is_active or self.activated_at_monotonic is None:
            return math.inf
        return max(0.0, float(t_now) - self.activated_at_monotonic)


@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""Informe físico RC-descompuesto de la actuación del interlock Crowbar BT151."""

    interlock_fired: bool
    actuation_latency_ns: float
    gpio: str
    device: str
    seed_sha256: str
    gate_charge_injected_nc: float
    t_isr_dispatch_ns: float = 0.0
    t_gpio_propagation_ns: float = 0.0
    t_gate_charge_ns: float = 0.0


@dataclass(frozen=True, slots=True)
class OctonionicAuditCertificate:
    r"""CERTIFICADO TERMINAL INMUTABLE DE GOBERNANZA TRILATERAL."""

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
    artin_relative_residual: float = 0.0
    moufang_left_defect: float = 0.0
    moufang_right_defect: float = 0.0
    moufang_central_defect: float = 0.0
    malcev_residual: float = 0.0
    fano_3form_value: float = 0.0
    laplacian_connectivity: float = 0.0
    kirchhoff_index: float = 0.0
    dirichlet_exergy: float = 0.0
    reasons: Tuple[str, ...] = ()
    conjuncts: Tuple[Tuple[str, str], ...] = ()
    t_isr_dispatch_ns: float = 0.0
    t_gpio_propagation_ns: float = 0.0
    t_gate_charge_ns: float = 0.0
    grace_is_active: bool = False
    triad_seal: str = ""
    engine_version: str = __version__


class Phase3_OODAActuator(Phase2_OctonionicDiagnostics):
    r"""
    FASE 3: Decide + Act.
    Hereda ontológicamente de la Fase 2.

    Categoría Functorial:
      \mathbf{OctonionicOrientationReport} \longrightarrow \mathbf{OctonionicAuditCertificate}.

    Responsabilidades axiomáticas:
      1. Morfismo de inicio `continue_from_orientation`.
      2. Resolución en \Omega_3 mediante MORFISMO DE KLEISLI PURO
         `decide_from_orientation`: \mathrm{Orientation} \times \Gamma \to
         (V, \Gamma'), con envoltorio *stateful* thread-safe opcional.
      3. Override HMAC ATADO AL CONTEXTO DECISIONAL (hash de orientación +
         conjuntos evaluados), reparando el esquema previamente roto que
         autenticaba el token contra sí mismo.
      4. Interlock Crowbar BT151 mediante MODELO RC-DESCOMPUESTO en 3
         subsistemas físicos (ISR Xtensa, propagación GPIO/APB, carga de
         compuerta), consistente con el agente soberano.
      5. Sellado criptográfico con *framing* de longitud (`_frame_bytes`).

    Morfismo Terminal Global: `audit_trilateral_cycle`.
    """

    __slots__ = (
        "_grace_max", "_default_asoc_threshold", "_grace_state", "_grace_lock",
        "_override_verifier", "_allowed_override_tokens", "_hmac_secret",
    )

    def __init__(
        self,
        tolerance: float = 1e-12,
        grace_period_seconds: float = 3600.0,
        asoc_threshold: float = 0.15,
        strict_axioms: bool = True,
        override_verifier: Optional[Callable[[str, str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        super().__init__(tolerance=tolerance, strict_axioms=strict_axioms)
        self._grace_max: Final[float] = float(grace_period_seconds)
        self._default_asoc_threshold: Final[float] = float(asoc_threshold)
        self._grace_state: OctonionicGraceState = OctonionicGraceState()
        self._grace_lock: Final[threading.RLock] = threading.RLock()
        self._override_verifier: Optional[Callable[[str, str], bool]] = override_verifier

        tokens = allowed_override_tokens
        if tokens is None:
            tokens = _LEGACY_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        self._allowed_override_tokens: Final[frozenset[str]] = frozenset(tokens)

        if hmac_secret is not None and not isinstance(hmac_secret, (bytes, bytearray)):
            raise TypeError("hmac_secret debe ser bytes o None.")
        self._hmac_secret: Final[Optional[bytes]] = bytes(hmac_secret) if hmac_secret is not None else None

    def reset_soft_veto(self) -> None:
        """Restablece (thread-safe) la ventana de gracia modal Γ."""
        with self._grace_lock:
            self._grace_state = OctonionicGraceState()

    @property
    def current_grace_state(self) -> OctonionicGraceState:
        with self._grace_lock:
            return self._grace_state

    def _compute_override_context_hash(self, orientation: OctonionicOrientationReport) -> str:
        r"""
        Ata el token de override al CONTEXTO DECISIONAL completo (hash de
        orientación + conjuntos evaluados), reparando el esquema previo que
        autenticaba el token contra una HMAC de sí mismo.
        """
        hasher = hashlib.sha256()
        hasher.update(_frame_bytes(b"OCTO_OVERRIDE_ORIENTATION", orientation.sha256_hash.encode("ascii")))
        for name, verdict in orientation.conjuncts:
            hasher.update(_frame_bytes(name.encode("ascii"), verdict.encode("ascii")))
        return hasher.hexdigest()

    def _verify_override(self, token: Optional[str], context_hash: str) -> bool:
        """Autenticación en tiempo constante del override, atada al contexto decisional."""
        if not token or not isinstance(token, str):
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token, context_hash))
            except Exception:
                logger.exception("Error en override_verifier.")
                return False

        token_b = token.encode("utf-8")
        for allowed in self._allowed_override_tokens:
            if hmac.compare_digest(token_b, allowed.encode("utf-8")):
                return True

        if self._hmac_secret is not None and context_hash:
            expected = hmac.new(self._hmac_secret, context_hash.encode("ascii"), hashlib.sha256).hexdigest()
            if hmac.compare_digest(token, expected):
                return True

        return False

    def decide_from_orientation(
        self,
        orientation: OctonionicOrientationReport,
        grace_state: OctonionicGraceState,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
    ) -> Tuple[HeytingVerdict, bool, bool, float, OctonionicGraceState]:
        r"""
        MORFISMO DE KLEISLI PURO:
          \mathrm{Orientation} \times \Gamma \to (V, \text{soft}, \text{hard}, t_{\text{rem}}, \Gamma').
        Sin efectos laterales de instancia.
        """
        now = time.monotonic() if curr_time is None else float(curr_time)
        verdict = orientation.omega_pre

        if verdict is HeytingVerdict.VETOED:
            return HeytingVerdict.VETOED, False, True, 0.0, grace_state.cleared()

        if verdict is HeytingVerdict.COHERENT:
            return HeytingVerdict.COHERENT, False, False, 0.0, grace_state.cleared()

        # Régimen DEGRADED
        context_hash = self._compute_override_context_hash(orientation)
        if override_token is not None and self._verify_override(override_token, context_hash):
            return HeytingVerdict.DEGRADED, False, False, 0.0, grace_state.cleared()

        if not grace_state.is_active:
            return HeytingVerdict.DEGRADED, True, False, self._grace_max, grace_state.with_activation(now)

        elapsed = grace_state.elapsed_since(now)
        time_remaining = max(0.0, self._grace_max - elapsed) if math.isfinite(elapsed) else 0.0

        if time_remaining <= self._tol:
            return HeytingVerdict.VETOED, False, True, 0.0, grace_state.cleared()

        return HeytingVerdict.DEGRADED, True, False, time_remaining, grace_state

    def _decide_stateful(
        self,
        orientation: OctonionicOrientationReport,
        override_token: Optional[str],
        curr_time: Optional[float],
    ) -> Tuple[HeytingVerdict, bool, bool, float]:
        """Envoltorio *stateful* thread-safe del morfismo puro `decide_from_orientation`."""
        with self._grace_lock:
            verdict, soft, hard, rem, new_state = self.decide_from_orientation(
                orientation, self._grace_state, override_token, curr_time
            )
            self._grace_state = new_state
            return verdict, soft, hard, rem

    def _model_crowbar_latency_ns(self, digest: bytes) -> Tuple[float, float, float, float]:
        r"""Modelo RC-descompuesto de latencia de interlock (idéntico en espíritu al agente soberano)."""
        f_isr = int.from_bytes(digest[0:2], "big") / 65535.0
        f_gpio = int.from_bytes(digest[2:4], "big") / 65535.0
        f_gate = int.from_bytes(digest[4:6], "big") / 65535.0

        n_cycles = _ISR_NOMINAL_CYCLES - _ISR_CYCLE_JITTER + 2.0 * _ISR_CYCLE_JITTER * f_isr
        t_isr = n_cycles * _ISR_CYCLE_TIME_NS

        slew_lo, slew_hi = _GPIO_PAD_SLEW_NS_RANGE
        t_gpio = _APB_CYCLE_TIME_NS + (slew_lo + (slew_hi - slew_lo) * f_gpio)

        v_supply = _CROWBAR_V_SUPPLY_NOMINAL * (1.0 + _CROWBAR_V_SUPPLY_TOLERANCE * (2.0 * f_gate - 1.0))
        rc_ns = _CROWBAR_R_GK_OHM * _CROWBAR_C_GK_FARAD * 1.0e9
        t_rc = rc_ns * math.log(v_supply / (v_supply - _CROWBAR_V_GT_VOLTS))

        total = t_isr + t_gpio + t_rc
        if not math.isfinite(total) or total >= _CROWBAR_IRAM_LATENCY_NS_LIMIT:
            raise OctonionicEngineError(
                f"Presupuesto de latencia ISR excedido: {total:.3f} ns >= {_CROWBAR_IRAM_LATENCY_NS_LIMIT} ns."
            )
        return float(t_isr), float(t_gpio), float(t_rc), float(total)

    def _act_crowbar(self, orientation: OctonionicOrientationReport, verdict: HeytingVerdict) -> CrowbarActuationReport:
        """Simula el interlock físico Crowbar BT151 mediante el modelo RC-descompuesto."""
        if verdict is not HeytingVerdict.VETOED:
            return CrowbarActuationReport(
                interlock_fired=False, actuation_latency_ns=0.0, gpio=_CROWBAR_GPIO,
                device=_CROWBAR_DEVICE, seed_sha256="", gate_charge_injected_nc=0.0,
            )

        sha = hashlib.sha256()
        sha.update(_frame_bytes(b"BT151_CROWBAR_OCTONIONIC_RC", orientation.sha256_hash.encode("ascii")))
        digest = sha.digest()

        t_isr, t_gpio, t_rc, latency = self._model_crowbar_latency_ns(digest)
        f_gate = int.from_bytes(digest[4:6], "big") / 65535.0
        v_supply_est = _CROWBAR_V_SUPPLY_NOMINAL * (1.0 + _CROWBAR_V_SUPPLY_TOLERANCE * (2.0 * f_gate - 1.0))
        gate_charge_nc = _CROWBAR_C_GK_FARAD * v_supply_est * 1.0e9

        logger.critical("╔══════════════════════════════════════════════════════════════╗")
        logger.critical("║ ¡INTERLOCK CROWBAR BT151 DISPARADO POR COLAPSO DE HEYTING!   ║")
        logger.critical(
            "║ - Latencia RC : t_isr=%.2f + t_gpio=%.2f + t_rc=%.2f = %.2f ns  ║", t_isr, t_gpio, t_rc, latency
        )
        logger.critical("║ - Protocolo   : Cortocircuito a tierra de bus trilateral.    ║")
        logger.critical("╚══════════════════════════════════════════════════════════════╝")

        return CrowbarActuationReport(
            interlock_fired=True, actuation_latency_ns=latency, gpio=_CROWBAR_GPIO,
            device=_CROWBAR_DEVICE, seed_sha256=digest.hex(), gate_charge_injected_nc=gate_charge_nc,
            t_isr_dispatch_ns=t_isr, t_gpio_propagation_ns=t_gpio, t_gate_charge_ns=t_rc,
        )

    def continue_from_orientation(
        self,
        orientation: OctonionicOrientationReport,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
        grace_state: Optional[OctonionicGraceState] = None,
    ) -> OctonionicAuditCertificate:
        r"""
        MORFISMO DE CONTINUACIÓN DE LA FASE 3.
        Si `grace_state` es `None`, usa el modo *stateful* interno thread-safe;
        si se provee, opera en modo PURO/FUNCIONAL.
        """
        if grace_state is not None:
            verdict, is_soft, is_hard, time_rem, _new_grace = self.decide_from_orientation(
                orientation, grace_state, override_token, curr_time
            )
        else:
            verdict, is_soft, is_hard, time_rem = self._decide_stateful(orientation, override_token, curr_time)

        actuation = self._act_crowbar(orientation, verdict)

        hasher = hashlib.sha256()
        hasher.update(_frame_bytes(b"OCTO_CERT_HEADER", (__version__ + "|" + verdict.value).encode("utf-8")))
        hasher.update(_frame_bytes(b"TRIAD_SEAL", orientation.triad.sha256_hash.encode("ascii")))
        hasher.update(_frame_bytes(b"ORIENT_SEAL", orientation.sha256_hash.encode("ascii")))
        hasher.update(_frame_bytes(b"ACTUATION_SEED", actuation.seed_sha256.encode("ascii")))

        scalar_chain: Tuple[Tuple[str, float], ...] = (
            ("comp_err", orientation.composition_error),
            ("comp_rel", orientation.composition_relative_error),
            ("asoc_norm", orientation.associator_norm),
            ("asoc_rel", orientation.associator_relative_norm),
            ("artin_res", orientation.artin_residual),
            ("artin_rel", orientation.artin_relative_residual),
            ("moufang_left", orientation.moufang_left_defect),
            ("moufang_right", orientation.moufang_right_defect),
            ("moufang_central", orientation.moufang_central_defect),
            ("malcev_res", orientation.triad.malcev_residual),
            ("fano_3form", orientation.triad.fano_3form_value),
            ("fiedler", orientation.laplacian_connectivity),
            ("kirchhoff", orientation.kirchhoff_index),
            ("dirichlet", orientation.dirichlet_exergy),
            ("omega_pre", orientation.omega_pre.omega),
            ("verdict_omega", verdict.omega),
            ("latency", actuation.actuation_latency_ns),
            ("t_isr", actuation.t_isr_dispatch_ns),
            ("t_gpio", actuation.t_gpio_propagation_ns),
            ("t_rc", actuation.t_gate_charge_ns),
            ("grace_remaining", time_rem),
        )
        for name, scalar in scalar_chain:
            hasher.update(_frame_bytes(name.encode("ascii"), _canonical_bytes(scalar)))
        for name, val in orientation.conjuncts:
            hasher.update(_frame_bytes(f"CONJUNCT_{name}".encode("ascii"), val.encode("ascii")))

        cryptographic_seal = hasher.hexdigest()

        return OctonionicAuditCertificate(
            heyting_verdict=verdict.value,
            associator_norm=orientation.associator_norm,
            is_associative_stable=orientation.is_associative_stable,
            composition_error=orientation.composition_error,
            is_cfl_stable=orientation.is_cfl_stable,
            is_soft_veto_active=is_soft, is_hard_veto_active=is_hard,
            actuation_latency_ns=actuation.actuation_latency_ns, time_grace_remaining=time_rem,
            cryptographic_seal=cryptographic_seal,
            composition_relative_error=orientation.composition_relative_error,
            associator_relative_norm=orientation.associator_relative_norm,
            contractor_norm=orientation.contractor_norm, supplier_norm=orientation.supplier_norm,
            interventor_norm=orientation.interventor_norm,
            heyting_omega=verdict.omega, omega_pre=orientation.omega_pre.omega,
            artin_residual=orientation.artin_residual, artin_relative_residual=orientation.artin_relative_residual,
            moufang_left_defect=orientation.moufang_left_defect,
            moufang_right_defect=orientation.moufang_right_defect,
            moufang_central_defect=orientation.moufang_central_defect,
            malcev_residual=orientation.triad.malcev_residual,
            fano_3form_value=orientation.triad.fano_3form_value,
            laplacian_connectivity=orientation.laplacian_connectivity,
            kirchhoff_index=orientation.kirchhoff_index, dirichlet_exergy=orientation.dirichlet_exergy,
            reasons=orientation.reasons, conjuncts=orientation.conjuncts,
            t_isr_dispatch_ns=actuation.t_isr_dispatch_ns,
            t_gpio_propagation_ns=actuation.t_gpio_propagation_ns,
            t_gate_charge_ns=actuation.t_gate_charge_ns,
            grace_is_active=is_soft,
            triad_seal=orientation.triad.sha256_hash,
            engine_version=__version__,
        )

    def audit_trilateral_cycle(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
        asoc_threshold: Optional[float] = None,
        override_token: Optional[str] = None,
        grace_state: Optional[OctonionicGraceState] = None,
    ) -> OctonionicAuditCertificate:
        r"""
        MORFISMO TERMINAL GLOBAL DEL RESOLVER OCTONIÓNICO.

        Ejecuta el ciclo covariante OODA trilateral en \mathbb{O}:
          (S_1, S_2, S_3)
          \xrightarrow{\text{Fase 1: Observe}} \mathbf{TriadReport}
          \xrightarrow{\text{Fase 2: Orient}} \mathbf{OrientationReport}
          \xrightarrow{\text{Fase 3: Decide+Act}} \mathbf{AuditCertificate}.

        `asoc_threshold=None` reutiliza el umbral configurado en el constructor
        (reparación del campo muerto `_asoc_limit` de v3.1.0).
        """
        effective_asoc_threshold = (
            self._default_asoc_threshold if asoc_threshold is None else float(asoc_threshold)
        )

        triad = self.synthesize_octonionic_triad(contractor_S, supplier_S, interventor_S)
        orientation = self.continue_from_triad_report(triad=triad, asoc_threshold=effective_asoc_threshold)
        certificate = self.continue_from_orientation(
            orientation=orientation, override_token=override_token, grace_state=grace_state
        )

        logger.debug(
            "Ciclo Trilateral 8D auditado | Veredicto: %s | Asoc: %.4e | Sello: %s",
            certificate.heyting_verdict, certificate.associator_norm, certificate.cryptographic_seal[:12],
        )
        return certificate


# ═══════════════════════════════════════════════════════════════════════════════
# §I. FACHADA SOBERANA PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════════
# Anidación Ontológica Estricta: Phase3 ⊏ Phase2 ⊏ Phase1.
OctonionicDependencyResolver = Phase3_OODAActuator

__all__ = [
    "OctonionicDependencyResolver",
    "OctonionicAlgebraKernel",
    "OctonionicState",
    "OctonionicTriadReport",
    "OctonionicOrientationReport",
    "OctonionicAuditCertificate",
    "OctonionicGraceState",
    "OctonionicThresholds",
    "BanachRegularityReport",
    "CrowbarActuationReport",
    "HeytingVerdict",
    "Phase1_OctonionicAlgebraKernel",
    "Phase2_OctonionicDiagnostics",
    "Phase3_OODAActuator",
    "OctonionicEngineError",
    "OctonionicDimensionError",
    "OctonionicNumericalSingularityError",
    "OctonionicCompositionError",
    "OctonionicArtinFailureError",
    "OctonionicFanoStructureError",
    "OctonionicHeytingAxiomError",
]