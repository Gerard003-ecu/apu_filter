# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Photinic Governance Satellite Engine                                ║
║ Ruta   : app/core/inmune_system/photinic_governance_satellite_engine.py      ║
║ Versión: 6.0.0-Validated-Channel-Bell-State-HMAC                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Motor de auditoría federada organizado en tres fases OODA:

  Fase 1 — Observe
    Valida y copia las políticas reales; calcula el operador de Gram federado,
    comprueba su simetría y positividad dentro de tolerancia, evalúa cocientes
    de normas y emite un expediente autenticado con HMAC.

  Fase 2 — Orient
    Calcula el proyector ortogonal sobre el espacio fila de las políticas,
    audita una familia explícita de operadores de Kraus y, opcionalmente,
    analiza un estado de dos qubits suministrado por el consumidor.

  Fase 3 — Act
    Comprueba las certificaciones recibidas, calcula el operador de Gram
    normalizado y su entropía espectral, y emite un certificado HMAC.

Precisiones de interpretación:
  • sqrt(Tr(S²)) y Tr(S)/||S||_F son invariantes del operador de Gram. Se
    conservan nombres históricos que contienen "Majorana", pero no representan
    por sí solos un espinor de Majorana ni una cantidad de N=1 SYM.
  • Los operadores K_k = P_k/sqrt(N) definidos aquí producen una operación
    completamente positiva. El chequeo sum K_k†K_k <= I certifica que es
    trace-nonincreasing; no certifica trace-preserving. La opción
    require_trace_preserving permite exigir TP para aceptar el resultado.
  • Un chequeo local de completitud de Kraus no demuestra no-señalización.
    is_causal_non_signaling permanece None porque las entradas no especifican
    una operación bipartita ni sus marginales.
  • El criterio de Horodecki se aplica solo si se proporciona una matriz de
    densidad válida de dos qubits (4x4). No se inventa una bipartición a partir
    de vectores de políticas.
  • La entropía de la matriz de Gram normalizada es una entropía espectral de
    ese operador. Sin una interpretación cuántica adicional no se afirma que
    sea la entropía física de un estado cuántico.
  • La entropía de entrelazamiento se devuelve solo cuando se suministra un
    estado bipartito puro; para estados mixtos no se infiere de los autovalores
    de TᵀT.
  • Sin clave HMAC inyectada se genera una clave aleatoria efímera. Ese modo es
    apropiado para desarrollo dentro de una instancia, pero no permite verificar
    firmas entre procesos o reinicios. En producción, inyectar una clave secreta
    gestionada externamente.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import secrets
import struct
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Final, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger("APU.Physics.PhotinicGovernanceSatelliteEngine")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y LÍMITES NUMÉRICOS
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_FLOAT_TINY: Final[float] = float(np.finfo(np.float64).tiny)

_DEFAULT_TOLERANCE: Final[float] = 1.0e-12
_DEFAULT_GRAM_REGULARIZATION: Final[float] = 1.0e-12
_GRAM_RECONSTRUCTION_TOLERANCE: Final[float] = 1.0e-9
_CHOІ_PSD_TOLERANCE: Final[float] = 1.0e-9
_CHANNEL_TNI_TOLERANCE: Final[float] = 1.0e-9
_BELL_STATE_TOLERANCE: Final[float] = 1.0e-10
_TSIRELSON_TOLERANCE: Final[float] = 1.0e-9

_TSIRELSON_BOUND: Final[float] = 2.0 * math.sqrt(2.0)
_BELL_CLASSICAL_BOUND: Final[float] = 2.0
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_VELTKAMP_SPLIT_CONSTANT: Final[float] = 134217729.0  # 2**27 + 1

_MAX_EXPLICIT_CHOI_DIMENSION: Final[int] = 16
_MIN_HMAC_KEY_BYTES: Final[int] = 32

# Nombre legado para compatibilidad con consumidores anteriores. No se usa
# como sustituto de una tolerancia relativa dependiente de escala.
_WILKINSON_FLOOR: Final[float] = 1.0e-15


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════

class PhotinicEngineError(Exception):
    """Excepción raíz del motor fotínico."""


class PhotinicInputError(PhotinicEngineError):
    """Entrada mal formada, no real o no finita."""


class PhotinicDimensionError(PhotinicEngineError):
    """Dimensiones incompatibles entre políticas y operadores."""


class PhotinicNumericalError(PhotinicEngineError):
    """Un cálculo no puede certificarse dentro del rango numérico disponible."""


class PhotinicIntegrityError(PhotinicEngineError):
    """Un expediente o certificado no supera la verificación HMAC."""


class PhotinicNonPositiveDefiniteError(PhotinicEngineError):
    """Violación de positividad del Gram, de una densidad o del Choi auditado."""


class PhotinicCausalViolationError(PhotinicEngineError):
    """Excepción raíz para fallos de admisibilidad de una operación."""


class PhotinicNonCausalChannelError(PhotinicCausalViolationError):
    """
    Nombre histórico conservado.

    La condición comprobada es sum_k K_k† K_k <= I (trace-nonincreasing);
    no es, por sí sola, una prueba de no-señalización relativista.
    """


class PhotinicTracePreservationError(PhotinicCausalViolationError):
    """La operación aceptada no satisface el requisito explícito de TP."""


class PhotinicTsirelsonViolationError(PhotinicCausalViolationError):
    """El resultado de CHSH supera la cota configurada para un estado cuántico."""


class PhotinicInvalidQuantumStateError(PhotinicInputError):
    """La matriz proporcionada no es una matriz de densidad válida de dos qubits."""


class PhotinicProjectorError(PhotinicNumericalError):
    """No se pudo certificar el proyector ortogonal calculado."""


# ══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE ESCALARES Y ARREGLOS
# ══════════════════════════════════════════════════════════════════════════════

def _finite_float(value: Any, name: str) -> float:
    """Convierte a float y rechaza booleanos, NaN e infinitos."""
    if isinstance(value, (bool, np.bool_)):
        raise PhotinicInputError(f"{name} debe ser un escalar real, no booleano.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhotinicInputError(f"{name} no es un escalar real válido.") from exc
    if not math.isfinite(result):
        raise PhotinicInputError(f"{name} debe ser finito.")
    return result


def _nonnegative_float(
    value: Any,
    name: str,
    *,
    allow_positive_infinity: bool = False,
) -> float:
    """Valida un escalar no negativo, permitiendo opcionalmente +inf."""
    if isinstance(value, (bool, np.bool_)):
        raise PhotinicInputError(f"{name} debe ser numérico, no booleano.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhotinicInputError(f"{name} no es numérico.") from exc

    if math.isnan(result) or result < 0.0:
        raise PhotinicInputError(f"{name} debe ser no negativo y no NaN.")
    if math.isinf(result) and not (
        allow_positive_infinity and result > 0.0
    ):
        raise PhotinicInputError(f"{name} debe ser finito.")
    return result


def _readonly_real_array(value: Any, name: str) -> NDArray[np.float64]:
    """
    Copia a float64 real, valida finitud y devuelve un arreglo de solo lectura.

    No acepta booleanos, complejos, cadenas ni arreglos de objetos.
    """
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PhotinicInputError(f"{name} no es convertible a arreglo.") from exc

    if raw.dtype.kind not in "fiu":
        raise PhotinicInputError(
            f"{name} debe tener dtype real numérico; recibido {raw.dtype}."
        )

    try:
        result = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhotinicInputError(f"{name} no puede convertirse a float64.") from exc

    if not np.all(np.isfinite(result)):
        raise PhotinicInputError(f"{name} contiene NaN o infinitos.")

    result[result == 0.0] = 0.0  # Canoniza -0.0 como +0.0.
    result.setflags(write=False)
    return result


def _readonly_complex_array(
    value: Any,
    name: str,
) -> NDArray[np.complex128]:
    """Copia a complex128 y valida finitud de partes real e imaginaria."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PhotinicInputError(f"{name} no es convertible a arreglo.") from exc

    if raw.dtype.kind not in "fiuc":
        raise PhotinicInputError(
            f"{name} debe ser numérico real o complejo; recibido {raw.dtype}."
        )

    try:
        result = np.array(raw, dtype=np.complex128, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhotinicInputError(f"{name} no puede convertirse a complex128.") from exc

    if not np.all(np.isfinite(result)):
        raise PhotinicInputError(f"{name} contiene NaN o infinitos.")

    result[result == 0.0] = 0.0 + 0.0j
    result.setflags(write=False)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SUMA COMPENSADA Y PRODUCTO PUNTO DEKKER
# ══════════════════════════════════════════════════════════════════════════════

class KahanNeumaierSum:
    r"""
    Suma de Neumaier y producto punto con transformación de producto de Dekker.

    La compensación reduce el error de redondeo en el dominio float64. No
    convierte las operaciones en aritmética exacta para entradas arbitrarias:
    ante overflow se falla explícitamente; en subnormales, el residuo del
    producto puede no ser representable en float64.
    """

    @staticmethod
    def sum(values: Sequence[float]) -> float:
        """Suma compensada de Neumaier; falla si un intermedio desborda."""
        total = 0.0
        correction = 0.0

        for raw_value in values:
            value = _finite_float(raw_value, "sumando")
            updated = total + value
            if not math.isfinite(updated):
                raise PhotinicNumericalError(
                    "Overflow durante la suma compensada de Neumaier."
                )

            if abs(total) >= abs(value):
                correction += (total - updated) + value
            else:
                correction += (value - updated) + total

            if not math.isfinite(correction):
                raise PhotinicNumericalError(
                    "Overflow en la corrección de Neumaier."
                )
            total = updated

        result = total + correction
        if not math.isfinite(result):
            raise PhotinicNumericalError(
                "El resultado de la suma compensada no es finito."
            )
        return float(result)

    @staticmethod
    def _veltkamp_split(value: float) -> Tuple[float, float]:
        """Separa un float en componentes alta y baja evitando splitter*value."""
        mantissa, exponent = math.frexp(value)
        product = _VELTKAMP_SPLIT_CONSTANT * mantissa
        high_mantissa = product - (product - mantissa)
        low_mantissa = mantissa - high_mantissa

        try:
            high = math.ldexp(high_mantissa, exponent)
            low = math.ldexp(low_mantissa, exponent)
        except OverflowError as exc:
            raise PhotinicNumericalError(
                "Overflow en la separación de Veltkamp."
            ) from exc

        if not math.isfinite(high) or not math.isfinite(low):
            raise PhotinicNumericalError(
                "La separación de Veltkamp produjo un valor no finito."
            )
        return high, low

    @staticmethod
    def _fraction_product_residual(a: float, b: float, rounded_product: float) -> float:
        """Ruta de respaldo exacta en racionales para casos extremos de Dekker."""
        exact_residual = (
            Fraction.from_float(a) * Fraction.from_float(b)
            - Fraction.from_float(rounded_product)
        )
        try:
            residual = float(exact_residual)
        except (OverflowError, ValueError) as exc:
            raise PhotinicNumericalError(
                "El residuo exacto del producto no cabe en float64."
            ) from exc
        if not math.isfinite(residual):
            raise PhotinicNumericalError(
                "El residuo del producto no es finito."
            )
        return residual

    @staticmethod
    def two_product_dekker(a: float, b: float) -> Tuple[float, float]:
        r"""
        Devuelve producto redondeado y residuo calculado por Dekker.

        Para resultados normales y sin overflow, el residuo corrige la parte
        baja del producto. Se usa una ruta racional en casos subnormales o si
        los intermedios de la separación no son representables.
        """
        a = _finite_float(a, "factor a")
        b = _finite_float(b, "factor b")
        product = a * b

        if not math.isfinite(product):
            raise PhotinicNumericalError("Overflow en el producto a*b.")
        if a == 0.0 or b == 0.0:
            return product, 0.0

        fma = getattr(math, "fma", None)
        if fma is not None:
            residual = fma(a, b, -product)
            if math.isfinite(residual):
                return product, residual

        if (
            abs(a) < _FLOAT_TINY
            or abs(b) < _FLOAT_TINY
            or abs(product) < _FLOAT_TINY
        ):
            return product, KahanNeumaierSum._fraction_product_residual(
                a, b, product
            )

        try:
            a_high, a_low = KahanNeumaierSum._veltkamp_split(a)
            b_high, b_low = KahanNeumaierSum._veltkamp_split(b)

            term_0 = a_high * b_high - product
            term_1 = a_high * b_low
            term_2 = a_low * b_high
            term_3 = a_low * b_low

            if not all(
                math.isfinite(term)
                for term in (term_0, term_1, term_2, term_3)
            ):
                raise PhotinicNumericalError(
                    "Intermedio no finito en TwoProduct de Dekker."
                )

            residual = ((term_0 + term_1) + term_2) + term_3
            if math.isfinite(residual):
                return product, residual
        except PhotinicNumericalError:
            pass

        return product, KahanNeumaierSum._fraction_product_residual(
            a, b, product
        )

    @staticmethod
    def compensated_dot(
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> float:
        """Producto punto con productos Dekker y acumulaciones Neumaier."""
        u_array = _readonly_real_array(u, "u")
        v_array = _readonly_real_array(v, "v")
        if u_array.shape != v_array.shape:
            raise PhotinicDimensionError(
                f"Formas incompatibles en producto punto: "
                f"{u_array.shape} != {v_array.shape}."
            )

        products: list[float] = []
        errors: list[float] = []
        for left, right in zip(u_array.ravel(), v_array.ravel()):
            product, error = KahanNeumaierSum.two_product_dekker(
                float(left), float(right)
            )
            products.append(product)
            errors.append(error)

        product_sum = KahanNeumaierSum.sum(products)
        error_sum = KahanNeumaierSum.sum(errors)
        return KahanNeumaierSum.sum((product_sum, error_sum))

    @staticmethod
    def compensated_l2_norm(vector: NDArray[np.float64]) -> float:
        """Norma 2 escalada para evitar overflow al elevar componentes al cuadrado."""
        array = _readonly_real_array(vector, "vector").ravel()
        if array.size == 0:
            return 0.0

        scale = float(np.max(np.abs(array)))
        if scale == 0.0:
            return 0.0

        scaled = array / scale
        sum_squares = KahanNeumaierSum.sum(
            float(value) * float(value) for value in scaled
        )
        factor = math.sqrt(sum_squares)

        if factor == 0.0:
            return 0.0
        if scale > np.finfo(np.float64).max / factor:
            raise PhotinicNumericalError("La norma 2 excede el rango float64.")
        return scale * factor


def _stable_norm(values: Any) -> float:
    """Norma euclídea/Frobenius escalada para arreglos reales."""
    array = np.asarray(values, dtype=np.float64)
    return KahanNeumaierSum.compensated_l2_norm(array.ravel())


def _stable_complex_norm(values: Any) -> float:
    """Norma Frobenius compleja estable, evaluada sobre partes real e imaginaria."""
    array = np.asarray(values, dtype=np.complex128)
    joined = np.concatenate(
        (array.real.ravel(), array.imag.ravel())
    )
    return KahanNeumaierSum.compensated_l2_norm(joined)


def _relative_defect(numerator: Any, reference: Any) -> float:
    """||numerator|| / max(||reference||, tiny), con control de rango."""
    numerator_norm = _stable_norm(numerator)
    reference_norm = _stable_norm(reference)
    denominator = max(reference_norm, _FLOAT_TINY)
    return numerator_norm / denominator


# ══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PhotinicObservationKernel:
    """
    Expediente de observación de la Fase 1.

    Los campos majorana_spinor_norm y majorana_chiral_parity se conservan por
    compatibilidad. Son, respectivamente, ||S||_F y Tr(S)/||S||_F; no son
    invariantes de un espinor físico sin datos espinoriales adicionales.
    """

    node_policies: Tuple[NDArray[np.float64], ...]
    majorana_spinor_norm: float
    majorana_chiral_parity: float
    banach_ratios: Tuple[float, ...]
    gram_matrix: NDArray[np.float64]
    gram_eigenvalues: NDArray[np.float64]
    gram_symmetry_defect: float
    gram_reconstruction_residual: float
    condition_number: float
    spectral_gap: float
    sha256_seal: str
    timestamp: float
    regularization: float = _DEFAULT_GRAM_REGULARIZATION
    validation_tolerance: float = _DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        policies = tuple(
            _readonly_real_array(policy, f"node_policies[{index}]")
            for index, policy in enumerate(self.node_policies)
        )
        if not policies:
            raise PhotinicDimensionError("El kernel no puede contener cero políticas.")

        dimension = policies[0].size
        if dimension == 0:
            raise PhotinicDimensionError("Las políticas no pueden tener dimensión cero.")
        for index, policy in enumerate(policies):
            if policy.ndim != 1 or policy.size != dimension:
                raise PhotinicDimensionError(
                    f"node_policies[{index}] no es un vector de dimensión {dimension}."
                )

        gram = _readonly_real_array(self.gram_matrix, "gram_matrix")
        eigenvalues = _readonly_real_array(
            self.gram_eigenvalues, "gram_eigenvalues"
        )
        if gram.shape != (dimension, dimension):
            raise PhotinicDimensionError(
                f"gram_matrix tiene shape={gram.shape}; "
                f"se esperaba ({dimension}, {dimension})."
            )
        if eigenvalues.shape != (dimension,):
            raise PhotinicDimensionError(
                f"gram_eigenvalues tiene shape={eigenvalues.shape}; "
                f"se esperaba ({dimension},)."
            )

        ratios = tuple(
            _nonnegative_float(value, f"banach_ratios[{index}]")
            for index, value in enumerate(self.banach_ratios)
        )
        if len(ratios) != len(policies):
            raise PhotinicDimensionError(
                "Debe existir un cociente de Banach por política."
            )

        object.__setattr__(self, "node_policies", policies)
        object.__setattr__(self, "banach_ratios", ratios)
        object.__setattr__(self, "gram_matrix", gram)
        object.__setattr__(self, "gram_eigenvalues", eigenvalues)

        object.__setattr__(
            self,
            "majorana_spinor_norm",
            _nonnegative_float(
                self.majorana_spinor_norm, "majorana_spinor_norm"
            ),
        )
        object.__setattr__(
            self,
            "majorana_chiral_parity",
            _finite_float(
                self.majorana_chiral_parity, "majorana_chiral_parity"
            ),
        )
        object.__setattr__(
            self,
            "gram_symmetry_defect",
            _nonnegative_float(
                self.gram_symmetry_defect, "gram_symmetry_defect"
            ),
        )
        object.__setattr__(
            self,
            "gram_reconstruction_residual",
            _nonnegative_float(
                self.gram_reconstruction_residual,
                "gram_reconstruction_residual",
            ),
        )
        object.__setattr__(
            self,
            "condition_number",
            _nonnegative_float(
                self.condition_number,
                "condition_number",
                allow_positive_infinity=True,
            ),
        )
        object.__setattr__(
            self,
            "spectral_gap",
            _nonnegative_float(
                self.spectral_gap,
                "spectral_gap",
                allow_positive_infinity=True,
            ),
        )
        object.__setattr__(
            self, "timestamp", _finite_float(self.timestamp, "timestamp")
        )

        regularization = _finite_float(self.regularization, "regularization")
        if regularization < 0.0:
            raise PhotinicInputError("regularization no puede ser negativa.")
        object.__setattr__(self, "regularization", regularization)

        validation_tolerance = _finite_float(
            self.validation_tolerance, "validation_tolerance"
        )
        if validation_tolerance <= 0.0:
            raise PhotinicInputError("validation_tolerance debe ser positiva.")
        object.__setattr__(
            self, "validation_tolerance", validation_tolerance
        )

        if not isinstance(self.sha256_seal, str):
            raise PhotinicInputError("sha256_seal debe ser texto.")

    @property
    def hmac_seal(self) -> str:
        """Alias semántico: sha256_seal contiene un HMAC-SHA256."""
        return self.sha256_seal

    @property
    def gram_frobenius_norm(self) -> float:
        """Alias explícito del invariante ||S||_F."""
        return self.majorana_spinor_norm

    @property
    def gram_trace_frobenius_ratio(self) -> float:
        """Alias explícito del cociente Tr(S)/||S||_F."""
        return self.majorana_chiral_parity


@dataclass(frozen=True, slots=True)
class PhotinicPolicyReport:
    """Reporte autenticable de los diagnósticos de la Fase 2."""

    kernel: PhotinicObservationKernel
    idempotence_residual: float
    trace_rank_mismatch: float

    # None significa que el Choi d²×d² no se materializó por límite de recursos.
    choi_min_eigenvalue: Optional[float]

    # Nombre legado: es ||sum K†K - I||_F, no un residuo de traza del Choi.
    choi_trace_residual: float

    # Nombre legado: lambda_min(I - sum K†K), que caracteriza TNI.
    choi_subunital_defect_min_eigenvalue: float

    # None cuando no se proporcionó un estado de dos qubits.
    tsirelson_chsh_value: Optional[float]
    horodecki_eigs: Optional[Tuple[float, float]]
    horodecki_bipartition_pairs: int

    is_choi_completely_positive: bool
    is_trace_non_increasing: bool

    # No se deduce de un conjunto local de operadores de Kraus.
    is_causal_non_signaling: Optional[bool]

    is_tsirelson_bounded: Optional[bool]
    is_classical_lhv: Optional[bool]
    timestamp: float

    is_trace_preserving: bool = False
    choi_trace: Optional[float] = None
    choi_materialized: bool = False
    projector_rank: int = 0
    is_projector_valid: bool = True

    bell_state: Optional[NDArray[np.complex128]] = None
    bell_correlation_tensor: Optional[NDArray[np.float64]] = None
    bell_state_is_physical: Optional[bool] = None
    bell_state_min_eigenvalue: Optional[float] = None
    bell_state_hermiticity_defect: Optional[float] = None

    hmac_seal: str = ""

    def __post_init__(self) -> None:
        if self.bell_state is not None:
            object.__setattr__(
                self,
                "bell_state",
                _readonly_complex_array(self.bell_state, "bell_state"),
            )
        if self.bell_correlation_tensor is not None:
            object.__setattr__(
                self,
                "bell_correlation_tensor",
                _readonly_real_array(
                    self.bell_correlation_tensor,
                    "bell_correlation_tensor",
                ),
            )
        if self.horodecki_eigs is not None:
            if len(self.horodecki_eigs) != 2:
                raise PhotinicDimensionError(
                    "horodecki_eigs debe contener exactamente dos autovalores."
                )
            object.__setattr__(
                self,
                "horodecki_eigs",
                (
                    _nonnegative_float(
                        self.horodecki_eigs[0], "horodecki_eigs[0]"
                    ),
                    _nonnegative_float(
                        self.horodecki_eigs[1], "horodecki_eigs[1]"
                    ),
                ),
            )
        object.__setattr__(
            self,
            "idempotence_residual",
            _nonnegative_float(
                self.idempotence_residual, "idempotence_residual"
            ),
        )
        object.__setattr__(
            self,
            "trace_rank_mismatch",
            _nonnegative_float(
                self.trace_rank_mismatch, "trace_rank_mismatch"
            ),
        )
        object.__setattr__(
            self,
            "choi_trace_residual",
            _nonnegative_float(
                self,  # placeholder, replaced immediately below
                "unused",
            ) if False else _nonnegative_float(
                self.choi_trace_residual, "choi_trace_residual"
            ),
        )
        object.__setattr__(
            self, "timestamp", _finite_float(self.timestamp, "timestamp")
        )

    @property
    def trace_preservation_residual(self) -> float:
        """Nombre semántico del residuo legado choi_trace_residual."""
        return self.choi_trace_residual

    @property
    def trace_nonincrease_defect_min_eigenvalue(self) -> float:
        """Nombre semántico de lambda_min(I - sum K†K)."""
        return self.choi_subunital_defect_min_eigenvalue

    @property
    def is_quantum_operation(self) -> bool:
        """Una operación CP y TNI; puede no preservar la traza."""
        return bool(
            self.is_choi_completely_positive
            and self.is_trace_non_increasing
        )

    @property
    def is_cptp(self) -> bool:
        """Verdadero solo si la operación reportada es CP y TP."""
        return bool(
            self.is_choi_completely_positive
            and self.is_trace_preserving
        )

    @property
    def bell_analysis_available(self) -> bool:
        """Indica si se auditó un estado explícito de dos qubits."""


@dataclass(frozen=True, slots=True)
class PhotinicEngineState:
    """Certificado terminal de la Fase 3."""

    kernel: PhotinicObservationKernel
    policy_report: PhotinicPolicyReport
    fpu_execution_time_ms: float

    # Operador de Gram normalizado y proyectado a PSD dentro de tolerancia.
    # No implica por sí solo un estado físico cuántico.
    federated_density_matrix: NDArray[np.float64]

    von_neumann_entropy: float
    quantum_purity: float

    # Solo existe para un estado de dos qubits puro suministrado explícitamente.
    bell_pair_entanglement_entropy: Optional[float]

    fpu_precision_drift: float
    cryptographic_seal: str
    phase_signature: str
    density_projection_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "federated_density_matrix",
            _readonly_real_array(
                self.federated_density_matrix,
                "federated_density_matrix",
            ),
        )
        object.__setattr__(
            self,
            "fpu_execution_time_ms",
            _nonnegative_float(
                self.fpu_execution_time_ms,
                "fpu_execution_time_ms",
            ),
        )
        object.__setattr__(
            self,
            "von_neumann_entropy",
            _nonnegative_float(
                self.von_neumann_entropy,
                "von_neumann_entropy",
            ),
        )
        object.__setattr__(
            self,
            "quantum_purity",
            _nonnegative_float(self.quantum_purity, "quantum_purity"),
        )
        object.__setattr__(
            self,
            "fpu_precision_drift",
            _nonnegative_float(
                self.fpu_precision_drift, "fpu_precision_drift"
            ),
        )
        object.__setattr__(
            self,
            "density_projection_residual",
            _nonnegative_float(
                self.density_projection_residual,
                "density_projection_residual",
            ),
        )
        if self.bell_pair_entanglement_entropy is not None:
            object.__setattr__(
                self,
                "bell_pair_entanglement_entropy",
                _nonnegative_float(
                    self.bell_pair_entanglement_entropy,
                    "bell_pair_entanglement_entropy",
                ),
            )


@dataclass(frozen=True, slots=True)
class _ChannelAudit:
    """Resultados internos de la auditoría de la familia de Kraus."""

    choi_min_eigenvalue: Optional[float]
    trace_preservation_residual: float
    trace_nonincrease_min_eigenvalue: float
    is_completely_positive: bool
    is_trace_non_increasing: bool
    is_trace_preserving: bool
    choi_trace: float
    choi_materialized: bool


@dataclass(frozen=True, slots=True)
class _BellAudit:
    """Resultados internos del estado bipartito explícito."""

    state: NDArray[np.complex128]
    correlation_tensor: NDArray[np.float64]
    chsh_value: float
    horodecki_eigs: Tuple[float, float]
    is_tsirelson_bounded: bool
    is_classical_lhv: bool
    minimum_state_eigenvalue: float
    hermiticity_defect: float


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE: POLÍTICAS, GRAM, INVARIANTES Y HMAC
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_PhotinicObservationKernel:
    """
    Fase 1 del pipeline.

    Construye S = mean_k(p_k p_kᵀ) + regularization*I, separando la
    regularización de la tolerancia de validación. El Gram se comprueba antes
    de cualquier diagonalización; la positividad se evalúa con tolerancia
    relativa a la escala espectral.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOLERANCE,
        hmac_key: Optional[bytes] = None,
        regularization: float = _DEFAULT_GRAM_REGULARIZATION,
    ) -> None:
        self._tol = _finite_float(tolerance, "tolerance")
        if not 0.0 < self._tol < 1.0:
            raise PhotinicInputError("tolerance debe satisfacer 0 < tolerance < 1.")

        self._regularization = _finite_float(
            regularization, "regularization"
        )
        if self._regularization < 0.0:
            raise PhotinicInputError("regularization no puede ser negativa.")

        if hmac_key is None:
            self._hmac_key = secrets.token_bytes(32)
            logger.warning(
                "HMAC con clave aleatoria efímera. Inyecte hmac_key desde un "
                "gestor de secretos para verificar expedientes entre procesos."
            )
        else:
            if not isinstance(hmac_key, (bytes, bytearray)):
                raise PhotinicInputError("hmac_key debe ser bytes.")
            self._hmac_key = bytes(hmac_key)
            if len(self._hmac_key) < _MIN_HMAC_KEY_BYTES:
                raise PhotinicInputError(
                    f"hmac_key debe tener al menos {_MIN_HMAC_KEY_BYTES} bytes."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Serialización canónica y sellado autenticado
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _float_token(value: float) -> bytes:
        """Codificación estable de float, incluyendo infinitos diagnósticos."""
        value = float(value)
        if math.isnan(value):
            raise PhotinicInputError("No se puede sellar un NaN.")
        if value == 0.0:
            value = 0.0
        if math.isinf(value):
            return b"+inf" if value > 0.0 else b"-inf"
        return value.hex().encode("ascii")

    @staticmethod
    def _optional_float_token(value: Optional[float]) -> bytes:
        return b"<none>" if value is None else (
            b"<float>" + Phase1_PhotinicObservationKernel._float_token(value)
        )

    @staticmethod
    def _bool_token(value: Optional[bool]) -> bytes:
        if value is None:
            return b"<none>"
        return b"<true>" if bool(value) else b"<false>"

    @staticmethod
    def _text_token(value: str) -> bytes:
        return value.encode("utf-8")

    @staticmethod
    def _array_token(array: Any) -> bytes:
        """Codifica forma, tipo real/complejo y datos little-endian contiguos."""
        raw = np.asarray(array)
        if np.iscomplexobj(raw):
            canonical = np.array(raw, dtype="<c16", copy=True, order="C")
            if not np.all(np.isfinite(canonical)):
                raise PhotinicInputError("No se puede sellar un arreglo no finito.")
            canonical[canonical == 0.0] = 0.0 + 0.0j
            type_tag = b"C"
        else:
            canonical = np.array(raw, dtype="<f8", copy=True, order="C")
            if not np.all(np.isfinite(canonical)):
                raise PhotinicInputError("No se puede sellar un arreglo no finito.")
            canonical[canonical == 0.0] = 0.0
            type_tag = b"R"

        shape_bytes = struct.pack("<I", canonical.ndim)
        for extent in canonical.shape:
            shape_bytes += struct.pack("<Q", int(extent))
        return type_tag + shape_bytes + canonical.tobytes(order="C")

    def _compute_hmac_seal(self, domain: bytes, *chunks: bytes) -> str:
        """HMAC-SHA256 con separación de dominio y fragmentos enmarcados."""
        if not isinstance(domain, bytes) or not domain:
            raise PhotinicInputError("El dominio HMAC debe ser bytes no vacíos.")
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)

        for chunk in (domain, *chunks):
            if not isinstance(chunk, bytes):
                raise PhotinicInputError("Los fragmentos HMAC deben ser bytes.")
            mac.update(struct.pack(">Q", len(chunk)))
            mac.update(chunk)
        return mac.hexdigest()

    def _observation_chunks(
        self,
        kernel: PhotinicObservationKernel,
    ) -> Tuple[bytes, ...]:
        chunks: list[bytes] = [
            b"schema=6",
            self._float_token(kernel.regularization),
            self._float_token(kernel.validation_tolerance),
            self._float_token(kernel.timestamp),
            str(len(kernel.node_policies)).encode("ascii"),
            str(kernel.gram_matrix.shape[0]).encode("ascii"),
        ]

        for policy in kernel.node_policies:
            chunks.append(self._array_token(policy))

        chunks.extend(
            (
                self._array_token(kernel.gram_matrix),
                self._array_token(kernel.gram_eigenvalues),
                self._float_token(kernel.majorana_spinor_norm),
                self._float_token(kernel.majorana_chiral_parity),
                self._float_token(kernel.gram_symmetry_defect),
                self._float_token(kernel.gram_reconstruction_residual),
                self._float_token(kernel.condition_number),
                self._float_token(kernel.spectral_gap),
            )
        )
        chunks.extend(
            self._float_token(value) for value in kernel.banach_ratios
        )
        return tuple(chunks)

    def verify_observation_kernel(
        self,
        kernel: PhotinicObservationKernel,
    ) -> bool:
        """Verifica el HMAC del expediente; no sustituye su procedencia confiable."""
        if not isinstance(kernel, PhotinicObservationKernel):
            return False
        try:
            expected = self._compute_hmac_seal(
                b"photinic-observation-kernel-v6",
                *self._observation_chunks(kernel),
            )
        except Exception:
            return False
        return hmac.compare_digest(kernel.sha256_seal, expected)

    def _require_valid_kernel(
        self,
        kernel: PhotinicObservationKernel,
    ) -> None:
        if not self.verify_observation_kernel(kernel):
            raise PhotinicIntegrityError(
                "El kernel de observación no está autenticado por esta instancia."
            )

    def _policy_report_chunks(
        self,
        report: PhotinicPolicyReport,
    ) -> Tuple[bytes, ...]:
        chunks: list[bytes] = [
            report.kernel.sha256_seal.encode("ascii"),
            self._float_token(report.idempotence_residual),
            self._float_token(report.trace_rank_mismatch),
            self._optional_float_token(report.choi_min_eigenvalue),
            self._float_token(report.choi_trace_residual),
            self._float_token(
                report.choi_subunital_defect_min_eigenvalue
            ),
            self._optional_float_token(report.tsirelson_chsh_value),
            self._float_token(report.timestamp),
            str(report.horodecki_bipartition_pairs).encode("ascii"),
            str(report.projector_rank).encode("ascii"),
            self._bool_token(report.is_choi_completely_positive),
            self._bool_token(report.is_trace_non_increasing),
            self._bool_token(report.is_trace_preserving),
            self._bool_token(report.is_causal_non_signaling),
            self._bool_token(report.is_tsirelson_bounded),
            self._bool_token(report.is_classical_lhv),
            self._bool_token(report.choi_materialized),
            self._bool_token(report.is_projector_valid),
            self._optional_float_token(report.choi_trace),
            self._bool_token(report.bell_state_is_physical),
            self._optional_float_token(report.bell_state_min_eigenvalue),
            self._optional_float_token(report.bell_state_hermiticity_defect),
        ]

        if report.horodecki_eigs is None:
            chunks.append(b"<no-horodecki-eigs>")
        else:
            chunks.extend(
                self._float_token(value)
                for value in report.horodecki_eigs
            )

        if report.bell_state is None:
            chunks.append(b"<no-bell-state>")
        else:
            chunks.append(self._array_token(report.bell_state))

        if report.bell_correlation_tensor is None:
            chunks.append(b"<no-correlation-tensor>")
        else:
            chunks.append(
                self._array_token(report.bell_correlation_tensor)
            )

        return tuple(chunks)

    def verify_policy_report(
        self,
        report: PhotinicPolicyReport,
    ) -> bool:
        """Verifica tanto el kernel incluido como el sello del reporte."""
        if not isinstance(report, PhotinicPolicyReport):
            return False
        if not self.verify_observation_kernel(report.kernel):
            return False

        try:
            expected = self._compute_hmac_seal(
                b"photinic-policy-report-v6",
                *self._policy_report_chunks(report),
            )
        except Exception:
            return False
        return hmac.compare_digest(report.hmac_seal, expected)

    # ──────────────────────────────────────────────────────────────────────────
    # Normas y auditoría de Gram
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_banach_regularity(
        self,
        vector: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula ||v||₁/||v||₂ para un vector real no nulo.

        Para v != 0, 1 <= ||v||₁/||v||₂ <= sqrt(d). El vector nulo tiene ratio
        indefinido matemáticamente; este método devuelve 0 como indicador
        explícito de nodo sin actividad.
        """
        value = _readonly_real_array(vector, "vector")
        if value.ndim != 1 or value.size == 0:
            raise PhotinicDimensionError(
                "La política debe ser un vector no vacío."
            )

        scale = float(np.max(np.abs(value)))
        if scale == 0.0:
            return 0.0

        scaled = value / scale
        l1_scaled = KahanNeumaierSum.sum(
            abs(float(component)) for component in scaled
        )
        l2_scaled = math.sqrt(
            KahanNeumaierSum.sum(
                float(component) * float(component)
                for component in scaled
            )
        )
        if l2_scaled == 0.0:
            return 0.0

        ratio = l1_scaled / l2_scaled
        lower = 1.0
        upper = math.sqrt(value.size)
        slack = 64.0 * _MACHINE_EPS * max(1.0, upper)

        if ratio < lower - slack or ratio > upper + slack:
            raise PhotinicNumericalError(
                f"El cociente l1/l2={ratio:.6e} viola sus cotas analíticas."
            )
        return float(ratio)

    def _audit_spectral_symmetry_defect(
        self,
        gram: NDArray[np.float64],
    ) -> float:
        """Calcula ||S-Sᵀ||F/||S||F y falla si excede la tolerancia."""
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise PhotinicDimensionError("El operador de Gram debe ser cuadrado.")

        denominator = max(_stable_norm(gram), _FLOAT_TINY)
        defect = _stable_norm(gram - gram.T) / denominator
        if not math.isfinite(defect) or defect > self._tol:
            raise PhotinicNumericalError(
                f"Defecto de simetría de Gram {defect:.3e} "
                f"> tolerancia {self._tol:.3e}."
            )
        return float(defect)

    def _audit_gram_reconstruction_consistency(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        gram: NDArray[np.float64],
    ) -> float:
        """
        Reconstruye cada entrada del Gram mediante productos punto compensados.

        La comparación descuenta la regularización diagonal. Esta segunda ruta
        es un contraste numérico, no una prueba de autenticidad criptográfica.
        """
        if not cleaned_policies:
            raise PhotinicDimensionError("No hay políticas que reconstruir.")

        stack = np.vstack(cleaned_policies)
        node_count, dimension = stack.shape
        reconstructed = np.zeros((dimension, dimension), dtype=np.float64)

        for i in range(dimension):
            for j in range(i, dimension):
                value = (
                    KahanNeumaierSum.compensated_dot(
                        stack[:, i], stack[:, j]
                    )
                    / float(node_count)
                )
                reconstructed[i, j] = value
                reconstructed[j, i] = value

        unregularized_gram = gram - (
            self._regularization * np.eye(dimension, dtype=np.float64)
        )
        residual = _stable_norm(reconstructed - unregularized_gram)
        reference = max(
            _stable_norm(reconstructed),
            _stable_norm(unregularized_gram),
            _FLOAT_TINY,
        )
        relative_residual = residual / reference
        allowed = max(
            _GRAM_RECONSTRUCTION_TOLERANCE,
            self._tol,
        )

        if not math.isfinite(relative_residual) or relative_residual > allowed:
            raise PhotinicNumericalError(
                "La reconstrucción compensada del Gram no coincide: "
                f"residuo relativo={relative_residual:.3e}, "
                f"límite={allowed:.3e}."
            )
        return float(relative_residual)

    def compute_majorana_sym_invariants(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        gram: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        Calcula invariantes del Gram con nombres históricos.

        Devuelve:
          1. ||S||_F = sqrt(Tr(SᵀS)); si S es simétrica, sqrt(Tr(S²)).
          2. Tr(S)/||S||_F, o cero si S es nulo.
          3. Residuo relativo de reconstrucción del Gram.

        Ninguno de estos cálculos, por sí solo, demuestra una interpretación
        espinorial de Majorana ni una simetría quiral gamma_5.
        """
        matrix = _readonly_real_array(gram, "gram")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise PhotinicDimensionError("gram debe ser una matriz cuadrada.")

        symmetry = _relative_defect(matrix - matrix.T, matrix)
        if symmetry > self._tol:
            raise PhotinicNumericalError(
                "compute_majorana_sym_invariants recibió un Gram no simétrico."
            )

        symmetric = 0.5 * matrix + 0.5 * matrix.T
        try:
            eigenvalues = np.linalg.eigvalsh(symmetric)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo diagonalizar el Gram."
            ) from exc

        gram_norm = _stable_norm(eigenvalues)
        trace = KahanNeumaierSum.sum(float(value) for value in eigenvalues)
        ratio = trace / gram_norm if gram_norm > 0.0 else 0.0

        reconstruction_residual = (
            self._audit_gram_reconstruction_consistency(
                cleaned_policies,
                symmetric,
            )
        )
        return float(gram_norm), float(ratio), reconstruction_residual

    def _normalize_policy_sequence(
        self,
        node_policies: Sequence[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], ...]:
        try:
            raw_policies = list(node_policies)
        except TypeError as exc:
            raise PhotinicInputError(
                "node_policies debe ser una secuencia de vectores."
            ) from exc

        if not raw_policies:
            raise PhotinicDimensionError(
                "El consorcio federado contiene cero políticas."
            )

        policies: list[NDArray[np.float64]] = []
        dimension: Optional[int] = None

        for index, raw_policy in enumerate(raw_policies):
            policy = _readonly_real_array(
                raw_policy, f"node_policies[{index}]"
            )
            if policy.ndim != 1 or policy.size == 0:
                raise PhotinicDimensionError(
                    f"node_policies[{index}] debe ser un vector no vacío."
                )
            if dimension is None:
                dimension = int(policy.size)
            elif policy.size != dimension:
                raise PhotinicDimensionError(
                    f"node_policies[{index}] tiene dimensión {policy.size}; "
                    f"se esperaba {dimension}."
                )
            policies.append(policy)

        return tuple(policies)

    # ──────────────────────────────────────────────────────────────────────────
    # Sellado canónico del expediente de observación
    # ──────────────────────────────────────────────────────────────────────────

    def observe_federated_nodes(
        self,
        node_policies: Sequence[NDArray[np.float64]],
    ) -> PhotinicObservationKernel:
        """Valida políticas, construye el Gram y emite un kernel HMAC-sellado."""
        policies = self._normalize_policy_sequence(node_policies)
        stack = np.vstack(policies)
        node_count, dimension = stack.shape

        ratios = tuple(
            self.evaluate_banach_regularity(policy)
            for policy in policies
        )

        with np.errstate(over="ignore", invalid="ignore"):
            gram_raw = (stack.T @ stack) / float(node_count)

        if not np.all(np.isfinite(gram_raw)):
            raise PhotinicNumericalError(
                "El producto PᵀP excede el rango float64."
            )

        symmetry_defect = self._audit_spectral_symmetry_defect(gram_raw)
        gram_symmetric = 0.5 * gram_raw + 0.5 * gram_raw.T
        gram = gram_symmetric + (
            self._regularization
            * np.eye(dimension, dtype=np.float64)
        )
        gram = 0.5 * gram + 0.5 * gram.T

        if not np.all(np.isfinite(gram)):
            raise PhotinicNumericalError(
                "El Gram regularizado contiene valores no finitos."
            )

        try:
            eigenvalues = np.linalg.eigvalsh(gram)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo calcular el espectro del Gram."
            ) from exc

        if not np.all(np.isfinite(eigenvalues)):
            raise PhotinicNumericalError(
                "El espectro del Gram contiene valores no finitos."
            )

        spectral_scale = max(
            float(np.max(np.abs(eigenvalues))),
            self._regularization,
            _FLOAT_TINY,
        )
        psd_tolerance = max(
            self._tol,
            64.0 * _MACHINE_EPS * max(1, dimension),
        ) * spectral_scale

        if float(eigenvalues[0]) < -psd_tolerance:
            raise PhotinicNonPositiveDefiniteError(
                f"El Gram no es PSD dentro de tolerancia: "
                f"lambda_min={eigenvalues[0]:.3e}, "
                f"tol={psd_tolerance:.3e}."
            )

        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
        condition_number = (
            lambda_max / lambda_min if lambda_min > 0.0 else math.inf
        )

        if condition_number > _CONDITION_NUMBER_MAX:
            logger.warning(
                "Gram mal condicionado: kappa=%.3e > %.3e",
                condition_number,
                _CONDITION_NUMBER_MAX,
            )

        # Convención explícita: gap entre los dos autovalores dominantes.
        # Para dimensión uno no existe segundo autovalor: gap = +inf.
        spectral_gap = (
            float(eigenvalues[-1] - eigenvalues[-2])
            if dimension > 1
            else math.inf
        )
        spectral_gap = max(spectral_gap, 0.0)

        majorana_proxy_norm, chiral_proxy_ratio, reconstruction_residual = (
            self.compute_majorana_sym_invariants(policies, gram)
        )

        kernel = PhotinicObservationKernel(
            node_policies=policies,
            majorana_spinor_norm=majorana_proxy_norm,
            majorana_chiral_parity=chiral_proxy_ratio,
            banach_ratios=ratios,
            gram_matrix=gram,
            gram_eigenvalues=eigenvalues,
            gram_symmetry_defect=symmetry_defect,
            gram_reconstruction_residual=reconstruction_residual,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            sha256_seal="",
            timestamp=time.time(),
            regularization=self._regularization,
            validation_tolerance=self._tol,
        )

        seal = self._compute_hmac_seal(
            b"photinic-observation-kernel-v6",
            *self._observation_chunks(kernel),
        )
        sealed_kernel = replace(kernel, sha256_seal=seal)

        logger.debug(
            "Fase 1 OK | nodes=%d | dim=%d | kappa=%.3e | "
            "lambda_min=%.3e | gram_residual=%.3e",
            node_count,
            dimension,
            condition_number,
            lambda_min,
            reconstruction_residual,
        )
        return sealed_kernel

    def canonize_photinic_observation_kernel(
        self,
        node_policies: Sequence[NDArray[np.float64]],
    ) -> PhotinicObservationKernel:
        """Método terminal de la Fase 1 y entrada canónica de la Fase 2."""
        return self.observe_federated_nodes(node_policies)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT: PROYECTOR, KRAUS, CHOI Y HORODECKI
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_PhotinicFederatedOrient(Phase1_PhotinicObservationKernel):
    """
    Fase 2 del pipeline.

    El proyector se calcula sobre el espacio fila de P. La familia de Kraus
    auditada es K_k = |p̂_k><p̂_k|/sqrt(N), con K_k=0 para una política nula.
    El análisis de Bell es independiente de esas políticas y requiere un estado
    de dos qubits explícito.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOLERANCE,
        hmac_key: Optional[bytes] = None,
        regularization: float = _DEFAULT_GRAM_REGULARIZATION,
        *,
        require_trace_preserving: bool = False,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            hmac_key=hmac_key,
            regularization=regularization,
        )
        if not isinstance(require_trace_preserving, (bool, np.bool_)):
            raise PhotinicInputError(
                "require_trace_preserving debe ser booleano."
            )
        self._require_trace_preserving = bool(require_trace_preserving)

    # ──────────────────────────────────────────────────────────────────────────
    # Proyector ortogonal de Grothendieck
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_grothendieck_projector(
        self,
        policies: Tuple[NDArray[np.float64], ...],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Construye Omega=V_rV_rᵀ mediante SVD de P.

        La tolerancia de rango es max(N,d)*eps*sigma_max. Si P=0, el rango es
        cero y Omega es el proyector nulo; no se fuerza rango uno.
        """
        if not policies:
            raise PhotinicDimensionError("Se requiere al menos una política.")

        matrix = np.vstack(policies)
        if matrix.ndim != 2:
            raise PhotinicDimensionError("La matriz de políticas debe ser 2D.")

        scale = float(np.max(np.abs(matrix)))
        dimension = matrix.shape[1]

        if scale == 0.0:
            projector = np.zeros(
                (dimension, dimension), dtype=np.float64
            )
            return projector, 0.0, 0.0

        scaled = matrix / scale
        try:
            _, singular_values, right_vectors = np.linalg.svd(
                scaled,
                full_matrices=False,
            )
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "La SVD del espacio de políticas falló."
            ) from exc

        if singular_values.size == 0 or singular_values[0] == 0.0:
            rank = 0
        else:
            rank_tolerance = (
                max(matrix.shape)
                * _MACHINE_EPS
                * float(singular_values[0])
            )
            rank = int(np.count_nonzero(singular_values > rank_tolerance))

        if rank == 0:
            projector = np.zeros(
                (dimension, dimension), dtype=np.float64
            )
        else:
            basis = right_vectors[:rank, :]
            projector = basis.T @ basis
            projector = 0.5 * projector + 0.5 * projector.T

        idempotence_residual = _stable_norm(
            projector @ projector - projector
        )
        trace_value = KahanNeumaierSum.sum(
            float(value) for value in np.diag(projector)
        )
        trace_rank_mismatch = abs(trace_value - float(rank))

        return (
            projector,
            float(idempotence_residual),
            float(trace_rank_mismatch),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Auditoría de operación cuántica mediante operadores de Kraus
    # ──────────────────────────────────────────────────────────────────────────

    def _audit_kraus_channel(
        self,
        policies: Tuple[NDArray[np.float64], ...],
    ) -> _ChannelAudit:
        """
        Audita CP y las condiciones TP/TNI de la familia K_k.

        CP se sigue de la representación de Kraus. Para dimension <= 16 se
        materializa además C = (1/d) sum_k |K_k>><<K_k| y se comprueba su
        espectro numérico.
        """
        if not policies:
            raise PhotinicDimensionError("Se requiere al menos una política.")

        node_count = len(policies)
        dimension = int(policies[0].size)
        effect_ld = np.zeros(
            (dimension, dimension), dtype=np.longdouble
        )

        materialize_choi = dimension <= _MAX_EXPLICIT_CHOI_DIMENSION
        kraus_operators: list[NDArray[np.float64]] = []

        for policy in policies:
            scale = float(np.max(np.abs(policy)))
            if scale == 0.0:
                unit = np.zeros(dimension, dtype=np.float64)
                projector = np.zeros(
                    (dimension, dimension), dtype=np.float64
                )
            else:
                scaled = np.asarray(policy / scale, dtype=np.float64)
                norm_squared = KahanNeumaierSum.sum(
                    float(component) * float(component)
                    for component in scaled
                )
                if norm_squared <= 0.0 or not math.isfinite(norm_squared):
                    raise PhotinicNumericalError(
                        "No se pudo normalizar una política no nula."
                    )
                unit = scaled / math.sqrt(norm_squared)
                projector = np.outer(unit, unit)
                projector = 0.5 * projector + 0.5 * projector.T

            effect_ld += (
                np.asarray(projector, dtype=np.longdouble)
                / np.longdouble(node_count)
            )

            if materialize_choi:
                kraus_operators.append(
                    projector / math.sqrt(float(node_count))
                )

        effect = np.asarray(effect_ld, dtype=np.float64)
        if not np.all(np.isfinite(effect)):
            raise PhotinicNumericalError(
                "La matriz de efecto de Kraus no es finita."
            )
        effect = 0.5 * effect + 0.5 * effect.T

        identity = np.eye(dimension, dtype=np.float64)
        defect = identity - effect

        try:
            defect_eigenvalues = np.linalg.eigvalsh(defect)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo diagonalizar I - sum K†K."
            ) from exc

        minimum_defect_eigenvalue = float(defect_eigenvalues[0])
        maximum_defect_magnitude = float(
            np.max(np.abs(defect_eigenvalues))
        )
        trace_residual = _stable_norm(effect - identity)

        tni_tolerance = max(
            _CHANNEL_TNI_TOLERANCE,
            self._tol,
        )
        is_trace_non_increasing = (
            minimum_defect_eigenvalue >= -tni_tolerance
        )

        tp_tolerance = max(
            self._tol,
            64.0 * _MACHINE_EPS * max(1, dimension),
        )
        is_trace_preserving = (
            maximum_defect_magnitude <= tp_tolerance
        )

        trace_effect = KahanNeumaierSum.sum(
            float(value) for value in np.diag(effect)
        )
        choi_trace = trace_effect / float(dimension)

        choi_min_eigenvalue: Optional[float]
        choi_materialized = False
        is_completely_positive = True  # Certificado analítico por Kraus.

        if materialize_choi:
            choi = np.zeros(
                (dimension * dimension, dimension * dimension),
                dtype=np.float64,
            )
            for kraus in kraus_operators:
                vectorized = kraus.reshape(-1, order="F")
                choi += np.outer(vectorized, vectorized) / float(dimension)

            choi = 0.5 * choi + 0.5 * choi.T
            try:
                choi_eigenvalues = np.linalg.eigvalsh(choi)
            except np.linalg.LinAlgError as exc:
                raise PhotinicNumericalError(
                    "No se pudo diagonalizar la matriz de Choi explícita."
                ) from exc

            if not np.all(np.isfinite(choi_eigenvalues)):
                raise PhotinicNumericalError(
                    "El espectro del Choi contiene valores no finitos."
                )

            choi_min_eigenvalue = float(choi_eigenvalues[0])
            choi_scale = max(
                float(np.max(np.abs(choi_eigenvalues))),
                1.0,
            )
            choi_tolerance = max(
                _CHOІ_PSD_TOLERANCE,
                self._tol,
            ) * choi_scale
            is_completely_positive = (
                choi_min_eigenvalue >= -choi_tolerance
            )
            choi_materialized = True
        else:
            # No se informa un autovalor ficticio. CP se deriva de la forma de
            # Kraus con operadores finitos construidos en esta rutina.
            choi_min_eigenvalue = None

        return _ChannelAudit(
            choi_min_eigenvalue=choi_min_eigenvalue,
            trace_preservation_residual=float(trace_residual),
            trace_nonincrease_min_eigenvalue=minimum_defect_eigenvalue,
            is_completely_positive=bool(is_completely_positive),
            is_trace_non_increasing=bool(is_trace_non_increasing),
            is_trace_preserving=bool(is_trace_preserving),
            choi_trace=float(choi_trace),
            choi_materialized=choi_materialized,
        )

    def _compute_choi_jamiolkowski_superoperator(
        self,
        policies: Tuple[NDArray[np.float64], ...],
    ) -> Tuple[Optional[float], float, float, bool, bool]:
        """
        Interfaz de compatibilidad para la auditoría de Kraus.

        Devuelve (lambda_min_Choi, residuo_TP, lambda_min(I-sum K†K), CP, TNI).
        """
        audit = self._audit_kraus_channel(policies)
        return (
            audit.choi_min_eigenvalue,
            audit.trace_preservation_residual,
            audit.trace_nonincrease_min_eigenvalue,
            audit.is_completely_positive,
            audit.is_trace_non_increasing,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Estado de Bell explícito y criterio de Horodecki
    # ──────────────────────────────────────────────────────────────────────────

    def _audit_bell_state(
        self,
        bell_state: Optional[Any],
    ) -> Optional[_BellAudit]:
        """
        Valida una matriz de densidad 4x4 y calcula su tensor de correlación.

        No se normaliza ni se modifica el estado para forzar la cota de Tsirelson.
        Solo se hermitiza si la desviación original ya está dentro de tolerancia.
        """
        if bell_state is None:
            return None

        rho_input = _readonly_complex_array(bell_state, "bell_state")
        if rho_input.shape != (4, 4):
            raise PhotinicDimensionError(
                "bell_state debe tener shape (4, 4), correspondiente a dos qubits."
            )

        rho = np.array(rho_input, dtype=np.complex128, copy=True)
        hermiticity_defect = (
            _stable_complex_norm(rho - rho.conj().T)
            / max(_stable_complex_norm(rho), _FLOAT_TINY)
        )
        if hermiticity_defect > max(self._tol, _BELL_STATE_TOLERANCE):
            raise PhotinicInvalidQuantumStateError(
                f"bell_state no es hermítica: defecto={hermiticity_defect:.3e}."
            )

        rho = 0.5 * rho + 0.5 * rho.conj().T
        trace_value = np.trace(rho)

        if abs(float(trace_value.imag)) > max(
            self._tol, _BELL_STATE_TOLERANCE
        ):
            raise PhotinicInvalidQuantumStateError(
                "La traza de bell_state tiene parte imaginaria no despreciable."
            )
        if abs(float(trace_value.real) - 1.0) > max(
            self._tol, _BELL_STATE_TOLERANCE
        ):
            raise PhotinicInvalidQuantumStateError(
                f"bell_state debe tener traza 1; traza={trace_value.real:.12g}."
            )

        try:
            state_eigenvalues = np.linalg.eigvalsh(rho)
        except np.linalg.LinAlgError as exc:
            raise PhotinicInvalidQuantumStateError(
                "No se pudo diagonalizar bell_state."
            ) from exc

        state_scale = max(
            float(np.max(np.abs(state_eigenvalues))),
            1.0,
        )
        state_psd_tolerance = max(
            self._tol, _BELL_STATE_TOLERANCE
        ) * state_scale
        minimum_state_eigenvalue = float(state_eigenvalues[0])

        if minimum_state_eigenvalue < -state_psd_tolerance:
            raise PhotinicInvalidQuantumStateError(
                "bell_state no es positiva semidefinida: "
                f"lambda_min={minimum_state_eigenvalue:.3e}."
            )

        sigma_x = np.array(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
        )
        sigma_y = np.array(
            [[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128
        )
        sigma_z = np.array(
            [[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128
        )
        pauli = (sigma_x, sigma_y, sigma_z)

        correlation = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                observable = np.kron(pauli[i], pauli[j])
                expectation = np.trace(rho @ observable)
                imaginary_tolerance = max(
                    self._tol,
                    _BELL_STATE_TOLERANCE,
                ) * max(1.0, abs(float(expectation.real)))
                if abs(float(expectation.imag)) > imaginary_tolerance:
                    raise PhotinicNumericalError(
                        "Una correlación Hermítica resultó compleja fuera de tolerancia."
                    )
                correlation[i, j] = float(expectation.real)

        try:
            singular_values = np.linalg.svd(
                correlation,
                compute_uv=False,
            )
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "La SVD del tensor de correlación de Bell falló."
            ) from exc

        squared = np.square(singular_values)
        u1 = float(squared[0]) if squared.size > 0 else 0.0
        u2 = float(squared[1]) if squared.size > 1 else 0.0
        chsh_value = 2.0 * math.sqrt(max(u1 + u2, 0.0))

        tsirelson_slack = max(
            _TSIRELSON_TOLERANCE,
            self._tol,
        )
        classical_slack = max(self._TSIRELSON_TOLERANCE, self._tol)

        return _BellAudit(
            state=rho,
            correlation_tensor=correlation,
            chsh_value=float(chsh_value),
            horodecki_eigs=(u1, u2),
            is_tsirelson_bounded=(
                chsh_value <= _TSIRELSON_BOUND + tsirelson_slack
            ),
            is_classical_lhv=(
                chsh_value <= _BELL_CLASSICAL_BOUND + classical_slack
            ),
            minimum_state_eigenvalue=minimum_state_eigenvalue,
            hermiticity_defect=float(hermiticity_defect),
        )

    def _compute_horodecki_tsirelson_bound(
        self,
        bell_state: Optional[Any],
    ) -> Tuple[
        Optional[float],
        Optional[Tuple[float, float]],
        int,
        Optional[bool],
        Optional[bool],
    ]:
        """
        Interfaz de Horodecki para una matriz 4x4 explícita.

        El último entero es 1 cuando se auditó la bipartición explícita y 0
        cuando no se proporcionó estado. No se divide ni se normaliza el valor
        de CHSH para imponer artificialmente la cota.
        """
        audit = self._audit_bell_state(bell_state)
        if audit is None:
            return None, None, 0, None, None
        return (
            audit.chsh_value,
            audit.horodecki_eigs,
            1,
            audit.is_tsirelson_bounded,
            audit.is_classical_lhv,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Orientación terminal y reporte autenticado
    # ──────────────────────────────────────────────────────────────────────────

    def orient_photinic_policy(
        self,
        kernel: PhotinicObservationKernel,
        bell_state: Optional[Any] = None,
    ) -> PhotinicPolicyReport:
        """Calcula diagnósticos de proyección, Kraus y, opcionalmente, Bell."""
        if not isinstance(kernel, PhotinicObservationKernel):
            raise PhotinicInputError(
                "kernel debe ser PhotinicObservationKernel."
            )
        self._require_valid_kernel(kernel)

        _, idempotence_residual, trace_rank_mismatch = (
            self._compute_grothendieck_projector(kernel.node_policies)
        )
        _, projector_diagonal_residual, _ = (
            self._compute_grothendieck_projector(kernel.node_policies)
        )
        # Se conserva una única magnitud de idempotencia: la segunda evaluación
        # determinista sirve como control de repetibilidad, no se suma al residuo.
        if abs(idempotence_residual - projector_diagonal_residual) > (
            64.0 * _MACHINE_EPS
        ):
            raise PhotinicNumericalError(
                "El cálculo repetido del proyector no es reproducible."
            )

        projector_trace = KahanNeumaierSum.sum(
            float(value)
            for value in np.diag(
                self._compute_grothendieck_projector(
                    kernel.node_policies
                )[0]
            )
        )
        projector_rank = max(0, int(round(projector_trace)))

        projector_tolerance = max(
            self._tol,
            64.0 * _MACHINE_EPS,
        ) * max(1.0, math.sqrt(float(max(projector_rank, 1))))
        is_projector_valid = (
            idempotence_residual <= projector_tolerance
            and trace_rank_mismatch <= projector_tolerance
        )

        channel = self._audit_kraus_channel(kernel.node_policies)
        bell_audit = self._audit_bell_state(bell_state)

        report = PhotinicPolicyReport(
            kernel=kernel,
            idempotence_residual=idempotence_residual,
            trace_rank_mismatch=trace_rank_mismatch,
            choi_min_eigenvalue=channel.choi_min_eigenvalue,
            choi_trace_residual=channel.trace_preservation_residual,
            choi_subunital_defect_min_eigenvalue=(
                channel.trace_nonincrease_min_eigenvalue
            ),
            tsirelson_chsh_value=(
                None if bell_audit is None else bell_audit.chsh_value
            ),
            horodecki_eigs=(
                None if bell_audit is None else bell_audit.horodecki_eigs
            ),
            horodecki_bipartition_pairs=(
                0 if bell_audit is None else 1
            ),
            is_choi_completely_positive=(
                channel.is_completely_positive
            ),
            is_trace_non_increasing=(
                channel.is_trace_non_increasing
            ),
            # No se infiere no-señalización a partir de estas entradas.
            is_causal_non_signaling=None,
            is_tsirelson_bounded=(
                None
                if bell_audit is None
                else bell_audit.is_tsirelson_bounded
            ),
            is_classical_lhv=(
                None if bell_audit is None else bell_audit.is_classical_lhv
            ),
            timestamp=time.time(),
            is_trace_preserving=channel.is_trace_preserving,
            choi_trace=channel.choi_trace,
            choi_materialized=channel.choi_materialized,
            projector_rank=projector_rank,
            is_projector_valid=is_projector_valid,
            bell_state=(
                None if bell_audit is None else bell_audit.state
            ),
            bell_correlation_tensor=(
                None
                if bell_audit is None
                else bell_audit.correlation_tensor
            ),
            bell_state_is_physical=(
                None if bell_audit is None else True
            ),
            bell_state_min_eigenvalue=(
                None
                if bell_audit is None
                else bell_audit.minimum_state_eigenvalue
            ),
            bell_state_hermiticity_defect=(
                None
                if bell_audit is None
                else bell_audit.hermiticity_defect
            ),
            hmac_seal="",
        )

        report_seal = self._compute_hmac_seal(
            b"photinic-policy-report-v6",
            *self._policy_report_chunks(report),
        )
        sealed_report = replace(report, hmac_seal=report_seal)

        logger.debug(
            "Fase 2 OK | projector_rank=%d | CP=%s | TNI=%s | TP=%s | "
            "CHSH=%s",
            projector_rank,
            sealed_report.is_choi_completely_positive,
            sealed_report.is_trace_non_increasing,
            sealed_report.is_trace_preserving,
            (
                "no disponible"
                if sealed_report.tsirelson_chsh_value is None
                else f"{sealed_report.tsirelson_chsh_value:.8f}"
            ),
        )
        return sealed_report

    def synthesize_photinic_orientation(
        self,
        kernel_or_policies: Union[
            PhotinicObservationKernel,
            Sequence[NDArray[np.float64]],
        ],
        bell_state: Optional[Any] = None,
    ) -> PhotinicPolicyReport:
        """Acopla la Fase 1 si recibe políticas crudas y ejecuta la Fase 2."""
        if isinstance(kernel_or_policies, PhotinicObservationKernel):
            kernel = kernel_or_policies
        else:
            kernel = self.canonize_photinic_observation_kernel(
                kernel_or_policies
            )
        return self.orient_photinic_policy(kernel, bell_state=bell_state)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — ACT: CERTIFICACIÓN, ENTROPÍA Y FIRMA TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_PhotinicGovernanceEngine(Phase2_PhotinicFederatedOrient):
    """Fase 3 y orquestador OODA terminal."""

    def _compute_density_matrix_and_projection_residual(
        self,
        kernel: PhotinicObservationKernel,
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Convierte el Gram a un operador PSD de traza uno.

        Si la diagonalización arroja autovalores negativos solo dentro de la
        tolerancia de redondeo, los proyecta a cero y registra la magnitud de
        la corrección. Una violación mayor se rechaza.
        """
        self._require_valid_kernel(kernel)
        gram = np.asarray(kernel.gram_matrix, dtype=np.float64)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo diagonalizar el Gram para formar la densidad."
            ) from exc

        scale = max(
            float(np.max(np.abs(eigenvalues))),
            _FLOAT_TINY,
        )
        psd_tolerance = max(
            self._tol,
            64.0 * _MACHINE_EPS * max(1, gram.shape[0]),
        ) * scale

        if float(eigenvalues[0]) < -psd_tolerance:
            raise PhotinicNonPositiveDefiniteError(
                "El Gram no puede convertirse en operador PSD: "
                f"lambda_min={eigenvalues[0]:.3e}."
            )

        positive_eigenvalues = np.maximum(eigenvalues, 0.0)
        normalization = KahanNeumaierSum.sum(
            float(value) for value in positive_eigenvalues
        )
        if normalization <= 0.0:
            raise PhotinicNonPositiveDefiniteError(
                "La traza PSD del Gram es nula; no se puede normalizar."
            )

        probabilities = positive_eigenvalues / normalization
        density = (
            eigenvectors * probabilities[np.newaxis, :]
        ) @ eigenvectors.T
        density = 0.5 * density + 0.5 * density.T

        raw_trace = KahanNeumaierSum.sum(
            float(value) for value in np.diag(gram)
        )
        if raw_trace <= 0.0:
            raise PhotinicNonPositiveDefiniteError(
                "La traza del Gram no es positiva."
            )
        raw_density = gram / raw_trace

        projection_residual = _relative_defect(
            density - raw_density,
            raw_density,
        )
        if not np.all(np.isfinite(density)):
            raise PhotinicNumericalError(
                "La matriz de densidad derivada del Gram no es finita."
            )
        return density, float(projection_residual)

    def compute_reduced_federated_density_matrix(
        self,
        kernel: PhotinicObservationKernel,
    ) -> NDArray[np.float64]:
        """
        Devuelve el Gram normalizado y proyectado a PSD dentro de tolerancia.

        El nombre se conserva por compatibilidad; su interpretación física como
        estado cuántico requiere una aplicación que defina dicho mapeo.
        """
        density, _ = self._compute_density_matrix_and_projection_residual(
            kernel
        )
        return _readonly_real_array(
            density, "federated_density_matrix"
        )

    def compute_von_neumann_entropy_full(
        self,
        rho: NDArray[Any],
    ) -> Tuple[float, float]:
        """
        Calcula -Tr(rho log rho) y Tr(rho²) sobre el espectro completo.

        Exige una matriz Hermítica PSD, normalizada a traza uno dentro de
        tolerancia. No renormaliza una entrada inválida.
        """
        density = _readonly_complex_array(rho, "rho")
        if density.ndim != 2 or density.shape[0] != density.shape[1]:
            raise PhotinicDimensionError("rho debe ser una matriz cuadrada.")

        matrix = np.asarray(density, dtype=np.complex128)
        hermiticity_defect = (
            _stable_complex_norm(matrix - matrix.conj().T)
            / max(_stable_complex_norm(matrix), _FLOAT_TINY)
        )
        allowed = max(self._tol, _BELL_STATE_TOLERANCE)
        if hermiticity_defect > allowed:
            raise PhotinicInvalidQuantumStateError(
                f"rho no es hermítica: defecto={hermiticity_defect:.3e}."
            )

        matrix = 0.5 * matrix + 0.5 * matrix.conj().T
        trace_value = np.trace(matrix)
        if abs(float(trace_value.imag)) > allowed:
            raise PhotinicInvalidQuantumStateError(
                "La traza de rho tiene parte imaginaria significativa."
            )
        if abs(float(trace_value.real) - 1.0) > allowed:
            raise PhotinicInvalidQuantumStateError(
                f"rho debe tener traza uno; recibido {trace_value.real:.12g}."
            )

        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo calcular el espectro de rho."
            ) from exc

        spectral_scale = max(
            float(np.max(np.abs(eigenvalues))),
            1.0,
        )
        if float(eigenvalues[0]) < -allowed * spectral_scale:
            raise PhotinicNonPositiveDefiniteError(
                f"rho no es PSD: lambda_min={eigenvalues[0]:.3e}."
            )

        probabilities = np.maximum(eigenvalues, 0.0)
        probability_sum = KahanNeumaierSum.sum(
            float(value) for value in probabilities
        )
        if abs(probability_sum - 1.0) > allowed * max(1, len(probabilities)):
            raise PhotinicInvalidQuantumStateError(
                "El espectro PSD de rho no suma uno dentro de tolerancia."
            )

        entropy_terms = [
            -float(value) * math.log(float(value))
            for value in probabilities
            if value > 0.0
        ]
        entropy = KahanNeumaierSum.sum(entropy_terms)
        purity = KahanNeumaierSum.sum(
            float(value) * float(value)
            for value in probabilities
        )
        return max(float(entropy), 0.0), float(purity)

    def compute_horodecki_entanglement_entropy(
        self,
        report: PhotinicPolicyReport,
    ) -> Optional[float]:
        """
        Entropía de entrelazamiento para un estado bipartito puro.

        Para un estado mixto no se identifica la entropía de una reducción con
        una medida general de entrelazamiento; en ese caso devuelve None.
        """
        if not isinstance(report, PhotinicPolicyReport):
            raise PhotinicInputError(
                "report debe ser PhotinicPolicyReport."
            )
        if report.bell_state is None:
            return None

        rho = np.asarray(report.bell_state, dtype=np.complex128)
        global_purity = float(np.trace(rho @ rho).real)
        pure_tolerance = max(
            10.0 * self._tol,
            10.0 * _BELL_STATE_TOLERANCE,
        )
        if abs(global_purity - 1.0) > pure_tolerance:
            return None

        # Índices rho[a,b,c,d]; se traza el segundo qubit b=d.
        rho_tensor = rho.reshape(2, 2, 2, 2)
        reduced_a = np.einsum(
            "abcb->ac",
            rho_tensor,
            optimize=True,
        )
        reduced_a = 0.5 * reduced_a + 0.5 * reduced_a.conj().T

        try:
            eigenvalues = np.linalg.eigvalsh(reduced_a)
        except np.linalg.LinAlgError as exc:
            raise PhotinicNumericalError(
                "No se pudo calcular el espectro reducido del estado de Bell."
            ) from exc

        if float(eigenvalues[0]) < -pure_tolerance:
            raise PhotinicNonPositiveDefiniteError(
                "La reducción del estado puro no es PSD."
            )

        probabilities = np.maximum(eigenvalues, 0.0)
        total = KahanNeumaierSum.sum(
            float(value) for value in probabilities
        )
        if total <= 0.0:
            raise PhotinicNonPositiveDefiniteError(
                "La reducción tiene traza nula."
            )

        # Corrige solo el redondeo de la traza reducida.
        probabilities = probabilities / total
        entropy = KahanNeumaierSum.sum(
            -float(value) * math.log(float(value))
            for value in probabilities
            if value > 0.0
        )
        return max(float(entropy), 0.0)

    def _policy_precision_drift(
        self,
        kernel: PhotinicObservationKernel,
        report: PhotinicPolicyReport,
        density_projection_residual: float,
    ) -> float:
        """Máximo de residuos normalizados, sin sumar magnitudes heterogéneas."""
        rank_scale = max(
            1.0,
            math.sqrt(float(max(report.projector_rank, 1))),
        )
        rank_denominator = max(1.0, float(report.projector_rank))

        components = [
            kernel.gram_symmetry_defect,
            kernel.gram_reconstruction_residual,
            report.idempotence_residual / rank_scale,
            report.trace_rank_mismatch / rank_denominator,
            density_projection_residual,
        ]

        if report.choi_min_eigenvalue is not None:
            components.append(max(0.0, -report.choi_min_eigenvalue))

        if not all(math.isfinite(value) for value in components):
            raise PhotinicNumericalError(
                "La deriva numérica contiene magnitudes no finitas."
            )
        return max(components, default=0.0)

    def _engine_state_chunks(
        self,
        state: PhotinicEngineState,
    ) -> Tuple[bytes, ...]:
        return (
            state.kernel.sha256_seal.encode("ascii"),
            state.policy_report.hmac_seal.encode("ascii"),
            self._array_token(state.federated_density_matrix),
            self._float_token(state.fpu_execution_time_ms),
            self._float_token(state.von_neumann_entropy),
            self._float_token(state.quantum_purity),
            self._optional_float_token(
                state.bell_pair_entanglement_entropy
            ),
            self._float_token(state.fpu_precision_drift),
            self._float_token(state.density_projection_residual),
            state.phase_signature.encode("utf-8"),
        )

    def verify_engine_state(self, state: PhotinicEngineState) -> bool:
        """Verifica el sello terminal y los sellos anidados."""
        if not isinstance(state, PhotinicEngineState):
            return False
        if not self.verify_observation_kernel(state.kernel):
            return False
        if not self.verify_policy_report(state.policy_report):
            return False

        try:
            expected = self._compute_hmac_seal(
                b"photinic-engine-state-v6",
                *self._engine_state_chunks(state),
            )
        except Exception:
            return False
        return hmac.compare_digest(state.cryptographic_seal, expected)

    def execute_photinic_audit(
        self,
        target: Union[
            PhotinicPolicyReport,
            PhotinicObservationKernel,
            Sequence[NDArray[np.float64]],
        ],
        *,
        bell_state: Optional[Any] = None,
    ) -> PhotinicEngineState:
        """
        Ejecuta la Fase 3 y, si es necesario, las fases precedentes.

        bell_state solo se admite cuando target es un kernel o una secuencia
        cruda. Un reporte ya orientado contiene sus propios diagnósticos Bell.
        """
        start = time.perf_counter()

        if isinstance(target, PhotinicPolicyReport):
            if bell_state is not None:
                raise PhotinicInputError(
                    "No pase bell_state junto a un PhotinicPolicyReport."
                )
            report = target
            kernel = report.kernel
        elif isinstance(target, PhotinicObservationKernel):
            kernel = target
            report = self.orient_photinic_policy(
                kernel,
                bell_state=bell_state,
            )
        else:
            report = self.synthesize_photinic_orientation(
                target,
                bell_state=bell_state,
            )
            kernel = report.kernel

        if not self.verify_policy_report(report):
            raise PhotinicIntegrityError(
                "El reporte de Fase 2 no supera la verificación HMAC."
            )

        if not report.is_projector_valid:
            raise PhotinicProjectorError(
                "El proyector no satisface idempotencia y traza-rango."
            )

        if not report.is_choi_completely_positive:
            minimum = report.choi_min_eigenvalue
            raise PhotinicNonPositiveDefiniteError(
                "El Choi materializado no supera la prueba PSD: "
                f"lambda_min={minimum}."
            )

        if not report.is_trace_non_increasing:
            raise PhotinicNonCausalChannelError(
                "La operación no es trace-nonincreasing: "
                "lambda_min(I - sum K†K)="
                f"{report.choi_subunital_defect_min_eigenvalue:.3e}."
            )

        if self._require_trace_preserving and not report.is_trace_preserving:
            raise PhotinicTracePreservationError(
                "La configuración exige TP, pero "
                f"||sum K†K-I||F={report.choi_trace_residual:.3e}."
            )

        if report.is_tsirelson_bounded is False:
            raise PhotinicTsirelsonViolationError(
                "El valor calculado de CHSH supera la cota de Tsirelson: "
                f"{report.tsirelson_chsh_value:.12g} > "
                f"{_TSIRELSON_BOUND:.12g}."
            )

        density, density_projection_residual = (
            self._compute_density_matrix_and_projection_residual(kernel)
        )
        von_neumann_entropy, purity = (
            self.compute_von_neumann_entropy_full(density)
        )
        bell_entanglement_entropy = (
            self.compute_horodecki_entanglement_entropy(report)
        )

        precision_drift = self._policy_precision_drift(
            kernel,
            report,
            density_projection_residual,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if report.tsirelson_chsh_value is None:
            bell_signature = "BELL=NA"
        else:
            bell_signature = (
                f"BELL={report.tsirelson_chsh_value:.8f}"
            )

        entanglement_signature = (
            "ENTANGLEMENT=NA"
            if bell_entanglement_entropy is None
            else f"ENTANGLEMENT={bell_entanglement_entropy:.8f}"
        )

        phase_signature = (
            f"PHOTINIC-v6:"
            f"CP={int(report.is_choi_completely_positive)}:"
            f"TNI={int(report.is_trace_non_increasing)}:"
            f"TP={int(report.is_trace_preserving)}:"
            f"{bell_signature}:"
            f"{entanglement_signature}:"
            f"S_GRAM={von_neumann_entropy:.8f}:"
            f"PURETY_GRAM={purity:.8f}"
        )

        unsigned_state = PhotinicEngineState(
            kernel=kernel,
            policy_report=report,
            fpu_execution_time_ms=float(elapsed_ms),
            federated_density_matrix=density,
            von_neumann_entropy=von_neumann_entropy,
            quantum_purity=purity,
            bell_pair_entanglement_entropy=bell_entanglement_entropy,
            fpu_precision_drift=precision_drift,
            cryptographic_seal="",
            phase_signature=phase_signature,
            density_projection_residual=density_projection_residual,
        )

        final_seal = self._compute_hmac_seal(
            b"photinic-engine-state-v6",
            *self._engine_state_chunks(unsigned_state),
        )
        state = replace(unsigned_state, cryptographic_seal=final_seal)

        logger.info(
            "Fase 3 OK | CP=%s | TNI=%s | TP=%s | drift=%.3e | "
            "elapsed=%.3f ms",
            report.is_choi_completely_positive,
            report.is_trace_non_increasing,
            report.is_trace_preserving,
            precision_drift,
            elapsed_ms,
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PÚBLICO
# ══════════════════════════════════════════════════════════════════════════════

class PhotinicGovernanceSatelliteEngine(Phase3_PhotinicGovernanceEngine):
    """Motor público de gobernanza fotínica federada."""

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOLERANCE,
        hmac_key: Optional[bytes] = None,
        regularization: float = _DEFAULT_GRAM_REGULARIZATION,
        *,
        require_trace_preserving: bool = False,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            hmac_key=hmac_key,
            regularization=regularization,
            require_trace_preserving=require_trace_preserving,
        )
        logger.info(
            "PhotinicGovernanceSatelliteEngine v6.0.0 inicializado. "
            "Canal CP/TNI auditado; TP configurable; CHSH requiere estado "
            "explícito de dos qubits."
        )


__all__ = [
    "PhotinicGovernanceSatelliteEngine",
    "PhotinicObservationKernel",
    "PhotinicPolicyReport",
    "PhotinicEngineState",
    "KahanNeumaierSum",
    "Phase1_PhotinicObservationKernel",
    "Phase2_PhotinicFederatedOrient",
    "Phase3_PhotinicGovernanceEngine",
    "PhotinicEngineError",
    "PhotinicInputError",
    "PhotinicDimensionError",
    "PhotinicNumericalError",
    "PhotinicIntegrityError",
    "PhotinicNonPositiveDefiniteError",
    "PhotinicCausalViolationError",
    "PhotinicNonCausalChannelError",
    "PhotinicTracePreservationError",
    "PhotinicTsirelsonViolationError",
    "PhotinicInvalidQuantumStateError",
    "PhotinicProjectorError",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    policies = [
        rng.normal(size=6)
        for _ in range(5)
    ]

    # Estado singlete explícito |ψ⁻><ψ⁻|. CHSH se calcula sobre este estado,
    # no sobre una bipartición inventada de los vectores de políticas.
    singlet = np.array(
        [0.0, 1.0, -1.0, 0.0],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    bell_state = np.outer(singlet, singlet.conj())

    engine = PhotinicGovernanceSatelliteEngine()
    state = engine.execute_photinic_audit(
        policies,
        bell_state=bell_state,
    )
    report = state.policy_report

    print("Fase 1 — Observe")
    print(f"  Número de nodos:             {len(state.kernel.node_policies)}")
    print(f"  Dimensión del Gram:          {state.kernel.gram_matrix.shape[0]}")
    print(f"  Condicionamiento del Gram:   {state.kernel.condition_number:.6e}")
    print(f"  Residuo reconstrucción Gram:{state.kernel.gram_reconstruction_residual:.6e}")
    print(f"  HMAC kernel:                 {state.kernel.sha256_seal[:24]}...")

    print("\nFase 2 — Orient")
    print(f"  Proyector válido:            {report.is_projector_valid}")
    print(f"  Canal CP:                    {report.is_choi_completely_positive}")
    print(f"  Operación TNI:               {report.is_trace_non_increasing}")
    print(f"  Operación TP:                {report.is_trace_preserving}")
    print(f"  Operación CPTP:              {report.is_cptp}")
    print(f"  CHSH:                        {report.tsirelson_chsh_value:.12f}")
    print(f"  Acotado por Tsirelson:       {report.is_tsirelson_bounded}")
    print(f"  No-señalización:             {report.is_causal_non_signaling}")

    print("\nFase 3 — Act")
    print(f"  Entropía espectral del Gram: {state.von_neumann_entropy:.8f}")
    print(f"  Pureza espectral del Gram:   {state.quantum_purity:.8f}")
    print(f"  Entropía de entrelazamiento: {state.bell_pair_entanglement_entropy}")
    print(f"  Deriva numérica:             {state.fpu_precision_drift:.6e}")
    print(f"  Estado HMAC válido:          {engine.verify_engine_state(state)}")
    print(f"  HMAC terminal:               {state.cryptographic_seal[:32]}...")

    assert report.is_choi_completely_positive
    assert report.is_trace_non_increasing
    assert report.is_tsirelson_bounded is True