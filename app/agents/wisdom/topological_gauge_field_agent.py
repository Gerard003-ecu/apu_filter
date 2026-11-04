# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Topological Gauge Field Agent                                    ║
║ Ubicación: app/agents/wisdom/topological_gauge_field_agent.py                ║
║ Versión  : 12.0.0 (NESTED-OODA-POINCARE-RSI3-FAIL-CLOSED)                   ║
║ Función  : Orquestador agéntico en tres fases anidadas:                      ║
║            Fase 1 — Observe: sanitización, isometría, RSI³-Lindstedt.       ║
║            Fase 2 — Orient: Yang–Mills, topología, Poincaré, QND, KAM.      ║
║            Fase 3 — Decide & Act: adjudicación conservadora, Crowbar,       ║
║            functor semántico y compromiso criptográfico del estado.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Alcance y cautelas
------------------
Este módulo valida estructuras numéricas y certificados emitidos por el motor.
Una firma HMAC acredita integridad respecto de una clave compartida; no demuestra
por sí sola una afirmación física o matemática. Del mismo modo, una secuencia
finita de residuos RSI³ sólo permite estimar contracción observada, no probar una
cota global de Lipschitz. Los datos de una sección de Poincaré son diagnósticos
numéricos, no una prueba del teorema de recurrencia. La latencia reportada por un
adaptador de hardware no certifica el WCET físico del circuito: esa garantía
requiere medición y verificación externas del ESP32, GPIO14 y el Crowbar.

Organización por fases anidadas
-------------------------------
  §1. Infraestructura, normalización, compromisos y certificados.
  §2. FASE 1 — GaugeAgentPhase1Observer.
      Su último método, continue_to_phase2, verifica y sella el puente Φ₁₂.
  §3. FASE 2 — GaugeAgentPhase2Orientor(GaugeAgentPhase1Observer).
      Comienza recibiendo Φ₁ y termina con continue_to_phase3, el puente Φ₂₃.
  §4. FASE 3 — GaugeAgentPhase3Decider(GaugeAgentPhase2Orientor).
      Comienza recibiendo Φ₂ y cierra la adjudicación y acción.
  §5. Fachada, exportación y autocomprobación.

La cadena de decisiones usa el orden de severidad
    VETOED < DEGRADED < COHERENT.
La adjudicación de gobernanza es el ínfimo (meet), por lo que un átomo más
favorable nunca puede borrar un veto previo.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import IntEnum
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════════════
# §0. MOTOR ESPECTRAL: IMPORTACIÓN COMPATIBLE Y FALLBACK AUTÓNOMO
# ══════════════════════════════════════════════════════════════════════════════

_ENGINE_IMPORTED: bool = False

try:
    from app.wisdom.topological_gauge_field_engine import (  # type: ignore
        HeytingOmega3,
        TopologicalGaugeFieldEngine,
        TopologicalGaugeFieldSnapshot,
    )
    _ENGINE_IMPORTED = True
except ImportError as _primary_import_error:
    try:
        from topological_gauge_field_engine import (  # type: ignore
            HeytingOmega3,
            TopologicalGaugeFieldEngine,
            TopologicalGaugeFieldSnapshot,
        )
        _ENGINE_IMPORTED = True
    except ImportError:
        _primary_import_error = _primary_import_error  # conserva el diagnóstico
        _ENGINE_IMPORTED = False


if not _ENGINE_IMPORTED:  # pragma: no cover — fallback para pruebas autónomas

    class HeytingOmega3(IntEnum):
        """Cadena de severidad: VETOED < DEGRADED < COHERENT."""

        VETOED = 0x00
        DEGRADED = 0x01
        COHERENT = 0x03

        def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return _omega_from_rank(max(_omega_rank(self), _omega_rank(other)))

        def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return _omega_from_rank(min(_omega_rank(self), _omega_rank(other)))

        def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
            # En una cadena, a -> b = ⊤ si a <= b; en otro caso b.
            if _omega_rank(self) <= _omega_rank(other):
                return HeytingOmega3.COHERENT
            return other

        def negate(self) -> "HeytingOmega3":
            # Negación de Heyting: a -> ⊥.
            return self.implies(HeytingOmega3.VETOED)

        @classmethod
        def fold_supremum(
            cls, verdicts: Sequence["HeytingOmega3"],
        ) -> "HeytingOmega3":
            acc = cls.VETOED
            for verdict in verdicts:
                acc = acc.join(verdict)
            return acc

        @classmethod
        def fold_infimum(
            cls, verdicts: Sequence["HeytingOmega3"],
        ) -> "HeytingOmega3":
            acc = cls.COHERENT
            for verdict in verdicts:
                acc = acc.meet(verdict)
            return acc

        @classmethod
        def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
            return cls.VETOED if int(legacy) == 0x02 else cls(int(legacy))

    class TopologicalGaugeFieldSnapshot:  # type: ignore[no-redef]
        """Marcador de tipo del fallback."""

    class TopologicalGaugeFieldEngine:  # type: ignore[no-redef]
        """
        Motor mínimo para ejecución local.

        No sustituye al motor espectral de producción ni simula una certificación
        física de hardware.
        """

        def __init__(
            self,
            dimension: int = 56,
            secret_key: bytes = b"APU_KEY",
        ) -> None:
            self.dimension = max(4, int(dimension))
            self.secret_key = bytes(secret_key)

        def sanitize_density_matrix(
            self, matrix: NDArray[Any],
        ) -> NDArray[np.complex128]:
            return _project_density_matrix(matrix)

        def build_gauge_connection_A(
            self, rho: NDArray[Any],
        ) -> NDArray[np.complex128]:
            rho_c = _project_density_matrix(rho)
            n = int(rho_c.shape[0])
            traceless = rho_c - (np.trace(rho_c) / n) * np.eye(
                n, dtype=np.complex128,
            )
            connection = 1j * traceless
            connection = 0.5 * (connection - connection.conj().T)
            connection -= (np.trace(connection) / n) * np.eye(
                n, dtype=np.complex128,
            )
            return np.asarray(connection, dtype=np.complex128)

        def observe_qnd_weak_measurement(
            self,
            phi_i: NDArray[Any],
            phi_f: NDArray[Any],
            observable: NDArray[Any],
        ) -> Tuple[complex, float, float]:
            """
            Devuelve (valor débil, retroacción reportada, varianza).

            La retroacción cero del fallback es una convención de prueba, no una
            medición de hardware.
            """
            initial = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
            final = np.asarray(phi_f, dtype=np.complex128).reshape(-1)
            operator = np.asarray(observable, dtype=np.complex128)
            overlap = np.vdot(final, initial)
            if not np.isfinite(overlap.real) or not np.isfinite(overlap.imag):
                return complex(float("nan"), float("nan")), float("nan"), float("nan")
            if abs(overlap) <= _WILKINSON_EPSILON:
                return complex(float("nan"), float("nan")), float("nan"), float("nan")

            weak_value = np.vdot(final, operator @ initial) / overlap
            mean = np.vdot(initial, operator @ initial)
            second = np.vdot(initial, operator @ operator @ initial)
            variance = float(np.real(second - mean * mean))
            return complex(weak_value), 0.0, max(0.0, variance)

        def trigger_hardware_interlock_esp32_crowbar(
            self,
            verdict: HeytingOmega3,
            gpio_pin: int = 14,
        ) -> Tuple[bool, float, str]:
            if verdict is HeytingOmega3.VETOED:
                return (
                    True,
                    42.0,
                    f"ESP32_CROWBAR_GPIO{int(gpio_pin)}_HIGH_BT151_TRIGGERED",
                )
            if verdict is HeytingOmega3.DEGRADED:
                return False, 120.0, "MONITORING_DEGRADED_PERFORMANCE"
            return False, 45.0, "NOMINAL_EXECUTION"

        def apply_visceral_functor_phi_sem(
            self,
            verdict: HeytingOmega3,
            energy: float,
            angle: float,
        ) -> Dict[str, Any]:
            common = {
                "llm_inference_savings_pct": 80.0,
                "kv_cache_compression_pct": 86.4,
                "yang_mills_energy_fpu": float(energy),
                "wilson_angle_rad": float(angle),
            }
            if verdict is HeytingOmega3.VETOED:
                return {
                    **common,
                    "wacc_protection_status": "BLINDADO AL 15.0%",
                    "net_liquidity_impact": "+11.5% EN TESORERÍA",
                    "executive_action": "DESEMBOLSO CERRADO",
                }
            if verdict is HeytingOmega3.DEGRADED:
                return {
                    **common,
                    "wacc_protection_status": "PRESERVADO EN 15.0%",
                    "net_liquidity_impact": "+8.5% RECIRCULANTE",
                    "executive_action": "VETO SUAVE",
                }
            return {
                **common,
                "wacc_protection_status": "ESTABLE AL 15.0%",
                "net_liquidity_impact": "+11.5% LIBRE",
                "executive_action": "FLUJO NOMINAL",
            }

        def evaluate_topological_gauge_field(
            self,
            rho: NDArray[Any],
            p_i: Any = None,
            p_f: Any = None,
            **_kwargs: Any,
        ) -> Tuple[Any, Dict[str, Any]]:
            class _DummySnapshot:
                snapshot_id = "GAUGE-DUMMY"
                yang_mills_energy = 0.001
                chern_simons_3form_higgs = 0.002
                chern_simons_anomaly = 0.0
                chern_hopf_class_c1 = 0.0
                wilson_loop_trace = 1.0 + 0.0j
                wilson_angle_rad = 0.0
                pontryagin_instanton_number = 0
                pontryagin_integrality_residual = 0.0
                stora_zumino_residual = 0.0
                weak_value_qnd_aav = 0.0 + 0.0j
                back_action_db = 0.0
                qnd_variance = 0.0
                qnd_valid = True
                topology_regime = "TRIVIAL_SU2_HOLONOMY"
                global_verdict = HeytingOmega3.COHERENT
                sha256_merkle_digest = "fallback"
                timestamp_utc = time.time()
                kam_worst_ratio = 0.0
                kam_satisfied = True
                bruno_satisfied = True
                melnikov_value = 0.0
                floquet_max_unit_deviation = 0.0
                floquet_is_neutral = True
                poincare_section_n_crossings = 0
                poincare_section_dispersion = 0.0
                mean_return_time = 0.0
                recurrence_time = 0.0
                poincare_data_valid = True
                poincare_map_symplectic_valid = True
                poincare_symplectic_defect = 0.0
                poincare_transversal = True
                banach_lipschitz_t3 = 0.0
                fubini_study_distance = 0.0
                novikov_krull_valuation = 1.0
                rsi3_valid = True
                rsi3_iterations = 1
                symplectic_residue = 0.0
                symplectic_defect = 0.0
                crowbar_latency_ns = 42.0
                hardware_crowbar_action = "FALLBACK"
                spectral_gap = 0.0
                poincare_cartan_invariant = 0.0
                chern_simons_algebraic = 0.002
                lindstedt_residual = 0.0

            snapshot = _DummySnapshot()
            return snapshot, self.apply_visceral_functor_phi_sem(
                HeytingOmega3.COHERENT, 0.001, 0.0,
            )


_ENGINE_HAS_ENVELOPES: bool = bool(
    _ENGINE_IMPORTED
    and hasattr(TopologicalGaugeFieldEngine, "emit_phase1_envelope")
    and hasattr(TopologicalGaugeFieldEngine, "emit_phase2_envelope")
)
_ENGINE_HAS_CONTINUE: bool = bool(
    _ENGINE_IMPORTED
    and hasattr(TopologicalGaugeFieldEngine, "continue_to_phase2")
)
_ENGINE_HAS_GOVERNANCE: bool = bool(
    _ENGINE_IMPORTED
    and hasattr(TopologicalGaugeFieldEngine, "emit_governance_state")
)

logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldAgent.v12")


# ══════════════════════════════════════════════════════════════════════════════
# §1. CONSTANTES, EXCEPCIONES Y PRIMITIVAS NUMÉRICAS/CRIPTOGRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

_AGENT_VERSION: Final[str] = "12.0.0"
_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-TGFA-v12.0.0|"
_DEFAULT_SECRET_KEY: Final[bytes] = (
    b"APU_FILTER_GAUGE_FIELD_SOVEREIGN_SECRET_KEY"
)

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_TRACE_CONSERVATION_TOL: Final[float] = 1.0e-10
_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_UNITARY_TOL: Final[float] = 1.0e-10

_NORM_A_SOFT: Final[float] = 10.0
_NORM_A_HARD: Final[float] = 100.0
_DENSITY_REPAIR_SOFT: Final[float] = 1.0e-6
_DENSITY_REPAIR_HARD: Final[float] = 0.25

_RSI3_MAX_ITER: Final[int] = 3
_RSI3_STEP_LIP_MAX: Final[float] = 0.95
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_BANACH_LIPSCHITZ_T3_MAX: Final[float] = _BANACH_LIPSCHITZ_MAX ** 3
_LINSTEDT_DAMP: Final[float] = 0.15
_LINSTEDT_RESIDUAL_TOL: Final[float] = 1.0e-9

_MELNIKOV_TOL: Final[float] = 0.5
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_TOPOLOGY_RESIDUAL_TOL: Final[float] = 1.0e-6
_QND_VARIANCE_TOL: Final[float] = 1.0e-10
_QND_BACK_ACTION_TOL_DB: Final[float] = 1.0e-6

_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0
_POSITRON_REST_MASS_ENERGY_KEV: Final[float] = 1022.0


class GaugeFieldAgentError(Exception):
    """Excepción raíz del agente de campo de gauge."""


class GaugeContractCorruptedError(GaugeFieldAgentError):
    """Firma HMAC, compromiso de datos o enlace de fase inválido."""


class GaugeIsometryViolationError(GaugeFieldAgentError):
    """Violación de un invariante de densidad o conexión de gauge."""


class SymplecticGaugeInvarianceError(GaugeFieldAgentError):
    """Violación numérica de la condición simpléctica."""


class EnvelopeChainViolationError(GaugeFieldAgentError):
    """La cadena de certificados/envelopes entre fases es inconsistente."""


class PoincareLindstedtResonanceError(GaugeFieldAgentError):
    """Divisores pequeños o datos de Lindstedt no resolubles."""


class RSI3ContractionFailureError(GaugeFieldAgentError):
    """La estimación observada de contracción excede el umbral configurado."""


def _omega_rank(verdict: HeytingOmega3) -> int:
    """
    Devuelve el rango semántico, independiente de los enteros de la enum externa.

    Rango 0 = VETOED; 1 = DEGRADED; 2 = COHERENT.
    """
    name = getattr(verdict, "name", None)
    by_name = {"VETOED": 0, "DEGRADED": 1, "COHERENT": 2}
    if name in by_name:
        return by_name[name]

    for candidate, rank in (
        (HeytingOmega3.VETOED, 0),
        (HeytingOmega3.DEGRADED, 1),
        (HeytingOmega3.COHERENT, 2),
    ):
        if verdict == candidate:
            return rank
    raise ValueError(f"Estado Heyting desconocido: {verdict!r}")


def _omega_from_rank(rank: int) -> HeytingOmega3:
    rank = int(rank)
    if rank <= 0:
        return HeytingOmega3.VETOED
    if rank == 1:
        return HeytingOmega3.DEGRADED
    return HeytingOmega3.COHERENT


def _omega_meet(left: HeytingOmega3, right: HeytingOmega3) -> HeytingOmega3:
    """Ínfimo conservador: toma el estado de mayor severidad."""
    return _omega_from_rank(min(_omega_rank(left), _omega_rank(right)))


def _omega_join(left: HeytingOmega3, right: HeytingOmega3) -> HeytingOmega3:
    """Supremo lógico de la cadena de estados."""
    return _omega_from_rank(max(_omega_rank(left), _omega_rank(right)))


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_complex(value: Any, default: complex = complex(float("nan"), float("nan"))) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError):
        return complex(default)
    return result


def _g(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _project_density_with_diagnostics(
    matrix: NDArray[Any],
) -> Tuple[NDArray[np.complex128], float, float]:
    """
    Proyecta una matriz finita a una matriz de densidad.

    Devuelve (rho, norma_de_reparación_relativa, autovalor_mínimo_final).
    Una matriz nula o completamente no positiva se sustituye por I/n; la
    reparación queda reflejada en la norma informada.
    """
    raw = np.asarray(matrix, dtype=np.complex128)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
        raise ValueError("La matriz de densidad debe ser cuadrada y no vacía.")
    if not np.all(np.isfinite(raw.real)) or not np.all(np.isfinite(raw.imag)):
        raise ValueError("La matriz de densidad contiene NaN o infinito.")

    hermitian = 0.5 * (raw + raw.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(clipped))

    if not math.isfinite(total) or total <= _SPECTRAL_PSD_FLOOR:
        rho = np.eye(raw.shape[0], dtype=np.complex128) / float(raw.shape[0])
    else:
        rho = (eigenvectors * clipped) @ eigenvectors.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        trace = float(np.real(np.trace(rho)))
        if not math.isfinite(trace) or trace <= _SPECTRAL_PSD_FLOOR:
            rho = np.eye(raw.shape[0], dtype=np.complex128) / float(raw.shape[0])
        else:
            rho /= trace

    denominator = max(1.0, float(la.norm(raw, "fro")))
    repair = float(la.norm(rho - raw, "fro")) / denominator
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))))
    return np.asarray(rho, dtype=np.complex128), repair, min_eigenvalue


def _project_density_matrix(matrix: NDArray[Any]) -> NDArray[np.complex128]:
    """Proyección local de una matriz a densidad positiva, hermítica y de traza uno."""
    return _project_density_with_diagnostics(matrix)[0]


def _matrix_hash(matrix: NDArray[Any]) -> str:
    """Hash de matriz complejo normalizado a complex128."""
    return _hash_state(np.asarray(matrix, dtype=np.complex128))


def _hash_state(*arrays: NDArray[Any]) -> str:
    """
    Hash sin ambigüedad de forma/tipo para arrays contiguos.

    El prefijo de cada array incluye dtype, shape y longitud; no basta con
    concatenar bytes crudos porque formas distintas pueden compartir bytes.
    """
    digest = hashlib.sha256()
    digest.update(b"APU-TGFA-ARRAY-STATE-v2|")
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array))
        if contiguous.dtype.hasobject:
            raise TypeError("No se permiten arrays object en el hash de estado.")
        metadata = json.dumps(
            {
                "dtype": contiguous.dtype.str,
                "shape": tuple(int(x) for x in contiguous.shape),
                "nbytes": int(contiguous.nbytes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _hash_text(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"APU-TGFA-TEXT-COMMITMENT-v2|")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _float_token(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()


def _canonical_value(
    value: Any,
    seen: Optional[set[int]] = None,
    depth: int = 0,
) -> Any:
    """Conversión determinista a una estructura serializable para compromisos."""
    if seen is None:
        seen = set()
    if depth > 32:
        return {"truncated": type(value).__qualname__}

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return {"float": _float_token(float(value))}
    if isinstance(value, (complex, np.complexfloating)):
        z = complex(value)
        return {"complex": [_float_token(z.real), _float_token(z.imag)]}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_hex": bytes(value).hex()}
    if isinstance(value, IntEnum):
        return {
            "enum": type(value).__qualname__,
            "name": getattr(value, "name", ""),
            "value": int(value),
        }
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), seen, depth + 1)

    identity = id(value)
    if identity in seen:
        return {"cycle": type(value).__qualname__}

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            return {
                "ndarray_object": _canonical_value(array.tolist(), seen, depth + 1),
                "shape": tuple(int(x) for x in array.shape),
            }
        return {
            "ndarray_dtype": array.dtype.str,
            "ndarray_shape": tuple(int(x) for x in array.shape),
            "ndarray_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    seen.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field_info.name: _canonical_value(
                    getattr(value, field_info.name), seen, depth + 1,
                )
                for field_info in fields(value)
                if field_info.name not in {"_lock", "_logger"}
            }
        if isinstance(value, Mapping):
            return {
                str(key): _canonical_value(value[key], seen, depth + 1)
                for key in sorted(value, key=lambda item: str(item))
            }
        if isinstance(value, (tuple, list)):
            return [_canonical_value(item, seen, depth + 1) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [_canonical_value(item, seen, depth + 1) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )

        attributes: Dict[str, Any] = {}
        if hasattr(value, "__dict__"):
            attributes.update(
                {
                    str(key): item
                    for key, item in vars(value).items()
                    if not str(key).startswith("_") and not callable(item)
                }
            )
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if (
                isinstance(slot, str)
                and not slot.startswith("_")
                and hasattr(value, slot)
                and slot not in attributes
            ):
                item = getattr(value, slot)
                if not callable(item):
                    attributes[slot] = item
        if attributes:
            return {
                "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "attributes": {
                    key: _canonical_value(attributes[key], seen, depth + 1)
                    for key in sorted(attributes)
                },
            }

        # No se usa repr(value): puede incluir direcciones de memoria no estables.
        return {"opaque_type": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        seen.discard(identity)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _engine_envelope_digest(envelope: Any) -> str:
    if envelope is None:
        return ""
    payload = _canonical_json(envelope).encode("utf-8")
    return hashlib.sha256(b"APU-TGFA-ENGINE-ENVELOPE-v2|" + payload).hexdigest()


def _phase1_chain_hash(cert: "GaugeFieldIsometryCertificate") -> str:
    return _hash_text(
        "PHASE1-CHAIN-v2",
        cert.rho_state_hash,
        cert.connection_state_hash,
        cert.engine_envelope_digest,
    )


def _phase2_chain_hash(cert: "Phase2GaugeDynamicsCertificate") -> str:
    excluded = {
        "hmac_phase2_signature",
        "chain_hash",
        "engine_envelope",
    }
    body = {
        item.name: getattr(cert, item.name)
        for item in fields(cert)
        if item.name not in excluded
    }
    return hashlib.sha256(
        b"APU-TGFA-PHASE2-CHAIN-v2|"
        + _canonical_json(body).encode("utf-8")
    ).hexdigest()


def _unpack_qnd(result: Any) -> Tuple[complex, float, float]:
    """Normaliza motores que devuelven (A_w,dB,var), (A_w,dB) o sólo A_w."""
    if not isinstance(result, (tuple, list)):
        return _safe_complex(result), float("nan"), float("nan")
    if len(result) >= 3:
        return (
            _safe_complex(result[0]),
            _safe_float(result[1]),
            _safe_float(result[2]),
        )
    if len(result) == 2:
        return _safe_complex(result[0]), _safe_float(result[1]), float("nan")
    return complex(float("nan"), float("nan")), float("nan"), float("nan")


def _unpack_crowbar(result: Any) -> Tuple[bool, float, str]:
    """Normaliza las respuestas históricas del adaptador Crowbar."""
    if not isinstance(result, (tuple, list)):
        return bool(result), float("nan"), "UNKNOWN"
    if len(result) >= 3:
        return bool(result[0]), _safe_float(result[1]), str(result[2])
    if len(result) == 2:
        latched, latency = bool(result[0]), _safe_float(result[1])
        if latched:
            action = "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
        elif math.isfinite(latency) and latency >= 100.0:
            action = "MONITORING_DEGRADED_PERFORMANCE"
        else:
            action = "NOMINAL_EXECUTION"
        return latched, latency, action
    return False, float("nan"), "UNKNOWN"


def _call_emit_phase1(
    engine: Any,
    rho: NDArray[Any],
    background_hamiltonian: Optional[NDArray[Any]],
) -> Any:
    method = getattr(engine, "emit_phase1_envelope")
    try:
        return method(rho, background_hamiltonian=background_hamiltonian)
    except TypeError as exc:
        # Sólo se reintenta por incompatibilidad de firma, no por un TypeError
        # interno arbitrario del motor.
        message = str(exc)
        if "background_hamiltonian" not in message and "keyword" not in message:
            raise
        return method(rho)


def _call_evaluate(
    engine: Any,
    rho: NDArray[Any],
    background_hamiltonian: Optional[NDArray[Any]],
) -> Tuple[Any, Dict[str, Any]]:
    method = getattr(engine, "evaluate_topological_gauge_field")
    try:
        result = method(rho, background_hamiltonian=background_hamiltonian)
    except TypeError as exc:
        message = str(exc)
        if "background_hamiltonian" not in message and "keyword" not in message:
            raise
        result = method(rho)

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return result[0], dict(result[1] or {})
    return result, {}


def _symplectic_matrix_defect(matrix: NDArray[Any]) -> float:
    """
    Calcula ||MᵀJM-J||_F / max(1,||M||_F²) para una matriz real 2n×2n.
    """
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        return float("inf")
    if raw.shape[0] % 2:
        return float("inf")
    if np.iscomplexobj(raw):
        if float(la.norm(np.imag(raw), "fro")) > _SYMPLECTIC_TOL:
            return float("inf")
        raw = np.real(raw)
    m = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(m)):
        return float("inf")
    n = m.shape[0] // 2
    identity = np.eye(n, dtype=np.float64)
    zero = np.zeros((n, n), dtype=np.float64)
    canonical_j = np.block([[zero, identity], [-identity, zero]])
    defect = m.T @ canonical_j @ m - canonical_j
    denominator = max(1.0, float(la.norm(m, "fro")) ** 2)
    return float(la.norm(defect, "fro")) / denominator


@dataclass(frozen=True, slots=True)
class _PoincareDiagnostics:
    """Diagnósticos observados de una sección/mapa de Poincaré."""

    n_crossings: int = 0
    return_times: Tuple[float, ...] = ()
    mean_return_time: float = 0.0
    recurrence_time: float = 0.0
    return_dispersion: float = 0.0
    data_valid: bool = False
    map_symplectic_defect: float = float("inf")
    map_symplectic_valid: bool = False
    transversal: bool = False
    valid: bool = False


def _poincare_diagnostics(section: Any) -> _PoincareDiagnostics:
    """
    Normaliza retornos y, si existe, verifica el jacobiano del mapa de retorno.

    La recurrencia aquí es empírica: no se infiere una demostración del teorema
    de recurrencia de Poincaré a partir de una serie finita.
    """
    if section is None:
        return _PoincareDiagnostics()

    n_crossings = _safe_int(
        _g(section, "n_crossings", _g(section, "poincare_section_n_crossings", -1)),
        -1,
    )
    raw_times = None
    for candidate in ("return_times", "inter_return_times", "flight_times"):
        value = _g(section, candidate, None)
        if value is not None:
            raw_times = value
            break

    times: Tuple[float, ...] = ()
    time_data_valid = True
    if raw_times is not None:
        try:
            array = np.asarray(raw_times, dtype=np.float64).reshape(-1)
            times = tuple(float(x) for x in array)
            time_data_valid = bool(
                np.all(np.isfinite(array)) and np.all(array > 0.0)
            )
        except (TypeError, ValueError):
            time_data_valid = False

    mean_return = _safe_float(
        _g(section, "mean_return_time", _g(section, "mean_return", float("nan"))),
    )
    recurrence = _safe_float(
        _g(section, "recurrence_time", _g(section, "first_recurrence_time", float("nan"))),
    )
    dispersion = _safe_float(
        _g(section, "return_dispersion", _g(section, "poincare_section_dispersion", float("nan"))),
    )

    if times:
        array = np.asarray(times, dtype=np.float64)
        empirical_mean = float(np.mean(array))
        empirical_recurrence = float(np.min(array))
        empirical_dispersion = (
            float(np.std(array) / empirical_mean)
            if empirical_mean > _WILKINSON_EPSILON
            else float("inf")
        )
        if not math.isfinite(mean_return):
            mean_return = empirical_mean
        if not math.isfinite(recurrence):
            recurrence = empirical_recurrence
        if not math.isfinite(dispersion):
            dispersion = empirical_dispersion

    if n_crossings < 0:
        n_crossings = len(times)

    explicit_data_valid = _g(
        section, "poincare_data_valid", _g(section, "data_valid", None),
    )
    finite_return_data = (
        math.isfinite(mean_return)
        and math.isfinite(recurrence)
        and math.isfinite(dispersion)
        and mean_return >= 0.0
        and recurrence >= 0.0
        and dispersion >= 0.0
    )
    if explicit_data_valid is None:
        data_valid = bool(
            time_data_valid
            and finite_return_data
            and (bool(times) or n_crossings == 0)
        )
    else:
        data_valid = bool(explicit_data_valid) and time_data_valid and finite_return_data

    matrix = None
    for candidate in (
        "return_map_jacobian",
        "poincare_map_jacobian",
        "monodromy_matrix",
        "return_map_matrix",
    ):
        matrix = _g(section, candidate, None)
        if matrix is not None:
            break

    if matrix is not None:
        map_defect = _symplectic_matrix_defect(np.asarray(matrix))
        map_valid = math.isfinite(map_defect) and map_defect <= _SYMPLECTIC_TOL
    else:
        map_defect = _safe_float(
            _g(
                section,
                "poincare_symplectic_defect",
                _g(section, "symplectic_defect", float("inf")),
            ),
        )
        explicit_map_valid = _g(
            section,
            "poincare_map_symplectic_valid",
            _g(section, "map_is_symplectic", None),
        )
        map_valid = bool(explicit_map_valid) and (
            math.isfinite(map_defect) and map_defect <= _SYMPLECTIC_TOL
        )

    transversal = bool(
        _g(section, "poincare_transversal", _g(section, "is_transversal", False))
    )
    valid = bool(data_valid and map_valid and transversal)

    return _PoincareDiagnostics(
        n_crossings=max(0, n_crossings),
        return_times=times,
        mean_return_time=mean_return if math.isfinite(mean_return) else 0.0,
        recurrence_time=recurrence if math.isfinite(recurrence) else 0.0,
        return_dispersion=dispersion if math.isfinite(dispersion) else 0.0,
        data_valid=data_valid,
        map_symplectic_defect=map_defect,
        map_symplectic_valid=map_valid,
        transversal=transversal,
        valid=valid,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §1.A. ADJUDICADOR HEYTING Ω₃
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3Adjudicator:
    """
    Retículo de tres estados con semántica conservadora explícita.

    Orden de severidad: VETOED < DEGRADED < COHERENT.
    `meet_atoms` fusiona restricciones de gobernanza; `join_atoms` representa
    el supremo matemático, pero no debe usarse para borrar un veto acumulado.
    """

    __slots__ = ()

    @staticmethod
    def fold_supremum(verdicts: Sequence[HeytingOmega3]) -> HeytingOmega3:
        if not verdicts:
            return HeytingOmega3.VETOED
        return _omega_from_rank(max(_omega_rank(v) for v in verdicts))

    @staticmethod
    def fold_infimum(verdicts: Sequence[HeytingOmega3]) -> HeytingOmega3:
        if not verdicts:
            return HeytingOmega3.COHERENT
        return _omega_from_rank(min(_omega_rank(v) for v in verdicts))

    @classmethod
    def adjudicate(cls, *atoms: HeytingOmega3) -> HeytingOmega3:
        """Adjudicación fail-closed: ínfimo de los átomos."""
        return cls.fold_infimum(atoms)

    @staticmethod
    def join_atoms(atoms: Sequence[HeytingOmega3]) -> HeytingOmega3:
        return HeytingOmega3Adjudicator.fold_supremum(atoms)

    @staticmethod
    def meet_atoms(atoms: Sequence[HeytingOmega3]) -> HeytingOmega3:
        return HeytingOmega3Adjudicator.fold_infimum(atoms)

    @staticmethod
    def truth_table_join() -> Tuple[Tuple[int, int, int], ...]:
        values = (
            HeytingOmega3.VETOED,
            HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT,
        )
        return tuple(
            tuple(int(_omega_join(a, b)) for b in values)
            for a in values
        )

    @staticmethod
    def truth_table_meet() -> Tuple[Tuple[int, int, int], ...]:
        values = (
            HeytingOmega3.VETOED,
            HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT,
        )
        return tuple(
            tuple(int(_omega_meet(a, b)) for b in values)
            for a in values
        )

    @staticmethod
    def truth_table_implies() -> Tuple[Tuple[int, int, int], ...]:
        values = (
            HeytingOmega3.VETOED,
            HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT,
        )
        top = HeytingOmega3.COHERENT
        rows = []
        for left in values:
            row = []
            for right in values:
                result = top if _omega_rank(left) <= _omega_rank(right) else right
                row.append(int(result))
            rows.append(tuple(row))
        return tuple(rows)

    @staticmethod
    def truth_table_negate() -> Tuple[int, int, int]:
        values = (
            HeytingOmega3.VETOED,
            HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT,
        )
        bottom = HeytingOmega3.VETOED
        top = HeytingOmega3.COHERENT
        return tuple(
            int(top if _omega_rank(value) == 0 else bottom)
            for value in values
        )  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# §1.B. CERTIFICADOS INMUTABLES Y ESTADO TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GaugeFieldIsometryCertificate:
    """FASE 1: isometría, estado de densidad, Lindstedt, envelope y HMAC."""

    registry_size: int
    dimension: int
    connection_norm_frobenius: float
    density_trace: float
    is_trace_conserved: bool
    phase1_verdict: HeytingOmega3
    hmac_phase1_signature: str

    is_anti_hermitian_A: bool = False
    is_traceless_A: bool = False
    symplectic_residue: float = float("inf")
    symplectic_defect: float = float("inf")
    is_symplectic_valid: bool = False
    poincare_residual: float = float("inf")
    poincare_is_resonant: bool = False
    energy_ym_baseline: float = 0.0
    engine_envelope: Optional[Any] = None
    chain_hash: str = ""

    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    is_kahler: bool = False
    is_quaternionic_proxy: bool = False
    cayley_unitary_defect: float = float("inf")
    lindstedt_is_solved: bool = False
    n_small_divisors: int = 0
    rsi3_iterations: int = 0
    rsi3_contraction: float = float("inf")
    symplectic_certified: bool = False

    density_min_eigenvalue: float = 0.0
    density_repair_norm: float = 0.0
    rsi3_valid: bool = False
    rsi3_residual_history: Tuple[float, ...] = ()
    rho_state_hash: str = ""
    connection_state_hash: str = ""
    engine_envelope_digest: str = ""


@dataclass(frozen=True, slots=True)
class Phase2GaugeDynamicsCertificate:
    """FASE 2: dinámica, invariantes topológicos, Poincaré, QND y HMAC."""

    yang_mills_energy: float
    chern_simons_3form_higgs: float
    wilson_angle_rad: float
    instanton_k: int
    stora_zumino_residual: float
    weak_value_qnd: complex
    positron_annihilations_count: int
    phase2_verdict: HeytingOmega3
    hmac_phase2_signature: str

    chern_simons_anomaly: float = 0.0
    c1_value: float = 0.0
    pontryagin_integrality_residual: float = 0.0
    kam_worst_ratio: float = float("inf")
    kam_satisfied: bool = False
    melnikov_value: float = float("inf")
    melnikov_preserved: bool = False
    floquet_max_unit_deviation: float = float("inf")
    floquet_neutral: bool = False
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    rsi3_valid: bool = False
    rsi3_iterations: int = 0
    banach_lipschitz_t3: float = float("inf")
    fubini_study_distance: float = float("inf")
    novikov_krull_valuation: float = float("-inf")
    back_action_db: float = float("inf")
    topology_regime: str = "UNKNOWN"
    engine_envelope: Optional[Any] = None
    chain_hash: str = ""

    dimension: int = 0
    bruno_satisfied: bool = False
    bruno_sum: float = float("inf")
    chern_simons_algebraic: float = 0.0
    chern_character_ch2: float = 0.0
    krein_collision: bool = False
    qnd_variance: float = float("inf")
    mean_return_time: float = 0.0
    recurrence_time: float = 0.0
    rsi3_contraction: float = float("inf")

    parent_phase1_chain_hash: str = ""
    parent_phase1_signature: str = ""
    engine_envelope_digest: str = ""
    qnd_valid: bool = False
    qnd_overlap_abs: float = 0.0
    poincare_data_valid: bool = False
    poincare_map_symplectic_valid: bool = False
    poincare_symplectic_defect: float = float("inf")
    poincare_transversal: bool = False
    poincare_return_times: Tuple[float, ...] = ()
    floquet_krein_collision: bool = False


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldSovereignState:
    """FASE 3: veredicto final, acción reportada y compromiso criptográfico."""

    phase1_cert: GaugeFieldIsometryCertificate
    phase2_cert: Phase2GaugeDynamicsCertificate
    global_verdict: HeytingOmega3
    crowbar_triggered: bool
    crowbar_latency_ns: float
    visceral_report: Dict[str, Any]
    sha256_merkle_digest: str
    provenance_hmac: str
    timestamp_utc: float

    crowbar_latency_verified: bool = False
    hardware_action: str = "UNKNOWN"
    heyting_join_atoms: Tuple[int, ...] = ()
    heyting_meet_atoms: Tuple[int, ...] = ()
    heyting_implies_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_negate_table: Tuple[int, int, int] = (0, 0, 0)
    is_epistemologically_valid: bool = False
    agent_version: str = _AGENT_VERSION

    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    bruno_satisfied: bool = False
    lindstedt_residual: float = 0.0
    qnd_variance: float = float("inf")
    hmac_verified: bool = False

    phase_chain_valid: bool = False
    crowbar_acknowledged: bool = False
    qnd_valid: bool = False
    poincare_valid: bool = False
    rsi3_valid: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — OBSERVE                                                          ║
# ║ Sanitización de ρ, conexión, envelope, RSI³-Lindstedt y firma HMAC.        ║
# ║ El último método es continue_to_phase2, puente autenticado hacia FASE 2.  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase1Observer:
    """
    Observador de fase 1.

    La conjugación RSI³ mantiene la positividad y la traza en aritmética exacta
    cuando el generador es antihermítico. En cómputo finito se verifica el
    defecto de unitariedad y se vuelve a proyectar ρ al cono de densidades.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)
        if not self.secret_key:
            raise ValueError("secret_key no puede ser vacío.")

        if background_hamiltonian is None:
            self._background_H: Optional[NDArray[np.complex128]] = None
        else:
            background = np.asarray(background_hamiltonian, dtype=np.complex128)
            if (
                background.ndim != 2
                or background.shape[0] != background.shape[1]
                or not np.all(np.isfinite(background.real))
                or not np.all(np.isfinite(background.imag))
            ):
                raise ValueError("background_hamiltonian debe ser cuadrado y finito.")
            self._background_H = np.array(background, copy=True)

        self.engine: Final[Any] = TopologicalGaugeFieldEngine(
            dimension=self.dimension,
            secret_key=self.secret_key,
        )
        self._phase1_cert: Optional[GaugeFieldIsometryCertificate] = None
        self._phase2_cert: Optional[Phase2GaugeDynamicsCertificate] = None
        self._adjudicator = HeytingOmega3Adjudicator()

    # ──────────────────────────────────────────────────────────────────────
    # 1.1. Serialización y autenticación de certificados
    # ──────────────────────────────────────────────────────────────────────

    def _compute_hmac(self, payload: str) -> str:
        return hmac.new(
            self.secret_key,
            _HMAC_DOMAIN + payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _verify_hmac(self, payload: str, signature: str) -> bool:
        if not isinstance(signature, str) or not signature:
            return False
        return hmac.compare_digest(self._compute_hmac(payload), signature)

    @staticmethod
    def _phase1_payload(cert: GaugeFieldIsometryCertificate) -> str:
        excluded = {"hmac_phase1_signature", "engine_envelope"}
        body = {
            item.name: getattr(cert, item.name)
            for item in fields(cert)
            if item.name not in excluded
        }
        return "P1:" + _canonical_json(body)

    @staticmethod
    def _phase2_payload(cert: Phase2GaugeDynamicsCertificate) -> str:
        excluded = {"hmac_phase2_signature", "engine_envelope"}
        body = {
            item.name: getattr(cert, item.name)
            for item in fields(cert)
            if item.name not in excluded
        }
        return "P2:" + _canonical_json(body)

    def _sign_phase1(
        self, cert: GaugeFieldIsometryCertificate,
    ) -> GaugeFieldIsometryCertificate:
        envelope_digest = _engine_envelope_digest(cert.engine_envelope)
        if cert.engine_envelope_digest != envelope_digest:
            raise GaugeContractCorruptedError(
                "El digest del envelope de fase 1 no coincide con su contenido."
            )
        chain_hash = _phase1_chain_hash(cert)
        unsigned = replace(
            cert,
            chain_hash=chain_hash,
            hmac_phase1_signature="",
        )
        return replace(
            unsigned,
            hmac_phase1_signature=self._compute_hmac(
                self._phase1_payload(unsigned),
            ),
        )

    def _sign_phase2(
        self, cert: Phase2GaugeDynamicsCertificate,
    ) -> Phase2GaugeDynamicsCertificate:
        envelope_digest = _engine_envelope_digest(cert.engine_envelope)
        if cert.engine_envelope_digest != envelope_digest:
            raise GaugeContractCorruptedError(
                "El digest del envelope de fase 2 no coincide con su contenido."
            )
        chained = replace(cert, chain_hash="", hmac_phase2_signature="")
        chained = replace(chained, chain_hash=_phase2_chain_hash(chained))
        return replace(
            chained,
            hmac_phase2_signature=self._compute_hmac(
                self._phase2_payload(chained),
            ),
        )

    def _verify_phase1_certificate(
        self, cert: GaugeFieldIsometryCertificate,
    ) -> None:
        if not isinstance(cert, GaugeFieldIsometryCertificate):
            raise GaugeContractCorruptedError("Tipo de certificado de fase 1 inválido.")
        if not self._verify_hmac(
            self._phase1_payload(cert), cert.hmac_phase1_signature,
        ):
            raise GaugeContractCorruptedError(
                "HMAC de fase 1 inválida o certificado alterado."
            )
        if _engine_envelope_digest(cert.engine_envelope) != cert.engine_envelope_digest:
            raise GaugeContractCorruptedError(
                "El envelope de fase 1 fue modificado después de firmarse."
            )
        if _phase1_chain_hash(cert) != cert.chain_hash:
            raise EnvelopeChainViolationError(
                "El hash de cadena de fase 1 no es consistente."
            )

        if cert.engine_envelope is not None:
            envelope_rho = _g(cert.engine_envelope, "rho_sanitized", None)
            envelope_A = _g(cert.engine_envelope, "connection_A", None)
            if envelope_rho is not None:
                if _matrix_hash(np.asarray(envelope_rho)) != cert.rho_state_hash:
                    raise EnvelopeChainViolationError(
                        "ρ del envelope no coincide con el compromiso firmado."
                    )
            if envelope_A is not None:
                if _matrix_hash(np.asarray(envelope_A)) != cert.connection_state_hash:
                    raise EnvelopeChainViolationError(
                        "A del envelope no coincide con el compromiso firmado."
                    )

    def _verify_phase2_certificate(
        self, cert: Phase2GaugeDynamicsCertificate,
    ) -> None:
        if not isinstance(cert, Phase2GaugeDynamicsCertificate):
            raise GaugeContractCorruptedError("Tipo de certificado de fase 2 inválido.")
        if not self._verify_hmac(
            self._phase2_payload(cert), cert.hmac_phase2_signature,
        ):
            raise GaugeContractCorruptedError(
                "HMAC de fase 2 inválida o certificado alterado."
            )
        if _engine_envelope_digest(cert.engine_envelope) != cert.engine_envelope_digest:
            raise GaugeContractCorruptedError(
                "El envelope de fase 2 fue modificado después de firmarse."
            )
        if _phase2_chain_hash(cert) != cert.chain_hash:
            raise EnvelopeChainViolationError(
                "El hash de cadena de fase 2 no es consistente."
            )
        if not cert.parent_phase1_chain_hash or not cert.parent_phase1_signature:
            raise EnvelopeChainViolationError(
                "El certificado de fase 2 carece del enlace a fase 1."
            )

    # ──────────────────────────────────────────────────────────────────────
    # 1.2. Sanitización de estado y consistencia dimensional
    # ──────────────────────────────────────────────────────────────────────

    def _sanitize_density(
        self, density_matrix: NDArray[Any],
    ) -> Tuple[NDArray[np.complex128], float, float]:
        raw = np.asarray(density_matrix, dtype=np.complex128)
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
            raise ValueError("density_matrix debe ser cuadrada y no vacía.")
        if self._background_H is not None and self._background_H.shape != raw.shape:
            raise ValueError(
                "background_hamiltonian y density_matrix deben tener la misma forma."
            )

        local, local_repair, local_min_eig = _project_density_with_diagnostics(raw)
        try:
            engine_result = self.engine.sanitize_density_matrix(local)
            candidate = np.asarray(engine_result, dtype=np.complex128)
            if candidate.shape != local.shape:
                raise ValueError("El sanitizador del motor cambió la dimensión de ρ.")
            sanitized, engine_repair, min_eig = _project_density_with_diagnostics(
                candidate,
            )
            total_repair = float(la.norm(sanitized - raw, "fro")) / max(
                1.0, float(la.norm(raw, "fro")),
            )
            return sanitized, max(local_repair, engine_repair, total_repair), min_eig
        except Exception as exc:
            logger.warning("Sanitizador del motor no disponible; proyección local: %s", exc)
            return local, local_repair, local_min_eig

    @staticmethod
    def _antihermitian_diagnostics(
        connection: NDArray[Any],
    ) -> Tuple[float, float, float]:
        matrix = np.asarray(connection, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return float("inf"), float("inf"), float("inf")
        if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
            return float("inf"), float("inf"), float("inf")
        scale = max(1.0, float(la.norm(matrix, "fro")))
        anti_residual = float(la.norm(matrix + matrix.conj().T, "fro")) / scale
        trace_residual = float(abs(np.trace(matrix))) / scale
        norm = float(la.norm(matrix, "fro"))
        return anti_residual, trace_residual, norm

    def _check_phase1_density_matches(
        self,
        cert1: GaugeFieldIsometryCertificate,
        rho: NDArray[np.complex128],
    ) -> None:
        envelope = cert1.engine_envelope
        envelope_rho = _g(envelope, "rho_sanitized", None) if envelope is not None else None
        if envelope_rho is not None:
            reference = _project_density_matrix(np.asarray(envelope_rho))
            if reference.shape != rho.shape or not np.allclose(
                reference, rho, rtol=1.0e-10, atol=1.0e-12,
            ):
                raise EnvelopeChainViolationError(
                    "La densidad de fase 2 no es la densidad firmada en fase 1."
                )
        elif _matrix_hash(rho) != cert1.rho_state_hash:
            raise EnvelopeChainViolationError(
                "La densidad de entrada no coincide con el hash firmado en fase 1."
            )

    # ──────────────────────────────────────────────────────────────────────
    # 1.3. RSI³: conjugación unitaria y estimación a posteriori
    # ──────────────────────────────────────────────────────────────────────

    def _rsi3_lindstedt_reobserve(
        self,
        rho: NDArray[np.complex128],
        envelope: Any,
    ) -> Tuple[
        NDArray[np.complex128],
        Any,
        List[float],
        float,
        int,
        bool,
        float,
    ]:
        """
        Aplica hasta tres conjugaciones RSI³ derivadas del operador de Lindstedt.

        El cociente de residuos es una estimación observada de contracción; no
        es una prueba de la constante global de Lipschitz de la aplicación.
        """
        current_rho = np.asarray(rho, dtype=np.complex128)
        current_envelope = envelope
        history: List[float] = []
        ratios: List[float] = []
        updates = 0
        max_unitary_defect = 0.0

        if current_envelope is None:
            return current_rho, current_envelope, history, float("inf"), 0, False, float("inf")

        for _ in range(_RSI3_MAX_ITER):
            pl = _g(current_envelope, "poincare_lindstedt", None)
            if pl is None:
                return (
                    current_rho,
                    current_envelope,
                    history,
                    float("inf"),
                    updates,
                    False,
                    max_unitary_defect,
                )

            residual = _safe_float(_g(pl, "residual_norm", float("nan")))
            solved = bool(_g(pl, "is_solved", False))
            if not math.isfinite(residual) or residual < 0.0:
                break
            if not history:
                history.append(residual)

            if residual <= _LINSTEDT_RESIDUAL_TOL and solved:
                break

            raw_w = _g(pl, "W", None)
            if raw_w is None:
                break
            w = np.asarray(raw_w, dtype=np.complex128)
            if w.shape != current_rho.shape:
                logger.warning(
                    "RSI³ detenido: W tiene forma %s, ρ tiene forma %s.",
                    w.shape,
                    current_rho.shape,
                )
                break
            if not np.all(np.isfinite(w.real)) or not np.all(np.isfinite(w.imag)):
                break

            generator = 0.5 * (w - w.conj().T)
            generator = 0.5 * (generator - generator.conj().T)
            try:
                unitary = la.expm(_LINSTEDT_DAMP * generator)
            except (la.LinAlgError, ValueError, FloatingPointError) as exc:
                logger.warning("Exponencial de Lindstedt falló: %s", exc)
                break

            identity = np.eye(unitary.shape[0], dtype=np.complex128)
            unitary_defect = float(
                la.norm(unitary.conj().T @ unitary - identity, "fro")
            )
            max_unitary_defect = max(max_unitary_defect, unitary_defect)
            if not math.isfinite(unitary_defect) or unitary_defect > _UNITARY_TOL:
                logger.warning(
                    "RSI³ detenido por defecto unitario %.3e.", unitary_defect,
                )
                break

            candidate = unitary @ current_rho @ unitary.conj().T
            try:
                next_rho, _, _ = self._sanitize_density(candidate)
            except Exception as exc:
                logger.warning("RSI³ no pudo normalizar el nuevo ρ: %s", exc)
                break

            if not hasattr(self.engine, "emit_phase1_envelope"):
                break
            try:
                next_envelope = _call_emit_phase1(
                    self.engine, next_rho, self._background_H,
                )
                if next_envelope is None:
                    break
                transition = getattr(self.engine, "continue_to_phase2", None)
                if callable(transition):
                    transitioned = transition(next_envelope)
                    if transitioned is not None and not isinstance(transitioned, bool):
                        next_envelope = transitioned
            except Exception as exc:
                logger.warning("RSI³ reobservación del motor falló: %s", exc)
                break

            next_pl = _g(next_envelope, "poincare_lindstedt", None)
            next_residual = _safe_float(
                _g(next_pl, "residual_norm", float("nan"))
                if next_pl is not None
                else float("nan")
            )
            if not math.isfinite(next_residual) or next_residual < 0.0:
                break

            if residual > _WILKINSON_EPSILON:
                ratios.append(next_residual / residual)
            history.append(next_residual)
            current_rho = next_rho
            current_envelope = next_envelope
            updates += 1

        final_pl = _g(current_envelope, "poincare_lindstedt", None)
        final_residual = _safe_float(
            _g(final_pl, "residual_norm", float("nan"))
            if final_pl is not None
            else float("nan")
        )
        final_solved = bool(_g(final_pl, "is_solved", False)) if final_pl is not None else False
        if math.isfinite(final_residual) and final_residual <= _LINSTEDT_RESIDUAL_TOL:
            final_solved = True

        observed_contraction = max(ratios) if ratios else 0.0
        ratios_valid = all(
            math.isfinite(value) and value <= _RSI3_STEP_LIP_MAX
            for value in ratios
        )
        valid = bool(
            final_pl is not None
            and math.isfinite(final_residual)
            and final_residual <= _LINSTEDT_RESIDUAL_TOL
            and final_solved
            and ratios_valid
            and max_unitary_defect <= _UNITARY_TOL
        )
        return (
            current_rho,
            current_envelope,
            history,
            float(observed_contraction),
            updates,
            valid,
            float(max_unitary_defect),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 1.4. Adjudicación de Observe y emisión del certificado
    # ──────────────────────────────────────────────────────────────────────

    def _verdict_phase1(
        self,
        *,
        trace_ok: bool,
        norm_a: float,
        antihermitian_ok: bool,
        traceless_ok: bool,
        symplectic_ok: bool,
        symplectic_certified: bool,
        resonant: bool,
        lindstedt_solved: bool,
        rsi3_valid: bool,
        density_repair: float,
    ) -> HeytingOmega3:
        if (
            not trace_ok
            or not math.isfinite(norm_a)
            or norm_a > _NORM_A_HARD
            or not antihermitian_ok
            or not traceless_ok
            or not math.isfinite(density_repair)
            or density_repair >= _DENSITY_REPAIR_HARD
        ):
            return HeytingOmega3.VETOED

        if (
            norm_a > _NORM_A_SOFT
            or not symplectic_ok
            or not symplectic_certified
            or resonant
            or not lindstedt_solved
            or not rsi3_valid
            or density_repair > _DENSITY_REPAIR_SOFT
        ):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def _phase1_envelope_connection(
        self, envelope: Any,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        raw_rho = _g(envelope, "rho_sanitized", None)
        raw_A = _g(envelope, "connection_A", None)
        if raw_rho is None or raw_A is None:
            raise GaugeFieldAgentError(
                "El envelope de fase 1 debe contener rho_sanitized y connection_A."
            )
        rho = np.asarray(raw_rho, dtype=np.complex128)
        connection = np.asarray(raw_A, dtype=np.complex128)
        if (
            rho.ndim != 2
            or rho.shape[0] != rho.shape[1]
            or connection.shape != rho.shape
            or not np.all(np.isfinite(rho.real))
            or not np.all(np.isfinite(rho.imag))
            or not np.all(np.isfinite(connection.real))
            or not np.all(np.isfinite(connection.imag))
        ):
            raise GaugeFieldAgentError("Envelope de fase 1 con matrices inválidas.")
        return rho, connection

    def emit_phase1_certificate(
        self,
        density_matrix: NDArray[Any],
    ) -> GaugeFieldIsometryCertificate:
        """
        Cierre operatorial de Observe: sanitiza, itera RSI³ y firma.

        La ausencia de un envelope verificable no se presenta como certificación
        simpléctica: el resultado se degrada de forma explícita.
        """
        rho, density_repair, density_min_eig = self._sanitize_density(density_matrix)
        trace = float(np.real(np.trace(rho)))
        trace_ok = (
            math.isfinite(trace)
            and abs(trace - 1.0) <= _TRACE_CONSERVATION_TOL
            and density_min_eig >= -_TRACE_CONSERVATION_TOL
        )

        envelope: Any = None
        history: List[float] = []
        rsi3_contraction = float("inf")
        rsi3_iterations = 0
        rsi3_valid = False
        unitary_defect = float("inf")

        if hasattr(self.engine, "emit_phase1_envelope"):
            try:
                envelope = _call_emit_phase1(
                    self.engine, rho, self._background_H,
                )
                if envelope is not None:
                    transition = getattr(self.engine, "continue_to_phase2", None)
                    if callable(transition):
                        transitioned = transition(envelope)
                        if transitioned is not None and not isinstance(transitioned, bool):
                            envelope = transitioned
                    (
                        rho,
                        envelope,
                        history,
                        rsi3_contraction,
                        rsi3_iterations,
                        rsi3_valid,
                        unitary_defect,
                    ) = self._rsi3_lindstedt_reobserve(rho, envelope)
                    trace = float(np.real(np.trace(rho)))
                    trace_ok = (
                        math.isfinite(trace)
                        and abs(trace - 1.0) <= _TRACE_CONSERVATION_TOL
                    )
                    density_min_eig = float(
                        np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)))
                    )
            except Exception as exc:
                logger.warning("Envelope de fase 1 no disponible: %s", exc)
                envelope = None

        if envelope is not None:
            try:
                envelope_rho, connection = self._phase1_envelope_connection(envelope)
                if not np.allclose(
                    _project_density_matrix(envelope_rho),
                    rho,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ):
                    raise EnvelopeChainViolationError(
                        "El envelope de fase 1 no corresponde al ρ reobservado."
                    )

                anti_residual, trace_residual, norm_a = (
                    self._antihermitian_diagnostics(connection)
                )
                antihermitian_ok = (
                    anti_residual <= _ANTI_HERM_TOL
                    and bool(_g(envelope, "is_anti_hermitian_A", False))
                )
                traceless_ok = (
                    trace_residual <= _TRACELESS_TOL
                    and bool(_g(envelope, "is_traceless_A", False))
                )

                symplectic_residue = _safe_float(
                    _g(envelope, "symplectic_residue", float("inf")),
                )
                symplectic_defect = _safe_float(
                    _g(envelope, "symplectic_defect", float("inf")),
                )
                unitary_defect_engine = _safe_float(
                    _g(envelope, "cayley_unitary_defect", unitary_defect),
                )
                if math.isfinite(unitary_defect_engine):
                    unitary_defect = max(
                        unitary_defect if math.isfinite(unitary_defect) else 0.0,
                        unitary_defect_engine,
                    )
                symplectic_certified = bool(
                    _g(envelope, "is_symplectic_valid", False)
                    and math.isfinite(symplectic_residue)
                    and math.isfinite(symplectic_defect)
                    and symplectic_residue <= _SYMPLECTIC_TOL
                    and symplectic_defect <= _SYMPLECTIC_TOL
                    and math.isfinite(unitary_defect)
                    and unitary_defect <= _UNITARY_TOL
                )
                symplectic_ok = symplectic_certified

                pl = _g(envelope, "poincare_lindstedt", None)
                poincare_residual = _safe_float(
                    _g(pl, "residual_norm", float("inf"))
                    if pl is not None
                    else float("inf")
                )
                resonant = bool(_g(pl, "is_resonant", True)) if pl is not None else True
                lindstedt_solved = bool(
                    _g(pl, "is_solved", False)
                    if pl is not None
                    else False
                )
                if (
                    math.isfinite(poincare_residual)
                    and poincare_residual <= _LINSTEDT_RESIDUAL_TOL
                ):
                    lindstedt_solved = True
                n_small_divisors = _safe_int(
                    _g(pl, "n_small_divisors", 0) if pl is not None else 0,
                    0,
                )

                energy_baseline = _safe_float(
                    _g(envelope, "energy_ym", 0.0), 0.0,
                )
                spectral_gap = _safe_float(_g(envelope, "spectral_gap", 0.0), 0.0)
                cartan_invariant = _safe_float(
                    _g(envelope, "poincare_cartan_invariant", 0.0), 0.0,
                )
                is_kahler = bool(_g(envelope, "is_kahler", False))
                is_quaternionic = bool(
                    _g(envelope, "is_quaternionic_proxy", False),
                )
                committed_rho = np.asarray(envelope_rho, dtype=np.complex128)
                committed_A = np.asarray(connection, dtype=np.complex128)
            except Exception as exc:
                logger.warning("Envelope de fase 1 inconsistente; modo degradado: %s", exc)
                envelope = None

        if envelope is None:
            connection = np.asarray(
                self.engine.build_gauge_connection_A(rho),
                dtype=np.complex128,
            )
            anti_residual, trace_residual, norm_a = (
                self._antihermitian_diagnostics(connection)
            )
            antihermitian_ok = anti_residual <= _ANTI_HERM_TOL
            traceless_ok = trace_residual <= _TRACELESS_TOL
            symplectic_residue = float("inf")
            symplectic_defect = float("inf")
            symplectic_ok = False
            symplectic_certified = False
            poincare_residual = float("inf")
            resonant = True
            lindstedt_solved = False
            n_small_divisors = 0
            energy_baseline = 0.0
            spectral_gap = 0.0
            cartan_invariant = 0.0
            is_kahler = False
            is_quaternionic = False
            unitary_defect = float("inf")
            rsi3_valid = False
            rsi3_contraction = float("inf")
            committed_rho = np.asarray(rho, dtype=np.complex128)
            committed_A = np.asarray(connection, dtype=np.complex128)

            curvature_method = getattr(
                self.engine, "compute_yang_mills_curvature_F", None,
            )
            if callable(curvature_method):
                try:
                    _, energy_baseline = curvature_method(connection, rho)
                    energy_baseline = _safe_float(energy_baseline, 0.0)
                except Exception:
                    energy_baseline = 0.0

        verdict = self._verdict_phase1(
            trace_ok=trace_ok,
            norm_a=norm_a,
            antihermitian_ok=antihermitian_ok,
            traceless_ok=traceless_ok,
            symplectic_ok=symplectic_ok,
            symplectic_certified=symplectic_certified,
            resonant=resonant,
            lindstedt_solved=lindstedt_solved,
            rsi3_valid=rsi3_valid,
            density_repair=density_repair,
        )

        envelope_digest = _engine_envelope_digest(envelope)
        unsigned = GaugeFieldIsometryCertificate(
            registry_size=self.dimension,
            dimension=int(rho.shape[0]),
            connection_norm_frobenius=float(norm_a),
            density_trace=float(trace),
            is_trace_conserved=bool(trace_ok),
            phase1_verdict=verdict,
            hmac_phase1_signature="",
            is_anti_hermitian_A=bool(antihermitian_ok),
            is_traceless_A=bool(traceless_ok),
            symplectic_residue=float(symplectic_residue),
            symplectic_defect=float(symplectic_defect),
            is_symplectic_valid=bool(symplectic_ok),
            poincare_residual=float(poincare_residual),
            poincare_is_resonant=bool(resonant),
            energy_ym_baseline=float(energy_baseline),
            engine_envelope=envelope,
            spectral_gap=float(spectral_gap),
            poincare_cartan_invariant=float(cartan_invariant),
            is_kahler=bool(is_kahler),
            is_quaternionic_proxy=bool(is_quaternionic),
            cayley_unitary_defect=float(unitary_defect),
            lindstedt_is_solved=bool(lindstedt_solved),
            n_small_divisors=int(n_small_divisors),
            rsi3_iterations=int(rsi3_iterations),
            rsi3_contraction=float(rsi3_contraction),
            symplectic_certified=bool(symplectic_certified),
            density_min_eigenvalue=float(density_min_eig),
            density_repair_norm=float(density_repair),
            rsi3_valid=bool(rsi3_valid),
            rsi3_residual_history=tuple(float(x) for x in history),
            rho_state_hash=_matrix_hash(committed_rho),
            connection_state_hash=_matrix_hash(committed_A),
            engine_envelope_digest=envelope_digest,
        )
        return self._sign_phase1(unsigned)

    def execute_phase1_gauge_isometry(
        self, density_matrix: NDArray[Any],
    ) -> Tuple[GaugeFieldIsometryCertificate, NDArray[np.complex128]]:
        """Compatibilidad: ejecuta Observe y devuelve además la conexión A."""
        cert1 = self.emit_phase1_certificate(density_matrix)
        if cert1.engine_envelope is not None:
            connection = _g(cert1.engine_envelope, "connection_A", None)
            if connection is not None:
                return cert1, np.asarray(connection, dtype=np.complex128)
        rho, _, _ = self._sanitize_density(density_matrix)
        return cert1, np.asarray(
            self.engine.build_gauge_connection_A(rho),
            dtype=np.complex128,
        )

    def consume_engine_phase1_envelope(
        self, engine_env: Any,
    ) -> GaugeFieldIsometryCertificate:
        """Adapta un envelope externo de fase 1 a un certificado firmado local."""
        if engine_env is None:
            raise GaugeFieldAgentError("engine_env es None.")

        raw_rho, connection = self._phase1_envelope_connection(engine_env)
        rho, repair, min_eig = _project_density_with_diagnostics(raw_rho)
        if not np.allclose(rho, raw_rho, rtol=1.0e-10, atol=1.0e-12):
            raise GaugeIsometryViolationError(
                "El rho_sanitized externo no es una densidad normalizada válida."
            )

        trace = float(np.real(np.trace(raw_rho)))
        trace_ok = (
            math.isfinite(trace)
            and abs(trace - 1.0) <= _TRACE_CONSERVATION_TOL
            and min_eig >= -_TRACE_CONSERVATION_TOL
        )
        anti_residual, trace_residual, norm_a = (
            self._antihermitian_diagnostics(connection)
        )
        anti_ok = (
            anti_residual <= _ANTI_HERM_TOL
            and bool(_g(engine_env, "is_anti_hermitian_A", False))
        )
        traceless_ok = (
            trace_residual <= _TRACELESS_TOL
            and bool(_g(engine_env, "is_traceless_A", False))
        )

        symplectic_residue = _safe_float(
            _g(engine_env, "symplectic_residue", float("inf")),
        )
        symplectic_defect = _safe_float(
            _g(engine_env, "symplectic_defect", float("inf")),
        )
        unitary_defect = _safe_float(
            _g(engine_env, "cayley_unitary_defect", float("inf")),
        )
        symplectic_ok = bool(
            _g(engine_env, "is_symplectic_valid", False)
            and math.isfinite(symplectic_residue)
            and math.isfinite(symplectic_defect)
            and math.isfinite(unitary_defect)
            and symplectic_residue <= _SYMPLECTIC_TOL
            and symplectic_defect <= _SYMPLECTIC_TOL
            and unitary_defect <= _UNITARY_TOL
        )

        pl = _g(engine_env, "poincare_lindstedt", None)
        pl_residual = _safe_float(
            _g(pl, "residual_norm", float("inf")) if pl is not None else float("inf"),
        )
        resonant = bool(_g(pl, "is_resonant", True)) if pl is not None else True
        lindstedt_solved = bool(_g(pl, "is_solved", False)) if pl is not None else False
        if math.isfinite(pl_residual) and pl_residual <= _LINSTEDT_RESIDUAL_TOL:
            lindstedt_solved = True

        rsi3 = _g(engine_env, "rsi3_certificate", None)
        rsi3_valid = bool(_g(rsi3, "is_valid", False))
        rsi3_lip = _safe_float(_g(rsi3, "lipschitz_t3", float("inf")))
        if rsi3 is None:
            rsi3_valid = bool(lindstedt_solved and not resonant)
            rsi3_lip = 0.0 if rsi3_valid else float("inf")
        if rsi3_valid and rsi3_lip > _BANACH_LIPSCHITZ_T3_MAX:
            rsi3_valid = False

        verdict = self._verdict_phase1(
            trace_ok=trace_ok,
            norm_a=norm_a,
            antihermitian_ok=anti_ok,
            traceless_ok=traceless_ok,
            symplectic_ok=symplectic_ok,
            symplectic_certified=True,
            resonant=resonant,
            lindstedt_solved=lindstedt_solved,
            rsi3_valid=rsi3_valid,
            density_repair=repair,
        )

        envelope_digest = _engine_envelope_digest(engine_env)
        unsigned = GaugeFieldIsometryCertificate(
            registry_size=self.dimension,
            dimension=int(raw_rho.shape[0]),
            connection_norm_frobenius=float(norm_a),
            density_trace=trace,
            is_trace_conserved=trace_ok,
            phase1_verdict=verdict,
            hmac_phase1_signature="",
            is_anti_hermitian_A=anti_ok,
            is_traceless_A=traceless_ok,
            symplectic_residue=symplectic_residue,
            symplectic_defect=symplectic_defect,
            is_symplectic_valid=symplectic_ok,
            poincare_residual=pl_residual,
            poincare_is_resonant=resonant,
            energy_ym_baseline=_safe_float(_g(engine_env, "energy_ym", 0.0), 0.0),
            engine_envelope=engine_env,
            spectral_gap=_safe_float(_g(engine_env, "spectral_gap", 0.0), 0.0),
            poincare_cartan_invariant=_safe_float(
                _g(engine_env, "poincare_cartan_invariant", 0.0), 0.0,
            ),
            is_kahler=bool(_g(engine_env, "is_kahler", False)),
            is_quaternionic_proxy=bool(
                _g(engine_env, "is_quaternionic_proxy", False),
            ),
            cayley_unitary_defect=unitary_defect,
            lindstedt_is_solved=lindstedt_solved,
            n_small_divisors=_safe_int(
                _g(pl, "n_small_divisors", 0) if pl is not None else 0,
            ),
            rsi3_iterations=_safe_int(_g(rsi3, "iterations", 0), 0),
            rsi3_contraction=rsi3_lip,
            symplectic_certified=True,
            density_min_eigenvalue=min_eig,
            density_repair_norm=repair,
            rsi3_valid=rsi3_valid,
            rho_state_hash=_matrix_hash(raw_rho),
            connection_state_hash=_matrix_hash(connection),
            engine_envelope_digest=envelope_digest,
        )
        return self._sign_phase1(unsigned)

    # ──────────────────────────────────────────────────────────────────────
    # 1.5. Puente autenticado Observe → Orient
    #      Este es el último método formal de la FASE 1.
    # ──────────────────────────────────────────────────────────────────────

    def continue_to_phase2(
        self, cert1: GaugeFieldIsometryCertificate,
    ) -> GaugeFieldIsometryCertificate:
        """
        Puente Φ₁₂: valida HMAC, digest del envelope, hash de cadena e invariantes.

        La fase siguiente no se inicia si el certificado está alterado o si la
        densidad declarada como coherente viola la conservación de traza.
        """
        self._verify_phase1_certificate(cert1)

        if (
            abs(cert1.density_trace - 1.0) > _TRACE_CONSERVATION_TOL
            and cert1.phase1_verdict is HeytingOmega3.COHERENT
        ):
            raise GaugeIsometryViolationError(
                f"Tr(ρ)={cert1.density_trace:.6e} no está en "
                f"1±{_TRACE_CONSERVATION_TOL:.1e}."
            )
        self._phase1_cert = cert1
        logger.debug(
            "Puente FASE 1→2 validado; firma=%s…",
            cert1.hmac_phase1_signature[:16],
        )
        return cert1


# ══════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ORIENT                                                           ║
# ║ Dinámica, topología, sección de Poincaré, QND y RSI³ del motor.             ║
# ║ Comienza recibiendo Φ₁ y termina con continue_to_phase3, puente Φ₂₃.       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase2Orientor(GaugeAgentPhase1Observer):
    """Orientador de fase 2; hereda validación y estado autenticado de fase 1."""

    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: GaugeFieldIsometryCertificate,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: Optional[int] = None,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> "GaugeAgentPhase2Orientor":
        """Apertura de fase 2 dual al puente autenticado Φ₁₂."""
        dim = int(dimension) if dimension is not None else int(cert1.dimension)
        instance = cls(
            dimension=dim,
            secret_key=secret_key,
            background_hamiltonian=background_hamiltonian,
        )
        instance.continue_to_phase2(cert1)
        return instance

    def continue_into_phase2_campaign_gate(
        self, cert1: GaugeFieldIsometryCertificate,
    ) -> bool:
        """Compatibilidad con la compuerta histórica Φ₁₂."""
        self.continue_to_phase2(cert1)
        return True

    # ──────────────────────────────────────────────────────────────────────
    # 2.1. Medición débil/QND
    # ──────────────────────────────────────────────────────────────────────

    def _observe_qnd_diagnostics(
        self,
        rho: NDArray[np.complex128],
        connection_A: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        """
        Evalúa A_w con H=-iA y verifica solapamiento, varianza y retroacción.

        El estado inicial se elige como autovector principal de ρ para evitar
        postselecciones numéricamente casi ortogonales. Si el motor no reporta
        retroacción, ésta queda como desconocida y QND no se declara válido.
        """
        matrix = np.asarray(connection_A, dtype=np.complex128)
        if matrix.shape != rho.shape:
            return {
                "a_weak": complex(float("nan"), float("nan")),
                "back_action": float("nan"),
                "qnd_var": float("inf"),
                "qnd_valid": False,
                "qnd_measured": False,
                "overlap_abs": 0.0,
            }

        observable = -1j * matrix
        observable = 0.5 * (observable + observable.conj().T)
        if (
            not np.all(np.isfinite(observable.real))
            or not np.all(np.isfinite(observable.imag))
        ):
            return {
                "a_weak": complex(float("nan"), float("nan")),
                "back_action": float("nan"),
                "qnd_var": float("inf"),
                "qnd_valid": False,
                "qnd_measured": False,
                "overlap_abs": 0.0,
            }

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (rho + rho.conj().T),
            )
            phi_i = np.asarray(
                eigenvectors[:, int(np.argmax(eigenvalues))],
                dtype=np.complex128,
            )
            pivot = int(np.argmax(np.abs(phi_i)))
            if abs(phi_i[pivot]) > _WILKINSON_EPSILON:
                phi_i *= np.exp(-1j * np.angle(phi_i[pivot]))

            phi_f = rho @ phi_i
            norm_final = float(np.linalg.norm(phi_f))
            if not math.isfinite(norm_final) or norm_final <= _WILKINSON_EPSILON:
                raise ValueError("Estado postseleccionado de norma nula.")
            phi_f /= norm_final

            overlap = np.vdot(phi_f, phi_i)
            overlap_abs = float(abs(overlap))
            if not math.isfinite(overlap_abs) or overlap_abs <= _WILKINSON_EPSILON:
                raise ValueError("Solapamiento AAV nulo o no finito.")

            weak_value = complex(np.vdot(phi_f, observable @ phi_i) / overlap)
            mean = np.vdot(phi_i, observable @ phi_i)
            second_moment = np.vdot(phi_i, observable @ observable @ phi_i)
            variance_raw = float(np.real(second_moment - mean * mean))
            variance_scale = max(
                1.0,
                float(la.norm(observable, "fro")) ** 2,
            )
            if variance_raw < -1.0e-10 * variance_scale:
                variance = float("inf")
            else:
                variance = max(0.0, variance_raw)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            logger.debug("Construcción AAV/QND falló: %s", exc)
            return {
                "a_weak": complex(float("nan"), float("nan")),
                "back_action": float("nan"),
                "qnd_var": float("inf"),
                "qnd_valid": False,
                "qnd_measured": False,
                "overlap_abs": 0.0,
            }

        method = getattr(self.engine, "observe_qnd_weak_measurement", None)
        measured = False
        back_action = float("nan")
        engine_variance = float("nan")
        engine_weak = complex(float("nan"), float("nan"))
        if callable(method):
            try:
                raw_result = method(
                    phi_i.reshape(-1, 1),
                    phi_f.reshape(-1, 1),
                    observable,
                )
                engine_weak, back_action, engine_variance = _unpack_qnd(raw_result)
                measured = bool(
                    math.isfinite(back_action)
                    and np.isfinite(engine_weak.real)
                    and np.isfinite(engine_weak.imag)
                )
            except Exception as exc:
                logger.debug("Medición QND del motor no disponible: %s", exc)

        weak_consistent = bool(
            measured
            and abs(engine_weak - weak_value)
            <= 1.0e-7 * max(1.0, abs(weak_value), abs(engine_weak))
        )
        variance_consistent = bool(
            measured
            and (
                not math.isfinite(engine_variance)
                or abs(engine_variance - variance)
                <= 1.0e-7 * max(1.0, variance, abs(engine_variance))
            )
        )
        variance_ok = (
            math.isfinite(variance)
            and variance <= _QND_VARIANCE_TOL * variance_scale
        )
        back_action_ok = (
            measured
            and math.isfinite(back_action)
            and abs(back_action) <= _QND_BACK_ACTION_TOL_DB
        )

        return {
            "a_weak": weak_value,
            "back_action": back_action,
            "qnd_var": variance,
            "qnd_valid": bool(
                overlap_abs > _WILKINSON_EPSILON
                and variance_ok
                and back_action_ok
                and weak_consistent
                and variance_consistent
            ),
            "qnd_measured": measured,
            "overlap_abs": overlap_abs,
        }

    def _observe_qnd_safe(
        self,
        rho: NDArray[np.complex128],
        connection_A: NDArray[np.complex128],
    ) -> Tuple[complex, float, float]:
        """Adaptador de compatibilidad que devuelve (A_w, dB, Var)."""
        observation = self._observe_qnd_diagnostics(rho, connection_A)
        return (
            complex(observation["a_weak"]),
            float(observation["back_action"]),
            float(observation["qnd_var"]),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 2.2. Lectura normalizada de certificados del motor
    # ──────────────────────────────────────────────────────────────────────

    def _extract_from_phase2_envelope(
        self,
        engine_p1_env: Any,
        engine_p2_env: Any,
        rho: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        kam = _g(engine_p2_env, "kam_certificate", None)
        mel = _g(engine_p2_env, "melnikov", None)
        floquet = _g(engine_p2_env, "floquet", None)
        section = _g(engine_p2_env, "poincare_section", None)
        rsi3 = _g(engine_p2_env, "rsi3_certificate", None)

        raw_A = _g(engine_p1_env, "connection_A", None)
        if raw_A is None:
            raise GaugeFieldAgentError(
                "El envelope de fase 1 no contiene connection_A."
            )
        connection_A = np.asarray(raw_A, dtype=np.complex128)
        qnd = self._observe_qnd_diagnostics(rho, connection_A)
        poincare = _poincare_diagnostics(section)

        lip_t3 = _safe_float(
            _g(rsi3, "lipschitz_t3", float("inf")),
        )
        rsi3_iterations = _safe_int(_g(rsi3, "iterations", 0), 0)
        fubini = _safe_float(
            _g(_g(rsi3, "surface3", None), "metric_value", float("inf")),
        )
        novikov = _safe_float(
            _g(_g(rsi3, "surface1", None), "metric_value", float("-inf")),
        )
        rsi3_claim = bool(_g(rsi3, "is_valid", False))
        rsi3_ok = bool(
            rsi3_claim
            and math.isfinite(lip_t3)
            and lip_t3 <= _BANACH_LIPSCHITZ_T3_MAX
            and 0 <= rsi3_iterations <= _RSI3_MAX_ITER
        )

        melnikov_value = _safe_float(
            _g(mel, "integral_value", float("inf")),
        )
        melnikov_ok = bool(
            _g(mel, "is_homoclinic_preserved", False)
            and math.isfinite(melnikov_value)
            and abs(melnikov_value) <= _MELNIKOV_TOL
        )
        floquet_deviation = _safe_float(
            _g(floquet, "max_unit_deviation", float("inf")),
        )
        krein_collision = bool(_g(floquet, "krein_collision", False))
        floquet_neutral = bool(
            _g(floquet, "is_neutral", False)
            and math.isfinite(floquet_deviation)
            and floquet_deviation <= _FLOQUET_UNIT_TOL
            and not krein_collision
        )

        fields_out: Dict[str, Any] = {
            "ym_energy": _safe_float(_g(engine_p1_env, "energy_ym", float("nan"))),
            "cs3": _safe_float(
                _g(engine_p2_env, "chern_simons_real", float("nan")),
            ),
            "cs_anomaly": _safe_float(
                _g(engine_p2_env, "chern_simons_anomaly", float("nan")),
            ),
            "w_angle": _safe_float(
                _g(engine_p2_env, "wilson_angle_rad", float("nan")),
            ),
            "c1": _safe_float(_g(engine_p2_env, "c1_value", float("nan"))),
            "k_inst": _safe_int(
                _g(engine_p2_env, "pontryagin_instanton_k", 0),
            ),
            "integrality": _safe_float(
                _g(engine_p2_env, "pontryagin_integrality_residual", float("inf")),
            ),
            "sz": _safe_float(
                _g(engine_p2_env, "stora_zumino_residual", float("inf")),
            ),
            "kam_worst": _safe_float(
                _g(kam, "worst_ratio", float("inf")),
            ),
            "kam_ok": bool(_g(kam, "is_satisfied", False)),
            "bruno_ok": bool(_g(kam, "bruno_satisfied", False)),
            "bruno_sum": _safe_float(_g(kam, "bruno_sum", float("inf"))),
            "melnikov": melnikov_value,
            "melnikov_ok": melnikov_ok,
            "floquet_dev": floquet_deviation,
            "floquet_neutral": floquet_neutral,
            "krein": krein_collision,
            "poincare_x": poincare.n_crossings,
            "poincare_disp": poincare.return_dispersion,
            "mean_ret": poincare.mean_return_time,
            "rec_time": poincare.recurrence_time,
            "poincare_data_valid": poincare.data_valid,
            "poincare_map_valid": poincare.map_symplectic_valid,
            "poincare_map_defect": poincare.map_symplectic_defect,
            "poincare_transversal": poincare.transversal,
            "poincare_valid": poincare.valid,
            "poincare_times": poincare.return_times,
            "rsi3_ok": rsi3_ok,
            "rsi3_iters": rsi3_iterations,
            "banach_t3": lip_t3,
            "fubini": fubini,
            "novikov": novikov,
            "contraction": _safe_float(
                _g(rsi3, "contraction_ratio", float("inf")),
            ),
            "back_action": float(qnd["back_action"]),
            "a_weak": complex(qnd["a_weak"]),
            "qnd_var": float(qnd["qnd_var"]),
            "qnd_valid": bool(qnd["qnd_valid"]),
            "qnd_overlap": float(qnd["overlap_abs"]),
            "cs_alg": _safe_float(
                _g(engine_p2_env, "chern_simons_algebraic", float("nan")),
            ),
            "ch2": _safe_float(
                _g(engine_p2_env, "chern_character_ch2", float("nan")),
            ),
            "regime": str(_g(engine_p2_env, "topology_regime", "UNKNOWN")),
        }
        return fields_out

    def _extract_from_snapshot(self, snapshot: Any) -> Dict[str, Any]:
        """Adapta snapshots v10/v11 y marca como desconocidos los campos ausentes."""
        poincare = _poincare_diagnostics(snapshot)
        melnikov = _safe_float(_g(snapshot, "melnikov_value", float("inf")))
        floquet_deviation = _safe_float(
            _g(snapshot, "floquet_max_unit_deviation", float("inf")),
        )
        krein_collision = bool(_g(snapshot, "krein_collision", False))
        lip_t3 = _safe_float(
            _g(snapshot, "banach_lipschitz_t3", float("inf")),
        )
        iterations = _safe_int(_g(snapshot, "rsi3_iterations", 0), 0)
        rsi3_ok = bool(
            _g(snapshot, "rsi3_valid", False)
            and math.isfinite(lip_t3)
            and lip_t3 <= _BANACH_LIPSCHITZ_T3_MAX
            and 0 <= iterations <= _RSI3_MAX_ITER
        )
        melnikov_ok = bool(
            abs(melnikov) <= _MELNIKOV_TOL
            and math.isfinite(melnikov)
        )
        floquet_neutral = bool(
            _g(snapshot, "floquet_is_neutral", False)
            and math.isfinite(floquet_deviation)
            and floquet_deviation <= _FLOQUET_UNIT_TOL
            and not krein_collision
        )

        return {
            "ym_energy": _safe_float(
                _g(snapshot, "yang_mills_energy", float("nan")),
            ),
            "cs3": _safe_float(
                _g(snapshot, "chern_simons_3form_higgs", float("nan")),
            ),
            "cs_anomaly": _safe_float(
                _g(snapshot, "chern_simons_anomaly", float("nan")),
            ),
            "w_angle": _safe_float(
                _g(snapshot, "wilson_angle_rad", float("nan")),
            ),
            "c1": _safe_float(
                _g(snapshot, "chern_hopf_class_c1", float("nan")),
            ),
            "k_inst": _safe_int(
                _g(snapshot, "pontryagin_instanton_number", 0),
            ),
            "integrality": _safe_float(
                _g(snapshot, "pontryagin_integrality_residual", float("inf")),
            ),
            "sz": _safe_float(
                _g(snapshot, "stora_zumino_residual", float("inf")),
            ),
            "kam_worst": _safe_float(
                _g(snapshot, "kam_worst_ratio", float("inf")),
            ),
            "kam_ok": bool(_g(snapshot, "kam_satisfied", False)),
            "bruno_ok": bool(_g(snapshot, "bruno_satisfied", False)),
            "bruno_sum": _safe_float(
                _g(snapshot, "bruno_sum", float("inf")),
            ),
            "melnikov": melnikov,
            "melnikov_ok": bool(
                _g(snapshot, "melnikov_preserved", melnikov_ok)
                and melnikov_ok
            ),
            "floquet_dev": floquet_deviation,
            "floquet_neutral": floquet_neutral,
            "krein": krein_collision,
            "poincare_x": poincare.n_crossings,
            "poincare_disp": poincare.return_dispersion,
            "mean_ret": poincare.mean_return_time,
            "rec_time": poincare.recurrence_time,
            "poincare_data_valid": poincare.data_valid,
            "poincare_map_valid": poincare.map_symplectic_valid,
            "poincare_map_defect": poincare.map_symplectic_defect,
            "poincare_transversal": poincare.transversal,
            "poincare_valid": poincare.valid,
            "poincare_times": poincare.return_times,
            "rsi3_ok": rsi3_ok,
            "rsi3_iters": iterations,
            "banach_t3": lip_t3,
            "fubini": _safe_float(
                _g(snapshot, "fubini_study_distance", float("inf")),
            ),
            "novikov": _safe_float(
                _g(snapshot, "novikov_krull_valuation", float("-inf")),
            ),
            "contraction": (
                lip_t3 ** (1.0 / 3.0)
                if math.isfinite(lip_t3) and lip_t3 > 0.0
                else 0.0
            ),
            "back_action": _safe_float(
                _g(snapshot, "back_action_db", float("inf")),
            ),
            "a_weak": _safe_complex(
                _g(snapshot, "weak_value_qnd_aav", complex(float("nan"), float("nan"))),
            ),
            "qnd_var": _safe_float(
                _g(snapshot, "qnd_variance", float("inf")),
            ),
            "qnd_valid": bool(_g(snapshot, "qnd_valid", False)),
            "qnd_overlap": _safe_float(
                _g(snapshot, "qnd_overlap_abs", 0.0), 0.0,
            ),
            "cs_alg": _safe_float(
                _g(snapshot, "chern_simons_algebraic", float("nan")),
            ),
            "ch2": _safe_float(
                _g(snapshot, "chern_character_ch2", float("nan")),
            ),
            "regime": str(_g(snapshot, "topology_regime", "UNKNOWN")),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 2.3. Fock, adjudicación de Orient y construcción del certificado
    # ──────────────────────────────────────────────────────────────────────

    def _fock_annihilation(
        self,
        electron_anomaly_count: int,
        positron_auth_signature: Optional[str],
    ) -> int:
        count = int(electron_anomaly_count)
        if count <= 0 or not positron_auth_signature:
            return 0
        auth_payload = f"AUTH_POSITRON:{count}"
        if not self._verify_hmac(auth_payload, positron_auth_signature):
            logger.warning("Firma de positrón inválida; no se autoriza la aniquilación.")
            return 0
        logger.info(
            "[FOCK] e⁻(%d)+e⁺(%d) → 2γ (%.1f keV reportados).",
            count,
            count,
            _POSITRON_REST_MASS_ENERGY_KEV,
        )
        return count

    def _phase2_atoms(
        self,
        cert1: GaugeFieldIsometryCertificate,
        fields_out: Mapping[str, Any],
    ) -> List[HeytingOmega3]:
        energy = _safe_float(fields_out.get("ym_energy"))
        if not math.isfinite(energy) or energy < -_TOPOLOGY_RESIDUAL_TOL:
            energy_atom = HeytingOmega3.VETOED
        elif energy <= 0.25:
            energy_atom = HeytingOmega3.COHERENT
        elif energy <= 0.85:
            energy_atom = HeytingOmega3.DEGRADED
        else:
            energy_atom = HeytingOmega3.VETOED

        instanton_atom = (
            HeytingOmega3.COHERENT
            if abs(_safe_int(fields_out.get("k_inst", 0))) <= 2
            else HeytingOmega3.VETOED
        )
        kam_atom = (
            HeytingOmega3.COHERENT
            if bool(fields_out.get("kam_ok", False))
            else HeytingOmega3.VETOED
        )
        bruno_atom = (
            HeytingOmega3.COHERENT
            if bool(fields_out.get("bruno_ok", False))
            else HeytingOmega3.DEGRADED
        )
        melnikov_atom = (
            HeytingOmega3.COHERENT
            if bool(fields_out.get("melnikov_ok", False))
            else HeytingOmega3.VETOED
        )
        floquet_atom = (
            HeytingOmega3.COHERENT
            if bool(fields_out.get("floquet_neutral", False))
            else HeytingOmega3.VETOED
        )
        rsi3_atom = (
            HeytingOmega3.COHERENT
            if bool(fields_out.get("rsi3_ok", False))
            else HeytingOmega3.VETOED
        )

        novikov = _safe_float(fields_out.get("novikov"))
        novikov_atom = (
            HeytingOmega3.COHERENT
            if math.isfinite(novikov) and novikov >= _KRULL_NOVIKOV_FLOOR
            else HeytingOmega3.DEGRADED
        )

        back_action = _safe_float(fields_out.get("back_action"))
        qnd_variance = _safe_float(fields_out.get("qnd_var"))
        if bool(fields_out.get("qnd_valid", False)):
            qnd_atom = HeytingOmega3.COHERENT
        elif (
            (math.isfinite(back_action) and abs(back_action) > _QND_BACK_ACTION_TOL_DB)
            or (
                math.isfinite(qnd_variance)
                and qnd_variance > 100.0 * _QND_VARIANCE_TOL
            )
        ):
            qnd_atom = HeytingOmega3.VETOED
        else:
            qnd_atom = HeytingOmega3.DEGRADED

        poincare_data_valid = bool(fields_out.get("poincare_data_valid", False))
        poincare_valid = bool(fields_out.get("poincare_valid", False))
        if poincare_valid:
            poincare_atom = HeytingOmega3.COHERENT
        elif poincare_data_valid:
            # Datos de retorno finitos pero sin certificado de mapa/transversalidad.
            poincare_atom = HeytingOmega3.DEGRADED
        else:
            poincare_atom = HeytingOmega3.DEGRADED

        topology_values = (
            _safe_float(fields_out.get("cs3")),
            _safe_float(fields_out.get("cs_anomaly")),
            _safe_float(fields_out.get("w_angle")),
            _safe_float(fields_out.get("c1")),
            _safe_float(fields_out.get("integrality")),
            _safe_float(fields_out.get("sz")),
            _safe_float(fields_out.get("cs_alg")),
            _safe_float(fields_out.get("ch2")),
        )
        if not all(math.isfinite(value) for value in topology_values):
            topology_atom = HeytingOmega3.DEGRADED
        elif (
            abs(topology_values[4]) > _TOPOLOGY_RESIDUAL_TOL
            or abs(topology_values[5]) > _TOPOLOGY_RESIDUAL_TOL
            or abs(topology_values[1]) > _TOPOLOGY_RESIDUAL_TOL
        ):
            topology_atom = HeytingOmega3.DEGRADED
        else:
            topology_atom = HeytingOmega3.COHERENT

        return [
            cert1.phase1_verdict,
            energy_atom,
            instanton_atom,
            topology_atom,
            kam_atom,
            bruno_atom,
            melnikov_atom,
            floquet_atom,
            rsi3_atom,
            novikov_atom,
            qnd_atom,
            poincare_atom,
        ]

    def _assemble_phase2_certificate(
        self,
        cert1: GaugeFieldIsometryCertificate,
        fields_out: Mapping[str, Any],
        engine_p2_env: Any,
        annihilations: int,
    ) -> Phase2GaugeDynamicsCertificate:
        atoms = self._phase2_atoms(cert1, fields_out)
        verdict = self._adjudicator.meet_atoms(atoms)

        unsigned = Phase2GaugeDynamicsCertificate(
            yang_mills_energy=_safe_float(fields_out.get("ym_energy")),
            chern_simons_3form_higgs=_safe_float(fields_out.get("cs3")),
            wilson_angle_rad=_safe_float(fields_out.get("w_angle")),
            instanton_k=_safe_int(fields_out.get("k_inst", 0)),
            stora_zumino_residual=_safe_float(fields_out.get("sz")),
            weak_value_qnd=_safe_complex(fields_out.get("a_weak")),
            positron_annihilations_count=int(annihilations),
            phase2_verdict=verdict,
            hmac_phase2_signature="",
            chern_simons_anomaly=_safe_float(fields_out.get("cs_anomaly")),
            c1_value=_safe_float(fields_out.get("c1")),
            pontryagin_integrality_residual=_safe_float(fields_out.get("integrality")),
            kam_worst_ratio=_safe_float(fields_out.get("kam_worst")),
            kam_satisfied=bool(fields_out.get("kam_ok", False)),
            melnikov_value=_safe_float(fields_out.get("melnikov")),
            melnikov_preserved=bool(fields_out.get("melnikov_ok", False)),
            floquet_max_unit_deviation=_safe_float(fields_out.get("floquet_dev")),
            floquet_neutral=bool(fields_out.get("floquet_neutral", False)),
            poincare_section_n_crossings=_safe_int(fields_out.get("poincare_x", 0)),
            poincare_section_dispersion=_safe_float(fields_out.get("poincare_disp")),
            rsi3_valid=bool(fields_out.get("rsi3_ok", False)),
            rsi3_iterations=_safe_int(fields_out.get("rsi3_iters", 0)),
            banach_lipschitz_t3=_safe_float(fields_out.get("banach_t3")),
            fubini_study_distance=_safe_float(fields_out.get("fubini")),
            novikov_krull_valuation=_safe_float(fields_out.get("novikov")),
            back_action_db=_safe_float(fields_out.get("back_action")),
            topology_regime=str(fields_out.get("regime", "UNKNOWN")),
            engine_envelope=engine_p2_env,
            dimension=int(cert1.dimension),
            bruno_satisfied=bool(fields_out.get("bruno_ok", False)),
            bruno_sum=_safe_float(fields_out.get("bruno_sum")),
            chern_simons_algebraic=_safe_float(fields_out.get("cs_alg")),
            chern_character_ch2=_safe_float(fields_out.get("ch2")),
            krein_collision=bool(fields_out.get("krein", False)),
            qnd_variance=_safe_float(fields_out.get("qnd_var")),
            mean_return_time=_safe_float(fields_out.get("mean_ret")),
            recurrence_time=_safe_float(fields_out.get("rec_time")),
            rsi3_contraction=_safe_float(fields_out.get("contraction")),
            parent_phase1_chain_hash=cert1.chain_hash,
            parent_phase1_signature=cert1.hmac_phase1_signature,
            engine_envelope_digest=_engine_envelope_digest(engine_p2_env),
            qnd_valid=bool(fields_out.get("qnd_valid", False)),
            qnd_overlap_abs=_safe_float(fields_out.get("qnd_overlap", 0.0), 0.0),
            poincare_data_valid=bool(fields_out.get("poincare_data_valid", False)),
            poincare_map_symplectic_valid=bool(
                fields_out.get("poincare_map_valid", False),
            ),
            poincare_symplectic_defect=_safe_float(
                fields_out.get("poincare_map_defect", float("inf")),
            ),
            poincare_transversal=bool(fields_out.get("poincare_transversal", False)),
            poincare_return_times=tuple(
                float(x) for x in fields_out.get("poincare_times", ())
            ),
            floquet_krein_collision=bool(fields_out.get("krein", False)),
        )
        return self._sign_phase2(unsigned)

    def emit_phase2_certificate(
        self,
        cert1: GaugeFieldIsometryCertificate,
        density_matrix: NDArray[Any],
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> Phase2GaugeDynamicsCertificate:
        """Cierre de Orient: valida Φ₁, mide, adjudica y firma Φ₂."""
        self.continue_to_phase2(cert1)
        rho, _, _ = self._sanitize_density(density_matrix)
        self._check_phase1_density_matches(cert1, rho)

        phase1_env = cert1.engine_envelope
        phase2_env: Any = None
        if (
            phase1_env is not None
            and hasattr(self.engine, "emit_phase2_envelope")
        ):
            try:
                phase2_env = self.engine.emit_phase2_envelope(phase1_env)
                if phase2_env is not None:
                    transition = getattr(self.engine, "continue_to_phase3", None)
                    if callable(transition):
                        transitioned = transition(phase2_env)
                        if transitioned is not None and not isinstance(transitioned, bool):
                            phase2_env = transitioned
            except Exception as exc:
                logger.warning("Envelope de fase 2 falló; se intentará snapshot: %s", exc)
                phase2_env = None

        if phase2_env is not None:
            fields_out = self._extract_from_phase2_envelope(
                phase1_env, phase2_env, rho,
            )
        else:
            try:
                snapshot, _ = _call_evaluate(
                    self.engine, rho, self._background_H,
                )
            except Exception as exc:
                raise GaugeFieldAgentError(
                    f"El motor no pudo evaluar el campo: {exc}"
                ) from exc

            fields_out = self._extract_from_snapshot(snapshot)
            raw_A = (
                _g(phase1_env, "connection_A", None)
                if phase1_env is not None
                else None
            )
            if raw_A is None:
                raw_A = self.engine.build_gauge_connection_A(rho)
            qnd = self._observe_qnd_diagnostics(
                rho, np.asarray(raw_A, dtype=np.complex128),
            )
            # Se usa la medición calculada en esta fase cuando existe. Si el motor
            # no la proporciona, el snapshot puede conservar su propio certificado.
            if qnd["qnd_measured"]:
                fields_out.update(
                    {
                        "a_weak": qnd["a_weak"],
                        "back_action": qnd["back_action"],
                        "qnd_var": qnd["qnd_var"],
                        "qnd_valid": qnd["qnd_valid"],
                        "qnd_overlap": qnd["overlap_abs"],
                    }
                )

        annihilations = self._fock_annihilation(
            electron_anomaly_count,
            positron_auth_signature,
        )
        cert2 = self._assemble_phase2_certificate(
            cert1, fields_out, phase2_env, annihilations,
        )
        return cert2

    def execute_phase2_gauge_dynamics(
        self,
        cert1: GaugeFieldIsometryCertificate,
        density_matrix: NDArray[Any],
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> Phase2GaugeDynamicsCertificate:
        """Compatibilidad: ejecuta Orient y devuelve Φ₂."""
        return self.emit_phase2_certificate(
            cert1,
            density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
        )

    def consume_engine_phase2_envelope(
        self, engine_env: Any,
    ) -> Phase2GaugeDynamicsCertificate:
        """Adapta un envelope externo de fase 2 y firma su certificado local."""
        if engine_env is None:
            raise GaugeFieldAgentError("engine_env es None.")
        phase1_env = _g(engine_env, "phase1", None)
        if phase1_env is None:
            raise GaugeFieldAgentError("El envelope de fase 2 no contiene phase1.")

        cert1 = self.consume_engine_phase1_envelope(phase1_env)
        self.continue_to_phase2(cert1)

        raw_rho = _g(phase1_env, "rho_sanitized", None)
        if raw_rho is None:
            raise GaugeFieldAgentError("phase1.rho_sanitized no está disponible.")
        rho, _, _ = self._sanitize_density(np.asarray(raw_rho))
        fields_out = self._extract_from_phase2_envelope(
            phase1_env, engine_env, rho,
        )
        return self._assemble_phase2_certificate(
            cert1,
            fields_out,
            engine_env,
            annihilations=0,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 2.4. Puente autenticado Orient → Decide & Act
    #      Este es el último método formal de la FASE 2.
    # ──────────────────────────────────────────────────────────────────────

    def continue_to_phase3(
        self, cert2: Phase2GaugeDynamicsCertificate,
    ) -> Phase2GaugeDynamicsCertificate:
        """
        Puente Φ₂₃: verifica firma, digest, cadena y enlace al certificado Φ₁.

        Una contracción RSI³ fuera del umbral no invalida la firma criptográfica,
        pero sí impide que el resultado se interprete como coherente.
        """
        self._verify_phase2_certificate(cert2)

        if self._phase1_cert is not None:
            if (
                cert2.parent_phase1_chain_hash != self._phase1_cert.chain_hash
                or cert2.parent_phase1_signature
                != self._phase1_cert.hmac_phase1_signature
            ):
                raise EnvelopeChainViolationError(
                    "El certificado de fase 2 no desciende del Φ₁ activo."
                )

        if (
            not math.isfinite(cert2.banach_lipschitz_t3)
            or cert2.banach_lipschitz_t3 > _BANACH_LIPSCHITZ_T3_MAX
        ):
            logger.warning(
                "RSI³ observado fuera del umbral: Lip(T³)=%.6g, máximo=%.6g.",
                cert2.banach_lipschitz_t3,
                _BANACH_LIPSCHITZ_T3_MAX,
            )

        self._phase2_cert = cert2
        logger.debug(
            "Puente FASE 2→3 validado; firma=%s…",
            cert2.hmac_phase2_signature[:16],
        )
        return cert2


# ══════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — DECIDE & ACT                                                     ║
# ║ Meet conservador, interlock, functor semántico y raíz de compromiso.       ║
# ║ Comienza recibiendo Φ₂ y cierra el estado soberano autenticado.             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase3Decider(GaugeAgentPhase2Orientor):
    """Decisor terminal de fase 3."""

    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: Phase2GaugeDynamicsCertificate,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: Optional[int] = None,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> "GaugeAgentPhase3Decider":
        """Apertura de fase 3 dual al puente autenticado Φ₂₃."""
        dim = int(dimension) if dimension is not None else int(cert2.dimension)
        instance = cls(
            dimension=dim,
            secret_key=secret_key,
            background_hamiltonian=background_hamiltonian,
        )
        instance.continue_to_phase3(cert2)
        return instance

    def continue_into_phase3_adjudication_gate(
        self, cert2: Phase2GaugeDynamicsCertificate,
    ) -> bool:
        """Compatibilidad con la compuerta histórica Φ₂₃."""
        self.continue_to_phase3(cert2)
        return True

    # ──────────────────────────────────────────────────────────────────────
    # 3.1. Interlock: respuesta reportada, validación fail-closed
    # ──────────────────────────────────────────────────────────────────────

    def _crowbar(
        self, verdict: HeytingOmega3,
    ) -> Tuple[bool, float, str]:
        """
        Solicita el interlock y devuelve el acuse reportado por el adaptador.

        El resultado del método no demuestra por sí solo una latencia física
        inferior al WCET. Ante una excepción o ausencia de acuse, no se inventa
        una activación exitosa.
        """
        method = getattr(
            self.engine,
            "trigger_hardware_interlock_esp32_crowbar",
            None,
        )
        if not callable(method):
            return False, float("inf"), "CROWBAR_ADAPTER_UNAVAILABLE"

        attempts = 2 if verdict is HeytingOmega3.VETOED else 1
        last_result: Tuple[bool, float, str] = (
            False,
            float("inf"),
            "CROWBAR_NO_ACK",
        )
        for _ in range(attempts):
            try:
                try:
                    raw_result = method(verdict, gpio_pin=14)
                except TypeError as exc:
                    message = str(exc)
                    if "gpio_pin" not in message and "keyword" not in message:
                        raise
                    raw_result = method(verdict)

                last_result = _unpack_crowbar(raw_result)
                latched, latency, action = last_result
                if (
                    verdict is not HeytingOmega3.VETOED
                    or (
                        latched
                        and math.isfinite(latency)
                        and latency >= 0.0
                    )
                ):
                    return latched, latency, action
            except Exception as exc:
                logger.critical("Fallo al solicitar Crowbar: %s", exc)
                last_result = (
                    False,
                    float("inf"),
                    "CROWBAR_COMMAND_FAILED",
                )
        return last_result

    # ──────────────────────────────────────────────────────────────────────
    # 3.2. Compromiso Merkle del estado terminal
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merkle_root(leaves: Sequence[bytes]) -> str:
        if not leaves:
            return hashlib.sha256(b"APU-TGFA-EMPTY-MERKLE-v1").hexdigest()
        nodes = [
            hashlib.sha256(b"APU-TGFA-MERKLE-LEAF-v1|" + leaf).digest()
            for leaf in leaves
        ]
        while len(nodes) > 1:
            if len(nodes) % 2:
                nodes.append(nodes[-1])
            nodes = [
                hashlib.sha256(
                    b"APU-TGFA-MERKLE-NODE-v1|" + nodes[index] + nodes[index + 1]
                ).digest()
                for index in range(0, len(nodes), 2)
            ]
        return nodes[0].hex()

    @classmethod
    def _state_merkle_digest(
        cls, state: TopologicalGaugeFieldSovereignState,
    ) -> str:
        excluded = {
            "phase1_cert",
            "phase2_cert",
            "sha256_merkle_digest",
            "provenance_hmac",
        }
        state_body = {
            item.name: getattr(state, item.name)
            for item in fields(state)
            if item.name not in excluded
        }
        phase1_leaf = (
            f"{state.phase1_cert.chain_hash}|"
            f"{state.phase1_cert.hmac_phase1_signature}"
        ).encode("utf-8")
        phase2_leaf = (
            f"{state.phase2_cert.chain_hash}|"
            f"{state.phase2_cert.hmac_phase2_signature}"
        ).encode("utf-8")
        state_leaf = _canonical_json(state_body).encode("utf-8")
        return cls._merkle_root((phase1_leaf, phase2_leaf, state_leaf))

    # ──────────────────────────────────────────────────────────────────────
    # 3.3. Adjudicación, acción y firma del estado soberano
    # ──────────────────────────────────────────────────────────────────────

    def emit_sovereign_state(
        self,
        cert1: GaugeFieldIsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        density_matrix: NDArray[Any],
    ) -> TopologicalGaugeFieldSovereignState:
        """
        Cierre de Decide & Act.

        Verifica ambas firmas y su enlace; adjudica el meet de los átomos;
        solicita Crowbar si existe veto; firma el estado final.
        """
        self._verify_phase1_certificate(cert1)
        self._verify_phase2_certificate(cert2)
        if (
            cert2.parent_phase1_chain_hash != cert1.chain_hash
            or cert2.parent_phase1_signature != cert1.hmac_phase1_signature
        ):
            raise EnvelopeChainViolationError(
                "Φ₂ no está enlazado criptográficamente al Φ₁ suministrado."
            )

        rho, _, _ = self._sanitize_density(density_matrix)
        self._check_phase1_density_matches(cert1, rho)

        atoms: List[HeytingOmega3] = [
            cert1.phase1_verdict,
            cert2.phase2_verdict,
            (
                HeytingOmega3.COHERENT
                if cert1.is_trace_conserved
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.is_symplectic_valid
                else HeytingOmega3.DEGRADED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.symplectic_certified
                else HeytingOmega3.DEGRADED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.is_anti_hermitian_A
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.is_traceless_A
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.lindstedt_is_solved
                else HeytingOmega3.DEGRADED
            ),
            (
                HeytingOmega3.COHERENT
                if cert1.rsi3_valid
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert2.kam_satisfied
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert2.bruno_satisfied
                else HeytingOmega3.DEGRADED
            ),
            (
                HeytingOmega3.COHERENT
                if cert2.melnikov_preserved
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert2.rsi3_valid
                and math.isfinite(cert2.banach_lipschitz_t3)
                and cert2.banach_lipschitz_t3 <= _BANACH_LIPSCHITZ_T3_MAX
                else HeytingOmega3.VETOED
            ),
            (
                HeytingOmega3.COHERENT
                if cert2.qnd_valid
                else HeytingOmega3.DEGRADED
            ),
            (
                HeytingOmega3.COHERENT
                if (
                    cert2.poincare_data_valid
                    and cert2.poincare_map_symplectic_valid
                    and cert2.poincare_transversal
                )
                else HeytingOmega3.DEGRADED
            ),
        ]
        global_verdict = self._adjudicator.meet_atoms(atoms)

        crowbar_triggered, latency_ns, hardware_action = self._crowbar(
            global_verdict,
        )
        latency_finite = math.isfinite(latency_ns) and latency_ns >= 0.0
        latency_under_absolute_limit = bool(
            latency_finite and latency_ns < _CROWBAR_TIMING_LIMIT_NS
        )
        if global_verdict is HeytingOmega3.VETOED:
            latency_verified = bool(
                crowbar_triggered
                and latency_finite
                and latency_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS
                and latency_under_absolute_limit
            )
        else:
            latency_verified = latency_under_absolute_limit

        if not latency_verified:
            logger.critical(
                "Crowbar/latencia no verificados: acción=%s, latencia=%s ns.",
                hardware_action,
                latency_ns,
            )
            global_verdict = _omega_meet(
                global_verdict,
                HeytingOmega3.VETOED,
            )
            if not crowbar_triggered:
                crowbar_triggered, latency_ns, hardware_action = self._crowbar(
                    HeytingOmega3.VETOED,
                )
                latency_finite = math.isfinite(latency_ns) and latency_ns >= 0.0
                latency_under_absolute_limit = bool(
                    latency_finite and latency_ns < _CROWBAR_TIMING_LIMIT_NS
                )
                latency_verified = bool(
                    crowbar_triggered
                    and latency_finite
                    and latency_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS
                    and latency_under_absolute_limit
                )

        try:
            visceral = dict(
                self.engine.apply_visceral_functor_phi_sem(
                    global_verdict,
                    cert2.yang_mills_energy,
                    cert2.wilson_angle_rad,
                )
            )
        except Exception as exc:
            logger.warning("Functor semántico no disponible: %s", exc)
            visceral = {"note": "visceral_functor_unavailable"}

        phase_chain_valid = bool(
            cert2.parent_phase1_chain_hash == cert1.chain_hash
            and cert2.parent_phase1_signature == cert1.hmac_phase1_signature
        )
        qnd_valid = bool(cert2.qnd_valid)
        poincare_valid = bool(
            cert2.poincare_data_valid
            and cert2.poincare_map_symplectic_valid
            and cert2.poincare_transversal
        )
        rsi3_valid = bool(
            cert1.rsi3_valid
            and cert2.rsi3_valid
            and math.isfinite(cert2.banach_lipschitz_t3)
            and cert2.banach_lipschitz_t3 <= _BANACH_LIPSCHITZ_T3_MAX
        )
        epistemologically_valid = bool(
            global_verdict is HeytingOmega3.COHERENT
            and phase_chain_valid
            and cert1.is_trace_conserved
            and cert1.symplectic_certified
            and qnd_valid
            and poincare_valid
            and rsi3_valid
            and latency_verified
        )

        now = float(time.time())
        atom_codes = tuple(int(atom) for atom in atoms)
        provisional = TopologicalGaugeFieldSovereignState(
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=global_verdict,
            crowbar_triggered=bool(crowbar_triggered),
            crowbar_latency_ns=float(latency_ns),
            visceral_report=visceral,
            sha256_merkle_digest="",
            provenance_hmac="",
            timestamp_utc=now,
            crowbar_latency_verified=bool(latency_verified),
            hardware_action=str(hardware_action),
            heyting_join_atoms=atom_codes,
            heyting_meet_atoms=atom_codes,
            heyting_implies_table=HeytingOmega3Adjudicator.truth_table_implies(),
            heyting_negate_table=HeytingOmega3Adjudicator.truth_table_negate(),
            is_epistemologically_valid=epistemologically_valid,
            agent_version=_AGENT_VERSION,
            spectral_gap=cert1.spectral_gap,
            poincare_cartan_invariant=cert1.poincare_cartan_invariant,
            bruno_satisfied=cert2.bruno_satisfied,
            lindstedt_residual=cert1.poincare_residual,
            qnd_variance=cert2.qnd_variance,
            hmac_verified=True,
            phase_chain_valid=phase_chain_valid,
            crowbar_acknowledged=bool(crowbar_triggered),
            qnd_valid=qnd_valid,
            poincare_valid=poincare_valid,
            rsi3_valid=rsi3_valid,
        )
        merkle_digest = self._state_merkle_digest(provisional)
        provenance = self._compute_hmac(f"STATE:{merkle_digest}")
        return replace(
            provisional,
            sha256_merkle_digest=merkle_digest,
            provenance_hmac=provenance,
        )

    def execute_phase3_adjudication(
        self,
        cert1: GaugeFieldIsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        density_matrix: NDArray[Any],
    ) -> TopologicalGaugeFieldSovereignState:
        """Compatibilidad: ejecuta Decide & Act y devuelve el estado soberano."""
        return self.emit_sovereign_state(cert1, cert2, density_matrix)

    def verify_sovereign_hmac(
        self, state: TopologicalGaugeFieldSovereignState,
    ) -> bool:
        """Verifica firmas de fases, enlace padre, raíz Merkle y HMAC terminal."""
        try:
            self._verify_phase1_certificate(state.phase1_cert)
            self._verify_phase2_certificate(state.phase2_cert)
            if (
                state.phase2_cert.parent_phase1_chain_hash
                != state.phase1_cert.chain_hash
                or state.phase2_cert.parent_phase1_signature
                != state.phase1_cert.hmac_phase1_signature
            ):
                return False

            expected_root = self._state_merkle_digest(state)
            if not hmac.compare_digest(
                expected_root,
                state.sha256_merkle_digest,
            ):
                return False
            expected_provenance = self._compute_hmac(f"STATE:{expected_root}")
            return hmac.compare_digest(
                expected_provenance,
                state.provenance_hmac,
            )
        except (GaugeFieldAgentError, TypeError, ValueError, AttributeError):
            return False


# ══════════════════════════════════════════════════════════════════════════════
# §5. FACHADA: ORQUESTACIÓN DEL CICLO ANIDADO OODA DE TRES FASES
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldAgent(GaugeAgentPhase3Decider):
    """
    Fachada pública del agente.

      FASE 1: emit_phase1_certificate → continue_to_phase2
      FASE 2: emit_phase2_certificate → continue_to_phase3
      FASE 3: emit_sovereign_state

    La clave por defecto se conserva por compatibilidad y sólo es apropiada para
    pruebas. En producción debe inyectarse una clave secreta gestionada fuera del
    código fuente.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        background_hamiltonian: Optional[NDArray[Any]] = None,
    ) -> None:
        super().__init__(
            dimension=dimension,
            secret_key=secret_key,
            background_hamiltonian=background_hamiltonian,
        )
        logger.info(
            "TopologicalGaugeFieldAgent v%s: dim=%d engine=%s envelopes=%s",
            _AGENT_VERSION,
            self.dimension,
            _ENGINE_IMPORTED,
            _ENGINE_HAS_ENVELOPES,
        )

    def govern_gauge_field(
        self,
        density_matrix: NDArray[Any],
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> TopologicalGaugeFieldSovereignState:
        """Ejecuta en orden los tres cierres y los dos puentes autenticados."""
        cert1 = self.emit_phase1_certificate(density_matrix)
        cert1 = self.continue_to_phase2(cert1)

        cert2 = self.emit_phase2_certificate(
            cert1,
            density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
        )
        cert2 = self.continue_to_phase3(cert2)

        return self.emit_sovereign_state(cert1, cert2, density_matrix)

    def issue_positron_auth(self, electron_anomaly_count: int) -> str:
        """Firma el contador para la autorización de la etapa Fock."""
        count = int(electron_anomaly_count)
        if count <= 0:
            raise ValueError("electron_anomaly_count debe ser positivo.")
        return self._compute_hmac(f"AUTH_POSITRON:{count}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__: List[str] = [
    "GaugeFieldAgentError",
    "GaugeContractCorruptedError",
    "GaugeIsometryViolationError",
    "SymplecticGaugeInvarianceError",
    "EnvelopeChainViolationError",
    "PoincareLindstedtResonanceError",
    "RSI3ContractionFailureError",
    "HeytingOmega3Adjudicator",
    "GaugeFieldIsometryCertificate",
    "Phase2GaugeDynamicsCertificate",
    "TopologicalGaugeFieldSovereignState",
    "GaugeAgentPhase1Observer",
    "GaugeAgentPhase2Orientor",
    "GaugeAgentPhase3Decider",
    "TopologicalGaugeFieldAgent",
    "HeytingOmega3",
    "TopologicalGaugeFieldEngine",
    "TopologicalGaugeFieldSnapshot",
]


# ══════════════════════════════════════════════════════════════════════════════
# AUTOCOMPROBACIÓN DETERMINISTA DEL CONTRATO SOFTWARE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 80)
    print("Topological Gauge Field Agent v12.0.0 — smoke test de contratos")
    print(
        f"engine={_ENGINE_IMPORTED}, envelopes={_ENGINE_HAS_ENVELOPES}, "
        f"continue={_ENGINE_HAS_CONTINUE}, governance={_ENGINE_HAS_GOVERNANCE}"
    )
    print("═" * 80)

    dimension = 16
    rng = np.random.default_rng(2026)
    gaussian = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    )
    density_sample = gaussian @ gaussian.conj().T
    density_sample /= np.trace(density_sample)

    agent = TopologicalGaugeFieldAgent(dimension=dimension)
    state = agent.govern_gauge_field(density_sample)
    hmac_ok = agent.verify_sovereign_hmac(state)

    print(f"Veredicto global      : {state.global_verdict.name}")
    print(f"Traza fase 1          : {state.phase1_cert.density_trace:.12f}")
    print(
        "Conexión antiherm./traceless: "
        f"{state.phase1_cert.is_anti_hermitian_A}/"
        f"{state.phase1_cert.is_traceless_A}"
    )
    print(
        "RSI³ fase 1 válido    : "
        f"{state.phase1_cert.rsi3_valid} "
        f"(iteraciones={state.phase1_cert.rsi3_iterations}, "
        f"contracción observada={state.phase1_cert.rsi3_contraction:.4g})"
    )
    print(
        "RSI³ motor Lip(T³)    : "
        f"{state.phase2_cert.banach_lipschitz_t3:.6g}"
    )
    print(
        "Poincaré válido       : "
        f"{state.poincare_valid} "
        f"(cruces={state.phase2_cert.poincare_section_n_crossings})"
    )
    print(
        "QND válido            : "
        f"{state.qnd_valid} "
        f"(var={state.qnd_variance:.4g})"
    )
    print(
        "Crowbar reportado     : "
        f"{state.crowbar_triggered}, "
        f"latencia={state.crowbar_latency_ns:.3f} ns, "
        f"verificación software={state.crowbar_latency_verified}"
    )
    print(
        "Cadena/HMAC/Merkle     : "
        f"{state.phase_chain_valid}/{hmac_ok}/"
        f"{state.sha256_merkle_digest[:24]}…"
    )

    assert hmac_ok, "La verificación criptográfica local debe ser reproducible."

    # La modificación de un campo firmado debe invalidar la verificación.
    tampered_cert1 = replace(
        state.phase1_cert,
        density_trace=state.phase1_cert.density_trace + 1.0e-3,
    )
    tampered_state = replace(state, phase1_cert=tampered_cert1)
    assert not agent.verify_sovereign_hmac(tampered_state)

    # Autorización de Fock: una firma inválida no autoriza la operación.
    bad_state = agent.govern_gauge_field(
        density_sample,
        electron_anomaly_count=3,
        positron_auth_signature="invalid",
    )
    valid_signature = agent.issue_positron_auth(3)
    good_state = agent.govern_gauge_field(
        density_sample,
        electron_anomaly_count=3,
        positron_auth_signature=valid_signature,
    )
    print(
        "Fock: firma inválida → "
        f"{bad_state.phase2_cert.positron_annihilations_count}; "
        f"firma válida → {good_state.phase2_cert.positron_annihilations_count}"
    )
    print("═" * 80)